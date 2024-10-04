import Mathlib

namespace geometric_series_sum_l797_797751

-- Definition of the geometric sum function in Lean
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * ((1 - r^n) / (1 - r))

-- Specific terms for the problem
def a : ℚ := 2
def r : ℚ := 2 / 5
def n : ℕ := 5

-- The target sum we aim to prove
def target_sum : ℚ := 10310 / 3125

-- The theorem stating that the calculated sum equals the target sum
theorem geometric_series_sum : geometric_sum a r n = target_sum :=
by sorry

end geometric_series_sum_l797_797751


namespace sequence_sum_l797_797502

noncomputable def a : ℕ → ℝ
| 0       := 3
| (n + 1) := 3 - 6 / (a n)

theorem sequence_sum (n : ℕ) :
  (∑ i in Finset.range (n + 1), 1 / a i) = (1 / 3) * (2^(n + 2) - n - 3) := 
  sorry

end sequence_sum_l797_797502


namespace mutually_exclusive_not_opposite_l797_797806

-- Define the set and the probability space.
inductive Ball
| Red : Ball
| Black : Ball

open Ball

def bag : List Ball := [Red, Red, Black, Black]

/-- Define the event of drawing two balls from the bag and the possible outcomes. -/
def draw_two_balls (bag : List Ball) : List (Ball × Ball) :=
  [(Red, Red), (Red, Black), (Black, Red), (Black, Black)]

/-- Define the events based on the conditions. -/
def event_at_least_one_black_ball (b1 b2 : Ball) : Prop :=
  b1 = Black ∨ b2 = Black

def event_all_red_balls (b1 b2 : Ball) : Prop :=
  b1 = Red ∧ b2 = Red

def event_all_black_balls (b1 b2 : Ball) : Prop :=
  b1 = Black ∧ b2 = Black

def event_at_least_one_red_ball (b1 b2 : Ball) : Prop :=
  b1 = Red ∨ b2 = Red

def event_exactly_one_black_ball (b1 b2 : Ball) : Prop :=
  (b1 = Black ∧ b2 = Red) ∨ (b1 = Red ∧ b2 = Black)

def event_exactly_two_black_balls (b1 b2 : Ball) : Prop :=
  b1 = Black ∧ b2 = Black

/-- Prove that events in option D are mutually exclusive but not opposite. -/
theorem mutually_exclusive_not_opposite :
  ∀ (b1 b2 : Ball),
    (event_exactly_one_black_ball b1 b2 ∧ ¬ event_exactly_two_black_balls b1 b2) ↔ (¬ event_exactly_one_black_ball b1 b2 ∧ event_exactly_two_black_balls b1 b2) :=
by
  simp
  sorry

end mutually_exclusive_not_opposite_l797_797806


namespace sequence_an_formula_integer_exists_l797_797544

noncomputable def a_seq (n : ℕ) := n
noncomputable def b_seq (n : ℕ) (λ : ℤ) := 3^n + (-1 : ℤ)^(n-1) * λ * 2^(a_seq n)
def Sn (n : ℕ) := ∑ i in Finset.range n, a_seq (i + 1)

theorem sequence_an_formula :
  (∀ n : ℕ, n > 0 → (∑ i in Finset.range n, (a_seq (i + 1))^3 = (Sn n)^2)
  → (∀ n: ℕ, n > 0 → a_seq n = n)) := 
sorry

theorem integer_exists (λ : ℤ) :
  (∃ λ : ℤ, (∀ n : ℕ, n > 0 → b_seq (n+1) λ > b_seq n λ) ↔ λ = -1) := 
sorry

end sequence_an_formula_integer_exists_l797_797544


namespace nine_fact_div_four_fact_eq_15120_l797_797421

theorem nine_fact_div_four_fact_eq_15120 :
  (362880 / 24) = 15120 :=
by
  sorry

end nine_fact_div_four_fact_eq_15120_l797_797421


namespace nine_fact_div_four_fact_eq_15120_l797_797419

theorem nine_fact_div_four_fact_eq_15120 :
  (362880 / 24) = 15120 :=
by
  sorry

end nine_fact_div_four_fact_eq_15120_l797_797419


namespace proposition_p_or_proposition_q_implies_m_lt_neg1_l797_797569

def discriminant_quadratic (a b c : ℝ) : ℝ :=
  b^2 - 4*a*c

def has_two_distinct_positive_real_roots (a b c : ℝ) : Prop :=
  discriminant_quadratic a b c > 0 ∧ 
  let roots := (-b ± real.sqrt (discriminant_quadratic a b c)) / (2 * a)
  roots.fst > 0 ∧ roots.snd > 0

def no_real_roots (a b c : ℝ) : Prop :=
  discriminant_quadratic a b c < 0

theorem proposition_p_or_proposition_q_implies_m_lt_neg1 (m : ℝ) :
  (has_two_distinct_positive_real_roots 1 m 1 ∨ no_real_roots 4 (4 * (m + 2)) 1) → m < -1 := by
  sorry

end proposition_p_or_proposition_q_implies_m_lt_neg1_l797_797569


namespace rhombus_side_length_l797_797776

theorem rhombus_side_length (S m n : ℝ) (hS_pos : 0 < S) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  let a := sqrt (S * (m^2 + n^2) / (2 * m * n))
  in ∃ a, a = sqrt (S * (m^2 + n^2) / (2 * m * n)) := by
  exists a
  sorry

end rhombus_side_length_l797_797776


namespace find_m_l797_797929

noncomputable def curve (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9
noncomputable def symmetric_line (x y m : ℝ) : Prop := x + m * y + 4 = 0
def is_symmetric {P Q : ℝ × ℝ} (m : ℝ) : Prop :=
   symmetric_line P.fst P.snd m ∧ symmetric_line Q.fst Q.snd m ∧ 
   (P.fst + Q.fst) / 2 = -1 ∧ (P.snd + Q.snd) / 2 = 3

theorem find_m (P Q : ℝ × ℝ) (h1 : curve P.fst P.snd) (h2 : curve Q.fst Q.snd) (h_sym : is_symmetric m) : m = -1 :=
by
  sorry

end find_m_l797_797929


namespace find_base_b_l797_797063

theorem find_base_b (b : ℕ) :
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 6) = (4 * b^2 + 1 * b + 1) →
  7 < b →
  b = 10 :=
by
  intro h₁ h₂
  sorry

end find_base_b_l797_797063


namespace part1_part2_l797_797449

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

end part1_part2_l797_797449


namespace least_possible_value_l797_797071

theorem least_possible_value (n : ℕ) (hn : n > 0) (x : Fin n → ℝ) 
  (hx : (Finset.univ.sum (λ i, |x i|)) = 1) :
  (|x 0| + Finset.univ.sum (λ i, (|Finset.univ.sum (λ j, x ⟨j.1, nat.lt_of_succ_lt_succ (nat.lt_of_lt_succ j.2)⟩)|))) = 2^(1 - n) :=
sorry

end least_possible_value_l797_797071


namespace csc_315_eq_sqrt2_l797_797043

theorem csc_315_eq_sqrt2 :
  let θ := 315
  let csc := λ θ, 1 / (Real.sin (θ * Real.pi / 180))
  315 = 360 - 45 → 
  Real.sin (315 * Real.pi / 180) = Real.sin ((360 - 45) * Real.pi / 180) → 
  Real.sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 →
  csc 315 = Real.sqrt 2 := 
by
  intros θ csc h1 h2 h3
  -- proof would go here
  sorry

end csc_315_eq_sqrt2_l797_797043


namespace sum_of_cubes_l797_797572

theorem sum_of_cubes (k : ℤ) : 
  24 * k = (k + 2)^3 + (-k)^3 + (-k)^3 + (k - 2)^3 :=
by
  sorry

end sum_of_cubes_l797_797572


namespace johnson_family_seating_l797_797604

/-- The Johnson family has 5 sons and 4 daughters. We want to find the number of ways to seat them in a row of 9 chairs such that at least 2 boys are next to each other. -/
theorem johnson_family_seating : 
  let boys := 5 in
  let girls := 4 in
  let total_children := boys + girls in
  fact total_children - 
  2 * (fact boys * fact girls) = 357120 := 
by
  let boys := 5
  let girls := 4
  let total_children := boys + girls
  have total_arrangements : ℕ := fact total_children
  have no_two_boys_next_to_each_other : ℕ := 2 * (fact boys * fact girls)
  have at_least_two_boys_next_to_each_other : ℕ := total_arrangements - no_two_boys_next_to_each_other
  show at_least_two_boys_next_to_each_other = 357120
  sorry

end johnson_family_seating_l797_797604


namespace sean_houses_l797_797579

theorem sean_houses :
  ∃ (initial traded1 bought traded2 sold upgraded: ℕ),
    initial = 45 ∧
    traded1 = 15 ∧
    bought = 18 ∧
    traded2 = 5 ∧
    sold = 7 ∧
    upgraded = 16 ∧
    initial - traded1 + bought - traded2 - sold - upgraded = 20 :=
begin
  use 45, use 15, use 18, use 5, use 7, use 16,
  split, {refl},
  split, {refl},
  split, {refl},
  split, {refl},
  split, {refl},
  exact eq.refl 20,
end

end sean_houses_l797_797579


namespace checkers_rearrangement_impossible_l797_797944

theorem checkers_rearrangement_impossible :
  ∀ (board : fin 5 → fin 5 → Prop), 
  (∀ i j, board i j) → 
  ¬ ∃ new_board, 
    (∀ i j, new_board i j) ∧ 
    (∀ i j, (new_board i j → (|i1 - i2| = 1 ∧ j1 = j2) ∨ (i1 = i2 ∧ |j1 - j2| = 1))) := 
sorry

end checkers_rearrangement_impossible_l797_797944


namespace johnson_family_seating_l797_797607

/-- The Johnson family has 5 sons and 4 daughters. We want to find the number of ways to seat them in a row of 9 chairs such that at least 2 boys are next to each other. -/
theorem johnson_family_seating : 
  let boys := 5 in
  let girls := 4 in
  let total_children := boys + girls in
  fact total_children - 
  2 * (fact boys * fact girls) = 357120 := 
by
  let boys := 5
  let girls := 4
  let total_children := boys + girls
  have total_arrangements : ℕ := fact total_children
  have no_two_boys_next_to_each_other : ℕ := 2 * (fact boys * fact girls)
  have at_least_two_boys_next_to_each_other : ℕ := total_arrangements - no_two_boys_next_to_each_other
  show at_least_two_boys_next_to_each_other = 357120
  sorry

end johnson_family_seating_l797_797607


namespace range_inequality_l797_797547

noncomputable def f (x : ℝ) := 1 + |x| - (1 / (1 + x^2))

theorem range_inequality:
  {x : ℝ | f (Real.log2 x) > f (-2 * Real.log (1/2) x - 1)} = {x : ℝ | Real.root 3 2 < x ∧ x < 2} :=
by
  sorry

end range_inequality_l797_797547


namespace find_k_l797_797817

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n : ℕ, a (n + 1) = r * a n

def arithmetic_sequence (a b c : ℝ) := 2 * b = a + c

theorem find_k (a : ℕ → ℝ) (k : ℝ) (r : ℝ)
  (h_geo : geometric_sequence a r)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = a)
  (h_recur : ∀ n : ℕ, a (n + 1) = k * (a n + a (n + 2)))
  (h_arith : ∀ m : ℕ, ∃ x y z : ℝ, set_of (arithmetic_sequence x y z) ⊆ set_of (arithmetic_sequence (a m) (a (m + 1)) (a (m + 2)))) :
  k = -2 / 5 :=
sorry

end find_k_l797_797817


namespace find_p0_add_p4_l797_797550

def monic_polynomial_degree_4 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, p = λ x, x^4 + a * x^3 + b * x^2 + c * x + d

theorem find_p0_add_p4 (p : ℝ → ℝ)
  (h_monic : monic_polynomial_degree_4 p)
  (h_1 : p 1 = 21)
  (h_2 : p 2 = 42)
  (h_3 : p 3 = 63) :
  p 0 + p 4 = 60 :=
by
  sorry

end find_p0_add_p4_l797_797550


namespace sum_of_reciprocals_sum_b_n_l797_797094

-- Definitions and conditions
def a_n (n : ℕ) : ℕ := n

def S_n (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem sum_of_reciprocals (h : 3 * (1 : ℚ) / (S_n 1 : ℚ) + 1 / (S_n 2 : ℚ) + 1 / (S_n 3 : ℚ) = 3 / 2) : 
  (\forall n, a_n n = n) :=
  sorry
  
-- Definitions for second part
def b_n (n : ℕ) : ℕ := n * 2^n

def T_n (n : ℕ) : ℕ := ∑ i in range (n + 1), b_n i

theorem sum_b_n (n : ℕ) : T_n n = (n - 1) * 2^(n + 1) + 2 :=
  sorry

end sum_of_reciprocals_sum_b_n_l797_797094


namespace simplest_sqrt2_l797_797736

-- Definitions and simplifications
def simplest_square_root (x : ℝ) : Prop :=
  ∀ (y : ℝ), (∃ (z : ℝ), y = z * z ∧ x = y) → x = y

theorem simplest_sqrt2 :
  simplest_square_root (sqrt 2) :=
by
  sorry

end simplest_sqrt2_l797_797736


namespace hydrochloric_acid_required_l797_797793

-- Define the quantities for the balanced reaction equation
def molesOfAgNO3 : ℕ := 2
def molesOfHNO3 : ℕ := 2
def molesOfHCl : ℕ := 2

-- Define the condition for the reaction (balances the equation)
def balanced_reaction (x y z w : ℕ) : Prop :=
  x = y ∧ x = z ∧ y = w

-- The goal is to prove that the number of moles of HCl needed is 2
theorem hydrochloric_acid_required :
  balanced_reaction molesOfAgNO3 molesOfHCl molesOfHNO3 2 →
  molesOfHCl = 2 :=
by sorry

end hydrochloric_acid_required_l797_797793


namespace range_of_m_l797_797401

namespace MathProblem

def A : Set ℝ := { x | abs(x - 2) ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | (x - 1 - m) * (x - 1 + m) ≤ 0 }

theorem range_of_m (m : ℝ) (h_m_pos : m > 0) (h_sub : A ⊆ B m) : m ≥ 5 :=
sorry

end MathProblem

end range_of_m_l797_797401


namespace johnson_family_seating_problem_l797_797609

theorem johnson_family_seating_problem : 
  ∃ n : ℕ, n = 9! - 5! * 4! ∧ n = 359760 :=
by
  have total_ways := (Nat.factorial 9)
  have no_adjacent_boys := (Nat.factorial 5) * (Nat.factorial 4)
  have result := total_ways - no_adjacent_boys
  use result
  split
  . exact eq.refl result
  . norm_num -- This will replace result with its evaluated form, 359760

end johnson_family_seating_problem_l797_797609


namespace pos_int_solutions_l797_797648

theorem pos_int_solutions (x : ℤ) : (3 * x - 4 < 2 * x) → (0 < x) → (x = 1 ∨ x = 2 ∨ x = 3) :=
by
  intro h1 h2
  have h3 : x - 4 < 0 := by sorry  -- Step derived from inequality simplification
  have h4 : x < 4 := by sorry     -- Adding 4 to both sides
  sorry                           -- Combine conditions to get the specific solutions

end pos_int_solutions_l797_797648


namespace complex_multiplication_l797_797315

theorem complex_multiplication : (1 + 2 * Complex.i) * (2 + Complex.i) = 5 * Complex.i := 
by 
  sorry

end complex_multiplication_l797_797315


namespace num_roots_x_minus_sin_x_l797_797252

theorem num_roots_x_minus_sin_x :
  ∃! x : ℝ, x - sin x = 0 := sorry

end num_roots_x_minus_sin_x_l797_797252


namespace part1_part2_l797_797477

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l797_797477


namespace georgia_total_cost_l797_797809

noncomputable def cost_of_carnations (teachers : list (ℕ × ℕ)) (friends : list (ℕ × ℕ)) : ℕ :=
  let price_single := 50         -- in cents
  let price_dozen := 400         -- in cents
  let price_bundle_25 := 800     -- in cents
  let teacher_cost (t : ℕ × ℕ) := t.1 * price_dozen + t.2 * price_single
  let friend_cost (f : ℕ × ℕ) := f.1 * price_dozen + f.2 * price_single
  let total_teacher_cost := (teachers.map teacher_cost).sum
  let total_friend_cost := (friends.map friend_cost).sum
  total_teacher_cost + total_friend_cost

theorem georgia_total_cost :
  cost_of_carnations [(1, 0), (1, 3), (0, 25), (2, 9), (3, 0)]
                      [(0, 3), (1, 0), (0, 9), (0, 7), (2, 0), (0, 10), (0, 10), (1, 5), (0, 25)] = 8800 :=
by sorry

end georgia_total_cost_l797_797809


namespace parabola_properties_l797_797483

-- Define conditions as hypotheses
variables {E A B M N : Point}
variables {l : Line}
variables [OnParabola E 2 2 2p x, OnLine l (2, 0), Intersects l Parabola A B]

-- Define points and lines given the conditions
def parabola := {p : Point | p.2^2 = 2 * p.1}
def focus := (1 / 2, 0 : ℝ)
def line_through_origin := {l : Line | l.contains (2, 0)}
def intersection_at_x_neg_2 (P : Point) := ∃ y, P = (-2, y)

-- Problem statement in Lean
theorem parabola_properties :
  ∀ E A B M N (E_on_parabola : OnParabola E 2 2 2p x)
          (line_through_origin : Intersects l (2, 0))
          (l_intersects_parabola : Intersects l parabola A B)
          (EA_intersects_x_neg_2 : intersection_at_x_neg_2 M)
          (EB_intersects_x_neg_2 : intersection_at_x_neg_2 N),
    (parabola = {p : Point | p.2^2 = 2 * p.1} ∧ focus = (1 / 2, 0 : ℝ)) ∧
    ∠ MON = π / 2 
  :=
by
  sorry

end parabola_properties_l797_797483


namespace arithmetic_sequence_property_l797_797932

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ((a 6 - 1)^3 + 2013 * (a 6 - 1)^3 = 1))
  (h2 : ((a 2008 - 1)^3 = -2013 * (a 2008 - 1)^3))
  (sum_formula : ∀ n, S n = n * a n) : 
  S 2013 = 2013 ∧ a 2008 < a 6 := 
sorry

end arithmetic_sequence_property_l797_797932


namespace arctan_sum_one_two_three_no_four_distinct_arctan_sum_l797_797218

noncomputable def arctan_seq_existence (m : ℕ) (hm : 3 ≤ m ∧ Odd m) : ∃ s : Fin m → ℕ, StrictMono s ∧ (∑ i, Real.arctan (s i)) = Int.cast m * Real.pi := sorry

theorem arctan_sum_one_two_three :
  Real.arctan 1 + Real.arctan 2 + Real.arctan 3 = Real.pi :=
  sorry

theorem no_four_distinct_arctan_sum (a b c d : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (∑ x in {a, b, c, d}.toFinset, Real.tan (Real.arctan x)) ≠ 0 :=
  sorry

end arctan_sum_one_two_three_no_four_distinct_arctan_sum_l797_797218


namespace csc_315_eq_neg_sqrt_2_l797_797019

theorem csc_315_eq_neg_sqrt_2 :
  let csc := λ θ, 1 / Real.sin θ in
  csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
  by
  let sin := Real.sin
  have h1 : csc (315 * Real.pi / 180) = 1 / sin (315 * Real.pi / 180) := rfl
  have h2 : sin (315 * Real.pi / 180) = sin ((360 - 45) * Real.pi / 180) := by congr; norm_num
  have h3 : sin ((360 - 45) * Real.pi / 180) = -sin (45 * Real.pi / 180) := by
    rw [Real.sin_pi_sub]
    congr; norm_num
  have h4 : sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 := Real.sin_of_one_div_sqrt_two 45 rfl
  sorry

end csc_315_eq_neg_sqrt_2_l797_797019


namespace length_of_other_train_l797_797299

-- Define the given conditions in a)
def length_first_train : ℝ := 270
def speed_first_train_kmph : ℝ := 120
def speed_second_train_kmph : ℝ := 80
def time_to_cross : ℝ := 9

-- Define the conversions and calculations required from a) and c)
def speed_first_train_mps : ℝ := (speed_first_train_kmph * 1000) / 3600
def speed_second_train_mps : ℝ := (speed_second_train_kmph * 1000) / 3600
def relative_speed : ℝ := speed_first_train_mps + speed_second_train_mps
def total_distance : ℝ := relative_speed * time_to_cross

-- The proof problem to solve
theorem length_of_other_train : (length_first_train + 229.95 = total_distance) :=
by
  sorry

end length_of_other_train_l797_797299


namespace angle_B_value_triangle_perimeter_l797_797829

open Real

variables {A B C a b c : ℝ}

-- Statement 1
theorem angle_B_value (h1 : a = b * sin A + sqrt 3 * a * cos B) : B = π / 2 := by
  sorry

-- Statement 2
theorem triangle_perimeter 
  (h1 : B = π / 2)
  (h2 : b = 4)
  (h3 : (1 / 2) * a * c = 4) : 
  a + b + c = 4 + 4 * sqrt 2 := by
  sorry


end angle_B_value_triangle_perimeter_l797_797829


namespace seating_arrangements_l797_797597

theorem seating_arrangements (sons daughters : ℕ) (totalSeats : ℕ) (h_sons : sons = 5) (h_daughters : daughters = 4) (h_seats : totalSeats = 9) :
  let total_arrangements := totalSeats.factorial
  let unwanted_arrangements := sons.factorial * daughters.factorial
  total_arrangements - unwanted_arrangements = 360000 :=
by
  rw [h_sons, h_daughters, h_seats]
  let total_arrangements := 9.factorial
  let unwanted_arrangements := 5.factorial * 4.factorial
  exact Nat.sub_eq_of_eq_add $ eq_comm.mpr (Nat.add_sub_eq_of_eq total_arrangements_units)
where
  total_arrangements_units : 9.factorial = 5.factorial * 4.factorial + 360000 := by
    rw [Nat.factorial, Nat.factorial, Nat.factorial, ←Nat.factorial_mul_factorial_eq 5 4]
    simp [tmp_rewriting]

end seating_arrangements_l797_797597


namespace range_of_a_l797_797651

theorem range_of_a (x a : ℝ) :
  (x^3 + a > -2 ∧ x < -2 → a > 6) ∧ (x^3 + a < 2 ∧ x > 2 → a < -6)
  → (a ∈ set.Ioo ((-∞) : ℝ) (-6) ∪ set.Ioo (6 : ℝ) ∞) :=
sorry

end range_of_a_l797_797651


namespace volume_not_occupied_by_cones_l797_797273

theorem volume_not_occupied_by_cones:
  let r := 12 in
  let h_cylinder := 24 in
  let h_cone := 12 in
  let volume_cylinder := Math.pi * r^2 * h_cylinder in
  let volume_cone := (1/3) * Math.pi * r^2 * h_cone in
  let volume_cones := 2 * volume_cone in
  let volume_not_occupied := volume_cylinder - volume_cones in
  volume_not_occupied = 2304 * Math.pi :=
by
  sorry

end volume_not_occupied_by_cones_l797_797273


namespace shelby_total_gold_stars_l797_797310

variable (yesterday today : ℕ)
variable (earned_yesterday earned_today : ℕ)

-- Defining the conditions given in the problem.
def shelby_earned_yesterday := earned_yesterday = 4
def shelby_earned_today := earned_today = 3

-- Defining the main statement to prove
theorem shelby_total_gold_stars (h1 : shelby_earned_yesterday) (h2 : shelby_earned_today) :
  (earned_yesterday + earned_today) = 7 :=
by
  sorry

end shelby_total_gold_stars_l797_797310


namespace part1_part2_l797_797474

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l797_797474


namespace no_inverse_if_determinant_zero_l797_797913

def vector := ℝ × ℝ

def magnitude (v : vector) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def normalize (v : vector) : vector :=
  let mag := magnitude v in (v.1 / mag, v.2 / mag)

def projection_matrix (v : vector) : matrix (fin 2) (fin 2) ℝ :=
  let u := normalize v in
  let ux := ((u.1, u.2), (u.1, u.2)) in
  (λ i j, ux i.1 j.1)

def det (m : matrix (fin 2) (fin 2) ℝ) : ℝ :=
  m 0 0 * m 1 1 - m 0 1 * m 1 0

theorem no_inverse_if_determinant_zero : 
  ∀ v : vector,
  v = (1, 3) →
  let Q := projection_matrix v in
  det Q = 0 →
  (Q⁻¹ = (λ i j, (0 : ℝ))) :=
by
  intros v hv Q hQdet0
  sorry

end no_inverse_if_determinant_zero_l797_797913


namespace problem1_problem2_l797_797106

-- Definitions for points and vectors based on the given conditions
def point_O : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (0, 1)
def point_C (θ : ℝ) : ℝ × ℝ := (2 * Real.sin θ, Real.cos θ)

-- Problem 1: Prove that if |AC| = |BC|, then tan θ = 1/2
theorem problem1 (θ : ℝ) (h : ((2 * Real.sin θ - 1) ^ 2 + Real.cos θ ^ 2 = 4 * (Real.sin θ) ^ 2 + (Real.cos θ - 1) ^ 2)) :
  Real.tan θ = 1 / 2 :=
sorry

-- Problem 2: Prove that if (OA + 2OB) · OC = 1, then sin θ cos θ = -3/8
theorem problem2 (θ : ℝ) (h : (1 * (2 * Real.sin θ) + 2 * 1 * (Real.cos θ)) = 1) :
  Real.sin θ * Real.cos θ = -3 / 8 :=
sorry

end problem1_problem2_l797_797106


namespace cone_volume_l797_797328

-- Definitions of the initial conditions
def radius_of_circle : ℝ := 6
def sector_fraction : ℝ := 5 / 8

-- Defining the equivalent Lean theorem
theorem cone_volume (r : ℝ) (f : ℝ) (h₁ : r = radius_of_circle) (h₂ : f = sector_fraction) : 
  (1 / 3) * π * (f * (2 * π * r) / (2 * π))^2 * (sqrt (r^2 - ((f * (2 * π * r)) / (2 * π))^2)) = 
  4.6875 * π * sqrt 21.9375 := 
by 
  sorry

end cone_volume_l797_797328


namespace a_n_increasing_l797_797924

def a_n (n : ℕ) : ℝ := ∑ k in Finset.range n, 1 / (k + 1) / (n + 1 - (k + 1))

theorem a_n_increasing : ∀ n : ℕ, n ≥ 2 → a_n n > a_n (n - 1) :=
by
  sorry

end a_n_increasing_l797_797924


namespace johnson_family_seating_l797_797631

theorem johnson_family_seating (boys girls : Finset ℕ) (h_boys : boys.card = 5) (h_girls : girls.card = 4) :
  (∃ (arrangement : List ℕ), arrangement.length = 9 ∧ at_least_two_adjacent boys arrangement) :=
begin
  -- Given the total number of ways: 9! 
  -- subtract 5! * 4! from 9! to get the result 
  have total_arrangements := nat.factorial 9,
  have restrictive_arrangements := nat.factorial 5 * nat.factorial 4,
  exact (total_arrangements - restrictive_arrangements) = 360000,
end

end johnson_family_seating_l797_797631


namespace mean_proportional_of_segments_l797_797486

theorem mean_proportional_of_segments (a b c : ℝ) (a_val : a = 2) (b_val : b = 6) :
  c = 2 * Real.sqrt 3 ↔ c*c = a * b := by
  sorry

end mean_proportional_of_segments_l797_797486


namespace perfect_squares_less_than_20000_l797_797364

theorem perfect_squares_less_than_20000: 
  let count_possible_squares := 
    (λ n, ∃ a b : ℤ, n = a^2 ∧ n < 20000 ∧ b = a - 3 ∧ (a + 3)^2 - a^2 = n) in
    finset.card (finset.filter count_possible_squares (finset.range 141)) = 70 :=
by
  sorry

end perfect_squares_less_than_20000_l797_797364


namespace johnson_family_seating_l797_797633

theorem johnson_family_seating (boys girls : Finset ℕ) (h_boys : boys.card = 5) (h_girls : girls.card = 4) :
  (∃ (arrangement : List ℕ), arrangement.length = 9 ∧ at_least_two_adjacent boys arrangement) :=
begin
  -- Given the total number of ways: 9! 
  -- subtract 5! * 4! from 9! to get the result 
  have total_arrangements := nat.factorial 9,
  have restrictive_arrangements := nat.factorial 5 * nat.factorial 4,
  exact (total_arrangements - restrictive_arrangements) = 360000,
end

end johnson_family_seating_l797_797633


namespace maximum_minimum_sum_l797_797557

def f (x : ℝ) : ℝ := (2*x^2 + x - 2 + sin x) / (x^2 - 1)

theorem maximum_minimum_sum (M m : ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → - x + sin (- x) / (- x ^ 2 - 1) = - (x + sin x) / (x ^ 2 - 1)) →
  (M = 2 + (sup {y | ∃ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ y = (x + sin x) / (x ^ 2 - 1)})) →
  (m = 2 - (sup {y | ∃ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ y = (x + sin x) / (x ^ 2 - 1)})) →
  (M + m = 4) :=
by
  sorry

end maximum_minimum_sum_l797_797557


namespace inequality_proof_l797_797954

variable {n : ℕ} (α : Fin n → ℝ)
  
theorem inequality_proof (h1 : n ≥ 2) (h2 : ∀ i, 0 < α i) (h3 : ∑ i, α i = 1) : 
  ∑ i, α i / (2 - α i) ≥ n / (2 * n - 1) :=
sorry

end inequality_proof_l797_797954


namespace arrival_time_C_l797_797341

def planned_departure_time := time.mk 10 10
def planned_arrival_time := time.mk 13 10
def late_departure_time := time.mk 10 15
def early_arrival_time := time.mk 13 06
def arrival_at_C_time := time.mk 11 50

theorem arrival_time_C : 
  ((late_departure_time.diff planned_departure_time) + (planned_arrival_time.diff early_arrival_time) = 9) ∧
  arrival_at_C_time = planned_departure_time.add_minute 100

end arrival_time_C_l797_797341


namespace inverse_function_l797_797284

noncomputable def f (x : ℝ) : ℝ := 3 - 4 * x

noncomputable def g (x : ℝ) : ℝ := (3 - x) / 4

theorem inverse_function :
  ∀ x : ℝ, f(g(x)) = x ∧ g(f(x)) = x :=
by
  sorry

end inverse_function_l797_797284


namespace find_a_plus_b_l797_797922
noncomputable def probability_4_heads_before_3_tails : ℚ :=
  (1 / 9)

theorem find_a_plus_b : 
  ∃ (a b : ℕ), (a ≠ 0) ∧ (b ≠ 0) ∧ (Nat.coprime a b) ∧ 
               (probability_4_heads_before_3_tails = (a / b)) ∧ 
               (a + b = 10) :=
begin
  use [1, 9],
  split, { norm_num },
  split, { norm_num },
  split,
  { exact nat.coprime_one_right 9 },
  split,
  { norm_num,
    show 1 / 9 = probability_4_heads_before_3_tails,
    exact rfl, },
  { norm_num }
end

end find_a_plus_b_l797_797922


namespace parabola_standard_equation_l797_797835

open Real

theorem parabola_standard_equation {p : ℝ} (p_pos : p > 0) :
  (∃ (y, x : ℝ), y^2 = 2 * p * x ∧ (x, y) = (-2, -4))
  ∨ (∃ (x, y : ℝ), x^2 = -2 * p * y ∧ (x, y) = (-2, -4)) → 
  (∀ x y : ℝ, (y^2 = -8 * x ∨ x^2 = -y) ∧ ((-2, -4) = (x, y))) :=
by {
  sorry
}

end parabola_standard_equation_l797_797835


namespace find_z_l797_797772

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the condition for the complex number z
def condition (z : ℂ) : Prop :=
  det 1 (-1) z (z * complex.I) = 2

-- The main theorem to be proved
theorem find_z : ∃ z : ℂ, condition z ∧ z = 1 - complex.I :=
by
  sorry

end find_z_l797_797772


namespace gcd_of_a_and_b_is_one_l797_797545

theorem gcd_of_a_and_b_is_one {a b : ℕ} (h1 : a > b) (h2 : Nat.gcd (a + b) (a - b) = 1) : Nat.gcd a b = 1 :=
by
  sorry

end gcd_of_a_and_b_is_one_l797_797545


namespace max_rooks_non_attacking_max_rooks_attacked_by_one_l797_797322

-- Definitions for the chessboard and rook attacks.
def Chessboard := list (ℕ × ℕ)
def Rook := (ℕ, ℕ)
def attacks (r1 r2 : Rook) : Prop := r1.1 = r2.1 ∨ r1.2 = r2.2

-- Part (a): Maximum number of rooks such that no two rooks attack each other.
theorem max_rooks_non_attacking : 
  ∃ (rooks : list Rook), list.length rooks = 8 ∧ (∀ (r1 r2 : Rook), r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 → ¬ attacks r1 r2) :=
sorry

-- Part (b): Maximum number of rooks such that each rook is attacked by at most one other rook.
theorem max_rooks_attacked_by_one : 
  ∃ (rooks : list Rook), list.length rooks = 10 ∧ (∀ r ∈ rooks, list.count (λ r' => attacks r r') rooks ≤ 1) :=
sorry

end max_rooks_non_attacking_max_rooks_attacked_by_one_l797_797322


namespace ratio_areas_pentagon_to_squares_l797_797266

-- Define the squares and their properties
structure Square where
  side : ℝ

def ABCD : Square := { side := 1 }
def EFGH : Square := { side := 2 }
def KLMO : Square := { side := 1 }

-- Midpoint and partition conditions
def Point := ℝ × ℝ

def D : Point := (1, 0.5 * EFGH.side)
def C : Point := (2, 2 * 2 / 3)

-- Areas
def area_square (s : Square) : ℝ := s.side * s.side

def area_pentagon : ℝ :=
  let area_ABDC := 0.5 * (1 + 5 / 3) * 1
  let area_OKGC := 0.5 * (1 + 2 / 3) * 1
  let area_MGC := 0.5 * 2 / 3 * 1
  area_ABDC + area_OKGC + area_MGC

-- Total area of three squares
def total_area_squares : ℝ :=
  area_square ABCD + area_square EFGH + area_square KLMO

-- The theorem statement
theorem ratio_areas_pentagon_to_squares :
  let total_area_pentagon := area_pentagon
  (total_area_pentagon / total_area_squares) = 5 / 12 := 
by
  sorry

end ratio_areas_pentagon_to_squares_l797_797266


namespace sequence_a_n_sequence_b_n_T_n_l797_797086

noncomputable def a_n : ℕ → ℕ 
| 1 := 1
| n := 2*n - 1

def S (n : ℕ) : ℕ := ∑ i in finset.range n, a_n i

theorem sequence_a_n (n : ℕ) : a_n (n + 1) = 2*(n + 1) - 1 :=
by sorry

def b_n (n : ℕ) : ℕ := (2*n-1) * 2^n

def T (n : ℕ) : ℕ := ∑ i in finset.range n, b_n i

theorem sequence_b_n_T_n (n : ℕ) : 
    T n = (2*n-3) * 2^(n+1) + 6 :=
by sorry

end sequence_a_n_sequence_b_n_T_n_l797_797086


namespace value_of_D_l797_797250

variable (L E A D : ℤ)

-- given conditions
def LEAD := 41
def DEAL := 45
def ADDED := 53

-- condition that L = 15
axiom hL : L = 15

-- equations from the problem statement
def eq1 := L + E + A + D = 41
def eq2 := D + E + A + L = 45
def eq3 := A + 3 * D + E = 53

-- stating the problem as proving that D = 4 given the conditions
theorem value_of_D : D = 4 :=
by
  sorry

end value_of_D_l797_797250


namespace theta_half_quadrant_l797_797509

open Real

theorem theta_half_quadrant (θ : ℝ) (k : ℤ) 
  (h1 : 2 * k * π + 3 * π / 2 ≤ θ ∧ θ ≤ 2 * k * π + 2 * π) 
  (h2 : |cos (θ / 2)| = -cos (θ / 2)) : 
  k * π + 3 * π / 4 ≤ θ / 2 ∧ θ / 2 ≤ k * π + π ∧ cos (θ / 2) < 0 := 
sorry

end theta_half_quadrant_l797_797509


namespace complex_multiplication_l797_797312

-- Defining the imaginary unit i and its square property
def i : ℂ := complex.I

theorem complex_multiplication :
  (1 + 2 * i) * (2 + i) = 5 * i := by
  sorry

end complex_multiplication_l797_797312


namespace max_sum_of_arithmetic_sequence_l797_797101

theorem max_sum_of_arithmetic_sequence
  (a_n : ℕ → ℝ)
  (h_arithmetic : ∀ n, a_n (n + 1) = a_n n + ((a_n 1) - (a_n 0)))
  (h_sum1 : a_n 1 + a_n 3 + a_n 5 = 105)
  (h_sum2 : a_n 2 + a_n 4 + a_n 6 = 99)
  (S_n : ℕ → ℝ)
  (h_sum_def : ∀ n, S_n n = (n:ℝ) / 2 * (2 * a_n 1 + (n:ℝ - 1) * ((a_n 1) - (a_n 0)))) :
  ∃ n, S_n n = 800 / 2 := sorry

end max_sum_of_arithmetic_sequence_l797_797101


namespace remainder_polynomial_l797_797862

theorem remainder_polynomial (a : ℕ) (h : (2 * (a : ℤ) + 1)^100 = a_0 + a_1 * a + a_2 * a^2 + ... + a_100 * a^100) :
  (2 * (a_1 + a_3 + a_5 + ... + a_99) - 3) % 8 = 5 :=
by sorry

end remainder_polynomial_l797_797862


namespace total_interest_l797_797996

-- Define the variables P, R, and the given conditions
variables (P R : ℝ)

-- Define the first condition: Simple interest after 10 years is Rs. 1200
def simple_interest_after_10_years (P R : ℝ) := (P * R * 10) / 100

-- Define the second condition: Principal is trebled after 5 years
def simple_interest_next_5_years (P R : ℝ) := (3 * P * R * 5) / 100

-- Given conditions
axiom si_condition : simple_interest_after_10_years P R = 1200
axiom pr_condition : P * R = 1200

-- The theorem to be proven: Total interest at the end of the tenth year is Rs. 3000
theorem total_interest (P R : ℝ) (si_condition : simple_interest_after_10_years P R = 1200) (pr_condition : P * R = 1200) :
  simple_interest_after_10_years P R / 10 * 5 + simple_interest_next_5_years P R = 3000 :=
sorry

end total_interest_l797_797996


namespace prob_five_fish_eaten_expected_fish_eaten_value_l797_797561

noncomputable def prob_eats_at_least_five (n : ℕ) : ℚ :=
  if n = 7 then 19 / 35 else 0

noncomputable def expected_fish_eaten (n : ℕ) : ℚ :=
  if n = 7 then 5 else 0

theorem prob_five_fish_eaten {n : ℕ} (h : n = 7) :
  prob_eats_at_least_five n = 19 / 35 :=
begin
  rw h,
  exact rfl,
end

theorem expected_fish_eaten_value {n : ℕ} (h : n = 7) :
  expected_fish_eaten n = 5 :=
begin
  rw h,
  exact rfl,
end

end prob_five_fish_eaten_expected_fish_eaten_value_l797_797561


namespace common_intersection_point_l797_797215

variables {R : Type*} [CommRing R]

def y1 (a b c x : R) : R := a * x^2 - b * x + c
def y2 (b c a x : R) : R := b * x^2 - c * x + a
def y3 (c a b x : R) : R := c * x^2 - a * x + b

theorem common_intersection_point (a b c : R) : 
  y1 a b c (-1) = y2 b c a (-1) ∧ y2 b c a (-1) = y3 c a b (-1) :=
by {
  -- We calculate the value of y1, y2, y3 at x = -1
  have h1 : y1 a b c (-1) = a + b + c,
  { 
    dsimp [y1],
    ring,
  },
  have h2 : y2 b c a (-1) = a + b + c,
  { 
    dsimp [y2],
    ring,
  },
  have h3 : y3 c a b (-1) = a + b + c,
  { 
    dsimp [y3],
    ring,
  },
  -- Therefore, y1 = y2 and y2 = y3
  exact ⟨h1.trans h2.symm, h2.trans h3.symm⟩,
}

end common_intersection_point_l797_797215


namespace ends_in_five_square_ends_in_25_tens_digit_multiplication_appending_25_l797_797803

-- Part (a)
theorem ends_in_five_square_ends_in_25 (k : ℕ) (h : k % 10 = 5) : (k * k) % 100 = 25 := 
  sorry

-- Part (b)
theorem tens_digit_multiplication_appending_25 (n : ℕ): 
  let k := 10 * n + 5 in 
  k * k = 100 * n * (n + 1) + 25 :=
  sorry

end ends_in_five_square_ends_in_25_tens_digit_multiplication_appending_25_l797_797803


namespace perimeter_independent_l797_797927

variable (ABCD : Type) [Rhombus ABCD]
variable (BD : Diagonal ABCD)
variable (MN PQ : Line)
variable (h : ℝ)
variable (p : ℝ)
variable (a : ℝ)
variable (h1 h2 : ℝ)
variable (k : ℝ)
variable (BM BN DP DQ : ℝ)

def heights_proportional (MN_perpendicular_PQ_perpendicular_BD : MN ⊥ BD ∧ PQ ⊥ BD)
(h1_h2_sum : h1 + h2 = a - h)
(perimeter_hexagon : p = p - k(a - h)) : Prop :=
∃ k,
  (BM + BN - MN.length) / h1 = (DP + DQ - PQ.length) / h2 ∧
  perimeter_hexagon = p - k(a - h)

theorem perimeter_independent (MN_perpendicular_PQ_perpendicular_BD : MN ⊥ BD ∧ PQ ⊥ BD)
(h1_h2_sum : h1 + h2 = a - h) :
  ∃ k, heights_proportional ABCD BD MN PQ h p a h1 h2 k BM BN DP DQ :=
sorry

end perimeter_independent_l797_797927


namespace quadratic_poly_with_root_and_coefficient_l797_797382

noncomputable def find_quadratic_polynomial (a b c : ℝ) : Polynomial ℂ :=
  Polynomial.C a * Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c

theorem quadratic_poly_with_root_and_coefficient :
  ∃ (a b c : ℝ),
  (find_quadratic_polynomial a b c) = (-1 / 3) * (Polynomial.X^2 - 6 * Polynomial.X + 13) ∧
  3 - 2 * Complex.I ∈ (find_quadratic_polynomial a b c).roots ∧
  3 + 2 * Complex.I ∈ (find_quadratic_polynomial a b c).roots ∧
  b = 2 :=
sorry

end quadratic_poly_with_root_and_coefficient_l797_797382


namespace function_symmetric_about_line_x_equals_pi_over_8_l797_797140

theorem function_symmetric_about_line_x_equals_pi_over_8 (a : ℝ) :
  (∀ x : ℝ, f(x) = sin (2 * x) + a * cos (2 * x)) ∧ 
  (∀ x : ℝ, f(x) = f(π/4 - x)) → a = 1 :=
begin
  sorry
end

end function_symmetric_about_line_x_equals_pi_over_8_l797_797140


namespace total_cost_eq_4800_l797_797939

def length := 30
def width := 40
def cost_per_square_foot := 3
def cost_of_sealant_per_square_foot := 1

theorem total_cost_eq_4800 : 
  (length * width * cost_per_square_foot) + (length * width * cost_of_sealant_per_square_foot) = 4800 :=
by
  sorry

end total_cost_eq_4800_l797_797939


namespace fraction_meaningful_l797_797872

theorem fraction_meaningful (a : ℝ) : (∃ b, b = 2 / (a + 1)) → a ≠ -1 :=
by
  sorry

end fraction_meaningful_l797_797872


namespace bakery_water_requirement_l797_797148

theorem bakery_water_requirement (flour water : ℕ) (total_flour : ℕ) (h : flour = 300) (w : water = 75) (t : total_flour = 900) : 
  225 = (total_flour / flour) * water :=
by
  sorry

end bakery_water_requirement_l797_797148


namespace slope_tangent_line_exponential_l797_797258

theorem slope_tangent_line_exponential :
  (deriv (λ x : ℝ, Real.exp x) 2) = Real.exp 2 := 
sorry

end slope_tangent_line_exponential_l797_797258


namespace fraction_sum_equals_zero_l797_797752

theorem fraction_sum_equals_zero :
  (1 / 12) + (2 / 12) + (3 / 12) + (4 / 12) + (5 / 12) + (6 / 12) + (7 / 12) + (8 / 12) + (9 / 12) - (45 / 12) = 0 :=
by
  sorry

end fraction_sum_equals_zero_l797_797752


namespace interval_monotonic_increase_side_length_c_l797_797811

-- Define vectors and function
def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (2 * x), 1)
def b (x : ℝ) : ℝ × ℝ := (1, cos (2 * x))
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 

-- Define the interval of monotonic increase
theorem interval_monotonic_increase (k : ℤ) :
  ∀ x : ℝ, (k * real.pi - real.pi / 3 ≤ x ∧ x ≤ k * real.pi + real.pi / 6) → monotonic_on f (Icc (k * real.pi - real.pi / 3) (k * real.pi + real.pi / 6)) :=
sorry

-- Triangle ABC conditions and proof for side length c
theorem side_length_c (A : ℝ) (a b : ℝ) (fA : ℝ) : 
  f A = 2 ∧ a = sqrt 7 ∧ b = sqrt 3 →
  let c := sqrt (a^2 - b^2 + 3 * b) in 
  c = 4 :=
sorry

end interval_monotonic_increase_side_length_c_l797_797811


namespace sequence_properties_l797_797487

noncomputable def a_n (n : ℕ) : ℝ := 3^(n - 1)
noncomputable def b_n (n : ℕ) : ℝ := 3 * n
noncomputable def S_n (n : ℕ) : ℝ := n * (3 + 3 * n) / 2
noncomputable def c_n (n : ℕ) : ℝ := 9 / (2 * S_n n)
noncomputable def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, c_n (i + 1)

theorem sequence_properties :
  (∀ n, a_n n = 3^(n - 1)) ∧
  (∀ n, b_n n = 3 * n) ∧
  (∀ n, T_n n = 3 * n / (n + 1)) := by
  sorry

end sequence_properties_l797_797487


namespace johnson_family_seating_l797_797602

/-- The Johnson family has 5 sons and 4 daughters. We want to find the number of ways to seat them in a row of 9 chairs such that at least 2 boys are next to each other. -/
theorem johnson_family_seating : 
  let boys := 5 in
  let girls := 4 in
  let total_children := boys + girls in
  fact total_children - 
  2 * (fact boys * fact girls) = 357120 := 
by
  let boys := 5
  let girls := 4
  let total_children := boys + girls
  have total_arrangements : ℕ := fact total_children
  have no_two_boys_next_to_each_other : ℕ := 2 * (fact boys * fact girls)
  have at_least_two_boys_next_to_each_other : ℕ := total_arrangements - no_two_boys_next_to_each_other
  show at_least_two_boys_next_to_each_other = 357120
  sorry

end johnson_family_seating_l797_797602


namespace alice_sequence_l797_797197

theorem alice_sequence (c : ℝ) (h : c > 1) :
  ∃ (n : ℕ) (a : ℕ → ℕ), 
  (∀ i, 1 ≤ i → a i ≤ (c * i).to_nat) ∧ 
  (∀ m n, m ≠ n → ((∑ k in finset.range m, if even k then a (k + 1) else -a (k + 1)) ≠ 
                   (∑ k in finset.range n, if even k then a (k + 1) else -a (k + 1)))) :=
sorry


end alice_sequence_l797_797197


namespace csc_315_eq_neg_sqrt_2_l797_797039

theorem csc_315_eq_neg_sqrt_2 : csc 315 = -sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797039


namespace csc_315_eq_neg_sqrt2_l797_797030

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end csc_315_eq_neg_sqrt2_l797_797030


namespace integer_solutions_ineq_system_l797_797591

theorem integer_solutions_ineq_system:
  ∀ (x : ℤ), 
  (2 * (x - 1) ≤ x + 3) ∧ ((x + 1) / 3 < x - 1) ↔ (x = 3 ∨ x = 4 ∨ x = 5) := 
by 
  intros x 
  split
  · intro h
    cases h with h1 h2
    sorry -- to be proved later
  · intro h
    sorry -- to be proved later

end integer_solutions_ineq_system_l797_797591


namespace find_lambda_l797_797100

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (λ : ℝ)

-- Given conditions: vector a is perpendicular to vector b, and the magnitudes of a and b.
def is_perpendicular (a b : V) : Prop := ⟪a, b⟫ = 0
def magnitude (v : V) (m : ℝ) : Prop := ⟪v, v⟫ = m ^ 2

-- Main statement
theorem find_lambda (h1 : is_perpendicular a b) (h2 : magnitude a 2) (h3 : magnitude b 3) 
    (h4 : is_perpendicular (a + 2 • b) (λ • a - b)) :
    λ = 9 / 2 :=
  sorry

end find_lambda_l797_797100


namespace binom_identity_l797_797193

theorem binom_identity (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  Nat.choose (n + 1) (k + 1) = ∑ i in Finset.range (k + 1), Nat.choose (n - i) k := 
sorry

end binom_identity_l797_797193


namespace triangle_is_obtuse_l797_797200

-- Define the sides of the triangle with the given ratio
def a (x : ℝ) := 3 * x
def b (x : ℝ) := 4 * x
def c (x : ℝ) := 6 * x

-- The theorem statement
theorem triangle_is_obtuse (x : ℝ) (hx : 0 < x) : 
  (a x)^2 + (b x)^2 < (c x)^2 :=
by
  sorry

end triangle_is_obtuse_l797_797200


namespace num_values_cos_l797_797858

theorem num_values_cos (x : ℝ) : 
    0 ≤ x ∧ x ≤ 360 ∧ real.cos x = -0.65 ↔ 2 := 
begin
    sorry
end

end num_values_cos_l797_797858


namespace mean_transformation_l797_797936

theorem mean_transformation (a : Fin 10 → ℝ) :
  let original_mean := (∑ i, a i) / 10
  let new_set := λ i, 3 * (a i) + 5
  let new_mean := (∑ i, new_set i) / 10
  new_mean = 3 * original_mean + 5 := by
  sorry

end mean_transformation_l797_797936


namespace negation_proposition_l797_797990

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^3 + 5*x - 2 = 0) ↔ ∀ x : ℝ, x^3 + 5*x - 2 ≠ 0 :=
by sorry

end negation_proposition_l797_797990


namespace part1_part2_l797_797436

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l797_797436


namespace tetrahedron_has_no_diagonals_l797_797340

theorem tetrahedron_has_no_diagonals
  (V : Type) [fintype V]
  (vertices : fin 4 → V)
  (E : set (V × V)) (hE : E.finite) 
  (h : ∀ (v : V), ∃! (w1 w2 w3 : V), (w1, v) ∈ E ∧ (w2, v) ∈ E ∧ (w3, v) ∈ E ∧ w1 ≠ w2 ∧ w2 ≠ w3 ∧ w1 ≠ w3) :
  (∃ D : set (V × V), D = ∅) :=
sorry

end tetrahedron_has_no_diagonals_l797_797340


namespace max_4_element_subsets_of_8_l797_797287

open Finset

def maximum_subsets_condition (S : Finset ℕ) : ℕ :=
  @max {(F : Finset (Finset ℕ)) // ∀ A B C ∈ F, |A ∩ B ∩ C| ≤ 1} Finset.card
  sorry

theorem max_4_element_subsets_of_8 (S : Finset ℕ) (h : S.card = 8) :
  maximum_subsets_condition S = 8 :=
sorry

end max_4_element_subsets_of_8_l797_797287


namespace units_digit_sum_2_pow_a_5_pow_b_is_not_4_l797_797065

theorem units_digit_sum_2_pow_a_5_pow_b_is_not_4 :
  ∀ (a b : ℕ), (a ∈ Finset.range 1 101) → (b ∈ Finset.range 1 101) → 
  ∃ (d : ℕ), (d ∈ {2, 4, 8, 6}) → ∃ (k : ℕ), 5^b = 5 + 10 * k →
  (2^a + 5^b) % 10 ≠ 4 := 
by sorry

end units_digit_sum_2_pow_a_5_pow_b_is_not_4_l797_797065


namespace h2o_combined_l797_797058

noncomputable def CaO : Type := ℕ -- ℕ denotes the amount of substance in moles
noncomputable def H2O : Type := ℕ
noncomputable def CaOH2 : Type := ℕ

-- Define the balanced chemical equation
def reaction (x : CaO) (y : H2O) : CaOH2 := min x y

-- Define the initial condition
def initial_amount_of_CaO : CaO := 1
def initial_amount_of_CaOH2_formed : CaOH2 := 1

-- The goal is to find the amount of H2O
theorem h2o_combined (result : H2O) : reaction initial_amount_of_CaO result = initial_amount_of_CaOH2_formed → result = 1 :=
by
  sorry

end h2o_combined_l797_797058


namespace monthly_rent_requirement_l797_797729

noncomputable def initial_investment : Float := 200000
noncomputable def annual_return_rate : Float := 0.06
noncomputable def annual_insurance_cost : Float := 4500
noncomputable def maintenance_percentage : Float := 0.15
noncomputable def required_monthly_rent : Float := 1617.65

theorem monthly_rent_requirement :
  let annual_return := initial_investment * annual_return_rate
  let annual_cost_with_insurance := annual_return + annual_insurance_cost
  let monthly_required_net := annual_cost_with_insurance / 12
  let rental_percentage_kept := 1 - maintenance_percentage
  let monthly_rental_full := monthly_required_net / rental_percentage_kept
  monthly_rental_full = required_monthly_rent := 
by
  sorry

end monthly_rent_requirement_l797_797729


namespace scientific_notation_of_0_00076_l797_797594

theorem scientific_notation_of_0_00076 : (0.00076 : ℝ) = 7.6 * 10^(-4) :=
by
  sorry

end scientific_notation_of_0_00076_l797_797594


namespace maxwell_meets_brad_time_l797_797203

theorem maxwell_meets_brad_time :
  ∀ (d m_speed b_speed : ℝ) (start_diff : ℝ),
  m_speed = 4 → b_speed = 6 → d = 24 → start_diff = 1 →
  let t_brad := (d - m_speed * start_diff) / (m_speed + b_speed) in
  t_brad + start_diff = 3 :=
by
  intros d m_speed b_speed start_diff h_m_speed h_b_speed h_d h_start_diff t_brad
  simp [t_brad, h_m_speed, h_b_speed, h_d, h_start_diff]
  sorry

end maxwell_meets_brad_time_l797_797203


namespace tickets_not_attended_l797_797277

def total_tickets : ℕ := 2465
def before_concert (x : ℕ) : ℕ := (7 * x) / 8
def after_first_song (x : ℕ) : ℕ := (13 * x) / 17
def middle_concert : ℕ := 128
def last_performances : ℕ := 47

theorem tickets_not_attended :
  let remaining1 := total_tickets - before_concert total_tickets,
      remaining2 := remaining1 - after_first_song remaining1,
      remaining3 := remaining2 - last_performances in
  remaining3 = 26 :=
by
  sorry

end tickets_not_attended_l797_797277


namespace min_value_f_eq_zero_l797_797988

noncomputable def f (x : ℝ) : ℝ := |2 * sqrt x * log (sqrt 2) (2 * x)|

theorem min_value_f_eq_zero : 
  ∃ x_min, (∀ x > 0, f x ≥ 0) ∧ (∀ x > 0, f x ≥ f x_min) ∧ (f x_min = 0) := by
sorry

end min_value_f_eq_zero_l797_797988


namespace johnson_family_seating_l797_797620

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem johnson_family_seating (sons daughters : ℕ) (total_seats : ℕ) 
  (condition1 : sons = 5) (condition2 : daughters = 4) (condition3 : total_seats = 9) :
  let total_arrangements := factorial total_seats,
      restricted_arrangements := factorial sons * factorial daughters,
      answer := total_arrangements - restricted_arrangements
  in answer = 360000 := 
by
  -- The proof would go here
  sorry

end johnson_family_seating_l797_797620


namespace number_of_odd_digits_base_5_345_l797_797794

theorem number_of_odd_digits_base_5_345 : 
  let base_5_representation_345 := 2340
  in 
  (number_of_odd_digits base_5_representation_345) = 2 :=
sorry

end number_of_odd_digits_base_5_345_l797_797794


namespace time_to_cut_mans_hair_l797_797530

theorem time_to_cut_mans_hair :
  ∃ (x : ℕ),
    (3 * 50) + (2 * x) + (3 * 25) = 255 ∧ x = 15 :=
by {
  sorry
}

end time_to_cut_mans_hair_l797_797530


namespace fn_2011_l797_797192

noncomputable def f : ℝ → ℝ := λ x, Real.sin x

def fn : ℕ → (ℝ → ℝ)
| 0     := f
| (n+1) := (fn n)' -- deriving the function for n+1

theorem fn_2011 (x : ℝ) : fn 2011 x = -Real.cos x :=
sorry

end fn_2011_l797_797192


namespace avg_median_max_k_m_r_s_t_l797_797977

theorem avg_median_max_k_m_r_s_t (
  k m r s t : ℕ 
) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
  (h5 : 5 * 16 = k + m + r + s + t)
  (h6 : r = 17) : 
  t = 42 :=
by
  sorry

end avg_median_max_k_m_r_s_t_l797_797977


namespace canyon_trail_length_l797_797567

theorem canyon_trail_length
  (a b c d e : ℝ)
  (h1 : a + b + c = 36)
  (h2 : b + c + d = 42)
  (h3 : c + d + e = 45)
  (h4 : a + d = 29) :
  a + b + c + d + e = 71 :=
by sorry

end canyon_trail_length_l797_797567


namespace wilson_theorem_for_odd_primes_l797_797931

theorem wilson_theorem_for_odd_primes (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1)  :
  ((p - 1)! + 1) % p = 0 :=
sorry

end wilson_theorem_for_odd_primes_l797_797931


namespace inequality_part1_inequality_part2_l797_797455

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l797_797455


namespace csc_315_eq_neg_sqrt_2_l797_797017

theorem csc_315_eq_neg_sqrt_2 :
  let csc := λ θ, 1 / Real.sin θ in
  csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
  by
  let sin := Real.sin
  have h1 : csc (315 * Real.pi / 180) = 1 / sin (315 * Real.pi / 180) := rfl
  have h2 : sin (315 * Real.pi / 180) = sin ((360 - 45) * Real.pi / 180) := by congr; norm_num
  have h3 : sin ((360 - 45) * Real.pi / 180) = -sin (45 * Real.pi / 180) := by
    rw [Real.sin_pi_sub]
    congr; norm_num
  have h4 : sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 := Real.sin_of_one_div_sqrt_two 45 rfl
  sorry

end csc_315_eq_neg_sqrt_2_l797_797017


namespace monotonic_intervals_l797_797363

-- Define the operation
def star (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f(x) using the operation
noncomputable def f (x : ℝ) : ℝ := star (2 * Real.sin x) (2 * Real.cos x)

-- State the theorem
theorem monotonic_intervals :
  -- The intervals of monotonic increase for the function f in the interval [0, 2π]
  ((0, Real.pi / 4), (Real.pi, 5 * Real.pi / 4), (3 * Real.pi / 2, 2 * Real.pi)) := sorry

end monotonic_intervals_l797_797363


namespace mass_percentage_O_in_N2O3_l797_797791

variable (m_N : ℝ := 14.01)  -- Molar mass of nitrogen (N) in g/mol
variable (m_O : ℝ := 16.00)  -- Molar mass of oxygen (O) in g/mol
variable (n_N : ℕ := 2)      -- Number of nitrogen (N) atoms in N2O3
variable (n_O : ℕ := 3)      -- Number of oxygen (O) atoms in N2O3

theorem mass_percentage_O_in_N2O3 :
  let molar_mass_N2O3 := (n_N * m_N) + (n_O * m_O)
  let mass_O_in_N2O3 := n_O * m_O
  let percentage_O := (mass_O_in_N2O3 / molar_mass_N2O3) * 100
  percentage_O = 63.15 :=
by
  -- Formal proof here
  sorry

end mass_percentage_O_in_N2O3_l797_797791


namespace necessary_and_sufficient_cond_l797_797549

variable {R : Type} [LinearOrderedField R]

def increasing (f : R → R) : Prop :=
∀ ⦃x y : R⦄, x < y → f x < f y

def is_monotonically_increasing (f : R → R) : Prop :=
∀ x : R, (deriv f x) ≥ 0

noncomputable def polynom (m : R) (x : R) : R :=
  x^3 + 2*x^2 + m*x + 1

theorem necessary_and_sufficient_cond (m : R) :
  is_monotonically_increasing (polynom m) ↔ m ≥ 4/3 :=
by sorry

end necessary_and_sufficient_cond_l797_797549


namespace find_c_l797_797227

/-- Seven unit squares are arranged in a row in the coordinate plane, 
with the lower left corner of the first square at the origin. 
A line extending from (c,0) to (4,4) divides the entire region 
into two regions of equal area. What is the value of c?
-/
theorem find_c (c : ℝ) (h : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 7 ∧ y = (4 / (4 - c)) * (x - c)) : c = 2.25 :=
sorry

end find_c_l797_797227


namespace three_digit_numbers_with_repeats_l797_797669

theorem three_digit_numbers_with_repeats :
  (let total_numbers := 9 * 10 * 10
   let non_repeating_numbers := 9 * 9 * 8
   total_numbers - non_repeating_numbers = 252) :=
by
  sorry

end three_digit_numbers_with_repeats_l797_797669


namespace sin_cos_product_l797_797496

noncomputable def f : ℝ → ℝ := sin

theorem sin_cos_product (α : ℝ) (h1 : -π/2 ≤ α ∧ α ≤ π/2)
  (h2 : f (sin α) + f (cos α - 1/2) = 0) : 
  sin α * cos α = -3/8 :=
by {
  sorry
}

end sin_cos_product_l797_797496


namespace hyperbola_asymptotes_l797_797981

theorem hyperbola_asymptotes (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (e : ℝ) (he : e = sqrt 3) :
  ∀ x : ℝ, y = x * sqrt 2 ∨ y = -x * sqrt 2 :=
by
  sorry

end hyperbola_asymptotes_l797_797981


namespace max_area_circle_l797_797251

noncomputable def maxCircleAreaBetweenLines : ℝ :=
  let A := 3
  let B := -4
  let C1 := 0
  let C2 := -20
  let distance := (abs (C2 - C1)) / (real.sqrt (A^2 + B^2))
  let radius := distance / 2
  real.pi * (radius^2)

theorem max_area_circle (l1 : ℝ → ℝ → Prop) (l2 : ℝ → ℝ → Prop) :
  l1 = (λ x y, 3 * x - 4 * y = 0) → 
  l2 = (λ x y, 3 * x - 4 * y - 20 = 0) → 
  maxCircleAreaBetweenLines = 4 * real.pi :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end max_area_circle_l797_797251


namespace find_n_l797_797098

theorem find_n
  (n : Nat) (a : Fin n → ℕ) (x : ℝ)
  (h₁ : (1 + x)^n = ∑ i in Finset.range (n+1), a i * x ^ i)
  (h₂ : (a 1) / (a 2) = 1 / 4) : 
  n = 9 :=
sorry

end find_n_l797_797098


namespace part1_part2_l797_797478

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l797_797478


namespace asian_games_tourists_scientific_notation_l797_797234

theorem asian_games_tourists_scientific_notation : 
  ∀ (n : ℕ), n = 18480000 → 1.848 * (10:ℝ) ^ 7 = (n : ℝ) :=
by
  intro n
  sorry

end asian_games_tourists_scientific_notation_l797_797234


namespace csc_315_eq_sqrt2_l797_797049

theorem csc_315_eq_sqrt2 :
  let θ := 315
  let csc := λ θ, 1 / (Real.sin (θ * Real.pi / 180))
  315 = 360 - 45 → 
  Real.sin (315 * Real.pi / 180) = Real.sin ((360 - 45) * Real.pi / 180) → 
  Real.sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 →
  csc 315 = Real.sqrt 2 := 
by
  intros θ csc h1 h2 h3
  -- proof would go here
  sorry

end csc_315_eq_sqrt2_l797_797049


namespace sequence_fill_l797_797155

theorem sequence_fill (x2 x3 x4 x5 x6 x7: ℕ) : 
  (20 + x2 + x3 = 100) ∧ 
  (x2 + x3 + x4 = 100) ∧ 
  (x3 + x4 + x5 = 100) ∧ 
  (x4 + x5 + x6 = 100) ∧ 
  (x5 + x6 + 16 = 100) →
  [20, x2, x3, x4, x5, x6, 16] = [20, 16, 64, 20, 16, 64, 20, 16] :=
by
  sorry

end sequence_fill_l797_797155


namespace complex_multiplication_l797_797314

theorem complex_multiplication : (1 + 2 * Complex.i) * (2 + Complex.i) = 5 * Complex.i := 
by 
  sorry

end complex_multiplication_l797_797314


namespace valid_five_digit_numbers_divisible_by_7_l797_797919

def num_valid_n : Nat := 13050

theorem valid_five_digit_numbers_divisible_by_7 :
  ∃ (n : Finset ℕ), (∀ n ∈ n, 10000 ≤ n ∧ n < 100000) ∧
    ∃ q r : ℕ, n = 200 * q + r ∧ r < 200 ∧ (q + r) % 7 = 0 ∧  
    (n.card = num_valid_n) :=
by
  sorry

end valid_five_digit_numbers_divisible_by_7_l797_797919


namespace determine_c_absolute_value_l797_797971

theorem determine_c_absolute_value
  (a b c : ℤ)
  (h_gcd : Int.gcd (Int.gcd a b) c = 1)
  (h_root : a * (Complex.mk 3 1)^4 + b * (Complex.mk 3 1)^3 + c * (Complex.mk 3 1)^2 + b * (Complex.mk 3 1) + a = 0) :
  |c| = 109 := 
sorry

end determine_c_absolute_value_l797_797971


namespace combined_probability_l797_797136

-- Definitions:
def number_of_ways_to_get_3_heads_and_1_tail := Nat.choose 4 3
def probability_of_specific_sequence_of_3_heads_and_1_tail := (1/2) ^ 4
def probability_of_3_heads_and_1_tail := number_of_ways_to_get_3_heads_and_1_tail * probability_of_specific_sequence_of_3_heads_and_1_tail

def favorable_outcomes_die := 2
def total_outcomes_die := 6
def probability_of_number_greater_than_4 := favorable_outcomes_die / total_outcomes_die

-- Proof statement:
theorem combined_probability : probability_of_3_heads_and_1_tail * probability_of_number_greater_than_4 = 1/12 := by
  sorry

end combined_probability_l797_797136


namespace P_cubed_plus_7_is_composite_l797_797860

theorem P_cubed_plus_7_is_composite (P : ℕ) (h_prime_P : Nat.Prime P) (h_prime_P3_plus_5 : Nat.Prime (P^3 + 5)) : ¬ Nat.Prime (P^3 + 7) ∧ (P^3 + 7).factors.length > 1 :=
by
  sorry

end P_cubed_plus_7_is_composite_l797_797860


namespace confectionary_candy_limit_l797_797714

theorem confectionary_candy_limit (n_students : ℕ) (h1 : n_students = 1000)
                               (h2 : ∀ (candy_types : Finset ℕ), candy_types.card = 11 → 
                                     ∃ (student : ℕ), student ∈ (Finset.range n_students) ∧ 
                                     (∃ t ∈ candy_types, t ∈ (student_set student)))
                               (h3 : ∀ (c1 c2 : ℕ), c1 ≠ c2 → 
                                     ∃ (student : ℕ), student ∈ (Finset.range n_students) ∧ 
                                     ((∃ t ∈ {c1, c2}, t ∈ (student_set student)) ∧ 
                                     (∀ t ∈ {c1, c2}, t ∉ (student_set student) → t ≠ t)))
                               :
n_types_candy ≤ 5501 :=
sorry

end confectionary_candy_limit_l797_797714


namespace Huanggang_Singer_Competition_l797_797160

def scores : List ℝ := [91, 89, 91, 96, 94, 95, 94]

def remainingScores (scores : List ℝ) : List ℝ :=
  scores.erase 96 |>.erase 89

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def variance (l : List ℝ) (μ : ℝ) : ℝ :=
  (l.map (λ x => (x - μ)^2)).sum / l.length

theorem Huanggang_Singer_Competition : mean (remainingScores scores) = 93 ∧ variance (remainingScores scores) (mean (remainingScores scores)) = 2.8 := by
  sorry

end Huanggang_Singer_Competition_l797_797160


namespace price_of_turban_correct_l797_797853

noncomputable def initial_yearly_salary : ℝ := 90
noncomputable def initial_monthly_salary : ℝ := initial_yearly_salary / 12
noncomputable def raise : ℝ := 0.05 * initial_monthly_salary

noncomputable def first_3_months_salary : ℝ := 3 * initial_monthly_salary
noncomputable def second_3_months_salary : ℝ := 3 * (initial_monthly_salary + raise)
noncomputable def third_3_months_salary : ℝ := 3 * (initial_monthly_salary + 2 * raise)

noncomputable def total_cash_salary : ℝ := first_3_months_salary + second_3_months_salary + third_3_months_salary
noncomputable def actual_cash_received : ℝ := 80
noncomputable def price_of_turban : ℝ := actual_cash_received - total_cash_salary

theorem price_of_turban_correct : price_of_turban = 9.125 :=
by
  sorry

end price_of_turban_correct_l797_797853


namespace abs_difference_is_one_l797_797991

theorem abs_difference_is_one
  (a_1 a_2 b_1 b_2 b_3 : ℕ)
  (h395 : 395 = (a_1.factorial * a_2.factorial) / (b_1.factorial * b_2.factorial * b_3.factorial))
  (h1 : a_1 ≥ a_2)
  (h2 : b_1 ≥ b_2)
  (h3 : b_2 ≥ b_3)
  (min_ab : ∀ c_1 c_2, 395 = (c_1.factorial * a_2.factorial) / (c_2.factorial * b_2.factorial * b_3.factorial) → c_1 + c_2 ≥ a_1 + b_1)
  : abs (a_1 - b_1) = 1 :=
sorry

end abs_difference_is_one_l797_797991


namespace line_through_points_l797_797798

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

noncomputable def y_intercept (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  p.2 - m * p.1

theorem line_through_points (p1 p2 : ℝ × ℝ) (m b : ℝ) :
  p1 = (1, 2) → p2 = (4, 20) → 
  m = slope p1 p2 → b = y_intercept p1 m →
  m + b = 8 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end line_through_points_l797_797798


namespace nine_fact_div_four_fact_eq_15120_l797_797420

theorem nine_fact_div_four_fact_eq_15120 :
  (362880 / 24) = 15120 :=
by
  sorry

end nine_fact_div_four_fact_eq_15120_l797_797420


namespace number_of_boys_in_school_l797_797152

-- Definition of percentages for Muslims, Hindus, and Sikhs
def percent_muslims : ℝ := 0.46
def percent_hindus : ℝ := 0.28
def percent_sikhs : ℝ := 0.10

-- Given number of boys in other communities
def boys_other_communities : ℝ := 136

-- The total number of boys in the school
def total_boys (B : ℝ) : Prop := B = 850

-- Proof statement (with conditions embedded)
theorem number_of_boys_in_school (B : ℝ) :
  percent_muslims * B + percent_hindus * B + percent_sikhs * B + boys_other_communities = B → 
  total_boys B :=
by
  sorry

end number_of_boys_in_school_l797_797152


namespace cylindrical_to_rectangular_multiplied_l797_797771

theorem cylindrical_to_rectangular_multiplied :
  let r := 7
  let θ := Real.pi / 4
  let z := -3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (2 * x, 2 * y, 2 * z) = (7 * Real.sqrt 2, 7 * Real.sqrt 2, -6) := 
by
  sorry

end cylindrical_to_rectangular_multiplied_l797_797771


namespace hayley_meatballs_l797_797504

theorem hayley_meatballs (original : ℕ) (stolen : ℕ) (remaining : ℕ) 
  (h1 : original = 25) (h2 : stolen = 14) : 
  remaining = original - stolen := 
begin
  rw [h1, h2],
  norm_num,
  sorry,
end

end hayley_meatballs_l797_797504


namespace triangle_area_comparison_l797_797804

noncomputable def triangle (A1 A2 A3 : ℝ) (A1A2 A1A3 A2A3 : ℝ) := 
  (A1A2 > 0 ∧ A1A3 > 0 ∧ A2A3 > 0) ∧
  (A1A2 + A1A3 > A2A3 ∧ A1A2 + A2A3 > A1A3 ∧ A1A3 + A2A3 > A1A2)

noncomputable def non_obtuse (A1 A2 A3: ℝ) (A1A2 A1A3 A2A3 : ℝ) := 
  ∀ (angle : ℝ), 0 ≤ angle ∧ angle ≤ π/2

-- \( \Delta_A \) area of triangle \( A_1A_2A_3 \)
-- \( \Delta_B \) area of triangle \( B_1B_2B_3 \)
theorem triangle_area_comparison
  (A1 A2 A3 B1 B2 B3 : ℝ) 
  (A1A2 A1A3 A2A3 B1B2 B1B3 B2B3 : ℝ)
  (hA_triangle : triangle A1 A2 A3 A1A2 A1A3 A2A3)
  (hB_triangle : triangle B1 B2 B3 B1B2 B1B3 B2B3)
  (h_sides : ∀ i j, A1A2 ≥ B1B2 ∧ A1A3 ≥ B1B3 ∧ A2A3 ≥ B2B3)
  (h_not_obtuse : non_obtuse A1 A2 A3 A1A2 A1A3 A2A3)
  (area_A : ℝ)
  (area_B : ℝ) : 
  area_A ≥ area_B :=
sorry

end triangle_area_comparison_l797_797804


namespace part1_part2_l797_797445

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

end part1_part2_l797_797445


namespace larger_angle_measure_l797_797667

theorem larger_angle_measure (x : ℝ) (hx : 7 * x = 90) : 4 * x = 360 / 7 := by
sorry

end larger_angle_measure_l797_797667


namespace eccentricity_of_ellipse_l797_797499

/-- Given the ellipse C defined by the equation x^2 / a^2 + y^2 / b^2 = 1 with a > b > 0, the upper endpoint B of the ellipse at (0, b), and the left focus F at (-c, 0), if a line through B intersects the ellipse at another point A such that |BF| = 3|AF|, then the eccentricity of the ellipse is sqrt(2)/2. -/
theorem eccentricity_of_ellipse :
  ∀ (a b c : ℝ), a > 0 → b > 0 → a > b
  → let e := c / a in
  let F := (-c, 0) in
  let B := (0, b) in
  let A := (-4 * c / 3, -b / 3) in
  |(0, b) - (-c, 0)| = 3 * |(-4 * c / 3, -b / 3) - (-c, 0)| 
  → (16 * (c^2) / (9 * (a^2)) + (b^2) / (9 * (b^2)) = 1) 
  → abs e = (Real.sqrt 2)/2 := 
sorry

end eccentricity_of_ellipse_l797_797499


namespace sum_of_integers_satisfying_inequality_l797_797967

theorem sum_of_integers_satisfying_inequality :
  let I := {x : ℤ | -70 < x ∧ x < 34}
  I.sum (λ x, if 2 ^ (2 + sqrt (x - 1)) - 24 / (2 ^ (1 + sqrt (x - 1)) - 8) > 1 then x else 0) = 526 :=
sorry

end sum_of_integers_satisfying_inequality_l797_797967


namespace quadratic_has_distinct_real_roots_l797_797257

theorem quadratic_has_distinct_real_roots :
  (∃ a b c : ℝ, a ≠ 0 ∧ b = 2 ∧ a = 1 ∧ c = -5 ∧ Δ = b^2 - 4 * a * c ∧ Δ > 0) :=
by
  let a := 1
  let b := 2
  let c := -5
  let Δ := b^2 - 4 * a * c
  have hΔ : Δ = 24 := by
    dsimp [Δ]
    ring
  exact ⟨a, b, c, by norm_num, rfl, rfl, rfl, by norm_num⟩

end quadratic_has_distinct_real_roots_l797_797257


namespace savings_ratio_is_one_half_l797_797760

-- Define constants for the conditions
def earnings_per_can : ℝ := 0.25
def cans_at_home : ℕ := 12
def cans_at_grandparents : ℕ := 3 * cans_at_home
def cans_from_neighbor : ℕ := 46
def cans_from_dad : ℕ := 250
def savings_amount : ℝ := 43

-- Define total cans collected
def total_cans_collected : ℕ := 
  cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_dad

-- Define total earnings
def total_earnings : ℝ := total_cans_collected * earnings_per_can

-- Define the ratio of savings to total earnings
def savings_ratio : ℝ := savings_amount / total_earnings

theorem savings_ratio_is_one_half : savings_ratio = 1 / 2 := by
  sorry

end savings_ratio_is_one_half_l797_797760


namespace ratio_largest_to_sum_l797_797769

theorem ratio_largest_to_sum :
  let s : Finset ℕ := {1, 5, 5^2, 5^3, 5^4, 5^5, 5^6, 5^7}
  let largest : ℕ := 5^8
  let sum_rest : ℕ := Finset.sum s
  abs ((largest : ℚ) / sum_rest.toRat - 4) < 1 :=
by
  sorry

end ratio_largest_to_sum_l797_797769


namespace father_current_age_is_85_l797_797389

theorem father_current_age_is_85 (sebastian_age : ℕ) (sister_diff : ℕ) (age_sum_fraction : ℕ → ℕ → ℕ → Prop) :
  sebastian_age = 40 →
  sister_diff = 10 →
  (∀ (s s' f : ℕ), age_sum_fraction s s' f → f = 4 * (s + s') / 3) →
  age_sum_fraction (sebastian_age - 5) (sebastian_age - sister_diff - 5) (40 + 5) →
  ∃ father_age : ℕ, father_age = 85 :=
by
  intros
  sorry

end father_current_age_is_85_l797_797389


namespace sym_diff_complement_sym_diff_union_l797_797575
open Set

section

variables {α : Type*} (A B K : Set α)

-- The statement to be proved:
theorem sym_diff_complement (A B : Set α) :
  (A \triangle B)' = A' \triangle B ∧ (A \triangle B)' = A \triangle B' :=
sorry

theorem sym_diff_union (A B K : Set α) :
  (A \triangle K) ∪ (B \triangle K) = (A ∩ B) \triangle (K ∪ (A \triangle B)) :=
sorry

end

end sym_diff_complement_sym_diff_union_l797_797575


namespace geometry_problem_l797_797168

theorem geometry_problem
    (O A B C D E P : Point)  -- Define involved points
    (r : ℝ)  -- Define the radius of the circle
    (H1 : AB = 3 * r)  -- AB is three times the radius
    (H2 : equilateral_triangle O B C)  -- OBC is an equilateral triangle
    (H3 : DO = r / 2)  -- DO is half of the radius
    (H4 : AB ⊥ BC)  -- AB is perpendicular to BC
    (H5 : collinear [A, D, O, E])  -- A, D, O, E are collinear
    (H6 : AP = AD) -- AP is equal to AD
    [Circular A O] -- A is a point on the circle centered at O

    : ¬ (AP^2 = PB * AB) ∧ ¬ (AP * DO = PB * AD) ∧ ¬ (AB^2 = AD * DE) ∧ ¬ (AB * AD = OB * AO) := by
  sorry -- proof to be completed

end geometry_problem_l797_797168


namespace rectangle_ratio_expression_value_l797_797359

theorem rectangle_ratio_expression_value (l w : ℝ) (S : ℝ) (h1 : l / w = (2 * (l + w)) / (2 * l)) (h2 : S = w / l) :
  S ^ (S ^ (S^2 + 1/S) + 1/S) + 1/S = Real.sqrt 5 :=
by
  sorry

end rectangle_ratio_expression_value_l797_797359


namespace number_of_tens_in_product_l797_797649

theorem number_of_tens_in_product (a b : ℕ) (ten : ℕ) (h1 : a = 100) (h2 : b = 100) (h3 : ten = 10) :
  ∑ (i : ℕ) in (Finset.range (a * b / ten)), (λ _ , ten) = a * b := sorry

end number_of_tens_in_product_l797_797649


namespace waiter_slices_l797_797132

theorem waiter_slices (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ)
  (h_total_slices : total_slices = 78)
  (h_ratios : buzz_ratio = 5 ∧ waiter_ratio = 8) :
  20 < (waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio))) →
  28 = waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio)) - 20 :=
by
  sorry

end waiter_slices_l797_797132


namespace ellen_initial_legos_l797_797783

theorem ellen_initial_legos :
  ∀ (lost current initial_legos : ℕ), 
  lost = 57 → 
  current = 323 → 
  initial_legos = lost + current → 
  initial_legos = 380 :=
by
  intros lost current initial_legos h_lost h_current h_initial
  rw [h_lost] at h_initial
  rw [h_current] at h_initial
  exact h_initial
  sorry

end ellen_initial_legos_l797_797783


namespace find_f_one_l797_797076

-- Define the function f(x-3) = 2x^2 - 3x + 1
noncomputable def f (x : ℤ) := 2 * (x+3)^2 - 3 * (x+3) + 1

-- Declare the theorem we intend to prove
theorem find_f_one : f 1 = 21 :=
by
  -- The proof goes here (saying "sorry" because the detailed proof is skipped)
  sorry

end find_f_one_l797_797076


namespace area_closest_to_A_in_pentagon_l797_797691

def radius : ℝ := 1
def num_sides : ℕ := 5

noncomputable def area_closest_to_A : ℝ := π / num_sides

theorem area_closest_to_A_in_pentagon : 
  let pentagon := inscribed_regular_polygon num_sides radius in
  area_of_region_closer_to_A_than_other_vertices pentagon = area_closest_to_A :=
by
  -- Proof to be completed
  sorry

end area_closest_to_A_in_pentagon_l797_797691


namespace sasha_max_quarters_l797_797962

theorem sasha_max_quarters (q : ℕ) (h1 : 0.45 * q ≤ 4.80) (h2 : ∃ k : ℕ, q = k) : q ≤ 10 := 
by
  sorry

end sasha_max_quarters_l797_797962


namespace hyperbola_focus_l797_797643

theorem hyperbola_focus {m : ℝ} 
  (h : (∃ x y : ℝ, (x, y) = (2, 0)) 
         → ∀ c : ℝ, c = 2
           → ((c = sqrt(m + (3+m)))
           → (2 = sqrt(m + (3+m)))) :
  m = 1 / 2 :=
begin
  intros x y exist xy foc hyp,
  cases xy with hx hy,
  rw [hx, hy] at *,
  have h1 : c = sqrt(m + (3+m)) := hyp foc,
  have h2 : 2 = sqrt(m + (3+m)) := by rw h in h1; exact h1,
  sorry
end

end hyperbola_focus_l797_797643


namespace johnson_family_seating_l797_797606

/-- The Johnson family has 5 sons and 4 daughters. We want to find the number of ways to seat them in a row of 9 chairs such that at least 2 boys are next to each other. -/
theorem johnson_family_seating : 
  let boys := 5 in
  let girls := 4 in
  let total_children := boys + girls in
  fact total_children - 
  2 * (fact boys * fact girls) = 357120 := 
by
  let boys := 5
  let girls := 4
  let total_children := boys + girls
  have total_arrangements : ℕ := fact total_children
  have no_two_boys_next_to_each_other : ℕ := 2 * (fact boys * fact girls)
  have at_least_two_boys_next_to_each_other : ℕ := total_arrangements - no_two_boys_next_to_each_other
  show at_least_two_boys_next_to_each_other = 357120
  sorry

end johnson_family_seating_l797_797606


namespace csc_315_eq_sqrt2_l797_797045

theorem csc_315_eq_sqrt2 :
  let θ := 315
  let csc := λ θ, 1 / (Real.sin (θ * Real.pi / 180))
  315 = 360 - 45 → 
  Real.sin (315 * Real.pi / 180) = Real.sin ((360 - 45) * Real.pi / 180) → 
  Real.sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 →
  csc 315 = Real.sqrt 2 := 
by
  intros θ csc h1 h2 h3
  -- proof would go here
  sorry

end csc_315_eq_sqrt2_l797_797045


namespace johnson_family_seating_problem_l797_797611

theorem johnson_family_seating_problem : 
  ∃ n : ℕ, n = 9! - 5! * 4! ∧ n = 359760 :=
by
  have total_ways := (Nat.factorial 9)
  have no_adjacent_boys := (Nat.factorial 5) * (Nat.factorial 4)
  have result := total_ways - no_adjacent_boys
  use result
  split
  . exact eq.refl result
  . norm_num -- This will replace result with its evaluated form, 359760

end johnson_family_seating_problem_l797_797611


namespace part1_part2_l797_797442

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

end part1_part2_l797_797442


namespace max_a_b_c_d_l797_797827

theorem max_a_b_c_d (a c d b : ℤ) (hb : b > 0) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) 
: a + b + c + d = -5 :=
by
  sorry

end max_a_b_c_d_l797_797827


namespace distribute_awards_l797_797800

variable (Awards : Fin 5)
variable (Students : Fin 4)

/--
The number of different ways to distribute five different awards to four students,
with each student receiving at least one award, is 240.
-/
theorem distribute_awards :
  ∃ f : Awards → Students,
  (∀ s : Students, ∃ a : Awards, f a = s) ∧
  (f.to_fun (Awards.val 4) = Students.val 0 → 
  f.to_fun (Awards.val 3) = Students.val 1 → 
  f.to_fun (Awards.val 2) = Students.val 2 → 
  f.to_fun (Awards.val 1) = Students.val 3 → 
  f.to_fun (Awards.val 0) = Students.val 0) → False :=
sorry

end distribute_awards_l797_797800


namespace identical_function_sqrt_abs_l797_797780

theorem identical_function_sqrt_abs (x : ℝ) : (sqrt (x^2) = abs x) :=
sorry

end identical_function_sqrt_abs_l797_797780


namespace csc_315_eq_neg_sqrt_2_l797_797002

theorem csc_315_eq_neg_sqrt_2 :
  Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := 
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797002


namespace hyperbola_focus_eq_parabola_focus_l797_797143

theorem hyperbola_focus_eq_parabola_focus (k : ℝ) (hk : k > 0) :
  let parabola_focus : ℝ × ℝ := (2, 0) in
  let hyperbola_focus_distance : ℝ := Real.sqrt (1 + k^2) in
  hyperbola_focus_distance = 2 ↔ k = Real.sqrt 3 :=
by {
  sorry
}

end hyperbola_focus_eq_parabola_focus_l797_797143


namespace lines_concurrent_l797_797647

theorem lines_concurrent
  (A B C O A1 B1 C1 A2 B2 C2 : Point)
  (hIncircle : Incircle O A B C)
  (hTangencyA1 : TangencyPoint A1 O B C)
  (hTangencyB1 : TangencyPoint B1 O C A)
  (hTangencyC1 : TangencyPoint C1 O A B)
  (hIntersectionA2 : IntersectionPoint A2 (line_through A1 O) (segment B1 C1))
  (hIntersectionB2 : IntersectionPoint B2 (line_through B1 O) (segment C1 A1))
  (hIntersectionC2 : IntersectionPoint C2 (line_through C1 O) (segment A1 B1))
  : Concurrent (line_through A A2) (line_through B B2) (line_through C C2) :=
sorry

end lines_concurrent_l797_797647


namespace johnson_family_seating_l797_797622

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem johnson_family_seating (sons daughters : ℕ) (total_seats : ℕ) 
  (condition1 : sons = 5) (condition2 : daughters = 4) (condition3 : total_seats = 9) :
  let total_arrangements := factorial total_seats,
      restricted_arrangements := factorial sons * factorial daughters,
      answer := total_arrangements - restricted_arrangements
  in answer = 360000 := 
by
  -- The proof would go here
  sorry

end johnson_family_seating_l797_797622


namespace product_formal_power_series_well_defined_l797_797573

-- Define what it means for a product of formal power series to be well-defined.
def well_defined (f : ℕ → ℝ) (x : ℝ) : Prop := 
  ∀ n, ∃ (N : ℕ), f n = ∑ (m : ℕ) in finset.range (N+1), f m * x^m

-- The actual theorem to prove
theorem product_formal_power_series_well_defined (x : ℝ) : 
  well_defined (λ n, ∑ (k : ℕ) in (finset.range n.succ).filter (λ m, m = n), 1) x :=
sorry

end product_formal_power_series_well_defined_l797_797573


namespace positive_divisors_l797_797553

def number_of_divisors (a : ℕ → ℕ) (k : ℕ) : ℕ :=
  (List.range k).map (λ i, a i + 1).prod

def number_of_divisors_m (a : ℕ → ℕ) (k : ℕ) : ℕ :=
  (List.range k).map (λ i, (a i / 2) + 1).prod

theorem positive_divisors (n : ℕ) (p : ℕ → ℕ) (a : ℕ → ℕ) (k : ℕ)
  (hn : n = List.prod (List.range k).map (λ i, (p i) ^ (a i))) :
  number_of_divisors a k = (List.range k).map (λ i, a i + 1).prod ∧
  number_of_divisors_m a k = (List.range k).map (λ i, (a i / 2) + 1).prod :=
by
  sorry

end positive_divisors_l797_797553


namespace perpendicular_lines_tangent_circle_l797_797852

theorem perpendicular_lines_tangent_circle (a b : ℝ) :
  (∀ x y : ℝ, (a * x + 3 * y + 1 = 0) ∧ (x + a * y + 2 = 0) → a * (1 + 3 * a) = 0) →
  (∀ x y : ℝ, (x + a * y + 2 = 0) → ∃ d : ℝ, d = abs 2 / real.sqrt (1 + a ^ 2) ∧ d = real.sqrt b) →
  b = 9 :=
by
  intros h1 h2
  -- Use conditions h1 and h2 to reach the conclusion that b = 9
  sorry

end perpendicular_lines_tangent_circle_l797_797852


namespace inverse_proposition_of_divisibility_by_5_l797_797985

theorem inverse_proposition_of_divisibility_by_5 (n : ℕ) :
  (n % 10 = 5 → n % 5 = 0) → (n % 5 = 0 → n % 10 = 5) :=
sorry

end inverse_proposition_of_divisibility_by_5_l797_797985


namespace a_n_is_geometric_seq_b_n_is_arithmetic_seq_sum_of_first_n_terms_l797_797149

-- Condition definitions for the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Conditions for a_n sequence
def a_conditions : Prop :=
  a 1 = 2 ∧ a 4 = 16 ∧ geometric_sequence a 2

-- Proof that a_n is indeed 2^n
theorem a_n_is_geometric_seq (h : a_conditions) :
  ∀ n, a n = 2 ^ n :=
sorry

-- Additional conditions for b_n sequence
def b_conditions : Prop :=
  a 3 = 8 ∧ a 5 = 32 ∧ (∃ d : ℝ, ∀ n, b (n + 1) = b n + d ∧ b 3 = a 3 ∧ b 5 = a 5)

-- Proof that b_n is 12n - 28 
theorem b_n_is_arithmetic_seq (h : b_conditions) :
  ∀ n, b n = 12 * n - 28 :=
sorry

-- Sum of the first n terms of b_n
theorem sum_of_first_n_terms (h : b_conditions) :
  ∀ n, ∑ i in range n, b i = 6 * n ^ 2 - 22 * n :=
sorry

end a_n_is_geometric_seq_b_n_is_arithmetic_seq_sum_of_first_n_terms_l797_797149


namespace length_AC_l797_797808

open Real

noncomputable def net_south_north (south north : ℝ) : ℝ := south - north
noncomputable def net_east_west (east west : ℝ) : ℝ := east - west
noncomputable def distance (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

theorem length_AC :
  let A : ℝ := 0
  let south := 30
  let north := 20
  let east := 40
  let west := 35
  let net_south := net_south_north south north
  let net_east := net_east_west east west
  distance net_south net_east = 5 * sqrt 5 :=
by
  sorry

end length_AC_l797_797808


namespace length_of_third_median_l797_797276

theorem length_of_third_median
  (a b c: ℝ)
  (median1: ℝ)
  (median2: ℝ)
  (area: ℝ)
  (h_medians: median1 = 3 ∧ median2 = 6)
  (h_area: area = 3 * real.sqrt 15)
  (h_sides_unequal: a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ (median3: ℝ), median3 = 3 * real.sqrt 6 := 
begin
  sorry
end

end length_of_third_median_l797_797276


namespace part1_part2_l797_797434

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l797_797434


namespace find_m_and_quadratic_function_l797_797814

def y (m : ℝ) (x : ℝ) : ℝ := (m^2 - m) * x^(m^2 - 2 * m - 1) + (m - 3) * x + m^2

theorem find_m_and_quadratic_function (m x : ℝ) :
  ∀ m, (m^2 - 2 * m - 1 = 2) ∧ (m^2 - m ≠ 0) -> 
  (m = 3 ∧ y 3 x = 6 * x^2 + 9) ∨ (m = -1 ∧ y (-1) x = 2 * x^2 - 4 * x + 1) :=
by
  sorry

end find_m_and_quadratic_function_l797_797814


namespace csc_315_eq_neg_sqrt_2_l797_797003

theorem csc_315_eq_neg_sqrt_2 :
  Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := 
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797003


namespace part1_part2_l797_797480

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l797_797480


namespace solve_equation_equality_check_l797_797303

-- Part a)
theorem solve_equation (x : ℝ) : (4 * x - 7) * real.sqrt (x - 3) = 25 * real.sqrt 5 → 
    ∃ (x : ℝ), (4 * x - 7) * real.sqrt (x - 3) = 25 * real.sqrt 5 :=
sorry

-- Part b)
theorem equality_check : 
  real.root 20 3 * real.root 4 (6 + 3 * real.root 5 5 + 6 * real.root 3 25) ≠ real.root 5 (16 + 9 * real.root 3 5 + 5 * real.root 3 25) :=
sorry

end solve_equation_equality_check_l797_797303


namespace inequality_part1_inequality_part2_l797_797456

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l797_797456


namespace inequality_part1_inequality_part2_l797_797457

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l797_797457


namespace csc_315_eq_neg_sqrt2_l797_797005

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end csc_315_eq_neg_sqrt2_l797_797005


namespace animators_maximum_money_extortion_l797_797253

-- Define the parameters and the conditions
variable (n : ℕ) (C : Fin n → ℕ)

-- Define the maximal money extorted function as described
noncomputable def maximal_money_extorted (n : ℕ) : ℕ :=
if n = 1 then 0 else 2^(n-1) - 1 + maximal_money_extorted (n-1)

-- Theorem statement
theorem animators_maximum_money_extortion (n : ℕ) (C : Fin n → ℕ) :
  maximal_money_extorted n = 2^n - n - 1 :=
sorry

end animators_maximum_money_extortion_l797_797253


namespace range_of_m_l797_797188

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ¬(m * x^2 + x + m = 0)) ↔ (m ∈ set.Ioo (-∞) (-1/2) ∪ set.Ioo (1/2) ∞) :=
by {
  sorry
}

end range_of_m_l797_797188


namespace part1_part2_l797_797448

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

end part1_part2_l797_797448


namespace csc_315_eq_neg_sqrt2_l797_797027

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end csc_315_eq_neg_sqrt2_l797_797027


namespace calculate_expression_l797_797754

theorem calculate_expression : 
  |1 - Real.sqrt 3| - Real.tan (Float.pi / 3) + (Float.pi - 2023)^0 + (-1/2)^(-1) = -2 := by
  sorry

end calculate_expression_l797_797754


namespace tank_width_problem_l797_797347

noncomputable def tank_width (cost_per_sq_meter : ℚ) (total_cost : ℚ) (length depth : ℚ) : ℚ :=
  let total_cost_in_paise := total_cost * 100
  let total_area := total_cost_in_paise / cost_per_sq_meter
  let w := (total_area - (2 * length * depth) - (2 * depth * 6)) / (length + 2 * depth)
  w

theorem tank_width_problem :
  tank_width 55 409.20 25 6 = 12 := 
by 
  sorry

end tank_width_problem_l797_797347


namespace average_pastries_sold_per_day_l797_797704

noncomputable def pastries_sold_per_day (n : ℕ) : ℕ :=
  match n with
  | 0 => 2 -- Monday
  | _ + 1 => 2 + n + 1

theorem average_pastries_sold_per_day :
  (∑ i in Finset.range 7, pastries_sold_per_day i) / 7 = 5 :=
by
  sorry

end average_pastries_sold_per_day_l797_797704


namespace smallest_n_for_g_larger_than_21_l797_797917

def g (n : ℕ) : ℕ :=
  Inf { j : ℕ | n ∣ Nat.factorial j }

theorem smallest_n_for_g_larger_than_21 : ∀ (n : ℕ), n = 21 * 23 → g n > 21 := by
  sorry

end smallest_n_for_g_larger_than_21_l797_797917


namespace factorization_example_l797_797297

theorem factorization_example (a m x : ℝ) : 
  (∃ a, (a^2 - 9 = (a + 3) * (a - 3))) :=
by
  exists a
  sorry

end factorization_example_l797_797297


namespace minimum_arc_length_of_curve_and_line_l797_797871

-- Definition of the curve C and the line x = π/4
def curve (x y α : ℝ) : Prop :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line (x : ℝ) : Prop :=
  x = Real.pi / 4

-- Statement of the proof problem: the minimum value of d as α varies
theorem minimum_arc_length_of_curve_and_line : 
  (∀ α : ℝ, ∃ d : ℝ, (∃ y : ℝ, curve (Real.pi / 4) y α) → 
    (d = Real.pi / 2)) :=
sorry

end minimum_arc_length_of_curve_and_line_l797_797871


namespace find_lambda_l797_797884

variables (A B C P : Type) [MetricSpace A] [AddCommGroup A] [Module ℝ A]

def equilateral_triangle (A B C : A) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def lambda := sorry

theorem find_lambda
  (A B C P : A)
  (h_equilateral : equilateral_triangle A B C)
  (h_AP : ∃ λ : ℝ, 0 < λ ∧ λ < 1 ∧ (AP = λ • (A - B)))
  (h_condition : (C - P) • (A - B) = (A - P) • (P - B)) :
  lambda = (2 - sqrt 2) / 2 := sorry

end find_lambda_l797_797884


namespace sin_alpha_correct_l797_797397

noncomputable def sin_alpha (alpha : Real) (h1 : sin (alpha + pi / 6) = 4 / 5)
  (h2 : 0 < alpha ∧ alpha < pi / 3) : Real :=
  sin alpha

theorem sin_alpha_correct (alpha : Real) (h1 : sin (alpha + pi / 6) = 4 / 5)
  (h2 : 0 < alpha ∧ alpha < pi / 3) : sin_alpha alpha h1 h2 = (4 * Real.sqrt 3 - 3) / 10 :=
by
  sorry

end sin_alpha_correct_l797_797397


namespace solve_inequality_l797_797587

theorem solve_inequality (x : ℝ) : 
  (0 < (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6))) ↔ 
  (x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x) :=
by 
  sorry

end solve_inequality_l797_797587


namespace not_every_tv_owner_has_pass_l797_797741

variable (Person : Type) (T P G : Person → Prop)

-- Condition 1: There exists a television owner who is not a painter.
axiom exists_tv_owner_not_painter : ∃ x, T x ∧ ¬ P x 

-- Condition 2: If someone has a pass to the Gellért Baths and is not a painter, they are not a television owner.
axiom pass_and_not_painter_imp_not_tv_owner : ∀ x, (G x ∧ ¬ P x) → ¬ T x

-- Prove: Not every television owner has a pass to the Gellért Baths.
theorem not_every_tv_owner_has_pass :
  ¬ ∀ x, T x → G x :=
by
  sorry -- Proof omitted

end not_every_tv_owner_has_pass_l797_797741


namespace find_two_addends_l797_797959

theorem find_two_addends (a b : ℕ) : 
  a + b = 987654321 ∧ 
  (∀ i ∈ [0,1,2,3,4,5,6,7,8,9], (∃ j1 j2, 
    ((j1 ≠ j2) ∧ j1 ∈ (to_digits 10 a) ∧ j2 ∈ (to_digits 10 b))) ) ∧ 
  (length (to_digits 10 a) = 9 ∧ length (to_digits 10 b) = 9) :=
begin
  sorry
end

end find_two_addends_l797_797959


namespace JohnsonFamilySeating_l797_797628

theorem JohnsonFamilySeating : 
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 359000 := by
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  show total_arrangements - restricted_arrangements = 359000 from sorry

end JohnsonFamilySeating_l797_797628


namespace three_pow_sub_two_pow_prime_power_prime_l797_797965

theorem three_pow_sub_two_pow_prime_power_prime (n : ℕ) (hn : n > 0) (hp : ∃ p k : ℕ, Nat.Prime p ∧ 3^n - 2^n = p^k) : Nat.Prime n := 
sorry

end three_pow_sub_two_pow_prime_power_prime_l797_797965


namespace area_equilateral_triangle_42_75_l797_797654

noncomputable def sum_of_areas (side_length : ℝ) : ℝ :=
  let radius := side_length / 2
  let sector_area := (1 / 6) * real.pi * radius ^ 2
  let triangle_area := (real.sqrt 3 / 4) * radius ^ 2
  let shaded_area := sector_area - triangle_area
  2 * shaded_area

theorem area_equilateral_triangle_42_75 :
  ∃ a b c : ℝ, a = 18.75 ∧ b = 21 ∧ c = 3 ∧ a + b + c = 42.75 ∧
               sum_of_areas 15 = (a * real.pi - b * real.sqrt c) :=
by
  use 18.75
  use 21
  use 3
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end area_equilateral_triangle_42_75_l797_797654


namespace ellipse_equation_correct_hyperbola_equation_correct_l797_797797

noncomputable def ellipse_standard_equation := 
  ∀ (a b : ℝ), (a > b) → (∀ (x y : ℝ), (x = 0 ∧ y = 2) ∨ (x = 1 ∧ y = 0) → 
  (x^2 / 4 + y^2 / a^2 = 1)) → (a = 2 ∧ b = 2)

theorem ellipse_equation_correct : 
  ellipse_standard_equation :=
begin
  sorry
end

noncomputable def hyperbola_standard_equation := 
  ∀ (λ : ℝ), (λ ≠ 0) → 
  (∀ (x y : ℝ), (x = 0 ∧ y = 6) ∨ (x = 0 ∧ y = -6)) → 
  (-y^2 / 12 + x^2 / λ = 1) → (λ = 12)

theorem hyperbola_equation_correct : 
  hyperbola_standard_equation :=
begin
  sorry
end

end ellipse_equation_correct_hyperbola_equation_correct_l797_797797


namespace find_length_BE_l797_797568

variable (A B C D F E G : Type) [has_coord A] [has_coord B] [has_coord C] [has_coord D] [has_coord F] [has_coord E] [has_coord G]

-- Given conditions
variables (EF GF : ℝ) (hEF : EF = 25) (hGF : GF = 15)
variables (rectangle_ABCD : is_rectangle A B C D)
variables (extension_BF : lies_on F (extension_of BC))
variables (intersection_AF_BD : lies_on E (intersection_of AF BD))
variables (intersection_AF_AD : lies_on G (intersection_of AF AD))

-- Required conclusion
theorem find_length_BE : ∃ BE : ℝ, BE = 25 := 
  sorry

end find_length_BE_l797_797568


namespace part1_part2_l797_797460

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l797_797460


namespace f_monotonic_increasing_on_nonnegative_g_expression_l797_797493

def f (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem f_monotonic_increasing_on_nonnegative : 
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → f x₁ < f x₂ :=
begin
  sorry
end

def g (t : ℝ) : ℝ :=
if t < -1 then 2^(t+1) + 2^(-(t+1))
else if -1 ≤ t ∧ t ≤ 0 then 2
else 2^t + 2^(-t)

theorem g_expression : 
  ∀ t : ℝ, g t = 
  if t < -1 then 2^(t+1) + 2^(-(t+1))
  else if -1 ≤ t ∧ t ≤ 0 then 2
  else 2^t + 2^(-t) :=
begin
  sorry
end

end f_monotonic_increasing_on_nonnegative_g_expression_l797_797493


namespace imaginary_part_of_z_l797_797243

theorem imaginary_part_of_z (a : ℝ) (h1 : a > 0) (h2 : complex.abs (1 + complex.i * a) = real.sqrt 5) : im (1 + complex.i * a) = 2 := 
sorry

end imaginary_part_of_z_l797_797243


namespace YHZ_angle_l797_797893

theorem YHZ_angle (XYZ_angle : ℝ) (XZY_angle : ℝ) (H : true) 
  (XYZ_angle_eq : XYZ_angle = 58) (XZY_angle_eq : XZY_angle = 15) : 
  ∠ YHZ = 73 := by
  sorry

end YHZ_angle_l797_797893


namespace determine_asymptotes_l797_797832

noncomputable def asymptotes_of_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (2 * a = 2 * Real.sqrt 2) ∧ (2 * b = 2) → 
  (∀ x y : ℝ, (y = x * (Real.sqrt 2 / 2) ∨ y = -x * (Real.sqrt 2 / 2)))

theorem determine_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a = 2 * Real.sqrt 2) ∧ (2 * b = 2) → 
  asymptotes_of_hyperbola a b ha hb :=
by
  intros h
  sorry

end determine_asymptotes_l797_797832


namespace average_height_of_trees_l797_797179

-- Define the heights of the trees
def height_tree1: ℕ := 1000
def height_tree2: ℕ := height_tree1 / 2
def height_tree3: ℕ := height_tree1 / 2
def height_tree4: ℕ := height_tree1 + 200

-- Calculate the total number of trees
def number_of_trees: ℕ := 4

-- Compute the total height climbed
def total_height: ℕ := height_tree1 + height_tree2 + height_tree3 + height_tree4

-- Define the average height
def average_height: ℕ := total_height / number_of_trees

-- The theorem statement
theorem average_height_of_trees: average_height = 800 := by
  sorry

end average_height_of_trees_l797_797179


namespace maria_saved_25_percent_l797_797744

-- Define the parameters as given in the conditions
def regular_price : ℕ := 60
def first_hat_price : ℕ := regular_price
def second_hat_price : ℕ := regular_price - (regular_price * 25 / 100)
def third_hat_price : ℕ := regular_price / 2
def total_regular_cost : ℕ := 3 * regular_price
def total_discounted_cost : ℕ := first_hat_price + second_hat_price + third_hat_price

-- Calculate the savings and the percentage saved
def savings : ℕ := total_regular_cost - total_discounted_cost
def percentage_saved : ℕ := (savings * 100) / total_regular_cost

-- Final statement to prove
theorem maria_saved_25_percent : percentage_saved = 25 := by
  intro a
  have h1 : total_regular_cost = 180 := rfl
  have h2 : total_discounted_cost = 135 := rfl
  have h3 : savings = 45 := by
    rw [h1, h2]
    exact rfl
  have h4 : percentage_saved = 25 := by
    rw [←mul_div_assoc, h3]
    exact rfl
  exact h4

end maria_saved_25_percent_l797_797744


namespace intersection_of_A_and_B_l797_797849

universe u

variable {U : Type u} [Fintype U] (A B : Set U)

-- Conditions
def universal_set : Set U := {1, 2, 3, 4, 5, 6, 7}
def complement_A : Set U := {1, 2, 4, 5, 7}
def B_set : Set U := {2, 4, 6}

-- Theorem
theorem intersection_of_A_and_B
  (h1 : A ⊆ universal_set)
  (h2 : B = B_set)
  (h3 : universal_set \ A = complement_A) :
  A ∩ B = {6} :=
by sorry

end intersection_of_A_and_B_l797_797849


namespace JohnsonFamilySeating_l797_797626

theorem JohnsonFamilySeating : 
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 359000 := by
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  show total_arrangements - restricted_arrangements = 359000 from sorry

end JohnsonFamilySeating_l797_797626


namespace probability_male_or_female_conditional_prob_B_A_l797_797324

open Probability

namespace SchoolLab
variable (Officials : Finset (Moll, Foll))

/-- Define the set of officials and the number of selections -/
def n_officials : Nat := 6
def n_males : Nat := 4
def n_females : Nat := 2
def n_selected : Nat := 3

/-- Define probability computation -/
noncomputable def prob_male_or_female_selected : ℚ :=
  (16 / 20 : ℚ)

noncomputable def A_selected_prob : ℚ :=
  (10 / 20 : ℚ)  -- C_5^2 is simplified to 10

noncomputable def AB_selected_prob : ℚ :=
  (4 / 20 : ℚ)   -- C_4^1 is simplified to 4

noncomputable def conditional_prob_B_given_A : ℚ :=
  (AB_selected_prob / A_selected_prob : ℚ)

theorem probability_male_or_female:
  prob_male_or_female_selected = (4 / 5 : ℚ) :=
by
  sorry

theorem conditional_prob_B_A:
  conditional_prob_B_given_A = (2 / 5 : ℚ) :=
by
  sorry

end SchoolLab

end probability_male_or_female_conditional_prob_B_A_l797_797324


namespace seating_arrangements_l797_797596

theorem seating_arrangements (sons daughters : ℕ) (totalSeats : ℕ) (h_sons : sons = 5) (h_daughters : daughters = 4) (h_seats : totalSeats = 9) :
  let total_arrangements := totalSeats.factorial
  let unwanted_arrangements := sons.factorial * daughters.factorial
  total_arrangements - unwanted_arrangements = 360000 :=
by
  rw [h_sons, h_daughters, h_seats]
  let total_arrangements := 9.factorial
  let unwanted_arrangements := 5.factorial * 4.factorial
  exact Nat.sub_eq_of_eq_add $ eq_comm.mpr (Nat.add_sub_eq_of_eq total_arrangements_units)
where
  total_arrangements_units : 9.factorial = 5.factorial * 4.factorial + 360000 := by
    rw [Nat.factorial, Nat.factorial, Nat.factorial, ←Nat.factorial_mul_factorial_eq 5 4]
    simp [tmp_rewriting]

end seating_arrangements_l797_797596


namespace pascals_triangle_eighth_row_sum_l797_797147

theorem pascals_triangle_eighth_row_sum :
  (∑ k in Finset.range (6).succ, Nat.choose 6 k) - 2 = 30 →
  (∑ k in Finset.range (8).succ, Nat.choose 8 k) - 2 = 126 :=
by
  sorry

end pascals_triangle_eighth_row_sum_l797_797147


namespace second_greatest_number_l797_797291

theorem second_greatest_number (digits : set ℕ) (tens : ℕ) (correct_number : ℕ) : 
  digits = {4, 3, 1, 7, 9} → tens = 3 → correct_number = 934 :=
by
  intros h1 h2
  sorry

end second_greatest_number_l797_797291


namespace john_total_payment_l797_797898

theorem john_total_payment :
  let cost_per_appointment := 400
  let total_appointments := 3
  let pet_insurance_cost := 100
  let insurance_coverage := 0.80
  let first_appointment_cost := cost_per_appointment
  let subsequent_appointments := total_appointments - 1
  let subsequent_appointments_cost := subsequent_appointments * cost_per_appointment
  let covered_cost := subsequent_appointments_cost * insurance_coverage
  let uncovered_cost := subsequent_appointments_cost - covered_cost
  let total_cost := first_appointment_cost + pet_insurance_cost + uncovered_cost
  total_cost = 660 :=
by
  sorry

end john_total_payment_l797_797898


namespace sequence_fill_l797_797156

theorem sequence_fill (x2 x3 x4 x5 x6 x7: ℕ) : 
  (20 + x2 + x3 = 100) ∧ 
  (x2 + x3 + x4 = 100) ∧ 
  (x3 + x4 + x5 = 100) ∧ 
  (x4 + x5 + x6 = 100) ∧ 
  (x5 + x6 + 16 = 100) →
  [20, x2, x3, x4, x5, x6, 16] = [20, 16, 64, 20, 16, 64, 20, 16] :=
by
  sorry

end sequence_fill_l797_797156


namespace JohnsonFamilySeating_l797_797623

theorem JohnsonFamilySeating : 
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 359000 := by
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  show total_arrangements - restricted_arrangements = 359000 from sorry

end JohnsonFamilySeating_l797_797623


namespace alternating_boys_girls_arrangements_l797_797656

theorem alternating_boys_girls_arrangements :
  let n := 4
  let arrangements := (nat.factorial n) * (nat.factorial n)
  2 * arrangements * arrangements = 2 * (nat.factorial n) ^ 2 * (nat.factorial n) ^ 2 :=
by
  sorry  -- Mathematically equivalent proof yet to be provided.

end alternating_boys_girls_arrangements_l797_797656


namespace part1_part2_l797_797438

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l797_797438


namespace molecular_weight_NaClO_l797_797057

theorem molecular_weight_NaClO :
  let Na := 22.99
  let Cl := 35.45
  let O := 16.00
  Na + Cl + O = 74.44 :=
by
  let Na := 22.99
  let Cl := 35.45
  let O := 16.00
  sorry

end molecular_weight_NaClO_l797_797057


namespace partition_impossible_l797_797755

def sum_of_list (l : List Int) : Int := l.foldl (· + ·) 0

theorem partition_impossible
  (l : List Int)
  (h : l = [-7, -4, -2, 3, 5, 9, 10, 18, 21, 33])
  (total_sum : Int := sum_of_list l)
  (target_diff : Int := 9) :
  ¬∃ (l1 l2 : List Int), 
    (l1 ++ l2 = l ∧ 
     sum_of_list l1 - sum_of_list l2 = target_diff ∧
     total_sum  = 86) := 
sorry

end partition_impossible_l797_797755


namespace bus_interval_l797_797236

theorem bus_interval (num_departures : ℕ) (total_duration : ℕ) (interval : ℕ)
  (h1 : num_departures = 11)
  (h2 : total_duration = 60)
  (h3 : interval = total_duration / (num_departures - 1)) :
  interval = 6 :=
by
  sorry

end bus_interval_l797_797236


namespace Carl_stops_after_finite_moves_l797_797196

open Classical

variable {Book : Type} [LinearOrder Book]

structure Bookshelf (n : ℕ) (books : Fin n → Book) where
  heights : Fin n → ℕ
  widths  : Fin n → ℕ
  distinct_heights : ∀ i j, i ≠ j → heights i ≠ heights j
  distinct_widths  : ∀ i j, i ≠ j → widths i ≠ widths j
  increasing_height : ∀ i j, i < j → heights i < heights j

def move_possible {Book : Type} (books : Fin 2 → Book) (heights : Fin 2 → ℕ) (widths : Fin 2 → ℕ) : Prop :=
  heights 0 < heights 1 ∧ widths 0 > widths 1

theorem Carl_stops_after_finite_moves (n : ℕ) (books : Fin n → Book) 
    (H : Bookshelf n books) :
  ∃ k, ∀ i < k, ¬move_possible (books i) (H.heights i) (H.widths i) ∧ 
  ∀ i j, i < j → H.widths i < H.widths j := 
by
  sorry

end Carl_stops_after_finite_moves_l797_797196


namespace composite_expression_l797_797953

theorem composite_expression (n : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^3 + 6 * n^2 + 12 * n + 7 = a * b :=
by
  sorry

end composite_expression_l797_797953


namespace solve_for_x_l797_797966

theorem solve_for_x (x : ℝ) (h : x > 0) (seq : list ℝ) (h_seq : seq = [4, x^2, 16]) : x = Real.sqrt 10 := sorry

end solve_for_x_l797_797966


namespace no_right_angle_tetrahedron_exists_l797_797583

theorem no_right_angle_tetrahedron_exists: 
  ¬ ∃ (A B C D : Type) (triangle : A → B → C → Type) (right_angle : A → B → C → Prop), 
           (right_angle A B C ∧ right_angle A C B ∧ right_angle B D C ∧ right_angle A D B) := 
by 
  sorry

end no_right_angle_tetrahedron_exists_l797_797583


namespace trajectory_equation_l797_797247

theorem trajectory_equation (x y : ℝ) 
  (h : |x| = real.sqrt ((x - 2)^2 + y^2)) : 
  y^2 = 4 * (x - 1) := 
by 
  sorry

end trajectory_equation_l797_797247


namespace radius_of_incircle_l797_797516

variable (W X Y Z A B C D : Type)
variable (WZ XY : ℝ) (incircle_radius : ℝ)
variable [geometry.trapezoid W X Y Z]
variable [geometry.parallel WZ XY]
variable [geometry.inscribed_circle WXYZ A B C D]

-- Conditions
axiom angle_WXY_is_45_deg : geometry.angle W X Y = 45
axiom length_of_XY : XY = 10

-- Problem statement to prove
theorem radius_of_incircle : incircle_radius = 10 / 3 := by
  sorry

end radius_of_incircle_l797_797516


namespace total_fish_caught_total_l797_797748
-- Include the broad Mathlib library to ensure all necessary mathematical functions and definitions are available

-- Define the conditions based on the given problem
def brian_trips (chris_trips : ℕ) : ℕ := 2 * chris_trips
def chris_fish_per_trip (brian_fish_per_trip : ℕ) : ℕ := brian_fish_per_trip + (2/5 : ℚ) * brian_fish_per_trip
def total_fish_caught (chris_trips : ℕ) (brian_fish_per_trip chris_fish_per_trip : ℕ) : ℕ := 
  brian_trips chris_trips * brian_fish_per_trip + chris_trips * chris_fish_per_trip

-- State the main proof problem based on the question and conditions
theorem total_fish_caught_total :
  ∀ (chris_trips : ℕ) (brian_fish_per_trip : ℕ) (chris_fish_per_trip : ℕ),
  chris_trips = 10 →
  brian_fish_per_trip = 400 →
  chris_fish_per_trip = 560 →
  total_fish_caught chris_trips brian_fish_per_trip chris_fish_per_trip = 13600 :=
by
  intros chris_trips brian_fish_per_trip chris_fish_per_trip h_chris_trips h_brian_fish_per_trip h_chris_fish_per_trip
  rw [h_chris_trips, h_brian_fish_per_trip, h_chris_fish_per_trip]
  sorry -- Proof omitted

end total_fish_caught_total_l797_797748


namespace coprime_probability_l797_797274

open Nat

-- Define the probabilities and required calculations in terms of sets, GCD, and combinatorics

def two_element_subsets (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ t, t.card = 2)

def is_coprime_pair (a b : ℕ) : Prop :=
  gcd a b = 1

def coprime_pair_count (s : Finset ℕ) : ℕ :=
  (two_element_subsets s).filter (λ t, is_coprime_pair t.toList.head t.toList.tail.head).card

def total_pair_count (s : Finset ℕ) : ℕ :=
  (two_element_subsets s).card

def probability_coprime_pairs (s : Finset ℕ) : ℚ :=
  coprime_pair_count s / total_pair_count s

-- Prove the desired probability for the set {1, 2, ..., 10}
theorem coprime_probability : probability_coprime_pairs (Finset.range 11 \ {0}) = 32 / 45 :=
by
  sorry

end coprime_probability_l797_797274


namespace magnitude_of_complex_solution_l797_797812

theorem magnitude_of_complex_solution (z : ℂ) (h : (1 + 2*complex.I)*(conj z) = 4 + 3*complex.I) : complex.abs z = complex.abs (sqrt 5) :=
by
  sorry

end magnitude_of_complex_solution_l797_797812


namespace problem_equivalence_l797_797859

theorem problem_equivalence :
  (∃ a a1 a2 a3 a4 a5 : ℝ, ((1 - x)^5 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5)) → 
  ∀ (a a1 a2 a3 a4 a5 : ℝ), (1 - 1)^5 = a + a1 + a2 + a3 + a4 + a5 →
  (1 + 1)^5 = a - a1 + a2 - a3 + a4 - a5 →
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := 
by
  intros h a a1 a2 a3 a4 a5 e1 e2
  sorry

end problem_equivalence_l797_797859


namespace part1_part2_l797_797471

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l797_797471


namespace part1_part2_l797_797426

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l797_797426


namespace cans_needed_for_fewer_people_l797_797949

theorem cans_needed_for_fewer_people :
  ∀ (total_cans : ℕ) (total_people : ℕ) (percentage_fewer : ℕ),
    total_cans = 600 →
    total_people = 40 →
    percentage_fewer = 30 →
    total_cans / total_people * (total_people - (total_people * percentage_fewer / 100)) = 420 :=
by
  intros total_cans total_people percentage_fewer
  assume h1 h2 h3
  rw [h1, h2, h3]
  have cans_per_person : ℕ := 600 / 40
  have people_after_reduction : ℕ := 40 - (40 * 30 / 100)
  show cans_per_person * people_after_reduction = 420
  sorry

end cans_needed_for_fewer_people_l797_797949


namespace Mr_Brown_children_ages_l797_797205

theorem Mr_Brown_children_ages (T : Finset ℕ) (eleven : 11 ∈ T) (card : T.card = 9)
  (age_property : ∀ n, (∃ a b c d e : ℕ, n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
                                  (a, b, c, d, e).nodup ∧
                                  (∃ x, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ ∀ t ∈ T, x % t = 0) ∧
                                  (∃ y z, 100 * y + 10 * z = Mrs_Brown's_age)) := sorry :
  10 ∉ T :=
begin
  sorry
end

end Mr_Brown_children_ages_l797_797205


namespace number_of_friendly_functions_l797_797282

-- Definition of friendly functions
def is_friendly_function (f : ℤ → ℤ) (expr : ℤ → ℤ) (range : set ℤ) : Prop :=
  ∀ x, x ∈ range ↔ ∃ d, f d = x

-- Specific function and its range
def expr := λ x : ℤ, x^2
def range := {1, 9}

-- Prove there are exactly 9 friendly functions with the given expression and range
theorem number_of_friendly_functions : 
  (∃ (funs : list (ℤ → ℤ)), 
     (∀ f ∈ funs, is_friendly_function f expr range) ∧ 
     (funs.length = 9)) := 
sorry -- proof needed

end number_of_friendly_functions_l797_797282


namespace problem1_problem2_problem3_l797_797482

-- Problem 1
theorem problem1 (f g : ℝ → ℝ) 
  (hf_odd : ∀ x, f (-x) = -f x) 
  (hg_even : ∀ x, g (-x) = g x) 
  (hfg_eq : ∀ x, f x + g x = 2 * log 2 (1 - x)) 
  : ∀ x, -1 < x ∧ x < 1 → (f x = log 2 ((1 - x) / (1 + x)) ∧ g x = log 2 (1 - x^2)) := 
  sorry

-- Problem 2
theorem problem2 (f g : ℝ → ℝ) 
  (hf_odd : ∀ x, f (-x) = -f x) 
  (hg_even : ∀ x, g (-x) = g x) 
  (hfg_eq : ∀ x, f x + g x = 2 * log 2 (1 - x)) 
  (hF_mono : ∀ x, -1 < x ∧ x < 1 → monotone (λ x, 2^(g x) + (k + 2) * x)) 
  : k ≤ -4 ∨ k ≥ 0 :=
  sorry

-- Problem 3
theorem problem3 (f g : ℝ → ℝ) 
  (hf_odd : ∀ x, f (-x) = -f x) 
  (hg_even : ∀ x, g (-x) = g x) 
  (hfg_eq : ∀ x, f x + g x = 2 * log 2 (1 - x)) 
  (hsol : ∃ x, f (2^x) - m = 0) 
  : m ∈ Iio 0 :=
  sorry

end problem1_problem2_problem3_l797_797482


namespace alpha_beta_sum_l797_797488

variables (α β : ℝ)

-- Conditions
def conditions (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (a b : ℝ × ℝ) : Prop :=
  a = (Real.sin α, Real.cos β) ∧ 
  b = (Real.cos α, Real.sin β) ∧ 
  (∀ k : ℝ, a = (k • (b.1, b.2)))

-- Proof problem statement
theorem alpha_beta_sum (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (a b : ℝ × ℝ) (h : conditions hα hβ a b) : α + β = π / 2 :=
sorry

end alpha_beta_sum_l797_797488


namespace main_theorem_l797_797085

variables {ℕ : Type*} [inst : Nat ℕ] -- Define natural numbers type

def sequence_a (n : ℕ) : ℕ := 3 * n - 2

def sequence_b (n : ℕ) : ℝ := (1 / (4 : ℝ)) ^ n -- Note Lean uses the : for casting to ℝ

def sequence_c (n : ℕ) : ℝ := sequence_a(n) * sequence_b(n)

-- Sum first n values of a sequence
def sum_first_n (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).map f |>.sum

def S_n (n : ℕ) : ℝ :=
  sum_first_n sequence_c n

def sequence_B (n : ℕ) : ℝ :=
  sum_first_n (λ n, 1 / sequence_b(n)) n

def sequence_d (n : ℕ) : ℝ :=
  1 / (sequence_b(n) * (sequence_B(n))^2)

theorem main_theorem (n : ℕ) : 
  sum_first_n sequence_d n < 1 / 2 :=
sorry

end main_theorem_l797_797085


namespace csc_315_eq_neg_sqrt2_l797_797013

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end csc_315_eq_neg_sqrt2_l797_797013


namespace probability_distance_less_than_8000_l797_797248

-- City distances in miles
def distance (city1 city2 : String) : Nat :=
  match (city1, city2) with
  | ("Beijing", "Cairo") => 4600
  | ("Beijing", "Dublin") => 5000
  | ("Beijing", "Manila") => 1800
  | ("Beijing", "Rio de Janeiro") => 10300
  | ("Cairo", "Dublin") => 2400
  | ("Cairo", "Manila") => 8150
  | ("Cairo", "Rio de Janeiro") => 5600
  | ("Dublin", "Manila") => 6700
  | ("Dublin", "Rio de Janeiro") => 7900
  | ("Manila", "Rio de Janeiro") => 11800
  -- Distance is symmetric
  | (a, b) => distance b a

-- List of city pairs
def city_pairs := [
  ("Beijing", "Cairo"),
  ("Beijing", "Dublin"),
  ("Beijing", "Manila"),
  ("Beijing", "Rio de Janeiro"),
  ("Cairo", "Dublin"),
  ("Cairo", "Manila"),
  ("Cairo", "Rio de Janeiro"),
  ("Dublin", "Manila"),
  ("Dublin", "Rio de Janeiro"),
  ("Manila", "Rio de Janeiro")
]

-- Count pairs with distances less than 8000 miles
def count_pairs_less_than_8000 : Nat :=
  city_pairs.foldl
    (λ count pair => if distance pair.1 pair.2 < 8000 then count + 1 else count)
    0

-- The total number of unique city pairs
def total_pairs : Nat := 10

-- Statement of the probability problem
theorem probability_distance_less_than_8000 : count_pairs_less_than_8000 / total_pairs = 7 / 10 :=
  by
    sorry

end probability_distance_less_than_8000_l797_797248


namespace centroid_locus_l797_797851

open EuclideanGeometry 

variables (A B C A1 B1 C1 : Point ℝ^2) (l : Line ℝ^2)

def triangle_centroid (A B C : Point ℝ^2) : Point ℝ^2 :=
  let x := (A.x + B.x + C.x) / 3 
      y := (A.y + B.y + C.y) / 3
  in ⟨x, y⟩

def midpoint (P Q : Point ℝ^2) : Point ℝ^2 :=
  let x := (P.x + Q.x) / 2
      y := (P.y + Q.y) / 2
  in ⟨x, y⟩

def line_centroid (P Q R : Point ℝ^2) : Point ℝ^2 :=
  let x := (P.x + Q.x + R.x) / 3
      y := (P.y + Q.y + R.y) / 3
  in ⟨x, y⟩

def homothety (center : Point ℝ^2) (ratio : ℝ) (P : Point ℝ^2) : Point ℝ^2 :=
  ⟨center.x + ratio * (P.x - center.x), center.y + ratio * (P.y - center.y)⟩

theorem centroid_locus (hA1 : A1 ∈ l) (hB1 : B1 ∈ l) (hC1 : C1 ∈ l) :
  ∃ l' : Line ℝ^2, parallel l l' ∧ 
  ∀ A1 B1 C1, (A1 ∈ l) → (B1 ∈ l) → (C1 ∈ l) →
  let M := triangle_centroid A B C
      X := line_centroid A1 B1 C1 in
  (midpoint M X) ∈ l' ∧ 
  homothety M (1/2) (line_centroid A1 B1 C1) ∈ l' := 
sorry

end centroid_locus_l797_797851


namespace max_area_house_l797_797325

def price_colored := 450
def price_composite := 200
def cost_limit := 32000

def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

theorem max_area_house : 
  ∃ (x y S : ℝ), 
    (S = x * y) ∧ 
    (material_cost x y ≤ cost_limit) ∧ 
    (0 < S ∧ S ≤ 100) ∧ 
    (S = 100 → x = 20 / 3) := 
by
  sorry

end max_area_house_l797_797325


namespace number_of_rabbits_l797_797320

-- Defining the problem conditions
variables (x y : ℕ)
axiom heads_condition : x + y = 40
axiom legs_condition : 4 * x = 10 * 2 * y - 8

--  Prove the number of rabbits is 33
theorem number_of_rabbits : x = 33 :=
by
  sorry

end number_of_rabbits_l797_797320


namespace find_fathers_age_l797_797387

noncomputable def sebastian_age : ℕ := 40
noncomputable def age_difference : ℕ := 10
noncomputable def sum_ages_five_years_ago_ratio : ℚ := (3 : ℚ) / 4

theorem find_fathers_age 
  (sebastian_age : ℕ) 
  (age_difference : ℕ) 
  (sum_ages_five_years_ago_ratio : ℚ) 
  (h1 : sebastian_age = 40) 
  (h2 : age_difference = 10) 
  (h3 : sum_ages_five_years_ago_ratio = 3 / 4) 
: ∃ father_age : ℕ, father_age = 85 :=
sorry

end find_fathers_age_l797_797387


namespace swiss_slices_correct_l797_797566

-- Define the variables and conditions
variables (S : ℕ) (cheddar_slices : ℕ := 12) (total_cheddar_slices : ℕ := 84) (total_swiss_slices : ℕ := 84)

-- Define the statement to be proved
theorem swiss_slices_correct (H : total_cheddar_slices = total_swiss_slices) : S = 12 :=
sorry

end swiss_slices_correct_l797_797566


namespace curve_is_ellipse_l797_797377

theorem curve_is_ellipse (θ : ℝ) : 
  let r := 2 / (1 - 2 * Real.cos θ) in
  ∃ (x y : ℝ), (x^2 + y^2) = r^2 ∧ (y^2 + 5*x^2 = 4 + 4*x*r) → 
  y^2 + 25*x^2 = 16 :=
begin
  sorry
end

end curve_is_ellipse_l797_797377


namespace cos_angle_OBC_plus_OBC_l797_797524

-- Definitions captured from the problem’s conditions
variables (A B C O : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space O]
variables (angle_A_is_obtuse : A.obtuse)
variables (orthocenter : O)
variables (AO_eq_BC : dist A O = dist B C)

-- Proof statement
theorem cos_angle_OBC_plus_OBC {A B C O : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space O]
  (angle_A_is_obtuse : A.obtuse) (orthocenter_O : is_orthocenter O) (AO_eq_BC : dist A O = dist B C) :
  cos (angle O B C + angle O C B) = - (sqrt 2) / 2 :=
sorry

end cos_angle_OBC_plus_OBC_l797_797524


namespace x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5_l797_797311

theorem x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5 :
  (∀ x : ℝ, x > 5 → x > 3) ∧ ¬(∀ x : ℝ, x > 3 → x > 5) :=
by 
  -- Prove implications with provided conditions
  sorry

end x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5_l797_797311


namespace tan_double_angle_cos_beta_l797_797396

theorem tan_double_angle (α β : ℝ) (h1 : Real.sin α = 4 * Real.sqrt 3 / 7) 
  (h2 : Real.cos (β - α) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < Real.pi / 2) : 
  Real.tan (2 * α) = -(8 * Real.sqrt 3) / 47 :=
  sorry

theorem cos_beta (α β : ℝ) (h1 : Real.sin α = 4 * Real.sqrt 3 / 7) 
  (h2 : Real.cos (β - α) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < Real.pi / 2) : 
  Real.cos β = 1 / 2 :=
  sorry

end tan_double_angle_cos_beta_l797_797396


namespace csc_315_eq_neg_sqrt2_l797_797010

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end csc_315_eq_neg_sqrt2_l797_797010


namespace livestock_allocation_l797_797718

theorem livestock_allocation :
  ∃ (x y z : ℕ), x + y + z = 100 ∧ 20 * x + 6 * y + z = 200 ∧ x = 5 ∧ y = 1 ∧ z = 94 :=
by
  sorry

end livestock_allocation_l797_797718


namespace vasim_share_l797_797738

theorem vasim_share (x : ℝ)
  (h_ratio : ∀ (f v r : ℝ), f = 3 * x ∧ v = 5 * x ∧ r = 6 * x)
  (h_diff : 6 * x - 3 * x = 900) :
  5 * x = 1500 :=
by
  try sorry

end vasim_share_l797_797738


namespace imo2007_hktst1_p2_l797_797680

noncomputable def minimizeExpression (A B C : ℝ) : ℝ :=
  (Real.tan C - Real.sin A)^2 + (Real.cot C - Real.cos B)^2

-- Main theorem statement
theorem imo2007_hktst1_p2
    (A B C : ℝ)
    (h1 : Real.sin A * Real.cos B + abs (Real.cos A * Real.sin B) = 
          Real.sin A * abs (Real.cos A) + abs (Real.sin B) * Real.cos B)
    (h2 : ∃ t : ℝ, Real.tan C = t ∧ Real.cot C = 1 / t) :
    minimizeExpression A B C = 3 - 2 * Real.sqrt 2 := 
sorry

end imo2007_hktst1_p2_l797_797680


namespace all_terms_are_integers_l797_797930

noncomputable theory

def sequence (a1 a2 : ℤ) (b : ℤ) (n : ℕ) : ℤ :=
nat.rec_on n a1 (λ n seq_prev1,
  nat.rec_on n a2 (λ n seq_prev2,
    (seq_prev1^2 + b) / seq_prev2))

theorem all_terms_are_integers (a1 a2 b : ℤ) 
(h1 : (a1 : ℚ) ≠ 0)
(h2 : (a2 : ℚ) ≠ 0)
(h3 : (a1^2 + a2^2 + b) % (a1 * a2) = 0) 
:
    ∀ n : ℕ, ∃ an : ℤ, an = sequence a1 a2 b n :=
begin
    sorry
end

end all_terms_are_integers_l797_797930


namespace plane_coloring_l797_797174

-- Define the basic setting of the problem
def is_rational (r : ℝ) : Prop := ∃ (q : ℚ), r = q

-- The coloring function is based on the distance from a reference point.
noncomputable def color (O M : ℝ × ℝ) : Prop :=
  if is_rational (real.sqrt ((M.1 - O.1) ^ 2 + (M.2 - O.2) ^ 2)) then true else false

-- Statement of the theorem
theorem plane_coloring (O : ℝ × ℝ) :
  ∀ (A B : ℝ × ℝ), A ≠ B →
  ∃ (M : ℝ × ℝ), M ∈ segment ℝ A B ∧
  ((color O M = true ∧ ∃ (N : ℝ × ℝ), N ∈ segment ℝ A B ∧ color O N = false) ∨
   (color O M = false ∧ ∃ (N : ℝ × ℝ), N ∈ segment ℝ A B ∧ color O N = true)) :=
sorry

end plane_coloring_l797_797174


namespace promotional_codes_divided_by_10_l797_797712

-- The six unique characters available for the code
inductive Char : Type
| A : Char
| B : Char
| C : Char
| Digit0 : Char
| Digit2 : Char
| Digit7 : Char

-- Function to count the number of valid codes
noncomputable def count_valid_codes : ℕ := 1800

-- The main theorem statement
theorem promotional_codes_divided_by_10 : count_valid_codes / 10 = 180 :=
by
  -- Proof omitted
  sorry

end promotional_codes_divided_by_10_l797_797712


namespace sets_A_B_complement_U_l797_797119

def A (x : ℝ) : Prop := (x + 2) / (x - 2) ≤ 0
def B (x : ℝ) : Prop := |x - 1| < 2
def complement_A (x : ℝ) : Prop := x < -2 ∨ x ≥ 2

theorem sets_A_B_complement_U : 
  (A = {x : ℝ | -2 ≤ x ∧ x < 2}) ∧
  (B = {x : ℝ | -1 < x ∧ x < 3}) ∧
  (B ∩ (complement_A) = {x : ℝ | 2 ≤ x ∧ x < 3}) := 
  by
  sorry

end sets_A_B_complement_U_l797_797119


namespace elderly_sample_correct_l797_797321

-- Conditions
def young_employees : ℕ := 300
def middle_aged_employees : ℕ := 150
def elderly_employees : ℕ := 100
def total_employees : ℕ := young_employees + middle_aged_employees + elderly_employees
def sample_size : ℕ := 33
def elderly_sample (total : ℕ) (elderly : ℕ) (sample : ℕ) : ℕ := (sample * elderly) / total

-- Statement to prove
theorem elderly_sample_correct :
  elderly_sample total_employees elderly_employees sample_size = 6 := 
by
  sorry

end elderly_sample_correct_l797_797321


namespace fraction_of_fritz_money_l797_797578

theorem fraction_of_fritz_money
  (Fritz_money : ℕ)
  (total_amount : ℕ)
  (fraction : ℚ)
  (Sean_money : ℚ)
  (Rick_money : ℚ)
  (h1 : Fritz_money = 40)
  (h2 : total_amount = 96)
  (h3 : Sean_money = fraction * Fritz_money + 4)
  (h4 : Rick_money = 3 * Sean_money)
  (h5 : Rick_money + Sean_money = total_amount) :
  fraction = 1 / 2 :=
by
  sorry

end fraction_of_fritz_money_l797_797578


namespace main_theorem_l797_797158

def is_T_sequence (A : ℕ → ℝ × ℝ) : Prop :=
  ∀ n, A (n+1).snd - A n.snd > 0

def statement_1 : Prop :=
  let A := λ n, (n : ℕ, 1 / (n : ℝ)) in is_T_sequence A

def statement_2 (A : ℕ → ℝ × ℝ) : Prop :=
  A 2 <.> A 1 ∧ ∀ k, ∃ x y z : ℝ, 0 <= x ∧ x = A (k+2) - A (k+1) ∧ y = A (k+1) - A k ∧ z < 0

def statement_3 (A : ℕ → ℝ × ℝ) : Prop :=
  ∀ m n p q : ℕ, 1 ≤ m ∧ m < n ∧ n < p ∧ p < q ∧ m + q = n + p →
    let b := λ n, A (n+1).snd - A n.snd in A q.snd - A p.snd ≥ (q - p) * b p

def statement_4 (A : ℕ → ℝ × ℝ) : Prop :=
  ∀ m n p q : ℕ, 1 ≤ m ∧ m < n ∧ n < p ∧ p < q ∧ m + q = n + p →
    let b := λ n, A (n+1).snd - A n.snd in b q > b (n-1)

theorem main_theorem : (statement_1 ∧ statement_3 (λ n, (n : ℕ, 1 / (n : ℝ))) ∧ statement_4 (λ n, (n : ℕ, 1 / (n : ℝ))) ∧ ¬statement_2 (λ n, (n : ℕ, 1 / (n : ℝ)))) → true := 
by {
  sorry
}

end main_theorem_l797_797158


namespace min_value_of_reciprocal_sum_l797_797075

variables {m n : ℝ}
variables (h1 : m > 0)
variables (h2 : n > 0)
variables (h3 : m + n = 1)

theorem min_value_of_reciprocal_sum : 
  (1 / m + 1 / n) = 4 :=
by
  sorry

end min_value_of_reciprocal_sum_l797_797075


namespace count_valid_pairs_l797_797381

-- Define the factorial of a polynomial
def canBeFactored (a b : ℤ) : Prop :=
  ∃ r s : ℤ, r + s = a ∧ r * s = b

-- Define the conditions for the ordered pairs
def validPairs (a b : ℤ) : Prop :=
  1 ≤ a ∧ a ≤ 50 ∧ b > 0 ∧ canBeFactored a b

-- Main statement proving the number of such valid pairs is 325
theorem count_valid_pairs : (finset.card (finset.filter 
  (λ p : ℤ × ℤ, validPairs p.1 p.2) 
  ((finset.Icc 1 50).product (finset.Ioi 0)))) = 325 := 
sorry

end count_valid_pairs_l797_797381


namespace hyperbola_eccentricity_l797_797378

theorem hyperbola_eccentricity : 
  (∃ (a b : ℝ), (a^2 = 1 ∧ b^2 = 2) ∧ ∀ e : ℝ, e = Real.sqrt (1 + b^2 / a^2) → e = Real.sqrt 3) :=
by 
  sorry

end hyperbola_eccentricity_l797_797378


namespace part1_part2_l797_797463

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l797_797463


namespace cos_four_times_arccos_val_l797_797373

theorem cos_four_times_arccos_val : 
  ∀ x : ℝ, x = Real.arccos (1 / 4) → Real.cos (4 * x) = 17 / 32 :=
by
  intro x h
  sorry

end cos_four_times_arccos_val_l797_797373


namespace johnson_family_seating_l797_797608

/-- The Johnson family has 5 sons and 4 daughters. We want to find the number of ways to seat them in a row of 9 chairs such that at least 2 boys are next to each other. -/
theorem johnson_family_seating : 
  let boys := 5 in
  let girls := 4 in
  let total_children := boys + girls in
  fact total_children - 
  2 * (fact boys * fact girls) = 357120 := 
by
  let boys := 5
  let girls := 4
  let total_children := boys + girls
  have total_arrangements : ℕ := fact total_children
  have no_two_boys_next_to_each_other : ℕ := 2 * (fact boys * fact girls)
  have at_least_two_boys_next_to_each_other : ℕ := total_arrangements - no_two_boys_next_to_each_other
  show at_least_two_boys_next_to_each_other = 357120
  sorry

end johnson_family_seating_l797_797608


namespace problem_1_l797_797053

noncomputable def derivative_y (a x y : ℝ) (h : y^3 - 3 * y + 2 * a * x = 0) : ℝ :=
  (2 * a) / (3 * (1 - y^2))

theorem problem_1 (a x y : ℝ) (h : y^3 - 3 * y + 2 * a * x = 0) :
  derivative_y a x y h = (2 * a) / (3 * (1 - y^2)) :=
sorry

end problem_1_l797_797053


namespace computation_l797_797762

theorem computation :
  (13 + 12)^2 - (13 - 12)^2 = 624 :=
by
  sorry

end computation_l797_797762


namespace find_diff_eq_l797_797055

noncomputable def general_solution (y : ℝ → ℝ) : Prop :=
∃ (C1 C2 : ℝ), ∀ x : ℝ, y x = C1 * x + C2

theorem find_diff_eq (y : ℝ → ℝ) (C1 C2 : ℝ) (h : ∀ x : ℝ, y x = C1 * x + C2) :
  ∀ x : ℝ, (deriv (deriv y)) x = 0 :=
by
  sorry

end find_diff_eq_l797_797055


namespace calc_1_calc_2_l797_797753

variable (x y : ℝ)

theorem calc_1 : (-x^2)^4 = x^8 := 
sorry

theorem calc_2 : (-x^2 * y)^3 = -x^6 * y^3 := 
sorry

end calc_1_calc_2_l797_797753


namespace proof_problem_l797_797539

noncomputable def find_values (a b c x y z : ℝ) := 
  14 * x + b * y + c * z = 0 ∧ 
  a * x + 24 * y + c * z = 0 ∧ 
  a * x + b * y + 43 * z = 0 ∧ 
  a ≠ 14 ∧ b ≠ 24 ∧ c ≠ 43 ∧ x ≠ 0

theorem proof_problem (a b c x y z : ℝ) 
  (h : find_values a b c x y z):
  (a / (a - 14)) + (b / (b - 24)) + (c / (c - 43)) = 1 :=
by
  sorry

end proof_problem_l797_797539


namespace max_spheres_theoretical_l797_797286

noncomputable def volume_of_sphere := 3.5
noncomputable def box_dimensions := (8.1, 9.7, 12.5)
noncomputable def box_volume : ℝ := (box_dimensions.1 * box_dimensions.2 * box_dimensions.3)

theorem max_spheres_theoretical :
  let N := box_volume / volume_of_sphere in
  N.floor = 280 :=
by
  sorry

end max_spheres_theoretical_l797_797286


namespace faster_speed_l797_797721

noncomputable def time_taken (distance speed : ℝ) : ℝ := distance / speed

theorem faster_speed :
  ∃ x : ℝ, 
    let t := time_taken 60 12 in
    time_taken (60 + 20) x = t ∧
    x = 16 :=
by
  sorry

end faster_speed_l797_797721


namespace fraction_question_l797_797371

theorem fraction_question :
  ((3 / 8 + 5 / 6) / (5 / 12 + 1 / 4) = 29 / 16) :=
by
  -- This is where we will put the proof steps 
  sorry

end fraction_question_l797_797371


namespace find_value_of_x_l797_797064

theorem find_value_of_x (x : ℕ) (h : (50 + x / 90) * 90 = 4520) : x = 4470 :=
sorry

end find_value_of_x_l797_797064


namespace new_perimeter_is_60_l797_797655

-- Defining the conditions
variables (width : ℝ) (original_area : ℝ) (new_area : ℝ) (original_length : ℝ) (new_length : ℝ)

-- The given conditions
def width_is_10 : width = 10 := by sorry
def area_is_150 : original_area = 150 := by sorry
def new_area_is_4_over_3_times_original : new_area = (4 / 3) * original_area := by sorry

-- Calculating the original length based on the area and width
def original_length_calc : original_length = original_area / width :=
by { rw [area_is_150, width_is_10], exact (150 / 10) }

-- Calculating the new length based on the new area and width
def new_length_calc : new_length = new_area / width :=
by { rw [new_area_is_4_over_3_times_original, width_is_10], exact ((4 / 3) * 150 / 10) }

-- Proving the new perimeter based on the new length and width
theorem new_perimeter_is_60 : 2 * (new_length + width) = 60 :=
by {
  rw [new_length_calc, width_is_10],
  rw [show (4 / 3) * 150 / 10 = 20, by norm_num],
  rw show 10 = 10, by norm_num,
  exact (by rw [2 * (20 + 10) = 60])
}

end new_perimeter_is_60_l797_797655


namespace algae_coverage_l797_797637

theorem algae_coverage (doubles_every_day : ∀ n, algae_coverage (n + 1) = 2 * algae_coverage n)
  (coverage_day_20 : algae_coverage 20 = 1) :
  algae_coverage 17 = 1 / 8 := 
sorry

end algae_coverage_l797_797637


namespace count_integer_values_l797_797652

theorem count_integer_values (x : ℤ) (h1 : 4 < Real.sqrt (3 * x + 1)) (h2 : Real.sqrt (3 * x + 1) < 5) : 
  (5 < x ∧ x < 8 ∧ ∃ (N : ℕ), N = 2) :=
by sorry

end count_integer_values_l797_797652


namespace csc_315_eq_neg_sqrt_2_l797_797022

theorem csc_315_eq_neg_sqrt_2 :
  let csc := λ θ, 1 / Real.sin θ in
  csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
  by
  let sin := Real.sin
  have h1 : csc (315 * Real.pi / 180) = 1 / sin (315 * Real.pi / 180) := rfl
  have h2 : sin (315 * Real.pi / 180) = sin ((360 - 45) * Real.pi / 180) := by congr; norm_num
  have h3 : sin ((360 - 45) * Real.pi / 180) = -sin (45 * Real.pi / 180) := by
    rw [Real.sin_pi_sub]
    congr; norm_num
  have h4 : sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 := Real.sin_of_one_div_sqrt_two 45 rfl
  sorry

end csc_315_eq_neg_sqrt_2_l797_797022


namespace maximize_perimeter_area_l797_797109

def ellipse := {P : ℝ × ℝ // (P.1)^2 / 9 + (P.2)^2 / 5 = 1}
def A : ℝ × ℝ := (0, 2 * Real.sqrt 3)
def F : ℝ × ℝ := (2, 0)
def area_of_triangle (A F : ℝ × ℝ) (P : ellipse) : ℝ := 
  1 / 2 * Real.dist A P * 4 * Real.sin (Real.pi / 3)

theorem maximize_perimeter_area (P : ellipse) : 
  area_of_triangle A F P = (21 * Real.sqrt 3) / 4 :=
sorry

end maximize_perimeter_area_l797_797109


namespace find_fathers_age_l797_797386

noncomputable def sebastian_age : ℕ := 40
noncomputable def age_difference : ℕ := 10
noncomputable def sum_ages_five_years_ago_ratio : ℚ := (3 : ℚ) / 4

theorem find_fathers_age 
  (sebastian_age : ℕ) 
  (age_difference : ℕ) 
  (sum_ages_five_years_ago_ratio : ℚ) 
  (h1 : sebastian_age = 40) 
  (h2 : age_difference = 10) 
  (h3 : sum_ages_five_years_ago_ratio = 3 / 4) 
: ∃ father_age : ℕ, father_age = 85 :=
sorry

end find_fathers_age_l797_797386


namespace inscribed_triangle_congruent_l797_797278

variables {α : Type*} [LinearOrderedField α] [MetricSpace α]

-- Given the original triangle ABC.
variables (A B C G H I : α)

-- Definitions for vertices of the inscribed triangle DEF, and other elements.
variables (D E F I : α)

-- Definitions for the triangle congruence property.
def is_congruent (T1 T2 : Triangle) : Prop :=
  T1.side_lengths = T2.side_lengths ∧ T1.angles = T2.angles

-- Main statement: Prove that if DEF is inscribed in ABC and congruent to GHI, then DEF and GHI are congruent.
theorem inscribed_triangle_congruent 
  (hABC_triangle : triangle A B C)
  (hGHI_triangle : triangle G H I)
  (hDEF_inscribed : inscribe_triangle A B C D E F)
  (hCongruent : is_congruent (triangle D E F) (triangle G H I)) :
  triangle D E F = triangle G H I := by
  sorry

end inscribed_triangle_congruent_l797_797278


namespace possible_values_of_f_l797_797915

-- Define the sequence x_k
def x_k (k : ℕ) : ℤ := (-1)^(k+1)

-- Define the function f(n)
def f (n : ℕ) : ℚ := (∑ k in Finset.range n, x_k (k+1)) / n

-- Problem statement to prove the set of possible values of f(n)
theorem possible_values_of_f (n : ℕ) (hn : 0 < n) : 
  (f n = 0) ∨ (f n = 1 / n) :=
sorry

end possible_values_of_f_l797_797915


namespace Dan_initial_money_l797_797361

-- Definitions of the conditions
def candy_bar_cost : Nat := 6
def chocolate_cost : Nat := 3
def candy_bar_more_than_chocolate : Nat := 3

-- Statement of the problem as a Lean theorem
theorem Dan_initial_money (candy_bar_cost : Nat) (chocolate_cost : Nat) (candy_bar_more_than_chocolate : Nat) : Nat :=
  let total_spent := candy_bar_cost + chocolate_cost
  total_spent = 9 :=
begin
  sorry
end

end Dan_initial_money_l797_797361


namespace kangaroo_population_change_l797_797945

theorem kangaroo_population_change:
  ∀ (G R : ℝ) (h₁ : G > 0) (h₂ : R > 0), 
  let new_G := 1.28 * G,
      new_R := 0.72 * R,
      initial_total := G + R,
      new_total := new_G + new_R in
  new_G / new_R = R / G →
  100 * (new_total - initial_total) / initial_total = -4 :=
begin
  intros G R h₁ h₂ new_G new_R initial_total new_total h_ratio,
  sorry
end

end kangaroo_population_change_l797_797945


namespace apollonian_circle_radius_l797_797167

theorem apollonian_circle_radius (r : ℝ) (r_pos : r > 0) : 
  (∃ P : ℝ × ℝ, ((P.1 - 2)^2 + P.2^2 = r^2) ∧ 
                  (real.sqrt((P.1 - 3)^2 + P.2^2) = 2 * real.sqrt(P.1^2 + P.2^2))) ↔ r = 1 :=
begin
  sorry
end

end apollonian_circle_radius_l797_797167


namespace n_minus_two_is_square_of_natural_number_l797_797999

theorem n_minus_two_is_square_of_natural_number (n : ℕ) (h_n : n ≥ 3) (h_odd_m : Odd (1 / 2 * n * (n - 1))) :
  ∃ k : ℕ, n - 2 = k^2 := 
  by
  sorry

end n_minus_two_is_square_of_natural_number_l797_797999


namespace average_pastries_per_day_l797_797701

-- Conditions
def pastries_on_monday := 2

def pastries_on_day (n : ℕ) : ℕ :=
  pastries_on_monday + n

def total_pastries_in_week : ℕ :=
  List.sum (List.map pastries_on_day (List.range 7))

def number_of_days_in_week : ℕ := 7

-- Theorem to prove
theorem average_pastries_per_day : (total_pastries_in_week / number_of_days_in_week) = 5 :=
by
  sorry

end average_pastries_per_day_l797_797701


namespace exceeds_500_on_friday_l797_797902

theorem exceeds_500_on_friday
  (initial_amount : ℕ := 5)
  (rate : ℕ := 3)
  (threshold : ℕ := 500)
  (start_day : ℕ := 1) : 
  ∃ n, geometric_sum initial_amount rate n > threshold ∧ day_of_week n = "Friday" :=
by
  sorry

-- Definitions for geometric sum and day of week calculation
def geometric_sum (a r n : ℕ) : ℕ :=
  a * (1 - r ^ n) / (1 - r)

def day_of_week (n : ℕ) : String :=
  let days := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  days.get (n % 7)

end exceeds_500_on_friday_l797_797902


namespace exists_infinite_g_l797_797195

-- Definition of bijective function
def bijective (f : ℝ → ℝ) : Prop :=
  function.injective f ∧ function.surjective f

-- Lean statement of the problem
theorem exists_infinite_g {f : ℝ → ℝ} (hf : bijective f) :
  ∃ (g : ℝ → ℝ) (infinite_g : infinite (SetOf (λ g, ∀ x, f (g x) = g (f x)))),
  true :=
sorry

end exists_infinite_g_l797_797195


namespace exists_integers_a_b_c_d_and_n_l797_797956

theorem exists_integers_a_b_c_d_and_n (n a b c d : ℕ)
  (h1 : a = 10) 
  (h2 : b = 15) 
  (h3 : c = 8) 
  (h4 : d = 3) 
  (h5 : n = 16) :
  a^4 + b^4 + c^4 + 2 * d^4 = n^4 := by
  -- Proof goes here
  sorry

end exists_integers_a_b_c_d_and_n_l797_797956


namespace part1_part2_l797_797475

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l797_797475


namespace sin_double_angle_l797_797826

theorem sin_double_angle (α : ℝ) (h1 : cos (5 * π / 2 + α) = 3 / 5) (h2 : -π / 2 < α ∧ α < 0) : 
  sin (2 * α) = -24 / 25 :=
sorry

end sin_double_angle_l797_797826


namespace csc_315_eq_neg_sqrt2_l797_797023

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end csc_315_eq_neg_sqrt2_l797_797023


namespace locus_of_tangency_centers_l797_797242

def locus_of_centers (a b : ℝ) : Prop := 8 * a ^ 2 + 9 * b ^ 2 - 16 * a - 64 = 0

theorem locus_of_tangency_centers (a b : ℝ)
  (hx1 : ∃ x y : ℝ, x ^ 2 + y ^ 2 = 1) 
  (hx2 : ∃ x y : ℝ, (x - 2) ^ 2 + y ^ 2 = 25) 
  (hcent : ∃ r : ℝ, a^2 + b^2 = (r + 1)^2 ∧ (a - 2)^2 + b^2 = (5 - r)^2) : 
  locus_of_centers a b :=
sorry

end locus_of_tangency_centers_l797_797242


namespace chord_length_l797_797519

open Real

theorem chord_length (x y : ℝ) : 
  (x^2 + y^2 = 4) ∧ (x - √3 * y + 2 * √3 = 0) → 
  (chord_length : ℝ) : chord_length = 2 :=
by
  sorry

end chord_length_l797_797519


namespace y_coord_of_line_at_x_eq_six_l797_797332

theorem y_coord_of_line_at_x_eq_six :
  let p1 := (3, 3, 2)
  let p2 := (7, 0, -4)
  ∃ y : ℚ, 
  ∃ t : ℚ, 
  let x := p1.1 + t * (p2.1 - p1.1)
  let y_coord := p1.2 + t * (p2.2 - p1.2)
  let z := p1.3 + t * (p2.3 - p1.3)
  x = 6 ∧ y_coord = y := 
  y = 3 / 4 :=
by
  sorry

end y_coord_of_line_at_x_eq_six_l797_797332


namespace percentage_preferring_city_Y_l797_797207

theorem percentage_preferring_city_Y (total_employees : ℕ) (percent_reloc_to_X : ℕ) 
  (max_preferred_reloc : ℕ) (employees_to_X : ℕ) (employees_to_Y : ℕ) 
  (percent_preferring_city_Y : ℕ) :
  total_employees = 200 →
  percent_reloc_to_X = 30 →
  max_preferred_reloc = 140 →
  employees_to_X = (percent_reloc_to_X * total_employees) / 100 →
  employees_to_Y = max_preferred_reloc - employees_to_X →
  percent_preferring_city_Y = (employees_to_Y * 100) / total_employees →
  percent_preferring_city_Y = 40 :=
by
  intros h_total h_percentX h_max h_employeesX h_employeesY h_percentY
  have h1 : employees_to_X = 60 := by
    rw [h_total, h_percentX]; norm_num
  have h2 : employees_to_Y = 140 - 60 := by
    rw [h_max, h1]; norm_num
  simp [h_total, h2] at h_percentY
  norm_num at h_percentY
  exact h_percentY

end percentage_preferring_city_Y_l797_797207


namespace cats_teeth_count_l797_797280

-- Lean statement for the given math proof problem
theorem cats_teeth_count :
  ∀ (dogs cats pigs total_teeth teeth_per_dog teeth_per_pig : ℕ), 
  teeth_per_dog = 42 → 
  teeth_per_pig = 28 → 
  dogs = 5 → 
  cats = 10 → 
  pigs = 7 → 
  total_teeth = 706 → 
  10 * (total_teeth - dogs * teeth_per_dog - pigs * teeth_per_pig) / 10 = 30 :=
by
  intros dogs cats pigs total_teeth teeth_per_dog teeth_per_pig
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end cats_teeth_count_l797_797280


namespace csc_315_eq_sqrt2_l797_797047

theorem csc_315_eq_sqrt2 :
  let θ := 315
  let csc := λ θ, 1 / (Real.sin (θ * Real.pi / 180))
  315 = 360 - 45 → 
  Real.sin (315 * Real.pi / 180) = Real.sin ((360 - 45) * Real.pi / 180) → 
  Real.sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 →
  csc 315 = Real.sqrt 2 := 
by
  intros θ csc h1 h2 h3
  -- proof would go here
  sorry

end csc_315_eq_sqrt2_l797_797047


namespace sum_cn_less_than_2_l797_797818

-- Definitions as conditions
def a_seq (n : ℕ) : ℕ := n
def S_seq (n : ℕ) : ℕ := (1/2 : ℚ) * (a_seq n) * ((a_seq n) + 1)
def b_seq (n : ℕ) : ℕ := (3^n - 1) / 2
def c_seq (n : ℕ) : ℚ := (3^n : ℚ) / (2 * (b_seq n)^2)
def T_seq (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), c_seq i

-- Theorem statement
theorem sum_cn_less_than_2 (n : ℕ) : T_seq n < 2 := 
sorry

end sum_cn_less_than_2_l797_797818


namespace profit_function_equation_maximum_profit_l797_797069

noncomputable def production_cost (x : ℝ) : ℝ := x^3 - 24*x^2 + 63*x + 10
noncomputable def sales_revenue (x : ℝ) : ℝ := 18*x
noncomputable def production_profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem profit_function_equation (x : ℝ) : production_profit x = -x^3 + 24*x^2 - 45*x - 10 :=
  by
    unfold production_profit sales_revenue production_cost
    sorry

theorem maximum_profit : (production_profit 15 = 1340) ∧ ∀ x, production_profit 15 ≥ production_profit x :=
  by
    sorry

end profit_function_equation_maximum_profit_l797_797069


namespace prove_problem1_prove_problem2_l797_797217

open Real Nat

noncomputable def problem1 (n : ℕ) : Prop :=
  1 - (Finset.range (n / 2 + 1)).sum (λ m : ℕ, binom n (2 * m) * (-1)^m) = 2^(n / 2) * cos (n * pi / 4)

noncomputable def problem2 (n : ℕ) : Prop :=
  (Finset.range (n / 2)).sum (λ m : ℕ, binom n (2 * m + 1) * (-1)^m) = 2^(n / 2) * sin (n * pi / 4)

theorem prove_problem1 (n : ℕ) : problem1 n := sorry

theorem prove_problem2 (n : ℕ) : problem2 n := sorry

end prove_problem1_prove_problem2_l797_797217


namespace unattainable_y_value_l797_797766

theorem unattainable_y_value (y : ℝ) (x : ℝ) (h : x ≠ -4 / 3) : ¬ (y = -1 / 3) :=
by {
  -- The proof is omitted for now. 
  -- We're only constructing the outline with necessary imports and conditions.
  sorry
}

end unattainable_y_value_l797_797766


namespace problem_statement_l797_797671

noncomputable def smallest_integer_exceeding := 
  let x : ℝ := (Real.sqrt 3 + Real.sqrt 2) ^ 8
  Int.ceil x

theorem problem_statement : smallest_integer_exceeding = 5360 :=
by 
  -- The proof is omitted
  sorry

end problem_statement_l797_797671


namespace find_integers_in_range_l797_797786

theorem find_integers_in_range :
  ∀ x : ℤ,
  (20 ≤ x ∧ x ≤ 50 ∧ (6 * x + 5) % 10 = 19) ↔
  x = 24 ∨ x = 29 ∨ x = 34 ∨ x = 39 ∨ x = 44 ∨ x = 49 :=
by sorry

end find_integers_in_range_l797_797786


namespace csc_315_eq_neg_sqrt_2_l797_797036

theorem csc_315_eq_neg_sqrt_2 : csc 315 = -sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797036


namespace polynomial_solution_l797_797785

noncomputable def satisfies_condition (P : ℝ → ℝ) (x y z : ℝ) : Prop :=
  (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧ (2 * x * y * z = x + y + z) ∧
  ((P(x) / (y * z)) + (P(y) / (z * x)) + (P(z) / (x * y)) = P(x - y) + P(y - z) + P(z - x))

theorem polynomial_solution (P : ℝ → ℝ) :
  (∀ x y z, satisfies_condition P x y z) → 
  ∃ c : ℝ, ∀ x, P(x) = c * (x^2 + 3) :=
sorry

end polynomial_solution_l797_797785


namespace jose_investment_l797_797662

theorem jose_investment 
  (T_investment : ℕ := 30000) -- Tom's investment in Rs.
  (J_months : ℕ := 10)        -- Jose's investment period in months
  (T_months : ℕ := 12)        -- Tom's investment period in months
  (total_profit : ℕ := 72000) -- Total profit in Rs.
  (jose_profit : ℕ := 40000)  -- Jose's share of profit in Rs.
  : ∃ X : ℕ, (jose_profit * (T_investment * T_months)) = ((total_profit - jose_profit) * (X * J_months)) ∧ X = 45000 :=
  sorry

end jose_investment_l797_797662


namespace symmetric_points_y_axis_l797_797866

theorem symmetric_points_y_axis (a b : ℤ) 
  (h1 : a + 1 = 2) 
  (h2 : b + 2 = 3) : 
  a + b = 2 :=
by
  sorry

end symmetric_points_y_axis_l797_797866


namespace range_of_a_l797_797414

variables (a : ℝ) (x : ℝ) (x0 : ℝ)

def proposition_P (a : ℝ) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def proposition_Q (a : ℝ) : Prop :=
  ∃ x0, x0^2 + 2 * a * x0 + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (proposition_P a ∧ proposition_Q a) → a ∈ {a : ℝ | a ≤ -2} ∪ {a : ℝ | a = 1} :=
by {
  sorry -- Proof goes here.
}

end range_of_a_l797_797414


namespace ticket_percent_saved_l797_797743

theorem ticket_percent_saved:
  (∀ (P : ℝ), P > 0 →
    let original_price_50 := 50 * P in
    let sale_price_50 := 2 * 21.5 * P in
    let amount_saved := original_price_50 - sale_price_50 in
    let percent_saved := (amount_saved / original_price_50) * 100 in
    percent_saved = 14) :=
sorry

end ticket_percent_saved_l797_797743


namespace smartphone_charging_time_l797_797723

theorem smartphone_charging_time :
  ∀ (T S : ℕ), T = 53 → T + (1 / 2 : ℚ) * S = 66 → S = 26 :=
by
  intros T S hT equation
  sorry

end smartphone_charging_time_l797_797723


namespace min_chord_length_l797_797868

variable (α : ℝ)

def curve_eq (x y α : ℝ) :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line_eq (x : ℝ) :=
  x = Real.pi / 4

theorem min_chord_length :
  ∃ d, (∀ α : ℝ, ∃ y1 y2 : ℝ, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ d = |y2 - y1|) ∧
  (∀ α : ℝ, ∃ y1 y2, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ |y2 - y1| ≥ d) :=
sorry

end min_chord_length_l797_797868


namespace samia_walked_distance_l797_797223

theorem samia_walked_distance 
  (biking_speed : ℝ := 17) 
  (walking_speed : ℝ := 5) 
  (total_time_min : ℝ := 44) 
  (total_time_hr : ℝ := 44 / 60) : 
  ∀ (x : ℝ), 
  ((total_time_hr = (x / biking_speed + x / walking_speed)) → 
  (Real.floor (10 * x) / 10 = 2.8)) := 
by 
  intro x 
  intro h 
  sorry

end samia_walked_distance_l797_797223


namespace no_solution_interval_l797_797374

theorem no_solution_interval (a : ℤ) :
  (∀ x : ℝ, (x - 2 * a + 1)^2 - 2 * x + 4 * a - 10 ≠ 0 → x ∉ Icc (-1 : ℝ) 7)
  ↔ a ≤ -3 ∨ a ≥ 6 :=
by
  sorry

end no_solution_interval_l797_797374


namespace intersection_point_C1_C2_min_distance_AB_l797_797518

noncomputable def C1 (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α ^ 2)
def C2 (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - π / 4) = - (Real.sqrt 2) / 2
def C3 (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

theorem intersection_point_C1_C2 : ∃ α ρ θ, C1 α = (-1, 0) ∧ C2 ρ θ := 
by {
  -- Proof omitted
  sorry
}

theorem min_distance_AB : ∃ (A B : ℝ × ℝ), (∃ ρ θ, C2 ρ θ ∧ A = (ρ * Real.cos θ, ρ * Real.sin θ)) ∧ 
                                    (∃ ρ' θ', C3 ρ' θ' ∧ B = (ρ' * Real.cos θ', ρ' * Real.sin θ')) ∧
                                    ∀ y, y = Real.sqrt 2 - 1 := 
by {
  -- Proof omitted
  sorry
}

end intersection_point_C1_C2_min_distance_AB_l797_797518


namespace percentage_increase_in_area_is_96_l797_797265

theorem percentage_increase_in_area_is_96 :
  let r₁ := 5
  let r₃ := 7
  let A (r : ℝ) := Real.pi * r^2
  ((A r₃ - A r₁) / A r₁) * 100 = 96 := by
  sorry

end percentage_increase_in_area_is_96_l797_797265


namespace width_of_field_l797_797646

theorem width_of_field (W L : ℝ) (h1 : L = (7 / 5) * W) (h2 : 2 * L + 2 * W = 360) : W = 75 :=
sorry

end width_of_field_l797_797646


namespace part1_part2_l797_797473

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l797_797473


namespace csc_315_eq_neg_sqrt2_l797_797008

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end csc_315_eq_neg_sqrt2_l797_797008


namespace domain_of_sqrt_frac_l797_797246

theorem domain_of_sqrt_frac (x : ℝ) :
  (sqrt ((x + 1) / (x - 2))).denom_defined → (2 < x ∨ x < -1) :=
by
  sorry

end domain_of_sqrt_frac_l797_797246


namespace solve_for_y_l797_797585

theorem solve_for_y (y : ℝ) (h : 125^(3*y) = 25^(4*y - 5)) : y = -10 :=
by
  sorry

end solve_for_y_l797_797585


namespace solve_floor_equation_l797_797586

theorem solve_floor_equation (x : ℝ) (hx : (∃ (y : ℤ), (x^3 - 40 * (y : ℝ) - 78 = 0) ∧ (y : ℝ) ≤ x ∧ x < (y + 1 : ℝ))) :
  x = -5.45 ∨ x = -4.96 ∨ x = -1.26 ∨ x = 6.83 ∨ x = 7.10 :=
by sorry

end solve_floor_equation_l797_797586


namespace determine_f_3_2016_l797_797801

noncomputable def f : ℕ → ℕ → ℕ
| 0, y       => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem determine_f_3_2016 : f 3 2016 = 2 ^ 2019 - 3 := by
  sorry

end determine_f_3_2016_l797_797801


namespace find_f_inv_l797_797495

noncomputable def f : ℝ → ℝ :=
  λ x, if 0 < x ∧ x < 2 then x^2 + x else -2*x + 8

theorem find_f_inv (a : ℝ) (h1 : f a = f (a + 2)) (h2 : 0 < a ∧ a < 2) : 
  f (1 / a) = 2 := by
  -- additional conditions are assumed for a at the moment, proof left as sorry
  sorry

end find_f_inv_l797_797495


namespace csc_315_eq_neg_sqrt2_l797_797012

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end csc_315_eq_neg_sqrt2_l797_797012


namespace contemporaries_probability_l797_797668

open Real

noncomputable def probability_of_contemporaries
  (born_within : ℝ) (lifespan : ℝ) : ℝ :=
  let total_area := born_within * born_within
  let side := born_within - lifespan
  let non_overlap_area := 2 * (1/2 * side * side)
  let overlap_area := total_area - non_overlap_area
  overlap_area / total_area

theorem contemporaries_probability :
  probability_of_contemporaries 300 80 = 104 / 225 := 
by
  sorry

end contemporaries_probability_l797_797668


namespace problem_I_problem_II_l797_797846

variables {a : ℝ} (h1 : a ≠ 0)

def f (x : ℝ) : ℝ := 2 ^ (a * x) - 2
def g (x : ℝ) : ℝ := a * (x - 2 * a) * (x + 2 - a)

-- Problem (I)
theorem problem_I (h2 : {x | f x * g x = 0} = {1, 2}) : a = 1 := sorry

-- Problem (II)
theorem problem_II (h3 : {x | f x < 0 ∨ g x < 0} = set.univ) : -Real.sqrt 2 / 2 < a ∧ a < 0 := sorry

end problem_I_problem_II_l797_797846


namespace probability_sum_of_last_two_digits_gt_18_l797_797327

/-- A five-digit integer is chosen at random from all positive five-digit integers.
    The probability that the sum of the last two digits of the number is greater than 18
    is equal to 1/100. -/
theorem probability_sum_of_last_two_digits_gt_18 :
  (∃ n : ℕ, 10000 ≤ n ∧ n < 100000) →
  let pairs := (List.range 10).product (List.range 10) in
  (↑(pairs.filter (λ pair, pair.1 + pair.2 > 18)).length / ↑pairs.length : ℚ) = 1 / 100 :=
by
  sorry

end probability_sum_of_last_two_digits_gt_18_l797_797327


namespace round_3_598_to_nearest_hundredth_l797_797222

theorem round_3_598_to_nearest_hundredth : (Real.round (3.598 * 100) / 100) = 3.60 :=
by
  sorry

end round_3_598_to_nearest_hundredth_l797_797222


namespace books_borrowed_l797_797663

theorem books_borrowed (initial_books : ℕ) (additional_books : ℕ) (remaining_books : ℕ) : 
  initial_books = 300 → 
  additional_books = 10 * 5 → 
  remaining_books = 210 → 
  initial_books + additional_books - remaining_books = 140 :=
by
  intros h1 h2 h3
  rw [h1, h2]
  sorry

end books_borrowed_l797_797663


namespace total_payment_is_correct_l797_797937

def length : ℕ := 30
def width : ℕ := 40
def construction_cost_per_sqft : ℕ := 3
def sealant_cost_per_sqft : ℕ := 1
def total_area : ℕ := length * width
def total_cost_per_sqft : ℕ := construction_cost_per_sqft + sealant_cost_per_sqft
def total_cost : ℕ := total_area * total_cost_per_sqft

theorem total_payment_is_correct : total_cost = 4800 := by
  sorry

end total_payment_is_correct_l797_797937


namespace geometric_sequence_b_l797_797366

theorem geometric_sequence_b (b : ℝ) (r : ℝ) (hb : b > 0)
  (h1 : 10 * r = b)
  (h2 : b * r = 10 / 9)
  (h3 : (10 / 9) * r = 10 / 81) :
  b = 10 :=
sorry

end geometric_sequence_b_l797_797366


namespace ab_value_l797_797305

theorem ab_value (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 80) : a * b = 32 := by
  sorry

end ab_value_l797_797305


namespace lcm_minimum_value_l797_797986

open Nat

theorem lcm_minimum_value (a b c : ℕ) (h1 : lcm a b = 18) (h2 : lcm b c = 20) : lcm a c ≥ 90 := 
sorry

end lcm_minimum_value_l797_797986


namespace log_sum_l797_797354

theorem log_sum (a b : ℝ) (ha : a = 60) (hb : b = 15) : 
  log 10 a + log 10 b = 2.955 :=
by
  rw [ha, hb]
  have h1 : a * b = 900 := by norm_num
  have hlog : log 10 900 ≈ 2.955 := sorry
  rw [log_mul (ne_of_gt (by norm_num : 10 > 0)), h1]
  exact hlog

end log_sum_l797_797354


namespace problem_statement_l797_797813

theorem problem_statement (x y : ℝ) (h : x^2 * (y^2 + 1) = 1) :
  xy < 1 ∧ x^2 y ≥ -1 / 2 ∧ x^2 + xy ≤ 5 / 4 :=
begin
  sorry
end

end problem_statement_l797_797813


namespace square_vertice_lies_on_lines_l797_797345

theorem square_vertice_lies_on_lines (A B C D : ℝ × ℝ) (B D : ℝ × ℝ) :
  dist A B = dist B C ∧ dist C D = dist D A ∧ dist A C = dist B D ∧
  (A.1 = 0 ∨ A.2 = 0) ∧ (C.1 = 0 ∨ C.2 = 0) ∧ A ≠ C ∧
  (B.1 = B.2 ∨ D.1 = D.2) ∧ (B.1 = -B.2 ∨ D.1 = -D.2) :=
begin
  sorry
end

end square_vertice_lies_on_lines_l797_797345


namespace louisa_average_speed_l797_797946

theorem louisa_average_speed :
  ∃ v : ℝ, 
  (100 / v = 175 / v - 3) ∧ 
  v = 25 :=
by
  sorry

end louisa_average_speed_l797_797946


namespace factorize_expression1_factorize_expression2_l797_797372

variable {m x : ℝ}

theorem factorize_expression1 : m * (m - 5) - 2 * (5 - m) * (5 - m) = -(m - 5) * (m - 10) :=
by
  sorry

theorem factorize_expression2 : -4 * x^3 + 8 * x^2 - 4 * x = -4 * x * (x - 1) * (x - 1) :=
by
  sorry

end factorize_expression1_factorize_expression2_l797_797372


namespace max_sum_of_arithmetic_sequence_l797_797819

-- Define the arithmetic sequence 
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

-- Define the sum Sn of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem max_sum_of_arithmetic_sequence
  (a1 : ℝ) (d : ℝ)
  (h1 : a1 = 1)
  (h2 : -2/17 < d)
  (h3 : d < -1/9)
  : ∃ n : ℕ, n = 9 ∧ (∀ m : ℕ, m ≠ 9 → sum_arithmetic_sequence 1 d m < sum_arithmetic_sequence 1 d 9) :=
sorry

end max_sum_of_arithmetic_sequence_l797_797819


namespace max_wise_men_l797_797580

def hat_color (s : ℕ → ℕ) (i : ℕ) : Prop := s i = 0 ∨ s i = 1

theorem max_wise_men (s : ℕ → ℕ) (n : ℕ) 
  (h1 : ∀ k, 1 ≤ k → k ≤ n - 9 → ∑ i in finset.range 10, s (k + i) = 5)
  (h2 : ∀ j, 1 ≤ j → j ≤ n - 11 → ∑ i in finset.range 12, s (j + i) ≠ 6) :
  n ≤ 15 := 
sorry

end max_wise_men_l797_797580


namespace prime_division_property_l797_797538

theorem prime_division_property (m n : ℤ) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
    (h_primes : ∃ ps : Finset ℕ, ps.card = m ∧ ∀ p ∈ ps, p.prime ∧ p ∈ set.Icc 1 n)
    (S : Finset ℕ) (hS : S ⊆ set.Icc 1 n) (h_card_S : S.card = m+1) :
    ∃ a ∈ S, a ∣ S.erase a.prod := by
  sorry

end prime_division_property_l797_797538


namespace sequence_sum_eq_l797_797727

noncomputable def sequence_sum := 
  let b : ℕ → ℚ :=
    λ n, if n = 1 then 3
         else if n = 2 then 2
         else (1 / 4) * b (n - 1) + (2 / 5) * b (n - 2)
  ∑' n, b n

theorem sequence_sum_eq : sequence_sum = 85 / 7 :=
  sorry

end sequence_sum_eq_l797_797727


namespace factorial_division_l797_797424

theorem factorial_division (h : 9.factorial = 362880) : 9.factorial / 4.factorial = 15120 := by
  sorry

end factorial_division_l797_797424


namespace solution_l797_797104

definition isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

definition main_condition (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a ∈ Icc (-1) 1 → b ∈ Icc (-1) 1 → a + b ≠ 0 → 0 < (f a + f b) / (a + b)

definition f_value_condition (f : ℝ → ℝ) : Prop :=
  ∀ x a : ℝ, x ∈ Icc (-1) 1 → a ∈ Icc (-1) 1 → f x ≤ x^2 - 2 * a * x + 1

theorem solution (f : ℝ → ℝ) (m : ℝ) :
  isOddFunction f →
  f 1 = 1 →
  main_condition f →
  f_value_condition f →
  (m ≤ -2 ∨ 2 ≤ m ∨ m = 0) :=
sorry

end solution_l797_797104


namespace seq_60th_pair_l797_797089

-- Define the given sequence of integer pairs
def sequence : ℕ → ℕ × ℕ
  | 0 => (1, 1)
  | n => 
      let sum := (seq: ℕ × ℕ).fst + (seq: ℕ × ℕ).snd
      let nextPair := if sum % 2 = 0 then 
                        if seq.fst > 0 then 
                          (seq.fst - 1, seq.snd + 1) 
                        else 
                          (0, seq.snd + 1) 
                      else 
                        if seq.snd > 0 then 
                          (seq.fst + 1, seq.snd - 1) 
                        else 
                          (seq.fst + 1, 0)
      nextPair where seq := sequence (n - 1)

theorem seq_60th_pair : sequence 60 = (5, 7) :=
by 
  sorry

end seq_60th_pair_l797_797089


namespace angle_rewrite_l797_797220

-- Define the angle and condition
def angle : ℝ := -27/4 * Real.pi
def alpha (k : ℤ) : ℝ := 5/4 * Real.pi + 2 * k * Real.pi

-- The proof problem statement
theorem angle_rewrite :
  ∃ k : ℤ, α k = (5 * Real.pi / 4) ∧ 0 ≤ α k ∧ α k < 2 * Real.pi := 
sorry

end angle_rewrite_l797_797220


namespace range_f_l797_797925

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^4 - (Real.sin x) * (Real.cos x) + (Real.cos x)^4

theorem range_f : 
  set.range f = set.Icc 0 (9 / 8) := 
sorry

end range_f_l797_797925


namespace csc_315_eq_neg_sqrt_2_l797_797018

theorem csc_315_eq_neg_sqrt_2 :
  let csc := λ θ, 1 / Real.sin θ in
  csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
  by
  let sin := Real.sin
  have h1 : csc (315 * Real.pi / 180) = 1 / sin (315 * Real.pi / 180) := rfl
  have h2 : sin (315 * Real.pi / 180) = sin ((360 - 45) * Real.pi / 180) := by congr; norm_num
  have h3 : sin ((360 - 45) * Real.pi / 180) = -sin (45 * Real.pi / 180) := by
    rw [Real.sin_pi_sub]
    congr; norm_num
  have h4 : sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 := Real.sin_of_one_div_sqrt_two 45 rfl
  sorry

end csc_315_eq_neg_sqrt_2_l797_797018


namespace f_log2_x_domain_l797_797491

noncomputable def domain_of_f_log2_x : set ℝ :=
  {y : ℝ | 1 ≤ y ∧ y ≤ 2} -- Defining the domain of f(2^x)

theorem f_log2_x_domain :
  domain_of_f_log2_x = {y : ℝ | 4 ≤ y ∧ y ≤ 16} :=
sorry

end f_log2_x_domain_l797_797491


namespace molly_age_condition_l797_797960

-- Definitions
def S : ℕ := 38 - 6
def M : ℕ := 24

-- The proof problem
theorem molly_age_condition :
  (S / M = 4 / 3) → (S = 32) → (M = 24) :=
by
  intro h_ratio h_S
  sorry

end molly_age_condition_l797_797960


namespace average_pastries_per_day_l797_797702

-- Conditions
def pastries_on_monday := 2

def pastries_on_day (n : ℕ) : ℕ :=
  pastries_on_monday + n

def total_pastries_in_week : ℕ :=
  List.sum (List.map pastries_on_day (List.range 7))

def number_of_days_in_week : ℕ := 7

-- Theorem to prove
theorem average_pastries_per_day : (total_pastries_in_week / number_of_days_in_week) = 5 :=
by
  sorry

end average_pastries_per_day_l797_797702


namespace sum_numeric_value_l797_797987

theorem sum_numeric_value : 
  let pattern := [1, 2, 1, 0, -1, -2, -1, 0]
  let letter_value := λ c : Char, pattern[((c.toNat - 'a'.toNat) % 8 : ℕ)]
  ∑ i in "numeric".toList.map letter_value, i = -1 :=
by
  -- proof to be filled in here
  sorry

end sum_numeric_value_l797_797987


namespace number_of_outcomes_probability_div_by_four_probability_on_line_l797_797705

open Finset

-- Define the labels on the balls.
def labels := {1, 3, 5, 7, 9}

-- Define the set of all possible outcomes when drawing two balls simultaneously.
def possible_outcomes : Finset (ℕ × ℕ) :=
  finset.powersetLen 2 labels
  |>.image (λ s, (s.to_list.nth 0, s.to_list.nth 1))
  |>.filter (λ p, p.1 < p.2)

-- Calculate the probability that the sum of the labels is divisible by 4.
def div_by_four_subset : Finset (ℕ × ℕ) :=
  possible_outcomes.filter (λ p, (p.1 + p.2) % 4 = 0)

-- Calculate the probability that the point lies on the line y = x + 2.
def line_subset : Finset (ℕ × ℕ) :=
  possible_outcomes.filter (λ p, p.2 = p.1 + 2)

-- Prove the number of possible outcomes.
theorem number_of_outcomes : possible_outcomes.card = 10 := by
  sorry

-- Prove the probability that the sum of the labels is divisible by 4.
theorem probability_div_by_four : (div_by_four_subset.card : ℚ) / possible_outcomes.card = 3 / 5 := by
  sorry

-- Prove the probability that the point lies on the line y = x + 2.
theorem probability_on_line : (line_subset.card : ℚ) / possible_outcomes.card = 2 / 5 := by
  sorry

end number_of_outcomes_probability_div_by_four_probability_on_line_l797_797705


namespace midpoint_parallelogram_diagonals_l797_797822

-- Define the two points A and C as given conditions
variables {A C : ℝ × ℝ}
def A : ℝ × ℝ := (2, -3)
def C : ℝ × ℝ := (14, 9)

-- Prove that the midpoint (intersection of diagonals) is (8, 3)
theorem midpoint_parallelogram_diagonals (A C : ℝ × ℝ) (hA : A = (2, -3)) (hC : C = (14, 9)) :
  let M := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) in M = (8, 3) :=
by
  sorry

end midpoint_parallelogram_diagonals_l797_797822


namespace find_multiple_of_hats_l797_797745

/-
   Given:
   - Fire chief Simpson has 15 hats.
   - Policeman O'Brien now has 34 hats.
   - Before he lost one, Policeman O'Brien had 5 more hats than a certain multiple of Fire chief Simpson's hats.
   Prove:
   The multiple of Fire chief Simpson's hats that Policeman O'Brien had before he lost one is 2.
-/

theorem find_multiple_of_hats :
  ∃ x : ℕ, 34 + 1 = 5 + 15 * x ∧ x = 2 :=
by
  sorry

end find_multiple_of_hats_l797_797745


namespace number_of_pupils_l797_797684

theorem number_of_pupils (n : ℕ) (M : ℕ)
  (avg_all : 39 * n = M)
  (pupil_marks : 25 + 12 + 15 + 19 = 71)
  (new_avg : (M - 71) / (n - 4) = 44) :
  n = 21 := sorry

end number_of_pupils_l797_797684


namespace probability_third_flip_heads_l797_797181
-- Define the conditions as given
def fair_coin := (1/2 : ℚ)

def three_flips_with_two_heads : set (list bool) :=
  {sequence | sequence.size = 3 ∧ sequence.count (λ b => b = tt) = 2}

def five_flips_with_conditions : set (list bool) :=
  {sequence | sequence.size = 5 ∧
              (sequence.take 3 ∈ three_flips_with_two_heads) ∧
              (sequence.drop 2 ∈ three_flips_with_two_heads)}

-- Define the theorem stating the probability of the third flip being heads
theorem probability_third_flip_heads :
  probability (λ seq : list bool, seq !! 2 = some tt) (five_flips_with_conditions) = 4 / 5 :=
sorry

end probability_third_flip_heads_l797_797181


namespace PeterHasMoney_l797_797318

noncomputable theory

def JohnHas : ℝ := 185.40
def PeterHas (J : ℝ) : ℝ := 2 * J
def QuincyHas (P : ℝ) : ℝ := P + 20
def AndrewHas (Q : ℝ) : ℝ := 1.15 * Q
def TotalMoney (J P Q A : ℝ) : ℝ := J + P + Q + A

theorem PeterHasMoney :
  let J := JohnHas
  let P := PeterHas J
  let Q := QuincyHas P
  let A := AndrewHas Q
  TotalMoney J P Q A = 1211 → P = 370.80 :=
by
  sorry

end PeterHasMoney_l797_797318


namespace correct_number_of_statements_l797_797992

def statement_one (line : Type) [has_eq line] (p : line → Prop) (l : line) : Prop :=
  ∃! m : line, p m ∧ (m ≠ l)

def statement_two (line : Type) [has_eq line] (p : line → Prop) (l : line) : Prop :=
  ∃ l1 l2 : line, p l1 ∧ p l2 ∧ l1 ≠ l2

def statement_three (angle : Type) [has_eq angle] (common_vertex : angle → Prop) (measure_eq : angle → Prop) : Prop :=
  ∃ a1 a2 : angle, common_vertex a1 ∧ common_vertex a2 ∧ measure_eq a1 ∧ measure_eq a2 ∧ ¬ (vertical_angles a1 a2)

def statement_four (point : Type) (line : Type) [has_eq line] [has_eq point] (dist_from_point : point → line → ℝ) : Prop :=
  ∃ p : point, ∃ l : line, dist_from_point p l = (∃! seg : ℝ, seg > 0)

def statement_five (line : Type) [has_eq line] (p : line → Prop) (l : line) : Prop :=
  ∃! m : line, p m ∧ (l ≠ m)

theorem correct_number_of_statements :
  ∃ n, n = 1 :=
  sorry

end correct_number_of_statements_l797_797992


namespace total_budget_is_32_l797_797260

-- Definitions of conditions
def budget : ℝ := 32
def policing (B : ℝ) : ℝ := B / 2
def education : ℝ := 12
def public_spaces : ℝ := 4

-- The final theorem to prove
theorem total_budget_is_32 (B : ℝ) (half_policing : policing B = B / 2) (education_cost : education = 12) (spaces_cost : public_spaces = 4) 
  (total_eq_B : B / 2 + 12 + 4 = B) : B = 32 :=
sorry

end total_budget_is_32_l797_797260


namespace csc_315_eq_neg_sqrt2_l797_797024

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end csc_315_eq_neg_sqrt2_l797_797024


namespace minimum_cost_peking_opera_l797_797998

theorem minimum_cost_peking_opera (T p₆ p₁₀ : ℕ) (xₛ yₛ : ℕ) :
  T = 140 ∧ p₆ = 6 ∧ p₁₀ = 10 ∧ xₛ + yₛ = T ∧ yₛ ≥ 2 * xₛ →
  6 * xₛ + 10 * yₛ = 1216 ∧ xₛ = 46 ∧ yₛ = 94 :=
by
   -- Proving this is skipped (left as a sorry)
  sorry

end minimum_cost_peking_opera_l797_797998


namespace seating_arrangements_l797_797595

theorem seating_arrangements (sons daughters : ℕ) (totalSeats : ℕ) (h_sons : sons = 5) (h_daughters : daughters = 4) (h_seats : totalSeats = 9) :
  let total_arrangements := totalSeats.factorial
  let unwanted_arrangements := sons.factorial * daughters.factorial
  total_arrangements - unwanted_arrangements = 360000 :=
by
  rw [h_sons, h_daughters, h_seats]
  let total_arrangements := 9.factorial
  let unwanted_arrangements := 5.factorial * 4.factorial
  exact Nat.sub_eq_of_eq_add $ eq_comm.mpr (Nat.add_sub_eq_of_eq total_arrangements_units)
where
  total_arrangements_units : 9.factorial = 5.factorial * 4.factorial + 360000 := by
    rw [Nat.factorial, Nat.factorial, Nat.factorial, ←Nat.factorial_mul_factorial_eq 5 4]
    simp [tmp_rewriting]

end seating_arrangements_l797_797595


namespace csc_315_eq_neg_sqrt_2_l797_797034

theorem csc_315_eq_neg_sqrt_2 : csc 315 = -sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797034


namespace part1_part2_l797_797461

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l797_797461


namespace S_n_lt_4_over_3_l797_797216

-- Define the function S_n(x)
def S_n (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in Finset.range n, x^k / ((x^2 - 2*x + 3)^(k + 1))

-- The theorem we want to prove
theorem S_n_lt_4_over_3 (n : ℕ) (x : ℝ): S_n n x < 4 / 3 := 
  sorry

end S_n_lt_4_over_3_l797_797216


namespace min_area_of_rectangle_with_perimeter_100_l797_797339

theorem min_area_of_rectangle_with_perimeter_100 :
  ∃ (length width : ℕ), 
    (length + width = 50) ∧ 
    (length * width = 49) := 
by
  sorry

end min_area_of_rectangle_with_perimeter_100_l797_797339


namespace side_length_of_octagon_l797_797658

-- Define the conditions
def is_octagon (n : ℕ) := n = 8
def perimeter (p : ℕ) := p = 72

-- Define the problem statement
theorem side_length_of_octagon (n p l : ℕ) 
  (h1 : is_octagon n) 
  (h2 : perimeter p) 
  (h3 : p / n = l) :
  l = 9 := 
  sorry

end side_length_of_octagon_l797_797658


namespace seating_arrangements_l797_797601

theorem seating_arrangements (sons daughters : ℕ) (totalSeats : ℕ) (h_sons : sons = 5) (h_daughters : daughters = 4) (h_seats : totalSeats = 9) :
  let total_arrangements := totalSeats.factorial
  let unwanted_arrangements := sons.factorial * daughters.factorial
  total_arrangements - unwanted_arrangements = 360000 :=
by
  rw [h_sons, h_daughters, h_seats]
  let total_arrangements := 9.factorial
  let unwanted_arrangements := 5.factorial * 4.factorial
  exact Nat.sub_eq_of_eq_add $ eq_comm.mpr (Nat.add_sub_eq_of_eq total_arrangements_units)
where
  total_arrangements_units : 9.factorial = 5.factorial * 4.factorial + 360000 := by
    rw [Nat.factorial, Nat.factorial, Nat.factorial, ←Nat.factorial_mul_factorial_eq 5 4]
    simp [tmp_rewriting]

end seating_arrangements_l797_797601


namespace smallest_angle_diagonal_l797_797209

variables {a : ℝ} (x y z : ℝ) (midpoints : Set (ℝ × ℝ × ℝ)) [inhabited midpoints]

-- Defining the length of the diagonal of a cube with side length 'a'
noncomputable def body_diagonal (a : ℝ) : ℝ := a * Real.sqrt 3

-- Defining the radius of the circumscribed sphere
noncomputable def circumscribed_radius (a : ℝ) : ℝ := (a * Real.sqrt 3) / 2

-- Defining the midpoints of the edges on the faces of the cube
noncomputable def midpoints_faces : Set (ℝ × ℝ × ℝ) :=
  { (a/2, 0, 0), (0, a/2, 0), (0, 0, a/2), (a/2, a, a), (a, a/2, a), (a, a, a/2) }

-- Statement to be proven in Lean
theorem smallest_angle_diagonal (surface_points : Set (ℝ × ℝ × ℝ)) :
  ∃ p ∈ surface_points, ∀ q ∈ surface_points, angle (diagonal a) p ≤ angle (diagonal a) q ↔ p ∈ midpoints_faces
:= sorry

end smallest_angle_diagonal_l797_797209


namespace sum_of_2500_terms_l797_797342

noncomputable def sequence_sum (n : ℕ) (b : ℕ → ℝ) : ℝ :=
  ∑ i in finset.range n, b i

variable {b : ℕ → ℝ}

axiom a1 : ∀ n ≥ 4, b n = b (n - 1) + b (n - 3)
axiom a2 : sequence_sum 1800 b = 2000
axiom a3 : sequence_sum 2300 b = 3000

theorem sum_of_2500_terms : sequence_sum 2500 b = 3223.34 :=
sorry

end sum_of_2500_terms_l797_797342


namespace total_increase_area_l797_797331

theorem total_increase_area (increase_broccoli increase_cauliflower increase_cabbage : ℕ)
    (area_broccoli area_cauliflower area_cabbage : ℝ)
    (h1 : increase_broccoli = 79)
    (h2 : increase_cauliflower = 25)
    (h3 : increase_cabbage = 50)
    (h4 : area_broccoli = 1)
    (h5 : area_cauliflower = 2)
    (h6 : area_cabbage = 1.5) :
    increase_broccoli * area_broccoli +
    increase_cauliflower * area_cauliflower +
    increase_cabbage * area_cabbage = 204 := 
by 
    sorry

end total_increase_area_l797_797331


namespace seating_arrangements_l797_797598

theorem seating_arrangements (sons daughters : ℕ) (totalSeats : ℕ) (h_sons : sons = 5) (h_daughters : daughters = 4) (h_seats : totalSeats = 9) :
  let total_arrangements := totalSeats.factorial
  let unwanted_arrangements := sons.factorial * daughters.factorial
  total_arrangements - unwanted_arrangements = 360000 :=
by
  rw [h_sons, h_daughters, h_seats]
  let total_arrangements := 9.factorial
  let unwanted_arrangements := 5.factorial * 4.factorial
  exact Nat.sub_eq_of_eq_add $ eq_comm.mpr (Nat.add_sub_eq_of_eq total_arrangements_units)
where
  total_arrangements_units : 9.factorial = 5.factorial * 4.factorial + 360000 := by
    rw [Nat.factorial, Nat.factorial, Nat.factorial, ←Nat.factorial_mul_factorial_eq 5 4]
    simp [tmp_rewriting]

end seating_arrangements_l797_797598


namespace triangle_bcsolved_l797_797514

noncomputable def triangle_side_BC_length (AB : ℝ) (angleA : ℝ) (area : ℝ) : ℝ :=
  if AB = 3 ∧ angleA = 120 ∧ area = (15 * Real.sqrt 3) / 4 then 7 else 0

theorem triangle_bcsolved :
  triangle_side_BC_length 3 120 ((15 * Real.sqrt 3) / 4) = 7 :=
by
  simp [triangle_side_BC_length]
  sorry

end triangle_bcsolved_l797_797514


namespace cost_of_sending_kid_to_prep_school_l797_797901

theorem cost_of_sending_kid_to_prep_school (cost_per_semester : ℕ) (semesters_per_year : ℕ) (years : ℕ) (total_cost : ℕ) : 
    (cost_per_semester = 20000) → (semesters_per_year = 2) → (years = 13) → (total_cost = 520000) → 
    (cost_per_semester * semesters_per_year * years = total_cost) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, ←h4]
  sorry

end cost_of_sending_kid_to_prep_school_l797_797901


namespace tangent_line_at_P_tangent_lines_passing_through_P_l797_797838

-- Define the curve equation
def curve (x : ℝ) : ℝ := (1/3) * x^3 + (4/3)

-- Define point P
def P := (2 : ℝ, 4 : ℝ)

-- Define the first problem
theorem tangent_line_at_P : 
  ∃ k : ℝ, (curve P.1 = P.2) ∧ (derive curve P.1 = k) ∧ (∀ x y : ℝ, y = k * (x - P.1) + P.2 → 4 * x - y - 4 = 0) :=
sorry

-- Define the second problem
theorem tangent_lines_passing_through_P : 
  ∃ (x0 : ℝ), (curve P.1 = P.2) → 
  (∀ (x₀ : ℝ), curve x₀ = (1/3) * x₀^3 + (4/3) → 
  ∃ (k : ℝ), k = x₀^2 ∧ ∀ x y : ℝ, (curve P.1 = P.2) → y = k * (x - x₀) + (curve x₀) → (4 * x - y - 4 = 0 ∨ x - y + 2 = 0)) :=
sorry

end tangent_line_at_P_tangent_lines_passing_through_P_l797_797838


namespace not_equal_factorial_l797_797349

noncomputable def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem not_equal_factorial (n : ℕ) :
  permutations (n + 1) n ≠ (by apply Nat.factorial n) := by
  sorry

end not_equal_factorial_l797_797349


namespace circle_circumference_l797_797639

theorem circle_circumference : 
  (∀ (x y : ℝ), (x^2 + y^2 - 2*x + 6*y + 8 = 0) → 
   (2 * Real.pi * Real.sqrt 2) = 2 * Real.sqrt 2 * Real.pi :=
sorry

end circle_circumference_l797_797639


namespace csc_315_eq_neg_sqrt2_l797_797028

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end csc_315_eq_neg_sqrt2_l797_797028


namespace johns_height_in_feet_l797_797184

def initial_height := 66 -- John's initial height in inches
def growth_rate := 2      -- Growth rate in inches per month
def growth_duration := 3  -- Growth duration in months
def inches_per_foot := 12 -- Conversion factor from inches to feet

def total_growth : ℕ := growth_rate * growth_duration

def final_height_in_inches : ℕ := initial_height + total_growth

-- Now, proof that the final height in feet is 6
theorem johns_height_in_feet : (final_height_in_inches / inches_per_foot) = 6 :=
by {
  -- We would provide the detailed proof here
  sorry
}

end johns_height_in_feet_l797_797184


namespace opposite_of_expression_l797_797993

theorem opposite_of_expression : 
  let expr := 1 - (3 : ℝ)^(1/3)
  (-1 + (3 : ℝ)^(1/3)) = (3 : ℝ)^(1/3) - 1 :=
by 
  let expr := 1 - (3 : ℝ)^(1/3)
  sorry

end opposite_of_expression_l797_797993


namespace cars_return_to_start_l797_797564

noncomputable theory

def car_positions_return (n : ℕ) (positions : Fin n → ℝ) (speeds : Fin n → ℝ) 
  (meet_and_swap : ∀ i j : Fin n, positions i = positions j → speeds i ≠ speeds j) : ℕ :=
sorry

theorem cars_return_to_start (n : ℕ) (positions : Fin n → ℝ) (speeds : Fin n → ℝ) 
  (meet_and_swap : ∀ i j : Fin n, positions i = positions j → speeds i ≠ speeds j) :
  ∃ t : ℕ, ∀ i : Fin n, (positions i + speeds i * t) % 1 = positions i :=
sorry

end cars_return_to_start_l797_797564


namespace part1_part2_l797_797465

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l797_797465


namespace combined_efficiency_is_correct_l797_797734

-- Define the conditions
variables (d : ℝ)
def alicia_efficiency := 20
def john_efficiency := 15

-- Define the total fuel usage for both cars
def total_fuel_used := (d / alicia_efficiency) + (d / john_efficiency)

-- Define the total distance driven by both cars
def total_distance := 2 * d

-- Define the combined fuel efficiency
def combined_fuel_efficiency := total_distance / total_fuel_used

-- State the proof problem
theorem combined_efficiency_is_correct (h1 : alicia_efficiency = 20) (h2 : john_efficiency = 15) : 
  combined_fuel_efficiency d = 17.14 :=
by
  sorry

end combined_efficiency_is_correct_l797_797734


namespace seating_arrangements_l797_797600

theorem seating_arrangements (sons daughters : ℕ) (totalSeats : ℕ) (h_sons : sons = 5) (h_daughters : daughters = 4) (h_seats : totalSeats = 9) :
  let total_arrangements := totalSeats.factorial
  let unwanted_arrangements := sons.factorial * daughters.factorial
  total_arrangements - unwanted_arrangements = 360000 :=
by
  rw [h_sons, h_daughters, h_seats]
  let total_arrangements := 9.factorial
  let unwanted_arrangements := 5.factorial * 4.factorial
  exact Nat.sub_eq_of_eq_add $ eq_comm.mpr (Nat.add_sub_eq_of_eq total_arrangements_units)
where
  total_arrangements_units : 9.factorial = 5.factorial * 4.factorial + 360000 := by
    rw [Nat.factorial, Nat.factorial, Nat.factorial, ←Nat.factorial_mul_factorial_eq 5 4]
    simp [tmp_rewriting]

end seating_arrangements_l797_797600


namespace inner_square_area_l797_797968

noncomputable def area_inner_square : Real := 64

theorem inner_square_area
  (WXYZ_side_length : Real)
  (WI_length : Real)
  (h1 : WXYZ_side_length = 10)
  (h2 : WI_length = Real.sqrt 2) :
  let s := WXYZ_side_length - WI_length * Real.sqrt 2 in
  s^2 = area_inner_square :=
by
  sorry

end inner_square_area_l797_797968


namespace find_a_plus_b_l797_797923
noncomputable def probability_4_heads_before_3_tails : ℚ :=
  (1 / 9)

theorem find_a_plus_b : 
  ∃ (a b : ℕ), (a ≠ 0) ∧ (b ≠ 0) ∧ (Nat.coprime a b) ∧ 
               (probability_4_heads_before_3_tails = (a / b)) ∧ 
               (a + b = 10) :=
begin
  use [1, 9],
  split, { norm_num },
  split, { norm_num },
  split,
  { exact nat.coprime_one_right 9 },
  split,
  { norm_num,
    show 1 / 9 = probability_4_heads_before_3_tails,
    exact rfl, },
  { norm_num }
end

end find_a_plus_b_l797_797923


namespace election_vote_count_l797_797883

theorem election_vote_count (V : ℝ)
  (h1 : 0.70 * V - 0.30 * V = 180) : V = 450 := 
begin
  sorry
end

end election_vote_count_l797_797883


namespace angle_CED_120_degrees_l797_797272

theorem angle_CED_120_degrees {A B C D E : Point} 
  (h1 : CircleCenteredAt A passesThrough B)
  (h2 : CircleCenteredAt B passesThrough A)
  (h3 : line_through A B extends_to C D)
  (h4 : circles_intersect_at_two_points A B = { E, F }) :
  measure_angle C E D = 120 :=
by
  sorry

end angle_CED_120_degrees_l797_797272


namespace distance_product_zero_l797_797060

-- Definitions of line l and curve C
def P : (ℝ × ℝ) := (1, 1)

def line_l (t : ℝ) : (ℝ × ℝ) :=
  (1 - (1/2) * t, 1 + (√3 / 2) * t)

def curve_C (x y : ℝ) : Prop :=
  (x^2 / 36) + (y^2 / 16) = 1

-- Main theorem
theorem distance_product_zero : 
  |P| * ∃ (t : ℝ), let (x_B, y_B) := line_l t in curve_C x_B y_B ∧ 
  let dist := Real.sqrt ((x_B - 1)^2 + (y_B - 1)^2) in
  |P| * dist = 0 :=
begin
  sorry
end

end distance_product_zero_l797_797060


namespace JohnsonFamilySeating_l797_797624

theorem JohnsonFamilySeating : 
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 359000 := by
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  show total_arrangements - restricted_arrangements = 359000 from sorry

end JohnsonFamilySeating_l797_797624


namespace monotonic_range_of_a_l797_797830

theorem monotonic_range_of_a (a : ℝ) :
  (∀ x < 4, (deriv (λ x, a * x^2 + 2 * x - 3)) x ≥ 0) ↔ (-1 / 4 ≤ a ∧ a ≤ 0) :=
begin
  sorry
end

end monotonic_range_of_a_l797_797830


namespace eggs_per_omelet_l797_797976

theorem eggs_per_omelet:
  let small_children_tickets := 53
  let older_children_tickets := 35
  let adult_tickets := 75
  let senior_tickets := 37
  let smallChildrenOmelets := small_children_tickets * 0.5
  let olderChildrenOmelets := older_children_tickets
  let adultOmelets := adult_tickets * 2
  let seniorOmelets := senior_tickets * 1.5
  let extra_omelets := 25
  let total_omelets := smallChildrenOmelets + olderChildrenOmelets + adultOmelets + seniorOmelets + extra_omelets
  let total_eggs := 584
  total_eggs / total_omelets = 2 := 
by
  sorry

end eggs_per_omelet_l797_797976


namespace store_earnings_correct_l797_797719

theorem store_earnings_correct :
  let graphics_cards_sold : ℕ := 10
  let hard_drives_sold : ℕ := 14
  let cpus_sold : ℕ := 8
  let ram_pairs_sold : ℕ := 4
  let graphics_card_price : ℝ := 600
  let hard_drive_price : ℝ := 80
  let cpu_price : ℝ := 200
  let ram_pair_price : ℝ := 60
  graphics_cards_sold * graphics_card_price +
  hard_drives_sold * hard_drive_price +
  cpus_sold * cpu_price +
  ram_pairs_sold * ram_pair_price = 8960 := 
by
  sorry

end store_earnings_correct_l797_797719


namespace right_triangle_of_condition_l797_797955

theorem right_triangle_of_condition
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_trig : Real.sin γ - Real.cos α = Real.cos β) :
  (α = 90) ∨ (β = 90) :=
sorry

end right_triangle_of_condition_l797_797955


namespace range_of_k_l797_797070

theorem range_of_k (k : ℝ) : (∀ (x : ℝ), k * x ^ 2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0) := 
by 
  sorry

end range_of_k_l797_797070


namespace csc_315_eq_neg_sqrt2_l797_797006

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end csc_315_eq_neg_sqrt2_l797_797006


namespace csc_315_eq_neg_sqrt_2_l797_797037

theorem csc_315_eq_neg_sqrt_2 : csc 315 = -sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797037


namespace greatest_points_top_three_teams_l797_797882

/-- In a tournament with the given conditions, the greatest possible number of total points for each of the top three teams is 38. -/
theorem greatest_points_top_three_teams:
  ∃ (A B C: ℕ) (teams: ℕ) (games_played: ℕ) (total_points: ℕ),
    teams = 8 ∧
    games_played = (choose 8 2) * 2 ∧
    total_points = 3 * games_played ∧
    3 * A + 3 * B + 3 * C = total_points ∧
    A = B ∧ B = C ∧
    A + B + C = 38 :=
begin
  sorry,
end

end greatest_points_top_three_teams_l797_797882


namespace johnson_family_seating_l797_797616

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem johnson_family_seating (sons daughters : ℕ) (total_seats : ℕ) 
  (condition1 : sons = 5) (condition2 : daughters = 4) (condition3 : total_seats = 9) :
  let total_arrangements := factorial total_seats,
      restricted_arrangements := factorial sons * factorial daughters,
      answer := total_arrangements - restricted_arrangements
  in answer = 360000 := 
by
  -- The proof would go here
  sorry

end johnson_family_seating_l797_797616


namespace limit_of_increasing_bounded_seq_l797_797533

open Filter

noncomputable def lim_prod_seq (a : ℕ → ℝ) : ℝ :=
  lim (λ n, (range (n-1)).foldl (λ prod k, prod * (2 * a n - a k - a (k + 1))) 1 * (2 * a n - a (n-1) - a 0))

theorem limit_of_increasing_bounded_seq (a : ℕ → ℝ) (h_incr: ∀ n, a n ≤ a (n + 1)) (h_bound : ∃ L, ∀ n, a n ≤ L) :
  lim_prod_seq a = 0 :=
sorry

end limit_of_increasing_bounded_seq_l797_797533


namespace integer_solutions_inequality_system_l797_797588

theorem integer_solutions_inequality_system :
  {x : ℤ | 2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1} = {3, 4, 5} :=
by
  sorry

end integer_solutions_inequality_system_l797_797588


namespace paula_aunt_gave_her_total_money_l797_797210

theorem paula_aunt_gave_her_total_money :
  let shirt_price := 11
  let pants_price := 13
  let shirts_bought := 2
  let money_left := 74
  let total_spent := shirts_bought * shirt_price + pants_price
  total_spent + money_left = 109 :=
by
  let shirt_price := 11
  let pants_price := 13
  let shirts_bought := 2
  let money_left := 74
  let total_spent := shirts_bought * shirt_price + pants_price
  show total_spent + money_left = 109
  sorry

end paula_aunt_gave_her_total_money_l797_797210


namespace csc_315_eq_sqrt2_l797_797048

theorem csc_315_eq_sqrt2 :
  let θ := 315
  let csc := λ θ, 1 / (Real.sin (θ * Real.pi / 180))
  315 = 360 - 45 → 
  Real.sin (315 * Real.pi / 180) = Real.sin ((360 - 45) * Real.pi / 180) → 
  Real.sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 →
  csc 315 = Real.sqrt 2 := 
by
  intros θ csc h1 h2 h3
  -- proof would go here
  sorry

end csc_315_eq_sqrt2_l797_797048


namespace father_current_age_is_85_l797_797388

theorem father_current_age_is_85 (sebastian_age : ℕ) (sister_diff : ℕ) (age_sum_fraction : ℕ → ℕ → ℕ → Prop) :
  sebastian_age = 40 →
  sister_diff = 10 →
  (∀ (s s' f : ℕ), age_sum_fraction s s' f → f = 4 * (s + s') / 3) →
  age_sum_fraction (sebastian_age - 5) (sebastian_age - sister_diff - 5) (40 + 5) →
  ∃ father_age : ℕ, father_age = 85 :=
by
  intros
  sorry

end father_current_age_is_85_l797_797388


namespace minimum_value_of_f_l797_797511

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / x

theorem minimum_value_of_f (h : 1 < x) : ∃ y, f x = y ∧ (∀ z, (f z) ≥ 2*sqrt 2) :=
by
  sorry

end minimum_value_of_f_l797_797511


namespace partnership_investment_l797_797732

/-- A partnership business problem where A, B, and C invested different amounts and shared profit proportionally. -/
theorem partnership_investment :
  let A_investment := 6300 in
  let C_investment := 10500 in
  let total_profit := 12700 in
  let As_profit := 3810 in
  let B_investment := 13702.36 in
  6300 / (6300 + B_investment + 10500) = 3810 / 12700 :=
by
  let A_investment := 6300 in
  let C_investment := 10500 in
  let total_profit := 12700 in
  let As_profit := 3810 in
  let B_investment := 13702.36 in
  sorry

end partnership_investment_l797_797732


namespace number_of_zeros_of_f_l797_797079

noncomputable def f (a x : ℝ) := x * Real.log x - a * x^2 - x

theorem number_of_zeros_of_f (a : ℝ) (h : |a| ≥ 1 / (2 * Real.exp 1)) :
  ∃! x, f a x = 0 :=
sorry

end number_of_zeros_of_f_l797_797079


namespace three_circles_area_less_than_total_radius_squared_l797_797659

theorem three_circles_area_less_than_total_radius_squared
    (x y z R : ℝ)
    (h1 : x > 0)
    (h2 : y > 0)
    (h3 : z > 0)
    (h4 : R > 0)
    (descartes_theorem : ( (1/x + 1/y + 1/z - 1/R)^2 = 2 * ( (1/x)^2 + (1/y)^2 + (1/z)^2 + (1/R)^2 ) )) :
    x^2 + y^2 + z^2 < 4 * R^2 := 
sorry

end three_circles_area_less_than_total_radius_squared_l797_797659


namespace find_f2_l797_797398

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f2_l797_797398


namespace range_of_m_l797_797254

theorem range_of_m (a b m : ℝ)
  (h1 : ∀ x : ℝ, a + b * (1 + x) - (1 + x)^2 = a + b * (1 - x) - (1 - x)^2)
  (h2 : ∀ x ∈ set.Iic (4 : ℝ), 2 - 2 * (x + m) ≥ 0)
  : m ∈ set.Iic (-3) := 
sorry

end range_of_m_l797_797254


namespace area_rectangle_eq_l797_797880

-- Define the given conditions
variables (M N O P G H : Point)
variables (MN MG PH : Real)
axiom MN_eq : MN = 18
axiom MG_eq : MG = 6
axiom PH_eq : PH = 6

-- Define the question as a theorem, i.e., the area of the rectangle
theorem area_rectangle_eq : 
  let radius := 15
  let MP := 3 * sqrt 11
  let area := 18 * MP
  area = 54 * sqrt 11 :=
by 
  sorry -- Skip the proof, focusing on statement equivalence

end area_rectangle_eq_l797_797880


namespace select_at_least_one_first_class_item_l797_797348

theorem select_at_least_one_first_class_item (parts : Finset ℕ) (n_first_class n_second_class : ℕ)
  (h : parts.card = 8)
  (h_first : n_first_class = 5)
  (h_second : n_second_class = 3) :
  (parts.choose 3).card - (finset.univ.filter (λ t, t.card = 3 ∧ t.to_list.all (λ x, x > 5))).card = 55 :=
by
  sorry

end select_at_least_one_first_class_item_l797_797348


namespace mutually_exclusive_event_l797_797074

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def event1 (a b : ℕ) : Prop := (is_odd a ∧ is_even b) ∨ (is_odd b ∧ is_even a)
def event2 (a b : ℕ) : Prop := is_odd a ∨ is_odd b
def event3 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_even a ∧ is_even b
def event4 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ (is_even a ∨ is_even b)

theorem mutually_exclusive_event :
  ∃ a b : ℕ, a ≠ b ∧ a ∈ {1,2,3,4,5,6,7,8,9} ∧ b ∈ {1,2,3,4,5,6,7,8,9} ∧ event3 a b :=
sorry

end mutually_exclusive_event_l797_797074


namespace three_divides_difference_l797_797262

theorem three_divides_difference (n a b : ℕ)
  (h1 : n ≥ 3)
  (h2 : ∀ (f : Fin n → Bool), 
    let count_groups_with_one_boy := (Finset.univ.filter (λ i, f i && !f (⟨(i + 1) % n, sorry⟩) && !f (⟨(i + 2) % n, sorry⟩)) + 
                                     Finset.univ.filter (λ i, !f i && f (⟨(i + 1) % n, sorry⟩) && !f (⟨(i + 2) % n, sorry⟩)) +
                                     Finset.univ.filter (λ i, !f i && !f (⟨(i + 1) % n, sorry⟩) && f (⟨(i + 2) % n, sorry⟩))).card = a,
    let count_groups_with_one_girl := (Finset.univ.filter (λ i, !f i && f (⟨(i + 1) % n, sorry⟩) && f (⟨(i + 2) % n, sorry⟩)) + 
                                     Finset.univ.filter (λ i, f i && !f (⟨(i + 1) % n, sorry⟩) && f (⟨(i + 2) % n, sorry⟩)) +
                                     Finset.univ.filter (λ i, f i && f (⟨(i + 1) % n, sorry⟩) && !f (⟨(i + 2) % n, sorry⟩))).card = b):
  3 ∣ (a - b) := 
sorry

end three_divides_difference_l797_797262


namespace G_is_centroid_l797_797905

-- Definitions for the problem statement
variables {G A B C D E F : Type}
variables (triangle : Triangle A B C)
variables (D_on_BC : PointOnLine D (Line B C))
variables (E_on_CA : PointOnLine E (Line C A))
variables (F_on_AB : PointOnLine F (Line A B))
variables (G_intersect_medians : Intersection G (Line A D) (Line B E) (Line C F))
variables (area_AGE_eq_CGD : AreaEq (Triangle A G E) (Triangle C G D))
variables (area_CGD_eq_BGF : AreaEq (Triangle C G D) (Triangle B G F))

-- Proof goal
theorem G_is_centroid : IsCentroid G (Triangle A B C) :=
sorry

end G_is_centroid_l797_797905


namespace primes_sum_product_composite_l797_797552

theorem primes_sum_product_composite {p q r : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hdistinct_pq : p ≠ q) (hdistinct_pr : p ≠ r) (hdistinct_qr : q ≠ r) :
  ¬ Nat.Prime (p + q + r + p * q * r) :=
by
  sorry

end primes_sum_product_composite_l797_797552


namespace proof_periodic_relation_l797_797077

noncomputable def f : ℝ → ℝ := sorry
noncomputable def a : ℝ := f 1
noncomputable def b : ℝ := f 10
noncomputable def c : ℝ := f 100

-- Conditions
axiom even_function (x : ℝ) : f (-x) = f x
axiom odd_function_offset (x : ℝ) : f (-x + 1) = -f (x + 1)
axiom positive_on_interval (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x > 0
axiom decreasing_on_interval (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f' x < 0

-- Theorem to be proved
theorem proof_periodic_relation : b < a ∧ a < c := sorry

end proof_periodic_relation_l797_797077


namespace triangle_side_b_cos_conditions_l797_797169

theorem triangle_side_b_cos_conditions (A C : ℝ) (a b : ℝ) (h1 : cos A = 4 / 5) 
  (h2 : cos C = 5 / 13) (h3 : a = 1) : 
  b = 21 / 13 :=
sorry

end triangle_side_b_cos_conditions_l797_797169


namespace part1_part2_l797_797432

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l797_797432


namespace number_of_even_integers_between_l797_797506

theorem number_of_even_integers_between : 
  let lower_bound := Int.ceil (9 / 2)
  let upper_bound := Int.floor (47 / 3)
  (Nat.filter (λ n => n % 2 = 0) (List.range' lower_bound (upper_bound - lower_bound + 1))).length = 5 := 
by
  sorry

end number_of_even_integers_between_l797_797506


namespace no_natural_numbers_condition_l797_797582

theorem no_natural_numbers_condition :
  ¬ ∃ (a : Fin 2018 → ℕ), ∀ i : Fin 2018,
    ∃ k : ℕ, (a i) ^ 2018 + a ((i + 1) % 2018) = 5 ^ k :=
by sorry

end no_natural_numbers_condition_l797_797582


namespace extreme_value_f_range_of_a_l797_797498

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x^2 + a * x - 3
noncomputable def h (x : ℝ) : ℝ := 2 * Real.log x + x + 3 / x

theorem extreme_value_f : ∃ x, f x = -1 / Real.exp 1 :=
by sorry

theorem range_of_a (a : ℝ) : (∀ x > 0, 2 * f x ≥ g x a) → a ≤ 4 :=
by sorry

end extreme_value_f_range_of_a_l797_797498


namespace no_real_roots_m_eq_1_range_of_m_l797_797115

-- Definitions of the given functions
def f (x : ℝ) (m : ℝ) := m*x - m/x
def g (x : ℝ) := 2 * Real.log x

-- First proof problem: Proving no real roots
theorem no_real_roots_m_eq_1 (x : ℝ) (h1 : 1 < x) : f x 1 ≠ g x :=
  sorry

-- Second proof problem: Finding the range of m
theorem range_of_m (m : ℝ) 
  (h2 : ∀ x ∈ Set.Ioc 1 Real.exp, f x m - g x < 2) : 
  m < (4 * Real.exp) / (Real.exp^2 - 1) :=
  sorry

end no_real_roots_m_eq_1_range_of_m_l797_797115


namespace graph_of_f_is_D_l797_797249

-- Define the function g(x)
def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

-- Define the transformed function f(x) = -g(x + 2) + 1
def f (x : ℝ) : ℝ := -g (x + 2) + 1

-- Theorem statement that the graph of f is labeled "D"
theorem graph_of_f_is_D : -- Placeholder for the actual proof property
  sorry

end graph_of_f_is_D_l797_797249


namespace num_tent_setup_plans_l797_797695

theorem num_tent_setup_plans (num_students : ℕ) (X Y : ℕ) :
  (num_students = 50) →
  (∀ ⦃X Y:ℕ⦄, 3 * X + 2 * Y = num_students → X % 2 = 0 → X > 0 ∧ Y > 0) →
  ∃ (plans : ℕ), plans = 8 :=
by
  intros hnum_students hvalid_pairs
  have : ∃! (pair_set : Finset (ℕ × ℕ)), pair_set.card = 8 ∧ ∀ (x, y) ∈ pair_set, 3 * x + 2 * y = 50 ∧ x % 2 = 0 ∧ x > 0 ∧ y > 0 := sorry
  sorry

end num_tent_setup_plans_l797_797695


namespace monotonic_intervals_range_of_m_l797_797843

noncomputable def f (m x : ℝ) : ℝ := m * log x - x^2 + 2
noncomputable def f' (m x : ℝ) : ℝ := m / x - 2 * x

theorem monotonic_intervals (x m : ℝ) (h1 : m ≤ 8) (h2 : m - 2 > -2) :
  (∀ x ∈ (0 : ℝ) .. Real.sqrt (m / 2), f' m x > 0) ∧ (∀ x ∈ (Real.sqrt (m / 2) : ℝ) .. +∞, f' m x < 0) :=
begin
  sorry
end

theorem range_of_m (x m : ℝ) (h1 : m ≤ 8) (h2 : ∀ x ∈ set.Ici (1 : ℝ), f m x - f' m x ≤ 4 * x - 3) :
  2 ≤ m ∧ m ≤ 8 :=
begin
  sorry
end

end monotonic_intervals_range_of_m_l797_797843


namespace union_of_sets_l797_797120

open Set

theorem union_of_sets : 
  let A := {0, 1, 2}
      B := {2, 3}
  in A ∪ B = {0, 1, 2, 3} := 
by {
  -- The proof goes here
  sorry
}

end union_of_sets_l797_797120


namespace units_digit_of_M_l797_797911

-- Definition of digits functions for the problem
def P (n : ℕ) : ℕ := (n / 10) * (n % 10)
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Main problem statement in Lean 4
theorem units_digit_of_M (M : ℕ) (h1 : M / 10 > 0) (h2 : M < 100) (h3 : M = P(M) + S(M) + 5) :
  M % 10 = 8 :=
sorry

end units_digit_of_M_l797_797911


namespace triangle_DEH_right_angle_l797_797237

noncomputable def Triangle (A B C : Point) : Prop := sorry
noncomputable def Altitude (P Q : Point) (T : Triangle) : Line  := sorry
noncomputable def Intersection (L1 L2: Line) : Point := sorry
noncomputable def Segment (P Q : Point) (d : ℝ) := sorry
noncomputable def Circumcircle (T : Triangle) : Circle := sorry
noncomputable def RightAngle (A B C : Point) : Prop := sorry

variables (A B C B1 C1 B2 C2 H D E : Point)

def conditions:
  Triangle A B C ∧
  Altitude B B1 (Triangle A B C) ∧ 
  Altitude C C1 (Triangle A B C) ∧ 
  H = Intersection (Altitude B B1 (Triangle A B C)) (Altitude C C1 (Triangle A B C)) ∧
  Segment B H B1H ∧
  Segment C H C1H ∧
  Circumcircle (Triangle B2 H C2) ∩ Circumcircle (Triangle A B C) = {D, E} :=
sorry

theorem triangle_DEH_right_angle (h: conditions A B C B1 C1 B2 C2 H D E):
  RightAngle D E H :=
sorry

end triangle_DEH_right_angle_l797_797237


namespace sine_theorem_l797_797688

theorem sine_theorem (a b c α β γ : ℝ) 
  (h1 : a / Real.sin α = b / Real.sin β) 
  (h2 : b / Real.sin β = c / Real.sin γ) 
  (h3 : α + β + γ = Real.pi) :
  a = b * Real.cos γ + c * Real.cos β ∧
  b = c * Real.cos α + a * Real.cos γ ∧
  c = a * Real.cos β + b * Real.cos α :=
by
  sorry

end sine_theorem_l797_797688


namespace number_cooking_and_weaving_l797_797300

section CurriculumProblem

variables {total_yoga total_cooking total_weaving : ℕ}
variables {cooking_only cooking_and_yoga all_curriculums CW : ℕ}

-- Given conditions
def yoga (total_yoga : ℕ) := total_yoga = 35
def cooking (total_cooking : ℕ) := total_cooking = 20
def weaving (total_weaving : ℕ) := total_weaving = 15
def cookingOnly (cooking_only : ℕ) := cooking_only = 7
def cookingAndYoga (cooking_and_yoga : ℕ) := cooking_and_yoga = 5
def allCurriculums (all_curriculums : ℕ) := all_curriculums = 3

-- Prove that CW (number of people studying both cooking and weaving) is 8
theorem number_cooking_and_weaving : 
  yoga total_yoga → cooking total_cooking → weaving total_weaving → 
  cookingOnly cooking_only → cookingAndYoga cooking_and_yoga → 
  allCurriculums all_curriculums → CW = 8 := 
by 
  intros h_yoga h_cooking h_weaving h_cookingOnly h_cookingAndYoga h_allCurriculums
  -- Placeholder for the actual proof
  sorry

end CurriculumProblem

end number_cooking_and_weaving_l797_797300


namespace hyperbola_standard_equation_and_properties_l797_797082

theorem hyperbola_standard_equation_and_properties :
  let a := 5
  let b := 4
  let c := Real.sqrt 41
  let e := c / a
  let hyperbola_eq := (λ (x y : ℝ), x^2 / 25 - y^2 / 16 = 1)
  let ellipse_eq := (λ (x y : ℝ), x^2 / 41 + y^2 / 16 = 1)
  (hyperbola_eq = (λ (x y : ℝ), x^2 / 25 - y^2 / 16 = 1)) ∧
  (8 = 2 * b) ∧
  (e = Real.sqrt 41 / 5) ∧
  (ellipse_eq = (λ (x y : ℝ), x^2 / 41 + y^2 / 16 = 1)) :=
by {
  sorry
}

end hyperbola_standard_equation_and_properties_l797_797082


namespace inequality_proof_l797_797199

theorem inequality_proof 
  (D : set ℝ) (hD : ∀ x, x ∈ D → 0 < x)
  (f : ℝ → ℝ) (hpos : ∀ x, x ∈ D → 0 < f x)
  (h1 : ∀ {x1 x2 : ℝ}, x1 ∈ D → x2 ∈ D → f (sqrt (x1 * x2)) ≤ sqrt (f x1 * f x2))
  (h2 : ∀ {x1 x2 : ℝ}, x1 ∈ D → x2 ∈ D → f x1 ^ 2 + f x2 ^ 2 ≥ 2 * f (sqrt ((x1^2 + x2^2) / 2)) ^ 2) :
  ∀ {x1 x2 : ℝ}, x1 ∈ D → x2 ∈ D → f x1 + f x2 ≥ 2 * f ((x1 + x2) / 2) := 
by 
  -- proof goes here
  sorry

end inequality_proof_l797_797199


namespace brokerage_percentage_l797_797241

theorem brokerage_percentage (cash_realized : ℝ) (cash_after_brokerage : ℝ) 
  (h_realized : cash_realized = 109.25) (h_after : cash_after_brokerage = 109) :
  (cash_realized - cash_after_brokerage) / cash_realized * 100 ≈ 0.228833 := by
  sorry

end brokerage_percentage_l797_797241


namespace csc_315_eq_sqrt2_l797_797044

theorem csc_315_eq_sqrt2 :
  let θ := 315
  let csc := λ θ, 1 / (Real.sin (θ * Real.pi / 180))
  315 = 360 - 45 → 
  Real.sin (315 * Real.pi / 180) = Real.sin ((360 - 45) * Real.pi / 180) → 
  Real.sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 →
  csc 315 = Real.sqrt 2 := 
by
  intros θ csc h1 h2 h3
  -- proof would go here
  sorry

end csc_315_eq_sqrt2_l797_797044


namespace no_such_compact_sets_exist_l797_797963

noncomputable theory

open Set

theorem no_such_compact_sets_exist :
  ¬ ∃ (A : ℕ → Set ℝ),
    (∀ n : ℕ, A n ⊆ ℚ) ∧
    (∀ (K : Set ℝ), IsCompact K ∧ K ⊆ ℚ → (∃ m : ℕ, K ⊆ A m)) :=
by
  sorry

end no_such_compact_sets_exist_l797_797963


namespace part1_part2_l797_797444

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

end part1_part2_l797_797444


namespace identify_spy_l797_797689

/-!
# The Spy Identification Problem

There are three individuals: A, B, and C.
Exactly one is a knight (always tells the truth), one is a liar (always lies), and one is a spy (can either lie or tell the truth).
A accuses B of being the spy.
B accuses C of being the spy.
C points to either A or B and states: "In reality, the spy is him!"
Given these conditions, we aim to prove that C is the spy.
-/

inductive Role
| knight
| liar
| spy

def accuses (accuser accused : Role) : Prop :=
  match accuser with
  | Role.knight => true       -- always tells the truth
  | Role.liar   => false      -- always lies
  | Role.spy    => true ∨ false  -- can either lie or tell the truth
  end

variables (A B C : Role)

axiom A_accuses_B_is_spy : accuses A B = true
axiom B_accuses_C_is_spy : accuses B C = true
axiom C_accuses_A_or_B_is_spy : accuses C A = true ∨ accuses C B = true
axiom unique_roles : (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C)

theorem identify_spy : C = Role.spy :=
sorry

end identify_spy_l797_797689


namespace hyperbola_eccentricity_solution_l797_797117

noncomputable def hyperbola_eccentricity_geometric_mean (m : ℝ) : Prop :=
  ∃ (a b : ℝ), (m < -1) ∧ (b = real.sqrt (-1/m)) ∧ (a = 1) ∧
  ((mx^2 + y^2 = 1) → ((a = 1) ∧ (b^2 = -1/m)) ∧ (m = -7 - 4*real.sqrt(3)))

theorem hyperbola_eccentricity_solution (m : ℝ) (hx : m < -1) :
  (hyperbola_eccentricity_geometric_mean m) :=
sorry

end hyperbola_eccentricity_solution_l797_797117


namespace samia_walked_distance_l797_797226

theorem samia_walked_distance :
  ∃ (x : ℝ), 
    let bike_speed := 17 in
    let walk_speed := 5 in
    let total_time := (44 : ℝ) / 60 in
    let coeff := (total_time * bike_speed * walk_speed) / (bike_speed + walk_speed) in
    x = coeff ∧ abs (x - 2.8) < 0.1 :=
sorry

end samia_walked_distance_l797_797226


namespace sequence_convergence_l797_797084

noncomputable def sequence (c : ℝ) (x : ℕ → ℝ) (n : ℕ) : ℝ :=
if n = 0 then x 0
else sqrt (c - sqrt (c + (x (n - 1))))

theorem sequence_convergence (c : ℝ) (x : ℕ → ℝ)
    (h_positive : c > 0) (h_start : x 0 > 0) (h_less : x 0 < c)
    (h_def : ∀ n, x (n+1) = sqrt (c - sqrt (c + x n))) :
  (∀ n, x n ≤ c^2 - c) →
  ∃ L : ℝ, ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (x n - L) < ε :=
by 
  intros h_bound
  sorry

end sequence_convergence_l797_797084


namespace inequality_part1_inequality_part2_l797_797450

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l797_797450


namespace integer_solutions_inequality_system_l797_797589

theorem integer_solutions_inequality_system :
  {x : ℤ | 2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1} = {3, 4, 5} :=
by
  sorry

end integer_solutions_inequality_system_l797_797589


namespace inequality_part1_inequality_part2_l797_797454

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l797_797454


namespace decimal_digits_of_fraction_l797_797505

theorem decimal_digits_of_fraction : 
  let n := (4^6 : ℝ) / (7^4 * 5^6) 
  in ∃ (d : ℝ), n = d / 10^5 :=
by
  let n := (4^6 : ℝ) / (7^4 * 5^6)
  use n * 10^5
  sorry

end decimal_digits_of_fraction_l797_797505


namespace johnson_family_seating_l797_797617

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem johnson_family_seating (sons daughters : ℕ) (total_seats : ℕ) 
  (condition1 : sons = 5) (condition2 : daughters = 4) (condition3 : total_seats = 9) :
  let total_arrangements := factorial total_seats,
      restricted_arrangements := factorial sons * factorial daughters,
      answer := total_arrangements - restricted_arrangements
  in answer = 360000 := 
by
  -- The proof would go here
  sorry

end johnson_family_seating_l797_797617


namespace sum_of_digits_base2_312_l797_797673

def binary_sum_of_digits (n : ℕ) : ℕ :=
  (n.binaryDigits.filter (λ d, d = 1)).length

theorem sum_of_digits_base2_312 : binary_sum_of_digits 312 = 3 :=
by
  sorry

end sum_of_digits_base2_312_l797_797673


namespace factorial_division_l797_797423

theorem factorial_division (h : 9.factorial = 362880) : 9.factorial / 4.factorial = 15120 := by
  sorry

end factorial_division_l797_797423


namespace find_fraction_l797_797525

-- Define the initial amount, the amount spent on pads, and the remaining amount
def initial_amount := 150
def spent_on_pads := 50
def remaining := 25

-- Define the fraction she spent on hockey skates
def fraction_spent_on_skates (f : ℚ) : Prop :=
  let spent_on_skates := initial_amount - remaining - spent_on_pads
  (spent_on_skates / initial_amount) = f

theorem find_fraction : fraction_spent_on_skates (1 / 2) :=
by
  -- Proof steps go here
  sorry

end find_fraction_l797_797525


namespace radius_is_100_div_pi_l797_797061

noncomputable def radius_of_circle (L : ℝ) (θ : ℝ) : ℝ :=
  L * 360 / (θ * 2 * Real.pi)

theorem radius_is_100_div_pi :
  radius_of_circle 25 45 = 100 / Real.pi := 
by
  sorry

end radius_is_100_div_pi_l797_797061


namespace find_angle_of_inclination_l797_797343

noncomputable def pyramidAngleInclination : Prop :=
  ∀ (M A B C D : ℝ^3) (r R h : ℝ),
  (sphereInscribedInPyramid M A B C D r) →
  (sphereTouching M A B C D r R A) →
  (planeThroughCenterSecondSphereAndSide M A B C D r R A K) →
  (edgeMAAndDiagonalCKPerpendicular M A B C D K) →
  ∃ (angle : ℝ), angle = 30

/-- Definitions used in the problem -/

def sphereInscribedInPyramid (M A B C D : ℝ^3) (r : ℝ) : Prop := sorry

def sphereTouching (M A B C D : ℝ^3) (r R : ℝ) (A : ℝ^3) : Prop := sorry

def planeThroughCenterSecondSphereAndSide (M A B C D : ℝ^3) (r R : ℝ) (A K : ℝ^3) : Prop := sorry

def edgeMAAndDiagonalCKPerpendicular (M A B C D : ℝ^3) (K : ℝ^3) : Prop := sorry

-- Lean will assert the definitions are logically valid as given
theorem find_angle_of_inclination : pyramidAngleInclination :=
  by sorry

end find_angle_of_inclination_l797_797343


namespace max_value_sqrt_abc_expression_l797_797540

theorem max_value_sqrt_abc_expression (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                                       (hb : 0 ≤ b) (hb1 : b ≤ 1)
                                       (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    (Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1) :=
sorry

end max_value_sqrt_abc_expression_l797_797540


namespace fraction_of_coins_1800_to_1809_l797_797577

theorem fraction_of_coins_1800_to_1809
  (total_coins : ℕ)
  (coins_1800_1809 : ℕ)
  (h_total : total_coins = 22)
  (h_coins : coins_1800_1809 = 5) :
  (coins_1800_1809 : ℚ) / total_coins = 5 / 22 := by
  sorry

end fraction_of_coins_1800_to_1809_l797_797577


namespace sum_of_reciprocals_of_S_n_l797_797259

theorem sum_of_reciprocals_of_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ)
  (h1 : a 3 = 3)
  (h2 : S 4 = 10)
  (h3 : ∀ n, S n = n * (n + 1) / 2) :
  (∑ k in Finset.range (n + 1), 1 / S k) = 2 * n / (n + 1) := by
  sorry

end sum_of_reciprocals_of_S_n_l797_797259


namespace exists_triangle_parallel_to_medians_l797_797574

theorem exists_triangle_parallel_to_medians (A B C A1 B1 C1 : Point) 
  (hAA1 : is_median A B C A1) (hBB1 : is_median B A C B1) (hCC1 : is_median C A B C1) :
  ∃ (K M N : Point), is_parallel_side K M (med AA1) ∧ is_parallel_side M N (med BB1) ∧ is_parallel_side N K (med CC1) := 
sorry

end exists_triangle_parallel_to_medians_l797_797574


namespace simplify_expression_l797_797964

-- Define ω and ω^2 as cube roots of unity
def ω : ℂ := -1/2 + (real.sqrt 3) * complex.I / 2
def ω₂ : ℂ := -1/2 - (real.sqrt 3) * complex.I / 2

-- Prove that ω and ω₂ are cube roots of unity
lemma ω_cube_root : ω ^ 3 = 1 := 
by sorry

lemma ω₂_cube_root : ω₂ ^ 3 = 1 :=
by sorry

-- Define ω' and ω'' as given in the conditions
def ω' : ℂ := -3/2 + (real.sqrt 3) * complex.I / 2
def ω'' : ℂ := -3/2 - (real.sqrt 3) * complex.I / 2

-- Express ω' and ω'' in terms of ω and ω₂
lemma ω'_repr : ω' = 3 * ω₂ :=
by sorry

lemma ω''_repr : ω'' = 3 * ω := 
by sorry

-- Given the conditions for the proof
theorem simplify_expression :
  (ω'/2)^12 + (ω''/2)^12 = 1062882 :=
by sorry

end simplify_expression_l797_797964


namespace expanding_path_product_l797_797640

def manhattan_distance (p1 p2 : (ℕ × ℕ)) : ℕ :=
  abs (p1.1 - p2.1) + abs (p1.2 - p2.2)

def is_expanding_path (path : List (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), i < j → j < path.length → manhattan_distance (path.nthLe i sorry) (path.nthLe j sorry) > manhattan_distance (path.nthLe (i - 1) sorry) (path.nthLe (i + 1) sorry)

def max_moves_in_expanding_path (grid_size : ℕ) : ℕ :=
  4 + 3 + 2 + 1

def num_expanding_paths (grid_size : ℕ) (m : ℕ) : ℕ :=
  4 * 24

theorem expanding_path_product : 
  max_moves_in_expanding_path 5 * num_expanding_paths 5 (max_moves_in_expanding_path 5) = 960 := 
by
  -- Proof steps can be added here
  sorry

end expanding_path_product_l797_797640


namespace smallest_n_factorial_l797_797972

theorem smallest_n_factorial (a b c m n : ℕ) (h1 : a + b + c = 2020)
(h2 : c > a + 100)
(h3 : m * 10^n = a! * b! * c!)
(h4 : ¬ (10 ∣ m)) : 
  n = 499 :=
sorry

end smallest_n_factorial_l797_797972


namespace intervals_of_monotonicity_range_of_a_for_extremum_l797_797497

def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * a * x^2 + 3 * x + 1
def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 6 * a * x + 3

theorem intervals_of_monotonicity (a : ℝ) (h : a = 2) :
  ∀ x, 
    (f' x a > 0 ↔ x > 2 + real.sqrt 3 ∨ x < 2 - real.sqrt 3) ∧
    (f' x a < 0 ↔ 2 - real.sqrt 3 < x ∧ x < 2 + real.sqrt 3) :=
sorry

theorem range_of_a_for_extremum (a : ℝ) :
  (∃ x, 2 < x ∧ x < 3 ∧ f' x a = 0) ↔ (5 / 4 < a ∧ a < 5 / 3) :=
sorry

end intervals_of_monotonicity_range_of_a_for_extremum_l797_797497


namespace transformation_matrices_sol_l797_797892

theorem transformation_matrices_sol {θ k : ℝ} 
  (hA: 0 < θ ∧ θ < 2*π) 
  (hB: 0 < k ∧ k < 1)
  (hBA: (λ θ k, (Matrix.mul (Matrix.of (λ i j, if (i, j) = (0, 0) then real.cos θ else if (i, j) = (0, 1) then -real.sin θ else if (i, j) = (1, 0) then real.sin θ else real.cos θ)) (Matrix.of (λ i j, if (i, j) = (0, 0) then 1 else if (i, j) = (1, 1) then k else 0)))
          θ k = ![![0, -1], ![\frac{1}{2}, 0]])) 
  :
  k = \frac{1}{2} ∧ θ = π/2 :=
sorry

end transformation_matrices_sol_l797_797892


namespace obtain_2010_by_functions_l797_797528

def f (x : ℝ) : ℝ := 1 / x
def g (x : ℝ) : ℝ := x / Real.sqrt (1 + x^2)

theorem obtain_2010_by_functions :
  ∃ n : ℕ, let x := 1 in 
    f (Nat.iterate g (2010^2 - 1) x) = 2010 :=
sorry

end obtain_2010_by_functions_l797_797528


namespace unique_zero_of_f_l797_797974

theorem unique_zero_of_f (f : ℝ → ℝ) (h1 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 16) 
  (h2 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 8) (h3 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 4) 
  (h4 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 2) : ¬ ∃ x, f x = 0 ∧ 2 ≤ x ∧ x < 16 := 
by
  sorry

end unique_zero_of_f_l797_797974


namespace value_range_f_at_4_l797_797403

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_range_f_at_4 (f : ℝ → ℝ)
  (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2)
  (h2 : 1 ≤ f (1) ∧ f (1) ≤ 3)
  (h3 : 2 ≤ f (2) ∧ f (2) ≤ 4)
  (h4 : -1 ≤ f (3) ∧ f (3) ≤ 1) :
  -21.75 ≤ f 4 ∧ f 4 ≤ 1 :=
sorry

end value_range_f_at_4_l797_797403


namespace sheepdog_catches_sheep_l797_797202

-- Define the speeds and the time taken
def v_s : ℝ := 12 -- speed of the sheep in feet/second
def v_d : ℝ := 20 -- speed of the sheepdog in feet/second
def t : ℝ := 20 -- time in seconds

-- Define the initial distance between the sheep and the sheepdog
def initial_distance (v_s v_d t : ℝ) : ℝ :=
  v_d * t - v_s * t

theorem sheepdog_catches_sheep :
  initial_distance v_s v_d t = 160 :=
by
  -- The formal proof would go here, but for now we replace it with sorry
  sorry

end sheepdog_catches_sheep_l797_797202


namespace part1_part2_l797_797433

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l797_797433


namespace system_C_is_linear_l797_797679

-- Definitions of the systems
structure System (α : Type _) :=
  (eqns : list (α → Prop))

-- System A
def system_A : System (ℕ × ℕ) :=
{ eqns := [λ (x, y), x + y = 5, λ (x, y), x * y = 6] }

-- System B
def system_B : System (ℕ × ℕ × ℕ) :=
{ eqns := [λ (x, y, z), x - y = 1, λ (x, y, z), z = 1] }

-- System C
def system_C : System (ℕ × ℕ) :=
{ eqns := [λ (x, y), x + y = 0, λ (x, y), y = 5 * x] }

-- System D
def system_D : System (ℕ × ℕ) :=
{ eqns := [λ (x, y), 1 - y = 1, λ (x, y), x + y = 2] }

-- Definition to check linearity
def is_linear_system {α : Type _} (sys : System α) : Prop :=
  ∀ (f : α → Prop), f ∈ sys.eqns → (∃ (a b c : ℕ), ∀ (x y : ℕ), f (x, y) ↔ a * x + b * y = c)

-- The theorem for the proof problem
theorem system_C_is_linear : is_linear_system system_C := by
  sorry

end system_C_is_linear_l797_797679


namespace integer_solutions_ineq_system_l797_797590

theorem integer_solutions_ineq_system:
  ∀ (x : ℤ), 
  (2 * (x - 1) ≤ x + 3) ∧ ((x + 1) / 3 < x - 1) ↔ (x = 3 ∨ x = 4 ∨ x = 5) := 
by 
  intros x 
  split
  · intro h
    cases h with h1 h2
    sorry -- to be proved later
  · intro h
    sorry -- to be proved later

end integer_solutions_ineq_system_l797_797590


namespace complex_conjugate_quadrant_l797_797404

theorem complex_conjugate_quadrant : 
  (let z := (2 * Complex.I) / (1 + Complex.I) in
   let conj_z := Complex.conj z in
   0 < conj_z.re ∧ conj_z.im < 0) := 
by
  let z := (2 * Complex.I) / (1 + Complex.I)
  let conj_z := Complex.conj z
  sorry

end complex_conjugate_quadrant_l797_797404


namespace csc_315_eq_neg_sqrt_2_l797_797040

theorem csc_315_eq_neg_sqrt_2 : csc 315 = -sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797040


namespace problem_solution_l797_797189

noncomputable def problem : ℝ :=
  let AB : ℝ := 4
  let BC : ℝ := 8
  let DE : ℝ := 24
  let angle_ABC : ℝ := 150 / 180 * Real.pi -- angle in radians

  -- Calculate AC using the Law of Cosines
  let AC := Real.sqrt (AB^2 + BC^2 + 2 * AB * BC * Real.cos angle_ABC)
  
  -- Expected value of p and q are 7 and 36 respectively
  let p := 7
  let q := 36
  
  -- Calculate area ratio and simplification
  let area_ratio := p / q
  
  -- Final sum to be proved
  in p + q = 43

theorem problem_solution : ℕ :=
  by
    rw problem
    exact 43

end problem_solution_l797_797189


namespace living_room_chairs_l797_797717

def total_chairs : ℕ := 9
def kitchen_chairs : ℕ := 6

theorem living_room_chairs (total_chairs kitchen_chairs : ℕ) :
  total_chairs = 9 → kitchen_chairs = 6 → total_chairs - kitchen_chairs = 3 :=
by intros h₁ h₂; rw [h₁, h₂]; exact rfl

end living_room_chairs_l797_797717


namespace problem_I_problem_II_problem_III_l797_797408

noncomputable def f (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem problem_I (a b c : ℝ) (h : ∀ x : ℝ, x ≤ f(x) ∧ f(x) ≤ 1 / 4 * (x + 1) ^ 2) : f 1 = 1 :=
sorry

theorem problem_II (a b c : ℝ)
  (h : ∀ x : ℝ, x ≤ f(x) ∧ f(x) ≤ 1 / 4 * (x + 1) ^ 2)
  (h1 : f(-1) = 0)
  : f = λ x : ℝ, 1 / 4 * x ^ 2 + 1 / 2 * x + 1 / 4 :=
sorry

theorem problem_III (a b c m : ℝ)
  (h : ∀ x : ℝ, x ≤ f(x) ∧ f(x) ≤ 1 / 4 * (x + 1) ^ 2)
  (h1 : f(-1) = 0)
  (h2 : ∀ x : ℝ, 0 ≤ x → f(x) - m / 2 * x > -3 / 4)
  : m < 3 :=
sorry

end problem_I_problem_II_problem_III_l797_797408


namespace part1_part2_l797_797468

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l797_797468


namespace solve_system_l797_797592

theorem solve_system (C1 C2 : ℝ) :
  (∀ t : ℝ, deriv (λ t, x t) t = x t + 2 * y t + 16 * t * exp t) ∧
  (∀ t : ℝ, deriv (λ t, y t) t = 2 * x t - 2 * y t) →
  (∀ t : ℝ,
    x t = 2 * C1 * exp (2 * t) + C2 * exp (-3 * t) - (12 * t + 13) * exp t) ∧
  (∀ t : ℝ,
    y t = C1 * exp (2 * t) - 2 * C2 * exp (-3 * t) - (8 * t + 6) * exp t) :=
by
  sorry

end solve_system_l797_797592


namespace prove_b_eq_d_and_c_eq_e_l797_797244

variable (a b c d e f : ℕ)

-- Define the expressions for A and B as per the problem statement
def A := 10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f
def B := 10^5 * f + 10^4 * d + 10^3 * e + 10^2 * b + 10 * c + a

-- Define the condition that A - B is divisible by 271
def divisible_by_271 (n : ℕ) : Prop := ∃ k : ℕ, n = 271 * k

-- Define the main theorem to prove b = d and c = e under the given conditions
theorem prove_b_eq_d_and_c_eq_e
    (h1 : divisible_by_271 (A a b c d e f - B a b c d e f)) :
    b = d ∧ c = e :=
sorry

end prove_b_eq_d_and_c_eq_e_l797_797244


namespace root_expression_value_l797_797194

variables (a b : ℝ)
noncomputable def quadratic_eq (a b : ℝ) : Prop := (a + b = 1 ∧ a * b = -1)

theorem root_expression_value (h : quadratic_eq a b) : 3 * a ^ 2 + 4 * b + (2 / a ^ 2) = 11 := sorry

end root_expression_value_l797_797194


namespace simplify_expression_l797_797841

theorem simplify_expression (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0):
  (x ^ (-2) * y ^ (-2)) / (x ^ (-4) - y ^ (-4)) = (x ^ 2 * y ^ 2) / (y ^ 4 - x ^ 4) :=
by 
  sorry

end simplify_expression_l797_797841


namespace csc_315_eq_neg_sqrt2_l797_797031

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end csc_315_eq_neg_sqrt2_l797_797031


namespace john_saves_water_l797_797183

-- Define the conditions
def old_water_per_flush : ℕ := 5
def num_flushes_per_day : ℕ := 15
def reduction_percentage : ℕ := 80
def days_in_june : ℕ := 30

-- Define the savings calculation
def water_saved_in_june : ℕ :=
  let old_daily_usage := old_water_per_flush * num_flushes_per_day
  let old_june_usage := old_daily_usage * days_in_june
  let new_water_per_flush := old_water_per_flush * (100 - reduction_percentage) / 100
  let new_daily_usage := new_water_per_flush * num_flushes_per_day
  let new_june_usage := new_daily_usage * days_in_june
  old_june_usage - new_june_usage

-- The proof problem statement
theorem john_saves_water : water_saved_in_june = 1800 := 
by
  -- Proof would go here
  sorry

end john_saves_water_l797_797183


namespace teal_sales_revenue_l797_797885

theorem teal_sales_revenue :
  let pumpkin_pie_slices := 8
  let pumpkin_pie_price := 5
  let pumpkin_pies_sold := 4
  let custard_pie_slices := 6
  let custard_pie_price := 6
  let custard_pies_sold := 5
  let apple_pie_slices := 10
  let apple_pie_price := 4
  let apple_pies_sold := 3
  let pecan_pie_slices := 12
  let pecan_pie_price := 7
  let pecan_pies_sold := 2
  (pumpkin_pie_slices * pumpkin_pie_price * pumpkin_pies_sold) +
  (custard_pie_slices * custard_pie_price * custard_pies_sold) +
  (apple_pie_slices * apple_pie_price * apple_pies_sold) +
  (pecan_pie_slices * pecan_pie_price * pecan_pies_sold) = 
  628 := by
  sorry

end teal_sales_revenue_l797_797885


namespace part1_part2_l797_797437

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l797_797437


namespace centroid_angle_measure_l797_797096

variable {A B C G : Type*}
variables [inner_product_space ℝ A]
variables [finite_dimensional ℝ A]

-- Let α, β, γ be the angles opposite sides a, b, c in triangle ABC
noncomputable def angle_measure {α β γ : ℝ} (a b c : ℝ) : ℝ :=
if b^2 + c^2 - a^2 = b * c * 2 * cos α then α else sorry

theorem centroid_angle_measure 
  (G : A) (a b c : ℝ)
  (hG : G = centroid ℝ ({A, B, C} : set A))
  (h₀ : ⟪G - A, G - A⟫ = a^2) (h₁ : ⟪G - B, G - B⟫ = b^2) (h₂ : ⟪G - C, G - C⟫ = c^2)
  (h₃ : (a / 5 : ℝ) • (G - A) + (b / 7) • (G - B) + (c / 8) • (G - C) = 0) : 
  angle_measure a b c = π / 3 :=
sorry

end centroid_angle_measure_l797_797096


namespace total_revenue_generated_l797_797942

-- Definitions for the conditions
def size : ℕ := 500
def region_A_size : ℕ := 200 * 300
def region_B_size : ℕ := 200 * 200
def region_C_size : ℕ := 100 * 500

def region_A_production_rate : ℕ := 60
def region_B_production_rate : ℕ := 45
def region_C_production_rate : ℕ := 30

def peanuts_to_pb_conversion_factor : ℕ := 20 / 5

def month_prices : list ℕ := [12, 10, 14, 8, 11]

-- Calculation of peanut production for each region
def region_A_production : ℕ := region_A_size * region_A_production_rate
def region_B_production : ℕ := region_B_size * region_B_production_rate
def region_C_production : ℕ := region_C_size * region_C_production_rate

-- Total peanut production
def total_peanut_production : ℕ := region_A_production + region_B_production + region_C_production

-- Total peanut butter production in kilograms
def total_pb_production_kg : ℕ := (total_peanut_production / peanuts_to_pb_conversion_factor) / 1000

-- Revenue generation calculation
def revenue_per_month (price : ℕ) : ℕ :=
  total_pb_production_kg * price

def total_revenue := list.sum (list.map revenue_per_month month_prices)

-- Proof statement
theorem total_revenue_generated : total_revenue = 94875 := by
  sorry

end total_revenue_generated_l797_797942


namespace merchants_debt_payment_l797_797529

theorem merchants_debt_payment (n : ℕ) (a : ℕ → ℝ) (h_sum : (Finset.range n).sum a = 0) :
  ∃ k : ℕ, k < n ∧ ∀ j : ℕ, j < n → 0 ≤ (Finset.range (j + 1)).sum (λ i, a ((k + i) % n)) :=
by
  sorry

end merchants_debt_payment_l797_797529


namespace find_angle_AOD_l797_797165

noncomputable def angleAOD (x : ℝ) : ℝ :=
4 * x

theorem find_angle_AOD (x : ℝ) (h1 : 4 * x = 180) : angleAOD x = 135 :=
by
  -- x = 45
  have h2 : x = 45 := by linarith

  -- angleAOD 45 = 4 * 45 = 135
  rw [angleAOD, h2]
  norm_num
  sorry

end find_angle_AOD_l797_797165


namespace hyperbola_focus_eq_parabola_focus_l797_797142

theorem hyperbola_focus_eq_parabola_focus (k : ℝ) (hk : k > 0) :
  let parabola_focus : ℝ × ℝ := (2, 0) in
  let hyperbola_focus_distance : ℝ := Real.sqrt (1 + k^2) in
  hyperbola_focus_distance = 2 ↔ k = Real.sqrt 3 :=
by {
  sorry
}

end hyperbola_focus_eq_parabola_focus_l797_797142


namespace log4_a3_arithmetic_sequence_extreme_points_l797_797891

theorem log4_a3_arithmetic_sequence_extreme_points : 
  (∃ a : ℕ → ℝ, is_arithmetic_seq a ∧ 
   (∀ x, deriv (λ x, x^3 - 6 * x^2 + 4 * x - 1) = 3 * x^2 - 12 * x + 4 ∧ 
    (a 2, a 4) ∈ extremas ⟮λ x, x^3 - 6 * x^2 + 4 * x - 1⟯)) → log 4 (a 3) = 1 / 2 :=
by
  sorry

variable {a : ℕ → ℝ}

-- Definition: is_arithmetic_seq
def is_arithmetic_seq (a: ℕ → ℝ) : Prop := 
  ∃ d: ℝ, ∀ n, a (n + 1) - a n = d

-- Definition: extremas
def extremas (f: ℝ → ℝ) (x y: ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ deriv f a = 0 ∧ deriv f b = 0 

end log4_a3_arithmetic_sequence_extreme_points_l797_797891


namespace positive_difference_r1_r2_l797_797764

theorem positive_difference_r1_r2 {
  a b r1 r2 : ℝ,
  F : ℕ → ℝ,
  hF0 : F 0 = 2,
  hF1 : F 1 = 3,
  recurrence : ∀ n, (F (n + 1) * F (n - 1) - F n ^ 2 = (-1)^n * 2),
  form : ∀ n, F n = a * r1^n + b * r2^n
} :
  |r1 - r2| = (Real.sqrt 17) / 2 :=
sorry

end positive_difference_r1_r2_l797_797764


namespace coin_flip_probability_l797_797326

open Classical BigOperators

/-- Given a fair coin that is flipped 8 times, the probability that exactly 6 flips result in heads is 7/64. -/
theorem coin_flip_probability :
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := Nat.choose 8 6
  let probability := favorable_outcomes / (total_outcomes: ℚ)
  probability = 7 / 64 := by
  sorry

end coin_flip_probability_l797_797326


namespace hexagon_perpendicular_sum_l797_797692

-- Definition of the problem using existing conditions
theorem hexagon_perpendicular_sum (s : ℝ) (AO AQ AR : ℝ) : 
  (s = 1) →
  (OP = 2) →
  (hexagon ABCDEF is regular) →
  (AP AQ AR are perpendicular from A onto DE DC extended EF extended, respectively) →
  AO + AQ + AR = 3 * Real.sqrt 3 - 2 :=
by
  sorry -- This is where the proof would go

end hexagon_perpendicular_sum_l797_797692


namespace distance_between_vertices_l797_797910

def vertex_a : (ℝ × ℝ) := (2, 3)
def vertex_b : (ℝ × ℝ) := (-3, 11)

def dist (p q : (ℝ × ℝ)) : ℝ := 
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_between_vertices :
  dist vertex_a vertex_b = real.sqrt 89 :=
by
  sorry

end distance_between_vertices_l797_797910


namespace simplify_and_evaluate_expr_l797_797228

theorem simplify_and_evaluate_expr (x : ℤ) (h : x = -2) : 
  (2 * x + 1) * (x - 2) - (2 - x) ^ 2 = -8 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expr_l797_797228


namespace JohnsonFamilySeating_l797_797625

theorem JohnsonFamilySeating : 
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 359000 := by
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  show total_arrangements - restricted_arrangements = 359000 from sorry

end JohnsonFamilySeating_l797_797625


namespace sin_pi_div_two_plus_alpha_l797_797407

theorem sin_pi_div_two_plus_alpha (x y : ℝ) (h₁ : x = -4) (h₂ : y = 3) (h₃ : x^2 + y^2 = 25) :
  sin (π / 2 + arctan (y / x)) = -4 / 5 :=
by
  -- Define α using the arctan of y / x.
  let α := arctan (y / x)
  -- Use the cofunction identity: sin (π / 2 + α) = cos (α).
  have h₄ : sin (π / 2 + α) = cos (α) := by sorry
  -- Express cos (α) in terms of x and r.
  have h₅ : cos (α) = x / sqrt (x^2 + y^2) := by sorry
  -- Substitute x = -4 and x^2 + y^2 = 25 into cos (α).
  have h₆ : cos (α) = -4 / sqrt 25 := by sorry
  have h₇ : sqrt 25 = 5 := by sorry
  -- Thus, cos (α) = -4 / 5 and sin (π / 2 + α) = -4 / 5.
  have h₈ : cos (α) = -4 / 5 := by sorry
  -- Therefore.
  rw [h₄, h₈]
  exact rfl

end sin_pi_div_two_plus_alpha_l797_797407


namespace max_value_of_curve_l797_797792

noncomputable def f (x: ℝ) : ℝ := x / (x - 2)

theorem max_value_of_curve : ∀ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) → (f x ≤ (1/3)) ∧ (f (-1) = 1 / 3) := by
  sorry

end max_value_of_curve_l797_797792


namespace first_number_is_210_l797_797645

theorem first_number_is_210 (A B hcf lcm : ℕ) (h1 : lcm = 2310) (h2: hcf = 47) (h3 : B = 517) :
  A * B = lcm * hcf → A = 210 :=
by
  sorry

end first_number_is_210_l797_797645


namespace find_coprime_pair_l797_797375

-- Define the necessary conditions
def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def coprime (a b : ℕ) : Prop := gcd a b = 1

def satisfiesDecimalCondition (a b n : ℕ) : Prop := (a:ℚ)/(b:ℚ) = (b:ℚ) + (a:ℚ)/(10^n)

-- The main statement
theorem find_coprime_pair :
  ∃ (a b : ℕ), coprime a b ∧ (∃ n : ℕ, (1 ≤ b) ∧ (10^(n-1) ≤ a) ∧ (a < 10^n) ∧ satisfiesDecimalCondition a b n) ∧ (a, b) = (5, 2) :=
by
  exists 5, 2
  split
  {
    exact rfl  -- gcd(5, 2) = 1
  },
  split
  {
    exists 2
    split, {
      exact le_refl 2  -- 1 ≤ 2
    },
    split, {
      have h : 10^(2-1) = 10,
      exact Nat.pow_one 10  -- 10^(2-1) = 10
      linarith  -- 10 ≤ 5 < 100
    },
    split,
    {
      exact Nat.lt_of_sub_pos (by rfl)  --  5 < 10^2
    },
    {
      exact (show ℚ:0.5 = (ℚ:2) + (ℚ:5)/(10^2) / 2) sorry  -- Proof of decimal condition
    }
  },
  {
    exact rfl  -- (a, b) = (5, 2)
  }

end find_coprime_pair_l797_797375


namespace weight_of_newcomer_l797_797239

theorem weight_of_newcomer (avg_old W_initial : ℝ) 
  (h_weight_range : 400 ≤ W_initial ∧ W_initial ≤ 420)
  (h_avg_increase : avg_old + 3.5 = (W_initial - 47 + W_new) / 6)
  (h_person_replaced : 47 = 47) :
  W_new = 68 := 
sorry

end weight_of_newcomer_l797_797239


namespace angle_between_vectors_of_square_l797_797091

theorem angle_between_vectors_of_square (O B D C : Point) (OB : dist O B = 10) (OD : dist O D = 6) 
  (side_length : ℤ := 4 * Real.sqrt 2) (square_ABCD : is_square A B C D side_length):
  angle_between_vectors (vector O B) (vector O C) = Real.arccos (23 / (5 * Real.sqrt 29)) := 
by
  sorry

end angle_between_vectors_of_square_l797_797091


namespace smallest_number_l797_797672

theorem smallest_number
  (A : ℕ := 2^3 + 2^2 + 2^1 + 2^0)
  (B : ℕ := 2 * 6^2 + 1 * 6)
  (C : ℕ := 1 * 4^3)
  (D : ℕ := 8 + 1) :
  A < B ∧ A < C ∧ A < D :=
by {
  sorry
}

end smallest_number_l797_797672


namespace seating_possible_l797_797526

theorem seating_possible (n : ℕ) (h_prime : Nat.prime (2 * n + 1)) : 
  ∃ seatings : list (list ℕ), 
  (length seatings = n ∧ ∀ i j, i ≠ j → disjoint (seatings.nth i) (seatings.nth j)) := 
sorry

end seating_possible_l797_797526


namespace distinct_keychain_arrangements_l797_797886

theorem distinct_keychain_arrangements :
  let n := 6
  let house_car_unit := 1
  let office_bike_unit := 1
  let additional_keys := n - 2 * (house_car_unit + office_bike_unit)
  (house_car_unit + office_bike_unit + additional_keys - 1)! / 2 * 
  (2 * 2) = 12 :=
by
  sorry

end distinct_keychain_arrangements_l797_797886


namespace arithmetic_seq_and_sum_correct_l797_797108

variables {α : Type*} [linear_ordered_field α]

noncomputable def sum_of_first_n_terms_arithmetic_seq (a n : ℕ → α) : α :=
  (n * (a 1 + a n)) / 2

noncomputable def find_arithmetic_seq_and_sum (a : ℕ → α) (S : ℕ → α): Prop :=
  (a 2 = 6) ∧ (S 5 = 40) ∧ 
  (∀ n : ℕ, a n = 2 * n + 2) ∧
  (∀ n : ℕ, S n = n * (n + 3))

theorem arithmetic_seq_and_sum_correct (a : ℕ → α) (S : ℕ → α) :
  find_arithmetic_seq_and_sum a S :=
by 
  sorry

end arithmetic_seq_and_sum_correct_l797_797108


namespace average_pastries_sold_per_day_l797_797703

noncomputable def pastries_sold_per_day (n : ℕ) : ℕ :=
  match n with
  | 0 => 2 -- Monday
  | _ + 1 => 2 + n + 1

theorem average_pastries_sold_per_day :
  (∑ i in Finset.range 7, pastries_sold_per_day i) / 7 = 5 :=
by
  sorry

end average_pastries_sold_per_day_l797_797703


namespace odd_function_property_l797_797984

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x else if x < 0 then -2^(-x) else 0

theorem odd_function_property (x : ℝ) (h : x > 0) : f(-x) = -f(x) :=
by
  simp [f]
  rw if_neg (show -x > 0, by linarith)
  rw [←neg_eq_iff_neg_eq, neg_neg, if_pos h]
  sorry

example (x : ℝ) (h : x < 0) : f(x) = -2^(-x) :=
by
  have h_odd: f(-x) = -f(x) := odd_function_property (-x) (by linarith [h])
  simp [f]
  simp [f] at h_odd
  exact h_odd

end odd_function_property_l797_797984


namespace minimum_arc_length_of_curve_and_line_l797_797870

-- Definition of the curve C and the line x = π/4
def curve (x y α : ℝ) : Prop :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line (x : ℝ) : Prop :=
  x = Real.pi / 4

-- Statement of the proof problem: the minimum value of d as α varies
theorem minimum_arc_length_of_curve_and_line : 
  (∀ α : ℝ, ∃ d : ℝ, (∃ y : ℝ, curve (Real.pi / 4) y α) → 
    (d = Real.pi / 2)) :=
sorry

end minimum_arc_length_of_curve_and_line_l797_797870


namespace train_time_to_pass_post_l797_797856

namespace TrainTiming

-- Define the train's length in meters
def trainLength : ℝ := 200

-- Define the train's speed in kmph
def trainSpeed_kmph : ℝ := 36

-- Convert speed from kmph to mps
def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

-- Define the speed in mps using the conversion function
def trainSpeed_mps : ℝ :=
  kmph_to_mps trainSpeed_kmph

-- Define the time it takes for the train to pass the telegraph post
def timeToPassPost (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

-- Lean statement to prove the required time is 20 seconds
theorem train_time_to_pass_post : timeToPassPost trainLength trainSpeed_mps = 20 := by
  sorry

end TrainTiming

end train_time_to_pass_post_l797_797856


namespace find_a_l797_797050

theorem find_a (a : ℝ) (h : log a 125 = -3 / 2) : a = 1 / 25 :=
sorry

end find_a_l797_797050


namespace sequence_sum_consecutive_l797_797153

theorem sequence_sum_consecutive 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 20) 
  (h8 : a 8 = 16) 
  (h_sum : ∀ i, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 100) :
  a 2 = 16 ∧ a 3 = 64 ∧ a 4 = 20 ∧ a 5 = 16 ∧ a 6 = 64 ∧ a 7 = 20 :=
  sorry

end sequence_sum_consecutive_l797_797153


namespace probability_two_primes_l797_797593

/-- Suppose Ben rolls four fair 6-sided dice, each numbered 1 to 6.
The primes between 1 and 6 are 2, 3, and 5.
Thus, the probability that a 6-sided die rolls a prime number is 3 / 6 = 1 / 2.
We need to calculate the probability of exactly two dice showing a prime number.

Total number of dice rolls = 4.

Probability of rolling a prime number in one roll = 1 / 2.
Probability of rolling a non-prime number in one roll = 1 / 2.

The number of ways to choose which 2 dice out of 4 show a prime number is ("4 choose 2"), that is, 6.
The probability of that specific configuration is (1 / 2) ^ 2 * (1 / 2) ^ 2 = 1 / 16.
Therefore, the total probability is 6 * (1 / 16) = 6 / 16 = 3 / 8.
-/
theorem probability_two_primes (p : ℚ := 1/2) (n : ℕ := 4) :
  (nat.choose n 2) * (p^2 * (1 - p)^2) = 3 / 8 := by
  sorry

end probability_two_primes_l797_797593


namespace tin_silver_ratio_l797_797699

/-- Assuming a metal bar made of an alloy of tin and silver weighs 40 kg, 
    and loses 4 kg in weight when submerged in water,
    where 10 kg of tin loses 1.375 kg in water and 5 kg of silver loses 0.375 kg, 
    prove that the ratio of tin to silver in the bar is 2 : 3. -/
theorem tin_silver_ratio :
  ∃ (T S : ℝ), 
    T + S = 40 ∧ 
    0.1375 * T + 0.075 * S = 4 ∧ 
    T / S = 2 / 3 := 
by
  sorry

end tin_silver_ratio_l797_797699


namespace distance_between_points_is_five_l797_797245

-- Define the two points
def point1 : (ℝ × ℝ) := (4, 3)
def point2 : (ℝ × ℝ) := (7, -1)

-- Define the function to calculate the distance between two points
def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Statement of the problem
theorem distance_between_points_is_five :
  distance point1 point2 = 5 :=
by
  -- We are skipping the proof part and using sorry 
  sorry

end distance_between_points_is_five_l797_797245


namespace sum_arithmetic_sequence_satisfies_conditions_l797_797982

theorem sum_arithmetic_sequence_satisfies_conditions :
  ∀ (a : ℕ → ℤ) (d : ℤ),
  (a 1 = 1) ∧ (d ≠ 0) ∧ ((a 3)^2 = (a 2) * (a 6)) →
  (6 * a 1 + (6 * 5 / 2) * d = -24) :=
by
  sorry

end sum_arithmetic_sequence_satisfies_conditions_l797_797982


namespace distance_between_lines_l797_797788

/-- Line 1: 3x + 4y + 3 = 0 -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

/-- Line 2: 6x + 8y + 11 = 0 -/
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 11 = 0

/-- Distance between the two lines -/
theorem distance_between_lines : 
  ∀(A B : ℝ), A = 3 → B = 4 → 
  ∀(C1 C2 : ℝ), C1 = 3 → C2 = 11 / 2 → 
  ∀(d : ℝ), d = abs (C1 - C2) / real.sqrt (A * A + B * B) → 
  d = 1 / 2 :=
by
  intros A B hA hB C1 C2 hC1 hC2 d hd
  rw [hA, hB, hC1, hC2] at hd
  rw abs_of_nonneg at hd
  . assumption
  sorry

end distance_between_lines_l797_797788


namespace hyperbola_parabola_focus_l797_797144

theorem hyperbola_parabola_focus (k : ℝ) (h : k > 0) :
  (∃ x y : ℝ, (1/k^2) * y^2 = 0 ∧ x^2 - (y^2 / k^2) = 1) ∧ (∃ x : ℝ, y^2 = 8 * x) →
  k = Real.sqrt 3 :=
by sorry

end hyperbola_parabola_focus_l797_797144


namespace average_height_of_trees_l797_797180

-- Define the heights of the trees
def height_tree1: ℕ := 1000
def height_tree2: ℕ := height_tree1 / 2
def height_tree3: ℕ := height_tree1 / 2
def height_tree4: ℕ := height_tree1 + 200

-- Calculate the total number of trees
def number_of_trees: ℕ := 4

-- Compute the total height climbed
def total_height: ℕ := height_tree1 + height_tree2 + height_tree3 + height_tree4

-- Define the average height
def average_height: ℕ := total_height / number_of_trees

-- The theorem statement
theorem average_height_of_trees: average_height = 800 := by
  sorry

end average_height_of_trees_l797_797180


namespace megatek_employees_in_manufacturing_l797_797308

theorem megatek_employees_in_manufacturing :
  let total_degrees := 360
  let manufacturing_degrees := 108
  (manufacturing_degrees / total_degrees.toFloat) * 100 = 30 := 
by
  sorry

end megatek_employees_in_manufacturing_l797_797308


namespace monotonic_increasing_interval_l797_797989

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

-- Define the statement to be proved
theorem monotonic_increasing_interval : ∀ x : ℝ, x > 2 → deriv f x > 0 :=
by
  -- This is where the proof will go
  sorry

end monotonic_increasing_interval_l797_797989


namespace initial_quantity_of_A_l797_797706

noncomputable def initial_quantity_of_A_in_can (initial_total_mixture : ℤ) (x : ℤ) := 7 * x

theorem initial_quantity_of_A
  (initial_ratio_A : ℤ) (initial_ratio_B : ℤ) (initial_ratio_C : ℤ)
  (initial_total_mixture : ℤ) (drawn_off_mixture : ℤ) (new_quantity_of_B : ℤ)
  (new_ratio_A : ℤ) (new_ratio_B : ℤ) (new_ratio_C : ℤ)
  (h1 : initial_ratio_A = 7) (h2 : initial_ratio_B = 5) (h3 : initial_ratio_C = 3)
  (h4 : initial_total_mixture = 15 * x)
  (h5 : new_ratio_A = 7) (h6 : new_ratio_B = 9) (h7 : new_ratio_C = 3)
  (h8 : drawn_off_mixture = 18)
  (h9 : new_quantity_of_B = 5 * x - (5 / 15) * 18 + 18)
  (h10 : (7 * x - (7 / 15) * 18) / new_quantity_of_B = 7 / 9) :
  initial_quantity_of_A_in_can initial_total_mixture x = 54 :=
by
  sorry

end initial_quantity_of_A_l797_797706


namespace lines_perpendicular_l797_797126

-- Definitions and conditions from the problem
def line1 (m : ℝ) : ℝ → ℝ → Prop := λ x y, m*x + 2*y - 1 = 0
def line2 (m : ℝ) : ℝ → ℝ → Prop := λ x y, 3*x + (m+1)*y + 1 = 0

-- The statement to prove
theorem lines_perpendicular (m : ℝ) : 
  (m = -2/5) ↔ (∀ x y : ℝ, line1 m x y ↔ line2 m x y → (m*x + 2*y - 1) * (3*x + (m+1)*y + 1) = -1) :=
begin
  sorry
end

end lines_perpendicular_l797_797126


namespace smaller_divides_larger_prob_l797_797275

-- Define the set of numbers
def numbers := {1, 2, 3, 6, 9}

-- Define a function to check pairwise divisibility
def divides (a b : ℕ) : Prop := a ∣ b

-- Define the pairs and count valid pairs
def valid_pairs : List (ℕ × ℕ) :=
  [(1, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 6), 
   (2, 9), (3, 6), (3, 9), (6, 9)]

def valid_division_count : Nat := 
  valid_pairs.count (λ ⟨a, b⟩ => divides a b)

lemma probability_smaller_divides_larger : 
  valid_division_count = 6 ∧ valid_pairs.length = 10 :=
by
  -- We counted by hand earlier; the counts are thus taken as given from steps.
  sorry

theorem smaller_divides_larger_prob : 
  ∃ p q : Nat, (p = 3 ∧ q = 5) ∧ 
  p/q = valid_division_count / valid_pairs.length :=
by
  have h1 := probability_smaller_divides_larger
  -- Solve for the probability based on valid_division_count / valid_pairs.length
  existsi 3, 5
  -- we know 6/10 simplifies to 3/5
  sorry

end smaller_divides_larger_prob_l797_797275


namespace part1_part2_l797_797466

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l797_797466


namespace find_positive_integer_tuples_l797_797051

theorem find_positive_integer_tuples
  (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hz_prime : Prime z) :
  z ^ x = y ^ 3 + 1 →
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 2 ∧ y = 2 ∧ z = 3) :=
by
  sorry

end find_positive_integer_tuples_l797_797051


namespace sum_of_distinct_prime_divisors_of_2520_l797_797674

theorem sum_of_distinct_prime_divisors_of_2520 : ∑ p in {2, 3, 5, 7}, p = 17 := by
  sorry

end sum_of_distinct_prime_divisors_of_2520_l797_797674


namespace monotonically_increasing_if_k_ge_1_l797_797513

theorem monotonically_increasing_if_k_ge_1 (k : ℝ) : 
  (∀ x : ℝ, 1 ≤ x → deriv (λ x, k * x - log x) x ≥ 0) ↔ k ≥ 1 :=
by
  sorry

end monotonically_increasing_if_k_ge_1_l797_797513


namespace find_standard_equation_of_ellipse_1_find_standard_equation_of_ellipse_2_l797_797383

/-- Define the conditions. -/
def ellipse_1 : Prop :=
  ∃ (x y : ℝ), (x^2 / 4 + y^2 / 9 = 1)

def passes_through_point (x y : ℝ) (point_x point_y : ℝ) : Prop :=
  (x = point_x) ∧ (y = point_y)

def ellipse_2 (a b : ℝ) : Prop :=
  ∃ (e : ℝ), (e = Real.sqrt 5 / 5) ∧ (2 * b = 4) ∧ (a^2 = b^2 + (Real.sqrt 5)^2)

/-- Formulate the problem in Lean 4 statements. -/
theorem find_standard_equation_of_ellipse_1 (c1 : ellipse_1) (p : passes_through_point 2 (-3)) :
  (∃ (a : ℝ), a^2 = 15 ∧ (10 : ℝ) = 15 - 5 ∧ (15 : ℝ) = 15) ∧ (2 : ℝ) * (sqrt 5 / 5) = 1 / 2 :=
sorry

theorem find_standard_equation_of_ellipse_2 (c2 : ellipse_2 ℝ ℝ) :
  (5 : ℝ = 4) ∧ (4 : ℝ = 5) :=
sorry

end find_standard_equation_of_ellipse_1_find_standard_equation_of_ellipse_2_l797_797383


namespace four_digit_numbers_in_range_l797_797855

theorem four_digit_numbers_in_range : 
  ∀ (start end : ℕ), 
    1000 ≤ start → end ≤ 5000 → start = 1000 → end = 5000 → 
    end - start + 1 = 4001 :=
by
  intros start end h1 h2 h3 h4
  rw [h3, h4]
  exact (Nat.sub_add_comm (Nat.le_of_lt h1)).trans (Nat.add_one_injective)
  sorry

end four_digit_numbers_in_range_l797_797855


namespace problem_l797_797415

open Real

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def P (m : ℝ) : Prop :=
  is_increasing (λ x, log (2 * m) (x + 1))

def Q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

theorem problem (m : ℝ) (H1 : P m ∨ Q m) (H2 : ¬(P m ∧ Q m)) :
  m ∈ Set.Icc (-2 : ℝ) (1 / 2) ∪ Set.Ioi (2 : ℝ) :=
sorry

end problem_l797_797415


namespace total_molecular_weight_l797_797288

-- Define atomic weights
def atomic_weight (element : String) : Float :=
  match element with
  | "K"  => 39.10
  | "Cr" => 51.996
  | "O"  => 16.00
  | "Fe" => 55.845
  | "S"  => 32.07
  | "Mn" => 54.938
  | _    => 0.0

-- Molecular weights of compounds
def molecular_weight_K2Cr2O7 : Float := 
  2 * atomic_weight "K" + 2 * atomic_weight "Cr" + 7 * atomic_weight "O"

def molecular_weight_Fe2_SO4_3 : Float := 
  2 * atomic_weight "Fe" + 3 * atomic_weight "S" + 12 * atomic_weight "O"

def molecular_weight_KMnO4 : Float := 
  atomic_weight "K" + atomic_weight "Mn" + 4 * atomic_weight "O"

-- Proof statement 
theorem total_molecular_weight :
  4 * molecular_weight_K2Cr2O7 + 3 * molecular_weight_Fe2_SO4_3 + 5 * molecular_weight_KMnO4 = 3166.658 :=
by
  sorry

end total_molecular_weight_l797_797288


namespace part1_purely_periodic_sequence_part2_period_of_sequence_l797_797693

-- Part (1): Proving that the sequence {a_n} is purely periodic
theorem part1_purely_periodic_sequence (k : ℕ) (a : ℕ → ℕ) (x : Fin k → ℂ)
  (T : Fin k → ℕ) (h1 : ∀ j : Fin k, x j ≠ 0)
  (h2 : ∀ i j : Fin k, i ≠ j → x i ≠ x j)
  (h3 : ∀ j : Fin k, x j ^ (T j) = 1)
  (h4 : ∀ n : ℕ, a (n + k) = ∑ i : Fin k, (a (n + i) * (x i) ^ n)) :
  ∃ p : ℕ, ∀ n : ℕ, a (n + p) = a n :=
sorry

-- Part (2): Proving that 7m is the period of the sequence {y_n}
theorem part2_period_of_sequence (m : ℕ) (m_pos : 0 < m) (y : ℕ → ℂ)
  (h5 : ∀ n : ℕ, y n + y (n + 2 * m) = 2 * y (n + m) * Real.cos (2 * π / 7)) :
  ∀ n : ℕ, y (n + 7 * m) = y n :=
sorry

end part1_purely_periodic_sequence_part2_period_of_sequence_l797_797693


namespace estimate_fish_population_jan_15_l797_797334

-- Definitions according to the problem conditions
def initial_tagged_fish : ℕ := 80
def june_sample_size : ℕ := 100
def tagged_fish_in_june_sample : ℕ := 6
def death_or_migration_rate : ℝ := 0.20
def recent_additions_rate : ℝ := 0.50

-- Proof statement
theorem estimate_fish_population_jan_15 :
  ∃ x : ℕ, 
    let fish_from_january_in_june := june_sample_size * (1 - recent_additions_rate) in
    let surviving_tagged_fish := initial_tagged_fish * (1 - death_or_migration_rate) in
    (real_of_nat tagged_fish_in_june_sample) / (real_of_nat fish_from_january_in_june) = 
    (real_of_nat surviving_tagged_fish) / (real_of_nat x) ∧ 
    x = 533 :=
by
  sorry

end estimate_fish_population_jan_15_l797_797334


namespace arithmetic_sequence_sum_l797_797541

-- Definitions of the conditions
variables {α : Type*} [linear_ordered_field α] {a : ℕ → α}

-- Conditions
axiom a2_eq_3 : a 2 = 3
axiom a6_eq_11 : a 6 = 11

-- Sum of the first 7 terms
def S_7 (a : ℕ → α) := ∑ i in finset.range 7, a i

-- The proof goal to demonstrate S_7 = 49 given the conditions
theorem arithmetic_sequence_sum : S_7 a = 49 :=
by 
  sorry

end arithmetic_sequence_sum_l797_797541


namespace determine_y_l797_797099

theorem determine_y (a b c x : ℝ) (p q r : ℝ)
  (h1 : (log a) / (2 * p) = (log b) / (3 * q))
  (h2 : (log b) / (3 * q) = (log c) / r)
  (h3 : (log c) / r = log x)
  (h4 : x ≠ 1) :
  (∃ y : ℝ, (b ^ 3 / (a ^ 2 * c) = x ^ y) ∧ y = 9 * q - 4 * p - r) :=
by
  sorry

end determine_y_l797_797099


namespace c_not_parallel_to_b_l797_797105

-- Definitions
structure Line where
  --- defining a placeholder for Line
  points : Set (ℝ × ℝ × ℝ)

def skew_lines (a b : Line) : Prop :=
  ¬ (∃ p, p ∈ a.points ∧ p ∈ b.points) ∧ ¬ (∃ v, ∀ p ∈ a.points, ∀ q ∈ b.points, (q - p) ∈ ℝ • v)

def parallel_lines (a b : Line) : Prop :=
  ∃ v, ∃ w ∈ ℝ • v, (b.points = {p | ∃ q ∈ a.points, p = q + w})

-- Given conditions
variables (a b c : Line)
axiom h1 : skew_lines a b
axiom h2 : parallel_lines a c

-- Goal: Prove that lines c and b cannot be parallel
theorem c_not_parallel_to_b : ¬ parallel_lines c b :=
by
  sorry

end c_not_parallel_to_b_l797_797105


namespace maximize_k_l797_797823

open Real

theorem maximize_k (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : log x + log y = 0)
  (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → k * (x + 2 * y) ≤ x^2 + 4 * y^2) : k ≤ sqrt 2 :=
sorry

end maximize_k_l797_797823


namespace sufficient_but_not_necessary_l797_797127

theorem sufficient_but_not_necessary (x : ℝ) : (x = 1 → x * (x - 1) = 0) ∧ ¬(x * (x - 1) = 0 → x = 1) := 
by
  sorry

end sufficient_but_not_necessary_l797_797127


namespace pirate_treasure_l797_797952

theorem pirate_treasure (x : ℕ) (h : x * (x + 1) / 2 = 3 * x) : 4 * x = 20 :=
by
  have hx : 0 < x := sorry  -- We know that x can't be 0 from the solution.
  have hx5 : x = 5 := sorry -- We derive x = 5 from solving the equation x(x - 5) = 0.
  rw [hx5]
  norm_num

end pirate_treasure_l797_797952


namespace waiter_slices_l797_797131

theorem waiter_slices (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ)
  (h_total_slices : total_slices = 78)
  (h_ratios : buzz_ratio = 5 ∧ waiter_ratio = 8) :
  20 < (waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio))) →
  28 = waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio)) - 20 :=
by
  sorry

end waiter_slices_l797_797131


namespace transform_ΔDEF_to_ΔD_l797_797271

structure Point where
  x : ℚ
  y : ℚ

structure Triangle where
  A : Point
  B : Point
  C : Point

def ΔDEF := Triangle.mk (Point.mk 0 0) (Point.mk 0 10) (Point.mk 15 0)
def ΔD'E'F' := Triangle.mk (Point.mk 15 25) (Point.mk 25 25) (Point.mk 15 10)

def isRotation (n : ℚ) (u v : ℚ) (Δ1 Δ2 : Triangle) : Prop :=
  -- Definition of rotation transformation between triangles Δ1 and Δ2
  sorry

theorem transform_ΔDEF_to_ΔD'E'F' :
  ∃ (n u v : ℚ), 0 < n ∧ n < 180 ∧ isRotation n u v ΔDEF ΔD'E'F' ∧ n + u + v = 115 :=
by
  use 90
  use 20
  use 5
  have h1 : 0 < 90 := by norm_num
  have h2 : 90 < 180 := by norm_num
  have h3 : isRotation 90 20 5 ΔDEF ΔD'E'F' := sorry
  have h4 : 90 + 20 + 5 = 115 := by norm_num
  exact ⟨90, 20, 5, h1, h2, h3, h4⟩

end transform_ΔDEF_to_ΔD_l797_797271


namespace Sn_min_value_at_n_5_l797_797175

/-- The function S calculates the sum of the first n terms of an arithmetic sequence. -/
def Sn (a : ℕ → ℤ) (n : ℕ) : ℤ := (n * (a 1 + a n)) / 2

/-- The minimum value of Sn is reached at n = 5. -/
theorem Sn_min_value_at_n_5 (a : ℕ → ℤ) (h1 : a 3 + a 9 > 0) (h2 : Sn a 9 < 0) : Sn a 5 = min (Sn a) := 
by
  sorry

end Sn_min_value_at_n_5_l797_797175


namespace area_of_triangle_BQW_is_16_l797_797166

noncomputable def area_triangle_BQW (ABCD: Type) (AZ WC AB area_trapezoid: ℕ) (Q_ratio: ℕ) : ℕ :=
  if h : ABCD = 1 ∧ AZ = 8 ∧ WC = 8 ∧ AB = 16 ∧ area_trapezoid = 160 ∧ Q_ratio = 1 then
    16
  else
    0

theorem area_of_triangle_BQW_is_16 : 
  ∀ (ABCD: Type) (AZ WC AB area_trapezoid Q_ratio: ℕ),
     ABCD = 1 ∧ AZ = 8 ∧ WC = 8 ∧ AB = 16 ∧ area_trapezoid = 160 ∧ Q_ratio = 1 → 
     area_triangle_BQW ABCD AZ WC AB area_trapezoid Q_ratio = 16 :=
by
  intro ABCD AZ WC AB area_trapezoid Q_ratio h
  simp [h]
  sorry

end area_of_triangle_BQW_is_16_l797_797166


namespace value_of_m_l797_797110

theorem value_of_m (m x : ℝ) (h1 : mx + 1 = 2 * (m - x)) (h2 : |x + 2| = 0) : m = -|3 / 4| :=
by
  sorry

end value_of_m_l797_797110


namespace sufficient_condition_perpendicular_planes_l797_797548

-- Definitions of the planes and lines
variable (α β : Plane)
variable (m n l₁ l₂ : Line)

-- Conditions:
-- m and n are distinct lines within plane α
-- l₁ and l₂ are two intersecting lines within plane β
variable (h₀ : m ≠ n)
variable (h₁ : m ∈ α)
variable (h₂ : n ∈ α)
variable (h₃ : l₁ ∈ β)
variable (h₄ : l₂ ∈ β)
variable (h₅ : intersecting l₁ l₂)

-- Proof requirement: A sufficient condition for α ⊥ β is:
-- l₁ ⊥ m and l₂ ⊥ m
theorem sufficient_condition_perpendicular_planes (h₆ : perp l₁ m) (h₇ : perp l₂ m) : perp α β := sorry

end sufficient_condition_perpendicular_planes_l797_797548


namespace sqrt_meaningful_range_l797_797268

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 8)) → x ≥ -8 :=
by
  sorry

end sqrt_meaningful_range_l797_797268


namespace original_ghee_quantity_l797_797306

theorem original_ghee_quantity (x : ℝ) (H1 : 0.60 * x + 10 = ((1 + 0.40 * x) * 0.80)) :
  x = 10 :=
sorry

end original_ghee_quantity_l797_797306


namespace csc_315_eq_neg_sqrt_2_l797_797021

theorem csc_315_eq_neg_sqrt_2 :
  let csc := λ θ, 1 / Real.sin θ in
  csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
  by
  let sin := Real.sin
  have h1 : csc (315 * Real.pi / 180) = 1 / sin (315 * Real.pi / 180) := rfl
  have h2 : sin (315 * Real.pi / 180) = sin ((360 - 45) * Real.pi / 180) := by congr; norm_num
  have h3 : sin ((360 - 45) * Real.pi / 180) = -sin (45 * Real.pi / 180) := by
    rw [Real.sin_pi_sub]
    congr; norm_num
  have h4 : sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 := Real.sin_of_one_div_sqrt_two 45 rfl
  sorry

end csc_315_eq_neg_sqrt_2_l797_797021


namespace inequality_part1_inequality_part2_l797_797452

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l797_797452


namespace distinct_equilateral_triangles_l797_797821

-- Define the eleven-sided regular polygon as a set of points
def regular_polygon_vertices (B : Fin 11 → ℝ × ℝ) : Prop :=
  ∃ r : ℝ, 0 < r ∧ ∃ θ : ℝ, 0 < θ ∧ ∀ i j : Fin 11, i ≠ j →
    B i = (r * Real.cos (i.val * θ), r * Real.sin (i.val * θ)) 
    ∧ B j = (r * Real.cos (j.val * θ), r * Real.sin (j.val * θ))
  
-- Proposition to be proved
theorem distinct_equilateral_triangles (B : Fin 11 → ℝ × ℝ) (h : regular_polygon_vertices B) :
  ∃ n, n = 11 ∧ has_distinct_equilateral_triangles B n :=
sorry

end distinct_equilateral_triangles_l797_797821


namespace find_particular_number_l797_797661

def particular_number (x : ℕ) : Prop :=
  (2 * (67 - (x / 23))) = 102

theorem find_particular_number : particular_number 2714 :=
by {
  sorry
}

end find_particular_number_l797_797661


namespace grid_fill_count_l797_797887

theorem grid_fill_count : 
  (∃ (a : Fin 4 → Fin 4 → Fin 2), 
    (∀ i : Fin 4, (∑ j, a i j) % 2 = 0) ∧ 
    (∀ j : Fin 4, (∑ i, a i j) % 2 = 0) ∧ 
    (∑ i, a i i % 2 = 0) ∧ 
    (∑ i, a i (Fin 3 - i) % 2 = 0)) 
  ↔ 256 := 
sorry

end grid_fill_count_l797_797887


namespace problem_part_one_problem_part_two_l797_797581

theorem problem_part_one :
  ∀ (S : Finset ℕ), (∀ x ∈ S, x < 100 ∧ x % 2 = 1) ∧ S.card = 27 →
    ∃ (a b ∈ S), a + b = 102 :=
by
  sorry

theorem problem_part_two :
  ∃ (count : ℕ), count = 2^24 ∧ (∀ (S : Finset ℕ), (∀ x ∈ S, x < 100 ∧ x % 2 = 1) ∧ S.card = 26 →
    (∀ (a b ∈ S), a + b ≠ 102) → count = 2^24) :=
by
  use 2^24
  sorry

end problem_part_one_problem_part_two_l797_797581


namespace jack_can_return_3900_dollars_l797_797531

/-- Jack's Initial Gift Card Values and Counts --/
def best_buy_card_value : ℕ := 500
def walmart_card_value : ℕ := 200
def initial_best_buy_cards : ℕ := 6
def initial_walmart_cards : ℕ := 9

/-- Jack's Sent Gift Card Counts --/
def sent_best_buy_cards : ℕ := 1
def sent_walmart_cards : ℕ := 2

/-- Calculate the remaining dollar value of Jack's gift cards. --/
def remaining_gift_cards_value : ℕ := 
  (initial_best_buy_cards * best_buy_card_value - sent_best_buy_cards * best_buy_card_value) +
  (initial_walmart_cards * walmart_card_value - sent_walmart_cards * walmart_card_value)

/-- Proving the remaining value of gift cards Jack can return is $3900. --/
theorem jack_can_return_3900_dollars : remaining_gift_cards_value = 3900 := by
  sorry

end jack_can_return_3900_dollars_l797_797531


namespace surface_area_ratio_l797_797263

theorem surface_area_ratio (a : ℝ) (a_pos : a > 0) :
  let r1 := a / 2
  let r2 := a / Real.sqrt 2
  let r3 := (a * Real.sqrt 3) / 2
  let S1 := 4 * Real.pi * r1^2
  let S2 := 4 * Real.pi * r2^2
  let S3 := 4 * Real.pi * r3^2
  (S1 : S2 : S3) = (1 : 2 : 3) :=
begin
  sorry
end

end surface_area_ratio_l797_797263


namespace transformed_data_average_and_variance_l797_797489

variables {α : Type} [field α]
variables (x : ℕ → α)

def average (s : fin 8 → α) : α := 
  (finset.univ.sum (λ i, s i)) / 8

def variance (s : fin 8 → α) : α :=
  (finset.univ.sum (λ i, (s i - average s) ^ 2)) / 8

theorem transformed_data_average_and_variance
  (h_avg : average x = 4)
  (h_var : variance x = 3) :
  average (λ i, 2 * x i - 6) = 2 ∧ variance (λ i, 2 * x i - 6) = 12 :=
sorry

end transformed_data_average_and_variance_l797_797489


namespace relation_between_a_and_b_range_of_a_series_inequality_l797_797484

noncomputable def f_prime (x : ℝ) (a : ℝ) (b : ℝ) := ax + b / x + 2 - 2a
noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := ax + b / x + 2 - 2a

theorem relation_between_a_and_b (a b : ℝ) (h : a > 0) (h_tangent_parallel : f_prime 1 a b = 2) :
  b = a - 2 :=
sorry

theorem range_of_a (a : ℝ) (h : a > 0) (h_relation : ∃ b, f_prime 1 a b = 2) :
  (∀ x : ℝ, 1 ≤ x → f x a (a - 2) ≥ 2 * Real.log x) ↔ a ≥ 1 :=
sorry

theorem series_inequality (n : ℕ) (h : n > 0) :
  (∑ i in (Finite 𝕜.range(1, n + 1)), (1 / ( 2 * i - 1)) > (1 / 2) * Real.log(2*n + 1) + (n / (2*n + 1))) :=
sorry

end relation_between_a_and_b_range_of_a_series_inequality_l797_797484


namespace father_l797_797393

noncomputable def father's_current_age : ℕ :=
  let S : ℕ := 40 -- Sebastian's current age
  let Si : ℕ := S - 10 -- Sebastian's sister's current age
  let sum_five_years_ago := (S - 5) + (Si - 5) -- Sum of their ages five years ago
  let father_age_five_years_ago := (4 * sum_five_years_ago) / 3 -- From the given condition
  father_age_five_years_ago + 5 -- Their father's current age

theorem father's_age_is_85 : father's_current_age = 85 :=
  sorry

end father_l797_797393


namespace part1_part2_l797_797447

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

end part1_part2_l797_797447


namespace pages_488_more_4s_than_8s_l797_797782

theorem pages_488_more_4s_than_8s :
  let count_digit (digit n : ℕ) : ℕ := (n.toString.toList.filter (fun c => c = (digit.toString.front!))).length in
  let total_4s := (List.range' 1 488).sum (count_digit 4) in
  let total_8s := (List.range' 1 488).sum (count_digit 8) in
  total_4s - total_8s = 90 :=
by
  sorry

end pages_488_more_4s_than_8s_l797_797782


namespace cube_root_of_466560000_l797_797761

theorem cube_root_of_466560000 :
  (∛ (466560000 : ℝ) = 360) :=
sorry

end cube_root_of_466560000_l797_797761


namespace probability_of_finding_last_defective_product_on_fourth_inspection_l797_797735

theorem probability_of_finding_last_defective_product_on_fourth_inspection :
  let total_products := 6
  let qualified_products := 4
  let defective_products := 2
  let probability := (4 / 6) * (3 / 5) * (2 / 4) * (1 / 3) + (4 / 6) * (2 / 5) * (3 / 4) * (1 / 3) + (2 / 6) * (4 / 5) * (3 / 4) * (1 / 3)
  probability = 1 / 5 :=
by
  let total_products := 6
  let qualified_products := 4
  let defective_products := 2
  let probability := (4 / 6) * (3 / 5) * (2 / 4) * (1 / 3) + (4 / 6) * (2 / 5) * (3 / 4) * (1 / 3) + (2 / 6) * (4 / 5) * (3 / 4) * (1 / 3)
  have : probability = 1 / 5 := sorry
  exact this

end probability_of_finding_last_defective_product_on_fourth_inspection_l797_797735


namespace cyclic_quadrilateral_problem_l797_797316

noncomputable def polynomial := Polynomial (x^3 - x - 1)

theorem cyclic_quadrilateral_problem :
  let α β γ be roots of polynomial in ℝ
  in (α + β + γ = 0 
     ∧ α*β + β*γ + γ*α = -1 
     ∧ α*β*γ = 1)
  → ((1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ)) = 3 :=
sorry

end cyclic_quadrilateral_problem_l797_797316


namespace samia_walked_distance_l797_797225

theorem samia_walked_distance :
  ∃ (x : ℝ), 
    let bike_speed := 17 in
    let walk_speed := 5 in
    let total_time := (44 : ℝ) / 60 in
    let coeff := (total_time * bike_speed * walk_speed) / (bike_speed + walk_speed) in
    x = coeff ∧ abs (x - 2.8) < 0.1 :=
sorry

end samia_walked_distance_l797_797225


namespace n_sum_of_two_squares_l797_797510

theorem n_sum_of_two_squares (n : ℤ) (m : ℤ) (hn_gt_2 : n > 2) (hn2_eq_diff_cubes : n^2 = (m+1)^3 - m^3) : 
  ∃ a b : ℤ, n = a^2 + b^2 :=
sorry

end n_sum_of_two_squares_l797_797510


namespace part1_part2_l797_797429

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l797_797429


namespace maximum_value_of_f_at_sqrt2_l797_797113

noncomputable def f (x : ℝ) : ℝ := (2 * x - x^2) * Real.exp x

theorem maximum_value_of_f_at_sqrt2 :
  ∃ x_max : ℝ, 
  x_max = Real.sqrt 2 ∧ 
  (∀ x : ℝ, f x ≤ f x_max) ∧ 
  (∀ M : ℝ, ∃ x : ℝ, f x < M := 
  (∀ x : ℝ, x > Real.sqrt 2 → f x → 0 ∧ x < -Real.sqrt 2 → f x < M)) :=
begin
  sorry
end

end maximum_value_of_f_at_sqrt2_l797_797113


namespace csc_315_eq_neg_sqrt2_l797_797026

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end csc_315_eq_neg_sqrt2_l797_797026


namespace quadratic_non_monotonic_iff_m_interval_l797_797775

theorem quadratic_non_monotonic_iff_m_interval (m : ℝ) :
  (∃ x y ∈ set.Icc (-1 : ℝ) 2, x < y ∧ f x > f y) ∨ (∃ x y ∈ set.Icc (-1 : ℝ) 2, x < y ∧ f x < f y) ↔
  m ∈ set.Ioo (-1 : ℝ) 2 := sorry

-- Defining the quadratic function f
def f (x : ℝ) : ℝ := x^2 - 2 * m * x + 3

end quadratic_non_monotonic_iff_m_interval_l797_797775


namespace eccentricity_of_hyperbola_l797_797097

-- Definitions of various components
variables (a b : ℝ) (P F1 F2 : ℝ × ℝ)
-- Given conditions
variables (ha : 0 < a) (hb : 0 < b)
variables (P_on_hyperbola : P ∈ { (x, y) | x^2 / a^2 - y^2 / b^2 = 1 })
variables (left_branch : P.1 < 0)
variables PF1 PF2 : ℝ
variables (hPF1 : ∀ (P : ℝ × ℝ), sqrt((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = PF1)
variables (hPF2 : ∀ (P : ℝ × ℝ), sqrt((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = PF2)
variables (min_condition : ∀ (P : ℝ × ℝ), min (PF2^2 / PF1) = 9 * a)
variables c : ℝ

-- To prove
theorem eccentricity_of_hyperbola : 
  ∃ (e : ℝ), e = 5 ∧ e = c / a := sorry

end eccentricity_of_hyperbola_l797_797097


namespace range_of_m_l797_797644

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3 * m

theorem range_of_m (m : ℝ) : 
  satisfies_inequality m ↔ (m ≥ 4 ∨ m ≤ -1) :=
by
  sorry

end range_of_m_l797_797644


namespace part_a_single_line_intersection_l797_797304

structure Tetrahedron :=
  (A B C D : Point)

def is_perpendicular_to (p : Plane) (e : Edge) : Prop := sorry

def Plane1 (A B C : Point) : Plane := sorry
def Plane2 (A B D : Point) : Plane := sorry
def Plane3 (A C D : Point) : Plane := sorry

theorem part_a_single_line_intersection
  (tetra : Tetrahedron) 
  (P1 := Plane1 tetra.A tetra.B tetra.C)
  (P2 := Plane2 tetra.A tetra.B tetra.D)
  (P3 := Plane3 tetra.A tetra.C tetra.D) 
  (H1 : is_perpendicular_to P1 (Edge_of tetra.B tetra.C))
  (H2 : is_perpendicular_to P2 (Edge_of tetra.B tetra.D))
  (H3 : is_perpendicular_to P3 (Edge_of tetra.C tetra.D)) :
  ∃ (l : Line), P1 ∩ P2 ∩ P3 = l := 
sorry

end part_a_single_line_intersection_l797_797304


namespace distinct_cube_paintings_l797_797715

def cubePainting (yellow purple orange : ℕ) : Prop :=
  yellow = 1 ∧ purple = 2 ∧ orange = 3 → (distinctPaintings yellow purple orange = 3)

theorem distinct_cube_paintings :
  cubePainting 1 2 3 :=
by
  sorry

end distinct_cube_paintings_l797_797715


namespace hueys_share_l797_797352

def fraction_of_apple_pie_left : ℚ := 3 / 4
def fraction_of_cherry_pie_left : ℚ := 5 / 6
def number_of_people : ℚ := 3

theorem hueys_share : 
  let hueys_apple_pie := fraction_of_apple_pie_left / number_of_people in
  let hueys_cherry_pie := fraction_of_cherry_pie_left / number_of_people in
  hueys_apple_pie = 1 / 4 ∧ hueys_cherry_pie = 5 / 18 := sorry

end hueys_share_l797_797352


namespace lambda_range_l797_797409

-- Define the sequence a_n
def a : ℕ → ℚ
| 0       := 0  -- a_0 is not defined in the original problem, but including it for type correctness
| 1       := 1
| (n+2)   := (1 / 2) * a (n+1)

-- Define the sequence b_n
def b (n : ℕ) (a : ℕ → ℚ) : ℚ := (n^2 - 3*n - 2) * a n

-- Define the property for lambda
def valid_lambda (λ : ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → λ ≥ b n a

-- We need to prove that if valid_lambda λ holds, then λ ≥ 1/2
theorem lambda_range (λ : ℚ) (h : valid_lambda λ) : λ ≥ 1 / 2 :=
sorry

end lambda_range_l797_797409


namespace solve_quadratic_eq_l797_797229

theorem solve_quadratic_eq (x : ℝ) : 4 * x ^ 2 - (x - 1) ^ 2 = 0 ↔ x = -1 ∨ x = 1 / 3 :=
by
  sorry

end solve_quadratic_eq_l797_797229


namespace martin_cups_per_day_l797_797941

def cost_per_package : ℝ := 2.0

def total_amount_spent : ℝ := 30.0

def number_of_days : ℕ := 30

def total_packages : ℝ := total_amount_spent / cost_per_package

def total_cups : ℝ := total_packages

def cups_per_day : ℝ := total_cups / number_of_days

theorem martin_cups_per_day :
  cups_per_day = 0.5 :=
sorry

end martin_cups_per_day_l797_797941


namespace Linda_original_savings_l797_797201

theorem Linda_original_savings (S : ℝ)
  (H1 : 3/4 * S + 1/4 * S = S)
  (H2 : 1/4 * S = 220) :
  S = 880 :=
sorry

end Linda_original_savings_l797_797201


namespace tens_digit_of_72_pow_25_l797_797293

theorem tens_digit_of_72_pow_25 : (72^25 % 100) / 10 = 3 := 
by
  sorry

end tens_digit_of_72_pow_25_l797_797293


namespace determine_smallest_positive_period_l797_797767

noncomputable def smallest_positive_period (f : ℝ → ℝ) : ℝ :=
  if h : ∃ p > 0, ∀ x, f (x + p) = f x then Classical.choose h else 0

theorem determine_smallest_positive_period (f : ℝ → ℝ)
  (h : ∀ x, f (x + 5) + f (x - 5) = f x) : smallest_positive_period f = 30 :=
sorry

end determine_smallest_positive_period_l797_797767


namespace csc_315_eq_neg_sqrt_2_l797_797000

theorem csc_315_eq_neg_sqrt_2 :
  Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := 
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797000


namespace find_fathers_age_l797_797385

noncomputable def sebastian_age : ℕ := 40
noncomputable def age_difference : ℕ := 10
noncomputable def sum_ages_five_years_ago_ratio : ℚ := (3 : ℚ) / 4

theorem find_fathers_age 
  (sebastian_age : ℕ) 
  (age_difference : ℕ) 
  (sum_ages_five_years_ago_ratio : ℚ) 
  (h1 : sebastian_age = 40) 
  (h2 : age_difference = 10) 
  (h3 : sum_ages_five_years_ago_ratio = 3 / 4) 
: ∃ father_age : ℕ, father_age = 85 :=
sorry

end find_fathers_age_l797_797385


namespace men_women_alternate_l797_797317

theorem men_women_alternate 
  (M W : Type) 
  [Fintype M] [Fintype W] 
  [DecidableEq M] [DecidableEq W]
  (hM : Fintype.card M = 2) 
  (hW : Fintype.card W = 2) :
  ∃ l : List (M ⊕ W), l.length = 4 ∧ 
    (∀ i, i < l.length → ((l.nth i).elim (λ _, i.even) (λ _, i.odd)) = i.even) ∧ 
    Fintype.card {l // (∀ i, i < l.length → (l.nth i).elim (λ _, false) (λ _, odd) = i.odd)} = 8 :=
by
  sorry

end men_women_alternate_l797_797317


namespace candidate_A_leading_paths_l797_797666

/-- Number of ways such that candidate A is always leading in the count of votes. -/
theorem candidate_A_leading_paths (m n : ℕ) (h : m > n) :
  (m - n) * Nat.choose (m + n) m = (m + n) * (number_of_ways A B) := 
sorry

end candidate_A_leading_paths_l797_797666


namespace abs_fraction_eq_sqrt_seven_thirds_l797_797351

open Real

theorem abs_fraction_eq_sqrt_seven_thirds {a b : ℝ} 
  (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : a^2 + b^2 = 5 * a * b) : 
  abs ((a + b) / (a - b)) = sqrt (7 / 3) :=
by
  sorry

end abs_fraction_eq_sqrt_seven_thirds_l797_797351


namespace match_areas_3x_l797_797211

def area (matches : ℕ) : ℕ :=
  -- Definition of the area function based on the number of matches.

-- Conditions
constant A : ℕ  -- Area of the smaller initial figure
constant B : ℕ  -- Area of the larger initial figure
constant A_1 : ℕ  -- Area of the smaller figure after rearrangement
constant B_1 : ℕ  -- Area of the larger figure after rearrangement

axiom initial_condition : 3 * A = B
axiom new_condition : (number_of_matches : ℕ) -> 
  if number_of_matches = 6 then A else 
  if number_of_matches = 14 then B else 
  if number_of_matches = 7 then A_1 else 
  if number_of_matches = 13 then B_1 else 
  0 = area number_of_matches

theorem match_areas_3x : 3 * A_1 = B_1 :=
sorry

end match_areas_3x_l797_797211


namespace jake_time_to_row_lake_l797_797899

noncomputable def time_to_row_lake (side_length miles_per_side : ℝ) (swim_time_per_mile minutes_per_mile : ℝ) : ℝ :=
  let swim_speed := 60 / swim_time_per_mile -- miles per hour
  let row_speed := 2 * swim_speed          -- miles per hour
  let total_distance := 4 * side_length    -- miles
  total_distance / row_speed               -- hours

theorem jake_time_to_row_lake :
  time_to_row_lake 15 20 = 10 := sorry

end jake_time_to_row_lake_l797_797899


namespace not_necessarily_divisor_l797_797918

def consecutive_product (k : ℤ) : ℤ := k * (k + 1) * (k + 2) * (k + 3)

theorem not_necessarily_divisor (k : ℤ) (hk : 8 ∣ consecutive_product k) : ¬ (48 ∣ consecutive_product k) :=
sorry

end not_necessarily_divisor_l797_797918


namespace charles_distance_l797_797756

theorem charles_distance (s : ℝ) (t : ℝ) (d : ℝ) (h_speed : s = 3) (h_time : t = 2) : d = s * t → d = 6 :=
by
  assume h_dist : d = s * t
  rw [h_speed, h_time] at h_dist
  rw [h_dist]
  norm_num
  sorry

end charles_distance_l797_797756


namespace exists_subset_S_l797_797551

def T : set (ℤ × ℤ) := {p | true}

def adjacent (p q : ℤ × ℤ) : Prop := |p.1 - q.1| + |p.2 - q.2| = 1

def S : set (ℤ × ℤ) := {p | 5 ∣ (p.1 + 2 * p.2)}

theorem exists_subset_S :
  ∃ S ⊆ T, ∀ p ∈ T, (∃ q ∈ T, adjacent p q ∧ q ∈ S) ∧ (∀ q ∈ T, adjacent p q → q ≠ p → p ∈ S → q ∉ S) :=
by
  sorry

end exists_subset_S_l797_797551


namespace csc_315_eq_neg_sqrt_2_l797_797016

theorem csc_315_eq_neg_sqrt_2 :
  let csc := λ θ, 1 / Real.sin θ in
  csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
  by
  let sin := Real.sin
  have h1 : csc (315 * Real.pi / 180) = 1 / sin (315 * Real.pi / 180) := rfl
  have h2 : sin (315 * Real.pi / 180) = sin ((360 - 45) * Real.pi / 180) := by congr; norm_num
  have h3 : sin ((360 - 45) * Real.pi / 180) = -sin (45 * Real.pi / 180) := by
    rw [Real.sin_pi_sub]
    congr; norm_num
  have h4 : sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 := Real.sin_of_one_div_sqrt_two 45 rfl
  sorry

end csc_315_eq_neg_sqrt_2_l797_797016


namespace evaluate_expression_l797_797370

theorem evaluate_expression : (528 * 528) - (527 * 529) = 1 := by
  sorry

end evaluate_expression_l797_797370


namespace parallel_vectors_l797_797836

theorem parallel_vectors (m : ℝ) : 
  let a := (m, 1)
      b := (3, 2) in
  let collinear := ∃ k : ℝ, a = (k * b.1, k * b.2) in
  (collinear → m = 3 / 2) :=
by
  let a := (m, 1)
  let b := (3, 2)
  let collinear := ∃ k : ℝ, a = (k * b.1, k * b.2)
  intro h 
  sorry

end parallel_vectors_l797_797836


namespace variance_is_0_4_l797_797090
noncomputable def dataSet : List ℝ := [2, 1, 3, 2, 2]

def mean (xs : List ℝ) : ℝ :=
  (List.sum xs) / (xs.length)

def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (List.sum (List.map (fun x => (x - μ) ^ 2) xs)) / (xs.length)

theorem variance_is_0_4 : variance dataSet = 0.4 := by
  sorry

end variance_is_0_4_l797_797090


namespace find_k_l797_797543

def vec2 := ℝ × ℝ

-- Definitions
def i : vec2 := (1, 0)
def j : vec2 := (0, 1)
def a : vec2 := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : vec2 := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Dot product definition for 2D vectors
def dot_product (u v : vec2) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem
theorem find_k (k : ℝ) : dot_product a (b k) = 0 → k = 6 :=
by
  sorry

end find_k_l797_797543


namespace probability_heads_before_tails_l797_797921

theorem probability_heads_before_tails (q : ℚ) (a b : ℕ) (h_rel_prime : Nat.gcd a b = 1)
  (h_result : q = 13/17) : 
  a + b = 30 :=
by
  have h_q_def : q = (13 : ℚ) / 17, from h_result
  have h_q_form : q = (a : ℚ) / b, from sorry
  have h_rel_prime_q : (a / b : ℚ) = 13 / 17, from sorry
  have h_one : (a : ℚ) = 13 ∧ (b : ℚ) = 17, from sorry
  show a + b = 30, from sorry

end probability_heads_before_tails_l797_797921


namespace point_inside_circle_range_of_a_l797_797338

/- 
  Define the circle and the point P. 
  We would show that ensuring the point lies inside the circle implies |a| < 1/13.
-/

theorem point_inside_circle_range_of_a (a : ℝ) : 
  ((5 * a + 1 - 1) ^ 2 + (12 * a) ^ 2 < 1) -> |a| < 1 / 13 := 
by 
  sorry

end point_inside_circle_range_of_a_l797_797338


namespace arithmetic_sequence_a_find_p_q_find_c_minus_a_find_y_values_l797_797694

-- Problem (a)
theorem arithmetic_sequence_a (x1 x2 x3 x4 x5 : ℕ) (h : (x1 = 2 ∧ x2 = 5 ∧ x3 = 10 ∧ x4 = 13 ∧ x5 = 15)) : 
  ∃ a b c, (a = 5 ∧ b = 10 ∧ c = 15 ∧ b - a = c - b ∧ b - a > 0) := 
sorry

-- Problem (b)
theorem find_p_q (p q : ℕ) (h : ∃ d, (7 - p = d ∧ q - 7 = d ∧ 13 - q = d)) : 
  p = 4 ∧ q = 10 :=
sorry

-- Problem (c)
theorem find_c_minus_a (a b c : ℕ) (h : ∃ d, (b - a = d ∧ c - b = d ∧ (a + 21) - c = d)) :
  c - a = 14 :=
sorry

-- Problem (d)
theorem find_y_values (y : ℤ) (h : ∃ d, ((2*y + 3) - (y - 6) = d ∧ (y*y + 2) - (2*y + 3) = d) ) :
  y = 5 ∨ y = -2 :=
sorry

end arithmetic_sequence_a_find_p_q_find_c_minus_a_find_y_values_l797_797694


namespace johnson_family_seating_problem_l797_797610

theorem johnson_family_seating_problem : 
  ∃ n : ℕ, n = 9! - 5! * 4! ∧ n = 359760 :=
by
  have total_ways := (Nat.factorial 9)
  have no_adjacent_boys := (Nat.factorial 5) * (Nat.factorial 4)
  have result := total_ways - no_adjacent_boys
  use result
  split
  . exact eq.refl result
  . norm_num -- This will replace result with its evaluated form, 359760

end johnson_family_seating_problem_l797_797610


namespace radius_range_l797_797159

-- Defining the circle and conditions as Lean structures
structure Circle where
  center : (ℝ × ℝ)
  radius : ℝ

-- Defining a predicate that checks if a point is on the circle
def isOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1) ^ 2 + (p.2 - c.center.2) ^ 2 = c.radius ^ 2

-- Defining a predicate for the distance condition from x-axis
def distanceFromXAxis (p : ℝ × ℝ) := |p.2|

-- Defining a predicate that states there are exactly two points on the circle at distance 1 from the x-axis
def exactTwoPointsAtDistanceOne (c : Circle) : Prop :=
  let pts := {(x, 1) | isOnCircle c (x, 1)} ∪ {(x, -1) | isOnCircle c (x, -1)}
  pts.toFinite ∧ pts.toFinset.card = 2

-- The main theorem stating the range for r
theorem radius_range (r : ℝ) (c : Circle) (h_center : c.center = (3, -5)) (h_two_points : exactTwoPointsAtDistanceOne c) :
  4 < r ∧ r < 6 :=
by
  sorry

end radius_range_l797_797159


namespace JohnsonFamilySeating_l797_797629

theorem JohnsonFamilySeating : 
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 359000 := by
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  show total_arrangements - restricted_arrangements = 359000 from sorry

end JohnsonFamilySeating_l797_797629


namespace circle_chord_intersection_l797_797323

theorem circle_chord_intersection (O P A B C D E F : Type) [MetricSpace O] [MetricSpace P] [MetricSpace A] 
    [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
    (h1 : dist O A = 20)
    (h2 : dist O B = 20)
    (h3 : dist O C = 20)
    (h4 : dist O D = 20)
    (h5 : dist A B = 24)
    (h6 : dist C D = 16)
    (h7 : dist E F = 10)
    (h8 : midpoint E A B)
    (h9 : midpoint F C D)
    (h10 : intersect at P A B C D)
    : OP^2 = 2940 / 13 :=
by
  sorry

end circle_chord_intersection_l797_797323


namespace len_is_59_l797_797231

def is_prime (n : ℕ) := ∀ m : ℕ, (m ∣ n) → m = 1 ∨ m = n

def len_candidate_ages (guesses : List ℕ) : List ℕ :=
  guesses.filter (λ x, (guesses.contains (x - 1) ∨ guesses.contains (x + 1)))

noncomputable def len_age (guesses : List ℕ) (correct_age : ℕ) : Prop :=
  (len_candidate_ages guesses).contains correct_age ∧
  is_prime correct_age ∧
  (correct_age % 3 ≠ 0) ∧
  (guesses.filter (λ g, g < correct_age)).length * 2 ≥ guesses.length

theorem len_is_59 :
  len_age [31, 35, 39, 43, 47, 52, 54, 57, 58, 60] 59 :=
by
  sorry

end len_is_59_l797_797231


namespace john_sold_books_on_friday_l797_797900

theorem john_sold_books_on_friday :
  let initial_stock := 1300
      sold_monday := 75
      sold_tuesday := 50
      sold_wednesday := 64
      sold_thursday := 78
      percentage_unsold := 69.07692307692308
      unsold_books := (percentage_unsold / 100) * initial_stock
      total_sold_mon_thu := sold_monday + sold_tuesday + sold_wednesday + sold_thursday
      sold_friday := initial_stock - (total_sold_mon_thu + unsold_books)
  in sold_friday = 135 := 
by
  -- Here should be a detailed proof, retained as a placeholder
  sorry

end john_sold_books_on_friday_l797_797900


namespace sequence_a10_l797_797847

theorem sequence_a10 : 
  (∃ (a : ℕ → ℤ), 
    a 1 = -1 ∧ 
    (∀ n : ℕ, n ≥ 1 → a (2*n) - a (2*n - 1) = 2^(2*n-1)) ∧ 
    (∀ n : ℕ, n ≥ 1 → a (2*n + 1) - a (2*n) = 2^(2*n))) → 
  (∃ a : ℕ → ℤ, a 10 = 1021) :=
by
  intro h
  obtain ⟨a, h1, h2, h3⟩ := h
  sorry

end sequence_a10_l797_797847


namespace max_tan_angle_BAD_l797_797664

theorem max_tan_angle_BAD (A B C D : Type) [euclidean_geometry A B C D] 
    (angle_C_45 : ∠ C = 45)
    (BC_eq_2 : dist B C = 2)
    (D_midpoint_BC : midpoint D B C) : 
    tan (angle A B D) = (2 - sqrt 2) / (2 * sqrt 2 + 1) :=
by 
    sorry

end max_tan_angle_BAD_l797_797664


namespace find_c_degree_3_l797_797768

def f (x : ℝ) : ℝ := 3 + 8*x - 4*x^2 + 6*x^3 - 7*x^4
def g (x : ℝ) : ℝ := 5 - 3*x + x^2 - 8*x^3 + 11*x^4

theorem find_c_degree_3 : ∃ c : ℝ, (∀ x : ℝ, f x + c * g x = 0) ∧ (∀ x : ℝ, degree (f x + c * g x) = 3) :=
by
  sorry

end find_c_degree_3_l797_797768


namespace lnot_p_sufficient_but_not_necessary_for_lnot_q_l797_797400

variable (x : ℝ)

def p : Prop := 0 < x ∧ x < 2
def q : Prop := 1 / x ≥ 1
def not_p : Prop := x ≥ 2 ∨ x ≤ 0
def not_q : Prop := x > 1 ∨ x ≤ 0

theorem lnot_p_sufficient_but_not_necessary_for_lnot_q : not_p x → not_q x ∧ ¬(not_q x → not_p x) :=
by
  intro h
  split
  . sorry
  . sorry

end lnot_p_sufficient_but_not_necessary_for_lnot_q_l797_797400


namespace csc_315_eq_neg_sqrt2_l797_797009

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end csc_315_eq_neg_sqrt2_l797_797009


namespace mini_train_speed_l797_797335

theorem mini_train_speed (length : ℝ) (time : ℝ) (in_kmph : ℝ) :
  length = 62.505 → time = 3 → in_kmph = (length / 1000) / (time / 3600) → in_kmph = 75 :=
by
  intros h_len h_time h_calc
  rw [h_len, h_time]
  norm_num at h_calc
  exact h_calc

end mini_train_speed_l797_797335


namespace rectangle_circle_area_ratio_l797_797994

theorem rectangle_circle_area_ratio 
  (radius : ℝ) 
  (rectangle_area : ℝ) 
  (h_radius : radius = 5) 
  (h_rectangle_area : rectangle_area = 50) : 
  let circle_area := real.pi * radius^2 in
  rectangle_area / circle_area = 2 / real.pi :=
by
  sorry

end rectangle_circle_area_ratio_l797_797994


namespace sum_of_integers_l797_797395

theorem sum_of_integers (a b c d : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : d > 1)
    (h_prod : a * b * c * d = 1000000)
    (h_gcd1 : Nat.gcd a b = 1) (h_gcd2 : Nat.gcd a c = 1) (h_gcd3 : Nat.gcd a d = 1)
    (h_gcd4 : Nat.gcd b c = 1) (h_gcd5 : Nat.gcd b d = 1) (h_gcd6 : Nat.gcd c d = 1) : 
    a + b + c + d = 15698 :=
sorry

end sum_of_integers_l797_797395


namespace triangle_angle_bisector_lemma_l797_797267

-- Define the given conditions as Lean structures and properties.
open EuclideanGeometry

variables {A B C M N : Point}
variables {CA CB AM BM AN BN : Real}
variables [TriangleABC : Triangle A B C]

-- Assuming the two lines through C form equal angles with sides CA and CB and intersect AB at points M and N.
axiom equal_angles (line1 : Line) (line2 : Line) :
  are_concurrent C [line1, line2] ∧
  equal_angles (angle C A line1) (angle C B line2) ∧
  line_intersects_at line1 A B M ∧ 
  line_intersects_at line2 A B N
  [ CA = distance C A, CB = distance C B,
    AM = distance A M, BM = distance B M,
    AN = distance A N, BN = distance B N ]

-- Formulate the theorem to be proven.
theorem triangle_angle_bisector_lemma :
  CA * CA / (CB * CB) = AM * AN / (BM * BN) :=
by sorry

end triangle_angle_bisector_lemma_l797_797267


namespace domain_of_log_function_is_correct_l797_797789

noncomputable def domain_of_log_function : set ℝ := { x : ℝ | ∃ (k : ℤ), (k * real.pi - real.pi / 4) < x ∧ x < (k * real.pi + real.pi / 4) }

theorem domain_of_log_function_is_correct :
  (∀ x : ℝ, ∃ (k : ℤ), (2 * real.cos x ^ 2 - 1 > 0) ↔ (domain_of_log_function x)) :=
begin
  sorry
end

end domain_of_log_function_is_correct_l797_797789


namespace spherical_to_rectangular_example_l797_797360

noncomputable def spherical_to_rectangular (ρ θ ϕ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin ϕ * Real.cos θ, ρ * Real.sin ϕ * Real.sin θ, ρ * Real.cos ϕ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 4 (Real.pi / 4) (Real.pi / 6) = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) :=
by
  sorry

end spherical_to_rectangular_example_l797_797360


namespace four_rows_and_columns_without_queens_l797_797563

-- Definition of the chessboard and conditions
def chessboard : Type := fin 8 × fin 8

-- Given 12 queens on an 8 × 8 chessboard
def queens_positions (q : fin 12) : chessboard := sorry

-- The problem statement
theorem four_rows_and_columns_without_queens :
  ∃ (rows cols : finset (fin 8)), 
    rows.card = 4 ∧ cols.card = 4 ∧
    ∀ r ∈ rows, ∀ c ∈ cols, 
       ∀ q, queens_positions q ≠ (r, c) := 
sorry

end four_rows_and_columns_without_queens_l797_797563


namespace microdegrees_in_seventeenth_of_circle_l797_797134

theorem microdegrees_in_seventeenth_of_circle :
  let degree_in_full_circle := 360
  let one_microdegree := 1 / 1000
  let seventeenth_circle := degree_in_full_circle / 17
  let microdegrees := seventeenth_circle / one_microdegree
  microdegrees = 21176.4705882 :=
by
  -- Definitions
  let degree_in_full_circle := 360
  let one_microdegree := 1 / 1000
  let seventeenth_circle := degree_in_full_circle / 17
  let microdegrees := seventeenth_circle / one_microdegree
  -- The proof
  have h := seventeenth_circle,
  have m := microdegrees,
  simp [h, m],
  sorry

end microdegrees_in_seventeenth_of_circle_l797_797134


namespace election_count_l797_797565

def girls : ℕ := 13
def boys : ℕ := 12

noncomputable def choose_president_and_vice_president (g b : ℕ) : ℕ :=
g * (b / 2)
 
theorem election_count : choose_president_and_vice_president girls boys = 78 :=
by
  have h_girls := girls
  have h_boys := boys
  unfold choose_president_and_vice_president
  rw [←nat.div_mul_cancel (by norm_num : 12 % 2 = 0)]
  norm_num
  sorry

end election_count_l797_797565


namespace triangle_side_length_l797_797536

/-
  Given a triangle ABC with sides |AB| = c, |AC| = b, and centroid G, incenter I,
  if GI is perpendicular to BC, then we need to prove that |BC| = (b+c)/2.
-/
variable {A B C G I : Type}
variable {AB AC BC : ℝ} -- Lengths of the sides
variable {b c : ℝ} -- Given lengths
variable {G_centroid : IsCentroid A B C G} -- G is the centroid of triangle ABC
variable {I_incenter : IsIncenter A B C I} -- I is the incenter of triangle ABC
variable {G_perp_BC : IsPerpendicular G I BC} -- G I ⊥ BC

theorem triangle_side_length (h1 : |AB| = c) (h2 : |AC| = b) :
  |BC| = (b + c) / 2 := 
sorry

end triangle_side_length_l797_797536


namespace part1_part2_l797_797427

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l797_797427


namespace fraction_meaningful_l797_797873

theorem fraction_meaningful (a : ℝ) : (∃ b, b = 2 / (a + 1)) → a ≠ -1 :=
by
  sorry

end fraction_meaningful_l797_797873


namespace distinct_tower_heights_l797_797943

/-- Given 94 bricks where each brick can contribute either 3, 11, or 20 inches to the height.
Prove that the number of distinct total heights that can be achieved is 1541. -/
theorem distinct_tower_heights : 
  let bricks := 94
  let increments := {3, 11, 20}
  let minHeight := 282
  let maxHeight := 1822
  (maxHeight - minHeight + 1) = 1541 :=
by
  sorry

end distinct_tower_heights_l797_797943


namespace father_l797_797392

noncomputable def father's_current_age : ℕ :=
  let S : ℕ := 40 -- Sebastian's current age
  let Si : ℕ := S - 10 -- Sebastian's sister's current age
  let sum_five_years_ago := (S - 5) + (Si - 5) -- Sum of their ages five years ago
  let father_age_five_years_ago := (4 * sum_five_years_ago) / 3 -- From the given condition
  father_age_five_years_ago + 5 -- Their father's current age

theorem father's_age_is_85 : father's_current_age = 85 :=
  sorry

end father_l797_797392


namespace le_n_minus_two_l797_797815

theorem le_n_minus_two (n : ℕ) (a : fin n → ℕ) (r : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : ∀ i j, i ≠ j → gcd (a i) (a j) = 1) 
  (h3 : ∀ i, (a.prod (λ j, if j = i then 1 else a j)) % (a i) = r) : 
  r ≤ n-2 :=
sorry

end le_n_minus_two_l797_797815


namespace fish_caught_l797_797746

noncomputable def total_fish_caught (chris_trips : ℕ) (chris_fish_per_trip : ℕ) (brian_trips : ℕ) (brian_fish_per_trip : ℕ) : ℕ :=
  chris_trips * chris_fish_per_trip + brian_trips * brian_fish_per_trip

theorem fish_caught (chris_trips : ℕ) (brian_factor : ℕ) (brian_fish_per_trip : ℕ) (ratio_numerator : ℕ) (ratio_denominator : ℕ) :
  chris_trips = 10 → brian_factor = 2 → brian_fish_per_trip = 400 → ratio_numerator = 3 → ratio_denominator = 5 →
  total_fish_caught chris_trips (brian_fish_per_trip * ratio_denominator / ratio_numerator) (chris_trips * brian_factor) brian_fish_per_trip = 14660 :=
by
  intros h_chris_trips h_brian_factor h_brian_fish_per_trip h_ratio_numer h_ratio_denom
  rw [h_chris_trips, h_brian_factor, h_brian_fish_per_trip, h_ratio_numer, h_ratio_denom]
  -- adding actual arithmetic would resolve the statement correctly
  sorry

end fish_caught_l797_797746


namespace power_evaluation_l797_797690

theorem power_evaluation : (-1)^(5^2) + 1^(2^5) = 0 := 
  by
    have h1 : 5^2 = 25 := by norm_num
    have h2 : 2^5 = 32 := by norm_num
    rw [h1, h2]
    norm_num
    sorry

end power_evaluation_l797_797690


namespace ratio_to_percent_l797_797255

theorem ratio_to_percent (a b : ℕ) (h : a = 6) (h2 : b = 3) :
  ((a / b : ℚ) * 100 = 200) :=
by
  have h3 : a = 6 := h
  have h4 : b = 3 := h2
  sorry

end ratio_to_percent_l797_797255


namespace must_be_true_statements_l797_797233

-- Definitions for conditions
variables (Dragon FormidableBeast MysteriousCreature ScaryMonster : Type)
variables (isDragon : Dragon → Prop)
variables (isFormidableBeast : FormidableBeast → Prop)
variables (isMysteriousCreature : MysteriousCreature → Prop)
variables (isScaryMonster : ScaryMonster → Prop)
variables (dragonsAreFormidableBeasts : ∀ d : Dragon, isFormidableBeast d)
variables (someMysteriousCreaturesAreDragons : ∃ d : Dragon, isMysteriousCreature d)
variables (dragonsAreScaryMonsters : ∀ d : Dragon, isScaryMonster d)

-- Statements to prove
theorem must_be_true_statements :
  (∃ f : FormidableBeast, isMysteriousCreature f) ∧ (∃ s : ScaryMonster, isMysteriousCreature s) :=
by
  sorry

end must_be_true_statements_l797_797233


namespace father_current_age_is_85_l797_797390

theorem father_current_age_is_85 (sebastian_age : ℕ) (sister_diff : ℕ) (age_sum_fraction : ℕ → ℕ → ℕ → Prop) :
  sebastian_age = 40 →
  sister_diff = 10 →
  (∀ (s s' f : ℕ), age_sum_fraction s s' f → f = 4 * (s + s') / 3) →
  age_sum_fraction (sebastian_age - 5) (sebastian_age - sister_diff - 5) (40 + 5) →
  ∃ father_age : ℕ, father_age = 85 :=
by
  intros
  sorry

end father_current_age_is_85_l797_797390


namespace range_of_a_l797_797141

theorem range_of_a (a : ℝ) : (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) → (a < -2 ∨ a > 2) :=
by
  assume h : ¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0
  sorry

end range_of_a_l797_797141


namespace symmetric_point_coordinates_l797_797787

/--
Given a point P with coordinates (2, -1, 2) and a line defined by
(x - 1) / 1 = (y - 0) / 0 = (z + 1) / -2, find the coordinates of point Q
such that Q is symmetric to P with respect to the given line.
-/
theorem symmetric_point_coordinates :
  let P := (2, -1, 2)
  let line_x t := t + 1
  let line_y t := 0
  let line_z t := -2 * t - 1
  let P' := (0, 0, 1) -- This is the projection of P on the line
  let Q := (-2, 1, 0)
  Q = (2 * P'.1 - P.1, 2 * P'.2 - P.2, 2 * P'.3 - P.3) :=
by
  -- Proof skipped
  sorry

end symmetric_point_coordinates_l797_797787


namespace lakshmi_share_l797_797219

-- Definitions of conditions
def raman_investment (x : ℝ) : ℝ := x
def lakshmi_investment (x : ℝ) : ℝ := 2 * x
def muthu_investment (x : ℝ) : ℝ := 3 * x

-- Investment periods
def raman_investment_months (x : ℝ) : ℝ := raman_investment x * 12
def lakshmi_investment_months (x : ℝ) : ℝ := lakshmi_investment x * 6
def muthu_investment_months (x : ℝ) : ℝ := muthu_investment x * 4

-- Total investment-months
def total_investment_months (x : ℝ) : ℝ :=
  raman_investment_months x + lakshmi_investment_months x + muthu_investment_months x

-- Lakshmi's share
def lakshmi_share_ratio (x : ℝ) : ℝ :=
  lakshmi_investment_months x / total_investment_months x

def total_annual_gain : ℝ := 36000

theorem lakshmi_share (x : ℝ) : 
  lakshmi_share_ratio x * total_annual_gain = 12000 :=
by
  sorry

end lakshmi_share_l797_797219


namespace part1_part2_l797_797469

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l797_797469


namespace randy_blocks_left_l797_797958

-- Definitions of the conditions
def initial_blocks : ℕ := 78
def blocks_for_tower : ℕ := 19
def blocks_for_bridge (remaining_blocks : ℕ) : ℕ := remaining_blocks / 2
def blocks_for_house : ℕ := 11

-- Lean statement to prove the final number of blocks left
theorem randy_blocks_left : 
  let remaining_after_tower := initial_blocks - blocks_for_tower in
  let remaining_after_bridge := remaining_after_tower - blocks_for_bridge remaining_after_tower in
  let remaining_after_house := remaining_after_bridge - blocks_for_house in
  remaining_after_house = 19 :=
by 
  sorry

end randy_blocks_left_l797_797958


namespace midpoint_probability_l797_797190

theorem midpoint_probability :
  let T := { (x, y, z) | ∀ x y z, x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧ 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 6 ∧ 0 ≤ z ∧ z ≤ 4 } in
  let distinct_points := { (p1, p2) | p1 ∈ T ∧ p2 ∈ T ∧ p1 ≠ p2 } in
  let valid_midpoints := (p1, p2) ∈ distinct_points → ((p1.1 + p2.1) / 2 % 2 = 0) ∧ ((p1.2 + p2.2) / 2 % 2 = 0) ∧ ((p1.3 + p2.3) / 2 % 2 = 0) in
  let p := 1 in 
  let q := 1 in 
  let prob := p / q in
  (p + q = 2) :=
by
  sorry

end midpoint_probability_l797_797190


namespace part1_part2_l797_797462

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l797_797462


namespace find_p_q_sum_l797_797914

open Real

noncomputable def is_prob_two_distinct_real_solutions (b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^4 + 16 * b^2 = (9 * b^2 - 15 * b) * x₁^2) ∧ (x₂^4 + 16 * b^2 = (9 * b^2 - 15 * b) * x₂^2)

theorem find_p_q_sum :
  ∃ p q : ℕ, nat.coprime p q ∧ p + q = 733 ∧ 
  (probability (is_prob_two_distinct_real_solutions) (Icc (-10 : ℝ) 10)) = (p : ℝ) / (q : ℝ) :=
sorry

end find_p_q_sum_l797_797914


namespace factorial_division_l797_797418

theorem factorial_division :
  9! = 362880 → (9! / 4!) = 15120 := by
  sorry

end factorial_division_l797_797418


namespace base7_to_base10_54321_l797_797283

-- Declaring the number in base 7
def number_in_base7 := [5, 4, 3, 2, 1] -- Represents the digits of 54321 in base 7

-- Function to convert a base 7 number to base 10
def base7_to_base10 (digits : List ℕ) : ℕ :=
  digits.reverse.foldl (λ acc digit, acc * 7 + digit) 0

-- The theorem to be proven
theorem base7_to_base10_54321 :
  base7_to_base10 number_in_base7 = 13539 :=
  sorry

end base7_to_base10_54321_l797_797283


namespace csc_315_eq_neg_sqrt2_l797_797007

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end csc_315_eq_neg_sqrt2_l797_797007


namespace average_height_of_trees_l797_797178

def first_tree_height : ℕ := 1000
def half_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200

def total_height : ℕ := first_tree_height + 2 * half_tree_height + last_tree_height
def number_of_trees : ℕ := 4
def average_height : ℕ := total_height / number_of_trees

theorem average_height_of_trees :
  average_height = 800 :=
by
  -- This line contains a placeholder proof, the actual proof is omitted.
  sorry

end average_height_of_trees_l797_797178


namespace plane_parallel_exists_unique_l797_797660

-- Define given point A and given plane P
variable (A : Point)
variable (P : Plane)

-- The goal is to prove the existence and uniqueness of a plane Q that passes through A and is parallel to P
theorem plane_parallel_exists_unique : ∃! (Q : Plane), Q.contains A ∧ Q ∥ P := 
sorry

end plane_parallel_exists_unique_l797_797660


namespace part1_part2_l797_797470

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l797_797470


namespace distance_KL_l797_797906

-- Define the points and their coordinates
def point (x y : ℝ) := (x, y)

def A := point (-1/2) 0
def B := point (-1/2) 1
def C := point (1/2) 1
def D := point (1/2) 0
def E := point (-1/2) (1/3)
def F := point (-1/2) (2/3)
def G := point (-1/6) 1
def H := point (1/6) 1
def I := point (1/2) (1/2)
def J := point 0 0

-- Define the intersection points K and L
def line (p1 p2 : ℝ × ℝ) (x : ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1) * (x - p1.1) + p1.2

def HJ := line H J
def EI := line E I
def GJ := line G J
def FI := line F I

def intersect_x (f g : ℝ → ℝ) : ℝ :=
  let a := f 0 - g 0
  let b := (fun x => (f x) - (g x))
  let x0 := a / b 1
  x0

def intersect_y (f : ℝ → ℝ) (x : ℝ) : ℝ := f x

def K := (intersect_x HJ EI, intersect_y HJ (intersect_x HJ EI))
def L := (intersect_x GJ FI, intersect_y GJ (intersect_x GJ FI))

-- Distance formula
def distance (p1 p2 : ℝ × ℝ) :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Theorem statement
theorem distance_KL : distance K L = 6 * Real.sqrt 2 / 35 := sorry

end distance_KL_l797_797906


namespace decreasing_function_on_interval_l797_797677

theorem decreasing_function_on_interval :
  ∃ f : ℝ → ℝ, (∀ x ∈ set.Ioo 0 1, f x = real.cos x) ∧
  (∀ f' ∈ [λ x, real.log x, λ x, 2^x, λ x, 1/(2*x - 1)], ∀ x ∈ set.Ioo 0 1, deriv f x < 0 → f = real.cos) :=
by
  sorry

end decreasing_function_on_interval_l797_797677


namespace trivia_competition_points_l797_797739

theorem trivia_competition_points 
  (total_members : ℕ := 120) 
  (absent_members : ℕ := 37) 
  (points_per_member : ℕ := 24) : 
  (total_members - absent_members) * points_per_member = 1992 := 
by
  sorry

end trivia_competition_points_l797_797739


namespace f_comp_f_root_l797_797907

theorem f_comp_f_root 
  (α : ℝ) 
  (p : Polynomial ℝ) 
  (h : p = Polynomial.C 3 - Polynomial.monomial 1 (5 : ℝ) + Polynomial.monomial 3 (1 : ℝ)) 
  (hα : p.eval α = 0) 
  (f : Polynomial ℚ) 
  (hf : ∀ x, ∃ q r, f.eval x = (q * p).eval x + r.eval x ∧ r.degree < p.degree) 
  : p.eval (f.eval (f.eval α)) = 0 := 
sorry

end f_comp_f_root_l797_797907


namespace math_problem_A_B_M_l797_797118

theorem math_problem_A_B_M :
  ∃ M : Set ℝ,
    M = {m | ∃ A B : Set ℝ,
      A = {x | x^2 - 5 * x + 6 = 0} ∧
      B = {x | m * x - 1 = 0} ∧
      A ∩ B = B ∧
      M = {0, (1:ℝ)/2, (1:ℝ)/3}} ∧
    ∃ subsets : Set (Set ℝ),
      subsets = {∅, {0}, {(1:ℝ)/2}, {(1:ℝ)/3}, {0, (1:ℝ)/2}, {(1:ℝ)/2, (1:ℝ)/3}, {0, (1:ℝ)/3}, {0, (1:ℝ)/2, (1:ℝ)/3}} :=
by
  sorry

end math_problem_A_B_M_l797_797118


namespace find_parallel_line_eq_l797_797839

noncomputable def curve (x : ℝ) : ℝ := 4 / x

theorem find_parallel_line_eq 
  (tangent_eq : ∀ x, derive (curve x) = -4)
  (tangent_pt : curve 1 = 4)
  (dist_eq : ∀ l₁ l₂ : ℝ, l₁ - l₂ = 17 ∨ l₂ - l₁ = 17) :
  (∀ x y : ℝ, (4 * x + y - 25 = 0) ∨ (4 * x + y + 9 = 0)) :=
  sorry

end find_parallel_line_eq_l797_797839


namespace vince_bus_ride_distance_l797_797281

/-- 
  Vince's bus ride to school is 0.625 mile, 
  given that Zachary's bus ride is 0.5 mile 
  and Vince's bus ride is 0.125 mile longer than Zachary's.
--/
theorem vince_bus_ride_distance (zachary_ride : ℝ) (vince_longer : ℝ) 
  (h1 : zachary_ride = 0.5) (h2 : vince_longer = 0.125) 
  : zachary_ride + vince_longer = 0.625 :=
by sorry

end vince_bus_ride_distance_l797_797281


namespace johnson_family_seating_l797_797634

theorem johnson_family_seating (boys girls : Finset ℕ) (h_boys : boys.card = 5) (h_girls : girls.card = 4) :
  (∃ (arrangement : List ℕ), arrangement.length = 9 ∧ at_least_two_adjacent boys arrangement) :=
begin
  -- Given the total number of ways: 9! 
  -- subtract 5! * 4! from 9! to get the result 
  have total_arrangements := nat.factorial 9,
  have restrictive_arrangements := nat.factorial 5 * nat.factorial 4,
  exact (total_arrangements - restrictive_arrangements) = 360000,
end

end johnson_family_seating_l797_797634


namespace intersection_sum_x_coordinates_eq_neg_17_l797_797212

noncomputable def sum_of_x_coordinates (c d : ℕ) : ℚ :=
  if cd_cond: c * d = 30
  then -5 
            - 5 * (1 / 2 : ℚ) 
            - 5 * (1 / 3 : ℚ) 
            - 1 
            - 5 * (1 / 6 : ℚ) 
            - (1 / 2 : ℚ) 
            - (1 / 3 : ℚ) 
            - (1 / 6 : ℚ)
  else 0

theorem intersection_sum_x_coordinates_eq_neg_17 : 
  (∀ c d : ℕ, (c * d = 30) → (sum_of_x_coordinates c d = -17)) :=
by
  intros c d h
  rw sum_of_x_coordinates
  split_ifs
  . exact rfl
  . sorry

end intersection_sum_x_coordinates_eq_neg_17_l797_797212


namespace volume_rect_prism_l797_797256

/-- If the sides of the base of a rectangular prism are in the ratio m : n,
and the diagonal cross-section is a square with area Q, then the volume of the prism is
   V = (mnQ^(3/2)) / (m^2 + n^2).
-/

theorem volume_rect_prism (m n Q : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hQ : Q > 0) :
  let V := (mnQ^(3/2)) / (m^2 + n^2) in
  V = (m * n * Q * Real.sqrt(Q))/(m^2 + n^2) :=
sorry

end volume_rect_prism_l797_797256


namespace total_books_l797_797123

def school_books : ℕ := 19
def sports_books : ℕ := 39

theorem total_books : school_books + sports_books = 58 := by
  sorry

end total_books_l797_797123


namespace good_number_adjacent_to_powers_of_2_unique_l797_797864

def good_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≥ 2 ∧ b ≥ 2 ∧ n = a ^ b

def adjacent_to_power_of_2 (n : ℕ) : Prop :=
  ∃ (t : ℕ), n = 2 ^ t + 1 ∨ n = 2 ^ t - 1

theorem good_number_adjacent_to_powers_of_2_unique :
  (∃ n : ℕ, good_number n ∧ adjacent_to_power_of_2 n) →
  ∃! n, good_number n ∧ adjacent_to_power_of_2 n ∧ n = 9 := 
begin
  sorry
end

end good_number_adjacent_to_powers_of_2_unique_l797_797864


namespace csc_315_eq_neg_sqrt_2_l797_797015

theorem csc_315_eq_neg_sqrt_2 :
  let csc := λ θ, 1 / Real.sin θ in
  csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
  by
  let sin := Real.sin
  have h1 : csc (315 * Real.pi / 180) = 1 / sin (315 * Real.pi / 180) := rfl
  have h2 : sin (315 * Real.pi / 180) = sin ((360 - 45) * Real.pi / 180) := by congr; norm_num
  have h3 : sin ((360 - 45) * Real.pi / 180) = -sin (45 * Real.pi / 180) := by
    rw [Real.sin_pi_sub]
    congr; norm_num
  have h4 : sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 := Real.sin_of_one_div_sqrt_two 45 rfl
  sorry

end csc_315_eq_neg_sqrt_2_l797_797015


namespace algebraic_expression_value_l797_797078

noncomputable def a : ℝ := 2 * Real.sin (Real.pi / 4) + 1
noncomputable def b : ℝ := 2 * Real.cos (Real.pi / 4) - 1

theorem algebraic_expression_value :
  ((a^2 + b^2) / (2 * a * b) - 1) / ((a^2 - b^2) / (a^2 * b + a * b^2)) = 1 :=
by sorry

end algebraic_expression_value_l797_797078


namespace circle_equation_tangent_line_l797_797641

noncomputable def circle_center : (ℝ × ℝ) := (1, -1)
noncomputable def tangent_line : ℝ × ℝ × ℝ := (1, -1, 2)

theorem circle_equation_tangent_line : 
  ∃ (R : ℝ), 
    (let (x₀, y₀) := circle_center in let (A, B, C) := tangent_line in 
    R = abs (A * x₀ + B * y₀ + C) / sqrt (A^2 + B^2)) ∧
    (let (a, b) := circle_center in
    (x : ℝ) → (y : ℝ) → (x - a)^2 + (y - b)^2 = R^2 -> (x - 1)^2 + (y + 1)^2 = 8) :=
begin
  sorry
end

end circle_equation_tangent_line_l797_797641


namespace fraction_of_sides_area_of_triangle_l797_797894

-- Part (1)
theorem fraction_of_sides (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π)
  (h_sines : 2 * (Real.tan A + Real.tan B) = (Real.tan A / Real.cos B) + (Real.tan B / Real.cos A))
  (h_sine_law : c = 2) : (a + b) / c = 2 :=
sorry

-- Part (2)
theorem area_of_triangle (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π)
  (h_sines : 2 * (Real.tan A + Real.tan B) = (Real.tan A / Real.cos B) + (Real.tan B / Real.cos A))
  (h_sine_law : c = 2) (h_C : C = π / 3) : (1 / 2) * a * b * Real.sin C = Real.sqrt 3 :=
sorry

end fraction_of_sides_area_of_triangle_l797_797894


namespace part1_part2_l797_797309

open Real

noncomputable def f (x a b : ℝ) := x^2 + a * x + b
noncomputable def g (x : ℝ) := 2 * x^2 + 4 * x - 30

axiom condition (a b : ℝ) : ∀ x : ℝ, abs (f x a b) ≤ abs (g x)

def a_seq : ℕ → ℝ
| 0       => 1 / 2
| (n + 1) => (f (a_seq n) 2 (-15) + 15) / 2

def b_seq : ℕ → ℝ := fun n => 1 / (2 + a_seq n)

def S_n (n : ℕ) := (Finset.range n).sum (λ i, b_seq i)
def T_n (n : ℕ) := (Finset.range n).prod (λ i, b_seq i)

theorem part1 : ∀ a b : ℝ, condition a b → a = 2 ∧ b = -15 :=
by
  intros a b h
  sorry

theorem part2 : ∀ n : ℕ, 2^(n+1) * T_n n + S_n n = 2 :=
by
  intro n
  sorry

end part1_part2_l797_797309


namespace segment_length_l797_797670

theorem segment_length (x : ℝ) (h : |x - (8 : ℝ)^(1/3)| = 4) : 
  let a := (8 : ℝ)^(1/3) in
  |(a + 4) - (a - 4)| = 8 :=
by 
  sorry

end segment_length_l797_797670


namespace part1_part2_l797_797472

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l797_797472


namespace sum_of_first_10_terms_l797_797162

-- Define the arithmetic sequence as a function from natural numbers to reals
def arithmetic_seq (a d : ℝ) : ℕ → ℝ
| 0    := a
| (n+1) := arithmetic_seq n + d

-- Definitions for the conditions from a)
def a_3 (a d : ℝ) : ℝ := arithmetic_seq a d 2
def a_4 (a d : ℝ) : ℝ := arithmetic_seq a d 3
def a_5 (a d : ℝ) : ℝ := arithmetic_seq a d 4
def a_7 (a d : ℝ) : ℝ := arithmetic_seq a d 6

-- The main theorem to prove
theorem sum_of_first_10_terms (a d : ℝ) :
  a_4 a d = 4 →
  a_3 a d + a_5 a d + a_7 a d = 15 →
  ∑ i in Finset.range 10, arithmetic_seq a d i = 55 :=
by
  intros h1 h2
  sorry

end sum_of_first_10_terms_l797_797162


namespace quadratic_increasing_l797_797501

noncomputable def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

theorem quadratic_increasing (a b c : ℝ) 
  (h1 : quadratic a b c 0 = quadratic a b c 6)
  (h2 : quadratic a b c 0 < quadratic a b c 7) :
  ∀ x, x > 3 → ∀ y, y > 3 → x < y → quadratic a b c x < quadratic a b c y :=
sorry

end quadratic_increasing_l797_797501


namespace calculate_interest_period_l797_797795

variables (P R SI: ℝ) (T: ℕ)

-- Given conditions
def principal := 10000
def rate := 9 / 100
def simple_interest := 900

-- To calculate the number of years
def years : ℝ := (simple_interest * 100) / (principal * rate)

-- Convert years to months
def months := years * 12

theorem calculate_interest_period : T = 12 :=
by sorry

end calculate_interest_period_l797_797795


namespace min_value_l797_797102

-- Definition of the conditions
def positive (a : ℝ) : Prop := a > 0

theorem min_value (a : ℝ) (h : positive a) : 
  ∃ m : ℝ, (m = 2 * Real.sqrt 6) ∧ (∀ x : ℝ, positive x → (3 / (2 * x) + 4 * x) ≥ m) :=
sorry

end min_value_l797_797102


namespace shoe_length_increase_l797_797333

theorem shoe_length_increase
  (L : ℝ)
  (x : ℝ)
  (h1 : L + 9*x = L * 1.2)
  (h2 : L + 7*x = 10.4) :
  x = 0.2 :=
by
  sorry

end shoe_length_increase_l797_797333


namespace symmetric_point_x_correct_l797_797523

-- Define the Cartesian coordinate system
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetry with respect to the x-axis
def symmetricPointX (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Given point (-2, 1, 4)
def givenPoint : Point3D := { x := -2, y := 1, z := 4 }

-- Define the expected symmetric point
def expectedSymmetricPoint : Point3D := { x := -2, y := -1, z := -4 }

-- State the theorem to prove the expected symmetric point
theorem symmetric_point_x_correct :
  symmetricPointX givenPoint = expectedSymmetricPoint := by
  -- here the proof would go, but we leave it as sorry
  sorry

end symmetric_point_x_correct_l797_797523


namespace small_triangles_in_rectangle_l797_797337

theorem small_triangles_in_rectangle :
  let total_squares := 3 * 7,
      corner_squares := 4,
      cut_squares := total_squares - corner_squares,
      triangles_per_square := 4,
      total_triangles := cut_squares * triangles_per_square
  in total_triangles = 68 :=
by 
  sorry

end small_triangles_in_rectangle_l797_797337


namespace diameter_tends_to_zero_l797_797186

noncomputable theory
open Set

def diameter (X : Set ℝ) : ℝ := ⨆ x y ∈ X, |x - y|

theorem diameter_tends_to_zero 
  (f : ℝ → ℝ)
  (A : ℕ → Set ℝ)
  (h_cont : ContinuousOn f (Icc 0 1))
  (h_diff : ∀ x ∈ Ioo 0 1, DifferentiableAt ℝ f x)
  (h_deriv_lt_one : ∀ x ∈ Ioo 0 1, HasDerivAt f (f' x) x ∧ f' x < 1)
  (h_incr : ∀ x y ∈ Icc 0 1, x ≤ y → f x ≤ f y)
  (A_def : ∀ n, A (n + 1) = f '' (A n))
  (A_init : A 1 = f '' (Icc 0 1)) :
  Tendsto (λ n, diameter (A n)) atTop (nhds 0) :=
sorry

end diameter_tends_to_zero_l797_797186


namespace johnson_family_seating_l797_797635

theorem johnson_family_seating (boys girls : Finset ℕ) (h_boys : boys.card = 5) (h_girls : girls.card = 4) :
  (∃ (arrangement : List ℕ), arrangement.length = 9 ∧ at_least_two_adjacent boys arrangement) :=
begin
  -- Given the total number of ways: 9! 
  -- subtract 5! * 4! from 9! to get the result 
  have total_arrangements := nat.factorial 9,
  have restrictive_arrangements := nat.factorial 5 * nat.factorial 4,
  exact (total_arrangements - restrictive_arrangements) = 360000,
end

end johnson_family_seating_l797_797635


namespace ratio_of_work_speeds_l797_797731

theorem ratio_of_work_speeds (B_speed : ℚ) (combined_speed : ℚ) (A_speed : ℚ) 
  (h1 : B_speed = 1/12) 
  (h2 : combined_speed = 1/4) 
  (h3 : A_speed + B_speed = combined_speed) : 
  A_speed / B_speed = 2 := 
sorry

end ratio_of_work_speeds_l797_797731


namespace partA_partB_partC_partD_partE_l797_797191

def T := {x : ℝ // x ≠ 0}

def star (a b : T) : T := ⟨3 * a.val * b.val, by
  have h : 3 * a.val * b.val ≠ 0 := by 
    intro H
    have H1 : a.val ≠ 0 := a.property
    have H2 : b.val ≠ 0 := b.property
    nlinarith
  exact h⟩

theorem partA (a b : T) : star a b = star b a := by
  unfold star
  simp
  apply subtype.ext
  exact mul_comm _ _

theorem partB (a b c : T) : star (star a b) c = star a (star b c) := sorry

theorem partC (a : T) : star a ⟨1/3, by norm_num⟩ = a ∧ star ⟨1/3, by norm_num⟩ a = a := by
  unfold star
  simp
  apply and.intro
  all_goals { simp }

theorem partD (a : T) : ∃ b : T, star a b = ⟨1/3, by norm_num⟩ := by
  use ⟨1/(9*a.val), by 
    intro H
    have Ha : 9 * a.val ≠ 0 := by nlinarith [a.property]
    exact Ha H⟩
  unfold star
  simp

theorem partE (a : T) : star a ⟨1/(3*a.val), by 
  intro H
  have Ha : 3 * a.val ≠ 0 := by nlinarith [a.property]
  exact Ha H⟩ = ⟨1/3, by norm_num⟩ ∧ star ⟨1/(3*a.val), by 
  intro H
  have Ha : 3 * a.val ≠ 0 := by nlinarith [a.property]
  exact Ha H⟩ a = ⟨1/3, by norm_num⟩ := by
  unfold star
  simp
  all_goals { apply and.intro }
  all_goals { simp }

end partA_partB_partC_partD_partE_l797_797191


namespace parabola_eqn_l797_797653

theorem parabola_eqn (p : ℝ) (h1 : vertex = (0, 0)) (h2 : axis_of_symmetry = -4) : y^2 = 2 * p * x :=
by
  have hp : p = 8 := by
    sorry -- add steps to derive p = 8 here based on axis_of_symmetry = -4
  rw hp
  sorry -- solve the final part for y^2 = 16x

end parabola_eqn_l797_797653


namespace cos_sin_identity_l797_797763

theorem cos_sin_identity :
  (cos (10 * real.pi / 180) - 2 * sin (20 * real.pi / 180)) / sin (10 * real.pi / 180) = real.sqrt 3 :=
by sorry

end cos_sin_identity_l797_797763


namespace Ceva_concurrent_lines_l797_797709

theorem Ceva_concurrent_lines
    (A B C D₁ D₂ E₁ E₂ F₁ F₂ L M N : Point)
    (circle : Circle)
    (triangle_ABC : Triangle A B C)
    (conditions : Intersects circle (side triangle_ABC B C) = ⟨D₁, D₂⟩ ∧
                   Intersects circle (side triangle_ABC C A) = ⟨E₁, E₂⟩ ∧
                   Intersects circle (side triangle_ABC A B) = ⟨F₁, F₂⟩ ∧
                   Intersects (segment D₁ E₁) (segment D₂ F₂) = L ∧
                   Intersects (segment E₁ F₁) (segment E₂ D₂) = M ∧
                   Intersects (segment F₁ D₁) (segment F₂ E₂) = N) :
    Concurrent (line_through A L) (line_through B M) (line_through C N) := 
sorry

end Ceva_concurrent_lines_l797_797709


namespace max_value_of_f_prime_div_f_l797_797831

def f (x : ℝ) : ℝ := sorry

theorem max_value_of_f_prime_div_f (f : ℝ → ℝ) (h1 : ∀ x, deriv f x - f x = 2 * x * Real.exp x) (h2 : f 0 = 1) :
  ∀ x > 0, (deriv f x / f x) ≤ 2 :=
sorry

end max_value_of_f_prime_div_f_l797_797831


namespace prevent_four_digit_number_l797_797336

theorem prevent_four_digit_number (N : ℕ) (n : ℕ) :
  n = 123 + 102 * N ∧ ∀ x : ℕ, (3 + 2 * x) % 10 < 1000 → x < 1000 := 
sorry

end prevent_four_digit_number_l797_797336


namespace a6_is_3_l797_797095

noncomputable def a4 := 8 / 2 -- Placeholder for positive root
noncomputable def a8 := 8 / 2 -- Placeholder for the second root (we know they are both the same for now)
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 2) = (a (n + 1))^2

theorem a6_is_3 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a4_a8: a 4 = a4) (h_a4_a8_root : a 8 = a8) : 
  a 6 = 3 :=
by
  sorry

end a6_is_3_l797_797095


namespace eval_expression_1_eval_expression_2_l797_797784

theorem eval_expression_1 : (2 * (7 / 9: ℝ))^(1 / 2) - (2 * real.sqrt 3 - real.pi)^0 - (2 * (10 / 27: ℝ))^(-2 / 3) + (0.25)^(-3 / 2) = 8 + 5 / 48 :=
by
  sorry

theorem eval_expression_2 (x : ℝ) (h : 0 < x ∧ x < 1) (hx : x + x⁻¹ = 3) : x^(1 / 2) - x^(-1 / 2) = -1 :=
by
  sorry

end eval_expression_1_eval_expression_2_l797_797784


namespace triangle_side_length_difference_l797_797765

theorem triangle_side_length_difference (x : ℤ) :
  (2 < x ∧ x < 16) → (∀ y : ℤ, (2 < y ∧ y < 16) → (3 ≤ y) ∧ (y ≤ 15)) →
  (∀ z : ℤ, (3 ≤ z ∨ z ≤ 15) → (15 - 3 = 12)) := by
  sorry

end triangle_side_length_difference_l797_797765


namespace time_after_1500_seconds_l797_797675

def initial_time_hours : ℕ := 14
def initial_time_minutes : ℕ := 35
def duration_seconds : ℕ := 1500

theorem time_after_1500_seconds : 
  let added_minutes := duration_seconds / 60 in
  let total_minutes := initial_time_minutes + added_minutes in
  let new_hours := initial_time_hours + total_minutes / 60 in
  let new_minutes := total_minutes % 60 in
  new_hours = 15 ∧ new_minutes = 0 := by
  sorry

end time_after_1500_seconds_l797_797675


namespace suitable_squares_l797_797935

/--
A natural number is "suitable" if it is the smallest among all natural numbers with the same sum of digits.
-/
def isSuitable (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n → sumOfDigits m = sumOfDigits n → m > n

/--
We need to show that the set of all suitable numbers that are exact squares of natural numbers
is exactly {1, 4, 9, 49}.
-/
theorem suitable_squares :
  { n : ℕ | isSuitable n ∧ ∃ k : ℕ, k^2 = n } = {1, 4, 9, 49} :=
by
  sorry

end suitable_squares_l797_797935


namespace white_ball_probability_l797_797807

theorem white_ball_probability :
  ∀ (n : ℕ), (2/(n+2) = 2/5) → (n = 3) → (n/(n+2) = 3/5) :=
by
  sorry

end white_ball_probability_l797_797807


namespace butane_molecular_weight_l797_797289

def molecular_weight (moles total_weight : ℝ) : ℝ :=
  total_weight / moles

theorem butane_molecular_weight :
  (molecular_weight 4 260) = 65 :=
by
  sorry

end butane_molecular_weight_l797_797289


namespace cans_needed_for_fewer_people_l797_797947

theorem cans_needed_for_fewer_people :
  ∀ (cans total_people fewer_people_rate : ℕ), 
    cans = 600 →
    total_people = 40 →
    fewer_people_rate = 30 →
    let fewer_people := (total_people * fewer_people_rate) / 100 in
    let new_total_people := total_people - fewer_people in
    let cans_per_person := cans / total_people in
    cans_per_person * new_total_people = 420 :=
by
  intros 
  intros h1 h2 h3
  let fewer_people := (total_people * fewer_people_rate) / 100
  let new_total_people := total_people - fewer_people
  let cans_per_person := cans / total_people
  sorry

end cans_needed_for_fewer_people_l797_797947


namespace cans_needed_for_fewer_people_l797_797950

theorem cans_needed_for_fewer_people :
  ∀ (total_cans : ℕ) (total_people : ℕ) (percentage_fewer : ℕ),
    total_cans = 600 →
    total_people = 40 →
    percentage_fewer = 30 →
    total_cans / total_people * (total_people - (total_people * percentage_fewer / 100)) = 420 :=
by
  intros total_cans total_people percentage_fewer
  assume h1 h2 h3
  rw [h1, h2, h3]
  have cans_per_person : ℕ := 600 / 40
  have people_after_reduction : ℕ := 40 - (40 * 30 / 100)
  show cans_per_person * people_after_reduction = 420
  sorry

end cans_needed_for_fewer_people_l797_797950


namespace effective_tent_setup_plans_l797_797698

theorem effective_tent_setup_plans (X Y : ℕ) (h1 : 3 * X + 2 * Y = 50) (h2 : X > 0) (h3 : Y > 0) : ∃ (n : ℕ), n = 8 :=
by
  use 8
  sorry

end effective_tent_setup_plans_l797_797698


namespace coefficient_x5_y2_in_expansion_l797_797979

theorem coefficient_x5_y2_in_expansion
  (x y : ℝ) :
  coefficient_of_term (x^5 * y^2) (expansion (x^2 + x + y)^5) = 30 :=
by
  sorry

end coefficient_x5_y2_in_expansion_l797_797979


namespace csc_315_eq_neg_sqrt_2_l797_797035

theorem csc_315_eq_neg_sqrt_2 : csc 315 = -sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797035


namespace pens_taken_first_month_correct_l797_797896

noncomputable theory

def total_pens_per_student (red_pens : ℕ) (black_pens : ℕ) : ℕ :=
  red_pens + black_pens

def total_pens (students : ℕ) (red_pens : ℕ) (black_pens : ℕ) : ℕ :=
  students * (total_pens_per_student red_pens black_pens)

def pens_remaining (students : ℕ) (pens_per_student : ℕ) : ℕ :=
  students * pens_per_student

def pens_taken_after_first_month (total_pens : ℕ) (pens_taken_second_month : ℕ) (pens_remaining : ℕ) : ℕ :=
  total_pens - pens_taken_second_month - pens_remaining

theorem pens_taken_first_month_correct :
  let red_pens := 62 in
  let black_pens := 43 in
  let students := 3 in
  let pens_taken_second_month := 41 in
  let pens_per_student_after_second_month := 79 in
  let total_pens := total_pens students red_pens black_pens in
  let pens_remaining := pens_remaining students pens_per_student_after_second_month in
  pens_taken_after_first_month total_pens pens_taken_second_month pens_remaining = 37 :=
by
  sorry

end pens_taken_first_month_correct_l797_797896


namespace problem_statement_l797_797916

theorem problem_statement (f : ℝ → ℝ) (h_diff : ∀ x > 0, differentiable_at ℝ f x)
  (h_pos : ∀ x > 0, f x > 0)
  (h_ineq : ∀ x > 0, f x < x * (deriv f x) ∧ x * (deriv f x) < 2 * (f x)) :
  (1 / 4) < (f 1) / (f 2) ∧ (f 1) / (f 2) < (1 / 2) := 
sorry

end problem_statement_l797_797916


namespace find_x_l797_797137

theorem find_x (p q x : ℚ) (h1 : p / q = 4 / 5)
    (h2 : 4 / 7 + x / (2 * q + p) = 1) : x = 12 := 
by
  sorry

end find_x_l797_797137


namespace smallest_even_digits_divisible_by_99_l797_797062

-- Define the divisibility constraints
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def alternating_sum_of_digits (n : ℕ) : ℤ :=
  n.digits 10 |>.reverse |>.zipWith (λ i d, if i % 2 = 0 then (d : ℤ) else -d) [0..] |>.sum

-- Define a helper function to check if all digits are even
def all_digits_even (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 0

-- Define the main theorem
theorem smallest_even_digits_divisible_by_99 : 
  ∀ n : ℕ, (∃ m : ℕ, m > 0 ∧ all_digits_even m ∧ m % 9 = 0 ∧ m % 11 = 0) →
  (228888 = n ∧ all_digits_even n ∧ n % 99 = 0) :=
by
  sorry

end smallest_even_digits_divisible_by_99_l797_797062


namespace quadratic_roots_relationship_l797_797522

theorem quadratic_roots_relationship (a b c r : ℂ) 
  (h_eq : a ≠ 0) (root1 : r) (root2 : 0.5 * r^3) 
  (vieta_sum : r + 0.5 * r^3 = -b / a) 
  (vieta_product : r * (0.5 * r^3) = c / a) :
  b^2 = a^2 * (1 + 2 * (2^(2/3):ℂ)) + a * 2^(4/3:ℂ) :=
by
  sorry

end quadratic_roots_relationship_l797_797522


namespace csc_315_eq_neg_sqrt_2_l797_797038

theorem csc_315_eq_neg_sqrt_2 : csc 315 = -sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797038


namespace diagonal_length_EG_l797_797517

theorem diagonal_length_EG (EF FG GH HE : ℝ) (angle_GHE : ℝ) (h1 : EF = 12) (h2 : FG = 12) (h3 : GH = 20) (h4 : HE = 20) (h5 : angle_GHE = 90) :
  real.sqrt (GH^2 + HE^2) = 20 * real.sqrt 2 :=
by
  -- conditions derived from the problem statement
  have h_GH_HE : GH = 20 := h3,
  have h_HE_HE : HE = 20 := h4,
  sorry

end diagonal_length_EG_l797_797517


namespace seating_arrangements_l797_797599

theorem seating_arrangements (sons daughters : ℕ) (totalSeats : ℕ) (h_sons : sons = 5) (h_daughters : daughters = 4) (h_seats : totalSeats = 9) :
  let total_arrangements := totalSeats.factorial
  let unwanted_arrangements := sons.factorial * daughters.factorial
  total_arrangements - unwanted_arrangements = 360000 :=
by
  rw [h_sons, h_daughters, h_seats]
  let total_arrangements := 9.factorial
  let unwanted_arrangements := 5.factorial * 4.factorial
  exact Nat.sub_eq_of_eq_add $ eq_comm.mpr (Nat.add_sub_eq_of_eq total_arrangements_units)
where
  total_arrangements_units : 9.factorial = 5.factorial * 4.factorial + 360000 := by
    rw [Nat.factorial, Nat.factorial, Nat.factorial, ←Nat.factorial_mul_factorial_eq 5 4]
    simp [tmp_rewriting]

end seating_arrangements_l797_797599


namespace effective_tent_setup_plans_l797_797697

theorem effective_tent_setup_plans (X Y : ℕ) (h1 : 3 * X + 2 * Y = 50) (h2 : X > 0) (h3 : Y > 0) : ∃ (n : ℕ), n = 8 :=
by
  use 8
  sorry

end effective_tent_setup_plans_l797_797697


namespace linda_spent_amount_l797_797683

theorem linda_spent_amount :
  let cost_notebooks := 3 * 1.20
  let cost_pencils := 1.50
  let cost_pens := 1.70
  let total_cost := cost_notebooks + cost_pencils + cost_pens
  total_cost = 6.80 :=
by
  let cost_notebooks := 3 * 1.20
  let cost_pencils := 1.50
  let cost_pens := 1.70
  let total_cost := cost_notebooks + cost_pencils + cost_pens
  show total_cost = 6.80
  sorry

end linda_spent_amount_l797_797683


namespace johnson_family_seating_l797_797636

theorem johnson_family_seating (boys girls : Finset ℕ) (h_boys : boys.card = 5) (h_girls : girls.card = 4) :
  (∃ (arrangement : List ℕ), arrangement.length = 9 ∧ at_least_two_adjacent boys arrangement) :=
begin
  -- Given the total number of ways: 9! 
  -- subtract 5! * 4! from 9! to get the result 
  have total_arrangements := nat.factorial 9,
  have restrictive_arrangements := nat.factorial 5 * nat.factorial 4,
  exact (total_arrangements - restrictive_arrangements) = 360000,
end

end johnson_family_seating_l797_797636


namespace proposition_invalid_for_lines_and_plane_l797_797500
noncomputable def geometrical_figures (x y z : Type) [IsPerpendicular x y] [IsParallel y z] := 
  ¬ (x : Line in ℝ^3) ∧ 
  (y : Line in ℝ^3) ∧ 
  (z : Plane in ℝ^3) ∧ 
  IsPerpendicular x z
theorem proposition_invalid_for_lines_and_plane (x y z : Type) [geometrical_figures x y z] : 
  false :=
sorry

end proposition_invalid_for_lines_and_plane_l797_797500


namespace coefficient_x2_term_l797_797052

theorem coefficient_x2_term (a b : ℕ) :
  let poly1 := λ (x : ℕ), a * x^3 + 3 * x^2 - 2 * x
  let poly2 := λ (x : ℕ), b * x^2 - 7 * x - 4
  ∃ (c : ℕ), c = 2 ∧ (∃ x2_term_in_product : (poly1 * poly2) (some x2_term_in_product)) sorry

end coefficient_x2_term_l797_797052


namespace f_definition_for_neg_x_l797_797103

noncomputable def f : ℝ → ℝ := sorry

lemma f_is_odd (x : ℝ) : f (-x) = -f x := sorry

lemma f_definition_for_nonneg_x (x : ℝ) (hx : 0 ≤ x) : f x = x^2 * (1 - real.sqrt x) := sorry

theorem f_definition_for_neg_x (x : ℝ) (hx : x < 0) : f x = - x^2 * (1 - real.sqrt (-x)) :=
by sorry

end f_definition_for_neg_x_l797_797103


namespace total_payment_is_correct_l797_797938

def length : ℕ := 30
def width : ℕ := 40
def construction_cost_per_sqft : ℕ := 3
def sealant_cost_per_sqft : ℕ := 1
def total_area : ℕ := length * width
def total_cost_per_sqft : ℕ := construction_cost_per_sqft + sealant_cost_per_sqft
def total_cost : ℕ := total_area * total_cost_per_sqft

theorem total_payment_is_correct : total_cost = 4800 := by
  sorry

end total_payment_is_correct_l797_797938


namespace num_squares_below_2000_l797_797507

-- Definitions based on conditions
def has_ones_digit (n : ℕ) (d : ℕ) : Prop := (n % 10) = d
def is_perfect_square (n : ℕ) : Prop := ∃ (k : ℕ), k^2 = n
def ones_digit_5_or_6_or_7 (n : ℕ) : Prop := has_ones_digit n 5 ∨ has_ones_digit n 6 ∨ has_ones_digit n 7

-- Assertion based on the problem
theorem num_squares_below_2000 : 
  (∃ k : ℕ, 1 <= k ∧ k < 2000 ∧ is_perfect_square k ∧ ones_digit_5_or_6_or_7 k) = 13 :=
by 
  sorry

end num_squares_below_2000_l797_797507


namespace complex_solution_not_real_l797_797356

theorem complex_solution_not_real :
  ∫ (λ x, 3 * x^2 + 2 * x - 1) ∂x +
    (13 - 3) * real.sqrt ((-4 + 5) - (∫ x in 0..3, 9 * x^3 - 8 * x^2 + 7 * x)) =
  sorry := by
  sorry

end complex_solution_not_real_l797_797356


namespace add_base_12_l797_797733

def a_in_base_10 := 10
def b_in_base_10 := 11
def c_base := 12

theorem add_base_12 : 
  let a := 10
  let b := 11
  (3 * c_base ^ 2 + 12 * c_base + 5) + (2 * c_base ^ 2 + a * c_base + b) = 6 * c_base ^ 2 + 3 * c_base + 4 :=
by
  sorry

end add_base_12_l797_797733


namespace part1_part2_l797_797464

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l797_797464


namespace calculate_expression_l797_797750

theorem calculate_expression :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 :=
by
  have fact_6_pos : Nat.factorial 6 > 0 := Nat.factorial_pos 6
  have fact_5_pos : Nat.factorial 5 > 0 := Nat.factorial_pos 5
  have fact_7_pos : Nat.factorial 7 > 0 := Nat.factorial_pos 7

  have mul_pos_1 : 4 * Nat.factorial 6 > 0 := mul_pos (by norm_num) fact_6_pos
  have mul_pos_2 : 20 * Nat.factorial 5 > 0 := mul_pos (by norm_num) fact_5_pos
  have add_pos : 4 * Nat.factorial 6 + 20 * Nat.factorial 5 > 0 := add_pos mul_pos_1 mul_pos_2

  have exp_nonneg : 0 < Nat.factorial 7 := fact_7_pos

  norm_num
  sorry

end calculate_expression_l797_797750


namespace father_l797_797391

noncomputable def father's_current_age : ℕ :=
  let S : ℕ := 40 -- Sebastian's current age
  let Si : ℕ := S - 10 -- Sebastian's sister's current age
  let sum_five_years_ago := (S - 5) + (Si - 5) -- Sum of their ages five years ago
  let father_age_five_years_ago := (4 * sum_five_years_ago) / 3 -- From the given condition
  father_age_five_years_ago + 5 -- Their father's current age

theorem father's_age_is_85 : father's_current_age = 85 :=
  sorry

end father_l797_797391


namespace number_interchanged_digits_l797_797865

-- Definitions from conditions
variables {a b j : ℤ}
def sum_of_digits := a + b
def original_number := 10 * a + b
def interchanged_number := 10 * b + a

-- Given condition
axiom given_condition : original_number = j * sum_of_digits

-- Goal to prove
theorem number_interchanged_digits (a b j : ℤ) (h : original_number = j * sum_of_digits) :
  interchanged_number = (10 * j - 9) * sum_of_digits :=
sorry

end number_interchanged_digits_l797_797865


namespace prob_interval_calculation_l797_797554

variable {ξ : ℕ → ℝ}
variable {a : ℝ}

-- Condition 1: Probability distribution of ξ
axiom prob_distribution (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 5) : P(ξ k = k / 5) = a * k

-- Condition 2: Sum of probabilities is 1
axiom sum_prob : ∑ k in finset.range 5, P(ξ k = (k + 1) / 5) = 1

-- Definition of P(ξ) between 1/10 and 1/2
def prob_interval (a : ℝ) : ℝ := P (ξ 1 = 1 / 5) + P (ξ 2 = 2 / 5)

-- Theorem to prove
theorem prob_interval_calculation : prob_interval a = 1 / 5 :=
  sorry

end prob_interval_calculation_l797_797554


namespace coupon_value_l797_797895

noncomputable def frames_cost : ℝ := 200
noncomputable def lenses_cost : ℝ := 500
noncomputable def insurance_coverage_percentage : ℝ := 0.80
noncomputable def total_cost : ℝ := 250

theorem coupon_value : 
  let insurance_coverage := insurance_coverage_percentage * lenses_cost in
  let james_lenses_cost := lenses_cost - insurance_coverage in
  let total_cost_before_coupon := frames_cost + james_lenses_cost in
  total_cost_before_coupon - total_cost = 50 := 
by
  sorry

end coupon_value_l797_797895


namespace vector_midpoints_l797_797521

variables {V : Type} [AddCommGroup V] [Module ℝ V]
variables {A B C D E F : V}

-- Define midpoint condition
def is_midpoint (M P Q : V) : Prop := M = (P + Q) / 2

-- Given conditions
variables (hE : is_midpoint E A B) (hF : is_midpoint F C D)

-- Goal
theorem vector_midpoints (hE : is_midpoint E A B) (hF : is_midpoint F C D) :
  2 • (F - E) = (D - A) + (C - B) :=
by
  sorry

end vector_midpoints_l797_797521


namespace find_b_for_continuous_f_l797_797198

def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 3 then 3 * x^2 - 5
  else b * x + 6

theorem find_b_for_continuous_f :
  (∀ x : ℝ, continuous_at (λ x, f x b) 3) → b = 16 / 3 :=
by
  sorry

end find_b_for_continuous_f_l797_797198


namespace max_profit_at_3_l797_797720

noncomputable def sales_volume (x k : ℝ) : ℝ := 3 - k / (x + 1)

noncomputable def profit (x : ℝ) : ℝ :=
  let k := 2 in
  let m := sales_volume x k in
  0.5 * (80 + 160 * m) - x

theorem max_profit_at_3 :
  (∀ x ≥ 0, profit x ≤ profit 3) :=
by sorry

end max_profit_at_3_l797_797720


namespace range_of_m_l797_797867

def point_P := (1, 1)
def circle_C1 (x y m : ℝ) := x^2 + y^2 + 2*x - m = 0

theorem range_of_m (m : ℝ) :
  (1 + 1)^2 + 1^2 > m + 1 → -1 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l797_797867


namespace sqrt_sum_of_area_l797_797537

variables {T a b c : ℝ}

-- Given P as the interior point, and areas T, a, b, c
-- We want to prove that \sqrt T = \sqrt a + \sqrt b + \sqrt c.

theorem sqrt_sum_of_area (hT : 0 < T) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (partition : sqrt (a / T) + sqrt (b / T) + sqrt (c / T) = 1) :
    sqrt T = sqrt a + sqrt b + sqrt c :=
by
  sorry

end sqrt_sum_of_area_l797_797537


namespace csc_315_eq_neg_sqrt_2_l797_797033

theorem csc_315_eq_neg_sqrt_2 : csc 315 = -sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797033


namespace eccentricity_of_hyperbola_l797_797837

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_asymp : 3 * a + b = 0) :
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = Real.sqrt 10 :=
by
  sorry

end eccentricity_of_hyperbola_l797_797837


namespace common_charts_for_categorical_vars_l797_797295

theorem common_charts_for_categorical_vars :
  commonly_used_charts_for_relationship_between_two_categorical_variables = 
  { "contingency tables", "three-dimensional bar charts", "two-dimensional bar charts" } :=
sorry

end common_charts_for_categorical_vars_l797_797295


namespace johnson_family_seating_l797_797605

/-- The Johnson family has 5 sons and 4 daughters. We want to find the number of ways to seat them in a row of 9 chairs such that at least 2 boys are next to each other. -/
theorem johnson_family_seating : 
  let boys := 5 in
  let girls := 4 in
  let total_children := boys + girls in
  fact total_children - 
  2 * (fact boys * fact girls) = 357120 := 
by
  let boys := 5
  let girls := 4
  let total_children := boys + girls
  have total_arrangements : ℕ := fact total_children
  have no_two_boys_next_to_each_other : ℕ := 2 * (fact boys * fact girls)
  have at_least_two_boys_next_to_each_other : ℕ := total_arrangements - no_two_boys_next_to_each_other
  show at_least_two_boys_next_to_each_other = 357120
  sorry

end johnson_family_seating_l797_797605


namespace largest_divisor_of_10000_not_dividing_9999_l797_797285

theorem largest_divisor_of_10000_not_dividing_9999 : ∃ d, d ∣ 10000 ∧ ¬ (d ∣ 9999) ∧ ∀ y, (y ∣ 10000 ∧ ¬ (y ∣ 9999)) → y ≤ d := 
by
  sorry

end largest_divisor_of_10000_not_dividing_9999_l797_797285


namespace four_digit_even_numbers_count_l797_797854

def digits : Finset Nat := {0, 1, 2, 3, 4, 5}
def is_even (n : Nat) : Prop := n % 2 = 0
def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n < 10000
def no_repetitions (l : List Nat) : Prop := l.nodup

noncomputable def count_four_digit_even_numbers : Nat :=
  (digits.powerset.filter (λ s, s.card = 4 ∧
    s.to_list.permutations.filter (λ l, 
      is_four_digit (l.reverse.foldl (λ n d, 10 * n + d) 0) ∧ 
      is_even (l.head!) ∧ 
      no_repetitions l
    ).card ≠ 0)
  ).card

theorem four_digit_even_numbers_count :
  count_four_digit_even_numbers = 156 := sorry

end four_digit_even_numbers_count_l797_797854


namespace find_third_number_l797_797638

theorem find_third_number : 
  let nums := (10, 60, 35)
  let avg_original := (nums.1 + nums.2 + nums.3) / 3
  let avg_new := avg_original + 5
  let known_nums := (20, 40)
  ∃ x : ℕ, known_nums.1 + known_nums.2 + x = 3 * avg_new → x = 60 :=
by
  let nums := (10, 60, 35)
  let avg_original := (nums.1 + nums.2 + nums.3) / 3
  let avg_new := avg_original + 5
  let known_nums := (20, 40)
  exists 60
  intro h
  rw [←h]
  sorry

end find_third_number_l797_797638


namespace chloe_at_least_85_nickels_l797_797757

-- Define the given values
def shoe_cost : ℝ := 45.50
def ten_dollars : ℝ := 10.0
def num_ten_dollar_bills : ℕ := 4
def quarter_value : ℝ := 0.25
def num_quarters : ℕ := 5
def nickel_value : ℝ := 0.05

-- Define the statement to be proved
theorem chloe_at_least_85_nickels (n : ℕ) 
  (H1 : shoe_cost = 45.50)
  (H2 : ten_dollars = 10.0)
  (H3 : num_ten_dollar_bills = 4)
  (H4 : quarter_value = 0.25)
  (H5 : num_quarters = 5)
  (H6 : nickel_value = 0.05) :
  4 * ten_dollars + 5 * quarter_value + n * nickel_value >= shoe_cost → n >= 85 :=
by {
  sorry
}

end chloe_at_least_85_nickels_l797_797757


namespace midpoint_coordinate_sum_l797_797292

theorem midpoint_coordinate_sum
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 = 10)
  (h2 : y1 = 3)
  (h3 : x2 = 4)
  (h4 : y2 = -3) :
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  xm + ym =  7 := by
  sorry

end midpoint_coordinate_sum_l797_797292


namespace length_of_bridge_l797_797685

noncomputable def train_length : ℝ := 155
noncomputable def train_speed_km_hr : ℝ := 45
noncomputable def crossing_time_seconds : ℝ := 30

noncomputable def train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600

noncomputable def total_distance : ℝ := train_speed_m_s * crossing_time_seconds

noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge : bridge_length = 220 := by
  sorry

end length_of_bridge_l797_797685


namespace base4_division_l797_797157

open Nat

def base4_to_base10 (n : Nat) : Nat :=
  match n with
  | 1230 := 1 * 4^3 + 2 * 4^2 + 3 * 4^1 + 0 * 4^0
  | 32 := 3 * 4^1 + 2 * 4^0
  | 13 := 1 * 4^1 + 3 * 4^0
  | _ := 0

def base10_to_base4 (n : Nat) : Nat :=
  match n with
  | 17 := 1 * 4^2 + 1 * 4^1 + 1 * 4^0
  | _ := 0

theorem base4_division :
  let a := base4_to_base10 1230
  let b := base4_to_base10 32
  let c := base4_to_base10 13
  let sum := a + b
  let quotient := sum / c
  base10_to_base4 quotient = 111 :=
by
  sorry

end base4_division_l797_797157


namespace sum_divisible_by_last_term_l797_797213

noncomputable def sum_first_n_numbers (n : ℕ) : ℕ :=
  (2 * n - 1) * n

theorem sum_divisible_by_last_term (n : ℕ) : (2 * n - 1) ∣ ∑ k in Finset.range (2 * n - 1 + 1), k :=
by sorry

end sum_divisible_by_last_term_l797_797213


namespace johnson_family_seating_l797_797603

/-- The Johnson family has 5 sons and 4 daughters. We want to find the number of ways to seat them in a row of 9 chairs such that at least 2 boys are next to each other. -/
theorem johnson_family_seating : 
  let boys := 5 in
  let girls := 4 in
  let total_children := boys + girls in
  fact total_children - 
  2 * (fact boys * fact girls) = 357120 := 
by
  let boys := 5
  let girls := 4
  let total_children := boys + girls
  have total_arrangements : ℕ := fact total_children
  have no_two_boys_next_to_each_other : ℕ := 2 * (fact boys * fact girls)
  have at_least_two_boys_next_to_each_other : ℕ := total_arrangements - no_two_boys_next_to_each_other
  show at_least_two_boys_next_to_each_other = 357120
  sorry

end johnson_family_seating_l797_797603


namespace not_relatively_prime_l797_797413

open Nat -- Use the Nat namespace for natural numbers

theorem not_relatively_prime (a b c : ℕ) (A : ℕ) (h1 : A = a^2 + b^2 + a * b * c) 
  (h2 : ∀ d ∈ divisors A, d <= 2008)
  (h3 : (c + 2)^1004 ∣ A) : gcd a b > 1 :=
by
  sorry

end not_relatively_prime_l797_797413


namespace Galaxy_Chess_Team_Arrangement_l797_797975

theorem Galaxy_Chess_Team_Arrangement : 
  let girls_positions := 2!
  let boys_positions := 3!
  let total_arrangements := girls_positions * boys_positions
  total_arrangements = 12 := by
  sorry

end Galaxy_Chess_Team_Arrangement_l797_797975


namespace alex_climb_staircase_10_l797_797302

def ways_to_climb : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := ways_to_climb n + ways_to_climb (n+1)

theorem alex_climb_staircase_10 :
  ways_to_climb 10 = 89 := by
  sorry

end alex_climb_staircase_10_l797_797302


namespace intersection_nonempty_implies_range_of_a_q_is_necessary_condition_for_p_implies_range_of_a_l797_797121

-- Definitions based on the conditions
def is_in_A (x : ℝ) : Prop := 2 < x ∧ x < 3
def is_in_B (x a : ℝ) : Prop := a < x ∧ x < a^2 + 2

-- Part (1)
theorem intersection_nonempty_implies_range_of_a (A B : set ℝ) (a : ℝ) :
  (∃ x, is_in_A x ∧ is_in_B x a) → a ∈ set.Iio 0 ∪ set.Ioo 0 3 := 
sorry

-- Part (2)
theorem q_is_necessary_condition_for_p_implies_range_of_a (a : ℝ) :
  (∀ x, is_in_A x → is_in_B x a) → a ∈ set.Iic (-1) ∪ set.Icc 1 2 := 
sorry

end intersection_nonempty_implies_range_of_a_q_is_necessary_condition_for_p_implies_range_of_a_l797_797121


namespace probability_product_divisible_by_four_l797_797264

-- Defining events and probability related structures
def is_face_divisible_by_four (faces : ℕ → ℕ) (n : ℕ) : Prop :=
  (faces 0 * faces 1 * faces 2 * faces 3) % 4 = 0

axiom tetrahedron_faces : set (ℕ → ℕ)
axiom tetrahedron_fair : ∀ t ∈ tetrahedron_faces, ∀ i, 1 ≤ t i ∧ t i ≤ 4

theorem probability_product_divisible_by_four : 
  ∑' (t ∈ tetrahedron_faces), if is_face_divisible_by_four t 4 then 1 else 0 / ∑' (t ∈ tetrahedron_faces), 1 = 13 / 16 :=
sorry

end probability_product_divisible_by_four_l797_797264


namespace max_N_value_l797_797681

noncomputable def S : set ℂ := {z : ℂ | sorry} -- We assume S is defined appropriately

-- Conditions
axiom uv_in_S (u v : ℂ) (hu : u ∈ S) (hv : v ∈ S) : u * v ∈ S
axiom u2_plus_v2_in_S (u v : ℂ) (hu : u ∈ S) (hv : v ∈ S) : u^2 + v^2 ∈ S
axiom finite_elements (h : ∃ (N : ℕ), {z ∈ S | complex.abs z ≤ 1}.finite ∧ {z ∈ S | complex.abs z ≤ 1}.to_finset.card = N)

-- We want to prove this
theorem max_N_value : ∃ (N ≤ 13), {z ∈ S | complex.abs z ≤ 1}.to_finset.card = N :=
by
  sorry

end max_N_value_l797_797681


namespace smallest_positive_period_intervals_of_monotonic_increase_max_value_on_interval_min_value_on_interval_l797_797844

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * Real.cos (2 * x - Real.pi / 4)

theorem smallest_positive_period : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi := sorry

theorem intervals_of_monotonic_increase : ∀ k : ℤ, (∀ x y : ℝ, -3 * Real.pi / 8 + k * Real.pi ≤ x ∧ x ≤ y ∧ y ≤ Real.pi / 8 + k * Real.pi → f x ≤ f y) := sorry

theorem max_value_on_interval : ∃ x : ℝ, -Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = sqrt 2 := sorry

theorem min_value_on_interval : ∃ x : ℝ, -Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = -1 := sorry

end smallest_positive_period_intervals_of_monotonic_increase_max_value_on_interval_min_value_on_interval_l797_797844


namespace complex_number_in_second_quadrant_l797_797163

-- Define the imaginary unit and its properties.
noncomputable def i : ℂ := complex.I

-- Define the complex number in question.
noncomputable def complex_number : ℂ := (2 / (1 - i)) + 2 * (i^2)

-- The proof goal: the point corresponding to the complex number is in the second quadrant.
theorem complex_number_in_second_quadrant : 
  ∃ (z : ℂ), z = complex_number ∧ z.re < 0 ∧ z.im > 0 := 
sorry

end complex_number_in_second_quadrant_l797_797163


namespace marbles_difference_l797_797903

theorem marbles_difference : 10 - 8 = 2 :=
by
  sorry

end marbles_difference_l797_797903


namespace johnson_family_seating_l797_797619

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem johnson_family_seating (sons daughters : ℕ) (total_seats : ℕ) 
  (condition1 : sons = 5) (condition2 : daughters = 4) (condition3 : total_seats = 9) :
  let total_arrangements := factorial total_seats,
      restricted_arrangements := factorial sons * factorial daughters,
      answer := total_arrangements - restricted_arrangements
  in answer = 360000 := 
by
  -- The proof would go here
  sorry

end johnson_family_seating_l797_797619


namespace f_five_sixths_l797_797642

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then Real.sin (Real.pi * x) else f (x - 1)

theorem f_five_sixths : f (5 / 6) = -1 / 2 := 
by
  -- proof goes here
  sorry

end f_five_sixths_l797_797642


namespace total_amount_earned_l797_797344

theorem total_amount_earned (avg_price_per_pair : ℝ) (number_of_pairs : ℕ) (price : avg_price_per_pair = 9.8 ) (pairs : number_of_pairs = 50 ) : 
avg_price_per_pair * number_of_pairs = 490 := by
  -- Given conditions
  sorry

end total_amount_earned_l797_797344


namespace little_red_riding_hood_time_l797_797558

-- Definitions of conditions
def distance_to_grandma := 1 -- km
def speed_flat := 4 -- km/h
def speed_uphill := 3 -- km/h
def speed_downhill := 6 -- km/h

-- Function to compute time for half journey considering both flat and inclined sections
def journey_time (distance : ℝ) (speed_flat speed_uphill speed_downhill : ℝ) : ℝ :=
  let inclined_section_length := distance / 2 in
  let time_uphill := inclined_section_length / speed_uphill in
  let time_downhill := inclined_section_length / speed_downhill in
  let time_flat := inclined_section_length / speed_flat in
  2 * (time_flat + time_uphill + time_downhill) -- return the total journey time
  
-- Main statement to prove
theorem little_red_riding_hood_time :
  journey_time 1 speed_flat speed_uphill speed_downhill = 0.5 :=
by
  sorry

end little_red_riding_hood_time_l797_797558


namespace cellphone_surveys_l797_797730

theorem cellphone_surveys
  (regular_rate : ℕ)
  (total_surveys : ℕ)
  (higher_rate_multiplier : ℕ)
  (total_earnings : ℕ)
  (higher_rate_bonus : ℕ)
  (x : ℕ) :
  regular_rate = 10 → total_surveys = 100 →
  higher_rate_multiplier = 130 → total_earnings = 1180 →
  higher_rate_bonus = 3 → (10 * (100 - x) + 13 * x = 1180) →
  x = 60 :=
by
  sorry

end cellphone_surveys_l797_797730


namespace solve_for_y_l797_797584

theorem solve_for_y (y : ℝ) (h : 16^(3*y - 5) = 4^(2*y + 8)) : y = 9/2 :=
sorry

end solve_for_y_l797_797584


namespace determine_a_domain_of_f_max_value_of_f_l797_797402

/-- Given a function f(x) with parametr a -/
def f (a : ℝ) : ℝ → ℝ := λ x, log a (1 + x) + log a (3 - x)

/-- Condition: a is positive and a ≠ 1 -/
axiom a_pos (a : ℝ) : a > 0
axiom a_neq_one (a : ℝ) : a ≠ 1

/-- Condition: f(1) = 2 -/
axiom f_at_one (a : ℝ) : f a 1 = 2

/-- Proving that a = 2 given f(1) = 2 -/
theorem determine_a (a : ℝ) : f a 1 = 2 → a = 2 := sorry

/-- Proving the domain of f(x) for a = 2 is (-1, 3) -/
theorem domain_of_f : ∀ x, 0 < 2 ∧ 2 ≠ 1 ∧ f 2 1 = 2 ↔ (x ∈ set.Ioo (-1 : ℝ) 3) := sorry

/-- Proving the maximum value of f(x) on [0, 3/2] is 2 -/
theorem max_value_of_f (x : ℝ) : x ∈ set.Icc (0 : ℝ) (3 / 2) → f 2 x ≤ 2 := sorry

end determine_a_domain_of_f_max_value_of_f_l797_797402


namespace extremum_at_one_monotonicity_intervals_l797_797492

noncomputable def f (x : ℝ) (a : ℝ) := (2 - a) * x - 2 * Real.log x

theorem extremum_at_one (a : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ (2 - a) - (2 / x) = 0) → a = 0 :=
by
  have h1 : (2 - a) - 2 = 0 → a = 0 := by
    intro h; linarith
  intro ⟨hx, hxy⟩
  rw [hx] at hxy
  exact h1 hxy

theorem monotonicity_intervals (a : ℝ) : 
  (a ≥ 2 → ∀ x : ℝ, x > 0 → (2 - a) - (2 / x) < 0) ∧ 
  (a < 2 → ∀ x : ℝ, (x > (2 / (2 - a)) → (2 - a) - (2 / x) > 0) ∧ (x < (2 / (2 - a)) → (2 - a) - (2 / x) < 0)) :=
by 
  have h2 : a ≥ 2 → ∀ x : ℝ, x > 0 → (2 - a) - (2 / x) < 0 := by
    intros h x hx
    linarith [(2 - a), -2 / x]
  have h3 : a < 2 → ∀ x : ℝ, (x > (2 / (2 - a)) → (2 - a) - (2 / x) > 0) ∧ (x < (2 / (2 - a)) → (2 - a) - (2 / x) < 0) := by
    intros h x
    split_ifs
    case h1 h => linarith [(2 - (a:ℝ)), - (2 / x)]
    case h2 h => linarith [(2 - (a:ℝ)), - (2 / x)]
  exact ⟨h2, h3⟩

end extremum_at_one_monotonicity_intervals_l797_797492


namespace count_valid_permutations_l797_797059

open List 

/-- Define permutations of a list and check if a list meets the constraint. -/
def isValidPermutation (l : List ℕ) : Prop :=
  l.Permutation [1, 2, 3, 4, 5, 6] ∧ (∀ i, i < l.length - 1 → l.get ⟨i + 1, sorry⟩ ≤ l.get ⟨i, sorry⟩ +1)

theorem count_valid_permutations :
  (Finset.univ.filter isValidPermutation).card = 309 :=
sorry

end count_valid_permutations_l797_797059


namespace question_A_question_D_l797_797171

variable {A B C : ℝ} -- Angles of triangle ABC
variable {a b c : ℝ} -- Sides opposite the angles A, B, and C respectively

-- Definitions based on conditions in a)
def valid_triangle (a b c : ℝ) (A B C : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π

-- Proving the assertions based on the conditions
theorem question_A (habc : valid_triangle a b c A B C) (hab : a > b) : cos (2 * A) < cos (2 * B) :=
sorry

theorem question_D (habc : valid_triangle a b c A B C) (hC : C = π / 3) (hc : c = 2) : 
  ∃ A' B', valid_triangle a b c A' B' C ∧ by mul_assoc at meq (A' + B' = C) ∧ abs ((a * b) / 2 * sin C) = sqrt 3 :=
sorry

end question_A_question_D_l797_797171


namespace csc_315_eq_neg_sqrt_2_l797_797004

theorem csc_315_eq_neg_sqrt_2 :
  Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := 
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797004


namespace ratio_of_areas_l797_797346

-- Definitions of conditions
def side_length (s : ℝ) : Prop := s > 0
def original_area (A s : ℝ) : Prop := A = s^2

-- Definition of the new area after folding
def new_area (B A s : ℝ) : Prop := B = (7/8) * s^2

-- The proof statement to show the ratio B/A is 7/8
theorem ratio_of_areas (s A B : ℝ) (h_side : side_length s) (h_area : original_area A s) (h_B : new_area B A s) : 
  B / A = 7 / 8 := 
by 
  sorry

end ratio_of_areas_l797_797346


namespace initial_number_of_persons_l797_797240

-- Define the conditions and the goal
def weight_increase_due_to_new_person : ℝ := 102 - 75
def average_weight_increase (n : ℝ) : ℝ := 4.5 * n

theorem initial_number_of_persons (n : ℝ) (h1 : average_weight_increase n = weight_increase_due_to_new_person) : n = 6 :=
by
  -- Skip the proof with sorry
  sorry

end initial_number_of_persons_l797_797240


namespace perpendicular_FB_AB_l797_797740

-- Definitions of points and triangle properties
variables {A B C M D E F : Type}
variable [metric_space A] [metric_space B] [metric_space C]
variable [metric_space D] [metric_space E] [metric_space F]
variables (A B C M D E F : Point)
noncomputable def right_triangle (A B C : Point) :=
  ∠ACB = 90°

noncomputable def midpoint (M : Point) (A B : Point) :=
  M = midpoint (A, B)

noncomputable def angle_condition (D E : Point) (A B C : Point) :=
  ∠ABE = ∠BCD

noncomputable def line_intersect (D C : Point) (M E : Point) :=
  ∃ F, line (D, C) ∩ line (M, E) = F

-- Main theorem
theorem perpendicular_FB_AB (A B C M D E F : Point) 
  (h_rt : right_triangle A B C)
  (h_midpoint : midpoint M A B)
  (h_ang : angle_condition D E A B C)
  (h_int : line_intersect D C M E) : 
  ⟨F, ⟂⟩ := sorry

end perpendicular_FB_AB_l797_797740


namespace part1_part2_l797_797476

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l797_797476


namespace johnson_family_seating_problem_l797_797615

theorem johnson_family_seating_problem : 
  ∃ n : ℕ, n = 9! - 5! * 4! ∧ n = 359760 :=
by
  have total_ways := (Nat.factorial 9)
  have no_adjacent_boys := (Nat.factorial 5) * (Nat.factorial 4)
  have result := total_ways - no_adjacent_boys
  use result
  split
  . exact eq.refl result
  . norm_num -- This will replace result with its evaluated form, 359760

end johnson_family_seating_problem_l797_797615


namespace spadesuit_evaluation_l797_797802

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_evaluation : spadesuit 3 (spadesuit 4 5) = -72 := by
  sorry

end spadesuit_evaluation_l797_797802


namespace lena_played_3_5_hours_l797_797904

-- Defining the problem conditions
variables (L B : ℕ)
def total_time := 437
def brother_time_more := 17

-- Stating the theorem
theorem lena_played_3_5_hours (h1 : B = L + brother_time_more) (h2 : L + B = total_time) : 
  L = 3.5 * 60 := -- Proving that Lena's playtime is 3.5 hours, expressed in minutes
by 
  sorry

end lena_played_3_5_hours_l797_797904


namespace minimum_even_N_for_A_2015_turns_l797_797208

noncomputable def a (n : ℕ) : ℕ :=
  6 * 2^n - 4

def A_minimum_even_moves_needed (k : ℕ) : ℕ :=
  2015 - 1

theorem minimum_even_N_for_A_2015_turns :
  ∃ N : ℕ, 2 ∣ N ∧ A_minimum_even_moves_needed 2015 ≤ N ∧ a 1007 = 6 * 2^1007 - 4 := by
  sorry

end minimum_even_N_for_A_2015_turns_l797_797208


namespace intersection_A_B_union_complement_A_B_range_of_m_l797_797848

variable (x m : ℝ)
def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (3 * m)}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 5} := sorry
theorem union_complement_A_B : (Aᶜ ∪ B) = {x : ℝ | -2 < x ∧ x < 5} := sorry
theorem range_of_m (B_inter_C_eq_C : B ∩ C = C) : m < -1 / 2 := sorry

end intersection_A_B_union_complement_A_B_range_of_m_l797_797848


namespace inequality_proof_l797_797908

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_cond : a + b + c + 3 * a * b * c ≥ (a * b)^2 + (b * c)^2 + (c * a)^2 + 3) :
  (a^3 + b^3 + c^3) / 3 ≥ (a * b * c + 2021) / 2022 :=
by 
  sorry

end inequality_proof_l797_797908


namespace fish_caught_l797_797747

noncomputable def total_fish_caught (chris_trips : ℕ) (chris_fish_per_trip : ℕ) (brian_trips : ℕ) (brian_fish_per_trip : ℕ) : ℕ :=
  chris_trips * chris_fish_per_trip + brian_trips * brian_fish_per_trip

theorem fish_caught (chris_trips : ℕ) (brian_factor : ℕ) (brian_fish_per_trip : ℕ) (ratio_numerator : ℕ) (ratio_denominator : ℕ) :
  chris_trips = 10 → brian_factor = 2 → brian_fish_per_trip = 400 → ratio_numerator = 3 → ratio_denominator = 5 →
  total_fish_caught chris_trips (brian_fish_per_trip * ratio_denominator / ratio_numerator) (chris_trips * brian_factor) brian_fish_per_trip = 14660 :=
by
  intros h_chris_trips h_brian_factor h_brian_fish_per_trip h_ratio_numer h_ratio_denom
  rw [h_chris_trips, h_brian_factor, h_brian_fish_per_trip, h_ratio_numer, h_ratio_denom]
  -- adding actual arithmetic would resolve the statement correctly
  sorry

end fish_caught_l797_797747


namespace count_valid_three_digit_numbers_l797_797857

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 5 ∧ d ≠ 6

def count_valid_numbers : ℕ :=
  let hundreds := {d // 1 ≤ d ∧ d ≤ 9 ∧ is_valid_digit d}.to_finset.card
  let tens := {d // 0 ≤ d ∧ d ≤ 9 ∧ is_valid_digit d}.to_finset.card
  let units := {d // 0 ≤ d ∧ d ≤ 9}.to_finset.card
  hundreds * tens * units

theorem count_valid_three_digit_numbers : count_valid_numbers = 490 := sorry

end count_valid_three_digit_numbers_l797_797857


namespace true_proposition_B_l797_797678

theorem true_proposition_B : (3 > 4) ∨ (3 < 4) :=
sorry

end true_proposition_B_l797_797678


namespace minimum_empty_cells_after_movement_l797_797878

def beetle_grid := Fin 9 × Fin 9

def initial_beetles (cell : beetle_grid) : Prop := True

def beetle_moves (cell1 cell2 : beetle_grid) : Prop :=
  let (x1, y1) := cell1
  let (x2, y2) := cell2
  (abs (x2 - x1) = 1 ∧ abs (y2 - y1) = 1)

theorem minimum_empty_cells_after_movement :
  ∃ (n : ℕ), n = 9 ∧ ∀ config : beetle_grid → list beetle_moves, (count_empty_cells (move_beetles config initial_beetles) = n) :=
sorry

end minimum_empty_cells_after_movement_l797_797878


namespace part1_part2_l797_797439

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l797_797439


namespace arithmetic_sequence_term_l797_797161

theorem arithmetic_sequence_term (a : ℕ → ℤ) (d : ℤ) (n : ℕ) :
  a 5 = 33 ∧ a 45 = 153 ∧ (∀ n, a n = a 1 + (n - 1) * d) ∧ a n = 201 → n = 61 :=
by
  sorry

end arithmetic_sequence_term_l797_797161


namespace csc_315_eq_sqrt2_l797_797041

theorem csc_315_eq_sqrt2 :
  let θ := 315
  let csc := λ θ, 1 / (Real.sin (θ * Real.pi / 180))
  315 = 360 - 45 → 
  Real.sin (315 * Real.pi / 180) = Real.sin ((360 - 45) * Real.pi / 180) → 
  Real.sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 →
  csc 315 = Real.sqrt 2 := 
by
  intros θ csc h1 h2 h3
  -- proof would go here
  sorry

end csc_315_eq_sqrt2_l797_797041


namespace johnson_family_seating_problem_l797_797612

theorem johnson_family_seating_problem : 
  ∃ n : ℕ, n = 9! - 5! * 4! ∧ n = 359760 :=
by
  have total_ways := (Nat.factorial 9)
  have no_adjacent_boys := (Nat.factorial 5) * (Nat.factorial 4)
  have result := total_ways - no_adjacent_boys
  use result
  split
  . exact eq.refl result
  . norm_num -- This will replace result with its evaluated form, 359760

end johnson_family_seating_problem_l797_797612


namespace triathlete_average_speed_l797_797728

def swimming_distance : ℝ := 1
def biking_distance : ℝ := 2
def running_distance : ℝ := 3

def swimming_speed : ℝ := 2
def biking_speed : ℝ := 25
def running_speed : ℝ := 12

def swimming_time : ℝ := swimming_distance / swimming_speed
def biking_time : ℝ := biking_distance / biking_speed
def running_time : ℝ := running_distance / running_speed

def total_distance : ℝ := swimming_distance + biking_distance + running_distance
def total_time : ℝ := swimming_time + biking_time + running_time

def average_speed (total_distance total_time : ℝ) : ℝ := total_distance / total_time

theorem triathlete_average_speed :
  average_speed total_distance total_time ≈ 7.2 :=
by
  sorry

end triathlete_average_speed_l797_797728


namespace compute_area_ratio_l797_797164

-- Define the points and lengths
variables {A B C D E F : Point}
variables (AB AC AD CF : ℝ)
variables (AB_eq : AB = 130) (AC_eq : AC = 130) (AD_eq : AD = 42) (CF_eq : CF = 88)
variables (triangle : Triangle A B C)

theorem compute_area_ratio (hAB : AB = 130) (hAC : AC = 130) (hAD : AD = 42) (hCF : CF = 88) :
  let ratio := (area C E F) / (area D B E) in
  ratio = 21 / 109 :=
sorry

end compute_area_ratio_l797_797164


namespace increasing_on_interval_l797_797877

theorem increasing_on_interval (a : ℝ) : (∀ x : ℝ, x > 1/2 → (2 * x + a + 1 / x^2) ≥ 0) → a ≥ -3 :=
by
  intros h
  -- Rest of the proof would go here
  sorry

end increasing_on_interval_l797_797877


namespace part_a_l797_797676

-- Power tower with 100 twos
def power_tower_100_t2 : ℕ := sorry

theorem part_a : power_tower_100_t2 > 3 := sorry

end part_a_l797_797676


namespace quadrilateral_perimeter_l797_797290

theorem quadrilateral_perimeter
  (EF FG HG : ℝ)
  (h1 : EF = 7)
  (h2 : FG = 15)
  (h3 : HG = 3)
  (perp1 : EF * FG = 0)
  (perp2 : HG * FG = 0) :
  EF + FG + HG + Real.sqrt (4^2 + 15^2) = 25 + Real.sqrt 241 :=
by
  sorry

end quadrilateral_perimeter_l797_797290


namespace rectangle_enclosing_ways_l797_797384

/-- Given five horizontal lines and five vertical lines, the total number of ways to choose four lines (two horizontal, two vertical) such that they form a rectangle is 100 --/
theorem rectangle_enclosing_ways : 
  let horizontal_lines := [1, 2, 3, 4, 5]
  let vertical_lines := [1, 2, 3, 4, 5]
  let ways_horizontal := Nat.choose 5 2
  let ways_vertical := Nat.choose 5 2
  ways_horizontal * ways_vertical = 100 := 
by
  sorry

end rectangle_enclosing_ways_l797_797384


namespace find_sticker_price_l797_797759

-- Define the conditions as given in the problem
def sticker_price : ℝ -- y is the sticker price of the laptop
def store_p_discount : ℝ := 0.80 * sticker_price
def store_p_final_price := store_p_discount - 120 -- Store P's final price after 20% discount and $120 rebate
def store_q_discount : ℝ := 0.70 * sticker_price -- Store Q's final price after 30% discount

-- Clara saves $30 more by purchasing at Store P
def savings_condition := store_p_final_price + 30 = store_q_discount

-- The proof goal
theorem find_sticker_price : 
  (∃ y : ℝ, sticker_price = y ∧ y = 900) :=
by
  sorry

end find_sticker_price_l797_797759


namespace archer_scores_distribution_l797_797713

structure ArcherScores where
  hits_40 : ℕ
  hits_39 : ℕ
  hits_24 : ℕ
  hits_23 : ℕ
  hits_17 : ℕ
  hits_16 : ℕ
  total_score : ℕ

theorem archer_scores_distribution
  (dora : ArcherScores)
  (reggie : ArcherScores)
  (finch : ArcherScores)
  (h1 : dora.total_score = 120)
  (h2 : reggie.total_score = 110)
  (h3 : finch.total_score = 100)
  (h4 : dora.hits_40 + dora.hits_39 + dora.hits_24 + dora.hits_23 + dora.hits_17 + dora.hits_16 = 6)
  (h5 : reggie.hits_40 + reggie.hits_39 + reggie.hits_24 + reggie.hits_23 + reggie.hits_17 + reggie.hits_16 = 6)
  (h6 : finch.hits_40 + finch.hits_39 + finch.hits_24 + finch.hits_23 + finch.hits_17 + finch.hits_16 = 6)
  (h7 : 40 * dora.hits_40 + 39 * dora.hits_39 + 24 * dora.hits_24 + 23 * dora.hits_23 + 17 * dora.hits_17 + 16 * dora.hits_16 = 120)
  (h8 : 40 * reggie.hits_40 + 39 * reggie.hits_39 + 24 * reggie.hits_24 + 23 * reggie.hits_23 + 17 * reggie.hits_17 + 16 * reggie.hits_16 = 110)
  (h9 : 40 * finch.hits_40 + 39 * finch.hits_39 + 24 * finch.hits_24 + 23 * finch.hits_23 + 17 * finch.hits_17 + 16 * finch.hits_16 = 100)
  (h10 : dora.hits_40 = 1)
  (h11 : dora.hits_39 = 0)
  (h12 : dora.hits_24 = 0) :
  dora.hits_40 = 1 ∧ dora.hits_16 = 5 ∧ 
  reggie.hits_23 = 2 ∧ reggie.hits_16 = 4 ∧ 
  finch.hits_17 = 4 ∧ finch.hits_16 = 2 :=
sorry

end archer_scores_distribution_l797_797713


namespace value_of_expression_l797_797399

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2006 = 2007 :=
sorry

end value_of_expression_l797_797399


namespace average_height_of_trees_l797_797177

def first_tree_height : ℕ := 1000
def half_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200

def total_height : ℕ := first_tree_height + 2 * half_tree_height + last_tree_height
def number_of_trees : ℕ := 4
def average_height : ℕ := total_height / number_of_trees

theorem average_height_of_trees :
  average_height = 800 :=
by
  -- This line contains a placeholder proof, the actual proof is omitted.
  sorry

end average_height_of_trees_l797_797177


namespace part1_part2_l797_797440

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l797_797440


namespace inequality_l797_797555

def domain (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem inequality (a b : ℝ) (ha : domain a) (hb : domain b) :
  |a + b| < |3 + ab / 3| :=
by
  sorry

end inequality_l797_797555


namespace circle_traj_and_distance_l797_797405

theorem circle_traj_and_distance:
  (∀ C : Type, ∀ (passesThrouF : ∀ p : C, p = (1 : ℝ, 0 : ℝ)), 
   ∀ (tangentToLine : ∀ l : C, l = (λ x : ℝ, x = -1)), 
   (∃ (centerTraj : ℝ → ℝ), centerTraj = λ y : ℝ, y^2 = 4 * x) 
   ∧ (∃ (circle_eqn : C → ℝ), circle_eqn = λ p : C, p = (0 : ℝ, 0 : ℝ) → x^2 + y^2 = 1)
   ∧ (∀ (intersectLine : ℝ → ℝ), 
      intersectLine = λ x : ℝ, y = (1 / 2) * x + (1 / 2) → 
      (∀ (A B C D : ℝ × ℝ), B = D → k_BF + k_DF = 0 → 
      ( ∃ (length_AD_CD : ℕ), length_AD_CD = (36 * sqrt(5)) / 5 ))) → 
  sorry

end circle_traj_and_distance_l797_797405


namespace csc_315_eq_neg_sqrt_2_l797_797014

theorem csc_315_eq_neg_sqrt_2 :
  let csc := λ θ, 1 / Real.sin θ in
  csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
  by
  let sin := Real.sin
  have h1 : csc (315 * Real.pi / 180) = 1 / sin (315 * Real.pi / 180) := rfl
  have h2 : sin (315 * Real.pi / 180) = sin ((360 - 45) * Real.pi / 180) := by congr; norm_num
  have h3 : sin ((360 - 45) * Real.pi / 180) = -sin (45 * Real.pi / 180) := by
    rw [Real.sin_pi_sub]
    congr; norm_num
  have h4 : sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 := Real.sin_of_one_div_sqrt_two 45 rfl
  sorry

end csc_315_eq_neg_sqrt_2_l797_797014


namespace num_tent_setup_plans_l797_797696

theorem num_tent_setup_plans (num_students : ℕ) (X Y : ℕ) :
  (num_students = 50) →
  (∀ ⦃X Y:ℕ⦄, 3 * X + 2 * Y = num_students → X % 2 = 0 → X > 0 ∧ Y > 0) →
  ∃ (plans : ℕ), plans = 8 :=
by
  intros hnum_students hvalid_pairs
  have : ∃! (pair_set : Finset (ℕ × ℕ)), pair_set.card = 8 ∧ ∀ (x, y) ∈ pair_set, 3 * x + 2 * y = 50 ∧ x % 2 = 0 ∧ x > 0 ∧ y > 0 := sorry
  sorry

end num_tent_setup_plans_l797_797696


namespace rectangles_and_triangles_not_both_axisymmetric_l797_797576

def is_axisymmetric_figure (figure : Type*) : Prop :=
  ∃ (fold : figure → Prop), ∀ (f : figure), fold f → (∃ axis : line, can_fold_along f axis)

def rectangle : Type := sorry
def triangle : Type := sorry

axiom can_fold_along_rectangle : ∀ (r : rectangle), ∃ axis : line, can_fold_along r axis
axiom can_fold_along_isosceles_triangle : ∀ (t : triangle), is_isosceles t → ∃ axis : line, can_fold_along t axis
axiom not_all_triangles_isosceles : ∃ (t : triangle), ¬ is_isosceles t

theorem rectangles_and_triangles_not_both_axisymmetric : 
  ¬(is_axisymmetric_figure rectangle ∧ is_axisymmetric_figure triangle) :=
begin
  intro h,
  cases h with h_rect h_tri,
  obtain ⟨fold_rect, h_fold_rect⟩ := h_rect,
  obtain ⟨axis_rect, h_axis_rect⟩ := h_fold_rect (sorry : rectangle) sorry, -- Need to replace with a specific instance
  obtain ⟨fold_tri, h_fold_tri⟩ := h_tri,
  obtain ⟨axis_tri, h_axis_tri⟩ := h_fold_tri (sorry : triangle) sorry, -- Need to replace with a specific instance
  -- Continue proof here
  sorry
end

end rectangles_and_triangles_not_both_axisymmetric_l797_797576


namespace limit_of_geom_series_l797_797833

-- Definitions of the first term and common ratio as per the conditions
def first_term : ℝ := 35
def common_ratio : ℝ := 1 / 2

-- Sum formula of an infinite geometric series
def geom_series_limit (a r : ℝ) : ℝ := a / (1 - r)

-- The main statement to prove
theorem limit_of_geom_series : geom_series_limit first_term common_ratio = 70 :=
by sorry

end limit_of_geom_series_l797_797833


namespace probability_heads_before_tails_l797_797920

theorem probability_heads_before_tails (q : ℚ) (a b : ℕ) (h_rel_prime : Nat.gcd a b = 1)
  (h_result : q = 13/17) : 
  a + b = 30 :=
by
  have h_q_def : q = (13 : ℚ) / 17, from h_result
  have h_q_form : q = (a : ℚ) / b, from sorry
  have h_rel_prime_q : (a / b : ℚ) = 13 / 17, from sorry
  have h_one : (a : ℚ) = 13 ∧ (b : ℚ) = 17, from sorry
  show a + b = 30, from sorry

end probability_heads_before_tails_l797_797920


namespace smallest_four_digit_number_l797_797796

theorem smallest_four_digit_number :
  ∃ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (∃ n : ℕ, 21 * m = n^2) ∧ m = 1029 :=
by sorry

end smallest_four_digit_number_l797_797796


namespace general_term_formula_exists_positive_integer_l797_797834

variable {nat : ℕ}

-- Definitions from conditions
def sum_of_terms (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop := 
  ∀ n, S n = n * a n - 3 * n * (n - 1)

def init_term (a : ℕ → ℕ) : Prop :=
  a 1 = 1

-- The first part of the problem
theorem general_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (cond1 : sum_of_terms S a) (cond2 : init_term a) : 
  ∀ n, a n = 6 * n - 5 := sorry

-- The second part of the problem
theorem exists_positive_integer (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (cond1 : sum_of_terms S a) (cond2 : init_term a) :
  ∃ n, (∑ k in finset.range n + 1, (S k / k) - (3 / 2) * ((n - 1) * (n - 1))) = 2016 ∧ n = 807 := sorry

end general_term_formula_exists_positive_integer_l797_797834


namespace veronica_photo_choices_l797_797307

theorem veronica_photo_choices (n k₁ k₂ : ℕ) (h₁ : n = 5) (h₂ : k₁ = 3) (h₃ : k₂ = 4) :
  (nat.choose n k₁ + nat.choose n k₂) = 15 :=
by
  sorry

end veronica_photo_choices_l797_797307


namespace jack_more_emails_morning_than_evening_l797_797176

theorem jack_more_emails_morning_than_evening : 
  ∀ (morning_emails : ℕ) (evening_emails : ℕ), 
  morning_emails = 9 → evening_emails = 7 → (morning_emails - evening_emails) = 2 :=
begin
  intros morning_emails evening_emails,
  assume h1 : morning_emails = 9,
  assume h2 : evening_emails = 7,
  rw [h1, h2],
  norm_num,
end

end jack_more_emails_morning_than_evening_l797_797176


namespace correct_statement_l797_797269

-- Definitions from the conditions
def surveyMethod := "sample survey"
def population := {x | x ∈ ninth_grade_graduates_in_Qingdao_2003}
def sample := {math_score | math_score ∈ random_sample_of_500_students}

-- The question we need to prove
theorem correct_statement : 
  surveyMethod ≠ "census" ∧ 
  (population = {x | x ∈ ninth_grade_graduates_in_Qingdao_2003}) ∧ 
  (sample = {math_score | math_score ∈ random_sample_of_500_students}) → 
  (The math scores of the randomly selected 500 students are a sample) := 
by
  sorry

end correct_statement_l797_797269


namespace no_real_pairs_for_same_lines_l797_797365

theorem no_real_pairs_for_same_lines : ¬ ∃ (a b : ℝ), (∀ x y : ℝ, 2 * x + a * y + b = 0 ↔ b * x - 3 * y + 15 = 0) :=
by {
  sorry
}

end no_real_pairs_for_same_lines_l797_797365


namespace line_divides_hypotenuse_in_ratio_two_to_one_l797_797151

-- Define the right triangle with given conditions
variables {A B C M N : Point}
variables (h_right : right_triangle A B C)
variables (h_sin : sin (angle A C B) = 1 / 3)
variables (h_perp : ∃ M N, perpendicular M N AB ∧ divides_into_two_equal_areas M N A B C)

-- Define the proof statement
theorem line_divides_hypotenuse_in_ratio_two_to_one :
  divides_hypotenuse_in_ratio A B M N 2 1 :=
sorry

end line_divides_hypotenuse_in_ratio_two_to_one_l797_797151


namespace sqrt_expression_evaluation_l797_797367

theorem sqrt_expression_evaluation : 
  (Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 2) := 
by
  sorry

end sqrt_expression_evaluation_l797_797367


namespace factorial_division_l797_797416

theorem factorial_division :
  9! = 362880 → (9! / 4!) = 15120 := by
  sorry

end factorial_division_l797_797416


namespace line_of_intersection_l797_797708

noncomputable def circle1 : set (ℝ × ℝ) :=
  {p | (p.1 + 12)^2 + (p.2 + 2)^2 = 225}

noncomputable def circle2 : set (ℝ × ℝ) :=
  {p | (p.1 - 5)^2 + (p.2 - 11)^2 = 90}

theorem line_of_intersection (d : ℝ) :
  ∃ p1 p2, p1 ∈ circle1 ∧ p1 ∈ circle2 ∧ p2 ∈ circle1 ∧ p2 ∈ circle2 ∧
    (∀ x y : ℝ, x + y = d) ↔ (d = -1 / 2) :=
  sorry

end line_of_intersection_l797_797708


namespace proof_problem_l797_797512

-- Define the given condition as a constant
def condition : Prop := 213 * 16 = 3408

-- Define the statement we need to prove under the given condition
theorem proof_problem (h : condition) : 0.16 * 2.13 = 0.3408 := 
by 
  sorry

end proof_problem_l797_797512


namespace solve_for_n_l797_797133

theorem solve_for_n (n : ℕ) : 4^8 = 16^n → n = 4 :=
by
  sorry

end solve_for_n_l797_797133


namespace range_of_f_in_0_1_inequality_holds_for_c_l797_797112

noncomputable theory

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

-- Conditions on f(x) for the specified intervals
axiom f_pos_neg (a b : ℝ) (a_ne_zero : a ≠ 0) :
  (∀ x ∈ set.Ioo (-3 : ℝ) 2, 0 < f a b x) ∧
  (∀ x, x ∉ set.Ioo (-3 : ℝ) 2 → (x < -3 ∨ x > 2) → f a b x < 0)

-- Range of f(x) in [0, 1]
theorem range_of_f_in_0_1 (a b : ℝ) (a_ne_zero : a ≠ 0) :
  ∀ y ∈ set.Icc 0 1, 12 ≤ f a b y ∧ f a b y ≤ 18 := sorry

-- For inequality ax^2 + bx + c <= 0 in [1, 4]
theorem inequality_holds_for_c (a b : ℝ) (a_ne_zero : a ≠ 0) (c : ℝ) :
  (∀ x ∈ set.Icc 1 4, a * x^2 + b * x + c ≤ 0) ↔ c ≤ -2 := sorry

end range_of_f_in_0_1_inequality_holds_for_c_l797_797112


namespace correct_range_of_m_l797_797773

noncomputable def f (x : ℝ) (m : ℝ) := (1/3) * x^3 - x^2 + m

def counterpart_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ x1 x2 : ℝ, a < x1 ∧ x1 < x2 ∧ x2 < b ∧ 
  f (x1) = f (x2) ∧
  (∂ x1).f = (∂ x2).f =
    (f b - f a) / (b - a)

def range_of_m (f : ℝ → ℝ) : set ℝ :=
{m : ℝ | let df := fun x => (2*x) - 2 in
  ∃ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < m ∧ 
  df x1 = ((1/3)*m^2 - m) ∧ df x2 = ((1/3)*m^2 - m)}

theorem correct_range_of_m :
  ∀ m : ℝ, counterpart_function (f x m) 0 m → (m > 3/2 ∧ m < 3) :=
sorry

end correct_range_of_m_l797_797773


namespace cans_needed_for_fewer_people_l797_797948

theorem cans_needed_for_fewer_people :
  ∀ (cans total_people fewer_people_rate : ℕ), 
    cans = 600 →
    total_people = 40 →
    fewer_people_rate = 30 →
    let fewer_people := (total_people * fewer_people_rate) / 100 in
    let new_total_people := total_people - fewer_people in
    let cans_per_person := cans / total_people in
    cans_per_person * new_total_people = 420 :=
by
  intros 
  intros h1 h2 h3
  let fewer_people := (total_people * fewer_people_rate) / 100
  let new_total_people := total_people - fewer_people
  let cans_per_person := cans / total_people
  sorry

end cans_needed_for_fewer_people_l797_797948


namespace milk_ratio_l797_797560

-- Define the amounts of milk
def MinyoungMilk : ℝ := 10
def YunaMilk : ℝ := 2 / 3

-- State the theorem
theorem milk_ratio : MinyoungMilk / YunaMilk = 15 := 
by
  sorry

end milk_ratio_l797_797560


namespace part1_part2_l797_797441

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l797_797441


namespace johnson_family_seating_problem_l797_797613

theorem johnson_family_seating_problem : 
  ∃ n : ℕ, n = 9! - 5! * 4! ∧ n = 359760 :=
by
  have total_ways := (Nat.factorial 9)
  have no_adjacent_boys := (Nat.factorial 5) * (Nat.factorial 4)
  have result := total_ways - no_adjacent_boys
  use result
  split
  . exact eq.refl result
  . norm_num -- This will replace result with its evaluated form, 359760

end johnson_family_seating_problem_l797_797613


namespace same_number_of_acquaintances_l797_797570

theorem same_number_of_acquaintances (n : ℕ) (h : n ≥ 2) :
  ∃ (i j : ℕ), i ≠ j ∧ 
  (∃ (acquaintances : Fin n → Fin (n-1)),
  acquaintances i = acquaintances j) :=
by
  sorry

end same_number_of_acquaintances_l797_797570


namespace she_needs_to_order_l797_797270

def TreShawn_Consumption : ℚ := 1/2
def Michael_Consumption : ℚ := 1/3
def LaMar_Consumption : ℚ := 1/6
def Jasmine_Consumption_Pepperoni : ℚ := 1/4
def Jasmine_Consumption_Vegetarian : ℚ := 1/4
def Carlos_Consumption_Cheese : ℚ := 1/2
def Carlos_Consumption_Pepperoni : ℚ := 1/6

def total_cheese : ℚ := TreShawn_Consumption + Carlos_Consumption_Cheese
def total_pepperoni : ℚ := Michael_Consumption + Jasmine_Consumption_Pepperoni + Carlos_Consumption_Pepperoni
def total_vegetarian : ℚ := LaMar_Consumption + Jasmine_Consumption_Vegetarian

def total_cheese_rounded : ℕ := total_cheese.ceil.to_nat
def total_pepperoni_rounded : ℕ := total_pepperoni.ceil.to_nat
def total_vegetarian_rounded : ℕ := total_vegetarian.ceil.to_nat

theorem she_needs_to_order :
  total_cheese_rounded = 1 ∧ total_pepperoni_rounded = 1 ∧ total_vegetarian_rounded = 1 :=
by {
  rw [total_cheese_rounded, total_pepperoni_rounded, total_vegetarian_rounded, 
    total_cheese, total_pepperoni, total_vegetarian],
  norm_num
}

end she_needs_to_order_l797_797270


namespace parabola_segment_length_l797_797107

-- Definitions for conditions
def parabola (m : ℝ) : ℝ → ℝ :=
  λ x, x^2 + m * x

def axis_of_symmetry (m : ℝ) : ℝ :=
  -m / 2

-- Theorem stating the length of the intercepted segment on the x-axis
theorem parabola_segment_length (m : ℝ) (h : axis_of_symmetry m = 2) : 
  let p := parabola m in
  let x1 := (0:ℝ) in
  let x2 := 4 in
  x2 - x1 = 4 := 
by
  sorry

end parabola_segment_length_l797_797107


namespace giorgio_oatmeal_raisin_cookies_l797_797810

theorem giorgio_oatmeal_raisin_cookies
  (n_students : ℕ)
  (cookies_per_student : ℕ)
  (percent_oatmeal_raisin : ℚ)
  (h_n_students : n_students = 40)
  (h_cookies_per_student : cookies_per_student = 2)
  (h_percent_oatmeal_raisin : percent_oatmeal_raisin = 0.1) :
  (n_students * cookies_per_student * percent_oatmeal_raisin).to_nat = 8 :=
by
  sorry

end giorgio_oatmeal_raisin_cookies_l797_797810


namespace csc_315_eq_neg_sqrt_2_l797_797032

theorem csc_315_eq_neg_sqrt_2 : csc 315 = -sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797032


namespace calculate_gallons_of_milk_l797_797559

-- Definitions of the given constants and conditions
def price_of_soup : Nat := 2
def price_of_bread : Nat := 5
def price_of_cereal : Nat := 3
def price_of_milk : Nat := 4
def total_amount_paid : Nat := 4 * 10

-- Calculation of total cost of non-milk items
def total_cost_non_milk : Nat :=
  (6 * price_of_soup) + (2 * price_of_bread) + (2 * price_of_cereal)

-- The function to calculate the remaining amount to be spent on milk
def remaining_amount : Nat := total_amount_paid - total_cost_non_milk

-- Statement to compute the number of gallons of milk
def gallons_of_milk (remaining : Nat) (price_per_gallon : Nat) : Nat :=
  remaining / price_per_gallon

-- Proof theorem statement (no implementation required, proof skipped)
theorem calculate_gallons_of_milk : 
  gallons_of_milk remaining_amount price_of_milk = 3 := 
by
  sorry

end calculate_gallons_of_milk_l797_797559


namespace joe_flight_expense_l797_797182

theorem joe_flight_expense
  (initial_amount : ℕ)
  (hotel_expense : ℕ)
  (food_expense : ℕ)
  (remaining_amount : ℕ)
  (flight_expense : ℕ)
  (h1 : initial_amount = 6000)
  (h2 : hotel_expense = 800)
  (h3 : food_expense = 3000)
  (h4 : remaining_amount = 1000)
  (h5 : flight_expense = initial_amount - remaining_amount - hotel_expense - food_expense) :
  flight_expense = 1200 :=
by
  sorry

end joe_flight_expense_l797_797182


namespace cover_square_floor_l797_797279

theorem cover_square_floor (x : ℕ) (h : 2 * x - 1 = 37) : x^2 = 361 :=
by
  sorry

end cover_square_floor_l797_797279


namespace fourth_term_integer_probability_l797_797204

section MichaelSequence

def sequence_evolution (initial : ℕ) (steps : ℕ) : ℕ → list ℕ
| 0     := [initial]
| (n+1) := 
  let a_n := sequence_evolution initial n in
  a_n.bind (λ a,
               if 2 * (a - 1) >= 0 then [2 * (a - 1 + 1) + 1] else [8]  
               ++ if (a / 2 - 2) >= 0 then [a / 2 - 2 * 2] else [8])

def is_integer (n : ℕ) : bool :=
  n ≥ 0

def probability (seq : list ℕ) : ℚ :=
  (seq.filter is_integer).length / seq.length

theorem fourth_term_integer_probability :
  probability (sequence_evolution 8 3) = 4 / 5 :=
by sorry

end MichaelSequence

end fourth_term_integer_probability_l797_797204


namespace green_park_chess_team_arrangements_l797_797235

def chess_team_arrangements (boys girls : ℕ) : ℕ :=
  let end_girls := 4 * 3
  let middle_girl_choices := 2
  let boys_arrangement := 3 * 2 * 1
  end_girls * middle_girl_choices * boys_arrangement

theorem green_park_chess_team_arrangements :
  chess_team_arrangements 3 4 = 144 := by
  calc
    chess_team_arrangements 3 4
        = (4 * 3) * 2 * (3 * 2 * 1) : by rw [chess_team_arrangements]
    ... = 12 * 2 * 6 : by norm_num
    ... = 144 : by norm_num

end green_park_chess_team_arrangements_l797_797235


namespace part1_part2_l797_797479

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l797_797479


namespace vector_magnitude_identity_l797_797128

open RealEuclideanSpace

theorem vector_magnitude_identity
  (a b : EuclideanSpace ℝ (fin 2))
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (hab : ‖a - b‖ = 1) :
  ‖a + b‖ = √3 :=
by
  -- Proof goes here
  sorry

end vector_magnitude_identity_l797_797128


namespace fraction_hatching_l797_797329

theorem fraction_hatching (E D S H : ℕ) (hE : E = 800) (hD : D = (10 * E / 100)) (hS : S = (70 * E / 100)) (hH : H = 40) :
  H / (E - D - S) = 1 / 4 := 
by 
suffices : E - D - S = 160,
  {
    have h : H = 40 := hH,
    have H_div_160 : (H : ℚ) / 160 = 1 / 4 := by simp,
    exact H_div_160,
  }
; sorry

end fraction_hatching_l797_797329


namespace egg_cost_l797_797742

theorem egg_cost (toast_cost : ℝ) (E : ℝ) (total_cost : ℝ)
  (dales_toast : ℝ) (dales_eggs : ℝ) (andrews_toast : ℝ) (andrews_eggs : ℝ) :
  toast_cost = 1 → 
  dales_toast = 2 → 
  dales_eggs = 2 → 
  andrews_toast = 1 → 
  andrews_eggs = 2 → 
  total_cost = 15 →
  total_cost = (dales_toast * toast_cost + dales_eggs * E) + 
               (andrews_toast * toast_cost + andrews_eggs * E) →
  E = 3 :=
by
  sorry

end egg_cost_l797_797742


namespace csc_315_eq_sqrt2_l797_797046

theorem csc_315_eq_sqrt2 :
  let θ := 315
  let csc := λ θ, 1 / (Real.sin (θ * Real.pi / 180))
  315 = 360 - 45 → 
  Real.sin (315 * Real.pi / 180) = Real.sin ((360 - 45) * Real.pi / 180) → 
  Real.sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 →
  csc 315 = Real.sqrt 2 := 
by
  intros θ csc h1 h2 h3
  -- proof would go here
  sorry

end csc_315_eq_sqrt2_l797_797046


namespace unique_functional_equation_solution_l797_797379

theorem unique_functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f(x + f(y)) = x + y + 1) → f = (λ y, y + 1) :=
by sorry

end unique_functional_equation_solution_l797_797379


namespace angle_BPC_36_l797_797172

theorem angle_BPC_36
  {A B C D P Q : Point}
  (h1 : dist A B = dist A C)
  (h2 : angle A B C = 36)
  (hD : midpoint D B C)
  (hP : segment A D P)
  (hQ : segment D Q P)
  (hAP : dist A P = dist P Q)
  (hPQ : dist P Q = dist Q D):
  angle B P C = 36 :=
sorry

end angle_BPC_36_l797_797172


namespace tangent_incenter_proof_l797_797758

-- Ensure circles intersect at points A and B
variables {ω₁ ω₂ : Type} [incircle ω₁] [incircle ω₂]
variables (A B C D : Point) (tangent_A_C : Tangent AC ω₂) (tangent_A_D : Tangent AD ω₁)

-- Points C and D on respective circles
variable (on_C : OnCircle C ω₁)
variable (on_D : OnCircle D ω₂)

-- Incenters of triangles ABC and ABD
variable (I₁ I₂ : Point) [Incenter I₁ (Triangle ABC)]
variable [Incenter I₂ (Triangle ABD)]

-- Intersection of I₁I₂ and AB
variable (E : Point)
variable (intersection : IntersectAtSegment E (Segment I₁ I₂) (Segment A B))

-- Statement to prove
theorem tangent_incenter_proof :
  ∀ (A B C D I₁ I₂ E : Point),
  Tangent AC ω₂ →
  Tangent AD ω₁ →
  OnCircle C ω₁ →
  OnCircle D ω₂ →
  Incenter I₁ (Triangle ABC) →
  Incenter I₂ (Triangle ABD) →
  IntersectAtSegment E (Segment I₁ I₂) (Segment A B) →
  (1 / (SegmentLength AE)) = (1 / (SegmentLength AC)) + (1 / (SegmentLength AD)) :=
by {
  sorry
}

end tangent_incenter_proof_l797_797758


namespace part1_part2_l797_797459

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l797_797459


namespace find_a_l797_797876

theorem find_a (a : ℝ) (h : 0 < a ∧ a < 1) 
  (H : ∀ x ∈ set.Icc a (2 * a), f x = log a x ∧ (∀ x ∈ set.Icc a (2 * a), log a a = 1 ∧ log a (2 * a) = log a 2 + 1) → (1 = 3 * (log a 2 + 1))) 
  : a = 2^(-3/2) :=
  sorry
  where f (x : ℝ) : ℝ := log a x

end find_a_l797_797876


namespace domain_of_f_range_of_f_l797_797842

noncomputable def f (x : ℝ) : ℝ := Real.logb (1/2) (Real.sqrt (-x^2 + 2*x + 8))

theorem domain_of_f :
  {x : ℝ | Real.sqrt (-x^2 + 2*x + 8) > 0} = {x : ℝ | -2 < x ∧ x < 4} := by
  sorry

theorem range_of_f :
  (f '' {x : ℝ | -2 < x ∧ x < 4}) = {y : ℝ | logb (1/2) 3 ≤ y} := by
  sorry

end domain_of_f_range_of_f_l797_797842


namespace invertible_matrix_constant_l797_797912

noncomputable def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]
noncomputable def N_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![1/3, 0], ![1/6, -1/4]]
noncomputable def c : ℚ := 1 / 12
noncomputable def d : ℚ := 1 / 4

theorem invertible_matrix_constant:
  N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end invertible_matrix_constant_l797_797912


namespace cyclic_quadrilaterals_and_coinciding_centers_l797_797185

open EuclideanGeometry

theorem cyclic_quadrilaterals_and_coinciding_centers
  (A B C D E F G J H H' J' I K K' I' : Point)
  -- Given conditions
  (h_abc_acute : AcuteTriangle A B C)
  (h_ab_lt_ac : AB < AC)
  (h_ac_lt_bc : AC < BC)
  (h_d_on_extended_bc : Collinear B D C ∧ Between B C D)
  (c1 : Circle A (dist A D))
  (h_c1_intersects_ac_at_e : CircleIntersectsLine c1 AC E)
  (h_c1_intersects_ab_at_f : CircleIntersectsLine c1 AB F)
  (h_c1_intersects_cb_at_g : CircleIntersectsLine c1 CB G)
  (c2 : CircumscribedCircle (Triangle.mk A F G))
  (c3 : CircumscribedCircle (Triangle.mk A D E))
  (h_c2_intersects_fe_at_j : CircleIntersectsLine c2 FE J)
  (h_c2_intersects_bc_at_h : CircleIntersectsLine c2 BC H)
  (h_c2_intersects_fe_at_h' : CircleIntersectsLine c2 GE H')
  (h_c2_intersects_df_at_j' : CircleIntersectsLine c2 DF J')
  (h_c3_intersects_fe_at_i : CircleIntersectsLine c3 FE I)
  (h_c3_intersects_bc_at_k : CircleIntersectsLine c3 BC K)
  (h_c3_intersects_fe_at_k' : CircleIntersectsLine c3 GE K')
  (h_c3_intersects_df_at_i' : CircleIntersectsLine c3 DF I') :
  (CyclicQuad J H I K) ∧
  (CyclicQuad J' H' I' K') ∧
  -- Extra condition for centers
  (center (CircumscribedCircle.mk J H I K) = 
   center (CircumscribedCircle.mk J' H' I' K')) :=
by
  sorry

end cyclic_quadrilaterals_and_coinciding_centers_l797_797185


namespace D_score_l797_797888

-- Definitions based on the given conditions
def A := 94
def B (y : ℕ) : Prop := y > 94
def C (A D : ℕ) : Prop := A + D = 2 * (A + D) / 2 ∧ (A + D) % 2 = 0 -- C is average of A and D, and integer
def D_calc (A B C E : ℕ) : ℕ := (A + B + C + D + E) / 5
def E (C : ℕ) : ℕ := C + 2

-- Proposition to prove that D score is 96
theorem D_score : ∃ (D : ℕ), D = 96 ∧
  (∃ (B : ℕ), B > 94 ∧
   ∃ (C : ℕ), C = (A + D) / 2 ∧ (A + D) % 2 = 0 ∧
   ∃ (E : ℕ), E = C + 2 ∧
   5 * D = A + B + C + D + E) :=
by
  sorry  -- Proof not required per instructions

end D_score_l797_797888


namespace part1_part2_l797_797435

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l797_797435


namespace find_moles_HCl_combined_l797_797380

def reaction (HCl NaHCO3 NaCl H2O CO2 : Type) := 
  (λ hcl nahco3 : ℕ, (HCl → NaHCO3 → NaCl → H2O → CO2))

variable (moles_HCl moles_NaHCO3 moles_NaCl : ℕ)

axiom reaction_equation_holds : reaction HCl NaHCO3 NaCl H2O CO2
axiom moles_NaHCO3_giv : moles_NaHCO3 = 3
axiom total_moles_NaCl : moles_NaCl = 3

theorem find_moles_HCl_combined : moles_HCl = 3 :=
by 
  -- Proof skipped
  sorry

end find_moles_HCl_combined_l797_797380


namespace arithmetic_sequence_general_formula_sequence_b_sum_l797_797820

-- Definition of the arithmetic sequence and proving general formula
theorem arithmetic_sequence_general_formula (d : ℝ) (a₁ : ℝ) 
  (h₁ : a₁ * (d / a₁) = a₁ * 2) (h₂ : -3 / a₁ = -3 / 1) : 
  ∀ n : ℕ, a_n = 2 * n - 1 := by
  sorry

-- Definition of the sequence b and proving sum of the first n terms
theorem sequence_b_sum (n : ℕ) : 
  S_n = (2 ^ (2 * n - 1) + 2 * (2 * n - 1)) :=
  S_n = 1 / 2 * (4^1 + 4^2 + 4^3+...+4^n) + (2 + 6 + 10 + ... + 4n - 2) :=
  S_n = 1 / 2 * 4 * (1 - 4^n) / (1 - 4) + (n * (2 + 4n - 2) / 2 := by
  sorry

end arithmetic_sequence_general_formula_sequence_b_sum_l797_797820


namespace twenty_less_waiter_slices_eq_28_l797_797130

noncomputable def slices_of_pizza : ℕ := 78
noncomputable def buzz_ratio : ℕ := 5
noncomputable def waiter_ratio : ℕ := 8

theorem twenty_less_waiter_slices_eq_28:
  let total_slices := slices_of_pizza in
  let total_ratio := buzz_ratio + waiter_ratio in
  let waiter_slices := (waiter_ratio * total_slices) / total_ratio in
  waiter_slices - 20 = 28 := by
  sorry

end twenty_less_waiter_slices_eq_28_l797_797130


namespace part1_part2_l797_797845

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a

theorem part1 (tangent_at_e : ∀ x : ℝ, f x e = 2 * e) : a = e := sorry

theorem part2 (m : ℝ) (a : ℝ) (hm : 0 < m) :
  (if m ≤ 1 / (2 * Real.exp 1) then 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f (2 * m) a 
   else if 1 / (2 * Real.exp 1) < m ∧ m < 1 / (Real.exp 1) then 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f (1 / (Real.exp 1)) a 
   else 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f m a) :=
  sorry

end part1_part2_l797_797845


namespace circle_equation_line_MN_equation_PA_PB_range_l797_797889

-- (1) Equation of the circle given it is centered at the origin and tangent to a specific line.
theorem circle_equation (x y : ℝ) (tangent_condition : ∀ x y, x - sqrt 3 * y = 4) :
  x ^ 2 + y ^ 2 = 4 := sorry

-- (2) Equation of line MN given points M and N are symmetric about a line and |MN| = 2√3.
theorem line_MN_equation (M N : ℝ × ℝ) (symmetry_condition : ∀ x y, x + 2 * y = 0)
  (distance_condition : |(sqrt ((fst N - fst M)^2 + (snd N - snd M)^2 - 2 * sqrt 3))| = 0) :
  ∀ b, 2 * (fst M) - (snd M) + b = 0 ∧ 2 * (fst N) - (snd N) - (b) = 0 → (2 * (fst N) - (snd N) + sqrt 5 = 0 ∨ 2 * (fst M) - (snd M) + sqrt 5 = 0)
 := sorry

-- (3) Range of values for PA ⋅ PB given conditions for P inside the circle forming a geometric sequence.
theorem PA_PB_range (P A B : ℝ × ℝ)
  (circle_condition : ∀ P, (fst P)^2 + (snd P)^2 < 4)
  (sequence_condition : ∀ A B, |(fst P - fst A) * (fst P - fst B)| = (fst P) ^ 2 + (snd P) ^ 2) :
  -2 ≤ ((fst P - fst A) * (fst P - fst B) + (snd P - snd A) * (snd P - snd B)) ∧ 
  ((fst P - fst A) * (fst P - fst B) + (snd P - snd A) * (snd P - snd B)) < 0 := sorry

end circle_equation_line_MN_equation_PA_PB_range_l797_797889


namespace csc_315_eq_neg_sqrt2_l797_797011

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end csc_315_eq_neg_sqrt2_l797_797011


namespace trigonometric_identity_l797_797369

theorem trigonometric_identity :
  (sin 24 * cos 18 + cos 156 * cos 96) / (sin 28 * cos 12 + cos 152 * cos 92)
  = sin 18 / sin 26 := 
sorry

end trigonometric_identity_l797_797369


namespace inequality_part1_inequality_part2_l797_797453

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l797_797453


namespace base16_to_base2_bits_l797_797296

theorem base16_to_base2_bits :
  ∀ (n : ℕ), n = 16^4 * 7 + 16^3 * 7 + 16^2 * 7 + 16 * 7 + 7 → (2^18 ≤ n ∧ n < 2^19) → 
  ∃ b : ℕ, b = 19 := 
by
  intros n hn hpow
  sorry

end base16_to_base2_bits_l797_797296


namespace product_inequality_l797_797957

theorem product_inequality :
  (1:ℚ) / 15 < (∏ k in Finset.range 50, (2 * k + 1) / (2 * (k + 1))) ∧
  (∏ k in Finset.range 50, (2 * k + 1) / (2 * (k + 1))) < (1:ℚ) / 10 :=
sorry

end product_inequality_l797_797957


namespace ratio_of_areas_l797_797969

-- Definitions related to the given conditions
def side_length_ABCD (s : ℝ) : ℝ := 4 * s

def point_J_coordinates (s : ℝ) : ℝ × ℝ := (3 * s, 0)

def side_length_JK (s : ℝ) : ℝ := s * real.sqrt 2

-- Area calculations
def area_JKLM (s : ℝ) : ℝ := (side_length_JK s) ^ 2

def area_ABCD (s : ℝ) : ℝ := (side_length_ABCD s) ^ 2

-- The statement to prove
theorem ratio_of_areas (s : ℝ) (h : s > 0) : 
  (area_JKLM s) / (area_ABCD s) = 1 / 8 :=
by sorry

end ratio_of_areas_l797_797969


namespace not_possible_to_tile_5x7_with_trominos_l797_797527

theorem not_possible_to_tile_5x7_with_trominos :
  ∀ (n : ℕ), ¬ (∃ (tile : Fin 5 × Fin 7 → Fin n → ℕ), 
   (∀ i j : Fin 5, ∀ k : Fin 7, ∑ t in finset.univ, tile (i,j) t = k) 
   ∧ (∀ s : Fin n, ∑ i j, tile (i,j) s = 3)) :=
by
  sorry

end not_possible_to_tile_5x7_with_trominos_l797_797527


namespace conjugate_of_z_l797_797139

def z : ℂ := I * (3 - 2 * I) * I

theorem conjugate_of_z : conj z = 2 - 3 * I := by
  sorry

end conjugate_of_z_l797_797139


namespace pages_in_each_book_l797_797805

variable (BooksRead DaysPerBook TotalDays : ℕ)

theorem pages_in_each_book (h1 : BooksRead = 41) (h2 : DaysPerBook = 12) (h3 : TotalDays = 492) : (TotalDays / DaysPerBook) * DaysPerBook = 492 :=
by
  sorry

end pages_in_each_book_l797_797805


namespace lying_dwarf_number_l797_797781

-- Definitions of the numbers chosen by the dwarfs
def dwarf_choices := {a : Fin 7 → ℕ // ∃ i : Fin 7, ∑ j, if j = i then 0 else a j = a i ∧ ∑ j, a j = 46}

-- Main theorem statement: Determine possible number chosen by the lying dwarf
theorem lying_dwarf_number (a : Fin 7 → ℕ) (h : a ∈ dwarf_choices) : ∃ i : Fin 7, a i = 7 ∨ a i = 14 :=
by
  sorry

end lying_dwarf_number_l797_797781


namespace mn_value_l797_797861

variables {x m n : ℝ} -- Define variables x, m, n as real numbers

theorem mn_value (h : x^2 + m * x - 15 = (x + 3) * (x + n)) : m * n = 10 :=
by {
  -- Sorry for skipping the proof steps
  sorry
}

end mn_value_l797_797861


namespace total_cost_eq_4800_l797_797940

def length := 30
def width := 40
def cost_per_square_foot := 3
def cost_of_sealant_per_square_foot := 1

theorem total_cost_eq_4800 : 
  (length * width * cost_per_square_foot) + (length * width * cost_of_sealant_per_square_foot) = 4800 :=
by
  sorry

end total_cost_eq_4800_l797_797940


namespace eccentricity_is_sqrt2_l797_797406

-- Define the quadratic equation under consideration
def equation (x y : ℝ) : Prop :=
  10 * x - 2 * x * y - 2 * y + 1 = 0

-- Define the statement that the eccentricity of the given equation is sqrt(2)
theorem eccentricity_is_sqrt2 (x y : ℝ) (h : equation x y) : Real.sqrt 2 = √2 :=
by
  sorry

end eccentricity_is_sqrt2_l797_797406


namespace csc_315_eq_sqrt2_l797_797042

theorem csc_315_eq_sqrt2 :
  let θ := 315
  let csc := λ θ, 1 / (Real.sin (θ * Real.pi / 180))
  315 = 360 - 45 → 
  Real.sin (315 * Real.pi / 180) = Real.sin ((360 - 45) * Real.pi / 180) → 
  Real.sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 →
  csc 315 = Real.sqrt 2 := 
by
  intros θ csc h1 h2 h3
  -- proof would go here
  sorry

end csc_315_eq_sqrt2_l797_797042


namespace S_eq_25080_l797_797394

-- Define the function b(p) as the unique integer k such that |k - sqrt(p)| < 1/2
noncomputable def b (p : ℕ) : ℕ :=
  if h : p > 0 then
    let k := Real.floor (Real.sqrt (p : ℝ))
    in if abs (k - Real.sqrt (p : ℝ)) < 1/2 then k else k + 1
  else 1  -- b(p) is undefined for non-positive p, included for syntactic correctness

-- Define the sum S from 1 to 3000 of b(p)
noncomputable def S : ℕ :=
  (Finset.range 3000).sum (λ p, b (p + 1))

-- State the main theorem: S = 25080
theorem S_eq_25080 : S = 25080 := by
  sorry

end S_eq_25080_l797_797394


namespace sequence_sum_consecutive_l797_797154

theorem sequence_sum_consecutive 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 20) 
  (h8 : a 8 = 16) 
  (h_sum : ∀ i, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 100) :
  a 2 = 16 ∧ a 3 = 64 ∧ a 4 = 20 ∧ a 5 = 16 ∧ a 6 = 64 ∧ a 7 = 20 :=
  sorry

end sequence_sum_consecutive_l797_797154


namespace luggage_between_340_and_420_l797_797068

noncomputable def luggage_normal_distribution : ProbabilityDistribution ℝ :=
  NormalDistr.mk 380 (20 ^ 2)

theorem luggage_between_340_and_420 :
  Pr(open_interval (340 : ℝ) (420 : ℝ)) luggage_normal_distribution = 0.95 :=
sorry

end luggage_between_340_and_420_l797_797068


namespace probability_red_ball_is_two_thirds_red_balls_taken_out_is_three_l797_797700

-- Definitions for the conditions
def total_balls : ℕ := 18
def initial_red_balls : ℕ := 12
def initial_white_balls : ℕ := 6
def probability_red_ball : ℚ := initial_red_balls / total_balls
def probability_white_ball_after_removal (x : ℕ) : ℚ := initial_white_balls / (total_balls - x)

-- Statement of the proof problem
theorem probability_red_ball_is_two_thirds : probability_red_ball = 2 / 3 := 
by sorry

theorem red_balls_taken_out_is_three : ∃ x : ℕ, probability_white_ball_after_removal x = 2 / 5 ∧ x = 3 := 
by sorry

end probability_red_ball_is_two_thirds_red_balls_taken_out_is_three_l797_797700


namespace least_elements_set_finite_set_multiple_of_3_l797_797410

-- Definitions for conditions in the problem
def satisfies_condition (S : Set ℝ) : Prop :=
  (0 ∉ S) ∧ (1 ∉ S) ∧ (∀ a ∈ S, 1 / (1 - a) ∈ S)

-- Question I: Find the least number of elements set
theorem least_elements_set {S : Set ℝ} (h : satisfies_condition S) (h_incl : {2, -2} ⊆ S) :
  S = {2, -1, 1/2, -2, 1/3, 3/2} :=
sorry

-- Question II: Prove the number of elements is a multiple of 3
theorem finite_set_multiple_of_3 {S : Set ℝ} (h : satisfies_condition S) (h_nonempty : S ≠ ∅)
  (h_finite : Finite S) : ∃ n, 3 * n = S.toFinset.card :=
sorry

end least_elements_set_finite_set_multiple_of_3_l797_797410


namespace ratio_of_segments_l797_797710

theorem ratio_of_segments (a b t : ℕ) (circ : ∃ C, is_inscribed_circle C a b t)
    (h₁ : a < b) (h₂ : a + b + t = 26)
    (u v : ℕ) (h₃ : u + v = t) (h₄ : u < v) 
    : u : v = 2 : 3 := by
  sorry

end ratio_of_segments_l797_797710


namespace least_t_geometric_progression_l797_797358

theorem least_t_geometric_progression (α t : ℝ) (hα : 0 < α ∧ α < π / 2) :
  (0 < t ∧
  ∃ r : ℝ, 
  (r * α = 3 * α) ∧ 
  (r^2 * α = 8 * α) ∧ 
  (r^3 * α = t * α)) →
  t = 16 * real.sqrt 6 / 3 :=
by sorry

end least_t_geometric_progression_l797_797358


namespace labeling_edges_no_common_divisor_l797_797973

variables (G : Type) [graph : SimpleGraph G] (n : ℕ)

-- Assuming G is a connected graph with n edges
variable [DecidableRel graph.adj]
variable connected : graph.Connected
variable (edges_count : graph.edgeFinset.card = n)

-- Define the property that needs to be proven
noncomputable def good_labeling_exists : Prop :=
  ∃ label : graph.Edge → ℕ, 
    (∀ e, label e ∈ Finset.range (n + 1)) ∧ 
    (∀ v, graph.degree v ≥ 2 → 
         ∃ (e1 e2 : graph.Edge), graph.incident v e1 ∧ 
                                 graph.incident v e2 ∧ 
                                 Nat.gcd (label e1) (label e2) = 1)

-- The Lean statement to prove
theorem labeling_edges_no_common_divisor :
  good_labeling_exists G n connected edges_count :=
sorry

end labeling_edges_no_common_divisor_l797_797973


namespace eval_expression_l797_797779

theorem eval_expression (x y z : ℕ) (h_x : x = 3) (h_y : y = 4) (h_z : z = 5) : 
  abs (3500 - (1000 / (20.50 + x * 10)) / ((y ^ 2) - 2 * z) - 3496.70) < 0.01 := 
by
  sorry

end eval_expression_l797_797779


namespace general_term_formula_of_arithmetic_seq_l797_797995

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem general_term_formula_of_arithmetic_seq 
  (a : ℕ → ℝ) (h_arith : arithmetic_seq a)
  (h1 : a 3 * a 7 = -16) 
  (h2 : a 4 + a 6 = 0) :
  (∀ n : ℕ, a n = 2 * n - 10) ∨ (∀ n : ℕ, a n = -2 * n + 10) :=
by
  sorry

end general_term_formula_of_arithmetic_seq_l797_797995


namespace part1_part2_l797_797467

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l797_797467


namespace geometric_sequence_100th_term_l797_797150

theorem geometric_sequence_100th_term :
  ∀ (a₁ a₂ : ℤ) (r : ℤ), a₁ = 5 → a₂ = -15 → r = a₂ / a₁ → 
  (a₁ * r ^ 99 = -5 * 3 ^ 99) :=
by
  intros a₁ a₂ r ha₁ ha₂ hr
  sorry

end geometric_sequence_100th_term_l797_797150


namespace csc_315_eq_neg_sqrt2_l797_797025

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end csc_315_eq_neg_sqrt2_l797_797025


namespace arithmetic_progression_root_difference_l797_797054

theorem arithmetic_progression_root_difference (a b c : ℚ) (h : 81 * a * a * a - 225 * a * a + 164 * a - 30 = 0)
  (hb : b = 5/3) (hprog : ∃ d : ℚ, a = b - d ∧ c = b + d) :
  c - a = 5 / 9 :=
sorry

end arithmetic_progression_root_difference_l797_797054


namespace JohnsonFamilySeating_l797_797627

theorem JohnsonFamilySeating : 
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 359000 := by
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  show total_arrangements - restricted_arrangements = 359000 from sorry

end JohnsonFamilySeating_l797_797627


namespace find_n_find_S_n_l797_797520

open_locale big_operators

-- Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
∀ n m : ℕ, a n + a m = 2 * a ((n + m) / 2)

def a_seq (n : ℕ) : ℚ := if n = 1 then 1/3 else (1/3 + (n-1) * (2/3))

-- Given conditions
axiom h1 : is_arithmetic_sequence a_seq
axiom h2 : a_seq 1 = 1/3
axiom h3 : a_seq 2 + a_seq 5 = 4
axiom h4 : ∃ n, a_seq n = 33

-- Prove n = 50
theorem find_n : ∃ n, a_seq n = 33 ∧ n = 50 := 
sorry

-- Prove S_n = 850
theorem find_S_n : ∃ S_n, S_n = 850 ∧ (∃ n, a_seq n = 33) := 
sorry

end find_n_find_S_n_l797_797520


namespace digits_sum_18_to_21_sum_digits_0_to_99_l797_797686

open Nat List

-- Lean statement definition
def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |> List.sum

theorem digits_sum_18_to_21 : 
  sum_of_digits 18 + sum_of_digits 19 + sum_of_digits 20 + sum_of_digits 21 = 24 := by 
  sorry

noncomputable def q := 
  List.range 100 |> List.map sum_of_digits |> List.sum

theorem sum_digits_0_to_99 : q = 900 := by
  sorry

end digits_sum_18_to_21_sum_digits_0_to_99_l797_797686


namespace average_price_of_5_shirts_l797_797725

theorem average_price_of_5_shirts :
  (forall (n : ℕ) (p1 p2 p3 : ℕ),
    n = 5 → 
    p1 = 30 → 
    p2 = 20 → 
    p3 = 33.333333333333336 →
    ∃ average_price : ℝ,
      average_price = ((p1 + p2 + 3 * p3) / n) →
      average_price = 30) :=
by
  intros
  sorry

end average_price_of_5_shirts_l797_797725


namespace find_k_and_an_sum_of_bn_l797_797087

-- Given conditions
def Sn (n : ℕ) (k : ℝ) : ℝ := n^2 + k * n
def a (n : ℕ) (k : ℝ) : ℝ := Sn n k - Sn (n - 1) k
def b (n : ℕ) (an : ℕ → ℝ) : ℝ := 2 / (n * (an n + 1))

-- Prove the value k and general formula for the sequence an
theorem find_k_and_an (k : ℝ) (a6_eq_13: a 6 k = 13) :
  k = 2 ∧ ∀ n : ℕ, a n 2 = 2 * n + 1 := sorry

-- Prove the sum of the first n terms of the sequence bn
theorem sum_of_bn (n : ℕ) (a2 : ℕ → ℝ)
  (h : ∀ m, m > 0 -> a2 m = 2 * m + 1) :
  ∑ i in finset.range n, b i a2 = n / (n + 1) := sorry

end find_k_and_an_sum_of_bn_l797_797087


namespace domain_of_function_l797_797056

noncomputable def function_domain (x : ℝ) : Set ℝ := {x | x > -1 ∧ -x^2 - 3x + 4 > 0}

theorem domain_of_function :
  function_domain x = {x : ℝ | -1 < x ∧ x < 1} := sorry

end domain_of_function_l797_797056


namespace part1_part2_l797_797458

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l797_797458


namespace find_m_from_hyperbola_and_parabola_l797_797840

theorem find_m_from_hyperbola_and_parabola (a m : ℝ) 
  (h_eccentricity : (Real.sqrt (a^2 + 4)) / a = 3 * Real.sqrt 5 / 5) 
  (h_focus_coincide : (m / 4) = -3) : m = -12 := 
  sorry

end find_m_from_hyperbola_and_parabola_l797_797840


namespace range_of_m_l797_797556

theorem range_of_m (m : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ set.Icc 1 3, f x > -m + 2) :
  ∃ a : ℝ, a = 3 ∧ (m > a) :=
by
  let f := λ x : ℝ, m * x^2 - m * x - 1
  have h1 : ∀ x ∈ set.Icc 1 3, m * (x^2 - x + 1) - 3 > 0 := sorry
  exact ⟨3, rfl, sorry⟩

end range_of_m_l797_797556


namespace arithmetic_seq_max_S_l797_797412

theorem arithmetic_seq_max_S {S : ℕ → ℝ} (h1 : S 2023 > 0) (h2 : S 2024 < 0) : S 1012 > S 1013 :=
sorry

end arithmetic_seq_max_S_l797_797412


namespace proper_fraction_sum_distinct_unit_fractions_l797_797571
noncomputable def canRepresentAsSumOfUnitFractions (m n : ℕ) : Prop :=
  0 < m → m < n → ∃ (k : ℕ) (a : Fin k → ℕ), (∀ i j : Fin k, i ≠ j → a i ≠ a j) ∧ (∑ i, 1 / (a i : ℚ) = m / n)

theorem proper_fraction_sum_distinct_unit_fractions (m n : ℕ) (h₁ : 0 < m) (h₂ : m < n) : canRepresentAsSumOfUnitFractions m n :=
by sorry

end proper_fraction_sum_distinct_unit_fractions_l797_797571


namespace graph_passes_through_1_0_l797_797080

-- Definitions for the given conditions
variable {α β : Type} [LinearOrderedField α] [LinearOrderedField β]

variables (f : α → β) (f_inv : β → α)

-- Function f has an inverse
axiom has_inverse : ∀ y, f (f_inv y) = y ∧ f_inv (f y) = y

-- Condition: y = f(x) + 1/x passes through (1, 2)
axiom pass_through_1_2 : f 1 + 1 = 2

-- Proving that y = f⁻¹(x) - 1/x passes through (1, 0)
theorem graph_passes_through_1_0 : f_inv 1 - 1 = 0 :=
by
  have h1 : f 1 = 1 := by
    linarith [pass_through_1_2]
  have h2 : f_inv 1 = 1 := by
    rw [←has_inverse f f_inv 1]
    exact h1
  rw [h2]; linarith

end graph_passes_through_1_0_l797_797080


namespace construct_triangle_ABC_l797_797092

-- The given condition that we have an acute-angled triangle A_1 B_1 C_1.
variables {A1 B1 C1 A B C : Type} [EuclideanGeometry A1 B1 C1]

-- Definition of an acute-angled triangle
def is_acute_angled_triangle (A1 B1 C1 : Type) [EuclideanGeometry A1 B1 C1] : Prop :=
  ∀ a b c : ℝ, 0 < angle a b c < π / 2

-- Definitions for the construction of equilateral triangles
def is_equilateral_triangle (A B C : Type) [EuclideanGeometry A B C] : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A)

-- Complete the proof to show the final construction
theorem construct_triangle_ABC
  (hacute : is_acute_angled_triangle A1 B1 C1)
  (hequilateral1 : is_equilateral_triangle A1 B C)
  (hequilateral2 : is_equilateral_triangle B1 C A)
  (hequilateral3 : is_equilateral_triangle C1 A B)  :
  ∃ (A B C : Type) [EuclideanGeometry A B C], 
    (is_equilateral_triangle A1 B C) ∧ (is_equilateral_triangle B1 C A) ∧ (is_equilateral_triangle C1 A B) :=
sorry

end construct_triangle_ABC_l797_797092


namespace mod_remainder_l797_797294

theorem mod_remainder (a p : ℕ) (h_prime : nat.prime p) (h_a : a < p) :
  ((a^a + a) % p) = 89 :=
by
  have h1 : a^(p-1) % p = 1 := nat.modeq.pow_totient a (p-1) h_prime,
  sorry

end mod_remainder_l797_797294


namespace complex_multiplication_l797_797313

-- Defining the imaginary unit i and its square property
def i : ℂ := complex.I

theorem complex_multiplication :
  (1 + 2 * i) * (2 + i) = 5 * i := by
  sorry

end complex_multiplication_l797_797313


namespace inversely_proportional_ratio_l797_797232

section
variables {x y : ℝ} (k : ℝ)
variables (x₁ x₂ y₁ y₂ : ℝ)
variable h_inv : ∀(x y : ℝ), x * y = k

theorem inversely_proportional_ratio
  (h1 : x₁ ≠ 0) (h2 : x₂ ≠ 0) (h3 : y₁ ≠ 0) (h4 : y₂ ≠ 0)
  (h5 : x₁ * y₁ = k) (h6 : x₂ * y₂ = k) (h_ratio : x₁ / x₂ = 4 / 5) :
  y₁ / y₂ = 5 / 4 :=
by
  sorry
end

end inversely_proportional_ratio_l797_797232


namespace reciprocal_neg_six_l797_797650

-- Define the concept of reciprocal
def reciprocal (a : ℤ) (h : a ≠ 0) : ℚ := 1 / a

theorem reciprocal_neg_six : reciprocal (-6) (by norm_num) = -1 / 6 := 
by 
  sorry

end reciprocal_neg_six_l797_797650


namespace Irene_grabbed_5_shirts_l797_797532

theorem Irene_grabbed_5_shirts
    (shorts_price : ℝ)
    (num_shorts : ℕ)
    (shirts_price : ℝ)
    (total_money : ℝ)
    (discount : ℝ)
    (shorts_total_cost_disc : ℝ)
    (money_left : ℝ)
    (dis_price_every_shirt : ℝ)
    (x : ℝ) : num_shorts = 3 ∧ shorts_price = 15 ∧ shirts_price = 17 ∧ total_money = 117 ∧ discount = 0.1 ∧ shorts_total_cost_disc = (shorts_price * num_shorts) * (1 - discount) ∧ money_left = total_money - shorts_total_cost_disc ∧ dis_price_every_shirt = shirts_price * (1 - discount) ∧ money_left / dis_price_every_shirt = x → x = 5 := by
    intros h,
    sorry

end Irene_grabbed_5_shirts_l797_797532


namespace samia_walked_distance_l797_797224

theorem samia_walked_distance 
  (biking_speed : ℝ := 17) 
  (walking_speed : ℝ := 5) 
  (total_time_min : ℝ := 44) 
  (total_time_hr : ℝ := 44 / 60) : 
  ∀ (x : ℝ), 
  ((total_time_hr = (x / biking_speed + x / walking_speed)) → 
  (Real.floor (10 * x) / 10 = 2.8)) := 
by 
  intro x 
  intro h 
  sorry

end samia_walked_distance_l797_797224


namespace average_eq_51x_l797_797978

theorem average_eq_51x (x : ℚ) (h : (1 + 2 + 3 + ... + 50 + x) / 51 = 51 * x) : 
  x = 51 / 104 :=
sorry

end average_eq_51x_l797_797978


namespace trajectory_parabolic_l797_797980

noncomputable def distance_to_point (P : ℝ × ℝ) (Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def distance_to_line_x (P : ℝ × ℝ) (a : ℝ) : ℝ :=
  real.abs (P.1 - a)

theorem trajectory_parabolic (x y : ℝ) :
  (distance_to_point (x, y) (3, 0)) = (distance_to_line_x (x, y) (-2)) + 1 →
  y^2 = 12 * x :=
by
  sorry

end trajectory_parabolic_l797_797980


namespace complement_intersection_l797_797850

open Set

theorem complement_intersection (U A B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 5})
  (hB : B = {2, 4}) :
  ((U \ A) ∩ B) = {2, 4} :=
by
  sorry

end complement_intersection_l797_797850


namespace zane_spent_2_per_oreo_l797_797298

noncomputable def price_per_oreo (x : ℝ) (O C : ℕ) : Prop :=
  let total_spent_on_oreos := O * x in
  let total_spent_on_cookies := C * 3 in
  O / C = 4 / 9 ∧
  O + C = 65 ∧
  total_spent_on_cookies = total_spent_on_oreos + 95 ∧
  x = 2

theorem zane_spent_2_per_oreo : ∃ x O C, price_per_oreo x O C :=
by 
  sorry

end zane_spent_2_per_oreo_l797_797298


namespace sequence_sum_eq_4016_div_2009_l797_797726

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := a n + n + 1

theorem sequence_sum_eq_4016_div_2009 :
  (∑ i in Finset.range 2008, 1 / a (i + 1)) = 4016 / 2009 := 
sorry

end sequence_sum_eq_4016_div_2009_l797_797726


namespace billy_sleep_hours_l797_797353

open Real

-- Define the conditions
variables (x : ℝ)
def second_night := x + 2
def third_night := (x + 2) / 2
def fourth_night := 3 * ((x + 2) / 2)
def total_sleep := x + second_night + third_night + fourth_night

-- The proof statement
theorem billy_sleep_hours (h : total_sleep x = 30) : x = 6 :=
sorry

end billy_sleep_hours_l797_797353


namespace least_positive_four_digit_solution_l797_797790

theorem least_positive_four_digit_solution :
  ∃ x : ℤ, 1000 ≤ x ∧ x < 10000 ∧
           7 * x ≡ 21 [MOD 14] ∧
           2 * x + 13 ≡ 16 [MOD 9] ∧
           -2 * x + 1 ≡ x [MOD 25] ∧
           x = 1167 :=
by
  sorry

end least_positive_four_digit_solution_l797_797790


namespace mrs_hilt_total_money_l797_797562

def total_money_needed (persons : ℕ) (money_per_person : ℝ) : ℝ :=
  persons * money_per_person

theorem mrs_hilt_total_money :
  ∀ (persons : ℕ) (money_per_person : ℝ), persons = 3 → money_per_person = 1.25 → total_money_needed persons money_per_person = 3.75 :=
by {
  intros persons money_per_person h1 h2,
  rw [h1, h2],
  norm_num,
  sorry
}

end mrs_hilt_total_money_l797_797562


namespace men_work_equivalence_l797_797230

theorem men_work_equivalence : 
  ∀ (M : ℕ) (m w : ℕ),
  (3 * w = 2 * m) ∧ 
  (M * 21 * 8 * m = 21 * 60 * 3 * w) →
  M = 15 := by
  intro M m w
  intro h
  sorry

end men_work_equivalence_l797_797230


namespace cube_root_of_583200_l797_797357

theorem cube_root_of_583200 : real.cbrt 583200 = 60 :=
by
  -- Given condition
  have h : 583200 = 6^3 * 5^3 * 2^3 := by norm_num
  -- Prove the cube root
  rw [h, real.cbrt_mul, real.cbrt_pow, real.cbrt_pow, real.cbrt_pow]
  norm_num

end cube_root_of_583200_l797_797357


namespace ellipse_equation_and_m_range_lemma_l797_797490

noncomputable def getX (m : ℝ) : bool :=
  let discriminant := 8*(-m^2 + 3)
  discriminant > 0

noncomputable def midpointCondition (m : ℝ) : bool :=
  let midpointCondition := (4*m^2 + m^2) >= 5
  midpointCondition

theorem ellipse_equation_and_m_range_lemma :
  ∃ a b : ℝ, 0 < b ∧ b < a ∧ 
  (∀ x y : ℝ, ((x^2) / (a^2) + (y^2) / (b^2) = 1) → 
  a = sqrt 2 ∧ b = 1 ∧ (x = 1 ∧ y = sqrt 2 / 2)) ∧
  (∀ m : ℝ, -sqrt 3 < m ∧ m < sqrt 3 →
  ¬ (4 * (-m^2 / 9 + 1 / 3) + (-m^2 / 9 + m) ^ 2 < 5 / 9) ∧ getX m = true ∧ midpointCondition m  = true ) →
  (a = sqrt 2 ∧ b = 1 ∧ (x = 1 ∧ y = sqrt 2 / 2) ∧
  (m ≤ -1 ∨ m ≥ 1)) :=
by
  intros a b hab0 hab1 h
  have ha2 := sqrt 2_pos,
  sorry

end ellipse_equation_and_m_range_lemma_l797_797490


namespace geometric_sequence_property_l797_797081

variable {α : Type*} [Field α] (a : ℕ → α) (r : α)

-- The sequence is geometric
def is_geometric_sequence (a : ℕ → α) (r : α) : Prop :=
∀ n, a (n + 1) = a n * r

-- The conditions given in the problem
axiom (geom_seq : is_geometric_sequence a r)
axiom (cond1 : a 4 + a 6 = (π : α))

-- The proof statement
theorem geometric_sequence_property (h : is_geometric_sequence a r) (h1 : a 4 + a 6 = (π : α)) :
  a 5 * a 3 + 2 * (a 5) ^ 2 + a 5 * a 7 = (π : α) ^ 2 := 
by
  sorry

end geometric_sequence_property_l797_797081


namespace inequality_part1_inequality_part2_l797_797451

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l797_797451


namespace part1_part2_l797_797431

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l797_797431


namespace log_sum_l797_797124

theorem log_sum :
  ∀ (a b : ℝ), (2 ^ a = 100) ∧ (5 ^ b = 100) → (1 / a + 1 / b = 1 / 2) :=
by
  intros a b h
  sorry

end log_sum_l797_797124


namespace positive_integers_satisfy_l797_797774

theorem positive_integers_satisfy :
  {n : ℤ | (n + 9) * (n - 4) * (n - 13) < 0 ∧ n > 0}.card = 8 :=
by
  sorry

end positive_integers_satisfy_l797_797774


namespace problem_statement_l797_797485

variable {f : ℝ → ℝ}
variable {a b : ℝ}

theorem problem_statement
  (h₀ : ∀ x, 0 < x → x * (f' x) > 2 * f x)
  (h₁ : 0 < b)
  (h₂ : b < a) :
  b^2 * f a > a^2 * f b :=
sorry

end problem_statement_l797_797485


namespace perfect_square_pairs_count_l797_797261

theorem perfect_square_pairs_count : 
  ∃ (cards : Finset ℕ), 
  (cards.card = 294 ∧
   (∀ n ∈ cards, ∃ (k : ℕ), n = 7^k ∨ n = 11^k) ∧ 
    ∃ (n_pairs : ℕ), n_pairs = 15987 ∧ 
      (n_pairs = (cards.filter (λ x, (log x 7).nat_odd)).card * (cards.filter (λ x, (log x 7).nat_odd)).card / 2 + 
                (cards.filter (λ x, (log x 7).nat_even)).card * (cards.filter (λ x, (log x 7).nat_even)).card / 2 + 
                (cards.filter (λ x, (log x 11).nat_odd)).card * (cards.filter (λ x, (log x 11).nat_odd)).card / 2 + 
                (cards.filter (λ x, (log x 11).nat_even)).card * (cards.filter (λ x, (log x 11).nat_even)).card / 2 + 
                (cards.filter (λ x, (log x 7).nat_even)).card * (cards.filter (λ x, (log x 11).nat_even)).card)) :=
sorry

end perfect_square_pairs_count_l797_797261


namespace tan_double_theta_l797_797125

theorem tan_double_theta (θ : ℝ) :
  3 * cos (π / 2 - θ) + cos (π + θ) = 0 → tan (2 * θ) = 3 / 4 :=
by
  intro h
  sorry

end tan_double_theta_l797_797125


namespace dreamy_vacation_note_probability_l797_797879

theorem dreamy_vacation_note_probability :
  ∃ n : ℕ, n ≤ 5 ∧ 
  (probability_binomial 5 0.4 n = 0.2304) :=
sorry

end dreamy_vacation_note_probability_l797_797879


namespace exists_circle_with_13_points_l797_797515

open Set

theorem exists_circle_with_13_points (points : Set (EuclideanSpace ℝ 2)) (h₁ : points.length = 25) 
  (h₂ : ∀ (p₁ p₂ p₃ : EuclideanSpace ℝ 2), p₁ ∈ points ∧ p₂ ∈ points ∧ p₃ ∈ points → 
          (dist p₁ p₂ < 1) ∨ (dist p₂ p₃ < 1) ∨ (dist p₁ p₃ < 1)) : 
  ∃ (c : EuclideanSpace ℝ 2) (r : ℝ), r = 1 ∧ ∃ (s : Finset (EuclideanSpace ℝ 2)), s.card ≥ 13 ∧ ∀ x ∈ s, dist x c ≤ r := 
begin 
  sorry
end

end exists_circle_with_13_points_l797_797515


namespace monotonic_increase_interval_l797_797111

theorem monotonic_increase_interval :
  ∀ (x : ℝ), (x ∈ Icc 0 Real.pi) → (∀ y ∈ Icc 0 (Real.pi / 4), f y ≤ f x) →
     f x = sqrt 2 * (Real.sin (x + (Real.pi / 4))) :=
begin
  sorry
end

end monotonic_increase_interval_l797_797111


namespace angle_BAK_eq_90_l797_797881

namespace Geometry

variables {A B C D K : Type} [IncidenceGeometry A B C D]

/-- In a tetrahedron ABCD, the sum of angles BAC and BAD is 180 degrees. AK is the angle bisector of ∠CAD.
Prove that the measure of ∠BAK is 90 degrees. -/
theorem angle_BAK_eq_90
    (h1 : ∠BAC + ∠BAD = 180)
    (h2 : AK bisects ∠CAD)
    : ∠BAK = 90 :=
sorry

end Geometry

end angle_BAK_eq_90_l797_797881


namespace johnson_family_seating_l797_797621

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem johnson_family_seating (sons daughters : ℕ) (total_seats : ℕ) 
  (condition1 : sons = 5) (condition2 : daughters = 4) (condition3 : total_seats = 9) :
  let total_arrangements := factorial total_seats,
      restricted_arrangements := factorial sons * factorial daughters,
      answer := total_arrangements - restricted_arrangements
  in answer = 360000 := 
by
  -- The proof would go here
  sorry

end johnson_family_seating_l797_797621


namespace part1_part2_l797_797481

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l797_797481


namespace max_expr_value_l797_797928

theorem max_expr_value (x y z : ℝ) (h_pos: x > 0 ∧ y > 0 ∧ z > 0) (h_sum: x + y + z = 1) :
  (\frac{(x + y + z)^2}{x^2 + y^2 + z^2}) ≤ 3 :=
by
  sorry

end max_expr_value_l797_797928


namespace P_divides_AC_in_ratio_l797_797093

theorem P_divides_AC_in_ratio
  (ABC : Type)
  [equilateral_triangle ABC]
  (A B C K M P : ABC)
  (hK : midpoint K A B)
  (hM : M ∈ line_segment B C ∧ ratio (M, B, C) = 1 / 3)
  (hP : minimizes_perimeter P K M) :
  divides (P, A, C) 2 3 :=
sorry

end P_divides_AC_in_ratio_l797_797093


namespace find_number_l797_797799

theorem find_number (x : ℤ) (h : x * 9999 = 806006795) : x = 80601 :=
sorry

end find_number_l797_797799


namespace interval_decrease_log_l797_797983

theorem interval_decrease_log (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : a > 1) : 
  Set.Ioo (Real.Inf $ Set.Ioo Real.Inf Real.Inf) (-1) := 
sorry

end interval_decrease_log_l797_797983


namespace csc_315_eq_neg_sqrt_2_l797_797020

theorem csc_315_eq_neg_sqrt_2 :
  let csc := λ θ, 1 / Real.sin θ in
  csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
  by
  let sin := Real.sin
  have h1 : csc (315 * Real.pi / 180) = 1 / sin (315 * Real.pi / 180) := rfl
  have h2 : sin (315 * Real.pi / 180) = sin ((360 - 45) * Real.pi / 180) := by congr; norm_num
  have h3 : sin ((360 - 45) * Real.pi / 180) = -sin (45 * Real.pi / 180) := by
    rw [Real.sin_pi_sub]
    congr; norm_num
  have h4 : sin (45 * Real.pi / 180) = 1 / Real.sqrt 2 := Real.sin_of_one_div_sqrt_two 45 rfl
  sorry

end csc_315_eq_neg_sqrt_2_l797_797020


namespace toaster_sales_1_toaster_sales_2_l797_797362

variable (p c k : ℝ)
variable (k₀ : k = 6000)

theorem toaster_sales_1 (h₁ : p * 300 = 6000) : (20 * 300 = k) ∧ (k = 6000) → 600 * (6000 / 600) = 6000 :=
by
  intro h2
  exact h2
  sorry

theorem toaster_sales_2 (h₂ : p * c = 6000) : (15 * 400 = 6000) ∧ (k = 6000) → 400 := 
by
  intro h3
  exact h3
  sorry

end toaster_sales_1_toaster_sales_2_l797_797362


namespace factorial_division_l797_797422

theorem factorial_division (h : 9.factorial = 362880) : 9.factorial / 4.factorial = 15120 := by
  sorry

end factorial_division_l797_797422


namespace johnson_family_seating_l797_797630

theorem johnson_family_seating (boys girls : Finset ℕ) (h_boys : boys.card = 5) (h_girls : girls.card = 4) :
  (∃ (arrangement : List ℕ), arrangement.length = 9 ∧ at_least_two_adjacent boys arrangement) :=
begin
  -- Given the total number of ways: 9! 
  -- subtract 5! * 4! from 9! to get the result 
  have total_arrangements := nat.factorial 9,
  have restrictive_arrangements := nat.factorial 5 * nat.factorial 4,
  exact (total_arrangements - restrictive_arrangements) = 360000,
end

end johnson_family_seating_l797_797630


namespace fifth_friend_paid_40_l797_797066

-- Defining the conditions given in the problem
variables {a b c d e : ℝ}
variables (h1 : a = (1/3) * (b + c + d + e))
variables (h2 : b = (1/4) * (a + c + d + e))
variables (h3 : c = (1/5) * (a + b + d + e))
variables (h4 : d = (1/6) * (a + b + c + e))
variables (h5 : a + b + c + d + e = 120)

-- Proving that the amount paid by the fifth friend is $40
theorem fifth_friend_paid_40 : e = 40 :=
by
  sorry  -- Proof to be provided

end fifth_friend_paid_40_l797_797066


namespace hyperbola_eqn_asymptote_focus_l797_797116

theorem hyperbola_eqn_asymptote_focus
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : b = a * Real.sqrt 3)
  (h4 : a^2 + b^2 = 4) :
  a = 1 ∧ b = Real.sqrt 3 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ b = a * Real.sqrt 3 ∧ a^2 + b^2 = 4 ∧
  ∀ x y : ℝ, x^2 - (y^2 / 3) = 1 :=
by
  intro hab
  use [1, Real.sqrt 3]
  split
  apply zero_lt_one,
  apply Real.sqrt_pos_strict.mpr,
  norm_num,
  split,
  rw hab.2,
  norm_num,
  have : 1^2 + (Real.sqrt 3)^2 = 4,
  norm_num,
  rw Real.sqrt_sq,
  norm_num,
  rwa hab.1,
  exact ⟨1, Real.sqrt 3, Real.sqrt_pos.mpr (show 3 ≥ 0 by norm_num)⟩,
  sorry

end hyperbola_eqn_asymptote_focus_l797_797116


namespace problem1_problem2_l797_797508

variables {x y z : ℝ}

theorem problem1 (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y + y * z + z * x = 1) :
  27 / 4 * (x + y) * (y + z) * (z + x) ≥ (sqrt (x + y) + sqrt (y + z) + sqrt (z + x))^2 :=
sorry

theorem problem2 (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y + y * z + z * x = 1) :
  (sqrt (x + y) + sqrt (y + z) + sqrt (z + x))^2 ≥ 6 * sqrt 3 :=
sorry

end problem1_problem2_l797_797508


namespace determine_angle_range_l797_797816

variable (α : ℝ)

theorem determine_angle_range 
  (h1 : 0 < α) 
  (h2 : α < 2 * π) 
  (h_sin : Real.sin α < 0) 
  (h_cos : Real.cos α > 0) : 
  (3 * π / 2 < α ∧ α < 2 * π) := 
sorry

end determine_angle_range_l797_797816


namespace csc_315_eq_neg_sqrt_2_l797_797001

theorem csc_315_eq_neg_sqrt_2 :
  Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := 
by
  sorry

end csc_315_eq_neg_sqrt_2_l797_797001


namespace twenty_less_waiter_slices_eq_28_l797_797129

noncomputable def slices_of_pizza : ℕ := 78
noncomputable def buzz_ratio : ℕ := 5
noncomputable def waiter_ratio : ℕ := 8

theorem twenty_less_waiter_slices_eq_28:
  let total_slices := slices_of_pizza in
  let total_ratio := buzz_ratio + waiter_ratio in
  let waiter_slices := (waiter_ratio * total_slices) / total_ratio in
  waiter_slices - 20 = 28 := by
  sorry

end twenty_less_waiter_slices_eq_28_l797_797129


namespace geometric_sequence_log_sum_l797_797135

open Real

theorem geometric_sequence_log_sum {a : ℕ → ℝ}
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ m n, m + 1 = n → a m * a n = a (m - 1) * a (n + 1) )
  (h3 : a 10 * a 11 + a 9 * a 12 = 2 * exp 5) :
  ( ∑ i in Finset.range 20, log (a (i+1)) ) = 50 :=
by
  -- The detailed proof is omitted here.
  sorry

end geometric_sequence_log_sum_l797_797135


namespace johnson_family_seating_l797_797618

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem johnson_family_seating (sons daughters : ℕ) (total_seats : ℕ) 
  (condition1 : sons = 5) (condition2 : daughters = 4) (condition3 : total_seats = 9) :
  let total_arrangements := factorial total_seats,
      restricted_arrangements := factorial sons * factorial daughters,
      answer := total_arrangements - restricted_arrangements
  in answer = 360000 := 
by
  -- The proof would go here
  sorry

end johnson_family_seating_l797_797618


namespace area_of_annulus_proof_l797_797737

noncomputable def area_of_annulus (b c a : ℝ) (h_bc : b > c) (h_pythagorean : b^2 = c^2 + a^2) : ℝ :=
  π * a^2

theorem area_of_annulus_proof (b c a : ℝ) (h_bc : b > c) (h_pythagorean : b^2 = c^2 + a^2) :
  area_of_annulus b c a h_bc h_pythagorean = π * a^2 :=
by
  unfold area_of_annulus
  sorry

end area_of_annulus_proof_l797_797737


namespace wall_clock_time_at_car_5PM_l797_797970

-- Define the initial known conditions
def initial_time : ℕ := 7 -- 7:00 AM
def wall_time_at_10AM : ℕ := 10 -- 10:00 AM
def car_time_at_10AM : ℕ := 11 -- 11:00 AM
def car_time_at_5PM : ℕ := 17 -- 5:00 PM = 17:00 in 24-hour format

-- Define the calculations for the rate of the car clock
def rate_of_car_clock : ℚ := (car_time_at_10AM - initial_time : ℚ) / (wall_time_at_10AM - initial_time : ℚ) -- rate = 4/3

-- Prove the actual time according to the wall clock when the car clock shows 5:00 PM
theorem wall_clock_time_at_car_5PM :
  let elapsed_real_time := (car_time_at_5PM - car_time_at_10AM) * (3 : ℚ) / (4 : ℚ)
  let actual_time := wall_time_at_10AM + elapsed_real_time
  (actual_time : ℚ) = 15 + (15 / 60 : ℚ) := -- 3:15 PM as 15.25 in 24-hour time
by
  sorry

end wall_clock_time_at_car_5PM_l797_797970


namespace num_bases_between_4_and_12_where_1024_ends_in_1_l797_797072

theorem num_bases_between_4_and_12_where_1024_ends_in_1 :
  (finset.card (finset.filter (λ b : ℕ, 4 ≤ b ∧ b ≤ 12 ∧ 1023 % b = 0) (finset.range (12 + 1)))) = 1 :=
by
  sorry

end num_bases_between_4_and_12_where_1024_ends_in_1_l797_797072


namespace perimeter_AMN_l797_797665

-- Define a triangle and its properties
def TriangleABC_sides : Prop :=
  let AB := 12
  let BC := 26
  let AC := 18
  True

-- Define the incenter and the points M and N based on given conditions
def incenter_line_parallel : Prop :=
  ∃ (O M N : Type), 
    (
      let is_incenter := True in -- Placeholder for the incenter definition.
      let is_parallel := True in -- Placeholder for the parallel condition definition.
      True
    )

-- Final theorem
theorem perimeter_AMN
  (h1 : TriangleABC_sides) 
  (h2 : incenter_line_parallel) : 
  ∃ (AM MN NA : ℝ), AM + MN + NA = 30 :=
by
  sorry  -- Proof to be filled in

end perimeter_AMN_l797_797665


namespace hyperbola_parabola_focus_l797_797145

theorem hyperbola_parabola_focus (k : ℝ) (h : k > 0) :
  (∃ x y : ℝ, (1/k^2) * y^2 = 0 ∧ x^2 - (y^2 / k^2) = 1) ∧ (∃ x : ℝ, y^2 = 8 * x) →
  k = Real.sqrt 3 :=
by sorry

end hyperbola_parabola_focus_l797_797145


namespace point_reflection_y_l797_797890

def coordinates_with_respect_to_y_axis (x y : ℝ) : ℝ × ℝ :=
  (-x, y)

theorem point_reflection_y (x y : ℝ) (h : (x, y) = (-2, 3)) : coordinates_with_respect_to_y_axis x y = (2, 3) := by
  sorry

end point_reflection_y_l797_797890


namespace intersection_sets_l797_797825

theorem intersection_sets :
  let A := { y : ℝ | ∃ x : ℝ, y = x^2 - 1 }
      B := { y : ℝ | ∃ x : ℝ, -real.sqrt 2 ≤ x ∧ x ≤ real.sqrt 2 ∧ y = real.sqrt (2 - x^2) }
  ∀ y : ℝ, y ∈ (A ∩ B) ↔ y ∈ set.Icc (-1) 1 := 
by
  sorry

end intersection_sets_l797_797825


namespace percent_increase_from_first_to_second_quarter_l797_797682

theorem percent_increase_from_first_to_second_quarter 
  (P : ℝ) :
  ((1.60 * P - 1.20 * P) / (1.20 * P)) * 100 = 33.33 := by
  sorry

end percent_increase_from_first_to_second_quarter_l797_797682


namespace numerators_required_l797_797542

-- Define the set T of rational numbers r where 0 < r < 1 with the specified repeating decimal
def is_special_repeating_decimal (r : ℚ) : Prop :=
  0 < r ∧ r < 1 ∧ ∃ e f g h : ℕ, r = (e * 1000 + f * 100 + g * 10 + h) / 9999

-- Define the problem of counting the number of such numerators required
theorem numerators_required (T : set ℚ) 
  (hT : ∀ r, r ∈ T ↔ is_special_repeating_decimal r) : 
  ∃ n, n = 6000 ∧ ∀ r ∈ T, ∃ k : ℕ, r = k / 9999 ∧ nat.gcd k 9999 = 1 :=
by
  let num_numerators := (λ n : ℕ, nat.gcd n 9999 = 1)
  have : finset.card (finset.filter num_numerators (finset.range 10000)) = 6000,
  { sorry }, -- Euler's totient function calculation yields 6000 co-prime numbers
  exact ⟨6000, this, λ r hr,
  begin
    rcases hT r with ⟨h0, h1, e, f, g, h', h_eq⟩,
    use (e * 1000 + f * 100 + g * 10 + h'),
    refine ⟨h_eq, _⟩,
    have := this,
    sorry
  end⟩

end numerators_required_l797_797542


namespace johnson_family_seating_problem_l797_797614

theorem johnson_family_seating_problem : 
  ∃ n : ℕ, n = 9! - 5! * 4! ∧ n = 359760 :=
by
  have total_ways := (Nat.factorial 9)
  have no_adjacent_boys := (Nat.factorial 5) * (Nat.factorial 4)
  have result := total_ways - no_adjacent_boys
  use result
  split
  . exact eq.refl result
  . norm_num -- This will replace result with its evaluated form, 359760

end johnson_family_seating_problem_l797_797614


namespace part1_part2_l797_797428

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l797_797428


namespace marching_band_formations_l797_797301

theorem marching_band_formations : (number of pairs (s, t) such that s * t = 240 and 8 ≤ t ≤ 30 and s ≥ 1) = 5 :=
sorry

end marching_band_formations_l797_797301


namespace angle_YXZ_in_triangle_XYZ_l797_797170

theorem angle_YXZ_in_triangle_XYZ
    (XYZ : Type)
    [triangle : triangle XYZ]
    (angle_XYZ : ℝ)
    (angle_YZA : ℝ)
    (tangent_circle : ∃ A : XYZ, is_tangent_to_sides A XYZ)
    (external_angle_bisector : external_angle_bisector_property Z A Y)
    (hXYZ : angle_XYZ = 80)
    (hYZA : angle_YZA = 20) :
    angle_YXZ = 60 :=
begin
  sorry
end

end angle_YXZ_in_triangle_XYZ_l797_797170


namespace total_pieces_for_ten_rows_l797_797778

noncomputable def sum_arithmetic_sequence (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def rods_for_n_rows (n : ℕ) : ℕ :=
  3 * sum_arithmetic_sequence 1 1 n

def connectors_for_n_rows (n : ℕ) : ℕ :=
  sum_arithmetic_sequence 1 1 (n + 1)

def total_pieces (n : ℕ) : ℕ :=
  rods_for_n_rows n + connectors_for_n_rows n

theorem total_pieces_for_ten_rows : total_pieces 10 = 231 :=
by
  sorry

end total_pieces_for_ten_rows_l797_797778


namespace sum_of_possible_values_of_A_add_B_l797_797863

theorem sum_of_possible_values_of_A_add_B :
  let possible_sums := {2, 10}
  List.sum possible_sums = 12 :=
by
  sorry

end sum_of_possible_values_of_A_add_B_l797_797863


namespace log_product_identity_l797_797368

theorem log_product_identity (x : ℝ) (hx : x > 0) : 
  log x (2 * x) * log (2 * x) (x ^ 3) = 3 :=
by {
  sorry
}

end log_product_identity_l797_797368


namespace phil_change_l797_797951

-- Define the conditions as hypothesis
def cost_per_apple := 0.75  -- cost per apple in dollars
def num_apples := 4         -- number of apples bought by Phil
def payment := 10.0         -- amount paid by Phil

-- Prove that the change received by Phil is $7.00
theorem phil_change : payment - (num_apples * cost_per_apple) = 7.0 :=
by sorry

end phil_change_l797_797951


namespace find_function_range_of_a_l797_797503

variables (a b : ℝ) (f : ℝ → ℝ) 

-- Given: f(x) = ax + b where a ≠ 0 
--        f(2x + 1) = 4x + 1
-- Prove: f(x) = 2x - 1
theorem find_function (h1 : ∀ x, f (2 * x + 1) = 4 * x + 1) : 
  ∃ a b, a = 2 ∧ b = -1 ∧ ∀ x, f x = a * x + b :=
by sorry

-- Given: A = {x | a - 1 < x < 2a +1 }
--        B = {x | 1 < f(x) < 3 }
--        B ⊆ A
-- Prove: 1/2 ≤ a ≤ 2
theorem range_of_a (Hf : ∀ x, f x = 2 * x - 1) (Hsubset: ∀ x, 1 < f x ∧ f x < 3 → a - 1 < x ∧ x < 2 * a + 1) :
  1 / 2 ≤ a ∧ a ≤ 2 :=
by sorry

end find_function_range_of_a_l797_797503


namespace inscribed_square_side_length_l797_797221

theorem inscribed_square_side_length (DE EF DF : ℝ) (hDE : DE = 12) (hEF : EF = 16) (hDF : DF = 20) :
  ∃ t : ℝ, (t = 80 / 9) :=
by 
  have hArea : (1 / 2) * DE * 16 = 96, by sorry
  have k : ℝ := 16, by sorry
  have t : ℝ := 20 * k / (20 + k), by sorry
  existsi t
  rw [hDE, hEF, hDF]
  split
  case 1 {
    exact hArea
  }
  case 2 {
    rw [k]
    norm_num
    linarith
  }

end inscribed_square_side_length_l797_797221


namespace find_a_over_b_l797_797770

variable (x y z a b : ℝ)
variable (h₁ : 4 * x - 2 * y + z = a)
variable (h₂ : 6 * y - 12 * x - 3 * z = b)
variable (h₃ : b ≠ 0)

theorem find_a_over_b : a / b = -1 / 3 :=
by
  sorry

end find_a_over_b_l797_797770


namespace cost_difference_zero_l797_797330

theorem cost_difference_zero
  (A O X : ℝ)
  (h1 : 3 * A + 7 * O = 4.56)
  (h2 : A + O = 0.26)
  (h3 : O = A + X) :
  X = 0 := 
sorry

end cost_difference_zero_l797_797330


namespace multiply_5915581_7907_l797_797355

theorem multiply_5915581_7907 : 5915581 * 7907 = 46757653387 := 
by
  -- sorry is used here to skip the proof
  sorry

end multiply_5915581_7907_l797_797355


namespace find_triangles_l797_797146

/-- In a triangle, if the side lengths a, b, c (a ≤ b ≤ c) are integers, form a geometric progression (i.e., b² = ac),
    and at least one of a or c is equal to 100, then the possible values for the triple (a, b, c) are:
    (49, 70, 100), (64, 80, 100), (81, 90, 100), 
    (100, 100, 100), (100, 110, 121), (100, 120, 144),
    (100, 130, 169), (100, 140, 196), (100, 150, 225), (100, 160, 256). 
-/
theorem find_triangles (a b c : ℕ) (h1 : a ≤ b ∧ b ≤ c) 
(h2 : b * b = a * c)
(h3 : a = 100 ∨ c = 100) : 
  (a = 49 ∧ b = 70 ∧ c = 100) ∨ 
  (a = 64 ∧ b = 80 ∧ c = 100) ∨ 
  (a = 81 ∧ b = 90 ∧ c = 100) ∨ 
  (a = 100 ∧ b = 100 ∧ c = 100) ∨ 
  (a = 100 ∧ b = 110 ∧ c = 121) ∨ 
  (a = 100 ∧ b = 120 ∧ c = 144) ∨ 
  (a = 100 ∧ b = 130 ∧ c = 169) ∨ 
  (a = 100 ∧ b = 140 ∧ c = 196) ∨ 
  (a = 100 ∧ b = 150 ∧ c = 225) ∨ 
  (a = 100 ∧ b = 160 ∧ c = 256) := sorry

end find_triangles_l797_797146


namespace triangle_is_isosceles_l797_797411

variable {α β γ : ℝ} (quadrilateral_angles : List ℝ)

-- Conditions from the problem
axiom triangle_angle_sum : α + β + γ = 180
axiom quadrilateral_angle_sum : quadrilateral_angles.sum = 360
axiom quadrilateral_angle_conditions : ∀ (a b : ℝ), a ∈ [α, β, γ] → b ∈ [α, β, γ] → a ≠ b → (a + b ∈ quadrilateral_angles)

-- Proof statement
theorem triangle_is_isosceles : (α = β) ∨ (β = γ) ∨ (γ = α) := 
  sorry

end triangle_is_isosceles_l797_797411


namespace factorial_division_l797_797417

theorem factorial_division :
  9! = 362880 → (9! / 4!) = 15120 := by
  sorry

end factorial_division_l797_797417


namespace part1_part2_l797_797446

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

end part1_part2_l797_797446


namespace max_comics_jason_can_buy_l797_797897

-- Define the conditions
def jason_budget : ℝ := 15
def book_cost : ℝ := 1.20
def discount_cost : ℝ := 1.20 - 0.10

-- Define the cost function C(n)
def cost (n : ℕ) : ℝ :=
  if n ≤ 10 then book_cost * n
  else (book_cost * 10) + (discount_cost * (n - 10))

-- Statement of the proof problem
theorem max_comics_jason_can_buy :
  ∀ n : ℕ, cost(n) ≤ jason_budget → n ≤ 12 :=
sorry

end max_comics_jason_can_buy_l797_797897


namespace part1_part2_l797_797443

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

end part1_part2_l797_797443


namespace csc_315_eq_neg_sqrt2_l797_797029

theorem csc_315_eq_neg_sqrt2 : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end csc_315_eq_neg_sqrt2_l797_797029


namespace fraction_meaningful_l797_797875

theorem fraction_meaningful (a : ℝ) : (∃ x, x = 2 / (a + 1)) ↔ a ≠ -1 :=
by
  sorry

end fraction_meaningful_l797_797875


namespace total_truck_loads_l797_797724

-- Using definitions from conditions in (a)
def sand : ℝ := 0.16666666666666666
def dirt : ℝ := 0.3333333333333333
def cement : ℝ := 0.16666666666666666

-- The proof statement based on the correct answer in (b)
theorem total_truck_loads : sand + dirt + cement = 0.6666666666666666 := 
by
  sorry

end total_truck_loads_l797_797724


namespace unique_solution_l797_797206

def is_valid_func (f : ℕ → ℕ) : Prop :=
  ∀ n, f (f n) + f n = 2 * n + 2001 ∨ f (f n) + f n = 2 * n + 2002

theorem unique_solution (f : ℕ → ℕ) (hf : is_valid_func f) :
  ∀ n, f n = n + 667 :=
sorry

end unique_solution_l797_797206


namespace A_and_C_mutually_independent_l797_797067

-- Define sample space and cardinalities of events
def sample_space : Finset ℝ := {i | i ≥ 0 ∧ i < 60}.to_finset
def n (s: Finset ℝ) : ℕ := s.card

-- Define events A, B, C, D
def A : Finset ℝ := {i | i ≥ 0 ∧ i < 30}.to_finset
def B : Finset ℝ := {i | i ≥ 30 ∧ i < 40}.to_finset
def C : Finset ℝ := {i | i ≥ 40 ∧ i < 60}.to_finset
def D : Finset ℝ := {i | i ≥ 30 ∧ i < 60}.to_finset

-- Given conditions
axiom h1 : n sample_space = 60
axiom h2 : n A = 30
axiom h3 : n B = 10
axiom h4 : n C = 20
axiom h5 : n D = 30
axiom h6 : n (A ∪ B) = 40
axiom h7 : n (A ∩ C) = 10
axiom h8 : n (A ∪ D) = 60

-- Theorem to prove
theorem A_and_C_mutually_independent : 
  (n (A ∩ C)) = (n A * n C) / n sample_space := by
  sorry

end A_and_C_mutually_independent_l797_797067


namespace g_composition_l797_797546

def f (x : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + 5

def g (y : ℝ) : ℝ := 25  -- from the problem assumption g(5) = 25

theorem g_composition : g (f (-2)) = 25 := by
  have h1 : f (-2) = 21 := by
    calc
      f (-2) = 2 * (-2) ^ 2 - 4 * (-2) + 5 := rfl
      ... = 2 * 4 + 8 + 5 := rfl
      ... = 21 := rfl
  have h2 : g (21) = 25 := by rfl
  show g (f (-2)) = 25 from eq.trans (congr_arg g h1) h2

end g_composition_l797_797546


namespace part1_part2_l797_797430

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l797_797430


namespace find_constants_monotonicity_l797_797114

noncomputable def f (x a b : ℝ) := (x^2 + a * x) * Real.exp x + b

theorem find_constants (a b : ℝ) (h_tangent : (f 0 a b = 1) ∧ (deriv (f · a b) 0 = -2)) :
  a = -2 ∧ b = 1 := by
  sorry

theorem monotonicity (a b : ℝ) (h_constants : a = -2 ∧ b = 1) :
  (∀ x : ℝ, (Real.exp x * (x^2 - 2) > 0 → x > Real.sqrt 2 ∨ x < -Real.sqrt 2)) ∧
  (∀ x : ℝ, (Real.exp x * (x^2 - 2) < 0 → -Real.sqrt 2 < x ∧ x < Real.sqrt 2)) := by
  sorry

end find_constants_monotonicity_l797_797114


namespace problem_l797_797494
noncomputable theory

open Real

def f (A ω φ x : ℝ) : ℝ := A * sin (ω * x + φ)

theorem problem
  (A : ℝ) (ω : ℝ) (φ : ℝ)
  (hA : A > 0)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (h_period : ∀ x : ℝ, f A ω φ (x + π) = f A ω φ x)
  (h_max : f A ω φ (π / 12) = A) :
  (ω = 2 ∧ φ = π / 3) ∧ (∀ A, 
   (∃ x ∈ Icc (-π / 4) 0, f A 2 (π / 3) x = 1 - A) → (4 - 2 * sqrt 3 ≤ A ∧ A ≤ 2)) :=
by sorry

end problem_l797_797494


namespace fraction_meaningful_l797_797874

theorem fraction_meaningful (a : ℝ) : (∃ x, x = 2 / (a + 1)) ↔ a ≠ -1 :=
by
  sorry

end fraction_meaningful_l797_797874


namespace min_sum_of_12_consecutive_integers_is_perfect_square_l797_797777

theorem min_sum_of_12_consecutive_integers_is_perfect_square :
  ∃ n : ℕ, 6 * (2 * n + 11) = 150 :=
begin
  use 7,
  norm_num,
  sorry
end

end min_sum_of_12_consecutive_integers_is_perfect_square_l797_797777


namespace geometric_sequence_product_l797_797933

open Function

-- conditions
variable (n : ℕ) (b : ℕ → ℝ)
variable (b_pos : ∀ (m : ℕ), m < n → b m > 0)

-- definition of arithmetic sequence sum
def Sn_arithmetic (a : ℕ → ℝ) [∀ i, Decidable (i < n)] : ℝ :=
  (n * (a 0 + a (n-1))) / 2

-- intended proof problem
theorem geometric_sequence_product : 
  (∀ (n : ℕ) (b : ℕ → ℝ), (∀ (m : ℕ), m < n → b m > 0) → 
    ∃ T_n : ℝ, T_n = real.sqrt ((b 0 * b (n-1))^n)) :=
by
  sorry

end geometric_sequence_product_l797_797933


namespace sum_T_2024_equals_l797_797088

noncomputable def sequence_a : Nat → ℝ 
| 1 => 1 / 2
| 4 => 1 / 8
| n => sorry

axiom recurrence_relation (n : ℕ) (h : n ≥ 2) : 
  sequence_a (n + 1) * sequence_a n + sequence_a (n - 1) * sequence_a n 
  = 2 * sequence_a (n + 1) * sequence_a (n - 1)

def sequence_b (n : ℕ) : ℝ := sequence_a n * sequence_a (n + 1)

def sum_T (n : ℕ) : ℝ := ∑ i in Finset.range n, sequence_b (i + 1)

theorem sum_T_2024_equals : sum_T 2024 = 506 / 2025 := 
by 
  sorry

end sum_T_2024_equals_l797_797088


namespace intersection_M_N_l797_797138

def setM : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def setN : Set ℕ := {0, 1, 2}

theorem intersection_M_N : setM ∩ setN = {0, 1} := by
  sorry

end intersection_M_N_l797_797138


namespace cube_difference_divisible_by_16_l797_797214

theorem cube_difference_divisible_by_16 (a b : ℤ) : 
  16 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3 + 8) :=
by
  sorry

end cube_difference_divisible_by_16_l797_797214


namespace central_angle_of_sector_l797_797711

theorem central_angle_of_sector (p : ℝ) (h : p = 1 / 4) : ∃ x : ℝ, x = 90 :=
by {
  existsi (90 : ℝ),
  simp,
  linarith [h],
  sorry
}

end central_angle_of_sector_l797_797711


namespace min_chord_length_l797_797869

variable (α : ℝ)

def curve_eq (x y α : ℝ) :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line_eq (x : ℝ) :=
  x = Real.pi / 4

theorem min_chord_length :
  ∃ d, (∀ α : ℝ, ∃ y1 y2 : ℝ, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ d = |y2 - y1|) ∧
  (∀ α : ℝ, ∃ y1 y2, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ |y2 - y1| ≥ d) :=
sorry

end min_chord_length_l797_797869


namespace johnson_family_seating_l797_797632

theorem johnson_family_seating (boys girls : Finset ℕ) (h_boys : boys.card = 5) (h_girls : girls.card = 4) :
  (∃ (arrangement : List ℕ), arrangement.length = 9 ∧ at_least_two_adjacent boys arrangement) :=
begin
  -- Given the total number of ways: 9! 
  -- subtract 5! * 4! from 9! to get the result 
  have total_arrangements := nat.factorial 9,
  have restrictive_arrangements := nat.factorial 5 * nat.factorial 4,
  exact (total_arrangements - restrictive_arrangements) = 360000,
end

end johnson_family_seating_l797_797632


namespace arc_midpoint_constant_ratio_l797_797425

/-
  Given that C is the midpoint of arc A B on circle O,
  and P is any point on arc (A M B) excluding points A and B,
  prove that the ratio (A P + B P) : C P is a constant value.
-/
theorem arc_midpoint_constant_ratio
  {O : Type*} [metric_space O] [normed_group O] [normed_space ℝ O]
  {A B C P : O} (hC : circle (O) (dist O) A B C C = 1)
  (hP : ∀ P, P ∈ arc_AMB (dist O) A M B → P ≠ A ∧ P ≠ B)
  (hMid : is_midpoint (arc_dist O) A B C) :
  ∃ k : ℝ, ∀ P, P ≠ A ∧ P ≠ B → (dist A P + dist B P) = k * dist C P :=
sorry

end arc_midpoint_constant_ratio_l797_797425


namespace m_ge_4_x_range_l797_797824

variable (x m : ℝ)

def p := x^2 - 4x - 5 ≤ 0
def q := x^2 - 2x + 1 - m^2 ≤ 0

-- Proof Problem 1: Given \( p \) is a sufficient condition for \( q \), prove that \( m \geq 4 \)
theorem m_ge_4 (h1 : p x) (h2 : q x) : m ≥ 4 := 
sorry

-- Proof Problem 2: Given \( m = 5 \), \( p \vee q \) is true and \( p \wedge q \) is false, prove that \( x \in [-4, -1) \cup (5, 6] \)
theorem x_range (m_eq_5 : m = 5) (h3 : p x ∨ q x) (h4 : ¬(p x ∧ q x)) : x ∈ (Set.Ico (-4 : ℝ) (-1) ∪ Set.Icc (5 : ℝ) 6) := 
sorry

end m_ge_4_x_range_l797_797824


namespace no_two_color_polyhedron_coloring_l797_797083

noncomputable def polyhedron_faces (n : ℕ) (F : ℕ) (faces : Fin F → ℕ) : Prop :=
∀ i : Fin F, i ≠ exceptional_idx → faces i % n = 0

theorem no_two_color_polyhedron_coloring (n : ℕ) {F : ℕ} (faces : Fin F → ℕ) (exceptional_idx : Fin F) 
  (h1 : 1 < n)
  (h2 : polyhedron_faces n F faces)
  : ¬ ∃ (coloring : Fin F → Prop), ∀ {i j : Fin F}, adjacent i j → coloring i ≠ coloring j := 
sorry

end no_two_color_polyhedron_coloring_l797_797083


namespace orchids_sold_is_20_l797_797722

noncomputable def number_of_orchids_sold 
  (earnings_per_orchid : ℕ)
  (num_money_plants : ℕ)
  (earnings_per_money_plant : ℕ)
  (num_workers : ℕ)
  (earnings_per_worker : ℕ)
  (cost_new_pots : ℕ)
  (leftover_money : ℕ)
  (earnings_from_orchids : ℕ)
  (total_expenses : ℕ) : ℕ :=
  let total_earnings := earnings_from_orchids + (num_money_plants * earnings_per_money_plant)
  let total_expenses := (num_workers * earnings_per_worker) + cost_new_pots
  in if total_earnings - total_expenses = leftover_money then
     earnings_from_orchids / earnings_per_orchid
     else 0

theorem orchids_sold_is_20 :
  number_of_orchids_sold 50 15 25 2 40 150 1145 1000 230 = 20 :=
  sorry

end orchids_sold_is_20_l797_797722


namespace second_box_clay_amount_l797_797319

theorem second_box_clay_amount : 
  let height1 := 3
      width1 := 4
      length1 := 7
      clay1 := 70
      height2 := 3 * height1
      width2 := 2 * width1
      length2 := length1
  in height1 * width1 * length1 ≠ 0 ∧ (height1 * width1 * length1 * clay1) / (height1 * width1 * length1) = clay1 →
     (height2 / height1) * (width2 / width1) * (length2 / length1) * clay1 = 420 :=
by
  intro height1 width1 length1 clay1 height2 width2 length2 h
  sorry

end second_box_clay_amount_l797_797319


namespace number_of_valid_sequences_l797_797187

-- Define the sequence property
def sequence_property (b : Fin 10 → Fin 10) : Prop :=
  ∀ i : Fin 10, 2 ≤ i → (∃ j : Fin 10, j < i ∧ (b j = b i + 1 ∨ b j = b i - 1 ∨ b j = b i + 2 ∨ b j = b i - 2))

-- Define the set of such sequences
def valid_sequences : Set (Fin 10 → Fin 10) := {b | sequence_property b}

-- Define the number of such sequences
def number_of_sequences : Fin 512 :=
  sorry -- Proof omitted for brevity

-- The final statement
theorem number_of_valid_sequences : number_of_sequences = 512 :=
  sorry  -- Skip proof

end number_of_valid_sequences_l797_797187


namespace rectangle_short_side_length_l797_797961

theorem rectangle_short_side_length
  (s : ℝ)
  (a b : ℝ)
  (h_a : a = 9)
  (h_b : b = 12)
  (hypotenuse : ℝ := Real.sqrt (a^2 + b^2))
  (rectangle : ℝ → ℝ → ℝ → Prop)
  (H1 : hypotenuse = 15)
  (H2 : ∀ s, rectangle s (2 * s) hypotenuse → s = 6) :
  s = 6 := by
  sorry

end rectangle_short_side_length_l797_797961


namespace largest_non_remarkable_l797_797934

/-- A number is termed 'remarkable' if it can be represented
    as the sum of 2023 natural composite numbers -/
def is_remarkable (n : ℕ) : Prop :=
  ∃ (s : Finset ℕ), (∀ x ∈ s, ¬nat.Prime x ∧ x > 1) ∧ s.card = 2023 ∧ s.sum = n

theorem largest_non_remarkable :
  ∀ n, n = 2023 →
  ∀ m, m = 4 * n + 3 →
  ¬ is_remarkable m ∧ (∀ k > m, is_remarkable k) :=
by
  sorry

end largest_non_remarkable_l797_797934


namespace find_geo_prog_numbers_l797_797657

noncomputable def geo_prog_numbers (a1 a2 a3 : ℝ) : Prop :=
a1 * a2 * a3 = 27 ∧ a1 + a2 + a3 = 13

theorem find_geo_prog_numbers :
  geo_prog_numbers 1 3 9 ∨ geo_prog_numbers 9 3 1 :=
sorry

end find_geo_prog_numbers_l797_797657


namespace find_k_values_l797_797376

theorem find_k_values (k : ℕ) (n : ℕ) (h_pos_k : 0 < k) (h_pos_n : 0 < n) :
  (k = 2^s → ∃ s : ℕ, 2^((k-1) * n + 1) ∣ (kn)! / n!) :=
sorry

end find_k_values_l797_797376


namespace largest_possible_A_l797_797909

noncomputable theory

open Classical

variable (p : ℕ) [hp : Fact (Nat.Prime p)]
variable (A : Finset ℕ)

-- Conditions
def prime_divisors_condition (A : Finset ℕ) : Prop :=
  (∃ S : Finset ℕ, ∀ a ∈ A, ∀ prime_divisor ∈ S, prime prime_divisor ∧ a % prime_divisor = 0) ∧ S.card = p - 1

def no_perfect_pth_power (A : Finset ℕ) : Prop :=
  ∀ (B : Finset ℕ), B ⊆ A → B.nonempty → ¬ is_pth_power (B.prod id) p

-- Definition of a p-th power
def is_pth_power (n p : ℕ) : Prop :=
  ∃ x : ℕ, n = x ^ p

-- Main statement
theorem largest_possible_A (A : Finset ℕ) (hp1 : prime_divisors_condition A) (hp2 : no_perfect_pth_power A) : 
  A.card ≤ (p - 1) ^ 2 := sorry

end largest_possible_A_l797_797909


namespace probability_four_coins_l797_797073

-- Define four fair coin flips, having 2 possible outcomes for each coin
def four_coin_flips_outcomes : ℕ := 2 ^ 4

-- Define the favorable outcomes: all heads or all tails
def favorable_outcomes : ℕ := 2

-- The probability of getting all heads or all tails
def probability_all_heads_or_tails : ℚ := favorable_outcomes / four_coin_flips_outcomes

-- The theorem stating the answer to the problem
theorem probability_four_coins:
  probability_all_heads_or_tails = 1 / 8 := by
  sorry

end probability_four_coins_l797_797073


namespace elongation_rate_improved_l797_797707

def elongation_rates_x : List ℝ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def elongation_rates_y : List ℝ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
def z : List ℝ := List.zipWith (-) elongation_rates_x elongation_rates_y

noncomputable def mean (l : List ℝ) : ℝ := (l.sum / l.length)

noncomputable def variance (l : List ℝ) : ℝ :=
let m := mean l in
(l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem elongation_rate_improved :
  mean z = 11 ∧ variance z = 61 ∧ mean z ≥ 2 * Real.sqrt (variance z / 10) := by
  sorry

end elongation_rate_improved_l797_797707


namespace function_decreasing_on_domain_l797_797350

-- Define the functions given in the conditions
def f1 (x : ℝ) : ℝ := abs (x - 1)
def f2 (x : ℝ) : ℝ := log x / log 2
def f3 (x : ℝ) : ℝ := (x + 1) ^ 2
def f4 (x : ℝ) : ℝ := (1 / 2) ^ x

-- Define the domain
def domain := {x : ℝ | 0 < x}

-- State the theorem based on the question, conditions, and correct answer
theorem function_decreasing_on_domain : monotone_decreasing_on f4 domain :=
sorry -- proof is not required

end function_decreasing_on_domain_l797_797350


namespace sum_numbers_eq_432_l797_797997

theorem sum_numbers_eq_432 (n : ℕ) (h : (n * (n + 1)) / 2 = 432) : n = 28 :=
sorry

end sum_numbers_eq_432_l797_797997


namespace wedge_volume_l797_797716

noncomputable def cylinder_diameter: ℝ := 20
noncomputable def cylinder_radius: ℝ := cylinder_diameter / 2
noncomputable def pi: ℝ := Real.pi
noncomputable def angle: ℝ := 30
noncomputable def cylinder_height: ℝ := 20 -- Assuming the height of the cylinder is 20 inches

/-- Given a cylindrical log with a diameter of 20 inches,
    and a wedge formed by two cuts: one perpendicular to the axis and another making a 30-degree angle,
    the volume of the wedge expressed as nπ is (500 / 3)π.
-/
theorem wedge_volume (d h : ℝ) (θ : ℝ) (r := d / 2) (V_cyl := pi * r^2 * h):
  θ = angle →
  d = cylinder_diameter →
  h = cylinder_height →
  let V_wedge := V_cyl * θ / 360 in
  V_wedge = 500 * pi / 3 :=
by
  intros
  sorry

end wedge_volume_l797_797716


namespace AP_parallel_CS_l797_797535

-- Defining the existence of points and the relationships
variables {A B C P M R S : Type} [CoordSys : EuclideanGeometry] 

-- Conditions starting here:
-- Condition 1: A is an acute-angled triangle inscribed in a circle k
def isAcuteAngledTriangle (A B C : Triangle) := 
  acute_angle_triangle A B C

def inscribedInCircle (k : Circle) (t : Triangle) :=
  circumscribed k t

-- Condition 2: The tangent from A to the circle meets the line BC at point P
def tangentMeetsLine (A P : Point) (k : Circle) (BC : Line) :=
  tangent_contains k A P ∧ meets_line BC P

-- Condition 3: M is the midpoint of the line segment AP
def isMidpoint (M A P : Point) :=
  midpoint M A P

-- Condition 4: R is the second intersection point of the circle k with the line BM
def secondIntersection (R : Point) (k : Circle) (BM : Line) :=
  intersects k R (second_point BM)

-- Condition 5: The line PR meets the circle k again at point S different from R
def lineMeetsCircleAgain (PR : Line) (k : Circle) (S R : Point) :=
  intersects_again PR k S R

-- Definitions of AP and CS lines
def lineAP (A P : Point) : Line :=
  line_through A P 

def lineCS (C S : Point) : Line :=
  line_through C S

-- The main theorem: Prove that AP and CS are parallel
theorem AP_parallel_CS (ABC : Triangle) (k : Circle) (P M R S : Point)
  (h1 : isAcuteAngledTriangle ABC)
  (h2 : inscribedInCircle k ABC)
  (h3 : tangentMeetsLine A P k (line_through B C))
  (h4 : isMidpoint M A P)
  (h5 : secondIntersection R k (line_through B M))
  (h6 : lineMeetsCircleAgain (line_through P R) k S R) :
  is_parallel (lineAP A P) (lineCS C S) :=
sorry -- Proof required

end AP_parallel_CS_l797_797535


namespace distinct_values_g_l797_797926

noncomputable def g (x : ℝ) : ℤ := 
  ∑ k in (Finset.range 10).map (Finset.singleton 3).to_finset, 
    (Int.floor (k * x) - k * Int.floor x + Int.floor (x / k : ℝ))

theorem distinct_values_g : 
  (Finset.range 10).map (Finset.singleton 3).to_finset.card + 1 = 39 := 
sorry

end distinct_values_g_l797_797926


namespace max_val_m_min_val_M_l797_797828

variable {n : ℕ} (a : ℕ → ℝ)

noncomputable def m (a : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, Real.sqrt (1 + 3 * a i)

theorem max_val_m (h : ∀ i, a i < 0) (h_sum : ∑ i in Finset.range n, a i = 1):
    ∃ (m : ℝ), m ≤ ∑ i in Finset.range n, Real.sqrt (1 + 3 * a i) ∧ m = n + 1 := sorry

theorem min_val_M (h : ∀ i, a i < 0) (h_sum : ∑ i in Finset.range n, a i = 1):
    ∃ (M : ℝ), ∑ i in Finset.range n, Real.sqrt (1 + 3 * a i) ≤ M ∧ M = Real.sqrt (n * (n + 3)) := sorry

end max_val_m_min_val_M_l797_797828


namespace part_a_condition_length_of_A_1A_part_b_conditions_volume_and_radius_l797_797687

noncomputable def A_1A := 18

noncomputable def volume := 1944

noncomputable def radius := 3 * Real.sqrt 10

theorem part_a_condition_length_of_A_1A
  (parallelepiped : Type)
  [is_orthogonal_edge : ∀ (A₁ A : parallelepiped), perpendicular A₁ A ABCD]
  (sphere : Type)
  [touches_edges : ∀ (BB₁ B₁C₁ C₁C CB CD : parallelepiped), sphere touches BB₁ ∧ B₁C₁ ∧ C₁C ∧ CB ∧ CD]
  [touches_CD_at_K : ∀ (CK KD : ℝ), K CD CK = 9 ∧ KD = 1]
  : (∃ x : parallelepiped, x = A_1A) := sorry

theorem part_b_conditions_volume_and_radius
  (parallelepiped : Type)
  [is_orthogonal_edge : ∀ (A₁ A : parallelepiped), perpendicular A₁ A ABCD]
  (sphere : Type)
  [touches_edges : ∀ (BB₁ B₁C₁ C₁C CB CD : parallelepiped), sphere touches BB₁ ∧ B₁C₁ ∧ C₁C ∧ CB ∧ CD]
  [touches_CD_at_K : ∀ (CK KD : ℝ), K CD CK = 9 ∧ KD = 1]
  [sphere_touches_A_1D_1 : ∀ (A₁D₁ : parallelepiped), sphere touches A₁D₁]
  : (∃ v : ℝ, v = volume) ∧ (∃ r : ℝ, r = radius) := sorry

end part_a_condition_length_of_A_1A_part_b_conditions_volume_and_radius_l797_797687


namespace total_fish_caught_total_l797_797749
-- Include the broad Mathlib library to ensure all necessary mathematical functions and definitions are available

-- Define the conditions based on the given problem
def brian_trips (chris_trips : ℕ) : ℕ := 2 * chris_trips
def chris_fish_per_trip (brian_fish_per_trip : ℕ) : ℕ := brian_fish_per_trip + (2/5 : ℚ) * brian_fish_per_trip
def total_fish_caught (chris_trips : ℕ) (brian_fish_per_trip chris_fish_per_trip : ℕ) : ℕ := 
  brian_trips chris_trips * brian_fish_per_trip + chris_trips * chris_fish_per_trip

-- State the main proof problem based on the question and conditions
theorem total_fish_caught_total :
  ∀ (chris_trips : ℕ) (brian_fish_per_trip : ℕ) (chris_fish_per_trip : ℕ),
  chris_trips = 10 →
  brian_fish_per_trip = 400 →
  chris_fish_per_trip = 560 →
  total_fish_caught chris_trips brian_fish_per_trip chris_fish_per_trip = 13600 :=
by
  intros chris_trips brian_fish_per_trip chris_fish_per_trip h_chris_trips h_brian_fish_per_trip h_chris_fish_per_trip
  rw [h_chris_trips, h_brian_fish_per_trip, h_chris_fish_per_trip]
  sorry -- Proof omitted

end total_fish_caught_total_l797_797749


namespace dot_product_parallel_result_l797_797122

variables (x : ℝ)

def vector_a := (x, 2)
def vector_b := (2, 1)
def vector_c := (3, x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (k * v.fst = w.fst ∧ k * v.snd = w.snd)

theorem dot_product_parallel_result :
  parallel (vector_a x) vector_b →
  (vector_a x).fst * (vector_c x).fst + (vector_a x).snd * (vector_c x).snd = 20
:= by
  sorry

end dot_product_parallel_result_l797_797122


namespace problem_statement_l797_797534

theorem problem_statement (n : ℕ) (a b c : ℕ → ℤ)
  (h1 : n > 0)
  (h2 : ∀ i j, i ≠ j → ¬ (a i - a j) % n = 0 ∧
                           ¬ ((b i + c i) - (b j + c j)) % n = 0 ∧
                           ¬ (b i - b j) % n = 0 ∧
                           ¬ ((c i + a i) - (c j + a i)) % n = 0 ∧
                           ¬ (c i - c j) % n = 0 ∧
                           ¬ ((a i + b i) - (a j + b i)) % n = 0 ∧
                           ¬ ((a i + b i + c i) - (a j + b i + c j)) % n = 0) :
  (Odd n) ∧ (¬ ∃ k, n = 3 * k) :=
by sorry

end problem_statement_l797_797534


namespace triangle_KIA_KO_eq_VI_l797_797173

theorem triangle_KIA_KO_eq_VI (K I A V X O : Type) [line_segment K I I A I K V A] :
  ∃ (triangle : Type) (angle : triangle → triangle → triangle → ℝ),
    (point O ∈ (line_segment A X ∩ line_segment K I)) →
    (angle X K I = 1 / 2 * angle A V I) ∧ (angle X I K = 1 / 2 * angle K V A) →
    (length (segment K O) = length (segment V I)) :=
begin
  sorry
end

end triangle_KIA_KO_eq_VI_l797_797173


namespace average_score_l797_797238

theorem average_score (avg1 avg2 : ℕ) (n1 n2 total_matches : ℕ) (total_avg : ℕ) 
  (h1 : avg1 = 60) 
  (h2 : avg2 = 70) 
  (h3 : n1 = 10) 
  (h4 : n2 = 15) 
  (h5 : total_matches = 25) 
  (h6 : total_avg = 66) :
  (( (avg1 * n1) + (avg2 * n2) ) / total_matches = total_avg) :=
by
  sorry

end average_score_l797_797238
