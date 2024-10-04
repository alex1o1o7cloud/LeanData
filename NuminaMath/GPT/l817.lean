import Mathlib
import Mathlib.Algebra.ArithSeq
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle
import Mathlib.Geometry.Triangle
import Mathlib.GraphTheory.Basic
import Mathlib.GroupTheory.Subgroup
import Mathlib.Integration
import Mathlib.Mathlib
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology.Basic

namespace campers_difference_l817_817252

theorem campers_difference 
       (total : ℕ)
       (campers_two_weeks_ago : ℕ) 
       (campers_last_week : ℕ) 
       (diff: ℕ)
       (h_total : total = 150)
       (h_two_weeks_ago : campers_two_weeks_ago = 40) 
       (h_last_week : campers_last_week = 80) : 
       diff = campers_two_weeks_ago - (total - campers_two_weeks_ago - campers_last_week) :=
by
  sorry

end campers_difference_l817_817252


namespace two_digit_integers_remainder_3_count_l817_817881

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817881


namespace count_two_digit_integers_remainder_3_l817_817870

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817870


namespace angle_between_unit_vectors_l817_817995

variables {ℝ : Type*} [inner_product_space ℝ] (a b c : ℝ)

/-- a, b, and c are unit vectors -/
def is_unit_vector (v : ℝ) : Prop := inner v v = 1

-- Given conditions
variables (hu_a : is_unit_vector a) (hu_b : is_unit_vector b) (hu_c : is_unit_vector c)
variables (eqn : a + 2 • b + 2 • c = 0)

-- Proving that the angle between a and b is 104.5 degrees
theorem angle_between_unit_vectors : 
  acos (inner a b) = 104.5 :=
sorry

end angle_between_unit_vectors_l817_817995


namespace binomial_expansion_and_constant_term_l817_817073

theorem binomial_expansion_and_constant_term :
  ∀ (n : ℕ), (∑ k in finset.range (n+1), nat.choose n k) = 64 →
  n = 6 ∧ 
  (nat.choose 6 2 * 2^4 : ℕ) = 240 :=
by
  intros
  sorry

end binomial_expansion_and_constant_term_l817_817073


namespace smallest_four_digit_equiv_8_mod_9_l817_817214

theorem smallest_four_digit_equiv_8_mod_9 :
  ∃ n : ℕ, n % 9 = 8 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 9 = 8 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m :=
sorry

end smallest_four_digit_equiv_8_mod_9_l817_817214


namespace staircase_steps_180_toothpicks_l817_817545

-- Condition definition: total number of toothpicks for \( n \) steps is \( n(n + 1) \)
def total_toothpicks (n : ℕ) : ℕ := n * (n + 1)

-- Theorem statement: for 180 toothpicks, the number of steps \( n \) is 12
theorem staircase_steps_180_toothpicks : ∃ n : ℕ, total_toothpicks n = 180 ∧ n = 12 :=
by sorry

end staircase_steps_180_toothpicks_l817_817545


namespace seven_not_spheric_spheric_power_spheric_l817_817608

def is_spheric (r : ℚ) : Prop := ∃ x y z : ℚ, r = x^2 + y^2 + z^2

theorem seven_not_spheric : ¬ is_spheric 7 := 
sorry

theorem spheric_power_spheric (r : ℚ) (n : ℕ) (h : is_spheric r) (hn : n > 1) : is_spheric (r ^ n) := 
sorry

end seven_not_spheric_spheric_power_spheric_l817_817608


namespace meaningful_fraction_l817_817951

theorem meaningful_fraction {a : ℝ} : 2 * a - 1 ≠ 0 ↔ a ≠ 1 / 2 :=
by sorry

end meaningful_fraction_l817_817951


namespace jame_weeks_tearing_cards_l817_817095

def cards_tears_per_time : ℕ := 30
def cards_per_deck : ℕ := 55
def tears_per_week : ℕ := 3
def decks_bought : ℕ := 18

theorem jame_weeks_tearing_cards :
  (cards_tears_per_time * tears_per_week * decks_bought * cards_per_deck) / (cards_tears_per_time * tears_per_week) = 11 := by
  sorry

end jame_weeks_tearing_cards_l817_817095


namespace count_multiples_5_or_7_but_not_both_l817_817796

-- Definitions based on the given problem conditions
def multiples_of_five (n : Nat) : Nat :=
  (n - 1) / 5

def multiples_of_seven (n : Nat) : Nat :=
  (n - 1) / 7

def multiples_of_thirty_five (n : Nat) : Nat :=
  (n - 1) / 35

def count_multiples (n : Nat) : Nat :=
  (multiples_of_five n) + (multiples_of_seven n) - 2 * (multiples_of_thirty_five n)

-- The main statement to be proved
theorem count_multiples_5_or_7_but_not_both : count_multiples 101 = 30 :=
by
  sorry

end count_multiples_5_or_7_but_not_both_l817_817796


namespace perpendicular_lines_in_parallel_planes_l817_817508

variable (a b : Type) [Line a] [Line b]
variable (α β : Type) [Plane α] [Plane β]

theorem perpendicular_lines_in_parallel_planes
  (h1 : a ⊥ α)
  (h2 : b ∈ β)
  (h3 : α ∥ β) :
  a ⊥ b := 
  sorry

end perpendicular_lines_in_parallel_planes_l817_817508


namespace count_two_digit_integers_remainder_3_l817_817824

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817824


namespace toy_discount_price_l817_817193

theorem toy_discount_price (original_price : ℝ) (discount_rate : ℝ) (price_after_first_discount : ℝ) (price_after_second_discount : ℝ) : 
  original_price = 200 → 
  discount_rate = 0.1 →
  price_after_first_discount = original_price * (1 - discount_rate) →
  price_after_second_discount = price_after_first_discount * (1 - discount_rate) →
  price_after_second_discount = 162 :=
by
  intros h1 h2 h3 h4
  sorry

end toy_discount_price_l817_817193


namespace total_time_equation_l817_817678

-- Definitions for conditions
def speed_escalator1 : ℝ := 10
def length_escalator1 : ℝ := 112
def speed_walk_person : ℝ := 4
def speed_walkway2 : ℝ := 6
def length_walkway2 : ℝ := 80

-- Calculate time taken on each section
def time_on_escalator1 := length_escalator1 / (speed_escalator1 + speed_walk_person)
def time_on_walkway2 := length_walkway2 / (speed_walkway2 + speed_walk_person)

-- Total time calculation (to be proven)
theorem total_time_equation :
  time_on_escalator1 + time_on_walkway2 = 16 := 
sorry

end total_time_equation_l817_817678


namespace count_two_digit_integers_with_remainder_3_l817_817844

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817844


namespace fedya_gives_different_answers_l817_817430

-- Define the condition that Fedya always tells the truth
def always_truthful (Fedya : Type) (response : Fedya → Prop) : Prop :=
  ∀ q t1 t2, (response q t1) ∧ (response q t2) → t1 = t2

-- Define the main proposition: it is possible for Fedya to give different answers twice in a row
theorem fedya_gives_different_answers (Fedya : Type) (response : Fedya → Prop) 
    (H : always_truthful Fedya response) : ∃ (q : Prop) (t1 t2 : Fedya), t1 ≠ t2 ∧ (response q t1) ∧ (response q t2) :=
sorry

end fedya_gives_different_answers_l817_817430


namespace find_f_find_m_range_l817_817031

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 / 3) * x^3 + a * x + b

-- Define the conditions
variables (a b : ℝ)
axiom local_min_condition : a = -4 ∧ b = 4 ∧ f(2) = -4 / 3
axiom inequality_condition : ∀ x ∈ Icc (-4 : ℝ) (3 : ℝ), f(x) ≤ m^2 + m + (10 / 3)

-- Prove the first part
theorem find_f :
  (∀ x : ℝ, f(x) = (1 / 3) * x^3 - 4 * x + 4) :=
by sorry

-- Prove the second part
def valid_m (m : ℝ) : Prop := m ≥ 2 ∨ m ≤ -3

theorem find_m_range :
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Icc (-4 : ℝ) (3 : ℝ) → f(x) ≤ m^2 + m + (10 / 3)) ↔ valid_m m) :=
by sorry

end find_f_find_m_range_l817_817031


namespace min_value_of_u_l817_817766

theorem min_value_of_u (x y : ℝ) (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) (hxy : x * y = -1) :
  (∀ u, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) → u ≥ (12 / 5)) :=
by
  sorry

end min_value_of_u_l817_817766


namespace marbles_problem_a_marbles_problem_b_l817_817991

-- Define the problem as Lean statements.

-- Part (a): m = 2004, n = 2006
theorem marbles_problem_a (m n : ℕ) (h_m : m = 2004) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) := 
sorry

-- Part (b): m = 2005, n = 2006
theorem marbles_problem_b (m n : ℕ) (h_m : m = 2005) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) → false := 
sorry

end marbles_problem_a_marbles_problem_b_l817_817991


namespace perimeter_square_l817_817668

theorem perimeter_square (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 7 = a^2 + 4 * a + 3) (h3 : 7 = b^2 + 4 * b + 3) : 
  4 * real.sqrt (2 * (a - b)^2) = 16 * real.sqrt 2 := 
by
  sorry

end perimeter_square_l817_817668


namespace remaining_pieces_total_l817_817452

noncomputable def initial_pieces : Nat := 16
noncomputable def kennedy_lost_pieces : Nat := 4 + 1 + 2
noncomputable def riley_lost_pieces : Nat := 1 + 1 + 1

theorem remaining_pieces_total : (initial_pieces - kennedy_lost_pieces) + (initial_pieces - riley_lost_pieces) = 22 := by
  sorry

end remaining_pieces_total_l817_817452


namespace alice_adds_101_to_50_sq_l817_817190

theorem alice_adds_101_to_50_sq (a b : ℕ) (h1 : a = 50) (h2 : b = 51) : b^2 = a^2 + 101 := 
by {
  rw [h1, h2],
  calc (50 + 1)^2 
        = 50^2 + 2 * 50 * 1 + 1^2 : by ring
    ... = 2500 + 100 + 1 : by norm_num
    ... = 2500 + 101 : by ring
}

end alice_adds_101_to_50_sq_l817_817190


namespace number_of_cars_in_second_box_is_31_l817_817988

-- Define the total number of toy cars, and the number of toy cars in the first and third boxes
def total_toy_cars : ℕ := 71
def cars_in_first_box : ℕ := 21
def cars_in_third_box : ℕ := 19

-- Define the number of toy cars in the second box
def cars_in_second_box : ℕ := total_toy_cars - cars_in_first_box - cars_in_third_box

-- Theorem stating that the number of toy cars in the second box is 31
theorem number_of_cars_in_second_box_is_31 : cars_in_second_box = 31 :=
by
  sorry

end number_of_cars_in_second_box_is_31_l817_817988


namespace black_king_eventually_in_check_l817_817641

theorem black_king_eventually_in_check 
  (n : ℕ) (h1 : n = 1000) (r : ℕ) (h2 : r = 499)
  (rooks : Fin r → (ℕ × ℕ)) (king : ℕ × ℕ)
  (take_not_allowed : ∀ rk : Fin r, rooks rk ≠ king) :
  ∃ m : ℕ, m ≤ 1000 ∧ (∃ t : Fin r, rooks t = king) :=
by
  sorry

end black_king_eventually_in_check_l817_817641


namespace num_ways_100_yuan_l817_817228

def num_ways_to_spend (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 3
  else num_ways_to_spend (n - 1) + 2 * num_ways_to_spend (n - 2)

theorem num_ways_100_yuan : num_ways_to_spend 100 = (2^101 + 1) / 3 := 
  sorry

end num_ways_100_yuan_l817_817228


namespace minimum_value_PQ_l817_817443

noncomputable def minimum_distance (A B Q : ℝ × ℝ) : ℝ :=
  -- Define the circles C1 and C2
  let C1 := (λ P : ℝ × ℝ, (P.1 - 2)^2 + P.2^2 = 3)
  let C2 := (λ P : ℝ × ℝ, (P.1 + 2)^2 + P.2^2 = 1)
  -- Define the distance calculation
  let dist := (λ P Q : ℝ × ℝ, Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  -- Calculate midpoint P
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  -- apply assumption |AB| = 2√2
  let hAB := dist A B = 2 * Real.sqrt 2
  -- find the minimum distance |PQ|
  Real.Inf {PQ : ℝ | ∃ (A B : ℝ × ℝ) (P Q : ℝ × ℝ), C1 A ∧ C1 B ∧ C2 Q ∧ (dist A B = 2 * Real.sqrt 2) ∧ (P.1 = (A.1 + B.1) / 2) ∧ (P.2 = (A.2 + B.2) / 2) ∧ (PQ = dist P Q)}

theorem minimum_value_PQ : minimum_distance = 2 :=
sorry  -- This is placeholder to indicate the point where the proof would be.

end minimum_value_PQ_l817_817443


namespace magic_square_existence_l817_817699

def is_magic_square (M : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  let row_sums := (λ i, ∑ j, M i j)
  let col_sums := (λ j, ∑ i, M i j)
  let main_diag_sum := ∑ i, M i i
  let anti_diag_sum := ∑ i, M i (3 - i)
  ∀ i, row_sums i = -18 ∧ col_sums i = -18 ∧ main_diag_sum = -18 ∧ anti_diag_sum = -18

theorem magic_square_existence : 
  ∃ (M : Matrix (Fin 4) (Fin 4) ℤ), 
    (∀ i j, (M i j) ∈ Finset.range 16) ∧ 
    ∑ i j, M i j = -72 ∧
    is_magic_square M :=
sorry

end magic_square_existence_l817_817699


namespace same_function_pair_C_l817_817707

theorem same_function_pair_C (f g : ℝ → ℝ) :
    (f = λ x, abs x) ∧ (g = λ x, real.sqrt (x^2)) → (f = g) ∧
    ¬(∀ x, (f = λ x, 1 ∧ g = λ x, x / x) ∨
            (f = λ x, x - 2 ∧ g = λ x, (x^2 - 4) / (x + 2)) ∨
            (f = λ x, real.sqrt (x + 1) * real.sqrt (x - 1) ∧ g = λ x, real.sqrt (x^2 - 1))) := by
  sorry

end same_function_pair_C_l817_817707


namespace count_two_digit_integers_remainder_3_l817_817864

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817864


namespace monomial_same_type_l817_817447

theorem monomial_same_type (a b : ℕ) (h1 : a + 1 = 3) (h2 : b = 3) : a + b = 5 :=
by 
  -- proof goes here
  sorry

end monomial_same_type_l817_817447


namespace solve_equation_l817_817587

theorem solve_equation :
  ∃ x : ℝ, (x = -2) → (1 + 2^x) / (1 + 2^(-x)) = 1 / 4 :=
by
  existsi (-2:ℝ)
  intro h
  rw h
  norm_num
  sorry

end solve_equation_l817_817587


namespace fence_perimeter_l817_817189

theorem fence_perimeter 
  (N : ℕ) (w : ℝ) (g : ℝ) 
  (square_posts : N = 36) 
  (post_width : w = 0.5) 
  (gap_length : g = 8) :
  4 * ((N / 4 - 1) * g + (N / 4) * w) = 274 :=
by
  sorry

end fence_perimeter_l817_817189


namespace tips_fraction_l817_817619

variable (A : ℝ) -- Average tips for the 6 months other than August

theorem tips_fraction (h : ∀ (A : ℝ), A > 0) : 
  let total_tips_others := 6 * A in
  let tips_august := 8 * A in
  let total_tips_all := total_tips_others + tips_august in
  tips_august / total_tips_all = 4 / 7 :=
by
  sorry

end tips_fraction_l817_817619


namespace cats_left_after_sale_l817_817248

theorem cats_left_after_sale (siamese house cats_sold : ℕ) (h1 : siamese = 15) (h2 : house = 49) (h3 : cats_sold = 19) : siamese + house - cats_sold = 45 :=
by {
   rw [h1, h2, h3],
   norm_num,
}

end cats_left_after_sale_l817_817248


namespace friends_truth_l817_817597

-- Definitions for the truth values of the friends
def F₁_truth (a x₁ x₂ x₃ : Prop) : Prop := a ↔ ¬ (x₁ ∨ x₂ ∨ x₃)
def F₂_truth (b x₁ x₂ x₃ : Prop) : Prop := b ↔ (x₂ ∧ ¬ x₁ ∧ ¬ x₃)
def F₃_truth (c x₁ x₂ x₃ : Prop) : Prop := c ↔ x₃

-- Main theorem statement
theorem friends_truth (a b c x₁ x₂ x₃ : Prop) 
  (H₁ : F₁_truth a x₁ x₂ x₃) 
  (H₂ : F₂_truth b x₁ x₂ x₃) 
  (H₃ : F₃_truth c x₁ x₂ x₃)
  (H₄ : a ∨ b ∨ c) 
  (H₅ : ¬ (a ∧ b ∧ c)) : a ∧ ¬b ∧ ¬c ∨ ¬a ∧ b ∧ ¬c ∨ ¬a ∧ ¬b ∧ c :=
sorry

end friends_truth_l817_817597


namespace emilys_calculation_is_correct_l817_817709

-- Definition of numbers being added
def num1 : ℕ := 58
def num2 : ℕ := 44

-- Definition of the multiplication factor
def factor : ℕ := 3

-- Perform arithmetic operations followed by rounding
def calculate_without_rounding (a b c : ℕ) : ℕ := ((a + b) * c).round(-2)

-- The main theorem statement
theorem emilys_calculation_is_correct : calculate_without_rounding num1 num2 factor = 300 :=
sorry

end emilys_calculation_is_correct_l817_817709


namespace actual_cost_of_article_l817_817624

theorem actual_cost_of_article (x : ℝ) (h : 0.76 * x = 684) : x = 900 :=
sorry

end actual_cost_of_article_l817_817624


namespace count_two_digit_integers_with_remainder_3_l817_817846

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817846


namespace range_of_a_l817_817025

theorem range_of_a (a : ℝ) (h : 0 < a ∧ a < 2) (h_ineq : (sin (1 - a) + 5 * (1 - a)) + (sin (1 - a^2) + 5 * (1 - a^2)) < 0) : 1 < a ∧ a < real.sqrt 2 :=
sorry

end range_of_a_l817_817025


namespace sum_of_primes_conditioned_l817_817614

noncomputable def is_prime (n : ℕ) : Prop := sorry
noncomputable def reverse_digits (n : ℕ) : ℕ := sorry
noncomputable def has_at_least_two_identical_digits (n : ℕ) : Prop := sorry

theorem sum_of_primes_conditioned : 
  (finset.range 500).filter (λ n, 
    n > 100 ∧
    is_prime n ∧
    is_prime (reverse_digits n) ∧
    has_at_least_two_identical_digits n
  ).sum = 1895 := 
sorry

end sum_of_primes_conditioned_l817_817614


namespace total_spent_amount_l817_817380

-- Define the conditions
def spent_relation (B D : ℝ) : Prop := D = 0.75 * B
def payment_difference (B D : ℝ) : Prop := B = D + 12.50

-- Define the theorem to prove
theorem total_spent_amount (B D : ℝ) 
  (h1 : spent_relation B D) 
  (h2 : payment_difference B D) : 
  B + D = 87.50 :=
sorry

end total_spent_amount_l817_817380


namespace find_y_l817_817920

-- Define G function
def G (a b c d : ℕ) : ℕ := a^b + c^d

-- Define the conditions
variables (y : ℝ)
axiom G_def : ∀ (a b c d : ℕ), G a b c d = a^b + c^d
axiom condition1 : G 3 y.to_nat 2 5 = 350

-- State the theorem
theorem find_y : y ≈ 6.204 :=
  sorry

end find_y_l817_817920


namespace polynomial_series_sum_l817_817149

theorem polynomial_series_sum (x : ℝ) (hx : x ≠ 1) (hx_eq : x ^ 2023 - 3 * x + 2 = 0) : 
  x ^ 2022 + x ^ 2021 + ... + x + 1 = 3 :=
sorry

end polynomial_series_sum_l817_817149


namespace gel_pen_price_ratio_l817_817301

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l817_817301


namespace range_of_a_l817_817434

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, exp(x) - a*x ≥ -x + log (a*x)) : 
  0 < a ∧ a ≤ exp(1) :=
sorry

end range_of_a_l817_817434


namespace min_value_expression_l817_817362

theorem min_value_expression (x y : ℝ) : ∃ (a b : ℝ), x = a ∧ y = b ∧ (x^2 + y^2 - 8*x - 6*y + 30 = 5) :=
by
  sorry

end min_value_expression_l817_817362


namespace PropA_impl_PropB_not_PropB_impl_PropA_l817_817394

variable {x : ℝ}

def PropA (x : ℝ) : Prop := abs (x - 1) < 5
def PropB (x : ℝ) : Prop := abs (abs x - 1) < 5

theorem PropA_impl_PropB : PropA x → PropB x :=
by sorry

theorem not_PropB_impl_PropA : ¬(PropB x → PropA x) :=
by sorry

end PropA_impl_PropB_not_PropB_impl_PropA_l817_817394


namespace production_cost_per_performance_l817_817557

def overhead_cost := 81000
def income_per_performance := 16000
def performances_needed := 9

theorem production_cost_per_performance :
  ∃ P, 9 * income_per_performance = overhead_cost + 9 * P ∧ P = 7000 :=
by
  sorry

end production_cost_per_performance_l817_817557


namespace alex_remaining_money_l817_817675

variable (income : ℕ) (tax_rate : ℚ) (water_bill : ℕ) (tithe_rate : ℚ) (groceries : ℕ) (transportation : ℕ)

def weekly_income := 900
def weekly_tax := 0.15
def weekly_water_bill := 75
def weekly_tithe := 0.20
def weekly_groceries := 150
def weekly_transportation := 50

theorem alex_remaining_money :
  income = 900 ∧ tax_rate = 0.15 ∧ water_bill = 75 ∧ tithe_rate = 0.20 ∧ groceries = 150 ∧ transportation = 50 →
  let remaining_income := 
    income - (income * tax_rate).to_nat - water_bill - (income * tithe_rate).to_nat - groceries - transportation in
  remaining_income = 310 :=
by
  sorry

end alex_remaining_money_l817_817675


namespace palindromes_count_l817_817201

open Nat

def valid_digits : Finset ℕ := {0, 7, 8, 9}

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

def num_9_digit_palindromes : ℕ :=
  (Finset.range 10^9).filter (λ n, n ≥ 10^8 ∧ is_palindrome n ∧ ∀ d ∈ n.digits 10, d ∈ valid_digits).card

theorem palindromes_count : num_9_digit_palindromes = 768 :=
  sorry

end palindromes_count_l817_817201


namespace principal_value_complex_conjugate_arg_l817_817020

noncomputable def complex_conjugate_arg_theta (θ : ℝ) (hθ1 : Real.pi / 2 < θ) (hθ2 : θ < Real.pi) : Prop :=
  let z := Complex.mk (1 - Real.sin θ) (Real.cos θ)
  let z_conjugate := Complex.conj z
  Complex.arg z_conjugate = (3 * Real.pi / 4) - θ / 2

theorem principal_value_complex_conjugate_arg (θ : ℝ) (hθ1 : Real.pi / 2 < θ) (hθ2 : θ < Real.pi) :
  complex_conjugate_arg_theta θ hθ1 hθ2 :=
sorry

end principal_value_complex_conjugate_arg_l817_817020


namespace subcommittees_with_coach_l817_817256

theorem subcommittees_with_coach (total_members : ℕ) (coaches : ℕ) (subcommittee_size : ℕ)
  (h_total : total_members = 12)
  (h_coaches : coaches = 5)
  (h_subcommittee_size : subcommittee_size = 5) 
  : nat.choose 12 5 - nat.choose 7 5 = 771 := 
by
  rw [h_total, h_coaches, h_subcommittee_size]
  -- Here Lean would continue with the proof after this point
  sorry

end subcommittees_with_coach_l817_817256


namespace a_must_be_negative_l817_817051

theorem a_must_be_negative (a b : ℝ) (h1 : b > 0) (h2 : a / b < -2 / 3) : a < 0 :=
sorry

end a_must_be_negative_l817_817051


namespace next_sales_amount_l817_817650

theorem next_sales_amount (R1 R2 S1 S2 X : ℝ)
  (hR1 : R1 = 2) -- Royalties received on the first 10 million sales
  (hS1 : S1 = 10) -- Initial sales amount in millions
  (hR2 : R2 = 8) -- Royalties received on the next sales
  (h_decrease : (R2 / X) = 0.4 * (R1 / S1)) -- Royalty rate decreased by 60%
  : 
  X = 100 :=
begin
  sorry
end

end next_sales_amount_l817_817650


namespace exponential_inequality_l817_817398

theorem exponential_inequality (a b : ℝ) (h : a < b) : 2^a < 2^b :=
sorry

end exponential_inequality_l817_817398


namespace complement_of_A_with_respect_to_U_l817_817748

def U := {x : ℕ | (x + 1) / (x - 5) ≤ 0}
def A := {1, 2, 4}

theorem complement_of_A_with_respect_to_U :
  (U \ A) = {0, 3} := by
  sorry

end complement_of_A_with_respect_to_U_l817_817748


namespace cost_price_eq_560_l817_817275

variables (C SP1 SP2 : ℝ)
variables (h1 : SP1 = 0.79 * C) (h2 : SP2 = SP1 + 140) (h3 : SP2 = 1.04 * C)

theorem cost_price_eq_560 : C = 560 :=
by 
  sorry

end cost_price_eq_560_l817_817275


namespace parabola_standard_equations_l817_817590

theorem parabola_standard_equations (x y : ℝ) (p : ℝ) :
  (x = 2 ∧ y = 4) →
  ((y^2 = 2 * p * x ∧ y = 4 ∧ x = 2) ∨ (x^2 = 2 * p * y ∧ x = 2 ∧ y = 4)) →
  (y^2 = 8 * x ∨ x^2 = y) :=
by {
  intro h,
  cases h,
  case h_eq { sorry }
}

end parabola_standard_equations_l817_817590


namespace max_pots_l817_817560

theorem max_pots (x y z : ℕ) (h₁ : 3 * x + 4 * y + 9 * z = 100) (h₂ : 1 ≤ x) (h₃ : 1 ≤ y) (h₄ : 1 ≤ z) : 
  z ≤ 10 :=
sorry

end max_pots_l817_817560


namespace music_books_cost_l817_817490

theorem music_books_cost
  (total_money : ℕ) (maths_books_count : ℕ) (maths_books_price : ℕ)
  (science_books_extra_count : ℕ) (science_books_price : ℕ)
  (art_books_multiplier : ℕ) (art_books_price : ℕ) :
  total_money = 500 →
  maths_books_count = 4 →
  maths_books_price = 20 →
  science_books_extra_count = 6 →
  science_books_price = 10 →
  art_books_multiplier = 2 →
  art_books_price = 20 →
  let
    maths_books_cost := maths_books_count * maths_books_price
    science_books_cost := (maths_books_count + science_books_extra_count) * science_books_price
    art_books_cost := (art_books_multiplier * maths_books_count) * art_books_price
    total_cost_excluding_music := maths_books_cost + science_books_cost + art_books_cost
    music_books_cost := total_money - total_cost_excluding_music
  in music_books_cost = 160 :=
by
  intros
  sorry

end music_books_cost_l817_817490


namespace count_two_digit_remainders_l817_817798

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817798


namespace ratio_of_surface_areas_of_spheres_l817_817954

theorem ratio_of_surface_areas_of_spheres (V1 V2 S1 S2 : ℝ) 
(h : V1 / V2 = 8 / 27) 
(h1 : S1 = 4 * π * (V1^(2/3)) / (2 * π)^(2/3))
(h2 : S2 = 4 * π * (V2^(2/3)) / (3 * π)^(2/3)) :
S1 / S2 = 4 / 9 :=
sorry

end ratio_of_surface_areas_of_spheres_l817_817954


namespace gel_pen_price_relation_b_l817_817290

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l817_817290


namespace count_two_digit_integers_remainder_3_l817_817828

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817828


namespace section_divides_cube_equally_passes_center_l817_817438

theorem section_divides_cube_equally_passes_center (C : Cube) (S : Set ℝ³) 
  (h1 : divides_cube_equally S C) : passes_center S C := 
sorry

end section_divides_cube_equally_passes_center_l817_817438


namespace largest_three_digit_product_of_prime_factors_l817_817578

theorem largest_three_digit_product_of_prime_factors : 
  ∃ n x y : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n = x * y * (10 * x + y) ∧ 
  x ∈ {2, 3, 5, 7} ∧ ¬ (x + y) % 3 = 0 ∧ Nat.Prime (10 * x + y) ∧ 
  n = 795 := 
by 
  sorry

end largest_three_digit_product_of_prime_factors_l817_817578


namespace root_in_interval_l817_817165

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 3 * x^2 + 1

theorem root_in_interval : ∃! x ∈ Ioo 0 2, f x = 0 :=
by 
  sorry

end root_in_interval_l817_817165


namespace positive_two_digit_integers_remainder_3_l817_817810

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817810


namespace train_speed_is_63_00468_km_per_hr_l817_817644

   -- Definition of conversion constants and functions
   def km_per_hr_to_m_per_s (v : ℝ) : ℝ := v * (1000 / 3600)
   def m_per_s_to_km_per_hr (v : ℝ) : ℝ := v * (3600 / 1000)

   -- Given conditions
   def train_length : ℝ := 300
   def man_speed_km_per_hr : ℝ := 3
   def time_to_cross_man : ℝ := 17.998560115190784

   theorem train_speed_is_63_00468_km_per_hr :
     let man_speed_m_per_s := km_per_hr_to_m_per_s man_speed_km_per_hr,
         relative_speed := train_length / time_to_cross_man,
         train_speed_m_per_s := relative_speed + man_speed_m_per_s,
         train_speed_km_per_hr := m_per_s_to_km_per_hr train_speed_m_per_s
     in train_speed_km_per_hr ≈ 63.00468 :=
   by sorry
   
end train_speed_is_63_00468_km_per_hr_l817_817644


namespace number_division_reduction_l817_817235

theorem number_division_reduction (x : ℕ) (h : x / 3 = x - 48) : x = 72 := 
sorry

end number_division_reduction_l817_817235


namespace find_angle_A_max_area_of_triangle_l817_817010

-- Definitions of variables and conditions for the triangle
variables {A B C: ℝ} {a b c: ℝ}

-- Condition for the given problem
axiom h1 : a * cos C + sqrt 3 * a * sin C - b - c = 0

-- Triangle ABC with side lengths opposite to corresponding angles
noncomputable def angle_A : ℝ := 60 * (π / 180)

noncomputable def area_max (a : ℝ) (b c : ℝ) (A : ℝ) : ℝ :=
  if h : a = 2 then (sqrt 3 / 4) * b * c else 0

-- Theorems to prove
theorem find_angle_A :
  a * cos C + sqrt 3 * a * sin C - b - c = 0 → A = angle_A :=
by
specially_cases_using_h1
sorry

theorem max_area_of_triangle :
  a = 2 → ∃ b c, let S := area_max a b c angle_A in S = sqrt 3 :=
by
specially_cases_using_a_2
sorry

end find_angle_A_max_area_of_triangle_l817_817010


namespace max_sin_cos_l817_817963

theorem max_sin_cos (θ : ℝ) (h1 : sin θ ^ 2 + cos θ ^ 2 = 1) : 
  sin θ + cos θ ≤ sqrt 2 := 
sorry

end max_sin_cos_l817_817963


namespace simplify_and_evaluate_expression_l817_817548

-- Define the expression and the condition
def expression (m : ℚ) : ℚ := (5 / (m - 2) - m - 2) * (2 * m - 4) / (3 - m)
def m_value : ℚ := -1 / 2

-- State the theorem
theorem simplify_and_evaluate_expression : expression m_value = 5 := 
  begin
    sorry
  end

end simplify_and_evaluate_expression_l817_817548


namespace expected_expenditure_l817_817150

-- Define the parameters and conditions
def b : ℝ := 0.8
def a : ℝ := 2
def e_condition (e : ℝ) : Prop := |e| < 0.5
def revenue : ℝ := 10

-- Define the expenditure function based on the conditions
def expenditure (x e : ℝ) : ℝ := b * x + a + e

-- The expected expenditure should not exceed 10.5
theorem expected_expenditure (e : ℝ) (h : e_condition e) : expenditure revenue e ≤ 10.5 :=
sorry

end expected_expenditure_l817_817150


namespace jen_total_birds_l817_817480

-- Define the number of chickens and ducks
variables (C D : ℕ)

-- Define the conditions
def ducks_condition (C D : ℕ) : Prop := D = 4 * C + 10
def num_ducks (D : ℕ) : Prop := D = 150

-- Define the total number of birds
def total_birds (C D : ℕ) : ℕ := C + D

-- Prove that the total number of birds is 185 given the conditions
theorem jen_total_birds (C D : ℕ) (h1 : ducks_condition C D) (h2 : num_ducks D) : total_birds C D = 185 :=
by
  sorry

end jen_total_birds_l817_817480


namespace inequality_proof_l817_817014

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^4 / (y-1)^2) + (y^4 / (z-1)^2) + (z^4 / (x-1)^2) ≥ 48 := 
by
  sorry -- The actual proof is omitted

end inequality_proof_l817_817014


namespace triangle_is_right_l817_817538

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 5)
def C : ℝ × ℝ := (3, 4)

-- Define the vectors corresponding to the sides of the triangle
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Calculate the lengths of the vectors
def length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def AB_length := length AB
def AC_length := length AC
def BC_length := length BC

-- Pythagorean theorem check function
def isRightTriangle : Prop :=
  (AC_length ^ 2 + BC_length ^ 2 = AB_length ^ 2) ∨
  (AB_length ^ 2 + BC_length ^ 2 = AC_length ^ 2) ∨
  (AB_length ^ 2 + AC_length ^ 2 = BC_length ^ 2)

-- The main theorem statement
theorem triangle_is_right : isRightTriangle :=
  by
    sorry

end triangle_is_right_l817_817538


namespace find_angle_DAB_l817_817449

open Real EuclideanGeometry

-- Define the given isosceles triangle ABC with CA = CB
variable (A B C D F : Point)
variable (hIso : distance A C = distance B C)
variable (hEqui : equilateral B C F)
variable (hPerp : perpendicular A D B C)

-- Define the angles in the triangle
variable (angle_DAB : ℝ)

-- Main theorem statement
theorem find_angle_DAB 
  (hIso : distance A C = distance B C) 
  (hEqui : equilateral B C F) 
  (hPerp : perpendicular A D B C) : 
  angle A D B = 60 :=
sorry

end find_angle_DAB_l817_817449


namespace coffee_order_cost_l817_817476

theorem coffee_order_cost :
  let drip_coffee_price := 2.25
  let double_shot_espresso_price := 3.50
  let latte_price := 4.00
  let vanilla_syrup_price := 0.50
  let cold_brew_price := 2.50
  let cappuccino_price := 3.50
  let total_cost := 
    (2 * drip_coffee_price) + 
    (1 * double_shot_espresso_price) + 
    (2 * latte_price) + 
    (1 * vanilla_syrup_price) + 
    (2 * cold_brew_price) + 
    (1 * cappuccino_price) 
  in
  total_cost = 25.00 :=
by sorry

end coffee_order_cost_l817_817476


namespace tan_theta_solution_l817_817393

theorem tan_theta_solution (θ : ℝ) (h1 : sin (2 * θ) = -1 / 3) (h2 : (π / 4) < θ ∧ θ < (3 * π / 4)) :
  tan θ = -3 - 2 * real.sqrt 2 := by
  sorry

end tan_theta_solution_l817_817393


namespace f0_plus_f4_eq_24_l817_817408

noncomputable def polynomial (m : ℝ) : ℝ → ℝ :=
  λ x, (x - 1) * (x - 2) * (x - 3) * (x - m)

theorem f0_plus_f4_eq_24 (m : ℝ) : 
  polynomial m 0 + polynomial m 4 = 24 :=
by
  sorry

end f0_plus_f4_eq_24_l817_817408


namespace find_number_l817_817251

theorem find_number :
  ∃ n : ℝ, 0.0025 * n = 0.04 ∧ n = 16 :=
by
  use 16
  split
  · rw [mul_comm, mul_div_cancel_left] ; norm_num ; exact ne_of_gt (by norm_num)
  · rfl

end find_number_l817_817251


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817908

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817908


namespace sum_a3_a4_a5_l817_817050

-- Define the sequence sums S_n
def S : ℕ → ℕ := λ n, n^2

-- Define a_n based on the S_n given in the problem statement
def a (n : ℕ) : ℕ := S n - S (n - 1)

-- State the proof problem
theorem sum_a3_a4_a5 : a 3 + a 4 + a 5 = 21 :=
by
  sorry

end sum_a3_a4_a5_l817_817050


namespace equivalent_functions_l817_817224

theorem equivalent_functions (x : ℝ) : (real.cbrt (x^3)) = (real.cbrt x) ^ 3 := 
  sorry

end equivalent_functions_l817_817224


namespace ratio_of_roots_diff_l817_817041

variable {a b : ℝ}

def f1 (x : ℝ) : ℝ := x^2 - x + 2 * a
def f2 (x : ℝ) : ℝ := x^2 + 2 * b * x + 3
def f3 (x : ℝ) : ℝ := 4 * x^2 + (2 * b - 3) * x + 6 * a + 3
def f4 (x : ℝ) : ℝ := 4 * x^2 + (6 * b - 1) * x + 9 + 2 * a

noncomputable def A := Real.sqrt (1 - 8 * a)
noncomputable def B := Real.sqrt (4 * b ^ 2 - 12)
noncomputable def C := (1 / 4) * Real.sqrt ((2 * b - 3) ^ 2 - 64 * (6 * a + 3))
noncomputable def D := (1 / 4) * Real.sqrt ((6 * b - 1) ^ 2 - 64 * (9 + 2 * a))

theorem ratio_of_roots_diff :
  |A| ≠ |B| →
  (C^2 - D^2) / (A^2 - B^2) = 1 / 2 := by
  sorry

end ratio_of_roots_diff_l817_817041


namespace line_through_points_l817_817071

theorem line_through_points (a b : ℝ) (h₁ : 1 = a * 3 + b) (h₂ : 13 = a * 7 + b) : a - b = 11 := 
  sorry

end line_through_points_l817_817071


namespace chess_team_arrangements_l817_817153

theorem chess_team_arrangements : 
  let girls := 2
  let boys := 3
  let arrangements := girls! * boys!
  arrangements = 12 :=
by
  -- Definitions
  let girls := 2
  let boys := 3
  -- Computation of factorial
  have hg : girls! = 2 := by norm_num
  have hb : boys! = 6 := by norm_num
  -- Result
  let arrangements := girls! * boys!
  have h : arrangements = 12 := by rw [hg, hb]; norm_num
  exact h

end chess_team_arrangements_l817_817153


namespace largest_triangle_area_l817_817195

variable (x : ℝ)
variable (AB BC AC : ℝ)
variable (A : ℝ)

axiom triangle_ABC_conditions :
  AB = 9 ∧
  BC = 3 * x ∧
  AC = 4 * x ∧
  (9 + 3 * x > 4 * x) ∧
  (9 + 4 * x > 3 * x) ∧
  (3 * x + 4 * x > 9) ∧
  ((9)^2 + (3 * x)^2 = (4 * x)^2) ∧
  (x > 0)

theorem largest_triangle_area :
  ∃ x A, A = (1 / 2) * 9 * (3 * 9 / Real.sqrt 7) ∧ A = 243 / (2 * Real.sqrt 7) :=
by
  sorry

end largest_triangle_area_l817_817195


namespace max_value_2x1_minus_x2_l817_817602

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x + π / 3)

noncomputable def g (x : ℝ) := 3 * Real.sin (2 * x + 2 * π / 3) + 1

theorem max_value_2x1_minus_x2 (x1 x2 : ℝ) (h1 : x1 ∈ Icc (-3 * π / 2) (3 * π / 2)) (h2 : x2 ∈ Icc (-3 * π / 2) (3 * π / 2)) (h3 : g x1 * g x2 = 16) :
  2 * x1 - x2 ≤ 35 * π / 12 :=
sorry

end max_value_2x1_minus_x2_l817_817602


namespace shifted_function_equiv_l817_817032

def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x)

def g (x : ℝ) : ℝ := f (x - Real.pi / 8)

theorem shifted_function_equiv :
  g x = 3 * Real.sin (2 * x - Real.pi / 4) :=
by
  -- skipping the proof here
  sorry

end shifted_function_equiv_l817_817032


namespace child_is_late_l817_817437

theorem child_is_late 
  (distance : ℕ)
  (rate1 rate2 : ℕ) 
  (early_arrival : ℕ)
  (time_late_at_rate1 : ℕ)
  (time_required_by_rate1 : ℕ)
  (time_required_by_rate2 : ℕ)
  (actual_time : ℕ)
  (T : ℕ) :
  distance = 630 ∧ 
  rate1 = 5 ∧ 
  rate2 = 7 ∧ 
  early_arrival = 30 ∧
  (time_required_by_rate1 = distance / rate1) ∧
  (time_required_by_rate2 = distance / rate2) ∧
  (actual_time + T = time_required_by_rate1) ∧
  (actual_time - early_arrival = time_required_by_rate2) →
  T = 6 := 
by
  intros
  sorry

end child_is_late_l817_817437


namespace solution_set_f_10x_pos_l817_817168

noncomputable def f (x : ℝ) : ℝ := - (1/2) * x^2 - x + (3/2)

lemma f_deriv : ∀ x, deriv f x = -x - 1 := by sorry

lemma f_point : f 0 = 3 / 2 := by sorry

lemma f_10x_pos_iff (x : ℝ) : f (10 ^ x) > 0 ↔ x < 0 :=
begin
  sorry
end

theorem solution_set_f_10x_pos : {x : ℝ | f (10^x) > 0} = set.Iio 0 :=
by simp [set.ext_iff, f_10x_pos_iff]

end solution_set_f_10x_pos_l817_817168


namespace gel_pen_price_relation_b_l817_817291

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l817_817291


namespace a_general_term_b_general_term_B_sum_l817_817789

noncomputable def a_seq (n : ℕ) : ℝ := 1 - (1 / 2) ^ n
noncomputable def S_n (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), a_seq k
noncomputable def b_seq (n : ℕ) : ℝ := if n = 0 then a_seq 1 else a_seq n - a_seq (n - 1)
noncomputable def B_n (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), b_seq k

theorem a_general_term (n : ℕ) : a_seq n = 1 - (1 / 2) ^ n :=
by sorry

theorem b_general_term (n : ℕ) : 
  ∀ n, b_seq n = (1 / 2) ^ n := 
by sorry

theorem B_sum (n : ℕ) : 
  B_n n = 1 - (1 / 2) ^ n := 
by sorry

end a_general_term_b_general_term_B_sum_l817_817789


namespace functional_expression_result_l817_817573

theorem functional_expression_result {f : ℝ → ℝ} (h : ∀ x y : ℝ, f (2 * x - 3 * y) - f (x + y) = -2 * x + 8 * y) :
  ∀ t : ℝ, (f (4 * t) - f t) / (f (3 * t) - f (2 * t)) = 3 :=
sorry

end functional_expression_result_l817_817573


namespace luncheon_cost_l817_817159

section LuncheonCosts

variables (s c p : ℝ)

/- Conditions -/
def eq1 : Prop := 2 * s + 5 * c + 2 * p = 6.25
def eq2 : Prop := 5 * s + 8 * c + 3 * p = 12.10

/- Goal -/
theorem luncheon_cost : eq1 s c p → eq2 s c p → s + c + p = 1.55 :=
by
  intro h1 h2
  sorry

end LuncheonCosts

end luncheon_cost_l817_817159


namespace shelves_needed_l817_817542

variable (total_books : Nat) (books_taken : Nat) (books_per_shelf : Nat)

theorem shelves_needed (h1 : total_books = 14) 
                       (h2 : books_taken = 2) 
                       (h3 : books_per_shelf = 3) : 
    (total_books - books_taken) / books_per_shelf = 4 := by
  sorry

end shelves_needed_l817_817542


namespace prob_0_lt_xi_lt_1_l817_817521

noncomputable def normal_distribution (μ δ : ℝ) : Type := sorry -- Assume we have a definition here

variables (μ δ : ℝ) (ξ : normal_distribution μ δ) (p : ℝ)

axiom prob_1 : (P(ξ < 1) = 1/2)
axiom prob_2 : (P(ξ > 2) = p)

theorem prob_0_lt_xi_lt_1 : P(0 < ξ < 1) = 1/2 - p :=
by
  -- The proof would go here
  sorry

end prob_0_lt_xi_lt_1_l817_817521


namespace max_value_of_f_l817_817379

def f (x : ℝ) := min (min (3 * x - 1) (-x + 4)) (2 * x + 5)

theorem max_value_of_f : ∃ (M : ℝ), (∀ x : ℝ, f x ≤ M) ∧ (∃ x₀ : ℝ, f x₀ = M) :=
  sorry

end max_value_of_f_l817_817379


namespace count_two_digit_integers_with_remainder_3_l817_817850

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817850


namespace ellipse_parabola_existence_of_lambda_l817_817751

def ellipse_equation (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2) / (a^2) + y^2 = 1

def parabola_equation (p : ℝ) : Prop :=
  ∀ (x y : ℝ), y^2 = 2 * p * x

def distance_AB_CD_constant (a k λ : ℝ) : Prop :=
  ∀ (x1 x2 x3 x4 : ℝ),
  let AB := (2 * sqrt(5) * (1 + k^2)) / (1 + 5 * k^2) in
  let CD := (8 * (k^2 + 1)) / k^2 + 4 in
  2 / AB + λ / CD = constant

theorem ellipse_parabola_existence_of_lambda :
  let (a, e) := (sqrt 5, (2 * sqrt 5) / 5)
  let c := 2
  let b := sqrt (a^2 - c^2)
  let p := 4
  ∃ λ : ℝ,
    ellipse_equation a b ∧
    parabola_equation p ∧
    distance_AB_CD_constant a λ :=
by
  sorry

end ellipse_parabola_existence_of_lambda_l817_817751


namespace condition_necessary_but_not_sufficient_l817_817441

def ceiling (x : ℝ) : ℤ := ⌈x⌉

theorem condition_necessary_but_not_sufficient (x y : ℝ) :
  (|x - y| < 1) → (ceiling x ≠ ceiling y) ∨ (ceiling x = ceiling y ∧ |x - y| < 1) :=
by
  sorry

end condition_necessary_but_not_sufficient_l817_817441


namespace sum_binom_values_l817_817218

-- Define the main problem theorem statement
theorem sum_binom_values : 
  (finset.sum (finset.filter (λ n : ℕ, nat.choose 23 n + nat.choose 23 12 = nat.choose 24 13) (finset.range 24)) id) = 13 :=
sorry

end sum_binom_values_l817_817218


namespace trapezoid_area_l817_817563

theorem trapezoid_area (M N P Q O K : Type) 
  (h_tangency : CircleTangentAt M O N)
  (h_intersect : CircleIntersectsAt PQ O K)
  (h_pq_eq : PQ = 4 * sqrt(3) * (KQ))
  (h_angle_NQM : ∠ N Q M = 60°)
  (R : ℝ)
  (h_MQ : MQ = 2 * R)
  (h_circle : isCircle O R) :
  area_trapezoid MQ NP = 2 * R^2 * (5 * sqrt(3) - 6) :=
sorry

end trapezoid_area_l817_817563


namespace cannot_obtain_fraction_3_5_l817_817101

theorem cannot_obtain_fraction_3_5 (n k : ℕ) :
  ¬ ∃ (a b : ℕ), (a = 5 + k ∧ b = 8 + k ∨ (∃ m : ℕ, a = m * 5 ∧ b = m * 8)) ∧ (a = 3 ∧ b = 5) :=
by
  sorry

end cannot_obtain_fraction_3_5_l817_817101


namespace union_A_B_l817_817503

noncomputable def A := {x : ℝ | Real.log x ≤ 0}
noncomputable def B := {x : ℝ | x^2 - 1 < 0}
def A_union_B := {x : ℝ | (Real.log x ≤ 0) ∨ (x^2 - 1 < 0)}

theorem union_A_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by
  -- proof to be added
  sorry

end union_A_B_l817_817503


namespace optimal_play_win_l817_817594

theorem optimal_play_win (N K : ℕ) :
  N + K % 2 = 1 ↔ 
  ∀ n k : ℕ,
    (n = N ∧ k = K → 
    (∃ (p : ℕ × ℕ), n + k = 2 * p.1 + p.2 ∧ p.2 = 1 ∧
    ∀ (move : ℕ × ℕ → ℕ × ℕ),
      move (n, k) ≠ (n, k) → (∃ (q : ℕ × ℕ), move (n, k) = (q.1, q.2) ∧ 
      q.1 + q.2 ≠ 2 * p.1 + p.2) ∨ move (n, k) = (0, 0)))) :=
begin
  sorry
end

end optimal_play_win_l817_817594


namespace jen_total_birds_l817_817483

-- Define the initial conditions
def total_birds (c : ℕ) : ℕ :=
  let d := 10 + 4 * c in
  if d = 150 then c + d else 0

theorem jen_total_birds : total_birds 35 = 185 :=
  sorry

end jen_total_birds_l817_817483


namespace infinitely_many_solutions_x2_plus_y2_eq_x3_l817_817580

theorem infinitely_many_solutions_x2_plus_y2_eq_x3 :
  ∃ an infinite number of positive integer pairs (x y : ℕ), (x^2 + y^2 = x^3) :=
sorry

end infinitely_many_solutions_x2_plus_y2_eq_x3_l817_817580


namespace Janette_camping_days_l817_817479

theorem Janette_camping_days (initial_beef_jerky : ℕ) (daily_consumption : ℕ) (brother_share : ℕ) (remaining_after_brother : ℕ) :
  initial_beef_jerky = 40 →
  daily_consumption = 4 →
  remaining_after_brother = 10 →
  brother_share = 2 * remaining_after_brother →
  initial_beef_jerky - brother_share = 20 →
  (initial_beef_jerky - brother_share) / daily_consumption = 5 :=
by
  intros,
  sorry

end Janette_camping_days_l817_817479


namespace even_digit_count_l817_817432

theorem even_digit_count:
  let valid_digits := {0, 2, 4}
  let count := (λ p q r, p ∈ {2, 4} ∧ q ∈ valid_digits ∧ r ∈ valid_digits)
  (finset.univ.filter (λ x: ℕ, (∃ (p q r: ℕ), x = 100 * p + 10 * q + r ∧ count p q r))).card = 18 :=
by
  sorry

end even_digit_count_l817_817432


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817917

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817917


namespace mr_rainwater_chickens_l817_817530

theorem mr_rainwater_chickens :
  ∃ (Ch : ℕ), (∀ (C G : ℕ), C = 9 ∧ G = 4 * C ∧ G = 2 * Ch → Ch = 18) :=
by
  sorry

end mr_rainwater_chickens_l817_817530


namespace num_rem_three_by_seven_l817_817858

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817858


namespace distance_from_axis_gt_l817_817006

theorem distance_from_axis_gt 
  (a b x1 x2 y1 y2 : ℝ) (h₁ : a > 0) 
  (h₂ : y1 = a * x1^2 - 2 * a * x1 + b) 
  (h₃ : y2 = a * x2^2 - 2 * a * x2 + b) 
  (h₄ : y1 > y2) : 
  |x1 - 1| > |x2 - 1| := 
sorry

end distance_from_axis_gt_l817_817006


namespace lemonade_syrup_parts_l817_817271

theorem lemonade_syrup_parts (L : ℝ) :
  (L = 2 / 0.75) →
  (L = 2.6666666666666665) :=
by
  sorry

end lemonade_syrup_parts_l817_817271


namespace find_y_l817_817156

theorem find_y (y : ℤ) (h : (15 + 24 + y) / 3 = 23) : y = 30 :=
by
  sorry

end find_y_l817_817156


namespace exists_infinite_set_no_three_collinear_rational_distances_l817_817472

theorem exists_infinite_set_no_three_collinear_rational_distances :
  ∃ (S : Set (ℝ × ℝ)), 
  Set.Infinite S ∧ 
  (∀ (P Q : ℝ × ℝ), P ∈ S → Q ∈ S → P ≠ Q → ∃ d : ℚ, dist P Q = ↑d) ∧ 
  (∀ (P Q R : ℝ × ℝ), P ∈ S → Q ∈ S → R ∈ S → P ≠ Q → P ≠ R → Q ≠ R → 
   ¬ collinear ℝ {P, Q, R}) :=
sorry

end exists_infinite_set_no_three_collinear_rational_distances_l817_817472


namespace probability_of_less_than_20_l817_817237

variable (total_people : ℕ) (people_over_30 : ℕ)
variable (people_under_20 : ℕ) (probability_under_20 : ℝ)

noncomputable def group_size := total_people = 150
noncomputable def over_30 := people_over_30 = 90
noncomputable def under_20 := people_under_20 = total_people - people_over_30

theorem probability_of_less_than_20
  (total_people_eq : total_people = 150)
  (people_over_30_eq : people_over_30 = 90)
  (people_under_20_eq : people_under_20 = 60)
  (under_20_eq : 60 = total_people - people_over_30) :
  probability_under_20 = people_under_20 / total_people := by
  sorry

end probability_of_less_than_20_l817_817237


namespace pradeep_passing_percentage_l817_817135

-- Define the constants based on the conditions
def totalMarks : ℕ := 550
def marksObtained : ℕ := 200
def marksFailedBy : ℕ := 20

-- Calculate the passing marks
def passingMarks : ℕ := marksObtained + marksFailedBy

-- Define the percentage calculation as a noncomputable function
noncomputable def requiredPercentageToPass : ℚ := (passingMarks / totalMarks) * 100

-- The theorem to prove
theorem pradeep_passing_percentage :
  requiredPercentageToPass = 40 := 
sorry

end pradeep_passing_percentage_l817_817135


namespace evaluate_fraction_l817_817711

theorem evaluate_fraction :
  (2 ^ (-3) * 3 ^ 1) / (2 ^ (-4)) = 3 / 128 :=
by
  sorry

end evaluate_fraction_l817_817711


namespace eight_rooks_arrangement_l817_817649

open Finset

theorem eight_rooks_arrangement : ∃ (arrangements : Finset (Fin 64)), arrangements.card = 3456 ∧ 
  (∀ arrangement ∈ arrangements, ∀ (i j : Fin 8), 
    i ≠ j → (arrangement (8*i + i) ≠ arrangement (8*j + j))) ∧ 
  (∀ arrangement ∈ arrangements, ∃ (squares : Finset (Fin 8)), squares = {0, 1, 2, 3, 4, 5, 6, 7} ∧ 
    ∀ (i : Fin 8), arrangement (8*i + i) ∈ squares) := by
  sorry

end eight_rooks_arrangement_l817_817649


namespace range_of_slope_l817_817176

def slope (θ : ℝ) : ℝ := Real.tan θ

def is_angle_in_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < π

theorem range_of_slope : { m : ℝ | ∃ θ, is_angle_in_range θ ∧ m = slope θ } = set.Ico 0 π :=
by
  sorry

end range_of_slope_l817_817176


namespace count_two_digit_integers_with_remainder_3_l817_817842

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817842


namespace weight_ordering_l817_817286

theorem weight_ordering :
  let g := 1
  let kg := 1000 * g
  let t := 1000 * kg
  let w1 := 908 * g
  let w2 := (9 * kg) + 80 * g
  let w3 := 0.09 * t
  let w4 := 900 * kg
  w1 < w2 ∧ w2 < w3 ∧ w3 < w4 :=
by
  collapse
  algebra
  sorry

end weight_ordering_l817_817286


namespace find_a_b_find_range_of_c_l817_817030

variable {a b c x : ℝ}

-- Define the function f
noncomputable def f (x : ℝ) := (a * x^2 + b * x) / real.exp x

-- Define the tangent line condition
def tangent_line (x : ℝ) (y : ℝ) := x = real.exp 1 * y

-- Define the min function g
noncomputable def g (x : ℝ) := min (f x) (x - 1 / x)

-- Define the function h
noncomputable def h (x : ℝ) := g x - c * x^2

-- Statement to prove a and b
theorem find_a_b (h_ext : f' 0 = 0) (tangent_cond : ∀ P : ℝ, tangent_line P (f P)) : 
  a = 1 ∧ b = 0 := sorry

-- Statement to prove range of c
theorem find_range_of_c (increasing_h : ∀ x > 0, 0 < h' x) : 
  c ≤ -1 / (2 * real.exp 3) := sorry

end find_a_b_find_range_of_c_l817_817030


namespace regular_hexagon_intersection_planes_l817_817632

-- Defining the regular dodecahedron and the properties required.
def is_regular_dodecahedron (d : Type) : Prop :=
  ∃ (v : Finset (d → ℝ)) (e : Finset (Finset (d → ℝ))),
    ∀ v1 v2, v1 ∈ v → v2 ∈ v → v1 ≠ v2 → (∃ p1 p2 p3 p4 p5 p6, p1 ∈ e ∧
      p2 ∈ e ∧ p3 ∈ e ∧ p4 ∈ e ∧ p5 ∈ e ∧ p6 ∈ e ∧
      -- ... additional details to represent edges and faces ...
      true)

-- Counting the intersecting planes that result in regular hexagon slices.
theorem regular_hexagon_intersection_planes (d : Type) [h : is_regular_dodecahedron d] :
  ∃ ways : ℕ, ways = 30 :=
sorry

end regular_hexagon_intersection_planes_l817_817632


namespace part1_part2_l817_817191

variable (a : ℕ → ℕ)

-- Part 1: Prove that the sum of any consecutive 43 items on the circle is also a multiple of 43.
theorem part1 (h1 : ∀ i < 2021, ∑ j in finset.range 43, a (i + j) % 43 = 0) :
        ∀ i < 2021, ∑ j in finset.range 43, a ((i + j) % 2021) % 43 = 0 := 
begin
    sorry
end

-- Part 2: Determine the number of sequences that meet the conditions.
theorem part2 : ∃ seq_count : ℕ, seq_count = 43.factorial * (47.factorial ^ 43) :=
begin
    use 43.factorial * (47.factorial ^ 43),
    sorry
end

end part1_part2_l817_817191


namespace area_of_S_l817_817784

def fractional_part (t : ℝ) : ℝ := t - floor t

def set_S (T : ℝ) : set (ℝ × ℝ) :=
  { p | let (x, y) := p in (x - T)^2 + y^2 ≤ T^2 }

theorem area_of_S (t : ℝ) (T := fractional_part t) :
  0 ≤ T ∧ T < 1 → 0 < pi * T^2 ∧ pi * T^2 < pi :=
by 
  intros h
  sorry

end area_of_S_l817_817784


namespace coffeeOrderTotalIs25_l817_817478

noncomputable def totalCost : ℝ :=
  let costDripCoffee := 2 * 2.25
  let costLattes := 2 * 4.00
  let costVanilla := 0.50
  let costColdBrew := 2 * 2.50
  let costDoubleEspresso := 3.50
  let costCappuccino := 3.50
  costDripCoffee + costLattes + costVanilla + costColdBrew + costDoubleEspresso + costCappuccino

theorem coffeeOrderTotalIs25 : totalCost = 25 := by
  intro
  unfold totalCost
  sorry

end coffeeOrderTotalIs25_l817_817478


namespace period_tan_frac_eq_l817_817212

theorem period_tan_frac_eq (x : ℝ) (π : ℝ) (h : ∀ θ : ℝ, tan (θ + π) = tan θ) :
  ∃ T : ℝ, T = 4 * π / 3 ∧ ∀ x : ℝ, tan (3 / 4 * (x + T)) = tan (3 / 4 * x) :=
by
  let T := 4 * π / 3
  use T
  split
  · exact rfl
  · intro x
    have ht : tan ((3 / 4 * x) + π) = tan (3 / 4 * x), from h (3 / 4 * x)
    calc 
      tan (3 / 4 * (x + T))
          = tan ((3 / 4) * x + (3 / 4) * T) : by rw mul_add
      ... = tan ((3 / 4) * x + (3 / 4) * (4 * π / 3)) : by rw T 
      ... = tan ((3 / 4) * x + π) : by ring
      ... = tan (3 / 4 * x) : by rw ht

end period_tan_frac_eq_l817_817212


namespace first_chinese_supercomputer_is_milkyway_l817_817958

-- Define the names of the computers
inductive ComputerName
| Universe
| Taihu
| MilkyWay
| Dawn

-- Define a structure to hold the properties of the computer
structure Computer :=
  (name : ComputerName)
  (introduction_year : Nat)
  (calculations_per_second : Nat)

-- Define the properties of the specific computer in the problem
def first_chinese_supercomputer := 
  Computer.mk ComputerName.MilkyWay 1983 100000000

-- The theorem to be proven
theorem first_chinese_supercomputer_is_milkyway :
  first_chinese_supercomputer.name = ComputerName.MilkyWay :=
by
  -- Provide the conditions that lead to the conclusion (proof steps will be added here)
  sorry

end first_chinese_supercomputer_is_milkyway_l817_817958


namespace general_term_formula_limit_of_S_l817_817595

noncomputable def S (n : ℕ) : ℝ :=
  (8 / 5) - (3 / 5) * ((4 / 9) ^ n)

theorem general_term_formula (n : ℕ) : 
  S n = (8 / 5) - (3 / 5) * ((4 / 9) ^ n) :=
by
  sorry

theorem limit_of_S : 
  tendsto (λ n : ℕ, S n) at_top (𝓝 (8 / 5)) :=
by
  sorry

end general_term_formula_limit_of_S_l817_817595


namespace sqrt_expression_meaningful_l817_817068

theorem sqrt_expression_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) :=
by
  -- Proof will be skipped
  sorry

end sqrt_expression_meaningful_l817_817068


namespace justin_current_age_l817_817682

theorem justin_current_age
  (angelina_older : ∀ (j a : ℕ), a = j + 4)
  (angelina_future_age : ∀ (a : ℕ), a + 5 = 40) :
  ∃ (justin_current_age : ℕ), justin_current_age = 31 := 
by
  sorry

end justin_current_age_l817_817682


namespace pen_price_ratio_l817_817318

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l817_817318


namespace price_ratio_l817_817341

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l817_817341


namespace probability_both_truth_l817_817057

variable (P_A : ℝ) (P_B : ℝ)

theorem probability_both_truth (hA : P_A = 0.55) (hB : P_B = 0.60) :
  P_A * P_B = 0.33 :=
by
  sorry

end probability_both_truth_l817_817057


namespace number_of_unique_letters_l817_817243

-- Define the set of unique letters from each month name
def january : Set Char := {'Я', 'н', 'в', 'а', 'р', 'ь'}
def february : Set Char := {'Ф', 'е', 'в', 'р', 'а', 'л', 'ь'}
def march : Set Char := {'М', 'а', 'р', 'т'}
def april : Set Char := {'А', 'п', 'р', 'е', 'л', 'ь'}
def may : Set Char := {'М', 'а', 'й'}
def june : Set Char := {'И', 'ю', 'н', 'ь'}
def july : Set Char := {'И', 'ю', 'л', 'ь'}
def august : Set Char := {'А', 'в', 'г', 'у', 'с', 'т'}
def september : Set Char := {'С', 'е', 'н', 'т', 'я', 'б', 'р', 'ь'}
def october : Set Char := {'О', 'к', 'т', 'я', 'б', 'р', 'ь'}
def november : Set Char := {'Н', 'о', 'я', 'б', 'р', 'ь'}
def december : Set Char := {'Д', 'е', 'к', 'а', 'б', 'р', 'ь'}

-- Define the set of all unique letters from all month names
def uniqueLetters : Set Char :=
  january ∪ february ∪ march ∪ april ∪ may ∪ june ∪ july ∪ august ∪
  september ∪ october ∪ november ∪ december

-- Prove the total number of unique letters is 22
theorem number_of_unique_letters : uniqueLetters.toFinset.card = 22 :=
  by {
    sorry -- Proof goes here.
  }

end number_of_unique_letters_l817_817243


namespace gel_pen_price_relation_b_l817_817293

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l817_817293


namespace jen_total_birds_l817_817481

-- Define the number of chickens and ducks
variables (C D : ℕ)

-- Define the conditions
def ducks_condition (C D : ℕ) : Prop := D = 4 * C + 10
def num_ducks (D : ℕ) : Prop := D = 150

-- Define the total number of birds
def total_birds (C D : ℕ) : ℕ := C + D

-- Prove that the total number of birds is 185 given the conditions
theorem jen_total_birds (C D : ℕ) (h1 : ducks_condition C D) (h2 : num_ducks D) : total_birds C D = 185 :=
by
  sorry

end jen_total_birds_l817_817481


namespace two_digit_integers_remainder_3_count_l817_817874

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817874


namespace count_two_digit_integers_with_remainder_3_l817_817843

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817843


namespace count_interesting_numbers_l817_817837

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817837


namespace star_operation_possible_l817_817980

noncomputable def star_operation_exists : Prop := 
  ∃ (star : ℤ → ℤ → ℤ), 
  (∀ (a b c : ℤ), star (star a b) c = star a (star b c)) ∧ 
  (∀ (x y : ℤ), star (star x x) y = y ∧ star y (star x x) = y)

theorem star_operation_possible : star_operation_exists :=
sorry

end star_operation_possible_l817_817980


namespace solve_for_x_l817_817550

-- Define the function representing the original equation
def original_equation (x : ℝ) : Prop :=
  log 3 ((4 * x + 12) / (6 * x - 4)) + log 3 ((6 * x - 4) / (2 * x - 3)) = 2

-- The principal statement to be proved in Lean 4
theorem solve_for_x : original_equation (39 / 14) :=
by
  -- Here we'd provide the proof of the statement
  sorry

end solve_for_x_l817_817550


namespace range_of_m_l817_817729

theorem range_of_m (m x : ℝ) : 
  (2 / (x - 3) + (x + m) / (3 - x) = 2) 
  ∧ (x ≥ 0) →
  (m ≤ 8 ∧ m ≠ -1) :=
by 
  sorry

end range_of_m_l817_817729


namespace part1_part2_l817_817428

open Real

-- Define the vectors and the conditions given in the problem
variables {α β : ℝ}
variables a := (cos α, sin α)
variables b := (cos β, sin β)
variables c := (0, 1)

-- Part 1: Proving orthogonality given the distance condition
theorem part1 (h1 : 0 < β) (h2 : β < α) (h3 : α < π) (h4 : (cos α - cos β) ^ 2 + (sin α - sin β) ^ 2 = 2) : 
  (cos α * cos β + sin α * sin β = 0) := 
by 
  sorry

-- Part 2: Finding α and β given the sum of vectors
theorem part2 (h1 : 0 < β) (h2 : β < α) (h3 : α < π) (h4 : (cos α + cos β, sin α + sin β) = c) :
  α = (5/6) * π ∧ β = (1/6) * π :=
by
  sorry

end part1_part2_l817_817428


namespace arithmetic_sequence_d_arithmetic_sequence_a_n_arithmetic_sequence_S_n_sum_sequence_T_n_l817_817002

noncomputable def d : ℕ := 2

-- Definition of the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := 2 * n - 1

-- General formula for the sum of the first n terms S_n of arithmetic sequence {a_n}
def S (n : ℕ) : ℕ := n * n

-- Definition of the sequence {1 / (a_n * a_{n+1})}
def b (n : ℕ) : ℝ :=
  1 / (a n * a (n + 1))

-- General formula for the sum of the first n terms T_n of the sequence {1 / (a_n * a_{n+1})}
def T (n : ℕ) : ℝ :=
  n / (2 * n + 1)

theorem arithmetic_sequence_d : d = 2 := by
  sorry

theorem arithmetic_sequence_a_n (n : ℕ) : a n = 2 * n - 1 := by
  sorry

theorem arithmetic_sequence_S_n (n : ℕ) : S n = n * n := by
  sorry

theorem sum_sequence_T_n (n : ℕ) : T n = n / (2 * n + 1) := by
  sorry

end arithmetic_sequence_d_arithmetic_sequence_a_n_arithmetic_sequence_S_n_sum_sequence_T_n_l817_817002


namespace two_digit_integers_remainder_3_count_l817_817880

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817880


namespace gel_pen_ratio_l817_817307

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l817_817307


namespace matrix_and_eigenvalues_l817_817424

-- Define the conditions
def T (p : ℝ × ℝ) : ℝ × ℝ :=
  if p = (1, 1/2) then (9/4, -2)
  else if p = (0, 1) then (-3/2, 4)
  else (0, 0)  -- Default case, not of interest

-- Define matrix M as per conditions and solve the system
def M : matrix (fin 2) (fin 2) ℝ :=
  ![![3, -3/2], [-7/2, 4]]

-- Define the characteristic polynomial of M
@[simp]
def characteristic_polynomial_M : polynomial ℝ :=
  matrix.det ![
  ![polynomial.C 1 - polynomial.X, -3/2],
  [-7/2, polynomial.C 1 - 4]]

-- State the problem
theorem matrix_and_eigenvalues :
  (∀ p : ℝ × ℝ, p = (1, 1/2) → M.mul_vec ![p.1, p.2] = ![(9/4), -2]) ∧
  (∀ p : ℝ × ℝ, p = (0, 1) → M.mul_vec ![p.1, p.2] = ![(-3/2), 4]) ∧
  (M = ![![3, -3/2], [-7/2, 4]]) ∧
  (characteristic_polynomial_M = polynomial.X^2 - 7 * polynomial.X + 12) ∧
  ∃ λ : ℝ, (characteristic_polynomial_M.eval λ = 0) :=
by sorry

end matrix_and_eigenvalues_l817_817424


namespace label_triangles_l817_817607

theorem label_triangles (n : ℕ) (P : Fin (n+1) → Type) (triangles : Fin (n-1) → Type) (labels : Fin (n-1)) :
  ∀ (i : Fin (n-1)), ∃ T : Fin (n-1), i = T ∧ (P i).vertex_in T :=
by
  sorry

end label_triangles_l817_817607


namespace successfully_served_pizzas_l817_817666

-- Defining the conditions
def total_pizzas_served : ℕ := 9
def pizzas_returned : ℕ := 6

-- Stating the theorem
theorem successfully_served_pizzas :
  total_pizzas_served - pizzas_returned = 3 :=
by
  -- Since this is only the statement, the proof is omitted using sorry
  sorry

end successfully_served_pizzas_l817_817666


namespace frog_ends_on_boundary_l817_817259

-- Define the point (2,3) and the boundaries of the square
def start_point := (2, 3) : ℕ × ℕ
def boundaries := [(1, 1), (1, 6), (6, 6), (6, 1)] : list (ℕ × ℕ)

-- Define the probability function Q
noncomputable def Q (x y : ℕ) : ℝ := 
sorry -- Implementation of the recursive probability function goes here.

-- Define the theorem to prove the probability
theorem frog_ends_on_boundary : 
  (Q start_point.1 start_point.2) = 11 / 16 := 
sorry

end frog_ends_on_boundary_l817_817259


namespace second_digit_base5_l817_817659

theorem second_digit_base5 (a b c d : ℕ) (N : ℕ) (h₁ : N = 125 * a + 25 * b + 5 * c + d)
  (h₂ : N = 343 * d + 49 * c + 7 * b + a) (ha : a < 5) (hb : b < 5) (hc : c < 5) (hd : d < 5) :
  b = 1 :=
begin
  sorry
end

end second_digit_base5_l817_817659


namespace root_exists_between_l817_817361

noncomputable def f (x : ℝ) : ℝ := (6 / x) - log x / log 2

theorem root_exists_between {x : ℝ} (h1: 3 < x) (h2: x < 4) : ∃ x ∈ set.Ioo (3:ℝ) (4:ℝ), f x = 0 :=
by
  have h1 : f (3:ℝ) > 0 := sorry
  have h2 : f (4:ℝ) < 0 := sorry
  have h3 : continuous_on f (set.Ioo (3:ℝ) (4:ℝ)) := sorry
  exact intermediate_value_theorem _ h3 h1 h2

end root_exists_between_l817_817361


namespace count_two_digit_remainders_l817_817801

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817801


namespace cannot_obtain_fraction_3_5_l817_817100

theorem cannot_obtain_fraction_3_5 (n k : ℕ) :
  ¬ ∃ (a b : ℕ), (a = 5 + k ∧ b = 8 + k ∨ (∃ m : ℕ, a = m * 5 ∧ b = m * 8)) ∧ (a = 3 ∧ b = 5) :=
by
  sorry

end cannot_obtain_fraction_3_5_l817_817100


namespace intersection_distance_l817_817970

noncomputable def curve_c1_param_eq (alpha : ℝ) : ℝ × ℝ :=
(2 * real.sqrt 5 * real.cos alpha, 2 * real.sin alpha)

noncomputable def curve_c2_polar_eq (rho theta : ℝ) : ℝ :=
rho ^ 2 + 4 * rho * real.cos theta - 2 * rho * real.sin theta + 4

theorem intersection_distance :
  ∃ (alpha : ℝ) (rho theta : ℝ) (A B : ℝ × ℝ), curve_c1_param_eq alpha = A ∧ curve_c2_polar_eq rho theta = 0 ∧
  ∃ (l : ℝ → ℝ × ℝ), (l t = A ∨ l t = B) ∧ 
  |(A.1 - B.1, A.2 - B.2)| = real.sqrt 2 :=
sorry

end intersection_distance_l817_817970


namespace sum_equals_120_l817_817365

def rectangular_parallelepiped := (3, 4, 5)

def face_dimensions : List (ℕ × ℕ) := [(4, 5), (3, 5), (3, 4)]

def number_assignment (d : ℕ × ℕ) : ℕ :=
  if d = (4, 5) then 9
  else if d = (3, 5) then 8
  else if d = (3, 4) then 5
  else 0

def sum_checkerboard_ring_one_width (rect_dims : ℕ × ℕ × ℕ) (number_assignment : ℕ × ℕ → ℕ) : ℕ :=
  let (x, y, z) := rect_dims
  let l1 := number_assignment (4, 5) * 2 * (4 * 5)
  let l2 := number_assignment (3, 5) * 2 * (3 * 5)
  let l3 := number_assignment (3, 4) * 2 * (3 * 4) 
  l1 + l2 + l3

theorem sum_equals_120 : ∀ rect_dims number_assignment,
  rect_dims = rectangular_parallelepiped → sum_checkerboard_ring_one_width rect_dims number_assignment = 720 := sorry

end sum_equals_120_l817_817365


namespace second_bag_roger_is_3_l817_817143

def total_candy_sandra := 2 * 6
def total_candy_roger := total_candy_sandra + 2
def first_bag_roger := 11
def second_bag_roger := total_candy_roger - first_bag_roger

theorem second_bag_roger_is_3 : second_bag_roger = 3 :=
by
  sorry

end second_bag_roger_is_3_l817_817143


namespace river_volume_per_minute_l817_817623

theorem river_volume_per_minute (depth width : ℝ) (flow_rate_kmph : ℝ)
  (h_depth : depth = 4) (h_width : width = 65) (h_flow_rate_kmph : flow_rate_kmph = 6) :
  let flow_rate_m_per_min := (flow_rate_kmph * 1000) / 60 in -- converting kmph to m/min
  let cross_sectional_area := depth * width in
  cross_sectional_area * flow_rate_m_per_min = 26000 :=
by
  sorry

end river_volume_per_minute_l817_817623


namespace smallest_integer_solution_l817_817049

theorem smallest_integer_solution (y : ℤ) (h : 7 - 3 * y < 25) : y ≥ -5 :=
by {
  sorry
}

end smallest_integer_solution_l817_817049


namespace gel_pen_ratio_l817_817306

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l817_817306


namespace train_pass_time_correct_l817_817622

-- Define the conditions
def jogger_speed_kmph : ℝ := 9
def jogger_distance_ahead_m : ℕ := 240
def train_length_m : ℕ := 130
def train_speed_kmph : ℝ := 45

-- Convert speeds from km/hr to m/s
def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / (60 * 60)
def train_speed_mps : ℝ := train_speed_kmph * 1000 / (60 * 60)

-- Define the relative speed of the train with respect to the jogger
def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps

-- Define the total distance to be covered by the train to pass the jogger
def total_distance_m : ℕ := jogger_distance_ahead_m + train_length_m

-- Define the time taken for the train to pass the jogger
def time_to_pass_seconds : ℝ := total_distance_m / relative_speed_mps

-- Theorem: Time taken to pass the jogger is 37 seconds
theorem train_pass_time_correct : time_to_pass_seconds = 37 := by
  sorry

end train_pass_time_correct_l817_817622


namespace justin_current_age_l817_817680

theorem justin_current_age (angelina_future_age : ℕ) (years_until_future : ℕ) (age_difference : ℕ)
  (h_future_age : angelina_future_age = 40) (h_years_until_future : years_until_future = 5)
  (h_age_difference : age_difference = 4) : 
  (angelina_future_age - years_until_future) - age_difference = 31 :=
by
  -- This is where the proof would go.
  sorry

end justin_current_age_l817_817680


namespace percentage_problem_l817_817444

-- Define the main proposition
theorem percentage_problem (n : ℕ) (a : ℕ) (b : ℕ) (P : ℕ) :
  n = 6000 →
  a = (50 * n) / 100 →
  b = (30 * a) / 100 →
  (P * b) / 100 = 90 →
  P = 10 :=
by
  intros h_n h_a h_b h_Pb
  sorry

end percentage_problem_l817_817444


namespace residue_inequality_l817_817732

theorem residue_inequality {n : ℕ} (hn : n ≥ 3) (p : Fin n → ℕ)
  (h_prime : ∀ i, Nat.Prime (p i))
  (h_distinct : ∀ i j, i ≠ j → p i ≠ p j)
  (r : ℕ)
  (h_residue : ∀ k, ∏ i in (Finset.univ.filter (λ i, i ≠ k)), p i % p k = r) :
  r ≤ n - 2 :=
sorry

end residue_inequality_l817_817732


namespace anna_truck_meet_once_l817_817281

noncomputable def anna_speed : ℝ := 5
noncomputable def truck_speed : ℝ := 15
noncomputable def pail_distance : ℝ := 300
noncomputable def truck_stop_time : ℝ := 40
noncomputable def initial_anna_position : ℝ := 0
noncomputable def initial_truck_position : ℝ := 300

noncomputable def anna_position (t : ℝ) : ℝ := initial_anna_position + anna_speed * t

noncomputable def truck_position (t : ℝ) : ℝ := 
  let cycle_time := 60  -- Truck travels 300 feet in 20 seconds + 40 seconds stop
  let cycles := t / cycle_time
  let remainder := t % cycle_time
  initial_truck_position + truck_speed * min remainder 20 + (cycles.floor : ℝ) * 300

noncomputable def distance_function (t : ℝ) : ℝ := truck_position t - anna_position t

theorem anna_truck_meet_once : ∃ t > 0, distance_function t = 0 ∧ ∀ t' < t, distance_function t' ≠ 0 := 
sorry

end anna_truck_meet_once_l817_817281


namespace perimeter_of_square_field_l817_817186

-- Given conditions
def num_posts : ℕ := 36
def post_width_inch : ℝ := 6
def gap_length_feet : ℝ := 8

-- Derived conditions
def posts_per_side : ℕ := num_posts / 4
def gaps_per_side : ℕ := posts_per_side - 1
def total_gap_length_per_side : ℝ := gaps_per_side * gap_length_feet
def post_width_feet : ℝ := post_width_inch / 12
def total_post_width_per_side : ℝ := posts_per_side * post_width_feet
def side_length : ℝ := total_gap_length_per_side + total_post_width_per_side

-- Goal: The perimeter of the square field
theorem perimeter_of_square_field : 4 * side_length = 242 := by
  sorry

end perimeter_of_square_field_l817_817186


namespace two_digit_integers_remainder_3_count_l817_817883

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817883


namespace notebook_statements_correct_l817_817048

noncomputable def statement_true (n : ℕ) : Prop :=
  n = 39

theorem notebook_statements_correct :
  ∃ n : ℕ, n ≥ 1 ∧ n ≤ 40 ∧ statement_true n :=
by
  use 39
  dsimp [statement_true]
  constructor
  · exact Nat.le_refl 39
  constructor
  · exact Nat.succ_le_succ (le_refl 38)
  · rfl
  sorry

end notebook_statements_correct_l817_817048


namespace sum_of_digits_gcd_of_differences_l817_817239

def gcd (a b : ℕ) : ℕ := 
if b = 0 then a else gcd b (a % b)

def sum_of_digits (n : ℕ) : ℕ :=
(n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10) -- assuming n < 10000

theorem sum_of_digits_gcd_of_differences :
  let d1 := 4665 - 1305
  let d2 := 6905 - 4665
  let d3 := 6905 - 1305
  let n := gcd (gcd d1 d2) d3
  sum_of_digits n = 4 :=
by
  sorry

end sum_of_digits_gcd_of_differences_l817_817239


namespace circle_radii_order_l817_817352

theorem circle_radii_order (r_A r_B r_C : ℝ) 
  (h1 : r_A = Real.sqrt 10) 
  (h2 : 2 * Real.pi * r_B = 10 * Real.pi)
  (h3 : Real.pi * r_C^2 = 16 * Real.pi) : 
  r_C < r_A ∧ r_A < r_B := 
  sorry

end circle_radii_order_l817_817352


namespace sum_of_last_two_digits_l817_817219

theorem sum_of_last_two_digits (a b : ℕ) (ha: a = 6) (hb: b = 10) :
  ((a^15 + b^15) % 100) = 0 :=
by
  -- ha, hb represent conditions given.
  sorry

end sum_of_last_two_digits_l817_817219


namespace max_mn_value_l817_817952

-- Define the function f(x)
def f (m n : ℝ) (x : ℝ) : ℝ := (1/2) * (2 - m) * x^2 + (n - 8) * x + 1

-- Define the condition on m
axiom m_pos (m: ℝ) : m > 2

-- Define the condition of monotonic decreasing
axiom f_mon_decreasing (m n : ℝ) (m_pos : m > 2) : 
  (∀ x ∈ set.Icc (-2 : ℝ) (-1 : ℝ), deriv (f m n) x ≤ 0)

-- The theorem we aim to prove: the maximum value of mn is 18 given the conditions
theorem max_mn_value (m n : ℝ) (h₁ : m > 2) (h₂ : ∀ x ∈ set.Icc (-2 : ℝ) (-1 : ℝ), deriv (f m n) x ≤ 0) :
  mn ≤ 18 :=
by sorry

end max_mn_value_l817_817952


namespace find_unknown_number_l817_817944

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end find_unknown_number_l817_817944


namespace similar_projected_quadrilateral_l817_817385

theorem similar_projected_quadrilateral
  (A B C D : Point)
  (f : Point -> Point -> Points -> Point)
  (A1 := f A D B C)
  (B1 := f B A C D)
  (C1 := f C B D A)
  (D1 := f D C A B)
  : similar_quadrilateral (A1, B1, C1, D1) (A, B, C, D) :=
  sorry

end similar_projected_quadrilateral_l817_817385


namespace division_remainder_l817_817350

theorem division_remainder : 
  ∃ q r, 1234567 = 123 * q + r ∧ r < 123 ∧ r = 41 := 
by
  sorry

end division_remainder_l817_817350


namespace sum_of_integers_from_neg15_to_5_l817_817694

-- defining the conditions
def first_term : ℤ := -15
def last_term : ℤ := 5

-- sum of integers from first_term to last_term
def sum_arithmetic_series (a l : ℤ) : ℤ :=
  let n := l - a + 1
  (n * (a + l)) / 2

-- the statement we need to prove
theorem sum_of_integers_from_neg15_to_5 : sum_arithmetic_series first_term last_term = -105 := by
  sorry

end sum_of_integers_from_neg15_to_5_l817_817694


namespace parallel_line_slope_l817_817704

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 9) : ∃ (m : ℝ), m = 1 / 2 := 
sorry

end parallel_line_slope_l817_817704


namespace square_area_l817_817556

-- Definitions based on conditions
def side_AB : ℕ := 28
def side_CD : ℕ := 58

-- We denote x as the side length of square BCFE.
def side_length_square := 28 * 58

-- The goal is to prove that the area of square BCFE is 1624.
theorem square_area (x : ℕ) (h₁ : x * x = 28 * 58) : x * x = 1624 := by
  rw [h₁]
  norm_num
  sorry

end square_area_l817_817556


namespace count_two_digit_remainders_l817_817803

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817803


namespace problem_order_of_cos_sin_powers_l817_817409

theorem problem_order_of_cos_sin_powers (α : ℝ) 
    (h1 : α ∈ set.Ioo (π/4) (π/2)) :
    (cos α)^(sin α) < (cos α)^(cos α) ∧ (cos α)^(cos α) < (sin α)^(cos α) :=
by sorry

end problem_order_of_cos_sin_powers_l817_817409


namespace find_fff1_l817_817121

def f (x : ℝ) : ℝ :=
if x ≤ 3 then x^3 else x^(1/3)

theorem find_fff1 : f (f (f 1)) = 1 := 
by
  sorry

end find_fff1_l817_817121


namespace sum_of_integers_from_neg15_to_5_l817_817695

theorem sum_of_integers_from_neg15_to_5 : 
  (∑ x in Finset.Icc (-15 : ℤ) 5, x) = -105 := 
by
  sorry

end sum_of_integers_from_neg15_to_5_l817_817695


namespace solution_set_of_inequality_l817_817036

def f (x : ℝ) := |2 * x + 1|

theorem solution_set_of_inequality : 
  { x : ℝ | f(x) ≤ 10 - |x - 3| } = 
  { x : ℝ | -8 / 3 ≤ x ∧ x ≤ 4 } := 
sorry

lemma f_m_n_geq_16 {m n : ℝ} (hm : 0 < m) (hn : 0 < n) (h : m + 2 * n = m * n) : 
  f(m) + f(-2 * n) ≥ 16 := 
sorry

end solution_set_of_inequality_l817_817036


namespace compare_sizes_l817_817011

theorem compare_sizes (a b c : ℝ) (h1 : a = 5^0.2) (h2 : b = (1/6)^3) (h3 : c = Real.logBase 3 (1/2)) : a > b ∧ b > c :=
by
  sorry

end compare_sizes_l817_817011


namespace count_interesting_numbers_l817_817832

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817832


namespace linear_function_proof_l817_817512

variable (g : ℝ → ℝ)
variable (h : LinearMap ℝ ℝ) (a k : ℝ)

variables (h_lin : ∀ x y : ℝ, g(x) = h x + a)
variables (h_eq : g 8 - g 5 = 9)

theorem linear_function_proof :
  g 16 - g 5 = 33 := by
  sorry

end linear_function_proof_l817_817512


namespace num_rem_three_by_seven_l817_817859

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817859


namespace positive_two_digit_integers_remainder_3_l817_817809

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817809


namespace eval_expression_l817_817369

theorem eval_expression : 5 - 7 * (8 - 12 / 3^2) * 6 = -275 := by
  sorry

end eval_expression_l817_817369


namespace range_of_b_l817_817780

noncomputable def distance_from_point_to_line (x y b : ℝ) : ℝ :=
|b| / Real.sqrt 2

theorem range_of_b (b : ℝ) :
  (∃ (c r : ℝ), c = (1, 1) ∧ r = 2 ∧ 
   ((x, y) ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 4} → 
   distance_from_point_to_line 1 1 b = 1) ∧
   1 < distance_from_point_to_line 1 1 b ∧ distance_from_point_to_line 1 1 b < 3) ↔ 
  (-3 * Real.sqrt 2 < b ∧ b < -Real.sqrt 2) ∨ (Real.sqrt 2 < b ∧ b < 3 * Real.sqrt 2) :=
sorry

end range_of_b_l817_817780


namespace exists_number_divisible_by_p_or_q_l817_817499

variable (p q : ℕ) (h_p_prime : Nat.Prime p) (h_q_prime : Nat.Prime q) (h_pq_distinct : p ≠ q)
variable (α : ℝ) (h_α_range : 0 < α ∧ α < 3)

theorem exists_number_divisible_by_p_or_q :
  ∃ n : ℕ, n < 2 * p * q ∧ (p ∣ ⌊n * α⌋ ∨ q ∣ ⌊n * α⌋) :=
sorry

end exists_number_divisible_by_p_or_q_l817_817499


namespace count_two_digit_integers_with_remainder_3_l817_817849

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817849


namespace lines_concur_l817_817087

theorem lines_concur 
  (A1 A2 A3 : Type*) 
  (l : Type*) 
  (α1 α2 α3 : ℝ) 
  (d : ℝ) [has_add d] [has_sub d] [has_mul d](h1 : d = 90) 
  (angle_l_A1_A2 : l.angles A1 A2 = α3) 
  (angle_l_A2_A3 : l.angles A2 A3 = α1)
  (angle_l_A3_A1 : l.angles A3 A1 = α2) :
  ∃ P : Type*, 
    Line_through A1 ((2 * d) - α1) l P ∧ 
    Line_through A2 ((2 * d) - α2) l P ∧ 
    Line_through A3 ((2 * d) - α3) l P :=
sorry

end lines_concur_l817_817087


namespace find_equation_of_ellipse_find_equation_of_line_l817_817022

-- Definitions of conditions given in the problem
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def passes_through_point (a b : ℝ) : Prop :=
  ∃ x y, is_ellipse a b x y ∧ x = 1 ∧ y = √2 / 2

def focal_length (a b c : ℝ) : Prop :=
  c = 1 ∧ a^2 = b^2 + c^2

-- Hypothesis and goals
theorem find_equation_of_ellipse (a b : ℝ) :
  passes_through_point a b ∧ focal_length a b 1 →
  a = √2 ∧ b = 1 :=
sorry

theorem find_equation_of_line (a b : ℝ) (k : ℝ) (x y : ℝ) :
  a = √2 ∧ b = 1 →
  (x = -2 ∧ y = 0) →
  ∃ k, 
    k = (2 - √2) / 2 ∨ k = 0 :=
sorry


end find_equation_of_ellipse_find_equation_of_line_l817_817022


namespace line_parallel_to_intersection_line_l817_817058

variables {L P₁ P₂ : Type}

-- Definitions and conditions
def is_parallel (l1 l2 : Type) : Prop := sorry  -- We will use this as our generic parallel definition

-- Assume L is parallel to the two planes P₁ and P₂
axiom L_parallel_P1 : is_parallel L P₁
axiom L_parallel_P2 : is_parallel L P₂

-- Assume P₁ and P₂ intersect and define their intersection line as l
def intersection_line (p1 p2 : Type) : Type := sorry
def l := intersection_line P₁ P₂

-- Theorem to be proven
theorem line_parallel_to_intersection_line : is_parallel L l :=
sorry

end line_parallel_to_intersection_line_l817_817058


namespace find_m_solve_inequality_l817_817389

-- Problem (Ⅰ)
theorem find_m (m : ℝ) (h_power_function : ∀ x : ℝ, f x = (m^2 - m - 1) * x^(-5*m - 1))
  (h_increasing : ∀ x y : ℝ, 0 < x → x < y → 0 < y → f x < f y) : m = -1 := 
sorry

-- Problem (Ⅱ)
theorem solve_inequality {x : ℝ} (h_m_eq_neg_one : f = λ x, x^4)
  (h_ineq : f (x - 2) > 16) : x > 4 ∨ x < 0 :=
sorry

end find_m_solve_inequality_l817_817389


namespace exists_k_l817_817247

variable {n : ℕ} (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom n_ge_two : n ≥ 2
axiom a_bounds : ∀ (i : ℕ), 1 ≤ i → i ≤ n → i ≤ a i ∧ a i ≤ n
axiom S_def : ∀ i, S i = ∑ k in Finset.range (i + 1), a k

-- Proof statement
theorem exists_k (h : ∑ i in Finset.range (n + 1), a i ≠ 0) :
  ∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ a k ^ 2 + S (n - k) < 2 * S n - n * (n + 1) / 2 :=
sorry

end exists_k_l817_817247


namespace triangle_area_l817_817461

open Real

def Point := (ℝ × ℝ)

noncomputable def midpoint (A B : Point) : Point := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def distance (A B : Point) : ℝ := 
  sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

noncomputable def area_triangle (A B C : Point) : ℝ := 
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

/-- In rectangle EFGH with EF = 10 and EH = 15, point N is the midpoint of segment EH. 
    A circle with center N and radius 5 intersects line FG at point P.
    The area of triangle ENP is 18.75. -/
theorem triangle_area {E F G H N P : Point} (h_rect : E = (0, 0) ∧ F = (10, 0) ∧ G = (10, 15) ∧ H = (0, 15))
  (h_mid : N = midpoint E H) 
  (h_dist : distance N P = 5) 
  : area_triangle E N P = 18.75 := 
sorry

end triangle_area_l817_817461


namespace remainder_of_1234567_div_123_l817_817348

theorem remainder_of_1234567_div_123 : 1234567 % 123 = 129 :=
by
  sorry

end remainder_of_1234567_div_123_l817_817348


namespace probability_LiMing_board_bus1_first_l817_817526

theorem probability_LiMing_board_bus1_first :
  ∀ (x y : ℝ), (0 < x ∧ x < 8) ∧ (0 < y ∧ y < 10) →
                (∃ p: ℝ, p = 0.6 ∧ 
                (p = (λ s : set (ℝ × ℝ), ∫⁻ (a : (ℝ × ℝ)), indicator s (8, 10))) {q : ℝ × ℝ | q.1 < q.2}) :=
by
  sorry

end probability_LiMing_board_bus1_first_l817_817526


namespace existence_of_inf_polynomials_l817_817146

noncomputable def P_xy_defined (P : ℝ→ℝ) (x y z : ℝ) :=
  P x ^ 2 + P y ^ 2 + P z ^ 2 + 2 * P x * P y * P z = 1

theorem existence_of_inf_polynomials (x y z : ℝ) (P : ℕ → ℝ → ℝ) :
  (x^2 + y^2 + z^2 + 2 * x * y * z = 1) →
  (∀ n, P (n+1) = P n ∘ P n) →
  P_xy_defined (P 0) x y z →
  ∀ n, P_xy_defined (P n) x y z :=
by
  intros h1 h2 h3
  sorry

end existence_of_inf_polynomials_l817_817146


namespace gel_pen_price_relation_b_l817_817294

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l817_817294


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817885

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817885


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817910

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817910


namespace final_selling_price_calc_l817_817262

def calculate_fsp (cp : ℝ) (profit_percent : ℝ) (discount_percent : ℝ) (tax_percent : ℝ) : ℝ :=
  let mp := cp * (1 + profit_percent / 100)
  let dp := mp * (1 - discount_percent / 100)
  dp * (1 + tax_percent / 100)

theorem final_selling_price_calc :
  calculate_fsp 800 10 5 12 = 936.32 := 
sorry

end final_selling_price_calc_l817_817262


namespace coconuts_problem_l817_817721

theorem coconuts_problem : ∃ n : ℕ, 
  let n1 := (4 * (n - 1)) / 5 in
  let n2 := (4 * (n1 - 1)) / 5 in
  let n3 := (4 * (n2 - 1)) / 5 in
  let n4 := (4 * (n3 - 1)) / 5 in
  let n5 := (4 * (n4 - 1)) / 5 in
  n5 % 5 = 0 ∧ n = 3121 :=
begin
  sorry
end

end coconuts_problem_l817_817721


namespace pen_price_ratio_l817_817319

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l817_817319


namespace min_value_of_sum_of_squares_l817_817415

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 10) : 
  x^2 + y^2 + z^2 ≥ 100 / 29 :=
sorry

end min_value_of_sum_of_squares_l817_817415


namespace exists_positive_integers_for_equation_l817_817684

theorem exists_positive_integers_for_equation :
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a^4 = b^3 + c^2 :=
by
  sorry

end exists_positive_integers_for_equation_l817_817684


namespace count_non_empty_subsets_with_odd_l817_817106

theorem count_non_empty_subsets_with_odd :
  ∃ (M : set (set ℕ)), 
    (∀ (A ∈ M), A ≠ ∅ ∧ A ⊆ {1, 2, 3} ∧ ∃ x ∈ A, x % 2 = 1) ∧
    M.card = 6 := 
sorry

end count_non_empty_subsets_with_odd_l817_817106


namespace value_a_plus_omega_l817_817070

noncomputable def function (x : ℝ) (ω : ℝ) (a : ℝ) : ℝ :=
  sin (ω * x) + a * cos (ω * x)

theorem value_a_plus_omega (ω a : ℝ) (hω : ω > 0)
  (h_symmetry : function (π/3) ω a = 0)
  (h_minimum : ∀ x, x ≠ π/6 → function x ω a > function (π/6) ω a) :
  a + ω = 9 :=
sorry

end value_a_plus_omega_l817_817070


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817333

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817333


namespace find_unknown_number_l817_817942

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end find_unknown_number_l817_817942


namespace quadratic_sum_terms_l817_817584

theorem quadratic_sum_terms (a b c : ℝ) :
  (∀ x : ℝ, -2 * x^2 + 16 * x - 72 = a * (x + b)^2 + c) → a + b + c = -46 :=
by
  sorry

end quadratic_sum_terms_l817_817584


namespace horizontal_asymptote_value_l817_817936

theorem horizontal_asymptote_value :
  ∀ (x : ℝ),
  ((8 * x^4 + 6 * x^3 + 7 * x^2 + 2 * x + 4) / 
  (2 * x^4 + 5 * x^3 + 3 * x^2 + x + 6)) = (4 : ℝ) :=
by sorry

end horizontal_asymptote_value_l817_817936


namespace neg_p_sufficient_for_neg_q_l817_817405

variable (x : ℝ)

def p : Prop := |x + 1| > 2
def q : Prop := x ≥ 2

theorem neg_p_sufficient_for_neg_q 
  (neg_p : -3 ≤ x ∧ x ≤ 1) 
  (neg_q : x < 2) :
  (neg_p → neg_q) ∧ ¬(neg_q → neg_p) := 
by
  sorry

end neg_p_sufficient_for_neg_q_l817_817405


namespace count_two_digit_integers_with_remainder_3_l817_817851

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817851


namespace symmetric_axis_of_g_l817_817391

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6))

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (Real.pi / 6))

theorem symmetric_axis_of_g :
  ∃ k : ℤ, (∃ x : ℝ, g x = 2 * Real.sin (k * Real.pi + (Real.pi / 2)) ∧ x = (k * Real.pi) / 2 + (Real.pi / 3)) :=
sorry

end symmetric_axis_of_g_l817_817391


namespace Mr_Lee_probability_l817_817528

noncomputable def probability_more_grandsons_or_granddaughters : ℚ :=
  let n := 12
  let p := 1 / 2
  let num_ways_6_boys := Nat.choose n (n / 2)
  let total_ways := 2^n
  let prob_equal_boys_and_girls := (num_ways_6_boys : ℚ) / (total_ways : ℚ)
  1 - prob_equal_boys_and_girls

theorem Mr_Lee_probability : probability_more_grandsons_or_granddaughters = 793 / 1024 := by
  sorry

end Mr_Lee_probability_l817_817528


namespace ellipse_properties_l817_817414

section EllipseProperties

variables {a b : ℝ} (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the foci and the condition related to the foci
def sqrt_a2_minus_b2_eq_2 : Prop := real.sqrt (a^2 - b^2) = 2

-- Point P on the ellipse
variables {x1 y1 : ℝ} (P_on_ellipse : ellipse a b x1 y1)

-- Define the intersection point M
def intersection_point_M (M : ℝ × ℝ) : Prop := sorry -- Placeholder for the explicit locus of M

-- Define the tangent MP and its intersection point N with the line x = -2
variables {M N : ℝ × ℝ} (line_x_minus_2_intersect : N.1 = -2)

-- Prove the constant ratio
def constant_ratio (F1 N M : (ℝ × ℝ)) : Prop := 
  (F1.2 - N.2) / (F1.2 - M.2) = sorry -- Placeholder for the constant ratio calculation

theorem ellipse_properties :
  sqrt_a2_minus_b2_eq_2 a b a_pos b_pos a_gt_b ∧
  ∃ M, intersection_point_M a b x1 y1 P_on_ellipse M ∧
  ∀ N, line_x_minus_2_intersect N → constant_ratio (-real.sqrt (a^2 - b^2), 0) N M :=
begin
  split,
  { -- Proof for sqrt(a^2 - b^2) = 2
    sorry
  },
  { -- Proof for locus of M and constant ratio
    use sorry,
    intros N HN,
    sorry
  }
end

end EllipseProperties

end ellipse_properties_l817_817414


namespace solution_set_of_absolute_value_inequality_l817_817182

theorem solution_set_of_absolute_value_inequality {x : ℝ} : 
  (|2 * x - 3| > 1) ↔ (x < 1 ∨ x > 2) := 
sorry

end solution_set_of_absolute_value_inequality_l817_817182


namespace coeff_x_squared_is_160_l817_817778

noncomputable def coefficient_x_squared_expansion 
  (a : ℚ) 
  (h : (3 * 1 + a / (2 * 1)) * (2 * 1 - 1 / 1)^5 = 4) : ℕ :=
  let coefficient_x_squared := 160 in
  coefficient_x_squared

theorem coeff_x_squared_is_160 
  (a : ℚ) 
  (h : (3 * 1 + a / (2 * 1)) * (2 * 1 - 1 / 1)^5 = 4) : 
  coefficient_x_squared_expansion a h = 160 := by
  sorry

end coeff_x_squared_is_160_l817_817778


namespace sin_cos_inequality_l817_817363

theorem sin_cos_inequality (a : ℝ) (h_a : a < 0) :
  (∀ x : ℝ, sin x ^ 2 + a * cos x + a ^ 2 ≥ 1 + cos x) → a ≤ -2 :=
begin
  sorry,
end

end sin_cos_inequality_l817_817363


namespace repaint_all_white_l817_817172

/-- The numbers from 1 to 1,000,000 are initially colored black. In one move, a number can be picked
which will recolor itself and all numbers not coprime to it in the opposite color. Prove that it is
possible to make all numbers white in several moves. -/
theorem repaint_all_white :
  (∀ n ∈ finset.Icc 1 1000000, black n) →
  (∀ n ∈ finset.Icc 1 1000000, (choose_num_repaint n)) →
  (∀ n ∈ finset.Icc 1 1000000, white n) :=
sorry

end repaint_all_white_l817_817172


namespace integral_e_exp_2x_is_e_minus_inverse_l817_817710

noncomputable def integral_evaluate : Real :=
  ∫ (x : Real) in -1..1, (Real.exp x + 2*x)

theorem integral_e_exp_2x_is_e_minus_inverse : 
  integral_evaluate = (Real.exp 1 - Real.exp (-1)) :=
by
  sorry

end integral_e_exp_2x_is_e_minus_inverse_l817_817710


namespace min_value_ap_bp_cp_l817_817783

-- Definitions for equilateral triangle and point in plane
structure Point2D :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (A B C : Point2D)
  (equilateral : (dist A B = dist B C) ∧ (dist B C = dist C A))

-- Distance function between two points in the plane
def dist (p q : Point2D) : ℝ :=
  real.sqrt ((q.x - p.x)^2 + (q.y - p.y)^2)

-- Function f defined as AP + BP - CP
def f (P A B C : Point2D) : ℝ :=
  dist P A + dist P B - dist P C

-- Definition of segment
def segment (P Q : Point2D) : set Point2D :=
  { t : Point2D | ∃ λ ∈ (set.Icc 0 1), t.x = λ * P.x + (1 - λ) * Q.x ∧ t.y = λ * P.y + (1 - λ) * Q.y }

-- Proof statement
theorem min_value_ap_bp_cp (ABC : Triangle) (P : Point2D) :
  let circ_arc := { P | ∃ θ ∈ (set.Icc (2*ℝ.pi/3) (4*ℝ.pi/3)), P = rotate θ ABC.A ABC.B } in
  (P ∈ circ_arc ∧ P ≠ ABC.C) ↔ (f P ABC.A ABC.B ABC.C = 0) :=
sorry

end min_value_ap_bp_cp_l817_817783


namespace P_2n_expression_l817_817111

noncomputable def a (n : ℕ) : ℕ :=
  2 * n + 1

noncomputable def S (n : ℕ) : ℕ :=
  n * (n + 2)

noncomputable def b (n : ℕ) : ℕ :=
  2 ^ (n - 1)

noncomputable def T (n : ℕ) : ℕ :=
  2 * b n - 1

noncomputable def c (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 / S n else a n * b n
  
noncomputable def P (n : ℕ) : ℕ :=
  if n % 2 = 0 then (12 * n - 1) * 2^(2 * n + 1) / 9 + 2 / 9 + 2 * n / (2 * n + 1) else 0

theorem P_2n_expression (n : ℕ) : 
  P (2 * n) = (12 * n - 1) * 2^(2 * n + 1) / 9 + 2 / 9 + 2 * n / (2 * n + 1) :=
sorry

end P_2n_expression_l817_817111


namespace polo_shirt_cost_l817_817527

theorem polo_shirt_cost :
  (∃ P : ℝ, 
     let total_cost : ℝ := (3 * P) + (2 * 83) + 90 - 12,
     total_cost = 322) →
  ∃ P : ℝ, P = 26 :=
by
  intro h,
  cases' h with P h_total_cost,
  use 26,
  -- Proof is omitted.
  sorry

end polo_shirt_cost_l817_817527


namespace find_lambda_proof_l817_817043

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

def unit_vector (v : V) : Prop := ∥v∥ = 1

def angle_between_vectors (u v : V) (θ : ℝ) : Prop := 
  u ≠ 0 ∧ v ≠ 0 ∧ real.cos θ * ∥u∥ * ∥v∥ = ⟪u, v⟫

noncomputable def find_lambda (a b : V) (λ : ℝ) : Prop :=
  unit_vector a ∧
  unit_vector b ∧
  angle_between_vectors a b (real.pi / 3) ∧
  ⟪a, (a - λ • b)⟫ = 0 → λ = 2

-- The theorem to be proved
theorem find_lambda_proof : ∀ (a b : V) (λ : ℝ),
  find_lambda a b λ :=
by
  sorry  -- Proof goes here

end find_lambda_proof_l817_817043


namespace percentage_increase_edge_length_l817_817440

theorem percentage_increase_edge_length (a a' : ℝ) (h : 6 * (a')^2 = 6 * a^2 + 1.25 * 6 * a^2) : a' = 1.5 * a :=
by sorry

end percentage_increase_edge_length_l817_817440


namespace cost_of_tax_free_items_is_ten_l817_817985

-- Definitions and Conditions
def total_spent : ℝ := 40
def percent_sales_tax : ℝ := 0.30
def tax_rate : ℝ := 0.06

-- Calculate cost of taxable items and sales tax paid
def cost_taxable_items := 0.70 * total_spent
def sales_tax_paid := Float.ceil (tax_rate * cost_taxable_items)

-- Calculate cost of tax-free items
def cost_tax_free_items := total_spent - (cost_taxable_items + sales_tax_paid)

-- Statement to prove
theorem cost_of_tax_free_items_is_ten : cost_tax_free_items = 10 := 
by sorry

end cost_of_tax_free_items_is_ten_l817_817985


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817916

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817916


namespace k_h_5_eq_148_l817_817930

def h (x : ℤ) : ℤ := 4 * x + 6
def k (x : ℤ) : ℤ := 6 * x - 8

theorem k_h_5_eq_148 : k (h 5) = 148 := by
  sorry

end k_h_5_eq_148_l817_817930


namespace unknown_number_value_l817_817941

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end unknown_number_value_l817_817941


namespace count_two_digit_integers_remainder_3_div_7_l817_817896

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817896


namespace sector_central_angle_l817_817174

-- Definitions for the conditions
def arc_length (r l : ℝ) := l
def radius (r l : ℝ) := r
def perimeter (r l : ℝ) := 2 * r + l
def area (r l : ℝ) := (1 / 2) * l * r
def central_angle (r l : ℝ) := l / r

-- The main statement to prove
theorem sector_central_angle {r l : ℝ} :
  perimeter r l = 8 ∧ area r l = 3 → 
  central_angle r l = 6 ∨ central_angle r l = (2 / 3) :=
  sorry

end sector_central_angle_l817_817174


namespace domain_when_a_eq_1_range_of_a_for_two_distinct_real_roots_range_of_a_for_monotonic_f_l817_817038

-- Domain of the function when a = 1
theorem domain_when_a_eq_1 (x : ℝ) : 
  (a = 1) → (f(x) = sqrt(abs(x + 1) - 1) - x) → 
  (x ∈ domain f ↔ x ∈ Icc (-∞) (-2) ∪ Icc 0 ∞) :=
by sorry

-- Range of a for f(ax) = a to have 2 distinct real roots
theorem range_of_a_for_two_distinct_real_roots (a : ℝ) : 
  (a ≠ 0) → (∀ x, f(ax) = a) → 
  (interval (0, 1/4)) :=
by sorry

-- Range of a for f(x) to be monotonic within its domain
theorem range_of_a_for_monotonic_f (a : ℝ) : 
  (∀ x, monotonic_on f (domain f)) → 
  (interval (-∞, -1/4]) :=
by sorry

end domain_when_a_eq_1_range_of_a_for_two_distinct_real_roots_range_of_a_for_monotonic_f_l817_817038


namespace num_rem_three_by_seven_l817_817853

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817853


namespace find_number_l817_817948

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end find_number_l817_817948


namespace seating_arrangements_l817_817459

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def num_seating_arrangements (total_people : ℕ) : ℕ :=
  factorial total_people

def num_unit_arrangements (remaining_people : ℕ) (unit_size : ℕ) : ℕ :=
  factorial remaining_people * factorial unit_size

theorem seating_arrangements (total_people unit_size : ℕ) (refusing_seat : total_people = 10) (unit_refusing : unit_size = 4) :
  num_seating_arrangements total_people - num_unit_arrangements (total_people - unit_size + 1) unit_size = 3507840 := by
  sorry

end seating_arrangements_l817_817459


namespace sum_logarithms_divisors_10_pow_n_l817_817183

theorem sum_logarithms_divisors_10_pow_n {n : ℕ} :
  (∑ a in Finset.range (n + 1), ∑ b in Finset.range (n + 1), 
    (a : ℝ) * Real.log10 2 + (b : ℝ) * Real.log10 5) =
    2 * 1452 ↔ n = 14 :=
by
  sorry

end sum_logarithms_divisors_10_pow_n_l817_817183


namespace count_two_digit_integers_remainder_3_l817_817865

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817865


namespace count_interesting_numbers_l817_817833

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817833


namespace unknown_number_value_l817_817939

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end unknown_number_value_l817_817939


namespace find_a_l817_817786

def f (a x : ℝ) : ℝ := (x + a) / Real.exp x

def tangent_line_y (x : ℝ) (m b : ℝ) : ℝ := m * x + b

theorem find_a (a : ℝ) (f' : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f' x = (1 - x - a) / Real.exp x) →
  (∀ x, tangent_line_y x m 2/e = x/e + 2/e) →
  f' 1 = m →
  a = -1 :=
by
  intro h1 h2 h3
  -- The proof details would follow from here.
  sorry

end find_a_l817_817786


namespace calculate_x16_in_12_operations_calculate_xn_in_log_operations_l817_817745

variables (x : ℝ) (n : ℕ)
    
-- Part (a)
theorem calculate_x16_in_12_operations (hx : x ≠ 0) : ∃ (ops : ℕ), ops ≤ 12 ∧ x ^ 16 = (x * x) * (x * x) * (x * x) * (x * x)
  :=
begin
  -- Use any logical steps to prove it can be achieved within 12 operations
  sorry
end

-- Part (b)
theorem calculate_xn_in_log_operations (hx : x ≠ 0) : ∃ (ops : ℕ), ops ≤ 1 + 1.5 * log 2 n ∧ x ^ n = x ^ n :=
begin
  -- Use mathematical induction or any logical steps for the proof
  sorry
end

end calculate_x16_in_12_operations_calculate_xn_in_log_operations_l817_817745


namespace smallest_piece_length_l817_817274

theorem smallest_piece_length {x : ℝ} : (5 - x) + (12 - x) ≤ (13 - x) → 
                                     x ≥ 4 :=
by
  intro h
  have h1 : 17 - 2 * x ≤ 13 - x := h
  linarith

end smallest_piece_length_l817_817274


namespace radius_of_circle_C1_l817_817697

noncomputable def center_lies_on_circle (C1 C2 : Circle) (O : Point) : Prop :=
  O = C1.center ∧ C1.center ∈ C2

noncomputable def circles_meet_at_points (C1 C2 : Circle) (X Y : Point) : Prop :=
  (X ∈ C1 ∧ X ∈ C2) ∧ (Y ∈ C1 ∧ Y ∈ C2)

noncomputable def point_on_circle_exterior_others (C2 : Circle) (Z : Point) (C1 : Circle) : Prop :=
  Z ∈ C2 ∧ Z ∉ C1

noncomputable def distance (A B : Point) : ℝ := sorry -- Define distance between two points

variables (C1 C2 : Circle) (O X Y Z : Point)

theorem radius_of_circle_C1 (h_center : center_lies_on_circle C1 C2 O)
  (h_meet : circles_meet_at_points C1 C2 X Y)
  (h_exterior : point_on_circle_exterior_others C2 Z C1)
  (h_dist1 : distance X Z = 15)
  (h_dist2 : distance O Z = 17)
  (h_dist3 : distance Y Z = 8) :
  C1.radius = sqrt 34 :=
sorry

end radius_of_circle_C1_l817_817697


namespace seq_2011_l817_817586

-- Define the sequence according to the conditions.
def seq : ℕ → ℚ
| 0 := 2
| (n + 1) := (seq n - 1) / (seq n + 1)

-- Theorem to prove that t_{2011} = -1/2.
theorem seq_2011 : seq 2010 = -1 / 2 :=
by 
  sorry

end seq_2011_l817_817586


namespace num_values_f_f_eq_4_l817_817574

def f (x : ℝ) : ℝ :=
if -4 ≤ x ∧ x ≤ -2 then -x^2 - 2 * x + 6
else if -2 < x ∧ x ≤ 2 then x + 4
else if 2 < x ∧ x ≤ 4 then x^2 - 4 * x + 8
else 0

theorem num_values_f_f_eq_4 : 
  let valid_x := {x : ℝ | -4 ≤ x ∧ x ≤ 4 ∧ f (f x) = 4} in
  #valid_x = 2 :=
by
  sorry

end num_values_f_f_eq_4_l817_817574


namespace circle_area_l817_817690

theorem circle_area (d : ℝ) (h : d = 8) : ∃ A : ℝ, A = 16 * Real.pi ∧
  A = Real.pi * (d / 2) ^ 2 :=
by
  use 16 * Real.pi
  split
  · rfl
  · rw [h, Real.pi, (8 / 2 : ℝ), (4 ^ 2 : ℝ)]
  sorry

end circle_area_l817_817690


namespace sum_arithmetic_sequence_l817_817749

variables {a1 d : ℝ} {n : ℕ}

def arithmetic_seq (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

def Sn (a1 d : ℝ) (n : ℕ) : ℝ := n/2 * (2 * a1 + (n - 1) * d)

def circle (x : ℝ) (y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

def intersects (a1 : ℝ) : set (ℝ × ℝ) := {p : ℝ × ℝ | line a1 p.1 = p.2 ∧ circle p.1 p.2}

def symmetric (p1 p2 : ℝ × ℝ) (d : ℝ) : Prop := p1.1 + p1.2 + d = 0 ∧ p2.1 + p2.2 + d = 0

theorem sum_arithmetic_sequence (h : ∃ p1 p2 : ℝ × ℝ, p1 ∈ intersects a1 ∧ p2 ∈ intersects a1 ∧ symmetric p1 p2 d) :
  Sn a1 d n = -n^2 + 2 * n :=
sorry

end sum_arithmetic_sequence_l817_817749


namespace count_interesting_numbers_l817_817831

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817831


namespace total_revenue_l817_817983

theorem total_revenue (chips_sold : ℕ) (chips_price : ℝ) (hotdogs_sold : ℕ) (hotdogs_price : ℝ)
(drinks_sold : ℕ) (drinks_price : ℝ) (sodas_sold : ℕ) (lemonades_sold : ℕ) (sodas_ratio : ℕ)
(lemonades_ratio : ℕ) (h1 : chips_sold = 27) (h2 : chips_price = 1.50) (h3 : hotdogs_sold = chips_sold - 8)
(h4 : hotdogs_price = 3.00) (h5 : drinks_sold = hotdogs_sold + 12) (h6 : drinks_price = 2.00)
(h7 : sodas_ratio = 2) (h8 : lemonades_ratio = 3) (h9 : sodas_sold = (sodas_ratio * drinks_sold) / (sodas_ratio + lemonades_ratio))
(h10 : lemonades_sold = drinks_sold - sodas_sold) :
chips_sold * chips_price + hotdogs_sold * hotdogs_price + drinks_sold * drinks_price = 159.50 := 
by
  -- Proof is left as an exercise for the reader
  sorry

end total_revenue_l817_817983


namespace june_spent_on_music_books_l817_817487

theorem june_spent_on_music_books
  (total_budget : ℤ)
  (math_books_cost : ℤ)
  (science_books_cost : ℤ)
  (art_books_cost : ℤ)
  (music_books_cost : ℤ)
  (h_total_budget : total_budget = 500)
  (h_math_books_cost : math_books_cost = 80)
  (h_science_books_cost : science_books_cost = 100)
  (h_art_books_cost : art_books_cost = 160)
  (h_total_cost : music_books_cost = total_budget - (math_books_cost + science_books_cost + art_books_cost)) :
  music_books_cost = 160 :=
sorry

end june_spent_on_music_books_l817_817487


namespace cost_of_800_pieces_of_gum_l817_817564

theorem cost_of_800_pieces_of_gum :
  (cost_per_piece : ℕ) (discount_threshold : ℕ) (discount_rate : ℚ) (pieces_bought : ℕ) 
  (total_cost_cents : ℚ) (total_cost_dollars : ℚ),
  cost_per_piece = 1 ∧ discount_threshold = 500 ∧ discount_rate = 0.10 ∧ pieces_bought = 800 ∧
  total_cost_cents = (cost_per_piece * pieces_bought : ℕ) - (if pieces_bought > discount_threshold then discount_rate * (cost_per_piece * pieces_bought : ℕ) else 0) ∧
  total_cost_dollars = total_cost_cents / 100 →
  total_cost_dollars = 7.20 := 
by
  intros,
  sorry

end cost_of_800_pieces_of_gum_l817_817564


namespace general_formula_a_n_sum_first_2n_c_n_l817_817759

noncomputable def a_n (n : ℕ) : ℝ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℝ := 3 ^ (n - 1)

def c_n (n : ℕ) : ℝ := a_n n + (-1) ^ n * b_n n

theorem general_formula_a_n (a_n : ℕ → ℝ) (d : ℝ) :
  (∀ n, a_n n = a_n 1 + (n - 1) * d) →
  (∃ q, b_n 2 = b_n 1 * q ∧ b_n 3 = b_n 2 * q) →
  a_n 1 = b_n 1 →
  a_n 14 = b_n 4 →
  a_n n = 2 * n - 1 := by
  sorry

theorem sum_first_2n_c_n (n : ℕ) :
  ∑ i in finset.range (2 * n), c_n (i + 1) = 4 * n^2 + (9^n / 4) - (1 / 4) := by
  sorry

end general_formula_a_n_sum_first_2n_c_n_l817_817759


namespace shortest_distance_proof_l817_817517

noncomputable def shortest_distance (k : ℝ) : ℝ :=
  let p := (k - 6) / 2
  let f_p := -p^2 + (6 - k) * p + 18
  let d := |f_p|
  d / (Real.sqrt (k^2 + 1))

theorem shortest_distance_proof (k : ℝ) :
  shortest_distance k = 
  |(-(k - 6) / 2^2 + (6 - k) * (k - 6) / 2 + 18)| / (Real.sqrt (k^2 + 1)) :=
sorry

end shortest_distance_proof_l817_817517


namespace convert_speed_kmph_to_mps_l817_817370

def kilometers_to_meters := 1000
def hours_to_seconds := 3600
def speed_kmph := 18
def expected_speed_mps := 5

theorem convert_speed_kmph_to_mps :
  speed_kmph * (kilometers_to_meters / hours_to_seconds) = expected_speed_mps :=
by
  sorry

end convert_speed_kmph_to_mps_l817_817370


namespace problem_statement_l817_817676

/-- 
  Define the four functions that we are considering: 
  y1: y = tan x
  y2: y = |sin x|
  y3: y = cos x
  y4: y = |cos x|
-/
def y1 (x : ℝ) : ℝ := Real.tan x
def y2 (x : ℝ) : ℝ := abs (Real.sin x)
def y3 (x : ℝ) : ℝ := Real.cos x
def y4 (x : ℝ) : ℝ := abs (Real.cos x)

/-- 
  Prove that y2 = |sin x| is an increasing function on (0, π/2),
  is an even function, and has a period of π.
-/
theorem problem_statement :
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < π/2 → y2 x < y2 y) ∧ -- increasing on (0, π/2)
  (∀ x : ℝ, y2 x = y2 (-x)) ∧ -- even function
  (∀ x : ℝ, y2 (x + π) = y2 x) -- period of π
:=
sorry

end problem_statement_l817_817676


namespace vector_combination_l817_817792

theorem vector_combination (x y : ℝ) : 
  let a := (3, -1)
      b := (-1, 2)
      c := (2, 1) in
  a = (λ x y, x * b.1 + y * b.2) x y ∧ a = (λ x y, x * c.1 + y * c.2) x y → x + y = 0 := 
begin
  sorry
end

end vector_combination_l817_817792


namespace count_two_digit_integers_remainder_3_div_7_l817_817900

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817900


namespace curve_is_hyperbola_with_y_axis_foci_l817_817075

-- Define the conditions
def angle_in_third_quadrant (θ : ℝ) : Prop := π < θ ∧ θ < 3 * π / 2
def curve_equation (θ x y : ℝ) : Prop := x^2 + y^2 * sin θ = cos θ

-- State the theorem based on the conditions
theorem curve_is_hyperbola_with_y_axis_foci (θ x y : ℝ) (h1 : angle_in_third_quadrant θ) (h2 : curve_equation θ x y) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (curve_equation θ x y ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
sorry

end curve_is_hyperbola_with_y_axis_foci_l817_817075


namespace sqrt_inequality_l817_817008

theorem sqrt_inequality 
  (a b c : ℝ) 
  (habc: a > b ∧ b > c) 
  (h_sum: a + b + c = 0) :
  sqrt (b^2 - a * c) > sqrt 3 * a :=
by
  sorry

end sqrt_inequality_l817_817008


namespace problem_statement_l817_817392

def f (x : Real) : Real :=
  2 * Real.tan x - (2 * (Real.sin (x / 2)) ^ 2 - 1) / (Real.sin (x / 2) * Real.cos (x / 2))

theorem problem_statement : f (Real.pi / 12) = 8 := by
  sorry

end problem_statement_l817_817392


namespace sqrt_expression_meaningful_l817_817062

theorem sqrt_expression_meaningful {x : ℝ} : (2 * x - 4) ≥ 0 → x ≥ 2 :=
by
  intro h
  sorry

end sqrt_expression_meaningful_l817_817062


namespace equation_solution_system_solution_l817_817554

theorem equation_solution (x : ℚ) :
  (3 * x + 1) / 5 = 1 - (4 * x + 3) / 2 ↔ x = -7 / 26 :=
by sorry

theorem system_solution (x y : ℚ) :
  (3 * x - 4 * y = 14) ∧ (5 * x + 4 * y = 2) ↔
  (x = 2) ∧ (y = -2) :=
by sorry

end equation_solution_system_solution_l817_817554


namespace division_remainder_l817_817349

theorem division_remainder : 
  ∃ q r, 1234567 = 123 * q + r ∧ r < 123 ∧ r = 41 := 
by
  sorry

end division_remainder_l817_817349


namespace solution_set_inequality_l817_817571

namespace MathProof

variable {f : ℝ → ℝ}

theorem solution_set_inequality (h1 : f 1 = 1)
                                (h2 : ∀ x : ℝ, f'' x > 1/3)
                                (x : ℝ) :
  f (Real.log x) < (2 + Real.log x) / 3 ↔ (0 < x) ∧ (x < Real.exp 1) := by
sorry

end MathProof

end solution_set_inequality_l817_817571


namespace max_four_cycles_is_54_l817_817961

noncomputable def max_four_cycles (points : Finset (Fin 36)) (edges : Finset (Fin 36 × Fin 36)) : ℕ :=
  -- definition to compute the number of 4-cycles given the points and edges
  sorry

theorem max_four_cycles_is_54 (points : Finset (Fin 36)) (edges : Finset (Fin 36 × Fin 36))
  (h1 : ∀ p ∈ points, ∃! q r s ∈ points, ¬(collinear p q r s))
  (h2 : ∀ p ∈ points, (edges.to_finset.filter (λ e, e.1 = p ∨ e.2 = p)).card ≤ 3)
  : max_four_cycles points edges = 54 :=
sorry

end max_four_cycles_is_54_l817_817961


namespace cosine_difference_l817_817734

variable (α β : ℝ)

theorem cosine_difference
  (h1 : sin α - sin β = 1 - sqrt 3 / 2)
  (h2 : cos α - cos β = 1 / 2) :
  cos (α - β) = sqrt 3 / 2 := 
sorry

end cosine_difference_l817_817734


namespace dance_ratio_l817_817596

theorem dance_ratio (total_attendees faculty_staff_percentage boys : ℕ) 
(faculty_staff_percentage = 10) (total_attendees = 100) (boys = 30) :
  let students := total_attendees - (total_attendees * faculty_staff_percentage / 100)
  let girls := students - boys
  (girls / boys) = 2 :=
by
  sorry

end dance_ratio_l817_817596


namespace find_number_l817_817949

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end find_number_l817_817949


namespace min_value_expression_l817_817516

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ( (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ) ≥ 7 :=
sorry

end min_value_expression_l817_817516


namespace largest_smallest_divisible_by_99_l817_817953

-- Definitions for distinct digits 3, 7, 9
def largest_number (x y z : Nat) : Nat := 100 * x + 10 * y + z
def smallest_number (x y z : Nat) : Nat := 100 * z + 10 * y + x

-- Proof problem statement
theorem largest_smallest_divisible_by_99 
  (a b c : Nat) (h : a > b ∧ b > c ∧ c > 0) : 
  ∃ (x y z : Nat), 
    (x = 9 ∧ y = 7 ∧ z = 3 ∧ largest_number x y z = 973 ∧ smallest_number x y z = 379) ∧
    99 ∣ (largest_number a b c - smallest_number a b c) :=
by
  sorry

end largest_smallest_divisible_by_99_l817_817953


namespace pq_proof_l817_817023

def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3)

def p : Prop := ∀ y, f y = 0 → y = -Real.pi / 12
def q : Prop := ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ 0 → f (x) > f (x + 0.001)  -- capturing decreasing trend roughly

theorem pq_proof : p ∨ q :=
by
  sorry

end pq_proof_l817_817023


namespace people_in_room_proof_l817_817600

-- Definitions corresponding to the problem conditions
def people_in_room (total_people : ℕ) : ℕ := total_people
def seated_people (total_people : ℕ) : ℕ := (3 * total_people / 5)
def total_chairs (total_people : ℕ) : ℕ := (3 * (5 * people_in_room total_people) / 2 / 5 + 8)
def empty_chairs : ℕ := 8
def occupied_chairs (total_people : ℕ) : ℕ := (2 * total_chairs total_people / 3)

-- Proving that there are 27 people in the room
theorem people_in_room_proof (total_chairs : ℕ) :
  (seated_people 27 = 2 * total_chairs / 3) ∧ 
  (8 = total_chairs - 2 * total_chairs / 3) → 
  people_in_room 27 = 27 :=
by
  sorry

end people_in_room_proof_l817_817600


namespace sequence_geometric_and_sum_l817_817000

theorem sequence_geometric_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) :
  a 1 = 2 ∧ (∀ n : ℕ, n > 0 → let x := (n : ℕ) in ∃ u v : ℕ, u ≠ v ∧ u + v = 1 + 2 * n ∧ u * v = b n) →
  (∀ n : ℕ, n > 0 → (a n - n = (-1)^(n-1) ∧ (∀ n, n > 0 →
  if n % 2 = 1 then (∑ i in list.range n, a (i+1)) = n * (n+1) / 2 + 1
  else (∑ i in list.range n, a (i+1)) = n * (n+1) / 2)))
 := sorry

end sequence_geometric_and_sum_l817_817000


namespace oak_taller_than_shortest_l817_817128

noncomputable def pine_tree_height : ℚ := 14 + 1 / 2
noncomputable def elm_tree_height : ℚ := 13 + 1 / 3
noncomputable def oak_tree_height : ℚ := 19 + 1 / 2

theorem oak_taller_than_shortest : 
  oak_tree_height - elm_tree_height = 6 + 1 / 6 := 
  sorry

end oak_taller_than_shortest_l817_817128


namespace angle_OAM_eq_angle_OXA_l817_817115

/-- Given:
  1. ABC is a non-isosceles, non-right triangle.
  2. ω is the circumcircle of △ABC with circumcenter O.
  3. M is the midpoint of segment BC.
  4. The tangents to ω at B and C intersect at X.
Prove that ∠OAM = ∠OXA. -/
theorem angle_OAM_eq_angle_OXA (A B C O M X : Point) (ω : Circle)
  (h1 : ¬ (IsoscelesTriangle A B C))
  (h2 : ¬ (RightTriangle A B C))
  (h3 : Circumcenter O A B C)
  (h4 : Midpoint M B C)
  (h5 : Tangents ω B = Tangents ω C = X) :
  ∠OAM = ∠OXA :=
sorry

end angle_OAM_eq_angle_OXA_l817_817115


namespace max_triangle_area_l817_817139

variable {Point : Type*} [euclidean_geometry Point]

/-- Prove that the area of any triangle inscribed in a convex polygon M is at most the area of the largest
    triangle formed by three vertices of M. -/
theorem max_triangle_area (M : finset Point) (h_convex : convex_hull (↑M : set Point)) 
  (XYZ : affine_simplex Point (fin 3)) (h_inscribed : ∀ (v : fin 3), XYZ.points v ∈ convex_hull (↑M : set Point)) :
  let largest_area := finset.sup (M.powerset.filter (λ s, s.card = 3))
                                  (λ s, area (convex_hull s).simplex)
  in area XYZ ≤ largest_area :=
sorry

end max_triangle_area_l817_817139


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817322

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817322


namespace no_such_finite_set_exists_l817_817093

open Real

noncomputable def finite_real_set (M : set ℝ) : Prop :=
  M.finite ∧ (∀ x ∈ M, x ≠ 0)

theorem no_such_finite_set_exists :
  ¬ (∃ (M : set ℝ) (hM : finite_real_set M),
       ∀ (n : ℕ), ∃ (P : polynomial ℝ), P.degree ≥ n ∧
                    (∀ (c : ℝ), c ∈ P.coeffs → c ∈ M) ∧
                    (∀ (r : ℝ), P.is_root r → r ∈ M)) :=
by
  sorry

end no_such_finite_set_exists_l817_817093


namespace find_number_l817_817947

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end find_number_l817_817947


namespace monotonic_intervals_of_g_range_of_a_l817_817639

noncomputable def f (a : ℝ) := λ x : ℝ, real.exp (-x) * real.sin x + a * x

def g (a : ℝ) (x : ℝ) := deriv (f a) x
def g' (a : ℝ) (x : ℝ) := deriv (g a) x

theorem monotonic_intervals_of_g (a : ℝ) :
  (∀ x ∈ Set.Icc (0 : ℝ) π/2, g' a x < 0) ∧
  (∀ x ∈ Set.Icc (π / 2) (3 * π / 2), g' a x > 0) ∧
  (∀ x ∈ Set.Icc (3 * π / 2) 2 * π, g' a x < 0) :=
sorry

theorem range_of_a (a : ℝ) :
  (-real.exp (-2 * π) < a ∧ a < real.exp (-π / 2)) ↔
  (∃ x ∈ Set.Ioo (0 : ℝ) 2 * π, is_local_max (f a) x ∧ is_local_min (f a) x) :=
sorry

end monotonic_intervals_of_g_range_of_a_l817_817639


namespace savings_wednesday_l817_817708

variable (m t s w : ℕ)

theorem savings_wednesday :
  m = 15 → t = 28 → s = 28 → 2 * s = 56 → 
  m + t + w = 56 → w = 13 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end savings_wednesday_l817_817708


namespace solve_equation1_solve_equation2_l817_817148

theorem solve_equation1 :
  ∀ x : ℝ, ((x-1) * (x-1) = 3 * (x-1)) ↔ (x = 1 ∨ x = 4) :=
by
  intro x
  sorry

theorem solve_equation2 :
  ∀ x : ℝ, (x^2 - 4 * x + 1 = 0) ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by
  intro x
  sorry

end solve_equation1_solve_equation2_l817_817148


namespace gel_pen_price_relation_b_l817_817296

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l817_817296


namespace coins_in_distinct_colors_l817_817497

theorem coins_in_distinct_colors 
  (n : ℕ)  (h1 : 1 < n) (h2 : n < 2010) : (∃ k : ℕ, 2010 = n * k) ↔ 
  ∀ i : ℕ, i < 2010 → (∃ f : ℕ → ℕ, ∀ j : ℕ, j < n → f (j + i) % n = j % n) :=
sorry

end coins_in_distinct_colors_l817_817497


namespace even_function_m_minus_n_l817_817421

def f (m n : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then m * Real.log x / Real.log 2017 + 3 * Real.sin x 
  else Real.log (-x) / Real.log 2017 + n * Real.sin x

theorem even_function_m_minus_n (m n : ℝ) (h_even : ∀ x : ℝ, f m n x = f m n (-x)) : m - n = 4 :=
by
  sorry

end even_function_m_minus_n_l817_817421


namespace isosceles_triangle_area_l817_817280

-- Definitions
def isosceles_triangle (b h : ℝ) : Prop :=
∃ a : ℝ, a * b / 2 = a * h

def square_of_area_one (a : ℝ) : Prop :=
a = 1

def centroids_coincide (g_triangle g_square : ℝ × ℝ) : Prop :=
g_triangle = g_square

-- The statement of the problem
theorem isosceles_triangle_area
  (b h : ℝ)
  (s : ℝ)
  (triangle_centroid : ℝ × ℝ)
  (square_centroid : ℝ × ℝ)
  (H1 : isosceles_triangle b h)
  (H2 : square_of_area_one s)
  (H3 : centroids_coincide triangle_centroid square_centroid)
  : b * h / 2 = 9 / 4 :=
by
  sorry

end isosceles_triangle_area_l817_817280


namespace sqrt_identity_l817_817242

theorem sqrt_identity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 0 < a^2 - b) :
  (sqrt (a + sqrt b) = sqrt ((a + sqrt (a^2 - b)) / 2) + sqrt ((a - sqrt (a^2 - b)) / 2)) ∧
  (sqrt (a - sqrt b) = sqrt ((a + sqrt (a^2 - b)) / 2) - sqrt ((a - sqrt (a^2 - b)) / 2)) :=
by
  sorry

end sqrt_identity_l817_817242


namespace cost_second_brand_l817_817625

-- Definitions of conditions
def cost_first_brand : ℝ := 200
def weight_first_brand : ℕ := 2
def weight_second_brand : ℕ := 3
def selling_price_per_kg : ℝ := 177
def profit_percentage : ℝ := 0.18

-- Theorem stating the problem
theorem cost_second_brand :
  let total_weight := weight_first_brand + weight_second_brand
  let total_selling_price := total_weight * selling_price_per_kg
  let cost_price := total_selling_price / (1 + profit_percentage)
  let total_cost := weight_first_brand * cost_first_brand + weight_second_brand * C
  total_cost = cost_price → 
  C = 350 / 3 :=
begin
  sorry
end

end cost_second_brand_l817_817625


namespace solve_for_b_l817_817575

theorem solve_for_b (b : ℚ) : 
  (∃ m1 m2 : ℚ, 3 * m1 - 2 * 1 + 4 = 0 ∧ 5 * m2 + b * 1 - 1 = 0 ∧ m1 * m2 = -1) → b = 15 / 2 :=
by
  sorry

end solve_for_b_l817_817575


namespace apex_on_circle_l817_817399

variables {Point : Type} [metric_space Point]

-- Given
def is_pyramid (P : Point) (A B C D : Point) : Prop := sorry

def lies_on_xy_plane (A B C D : Point) : Prop := sorry

def intersection_is_rectangle (P : Point) (A B C D : Point) : Prop := sorry

def intersection_points (A B C D : Point) : Point × Point := sorry

def collineation_axis (E F : Point) : Prop := sorry

def thales_theorem (E F M : Point) : Prop := sorry

def cutting_plane_parallel (q M : Point) : Prop := sorry

-- To prove
theorem apex_on_circle
  (P A B C D : Point)
  (h1 : is_pyramid P A B C D)
  (h2 : lies_on_xy_plane A B C D)
  (h3 : intersection_is_rectangle P A B C D) :
  ∃ E F M : Point, 
  (intersection_points A B C D = (E, F)) ∧
  (collineation_axis E F) ∧
  (thales_theorem E F M) ∧
  (cutting_plane_parallel q M) :=
sorry

end apex_on_circle_l817_817399


namespace present_age_of_A_l817_817562

theorem present_age_of_A {x : ℕ} (h₁ : ∃ (x : ℕ), 5 * x = A ∧ 3 * x = B)
                         (h₂ : ∀ (A B : ℕ), (A + 6) / (B + 6) = 7 / 5) : A = 15 :=
by sorry

end present_age_of_A_l817_817562


namespace find_fourth_vertex_l817_817746

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_parallelogram (A B C D : Point) : Prop :=
  is_midpoint ({x := 0, y := -9}) A C ∧ is_midpoint ({x := 2, y := 6}) B D ∧
  is_midpoint ({x := 4, y := 5}) C D ∧ is_midpoint ({x := 0, y := -9}) A D

theorem find_fourth_vertex :
  ∃ D : Point,
    (is_parallelogram ({x := 0, y := -9}) ({x := 2, y := 6}) ({x := 4, y := 5}) D)
    ∧ ((D = {x := 2, y := -10}) ∨ (D = {x := -2, y := -8}) ∨ (D = {x := 6, y := 20})) :=
sorry

end find_fourth_vertex_l817_817746


namespace range_of_k_l817_817513

def h (x : ℝ) : ℝ := 5 * x + 3
def k (x : ℝ) : ℝ := h(h(h(x)))

theorem range_of_k :
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 3 → -32 ≤ k(x) ∧ k(x) ≤ 468 :=
by
  intros x hx
  sorry

end range_of_k_l817_817513


namespace evaluate_expression_l817_817368

theorem evaluate_expression : (900^2 / (153^2 - 147^2)) = 450 := by
  sorry

end evaluate_expression_l817_817368


namespace increase_corrosion_with_more_active_metal_rivets_l817_817208

-- Definitions representing conditions
def corrosion_inhibitor (P : Type) : Prop := true
def more_active_metal_rivets (P : Type) : Prop := true
def less_active_metal_rivets (P : Type) : Prop := true
def painted_parts (P : Type) : Prop := true

-- Main theorem statement
theorem increase_corrosion_with_more_active_metal_rivets (P : Type) 
  (h1 : corrosion_inhibitor P)
  (h2 : more_active_metal_rivets P)
  (h3 : less_active_metal_rivets P)
  (h4 : painted_parts P) : 
  more_active_metal_rivets P :=
by {
  -- proof goes here
  sorry
}

end increase_corrosion_with_more_active_metal_rivets_l817_817208


namespace two_digit_integers_remainder_3_count_l817_817876

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817876


namespace side_length_of_triangle_ABC_l817_817507

noncomputable def side_length_of_equilateral_triangle {A B C Q : Type} (dAQ : ℝ) (dBQ : ℝ) (dCQ : ℝ) : ℝ :=
if h : (dAQ = 2 ∧ dBQ = sqrt 5 ∧ dCQ = 3) then sqrt 30 else 0

-- The proof statement with given conditions and the result to be proved.
theorem side_length_of_triangle_ABC (h : (dAQ = 2 ∧ dBQ = sqrt 5 ∧ dCQ = 3)) : 
  side_length_of_equilateral_triangle 2 (sqrt 5) 3 = sqrt 30 :=
sorry

end side_length_of_triangle_ABC_l817_817507


namespace angle_AFB_constant_l817_817403

noncomputable def a : ℝ := 2
noncomputable def e : ℝ := sqrt 3 / 2
noncomputable def c : ℝ := e * a
noncomputable def b : ℝ := sqrt (a^2 - c^2)

lemma equation_ellipse : 
  (∀ x y : ℝ, (x^2 / (a^2)) + y^2 = 1 ↔ (x^2 / 4) + y^2 = 1) := by
sorry

theorem angle_AFB_constant (F : ℝ × ℝ) (x0 y0 : ℝ) (l : ℝ × ℝ → ℝ) 
  (tangent_l_ellipse : l (x0, y0) = 0) 
  (A B: ℝ × ℝ)
  (hx1 : A.fst = 2) (hx2 : B.fst = -2) :  
  ∀ (slope : ℝ), ∠AFO = (π / 2) := by
sorry

end angle_AFB_constant_l817_817403


namespace tan_half_theta_l817_817388

theorem tan_half_theta {θ : ℝ} (h : 2 * sin θ = 1 + cos θ) :
  tan (θ / 2) = 1 / 2 ∨ ¬ (∃ y : ℝ, tan (θ / 2) = y) :=
by
  -- Proof to be filled in
  sorry

end tan_half_theta_l817_817388


namespace exam_full_marks_l817_817458

variables {A B C D F : ℝ}

theorem exam_full_marks
  (hA : A = 0.90 * B)
  (hB : B = 1.25 * C)
  (hC : C = 0.80 * D)
  (hA_val : A = 360)
  (hD : D = 0.80 * F) 
  : F = 500 :=
sorry

end exam_full_marks_l817_817458


namespace balanced_sequence_count_balanced_sequence_count_with_an_l817_817492

-- Define the balanced sequence property
def isBalancedSequence (n : ℕ) (seq : List ℕ) : Prop :=
  seq.length = n ∧ ∀ k ∈ List.range n, seq.get k = (List.take (k + 1) seq).to_finset.card

-- Prove the number of balanced sequences of length n is 2^(n-1)
theorem balanced_sequence_count (n : ℕ) (h : n > 0) :
  ∃ s, isBalancedSequence n s ∧ Nat.card (s.to_finset) = 2^(n-1) := sorry

-- Prove the number of balanced sequences of length n such that a_n = m is (n-1 choose m-1)
theorem balanced_sequence_count_with_an (n m : ℕ) (h1 : n > 0) (h2 : m > 0) :
  ∃ s, isBalancedSequence n s ∧ s.get (n - 1) = m ∧ Nat.card (s.to_finset) = Nat.choose (n-1) (m-1) := sorry

end balanced_sequence_count_balanced_sequence_count_with_an_l817_817492


namespace count_two_digit_integers_remainder_3_l817_817827

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817827


namespace amount_invested_l817_817950

variables (P y : ℝ)

-- Conditions
def condition1 : Prop := 800 = P * (2 * y) / 100
def condition2 : Prop := 820 = P * ((1 + y / 100) ^ 2 - 1)

-- The proof we seek
theorem amount_invested (h1 : condition1 P y) (h2 : condition2 P y) : P = 8000 :=
by
  -- Place the proof here
  sorry

end amount_invested_l817_817950


namespace case_n_eq_50_case_n_eq_51_l817_817593

-- Definitions based on conditions
def weights (n : ℕ) := list ℕ
def total_weight (ws : weights n) : ℕ := ws.sum

-- Problem statements
theorem case_n_eq_50 (ws : weights 50) (h1 : total_weight ws = 100) : 
  ∃ S1 S2, S1 ∪ S2 = ws ∧ S1 ≠ S2 ∧ S1.sum = 50 ∧ S2.sum = 50 → False :=
sorry

theorem case_n_eq_51 (ws : weights 51) (h1 : total_weight ws = 100) : 
  ∃ S1 S2, S1 ∪ S2 = ws ∧ S1.sum = 50 ∧ S2.sum = 50 :=
sorry

end case_n_eq_50_case_n_eq_51_l817_817593


namespace geometric_sequence_sum_5_l817_817973

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ i j : ℕ, ∃ r : ℝ, a (i + 1) = a i * r ∧ a (j + 1) = a j * r

theorem geometric_sequence_sum_5
  (a : ℕ → ℝ)
  (h : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : a 2 * a 6 + 2 * a 4 * a 5 + (a 5) ^ 2 = 25) :
  a 4 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_5_l817_817973


namespace last_digit_of_expression_l817_817054

-- Conditions
def a : ℤ := 25
def b : ℤ := -3

-- Statement to be proved
theorem last_digit_of_expression :
  (a ^ 1999 + b ^ 2002) % 10 = 4 :=
by
  -- proof would go here
  sorry

end last_digit_of_expression_l817_817054


namespace total_cost_eq_898_80_l817_817565

theorem total_cost_eq_898_80 (M R F : ℝ)
  (h1 : 10 * M = 24 * R)
  (h2 : 6 * F = 2 * R)
  (h3 : F = 21) :
  4 * M + 3 * R + 5 * F = 898.80 :=
by
  sorry

end total_cost_eq_898_80_l817_817565


namespace smallest_y_l817_817213

theorem smallest_y (y : ℤ) :
  (∃ k : ℤ, y^2 + 3*y + 7 = k*(y-2)) ↔ y = -15 :=
sorry

end smallest_y_l817_817213


namespace count_two_digit_integers_remainder_3_l817_817822

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817822


namespace solve_for_x_l817_817551

-- Define the function representing the original equation
def original_equation (x : ℝ) : Prop :=
  log 3 ((4 * x + 12) / (6 * x - 4)) + log 3 ((6 * x - 4) / (2 * x - 3)) = 2

-- The principal statement to be proved in Lean 4
theorem solve_for_x : original_equation (39 / 14) :=
by
  -- Here we'd provide the proof of the statement
  sorry

end solve_for_x_l817_817551


namespace gel_pen_ratio_l817_817310

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l817_817310


namespace sin_cos_identity_l817_817738

theorem sin_cos_identity (x : ℝ) : 
  sin x ^ 6 + cos x ^ 6 + sin x ^ 2 = 2 * sin x ^ 4 + cos x ^ 4 := 
sorry

end sin_cos_identity_l817_817738


namespace perimeter_of_square_field_l817_817187

-- Given conditions
def num_posts : ℕ := 36
def post_width_inch : ℝ := 6
def gap_length_feet : ℝ := 8

-- Derived conditions
def posts_per_side : ℕ := num_posts / 4
def gaps_per_side : ℕ := posts_per_side - 1
def total_gap_length_per_side : ℝ := gaps_per_side * gap_length_feet
def post_width_feet : ℝ := post_width_inch / 12
def total_post_width_per_side : ℝ := posts_per_side * post_width_feet
def side_length : ℝ := total_gap_length_per_side + total_post_width_per_side

-- Goal: The perimeter of the square field
theorem perimeter_of_square_field : 4 * side_length = 242 := by
  sorry

end perimeter_of_square_field_l817_817187


namespace gel_pen_price_ratio_l817_817300

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l817_817300


namespace tournament_participants_l817_817965

theorem tournament_participants (n : ℕ) (h₁ : 2 * (n * (n - 1) / 2 + 4) - (n - 2) * (n - 3) - 16 = 124) : n = 13 :=
sorry

end tournament_participants_l817_817965


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817886

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817886


namespace count_two_digit_remainders_l817_817799

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817799


namespace sum_of_angles_l817_817658

-- Define the circle with a radius of 10 meters
structure Circle :=
  (radius : ℝ)

-- Define the path with total distance covered
structure Path :=
  (total_distance : ℝ)
  (angles : list ℝ)

-- Define the main theorem to prove the sum of the angles is at least 2998 radians
theorem sum_of_angles {arena : Circle} {lion_path : Path}
  (h1 : arena.radius = 10) 
  (h2 : lion_path.total_distance = 30000) 
  (h3 : ∀ θ ∈ lion_path.angles, 0 ≤ θ) : 
  lion_path.angles.sum ≥ 2998 :=
by
  sorry

end sum_of_angles_l817_817658


namespace Joan_spent_68_353_on_clothing_l817_817485

theorem Joan_spent_68_353_on_clothing :
  let shorts := 15.00
  let jacket := 14.82 * 0.9
  let shirt := 12.51 * 0.5
  let shoes := 21.67 - 3
  let hat := 8.75
  let belt := 6.34
  shorts + jacket + shirt + shoes + hat + belt = 68.353 :=
sorry

end Joan_spent_68_353_on_clothing_l817_817485


namespace tangent_line_exp_l817_817018

theorem tangent_line_exp (k : ℝ) : (∃ (x₀ : ℝ), y = k * x₀ ∧ y = e ^ x₀) →
  k = real.exp 1 :=
sorry

end tangent_line_exp_l817_817018


namespace count_interesting_numbers_l817_817834

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817834


namespace logarithmic_inequality_solution_l817_817621

theorem logarithmic_inequality_solution :
  {x : ℝ | 8.24 * log 3 ((3 * x - 5) / (x + 1)) ≤ 1} = {x : ℝ | x > 5/3} :=
by
  sorry

end logarithmic_inequality_solution_l817_817621


namespace crease_length_l817_817660

noncomputable def fold_creased_length (a b c : ℝ) : ℝ :=
  if h : a^2 + b^2 = c^2 then
    let midpoint := c / 2
    let crease := Real.sqrt (a^2 + midpoint^2)
    crease
  else 0

theorem crease_length (a b c : ℝ) (h : a = 5) (hb : b = 12) (hc : c = 13) (hypotenuse_cond : a^2 + b^2 = c^2) :
  fold_creased_length a b c = 8.2 :=
by
  rw [h, hb, hc]
  have hypotenuse_cond' : 5^2 + 12^2 = 13^2 := by norm_num
  rw fold_creased_length
  split_ifs
  · norm_num
  · contradiction

end crease_length_l817_817660


namespace find_triples_l817_817518

noncomputable def polynomial : (ℂ → ℂ → ℂ → (ℂ → ℂ)) :=
  λ a b c, λ x, x^4 - a * x^3 - b * x + c

theorem find_triples (a b c : ℂ) :
  (∀ (x : ℂ), polynomial a b c x = 0 → (x = a ∨ x = b ∨ x = c ∨ ∃ d, d = -(b + c)))
  →
  -- Possible sets of values for (a, b, c):
  (a, b, c) = (a, 0, 0) ∨
  (a, b, c) = (\(-1 + complex.I * complex.sqrt 3) / 2, 1, (-1 + complex.I * complex.sqrt 3) / 2) ∨
  (a, b, c) = (\(-1 - complex.I * complex.sqrt 3) / 2, 1, (-1 - complex.I * complex.sqrt 3) / 2) ∨
  (a, b, c) = (1 - complex.I * complex.sqrt 3 / 2, -1, 1 + complex.sqrt 3 / 2) ∨
  (a, b, c) = (1 + complex.I * complex.sqrt 3 / 2, -1, 1 - complex.sqrt 3 / 2) :=
sorry

end find_triples_l817_817518


namespace desktop_revenue_l817_817529

def total_computers_sold_per_week : ℕ := 72

def week1_sales_distribution : {laptops : ℚ // laptops = 1/2} ∧ {netbooks : ℚ // netbooks = 1/3} ∧ {desktops : ℚ // desktops = 1 - (1/2 + 1/3)}
def week2_sales_distribution : {laptops : ℚ // laptops = 0.4} ∧ {netbooks : ℚ // netbooks = 0.2} ∧ {desktops : ℚ // desktops = 1 - (0.4 + 0.2)}
def week3_sales_distribution : {laptops : ℚ // laptops = 0.3} ∧ {netbooks : ℚ // netbooks = 0.5} ∧ {desktops : ℚ // desktops = 1 - (0.3 + 0.5)}
def week4_sales_distribution : {laptops : ℚ // laptops = 0.1} ∧ {netbooks : ℚ // netbooks = 0.25} ∧ {desktops : ℚ // desktops = 1 - (0.1 + 0.25)}

def desktop_price : ℚ := 1000

theorem desktop_revenue (w1 w2 w3 w4 : {d : ℚ // 0 ≤ d ∧ d ≤ 72}) 
  (hw1 : w1.val = total_computers_sold_per_week * (1 - (1/2 + 1/3)))
  (hw2 : w2.val = total_computers_sold_per_week * (1 - (0.4 + 0.2)))
  (hw3 : w3.val = total_computers_sold_per_week * (1 - (0.3 + 0.5)))
  (hw4 : w4.val = total_computers_sold_per_week * (1 - (0.1 + 0.25))) :
  w1.val * desktop_price + w2.val * desktop_price + w3.val * desktop_price + w4.val * desktop_price = 102000 := 
sorry

end desktop_revenue_l817_817529


namespace pen_price_ratio_l817_817321

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l817_817321


namespace volume_shaded_part_rotated_l817_817288

noncomputable def volume_of_solid_rotated_around_CD (BC AB : ℝ) (pi_val : ℝ) : ℝ :=
  let r := BC / 2
  let volume_cone := (1 / 3) * pi_val * (r ^ 2) * AB
  2 * volume_cone

theorem volume_shaded_part_rotated (BC AB : ℝ) (pi_val : ℝ) :
  BC = 6 → AB = 10 → pi_val = 3.14 → 
  volume_of_solid_rotated_around_CD BC AB pi_val = 188.4 :=
by
  intros hBC hAB hPi
  rw [hBC, hAB, hPi]
  have r := 6 / 2 
  have volume_cone := (1 / 3) * 3.14 * (r ^ 2) * 10
  have volume := 2 * volume_cone
  sorry

end volume_shaded_part_rotated_l817_817288


namespace correctness_of_statements_l817_817028

noncomputable def f (x : ℝ) : ℝ := real.exp x - real.exp (-x)

theorem correctness_of_statements :
  (∀ x, f (-x) = -f x) ∧ 
  (∃ x, f x = x^2 + 2*x → false) ∧ 
  (∀ x, 0 < f x) ∧ 
  (∀ (x : ℝ), 0 < x → f x > 2 * x) :=
by
  sorry

end correctness_of_statements_l817_817028


namespace ticket_sales_revenue_l817_817673

theorem ticket_sales_revenue (total_tickets advance_tickets same_day_tickets price_advance price_same_day: ℕ) 
    (h1: total_tickets = 60) 
    (h2: price_advance = 20) 
    (h3: price_same_day = 30) 
    (h4: advance_tickets = 20) 
    (h5: same_day_tickets = total_tickets - advance_tickets):
    advance_tickets * price_advance + same_day_tickets * price_same_day = 1600 := 
by
  sorry

end ticket_sales_revenue_l817_817673


namespace sum_a_2b_eq_1_l817_817509

-- Given conditions
variables (a b : ℝ) (h : {0, b, b / a} = {1, a, a + b})

-- Theorem statement: Prove a + 2b = 1
theorem sum_a_2b_eq_1 (ha : a ≠ 0) : a + 2b = 1 :=
sorry

end sum_a_2b_eq_1_l817_817509


namespace pizza_longest_segment_squared_l817_817269

theorem pizza_longest_segment_squared :
  ∀ (d : ℝ) (n : ℕ) (m : ℝ), d = 16 → n = 4 → m = 8 * Real.sqrt 2 → m^2 = 128 :=
by
  intros d n m h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  -- Continue the proof or add sorry to skip the proof
  sorry

end pizza_longest_segment_squared_l817_817269


namespace gel_pen_price_ratio_l817_817302

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l817_817302


namespace passing_train_speed_l817_817670

noncomputable def speed_of_passing_train (v_p : ℝ) (t : ℝ) (L : ℝ) : ℝ :=
  let relative_speed := (L / 1000) / (t / 3600) in
  relative_speed - v_p

theorem passing_train_speed :
  let v_p := 40 -- passenger's train speed in km/h
  let t := 3 / 3600 -- time in hours
  let L := 75 / 1000 -- length in km
  speed_of_passing_train v_p t 0.075 = 50 :=
by
  sorry

end passing_train_speed_l817_817670


namespace raking_yard_time_l817_817473

theorem raking_yard_time (your_rate : ℚ) (brother_rate : ℚ) (combined_rate : ℚ) (combined_time : ℚ) :
  your_rate = 1 / 30 ∧ 
  brother_rate = 1 / 45 ∧ 
  combined_rate = your_rate + brother_rate ∧ 
  combined_time = 1 / combined_rate → 
  combined_time = 18 := 
by 
  sorry

end raking_yard_time_l817_817473


namespace dart_distribution_count_l817_817687
-- Importing the full Mathlib library ensures all necessary definitions and theorems are available.

-- Defining the equivalent proof problem within Lean 4.
theorem dart_distribution_count :
  (list_of_partitions 4 5).length = 5 :=
sorry

end dart_distribution_count_l817_817687


namespace fish_tank_ratio_l817_817559

theorem fish_tank_ratio :
  ∀ (F1 F2 F3: ℕ),
  F1 = 15 →
  F3 = 10 →
  (F3 = (1 / 3 * F2)) →
  F2 / F1 = 2 :=
by
  intros F1 F2 F3 hF1 hF3 hF2
  sorry

end fish_tank_ratio_l817_817559


namespace maximize_binomial_probability_l817_817960

open Nat

theorem maximize_binomial_probability :
  ∀ k : ℕ, (k ≤ 5) ∧
  (P_formula k = (choose 5 k) * (1/4 : ℚ)^k * (3/4 : ℚ)^(5 - k)) →
  P_formula 1 = max (P_formula 0)
    (max (P_formula 1)
      (max (P_formula 2)
        (max (P_formula 3)
          (max (P_formula 4) (P_formula 5))))) :=
  by -- structure and conditions of the proof go here
  sorry

def P_formula (k : ℕ) : ℚ :=
  if k ≤ 5 then (choose 5 k) * (1/4 : ℚ)^k * (3/4 : ℚ)^(5 - k)
  else 0

end maximize_binomial_probability_l817_817960


namespace coffeeOrderTotalIs25_l817_817477

noncomputable def totalCost : ℝ :=
  let costDripCoffee := 2 * 2.25
  let costLattes := 2 * 4.00
  let costVanilla := 0.50
  let costColdBrew := 2 * 2.50
  let costDoubleEspresso := 3.50
  let costCappuccino := 3.50
  costDripCoffee + costLattes + costVanilla + costColdBrew + costDoubleEspresso + costCappuccino

theorem coffeeOrderTotalIs25 : totalCost = 25 := by
  intro
  unfold totalCost
  sorry

end coffeeOrderTotalIs25_l817_817477


namespace find_equation_of_line_l817_817656

-- Given the conditions of the problem.
def passes_point (P: Point ℝ) (l: Line ℝ) : Prop :=
  l contains P

def equal_intercepts (l: Line ℝ) : Prop :=
  ∃ a: ℝ, l = { p | p.x + p.y = a }

-- Statement of the problem (no proof required, just the form of the statement)
theorem find_equation_of_line (P : Point ℝ) (l : Line ℝ)
  (h1 : P = (-3, -2))
  (h2 : passes_point P l)
  (h3 : equal_intercepts l) :
  l = { p | p.x + p.y = -5 } ∨ l = {p | 2 * p.x - 3 * p.y = 0 } :=
sorry

end find_equation_of_line_l817_817656


namespace hexagon_diagonals_concurrent_l817_817992

   -- Define the condition that the hexagon ABCDEF is convex and has all equal sides
   variables (A B C D E F : Type*)
   [hex : convex_hexagon_with_equal_sides A B C D E F] 

   -- Define the condition on the angles in the hexagon
   axiom angles_sum_condition : (∠A + ∠C + ∠E = ∠B + ∠D + ∠F)

   -- Problem to prove: AD, BE, and CF are concurrent
   theorem hexagon_diagonals_concurrent (h : convex_hexagon_with_equal_sides A B C D E F) 
       (h_angle_sum : ∠A + ∠C + ∠E = ∠B + ∠D + ∠F) : 
       concurrent AD BE CF := 
   by 
       sorry
   
end hexagon_diagonals_concurrent_l817_817992


namespace no_prime_p_gt_7_with_few_divisors_l817_817141

-- Define the notion of a prime number.
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the divisor count function.
def num_divisors (n : ℕ) : ℕ := (list.range (n + 1)).count (λ d, n % d = 0)

theorem no_prime_p_gt_7_with_few_divisors (p : ℕ) (hp_prime : is_prime p) (hp_gt_7 : p > 7) :
    ¬ (num_divisors (p^12 + 5039 * 5041) < 120) := sorry

end no_prime_p_gt_7_with_few_divisors_l817_817141


namespace count_whole_numbers_in_interval_l817_817047

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.exp 1
  let lower_bound := Int.ceil a
  let upper_bound := Int.floor b
  (upper_bound - lower_bound + 1) = 7 := by
  let a := 7 / 4
  let b := 3 * Real.exp 1
  let lower_bound := Int.ceil a
  let upper_bound := Int.floor b
  sorry

end count_whole_numbers_in_interval_l817_817047


namespace max_days_proof_l817_817162

-- Define a graph with n vertices and bidirectional edges
structure Graph (V : Type) :=
  (adj : V → V → Prop)
  (sym : ∀ {u v : V}, adj u v → adj v u)

-- Define the problem conditions
def airport_problem (V : Type) (n : ℕ) [finite V] [fintype V] (G : Graph V) : Prop :=
  n = fintype.card V ∧
  n ≥ 3 ∧
  (∃ D, ∀ v : V, D v = card (finset.filter (G.adj v) (finset.univ : finset V))) ∧
  (∀ (t : ℕ), t < n - 3 → ∃ (v : V), D v = max (λ v : V, D v))

-- Define the maximum number of days for each n
def max_days (n : ℕ) : ℕ :=
  if n = 3 then 1 else n - 3

-- Lean theorem stating the equivalence of condition and answer
theorem max_days_proof (V : Type) (n : ℕ) [finite V] [fintype V] (G : Graph V)
  (cond : airport_problem V n G) : 
  cond → max_days n = (if n = 3 then 1 else n - 3) :=
sorry

end max_days_proof_l817_817162


namespace abc_value_l817_817052

theorem abc_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (hab : a * b = 30 * real.sqrt 5) (hac : a * c = 45 * real.sqrt 5)
  (hbc : b * c = 40 * real.sqrt 5) : 
  a * b * c = 300 * real.sqrt 3 * real.sqrt (real.sqrt 5) :=
by
  sorry

end abc_value_l817_817052


namespace meal_combinations_l817_817227

def number_of_menu_items : ℕ := 15

theorem meal_combinations (different_orderings : ∀ Yann Camille : ℕ, Yann ≠ Camille → Yann ≤ number_of_menu_items ∧ Camille ≤ number_of_menu_items) : 
  (number_of_menu_items * (number_of_menu_items - 1)) = 210 :=
by sorry

end meal_combinations_l817_817227


namespace interval_of_monotonic_increase_l817_817360

def f : ℝ → ℝ := λ x, abs (2^(-x) - 2)

theorem interval_of_monotonic_increase : ∀ x1 x2 : ℝ, (x1 > -1 → x2 > -1) → (x1 < x2 → f x1 < f x2) := sorry

end interval_of_monotonic_increase_l817_817360


namespace remainder_of_1234567_div_123_l817_817347

theorem remainder_of_1234567_div_123 : 1234567 % 123 = 129 :=
by
  sorry

end remainder_of_1234567_div_123_l817_817347


namespace two_digit_integers_remainder_3_count_l817_817878

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817878


namespace op_star_algebraic_l817_817701

def op_star (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3 * (op_star (n - 1))

theorem op_star_algebraic (n : ℕ) (h : n > 0) : 
  op_star n = 3^(n-1) :=
by
  sorry

end op_star_algebraic_l817_817701


namespace horner_eval_v3_at_minus4_l817_817605

def f (x : ℤ) : ℤ := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

def horner_form (x : ℤ) : ℤ :=
  let a6 := 3
  let a5 := 5
  let a4 := 6
  let a3 := 79
  let a2 := -8
  let a1 := 35
  let a0 := 12
  let v := a6
  let v1 := v * x + a5
  let v2 := v1 * x + a4
  let v3 := v2 * x + a3
  let v4 := v3 * x + a2
  let v5 := v4 * x + a1
  let v6 := v5 * x + a0
  v3

theorem horner_eval_v3_at_minus4 :
  horner_form (-4) = -57 :=
by
  sorry

end horner_eval_v3_at_minus4_l817_817605


namespace length_of_train_is_90m_l817_817231

-- Define the speed of the train in km/hr
def speed_km_per_hr := 36

-- Define the time it takes to cross a pole in seconds
def time_seconds := 9

-- Convert speed from km/hr to m/s
def speed_m_per_s := speed_km_per_hr * (1000 / 3600)

-- Define the length of the train
def length_of_train := speed_m_per_s * time_seconds

-- Provide the theorem statement
theorem length_of_train_is_90m : length_of_train = 90 := by
  calc
    length_of_train = speed_m_per_s * time_seconds      : rfl
               ...  = (36 * (1000 / 3600)) * 9          : by sorry
               ...  = 10 * 9                            : by sorry
               ...  = 90                                : rfl

end length_of_train_is_90m_l817_817231


namespace angle_between_vectors_l817_817059

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variable (h1 : inner (a - 4 • b) a = 0)
variable (h2 : inner (b - a) b = 0)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)

theorem angle_between_vectors (ha : a ≠ 0) (hb : b ≠ 0) (h1 : inner (a - 4 • b) a = 0) (h2 : inner (b - a) b = 0) :
  real.angle a b = real.pi / 3 :=
sorry

end angle_between_vectors_l817_817059


namespace num_rem_three_by_seven_l817_817854

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817854


namespace find_number_l817_817946

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end find_number_l817_817946


namespace part1_l817_817420

theorem part1 (a : ℝ) : 
  (∀ x ∈ Set.Ici (1/2 : ℝ), 2 * x + a / (x + 1) ≥ 0) → a ≥ -3 / 2 :=
sorry

end part1_l817_817420


namespace k_h_5_eq_148_l817_817929

def h (x : ℤ) : ℤ := 4 * x + 6
def k (x : ℤ) : ℤ := 6 * x - 8

theorem k_h_5_eq_148 : k (h 5) = 148 := by
  sorry

end k_h_5_eq_148_l817_817929


namespace probability_divisor_of_12_on_8_sided_die_l817_817677

theorem probability_divisor_of_12_on_8_sided_die :
  (∃ (num favorable: ℕ), favorable = multiset.card { x | x ∣ 12 ∧ x ∈ {1, 2, 3, 4, 5, 6, 7, 8} } ∧ ∑ x in {1, 2, 3, 4, 5, 6, 7, 8}, 1 = 8 ∧ favorable = 5) →
  (5 / 8 : ℚ) = 5 / 8 :=
begin
  sorry
end

end probability_divisor_of_12_on_8_sided_die_l817_817677


namespace square_root_of_expression_l817_817016

-- Define the initial conditions
variables (a b c : ℕ)

-- Provide the definition from the conditions
def condition1 := (5 * a + 2) = 27
def condition2 := (3 * a + b - 1) = 16
def condition3 := c = Int.floor (Real.sqrt 13)

-- Prove the final statement
theorem square_root_of_expression :
  condition1 →
  condition2 →
  condition3 →
  Real.sqrt (3 * a - b + c) = 4 ∨ Real.sqrt (3 * a - b + c) = -4 :=
by
  intros h1 h2 h3
  have h_a : a = 5 := by sorry
  have h_b : b = 2 := by sorry
  have h_c : c = 3 := by sorry
  rw [h_a, h_b, h_c]
  have : 3 * 5 - 2 + 3 = 16 := by norm_num
  rw this
  norm_num
  right
  norm_num
  sorry

end square_root_of_expression_l817_817016


namespace shopkeeper_secret_code_l817_817631

theorem shopkeeper_secret_code (digit_mapping : Char → ℕ)
  (h_distinct : ∀ x y : Char, x ≠ y → digit_mapping x ≠ digit_mapping y)
  (h_keyword : ∀ c, c ∈ "ПОДСВЕЧНИК".to_list → digit_mapping c ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 0})
  (h_length : "ПОДСВЕЧНИК".to_list.length = 10):
  digit_mapping 'Д' = 2 ∧ digit_mapping 'Е' = 0 ∧ digit_mapping 'С' = 3 ∧
  digit_mapping 'К' = 9 ∧ digit_mapping 'Ч' = 8 ∧ digit_mapping 'И' = 4 ∧
  digit_mapping 'Н' = 6 ∧ digit_mapping 'В' = 7 ∧ digit_mapping 'О' = 5 ∧
  digit_mapping 'П' = 1 :=
by
  sorry

end shopkeeper_secret_code_l817_817631


namespace ratio_pressures_l817_817962

-- Define the conditions as given in the problem
def ratio_areas := (3, 2, 1)
def pressure (F : ℝ) (S : ℝ) := F / S

-- Assume the areas of the faces
def area_A := 3 * a
def area_B := 2 * a
def area_C := a

-- Define the pressures based on the given areas and force F
def P_A := pressure F area_A
def P_B := pressure F area_B
def P_C := pressure F area_C

-- The hypothesis stating the given ratio of areas
def hyp_areas_ratio := ratio_areas = (3, 2, 1)

-- The goal is to prove the ratio of the pressures
theorem ratio_pressures (a F : ℝ) (h : hyp_areas_ratio) : 
  (P_A / P_B / P_C) = (2 / 3 / 6) := by
  sorry  -- Proof to be provided

end ratio_pressures_l817_817962


namespace f_f_3_eq_neg_3_l817_817773

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 3 then 2^x - 1 else x - 5

theorem f_f_3_eq_neg_3 
  (h : is_odd_function f) : 
  f (f 3) = -3 :=
by
  sorry

end f_f_3_eq_neg_3_l817_817773


namespace num_rem_three_by_seven_l817_817861

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817861


namespace number_multiplied_by_six_l817_817616

theorem number_multiplied_by_six (n : ℕ) (h : n / 11 = 2) : n * 6 = 132 := by
  have hn : n = 2 * 11 := sorry
  rw [hn]
  norm_num

end number_multiplied_by_six_l817_817616


namespace remainder_mod_7_l817_817375

def A : ℕ := ∑ i in (Finset.range 10).map (λ i, 10^(10^i))

theorem remainder_mod_7 : A % 7 = 5 :=
by
  sorry

end remainder_mod_7_l817_817375


namespace fraction_of_remaining_paint_used_l817_817987

-- Define the initial amount of paint
def initial_paint : ℝ := 360

-- Define the fraction of paint used in the first week
def first_week_usage_fraction : ℝ := 1 / 4

-- Define the total paint used
def total_paint_used : ℝ := 128.57

-- Define the amount of paint used in the first week
def first_week_usage := initial_paint * first_week_usage_fraction

-- Define the remaining paint after the first week
def remaining_paint := initial_paint - first_week_usage

-- Define the fraction of remaining paint used in the second week
def second_week_usage_fraction := total_paint_used / remaining_paint

-- State the theorem
theorem fraction_of_remaining_paint_used : second_week_usage_fraction = 119/250 :=
by
  sorry

end fraction_of_remaining_paint_used_l817_817987


namespace gel_pen_price_ratio_l817_817299

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l817_817299


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817911

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817911


namespace min_rooms_l817_817207

theorem min_rooms (n : ℕ) (doors_internal : ℕ) (doors_external : ℕ)
  (h1 : ∀ x y, x ≠ y → doors_internal ≤ (n * (n - 1) / 2))
  (h2 : doors_external ≤ n)
  (h3 : doors_internal + doors_external = 12) :
  n ≥ 5 :=
by
  sorry

end min_rooms_l817_817207


namespace binomial_expansion_solution_l817_817413

theorem binomial_expansion_solution
    (n : ℕ)
    (n_pos : 0 < n)
    (h_ratio : binom n 1 / binom n 2 = 2 / 5) :
    n = 6 ∧
    term_with_x3 n = 240 * x^3 ∧
    2^6 * (binom 6 0) + 2^5 * (binom 6 1) + 2^4 * (binom 6 2) + 2^3 * (binom 6 3) + 2^2 * (binom 6 4) + 2 * (binom 6 5) = 728 :=
by
  sorry

end binomial_expansion_solution_l817_817413


namespace amount_lent_to_B_l817_817655

variable (P_B : ℝ) (P_C : ℝ) (r : ℝ) (t_B : ℝ) (t_C : ℝ) (total_interest : ℝ)

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  (P * r * t) / 100

theorem amount_lent_to_B (h1 : P_C = 3000)
                        (h2 : r = 8)
                        (h3 : t_B = 2)
                        (h4 : t_C = 4)
                        (h5 : total_interest = 1760)
                        (h6 : simple_interest P_C r t_C = 960) :
  simple_interest P_B r t_B + 960 = 1760 → P_B = 5000 :=
by
  intro h
  have h7 : 960 + simple_interest P_B r t_B = 1760 := by sorry
  have h8 : simple_interest P_B r t_B = 800 := by sorry
  have h9 : (P_B * r * t_B) / 100 = 800 := by sorry
  have h10 : P_B * 16 / 100 = 800 := by sorry
  have h11 : P_B * 16 = 80000 := by sorry
  have h12 : P_B = 5000 := by sorry
  exact h12

end amount_lent_to_B_l817_817655


namespace triangle_obtuse_of_cos_relation_l817_817009

theorem triangle_obtuse_of_cos_relation
  (a b c : ℝ)
  (A B C : ℝ)
  (hTriangle : A + B + C = Real.pi)
  (hSides : a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
  (hSides' : b^2 = a^2 + c^2 - 2*a*c*Real.cos B)
  (hSides'' : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)
  (hRelation : a * Real.cos C = b + 2/3 * c) :
 ∃ (A' : ℝ), A' = A ∧ A > (Real.pi / 2) := 
sorry

end triangle_obtuse_of_cos_relation_l817_817009


namespace count_two_digit_integers_remainder_3_div_7_l817_817897

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817897


namespace train_length_is_199_95_l817_817669

noncomputable def convert_speed_to_m_s (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000) / 3600

noncomputable def length_of_train (bridge_length : ℝ) (time_seconds : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := convert_speed_to_m_s speed_kmh
  speed_ms * time_seconds - bridge_length

theorem train_length_is_199_95 :
  length_of_train 300 45 40 = 199.95 := by
  sorry

end train_length_is_199_95_l817_817669


namespace problem1_problem2_l817_817638

-- Define the trigonometric functions involved
def trig_ex1 : ℝ := sin (270 * (π / 180)) + tan (765 * (π / 180)) + tan (225 * (π / 180)) + cos (240 * (π / 180))

theorem problem1 : trig_ex1 = 1 / 2 :=
by
  sorry

variables (α : ℝ)

-- Define the trigonometric functions involved for the second problem
def trig_ex2 : ℝ := 
  (- sin (180 * (π / 180) + α) + sin (- α) - tan (360 * (π / 180) + α)) /
  (tan (α + 180 * (π / 180)) + cos (- α) + cos (180 * (π / 180) - α))

theorem problem2 (α : ℝ) : trig_ex2 α = -1 :=
by
  sorry

end problem1_problem2_l817_817638


namespace volume_of_solid_l817_817082

variable (r α : ℝ)
variable (V : ℝ)

theorem volume_of_solid :
  let sin_half_alpha := Real.sin (α / 2) in
  V = (sin_half_alpha^2 / Real.sqrt 3) * (3 - 4 * sin_half_alpha^2) :=
sorry

end volume_of_solid_l817_817082


namespace distinguishable_arrangements_l817_817793

theorem distinguishable_arrangements :
  let brown := 1
  let purple := 2
  let green := 3
  let yellow := 3
  let total := brown + purple + green + yellow
  total = 9 →
  (Nat.factorial total) / ((Nat.factorial brown) * (Nat.factorial purple) * (Nat.factorial green) * (Nat.factorial yellow)) = 5040 :=
by
  intros brown purple green yellow total h
  rw [Nat.factorial, Nat.factorial_div,
      Nat.factorial_mul_factorial, h]
  simp only [Nat.factorial_succ, Nat.factorial]
  sorry

end distinguishable_arrangements_l817_817793


namespace cory_fruit_schedule_l817_817700

theorem cory_fruit_schedule :
  -- Define the total number of distinct ways Cory can eat his fruits
  ∃ (num_ways : ℕ),
    -- Cory has 4 apples, 3 oranges, 3 bananas, and 4 grapes (14 pieces of fruit in total)
    let apples := 4,
        oranges := 3,
        bananas := 3,
        grapes := 4,
        total_fruit := apples + oranges + bananas + grapes,
        total_meals := 14 in
    -- Number of distinct ways Cory can consume the fruits such that he eats two fruits a day for 7 days
    total_fruit = total_meals ∧
    num_ways = nat.factorial total_meals / (nat.factorial apples * nat.factorial oranges * nat.factorial bananas * nat.factorial grapes) ∧
    num_ways = 4204200 :=
begin
  -- Declare that such a number exists
  use 4204200,
  -- Define the values
  let apples := 4,
  let oranges := 3,
  let bananas := 3,
  let grapes := 4,
  let total_fruit := apples + oranges + bananas + grapes,
  let total_meals := 14,
  
  -- Show the conditions
  split,
  { -- Show that the total fruit equals the total required meals
    exact (rfl : total_fruit = total_meals),
  },
  split,
  { -- Show that the number of ways to arrange the fruits is what we calculated
    exact (show 14! / (4! * 3! * 3! * 4!) = 4204200, by sorry),
  },
  -- Finally, declare the result equals 4204200
  exact rfl,
end

end cory_fruit_schedule_l817_817700


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817894

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817894


namespace count_two_digit_remainders_l817_817800

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817800


namespace brocard_circumradius_product_l817_817108

theorem brocard_circumradius_product (ABC : Triangle) (P : Point) 
    (hP : isBrocardPoint ABC P) 
    (R1 R2 R3 : ℝ) (hR1 : circumradius (triangle.mk ABC.a ABC.b P) = R1)
    (hR2 : circumradius (triangle.mk ABC.b ABC.c P) = R2)
    (hR3 : circumradius (triangle.mk ABC.c ABC.a P) = R3)
    (R : ℝ) (hR : circumradius ABC = R) :
    R1 * R2 * R3 = R^3 :=
by sorry

end brocard_circumradius_product_l817_817108


namespace arrange_circles_in_rectangle_l817_817138

/-- Prove that it is possible to arrange non-overlapping circles 
in a rectangle of area 1 such that the sum of their radii equals 100. -/
theorem arrange_circles_in_rectangle (a b : ℝ) (h : a * b = 1) : 
  ∃ radii : ℝ → ℝ, (∀ x, radii x ≥ 0) ∧ (∑' x, radii x = 100) := 
begin
  sorry
end

end arrange_circles_in_rectangle_l817_817138


namespace intersection_point_of_circumcircles_l817_817494

variables {A B C D O₁ O₂ : ℝ}
variables [convex_quadrilateral : A B C D]
variables (h1 : dist A D = dist B C)
variables (h2 : angle B A C + angle D C A = 180)
variables (h3 : angle B A C ≠ 90)
noncomputable def O₁ := circumcenter B C A
noncomputable def O₂ := circumcenter C A D

theorem intersection_point_of_circumcircles (h₄ : convex_quadrilateral A B C D)
  (h₁ : dist A D = dist B C)
  (h₂ : angle B A C + angle D C A = 180)
  (h₃ : angle B A C ≠ 90) :
  ∃ T, T ∈ circumcircle O₁ B C ∧ T ∈ circumcircle O₂ A D ∧ T ∈ line A C :=
begin
  sorry
end

end intersection_point_of_circumcircles_l817_817494


namespace ratio_sunday_to_weekday_paper_l817_817098

noncomputable def weight_monday_saturday : ℕ := 8
noncomputable def papers_per_day : ℕ := 250
noncomputable def weeks : ℕ := 10
noncomputable def revenue_dollars : ℕ := 100
noncomputable def ton_in_pounds : ℕ := 2000
noncomputable def pound_in_ounces : ℕ := 16
noncomputable def total_sundays : ℕ := weeks
noncomputable def one_ton_paper_in_ounces : ℕ := ton_in_pounds * pound_in_ounces

theorem ratio_sunday_to_weekday_paper :
  let total_weekday_papers := papers_per_day * 6 * weeks in
  let weight_weekday_papers_ounces := total_weekday_papers * weight_monday_saturday in
  let weight_weekday_papers_pounds := weight_weekday_papers_ounces / pound_in_ounces in
  let weight_weekday_papers_tons := weight_weekday_papers_pounds / ton_in_pounds in
  let total_sunday_papers := papers_per_day * total_sundays in
  let weight_sunday_papers_ounces := one_ton_paper_in_ounces in
  let weight_sunday_paper := weight_sunday_papers_ounces / total_sunday_papers in
  weight_sunday_paper / (weight_monday_saturday : ℝ) = 1.6 :=
by
  sorry

end ratio_sunday_to_weekday_paper_l817_817098


namespace count_two_digit_remainders_l817_817804

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817804


namespace compare_areas_l817_817166

def area_of_sector (r : ℝ) (θ : ℝ) : ℝ :=
  θ * r^2 / 2

noncomputable def area_dotted_segment : ℝ := sorry -- Assume the function exists

noncomputable def area_figure1 (r : ℝ) : ℝ :=
  area_of_sector r (π / 3) - area_dotted_segment

noncomputable def area_triangular_shapes (r : ℝ) : ℝ := sorry -- Assume area of triangular shapes

noncomputable def area_small_segments (r : ℝ) : ℝ := sorry -- Assume area of small segments

noncomputable def area_figure2 (r : ℝ) : ℝ :=
  area_triangular_shapes r - area_small_segments r

theorem compare_areas (r : ℝ) (A1 A2 : ℝ) (h1 : A1 = area_figure1 r) (h2 : A2 = area_figure2 r) :
  A1 > A2 := 
sorry -- Proof required

end compare_areas_l817_817166


namespace find_square_digit_l817_817152

def is_multiple_of_6 (n : ℕ) : Prop :=
  n % 6 = 0

def is_digit (d : ℕ) : Prop :=
  d < 10

def even_digit (d : ℕ) : Prop :=
  d % 2 = 0

noncomputable def sum_of_digits (n : ℕ) (d : ℕ) :=
  5 + 2 + 2 + 8 + d

theorem find_square_digit :
  ∃ (square : ℕ), is_digit square ∧ is_multiple_of_6 (52 * 1000 + 28 * 10 + square)
:=
begin
  use 4,
  split,
  { exact dec_trivial, -- 4 is a digit
  { rw nat.add_mul_mod_self_left,
    exact dec_trivial, -- Proof irrespective, as 52184 % 6 == 0 holds
    sorry }
end

end find_square_digit_l817_817152


namespace problem_statement_l817_817383

noncomputable def quadrilateral_is_rectangle (OA OB OC OD : ℝ) (O A B C D : ℝ) : Prop :=
  OA = OB ∧ OA = OC ∧ OA = OD ∧
  (OA + OB + OC + OD = 0) ∧
  (∀ r, OA = r ∧ OB = r ∧ OC = r ∧ OD = r) →
  is_rectangle A B C D

theorem problem_statement (OA OB OC OD : ℝ) (O A B C D: ℝ) (h1 : OA = OB)
  (h2 : OA = OC) (h3 : OA = OD) (h4 : OA + OB + OC + OD = 0)
  (h5 : ∀ r : ℝ, OA = r ∧ OB = r ∧ OC = r ∧ OD = r) : quadrilateral_is_rectangle OA OB OC OD O A B C D :=
begin
  sorry
end

end problem_statement_l817_817383


namespace count_two_digit_integers_remainder_3_l817_817871

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817871


namespace two_digit_integers_remainder_3_count_l817_817882

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817882


namespace intersecting_circles_CD_squared_l817_817603

theorem intersecting_circles_CD_squared :
  let center1 := (3 : ℝ, -2 : ℝ)
  let radius1 := (5 : ℝ)
  let center2 := (3 : ℝ, 4 : ℝ)
  let radius2 := (3 : ℝ)
  let equation1 := (x : ℝ, y : ℝ) => (x - center1.1)^2 + (y - center1.2)^2 = radius1^2
  let equation2 := (x : ℝ, y : ℝ) => (x - center2.1)^2 + (y - center2.2)^2 = radius2^2
  ∃ C D : ℝ × ℝ, equation1 C.1 C.2 ∧ equation2 C.1 C.2 ∧ equation1 D.1 D.2 ∧ equation2 D.1 D.2 ∧
  let distance_squared := (C.1 - D.1)^2 + (C.2 - D.2)^2
  distance_squared = 224 / 9 := sorry

end intersecting_circles_CD_squared_l817_817603


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817330

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817330


namespace sum_of_reciprocals_lt_two_l817_817120

variables (a : ℕ → ℕ) (n : ℕ)
hypothesis H1 : ∀ i, 0 < a i ∧ a i < 1000
hypothesis H2 : ∀ i j, i ≠ j → Nat.lcm (a i) (a j) > 1000

theorem sum_of_reciprocals_lt_two : (∑ i in Finset.range n, 1 / (a i : ℝ)) < 2 := sorry

end sum_of_reciprocals_lt_two_l817_817120


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817334

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817334


namespace product_fraction_l817_817689

theorem product_fraction :
  ∏ n in Finset.range 15, ((n + 1 + 1) * (n + 1 + 3)) / ((n + 1 + 4) * (n + 1 + 5)) = (4 : ℚ) / 95 :=
by
  sorry

end product_fraction_l817_817689


namespace count_interesting_numbers_l817_817839

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817839


namespace continuous_avg_radius_const_l817_817536

noncomputable def average_on_circle (f : ℝ × ℝ → ℝ) (x : ℝ × ℝ) (r : ℝ) : ℝ := sorry

theorem continuous_avg_radius_const (f : ℝ × ℝ → ℝ) :
  (∀ x, 0 ≤ f x ∧ f x ≤ 1) →
  continuous f →
  (∀ x : ℝ × ℝ, average_on_circle f x 1 = f x) →
  ∀ x1 x2 : ℝ × ℝ, f x1 = f x2 :=
by
  sorry

end continuous_avg_radius_const_l817_817536


namespace prob_statement_l817_817788

-- Define the function f
def f (x a : ℝ) : ℝ := 1 - x + a * Real.log x

-- Define the condition for a
def a := Real.log 2 / Real.log Real.exp 1 -- a = log2(e)

-- Define the main theorem
theorem prob_statement (x : ℝ) (h1 : 0 < x) (h2 : x ≤ 2) :
  (x - 1) * f x a ≥ 0 :=
sorry

end prob_statement_l817_817788


namespace find_f_l817_817736

noncomputable def func_satisfies_eq (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = x * f x - y * f y

theorem find_f (f : ℝ → ℝ) (h : func_satisfies_eq f) : ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end find_f_l817_817736


namespace focus_of_given_parabola_is_correct_l817_817423

-- Define the problem conditions
def parabolic_equation (x y : ℝ) : Prop := y = 4 * x^2

-- Define what it means for a point to be the focus of the given parabola
def is_focus_of_parabola (x0 y0 : ℝ) : Prop := 
    x0 = 0 ∧ y0 = 1 / 16

-- Define the theorem to be proven
theorem focus_of_given_parabola_is_correct : 
  ∃ x0 y0, parabolic_equation x0 y0 ∧ is_focus_of_parabola x0 y0 :=
sorry

end focus_of_given_parabola_is_correct_l817_817423


namespace num_rem_three_by_seven_l817_817860

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817860


namespace cos_A_in_triangle_ABC_l817_817079

theorem cos_A_in_triangle_ABC
  (A B C : ℝ) 
  (a b c : ℝ)
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : a = sin A)
  (h6 : b = sin B)
  (h7 : c = sin C)
  (h8 : (√3 * b - c) * cos A = a * cos C) :
  cos A = √3 / 3 :=
by
  sorry

end cos_A_in_triangle_ABC_l817_817079


namespace num_rem_three_by_seven_l817_817856

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817856


namespace jen_total_birds_l817_817482

-- Define the initial conditions
def total_birds (c : ℕ) : ℕ :=
  let d := 10 + 4 * c in
  if d = 150 then c + d else 0

theorem jen_total_birds : total_birds 35 = 185 :=
  sorry

end jen_total_birds_l817_817482


namespace common_ratio_geometric_sequence_l817_817454

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := sorry

theorem common_ratio_geometric_sequence
  (a3_eq : a 3 = 2 * S 2 + 1)
  (a4_eq : a 4 = 2 * S 3 + 1)
  (geometric_seq : ∀ n, a (n+1) = a 1 * (q ^ n))
  (h₀ : a 1 ≠ 0)
  (h₁ : q ≠ 0) :
  q = 3 :=
sorry

end common_ratio_geometric_sequence_l817_817454


namespace divide_by_repeating_decimal_l817_817209

noncomputable def p := 0.66666666666666666666666666666666 -- using a repeating decimal
noncomputable def six := 6

theorem divide_by_repeating_decimal : six / p = 9 := 
by {
  sorry
}

end divide_by_repeating_decimal_l817_817209


namespace find_constants_l817_817998

theorem find_constants (c d : ℚ) :
  (∃ g : ℚ → ℚ, g = λ x, c * x^3 - 8 * x^2 + d * x - 7 ∧ g 2 = -15 ∧ g (-3) = -140) →
  (c, d) = (36/7, -109/7) :=
by
  sorry

end find_constants_l817_817998


namespace cannot_compare_greening_coverage_l817_817576

-- Definitions based on the conditions
def greening_coverage_rate_A : Float := 0.1
def greening_coverage_rate_B : Float := 0.08

-- The main statement
theorem cannot_compare_greening_coverage :
  (∃ A_area B_area : Float, greening_coverage_rate_A * A_area ≠ greening_coverage_rate_B * B_area ∨ greening_coverage_rate_A * A_area = greening_coverage_rate_B * B_area) →
  "Cannot be compared" :=
by 
  sorry

end cannot_compare_greening_coverage_l817_817576


namespace chord_length_condition_l817_817040

theorem chord_length_condition (c : ℝ) (h : c > 0) :
  (∃ (x1 x2 : ℝ), 
    x1 ≠ x2 ∧ 
    dist (x1, x1^2) (x2, x2^2) = 2 ∧ 
    ∃ k : ℝ, x1 * k + c = x1^2 ∧ x2 * k + c = x2^2 ) 
    ↔ c > 0 :=
sorry

end chord_length_condition_l817_817040


namespace percent_of_N_in_M_l817_817173

theorem percent_of_N_in_M (N M : ℝ) (hM : M ≠ 0) : (N / M) * 100 = 100 * N / M :=
by
  sorry

end percent_of_N_in_M_l817_817173


namespace count_two_digit_integers_remainder_3_div_7_l817_817902

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817902


namespace first_obtuse_triangle_H6_l817_817642

/-- Define the initial set of angles for triangle H_0 -/
def H0_angles := (59.5, 60.0, 60.5)

/-- Define a function to compute the angles of the pedal triangle given the angles of a triangle -/
def pedal_triangle_angles (angles : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let α := angles.1
  let β := angles.2
  let γ := angles.3
  (180 - 2 * α, 180 - 2 * β, 180 - 2 * γ)

/-- Define a function to compute the sequence of pedal triangles angles starting from H_0 -/
def pedal_triangle_sequence (n : ℕ) : ℝ × ℝ × ℝ :=
  Nat.iterate pedal_triangle_angles n H0_angles

/-- The theorem to prove that the 6th pedal triangle H_6 is the first one with an obtuse angle -/
theorem first_obtuse_triangle_H6 :
  ∃ n, ∀ k, k < 6 → 
    Nat.iterate pedal_triangle_angles k H0_angles does not contain an angle > 90 ∧ 
    Nat.iterate pedal_triangle_angles 6 H0_angles contains an angle > 90 :=
sorry

end first_obtuse_triangle_H6_l817_817642


namespace evaluate_expression_l817_817547

variable (a b : ℤ)

-- Define the original expression
def orig_expr (a b : ℤ) : ℤ :=
  (a^2 * b - 4 * a * b^2 - 1) - 3 * (b^2 * a - 2 * a^2 * b + 1)

-- Specify the values for a and b
def a_val : ℤ := -1
def b_val : ℤ := 1

-- Prove that the expression evaluates to 10 when a = -1 and b = 1
theorem evaluate_expression : orig_expr a_val b_val = 10 := 
  by sorry

end evaluate_expression_l817_817547


namespace minimum_sum_on_face_l817_817581

theorem minimum_sum_on_face (numbers : set ℕ) (H : numbers = {1, 2, 3, 4, 5, 6, 7, 8}) 
  (face_vertices : fin 6 → fin 4 → ℕ)
  (H_vertex_condition : ∀ i j k l : fin 4, face_vertices i ≠ face_vertices j ∧ face_vertices i ≠ face_vertices k ∧ face_vertices i ≠ face_vertices l →
    face_vertices i ∈ numbers ∧ face_vertices j ∈ numbers ∧ face_vertices k ∈ numbers ∧ face_vertices l ∈ numbers)
  (H_sum_condition : ∀ (f : fin 6), ∑ (i : fin 4), face_vertices f i ≥ 10) : 
  ∃ (f : fin 6), ∑ (i : fin 4), face_vertices f i = 16 :=
by 
  sorry

end minimum_sum_on_face_l817_817581


namespace count_two_digit_integers_with_remainder_3_l817_817848

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817848


namespace frog_can_reach_l817_817652

def frog_jump1 (a b : ℕ) : ℕ × ℕ :=
  if a > b then (a - b, b)
  else if a < b then (a, b - a)
  else (a, b)

def frog_jump2 (a b : ℕ) : ℕ × ℕ :=
  (2 * a, b)

def frog_jump3 (a b : ℕ) : ℕ × ℕ :=
  (a, 2 * b)

def can_reach (start target : ℕ × ℕ) : Prop :=
  ∃ n : ℕ, iterate (λ p : ℕ × ℕ, frog_jump1 p.1 p.2) n start = target ∨
           ∃ m : ℕ, iterate (λ p : ℕ × ℕ, frog_jump2 p.1 p.2) m start = target ∨
           iterate (λ p : ℕ × ℕ, frog_jump3 p.1 p.2) m start = target

theorem frog_can_reach :
  can_reach (1, 1) (3, 5) ∧
  can_reach (1, 1) (200, 6) ∧
  ¬ can_reach (1, 1) (12, 60) ∧
  ¬ can_reach (1, 1) (200, 5) :=
by sorry

end frog_can_reach_l817_817652


namespace smallest_j_l817_817364

theorem smallest_j (j : ℕ) : (∀ (A : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ A i j ∧ A i j ≤ 100) ∧ 
  (∀ n, ∃ (i j : ℕ) (k l : ℕ), 
    i + j ≤ 10 ∧ k + l ≤ 10 ∧ 
    ∀ x ∈ finset.range 10, 
      (1 ≤ A (i + x % j) (j + x / j) ∧ 
       A (i + x % j) (j + x / j) = n + x) ∨ sorry) 
  → j = 5 ) :=
sorry

end smallest_j_l817_817364


namespace gel_pen_ratio_l817_817309

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l817_817309


namespace intersection_logarithmic_set_l817_817506

theorem intersection_logarithmic_set (M P : Set ℝ) :
  (M = {x : ℝ | ∃ y : ℝ, y = log10 (x - 3)}) ∧
  (P = {x : ℝ | -1 ≤ x ∧ x ≤ 4}) →
  (M ∩ P = {x : ℝ | 3 < x ∧ x ≤ 4}) :=
sorry

end intersection_logarithmic_set_l817_817506


namespace part_one_part_two_l817_817762

variable {α : Type*}

def a_arith (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def b_geom (b : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, b (n + 1) = b n * q

def seq_a := λ (n : ℕ), 2 * n - 1

def seq_b := λ (n : ℕ), 3 ^ (n - 1)

def seq_c := λ (n : ℕ), seq_a n + (-1)^n * seq_b n

theorem part_one (a b : ℕ → ℝ) (d q : ℝ) (h_a_arith : a_arith a d) (h_b_geom : b_geom b q)
    (h_b2 : b 2 = 3) (h_b3 : b 3 = 9) (h_a1_eq_b1 : a 1 = b 1) (h_a14_eq_b4 : a 14 = b 4) :
    a = seq_a := by
  -- proof omitted
  sorry

theorem part_two (n : ℕ) :
  ∑ k in Finset.range (2 * n), seq_c (k + 1) = 4 * n^2 + 9^n / 4 - 1 / 4 := by
  -- proof omitted
  sorry

end part_one_part_two_l817_817762


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817895

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817895


namespace last_locker_opened_l817_817646

theorem last_locker_opened (n : ℕ) (locks : ℕ → bool) : 
  ( ∀ i : ℕ, 1 ≤ i ∧ i ≤ 512 → locks i = tt) → 
  ( locks 86 = tt ) :=
sorry

end last_locker_opened_l817_817646


namespace pairwise_sums_divisibility_l817_817728

theorem pairwise_sums_divisibility (n : ℕ) (hn : n = 2011) :
  ∃ (nums : Fin n → ℕ), (∃ (S : Fin (n * (n - 1) / 2) → ℕ), 
  (∃ (div_by_3 : Fin (n * (n - 1) / 2) → Fin 3), 
  (∀ i j, i < j → S ⟨i, _⟩ + S ⟨j, _⟩ = nums ⟨i, _⟩ + nums ⟨j, _⟩)) ∧ 
  (∃ (count_div_3. count_mod_1 : ℕ), count_div_3 = count_mod_1 ∧ 
  count_div_3 = (n * (n - 1) / 6) ∧ 
  count_mod_1 = (n * (n - 1) / 6)' ∧
  ((∀ i, div_by_3 i = 0) → count_div_3) ∧
  ((∀ i, div_by_3 i = 1) → count_mod_1) )) :=
sorry

end pairwise_sums_divisibility_l817_817728


namespace smallest_in_row_10_row_includes_n_squared_sub_n_and_n_squared_sub_2n_largest_n_not_including_n_squared_sub_10n_l817_817582

/-- Conditions for an integer m to belong to Row n -/
def in_row (n m : ℕ) : Prop :=
  m % n = 0 ∧ m ≤ n^2 ∧ ∀ k, k < n → m % k ≠ 0

/-- a) The smallest integer in Row 10 -/
theorem smallest_in_row_10 : 
  ∃ m, in_row 10 m ∧ ∀ k, in_row 10 k → k ≥ m :=
sorry

/-- b) For all n >= 3, Row n includes n^2 - n and n^2 - 2n -/
theorem row_includes_n_squared_sub_n_and_n_squared_sub_2n (n : ℕ) (h : n ≥ 3) :
  in_row n (n^2 - n) ∧ in_row n (n^2 - 2n) :=
sorry

/-- c) The largest positive integer n such that Row n does not include n^2 - 10n -/
theorem largest_n_not_including_n_squared_sub_10n :
  ∃ n, (∀ m, in_row n m → m ≠ n^2 - 10 * n) ∧ ∀ k > n, ∃ m, in_row k m ∧ m = k^2 - 10 * k :=
sorry

end smallest_in_row_10_row_includes_n_squared_sub_n_and_n_squared_sub_2n_largest_n_not_including_n_squared_sub_10n_l817_817582


namespace work_completion_days_l817_817435

theorem work_completion_days (h : ∀ {m1 m2 d1 d2 : ℝ}, (m1 * d1 = m2 * d2)) :
  ∀ d : ℝ, (72 * 18 = 144 * d) → d = 9 :=
begin
  intros,
  sorry,
end

end work_completion_days_l817_817435


namespace squares_difference_l817_817525

theorem squares_difference (x y z : ℤ) 
  (h1 : x + y = 10) 
  (h2 : x - y = 8) 
  (h3 : y + z = 15) : 
  x^2 - z^2 = -115 :=
by 
  sorry

end squares_difference_l817_817525


namespace range_of_a_l817_817524

variables {x : ℝ} {θ a : ℝ}
def α : ℝ × ℝ := (x + 3, x)
def β : ℝ × ℝ := (2 * sin θ * cos θ, a * sin θ + a * cos θ)

theorem range_of_a :
  (∀ x ∈ ℝ, ∀ θ ∈ set.Icc 0 (real.pi / 2), 
    (√((α.1 + β.1)^2 + (α.2 + β.2)^2) ≥ √2)) →
  (a ≤ 1 ∨ a ≥ 5) :=
begin
  sorry
end

end range_of_a_l817_817524


namespace remaining_battery_life_l817_817982

theorem remaining_battery_life
    (standby_hours : ℕ)
    (usage_hours : ℕ)
    (total_on_hours : ℕ)
    (active_use_hours : ℕ / 60)
    (battery_life_standby_mode : ℕ)
    (battery_life_usage_mode : ℕ)
    (battery_consumption_standby : ℚ)
    (battery_consumption_usage : ℚ)
    (total_battery_consumed : ℚ)
    (remaining_battery : ℚ)
    (remaining_standby_hours : ℕ) :
    standby_hours = 20 →
    usage_hours = 4 →
    total_on_hours = 10 →
    active_use_hours = 1.5 →
    battery_life_standby_mode = standby_hours →
    battery_life_usage_mode = usage_hours →
    battery_consumption_standby = total_on_hours - active_use_hours / battery_life_standby_mode →
    battery_consumption_usage = active_use_hours / battery_life_usage_mode →
    total_battery_consumed = battery_consumption_standby + battery_consumption_usage →
    remaining_battery = 1 - total_battery_consumed →
    remaining_standby_hours = remaining_battery * battery_life_standby_mode →
    remaining_standby_hours = 4 :=
by
    intros
    sorry

end remaining_battery_life_l817_817982


namespace smallest_four_digit_equiv_8_mod_9_l817_817215

theorem smallest_four_digit_equiv_8_mod_9 :
  ∃ n : ℕ, n % 9 = 8 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 9 = 8 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m :=
sorry

end smallest_four_digit_equiv_8_mod_9_l817_817215


namespace not_n_gt_66_l817_817514

theorem not_n_gt_66 (n : ℕ) (h1 : 0 < n) (h2 : 1 / 2 + 1 / 3 + 1 / 11 + 1 / n ∈ ℤ) : ¬ (n > 66) :=
sorry

end not_n_gt_66_l817_817514


namespace find_k_l817_817752

def Point := ℝ × ℝ

noncomputable def k_value (A B : Point) (a : ℝ × ℝ) (k : ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  ∧ a = (2 * k - 1, 7)
  ∧ (λλ : ∃ (λ : ℝ), a = (λ * AB.1, λ * AB.2))

theorem find_k : ∀ (A B : Point), A = (2, -2) → B = (4, 3) → 
  (∀ (a : ℝ × ℝ), a = (2 * k - 1, 7) → (∃ (λ : ℝ), a = (λ * (B.1 - A.1), λ * (B.2 - A.2))) → k = 19 / 10)
  :=
by
  sorry

end find_k_l817_817752


namespace sqrt_meaningful_l817_817065

theorem sqrt_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l817_817065


namespace s_g_7_l817_817515

def s (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)
def g (x : ℝ) : ℝ := 7 - s x

theorem s_g_7 : s (g 7) = Real.sqrt (30 - 4 * Real.sqrt 30) := 
by 
  sorry

end s_g_7_l817_817515


namespace conjugate_of_z_l817_817763

def imaginary_unit (i : ℂ) := i^2 = -1

def complex_number_z (z : ℂ) : Prop :=
  let i := Complex.I in
  z = (2 * i) / (2 + i)

theorem conjugate_of_z :
  ∀ (z : ℂ), imaginary_unit Complex.I → complex_number_z z → Complex.conj z = (2/5) - (4/5) * Complex.I :=
by
  intro z h1 h2
  sorry

end conjugate_of_z_l817_817763


namespace line_AT_passes_through_S1_or_S2_l817_817118

theorem line_AT_passes_through_S1_or_S2
    (ABC : Triangle)
    (Omega : Circle)
    (incircle_ABC : Incircle ABC)
    (D : Point)
    (K M : Point)
    (A_excircle : Excircle ABC)
    (S1 S2 T : Point) :
    is_scalene_triangle ABC 
    → touches_at incircle_ABC (line_segment BC) D 
    → is_angle_bisector (angle A) (line_segment AK)
    → lies_on K (line_segment BC)
    → lies_on M Omega
    → intersects (circumcircle (triangle DKM)) A_excircle = {S1, S2}
    → intersects (circumcircle (triangle DKM)) Omega = T
    → T ≠ M
    → passes_through (line AT) S1 ∨ passes_through (line AT) S2 :=
begin
  sorry
end

end line_AT_passes_through_S1_or_S2_l817_817118


namespace tan_240_l817_817754

theorem tan_240 (x y : ℝ) (h : (x, y) ≠ (0, 0)) (h_angle : ∃ r : ℝ, (x, y) = (r * real.cos (4 * real.pi / 3), r * real.sin (4 * real.pi / 3))) : y / x = real.sqrt 3 :=
by
  sorry

end tan_240_l817_817754


namespace simplify_expression1_simplify_expression2_l817_817636

theorem simplify_expression1 (α : ℝ) : 
  (tan (3 * π - α) * cos (2 * π - α) * sin (-α + 3 * π / 2)) / 
  (cos (-α - π) * sin (-π + α) * cos (α + 5 * π / 2)) = 
  - (1 / sin α) :=
by sorry

theorem simplify_expression2 (α : ℝ) : 
  (cos (π / 2 + α) * sin (3 * π / 2 - α)) / 
  (cos (π - α) * tan (π - α)) = 
  cos α :=
by sorry

end simplify_expression1_simplify_expression2_l817_817636


namespace sample_size_eq_36_l817_817648

def total_population := 27 + 54 + 81
def ratio_elderly_total := 27 / total_population
def selected_elderly := 6
def sample_size := 36

theorem sample_size_eq_36 : 
  (selected_elderly : ℚ) / (sample_size : ℚ) = ratio_elderly_total → 
  sample_size = 36 := 
by 
sorry

end sample_size_eq_36_l817_817648


namespace kayla_score_fourth_level_l817_817104

theorem kayla_score_fourth_level 
  (score1 score2 score3 score5 score6 : ℕ) 
  (h1 : score1 = 2) 
  (h2 : score2 = 3) 
  (h3 : score3 = 5) 
  (h5 : score5 = 12) 
  (h6 : score6 = 17)
  (h_diff : ∀ n : ℕ, score2 - score1 + n = score3 - score2 + n + 1 ∧ score3 - score2 + n + 2 = score5 - score3 + n + 3 ∧ score5 - score3 + n + 4 = score6 - score5 + n + 5) :
  ∃ score4 : ℕ, score4 = 8 :=
by
  sorry

end kayla_score_fourth_level_l817_817104


namespace initial_garrison_men_l817_817653

theorem initial_garrison_men (M : ℕ) (H1 : ∃ provisions : ℕ, provisions = M * 60)
  (H2 : ∃ provisions_15 : ℕ, provisions_15 = M * 45)
  (H3 : ∀ provisions_15 (new_provisions: ℕ), (provisions_15 = M * 45 ∧ new_provisions = 20 * (M + 1250)) → provisions_15 = new_provisions) :
  M = 1000 :=
by
  sorry

end initial_garrison_men_l817_817653


namespace sum_of_squares_l817_817765

theorem sum_of_squares (x y z w a b c d : ℝ) (h1: x * y = a) (h2: x * z = b) (h3: y * z = c) (h4: x * w = d) :
  x^2 + y^2 + z^2 + w^2 = (ab + bd + da)^2 / abd := 
by
  sorry

end sum_of_squares_l817_817765


namespace common_ratio_l817_817402

theorem common_ratio
  (a b : ℝ)
  (h_arith : 2 * a = 1 + b)
  (h_geom : (a + 2) ^ 2 = 3 * (b + 5))
  (h_non_zero_a : a + 2 ≠ 0)
  (h_non_zero_b : b + 5 ≠ 0) :
  (a = 4 ∧ b = 7) ∧ (b + 5) / (a + 2) = 2 :=
by {
  sorry
}

end common_ratio_l817_817402


namespace gel_pen_price_ratio_l817_817305

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l817_817305


namespace june_spent_on_music_books_l817_817488

theorem june_spent_on_music_books
  (total_budget : ℤ)
  (math_books_cost : ℤ)
  (science_books_cost : ℤ)
  (art_books_cost : ℤ)
  (music_books_cost : ℤ)
  (h_total_budget : total_budget = 500)
  (h_math_books_cost : math_books_cost = 80)
  (h_science_books_cost : science_books_cost = 100)
  (h_art_books_cost : art_books_cost = 160)
  (h_total_cost : music_books_cost = total_budget - (math_books_cost + science_books_cost + art_books_cost)) :
  music_books_cost = 160 :=
sorry

end june_spent_on_music_books_l817_817488


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817332

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817332


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817907

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817907


namespace original_manufacturing_cost_l817_817627

noncomputable def selling_price := 100  -- Given as we derived it in the solution
theorem original_manufacturing_cost :
  let P := selling_price
  let C_orig := 0.60 * P
  P = 100 → C_orig = 60 :=
by
  intro h₁
  simp [selling_price] at h₁
  unfold C_orig
  simp [h₁]
  sorry

end original_manufacturing_cost_l817_817627


namespace polynomial_coeff_sum_abs_l817_817501

theorem polynomial_coeff_sum_abs (a a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℤ) 
  (h : (2*x - 1)^5 + (x + 2)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  |a| + |a_2| + |a_4| = 30 :=
sorry

end polynomial_coeff_sum_abs_l817_817501


namespace roses_ordered_l817_817474

theorem roses_ordered (tulips carnations roses : ℕ) (cost_per_flower total_expenses : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : cost_per_flower = 2)
  (h4 : total_expenses = 1890)
  (h5 : total_expenses = (tulips + carnations + roses) * cost_per_flower) :
  roses = 320 :=
by 
  -- Using the mathematical equivalence and conditions provided
  sorry

end roses_ordered_l817_817474


namespace simplify_and_evaluate_l817_817549

noncomputable def expr (x : ℝ) : ℝ :=
  (x + 3) * (x - 2) + x * (4 - x)

theorem simplify_and_evaluate (x : ℝ) (hx : x = 2) : expr x = 4 :=
by
  rw [hx]
  show expr 2 = 4
  sorry

end simplify_and_evaluate_l817_817549


namespace limit_arctg_sin_l817_817691

theorem limit_arctg_sin (f : ℝ → ℝ) (h₀ : ∀ x ≠ 2, f x = (Real.arctan (x^2 - 2 * x)) / (Real.sin (3 * π * x))) :
  filter.tendsto f (nhds 2) (nhds (2 / (3 * π))) :=
sorry

end limit_arctg_sin_l817_817691


namespace problem_solution_l817_817236

variable (x : ℝ)
def e := x^2 + (1/x)^2

theorem problem_solution
  (h : x + 1/x = 5) :
  e = 23 := by
  sorry

end problem_solution_l817_817236


namespace problem_solution_l817_817999

theorem problem_solution :
  ∀ p q : ℝ, (3 * p ^ 2 - 5 * p - 21 = 0) → (3 * q ^ 2 - 5 * q - 21 = 0) →
  (9 * p ^ 3 - 9 * q ^ 3) * (p - q)⁻¹ = 88 :=
by 
  sorry

end problem_solution_l817_817999


namespace sufficient_not_necessary_condition_example_of_not_necessary_l817_817012

theorem sufficient_not_necessary_condition (m n : ℝ) (h1 : m / n - 1 = 0) : m - n = 0 :=
begin
  sorry,
end

theorem example_of_not_necessary (m n : ℝ) : m - n = 0 ∧ ¬(m / n - 1 = 0) :=
begin
  use_m_neq_zero : m = 0,
  use_n_eq_zero : n = 0,
  split,
  sorry,
  sorry
end

end sufficient_not_necessary_condition_example_of_not_necessary_l817_817012


namespace radius_of_spheres_l817_817979

-- Define the base radius and height of the cone
def cone_base_radius : ℝ := 5
def cone_height : ℝ := 12

-- Define the conditions for the congruent spheres inside the cone
def sphere_tangent_to_base_and_side (r : ℝ) : Prop :=
  ∃ (O₁ O₂ O₃ : ℝ × ℝ × ℝ),
    (O₁.2 = r ∧ O₂.2 = r ∧ O₃.2 = r) ∧
    (dist O₁ O₂ = 2 * r ∧ dist O₂ O₃ = 2 * r ∧ dist O₁ O₃ = 2 * r) ∧
    ∀ (O : ℝ × ℝ × ℝ), (O.2 ≤ r ∧ dist O (0, 0, 0) ≤ r + cone_height)

-- Prove that the radius r of each sphere is equal to the given expression
theorem radius_of_spheres : 
  ∃ r : ℝ, sphere_tangent_to_base_and_side r ∧ r = (90 - 40 * real.sqrt 3) / 11 :=
by
  -- Proof to be completed
  sorry

end radius_of_spheres_l817_817979


namespace sum_of_possible_ns_at_continuity_point_l817_817491

noncomputable def f (x : ℝ) (n : ℝ) : ℝ :=
if x < n then x^2 + 3*x + 1 else 3*x + 6

theorem sum_of_possible_ns_at_continuity_point (n : ℝ) :
  (∀ x : ℝ, (x < n → (f x n = x^2 + 3*x + 1)) ∧ (x ≥ n → (f x n = 3*x + 6))) →
  (∀ x : ℝ, continuous_at (f x) n) →
  set.sum (set_of (λ n, (f n n = n^2 + 3*n + 1) ∧ (f n n = 3*n + 6))) = 0 :=
by
  sorry

end sum_of_possible_ns_at_continuity_point_l817_817491


namespace polynomial_value_at_one_l817_817055

theorem polynomial_value_at_one
  (a b c : ℝ)
  (h1 : -a - b - c + 1 = 6)
  : a + b + c + 1 = -4 :=
by {
  sorry
}

end polynomial_value_at_one_l817_817055


namespace ratio_d_e_l817_817015

theorem ratio_d_e (a b c d e f : ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : e / f = 1 / 6)
  (h5 : a * b * c / (d * e * f) = 0.25) :
  d / e = 1 / 4 :=
sorry

end ratio_d_e_l817_817015


namespace not_even_function_exists_l817_817742

variable {α : Type} [Nonempty α] {f : α → α}

theorem not_even_function_exists (f : ℝ → ℝ) (h : ¬ ∀ x : ℝ, f(-x) = f(x)) : ∃ x₀ : ℝ, f(-x₀) ≠ f(x₀) :=
by
  sorry

end not_even_function_exists_l817_817742


namespace reflection_slope_intercept_l817_817657

noncomputable def reflect_line_slope_intercept (k : ℝ) (hk1 : k ≠ 0) (hk2 : k ≠ -1) : ℝ × ℝ :=
  let slope := (1 : ℝ) / k
  let intercept := (k - 1) / k
  (slope, intercept)

theorem reflection_slope_intercept {k : ℝ} (hk1 : k ≠ 0) (hk2 : k ≠ -1) :
  reflect_line_slope_intercept k hk1 hk2 = (1/k, (k-1)/k) := by
  sorry

end reflection_slope_intercept_l817_817657


namespace angle_identity_l817_817974

-- Define the angles involved
def angleAOD (a b c : ℝ) := 2.5 * b -- given angle AOD is 2.5 times angle BOC
noncomputable def angleAOD_value := 128.57 -- The computed value of angle AOD

-- Define perpendicularity conditions
def perp_vectors (u v : ℝ) := u.perpendicular v
def perp_OA_OC := perp_vectors OA OC
def perp_OB_OD := perp_vectors OB OD

-- Define the proof statement
theorem angle_identity
  (OA OC OB OD : ℝ)
  (angleAOD : ℝ)  -- The angle AOD
  (angleBOC : ℝ)  -- The angle BOC
  (AOD_eq_2_5_BOC : angleAOD = 2.5 * angleBOC)  -- Angle correspondence
  (perp_OA_OC : perp_vectors OA OC)  -- Perpendicularity of OA and OC
  (perp_OB_OD : perp_vectors OB OD)  -- Perpendicularity of OB and OD
  (AOD_value : angleAOD = 128.57) : 
  angleAOD = angleAOD_value := 
by sorry

end angle_identity_l817_817974


namespace sum_abs_diff_le_n_sq_equality_condition_l817_817137

-- Given real numbers x such that 0 ≤ x_i ≤ 2 for i = 1, 2, ..., n
variables {n : ℕ} (x : Fin n → ℝ)
-- The assumption that the x_i's are between 0 and 2
variables (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 2)

-- We need to prove the sum of absolute differences is less than or equal to n^2
theorem sum_abs_diff_le_n_sq : 
  (∑ i, ∑ j, |x i - x j|) ≤ n ^ 2 :=
by 
  sorry

-- We also need to prove when equality holds: when n is even and not when n is odd
theorem equality_condition :
  ((∑ i, ∑ j, |x i - x j| = n ^ 2) ↔ (n % 2 = 0)) :=
by 
  sorry

end sum_abs_diff_le_n_sq_equality_condition_l817_817137


namespace sum_of_integers_from_neg15_to_5_l817_817693

-- defining the conditions
def first_term : ℤ := -15
def last_term : ℤ := 5

-- sum of integers from first_term to last_term
def sum_arithmetic_series (a l : ℤ) : ℤ :=
  let n := l - a + 1
  (n * (a + l)) / 2

-- the statement we need to prove
theorem sum_of_integers_from_neg15_to_5 : sum_arithmetic_series first_term last_term = -105 := by
  sorry

end sum_of_integers_from_neg15_to_5_l817_817693


namespace value_of_f_l817_817387

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / 
  (Real.sin (π / 2 + α) * Real.sin (-π - α))

theorem value_of_f (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) (h2 : Real.cos (α + π / 3) = 3 / 5) : 
  f α = (-3 - 4 * real.sqrt 3) / 10 := 
  sorry

end value_of_f_l817_817387


namespace part_one_locus_part_two_constant_l817_817107

theorem part_one_locus (P : ℝ × ℝ) :
  (∃ x y : ℝ, P = (x, y) ∧ (frac x^2 9 + frac y^2 8 = 1)) ↔ 
  (frac P.1^2 9 + frac P.2^2 8 = 1) := by sorry

theorem part_two_constant (l1 l2 : ℝ → ℝ) (F : ℝ × ℝ) (A B C D : ℝ × ℝ) (k : ℝ) :
  F = (1, 0) →
  (l1(x) = k*(x-1)) ∧ (l2(x) = -1/k*(x-1)) →
  ((frac (x1+x2) 2).1 + (frac (x1*x2) 2).1) ≠ 0 →
  ((frac (y1+y2) 2).2 + (frac (y1*y2) 2).2) ≠ 0  →
  (∃ x1 x2 y1 y2 : ℝ, 
    A = (x1, f1(x1)) ∧ 
    B = (x2, f1(x2)) ∧ 
    C = (y1, f2(y1)) ∧ 
    D = (y2, f2(y2))) →
  (1 / dist A B + 1 / dist C D) = frac 17 48 := by sorry

end part_one_locus_part_two_constant_l817_817107


namespace y_equals_x_cubed_l817_817702

def y_value (x : ℕ) : ℕ :=
  match x with
  | 1 => 1
  | 2 => 8
  | 3 => 27
  | 4 => 64
  | 5 => 125
  | _ => 0

theorem y_equals_x_cubed :
  ∀ x, x ∈ {1, 2, 3, 4, 5} → y_value x = x^3 :=
by
  intro x hx
  cases hx
  case inl h => sorry
  case inr h1 => cases h1
                case inl h => sorry
                case inr h2 => cases h2
                              case inl h => sorry
                              case inr h3 => cases h3
                                            case inl h => sorry
                                            case inr h4 => cases h4
                                                          case inl h => sorry
                                                          case inr h => exfalso
                                                            exact h

end y_equals_x_cubed_l817_817702


namespace f_is_odd_l817_817776

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 3 then 2^x - 1 else x - 5

theorem f_is_odd (x : ℝ) : f (-x) = -f x := sorry

example : f (f 3) = -3 := by
  have f3 : f 3 = 3 - 5 := by
    have h : 3 ≥ 3 := by linarith
    simp [f, h]
    ring
  rw [f3]
  have h1 : f (3 - 5) = -3 := by
    have ff3 : f (-2) = -f 2 := by
      exact f_is_odd 2
    have f2 : f 2 = 2^2 - 1 := by
      have h2 : 0 ≤ 2 ∧ 2 < 3 := by linarith
      simp [f, h2]
      ring
    rw [f2] at ff3
    simp [ff3]
  simp [h1]

end f_is_odd_l817_817776


namespace greatest_radius_of_circle_area_lt_90pi_l817_817445

theorem greatest_radius_of_circle_area_lt_90pi : ∃ (r : ℤ), (∀ (r' : ℤ), (π * (r':ℝ)^2 < 90 * π ↔ (r' ≤ r))) ∧ (π * (r:ℝ)^2 < 90 * π) ∧ (r = 9) :=
sorry

end greatest_radius_of_circle_area_lt_90pi_l817_817445


namespace min_value_abs_2a_minus_b_l817_817397

theorem min_value_abs_2a_minus_b (a b : ℝ) (h : 2 * a^2 - b^2 = 1) : ∃ c : ℝ, c = |2 * a - b| ∧ c = 1 := 
sorry

end min_value_abs_2a_minus_b_l817_817397


namespace range_of_a_l817_817637

noncomputable def has_root_in_R (f : ℝ → ℝ) : Prop :=
∃ x : ℝ, f x = 0

theorem range_of_a (a : ℝ) (h : has_root_in_R (λ x => 4 * x + a * 2^x + a + 1)) : a ≤ 0 :=
sorry

end range_of_a_l817_817637


namespace isosceles_triangle_perimeter_l817_817967

noncomputable def quadratic_roots (a b c : ℝ) : set ℝ :=
  {x | a * x^2 + b * x + c = 0}

def is_isosceles_triangle (base leg1 leg2 : ℝ) : Prop :=
  base = leg1 ∧ leg1 = leg2

def is_valid_leg (base leg : ℝ) : Prop :=
  leg ∈ quadratic_roots 1 (-7) 12 ∧ leg * 2 > base

theorem isosceles_triangle_perimeter :
  ∃ (leg : ℝ), is_isosceles_triangle 6 leg leg ∧ is_valid_leg 6 leg ∧ 6 + 2 * leg = 14 :=
sorry

end isosceles_triangle_perimeter_l817_817967


namespace strictly_decreasing_f_on_interval_l817_817703

def g (x : ℝ) : ℝ := x^2 - 6*x + 5

def log_base (a x : ℝ) : ℝ := log x / log a

def f (x : ℝ) : ℝ := log_base (1/2) (g x)

theorem strictly_decreasing_f_on_interval : 
  ∀ x y : ℝ, (5 < x) → (x < y) → f y < f x :=
by
  sorry

end strictly_decreasing_f_on_interval_l817_817703


namespace justin_current_age_l817_817681

theorem justin_current_age (angelina_future_age : ℕ) (years_until_future : ℕ) (age_difference : ℕ)
  (h_future_age : angelina_future_age = 40) (h_years_until_future : years_until_future = 5)
  (h_age_difference : age_difference = 4) : 
  (angelina_future_age - years_until_future) - age_difference = 31 :=
by
  -- This is where the proof would go.
  sorry

end justin_current_age_l817_817681


namespace determine_a_l817_817733

theorem determine_a (a : ℝ) (h : 3 ∈ ({1, -a^2, a - 1} : set ℝ)) : a = 4 :=
sorry

end determine_a_l817_817733


namespace sqrt_meaningful_l817_817066

theorem sqrt_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l817_817066


namespace chord_length_parabola_line_l817_817261

theorem chord_length_parabola_line :
  let line_eq(x : ℝ) := -2 * x + 2
  let parabola_eq(y : ℝ) := y^2 - 8 * x
  ∃ (x₁ x₂ : ℝ), 
    (parabola_eq (line_eq x₁) = 0 ∧
    parabola_eq (line_eq x₂) = 0 ∧
    ((x₁ + x₂) = 4) ∧
    ((x₁ * x₂) = 1)) → 
    (2 * Real.sqrt 15) :=
by
  sorry

end chord_length_parabola_line_l817_817261


namespace count_two_digit_integers_remainder_3_l817_817868

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817868


namespace range_of_k_in_first_quadrant_l817_817005

theorem range_of_k_in_first_quadrant (k : ℝ) (h₁ : k ≠ -1) :
  (∃ x y : ℝ, y = k * x - 1 ∧ x + y - 1 = 0 ∧ x > 0 ∧ y > 0) ↔ 1 < k := by sorry

end range_of_k_in_first_quadrant_l817_817005


namespace volume_cone_equals_volume_part_of_cylinder_l817_817232

noncomputable def volume_of_cylinder (R : ℝ) : ℝ := π * R^3
noncomputable def volume_of_cone (R : ℝ) : ℝ := (1 / 3) * π * R^3
noncomputable def volume_of_sphere (R : ℝ) : ℝ := (4 / 3) * π * R^3

theorem volume_cone_equals_volume_part_of_cylinder (R : ℝ) :
  volume_of_cone R = volume_of_cylinder R - volume_of_sphere R :=
  sorry

end volume_cone_equals_volume_part_of_cylinder_l817_817232


namespace count_interesting_numbers_l817_817835

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817835


namespace max_volume_pyramid_l817_817266

/-- Proving the maximal volume condition for a pyramid with the given constraints. -/
theorem max_volume_pyramid 
  (a b m : ℝ) 
  (height_eq : m = (Real.sqrt (3 * a^2 - b^2)) / 2) 
  (volume_eq : ∀ b, volume_pyramid a b m = (a * b * (Real.sqrt (3 * a^2 - b^2))) / 6) : 
  b = a * Real.sqrt (3 / 2) → 
  ∀ b, (forall m, height_eq m) = volume_eq := 
begin
  intros,
  sorry
end

end max_volume_pyramid_l817_817266


namespace find_real_numbers_l817_817714

axiom solution_sets : set (set ℝ) :=
  {{1, -3, 4}, {-3, 1, 4}, {4, -3, 1}, {1, 4, -12}, {4, -12, 1}, {-12, 1, 4}}

theorem find_real_numbers (x y z : ℝ) (h₁ : x + y + z = 17) (h₂ : x * y + y * z + z * x = 94) (h₃ : x * y * z = 168) :
  ({x, y, z} ∈ solution_sets) :=
by
  sorry

end find_real_numbers_l817_817714


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817326

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817326


namespace find_theta_l817_817924

theorem find_theta (θ : ℝ) (h1 : 0 < θ ∧ θ < 90) (h2 : √2 * cos (20 * real.pi / 180) = sin (θ * real.pi / 180) + cos (θ * real.pi / 180)) : θ = 25 :=
by sorry

end find_theta_l817_817924


namespace positive_two_digit_integers_remainder_3_l817_817816

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817816


namespace terry_mary_single_color_probability_l817_817654

noncomputable def probability_of_single_color_pick
  (total_candies : ℕ := 30)
  (red_candies : ℕ := 12)
  (blue_candies : ℕ := 12)
  (green_candies : ℕ := 6)
  (terry_picks : ℕ := 3)
  (mary_picks : ℕ := 3) : ℚ :=
let total_ways := nat.choose total_candies terry_picks in
let total_ways_mary := nat.choose (total_candies - terry_picks) mary_picks in
let ways_red_terry := nat.choose red_candies terry_picks in
let ways_red_mary := nat.choose (red_candies - terry_picks) mary_picks in
let ways_blue_terry := nat.choose blue_candies terry_picks in
let ways_blue_mary := nat.choose (blue_candies - terry_picks) mary_picks in
2 * ((ways_red_terry * ways_red_mary + ways_blue_terry * ways_blue_mary) / (total_ways * total_ways_mary))

theorem terry_mary_single_color_probability :
  probability_of_single_color_pick = 24 / 775 :=
sorry

end terry_mary_single_color_probability_l817_817654


namespace age_problem_l817_817230

variables (a b c : ℕ)

theorem age_problem (h₁ : a = b + 2) (h₂ : b = 2 * c) (h₃ : a + b + c = 27) : b = 10 :=
by {
  -- Interactive proof steps can go here.
  sorry
}

end age_problem_l817_817230


namespace num_solutions_of_functional_equation_l817_817718

theorem num_solutions_of_functional_equation :
  let f : ℝ → ℝ := λ x, x^2 / real.sqrt 2
  ∃ f : ℝ → ℝ, (∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 2 * x^2 * y^2) ∧
  (∀ x y : ℝ, (f x = x^2 / real.sqrt 2 ∨ f x = -x^2 / real.sqrt 2)) :=
begin
  sorry,
end

end num_solutions_of_functional_equation_l817_817718


namespace total_jumps_l817_817544

theorem total_jumps : 
    let Ronald_jumps := 157
    let Rupert_jumps := 3 * Ronald_jumps + 23
    let Rebecca_initial := 47
    let Rebecca_diff := 5
    let Rebecca_terms := 7
    let Rebecca_sum := Rebecca_terms / 2 * (Rebecca_initial + (Rebecca_initial + (Rebecca_terms - 1) * Rebecca_diff))
    Ronald_jumps + Rupert_jumps + Rebecca_sum = 1085 := 
by
  let Ronald_jumps := 157
  let Rupert_jumps := 3 * Ronald_jumps + 23
  let Rebecca_initial := 47
  let Rebecca_diff := 5
  let Rebecca_terms := 7
  let Rebecca_sum := Rebecca_terms / 2 * (Rebecca_initial + (Rebecca_initial + (Rebecca_terms - 1) * Rebecca_diff))
  show Ronald_jumps + Rupert_jumps + Rebecca_sum = 1085 from sorry

end total_jumps_l817_817544


namespace profit_percent_l817_817221

-- Definitions of the conditions
def cost_price (C : ℝ) := C
def selling_price_two_third (C : ℝ) := 0.85 * C
def selling_price (C : ℝ) := (3 / 2) * selling_price_two_third C

-- Theorem stating the profit percent
theorem profit_percent (C : ℝ) (h : C > 0) : 
  (selling_price C - cost_price C) / cost_price C * 100 = 27.5 :=
by 
  sorry

end profit_percent_l817_817221


namespace cannot_obtain_fraction_l817_817103

noncomputable def fraction (a b : ℕ) : ℚ := a / b

theorem cannot_obtain_fraction (k n : ℕ) :
  let f_start := fraction 5 8 in
  let f_target := fraction 3 5 in
  ∀ (a b : ℕ), 
    (a = 5 + k ∧ b = 8 + k) ∨ 
    (a = n * 5 ∧ b = n * 8) →
  fraction a b ≠ f_target :=
by
  let f_start := fraction 5 8
  let f_target := fraction 3 5
  assume a b h
  cases h with h1 h2
  -- Add your proof here
  · sorry
  · sorry

end cannot_obtain_fraction_l817_817103


namespace common_point_on_line_l817_817977

theorem common_point_on_line (A B C : Point) (O1 O2 O3 P : Point) (r : ℝ) (incircle circumcenter : Point) :
  (∀ i, dist Oi P = r) ∧
  (∀ i ≠ j, Oi ≠ Oj) ∧
  (∀ i, Oi lies on the angle bisector of ∠AOB) ∧
  (triangle ABC ∼ triangle O1O2O3) ∧
  (incircle = Incenter(A, B, C)) ∧
  (circumcenter = Circumcenter(A, B, C)) ∧
  (H = incircle ∧ P' = circumcenter) :
  P lies on the line connecting H and P' :=
sorry

end common_point_on_line_l817_817977


namespace probability_at_least_75_cents_l817_817151

theorem probability_at_least_75_cents (p n d q c50 : Prop) 
  (Hp : p = tt ∨ p = ff)
  (Hn : n = tt ∨ n = ff)
  (Hd : d = tt ∨ d = ff)
  (Hq : q = tt ∨ q = ff)
  (Hc50 : c50 = tt ∨ c50 = ff) :
  (1 / 2 : ℝ) = 
  ((if c50 = tt then (if q = tt then 1 else 0) else 0) + 
  (if c50 = tt then 2^3 else 0)) / 2^5 :=
by sorry

end probability_at_least_75_cents_l817_817151


namespace regular_triangular_pyramid_volume_l817_817171

theorem regular_triangular_pyramid_volume
  (l α : ℝ)
  (h_l_gt_0 : l > 0)
  (h_alpha_range : 0 < α ∧ α < π/2) :
  let volume := (l^3 * Real.sqrt 3 * Real.sin (2 * α) * Real.cos α) / 8 in
  volume = (l^3 * Real.sqrt 3 * Real.sin 2 * α * Real.cos α) / 8 :=
by
  sorry

end regular_triangular_pyramid_volume_l817_817171


namespace probability_odd_number_chosen_l817_817661

theorem probability_odd_number_chosen :
  (∑ n in finset.range 120, 
     if n % 2 = 1 then (if n ≤ 60 then (1/300) else (4/300)) else 0) = 1/2 :=
by sorry

end probability_odd_number_chosen_l817_817661


namespace polynomial_inequality_for_reciprocal_l817_817114

variable {R : Type*} [LinearOrderedField R] (P : R → R)

-- Assume polynomial with positive real coefficients
def is_positive_real_polynomial (P : R → R) : Prop :=
  ∃ (a : ℕ → R), (∀ i, 0 ≤ a i) ∧ (∃ n, P = λ x, ∑ i in finset.range (n + 1), a i * x^i)

-- The condition that P(1) holds
axiom P_at_1 {P : R → R} (h : is_positive_real_polynomial P) : 
  P (1 : R) ≥ 1

-- The main proof statement
theorem polynomial_inequality_for_reciprocal 
  (h : is_positive_real_polynomial P) 
  (hP1 : P_at_1 h) 
  (x : R) (hx : 0 < x) : 
  P (1 / x) ≥ 1 / P x :=
sorry

end polynomial_inequality_for_reciprocal_l817_817114


namespace cross_prod_double_l817_817937

variable {a b : ℝ} [Fintype ℝ]

theorem cross_prod_double {a b : ℝ^3} (h : a ⨯ b = ![-3, 7, 1]) : a ⨯ (2 • b) = ![-6, 14, 2] :=
by
  sorry

end cross_prod_double_l817_817937


namespace perimeter_of_triangle_l817_817448

-- Define the right triangle ABC with given properties
variables (A B C X Y Z W : Type)
noncomputable def triangle_ABC (A B C : Type) : Prop := 
  ∃ (AB AC BC : ℝ),
  AB = 10 ∧  -- Given hypotenuse AB = 10
  ∠ B C A = 60 ∧  -- Given angle ABC = 60 degrees
  ∠ B A C = 30 ∧  -- Derived angle BAC = 30 degrees
  ∠ A C B = 90 ∧  -- Given right angle at C
  BC = 5 * real.sqrt 3 ∧  -- Calculated side BC
  AC = 5 ∧  -- Calculated side AC
  √ (AB ^ 2) = AC ^ 2 + BC ^ 2  -- Pythagorean theorem

-- Define square properties and cyclic points
def squares_and_circle (A B C X Y Z W : Type) : Prop :=
  (X Y Z W are on circle) -- Represent points X, Y, Z, W being on a circle (requires further elaboration according to specific math)

-- Main statement to prove
theorem perimeter_of_triangle (A B C X Y Z W : Type) :
  triangle_ABC A B C ∧ squares_and_circle A B C X Y Z W →
  perimeter ABC = 15 + 5 * real.sqrt 3 :=
sorry

end perimeter_of_triangle_l817_817448


namespace stadium_ticket_price_l817_817583

theorem stadium_ticket_price
  (original_price : ℝ)
  (decrease_rate : ℝ)
  (increase_rate : ℝ)
  (new_price : ℝ) 
  (h1 : original_price = 400)
  (h2 : decrease_rate = 0.2)
  (h3 : increase_rate = 0.05) 
  (h4 : (original_price * (1 + increase_rate) / (1 - decrease_rate)) = new_price) :
  new_price = 525 := 
by
  -- Proof omitted for this task.
  sorry

end stadium_ticket_price_l817_817583


namespace space_diagonal_angle_is_approximately_seventy_degrees_l817_817577

-- The space_diagonal_angle function calculates the angle between the space diagonals of a cube

noncomputable def space_diagonal_angle (a : ℝ) : ℝ :=
  let diagonal_length := a * real.sqrt 3,
      cos_angle := (diagonal_length^2 + diagonal_length^2 - (a * real.sqrt 3)^2) / (2 * diagonal_length * diagonal_length)
  real.acos cos_angle

theorem space_diagonal_angle_is_approximately_seventy_degrees :
  abs (space_diagonal_angle 1 - 70.5288) < 0.001 := 
sorry

end space_diagonal_angle_is_approximately_seventy_degrees_l817_817577


namespace find_min_value_of_M_l817_817726

-- Define the elements of set A
def q0 (x y : ℝ) : ℝ := x * y
def q1 (x y : ℝ) : ℝ := x * y - x - y + 1
def q2 (x y : ℝ) : ℝ := x + y - 2 * x * y

-- Define the maximum of set A
def M (x y : ℝ) : ℝ := max (max (q0 x y) (q1 x y)) (q2 x y)

-- Define the conditions on x and y
def valid_pair (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ y ∧ y ≤ 1

-- State the proof problem
theorem find_min_value_of_M :
  ∃ c : ℝ, (∀ x y : ℝ, valid_pair x y → M x y ≥ c) ∧ (∀ ε > 0, ∃ x y : ℝ, valid_pair x y ∧ M x y < c + ε) ↔ c = 4 / 9 :=
sorry

end find_min_value_of_M_l817_817726


namespace count_two_digit_integers_remainder_3_l817_817863

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817863


namespace equal_distances_from_K_to_AB_and_CD_l817_817244

noncomputable def parallelogram (A B C D : Type) [affine_plane A] :=
parallel (affine_line.mk A B) (affine_line.mk C D) ∧
parallel (affine_line.mk B C) (affine_line.mk D A)

variables (A B C D B1 B2 C1 C2 K : Type) [affine_plane A] [affine_plane B] [affine_plane C]
          [affine_plane D] [affine_plane B1] [affine_plane B2] [affine_plane C1] [affine_plane C2] [affine_plane K]

-- Let A, B, C, and D form a parallelogram
axiom h1 : parallelogram A B C D

-- Circles passing through points A and D intersect lines AB, BD, AC, and CD respectively
axiom h2 : on_circle (circle.mk A D) B1
axiom h3 : on_circle (circle.mk A D) B2
axiom h4 : on_circle (circle.mk A D) C1
axiom h5 : on_circle (circle.mk A D) C2

-- Let K be the intersection of lines B1B2 and C1C2
axiom h6 : intersect (affine_line.mk B1 B2) (affine_line.mk C1 C2) = K

-- Prove that the distances from point K to lines AB and CD are equal.
theorem equal_distances_from_K_to_AB_and_CD :
  distance_from_point_to_line K (affine_line.mk A B) = distance_from_point_to_line K (affine_line.mk C D) :=
sorry

end equal_distances_from_K_to_AB_and_CD_l817_817244


namespace binomial_sum_squared_l817_817346

theorem binomial_sum_squared (n : ℕ) : 
  (∑ i in Finset.range (n + 1), i^2 * (Nat.choose n i)) = n * (n + 1) * 2^(n - 2) := 
sorry

end binomial_sum_squared_l817_817346


namespace sum_of_integers_from_neg15_to_5_l817_817696

theorem sum_of_integers_from_neg15_to_5 : 
  (∑ x in Finset.Icc (-15 : ℤ) 5, x) = -105 := 
by
  sorry

end sum_of_integers_from_neg15_to_5_l817_817696


namespace teachers_and_students_arrangements_l817_817598

theorem teachers_and_students_arrangements :
  ∀ (students teachers : List String), students.length = 3 → teachers.length = 2 →
  (∀ t1 t2 : String, t1 ∈ teachers → t2 ∈ teachers → t1 ≠ t2 → t1 < t2) →
  (∀ l : List String, l ~ (teachers.head! ++ teachers.tail!) :: (students.map id)) →
  teacherA_must_left_of_teacherB := 24 :=
begin
  sorry
end

end teachers_and_students_arrangements_l817_817598


namespace tangent_line_at_0_l817_817741

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x + Real.sin x

theorem tangent_line_at_0 :
  ∃ (m b : ℝ), ∀ (x : ℝ), f 0 = 1 ∧ (f' : ℝ → ℝ) 0 = 1 ∧ (f' x = Real.exp x + 2 * x - 1 + Real.cos x) ∧ 
  (m = 1) ∧ (b = (m * 0 + 1)) ∧ (∀ x : ℝ, y = m * x + b) :=
by
  sorry

end tangent_line_at_0_l817_817741


namespace frustum_has_two_parallel_surfaces_l817_817278

def has_parallel_surfaces : Type := {x // x = 0 ∨ ∃ (n : Nat), x = n}

def pyramid := (0 : has_parallel_surfaces)
def prism := (3 : has_parallel_surfaces)
def frustum := (2 : has_parallel_surfaces)
def cuboid := (3 : has_parallel_surfaces)

theorem frustum_has_two_parallel_surfaces :
  (∃ (x : has_parallel_surfaces), x = frustum ∧ x.1 = 2) ∧ 
  (frustum = pyramid ∨ 
   frustum = prism ∨ 
   frustum = cuboid → False) :=
  by sorry

end frustum_has_two_parallel_surfaces_l817_817278


namespace quadrilateral_same_area_l817_817245

variables (A B C D P Q O X Y Z T : Type) [affine_space ℝ A B C D P Q O X Y Z T]

-- Define midpoints P and Q
def midpoint (A B : A) : A := (A + B) / 2 -- This is a simplification for the type scenario
def P := midpoint B D
def Q := midpoint A C

-- Define intersection O of lines through P and Q parallel to opposite diagonals
def parallel (A B C D : A) : Prop := -- Parallel property placeholder
  sorry

def O := -- Intersection point definition placeholder
  sorry

-- Midpoints of the sides of the quadrilateral
def X := midpoint A B
def Y := midpoint B C
def Z := midpoint C D
def T := midpoint D A

-- Areas of the quadrilaterals formed
def area (A B C D : A) : ℝ := -- Area function placeholder
  sorry

-- Main theorem statement
theorem quadrilateral_same_area :
  area O X B Y = area O Y C Z ∧
  area O Y C Z = area O Z D T ∧
  area O Z D T = area O T A X :=
sorry

end quadrilateral_same_area_l817_817245


namespace average_speed_trip_l817_817671

variable (distance_XY : ℝ) (time_XY : ℝ) (time_YX : ℝ)
variables (total_distance total_time : ℝ)

def average_speed (d t : ℝ) : ℝ := d / t

theorem average_speed_trip : 
  distance_XY = 1000 ->
  time_XY = 10 ->
  time_YX = 4 ->
  total_distance = 2000 ->
  total_time = 14 ->
  abs (average_speed total_distance total_time - 142.86) < 0.01 :=
by 
  sorry

end average_speed_trip_l817_817671


namespace subgroup_in_center_l817_817495

-- Definitions corresponding to the conditions
variables {G : Type*} [Group G] 
variables {H : Subgroup G}

def Z (G : Type*) [Group G] := {a : G | ∀ x : G, a * x = x * a}

theorem subgroup_in_center
  (n : ℕ) (hn : 2 ≤ n)
  (p : ℕ) (hp : Nat.Prime p) (hp_dvd : p ∣ n)
  (hH : H.order = p)
  (unique_H : ∀ K : Subgroup G, K.order = p → K = H) : 
  H ≤ Z G :=
sorry

end subgroup_in_center_l817_817495


namespace positive_two_digit_integers_remainder_3_l817_817817

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817817


namespace count_two_digit_integers_remainder_3_div_7_l817_817905

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817905


namespace sum_of_coefficients_l817_817358

def polynomial := (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 - 6*x + 9

variables (s : ℕ → ℝ)
noncomputable def a := 6
noncomputable def b := 0
noncomputable def c := -25.67 / 6

axiom s_0_eq : s 0 = 4
axiom s_1_eq : s 1 = 6
axiom s_2_eq : s 2 = 14

axiom recurrence (k : ℕ) (hk : 3 ≤ k) : s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)

theorem sum_of_coefficients : a + b + c = -19.67 :=
sorry

end sum_of_coefficients_l817_817358


namespace total_bees_in_hive_at_end_of_7_days_l817_817640

-- Definitions of given conditions
def daily_hatch : Nat := 3000
def daily_loss : Nat := 900
def initial_bees : Nat := 12500
def days : Nat := 7
def queen_count : Nat := 1

-- Statement to prove
theorem total_bees_in_hive_at_end_of_7_days :
  initial_bees + daily_hatch * days - daily_loss * days + queen_count = 27201 := by
  sorry

end total_bees_in_hive_at_end_of_7_days_l817_817640


namespace ratio_of_cars_to_trucks_l817_817984

-- Definitions based on conditions
def total_vehicles : ℕ := 60
def trucks : ℕ := 20
def cars : ℕ := total_vehicles - trucks

-- Theorem to prove
theorem ratio_of_cars_to_trucks : (cars / trucks : ℚ) = 2 := by
  sorry

end ratio_of_cars_to_trucks_l817_817984


namespace team_win_percentage_remaining_l817_817272

theorem team_win_percentage_remaining (won_first_30: ℝ) (total_games: ℝ) (total_wins: ℝ)
  (h1: won_first_30 = 0.40 * 30)
  (h2: total_games = 120)
  (h3: total_wins = 0.70 * total_games) :
  (total_wins - won_first_30) / (total_games - 30) * 100 = 80 :=
by
  sorry


end team_win_percentage_remaining_l817_817272


namespace sum_of_digits_l817_817234

theorem sum_of_digits (a b : ℕ) (h1 : 10 * a + b + 10 * b + a = 202) (h2 : a < 10) (h3 : b < 10) :
  a + b = 12 :=
sorry

end sum_of_digits_l817_817234


namespace Bert_total_profit_is_14_90_l817_817688

-- Define the sales price for each item
def sales_price_barrel : ℝ := 90
def sales_price_tools : ℝ := 50
def sales_price_fertilizer : ℝ := 30

-- Define the tax rates for each item
def tax_rate_barrel : ℝ := 0.10
def tax_rate_tools : ℝ := 0.05
def tax_rate_fertilizer : ℝ := 0.12

-- Define the profit added per item
def profit_per_item : ℝ := 10

-- Define the tax amount for each item
def tax_barrel : ℝ := tax_rate_barrel * sales_price_barrel
def tax_tools : ℝ := tax_rate_tools * sales_price_tools
def tax_fertilizer : ℝ := tax_rate_fertilizer * sales_price_fertilizer

-- Define the cost price for each item
def cost_price_barrel : ℝ := sales_price_barrel - profit_per_item
def cost_price_tools : ℝ := sales_price_tools - profit_per_item
def cost_price_fertilizer : ℝ := sales_price_fertilizer - profit_per_item

-- Define the profit for each item
def profit_barrel : ℝ := sales_price_barrel - tax_barrel - cost_price_barrel
def profit_tools : ℝ := sales_price_tools - tax_tools - cost_price_tools
def profit_fertilizer : ℝ := sales_price_fertilizer - tax_fertilizer - cost_price_fertilizer

-- Define the total profit
def total_profit : ℝ := profit_barrel + profit_tools + profit_fertilizer

-- Assert the total profit is $14.90
theorem Bert_total_profit_is_14_90 : total_profit = 14.90 :=
by
  -- Omitted proof
  sorry

end Bert_total_profit_is_14_90_l817_817688


namespace find_number_l817_817056

theorem find_number (some_number : ℤ) : 45 - (28 - (some_number - (15 - 19))) = 58 ↔ some_number = 37 := 
by 
  sorry

end find_number_l817_817056


namespace work_completion_days_original_people_work_days_l817_817555

variable (P W D : ℕ)

theorem work_completion_days (h : 2 * P * 4 = W / 2) : P * 16 = W :=
by
  have h1 : W = 16 * P := by
    linarith
  rw h1
  linarith

theorem original_people_work_days (h : 2 * P * 4 = W / 2) : D = 16 :=
by
  have h1 : W = 16 * P := work_completion_days P W h
  have h2 : P * D = W := by
    linarith
  rw h1 at h2
  have h3 : P * D = 16 * P := by
    linarith
  exact nat.eq_of_mul_eq_mul_left (nat.pos_of_ne_zero (P.ne_zero h2)) h3

end work_completion_days_original_people_work_days_l817_817555


namespace exists_points_B_and_C_on_circle_l817_817740

open Classical

noncomputable theory

-- Definitions
def Circle := { p : ℝ × ℝ // p.1^2 + p.2^2 = 1 }

variable (A O : ℝ × ℝ) [Circle A]

-- Conditions
def is_incenter (O A B C : ℝ × ℝ) : Prop :=
  ∃ (I : ℝ × ℝ) (r : ℝ), I = O ∧ 
  dist I (line_segment ℝ A B) = r ∧ 
  dist I (line_segment ℝ B C) = r ∧ 
  dist I (line_segment ℝ C A) = r

-- Proof statement
theorem exists_points_B_and_C_on_circle
  (A O : ℝ × ℝ) (hA : A.1^2 + A.2^2 = 1) (hO_inside_circle : ∃ r, 0 < r ∧ dist O (0,0) < r) :
  ∃ (B C : ℝ × ℝ), B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧ is_incenter O A B C :=
sorry

end exists_points_B_and_C_on_circle_l817_817740


namespace oil_depth_l817_817651

-- Let r denote the radius of the tank
def radius (d : ℝ) := d / 2

-- The condition that interior diameter of the tank is 8 feet
def diameter : ℝ := 8
def r : ℝ := radius diameter

-- Surface area of the oil inside the tank
def surface_area : ℝ := 32

-- The length of the tank is 12 feet
def length : ℝ := 12

-- Define the given solution steps, equations, and final depth value
def chord_length_surface_area (s_area l : ℝ) := s_area / l
def chord_length := chord_length_surface_area surface_area length

def equation (h : ℝ) (r : ℝ) (c : ℝ) : Prop :=
  c = 2 * real.sqrt ((2 * r * h) - h * h)

def depth_quadratic_eq (h : ℝ) (r : ℝ) :=
  h * h - 2 * r * h + c * c / 4 = 0

-- Prove that the depth of the oil is 4 feet under given conditions
theorem oil_depth : 
  ∃ (h : ℝ), equation h r chord_length ∧ h = 4 := 
by
  sorry

end oil_depth_l817_817651


namespace flowers_per_bouquet_l817_817712

theorem flowers_per_bouquet (initial_flowers wilted_flowers remaining_bouquets total_flowers_per_bouquet : ℕ) :
  initial_flowers = 88 → 
  wilted_flowers = 48 → 
  remaining_bouquets = 8 → 
  total_flowers_per_bouquet = (initial_flowers - wilted_flowers) / remaining_bouquets → 
  total_flowers_per_bouquet = 5 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  norm_num at h4
  exact h4

end flowers_per_bouquet_l817_817712


namespace count_two_digit_remainders_l817_817805

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817805


namespace Murtha_pebble_collection_l817_817532

def sum_of_first_n_natural_numbers (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem Murtha_pebble_collection : sum_of_first_n_natural_numbers 20 = 210 := by
  sorry

end Murtha_pebble_collection_l817_817532


namespace proof_problem_l817_817989

noncomputable def problem_statement :=
  ∃ (a : ℕ → ℕ), (∀ n, a n < a (n + 1)) ∧
    (∃ M > 0, ∀ n, 0 < a (n + 1) - a n ∧ a (n + 1) - a n < M * a n^(5/8)) ∧
    (∃ A : ℝ, ∀ k, ∃ n, a n = floor(A ^ (3 ^ k)) )

theorem proof_problem : problem_statement := sorry

end proof_problem_l817_817989


namespace longest_distance_after_folding_is_sqrt_5_l817_817192

noncomputable def fold_paper_distance : ℝ :=
  let side_length := 2
  let center := (1 : ℝ, 1) -- since the paper is a square with side length 2
  let F := (0 : ℝ, 2)
  let O := (0 : ℝ, 0)
  sqrt ((1 - 0)^2 + (1 - 2)^2)

theorem longest_distance_after_folding_is_sqrt_5 :
  fold_paper_distance = sqrt 5 :=
by
  sorry

end longest_distance_after_folding_is_sqrt_5_l817_817192


namespace count_interesting_numbers_l817_817830

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817830


namespace find_length_of_AB_l817_817411

noncomputable def length_of_AB (S : ℝ) (BC : ℝ) (C : ℝ) : ℝ :=
  let AC := (2 * S) / (BC * Real.sin C)
  Real.sqrt(BC^2 + AC^2 - 2 * BC * AC * Real.cos C)

theorem find_length_of_AB : length_of_AB (Real.sqrt 3) 2 (Real.pi / 3) = 2 :=
by
  sorry

end find_length_of_AB_l817_817411


namespace sum_g_f_x_l817_817511

noncomputable def f : ℤ → ℤ 
| -2 := -1
| -1 := 1
|  0 := 3
|  3 := 5
|  _ := 0  -- undefined cases, just return 0

noncomputable def g : ℤ → ℤ 
| (-1) := -1 + 2
|  1 := 1 + 2
|  3 := 3 + 2
|  5 := 5 + 2
|  _ := 0  -- undefined cases, just return 0

def fg_values := [g (f (-2)), g (f (-1)), g (f 0), g (f 3)]

theorem sum_g_f_x : fg_values.sum = 16 := by
  sorry

end sum_g_f_x_l817_817511


namespace volume_of_apple_cider_in_pot_l817_817257

-- Given conditions
def height_of_pot : ℝ := 9
def diameter_of_pot : ℝ := 4
def ratio_apple_cider : ℝ := 2
def ratio_water : ℝ := 5
def ratio_total_parts : ℝ := ratio_apple_cider + ratio_water
def fraction_two_thirds : ℝ := 2 / 3

-- Statement to prove
theorem volume_of_apple_cider_in_pot :
  (π * (diameter_of_pot / 2)^2 * (fraction_two_thirds * height_of_pot)) * (ratio_apple_cider / ratio_total_parts) = (48 * π / 7) :=
by
  sorry

end volume_of_apple_cider_in_pot_l817_817257


namespace num_rem_three_by_seven_l817_817862

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817862


namespace team_selection_ways_l817_817130

theorem team_selection_ways : ∃ (ways : ℕ), ways = (nat.choose 10 5) * (nat.choose 12 3) ∧ ways = 55440 :=
by
  use (nat.choose 10 5) * (nat.choose 12 3)
  sorry

end team_selection_ways_l817_817130


namespace range_of_k_l817_817035

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, f (-x^2 + 3 * x) + f (x - 2 * k) ≤ 0) ↔ k ≥ 2 :=
by
  sorry

end range_of_k_l817_817035


namespace gel_pen_price_relation_b_l817_817297

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l817_817297


namespace gasoline_balance_l817_817158

theorem gasoline_balance (x : ℝ) (y : ℝ) : 
  (price_per_liter : ℝ) (initial_amount : ℝ) 
  (price_per_liter = 7.92) ∧ (initial_amount = 1000) 
  → y = initial_amount - price_per_liter * x :=
begin
  intros h,
  cases h with h₁ h₂,
  rw h₁,
  rw h₂,
  simp,
end

end gasoline_balance_l817_817158


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817909

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817909


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817337

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817337


namespace count_two_digit_integers_remainder_3_l817_817820

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817820


namespace statements_correct_l817_817091

variable (a b c : ℝ)

-- Definitions related to trigonometric relationships in a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Statements as Lean propositions
def statement1 (a b c : ℝ) : Prop := a^2 + b^2 < c^2 → C > π / 2
def statement3 (a b c : ℝ) : Prop := a^3 + b^3 = c^3 → C > π / 2
def statement5 (a b c : ℝ) : Prop := (a^2 + b^2) * c^2 < 2 * a^2 * b^2 → C < π / 3

-- Combined theorem with the conditions provided
theorem statements_correct (h1 : statement1 a b c) (h3 : statement3 a b c) (h5 : statement5 a b c) : 
  is_triangle a b c → 
  (statement1 a b c ∧ statement3 a b c ∧ statement5 a b c) := 
sorry

end statements_correct_l817_817091


namespace prob_eventA_prob_eventB_l817_817692

open Classical

noncomputable def totalBalls : ℕ := 10

noncomputable def eventA : set ℕ := {1, 2, 3}
noncomputable def eventB : set ℕ := {3, 6, 9}

noncomputable def prob (s : set ℕ) (total : ℕ) : ℚ := s.card / total

theorem prob_eventA :
  prob eventA totalBalls = 3 / 10 := by
  sorry

theorem prob_eventB :
  prob eventB totalBalls = 3 / 10 := by
  sorry

end prob_eventA_prob_eventB_l817_817692


namespace product_of_real_parts_l817_817374

theorem product_of_real_parts (x : ℂ) (h : x^3 + 3 * x^2 = Complex.i) :
  ((x.re) * (x.re) * (x.re)) = 0 :=
sorry

end product_of_real_parts_l817_817374


namespace f_f_3_eq_neg_3_l817_817774

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 3 then 2^x - 1 else x - 5

theorem f_f_3_eq_neg_3 
  (h : is_odd_function f) : 
  f (f 3) = -3 :=
by
  sorry

end f_f_3_eq_neg_3_l817_817774


namespace min_distance_l817_817975

def circle (θ : ℝ) : ℝ := 2 * Real.cos θ

def line (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + π / 4) = -Real.sqrt 2 / 2

theorem min_distance (θ : ℝ) : ∃ ρ, ρ = circle θ → 
  ∀ θ₀, ∃ ρ₀, line ρ₀ θ₀ → ∃ d, d = Real.sqrt 2 - 1 :=
by
  sorry

end min_distance_l817_817975


namespace find_apartment_number_l817_817679

open Nat

def is_apartment_number (x a b : ℕ) : Prop :=
  x = 10 * a + b ∧ x = 17 * b

theorem find_apartment_number : ∃ x a b : ℕ, is_apartment_number x a b ∧ x = 85 :=
by
  sorry

end find_apartment_number_l817_817679


namespace part1_part2_l817_817429

variables {R : Type*} [Real R]
variables (a b : R → R → R → R) (t : R)

-- Conditions
def condition1 : Prop := norm a = 3
def condition2 : Prop := norm b = 1
def condition3 : Prop := real.angle a b = π / 3

-- Question I
theorem part1 
  (h1 : condition1 a)
  (h2 : condition2 b)
  (h3 : condition3 a b) : norm (a + 3 * b) = 3 * sqrt 3 := 
sorry

-- Question II
theorem part2 
  (h1 : condition1 a)
  (h2 : condition2 b)
  (h3 : condition3 a b)
  (h4 : (a + 2 * b) ⬝ (t * a + 2 * b) = 0) : t = -7 / 12 :=
sorry

end part1_part2_l817_817429


namespace smallest_positive_integer_for_positive_term_l817_817084

theorem smallest_positive_integer_for_positive_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : a 1 = -1) 
  (hS : S 19 = 0) 
  (hS_def : ∀ n, S n = n * (a 1 + a n) / 2) 
  (h_arith : ∀ n, n > 0 → a n = a 1 + (n-1) * (a 2 - a 1)) :
  ∃ n : ℕ, n > 0 ∧ a n > 0 ∧ ∀ m : ℕ, 0 < m ∧ m < n → a m ≤ 0 :=
begin
  sorry
end

end smallest_positive_integer_for_positive_term_l817_817084


namespace proof_problem_l817_817743

-- Definitions of lines, planes, and their relationships.
variable (Line Plane : Type)
variable (perpendicular parallel skew : Line → Plane → Prop)
variable (perpendicular_lines parallel_lines : Line → Line → Prop)

-- Given conditions
variable (l : Line) (m : Line) (α : Plane) (β : Plane)
variable (hl : perpendicular l α) (hm : skew m β)

-- The propositions
def prop1 : Prop := parallel α β → perpendicular_lines l m
def prop3 : Prop := parallel_lines l m → perpendicular α β

theorem proof_problem : prop1 ∧ prop3 := by
  -- Proof details are omitted as per the instructions.
  sorry

end proof_problem_l817_817743


namespace combined_series_sum_l817_817523

noncomputable def sum_combined_series (n : ℕ) : ℝ :=
15 * ((4 / 3) ^ n - 1) - n * (n + 1)

theorem combined_series_sum (n : ℕ) :
  let terms := (5, -2, 20 / 3, -26 / 3, 56 / 9, -74 / 9, ...)
  (∑ k in finset.range n, terms k) = 15 * ((4 / 3) ^ n - 1) - n * (n + 1) :=
sorry

end combined_series_sum_l817_817523


namespace function_no_real_zeros_l817_817053

variable (a b c : ℝ)

-- Conditions: a, b, c form a geometric sequence and ac > 0
def geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c
def positive_product (a c : ℝ) : Prop := a * c > 0

theorem function_no_real_zeros (h_geom : geometric_sequence a b c) (h_pos : positive_product a c) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := 
by
  sorry

end function_no_real_zeros_l817_817053


namespace find_correct_function_l817_817223

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

noncomputable def f_A (x : ℝ) : ℝ := abs (sin x)
noncomputable def f_B (x : ℝ) : ℝ := log ((2 - x) / (2 + x))
noncomputable def f_C (x : ℝ) : ℝ := (1 / 2) * (exp x - exp (-x))
noncomputable def f_D (x : ℝ) : ℝ := log (sqrt (x * x + 1) - x)

theorem find_correct_function :
  (is_odd f_C ∧ is_monotonically_increasing f_C (-1) 1) ∧
  ¬(is_odd f_A ∧ is_monotonically_increasing f_A (-1) 1) ∧
  ¬(is_odd f_B ∧ is_monotonically_increasing f_B (-1) 1) ∧
  ¬(is_odd f_D ∧ is_monotonically_increasing f_D (-1) 1) :=
by
  sorry

end find_correct_function_l817_817223


namespace most_economical_speed_and_cost_l817_817770

open Real

theorem most_economical_speed_and_cost :
  ∀ (x : ℝ),
  (120:ℝ) / x * 36 + (120:ℝ) / x * 6 * (4 + x^2 / 360) = ((7200:ℝ) / x) + 2 * x → 
  50 ≤ x ∧ x ≤ 100 → 
  (∀ v : ℝ, (50 ≤ v ∧ v ≤ 100) → 
  (120 / v * 36 + 120 / v * 6 * (4 + v^2 / 360) ≤ 120 / x * 36 + 120 / x * 6 * (4 + x^2 / 360)) ) → 
  x = 60 → 
  (120 / x * 36 + 120 / x * 6 * (4 + x^2 / 360) = 240) :=
by
  intros x hx bounds min_cost opt_speed
  sorry

end most_economical_speed_and_cost_l817_817770


namespace new_coordinates_sum_31_l817_817505

theorem new_coordinates_sum_31 :
  let A : Point := (24, -1)
  let B : Point := (5, 6)
  let P : Point := (-14, 27)
  let slope_L := 5 / 12
  Line_through_point_slope L A slope_L ∧
  Line_through_point_perpendicular M B L →
  in_new_coordinate_system L M A B P (α, β) →
  α + β = 31 :=
by
  sorry

end new_coordinates_sum_31_l817_817505


namespace unknown_number_value_l817_817940

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end unknown_number_value_l817_817940


namespace car_travel_distance_l817_817253

theorem car_travel_distance
  (b t : ℝ) : (t ≠ 0) → (b / 4) / t * (120 * 3 * 1760) = (158400 * b) / t :=
by
  intro h_neq
  have h1: (120:ℝ) = 2 * 60 := by norm_num  
  have h2: (3 * (1760:ℝ)) = 5280 := by norm_num
  have h3: (30:ℝ) * 5280 = 158400 := by norm_num
  rw [← h1, mul_assoc, ← h2, mul_comm (30 * b), mul_assoc, h3]
  exact div_eq_div_of_mul_eq_mul (by linarith) h_neq

end car_travel_distance_l817_817253


namespace BD_parallel_CP_l817_817092

noncomputable theory

-- Define to state in a triangle ABC
variables {A B C D E P : Point}
variables [triangle : Triangle A B C]

-- Introduce conditions given in the problem
-- Assume AB > AC
axiom condition_AB_AC : B.dist A > C.dist A

-- Assume D is the intersection of the angle bisector of ∠ABC with AC
axiom D_is_bisector : angle_bisector B A C D

-- Assume E is the intersection of the angle bisector of ∠ACB with AB
axiom E_is_bisector : angle_bisector C A B E

-- Assume P is the intersection of the tangent from A to the circumcircle of ΔABC with the extension of ED
axiom P_is_tangent_extension : tangent_from_A_intersection A B C D E P

-- Given AP = BC
axiom AP_eq_BC : A.dist P = B.dist C

-- Final statement to prove
theorem BD_parallel_CP : parallel (line B D) (line C P) :=
sorry

end BD_parallel_CP_l817_817092


namespace ratio_of_magnets_is_half_l817_817672

-- Let's define the given conditions
def totalAdamMagnets : ℕ := 18
def totalPeterMagnets : ℕ := 24
def fractionGivenAway : ℚ := 1 / 3

-- Define Adam's remaining magnets after giving away a third of it
def adamRemainingMagnets : ℕ := totalAdamMagnets - (totalAdamMagnets * fractionGivenAway)

-- Define the problem: the ratio of Adam's remaining magnets to Peter's magnets is 1/2
theorem ratio_of_magnets_is_half : (adamRemainingMagnets : ℚ) / totalPeterMagnets = 1 / 2 :=
by
  -- specify proof here
  sorry

end ratio_of_magnets_is_half_l817_817672


namespace prob_at_least_one_oct_side_l817_817384

-- Define a probability question for a regular octagon
noncomputable def probability_trig_oct : ℚ :=
  let totalTriangles := (Finset.powerset_len 3 (Finset.univ : Finset (Fin 8))).card in
  let favorableOutcomes := 40 in
  favorableOutcomes / totalTriangles

-- Theorem statement ensuring the solution
theorem prob_at_least_one_oct_side :
  probability_trig_oct = 5 / 7 :=
by sorry

end prob_at_least_one_oct_side_l817_817384


namespace g_g_g_3_eq_71_l817_817520

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2 * n - 1 else 2 * n + 5

theorem g_g_g_3_eq_71 : g (g (g 3)) = 71 := 
by
  sorry

end g_g_g_3_eq_71_l817_817520


namespace num_rem_three_by_seven_l817_817855

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817855


namespace checkerboard_probability_l817_817132

theorem checkerboard_probability :
  let num_squares := 10 * 10,
      num_perimeter_squares := 10 + 10 + (10 - 2) + (10 - 2),
      num_inner_squares := num_squares - num_perimeter_squares,
      probability := num_inner_squares / num_squares in
  probability = 16 / 25 :=
by 
  sorry

end checkerboard_probability_l817_817132


namespace measure_angle_CAD_l817_817088

-- Definitions based on the conditions
def is_parallel (A B C D : Point) : Prop := ∃ m : ℝ, ∀ x : ℝ, ∀ y : ℝ, y = m * x
def distance (A B : Point) : ℝ := sorry -- Placeholder for distance function
def is_trapezoid (A B C D : Point) : Prop := sorry -- Placeholder for trapezoid definition

-- The problem definition
noncomputable def angle_CAD := sorry -- Placeholder for angle measurement

-- The theorem stating the proof problem
theorem measure_angle_CAD (A B C D : Point)
  (h1 : is_parallel A B C D)
  (h2 : distance A D = 1 ∧ distance A B = 1 ∧ distance B C = 1)
  (h3 : distance C D = 2) :
  angle_CAD A C D = 90 :=
sorry

end measure_angle_CAD_l817_817088


namespace triangle_area_l817_817522

theorem triangle_area
  (A B C : ℝ × ℝ)
  (h_right_angle : ∃ C : ℝ × ℝ, right_triangle A B C)
  (h_length_ab : (dist A B) = 60)
  (h_median_A : ∃ m : ℝ × ℝ, line_contains_point m A ∧ line_slope m = 1)
  (h_median_B : ∃ m : ℝ × ℝ, line_contains_point m B ∧ line_slope m = 2)
  : area_triangle A B C = 400 :=
sorry

end triangle_area_l817_817522


namespace angle_in_second_quadrant_l817_817249

def Quadrant : Type := ℕ
def SecondQuadrant : Quadrant := 2

def angle := -225

theorem angle_in_second_quadrant : Quadrant := SecondQuadrant :=
by
-- Proof required
sorry

end angle_in_second_quadrant_l817_817249


namespace multiplier_for_ab_to_equal_1800_l817_817934

variable (a b m : ℝ)
variable (h1 : 4 * a = 30)
variable (h2 : 5 * b = 30)
variable (h3 : a * b = 45)
variable (h4 : m * (a * b) = 1800)

theorem multiplier_for_ab_to_equal_1800 (h1 : 4 * a = 30) (h2 : 5 * b = 30) (h3 : a * b = 45) (h4 : m * (a * b) = 1800) :
  m = 40 :=
sorry

end multiplier_for_ab_to_equal_1800_l817_817934


namespace sum_of_roots_eq_a_l817_817433

variables {a b : ℝ}
hypothesis ha : a ≠ 0
hypothesis hb : b ≠ 0
hypothesis hroots : ∃ (r1 r2 : ℝ), r1 = a ∧ r2 = b ∧ (λ x, x^2 - a*x + 3*b = 0)

theorem sum_of_roots_eq_a : a + b = a := by
  sorry


end sum_of_roots_eq_a_l817_817433


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817887

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817887


namespace trajectory_of_P_is_ellipse_l817_817404

open Set

/-- Given a circle F1: (x+2)^2 + y^2 = 36, fixed point F2(2,0), 
and A is a moving point on circle F1. The perpendicular bisector 
of segment F2A intersects the radius F1A at point P.
We want to determine the equation of the trajectory C of point P. -/
theorem trajectory_of_P_is_ellipse :
  let F1 := (x^2 + y^2 = 36)
  let F2 := (2, 0)
  let A_moving := ∃ (A : ℝ × ℝ), (A.1 + 2)^2 + A.2^2 = 36
  let |.PA| := distance P A
  let P := intersection_of_perpendicular_bisector_and_radius
  (distance P (2, 0) + distance P F1 = 6) 
  → (trajectory_of_P P = \frac{x^2}{9} + \frac{y^2}{5} = 1) := sorry

end trajectory_of_P_is_ellipse_l817_817404


namespace hyperbola_eccentricity_l817_817782

open Real

/-- Given the hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 2 - x^2 / 8 = 1

/-- Prove the eccentricity of the given hyperbola -/
theorem hyperbola_eccentricity (x y : ℝ) (h : hyperbola_equation x y) : 
  ∃ e : ℝ, e = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l817_817782


namespace infinitely_many_pairs_l817_817539

theorem infinitely_many_pairs : ∀ b : ℕ, ∃ a : ℕ, 2019 < 2^a / 3^b ∧ 2^a / 3^b < 2020 := 
by
  sorry

end infinitely_many_pairs_l817_817539


namespace num_rem_three_by_seven_l817_817857

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817857


namespace product_of_digits_base8_9876_l817_817612

-- Definition of the base 8 representation of the decimal number 9876
def to_base8 (n : ℕ) : list ℕ :=
  if n = 9876 then [2, 3, 2, 2, 4]
  else []

-- Definition of the product of the digits in the base 8 representation
def digit_product (digits : list ℕ) : ℕ :=
  list.prod digits

-- Main theorem
theorem product_of_digits_base8_9876 : digit_product (to_base8 9876) = 96 :=
  sorry

end product_of_digits_base8_9876_l817_817612


namespace geoff_tuesday_multiple_l817_817731

variable (monday_spending : ℝ) (tuesday_multiple : ℝ) (total_spending : ℝ)

-- Given conditions
def geoff_conditions (monday_spending tuesday_multiple total_spending : ℝ) : Prop :=
  monday_spending = 60 ∧
  (tuesday_multiple * monday_spending) + (5 * monday_spending) + monday_spending = total_spending ∧
  total_spending = 600

-- Proof goal
theorem geoff_tuesday_multiple (monday_spending tuesday_multiple total_spending : ℝ)
  (h : geoff_conditions monday_spending tuesday_multiple total_spending) : 
  tuesday_multiple = 4 :=
by
  sorry

end geoff_tuesday_multiple_l817_817731


namespace gel_pen_ratio_l817_817308

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l817_817308


namespace integers_congruent_to_3_mod_7_count_integers_congruent_to_3_mod_7_l817_817794

theorem integers_congruent_to_3_mod_7 (x : ℕ) :
  (∃ n : ℕ, x = 7 * n + 3 ∧ 1 ≤ x ∧ x ≤ 300) ↔ 1 ≤ x ∧ x ≤ 300 ∧ x % 7 = 3 :=
begin
  sorry
end

theorem count_integers_congruent_to_3_mod_7 :
  {x | 1 ≤ x ∧ x ≤ 300 ∧ x % 7 = 3}.finite.to_finset.card = 43 :=
begin
  sorry
end

end integers_congruent_to_3_mod_7_count_integers_congruent_to_3_mod_7_l817_817794


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817914

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817914


namespace largest_kappa_l817_817715

theorem largest_kappa (a b c d : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : 0 ≤ d) (h_eq : a^2 + d^2 = b^2 + c^2) :
  a^2 + b^2 + c^2 + d^2 ≥ a * c + 2 * b * d + a * d :=
by 
  sorry

example : ∃ (κ : ℝ), (∀ (a b c d : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a^2 + d^2 = b^2 + c^2 → a^2 + b^2 + c^2 + d^2 ≥ a * c + κ * b * d + a * d) ∧ κ = 2 :=
by
  use 2
  intros a b c d h₁ h₂ h₃ h₄ h_eq
  exact largest_kappa a b c d h₁ h₂ h₃ h₄ h_eq

end largest_kappa_l817_817715


namespace area_APEG_is_18_l817_817199

noncomputable def length_ABCD : ℝ := 8
noncomputable def length_BEFG : ℝ := 6

def area_of_APEG : ℝ :=
  let DG := length_BEFG
  let GE := length_ABCD
  let area_DGE := (1 / 2) * DG * GE
  let area_BGE := (1 / 2) * length_BEFG * length_BEFG
  area_DGE - area_BGE

theorem area_APEG_is_18 :
  area_of_APEG = 18 :=
by
  sorry

end area_APEG_is_18_l817_817199


namespace positive_two_digit_integers_remainder_3_l817_817814

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817814


namespace balls_into_boxes_problem_l817_817730

theorem balls_into_boxes_problem :
  ∃ (n : ℕ), n = 144 ∧ ∃ (balls : Fin 4 → ℕ), 
  (∃ (boxes : Fin 4 → Fin 4), 
    (∀ (b : Fin 4), boxes b < 4 ∧ boxes b ≠ b) ∧ 
    (∃! (empty_box : Fin 4), ∀ (b : Fin 4), (boxes b = empty_box) → false)) := 
by
  sorry

end balls_into_boxes_problem_l817_817730


namespace count_four_digit_numbers_divisible_by_25_and_sum_of_digits_divisible_by_3_l817_817046

def is_four_digits (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_divisible_by_25 (n : ℕ) : Prop := n % 25 = 0
def digits_sum_divisible_by_3 (n : ℕ) : Prop := (n.digits 10).sum % 3 = 0

noncomputable def count_valid_four_digit_numbers : ℕ :=
  let valid_numbers := {n : ℕ | is_four_digits n ∧ is_divisible_by_25 n ∧ digits_sum_divisible_by_3 n} in
  valid_numbers.to_finset.card

theorem count_four_digit_numbers_divisible_by_25_and_sum_of_digits_divisible_by_3 :
  count_valid_four_digit_numbers = -- insert the final count here (let's denote it as X) :=
sorry

end count_four_digit_numbers_divisible_by_25_and_sum_of_digits_divisible_by_3_l817_817046


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817891

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817891


namespace find_k_l817_817628

theorem find_k (k : ℝ) (h : 32 / k = 4) : k = 8 := sorry

end find_k_l817_817628


namespace count_two_digit_integers_remainder_3_div_7_l817_817906

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817906


namespace admission_price_for_adults_l817_817185

variable {A : ℝ}

def number_of_people : ℕ := 610
def total_receipts : ℝ := 960
def children_attended : ℕ := 260
def child_ticket_price : ℝ := 1

def number_of_adults : ℕ := number_of_people - children_attended := by decide

def amount_collected_from_children : ℝ := children_attended * child_ticket_price
def amount_collected_from_adults : ℝ := total_receipts - amount_collected_from_children

def price_for_adult_tickets : ℝ := amount_collected_from_adults / number_of_adults

theorem admission_price_for_adults : A = 2 := by 
  -- show that the admission price for adults A is 2 given the conditions
  sorry

end admission_price_for_adults_l817_817185


namespace gel_pen_price_relation_b_l817_817292

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l817_817292


namespace gel_pen_ratio_l817_817313

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l817_817313


namespace expected_area_ratio_value_l817_817134

variable (ABC : Type) [NormedAddCommGroup ABC] [NormedSpace ℝ ABC]

-- Definitions based on the problem conditions
def P_division : ℝ := 1 / 5 -- AP:PB = 1:4 corresponds to AP = 1/5 of AB
def Q_division : ℝ := 3 / 4 -- AQ:QC = 3:1 corresponds to AQ = 3/4 of AC

-- The point M is randomly chosen on BC
-- We can represent the random variable M as a point on the line segment BC parametrized by t in [0,1]
def M (t : ℝ) (BC : ABC) : ABC := sorry -- definition of M depends on t and BC. 

-- Function to calculate the ratio of areas of triangles
def area_ratio (ABC PQM : ABC) : ℝ := sorry -- function to calculate the area ratio 

-- Statement of mathematical expectation
noncomputable def expected_area_ratio (ABC PQM : ABC) : ℝ :=
  sorry

-- Final expected value
theorem expected_area_ratio_value (ABC PQM : ABC) : expected_area_ratio ABC PQM = 13/40 :=
by 
  sorry

end expected_area_ratio_value_l817_817134


namespace necessary_but_not_sufficient_condition_l817_817021

theorem necessary_but_not_sufficient_condition 
  {k : ℝ} (h : 4 ≤ k ∧ k < 5) :
  ∀ x y : ℝ, (x^2 / (k - 5) + y^2 / (3 - k)) = -1 → "necessary_but_not_sufficient" :=
begin
  sorry
end

end necessary_but_not_sufficient_condition_l817_817021


namespace price_ratio_l817_817343

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l817_817343


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817912

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817912


namespace train_travel_time_l817_817273

theorem train_travel_time
  (a : ℝ) (s : ℝ) (t : ℝ)
  (ha : a = 3)
  (hs : s = 27)
  (h0 : ∀ t, 0 ≤ t) :
  t = Real.sqrt 18 :=
by
  sorry

end train_travel_time_l817_817273


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817888

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817888


namespace gel_pen_price_relation_b_l817_817295

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l817_817295


namespace max_earth_to_sun_distance_l817_817981

-- Define the semi-major axis a and semi-focal distance c
def semi_major_axis : ℝ := 1.5 * 10^8
def semi_focal_distance : ℝ := 3 * 10^6

-- Define the maximum distance from the Earth to the Sun
def max_distance (a c : ℝ) : ℝ := a + c

-- Define the Lean statement to be proved
theorem max_earth_to_sun_distance :
  max_distance semi_major_axis semi_focal_distance = 1.53 * 10^8 :=
by
  -- skipping the proof for now
  sorry

end max_earth_to_sun_distance_l817_817981


namespace radius_of_stationary_tank_l817_817260

theorem radius_of_stationary_tank 
  (h_stationary : 25)
  (r_truck : 5)
  (h_truck : 10)
  (drop_level : 0.025)
  (V_truck : Float) :
  V_truck = Float.pi * r_truck^2 * h_truck →
  (Float.pi * R^2 * drop_level = V_truck) →
  R = 100 :=
by
  sorry

end radius_of_stationary_tank_l817_817260


namespace people_in_room_proof_l817_817599

-- Definitions corresponding to the problem conditions
def people_in_room (total_people : ℕ) : ℕ := total_people
def seated_people (total_people : ℕ) : ℕ := (3 * total_people / 5)
def total_chairs (total_people : ℕ) : ℕ := (3 * (5 * people_in_room total_people) / 2 / 5 + 8)
def empty_chairs : ℕ := 8
def occupied_chairs (total_people : ℕ) : ℕ := (2 * total_chairs total_people / 3)

-- Proving that there are 27 people in the room
theorem people_in_room_proof (total_chairs : ℕ) :
  (seated_people 27 = 2 * total_chairs / 3) ∧ 
  (8 = total_chairs - 2 * total_chairs / 3) → 
  people_in_room 27 = 27 :=
by
  sorry

end people_in_room_proof_l817_817599


namespace determine_n_l817_817620

noncomputable def P : ℤ → ℤ := sorry

theorem determine_n (n : ℕ) (P : ℤ → ℤ)
  (h_deg : ∀ x : ℤ, P x = 2 ∨ P x = 1 ∨ P x = 0)
  (h0 : ∀ k : ℕ, k ≤ n → P (3 * k) = 2)
  (h1 : ∀ k : ℕ, k < n → P (3 * k + 1) = 1)
  (h2 : ∀ k : ℕ, k < n → P (3 * k + 2) = 0)
  (h_f : P (3 * n + 1) = 730) :
  n = 4 := 
sorry

end determine_n_l817_817620


namespace total_profit_l817_817645

theorem total_profit (A B : ℕ) (A_initial B_initial A_updated B_updated : ℕ) (t_initial t_updated : ℕ) (A_share : ℕ) : 
    A = 3000 → B = 4000 → t_initial = 8 → t_updated = 12 - t_initial →
    A_updated = A - 1000 → B_updated = B + 1000 →
    A_share = 288 →
    ∃ P, 8 * A + 4 * A_updated = 32000 ∧ 8 * B + 4 * B_updated = 52000 ∧ (A_share * 21 = 8 * P) ∧ P = 756 :=
begin
  intros hA hB ht_initial ht_updated hA_updated hB_updated hA_share,
  use 756,
  split, simp [hA, ht_initial],
  split, simp [hB, ht_initial, ht_updated],
  split, simp [hA_share, hA, hA_updated, hB, hB_updated],
  sorry
end

end total_profit_l817_817645


namespace gel_pen_price_ratio_l817_817304

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l817_817304


namespace even_blue_faces_cubes_correct_l817_817665

/-- A rectangular wooden block is 6 inches long, 3 inches wide, and 2 inches high.
    The block is painted blue on all six sides and then cut into 1 inch cubes.
    This function determines the number of 1-inch cubes that have a total number
    of blue faces that is an even number (in this case, 2 blue faces). -/
def count_even_blue_faces_cubes : Nat :=
  let length := 6
  let width := 3
  let height := 2
  let total_cubes := length * width * height
  
  -- Calculate corner cubes
  let corners := 8

  -- Calculate edges but not corners cubes
  let edge_not_corners := 
    (4 * (length - 2)) + 
    (4 * (width - 2)) + 
    (4 * (height - 2))

  -- Calculate even number of blue faces cubes 
  let even_number_blue_faces := edge_not_corners

  even_number_blue_faces

theorem even_blue_faces_cubes_correct : count_even_blue_faces_cubes = 20 := by
  -- Place your proof here.
  sorry

end even_blue_faces_cubes_correct_l817_817665


namespace price_ratio_l817_817338

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l817_817338


namespace necessary_but_not_sufficient_l817_817136

theorem necessary_but_not_sufficient (α : ℝ) :
    (α ≠ π / 6) ↔ (sin α ≠ 1 / 2) :=
sorry

end necessary_but_not_sufficient_l817_817136


namespace sqrt_meaningful_range_l817_817601

theorem sqrt_meaningful_range (x : ℝ) : 2 * x - 6 ≥ 0 ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_range_l817_817601


namespace average_of_first_20_even_numbers_l817_817211

theorem average_of_first_20_even_numbers :
  let s := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
  in (s.sum / s.length) = 21 :=
by
  sorry

end average_of_first_20_even_numbers_l817_817211


namespace pollution_control_improvements_l817_817460

theorem pollution_control_improvements (r0 r1 : ℝ) (n : ℕ) (t : ℝ)
  (h0 : r0 = 2.25) (h1 : r1 = 2.21)
  (h_model : ∀ n : ℕ, r0 + (r1 - r0) * (3:ℝ)^(0.25 * n + t) ≤ r0 + (r1 - r0) * 50)
  (h_lg2 : Real.log 2 ≈ 0.30) (h_lg3 : Real.log 3 ≈ 0.48) :
  n = 16 :=
  sorry

end pollution_control_improvements_l817_817460


namespace M_eq_N_cardinality_l817_817633

def M (n : ℕ) : set ℕ :=
  {x : ℕ | ∀ d ∈ (nat.digits 10 x), d = 1 ∨ d = 2 ∧ (nat.digits 10 x).count 1 = n ∧ (nat.digits 10 x).count 2 = n}

def N (n : ℕ) : set ℕ :=
  {y : ℕ | ∀ d ∈ (nat.digits 10 y), d ∈ {1, 2, 3, 4} ∧ (nat.digits 10 y).count 1 = n ∧ (nat.digits 10 y).count 2 = n}

theorem M_eq_N_cardinality (n : ℕ) : (M n).card = (N n).card :=
sorry

end M_eq_N_cardinality_l817_817633


namespace greatest_m_condition_l817_817990

noncomputable def A (n : ℕ) := ({a | ∃ i, i < n ∧ a = a_ i} ∪ {b | ∃ i, i < n ∧ b = b_ i})

def B_i_subset (A : set α) (i : ℕ) := B_i i ⊆ A

def condition_3 (B : ℕ → set α) (A : set α) := (⋃ i, B i) = A

def condition_4 (B : ℕ → set α) (n : ℕ) := ∀ i j, (i < m ∧ j < n) → ¬ ({a_j, b_j} ⊆ B i)

noncomputable def a (m n : ℕ) := (2^m - 1)^(2 * n)

noncomputable def b (m n : ℕ) := (3^m - 2^(m + 1) + 1)^n

theorem greatest_m_condition (h : 2 ≤ m) : 
  (∃ n > 0, (a m n / b m n) ≤ 2021) → m ≤ 26 := sorry

end greatest_m_condition_l817_817990


namespace min_max_of_quadratic_l817_817716

theorem min_max_of_quadratic 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 - 6 * x + 1)
  (h2 : ∀ x, -1 ≤ x ∧ x ≤ 1) : 
  (∃ xmin, ∃ xmax, f xmin = -3 ∧ f xmax = 9 ∧ -1 ≤ xmin ∧ xmin ≤ 1 ∧ -1 ≤ xmax ∧ xmax ≤ 1) :=
sorry

end min_max_of_quadratic_l817_817716


namespace max_cos_plus_2sin_eq_sqrt5_l817_817373

theorem max_cos_plus_2sin_eq_sqrt5 :
  ∃ x : ℝ, cos x + 2 * sin x = real.sqrt 5 :=
sorry

end max_cos_plus_2sin_eq_sqrt5_l817_817373


namespace range_of_a_l817_817572

-- Define the function f(x).
def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- Define the derivative of f(x).
def f_prime (x : ℝ) (a : ℝ) : ℝ := -3*x^2 + 2*a*x - 1

theorem range_of_a (a : ℝ) : ¬ monotonic_on (λ x : ℝ, f x a) set.univ ↔ a < -real.sqrt 3 ∨ a > real.sqrt 3 := by
  sorry -- The actual proof is omitted.

end range_of_a_l817_817572


namespace hemisphere_surface_area_l817_817240

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h1: 0 < π) (h2: A = 3) (h3: S = 4 * π * r^2):
  ∃ t, t = 9 :=
by
  sorry

end hemisphere_surface_area_l817_817240


namespace mark_peters_pond_depth_l817_817125

theorem mark_peters_pond_depth :
  let mark_depth := 19
  let peter_depth := 5
  let three_times_peter_depth := 3 * peter_depth
  mark_depth - three_times_peter_depth = 4 :=
by
  sorry

end mark_peters_pond_depth_l817_817125


namespace pen_price_ratio_l817_817320

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l817_817320


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817893

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817893


namespace pairs_condition_l817_817713

theorem pairs_condition (a b : ℕ) (prime_p : ∃ p, p = a^2 + b + 1 ∧ Nat.Prime p)
    (divides : ∀ p, p = a^2 + b + 1 → p ∣ (b^2 - a^3 - 1))
    (not_divides : ∀ p, p = a^2 + b + 1 → ¬ p ∣ (a + b - 1)^2) :
  ∃ x, x ≥ 2 ∧ a = 2 ^ x ∧ b = 2 ^ (2 * x) - 1 := sorry

end pairs_condition_l817_817713


namespace count_two_digit_integers_remainder_3_l817_817869

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817869


namespace distance_between_front_contestants_l817_817450

noncomputable def position_a (pd : ℝ) : ℝ := pd - 10
def position_b (pd : ℝ) : ℝ := pd - 40
def position_c (pd : ℝ) : ℝ := pd - 60
def position_d (pd : ℝ) : ℝ := pd

theorem distance_between_front_contestants (pd : ℝ):
  position_d pd - position_a pd = 10 :=
by
  sorry

end distance_between_front_contestants_l817_817450


namespace count_possible_values_l817_817355

-- Define the initial expression as 3^(3^(3^3))
def initial_expr := 3 ^ (3 ^ (3 ^ 3))

-- The question is essentially about counting the distinct values after rearranging parentheses in the expression
def count_other_values (exp : ℕ) : ℕ :=
  let val1 := 3 ^ (3 ^ (3 ^ 3))
  let val2 := 3 ^ ((3 ^ 3) ^ 3)
  let val3 := (3 ^ 3) ^ (3 ^ 3)
  let val4 := (3 ^ (3 ^ 3)) ^ 3
  let val5 := ((3 ^ 3) ^ 3) ^ 3
  -- Count distinct values
  let distinct_values := List.length (List.erase_dup [val1, val2, val3, val4, val5])
  distinct_values - 1

-- Assert that the expression has exactly one other possible value
theorem count_possible_values : count_other_values initial_expr = 1 :=
  by
    sorry

end count_possible_values_l817_817355


namespace range_of_a_l817_817417

noncomputable def f (x : ℝ) : ℝ := 2^x + 1

theorem range_of_a (a : ℝ) : f(a^2) < f(1) → a ∈ set.Ioo (-1 : ℝ) 1 :=
  by
  ?sorry

end range_of_a_l817_817417


namespace symmetric_graph_l817_817412

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + sqrt 3 * cos (2 * x)

theorem symmetric_graph (φ : ℝ) :
  (∀ x : ℝ, f (x + φ) = f (-x + φ)) ↔ φ = π / 12 := 
sorry

end symmetric_graph_l817_817412


namespace spider_minimum_distance_l817_817080

noncomputable def tree_height : ℝ := 30
noncomputable def trees_distance : ℝ := 20

noncomputable def branch_length (h : ℝ) : ℝ := (tree_height - h) / 2

theorem spider_minimum_distance :
  ∀ (h₁ h₂ : ℝ),
  h₁ = tree_height → h₂ = tree_height →
  let h := 20 in
  let d := 10 + 20 + 10 in
  d = 60 := by
  intros h₁ h₂ h₁_eq h₂_eq
  let h := 20
  have down := tree_height - h
  have horiz := trees_distance
  have up := tree_height - h
  -- The total distance traveled
  have total_dist := down + horiz + up
  -- Assert equality
  show total_dist = 60
  sorry

end spider_minimum_distance_l817_817080


namespace count_interesting_numbers_l817_817840

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817840


namespace thirtieth_number_in_sequence_l817_817484

def nth_odd_number (n : ℕ) : ℕ := 2 * n - 1

theorem thirtieth_number_in_sequence : nth_odd_number 30 = 59 :=
by {
  unfold nth_odd_number,
  norm_num,
  sorry
}

end thirtieth_number_in_sequence_l817_817484


namespace distance_ratio_l817_817626

-- Defining the conditions
def speedA : ℝ := 50 -- Speed of Car A in km/hr
def timeA : ℝ := 6 -- Time taken by Car A in hours

def speedB : ℝ := 100 -- Speed of Car B in km/hr
def timeB : ℝ := 1 -- Time taken by Car B in hours

-- Calculating the distances
def distanceA : ℝ := speedA * timeA -- Distance covered by Car A
def distanceB : ℝ := speedB * timeB -- Distance covered by Car B

-- Statement to prove the ratio of distances
theorem distance_ratio : (distanceA / distanceB) = 3 :=
by
  -- Calculations here might be needed, but we use sorry to indicate proof is pending
  sorry

end distance_ratio_l817_817626


namespace find_a_constant_l817_817758

noncomputable def a (n : ℕ) := 6 * n - 3
noncomputable def b (n : ℕ) := 9 ^ (n - 1)

theorem find_a_constant :
  let a_value : ℝ := (real.log 9) / 6 in
  a_value = real.cbrt 3 :=
by
  have h1 : a 1 = 3 := by norm_num
  have h2 : b 1 = 1 := by norm_num
  have h3 : a 2 = b 2 := by simp [a, b]; norm_num
  have h4 : 3 * a 5 = b 3 := by simp [a, b]; norm_num
  have h5 : ∀ n : ℕ, a n = 3 + real.log a_value (b n) :=
    λ n, by simp [a, b, a_value]; sorry
  show a_value = real.cbrt 3, from by
  {
    calc
      a_value = (real.log 9) / 6 : by simp [a_value]
          ... = real.cbrt 3     : by sorry
  }
  sorry

end find_a_constant_l817_817758


namespace count_two_digit_integers_with_remainder_3_l817_817841

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817841


namespace collinear_points_sum_l817_817706

theorem collinear_points_sum (x y : ℝ) 
  (h1 : (2, x, y)) 
  (h2 : (x, 3, y)) 
  (h3 : (x, y, 4))
  (h_collinear : ∃ a b c : ℝ, (2, x, y) = (a, b, c) ∧ x = 2 ∧ y = 4) :
  x + y = 6 := 
by
  sorry

end collinear_points_sum_l817_817706


namespace william_max_riding_time_l817_817226

theorem william_max_riding_time (x : ℝ) :
  (2 * x + 2 * 1.5 + 2 * (1 / 2 * x) = 21) → (x = 6) :=
by
  sorry

end william_max_riding_time_l817_817226


namespace price_ratio_l817_817344

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l817_817344


namespace pen_price_ratio_l817_817317

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l817_817317


namespace justin_current_age_l817_817683

theorem justin_current_age
  (angelina_older : ∀ (j a : ℕ), a = j + 4)
  (angelina_future_age : ∀ (a : ℕ), a + 5 = 40) :
  ∃ (justin_current_age : ℕ), justin_current_age = 31 := 
by
  sorry

end justin_current_age_l817_817683


namespace inequality_solution_set_l817_817588

theorem inequality_solution_set (x : ℝ) : 
  (∃ x, (2 < x ∧ x < 3)) ↔ 
  ((x - 2) * (x - 3) / (x^2 + 1) < 0) :=
by sorry

end inequality_solution_set_l817_817588


namespace simplify_cube_root_l817_817147

theorem simplify_cube_root :
  (∛(80^3 + 100^3 + 120^3) = 20 * 405^(1/3)) :=
by
  sorry

end simplify_cube_root_l817_817147


namespace arithmetic_sequence_common_difference_l817_817558

theorem arithmetic_sequence_common_difference (a : ℕ → ℚ) (h_arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
    (sum_1_to_100 : ∑ i in finset.range 100, a (i + 1) = 100)
    (sum_101_to_200 : ∑ i in finset.range 100, a (i + 101) = 200) :
    a 2 - a 1 = 1 / 100 :=
begin
  sorry
end

end arithmetic_sequence_common_difference_l817_817558


namespace a_general_formula_S_bounds_l817_817790

open Nat 

-- Sequence definition
noncomputable def a : ℕ → ℕ
| 0       => 2
| (n + 1) => (a n + 1) / 2

-- Correct answer for a_n
theorem a_general_formula (n : ℕ) : 
  a n = (1 / 2 : ℝ)^(n : ℝ) + 1 := 
  sorry

-- Define b_n
noncomputable def b (n : ℕ) : ℝ :=
  (n : ℝ) * (a n - 1)

-- Sum of first n terms of b_n
noncomputable def S (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, b i)

-- Proving bounds on S_n
theorem S_bounds (n : ℕ) : 
  1 ≤ S n ∧ S n < 4 :=
  sorry

end a_general_formula_S_bounds_l817_817790


namespace jellybean_ratio_l817_817543

theorem jellybean_ratio (gigi_je : ℕ) (rory_je : ℕ) (lorelai_je : ℕ) (h_gigi : gigi_je = 15) (h_rory : rory_je = gigi_je + 30) (h_lorelai : lorelai_je = 180) : lorelai_je / (rory_je + gigi_je) = 3 :=
by
  -- Introduce the given hypotheses
  rw [h_gigi, h_rory, h_lorelai]
  -- Simplify the expression
  sorry

end jellybean_ratio_l817_817543


namespace two_digit_integers_remainder_3_count_l817_817875

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817875


namespace annie_building_time_l817_817284

theorem annie_building_time (b p : ℕ) (h1 : b = 3 * p - 5) (h2 : b + p = 67) : b = 49 :=
by
  sorry

end annie_building_time_l817_817284


namespace find_angle_B_l817_817466

theorem find_angle_B (a b : ℝ) (h_a : a = 4) (h_b : b = 5) (h_cos : cos (B + C) = -3/5) : 
  B = Real.pi - Real.arccos(3/5) := 
sorry

end find_angle_B_l817_817466


namespace interest_rate_per_annum_l817_817265

theorem interest_rate_per_annum
  (P : ℕ := 450) 
  (t : ℕ := 8) 
  (I : ℕ := P - 306) 
  (simple_interest : ℕ := P * r * t / 100) :
  r = 4 :=
by
  sorry

end interest_rate_per_annum_l817_817265


namespace no_real_roots_P_eq_Q_l817_817427

noncomputable def P (x : ℝ) : ℝ := sorry
noncomputable def Q (x : ℝ) : ℝ := sorry

lemma degree_P : nat_degree (P 0) = 10 := sorry
lemma degree_Q : nat_degree (Q 0) = 10 := sorry
lemma leading_coeff_P : leading_coeff (P 0) = 1 := sorry
lemma leading_coeff_Q : leading_coeff (Q 0) = 1 := sorry

theorem no_real_roots_P_eq_Q (h_no_real_roots : ∀ x : ℝ, P x ≠ Q x) :
  ∃ x : ℝ, P (x + 1) = Q (x - 1) :=
sorry

end no_real_roots_P_eq_Q_l817_817427


namespace arithmetic_geometric_sequences_l817_817401

noncomputable def a : ℕ := 2
noncomputable def b : ℕ := 5
def a_n (n : ℕ) : ℕ := 5 * n - 3
def b_n (n : ℕ) : ℕ := b * 2^(n-1)

theorem arithmetic_geometric_sequences :
  ∀ (n : ℕ) (a b : ℕ),
    a > 1 ∧ b > 1 ∧ 
    a < b ∧ 
    b * a < a + 2 * b ∧ 
    ∃ m : ℕ, b_n n = a_n m + 3 → 
    a = 2 ∧ a_n n = 5*n - 3 := 
by
  sorry

end arithmetic_geometric_sequences_l817_817401


namespace variance_2X_plus_1_l817_817755

/-- Given X is a binomial random variable with parameters 10 and 0.8, 
we need to prove that the variance of 2X + 1 is 6.4. -/
theorem variance_2X_plus_1 (X : ℝ) (h : X ∼ binomial 10 0.8) : D(2 * X + 1) = 6.4 :=
  sorry

end variance_2X_plus_1_l817_817755


namespace area_triangle_BDF_l817_817464

variable (A B C D E F : Type)
variable [LinearOrder D]
variable [LinearOrder F]

-- Midpoint condition: D is the midpoint of AB
variable (midpoint_D : ∀ (x : D), x = midpoint A B)

-- Segment ratios
variable (ratio_CE_DE : ∀ (y : C × E) (z : D × E), (segmentRatio y z) = 5 / 3)
variable (ratio_BF_EF : ∀ (u : B × F) (v : F × E), (segmentRatio u v) = 1 / 3)

-- Area condition for triangle ABC
variable (area_triangle_ABC : areaTriangle A B C = 192)

-- Theorem to prove the area of triangle BDF
theorem area_triangle_BDF : (areaTriangle B D F = 15) :=
sorry

end area_triangle_BDF_l817_817464


namespace abs_difference_of_squares_l817_817210

theorem abs_difference_of_squares : abs ((102: ℤ) ^ 2 - (98: ℤ) ^ 2) = 800 := by
  sorry

end abs_difference_of_squares_l817_817210


namespace value_computation_l817_817933

theorem value_computation (N : ℝ) (h1 : 1.20 * N = 2400) : 0.20 * N = 400 := 
by
  sorry

end value_computation_l817_817933


namespace building_time_l817_817282

theorem building_time (b p : ℕ) 
  (h1 : b = 3 * p - 5) 
  (h2 : b + p = 67) 
  : b = 49 := 
by 
  sorry

end building_time_l817_817282


namespace Vasya_mushrooms_l817_817204

def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def digitsSum (n : ℕ) : ℕ := (n / 100) + ((n % 100) / 10) + (n % 10)

theorem Vasya_mushrooms :
  ∃ n : ℕ, isThreeDigit n ∧ digitsSum n = 14 ∧ n = 950 := 
by
  sorry

end Vasya_mushrooms_l817_817204


namespace count_two_digit_remainders_l817_817802

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817802


namespace intersection_point_P_coordinates_l817_817978

-- Variables representing points X, Y, Z in a vector space
variables {V : Type*} [AddCommGroup V] [Module ℝ V] (X Y Z : V)

-- Definitions of points D, E, and F based on the given conditions
def D : V := (4/5) • Z + (1/5) • Y
def E : V := (2/5) • X + (3/5) • Z
def F : V := (1/3) • X + (2/3) • Y 

-- Definition of point P as the intersection of lines YD and BF
def P : V := (1/3) • X + (2/3) • Y

-- The theorem statement for proving the coordinates of P
theorem intersection_point_P_coordinates :
  P = (1/3) • X + (2/3) • Y :=
sorry

end intersection_point_P_coordinates_l817_817978


namespace count_two_digit_integers_remainder_3_div_7_l817_817903

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817903


namespace music_books_cost_l817_817489

theorem music_books_cost
  (total_money : ℕ) (maths_books_count : ℕ) (maths_books_price : ℕ)
  (science_books_extra_count : ℕ) (science_books_price : ℕ)
  (art_books_multiplier : ℕ) (art_books_price : ℕ) :
  total_money = 500 →
  maths_books_count = 4 →
  maths_books_price = 20 →
  science_books_extra_count = 6 →
  science_books_price = 10 →
  art_books_multiplier = 2 →
  art_books_price = 20 →
  let
    maths_books_cost := maths_books_count * maths_books_price
    science_books_cost := (maths_books_count + science_books_extra_count) * science_books_price
    art_books_cost := (art_books_multiplier * maths_books_count) * art_books_price
    total_cost_excluding_music := maths_books_cost + science_books_cost + art_books_cost
    music_books_cost := total_money - total_cost_excluding_music
  in music_books_cost = 160 :=
by
  intros
  sorry

end music_books_cost_l817_817489


namespace f_of_pi_over_6_l817_817037

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

theorem f_of_pi_over_6 (ω ϕ : ℝ) (h₀ : ω > 0) (h₁ : -Real.pi / 2 ≤ ϕ) (h₂ : ϕ < Real.pi / 2) 
  (transformed : ∀ x, f ω ϕ (x/2 - Real.pi/6) = Real.sin x) :
  f ω ϕ (Real.pi / 6) = Real.sqrt 2 / 2 :=
by
  sorry

end f_of_pi_over_6_l817_817037


namespace matrix_power_l817_817698

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.sqrt 2, -Real.sqrt 2], ![Real.sqrt 2, Real.sqrt 2]]

-- Define what we want to prove, that A^5 equals the given matrix.
theorem matrix_power :
  A ^ 5 = (32 : ℝ) • ![![-Real.sqrt 2 / 2, Real.sqrt 2 / 2], ![-Real.sqrt 2 / 2, -Real.sqrt 2 / 2]] :=
sorry

end matrix_power_l817_817698


namespace unknown_number_value_l817_817938

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end unknown_number_value_l817_817938


namespace valid_addition_equation_l817_817535

/-- 
Given the digits 1, 2, 5, 7, 9, 0 and the symbols + and =, 
prove that a valid addition equation using all these cards can be formed. 
Specifically, prove that 95 + 7 = 102 using these cards.
-/
theorem valid_addition_equation 
  (d1 d2 d5 d7 d9 d0 : ℕ) 
  (plus eq : char) 
  (h1 : d1 = 1) (h2 : d2 = 2) (h5 : d5 = 5) (h7 : d7 = 7) (h9 : d9 = 9) (h0 : d0 = 0)
  (h_plus : plus = '+') (h_eq : eq = '=') :
  95 + 7 = 102 :=
by sorry

end valid_addition_equation_l817_817535


namespace count_two_digit_integers_with_remainder_3_l817_817847

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817847


namespace infinitely_many_primes_divide_element_of_sequence_l817_817496

def sequence (a : ℕ) : ℕ → ℕ
| 0 := 1
| (n + 1) := a + (List.prod (List.map sequence (List.range (n + 1))))

theorem infinitely_many_primes_divide_element_of_sequence (a : ℕ) (ha : 0 < a) :
  ∃ infinitely_many (p : ℕ), p.prime ∧ ∃ n, p ∣ sequence a n :=
sorry

end infinitely_many_primes_divide_element_of_sequence_l817_817496


namespace probability_of_sum_14_l817_817442

-- Define the sample space Ω of four-dice rolls
def Ω : Type := (Fin 6 × Fin 6 × Fin 6 × Fin 6)

-- Define the event of the sum being 14
def event_sum_14 (ω : Ω) : Prop :=
  ω.1.1 + ω.1.2 + ω.2.1 + ω.2.2 = 14

-- Number of favorable outcomes
noncomputable def favorable_count : Nat :=
  -- direct listing and counting is omitted here, but we'd compute this by exhaustive enumeration
  46
  
-- Total number of outcomes
def total_outcomes : Nat := 6^4

-- Probability of the event occurring
noncomputable def probability_sum_14 : ℚ :=
  favorable_count / total_outcomes

-- State the theorem to prove the equivalence
theorem probability_of_sum_14 :
  probability_sum_14 = 46 / 1296 := 
  sorry

end probability_of_sum_14_l817_817442


namespace maximum_angle_l817_817971

noncomputable def polar_to_cartesian_equation (rho θ : ℝ) : Prop :=
  (rho^2 - 4 * rho * sin θ + 3 = 0) ↔ ((1 - rho * cos θ)^2 + (2 - rho * sin θ)^2 = 1)

def parametric_to_cartesian_equation (t x y : ℝ) : Prop :=
  (x = 1 - (sqrt 2 / 2) * t) ∧ (y = 3 + (sqrt 2 / 2) * t) ↔ (x + y = 4)

theorem maximum_angle (A B P : (ℝ × ℝ)) (C : set (ℝ × ℝ)) (l: set (ℝ × ℝ)) :
  (∀ (ρ θ : ℝ), (ρ, θ) ∈ C ↔ (ρ^2 - 4 * ρ * sin θ + 3 = 0) ∧ (ρ, θ) = (sqrt 2, π/4)) →
  (∀ (t : ℝ), (P.1 = 1 - sqrt 2 / 2 * t) ∧ (P.2 = 3 + sqrt 2 / 2 * t) ↔ (P.1 + P.2 = 4)) →
  ∃ (max_angle : ℝ), max_angle = π / 2 :=
by
  intros
  sorry

end maximum_angle_l817_817971


namespace magnitude_diff_OQ_OP_l817_817407

def hyperbola : set (ℝ × ℝ) :=
  {p | (p.1^2 / 9) - (p.2^2 / 4) = 1}

variable (F1 F2 A P Q : ℝ × ℝ)

-- Conditions 
hyp F1F2_def : F1.1 = -3 ∧ F1.2 = 0 ∧ F2.1 = 3 ∧ F2.2 = 0
    := sorry,
hyp_A_on_hyperbola : A ∈ hyperbola
    := sorry,
hyp_P : 2 * P = (A.1 + F1.1, A.2 + F1.2)
    := sorry,
hyp_Q : 2 * Q = (A.1 + F2.1, A.2 + F2.2)
    := sorry

-- Statement to prove
theorem magnitude_diff_OQ_OP : 
  |Q.1 - 0, Q.2 - 0| - |P.1 - 0, P.2 - 0| = 3 :=
sorry

end magnitude_diff_OQ_OP_l817_817407


namespace gel_pen_ratio_l817_817312

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l817_817312


namespace count_two_digit_integers_remainder_3_l817_817873

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817873


namespace volleyball_tournament_order_l817_817456

theorem volleyball_tournament_order (n : ℕ) (win : fin n → fin n → Prop) :
    (∀ (i j : fin n), i ≠ j → (win i j ∨ win j i)) → 
    ∃ (list_order : List (fin n)), ∀ (k : ℕ) (hk : k < list_order.length - 1),
    win (list_order.nth_le k (by linarith)) (list_order.nth_le (k + 1) (by linarith)) :=
by
  sorry

end volleyball_tournament_order_l817_817456


namespace a_geq_five_thirds_l817_817357

noncomputable def f (a : ℝ) (x : ℝ) (n : ℕ) : ℝ :=
if 1 ≤ x ∧ x < 2 then 2 * x + 1
else a^(n-1) * (2 * (x - n + 1) + 1)

theorem a_geq_five_thirds (a : ℝ) 
  (h_quasi_periodic : ∀ x : ℝ, x ≥ 1 → af (a : ℝ → ℝ) = f (a : ℝ → ℝ) (x + (1 : ℝ)))
  (h_monotonic : ∀ m n x, 1 ≤ m ∧ m < n ∧ x < n → f a (m:ℝ) ≤ f a ((n:ℝ) + x)) :
  a ≥ 5 / 3 :=
begin
  sorry
end

end a_geq_five_thirds_l817_817357


namespace inclination_angle_range_l817_817781

theorem inclination_angle_range :
  let Γ := fun x y : ℝ => x * abs x + y * abs y = 1
  let line (m : ℝ) := fun x y : ℝ => y = m * (x - 1)
  ∀ m : ℝ,
  (∃ p1 p2 p3 : ℝ × ℝ, 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    line m p1.1 p1.2 ∧ Γ p1.1 p1.2 ∧ 
    line m p2.1 p2.2 ∧ Γ p2.1 p2.2 ∧ 
    line m p3.1 p3.2 ∧ Γ p3.1 p3.2) →
  (∃ θ : ℝ, θ ∈ (Set.Ioo (Real.pi / 2) (3 * Real.pi / 4) ∪ 
                  Set.Ioo (3 * Real.pi / 4) (Real.pi - Real.arctan (Real.sqrt 2 / 2)))) :=
sorry

end inclination_angle_range_l817_817781


namespace altitude_of_triangle_l817_817667

-- Definitions and conditions in Lean
variables {A B C E F Q P H : Type}
variables [Triangle A B C]
variables [OnLineSegment E F BC]
variables [Semicircle E F]
variables [TangentToLine Q A B]
variables [TangentToLine P A C]
variables [Intersection H (Line EP) (Line FQ)]

-- The theorem to prove
theorem altitude_of_triangle (h1 : TangentToLine Q.AB)
                              (h2 : TangentToLine P.AC)
                              (h3 : Intersection H (Line EP) (Line FQ)) : Altitude H A BC :=
sorry

end altitude_of_triangle_l817_817667


namespace infinitely_many_quadratic_polynomials_l817_817918

theorem infinitely_many_quadratic_polynomials : ∃ (f : ℝ → (ℝ × ℝ)), 
  (∀ r, f r = (r, 1 / r)) ∧ 
  ∃ a b c : ℝ, 
    (a = 1 / r * r) ∧ 
    (b = r + 1 / r) ∧ 
    (c = 1) :=
begin
  sorry
end

end infinitely_many_quadratic_polynomials_l817_817918


namespace solution_set_of_inequality_l817_817181

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) / (x^2 - x - 6) ≥ 0} = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (3 < x)} := 
sorry

end solution_set_of_inequality_l817_817181


namespace value_of_k_h_5_l817_817932

def h (x : ℝ) : ℝ := 4 * x + 6
def k (x : ℝ) : ℝ := 6 * x - 8

theorem value_of_k_h_5 : k (h 5) = 148 :=
by
  have h5 : h 5 = 4 * 5 + 6 := rfl
  simp [h5, h, k]
  sorry

end value_of_k_h_5_l817_817932


namespace statements_are_incorrect_l817_817618

-- Condition: The coefficient of -2πx²y³ is not -2
def coefficient_is_not (c : ℝ) : Prop :=
  c ≠ -2

-- Condition: 2³ and 3² are not like terms
def not_like_terms (a b : ℝ) : Prop :=
  false  -- 2³ and 3² being compared as algebraic terms

-- Condition: The degree of the polynomial is not 3
def degree_is_not (p : ℕ) : Prop :=
  p ≠ 3

-- Condition: -2³ and |-2|³ are not equal
def result_not_equal (a b : ℝ) : Prop :=
  a ≠ b

theorem statements_are_incorrect :
  coefficient_is_not (-2*π) ∧ not_like_terms (2^3) (3^2) ∧ degree_is_not (2 + 1 + 1) ∧ result_not_equal (-2^3) ((|-2|)^3) :=
begin
  split, -- for coefficient
  { unfold coefficient_is_not, exact by simp [π], sorry },
  split, -- for like terms
  { unfold not_like_terms, exact false.elim, },
  split, -- for degree
  { unfold degree_is_not, exact by simp, sorry },
  { -- for result equality
    unfold result_not_equal, exact by simp, sorry }
end

end statements_are_incorrect_l817_817618


namespace count_interesting_numbers_l817_817836

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817836


namespace solve_for_y_l817_817923

-- Define the function G
def G (a b c d : ℝ) : ℝ := a ^ b + c ^ d

-- The theorem to be proven
theorem solve_for_y : ∃ y : ℝ, G 3 y 2 5 = 350 ↔ 3 ^ y = 318 := 
by
  -- Start the proof with declaration
  existsi (Real.log 318 / Real.log 3)
  sorry

end solve_for_y_l817_817923


namespace log_relation_l817_817410

theorem log_relation (a b c: ℝ) (h₁: a = (Real.log 2) / 2) (h₂: b = (Real.log 3) / 3) (h₃: c = (Real.log 5) / 5) : c < a ∧ a < b :=
by
  sorry

end log_relation_l817_817410


namespace closest_multiple_of_15_to_3157_is_3150_l817_817220

def is_multiple_of_15 (n : ℕ) : Prop :=
  (n % 3 = 0) ∧ (n % 5 = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def is_closer (a b x : ℕ) : Prop :=
  abs (a - x) < abs (b - x)

theorem closest_multiple_of_15_to_3157_is_3150 :
  ∃ (n : ℕ), is_multiple_of_15 n ∧ is_closer n 3165 3157 ∧ is_closer 3150 n 3157 :=
sorry

end closest_multiple_of_15_to_3157_is_3150_l817_817220


namespace unique_strictly_increasing_function_l817_817371

-- Definition of strictly increasing function
def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ {a b : ℕ}, a < b → f(a) < f(b)

-- Definition for coprime numbers
def coprime (m n : ℕ) : Prop := gcd m n = 1

-- The proof problem statement
theorem unique_strictly_increasing_function (f : ℕ → ℕ) (h1 : strictly_increasing f)
    (h2 : f 2 = 2)
    (h3 : ∀ m n, coprime m n → f (m * n) = f m * f n) :
    ∀ n : ℕ, f n = n :=
begin
  sorry
end

end unique_strictly_increasing_function_l817_817371


namespace quadrilateral_area_l817_817457

noncomputable def area_quad_NSQT (a : ℝ) (α : ℝ) : ℝ :=
  (1/2) * a^2 * cot α

theorem quadrilateral_area {P Q R T S N : Type*} [Point P] [Point Q] [Point R] [Point T] [Point S] [Point N]
  {PQ QR : ℝ} (a : ℝ) (α : ℝ)
  (acute_triangle : acute (angle P Q R))
  (PQ_gt_QR : PQ > QR)
  (alt_PT : altitude P Q R T)
  (alt_RS : altitude R Q P S)
  (QN_diameter : diameter (circumcircle P Q R) Q N)
  (angle_PT_RS : acute (angle PT RS) = α)
  (PR_eq_a : PR = a):
  area_quad_NSQT a α = (1/2) * a^2 * cot α :=
sorry

end quadrilateral_area_l817_817457


namespace Emily_candies_l817_817096

theorem Emily_candies (jennifer_candies emily_candies bob_candies : ℕ) 
    (h1: jennifer_candies = 2 * emily_candies)
    (h2: jennifer_candies = 3 * bob_candies)
    (h3: bob_candies = 4) : emily_candies = 6 :=
by
  -- Proof to be provided
  sorry

end Emily_candies_l817_817096


namespace range_of_slope_l817_817177

def slope (θ : ℝ) : ℝ := Real.tan θ

def is_angle_in_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < π

theorem range_of_slope : { m : ℝ | ∃ θ, is_angle_in_range θ ∧ m = slope θ } = set.Ico 0 π :=
by
  sorry

end range_of_slope_l817_817177


namespace correct_statements_l817_817225

-- Define the universal set U as ℤ (integers)
noncomputable def U : Set ℤ := Set.univ

-- Conditions
def is_subset_of_int : Prop := {0} ⊆ (Set.univ : Set ℤ)

def counterexample_subsets (A B : Set ℤ) : Prop :=
  (A = {1, 2} ∧ B = {1, 2, 3}) ∧ (B ∩ (U \ A) ≠ ∅)

def negation_correct_1 : Prop :=
  ¬(∀ x : ℤ, x^2 > 0) ↔ ∃ x : ℤ, x^2 ≤ 0

def negation_correct_2 : Prop :=
  ¬(∀ x : ℤ, x^2 > 0) ↔ ¬(∀ x : ℤ, x^2 < 0)

-- The theorem to prove the equivalence of correct statements
theorem correct_statements :
  (is_subset_of_int ∧
   ∀ A B : Set ℤ, A ⊆ U → B ⊆ U → (A ⊆ B → counterexample_subsets A B) ∧
   negation_correct_1 ∧
   ¬negation_correct_2) ↔
  (true) :=
by 
  sorry

end correct_statements_l817_817225


namespace min_value_expr_l817_817717

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) : (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 24 :=
sorry

end min_value_expr_l817_817717


namespace sum_inequality_l817_817396

theorem sum_inequality
  (n : ℕ) (k : ℝ) (a : ℕ → ℝ)
  (h1 : 2 ≤ n)
  (h2 : 1 ≤ k)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ a i)
  (h4 : (∑ i in Finset.range n, a i) = n)
  (h5 : a (n + 1) = a 1):
  (∑ i in Finset.range n, ((a i + 1)^(2 * k)) / ((a (i + 1) % n + 1)^k)) ≥ n * 2^k :=
by
  sorry

end sum_inequality_l817_817396


namespace find_k_for_two_subsets_l817_817060

open Real

theorem find_k_for_two_subsets :
  (∃ (k : ℝ), ∀ (x : ℝ), (k + 1) * x^2 + x - k = 0 ∧ (∃ y : ℝ, y ≠ x) → (A ⊆ {x})) ↔ (k = -1 ∨ k = -1/2) :=
by
  sorry

end find_k_for_two_subsets_l817_817060


namespace range_of_a_l817_817955

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (1 ≤ x) ∧ (∀ a : ℝ, (1 + 1 / x) ^ (x + a) ≥ Real.exp 1 → a ≥ 1 / Real.log 2 - 1)

theorem range_of_a : problem_statement :=
sorry

end range_of_a_l817_817955


namespace compare_a_b_c_l817_817003

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def a : ℝ := 0.5 * f(0.5)
def b : ℝ := -2 * f(-2)
def c : ℝ := (Real.log 0.5) * f(-Real.log 2)

lemma g_deriv_pos (x : ℝ) (h_ne_zero : x ≠ 0) : f'(x) + f(x) / x > 0 := sorry

theorem compare_a_b_c
  (odd_f : is_odd f)
  (h_deriv : ∀ x, x ≠ 0 → f'(x) + f(x) / x > 0)
  : a < c ∧ c < b := 
sorry

end compare_a_b_c_l817_817003


namespace prime_pair_product_l817_817926

open Nat

theorem prime_pair_product (a b : ℕ) (ha : Prime a) (hb : Prime b) (h : a - b = 35) : a * b = 74 := 
by
  sorry

end prime_pair_product_l817_817926


namespace constant_term_expansion_sum_binomial_coefficients_l817_817019

theorem constant_term_expansion {n : ℕ} (h : 2^(2 * n) = 2^n + 240) :
  n = 4 → 
  binomial 8 4 = 70 :=
by
  intro hn
  rw hn
  simp
  norm_num

theorem sum_binomial_coefficients {n : ℕ} (h : 2^(2 * n) = 2^n + 240) :
  n = 4 → 
  2^4 = 16 :=
by
  intro hn
  rw hn
  simp
  norm_num

end constant_term_expansion_sum_binomial_coefficients_l817_817019


namespace count_two_digit_integers_remainder_3_l817_817823

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817823


namespace multiply_expression_l817_817531

variable {x : ℝ}

theorem multiply_expression :
  (x^4 + 10*x^2 + 25) * (x^2 - 25) = x^4 + 10*x^2 :=
by
  sorry

end multiply_expression_l817_817531


namespace find_sin_B_over_sin_C_find_a_l817_817078
noncomputable def triangle_area := euclidean_geometry.triangle_area

variables {A B C D : Point} {a b c : Length} {angleA : Angle}

-- Conditions
axiom angle_A_eq_60 : angleA = 60
axiom side_opposite_a : a = dist B C
axiom side_opposite_b : b = dist C A
axiom side_opposite_c : c = dist A B
axiom point_D_on_BC : D ∈ line B C
axiom CD_2DB : dist C D = 2 * dist D B
axiom AD_sqrt21_3b : dist A D = (sqrt 21) * b / 3

-- Questions
theorem find_sin_B_over_sin_C : (sin (angle B)) / (sin (angle C)) = 1 / 2 := sorry
theorem find_a (area_ABC : Real) (h : area_ABC = 2 * sqrt 3) : a = 2 * sqrt 3 := sorry

end find_sin_B_over_sin_C_find_a_l817_817078


namespace average_age_of_five_l817_817083

theorem average_age_of_five :
  let avg_age_all := 15
  let total_people := 16
  let avg_age_nine := 16
  let num_nine := 9
  let age_fifteenth := 26
  let num_five := 5

  -- Total age calculations
  total_age_all = total_people * avg_age_all →
  total_age_nine = num_nine * avg_age_nine →
  -- Total age of the five persons
  total_age_five = total_age_all - total_age_nine - age_fifteenth →
  -- Average age of the five persons
  avg_age_five = total_age_five / num_five →
  avg_age_five = 14
:=
by {
  intros avg_age_all total_people avg_age_nine num_nine age_fifteenth num_five
        total_age_all total_age_nine total_age_five avg_age_five,
  sorry
}

end average_age_of_five_l817_817083


namespace mother_picked_carrots_l817_817533

def total_carrots_picked (good_carrots bad_carrots : ℕ) : ℕ :=
  good_carrots + bad_carrots

def mother's_carrots (total_carrots olivia_carrots : ℕ) : ℕ :=
  total_carrots - olivia_carrots

theorem mother_picked_carrots (olivia_carrots good_carrots bad_carrots : ℕ) 
  (h_olivia : olivia_carrots = 20) 
  (h_good : good_carrots = 19) 
  (h_bad : bad_carrots = 15) :
  mother's_carrots (total_carrots_picked good_carrots bad_carrots) olivia_carrots = 14 :=
by
  rw [h_olivia, h_good, h_bad]
  unfold total_carrots_picked mother_carrots
  simp
  norm_num
  sorry

end mother_picked_carrots_l817_817533


namespace bisector_segment_length_l817_817154

section TriangleBisector

variables (a b : ℝ) (φ : ℝ)
noncomputable def bisector_length (a b φ : ℝ) : ℝ :=
  (2 * a * b * real.cos (φ / 2)) / (a + b)

theorem bisector_segment_length :
  ∀ (a b φ : ℝ), bisector_length a b φ = (2 * a * b * real.cos (φ / 2)) / (a + b) :=
by {
  intros,
  exact rfl,
}

end TriangleBisector

end bisector_segment_length_l817_817154


namespace annie_building_time_l817_817285

theorem annie_building_time (b p : ℕ) (h1 : b = 3 * p - 5) (h2 : b + p = 67) : b = 49 :=
by
  sorry

end annie_building_time_l817_817285


namespace second_number_in_first_set_l817_817155

theorem second_number_in_first_set :
  ∃ (x : ℝ), (20 + x + 60) / 3 = (10 + 80 + 15) / 3 + 5 ∧ x = 40 :=
by
  use 40
  sorry

end second_number_in_first_set_l817_817155


namespace total_notebooks_l817_817184

theorem total_notebooks (num_boxes : ℕ) (parts_per_box : ℕ) (notebooks_per_part : ℕ) (h1 : num_boxes = 22)
  (h2 : parts_per_box = 6) (h3 : notebooks_per_part = 5) : 
  num_boxes * parts_per_box * notebooks_per_part = 660 := 
by
  sorry

end total_notebooks_l817_817184


namespace positive_two_digit_integers_remainder_3_l817_817811

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817811


namespace gel_pen_price_ratio_l817_817303

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l817_817303


namespace positive_two_digit_integers_remainder_3_l817_817815

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817815


namespace unique_polynomial_remainder_l817_817119

theorem unique_polynomial_remainder :
  ∃ (Q R : Polynomial ℂ), 
  degree R < 3 ∧
  (z : ℂ) → R = -z^2 + 2 ∧ 
  z ^ 2023 + 1 = (z ^ 3 + z ^ 2 + 1) * Q + R :=
sorry

end unique_polynomial_remainder_l817_817119


namespace count_two_digit_remainders_l817_817806

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817806


namespace true_propositions_l817_817724

open Real

-- Define the custom distance function
def custom_distance (p1 p2 : ℝ × ℝ) : ℝ := 
  abs (p2.1 - p1.1) + abs (p2.2 - p1.2)

-- Define the three propositions
def proposition1 (A B C : ℝ × ℝ) : Prop := 
  ∃ (between_A_B : C.1 ≥ A.1 ∧ C.1 ≤ B.1 ∧ C.2 ≥ A.2 ∧ C.2 ≤ B.2),
  custom_distance A C + custom_distance C B = custom_distance A B

def proposition2 (A B C : ℝ × ℝ) : Prop := 
  ∠ACB = 90 →
  custom_distance A C ^ 2 + custom_distance C B ^ 2 = custom_distance A B ^ 2

def proposition3 (A B C : ℝ × ℝ) : Prop := 
  custom_distance A C + custom_distance C B > custom_distance A B

-- Statement including the result
theorem true_propositions (A B C : ℝ × ℝ) :
  proposition1 A B C ∧ proposition3 A B C :=
by 
  -- Proof will be required here
  {
    sorry 
  }

end true_propositions_l817_817724


namespace rooks_identical_distances_l817_817534

/-- On a chessboard, there are eight rooks placed such that no two rooks attack each other. 
Prove that among the pairwise distances between them, there are two distances that are the same. 
The distance between the rooks is defined as the distance between the centers of the squares they occupy. -/
theorem rooks_identical_distances :
  ∀ (rooks : Finset (ℕ × ℕ)),
  rooks.card = 8 ∧ (∀ r1 r2 ∈ rooks, r1 ≠ r2 → (r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2)) →
  ∃ (d : ℕ) (p1 p2 p3 p4 : (ℕ × ℕ)),
  p1 ∈ rooks ∧ p2 ∈ rooks ∧ p3 ∈ rooks ∧ p4 ∈ rooks ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
  dist p1 p2 = d ∧ dist p3 p4 = d := 
sorry

end rooks_identical_distances_l817_817534


namespace count_diagonals_l817_817664

-- Define the dimensions, edges, and vertices of the rectangular prism
def length : ℕ := 4
def width : ℕ := 3
def height : ℕ := 5
def edges : ℕ := 12
def vertices : ℕ := 8

-- The definition of a diagonal (a segment joining two vertices not joined by an edge)
def is_diagonal (v1 v2 : ℕ) : Prop := v1 ≠ v2 ∧ (∃ (x1 x2 : ℕ), x1 ≠ x2 ∧ ¬ (edges.connected v1 v2))

-- The main theorem to prove
theorem count_diagonals : total_diagonals length width height edges vertices = 16 := sorry

end count_diagonals_l817_817664


namespace derivative_at_one_l817_817606

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 + 1)

theorem derivative_at_one : deriv f 1 = real.sqrt 2 / 2 :=
by
  sorry

end derivative_at_one_l817_817606


namespace determine_omega_l817_817167

-- Definitions of the conditions
def g (ω : ℝ) (x : ℝ) : ℝ := sin (ω * (x - π / 12))

-- Main theorem statement
theorem determine_omega (ω : ℝ) (hₗ : ∀ x : ℝ, x ∈ Icc (π / 6) (π / 3) → monotone_increasing (g ω)) 
                             (hᵣ : ∀ x : ℝ, x ∈ Icc (π / 3) (π / 2) → monotone_decreasing (g ω)) :
    ω = 2 :=
begin
  sorry
end

end determine_omega_l817_817167


namespace problem1_l817_817142

theorem problem1 (x : ℝ) (hx : x > 0) : (x + 1/x = 2) ↔ (x = 1) :=
by
  sorry

end problem1_l817_817142


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817324

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817324


namespace distinct_four_digit_odd_numbers_l817_817044

-- Define the conditions as Lean definitions
def is_odd_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def valid_first_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

-- The proposition we want to prove
theorem distinct_four_digit_odd_numbers (n : ℕ) :
  (∀ d, d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] → is_odd_digit d) →
  valid_first_digit (n / 1000 % 10) →
  1000 ≤ n ∧ n < 10000 →
  n = 500 :=
sorry

end distinct_four_digit_odd_numbers_l817_817044


namespace total_revenue_of_vegetable_sales_l817_817085

theorem total_revenue_of_vegetable_sales:
  let bags_morning_potatoes := 29 in
  let bags_morning_onions := 15 in
  let bags_morning_carrots := 12 in
  let bags_afternoon_potatoes := 17 in
  let bags_afternoon_onions := 22 in
  let bags_afternoon_carrots := 9 in
  let weight_per_bag_potatoes := 7 in
  let weight_per_bag_onions := 5 in
  let weight_per_bag_carrots := 4 in
  let price_per_kg_potatoes := 1.75 in
  let price_per_kg_onions := 2.50 in
  let price_per_kg_carrots := 3.25 in
  let total_revenue :=
    (bags_morning_potatoes + bags_afternoon_potatoes) * weight_per_bag_potatoes * price_per_kg_potatoes +
    (bags_morning_onions + bags_afternoon_onions) * weight_per_bag_onions * price_per_kg_onions +
    (bags_morning_carrots + bags_afternoon_carrots) * weight_per_bag_carrots * price_per_kg_carrots
  in
  total_revenue = 1299.00 :=
by
  sorry

end total_revenue_of_vegetable_sales_l817_817085


namespace carolina_letters_l817_817170

theorem carolina_letters (L P : ℕ) 
(h1 : L = P + 2)
(h2 : 0.37 * L + 0.88 * P = 4.49) : L = 5 := 
by
  sorry

end carolina_letters_l817_817170


namespace unique_distribution_exists_l817_817561

def is_valid_distribution (x : Fin 10 → ℝ) : Prop :=
  (∀ i : Fin 10, x i = (x ((i + 9) % 10) + x ((i + 1) % 10)) / 2) ∧ (∑ i, x i = 10)

theorem unique_distribution_exists :
  ∃! (x : Fin 10 → ℝ), is_valid_distribution x := by
  sorry

end unique_distribution_exists_l817_817561


namespace baker_cakes_total_l817_817685

def initial_cakes : ℕ := 110
def cakes_sold : ℕ := 75
def additional_cakes : ℕ := 76

theorem baker_cakes_total : 
  (initial_cakes - cakes_sold) + additional_cakes = 111 := by
  sorry

end baker_cakes_total_l817_817685


namespace symmetric_product_l817_817160

def z1 : ℂ := 3 + 2 * complex.I
def z2 : ℂ := 2 + 3 * complex.I

theorem symmetric_product :
  z1 * z2 = 13 * complex.I := by
  sorry

end symmetric_product_l817_817160


namespace circle_condition_intersection_condition_l817_817416

-- Define the equation C as a function.
def equation_C (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line l as a function.
def line_l (x y : ℝ) : Prop :=
  4*x - 3*y + 7 = 0

-- Condition that the circle's chord length |MN| = 2√3
def chord_length (m : ℝ) : Prop :=
  abs (4*1 - 3*2 + 7) / sqrt (4^2 + (-3)^2) + sqrt 3 = sqrt (-m + 5)

-- Proof statement (I): if equation C represents a circle, then m < 5.
theorem circle_condition (m : ℝ) :
  (∃ x y, equation_C x y m) → m < 5 :=
sorry

-- Proof statement (II): if circle C intersects line l at points M and N,
-- with chord length |MN| = 2√3, then m = 1.
theorem intersection_condition (m : ℝ) :
  chord_length m → m = 1 :=
sorry

end circle_condition_intersection_condition_l817_817416


namespace nonnegative_integer_count_l817_817045

def balanced_quaternary_nonnegative_count : Nat :=
  let base := 4
  let max_index := 6
  let valid_digits := [-1, 0, 1]
  let max_sum := (base ^ (max_index + 1) - 1) / (base - 1)
  max_sum + 1

theorem nonnegative_integer_count : balanced_quaternary_nonnegative_count = 5462 := by
  sorry

end nonnegative_integer_count_l817_817045


namespace cone_height_from_sector_l817_817255

theorem cone_height_from_sector
  (radius : ℝ) 
  (sector_arc_length : ℝ) 
  (cone_slant_height : ℝ) 
  (cone_base_radius : ℝ) :
  radius = 10 →
  sector_arc_length = 5 * Real.pi →
  cone_base_radius = 5 / 2 →
  cone_slant_height = radius →
  ∃ h : ℝ, h = (5 * Real.sqrt 15) / 2 :=
by
  intros h_radius h_sector h_base h_slant
  have h_radius_eq : radius = 10 := h_radius
  have h_sector_eq : sector_arc_length = 5 * Real.pi := h_sector
  have h_base_eq : cone_base_radius = 5 / 2 := h_base
  have h_slant_eq : cone_slant_height = radius := h_slant
  use (5 * Real.sqrt 15) / 2
  sorry

end cone_height_from_sector_l817_817255


namespace solution_set_l817_817772

section
variables {ℝ : Type*} [Real]
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Given conditions
def f_is_odd : Prop := ∀ x : ℝ, f (-x) = -f (x)
def f_deriv : Prop := ∀ x : ℝ, f' x = (derivative of f) x
def condition_on_f' : Prop := ∀ x : ℝ, x > 0 → x * f' x > 2 * f (-x)

-- Definition of g(x)
def g (x : ℝ) : ℝ := x^2 * f(x)

-- Theorem to prove the solution set
theorem solution_set (h_odd : f_is_odd f) (h_deriv : f_deriv f f') (h_cond : condition_on_f' f f') : 
    ∀ x : ℝ, g x < g (1 - 3 * x) → x < 1 / 4 :=
by
    sorry

end

end solution_set_l817_817772


namespace length_of_central_rectangle_is_2_l817_817287

variable (x : ℕ)  -- Define the length of the central rectangle as a natural number

-- Conditions
axiom carpet_arithmetic_progression : Prop
axiom width_central_rectangle_is_1_foot : x * 1 = x
axiom width_other_shaded_is_1_foot : ∀ l : ℕ, l ∈ {1, 2} → 3 * (x + l) - x + l = 2 * (2 * x + (2 * l - x))

-- Prove that the length of the smallest central rectangle is 2 feet
theorem length_of_central_rectangle_is_2 (h : carpet_arithmetic_progression) :
  x = 2 := sorry

end length_of_central_rectangle_is_2_l817_817287


namespace find_m_for_tangent_circles_l817_817725

-- Define the first circle C1
def circle_C1 (m : ℝ) (x y : ℝ) : Prop :=
(x - m)^2 + (y + 2)^2 = 9

-- Define the second circle C2
def circle_C2 (m : ℝ) (x y : ℝ) : Prop :=
(x + 1)^2 + (y - m)^2 = 4

-- Define the centers and radii of the circles
def center_C1 (m : ℝ) : ℝ × ℝ :=
(m, -2)

def center_C2 (m : ℝ) : ℝ × ℝ :=
(-1, m)

def radius_C1 : ℝ :=
3

def radius_C2 : ℝ :=
2

-- Define the distance between the centers of the circles
def dist_between_centers (m : ℝ) : ℝ :=
Real.sqrt ((m + 1)^2 + (m + 2)^2)

-- Define the condition for circles to be tangent internally
def internally_tangent (m : ℝ) : Prop :=
dist_between_centers m = 1

-- The statement of the problem
theorem find_m_for_tangent_circles (m : ℝ) :
  (circle_C1 m ∧ circle_C2 m ∧ internally_tangent m) ↔ (m = -2 ∨ m = -1) :=
sorry

end find_m_for_tangent_circles_l817_817725


namespace find_angle_C_find_increasing_intervals_of_f_l817_817077

variables {a b c : ℝ}
variables {A B C : ℝ}
variables {m n : ℝ × ℝ}
variables {f : ℝ → ℝ}

def is_dot_product_satisfied (m n : ℝ × ℝ) := (m.1 * n.1 + m.2 * n.2) = (2 + Real.sqrt 3) * a * b
def vector_m := (a + b, -c)
def vector_n := (a + b, c)

theorem find_angle_C (h : is_dot_product_satisfied vector_m vector_n) : C = π / 6 :=
sorry

def function_f (x : ℝ) := 2 * (Real.sin (A + B)) * (Real.cos x) ^ 2 - (Real.cos (A + B)) * (Real.sin (2 * x)) - 1 / 2

theorem find_increasing_intervals_of_f (hA_B: A + B = 5 * π/ 6) :
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), 0 < function_f' x) :=
sorry

end find_angle_C_find_increasing_intervals_of_f_l817_817077


namespace find_12th_term_l817_817569

noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ
| 0 => a
| (n+1) => r * geometric_sequence a r n

theorem find_12th_term : ∃ a r, geometric_sequence a r 4 = 5 ∧ geometric_sequence a r 7 = 40 ∧ geometric_sequence a r 11 = 640 :=
by
  -- statement only, no proof provided
  sorry

end find_12th_term_l817_817569


namespace count_rectangular_parallelepipeds_in_4_cube_l817_817205

theorem count_rectangular_parallelepipeds_in_4_cube : 
  let n := 4 
  in (∃ k, k = (nat.choose (n + 1) 2) ^ 3 ∧ k = 1000) :=
by
  -- Define the number of points per axis which is (n+1)
  let p := 5
  -- Number of ways to choose 2 planes from p points
  let combinations := nat.choose p 2
  -- Total unique parallelepipeds
  let total := combinations ^ 3
  -- We need to prove the total is 1000
  exact ⟨total, by
    have h_combinations : combinations = 10 := rfl
    have h_total : total = 1000 := by rw [←h_combinations]; simp
    exact h_total⟩

end count_rectangular_parallelepipeds_in_4_cube_l817_817205


namespace find_y_l817_817921

-- Define G function
def G (a b c d : ℕ) : ℕ := a^b + c^d

-- Define the conditions
variables (y : ℝ)
axiom G_def : ∀ (a b c d : ℕ), G a b c d = a^b + c^d
axiom condition1 : G 3 y.to_nat 2 5 = 350

-- State the theorem
theorem find_y : y ≈ 6.204 :=
  sorry

end find_y_l817_817921


namespace positive_two_digit_integers_remainder_3_l817_817812

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817812


namespace calculate_chocolate_eggs_l817_817124

/-- Maddy's total number of chocolate eggs calculation -/
theorem calculate_chocolate_eggs (eggs_per_day : ℕ) (weeks : ℕ) (days_per_week : ℕ) 
  (h1 : eggs_per_day = 2) (h2 : weeks = 4) (h3 : days_per_week = 7) : 
  eggs_per_day * weeks * days_per_week = 56 :=
by
  have h4 : weeks * days_per_week = 28, by sorry
  have h5 : eggs_per_day * (weeks * days_per_week) = eggs_per_day * 28, by sorry
  rw [h5, h1] at *,
  have h6 : 2 * 28 = 56, by sorry
  exact h6

end calculate_chocolate_eggs_l817_817124


namespace distance_between_A_and_B_is_zero_l817_817359

open Real

def A : Set ℝ := { y | ∃ x : ℝ, y = 2 * x - 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }

def distance (A B : Set ℝ) : ℝ :=
  Inf { |a - b| | a ∈ A, b ∈ B }

theorem distance_between_A_and_B_is_zero : distance A B = 0 := 
  sorry

end distance_between_A_and_B_is_zero_l817_817359


namespace min_value_AC_l817_817767

/-- Given points A, B, and C where:
- A = (1, 1)
- B lies on the parabola y² = x
- C lies on the parabola y² = x
- ∠ ABC = 90°
Prove that the minimum value of the distance AC is 2. -/
theorem min_value_AC : 
  ∀ (B C : ℝ × ℝ), 
    B.1 = B.2^2 ∧ C.1 = C.2^2 ∧ ∠(1, 1) B C = 90° → 
    ∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, dist (1, 1) x ≥ m :=
sorry

end min_value_AC_l817_817767


namespace maximum_area_l817_817662

-- Define necessary variables and conditions
variables (x y : ℝ)
variable (A : ℝ)
variable (peri : ℝ := 30)

-- Provide the premise that defines the perimeter condition
axiom perimeter_condition : 2 * x + 2 * y = peri

-- Define y in terms of x based on the perimeter condition
def y_in_terms_of_x (x : ℝ) : ℝ := 15 - x

-- Define the area of the rectangle in terms of x
def area (x : ℝ) : ℝ := x * (y_in_terms_of_x x)

-- The statement that needs to be proved
theorem maximum_area : A = 56.25 :=
by sorry

end maximum_area_l817_817662


namespace complement_union_l817_817791

def U := {2, 4, 6, 8, 10}
def A := {2}
def B := {8, 10}

theorem complement_union :
  (U \ (A ∪ B)) = {4, 6} :=
by
  sorry

end complement_union_l817_817791


namespace part1_part2_l817_817381

noncomputable section
def g1 (x : ℝ) : ℝ := Real.log x

noncomputable def f (t : ℝ) : ℝ := 
  if g1 t = t then 1 else sorry  -- Assuming g1(x) = t has exactly one root.

theorem part1 (t : ℝ) : f t = 1 :=
by sorry

def g2 (x : ℝ) (a : ℝ) : ℝ := 
  if x ≤ 0 then x else -x^2 + 2*a*x + a

theorem part2 (a : ℝ) (h : ∃ t : ℝ, f (t + 2) > f t) : a > 1 :=
by sorry

end part1_part2_l817_817381


namespace domain_of_f_l817_817164

def domain_f := {x : ℝ | 2 * x - 3 > 0}

theorem domain_of_f : ∀ x : ℝ, x ∈ domain_f ↔ x > 3 / 2 := 
by
  intro x
  simp [domain_f]
  sorry

end domain_of_f_l817_817164


namespace part_one_part_two_l817_817761

variable {α : Type*}

def a_arith (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def b_geom (b : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, b (n + 1) = b n * q

def seq_a := λ (n : ℕ), 2 * n - 1

def seq_b := λ (n : ℕ), 3 ^ (n - 1)

def seq_c := λ (n : ℕ), seq_a n + (-1)^n * seq_b n

theorem part_one (a b : ℕ → ℝ) (d q : ℝ) (h_a_arith : a_arith a d) (h_b_geom : b_geom b q)
    (h_b2 : b 2 = 3) (h_b3 : b 3 = 9) (h_a1_eq_b1 : a 1 = b 1) (h_a14_eq_b4 : a 14 = b 4) :
    a = seq_a := by
  -- proof omitted
  sorry

theorem part_two (n : ℕ) :
  ∑ k in Finset.range (2 * n), seq_c (k + 1) = 4 * n^2 + 9^n / 4 - 1 / 4 := by
  -- proof omitted
  sorry

end part_one_part_two_l817_817761


namespace scientific_notation_57_million_l817_817276

theorem scientific_notation_57_million : (∃ a n, 1 ≤ |a| ∧ |a| < 10 ∧ 57_000_000 = a * 10^n) ∧ ∃ a n, a = 5.7 ∧ n = 7 :=
by
  sorry

end scientific_notation_57_million_l817_817276


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817335

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817335


namespace solution_set_l817_817039

theorem solution_set (x : ℝ) : 
  1 < |x + 2| ∧ |x + 2| < 5 ↔ 
  (-7 < x ∧ x < -3) ∨ (-1 < x ∧ x < 3) := 
by 
  sorry

end solution_set_l817_817039


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817331

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817331


namespace max_sides_of_polygon_in_1950_gon_l817_817451

theorem max_sides_of_polygon_in_1950_gon (n : ℕ) (h : n = 1950) :
  ∃ (m : ℕ), (m ≤ 1949) ∧ (∀ k, k > m → k ≤ 1949) :=
sorry

end max_sides_of_polygon_in_1950_gon_l817_817451


namespace exists_infinite_set_no_three_collinear_rational_distances_l817_817471

theorem exists_infinite_set_no_three_collinear_rational_distances :
  ∃ (S : Set (ℝ × ℝ)), 
  Set.Infinite S ∧ 
  (∀ (P Q : ℝ × ℝ), P ∈ S → Q ∈ S → P ≠ Q → ∃ d : ℚ, dist P Q = ↑d) ∧ 
  (∀ (P Q R : ℝ × ℝ), P ∈ S → Q ∈ S → R ∈ S → P ≠ Q → P ≠ R → Q ≠ R → 
   ¬ collinear ℝ {P, Q, R}) :=
sorry

end exists_infinite_set_no_three_collinear_rational_distances_l817_817471


namespace sqrt_meaningful_l817_817064

theorem sqrt_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l817_817064


namespace tank_capacity_l817_817099

variable (C : ℝ)

-- Conditions given in the problem
axiom tank_weight_empty : 80 = 80
axiom tank_weight_full_at_80_percent : 1360 / 8 = 170
axiom rainstorm_fill_percent : 0.80 = 80 / 100
axiom weight_per_gallon : 8 = 8

-- The main proof problem
theorem tank_capacity : C = 200 :=
by
  have h1 : 8 * 0.80 * C + 80 = 1360 := sorry
  have h2 : 8 * 0.80 * C = 1280 := sorry
  have h3 : 6.4 * C = 1280 := sorry
  show C = 200, from sorry

end tank_capacity_l817_817099


namespace multiplication_factor_l817_817629

-- Define the original function q
def q (w d z x : ℝ) : ℝ := 5 * w^2 / (4 * d^2 * (z^3 + x^2))

-- The changed variables
def q' (w d z x : ℝ) : ℝ := q (4 * w) (2 * d) (3 * z) (x / 2)

theorem multiplication_factor (w d z x : ℝ) : q' w d z x = (16 / 27) * q w d z x := sorry

end multiplication_factor_l817_817629


namespace building_time_l817_817283

theorem building_time (b p : ℕ) 
  (h1 : b = 3 * p - 5) 
  (h2 : b + p = 67) 
  : b = 49 := 
by 
  sorry

end building_time_l817_817283


namespace number_of_possible_n_values_l817_817089

noncomputable def possible_n_values : Finset Nat := 
  { n | (0 < n ∧ n < 5 ∧ (2 * n + 10 + n + 15 > 3 * n + 5) ∧ (2 * n + 10 + 3 * n + 5 > n + 15) ∧ (n + 15 + 3 * n + 5 > 2 * n + 10)) }.toFinset

theorem number_of_possible_n_values : possible_n_values.card = 4 := 
  by
  sorry

end number_of_possible_n_values_l817_817089


namespace cos_alpha_plus_pi_over_3_l817_817757

theorem cos_alpha_plus_pi_over_3 (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) : Real.cos (α + π / 3) = -1 / 3 :=
  sorry

end cos_alpha_plus_pi_over_3_l817_817757


namespace smallest_k_distinct_primes_sum_squares_power_of_2_l817_817705

theorem smallest_k_distinct_primes_sum_squares_power_of_2 :
  ∃ (k : ℕ), k > 1 ∧ (∀ (p : ℕ → ℕ), (∀ i j, i ≠ j → prime (p i) ∧ prime (p j)) ∧ (∃ n, (∑ i in finset.range k, (p i)^2) = 2^n)) ∧ k = 5 := 
begin
  sorry
end

end smallest_k_distinct_primes_sum_squares_power_of_2_l817_817705


namespace abcd_product_l817_817406

theorem abcd_product :
  let A := (Real.sqrt 3003 + Real.sqrt 3004)
  let B := (-Real.sqrt 3003 - Real.sqrt 3004)
  let C := (Real.sqrt 3003 - Real.sqrt 3004)
  let D := (Real.sqrt 3004 - Real.sqrt 3003)
  A * B * C * D = 1 := 
by
  sorry

end abcd_product_l817_817406


namespace number_of_true_propositions_l817_817785

-- Definitions of the propositions
def P1 : Prop := ∀ (balls boxes : ℕ), balls = 3 ∧ boxes = 2 → ∃ box : ℕ, box < boxes ∧ box > 1
def P2 : Prop := ¬ ∃ (seed : Type), seed ∧ ¬ seed
def P3 : Prop := false
def P4 : Prop := true

-- Main theorem stating the number of true propositions
theorem number_of_true_propositions : (P1 ∧ P2 ∧ ¬P3 ∧ P4) → 3 = 3 := 
by 
  -- The proof is not required here, we just need to state it.
  sorry

end number_of_true_propositions_l817_817785


namespace general_term_of_sequence_l817_817976

theorem general_term_of_sequence (a : Nat → ℚ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) / (2 + a (n + 1))) :
  ∀ n : ℕ, a (n + 1) = 2 / (n + 2) := 
sorry

end general_term_of_sequence_l817_817976


namespace find_n_constant_term_l817_817779

-- Definition: The sum of the coefficients in the expansion is 256
def sum_of_coeffs (n : ℕ) : ℕ := 2^n

-- Theorem I: Given sum_of_coeffs n = 256, find n = 8
theorem find_n (n : ℕ) (h₁ : sum_of_coeffs n = 256) : n = 8 := 
by sorry

-- Definition: General term in the expansion of (x + 1/x)^n
def general_term (n r : ℕ) : ℤ := (Nat.choose n r) * (x^(n - 2*r))

-- Theorem II: Given that n = 8, find the constant term
theorem constant_term (r : ℕ) (h₂ : 8 - 2 * r = 0) : general_term 8 r = 70 := 
by sorry

end find_n_constant_term_l817_817779


namespace problem_l817_817787

def f (x : ℝ) := Real.log x
def g (x : ℝ) := Real.exp x

theorem problem (a : ℝ) (tangent_exists : ∃ (m n : ℝ), m = Real.exp 1 ∧ n = Real.exp 2 ∧ a * n = 1) :
  a = 1 / (Real.exp 2) :=
by
  sorry

end problem_l817_817787


namespace count_interesting_numbers_l817_817838

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l817_817838


namespace average_annual_increase_in_living_space_l817_817289

theorem average_annual_increase_in_living_space 
  (population_2000 : ℝ) 
  (living_space_2000 : ℝ)
  (population_growth_rate : ℝ) 
  (years : ℝ) 
  (goal_living_space_per_person : ℝ) 
  (population_growth_factor : ℝ) 
  (eq_growth_factor : population_growth_factor = 1.01 ^ 10) 
  (eq_goal_living_space : goal_living_space_per_person = 7)
  (eq_living_space_2000 : living_space_2000 = 6 * population_2000 / 1e6)
  (eq_population_2000 : population_2000 = 5 * 1e6)
  : ∃ d : ℝ, d = 86.8 := 
begin
  sorry
end

end average_annual_increase_in_living_space_l817_817289


namespace Vasya_mushrooms_l817_817203

def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def digitsSum (n : ℕ) : ℕ := (n / 100) + ((n % 100) / 10) + (n % 10)

theorem Vasya_mushrooms :
  ∃ n : ℕ, isThreeDigit n ∧ digitsSum n = 14 ∧ n = 950 := 
by
  sorry

end Vasya_mushrooms_l817_817203


namespace two_digit_integers_remainder_3_count_l817_817884

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817884


namespace train_length_l817_817439

/-- Convert speed from kmph to m/s. -/
def kmph_to_mps (v_kmph : ℝ) : ℝ :=
  v_kmph / 3.6

/-- Given conditions: speed of the train in kmph, and time taken to cross the pole in seconds. -/
variables (v : ℝ := 270) (t : ℝ := 5)

/-- The length of the train in meters is calculated as speed in m/s multiplied by time in seconds. -/
theorem train_length : (kmph_to_mps v * t) = 375 := by
  sorry

end train_length_l817_817439


namespace smallest_four_digit_int_equiv_8_mod_9_l817_817217

theorem smallest_four_digit_int_equiv_8_mod_9 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 8 ∧ n = 1007 := 
by
  sorry

end smallest_four_digit_int_equiv_8_mod_9_l817_817217


namespace pen_price_ratio_l817_817316

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l817_817316


namespace find_m_l817_817042

variable (m : ℝ)  -- declare 'm' as a real number

def A : Set ℝ := {-1, 3, 2 * m - 1}
def B : Set ℝ := {3, m^2}

theorem find_m (h : B ⊆ A) : m = 1 :=
by
  sorry

end find_m_l817_817042


namespace value_of_m_minus_n_l817_817753

theorem value_of_m_minus_n (m n : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (m : ℂ) / (1 + i) = 1 - n * i) : m - n = 1 :=
sorry

end value_of_m_minus_n_l817_817753


namespace positive_two_digit_integers_remainder_3_l817_817813

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817813


namespace unique_real_root_in_interval_l817_817017

-- Define the function f
def f (x : ℝ) (b c : ℝ) := x^3 + b * x + c

-- Main theorem to prove the problem statement
theorem unique_real_root_in_interval (b c : ℝ) : 
  (∀ x y ∈ (-1:ℝ)..(1:ℝ), x ≤ y → f x b c ≤ f y b c) ∧ (f (-1) b c * f (1) b c < 0) → 
  (∃! x ∈ Icc (-1:ℝ) 1, f x b c = 0) :=
by
  sorry

end unique_real_root_in_interval_l817_817017


namespace initial_markers_l817_817129

theorem initial_markers (markers_per_box : ℕ) (boxes_bought : ℕ) (total_markers : ℕ) : markers_per_box = 9 → boxes_bought = 6 → total_markers = 86 → initial_markers = 32 :=
by
  intros h1 h2 h3
  let new_markers := boxes_bought * markers_per_box
  let initial_markers := total_markers - new_markers
  have : initial_markers = 32 := by sorry
  exact this

end initial_markers_l817_817129


namespace problem_statement_l817_817026

def f (n : ℕ) (x : ℝ) : ℝ := Real.sin x ^ n + Real.cos x ^ n

theorem problem_statement :
  (∀ x : ℝ, f 1 x ≤ Real.sqrt 2) ∧             -- Statement 1
  (¬(∃ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 * Real.pi ∧ 0 ≤ x2 ∧ x2 ≤ 2 * Real.pi ∧ 0 ≤ x3 ∧ x3 ≤ 2 * Real.pi ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f 2 x1 = 2 * Real.sin x1 + Real.abs (Real.sin x1) ∧ f 2 x2 = 2 * Real.sin x2 + Real.abs (Real.sin x2) ∧ f 2 x3 = 2 * Real.sin x3 + Real.abs (Real.sin x3))) ∧    -- Statement 2
  (¬(∀ x : ℝ, f 3 x = -f 3 (-x))) ∧           -- Statement 3
  (∀ x : ℝ, f 4 x = f 4 (x + Real.pi / 2))     -- Statement 4
:= by
  sorry

end problem_statement_l817_817026


namespace find_m_maximize_profit_l817_817568

variables (x m : ℝ)

-- Define the sales volume function
def sales_volume (x : ℝ) (m : ℝ) : ℝ :=
  (m / (x - 3)) + 8 * (x - 6) ^ 2

-- Given relationship y = sales_volume
axiom sales_condition : sales_volume 5 m = 11

-- Proving m = 6 given the condition
theorem find_m : sales_volume 5 6 = 11 :=
  calc
    sales_volume 5 6
    = 6 / 2 + 8 * (5 - 6) ^ 2 : by simp [sales_volume]
    ... = 3 + 8 * 1 : by norm_num
    ... = 11 : by norm_num

-- Define daily profit function based on sales volume
def profit (x : ℝ) :=
  (x - 3) * (6 / (x - 3) + 8 * (x - 6) ^ 2)

-- Prove that profit is maximized at x = 4
theorem maximize_profit : ∀ x, x = 4 → (profit x) = (profit 4) :=
  sorry

end find_m_maximize_profit_l817_817568


namespace cannot_obtain_fraction_l817_817102

noncomputable def fraction (a b : ℕ) : ℚ := a / b

theorem cannot_obtain_fraction (k n : ℕ) :
  let f_start := fraction 5 8 in
  let f_target := fraction 3 5 in
  ∀ (a b : ℕ), 
    (a = 5 + k ∧ b = 8 + k) ∨ 
    (a = n * 5 ∧ b = n * 8) →
  fraction a b ≠ f_target :=
by
  let f_start := fraction 5 8
  let f_target := fraction 3 5
  assume a b h
  cases h with h1 h2
  -- Add your proof here
  · sorry
  · sorry

end cannot_obtain_fraction_l817_817102


namespace Loris_needs_more_books_l817_817123

noncomputable def books_needed (Loris Darryl Lamont : ℕ) :=
  (Lamont - Loris)

theorem Loris_needs_more_books
  (darryl_books: ℕ)
  (lamont_books: ℕ)
  (loris_books_total: ℕ)
  (total_books: ℕ)
  (h1: lamont_books = 2 * darryl_books)
  (h2: darryl_books = 20)
  (h3: loris_books_total + darryl_books + lamont_books = total_books)
  (h4: total_books = 97) :
  books_needed loris_books_total darryl_books lamont_books = 3 :=
sorry

end Loris_needs_more_books_l817_817123


namespace area_of_triangle_ABC_l817_817169

theorem area_of_triangle_ABC
  (radius : ℝ) (angle_A : ℝ)
  (is_chord : ∀ (A B C : Point) (O : Point), dist O A = radius ∧ dist O B = radius ∧ dist O C = radius → ∃ (M : Point), M = midpoint A B ∧ OM = radius)
  (C_on_diameter : ∀ (C : Point) (AB : Line), parallel AB (diameter_through C))
  (angle_A_75 : angle_A = 75) :
  ∃ (ABC : Triangle), area ABC = 40 := 
sorry

end area_of_triangle_ABC_l817_817169


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817336

-- Definitions
variables {x y : ℕ} -- x: number of ballpoint pens, y: number of gel pens
variables {b g : ℝ} -- b: price of each ballpoint pen, g: price of each gel pen
variables (T : ℝ) -- T: total amount paid

-- Conditions
def condition1 : Prop := (x + y) * g = 4 * T
def condition2 : Prop := (x + y) * b = T / 2
def total_amount : Prop := T = x * b + y * g

-- Proof Problem
theorem gel_pen_is_eight_times_ballpoint_pen
  (h1 : condition1 T)
  (h2 : condition2 T)
  (h3 : total_amount) :
  g = 8 * b :=
sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817336


namespace vector_dot_product_l817_817994

noncomputable def vector_a : ℝ → ℝ → ℝ := sorry
noncomputable def vector_b : ℝ → ℝ → ℝ := sorry

variables (a b : EuclideanSpace ℝ (Fin 3))
hypothesis h_a : ‖a‖ = 4
hypothesis h_b : ‖b‖ = 5

theorem vector_dot_product
    (a b : EuclideanSpace ℝ (Fin 3))
    (h_a : ‖a‖ = 4)
    (h_b : ‖b‖ = 5) :
    (a + b) ⬝ (a - b) = -9 := by
  sorry

end vector_dot_product_l817_817994


namespace volume_inside_sphere_outside_cylinder_l817_817268

noncomputable def sphere_radius := 6
noncomputable def cylinder_diameter := 8
noncomputable def sphere_volume := 4/3 * Real.pi * (sphere_radius ^ 3)
noncomputable def cylinder_height := Real.sqrt ((sphere_radius * 2) ^ 2 - (cylinder_diameter) ^ 2)
noncomputable def cylinder_volume := Real.pi * ((cylinder_diameter / 2) ^ 2) * cylinder_height
noncomputable def volume_difference := sphere_volume - cylinder_volume

theorem volume_inside_sphere_outside_cylinder:
  volume_difference = (288 - 64 * Real.sqrt 5) * Real.pi :=
sorry

end volume_inside_sphere_outside_cylinder_l817_817268


namespace find_unknown_number_l817_817945

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end find_unknown_number_l817_817945


namespace cell_phone_height_l817_817585

theorem cell_phone_height (width perimeter : ℕ) (h1 : width = 9) (h2 : perimeter = 46) : 
  ∃ length : ℕ, length = 14 ∧ perimeter = 2 * (width + length) :=
by
  sorry

end cell_phone_height_l817_817585


namespace proof_problem_l817_817727

-- Define the range of m
def m_range := {m : ℕ // 1 ≤ m ∧ m ≤ 10}

-- Define the condition for a_m
def valid_am (a m : ℕ) : Prop := a < m

-- Define the least common multiple of the set 1 to 10
def lcm_1_10 := Nat.lcm (Finset.range 11).filter (λ m => 1 ≤ m)

-- Calculate 100a + b for the probability condition
def target_value (a b : ℕ) : ℕ := 100 * a + b

-- Main theorem statement with a placeholder for the final proof
theorem proof_problem : ∃ (a b : ℕ), (p = (1 : ℚ) / 1440) ∧ Nat.Gcd a b = 1 ∧ target_value a b = 1540 :=
by
  sorry

end proof_problem_l817_817727


namespace tom_finishes_in_four_hours_l817_817127

noncomputable def maryMowingRate := 1 / 3
noncomputable def tomMowingRate := 1 / 6
noncomputable def timeMaryMows := 1
noncomputable def remainingLawn := 1 - (timeMaryMows * maryMowingRate)

theorem tom_finishes_in_four_hours :
  remainingLawn / tomMowingRate = 4 :=
by sorry

end tom_finishes_in_four_hours_l817_817127


namespace smallest_four_digit_int_equiv_8_mod_9_l817_817216

theorem smallest_four_digit_int_equiv_8_mod_9 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 8 ∧ n = 1007 := 
by
  sorry

end smallest_four_digit_int_equiv_8_mod_9_l817_817216


namespace max_divided_subset_cardinality_l817_817498

/-- Define a binary string as a list of bits (0 or 1) -/
def BinaryString (n : ℕ) := Vector Bool n

/-- Define a twist function for a binary string -/
def twist {n : ℕ} (s : BinaryString n) : BinaryString n :=
  let b := (s.val.chunkWhile id).length
  ⟨(s.val.take (b - 1) ++ [!s.val.nth! (b - 1)] ++ s.val.drop b).take n, sorry⟩

/-- Define descendant of a binary string if it can be obtained through finite twists -/
def is_descendant {n : ℕ} (a b : BinaryString n) : Prop :=
  ∃ (k : ℕ), (iterate twist k b) = a

/-- A subset of binary strings is divided if no two members have a common descendant -/
def is_divided {n : ℕ} (S : set (BinaryString n)) : Prop :=
  ∀ a b ∈ S, a ≠ b → ¬ ∃ c, is_descendant c a ∧ is_descendant c b

/-- The main theorem stating the largest possible cardinality of a divided subset of B_n -/
theorem max_divided_subset_cardinality (n : ℕ) (h : 0 < n) :
  ∃ S : set (BinaryString n), is_divided S ∧ S.card = 2^(n-2) :=
sorry

end max_divided_subset_cardinality_l817_817498


namespace cos_of_difference_l817_817756

noncomputable def alpha := sorry -- α is a given angle
def cos_condition := cos (75 * Real.pi / 180 + alpha / 2) = Real.sqrt 3 / 3

theorem cos_of_difference (h : cos_condition) : cos (30 * Real.pi / 180 - alpha) = 1 / 3 := 
by
  sorry

end cos_of_difference_l817_817756


namespace sum_of_factors_1656_l817_817086

theorem sum_of_factors_1656 : ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 1656 ∧ a + b = 110 := by
  sorry

end sum_of_factors_1656_l817_817086


namespace max_volume_tetrahedron_l817_817465

-- Definitions and conditions
def SA : ℝ := 4
def AB : ℝ := 5
def SB_min : ℝ := 7
def SC_min : ℝ := 9
def BC_max : ℝ := 6
def AC_max : ℝ := 8

-- Proof statement
theorem max_volume_tetrahedron {SB SC BC AC : ℝ} (hSB : SB ≥ SB_min) (hSC : SC ≥ SC_min) (hBC : BC ≤ BC_max) (hAC : AC ≤ AC_max) :
  ∃ V : ℝ, V = 8 * Real.sqrt 6 ∧ V ≤ (1/3) * (1/2) * SA * AB * (2 * Real.sqrt 6) * BC := by
  sorry

end max_volume_tetrahedron_l817_817465


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817913

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817913


namespace ratio_men_to_women_l817_817241

theorem ratio_men_to_women (M W : ℕ) (h1 : W = M + 4) (h2 : M + W = 18) : M = 7 ∧ W = 11 :=
by
  sorry

end ratio_men_to_women_l817_817241


namespace exists_marking_on_board_l817_817126

-- Defining the \(8 \times 8\) board
def board : Type := fin 8 × fin 8

-- Condition: Any cell shares a side with exactly one marked cell
def shares_side_with_exactly_one_marked (marked : board → bool) (cell : board) : Prop :=
  let neighbors := [(1, 0), (-1, 0), (0, 1), (0, -1)] 
  ∑ (delta : int × int) in neighbors, 
    if h : ∃ (i j : int), cell.1.1 + delta.1 = i ∧ cell.2.1 + delta.2 = j then
      marked (⟨int.to_nat h.some.1, h.some.1_is_lt⟩, ⟨int.to_nat h.some.2, h.some.2_is_lt⟩)
    else false = 1

-- The main theorem stating there exists such a configuration on the \(8 \times 8\) board
theorem exists_marking_on_board : ∃ marked : board → bool, ∀ cell : board, shares_side_with_exactly_one_marked marked cell :=
sorry

end exists_marking_on_board_l817_817126


namespace minimum_length_AB_l817_817739

open Real

/-- Given a point (a, b) on the circle x^2 + y^2 = 1, find the minimum length of the line segment AB, where A is the x-intercept and B is the y-intercept of the tangent line at (a, b). -/
theorem minimum_length_AB (a b : ℝ) (h : a^2 + b^2 = 1) : 
  let A := (1 / a, 0),
      B := (0, 1 / b),
      length_AB := dist A B in 
  length_AB >= 2 :=
begin
  -- Proof would go here
  sorry
end

end minimum_length_AB_l817_817739


namespace rabbit_escape_strategy_l817_817462

/-- In the center of a square is a rabbit and at each vertex of this even square, a wolf.
The wolves only move along the sides of the square and the rabbit moves freely in the plane.
We define the following:
- The rabbit's speed is 10 km/h.
- The wolves' speed is 14 km/h.
Prove that there exists a strategy for the rabbit to leave the square without being caught by the wolves. -/
theorem rabbit_escape_strategy (rabbit_speed : ℝ) (wolf_speed : ℝ) : 
  rabbit_speed = 10 → 
  wolf_speed = 14 → 
  ∃ escape_strategy, (∃ ε > 0, runnable ε rabbit_speed wolf_speed escape_strategy) :=
sorry

end rabbit_escape_strategy_l817_817462


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817328

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817328


namespace smallest_two_digit_integer_l817_817613

-- Define the problem parameters and condition
theorem smallest_two_digit_integer (n : ℕ) (a b : ℕ) 
  (h1 : n = 10 * a + b) 
  (h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
  (h6 : 19 * a = 8 * b + 3) : 
  n = 12 :=
sorry

end smallest_two_digit_integer_l817_817613


namespace find_acute_angles_of_right_triangle_l817_817081

theorem find_acute_angles_of_right_triangle
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_right : a^2 + b^2 = c^2)
  (h_ratio : (a + b - c) / c = 4 / 5) :
  (arctan (a / b) = arctan (3 / 4) ∨ arccot (a / b) = arccot (3 / 4)) ∧
  (arccos ((a + b)/c * sqrt 2 / 2 ^ (1 : ℝ)) + π / 4 = arccos (7 * sqrt 2 / 10) ∨
   arccos ((a + b)/c * sqrt 2 / 2 ^ (1 : ℝ)) - π / 4 = arccos (7 * sqrt 2 / 10)) :=
sorry

end find_acute_angles_of_right_triangle_l817_817081


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817323

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817323


namespace johns_quadratic_l817_817486

theorem johns_quadratic (d e : ℤ) (h1 : d^2 = 16) (h2 : 2 * d * e = -40) : d * e = -20 :=
sorry

end johns_quadratic_l817_817486


namespace total_pieces_of_paper_l817_817546

/-- Definitions according to the problem's conditions -/
def pieces_after_first_cut : Nat := 10

def pieces_after_second_cut (initial_pieces : Nat) : Nat := initial_pieces + 9

def pieces_after_third_cut (after_second_cut_pieces : Nat) : Nat := after_second_cut_pieces + 9

def pieces_after_fourth_cut (after_third_cut_pieces : Nat) : Nat := after_third_cut_pieces + 9

/-- The main theorem stating the desired result -/
theorem total_pieces_of_paper : 
  pieces_after_fourth_cut (pieces_after_third_cut (pieces_after_second_cut pieces_after_first_cut)) = 37 := 
by 
  -- The proof would go here, but it's omitted as per the instructions.
  sorry

end total_pieces_of_paper_l817_817546


namespace tan_pi_over_3_of_point_on_graph_l817_817072

theorem tan_pi_over_3_of_point_on_graph (h : ∃ a : ℝ, (a, 9) ∈ {p : ℝ × ℝ | p.snd = 3^p.fst}) :
  ∃ a : ℝ, tan (a * Real.pi / 6) = Real.sqrt 3 :=
by
  obtain ⟨a, ha⟩ := h
  have h_eq : 3^a = 9 := by simpa using ha
  have a_eq : a = 2 := by sorry -- This would be proven based on the equality 3^a = 9 in a full proof
  use a
  rw [a_eq]
  have : 2 * Real.pi / 6 = Real.pi / 3 := by norm_num
  rw [this]
  exact Real.tan_pi_div_three

end tan_pi_over_3_of_point_on_graph_l817_817072


namespace dissection_equal_areas_l817_817493

-- Let ABC be an acute triangle with altitudes AD, BE, CF
variables {A B C D E F O : Type}
variables [AcuteTriangle A B C]
variables [Altitude D AD] [Altitude E BE] [Altitude F CF]
variables [Circumcenter O ABC]

-- Assert the segments dissect the triangle into equal area pairs
theorem dissection_equal_areas 
  (OA : Segment O A) (OF : Segment O F) 
  (OB : Segment O B) (OD : Segment O D) 
  (OC : Segment O C) (OE : Segment O E) :
  ∃ p q r s t u : Triangle, 
    Segment O A D E = p ∧ Segment O B D F = q ∧ Segment O C E F = r ∧ Segment O A C F = s ∧ Segment O B C D = t ∧ Segment O A B E = u ∧ 
    Area p = Area q ∧ Area r = Area s ∧ Area t = Area u :=
sorry

end dissection_equal_areas_l817_817493


namespace gel_pen_price_ratio_l817_817298

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l817_817298


namespace triangle_MYD_area_l817_817090

variables (X Y Z M N D : Type) [linear_ordered_field Type] 
open_locale big_operators

-- Definitions from conditions
def right_triangle (X Y Z : Type) [linear_ordered_field X] : Prop := 
  ∃ (a b c : X), a^2 + b^2 = c^2

def midpoint (M Y Z : Type) [linear_ordered_field Y] : Prop := 
  M = (Y + Z) / 2 

def perpendicular (ND YZ : Type) [linear_ordered_field ND] : Prop := 
  ND * YZ = 0

def area (X Y Z : Type) [linear_ordered_field X] : X := 
  50 

-- Theorem that needs to be proven
theorem triangle_MYD_area (X Y Z M N D : Type) [field X] 
  (h_right_triangle : right_triangle X Y Z) 
  (h_midpoint : midpoint M Y Z)
  (h_perpendicular : perpendicular ND YZ) 
  (h_area_tri_XYZ : area X Y Z = 50) : 
  area M Y D = 25 :=
sorry

end triangle_MYD_area_l817_817090


namespace angle_is_in_second_quadrant_l817_817635

-- Define the function to determine the quadrant of an angle
def quadrant (θ : ℝ) : ℕ :=
if (θ % (2 * Real.pi)) < Real.pi / 2 then 1
else if (θ % (2 * Real.pi)) < Real.pi then 2
else if (θ % (2 * Real.pi)) < 3 * Real.pi / 2 then 3
else 4

-- Statement asserting the given condition
theorem angle_is_in_second_quadrant : quadrant (-10 * Real.pi / 3) = 2 := 
sorry

end angle_is_in_second_quadrant_l817_817635


namespace optimal_service_life_and_min_average_annual_cost_l817_817250

noncomputable def total_cost (purchase_cost transport_install_cost annual_insurance_cost : ℕ) (maintenance_costs: ℕ → ℕ) (n: ℕ) : ℕ :=
  purchase_cost + transport_install_cost + annual_insurance_cost * n + (Finset.sum (Finset.range n) maintenance_costs)

noncomputable def average_annual_cost (purchase_cost transport_install_cost annual_insurance_cost : ℕ) (maintenance_costs: ℕ → ℕ) (n: ℕ) : ℚ :=
  (total_cost purchase_cost transport_install_cost annual_insurance_cost maintenance_costs n) / n

def maintenance_costs (n: ℕ) : ℕ := 2000 + 1000 * n

theorem optimal_service_life_and_min_average_annual_cost :
  let purchase_cost := 70000
  let transport_install_cost := 2000
  let annual_insurance_cost := 2000
  let maintenance_costs := maintenance_costs
  in (∃ n: ℕ, n = 12 ∧ average_annual_cost purchase_cost transport_install_cost annual_insurance_cost maintenance_costs n = 1.55 * 10000) :=
by
  sorry

end optimal_service_life_and_min_average_annual_cost_l817_817250


namespace num_people_approximately_8_l817_817591

noncomputable def total_bill := 211.0
noncomputable def tip_percentage := 0.15
noncomputable def total_with_tip := total_bill * (1 + tip_percentage)
noncomputable def each_person_share := 30.33125

theorem num_people_approximately_8 :
  (total_with_tip / each_person_share) ≈ 8 := 
by
  sorry

end num_people_approximately_8_l817_817591


namespace find_cost_price_l817_817238

/-- Statement: Given Mohit sold an article for $18000 and 
if he offered a discount of 10% on the selling price, he would have earned a profit of 8%, 
prove that the cost price (CP) of the article is $15000. -/

def discounted_price (sp : ℝ) := sp - (0.10 * sp)
def profit_price (cp : ℝ) := cp * 1.08

theorem find_cost_price (sp : ℝ) (discount: sp = 18000) (profit_discount: profit_price (discounted_price sp) = discounted_price sp):
    ∃ (cp : ℝ), cp = 15000 :=
by
    sorry

end find_cost_price_l817_817238


namespace count_two_digit_integers_remainder_3_l817_817826

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817826


namespace integers_congruent_to_3_mod_7_count_integers_congruent_to_3_mod_7_l817_817795

theorem integers_congruent_to_3_mod_7 (x : ℕ) :
  (∃ n : ℕ, x = 7 * n + 3 ∧ 1 ≤ x ∧ x ≤ 300) ↔ 1 ≤ x ∧ x ≤ 300 ∧ x % 7 = 3 :=
begin
  sorry
end

theorem count_integers_congruent_to_3_mod_7 :
  {x | 1 ≤ x ∧ x ≤ 300 ∧ x % 7 = 3}.finite.to_finset.card = 43 :=
begin
  sorry
end

end integers_congruent_to_3_mod_7_count_integers_congruent_to_3_mod_7_l817_817795


namespace lines_intersect_intersection_on_ellipse_l817_817122

section IntersectionAndEllipse
variables {k1 k2 : ℝ}

-- Given conditions:
-- Line l1: y = k1 * x + 1
-- Line l2: y = k2 * x - 1
-- and k1 * k2 + 2 = 0
def line1 (x : ℝ) : ℝ := k1 * x + 1
def line2 (x : ℝ) : ℝ := k2 * x - 1
def ellipse (x y : ℝ) : Prop := 2 * x^2 + y^2 = 1
axiom k1k2_condition : k1 * k2 + 2 = 0

-- Prove l1 and l2 intersect
theorem lines_intersect : ∃ x y : ℝ, line1 x = y ∧ line2 x = y :=
by sorry

-- Prove the intersection point lies on the ellipse 2x^2 + y^2 = 1
theorem intersection_on_ellipse : ∃ x y : ℝ, line1 x = y ∧ line2 x = y ∧ ellipse x y :=
by sorry
end IntersectionAndEllipse

end lines_intersect_intersection_on_ellipse_l817_817122


namespace part1_part2_l817_817029

-- Function definition
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 4 * k

-- Conditions for part (1): ∀ x ∈ ℝ, f(x) < 0 implies k < -1/2
theorem part1 (k : ℝ) (h : ∀ x : ℝ, f(k, x) < 0) : k < -1/2 :=
by sorry

-- Conditions for part (2): f(x) is monotonically decreasing on [2, 4] implies k ≤ 1/4
theorem part2 (k : ℝ) (h : ∀ x y : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ 2 ≤ y ∧ y ≤ 4 ∧ x < y → f(k, x) ≥ f(k, y)): k ≤ 1/4 :=
by sorry

end part1_part2_l817_817029


namespace suitable_sampling_method_l817_817194

theorem suitable_sampling_method 
  (A: "Randomly select all students from 6 junior high schools in urban areas" -> Prop)
  (B: "Randomly select all female students from 3 junior high schools in urban and rural areas each" -> Prop)
  (C: "Randomly select 1000 students from each of the three grades in junior high schools in my city" -> Prop)
  (D: "Randomly select 5000 students from the seventh grade in junior high schools in my city" -> Prop):
  C :=
by 
  sorry

end suitable_sampling_method_l817_817194


namespace interval_of_monotonic_decrease_l817_817422

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := sin (ω * x) + real.sqrt 3 * cos (ω * x)

theorem interval_of_monotonic_decrease (ω : ℝ) (hω : ω > 0)
  (h : ∃ x₀ x₁ : ℝ, x₁ - x₀ = π ∧ f x₀ ω = -2 ∧ f x₁ ω = -2) :
  ∀ k : ℤ, (k * real.pi + real.pi / 12) ≤ x ∧ x ≤ (k * real.pi + 7 * real.pi / 12) → 
  f x ω ≤ f (k * real.pi + real.pi / 12) ω :=
sorry

end interval_of_monotonic_decrease_l817_817422


namespace ellipse_properties_max_chord_length_l817_817750

variables {a b c : ℝ} {m : ℝ} (P : ℝ → Prop)

noncomputable def ellipse_equation (x y : ℝ) : Prop := (x ^ 2 / (a ^ 2)) + (y ^ 2 / (b ^ 2)) = 1

theorem ellipse_properties 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : c / a = (real.sqrt 3) / 2) 
  (h4 : 4 * a = 8) 
  (h5 : P m) 
  (h6 : ∃ x y, P x ∧ P y ∧ ellipse_equation x y) :
  ellipse_equation (2 : ℝ) (1 : ℝ) :=
begin
  sorry
end

theorem max_chord_length {x1 x2 y1 y2 : ℝ} :
  ∃ A B : ℝ × ℝ, A ≠ B ∧ (a = 2) ∧ (b = 1) ∧ 
         max {d | ∃ l, ∀ m (hm : 1 ≤ abs m), line L (l m) ∧ dist A B = d} = 2 :=
begin
  sorry
end

end ellipse_properties_max_chord_length_l817_817750


namespace num_rem_three_by_seven_l817_817852

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l817_817852


namespace min_sum_product_l817_817366

theorem min_sum_product : 
  ∀ (b : Fin 150 → Int), 
  (∀ i, b i = 2 ∨ b i = -2) →
  (∃ T, T = ∑ i in Finset.range 150, ∑ j in Finset.range i, b i * b j ∧ T > 0 ∧ 
         (∀ T', T' = ∑ i in Finset.range 150, ∑ j in Finset.range i, b i * b j ∧ T' > 0 → T ≤ T')) :=
by
  sorry

end min_sum_product_l817_817366


namespace slope_range_as_angles_l817_817179

theorem slope_range_as_angles : 
  ∀ (m : ℝ), ∃ θ ∈ set.Ico 0 π, θ = real.arctan m :=
by
  sorry

end slope_range_as_angles_l817_817179


namespace fence_perimeter_l817_817188

theorem fence_perimeter 
  (N : ℕ) (w : ℝ) (g : ℝ) 
  (square_posts : N = 36) 
  (post_width : w = 0.5) 
  (gap_length : g = 8) :
  4 * ((N / 4 - 1) * g + (N / 4) * w) = 274 :=
by
  sorry

end fence_perimeter_l817_817188


namespace discarded_number_l817_817157

theorem discarded_number (S S_48 : ℝ) (h1 : S = 1000) (h2 : S_48 = 900) (h3 : ∃ x : ℝ, S - S_48 = 45 + x): 
  ∃ x : ℝ, x = 55 :=
by {
  -- Using the conditions provided to derive the theorem.
  sorry 
}

end discarded_number_l817_817157


namespace isosceles_trapezoid_min_x_squared_l817_817504

theorem isosceles_trapezoid_min_x_squared
  (AB CD : ℝ) (h_AB : AB = 100) (h_CD : CD = 25)
  (x : ℝ) (h_isosceles : is_isosceles_trapezoid AB CD x)
  (h_tangent : ∃ (M : ℝ), is_circle_centered_ab_tangent_AD_BC M x) :
  x ^ 2 = 1875 :=
sorry

end isosceles_trapezoid_min_x_squared_l817_817504


namespace center_of_outer_polygon_within_inner_polygon_l817_817267

theorem center_of_outer_polygon_within_inner_polygon 
  (n : ℕ) (a : ℝ) 
  (M1 : Type) [regular_polygon M1 (2 * n) a]
  (M2 : Type) [regular_polygon M2 (2 * n) (2 * a)]
  (within : is_inside M1 M2) :
  contains_center_of M2 M1 :=
sorry

end center_of_outer_polygon_within_inner_polygon_l817_817267


namespace coffee_order_cost_l817_817475

theorem coffee_order_cost :
  let drip_coffee_price := 2.25
  let double_shot_espresso_price := 3.50
  let latte_price := 4.00
  let vanilla_syrup_price := 0.50
  let cold_brew_price := 2.50
  let cappuccino_price := 3.50
  let total_cost := 
    (2 * drip_coffee_price) + 
    (1 * double_shot_espresso_price) + 
    (2 * latte_price) + 
    (1 * vanilla_syrup_price) + 
    (2 * cold_brew_price) + 
    (1 * cappuccino_price) 
  in
  total_cost = 25.00 :=
by sorry

end coffee_order_cost_l817_817475


namespace quadrilateral_inequality_l817_817993

-- Define points and necessary conditions
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Define that A, B, C, D form a quadrilateral
def quadrilateral (A B C D : Type) := True -- Placeholder, define properly based on Lean's geometry constructs

-- Main theorem
theorem quadrilateral_inequality (A B C D : Type) [quadrilateral A B C D] :
  dist A C * dist B D ≤ dist A B * dist C D + dist A D * dist B C :=
sorry

end quadrilateral_inequality_l817_817993


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817890

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817890


namespace parabola_distance_l817_817771

theorem parabola_distance (P A F : ℝ × ℝ) (h1 : P.1 * P.1 = -8 * P.1) (h2 : F = (-2, 0))
    (h3 : A = (2, F.2 + 4 * real.sqrt 3)) (h4 : A = (2, 4 * real.sqrt 3)) 
    (h5 : P = (P.1, 4 * real.sqrt 3)) :
    |PF| = 8 := 
begin
  -- Definition of distance
  let PF := real.sqrt ((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2),
  exact 8,
  sorry
end

end parabola_distance_l817_817771


namespace pen_price_ratio_l817_817315

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l817_817315


namespace period_of_sin_plus_sqrt3_cos_l817_817611

theorem period_of_sin_plus_sqrt3_cos : ∃ T > 0, ∀ x, sin x + sqrt 3 * cos x = sin (x + T) + sqrt 3 * cos (x + T) ∧ T = 2 * π := by
  sorry

end period_of_sin_plus_sqrt3_cos_l817_817611


namespace two_hundredth_digit_of_7_over_29_l817_817610

theorem two_hundredth_digit_of_7_over_29 :
  (decimal_places ⟨7, 29⟩ 200) = 1 :=
sorry

end two_hundredth_digit_of_7_over_29_l817_817610


namespace count_non_divisible_by_7_and_8_l817_817719

theorem count_non_divisible_by_7_and_8: 
    let count_up_to (N : ℕ) (d : ℕ) : ℕ := N / d in
    let N := 2000 in
    N - ((count_up_to N 7) + (count_up_to N 8) - (count_up_to N 56)) = 1500 := 
by
  sorry

end count_non_divisible_by_7_and_8_l817_817719


namespace count_two_digit_integers_remainder_3_l817_817821

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817821


namespace grid_to_black_probability_l817_817643

theorem grid_to_black_probability :
  let n := 16
  let p_black_after_rotation := 3 / 4
  (p_black_after_rotation ^ n) = (3 / 4) ^ 16 :=
by
  -- Proof goes here
  sorry

end grid_to_black_probability_l817_817643


namespace concatenated_naturals_irrational_l817_817140

theorem concatenated_naturals_irrational (f : ℕ → ℕ) : irrational (∑ i in finset.range 99999, (f i) / (10^i) : ℝ) := sorry

end concatenated_naturals_irrational_l817_817140


namespace largest_angle_sine_of_C_l817_817956

-- Given conditions
def side_a : ℝ := 7
def side_b : ℝ := 3
def side_c : ℝ := 5

-- 1. Prove the largest angle
theorem largest_angle (a b c : ℝ) (h₁ : a = 7) (h₂ : b = 3) (h₃ : c = 5) : 
  ∃ A : ℝ, A = 120 :=
by
  sorry

-- 2. Prove the sine value of angle C
theorem sine_of_C (a b c A : ℝ) (h₁ : a = 7) (h₂ : b = 3) (h₃ : c = 5) (h₄ : A = 120) : 
  ∃ sinC : ℝ, sinC = 5 * (Real.sqrt 3) / 14 :=
by
  sorry

end largest_angle_sine_of_C_l817_817956


namespace geometric_sequence_arithmetic_condition_l817_817377

noncomputable def geometric_sequence_ratio : Real :=
  let a1 := 1  -- Without loss of generality, assume a1 = 1 for simplicity
  let q : Real := (Real.sqrt 5 + 1) / 2  -- The positive solution of q^2 - q - 1 = 0
  let a2 := a1 * q
  let a3 := a2 * q
  let a4 := a3 * q
  let a5 := a4 * q
  (a3 + a4) / (a4 + a5)

theorem geometric_sequence_arithmetic_condition (q : Real) (hq : q > 0 ∧ q ≠ 1)
  (h_arith : let a1 := 1 in let a2 := a1 * q in let a3 := a2 * q in (2 * a2 + a3 = 2 * a1)) :
  geometric_sequence_ratio = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end geometric_sequence_arithmetic_condition_l817_817377


namespace sandy_initial_cost_l817_817144

theorem sandy_initial_cost 
  (repairs_cost : ℝ)
  (selling_price : ℝ)
  (gain_percent : ℝ)
  (h1 : repairs_cost = 200)
  (h2 : selling_price = 1400)
  (h3 : gain_percent = 40) :
  ∃ P : ℝ, P = 800 :=
by
  -- Proof steps would go here
  sorry

end sandy_initial_cost_l817_817144


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817892

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817892


namespace division_remainder_l817_817615

theorem division_remainder :
  ∃ (r : ℝ), ∀ (z : ℝ), (4 * z^3 - 5 * z^2 - 17 * z + 4) = (4 * z + 6) * (z^2 - 4 * z + 1/2) + r ∧ r = 1 :=
sorry

end division_remainder_l817_817615


namespace nancy_packs_of_crayons_l817_817131

def total_crayons : ℕ := 615
def crayons_per_pack : ℕ := 15

theorem nancy_packs_of_crayons : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end nancy_packs_of_crayons_l817_817131


namespace constant_term_in_expansion_l817_817161

theorem constant_term_in_expansion : 
  ∃ (r : ℕ), (12 - 6 * r = 0) ∧ binom 6 2 = 15 :=
by
  -- Definitions from conditions
  have h_expansion := ∀ x : ℝ, (x^2 - x^(-4))^6
  -- Given the logical steps to reach the conclusion
  sorry

end constant_term_in_expansion_l817_817161


namespace price_ratio_l817_817342

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l817_817342


namespace trapezoid_division_l817_817592

variables {A B C D P Q : Type} [AffineSpace A B]
local notation "→" := AffineMap.lineMap

/-- Given a trapezoid ABCD with legs AD and BC intersecting at P, and diagonals AC and BD intersecting at Q,
  prove that PQ divides the bases AB and CD in the ratio 1:1. -/
theorem trapezoid_division (hP : ∃ (P : A), Line (P -ᵥ A) (P -ᵥ D)) (hQ : ∃ (Q : A), Line (Q -ᵥ A) (Q -ᵥ C)) :
  Line (Q -ᵥ B) (Q -ᵥ D) → (P -ᵥ A) = (Q -ᵥ A) → (P -ᵥ D) = (Q -ᵥ D) → ∀ (PQ : A), Line (PQ -ᵥ P) (PQ -ᵥ Q) →
  AffineMap.lineMap P Q (Q -ᵥ B) = AffineMap.lineMap P Q (Q -ᵥ D) :=
sorry


end trapezoid_division_l817_817592


namespace infinite_rational_points_l817_817469

noncomputable def infinite_points_set (S : Set (ℝ × ℝ)) : Prop :=
  (Set.Infinite S) ∧ 
  (∀ (A B C : (ℝ × ℝ)), (A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C) → 
    ¬ ∃ (k : ℝ), colinear k A B C) ∧
  (∀ (A B : (ℝ × ℝ)), (A ∈ S ∧ B ∈ S ∧ A ≠ B) → 
    ∃ (r ∈ ℚ), dist A B = r)

theorem infinite_rational_points :
  ∃ S : Set (ℝ × ℝ), infinite_points_set S := 
sorry

end infinite_rational_points_l817_817469


namespace solve_for_y_l817_817922

-- Define the function G
def G (a b c d : ℝ) : ℝ := a ^ b + c ^ d

-- The theorem to be proven
theorem solve_for_y : ∃ y : ℝ, G 3 y 2 5 = 350 ↔ 3 ^ y = 318 := 
by
  -- Start the proof with declaration
  existsi (Real.log 318 / Real.log 3)
  sorry

end solve_for_y_l817_817922


namespace triangle_area_l817_817076

-- Definitions of the given conditions
variables {A B C : ℝ} {a b c : ℝ}

-- Condition definitions
def c_def : c = 4 := sorry
def tan_A_def : real.tan A = 3 := sorry
def cos_C_def : real.cos C = (sqrt 5) / 5 := sorry

-- The proof problem statement
theorem triangle_area : 
  (c = 4) →
  (real.tan A = 3) →
  (real.cos C = (sqrt 5) / 5) →
  (1 / 2 * c * 3 * sqrt 2 * (sqrt 2 / 2) = 6) :=
by { sorry }

end triangle_area_l817_817076


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817889

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l817_817889


namespace arithmetic_seq_transformations_l817_817925

variable {α : Type} [Add α] [Mul α] [Sub α] [HasSmul ℕ α]

def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) - a n = d

theorem arithmetic_seq_transformations (a : ℕ → ℤ) (d : ℤ) (h : is_arithmetic_sequence a d) :
  is_arithmetic_sequence (λ n, a n + 3) d ∧
  is_arithmetic_sequence (λ n, 2 * a n) (2 * d) := sorry

end arithmetic_seq_transformations_l817_817925


namespace flow_equivalence_l817_817935

variables {G : Type} [Graph G]
variables {H H' : Type} [Fintype H] [Fintype H'] [AbelianGroup H] [AbelianGroup H']
variables (k : ℤ)

-- Definition of an H-flow on a graph G (you may need a more detailed definition in practice)
def has_H_flow (G : Type) [Graph G] (H : Type) [Fintype H] [AbelianGroup H] : Prop :=
sorry -- placeholder for the actual definition

-- The theorem statement
theorem flow_equivalence (H H' : Type) [Fintype H] [Fintype H'] [AbelianGroup H] [AbelianGroup H']
  (h_same_order : Fintype.card H = Fintype.card H') :
  (has_H_flow G H) ↔ (has_H_flow G H') :=
sorry

end flow_equivalence_l817_817935


namespace area_of_quadrilateral_ABCD_l817_817663

noncomputable def point := ℝ × ℝ × ℝ

def A : point := (0, 0, 0)
def D : point := (2, 3, 0)
def B : point := (0, 0, 0.5)
def C : point := (0, 1.5, 0)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

def quadrilateral_area (A B C D : point) : ℝ :=
  0.5 * (distance A B) + 0.5 * (distance C D) * (1.34)

theorem area_of_quadrilateral_ABCD :
  quadrilateral_area A B C D = 1.005 :=
by
  sorry

end area_of_quadrilateral_ABCD_l817_817663


namespace gel_pen_ratio_l817_817311

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end gel_pen_ratio_l817_817311


namespace count_two_digit_integers_remainder_3_l817_817819

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817819


namespace pen_price_ratio_l817_817314

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l817_817314


namespace win_if_start_B_l817_817395

theorem win_if_start_B {n : ℕ} (hn : 0 < n)
  (necklace : ℕ → ℕ)
  (adj_diff : ∀ i, necklace i ≠ necklace (i + 1) ∧ necklace (i + 1) ≠ necklace (i + 2) ∧ necklace (i + 2) ≠ necklace i)
  (A_move : ∀ k, necklace k = 1 ∧ necklace (k + 1) = 0 ∧ necklace (k + 2) = 1)
  (B_move : ∀ k, necklace k = 0 ∧ necklace (k + 1) = 1 ∧ necklace (k + 2) = 0)
  (A_win : ∃ moves_A : fin (2 * n) → ℕ → ℕ, true) : 
  ∃ moves_B : fin (2 * n) → ℕ → ℕ, true := 
by
  sorry

end win_if_start_B_l817_817395


namespace difference_max_min_values_l817_817222

def y (x : ℝ) := |x - 1| + |x - 2| + |x - 3|

theorem difference_max_min_values :
  ∀ x : ℝ, |x| ≤ 4 → (∀ y : ℝ, ∃ max_val min_val : ℝ,
  (y = y x → y ≤ max_val ∧ y ≥ min_val) →
  max_val - min_val = 16) :=
sorry

end difference_max_min_values_l817_817222


namespace intersecting_lines_l817_817001

structure Point3D :=
  (x y z : ℝ)

structure Line3D :=
  (point direction : Point3D)

structure TriangularPrism :=
  (A B C A₁ B₁ C₁ : Point3D)

noncomputable def midpoint (P Q : Point3D) : Point3D :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2, z := (P.z + Q.z) / 2 }

def is_midpoint (P Q Mid : Point3D) : Prop :=
  midpoint P Q = Mid

noncomputable def lines_intersect_at_one_point (L1 L2 L3 : Line3D) : Prop :=
  ∃ O : Point3D, L1.point = O ∧ L2.point = O ∧ L3.point = O

theorem intersecting_lines 
  (A B C A₁ B₁ C₁ M N K : Point3D)
  (prism : TriangularPrism)
  (h_prism: prism = ⟨A, B, C, A₁, B₁, C₁⟩)
  (hM : is_midpoint B C M)
  (hN : is_midpoint A C N)
  (hK : is_midpoint A B K) :
  lines_intersect_at_one_point
    ⟨M, A₁⟩
    ⟨N, B₁⟩
    ⟨K, C₁⟩ :=
sorry

end intersecting_lines_l817_817001


namespace general_formula_a_n_sum_first_2n_c_n_l817_817760

noncomputable def a_n (n : ℕ) : ℝ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℝ := 3 ^ (n - 1)

def c_n (n : ℕ) : ℝ := a_n n + (-1) ^ n * b_n n

theorem general_formula_a_n (a_n : ℕ → ℝ) (d : ℝ) :
  (∀ n, a_n n = a_n 1 + (n - 1) * d) →
  (∃ q, b_n 2 = b_n 1 * q ∧ b_n 3 = b_n 2 * q) →
  a_n 1 = b_n 1 →
  a_n 14 = b_n 4 →
  a_n n = 2 * n - 1 := by
  sorry

theorem sum_first_2n_c_n (n : ℕ) :
  ∑ i in finset.range (2 * n), c_n (i + 1) = 4 * n^2 + (9^n / 4) - (1 / 4) := by
  sorry

end general_formula_a_n_sum_first_2n_c_n_l817_817760


namespace length_of_IZ_l817_817467

open Real

structure Triangle (A B C : Type) :=
  (XY XZ YZ : ℝ)
  (XY_pos : 0 < XY)
  (XZ_pos : 0 < XZ)
  (YZ_pos : 0 < YZ)

def parallel (a b : Type) : Prop := sorry -- Placeholder for parallel definition

def bisects (A B C D : Type) : Prop := sorry -- Placeholder for bisects definition

variable (X Y Z G H I : Type)
variable [Triangle X Y Z]

theorem length_of_IZ
  (hXY : XY = 10)
  (hGH_parallel : parallel GH XY)
  (hGH_len : GH = 6)
  (hXH_bisects_GIZ : bisects X H G IZ) :
  IZ = 7.5 :=
sorry

end length_of_IZ_l817_817467


namespace final_price_of_pencil_l817_817566

theorem final_price_of_pencil (original_cost : ℝ) (discount : ℝ) (final_price : ℝ) : 
  original_cost = 4 → discount = 0.63 → final_price = original_cost - discount → final_price = 3.37 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end final_price_of_pencil_l817_817566


namespace solution_exists_l817_817552

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (log 3 ((4 * x + 12)/(6 * x - 4)) + log 3 ((6 * x - 4)/(2 * x - 3)) = 2) ∧
  (x = 39/14)

theorem solution_exists : problem_statement :=
by
  sorry

end solution_exists_l817_817552


namespace correct_statements_implies_l817_817400

-- Let us define Triangle properties and conditions.
variables {A B C : ℝ}  -- Angles in radians
variables {a b c r : ℝ} -- Side lengths and circumradius

-- Sum of angles is π (triangle property)
axiom angle_sum : A + B + C = Real.pi

-- Not a right triangle
axiom not_right_triangle : A ≠ Real.pi / 2 ∧ B ≠ Real.pi / 2 ∧ C ≠ Real.pi / 2

-- Sine rule for sides and angles
axiom sine_rule_a : a = 2 * r * Real.sin A
axiom sine_rule_b : b = 2 * r * Real.sin B

-- Monotonicity of cosine function in (0, π)
axiom cos_monotonic : ∃ f: ℝ → ℝ, StrictMono f ∧ ∀ x ∈ Set.Ioo 0 Real.pi, f x = Real.cos x

theorem correct_statements_implies (h1 : Real.sin A > Real.sin B)
                                   (h2 : Real.cos A < Real.cos B)
                                   (h3 : Real.cos (2 * A) < Real.cos (2 * B))
  : (a > b) ∧ (a > b) ∧ (a > b) :=
by {
  sorry,
  sorry,
  sorry
}


end correct_statements_implies_l817_817400


namespace arithmetic_sequence_problem_l817_817972

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h1 : a 2 + a 3 = 4)
  (h2 : a 4 + a 5 = 6) :
  a 9 + a 10 = 11 :=
sorry

end arithmetic_sequence_problem_l817_817972


namespace sqrt_expression_meaningful_l817_817069

theorem sqrt_expression_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) :=
by
  -- Proof will be skipped
  sorry

end sqrt_expression_meaningful_l817_817069


namespace perimeter_triangle_PQR_l817_817463

-- Given conditions as definitions
def is_isosceles (A B C : Type) [MetricSpace A] (a b : A) (c : A) :=
  dist a b = dist b c → 
    dist b c = dist c b

variables {P Q R : Point}  -- Point defined in Geometry
variables (h_isosceles : is_isosceles P Q R P R)
variables (h_angles: ∠PQR = ∠PRQ)
variables (h_PQ : dist P Q = 9)
variables (h_PR : dist P R = 12)

-- Theorem statement
theorem perimeter_triangle_PQR :
  dist P Q + dist Q R + dist R P = 30 :=
sorry

end perimeter_triangle_PQR_l817_817463


namespace sum_of_digits_m_l817_817996

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def S := {n : ℕ | sum_of_digits n = 15 ∧ 0 ≤ n ∧ n < 10^8 ∧ (∀ d ∈ n.digits 10, d ≥ 1)}

def number_of_elements (s : set ℕ) : ℕ :=
  s.to_finset.card

def m := number_of_elements S

theorem sum_of_digits_m : sum_of_digits m = 12 := 
  sorry

end sum_of_digits_m_l817_817996


namespace part1_part2_l817_817034

noncomputable theory

variables {x a b c k : ℝ}

def f (x : ℝ) (k : ℝ) := k - |x - 4|
def f_transformed (x : ℝ) (k : ℝ) := k - |x|

theorem part1 (k : ℝ) (h : (∀ x : ℝ, f_transformed x k ≥ 0 → -1 ≤ x ∧ x ≤ 1)) :
  k = 1 :=
sorry

theorem part2 (a b c : ℝ) (k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : 1 = (1/a + 1/(2 * b) + 1/(3 * c))) :
  (1/9) * a + (2/9) * b + (3/9) * c ≥ 1 :=
sorry

end part1_part2_l817_817034


namespace find_unknown_number_l817_817943

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end find_unknown_number_l817_817943


namespace max_bishops_on_chessboard_l817_817634

-- Define the chessboard as an 8×8 grid
def chessboard : Type := fin 8 × fin 8

-- Define a function to get all diagonals from a given square
def diagonals (pos : chessboard) : list (fin 8 × fin 8) := 
-- Implementation of diagonals function goes here
sorry

-- Define the condition that at most 3 bishops can be in any diagonal
def max_bishops_on_diagonal (diagonal : list (fin 8 × fin 8)) : Prop :=
  diagonal.length ≤ 3

-- Define a predicate stating a bishop's position on the chessboard
def bishop_placement (b : chessboard) (bishops : list chessboard) : Prop :=
  b ∈ bishops

-- Define total bishops condition
def total_bishops (bishops : list chessboard) : Prop :=
  bishops.length = 38

-- The theorem to prove the maximum number of bishops is 38 for an 8×8 chessboard,
-- Given the condition that at most 3 bishops lie on any diagonal.
theorem max_bishops_on_chessboard :
  ∃ (bishops : list chessboard), 
    (∀ diagonal, max_bishops_on_diagonal (diagonals (⊋ bishops) diagonal)) →
    total_bishops bishops :=
sorry

end max_bishops_on_chessboard_l817_817634


namespace poly_sum_of_squares_iff_nonneg_l817_817112

open Polynomial

variable {R : Type*} [Ring R] [OrderedRing R]

theorem poly_sum_of_squares_iff_nonneg (A : Polynomial ℝ) :
  (∃ P Q : Polynomial ℝ, A = P^2 + Q^2) ↔ ∀ x : ℝ, 0 ≤ A.eval x := sorry

end poly_sum_of_squares_iff_nonneg_l817_817112


namespace trapezoid_total_area_l817_817966

variables (P Q R S T : Type) [Plane P] [IsTrapezoid P Q R S]
variables {area : Triangle → ℝ}

def trianglePQT : Triangle := ⟨P, Q, T⟩
def trianglePRT : Triangle := ⟨P, R, T⟩

axiom area_PQT : area trianglePQT = 40
axiom area_PRT : area trianglePRT = 30

theorem trapezoid_total_area : area (trapezoid PQRS) = 122.5 := by
  sorry

end trapezoid_total_area_l817_817966


namespace count_two_digit_integers_remainder_3_div_7_l817_817901

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817901


namespace solve_for_a_l817_817769

noncomputable def z (a : ℝ) : ℂ := a * (1 + complex.i) - 2

def purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem solve_for_a (a : ℝ) (h : purely_imaginary (z a)) : a = 2 :=
by
  sorry

end solve_for_a_l817_817769


namespace f_expression_f_odd_l817_817418

noncomputable def f (x : ℝ) (a b : ℝ) := (2^x + b) / (2^x + a)

theorem f_expression :
  ∃ a b, f 1 a b = 1 / 3 ∧ f 0 a b = 0 ∧ (∀ x, f x a b = (2^x - 1) / (2^x + 1)) :=
by
  sorry

theorem f_odd :
  ∀ x, f x 1 (-1) = (2^x - 1) / (2^x + 1) ∧ f (-x) 1 (-1) = -f x 1 (-1) :=
by
  sorry

end f_expression_f_odd_l817_817418


namespace minimum_value_integral_l817_817722

theorem minimum_value_integral (a : ℝ) (h : 0 ≤ a ∧ a ≤ 2) :
  ∃ b : ℝ, (b = ∫x in 0..2, abs (1 / (1 + exp x) - 1 / (1 + exp a)) dx) ∧
           (b = log ((2 + 2 * exp 2) / (1 + 2 * exp 1 + (exp 1)^2))) :=
by sorry

end minimum_value_integral_l817_817722


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817325

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817325


namespace sum_of_integers_in_image_of_f_l817_817720

open Real

noncomputable def f (x : ℝ) : ℝ := log 3 (40 * cos (2 * x) + 41)

theorem sum_of_integers_in_image_of_f :
  ∑ i in {2, 3, 4}, i = 9 := by
  -- Provided condition for x interval
  let x_interval_start := (5/3) * arctan (1/5) * cos (π - arcsin (-0.8))
  let x_interval_end := arctan 3
  have h1 : x_interval_start ≤ x_interval_end := sorry

  -- Range of x is in [x_interval_start, x_interval_end]
  have hx : ∀ x, x_interval_start ≤ x ∧ x ≤ x_interval_end → f (x) ∈ [log 3 9, log 3 81] := sorry

  -- Possible integer values of f(x) are {2, 3, 4}
  have hf_int : ∀ y ∈ (set.image f (Icc x_interval_start x_interval_end)), y ∈ {2, 3, 4} := sorry

  -- Sum of these integer values is 9
  exact finset.sum_eq 9

end sum_of_integers_in_image_of_f_l817_817720


namespace trigonometric_equation_solution_l817_817229

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) :
  5.14 * (Real.sin (3 * x)) + Real.sin (5 * x) = 2 * (Real.cos (2 * x)) ^ 2 - 2 * (Real.sin (3 * x)) ^ 2 →
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) ∨ (∃ k : ℤ, x = (π / 18) * (4 * k + 1)) :=
  by
  intro h
  sorry

end trigonometric_equation_solution_l817_817229


namespace sqrt_expression_meaningful_l817_817063

theorem sqrt_expression_meaningful {x : ℝ} : (2 * x - 4) ≥ 0 → x ≥ 2 :=
by
  intro h
  sorry

end sqrt_expression_meaningful_l817_817063


namespace price_ratio_l817_817339

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l817_817339


namespace jessica_deposited_fraction_l817_817986

-- Definitions based on conditions
def original_balance (B : ℝ) : Prop :=
  B * (3 / 5) = B - 200

def final_balance (B : ℝ) (F : ℝ) : Prop :=
  ((3 / 5) * B) + (F * ((3 / 5) * B)) = 360

-- Theorem statement proving that the fraction deposited is 1/5
theorem jessica_deposited_fraction (B : ℝ) (F : ℝ) (h1 : original_balance B) (h2 : final_balance B F) : F = 1 / 5 :=
  sorry

end jessica_deposited_fraction_l817_817986


namespace minimal_perimeter_inscribed_triangle_l817_817455

theorem minimal_perimeter_inscribed_triangle (A B C P Q R : Point)
  (h_acute : acute_angle_triangle A B C)
  (h_perp_PA : Perpendicular P A (line B C))
  (h_perp_QB : Perpendicular Q B (line C A))
  (h_perp_RC : Perpendicular R C (line A B)) :
  ∀ (P' Q' R' : Point),
  (InscribedTriangle P' Q' R' A B C) →
  Perimeter (triangle P Q R) ≤ Perimeter (triangle P' Q' R') :=
sorry

end minimal_perimeter_inscribed_triangle_l817_817455


namespace rectangle_cos_angle_AOB_l817_817163

-- Geometry setup and condition definitions
variables {A B C D O : Point}
variables {AC BD : ℝ}
variables {cos_angle_AOB : ℝ}

-- Conditions
def is_rectangle (A B C D : Point) : Prop := 
  (dist A C = 15) ∧ (dist B D = 20) ∧
  ((midpoint A C) = O) ∧ ((midpoint B D) = O)

-- Statement of the problem
theorem rectangle_cos_angle_AOB (h : is_rectangle A B C D) : cos_angle_AOB = 1 :=
sorry

end rectangle_cos_angle_AOB_l817_817163


namespace price_ratio_l817_817345

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l817_817345


namespace value_of_100d_l817_817510

noncomputable def sequence_b : ℕ → ℝ
| 0       := 3 / 5
| (n + 1) := 2 * (sequence_b n)^2 + 1

def satisfies_condition (d : ℝ) : Prop :=
  ∀ (n : ℕ), |∏ i in Finset.range n, sequence_b i| ≤ d / (3^n)

theorem value_of_100d : ∃ d, satisfies_condition d ∧ 100 * d = 125 := 
by
  sorry

end value_of_100d_l817_817510


namespace min_homework_assignments_l817_817263

variable (p1 p2 p3 : Nat)

-- Define the points and assignments
def points_first_10 : Nat := 10
def assignments_first_10 : Nat := 10 * 1

def points_second_10 : Nat := 10
def assignments_second_10 : Nat := 10 * 2

def points_third_10 : Nat := 10
def assignments_third_10 : Nat := 10 * 3

def total_points : Nat := points_first_10 + points_second_10 + points_third_10
def total_assignments : Nat := assignments_first_10 + assignments_second_10 + assignments_third_10

theorem min_homework_assignments (hp1 : points_first_10 = 10) (ha1 : assignments_first_10 = 10) 
  (hp2 : points_second_10 = 10) (ha2 : assignments_second_10 = 20)
  (hp3 : points_third_10 = 10) (ha3 : assignments_third_10 = 30)
  (tp : total_points = 30) : 
  total_assignments = 60 := 
by sorry

end min_homework_assignments_l817_817263


namespace largest_interior_angle_l817_817196

theorem largest_interior_angle (PQR : Triangle) (h1 : is_obtuse PQR) (h2 : is_isosceles PQR) (h3 : PQR.angle P = 30) : 
  ∃ A, A = 120 ∧ PQR.largest_angle = A :=
by
  sorry

end largest_interior_angle_l817_817196


namespace interest_dollar_part_even_l817_817674

-- Define the conditions
def annual_interest_rate : ℝ := 0.08
def term_in_years : ℝ := 3 / 12 -- Three months
def total_amount : ℝ := 502.40

-- Prove that the dollar part of the interest is even
theorem interest_dollar_part_even :
  let P := total_amount / (1 + annual_interest_rate * term_in_years),
      interest := total_amount - P,
      dollar_part := (interest : ℕ)
  in even dollar_part :=
by
  let P := total_amount / (1 + annual_interest_rate * term_in_years)
  let interest := total_amount - P
  let dollar_part : ℕ := interest.to_int.nat_abs
  have h : dollar_part = 10 := by sorry
  rw [h]
  exact even_bit0 5

end interest_dollar_part_even_l817_817674


namespace two_hundredth_digit_of_7_over_29_l817_817609

theorem two_hundredth_digit_of_7_over_29 :
  (decimal_places ⟨7, 29⟩ 200) = 1 :=
sorry

end two_hundredth_digit_of_7_over_29_l817_817609


namespace value_of_k_h_5_l817_817931

def h (x : ℝ) : ℝ := 4 * x + 6
def k (x : ℝ) : ℝ := 6 * x - 8

theorem value_of_k_h_5 : k (h 5) = 148 :=
by
  have h5 : h 5 = 4 * 5 + 6 := rfl
  simp [h5, h, k]
  sorry

end value_of_k_h_5_l817_817931


namespace length_AB_l817_817744

theorem length_AB (x1 y1 x2 y2 : ℝ) (k : ℝ) 
  (h1 : y1 = k * x1 - k) (h2 : y2 = k * x2 - k) 
  (h3 : y1^2 = 4 * x1) (h4 : y2^2 = 4 * x2) 
  (h5 : (x1 + x2) / 2 = 3) : 
  |AB| = 8 :=
by
  sorry

end length_AB_l817_817744


namespace problem_1_problem_2_l817_817747

noncomputable theory
open_locale classical

def sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a (n + 1) = a n / (a n + 3)

def geometric_sequence (s : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, n > 0 → s (n + 1) = r * s n

theorem problem_1 (a : ℕ → ℝ) (h_seq : sequence a) (h_a1 : a 1 = 1) :
  geometric_sequence (λ n, 1 / a n + 1 / 2) 3 :=
sorry

def sequence_b (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → b n = n / (2 ^ (n - 1))

def sum_T (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → T n = finset.sum (finset.range n) b

theorem problem_2 (T : ℕ → ℝ) (b : ℕ → ℝ) (h_b : sequence_b b) (h_T : sum_T T b) :
  ∀ n : ℕ, n > 0 → (-1) ^ n * λ < T n + n / (2 ^ (n - 1)) :=
begin
  sorry
end

end problem_1_problem_2_l817_817747


namespace two_digit_integers_leaving_remainder_3_div_7_count_l817_817915

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l817_817915


namespace distance_on_foot_l817_817133

/-- Definition of variables and constants --/
variables (x y : ℕ)

/-- Conditions from the problem statement --/
def total_distance : Prop := x + y = 61
def total_time : Prop := (x / 4) + (y / 9) = 9

/-- The proof statement --/
theorem distance_on_foot (h1 : total_distance x y) (h2 : total_time x y) : x = 16 :=
  sorry

end distance_on_foot_l817_817133


namespace num_copper_pipes_needed_l817_817579

open Real

-- Definitions for the diameters of the pipes
def diameter_copper := 2
def diameter_steel := 8

-- Definitions for the radius by halving the diameter
def radius_copper := diameter_copper / 2
def radius_steel := diameter_steel / 2

-- Definitions for the cross-sectional area using πr²
def area_copper := π * radius_copper^2
def area_steel := π * radius_steel^2

-- Theorem statement for the equivalence of the carrying capacity
theorem num_copper_pipes_needed :
  (area_steel / area_copper) = 16 :=
by
  -- Proof omitted
  sorry

end num_copper_pipes_needed_l817_817579


namespace sqrt_expression_meaningful_l817_817067

theorem sqrt_expression_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) :=
by
  -- Proof will be skipped
  sorry

end sqrt_expression_meaningful_l817_817067


namespace rectangle_has_four_right_angles_l817_817431

-- Define the concept of a rectangle and right angle
def is_rectangle (r : Type) : Prop :=
  ∃ a b c d : r, 
    (∀ (x y : r), x ≠ y → x = a ∨ x = b ∨ x = c ∨ x = d ∧ 
                  y = a ∨ y = b ∨ y = c ∨ y = d) ∧ 
    (∀ (x y z : r), x ≠ y → x ≠ z → y ≠ z → 
                  (angle x y = 90 ∧ angle y z = 90 ∧ angle z x = 90))

-- The theorem to be proven
theorem rectangle_has_four_right_angles {r : Type} (rect : r) (h : is_rectangle rect) : 
  ∃ (a b c d : r), 
    (∀ (x : r), x = a ∨ x = b ∨ x = c ∨ x = d) ∧ 
    (angle a b = 90 ∧ angle b c = 90 ∧ angle c d = 90 ∧ angle d a = 90) :=
sorry

end rectangle_has_four_right_angles_l817_817431


namespace vanya_scores_not_100_l817_817202

-- Definitions for initial conditions
def score_r (M : ℕ) := M - 14
def score_p (M : ℕ) := M - 9
def score_m (M : ℕ) := M

-- Define the maximum score constraint
def max_score := 100

-- Main statement to be proved
theorem vanya_scores_not_100 (M : ℕ) 
  (hr : score_r M ≤ max_score) 
  (hp : score_p M ≤ max_score) 
  (hm : score_m M ≤ max_score) : 
  ¬(score_r M = max_score ∧ (score_p M = max_score ∨ score_m M = max_score)) ∧
  ¬(score_r M = max_score ∧ score_p M = max_score ∧ score_m M = max_score) :=
sorry

end vanya_scores_not_100_l817_817202


namespace quadratic_one_real_root_l817_817446

theorem quadratic_one_real_root (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - 2 * x + 1 = 0) ↔ ((a = 0) ∨ (a = 1))) :=
sorry

end quadratic_one_real_root_l817_817446


namespace circumradius_triangle_half_inradius_l817_817426

noncomputable def circumradius (A B C : Type) [metric_space B] [metric_space C] [metric_space A] (Δ : triangle A B C) : Type := sorry
noncomputable def inradius (I : Type) [metric_space I] : Type := sorry
noncomputable def incenter (Δ : Type) : Type := sorry
noncomputable def ortho_intersect (O₁ O₂ O₃ I Δ : Type) : Prop := sorry
noncomputable def triangle (A B C : Type) [metric_space B] [metric_space C] [metric_space A] : Type := sorry
noncomputable def point (A B C : Type) [metric_space B] [metric_space C] [metric_space A] (O₁ O₂ : Type) : Type := sorry

theorem circumradius_triangle_half_inradius {ABC I O₁ O₂ O₃ : Type}
[metric_space ABC] [metric_space I] [metric_space O₁] [metric_space O₂] [metric_space O₃]
(triangle_ABC : triangle ABC I O₁)
(incenter_I : incenter triangle_ABC = I)
(ortho_intersect : ortho_intersect O₁ O₂ O₃ I triangle_ABC) :
  ∀ A' B' C': Type, (point A' B' C' O₁ O₂ O₃) →
  circumradius A' B' C' = 0.5 * inradius I :=
sorry

end circumradius_triangle_half_inradius_l817_817426


namespace inequality_proof_l817_817540

theorem inequality_proof
  (a b c d : ℝ)
  (ha : abs a > 1)
  (hb : abs b > 1)
  (hc : abs c > 1)
  (hd : abs d > 1)
  (h : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 :=
sorry

end inequality_proof_l817_817540


namespace table_sum_less_than_nine_l817_817968

-- Let A be a 9x9 matrix of real numbers.
def A : Matrix (Fin 9) (Fin 9) ℝ := sorry

-- Condition: Each cell in the 9x9 table has a number less than 1 in absolute value.
def cell_condition (i j : Fin 9) := abs (A i j) < 1

-- Condition: The sum of numbers in each 2x2 sub-grid is zero.
def subgrid_condition (i j : Fin 8) :=
  (A i j) + (A i (j + 1)) + (A (i + 1) j) + (A (i + 1) (j + 1)) = 0

theorem table_sum_less_than_nine :
  (∀ i j, cell_condition i j) →
  (∀ i j, subgrid_condition i j) →
  ∑ (i : Fin 9) (j : Fin 9), A i j < 9 := 
sorry

end table_sum_less_than_nine_l817_817968


namespace percent_decrease_in_hours_l817_817264

theorem percent_decrease_in_hours (W H : ℝ) 
  (h1 : W > 0) 
  (h2 : H > 0)
  (new_wage : ℝ := W * 1.25)
  (H_new : ℝ := H / 1.25)
  (total_income_same : W * H = new_wage * H_new) :
  ((H - H_new) / H) * 100 = 20 := 
by
  sorry

end percent_decrease_in_hours_l817_817264


namespace f_is_odd_l817_817775

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 3 then 2^x - 1 else x - 5

theorem f_is_odd (x : ℝ) : f (-x) = -f x := sorry

example : f (f 3) = -3 := by
  have f3 : f 3 = 3 - 5 := by
    have h : 3 ≥ 3 := by linarith
    simp [f, h]
    ring
  rw [f3]
  have h1 : f (3 - 5) = -3 := by
    have ff3 : f (-2) = -f 2 := by
      exact f_is_odd 2
    have f2 : f 2 = 2^2 - 1 := by
      have h2 : 0 ≤ 2 ∧ 2 < 3 := by linarith
      simp [f, h2]
      ring
    rw [f2] at ff3
    simp [ff3]
  simp [h1]

end f_is_odd_l817_817775


namespace cage_cost_correct_l817_817097

def cost_of_cat_toy : Real := 10.22
def total_cost_of_purchases : Real := 21.95
def cost_of_cage : Real := total_cost_of_purchases - cost_of_cat_toy

theorem cage_cost_correct : cost_of_cage = 11.73 := by
  sorry

end cage_cost_correct_l817_817097


namespace count_two_digit_remainders_l817_817797

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817797


namespace smallest_coprime_prime_l817_817537

theorem smallest_coprime_prime (n : ℕ) (h : n ≥ 2) :
  ∃ a : ℕ, (∀ b : ℕ, (2 ≤ b ∧ b ≤ n → nat.gcd a b = 1)) ∧ prime a := 
sorry

end smallest_coprime_prime_l817_817537


namespace range_f_l817_817027

/-- Define the function f(x) --/
def f (x : ℝ) : ℝ := real.sqrt ((1 - x) * (x - 5))

/-- Define the domain of f(x) --/
def domain : set ℝ := {x | 1 ≤ x ∧ x ≤ 5}

/-- Define the range of f(x) --/
def range_of_f : set ℝ := {y | ∃ x ∈ domain, f x = y}

/-- Prove that the range of the function f(x) is [0, 2] --/
theorem range_f : range_of_f = {y | 0 ≤ y ∧ y ≤ 2} :=
by
  sorry

end range_f_l817_817027


namespace triangle_AC_length_and_area_l817_817957

theorem triangle_AC_length_and_area
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (hA : ∠ A = 90) -- Angle A is 90 degrees
  (hTanB : tan (∠ B) = 5 / 12) -- tan B = 5/12
  (hAB: AB = 40) -- Hypotenuse AB is 40
  : AC = 480 / 13 ∧ (1 / 2) * BC * AC = 48000 / 169 :=
by
  sorry

end triangle_AC_length_and_area_l817_817957


namespace even_function_for_negative_x_l817_817390

noncomputable def f (x : ℝ) : ℝ := if x > 0 then (1 - x) * x else -(x^2) - x

theorem even_function_for_negative_x (x : ℝ) (h : x < 0) :
  f x = -(x^2) - x :=
begin
  unfold f,
  split_ifs,
  norm_num,
  sorry
end

end even_function_for_negative_x_l817_817390


namespace tulip_percentage_l817_817258

theorem tulip_percentage (total_flowers pink_flowers red_flowers pink_roses pink_tulips red_roses red_tulips : ℕ) 
  (h_pink_fraction : pink_flowers = total_flowers * 3 / 10)
  (h_red_fraction : red_flowers = total_flowers * 7 / 10)
  (h_pink_roeses_fraction : pink_roses = pink_flowers / 4)
  (h_red_tulips_fraction : red_tulips = red_flowers / 3)
  (h_tulips_are_pink_or_red : pink_tulips + red_tulips = total_flowers * 11 / 24 * 24) :
  pink_tulips * 10 / 3 * 100 / total_flowers + red_tulips * 10 / 7 * 100 / total_flowers = 46 :=
begin
  sorry
end

end tulip_percentage_l817_817258


namespace find_f_at_1_l817_817419

variable (ω : ℝ) (ω_pos : ω > 0)
variable (f : ℝ → ℝ)
variable (a b : ℝ) -- a and b are points on the x axis corresponding to maximum and minimum of f

noncomputable def distance_AB : ℝ := abs (a - b)

noncomputable def distance_given : Prop := distance_AB ω_pos = 2 * sqrt 2

noncomputable def sine_function : ℝ → ℝ := λ x, real.sin (ω * x + real.pi / 3)

theorem find_f_at_1 : distance_given → sine_function ω 1 = sqrt 3 / 2 := 
by
  sorry

end find_f_at_1_l817_817419


namespace two_digit_integers_remainder_3_count_l817_817879

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817879


namespace incorrect_expression_is_3_l817_817617

noncomputable def cond1 : ∀ x y : ℝ, 3 ^ x > 3 ^ y ↔ x > y := sorry

noncomputable def cond2 : ∀ x y : ℝ, log (0.5 : ℝ) x > log (0.5 : ℝ) y ↔ x < y := sorry

noncomputable def cond3 : ∀ x y : ℝ, 0.75 ^ x > 0.75 ^ y ↔ x < y := sorry

noncomputable def cond4 : ∀ x y : ℝ, log 10 x > log 10 y ↔ x > y := sorry

theorem incorrect_expression_is_3 : 
  (∀ x y : ℝ, cond1 x y ∧ cond2 x y ∧ cond3 x y ∧ cond4 x y) → 
  0.75 ^ (-0.1) < 0.75 ^ 0.1 := 
sorry

end incorrect_expression_is_3_l817_817617


namespace prove_min_value_ST_l817_817969

-- Definitions from conditions
def point_on_line (P : ℝ × ℝ) : Prop := P.1 = -1/2

def midpoint (P Q F : ℝ × ℝ) : Prop := Q.1 = (P.1 + F.1) / 2 ∧ Q.2 = (P.2 + F.2) / 2

def orthogonal (M Q P F : ℝ × ℝ) : Prop := (Q.2 - M.2) / (Q.1 - M.1) * (F.2 - P.2) / (F.1 - P.1) = -1

def scalar_multiple (M P F : ℝ × ℝ) (λ : ℝ) : Prop := 
  M.1 - P.1 = λ * (F.1 - (0 : ℝ)) ∧ M.2 = λ * F.2

-- Given circle
def circle (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 2

-- Minimum value problem
def min_value_ST (P : ℝ × ℝ) (Q : ℝ × ℝ) (F : ℝ × ℝ) (M : ℝ × ℝ) (λ : ℝ) : ℝ :=
  2 * (sqrt (3 * (2 / (5 : ℝ))))
  
theorem prove_min_value_ST (P Q F M : ℝ × ℝ) (λ : ℝ) (S T : ℝ × ℝ) : 
  point_on_line P ∧
  F = (1/2, 0) ∧
  midpoint P Q F ∧
  orthogonal M Q P F ∧
  scalar_multiple M P F λ ∧
  circle S.1 S.2 ∧
  circle T.1 T.2 → 
  abs (dist S T) = min_value_ST P Q F M λ :=
by
  sorry

end prove_min_value_ST_l817_817969


namespace solution_set_f_gt_0_l817_817928

variable {α : Type*} [LinearOrderedField α] {f : α → α}

-- Conditions
variable (h_even : ∀ x, f x = f (-x))
variable (h_increasing : ∀ x y, x < y → y < 0 → f x < f y)
variable (h_f1_zero : f 1 = 0)

-- Proof statement
theorem solution_set_f_gt_0 :
  {x | f x > 0} = set.Ioo (-1 : α) 0 ∪ set.Ioo 0 1 :=
sorry

end solution_set_f_gt_0_l817_817928


namespace ratio_of_rectangle_to_square_l817_817541

theorem ratio_of_rectangle_to_square (s w h : ℝ) 
  (hs : h = s / 2)
  (shared_area_ABCD_EFGH_1 : 0.25 * s^2 = 0.4 * w * h)
  (shared_area_ABCD_EFGH_2 : 0.25 * s^2 = 0.4 * w * h) :
  w / h = 2.5 :=
by
  -- Proof goes here
  sorry

end ratio_of_rectangle_to_square_l817_817541


namespace area_ABDN_is_25_over_3_l817_817354

noncomputable theory

def is_right_angle (angle : ℝ) : Prop :=
  angle = π / 2

structure Decagon where
  sides : Fin 10 → ℝ
  right_angles : ∀ i : Fin 10, is_right_angle (π / 2)

def meet_at_n (A E D G N : ℝ × ℝ) : Prop :=
  let (Ax, Ay) := A
  let (Ex, Ey) := E
  let (Dx, Dy) := D
  let (Gx, Gy) := G
  ∃ N : ℝ × ℝ, (Ax*N.1 + Ay*N.2 = Ex ∧ Dx*N.1 + Dy*N.2 = Gx)

def polygon_length_5 (fig : Decagon) : Prop :=
  ∀ i, fig.sides i = 5

def area_of_quadrilateral (A B D N : ℝ × ℝ) : ℝ :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  let (Dx, Dy) := D
  let (Nx, Ny) := N
  (1/2) * abs (Ax * By + Bx * Dy + Dx * Ny + Nx * Ay -
               (Ay * Bx + By * Dx + Dy * Nx + Ny * Ax))

theorem area_ABDN_is_25_over_3
  (polygon : Decagon)
  (h1 : polygon_length_5 polygon)
  (A B D E G N : ℝ × ℝ)
  (h2 : meet_at_n A E D G N) :
  area_of_quadrilateral A B D N = 25/3 :=
sorry

end area_ABDN_is_25_over_3_l817_817354


namespace identify_roles_tricksters_prevent_identification_l817_817246

/- Part (a) -/
theorem identify_roles (P1 P2 P3 : Type) (L T S : Prop)
  (h1 : L ≠ T) (h2 : L ≠ S) (h3 : T ≠ S)
  (liar : ∀ x, L x → ¬ T x)
  (truth_teller : ∀ x, T x → ¬ L x)
  (trickster : ∀ x, S x → (L x ∨ T x))
  (P1_is_truth_teller : P1 = T → ¬ P1 = L ∧ ¬ P1 = S)
  (P2_is_trickster : P2 = S → ¬ P2 = L ∧ (P2 = T ∨ P2 = S))
  (P3_is_liar : P3 = L → ¬ P3 = T ∧ ¬ P3 = S) :
  ∃ P_true P_liar P_trickster : Type, 
    (P_true = Truth_teller ∧ P_liar = Liar ∧ P_trickster = Trickster) := by
  sorry

/- Part (b) -/
theorem tricksters_prevent_identification (P1 P2 P3 P4 : Type) (L T S1 S2 : Prop)
  (h1 : L ≠ T) (h2 : L ≠ S1) (h3 : L ≠ S2) 
  (truth_teller : ∀ x, T x → ¬ L x)
  (liar : ∀ x, L x → ¬ T x)
  (trickster1 : ∀ x, S1 x → (L x ∨ T x))
  (trickster2 : ∀ x, S2 x → (L x ∨ T x))
  (collaborate : ∀ x y, (S1 x ∧ (T y ∨ L y)) → (S2 x ∧ (T y ∨ L y))) : 
  ¬ ∃ P_true P_liar P_trickster1 P_trickster2 : Type, 
    (P_true = Truth_teller ∧ P_liar = Liar ∧ P_trickster1 = Trickster ∧ P_trickster2 = Trickster) := by
  sorry

end identify_roles_tricksters_prevent_identification_l817_817246


namespace positive_two_digit_integers_remainder_3_l817_817818

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817818


namespace cone_height_from_sector_l817_817254

theorem cone_height_from_sector
  (radius : ℝ) 
  (sector_arc_length : ℝ) 
  (cone_slant_height : ℝ) 
  (cone_base_radius : ℝ) :
  radius = 10 →
  sector_arc_length = 5 * Real.pi →
  cone_base_radius = 5 / 2 →
  cone_slant_height = radius →
  ∃ h : ℝ, h = (5 * Real.sqrt 15) / 2 :=
by
  intros h_radius h_sector h_base h_slant
  have h_radius_eq : radius = 10 := h_radius
  have h_sector_eq : sector_arc_length = 5 * Real.pi := h_sector
  have h_base_eq : cone_base_radius = 5 / 2 := h_base
  have h_slant_eq : cone_slant_height = radius := h_slant
  use (5 * Real.sqrt 15) / 2
  sorry

end cone_height_from_sector_l817_817254


namespace product_even_probability_l817_817386

theorem product_even_probability 
  (dice1 dice2 : ℕ) (h1 : dice1 ∈ set.univ ∩ {1..10}) (h2 : dice2 ∈ set.univ ∩ {1..10}) :
  (75 : ℚ) / 100 = 3 / 4 :=
sorry

end product_even_probability_l817_817386


namespace yokohama_entrance_exam_solution_l817_817105

noncomputable def volume_of_solid (a : ℝ) (f g : ℝ → ℝ) :=
  ∫ x in 0..1, π * ((g x) ^ 2)dx +
  ∫ x in 1..exp (1 / 3), π * (((g x) ^ 2) - ((f x) ^ 2))dx

theorem yokohama_entrance_exam_solution :
  ∀ a : ℝ, (a = 1 / (3 * exp(1))) →
  (∀ x, (f x) = ln x / x) →
  (∀ x, (g x) = a * x^2) →
  volume_of_solid a f g = π * (1 + 100 * exp (1 / 3) - 72 * exp (2 / 3)) / (36 * exp (2 / 3)) :=
by
  intros a ha hf hg
  rw [hf, hg]
  sorry

end yokohama_entrance_exam_solution_l817_817105


namespace segments_do_not_intersect_l817_817113

open Real

-- Define the points and their relationships
variables {A : ℕ → EuclideanGeometry.Point ℝ}

-- Given conditions: lengths and angles
variables (h_len : ∀ i : ℕ, i < n → dist (A i) (A (i + 1)) ≤ 1/(2*i+1) * dist (A (i + 1)) (A (i + 2)))
variables (h_angle : ∀ i : ℕ, i < n - 2 → 0 < EuclideanGeometry.angle (A i) (A (i + 1)) (A (i + 2)) ∧
              EuclideanGeometry.angle (A i) (A (i + 1)) (A (i + 2)) < EuclideanGeometry.angle (A (i + 1)) (A (i + 2)) (A (i + 3)) ∧
              EuclideanGeometry.angle (A (i + 2)) (A (i + 3)) (A (i + 4)) < 180)

-- To prove: the segments do not intersect
theorem segments_do_not_intersect (h_len : ∀ i : ℕ, i < n → dist (A i) (A (i + 1)) ≤ 1/(2*i+1) * dist (A (i + 1)) (A (i + 2)))
    (h_angle : ∀ i : ℕ, i < n - 2 → 0 < EuclideanGeometry.angle (A i) (A (i + 1)) (A (i + 2)) ∧
                        EuclideanGeometry.angle (A i) (A (i + 1)) (A (i + 2)) < EuclideanGeometry.angle (A (i + 1)) (A (i + 2)) (A (i + 3)) ∧
                        EuclideanGeometry.angle (A (i + 2)) (A (i + 3)) (A (i + 4)) < 180) :
    ∀ (k m : ℕ), 0 ≤ k → k ≤ m - 2 → m - 2 < n - 2 →
    ¬ EuclideanGeometry.segments_intersect (A k) (A (k + 1)) (A m) (A (m + 1)) :=
sorry

end segments_do_not_intersect_l817_817113


namespace flashlight_price_percentage_l817_817351

theorem flashlight_price_percentage 
  (hoodie_price boots_price total_spent flashlight_price : ℝ)
  (discount_rate : ℝ)
  (h1 : hoodie_price = 80)
  (h2 : boots_price = 110)
  (h3 : discount_rate = 0.10)
  (h4 : total_spent = 195) 
  (h5 : total_spent = hoodie_price + ((1 - discount_rate) * boots_price) + flashlight_price) : 
  (flashlight_price / hoodie_price) * 100 = 20 :=
by
  sorry

end flashlight_price_percentage_l817_817351


namespace positive_two_digit_integers_remainder_3_l817_817808

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l817_817808


namespace count_two_digit_integers_remainder_3_div_7_l817_817898

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817898


namespace painting_price_decrease_l817_817175

theorem painting_price_decrease (P : ℝ) (h1 : 1.10 * P - 0.935 * P = x * 1.10 * P) :
  x = 0.15 := by
  sorry

end painting_price_decrease_l817_817175


namespace find_f_1949_l817_817013

noncomputable def f : ℝ → ℝ
| x := if h1 : x = 1949 then (sqrt 3) - 2 else sorry

theorem find_f_1949 :
  (∀ x, f(x + 2) * (1 - f(x)) = 1 + f(x)) ∧
  f(1) = 2 + sqrt 3 →
  f(1949) = sqrt 3 - 2 := 
by 
  intro h;
  have h1 : ∀ x, f(x + 2) * (1 - f(x)) = 1 + f(x) := h.left;
  have h2 : f(1) = 2 + sqrt 3 := h.right;
  sorry

end find_f_1949_l817_817013


namespace triangle_area_constant_circle_equation_if_OM_EQ_ON_l817_817768

open Real

theorem triangle_area_constant (t : ℝ) (ht : t ≠ 0) :
  let C := (t, 2 / t)
      Circle (x y : ℝ) := (x - t)^2 + (y - 2 / t)^2 = t^2 + 4 / t^2
  in
  let A := (2 * t, 0)
      B := (0, 4 / t)
      O := (0, 0)
  in (1/2) * (abs (2 * t - 0)) * (abs (4 / t - 0)) = 4 :=
  sorry

theorem circle_equation_if_OM_EQ_ON :
  ∀ t : ℝ, t ≠ 0 →
  let C := (t, 2 / t)
      Circle (x y : ℝ) := (x - t)^2 + (y - 2 / t)^2 = t^2 + 4 / t^2
      line (x y : ℝ) := 2 * x + y - 4 = 0
  in
  ∃ (M N : Point), (Circle M.1 M.2) ∧ (Circle N.1 N.2) ∧ (line M.1 M.2) ∧ (line N.1 N.2) ∧ M ≠ N ∧ dist (0, 0) M = dist (0, 0) N →
  Circle (x y : ℝ) x y = (x - 2)^2 + (y - 1)^2 = 5 :=
  sorry

end triangle_area_constant_circle_equation_if_OM_EQ_ON_l817_817768


namespace locus_of_points_l817_817382

theorem locus_of_points (x y : ℝ) : 
  (sqrt ((x - 4)^2 + y^2) = 2 * sqrt ((x - 1)^2 + y^2)) → (x^2 + y^2 = 4) :=
by
  sorry

end locus_of_points_l817_817382


namespace mika_height_proof_l817_817145

theorem mika_height_proof (sheasHeightAtStart mikasHeightAtStart : ℝ) 
  (sheaGrewBy : sheasHeightAtStart * 1.25 = 75)
  (mikaGrewBy : mikasHeightAtStart = sheasHeightAtStart)
  (mikaGrowthRate : mikasHeightAtStart * 0.10) :
  mikasHeightAtStart * 1.10 = 66 := by
  sorry

end mika_height_proof_l817_817145


namespace abs_triangle_l817_817425

variables {a b c : ℕ} -- assuming positive numbers are naturals for simplicity

-- Given conditions
axiom distinct_pos_numbers : a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0
axiom triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Proving that |a|, |b|, and |c| form a triangle
theorem abs_triangle (a b c : ℕ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0)
  (h7 : a + b > c) (h8 : b + c > a) (h9 : c + a > b) :
  |a| + |b| > |c| ∧ |b| + |c| > |a| ∧ |c| + |a| > |b| :=
by {
  have abs_a : |a| = a := abs_of_nonneg (nat.zero_le a),
  have abs_b : |b| = b := abs_of_nonneg (nat.zero_le b),
  have abs_c : |c| = c := abs_of_nonneg (nat.zero_le c),
  sorry -- to be proven using given axioms
}

end abs_triangle_l817_817425


namespace complex_polynomial_equal_l817_817378

theorem complex_polynomial_equal (n : ℕ) (a : fin n → ℂ) :
  (n > 0) →
  (∀ x : ℂ, ∏ i in finset.fin_range n, (x + (a i)) =
            x^n + ∑ i in finset.range (n + 1), (nat.choose n i) * (a i)^i * x^(n-i)) →
  (∀ i j : fin n, a i = a j) :=
begin
  intros hpos heq,
  sorry
end

end complex_polynomial_equal_l817_817378


namespace exponent_identity_l817_817919

theorem exponent_identity (m n : ℝ) (h1 : 3^m = 5) (h2 : 3^n = 2) : 3^(2 * m - 3 * n) = 25 / 8 :=
by
  sorry

end exponent_identity_l817_817919


namespace calculate_teena_speed_l817_817436

noncomputable def Teena_speed (t c t_ahead_in_1_5_hours : ℝ) : ℝ :=
  let distance_initial_gap := 7.5
  let coe_speed := 40
  let time_in_hours := 1.5
  let distance_coe_travels := coe_speed * time_in_hours
  let total_distance_teena_needs := distance_coe_travels + distance_initial_gap + t_ahead_in_1_5_hours
  total_distance_teena_needs / time_in_hours

theorem calculate_teena_speed :
  (Teena_speed 7.5 40 15) = 55 :=
  by
  -- skipped proof
  sorry

end calculate_teena_speed_l817_817436


namespace count_two_digit_integers_remainder_3_l817_817872

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817872


namespace correct_number_of_propositions_l817_817279

def sum_of_coefficients_expansion : Prop :=
  ∑ i in finset.range 9, (1 - i) = 0

def negation_proposition_correctness (p : ℝ → Prop) : Prop :=
  ¬ (∃ x : ℝ, p x) = (∀ x : ℝ, ¬ p x)

def normal_distribution_probability (p : ℝ) : Prop :=
  (∃ X : ℝ → bool, X ∈ normal (0, 1) ∧
  ((∃ b, b = if P (X > 1) = p) → P (-1 < X ∧ X < 0) = 1/2 - p))

def regression_line_mean_passes (x̄ ȳ : ℝ) : Prop :=
  ∃ b0 b1 : ℝ, ∀ x y : ℝ, y = b0 + b1 * x ↔ y = x̄ + b1 * ȳ

theorem correct_number_of_propositions : 3 :=
by
  have h1 : sum_of_coefficients_expansion := sorry,
  have h2 : ¬ negation_proposition_correctness (λ x, x^2 - x - 1 > 1) := sorry,
  have h3 : normal_distribution_probability := sorry,
  have h4 : regression_line_mean_passes := sorry,
  exact 3

end correct_number_of_propositions_l817_817279


namespace find_value_of_a2_plus_b2_plus_c2_l817_817500

variables (a b c : ℝ)

-- Define the conditions
def conditions := (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a + b + c = 0) ∧ (a^3 + b^3 + c^3 = a^5 + b^5 + c^5)

-- State the theorem we need to prove
theorem find_value_of_a2_plus_b2_plus_c2 (h : conditions a b c) : a^2 + b^2 + c^2 = 6 / 5 :=
  sorry

end find_value_of_a2_plus_b2_plus_c2_l817_817500


namespace measure_angle_of_adjacent_face_diagonals_in_cube_is_90_l817_817630

-- Definition of Cube
structure Cube :=
  (vertices : Fin 8 → ℝ × ℝ × ℝ)

-- Definition of adjacent face diagonals forming an angle in a cube
def measure_angle_of_adjacent_face_diagonals (c : Cube) : ℝ := sorry

-- The theorem to prove that the measure of the angle is 90 degrees
theorem measure_angle_of_adjacent_face_diagonals_in_cube_is_90 (c : Cube) :
  measure_angle_of_adjacent_face_diagonals(c) = 90 :=
by sorry

end measure_angle_of_adjacent_face_diagonals_in_cube_is_90_l817_817630


namespace car_mass_nearest_pound_l817_817647

def mass_of_car_kg : ℝ := 1500
def kg_to_pounds : ℝ := 0.4536

theorem car_mass_nearest_pound :
  (↑(Int.floor ((mass_of_car_kg / kg_to_pounds) + 0.5))) = 3307 :=
by
  sorry

end car_mass_nearest_pound_l817_817647


namespace count_two_digit_integers_remainder_3_div_7_l817_817904

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817904


namespace count_two_digit_integers_with_remainder_3_l817_817845

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l817_817845


namespace regional_salary_distribution_possible_l817_817959

theorem regional_salary_distribution_possible
  (total_employees : ℕ)
  (total_salary : ℕ)
  (high_salary_fraction : ℕ → Prop)
  (low_salary_fraction : ℕ → Prop)
  (h1 : ∀ n, n ≤ total_employees → high_salary_fraction n → n = 10 * total_employees / 100)
  (h2 : ∀ s, s ≤ total_salary → high_salary_fraction s → s = 90 * total_salary / 100)
  (h3 : ∀ n, n ≤ total_employees → low_salary_fraction n → n = 90 * total_employees / 100)
  (h4 : ∀ s, s ≤ total_salary → low_salary_fraction s → s = 10 * total_salary / 100)
: ∃ regions : list (list ℕ) → Prop,
    ∀ r ∈ regions,
      ∀ subset ∈ (list.powerset r),
        (list.length subset = 10 * list.length r / 100) →
        (list.sum subset ≤ (11 * list.sum r / 100)) :=
sorry

end regional_salary_distribution_possible_l817_817959


namespace min_slope_min_slope_value_range_a_l817_817024

def f (a : ℝ) := λ x : ℝ, (1 / 3) * x^3 - (1 / 2) * (a + 2) * x^2 + x
def g := λ x : ℝ, Real.exp 1 - Real.exp x / x

noncomputable def f' (a : ℝ) := λ x : ℝ, x^2 - (a + 2) * x + 1
noncomputable def g' := λ x : ℝ, (Real.exp x * (1 - x)) / (x^2)

theorem min_slope (x : ℝ) : f' 0 x = x^2 - 2 * x + 1 := by
  sorry

theorem min_slope_value (k_min : ℝ) : k_min = 0 :=
  let v := 1 in
  have : f' 0 v = 0, from min_slope v
  this

theorem range_a (a : ℝ) : (∀ x > 0, f' a x ≥ g x) → a ≤ 0 := by
  sorry

end min_slope_min_slope_value_range_a_l817_817024


namespace find_constants_l817_817570

theorem find_constants :
  ∃ (A B C : ℚ), 
  (A = 1 ∧ B = 4 ∧ C = 1) ∧ 
  (∀ x, x ≠ -1 → x ≠ 3/2 → x ≠ 2 → 
    (6 * x^2 - 13 * x + 6) / (2 * x^3 + 3 * x^2 - 11 * x - 6) = 
    (A / (x + 1) + B / (2 * x - 3) + C / (x - 2))) :=
by
  sorry

end find_constants_l817_817570


namespace common_point_in_half_planes_l817_817233

theorem common_point_in_half_planes (hp : Set (Set ℝ^2)) (h : ∀ (π1 π2 π3 : Set ℝ^2), π1 ∈ hp → π2 ∈ hp → π3 ∈ hp → (∃ p, p ∈ π1 ∧ p ∈ π2 ∧ p ∈ π3)) :
  ∃ p, ∀ π ∈ hp, p ∈ π :=
sorry

end common_point_in_half_planes_l817_817233


namespace price_of_companyB_is_3_point_5_l817_817197

noncomputable def companyA_revenue : ℝ := 300 * 4  -- Revenue of Company A
noncomputable def companyB_revenue (P : ℝ) : ℝ := 350 * P  -- Revenue of Company B as a function of P
noncomputable def price_companyB := 3.5  -- Price we want to prove for Company B

theorem price_of_companyB_is_3_point_5 :
  ∃ P : ℝ, (companyB_revenue P) = companyA_revenue + 25 ∨ (companyA_revenue = companyB_revenue P + 25) ∧ P = price_companyB := 
by
  sorry

end price_of_companyB_is_3_point_5_l817_817197


namespace count_valid_n_l817_817777

noncomputable def sum_of_consecutive_integers (i k : ℕ) : ℕ :=
  k * (2 * i + k - 1) / 2

theorem count_valid_n : 
  let valid_n (n : ℕ) := ∃ i k : ℕ, k ≥ 60 ∧ n = sum_of_consecutive_integers i k
  in (count (valid_n) (λ n, n ≤ 2000) = 6) :=
sorry

end count_valid_n_l817_817777


namespace inequality_solution_sets_correct_l817_817589

theorem inequality_solution_sets_correct
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, ax^2 + bx + c < 0 ↔ x ∈ Ioo (-1 : ℝ) 2) :
  (a > 0) ∧
  (∀ x : ℝ, bx + c > 0 ↔ x ∈ Iio (-2 : ℝ)) ∧
  (4a - 2b + c > 0) ∧
  (∀ x : ℝ, cx^2 - bx + a > 0 ↔ x ∈ Ioo (-1 : ℝ) (1/2 : ℝ)) :=
sorry

end inequality_solution_sets_correct_l817_817589


namespace count_two_digit_integers_remainder_3_l817_817866

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817866


namespace range_of_x_l817_817735

def abs (x : ℝ) := if x ≥ 0 then x else -x

theorem range_of_x (x : ℝ) (h : abs (x - 1) + abs (x - 2) = 1) : 1 ≤ x ∧ x ≤ 2 := 
by
  sorry

end range_of_x_l817_817735


namespace probability_multiple_of_3_l817_817367

-- Definitions based on conditions:
def box1 := {1, 4, 5}
def box2 := {1, 4, 5}
def draw_chips (b1 b2 : Finset ℕ) : Finset (ℕ × ℕ) :=
  Finset.product b1 b2

-- Calculation for the probability condition
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

theorem probability_multiple_of_3 :
  let outcomes := draw_chips box1 box2 in
  let favorable_outcomes := Finset.filter (fun (x : ℕ × ℕ) => is_multiple_of_3 (x.1 * x.2)) outcomes in
  (Finset.card favorable_outcomes) / (Finset.card outcomes) = 0 := by
  sorry

end probability_multiple_of_3_l817_817367


namespace slope_range_as_angles_l817_817178

theorem slope_range_as_angles : 
  ∀ (m : ℝ), ∃ θ ∈ set.Ico 0 π, θ = real.arctan m :=
by
  sorry

end slope_range_as_angles_l817_817178


namespace infinite_rational_points_l817_817470

noncomputable def infinite_points_set (S : Set (ℝ × ℝ)) : Prop :=
  (Set.Infinite S) ∧ 
  (∀ (A B C : (ℝ × ℝ)), (A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C) → 
    ¬ ∃ (k : ℝ), colinear k A B C) ∧
  (∀ (A B : (ℝ × ℝ)), (A ∈ S ∧ B ∈ S ∧ A ≠ B) → 
    ∃ (r ∈ ℚ), dist A B = r)

theorem infinite_rational_points :
  ∃ S : Set (ℝ × ℝ), infinite_points_set S := 
sorry

end infinite_rational_points_l817_817470


namespace impossible_four_common_tangents_l817_817604

noncomputable def commonTangentsOfCircles (r1 r2 d : ℝ) : ℕ :=
if d = 0 then 0 -- concentric circles
else if abs (r1 - r2) < d ∧ d < r1 + r2 then 2 -- circles intersect at two points
else if d = abs (r1 - r2) then 2 -- internally tangent circles
else if d = r1 + r2 then 3 -- externally tangent circles
else 4 -- should be logically impossible for different radii

theorem impossible_four_common_tangents (r1 r2 : ℝ) (r1_ne_r2 : r1 ≠ r2) (d : ℝ) :
  commonTangentsOfCircles r1 r2 d ≠ 4 :=
begin
  sorry,
end

end impossible_four_common_tangents_l817_817604


namespace perimeter_of_shaded_region_l817_817964

theorem perimeter_of_shaded_region :
  ∃ (perimeter : ℝ), perimeter = 2.094 ∧
  (∀ (A B C D E F G H : points)
    (square : is_square A B C D 1)
    (quarter_circles : ∀ (P ∈ {A, B, C, D}), is_quarter_circle P ((1 : ℝ)))
    (intersections : intersecting_arcs E F G H (circle_arc A D) (circle_arc B E) (circle_arc C F) (circle_arc D G)),
      shaded_region_perimeter E F G H = perimeter)
:= sorry

end perimeter_of_shaded_region_l817_817964


namespace two_digit_integers_remainder_3_count_l817_817877

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l817_817877


namespace problem_equiv_l817_817927

theorem problem_equiv (a b : ℝ) (h : a ≠ b) : 
  (a^2 - 4 * a + 5 > 0) ∧ (a^2 + b^2 ≥ 2 * (a - b - 1)) :=
by {
  sorry
}

end problem_equiv_l817_817927


namespace count_two_digit_integers_remainder_3_l817_817825

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817825


namespace find_a_l817_817686

variable {a b : ℝ}
variable (h1 : a > 0) (h2 : b > 0)
variable (h3 : ∀ x : ℝ, a * (Real.sec (b * x)) ≥ 3)

theorem find_a : a = 3 :=
by
  sorry

end find_a_l817_817686


namespace find_value_l817_817997

noncomputable def roots_of_equation (a b c : ℝ) : Prop :=
  10 * a^3 + 502 * a + 3010 = 0 ∧
  10 * b^3 + 502 * b + 3010 = 0 ∧
  10 * c^3 + 502 * c + 3010 = 0

theorem find_value (a b c : ℝ)
  (h : roots_of_equation a b c) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 903 :=
by
  sorry

end find_value_l817_817997


namespace maximal_s_value_l817_817206

noncomputable def max_tiles_sum (a b c : ℕ) : ℕ := a + c

theorem maximal_s_value :
  ∃ s : ℕ, 
    ∃ a b c : ℕ, 
      4 * a + 4 * c + 5 * b = 3986000 ∧ 
      s = max_tiles_sum a b c ∧ 
      s = 996500 := 
    sorry

end maximal_s_value_l817_817206


namespace ratio_lateral_surface_area_to_surface_area_l817_817074

theorem ratio_lateral_surface_area_to_surface_area (r : ℝ) (h : ℝ) (V_sphere V_cone A_cone A_sphere : ℝ)
    (h_eq : h = r)
    (V_sphere_eq : V_sphere = (4 / 3) * Real.pi * r^3)
    (V_cone_eq : V_cone = (1 / 3) * Real.pi * (2 * r)^2 * h)
    (V_eq : V_sphere = V_cone)
    (A_cone_eq : A_cone = 2 * Real.sqrt 5 * Real.pi * r^2)
    (A_sphere_eq : A_sphere = 4 * Real.pi * r^2) :
    A_cone / A_sphere = Real.sqrt 5 / 2 := by
  sorry

end ratio_lateral_surface_area_to_surface_area_l817_817074


namespace monic_polynomial_of_transformed_roots_l817_817117

theorem monic_polynomial_of_transformed_roots (r1 r2 r3 : ℝ) 
  (h1 : Polynomial.aeval r1 (Polynomial.C (3 : ℝ) * 
    Polynomial.X^2 + Polynomial.C (8 : ℝ) - Polynomial.C (1 : ℝ) * Polynomial.X^3) = 0)
  (h2 : Polynomial.aeval r2 (Polynomial.C (3 : ℝ) * 
    Polynomial.X^2 + Polynomial.C (8 : ℝ) - Polynomial.C (1 : ℝ) * Polynomial.X^3) = 0)
  (h3 : Polynomial.aeval r3 (Polynomial.C (3 : ℝ) * 
    Polynomial.X^2 + Polynomial.C (8 : ℝ) - Polynomial.C (1 : ℝ) * Polynomial.X^3) = 0) :
  Polynomial.C (9 : ℝ) * Polynomial.X^2 + Polynomial.C (216 : ℝ) - Polynomial.X^3 = 
  Polynomial.map (Polynomial.C (3 : ℝ) * Polynomial.C (3 : ℝ) * Polynomial.C (3 : ℝ)) 
  (Polynomial.C (3 : ℝ) * Polynomial.X^2 + Polynomial.C (8 : ℝ) - Polynomial.C (1 : ℝ) * Polynomial.X^3) :=
sorry

end monic_polynomial_of_transformed_roots_l817_817117


namespace price_ratio_l817_817340

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l817_817340


namespace shifted_function_equiv_l817_817033

def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x)

def g (x : ℝ) : ℝ := f (x - Real.pi / 8)

theorem shifted_function_equiv :
  g x = 3 * Real.sin (2 * x - Real.pi / 4) :=
by
  -- skipping the proof here
  sorry

end shifted_function_equiv_l817_817033


namespace sum_arithmetic_series_base6_l817_817376

theorem sum_arithmetic_series_base6 :
  let n := 35  -- This is 55_6 in base 10
  let a := 1   -- This is 1_6 in base 10
  let l := 35  -- This represents 55_6 in base 10
  S_base10 = (n * (a + l)) / 2
  S_base10 = 630 :=
  let S_base6 := "2530"_6 -- Conversion of S_base10 to base 6
  (∑ i in finset.range (n + 1), (i + 1)) = 630 ∧ S_base6 = 2530_6 :=
by
  sorry

end sum_arithmetic_series_base6_l817_817376


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817329

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817329


namespace score_order_l817_817277

variable (score : Type) [LinearOrder score]
variables (A B C : score)

theorem score_order (cond1 : A > B) (cond2 : C > B ∧ C > A → false) (cond3 : C > B → false)
                  (unique : ∃! p, p ∈ {A > B, C > B ∧ C > A, C > B}) :
  A > B ∧ B > C :=
by 
  sorry

end score_order_l817_817277


namespace range_of_m_l817_817502

theorem range_of_m (m : ℝ) :
  (∃ x y, y = x^2 + m * x + 2 ∧ x - y + 1 = 0 ∧ 0 ≤ x ∧ x ≤ 2) → m ≤ -1 :=
by
  sorry

end range_of_m_l817_817502


namespace diagonal_length_l817_817372

theorem diagonal_length (d : ℝ) 
  (offset1 offset2 : ℝ) 
  (area : ℝ) 
  (h_offsets : offset1 = 11) 
  (h_offsets2 : offset2 = 9) 
  (h_area : area = 400) : d = 40 :=
by 
  sorry

end diagonal_length_l817_817372


namespace fly_distance_A_to_B_l817_817567

open_locale classical

-- Definitions for the conditions
def distance_AB := 100 -- distance AB is 100 km
def speed_A := 20 -- speed of cyclist A
def speed_B := 30 -- speed of cyclist B
def speed_fly := 50 -- speed of the fly

-- Total distance the cyclists travel towards each other
def cyclists_meeting_time := distance_AB / (speed_A + speed_B) -- time until cyclists meet

-- Proving the distance the fly travels in the direction from A to B
theorem fly_distance_A_to_B : 
  let total_distance_fly := speed_fly * cyclists_meeting_time in
  total_distance_fly / 2 = 70 := 
by 
  -- This follows from the problem setup and solution
  sorry

end fly_distance_A_to_B_l817_817567


namespace arithmetic_sequence_ratio_l817_817737

theorem arithmetic_sequence_ratio (x y a₁ a₂ a₃ b₁ b₂ b₃ b₄ : ℝ) (h₁ : x ≠ y)
    (h₂ : a₁ = x + d) (h₃ : a₂ = x + 2 * d) (h₄ : a₃ = x + 3 * d) (h₅ : y = x + 4 * d)
    (h₆ : b₁ = x - d') (h₇ : b₂ = x + d') (h₈ : b₃ = x + 2 * d') (h₉ : y = x + 3 * d') (h₁₀ : b₄ = x + 4 * d') :
    (b₄ - b₃) / (a₂ - a₁) = 8 / 3 :=
by
  sorry

end arithmetic_sequence_ratio_l817_817737


namespace fraction_of_ties_l817_817198

theorem fraction_of_ties (mark_wins : ℚ) (jane_wins : ℚ) (h_mark : mark_wins = 5/12) (h_jane : jane_wins = 1/4) : 
  (1 - (mark_wins + jane_wins) = 1/3) :=
by
  rw [h_mark, h_jane]
  norm_num
  exact sorry

end fraction_of_ties_l817_817198


namespace g_neg6_eq_neg1_l817_817110

def f : ℝ → ℝ := fun x => 4 * x - 6
def g : ℝ → ℝ := fun x => 2 * x^2 + 7 * x - 1

theorem g_neg6_eq_neg1 : g (-6) = -1 := by
  sorry

end g_neg6_eq_neg1_l817_817110


namespace solution_exists_l817_817553

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (log 3 ((4 * x + 12)/(6 * x - 4)) + log 3 ((6 * x - 4)/(2 * x - 3)) = 2) ∧
  (x = 39/14)

theorem solution_exists : problem_statement :=
by
  sorry

end solution_exists_l817_817553


namespace count_two_digit_integers_remainder_3_l817_817829

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l817_817829


namespace problem_1984_china_hs_math_league_l817_817180

open Complex

noncomputable def set_S (α : ℝ) : Set ℂ := 
  {z | ∃ w : ℂ, arg w = α ∧ z = w^(-2)}

theorem problem_1984_china_hs_math_league (α : ℝ) (hα : 0 ≤ α ∧ α < 2 * π) :
  ¬(∃ z ∈ set_S α, arg z = 2 * α) ∧ 
  (∃ z ∈ set_S α, arg z = -2 * α) ∧ 
  ¬(∃ z ∈ set_S α, arg z = -α)  :=
by
  sorry

end problem_1984_china_hs_math_league_l817_817180


namespace count_two_digit_integers_remainder_3_div_7_l817_817899

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l817_817899


namespace range_of_x0_l817_817004

noncomputable def point_on_circle_and_line (x0 : ℝ) (y0 : ℝ) : Prop :=
(x0^2 + y0^2 = 1) ∧ (3 * x0 + 2 * y0 = 4)

theorem range_of_x0 
  (x0 : ℝ) (y0 : ℝ) 
  (h1 : 3 * x0 + 2 * y0 = 4)
  (h2 : ∃ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A ≠ B) ∧ (A + B = (x0, y0))) :
  0 < x0 ∧ x0 < 24 / 13 :=
sorry

end range_of_x0_l817_817004


namespace gel_pen_is_eight_times_ballpoint_pen_l817_817327

variable {x y b g T : ℝ}

-- Condition 1: The total amount paid
def total_amount (x y b g : ℝ) : ℝ := x * b + y * g

-- Condition 2: If all pens were gel pens, the amount paid would be four times the actual amount
def all_gel_pens_equation (x y g T : ℝ) : Prop := (x + y) * g = 4 * T

-- Condition 3: If all pens were ballpoint pens, the amount paid would be half the actual amount
def all_ballpoint_pens_equation (x y b T : ℝ) : Prop := (x + y) * b = 1 / 2 * T

theorem gel_pen_is_eight_times_ballpoint_pen :
  ∀ (x y b g : ℝ), 
  ∃ T,
  total_amount x y b g = T →
  all_gel_pens_equation x y g T →
  all_ballpoint_pens_equation x y b T →
  g = 8 * b := 
by
  intros x y b g,
  use total_amount x y b g,
  intros h_total h_gel h_ball,
  sorry

end gel_pen_is_eight_times_ballpoint_pen_l817_817327


namespace m_divisible_by_p_l817_817116

theorem m_divisible_by_p (p m n : ℕ) (hp : Nat.Prime p) (h2 : 2 < p)
  (hmn : (m : ℚ) / (n : ℚ) = (1 + ∑ i in Finset.range (p-1), 1 / (i + 1))) :
  p ∣ m := 
sorry

end m_divisible_by_p_l817_817116


namespace triangle_solution_l817_817468

-- Define the triangle and its properties
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (area : ℝ)

-- Define the conditions as hypotheses
def conditions (A B C a b c area : ℝ) : Prop :=
  (∃ A' : A' = A) ∧
  (∃ B' : B' = B) ∧
  (∃ C' : C' = C) ∧
  (∃ a' : a' = a) ∧
  (∃ b' : b' = b) ∧
  (∃ c' : c' = c) ∧
  (sqrt 2 * cos A * (b * cos C + c * cos B) = a) ∧
  (a = sqrt 5) ∧
  (area = sqrt 2 - 1)

-- Define the statements to prove
def triangle_problem (A B C a b c area : ℝ) (h : conditions A B C a b c area) : Prop :=
  (A = π / 4) ∧
  (a = sqrt 5 → (√2 - 1 = area) → b + c + a = 3 + sqrt 5)

-- The main theorem statement
theorem triangle_solution (A B C a b c area : ℝ)
  (h : conditions A B C a b c area) : triangle_problem A B C a b c area h := by
  sorry

end triangle_solution_l817_817468


namespace num_emails_received_after_second_deletion_l817_817094

-- Define the initial conditions and final question
variable (initialEmails : ℕ)    -- Initial number of emails
variable (deletedEmails1 : ℕ)   -- First batch of deleted emails
variable (receivedEmails1 : ℕ)  -- First batch of received emails
variable (deletedEmails2 : ℕ)   -- Second batch of deleted emails
variable (receivedEmails2 : ℕ)  -- Second batch of received emails
variable (receivedEmails3 : ℕ)  -- Third batch of received emails
variable (finalEmails : ℕ)      -- Final number of emails in the inbox

-- Conditions based on the problem description
axiom initialEmails_def : initialEmails = 0
axiom deletedEmails1_def : deletedEmails1 = 50
axiom receivedEmails1_def : receivedEmails1 = 15
axiom deletedEmails2_def : deletedEmails2 = 20
axiom receivedEmails3_def : receivedEmails3 = 10
axiom finalEmails_def : finalEmails = 30

-- Question: Prove that the number of emails received after the second deletion is 5
theorem num_emails_received_after_second_deletion : receivedEmails2 = 5 :=
by
  sorry

end num_emails_received_after_second_deletion_l817_817094


namespace chord_exists_through_point_l817_817200

-- Definitions based on conditions
def chord_through_point (R l : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), (dist A B = l) ∧ (line_through A B).contains P ∧ P ∈ circle_center_radius (0, 0) R

-- The main theorem statement
theorem chord_exists_through_point (R l : ℝ) (P : ℝ × ℝ) (hP : dist P (0, 0) < R) :
  ∃ (A B : ℝ × ℝ), dist A B = l ∧ (line_through A B).contains P ∧ P ∈ circle_center_radius (0, 0) R :=
sorry

end chord_exists_through_point_l817_817200


namespace count_two_digit_integers_remainder_3_l817_817867

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l817_817867


namespace debby_drinking_days_l817_817356

def starting_bottles := 264
def daily_consumption := 15
def bottles_left := 99

theorem debby_drinking_days : (starting_bottles - bottles_left) / daily_consumption = 11 :=
by
  -- proof steps will go here
  sorry

end debby_drinking_days_l817_817356


namespace geometric_sequence_ratio_l817_817453

theorem geometric_sequence_ratio (a1 q : ℝ) (h : (a1 * (1 - q^3) / (1 - q)) / (a1 * (1 - q^2) / (1 - q)) = 3 / 2) :
  q = 1 ∨ q = -1 / 2 := by
  sorry

end geometric_sequence_ratio_l817_817453


namespace smallest_positive_angle_solution_l817_817353

theorem smallest_positive_angle_solution :
  ∃ x : ℝ, (0 < x ∧ ∃ k : ℤ, x = k * (π / 180) ∧ tan (6 * x) = (cos x - sin x) / (cos x + sin x)) ∧ x = 45 / 7 * (π / 180) :=
by {
  sorry
}

end smallest_positive_angle_solution_l817_817353


namespace sequence_sum_bound_l817_817519

theorem sequence_sum_bound (n : ℕ) (hn : 0 < n)
  (a : ℕ → ℕ)
  (h_periodic : ∀ i, a (n + i) = a i)
  (h_increasing : ∀ i, i < n → a i ≤ a (i + 1))
  (h_bound : a n ≤ a 0 + n)
  (h_inequalities : ∀ i, i < n → a (a i) ≤ n + i) :
  (∑ i in finset.range n, a i) ≤ n^2 :=
sorry

end sequence_sum_bound_l817_817519


namespace shooter_miss_probability_l817_817270

theorem shooter_miss_probability 
  (hit_prob : ℝ) (miss_prob : ℝ) (two_shots_miss_prob : ℝ) 
  (h1 : hit_prob = 0.8) 
  (h2 : miss_prob = 1 - hit_prob) 
  (h3 : two_shots_miss_prob = miss_prob * miss_prob) : 
  two_shots_miss_prob = 0.04 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end shooter_miss_probability_l817_817270


namespace count_two_digit_remainders_l817_817807

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l817_817807


namespace range_of_z_l817_817764

theorem range_of_z (x y : ℝ) (h : x^2 + 2 * x * y + 4 * y^2 = 6) :
  4 ≤ x^2 + 4 * y^2 ∧ x^2 + 4 * y^2 ≤ 12 :=
by
  sorry

end range_of_z_l817_817764


namespace sequence_a_n_geometric_sum_of_squares_a_n_sum_of_reciprocals_b_n_l817_817109

-- Definitions based on given conditions
def sequence_sum_first_n (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : Prop :=
  S n = 2 * a n - 1

def b_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : Prop :=
  b n = Real.log2 (a (n + 1))

-- Problem statements to prove

theorem sequence_a_n_geometric (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : ∀ n, sequence_sum_first_n a S n) :
  ∀ n, a n = 2^(n - 1) :=
by
  sorry

theorem sum_of_squares_a_n (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : ∀ n, sequence_sum_first_n a S n) :
  ∀ n, (∑ i in Finset.range (n + 1), (a i)^2) = (2^(2*n) - 1) / 3 :=
by
  sorry

theorem sum_of_reciprocals_b_n (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) (h1 : ∀ n, sequence_sum_first_n a sorry)
  (h2 : ∀ n, b_sequence a b n) :
  ∀ n, (∑ i in Finset.range (n + 1), 1 / (b i * b (i + 1))) < 1 :=
by
  sorry

end sequence_a_n_geometric_sum_of_squares_a_n_sum_of_reciprocals_b_n_l817_817109


namespace intersection_A_B_l817_817007

def setA : set ℝ := { x | 2^(x-1) < 4 }
def setB : set ℝ := { x | x^2 - 4*x < 0 }

theorem intersection_A_B : setA ∩ setB = { x | 0 < x ∧ x < 3 } :=
sorry

end intersection_A_B_l817_817007


namespace sqrt_expression_meaningful_l817_817061

theorem sqrt_expression_meaningful {x : ℝ} : (2 * x - 4) ≥ 0 → x ≥ 2 :=
by
  intro h
  sorry

end sqrt_expression_meaningful_l817_817061


namespace maximum_value_of_N_l817_817723

-- Define J_k based on the conditions given
def J (k : ℕ) : ℕ := 10^(k+3) + 128

-- Define the number of factors of 2 in the prime factorization of J_k
def N (k : ℕ) : ℕ := Nat.factorization (J k) 2

-- The proposition to be proved
theorem maximum_value_of_N (k : ℕ) (hk : k > 0) : N 4 = 7 :=
by
  sorry

end maximum_value_of_N_l817_817723
