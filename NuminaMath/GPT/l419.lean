import Mathlib

namespace max_possible_y_l419_419724

theorem max_possible_y (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : y ≤ 18 :=
begin
  sorry
end

end max_possible_y_l419_419724


namespace prove_eccentricity_5_l419_419397

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (P F1 F2 : ℝ × ℝ) :=
  ∃ (e : ℝ), e = 5 ∧ 
  (a > 0 ∧ b > 0) ∧
  ((P.1^2 / a^2) - (P.2^2 / b^2) = 1) ∧
  (let PF1 := dist P F1 in let PF2 := dist P F2 in 
   let F1F2 := dist F1 F2 in 
   F1F2 = PF1 + PF1 - PF2 ∧ 
   ∠ P F1 F2 = π / 2 ∧ 
   e = (F1.1 - F2.1)^2 / a^2 + 1)
  
theorem prove_eccentricity_5 (a b : ℝ) (P F1 F2 : ℝ × ℝ) : 
  eccentricity_of_hyperbola a b P F1 F2 :=
sorry

end prove_eccentricity_5_l419_419397


namespace was_not_speeding_l419_419009

theorem was_not_speeding (x s : ℝ) (s_obs : ℝ := 26.5) (x_limit : ℝ := 120)
  (brake_dist_eq : s = 0.01 * x + 0.002 * x^2) : s_obs < 30 → x ≤ x_limit :=
sorry

end was_not_speeding_l419_419009


namespace intersection_of_sets_l419_419518

theorem intersection_of_sets :
  { x : ℝ | x^2 - x - 6 < 0 } ∩ { x : ℝ | x^2 + 2x - 8 > 0 } = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_of_sets_l419_419518


namespace g_increasing_l419_419152

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + a * x + 2
noncomputable def f' (a x : ℝ) : ℝ := x^2 - 2 * a * x + a
noncomputable def g (a x : ℝ) : ℝ := f' a x / x

theorem g_increasing (a : ℝ) (h : a < 0) : ∀ x ∈ Set.Ioi 1, has_deriv_at (λ x, g a x) (1 - a / (x^2)) x ∧ (1 - a / (x^2)) > 0 :=
by
  intros x hx
  have g_x := f' a x / x
  have g'_x := 1 - a / (x^2)
  split
  - apply has_deriv_at_div
    -- Ensure derivatives are well defined
    -- Filling this out can rely on internal mathlib properties
    sorry
  - apply h
    -- Ensuring a < 0 implies positive derivative for domain (1, +∞)
    sorry

end g_increasing_l419_419152


namespace sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14_l419_419796

noncomputable def value_x : ℝ := real.sqrt (20 - 8 * real.sqrt 5) + real.sqrt (20 + 8 * real.sqrt 5)

theorem sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14 :
  value_x = 2 * real.sqrt 14 :=
by
  sorry

end sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14_l419_419796


namespace number_of_solutions_l419_419201

def equation_numerator (x : ℕ) : ℕ := (List.product (List.map (λ n, x - n) (List.range 200).map (1 + .)))
def equation_denominator (x : ℕ) : ℕ := (List.product (List.map (λ n, x - n^2) (List.range 10).map (1 + .))) * (x - 11^3) * (x - 12^3) * (x - 13^3)

theorem number_of_solutions : 
  ∃ n, equation_denominator n ≠ 0 ∧ equation_numerator n = 0 ∧ 
       (finset.card (finset.filter (λ x, equation_numerator x = 0 ∧ equation_denominator x ≠ 0) (finset.Icc 1 200))) = 190 :=
begin
  sorry
end

end number_of_solutions_l419_419201


namespace diagonals_of_octagon_l419_419546

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l419_419546


namespace always_non_monotonic_l419_419179

noncomputable def f (a : ℝ) (t : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ t then (2 * a - 1) * x + 3 * a - 4 else x^3 - x

theorem always_non_monotonic (a t : ℝ) (h : a ≤ 1/2) : 
  ¬ (∀ x y : ℝ, x < y → f a t x < f a t y ∨ f a t x > f a t y) :=
begin
  sorry
end

end always_non_monotonic_l419_419179


namespace fifth_equation_value_of_selected_sum_l419_419654

-- Definition of the sum of cubes up to n,
-- and the sum of the first n numbers, for use in our proofs
def sum_of_cubes : ℕ → ℕ
| 0       := 0
| (n + 1) := (n + 1)^3 + sum_of_cubes n

def sum_of_first_n : ℕ → ℕ
| 0       := 0
| (n + 1) := (n + 1) + sum_of_first_n n

-- The fifth equation in the pattern:
theorem fifth_equation :
  sum_of_cubes 5 = sum_of_first_n 5 ^ 2 :=
by 
  -- Here we would provide the actual math proof,
  -- but we're using sorry to indicate the proof is omitted.
  sorry

-- The value of 6^3 + 7^3 + 8^3 + 9^3 + 10^3:
theorem value_of_selected_sum :
  (6^3 + 7^3 + 8^3 + 9^3 + 10^3) = (sum_of_first_n 10)^2 - (sum_of_first_n 5)^2 :=
by 
  -- Here we would provide the actual math proof,
  -- but we're using sorry to indicate the proof is omitted.
  sorry

end fifth_equation_value_of_selected_sum_l419_419654


namespace gcd_a_b_l419_419468

def a (n : ℤ) : ℤ := n^5 + 6 * n^3 + 8 * n
def b (n : ℤ) : ℤ := n^4 + 4 * n^2 + 3

theorem gcd_a_b (n : ℤ) : ∃ d : ℤ, d = Int.gcd (a n) (b n) ∧ (d = 1 ∨ d = 3) :=
by
  sorry

end gcd_a_b_l419_419468


namespace no_absolute_winner_prob_l419_419899

open_locale probability

-- Define the probability of Alyosha winning against Borya
def P_A_wins_B : ℝ := 0.6

-- Define the probability of Borya winning against Vasya
def P_B_wins_V : ℝ := 0.4

-- There are no ties, and each player plays with each other once
-- Conditions ensure that all pairs have played exactly once

-- Define the event that there will be no absolute winner
def P_no_absolute_winner : ℝ := P_A_wins_B * P_B_wins_V * 1 + P_A_wins_B * (1 - P_B_wins_V) * (1 - 1)

-- Statement of the problem: Prove that the probability of event C is 0.24
theorem no_absolute_winner_prob :
  P_no_absolute_winner = 0.24 :=
  by
    -- Placeholder for proof
    sorry

end no_absolute_winner_prob_l419_419899


namespace variance_transformation_l419_419634

noncomputable def variance (n : Nat) (data : Fin n -> ℝ) : ℝ :=
  let mean := (Array.foldl (fun acc x => acc + data x) 0 (Array.range n)) / (n : ℝ)
  (Array.foldl (fun acc x => acc + (data x - mean) ^ 2) 0 (Array.range n)) / (n : ℝ)

theorem variance_transformation (k : Fin 8 → ℝ) 
  (h : variance 8 k = 3) : variance 8 (fun i => 2 * (k i - 3)) = 12 := by
  sorry

end variance_transformation_l419_419634


namespace second_largest_is_13_l419_419760

def numbers : List ℕ := [10, 11, 12, 13, 14]

theorem second_largest_is_13 (h : numbers = [10, 11, 12, 13, 14]) : 13 = List.sort (· <= ·) numbers).reverse.nth 1 :=
sorry

end second_largest_is_13_l419_419760


namespace first_player_always_wins_l419_419338

theorem first_player_always_wins (A B : ℤ) (hA : A ≠ 0) (hB : B ≠ 0) : A + B + 1998 = 0 → 
  (∃ (a b c : ℤ), (a = A ∨ a = B ∨ a = 1998) ∧ (b = A ∨ b = B ∨ b = 1998) ∧ (c = A ∨ c = B ∨ c = 1998) ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (∃ (r1 r2 : ℚ), r1 ≠ r2 ∧ r1 * r1 * a + r1 * b + c = 0 ∧ r2 * r2 * a + r2 * b + c = 0)) :=
sorry

end first_player_always_wins_l419_419338


namespace rowing_time_l419_419055

def Vm : ℝ := 5 -- Speed in still water (kmph)
def Vc : ℝ := 1 -- Speed of current (kmph)
def D : ℝ := 2.4 -- Distance to the place (km)

def Vup : ℝ := Vm - Vc -- Effective speed upstream
def Vdown : ℝ := Vm + Vc -- Effective speed downstream

def Tup : ℝ := D / Vup -- Time to row upstream
def Tdown : ℝ := D / Vdown -- Time to row downstream

def Ttotal : ℝ := Tup + Tdown -- Total time to row to the place and back

theorem rowing_time : Ttotal = 1 := by
  -- The details of the proof will be filled in here
  sorry

end rowing_time_l419_419055


namespace sum_gcd_lcm_l419_419347

theorem sum_gcd_lcm (a₁ a₂ : ℕ) (h₁ : a₁ = 36) (h₂ : a₂ = 495) :
  Nat.gcd a₁ a₂ + Nat.lcm a₁ a₂ = 1989 :=
by
  -- Proof can be added here
  sorry

end sum_gcd_lcm_l419_419347


namespace shortest_tree_height_proof_l419_419331

def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

theorem shortest_tree_height_proof : shortest_tree_height = 50 := by
  sorry

end shortest_tree_height_proof_l419_419331


namespace smallest_positive_difference_l419_419274

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) : (∃ n : ℤ, n > 0 ∧ n = a - b) → n = 17 :=
by sorry

end smallest_positive_difference_l419_419274


namespace sin_of_angle_in_right_triangle_l419_419653

theorem sin_of_angle_in_right_triangle
  (A B C : ℝ)
  (h_right : C = 90)
  (h_angle_A : A = 30) :
  sin (60 : ℝ) = (√3 / 2) := by
  sorry

end sin_of_angle_in_right_triangle_l419_419653


namespace minimum_pieces_to_find_fish_and_sausage_l419_419816

theorem minimum_pieces_to_find_fish_and_sausage :
  ∀ (grid : ℕ → ℕ → Prop), 
    (∀ i j, 0 ≤ i ∧ i < 8 ∧ 0 ≤ j ∧ j < 8 → i ≠ j → 
      (∃ i1 j1 i2 j2, grid i1 j1 = "P" ∧ grid i2 j2 = "K")) → 
    (∀ i j, 0 ≤ i ∧ i < 6 ∧ 0 ≤ j ∧ j < 6 → 
      (∃ i1 j1 i2 j2, 0 ≤ i1 ∧ i1 < 6 ∧ 0 ≤ j1 ∧ j1 < 6 ∧ 
       grid i1 j1 = "P" ∧ grid i2 j2 = "P")) → 
    (∀ i j, 0 ≤ i ∧ i < 3 ∧ 0 ≤ j ∧ j < 3 → 
      (∑ i1 j1, grid i1 j1 = "K") ≤ 1) → 
    (∃ (coords : finset (ℕ × ℕ)), 
      coords.card = 5 ∧ 
      ∀ (i j : ℕ), (i, j) ∈ coords → 
      grid i j = "P" ∧ grid i j = "K") :=
sorry

end minimum_pieces_to_find_fish_and_sausage_l419_419816


namespace solve_for_x_l419_419619

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end solve_for_x_l419_419619


namespace remainder_of_sum_division_l419_419790

def a1 : ℕ := 2101
def a2 : ℕ := 2103
def a3 : ℕ := 2105
def a4 : ℕ := 2107
def a5 : ℕ := 2109
def n : ℕ := 12

theorem remainder_of_sum_division : ((a1 + a2 + a3 + a4 + a5) % n) = 1 :=
by {
  sorry
}

end remainder_of_sum_division_l419_419790


namespace find_x_l419_419457

theorem find_x :
  (∃ x : ℝ, real.cbrt (5 - x/3) = -4) → x = 207 :=
by
  intro hx
  rcases hx with ⟨x, H⟩
  sorry

end find_x_l419_419457


namespace proper_subset_count_l419_419519

open Set

def number_of_proper_subsets {α : Type*} (s : Set α) : ℕ :=
  2 ^ s.ncard - 1

theorem proper_subset_count :
  let M := ({1, 3} : Set ℕ)
  let N := {x | 0 < x ∧ x < 3 ∧ x ∈ ℤ}
  let P := M ∪ N
  number_of_proper_subsets P = 7 :=
by
  sorry

end proper_subset_count_l419_419519


namespace total_spent_l419_419767

-- Constants representing the conditions from the problem
def cost_per_deck : ℕ := 8
def tom_decks : ℕ := 3
def friend_decks : ℕ := 5

-- Theorem stating the total amount spent by Tom and his friend
theorem total_spent : tom_decks * cost_per_deck + friend_decks * cost_per_deck = 64 := by
  sorry

end total_spent_l419_419767


namespace Marcy_120_votes_l419_419225

-- Definitions based on conditions
def votes (name : String) : ℕ := sorry -- placeholder definition

-- Conditions
def Joey_votes := votes "Joey" = 8
def Jill_votes := votes "Jill" = votes "Joey" + 4
def Barry_votes := votes "Barry" = 2 * (votes "Joey" + votes "Jill")
def Marcy_votes := votes "Marcy" = 3 * votes "Barry"
def Tim_votes := votes "Tim" = votes "Marcy" / 2
def Sam_votes := votes "Sam" = votes "Tim" + 10

-- Theorem to prove
theorem Marcy_120_votes : Joey_votes → Jill_votes → Barry_votes → Marcy_votes → Tim_votes → Sam_votes → votes "Marcy" = 120 := by
  intros
  -- Skipping the proof
  sorry

end Marcy_120_votes_l419_419225


namespace quadrilateral_BX_eq_XC_l419_419675

theorem quadrilateral_BX_eq_XC
  (A B C D X : Point)
  (theta : ℝ)
  (h1 : convex_quadrilateral A B C D)
  (h2 : angle A B C = theta)
  (h3 : angle B C D = theta)
  (h4 : angle X A D = 90 - theta)
  (h5 : angle X D A = 90 - theta) :
  dist B X = dist X C :=
sorry

end quadrilateral_BX_eq_XC_l419_419675


namespace gazprom_rd_expense_l419_419934

theorem gazprom_rd_expense
  (R_and_D_t : ℝ) (ΔAPL_t_plus_1 : ℝ)
  (h1 : R_and_D_t = 3289.31)
  (h2 : ΔAPL_t_plus_1 = 1.55) :
  R_and_D_t / ΔAPL_t_plus_1 = 2122 := 
by
  sorry

end gazprom_rd_expense_l419_419934


namespace compute_expr_l419_419696

theorem compute_expr (x : ℝ) (h : x = 1 + sqrt (3) / (1 + sqrt (3) / (1 + sqrt (3) / (1 + ...)))) :
  ∃ A B C : ℤ, ((B ≠ 0) → (∀ p : ℕ, p.prime → ¬ (p^2 ∣ B))) ∧ (∀ p : ℕ, p.prime → ¬ (p^2 ∣ B)) 
  ∧ abs A + abs B + abs C = 17 ∧ (1 / ((x + 1) * (x - 3)) = (A + sqrt (B)) / C) :=
by
  sorry

end compute_expr_l419_419696


namespace octagon_has_20_diagonals_l419_419565

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l419_419565


namespace triangle_area_l419_419694

theorem triangle_area (ABC : Type) [triangle ABC] (I :  incenter ABC) (O : circumcenter ABC) 
  (H: angle O I A = π / 2) 
  (hAI : dist I A = 97) 
  (hBC : dist B C = 144) :
  area ABC = 14040 :=
sorry

end triangle_area_l419_419694


namespace subscription_period_is_18_l419_419391

-- Define the parameters and the condition
def normal_price : ℕ := 34
def issue_discount : ℝ := 0.25
def promotional_savings : ℝ := 9

-- Define the subscription period
def subscription_period (x : ℕ) : Prop :=
  2 * (x:ℝ) * issue_discount = promotional_savings

-- Theorem statement
theorem subscription_period_is_18 : ∃ (x : ℕ), subscription_period x ∧ x = 18 :=
by
  existsi 18
  split
  { unfold subscription_period
    norm_num }
  { refl
  }

end subscription_period_is_18_l419_419391


namespace alton_weekly_profit_l419_419893

-- Definitions of the given conditions
def dailyEarnings : ℕ := 8
def daysInWeek : ℕ := 7
def weeklyRent : ℕ := 20

-- The proof problem: Prove that the total profit every week is $36
theorem alton_weekly_profit : (dailyEarnings * daysInWeek) - weeklyRent = 36 := by
  sorry

end alton_weekly_profit_l419_419893


namespace isosceles_triangle_circle_area_l419_419829

theorem isosceles_triangle_circle_area 
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : a = 4) 
  (h3 : c = 3) 
  (h4 : a = 4) 
  (h5 : b = 4)
  (h6 : c ≠ a)
  (h7 : c ≠ b) :
  let r := 4 in π * r ^ 2 = 16 * π :=
by
  sorry

end isosceles_triangle_circle_area_l419_419829


namespace find_a_l419_419606

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 :=
by
  sorry

end find_a_l419_419606


namespace baker_batches_chocolate_chip_l419_419024

noncomputable def number_of_batches (total_cookies : ℕ) (oatmeal_cookies : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  (total_cookies - oatmeal_cookies) / cookies_per_batch

theorem baker_batches_chocolate_chip (total_cookies oatmeal_cookies cookies_per_batch : ℕ) 
  (h_total : total_cookies = 10) 
  (h_oatmeal : oatmeal_cookies = 4) 
  (h_batch : cookies_per_batch = 3) : 
  number_of_batches total_cookies oatmeal_cookies cookies_per_batch = 2 :=
by
  sorry

end baker_batches_chocolate_chip_l419_419024


namespace transformed_parabola_l419_419655

theorem transformed_parabola (x y : ℝ) : 
  (y = 2 * (x - 1)^2 + 3) → (y = 2 * (x + 1)^2 + 2) :=
by
  sorry

end transformed_parabola_l419_419655


namespace correct_statements_are_1_and_4_l419_419156

open Set

variables {Point Line Plane : Type}

-- Define conditions
variable (l : Line)
variable (m : Line)
variable (alpha : Plane)
variable (not_in_plane : l ∉ alpha)
variable (in_plane : m ∈ alpha)

-- Define the given statements as propositions
def S1 : Prop := l ⊥ alpha → l ⊥ m
def S2 : Prop := l ∥ alpha → l ∥ m
def S3 : Prop := l ⊥ m → l ⊥ alpha
def S4 : Prop := l ∥ m → l ∥ alpha

-- Proof problem
theorem correct_statements_are_1_and_4 : S1 ∧ S4 :=
by
  sorry

end correct_statements_are_1_and_4_l419_419156


namespace average_of_7_consec_with_max_23_l419_419763

def consecutive_sum (a : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a + n - 1)) / 2

theorem average_of_7_consec_with_max_23 : 
  (∃ (S : Finset ℕ) (a : ℕ), S = Finset.range 7 \map (λ i, a + i) ∧ a + 6 = 23 ∧ (S.sum id) / 7 = 20) :=
by
  sorry

end average_of_7_consec_with_max_23_l419_419763


namespace complex_square_real_iff_l419_419743

-- Definitions
def complex_sq_real_condition (a b : ℝ) : Prop :=
  (a + b * complex.I) ^ 2 ∈ ℝ ↔ a = 0 ∨ b = 0

theorem complex_square_real_iff (a b : ℝ) :
  complex_sq_real_condition a b :=
by
  sorry

end complex_square_real_iff_l419_419743


namespace perpendicular_points_on_same_circle_l419_419882

variables (A B C D O P Q R S : Point)
variables (h1 : CyclicQuadrilateral ABCD)
variables (h2 : Center O ABCD)
variables (h3 : PerpendicularFrom O A B P)
variables (h4 : PerpendicularFrom O C D R)
variables (h5 : IntersectsDiagonals P A C Q B D)
variables (h6 : IntersectsDiagonals R A C S B D)
variables (h7 : AngleEq O A B D)
variables (h8 : AngleEq O C D A)

theorem perpendicular_points_on_same_circle :
  OnSameCircle P Q R S :=
by sorry

end perpendicular_points_on_same_circle_l419_419882


namespace final_weight_is_correct_l419_419888

-- Define the initial weight of marble
def initial_weight := 300.0

-- Define the percentage reductions each week
def first_week_reduction := 0.3 * initial_weight
def second_week_reduction := 0.3 * (initial_weight - first_week_reduction)
def third_week_reduction := 0.15 * (initial_weight - first_week_reduction - second_week_reduction)

-- Calculate the final weight of the statue
def final_weight := initial_weight - first_week_reduction - second_week_reduction - third_week_reduction

-- The statement to prove
theorem final_weight_is_correct : final_weight = 124.95 := by
  -- Here would be the proof, which we are omitting
  sorry

end final_weight_is_correct_l419_419888


namespace hyperbola_problem_l419_419326

noncomputable def hyperbola_distance (F1 F2 P : Point) (a : ℝ) : Prop :=
  let d1 := distance P F1
  let d2 := distance P F2
  d1 = 12 ∧ (d1 - d2 = 2*a ∨ d2 - d1 = 2*a) ∧ (d2 = 22 ∨ d2 = 2)

theorem hyperbola_problem (F1 F2 P : Point) (a : ℝ) :
  hyperbola_distance F1 F2 P a :=
begin
  sorry,
end

end hyperbola_problem_l419_419326


namespace degree_of_polynomial_l419_419344

theorem degree_of_polynomial (p : ℕ) (n : ℕ) (m : ℕ) (h : p = 3) (k : n = 15) :
  m = p * n := by
  sorry

-- Given p = 3 (degree of 5x^3) and n = 15 (exponent in (5x^3 + 7)^15)
-- Prove that m = 45 (degree of (5x^3 + 7)^15)
noncomputable def main_theorem : Prop :=
  (degree_of_polynomial 3 15 45 rfl rfl)

end degree_of_polynomial_l419_419344


namespace whale_plankton_consumption_l419_419408

theorem whale_plankton_consumption
  (P : ℕ) -- Amount of plankton consumed in the first hour
  (h1 : ∀ n : ℕ, 1 ≤ n → n ≤ 9 → P + (n - 1) * 3 ∈ ℕ) -- Plankton consumption follows an arithmetic sequence over 9 hours
  (h2 : (finset.range 9).sum (λ n, P + n * 3) = 450) -- Total plankton consumption over 9 hours is 450 kilos
  (h3 : P + 15 = 53) -- On the sixth hour, the whale consumed 53 kilos
  : P = 38 := 
sorry

end whale_plankton_consumption_l419_419408


namespace find_value_l419_419505

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then 1 + 3^x else 2 * x - 1

theorem find_value : f(f(0) + 2) = 4 := by
  sorry

end find_value_l419_419505


namespace payment_ways_l419_419805

-- Define basic conditions and variables
variables {x y z : ℕ}

-- Define the main problem as a Lean statement
theorem payment_ways : 
  ∃ (n : ℕ), n = 9 ∧ 
             (∀ x y z : ℕ, 
              x + y + z ≤ 10 ∧ 
              x + 2 * y + 5 * z = 18 ∧ 
              x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
              (x > 0 ∨ y > 0) ∧ (y > 0 ∨ z > 0) ∧ (z > 0 ∨ x > 0) → 
              n = 9) := 
sorry

end payment_ways_l419_419805


namespace find_t_l419_419174

def z1 : ℂ := 3 + 4 * complex.I
def z2 (t : ℝ) : ℂ := t + 4 * complex.I
def z1_conjugate_z2 (t : ℝ) : ℂ := z1 * conj (z2 t)

theorem find_t (t : ℝ) (h : imag_part (z1_conjugate_z2 t) = 0) : t = 3 :=
by
  sorry

end find_t_l419_419174


namespace isosceles_triangle_circle_area_l419_419824

theorem isosceles_triangle_circle_area 
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : a = 4) 
  (h3 : c = 3) 
  (h4 : a = 4) 
  (h5 : b = 4)
  (h6 : c ≠ a)
  (h7 : c ≠ b) :
  let r := 4 in π * r ^ 2 = 16 * π :=
by
  sorry

end isosceles_triangle_circle_area_l419_419824


namespace unique_solution_l419_419461

theorem unique_solution : ∀ (x y z : ℕ), 
  x > 0 → y > 0 → z > 0 → 
  x^2 = 2 * (y + z) → 
  x^6 = y^6 + z^6 + 31 * (y^2 + z^2) → 
  (x, y, z) = (2, 1, 1) :=
by sorry

end unique_solution_l419_419461


namespace equation_of_line_l419_419970

theorem equation_of_line {C : ℝ} (x y : ℝ) (h_diff : ∀ x, deriv (λ x, x^2 - 3*x + C) x = 2*x - 3) (hx : x = 1) (hy : y = 3) :
  ∃ C, (y = 1^2 - 3*1 + C) ∧ (y = x^2 - 3*x + C) := 
sorry

end equation_of_line_l419_419970


namespace no_absolute_winner_prob_l419_419910

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end no_absolute_winner_prob_l419_419910


namespace shortest_wire_length_l419_419772

-- Given: two cylindrical poles
def radius1 : ℝ := 2
def radius2 : ℝ := 10

-- The shortest wire length enclosing both poles is
noncomputable def wire_length : ℝ :=
  8 * Real.sqrt 5 + (44 * Real.pi) / 3

theorem shortest_wire_length :
  let distance_between_centers : ℝ := radius1 + radius2,
      straight_section_length : ℝ := 2 * Real.sqrt (distance_between_centers^2 - (radius2 - radius1)^2),
      small_circle_arc : ℝ := (2 * Real.pi * radius1 * 120) / 360,
      large_circle_arc : ℝ := (2 * Real.pi * radius2 * 240) / 360
  in (straight_section_length + small_circle_arc + large_circle_arc) = wire_length :=
by
  sorry

end shortest_wire_length_l419_419772


namespace impossibility_of_grid_filling_l419_419668

theorem impossibility_of_grid_filling : 
  ¬ (∃ (grid : Array (Array Nat)), 
    (∀ i j, 0 ≤ i ∧ i + 3 < 100 ∧ 0 ≤ j ∧ j + 4 < 100 → 
      (∑ x in range 3, ∑ y in range 4, if grid[i + x][j + y] = 0 then 1 else 0) = 3 ∧
      (∑ x in range 3, ∑ y in range 4, if grid[i + x][j + y] = 1 then 1 else 0) = 4 ∧
      (∑ x in range 3, ∑ y in range 4, if grid[i + x][j + y] = 2 then 1 else 0) = 5)) :=
sorry

end impossibility_of_grid_filling_l419_419668


namespace restaurant_june_production_l419_419401

-- Define the given conditions
def daily_hot_dogs := 60
def daily_pizzas := daily_hot_dogs + 40
def june_days := 30
def daily_total := daily_hot_dogs + daily_pizzas
def june_total := daily_total * june_days

-- The goal is to prove that the total number of pizzas and hot dogs made in June is 4800
theorem restaurant_june_production : june_total = 4800 := by
  -- Sorry to skip proof
  sorry

end restaurant_june_production_l419_419401


namespace acute_triangle_segments_inside_l419_419054

theorem acute_triangle_segments_inside (ABC : Type) [triangle ABC] :
  (∀ (T : triangle) (hA : acute_angle T.A) (hB : acute_angle T.B) (hC : acute_angle T.C),
    altitudes T ⊆ interior T ∧ angle_bisectors T ⊆ interior T ∧ medians T ⊆ interior T) :=
sorry

end acute_triangle_segments_inside_l419_419054


namespace commission_percentage_l419_419047

theorem commission_percentage (cost_price profit_percentage selling_price : ℝ) 
                              (h1 : cost_price = 20)
                              (h2 : profit_percentage = 0.20)
                              (h3 : selling_price = 30) :
  ∃ commission_percentage, commission_percentage = 25 :=
by
  -- To achieve a 20% profit on the cost price of $20
  let profit_amount := profit_percentage * cost_price
  
  -- The required selling price without commission
  let desired_selling_price := cost_price + profit_amount
  
  -- Commission in terms of percentage on the desired selling price
  let commission := (selling_price - desired_selling_price) / desired_selling_price * 100

  -- We expect that commission should be 25
  use commission
  sorry  -- Proof to show commission = 25 is not required here

end commission_percentage_l419_419047


namespace small_box_balls_l419_419048

theorem small_box_balls (big_box_size small_box_balls total_balls : ℕ) 
    (least_unboxed : total_balls % big_box_size ≥ least_unboxed)
    [∀ m n : ℕ, fact (nat.gcd m n = 1)] :
  (big_box_size = 25) →
  (least_unboxed = 5) →
  (total_balls = 104) →
  small_box_balls = 12 :=
begin
  sorry
end

end small_box_balls_l419_419048


namespace number_of_diagonals_in_octagon_l419_419527

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l419_419527


namespace solve_for_a_l419_419604

-- Given conditions
variables (a b d : ℕ)
hypotheses
  (h1 : a + b = d)
  (h2 : b + d = 7)
  (h3 : d = 4)

-- Prove that a = 1
theorem solve_for_a : a = 1 :=
by {
  sorry
}

end solve_for_a_l419_419604


namespace find_r_over_s_at_2_l419_419734

noncomputable def r (x : ℝ) := 6 * x
noncomputable def s (x : ℝ) := (x + 4) * (x - 1)

theorem find_r_over_s_at_2 :
  r 2 / s 2 = 2 :=
by
  -- The corresponding steps to show this theorem.
  sorry

end find_r_over_s_at_2_l419_419734


namespace problem_statement_l419_419632

noncomputable def f (ω x : ℝ) : ℝ := 3 * Real.cos (ω * x + π / 6) - Real.sin (ω * x - π / 3)

theorem problem_statement (ω : ℝ) (hω : ω > 0) 
  (hperiod : ∀ (x : ℝ), f ω (x + π / ω) = f ω x) :
  ∃ x ∈ Set.Icc 0 (π / 2), f ω x = 2 * Real.sqrt 3 :=
by
  sorry

end problem_statement_l419_419632


namespace arithmetic_sequence_sum_l419_419657

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : a 2 + a 3 = 13) : 
  (finset.range 10).sum (λ n, a (n + 1)) = 155 := 
by
  sorry

end arithmetic_sequence_sum_l419_419657


namespace card_deck_initial_count_l419_419383

theorem card_deck_initial_count 
  (r b : ℕ)
  (h1 : r / (r + b) = 1 / 4)
  (h2 : r / (r + (b + 6)) = 1 / 5) : 
  r + b = 24 :=
by
  sorry

end card_deck_initial_count_l419_419383


namespace bottles_in_a_box_l419_419377

theorem bottles_in_a_box (x : ℕ) (h : 26 / x - 26 / (x + 3) = 0.6) : x = 10 :=
by sorry

end bottles_in_a_box_l419_419377


namespace find_k_for_circle_l419_419982

def represents_circle_of_radius (eq_left : ℝ → ℝ → ℝ) (radius : ℝ) : Prop :=
  ∃ (h : ℝ) (k : ℝ), ∀ x y : ℝ, eq_left x y = (x + h)^2 + (y + k)^2 - radius^2

theorem find_k_for_circle :
  represents_circle_of_radius (λ x y, x^2 + 8*x + y^2 + 10*y - 59) 10 :=
by
  sorry

end find_k_for_circle_l419_419982


namespace find_x_when_y_equals_two_l419_419626

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end find_x_when_y_equals_two_l419_419626


namespace tangent_projection_sum_l419_419157

theorem tangent_projection_sum 
  (O : Point) (r : ℝ) (A P : Point) (d : ℝ) 
  (h_circle : dist O A = r) 
  (h_tangent : tangent_at_point P O r) 
  (A' : Point) 
  (h_projection : orthogonal_projection A' A P) 
  (h_distance : PA' + A' A = d) : 
  ∃ P, 
    (d < 2 * r ∨ d = r + real.sqrt 2 * r) ∧ num_solutions = 2 ∨
    d = 2 * r ∧ num_solutions = 3 ∨
    (2 * r < d ∧ d < r + real.sqrt 2 * r) ∧ num_solutions = 4 ∨
    d > r + real.sqrt 2 * r ∧ num_solutions = 0 := 
sorry

end tangent_projection_sum_l419_419157


namespace smallest_blocks_needed_for_wall_l419_419030

theorem smallest_blocks_needed_for_wall :
  ∀ (height length: ℕ) (block_sizes: list ℕ),
  height = 9 →
  length = 120 →
  block_sizes = [2, 1.5, 1] →
  (∀ row, row < height →
    let 
      number_of_blocks := 
        if even row then 
          60 
        else 
          60
    in length = number_of_blocks * 2) →
  540 = height * (length / 2) :=
by
  intros height length block_sizes h_height h_length h_block_sizes h_blocks_per_row
  sorry

end smallest_blocks_needed_for_wall_l419_419030


namespace projection_length_invariant_l419_419483

theorem projection_length_invariant 
  (A B C O : Point) (E F : Point)
  (hABC_acute : ∠BAC < 90 ∧ ∠ABC < 90 ∧ ∠ACB < 90)
  (hCircumcenter : O = circumcenter A B C)
  (D : Point) (hAOD_BC : line_through A O ∩ line_through B C = D)
  (hConcyclic : cyclic A E D F)
  : ∀ E' F' : Point, 
      (E' ∈ AB ∧ F' ∈ AC ∧ cyclic A E' D F') →
      projection_length EF BC = projection_length E'F' BC := 
sorry

end projection_length_invariant_l419_419483


namespace moles_NaNO3_l419_419464

def reaction (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) : Prop :=
  HNO3 + NaHCO3 = NaNO3 + CO2 + H2O

constants (HNO3 : ℕ) (NaHCO3 : ℕ) (NaNO3 : ℕ) (CO2 : ℕ) (H2O : ℕ)

hypothesis h1 : HNO3 = 1
hypothesis h2 : NaHCO3 = 1

theorem moles_NaNO3 : NaNO3 = 1 :=
by sorry

end moles_NaNO3_l419_419464


namespace octagon_diagonals_l419_419540

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l419_419540


namespace abc_sum_is_twelve_l419_419697

theorem abc_sum_is_twelve
  (f : ℤ → ℤ)
  (a b c : ℕ)
  (h1 : f 1 = 10)
  (h2 : f 0 = 8)
  (h3 : f (-3) = -28)
  (h4 : ∀ x, x > 0 → f x = 2 * a * x + 6)
  (h5 : f 0 = a^2 * b)
  (h6 : ∀ x, x < 0 → f x = 2 * b * x + 2 * c)
  : a + b + c = 12 := sorry

end abc_sum_is_twelve_l419_419697


namespace weight_of_7th_person_l419_419365

theorem weight_of_7th_person (n : ℕ) (a b : ℝ) (H_n : n = 6) (H_a : a = 154) (H_b : b = 151) :
  let W := n * a in
  let X := (b * (n + 1)) - W in
  X = 133 := by
  have H_W : W = 924 := by 
    rw [H_n, H_a]
    rfl
  rw [H_W, H_b, H_n, H_a]
  rfl

end weight_of_7th_person_l419_419365


namespace number_of_diagonals_in_octagon_l419_419524

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l419_419524


namespace solve_for_x_l419_419621

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end solve_for_x_l419_419621


namespace values_with_max_occurrences_l419_419416

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 1)^2 / (x * (x^2 - 1))

theorem values_with_max_occurrences : 
 (set_of (λ a: ℝ, ∃ s: finset ℝ, (∀ x ∈ s, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) ∧ 
                  s.card = 2 ∧ ∀ x ∈ ({x : ℝ | f x = a} : set ℝ), x ∈ s)) = 
 {[a : ℝ | a < -4].union {[a : ℝ | a > 4]]} :=
sorry

end values_with_max_occurrences_l419_419416


namespace chloe_needs_100_nickels_l419_419938

theorem chloe_needs_100_nickels (n : ℕ) : 4 * 10 + 5 * 0.5 + n * 0.05 ≥ 47.50 → n ≥ 100 :=
by
  sorry

end chloe_needs_100_nickels_l419_419938


namespace minimum_distance_to_line_l419_419993

noncomputable def curve (x : ℝ) : ℝ :=
  x^2 - 2 * log (sqrt x)

noncomputable def line : ℝ × ℝ → ℝ :=
  λ P, 4 * P.1 + 4 * P.2 + 1

theorem minimum_distance_to_line (P : ℝ × ℝ)
  (hP : P.2 = curve P.1)
  : distance_to_line P line = (sqrt 2 / 2) * (1 + log 2) := 
sorry

noncomputable def distance_to_line (P : ℝ × ℝ) (line : ℝ × ℝ → ℝ) : ℝ :=
  abs (line P) / sqrt (4^2 + 4^2)

end minimum_distance_to_line_l419_419993


namespace remainder_correct_l419_419466

noncomputable def polynomial_remainder : Polynomial ℤ :=
  let f := Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X^4
  let g := (Polynomial.X - 3)^2
  Polynomial.remainder f g

theorem remainder_correct :
  polynomial_remainder = Polynomial.C (-161) + Polynomial.C 81 * Polynomial.X :=
by sorry

end remainder_correct_l419_419466


namespace volume_maximization_l419_419889

noncomputable def height_max_volume (total_length : ℝ) (side_difference : ℝ) : ℝ :=
  if total_length = 14.8 ∧ side_difference = 0.5 then 1.2 else sorry

noncomputable def max_volume (total_length : ℝ) (side_difference : ℝ) : ℝ :=
  if total_length = 14.8 ∧ side_difference = 0.5 then 2.2 else sorry

theorem volume_maximization (total_length : ℝ) (side_difference : ℝ) :
  total_length = 14.8 ∧ side_difference = 0.5 →
  (height_max_volume total_length side_difference = 1.2) ∧
  (max_volume total_length side_difference = 2.2) :=
by
  intro h
  simp [height_max_volume, max_volume, h]
  sorry

end volume_maximization_l419_419889


namespace inscribed_triangle_area_l419_419891

noncomputable def triangle_inscribed_area (r : ℝ) (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem inscribed_triangle_area :
  ∀ (r : ℝ), r = 12 / Real.pi →
  let a := 2 * r * Real.sin (60 * Real.pi / 180) in
  let b := 2 * r * Real.sin (75 * Real.pi / 180) in
  let c := 2 * r * Real.sin (45 * Real.pi / 180) in
  b = c →
  triangle_inscribed_area r a b c = sorry :=
by
  intro r h rfl B_eq_C
  let a := 2 * r * Real.sin (60 * Real.pi / 180)
  let b := 2 * r * Real.sin (75 * Real.pi / 180)
  let c := 2 * r * Real.sin (45 * Real.pi / 180)
  exact sorry

end inscribed_triangle_area_l419_419891


namespace probability_inequality_l419_419398

theorem probability_inequality (x : ℝ) (h : x ∈ Set.Icc (-4 : ℝ) 4) :
  let favorable_interval := Set.Icc (-1 : ℝ) 3
  let total_interval := Set.Icc (-4 : ℝ) 4
  let favorable_length := ∥3 - (-1)∥ -- Length of [-1, 3]
  let total_length := ∥4 - (-4)∥   -- Length of [-4, 4]
  let prob := favorable_length / total_length
  Set.Icc (-1 : ℝ) 3 ⊂ Set {x | x^2 - 2 * x - 3 ≤ 0} →
  prob = (1 / 2) :=
by
  sorry

end probability_inequality_l419_419398


namespace sum_of_c_n_is_Q_l419_419995

-- Definitions
def arithmetic_sequence (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def sum_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop := ∀ n, S n = (n * (a 1) + (n * (n - 1)) / 2 * (a (n + 1) - a n))

variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {c : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {T : ℕ → ℕ}

-- Arithmetic sequence conditions
axiom h1 : arithmetic_sequence a
axiom h2 : a 3 = 5
axiom h3 : S 6 - S 3 = 27

-- Sequence with product of the first n terms
axiom h4 : T n = 3^(n*(n+1)/2)

-- Sequences definitions
def a_n (n : ℕ) : ℕ := 2*n - 1
def b_n (n : ℕ) : ℕ := 3^n
def c_n (n : ℕ) : ℕ := (a n * b n) / (n^2 + n)

-- Sum of the first n terms of c_n
def Q (n : ℕ) : ℕ := (3^(n+1) / (n+1) - 3)

-- Theorem statement
theorem sum_of_c_n_is_Q (n : ℕ) (a_n_def : ∀ n, a n = 2*n - 1) (b_n_def : ∀ n, b n = 3^n) :
  ∑ i in range n, c_n i = Q n :=
sorry

end sum_of_c_n_is_Q_l419_419995


namespace compute_f_g2_l419_419275

def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 18 / Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x^2 - 3 * x - 4

theorem compute_f_g2 : f (g 2) = 12 * Real.sqrt 2 := by
  sorry

end compute_f_g2_l419_419275


namespace compute_ns_l419_419267

noncomputable def f : ℝ → ℝ :=
sorry

-- Defining the functional equation as a condition
def functional_equation (f : ℝ → ℝ) :=
∀ x y z : ℝ, f (x^2 + y^2 * f z) = x * f x + z * f (y^2)

-- Proving that the number of possible values of f(5) is 2
-- and their sum is 5, thus n * s = 10
theorem compute_ns (f : ℝ → ℝ) (hf : functional_equation f) : 2 * 5 = 10 :=
sorry

end compute_ns_l419_419267


namespace sixth_episode_length_correct_l419_419248

-- Define episode lengths and breaks
def episode_lengths : List ℕ := [58, 62, 65, 71, 79]
def break_length : ℕ := 12
def total_length_desired : ℕ := 540

-- Define function to find the length of the sixth episode
def length_of_sixth_episode (total_length_desired : ℕ) (episode_lengths : List ℕ) (break_length : ℕ) : ℕ :=
  total_length_desired - (episode_lengths.sum + episode_lengths.length * break_length)

-- Prove the length of the sixth episode
theorem sixth_episode_length_correct :
  length_of_sixth_episode total_length_desired episode_lengths break_length = 145 := by
  sorry

end sixth_episode_length_correct_l419_419248


namespace jeremy_school_distance_l419_419250

variables (v d : ℝ)

theorem jeremy_school_distance (h1 : 30 / 60 = (1 : ℝ) / 2)
                               (h2 : 18 / 60 = (3 : ℝ) / 10)
                               (h3 : d = v * (1 / 2))
                               (h4 : d = (v + 15) * (3 / 10))
                               : d = 11.25 :=
by
  have : v * (1 / 2) = (v + 15) * (3 / 10), from h4.symm ▸ h3,
  linarith,
  sorry

end jeremy_school_distance_l419_419250


namespace geometric_series_sum_l419_419429

theorem geometric_series_sum :
  (∑ k in Finset.range 101, (5:ℕ) ^ k) = (5 ^ 101 - 1) / 4 :=
by
  -- This is where the proof would go.
  sorry

end geometric_series_sum_l419_419429


namespace sugar_theft_problem_l419_419019

-- Define the statements by Gercoginya and the Cook
def gercoginya_statement := "The cook did not steal the sugar"
def cook_statement := "The sugar was stolen by Gercoginya"

-- Define the thief and truth/lie conditions
def thief_lies (x: String) : Prop := x = "The cook stole the sugar"
def other_truth_or_lie (x y: String) : Prop := x = "The sugar was stolen by Gercoginya" ∨ x = "The sugar was not stolen by Gercoginya"

-- The main proof problem to be solved
theorem sugar_theft_problem : 
  ∃ thief : String, 
    (thief = "cook" ∧ thief_lies gercoginya_statement ∧ other_truth_or_lie cook_statement gercoginya_statement) ∨ 
    (thief = "gercoginya" ∧ thief_lies cook_statement ∧ other_truth_or_lie gercoginya_statement cook_statement) :=
sorry

end sugar_theft_problem_l419_419019


namespace identify_compound_l419_419972

noncomputable def mass_percentage {compound : Type} (el : Type) [HasMass el] [HasComposition compound el] : compound → el → ℝ :=
  by sorry

theorem identify_compound (compound : Type) (O : compound) (mass_percent_O : ℝ) 
  (h: mass_percent_O = 58.33) : 
  ∃ compound : Type, ∀ O : compound, mass_percentage O = mass_percent_O → false :=
  by sorry

end identify_compound_l419_419972


namespace circle_area_isosceles_triangle_l419_419840

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end circle_area_isosceles_triangle_l419_419840


namespace factorial_division_l419_419940

theorem factorial_division :
  (10! / (7! * 2!)) = 360 := by
  sorry

end factorial_division_l419_419940


namespace birth_rate_in_renowned_city_l419_419227

theorem birth_rate_in_renowned_city :
  ∀ (B : ℕ),
  (net_increase_one_day : ℕ) (death_rate : ℕ) (seconds_per_day : ℕ)
  (two_second_intervals : ℕ)
  (net_increase_two_seconds : ℕ),
  net_increase_one_day = 259200 →
  death_rate = 1 →
  seconds_per_day = 86400 →
  two_second_intervals = seconds_per_day / 2 →
  net_increase_two_seconds = net_increase_one_day / two_second_intervals →
  B - death_rate = net_increase_two_seconds →
  B = 7 :=
begin
  sorry
end

end birth_rate_in_renowned_city_l419_419227


namespace general_formula_arithmetic_seq_sum_Tn_formula_l419_419996

variables (S : ℕ → ℝ) (a : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions from the problem
axiom S3_condition : S 3 = 15
axiom a4_a5_condition : a 4 + a 5 = 20

-- Definitions related to arithmetic sequence
def arithmetic_seq (a1 d : ℝ) (n : ℕ) : ℝ := a1 + d * n

-- The target general formula for the arithmetic sequence a_n = 2n + 1
def target_arithmetic_seq (n : ℕ) : ℝ := 2 * n - 1

-- Prove the general formula for a_n
theorem general_formula_arithmetic_seq
  (a1 d n : ℝ)
  (S3_condition : 3 * a1 + (2 * 3 - 3) * d = 15)
  (a4_a5_condition : 2 * a1 + (2 * 4 - 3) * d + (2 * 5 - 3) * d = 20) :
  arithmetic_seq a1 d = target_arithmetic_seq :=
sorry

-- Definition for Tn based on given arithmetic sequence
def target_Tn (n : ℕ) : ℝ := n / (6 * n + 9)

-- Prove the sum T_n = n / (6n + 9)
theorem sum_Tn_formula
  (T : ℕ → ℝ)
  (a : ℕ → ℝ)
  (h_arb_seq : ∀ n, a n = 2 * n - 1)
  (sum_T_condition : ∀ n, T n = (1 / (a n * a (n + 1)))) :
  ∀ n, T n = target_Tn n :=
sorry

end general_formula_arithmetic_seq_sum_Tn_formula_l419_419996


namespace parallelogram_slope_sum_l419_419162

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  (x^2)/4 + y^2 = 1

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

noncomputable def is_parallelogram (O A E B : ℝ × ℝ) : Prop :=
  let (ox, oy) := O in
  let (ax, ay) := A in
  let (ex, ey) := E in
  let (bx, by) := B in
  ex = ox + ax + bx ∧ ey = oy + ay + by

noncomputable def slopes_sum (O A B : ℝ × ℝ) (k1 k2 : ℝ) : Prop :=
  let (ox, oy) := O in
  let (ax, ay) := A in
  let (bx, by) := B in
  (ay - oy) / (ax - ox) = k1 ∧ (by - oy) / (bx - ox) = k2 ∧
  (ay - oy) / (ax - ox) + (by - oy) / (bx - ox) = k1 + k2

theorem parallelogram_slope_sum
  (A B E: ℝ × ℝ)
  (k1 k2 : ℝ) :
  ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧ circle_eq E.1 E.2 ∧
  (is_parallelogram (0, 0) A E B) ∧
  slopes_sum (0, 0) A B k1 k2 ∧
  (let m := sqrt 2/2 in
     (A.2 - B.2) / (A.1 - B.1) = m) →
  k1 + k2 = - (4 * sqrt 2 / 5) := sorry

end parallelogram_slope_sum_l419_419162


namespace degree_of_polynomial_raised_to_power_l419_419342

def polynomial_degree (p : Polynomial ℤ) : ℕ := p.natDegree

theorem degree_of_polynomial_raised_to_power :
  let p : Polynomial ℤ := Polynomial.C 5 * Polynomial.X ^ 3 + Polynomial.C 7
  in polynomial_degree (p ^ 15) = 45 :=
by
  sorry

end degree_of_polynomial_raised_to_power_l419_419342


namespace sqrt_sum_simplification_l419_419794

theorem sqrt_sum_simplification : 
  sqrt (20 - 8 * sqrt 5) + sqrt (20 + 8 * sqrt 5) = 2 * sqrt 14 := 
by
  sorry

end sqrt_sum_simplification_l419_419794


namespace infinite_power_tower_solution_l419_419102

theorem infinite_power_tower_solution (x : ℝ) (y : ℝ) (h1 : y = x ^ x ^ x ^ x ^ x ^ x ^ x ^ x ^ x ^ x) (h2 : y = 4) : x = Real.sqrt 2 :=
by
  sorry

end infinite_power_tower_solution_l419_419102


namespace geometric_sequence_value_of_b_l419_419747

-- Definitions
def is_geometric_sequence (a b c : ℝ) := 
  ∃ r : ℝ, a * r = b ∧ b * r = c

-- Theorem statement
theorem geometric_sequence_value_of_b (b : ℝ) (h : b > 0) 
  (h_seq : is_geometric_sequence 15 b 1) : b = Real.sqrt 15 :=
by
  sorry

end geometric_sequence_value_of_b_l419_419747


namespace area_of_circle_passing_through_vertices_l419_419846

noncomputable def circle_area_through_isosceles_triangle_vertices 
  (a b c : ℝ) (h_isosceles: (a = b) (h_sides: a = 4) (h_base: c = 3) : ℝ :=
π *(√((4^2 - (3/2)^2)/2 + (3/2))^2

theorem area_of_circle_passing_through_vertices :
  circle_area_through_isosceles_triangle_vertices 4 4 3 = 5.6875 * π :=
sorry

end area_of_circle_passing_through_vertices_l419_419846


namespace appropriate_sampling_methods_l419_419339

-- Definitions to represent the conditions
def classes := 15
def chosen_classes := 2

def total_stores := 1500
def ratio_large_medium_small := (1, 5, 9)
def chosen_stores := 15

-- Statement to represent what we need to prove
theorem appropriate_sampling_methods : 
  appropriate_method (classes, chosen_classes) (total_stores, ratio_large_medium_small, chosen_stores) = (simple_random_sampling, stratified_sampling) :=
sorry

end appropriate_sampling_methods_l419_419339


namespace alberto_percentage_correct_l419_419313

-- Conditions as Lean definitions
def first_part_questions : ℕ := 30
def second_part_questions : ℕ := 50
def correct_first_part : ℕ := (70 * first_part_questions) / 100
def correct_second_part : ℕ := (40 * second_part_questions) / 100
def total_correct : ℕ := correct_first_part + correct_second_part
def total_questions : ℕ := first_part_questions + second_part_questions
def percentage_correct : ℚ := (total_correct * 100) / total_questions

-- Statement to be proved
theorem alberto_percentage_correct : percentage_correct ≈ 51 := by
  sorry

end alberto_percentage_correct_l419_419313


namespace no_absolute_winner_l419_419902

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end no_absolute_winner_l419_419902


namespace max_displayed_games_l419_419078

def action_games : ℕ := 73
def adventure_games : ℕ := 51
def simulation_games : ℕ := 39
def action_shelves : ℕ := 2
def action_shelf_capacity : ℕ := 30
def adventure_shelves : ℕ := 1
def adventure_shelf_capacity : ℕ := 45
def simulation_shelves : ℕ := 1
def simulation_shelf_capacity : ℕ := 35
def special_display_requirement : ℕ := 10

theorem max_displayed_games : 
  (action_games - special_display_requirement) <= action_shelves * action_shelf_capacity ∧ 
  (adventure_games - special_display_requirement) <= adventure_shelf_capacity ∧ 
  (simulation_games - special_display_requirement) <= simulation_shelf_capacity →
  ((special_display_requirement * 3) + 
   (min (action_games - special_display_requirement) (action_shelves * action_shelf_capacity)) + 
   (min (adventure_games - special_display_requirement) adventure_shelf_capacity) + 
   (min (simulation_games - special_display_requirement) simulation_shelf_capacity) = 160) :=
begin
  intros h,
  sorry
end

end max_displayed_games_l419_419078


namespace find_initial_tax_rate_l419_419406

-- Define the conditions
def annualIncome : ℝ := 45000
def newTaxRate : ℝ := 33 / 100
def differentialSavings : ℝ := 3150

-- Define the initial tax rate to be determined
def initialTaxRate := 40 / 100

-- The proof problem statement
theorem find_initial_tax_rate :
  ∃ R : ℝ, (∃ (R_percentage : ℝ), R = R_percentage / 100 ∧ annualIncome * R - annualIncome * newTaxRate = differentialSavings) ∧ R = initialTaxRate / 100
:= by
  sorry

end find_initial_tax_rate_l419_419406


namespace distance_between_red_lights_l419_419740

theorem distance_between_red_lights : 
  ∀ (distance_between_lights inches_per_foot : ℕ), 
    distance_between_lights = 8 → 
    inches_per_foot = 12 → 
  let positions (n : ℕ) := 5 * n + (n % 3) + 1 in
  let distance_in_feet (pos1 pos2 : ℕ) := ((pos2 - pos1) * distance_between_lights) / inches_per_foot in
  distance_in_feet (positions 4) (positions 20) = 56 :=
by
  intro distance_between_lights inches_per_foot h1 h2
  let positions := λ n, 5 * n + (n % 3) + 1
  let distance_in_feet := λ pos1 pos2, ((pos2 - pos1) * distance_between_lights) / inches_per_foot
  have h_positions_5th_red := positions 4 -- Position of 5th red light
  have h_positions_25th_red := positions 24 -- Position of 25th red light
  have h_distance_between := (distance_between_lights * (h_positions_25th_red - h_positions_5th_red)) / 12
  exact h_distance_between

#print axioms distance_between_red_lights -- No axioms here

end distance_between_red_lights_l419_419740


namespace distance_is_twenty_cm_l419_419202

noncomputable def distance_between_pictures_and_board (picture_width: ℕ) (board_width_m: ℕ) (board_width_cm: ℕ) (number_of_pictures: ℕ) : ℕ :=
  let board_total_width := board_width_m * 100 + board_width_cm
  let total_pictures_width := number_of_pictures * picture_width
  let total_distance := board_total_width - total_pictures_width
  let total_gaps := number_of_pictures + 1
  total_distance / total_gaps

theorem distance_is_twenty_cm :
  distance_between_pictures_and_board 30 3 20 6 = 20 :=
by
  sorry

end distance_is_twenty_cm_l419_419202


namespace ship_chasing_distance_l419_419722

/--
Ship \( P \) spots ship \( Q \), which is moving in a direction perpendicular to \( PQ \),
maintaining its course. Ship \( P \) chases \( Q \), always heading directly towards \( Q \);
the speed of both ships at any moment is the same (but can change over time). Given that
initially the distance \( PQ \) was 10 nautical miles, prove that the distance between \( P \)
and \( Q \) will eventually be 5 nautical miles.
-/
theorem ship_chasing_distance (initial_distance : ℝ) (h_perpendicular : ∀ t : ℝ, t ≥ 0 → ⟪P(t) - Q(t), PQ(t)⟫ = 0)
  (h_chasing : ∀ t : ℝ, t ≥ 0 → P(t) ∈ line_through Q(t) (PQ(t)))
  (h_same_speed : ∀ t : ℝ, t ≥ 0 → speed_P(t) = speed_Q(t)) (h_initial_distance : initial_distance = 10) :
  ∃ t : ℝ, distance (P(t)) (Q(t)) = 5 :=
begin
  sorry
end

end ship_chasing_distance_l419_419722


namespace find_x_when_y_equals_two_l419_419624

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end find_x_when_y_equals_two_l419_419624


namespace product_of_sequence_equals_8_over_15_l419_419348

theorem product_of_sequence_equals_8_over_15 :
  (∏ k in Finset.range 14 + 2, (1 - (1 / (k: ℝ) ^ 2)) = (8 / 15)) :=
by
  sorry

end product_of_sequence_equals_8_over_15_l419_419348


namespace find_a_Sn_equals_lambda_range_l419_419182

noncomputable def f (x a : ℝ) : ℝ := log 2 ((sqrt 2 * x) / (a - x))

def point (x y : ℝ) : Prop := f x y = y

def A := (0.5, 0.5)

theorem find_a : (f (0.5) a = 0.5) ∧ (∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < 1) → (0 < x2 ∧ x2 < 1) → (f x1 a + f x2 a = 1) → (x1 + x2 = 1)) → a = 1 :=
sorry

theorem Sn_equals : (∀ x a, f x a = x → a = 1) ∧ (∀ n : ℕ, 1 < n → S n = f (1 / n) 1 + f (2 / n) 1 + ... + f ((n - 1) / n) 1) → S n = (n - 1) / 2 :=
sorry

theorem lambda_range : a1 = 2 / 3 ∧ (∀ n : ℕ, n > 0 → 1 / an = (Sn + 1) * (Sn_plus1 + 1)) ∧
(Tn := sum_of_seq a1 n) ∧ (∀ n : ℕ, n > 0 → Tn < λ * (Sn_plus1 + 1)) → λ > 1 / 2 :=
sorry

end find_a_Sn_equals_lambda_range_l419_419182


namespace no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime_l419_419471

theorem no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime :
  ¬∃ n : ℕ, 2 ≤ n ∧ Nat.Prime (n^4 + n^2 + 1) :=
sorry

end no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime_l419_419471


namespace pipe_c_empty_time_l419_419334

theorem pipe_c_empty_time (x : ℝ) :
  (4/20 + 4/30 + 4/x) * 3 = 1 → x = 6 :=
by
  sorry

end pipe_c_empty_time_l419_419334


namespace no_absolute_winner_l419_419904

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end no_absolute_winner_l419_419904


namespace transform_function_l419_419138

noncomputable def transformedFunction (x : ℝ) : ℝ := 2 * cos (4 * x)

theorem transform_function :
  ∀ (x : ℝ),
    let originalFunction := 2 * sin (2 * x + π / 6)
    let shiftedFunction := 2 * sin (2 * (x + π / 6) + π / 6)
    let stretchedFunction := 2 * cos (2 * (x + π / 6))
    let halfLengthFunction := 2 * cos (4 * x)
    halfLengthFunction = transformedFunction x := by
  intros
  have h1: 2 * sin (2 * (x + π / 6) + π / 6) = 2 * cos (2 * x) := sorry
  have h2: 2 * cos (2 * x) = 2 * cos (4 * x) := sorry
  rw [h1, h2]
  refl

end transform_function_l419_419138


namespace find_current_time_l419_419246

open Real

/-- Define the variables and conditions -/
noncomputable def currentTimeInMinutesAfter10 (t : ℝ) : Prop :=
  let hour_hand_pos_four_minutes_ago := 28 + 0.5 * t
  let minute_hand_pos_eight_minutes_later := 6 * (t + 8)
  abs (minute_hand_pos_eight_minutes_later - hour_hand_pos_four_minutes_ago) = 180

/-- Define the proof problem -/
theorem find_current_time : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 60 ∧ currentTimeInMinutesAfter10 t ∧ t ≈ 29.09 :=
sorry

end find_current_time_l419_419246


namespace quadratic_polynomial_fourth_power_l419_419756

theorem quadratic_polynomial_fourth_power {a b c : ℤ} (h : ∀ x : ℤ, ∃ k : ℤ, ax^2 + bx + c = k^4) : a = 0 ∧ b = 0 :=
sorry

end quadratic_polynomial_fourth_power_l419_419756


namespace geom_solid_area_volume_l419_419482

-- Define the concepts in Lean 4
variables {AB BC AD : ℝ}
variables (h1 : AB = 1) (h2 : BC = 2) (h3 : AD = 2) 
variables (h4 : BC = 2 * AB) (h5 : AD = BC) (h6 : AB ⟂ BC) (h7 : BC ∥ AD)

noncomputable def surface_area : ℝ := (5 + 3 * real.sqrt 2) * real.pi
noncomputable def volume : ℝ := (7 / 3) * real.pi

-- The proof problem in Lean 4 statement
theorem geom_solid_area_volume :
  let geometric_solid := rotate_trapezoid_around_AB AB BC AD in
  (surface_area geometric_solid = (5 + 3 * real.sqrt 2) * real.pi) ∧ 
  (volume geometric_solid = (7 / 3) * real.pi) :=
sorry

end geom_solid_area_volume_l419_419482


namespace circle_area_isosceles_triangle_l419_419838

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end circle_area_isosceles_triangle_l419_419838


namespace smallest_coconuts_l419_419978

/-
  Define the conditions and the proof goal.
  The formalization captures the essence of the problem conditions and the required proof.
-/

def is_valid_initial_coconuts (n : ℕ) : Prop :=
  -- First sailor's action
  let n1 := (4 * (n - 1)) / 5 in
  -- Second sailor's action
  let n2 := (4 * (n1 - 1)) / 5 in
  -- Third sailor's action
  let n3 := (4 * (n2 - 1)) / 5 in
  -- Fourth sailor's action
  let n4 := (4 * (n3 - 1)) / 5 in
  -- Fifth sailor's action
  let n5 := (4 * (n4 - 1)) / 5 in
  -- In the morning, the remaining coconuts should be divisible by 5
  n5 % 5 = 0

theorem smallest_coconuts : ∃ n : ℕ, is_valid_initial_coconuts n ∧ n = 3121 :=
by
  -- Declare a proof search here (this is normally where you would provide the proof)
  sorry

end smallest_coconuts_l419_419978


namespace minimum_value_l419_419463

open Real

def f (x : ℝ) : ℝ := x / exp(x)

theorem minimum_value : ∃ x ∈ Icc (2:ℝ) 4, ∀ y ∈ Icc (2:ℝ) 4, f(x) ≤ f(y) ∧ f(x) = 2 / exp(2) :=
by
  sorry

end minimum_value_l419_419463


namespace more_males_l419_419085

theorem more_males {Total_attendees Male_attendees : ℕ} (h1 : Total_attendees = 120) (h2 : Male_attendees = 62) :
  Male_attendees - (Total_attendees - Male_attendees) = 4 :=
by
  sorry

end more_males_l419_419085


namespace javier_attractions_l419_419249

theorem javier_attractions (A B C D E F : Type) :
  let attractions := [A, B, C, D, E, F]
  let condition := [F, E] ∈ [attractions.take 5, attractions.drop 5]
  ∃ σ : [A, B, C, D, E, F] → [A, B, C, D, E, F], 
    (∀ (i : Fin 5), σ (attractions.nth i) ∈ attractions) ∧
    ((σ (attractions.nth 5) = F ∧ σ (attractions.nth 4) = E) 
    ∨ (σ (attractions.nth 5) = E ∧ σ (attractions.nth 4) = F)) ∧
    (σ (attractions.take 4).permutations.card * 2 = 240) :=
  sorry

end javier_attractions_l419_419249


namespace length_of_platform_l419_419032

theorem length_of_platform 
  (speed_train_kmph : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_man : ℝ)
  (conversion_factor : ℝ)
  (speed_train_mps : ℝ)
  (length_train : ℝ)
  (total_distance : ℝ)
  (length_platform : ℝ) :
  speed_train_kmph = 150 →
  time_cross_platform = 45 →
  time_cross_man = 20 →
  conversion_factor = (1000 / 3600) →
  speed_train_mps = speed_train_kmph * conversion_factor →
  length_train = speed_train_mps * time_cross_man →
  total_distance = speed_train_mps * time_cross_platform →
  length_platform = total_distance - length_train →
  length_platform = 1041.75 :=
by sorry

end length_of_platform_l419_419032


namespace rohan_coconut_farm_size_l419_419719

theorem rohan_coconut_farm_size 
  (coconut_trees_per_sqm : ℕ)
  (coconuts_per_tree : ℕ)
  (harvest_months : ℕ)
  (coconut_cost : ℝ)
  (total_earnings : ℝ)
  (harvests_in_6_months : ℕ)
  (earnings_after_6_months : ℝ) :
  (coconut_trees_per_sqm = 2) →
  (coconuts_per_tree = 6) →
  (harvest_months = 3) →
  (coconut_cost = 0.5) →
  (total_earnings = 240) →
  (harvests_in_6_months = 6 / harvest_months) →
  (earnings_after_6_months = coconut_trees_per_sqm * coconuts_per_tree * coconut_cost * harvests_in_6_months) →
  (total_earnings / earnings_after_6_months = 20) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  unfold coconut_trees_per_sqm coconuts_per_tree harvest_months coconut_cost total_earnings harvests_in_6_months earnings_after_6_months
  sorry

end rohan_coconut_farm_size_l419_419719


namespace find_a_l419_419439

def F (a b c : ℕ) : ℕ := a * b^3 + c^2

theorem find_a :
  let a := (5 : ℚ) / 19
  F a 3 2 = F a 2 3 :=
by 
  sorry

end find_a_l419_419439


namespace octagon_diagonals_l419_419533

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l419_419533


namespace total_cost_is_correct_l419_419071

def cost_price (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage / 100)

theorem total_cost_is_correct :
  let cp1 := cost_price 120 25 in
  let cp2 := cost_price 225 40 in
  let cp3 := cost_price 450 20 in
  cp1 + cp2 + cp3 = 631.71 := 
by
  have cp1 : ℝ := cost_price 120 25
  have cp2 : ℝ := cost_price 225 40
  have cp3 : ℝ := cost_price 450 20
  have total_cost : ℝ := cp1 + cp2 + cp3
  show 631.71,
  sorry

end total_cost_is_correct_l419_419071


namespace original_engineers_from_university_A_fraction_l419_419422

-- Definitions and conditions
variables (X : ℕ) -- Number of original network engineers from University A
def new_hires := 8
def original_engineers := 20
def total_engineers := 28
def university_A_fraction := 0.75
def total_university_A := university_A_fraction * total_engineers

-- Statement of the problem
theorem original_engineers_from_university_A_fraction
  (h1 : total_university_A = X + new_hires) : 
    X = 13 ∧ (X / original_engineers : ℚ) = 13 / 20 :=
by {
  sorry
}

end original_engineers_from_university_A_fraction_l419_419422


namespace actual_cost_of_article_l419_419014

theorem actual_cost_of_article (x : ℝ) (h : 0.76 * x = 760) : x = 1000 :=
by 
  sorry

end actual_cost_of_article_l419_419014


namespace quadrilateral_perimeter_l419_419310

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2))

def PQR_perimeter (PQ QR PR : ℝ) : ℝ :=
  PQ + QR + PR

def PRS_perimeter (S : ℝ) (scaling_factor : ℝ) (PQR_perim : ℝ) : ℝ :=
  scaling_factor * PQR_perim

theorem quadrilateral_perimeter (PQ QR : ℝ) (PR_angle : ℝ) (scaling_factor : ℝ) :
  let PR := distance 0 0 PQ QR,
  let perimeter_PQR := PQR_perimeter PQ QR PR,
  let perimeter_PRS := PRS_perimeter 5 (5/3) perimeter_PQR,
  let PQRS_perimeter := perimeter_PQR + perimeter_PRS - PR in
  PQRS_perimeter = 22 :=
by
  let PR := distance 0 0 PQ QR,
  let perimeter_PQR := PQR_perimeter PQ QR PR,
  let perimeter_PRS := PRS_perimeter 5 (5/3) perimeter_PQR,
  let PQRS_perimeter := perimeter_PQR + perimeter_PRS - PR,
  have h1 : PQRS_perimeter = 22 := by sorry,
  exact h1

end quadrilateral_perimeter_l419_419310


namespace find_B_investment_l419_419812

def A_investment : ℝ := 24000
def C_investment : ℝ := 36000
def C_profit : ℝ := 36000
def total_profit : ℝ := 92000
def B_investment := 32000

theorem find_B_investment (B_investment_unknown : ℝ) :
  (C_investment / C_profit) = ((A_investment + B_investment_unknown + C_investment) / total_profit) →
  B_investment_unknown = B_investment := 
by 
  -- Mathematical equivalence to the given problem
  -- Proof omitted since only the statement is required
  sorry

end find_B_investment_l419_419812


namespace minimum_sum_of_distances_is_two_l419_419190

-- Definitions for the given lines and parabola
def line1 (x y : ℝ) : Prop := 4 * x - 3 * y + 6 = 0
def line2 (x : ℝ) : Prop := x = -1
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Distance functions
def distance_to_line1 (x y : ℝ) : ℝ := (abs (4 * x - 3 * y + 6)) / 5
def distance_to_line2 (x : ℝ) : ℝ := abs (x + 1)

-- Sum of distances
def sum_of_distances (a : ℝ) : ℝ :=
  let x := a^2
  let y := 2 * a
  distance_to_line1 x y + distance_to_line2 x

-- Prove that the minimum sum of distances is 2
theorem minimum_sum_of_distances_is_two : (∀ a : ℝ, parabola (a^2) (2 * a)) →  ∃ a, sum_of_distances a = 2 := by
  sorry

end minimum_sum_of_distances_is_two_l419_419190


namespace range_of_k_for_monotonic_increasing_l419_419217

noncomputable def f (x k : ℝ) := exp(x) + k * x - log(x)
noncomputable def g (x : ℝ) := (1 / x) - exp(x)

theorem range_of_k_for_monotonic_increasing (k : ℝ) :
  (∀ x > 1, exp(x) + k - (1 / x) ≥ 0) ↔ k ≥ 1 - exp(1) := by
  sorry

end range_of_k_for_monotonic_increasing_l419_419217


namespace find_breadth_of_cuboid_l419_419133

/-- Given a cuboid with surface area 700 square meters,
    length 12 meters and height 7 meters, the breadth is 14 meters. -/
theorem find_breadth_of_cuboid (SA l h : ℝ) (h_SA : SA = 700) (h_l : l = 12) (h_h : h = 7) :
  ∃ w : ℝ, 2 * (l * w + l * h + h * w) = SA ∧ w = 14 :=
by
  use 14
  split
  · sorry
  · refl

end find_breadth_of_cuboid_l419_419133


namespace sara_oil_usage_l419_419720

theorem sara_oil_usage:
  let Ron_oil := (3 / 8 : ℚ)
  let Sara_fraction := (5 / 6 : ℚ)
  let Sara_oil_usage := Sara_fraction * Ron_oil
  Sara_oil_usage = (5 / 16 : ℚ) :=
begin
  sorry
end

end sara_oil_usage_l419_419720


namespace problem_equivalence_l419_419235

def curve_C_parametric (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos α, sin α)

def line_l_polar (ρ θ : ℝ) : Prop :=
  (sqrt 2 / 2) * ρ * cos (θ + π / 4) = -1

def line_l_cartesian (x y : ℝ) : Prop :=
  x - y + 2 = 0

theorem problem_equivalence (M A B : ℝ × ℝ) (α t : ℝ) :
  (curve_C_parametric α = (x, y)) →
  (line_l_polar ρ θ) →
  (M = (-1, 0)) →
  (x = -1 + (sqrt 2 / 2) * t) →
  (y = (sqrt 2 / 2) * t) →
  (frac x^2 / 3 + y^2 = 1) →
  (line_l_cartesian x y) →
  (let (t₁, t₂) := (roots_of_quadratic (2, -sqrt 2, -2)) in t₁ * t₂ = -1) →
  (|dist M A| * |dist M B| = 1) :=
sorry

end problem_equivalence_l419_419235


namespace cannot_be_sum_of_six_consecutive_odd_integers_l419_419352

theorem cannot_be_sum_of_six_consecutive_odd_integers (S : ℕ) :
  (S = 90 ∨ S = 150) ->
  ∀ n : ℤ, ¬(S = n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10)) :=
by
  intro h
  intro n
  cases h
  case inl => 
    sorry
  case inr => 
    sorry

end cannot_be_sum_of_six_consecutive_odd_integers_l419_419352


namespace polynomial_min_k_eq_l419_419098

theorem polynomial_min_k_eq {k : ℝ} :
  (∃ k : ℝ, ∀ x y : ℝ, 9 * x^2 - 12 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 9 * y + 12 >= 0)
  ↔ k = (Real.sqrt 3) / 4 :=
sorry

end polynomial_min_k_eq_l419_419098


namespace number_of_diagonals_in_octagon_l419_419526

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l419_419526


namespace instantaneous_velocity_at_t2_l419_419395

noncomputable def s (t : ℝ) : ℝ := t^3 - t^2 + 2 * t

theorem instantaneous_velocity_at_t2 : 
  deriv s 2 = 10 := 
by
  sorry

end instantaneous_velocity_at_t2_l419_419395


namespace polynomial_degree_ge_prime_minus_one_l419_419277

theorem polynomial_degree_ge_prime_minus_one 
  (p : ℕ) [hp : Fact (Nat.Prime p)]
  (f : Polynomial ℤ)
  (hf0 : f.coeff 0 = 0) -- f(0) = 0
  (hf1 : f.coeff 1 = 1) -- f(1) = 1
  (hfn : ∀ n : ℕ, f.eval n % ↑p = 0 ∨ f.eval n % ↑p = 1)  -- for every positive integer n, f(n) mod p is 0 or 1
  : f.natDegree ≥ p-1 :=
by
  sorry

end polynomial_degree_ge_prime_minus_one_l419_419277


namespace eval_complex_product_l419_419122

def z1 : ℂ := 3 * Real.sqrt 5 - 5 * Complex.i
def z2 : ℂ := 2 * Real.sqrt 2 + 4 * Complex.i

theorem eval_complex_product :
  abs (z1 * z2) = 8 * Real.sqrt 105 :=
by
  sorry

end eval_complex_product_l419_419122


namespace complex_numbers_max_bound_l419_419492

open Complex

theorem complex_numbers_max_bound (z1 z2 z3 : ℂ)
  (h : {abs (z1 + z2 + z3), abs (-z1 + z2 + z3), abs (z1 - z2 + z3), abs (z1 + z2 - z3)} = {98, 84, 42, 28}) :
  max (abs (z1^2 * (2 - z2^2) - z3^2)) (max (abs (z2^2 * (2 - z3^2) - z1^2)) (abs (2 * z3^2 * (z1^2 + 1) - z2^2))) ≥ 2016 :=
by
  sorry

end complex_numbers_max_bound_l419_419492


namespace octagon_diagonals_l419_419534

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l419_419534


namespace find_divisor_l419_419361

theorem find_divisor (remainder quotient dividend divisor : ℕ) 
  (h_rem : remainder = 8)
  (h_quot : quotient = 43)
  (h_div : dividend = 997)
  (h_eq : dividend = divisor * quotient + remainder) : 
  divisor = 23 :=
by
  sorry

end find_divisor_l419_419361


namespace weekly_pizza_dough_batches_l419_419427

open Nat

def daily_flour_usage (day: String) : ℕ × ℕ × ℕ :=
  match day with
  | "Monday"    => (4, 3, 2)
  | "Tuesday"   => (6, 2, 1)
  | "Wednesday" => (5, 1, 2)
  | "Thursday"  => (3, 4, 3)
  | "Friday"    => (7, 1, 2)
  | "Saturday"  => (5, 3, 1)
  | "Sunday"    => (2, 4, 2)
  | _ => (0, 0, 0)

def weekly_flour_usages : List (ℕ × ℕ × ℕ) :=
  List.map daily_flour_usage ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def total_regular_dough_batches (weekly_usages : List (ℕ × ℕ × ℕ)) : ℕ :=
  weekly_usages.foldr (λ (usage : ℕ × ℕ × ℕ) acc => acc + 15 * fst usage) 0

def total_glutenfree_dough_batches (weekly_usages : List (ℕ × ℕ × ℕ)) : ℕ :=
  weekly_usages.foldr (λ (usage : ℕ × ℕ × ℕ) acc => acc + 10 * (snd usage).fst) 0

def total_wholewheat_dough_batches (weekly_usages : List (ℕ × ℕ × ℕ)) : ℕ :=
  weekly_usages.foldr 
    (λ (usage : ℕ × ℕ × ℕ) acc => 
     let regular_sacks := usage.1
     let raw_wholewheat_sacks := usage.2.1
     let converted_wholewheat_sacks := regular_sacks / 2
     acc + 12 * (raw_wholewheat_sacks + converted_wholewheat_sacks)
    ) 0

theorem weekly_pizza_dough_batches :
  total_regular_dough_batches weekly_flour_usages = 480 ∧
  total_glutenfree_dough_batches weekly_flour_usages = 180 ∧
  total_wholewheat_dough_batches weekly_flour_usages = 186 :=
  sorry

end weekly_pizza_dough_batches_l419_419427


namespace vector_magnitude_proof_l419_419493

noncomputable def vec_a : ℝ × ℝ := (real.sqrt 3, 1)
noncomputable def vec_b_magnitude : ℝ := 1
noncomputable def angle_between_ab : ℝ := real.pi / 3  -- 60 degrees in radians

theorem vector_magnitude_proof :
  let b := (1 / vec_b_magnitude, 0) in  -- Dummy placeholder for vector b with magnitude 1
  let vec_b := (real.cos angle_between_ab * vec_b_magnitude,
                            real.sin angle_between_ab * vec_b_magnitude) in
  | (prod.fst vec_a + 2 * prod.fst vec_b)^2 + (prod.snd vec_a + 2 * prod.snd vec_b)^2 |
  = 2 * real.sqrt 3 :=
by
  sorry

end vector_magnitude_proof_l419_419493


namespace mr_evans_mistake_l419_419287

variable {N : ℕ} (a : Fin (2 * N) → ℝ)

theorem mr_evans_mistake
  (h1 : ∑ i, a i = 116 * N) 
  (h2 : (a (N - 1) + a N) / 2 = 80)
  (h3 : a (2 * N - 1) - a 0 = 40) : False := by
  sorry

end mr_evans_mistake_l419_419287


namespace num_diagonals_octagon_l419_419569

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l419_419569


namespace factorize_expression_l419_419961

variable (a : ℝ) (b : ℝ)

theorem factorize_expression : 2 * a - 8 * a * b^2 = 2 * a * (1 - 2 * b) * (1 + 2 * b) := by
  sorry

end factorize_expression_l419_419961


namespace sum_linear_sequence_l419_419929

def is_linear_function (f : ℕ → ℕ) : Prop :=
  ∃ k b : ℕ, f = λ x, k * x + b

theorem sum_linear_sequence (f : ℕ → ℕ) (n : ℕ)
  (hf_lin : is_linear_function f)
  (hf0 : f 0 = 1)
  (hf_geom : ∃ r : ℕ, f 1 * r = f 4 ∧ f 4 * r = f 13) :
  (Finset.range n).sum (λ i, f (2 * i + 2)) = n * (2 * n + 3) := by
  sorry

end sum_linear_sequence_l419_419929


namespace oranges_in_bin_l419_419070

theorem oranges_in_bin (initial_oranges : ℕ) (percent_thrown : ℝ) (factor_new : ℝ) 
  (h_initial : initial_oranges = 50) (h_percent : percent_thrown = 0.20) (h_factor : factor_new = 1.5) :
  (initial_oranges - (initial_oranges * percent_thrown).toNat) + ((initial_oranges - (initial_oranges * percent_thrown).toNat) * factor_new).toNat = 100 :=
by
  sorry

end oranges_in_bin_l419_419070


namespace area_of_circumcircle_of_isosceles_triangle_l419_419833

theorem area_of_circumcircle_of_isosceles_triangle :
  let AB := 4
  let AC := 4
  let BC := 3
  let AD := (√(AB^2 - (BC/2)^2))
  let radius := AD
  let area := π * radius^2
  area = 16 * π :=
by
  sorry

end area_of_circumcircle_of_isosceles_triangle_l419_419833


namespace num_ordered_pairs_c_d_l419_419103

def is_solution (c d x y : ℤ) : Prop :=
  c * x + d * y = 2 ∧ x^2 + y^2 = 65

theorem num_ordered_pairs_c_d : 
  ∃ (S : Finset (ℤ × ℤ)), S.card = 136 ∧ 
  ∀ (c d : ℤ), (c, d) ∈ S ↔ ∃ (x y : ℤ), is_solution c d x y :=
sorry

end num_ordered_pairs_c_d_l419_419103


namespace mr_a_net_gain_l419_419707

theorem mr_a_net_gain 
  (initial_value : ℝ)
  (sale_profit_percentage : ℝ)
  (buyback_loss_percentage : ℝ)
  (final_sale_price : ℝ) 
  (buyback_price : ℝ)
  (net_gain : ℝ) :
  initial_value = 12000 →
  sale_profit_percentage = 0.15 →
  buyback_loss_percentage = 0.12 →
  final_sale_price = initial_value * (1 + sale_profit_percentage) →
  buyback_price = final_sale_price * (1 - buyback_loss_percentage) →
  net_gain = final_sale_price - buyback_price →
  net_gain = 1656 :=
by
  sorry

end mr_a_net_gain_l419_419707


namespace trajectory_of_point_P_l419_419196

noncomputable def F1 : ℝ × ℝ := (-5, 0)
noncomputable def F2 : ℝ × ℝ := (5, 0)
def dist (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem trajectory_of_point_P (P : ℝ × ℝ) (a : ℝ) (ha1 : a = 3) (ha2 : a = 5) :
  (dist P F1 - dist P F2 = 6 → true) ∧ (dist P F1 - dist P F2 = 10 → true) :=
by
  sorry

end trajectory_of_point_P_l419_419196


namespace number_of_diagonals_in_octagon_l419_419528

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l419_419528


namespace length_of_bridge_l419_419072

theorem length_of_bridge (L_t : ℝ) (t1 t2 : ℝ) (V : ℝ) (L_b : ℝ) (hL_t : L_t = 833.33)
  (ht1 : t1 = 30) (ht2 : t2 = 120) (hV : V = L_t / t1) (hL_b : L_b = 2500) :
  L_b = ((V * t2) - L_t) :=
by
  have V : V = L_t / t1 := by exact hV
  have proof := calc
    L_b = 833.33 * 3 : by sorry -- This represents the structured steps but deferred for now
  exact hL_b -- Conclude the theorem by the provided length of the bridge.

end length_of_bridge_l419_419072


namespace complex_symmetry_l419_419658

theorem complex_symmetry (z : ℂ) (h : z = 1 - (1 - z)) : z = 1 - I :=
by
  have h1 : 1 - (1 - I) = I := by simp
  rw h1 at h
  rw sub_self at h
  assumption

end complex_symmetry_l419_419658


namespace max_a_such_that_f_geq_a_min_value_under_constraint_l419_419020

-- Problem (1)
theorem max_a_such_that_f_geq_a :
  ∃ (a : ℝ), (∀ (x : ℝ), |x - (5/2)| + |x - a| ≥ a) ∧ a = 5 / 4 := sorry

-- Problem (2)
theorem min_value_under_constraint :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2 * y + 3 * z = 1 ∧
  (3 / x + 2 / y + 1 / z) = 16 + 8 * Real.sqrt 3 := sorry

end max_a_such_that_f_geq_a_min_value_under_constraint_l419_419020


namespace alternate_multiple_exists_l419_419779

-- Define what it means for a number to have alternating digits
def is_alternate_digits (n : ℕ) : Prop :=
  ∃ k, k > 0 ∧ 
  let digits := (to_digits 10 n) in 
  (∀ i : ℕ, i < digits.length → (i % 2 = 0 → digits.get i % 2 = 1) ∧ (i % 2 = 1 → digits.get i % 2 = 0))

-- The statement we need to prove
theorem alternate_multiple_exists (n : ℕ) : ¬ (∃ m : ℕ, m > 0 ∧ n = 20 * m) → ∃ k: ℕ, k > 0 ∧ n ∣ k ∧ is_alternate_digits k :=
by
  sorry

end alternate_multiple_exists_l419_419779


namespace values_with_max_occurrences_l419_419415

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 1)^2 / (x * (x^2 - 1))

theorem values_with_max_occurrences : 
 (set_of (λ a: ℝ, ∃ s: finset ℝ, (∀ x ∈ s, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) ∧ 
                  s.card = 2 ∧ ∀ x ∈ ({x : ℝ | f x = a} : set ℝ), x ∈ s)) = 
 {[a : ℝ | a < -4].union {[a : ℝ | a > 4]]} :=
sorry

end values_with_max_occurrences_l419_419415


namespace sum_of_sequence_l419_419351

-- Defining the conditions as given in the problem
def S_n (n : ℕ) : ℝ := (1 - (n+1) * 1^n + n * 1^(n+1)) / (1 - 1)^2

theorem sum_of_sequence (n : ℕ) (h1 : n ≥ 4) : S_n n = n * (n + 3) * 2^(n-2) :=
sorry

end sum_of_sequence_l419_419351


namespace paige_bouquets_l419_419368

theorem paige_bouquets (total_flowers wilted_flowers flowers_per_bouquet : ℕ) (h1 : total_flowers = 53) (h2 : wilted_flowers = 18) (h3 : flowers_per_bouquet = 7) :
  (total_flowers - wilted_flowers) / flowers_per_bouquet = 5 :=
by
  simp [h1, h2, h3]
  norm_num
  sorry

end paige_bouquets_l419_419368


namespace number_of_diagonals_in_octagon_l419_419531

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l419_419531


namespace probability_of_closer_to_D_in_triangle_DEF_l419_419242

noncomputable def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

theorem probability_of_closer_to_D_in_triangle_DEF :
  let D := (0, 0)
  let E := (0, 6)
  let F := (8, 0)
  let M := ((D.1 + F.1) / 2, (D.2 + F.2) / 2)
  let N := ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  let area_DEF := triangle_area D E F
  let area_DMN := triangle_area D M N
  area_DMN / area_DEF = 1 / 4 := by
    sorry

end probability_of_closer_to_D_in_triangle_DEF_l419_419242


namespace complex_magnitude_product_l419_419116

noncomputable def z1 : ℂ := 3 * Real.sqrt 5 - 5 * Complex.i
noncomputable def z2 : ℂ := 2 * Real.sqrt 2 + 4 * Complex.i
noncomputable def magnitude (z : ℂ) : ℝ := Complex.abs z

theorem complex_magnitude_product :
  magnitude (z1 * z2) = 12 * Real.sqrt 35 :=
by
  have z1 := z1
  have z2 := z2
  sorry

end complex_magnitude_product_l419_419116


namespace number_of_boys_is_fifty_two_l419_419015

theorem number_of_boys_is_fifty_two
  (circle : list ℕ)
  (h1 : ∀ i, i ∈ circle → i > 0)
  (h2 : ∃ n, length circle = n)
  (h3 : ∀ (n : ℕ), 1 ≤ n → n ≤ length circle 
         → ((circle.get ⟨7, by linarith⟩) = ((circle.get ⟨32, by linarith⟩) % (length circle))))
  : length circle = 52 := by
  sorry

end number_of_boys_is_fifty_two_l419_419015


namespace octagon_diagonals_20_l419_419550

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l419_419550


namespace teacher_student_relation_l419_419648

theorem teacher_student_relation
  (m n k ℓ : ℕ)
  (H1 : ∀ t : ℕ, t ≤ m)
  (H2 : ∀ s : ℕ, s ≤ n)
  (H3 : ∀ t : ℕ, ∃ S : set ℕ, S.card = k ∧ ∀ s ∈ S, s ≤ n)
  (H4 : ∀ s1 s2 : ℕ, s1 ≠ s2 → ∃ T : set ℕ, T.card = ℓ ∧ ∀ t ∈ T, t ≤ m) :
  m * k * (k - 1) = ℓ * n * (n - 1) :=
sorry

end teacher_student_relation_l419_419648


namespace num_necklaces_5_diamonds_l419_419892

-- Define the total number of necklaces
def total_necklaces : ℕ := 20

-- Define the total number of diamonds
def total_diamonds : ℕ := 79

-- Define the number of necklaces with 2 diamonds
variable (x : ℕ)

-- Define the number of necklaces with 5 diamonds
variable (y : ℕ)

-- State the condition for the total number of necklaces
def condition1 : Prop := x + y = total_necklaces

-- State the condition for the number of diamonds
def condition2 : Prop := 2 * x + 5 * y = total_diamonds

-- Define the main theorem to prove
theorem num_necklaces_5_diamonds : condition1 ∧ condition2 → y = 13 := by
  sorry

end num_necklaces_5_diamonds_l419_419892


namespace find_x_l419_419458

theorem find_x :
  (∃ x : ℝ, real.cbrt (5 - x/3) = -4) → x = 207 :=
by
  intro hx
  rcases hx with ⟨x, H⟩
  sorry

end find_x_l419_419458


namespace max_value_of_M_over_AB_l419_419488

theorem max_value_of_M_over_AB 
  (A B C : ℝ)
  (h_angles_sum : A + B + C = π)
  (h_alpha : ℝ × ℝ)
  (h_alpha_def : h_alpha = (cos ((A - B) / 2), sqrt 3 * sin ((A + B) / 2)))
  (h_alpha_norm : |h_alpha| = sqrt 2)
  (h_arithmetic_sequence : ∃ M : ℝ × ℝ, (M.x, M.y, |M - ⟨A, B⟩|) forms_arithmetic_seq) :
  ∀ M : ℝ × ℝ, angle C is_maximized M → 
  (|M - ⟨C⟩| / |⟨A, B⟩|) = (2 * sqrt 3 + sqrt 2) / 4 :=
sorry

end max_value_of_M_over_AB_l419_419488


namespace fred_added_nine_l419_419301

def onions_in_basket (initial_onions : ℕ) (added_by_sara : ℕ) (taken_by_sally : ℕ) (added_by_fred : ℕ) : ℕ :=
  initial_onions + added_by_sara - taken_by_sally + added_by_fred

theorem fred_added_nine : ∀ (S F : ℕ), onions_in_basket S 4 5 F = S + 8 → F = 9 :=
by
  intros S F h
  sorry

end fred_added_nine_l419_419301


namespace new_cube_volume_l419_419357

-- Definitions of the conditions
def original_volume := 216
def scale_factor := 2

-- Given that the original cube's volume is 216 cubic feet and the new cube's dimensions
-- are each twice that of the original cube, prove that the volume of the new cube is 1728 cubic feet.
theorem new_cube_volume : ∀ (original_volume : ℝ) (scale_factor : ℝ), 
                           original_volume = 216 → 
                           scale_factor = 2 →
                           (scale_factor * (^(3: ℝ) (original_volume^(1/3: ℝ))))^(3: ℝ) = 1728 :=
by
  intro original_volume scale_factor h1 h2
  sorry

end new_cube_volume_l419_419357


namespace no_absolute_winner_prob_l419_419897

open_locale probability

-- Define the probability of Alyosha winning against Borya
def P_A_wins_B : ℝ := 0.6

-- Define the probability of Borya winning against Vasya
def P_B_wins_V : ℝ := 0.4

-- There are no ties, and each player plays with each other once
-- Conditions ensure that all pairs have played exactly once

-- Define the event that there will be no absolute winner
def P_no_absolute_winner : ℝ := P_A_wins_B * P_B_wins_V * 1 + P_A_wins_B * (1 - P_B_wins_V) * (1 - 1)

-- Statement of the problem: Prove that the probability of event C is 0.24
theorem no_absolute_winner_prob :
  P_no_absolute_winner = 0.24 :=
  by
    -- Placeholder for proof
    sorry

end no_absolute_winner_prob_l419_419897


namespace bottles_per_case_l419_419876

theorem bottles_per_case (total_bottles : ℕ) (total_cases : ℕ) (h1 : total_bottles = 60000) (h2 : total_cases = 12000) :
  total_bottles / total_cases = 5 :=
by
  -- Using the given problem, so steps from the solution are not required here
  sorry

end bottles_per_case_l419_419876


namespace cory_needs_more_money_l419_419436

def money_needed_to_buy_candies (current_money : ℝ) (num_packs : ℕ) (cost_per_pack : ℝ) : ℝ :=
  (num_packs * cost_per_pack) - current_money

theorem cory_needs_more_money (current_money num_packs cost_per_pack : ℝ) (h_current_money : current_money = 20) (h_num_packs : num_packs = 2) (h_cost_per_pack: cost_per_pack = 49) :
  money_needed_to_buy_candies current_money num_packs cost_per_pack = 78 := by
  sorry

end cory_needs_more_money_l419_419436


namespace problem_I_problem_II_l419_419163

theorem problem_I (a b p : ℝ) (F_2 M : ℝ × ℝ)
(h1 : a > b) (h2 : b > 0) (h3 : p > 0)
(h4 : (F_2.1)^2 / a^2 + (F_2.2)^2 / b^2 = 1)
(h5 : M.2^2 = 2 * p * M.1)
(h6 : M.1 = abs (M.2 - F_2.2) - 1)
(h7 : (|F_2.1 - 1|) = 5 / 2) :
    p = 2 ∧ ∃ f : ℝ × ℝ, (f.1)^2 / 9 + (f.2)^2 / 8 = 1 := sorry

theorem problem_II (k m x_0 : ℝ) 
(h8 : k ≠ 0) 
(h9 : m ≠ 0) 
(h10 : km = 1) 
(h11: ∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m) ∧
    ((A.1)^2 / 9 + (A.2)^2 / 8 = 1) ∧
    ((B.1)^2 / 9 + (B.2)^2 / 8 = 1) ∧
    (x_0 = (A.1 + B.1) / 2)) :
  -1 < x_0 ∧ x_0 < 0 := sorry

end problem_I_problem_II_l419_419163


namespace radius_of_concentric_circle_l419_419380

theorem radius_of_concentric_circle
  (side_length : ℝ)
  (h_side : side_length = 3)
  (probability_visible : ℝ)
  (h_probability : probability_visible = 1 / 2) : 
  ∃ (r : ℝ), r = 6 * real.sqrt 2 - real.sqrt 3 :=
sorry

end radius_of_concentric_circle_l419_419380


namespace solve_for_x_l419_419617

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end solve_for_x_l419_419617


namespace travel_distance_apart_l419_419075

theorem travel_distance_apart
  (speed_adam : ℕ) (speed_simon : ℕ) (distance : ℝ) (time : ℝ)
  (h_speed_adam : speed_adam = 10)
  (h_speed_simon : speed_simon = 8)
  (h_distance : distance = 80)
  (h_time : time = (40 * real.sqrt 41) / 41) :
  real.sqrt ((speed_adam * time)^2 + (speed_simon * time)^2) = distance :=
by
  sorry

end travel_distance_apart_l419_419075


namespace intersection_lies_on_Gamma_l419_419161

section GeometryProof

variables {A B C I M D F E X : Point}
variables (Gamma : Circle) (ABC : Triangle A B C)

-- Given conditions
variables (circumcenter_Gamma : Circumcenter Gamma ABC)
variables (incenter_I : Incenter I ABC)
variables (midpoint_M : Midpoint M B C)
variables (projection_D : Projection D I B C)
variables (perpendicular_line : PerpendicularLine I (AI : Line A I) (F E))
variables (circumcircle_AEF : Circumcircle (A E F) (circle_AEF : Circle))
variables (intersection_X : X ∈ circumcircle_AEF ∩ Gamma ∧ X ≠ A)
variables (line_XD : Line X D)
variables (line_AM : Line A M)

-- Question to be proved
theorem intersection_lies_on_Gamma :
    lies_on (intersection_of line_XD line_AM) Gamma :=
sorry

end GeometryProof

end intersection_lies_on_Gamma_l419_419161


namespace solve_for_x_l419_419622

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end solve_for_x_l419_419622


namespace line_parallel_or_subset_plane_l419_419679

-- Definitions for vectors a and n
def a : EuclideanSpace ℝ (Fin 3) := ![3, -2, -1]
def n : EuclideanSpace ℝ (Fin 3) := ![1, 2, -1]

-- The theorem to be proven
theorem line_parallel_or_subset_plane :
  let a := ![3, -2, -1]
  let n := ![1, 2, -1]
  dot_product n a = 0 → (line_parallel α || line_subset α) :=
    sorry

end line_parallel_or_subset_plane_l419_419679


namespace john_mows_lawn_in_2_79_hours_l419_419255

def lawn_mowing_time (lawn_length lawn_width swath_width_overlap walk_speed: ℝ) : ℝ :=
  let effective_swath_width := (swath_width_overlap - 6) / 12
  let num_strips := (lawn_width / effective_swath_width).ceil
  let total_distance := num_strips * lawn_length
  total_distance / walk_speed

theorem john_mows_lawn_in_2_79_hours :
  ∀ (lawn_length lawn_width : ℝ)
    (swath_width_overlap : ℝ)
    (walk_speed : ℝ),
  lawn_length = 120 →
  lawn_width = 200 →
  swath_width_overlap = 32 →
  walk_speed = 4000 →
  abs (lawn_mowing_time lawn_length lawn_width swath_width_overlap walk_speed - 2.79) < 0.01 :=
by
  intros
  unfold lawn_mowing_time
  rw [h1, h2, h3, h4]
  -- We'll finish the proof later
  sorry

end john_mows_lawn_in_2_79_hours_l419_419255


namespace det_dilation_matrix_scale5_l419_419678

theorem det_dilation_matrix_scale5 : 
  let E := Matrix.diagonal ![5, 5, 5] in
  Matrix.det E = 125 := 
by
  let E := Matrix.diagonal ![5, 5, 5]
  sorry

end det_dilation_matrix_scale5_l419_419678


namespace number_of_valid_n_l419_419598

-- Define the conditions of the problem as Lean definitions.
def condition1 (n : ℕ) : Prop :=
  150 > n

def condition2 (n : ℕ) : Prop :=
  n > 27

-- Define the final theorem to prove the number of valid n
theorem number_of_valid_n : {n : ℕ | condition1 n ∧ condition2 n}.card = 122 :=
by
  -- Replace this part with the actual proof
  sorry

end number_of_valid_n_l419_419598


namespace determinant_is_16_x_plus_1_l419_419956

noncomputable def determinant_matrix (x : ℝ) : ℝ :=
  Matrix.det !![![x + 2, x, x], ![x, x + 2, x], ![x, x, x + 2]]

theorem determinant_is_16_x_plus_1 (x : ℝ) : determinant_matrix x = 16 * (x + 1) :=
by
  sorry

end determinant_is_16_x_plus_1_l419_419956


namespace total_tennis_balls_used_l419_419645

def num_rounds : ℕ := 7
def games_per_round : list ℕ := [64, 32, 16, 8, 4, 2, 1]
def cans_per_game : ℕ := 6
def balls_per_can : ℕ := 4

theorem total_tennis_balls_used :
  ∑ (games_in_round : ℕ) in games_per_round, (games_in_round * cans_per_game) * balls_per_can = 3048 :=
by simp [games_per_round]
   sorry

end total_tennis_balls_used_l419_419645


namespace parabola_distance_sum_l419_419491

theorem parabola_distance_sum
  (P : ℕ → ℝ × ℝ) (F : ℝ × ℝ)
  (x : ℕ → ℝ)
  (hP : ∀ i, P i = (x i, (4 * (x i))^0.5))
  (hF : F = (1 / 4, 0))
  (hx_sum : ∑ i in finset.range n, x i = 20) :
  ∑ i in finset.range n, (|P i.1 - F.1|) = n + 20 :=
sorry

end parabola_distance_sum_l419_419491


namespace line_does_not_pass_through_second_quadrant_l419_419176

theorem line_does_not_pass_through_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ x - y - a^2 = 0 := 
by
  sorry

end line_does_not_pass_through_second_quadrant_l419_419176


namespace Sperner_theorem_example_l419_419785

theorem Sperner_theorem_example :
  ∀ (S : Finset (Finset ℕ)), (S.card = 10) →
  (∀ (A B : Finset ℕ), A ∈ S → B ∈ S → A ⊆ B → A = B) → S.card = 252 :=
by sorry

end Sperner_theorem_example_l419_419785


namespace find_sin_l419_419988

def sin_value (α : ℝ) : Prop :=
  sin (α + π / 3) = 1 / 3

theorem find_sin (α : ℝ) (h : sin_value α) : sin (2 * α - 5 * π / 6) = 7 / 9 :=
  sorry

end find_sin_l419_419988


namespace range_of_m_l419_419185

noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if h : -1 < x ∧ x ≤ 0 then (1 / (x + 1)) - 3
  else if h : 0 < x ∧ x ≤ 1 then x^2 - 3 * x + 2
  else 0  -- Outside the domain conditions given, return 0

theorem range_of_m (g : ℝ → ℝ) (h_g : ∀ x, g x = piecewise_function x) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 - m * x1 - m = 0 ∧ g x2 - m * x2 - m = 0) ↔
  m ∈ (- (9 / 4) : ℝ, -2] ∪ [0, 2 : ℝ) := by
  sorry

end range_of_m_l419_419185


namespace number_of_factors_of_x_l419_419698

theorem number_of_factors_of_x (a b c : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) (h3 : Nat.Prime c) (h4 : a < b) (h5 : b < c) (h6 : ¬ a = b) (h7 : ¬ b = c) (h8 : ¬ a = c) :
  let x := 2^2 * a^3 * b^2 * c^4
  let num_factors := (2 + 1) * (3 + 1) * (2 + 1) * (4 + 1)
  num_factors = 180 := by
sorry

end number_of_factors_of_x_l419_419698


namespace joohan_choices_l419_419673

theorem joohan_choices : 
  (number_of_hats : ℕ) (number_of_shoes : ℕ) 
  (hats_condition : number_of_hats = 4) 
  (shoes_condition : number_of_shoes = 3) : 
  (number_of_hats * number_of_shoes = 12) :=
by 
  intros number_of_hats number_of_shoes hats_condition shoes_condition
  rw [hats_condition, shoes_condition]
  exact mul_eq_12 4 3 sorry

end joohan_choices_l419_419673


namespace probability_no_absolute_winner_l419_419914

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end probability_no_absolute_winner_l419_419914


namespace octagon_diagonals_l419_419532

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l419_419532


namespace probability_of_person_A_being_selected_l419_419388

theorem probability_of_person_A_being_selected :
  let people := ['A', 'B', 'C', 'D'] in
  let total_outcomes := Nat.choose 4 2 in
  let favorable_outcomes := Nat.choose 3 1 in
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 2 :=
by
  let  people := ['A', 'B', 'C', 'D']
  let total_outcomes := Nat.choose 4 2
  let favorable_outcomes := Nat.choose 3 1
  have total_outcomes_eq : total_outcomes = 6 := by sorry
  have favorable_outcomes_eq : favorable_outcomes = 3 := by sorry
  calc
    (favorable_outcomes / total_outcomes : ℚ)
      = (3 / 6 : ℚ) : by rw [favorable_outcomes_eq, total_outcomes_eq]
  ... = 1 / 2 : by norm_num

end probability_of_person_A_being_selected_l419_419388


namespace range_of_m_l419_419998

variables (m : ℝ)

def P : Prop := 4 * m^2 - 4 * m < 0
def Q : Prop := ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

theorem range_of_m (h1 : P ∨ Q) (h2 : ¬ P ∧ ¬ Q) : -2 ≤ m ∧ m ≤ 0 ∨ 1 ≤ m ∧ m ≤ 2 := 
sorry

end range_of_m_l419_419998


namespace area_of_circumcircle_of_isosceles_triangle_l419_419834

theorem area_of_circumcircle_of_isosceles_triangle :
  let AB := 4
  let AC := 4
  let BC := 3
  let AD := (√(AB^2 - (BC/2)^2))
  let radius := AD
  let area := π * radius^2
  area = 16 * π :=
by
  sorry

end area_of_circumcircle_of_isosceles_triangle_l419_419834


namespace probability_no_absolute_winner_l419_419915

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end probability_no_absolute_winner_l419_419915


namespace sqrt_sum_simplification_l419_419792

theorem sqrt_sum_simplification : 
  sqrt (20 - 8 * sqrt 5) + sqrt (20 + 8 * sqrt 5) = 2 * sqrt 14 := 
by
  sorry

end sqrt_sum_simplification_l419_419792


namespace convex_function_general_inequality_l419_419154

def f (x : ℝ) (m : ℝ) : ℝ := x^2 + (Real.sqrt m) * x + m + 1

theorem convex_function_general_inequality (m : ℝ) (n : ℕ) (x : Fin n.succ → ℝ) (hx : ∀ i, 0 < x i) (hm : 0 ≤ m) :
  f (Real.sqrt (Finset.univ.prod x).toReal n) m ≤ Real.sqrt (Finset.univ.prod (λ i, f (x i) m)) n ∧
  (f (Real.sqrt (Finset.univ.prod x).toReal n) m = Real.sqrt (Finset.univ.prod (λ i, f (x i) m)) n ↔ ∀ i, x i = x 0) :=
    by sorry

end convex_function_general_inequality_l419_419154


namespace sequence_odd_part_l419_419423

def is_odd_part (x y : ℕ) : Prop :=
  ∃ b : ℕ, x = 2^b * y ∧ odd y

def odd_part (n : ℕ) : ℕ :=
  n / 2^n.trailingZeroBits

theorem sequence_odd_part (n : ℕ) (h_odd_n : odd n) :
  let a : ℕ → ℕ := λ k, if k = 0 then 2 * n - 1 else odd_part (3 * a (k-1) + 1) in
  a n = 3^n - 1 :=
by
  sorry

end sequence_odd_part_l419_419423


namespace complex_division_l419_419976

theorem complex_division : (1 + 2 * complex.i) / complex.i = 2 - complex.i :=
by
  -- The proof is omitted.
  sorry

end complex_division_l419_419976


namespace volume_of_tetrahedron_A1A2A3A4_height_of_tetrahedron_from_A4_l419_419367

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A1 := Point3D.mk 2 (-4) (-3)
def A2 := Point3D.mk 5 (-6) 0
def A3 := Point3D.mk (-1) 3 (-3)
def A4 := Point3D.mk (-10) (-8) 7

def vector_sub (p1 p2 : Point3D) : Point3D :=
  Point3D.mk (p1.x - p2.x) (p1.y - p2.y) (p1.z - p2.z)

def scalar_triple_product (u v w : Point3D) : ℝ :=
  u.x * (v.y * w.z - v.z * w.y) -
  u.y * (v.x * w.z - v.z * w.x) +
  u.z * (v.x * w.y - v.y * w.x)

noncomputable def volume_of_tetrahedron (A1 A2 A3 A4 : Point3D) : ℝ :=
  let v12 := vector_sub A2 A1
  let v13 := vector_sub A3 A1
  let v14 := vector_sub A4 A1
  (1 / 6) * (scalar_triple_product v12 v13 v14).abs

noncomputable def base_triangle_area (A1 A2 A3 : Point3D) : ℝ :=
  let v12 := vector_sub A2 A1
  let v13 := vector_sub A3 A1
  (1 / 2) * (Math.sqrt ((v12.y * v13.z - v12.z * v13.y) ^ 2 + 
                        (v12.z * v13.x - v12.x * v13.z) ^ 2 + 
                        (v12.x * v13.y - v12.y * v13.x) ^ 2))

noncomputable def height (V S : ℝ) : ℝ := (3 * V) / S

theorem volume_of_tetrahedron_A1A2A3A4 :
  volume_of_tetrahedron A1 A2 A3 A4 = 31.5 := by
  sorry

theorem height_of_tetrahedron_from_A4 :
  height (volume_of_tetrahedron A1 A2 A3 A4) (base_triangle_area A1 A2 A3) = 189 / Math.sqrt 747 := by
  sorry

end volume_of_tetrahedron_A1A2A3A4_height_of_tetrahedron_from_A4_l419_419367


namespace function_even_l419_419693

theorem function_even (n : ℤ) (h : 30 ∣ n)
    (h_prop: (1 : ℝ)^n^2 + (-1: ℝ)^n^2 = 2 * ((1: ℝ)^n + (-1: ℝ)^n - 1)) :
    ∀ x : ℝ, (x^n = (-x)^n) :=
by
    sorry

end function_even_l419_419693


namespace max_T_n_l419_419265

def T (n : ℕ) : ℝ := n * (2021 / 2022) ^ (n - 1)

theorem max_T_n (n : ℕ) : (n = 2021 ∨ n = 2022) ↔ (∀ k : ℕ, T k ≤ T n) := by
  sorry

end max_T_n_l419_419265


namespace monic_polynomials_closed_under_mul_l419_419448

noncomputable def is_monic (f : ℤ[X]) : Prop :=
f.leading_coeff = 1

noncomputable def closed_under_mul (f : ℤ[X]) : Prop :=
∀ x y : ℤ, x ∈ (polynomial.eval₂ ring_hom.id id f '' (set.univ : set ℤ)) →
          y ∈ (polynomial.eval₂ ring_hom.id id f '' (set.univ : set ℤ)) →
          x * y ∈ (polynomial.eval₂ ring_hom.id id f '' (set.univ : set ℤ))

theorem monic_polynomials_closed_under_mul (f : ℤ[X]) :
is_monic f ∧ closed_under_mul f ↔ 
  (f = 0 ∨ f = 1 ∨ f = -1 ∨ ∃ (a : ℤ) (k : ℕ), k > 0 ∧ f = polynomial.monomial k 1 + polynomial.monomial 0 a) :=
sorry

end monic_polynomials_closed_under_mul_l419_419448


namespace probability_is_correct_l419_419376

def num_red : ℕ := 7
def num_green : ℕ := 9
def num_yellow : ℕ := 10
def num_blue : ℕ := 5
def num_purple : ℕ := 3

def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue + num_purple

def num_blue_or_purple : ℕ := num_blue + num_purple

-- Probability of selecting a blue or purple jelly bean
def probability_blue_or_purple : ℚ := num_blue_or_purple / total_jelly_beans

theorem probability_is_correct :
  probability_blue_or_purple = 4 / 17 := sorry

end probability_is_correct_l419_419376


namespace positive_difference_single_fraction_simplest_form_l419_419788

theorem positive_difference_single_fraction_simplest_form (n : ℤ) :
  let expr := n / (n + 1 - (n + 2) / (n + 3)) in
  let numerator := n * (n + 3) / (n^2 + 3 * n + 1) in
  let denominator := (n^2 + 3 * n + 1) / (n^2 + 3 * n + 1) in
  |numerator - denominator| = 1 :=
by
  let expr := n / (n + 1 - (n + 2) / (n + 3))
  let numerator := n * (n + 3) / (n^2 + 3 * n + 1)
  let denominator := (n^2 + 3 * n + 1) / (n^2 + 3 * n + 1)
  sorry

end positive_difference_single_fraction_simplest_form_l419_419788


namespace diagonals_of_octagon_l419_419541

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l419_419541


namespace find_coefficient_of_x3_l419_419752

noncomputable def coefficient_x3 (a : ℚ) : ℚ :=
  let poly := (a * X - 3 / (4 * X) + 2 / 3) * (X - 3 / X) ^ 6
  poly.coeff 3

theorem find_coefficient_of_x3 (a : ℚ) 
  (H : ∑ i in (range ((a * X - 3 / (4 * X) + 2 / 3) * (X - 3 / X) ^ 6).support, 
    ((a * X - 3 / (4 * X) + 2 / 3) * (X - 3 / X) ^ 6).coeff i) = 16) :
  coefficient_x3 a = 117 / 2 :=
sorry

end find_coefficient_of_x3_l419_419752


namespace brianna_remaining_money_l419_419426

-- Given conditions
variables {m c n : ℝ}
-- Conditions: Brianna uses one quarter of her money to buy one quarter of the albums
def condition1 : Prop := (1 / 4) * m = (1 / 4) * n * c
-- All albums have an equal price

-- Proof statement
theorem brianna_remaining_money (h : condition1) : m - n * c = 0 :=
by
  sorry

end brianna_remaining_money_l419_419426


namespace point_P_inside_circle_l419_419188

theorem point_P_inside_circle (
  a b c x1 x2 : ℝ
  (h1 : a > 0)
  (h2 : b > 0)
  (h_hyperbola : ∀ x y, (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1))
  (h_eccentricity : ∀ e, e = real.sqrt 2 → c = real.sqrt 2 * a)
  (h_quadratic_eq : ∀ x, a * x ^ 2 - b * x - c = 0)
) : x1 ^ 2 + x2 ^ 2 < 8 :=
sorry

end point_P_inside_circle_l419_419188


namespace S_15_is_1695_l419_419946

-- Define the nth set of consecutive integers
def nth_set (n : ℕ) : Set ℕ :=
  let start := 1 + (n * (n - 1)) / 2
  { start + i | i in Finset.range n }

-- Define the sum of elements in the nth set
def S (n : ℕ) : ℕ :=
  (nth_set n).sum id

-- Theorem statement for S_15
theorem S_15_is_1695 : S 15 = 1695 :=
by
  sorry

end S_15_is_1695_l419_419946


namespace range_of_log_of_sqrt_sin_l419_419005

noncomputable def log_of_sqrt_sin (x : ℝ) : ℝ := log 10 (sqrt (sin x))

theorem range_of_log_of_sqrt_sin :
    (∀ x : ℝ, 0 < x ∧ x < real.pi → log_of_sqrt_sin x < 0) := by
  sorry

end range_of_log_of_sqrt_sin_l419_419005


namespace determinant_of_matrix_l419_419958

theorem determinant_of_matrix : ∀ (x : ℝ),
  Matrix.det !![
    [x + 2, x, x],
    [x, x + 2, x],
    [x, x, x + 2]
  ] = 8 * x + 8 :=
by
  intros x
  sorry

end determinant_of_matrix_l419_419958


namespace problem_solution_1_problem_solution_2_l419_419700

variables {a b c x y k : Real}

def ellipse_equation (a b : Real) : Prop :=
  b > 0 ∧ a > b ∧ a = 2 ∧ b = 1 ∧ 
  (∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1) ↔ ((x^2)/4 + y^2 = 1)

 def slopes_condition (k : Real) : Prop :=
  (k > sqrt 3 / 2 ∨ k < - sqrt 3 / 2) ∧ 
  (k < 2 ∧ k > -2) ↔ ((k > 0 ∨ k < 0)) → 
  - sqrt 3 / 6 < k ∧ k < sqrt 3 / 6 ∨ 1 / 8 < k ∧ k < sqrt 3 / 6

theorem problem_solution_1 :
  let C := ellipse_equation 2 1
  in  C :=
sorry

theorem problem_solution_2 :
  let slope_range := slopes_condition
  in slope_range :=
sorry

end problem_solution_1_problem_solution_2_l419_419700


namespace binary_sum_eq_669_l419_419931

def binary111111111 : ℕ := 511
def binary1111111 : ℕ := 127
def binary11111 : ℕ := 31

theorem binary_sum_eq_669 :
  binary111111111 + binary1111111 + binary11111 = 669 :=
by
  sorry

end binary_sum_eq_669_l419_419931


namespace jenna_rolls_more_2s_than_5s_l419_419213

theorem jenna_rolls_more_2s_than_5s:
  let total_outcomes := 6^5 in
  let equal_2s_5s_outcomes := 2334 in
  (0 : ℝ) < 1 ∧
  (total_outcomes = 7776) ∧
  (equal_2s_5s_outcomes = 2334) →
  (1 - (equal_2s_5s_outcomes / total_outcomes) ≠ 0) →
  ((1 / 2) * (1 - (equal_2s_5s_outcomes / total_outcomes)) = 2721 / 7776) :=
by {
  sorry
}

end jenna_rolls_more_2s_than_5s_l419_419213


namespace cos_sum_of_triangle_angles_lt_two_l419_419297

theorem cos_sum_of_triangle_angles_lt_two
  (α β γ : ℝ)
  (h_sum : α + β + γ = π) 
  (hα_pos : 0 < α) (h_mid : α < π) 
  (hβ_pos : 0 < β) (h2 : β < π) 
  (hγ_pos : 0 < γ) (hends : γ < π):
  cos α + cos β + cos γ < 2 := 
by
  sorry

end cos_sum_of_triangle_angles_lt_two_l419_419297


namespace arithmetic_sequence_b1_l419_419142

theorem arithmetic_sequence_b1 
  (b : ℕ → ℝ) 
  (U : ℕ → ℝ)
  (U2023 : ℝ) 
  (b2023 : ℝ)
  (hb2023 : b 2023 = b 1 + 2022 * (b 2 - b 1))
  (hU2023 : U 2023 = 2023 * (b 1 + 1011 * (b 2 - b 1))) 
  (hUn : ∀ n, U n = (n * (2 * b 1 + (n - 1) * (b 2 - b 1)) / 2)) :
  b 1 = (U 2023 - 2023 * b 2023) / 2023 :=
by
  sorry

end arithmetic_sequence_b1_l419_419142


namespace circle_center_radius_l419_419953

def circle_proof_problem (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 2 * y = 0

theorem circle_center_radius :
  ∃ (h k r : ℝ), circle_proof_problem (x - h) (y - k) ∧ h = 2 ∧ k = -1 ∧ r = sqrt 5 :=
by
  sorry

end circle_center_radius_l419_419953


namespace solve_weights_problem_l419_419704

variable (a b c d : ℕ) 

def weights_problem := 
  a + b = 280 ∧ 
  a + d = 300 ∧ 
  c + d = 290 → 
  b + c = 270

theorem solve_weights_problem (a b c d : ℕ) : weights_problem a b c d :=
 by
  sorry

end solve_weights_problem_l419_419704


namespace limit_sum_arith_seq_l419_419751

def a (n : ℕ) : ℕ := 2 * n - 1
def S (n : ℕ) : ℕ := n ^ 2

theorem limit_sum_arith_seq :
  (filter.tendsto (λ n, (S n : ℝ) / (a n : ℝ)^2) filter.at_top (nhds (1 / 4))) :=
sorry

end limit_sum_arith_seq_l419_419751


namespace incorrect_statement_b_l419_419065

def temperature_sound_data : List (Int × Int) := [
  (-20, 318), 
  (-10, 324), 
  (0, 330), 
  (10, 336), 
  (20, 342), 
  (30, 348)
]

def statement_a_correct : Prop := 
  ∃ f : Int → Int, (∀ x, ∃ y, (x, y) ∈ temperature_sound_data)

def statement_b_incorrect : Prop := 
  ¬(∀ t₁ t₂ v₁ v₂, (t₁ > t₂ ∧ (t₁, v₁) ∈ temperature_sound_data ∧ (t₂, v₂) ∈ temperature_sound_data) → v₁ < v₂)

def statement_c_correct : Prop := 
  ∃ (v : Int), (20, v) ∈ temperature_sound_data ∧ v = 342

def statement_d_correct : Prop := 
  ∀ (t₁ t₂ : Int), (t₁ - t₂ = 10 ∧ (t₁, 336) ∈ temperature_sound_data ∧ (t₂, v) ∈ temperature_sound_data) → (v = t₁ - 6)

theorem incorrect_statement_b : statement_b_incorrect :=
sorry

end incorrect_statement_b_l419_419065


namespace rd_expense_necessary_for_increase_l419_419932

theorem rd_expense_necessary_for_increase :
  ∀ (R_and_D_t : ℝ) (delta_APL_t1 : ℝ),
  R_and_D_t = 3289.31 → delta_APL_t1 = 1.55 →
  R_and_D_t / delta_APL_t1 = 2122 := 
by
  intros R_and_D_t delta_APL_t1 hR hD
  rw [hR, hD]
  norm_num
  sorry

end rd_expense_necessary_for_increase_l419_419932


namespace probability_valid_integer_l419_419080

-- Define the range of integers
def in_range (n : ℕ) : Prop := 2000 ≤ n ∧ n ≤ 8999

-- Define the property of being even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the property of having all distinct digits
def distinct_digits (n : ℕ) : Prop :=
  let digits := (to_digits n) in -- to_digits function needs to be defined to get the list of digits
  nodup digits

-- Define the property of not containing the digit 5
def no_five (n : ℕ) : Prop :=
  let digits := (to_digits n) in
  ¬(5 ∈ digits)

-- Define the combined properties
def valid_integer (n : ℕ) : Prop :=
  in_range n ∧ is_even n ∧ distinct_digits n ∧ no_five n

-- Define the count of valid integers satisfying all conditions
def valid_integer_count : ℕ :=
  (list.range 7000).countp (λ n, valid_integer (n + 2000))

theorem probability_valid_integer :
  (valid_integer_count : ℚ) / 7000 = 3 / 28 :=
sorry

end probability_valid_integer_l419_419080


namespace clap_hands_time_and_distance_l419_419023

theorem clap_hands_time_and_distance :
  (∀ (circumference : ℝ) (time_per_lap_a : ℝ) (time_per_lap_b : ℝ)
      (laps_before_reverse : ℕ) (num_claps A : ℝ),
    circumference = 400 ∧ 
    time_per_lap_a = 4 ∧ 
    time_per_lap_b = 7 ∧
    laps_before_reverse = 10 ∧
    num_claps = 15 →
    A = 66 + (2 / 11) →
    let speed_b := (circumference / time_per_lap_b) in
    B = speed_b * A →
    B = 3781 + (9 / 11)) := sorry

end clap_hands_time_and_distance_l419_419023


namespace magnitude_of_product_l419_419121

-- Definitions of the complex numbers involved
def z1 : ℂ := 3 * real.sqrt 5 - 5 * complex.I
def z2 : ℂ := 2 * real.sqrt 2 + 4 * complex.I

-- The statement to be proven
theorem magnitude_of_product :
  complex.abs (z1 * z2) = real.sqrt 1680 :=
by
  sorry

end magnitude_of_product_l419_419121


namespace octagon_has_20_diagonals_l419_419563

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l419_419563


namespace sum_of_roots_l419_419320

theorem sum_of_roots (a b p c d q : ℝ) (h1 : a ≠ 0) 
  (h2 : ∀ x, x^2 + 2020 * a * x + c = 0 → x = p ∨ x = b) 
  (h3 : ∃ x1 x2, ax^2 + bx + d = 0 → x = x1 ∨ x = x2 ∧ x1 ≠ x2) 
  (h4 : ∃ x3 x4, ax^2 + px + q = 0 → x = x3 ∨ x = x4 ∧ x3 ≠ x4) : 
  (∑ x in {x1, x2, x3, x4}, x) = 2020 :=
by
  sorry

end sum_of_roots_l419_419320


namespace volume_is_correct_l419_419012

-- Define the dimensions of the original rectangular sheet
def original_length : ℕ := 48
def original_width : ℕ := 36

-- Define the side length of the square cut from each corner
def square_side : ℕ := 8

-- Compute the new dimensions of the base of the box
def new_length : ℕ := original_length - 2 * square_side
def new_width : ℕ := original_width - 2 * square_side

-- The height of the box is the side length of the square cut
def height : ℕ := square_side

-- Compute the volume of the box
def volume_of_box : ℕ := new_length * new_width * height

-- The theorem to be proved: the volume of the resulting open box is 5120 m^3
theorem volume_is_correct : volume_of_box = 5120 := by
  sorry

end volume_is_correct_l419_419012


namespace mutually_exclusive_event_A_l419_419146

noncomputable theory

-- Definitions
def bag : Set (List (string × ℕ)) := {("Black", 4), ("White", 2)}

def draw := List (string × ℕ)

def event_A (draw : List (string × ℕ)) : Prop :=
  let white_balls := draw.count (λ(⟨c, _⟩), c = "White")
  white_balls ≤ 1

def mutually_exclusive_event_to_A (draw : List (string × ℕ)) : Prop :=
  let white_balls := draw.count (λ(⟨c, _⟩), c = "White")
  let black_balls := draw.count (λ(⟨c, _⟩), c = "Black")
  white_balls = 2 ∧ black_balls = 1

-- Lean statement
theorem mutually_exclusive_event_A :
  ∀ (draw : List (string × ℕ)),
    mutually_exclusive_event_to_A draw ↔ ¬ event_A draw :=
sorry

end mutually_exclusive_event_A_l419_419146


namespace octagon_diagonals_20_l419_419553

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l419_419553


namespace number_of_chairs_borrowed_l419_419222

-- Define the conditions
def red_chairs := 4
def yellow_chairs := 2 * red_chairs
def blue_chairs := yellow_chairs - 2
def total_initial_chairs : Nat := red_chairs + yellow_chairs + blue_chairs
def chairs_left_in_the_afternoon := 15

-- Define the question
def chairs_borrowed_by_Lisa : Nat := total_initial_chairs - chairs_left_in_the_afternoon

-- The theorem to state the proof problem
theorem number_of_chairs_borrowed : chairs_borrowed_by_Lisa = 3 := by
  -- Proof to be added
  sorry

end number_of_chairs_borrowed_l419_419222


namespace circle_and_tangent_lines_l419_419494

open Real

noncomputable def equation_of_circle_center_on_line (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), (x - a)^2 + (y - (a + 1))^2 = 2 ∧ (a = 4) ∧ (b = 5)

noncomputable def tangent_line_through_point (x y : ℝ) : Prop :=
  y = x - 1 ∨ y = (23 / 7) * x - (23 / 7)

theorem circle_and_tangent_lines :
  (∃ (a b : ℝ), (a = 4) ∧ (b = 5) ∧ (∀ x y : ℝ, equation_of_circle_center_on_line x y)) ∧
  (∀ x y : ℝ, tangent_line_through_point x y) := 
  by
  sorry

end circle_and_tangent_lines_l419_419494


namespace rachels_game_final_configurations_l419_419299

-- Define the number of cells in the grid
def n : ℕ := 2011

-- Define the number of moves needed
def moves_needed : ℕ := n - 3

-- Define a function that counts the number of distinct final configurations
-- based on the number of fights (f) possible in the given moves.
def final_configurations : ℕ := moves_needed + 1

theorem rachels_game_final_configurations : final_configurations = 2009 :=
by
  -- Calculation shows that moves_needed = 2008 and therefore final_configurations = 2008 + 1 = 2009.
  sorry

end rachels_game_final_configurations_l419_419299


namespace square_side_reduction_l419_419325

theorem square_side_reduction (s : ℝ) (h : s > 0) :
    let new_side := 1.2 * s
    let reduction_factor := 1 / 1.2
    let reduced_side := reduction_factor * s
    reduction_percentage := (1 - reduction_factor) * 100
    increase_percentage := 20
in 
  (new_side * reduced_side = s * s) ∧ (increase_percentage = 20)
  → reduction_percentage = 16.67 := sorry

end square_side_reduction_l419_419325


namespace interval_monotonically_increasing_find_cos_2x0_l419_419187

noncomputable def f (x : Real) : Real := Real.sin x * Real.cos x - Real.sqrt 3 * (1 - Real.cos x ^ 2) + Real.sqrt 3 * Real.cos x ^ 2 - 3 * Real.sin x * Real.cos x

theorem interval_monotonically_increasing : 
  (∀ x ∈ (Icc (Real.pi / 4) (3 * Real.pi / 4) ∪ Icc (5 * Real.pi / 4) (7 * Real.pi / 4) ∪ Icc (9 * Real.pi / 4) (11 * Real.pi / 4)), deriv f x > 0) :=
sorry

theorem find_cos_2x0 (x0 : Real) (hx0 : x0 ∈ Icc 0 (Real.pi / 2)) (hf : f x0 = 6 / 5) : 
  Real.cos (2 * x0) = (4 + 3 * Real.sqrt 3) / 10 :=
sorry

end interval_monotonically_increasing_find_cos_2x0_l419_419187


namespace sum_triples_eq_1_over_13397_l419_419939

theorem sum_triples_eq_1_over_13397 :
  (∑ (a b c : ℕ) in finset.filter (λ t : ℕ × ℕ × ℕ, 1 ≤ t.1 ∧ t.1 < t.2 ∧ t.2 < t.3) (finset.Icc 1 (finset.range 10000) ×ˢ finset.Icc 1 (finset.range 10000) ×ˢ finset.Icc 1 (finset.range 10000)), 
  (1 : ℝ) / (3^a * 4^b * 6^c)) = 1 / 13397 := by
  sorry

end sum_triples_eq_1_over_13397_l419_419939


namespace problem1_problem2_l419_419370

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (hx : ∀ x, x ≥ 0 → f (real.sqrt x) = x - 1) :
  ∀ x, x ≥ 0 → f x = x^2 - 1 := by
  sorry

-- Problem 2
theorem problem2 (f : ℝ → ℝ) (hf : ∀ x, f x = a * x + b)
  (hf_eqn : ∀ x, f (f x) = f x + 2) :
  f = (λ x, x + 2) := by
  sorry

end problem1_problem2_l419_419370


namespace octagon_has_20_diagonals_l419_419561

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l419_419561


namespace octagon_diagonals_l419_419594

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l419_419594


namespace symmetric_polynomial_problem_l419_419692

noncomputable def x_i (i : ℕ) : ℝ := sorry

theorem symmetric_polynomial_problem 
  (h1 : ∑ i in (Finset.range 4).map (λ i, i + 1), x_i i = 0)
  (h2 : ∑ i in (Finset.range 4).map (λ i, i + 1), (x_i i) ^ 7 = 0) :
  let u := x_i 3 * (x_i 3 + x_i 0) * (x_i 3 + x_i 1) * (x_i 3 + x_i 2) in
  u = 0 :=
by
  sorry

end symmetric_polynomial_problem_l419_419692


namespace area_of_circle_passing_through_vertices_l419_419849

noncomputable def circle_area_through_isosceles_triangle_vertices 
  (a b c : ℝ) (h_isosceles: (a = b) (h_sides: a = 4) (h_base: c = 3) : ℝ :=
π *(√((4^2 - (3/2)^2)/2 + (3/2))^2

theorem area_of_circle_passing_through_vertices :
  circle_area_through_isosceles_triangle_vertices 4 4 3 = 5.6875 * π :=
sorry

end area_of_circle_passing_through_vertices_l419_419849


namespace cory_needs_additional_money_l419_419437

theorem cory_needs_additional_money :
  ∀ (price_per_pack : ℝ) (number_of_packs : ℕ) (money_cory_has : ℝ), 
  price_per_pack = 49 → 
  number_of_packs = 2 → 
  money_cory_has = 20 → 
  let total_cost := price_per_pack * number_of_packs 
  let additional_money_needed := total_cost - money_cory_has 
  additional_money_needed = 78 :=
by
  intros price_per_pack number_of_packs money_cory_has h_price h_packs h_money_cory_has
  simp [h_price, h_packs, h_money_cory_has]
  have total_cost : total_cost = 49 * 2 := by simp [total_cost]
  have additional_money_needed : additional_money_needed = 49 * 2 - 20 := by simp [additional_money_needed, total_cost]
  exact additional_money_needed

-- Stated the theorem
{ sorry }

end cory_needs_additional_money_l419_419437


namespace always_lit_square_l419_419432

theorem always_lit_square 
  (m n : ℕ)
  (screen : Fin m → Fin n → Bool)  -- represents whether each unit square is lit (True) or not (False)
  (h1 : ∃ count_lit, count_lit > (m - 1) * (n - 1))  -- More than (m-1) * (n-1) unit squares are lit
  (h2 : ∀ i j : Fin (m-1), ∀ k l : Fin (n-1), (¬ screen i j) ∧ (¬ screen (i+1 % m) j) ∧ (¬ screen i (j+1 % n)) → ¬ screen (i+1 % m) (j+1 % n)) -- 3 unlit squares in a 2x2 imply the 4th is also unlit
  : ∃ (i : Fin m) (j : Fin n), screen i j = true := 
sorry

end always_lit_square_l419_419432


namespace range_of_a1_l419_419643

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

theorem range_of_a1 (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h_geometric_sequence : ∀ n, S n = geometric_sequence_sum a1 q n)
  (h_limit : tendsto S at_top (𝓝 (1 / a1)))
  (h_condition : a1 > 1) :
  a1 ∈ set.Ioo 1 (real.sqrt 2) :=
sorry

end range_of_a1_l419_419643


namespace sum_of_fractions_and_decimal_l419_419759

theorem sum_of_fractions_and_decimal :
  (6 / 5 : ℝ) + (1 / 10 : ℝ) + 1.56 = 2.86 :=
by
  sorry

end sum_of_fractions_and_decimal_l419_419759


namespace pieces_per_pan_of_brownies_l419_419768

theorem pieces_per_pan_of_brownies (total_guests guests_ala_mode additional_guests total_scoops_per_tub total_tubs_eaten total_pans guests_per_pan second_pan_percentage consumed_pans : ℝ)
    (h1 : total_guests = guests_ala_mode + additional_guests)
    (h2 : total_scoops_per_tub * total_tubs_eaten = guests_ala_mode * 2)
    (h3 : consumed_pans = 1 + second_pan_percentage)
    (h4 : second_pan_percentage = 0.75)
    (h5 : total_guests = guests_per_pan * consumed_pans)
    (h6 : guests_per_pan = 28)
    : total_guests / consumed_pans = 16 :=
by
  have h7 : total_scoops_per_tub * total_tubs_eaten = 48 := by sorry
  have h8 : guests_ala_mode = 24 := by sorry
  have h9 : total_guests = 28 := by sorry
  have h10 : consumed_pans = 1.75 := by sorry
  have h11 : guests_per_pan = 28 := by sorry
  sorry


end pieces_per_pan_of_brownies_l419_419768


namespace union_M_N_l419_419264

def M := {x : ℝ | x^2 = x}
def N := {x : ℝ | 1 < 2^x ∧ 2^x < 2}

theorem union_M_N : M ∪ N = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end union_M_N_l419_419264


namespace square_cut_into_triangles_l419_419394

theorem square_cut_into_triangles (n : ℕ) (hn : n = 1965) :
  ∃ (y x : ℕ), 
  y = 5896 ∧ x = 3932 ∧ 
  ∀ x t, (x * 180 = (n + 1) * 360)  → (x = 2 * (n + 1)) ∧ 
  (3 * t = 4 + 2 * y) → (t = x) :=
begin
  -- definition and condition (n should be 1965)
  have hn_eq : n = 1965, from hn,
  -- sorry is used because we do not need to supply the proof
  sorry 
end

end square_cut_into_triangles_l419_419394


namespace problem_solution_l419_419682

open Real Polynomial

theorem problem_solution (b c : ℝ) (h1 : b = c^2 + 1) (h2 : discriminant (X^2 + C b * X + C c) = 0) :
  c = 1 :=
by {
  sorry
}

end problem_solution_l419_419682


namespace proof_problem_l419_419992

noncomputable def f (x : ℝ) : ℝ := sorry

def condition_1 := ∀ x, f (1 + x) + f (1 - x) = 0
def condition_2 := ∀ x, f (x - 1) = f (2 - x)
def condition_3 := ∀ x, (0 ≤ x ∧ x ≤ 2) → f x = a * x^2 + b
def condition_4 := f 0 + f 3 = 2
def condition_5 : f (7 / 2) = 7 / 6 := sorry

theorem proof_problem :
  (∀ x, f (1 + x) + f (1 - x) = 0) →
  (∀ x, f (x - 1) = f (2 - x)) →
  (∀ x, (0 ≤ x ∧ x ≤ 2) → f x = a * x^2 + b) →
  (f 0 + f 3 = 2) →
  f (7 / 2) = 7 / 6 :=
by
  intro h1 h2 h3 h4
  exact condition_5

end proof_problem_l419_419992


namespace chord_length_l419_419333

-- Define the structure of the problem
structure Circle (r : ℝ) (center : ℝ × ℝ)

-- Given conditions
def C1 : Circle := {r := 6, center := (0, 0)}
def C2 : Circle := {r := 8, center := (14, 0)} -- since C1 and C2 are externally tangent and radii add up
def C3 : Circle := {r := 28, center := (7, 0)} -- placing the center at midpoint

-- The proof goal
theorem chord_length (C1 C2 C3 : Circle) (h1 : C1.r = 6) (h2 : C2.r = 8) (h3 : C3.r = 28)
  (h4 : C1.center.2 = 0) (h5 : C2.center.2 = 0) (h6 : C3.center.2 = 0)
  (h7 : C2.center.1 - C1.center.1 = C1.r + C2.r)
  (h8 : (C3.center.1 - C1.center.1)^2 + 0^2 = C3.r^2 - C1.r^2) : 
  ∃ (m n p : ℕ), (m + n + p = 94) ∧ (C3.r = 28) ∧ (∃ (AB : ℝ), AB = (40 * (sqrt 47)) / 7) :=
sorry

end chord_length_l419_419333


namespace odd_function_iff_l419_419317

def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_iff (a b : ℝ) : 
  (∀ x, f x a b = -f (-x) a b) ↔ (a ^ 2 + b ^ 2 = 0) :=
by
  sorry

end odd_function_iff_l419_419317


namespace area_of_circle_passing_through_vertices_l419_419851

noncomputable def circle_area_through_isosceles_triangle_vertices 
  (a b c : ℝ) (h_isosceles: (a = b) (h_sides: a = 4) (h_base: c = 3) : ℝ :=
π *(√((4^2 - (3/2)^2)/2 + (3/2))^2

theorem area_of_circle_passing_through_vertices :
  circle_area_through_isosceles_triangle_vertices 4 4 3 = 5.6875 * π :=
sorry

end area_of_circle_passing_through_vertices_l419_419851


namespace function_B_is_periodic_with_pi_l419_419077

def f (x : ℝ) : ℝ := (sin x)^2 - (sqrt 3) * (cos x)^2

theorem function_B_is_periodic_with_pi :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = π := sorry

end function_B_is_periodic_with_pi_l419_419077


namespace election_voters_Sobel_percentage_l419_419231

theorem election_voters_Sobel_percentage : 
  ∀ (total_voters : ℚ) (male_percentage : ℚ) (female_voters_Lange_percentage : ℚ) 
    (male_voters_Sobel_percentage : ℚ), 
  male_percentage = 0.6 → 
  female_voters_Lange_percentage = 0.35 →
  male_voters_Sobel_percentage = 0.44 →
  (total_voters * male_percentage * male_voters_Sobel_percentage + 
  total_voters * (1 - male_percentage) * (1 - female_voters_Lange_percentage)) / total_voters * 100 = 52.4 :=
begin
  intros total_voters male_percentage female_voters_Lange_percentage male_voters_Sobel_percentage,
  assume h1 h2 h3,
  calc
  (total_voters * male_percentage * male_voters_Sobel_percentage + 
  total_voters * (1 - male_percentage) * (1 - female_voters_Lange_percentage)) / total_voters * 100
    = ((60 / 100) * 0.44 * total_voters + (40 / 100) * 0.65 * total_voters) / total_voters * 100 : by rw [h1, h2, h3]
  ... = 52.4 : by { sorry }, 
end

end election_voters_Sobel_percentage_l419_419231


namespace tan_add_pi_over_3_l419_419208

variable (y : ℝ)

theorem tan_add_pi_over_3 (h : Real.tan y = 3) : 
  Real.tan (y + π / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) := 
by
  sorry

end tan_add_pi_over_3_l419_419208


namespace problems_without_conditional_statements_l419_419177

theorem problems_without_conditional_statements:
  (∀ {a b c : ℝ}, ax^2 + bx + c = 0 → requires_conditional_statements) ∧
  (∀ {O : Point} {r : ℝ} {L : Line}, positional_relationship O r L → requires_conditional_statements) ∧
  (∀ {s1 s2 s3 : Score}, rank_students s1 s2 s3 → requires_conditional_statements) ∧
  (∀ {P Q : Point}, distance_between P Q → ¬ requires_conditional_statements) →
  number_of_problems_without_conditional_statements = 1 := 
sorry

end problems_without_conditional_statements_l419_419177


namespace cube_sphere_volume_ratio_l419_419757

theorem cube_sphere_volume_ratio (a : ℝ) : 
  let edge_length := 2 * a in
  let inscribed_radius := a in
  let circumscribed_radius := sqrt 3 * a in
  (inscribed_radius / circumscribed_radius)^3 = 1 / (3 * sqrt 3) :=
by sorry

end cube_sphere_volume_ratio_l419_419757


namespace quadratic_b_value_l419_419285

theorem quadratic_b_value {b m : ℝ} (h : ∀ x, x^2 + b * x + 44 = (x + m)^2 + 8) : b = 12 :=
by
  -- hint for proving: expand (x+m)^2 + 8 and equate it with x^2 + bx + 44 to solve for b 
  sorry

end quadratic_b_value_l419_419285


namespace octagon_has_20_diagonals_l419_419585

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l419_419585


namespace find_AX_l419_419447

theorem find_AX (AC BC BX : ℝ) (hAC : AC = 27) (hBC : BC = 36) (hBX : BX = 30) :
  AX = 67.5 :=
by
  let AX := (AC * BX) / BC
  have h : AX = (27 * 30) / 36, from sorry
  simp [AX] at h
  exact h

end find_AX_l419_419447


namespace fraction_comparison_l419_419353

theorem fraction_comparison :
  let f := (5 / 4 : ℚ)
  (10 / 8 : ℚ) = f ∧
  (1 + 1 / 4 : ℚ) = f ∧
  (1 + 3 / 12 : ℚ) = f ∧
  (1 + 10 / 40 : ℚ) = f →
  (1 + 1 / 5 : ℚ) ≠ f :=
by 
  intros f h,
  sorry

end fraction_comparison_l419_419353


namespace find_X_l419_419237

variable (E X : ℕ)

-- Theorem statement
theorem find_X (hE : E = 9)
              (hSum : E * 100 + E * 10 + E + E * 100 + E * 10 + E = 1798) :
              X = 7 :=
sorry

end find_X_l419_419237


namespace shooting_probability_l419_419396

theorem shooting_probability (P : ℝ) (shots : ℕ) (hP : P = 0.7) (hshots : shots = 2) : 
  (1 - ((1 - P) * (1 - P))) = 0.91 :=
by 
  rw [hP, hshots]
  sorry

end shooting_probability_l419_419396


namespace sum_of_cubes_mod_five_l419_419683

theorem sum_of_cubes_mod_five
  (b : Fin 1024 → ℕ)
  (h_inc : ∀ i j : Fin 1024, i < j → b i < b j)
  (h_sum : ∑ i, b i = 2^1024) :
  (∑ i, (b i)^3) % 5 = 1 := by
  sorry

end sum_of_cubes_mod_five_l419_419683


namespace num_diagonals_octagon_l419_419572

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l419_419572


namespace problem_l419_419986

theorem problem (x : ℝ) (h : 3 * x^2 - 2 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2 / 3) = 3 :=
by
  sorry

end problem_l419_419986


namespace number_of_A_not_B_or_C_l419_419230

open Finset

variable {α : Type} (U A B C : Finset α)

-- Variables definition
variables (hU : card U = 300)
variables (hA : card A = 80)
variables (hB : card B = 70)
variables (hC : card C = 60)
variables (hAB : card (A ∩ B) = 30)
variables (hAC : card (A ∩ C) = 25)
variables (hBC : card (B ∩ C) = 20)
variables (hABC : card (A ∩ B ∩ C) = 15)
variables (hNotABC : card (U \ (A ∪ B ∪ C)) = 65)

theorem number_of_A_not_B_or_C : card (A \ (B ∪ C)) = 40 :=
sorry

end number_of_A_not_B_or_C_l419_419230


namespace determinant_is_16_x_plus_1_l419_419957

noncomputable def determinant_matrix (x : ℝ) : ℝ :=
  Matrix.det !![![x + 2, x, x], ![x, x + 2, x], ![x, x, x + 2]]

theorem determinant_is_16_x_plus_1 (x : ℝ) : determinant_matrix x = 16 * (x + 1) :=
by
  sorry

end determinant_is_16_x_plus_1_l419_419957


namespace intersect_at_common_point_l419_419382

variable {P : Type} [EuclideanAffineSpace P]

-- Definitions for the given convex solid
variable (A1 B1 C1 A2 B2 C2 : P)
-- Intersection points of the diagonals
variable (A3 B3 C3 : P)

-- Convex solid condition assumptions
axiom convex_solid : (∃ Q1 Q2 Q3: P, Q1 ≠ Q2 ∧ Q2 ≠ Q3 ∧ Q1 ≠ Q3) → convex (convex_hull ℝ ({A1, B1, C1, A2, B2, C2}: set P))
axiom intersection_points : (diagonal_intersect (quadrilateral (B1, B2, C2, C1))) A3 ∧ 
                            (diagonal_intersect (quadrilateral (C1, C2, A2, A1))) B3 ∧ 
                            (diagonal_intersect (quadrilateral (A1, A2, B2, B1))) C3

-- Statement to prove
theorem intersect_at_common_point 
  (h1 : line_through A1 A3) 
  (h2 : line_through B1 B3)
  (h3 : line_through C1 C3) :
  ∃ M : P, (M ∈ A1 <--[A3]) ∧ (M ∈ B1 <--[B3]) ∧ (M ∈ C1 <--[C3]) :=
sorry

end intersect_at_common_point_l419_419382


namespace calculate_expression_l419_419936

open Real

theorem calculate_expression : 1.1^0 + 0.5^(-2) - log 25 - 2 * log 2 = 1 :=
by
  sorry

end calculate_expression_l419_419936


namespace angle_B_is_2π_div_3_area_triangle_ABC_l419_419994

-- Definitions and conditions
variables {A B C a b c : ℝ}
variables (triangle_ABC : Type) [IsTriangle triangle_ABC (∠ A B C)] 
variables (side_a : Side triangle_ABC A b) 
variables (side_b : Side triangle_ABC B a) 
variables (side_c : Side triangle_ABC C c)
variables (eq1 : a + 2 * c = 2 * b * Real.cos A) 

-- Theorem: Proving B = 2π/3 given the conditions
theorem angle_B_is_2π_div_3 (h : a + 2 * c = 2 * b * Real.cos A) : ∠ B = 2 * π / 3 :=
  sorry

-- Additional conditions
variables (b_eq : b = 2 * Real.sqrt 3)
variables (sum_a_c_eq : a + c = 4)

-- Theorem: Proving the area of triangle ABC is √3 given additional conditions
theorem area_triangle_ABC (h1: b = 2 * Real.sqrt 3) (h2: a + c = 4) (h3 : ∠ B = 2 * π / 3) : 
  area triangle_ABC = Real.sqrt 3 :=
  sorry

end angle_B_is_2π_div_3_area_triangle_ABC_l419_419994


namespace total_cost_expression_minimize_total_cost_l419_419050

-- Define known parameters and conditions
def truck_speed := ℝ
def distance := 130
def min_speed := 50
def max_speed := 100
def gas_price := 8
def gas_consumption_rate (x : truck_speed) := 2 + x^2 / 360
def driver_wage := 80
def truck_speed_range (x : truck_speed) := min_speed ≤ x ∧ x ≤ max_speed

-- Define the expression for total cost y in terms of x
def total_cost (x : truck_speed) := 
  130 * (96 / x + x / 45)

-- Formulate the theorem to prove the expression for total cost
theorem total_cost_expression (x : truck_speed) : 
  truck_speed_range x →
  total_cost x = 130 * (96 / x + x / 45) :=
by
  sorry

-- Define the optimal speed that minimizes cost
def optimal_speed := 12 * real.sqrt 30

-- Define the minimum cost associated with the optimal speed
def minimum_cost := (208 / 3) * real.sqrt 30

-- Formulate the theorem to prove the optimal speed and minimum cost
theorem minimize_total_cost :
  total_cost optimal_speed = minimum_cost :=
by
  sorry

end total_cost_expression_minimize_total_cost_l419_419050


namespace rate_of_mixed_oil_l419_419209

/--
If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 68 per litre, 
8 litres of a third oil at Rs. 42 per litre, and 7 litres of a fourth oil at Rs. 62 per litre, 
then the rate of the mixed oil per litre is Rs. 53.67.
-/
theorem rate_of_mixed_oil :
  let cost1 := 10 * 50
  let cost2 := 5 * 68
  let cost3 := 8 * 42
  let cost4 := 7 * 62
  let total_cost := cost1 + cost2 + cost3 + cost4
  let total_volume := 10 + 5 + 8 + 7
  let rate_per_litre := total_cost / total_volume
  rate_per_litre = 53.67 :=
by
  intros
  sorry

end rate_of_mixed_oil_l419_419209


namespace coin_toss_sequences_count_l419_419228

theorem coin_toss_sequences_count :
  (∃ (seq : List Char), 
    seq.length = 15 ∧ 
    (seq == ['H', 'H']) = 5 ∧ 
    (seq == ['H', 'T']) = 3 ∧ 
    (seq == ['T', 'H']) = 2 ∧ 
    (seq == ['T', 'T']) = 4) → 
  (count_sequences == 775360) :=
by
  sorry

end coin_toss_sequences_count_l419_419228


namespace problem_statement_l419_419515

def curve_c_parametric (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sin α)

def line_l_polar_equation (ρ θ : ℝ) : Prop :=
  sqrt 2 * ρ * Real.sin (θ + Real.pi / 4) = 3

def curve_c_general (x y : ℝ) : Prop :=
  (x ^ 2) / 4 + y ^ 2 = 1

def line_l_cartesian (x y : ℝ) : Prop :=
  x + y - 3 = 0

def distance_from_point_to_line (x y : ℝ) : ℝ :=
  (|x + y - 3| / Real.sqrt 2)

def maximum_distance (d : ℝ) : Prop :=
  d = (Real.sqrt 10 + 3 * Real.sqrt 2) / 2

theorem problem_statement :
  (∀ α, curve_c_general (2 * Real.cos α) (Real.sin α)) ∧
  (∀ ρ θ, line_l_polar_equation ρ θ → line_l_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∀ α, ∃ d, d = distance_from_point_to_line (2 * Real.cos α) (Real.sin α) ∧ d ≤ (Real.sqrt 10 + 3 * Real.sqrt 2) / 2) :=
by
  -- Proof omitted
  sorry

end problem_statement_l419_419515


namespace outfit_combinations_l419_419807

theorem outfit_combinations 
  (shirts : Fin 5)
  (pants : Fin 6)
  (restricted_shirt : Fin 1)
  (restricted_pants : Fin 2) :
  ∃ total_combinations : ℕ, total_combinations = 28 :=
sorry

end outfit_combinations_l419_419807


namespace smallest_positive_period_of_f_max_value_of_f_in_interval_l419_419178

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x) * (Real.cos (x - (Real.pi / 6))) + 1

theorem smallest_positive_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi :=
by
  sorry

theorem max_value_of_f_in_interval : ∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x ≤ (√3 + 2) :=
by
  sorry

end smallest_positive_period_of_f_max_value_of_f_in_interval_l419_419178


namespace num_integer_terms_sequence_l419_419947

noncomputable def sequence_starting_at_8820 : Nat := 8820

def divide_by_5 (n : Nat) : Nat := n / 5

theorem num_integer_terms_sequence :
  let seq := [sequence_starting_at_8820, divide_by_5 sequence_starting_at_8820]
  seq = [8820, 1764] →
  seq.length = 2 := by
  sorry

end num_integer_terms_sequence_l419_419947


namespace triangle_area_3_4_5_l419_419407

noncomputable def triangle_area (r : ℝ) (a b c : ℝ) : ℝ :=
  if a ^ 2 + b ^ 2 = c ^ 2 then
    let x := 6 / 5 in
    let base := 3 * x in
    let height := 4 * x in
    (1 / 2) * base * height
  else 0

theorem triangle_area_3_4_5 (r : ℝ) (a b c : ℝ) 
  (h1 : r = 3) 
  (h2 : a = 3 * (6 / 5)) 
  (h3 : b = 4 * (6 / 5)) 
  (h4 : c = 5 * (6 / 5))
  (h5 : a ^ 2 + b ^ 2 = c ^ 2) :
  triangle_area r a b c = 8.64 :=
by
  have x := 6 / 5
  have base := 3 * x
  have height := 4 * x
  have area := (1 / 2) * base * height
  have h_area : area = 8.64 :=
    by
      simp [x, base, height, area]
  rw [triangle_area, if_pos h5]
  exact h_area

end triangle_area_3_4_5_l419_419407


namespace incorrect_number_calculation_l419_419309

theorem incorrect_number_calculation (X : ℕ) (incorrect_avg correct_avg : ℕ) 
  (incorrect_avg = 19) (correct_avg = 24) (correct_number = 76) (num_elements = 10) :
  X = 26 := by 
  sorry

end incorrect_number_calculation_l419_419309


namespace complex_number_solution_l419_419990

theorem complex_number_solution (z : ℂ) (h : 2 * complex.I / z = 1 - complex.I) : z = -1 + complex.I :=
sorry

end complex_number_solution_l419_419990


namespace octagon_has_20_diagonals_l419_419579

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l419_419579


namespace count_of_chipped_marbles_l419_419725

def bags : List ℕ := [15, 18, 20, 22, 24, 27, 30, 32, 35, 37]

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem count_of_chipped_marbles (chipped_marbles : ℕ) :
  chipped_marbles ∈ {15, 20, 30, 35} ↔
  (chipped_marbles ∈ bags ∧ is_divisible_by_5 (260 - chipped_marbles)) :=
by
  sorry

end count_of_chipped_marbles_l419_419725


namespace roots_of_polynomial_l419_419975

noncomputable def f (x : ℝ) : ℝ := 8 * x^5 + 35 * x^4 - 94 * x^3 + 63 * x^2 - 12 * x - 24

def f_roots : List ℝ := [-1/2, 3/2, -4, (-25 + Real.sqrt 641)/4, (-25 - Real.sqrt 641)/4]

theorem roots_of_polynomial : ∀ r ∈ f_roots, f r = 0 :=
by
  sorry

end roots_of_polynomial_l419_419975


namespace foldable_positions_l419_419049

/-- 
A flat layout consists of 4 congruent squares arranged to form a cross shape.
A fifth congruent square is to be attached at one of the ten positions indicated on the edges of this cross.
We want to prove that 8 of the ten positions will allow the arrangement to be folded into a cube with one face missing.
-/
theorem foldable_positions {squares : ℕ} (flat_layout : squares = 5)
    (positions : fin 10)
    (cross_shape_formed : Prop)
    : 
    (count (λ pos, foldable_into_cube_missing_one_face pos) (finset.range 10)) = 8 :=
sorry

end foldable_positions_l419_419049


namespace area_of_circumscribed_circle_isosceles_triangle_l419_419854

theorem area_of_circumscribed_circle_isosceles_triangle :
  ∃ (r : ℝ), (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi) :=
by
  -- Consider the isosceles triangle conditions
  let a : ℝ := 4
  let b : ℝ := 4
  let c : ℝ := 3
  let BD := Real.sqrt(a^2 - (c/2)^2)
  let r := 8 / BD
  have h1 : BD = Real.sqrt 13.75 := by 
    -- Calculate the altitude BD
    calc
      BD = Real.sqrt(a^2 -  (c/2)^2) : rfl
      ... = Real.sqrt(16 - (3/2)^2) : rfl
      ... = Real.sqrt 13.75 : rfl
  
  use r
  have h2 : r = 8 / Real.sqrt 13.75 := by 
    -- Simplify the radius expression
    sorry

  have h3 : Real.pi * r ^ 2 = 256 / 55 * Real.pi := by 
    -- Calculate the area
    calc
      Real.pi * r ^ 2 = Real.pi * (8 / Real.sqrt 13.75) ^ 2 : by rw h2
      ... = Real.pi * (64 / 13.75) : by rw [pow_two, mul_div_assoc, mul_one, div_mul_div_same]
      ... = (256 / 54.6875) * Real.pi : by rw mul_comm
      ...   = (256 / 55) * Real.pi : by norm_num
    sorry
  
  show (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi),
  from ⟨h2, h3⟩
  sorry

end area_of_circumscribed_circle_isosceles_triangle_l419_419854


namespace part1_average_decrease_rate_part2_unit_price_reduction_l419_419040

-- Part 1: Prove the average decrease rate is 10%
theorem part1_average_decrease_rate (p0 p2 : ℝ) (x : ℝ) 
    (h1 : p0 = 200) 
    (h2 : p2 = 162) 
    (hx : (1 - x)^2 = p2 / p0) : x = 0.1 :=
by {
    sorry
}

-- Part 2: Prove the unit price reduction should be 15 yuan
theorem part2_unit_price_reduction (p_sell p_factory profit : ℝ) (n_initial dn m : ℝ)
    (h3 : p_sell = 200)
    (h4 : p_factory = 162)
    (h5 : n_initial = 20)
    (h6 : dn = 10)
    (h7 : profit = 1150)
    (hx : (38 - m) * (n_initial + 2 * m) = profit) : m = 15 :=
by {
    sorry
}

end part1_average_decrease_rate_part2_unit_price_reduction_l419_419040


namespace find_c_l419_419104

def f (x: ℝ) : ℝ := 5 * x^3 - 1 / x + 3
def g (x: ℝ) (k: ℝ) (c: ℝ) : ℝ := x^2 - k * x + c

theorem find_c (c : ℝ) (h1: f 2 - g 2 1 c = 2) : c = 38.5 :=
sorry

end find_c_l419_419104


namespace sum_f_1_to_1990_l419_419659

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

theorem sum_f_1_to_1990 : (Finset.range 1990).sum f = 1326 :=
  by
  sorry

end sum_f_1_to_1990_l419_419659


namespace no_absolute_winner_prob_l419_419900

open_locale probability

-- Define the probability of Alyosha winning against Borya
def P_A_wins_B : ℝ := 0.6

-- Define the probability of Borya winning against Vasya
def P_B_wins_V : ℝ := 0.4

-- There are no ties, and each player plays with each other once
-- Conditions ensure that all pairs have played exactly once

-- Define the event that there will be no absolute winner
def P_no_absolute_winner : ℝ := P_A_wins_B * P_B_wins_V * 1 + P_A_wins_B * (1 - P_B_wins_V) * (1 - 1)

-- Statement of the problem: Prove that the probability of event C is 0.24
theorem no_absolute_winner_prob :
  P_no_absolute_winner = 0.24 :=
  by
    -- Placeholder for proof
    sorry

end no_absolute_winner_prob_l419_419900


namespace minimum_num_of_small_triangles_l419_419003

-- Define the area of an equilateral triangle with a given side length
def area_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (√3 / 4) * side_length^2

-- Define the side lengths of the small and large triangles
def small_triangle_side : ℝ := 2
def large_triangle_side : ℝ := 12

-- Define the corresponding areas of the small and large triangles
def small_triangle_area : ℝ := area_of_equilateral_triangle small_triangle_side
def large_triangle_area : ℝ := area_of_equilateral_triangle large_triangle_side

-- Define the number of small triangles required to cover the large triangle
def num_small_triangles : ℝ := large_triangle_area / small_triangle_area

-- State the theorem
theorem minimum_num_of_small_triangles :
  num_small_triangles = 36 := by
sorry

end minimum_num_of_small_triangles_l419_419003


namespace area_of_circle_passing_through_vertices_l419_419845

noncomputable def circle_area_through_isosceles_triangle_vertices 
  (a b c : ℝ) (h_isosceles: (a = b) (h_sides: a = 4) (h_base: c = 3) : ℝ :=
π *(√((4^2 - (3/2)^2)/2 + (3/2))^2

theorem area_of_circle_passing_through_vertices :
  circle_area_through_isosceles_triangle_vertices 4 4 3 = 5.6875 * π :=
sorry

end area_of_circle_passing_through_vertices_l419_419845


namespace books_sold_on_friday_l419_419672

theorem books_sold_on_friday 
  (total_books : ℕ) 
  (sold_mon : ℕ) 
  (sold_tue : ℕ) 
  (sold_wed : ℕ) 
  (sold_thu : ℕ) 
  (percentage_unsold : ℝ) :
  total_books = 1400 →
  sold_mon = 75 →
  sold_tue = 50 →
  sold_wed = 64 →
  sold_thu = 78 →
  percentage_unsold = 71.28571428571429 →
  let sold_percentage := 100 - percentage_unsold,
      total_books_sold := (sold_percentage / 100) * total_books,
      sold_mon_to_thu := sold_mon + sold_tue + sold_wed + sold_thu,
      sold_fri := total_books_sold - sold_mon_to_thu in
  sold_fri = 135 :=
by
  intros h1 h2 h3 h4 h5 h6;
  let sold_percentage := 100 - percentage_unsold;
  let total_books_sold := (sold_percentage / 100) * total_books;
  let sold_mon_to_thu := sold_mon + sold_tue + sold_wed + sold_thu;
  let sold_fri := total_books_sold - sold_mon_to_thu;
  ...

sorry

end books_sold_on_friday_l419_419672


namespace find_abc_l419_419964

theorem find_abc (a b c : ℤ) (h : 2^a + 9^b = 2 * 5^c - 7) : a = 1 ∧ b = 0 ∧ c = 1 :=
by sorry

end find_abc_l419_419964


namespace area_of_circumcircle_of_isosceles_triangle_l419_419835

theorem area_of_circumcircle_of_isosceles_triangle :
  let AB := 4
  let AC := 4
  let BC := 3
  let AD := (√(AB^2 - (BC/2)^2))
  let radius := AD
  let area := π * radius^2
  area = 16 * π :=
by
  sorry

end area_of_circumcircle_of_isosceles_triangle_l419_419835


namespace local_extrema_l419_419733

-- Defining the function y = 1 + 3x - x^3
def y (x : ℝ) : ℝ := 1 + 3 * x - x ^ 3

-- Statement of the problem to be proved
theorem local_extrema :
  (∃ x : ℝ, x = -1 ∧ y x = -1 ∧ ∀ ε > 0, ∃ δ > 0, ∀ z, abs (z + 1) < δ → y z ≥ y (-1)) ∧
  (∃ x : ℝ, x = 1 ∧ y x = 3 ∧ ∀ ε > 0, ∃ δ > 0, ∀ z, abs (z - 1) < δ → y z ≤ y 1) :=
by sorry

end local_extrema_l419_419733


namespace Mitch_macarons_proof_l419_419286

variables (M J Mi R : ℕ)

-- Conditions
def Joshua_macarons := J = M + 6
def Miles_macarons := Mi = 2 * (M + 6)
def Renz_macarons := R = (3 * (Mi / 2)) - 1
def total_macarons := M + J + Mi + R = 68 * 2

-- Theorem statement
theorem Mitch_macarons_proof :
  (∃ (M J Mi R : ℕ), Joshua_macarons M J ∧ Miles_macarons M Mi ∧ Renz_macarons Mi R ∧ total_macarons M J Mi R) →
  M = 20 :=
begin
  sorry
end

end Mitch_macarons_proof_l419_419286


namespace circumcircle_area_of_isosceles_triangle_l419_419867

open Real

theorem circumcircle_area_of_isosceles_triangle:
  (∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
    (dist A B = 4) ∧ (dist A C = 4) ∧ (dist B C = 3) →
    (circle_area A B C = (256 / 13.75) * π)) :=
by sorry

end circumcircle_area_of_isosceles_triangle_l419_419867


namespace right_triangle_sides_l419_419316

theorem right_triangle_sides (a b c : ℕ) (h1 : a < b) 
  (h2 : 2 * c / 2 = c) 
  (h3 : exists x y, (x + y = 8 ∧ a < b) ∨ (x + y = 9 ∧ a < b)) 
  (h4 : a^2 + b^2 = c^2) : 
  a = 3 ∧ b = 4 ∧ c = 5 := 
by
  sorry

end right_triangle_sides_l419_419316


namespace question_1_question_2_l419_419703

variable (m x : ℝ)
def f (x : ℝ) := |x + m|

theorem question_1 (h : f 1 + f (-2) ≥ 5) : 
  m ≤ -2 ∨ m ≥ 3 := sorry

theorem question_2 (hx : x ≠ 0) : 
  f (1 / x) + f (-x) ≥ 2 := sorry

end question_1_question_2_l419_419703


namespace cricket_team_final_winning_percentage_l419_419636

theorem cricket_team_final_winning_percentage:
  ∀ (initial_games won_initial won_streak total_games total_won final_percentage : ℕ),
    initial_games = 120 →
    won_initial = Nat.floor (0.26 * initial_games) →
    won_streak = 65 →
    total_games = initial_games + won_streak →
    total_won = won_initial + won_streak →
    final_percentage = (total_won * 100) / total_games →
    final_percentage = 51 := by
  sorry

end cricket_team_final_winning_percentage_l419_419636


namespace octagon_diagonals_l419_419592

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l419_419592


namespace heights_in_pentagon_impossible_l419_419647

theorem heights_in_pentagon_impossible (h : fin 5 → ℝ) : 
  ¬(∀ (i j k l m : fin 5),
    (h i + h j > h k + h l) ∧ (h i + h j < h l + h m) ∨ 
    (h i + h j < h k + h l) ∧ (h i + h j > h l + h m)) :=
sorry

end heights_in_pentagon_impossible_l419_419647


namespace functional_equation_solution_l419_419440

theorem functional_equation_solution (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x - 1 - f x) = f x - 1 - x)
  (h2 : {y : ℝ | ∃ x : ℝ, x ≠ 0 ∧ y = f x / x}.finite) :
  ∀ x : ℝ, f x = x :=
by sorry

end functional_equation_solution_l419_419440


namespace solve_eq_l419_419456

theorem solve_eq (x : ℝ) : (∛(5 - x / 3) = -4) → x = 207 :=
by {
    intro hyp,
    -- proof steps here
    sorry
}

end solve_eq_l419_419456


namespace union_M_N_inter_complement_M_N_union_complement_M_N_l419_419172

open Set

variable (U : Set ℝ) (M : Set ℝ) (N : Set ℝ)

noncomputable def universal_set := U = univ

def set_M := M = {x : ℝ | x ≤ 3}
def set_N := N = {x : ℝ | x < 1}

theorem union_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    M ∪ N = {x : ℝ | x ≤ 3} :=
by sorry

theorem inter_complement_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    (U \ M) ∩ N = ∅ :=
by sorry

theorem union_complement_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    (U \ M) ∪ (U \ N) = {x : ℝ | x ≥ 1} :=
by sorry

end union_M_N_inter_complement_M_N_union_complement_M_N_l419_419172


namespace cistern_empty_time_due_to_leak_l419_419810

def cistern_fill_time_without_leak : ℝ := 8 -- 8 hours to fill normally
def cistern_fill_time_with_leak : ℝ := 10 -- 10 hours to fill with leak

theorem cistern_empty_time_due_to_leak : 
  ∀ (R L : ℝ), 
    R = 1 / cistern_fill_time_without_leak → 
    (R - L) * cistern_fill_time_with_leak = 1 →
    (1 / L) = 40 :=
begin
  intros R L h_R h,
  sorry
end

end cistern_empty_time_due_to_leak_l419_419810


namespace number_of_ordered_quadruples_l419_419685

theorem number_of_ordered_quadruples (x1 x2 x3 x4 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h_sum : x1 + x2 + x3 + x4 = 100) : 
  ∃ n : ℕ, n = 156849 := 
by 
  sorry

end number_of_ordered_quadruples_l419_419685


namespace max_occurring_values_l419_419412

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 1)^2 / (x * (x^2 - 1))

theorem max_occurring_values:
  {a : ℝ} (h : ∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 ∧ f x = a) → (|a| > 4) ↔ a ∈ (-∞, -4) ∪ (4, ∞) := 
by
  sorry

end max_occurring_values_l419_419412


namespace laura_equity_investment_l419_419257

theorem laura_equity_investment:
  ∃ (d : ℝ), d + 3 * d = 250000 ∧ 3 * d = 187500 :=
begin
  sorry
end

end laura_equity_investment_l419_419257


namespace defective_probability_bayesian_probabilities_l419_419378

noncomputable def output_proportion_A : ℝ := 0.25
noncomputable def output_proportion_B : ℝ := 0.35
noncomputable def output_proportion_C : ℝ := 0.40

noncomputable def defect_rate_A : ℝ := 0.05
noncomputable def defect_rate_B : ℝ := 0.04
noncomputable def defect_rate_C : ℝ := 0.02

noncomputable def probability_defective : ℝ :=
  output_proportion_A * defect_rate_A +
  output_proportion_B * defect_rate_B +
  output_proportion_C * defect_rate_C 

theorem defective_probability :
  probability_defective = 0.0345 := 
  by sorry

noncomputable def P_A_given_defective : ℝ :=
  (output_proportion_A * defect_rate_A) / probability_defective

noncomputable def P_B_given_defective : ℝ :=
  (output_proportion_B * defect_rate_B) / probability_defective

noncomputable def P_C_given_defective : ℝ :=
  (output_proportion_C * defect_rate_C) / probability_defective

theorem bayesian_probabilities :
  P_A_given_defective = 25 / 69 ∧
  P_B_given_defective = 28 / 69 ∧
  P_C_given_defective = 16 / 69 :=
  by sorry

end defective_probability_bayesian_probabilities_l419_419378


namespace hexagon_diagonals_concurrent_l419_419232

theorem hexagon_diagonals_concurrent
  (nonconvex_nonself_intersecting : ¬ (convex ABCDEF) ∧ ¬ (self_intersecting ABCDEF))
  (no_opposite_sides_parallel : ∀ (opposite_sides : (side_of hexagon × side_of hexagon)), ¬ (parallel opposite_sides.1 opposite_sides.2))
  (angle_conditions : (∠A = 3 * ∠D) ∧ (∠C = 3 * ∠F) ∧ (∠E = 3 * ∠B))
  (side_lengths_equal : (AB = DE) ∧ (BC = EF) ∧ (CD = FA)) :
  concurrent (diagonal AD) (diagonal BE) (diagonal CF) :=
sorry

end hexagon_diagonals_concurrent_l419_419232


namespace possible_n_values_l419_419951

theorem possible_n_values (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 → n = 1 ∨ n = 3 :=
by 
  sorry

end possible_n_values_l419_419951


namespace eigenvalues_of_inverse_matrix_l419_419514

theorem eigenvalues_of_inverse_matrix (M : Matrix (Fin 2) (Fin 2) ℝ) (hM : M = ![![1, 0], ![2, 2]]) :
  eigenvalues M⁻¹ = {1, 1/2} :=
by
  sorry

end eigenvalues_of_inverse_matrix_l419_419514


namespace power_of_prime_calculate_729_to_two_thirds_l419_419089

theorem power_of_prime (a : ℝ) (b : ℝ) (c : ℝ) (h : a = b^c) (e : ℝ) :
  a ^ e = b ^ (c * e) := sorry

theorem calculate_729_to_two_thirds : (729 : ℝ) ^ (2 / 3) = 81 := by
  have h : 729 = (3 : ℝ) ^ 6 := by norm_num
  exact power_of_prime (729 : ℝ) (3 : ℝ) (6 : ℝ) h (2 / 3)

end power_of_prime_calculate_729_to_two_thirds_l419_419089


namespace distance_between_x_intercepts_is_ten_l419_419336

noncomputable def x_intercept_dist {k1 k2 : ℂ} (P : ℂ) (a b : ℂ) : ℂ :=
  |a - b|

theorem distance_between_x_intercepts_is_ten :
  ∀ (k1 k2 : ℂ) (P : ℂ) (a b : ℂ),
    k1 = 2 →
    k2 = 6 →
    P = (40 + 30 * I) →
    a = 25 →
    b = 35 →
    x_intercept_dist P a b = 10 :=
by sorry

end distance_between_x_intercepts_is_ten_l419_419336


namespace octagon_has_20_diagonals_l419_419567

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l419_419567


namespace gcd_9011_4379_l419_419782

def a : ℕ := 9011
def b : ℕ := 4379

theorem gcd_9011_4379 : Nat.gcd a b = 1 := by
  sorry

end gcd_9011_4379_l419_419782


namespace piecewise_f_solution_set_l419_419699

noncomputable def min (p q : ℝ) : ℝ :=
if p ≤ q then p else q

noncomputable def f (x : ℝ) : ℝ :=
min (3 + log x / log (1 / 4)) (log x / log 2)

theorem piecewise_f : ∀ x : ℝ, (0 < x → x < 2 → f x = log x / log 2) ∧ (2 ≤ x → f x = 3 + log x / log (1 / 4)) :=
by
  intros x hx
  split
  {
    assume hlt
    sorry
  }
  {
    assume hge
    sorry
  }

theorem solution_set : {x : ℝ | 0 < f x ∧ f x < 2} = (λ x : ℝ, 1 < x ∧ x < 2) ∪ (λ x : ℝ, 4 < x ∧ x < 64) :=
by sorry

end piecewise_f_solution_set_l419_419699


namespace num_valid_k_l419_419143

theorem num_valid_k :
  let k := λ (a b c : ℕ), 2^a * 3^b * 5^c in
  (9^9 = 3^18) ∧
  (16^10 = 2^40) ∧
  (36^10 = 2^10 * 3^20) ∧
  (∀ a b c, (36^10 = (Nat.lcm (Nat.gcd (9^9) (16^10)))) = (Nat.lcm (Nat.gcd (k a b c)))) ->
  ∃ (a_vals : Finset ℕ) (b_val : ℕ) (c_val : ℕ),
    (a_vals.card = 11) ∧
    (∀ a ∈ a_vals, a ≤ 10) ∧
    (b_val = 20) ∧
    (c_val = 0) :=
by 
  sorry

end num_valid_k_l419_419143


namespace lg_properties_l419_419979

noncomputable def f : ℝ → ℝ := λ x, Real.log x

theorem lg_properties (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h_ne : x₁ ≠ x₂):
  (f (x₁ * x₂) = f x₁ + f x₂) ∧
  ((f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
  (f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2) :=
by
  sorry

end lg_properties_l419_419979


namespace jerry_gets_logs_l419_419671

def logs_per_pine_tree : ℕ := 80
def logs_per_maple_tree : ℕ := 60
def logs_per_walnut_tree : ℕ := 100
def logs_per_oak_tree : ℕ := 90
def logs_per_birch_tree : ℕ := 55

def pine_trees_cut : ℕ := 8
def maple_trees_cut : ℕ := 3
def walnut_trees_cut : ℕ := 4
def oak_trees_cut : ℕ := 7
def birch_trees_cut : ℕ := 5

def total_logs : ℕ :=
  pine_trees_cut * logs_per_pine_tree +
  maple_trees_cut * logs_per_maple_tree +
  walnut_trees_cut * logs_per_walnut_tree +
  oak_trees_cut * logs_per_oak_tree +
  birch_trees_cut * logs_per_birch_tree

theorem jerry_gets_logs : total_logs = 2125 :=
by
  sorry

end jerry_gets_logs_l419_419671


namespace num_diagonals_octagon_l419_419573

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l419_419573


namespace points_XODP_concyclic_l419_419241

-- Given a triangle \(ABC\) with circumcenter \(O\)
variables {A B C O : Point}
-- Midpoints of sides \(BC\), \(CA\), and \(AB\)
variables {D E F : Point}
-- \(D, E, F\) are midpoints
axiom midpoint_D : midpoint D B C
axiom midpoint_E : midpoint E C A
axiom midpoint_F : midpoint F A B
-- \(X\) is an interior point such that \(\angle AEX = \angle AFX\)
variables {X : Point}
axiom angle_AEX_eq_AFX : angle A E X = angle A F X
-- \(P\) is the intersection of \(AX\) and the circumcircle of \(\triangle ABC\)
variables {P : Point}
axiom intersection_AX_circumcircle : intersect_circle_ax A X P O (circumcircle A B C)

-- The proof problem statement
theorem points_XODP_concyclic :
  concyclic X O D P :=
sorry

end points_XODP_concyclic_l419_419241


namespace max_values_of_f_inequality_l419_419418

def f (x : ℝ) : ℝ := (x^2 + 1)^2 / (x * (x^2 - 1))

theorem max_values_of_f_inequality (a : ℝ) :
  |a| > 4 ↔ ∃ x : ℝ, f x = a := sorry

end max_values_of_f_inequality_l419_419418


namespace Henry_added_water_l419_419522

theorem Henry_added_water
  (initial_fullness : ℚ) (final_fullness : ℚ) (tank_capacity : ℚ) (initial_volume : ℚ) (final_volume : ℚ) : 
  initial_fullness = 3 / 4 → final_fullness = 7 / 8 → tank_capacity = 64 →
  initial_volume = initial_fullness * tank_capacity → final_volume = final_fullness * tank_capacity → 
  final_volume - initial_volume = 8 := 
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  rw [h4, h5]
  norm_num
  sorry

end Henry_added_water_l419_419522


namespace integer_averages_equal_or_alternating_l419_419997

theorem integer_averages_equal_or_alternating {n : ℕ} (a : fin n → ℤ) 
  (h : ∀ i : fin n, (a i + a ((i + 1) % n)) / 2 ∈ ℤ) :
  (∀ i j, a i = a j) ∨ (∀ i, a i = a ((i + 2) % n)) :=
sorry

end integer_averages_equal_or_alternating_l419_419997


namespace octagon_diagonals_l419_419588

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l419_419588


namespace gcd_a2_13a_36_a_6_eq_6_l419_419490

namespace GCDProblem

variable (a : ℕ)
variable (h : ∃ k, a = 1632 * k)

theorem gcd_a2_13a_36_a_6_eq_6 (ha : ∃ k : ℕ, a = 1632 * k) : 
  Int.gcd (a^2 + 13 * a + 36 : Int) (a + 6 : Int) = 6 := by
  sorry

end GCDProblem

end gcd_a2_13a_36_a_6_eq_6_l419_419490


namespace restaurant_june_production_l419_419402

-- Define the given conditions
def daily_hot_dogs := 60
def daily_pizzas := daily_hot_dogs + 40
def june_days := 30
def daily_total := daily_hot_dogs + daily_pizzas
def june_total := daily_total * june_days

-- The goal is to prove that the total number of pizzas and hot dogs made in June is 4800
theorem restaurant_june_production : june_total = 4800 := by
  -- Sorry to skip proof
  sorry

end restaurant_june_production_l419_419402


namespace find_values_of_a_and_b_f_is_increasing_solution_set_of_inequality_l419_419496

-- Given conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (3^x + b) / (3^x + a)

-- Part (1): Prove the values of a and b
theorem find_values_of_a_and_b  (h_odd : ∀ x : ℝ, f x b a = -f (-x) b a) :
  a = 1 ∧ b = -1 :=
sorry

-- Part (2): Prove that f(x) is increasing
theorem f_is_increasing (a b : ℝ) (h_odd : ∀ x : ℝ, f x b a = -f (-x) b a) (ha : a = 1) (hb : b = -1) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ b a < f x₂ b a :=
sorry

-- Part (3): Prove the solution set of the given inequality
theorem solution_set_of_inequality (a b : ℝ) (h_odd : ∀ x : ℝ, f x b a = -f (-x) b a) (ha : a = 1) (hb : b = -1) :
  {x : ℝ | f (Real.log (3 - x) / Real.log 2) b a + f ((1 / 3) * Real.log (3 - x) / Real.log 2 - (2 / 3)) b a ≥ 0} = set.Ico (5 / 2) 3 :=
sorry

end find_values_of_a_and_b_f_is_increasing_solution_set_of_inequality_l419_419496


namespace total_bananas_l419_419804

theorem total_bananas (W C L We Ce Le : ℕ)
  (hW : W = 48) (hC : C = 35) (hL : L = 27)
  (hWe : We = 12) (hCe : Ce = 14) (hLe : Le = 5) :
  W + C + L = 110 :=
by
  rw [hW, hC, hL]
  exact rfl

end total_bananas_l419_419804


namespace number_of_restaurants_l419_419521

def first_restaurant_meals_per_day := 20
def second_restaurant_meals_per_day := 40
def third_restaurant_meals_per_day := 50
def total_meals_per_week := 770

theorem number_of_restaurants :
  (first_restaurant_meals_per_day * 7) + 
  (second_restaurant_meals_per_day * 7) + 
  (third_restaurant_meals_per_day * 7) = total_meals_per_week → 
  3 = 3 :=
by 
  intros h
  sorry

end number_of_restaurants_l419_419521


namespace isosceles_triangle_circumcircle_area_l419_419863

noncomputable def area_of_circumcircle (a b c : ℝ) : ℝ :=
  let BD := real.sqrt (a^2 - ((c / 2)^2))
  let OD := (2 / 3) * BD
  let r := real.sqrt (a^2 - OD^2)
  real.pi * r^2

theorem isosceles_triangle_circumcircle_area :
  area_of_circumcircle 4 4 3 = 9.8889 * real.pi :=
sorry

end isosceles_triangle_circumcircle_area_l419_419863


namespace ant_minimum_travel_time_l419_419730

-- Conditions from the problem
def block_length : ℝ := 12
def block_width : ℝ := 4
def block_height : ℝ := 2
def speed_ascending : ℝ := 2
def speed_descending : ℝ := 3
def speed_horizontal : ℝ := 4

-- Given and conditions
noncomputable def travel_time (d1 d2 d3 d4 d5 d6 d7 : ℝ) : ℝ :=
  (d1 / speed_ascending) + (d2 / speed_horizontal) + (d3 / speed_descending) +
  (d4 / speed_horizontal) + (d5 / speed_ascending) + (d6 / speed_horizontal) +
  (d7 / speed_descending)

-- Specific distances for each segment from the solution steps
def d1 : ℝ := block_height
def d2 : ℝ := 5 * block_width
def d3 : ℝ := block_height
def d4 : ℝ := 2 * block_length
def d5 : ℝ := block_height
def d6 : ℝ := 5 * block_width
def d7 : ℝ := block_height

-- Target value to prove
def minimal_time : ℝ := (2 + (4 / 3) + 16 : ℝ)

-- Main theorem statement
theorem ant_minimum_travel_time : travel_time d1 d2 d3 d4 d5 d6 d7 = minimal_time :=
by
  -- proof steps would be here
  sorry

end ant_minimum_travel_time_l419_419730


namespace angle_APB_is_60_degrees_l419_419062

-- Define the given conditions
def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 = 0

def is_tangent_to_circle (A : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = r^2

def line_perpendicular (P : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  let CP := (P.1 - C.1, P.2 - C.2)
  in CP.1 + CP.2 = 0

-- Define the coordinates of the circle's center and the radius
def center : ℝ × ℝ := (-1, 5)
def radius : ℝ := real.sqrt 2

-- Proof problem statement
theorem angle_APB_is_60_degrees (P A B : ℝ × ℝ) 
    (h1 : point_on_line P) 
    (h2 : is_tangent_to_circle A center radius) 
    (h3 : is_tangent_to_circle B center radius)
    (h4 : line_perpendicular P center):
    angle A P B = 60 :=
sorry

end angle_APB_is_60_degrees_l419_419062


namespace profit_percent_is_25_l419_419215

noncomputable def SP : ℝ := sorry
noncomputable def CP : ℝ := 0.80 * SP
noncomputable def Profit : ℝ := SP - CP
noncomputable def ProfitPercent : ℝ := (Profit / CP) * 100

theorem profit_percent_is_25 :
  ProfitPercent = 25 :=
by
  sorry

end profit_percent_is_25_l419_419215


namespace triangle_area_is_integer_l419_419764

theorem triangle_area_is_integer (x1 x2 x3 y1 y2 y3 : ℤ) 
  (hx_even : (x1 + x2 + x3) % 2 = 0) 
  (hy_even : (y1 + y2 + y3) % 2 = 0) : 
  ∃ k : ℤ, 
    abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) = 2 * k := 
sorry

end triangle_area_is_integer_l419_419764


namespace isosceles_right_triangle_dot_product_l419_419233

theorem isosceles_right_triangle_dot_product 
  (A B C D : Type) 
  [vector_space ℝ A] 
  (AB AC AD : A)
  (h1 : is_isosceles_right_triangle A B C)
  (h2 : is_midpoint D B C)
  (h3 : ∥AB∥ = 2)
  (h4 : ∥AC∥ = 2)
  (h5 : angle D A B = π / 4)
  (h6 : ∥AD∥ = sqrt 2) :
  (AB + AC) • AD = 4 :=
sorry

end isosceles_right_triangle_dot_product_l419_419233


namespace line_through_point_with_angle_l419_419732

theorem line_through_point_with_angle :
  ∀ (k : ℝ) (x y : ℝ),
    let P := (-1 : ℝ, real.sqrt 3)
    let θ := real.pi / 6
    let L₁ := (λ x : ℝ, real.sqrt 3 * x - 1)
    let L₀ := (λ x : ℝ, k * x + real.sqrt 3)
    ((∀ (x y : ℝ), y - real.sqrt 3 = k * (x + 1)) ∨
     (Λx : ℝ, x = -1)) :=
  sorry

end line_through_point_with_angle_l419_419732


namespace original_mixture_litres_l419_419033

theorem original_mixture_litres 
  (x : ℝ)
  (h1 : 0.20 * x = 0.15 * (x + 5)) :
  x = 15 :=
sorry

end original_mixture_litres_l419_419033


namespace slope_angle_range_of_tangent_line_l419_419748

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - x^2

noncomputable def f' (x : ℝ) : ℝ := x^2 - 2 * x

theorem slope_angle_range_of_tangent_line :
  (∃ α : ℝ, 0 ≤ α ∧ α < π ∧ tan α = f' x) ↔ 
    (0 ≤ α ∧ α < π / 2) ∨ (3 * π / 4 ≤ α ∧ α < π) :=
  sorry

end slope_angle_range_of_tangent_line_l419_419748


namespace octagon_has_20_diagonals_l419_419581

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l419_419581


namespace tan_alpha_value_l419_419149

theorem tan_alpha_value (α : ℝ) (h : (sin α - cos α) / (2 * sin α + 3 * cos α) = 1 / 5) : 
  tan α = 8 / 3 :=
by
  sorry

end tan_alpha_value_l419_419149


namespace soldiers_arrival_time_l419_419769

open Function

theorem soldiers_arrival_time
    (num_soldiers : ℕ) (distance : ℝ) (car_speed : ℝ) (car_capacity : ℕ) (walk_speed : ℝ) (start_time : ℝ) :
    num_soldiers = 12 →
    distance = 20 →
    car_speed = 20 →
    car_capacity = 4 →
    walk_speed = 4 →
    start_time = 0 →
    ∃ arrival_time, arrival_time = 2 + 36/60 :=
by
  intros
  sorry

end soldiers_arrival_time_l419_419769


namespace area_of_circumcircle_of_isosceles_triangle_l419_419831

theorem area_of_circumcircle_of_isosceles_triangle :
  let AB := 4
  let AC := 4
  let BC := 3
  let AD := (√(AB^2 - (BC/2)^2))
  let radius := AD
  let area := π * radius^2
  area = 16 * π :=
by
  sorry

end area_of_circumcircle_of_isosceles_triangle_l419_419831


namespace sum_and_average_of_divisors_of_30_l419_419428

-- Define the problem as verifying the sum and average of pozitivie divisors of 30.

def positive_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n + 1))

def sum_divisors (n : ℕ) : ℕ :=
  (positive_divisors n).sum

def average_divisors (n : ℕ) : ℤ :=
  let sum := sum_divisors n
  let count := (positive_divisors n).length
  sum.toNat / count

theorem sum_and_average_of_divisors_of_30 :
  sum_divisors 30 = 72 ∧ average_divisors 30 = 9 := 
by
  sorry

end sum_and_average_of_divisors_of_30_l419_419428


namespace fish_truck_cod_pieces_l419_419878

theorem fish_truck_cod_pieces (total_fish : ℕ) (haddock_percent : ℝ) (halibut_percent : ℝ) 
  (haddock_sum : haddock_percent = 0.4) (halibut_sum: halibut_percent = 0.4) (total_fish_sold: total_fish = 220) :
  let cod_percent := 1 - haddock_percent - halibut_percent in
  let cod_pieces := cod_percent * total_fish in
  cod_pieces = 44 :=
sorry

end fish_truck_cod_pieces_l419_419878


namespace chloe_points_first_round_l419_419430

theorem chloe_points_first_round 
  (P : ℕ)
  (second_round_points : ℕ := 50)
  (lost_points : ℕ := 4)
  (total_points : ℕ := 86)
  (h : P + second_round_points - lost_points = total_points) : 
  P = 40 := 
by 
  sorry

end chloe_points_first_round_l419_419430


namespace roots_real_and_equal_l419_419945

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem roots_real_and_equal (a k : ℝ) (ha : a > 0) (hk : k > 0)
  (hD : discriminant a (-6 * real.sqrt k) (18 * k) = 0) :
  ∃ x : ℝ, x = 6 * real.sqrt k :=
by
  sorry

end roots_real_and_equal_l419_419945


namespace tendsto_in_prob_zero_l419_419695

open ProbabilityTheory MeasureTheory Filter

-- Assume we already have definitions from the mathlib for non-negative random variables and sigma algebras

noncomputable def nonneg_random_vars (Ω : Type) (ℱ : σ-Algebra Ω) : Type := sorry
noncomputable def sigma_algebras (Ω : Type) (ℱ : σ-Algebra Ω) : Type := sorry

theorem tendsto_in_prob_zero 
  {Ω : Type} 
  {ℱ : Type} 
  [measure_space Ω]
  (ξ : ℕ → nonneg_random_vars Ω ℱ)
  (𝓕 : ℕ → sigma_algebras Ω ℱ)
  (h : ∀ n, 𝓕 n) 
  (h_cond : ∀ ε > 0, Tendsto (λ n, P(ξ n > ε)) at_top (𝓝 0)) :
  Tendsto (λ n, ξ n) at_top (𝓝 0) :=
sorry

end tendsto_in_prob_zero_l419_419695


namespace scientific_notation_of_3300000000_l419_419926

theorem scientific_notation_of_3300000000 :
  3300000000 = 3.3 * 10^9 :=
sorry

end scientific_notation_of_3300000000_l419_419926


namespace area_of_circle_passing_through_vertices_l419_419848

noncomputable def circle_area_through_isosceles_triangle_vertices 
  (a b c : ℝ) (h_isosceles: (a = b) (h_sides: a = 4) (h_base: c = 3) : ℝ :=
π *(√((4^2 - (3/2)^2)/2 + (3/2))^2

theorem area_of_circle_passing_through_vertices :
  circle_area_through_isosceles_triangle_vertices 4 4 3 = 5.6875 * π :=
sorry

end area_of_circle_passing_through_vertices_l419_419848


namespace range_of_a_l419_419192

variable {a : ℝ}

theorem range_of_a (h : ∃ x ∈ set.Icc 1 2, x^2 + 2 * x + a ≥ 0) : -8 ≤ a :=
sorry

end range_of_a_l419_419192


namespace starting_lineup_ways_l419_419887

-- Define the conditions
def soccer_team_size : ℕ := 16
def quadruplets_count : ℕ := 4
def chosen_players : ℕ := 7
def max_quadruplets : ℕ := 2

-- Main theorem stating the problem
theorem starting_lineup_ways :
  ∑ (k in finset.range (max_quadruplets + 1)), (nat.choose quadruplets_count k) * (nat.choose (soccer_team_size - quadruplets_count) (chosen_players - k)) = 9240 :=
sorry

end starting_lineup_ways_l419_419887


namespace function_increasing_l419_419633

noncomputable def f (a x : ℝ) := x^2 + a * x + 1 / x

theorem function_increasing (a : ℝ) :
  (∀ x, (1 / 3) < x → 0 ≤ (2 * x + a - 1 / x^2)) → a ≥ 25 / 3 :=
by
  sorry

end function_increasing_l419_419633


namespace inequality_solution_l419_419219

theorem inequality_solution (a b : ℝ)
  (h₁ : ∀ x, - (1 : ℝ) / 2 < x ∧ x < (1 : ℝ) / 3 → ax^2 + bx + (2 : ℝ) > 0)
  (h₂ : - (1 : ℝ) / 2 = -(b / a))
  (h₃ : (- (1 : ℝ) / 6) = 2 / a) :
  a - b = -10 :=
sorry

end inequality_solution_l419_419219


namespace find_length_ad_l419_419665

theorem find_length_ad
  (A B C D X : Type)
  (angle_bad : ℝ) (angle_abc : ℝ) (angle_bcd : ℝ)
  (ab : ℝ) (cd : ℝ)
  (AX DX AD : ℝ)
  (h1 : angle_bad = 60)
  (h2 : angle_abc = 30)
  (h3 : angle_bcd = 30)
  (h4 : ab = 15)
  (h5 : cd = 8)
  (h6 : AX = ab / 2)
  (h7 : DX = cd / 2)
  (h8: AD = AX - DX) :
  AD = 3.5 :=
begin
  sorry
end

end find_length_ad_l419_419665


namespace octagon_has_20_diagonals_l419_419578

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l419_419578


namespace complement_intersection_l419_419195

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = {1, 2, 3})
variable (hB : B = {2, 3, 4})

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 4} :=
by
  sorry

end complement_intersection_l419_419195


namespace mass_percentage_Al_in_mixture_is_correct_l419_419442

-- Define the atomic masses
def mass_Al := 26.98 -- g/mol
def mass_S  := 32.06 -- g/mol
def mass_Ca := 40.08 -- g/mol
def mass_C  := 12.01 -- g/mol
def mass_O  := 16.00 -- g/mol
def mass_K  := 39.10 -- g/mol
def mass_Cl := 35.45 -- g/mol

-- Define the molar masses of compounds
def molar_mass_Al2S3 := 2 * mass_Al + 3 * mass_S
def molar_mass_CaCO3 := mass_Ca + mass_C + 3 * mass_O
def molar_mass_KCl   := mass_K + mass_Cl

-- Define the number of moles in the mixture
def moles_Al2S3 := 2
def moles_CaCO3 := 3
def moles_KCl   := 5

-- Compute total mass of each compound in the mixture
def mass_Al2S3 := moles_Al2S3 * molar_mass_Al2S3
def mass_CaCO3 := moles_CaCO3 * molar_mass_CaCO3
def mass_KCl   := moles_KCl * molar_mass_KCl

-- Compute total mass of the mixture
def total_mass_mixture := mass_Al2S3 + mass_CaCO3 + mass_KCl

-- Compute total mass of Al in the mixture
def moles_Al := 2 * moles_Al2S3
def mass_Al_total := moles_Al * mass_Al

-- Compute the mass percentage of Al in the mixture
def mass_percentage_Al := (mass_Al_total / total_mass_mixture) * 100

theorem mass_percentage_Al_in_mixture_is_correct :
  mass_percentage_Al = 11.09 := by
  sorry

end mass_percentage_Al_in_mixture_is_correct_l419_419442


namespace batsman_11th_inning_score_l419_419025

noncomputable def batsman_score (A : ℤ) (score_11th_inning : ℤ) : Prop := 
  let total_runs_10_innings := 10 * A in
  let total_runs_11_innings := total_runs_10_innings + score_11th_inning in
  total_runs_11_innings / 11 = 60 ∧ total_runs_10_innings / 10 = A

theorem batsman_11th_inning_score (A : ℤ) (score_11th_inning : ℤ) 
  (hA : A = 60 - 5) 
  (h_score : batsman_score A score_11th_inning) : 
  score_11th_inning = 110 :=
by 
  obtain ⟨h_avg_11, h_avg_10⟩ := h_score
  sorry

end batsman_11th_inning_score_l419_419025


namespace students_candies_shared_l419_419425

theorem students_candies_shared (students candies : ℕ) (h_students : students = 15) (h_candies : candies = 100) :
  ∃ (a : ℕ → ℕ), (∀ i j : ℕ, i < students → j < students → i ≠ j → a i ≠ a j) → (∑ i in finset.range students, a i) > candies :=
by
  intros
  use sorry

end students_candies_shared_l419_419425


namespace find_x_l419_419615

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end find_x_l419_419615


namespace octagon_diagonals_l419_419535

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l419_419535


namespace angle_A_is_30_l419_419164

theorem angle_A_is_30 (a b : ℝ) (B : ℝ) (h₁ : a = 1) (h₂ : b = Real.sqrt 2) (h₃ : B = 45) :
    ∃ A : ℝ, A = 30 :=
begin
  sorry
end

end angle_A_is_30_l419_419164


namespace minimize_quadrilateral_area_l419_419166

noncomputable def l1 (k : ℝ) (x y : ℝ) : Prop := k * x - 2 * y - 2 * k + 8 = 0
noncomputable def l2 (k : ℝ) (x y : ℝ) : Prop := 2 * x + k^2 * y - 4 * k^2 - 4 = 0

theorem minimize_quadrilateral_area (k : ℝ) (h1 : 0 < k) (h2 : k < 4) :
  minimize (area (quadrilateral_with_axes (l1 k) (l2 k))) k :=
by
  sorry

end minimize_quadrilateral_area_l419_419166


namespace constant_remainder_of_division_l419_419443

theorem constant_remainder_of_division (a : ℤ) :
  let p := 12 * Polynomial.C (Polynomials.of_int 1) * Polynomial.X^3
          - 9 * Polynomial.C (Polynomials.of_int 1) * Polynomial.X^2
          + a * Polynomial.C (Polynomials.of_int 1) * Polynomial.X
          + 8 * Polynomial.C (Polynomials.of_int 1),
      q := 3 * Polynomial.C (Polynomials.of_int 1) * Polynomial.X^2
          - 4 * Polynomial.C (Polynomials.of_int 1) * Polynomial.X
          + 2 * Polynomial.C (Polynomials.of_int 1)
  in (Polynomial.euclidean_domain.mod p q).degree = 0 → a = -7 :=
by
  intros p q h
  sorry

end constant_remainder_of_division_l419_419443


namespace min_value_expression_l419_419942

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom arithmetic_sequence (n : ℕ) : a (n+1) = a n + d
axiom pos_sequence : ∀ n, a n > 0
axiom sum_of_first_n_terms (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom sum_S_2017 : S 2017 = 4034
axiom sum_a1_a2017 : a 1 + a 2017 = 4

theorem min_value_expression : ∃ x : ℝ, x = ∑ y in [1, 9], (if y = 1 then 1 / a 9 else 9 / a 2009) ∧ x = 4 :=
by
  sorry

end min_value_expression_l419_419942


namespace one_over_x_plus_one_over_y_eq_fifteen_l419_419503

theorem one_over_x_plus_one_over_y_eq_fifteen
  (x y : ℝ)
  (h1 : xy > 0)
  (h2 : 1 / xy = 5)
  (h3 : (x + y) / 5 = 0.6) : 
  (1 / x) + (1 / y) = 15 := 
by
  sorry

end one_over_x_plus_one_over_y_eq_fifteen_l419_419503


namespace coloring_ways_l419_419097

-- Define the function that checks valid coloring
noncomputable def valid_coloring (colors : Fin 6 → Fin 3) : Prop :=
  colors 0 = 0 ∧ -- The central pentagon is colored red
  (colors 1 ≠ colors 0 ∧ colors 2 ≠ colors 1 ∧ 
   colors 3 ≠ colors 2 ∧ colors 4 ≠ colors 3 ∧ 
   colors 5 ≠ colors 4 ∧ colors 1 ≠ colors 5) -- No two adjacent polygons have the same color

-- Define the main theorem
theorem coloring_ways (f : Fin 6 → Fin 3) (h : valid_coloring f) : 
  ∃! (f : Fin 6 → Fin 3), valid_coloring f := by
  sorry

end coloring_ways_l419_419097


namespace probability_no_absolute_winner_l419_419916

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end probability_no_absolute_winner_l419_419916


namespace coeffs_sum_eq_40_l419_419486

theorem coeffs_sum_eq_40 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℝ)
  (h : (2 * x - 1) ^ 5 = a_0 * x ^ 5 + a_1 * x ^ 4 + a_2 * x ^ 3 + a_3 * x ^ 2 + a_4 * x + a_5) :
  a_2 + a_3 = 40 :=
sorry

end coeffs_sum_eq_40_l419_419486


namespace length_of_platform_l419_419809

-- Definitions based on the problem conditions
def train_length : ℝ := 300
def platform_crossing_time : ℝ := 39
def signal_pole_crossing_time : ℝ := 18

-- The main theorem statement
theorem length_of_platform : ∀ (L : ℝ), train_length + L = (train_length / signal_pole_crossing_time) * platform_crossing_time → L = 350.13 :=
by
  intro L h
  sorry

end length_of_platform_l419_419809


namespace john_average_speed_proof_l419_419254

def johns_average_speed (start_time end_time : Time) (distance : ℝ) : ℝ :=
  let duration := (end_time - start_time).toHours
  distance / duration

theorem john_average_speed_proof : 
  johns_average_speed ⟨8, 15⟩ ⟨12, 45⟩ 210 = 140 / 3 := by
  sorry

end john_average_speed_proof_l419_419254


namespace num_diagonals_octagon_l419_419571

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l419_419571


namespace circumcircle_area_of_isosceles_triangle_l419_419870

open Real

theorem circumcircle_area_of_isosceles_triangle:
  (∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
    (dist A B = 4) ∧ (dist A C = 4) ∧ (dist B C = 3) →
    (circle_area A B C = (256 / 13.75) * π)) :=
by sorry

end circumcircle_area_of_isosceles_triangle_l419_419870


namespace factor_polynomial_l419_419446

theorem factor_polynomial : 
  (x : ℝ) → (x^2 - 6 * x + 9 - 49 * x^4) = (-7 * x^2 + x - 3) * (7 * x^2 + x - 3) :=
by
  sorry

end factor_polynomial_l419_419446


namespace sin_cos_sum_l419_419501

-- Define hypotheses for the problem
variables {α : ℝ}
def x := 3
def y := -4
def r := Real.sqrt (x^2 + y^2)

-- Lean statement for the problem
theorem sin_cos_sum (cosα : ℝ) (sinα : ℝ) (h_cos : cosα = x / r) (h_sin : sinα = y / r) :
  sinα + cosα = -1 / 5 :=
by
  sorry

end sin_cos_sum_l419_419501


namespace product_of_decimals_l419_419726

theorem product_of_decimals :
  (8 : ℚ) * (1 / 4 : ℚ) * (2 : ℚ) * (1 / 8 : ℚ) = 1 / 2 := by
  sorry

end product_of_decimals_l419_419726


namespace isosceles_triangle_circle_area_l419_419826

theorem isosceles_triangle_circle_area 
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : a = 4) 
  (h3 : c = 3) 
  (h4 : a = 4) 
  (h5 : b = 4)
  (h6 : c ≠ a)
  (h7 : c ≠ b) :
  let r := 4 in π * r ^ 2 = 16 * π :=
by
  sorry

end isosceles_triangle_circle_area_l419_419826


namespace proof_problem_l419_419280

theorem proof_problem 
  (a b : ℝ) 
  (h₁ : a = 2003^1004 - 2003^(-1004)) 
  (h₂ : b = 2003^1004 + 2003^(-1004)) : 
  a ^ 2 - b ^ 2 = -4 :=
sorry

end proof_problem_l419_419280


namespace min_sum_of_dimensions_l419_419311

theorem min_sum_of_dimensions (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 3003) :
  a + b + c = 45 := sorry

end min_sum_of_dimensions_l419_419311


namespace number_of_diagonals_in_octagon_l419_419530

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l419_419530


namespace surface_area_of_sphere_from_rectangular_solid_l419_419739

theorem surface_area_of_sphere_from_rectangular_solid :
  ∀ (length width height : ℝ), 
    length = 3 → 
    width = 2 → 
    height = 1 →
    ∃ (r : ℝ), 
      2 * r = real.sqrt (length^2 + width^2 + height^2) ∧
      4 * real.pi * r^2 = 14 * real.pi :=
by 
  intros length width height hlength hwidth hheight
  have h_body_diagonal : real.sqrt (length^2 + width^2 + height^2) = real.sqrt 14,
  { rw [hlength, hwidth, hheight], norm_num, }
  use real.sqrt 14 / 2,
  split,
  { rw [←h_body_diagonal], ring, },
  { field_simp, norm_num, ring, }

end surface_area_of_sphere_from_rectangular_solid_l419_419739


namespace diagonals_of_octagon_l419_419544

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l419_419544


namespace circumcircle_area_of_isosceles_triangle_l419_419868

open Real

theorem circumcircle_area_of_isosceles_triangle:
  (∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
    (dist A B = 4) ∧ (dist A C = 4) ∧ (dist B C = 3) →
    (circle_area A B C = (256 / 13.75) * π)) :=
by sorry

end circumcircle_area_of_isosceles_triangle_l419_419868


namespace solve_eq_l419_419455

theorem solve_eq (x : ℝ) : (∛(5 - x / 3) = -4) → x = 207 :=
by {
    intro hyp,
    -- proof steps here
    sorry
}

end solve_eq_l419_419455


namespace no_absolute_winner_probability_l419_419917

-- Define the probabilities of matches
def P_AB : ℝ := 0.6  -- Probability Alyosha wins against Borya
def P_BV : ℝ := 0.4  -- Probability Borya wins against Vasya

-- Define the event C that there is no absolute winner
def event_C (P_AV : ℝ) (P_VB : ℝ) : ℝ :=
  let scenario1 := P_AB * P_BV * P_AV in
  let scenario2 := P_AB * P_VB * (1 - P_AV) in
  scenario1 + scenario2

-- Main theorem to prove
theorem no_absolute_winner_probability : 
  event_C 1 0.6 = 0.24 :=
by
  rw [event_C]
  simp
  norm_num
  sorry

end no_absolute_winner_probability_l419_419917


namespace points_per_bag_l419_419340

/-
Wendy had 11 bags but didn't recycle 2 of them. She would have earned 
45 points for recycling all 11 bags. Prove that Wendy earns 5 points 
per bag of cans she recycles.
-/

def total_bags : Nat := 11
def unrecycled_bags : Nat := 2
def recycled_bags : Nat := total_bags - unrecycled_bags
def total_points : Nat := 45

theorem points_per_bag : total_points / recycled_bags = 5 := by
  sorry

end points_per_bag_l419_419340


namespace find_f_4_l419_419610

noncomputable def f (x : ℕ) (a b c : ℕ) : ℕ := 2 * a * x + b * x + c

theorem find_f_4
  (a b c : ℕ)
  (f1 : f 1 a b c = 10)
  (f2 : f 2 a b c = 20) :
  f 4 a b c = 40 :=
sorry

end find_f_4_l419_419610


namespace midpoint_lemma_l419_419969

def point := (ℝ × ℝ × ℝ)

def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

theorem midpoint_lemma : 
  midpoint (2, -4, 7) (8, 0, 5) = (5, -2, 6) :=
by 
  sorry

end midpoint_lemma_l419_419969


namespace slopes_sum_constant_l419_419504

variables 
  (a b x y : ℝ)
  (N P A B : ℝ × ℝ)
  (k₁ k₂ : ℝ)
  (e : ℝ := (Math.sqrt 2) /2)
  (C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1})
  -- Define conditions
  (h1 : a = Math.sqrt 8)
  (h2 : b = Math.sqrt 4)
  (h3 : ∀ (p : ℝ × ℝ), p ∈ C ↔ 
    (p.1^2 / (Math.sqrt 8)^2 + p.2^2 / (Math.sqrt 4)^2) = 1)
  (h4 : N = (0, 2))
  (h5 : P = (-1, -2))
  (h6 : ∀ A B, A ∈ C ∧ B ∈ C ∧ A ≠ N ∧ B ≠ N → (k₁ k₂ : ℝ))

theorem slopes_sum_constant 
  (hA : A ∈ C) (hB : B ∈ C)
  (hA_not_N : A ≠ N) (hB_not_N : B ≠ N) : 
  k₁ + k₂ = 4 :=
  sorry

end slopes_sum_constant_l419_419504


namespace count_congruent_mod_7_in_250_l419_419597

theorem count_congruent_mod_7_in_250 :
  ∃ n, n = 36 ∧ ∀ x ∈ (Finset.range 250).map (λ x, x+1), x % 7 = 1 → ∃ k, x = 7 * k + 1 :=
by
  sorry

end count_congruent_mod_7_in_250_l419_419597


namespace circumcircle_area_of_isosceles_triangle_l419_419871

open Real

theorem circumcircle_area_of_isosceles_triangle:
  (∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
    (dist A B = 4) ∧ (dist A C = 4) ∧ (dist B C = 3) →
    (circle_area A B C = (256 / 13.75) * π)) :=
by sorry

end circumcircle_area_of_isosceles_triangle_l419_419871


namespace no_generating_combination_l419_419874

-- Representing Rubik's Cube state as a type (assume a type exists)
axiom CubeState : Type

-- A combination of turns represented as a function on states
axiom A : CubeState → CubeState

-- Simple rotations
axiom P : CubeState → CubeState
axiom Q : CubeState → CubeState

-- Rubik's Cube property of generating combination (assuming generating implies all states achievable)
def is_generating (A : CubeState → CubeState) :=
  ∀ X : CubeState, ∃ m n : ℕ, P X = A^[m] X ∧ Q X = A^[n] X

-- Non-commutativity condition
axiom non_commutativity : ∀ X : CubeState, P (Q X) ≠ Q (P X)

-- Formal statement of the problem
theorem no_generating_combination : ¬ ∃ A : CubeState → CubeState, is_generating A :=
by sorry

end no_generating_combination_l419_419874


namespace purely_imaginary_z_abs_one_l419_419173

noncomputable def complex_z (a : ℝ) : ℂ := (a + complex.I) / (1 + complex.I)

theorem purely_imaginary_z_abs_one (a : ℝ) (h : complex_z a = complex.I * (real.to_complex (1 - a) / 2)) : abs (complex_z a) = 1 :=
by
  sorry

end purely_imaginary_z_abs_one_l419_419173


namespace circle_area_isosceles_triangle_l419_419841

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end circle_area_isosceles_triangle_l419_419841


namespace num_valid_two_digit_numbers_l419_419753

def isValid (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (n - (a + b)) % 3 = 0

theorem num_valid_two_digit_numbers : (finset.filter isValid (finset.range 100)).card = 90 := by
  sorry

end num_valid_two_digit_numbers_l419_419753


namespace run_rate_remaining_l419_419362

variable (run_rate_first_10_overs : ℝ) (target_runs : ℝ)
variable (overs_first_segment : ℕ) (remaining_overs : ℕ)

-- Define the conditions based on the problem
def conditions : Prop :=
  run_rate_first_10_overs = 3.2 ∧
  target_runs = 272 ∧
  overs_first_segment = 10 ∧
  remaining_overs = 40

-- Prove the required run rate for the remaining 40 overs
theorem run_rate_remaining (h : conditions) : 
  (target_runs - run_rate_first_10_overs * overs_first_segment) / remaining_overs = 6 := by
  sorry

end run_rate_remaining_l419_419362


namespace slope_angle_range_l419_419629

open Real

theorem slope_angle_range (m : ℝ) (θ : ℝ) (h0 : 0 ≤ θ) (h1 : θ < π) 
    (hslope : tan θ = 1 - m^2) : θ ∈ set.Icc 0 (π / 4) ∪ set.Ioo (π / 2) π :=
by
  -- We are not providing the proof, so adding sorry
  sorry

end slope_angle_range_l419_419629


namespace find_angle_BAC_l419_419239

-- Define the angles as constants
constant angle_ACE : ℝ := 100
constant angle_ADC : ℝ := 110
constant angle_ACB : ℝ := 70
constant angle_ABC : ℝ := 50

-- Define the parallel lines relation
constant AB_parallel_DC : Prop

-- Define the angles in the triangle ABC
constant angle_BAC : ℝ

-- Define the problem statement as a theorem to be proved
theorem find_angle_BAC 
(h1 : AB_parallel_DC) 
(h2 : angle_ACE = 100) 
(h3 : angle_ADC = 110) 
(h4 : angle_ACB = 70) 
(h5 : angle_ABC = 50) : 
angle_BAC = 60 := 
sorry

end find_angle_BAC_l419_419239


namespace octagon_diagonals_l419_419539

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l419_419539


namespace incenter_of_triangle_CEF_l419_419375

-- Definitions based on given conditions
variables {Γ : Type} [MetricSpace Γ] [NormedAddCommGroup Γ] [NormedSpace ℝ Γ]
variables {A B C D E F I O : Γ}
variable (circle : ∀ {X Y : Γ}, CircularArc O Y X Y)
variable (is_diameter : BC radius circle)
variable (on_circle : A ∈ circle)
variable (angle_condition : 0 < ∠AOB ∧ ∠AOB < 120 * (π / 180))
variable (midpoint : Midpoint D AB ∧ D ∉ C)
variable (parallel_condition : ParallelThru O DA intersects AC I)
variable (perpendicular_bisector : PerpendicularBisector O A intersect_circle E F)

-- Statement to be proven
theorem incenter_of_triangle_CEF :
  IsIncenter I (Triangle C E F) :=
sorry

end incenter_of_triangle_CEF_l419_419375


namespace log_equation_solution_l419_419602

theorem log_equation_solution (x : ℝ) (h : log 4 (log 3 x) = 1) : x = 81 := 
by 
  sorry

end log_equation_solution_l419_419602


namespace gazprom_rd_expense_l419_419935

theorem gazprom_rd_expense
  (R_and_D_t : ℝ) (ΔAPL_t_plus_1 : ℝ)
  (h1 : R_and_D_t = 3289.31)
  (h2 : ΔAPL_t_plus_1 = 1.55) :
  R_and_D_t / ΔAPL_t_plus_1 = 2122 := 
by
  sorry

end gazprom_rd_expense_l419_419935


namespace intersection_count_l419_419318

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_count : ∃! (x1 x2 : ℝ), 
  x1 > 0 ∧ x2 > 0 ∧ f x1 = g x1 ∧ f x2 = g x2 ∧ x1 ≠ x2 :=
sorry

end intersection_count_l419_419318


namespace CP_perpendicular_AM_l419_419481

-- Given conditions
def center_O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (-2, 0)
def B : (ℝ × ℝ) := (2, 0)
def C : (ℝ × ℝ) := (1, 0)

-- Define points M and P
def M (x y : ℝ) : (ℝ × ℝ) := (x, y)
def P (x y : ℝ) : (ℝ × ℝ) := (x, -y)

-- Check that CP is perpendicular to AM
theorem CP_perpendicular_AM (x y : ℝ) :
  let AM := ((x + 2, y) : ℝ × ℝ),
      CP := ((x - 1, -y) : ℝ × ℝ) in
  AM.1 * CP.1 + AM.2 * CP.2 = 0 :=
by
  sorry

end CP_perpendicular_AM_l419_419481


namespace problem_l419_419684

theorem problem
  (g : ℕ → ℕ)
  (h : ∀ a b : ℕ, 2 * g (a^2 + b^2) = (g a)^2 + (g b)^2):
  let m := {(x : ℕ) | (x = g 16)}.size
  ∧ let t := ∑ x in {(x : ℕ) | (x = g 16)}, x
  in m * t = 387 := by
  sorry

end problem_l419_419684


namespace range_of_expression_l419_419186

noncomputable def f (x : ℝ) : ℝ := 2 + real.log x / real.log 3

theorem range_of_expression :
  ∀ x ∈ set.Icc 1 9, 6 ≤ (f x)^2 + f (x^2) ∧ (f x)^2 + f (x^2) ≤ 22 :=
by
  intro x hx
  sorry

end range_of_expression_l419_419186


namespace cory_needs_additional_money_l419_419438

theorem cory_needs_additional_money :
  ∀ (price_per_pack : ℝ) (number_of_packs : ℕ) (money_cory_has : ℝ), 
  price_per_pack = 49 → 
  number_of_packs = 2 → 
  money_cory_has = 20 → 
  let total_cost := price_per_pack * number_of_packs 
  let additional_money_needed := total_cost - money_cory_has 
  additional_money_needed = 78 :=
by
  intros price_per_pack number_of_packs money_cory_has h_price h_packs h_money_cory_has
  simp [h_price, h_packs, h_money_cory_has]
  have total_cost : total_cost = 49 * 2 := by simp [total_cost]
  have additional_money_needed : additional_money_needed = 49 * 2 - 20 := by simp [additional_money_needed, total_cost]
  exact additional_money_needed

-- Stated the theorem
{ sorry }

end cory_needs_additional_money_l419_419438


namespace normal_distribution_calculation_l419_419198

noncomputable def normal_probability : Prop :=
  let μ : ℝ := 5
  let δ : ℝ := 2
  let X := pdf.normal μ δ in
  𝔼[X] = 5 ∧ Var[X] = 4 ∧ Pr (3 < X ≤ 7) = 0.6826

theorem normal_distribution_calculation : normal_probability :=
by sorry

end normal_distribution_calculation_l419_419198


namespace gcd_of_840_and_1764_l419_419736

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_of_840_and_1764_l419_419736


namespace quadratic_inequalities_not_equiv_l419_419262

noncomputable def A (a₁ b₁ c₁ : ℝ) : set ℝ := 
  {x : ℝ | a₁ * x^2 + b₁ * x + c₁ > 0}

noncomputable def B (a₂ b₂ c₂ : ℝ) : set ℝ :=
  {x : ℝ | a₂ * x^2 + b₂ * x + c₂ > 0}

theorem quadratic_inequalities_not_equiv 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : a₁ ≠ 0) (h₂ : b₁ ≠ 0) (h₃ : c₁ ≠ 0)
  (h₄ : a₂ ≠ 0) (h₅ : b₂ ≠ 0) (h₆ : c₂ ≠ 0)
  (h : a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂) : 
  A a₁ b₁ c₁ ≠ B a₂ b₂ c₂ := 
sorry

end quadratic_inequalities_not_equiv_l419_419262


namespace max_crosses_4x10_proof_impossible_5x10_proof_l419_419358

-- Define the table size and conditions for the number of crosses in each row and column
def table4x10 := (4, 10)
def condition (n m : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) := 
  (∀ i : ℕ, i < n → a i % 2 = 1) ∧ (∀ j : ℕ, j < m → b j % 2 = 1)

-- Define the maximum number of crosses for the 4x10 table
def max_crosses_4x10 := 30

-- State the problem for the 4x10 table
theorem max_crosses_4x10_proof : 
  ∃ a b : ℕ → ℕ, 
    condition 4 10 a b ∧ (∑ i in Finset.range 4, a i) + (∑ j in Finset.range 10, b j) = max_crosses_4x10 :=
sorry

-- Define the table size and conditions for the number of crosses in each row and column
def table5x10 := (5, 10)

-- Prove the impossibility for the 5x10 table
theorem impossible_5x10_proof :
  ¬ ∃ a b : ℕ → ℕ, condition 5 10 a b :=
sorry

end max_crosses_4x10_proof_impossible_5x10_proof_l419_419358


namespace A_neg10_3_eq_neg1320_l419_419487

noncomputable def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else x * A (x - 1) (m - 1)

theorem A_neg10_3_eq_neg1320 : A (-10) 3 = -1320 := 
by
  sorry

end A_neg10_3_eq_neg1320_l419_419487


namespace container_marbles_proportional_l419_419042

theorem container_marbles_proportional (V1 V2 : ℕ) (M1 M2 : ℕ)
(h1 : V1 = 24) (h2 : M1 = 75) (h3 : V2 = 72) (h4 : V1 * M2 = V2 * M1) :
  M2 = 225 :=
by {
  -- Given conditions
  sorry
}

end container_marbles_proportional_l419_419042


namespace sum_of_roots_l419_419611

variable {p m n : ℝ}

axiom roots_condition (h : m * n = 4) : m + n = -4

theorem sum_of_roots (h : m * n = 4) : m + n = -4 := 
  roots_condition h

end sum_of_roots_l419_419611


namespace power_of_prime_calculate_729_to_two_thirds_l419_419090

theorem power_of_prime (a : ℝ) (b : ℝ) (c : ℝ) (h : a = b^c) (e : ℝ) :
  a ^ e = b ^ (c * e) := sorry

theorem calculate_729_to_two_thirds : (729 : ℝ) ^ (2 / 3) = 81 := by
  have h : 729 = (3 : ℝ) ^ 6 := by norm_num
  exact power_of_prime (729 : ℝ) (3 : ℝ) (6 : ℝ) h (2 / 3)

end power_of_prime_calculate_729_to_two_thirds_l419_419090


namespace exists_subset_S_l419_419261

-- Given sets A and B are non-empty finite sets.
variables {A B : Finset ℕ} 
  (hA : A.nonempty) 
  (hB : B.nonempty)

-- Given a non-negative integer n.
variable {n : ℕ} (hn : 0 ≤ n)

-- Main theorem statement.
theorem exists_subset_S (n : ℕ) (hA : A.nonempty) (hB : B.nonempty) (hn : 0 ≤ n) : 
  ∃ S ⊆ (A + B), ∃ S : Finset ℕ, S ⊆ (A + B) ∧ |A + B + (Finset.range (n+1)).sum (λ _, S)| ≤ |A + B| ^ (n + 1) :=
sorry

end exists_subset_S_l419_419261


namespace find_PQ_length_l419_419927

noncomputable theory

open Classical

variables (ABC circ incircle : Triangle) (T P Q : Point)

def is_equilateral (ABC : Triangle) : Prop :=
  ∀ (A B C : Point), dist A B = dist B C ∧ dist B C = dist C A

def is_inscribed (ABC : Triangle) (circ : Circle) : Prop :=
  ∀ vertex ∈ {ABC.vertices}, Circle.contains circ vertex

def is_tangent (incircle : Circle) (line : Line) (tangent_point : Point) : Prop :=
  Circle.tangent_at incircle line tangent_point

def is_tangent_at_point (circ1 circ2 : Circle) (point : Point) : Prop :=
  Circle.contains circ1 point ∧ Circle.contains circ2 point

variables (BC_len : ℝ)
  
axiom eq_triangle_AB (ABC : Triangle) : is_equilateral ABC 
axiom inscribed_ABC (ABC : Triangle) (circ : Circle): is_inscribed ABC circ
axiom tangent_incirc_sides 
  (AB AC : Line) 
  (P Q : Point) 
  (incircle : Circle)  : 
  is_tangent incircle AB P ∧ is_tangent incircle AC Q
axiom tangent_circles 
  (incircle circ : Circle) 
  (T : Point) : 
  is_tangent_at_point incircle circ T
axiom BC_eq_12 (L : IsEquilateral ABC) : dist L.B L.C = BC_len

theorem find_PQ_length :
  ∀ (ABC : Triangle) 
    (circ incircle : Circle) 
    (T P Q : Point), 
    is_equilateral ABC → 
    is_inscribed ABC circ → 
    is_tangent incircle (ABC.side AB) P → 
    is_tangent incircle (ABC.side AC) Q → 
    is_tangent_at_point incircle circ T → 
    (BC_len = 12) → 
    dist P Q = 8 := 
by 
  intros ABC circ incircle T P Q eq_tri ins_circ tang_AB_P tang_AC_Q tang_circles_T BC_is_12
  sorry

end find_PQ_length_l419_419927


namespace area_of_circumscribed_circle_isosceles_triangle_l419_419855

theorem area_of_circumscribed_circle_isosceles_triangle :
  ∃ (r : ℝ), (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi) :=
by
  -- Consider the isosceles triangle conditions
  let a : ℝ := 4
  let b : ℝ := 4
  let c : ℝ := 3
  let BD := Real.sqrt(a^2 - (c/2)^2)
  let r := 8 / BD
  have h1 : BD = Real.sqrt 13.75 := by 
    -- Calculate the altitude BD
    calc
      BD = Real.sqrt(a^2 -  (c/2)^2) : rfl
      ... = Real.sqrt(16 - (3/2)^2) : rfl
      ... = Real.sqrt 13.75 : rfl
  
  use r
  have h2 : r = 8 / Real.sqrt 13.75 := by 
    -- Simplify the radius expression
    sorry

  have h3 : Real.pi * r ^ 2 = 256 / 55 * Real.pi := by 
    -- Calculate the area
    calc
      Real.pi * r ^ 2 = Real.pi * (8 / Real.sqrt 13.75) ^ 2 : by rw h2
      ... = Real.pi * (64 / 13.75) : by rw [pow_two, mul_div_assoc, mul_one, div_mul_div_same]
      ... = (256 / 54.6875) * Real.pi : by rw mul_comm
      ...   = (256 / 55) * Real.pi : by norm_num
    sorry
  
  show (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi),
  from ⟨h2, h3⟩
  sorry

end area_of_circumscribed_circle_isosceles_triangle_l419_419855


namespace green_balls_count_l419_419821

theorem green_balls_count :
  ∀ (total_balls white_balls yellow_balls red_balls purple_balls : ℕ),
    total_balls = 100 →
    white_balls = 10 →
    yellow_balls = 10 →
    red_balls = 47 →
    purple_balls = 3 →
    (50 / total_balls = 0.5) →
    (50 - white_balls - yellow_balls = 30) :=
by
  intros total_balls white_balls yellow_balls red_balls purple_balls
  simp
  assume h1 h2 h3 h4 h5 h6,
  sorry

end green_balls_count_l419_419821


namespace postman_b_returns_first_l419_419774

theorem postman_b_returns_first (v : ℝ) (h : v > 5) :
  let tA := (10 * v) / (v^2 - 25)
  let tB := 10 / v
  tA > tB :=
by
  let tA := (10 * v) / (v^2 - 25)
  let tB := 10 / v
  have : tA > tB := 10 * v > 10 * (v^2 - 25) / v
  sorry

end postman_b_returns_first_l419_419774


namespace salary_increase_after_five_years_l419_419288

theorem salary_increase_after_five_years (S : ℝ) : 
  let final_salary := S * (1.12)^5
  let increase := final_salary - S
  let percent_increase := (increase / S) * 100
  percent_increase = 76.23 :=
by
  let final_salary := S * (1.12)^5
  let increase := final_salary - S
  let percent_increase := (increase / S) * 100
  sorry

end salary_increase_after_five_years_l419_419288


namespace count_three_digit_numbers_with_no_four_with_at_least_one_six_l419_419599

def three_digit_numbers (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def contains_digit (d : ℕ) (n : ℕ) : Prop := 
  (n / 100 = d) ∨ ((n / 10) % 10 = d) ∨ (n % 10 = d)

def no_four_as_digit (n : ℕ) : Prop := 
  ¬ contains_digit 4 n

def at_least_one_six (n : ℕ) : Prop := 
  contains_digit 6 n

theorem count_three_digit_numbers_with_no_four_with_at_least_one_six :
  ∃ (n : ℕ), n = 200 ∧ 
  (∀ k, three_digit_numbers k → no_four_as_digit k → at_least_one_six k → k ∈ finset.range n) := 
sorry

end count_three_digit_numbers_with_no_four_with_at_least_one_six_l419_419599


namespace perimeter_of_semicircles_l419_419400

theorem perimeter_of_semicircles 
  (side_length : ℝ) 
  (h : side_length = 4 / π) : 
  let diameter := side_length in
  let full_circle_circumference := π * diameter in
  let semicircle_perimeter := full_circle_circumference / 2 in
  let total_perimeter := 4 * semicircle_perimeter in
  total_perimeter = 8 :=
by
  sorry

end perimeter_of_semicircles_l419_419400


namespace sum_a_b_c_l419_419750

noncomputable def ratio_of_areas : ℚ := 27 / 50
noncomputable def ratio_of_side_lengths_simplified := 3 * Real.sqrt 6 / 10

theorem sum_a_b_c : (let a := 3 in let b := 6 in let c := 10 in a + b + c) = 19 :=
by
  have h1 : ratio_of_areas = 27 / 50 := rfl
  have h2 : ratio_of_side_lengths_simplified = 3 * Real.sqrt 6 / 10 := rfl
  sorry

end sum_a_b_c_l419_419750


namespace kernel_selects_white_probability_l419_419820

open Probability

def kernel_popping_problem 
  (P_white_kernels : ℝ)
  (P_yellow_kernels : ℝ)
  (P_white_pops : ℝ)
  (P_yellow_pops : ℝ)
  (P_popped : ℝ) : Prop :=
P_white_kernels = 2 / 3 ∧
P_yellow_kernels = 1 / 3 ∧
P_white_pops = 1 / 2 ∧
P_yellow_pops = 2 / 3 ∧
P_popped = (P_white_kernels * P_white_pops) + (P_yellow_kernels * P_yellow_pops) ∧
(P_white_kernels * P_white_pops) / P_popped = 3 / 5

theorem kernel_selects_white_probability :
  kernel_popping_problem (2 / 3) (1 / 3) (1 / 2) (2 / 3) ((2 / 3) * (1 / 2) + (1 / 3) * (2 / 3)) :=
begin
  sorry
end

end kernel_selects_white_probability_l419_419820


namespace jia_initial_speed_distance_A_to_D_l419_419712

noncomputable def initial_speed_Jia := 125
noncomputable def distance_AD := 1880

variables (A B C D : Type)
variables (v_Yi_initial : ℝ) (v_Yi_after_Bing : ℝ)
variables (v_Bing : ℝ) (v_Jia_after_Bing : ℝ)
variables (v_Jia_after_Yi : ℝ) (v_Jia_initial : ℝ)
variables (t_catch_Bing : ℝ) (t_total_catch_Yi : ℝ)
variables (distance_CD : ℝ) (distance_AD_calculated : ℝ)

axiom A1 : v_Yi_initial = 60
axiom A2 : v_Yi_after_Bing = 60 * 0.75
axiom A3 : v_Bing = 45
axiom A4 : v_Jia_after_Bing = 45
axiom A5 : v_Jia_after_Yi = v_Jia_after_Bing / 0.6
axiom A6 : v_Jia_initial = v_Jia_after_Yi / 0.6
axiom A7 : t_catch_Bing = 9
axiom A8 : distance_CD = 75 * 9 + 45 * 9 + 50
axiom A9 : distance_AD_calculated = distance_CD + (v_Jia_initial * 6)

theorem jia_initial_speed : v_Jia_initial = initial_speed_Jia := by
  apply A1
  apply A2
  apply A3
  apply A4
  apply A5
  apply A6
  apply A7
  apply A8
  apply A9
  sorry

theorem distance_A_to_D : distance_AD_calculated = distance_AD := by
  apply A1
  apply A2
  apply A3
  apply A4
  apply A5
  apply A6
  apply A7
  apply A8
  apply A9
  sorry

end jia_initial_speed_distance_A_to_D_l419_419712


namespace lcm_924_660_eq_4620_l419_419784

theorem lcm_924_660_eq_4620 : Nat.lcm 924 660 = 4620 := 
by
  sorry

end lcm_924_660_eq_4620_l419_419784


namespace cos_angle_GAC_is_sqrt3_div_3_l419_419373

-- Define a structure for a cube and its side length
structure Cube :=
  (side_length : ℝ)
  (diagonal_length : ℝ := side_length * sqrt 3)
  (edge_length : ℝ := side_length)

-- Define the cosine function for the angle GAC
def cos_angle_GAC (cube : Cube) : ℝ :=
  cube.edge_length / cube.diagonal_length

-- The theorem to prove
theorem cos_angle_GAC_is_sqrt3_div_3 (cube : Cube) :
  cos_angle_GAC cube = sqrt (3 : ℝ) / 3 := by
  sorry

end cos_angle_GAC_is_sqrt3_div_3_l419_419373


namespace arccos_solution_l419_419723

theorem arccos_solution (x : ℝ) (h : arccos (2 * x) - arccos x = π / 3) : x = -1 / 2 :=
sorry

end arccos_solution_l419_419723


namespace no_absolute_winner_prob_l419_419908

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end no_absolute_winner_prob_l419_419908


namespace xiaohu_fine_l419_419355

theorem xiaohu_fine :
  let pages := 30 - 15 + 1 in
  let sheets := pages / 2 in
  let cost_per_sheet := 16 in
  sheets * cost_per_sheet = 128 :=
by
  let pages := 30 - 15 + 1
  let sheets := pages / 2
  let cost_per_sheet := 16
  calc
    sheets * cost_per_sheet = (30 - 15 + 1) / 2 * 16 : by rfl
    ... = 128 : by norm_num

end xiaohu_fine_l419_419355


namespace number_division_l419_419319

theorem number_division (m k n : ℤ) (h : n = m * k + 1) : n = m * k + 1 :=
by
  exact h

end number_division_l419_419319


namespace octagon_diagonals_20_l419_419554

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l419_419554


namespace partI_partII_l419_419520

def vec_m (x : ℝ) : ℝ × ℝ := (sin x, -1)
def vec_n (x : ℝ) : ℝ × ℝ := (√3 * cos x, -1 / 2)

noncomputable def f (x : ℝ) : ℝ := (vec_m x).1 ^ 2 + (vec_m x).2 ^ 2 + ((vec_m x).fst * (vec_n x).fst + (vec_m x).snd * (vec_n x).snd) - 2

theorem partI : (∀ x, f x ≤ 1) ∧ (∃ k : ℤ, ∀ x, (x = k * π + π / 3) → f x = 1) := sorry

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions for part II
hypothesisB : B = π / 3
hypothesis_sequence : b ^ 2 = a * c
hypothesis_geometry : a  = b * b / c

theorem partII : ∀ (A B C a b c : ℝ), 
  B = π / 3 →
  b ^ 2 = a * c →
  sin B ^ 2 = sin A * sin C →
  (1 / tan A) + (1 / tan C) = 2 * √3 / 3 := sorry

end partI_partII_l419_419520


namespace length_of_BC_l419_419224

-- Definitions for the given conditions
variable {O A B C D : Type}
variable [metric_space O]
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variable [has_dist O A] [has_dist O B] [has_dist O C] [has_dist A C] [has_dist B C] [has_dist D A]
variable (BO AO CO : ℝ)
variable (angle_ABO angle_BOC : ℝ)
variable (arc_CD : ℝ)

-- Assume the given conditions
axiom cond_center_o : O
axiom cond_diameter_ad : O
axiom cond_chord_abc : O
axiom cond_BO : BO = 6
axiom cond_angle_ABO : angle_ABO = π / 2
axiom cond_arc_CD : arc_CD = π / 2

-- Main theorem statement: finding the length of BC
theorem length_of_BC :
  let BC := Real.sqrt (72 - 36 * Real.sqrt 2) in
  dist B C = 6 * Real.sqrt (2 - Real.sqrt 2) := 
begin
  sorry
end

end length_of_BC_l419_419224


namespace solve_for_n_l419_419983

theorem solve_for_n : ∃ n : ℤ, 3^3 - 5 = 4^2 + n ∧ n = 6 := 
by
  use 6
  sorry

end solve_for_n_l419_419983


namespace container_marbles_volume_l419_419044

theorem container_marbles_volume {V₁ V₂ m₁ m₂ : ℕ} 
  (h₁ : V₁ = 24) (h₂ : m₁ = 75) (h₃ : V₂ = 72) :
  m₂ = 225 :=
by
  have proportion := (m₁ : ℚ) / V₁
  have proportion2 := (m₂ : ℚ) / V₂
  have h4 := proportion = proportion2
  sorry

end container_marbles_volume_l419_419044


namespace kamal_salary_loss_l419_419814

def percentage_loss (S : ℝ) : ℝ := ((S - (0.65 * S)) / S) * 100

theorem kamal_salary_loss : ∀ (S : ℝ) (hS : S > 0),
  percentage_loss S = 35 := by
  sorry

end kamal_salary_loss_l419_419814


namespace area_of_circumcircle_of_isosceles_triangle_l419_419832

theorem area_of_circumcircle_of_isosceles_triangle :
  let AB := 4
  let AC := 4
  let BC := 3
  let AD := (√(AB^2 - (BC/2)^2))
  let radius := AD
  let area := π * radius^2
  area = 16 * π :=
by
  sorry

end area_of_circumcircle_of_isosceles_triangle_l419_419832


namespace binomial_inequality_l419_419713

theorem binomial_inequality (n : ℤ) (x : ℝ) (hn : n ≥ 2) (hx : |x| < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := 
sorry

end binomial_inequality_l419_419713


namespace diagonals_of_octagon_l419_419547

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l419_419547


namespace angle_C_90_sum_a_b_4_l419_419663

theorem angle_C_90 (a b c : ℝ) (h : a^2 - a * b + b^2 = c^2) : 
  ∠(((a, 0), (0, b), (0, 0)) : ℝ) = 90 := 
by
  sorry

theorem sum_a_b_4 (a b : ℝ) (c : ℝ) (S : ℝ) (h1 : c = 2) (h2 : S = 1/2 * a * b) (h3 : ab = 4) :
  a + b = 4 :=
by
  sorry

end angle_C_90_sum_a_b_4_l419_419663


namespace quadratic_has_two_roots_l419_419669

theorem quadratic_has_two_roots {a b c x1 x2 : ℝ} 
    (h1 : a ≠ 0) 
    (h2 : a * x1^2 + b * x1 + c = 0) 
    (h3 : a * x2^2 + b * x2 + c = 0) 
    (h4 : x1 ≠ x2)
    (h5 : x1^3 - x2^3 = 2011) : 
    let D := (2 * b)^2 - 4 * a * (4 * c) in 
  D > 0 := 
by 
  sorry

end quadratic_has_two_roots_l419_419669


namespace polygon_area_l419_419955

theorem polygon_area :
  let a := (0, 0)
  let b := (10, 0)
  let c := (20, 0)
  let d := (30, 0)
  let e := (0, 10)
  let f := (10, 10)
  let g := (20, 10)
  let h := (30, 10)
  let i := (0, 20)
  let j := (10, 20)
  let k := (20, 20)
  let l := (30, 20)
  let m := (0, 30)
  let n := (10, 30)
  let o := (20, 30)
  let p := (30, 30)
  in 
  let AEIM_area := 3 * 1
  let MNOP_area := 1 * 3
  let PHD_area := 3 * 3
  AEIM_area + MNOP_area + PHD_area = 15 :=
sorry

end polygon_area_l419_419955


namespace average_decrease_rate_required_price_reduction_l419_419039

-- Define the conditions
def factory_price_2019 : ℝ := 200
def factory_price_2021 : ℝ := 162
def daily_sold_2019 : ℕ := 20
def price_increase_per_reduction : ℕ := 10
def price_reduction_per_unit : ℝ := 5
def target_daily_profit : ℝ := 1150

-- Part 1: Prove the average decrease rate
theorem average_decrease_rate : 
  ∃ (x : ℝ), (factory_price_2019 * (1 - x)^2 = factory_price_2021) ∧ x = 0.1 :=
begin
  sorry
end

-- Part 2: Prove the required unit price reduction
theorem required_price_reduction :
  ∃ (m : ℝ), ((38 - m) * (daily_sold_2019 + 2 * m / price_reduction_per_unit) = target_daily_profit) ∧ m = 15 :=
begin
  sorry
end

end average_decrease_rate_required_price_reduction_l419_419039


namespace octagon_diagonals_20_l419_419552

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l419_419552


namespace art_club_activity_l419_419644

theorem art_club_activity (n p s b : ℕ) (h1 : n = 150) (h2 : p = 80) (h3 : s = 60) (h4 : b = 20) :
  (n - (p + s - b) = 30) :=
by
  sorry

end art_club_activity_l419_419644


namespace required_minimal_road_colors_l419_419053

def minimal_road_colors (towns roads colors : ℕ) (connects : ℕ → ℕ → Prop): Prop :=
  ∀ (A B : ℕ) (A_town B_town : A < towns ∧ B < towns) 
  (path : list ℕ) (road_steps : ∀ n ∈ path, connects n ((n + 101) % towns)),
  (list.nodup (path.map colors)) → towns = 2021 → roads = 101 → ∃ m, colors = m + 1 ∧ m ≥ 20

theorem required_minimal_road_colors :
  minimal_road_colors 2021 101 21 sorry :=
  begin
    sorry
  end

end required_minimal_road_colors_l419_419053


namespace Kolya_max_chords_l419_419295

-- Defining the problem conditions
def points := 2006
def colors := 17
def points_per_color := points / colors

-- Theorem statement
theorem Kolya_max_chords : ∀ (points : ℕ) (colors : ℕ), 
  points = 2006 → colors = 17 → points % colors = 0 → 
  (∀ chords, 
    (∀ c ∈ chords, (c.left.color = c.right.color) ∧ (∀ d ≠ c, ¬ (c.intersect d))) → 
    ∃ m, m = 117) :=
by
  intros points colors hp hc hmod chords hchords
  -- The actual proof would go here
  sorry

end Kolya_max_chords_l419_419295


namespace octagon_diagonals_20_l419_419556

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l419_419556


namespace octagon_diagonals_l419_419590

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l419_419590


namespace power_function_through_point_l419_419169

theorem power_function_through_point :
  ∃ (α : ℝ), (∀ x : ℝ, x > 0 → f x = x ^ α) ∧ f 2 = 4 → f x = x ^ 2 :=
by
  sorry

end power_function_through_point_l419_419169


namespace bees_fewer_than_flowers_l419_419817

theorem bees_fewer_than_flowers : 
  let flowers := 5 in
  let bees := 3 in
  bees = flowers - 2 :=
by
  sorry

end bees_fewer_than_flowers_l419_419817


namespace sum_of_extreme_values_eq_four_l419_419271

-- Given conditions in problem statement
variables (x y z : ℝ)
variables (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 8)

-- Statement to be proved: sum of smallest and largest possible values of x is 4
theorem sum_of_extreme_values_eq_four : m + M = 4 :=
sorry

end sum_of_extreme_values_eq_four_l419_419271


namespace degree_of_polynomial_l419_419345

theorem degree_of_polynomial (p : ℕ) (n : ℕ) (m : ℕ) (h : p = 3) (k : n = 15) :
  m = p * n := by
  sorry

-- Given p = 3 (degree of 5x^3) and n = 15 (exponent in (5x^3 + 7)^15)
-- Prove that m = 45 (degree of (5x^3 + 7)^15)
noncomputable def main_theorem : Prop :=
  (degree_of_polynomial 3 15 45 rfl rfl)

end degree_of_polynomial_l419_419345


namespace no_line_intersects_all_segments_interior_l419_419329

-- Define the structure of required points and line segments
structure PlanePoint where
  x : ℝ
  y : ℝ

structure LineSegment where
  start : PlanePoint
  end : PlanePoint

-- Given conditions
constant points : Fin 1001 → PlanePoint
constant line_segments : Fin 1001 → LineSegment
axiom condition_no_three_collinear (i j k : Fin 1001) : 
  (points i).x * ((points j).y - (points k).y) + 
  (points j).x * ((points k).y - (points i).y) + 
  (points k).x * ((points i).y - (points j).y) ≠ 0

axiom point_end_segment (i : Fin 1001) : 
  (∃ j, line_segments j = ⟨points i, _⟩) ∧ 
  (∃ k, line_segments k = ⟨_, points i⟩)

-- Proof statement
theorem no_line_intersects_all_segments_interior :
  ¬ ∃ L : ℝ → PlanePoint, 
    ∀ s : Fin 1001, (let inter := (L some_r) in 
      inter ≠ line_segments s.start ∧ inter ≠ line_segments s.end) :=
sorry

end no_line_intersects_all_segments_interior_l419_419329


namespace area_of_circumscribed_circle_isosceles_triangle_l419_419858

theorem area_of_circumscribed_circle_isosceles_triangle :
  ∃ (r : ℝ), (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi) :=
by
  -- Consider the isosceles triangle conditions
  let a : ℝ := 4
  let b : ℝ := 4
  let c : ℝ := 3
  let BD := Real.sqrt(a^2 - (c/2)^2)
  let r := 8 / BD
  have h1 : BD = Real.sqrt 13.75 := by 
    -- Calculate the altitude BD
    calc
      BD = Real.sqrt(a^2 -  (c/2)^2) : rfl
      ... = Real.sqrt(16 - (3/2)^2) : rfl
      ... = Real.sqrt 13.75 : rfl
  
  use r
  have h2 : r = 8 / Real.sqrt 13.75 := by 
    -- Simplify the radius expression
    sorry

  have h3 : Real.pi * r ^ 2 = 256 / 55 * Real.pi := by 
    -- Calculate the area
    calc
      Real.pi * r ^ 2 = Real.pi * (8 / Real.sqrt 13.75) ^ 2 : by rw h2
      ... = Real.pi * (64 / 13.75) : by rw [pow_two, mul_div_assoc, mul_one, div_mul_div_same]
      ... = (256 / 54.6875) * Real.pi : by rw mul_comm
      ...   = (256 / 55) * Real.pi : by norm_num
    sorry
  
  show (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi),
  from ⟨h2, h3⟩
  sorry

end area_of_circumscribed_circle_isosceles_triangle_l419_419858


namespace frequency_approaches_probability_not_frequency_is_probability_frequency_not_independent_of_trials_probability_in_interval_l419_419354

def frequency (n : ℕ) : ℝ := sorry  -- Placeholder definition
def probability : ℝ := sorry  -- Placeholder definition

-- Assume some basic facts about probabilities
axiom prob_nonneg : ∀ (A : Prop), probability >= 0
axiom prob_le_one : ∀ (A : Prop), probability <= 1

theorem frequency_approaches_probability (n : ℕ) :
  (B : ∀ n, frequency(n) → probability) := sorry

theorem not_frequency_is_probability :
  (A : ¬ frequency (n) = probability) := sorry

theorem frequency_not_independent_of_trials (n : ℕ) :
  (C : ∀ n, frequency(n) depends_on n) := sorry

theorem probability_in_interval :
  (D : ∀ A, probability ∈ set.Icc 0 1) := sorry

end frequency_approaches_probability_not_frequency_is_probability_frequency_not_independent_of_trials_probability_in_interval_l419_419354


namespace find_smaller_number_l419_419131

-- Define the two numbers such that one is 3 times the other
def numbers (x : ℝ) := (x, 3 * x)

-- Define the condition that the sum of the two numbers is 14
def sum_condition (x y : ℝ) : Prop := x + y = 14

-- The theorem we want to prove
theorem find_smaller_number (x : ℝ) (hx : sum_condition x (3 * x)) : x = 3.5 :=
by
  -- Proof goes here
  sorry

end find_smaller_number_l419_419131


namespace relative_error_comparison_l419_419421

theorem relative_error_comparison :
  let e₁ := 0.05
  let l₁ := 25.0
  let e₂ := 0.4
  let l₂ := 200.0
  let relative_error (e l : ℝ) : ℝ := (e / l) * 100
  (relative_error e₁ l₁ = relative_error e₂ l₂) :=
by
  sorry

end relative_error_comparison_l419_419421


namespace sin_cos_difference_l419_419327

theorem sin_cos_difference (h: sin 60 = real.sqrt 3 / 2) : 
  sin 80 * cos 20 - cos 80 * sin 20 = real.sqrt 3 / 2 :=
by
  sorry

end sin_cos_difference_l419_419327


namespace green_triangle_area_percentage_l419_419405

-- Define the side length of the flag
variable (k : ℝ) (A_flag : ℝ)
-- Set the area of the flag as the square of the side length
def area_flag : Prop := A_flag = k ^ 2

-- The red cross and green triangle together cover 25% of the flag's area
def cross_triangle_coverage : Prop := (0.25 * A_flag)  

-- Assume area of the green triangle is 4% of the area of the flag
def green_triangle_coverage : Prop := ∃ (x : ℝ), x = (0.04 * A_flag)

-- The main proposition: proving the green triangle covers 4% of the flag's area
theorem green_triangle_area_percentage : area_flag k A_flag ∧ cross_triangle_coverage A_flag → green_triangle_coverage A_flag :=
by
  sorry

end green_triangle_area_percentage_l419_419405


namespace avg_speed_last_40_min_is_70_l419_419253

noncomputable def avg_speed_last_interval
  (total_distance : ℝ) (total_time : ℝ)
  (speed_first_40_min : ℝ) (time_first_40_min : ℝ)
  (speed_second_40_min : ℝ) (time_second_40_min : ℝ) : ℝ :=
  let time_last_40_min := total_time - (time_first_40_min + time_second_40_min)
  let distance_first_40_min := speed_first_40_min * time_first_40_min
  let distance_second_40_min := speed_second_40_min * time_second_40_min
  let distance_last_40_min := total_distance - (distance_first_40_min + distance_second_40_min)
  distance_last_40_min / time_last_40_min

theorem avg_speed_last_40_min_is_70
  (h_total_distance : total_distance = 120)
  (h_total_time : total_time = 2)
  (h_speed_first_40_min : speed_first_40_min = 50)
  (h_time_first_40_min : time_first_40_min = 2 / 3)
  (h_speed_second_40_min : speed_second_40_min = 60)
  (h_time_second_40_min : time_second_40_min = 2 / 3) :
  avg_speed_last_interval 120 2 50 (2 / 3) 60 (2 / 3) = 70 :=
by
  sorry

end avg_speed_last_40_min_is_70_l419_419253


namespace cos_A_eq_neg_quarter_l419_419221

-- Definitions of angles and sides in the triangle
variables (A B C : ℝ)
variables (a b c : ℝ)

-- Conditions from the math problem
axiom sin_arithmetic_sequence : 2 * Real.sin B = Real.sin A + Real.sin C
axiom side_relation : a = 2 * c

-- Question to be proved as Lean 4 statement
theorem cos_A_eq_neg_quarter (h1 : ∀ {x y z : ℝ}, 2 * y = x + z) 
                              (h2 : ∀ {a b c : ℝ}, a = 2 * c) : 
                              Real.cos A = -1/4 := 
sorry

end cos_A_eq_neg_quarter_l419_419221


namespace domain_of_g_l419_419136

noncomputable def g (x : ℝ) : ℝ := real.sqrt (-6 * x^2 - 7 * x + 12)

def quadratic_nonnegative (x : ℝ) : Prop := -6 * x^2 - 7 * x + 12 ≥ 0

theorem domain_of_g : ∀ x : ℝ, quadratic_nonnegative x → -2 ≤ x ∧ x ≤ 1 / 3 :=
begin
  sorry
end

end domain_of_g_l419_419136


namespace ratio_of_areas_l419_419749

theorem ratio_of_areas (s L : ℝ) (h1 : (π * L^2) / (π * s^2) = 9 / 4) : L - s = (1/2) * s :=
by
  sorry

end ratio_of_areas_l419_419749


namespace math_problem_l419_419011

-- Define the conditions
def conditions (z : ℝ) : Prop :=
  z ≠ 3 / 2 ∧ z ≠ -3 / 2 ∧ z ≠ 0

-- Define the problem
noncomputable def problem (z : ℝ) : ℝ :=
  (Real.cbrt ((8 * z^3 + 24 * z^2 + 18 * z) / (2 * z - 3)) - 
  Real.cbrt ((8 * z^3 - 24 * z^2 + 18 * z) / (2 * z + 3))) - 
  ((1 / 2) * Real.cbrt ((2 * z) / 27 - (1 / (6 * z))))⁻¹

-- Statement of the theorem
theorem math_problem (z : ℝ) (h : conditions z) : problem z = 0 :=
  sorry

end math_problem_l419_419011


namespace pow_of_729_l419_419087

theorem pow_of_729 : (729 : ℝ) ^ (2 / 3) = 81 :=
by sorry

end pow_of_729_l419_419087


namespace sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14_l419_419795

noncomputable def value_x : ℝ := real.sqrt (20 - 8 * real.sqrt 5) + real.sqrt (20 + 8 * real.sqrt 5)

theorem sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14 :
  value_x = 2 * real.sqrt 14 :=
by
  sorry

end sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14_l419_419795


namespace exists_list_with_all_players_l419_419229

-- Definitions and assumptions
variable {Player : Type} 

-- Each player plays against every other player exactly once, and there are no ties.
-- Defining defeats relationship
def defeats (p1 p2 : Player) : Prop :=
  sorry -- Assume some ordering or wins relationship

-- Defining the list of defeats
def list_of_defeats (p : Player) : Set Player :=
  { q | defeats p q ∨ (∃ r, defeats p r ∧ defeats r q) }

-- Main theorem to be proven
theorem exists_list_with_all_players (players : Set Player) :
  (∀ p q : Player, p ∈ players → q ∈ players → p ≠ q → (defeats p q ∨ defeats q p)) →
  ∃ p : Player, (list_of_defeats p) = players \ {p} :=
by
  sorry

end exists_list_with_all_players_l419_419229


namespace diagonals_of_octagon_l419_419542

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l419_419542


namespace find_x_when_y_equals_two_l419_419625

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end find_x_when_y_equals_two_l419_419625


namespace circle_area_isosceles_triangle_l419_419842

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end circle_area_isosceles_triangle_l419_419842


namespace exists_n_ge_3_f_n_ge_n_pow_0_01_l419_419276

noncomputable def f (n : ℕ) : ℕ :=
  Inf {m : ℕ | ∀ a b ∈ Finset.filter (λ x, n % x = 0) (Finset.range (n + 1)), a ≠ b → a % m ≠ b % m}

-- Prove there exists an integer N = 3 such that for all n ≥ N, f(n) ≥ n^0.01
theorem exists_n_ge_3_f_n_ge_n_pow_0_01 :
  ∃ N : ℕ, (N = 3) ∧ ∀ n : ℕ, n ≥ N → f n ≥ n ^ 0.01 :=
by
  sorry

end exists_n_ge_3_f_n_ge_n_pow_0_01_l419_419276


namespace no_n_real_roots_l419_419688

noncomputable def real_polynomial (n : ℕ) (a : Fin n → ℝ) : Polynomial ℝ := 
  Polynomial.mk (List.ofFn (λ i, if i = 0 then 1 else if i = 1 then a 0 else a (i - 1)))

theorem no_n_real_roots
  (n : ℕ)
  (a : Fin n → ℝ)
  (hcond : (a 0) ^ 2 < (2 * n) / (n - 1) * a 1)
  (hroot : ∃ r : Fin n → ℝ, (∀ i j : Fin n, i ≠ j → r i ≠ r j) ∧ real_polynomial n a = Polynomial.prod (λ i, Polynomial.x - Polynomial.C (r i))) :
  False := 
sorry

end no_n_real_roots_l419_419688


namespace number_of_small_spheres_l419_419879

-- Define the diameters of the large and small spheres.
def diameter_large_sphere : ℝ := 10
def diameter_small_sphere : ℝ := 2

-- Define the radius of the large and small spheres.
def radius_large_sphere : ℝ := diameter_large_sphere / 2
def radius_small_sphere : ℝ := diameter_small_sphere / 2

-- Define the volume formula for a sphere.
def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r ^ 3)

-- Define the volumes of the large and small spheres.
def volume_large_sphere : ℝ := volume radius_large_sphere
def volume_small_sphere : ℝ := volume radius_small_sphere

-- The main theorem: Prove the number of small spheres.
theorem number_of_small_spheres : volume_large_sphere / volume_small_sphere = 125 := by
  sorry

end number_of_small_spheres_l419_419879


namespace annual_income_from_investment_l419_419965

-- Define the conditions
def investment_amount : ℝ := 6800
def stock_price : ℝ := 136
def dividend_rate : ℝ := 0.60
def par_value : ℝ := 100

-- Define the theorem to be proven
theorem annual_income_from_investment (investment_amount = 6800) 
                                       (stock_price = 136)
                                       (dividend_rate = 0.60)
                                       (par_value = 100) :
  investment_amount / stock_price * (par_value * dividend_rate) = 3000 := by
  sorry

end annual_income_from_investment_l419_419965


namespace count_valid_Qs_l419_419279

noncomputable def P := λ x : ℝ, (x - 1) * (x - 2) * (x - 3) * (x - 4)

theorem count_valid_Qs : 
  ∃ Qs : set (ℝ → ℝ), (∀ Q ∈ Qs, ∃ R : ℝ → ℝ, degree R = 4 ∧ (∀ x, P (Q x) = P x * R x)) ∧ card Qs = 254 :=
sorry

end count_valid_Qs_l419_419279


namespace friend_selling_price_l419_419392

-- Define the conditions
def CP : ℝ := 51136.36
def loss_percent : ℝ := 0.12
def gain_percent : ℝ := 0.20

-- Define the selling prices SP1 and SP2
def SP1 := CP * (1 - loss_percent)
def SP2 := SP1 * (1 + gain_percent)

-- State the theorem
theorem friend_selling_price : SP2 = 54000 := 
by sorry

end friend_selling_price_l419_419392


namespace find_base_b_l419_419129

theorem find_base_b (b : Real) : log b 64 = -4/3 → b = 1/32 :=
by
  sorry

end find_base_b_l419_419129


namespace who_is_lunatic_l419_419806

def Person := Type
variables (priest liar lunatic p1 p2 p3 : Person)

-- Conditions
-- The priest always tells the truth
def always_tells_truth (x : Person) := x = priest

-- The liar always lies
def always_lies (x : Person) := x = liar

-- The lunatic sometimes tells the truth and sometimes lies
def sometimes_tells_truth (x : Person) := x = lunatic

-- Statements made by the people
def statement_1 (x : Person) := x = lunatic
def statement_2 (y : Person) (x : Person) := x ≠ lunatic
def statement_3 (z : Person) := z = lunatic

theorem who_is_lunatic
  (H1 : (priest = p1 ∨ liar = p1 ∨ lunatic = p1))
  (H2 : (priest = p2 ∨ liar = p2 ∨ lunatic = p2))
  (H3 : (priest = p3 ∨ liar = p3 ∨ lunatic = p3))
  (H4 : statement_1 p1)
  (H5 : statement_2 p2 p1)
  (H6 : statement_3 p3)
  (H7 : always_tells_truth priest)
  (H8 : always_lies liar)
  (H9 : sometimes_tells_truth lunatic) :
  p3 = lunatic := sorry

end who_is_lunatic_l419_419806


namespace f_even_range_a_l419_419508

open Real
open Finset

-- Definitions of the functions
def f (x : ℝ) : ℝ := log (9 ^ x + 1) / log 3 - x
def g (x : ℝ) (a : ℝ) : ℝ := log (a + 2 - (a + 4) / (3 ^ x)) / log 3

-- Proofs to be provided
theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by sorry

theorem range_a (a : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x ≥ g x a) ↔ (-20/3 : ℝ) ≤ a ∧ a ≤ 4 :=
by sorry

end f_even_range_a_l419_419508


namespace original_annual_pension_l419_419079

theorem original_annual_pension (k x c d r s : ℝ) (h1 : k * (x + c) ^ (3/4) = k * x ^ (3/4) + r)
  (h2 : k * (x + d) ^ (3/4) = k * x ^ (3/4) + s) :
  k * x ^ (3/4) = (r - s) / (0.75 * (d - c)) :=
by sorry

end original_annual_pension_l419_419079


namespace probability_of_two_evens_in_five_l419_419474

-- Define the initial set of numbers
def num_set : set ℕ := {1, 2, 3, 4, 5}

-- Define a condition for even numbers in the set
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the count of ways to select two elements from a given set
def combination (n r : ℕ) : ℕ := n.choose r

-- The probability calculation function (stub)
noncomputable def probability_even_in_two_selected (set : set ℕ) : ℚ :=
let total_ways := combination (set.to_finset.card) 2 in
let even_ways := combination (set.filter is_even).to_finset.card 2 in
(even_ways : ℚ) / total_ways

-- The theorem statement to prove
theorem probability_of_two_evens_in_five :
  probability_even_in_two_selected num_set = 1 / 10 :=
by
  sorry

end probability_of_two_evens_in_five_l419_419474


namespace recurring_decimal_to_fraction_l419_419126

theorem recurring_decimal_to_fraction : ∀ x : ℝ, (x = 7 + (1/3 : ℝ)) → x = (22/3 : ℝ) :=
by
  sorry

end recurring_decimal_to_fraction_l419_419126


namespace jinho_remaining_money_l419_419251

def jinho_initial_money : ℕ := 2500
def cost_per_eraser : ℕ := 120
def erasers_bought : ℕ := 5
def cost_per_pencil : ℕ := 350
def pencils_bought : ℕ := 3

theorem jinho_remaining_money :
  jinho_initial_money - (erasers_bought * cost_per_eraser + pencils_bought * cost_per_pencil) = 850 :=
by
  sorry

end jinho_remaining_money_l419_419251


namespace no_absolute_winner_l419_419905

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end no_absolute_winner_l419_419905


namespace find_x_when_y_equals_two_l419_419627

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end find_x_when_y_equals_two_l419_419627


namespace octagon_diagonals_20_l419_419555

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l419_419555


namespace exists_48_good_perfect_square_l419_419303

-- Problem definition
def is_k_good (k : ℕ) (n : ℕ) : Prop :=
  ∃ (y z : ℕ), y = k * z ∧ (y * 10 ^ (Nat.log10 z + 1) + z = n)

-- Lean statement
theorem exists_48_good_perfect_square : ∃ (n : ℕ), is_k_good 48 n ∧ ∃ (k : ℕ), n = k * k := 
sorry

end exists_48_good_perfect_square_l419_419303


namespace sqrt_sum_simplification_l419_419791

theorem sqrt_sum_simplification : 
  sqrt (20 - 8 * sqrt 5) + sqrt (20 + 8 * sqrt 5) = 2 * sqrt 14 := 
by
  sorry

end sqrt_sum_simplification_l419_419791


namespace find_x_l419_419459

theorem find_x :
  (∃ x : ℝ, real.cbrt (5 - x/3) = -4) → x = 207 :=
by
  intro hx
  rcases hx with ⟨x, H⟩
  sorry

end find_x_l419_419459


namespace find_b_l419_419204

noncomputable def a : ℝ × ℝ := (-1, 2)
noncomputable def b (λ : ℝ) : ℝ × ℝ := (λ, -2 * λ)
noncomputable def b_magnitude (λ : ℝ) : ℝ := real.sqrt ((λ^2) + (-2 * λ)^2)

theorem find_b (λ : ℝ) (h1 : λ > 0) (h2 : b_magnitude λ = 3 * real.sqrt 5) :
  b λ = (3, -6) := sorry

end find_b_l419_419204


namespace number_of_correct_propositions_l419_419476

-- Definitions for the problem as per the conditions
variables {l m : Line} {α β : Plane}

-- Propositions from conditions
def prop1 : Prop := ∀ (l1 l2 : Line), (l ⟂ l1) ∧ (l ⟂ l2) ∧ (l1 ∈ α) ∧ (l2 ∈ α) → l ⟂ α
def prop2 : Prop := (m ∥ α) ∧ (l ⟂ α) → m ⟂ l
def prop3 : Prop := l ∥ α → ∀ l', l' ∈ α → l ∥ l'
def prop4 : Prop := (m ∈ α) ∧ (l ∈ β) ∧ (α ∥ β) → m ∥ l

-- The statement to be proven
theorem number_of_correct_propositions :
  ((¬ prop1) ∧ (prop2) ∧ (¬ prop3) ∧ (¬ prop4)) ↔ true := 
by sorry

end number_of_correct_propositions_l419_419476


namespace min_a1_l419_419681

noncomputable def a : ℕ → ℝ
noncomputable def a_zero (a1 : ℝ) : ℕ → ℝ 
noncomputable def a_pos (a1 : ℝ) (n : ℕ) : Prop 
noncomputable def a_cond (a1 : ℝ) (n : ℕ) : Prop 

theorem min_a1 :
  let a : ℕ → ℝ := λ n, ((n+1: ℕ).choose (1: ℕ)) * 5^(n-1) * (a1) - ∑ i in range n, i^2 
  ∀ n > 1, (a n = 5 * a (n - 1) - (n: ℝ) ^ 2) → 
  (∀ n, a n > 0) →
  a 1 = 25 / 4
  := 
  sorry

end min_a1_l419_419681


namespace octagon_diagonals_20_l419_419558

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l419_419558


namespace replace_asterisks_identity_l419_419300

theorem replace_asterisks_identity :
  ∃ (A B : ℝ), A = 2 ∧ B = 5 ∧
  (∀ x : ℝ, (3 * x - A) * (2 * x + 5) - x = 6 * x^2 + 2 * (5 * x - B)) :=
begin
  use [2, 5],
  split, {refl},
  split, {refl},
  intro x,
  calc
    (3 * x - 2) * (2 * x + 5) - x
        = 6 * x^2 + 14 * x - 4 * x - 10 - x : by ring
    ... = 6 * x^2 + 10 * x - 10 : by ring
end

end replace_asterisks_identity_l419_419300


namespace solve_cubic_root_eq_l419_419450

theorem solve_cubic_root_eq (x : ℝ) : (real.cbrt (5 - x / 3) = -4) → x = 207 :=
by
  sorry

end solve_cubic_root_eq_l419_419450


namespace sum_of_solutions_l419_419467

theorem sum_of_solutions (x : ℝ) (h : 8^(x^2 - 4 * x - 3) = 16^(x - 3)) :
  ∑ x in {x | 8^(x^2 - 4 * x - 3) = 16^(x - 3)}, x = 16 / 3 :=
by sorry

end sum_of_solutions_l419_419467


namespace find_number_l419_419813

theorem find_number (x : ℝ) : 0.40 * x = 0.80 * 5 + 2 → x = 15 :=
by
  intros h
  sorry

end find_number_l419_419813


namespace general_term_formula_find_m_exists_term_l419_419159

def arithmetic_sequence (a d : ℕ) : ℕ → ℕ
| 0       := a
| (n + 1) := arithmetic_sequence a d n + d

def geometric_sequence (a r : ℕ) : ℕ → ℕ
| 0       := a
| (n + 1) := geometric_sequence a r n * r

def sequence (n : ℕ) : ℕ :=
if even n then
  geometric_sequence 2 3 (n / 2 - 1)
else
  arithmetic_sequence 1 2 (n / 2)

def sequence_sum (n : ℕ) : ℕ :=
(n+1).sum (λ k, sequence k)

axiom Sn_condition : sequence_sum 5 = 2 * sequence 4 + sequence 5
axiom a9_condition : sequence 9 = sequence 3 + sequence 4

theorem general_term_formula 
  (n : ℕ) : (sequence n = if even n then 2 * 3^(n / 2 - 1) else 2 * (n / 2) - 1) :=
sorry

theorem find_m 
  (m : ℕ)
  (h : sequence m * sequence (m+1) = sequence (m+2)) : m = 2 :=
sorry

theorem exists_term 
  (m : ℕ) 
  (S : ℕ → ℕ) 
  (h1 : S 2 = 5) 
  (h2 : (S 1) = 3): 
  ∃ (k : ℕ), ∃L:ℕ, L = sequence k :=
sorry

end general_term_formula_find_m_exists_term_l419_419159


namespace correct_order_of_figures_and_colors_l419_419290

-- Definitions based on the conditions
def is_between (x y z : Nat) := x < y ∧ y < z
def is_right_of (x y : Nat) := x > y
def not_adjacent (x y : Nat) := abs (x - y) ≠ 1
def not_edge (x : Nat) := x ≠ 1 ∧ x ≠ 4

-- Predicate to determine if the arrangement is correct
def valid_arrangement (pos_color : Nat → String) (pos_shape : Nat → String) : Prop :=
  (∃ r b g, 
    pos_color(r) = "Red" ∧ pos_color(b) = "Blue" ∧ pos_color(g) = "Green" ∧
    is_between b r g) ∧
  (∃ y rh,
    pos_color(y) = "Yellow" ∧ pos_shape(rh) = "Rhombus" ∧
    is_right_of rh y) ∧
  (∃ c t rh,
    pos_shape(c) = "Circle" ∧ pos_shape(t) = "Triangle" ∧ pos_shape(rh) = "Rhombus" ∧
    is_right_of c t ∧ is_right_of c rh) ∧
  (∃ t,
    pos_shape(t) = "Triangle" ∧ not_edge t) ∧
  (∃ b y,
    pos_color(b) = "Blue" ∧ pos_color(y) = "Yellow" ∧
    not_adjacent b y)

-- Final theorem statement
theorem correct_order_of_figures_and_colors :
  (pos_color pos_shape : Nat → String) (colors : Nat → String) (shapes : Nat → String),
    pos_color(1) = "Yellow" ∧ pos_shape(1) = "Rectangle" ∧
    pos_color(2) = "Green" ∧ pos_shape(2) = "Rhombus" ∧
    pos_color(3) = "Red" ∧ pos_shape(3) = "Triangle" ∧
    pos_color(4) = "Blue" ∧ pos_shape(4) = "Circle" →
  valid_arrangement pos_color pos_shape := 
by
  sorry

end correct_order_of_figures_and_colors_l419_419290


namespace lucy_fewer_heads_12_l419_419789

noncomputable def probability_fewer_heads (n_flips : Nat) : ℚ :=
  let total_outcomes := 2^n_flips
  let favorable_outcomes := (Matrix.binom n_flips (n_flips / 2))
  let equal_heads_tails_prob := favorable_outcomes / total_outcomes
  (1 - equal_heads_tails_prob) / 2

theorem lucy_fewer_heads_12 : probability_fewer_heads 12 = 793 / 2048 := by
  sorry

end lucy_fewer_heads_12_l419_419789


namespace probability_no_replant_for_pit_distribution_and_expectation_of_X_l419_419822

-- Definitions from the conditions
def n : ℕ := 4
def seedGerminationProbability : ℚ := 1 / 2
def atLeastTwoSeedsGerminate (pit : ℕ → bool) : bool :=
  pit.count id >= 2

-- Questions translated to Lean statements
theorem probability_no_replant_for_pit : 
  let p_replant := (1 / 8 : ℚ) + (3 / 8 : ℚ),
      p_no_replant := 1 - p_replant in 
  p_no_replant = 1 / 2 :=
by sorry

theorem distribution_and_expectation_of_X (X : ℕ → Prop) :
  (∀ k, X k ↔ (X k) ∈ (Set.ofFinset (Finset.Icc 0 n) (λ k, binomialPdf n seedGerminationProbability k))) ∧
  (∑ k in Finset.range (n + 1), k * (binomialPdf n seedGerminationProbability k) = (2 : ℚ)) :=
by sorry

end probability_no_replant_for_pit_distribution_and_expectation_of_X_l419_419822


namespace function_maximum_value_l419_419107

noncomputable def domain_condition (x : ℝ) : Prop :=
  3 - 4 * x + x^2 > 0

noncomputable def f (x : ℝ) : ℝ :=
  2^x + 2 - 3 * 4^x

theorem function_maximum_value :
  let M := { x : ℝ | x < 1 ∨ x > 3 } in
  (∀ x, domain_condition x → (x ∈ M)) ∧ (∀ x ∈ M, f x ≤ -8) :=
by
  intros
  sorry

end function_maximum_value_l419_419107


namespace complex_magnitude_product_l419_419117

noncomputable def z1 : ℂ := 3 * Real.sqrt 5 - 5 * Complex.i
noncomputable def z2 : ℂ := 2 * Real.sqrt 2 + 4 * Complex.i
noncomputable def magnitude (z : ℂ) : ℝ := Complex.abs z

theorem complex_magnitude_product :
  magnitude (z1 * z2) = 12 * Real.sqrt 35 :=
by
  have z1 := z1
  have z2 := z2
  sorry

end complex_magnitude_product_l419_419117


namespace octagon_diagonals_l419_419538

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l419_419538


namespace kim_money_l419_419256

theorem kim_money (S P K A : ℝ) (h1 : K = 1.40 * S) (h2 : S = 0.80 * P) (h3 : A = 1.25 * (S + K)) (h4 : S + P + A = 3.60) : K = 0.96 :=
by
  sorry

end kim_money_l419_419256


namespace sophomores_playing_instrument_l419_419885

theorem sophomores_playing_instrument (j s : ℕ) (total_students : ℕ) (p_juniors_play : ℚ) (p_sophomores_play : ℚ) (p_total_play : ℚ) 
  (h1 : total_students = 600)
  (h2 : p_juniors_play = 0.55)
  (h3 : p_sophomores_play = 0.25)
  (h4 : p_total_play = 0.505)
  (h5 : j + s = total_students)
  (h6 : (p_juniors_play * j + p_sophomores_play * s).toReal = (p_total_play * total_students).toReal) :
  s * 0.25 = 23 :=
by
  sorry

end sophomores_playing_instrument_l419_419885


namespace find_a6_l419_419516

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 7 ∧ ∀ n ≥ 2, a n = (1/2) * a (n-1) + 3

theorem find_a6 (a : ℕ → ℚ) (h : sequence a) : a 6 = 193 / 32 :=
by
  sorry

end find_a6_l419_419516


namespace luke_total_coins_l419_419284

noncomputable def total_coins : ℕ :=
let quarters := 4 * 2 + 6 * 3 + 8 in
let dimes := 3 + 5 + 2 * 2 in
let nickels := 5 * 3 + 7 * 2 + 10 in
let pennies := 12 + 8 + 20 in
let half_dollars := 2 + 4 in
quarters + dimes + nickels + pennies + half_dollars

theorem luke_total_coins : total_coins = 131 :=
by
  sorry

end luke_total_coins_l419_419284


namespace cyclic_FGH_I_l419_419100

variables {A B C D E F G H I : Point}
variables {Triangle: triangle_class A B C}
variables {Altitudes: ∀ {X Y Z : Point}, altitude_class X Y Z}
variables {AD_altitude: Altitudes A D C}
variables {BE_altitude: Altitudes B E C}

theorem cyclic_FGH_I :
  is_acute_triangle A B C →  -- condition (1)
  foot_of_altitude B C = D →  -- condition (2)
  foot_of_altitude A C = E →  -- condition (2)
  on_segment A D F →          -- condition (3)
  on_segment B E G →          -- condition (3)
  ratio_eq (AF / FD) (BG / GE) → -- condition (3)
  intersection (line C F) (line B E) = H →  -- condition (4)
  intersection (line C G) (line A D) = I →  -- condition (5)
  cyclic F G H I :=             -- question
  sorry

end cyclic_FGH_I_l419_419100


namespace argument_friends_count_l419_419247

-- Define the conditions
def original_friends: ℕ := 20
def current_friends: ℕ := 19
def new_friend: ℕ := 1

-- Define the statement that needs to be proved
theorem argument_friends_count : 
  (original_friends + new_friend - current_friends = 1) :=
by
  -- Placeholder for the proof
  sorry

end argument_friends_count_l419_419247


namespace ali_total_money_l419_419410

-- Definitions based on conditions
def bills_of_5_dollars : ℕ := 7
def bills_of_10_dollars : ℕ := 1
def value_of_5_dollar_bill : ℕ := 5
def value_of_10_dollar_bill : ℕ := 10

-- Prove that Ali's total amount of money is $45
theorem ali_total_money : (bills_of_5_dollars * value_of_5_dollar_bill) + (bills_of_10_dollars * value_of_10_dollar_bill) = 45 := 
by
  sorry

end ali_total_money_l419_419410


namespace more_uniform_team_l419_419332

-- Define the parameters and the variances
def average_height := 1.85
def variance_team_A := 0.32
def variance_team_B := 0.26

-- Main theorem statement
theorem more_uniform_team : variance_team_B < variance_team_A → "Team B" = "Team with more uniform heights" :=
by
  -- Placeholder for the actual proof
  sorry

end more_uniform_team_l419_419332


namespace investment_worth_approx_28_years_l419_419017

noncomputable def investment_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) (tripling_period : ℝ) : ℝ :=
let triple_count := t / tripling_period in
P * (3 ^ (nat.floor triple_count))

theorem investment_worth_approx_28_years 
(P : ℝ := 2200) (r : ℝ := 0.08) (n : ℕ := 1) (t : ℝ := 28) 
(tripling_period : ℝ := 112 / 8) :
investment_value P r n t tripling_period ≈ 19800 :=
by
  sorry

end investment_worth_approx_28_years_l419_419017


namespace diagonals_of_octagon_l419_419543

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l419_419543


namespace part1_part2_l419_419511

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |2 * x - 1|

theorem part1 : {x : ℝ | f x ≤ 5} = {x : ℝ | -7 / 4 ≤ x ∧ x ≤ 3 / 4} :=
sorry

theorem part2 (h : ∃ x : ℝ, f x < |m - 2|) : m > 6 ∨ m < -2 :=
sorry

end part1_part2_l419_419511


namespace movie_theater_ticket_sales_l419_419060

theorem movie_theater_ticket_sales 
  (A C : ℤ) 
  (h1 : A + C = 900) 
  (h2 : 7 * A + 4 * C = 5100) : 
  A = 500 := 
sorry

end movie_theater_ticket_sales_l419_419060


namespace inequality_proof_l419_419609

variable {a b c : ℝ}

theorem inequality_proof (h : a > b) : (a / (c^2 + 1)) > (b / (c^2 + 1)) := by
  sorry

end inequality_proof_l419_419609


namespace bakery_profit_l419_419292

noncomputable def revenue_per_piece : ℝ := 4
noncomputable def pieces_per_pie : ℕ := 3
noncomputable def pies_per_hour : ℕ := 12
noncomputable def cost_per_pie : ℝ := 0.5

theorem bakery_profit (pieces_per_pie_pos : 0 < pieces_per_pie) 
                      (pies_per_hour_pos : 0 < pies_per_hour) 
                      (cost_per_pie_pos : 0 < cost_per_pie) :
  pies_per_hour * (pieces_per_pie * revenue_per_piece) - (pies_per_hour * cost_per_pie) = 138 := 
sorry

end bakery_profit_l419_419292


namespace product_a_n_eq_m_div_factorial_l419_419469

noncomputable def a_n (n : ℕ) (hn : n ≥ 3) : ℚ :=
  (n^2 + 2*n + 1) / (n^3 - 1)

theorem product_a_n_eq_m_div_factorial :
  (∏ n in finset.range' 3 48, a_n n (by linarith [n])) = (200 : ℚ) / (nat.factorial 50) :=
by
  sorry

end product_a_n_eq_m_div_factorial_l419_419469


namespace parabola_directrix_y_neg1_l419_419312

-- We define the problem given the conditions.
def parabola_directrix (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 = 4 * y → y = -p

-- Now we state what needs to be proved.
theorem parabola_directrix_y_neg1 : parabola_directrix 1 :=
by
  sorry

end parabola_directrix_y_neg1_l419_419312


namespace repeating_decimal_to_fraction_l419_419128

theorem repeating_decimal_to_fraction : 
  (∃ (x : ℚ), x = 7 + 3 / 9) → 7 + 3 / 9 = 22 / 3 :=
by
  intros h
  sorry

end repeating_decimal_to_fraction_l419_419128


namespace rounding_accuracy_6_18_times_10_to_4_l419_419420

theorem rounding_accuracy_6_18_times_10_to_4 :
  ∃ (place : String), place = "hundred place" ∧ 6.18 * 10^4 = 61800 ∧ 
  (∀ n : ℕ, 6.18 * 10^4 = n * 10^2) := sorry

end rounding_accuracy_6_18_times_10_to_4_l419_419420


namespace num_diagonals_octagon_l419_419575

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l419_419575


namespace part_I_part_II_l419_419507

noncomputable def f (k : ℝ) (x : ℝ) := Real.log (2 * x + 1) - k * x

theorem part_I (h : ∃ (k : ℝ), k > 0 ∧ ∀ x, deriv (f k) x = 0 → x = 0) : k = 2 := 
  sorry

theorem part_II (n : ℕ) (h : n > 0) (m : ℝ) (h₁ : m > 1) : 
  ∏ i in Finset.range n, (1 + 1 / m^(i+1)) < Real.exp (1 / (m - 1)) :=
  sorry

end part_I_part_II_l419_419507


namespace restaurant_total_pizzas_and_hotdogs_in_June_l419_419403

theorem restaurant_total_pizzas_and_hotdogs_in_June
  (hotdogs_daily : ℕ)
  (extra_pizzas : ℕ)
  (days_in_June : ℕ)
  (hotdogs_daily = 60)
  (extra_pizzas = 40)
  (days_in_June = 30) :
  (hotdogs_daily + extra_pizzas) * days_in_June = 4800 :=
by
  sorry

end restaurant_total_pizzas_and_hotdogs_in_June_l419_419403


namespace area_of_union_of_five_equilateral_triangles_l419_419139

theorem area_of_union_of_five_equilateral_triangles :
  let s := 4
  let area_of_one_triangle := (sqrt 3 / 4) * s^2
  let total_area_without_overlaps := 5 * area_of_one_triangle
  let overlapping_area_of_one_triangle := (sqrt 3 / 4) * (s / 2)^2
  let total_overlapping_area := 4 * overlapping_area_of_one_triangle
  let net_area := total_area_without_overlaps - total_overlapping_area
  in net_area = 16 * sqrt 3 := 
by
  sorry

end area_of_union_of_five_equilateral_triangles_l419_419139


namespace curve_cannot_intersect_each_edge_once_l419_419478

-- Define the graph structure and vertices/edges
structure Graph where
  vertices : Fin 12
  edges : Fin 16

-- Properties of the curve
structure Curve (G : Graph) where
  does_not_pass_through_vertex : Prop
  intersects_edge_exactly_once : Fin 16 → Prop

-- Hypotheses: Curve properties and graph structure
variables (G : Graph) (γ : Curve G)
  (h₁ : γ.does_not_pass_through_vertex)
  (h₂ : ∀ e, γ.intersects_edge_exactly_once e)

-- The theorem to be proven
theorem curve_cannot_intersect_each_edge_once :
    ∀ γ, γ.does_not_pass_through_vertex → (∃ e, ¬γ.intersects_edge_exactly_once e) :=
by
  intro γ h₁
  sorry

end curve_cannot_intersect_each_edge_once_l419_419478


namespace solve_for_x_l419_419623

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end solve_for_x_l419_419623


namespace num_integer_solutions_l419_419236

-- Definitions (conditions)
def line1 (x : ℝ) : ℝ := 2 * x - 1
def line2 (k x : ℝ) : ℝ := k * x + k

-- The intersection point must be an integer point
def is_integer_point (x y : ℤ) : Prop := 
  ∃ (kx : ℝ) (k : ℝ), line1 kx = y ∧ line2 k kx = y

-- Proof that there are exactly 4 integer values of k satisfying the condition
theorem num_integer_solutions : 
  ∃ (n : ℕ), n = 4 ∧
  ∀ (k : ℤ), 
  (∀ (x y : ℤ), is_integer_point x y → (k = -1 ∨ k = 1 ∨ k = 3 ∨ k = 5)) :=
begin
  sorry
end

end num_integer_solutions_l419_419236


namespace restaurant_total_pizzas_and_hotdogs_in_June_l419_419404

theorem restaurant_total_pizzas_and_hotdogs_in_June
  (hotdogs_daily : ℕ)
  (extra_pizzas : ℕ)
  (days_in_June : ℕ)
  (hotdogs_daily = 60)
  (extra_pizzas = 40)
  (days_in_June = 30) :
  (hotdogs_daily + extra_pizzas) * days_in_June = 4800 :=
by
  sorry

end restaurant_total_pizzas_and_hotdogs_in_June_l419_419404


namespace find_sum_l419_419890

theorem find_sum (P : ℕ) (h_total : P * (4/100 + 6/100 + 8/100) = 2700) : P = 15000 :=
by
  sorry

end find_sum_l419_419890


namespace find_volume_of_12_percent_solution_l419_419016

variable (x y : ℝ)

theorem find_volume_of_12_percent_solution
  (h1 : x + y = 60)
  (h2 : 0.02 * x + 0.12 * y = 3) :
  y = 18 := 
sorry

end find_volume_of_12_percent_solution_l419_419016


namespace octagon_has_20_diagonals_l419_419580

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l419_419580


namespace solve_for_y_l419_419145

theorem solve_for_y (x y : ℝ) : 3 * x + 5 * y = 10 → y = 2 - (3 / 5) * x :=
by 
  -- proof steps would be filled here
  sorry

end solve_for_y_l419_419145


namespace extreme_values_of_f_comparison_l419_419506

noncomputable def f (x a : ℝ) : ℝ := log x - a * x^2

noncomputable def g (x a : ℝ) : ℝ := f x a + a * x^2 - x

theorem extreme_values_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, x > 0 → ¬(∃ z : ℝ, x > z ∧ f z a > f x a ∧ f z a > f x a)) ∧
  (a > 0 → ∃ x_max, (∀ x : ℝ, x > 0 → x ≤ x_max → f x a ≤ f x_max a) ∧
    f (1 / (real.sqrt (2 * a))) a = - (1 / 2) * (log (2 * a) + 1)) :=
sorry

theorem comparison (x1 x2 a : ℝ) (h1 : x1 > x2) (h2 : x2 > 0) :
  (x1 / (x1^2 + x2^2) - (g x1 a - g x2 a) / (x1 - x2)) < 1 :=
sorry

end extreme_values_of_f_comparison_l419_419506


namespace range_of_a_to_make_f_increasing_l419_419153

def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a ^ x

theorem range_of_a_to_make_f_increasing :
  {a | a > 0 ∧ a ≠ 1 ∧ (∀ x y : ℝ, x < y → f x a < f y a)} = {a | 3 / 2 ≤ a ∧ a < 2} :=
by
  sorry

end range_of_a_to_make_f_increasing_l419_419153


namespace octagon_has_20_diagonals_l419_419564

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l419_419564


namespace arithmetic_sequence_problem_l419_419650

open BigOperators

structure arithmetic_sequence (a : ℕ → ℚ) : Prop :=
(common_difference : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d)

def sum_to_n (a : ℕ → ℚ) (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), a i

theorem arithmetic_sequence_problem (a : ℕ → ℚ)
  (h_seq : arithmetic_sequence a) 
  (h_sum_relation : sum_to_n a 84 = 4 * sum_to_n a 41) :
  a 7 + a 8 + a 19 + a 20 = 12 :=
sorry

end arithmetic_sequence_problem_l419_419650


namespace total_rounds_proof_l419_419387

-- Define the initial token counts
def initial_tokens : (ℕ × ℕ × ℕ) := (15, 14, 13)

-- Define the game rule as a function that takes the token counts and returns the updated counts after one round
def game_rule : ℕ × ℕ × ℕ → ℕ × ℕ × ℕ
| (a, b, c) :=
  if a ≥ b ∧ a ≥ c then
    (a - 3, b + 1, c + 1)
  else if b ≥ a ∧ b ≥ c then
    (a + 1, b - 3, c + 1)
  else
    (a + 1, b + 1, c - 3)

-- Define the function to compute the total number of rounds
def total_rounds : ℕ := 37

-- Theorem stating the total number of rounds until a player runs out of tokens is 37
theorem total_rounds_proof :
  (∃ rounds: ℕ, rounds = total_rounds ∧
   (let final_counts := (list.repeat game_rule total_rounds).foldl (λ acc f, f acc) initial_tokens
    in final_counts.1 = 0 ∨ final_counts.2 = 0 ∨ final_counts.3 = 0)) := by
  sorry

end total_rounds_proof_l419_419387


namespace game_ends_after_37_rounds_l419_419385

-- Type definition for players
inductive Player
| A
| B
| C

-- Function to get the initial tokens
def initialTokens : Player → ℕ
| Player.A => 15
| Player.B => 14
| Player.C => 13

-- Function to simulate one round
def round (tokens : Player → ℕ) (mostTokensPlayer : Player) : Player → ℕ
| Player.A =>
  if mostTokensPlayer = Player.A then tokens Player.A - 3 else tokens Player.A + 1
| Player.B =>
  if mostTokensPlayer = Player.B then tokens Player.B - 3 else tokens Player.B + 1
| Player.C =>
  if mostTokensPlayer = Player.C then tokens Player.C - 3 else tokens Player.C + 1

-- Function to find the player with the most tokens
def mostTokensPlayer (tokens : Player → ℕ) : Player :=
if tokens Player.A ≥ tokens Player.B ∧ tokens Player.A ≥ tokens Player.C then Player.A
else if tokens Player.B ≥ tokens Player.C then Player.B
else Player.C

-- Function to simulate multiple rounds
def simulateRounds (tokens : Player → ℕ) (rounds : ℕ) : Player → ℕ :=
match rounds with
| 0 => tokens
| n + 1 =>
  let mPlayer := mostTokensPlayer tokens
  simulateRounds (round tokens mPlayer) n

-- Proof that the game ends after 37 rounds
theorem game_ends_after_37_rounds :
  ∃ n, ∀ tokens : Player → ℕ,
  tokens = initialTokens →
  simulateRounds tokens 37 Player.A = 0 ∨
  simulateRounds tokens 37 Player.B = 0 ∨
  simulateRounds tokens 37 Player.C = 0 :=
by
  sorry

end game_ends_after_37_rounds_l419_419385


namespace inradius_right_triangle_l419_419664

theorem inradius_right_triangle (a b c : ℝ) (h₁ : (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 4 ∧ b = 3 ∧ c = 5)) :
  let s := (a + b + c) / 2 in
  let A := s * (s - a) * (s - b) * (s - c) in
  sqrt A / s = 1 :=
by
  sorry

end inradius_right_triangle_l419_419664


namespace no_absolute_winner_probability_l419_419921

-- Define the probabilities of matches
def P_AB : ℝ := 0.6  -- Probability Alyosha wins against Borya
def P_BV : ℝ := 0.4  -- Probability Borya wins against Vasya

-- Define the event C that there is no absolute winner
def event_C (P_AV : ℝ) (P_VB : ℝ) : ℝ :=
  let scenario1 := P_AB * P_BV * P_AV in
  let scenario2 := P_AB * P_VB * (1 - P_AV) in
  scenario1 + scenario2

-- Main theorem to prove
theorem no_absolute_winner_probability : 
  event_C 1 0.6 = 0.24 :=
by
  rw [event_C]
  simp
  norm_num
  sorry

end no_absolute_winner_probability_l419_419921


namespace candy_bar_calories_l419_419758

theorem candy_bar_calories:
  ∀ (calories_per_candy_bar : ℕ) (num_candy_bars : ℕ), 
  calories_per_candy_bar = 3 → 
  num_candy_bars = 5 → 
  calories_per_candy_bar * num_candy_bars = 15 :=
by
  sorry

end candy_bar_calories_l419_419758


namespace car_speed_problem_l419_419770

theorem car_speed_problem
  (v : ℝ)
  (time := 3.5)
  (distance := 385)
  (relative_speed := 52 + v) :
  relative_speed * time = distance → v = 58 :=
by
  intros h
  have h1 : 52 + v = (385 / 3.5) := sorry
  have h2 : 203 = 52 + v * 3.5 := sorry
  have h3 : v * 3.5 = 203 - 182 := sorry
  have h4 : v = (203 - 182) / 3.5 := sorry
  exact h4 -- actually proving the result using basic algebra

-- The proof steps to solve for v would go here using Lean tactics

end car_speed_problem_l419_419770


namespace combinations_x_eq_2_or_8_l419_419601

theorem combinations_x_eq_2_or_8 (x : ℕ) (h_pos : 0 < x) (h_comb : Nat.choose 10 x = Nat.choose 10 2) : x = 2 ∨ x = 8 :=
sorry

end combinations_x_eq_2_or_8_l419_419601


namespace sequence_a_2014_l419_419308

noncomputable def a_n (n : ℕ) : ℕ := sorry

theorem sequence_a_2014 :
  (∀ n : ℕ, a_n n ≤ a_n (n + 1)) →
  (∀ k : ℕ, k > 0 → ∑ i in finset.range (2 * k - 1 + 1), (a_n i = k) = 2 * k - 1) →
  a_n 2014 = 45 :=
by
  sorry

end sequence_a_2014_l419_419308


namespace optimal_removal_maximizes_probability_l419_419778

theorem optimal_removal_maximizes_probability :
  ∀ (lst : List ℤ) (n : ℤ),
  lst = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] →
  list.length (list.filter (λ (x : ℤ) (y : ℤ), x ≠ y ∧ x + y = 12) (list.combine lst lst)) <
  list.length (list.filter (λ (x : ℤ) (y : ℤ), x ≠ y ∧ x + y = 12) (list.combine (list.filter (≠ 6) lst))) := sorry

end optimal_removal_maximizes_probability_l419_419778


namespace vertex_angle_double_angle_triangle_l419_419950

theorem vertex_angle_double_angle_triangle 
  {α β : ℝ} (h1 : α + β + β = 180) (h2 : α = 2 * β ∨ β = 2 * α) :
  α = 36 ∨ α = 90 :=
by
  sorry

end vertex_angle_double_angle_triangle_l419_419950


namespace smallest_possible_third_term_l419_419068

theorem smallest_possible_third_term :
  ∃ (d : ℝ), (d = -3 + Real.sqrt 134 ∨ d = -3 - Real.sqrt 134) ∧ 
  (7, 7 + d + 3, 7 + 2 * d + 18) = (7, 10 + d, 25 + 2 * d) ∧ 
  min (25 + 2 * (-3 + Real.sqrt 134)) (25 + 2 * (-3 - Real.sqrt 134)) = 19 + 2 * Real.sqrt 134 :=
by
  sorry

end smallest_possible_third_term_l419_419068


namespace find_x_l419_419460

theorem find_x :
  (∃ x : ℝ, real.cbrt (5 - x/3) = -4) → x = 207 :=
by
  intro hx
  rcases hx with ⟨x, H⟩
  sorry

end find_x_l419_419460


namespace sufficient_but_not_necessary_not_necessary_l419_419369

-- Conditions
def condition_1 (x : ℝ) : Prop := x > 3
def condition_2 (x : ℝ) : Prop := x^2 - 5 * x + 6 > 0

-- Theorem statement
theorem sufficient_but_not_necessary (x : ℝ) : condition_1 x → condition_2 x :=
sorry

theorem not_necessary (x : ℝ) : condition_2 x → ∃ y : ℝ, ¬ condition_1 y ∧ condition_2 y :=
sorry

end sufficient_but_not_necessary_not_necessary_l419_419369


namespace cylinder_radius_original_l419_419244

theorem cylinder_radius_original (r : ℝ) (h : ℝ) (h_given : h = 4) 
    (V_increase_radius : π * (r + 4) ^ 2 * h = π * r ^ 2 * (h + 4)) : 
    r = 12 := 
  by
    sorry

end cylinder_radius_original_l419_419244


namespace count_valid_integers_how_many_valid_integers_l419_419596

theorem count_valid_integers (m : Int) : (m ≠ 0 ∧ (1 / (|m| : ℝ)) ≥ (1 / 6)) → m ≠ 0 → sorry :=
begin
  sorry
end

theorem how_many_valid_integers : ∃! n : Int, n = 12 :=
begin
  use 12,
  split,
  -- Proof that 12 is the only solution goes here (omitted as per instruction)
  sorry,
  intro b,
  -- Proof that if b = 12, then the premise holds (omitted as per instruction)
  sorry,
end

end count_valid_integers_how_many_valid_integers_l419_419596


namespace intersection_of_A_and_B_l419_419701

def A : Set ℕ := {a, b, c, d, e} -- Note: Adjust types if necessary (e.g., ℤ or α)
def B : Set ℕ := {d, f, g}      -- Note: Adjust types if necessary (e.g., ℤ or α)

theorem intersection_of_A_and_B : A ∩ B = {d} :=
by
  sorry

end intersection_of_A_and_B_l419_419701


namespace octagon_diagonals_l419_419537

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l419_419537


namespace symmetry_center_tan_l419_419755

open Real

theorem symmetry_center_tan (k : ℤ) : 
  let x := (k * π / 4 - π / 8) in
  ∃ c : ℝ, y = tan (2 * x + π / 4) →
  (c, 0) = (k * π / 4 - π / 8, 0) :=
by 
  intro k
  let x := k * π / 4 - π / 8
  exists x
  intro h
  sorry

end symmetry_center_tan_l419_419755


namespace isosceles_triangle_circumcircle_area_l419_419862

noncomputable def area_of_circumcircle (a b c : ℝ) : ℝ :=
  let BD := real.sqrt (a^2 - ((c / 2)^2))
  let OD := (2 / 3) * BD
  let r := real.sqrt (a^2 - OD^2)
  real.pi * r^2

theorem isosceles_triangle_circumcircle_area :
  area_of_circumcircle 4 4 3 = 9.8889 * real.pi :=
sorry

end isosceles_triangle_circumcircle_area_l419_419862


namespace area_of_circumcircle_of_isosceles_triangle_l419_419837

theorem area_of_circumcircle_of_isosceles_triangle :
  let AB := 4
  let AC := 4
  let BC := 3
  let AD := (√(AB^2 - (BC/2)^2))
  let radius := AD
  let area := π * radius^2
  area = 16 * π :=
by
  sorry

end area_of_circumcircle_of_isosceles_triangle_l419_419837


namespace octagon_has_20_diagonals_l419_419577

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l419_419577


namespace triangle_max_area_l419_419160

def area_max (a b c : ℝ) : ℝ := (1 / 2) * b * c * Real.sin (Real.pi / 3)

theorem triangle_max_area (a b c : ℝ) (h1 : a = 2) (h2 : (a - b + c) / c = b / (a + b - c)) :
  ∃ b c, area_max a b c = Real.sqrt 3 := sorry

end triangle_max_area_l419_419160


namespace trigonometric_proof_l419_419489

theorem trigonometric_proof (α : Real) (h_cos : cos α = -sqrt 5 / 5) (h_interval : π < α ∧ α < 3 * π / 2) :
  (sin α = -2 * sqrt 5 / 5) ∧ (sin (π + α) + 2 * sin (3 * π / 2 + α)) / (cos (3 * π - α) + 1) = sqrt 5 - 1 :=
by
  sorry

end trigonometric_proof_l419_419489


namespace circle_equation_radius_l419_419473

theorem circle_equation_radius (k : ℝ) :
  (∃ x y : ℝ, x^2 + 6 * x + y^2 + 8 * y - k = 0) -> k = 75 :=
begin
  intro h,
  sorry
end

end circle_equation_radius_l419_419473


namespace fish_fishermen_problem_l419_419762

theorem fish_fishermen_problem (h: ℕ) (r: ℕ) (w_h: ℕ) (w_r: ℕ) (claimed_weight: ℕ) (total_real_weight: ℕ) 
  (total_fishermen: ℕ) :
  -- conditions
  (claimed_weight = 60) →
  (total_real_weight = 120) →
  (total_fishermen = 10) →
  (w_h = 30) →
  (w_r < 60 / 7) →
  (h + r = total_fishermen) →
  (2 * w_h * h + r * claimed_weight = claimed_weight * total_fishermen) →
  -- prove the number of regular fishermen
  (r = 7 ∨ r = 8) :=
sorry

end fish_fishermen_problem_l419_419762


namespace clock_time_l419_419112

-- Let's define the conditions first.
def same_length_hands : Prop := ∀ (h m s : ℕ), h = m ∨ h = s ∨ m = s
def no_numbers_on_dial : Prop := true
def unclear_top_position : Prop := true
def no_coinciding_hands : Prop := true

-- The target theorem to prove.
theorem clock_time : same_length_hands ∧ no_numbers_on_dial ∧ unclear_top_position ∧ no_coinciding_hands → (∃ (h m : ℕ), h = 4 ∧ m = 50) :=
begin
  sorry
end

end clock_time_l419_419112


namespace trajectory_of_M_max_area_difference_l419_419379

-- Step 1: Definitions based on conditions (circle equation, points, etc.)
def circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0
def F1 : ℝ × ℝ := (1, 0)
def F2 : ℝ × ℝ := (-1, 0)
def radius : ℝ := 4

-- Statement for question 1 (trajectory of point M)
theorem trajectory_of_M :
  ∀ (M : ℝ × ℝ), circle M.1 M.2 →
  (∃ T, (T intersects the circle at P and Q) ∧ 
         (the perpendicular bisector of segment PF2 intersects segment PF1 at M) ∧
         |dist M F1 + dist M F2| = radius ∧
         (T corresponds to the ellipse equation (x^2)/4 + (y^2)/3 = 1)) := 
sorry

-- Step 2: Definitions based on the additional line intersections and calculations
def line_l (m y : ℝ) : ℝ := m*y - 1
def trajectory (x y : ℝ) : Prop := (x^2)/4 + (y^2)/3 = 1

-- Statement for question 2 (maximum value of S1 - S2)
theorem max_area_difference :
  ∀ (S1 S2 : ℝ) (m : ℝ),
  |S1 - S2| = ∃ (AB : ℝ), 1/2 * |AB| * (|y1 + y2|) * 12|m|/(4 + 3 * m^2) ∧
  (line_l intersects trajectory at points C and D)  ∧
  max (|S1 - S2|) = √3 :=
sorry

end trajectory_of_M_max_area_difference_l419_419379


namespace isosceles_triangle_circumcircle_area_l419_419865

noncomputable def area_of_circumcircle (a b c : ℝ) : ℝ :=
  let BD := real.sqrt (a^2 - ((c / 2)^2))
  let OD := (2 / 3) * BD
  let r := real.sqrt (a^2 - OD^2)
  real.pi * r^2

theorem isosceles_triangle_circumcircle_area :
  area_of_circumcircle 4 4 3 = 9.8889 * real.pi :=
sorry

end isosceles_triangle_circumcircle_area_l419_419865


namespace no_absolute_winner_prob_l419_419898

open_locale probability

-- Define the probability of Alyosha winning against Borya
def P_A_wins_B : ℝ := 0.6

-- Define the probability of Borya winning against Vasya
def P_B_wins_V : ℝ := 0.4

-- There are no ties, and each player plays with each other once
-- Conditions ensure that all pairs have played exactly once

-- Define the event that there will be no absolute winner
def P_no_absolute_winner : ℝ := P_A_wins_B * P_B_wins_V * 1 + P_A_wins_B * (1 - P_B_wins_V) * (1 - 1)

-- Statement of the problem: Prove that the probability of event C is 0.24
theorem no_absolute_winner_prob :
  P_no_absolute_winner = 0.24 :=
  by
    -- Placeholder for proof
    sorry

end no_absolute_winner_prob_l419_419898


namespace best_in_district_round_l419_419710

-- Assume a structure that lets us refer to positions
inductive Position
| first
| second
| third
| last

open Position

-- Definitions of the statements
def Eva (p : Position → Prop) := ¬ (p first) ∧ ¬ (p last)
def Mojmir (p : Position → Prop) := ¬ (p last)
def Karel (p : Position → Prop) := p first
def Peter (p : Position → Prop) := p last

-- The main hypothesis
def exactly_one_lie (p : Position → Prop) :=
  (Eva p ∧ Mojmir p ∧ Karel p ∧ ¬ (Peter p)) ∨
  (Eva p ∧ Mojmir p ∧ ¬ (Karel p) ∧ Peter p) ∨
  (Eva p ∧ ¬ (Mojmir p) ∧ Karel p ∧ Peter p) ∨
  (¬ (Eva p) ∧ Mojmir p ∧ Karel p ∧ Peter p)

theorem best_in_district_round :
  ∃ (p : Position → Prop),
    (Eva p ∧ Mojmir p ∧ ¬ (Karel p) ∧ Peter p) ∧ exactly_one_lie p :=
by
  sorry

end best_in_district_round_l419_419710


namespace subsets_of_M_l419_419989

noncomputable def discriminant (a : ℝ) : ℝ :=
  4 * (a + 1) ^ 2 - 4

def num_subsets (a : ℝ) (h : a ≠ 0) : ℕ :=
  if h1 : discriminant a > 0 then 4
  else if h2 : discriminant a = 0 then 2
  else 1

theorem subsets_of_M (a : ℝ) (h : a ≠ 0) :
  num_subsets a h = 1 ∨ num_subsets a h = 2 ∨ num_subsets a h = 4 :=
sorry

end subsets_of_M_l419_419989


namespace determine_blue_numbers_l419_419709

-- Problem statement rewritten as a Lean 4 formal statement
theorem determine_blue_numbers (f : Fin 1001 → Fin 501) :
  ∃ g : Fin 1001 → Fin 1001, 
    (∀ C : Fin 1001, f C = (card (filter (λ x : Fin 1001, C < x) (take 500 (drop (C + 1) (range 1001)))))) ∧
    (injective g ∧ (∀ C : Fin 1001, ((g ∘ f) C) = C)) :=
sorry

end determine_blue_numbers_l419_419709


namespace trig_identity_example_l419_419109

theorem trig_identity_example : 
  cos (Real.pi / 180 * 25) * cos (Real.pi / 180 * 85) + sin (Real.pi / 180 * 25) * sin (Real.pi / 180 * 85) = 1 / 2 :=
by
  sorry

end trig_identity_example_l419_419109


namespace sqrt_sum_simplification_l419_419793

theorem sqrt_sum_simplification : 
  sqrt (20 - 8 * sqrt 5) + sqrt (20 + 8 * sqrt 5) = 2 * sqrt 14 := 
by
  sorry

end sqrt_sum_simplification_l419_419793


namespace cos_angle_GAC_is_sqrt3_div_3_l419_419374

-- Define a structure for a cube and its side length
structure Cube :=
  (side_length : ℝ)
  (diagonal_length : ℝ := side_length * sqrt 3)
  (edge_length : ℝ := side_length)

-- Define the cosine function for the angle GAC
def cos_angle_GAC (cube : Cube) : ℝ :=
  cube.edge_length / cube.diagonal_length

-- The theorem to prove
theorem cos_angle_GAC_is_sqrt3_div_3 (cube : Cube) :
  cos_angle_GAC cube = sqrt (3 : ℝ) / 3 := by
  sorry

end cos_angle_GAC_is_sqrt3_div_3_l419_419374


namespace hyperbola_eccentricity_l419_419631

theorem hyperbola_eccentricity {a b c : ℝ} (ha : a > 0) (hb : b > 0)
    (h1 : ∀ x : ℝ, y = (b / a) * x → y = x^2 + 1 / 16) :
        4 * c^2 = 5 * a^2 → eccentricity = sqrt 5 / 2 :=
begin
  sorry
end

end hyperbola_eccentricity_l419_419631


namespace interval_with_highest_average_speed_l419_419735

-- Definitions according to conditions
def ΔDistance (start_time end_time : ℝ) : ℝ := -- Assume a function that gives the change in distance
  sorry

def average_speed (start_time end_time : ℝ) : ℝ :=
  ΔDistance start_time end_time / (end_time - start_time)

noncomputable def highest_average_speed_interval : ℝ :=
  let intervals := [(0, 2), (3, 5), (4, 6)]
  let avg_speeds := intervals.map (λ interval, average_speed interval.1 interval.2)
  intervals.zip avg_speeds |> list.maximum_by (λ pair, pair.2) |> (λ pair, pair.1.1)

theorem interval_with_highest_average_speed :
  highest_average_speed_interval = 3 :=
by
  sorry

end interval_with_highest_average_speed_l419_419735


namespace area_of_region_l419_419462
noncomputable theory

def fractional_part (x : ℝ) : ℝ := x - x.floor

theorem area_of_region :
  (∫ x in 0..∞, ∫ y in 0..∞, if 120 * fractional_part x ≥ 2 * (x.floor : ℝ) + (y.floor : ℝ) then 1 else 0) = 295860 :=
by 
  sorry

end area_of_region_l419_419462


namespace sqrt_sum_eq_l419_419799

theorem sqrt_sum_eq : 
  sqrt (20 - 8 * sqrt 5) + sqrt (20 + 8 * sqrt 5) = 2 * sqrt 10 := 
by 
  sorry

end sqrt_sum_eq_l419_419799


namespace find_k_l419_419270

-- Definitions based on conditions
def g (x : ℕ) (a b c : ℤ) := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (g : ℕ → ℤ := g)
  (g2_zero : g 2 a b c = 0)
  (g9_bounds : 110 < g 9 a b c ∧ g 9 a b c < 120)
  (g10_bounds : 130 < g 10 a b c ∧ g 10 a b c < 140)
  (k : ℤ)
  (g100_bounds : 6000 * k < g 100 a b c ∧ g 100 a b c < 6000 * (k + 1)) :
  k = 1 :=
begin
  -- Proof goes here
  sorry
end

end find_k_l419_419270


namespace members_of_A_l419_419364

variable {U : Type}  -- U is a type representing the universal set

-- Define the relevant sets
variable (A B : Set U)

-- Conditions given in the problem
variable (hU : card U = 192)
variable (hB : card B = 49)
variable (hNeither : card (U \ (A ∪ B)) = 59)
variable (hA_inter_B : card (A ∩ B) = 23)

-- The statement to prove
theorem members_of_A (h_card_U : ∑ u in U, 1 = 192)
                     (h_card_B : ∑ b in B, 1 = 49)
                     (h_card_neither : ∑ u in (U \ A ∪ B), 1 = 59)
                     (h_card_A_inter_B : ∑ a in A ∩ B, 1 = 23) :
  ∑ a in A, 1 = 107 :=
 by
 sorry

end members_of_A_l419_419364


namespace find_k_l419_419928

open BigOperators

noncomputable
def hyperbola_property (k : ℝ) (x a b c : ℝ) : Prop :=
  k > 0 ∧
  (a / 2, b / 2) = (a / 2, k / a / 2) ∧ -- midpoint condition
  abs (a * b) / 2 = 3 ∧                -- area condition
  b = k / a                            -- point B on the hyperbola

theorem find_k (k : ℝ) (x a b c : ℝ) : hyperbola_property k x a b c → k = 2 :=
by
  sorry

end find_k_l419_419928


namespace gcd_factorial_8_10_l419_419783

theorem gcd_factorial_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end gcd_factorial_8_10_l419_419783


namespace trig_solution_l419_419306

noncomputable def solve_trig_equation (x : ℝ) : Prop :=
  sin x ^ 4 + cos x ^ 4 - 2 * sin (2 * x) + (3 / 4) * sin (2 * x) ^ 2 = 0

theorem trig_solution : ∃ (n : ℤ), 
  (solve_trig_equation (16.2 * (π / 180) + 180 * n * (π / 180)) ∧ solve_trig_equation (73.8 * (π / 180) + 180 * n * (π / 180))) :=
begin
  sorry,
end

end trig_solution_l419_419306


namespace area_of_circumcircle_of_isosceles_triangle_l419_419836

theorem area_of_circumcircle_of_isosceles_triangle :
  let AB := 4
  let AC := 4
  let BC := 3
  let AD := (√(AB^2 - (BC/2)^2))
  let radius := AD
  let area := π * radius^2
  area = 16 * π :=
by
  sorry

end area_of_circumcircle_of_isosceles_triangle_l419_419836


namespace cost_of_pencil_l419_419977

variables (x y : ℚ)

lemma pencil_cost_equation1 : 5 * x + 4 * y = 340 := sorry
lemma pencil_cost_equation2 : 3 * x + 6 * y = 264 := sorry

theorem cost_of_pencil : y = 50 / 3 := by
  have h1 : 5 * x + 4 * y = 340 := pencil_cost_equation1
  have h2 : 3 * x + 6 * y = 264 := pencil_cost_equation2
  -- rest of the proof omitted
  sorry

end cost_of_pencil_l419_419977


namespace octagon_diagonals_l419_419536

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l419_419536


namespace octagon_diagonals_l419_419591

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l419_419591


namespace verify_f_l419_419689

def f (x : ℝ) : ℝ :=
if x ≠ 1 then 2 / (1 - x) else 1

theorem verify_f (x : ℝ) : f x + x * f (2 - x) = 2 :=
by
  simp [f];
  split_ifs with h;
  {
    sorry
  };
  {
    sorry
  }

end verify_f_l419_419689


namespace mixed_oil_rate_l419_419212

def rate_per_litre_mixed_oil (v1 v2 v3 v4 p1 p2 p3 p4 : ℕ) :=
  (v1 * p1 + v2 * p2 + v3 * p3 + v4 * p4) / (v1 + v2 + v3 + v4)

theorem mixed_oil_rate :
  rate_per_litre_mixed_oil 10 5 8 7 50 68 42 62 = 53.67 :=
by
  sorry

end mixed_oil_rate_l419_419212


namespace axis_of_symmetry_is_correct_l419_419731

noncomputable def axis_of_symmetry : ℝ :=
  let f (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)
  π / 12

theorem axis_of_symmetry_is_correct :
  axis_of_symmetry = π / 12 :=
sorry

end axis_of_symmetry_is_correct_l419_419731


namespace part_I_part_II_l419_419184

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := m * x * real.log x + x^2

theorem part_I (m : ℝ) (h_m : m = -2) :
  ∀ x > 0, (derivative (f x (-2))) x > 0 := sorry

theorem part_II {x1 x2 : ℝ} (m : ℝ) (h_f_x1 : f x1 m = 0) (h_f_x2 : f x2 m = 0) (h x_ne_x : x1 ≠ x2) :
  real.log x1 + real.log x2 > 2 := sorry

end part_I_part_II_l419_419184


namespace relationship_between_a_b_c_l419_419484

noncomputable def a : ℝ := real.sqrt (1 / 2)
noncomputable def b : ℝ := real.log 3 / real.log 2
noncomputable def c : ℝ := real.log 7 / (2 * real.log 2)

theorem relationship_between_a_b_c : a < c ∧ c < b :=
by
  -- We start by noting that a < 1
  have ha : a < 1 := real.sqrt_lt (1 / 2) 1 (by norm_num),
  -- Next, note that c > 1 since log_2 (sqrt 7) > log_2 2
  have hc : c > 1 := by
    rw c,
    exact (div_pos (real.log_pos (by norm_num: 7 > 1)) (mul_pos (by norm_num: 2 > 0) (real.log_pos (by norm_num: 2 > 1)))),
  -- Finally we use the fact that the logarithm is an increasing function to show that c > b
  have hb : c > b := by
    rw [b, c, div_lt_div_iff, real.log_lt_log (by norm_num: 3 > 0) (real.sqrt_pos.2 (by norm_num: 7 > 0)) (by norm_num: 2 > 0)],
    exact (by norm_num),
    exact (real.log_pos (by norm_num: 2 > 1)),
  exact ⟨ha.trans hc, hc.trans hb⟩,

end relationship_between_a_b_c_l419_419484


namespace average_of_first_5_primes_gt_50_l419_419966

theorem average_of_first_5_primes_gt_50 : 
  let primes := [53, 59, 61, 67, 71] in
  ∀ n ∈ primes, Prime n ∧ 50 < n → 
  (primes.sum / primes.length = 62.2) :=
by 
  intros primes h_prime h_gt_50
  sorry

end average_of_first_5_primes_gt_50_l419_419966


namespace octagon_has_20_diagonals_l419_419584

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l419_419584


namespace vinnie_exceeded_words_by_l419_419776

def words_on_friday := 450
def words_on_saturday := 650
def words_on_sunday := 300

def articles_on_friday := 25
def articles_on_saturday := 40
def articles_on_sunday := 15

def total_word_limit := 1000

-- The theorem we want to prove
theorem vinnie_exceeded_words_by :
  let total_words := words_on_friday + words_on_saturday + words_on_sunday
  let total_articles := articles_on_friday + articles_on_saturday + articles_on_sunday
  let words_without_articles := total_words - total_articles
  words_without_articles > total_word_limit → words_without_articles - total_word_limit = 320 := 
by
  let total_words := words_on_friday + words_on_saturday + words_on_sunday
  let total_articles := articles_on_friday + articles_on_saturday + articles_on_sunday
  let words_without_articles := total_words - total_articles
  intro h
  have : words_without_articles = 1320 := by
    simp [total_words, total_articles, words_without_articles]
  rw this at h
  simp [total_word_limit, words_without_articles]
  sorry

end vinnie_exceeded_words_by_l419_419776


namespace total_songs_in_june_l419_419777

-- Define the conditions
def Vivian_daily_songs : ℕ := 10
def Clara_daily_songs : ℕ := Vivian_daily_songs - 2
def Lucas_daily_songs : ℕ := Vivian_daily_songs + 5
def total_play_days_in_june : ℕ := 30 - 8 - 1

-- Total songs listened to in June
def total_songs_Vivian : ℕ := Vivian_daily_songs * total_play_days_in_june
def total_songs_Clara : ℕ := Clara_daily_songs * total_play_days_in_june
def total_songs_Lucas : ℕ := Lucas_daily_songs * total_play_days_in_june

-- The total number of songs listened to by all three
def total_songs_all_three : ℕ := total_songs_Vivian + total_songs_Clara + total_songs_Lucas

-- The proof problem
theorem total_songs_in_june : total_songs_all_three = 693 := by
  -- Placeholder for the proof
  sorry

end total_songs_in_june_l419_419777


namespace min_distance_curve_points_l419_419167

open Real

theorem min_distance_curve_points :
  (∀ x : ℝ, min_dist (P := (x, exp x)) (Q := (log x)) (PQ := P - Q) = sqrt 2) := 
sorry

end min_distance_curve_points_l419_419167


namespace line_intersects_circle_at_two_points_l419_419498

-- Definitions based on given conditions
def radius (r : ℝ) : Prop := r = 6.5
def distance_from_center_to_line (d : ℝ) : Prop := d = 4.5

-- Theorem statement
theorem line_intersects_circle_at_two_points (r d : ℝ) (hr : radius r) (hd : distance_from_center_to_line d) : 
  d < r → ∃(p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ (∥p1∥ = r ∧ ∥p2∥ = r) := 
by 
  sorry

end line_intersects_circle_at_two_points_l419_419498


namespace probability_of_divisible_by_7_is_13_over_90_l419_419214

def is_six_digit_sum_eq_31 (n : ℕ) : Prop :=
  let digits := n.digits 10
  n >= 100000 ∧ n < 1000000 ∧ digits.sum = 31

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

noncomputable def probability_divisible_by_7 : ℚ :=
  let all_numbers := {n | is_six_digit_sum_eq_31 n}
  let numbers_div_by_7 := {n | is_six_digit_sum_eq_31 n ∧ is_divisible_by_7 n}
  (numbers_div_by_7.card : ℚ) / (all_numbers.card : ℚ)

theorem probability_of_divisible_by_7_is_13_over_90 :
  probability_divisible_by_7 = 13 / 90 := sorry

end probability_of_divisible_by_7_is_13_over_90_l419_419214


namespace find_x_l419_419614

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end find_x_l419_419614


namespace case1_DC_correct_case2_DC_correct_l419_419105

-- Case 1
theorem case1_DC_correct (AB AD : ℝ) (HM BD DH MD : ℝ) (hAB : AB = 10) (hAD : AD = 4)
  (hHM : HM = 6 / 5) (hBD : BD = 2 * Real.sqrt 21) (hDH : DH = 4 * Real.sqrt 21 / 5)
  (hMD : MD = 6 * (Real.sqrt 21 - 1) / 5):
  (BD - HM : ℝ) == (8 * Real.sqrt 21 - 12) / 5 :=
by {
  sorry
}

-- Case 2
theorem case2_DC_correct (AB AD : ℝ) (HM BD DH MD : ℝ) (hAB : AB = 8 * Real.sqrt 2) (hAD : AD = 4)
  (hHM : HM = Real.sqrt 2) (hBD : BD = 4 * Real.sqrt 7) (hDH : DH = Real.sqrt 14)
  (hMD : MD = Real.sqrt 14 - Real.sqrt 2):
  (BD - HM : ℝ) == 2 * Real.sqrt 14 - 2 * Real.sqrt 2 :=
by {
  sorry
}

end case1_DC_correct_case2_DC_correct_l419_419105


namespace point_in_second_quadrant_l419_419477

-- Define the complex numbers z1 and z2
def z1 : ℂ := 2 + complex.I
def z2 : ℂ := 1 + 2 * complex.I

-- Define the complex number z as the difference between z2 and z1
def z : ℂ := z2 - z1

-- Define the function to determine in which quadrant a point lies
def quadrant (p : ℝ × ℝ) : String :=
  match p with
  | (x, y) =>
    if x > 0 ∧ y > 0 then "First quadrant"
    else if x < 0 ∧ y > 0 then "Second quadrant"
    else if x < 0 ∧ y < 0 then "Third quadrant"
    else if x > 0 ∧ y < 0 then "Fourth quadrant"
    else "On an axis"

-- Extract the real and imaginary parts of z as coordinates
def z_coord : ℝ × ℝ := (z.re, z.im)

-- The theorem statement to prove the point corresponding to z is in the second quadrant
theorem point_in_second_quadrant : quadrant z_coord = "Second quadrant" := by
  sorry

end point_in_second_quadrant_l419_419477


namespace book_pages_total_l419_419115

theorem book_pages_total
  (pages_read_first_day : ℚ) (total_pages : ℚ) (pages_read_second_day : ℚ)
  (rem_read_ratio : ℚ) (read_ratio_mult : ℚ)
  (book_ratio: ℚ) (read_pages_ratio: ℚ)
  (read_second_day_ratio: ℚ):
  pages_read_first_day = 1 / 6 →
  pages_read_second_day = 42 →
  rem_read_ratio = 3 →
  read_ratio_mult = (2 / 6) →
  book_ratio = 3 / 5 →
  read_pages_ratio = 2 / 5 →
  read_second_day_ratio = (2 / 5 - 1 / 6) →
  total_pages = pages_read_second_day / read_second_day_ratio  →
  total_pages = 126 :=
by sorry

end book_pages_total_l419_419115


namespace isosceles_triangle_has_perimeter_22_l419_419034

noncomputable def isosceles_triangle_perimeter (a b : ℕ) : ℕ :=
if a + a > b ∧ a + b > a ∧ b + b > a then a + a + b else 0

theorem isosceles_triangle_has_perimeter_22 :
  isosceles_triangle_perimeter 9 4 = 22 :=
by 
  -- Add a note for clarity; this will be completed via 'sorry'
  -- Prove that with side lengths 9 and 4 (with 9 being the equal sides),
  -- they form a valid triangle and its perimeter is 22
  sorry

end isosceles_triangle_has_perimeter_22_l419_419034


namespace find_a_n_l419_419191

variable (a : ℕ → ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a1_eq : a 1 = 1
axiom rec_relation : ∀ n, a n * (n * a n - a (n + 1)) = (n + 1) * (a (n + 1)) ^ 2

theorem find_a_n : ∀ n, a n = 1 / n := by
  sorry

end find_a_n_l419_419191


namespace tree_distance_eight_trees_l419_419444

theorem tree_distance_eight_trees (d : ℕ) (h₁ : eight_trees ≠ set.empty)
  (h₂ : ∃ t1 t2 t3 t4 t5 t6 t7 t8, 
      t1 ≠ t2 ∧ t2 ≠ t3 ∧ t3 ≠ t4 ∧ t4 ≠ t5 ∧ t5 ≠ t6 ∧ t6 ≠ t7 ∧ t7 ≠ t8 ∧ 
      distance(t1, t5) = 100 ∧ equal_spacing(t1, t2, t3, t4, t5, t6, t7, t8)) 
  : distance(first_tree : ℕ, last_tree : ℕ) = 175 :=
sorry

end tree_distance_eight_trees_l419_419444


namespace determinant_of_matrix_l419_419959

theorem determinant_of_matrix : ∀ (x : ℝ),
  Matrix.det !![
    [x + 2, x, x],
    [x, x + 2, x],
    [x, x, x + 2]
  ] = 8 * x + 8 :=
by
  intros x
  sorry

end determinant_of_matrix_l419_419959


namespace line_intersects_circle_at_two_points_l419_419497

-- Definitions based on given conditions
def radius (r : ℝ) : Prop := r = 6.5
def distance_from_center_to_line (d : ℝ) : Prop := d = 4.5

-- Theorem statement
theorem line_intersects_circle_at_two_points (r d : ℝ) (hr : radius r) (hd : distance_from_center_to_line d) : 
  d < r → ∃(p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ (∥p1∥ = r ∧ ∥p2∥ = r) := 
by 
  sorry

end line_intersects_circle_at_two_points_l419_419497


namespace not_all_roots_real_l419_419479

theorem not_all_roots_real (a b c d e : ℝ) (h : 2 * a^2 < 5 * b) :
    ¬ ∀ x : ℝ, (polynomial.eval x (polynomial.C e + polynomial.C d * polynomial.X + polynomial.C c * polynomial.X^2 + polynomial.C b * polynomial.X^3 + polynomial.C a * polynomial.X^4 + polynomial.X^5)) = 0 :=
by 
  sorry

end not_all_roots_real_l419_419479


namespace octagon_diagonals_l419_419589

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l419_419589


namespace cannot_satisfy_congruence_of_right_angled_triangles_l419_419922

-- Define the congruence conditions for right-angled triangles
def two_legs_congruent (Δ1 Δ2 : Triangle) : Prop := 
  Δ1.leg1 = Δ2.leg1 ∧ Δ1.leg2 = Δ2.leg2

def hypotenuse_one_leg_congruent (Δ1 Δ2 : Triangle) : Prop := 
  Δ1.hypotenuse = Δ2.hypotenuse ∧ (Δ1.leg1 = Δ2.leg1 ∨ Δ1.leg2 = Δ2.leg2)

def one_acute_angle_one_leg_congruent (Δ1 Δ2 : Triangle) : Prop :=
  (Δ1.acute_angle1 = Δ2.acute_angle1 ∨ Δ1.acute_angle2 = Δ2.acute_angle2) 
  ∧ (Δ1.leg1 = Δ2.leg1 ∨ Δ1.leg2 = Δ2.leg2)

def two_acute_angles_congruent (Δ1 Δ2 : Triangle) : Prop :=
  (Δ1.acute_angle1 = Δ2.acute_angle1 ∧ Δ1.acute_angle2 = Δ2.acute_angle2)
  ∨ (Δ1.acute_angle1 = Δ2.acute_angle2 ∧ Δ1.acute_angle2 = Δ2.acute_angle1)

-- Define the congruence of triangles
def congruent (Δ1 Δ2 : Triangle) : Prop :=
  Δ1.leg1 = Δ2.leg1 ∧ Δ1.leg2 = Δ2.leg2 ∧ Δ1.hypotenuse = Δ2.hypotenuse

-- Define the right-angled triangle
structure Triangle :=
  (leg1 leg2 hypotenuse : Real)
  (acute_angle1 acute_angle2 : Real)
  (right_angle : Real := π / 2)

-- Theorem statement
theorem cannot_satisfy_congruence_of_right_angled_triangles 
  (Δ1 Δ2 : Triangle) :
  two_acute_angles_congruent Δ1 Δ2 → ¬congruent Δ1 Δ2 :=
sorry

end cannot_satisfy_congruence_of_right_angled_triangles_l419_419922


namespace value_of_a_l419_419948

noncomputable def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ :=
  a * b^3 + c

theorem value_of_a :
  F a 2 3 = F a 3 4 → a = -1 / 19 :=
by
  sorry

end value_of_a_l419_419948


namespace probability_of_point_satisfying_x_lt_2y_l419_419063

open Set

noncomputable def area_of_triangle (a b c : Point ℝ) : ℝ :=
  abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2

noncomputable def probability_x_lt_2y : ℝ :=
  let rectangle := {p : Point ℝ | 0 <= p.x ∧ p.x <= 6 ∧ 0 <= p.y ∧ p.y <= 1}
  let triangle := {p : Point ℝ | p ∈ rectangle ∧ p.x < 2 * p.y}
  let area_rectangle := 6 * 1
  let area_triangle := area_of_triangle ⟨0, 0⟩ ⟨2, 1⟩ ⟨0, 1⟩
  area_triangle / area_rectangle

theorem probability_of_point_satisfying_x_lt_2y : probability_x_lt_2y = 1 / 6 :=
  sorry

end probability_of_point_satisfying_x_lt_2y_l419_419063


namespace induced_subgraph_isomorphic_l419_419980

variable {p : ℝ} (hp : 0 < p ∧ p < 1) (H : Type) [graph H]

theorem induced_subgraph_isomorphic (G : graph (fin n)) : 
  ∀ p ∈ (0, 1), ∀ (H : Type) [graph H], 
  (∀ n, ∃ G ∈ 𝓖(n, p), P[H ⊆ G]) :=
by
  sorry

end induced_subgraph_isomorphic_l419_419980


namespace quadratic_non_residue_219_mod_383_l419_419667

theorem quadratic_non_residue_219_mod_383 : legendreSymbol 219 383 = -1 := by
  -- Define the conditions
  have div_219 : 219 = 3 * 73 := by sorry
  have legendreSymbol_3_383 : legendreSymbol 3 383 = 1 := by sorry
  have legendreSymbol_73_383 : legendreSymbol 73 383 = -1 := by sorry
  -- Combine the results
  exact legendreSymbol.mul legendreSymbol_3_383 legendreSymbol_73_383

end quadratic_non_residue_219_mod_383_l419_419667


namespace circumference_to_diameter_ratio_l419_419729

theorem circumference_to_diameter_ratio (C D : ℝ) (hC : C = 314) (hD : D = 100) : C / D = 3.14 :=
by 
  rw [hC, hD]
  exact div_self hD

end circumference_to_diameter_ratio_l419_419729


namespace min_tetrahedrons_to_decompose_cube_l419_419346

theorem min_tetrahedrons_to_decompose_cube : 
  ∀ (cube : Type), (∃ (tetrahedrons : list cube), cube_to_tetrahedrons cube tetrahedrons ∧ length tetrahedrons = 5) ∧
  (∀ (tetrahedrons : list cube), cube_to_tetrahedrons cube tetrahedrons → length tetrahedrons ≥ 5) :=
begin
  sorry
end

end min_tetrahedrons_to_decompose_cube_l419_419346


namespace rowing_trip_time_l419_419057

def rowing_time_to_and_back (rowing_speed current_speed distance : ℝ) : ℝ :=
  let downstream_speed := rowing_speed + current_speed
  let upstream_speed := rowing_speed - current_speed
  let time_downstream := distance / downstream_speed
  let time_upstream := distance / upstream_speed
  time_downstream + time_upstream

theorem rowing_trip_time 
  (rowing_speed : ℝ := 5) 
  (current_speed : ℝ := 1)
  (distance : ℝ := 2.4)
  : rowing_time_to_and_back rowing_speed current_speed distance = 1 := 
by
  sorry

end rowing_trip_time_l419_419057


namespace volume_cube_equals_eight_l419_419291

noncomputable def original_volume_cube : ℕ :=
  let s : ℕ := 2 in
  s^3

theorem volume_cube_equals_eight :
  let s : ℕ := 2 in
  let Vcube := s^3 in
  let Vnew := (s+2)*(s-3)*s in
  Vcube = 8 → Vnew + 8 = Vcube :=
by
  sorry

end volume_cube_equals_eight_l419_419291


namespace sixth_number_is_correct_l419_419328

noncomputable def find_sixth_number
  (weighted_avg_11 : ℚ)
  (weighted_avg_first6 : ℚ)
  (weighted_avg_last6 : ℚ)
  (sum_weights_first6 : ℚ)
  (sum_weights_last6 : ℚ)
  (W : ℚ)
  (W_pos : W > 0) : ℚ :=
  let sum_first6 := weighted_avg_first6 * sum_weights_first6 in
  let sum_last6 := weighted_avg_last6 * sum_weights_last6 in
  let sum_all := weighted_avg_11 * (sum_weights_first6 + sum_weights_last6) in
  let sixth_weight := W in
  let weighted_sum_first6 := sum_first6 - (sum_first6 / sum_weights_first6) * sixth_weight in
  let weighted_sum_last6 := sum_last6 - (sum_last6 / sum_weights_last6) * sixth_weight in
  have eqn_main : (weighted_sum_first6 + weighted_sum_last6) = sum_all := by
  { sorry },
  (sum_all - (weighted_sum_first6 + weighted_sum_last6)) / (2 * sixth_weight)
  
theorem sixth_number_is_correct (weighted_avg_11 : ℚ) 
  (weighted_avg_first6 : ℚ) 
  (weighted_avg_last6 : ℚ) 
  (sum_weights_first6 : ℚ) 
  (sum_weights_last6 : ℚ) 
  (W : ℚ)
  (W_pos : W > 0) 
  (weighted_avg_11_def : weighted_avg_11 = 60)
  (weighted_avg_first6_def : weighted_avg_first6 = 58)
  (weighted_avg_last6_def : weighted_avg_last6 = 65)
  (sum_weights_first6_def : sum_weights_first6 = 2 * W)
  (sum_weights_last6_def : sum_weights_last6 = 3 * W)
  : find_sixth_number weighted_avg_11 weighted_avg_first6 weighted_avg_last6 sum_weights_first6 sum_weights_last6 W W_pos = 5.5 := 
by 
  sorry

end sixth_number_is_correct_l419_419328


namespace machines_make_2550_copies_l419_419877

def total_copies (rate1 rate2 : ℕ) (time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

theorem machines_make_2550_copies :
  total_copies 30 55 30 = 2550 :=
by
  unfold total_copies
  decide

end machines_make_2550_copies_l419_419877


namespace proper_subset_count_l419_419200

def S : set ℕ := {1, 2}

theorem proper_subset_count : finset.card (finset.powerset S).erase ∅ = 3 :=
by
  sorry

end proper_subset_count_l419_419200


namespace tangent_intersection_l419_419510

def f (x : ℝ) (a : ℝ) : ℝ := x - (Real.log x) / (a * x)

-- The derivative of the function
def f' (x : ℝ) (a : ℝ) : ℝ := 1 - (1 / a) * (1 - Real.log x) / (x * x)

-- The slopes of the tangent lines at the given points
def slope_l1 (a : ℝ) : ℝ := f' 1 a
def slope_l2 (a : ℝ) : ℝ := f' Real.exp a

-- The condition that the tangents are perpendicular
def perpendicular (a : ℝ) : Prop := slope_l1 a * slope_l2 a = -1

-- The intersection point of the tangent lines given a
def intersection (a : ℝ) : Prod ℝ ℝ :=
  let xa := 1 + 1 / Real.exp
  let ya := 1 - 1 / Real.exp
  (xa, ya)

-- Main theorem statement to prove
theorem tangent_intersection : ∀ a : ℝ,
  a ≠ 0 →
  perpendicular (1/2) →
  intersection (1/2) = (1 + 1 / Real.exp, 1 - 1 / Real.exp) :=
by
  intros a ha h_perp
  sorry

end tangent_intersection_l419_419510


namespace circle_area_isosceles_triangle_l419_419843

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end circle_area_isosceles_triangle_l419_419843


namespace books_distribution_l419_419220

theorem books_distribution (books : Finset ℕ) (students : Finset ℕ)
  (h_books : books.card = 5) (h_students : students.card = 3) :
  (∃ f : ℕ → ℕ, (∀ s ∈ students, 1 ≤ (books.filter (λ b, f b = s)).card) ∧ 
                (books.image f = students)) →
  ∃ n, n = 150 :=
by
  intro h
  use 150
  sorry

end books_distribution_l419_419220


namespace domain_of_g_l419_419941

noncomputable def g : ℝ → ℝ := sorry

theorem domain_of_g (x : ℝ) (hx : x ≠ 0) :
  g(x) + g(1 / x) = 3 * x ↔ (x = 1 ∨ x = -1) :=
sorry

end domain_of_g_l419_419941


namespace smallest_n_l419_419006

theorem smallest_n (n : ℕ) (h : n > 0): 
  ( ∃ n ≥ 1, (sqrt n: ℝ) - (sqrt (n - 1): ℝ) < 0.005) ∧ 
  (∀ m ≥ 1, (sqrt m: ℝ) - (sqrt (m - 1): ℝ) < 0.005 → m ≥ n) → 
  n = 10001 := 
by
  sorry

end smallest_n_l419_419006


namespace A_inter_B_domain_l419_419475

def A_domain : Set ℝ := {x : ℝ | x^2 + x - 2 >= 0}
def B_domain : Set ℝ := {x : ℝ | (2*x + 6)/(3 - x) >= 0 ∧ x ≠ -2}

theorem A_inter_B_domain :
  (A_domain ∩ B_domain) = {x : ℝ | (1 <= x ∧ x < 3) ∨ (-3 <= x ∧ x < -2)} :=
by
  sorry

end A_inter_B_domain_l419_419475


namespace face_value_of_shares_l419_419875

-- Define the problem conditions
variables (F : ℝ) (D R : ℝ)

-- Assume conditions
axiom h1 : D = 0.155 * F
axiom h2 : R = 0.25 * 31
axiom h3 : D = R

-- State the theorem
theorem face_value_of_shares : F = 50 :=
by 
  -- Here should be the proof which we are skipping
  sorry

end face_value_of_shares_l419_419875


namespace arithmetic_sequence_a12_l419_419656

variable (a : ℕ → ℤ) (n : ℕ)

def arithmetic_sequence := ∀ n, a (n + 1) - a n = d

theorem arithmetic_sequence_a12
  (h1 : a 3 + a 4 + a 5 = 3)
  (h2 : a 8 = 8)
  (arithmetic_seq : arithmetic_sequence a) :
  a 12 = 15 := by
  -- We can use the properties of arithmetic sequences here
  sorry

end arithmetic_sequence_a12_l419_419656


namespace total_cookies_and_brownies_l419_419635

-- Define the conditions
def bagsOfCookies : ℕ := 272
def cookiesPerBag : ℕ := 45
def bagsOfBrownies : ℕ := 158
def browniesPerBag : ℕ := 32

-- Define the total cookies, total brownies, and total items
def totalCookies := bagsOfCookies * cookiesPerBag
def totalBrownies := bagsOfBrownies * browniesPerBag
def totalItems := totalCookies + totalBrownies

-- State the theorem to prove
theorem total_cookies_and_brownies : totalItems = 17296 := by
  sorry

end total_cookies_and_brownies_l419_419635


namespace general_formula_find_m_l419_419158

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 0 else 4 ^ n

theorem general_formula (n : ℕ) (h : 1 ≤ n) : 3 * (Finset.sum (Finset.range n) sequence) = 4 * sequence n - sequence 1 :=
sorry

theorem find_m (m : ℕ) (hm : 1 ≤ m) :
  (Finset.sum (Finset.range (2 * m - m)) (λ k, (sequence (m + k))^.nat.sqrt)) = 56 :=
sorry

end general_formula_find_m_l419_419158


namespace imo_39th_preliminary_problem_l419_419084

theorem imo_39th_preliminary_problem
  (circle : Type)
  [metric_space circle]
  [normed_add_comm_group circle]
  [finite_dimensional ℝ circle]
  (A B P C D E F K : circle)
  (hAB: A ≠ B)
  (hAB_diameter : metric.segment A B = ⊙)
  (hP : collinear A B P)
  (hPC_tangent : is_tangent P C circle)
  (hD_sym : reflection A B C = D)
  (hCE_perp_AD : CE ⊥ AD)
  (hF_midpoint : midpoint C E = F)
  (hAF_intersect_K : intersects AF circle at K) :
  is_tangent AP (circumcircle (triangle P C K)) :=
sorry

end imo_39th_preliminary_problem_l419_419084


namespace train_length_l419_419366

-- Definitions based on conditions
def faster_train_speed := 46 -- speed in km/hr
def slower_train_speed := 36 -- speed in km/hr
def time_to_pass := 72 -- time in seconds
def relative_speed_kmph := faster_train_speed - slower_train_speed
def relative_speed_mps : ℚ := (relative_speed_kmph * 1000) / 3600

theorem train_length :
  ∃ L : ℚ, (2 * L = relative_speed_mps * time_to_pass / 1) ∧ L = 100 := 
by
  sorry

end train_length_l419_419366


namespace decreasing_interval_l419_419741

noncomputable def function := λ x : Real, log 2 (3 * x^2 - 7 * x + 2)

theorem decreasing_interval :
  ∀ x : Real, 
    (x < (1 / 3)) → 
    (∀ x1 x2 : Real, x1 < x2 → function x2 < function x1) :=
by
  intros x h
  sorry

end decreasing_interval_l419_419741


namespace max_body_diagonal_angle_l419_419171

noncomputable def rectangular_solid : Type := { a b c : ℝ // a ≤ b ∧ b ≤ c ∧ ab + ac + bc = 45/4 ∧ a + b + c = 6 }

noncomputable def body_diagonal_angle (s : rectangular_solid) : ℝ :=
  let ⟨a, b, c, _⟩ := s in
  let diagonal := real.sqrt (a^2 + b^2 + c^2) in
  real.arccos ((b + c) / diagonal)

theorem max_body_diagonal_angle :
  ∃ s : rectangular_solid, body_diagonal_angle s = real.arccos (real.sqrt 6 / 9) :=
sorry

end max_body_diagonal_angle_l419_419171


namespace triangle_centers_l419_419074

theorem triangle_centers (A B C : Type) [Angle : 58°] [Angle : 59°] [Angle : 63°] (Incenter : Type) 
(Circumcenter : Type) (P1 P2 : Type):
  ∃ (P1 = Incenter) (P2 = Circumcenter), true := sorry

end triangle_centers_l419_419074


namespace horner_eval_l419_419480

noncomputable def f (x : ℤ) : ℤ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_eval :
  let x := 3 in
  let v_0 := 1 in
  let v_1 := v_0 * x + 0 in
  let v_2 := v_1 * x + 2 in
  let v_3 := v_2 * x + 3 in
  v_3 = 36 :=
by
  sorry

end horner_eval_l419_419480


namespace blankets_collected_l419_419984

theorem blankets_collected (team_size : ℕ) (blankets_first_day_per_person : ℕ) (total_blankets : ℕ) (tripled_factor : ℕ) :
  team_size = 15 →
  blankets_first_day_per_person = 2 →
  total_blankets = 142 →
  tripled_factor = 3 →
  let blankets_first_day := team_size * blankets_first_day_per_person in
  let blankets_second_day := blankets_first_day * tripled_factor in
  let blankets_first_two_days := blankets_first_day + blankets_second_day in
  let blankets_last_day := total_blankets - blankets_first_two_days in
  blankets_last_day = 22 :=
by {
  intros h1 h2 h3 h4,
  simp [h1, h2, h3, h4],
  sorry
}

end blankets_collected_l419_419984


namespace octagon_has_20_diagonals_l419_419560

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l419_419560


namespace paula_painter_lunch_break_l419_419294

-- Define constants
variables (p h : ℝ) (L : ℝ)

-- Conditions based on the problem
theorem paula_painter_lunch_break :
  (9 - L) * (p + h) = 0.4 →     -- Monday's condition
  (7 - L) * h = 0.3 →           -- Tuesday's condition
  (10 - L) * p = 0.3 →          -- Wednesday's condition
  L * 60 = 36 :=                -- Prove: L in minutes is 36
begin
  sorry
end

end paula_painter_lunch_break_l419_419294


namespace six_digit_divisibility_by_37_l419_419811

theorem six_digit_divisibility_by_37 (a b c d e f : ℕ) (H : (100 * a + 10 * b + c + 100 * d + 10 * e + f) % 37 = 0) : 
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) % 37 = 0 := 
sorry

end six_digit_divisibility_by_37_l419_419811


namespace product_evaluation_l419_419960

theorem product_evaluation :
  (5 - 2) * (5 + 2) * (5 ^ 2 + 2 ^ 2) * (5 ^ 4 + 2 ^ 4) * (5 ^ 8 + 2 ^ 8) * (5 ^ 16 + 2 ^ 16) * (5 ^ 32 + 2 ^ 32) * (5 ^ 64 + 2 ^ 64) = 609 * 641 * 390881 * ... :=
begin
  sorry
end

end product_evaluation_l419_419960


namespace maximal_K4_free_constructible_l419_419052

-- Definition of a graph with vertices and edges
structure Graph :=
  (V : Type) -- Type of the vertices
  (E : V → V → Prop) -- Edges

-- Definition of complete graph K4
def K4 : Graph :=
{ V := fin 4, E := λ u v, true }

-- Definition of K4-free graph
def is_K4_free (G : Graph) : Prop :=
  ¬ ∃ (f : fin 4 → G.V), ∀ i j, i ≠ j → G.E (f i) (f j)

-- Definition of maximal K4-free graph
def is_maximal_K4_free (G : Graph) : Prop :=
  is_K4_free G ∧ ∀ v w, ¬ G.E v w → ¬ is_K4_free {G with E := λ u u', G.E u u' ∨ (u = v ∧ u' = w)}

-- Definition based on starting point of triangle and gluing along K2
def can_be_constructed_from_triangle (G : Graph) : Prop :=
  ∃ (steps : ℕ) (f : fin steps → Graph),
    f 0 = {V := fin 3, E := λ u v, true} ∧
    f steps = G ∧
    ∀ i, ∃ (u v : fin (fin.size (f i).V)), u ≠ v ∧ is_K4_free (f (i+1))

-- Statement of the problem in Lean
theorem maximal_K4_free_constructible :
  ∀ (G : Graph), (∀ v, ∃ w, v ≠ w) → is_maximal_K4_free G → can_be_constructed_from_triangle G :=
sorry

end maximal_K4_free_constructible_l419_419052


namespace isosceles_triangle_circumcircle_area_l419_419859

noncomputable def area_of_circumcircle (a b c : ℝ) : ℝ :=
  let BD := real.sqrt (a^2 - ((c / 2)^2))
  let OD := (2 / 3) * BD
  let r := real.sqrt (a^2 - OD^2)
  real.pi * r^2

theorem isosceles_triangle_circumcircle_area :
  area_of_circumcircle 4 4 3 = 9.8889 * real.pi :=
sorry

end isosceles_triangle_circumcircle_area_l419_419859


namespace find_t_minus_s_l419_419066

noncomputable def total_students : ℕ := 120
noncomputable def total_teachers : ℕ := 6
noncomputable def enrollments : List ℕ := [60, 30, 15, 10, 3, 2]

noncomputable def t : ℚ := (enrollments.sum : ℚ) / total_teachers
noncomputable def s : ℚ := (enrollments.map (λ n, (n * n : ℚ) / total_students)).sum

theorem find_t_minus_s : t - s = -20.316 :=
by
  have h_t : t = 20 := by sorry
  have h_s : s = 40.316 := by sorry
  rw [h_t, h_s]
  norm_num
  sorry

end find_t_minus_s_l419_419066


namespace determine_c_for_quadratic_eq_l419_419660

theorem determine_c_for_quadratic_eq (x1 x2 c : ℝ) 
  (h1 : x1 + x2 = 2)
  (h2 : x1 * x2 = c)
  (h3 : 7 * x2 - 4 * x1 = 47) : 
  c = -15 :=
sorry

end determine_c_for_quadratic_eq_l419_419660


namespace savings_equal_l419_419819

noncomputable def A_savings (A_salary : ℝ) : ℝ := A_salary - 0.95 * A_salary
noncomputable def B_savings (B_salary : ℝ) : ℝ := B_salary - 0.85 * B_salary
noncomputable def B_salary (total_salary A_salary : ℝ) : ℝ := total_salary - A_salary

theorem savings_equal (total_salary A_salary B_salary : ℝ)
  (h1 : total_salary = 4000)
  (h2 : A_salary = 3000)
  (h3 : B_salary = B_salary total_salary A_salary)
  : A_savings A_salary = B_savings B_salary :=
by
  sorry

end savings_equal_l419_419819


namespace octagon_has_20_diagonals_l419_419582

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l419_419582


namespace distance_between_disney_and_london_l419_419818

/-- Problem Statement
20 birds migrate on a seasonal basis from lake Jim to another, searching for food. They fly from lake Jim to lake Disney in one season,
which is 50 miles apart. The next season they fly from lake Disney to lake London, a certain distance apart. 
The combined distance all the birds traveled in the two seasons is 2200 miles.
Prove the distance between lake Disney and lake London is 60 miles
--/

theorem distance_between_disney_and_london
  (num_birds : ℕ)
  (distance_jim_disney : ℕ)
  (total_distance : ℕ) :
  num_birds = 20 →
  distance_jim_disney = 50 →
  total_distance = 2200 →
  let D := 60 in
  (num_birds * distance_jim_disney + num_birds * D = total_distance) →
  D = 60 :=
begin
  intros,
  sorry
end

end distance_between_disney_and_london_l419_419818


namespace range_of_f_l419_419321

noncomputable def f (x : ℝ) : ℝ := (1/2) * sin (2 * x) * tan x + 2 * sin x * tan (x / 2)

theorem range_of_f : set.range f = set.Ico 0 3 ∪ set.Ioo 3 4 :=
sorry

end range_of_f_l419_419321


namespace little_john_initial_amount_l419_419283

theorem little_john_initial_amount :
  ∀ (spent_on_sweets : ℝ) (given_to_first_friend : ℝ) (given_to_second_friend : ℝ) (remaining_amount : ℝ),
    spent_on_sweets = 1.05 → 
    given_to_first_friend = 1.00 → 
    given_to_second_friend = 1.00 → 
    remaining_amount = 4.05 → 
    spent_on_sweets + given_to_first_friend + given_to_second_friend + remaining_amount = 7.10 :=
by
  intros spent_on_sweets given_to_first_friend given_to_second_friend remaining_amount 
  intros hsweets hfirst hsecond hremain
  rw [hsweets, hfirst, hsecond, hremain]
  norm_num
  sorry

end little_john_initial_amount_l419_419283


namespace ratio_largest_element_l419_419433

theorem ratio_largest_element (S : Set ℕ) (largest : ℕ) (sum_others : ℕ) :
  S = {n | ∃ i : ℕ, i ≤ 20 ∧ n = 2^i} →
  largest = 2^20 →
  sum_others = ∑ i in Finset.range 20, 2^i →
  (largest : ℝ) / sum_others = 1 := by
  intros S_def largest_def sum_others_def
  sorry

end ratio_largest_element_l419_419433


namespace find_x_l419_419613

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end find_x_l419_419613


namespace octagon_diagonals_l419_419586

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l419_419586


namespace three_digit_divisible_by_11_l419_419714

theorem three_digit_divisible_by_11
  (x y z : ℕ) (h1 : y = x + z) : (100 * x + 10 * y + z) % 11 = 0 :=
by
  sorry

end three_digit_divisible_by_11_l419_419714


namespace number_of_diagonals_in_octagon_l419_419525

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l419_419525


namespace rowing_time_l419_419056

def Vm : ℝ := 5 -- Speed in still water (kmph)
def Vc : ℝ := 1 -- Speed of current (kmph)
def D : ℝ := 2.4 -- Distance to the place (km)

def Vup : ℝ := Vm - Vc -- Effective speed upstream
def Vdown : ℝ := Vm + Vc -- Effective speed downstream

def Tup : ℝ := D / Vup -- Time to row upstream
def Tdown : ℝ := D / Vdown -- Time to row downstream

def Ttotal : ℝ := Tup + Tdown -- Total time to row to the place and back

theorem rowing_time : Ttotal = 1 := by
  -- The details of the proof will be filled in here
  sorry

end rowing_time_l419_419056


namespace sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14_l419_419797

noncomputable def value_x : ℝ := real.sqrt (20 - 8 * real.sqrt 5) + real.sqrt (20 + 8 * real.sqrt 5)

theorem sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14 :
  value_x = 2 * real.sqrt 14 :=
by
  sorry

end sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14_l419_419797


namespace sqrt_sum_eq_l419_419802

theorem sqrt_sum_eq : 
  sqrt (20 - 8 * sqrt 5) + sqrt (20 + 8 * sqrt 5) = 2 * sqrt 10 := 
by 
  sorry

end sqrt_sum_eq_l419_419802


namespace mary_investment_l419_419706

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem mary_investment :
  ∃ (P : ℝ), P = 51346 ∧ compound_interest P 0.10 12 7 = 100000 :=
by
  sorry

end mary_investment_l419_419706


namespace container_marbles_volume_l419_419045

theorem container_marbles_volume {V₁ V₂ m₁ m₂ : ℕ} 
  (h₁ : V₁ = 24) (h₂ : m₁ = 75) (h₃ : V₂ = 72) :
  m₂ = 225 :=
by
  have proportion := (m₁ : ℚ) / V₁
  have proportion2 := (m₂ : ℚ) / V₂
  have h4 := proportion = proportion2
  sorry

end container_marbles_volume_l419_419045


namespace max_value_is_one_sixteenth_l419_419786

noncomputable def max_value_function : ℝ → ℝ := 
  λ t, ((3 ^ t - 4 * t) * t) / (9 ^ t)

theorem max_value_is_one_sixteenth : ∃ t : ℝ, max_value_function t = 1 / 16 :=
sorry

end max_value_is_one_sixteenth_l419_419786


namespace systematic_sampling_l419_419765

theorem systematic_sampling :
  ∃ (s : Finset ℕ), -- Existence of a set with natural numbers
  (∀ x ∈ s, 1 ≤ x ∧ x ≤ 50) ∧ -- Each selected bag is within the numbered range 1 to 50
  (s.card = 5) ∧ -- Exactly 5 bags are selected
  (∀ (i j : ℕ), i ∈ s → j ∈ s → i < j → j = i + 10) := -- Systematic sampling with interval of 10
begin
  use {7, 17, 27, 37, 47},
  split,
  { intros x hx, 
    fin_cases hx; repeat { constructor }; norm_num },
  split,
  { norm_num },
  { intros i j hi hj hij,
    fin_cases hi;
    fin_cases hj;
    norm_num }
end

end systematic_sampling_l419_419765


namespace pat_mark_ratio_l419_419293

theorem pat_mark_ratio :
  ∃ K P M : ℕ, P + K + M = 189 ∧ P = 2 * K ∧ M = K + 105 ∧ P / gcd P M = 1 ∧ M / gcd P M = 3 :=
by
  sorry

end pat_mark_ratio_l419_419293


namespace find_a_l419_419607

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 :=
by
  sorry

end find_a_l419_419607


namespace series_positive_l419_419278

variable (x : ℝ) (n : ℕ)
noncomputable def f (x : ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (sin ((2 * i + 1) * x) / (2 * i + 1))

theorem series_positive {x : ℝ} (hx : 0 < x ∧ x < π) (n : ℕ) : 0 < f x n := 
sorry

end series_positive_l419_419278


namespace f_equiv_zero_l419_419106

def f : ℕ → ℕ 

axiom f_mul (m n : ℕ) : f (m * n) = f m + f n
axiom f_2008_zero : f 2008 = 0
axiom f_mod (n : ℕ) (h : n % 2008 = 39) : f n = 0

theorem f_equiv_zero (n : ℕ) : f n = 0 :=
by
  sorry

end f_equiv_zero_l419_419106


namespace log_eq_system_solution_l419_419356

theorem log_eq_system_solution (x y : ℝ) 
  (h1 : log (x^2 + y^2) = 1 + log 8)
  (h2 : log (x + y) - log (x - y) = log 3)
  (h3 : x + y > 0)
  (h4 : x - y > 0) : 
  (x, y) = (8, 4) := 
  by 
    sorry

end log_eq_system_solution_l419_419356


namespace range_of_function_l419_419322

theorem range_of_function : 
  ∀ (x : ℝ), x ∈ set.Icc (-real.pi / 6) (real.pi / 6) → (1 + 2 * real.sin x) ∈ set.Icc 0 2 :=
by
  intros
  sorry

end range_of_function_l419_419322


namespace distinct_real_roots_f2004_l419_419258

noncomputable def f1 (x : ℝ) : ℝ := x^2 - 1

noncomputable def f : ℕ → (ℝ → ℝ)
| 0       := λ x, x  -- This holds for n = 0 for technical reasons and won't actually be used
| 1       := f1
| (n + 2) := λ x, f (n + 1) (f1 x)

theorem distinct_real_roots_f2004 :
  (f 2004) = λ x, 0 := 
sorry

end distinct_real_roots_f2004_l419_419258


namespace solve_for_x_l419_419620

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end solve_for_x_l419_419620


namespace calculate_expression_l419_419151

theorem calculate_expression (a b c : ℕ) (h1 : a = 2011) (h2 : b = 2012) (h3 : c = 2013) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end calculate_expression_l419_419151


namespace shopper_percentage_saved_l419_419069

def original_price (amount_spent amount_saved : ℝ) := amount_spent + amount_saved

def percentage_saved (amount_saved original_price : ℝ) := (amount_saved / original_price) * 100

theorem shopper_percentage_saved 
  (amount_saved : ℝ) (final_price : ℝ) 
  (h_amount_saved : amount_saved = 3.75)
  (h_final_price : final_price = 50) :
  percentage_saved amount_saved (original_price final_price amount_saved) = 7 :=
by
  sorry

end shopper_percentage_saved_l419_419069


namespace number_of_triangles_l419_419600

/-- 
  This statement defines and verifies the number of triangles 
  in the given geometric figure.
-/
theorem number_of_triangles (rectangle : Set ℝ) : 
  (exists lines : Set (List (ℝ × ℝ)), -- assuming a set of lines dividing the rectangle
    let small_right_triangles := 40
    let intermediate_isosceles_triangles := 8
    let intermediate_triangles := 10
    let larger_right_triangles := 20
    let largest_isosceles_triangles := 5
    small_right_triangles + intermediate_isosceles_triangles + intermediate_triangles + larger_right_triangles + largest_isosceles_triangles = 83) :=
sorry

end number_of_triangles_l419_419600


namespace max_occurring_values_l419_419413

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 1)^2 / (x * (x^2 - 1))

theorem max_occurring_values:
  {a : ℝ} (h : ∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 ∧ f x = a) → (|a| > 4) ↔ a ∈ (-∞, -4) ∪ (4, ∞) := 
by
  sorry

end max_occurring_values_l419_419413


namespace solve_for_a_l419_419603

-- Given conditions
variables (a b d : ℕ)
hypotheses
  (h1 : a + b = d)
  (h2 : b + d = 7)
  (h3 : d = 4)

-- Prove that a = 1
theorem solve_for_a : a = 1 :=
by {
  sorry
}

end solve_for_a_l419_419603


namespace part1_solution_part2_solution_part3_solution_part4_solution_l419_419155

-- Definition and conditions for part 1
def g (x : ℝ) (a b : ℝ) : ℝ := a * x^2 - 2 * a * x + 1 + b
axiom a_pos : ∀ a, a > 0

-- Part 1 question
theorem part1_solution (a b : ℝ) (h_max : ∀ x ∈ set.Icc 2 3, g x a b ≤ 4)
  (h_min : ∀ x ∈ set.Icc 2 3, g x a b ≥ 1) : a = 1 ∧ b = 0 := sorry

-- Part 2 question
def h (x : ℝ) (a : ℝ) : ℝ := a * x^2 - 2 * a * x + 2
theorem part2_solution (a : ℝ) (b_1 : b = 1) (cond : ∀ x ∈ set.Ico 1 2, h x a ≥ 0) : a ≤ 2 := sorry

-- Part 3 question
theorem part3_solution (a : ℝ) (b_1 : b = 1)
  (cond : ∀ x ∈ set.Icc 2 3, g x a 1 ≥ 0) : x ≤ 1 - real.sqrt 3 / 3 ∨ x ≥ 1 + real.sqrt 3 / 3 := sorry

-- Part 4 question
def f (x : ℝ) (a b : ℝ) : ℝ := g (|x|) a b
theorem part4_solution (k : ℝ) (a : ℝ) (b : ℝ)
  (cond : a = 1) (cond : b = 0) (h_gt : f (real.log 2 k) 1 0 > f 2 1 0) :
  k > 4 ∨ k < 1 / 4 := sorry

end part1_solution_part2_solution_part3_solution_part4_solution_l419_419155


namespace coefficient_of_x_l419_419967

theorem coefficient_of_x :
  ∀ (x : ℝ), 5 * (x - 6) + 6 * (8 - 3 * x^2 + 7 * x) - 9 * (4 * x - 3) = 11 * x + _ := by
  intro x
  sorry

end coefficient_of_x_l419_419967


namespace geometric_sequence_sum_l419_419472

theorem geometric_sequence_sum (S : ℕ → ℝ) (a : ℝ) :
    (∀ n, S n = a * 2^n + a - 2) →
    (∃ a, a = 1) :=
begin
  intro h,
  use 1,
  sorry
end

end geometric_sequence_sum_l419_419472


namespace second_derivative_at_pi_over_three_l419_419181

noncomputable def f : ℝ → ℝ := λ x, Real.cos x

theorem second_derivative_at_pi_over_three : (deriv^[2] f) (Real.pi / 3) = - (1 / 2) :=
by
  sorry

end second_derivative_at_pi_over_three_l419_419181


namespace no_absolute_winner_prob_l419_419907

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end no_absolute_winner_prob_l419_419907


namespace range_of_lambda_l419_419987

-- Definitions of the vectors and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (1, 1)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Acute angle condition
def acute_angle_condition (λ : ℝ) : Prop :=
  dot_product vector_a (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2) > 0

-- Non-collinearity condition
def non_collinearity_condition (λ : ℝ) : Prop :=
  λ ≠ 0

-- The final theorem
theorem range_of_lambda (λ : ℝ) :
  acute_angle_condition λ ∧ non_collinearity_condition λ ↔ - (5 / 3) < λ ∧ λ ≠ 0 :=
begin
  sorry
end

end range_of_lambda_l419_419987


namespace ReuleauxTriangleFitsAll_l419_419350

-- Assume definitions for fits into various slots

def FitsTriangular (s : Type) : Prop := sorry
def FitsSquare (s : Type) : Prop := sorry
def FitsCircular (s : Type) : Prop := sorry
def ReuleauxTriangle (s : Type) : Prop := sorry

theorem ReuleauxTriangleFitsAll (s : Type) (h : ReuleauxTriangle s) : 
  FitsTriangular s ∧ FitsSquare s ∧ FitsCircular s := 
  sorry

end ReuleauxTriangleFitsAll_l419_419350


namespace octagon_has_20_diagonals_l419_419559

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l419_419559


namespace z1z2_eq_1_3i_l419_419266

noncomputable def z1 (a : ℝ) := complex.mk a (-2)
noncomputable def z2 (a : ℝ) := complex.mk (-1) a

theorem z1z2_eq_1_3i (a : ℝ) (z1_is_pure_imaginary : z1 a + z2 a = complex.mk 0 ((a - 2) : ℝ)) :
  (z1 1) * (z2 1) = complex.mk 1 3 :=
by {
  sorry
}

end z1z2_eq_1_3i_l419_419266


namespace num_diagonals_octagon_l419_419570

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l419_419570


namespace no_absolute_winner_prob_l419_419901

open_locale probability

-- Define the probability of Alyosha winning against Borya
def P_A_wins_B : ℝ := 0.6

-- Define the probability of Borya winning against Vasya
def P_B_wins_V : ℝ := 0.4

-- There are no ties, and each player plays with each other once
-- Conditions ensure that all pairs have played exactly once

-- Define the event that there will be no absolute winner
def P_no_absolute_winner : ℝ := P_A_wins_B * P_B_wins_V * 1 + P_A_wins_B * (1 - P_B_wins_V) * (1 - 1)

-- Statement of the problem: Prove that the probability of event C is 0.24
theorem no_absolute_winner_prob :
  P_no_absolute_winner = 0.24 :=
  by
    -- Placeholder for proof
    sorry

end no_absolute_winner_prob_l419_419901


namespace number_of_men_in_first_group_l419_419035

theorem number_of_men_in_first_group : 
  (∃ M : ℕ, (M * 66) = (86 * 283.8 ∧ M = 369)) :=
by 
  sorry

end number_of_men_in_first_group_l419_419035


namespace pascal_hexagon_l419_419389

-- Define the elements involved in the problem
variables {A B C D E F : Type} 

-- Define the geometric configurations
def inscribed_in_circle (A B C D E F : Point) (S : Circle) : Prop := sorry

-- Define collinearity condition
def collinear (points : set Point) : Prop := sorry

-- State Pascal's theorem for the given problem

theorem pascal_hexagon (A B C D E F : Point) (S : Circle)
  (h_inscribed : inscribed_in_circle A B C D E F S) :
  collinear {point_of_intersection (line_through A B) (line_through D E),
             point_of_intersection (line_through B C) (line_through E F),
             point_of_intersection (line_through C D) (line_through F A)} :=
sorry


end pascal_hexagon_l419_419389


namespace area_between_chords_l419_419641

theorem area_between_chords (r d : ℝ) (h_r : r = 8) (h_d : d = 8) : 
  let area := 32 * Real.sqrt 3 + (64 / 3) * Real.pi in
  area = 32 * Real.sqrt 3 + 21.3333333333333 * Real.pi :=
by sorry

end area_between_chords_l419_419641


namespace composition_of_parallel_reflections_is_translation_translation_as_composition_of_reflections_l419_419013

-- Definitions for the conditions in part (a)
def Line (P : Type*) := P → Prop
def Reflection (P : Type*) (l : Line P) (A : P) : P := sorry
def Parallel (P : Type*) (l1 l2 : Line P) : Prop := sorry

-- Part (a)
theorem composition_of_parallel_reflections_is_translation
  (P : Type*) [MetricSpace P] (s s' : Line P) (h_parallel: Parallel P s s') 
  (A A1 A' : P) :
  Reflection P s A = A1 →
  Reflection P s' A1 = A' → 
  ∃ (M M': P), s M ∧ s' M' ∧
    let vec_a := 2 * dist M M' in  dist A A' = vec_a :=
sorry

-- Part (b)
theorem translation_as_composition_of_reflections
  (P : Type*) [MetricSpace P] (A A' : P) (s : Line P)
  (vec_a : ℝ) : 
  ∃ (s' : Line P), Parallel P s s' ∧ ∀ B, 
    let M := Reflection P s B,
        M' := Reflection P s' M in 
    B = A → M' = A' :=
sorry

end composition_of_parallel_reflections_is_translation_translation_as_composition_of_reflections_l419_419013


namespace bottom_right_corner_value_l419_419705

variable (a b c x : ℕ)

/--
Conditions:
- The sums of the numbers in each of the four 2x2 grids forming part of the 3x3 grid are equal.
- Known values for corners: a, b, and c.
Conclusion:
- The bottom right corner value x must be 0.
-/

theorem bottom_right_corner_value (S: ℕ) (A B C D E: ℕ) :
  S = a + A + B + C →
  S = A + b + C + D →
  S = B + C + c + E →
  S = C + D + E + x →
  x = 0 :=
by
  sorry

end bottom_right_corner_value_l419_419705


namespace roots_sum_fraction_eq_l419_419260

theorem roots_sum_fraction_eq :
  let x_roots := { x | x^84 + 7 * x - 6 = 0 }
  in (∑ x in x_roots, x / (x - 1)) = 77 / 2 :=
sorry

end roots_sum_fraction_eq_l419_419260


namespace circles_ordered_by_radius_l419_419434

noncomputable def radius_A : ℝ := 2
noncomputable def radius_B : ℝ := (6 * π) / (2 * π)
noncomputable def radius_C : ℝ := Real.sqrt (16 * π / π)

theorem circles_ordered_by_radius :
  [radius_A, radius_B, radius_C].sorted (≤) :=
by {
  have h_rB: radius_B = 3 := by sorry,
  have h_rC: radius_C = 4 := by sorry,
  exact List.sorted_insert radius_A (List.sorted_cons h_rB (List.sorted_single h_rC))
}

end circles_ordered_by_radius_l419_419434


namespace gray_area_is_correct_l419_419092

-- Define the conditions as per the given problem
def radius_smaller_circle (d_s : ℝ) : ℝ := d_s / 2
def radius_larger_circle (r_s : ℝ) : ℝ := 3 * r_s
def area_circle (r : ℝ) : ℝ := Real.pi * r^2

-- The proof problem translated to Lean 4 statement
theorem gray_area_is_correct (d_s : ℝ) (r_s r_l A_s A_l A_gray : ℝ) (h1: d_s = 6) 
                            (h2: r_s = radius_smaller_circle d_s) 
                            (h3: r_l = radius_larger_circle r_s) 
                            (h4: A_s = area_circle r_s) 
                            (h5: A_l = area_circle r_l) 
                            (h6: A_gray = A_l - A_s) 
                            : A_gray = 72 * Real.pi := 
by
  -- Proof is omitted
  sorry

end gray_area_is_correct_l419_419092


namespace minimum_disks_needed_l419_419674

theorem minimum_disks_needed :
  ∀ (n_files : ℕ) (disk_space : ℝ) (mb_files_1 : ℕ) (size_file_1 : ℝ) (mb_files_2 : ℕ) (size_file_2 : ℝ) (remaining_files : ℕ) (size_remaining_files : ℝ),
    n_files = 30 →
    disk_space = 1.5 →
    mb_files_1 = 4 →
    size_file_1 = 1.0 →
    mb_files_2 = 10 →
    size_file_2 = 0.6 →
    remaining_files = 16 →
    size_remaining_files = 0.5 →
    ∃ (min_disks : ℕ), min_disks = 13 :=
by
  sorry

end minimum_disks_needed_l419_419674


namespace circle_area_isosceles_triangle_l419_419839

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end circle_area_isosceles_triangle_l419_419839


namespace circle_chord_distance_l419_419640

theorem circle_chord_distance (R θ : ℝ) (hR : R = 1.4) (hθ : θ = 120) :
  ∃ d : ℝ, d = 0.7 ∧ ∀ (O A B M : point) (hO : O = center)
  (hM : midpoint A B = M) (hAMO : right_triangle O M A)
  (h∠AOB : ∠AOB = θ) (chord_subtends : A, B on_perpendicular_bisector_of_center ch):
    distance O M = d := by
  sorry

end circle_chord_distance_l419_419640


namespace teachers_students_count_l419_419114

theorem teachers_students_count :
  ∃ (x y : ℕ), (x + y = 31) ∧ (x + (15 + x) = 31) ∧ (x = 8) ∧ (y = 23) := 
by
  -- define the total number of participants
  have h1 : ∀ (x y: ℕ), x + y = 31 := sorry

  -- define the relationship for teachers
  have h2 : ∀ (x : ℕ), x + (15 + x) = 31 := sorry

  -- solving the equation for x
  have h3: ∀ (x : ℕ), x = 8 := sorry

  -- calculate the number of students based on x
  have h4: ∀ (y : ℕ), y = 31 - 8 := sorry

  -- combining all the information to show there exist x and y
  use 8
  use 31 - 8
  split
  · exact h1 8 (31 - 8)
  split
  · exact h2 8
  split
  · exact h3 8
  · exact h4 (31 - 8)

end teachers_students_count_l419_419114


namespace log_base_4_sqrt_inv_16_l419_419445

theorem log_base_4_sqrt_inv_16 : log 4 (sqrt (1 / 16)) = -1 := by
  sorry

end log_base_4_sqrt_inv_16_l419_419445


namespace no_absolute_winner_probability_l419_419919

-- Define the probabilities of matches
def P_AB : ℝ := 0.6  -- Probability Alyosha wins against Borya
def P_BV : ℝ := 0.4  -- Probability Borya wins against Vasya

-- Define the event C that there is no absolute winner
def event_C (P_AV : ℝ) (P_VB : ℝ) : ℝ :=
  let scenario1 := P_AB * P_BV * P_AV in
  let scenario2 := P_AB * P_VB * (1 - P_AV) in
  scenario1 + scenario2

-- Main theorem to prove
theorem no_absolute_winner_probability : 
  event_C 1 0.6 = 0.24 :=
by
  rw [event_C]
  simp
  norm_num
  sorry

end no_absolute_winner_probability_l419_419919


namespace haley_spent_32_dollars_l419_419199

theorem haley_spent_32_dollars 
  (ticket_price : ℕ) 
  (tickets_for_friends : ℕ) 
  (extra_tickets : ℕ) 
  (total_tickets : ℕ)
  (total_cost : ℕ) 
  (h1 : ticket_price = 4) 
  (h2 : tickets_for_friends = 3) 
  (h3 : extra_tickets = 5) 
  (h4 : total_tickets = tickets_for_friends + extra_tickets) 
  (h5 : total_cost = total_tickets * ticket_price): 
  total_cost = 32 :=
begin
  -- The proof would go here
  sorry
end

end haley_spent_32_dollars_l419_419199


namespace polyhedron_volume_correct_polyhedron_surface_area_correct_l419_419046

noncomputable def polyhedron_volume (a b c d : ℝ) : ℝ :=
  let a_b := a * b
  (a_b * d / 3) - (0.5 * (a / 2) * (b / 2) * (c / 3))

noncomputable def polyhedron_surface_area (a b c d : ℝ) : ℝ :=
  let term1 := (b / (16 * c)) * (2 * d - c) * sqrt (a^2 + 16 * c^2)
  let term2 := (a / (16 * c)) * (2 * d - c) * sqrt (b^2 + 16 * c^2)
  let term3 := (b / (8 * c)) * sqrt (a^2 * (d - 4 * c)^2 + 16 * c^2 * d^2)
  let term4 := (a / (8 * c)) * sqrt (b^2 * (d - 4 * c)^2 + 16 * c^2 * d^2)
  let term5 := 0.25 * c * sqrt (a^2 + b^2)
  let term6 := 7 * a * b / 8
  term1 + term2 + term3 + term4 + term5 + term6

theorem polyhedron_volume_correct :
  polyhedron_volume 20 30 15 25 = 4625 := by
  sorry

theorem polyhedron_surface_area_correct :
  polyhedron_surface_area 20 30 15 25 = 1851.6 := by
  sorry

end polyhedron_volume_correct_polyhedron_surface_area_correct_l419_419046


namespace ratio_triangle_trapezoid_l419_419662

-- Define necessary structures and properties
structure Trapezoid (P Q R S T : Type _) [Add R S T] :=
(base1_len : ℕ)
(base2_len : ℕ)
(extended_legs : ∃ P Q, True)

-- The main theorem statement to prove the ratio
theorem ratio_triangle_trapezoid (P Q R S T : Type _) [Trapezoid P Q R S T]
    (hPQ : P Q = 10) (hRS : R S = 23) (hExt : Trapezoid.extended_legs P Q R S T) :
    ∃ ratio : ℚ, ratio = 100 / 429 :=
begin
  use 100 / 429,
  sorry
end

end ratio_triangle_trapezoid_l419_419662


namespace sequence_v5_value_l419_419677

theorem sequence_v5_value (v : ℕ → ℝ) (h_rec : ∀ n, v (n + 2) = 3 * v (n + 1) - v n)
  (h_v3 : v 3 = 17) (h_v6 : v 6 = 524) : v 5 = 198.625 :=
sorry

end sequence_v5_value_l419_419677


namespace isosceles_triangle_circle_area_l419_419825

theorem isosceles_triangle_circle_area 
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : a = 4) 
  (h3 : c = 3) 
  (h4 : a = 4) 
  (h5 : b = 4)
  (h6 : c ≠ a)
  (h7 : c ≠ b) :
  let r := 4 in π * r ^ 2 = 16 * π :=
by
  sorry

end isosceles_triangle_circle_area_l419_419825


namespace trapezoid_area_calc_l419_419091

variable {a c m b : ℝ}

-- Define the conditions
def is_isosceles_trapezoid (a c m b : ℝ) : Prop :=
  (a > c) ∧ (m = b - c) ∧ (m = (a^2 - 2*a*c + 5*c^2) / (8 * c))

-- Define the area calculation
def trapezoid_area (a c m b : ℝ) : ℝ :=
  (a + c) / 2 * m

-- The main theorem
theorem trapezoid_area_calc (a : ℝ) (ha : a > 0) :
  is_isosceles_trapezoid a (a / 6) ((a / 2) + (a - (a / 6))^2 / (8 * (a / 6))) ((a / 2) + (a - (a / 6))^2 / (8 * (a / 6))) →
  trapezoid_area a (a / 6) ((a / 2) + (a - (a / 6))^2 / (8 * (a / 6))) ((a / 2) + (a - (a / 6))^2 / (8 * (a / 6))) = (49 * a^2) / 192 :=
by
  intros h
  sorry

end trapezoid_area_calc_l419_419091


namespace rate_of_mixed_oil_l419_419210

/--
If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 68 per litre, 
8 litres of a third oil at Rs. 42 per litre, and 7 litres of a fourth oil at Rs. 62 per litre, 
then the rate of the mixed oil per litre is Rs. 53.67.
-/
theorem rate_of_mixed_oil :
  let cost1 := 10 * 50
  let cost2 := 5 * 68
  let cost3 := 8 * 42
  let cost4 := 7 * 62
  let total_cost := cost1 + cost2 + cost3 + cost4
  let total_volume := 10 + 5 + 8 + 7
  let rate_per_litre := total_cost / total_volume
  rate_per_litre = 53.67 :=
by
  intros
  sorry

end rate_of_mixed_oil_l419_419210


namespace number_of_ordered_triplets_l419_419974

theorem number_of_ordered_triplets :
  ∃ (count : ℕ), count = (finset.range 601).sum (λ d, 
    finset.sum (finset.range (601 - 5 * d)).sum (λ p, 
      finset.sum (finset.range (601 - 5 * p - 3 * d)).sum (λ n, 
        (601 - 5 * p - 3 * d - n).succ))) :=
sorry

end number_of_ordered_triplets_l419_419974


namespace equation1_sol_equation2_sol_equation3_sol_l419_419307

theorem equation1_sol (x : ℝ) : 9 * x^2 - (x - 1)^2 = 0 ↔ (x = -0.5 ∨ x = 0.25) :=
sorry

theorem equation2_sol (x : ℝ) : (x * (x - 3) = 10) ↔ (x = 5 ∨ x = -2) :=
sorry

theorem equation3_sol (x : ℝ) : (x + 3)^2 = 2 * x + 5 ↔ (x = -2) :=
sorry

end equation1_sol_equation2_sol_equation3_sol_l419_419307


namespace isosceles_triangle_circle_area_l419_419828

theorem isosceles_triangle_circle_area 
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : a = 4) 
  (h3 : c = 3) 
  (h4 : a = 4) 
  (h5 : b = 4)
  (h6 : c ≠ a)
  (h7 : c ≠ b) :
  let r := 4 in π * r ^ 2 = 16 * π :=
by
  sorry

end isosceles_triangle_circle_area_l419_419828


namespace intersection_is_ge_negative_one_l419_419281

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4*x + 3}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

theorem intersection_is_ge_negative_one : M ∩ N = {y | y ≥ -1} := by
  sorry

end intersection_is_ge_negative_one_l419_419281


namespace average_first_16_even_numbers_divisible_by_five_l419_419780

def even_numbers_divisible_by_five (n : ℕ) : ℕ := 10 * (n + 1)

noncomputable def sum_even_numbers_divisible_by_five (m : ℕ) : ℕ := 
  (finset.range m).sum (λ n, even_numbers_divisible_by_five n)

theorem average_first_16_even_numbers_divisible_by_five :
  (sum_even_numbers_divisible_by_five 16 : ℝ) / 16 = 85 :=
by
  sorry

end average_first_16_even_numbers_divisible_by_five_l419_419780


namespace polynomial_remainder_l419_419137
-- Importing the broader library needed

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^5 + 2 * x^2 + 3

-- The statement of the theorem
theorem polynomial_remainder :
  p 2 = 43 :=
sorry

end polynomial_remainder_l419_419137


namespace perfect_square_solutions_l419_419130

theorem perfect_square_solutions :
  {n : ℕ | ∃ m : ℕ, n^2 + 77 * n = m^2} = {4, 99, 175, 1444} :=
by
  sorry

end perfect_square_solutions_l419_419130


namespace import_tax_excess_amount_l419_419393

theorem import_tax_excess_amount 
    (tax_rate : ℝ) 
    (tax_paid : ℝ) 
    (total_value : ℝ)
    (X : ℝ) 
    (h1 : tax_rate = 0.07)
    (h2 : tax_paid = 109.2)
    (h3 : total_value = 2560) 
    (eq1 : tax_rate * (total_value - X) = tax_paid) :
    X = 1000 := sorry

end import_tax_excess_amount_l419_419393


namespace sum_all_satisfy_l419_419269

-- Declare the assumptions
variables {f : ℝ → ℝ}

-- Conditions given in the problem
axiom cont_even (hf : continuous f) : ∀ x, f x = f (-x)
axiom mont_pos (hf : continuous f) : ∀ x y, 0 < x → x < y → f x ≤ f y

-- Main theorem statement
theorem sum_all_satisfy : (∑ x in {x | f x = f ((x + 1) / (2 * x + 4))}, x) = -4 := sorry

end sum_all_satisfy_l419_419269


namespace quadratic_min_value_proof_l419_419144

noncomputable def quadratic_min_value (a b c : ℝ) : Prop :=
  ∀ x : ℝ, a * x ^ 2 + b * x + c ≥ 0

theorem quadratic_min_value_proof (a b c : ℝ) (h_eqn : quadratic_min_value a b c)
  (h_ba : b > a) (h_a_pos : a > 0) : 
  ∀ x : ℝ, h_eqn x → (∀ x, 
    a * x ^ 2 + b * x + c ≥ 0) → (b > a) → (a > 0) → 
    (a + b + c) / (b - a) ≥ 3 :=
by
  intros x hx_eq h_quad_eq h_ba_eq h_a_pos_eq
  sorry

end quadratic_min_value_proof_l419_419144


namespace color_of_twenty_sixth_card_l419_419711

noncomputable def color_of_card (n : ℕ) : string := sorry

-- Define the given conditions
axiom constraint :
  ∀ n : ℕ, (n > 0) → color_of_card n ≠ color_of_card (n + 1)

axiom tenth_card_red : color_of_card 10 = "red"
axiom eleventh_card_red : color_of_card 11 = "red"
axiom twenty_fifth_card_black : color_of_card 25 = "black"

-- Prove the color of the twenty-sixth card
theorem color_of_twenty_sixth_card : color_of_card 26 = "red" := sorry

end color_of_twenty_sixth_card_l419_419711


namespace octagon_has_20_diagonals_l419_419583

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l419_419583


namespace sum_of_x_l419_419268

def f (x : ℝ) : ℝ := 20 * x - 5
noncomputable def f_inv (y : ℝ) : ℝ := (y + 5) / 20

theorem sum_of_x :
  let f (x : ℝ) := 20 * x - 5 in
  let f_inv (y : ℝ) := (y + 5) / 20 in
  (∑ x in { x : ℝ | f_inv x = f (1 / (3 * x)) }.to_finset) = -105 :=
by
  sorry

end sum_of_x_l419_419268


namespace infinite_k_exists_l419_419715

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℤ 
| 0 => 0
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Define the periodicity modulo 2017
noncomputable def periodic_sha (n : ℕ) : Prop :=
  ∃ (P : ℕ), ∀ (m : ℕ), fibonacci (m + P) % 2017 = fibonacci m % 2017

-- Define negative indices
def fib_neg_ind (n : ℤ) : ℤ :=
  if (n ≥ 0) then fibonacci (n.to_nat)
  else (-1)^(abs(n).to_nat + 1) * fibonacci (abs(n).to_nat)

-- Define the condition of the problem
def condition (k : ℤ) : Prop :=
  fib_neg_ind k % 2017 = -5 % 2017

theorem infinite_k_exists :
  (∃ (inf_sequence : ℕ → ℤ), ∀ n, condition (inf_sequence n)) :=
sorry

end infinite_k_exists_l419_419715


namespace largest_n_divides_30_factorial_l419_419971

theorem largest_n_divides_30_factorial : ∃ n : ℕ, (18^n ∣ nat.factorial 30) ∧ ∀ m : ℕ, (18^m ∣ nat.factorial 30) → m ≤ 7 :=
by
  have two_power_30_factorial : ∑ k in finset.range (nat.log 2 30 + 1), 30 / 2^k = 26 := sorry
  have three_power_30_factorial : ∑ k in finset.range (nat.log 3 30 + 1), 30 / 3^k = 14 := sorry
  use 7
  split
  ·
    --
    sorry
  ·
    --
    sorry

end largest_n_divides_30_factorial_l419_419971


namespace correct_propositions_about_binomial_expansion_l419_419718

theorem correct_propositions_about_binomial_expansion 
  (x : ℝ) : 
  (∑ k in range (24), (if k = 0 then 0 else binom 23 k * (-1)^k) = 1) ∧
  ((∑ k in range (24), (binom 23 k * x^(23 - k) * (-1)^k) % 24 = 23) → 
   (abs (binom 23 12) < binom 23 13)) :=
by
  sorry

end correct_propositions_about_binomial_expansion_l419_419718


namespace octagon_has_20_diagonals_l419_419566

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l419_419566


namespace cos_angle_GAC_in_cube_l419_419371

theorem cos_angle_GAC_in_cube (A B C D E F G H : Point) (s : ℝ) (h1 : s = 1)
  (h_cube : cube A B C D E F G H s) :
  cos (angle G A C) = (Real.sqrt 3) / 3 :=
sorry

end cos_angle_GAC_in_cube_l419_419371


namespace Alton_profit_l419_419896

variable (earnings_per_day : ℕ)
variable (days_per_week : ℕ)
variable (rent_per_week : ℕ)

theorem Alton_profit (h1 : earnings_per_day = 8) (h2 : days_per_week = 7) (h3 : rent_per_week = 20) :
  earnings_per_day * days_per_week - rent_per_week = 36 := 
by sorry

end Alton_profit_l419_419896


namespace find_CE_l419_419238

-- Definitions of triangles and conditions
def right_angle_triangle (A B C : Type) := ∃ (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ),
  angle_A = 90 ∧ angle_B + angle_C = 90

-- Definition of is45degrees
def is45degrees (angle : ℝ) : Prop := angle = 45.0

-- Conditions extracted
variable (A B C D E : Type)
variable (AE BE CE : ℝ)
variable [h1 : right_angle_triangle A B E]
variable [h2 : right_angle_triangle B C E]
variable [h3 : right_angle_triangle C D E]
variable [h4 : is45degrees 45.0] -- For 45° angles
variable (h_ae : AE = 40)

-- Main statement
theorem find_CE : CE = 20 * (Real.sqrt 2) := by
  sorry

end find_CE_l419_419238


namespace number_of_solutions_eq_l419_419690

open Nat

theorem number_of_solutions_eq (n : ℕ) : 
  ∃ N, (∀ (x : ℝ), 1 ≤ x ∧ x ≤ n → x^2 - ⌊x^2⌋ = (x - ⌊x⌋)^2) → N = n^2 - n + 1 :=
by sorry

end number_of_solutions_eq_l419_419690


namespace alton_weekly_profit_l419_419894

-- Definitions of the given conditions
def dailyEarnings : ℕ := 8
def daysInWeek : ℕ := 7
def weeklyRent : ℕ := 20

-- The proof problem: Prove that the total profit every week is $36
theorem alton_weekly_profit : (dailyEarnings * daysInWeek) - weeklyRent = 36 := by
  sorry

end alton_weekly_profit_l419_419894


namespace value_of_a_l419_419175

theorem value_of_a (a : ℝ) :
  let y := λ x : ℝ, x^3 + 2 * x + 1
  let tangent_slope := 3 * (1 : ℝ)^2 + 2
  let line_perpendicular := ∀ y x, ax - 2 * y - 3 = 0
  tangent_slope = 5 ∧ (5 * a / 2 = -1) → a = -2/5 :=
by
  intros
  sorry

end value_of_a_l419_419175


namespace area_of_circumscribed_circle_isosceles_triangle_l419_419853

theorem area_of_circumscribed_circle_isosceles_triangle :
  ∃ (r : ℝ), (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi) :=
by
  -- Consider the isosceles triangle conditions
  let a : ℝ := 4
  let b : ℝ := 4
  let c : ℝ := 3
  let BD := Real.sqrt(a^2 - (c/2)^2)
  let r := 8 / BD
  have h1 : BD = Real.sqrt 13.75 := by 
    -- Calculate the altitude BD
    calc
      BD = Real.sqrt(a^2 -  (c/2)^2) : rfl
      ... = Real.sqrt(16 - (3/2)^2) : rfl
      ... = Real.sqrt 13.75 : rfl
  
  use r
  have h2 : r = 8 / Real.sqrt 13.75 := by 
    -- Simplify the radius expression
    sorry

  have h3 : Real.pi * r ^ 2 = 256 / 55 * Real.pi := by 
    -- Calculate the area
    calc
      Real.pi * r ^ 2 = Real.pi * (8 / Real.sqrt 13.75) ^ 2 : by rw h2
      ... = Real.pi * (64 / 13.75) : by rw [pow_two, mul_div_assoc, mul_one, div_mul_div_same]
      ... = (256 / 54.6875) * Real.pi : by rw mul_comm
      ...   = (256 / 55) * Real.pi : by norm_num
    sorry
  
  show (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi),
  from ⟨h2, h3⟩
  sorry

end area_of_circumscribed_circle_isosceles_triangle_l419_419853


namespace find_f_of_3_l419_419170

-- Define assumptions
def power_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = √x

axiom passes_through : power_function f → f(2) = √2

-- The statement to prove
theorem find_f_of_3 (f : ℝ → ℝ) (h : power_function f) (hpass : passes_through f) : f 3 = √3 := 
  sorry

end find_f_of_3_l419_419170


namespace chef_bought_0_52_kg_l419_419036

def total_weight_of_nuts (a p : ℝ) : ℝ := a + p

theorem chef_bought_0_52_kg (a p : ℝ) (h_a : a = 0.14) (h_p : p = 0.38) :
  total_weight_of_nuts a p = 0.52 :=
by
  rw [total_weight_of_nuts, h_a, h_p]
  norm_num
  sorry

end chef_bought_0_52_kg_l419_419036


namespace regression_line_estimation_l419_419324

theorem regression_line_estimation :
  (λ x : ℝ, 0.5 * x - 0.81) 25 = 11.69 :=
by
  -- proof to be provided
  sorry

end regression_line_estimation_l419_419324


namespace mixed_oil_rate_l419_419211

def rate_per_litre_mixed_oil (v1 v2 v3 v4 p1 p2 p3 p4 : ℕ) :=
  (v1 * p1 + v2 * p2 + v3 * p3 + v4 * p4) / (v1 + v2 + v3 + v4)

theorem mixed_oil_rate :
  rate_per_litre_mixed_oil 10 5 8 7 50 68 42 62 = 53.67 :=
by
  sorry

end mixed_oil_rate_l419_419211


namespace dot_product_of_hyperbola_point_l419_419263

-- Definitions and given conditions
def hyperbola (x y : ℝ) := x^2 / 3 - y^2 = 1
def F1 := (-2 : ℝ, 0 : ℝ)
def F2 := (2 : ℝ, 0 : ℝ)

-- The area of triangle F1PF2 is 2
def triangle_area (P : ℝ × ℝ) :=
  let ⟨x, y⟩ := P in
  abs y * 4 / 2 = 2

-- The dot product condition to be proved
theorem dot_product_of_hyperbola_point (x y : ℝ) (hx : hyperbola x y) (harea : triangle_area (x, y)) :
  (x - 2) * (x - -2) + y^2 = 3 := 
sorry

end dot_product_of_hyperbola_point_l419_419263


namespace final_theorem_l419_419140

-- Definitions for the conditions in the problem.
def people : Finset String := { "A", "B", "C", "D", "E" }

-- Proof problem
noncomputable def problem_statement_A : Prop :=
  let adjacent_AB := { "A", "B" } ∪ people \ { "A", "B" }
  (adjacent_AB.card = 4) ∧ 
  (4.factorial = 24)

noncomputable def problem_statement_B : Prop :=
  (4.factorial = 24) ∧ (permit B.at_left ∨ permit A.at_left ∧ not (permit A.at_right)) 
 
-- Translation and correction of statement C
noncomputable def problem_statement_C : Prop :=
  (3.factorial * Nat.choose 4 2 = 36) -- matches the corrected number of ways for C

noncomputable def problem_statement_D : Prop :=
  (5.factorial / 3.factorial = 20) -- matches the number of ways for D

-- Combining the statements into a concluding proposition
theorem final_theorem : problem_statement_A ∧ problem_statement_C ∧ problem_statement_D :=
by {
  apply and.intro,
  {
    exact ⟨by simp [people], rfl⟩,
  },
  apply and.intro,
  {
    exact rfl,
  },
  {
    exact rfl,
  }
}

end final_theorem_l419_419140


namespace sum_area_bounds_l419_419203

theorem sum_area_bounds (n : ℕ) (h : ℝ) (x : Fin (2 * n + 2) → ℝ)
  (h0 : x 0 = 0) (h1 : x ( ⟨2 * n + 1, by linarith⟩) = 1)
  (h2 : ∀ i : Fin (2 * n + 1), x i < x (i + 1))
  (h3 : ∀ i : Fin (2 * n + 1), x (i + 1) - x i ≤ h) :
  (1 - h) / 2 < ∑ i in Finset.range n, x ⟨2 * i + 1, by linarith⟩ * (x ⟨2 * i + 2, by linarith⟩ - x ⟨2 * i, by linarith⟩)
  ∧ ∑ i in Finset.range n, x ⟨2 * i + 1, by linarith⟩ * (x ⟨2 * i + 2, by linarith⟩ - x ⟨2 * i, by linarith⟩) ≤ (1 + h) / 2 :=
begin
  sorry
end

end sum_area_bounds_l419_419203


namespace total_interest_received_l419_419880

def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem total_interest_received :
    let P_B := 5000
    let T_B := 2
    let P_C := 3000
    let T_C := 4
    let R := 12
    let SI_B := simple_interest P_B R T_B
    let SI_C := simple_interest P_C R T_C
    let Total_Interest := SI_B + SI_C
    Total_Interest = 2440 := by
  sorry

end total_interest_received_l419_419880


namespace pow_of_729_l419_419088

theorem pow_of_729 : (729 : ℝ) ^ (2 / 3) = 81 :=
by sorry

end pow_of_729_l419_419088


namespace hyperbola_asymptote_product_l419_419218

theorem hyperbola_asymptote_product (k1 k2 : ℝ) (h1 : k1 = 1) (h2 : k2 = -1) :
  k1 * k2 = -1 :=
by
  rw [h1, h2]
  norm_num

end hyperbola_asymptote_product_l419_419218


namespace parabola_circle_tangency_l419_419037

theorem parabola_circle_tangency (a b r : ℝ) 
  (h1 : ∀ x : ℝ, y = x^2) 
  (h2 : (0, b) ∈ circle) 
  (h3: circle = {p: ℝ × ℝ | (p.1)^2 + (p.2 - b)^2 = r}) 
  (h4: ∀ v :ℝ , pointOfTangency) 
  : b - a^2 = 1/2 :=
sorry

end parabola_circle_tangency_l419_419037


namespace holiday_customers_l419_419028

theorem holiday_customers (people_per_hour : ℕ) (hours : ℕ) (regular_rate : people_per_hour = 175)
  (holiday_rate : people_per_hour * 2 = 2 * 175) : 
  people_per_hour * 2 * hours = 2800 := 
by
  have h1 : people_per_hour = 175 := regular_rate
  have h2 : people_per_hour * 2 = 350 := by rw [h1, holiday_rate]
  have h3 : 350 * hours = 2800 := by
    norm_num at *
    exact Eq.refl 2800
  exact h3

end holiday_customers_l419_419028


namespace number_of_correct_expressions_is_2_l419_419923

-- Define the expressions as propositions
def expr1 : Prop := {0} ∈ ({0,1,2} : set nat)
def expr2 : Prop := (∅ : set nat) ⊆ {0}
def expr3 : Prop := {0,1,2} ⊆ ({1,2,0} : set nat)
def expr4 : Prop := 0 ∈ (∅ : set nat)
def expr5 : Prop := (0 : set nat) ∩ (∅ : set nat) = ∅

-- Define the main theorem that the number of correct expressions is 2
theorem number_of_correct_expressions_is_2 : 
  (if expr1 then 1 else 0) + 
  (if expr2 then 1 else 0) + 
  (if expr3 then 1 else 0) + 
  (if expr4 then 1 else 0) + 
  (if expr5 then 1 else 0) = 2 :=
sorry

end number_of_correct_expressions_is_2_l419_419923


namespace bijection_exists_l419_419924

structure LatticePath (n : ℕ) :=
(steps : List (ℕ × ℕ))
(start_at_origin : steps.head = (0, 0))
(up_steps : steps.filter (λ xy, xy.1 + 1 = xy.2).length = n)
(down_steps : steps.filter (λ xy, xy.1 - 1 = xy.2).length = n)

def is_downramp_of_even_length (path : LatticePath n) (m : ℕ) : Prop :=
∃ i, path.steps.nth i = (0, 0) ∧ (path.steps.nth (i + m + 1)) = (0, 0) ∧ m % 2 = 0

def S (n : ℕ) := { path : LatticePath (n - 1) // ¬ ∃ m, is_downramp_of_even_length path m }
def T (n : ℕ) := { path : LatticePath n // ¬ ∃ m, is_downramp_of_even_length path m }

theorem bijection_exists (n : ℕ) : ∃ f : S n → T n, function.bijective f :=
sorry

end bijection_exists_l419_419924


namespace Paul_cousin_points_l419_419010

variable (Paul_points : ℕ)
variable (total_points : ℕ)

theorem Paul_cousin_points : Paul_points = 3103 → total_points = 5816 → (total_points - Paul_points) = 2713 := by
  intros h1 h2
  rw [h1, h2]
  calc
    5816 - 3103 = 2713 : by norm_num

end Paul_cousin_points_l419_419010


namespace isosceles_triangle_circle_area_l419_419830

theorem isosceles_triangle_circle_area 
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : a = 4) 
  (h3 : c = 3) 
  (h4 : a = 4) 
  (h5 : b = 4)
  (h6 : c ≠ a)
  (h7 : c ≠ b) :
  let r := 4 in π * r ^ 2 = 16 * π :=
by
  sorry

end isosceles_triangle_circle_area_l419_419830


namespace binomial_coefficient_x3_l419_419968

theorem binomial_coefficient_x3 (x : ℝ) :
  coefficient (expansion (1 + 2*x) 5) 3 = 80 :=
sorry

end binomial_coefficient_x3_l419_419968


namespace total_time_correct_l419_419095

-- Definitions based on problem conditions
def first_time : ℕ := 15
def time_increment : ℕ := 7
def number_of_flights : ℕ := 7

-- Time taken for a specific flight
def time_for_nth_flight (n : ℕ) : ℕ := first_time + (n - 1) * time_increment

-- Sum of the times for the first seven flights
def total_time : ℕ := (number_of_flights * (first_time + time_for_nth_flight number_of_flights)) / 2

-- Statement to be proven
theorem total_time_correct : total_time = 252 := 
by
  sorry

end total_time_correct_l419_419095


namespace has_solution_set_range_a_l419_419189

noncomputable def solution_set (a x : ℝ) : Set ℝ := 
  { x | (ax - (a - 2)) * (x + 1) > 0 }

theorem has_solution_set (a : ℝ) (P : Set ℝ) :
  (∀ x, (a > 1 → (solution_set a x → (x < -1 ∨ x > (a - 2) / a))) ∧
        (a = 1 → (solution_set a x → (x ≠ -1))) ∧
        (0 < a ∧ a < 1 → (solution_set a x → (x < (a - 2) / a ∨ x > -1)))) ↔
  (P = {x | a > 1 ∧ (x < -1 ∨ x > (a - 2) / a) ∨ 
              a = 1 ∧ (x ≠ -1) ∨ 
              (0 < a ∧ a < 1) ∧ (x < (a - 2) / a ∨ x > -1)) } := 
sorry

theorem range_a (P : Set ℝ) : 
  {x : ℝ | -3 < x ∧ x < -1} ⊆ P → 
  (∀ a : ℝ, a ≥ 1) := 
sorry

end has_solution_set_range_a_l419_419189


namespace diagonals_of_60_sided_polygon_exterior_angle_of_60_sided_polygon_l419_419226

noncomputable def diagonals_in_regular_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

noncomputable def exterior_angle (n : ℕ) : ℝ :=
  360.0 / n

theorem diagonals_of_60_sided_polygon :
  diagonals_in_regular_polygon 60 = 1710 :=
by
  sorry

theorem exterior_angle_of_60_sided_polygon :
  exterior_angle 60 = 6.0 :=
by
  sorry

end diagonals_of_60_sided_polygon_exterior_angle_of_60_sided_polygon_l419_419226


namespace ratio_of_combined_areas_l419_419761

theorem ratio_of_combined_areas :
  let side_length_A := 36
  let side_length_B := 42
  let side_length_C := 48
  let area_A := side_length_A ^ 2
  let area_B := side_length_B ^ 2
  let area_C := side_length_C ^ 2
  let combined_area_AC := area_A + area_C
  (combined_area_AC : ℚ) / area_B = 20 / 7 :=
by {
  let side_length_A := 36
  let side_length_B := 42
  let side_length_C := 48
  let area_A := side_length_A ^ 2
  let area_B := side_length_B ^ 2
  let area_C := side_length_C ^ 2
  let combined_area_AC := area_A + area_C
  sorry
}

end ratio_of_combined_areas_l419_419761


namespace base7_to_base10_eq_l419_419341

theorem base7_to_base10_eq : 
  let a := 6 * 7^0 + 3 * 7^1 + 4 * 7^2 + 5 * 7^3 + 2 * 7^4
  in a = 6740 :=
by
  let a := 6 * 7^0 + 3 * 7^1 + 4 * 7^2 + 5 * 7^3 + 2 * 7^4
  have : a = 6740 := by sorry
  exact this

end base7_to_base10_eq_l419_419341


namespace angle_CKB_proof_l419_419083

noncomputable def angle_CKB 
  (A B C D M K : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AD_length : Real) (AB_length : Real) (angle_ADC : Real) 
  (is_midpoint : A → D → M → Prop) 
  (parallelogram_ABCD : A → B → C → D → Prop)
  (angle_bisector : B → C → D → Segment → K → Prop) 
  (BM_segment : B → M → Segment) : Real :=
  if parallelogram_ABCD A B C D ∧ 
     is_midpoint A D M ∧
     angle_ADC = 60 ∧
     AD_length = 2 ∧
     AB_length = sqrt 3 + 1 ∧
     (∃ K, angle_bisector B C D (BM_segment B M) K)
  then 75
  else 0

theorem angle_CKB_proof 
  (A B C D M K : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AD_length : Real) (AB_length : Real) (angle_ADC : Real) 
  (is_midpoint : A → D → M → Prop) 
  (parallelogram_ABCD : A → B → C → D → Prop)
  (angle_bisector : B → C → D → Segment → K → Prop) 
  (BM_segment : B → M → Segment) :
  parallelogram_ABCD A B C D ∧ 
  is_midpoint A D M ∧
  angle_ADC = 60 ∧
  AD_length = 2 ∧
  AB_length = sqrt 3 + 1 ∧
  (∃ K, angle_bisector B C D (BM_segment B M) K) →
  angle_CKB A B C D M K AD_length AB_length angle_ADC 
      is_midpoint parallelogram_ABCD angle_bisector BM_segment = 75 :=
by
  sorry

end angle_CKB_proof_l419_419083


namespace area_of_circumscribed_circle_isosceles_triangle_l419_419857

theorem area_of_circumscribed_circle_isosceles_triangle :
  ∃ (r : ℝ), (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi) :=
by
  -- Consider the isosceles triangle conditions
  let a : ℝ := 4
  let b : ℝ := 4
  let c : ℝ := 3
  let BD := Real.sqrt(a^2 - (c/2)^2)
  let r := 8 / BD
  have h1 : BD = Real.sqrt 13.75 := by 
    -- Calculate the altitude BD
    calc
      BD = Real.sqrt(a^2 -  (c/2)^2) : rfl
      ... = Real.sqrt(16 - (3/2)^2) : rfl
      ... = Real.sqrt 13.75 : rfl
  
  use r
  have h2 : r = 8 / Real.sqrt 13.75 := by 
    -- Simplify the radius expression
    sorry

  have h3 : Real.pi * r ^ 2 = 256 / 55 * Real.pi := by 
    -- Calculate the area
    calc
      Real.pi * r ^ 2 = Real.pi * (8 / Real.sqrt 13.75) ^ 2 : by rw h2
      ... = Real.pi * (64 / 13.75) : by rw [pow_two, mul_div_assoc, mul_one, div_mul_div_same]
      ... = (256 / 54.6875) * Real.pi : by rw mul_comm
      ...   = (256 / 55) * Real.pi : by norm_num
    sorry
  
  show (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi),
  from ⟨h2, h3⟩
  sorry

end area_of_circumscribed_circle_isosceles_triangle_l419_419857


namespace quadrilateral_square_l419_419686

variables {P A1 A2 A3 A4 B1 B2 B3 : Type}
variables {R1 R2 : ℝ}

-- Conditions
def ConcentricCircles (C1 C2 : Type) (P : Type) (R1 R2 : ℝ) : Prop :=
  ∀ c1 c2, (c1 = C1) ∧ (c2 = C2) ∧ center c1 = (P) ∧ center c2 = (P) ∧ radius c2 = 2 * radius c1

def QuadrilateralInscribedInCircle (C1 : Type) (A1 A2 A3 A4 : Type) : Prop :=
 ∀ q, q = quadrilateral (A1) (A2) (A3) (A4) ∧ inscribed q (C1)

def SideExtensionsIntersectCircle (A1 A2 A3 A4 P C2 : Type) (B1 B2 B3: Type) : Prop :=
 ∀ e1 e2 e3, 
 (extend (A4, A1) = e1) ∧ (extend (A1, A2) = e2) ∧ (extend (A2, A3) = e3) ∧
 intersection e1 C2 = B1 ∧ intersection e2 C2 = B2 ∧ intersection e3 C2 = B3

-- Problem statement
theorem quadrilateral_square
  (C1 C2 : Type)
  (P : Type)
  (R1 R2 : ℝ)
  (A1 A2 A3 A4 B1 B2 B3 : Type)
  (con_circs : ConcentricCircles C1 C2 P R1 R2)
  (quad_inscribed : QuadrilateralInscribedInCircle C1 A1 A2 A3 A4)
  (extensions : SideExtensionsIntersectCircle A1 A2 A3 A4 P C2 B1 B2 B3) :
  is_square (quadrilateral A1 A2 A3 A4) := sorry

end quadrilateral_square_l419_419686


namespace abs_fraction_lt_one_l419_419485

theorem abs_fraction_lt_one (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) : 
  |(x - y) / (1 - x * y)| < 1 := 
sorry

end abs_fraction_lt_one_l419_419485


namespace eval_complex_product_l419_419124

def z1 : ℂ := 3 * Real.sqrt 5 - 5 * Complex.i
def z2 : ℂ := 2 * Real.sqrt 2 + 4 * Complex.i

theorem eval_complex_product :
  abs (z1 * z2) = 8 * Real.sqrt 105 :=
by
  sorry

end eval_complex_product_l419_419124


namespace max_value_f_on_interval_l419_419315

open Real

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_f_on_interval :
  ∃ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), ∀ y ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f y ≤ f x ∧ f x = 23 := by
  sorry

end max_value_f_on_interval_l419_419315


namespace probability_A_second_day_probability_at_least_one_Western_l419_419823

namespace SchoolRestaurants

-- Given Conditions
def p_first_day_A : ℝ := 0.5
def p_A_to_A : ℝ := 0.4
def p_B_to_A : ℝ := 0.8
def p_first_day_B : ℝ := 0.5
def num_Western_desserts : ℕ := 4
def num_Chinese_desserts : ℕ := 6
def total_desserts : ℕ := num_Western_desserts + num_Chinese_desserts
def desserts_chosen : ℕ := 3

-- Part 1: Probability of going to restaurant A on the second day
theorem probability_A_second_day : 
  (p_first_day_A * p_A_to_A + p_first_day_B * p_B_to_A) = 0.6 :=
  sorry

-- Part 2: Probability of choosing at least one Western dessert
noncomputable def C (n k : ℕ) : ℝ := nat.choose n k

theorem probability_at_least_one_Western :
  (1 - (C num_Chinese_desserts desserts_chosen / C total_desserts desserts_chosen)) = 5 / 6 :=
  sorry

end SchoolRestaurants

end probability_A_second_day_probability_at_least_one_Western_l419_419823


namespace no_absolute_winner_prob_l419_419911

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end no_absolute_winner_prob_l419_419911


namespace diagonals_of_octagon_l419_419545

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l419_419545


namespace max_value_of_g_l419_419949

noncomputable def g : ℕ+ → ℕ
| ⟨n, h⟩ := if n < 15 then n + 12 else g ⟨n - 7, Nat.sub_pos_of_lt h⟩

theorem max_value_of_g : ∀ n : ℕ+, g n ≤ 26 :=
sorry

end max_value_of_g_l419_419949


namespace perimeter_large_star_l419_419737

theorem perimeter_large_star (n m : ℕ) (P : ℕ)
  (triangle_perimeter : ℕ) (quad_perimeter : ℕ) (small_star_perimeter : ℕ)
  (hn : n = 5) (hm : m = 5)
  (h_triangle_perimeter : triangle_perimeter = 7)
  (h_quad_perimeter : quad_perimeter = 18)
  (h_small_star_perimeter : small_star_perimeter = 3) :
  m * quad_perimeter + small_star_perimeter = n * triangle_perimeter + P → P = 58 :=
by 
  -- Placeholder proof
  sorry

end perimeter_large_star_l419_419737


namespace mps_to_kmph_conversion_l419_419360

/-- Define the conversion factor from meters per second to kilometers per hour. -/
def mps_to_kmph : ℝ := 3.6

/-- Define the speed in meters per second. -/
def speed_mps : ℝ := 5

/-- Define the converted speed in kilometers per hour. -/
def speed_kmph : ℝ := 18

/-- Statement asserting the conversion from meters per second to kilometers per hour. -/
theorem mps_to_kmph_conversion : speed_mps * mps_to_kmph = speed_kmph := by 
  sorry

end mps_to_kmph_conversion_l419_419360


namespace octagon_diagonals_l419_419587

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l419_419587


namespace shifted_function_is_odd_l419_419282

theorem shifted_function_is_odd (φ : ℝ) (hφ : |φ| < (Real.pi / 2)) :
  let f := λ x, Real.sin (2 * x + φ)
  in ∃ g, (∀ x, g x = Real.sin (2 * x + φ + 2 * Real.pi / 3)) ∧ (∀ x, g (-x) = -g x) → φ = Real.pi / 3 :=
by
  -- Definitions used in conditions.
  let f := λ x, Real.sin (2 * x + φ)
  exists.intro (λ x, Real.sin (2 * x + φ + 2 * Real.pi / 3))
  intros h1 h2
  -- Apply odd function property and solve.
  sorry

end shifted_function_is_odd_l419_419282


namespace sin_eq_sin_sin_solution_unique_l419_419952

open Real

theorem sin_eq_sin_sin_solution_unique :
  (∃! x ∈ set.Icc 0 (asin 0.99), sin x = sin (sin x)) :=
by
  have h1 : ∀ (θ : ℝ), 0 < θ ∧ θ < π / 2 → sin θ > θ :=
    sorry
  let S x := sin x - x
  have h2 : ∀ {x y}, 0 ≤ x → x < y → y < π / 2 → S x < S y :=
    sorry
  have h3 : 0 ≤ S(0) := sorry
  sorry

end sin_eq_sin_sin_solution_unique_l419_419952


namespace simplify_expression_l419_419304

variable (a b : ℝ)

theorem simplify_expression :
  3 * a * (3 * a^3 + 2 * a^2) - 2 * a^2 * (b^2 + 1) = 9 * a^4 + 6 * a^3 - 2 * a^2 * b^2 - 2 * a^2 :=
by
  sorry

end simplify_expression_l419_419304


namespace problem_a_problem_b_l419_419259
-- Import the entire math library to ensure all necessary functionality is included

-- Define the problem context
variables {x y z : ℝ}

-- State the conditions as definitions
def conditions (x y z : ℝ) : Prop :=
  (x ≤ y) ∧ (y ≤ z) ∧ (x + y + z = 12) ∧ (x^2 + y^2 + z^2 = 54)

-- State the formal proof problems
theorem problem_a (h : conditions x y z) : x ≤ 3 ∧ 5 ≤ z :=
sorry

theorem problem_b (h : conditions x y z) : 
  9 ≤ x * y ∧ x * y ≤ 25 ∧
  9 ≤ y * z ∧ y * z ≤ 25 ∧
  9 ≤ z * x ∧ z * x ≤ 25 :=
sorry

end problem_a_problem_b_l419_419259


namespace sqrt_sum_eq_l419_419801

theorem sqrt_sum_eq : 
  sqrt (20 - 8 * sqrt 5) + sqrt (20 + 8 * sqrt 5) = 2 * sqrt 10 := 
by 
  sorry

end sqrt_sum_eq_l419_419801


namespace cookies_per_person_l419_419808

-- Definitions based on conditions
def cookies_total : ℕ := 144
def people_count : ℕ := 6

-- The goal is to prove the number of cookies per person
theorem cookies_per_person : cookies_total / people_count = 24 :=
by
  sorry

end cookies_per_person_l419_419808


namespace total_handshakes_eq_900_l419_419930

def num_boys : ℕ := 25
def handshakes_per_pair : ℕ := 3

theorem total_handshakes_eq_900 : (num_boys * (num_boys - 1) / 2) * handshakes_per_pair = 900 := by
  sorry

end total_handshakes_eq_900_l419_419930


namespace find_fourth_vertex_of_regular_tetrahedron_l419_419064

def is_regular_tetrahedron (a b c d : ℝ × ℝ × ℝ) : Prop :=
  let dist (p q : ℝ × ℝ × ℝ) : ℝ := real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2)
  dist a b = dist b c ∧
  dist b c = dist c d ∧
  dist c d = dist d a ∧
  dist a c = dist b d ∧
  dist a d = dist b c

theorem find_fourth_vertex_of_regular_tetrahedron :
  ∃ x y z : ℤ, is_regular_tetrahedron (1, 0, 0) (0, 1, 0) (0, 0, 1) (x, y, z) ∧ (x, y, z) = (1, 1, 1) :=
by {
  sorry
}

end find_fourth_vertex_of_regular_tetrahedron_l419_419064


namespace no_absolute_winner_probability_l419_419920

-- Define the probabilities of matches
def P_AB : ℝ := 0.6  -- Probability Alyosha wins against Borya
def P_BV : ℝ := 0.4  -- Probability Borya wins against Vasya

-- Define the event C that there is no absolute winner
def event_C (P_AV : ℝ) (P_VB : ℝ) : ℝ :=
  let scenario1 := P_AB * P_BV * P_AV in
  let scenario2 := P_AB * P_VB * (1 - P_AV) in
  scenario1 + scenario2

-- Main theorem to prove
theorem no_absolute_winner_probability : 
  event_C 1 0.6 = 0.24 :=
by
  rw [event_C]
  simp
  norm_num
  sorry

end no_absolute_winner_probability_l419_419920


namespace monochromatic_solution_exists_l419_419022

theorem monochromatic_solution_exists :
  ∀ (c : ℕ → bool), 
  (∀ x y z, x + y = z → x ∈ {1, 2, 3, 4, 5} ∧ y ∈ {1, 2, 3, 4, 5} ∧ z ∈ {1, 2, 3, 4, 5} → c x ≠ c y ∨ c y ≠ c z ∨ c x ≠ c z) →
  False := by
  sorry

end monochromatic_solution_exists_l419_419022


namespace centers_of_circumscribed_circles_coincide_l419_419243

variables {A B C D E F X Y : Type} [Inhabited B] [LinearOrder B] [Field B]
variables {CF AD BE : B} 

-- Triangle ABC with medians CF, AD, BE and symmetric points X, Y
def is_median (A B C M : B) : Prop := 2*M = A + C ∧ 2*M = A + B 

def symmetric_points (F : B) (AD BE F : B) : X Y : B := 
  X = 2*AD - F ∧ Y = 2*BE - F

-- Main statement
theorem centers_of_circumscribed_circles_coincide :
  ∀ A B C D E F X Y : B,
    is_median A B C CF →
    is_median A B C AD →
    is_median B C A BE →
    symmetric_points F AD BE X Y → 
    circumcenter B E X = circumcenter A D Y :=
sorry

end centers_of_circumscribed_circles_coincide_l419_419243


namespace cost_of_fencing_is_289_l419_419134

def side_lengths : List ℕ := [10, 20, 15, 18, 12, 22]

def cost_per_meter : List ℚ := [3, 2, 4, 3.5, 2.5, 3]

def cost_of_side (length : ℕ) (rate : ℚ) : ℚ :=
  (length : ℚ) * rate

def total_cost : ℚ :=
  List.zipWith cost_of_side side_lengths cost_per_meter |>.sum

theorem cost_of_fencing_is_289 : total_cost = 289 := by
  sorry

end cost_of_fencing_is_289_l419_419134


namespace proof_polygon_7_sides_l419_419746

noncomputable def polygon_sides (A B C D : ℝ) :=
  ∃ n : ℕ, regular_polygon n ∧
  (1 / distance A B) = (1 / distance A C) + (1 / distance A D) ∧
  n = 7

theorem proof_polygon_7_sides (A B C D : ℝ) (h : polygon_sides A B C D) : 
  ∃ n : ℕ, regular_polygon n ∧ n = 7 :=
  sorry

end proof_polygon_7_sides_l419_419746


namespace isosceles_triangle_circumcircle_area_l419_419861

noncomputable def area_of_circumcircle (a b c : ℝ) : ℝ :=
  let BD := real.sqrt (a^2 - ((c / 2)^2))
  let OD := (2 / 3) * BD
  let r := real.sqrt (a^2 - OD^2)
  real.pi * r^2

theorem isosceles_triangle_circumcircle_area :
  area_of_circumcircle 4 4 3 = 9.8889 * real.pi :=
sorry

end isosceles_triangle_circumcircle_area_l419_419861


namespace purple_to_seafoam_valley_ratio_l419_419716

theorem purple_to_seafoam_valley_ratio (azure_skirts : ℕ) (purple_skirts : ℕ) :
  (azure_skirts = 60) → (purple_skirts = 10) → 
  (S = (2 / 3 : ℚ) * 60) → (S = 40) → 
  (purple_skirts / S = 1 / 4) :=
begin
  intros h_azure h_purple h_seafoam_valley1 h_seafoam_valley2,
  -- Proving the ratio
  sorry
end

end purple_to_seafoam_valley_ratio_l419_419716


namespace least_number_to_subtract_l419_419018

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (q : ℕ) :
  n = 13294 →
  d = 97 →
  n = d * q + r →
  r = 5 →
  ∃ k : ℕ, n - k = d * (q : ℕ) :=
begin
  intros h_n h_d h_div h_r,
  use r,
  rw [h_r, h_n, h_d, h_div],
  sorry
end

end least_number_to_subtract_l419_419018


namespace find_coeff_x4_in_expansion_l419_419180

noncomputable def f (x : ℝ) : ℝ := 10 * Real.sin x + (1/6) * x^3

def tangent_slope (x : ℝ) : ℝ := deriv f x

theorem find_coeff_x4_in_expansion :
  let n := 10 in
  ∃ c : ℝ, c = 135 ∧
    (∀ (x : ℝ), (1 + x + x^2) * (1 - x) ^ n = ∑ i in Finset.range(5), c * x ^ 4 + c * x ^ (i) + sorry) := 
sorry

end find_coeff_x4_in_expansion_l419_419180


namespace tan_ratio_l419_419691

-- Given conditions
variables {p q : ℝ} (h1 : Real.cos (p + q) = 1 / 3) (h2 : Real.cos (p - q) = 2 / 3)

-- The theorem we need to prove
theorem tan_ratio (h1 : Real.cos (p + q) = 1 / 3) (h2 : Real.cos (p - q) = 2 / 3) : 
  Real.tan p / Real.tan q = -1 / 3 :=
sorry

end tan_ratio_l419_419691


namespace cos_beta_l419_419207

variable {α β : ℝ}

-- Conditions
def acute_angle (x : ℝ) := 0 < x ∧ x < π / 2

axiom sin_alpha : sin α = (2 * real.sqrt 5) / 5
axiom cos_alpha_beta : cos (α + β) = -4 / 5
axiom acute_angles : acute_angle α ∧ acute_angle β

-- Theorem
theorem cos_beta :
  ∃ β, (acute_angle β) ∧ (cos β = (2 * real.sqrt 5) / 25) :=
sorry

end cos_beta_l419_419207


namespace center_of_symmetry_of_g_l419_419721

noncomputable def g (x : ℝ) : ℝ := 2 * cos (2 * x - (2 * Real.pi / 3)) - 1

theorem center_of_symmetry_of_g : 
  ∃ k : ℤ, (2 * (k * Real.pi / 2 + Real.pi / 12) - (2 * Real.pi / 3) = k * Real.pi + (Real.pi / 2)) ∧
             g (Real.pi / 12) = -1 :=
by
  sorry

end center_of_symmetry_of_g_l419_419721


namespace radius_circle_2016_l419_419638

-- Define the constants and parameters
variable (a : ℝ)

-- Define the recursive radius function
def radius : ℕ → ℝ
| 1     := 1 / (2 * a)
| (n+1) := radius n / 2

-- Prove the radius of the 2016th circle is (1 / (2 ^ 2016 * a))
theorem radius_circle_2016 : radius a 2016 = 1 / (2 ^ 2016 * a) :=
by
  sorry    -- Proof omitted

end radius_circle_2016_l419_419638


namespace sqrt_cube_sqrt_eq_fraction_l419_419206

noncomputable def sqrt_cube_sqrt (N : ℝ) (h : N > 1) : ℝ :=
  sqrt (N * sqrt (N * sqrt N))

theorem sqrt_cube_sqrt_eq_fraction (N : ℝ) (h : N > 1) : sqrt_cube_sqrt N h = N ^ (7 / 8) :=
by
  sorry

end sqrt_cube_sqrt_eq_fraction_l419_419206


namespace octagon_diagonals_20_l419_419557

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l419_419557


namespace equation_is_hyperbola_l419_419441

-- Define the condition for the given equation
def equation (x y : ℝ) : Prop := x^2 - 64*y^2 - 12*x + 16*y + 36 = 0

-- Define the predicate for hyperbola
def is_hyperbola (x y : ℝ) : Prop :=
  ∃ (h k : ℝ) (a b : ℝ) (h : a > 0 ∧ b > 0), 
  (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

-- Main theorem statement
theorem equation_is_hyperbola :
  ∀ (x y : ℝ), equation x y → is_hyperbola x y :=
sorry

end equation_is_hyperbola_l419_419441


namespace greatest_value_l419_419676

theorem greatest_value
  (m : ℕ) (a : ℝ) (h_a : 0 < a) (p : ℕ) (h_p1 : 1 ≤ p) (h_p2 : p ≤ m)
  (m_i : Fin p → ℕ)
  (h_sum : (∑ i, m_i i) = m) :
  ∃ m_i : Fin p → ℕ, 
    (∑ i, a ^ m_i i) = (p-1) * a + a ^ (m - (p-1)) ∧ 
    (∀ i, m_i i > 0) ∧ 
    (∑ i, m_i i = m) := 
sorry

end greatest_value_l419_419676


namespace quadrilateral_perimeter_l419_419652

-- Define points A, B, C, and D
variables (A B C D : Type)

-- Define distances and angles
variables (d_AB d_BC d_BD d_AD : ℝ)
variables (angle_B angle_AC_BD : ℝ)

-- Define specific values
def AB := 15
def BC := 20
def BD := 17
def π := real.pi

-- Define the constraints based on the problem
def conditions : Prop :=
  angle_B = π / 2 ∧
  angle_AC_BD = π / 2 ∧
  AB = 15 ∧
  BC = 20 ∧
  BD = 17

-- Define and calculate AC and AD using Pythagorean theorem
def AC : ℝ := real.sqrt (AB^2 + BC^2)
def AD : ℝ := real.sqrt (AB^2 + BD^2)

-- Calculate CD using given conditions
def perimeter (A B C D : Type) : ℝ := AB + BC + AC + AD

-- Problem statement to be proved
theorem quadrilateral_perimeter (A B C D : Type) :
  conditions → perimeter A B C D = 60 := 
begin
  sorry
end

end quadrilateral_perimeter_l419_419652


namespace octagon_diagonals_20_l419_419551

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l419_419551


namespace number_of_x_satisfying_condition_l419_419470

theorem number_of_x_satisfying_condition : 
  {x : Int | x^4 - 55 * x^2 + 54 < 0 }.card = 12 := 
by
  sorry

end number_of_x_satisfying_condition_l419_419470


namespace probability_no_absolute_winner_l419_419913

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end probability_no_absolute_winner_l419_419913


namespace sum_binom_mod_500_l419_419096

def complex_forth_root_unity (ω : ℂ) : Prop :=
  ω^4 = 1 ∧ ω^0 = 1 ∧ ω^1 = complex.I ∧ ω^2 = -1 ∧ ω^3 = -complex.I

theorem sum_binom_mod_500 (ω : ℂ) (h : complex_forth_root_unity ω) :
  (∑ i in Finset.range 1007, Nat.choose 2013 (4 * i) % 500) = 86 :=
  sorry

end sum_binom_mod_500_l419_419096


namespace sufficient_but_not_necessary_condition_for_subset_l419_419517

variable {A B : Set ℕ}
variable {a : ℕ}

theorem sufficient_but_not_necessary_condition_for_subset (hA : A = {1, a}) (hB : B = {1, 2, 3}) :
  (a = 3 → A ⊆ B) ∧ (A ⊆ B → (a = 3 ∨ a = 2)) ∧ ¬(A ⊆ B → a = 3) := by
sorry

end sufficient_but_not_necessary_condition_for_subset_l419_419517


namespace proof_problem_l419_419168

variables {α : Type*} [linear_ordered_field α] (f : α → α) (M N : α)

noncomputable theory

-- Conditions
def is_increasing_on (f : α → α) (a b : α) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_continuous_on (f : α → α) (a b : α) : Prop :=
  ∀ x ∈ set.Icc a b, continuous_at f x

variables (x : α)

theorem proof_problem :
  (is_increasing_on f 0 2) ∧ (is_continuous_on f 0 2) ∧ (f 0 = M) ∧ (f 2 = N) ∧ (0 < M) ∧ (0 < N) →
  (∃ x ∈ set.Icc (0 : α) 2, f x = (M + N) / 2) ∧
  (∃ x ∈ set.Icc (0 : α) 2, f x = real.sqrt (M * N)) ∧
  ¬(∃ x ∈ set.Icc (0 : α) 2, f x = 2 / (1 / M + 1 / N)) ∧
  ¬(∃ x ∈ set.Icc (0 : α) 2, f x = real.sqrt ((M ^ 2 + N ^ 2) / 2)) :=
begin
  sorry
end

end proof_problem_l419_419168


namespace find_norm_of_vector_expression_l419_419197

variables (a b : ℝ × ℝ) (λ : ℝ)

def vector_perp (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

noncomputable def norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

theorem find_norm_of_vector_expression
  (h1 : a = (2, 2)) 
  (h2 : b = (1, -1))
  (h3 : vector_perp (a.1 + λ * b.1, a.2 + λ * b.2) b) :
  norm (2 * a.1 - λ * b.1, 2 * a.2 - λ * b.2) = 4 * real.sqrt 2 :=
sorry

end find_norm_of_vector_expression_l419_419197


namespace perimeter_of_ABCD_l419_419651

noncomputable def length_AB := 15
noncomputable def length_BC := 20
noncomputable def length_CD := 9

theorem perimeter_of_ABCD :
  let AD := real.sqrt (length_AB^2 + length_BC^2 + length_CD^2)
  in perimeter_of_ABCD = 44 + AD :=
begin
  have h_AB := length_AB,
  have h_BC := length_BC,
  have h_CD := length_CD,
  let AD := real.sqrt (h_AB ^ 2 + h_BC ^ 2 + h_CD ^ 2),
  have h_AD := AD,
  let perimeter := h_AB + h_BC + h_CD + h_AD,
  show perimeter = 44 + real.sqrt (length_AB^2 + length_BC^2 + length_CD^2),
  sorry
end

end perimeter_of_ABCD_l419_419651


namespace percentage_P_is_40_percent_l419_419363

variables (P Q : ℝ)

def carbonated_water_equation : Prop :=
  0.80 * P + 0.55 * Q = 0.65 * (P + Q)

def percentage_P_in_mixture : ℝ :=
  P / (P + Q)

theorem percentage_P_is_40_percent
  (h : carbonated_water_equation P Q) :
  percentage_P_in_mixture P Q = 0.4 := 
sorry

end percentage_P_is_40_percent_l419_419363


namespace circle_area_isosceles_triangle_l419_419844

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end circle_area_isosceles_triangle_l419_419844


namespace find_angle_C_l419_419424

variables (A D C B O : Type)
variables (α β γ δ ε : Type) -- angle types

-- Conditions
def bisects_DO_ADC (DO: A → D → C → Prop) := ∀ {a d c : A}, DO a d c → angle ADC a d c = 2 * angle ODA d a
def bisects_BO_ABC (BO: A → B → C → Prop) := ∀ {a b c : A}, BO a b c → angle ABC a b c = 2 * angle OBC b a

-- Given Constants
axiom angle_A_35 (A : α) : angle A = 35 
axiom angle_O_42 (O : α) : angle O = 42 

-- Question: Find angle C
theorem find_angle_C (A D C B O : Type) (bisects_DO_ADC: A → D → C → Prop) (bisects_BO_ABC: A → B → C → Prop):
  angle C = 49 :=
sorry

end find_angle_C_l419_419424


namespace octagon_has_20_diagonals_l419_419562

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l419_419562


namespace probability_percussion_instruments_l419_419240

-- Define sounds set and classifications
def sounds_set := {"metal", "stone", "wood", "earth", "bamboo", "silk"}
def percussion_set := {"metal", "stone", "wood"}

-- Define a function to calculate binomial coefficient
def binom : ℕ → ℕ → ℕ
| n, 0     := 1
| 0, k     := 0
| n+1, k+1 := binom n k + binom n (k+1)

-- Define the Lean statement for the proof
theorem probability_percussion_instruments :
  let total_combinations := binom 6 2 in
  let favorable_outcomes := binom 3 2 in
  (favorable_outcomes : ℚ) / (total_combinations : ℚ) = 1 / 5 :=
by
  sorry

end probability_percussion_instruments_l419_419240


namespace proof_problem_l419_419082

noncomputable def a_seq (n : ℕ) : ℝ := sorry -- Define the sequence a(n)

def condition_1 (a : ℕ → ℝ) : Prop :=
  ∀ n, ∃ k, ∃ r, (k * (k - 1)) / 2 < n ∧ n ≤ (k * (k + 1)) / 2 ∧ 
                (∀ m, m > 1 → a (m * (m - 1) / 2 + 1) = 4 + 3 * (m - 2)) ∧ 
                a ((m * (m - 1)) / 2 + 1) = a ((m * (m - 1)) / 2 + 1 + (k - 1)) * (1 / 2) ^ (i - 1)

theorem proof_problem (a : ℕ → ℝ) 
      (h1 : condition_1 a) 
      (h2 : a 2 = 4) 
      (h3 : a 10 = 10) : 
    (a 1 = 1) ∧ 
    (∀ n, n ≥ 1 → a (n ^ 2) < a (n ^ 2 + 1)) ∧ 
    (∀ n, n ≥ 1 → a (n ^ 2) = (3 * n - 2) * (1 / 2) ^ (2 * n - 2)) := 
sorry

end proof_problem_l419_419082


namespace angle_ratio_value_l419_419234

variables {α : Type} [linear_ordered_field α]

-- Define the conditions of the problem
variables (ABCD : Type) [parallelogram ABCD]
variables (A B C D O : ABCD)
variables (theta : α)
variables (CAB DBC DBA ACB AOB : angle)

-- Conditions
-- 1. ABCD is a parallelogram
-- 2. O is the intersection of diagonals AC and BD
-- 3. angle CAB = angle DBC = 3 * angle DBA
axiom h1 : CAB = 3 * DBA
axiom h2 : DBC = 3 * DBA

-- Define angles needed for the solution
axiom h3 : ACB = 4 * DBA - DBA -- Simplified from steps derived

-- Define the ratio of angles
noncomputable def angle_ratio : α := ACB / AOB

-- The proof statement
theorem angle_ratio_value : angle_ratio = (5 / 8) := by
  sorry

end angle_ratio_value_l419_419234


namespace range_of_2a_plus_b_l419_419985

theorem range_of_2a_plus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 2) :
  0 < 2 * a + b ∧ 2 * a + b < 10 :=
sorry

end range_of_2a_plus_b_l419_419985


namespace game_ends_after_37_rounds_l419_419384

-- Type definition for players
inductive Player
| A
| B
| C

-- Function to get the initial tokens
def initialTokens : Player → ℕ
| Player.A => 15
| Player.B => 14
| Player.C => 13

-- Function to simulate one round
def round (tokens : Player → ℕ) (mostTokensPlayer : Player) : Player → ℕ
| Player.A =>
  if mostTokensPlayer = Player.A then tokens Player.A - 3 else tokens Player.A + 1
| Player.B =>
  if mostTokensPlayer = Player.B then tokens Player.B - 3 else tokens Player.B + 1
| Player.C =>
  if mostTokensPlayer = Player.C then tokens Player.C - 3 else tokens Player.C + 1

-- Function to find the player with the most tokens
def mostTokensPlayer (tokens : Player → ℕ) : Player :=
if tokens Player.A ≥ tokens Player.B ∧ tokens Player.A ≥ tokens Player.C then Player.A
else if tokens Player.B ≥ tokens Player.C then Player.B
else Player.C

-- Function to simulate multiple rounds
def simulateRounds (tokens : Player → ℕ) (rounds : ℕ) : Player → ℕ :=
match rounds with
| 0 => tokens
| n + 1 =>
  let mPlayer := mostTokensPlayer tokens
  simulateRounds (round tokens mPlayer) n

-- Proof that the game ends after 37 rounds
theorem game_ends_after_37_rounds :
  ∃ n, ∀ tokens : Player → ℕ,
  tokens = initialTokens →
  simulateRounds tokens 37 Player.A = 0 ∨
  simulateRounds tokens 37 Player.B = 0 ∨
  simulateRounds tokens 37 Player.C = 0 :=
by
  sorry

end game_ends_after_37_rounds_l419_419384


namespace max_occurring_values_l419_419411

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 1)^2 / (x * (x^2 - 1))

theorem max_occurring_values:
  {a : ℝ} (h : ∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 ∧ f x = a) → (|a| > 4) ↔ a ∈ (-∞, -4) ∪ (4, ∞) := 
by
  sorry

end max_occurring_values_l419_419411


namespace faster_train_length_l419_419001

noncomputable def length_of_faster_train (speed_faster_train_kmph : ℕ) 
                                         (speed_slower_train_kmph : ℕ) 
                                         (time_seconds : ℕ) : ℝ := 
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * (5.0 / 18.0)
  relative_speed_mps * time_seconds

theorem faster_train_length (speed_faster_train_kmph : ℕ)
                            (speed_slower_train_kmph : ℕ)
                            (time_seconds : ℕ)
                            (h_faster_speed : speed_faster_train_kmph = 120)
                            (h_slower_speed : speed_slower_train_kmph = 80)
                            (h_time : time_seconds = 30) :
  length_of_faster_train speed_faster_train_kmph speed_slower_train_kmph time_seconds = 333.3 :=
by
  sorry

end faster_train_length_l419_419001


namespace number_of_diagonals_in_octagon_l419_419529

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l419_419529


namespace period_of_y_l419_419004

-- Define the function y as given
def y (x : ℝ) : ℝ := (Real.tan x + Real.cot x)^2

-- The period assertion for the function y
theorem period_of_y : ∃ T > 0, ∀ x, y (x + T) = y x := 
  exists.intro π (by
    have h1 : π > 0 := Real.pi_pos,
    exact h1,
    intro x,
    sorry)

end period_of_y_l419_419004


namespace area_of_table_l419_419883

-- Definitions of the given conditions
def free_side_conditions (L W : ℝ) : Prop :=
  (L = 2 * W) ∧ (2 * W + L = 32)

-- Statement to prove the area of the rectangular table
theorem area_of_table {L W : ℝ} (h : free_side_conditions L W) : L * W = 128 := by
  sorry

end area_of_table_l419_419883


namespace num_diagonals_octagon_l419_419576

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l419_419576


namespace random_event_is_C_l419_419803

noncomputable def is_random_event (event: Prop) : Prop := ∃ outcome, outcome ∈ event ∧ outcome ≠ outcome

noncomputable def option_A := false
noncomputable def option_B := ∀ (T: Triangle), sum_interior_angles T = 180
noncomputable def option_C := ∀ (L1 L2 L3 : Line), (parallel L1 L2 ∧ transversal L3 L1 L2) → ∃ (A1 A2: Angle), corresponding_angles A1 A2 ∧ ∠A1 ≠ ∠A2
noncomputable def option_D := ∀ (Bag: Bag), count_black_balls Bag = 6 ∧ count_white_balls Bag = 2 → (Bag.ball_colors = {black, white})

theorem random_event_is_C : is_random_event option_C := sorry

end random_event_is_C_l419_419803


namespace sugar_to_cream_cheese_ratio_l419_419008

-- Add the given conditions from the problem
def vanilla_to_cream_cheese_ratio : ℝ := 1 / 2  -- 1 teaspoon of vanilla per 2 cups of cream cheese
def eggs_to_vanilla_ratio : ℝ := 2             -- 2 eggs per 1 teaspoon of vanilla
def cups_of_sugar : ℝ := 2
def number_of_eggs : ℝ := 8

-- Define the theorem to prove the ratio of sugar to cream cheese
theorem sugar_to_cream_cheese_ratio : 
  (number_of_eggs / eggs_to_vanilla_ratio) * 2 = 8 →
  cups_of_sugar / 8 = 1 / 4 :=
by 
  intros h,
  sorry

end sugar_to_cream_cheese_ratio_l419_419008


namespace diameter_perpendicular_to_chord_l419_419298

-- Define a structure for Circle and relevant points
structure Circle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] :=
(center : α)
(radius : ℝ)

variables {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]
variables (C : Circle α) (A B M O : α)

-- Define that A and B lie on the circle that is centered at O
def is_on_circle (p : α) (C : Circle α) : Prop :=
dist p C.center = C.radius

-- Define midpoint
def midpoint (A B : α) : α := (A + B) / 2

-- Main theorem statement
theorem diameter_perpendicular_to_chord (hA : is_on_circle A C) (hB : is_on_circle B C)
  (hM : M = midpoint A B) (hO : O = C.center) :
  A ≠ B →
  (∃ D : α, D = C.center + C.center - O ∧ ∀ A B : α, M = midpoint A B → ⟪O - M, A - B⟫ = 0) :=
sorry

end diameter_perpendicular_to_chord_l419_419298


namespace circumcircle_area_of_isosceles_triangle_l419_419869

open Real

theorem circumcircle_area_of_isosceles_triangle:
  (∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
    (dist A B = 4) ∧ (dist A C = 4) ∧ (dist B C = 3) →
    (circle_area A B C = (256 / 13.75) * π)) :=
by sorry

end circumcircle_area_of_isosceles_triangle_l419_419869


namespace initial_population_l419_419245

theorem initial_population (a b c : ℕ) 
  (h1 : a * a + 200 = b * b + 1)
  (h2 : b * b + 301 = c * c)
  (h3 : (exists k, a = 5 * k) ∨ (exists k, a = 8 * k) ∨ (exists k, a = 10 * k) ∨ (exists k, a = 14 * k) ∨ (exists k, a = 21 * k)) :
  a = 99 :=
begin
  sorry
end

end initial_population_l419_419245


namespace expression_value_is_correct_l419_419093

def calculate_expression : ℝ := -|(-5: ℝ)| + (-3: ℝ)^3 / (-2: ℝ)^2

theorem expression_value_is_correct : calculate_expression = 1.75 :=
by
  sorry

end expression_value_is_correct_l419_419093


namespace gillians_usual_monthly_bill_l419_419148

noncomputable def usual_monthly_bill (x : ℝ) : Prop :=
  let increased_monthly_bill := 1.10 * x
  let yearly_bill := 12 * increased_monthly_bill
  yearly_bill = 660

theorem gillians_usual_monthly_bill : ∃ x : ℝ, usual_monthly_bill x ∧ x = 50 :=
by
  use 50
  split
  focus
    sorry
  rfl

end gillians_usual_monthly_bill_l419_419148


namespace value_of_f_csc_squared_l419_419141

noncomputable def f (x : ℝ) : ℝ := if x ≠ 0 ∧ x ≠ 1 then 1 / x else 0

lemma csc_sq_identity (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  (f (x / (x - 1)) = 1 / x) := 
  by sorry

theorem value_of_f_csc_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π / 2) :
  f ((1 / (Real.sin t) ^ 2)) = - (Real.cos t) ^ 2 :=
  by sorry

end value_of_f_csc_squared_l419_419141


namespace downstream_distance_l419_419409

variables (v c d : ℝ)
variable (h_speed_current : c = 2.5)
variable (h_time_upstream : 36 / (v - c) = 9)
variable (h_time_downstream : d / (v + c) = 9)

theorem downstream_distance :
  d = 81 :=
by
  have h_v : v = 6.5 :=
    calc
      v = 36 / 9 + c : by sorry
      ... = 4 + 2.5 : by simp [h_speed_current]
      ... = 6.5 : by norm_num
  have h_d : d = 9 * (v + c) :=
    calc
      d = 9 * (v + c) : by field_simp [h_time_downstream]
  rw [h_v, h_speed_current] at h_d
  exact h_d

end downstream_distance_l419_419409


namespace magnitude_of_product_l419_419120

-- Definitions of the complex numbers involved
def z1 : ℂ := 3 * real.sqrt 5 - 5 * complex.I
def z2 : ℂ := 2 * real.sqrt 2 + 4 * complex.I

-- The statement to be proven
theorem magnitude_of_product :
  complex.abs (z1 * z2) = real.sqrt 1680 :=
by
  sorry

end magnitude_of_product_l419_419120


namespace consecutive_numbers_product_l419_419359

theorem consecutive_numbers_product (a b c d : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c + 1 = d) (h4 : a + d = 109) :
  b * c = 2970 :=
by {
  -- Proof goes here
  sorry
}

end consecutive_numbers_product_l419_419359


namespace ratio_of_books_l419_419670

theorem ratio_of_books (longest_pages : ℕ) (middle_pages : ℕ) (shortest_pages : ℕ) :
  longest_pages = 396 ∧ middle_pages = 297 ∧ shortest_pages = longest_pages / 4 →
  (middle_pages / shortest_pages = 3) :=
by
  intros h
  obtain ⟨h_longest, h_middle, h_shortest⟩ := h
  sorry

end ratio_of_books_l419_419670


namespace complement_U_A_eq_l419_419194
noncomputable def U := {x : ℝ | x ≥ -2}
noncomputable def A := {x : ℝ | x > -1}
noncomputable def comp_U_A := {x ∈ U | x ∉ A}

theorem complement_U_A_eq : comp_U_A = {x : ℝ | -2 ≤ x ∧ x < -1} :=
by sorry

end complement_U_A_eq_l419_419194


namespace find_r_l419_419495

theorem find_r (r : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = 4) → 
  (∀ (x y : ℝ), (x + 4)^2 + (y - 3)^2 = r^2) →
  (∀ x1 y1 x2 y2: ℝ, 
    (x2 - x1)^2 + (y2 - y1)^2 = 25) →
  (2 + |r| = 5) →
  (r = 3 ∨ r = -3) :=
by
  sorry

end find_r_l419_419495


namespace no_absolute_winner_prob_l419_419909

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end no_absolute_winner_prob_l419_419909


namespace lcm_of_nt_and_16_l419_419337

open Int

def n : ℤ := 24
def m : ℤ := 16
def gcf_n_m : ℤ := 8

theorem lcm_of_nt_and_16 :
  (gcd n m = gcf_n_m) →
  (Nat.lcm n.nat_abs m.nat_abs = 48) :=
by
  intro h
  sorry

end lcm_of_nt_and_16_l419_419337


namespace solve_for_x_l419_419618

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end solve_for_x_l419_419618


namespace nathan_final_temperature_l419_419289

theorem nathan_final_temperature : ∃ (final_temp : ℝ), final_temp = 77.4 :=
  let initial_temp : ℝ := 50
  let type_a_increase : ℝ := 2
  let type_b_increase : ℝ := 3.5
  let type_c_increase : ℝ := 4.8
  let type_d_increase : ℝ := 7.2
  let type_a_quantity : ℚ := 6
  let type_b_quantity : ℚ := 5
  let type_c_quantity : ℚ := 9
  let type_d_quantity : ℚ := 3
  let temp_after_a := initial_temp + 3 * type_a_increase
  let temp_after_b := temp_after_a + 2 * type_b_increase
  let temp_after_c := temp_after_b + 3 * type_c_increase
  let final_temp := temp_after_c
  ⟨final_temp, sorry⟩

end nathan_final_temperature_l419_419289


namespace probability_no_absolute_winner_l419_419912

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end probability_no_absolute_winner_l419_419912


namespace four_digit_numbers_with_digit_sum_3_l419_419595

def sum_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  d1 + d2 + d3 + d4

theorem four_digit_numbers_with_digit_sum_3 : 
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ sum_of_digits n = 3}.card = 10 :=
by
  sorry

end four_digit_numbers_with_digit_sum_3_l419_419595


namespace base_2_to_base_4_conversion_l419_419781

theorem base_2_to_base_4_conversion : ∀ (b₂ : Nat), b₂ = 0b11011000 → (Nat.digits 4 b₂) = [0, 2, 1, 3] :=
by
  intros b₂ h
  have : b₂ = 216 := by
    calc b₂ = 0b11011000 : h
        ... = 1 * 2^7 + 1 * 2^6 + 0 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 0 * 2^0 : by norm_num
        ... = 128 + 64 + 8 + 4 + 2 : by norm_num
        ... = 216 : by norm_num
  rw this
  norm_num
  sorry

end base_2_to_base_4_conversion_l419_419781


namespace rd_expense_necessary_for_increase_l419_419933

theorem rd_expense_necessary_for_increase :
  ∀ (R_and_D_t : ℝ) (delta_APL_t1 : ℝ),
  R_and_D_t = 3289.31 → delta_APL_t1 = 1.55 →
  R_and_D_t / delta_APL_t1 = 2122 := 
by
  intros R_and_D_t delta_APL_t1 hR hD
  rw [hR, hD]
  norm_num
  sorry

end rd_expense_necessary_for_increase_l419_419933


namespace recurring_decimal_to_fraction_l419_419125

theorem recurring_decimal_to_fraction : ∀ x : ℝ, (x = 7 + (1/3 : ℝ)) → x = (22/3 : ℝ) :=
by
  sorry

end recurring_decimal_to_fraction_l419_419125


namespace max_sum_permutations_l419_419273

open List

theorem max_sum_permutations : 
  let s := [1, 2, 3, 4, 5, 6]
  let sum_fun (l : List Nat) : Nat := 
    match l with 
    | [a, b, c, d, e, f] => a * b + b * c + c * d + d * e + e * f + f * a
    | _ => 0
  let P := s.permutations.map sum_fun |>.maximumD 0
  let Q := s.permutations.filter (fun l => sum_fun l = P) |>.length
  in P + Q = 122 :=
by
  sorry

end max_sum_permutations_l419_419273


namespace octagon_diagonals_l419_419593

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l419_419593


namespace length_of_second_platform_is_correct_l419_419073

-- Define the constants
def lt : ℕ := 70  -- Length of the train
def l1 : ℕ := 170  -- Length of the first platform
def t1 : ℕ := 15  -- Time to cross the first platform
def t2 : ℕ := 20  -- Time to cross the second platform

-- Calculate the speed of the train
def v : ℕ := (lt + l1) / t1

-- Define the length of the second platform
def l2 : ℕ := 250

-- The proof statement
theorem length_of_second_platform_is_correct : lt + l2 = v * t2 := sorry

end length_of_second_platform_is_correct_l419_419073


namespace planting_trees_system_of_equations_l419_419296

/-- This formalizes the problem where we have 20 young pioneers in total, 
each boy planted 3 trees, each girl planted 2 trees,
and together they planted a total of 52 tree seedlings.
We need to formalize proving that the system of linear equations is as follows:
x + y = 20
3x + 2y = 52
-/
theorem planting_trees_system_of_equations (x y : ℕ) (h1 : x + y = 20)
  (h2 : 3 * x + 2 * y = 52) : 
  (x + y = 20 ∧ 3 * x + 2 * y = 52) :=
by
  exact ⟨h1, h2⟩

end planting_trees_system_of_equations_l419_419296


namespace find_values_of_a_and_b_l419_419745

-- Definition of the problem and required conditions:
def symmetric_point (a b : ℝ) : Prop :=
  (a = -2) ∧ (b = -3)

theorem find_values_of_a_and_b (a b : ℝ) 
  (h : (a, -3) = (-2, -3) ∨ (2, b) = (2, -3) ∧ (a = -2)) :
  symmetric_point a b :=
by
  sorry

end find_values_of_a_and_b_l419_419745


namespace holiday_customers_l419_419027

theorem holiday_customers (h1 : ∀ t : ℕ, t = 1 → NumberOfPeoplePerHour = 175) 
                          (h2 : NumberOfPeoplePerHourDuringHoliday = 2 * NumberOfPeoplePerHour) :
  NumberOfPeopleIn8Hours = 8 * NumberOfPeoplePerHourDuringHoliday :=
by 
  -- Definition of Number of People Per Hour
  let NumberOfPeoplePerHour := 175
  
  -- Condition: Number of People Per Hour doubles during holiday season
  let NumberOfPeoplePerHourDuringHoliday := 2 * NumberOfPeoplePerHour

  -- Calculation for 8 hours
  let NumberOfPeopleIn8Hours := 8 * NumberOfPeoplePerHourDuringHoliday

  show NumberOfPeopleIn8Hours = 2800, from sorry

end holiday_customers_l419_419027


namespace isosceles_triangle_circumcircle_area_l419_419864

noncomputable def area_of_circumcircle (a b c : ℝ) : ℝ :=
  let BD := real.sqrt (a^2 - ((c / 2)^2))
  let OD := (2 / 3) * BD
  let r := real.sqrt (a^2 - OD^2)
  real.pi * r^2

theorem isosceles_triangle_circumcircle_area :
  area_of_circumcircle 4 4 3 = 9.8889 * real.pi :=
sorry

end isosceles_triangle_circumcircle_area_l419_419864


namespace sum_of_elements_l419_419687

variable {α : Type*} [add_comm_group α]

theorem sum_of_elements {a1 a2 a3 a4 : α} (h : (7 : ℝ) * ((a1 : ℝ) + a2 + a3 + a4) = 28) : (a1 + a2 + a3 + a4 = 4) :=
by
  sorry

end sum_of_elements_l419_419687


namespace sum_of_segments_l419_419002

noncomputable def segment_sum (AB_len CB_len FG_len : ℕ) : ℝ :=
  199 * (Real.sqrt (AB_len * AB_len + CB_len * CB_len) +
         Real.sqrt (AB_len * AB_len + FG_len * FG_len))

theorem sum_of_segments : segment_sum 5 6 8 = 199 * (Real.sqrt 61 + Real.sqrt 89) :=
by
  sorry

end sum_of_segments_l419_419002


namespace average_weight_of_eight_boys_l419_419728

theorem average_weight_of_eight_boys :
  let avg16 := 50.25
  let avg24 := 48.55
  let total_weight_16 := 16 * avg16
  let total_weight_all := 24 * avg24
  let W := (total_weight_all - total_weight_16) / 8
  W = 45.15 :=
by
  sorry

end average_weight_of_eight_boys_l419_419728


namespace average_decrease_rate_required_price_reduction_l419_419038

-- Define the conditions
def factory_price_2019 : ℝ := 200
def factory_price_2021 : ℝ := 162
def daily_sold_2019 : ℕ := 20
def price_increase_per_reduction : ℕ := 10
def price_reduction_per_unit : ℝ := 5
def target_daily_profit : ℝ := 1150

-- Part 1: Prove the average decrease rate
theorem average_decrease_rate : 
  ∃ (x : ℝ), (factory_price_2019 * (1 - x)^2 = factory_price_2021) ∧ x = 0.1 :=
begin
  sorry
end

-- Part 2: Prove the required unit price reduction
theorem required_price_reduction :
  ∃ (m : ℝ), ((38 - m) * (daily_sold_2019 + 2 * m / price_reduction_per_unit) = target_daily_profit) ∧ m = 15 :=
begin
  sorry
end

end average_decrease_rate_required_price_reduction_l419_419038


namespace no_subset_superset_or_equality_l419_419702

open Set

def P : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2}
def Q : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ x : ℝ, ∃ y : ℝ, p = (x, y) ∧ y = x^2}

theorem no_subset_superset_or_equality (P Q : Type) :
  ¬(P ⊆ Q ∨ Q ⊆ P ∨ P = Q) :=
sorry

end no_subset_superset_or_equality_l419_419702


namespace parabola_focus_l419_419502

-- Define the hyperbola parameters and conditions
variable (a b p : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hp : 0 < p)
variable (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → x > 0 ∧ y > 0)
variable (eccentricity_eq : b / a = sqrt 3)
variable (area_eq : ∀ A B : ℝ × ℝ, let y₁ := (B.snd - A.snd)
                                    let x₁ := (B.fst - A.fst)
                                    1/2 * abs (y₁ * x₁) = 2 * sqrt 3)

-- Define the statement to prove the focus of the parabola
theorem parabola_focus : (∀ p > 0, focus = (sqrt 2, 0)) :=
sorry  -- Proof to be provided

end parabola_focus_l419_419502


namespace simplify_and_evaluate_l419_419305

-- Definitions of given conditions
def a := 1
def b := 2

-- Statement of the theorem
theorem simplify_and_evaluate : (a * b + (a^2 - a * b) - (a^2 - 2 * a * b) = 4) :=
by
  -- Using sorry to indicate the proof is to be completed
  sorry

end simplify_and_evaluate_l419_419305


namespace max_values_of_f_inequality_l419_419419

def f (x : ℝ) : ℝ := (x^2 + 1)^2 / (x * (x^2 - 1))

theorem max_values_of_f_inequality (a : ℝ) :
  |a| > 4 ↔ ∃ x : ℝ, f x = a := sorry

end max_values_of_f_inequality_l419_419419


namespace tangent_line_intersect_x_l419_419873

noncomputable def tangent_intercept_x : ℚ := 9/2

theorem tangent_line_intersect_x (x : ℚ)
  (h₁ : x > 0)
  (h₂ : ∃ r₁ r₂ d : ℚ, r₁ = 3 ∧ r₂ = 5 ∧ d = 12 ∧ x = (r₂ * d) / (r₁ + r₂)) :
  x = tangent_intercept_x :=
by
  sorry

end tangent_line_intersect_x_l419_419873


namespace ammonia_produced_l419_419973

theorem ammonia_produced (KOH NH4I KI NH3 H2O : Type) 
  [HasAdd KOH] [HasAdd NH4I] [HasAdd KI] [HasAdd NH3] [HasAdd H2O] :
  (3 * KOH) + (3 * NH4I) = (3 * KI) + (3 * NH3) + (3 * H2O) → 
  3 * NH4I + KOH = 3 * NH3 :=
by 
  sorry

end ammonia_produced_l419_419973


namespace placemat_length_l419_419884

noncomputable def calculate_placemat_length
    (R : ℝ)
    (num_mats : ℕ)
    (mat_width : ℝ)
    (overlap_ratio : ℝ) : ℝ := 
    let circumference := 2 * Real.pi * R
    let arc_length := circumference / num_mats
    let angle := 2 * Real.pi / num_mats
    let chord_length := 2 * R * Real.sin (angle / 2)
    let effective_mat_length := chord_length / (1 - overlap_ratio * 2)
    effective_mat_length

theorem placemat_length (R : ℝ) (num_mats : ℕ) (mat_width : ℝ) (overlap_ratio : ℝ): 
    R = 5 ∧ num_mats = 8 ∧ mat_width = 2 ∧ overlap_ratio = (1 / 4)
    → calculate_placemat_length R num_mats mat_width overlap_ratio = 7.654 :=
by
  sorry

end placemat_length_l419_419884


namespace rowing_trip_time_l419_419058

def rowing_time_to_and_back (rowing_speed current_speed distance : ℝ) : ℝ :=
  let downstream_speed := rowing_speed + current_speed
  let upstream_speed := rowing_speed - current_speed
  let time_downstream := distance / downstream_speed
  let time_upstream := distance / upstream_speed
  time_downstream + time_upstream

theorem rowing_trip_time 
  (rowing_speed : ℝ := 5) 
  (current_speed : ℝ := 1)
  (distance : ℝ := 2.4)
  : rowing_time_to_and_back rowing_speed current_speed distance = 1 := 
by
  sorry

end rowing_trip_time_l419_419058


namespace eval_expression_l419_419981

-- Define floor function as [x]
def floor (x : ℝ) : ℤ := Int.floor x

theorem eval_expression
  (h1 : floor 6.5 = 6)
  (h2 : floor (2 / 3) = 0)
  (h3 : floor 2 = 2)
  (h4 : floor 8.4 = 8) :
  (6 : ℝ) * (0 : ℝ) + (2 : ℝ) * 7.2 + (8 : ℝ) - 6.0 = 16.4 :=
by
  sorry

end eval_expression_l419_419981


namespace probability_of_six_on_fair_die_l419_419000

theorem probability_of_six_on_fair_die :
  let num_faces := 6 in
  let favorable_outcomes := 1 in
  1 / num_faces = 1 / 6 :=
by
  sorry

end probability_of_six_on_fair_die_l419_419000


namespace pages_already_read_l419_419937

theorem pages_already_read (total_pages : ℕ) (pages_left : ℕ) (h_total : total_pages = 563) (h_left : pages_left = 416) :
  total_pages - pages_left = 147 :=
by
  sorry

end pages_already_read_l419_419937


namespace sqrt_sum_eq_l419_419800

theorem sqrt_sum_eq : 
  sqrt (20 - 8 * sqrt 5) + sqrt (20 + 8 * sqrt 5) = 2 * sqrt 10 := 
by 
  sorry

end sqrt_sum_eq_l419_419800


namespace isosceles_triangle_circle_area_l419_419827

theorem isosceles_triangle_circle_area 
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : a = 4) 
  (h3 : c = 3) 
  (h4 : a = 4) 
  (h5 : b = 4)
  (h6 : c ≠ a)
  (h7 : c ≠ b) :
  let r := 4 in π * r ^ 2 = 16 * π :=
by
  sorry

end isosceles_triangle_circle_area_l419_419827


namespace path_length_of_B_is_4_l419_419925

variables (BD : ℝ) (AD PQ : ℝ) 
variable (semicircle_radius : BD = 4 / π)

theorem path_length_of_B_is_4 (BD_eq : BD = 4 / π) :
  let B_path_length := 2 * (π * BD / 2) in
  (B_path_length = 4) :=
by
  sorry

end path_length_of_B_is_4_l419_419925


namespace arithmetic_geometric_sequences_correct_sum_c_seq_correct_l419_419499

-- Define the sequences and their constraints
def arithmetic_seq (a₁ d : ℕ → ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

def geometric_seq (b₁ q : ℕ → ℤ) (n : ℕ) : ℤ :=
  b₁ * q^(n - 1)

-- Define the initial conditions
variables a₁ d b₁ q : ℕ → ℤ
variables S₃ b₃ a₃ : ℤ

axiom h1 (b₁ = 2 ∨ a₁ = -1): b₁ = 2 ∧ -2 * a₁ = 2
axiom h2 : (a₃ = ∑ i in finset.range 3, arithmetic_seq a₁ d i)
axiom h3 : b₃ = ∑ i in finset.range 3, geometric_seq b₁ q i
axiom h4 : S₃ + 2 * b₃ = 7

-- Define and prove the arithmetic and geometric sequences based on the constraints
theorem arithmetic_geometric_sequences_correct :
  (∀ n, arithmetic_seq (-1) (-2) n = 2n - 3 ) ∧ 
  (∀ n, geometric_seq 2 2 n = 2^n) :=
sorry

-- Define the sequence cn
def c_seq (n : ℕ) : ℤ :=
  (-1)^(n-1) * (arithmetic_seq (-1) (-2) n) / (geometric_seq 2 2 n)

def T_n (n : ℕ) : ℤ :=
  ∑ i in finset.range n, c_seq i

-- Prove the sum of the first n terms for the sequence
theorem sum_c_seq_correct (n : ℕ) :
  T_n n = -5 / 9 + 2 / 9 * (-1 / 2)^(n - 1) + (-1)^(n - 1) * (2 * n - 3) / (3 * 2^n) :=
sorry

end arithmetic_geometric_sequences_correct_sum_c_seq_correct_l419_419499


namespace adults_more_than_children_l419_419061

-- Definitions based on the conditions
def number_of_men : ℕ := 120
def difference_men_women : ℕ := 80
def total_persons : ℕ := 240

-- Hypotheses based on the conditions
variable (M W A C : ℕ)
hypothesis (h1 : M = number_of_men)
hypothesis (h2 : M = W + difference_men_women)
hypothesis (h3 : M + W + C = total_persons)

-- Statement of the proof
theorem adults_more_than_children : A - C = 80 :=
by
  -- The proof will be filled in here
  sorry

end adults_more_than_children_l419_419061


namespace bingo_first_column_permutations_l419_419637

theorem bingo_first_column_permutations : 
  let first_column_set := {n | 10 ≤ n ∧ n ≤ 25} in
  fintype.card (finset.pi finset.univ (λ _, first_column_set).erase_none).subtype = 5 →
  fintype.card {s : fin (5) → first_column_set // function.injective s} = 524160 :=
by
  let first_column_set := {n | 10 ≤ n ∧ n ≤ 25}
  let num_possibilities := list.permutations (finset.range 16).erase_none |>.length
  num_possibilities = 524160
  sorry

end bingo_first_column_permutations_l419_419637


namespace smallest_divisor_l419_419738

theorem smallest_divisor (n : ℕ) (h1 : n = 999) :
  ∃ d : ℕ, 2.45 ≤ (999 : ℝ) / d ∧ (999 : ℝ) / d < 2.55 ∧ d = 392 :=
by
  sorry

end smallest_divisor_l419_419738


namespace positive_difference_l419_419787

theorem positive_difference (a b : ℕ) (h1 : a = (6^2 + 6^2) / 6) (h2 : b = (6^2 * 6^2) / 6) : a < b ∧ b - a = 204 :=
by
  sorry

end positive_difference_l419_419787


namespace max_sqrt_sum_l419_419021

theorem max_sqrt_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) :
  sqrt (3 * x) + sqrt (4 * y) ≤ sqrt 7 := by
sorry

end max_sqrt_sum_l419_419021


namespace polynomial_roots_diff_l419_419135

open Polynomial

noncomputable def polynomial := (16 : ℚ) * X^3 - 20 * X^2 + 9 * X - 1

theorem polynomial_roots_diff :
  (∀ a b c : ℚ, a ≠ b → b ≠ c → a ≠ c →
    is_root polynomial a ∧ is_root polynomial b ∧ is_root polynomial c ∧ 
    (b = a + (1 / 2)) ∧ (c = a + 2 * (1 / 2)) → 
     (c - a) = (1 / 2)) :=
sorry

end polynomial_roots_diff_l419_419135


namespace no_absolute_winner_l419_419903

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end no_absolute_winner_l419_419903


namespace solve_for_a_l419_419605

-- Given conditions
variables (a b d : ℕ)
hypotheses
  (h1 : a + b = d)
  (h2 : b + d = 7)
  (h3 : d = 4)

-- Prove that a = 1
theorem solve_for_a : a = 1 :=
by {
  sorry
}

end solve_for_a_l419_419605


namespace eval_complex_product_l419_419123

def z1 : ℂ := 3 * Real.sqrt 5 - 5 * Complex.i
def z2 : ℂ := 2 * Real.sqrt 2 + 4 * Complex.i

theorem eval_complex_product :
  abs (z1 * z2) = 8 * Real.sqrt 105 :=
by
  sorry

end eval_complex_product_l419_419123


namespace divisibility_of_2b_by_a_l419_419165

theorem divisibility_of_2b_by_a (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b)
  (h_cond : ∃ᶠ m in at_top, ∃ᶠ n in at_top, (∃ k₁ : ℕ, m^2 + a * n + b = k₁^2) ∧ (∃ k₂ : ℕ, n^2 + a * m + b = k₂^2)) :
  a ∣ 2 * b :=
sorry

end divisibility_of_2b_by_a_l419_419165


namespace prob_six_largest_is_correct_l419_419031

noncomputable def probability_six_is_largest : ℚ :=
  let total_ways := (Finset.card (Finset.powersetLen 3 (Finset.range 7))) in
  let favorable_ways := (Finset.card (Finset.powersetLen 3 (Finset.range 6))) in
  (favorable_ways : ℚ) / total_ways

theorem prob_six_largest_is_correct : probability_six_is_largest = 4 / 7 := by
  sorry

end prob_six_largest_is_correct_l419_419031


namespace area_of_circle_passing_through_vertices_l419_419850

noncomputable def circle_area_through_isosceles_triangle_vertices 
  (a b c : ℝ) (h_isosceles: (a = b) (h_sides: a = 4) (h_base: c = 3) : ℝ :=
π *(√((4^2 - (3/2)^2)/2 + (3/2))^2

theorem area_of_circle_passing_through_vertices :
  circle_area_through_isosceles_triangle_vertices 4 4 3 = 5.6875 * π :=
sorry

end area_of_circle_passing_through_vertices_l419_419850


namespace lcm_of_numbers_l419_419815

theorem lcm_of_numbers (a b c d : ℕ) (h1 : a = 8) (h2 : b = 24) (h3 : c = 36) (h4 : d = 54) :
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 216 := 
by 
  sorry

end lcm_of_numbers_l419_419815


namespace problem_equivalence_l419_419111

theorem problem_equivalence (n : ℕ) (H₁ : 2 * 2006 = 1) (H₂ : ∀ n : ℕ, (2 * n + 2) * 2006 = 3 * (2 * n * 2006)) :
  2008 * 2006 = 3 ^ 1003 :=
by
  sorry

end problem_equivalence_l419_419111


namespace max_values_of_f_inequality_l419_419417

def f (x : ℝ) : ℝ := (x^2 + 1)^2 / (x * (x^2 - 1))

theorem max_values_of_f_inequality (a : ℝ) :
  |a| > 4 ↔ ∃ x : ℝ, f x = a := sorry

end max_values_of_f_inequality_l419_419417


namespace area_of_region_l419_419132

theorem area_of_region : 
  (let area := 1 / 2 * abs (6 * 3) in
    ∀ x y : ℝ, |x + 2 * y| + |x - 2 * y| ≤ 6 → area = 9) := 
begin
  -- proof omitted; to be provided by user
  sorry
end

end area_of_region_l419_419132


namespace exist_extreme_value_if_a_eq_1_exist_a_such_that_f_geq_g_ax_l419_419150

section Problem1

variable (a : ℝ) (x : ℝ)
def f (x : ℝ) := a * x^2 - x
def g (x : ℝ) := Real.log x

-- (1) If a=1, find the extreme value of the function y=f(x)-3g(x)
theorem exist_extreme_value_if_a_eq_1 :
  (a = 1) → ∃ (m : ℝ), (∀ x, y = (f x - 3 * g x) ≥ m) ∧ 
  (y = (f x - 3 * g x) does not have a maximum value) := sorry

-- (2) Does there exist a real number a such that f(x) ≥ g(ax) holds? 
--     If so, find the set of values for a; otherwise, explain the reason.
theorem exist_a_such_that_f_geq_g_ax :
  ∃ (a : ℝ), (∀ x, f x ≥ g (a * x)) ↔ a = 1 := sorry

end Problem1

end exist_extreme_value_if_a_eq_1_exist_a_such_that_f_geq_g_ax_l419_419150


namespace isosceles_triangle_circumcircle_area_l419_419860

noncomputable def area_of_circumcircle (a b c : ℝ) : ℝ :=
  let BD := real.sqrt (a^2 - ((c / 2)^2))
  let OD := (2 / 3) * BD
  let r := real.sqrt (a^2 - OD^2)
  real.pi * r^2

theorem isosceles_triangle_circumcircle_area :
  area_of_circumcircle 4 4 3 = 9.8889 * real.pi :=
sorry

end isosceles_triangle_circumcircle_area_l419_419860


namespace part1_inter_complement_part2_range_l419_419999

-- Part 1 proof problem in Lean 4 statement
theorem part1_inter_complement : 
  let A := {x : ℝ | x^2 - 5 * x + 4 ≤ 0},
      B := {x : ℝ | (x - Real.sqrt 2) * (x - (Real.sqrt 2)^2 - 1) < 0},
      complement_B := {x : ℝ | x ≤ Real.sqrt 2 ∨ x ≥ 3} 
  in A ∩ complement_B = {x : ℝ | 1 ≤ x ∧ x ≤ Real.sqrt 2 ∨ 3 ≤ x ∧ x ≤ 4} := 
begin
  sorry
end

-- Part 2 proof problem in Lean 4 statement
theorem part2_range : 
  let A := {x : ℝ | x^2 - 5 * x + 4 ≤ 0},
      B := λ a : ℝ, {x : ℝ | (x - a) * (x - (a^2 + 1)) < 0}
  in ∀ a : ℝ, B a ⊆ A ↔ 1 ≤ a ∧ a ≤ Real.sqrt 3 :=
begin
  sorry
end

end part1_inter_complement_part2_range_l419_419999


namespace S_invariant_l419_419399

-- Definitions of S
def S := {z : ℂ | ∃ x y : ℝ, z = x + iy ∧ x^2 + y^2 ≤ 4}

-- Definition of transformation
def T (z : ℂ) := (1 / 2 + 1 / 2 * complex.I) * z

-- Definition of the subset condition
def in_S (z : ℂ) : Prop :=
  ∃ x y : ℝ, z = x + iy ∧ x^2 + y^2 ≤ 4

-- Proof problem statement
theorem S_invariant (z : ℂ) (hz : z ∈ S) : T z ∈ S := sorry

end S_invariant_l419_419399


namespace area_of_circle_passing_through_vertices_l419_419847

noncomputable def circle_area_through_isosceles_triangle_vertices 
  (a b c : ℝ) (h_isosceles: (a = b) (h_sides: a = 4) (h_base: c = 3) : ℝ :=
π *(√((4^2 - (3/2)^2)/2 + (3/2))^2

theorem area_of_circle_passing_through_vertices :
  circle_area_through_isosceles_triangle_vertices 4 4 3 = 5.6875 * π :=
sorry

end area_of_circle_passing_through_vertices_l419_419847


namespace repeating_decimal_to_fraction_l419_419127

theorem repeating_decimal_to_fraction : 
  (∃ (x : ℚ), x = 7 + 3 / 9) → 7 + 3 / 9 = 22 / 3 :=
by
  intros h
  sorry

end repeating_decimal_to_fraction_l419_419127


namespace resistor_value_l419_419773

-- Definitions based on given conditions
def U : ℝ := 9 -- Volt reading by the voltmeter
def I : ℝ := 2 -- Current reading by the ammeter
def U_total : ℝ := 2 * U -- Total voltage in the series circuit

-- Stating the theorem
theorem resistor_value (R₀ : ℝ) :
  (U_total = I * (2 * R₀)) → R₀ = 9 :=
by
  intro h
  sorry

end resistor_value_l419_419773


namespace complex_magnitude_product_l419_419118

noncomputable def z1 : ℂ := 3 * Real.sqrt 5 - 5 * Complex.i
noncomputable def z2 : ℂ := 2 * Real.sqrt 2 + 4 * Complex.i
noncomputable def magnitude (z : ℂ) : ℝ := Complex.abs z

theorem complex_magnitude_product :
  magnitude (z1 * z2) = 12 * Real.sqrt 35 :=
by
  have z1 := z1
  have z2 := z2
  sorry

end complex_magnitude_product_l419_419118


namespace circumference_area_equal_numerically_but_different_units_l419_419381

-- Definitions for the conditions
def radius : ℝ := 2
def pi_val : ℝ := Real.pi
def circumference (r : ℝ) : ℝ := 2 * pi_val * r
def area_of_circle (r : ℝ) : ℝ := pi_val * r * r

-- The theorem to prove
theorem circumference_area_equal_numerically_but_different_units :
  circumference radius = area_of_circle radius :=
by
  -- The proof content is omitted as per the instructions
  sorry

end circumference_area_equal_numerically_but_different_units_l419_419381


namespace count_statements_implying_implication_l419_419943

variables (p q r : Prop)

def statement1 := p ∧ q ∧ ¬ r
def statement2 := ¬ p ∧ q ∧ r
def statement3 := p ∧ ¬ q ∧ r
def statement4 := ¬ p ∧ ¬ q ∧ r

theorem count_statements_implying_implication :
  (statement1 p q r → (p ∧ q → r)) +
  (statement2 p q r → (p ∧ q → r)) +
  (statement3 p q r → (p ∧ q → r)) +
  (statement4 p q r → (p ∧ q → r)) = 3 :=
sorry

end count_statements_implying_implication_l419_419943


namespace sqrt_three_irrational_sqrt_three_code_l419_419744

theorem sqrt_three_irrational :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p * p = 3 * q * q) :=
by
  sorry

theorem sqrt_three_code {
  ∃ (f : ℕ → ℕ), (∀ m n, m > n → f m ≠ f n) } :=
by
  sorry

end sqrt_three_irrational_sqrt_three_code_l419_419744


namespace premium_rate_l419_419059

theorem premium_rate (P : ℝ) : (14400 / (100 + P)) * 5 = 600 → P = 20 :=
by
  intro h
  sorry

end premium_rate_l419_419059


namespace solve_eq_l419_419454

theorem solve_eq (x : ℝ) : (∛(5 - x / 3) = -4) → x = 207 :=
by {
    intro hyp,
    -- proof steps here
    sorry
}

end solve_eq_l419_419454


namespace maximum_non_managers_has_bound_l419_419639

-- Definition of the problem
def ratio_greater (m n : ℕ) : Prop := m * 24 > 7 * n

def maximum_non_managers (m n : ℕ) : Prop := ratio_greater m n ∧ m = 8 ∧ n ≤ 27

-- Lean statement for the problem
theorem maximum_non_managers_has_bound 
    (m : ℕ) 
    (n : ℕ) 
    (h1 : ratio_greater m n) 
    (h2 : m = 8) : 
    n ≤ 27 :=
begin
  sorry
end

end maximum_non_managers_has_bound_l419_419639


namespace num_diagonals_octagon_l419_419568

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l419_419568


namespace solve_for_x_l419_419616

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end solve_for_x_l419_419616


namespace area_of_circumscribed_circle_isosceles_triangle_l419_419856

theorem area_of_circumscribed_circle_isosceles_triangle :
  ∃ (r : ℝ), (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi) :=
by
  -- Consider the isosceles triangle conditions
  let a : ℝ := 4
  let b : ℝ := 4
  let c : ℝ := 3
  let BD := Real.sqrt(a^2 - (c/2)^2)
  let r := 8 / BD
  have h1 : BD = Real.sqrt 13.75 := by 
    -- Calculate the altitude BD
    calc
      BD = Real.sqrt(a^2 -  (c/2)^2) : rfl
      ... = Real.sqrt(16 - (3/2)^2) : rfl
      ... = Real.sqrt 13.75 : rfl
  
  use r
  have h2 : r = 8 / Real.sqrt 13.75 := by 
    -- Simplify the radius expression
    sorry

  have h3 : Real.pi * r ^ 2 = 256 / 55 * Real.pi := by 
    -- Calculate the area
    calc
      Real.pi * r ^ 2 = Real.pi * (8 / Real.sqrt 13.75) ^ 2 : by rw h2
      ... = Real.pi * (64 / 13.75) : by rw [pow_two, mul_div_assoc, mul_one, div_mul_div_same]
      ... = (256 / 54.6875) * Real.pi : by rw mul_comm
      ...   = (256 / 55) * Real.pi : by norm_num
    sorry
  
  show (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi),
  from ⟨h2, h3⟩
  sorry

end area_of_circumscribed_circle_isosceles_triangle_l419_419856


namespace find_ordered_pair_l419_419465

theorem find_ordered_pair : ∃ m n : ℤ, 0 < m ∧ 0 < n ∧ 14 * m * n = 55 - 7 * m - 2 * n ∧ m = 1 ∧ n = 3 := by
  -- Defining m and n with their constraints
  let m := 1
  let n := 3
  have h1 : 0 < m := by norm_num
  have h2 : 0 < n := by norm_num
  -- Showing the equation holds
  have h3 : 14 * m * n = 55 - 7 * m - 2 * n := by
    calc
      14 * m * n = 14 * 1 * 3 := by congr; norm_num
             ... = 42 := by norm_num
      55 - 7 * m - 2 * n = 55 - 7 * 1 - 2 * 3 := by congr; norm_num
                     ... = 55 - 7 - 6 := by norm_num
                     ... = 42 := by norm_num
  -- Combining constraints to create the proof
  use m, n
  exact and.intro h1 (and.intro h2 (and.intro h3 (and.intro rfl rfl)))

end find_ordered_pair_l419_419465


namespace solve_cubic_root_eq_l419_419449

theorem solve_cubic_root_eq (x : ℝ) : (real.cbrt (5 - x / 3) = -4) → x = 207 :=
by
  sorry

end solve_cubic_root_eq_l419_419449


namespace insect_reaches_northernmost_point_l419_419390

-- Definitions and conditions
def radius_of_disk : ℝ := 3 / 2
def angular_velocity : ℝ := 2 * real.pi / 15
def initial_position : ℝ := -radius_of_disk -- southernmost point
def insect_speed : ℝ := 1 -- inches per second

-- Main theorem to prove
theorem insect_reaches_northernmost_point :
  ∃ t : ℝ, 
  initial_position + insect_speed * t = radius_of_disk ∧ 
  (initial_position + insect_speed * t = -cos (angular_velocity * t + real.pi)) :=
sorry 

end insect_reaches_northernmost_point_l419_419390


namespace holiday_customers_l419_419026

theorem holiday_customers (h1 : ∀ t : ℕ, t = 1 → NumberOfPeoplePerHour = 175) 
                          (h2 : NumberOfPeoplePerHourDuringHoliday = 2 * NumberOfPeoplePerHour) :
  NumberOfPeopleIn8Hours = 8 * NumberOfPeoplePerHourDuringHoliday :=
by 
  -- Definition of Number of People Per Hour
  let NumberOfPeoplePerHour := 175
  
  -- Condition: Number of People Per Hour doubles during holiday season
  let NumberOfPeoplePerHourDuringHoliday := 2 * NumberOfPeoplePerHour

  -- Calculation for 8 hours
  let NumberOfPeopleIn8Hours := 8 * NumberOfPeoplePerHourDuringHoliday

  show NumberOfPeopleIn8Hours = 2800, from sorry

end holiday_customers_l419_419026


namespace cory_needs_more_money_l419_419435

def money_needed_to_buy_candies (current_money : ℝ) (num_packs : ℕ) (cost_per_pack : ℝ) : ℝ :=
  (num_packs * cost_per_pack) - current_money

theorem cory_needs_more_money (current_money num_packs cost_per_pack : ℝ) (h_current_money : current_money = 20) (h_num_packs : num_packs = 2) (h_cost_per_pack: cost_per_pack = 49) :
  money_needed_to_buy_candies current_money num_packs cost_per_pack = 78 := by
  sorry

end cory_needs_more_money_l419_419435


namespace line_bisects_angle_DAB_l419_419101

open Real EuclideanGeometry

variables {A B C D E : Point}
variables (ℓ : Line)

-- Assumptions
axiom parallelogramABCD : parallelogram A B C D
axiom cyclicQuadrilateralBCED : cyclicQuadrilateral B C E D
axiom lineThroughA : ℓ.contains A
axiom intersectsDCatF : ∃ F : Point, F ≠ D ∧ F ≠ C ∧ F ∈ segment D C ∧ ℓ.contains F
axiom intersectsBCatG : ∃ G : Point, ℓ.contains G ∧ G ∈ lineThrough B C
axiom EF_eq_EG_EC : ∀ F G : Point, EF = EG ∧ EF = EC

-- Goal
theorem line_bisects_angle_DAB (ℓ : Line) :
  bisects_angle ℓ ∠ D A B :=
begin
  sorry
end

end line_bisects_angle_DAB_l419_419101


namespace sequence_value_l419_419110

theorem sequence_value (a b c d x : ℕ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 17) (h4 : d = 33)
  (h5 : b - a = 4) (h6 : c - b = 8) (h7 : d - c = 16) (h8 : x - d = 32) : x = 65 := by
  sorry

end sequence_value_l419_419110


namespace project_completion_time_l419_419881

theorem project_completion_time (initial_workers : ℕ) (initial_days : ℕ) (extra_workers : ℕ) (extra_days : ℕ) : 
  initial_workers = 10 →
  initial_days = 15 →
  extra_workers = 5 →
  extra_days = 5 →
  total_days = 6 := by
  sorry

end project_completion_time_l419_419881


namespace passes_through_intersection_of_circumcircles_l419_419727

-- Define the acute-angled triangle ABC with angle BAC = 60°
variables (A B C : Point)
variables [triangle : Triangle ABC]
variables (BB1 CC1 I : Line)
variables (B' C' : Point)
variable (parallelogram : Parallelogram AB' IC')

-- Define angle bisectors
variables [angleBisectorBB1 : AngleBisector B B1]
variables [angleBisectorCC1 : AngleBisector C C1]

-- Provided angle BAC is 60 degrees
variable (angleBAC : Angle A B C = 60)

theorem passes_through_intersection_of_circumcircles :
  Line B' C' passes_through (Circumcircle B C C1 ∩ Circumcircle C B B1) := 
sorry

end passes_through_intersection_of_circumcircles_l419_419727


namespace determine_xy_with_B_l419_419081

theorem determine_xy_with_B (x y : ℕ) :
  ∃ (B : ℕ × ℕ → ℕ), (B (x, y) = (x + y) * (x + y + 1) - y) ∧
  (x = B (x, y) - (int.floor (real.sqrt (B (x, y))).toNat)^2) ∧
  (y = int.floor (real.sqrt (B (x, y))).toNat - x) :=
by
  sorry

end determine_xy_with_B_l419_419081


namespace find_least_n_l419_419099

def sequence_x (n : ℕ) : ℝ :=
  match n with
  | 1 => 1
  | 2 => 1
  | 3 => 2 / 3
  | _ => sequence_x(n-1)^2 * sequence_x(n-2) / (2 * sequence_x(n-2)^2 - sequence_x(n-1) * sequence_x(n-3))

theorem find_least_n (HN : ∃ n : ℕ, sequence_x n ≤ 1 / 10^6) : ∃ n : ℕ, sequence_x n ≤ 1 / 10^6 ∧ ∀ m < n, sequence_x m > 1 / 10^6 :=
begin
  use 13,
  simp [sequence_x],
  sorry
end

end find_least_n_l419_419099


namespace no_integer_roots_of_quadratic_l419_419991

theorem no_integer_roots_of_quadratic
  (a b c : ℤ) (f : ℤ → ℤ)
  (h_def : ∀ x, f x = a * x * x + b * x + c)
  (h_a_nonzero : a ≠ 0)
  (h_f0_odd : Odd (f 0))
  (h_f1_odd : Odd (f 1)) :
  ∀ x : ℤ, f x ≠ 0 :=
by
  sorry

end no_integer_roots_of_quadratic_l419_419991


namespace larger_factor_of_lcm_l419_419314

theorem larger_factor_of_lcm (A B : ℕ) (hcf lcm : ℕ)
  (hcf_cond : Nat.gcd A B = 23)
  (A_cond : A = 322)
  (lcm_cond : lcm = 23 * 13 * 14) :
  ∃ X, LCM (A, B) = 23 * 13 * X ∧ X = 14 :=
by
  sorry

end larger_factor_of_lcm_l419_419314


namespace degree_of_polynomial_raised_to_power_l419_419343

def polynomial_degree (p : Polynomial ℤ) : ℕ := p.natDegree

theorem degree_of_polynomial_raised_to_power :
  let p : Polynomial ℤ := Polynomial.C 5 * Polynomial.X ^ 3 + Polynomial.C 7
  in polynomial_degree (p ^ 15) = 45 :=
by
  sorry

end degree_of_polynomial_raised_to_power_l419_419343


namespace solution_set_of_inequality_l419_419500

theorem solution_set_of_inequality
  (a b : ℝ)
  (h1 : a < 0) 
  (h2 : b / a = 1) :
  { x : ℝ | (x - 1) * (a * x + b) < 0 } = { x : ℝ | x < -1 } ∪ {x : ℝ | 1 < x} :=
by
  sorry

end solution_set_of_inequality_l419_419500


namespace calculation_1_calculation_2_calculation_3_calculation_4_l419_419094

theorem calculation_1 : -3 - (-4) = 1 :=
by sorry

theorem calculation_2 : -1/3 + (-4/3) = -5/3 :=
by sorry

theorem calculation_3 : (-2) * (-3) * (-5) = -30 :=
by sorry

theorem calculation_4 : 15 / 4 * (-1/4) = -15/16 :=
by sorry

end calculation_1_calculation_2_calculation_3_calculation_4_l419_419094


namespace problem_l419_419205

theorem problem (x n : ℕ) (h : (1 + x)^n - 1 % 7 = 0) : x = 5 ∧ n = 4 :=
begin
  sorry
end

end problem_l419_419205


namespace transform_cos_to_sin_shift_right_l419_419766

theorem transform_cos_to_sin_shift_right (x : ℝ) : 
  ∃ c : ℝ, (y = cos (2 * x - π / 3)) → (y = sin (2 * (x + c))) := 
sorry

end transform_cos_to_sin_shift_right_l419_419766


namespace diagonals_of_octagon_l419_419548

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l419_419548


namespace smallest_a1_l419_419680

noncomputable def a_seq (a : ℝ) (n : ℕ) : ℝ :=
if n = 1 then a else 13 * (a_seq a (n - 1)) - 3 * (n : ℝ)

theorem smallest_a1 (a1 : ℝ) :
    (∀ n > 1, a_seq a1 n > 0) ∧ a1 < 1 → a1 = 51 / 100 :=
sorry

end smallest_a1_l419_419680


namespace population_risk_factors_l419_419649

theorem population_risk_factors (P_one : ℚ) (P_two : ℚ) (P_A_given_BC : ℚ) : 
  P_one = 0.08 → P_two = 0.12 → P_A_given_BC = 3/5 →
  let P_NONE_given_not_B := 109 / 182 in
  let p := 109 in
  let q := 182 in
  p + q = 291 := 
begin
  intros h1 h2 h3,
  have P_NONE_given_not_B_def : P_NONE_given_not_B = 109 / 182 := rfl,
  have p_def : p = 109 := rfl,
  have q_def : q = 182 := rfl,
  rw [p_def, q_def],
  exact rfl,
end

end population_risk_factors_l419_419649


namespace nancy_flooring_l419_419708

theorem nancy_flooring (total_area : ℝ) (hallway_length : ℝ) (hallway_width : ℝ)
  (hallway_area : ℝ) (central_area : ℝ) (L : ℝ) :
  total_area = 124 ∧ hallway_length = 6 ∧ hallway_width = 4 ∧
  hallway_area = hallway_length * hallway_width ∧
  central_area = total_area - hallway_area ∧
  central_area = L^2 → L = 10 :=
begin
  sorry
end

end nancy_flooring_l419_419708


namespace joan_seashells_l419_419252

/-- Prove that Joan has 36 seashells given the initial conditions. -/
theorem joan_seashells :
  let initial_seashells := 79
  let given_mike := 63
  let found_more := 45
  let traded_seashells := 20
  let lost_seashells := 5
  (initial_seashells - given_mike + found_more - traded_seashells - lost_seashells) = 36 :=
by
  sorry

end joan_seashells_l419_419252


namespace Alton_profit_l419_419895

variable (earnings_per_day : ℕ)
variable (days_per_week : ℕ)
variable (rent_per_week : ℕ)

theorem Alton_profit (h1 : earnings_per_day = 8) (h2 : days_per_week = 7) (h3 : rent_per_week = 20) :
  earnings_per_day * days_per_week - rent_per_week = 36 := 
by sorry

end Alton_profit_l419_419895


namespace solve_complex_addition_l419_419349

noncomputable def complex_addition : Prop :=
  let i := Complex.I
  let z1 := 3 - 5 * i
  let z2 := -1 + 12 * i
  let result := 2 + 7 * i
  z1 + z2 = result

theorem solve_complex_addition :
  complex_addition :=
by
  sorry

end solve_complex_addition_l419_419349


namespace no_absolute_winner_probability_l419_419918

-- Define the probabilities of matches
def P_AB : ℝ := 0.6  -- Probability Alyosha wins against Borya
def P_BV : ℝ := 0.4  -- Probability Borya wins against Vasya

-- Define the event C that there is no absolute winner
def event_C (P_AV : ℝ) (P_VB : ℝ) : ℝ :=
  let scenario1 := P_AB * P_BV * P_AV in
  let scenario2 := P_AB * P_VB * (1 - P_AV) in
  scenario1 + scenario2

-- Main theorem to prove
theorem no_absolute_winner_probability : 
  event_C 1 0.6 = 0.24 :=
by
  rw [event_C]
  simp
  norm_num
  sorry

end no_absolute_winner_probability_l419_419918


namespace ratio_of_land_values_l419_419742

-- Define the conditions
def moon_surface_area := (1/5 : ℝ) * 200
def earth_value := 80
def moon_value := 96

-- Theorem statement
theorem ratio_of_land_values : (moon_value / earth_value) = 1.2 := by
  sorry

end ratio_of_land_values_l419_419742


namespace solve_eq_l419_419453

theorem solve_eq (x : ℝ) : (∛(5 - x / 3) = -4) → x = 207 :=
by {
    intro hyp,
    -- proof steps here
    sorry
}

end solve_eq_l419_419453


namespace find_a_l419_419608

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 :=
by
  sorry

end find_a_l419_419608


namespace _l419_419962

lemma no_rational_roots (P : ℚ[X]) (hP : P = 3 * X^4 - 4 * X^3 - 9 * X^2 + 10 * X + 5) : ¬ ∃ r : ℚ, P.eval r = 0 :=
by
  -- Polynomial definition and theorem to prove
  sorry

end _l419_419962


namespace allie_betty_total_points_product_l419_419646

def score (n : Nat) : Nat :=
  if n % 3 == 0 then 9
  else if n % 2 == 0 then 3
  else if n % 2 == 1 then 1
  else 0

def allie_points : List Nat := [5, 2, 6, 1, 3]
def betty_points : List Nat := [6, 4, 1, 2, 5]

def total_points (rolls: List Nat) : Nat :=
  rolls.foldl (λ acc n => acc + score n) 0

theorem allie_betty_total_points_product : 
  total_points allie_points * total_points betty_points = 391 := by
  sorry

end allie_betty_total_points_product_l419_419646


namespace profit_is_480_l419_419067

-- Define the conditions
def price_per_six_bars := 3
def price_per_bar := price_per_six_bars / 6
def number_of_bars := 1200
def discount_threshold := 1000
def discount_rate := 0.05
def price_first_600_bars := 1.00
def price_remaining_bars := 0.75

def total_cost_without_discount := number_of_bars * price_per_bar
def total_cost_with_discount := if number_of_bars > discount_threshold then total_cost_without_discount * (1 - discount_rate) else total_cost_without_discount

def revenue_first_600_bars := min number_of_bars 600 * price_first_600_bars
def revenue_remaining_bars := (number_of_bars - min number_of_bars 600) * price_remaining_bars
def total_revenue := revenue_first_600_bars + revenue_remaining_bars

def profit := total_revenue - total_cost_with_discount

-- Prove that the calculated profit is 480 dollars
theorem profit_is_480 : profit = 480 := by
  sorry

end profit_is_480_l419_419067


namespace hyperbola_eccentricity_is_sqrt_10_over_3_l419_419512

variable (a b : ℝ)

def is_hyperbola (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def is_perpendicular_to_line (b a : ℝ) : Prop :=
  b = a / 3

def hyper_eccentricity (a b : ℝ) : ℝ :=
  real.sqrt ((a^2 + b^2) / a^2)

theorem hyperbola_eccentricity_is_sqrt_10_over_3:
  ∀ a b : ℝ,
  is_hyperbola 0 0 a b → is_perpendicular_to_line b a →
  hyper_eccentricity a b = real.sqrt 10 / 3 :=
by
  intros
  sorry

end hyperbola_eccentricity_is_sqrt_10_over_3_l419_419512


namespace container_marbles_proportional_l419_419043

theorem container_marbles_proportional (V1 V2 : ℕ) (M1 M2 : ℕ)
(h1 : V1 = 24) (h2 : M1 = 75) (h3 : V2 = 72) (h4 : V1 * M2 = V2 * M1) :
  M2 = 225 :=
by {
  -- Given conditions
  sorry
}

end container_marbles_proportional_l419_419043


namespace inclination_angle_range_l419_419513

theorem inclination_angle_range (k : ℝ) (θ : ℝ) :
  (∀ x y : ℝ, (y = k * x - real.sqrt 3) ∧ (x + y = 3) ∧ (x > 0 ∧ y > 0) → θ = real.arctan k)
  → (π / 6 < θ ∧ θ < π / 2) :=
sorry

end inclination_angle_range_l419_419513


namespace sequence_general_formula_l419_419754

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3 * sequence (n - 1)

def sum_sequence (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence (i + 1)

theorem sequence_general_formula (n : ℕ) (h_pos : 0 < n) :
  let a_n := sequence n in
  let S_n := sum_sequence n in
  a_1 = 1 ∧ (∀ n ≥ 1, a_{n+1} = 2 * S_n + 1) → a_n = 3^(n-1) :=
by
  intros a_n S_n h_pos h
  sorry

end sequence_general_formula_l419_419754


namespace number_of_diagonals_in_octagon_l419_419523

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l419_419523


namespace circumcircle_area_of_isosceles_triangle_l419_419866

open Real

theorem circumcircle_area_of_isosceles_triangle:
  (∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
    (dist A B = 4) ∧ (dist A C = 4) ∧ (dist B C = 3) →
    (circle_area A B C = (256 / 13.75) * π)) :=
by sorry

end circumcircle_area_of_isosceles_triangle_l419_419866


namespace num_diagonals_octagon_l419_419574

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l419_419574


namespace max_value_sin_function_l419_419509

theorem max_value_sin_function (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2)
  (h₁ : ∀ x, -f(x) = f(-x)) : 
  ∃ x, f(x) = sin x * sin (x + 3 * θ) ∧ (∀ y, f(y) ≤ 1 / 2) :=
begin
  use sorry, -- place where x achieves the max value
  split,
  { exact sorry, -- check that f(x) = sin x * sin (x + 3 * θ)
  },
  { intro y,
    exact sorry, -- check that for any y, f(y) <= 1 / 2
  }
end

end max_value_sin_function_l419_419509


namespace certain_number_divided_by_two_l419_419628

theorem certain_number_divided_by_two (x : ℝ) (h : x / 2 + x + 2 = 62) : x = 40 :=
sorry

end certain_number_divided_by_two_l419_419628


namespace sequence_equality_l419_419963

theorem sequence_equality {n : ℕ} (h₁ : ∀ i, 1 ≤ i ∧ i ≤ n → 1 ≤ (a : ℕ) i ∧ (a : ℕ) i ≤ n)
  (h₂ : ∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → |(a : ℕ) i - (a : ℕ) j| = |i - j|) : 
  ∀ i, 1 ≤ i ∧ i ≤ n → (a i) = i :=
by
  sorry

end sequence_equality_l419_419963


namespace identify_f_l419_419216

noncomputable def f (x : ℝ) : ℝ := sorry

-- Given conditions
axiom domain_of_f : ∀ x, x ∈ ℝ
axiom additivity : ∀ m n : ℝ, f(m + n) = f(m) + f(n) - 6
axiom f_neg1 : 0 < f(-1) ∧ f(-1) ≤ 5
axiom positive_x : ∀ x : ℝ, x > -1 → f(x) > 0

-- To be proven
theorem identify_f : ∃ k : ℤ, 1 ≤ k ∧ k ≤ 5 ∧ ∀ x : ℝ, f(x) = k * x + 6 := sorry

end identify_f_l419_419216


namespace symmetry_axis_of_g_l419_419335

-- Define the initial function f
def f (x : ℝ) : ℝ := 2 * Real.sin ((2/3) * x + (3 * Real.pi / 4))

-- Define the transformed function g
def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x)

-- Lean statement to verify the correct symmetry axis of g
theorem symmetry_axis_of_g : Set.SymmetryAxis g (Real.pi / 2) :=
sorry

end symmetry_axis_of_g_l419_419335


namespace part1_average_decrease_rate_part2_unit_price_reduction_l419_419041

-- Part 1: Prove the average decrease rate is 10%
theorem part1_average_decrease_rate (p0 p2 : ℝ) (x : ℝ) 
    (h1 : p0 = 200) 
    (h2 : p2 = 162) 
    (hx : (1 - x)^2 = p2 / p0) : x = 0.1 :=
by {
    sorry
}

-- Part 2: Prove the unit price reduction should be 15 yuan
theorem part2_unit_price_reduction (p_sell p_factory profit : ℝ) (n_initial dn m : ℝ)
    (h3 : p_sell = 200)
    (h4 : p_factory = 162)
    (h5 : n_initial = 20)
    (h6 : dn = 10)
    (h7 : profit = 1150)
    (hx : (38 - m) * (n_initial + 2 * m) = profit) : m = 15 :=
by {
    sorry
}

end part1_average_decrease_rate_part2_unit_price_reduction_l419_419041


namespace total_rounds_proof_l419_419386

-- Define the initial token counts
def initial_tokens : (ℕ × ℕ × ℕ) := (15, 14, 13)

-- Define the game rule as a function that takes the token counts and returns the updated counts after one round
def game_rule : ℕ × ℕ × ℕ → ℕ × ℕ × ℕ
| (a, b, c) :=
  if a ≥ b ∧ a ≥ c then
    (a - 3, b + 1, c + 1)
  else if b ≥ a ∧ b ≥ c then
    (a + 1, b - 3, c + 1)
  else
    (a + 1, b + 1, c - 3)

-- Define the function to compute the total number of rounds
def total_rounds : ℕ := 37

-- Theorem stating the total number of rounds until a player runs out of tokens is 37
theorem total_rounds_proof :
  (∃ rounds: ℕ, rounds = total_rounds ∧
   (let final_counts := (list.repeat game_rule total_rounds).foldl (λ acc f, f acc) initial_tokens
    in final_counts.1 = 0 ∨ final_counts.2 = 0 ∨ final_counts.3 = 0)) := by
  sorry

end total_rounds_proof_l419_419386


namespace magnitude_of_product_l419_419119

-- Definitions of the complex numbers involved
def z1 : ℂ := 3 * real.sqrt 5 - 5 * complex.I
def z2 : ℂ := 2 * real.sqrt 2 + 4 * complex.I

-- The statement to be proven
theorem magnitude_of_product :
  complex.abs (z1 * z2) = real.sqrt 1680 :=
by
  sorry

end magnitude_of_product_l419_419119


namespace correct_propositions_l419_419051

def f1 (x : ℝ) : ℝ := if x > 0 then log x else 1
def g1 (x : ℝ) : ℝ := -2

def f2 (x : ℝ) : ℝ := x + sin x
def g2 (x : ℝ) : ℝ := x - 1

def f3 (x : ℝ) : ℝ := exp x
def g3 (a : ℝ) (x : ℝ) : ℝ := a * x

def f4 (x : ℝ) : ℝ := 2 * x
def g4 (x : ℝ) : ℝ := 2 * x - 1

def isSupportingFunction (f g : ℝ → ℝ) : Prop := ∀ x, f x ≥ g x

theorem correct_propositions :
  (¬ isSupportingFunction f1 g1) ∧
  (isSupportingFunction f2 g2) ∧
  (∀ a, (0 ≤ a ∧ a ≤ exp 1) → isSupportingFunction f3 (g3 a)) ∧
  (¬ ∀ f : ℝ → ℝ, (∀ x, f x ∈ set.univ) → ∃ (a b : ℝ), ∀ x, f x ≥ a * x + b) →
  true := 
by {
  sorry
}

end correct_propositions_l419_419051


namespace horner_method_operations_l419_419775

theorem horner_method_operations (x : ℕ) (hx : x = 5) :
  let f := λ x, 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1 in
  ∃ (multiplications additions : ℕ), multiplications = 5 ∧ additions = 5 :=
by
  let f := λ x, 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1
  use 5
  use 5
  exact ⟨rfl, rfl⟩

end horner_method_operations_l419_419775


namespace sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14_l419_419798

noncomputable def value_x : ℝ := real.sqrt (20 - 8 * real.sqrt 5) + real.sqrt (20 + 8 * real.sqrt 5)

theorem sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14 :
  value_x = 2 * real.sqrt 14 :=
by
  sorry

end sqrt_20_minus_8sqrt5_add_sqrt_20_plus_8sqrt5_eq_2sqrt14_l419_419798


namespace proof_problem_l419_419183

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem proof_problem (a b : ℝ) :
  f 2016 a b + f (-2016) a b + f' 2017 a b - f' (-2017) a b = 8 := by
  sorry

end proof_problem_l419_419183


namespace find_x_l419_419612

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end find_x_l419_419612


namespace meeting_time_l419_419086

/--
Betty, Charles, and Lisa all start jogging around a park at 6:00 AM. Betty completes a lap every 5 minutes, 
Charles every 8 minutes, and Lisa every 9 minutes, but after every two laps, Lisa takes a 3-minute break.
Prove that the earliest time when all three meet back at the starting point is 1:00 PM (420 minutes after 6:00 AM).
-/
def lap_time_betty := 5
def lap_time_charles := 8
def lap_time_lisa := 9
def break_lisa := 3

-- Define a function that calculates the least common multiple of two natural numbers.
noncomputable def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem meeting_time : 
    lcm (lcm lap_time_betty lap_time_charles) ((lap_time_lisa * 2 + break_lisa) / 2) = 420 :=
by 
    sorry

end meeting_time_l419_419086


namespace intersection_nonempty_iff_k_in_range_l419_419193

def M : set (ℝ × ℝ) := { p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ y = sqrt (2 * x - x^2) }

def N (k : ℝ) : set (ℝ × ℝ) := { p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ y = k * (x + 1) }

theorem intersection_nonempty_iff_k_in_range (k : ℝ) :
  (∃ p, p ∈ M ∧ p ∈ N k) ↔ 0 ≤ k ∧ k ≤ sqrt 3 / 3 :=
sorry

end intersection_nonempty_iff_k_in_range_l419_419193


namespace product_even_probability_l419_419147

theorem product_even_probability :
  let total_outcomes := 216
  let probability_geoff_odd := (3 / 6) * (3 / 6)
  let probability_geoff_even := 1 - probability_geoff_odd
  probability_geoff_even = 3 / 4 :=
by
  sorry

end product_even_probability_l419_419147


namespace area_of_circumscribed_circle_isosceles_triangle_l419_419852

theorem area_of_circumscribed_circle_isosceles_triangle :
  ∃ (r : ℝ), (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi) :=
by
  -- Consider the isosceles triangle conditions
  let a : ℝ := 4
  let b : ℝ := 4
  let c : ℝ := 3
  let BD := Real.sqrt(a^2 - (c/2)^2)
  let r := 8 / BD
  have h1 : BD = Real.sqrt 13.75 := by 
    -- Calculate the altitude BD
    calc
      BD = Real.sqrt(a^2 -  (c/2)^2) : rfl
      ... = Real.sqrt(16 - (3/2)^2) : rfl
      ... = Real.sqrt 13.75 : rfl
  
  use r
  have h2 : r = 8 / Real.sqrt 13.75 := by 
    -- Simplify the radius expression
    sorry

  have h3 : Real.pi * r ^ 2 = 256 / 55 * Real.pi := by 
    -- Calculate the area
    calc
      Real.pi * r ^ 2 = Real.pi * (8 / Real.sqrt 13.75) ^ 2 : by rw h2
      ... = Real.pi * (64 / 13.75) : by rw [pow_two, mul_div_assoc, mul_one, div_mul_div_same]
      ... = (256 / 54.6875) * Real.pi : by rw mul_comm
      ...   = (256 / 55) * Real.pi : by norm_num
    sorry
  
  show (r = 8 / Real.sqrt 13.75) ∧ (Real.pi * r ^ 2 = 256 / 55 * Real.pi),
  from ⟨h2, h3⟩
  sorry

end area_of_circumscribed_circle_isosceles_triangle_l419_419852


namespace smallest_k_cubed_sum_multiple_of_360_l419_419954

theorem smallest_k_cubed_sum_multiple_of_360 :
  ∃ k : ℕ, (1^3 + 2^3 + 3^3 + ... + k^3 = (k * (k + 1) / 2)^2) ∧ (k : ℕ) % 360 = 0 ∧ k = 38 :=
sorry

end smallest_k_cubed_sum_multiple_of_360_l419_419954


namespace distribute_students_l419_419113

theorem distribute_students : 
  (∀ (students : Fin 4), 
    (∀ (A B C : Fin 3), 
      (A < 1) ∧ (B ≥ 1) ∧ (C ≥ 1) ∧ (students 0 ≠ C) →
      fintype.card {s : set (Fin 4 × Fin 3) | ∀ x, s x.1 = x.2} = 12)) :=
sorry

end distribute_students_l419_419113


namespace sin_range_l419_419323

theorem sin_range : 
  (∀ y, (∃ x, x ∈ set.Icc (π / 6) (2 * π / 3) ∧ y = sin x) ↔ y ∈ set.Icc (1 / 2) 1) := 
by 
  sorry

end sin_range_l419_419323


namespace infinite_solutions_l419_419302

theorem infinite_solutions (m : ℕ) (x y z : ℤ) 
  (hx : x = 2^(5*(3*m+2))) 
  (hy : y = 2^(2*(3*m+2))) 
  (hz : z = 2^(10*m+7)) : 
  x^2 + y^5 = z^3 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 := 
begin
  sorry
end

end infinite_solutions_l419_419302


namespace sum_of_digits_base_7_of_2019_l419_419007

theorem sum_of_digits_base_7_of_2019 : 
  let base_10_number := 2019
  let base := 7
  let sum_of_digits := 15
in (∃ (digits : List ℕ), 
    (base_10_number = digits.foldr (λ (d n: ℕ), d + n * base) 0) ∧
    (∑ d in digits, d) = sum_of_digits) :=
sorry

end sum_of_digits_base_7_of_2019_l419_419007


namespace number_of_classmates_l419_419330

theorem number_of_classmates (n m : ℕ) (h₁ : n < 100) (h₂ : m = 9)
:(2 ^ 6 - 1) = 63 → 63 / m = 7 := by
  intros 
  sorry

end number_of_classmates_l419_419330


namespace no_absolute_winner_l419_419906

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end no_absolute_winner_l419_419906


namespace solution_l419_419661

open List

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def problem_statement : Prop :=
  ∃ (second_row : List ℕ), 
    second_row.perm [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧ 
    (∀ i, i ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] →
      is_perfect_square (i + nth second_row (i - 1)).getD 0)

theorem solution : problem_statement := 
  sorry

end solution_l419_419661


namespace values_with_max_occurrences_l419_419414

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 1)^2 / (x * (x^2 - 1))

theorem values_with_max_occurrences : 
 (set_of (λ a: ℝ, ∃ s: finset ℝ, (∀ x ∈ s, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) ∧ 
                  s.card = 2 ∧ ∀ x ∈ ({x : ℝ | f x = a} : set ℝ), x ∈ s)) = 
 {[a : ℝ | a < -4].union {[a : ℝ | a > 4]]} :=
sorry

end values_with_max_occurrences_l419_419414


namespace poles_inside_base_l419_419886

theorem poles_inside_base :
  ∃ n : ℕ, 2015 + n ≡ 0 [MOD 36] ∧ n = 1 :=
sorry

end poles_inside_base_l419_419886


namespace probability_distribution_l419_419717

def Player : Type := {R : Unit, S : Unit, T : Unit, U : Unit}

def initialMoney (p : Player) : ℕ := 1

noncomputable def transitionProbability : ℝ := 2 / 27

theorem probability_distribution (p : Player) (n : ℕ) (h : n = 1008) :
  ∃ (prob : ℝ), prob = (2 / 27) := by
  sorry

end probability_distribution_l419_419717


namespace circle_tangent_to_ellipse_l419_419771

theorem circle_tangent_to_ellipse {r : ℝ} 
  (h1: ∀ p: ℝ × ℝ, p ≠ (0, 0) → ((p.1 - r)^2 + p.2^2 = r^2 → p.1^2 + 4 * p.2^2 = 8))
  (h2: ∃ p: ℝ × ℝ, p ≠ (0, 0) ∧ ((p.1 - r)^2 + p.2^2 = r^2 ∧ p.1^2 + 4 * p.2^2 = 8)):
  r = Real.sqrt (3 / 2) :=
by
  sorry

end circle_tangent_to_ellipse_l419_419771


namespace maximum_annual_profit_l419_419076

def annual_profit (x : ℕ) : ℝ :=
  if 0 < x ∧ x < 80 then
    - (1/2) * x^2 + 60 * x - 500
  else
    - x - 8100 / x + 1680

theorem maximum_annual_profit :
  ∃ (x : ℕ), x = 90 ∧ annual_profit x = 1500 :=
by
  existsi 90
  split
  sorry
  sorry

end maximum_annual_profit_l419_419076


namespace solve_cubic_root_eq_l419_419451

theorem solve_cubic_root_eq (x : ℝ) : (real.cbrt (5 - x / 3) = -4) → x = 207 :=
by
  sorry

end solve_cubic_root_eq_l419_419451


namespace diagonals_of_octagon_l419_419549

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l419_419549


namespace series_diverges_l419_419666

noncomputable def term (n : ℕ) : ℝ := (n + 1)! / (n * 10^n)

theorem series_diverges : ¬ ∃ l : ℝ, tendsto (λ n, (∑ i in range n, term i)) atTop (𝓝 l) := 
by
  sorry

end series_diverges_l419_419666


namespace subtended_angle_sine_l419_419223

-- Define the parameters
def radius : ℝ := 10
def chord_PQ_length : ℝ := 24
def segment_PT_QT_length : ℝ := 12

-- Define the relevant points and segments
axiom exists_points (P Q R S T : Type) 
  (circle : ∀ x : P, ∀ y : P, (x = y) → False) -- Placeholder representation of the circle
  (intersection_point : P → P → P → Prop) -- Definition of intersection for segments

-- Define a theorem to assert the conditions
theorem subtended_angle_sine (P Q R S T : Type)
  (circle : ∀ x : P, ∀ y : P, (x = y) → False)
  (center O : P)
  (radius_OP : O → P → ℝ := (λ O P, radius))
  (intersection : intersection_point P Q T)
  (chord_PQ_length_given : ∀ (P Q : P), O → O → O → ℝ) 
  (midpoint_T : O → O → O → O → Prop) -- Definition indicating midpoint
  (RS_unique_bisector : ∀ (T : P), midpoint_T P T T Q → midpoint_T R S S T → P = P = False)
  (segment_PT_QT : O → O → ℝ := segment_PT_QT_length) : 
  sin (angle O P R) = 1/2 := 
sorry

end subtended_angle_sine_l419_419223


namespace cyclic_sum_inequality_l419_419630

theorem cyclic_sum_inequality (x y z : ℝ) (hp : x > 0 ∧ y > 0 ∧ z > 0) (h : x + y + z = 3) : 
  (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) < (3 + x * y + y * z + z * x) := by
  sorry

end cyclic_sum_inequality_l419_419630


namespace city_division_exists_l419_419642

-- Define the problem conditions and prove the required statement
theorem city_division_exists (squares : Type) (streets : squares → squares → Prop)
  (h_outgoing: ∀ (s : squares), ∃ t u : squares, streets s t ∧ streets s u) :
  ∃ (districts : squares → ℕ), (∀ (s t : squares), districts s ≠ districts t → streets s t ∨ streets t s) ∧
  (∀ (i j : ℕ), i ≠ j → ∀ (s t : squares), districts s = i → districts t = j → streets s t ∨ streets t s) ∧
  (∃ m : ℕ, m = 1014) :=
sorry

end city_division_exists_l419_419642


namespace perpendicular_OP_AB_l419_419272

variables {P A B C D X O : Type}

-- Definitions based on conditions
def point_inside_quadrilateral (P : Type) (A B C D : Type) : Prop :=
  sorry -- Definition of P being inside quadrilateral ABCD

def circumcircles_congruent (P A B C D : Type) : Prop :=
  sorry -- Definition that circumcircles of triangles PDA, PAB, and PBC are pairwise distinct but congruent

def lines_intersect_at (AD BC : Type) (X : Type) : Prop :=
  sorry -- Definition of lines AD and BC intersecting at point X

def circumcenter (X C D O : Type) : Prop :=
  sorry -- Definition that O is the circumcenter of triangle XCD

-- Main theorem statement
theorem perpendicular_OP_AB
  (P A B C D X O : Type)
  (h1 : point_inside_quadrilateral P A B C D)
  (h2 : circumcircles_congruent P A B C D)
  (h3 : lines_intersect_at A D B C X)
  (h4 : circumcenter X C D O) :
  perp P O A B :=
begin
  sorry -- Proof goes here
end

end perpendicular_OP_AB_l419_419272


namespace substitution_preserves_range_l419_419944

-- Definitions for the problem conditions
def f (x a b : ℝ) : ℝ := 3 * x^2 + a * x + b

def g1 (t : ℝ) : ℝ := log (1 / 2) t
def g2 (t : ℝ) : ℝ := (1 / 2)^t
def g3 (t : ℝ) : ℝ := (t - 1)^2
def g4 (t : ℝ) : ℝ := cos t

-- Statement of the proof problem
theorem substitution_preserves_range (a b : ℝ) :
  (∀ y : ℝ, ∃ t : ℝ, y = g2 t) :=
sorry

end substitution_preserves_range_l419_419944


namespace cos_angle_GAC_in_cube_l419_419372

theorem cos_angle_GAC_in_cube (A B C D E F G H : Point) (s : ℝ) (h1 : s = 1)
  (h_cube : cube A B C D E F G H s) :
  cos (angle G A C) = (Real.sqrt 3) / 3 :=
sorry

end cos_angle_GAC_in_cube_l419_419372


namespace circumcircle_area_of_isosceles_triangle_l419_419872

open Real

theorem circumcircle_area_of_isosceles_triangle:
  (∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
    (dist A B = 4) ∧ (dist A C = 4) ∧ (dist B C = 3) →
    (circle_area A B C = (256 / 13.75) * π)) :=
by sorry

end circumcircle_area_of_isosceles_triangle_l419_419872


namespace smallest_delicious_integer_is_minus_2022_l419_419108

def smallest_delicious_integer (sum_target : ℤ) : ℤ :=
  -2022

theorem smallest_delicious_integer_is_minus_2022
  (B : ℤ)
  (h : ∃ (s : List ℤ), s.sum = 2023 ∧ B ∈ s) :
  B = -2022 :=
sorry

end smallest_delicious_integer_is_minus_2022_l419_419108


namespace holiday_customers_l419_419029

theorem holiday_customers (people_per_hour : ℕ) (hours : ℕ) (regular_rate : people_per_hour = 175)
  (holiday_rate : people_per_hour * 2 = 2 * 175) : 
  people_per_hour * 2 * hours = 2800 := 
by
  have h1 : people_per_hour = 175 := regular_rate
  have h2 : people_per_hour * 2 = 350 := by rw [h1, holiday_rate]
  have h3 : 350 * hours = 2800 := by
    norm_num at *
    exact Eq.refl 2800
  exact h3

end holiday_customers_l419_419029


namespace solve_cubic_root_eq_l419_419452

theorem solve_cubic_root_eq (x : ℝ) : (real.cbrt (5 - x / 3) = -4) → x = 207 :=
by
  sorry

end solve_cubic_root_eq_l419_419452


namespace find_second_game_points_l419_419431

-- Define Clayton's points for respective games
def first_game_points := 10
def third_game_points := 6

-- Define the points in the second game as P
variable (P : ℕ)

-- Define the points in the fourth game based on the average of first three games
def fourth_game_points := (first_game_points + P + third_game_points) / 3

-- Define the total points over four games
def total_points := first_game_points + P + third_game_points + fourth_game_points

-- Based on the total points, prove P = 14
theorem find_second_game_points (P : ℕ) (h : total_points P = 40) : P = 14 :=
  by
    sorry

end find_second_game_points_l419_419431
