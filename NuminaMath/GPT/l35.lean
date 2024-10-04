import Mathlib

namespace center_of_symmetry_f_l35_35200

noncomputable def f : ℝ → ℝ := λ x, (1/2) * Real.tan (5 * x + (Real.pi / 4))

theorem center_of_symmetry_f (k : ℤ) : ∃ x : ℝ, ∃ y : ℝ, y = 0 ∧ x = (k * Real.pi / 10) - (Real.pi / 20) :=
  by sorry

end center_of_symmetry_f_l35_35200


namespace find_angle_APB_l35_35320

open Real
open EuclideanGeometry

def PA_tangent_to_semicircle_SAR (P A S R : Point) : Prop := sorry
def PB_tangent_to_semicircle_RBT (P B R T : Point) : Prop := sorry
def SRT_is_straight_line (S R T : Point) : Prop := collinear {S, R, T}
def arc_measures_70_deg (A S : Point) (r : ℝ) : Prop := sorry
def arc_measures_50_deg (B T : Point) (r : ℝ) : Prop := sorry

theorem find_angle_APB (P A B S R T : Point) (r1 r2 : ℝ) 
  (h1 : PA_tangent_to_semicircle_SAR P A S R)
  (h2 : PB_tangent_to_semicircle_RBT P B R T)
  (h3 : SRT_is_straight_line S R T)
  (h4 : arc_measures_70_deg A S r1)
  (h5 : arc_measures_50_deg B T r2) : 
  ∠APB = 120 :=
sorry

end find_angle_APB_l35_35320


namespace find_coordinates_Q_l35_35723

def P : ℝ × ℝ := (2, 1)

def is_parallel_to_x_axis (P Q : ℝ × ℝ) : Prop :=
  P.2 = Q.2

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem find_coordinates_Q (Q : ℝ × ℝ) :
  is_parallel_to_x_axis P Q ∧ distance P Q = 3 → 
  (Q = (5, 1) ∨ Q = (-1, 1)) :=
begin
  sorry
end

end find_coordinates_Q_l35_35723


namespace square_of_repeating_decimal_l35_35555

theorem square_of_repeating_decimal :
  let x := (1 / 3 : ℚ) in
  let student_result := (0.3 * 0.3 : ℚ) in
  let actual_result := x * x in
  actual_result = 0.1111111111 :=
by
  sorry

end square_of_repeating_decimal_l35_35555


namespace antifreeze_replacement_percentage_l35_35895

/-- Conditions and given data:
1. The mixture in the car is 10% antifreeze.
2. The radiator contains 4 gallons of fluid.
3. We need the final mixture to be 50% antifreeze.
4. We are to drain 2.2857 gallons from the radiator.

Question: What is the percentage of antifreeze in the replacement mixture?
Answer: The percentage of antifreeze in the replacement mixture should be approximately 80%. 
-/

theorem antifreeze_replacement_percentage :
  ∀ (initial_percentage: ℝ) (total_fluid: ℝ) (drained_fluid: ℝ) (desired_percentage: ℝ),
    initial_percentage = 0.10 →
    total_fluid = 4 →
    drained_fluid = 2.2857 →
    desired_percentage = 0.50 →
    ((desired_percentage * total_fluid) - ((total_fluid - drained_fluid) * initial_percentage))
    / drained_fluid = 0.80 := 
by
  intros initial_percentage total_fluid drained_fluid desired_percentage
  assume h1 h2 h3 h4
  -- Proof omitted
  sorry

end antifreeze_replacement_percentage_l35_35895


namespace shadow_length_of_flagpole_is_correct_l35_35535

noncomputable def length_of_shadow_flagpole : ℕ :=
  let h_flagpole : ℕ := 18
  let shadow_building : ℕ := 60
  let h_building : ℕ := 24
  let similar_conditions : Prop := true
  45

theorem shadow_length_of_flagpole_is_correct :
  length_of_shadow_flagpole = 45 := by
  sorry

end shadow_length_of_flagpole_is_correct_l35_35535


namespace pizza_ordering_l35_35626

theorem pizza_ordering :
  ∀ (Alex Beth Cyril Eliza Dan : ℚ),
  Alex = 1/6 →
  Beth = 1/4 →
  Cyril = 1/3 →
  Dan = 0 →
  Eliza = 1 - (Alex + Beth + Cyril + Dan) →
  [Cyril, Beth, Eliza, Alex, Dan].sorted (≥) := 
by
  assume Alex Beth Cyril Eliza Dan hAlex hBeth hCyril hDan hEliza
  sorry

end pizza_ordering_l35_35626


namespace cranberries_left_l35_35173

def initial_cranberries : ℕ := 60000
def harvested_by_humans : ℕ := initial_cranberries * 40 / 100
def eaten_by_elk : ℕ := 20000

theorem cranberries_left (c : ℕ) : c = initial_cranberries - harvested_by_humans - eaten_by_elk → c = 16000 := by
  sorry

end cranberries_left_l35_35173


namespace num_of_sets_l35_35286

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B : Set ℕ := {x | 0 < x ∧ x < 5}

lemma count_sets (C : Set ℕ) : A ⊆ C → C ⊆ B → (C = {1, 2} ∨ C = {1, 2, 3} ∨ C = {1, 2, 4} ∨ C = {1, 2, 3, 4}) :=
by sorry

theorem num_of_sets : (∃ (C : Set ℕ), A ⊆ C ∧ C ⊆ B) ↔ 4 :=
by {
  have h1 : A = {1, 2},
  { ext, split; intro h,
    { simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff] at h,
      rcases h with rfl | rfl;
      simp },
    { simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff],
      intro hx, fin_cases hx; simp },
  },
  have h2 : B = {1, 2, 3, 4},
  { ext, split; intro h,
    { simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff] at h,
      finish },
    { simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff],
      finish },
  },
  use 4,
  split,
  { intro h,
    cases h with C hC,
    have hC' := count_sets C, exact sorry },
  { intro h, sorry }
}

end num_of_sets_l35_35286


namespace sin_y_eq_neg_one_l35_35732

noncomputable def α := Real.arccos (-1 / 5)

theorem sin_y_eq_neg_one (x y z : ℝ) (h1 : x = y - α) (h2 : z = y + α)
  (h3 : (2 + Real.sin x) * (2 + Real.sin z) = (2 + Real.sin y) ^ 2) : Real.sin y = -1 :=
sorry

end sin_y_eq_neg_one_l35_35732


namespace arith_seq_ratio_l35_35652

-- Let aₙ be an arithmetic sequence with a first term a₁ and a common difference d
def arith_seq (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d
-- The sum of the first n terms of the arithmetic sequence
def sum_arith_seq (a1 d : ℝ) (n : ℕ) : ℝ := n * a1 + (n * (n - 1) * d) / 2

theorem arith_seq_ratio 
  (a1 d : ℝ)
  (h : (5 * a1 + 10 * d) / (3 * a1 + 3 * d) = 3) : 
  (arith_seq a1 d 5 / arith_seq a1 d 3) = 17/9 :=
by
  sorry

end arith_seq_ratio_l35_35652


namespace find_perpendicular_line_through_point_l35_35617

theorem find_perpendicular_line_through_point (A : ℝ × ℝ) (L : ℝ → ℝ → Prop) : 
  (A = (1, 2)) →
  (∀ x y, L x y ↔ x + 2 * y = 1) →
  ∃ k b, (∀ x y, y = k * x + b ↔ 2 * x - y = 0) ∧
         (∀ x y, y = k * x + b → y - 2 = k * (x - 1)) ∧
         ∃ x y, L x y ∧ y = k * x + b :=
  by
  intros hA hL
  use [2, 0]  -- slope k = 2 and y-intercept b = 0 for 2x - y = 0
  split
  { intros x y
    split
    { intros hxy
      exact sorry  -- proof that 2x - y = 0 is equivalent to y = 2x
    }
    { intros hxy
      exact sorry  -- proof that y = 2x implies 2x - y = 0
    }
  }
  split
  { intros x y hxy
    exact sorry  -- point-slope form verification
  }
  { use (1, 2), 1 + 2 * 2 -- point on L and verifying y = kx + b
    exact sorry  -- proof to show a point on L is also on the desired line
  }

end find_perpendicular_line_through_point_l35_35617


namespace expression_evaluation_l35_35640

-- Define the variables and the given condition
variables (x y : ℝ)

-- Define the equation condition
def equation_condition : Prop := x - 3 * y = 4

-- State the theorem
theorem expression_evaluation (h : equation_condition x y) : 15 * y - 5 * x + 6 = -14 :=
by
  sorry

end expression_evaluation_l35_35640


namespace foci_distance_l35_35983

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 - 32 * y = 144

-- State the problem to prove
theorem foci_distance :
  ∀ x y : ℝ, hyperbola_eq x y → distance_between_foci (9 * x^2 - 18 * x - 16 * y^2 - 32 * y = 144) = (sqrt 3425) / 72 :=
by
  sorry

end foci_distance_l35_35983


namespace intersection_lines_l35_35304

theorem intersection_lines (a : ℝ) :
  (∃ p : ℝ × ℝ, 
    (∃ x y, p = (x, y) ∧ (l₁ : a * x + 2 * y + 6 = 0) ∧ 
    (l₂ : x + y - 4 = 0) ∧ (l₃ : 2 * x - y + 1 = 0))) → a = -12 :=
by
  sorry

end intersection_lines_l35_35304


namespace smallest_possible_d_l35_35844

theorem smallest_possible_d : 
  ∃ d, 
  (∀ (d : ℕ), d ≥ 1) ∧  -- To ensure d is a natural number and at least 1 (trivial condition).
  (∃ l : List ℕ, l.length = 99 ∧ l.nodup) ∧  -- List of 99 distinct numbers.
  ((l' : List (ℕ × ℕ)) , l'.length = 4851 ∧ ∀ {a b}, a ≠ b → (a, b) ∈ l' → a ≠ b → |a - b| = d) ∧ -- 4851 differences
  ((l1 : List (ℕ × ℕ)), l1.count (1, 1) = 85) -- Number 1 appears 85 times
  → 
  d = 7 := 
sorry

end smallest_possible_d_l35_35844


namespace teresa_age_at_michiko_birth_l35_35802

noncomputable def Teresa_age_now : ℕ := 59
noncomputable def Morio_age_now : ℕ := 71
noncomputable def Morio_age_at_Michiko_birth : ℕ := 38

theorem teresa_age_at_michiko_birth :
  (Teresa_age_now - (Morio_age_now - Morio_age_at_Michiko_birth)) = 26 := 
by
  sorry

end teresa_age_at_michiko_birth_l35_35802


namespace fraction_of_garden_occupied_by_flowerbeds_is_correct_l35_35900

noncomputable def garden_fraction_occupied : ℚ :=
  let garden_length := 28
  let garden_shorter_length := 18
  let triangle_leg := (garden_length - garden_shorter_length) / 2
  let triangle_area := 1 / 2 * triangle_leg^2
  let flowerbeds_area := 2 * triangle_area
  let garden_width : ℚ := 5  -- Assuming the height of the trapezoid as part of the garden rest
  let garden_area := garden_length * garden_width
  flowerbeds_area / garden_area

theorem fraction_of_garden_occupied_by_flowerbeds_is_correct :
  garden_fraction_occupied = 5 / 28 := by
  sorry

end fraction_of_garden_occupied_by_flowerbeds_is_correct_l35_35900


namespace intersection_on_circumcircle_l35_35756

variable {P Q M N : Type}
variable (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C]

-- Conditions
axiom condition1 : (P ∈ A) (Q ∈ B) (M ∈ P) (N ∈ Q) 
axiom condition2 : ∠APB = ∠BCA
axiom condition3 : ∠CAQ = ∠ABC
axiom condition4 : IsMidpoint P (A, M)
axiom condition5 : IsMidpoint Q (A, N)

-- Prove
theorem intersection_on_circumcircle : Intersection (Line B M) (Line C N) ∈ Circumcircle A B C :=
sorry

end intersection_on_circumcircle_l35_35756


namespace prime_factors_sum_l35_35842

theorem prime_factors_sum (x y : ℕ) (h1 : log 10 x + 2 * log 10 (Nat.gcd x y) = 60) 
                             (h2 : log 10 y + 2 * log 10 (Nat.lcm x y) = 570) 
                             (hx_pos : 0 < x) (hy_pos : 0 < y) : 
    let m := Nat.card (Nat.factors x),
        n := Nat.card (Nat.factors y) in 3 * m + 2 * n = 880 := 

sorry

end prime_factors_sum_l35_35842


namespace tye_bills_l35_35072

theorem tye_bills : 
  ∀ (total_amount withdrawn_money_per_bank bill_value number_of_banks: ℕ), 
  withdrawn_money_per_bank = 300 → 
  bill_value = 20 → 
  number_of_banks = 2 → 
  total_amount = withdrawn_money_per_bank * number_of_banks → 
  (total_amount / bill_value) = 30 :=
by
  intros total_amount withdrawn_money_per_bank bill_value number_of_banks 
  intro h_withdrawn_eq_300
  intro h_bill_eq_20
  intro h_banks_eq_2
  intro h_total_eq_mult
  have h_total_eq_600 : total_amount = 600 := by rw [h_total_eq_mult, h_withdrawn_eq_300, h_banks_eq_2]; norm_num
  have h_bills_eq_30 := h_total_eq_600.symm ▸ div_eq_of_eq_mul_left (ne_of_gt (by norm_num : 0 < 20)) (by norm_num : 600 = 20 * 30)
  exact h_bills_eq_30
  sorry

end tye_bills_l35_35072


namespace calculate_n_l35_35936

theorem calculate_n (n : ℤ) (h : 2^n = 2 * (16^2 : ℤ) * (64^3 : ℤ)) : n = 27 :=
by 
sorry

end calculate_n_l35_35936


namespace solve_inequality_l35_35981

open Real

theorem solve_inequality (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4) ↔ 
  (x ∈ Set.Ioo (-∞) (-2) ∪ Set.Ioo (-1) 0 ∪ Set.Ioo 2 ∞) := 
by
  sorry

end solve_inequality_l35_35981


namespace range_of_abs_z3_l35_35727

open Complex

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry

-- Hypotheses
def abs_z1_eq_sqrt2 : |z1| = Real.sqrt 2 := sorry
def abs_z2_eq_sqrt2 : |z2| = Real.sqrt 2 := sorry
def orthogonal_OZ1_OZ2 : re (z1 * conj z2) = 0 := sorry
def abs_z1_plus_z2_minus_z3_eq_1 : |z1 + z2 - z3| = 1 := sorry

theorem range_of_abs_z3 :
  1 ≤ |z3| ∧ |z3| ≤ 3 := sorry

end range_of_abs_z3_l35_35727


namespace proof_num_solutions_l35_35297

noncomputable def num_solutions : ℕ :=
  let condition1 (x y : ℝ) : Prop := 2 * x + 5 * y = 5
  let condition2 (x y : ℝ) : Prop := abs (abs x - 2 * abs y) = 2
  let satisfies (x y : ℝ) : Prop := condition1 x y ∧ condition2 x y
  set.count (λ (p : ℝ × ℝ), satisfies p.1 p.2)

theorem proof_num_solutions : num_solutions = 2 :=
by
  sorry

end proof_num_solutions_l35_35297


namespace exists_pos_int_K_l35_35745

theorem exists_pos_int_K
  (n : ℕ)
  (x : Fin n → ℝ)
  (X : ℝ)
  (h_mean : (1 / n) * (∑ i in Finset.range n, x i) = X) :
  ∃ K : ℕ, K ≤ n ∧ ∀ i : ℕ, 0 ≤ i ∧ i < K → (1 / (K - i)) * (∑ j in Finset.range (K - i), x (i + 1 + j)) ≤ X :=
sorry

end exists_pos_int_K_l35_35745


namespace largest_prime_factor_of_binomial_l35_35093

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_prime_factor_of_binomial {p : ℕ} (hp : Nat.Prime p) (hp_range : 10 ≤ p ∧ p < 100) :
  p ∣ binomial 300 150 → p = 97 :=
by
suffices ∀ q : ℕ, Nat.Prime q → 10 ≤ q ∧ q < 100 → q ∣ binomial 300 150 → q ≤ 97
from fun h => le_antisymm (this p hp hp_range h) (le_of_eq (rfl : 97 = 97))
intro q hq hq_range hq_div
sorry

end largest_prime_factor_of_binomial_l35_35093


namespace sphere_surface_area_l35_35423

theorem sphere_surface_area (R : ℝ) (h : (4 / 3) * π * R^3 = (32 / 3) * π) : 4 * π * R^2 = 16 * π :=
sorry

end sphere_surface_area_l35_35423


namespace figure_perimeter_l35_35807

theorem figure_perimeter (area_total : ℕ) (n_squares : ℕ) (rows : ℕ) (columns : ℕ) 
  (h₁ : area_total = 150) (h₂ : n_squares = 6) (h₃ : rows = 2) (h₄ : columns = 3) :
  let area_square := area_total / n_squares in
  let side_square := nat.sqrt area_square in
  let perimeter := 2 * (rows * side_square + columns * side_square) in
  perimeter = 40 :=
by
  sorry

end figure_perimeter_l35_35807


namespace cranberries_left_l35_35168

theorem cranberries_left (total_cranberries : ℕ) (harvested_percent: ℝ) (cranberries_eaten : ℕ) 
  (h1 : total_cranberries = 60000) 
  (h2 : harvested_percent = 0.40) 
  (h3 : cranberries_eaten = 20000) : 
  total_cranberries - (harvested_percent * total_cranberries).to_nat - cranberries_eaten = 16000 := 
by 
  sorry

end cranberries_left_l35_35168


namespace hyperbola_parabola_intersection_l35_35243

def parabola_eq (y x : ℝ) : ℝ := y^2 - 4 * x

theorem hyperbola_parabola_intersection
    (a b p : ℝ)
    (hyp_hyperbola : ∀ (x y : ℝ), (x^2) / (a^2) - (y^2) / (b^2) = 1)
    (eccentricity_eq : b / a = sqrt 3)
    (hyp_parabola_directrix : ∀ (x y : ℝ), y^2 = 2 * p * x)
    (area_triangle : sqrt 3) :
    p = 2 ∧ ∀ y x, parabola_eq y x = 0 :=
by
  sorry

end hyperbola_parabola_intersection_l35_35243


namespace cost_of_six_hotdogs_and_seven_burgers_l35_35336

theorem cost_of_six_hotdogs_and_seven_burgers :
  ∀ (h b : ℝ), 4 * h + 5 * b = 3.75 → 5 * h + 3 * b = 3.45 → 6 * h + 7 * b = 5.43 :=
by
  intros h b h_eqn b_eqn
  sorry

end cost_of_six_hotdogs_and_seven_burgers_l35_35336


namespace minimum_omega_for_maximums_minimum_interval_for_solutions_l35_35242

theorem minimum_omega_for_maximums :
  ∃ ω: ℝ, (ω > 0 ∧ (∀ x: ℝ, (0 ≤ x ∧ x ≤ 1) → (∃ t: ℝ, t < ω ∧ 2* Real.sin (ω * x + Real.pi / 6) = t))) ↔ ω = 55 * Real.pi / 3 :=
by
  sorry

theorem minimum_interval_for_solutions :
  ∃ (m n: ℝ), (m < n ∧ ∀ x: ℝ, (m ≤ x ∧ x ≤ n) → (2 * Real.sin(2*x + Real.pi / 3) = -1)) ↔ n - m = 28 * Real.pi / 3 :=
by 
  sorry

end minimum_omega_for_maximums_minimum_interval_for_solutions_l35_35242


namespace uniformColorGridPossible_l35_35718

noncomputable def canPaintUniformColor (n : Nat) (G : Matrix (Fin n) (Fin n) (Fin (n - 1))) : Prop :=
  ∀ (row : Fin n), ∃ (c : Fin (n - 1)), ∀ (col : Fin n), G row col = c

theorem uniformColorGridPossible (n : Nat) (G : Matrix (Fin n) (Fin n) (Fin (n - 1))) :
  (∀ r : Fin n, ∃ c₁ c₂ : Fin n, c₁ ≠ c₂ ∧ G r c₁ = G r c₂) ∧
  (∀ c : Fin n, ∃ r₁ r₂ : Fin n, r₁ ≠ r₂ ∧ G r₁ c = G r₂ c) →
  ∃ c : Fin (n - 1), ∀ (row col : Fin n), G row col = c := by
  sorry

end uniformColorGridPossible_l35_35718


namespace fill_6x6_square_with_tiles_l35_35121

-- Definitions for the problem conditions
def is_L_shaped_tile (t : Tile) : Prop := sorry
def is_rectangular_tile (t : Tile) : Prop := sorry
def is_6x6_square (s : Square) : Prop := sorry

-- The main theorem statement representing the proof problem
theorem fill_6x6_square_with_tiles (k : ℕ) (s : Square) (t : Tile) : 
  (is_6x6_square s) → 
  (k ∈ {2, 4, 5, 6, 7, 8, 9, 10, 11, 12}) → 
  (count_tiles t s = 12) → 
  (count_L_shaped_tiles t s = k) → 
  (count_rectangular_tiles t s = 12 - k) → 
  ∃ (tiling : Tiling), fills_square tiling s :=
sorry

end fill_6x6_square_with_tiles_l35_35121


namespace cranberries_left_in_bog_l35_35165

theorem cranberries_left_in_bog
  (total_cranberries : ℕ)
  (percentage_harvested_by_humans : ℕ)
  (cranberries_eaten_by_elk : ℕ)
  (initial_count : total_cranberries = 60000)
  (harvested_percentage : percentage_harvested_by_humans = 40)
  (eaten_by_elk : cranberries_eaten_by_elk = 20000) :
  total_cranberries - (total_cranberries * percentage_harvested_by_humans / 100) - cranberries_eaten_by_elk = 16000 :=
by
  sorry

end cranberries_left_in_bog_l35_35165


namespace largest_two_digit_prime_factor_of_binom_300_150_l35_35081

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

def is_prime (p : ℕ) : Prop :=
  nat.prime p

def is_two_digit (p : ℕ) : Prop :=
  10 ≤ p ∧ p < 100

def less_than_300_divided_by_three (p : ℕ) : Prop :=
  3 * p < 300

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  let n := binomial_coefficient 300 150 in
  ∃ p, is_prime p ∧ is_two_digit p ∧ less_than_300_divided_by_three p ∧
        ∀ q, is_prime q ∧ is_two_digit q ∧ less_than_300_divided_by_three q → q ≤ p ∧ n % p = 0 ∧ p = 97 :=
by
  sorry

end largest_two_digit_prime_factor_of_binom_300_150_l35_35081


namespace remaining_area_of_cloth_l35_35556

theorem remaining_area_of_cloth (original_length : ℝ) (trim1 trim2 : ℝ) (final_length final_width remaining_area : ℝ) :
  original_length = 18 →
  trim1 = 4 →
  trim2 = 3 →
  final_length = original_length - trim1 →
  final_width = original_length - trim2 →
  remaining_area = final_length * final_width →
  remaining_area = 210 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3] at h4
  rw [h1, h3] at h5
  rw [h4, h5] at h6
  sorry

end remaining_area_of_cloth_l35_35556


namespace number_of_people_in_group_l35_35395

theorem number_of_people_in_group 
    (N : ℕ)
    (old_person_weight : ℕ) (new_person_weight : ℕ)
    (average_weight_increase : ℕ) :
    old_person_weight = 70 →
    new_person_weight = 94 →
    average_weight_increase = 3 →
    N * average_weight_increase = new_person_weight - old_person_weight →
    N = 8 :=
by
  sorry

end number_of_people_in_group_l35_35395


namespace student_receives_gbp_l35_35913

def exchange_rate (eur_to_gbp : ℝ) := eur_to_gbp = 0.85
def exchange_fee (fee_percentage : ℝ) := fee_percentage = 0.05
def amount_to_exchange (amount_eur : ℝ) := amount_eur = 100
def effective_amount (amount_eur fee_percentage : ℝ) := amount_eur * (1 - fee_percentage)
def final_amount (effective_amount eur_to_gbp : ℝ) := effective_amount * eur_to_gbp

theorem student_receives_gbp :
  ∀ (eur_to_gbp fee_percentage amount_eur : ℝ),
    exchange_rate eur_to_gbp →
    exchange_fee fee_percentage →
    amount_to_exchange amount_eur →
    final_amount (effective_amount amount_eur fee_percentage) eur_to_gbp = 80.75 :=
by
  intros eur_to_gbp fee_percentage amount_eur h_eur_to_gbp h_fee h_amount_eur
  rw [exchange_rate, exchange_fee, amount_to_exchange] at h_eur_to_gbp h_fee h_amount_eur
  sorry

end student_receives_gbp_l35_35913


namespace pentagon_angles_l35_35164

def is_point_in_convex_pentagon (O A B C D E : Point) : Prop := sorry
def angle (A B C : Point) : ℝ := sorry -- Assume definition of angle in radians

theorem pentagon_angles (O A B C D E: Point) (hO : is_point_in_convex_pentagon O A B C D E)
  (h1: angle A O B = angle B O C) (h2: angle B O C = angle C O D)
  (h3: angle C O D = angle D O E) (h4: angle D O E = angle E O A) :
  (angle E O A = angle A O B) ∨ (angle E O A + angle A O B = π) :=
sorry

end pentagon_angles_l35_35164


namespace find_correct_statements_l35_35828

open_locale classical

variable {V : Type*} [add_comm_group V] [vector_space ℝ V]

variables (A B C : V)
variables (zero_vec : V)
variables (zero_scalar : ℝ)

-- Definitions from the problem (conditions).
def statement_1 : Prop := A + -A = 0
def statement_2 : Prop := (C - A) + A = C
def statement_3 : Prop := A - C = C - A
def statement_4 : Prop := zero_scalar • A = zero_vec

-- theorem
theorem find_correct_statements (h1 : statement_1) (h2 : statement_2) : (¬ statement_3) ∧ (¬ statement_4) → 2 = 2 := 
begin
  sorry
end

end find_correct_statements_l35_35828


namespace minimize_transportation_cost_l35_35885

noncomputable def transportation_cost (x : ℝ) (distance : ℝ) (k : ℝ) (other_expense : ℝ) : ℝ :=
  k * (x * distance / x^2 + other_expense * distance / x)

theorem minimize_transportation_cost :
  ∀ (distance : ℝ) (max_speed : ℝ) (k : ℝ) (other_expense : ℝ) (x : ℝ),
  0 < x ∧ x ≤ max_speed ∧ max_speed = 50 ∧ distance = 300 ∧ k = 0.5 ∧ other_expense = 800 →
  transportation_cost x distance k other_expense = 150 * (x + 1600 / x) ∧
  (∀ y, (0 < y ∧ y ≤ max_speed) → transportation_cost y distance k other_expense ≥ 12000) ∧
  (transportation_cost 40 distance k other_expense = 12000)
  := 
  by intros distance max_speed k other_expense x H;
     sorry

end minimize_transportation_cost_l35_35885


namespace matrix_singular_and_zero_inverse_l35_35618

/-
Matrix under consideration:
\[
\begin{pmatrix} 8 & -6 \\ -4 & 3 \end{pmatrix}
\]

Prove that this matrix is singular.
-/

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![8, -6],
    ![-4, 3]]

theorem matrix_singular_and_zero_inverse :
  det A = 0 ∧ (A⁻¹ = ![![0, 0], ![0, 0]]) := by
  -- The proof will go here
  sorry

end matrix_singular_and_zero_inverse_l35_35618


namespace cranberries_left_in_bog_l35_35166

theorem cranberries_left_in_bog
  (total_cranberries : ℕ)
  (percentage_harvested_by_humans : ℕ)
  (cranberries_eaten_by_elk : ℕ)
  (initial_count : total_cranberries = 60000)
  (harvested_percentage : percentage_harvested_by_humans = 40)
  (eaten_by_elk : cranberries_eaten_by_elk = 20000) :
  total_cranberries - (total_cranberries * percentage_harvested_by_humans / 100) - cranberries_eaten_by_elk = 16000 :=
by
  sorry

end cranberries_left_in_bog_l35_35166


namespace minimum_amount_needed_l35_35930

def house_price : ℝ := 320000
def discount_rate : ℝ := 0.96
def deed_tax_rate : ℝ := 0.015

theorem minimum_amount_needed : 
  let discounted_price := house_price * discount_rate in
  let deed_tax := discounted_price * deed_tax_rate in
  let total_amount := discounted_price + deed_tax in
  total_amount = 311808 := by
  sorry

end minimum_amount_needed_l35_35930


namespace find_multiple_l35_35925

-- Define the conditions
def ReetaPencils : ℕ := 20
def TotalPencils : ℕ := 64

-- Define the question and proof statement
theorem find_multiple (AnikaPencils : ℕ) (M : ℕ) :
  AnikaPencils = ReetaPencils * M + 4 →
  AnikaPencils + ReetaPencils = TotalPencils →
  M = 2 :=
by
  intros hAnika hTotal
  -- Skip the proof
  sorry

end find_multiple_l35_35925


namespace sky_falls_distance_l35_35158

def distance_from_city (x : ℕ) (y : ℕ) : Prop := 50 * x = y

theorem sky_falls_distance :
    ∃ D_s : ℕ, distance_from_city D_s 400 ∧ D_s = 8 :=
by
  sorry

end sky_falls_distance_l35_35158


namespace cyclic_sum_inequality_l35_35760

variable {x y z : ℝ}

theorem cyclic_sum_inequality (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  (x * y / z + y * z / x + z * x / y) > 2 * real.cbrt (x^3 + y^3 + z^3) := by
  sorry

end cyclic_sum_inequality_l35_35760


namespace correct_statements_count_l35_35103

-- Conditions
def statement_1 (a b : ℚ) : Prop := (a + b) > a ∧ (a + b) > b
def statement_2 (a b : ℤ) : Prop := ∀ (a > 0 ∧ b < 0) -> ((a + b) > 0)
def statement_3 (a b : ℤ) : Prop := (a < 0 ∧ b < 0) -> ( |a + b| = |a| + |b|)
def statement_4 (a b : ℤ) : Prop := (a > 0 ∧ b > 0) -> (a + b > 0)
def statement_5 (a b : ℤ) : Prop := (a < 0 ∧ b < 0) -> (a + b = a - |b|)
def statement_6 (a b : ℤ) : Prop := (a > 0 ∧ b < 0) -> (a + b = 0)

-- Proof problem
theorem correct_statements_count : 
  (¬ (∀ a b : ℚ, statement_1 a b)) ∧
  (¬ (∀ a b : ℤ, statement_2 a b)) ∧
  (∀ a b : ℤ, statement_3 a b) ∧
  (∀ a b : ℤ, statement_4 a b) ∧
  (¬ (∀ a b : ℤ, statement_5 a b)) ∧
  (¬ (∀ a b : ℤ, statement_6 a b)) → 2 = 2 := 
  by
  sorry

end correct_statements_count_l35_35103


namespace calculate_second_train_start_time_l35_35068

noncomputable def second_train_start_time (dist_pq : ℕ) 
  (start_time_p : ℕ) (speed_p : ℕ) (speed_q : ℕ) (meeting_time : ℕ) : ℕ :=
  let distance_first_train := speed_p * (meeting_time - start_time_p) in
  let distance_second_train := dist_pq - distance_first_train in
  let time_second_train := distance_second_train / speed_q in
  meeting_time - time_second_train

theorem calculate_second_train_start_time :
  second_train_start_time 200 7 20 25 12 = 8 :=
  by
    sorry

end calculate_second_train_start_time_l35_35068


namespace smallest_two_digit_number_product_12_l35_35474

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l35_35474


namespace distance_from_hut_to_station_l35_35929

variable (t s : ℝ)

theorem distance_from_hut_to_station
  (h1 : s / 4 = t + 3 / 4)
  (h2 : s / 6 = t - 1 / 2) :
  s = 15 := by
  sorry

end distance_from_hut_to_station_l35_35929


namespace solution_one_solution_two_l35_35186

noncomputable def problem_one : ℝ :=
  0.25 * ((-1 / 2) ^ (-4)) / ((sqrt 5 - 1) ^ 0) - ((1 / 16) ^ (-1 / 2))

theorem solution_one : problem_one = 0 := by
  sorry

noncomputable def problem_two : ℝ :=
  (real.log 2 + real.log 5 - real.log 8) / (real.log 50 - real.log 40)

theorem solution_two : problem_two = 1 := by
  sorry

end solution_one_solution_two_l35_35186


namespace max_elements_in_S_min_elements_in_S_l35_35389

/-- Let's define the conditions. -/
def A : Set ℕ := {a : ℕ | a < 100}

/-- The set S consisting of all numbers of the form x + y, where x, y ∈ A. -/
def S (A : Set ℕ) : Set ℕ := {s | ∃ x y ∈ A, s = x + y}

/-- Proving the maximum number of distinct elements in S can be 5050. -/
theorem max_elements_in_S (hA : A.card = 100) : (S A).card ≤ 5050 :=
by
  sorry

/-- Proving the minimum number of distinct elements in S can be 199. -/
theorem min_elements_in_S (hA : A.card = 100) : 199 ≤ (S A).card :=
by
  sorry

end max_elements_in_S_min_elements_in_S_l35_35389


namespace max_condition_part1_part2_l35_35274

noncomputable def f (x : Real) (m : Real) : Real :=
  2 * Real.sin (2 * x + Real.pi / 6) + m

theorem max_condition (x : Real) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) (m : Real) :
  ∃ m, ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x m ≤ 6 := sorry

theorem part1 (k : ℤ) (x : Real) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  2 * Real.sin (2 * x + Real.pi / 6) + 4 ∈ Icc (Real.pi / 6 + k * Real.pi) (2 * Real.pi / 3 + k * Real.pi) := sorry

theorem part2 (k : ℤ) (x : Real) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  2 * Real.sin (2 * x + Real.pi / 6) + 4 ≤ 3 → x ∈ Icc (Real.pi / 2 + k * Real.pi) (5 * Real.pi / 6 + k * Real.pi) := sorry

end max_condition_part1_part2_l35_35274


namespace max_tulips_l35_35446

theorem max_tulips (r y n : ℕ) (hr : r = y + 1) (hc : 50 * y + 31 * r ≤ 600) :
  r + y = 2 * y + 1 → n = 2 * y + 1 → n = 15 :=
by
  -- Given the constraints, we need to identify the maximum n given the costs and relationships.
  sorry

end max_tulips_l35_35446


namespace tapanga_corey_candies_l35_35033

theorem tapanga_corey_candies (corey_candies : ℕ) (tapanga_candies : ℕ) 
                              (h1 : corey_candies = 29) 
                              (h2 : tapanga_candies = corey_candies + 8) : 
                              corey_candies + tapanga_candies = 66 :=
by
  rw [h1, h2]
  sorry

end tapanga_corey_candies_l35_35033


namespace sum_permutations_999_l35_35627

-- Definitions of S_n, f(π), and g(π)
def S_n (n : ℕ) : Finset (Equiv.Perm (Fin n)) := Finset.univ

def f {n : ℕ} (π : Equiv.Perm (Fin n)) : ℕ :=
  Finset.card { p : Fin n × Fin n // p.1 < p.2 ∧ π p.1 > π p.2 }

def g {n : ℕ} (π : Equiv.Perm (Fin n)) : ℕ :=
  Finset.card { k : Fin n // π k = k + 1 ∨ π k = k - 1 }

theorem sum_permutations_999 : 
  ∑ π in S_n 999, (-1) ^ (f π + g π) = 995 * 2 ^ 998 := 
sorry

end sum_permutations_999_l35_35627


namespace triangle_equilateral_iff_l35_35355

noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

def triangleIsEquilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

theorem triangle_equilateral_iff (a1 a2 a3 h1 h2 h3 S : ℝ) (A1A2 A2A3 A3A1 : ℝ) :
  S = 1 / 6 * (A1A2 * h1 + A2A3 * h2 + A3A1 * h3) ↔ triangleIsEquilateral A1A2 A2A3 A3A1 :=
sorry

end triangle_equilateral_iff_l35_35355


namespace find_primes_l35_35612

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

/- Define the three conditions -/
def condition1 (p q r : ℕ) : Prop := divides p (1 + q ^ r)
def condition2 (p q r : ℕ) : Prop := divides q (1 + r ^ p)
def condition3 (p q r : ℕ) : Prop := divides r (1 + p ^ q)

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ condition1 p q r ∧ condition2 p q r ∧ condition3 p q r

theorem find_primes (p q r : ℕ) :
  satisfies_conditions p q r ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) :=
by
  sorry

end find_primes_l35_35612


namespace find_missing_number_l35_35117

theorem find_missing_number:
  ∃ x : ℕ, (306 / 34) * 15 + x = 405 := sorry

end find_missing_number_l35_35117


namespace probability_AC_adjacent_BE_not_adjacent_l35_35425

-- Define the 5 students as elements of a set
inductive Student
| A | B | C | D | E
deriving DecidableEq

open Student

-- Define a function to count valid arrangements with given conditions
def count_valid_arrangements (students : List Student) : Nat := sorry

-- Define the total number of permutations of 5 elements
def total_permutations : Nat := 5!

-- The proof statement
theorem probability_AC_adjacent_BE_not_adjacent :
  (count_valid_arrangements [A, B, C, D, E] / (total_permutations : ℚ) = 1 / 5) :=
by
  sorry

end probability_AC_adjacent_BE_not_adjacent_l35_35425


namespace incorrect_statement_l35_35765

variable (f : ℝ → ℝ)
variable (k : ℝ)
variable (h₁ : f 0 = -1)
variable (h₂ : ∀ x, f' x > k)
variable (h₃ : k > 1)

theorem incorrect_statement :
  ¬ f (1 / (k - 1)) < 1 / (k - 1) :=
sorry

end incorrect_statement_l35_35765


namespace cosine_fourier_transform_l35_35223

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then 0
  else if 1 < x ∧ x < 2 then 1
  else if 2 < x then 0
  else 0 -- To handle values outside the specified ranges

-- Condition: ∫_{0}^{∞} |f(x)| dx < ∞
def integral_condition : Prop :=
  ∫ x in 0..∞, |f x| < ∞

-- The statement of the theorem
theorem cosine_fourier_transform (h : integral_condition) :
  ∀ p > 0, F(p) = sqrt(2 / π) * (sin(2 * p) - sin(p)) / p :=
by
  sorry

end cosine_fourier_transform_l35_35223


namespace first_term_geometric_series_l35_35923

theorem first_term_geometric_series (r a S : ℝ)
  (h_r : r = 1 / 3)
  (h_S : S = 27)
  (h_S_formula : S = a / (1 - r)) : 
  a = 18 := by
  rw [h_r, h_S] at h_S_formula
  have h : 27 = a / (2 / 3) := h_S_formula
  rw div_eq_iff_mul_eq at h
  norm_num at h
  exact h.symm


end first_term_geometric_series_l35_35923


namespace find_max_marks_l35_35000

variable (marks_scored : ℕ) -- 212
variable (shortfall : ℕ) -- 22
variable (pass_percentage : ℝ) -- 0.30

theorem find_max_marks (h_marks : marks_scored = 212) 
                       (h_short : shortfall = 22) 
                       (h_pass : pass_percentage = 0.30) : 
  ∃ M : ℝ, M = 780 :=
by {
  sorry
}

end find_max_marks_l35_35000


namespace least_number_to_subtract_l35_35100

theorem least_number_to_subtract (n : ℕ) (h : n = 652543) : 
  ∃ x : ℕ, x = 7 ∧ (n - x) % 12 = 0 :=
by
  sorry

end least_number_to_subtract_l35_35100


namespace division_value_l35_35703

theorem division_value (a b c : ℝ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 7 / 2) : 
  c / a = 6 / 35 := 
by
  sorry

end division_value_l35_35703


namespace smallest_two_digit_l35_35461

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l35_35461


namespace star_j_l35_35197

def star (x y : ℝ) : ℝ := x^3 - x * y

theorem star_j (j : ℝ) : star j (star j j) = 2 * j^3 - j^4 := 
by
  sorry

end star_j_l35_35197


namespace quadrilateral_count_correct_triangle_count_correct_intersection_triangle_count_correct_l35_35520

-- 1. Problem: Count of quadrilaterals from 12 points in a semicircle
def semicircle_points : ℕ := 12
def quadrilaterals_from_semicircle_points : ℕ :=
  let points_on_semicircle := 8
  let points_on_diameter := 4
  360 -- This corresponds to the final computed count, skipping calculation details

theorem quadrilateral_count_correct :
  quadrilaterals_from_semicircle_points = 360 := sorry

-- 2. Problem: Count of triangles from 10 points along an angle
def angle_points : ℕ := 10
def triangles_from_angle_points : ℕ :=
  let points_on_one_side := 5
  let points_on_other_side := 4
  90 -- This corresponds to the final computed count, skipping calculation details

theorem triangle_count_correct :
  triangles_from_angle_points = 90 := sorry

-- 3. Problem: Count of triangles from intersection points of parallel lines
def intersection_points : ℕ := 12
def triangles_from_intersections : ℕ :=
  let line_set_1_count := 3
  let line_set_2_count := 4
  200 -- This corresponds to the final computed count, skipping calculation details

theorem intersection_triangle_count_correct :
  triangles_from_intersections = 200 := sorry

end quadrilateral_count_correct_triangle_count_correct_intersection_triangle_count_correct_l35_35520


namespace mechanic_hourly_rate_l35_35893

def total_cost : ℝ := 9220
def parts_cost : ℝ := 2500
def hours_per_day : ℕ := 8
def days : ℕ := 14
def labor_cost : ℝ := total_cost - parts_cost
def total_hours_worked : ℕ := hours_per_day * days
def hourly_rate : ℝ := labor_cost / total_hours_worked

theorem mechanic_hourly_rate : hourly_rate = 60 := 
by 
  -- We complete the proof here
  sorry

end mechanic_hourly_rate_l35_35893


namespace sum_of_distances_l35_35364

theorem sum_of_distances (a b : ℤ) (k : ℕ) 
  (h1 : |k - a| + |(k + 1) - a| + |(k + 2) - a| + |(k + 3) - a| + |(k + 4) - a| + |(k + 5) - a| + |(k + 6) - a| = 609)
  (h2 : |k - b| + |(k + 1) - b| + |(k + 2) - b| + |(k + 3) - b| + |(k + 4) - b| + |(k + 5) - b| + |(k + 6) - b| = 721)
  (h3 : a + b = 192) :
  a = 1 ∨ a = 104 ∨ a = 191 := 
sorry

end sum_of_distances_l35_35364


namespace find_function_l35_35360

-- Define the set of positive integers
def nat := { n : ℕ // 0 < n }

-- Define the function type
def func := nat → nat

-- Define the condition
def condition (f : func) :=
  ∀ x y : nat, ∃ z : ℤ, (x.1^2 - y.1^2 + 2 * ↑y.1 * (↑(f x).1 + ↑(f y).1)) = z^2

-- Define the theorem
theorem find_function :
  ∀ (f : func), condition f → ∃ k : nat, ∀ n : nat, f n = k * n :=
sorry

end find_function_l35_35360


namespace find_c_l35_35678

-- Define the function f(x)
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- Define the first derivative of f(x)
def f_prime (x c : ℝ) : ℝ := 3 * x ^ 2 - 4 * c * x + c ^ 2

-- Define the condition that f(x) has a local maximum at x = 2
def is_local_max (f' : ℝ → ℝ) (x0 : ℝ) : Prop :=
  f' x0 = 0 ∧ (∀ x, x < x0 → f' x > 0) ∧ (∀ x, x > x0 → f' x < 0)

-- The main theorem stating the equivalent proof problem
theorem find_c (c : ℝ) : is_local_max (f_prime 2) 2 → c = 6 := 
  sorry

end find_c_l35_35678


namespace yvonne_words_is_400_l35_35106

-- Definitions directly from conditions
constant y : ℕ
constant janna_words : ℕ := y + 150
constant removed_words : ℕ := 20
constant added_words : ℕ := 2 * removed_words
constant total_words_after_editing : ℕ := y + janna_words - removed_words + added_words + 30

-- The target theorem to prove
theorem yvonne_words_is_400 (h : total_words_after_editing = 1000) : y = 400 :=
  sorry

end yvonne_words_is_400_l35_35106


namespace total_yield_correct_l35_35341

/-- Jorge's total property in acres -/
def total_acres : ℕ := 60

/-- Yield per acre in good soil -/
def yield_good_soil : ℕ := 400

/-- Yield per acre in clay-rich soil -/
def yield_clay_rich_soil : ℕ := yield_good_soil / 2

/-- Fraction of land that is clay-rich soil -/
def fraction_clay_rich : ℚ := 1 / 3

/-- Acres of clay-rich soil -/
def acres_clay_rich : ℕ := total_acres * fraction_clay_rich

/-- Acres of good soil -/
def acres_good_soil : ℕ := total_acres - acres_clay_rich

/-- Total yield of corn -/
def total_yield : ℕ := (yield_good_soil * acres_good_soil) + (yield_clay_rich_soil * acres_clay_rich)

theorem total_yield_correct : total_yield = 20000 := by
  sorry

end total_yield_correct_l35_35341


namespace number_of_5_7_17_safe_numbers_le20000_is_4032_l35_35989

def is_p_safe (n p : ℕ) : Prop :=
  ∀ m : ℕ, m > 0 → abs (n - m * p) > 2

def count_p_safe_numbers (p : ℕ) (n : ℕ) : ℕ :=
  Nat.card { x : ℕ // x ≤ n ∧ is_p_safe x p }

theorem number_of_5_7_17_safe_numbers_le20000_is_4032 :
  count_p_safe_numbers 7 20000 + count_p_safe_numbers 17 20000 - 
  count_p_safe_numbers 119 20000 = 4032 :=
by
  sorry

end number_of_5_7_17_safe_numbers_le20000_is_4032_l35_35989


namespace square_perimeter_diagonal_20_l35_35110

theorem square_perimeter_diagonal_20 :
  ∀ (d : ℝ), d = 20 → ∃ (p : ℝ), p = 40 * real.sqrt 2 ∧ 
  (∃ (s : ℝ), s^2 + s^2 = d^2 ∧ p = 4 * s) :=
by {
  sorry
}

end square_perimeter_diagonal_20_l35_35110


namespace intersection_M_N_l35_35766

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l35_35766


namespace frame_ratio_correct_l35_35922

def painting_width : ℝ := 18
def painting_height : ℝ := 24
def painting_area : ℝ := painting_width * painting_height

def wood_side_width (x : ℝ) := x / 2
def wood_top_bottom_width (x : ℝ) := x

def total_height (x : ℝ) := painting_width + 2 * wood_top_bottom_width(x)
def total_width (x : ℝ) := painting_height + 2 * wood_side_width(x)

noncomputable def frame_area (x : ℝ) := total_height(x) * total_width(x) - painting_area

theorem frame_ratio_correct (x : ℝ) (h : frame_area(x) = painting_area) : 
  (painting_width + 2 * wood_top_bottom_width(x)) / (painting_height + 2 * wood_side_width(x)) = 2 / 3 :=
sorry

end frame_ratio_correct_l35_35922


namespace ratio_of_height_to_sum_perimeter_and_height_l35_35903

def length := 25
def width := 15
def height := 10

theorem ratio_of_height_to_sum_perimeter_and_height : 
  let perimeter := 2 * (length + width)
  let sum := perimeter + height
  (height : ℚ) / sum = 1 / 9 :=
by
  let perimeter := 2 * (length + width)
  let sum := perimeter + height
  have h : (height : ℚ) / sum = 1 / 9 := by sorry
  exact h

end ratio_of_height_to_sum_perimeter_and_height_l35_35903


namespace no_unique_set_l35_35514

def is_matrix_valid (A : Matrix (Fin n) (Fin n) ℕ := 
  ∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 
    (A i j = (i + j - 1) % n ∨ (A i j = n ∧ (i + j - 1) % n = 0))

theorem no_unique_set (n : ℕ) (h_even: n % 2 = 0) (h_pos: 0 < n) :
  ∀ (A : Matrix (Fin n) (Fin n) ℕ), (is_matrix_valid A) →
    ¬ (∃ (S : Finset (Fin n)), S.card = n ∧ (∀ {i j : Fin n}, i ≠ j →  ∃ k, A(i,j) ∈ S)) :=
by
  sorry

end no_unique_set_l35_35514


namespace smallest_two_digit_number_product_12_l35_35484

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l35_35484


namespace harmonic_series_inequality_l35_35016

theorem harmonic_series_inequality (n : ℕ) (h : n > 1) : 
  1 + ∑ i in range (2^n - 2), (1 / (i + 1 : ℝ)) < n :=
begin
  sorry,
end

end harmonic_series_inequality_l35_35016


namespace sin_alpha_minus_beta_l35_35998

theorem sin_alpha_minus_beta :
  ∀ (α β : ℝ),
    (cos (α - (Real.pi / 3)) = 2 / 3) →
    (cos (β + (Real.pi / 6)) = -2 / 3) →
    (0 < α ∧ α < Real.pi / 2) →
    (Real.pi / 2 < β ∧ β < Real.pi) →
    sin (α - β) = -1 := by
  intros
  sorry

end sin_alpha_minus_beta_l35_35998


namespace convex_polygon_in_rectangle_l35_35024

theorem convex_polygon_in_rectangle {P : Set ℝ^2} (hP_convex : convex P) (hP_area : measure_theory.measure_of P = 1) :
  ∃ R : Set ℝ^2, is_rectangle R ∧ measure_theory.measure_of R ≤ 2 ∧ P ⊆ R :=
sorry

end convex_polygon_in_rectangle_l35_35024


namespace probability_multiple_of_3_or_4_l35_35829

theorem probability_multiple_of_3_or_4 : 
  (∑ i in (finset.range 30).filter (λ x, x % 3 = 0 ∨ x % 4 = 0) (1 : ℚ)) / 30 = 1 / 2 :=
by
  sorry

end probability_multiple_of_3_or_4_l35_35829


namespace codecracker_total_combinations_l35_35312

theorem codecracker_total_combinations (colors slots : ℕ) (h_colors : colors = 6) (h_slots : slots = 5) :
  colors ^ slots = 7776 :=
by
  rw [h_colors, h_slots]
  norm_num

end codecracker_total_combinations_l35_35312


namespace volume_of_rectangular_prism_l35_35902

-- Defining the conditions as assumptions
variables (l w h : ℝ) 
variable (lw_eq : l * w = 10)
variable (wh_eq : w * h = 14)
variable (lh_eq : l * h = 35)

-- Stating the theorem to prove
theorem volume_of_rectangular_prism : l * w * h = 70 :=
by
  have lw := lw_eq
  have wh := wh_eq
  have lh := lh_eq
  sorry

end volume_of_rectangular_prism_l35_35902


namespace theater_ticket_sales_l35_35062

theorem theater_ticket_sales (x y : ℕ) (h1 : x + y = 175) (h2 : 6 * x + 2 * y = 750) : y = 75 :=
sorry

end theater_ticket_sales_l35_35062


namespace smallest_difference_l35_35868

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) : a - b = 4 :=
by
  sorry

end smallest_difference_l35_35868


namespace problem_1_problem_2_l35_35271

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else log a x

theorem problem_1 (h1 : f 2 (f 2 2) = 0) : f 2 (f 2 2) = 0 :=
  by
  exact h1

theorem problem_2 {a : ℝ} : 
  (∀ x y, x < y → f a x ≤ f a y ∨ f a x ≥ f a y ) ∧ (∀ x y, x < y → f a x < f a y ∨ f a x > f a y) → 
  ( 1/7 ≤ a ∧ a < 1/3 ∨ a ∈ ∅ ) :=
  sorry

end problem_1_problem_2_l35_35271


namespace trapezium_division_l35_35155

theorem trapezium_division (h : ℝ) (m n : ℕ) (h_pos : 0 < h) 
  (areas_equal : 4 / (3 * ↑m) = 7 / (6 * ↑n)) :
  m + n = 15 := by
  sorry

end trapezium_division_l35_35155


namespace workers_work_5_days_a_week_l35_35130

def total_weekly_toys : ℕ := 5500
def daily_toys : ℕ := 1100
def days_worked : ℕ := total_weekly_toys / daily_toys

theorem workers_work_5_days_a_week : days_worked = 5 := 
by 
  sorry

end workers_work_5_days_a_week_l35_35130


namespace cubic_expansion_solution_l35_35491

theorem cubic_expansion_solution (x y : ℕ) (h_x : x = 27) (h_y : y = 9) : 
  x^3 + 3 * x^2 * y + 3 * x * y^2 + y^3 = 46656 :=
by
  sorry

end cubic_expansion_solution_l35_35491


namespace same_color_combination_sum_l35_35889

theorem same_color_combination_sum (m n : ℕ) (coprime_mn : Nat.gcd m n = 1)
  (prob_together : ∀ (total_candies : ℕ), total_candies = 20 →
    let terry_red := Nat.choose 8 2;
    let total_cases := Nat.choose total_candies 2;
    let prob_terry_red := terry_red / total_cases;
    
    let mary_red_given_terry := Nat.choose 6 2;
    let reduced_total_cases := Nat.choose 18 2;
    let prob_mary_red_given_terry := mary_red_given_terry / reduced_total_cases;
    
    let both_red := prob_terry_red * prob_mary_red_given_terry;
    
    let terry_blue := Nat.choose 12 2;
    let prob_terry_blue := terry_blue / total_cases;
    
    let mary_blue_given_terry := Nat.choose 10 2;
    let prob_mary_blue_given_terry := mary_blue_given_terry / reduced_total_cases;
    
    let both_blue := prob_terry_blue * prob_mary_blue_given_terry;
    
    let mixed_red_blue := Nat.choose 8 1 * Nat.choose 12 1;
    let prob_mixed_red_blue := mixed_red_blue / total_cases;
    let both_mixed := prob_mixed_red_blue;
    
    let prob_same_combination := both_red + both_blue + both_mixed;
    
    prob_same_combination = m / n
  ) :
  m + n = 5714 :=
by
  sorry

end same_color_combination_sum_l35_35889


namespace twelve_integers_divisible_by_eleven_l35_35643

theorem twelve_integers_divisible_by_eleven (a : Fin 12 → ℤ) : 
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
by
  sorry

end twelve_integers_divisible_by_eleven_l35_35643


namespace gina_credits_l35_35635

theorem gina_credits
  (cost_per_credit : ℕ := 450)
  (num_textbooks : ℕ := 5)
  (cost_per_textbook : ℕ := 120)
  (facilities_fee : ℕ := 200)
  (total_spent : ℕ := 7100) :
  ∃ (c : ℕ), 450 * c + 120 * 5 + 200 = 7100 ∧ c = 14 :=
begin
  use 14,
  -- Proof details are omitted
  sorry,
end

end gina_credits_l35_35635


namespace range_of_m_l35_35672

theorem range_of_m
  (m : ℝ)
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m * x₁ + 1 = 0 ∧ x₂^2 + m * x₂ + 1 = 0) :
  m ∈ set.Ioi 2 ∪ set.Iio (-2) :=
by 
  sorry

end range_of_m_l35_35672


namespace cristina_running_pace_4point2_l35_35005

theorem cristina_running_pace_4point2 :
  ∀ (nicky_pace head_start time_after_start cristina_pace : ℝ),
    nicky_pace = 3 →
    head_start = 12 →
    time_after_start = 30 →
    cristina_pace = 4.2 →
    (time_after_start = head_start + 30 →
    cristina_pace * time_after_start = nicky_pace * (head_start + 30)) :=
by
  sorry

end cristina_running_pace_4point2_l35_35005


namespace log_expression_zero_l35_35876

theorem log_expression_zero (log : Real → Real) (exp : Real → Real) (log_mul : ∀ a b, log (a * b) = log a + log b) :
  log 2 ^ 2 + log 2 * log 50 - log 4 = 0 :=
by
  sorry

end log_expression_zero_l35_35876


namespace diameter_of_ring_X_l35_35145

def diameter_ring_Y : ℝ := 18
def fraction_not_covered : ℝ := 0.2098765432098765

theorem diameter_of_ring_X :
  let fraction_covered := 1 - fraction_not_covered
  let area_Y := π * (diameter_ring_Y / 2) ^ 2
  let area_X := area_Y / fraction_covered
  let D := 2 * real.sqrt (area_X / π)
  D ≈ 20.245029645310668 :=
by
  let fraction_covered := 1 - fraction_not_covered
  let area_Y := π * (diameter_ring_Y / 2) ^ 2
  let area_X := area_Y / fraction_covered
  let D := 2 * real.sqrt (area_X / π)
  exact sorry

end diameter_of_ring_X_l35_35145


namespace part_a_part_b_part_c_part_d_l35_35293

variable (a : ℝ × ℝ) (b : ℝ × ℝ)

def vec_a : ℝ × ℝ := (3, 5)
def vec_b : ℝ × ℝ := (2, -7)

-- Part a: Prove \(\vec{a} + \vec{b} = (5, -2)\)
theorem part_a : vec_a + vec_b = (5, -2) := by
  sorry

-- Part b: Prove \(\vec{a} - \vec{b} = (1, 12)\)
theorem part_b : vec_a - vec_b = (1, 12) := by
  sorry

-- Part c: Prove \(4 \vec{a} = (12, 20)\)
theorem part_c : 4 • vec_a = (12, 20) := by
  sorry

-- Part d: Prove \(-0.5 \vec{b} = (-1, 3.5)\)
theorem part_d : (-0.5) • vec_b = (-1, 3.5) := by
  sorry

end part_a_part_b_part_c_part_d_l35_35293


namespace dividend_calculation_l35_35453

theorem dividend_calculation :
  let divisor := 17
  let quotient := 9
  let remainder := 6
  let dividend := 159
  (divisor * quotient) + remainder = dividend :=
by
  sorry

end dividend_calculation_l35_35453


namespace program_output_l35_35455

-- Define initial condition
def initial_i : ℕ := 1

-- Looping function
def loop (i S : ℕ) : ℕ × ℕ :=
  let i := i + 2
  let S := 2 * i + 3
  let i := i - 1
  (i, S)

-- Recursive function to simulate the loop until the condition is met
def run_loop (i S : ℕ) : ℕ :=
  if i < 8 then
    let (new_i, new_S) := loop i S
    run_loop new_i new_S
  else S

-- Main theorem to prove the final value of S
theorem program_output : run_loop initial_i 0 = 21 :=
  sorry

end program_output_l35_35455


namespace count_numbers_with_3_or_5_in_base_7_l35_35687

/-- 
The problem statement is to prove that among the first 2401 positive integers in base 7, 
the count of those that include at least one digit of 3 or 5 is 1377.
-/
theorem count_numbers_with_3_or_5_in_base_7 :
  ∃ n : ℕ, n = 2401 ∧ (count_with_digit_3_or_5_in_base_7 n) = 1377 :=
by
  have n := 2401
  have h : count_with_digit_3_or_5_in_base_7 n = 1377 := sorry
  use n
  exact ⟨rfl, h⟩

end count_numbers_with_3_or_5_in_base_7_l35_35687


namespace projection_is_correct_l35_35257

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let scalar := (dot_product a b) / (magnitude b)^2 in
  (scalar * b.1, scalar * b.2, scalar * b.3)

def verify_projection : Prop :=
  let a := (0, 1, 2 : ℝ) in
  let b := (-1, 2, 2 : ℝ) in
  projection a b = (-2 / 3, 4 / 3, 4 / 3)

theorem projection_is_correct : verify_projection := by
  sorry

end projection_is_correct_l35_35257


namespace babysitter_earnings_l35_35879

theorem babysitter_earnings (h : ℤ) (regular_rate overtime_rate : ℤ) (weekly_hours : ℤ)
  (h_regular_rate : regular_rate = 16)
  (h_overtime_rate : overtime_rate = regular_rate + regular_rate * 75 / 100)
  (h_weekly_hours : weekly_hours = 40) :
  let regular_hours := 30,
      overtime_hours := weekly_hours - regular_hours,
      regular_earnings := regular_hours * regular_rate,
      overtime_earnings := overtime_hours * overtime_rate,
      total_earnings := regular_earnings + overtime_earnings in
  total_earnings = 760 := 
by {
  sorry
}

end babysitter_earnings_l35_35879


namespace range_of_f_l35_35623

noncomputable def f (x : ℝ) : ℝ := real.arcsin x + real.arccos x + real.arctan x + real.arccot x

theorem range_of_f : set.range f = set.Icc real.pi (3 * real.pi / 2) :=
sorry

end range_of_f_l35_35623


namespace rectangle_solution_l35_35450

theorem rectangle_solution (x : ℝ) :
  (3 * x - 5) * (x + 7) = 15 * x - 14 → x = 3 :=
begin
  sorry
end

end rectangle_solution_l35_35450


namespace largest_2_digit_prime_factor_of_binom_l35_35079

open Nat

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

/-- Definition of the problem conditions -/
def problem_conditions : Prop :=
  let n := binom 300 150
  ∃ p : ℕ, Prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p ≤ 75 ∨ 3 * p < 300) ∧ p = 97

/-- Statement of the proof problem -/
theorem largest_2_digit_prime_factor_of_binom : problem_conditions := 
  sorry

end largest_2_digit_prime_factor_of_binom_l35_35079


namespace plane_divides_diagonal_ratio_cross_section_area_l35_35061

-- Proof Problem 1: Proving the ratio in which the plane divides the diagonal DB_1
theorem plane_divides_diagonal_ratio (a : ℝ) (ABCD A1 B1 C1 D1 : ℝ) (M : Point)
  (h1 : M = midpoint AB)
  (h2 : Plane M parallel_to line BD1)
  (h3 : Plane M parallel_to line A1C1) : 
  divides_diagonal DB1 M (3/5) :=
sorry

-- Proof Problem 2: Proving the area of the resulting cross-section
theorem cross_section_area (a : ℝ) (ABCD A1 B1 C1 D1 : ℝ) (M : Point)
  (h1 : M = midpoint AB)
  (h2 : Plane M parallel_to line BD1)
  (h3 : Plane M parallel_to line A1C1) : 
  area_of_cross_section (Plane M) (7 * a^2 * √6 / 16) :=
sorry

end plane_divides_diagonal_ratio_cross_section_area_l35_35061


namespace expand_polynomials_l35_35212

-- Define the given polynomials
def poly1 (x : ℝ) : ℝ := 12 * x^2 + 5 * x - 3
def poly2 (x : ℝ) : ℝ := 3 * x^3 + 2

-- Define the expected result of the polynomial multiplication
def expected (x : ℝ) : ℝ := 36 * x^5 + 15 * x^4 - 9 * x^3 + 24 * x^2 + 10 * x - 6

-- State the theorem
theorem expand_polynomials (x : ℝ) :
  (poly1 x) * (poly2 x) = expected x :=
by
  sorry

end expand_polynomials_l35_35212


namespace slope_of_tangent_line_at_origin_l35_35205

theorem slope_of_tangent_line_at_origin : 
  (deriv (λ x : ℝ, Real.exp x) 0) = 1 :=
by
  sorry

end slope_of_tangent_line_at_origin_l35_35205


namespace subset_sum_even_count_l35_35688

open Finset

def S : Finset ℕ := {48, 51, 79, 103, 124, 137, 161}

theorem subset_sum_even_count :
  (S.subsets 4).count (λ t, t.sum % 2 = 0) = 10 := 
by
  sorry

end subset_sum_even_count_l35_35688


namespace graph_shift_l35_35405

noncomputable def g (x : ℝ) : ℝ := 
  if -2 ≤ x ∧ x ≤ 1 then -x 
  else if 1 < x ∧ x ≤ 3 then real.sqrt (4 - (x - 1)^2)
  else if 3 < x ∧ x ≤ 5 then x - 3
  else 0 

theorem graph_shift : 
  ∀ x : ℝ, g (x - 3) = if -2 ≤ (x - 3) ∧ (x - 3) ≤ 1 then -(x - 3)
                       else if 1 < (x - 3) ∧ (x - 3) ≤ 3 then real.sqrt (4 - ((x - 1) - 3)^2)
                       else if 3 < (x - 3) ∧ (x - 3) ≤ 5 then (x - 3) - 3
                       else 0 := 
by
  sorry

end graph_shift_l35_35405


namespace determine_k_l35_35725

noncomputable def line_intersects_circle (x y k : ℝ) := 
  x^2 + y^2 = 4 ∧ x - k*y + 1 = 0

theorem determine_k (k : ℝ) :
  (∃ x y : ℝ, line_intersects_circle x y k) ∧ 
  (∃ M : ℝ × ℝ, (∃ A B : ℝ × ℝ, 
    line_intersects_circle A.1 A.2 k ∧ 
    line_intersects_circle B.1 B.2 k ∧
    M = (A.1 + B.1, A.2 + B.2))) ∧ 
  (M.1^2 + M.2^2 = 4) →
  k = 0 :=
by
  sorry

end determine_k_l35_35725


namespace max_product_l35_35202

theorem max_product (a b : ℝ) (h1 : 9 * a ^ 2 + 16 * b ^ 2 = 25) (h2 : a > 0) (h3 : b > 0) :
  a * b ≤ 25 / 24 :=
sorry

end max_product_l35_35202


namespace sum_not_divisible_by_210_sum_divisible_by_11_only_for_3k_plus_1_l35_35380

theorem sum_not_divisible_by_210 (n : ℕ) :
  ¬ (210 ∣ ∑ k in Finset.range (n+1), 2^(3 * k) * Nat.choose (2 * n + 1) (2 * k + 1)) :=
sorry

theorem sum_divisible_by_11_only_for_3k_plus_1 (n : ℕ) :
  (11 ∣ ∑ k in Finset.range (n+1), 2^(3 * k) * Nat.choose (2 * n + 1) (2 * k + 1)) ↔ ∃ k : ℕ, n = 3 * k + 1 :=
sorry

end sum_not_divisible_by_210_sum_divisible_by_11_only_for_3k_plus_1_l35_35380


namespace modulus_value_l35_35231

theorem modulus_value (m : ℝ) (h : |4 + m * complex.I| = 4 * real.sqrt 13) : m = 8 * real.sqrt 3 :=
sorry

end modulus_value_l35_35231


namespace negation_proposition_l35_35825

theorem negation_proposition (m : ℤ) :
  ¬(∃ x : ℤ, x^2 + 2*x + m < 0) ↔ ∀ x : ℤ, x^2 + 2*x + m ≥ 0 :=
by
  sorry

end negation_proposition_l35_35825


namespace triangle_area_l35_35403

-- Definitions of the given conditions
def parabola (p : ℝ) : ℝ → ℝ := λ y, (1 / (2*p)) * y^2
def hyperbola (a b : ℝ) : ℝ × ℝ := (a, b)
def asymptote (x : ℝ) (a b : ℝ) : ℝ := (b / a) * x

-- The proof problem in Lean 4 statement
theorem triangle_area (a b : ℝ)
  (h1 : a^2 = 3) 
  (h2 : b^2 = 1) 
  (h3 : ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 2) 
  (h4 : ∃ p : ℝ, 2*p = 4 ∧ parabola p = λ y, y^2 = 8*x) :
  let directrix := -2 in
  let focus := (2, 0) in
  let y_intersect := λ x, asymptote x a b in
  let A := (directrix, y_intersect directrix) in
  let B := (directrix, - y_intersect directrix) in
  ∃ S : ℝ, S = 1/2 * |B.2 - A.2| * 2 ∧ S = 4*sqrt(3)/3 :=
sorry

end triangle_area_l35_35403


namespace remainder_of_98_mul_102_mod_11_l35_35097

theorem remainder_of_98_mul_102_mod_11 : (98 * 102) % 11 = 6 := 
by
sory

end remainder_of_98_mul_102_mod_11_l35_35097


namespace num_valid_points_l35_35525

-- Conditions
def A := (-2, 3)
def C := (4, -3)
def valid_path (x y : ℤ) : Prop := abs (x + 2) + abs (y - 3) + abs (x - 4) + abs (y + 3) ≤ 25

-- Goal statement
theorem num_valid_points : (finset.filter (λ (p : ℤ × ℤ), valid_path p.1 p.2) 
                                      (finset.product (finset.range 7).map (λ x, x - 2))
                                      (finset.range 19).map (λ y, y - 9)).card = 193 :=
sorry

end num_valid_points_l35_35525


namespace min_k_exists_l35_35744

variable (X : Type) [Fintype X] (n : ℕ) (f : X → X)

axiom h1 : ∀ x, f x ≠ x
axiom h2 : ∀ A : Finset X, (A.card = 40) → (A ∩ A.image f).nonempty

-- The mathematically equivalent proof problem in Lean 4 statement
theorem min_k_exists (hn : Fintype.card X = 100) : ∃ B : Finset X, B.card = 69 ∧ (B ∪ B.image f) = Finset.univ := 
sorry

end min_k_exists_l35_35744


namespace time_to_clear_l35_35438

noncomputable def length_of_train1 : ℝ := 161
noncomputable def length_of_train2 : ℝ := 165
noncomputable def speed_of_train1_kmh : ℝ := 80
noncomputable def speed_of_train2_kmh : ℝ := 65

noncomputable def total_distance : ℝ := length_of_train1 + length_of_train2
noncomputable def relative_speed_kmh : ℝ := speed_of_train1_kmh + speed_of_train2_kmh
noncomputable def relative_speed_ms : ℝ := relative_speed_kmh * (1000 / 3600)

theorem time_to_clear : total_distance / relative_speed_ms ≈ 8.09 := by
  sorry

end time_to_clear_l35_35438


namespace area_of_path_l35_35143

theorem area_of_path (l_field w_field path_width : ℝ) 
  (h_field : l_field = 60) (w_field : w_field = 55) (h_path : path_width = 2.5) : 
  let l_total := l_field + 2 * path_width
      w_total := w_field + 2 * path_width
      area_total := l_total * w_total
      area_field := l_field * w_field
      area_path := area_total - area_field
  in area_path = 600 :=
by
  sorry

end area_of_path_l35_35143


namespace largest_number_among_selected_students_l35_35716

def total_students := 80

def smallest_numbers (x y : ℕ) : Prop :=
  x = 6 ∧ y = 14

noncomputable def selected_students (n : ℕ) : ℕ :=
  6 + (n - 1) * 8

theorem largest_number_among_selected_students :
  ∀ (x y : ℕ), smallest_numbers x y → (selected_students 10 = 78) :=
by
  intros x y h
  rw [smallest_numbers] at h
  have h1 : x = 6 := h.1
  have h2 : y = 14 := h.2
  exact rfl

#check largest_number_among_selected_students

end largest_number_among_selected_students_l35_35716


namespace convert_base_10_to_base_5_l35_35194

theorem convert_base_10_to_base_5 :
  (256 : ℕ) = 2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0 :=
by
  sorry

end convert_base_10_to_base_5_l35_35194


namespace gcd_g102_g103_l35_35353

def g (x : ℕ) : ℕ := x^2 - x + 2007

theorem gcd_g102_g103 : 
  Nat.gcd (g 102) (g 103) = 3 :=
by
  sorry

end gcd_g102_g103_l35_35353


namespace correct_payment_l35_35126

-- Define the given conditions
def discount (amount : ℝ) : ℝ :=
  if amount < 200 then
    amount
  else if amount ≤ 600 then
    amount * 0.9
  else
    amount * 0.8

-- Given values
def first_purchase : ℝ := 168
def second_purchase : ℝ := 423

-- Calculating the total value without discount
def actual_second_purchase : ℝ := 423 / 0.9
def total_value : ℝ := first_purchase + actual_second_purchase

-- The amount to be paid
def total_payment := discount total_value

-- The goal or statement for the proof
theorem correct_payment:
  total_payment = 510.4 :=
by
  sorry

end correct_payment_l35_35126


namespace smallest_product_is_298150_l35_35783

def digits : List ℕ := [5, 6, 7, 8, 9, 0]

theorem smallest_product_is_298150 :
  ∃ (a b c : ℕ), 
    a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c = 298150) :=
sorry

end smallest_product_is_298150_l35_35783


namespace find_19a_20b_21c_l35_35342

theorem find_19a_20b_21c (a b c : ℕ) (h₁ : 29 * a + 30 * b + 31 * c = 366) 
  (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : 19 * a + 20 * b + 21 * c = 246 := 
sorry

end find_19a_20b_21c_l35_35342


namespace num_terms_in_sequence_l35_35957

theorem num_terms_in_sequence :
  ∀ (a d l : ℤ), a = -53 → d = 5 → l = 87 →
  ∃ n : ℕ, l = a + (n - 1) * d ∧ l ∈ (list.map (λ k, a + k * d) (list.range n)) ∧
  (list.length (list.filter (λ x, x ≤ l) (list.map (λ k, a + k * d) (list.range 30)))) = 29 :=
by
  intros a d l H1 H2 H3
  use 29
  split
  sorry
  split
  sorry
  sorry

end num_terms_in_sequence_l35_35957


namespace susan_more_cats_than_bob_after_exchanges_l35_35801

theorem susan_more_cats_than_bob_after_exchanges :
  ∀ (susan_initial bob_initial emma_initial susan_gift bob_gift emma_gift susan_to_bob emma_to_susan emma_to_bob : ℕ),
    susan_initial = 21 → 
    bob_initial = 3 → 
    emma_initial = 8 →
    susan_gift = 12 → 
    bob_gift = 14 → 
    emma_gift = 6 → 
    susan_to_bob = 6 → 
    emma_to_susan = 5 → 
    emma_to_bob = 3 → 
    let susan_final := susan_initial + susan_gift - susan_to_bob + emma_to_susan,
        bob_final := bob_initial + bob_gift + susan_to_bob + emma_to_bob in
    susan_final - bob_final = 6 :=
by
  intros
  sorry

end susan_more_cats_than_bob_after_exchanges_l35_35801


namespace interval_real_cardinality_l35_35788

noncomputable def f : ℝ → ℝ := λ x, Real.tan (Real.pi * x / 2)

theorem interval_real_cardinality :
  ∃ (f: ℝ → ℝ), Bijection f ∧ (∀ x, x ∈ Ioo (-1:ℝ) 1 → True) :=
begin
  use f,
  split,
  { -- Prove that f is injective
    intros x1 x2 h,
    simp [f, h],
    sorry,
  },
  { -- Prove that f is surjective
    intro y,
    use (2 / Real.pi * Real.arctan y),
    sorry,
  }
end

end interval_real_cardinality_l35_35788


namespace eventually_zero_implies_rational_l35_35759

noncomputable def prime_seq (n : ℕ) : ℕ := sorry

def fractional_part (x : ℝ) : ℝ :=
  x - x.floor

def sequence_x (x0 : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then x0
  else
    let x_prev := sequence_x x0 (k - 1)
    in if x_prev = 0 then 0 else fractional_part (prime_seq k / x_prev)

theorem eventually_zero_implies_rational (x0 : ℝ) (H1 : 0 < x0) (H2 : x0 < 1) (H3 : ∃N, ∀n ≥ N, sequence_x x0 n = 0) : 
  ∃ a b : ℕ, b ≠ 0 ∧ x0 = a / b :=
sorry

end eventually_zero_implies_rational_l35_35759


namespace alternating_sum_eq_970299_l35_35204

def alternating_sum (n : ℕ) : ℤ :=
  ∑ k in Finset.range (n + 1),
    if (is_square (integer.of_nat k)) then (-1) ^ (integer.sqrt (integer.of_nat k) : ℤ)
    else ((-1) ^ (integer.sqrt (integer.of_nat k) : ℤ)) * (integer.of_nat k)

theorem alternating_sum_eq_970299 :
  alternating_sum 9801 = 970299 :=
by
  sorry

end alternating_sum_eq_970299_l35_35204


namespace average_age_of_team_is_23_l35_35808

noncomputable def average_age_team (A : ℝ) : Prop :=
  let captain_age := 27
  let wicket_keeper_age := 28
  let team_size := 11
  let remaining_players := team_size - 2
  let remaining_average_age := A - 1
  11 * A = 55 + 9 * (A - 1)

theorem average_age_of_team_is_23 : average_age_team 23 := by
  sorry

end average_age_of_team_is_23_l35_35808


namespace relationship_among_a_b_c_l35_35399

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

def a := Real.exp 2012 * f 0
def b := Real.exp 2011 * f 1
def c := Real.exp 1000 * f 1012

theorem relationship_among_a_b_c
  (H : ∀ x : ℝ, f'(x) - f(x) < 0) : a > b ∧ b > c :=
sorry

end relationship_among_a_b_c_l35_35399


namespace least_multiple_of_9_not_lucky_integer_l35_35138

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10 in
  digits.foldr (λ d acc => d + acc) 0

def is_lucky_integer (n : ℕ) : Prop :=
  n > 0 ∧ (n % sum_of_digits n = 0)

def is_multiple_of_9 (n : ℕ) : Prop :=
  n > 0 ∧ (n % 9 = 0)

theorem least_multiple_of_9_not_lucky_integer : ∃ n, is_multiple_of_9 n ∧ ¬ is_lucky_integer n ∧ ∀ m, is_multiple_of_9 m ∧ ¬ is_lucky_integer m → n ≤ m :=
  sorry

end least_multiple_of_9_not_lucky_integer_l35_35138


namespace max_enclosed_area_l35_35846

theorem max_enclosed_area :
  ∃ (l w : ℝ), 2 * l + 2 * w = 400 ∧ l ≥ 90 ∧ w ≥ 50 ∧ l * w = 10000 :=
by
  have h₁ : ∃ l w : ℝ, 2 * l + 2 * w = 400 := sorry,
  have h₂ : ∃ l w : ℝ, l * w = 10000 := sorry,
  have h₃ : ∀ l : ℝ, l ≥ 90 := sorry,
  have h₄ : ∀ w : ℝ, w ≥ 50 := sorry,
  sorry

end max_enclosed_area_l35_35846


namespace magnitude_difference_l35_35601

open Complex

noncomputable def c1 : ℂ := 18 - 5 * I
noncomputable def c2 : ℂ := 14 + 6 * I
noncomputable def c3 : ℂ := 3 - 12 * I
noncomputable def c4 : ℂ := 4 + 9 * I

theorem magnitude_difference : 
  Complex.abs ((c1 * c2) - (c3 * c4)) = Real.sqrt 146365 :=
by
  sorry

end magnitude_difference_l35_35601


namespace second_train_length_l35_35069

variable (L : ℝ)

theorem second_train_length :
  (∀ (time : ℝ) (speed_train1_kmph speed_train2_kmph length_train1_m : ℝ),
    speed_train1_kmph = 42 ∧
    speed_train2_kmph = 30 ∧
    length_train1_m = 100 ∧
    time = 14.998800095992321 →
      let speed_train1_ms := speed_train1_kmph * (1000 / 3600),
          speed_train2_ms := speed_train2_kmph * (1000 / 3600),
          relative_speed_ms := speed_train1_ms + speed_train2_ms,
          total_distance := relative_speed_ms * time,
          length_train2_m := total_distance - length_train1_m
      in length_train2_m = 199.9760019198464) := sorry

end second_train_length_l35_35069


namespace train_crosses_platform_in_34_seconds_l35_35560

theorem train_crosses_platform_in_34_seconds 
    (train_speed_kmph : ℕ) 
    (time_cross_man_sec : ℕ) 
    (platform_length_m : ℕ) 
    (h_speed : train_speed_kmph = 72) 
    (h_time : time_cross_man_sec = 18) 
    (h_platform_length : platform_length_m = 320) 
    : (platform_length_m + (train_speed_kmph * 1000 / 3600) * time_cross_man_sec) / (train_speed_kmph * 1000 / 3600) = 34 :=
by
    sorry

end train_crosses_platform_in_34_seconds_l35_35560


namespace perpendicular_line_sufficient_condition_l35_35591

theorem perpendicular_line_sufficient_condition (a : ℝ) :
  (-a) * ((a + 2) / 3) = -1 ↔ (a = -3 ∨ a = 1) :=
by {
  sorry
}

#print perpendicular_line_sufficient_condition

end perpendicular_line_sufficient_condition_l35_35591


namespace comparison_of_A_and_B_l35_35999

noncomputable def A (m : ℝ) : ℝ := Real.sqrt (m + 1) - Real.sqrt m
noncomputable def B (m : ℝ) : ℝ := Real.sqrt m - Real.sqrt (m - 1)

theorem comparison_of_A_and_B (m : ℝ) (h : m > 1) : A m < B m :=
by
  sorry

end comparison_of_A_and_B_l35_35999


namespace polygon_pairs_count_l35_35361

theorem polygon_pairs_count :
  ∀ (a b : ℝ) (m n : ℕ), a ≠ b → a > 0 → b > 0 → m > 0 → n > 0 → m ≤ 100 → n ≤ 100 →
  (∃ (P : polygon (m+n)), P.inscribed (a+b) ∧ P.side_lengths = list.repeat a m ++ list.repeat b n) →
  ∑ x in finset.range 101, ∑ y in finset.range 101, if x > 0 ∧ y > 0 ∧ (min x y < 6 ∨ max x y > 7) then 1 else 0 = 940 :=
by sorry

end polygon_pairs_count_l35_35361


namespace find_ratio_l35_35051

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Given conditions
axiom sum_arithmetic_a (n : ℕ) : S n = n / 2 * (a 1 + a n)
axiom sum_arithmetic_b (n : ℕ) : T n = n / 2 * (b 1 + b n)
axiom sum_ratios (n : ℕ) : S n / T n = (2 * n + 1) / (3 * n + 2)

-- The proof problem
theorem find_ratio : (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := 
sorry

end find_ratio_l35_35051


namespace find_D_l35_35914

variable (A B C D E F G H I J : ℕ)

noncomputable def unique_digits : Prop :=
  list.nodup [A, B, C, D, E, F, G, H, I, J]

noncomputable def consecutive_even (x y z : ℕ) : Prop :=
  x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧ x = y + 2 ∧ y = z + 2

noncomputable def consecutive_odd (x y z : ℕ) : Prop :=
  x % 2 = 1 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧ x = y + 2 ∧ y = z + 2

theorem find_D :
  unique_digits A B C D E F G H I J →
  A > B → B > C →
  D > E → E > F →
  G > H → H > I → I > J →
  consecutive_even A B C →
  consecutive_odd D E F →
  H + I + J = 9 →
  D = 9 :=
sorry

end find_D_l35_35914


namespace no_integer_length_tangent_l35_35995

theorem no_integer_length_tangent (circumference : ℝ) (m n : ℕ) (t_1 : ℝ)
  (h₀ : circumference = 24) 
  (h₁ : ∃ P, ∃ t : ℝ, t = t_1 ∧ isTangent P t)
  (h₂ : odd m)
  (h₃ : n = 24 - m)
  (h₄ : t_1^2 = m * n) : t_1 ∈ ℕ → false :=
by
  sorry

end no_integer_length_tangent_l35_35995


namespace remainder_squared_mod_five_l35_35867

theorem remainder_squared_mod_five (n k : ℤ) (h : n = 5 * k + 3) : ((n - 1) ^ 2) % 5 = 4 :=
by
  sorry

end remainder_squared_mod_five_l35_35867


namespace triangle_area_l35_35249

-- Define triangle ABC with internal angles A, B, C and sides a, b, c
variables (A B C : ℝ) -- angles
variables (a b c : ℝ) -- sides

-- Define the given conditions as hypotheses
hypothesis h1 : a^2 * sin C = 4 * sin A
hypothesis h2 : (c*a + c*b) * (sin A - sin B) = sin C * (2 * real.sqrt 7 - c^2)

-- Prove that the area of triangle ABC is 3/2
theorem triangle_area (A B C a b c : ℝ) 
  (h1 : a^2 * sin C = 4 * sin A)
  (h2 : (c*a + c*b) * (sin A - sin B) = sin C * (2 * real.sqrt 7 - c^2)) :
  1/2 * a * c * sin B = 3/2 := 
sorry

end triangle_area_l35_35249


namespace toothpaste_last_day_l35_35421

theorem toothpaste_last_day (total_toothpaste : ℝ)
  (dad_use_per_brush : ℝ) (dad_brushes_per_day : ℕ)
  (mom_use_per_brush : ℝ) (mom_brushes_per_day : ℕ)
  (anne_use_per_brush : ℝ) (anne_brushes_per_day : ℕ)
  (brother_use_per_brush : ℝ) (brother_brushes_per_day : ℕ)
  (sister_use_per_brush : ℝ) (sister_brushes_per_day : ℕ)
  (grandfather_use_per_brush : ℝ) (grandfather_brushes_per_day : ℕ)
  (guest_use_per_brush : ℝ) (guest_brushes_per_day : ℕ) (guest_days : ℕ)
  (total_usage_per_day : ℝ) :
  total_toothpaste = 80 →
  dad_use_per_brush * dad_brushes_per_day = 16 →
  mom_use_per_brush * mom_brushes_per_day = 12 →
  anne_use_per_brush * anne_brushes_per_day = 8 →
  brother_use_per_brush * brother_brushes_per_day = 4 →
  sister_use_per_brush * sister_brushes_per_day = 2 →
  grandfather_use_per_brush * grandfather_brushes_per_day = 6 →
  guest_use_per_brush * guest_brushes_per_day * guest_days = 6 * 4 →
  total_usage_per_day = 54 →
  80 / 54 = 1 → 
  total_toothpaste / total_usage_per_day = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end toothpaste_last_day_l35_35421


namespace ratio_of_rises_l35_35437

-- Define necessary constants and variables
variables (r1 r2 rm : ℝ)
variables (h1 h2 : ℝ) -- initial heights of the liquid
variables (h1' h2' : ℝ) -- heights of the liquid after the marble is submerged

-- Conditions
def cone_conditions : Prop :=
  r1 = 4 ∧ r2 = 8 ∧ rm = 2

-- Initial volumes of liquid in the cones
def volume_liquid (r h : ℝ) := (1/3) * π * r^2 * h

-- Volume displaced by one marble
def volume_marble (r : ℝ) := (4/3) * π * r^3

-- Final volume equations after marble is submerged
def final_volume_liquid (r h hf : ℝ) :=
  volume_liquid r hf = volume_liquid r h + volume_marble rm

-- Definitions for the rises in liquid levels
def rise_liquid_level (h h' : ℝ) := h' - h

-- Define the statement to prove
theorem ratio_of_rises (h1 h2 h1' h2' : ℝ) :
  cone_conditions ∧ final_volume_liquid r1 h1 h1' ∧ final_volume_liquid r2 h2 h2' ∧
    rise_liquid_level h1 h1' = 2 ∧ rise_liquid_level h2 h2' = 0.5 →
  (rise_liquid_level h1 h1') / (rise_liquid_level h2 h2') = 4 :=
by
  sorry

end ratio_of_rises_l35_35437


namespace david_marks_in_physics_l35_35952

theorem david_marks_in_physics :
  ∀ (E M P C B average n : ℕ),
    E = 96 →
    M = 95 →
    C = 97 →
    B = 95 →
    average = 93 →
    n = 5 →
    (average * n - (E + M + C + B) = P) →
    P = 82 :=
by
  intros E M P C B average n hE hM hC hB havg hn hcalc
  rw [hE, hM, hC, hB, havg, hn] at hcalc
  calc 93 * 5 - (96 + 95 + 97 + 95) = 465 - 383 := by norm_num
                             ... = 82           := by norm_num
  exact hcalc

end david_marks_in_physics_l35_35952


namespace intersection_complement_l35_35350

open Set

variable {U : Type*} [hU : Nonempty U] -- Assumed to be nonempty.

-- Definitions of sets A and B
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x ≤ -1}

-- Definition of the Universe U
def U : Set ℝ := univ

-- Proof statement
theorem intersection_complement :
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end intersection_complement_l35_35350


namespace floor_equation_solution_l35_35609

open Int

theorem floor_equation_solution (x : ℝ) :
  (⌊ ⌊ 3 * x ⌋ - 1/2 ⌋ = ⌊ x + 4 ⌋) ↔ (7/3 ≤ x ∧ x < 3) := sorry

end floor_equation_solution_l35_35609


namespace exists_difference_divisible_by_11_l35_35641

theorem exists_difference_divisible_by_11 (a : Fin 12 → ℤ) :
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
  sorry

end exists_difference_divisible_by_11_l35_35641


namespace log_expression_evaluation_l35_35391

theorem log_expression_evaluation : 
  (4 * Real.log 2 + 3 * Real.log 5 - Real.log (1/5)) = 4 := 
  sorry

end log_expression_evaluation_l35_35391


namespace evaluate_expression_l35_35602

theorem evaluate_expression (b : ℝ) (hb : b ≠ 0) :
  (1 / 25 * b^0) + ((1 / (25 * b))^0) - (125^(-1 / 3)) - (50^(-1 / 2)) = 1 - (√2 / 10) :=
by
  -- Proof goes here
  sorry

end evaluate_expression_l35_35602


namespace area_of_triangle_ABC_is_24_l35_35222

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (10, 6)

-- Define the area calculation
def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  0.5 * |(v.1 * w.2 - v.2 * w.1)|

theorem area_of_triangle_ABC_is_24 :
  triangleArea A B C = 24 := by
  sorry

end area_of_triangle_ABC_is_24_l35_35222


namespace smallest_two_digit_number_product_12_l35_35477

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l35_35477


namespace smallest_two_digit_number_product_12_l35_35485

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l35_35485


namespace inequality_holds_l35_35299

variables (a b c : ℝ)

theorem inequality_holds 
  (h1 : a > b) : 
  a / (c^2 + 1) > b / (c^2 + 1) :=
sorry

end inequality_holds_l35_35299


namespace min_expression_value_l35_35659

theorem min_expression_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + 2 * b = 1) : 
  ∃ x, (x = (a^2 + 1) / a + (2 * b^2 + 1) / b) ∧ x = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end min_expression_value_l35_35659


namespace distance_between_Sneezy_and_Grumpy_is_8_l35_35059

variables (DS DV SP VP: ℕ) (SV: ℕ)

theorem distance_between_Sneezy_and_Grumpy_is_8
  (hDS : DS = 5)
  (hDV : DV = 4)
  (hSP : SP = 10)
  (hVP : VP = 17)
  (hSV_condition1 : SV + SP > VP)
  (hSV_condition2 : SV < DS + DV)
  (hSV_condition3 : 7 < SV) :
  SV = 8 := 
sorry

end distance_between_Sneezy_and_Grumpy_is_8_l35_35059


namespace ellipse_foci_distance_l35_35040

theorem ellipse_foci_distance :
  (∀ x y : ℝ, sqrt ((x - 4)^2 + (y - 5)^2) + sqrt ((x + 6)^2 + (y - 9)^2) = 24) →
  (dist (4, 5) (-6, 9) = 2 * sqrt 29) :=
by
  intro h
  sorry

end ellipse_foci_distance_l35_35040


namespace seq_10_is_4_l35_35978

-- Define the sequence with given properties
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 4
  | (n + 2) => if n % 2 = 0 then 4 else 3

-- Theorem statement: The 10th term of the sequence is 4
theorem seq_10_is_4 : seq 9 = 4 :=
by sorry

end seq_10_is_4_l35_35978


namespace equivalent_annual_rate_approx_l35_35594

noncomputable def annual_rate : ℝ := 0.045
noncomputable def days_in_year : ℝ := 365
noncomputable def daily_rate : ℝ := annual_rate / days_in_year
noncomputable def equivalent_annual_rate : ℝ := (1 + daily_rate) ^ days_in_year - 1

theorem equivalent_annual_rate_approx :
  abs (equivalent_annual_rate - 0.0459) < 0.0001 :=
by sorry

end equivalent_annual_rate_approx_l35_35594


namespace initial_quantity_of_A_l35_35884

theorem initial_quantity_of_A (x : ℚ) 
    (h1 : 7 * x = a)
    (h2 : 5 * x = b)
    (h3 : a + b = 12 * x)
    (h4 : a' = a - (7 / 12) * 9)
    (h5 : b' = b - (5 / 12) * 9 + 9)
    (h6 : a' / b' = 7 / 9) : 
    a = 23.625 := 
sorry

end initial_quantity_of_A_l35_35884


namespace range_a_l35_35277

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(1 - x) else 1 - log x / log 2

theorem range_a (a : ℝ) (h : |f a| ≥ 2) : 
  a ∈ Iic (1/2) ∪ Ici 8 := 
sorry

end range_a_l35_35277


namespace distance_from_x_axis_of_intersection_point_l35_35820

theorem distance_from_x_axis_of_intersection_point :
  ∃ x ∈ Ioc 0 (π/2), 2 + 3 * cos (2 * x) = 3 * √3 * sin x ∧ 3 = 3 := 
sorry

end distance_from_x_axis_of_intersection_point_l35_35820


namespace total_distance_between_A_and_B_l35_35515

/-- Xiao Li drove from location A to location B. Two hours after departure, the car broke down at location C, and it took 40 minutes to repair. 
After the repair, the speed was only 75% of the normal speed, resulting in arrival at location B being 2 hours later than planned. If the car had 
instead broken down at location D, which is 72 kilometers past location C, with the same repair time of 40 minutes and the speed after the repair 
still being 75% of the normal speed, then the arrival at location B would be only 1.5 hours later than planned. Determine the total distance in 
kilometers between location A and location B. -/
theorem total_distance_between_A_and_B (v d : ℝ) (normal_speed : v > 0) :
  let repair_time := 2/3,
      reduced_speed := 3/4 * v,
      delay1 := 2,
      delay2 := 1.5 in
  -- Time for repair at C
  (2 + repair_time + (d - 2 * v) / reduced_speed - d / v = delay1) ∧
  -- Time for repair at D (72 km past C)
  ((2 + (72 / v) + repair_time + (d - 2 * v - 72) / reduced_speed - d / v = delay2)) →
  d = 288 := sorry

end total_distance_between_A_and_B_l35_35515


namespace minimum_length_of_RQ_l35_35327

noncomputable def minimum_RQ_length (b c alpha : ℝ) : ℝ :=
  b * c * Real.sin alpha / Real.sqrt (b^2 + c^2 + 2 * b * c * Real.cos alpha)

theorem minimum_length_of_RQ
  (AC AB : ℝ) (angleBAC : ℝ)
  (P : ℝ → Prop) (PQ_parallel_AB : Prop) (PR_parallel_AC : Prop)
  : P(AB) ∧ P(AC) ∧ ∀ (Q R : Point), PQ_parallel_AB ∧ PR_parallel_AC ∧
    Q ∈ line AC ∧ R ∈ line AB →
  minimum_RQ_length AC AB angleBAC :=
by sorry

end minimum_length_of_RQ_l35_35327


namespace population_in_2070_l35_35980

noncomputable def population : ℕ → ℕ
| 1960            := 150
| (t + 20) := 2 * population t
| _              := 0

theorem population_in_2070 : population 2070 = 4800 :=
by
  -- from the given conditions, implement the calculations
  have p1960 : population 1960 = 150 := rfl
  have p1980 : population 1980 = 2 * population 1960 := rfl
  have p2000 : population 2000 = 2 * population 1980 := rfl
  have p2020 : population 2020 = 2 * population 2000 := rfl
  have p2040 : population 2040 = 2 * population 2020 := rfl
  have p2060 : population 2060 = 2 * population 2040 := rfl
  have p2070 : population 2070 = 2 * population 2060 := rfl
  sorry

end population_in_2070_l35_35980


namespace determine_target_function_l35_35513

noncomputable def target_function_form (a x y : ℝ) : Prop :=
  y = a * (x + 2) ^ 2 + 2

def perpendicular_tangents (a x0 y0 : ℝ) : Prop :=
  ∃ u0 z1 z2, 
  u0 = (y0 - 2) ∧
  2 * a * z1 * z2 = -1 ∧
  z1 + z2 = 2 * x0 ∧
  y0 = a * z1 * z2 + 2

def logarithmic_condition (x y : ℝ) : Prop :=
  log (x - x^2 + 3) (y - 6) =
  log (x - x^2 + 3) ((|2 * x + 6| - |2 * x + 3|) / (3 * x + 7.5) * sqrt (x ^ 2 + 5 * x + 6.25))

theorem determine_target_function (x0 y0 : ℝ) (h_tangent : perpendicular_tangents (-0.05) x0 y0) (h_log : logarithmic_condition x0 y0) :
  target_function_form (-0.05) x0 y0 :=
by
  sorry

end determine_target_function_l35_35513


namespace max_tulips_count_l35_35444

theorem max_tulips_count : ∃ (r y n : ℕ), 
  n = r + y ∧ 
  n % 2 = 1 ∧ 
  |r - y| = 1 ∧ 
  50 * y + 31 * r ≤ 600 ∧ 
  n = 15 := 
by
  sorry

end max_tulips_count_l35_35444


namespace katie_ds_games_l35_35338

theorem katie_ds_games (new_friends_games old_friends_games total_friends_games katie_games : ℕ) 
  (h1 : new_friends_games = 88)
  (h2 : old_friends_games = 53)
  (h3 : total_friends_games = 141)
  (h4 : total_friends_games = new_friends_games + old_friends_games + katie_games) :
  katie_games = 0 :=
by
  sorry

end katie_ds_games_l35_35338


namespace problem1_problem2_l35_35578

-- Lean statement for Problem 1
theorem problem1 : (-2: ℝ)^2 + (-1 / (2: ℝ))^4 + (3 - real.pi)^0 = -47 / 16 := by
  sorry

-- Lean statement for Problem 2
theorem problem2 : (5: ℝ)^2022 * (-1 / (5: ℝ))^2023 = -1 / 5 := by
  sorry

end problem1_problem2_l35_35578


namespace num_elements_in_A_inter_B_l35_35683

-- Define the sets A and B according to the given conditions
def A := { x : ℕ | ∃ n : ℕ, x = 3 * n + 2 }
def B := {6, 8, 10, 12, 14}

-- Statement: Prove that the number of elements in the intersection of set A and set B is 2
theorem num_elements_in_A_inter_B : finset.card (finset.filter (λ x, x ∈ B) (finset.filter (λ x, ∃ n, x = 3 * n + 2) (finset.range (15)))) = 2 := 
sorry

end num_elements_in_A_inter_B_l35_35683


namespace length_of_AB_l35_35540

theorem length_of_AB (x1 y1 x2 y2 : ℝ) 
  (h_parabola_A : y1^2 = 8 * x1) 
  (h_focus_line_A : y1 = 2 * (x1 - 2)) 
  (h_parabola_B : y2^2 = 8 * x2) 
  (h_focus_line_B : y2 = 2 * (x2 - 2)) 
  (h_sum_x : x1 + x2 = 6) : 
  |x1 - x2| = 10 :=
sorry

end length_of_AB_l35_35540


namespace sequence_nth_term_l35_35322

/-- The nth term of the sequence {a_n} defined by a_1 = 1 and
    the recurrence relation a_{n+1} = 2a_n + 2 for all n ∈ ℕ*,
    is given by the formula a_n = 3 * 2 ^ (n - 1) - 2. -/
theorem sequence_nth_term (n : ℕ) (h : n > 0) : 
  ∃ (a : ℕ → ℤ), a 1 = 1 ∧ (∀ n > 0, a (n + 1) = 2 * a n + 2) ∧ a n = 3 * 2 ^ (n - 1) - 2 :=
  sorry

end sequence_nth_term_l35_35322


namespace island_puzzle_solution_l35_35010

structure Person :=
(is_good : Prop)
(is_boy : Prop)

def Ali := {is_good := ¬ (Ali.is_good ∧ Bali.is_good), is_boy := true}
def Bali := {is_good := ¬ Ali.is_good, is_boy := true}

theorem island_puzzle_solution (Ali Bali : Person)
  (Ali_statement : Ali.is_good ↔ Ali.is_good ∧ Bali.is_good)
  (Bali_statement : Bali.is_good ↔ Ali.is_boy ∧ Bali.is_boy)
  (truth_telling : Ali.is_good → (Ali_statement ↔ Ali.is_good ∧ Bali.is_good))
  (truth_telling : Bali.is_good → (Bali_statement ↔ Ali.is_boy ∧ Bali.is_boy))
  (lying : ¬Ali.is_good → ¬(Ali_statement ↔ Ali.is_good ∧ Bali.is_good))
  (lying : ¬Bali.is_good → ¬(Bali_statement ↔ Ali.is_boy ∧ Bali.is_boy)) :
  (Ali.is_good = false ∧ Bali.is_good = true) ∧
  (Ali.is_boy = true ∧ Bali.is_boy = true) :=
by
  sorry

end island_puzzle_solution_l35_35010


namespace point_on_graph_l35_35265

theorem point_on_graph (g : ℝ → ℝ) (h : g 8 = 10) :
  ∃ x y : ℝ, 3 * y = g (3 * x - 1) + 3 ∧ x = 3 ∧ y = 13 / 3 ∧ x + y = 22 / 3 :=
by
  sorry

end point_on_graph_l35_35265


namespace find_t_l35_35404

def g (t x : ℝ) : ℝ := (Real.sin x) * (Real.logb 2 ((Real.sqrt (x^2 + 2 * t)) + x))

theorem find_t (t : ℝ) : (∀ x : ℝ, g t x = g t (-x)) → t = 1 / 2 :=
by
  sorry

end find_t_l35_35404


namespace regression_line_intercept_l35_35632

theorem regression_line_intercept
  (x : ℕ → ℝ)
  (y : ℕ → ℝ)
  (h_x_sum : x 1 + x 2 + x 3 + x 4 + x 5 + x 6 = 10)
  (h_y_sum : y 1 + y 2 + y 3 + y 4 + y 5 + y 6 = 4) :
  ∃ a : ℝ, (∀ i, y i = (1 / 4) * x i + a) → a = 1 / 4 :=
by
  sorry

end regression_line_intercept_l35_35632


namespace seq_10_is_4_l35_35977

-- Define the sequence with given properties
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 4
  | (n + 2) => if n % 2 = 0 then 4 else 3

-- Theorem statement: The 10th term of the sequence is 4
theorem seq_10_is_4 : seq 9 = 4 :=
by sorry

end seq_10_is_4_l35_35977


namespace eight_p_plus_one_composite_l35_35017

open Nat

theorem eight_p_plus_one_composite (p : ℕ) (hp : Prime p) (h8p_1 : Prime (8 * p - 1)) : ¬Prime (8 * p + 1) :=
by
  sorry

end eight_p_plus_one_composite_l35_35017


namespace ethanol_to_acetic_acid_l35_35982

theorem ethanol_to_acetic_acid
  (ethanol moles : ℕ)
  (oxygen moles : ℕ)
  (h_ethanol : ethanol moles = 3)
  (h_oxygen : oxygen moles = 3) :
  ∃ acetic_acid moles, acetic_acid moles = 3 :=
by
  sorry

end ethanol_to_acetic_acid_l35_35982


namespace wall_length_proof_l35_35063

-- Define the conditions from the problem
def wall_height : ℝ := 100 -- Height in cm
def wall_thickness : ℝ := 5 -- Thickness in cm
def brick_length : ℝ := 25 -- Brick length in cm
def brick_width : ℝ := 11 -- Brick width in cm
def brick_height : ℝ := 6 -- Brick height in cm
def number_of_bricks : ℝ := 242.42424242424244

-- Calculate the volume of one brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Calculate the total volume of the bricks
def total_brick_volume : ℝ := brick_volume * number_of_bricks

-- Define the proof problem
theorem wall_length_proof : total_brick_volume = wall_height * wall_thickness * 800 :=
sorry

end wall_length_proof_l35_35063


namespace smallest_partition_l35_35417

theorem smallest_partition (S : Finset ℕ) (hS : S = Finset.range 2023)
  (n : ℕ)
  (S_i : Fin n → Finset ℕ)
  (h_partition : ∀ (i : Fin n), S_i i ⊆ S ∧ (∀ (i j : Fin n), i ≠ j → Disjoint (S_i i) (S_i j)))
  (h_condition : ∀ i : Fin n, ((∀ x y ∈ S_i i, x ≠ y → Nat.gcd x y > 1) ∨ (∀ x y ∈ S_i i, x ≠ y → Nat.gcd x y = 1))) :
  n ≥ 14 := 
sorry

end smallest_partition_l35_35417


namespace inverse_function_logarithm_l35_35710

theorem inverse_function_logarithm (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) (f : ℝ → ℝ) (h_inv : ∀ x : ℝ, f (a ^ x) = x) (h_f_4 : f 4 = -2) :
  f = (λ x, log (1 / 2) x) :=
sorry

end inverse_function_logarithm_l35_35710


namespace problem1_problem2_l35_35192

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else n^2 * ∑ k in range (n - 1), (1 / (k + 2)^2)

theorem problem1 (n : ℕ) (hn : n ≥ 2) :
  (a n + 1) / a (n + 1) = n^2 / (n + 1)^2 := by
  sorry

theorem problem2 (n : ℕ) :
  (∏ i in range n, (1 + 1 / a (i + 1))) < 4 := by
  sorry

end problem1_problem2_l35_35192


namespace bounded_sequence_range_l35_35241

theorem bounded_sequence_range (a : ℝ) (a_n : ℕ → ℝ) (h1 : a_n 1 = a)
    (hrec : ∀ n : ℕ, a_n (n + 1) = 3 * (a_n n)^3 - 7 * (a_n n)^2 + 5 * (a_n n))
    (bounded : ∃ M : ℝ, ∀ n : ℕ, abs (a_n n) ≤ M) :
    0 ≤ a ∧ a ≤ 4/3 :=
by
  sorry

end bounded_sequence_range_l35_35241


namespace no_integer_solutions_2_pow_2x_minus_3_pow_2y_eq_85_l35_35586

theorem no_integer_solutions_2_pow_2x_minus_3_pow_2y_eq_85 :
  ∀ x y : ℤ, 2^(2*x) - 3^(2*y) ≠ 85 :=
by
  intro x y
  sorry

end no_integer_solutions_2_pow_2x_minus_3_pow_2y_eq_85_l35_35586


namespace find_max_min_difference_l35_35038

noncomputable def f (x : ℝ) : ℝ :=
  3 - Real.sin x - 2 * (Real.cos x) ^ 2

theorem find_max_min_difference :
  ∀ x ∈ Set.Icc (Real.pi / 6) (7 * Real.pi / 6),
  let f_val := f x in
  ∃ a b : ℝ, 
    (a = 2 ∧ b = 7 / 8 ∧ f_val ∈ Union (Interval a b) ) → 
    (max (univ.image f) - min (univ.image f)) = 9 / 8 := 
sorry

end find_max_min_difference_l35_35038


namespace most_suitable_graph_for_height_change_l35_35041

-- Definitions of graph types
def BarGraph (usage : String) := usage = "compare quantities among different groups at a specific time point"
def LineGraph (usage : String) := usage = "show trends or changes over time"
def PieChart (usage : String) := usage = "show a part-to-whole relationship at a single point in time"

-- Proof statement
theorem most_suitable_graph_for_height_change (height_change : String) :
  (height_change = "show trends or changes over time") → True := by trivial

-- Applying the definitions to conditions
example : most_suitable_graph_for_height_change "show trends or changes over time" sorry

end most_suitable_graph_for_height_change_l35_35041


namespace union_sets_l35_35362

def A := { x : ℝ | x^2 ≤ 1 }
def B := { x : ℝ | 0 < x }

theorem union_sets : A ∪ B = { x | -1 ≤ x } :=
by {
  sorry -- Proof is omitted as per the instructions
}

end union_sets_l35_35362


namespace find_vertex_parabola_l35_35673

-- Define the quadratic equation of the parabola
def parabola_eq (x y : ℝ) : Prop := x^2 - 4 * x + 3 * y + 10 = 0

-- Definition of the vertex of the parabola
def is_vertex (v : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), parabola_eq x y → v = (2, -2)

-- The main statement we want to prove
theorem find_vertex_parabola : 
  ∃ v : ℝ × ℝ, is_vertex v :=
by
  use (2, -2)
  intros x y hyp
  sorry

end find_vertex_parabola_l35_35673


namespace smallest_two_digit_l35_35462

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l35_35462


namespace nine_point_circle_center_intersection_l35_35315

noncomputable def acute_angled_triangle (A B C H : Point) (nine_point_center : Point) : Prop :=
  is_orthocenter H (Triangle A B C) ∧
  acute_angled (Triangle A B C) ∧
  let M_N_J_L := intersection_points_of_perpendicular_bisectors (A B C) H in
  let hexagon := convex_hexagon M_N_J_L in
  all_perpendicular_bisectors_intersect_at_single_point_no_interior (hexagon) nine_point_center

theorem nine_point_circle_center_intersection
  (A B C H nine_point_center : Point)
  (condition_orthocenter : is_orthocenter H (Triangle A B C))
  (condition_acute : acute_angled (Triangle A B C))
  (condition_hexagon_form : let M_N_J_L := intersection_points_of_perpendicular_bisectors (A B C) H in
    let hexagon := convex_hexagon M_N_J_L in
    true) :
  acute_angled_triangle A B C H nine_point_center :=
begin
  sorry -- Proof omitted
end

end nine_point_circle_center_intersection_l35_35315


namespace smallest_positive_period_maximum_area_of_triangle_l35_35769

def f (x : ℝ) : ℝ := 1 - 2 * (Real.sin x) ^ 2 - Real.cos (2 * x + (π / 3))

theorem smallest_positive_period :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ π) ∧ (∃ T : ℝ, T = π) :=
by
  sorry

variables (a b c A B C : ℝ)
def area_of_triangle (a b c : ℝ) (B : ℝ) : ℝ := 1 / 2 * a * c * Real.sin B

theorem maximum_area_of_triangle :
  b = 5 ∧ f (B / 2) = 1 ∧ B > 0 ∧ B < π ∧ (Real.cos B = 1 / 2) →
  (∀ a c : ℝ, (a ^ 2 + c ^ 2 - a * c = 25) → area_of_triangle a b c B ≤ 25 * (Real.sqrt 3) / 4) ∧
  (∃ a c : ℝ, b = 5 ∧ a = 5 ∧ c = 5 ∧ area_of_triangle a b c B = 25 * (Real.sqrt 3) / 4) :=
by
  sorry

end smallest_positive_period_maximum_area_of_triangle_l35_35769


namespace find_sin2alpha_cos4alpha_l35_35648

noncomputable def tan (α : ℝ) : ℝ := sin α / cos α
noncomputable def cot (α : ℝ) : ℝ := cos α / sin α

theorem find_sin2alpha_cos4alpha (α : ℝ) (h : (2 - real.sqrt 3 : ℝ) * (2 + real.sqrt 3 : ℝ) = 1 ∧ 2 * (2 - real.sqrt 3) + 2 * (2 + real.sqrt 3) - 4 = 0) :
  sin (2 * α) = 1 / 2 ∧ cos (4 * α) = 1 / 2 :=
by {
  sorry   -- Complete the proof here
}

end find_sin2alpha_cos4alpha_l35_35648


namespace find_1995th_remaining_number_l35_35233

theorem find_1995th_remaining_number :
  let seq := { n : ℕ | ¬ (n % 4 = 0 ∨ n % 7 = 0) ∨ n % 5 = 0 }
  (seq.to_finset.sort (≤)).nth (1994) = some 2795 :=
by
  have seq := { n : ℕ | ¬ (n % 4 = 0 ∨ n % 7 = 0) ∨ n % 5 = 0 }
  sorry

end find_1995th_remaining_number_l35_35233


namespace part1_part2_l35_35502

def isOddPrime (p : ℕ) := nat.prime p ∧ p % 2 = 1
def satisfies_Kp (p n : ℕ) := ∃ (parts : list (list ℕ)), 
  (parts.length = p ∧ (∀ part ∈ parts, list.sum part = list.sum (list.range (n+1)) / p) ∧ parts.bind id = list.range (n+1))

-- Part 1: If n satisfies K_p, then n or n + 1 is a multiple of p
theorem part1 (p n : ℕ) (hp : isOddPrime p) (hn : satisfies_Kp p n) : n % p = 0 ∨ (n + 1) % p = 0 :=
sorry

-- Part 2: If n is a multiple of 2p, then n satisfies K_p
theorem part2 (p n : ℕ) (hp : isOddPrime p) (hn : n % (2 * p) = 0) : satisfies_Kp p n :=
sorry

end part1_part2_l35_35502


namespace find_lambda_l35_35252

theorem find_lambda {λ : ℝ} :
  let A := (-1 : ℝ, -1 : ℝ),
      B := (1 : ℝ, 3 : ℝ),
      C := (2 : ℝ, λ),
      AB := (B.1 - A.1, B.2 - A.2),
      AC := (C.1 - A.1, C.2 - A.2) in
  (∃ k : ℝ, AC = (k * AB.1, k * AB.2)) → λ = 5 :=
by
  sorry

end find_lambda_l35_35252


namespace chords_intersect_probability_l35_35009

theorem chords_intersect_probability (n : ℕ) (hn : n = 1996) (A B C D : fin n) :
  ∃ p : ℚ, p = 1 / 3 :=
by {
  -- converting the given data in terms of lean variables
  have h := hn.symm,
  rw ←h at A B C D,
  sorry
}

end chords_intersect_probability_l35_35009


namespace playground_perimeter_is_correct_l35_35823

-- Definition of given conditions
def length_of_playground : ℕ := 110
def width_of_playground : ℕ := length_of_playground - 15

-- Statement of the problem to prove
theorem playground_perimeter_is_correct :
  2 * (length_of_playground + width_of_playground) = 230 := 
by
  sorry

end playground_perimeter_is_correct_l35_35823


namespace marble_selection_probability_l35_35524

theorem marble_selection_probability :
  let total_marbles := 9
  let selected_marbles := 4
  let total_ways := Nat.choose total_marbles selected_marbles
  let red_marbles := 3
  let blue_marbles := 3
  let green_marbles := 3
  let ways_one_red := Nat.choose red_marbles 1
  let ways_two_blue := Nat.choose blue_marbles 2
  let ways_one_green := Nat.choose green_marbles 1
  let favorable_outcomes := ways_one_red * ways_two_blue * ways_one_green
  (favorable_outcomes : ℚ) / total_ways = 3 / 14 :=
by
  sorry

end marble_selection_probability_l35_35524


namespace max_f_value_is_m_solution_set_f_less_than_1_max_ab_bc_value_l35_35279

-- Condition 1
def f (x : ℝ) : ℝ := |x - 3| - 2 * |x + 1|

-- Condition 2
def m : ℝ := 4

-- Theorem 1: maximum value of f(x) is m = 4
theorem max_f_value_is_m : ∃ x : ℝ, f x = m := sorry

-- Theorem 2: solution set of f(x) < 1
theorem solution_set_f_less_than_1 : {x : ℝ | f x < 1} = {x : ℝ | x < -4 ∨ (0 < x ∧ x < 3)} := sorry

-- Condition 3
variables (a b c : ℝ)
axiom ab_bound : a > 0 ∧ b > 0 ∧ a^2 + 2*b^2 + c^2 = 4

-- Theorem 3: maximum value of ab + bc is 2
theorem max_ab_bc_value : ab_bound a b c → ∃ max_ab_bc : ℝ, max_ab_bc = (ab + bc) ∧ max_ab_bc = 2 := sorry

end max_f_value_is_m_solution_set_f_less_than_1_max_ab_bc_value_l35_35279


namespace order_of_a_b_c_l35_35655

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := 1 / ∫ x in 0..Real.pi, x

theorem order_of_a_b_c : a > b ∧ b > c := by
  sorry

end order_of_a_b_c_l35_35655


namespace common_elements_count_l35_35296

-- Define the sequence conditions
def seq1 (n : ℕ) : ℕ := 6 * n + 4
def seq2 (m : ℕ) : ℕ := 11 * m - 1

-- State the problem as a Lean theorem
theorem common_elements_count : ∃ k : ℕ, 
  (∀ n m : ℕ, seq1 n = seq2 m → (1 ≤ n ∧ n ≤ 166) ∧ (1 ≤ m ∧ m ≤ 91)) → 
  finset.card (finset.filter (λ n, ∃ (m : ℕ), seq1 n = seq2 m) (finset.range 167)) = 16 :=
by { sorry }

end common_elements_count_l35_35296


namespace discount_percentage_for_two_pairs_of_jeans_l35_35149

theorem discount_percentage_for_two_pairs_of_jeans
  (price_per_pair : ℕ := 40)
  (price_for_three_pairs : ℕ := 112)
  (discount : ℕ := 8)
  (original_price_for_two_pairs : ℕ := price_per_pair * 2)
  (discount_percentage : ℕ := (discount * 100) / original_price_for_two_pairs) :
  discount_percentage = 10 := 
by
  sorry

end discount_percentage_for_two_pairs_of_jeans_l35_35149


namespace ellipse_equation_midpoint_coordinates_l35_35764

noncomputable def ellipse_c := {x : ℝ × ℝ | (x.1^2 / 25) + (x.2^2 / 16) = 1}

theorem ellipse_equation (a b : ℝ) (h1 : a = 5) (h2 : b = 4) :
    ∀ x y : ℝ, x = 0 → y = 4 → (y^2 / b^2 = 1) ∧ (e = 3 / 5) → 
      (a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) := 
sorry

theorem midpoint_coordinates (a b : ℝ) (h1 : a = 5) (h2 : b = 4) :
    ∀ x y x1 x2 y1 y2 : ℝ, 
    (y = 4 / 5 * (x - 3)) → 
    (y1 = 4 / 5 * (x1 - 3)) ∧ (y2 = 4 / 5 * (x2 - 3)) ∧ 
    (x1^2 / a^2) + ((y1 - 3)^2 / b^2) = 1 ∧ (x2^2 / a^2) + ((y2 - 3)^2 / b^2) = 1 ∧ 
    (x1 + x2 = 3) → 
    ((x1 + x2) / 2 = 3 / 2) ∧ ((y1 + y2) / 2 = -6 / 5) := 
sorry

end ellipse_equation_midpoint_coordinates_l35_35764


namespace initial_strawberries_l35_35376

variable initial : ℕ

theorem initial_strawberries 
  (h1 : initial + 78 = 120) : initial = 42 :=
by
  sorry

end initial_strawberries_l35_35376


namespace original_deck_size_l35_35533

noncomputable def initial_card_probability (r b : ℕ) : Prop := r / (r + b) = 2 / 5
noncomputable def new_card_probability (r b : ℕ) : Prop := r / (r + (b + 6)) = 1 / 3

theorem original_deck_size :
  ∃ (r b : ℕ), initial_card_probability r b ∧ new_card_probability r b ∧ (r + b = 5) :=
by {
  sorry,
}

end original_deck_size_l35_35533


namespace ball_never_returns_l35_35882

-- Define our conditions as stated in a)
def isPerpendicular (a b : ℕ) : Prop := (a + b = 90) ∨ (a + b = 270)

-- Define our billiard polygon
structure BilliardPolygon :=
(vertexes : List (ℕ × ℕ)) -- each vertex is a pair (x, y)
(adjacent_perpendicular : ∀ (i : ℕ), isPerpendicular (vertexes.get (i % vertexes.length)).1 (vertexes.get ((i+1) % vertexes.length)).1)

-- Define the reflection law
def reflection_law (angle_incidence angle_reflection : ℕ) : Prop := angle_incidence = angle_reflection

-- Define the situation where the ball starts at A and its internal angle is 90°
structure BallAtVertexA (P : BilliardPolygon) :=
(vertex_A : ℕ)
(internal_angle_A : isPerpendicular (P.vertexes.get vertex_A).1 90)

-- Define that the ball follows the reflection law
structure BallPath (P : BilliardPolygon) (A : BallAtVertexA P) :=
(trajectory : List (ℕ × ℕ))
(reflects_correctly : ∀ (i : ℕ) (h : i < trajectory.length - 1), reflection_law (trajectory.get i).1 (trajectory.get (i+1)).1)

-- The statement we need to prove
theorem ball_never_returns (P : BilliardPolygon) (A : BallAtVertexA P) (path : BallPath P A) : 
  ¬ ∃ (n : ℕ), path.trajectory.get n = P.vertexes.get A.vertex_A := 
by sorry -- proof is omitted, as per the requirement

end ball_never_returns_l35_35882


namespace Zara_total_earnings_l35_35321

noncomputable def Zara.hourly_wage := 53.20 / 6 -- hourly wage (x)
def Zara.first_week_hours := 18
def Zara.second_week_hours := 24
def Zara.earnings_week1 := Zara.first_week_hours * Zara.hourly_wage
def Zara.earnings_week2 := Zara.second_week_hours * Zara.hourly_wage
def Zara.total_earnings := Zara.earnings_week1 + Zara.earnings_week2

theorem Zara_total_earnings (x := 53.20 / 6) : Zara.total_earnings = 371.60 :=
by
  sorry

end Zara_total_earnings_l35_35321


namespace sum_G_0_to_5_l35_35356

def G : ℕ → ℕ
| 0       := 1
| 1       := 4
| (n + 2) := 3 * G (n + 1) - 2 * G n

theorem sum_G_0_to_5 : ∑ n in Finset.range 6, G n = 69 := by
  sorry

end sum_G_0_to_5_l35_35356


namespace cos_750_eq_sqrt3_div_2_l35_35517

theorem cos_750_eq_sqrt3_div_2 : cos (750 * real.pi / 180) = real.sqrt 3 / 2 :=
by 
  sorry

end cos_750_eq_sqrt3_div_2_l35_35517


namespace largest_angle_l35_35052

-- Definitions for our conditions
def right_angle : ℝ := 90
def sum_of_two_angles (a b : ℝ) : Prop := a + b = (4 / 3) * right_angle
def angle_difference (a b : ℝ) : Prop := b = a + 40

-- Statement of the problem to be proved
theorem largest_angle (a b c : ℝ) (h_sum : sum_of_two_angles a b) (h_diff : angle_difference a b) (h_triangle : a + b + c = 180) : c = 80 :=
by sorry

end largest_angle_l35_35052


namespace magnanimous_positive_integers_count_l35_35579

-- Define what it means for an integer to be "magnanimous"
def is_magnanimous (n : ℕ) : Prop :=
  if n < 10 then True
  else let digits := n.digits 10 in
  (digits.sorted = digits ∨ digits.sorted.reverse = digits ∨ 
  ∃ (k : ℕ), k > 0 ∧ digits.reverse.take (k - 1) = digits.sorted ∧ digits.get (k - 1) = 0)

-- Prove that there are 1030 magnanimous positive integers
theorem magnanimous_positive_integers_count :
  { n : ℕ | is_magnanimous n }.to_finset.card = 1030 :=
sorry

end magnanimous_positive_integers_count_l35_35579


namespace tenth_term_is_four_l35_35973

noncomputable def a : ℕ → ℝ
| 0     := 3
| 1     := 4
| (n + 1) := 12 / a n

theorem tenth_term_is_four : a 9 = 4 :=
by
  sorry

end tenth_term_is_four_l35_35973


namespace no_intersection_of_curves_l35_35988

theorem no_intersection_of_curves :
  ∀ x y : ℝ, ¬ (3 * x^2 + 2 * y^2 = 4 ∧ 6 * x^2 + 3 * y^2 = 9) :=
by sorry

end no_intersection_of_curves_l35_35988


namespace capacitor_capacitance_l35_35848

theorem capacitor_capacitance 
  (U ε Q : ℝ) 
  (hQ : Q = (U^2 * (ε - 1)^2 * C) /  (2 * ε * (ε + 1)))
  : C = (2 * ε * (ε + 1) * Q) / (U^2 * (ε - 1)^2) :=
by
  sorry

end capacitor_capacitance_l35_35848


namespace minimum_N_for_rectangle_l35_35150

theorem minimum_N_for_rectangle (N : ℕ) (h : N ≥ 102) :
  ∃ lengths : fin N → ℕ, (∑ i, lengths i = 200) ∧
  ∀ lengths, (∑ i, lengths i = 200 → ∃ (A B C D : finset ℕ), 
    A ∪ B ∪ C ∪ D = finset.univ ∧
    ∀ X Y, A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ A ∩ D = ∅ ∧ B ∩ C = ∅ ∧ B ∩ D = ∅ ∧ C ∩ D = ∅ ∧
    (∃ a b, a ∈ A ∧ b ∈ B ∧ lengths a + lengths b = 100) ∧
    (∃ c d, c ∈ C ∧ d ∈ D ∧ lengths c + lengths d = 100)))
  :=
by sorry

end minimum_N_for_rectangle_l35_35150


namespace always_positive_l35_35767

noncomputable def f : ℝ → ℝ :=
sorry

theorem always_positive (h_diff : differentiable ℝ f)
  (h_ineq : ∀ x : ℝ, 2 * f(x) + x * (derivative f x) > x^2) :
  ∀ x : ℝ, f(x) > 0 :=
sorry

end always_positive_l35_35767


namespace problem_remainder_mod_1991_l35_35357

/-- Given \( b = 1^{2} - 2^{2} + 3^{2} - 4^{2} + \cdots - 1988^{2} + 1989^{2} \), 
    prove that \( b \mod 1991 = 1 \). -/
theorem problem_remainder_mod_1991 :
  let b := (Finset.range 1990).sum (λ n, if n % 2 = 0 then (n + 1) ^ 2 else -(n + 1) ^ 2)
  in b % 1991 = 1 :=
sorry

end problem_remainder_mod_1991_l35_35357


namespace sum_of_solutions_l35_35351

def greatest_int (a : ℚ) : ℤ := int.floor a
def equation (x : ℚ) : Prop := greatest_int (3 * x + 1) = 2 * x - 1 / 2

theorem sum_of_solutions : 
  (∑ x in {x : ℚ | equation x}, x) = -2 := 
sorry

end sum_of_solutions_l35_35351


namespace tenth_term_of_sequence_l35_35966

noncomputable def sequence : ℕ → ℚ
| 0       := 3
| 1       := 4
| (n + 2) := 12 / (sequence n)

theorem tenth_term_of_sequence : sequence 9 = 4 :=
by
  sorry

end tenth_term_of_sequence_l35_35966


namespace max_students_distributing_pens_and_pencils_l35_35509

theorem max_students_distributing_pens_and_pencils :
  Nat.gcd 1001 910 = 91 :=
by
  -- remaining proof required
  sorry

end max_students_distributing_pens_and_pencils_l35_35509


namespace paper_left_after_notebooks_l35_35023

variable (S : ℕ) (N : ℕ) (P : ℕ)
variables (h_initial : S = 100) (h_notebooks : N = 3) (h_pages_per_notebook : P = 30)

theorem paper_left_after_notebooks : S - (N * P) = 10 :=
by
  rw [h_initial, h_notebooks, h_pages_per_notebook]
  norm_num
  sorry

end paper_left_after_notebooks_l35_35023


namespace rhombus_diagonals_not_equal_l35_35104

def is_rhombus (A B C D : Type) [HasEq A] [HasEq B] [HasEq C] [HasEq D] : Prop :=
  (A = B) ∧ (B = C) ∧ (C = D) ∧ (D = A) ∧   
  (diagonal_bisect A B C D) ∧ 
  (diagonal_bisect_angles A B C D)

def diagonal_bisect (A B C D : Type) : Prop :=
  -- Assuming the definition of diagonals bisecting each other
  sorry

def diagonal_bisect_angles (A B C D : Type): Prop :=
  -- Assuming the definition of diagonals bisecting one pair of opposite angles
  sorry

theorem rhombus_diagonals_not_equal (A B C D : Type) [HasEq A] [HasEq B] [HasEq C] [HasEq D]:
  is_rhombus A B C D → ¬ (diagonal_equal A B C D) :=
begin
  intro h,
  -- Assuming the definition of diagonals equality
  sorry
end

end rhombus_diagonals_not_equal_l35_35104


namespace tetrahedron_volume_l35_35216

theorem tetrahedron_volume (angle_ABC_BCD : Real.Angle := π / 4)
  (area_ABC : ℝ := 150)
  (area_BCD : ℝ := 90)
  (BC : ℝ := 12) :
  volume_tetrahedron (angle_ABC_BCD) (area_ABC) (area_BCD) (BC) = 375 * Real.sqrt 2 :=
sorry

end tetrahedron_volume_l35_35216


namespace smallest_8_digit_multiple_of_360_with_unique_digits_largest_8_digit_multiple_of_360_with_unique_digits_l35_35624

/-- Predicate that checks if a number is an 8-digit multiple of 360 with all unique digits --/
def isEightDigitMultipleOf360WithUniqueDigits (n : ℕ) : Prop :=
  (10000000 ≤ n) ∧ (n < 100000000) ∧  -- n is 8-digit
  (n % 360 = 0) ∧  -- n is a multiple of 360
  (let digits := (to_digits 10 n).nodup in digits) ∧  -- All digits are unique
  (10 ∣ n)

/-- Predicate that checks if all digits of the number are unique --/
def unique_digits (n : ℕ) := nodup (to_digits 10 n)

/-- The smallest 8-digit multiple of 360 with unique digits --/
theorem smallest_8_digit_multiple_of_360_with_unique_digits :
  ∃ n, isEightDigitMultipleOf360WithUniqueDigits n ∧ n = 12378960 :=
by
  sorry

/-- The largest 8-digit multiple of 360 with unique digits --/
theorem largest_8_digit_multiple_of_360_with_unique_digits :
  ∃ n, isEightDigitMultipleOf360WithUniqueDigits n ∧ n = 98763120 :=
by
  sorry

end smallest_8_digit_multiple_of_360_with_unique_digits_largest_8_digit_multiple_of_360_with_unique_digits_l35_35624


namespace space_explorer_acquisition_l35_35148

theorem space_explorer_acquisition :
  let minerals := 4*6^0 + 2*6^1 + 5*6^2 + 1*6^3,
      artifacts := 5*6^0 + 0*6^1 + 3*6^2,
      matter := 2*6^0 + 3*6^1 + 4*6^2 + 1*6^3
  in minerals + artifacts + matter = 905 :=
by
  let minerals := 4*6^0 + 2*6^1 + 5*6^2 + 1*6^3
  let artifacts := 5*6^0 + 0*6^1 + 3*6^2
  let matter := 2*6^0 + 3*6^1 + 4*6^2 + 1*6^3
  have h1 : minerals = 412 := by sorry
  have h2 : artifacts = 113 := by sorry
  have h3 : matter = 380 := by sorry
  show minerals + artifacts + matter = 905 from by
    calc
      minerals + artifacts + matter
      = 412 + 113 + 380 : by rw [h1, h2, h3]
      = 905 : by sorry


end space_explorer_acquisition_l35_35148


namespace digit_B_in_30624B080_is_9_l35_35309

-- Defining the number of books and books to be selected
def number_of_books : ℕ := 20
def books_to_be_selected : ℕ := 8

-- The given statement of the proof problem
theorem digit_B_in_30624B080_is_9 :
  (∃ B : ℕ, 20.choose 8 = 125970 ∧ B = 9) :=
by
  sorry

end digit_B_in_30624B080_is_9_l35_35309


namespace peter_bought_8_small_glasses_l35_35780

theorem peter_bought_8_small_glasses
  (cost_small cost_large: ℕ)
  (total_money money_left: ℕ)
  (num_large_glasses: ℕ)
  (cost_large_glasses: ℕ):
  cost_small = 3 →
  cost_large = 5 →
  total_money = 50 →
  money_left = 1 →
  num_large_glasses = 5 →
  cost_large_glasses = num_large_glasses * cost_large →
  (total_money - money_left - cost_large_glasses) / cost_small = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  sorry

end peter_bought_8_small_glasses_l35_35780


namespace elevator_travel_time_l35_35773

noncomputable def total_time_in_hours (floors : ℕ) (time_first_half : ℕ) (time_next_floors_per_floor : ℕ) (next_floors : ℕ) (time_final_floors_per_floor : ℕ) (final_floors : ℕ) : ℕ :=
  let time_first_part := time_first_half
  let time_next_part := time_next_floors_per_floor * next_floors
  let time_final_part := time_final_floors_per_floor * final_floors
  (time_first_part + time_next_part + time_final_part) / 60

theorem elevator_travel_time :
  total_time_in_hours 20 15 5 5 16 5 = 2 := 
by
  sorry

end elevator_travel_time_l35_35773


namespace slope_of_tangent_line_at_origin_l35_35206

theorem slope_of_tangent_line_at_origin : 
  (deriv (λ x : ℝ, Real.exp x) 0) = 1 :=
by
  sorry

end slope_of_tangent_line_at_origin_l35_35206


namespace packages_katie_can_make_l35_35738

-- Definition of the given conditions
def number_of_cupcakes_baked := 18
def cupcakes_eaten_by_todd := 8
def cupcakes_per_package := 2

-- The main statement to prove
theorem packages_katie_can_make : 
  (number_of_cupcakes_baked - cupcakes_eaten_by_todd) / cupcakes_per_package = 5 :=
by
  -- Use sorry to skip the proof
  sorry

end packages_katie_can_make_l35_35738


namespace qin_jiushao_algorithm_v2_l35_35851

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

-- Define the value x to evaluate the polynomial at
def x0 : ℝ := -1

-- Define the intermediate value v2 according to Horner's rule
def v1 : ℝ := 2 * x0^4 - 3 * x0^3 + x0^2
def v2 : ℝ := v1 * x0 + 2

theorem qin_jiushao_algorithm_v2 : v2 = -4 := 
by 
  -- The proof will be here, for now we place sorry.
  sorry

end qin_jiushao_algorithm_v2_l35_35851


namespace classify_numbers_l35_35219

noncomputable theory

def set_of_positive_numbers : set ℝ := {x | x > 0}
def set_of_integers : set ℤ := {x | true} -- All integers
def set_of_fractions : set ℚ := {x | true} -- All fractions
def set_of_positive_integers : set ℕ := {x | x > 0}
def set_of_non_negative_rationals : set ℚ := {x | x ≥ 0}

def numbers : list ℝ := [-3, -1, 0, 20, 1/4, -6.5, 17/100, -8.5, 7, 3.14, 16, -3.14]

theorem classify_numbers :
  {x ∈ numbers | x > 0} = {20, 1/4, 17/100, 7, 16, 3.14} ∧
  {x ∈ numbers | is_int x} = {-3, -1, 0, 20, 7, 16} ∧
  {x ∈ numbers | is_rat x} = {1/4, -6.5, 17/100, -8.5, -3.14} ∧
  {x ∈ numbers | is_nat x ∧ x > 0} = {20, 7, 16} ∧
  {x ∈ numbers | rational x ∧ x ≥ 0} = {0, 20, 1/4, 17/100, 7, 16} :=
sorry

#check classify_numbers

end classify_numbers_l35_35219


namespace negation_of_proposition_l35_35826

theorem negation_of_proposition (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a * x + 1 ≥ 0) :=
by sorry

end negation_of_proposition_l35_35826


namespace points_on_line_with_slope_two_l35_35323

noncomputable def sequence (a_1 : ℝ) : ℕ → ℝ
| 0     := 0   -- This case is trivial since we start from a_1
| 1     := a_1
| (n+1) := sequence a_1 n + 2

theorem points_on_line_with_slope_two (a_1 : ℝ) :
  ∀ n : ℕ, n ≥ 1 → ∃ m b : ℝ, (sequence a_1 n) = m * (n:ℝ) + b ∧ m = 2 :=
by
  intros
  sorry

end points_on_line_with_slope_two_l35_35323


namespace ball_2_probability_given_sum_6_l35_35060

def Ball := {n : ℕ // n = 1 ∨ n = 2 ∨ n = 3}

def draw_ball (b : Ball) : ℕ := b.val

def experiment : list Ball := [⟨1, Or.inl rfl⟩, ⟨2, Or.inr (Or.inl rfl)⟩, ⟨3, Or.inr (Or.inr rfl)⟩]

def outcomes := list (list Ball)

def possible_draws : ℕ := 3

def all_possible_outcomes : list (list Ball) :=
(list.replicateM 3 experiment)

def sum_of_draws (l : list Ball) : ℕ :=
(l.map draw_ball).sum

def draws_sum_to_six : list (list Ball) := 
(filter (λ l , sum_of_draws l = 6) all_possible_outcomes)

def favorable_outcome : list Ball := [⟨2, Or.inr (Or.inl rfl)⟩, ⟨2, Or.inr (Or.inl rfl)⟩, ⟨2, Or.inr (Or.inl rfl)⟩]

def probability_event : ℚ :=
  1 / (draws_sum_to_six.length : ℚ)

theorem ball_2_probability_given_sum_6 :
  probability_event = 1 / 7 :=
sorry

end ball_2_probability_given_sum_6_l35_35060


namespace max_value_frac_c_b_b_c_l35_35306

theorem max_value_frac_c_b_b_c 
  (a b c : ℝ) 
  (h : c / 2 = b) 
  (A B C : ℝ)
  (h₁ : angle_A == A) 
  (h₂ : angle_B == B) 
  (h₃ : angle_C == C)
  (h₄ : a^2 = 2 * b * c * Real.sin A) :
  ∃ A, (A = π / 4) → ∀ b c, (c / b + b / c) ≤ 2 * sqrt 2 := by
  sorry

end max_value_frac_c_b_b_c_l35_35306


namespace cheryl_material_need_l35_35188

-- Cheryl's conditions
def cheryl_material_used (x : ℚ) : Prop :=
  x + 2/3 - 4/9 = 2/3

-- The proof problem statement
theorem cheryl_material_need : ∃ x : ℚ, cheryl_material_used x ∧ x = 4/9 :=
  sorry

end cheryl_material_need_l35_35188


namespace smallest_two_digit_product_12_l35_35472

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l35_35472


namespace arithmetic_seq_sum_l35_35938

theorem arithmetic_seq_sum :
  let a := 2
  let d := 4
  let n := (42 - 2) / 4 + 1
  (n * (a + 42)) / 2 = 242 :=
by
  let a := 2
  let d := 4
  let n := (42 - 2) / 4 + 1
  have h1 : n = 11 := by sorry
  have h2 : (n * (a + 42)) / 2 = (11 * (2 + 42)) / 2 := by sorry
  have h3 : (11 * (2 + 42)) / 2 = 242 := by sorry
  show (n * (a + 42)) / 2 = 242
  from Eq.trans h2 h3

end arithmetic_seq_sum_l35_35938


namespace area_of_triangle_ABC_eq_sqrt_3_l35_35268

theorem area_of_triangle_ABC_eq_sqrt_3 (x y : ℝ) (A B C : ℝ × ℝ) :
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 + 4*p.2 + 1 = 0 } in
  let A := (0 : ℝ, b : ℝ) in
  let B := (c : ℝ, 0 : ℝ) in
  let C := (d : ℝ, e : ℝ) in
  A ∈ circle ∧ B ∈ circle ∧ C ∈ circle →
  d = 1 ∧ e = 0 ∧ b = 2 ∧ c = 2 * sqrt 3 →
  let area_of_triangle := 1 / 2 * abs ((b - 0) * (1 - d)) in
  area_of_triangle = sqrt 3 :=
begin
  sorry
end

end area_of_triangle_ABC_eq_sqrt_3_l35_35268


namespace probability_AC_adjacent_BE_not_adjacent_l35_35424

-- Define the 5 students as elements of a set
inductive Student
| A | B | C | D | E
deriving DecidableEq

open Student

-- Define a function to count valid arrangements with given conditions
def count_valid_arrangements (students : List Student) : Nat := sorry

-- Define the total number of permutations of 5 elements
def total_permutations : Nat := 5!

-- The proof statement
theorem probability_AC_adjacent_BE_not_adjacent :
  (count_valid_arrangements [A, B, C, D, E] / (total_permutations : ℚ) = 1 / 5) :=
by
  sorry

end probability_AC_adjacent_BE_not_adjacent_l35_35424


namespace derivative_at_1_derivative_at_neg_2_derivative_at_x0_l35_35852

noncomputable def f (x : ℝ) : ℝ := 2 / x + x

theorem derivative_at_1 : (deriv f 1) = -1 :=
sorry

theorem derivative_at_neg_2 : (deriv f (-2)) = 1 / 2 :=
sorry

theorem derivative_at_x0 (x0 : ℝ) : (deriv f x0) = -2 / (x0^2) + 1 :=
sorry

end derivative_at_1_derivative_at_neg_2_derivative_at_x0_l35_35852


namespace smallest_x_satisfies_conditions_l35_35526

variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
def cond1 : Prop := ∀ x > 0, g (4 * x) = 4 * g x
def cond2 : Prop := ∀ x, 2 ≤ x ∧ x ≤ 6 → g x = 2 - |x - 4|

-- Hypothesis stating conditions
axiom h1 : cond1 g
axiom h2 : cond2 g

-- Given equation for g(2022)
def g_2022 : ℝ := 4^5 * (2 - |2022 / 4^5 - 4|)

-- Target: Find the smallest x such that g(x) = g(2022)
def is_solution (x : ℝ) : Prop := g x = g_2022

-- Assertion: The smallest x satisfying the conditions and achieving the target value
theorem smallest_x_satisfies_conditions : is_solution g 2099 :=
by 
  sorry

end smallest_x_satisfies_conditions_l35_35526


namespace total_markings_l35_35544

theorem total_markings (L : ℝ) (hL: L = 1) : 
  set.countable ({x | x ∈ {x : ℝ | ∃ (n : ℕ), x = n/3 ∧ 0 ≤ x ∧ x ≤ L} ∪ {x : ℝ | ∃ (m : ℕ), x = m/4 ∧ 0 ≤ x ∧ x ≤ L}}) = 6 :=
by
  have subset_1 : {x | x ∈ {x : ℝ | ∃ (n : ℕ), x = n/3 ∧ 0 ≤ x ∧ x ≤ L}} = {0, 1/3, 2/3, 1},
    sorry,
  have subset_2 : {x | x ∈ {x : ℝ | ∃ (m : ℕ), x = m/4 ∧ 0 ≤ x ∧ x ≤ L}} = {0, 1/4, 1/2, 3/4, 1},
    sorry,
  have all_markings := {0, 1/3, 2/3, 1} ∪ {0, 1/4, 1/2, 3/4, 1},
    sorry,
  have unique_markings := {0, 1/3, 2/3, 1, 1/4, 3/4},
    sorry,
  have count_unique : set.countable ({0, 1/3, 2/3, 1, 1/4, 3/4}) = 6, 
    sorry,
  exact count_unique

end total_markings_l35_35544


namespace general_term_and_sum_formula_smallest_n_satisfying_inequality_l35_35651

variable {a : ℕ → ℕ}
variable {S : ℕ → ℚ}
variable {b : ℕ → ℚ}

-- Given conditions
axiom arithmetic_sequence (d : ℕ) (h_d : d ≠ 0) (n : ℕ) :
  a 1 = 1 ∧ (a (n + 1) = a n + d)

axiom geometric_property (n : ℕ) :
  (a 1) * (a 4) = (a 2) ^ 2

-- Prove the general term and sum formula
theorem general_term_and_sum_formula (d : ℕ) (h_d : d ≠ 0) :
  (∀ n, a n = n) ∧ (∀ n, S n = n * (n + 1) / 2) :=
sorry

-- Given generalized sequence
def b (n : ℕ) : ℚ :=
  1 / (S n)

-- Prove the smallest n satisfying the inequality
theorem smallest_n_satisfying_inequality :
  ∃ (n : ℕ), (∑ i in Finset.range (n + 1), b i) > 9/5 ∧ n = 10 :=
sorry

end general_term_and_sum_formula_smallest_n_satisfying_inequality_l35_35651


namespace probability_first_ge_second_l35_35131

-- Define the number of faces
def faces : ℕ := 10

-- Define the total number of outcomes excluding the duplicates
def total_outcomes : ℕ := faces * faces - faces

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ := 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- The statement we want to prove
theorem probability_first_ge_second :
  probability = 11 / 18 :=
sorry

end probability_first_ge_second_l35_35131


namespace building_height_l35_35505

noncomputable def height_of_building (flagpole_height shadow_of_flagpole shadow_of_building : ℝ) : ℝ :=
  (flagpole_height / shadow_of_flagpole) * shadow_of_building

theorem building_height : height_of_building 18 45 60 = 24 := by {
  sorry
}

end building_height_l35_35505


namespace choose_rows_zero_sum_l35_35522
open Matrix

variable {n : ℕ} (h : 1 ≤ n)
variable (board : Fin (2^n) → Fin n → ℤ)

-- Assuming the conditions
axiom (unique_sequences : ∀ i : Fin (2^n), ∀ j : Fin (2^n), (i ≠ j) → (∀ k : Fin n, board i k ≠ board j k))

-- Zero replaced condition
axiom (inclusion_property : ∀ i : Fin (2^n), ∀ j : Fin n, board i j = 1 ∨ board i j = -1 ∨ board i j = 0)

theorem choose_rows_zero_sum : ∃ (chosen_rows : Finset (Fin (2^n))),
  ∀ j : Fin n, (∑ i in chosen_rows, board i j) = 0 :=
sorry

end choose_rows_zero_sum_l35_35522


namespace find_f_neg1_l35_35300

theorem find_f_neg1 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (tan x) = sin (2 * x)) : f (-1) = -1 :=
by {
  sorry
}

end find_f_neg1_l35_35300


namespace customer_paid_correct_amount_l35_35824

noncomputable def final_price_paid (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
let price_after_first_discount := original_price - (discount1 / 100) * original_price in
price_after_first_discount - (discount2 / 100) * price_after_first_discount

theorem customer_paid_correct_amount :
  final_price_paid 70 10 4.999999999999997 = 59.85 :=
by
  sorry

end customer_paid_correct_amount_l35_35824


namespace prob_AC_adjacent_BE_not_adjacent_l35_35427

open Finset
open Perm
open Probability

-- Define the students as a Finset
def students : Finset (Fin 5) := {0, 1, 2, 3, 4}

-- Define that A is 0, B is 1, C is 2, D is 3, E is 4
def A := 0
def B := 1
def C := 2
def D := 3
def E := 4

-- Define the event of A and C being adjacent
def adjacent (x y : Fin 5) (p : List (Fin 5)) : Prop := 
  (List.indexOf x p + 1 = List.indexOf y p) ∨ (List.indexOf y p + 1 = List.indexOf x p)

-- Define the event of B and E not being adjacent
def not_adjacent (x y : Fin 5) (p : List (Fin 5)) : Prop := 
  ¬ adjacent x y p

-- Lean 4 statement: Calculate the probability that A and C are adjacent while B and E are not adjacent
noncomputable def probabilityAC_adjacent_BE_not_adjacent : ℚ :=
  let total_permutations := (univ.perm 5).toFinset.card
  let valid_permutations := (univ.filter (λ p, adjacent A C p ∧ not_adjacent B E p)).perm.toFinset.card
  valid_permutations / total_permutations

theorem prob_AC_adjacent_BE_not_adjacent : probabilityAC_adjacent_BE_not_adjacent = 1/5 :=
  sorry

end prob_AC_adjacent_BE_not_adjacent_l35_35427


namespace line_OP_correct_line_intersects_circle_l35_35314

-- Define the given initial conditions
def polar_coords_M := (2 : ℝ, 0 : ℝ)
def polar_coords_N := (2 * real.sqrt 3 / 3, real.pi / 2)
def parametric_circle (theta : ℝ) := (2 + 2 * real.cos theta, -real.sqrt 3 + 2 * real.sin theta)

-- Translate to Cartesian coordinates
def cartesian_coords_M := (2 : ℝ, 0 : ℝ)
def cartesian_coords_N := (0 : ℝ, 2 * real.sqrt 3 / 3)

-- Define the midpoint P
def midpoint_P := ((cartesian_coords_M.1 + cartesian_coords_N.1) / 2, (cartesian_coords_M.2 + cartesian_coords_N.2) / 2)

-- Define line OP equation
def line_OP (x : ℝ) : ℝ := real.sqrt 3 / 3 * x

-- Define the Cartesian equation of circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y + real.sqrt 3)^2 = 4

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := real.sqrt 3 * x + 3 * y - 2 * real.sqrt 3 = 0

-- Define the distance from the center of the circle to line l
def distance_center_to_line : ℝ := abs (2 * real.sqrt 3 - 3 * real.sqrt 3 - 2 * real.sqrt 3) / real.sqrt (real.sqrt 3 ^ 2 + 3 ^ 2)

-- Proof Problem Statements

-- (1) Proof the equation of OP
theorem line_OP_correct : ∀ x : ℝ, line_OP x = real.sqrt 3 / 3 * x := by sorry

-- (2) Proof the positional relationship
theorem line_intersects_circle : ∀ (x y : ℝ), circle_C x y ∧ line_l x y → distance_center_to_line < 2 := by sorry

end line_OP_correct_line_intersects_circle_l35_35314


namespace smallest_two_digit_product_12_l35_35469

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l35_35469


namespace problem1_problem2_l35_35113

-- For problem (1)
theorem problem1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := sorry

-- For problem (2)
theorem problem2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : b^2 = a * c) :
  a^2 + b^2 + c^2 > (a - b + c)^2 := sorry

end problem1_problem2_l35_35113


namespace Joan_attended_games_l35_35734

def total_games : ℕ := 864
def games_missed_by_Joan : ℕ := 469
def games_attended_by_Joan : ℕ := total_games - games_missed_by_Joan

theorem Joan_attended_games : games_attended_by_Joan = 395 := 
by 
  -- Proof omitted
  sorry

end Joan_attended_games_l35_35734


namespace smallest_two_digit_number_product_12_l35_35478

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l35_35478


namespace sum_a1_to_a5_l35_35234

noncomputable def f (x : ℝ) : ℝ := (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5
noncomputable def g (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) : ℝ := a_0 + a_1 * (1 - x) + a_2 * (1 - x)^2 + a_3 * (1 - x)^3 + a_4 * (1 - x)^4 + a_5 * (1 - x)^5

theorem sum_a1_to_a5 (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, f x = g x a_0 a_1 a_2 a_3 a_4 a_5) →
  (f 1 = g 1 a_0 a_1 a_2 a_3 a_4 a_5) →
  (f 0 = g 0 a_0 a_1 a_2 a_3 a_4 a_5) →
  a_0 = 62 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -57 :=
by
  intro hf1 hf2 hf3 ha0 hsum
  sorry

end sum_a1_to_a5_l35_35234


namespace total_budget_is_correct_l35_35004

-- Define the costs of TV, fridge, and computer based on the given conditions
def cost_tv : ℕ := 600
def cost_computer : ℕ := 250
def cost_fridge : ℕ := cost_computer + 500

-- Statement to prove the total budget
theorem total_budget_is_correct : cost_tv + cost_computer + cost_fridge = 1600 :=
by
  sorry

end total_budget_is_correct_l35_35004


namespace find_PB_l35_35877

variable {A B C D P : Type} [EuclideanGeometry.Point A] [EuclideanGeometry.Point B] [EuclideanGeometry.Point C] [EuclideanGeometry.Point D] [EuclideanGeometry.Point P]

open EuclideanGeometry

-- Define the distances
def PA : ℝ := 5
def PD : ℝ := 12
def PC : ℝ := 13

-- Define PB as the unknown variable x
noncomputable def PB : ℝ := x

-- The goal is to prove PB = 5√2 given the distances
theorem find_PB (h1 : dist P A = 5) (h2 : dist P D = 12) (h3 : dist P C = 13) : dist P B = 5 * Real.sqrt 2 := by
  sorry

end find_PB_l35_35877


namespace wesley_breenah_ages_l35_35855

theorem wesley_breenah_ages (w b : ℕ) (h₁ : w = 15) (h₂ : b = 7) (h₃ : w + b = 22) :
  ∃ n : ℕ, 2 * (w + b) = (w + n) + (b + n) := by
  exists 11
  sorry

end wesley_breenah_ages_l35_35855


namespace factorial_simplification_l35_35184

theorem factorial_simplification :
  8.factorial - 6 * 7.factorial - 2 * 7.factorial = 0 :=
by
  sorry

end factorial_simplification_l35_35184


namespace tenth_term_of_sequence_l35_35970

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 0 then 3
else if n = 1 then 4
else 12 / sequence (n - 1)

theorem tenth_term_of_sequence :
  sequence 9 = 4 :=
sorry

end tenth_term_of_sequence_l35_35970


namespace gain_percent_is_one_l35_35109

def gain (paise: ℕ) : ℝ := paise / 100

theorem gain_percent_is_one :
  let gain := gain 70 in
  let cost_price : ℝ := 70 in
  (gain / cost_price) * 100 = 1 :=
by
  let gain := gain 70
  let cost_price : ℝ := 70
  sorry

end gain_percent_is_one_l35_35109


namespace equal_amount_deposit_l35_35812

noncomputable def P : ℝ := 148

def SI (P R T : ℝ) : ℝ := P * R * T / 100

def SI1 (P : ℝ) : ℝ := SI P 15 3.5
def SI2 (P : ℝ) : ℝ := SI P 15 10

theorem equal_amount_deposit (P: ℝ) (R T1 T2 : ℝ) 
  (cond1: R = 15) 
  (cond2: T1 = 3.5) 
  (cond3: T2 = 10)
  (cond4: SI2 P - SI1 P = 144) 
  : P = 148 := 
by 
  sorry

end equal_amount_deposit_l35_35812


namespace distance_between_parallel_tangent_lines_l35_35152

-- Given conditions
def point_M := (0, 2)
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 25
def line_l_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0
def parallel_lines (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

-- Defining the initial problem
theorem distance_between_parallel_tangent_lines :
  ∃ (l l' : ℝ → ℝ → Prop),
  l = line_l_eq 4 (-3) 6 ∧
  l' = line_l_eq 4 (-3) 2 ∧
  ∀ (x y : ℝ), circle_C x y → line_l_eq 4 (-3) 6 x y →
  dist_between_lines l l' = 4 / 5 :=
by
  -- Proof statement placeholder
  sorry

end distance_between_parallel_tangent_lines_l35_35152


namespace total_surface_area_correct_l35_35539

-- Definitions for side lengths of the cubes
def side_length_large := 5
def side_length_medium := 2
def side_length_small := 1

-- Surface area calculation for a single cube
def surface_area (side_length : ℕ) : ℕ := 6 * side_length^2

-- Surface areas for each size of the cube
def surface_area_large := surface_area side_length_large
def surface_area_medium := surface_area side_length_medium
def surface_area_small := surface_area side_length_small

-- Total surface areas for medium and small cubes
def surface_area_medium_total := 4 * surface_area_medium
def surface_area_small_total := 4 * surface_area_small

-- Total surface area of the structure
def total_surface_area := surface_area_large + surface_area_medium_total + surface_area_small_total

-- Expected result
def expected_surface_area := 270

-- Proof statement
theorem total_surface_area_correct : total_surface_area = expected_surface_area := by
  sorry

end total_surface_area_correct_l35_35539


namespace smallest_two_digit_l35_35463

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l35_35463


namespace relationship_among_a_b_c_l35_35237

noncomputable def a : ℝ := Real.logb 4 (1 / 3)
noncomputable def b : ℝ := Real.log 10 5
noncomputable def c : ℝ := ∫ x in 0..1, x

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  sorry

end relationship_among_a_b_c_l35_35237


namespace molecular_weight_CxH1O_l35_35225

theorem molecular_weight_CxH1O (x : ℤ) (hx : 0 ≤ x) :
  12.01 * x + 1.008 + 16.00 = 65 → x = 4 :=
by sorry

end molecular_weight_CxH1O_l35_35225


namespace derangements_formula_l35_35762

def derangements (n : ℕ) : ℕ :=
  n! * (Finset.range (n + 1)).sum (λ k => (-1 : ℤ)^k / k!)

theorem derangements_formula (n : ℕ) : derangements n = (Finset.range (n + 1)).sum (λ k => nat.factorial n * ((-1 : ℤ)^k / nat.factorial k)) := by
  sorry

end derangements_formula_l35_35762


namespace total_cats_l35_35774

variable (initialCats : ℝ)
variable (boughtCats : ℝ)

theorem total_cats (h1 : initialCats = 11.0) (h2 : boughtCats = 43.0) :
    initialCats + boughtCats = 54.0 :=
by
  sorry

end total_cats_l35_35774


namespace one_sided_probability_l35_35128

def regular_polygon (n : ℕ) : Prop := 2 < n

def circumscribed_around_circle (n : ℕ) : Prop := regular_polygon n

def one_sided (vertices : set (ℝ × ℝ)) : Prop := 
  ∃ (semicircle : set (ℝ × ℝ)), vertices ⊆ semicircle

theorem one_sided_probability (n : ℕ) (h1 : regular_polygon (2 * n)) 
  (h2 : circumscribed_around_circle (2 * n)) :
  let total_triplets := (2 * n) * (2 * n - 1) * (2 * n - 2) / 6 in
  let one_sided_triplets := 6 * n^2 * (n - 1) in
  (one_sided_triplets : ℚ) / total_triplets = 3 * n / (2 * (2 * n - 1)) :=
by {
  sorry
}

end one_sided_probability_l35_35128


namespace count_valid_integers_l35_35629

theorem count_valid_integers :
  {n | (1 ≤ n ∧ n ≤ 100) ∧ (∃ k : ℕ, k * ((n + 1)! ^ (n + 1)) = ((n + 1)^2 - 1)!)}.card = 97 :=
by sorry

end count_valid_integers_l35_35629


namespace range_distance_PQ_l35_35253

noncomputable def point_P (α : ℝ) : ℝ × ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α, 1)
noncomputable def point_Q (β : ℝ) : ℝ × ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β, 1)

noncomputable def distance_PQ (α β : ℝ) : ℝ :=
  Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 +
             (3 * Real.sin α - 2 * Real.sin β)^2 +
             (1 - 1)^2)

theorem range_distance_PQ : 
  ∀ α β : ℝ, 1 ≤ distance_PQ α β ∧ distance_PQ α β ≤ 5 := 
by
  intros
  sorry

end range_distance_PQ_l35_35253


namespace sqrt_of_26244_div_by_100_l35_35695

theorem sqrt_of_26244_div_by_100 (h : Real.sqrt 262.44 = 16.2) : Real.sqrt 2.6244 = 1.62 :=
sorry

end sqrt_of_26244_div_by_100_l35_35695


namespace smallest_two_digit_number_product_12_l35_35475

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l35_35475


namespace projection_matrix_onto_vector_l35_35987

theorem projection_matrix_onto_vector : 
  let v := ⟨2, -3⟩ : ℝ × ℝ,
      m := 4/13, n := -6/13, o := 9/13 in
  matrix.of (λ i j, if i = 0 then (if j = 0 then m else n) else (if j = 0 then n else o)) = 
  matrix.of (λ i j, if i = 0 then (if j = 0 then 4/13 else -6/13) else (if j = 0 then -6/13 else 9/13)) :=
by sorry

end projection_matrix_onto_vector_l35_35987


namespace percent_both_correct_l35_35863

-- Definitions of the given percentages
def A : ℝ := 75
def B : ℝ := 25
def N : ℝ := 20

-- The proof problem statement
theorem percent_both_correct (A B N : ℝ) (hA : A = 75) (hB : B = 25) (hN : N = 20) : A + B - N - 100 = 20 :=
by
  sorry

end percent_both_correct_l35_35863


namespace midpoint_reflection_sum_l35_35379

/-- 
Points P and R are located at (2, 1) and (12, 15) respectively. 
Point M is the midpoint of segment PR. 
Segment PR is reflected over the y-axis.
We want to prove that the sum of the coordinates of the image of point M (the midpoint of the reflected segment) is 1.
-/
theorem midpoint_reflection_sum : 
  let P := (2, 1)
  let R := (12, 15)
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let P_image := (-P.1, P.2)
  let R_image := (-R.1, R.2)
  let M' := ((P_image.1 + R_image.1) / 2, (P_image.2 + R_image.2) / 2)
  (M'.1 + M'.2) = 1 :=
by
  sorry

end midpoint_reflection_sum_l35_35379


namespace chess_tournament_possible_l35_35845

theorem chess_tournament_possible :
  ∃ A_wins A_points C_wins C_points ,
    A_wins < B_wins ∧ A_wins < C_wins ∧ A_points > B_points ∧ A_points > C_points ∧
    C_wins > A_wins ∧ C_wins > B_wins ∧ C_points < A_points ∧ C_points < B_points ∧
    (3*W <= 6 ∧ 3* (W_points /2) <= total_points 
    ∧ W_points = A_wins + B_wins + C_wins 
    ∧ total_points = A_points + B_points + C_points).
sorry

end chess_tournament_possible_l35_35845


namespace product_of_N1_N2_l35_35352

theorem product_of_N1_N2 :
  (∃ (N1 N2 : ℤ),
    (∀ (x : ℚ),
      (47 * x - 35) * (x - 1) * (x - 2) = N1 * (x - 2) * (x - 1) + N2 * (x - 1) * (x - 2)) ∧
    N1 * N2 = -708) :=
sorry

end product_of_N1_N2_l35_35352


namespace tye_bills_l35_35073

theorem tye_bills : 
  ∀ (total_amount withdrawn_money_per_bank bill_value number_of_banks: ℕ), 
  withdrawn_money_per_bank = 300 → 
  bill_value = 20 → 
  number_of_banks = 2 → 
  total_amount = withdrawn_money_per_bank * number_of_banks → 
  (total_amount / bill_value) = 30 :=
by
  intros total_amount withdrawn_money_per_bank bill_value number_of_banks 
  intro h_withdrawn_eq_300
  intro h_bill_eq_20
  intro h_banks_eq_2
  intro h_total_eq_mult
  have h_total_eq_600 : total_amount = 600 := by rw [h_total_eq_mult, h_withdrawn_eq_300, h_banks_eq_2]; norm_num
  have h_bills_eq_30 := h_total_eq_600.symm ▸ div_eq_of_eq_mul_left (ne_of_gt (by norm_num : 0 < 20)) (by norm_num : 600 = 20 * 30)
  exact h_bills_eq_30
  sorry

end tye_bills_l35_35073


namespace count_valid_kid_group_sizes_l35_35449

-- Definition of the problem conditions
def isValidKidGroupSize (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ (n % 7 = 0 ∨ n % 7 = 1)

-- The main theorem statement
theorem count_valid_kid_group_sizes : (Finset.filter isValidKidGroupSize (Finset.range 100)).card = 26 := 
sorry

end count_valid_kid_group_sizes_l35_35449


namespace articles_produced_by_y_men_l35_35948

-- Definitions and conditions
def articles_produced (men hours days efficiency : Nat) : Nat := men * hours * days * efficiency

-- Given condition: x men, x hours, x days -> x^2 articles
def base_production_rate (x : Nat) : Nat := x^2 / (x * x * x)

-- Given an efficiency factor for more than 10 men
def efficiency_factor (y : Nat) : Real :=
  if y <= 10 then 1
  else 0.5 * ((y - 10) / 10)

-- Given facts: x = 10 and y = 20
def x : Nat := 10
def y : Nat := 20

-- Proof statement
theorem articles_produced_by_y_men : 
  articles_produced y y y (base_production_rate x * efficiency_factor y) = 400 :=
by
  -- Required proof
  sorry

end articles_produced_by_y_men_l35_35948


namespace find_angle_DAF_l35_35730

-- Definitions of the conditions in triangle ABC
variables (A B C O D F : Type)
  [is_triangle A B C] -- assume A, B, C form a triangle
  (angleACB : angle A C B = 40) 
  (angleCBA : angle C B A = 80)
  (D_perpendicular : is_foot D A B C) -- D is the foot of the perpendicular from A to BC
  (O_circumcenter : is_circumcenter O A B C) -- O is the center of the circle circumscribed about triangle ABC
  (F_midpoint : is_midpoint F B C) -- F is the midpoint of segment BC

-- The theorem to be proved
theorem find_angle_DAF : angle D A F = 20 := by
  sorry

end find_angle_DAF_l35_35730


namespace largest_2_digit_prime_factor_of_binom_l35_35075

open Nat

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

/-- Definition of the problem conditions -/
def problem_conditions : Prop :=
  let n := binom 300 150
  ∃ p : ℕ, Prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p ≤ 75 ∨ 3 * p < 300) ∧ p = 97

/-- Statement of the proof problem -/
theorem largest_2_digit_prime_factor_of_binom : problem_conditions := 
  sorry

end largest_2_digit_prime_factor_of_binom_l35_35075


namespace range_of_a_opposite_sides_l35_35266

theorem range_of_a_opposite_sides (a : ℝ) :
  let f1 := 3 * 3 - 2 * 1 + a,
      f2 := 3 * 4 - 2 * 6 + a in
  f1 * f2 < 0 ↔ -7 < a ∧ a < 0 :=
by
  let f1 := 3 * 3 - 2 * 1 + a
  let f2 := 3 * 4 - 2 * 6 + a
  sorry

end range_of_a_opposite_sides_l35_35266


namespace line_passes_through_fixed_point_find_perimeter_of_triangle_l35_35663

theorem line_passes_through_fixed_point (a : ℝ) :
  ∃ P : ℝ × ℝ, (P = (2, 3) ∧ ∀ a : ℝ, (a + 1) * P.1 + P.2 - 5 - 2 * a = 0) :=
begin
  use (2, 3),
  split,
  { refl },
  { intro a,
    linarith }
end

theorem find_perimeter_of_triangle (a : ℝ) :
  ∃ (x_A y_B : ℝ), ((y_B = 5 + 2 * a) ∧ (x_A = (5 + 2 * a) / (a + 1))) ∧ (1 / 2 * x_A * y_B = 12)
  → (a = 1/2) ∧ (x_A = 4) ∧ (y_B = 6) ∧ (p = 10 + 2 * real.sqrt 13) :=
begin
  sorry
end

end line_passes_through_fixed_point_find_perimeter_of_triangle_l35_35663


namespace sum_of_repeating_decimals_l35_35606

theorem sum_of_repeating_decimals : (0.6666.repeating + 0.4444.repeating : ℝ) = (10 / 9 : ℝ) :=
by
  sorry

end sum_of_repeating_decimals_l35_35606


namespace eggs_collection_l35_35182

theorem eggs_collection (b : ℕ) (c : ℕ) (t : ℕ) 
  (h₁ : b = 6) 
  (h₂ : c = 3 * b) 
  (h₃ : t = b - 4) : 
  b + c + t = 26 :=
by
  sorry

end eggs_collection_l35_35182


namespace HarryWorked35_l35_35507

-- Define constants and variables
constant x : ℝ
constant HarryPay JamesPay OliviaPay TotalPay : ℝ
constant HarryHours JamesHours OliviaHours : ℝ

-- Define initial conditions
axiom JamesWorked : JamesHours = 41
axiom OliviaWorked : OliviaHours = 26
axiom TotalAmountPaid : TotalPay = 5000
axiom EqualPay : HarryPay = JamesPay

-- Define pay calculations
def JamesPayCalc : ℝ := 40 * x + (JamesHours - 40) * 2 * x
def OliviaPayCalc : ℝ := 15 * x + (OliviaHours - 15) * 2 * x

-- Define equation for total payment
axiom TotalPayEq : HarryPay + JamesPay + OliviaPay = TotalPay

-- Define equation strings for Harry's pay
axiom HarryPayEq : HarryPay = 21 * x + (HarryHours - 21) * 1.5 * x

-- Theorem to prove Harry worked 35 hours
theorem HarryWorked35 : HarryHours = 35 := by
  -- Proof required but skipped using sorry
  sorry

end HarryWorked35_l35_35507


namespace tournament_probability_l35_35516

theorem tournament_probability :
  (64 : ℕ) -> ∀ (rounds : ℕ) (winners_compete : bool) (higher_wins : bool),
  rounds = 6 → winners_compete → higher_wins → 
  (∃ (prob : ℚ), prob = 512 / 1953) :=
by
  intros teams rounds winners_compete higher_wins h1 h2 h3
  sorry

end tournament_probability_l35_35516


namespace maximum_median_sum_l35_35284

open List

def groups : List (List ℕ) := 
  [ [50, 49, 48, 47, 46],
    [45, 44, 43, 42, 41],
    [40, 39, 38, 37, 36],
    [35, 34, 33, 32, 31],
    [30, 29, 28, 27, 26],
    [25, 24, 23, 22, 21],
    [20, 19, 18, 17, 16],
    [15, 14, 13, 12, 11],
    [10, 9, 8, 7, 6],
    [5, 4, 3, 2, 1] ]

noncomputable def medians (gs : List (List ℕ)) : List ℕ := 
  gs.map (λ g => g.nthLe 2 (by linarith [g.length]))

noncomputable def sum_of_medians (gs : List (List ℕ)) : ℕ := 
  (medians gs).sum

theorem maximum_median_sum :
  sum_of_medians groups = 345 := by 
  sorry

end maximum_median_sum_l35_35284


namespace max_tulips_l35_35441

theorem max_tulips (r y : ℕ) (h₁ : r + y = 2 * (y : ℕ) + 1) (h₂ : |r - y| = 1) (h₃ : 50 * y + 31 * r ≤ 600) :
    r + y = 15 :=
sorry

end max_tulips_l35_35441


namespace sum_of_nu_lcm_18_eq_90_l35_35098

open Nat

theorem sum_of_nu_lcm_18_eq_90 (positive_ints : List ℕ) (h1 : ∀ (n : ℕ), n ∈ positive_ints → n > 0)
  (h2 : ∀ (n : ℕ), n ∈ positive_ints → lcm n 18 = 90) : 
  positive_ints.sum = 195 :=
by
  sorry

end sum_of_nu_lcm_18_eq_90_l35_35098


namespace min_g_difference_l35_35280

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + a * x - (a + 2) * Real.log x

noncomputable def g (x a b : ℝ) : ℝ :=
  f x a + (a + 4) * Real.log x - (a + 2 * b - 2) * x

theorem min_g_difference (a b x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) (h4 : b ≥ 1 + 4 * sqrt 3 / 3) :
  g x1 a b - g x2 a b = (8 / 3 - 2 * Real.log 3) :=
sorry

end min_g_difference_l35_35280


namespace find_k_l35_35680

theorem find_k
  (k : ℝ)
  (h1 : k > 1)
  (h2 : ∃ x1 y1 x2 y2, (x1 - 1)^2 + (y1 - 2)^2 = 9 ∧ (x2 - 1)^2 + (y2 - 2)^2 = 9 ∧ y1 = k * x1 + 3 ∧ y2 = k * x2 + 3)
  (h3 : dist (x1, y1) (x2, y2) = 12 * sqrt 5 / 5) :
  k = 2 :=
by
  sorry

end find_k_l35_35680


namespace four_digit_non_convertible_to_1992_multiple_l35_35592

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_multiple_of_1992 (n : ℕ) : Prop :=
  n % 1992 = 0

def reachable (n m : ℕ) (k : ℕ) : Prop :=
  ∃ x y z : ℕ, 
    x ≠ m ∧ y ≠ m ∧ z ≠ m ∧
    (n + x * 10^(k-1) + y * 10^(k-2) + z * 10^(k-3)) % 1992 = 0 ∧
    n + x * 10^(k-1) + y * 10^(k-2) + z * 10^(k-3) < 10000

theorem four_digit_non_convertible_to_1992_multiple :
  ∃ n : ℕ, is_four_digit n ∧ (∀ m : ℕ, is_four_digit m ∧ is_multiple_of_1992 m → ¬ reachable n m 3) :=
sorry

end four_digit_non_convertible_to_1992_multiple_l35_35592


namespace cost_per_person_rounded_is_8_78_l35_35736

noncomputable def total_cost_before_discount : ℝ :=
  let c_cupcakes := 3.5 * 1.50
  let c_pastries := 2.25 * 2.75
  let c_muffins := 5 * 2.10
  c_cupcakes + c_pastries + c_muffins

noncomputable def total_cost_after_discount : ℝ :=
  let discount_amount := 0.20 * total_cost_before_discount
  total_cost_before_discount - discount_amount

noncomputable def cost_per_person : ℝ :=
  total_cost_after_discount / 2

noncomputable def round_to_nearest_cent (x : ℝ) : ℝ :=
  Real.round (x * 100) / 100

theorem cost_per_person_rounded_is_8_78 : round_to_nearest_cent cost_per_person = 8.78 := by
  sorry

end cost_per_person_rounded_is_8_78_l35_35736


namespace slope_l3_l35_35365

noncomputable def slope_of_l3 : ℝ :=
  let A := (-2 : ℝ, -3 : ℝ) in
  let B := (2 : ℝ, 2 : ℝ) in
  let C := (18/5 : ℝ, 2 : ℝ) in
  let slope := (C.2 - A.2) / (C.1 - A.1) in
    slope
  
theorem slope_l3 : slope_of_l3 = 25/28 :=
sorry

end slope_l3_l35_35365


namespace smallest_two_digit_number_product_12_l35_35486

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l35_35486


namespace smallest_n_for_unity_roots_l35_35456

theorem smallest_n_for_unity_roots (n : ℕ) :
  (∀ x, (x ^ 5 - x ^ 3 + x = 0) → 
       (∃ k : ℕ, x = exp (2 * π * I * k / n))) ↔ n = 12 := 
sorry

end smallest_n_for_unity_roots_l35_35456


namespace maximum_sum_of_diagonals_of_rhombus_l35_35546

noncomputable def rhombus_side_length : ℝ := 5
noncomputable def diagonal_bd_max_length : ℝ := 6
noncomputable def diagonal_ac_min_length : ℝ := 6
noncomputable def max_diagonal_sum : ℝ := 14

theorem maximum_sum_of_diagonals_of_rhombus :
  ∀ (s bd ac : ℝ), 
  s = rhombus_side_length → 
  bd ≤ diagonal_bd_max_length → 
  ac ≥ diagonal_ac_min_length → 
  bd + ac ≤ max_diagonal_sum → 
  max_diagonal_sum = 14 :=
by
  sorry

end maximum_sum_of_diagonals_of_rhombus_l35_35546


namespace total_eggs_collected_l35_35179

def benjamin_collects : Nat := 6
def carla_collects := 3 * benjamin_collects
def trisha_collects := benjamin_collects - 4

theorem total_eggs_collected :
  benjamin_collects + carla_collects + trisha_collects = 26 := by
  sorry

end total_eggs_collected_l35_35179


namespace cylinder_surface_area_l35_35418

theorem cylinder_surface_area (l w : ℝ) (h₁ : l = 6 * real.pi) (h₂ : w = 4 * real.pi) :
  (∃ r, l = 2 * real.pi * r ∧ (4 * real.pi * l + 2 * real.pi * r^2 = 24 * real.pi^2 + 18 * real.pi)) ∨
  (∃ r, w = 2 * real.pi * r ∧ (4 * real.pi * l + 2 * real.pi * r^2 = 24 * real.pi^2 + 8 * real.pi)) :=
sorry

end cylinder_surface_area_l35_35418


namespace range_of_f_l35_35414

def cos_range : Set ℝ :=
  {y : ℝ | -1 ≤ y ∧ y ≤ 1}

def f (x : ℝ) : ℝ :=
  (2 - Real.cos x) / (2 + Real.cos x)

theorem range_of_f :
  (∀ x, Real.cos x ∈ cos_range) →
  Set.range f = Icc (1 / 3) 3 := by
  intros h_cos
  sorry

end range_of_f_l35_35414


namespace distance_from_point_to_line_eq_l35_35615

noncomputable def point_on_line (t : ℝ) : ℝ × ℝ × ℝ :=
  (-t + 4, 4 * t - 3, 3 * t + 2)

noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem distance_from_point_to_line_eq :
  distance 2 1 (-5) (fst (point_on_line (-5 / 26)))  (snd (point_on_line (-5 / 26))) zvalue = 
  Real.sqrt (34489 / 676) :=
by
  sorry

end distance_from_point_to_line_eq_l35_35615


namespace fraction_of_number_is_one_fifth_l35_35101

theorem fraction_of_number_is_one_fifth (N : ℕ) (f : ℚ) 
    (hN : N = 90) 
    (h : 3 + (1 / 2) * (1 / 3) * f * N = (1 / 15) * N) : 
  f = 1 / 5 := by 
  sorry

end fraction_of_number_is_one_fifth_l35_35101


namespace sugar_snap_peas_l35_35572

theorem sugar_snap_peas (P : ℕ) (h1 : P / 7 = 72 / 9) : P = 56 := 
sorry

end sugar_snap_peas_l35_35572


namespace evaluate_magnitude_l35_35211

theorem evaluate_magnitude (ω : ℂ) (hω : ω = 7 + 3 * complex.I) :
  complex.abs (ω^2 + 8 * ω + 85) = real.sqrt 30277 := 
sorry

end evaluate_magnitude_l35_35211


namespace max_value_x_plus_2y_l35_35254

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x * y = 4) :
  x + 2 * y ≤ 4 :=
sorry

end max_value_x_plus_2y_l35_35254


namespace ratio_of_white_marbles_l35_35997

theorem ratio_of_white_marbles (total_marbles yellow_marbles red_marbles : ℕ)
    (h1 : total_marbles = 50)
    (h2 : yellow_marbles = 12)
    (h3 : red_marbles = 7)
    (green_marbles : ℕ)
    (h4 : green_marbles = yellow_marbles - yellow_marbles / 2) :
    (total_marbles - (yellow_marbles + green_marbles + red_marbles)) / total_marbles = 1 / 2 :=
by
  sorry

end ratio_of_white_marbles_l35_35997


namespace rearrangement_impossible_l35_35416

theorem rearrangement_impossible (n : ℕ) (hn : n = 1986) :
  ¬ (∃ (a : list ℕ), a.length = 2 * n ∧ (∀ k ∈ list.range (n + 1), list.count a k = 2) ∧ ∀ k ∈ list.range (n + 1), 
    ∃ i j, i < j ∧ j - i = k + 1 ∧ a[i] = k ∧ a[j] = k) :=
by
  sorry

end rearrangement_impossible_l35_35416


namespace cross_product_antisymmetric_cross_product_scalar_multiplication_cross_product_distributive_l35_35382

variables (a b c : ℝ → ℝ → ℝ → Prop)

-- Part (a)
theorem cross_product_antisymmetric (a b : ℝ × ℝ × ℝ) : a × b = - (b × a) :=
sorry

-- Part (b)
theorem cross_product_scalar_multiplication (a b : ℝ × ℝ × ℝ) (λ μ : ℝ) : (λ • a) × (μ • b) = λ * μ • (a × b) :=
sorry

-- Part (c)
theorem cross_product_distributive (a b c : ℝ × ℝ × ℝ) : a × (b + c) = (a × b) + (a × c) :=
sorry

end cross_product_antisymmetric_cross_product_scalar_multiplication_cross_product_distributive_l35_35382


namespace find_numer_denom_n_l35_35492

theorem find_numer_denom_n (n : ℕ) 
    (h : (2 + n) / (7 + n) = (3 : ℤ) / 4) : n = 13 := sorry

end find_numer_denom_n_l35_35492


namespace multiple_of_9_is_multiple_of_3_l35_35874

theorem multiple_of_9_is_multiple_of_3 (n : ℤ) (h : ∃ k : ℤ, n = 9 * k) : ∃ m : ℤ, n = 3 * m :=
by
  sorry

end multiple_of_9_is_multiple_of_3_l35_35874


namespace factorize_expression_l35_35218

theorem factorize_expression (x y : ℝ) : 
  x^3 - x*y^2 = x * (x + y) * (x - y) :=
sorry

end factorize_expression_l35_35218


namespace marble_statue_final_weight_l35_35911

noncomputable def final_weight_after_weeks (initial_weight : ℚ) (cuts : List ℚ) : ℚ :=
  List.foldl (λ w p => w - (p / 100) * w) initial_weight cuts

theorem marble_statue_final_weight :
  final_weight_after_weeks 250 [30, 20, 25, 15, 10, 5, 3] ≈ 74.02 :=
sorry

end marble_statue_final_weight_l35_35911


namespace four_digit_odd_number_count_l35_35633

theorem four_digit_odd_number_count :
  ∃ n : ℕ, (n = 8 ∧
  ∀ (d1 d2 d3 d4 : ℕ), 
  d1 ∈ {0, 1, 2, 3} ∧ d2 ∈ {0, 1, 2, 3} ∧ d3 ∈ {0, 1, 2, 3} ∧ d4 ∈ {0, 1, 2, 3} ∧ 
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧ 
  (d1 ≠ 0 ∧ (d4 % 2 = 1 | d4 % 2 = 0)) ∧ 
  (∃ k : ℕ, k = ite (d4 % 2 = 1) 4 0)) :=
proof
  sorry
end

end four_digit_odd_number_count_l35_35633


namespace corrected_mean_l35_35866

theorem corrected_mean (n : ℕ) (mean incorrect correct : ℝ) (h : n = 50 ∧ mean = 36 ∧ incorrect = 23 ∧ correct = 48) :
  let original_sum := mean * (n : ℝ) in
  let diff := correct - incorrect in
  let corrected_sum := original_sum + diff in
  let new_mean := corrected_sum / (n : ℝ) in
  new_mean = 36.5 :=
by
  -- Assume the necessary conditions
  obtain ⟨h1, h2, h3, h4⟩ := h
  -- Use provided conditions to derive values
  let original_sum := 36 * 50
  let diff := 48 - 23
  let corrected_sum := original_sum + diff
  let new_mean := corrected_sum / 50
  -- Prove that the new mean is 36.5
  sorry

end corrected_mean_l35_35866


namespace number_of_guests_l35_35573

theorem number_of_guests (cookies_per_guest : ℕ) (total_cookies : ℕ) (H1 : cookies_per_guest = 2) (H2 : total_cookies = 10) : total_cookies / cookies_per_guest = 5 :=
by
  rw [H1, H2]
  rfl

end number_of_guests_l35_35573


namespace total_save_percentage_l35_35132

theorem total_save_percentage :
  let original_jacket_price := 80
  let original_shirt_price := 40
  let jacket_discount := 0.40
  let shirt_discount := 0.55
  let original_total_cost := original_jacket_price + original_shirt_price
  let jacket_savings := original_jacket_price * jacket_discount
  let shirt_savings := original_shirt_price * shirt_discount
  let total_savings := jacket_savings + shirt_savings
  (total_savings / original_total_cost) * 100 = 45 :=
by
  let original_jacket_price := 80
  let original_shirt_price := 40
  let jacket_discount := 0.40
  let shirt_discount := 0.55
  let original_total_cost := original_jacket_price + original_shirt_price
  let jacket_savings := original_jacket_price * jacket_discount
  let shirt_savings := original_shirt_price * shirt_discount
  let total_savings := jacket_savings + shirt_savings
  have h : (total_savings / original_total_cost) * 100 = 45 := sorry
  exact h

end total_save_percentage_l35_35132


namespace hiking_distance_l35_35931

open Real

-- Defining the given conditions
def east_distance_1 : ℝ := 4
def hypotenuse : ℝ := 6
def angle : ℝ := π / 3 -- 60 degrees in radians
def east_distance_2 : ℝ := hypotenuse / 2  -- half of the hypotenuse in a 30-60-90 triangle
def north_distance : ℝ := east_distance_2 * sqrt 3

-- Total eastward distance
def total_east_distance : ℝ := east_distance_1 + east_distance_2

-- Correct answer
theorem hiking_distance 
  : sqrt (total_east_distance^2 + north_distance^2) = 2 * sqrt 19 := sorry

end hiking_distance_l35_35931


namespace taxi_fare_distance_l35_35420

theorem taxi_fare_distance (x : ℝ) : 
  (8 + if x ≤ 3 then 0 else if x ≤ 8 then 2.15 * (x - 3) else 2.15 * 5 + 2.85 * (x - 8)) + 1 = 31.15 → x = 11.98 :=
by 
  sorry

end taxi_fare_distance_l35_35420


namespace angle_between_vectors_l35_35291

variable {α : Type*} [InnerProductSpace ℝ α]

-- Define vectors a and b
variables (a b : α)

-- Non-zero condition
variable (h₁a : a ≠ 0)
variable (h₁b : b ≠ 0)

-- Conditions given in the problem
variable (h₂ : ‖a + b‖ = 2 * ‖a‖)
variable (h₃ : ‖a - b‖ = 2 * ‖a‖)

-- The statement to be proved
theorem angle_between_vectors (h₁a : a ≠ 0) (h₁b : b ≠ 0) (h₂ : ‖a + b‖ = 2 * ‖a‖) 
  (h₃ : ‖a - b‖ = 2 * ‖a‖) :
  let θ := real.arccos (((a + b) ⬝ (b - a)) / (‖a + b‖ * ‖b - a‖))
  in θ = real.pi / 3 :=
sorry

end angle_between_vectors_l35_35291


namespace polynomial_is_constant_l35_35519

-- Definitions of ℝ and the polynomial mapping P from ℝ × ℝ → ℝ × ℝ
noncomputable def P : ℝ × ℝ → ℝ × ℝ :=
  sorry

-- Conditions
axiom P_polynomial : polynomial ℝ (P x y)
axiom P_condition : ∀ (x y : ℝ), P (x, y) = P (x + y, x - y)

-- Theorem to be proved
theorem polynomial_is_constant :
  ∃ (a b : ℝ), ∀ (x y : ℝ), P (x, y) = (a, b) :=
sorry

end polynomial_is_constant_l35_35519


namespace worker_efficiency_l35_35561

theorem worker_efficiency (Wq : ℝ) (x : ℝ) : 
  (1.4 * (1 / x) = 1 / (1.4 * x)) → 
  (14 * (1 / x + 1 / (1.4 * x)) = 1) → 
  x = 24 :=
by
  sorry

end worker_efficiency_l35_35561


namespace algebraic_expression_evaluation_l35_35660

open Real

noncomputable def x : ℝ := 2 - sqrt 3

theorem algebraic_expression_evaluation :
  (7 + 4 * sqrt 3) * x^2 - (2 + sqrt 3) * x + sqrt 3 = 2 + sqrt 3 :=
by
  sorry

end algebraic_expression_evaluation_l35_35660


namespace cost_of_soccer_ball_l35_35521

theorem cost_of_soccer_ball
  (F S : ℝ)
  (h1 : 3 * F + S = 155)
  (h2 : 2 * F + 3 * S = 220) :
  S = 50 :=
sorry

end cost_of_soccer_ball_l35_35521


namespace find_k_l35_35037

noncomputable def a_squared : ℝ := 9
noncomputable def b_squared (k : ℝ) : ℝ := 4 + k
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

noncomputable def c_squared_1 (k : ℝ) : ℝ := 5 - k
noncomputable def c_squared_2 (k : ℝ) : ℝ := k - 5

theorem find_k (k : ℝ) :
  (eccentricity (Real.sqrt (c_squared_1 k)) (Real.sqrt a_squared) = 4 / 5 →
   k = -19 / 25) ∨ 
  (eccentricity (Real.sqrt (c_squared_2 k)) (Real.sqrt (b_squared k)) = 4 / 5 →
   k = 21) :=
sorry

end find_k_l35_35037


namespace num_of_sets_l35_35287

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B : Set ℕ := {x | 0 < x ∧ x < 5}

lemma count_sets (C : Set ℕ) : A ⊆ C → C ⊆ B → (C = {1, 2} ∨ C = {1, 2, 3} ∨ C = {1, 2, 4} ∨ C = {1, 2, 3, 4}) :=
by sorry

theorem num_of_sets : (∃ (C : Set ℕ), A ⊆ C ∧ C ⊆ B) ↔ 4 :=
by {
  have h1 : A = {1, 2},
  { ext, split; intro h,
    { simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff] at h,
      rcases h with rfl | rfl;
      simp },
    { simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff],
      intro hx, fin_cases hx; simp },
  },
  have h2 : B = {1, 2, 3, 4},
  { ext, split; intro h,
    { simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff] at h,
      finish },
    { simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff],
      finish },
  },
  use 4,
  split,
  { intro h,
    cases h with C hC,
    have hC' := count_sets C, exact sorry },
  { intro h, sorry }
}

end num_of_sets_l35_35287


namespace problem_l35_35993

theorem problem (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h1 : p + 5 < q)
  (h2 : (p + (p + 2) + (p + 5) + q + (q + 1) + (2 * q - 1)) / 6 = q)
  (h3 : (p + 5 + q) / 2 = q) : p + q = 11 :=
by sorry

end problem_l35_35993


namespace yellow_ball_probability_l35_35950

theorem yellow_ball_probability :
  let X_blue := 7
  let X_yellow := 3
  let Y_blue := 5
  let Y_yellow := 5
  let Z_blue := 8
  let Z_yellow := 2
  let P_X := (X_yellow : ℚ) / (X_blue + X_yellow)
  let P_Y := (Y_yellow : ℚ) / (Y_blue + Y_yellow)
  let P_Z := (Z_yellow : ℚ) / (Z_blue + Z_yellow)
  let P_Yellow := (1 / 3) * P_X + (1 / 3) * P_Y + (1 / 3) * P_Z
  in P_Yellow = (1 / 3) :=
by
  sorry

end yellow_ball_probability_l35_35950


namespace find_a_b_and_tangent_lines_l35_35264

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 1

theorem find_a_b_and_tangent_lines (a b : ℝ) :
  (3 * (-2 / 3)^2 + 2 * a * (-2 / 3) + b = 0) ∧
  (3 * 1^2 + 2 * a * 1 + b = 0) →
  a = -1 / 2 ∧ b = -2 ∧
  (∀ t : ℝ, f t a b = (t^3 + (a - 1 / 2) * t^2 - 2 * t + 1) → 
     (f t a b - (3 * t^2 - t - 2) * (0 - t) = 1) →
       (3 * t^2 - t - 2 = (t * (3 * (t - t))) ) → 
          ((2 * 0 + f 0 a b) = 1) ∨ (33 * 0 + 16 * 1 - 16 = 1)) :=
sorry

end find_a_b_and_tangent_lines_l35_35264


namespace max_ice_creams_l35_35501

theorem max_ice_creams (total_budget : ℕ) (pancake_cost : ℕ) (ice_cream_cost : ℕ) (num_pancakes : ℕ) :
  total_budget = 60 → pancake_cost = 5 → ice_cream_cost = 8 → num_pancakes = 5 →
  ∃ (max_ice_creams : ℕ), max_ice_creams = 4 ∧ 8 * max_ice_creams + 5 * 5 ≤ 60 :=
by
  intros
  use 4
  split
  · rfl
  · sorry

end max_ice_creams_l35_35501


namespace population_size_in_15th_year_l35_35046

theorem population_size_in_15th_year
  (a : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = a * Real.logb 2 (x + 1))
  (h2 : y 1 = 100) :
  y 15 = 400 :=
by
  sorry

end population_size_in_15th_year_l35_35046


namespace journey_speed_first_half_l35_35542

noncomputable def speed_first_half (total_time : ℝ) (total_distance : ℝ) (second_half_speed : ℝ) : ℝ :=
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let second_half_time := second_half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  first_half_distance / first_half_time

theorem journey_speed_first_half
  (total_time : ℝ) (total_distance : ℝ) (second_half_speed : ℝ)
  (h1 : total_time = 10)
  (h2 : total_distance = 224)
  (h3 : second_half_speed = 24) :
  speed_first_half total_time total_distance second_half_speed = 21 := by
  sorry

end journey_speed_first_half_l35_35542


namespace binom_20_10_eq_184756_l35_35681

theorem binom_20_10_eq_184756 
  (h1 : Nat.choose 19 9 = 92378)
  (h2 : Nat.choose 19 10 = Nat.choose 19 9) : 
  Nat.choose 20 10 = 184756 := 
by
  sorry

end binom_20_10_eq_184756_l35_35681


namespace smallest_two_digit_number_product_12_l35_35479

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l35_35479


namespace total_number_of_coins_l35_35891

/--
A man has $0.70 in dimes and nickels, and exactly 2 nickels.
Prove that the total number of coins he has is 8.
-/
theorem total_number_of_coins (d n : ℕ) (hv : 0.10 * d + 0.05 * n = 0.70) (hn : n = 2) : d + n = 8 := 
by 
  sorry

end total_number_of_coins_l35_35891


namespace tetrahedron_properties_l35_35870

def point := (ℝ × ℝ × ℝ)

noncomputable def vector_from (A B : point) : point :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

noncomputable def cross_product (u v : point) : point :=
  (u.2 * v.3 - u.3 * v.2, 
   u.3 * v.1 - u.1 * v.3, 
   u.1 * v.2 - u.2 * v.1)

noncomputable def dot_product (u v : point) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude (v : point) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def volume_tetrahedron (A1 A2 A3 A4 : point) : ℝ :=
  let a := vector_from A1 A2
  let b := vector_from A1 A3
  let c := vector_from A1 A4
  (1 / 6) * Real.abs (dot_product a (cross_product b c))

noncomputable def area_triangle (A1 A2 A3 : point) : ℝ :=
  let a := vector_from A1 A2
  let b := vector_from A1 A3
  (1 / 2) * magnitude (cross_product a b)

theorem tetrahedron_properties :
  let A1 := (2, -1, -2)
  let A2 := (1, 2, 1)
  let A3 := (5, 0, -6)
  let A4 := (-10, 9, -7)
  volume_tetrahedron A1 A2 A3 A4 = 140 / 3 ∧
  (3 * volume_tetrahedron A1 A2 A3 A4) / area_triangle A1 A2 A3 = 4 * Real.sqrt 14 :=
by
  let A1 := (2, -1, -2)
  let A2 := (1, 2, 1)
  let A3 := (5, 0, -6)
  let A4 := (-10, 9, -7)
  have vol := volume_tetrahedron A1 A2 A3 A4
  have area := area_triangle A1 A2 A3
  sorry

end tetrahedron_properties_l35_35870


namespace correct_statement_l35_35384

open Classical

noncomputable theory

def line := ℝ → ℝ

def hasAngleOfInclination (l : line) : Prop :=
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < π

def hasSlope (l : line) : Prop :=
  ∃ m : ℝ, l = λ x, m * x

theorem correct_statement (l : line) :
  (hasAngleOfInclination l ∧ ¬ hasSlope l) ∨ (hasAngleOfInclination l ∧ hasSlope l) :=
by
  sorry

end correct_statement_l35_35384


namespace sequence_non_periodic_l35_35246

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1
  else (n - 1) * Real.sin (sequence (n - 1)) + 1

theorem sequence_non_periodic : ¬ ∃ T > 0, ∀ n ≥ T, sequence n = sequence (n + T) ∧ sequence (n + 1) = sequence (n + T + 1) :=
sorry

end sequence_non_periodic_l35_35246


namespace log_base_4_of_4096_l35_35599

theorem log_base_4_of_4096 : Real.log 4096 / Real.log 4 = 6 := sorry

end log_base_4_of_4096_l35_35599


namespace area_triangle_DOB_l35_35014

variables {k p : ℝ}
variables (O D B : ℝ × ℝ)

-- Define the points
def O := (0, 0)
def B := (24, 0)
def D := (0, k * p)

-- Condition on k
axiom h_k : 0 < k ∧ k < 1

-- The area of triangle DOB should be 12 * k * p
theorem area_triangle_DOB (hD: D = (0, k * p)) : 
  let base := 24 in 
  let height := k * p in 
  (1 / 2) * base * height = 12 * k * p :=
by 
  sorry

end area_triangle_DOB_l35_35014


namespace cranberries_left_l35_35172

def initial_cranberries : ℕ := 60000
def harvested_by_humans : ℕ := initial_cranberries * 40 / 100
def eaten_by_elk : ℕ := 20000

theorem cranberries_left (c : ℕ) : c = initial_cranberries - harvested_by_humans - eaten_by_elk → c = 16000 := by
  sorry

end cranberries_left_l35_35172


namespace even_power_divisible_l35_35439

theorem even_power_divisible (x y : ℤ) (n : ℕ) (hn : n % 2 = 0) : x^n - y^n = (x + y) * k
by sorry

end even_power_divisible_l35_35439


namespace smallest_two_digit_l35_35460

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l35_35460


namespace triangle_equality_statement_l35_35422

-- Definitions corresponding to the conditions
structure Point where
  x : ℝ
  y : ℝ

def triangle (A B C : Point) : Prop := 
  ¬ collinear A B C

def equilateral (A B C : Point) : Prop := 
  (dist A B = dist B C) ∧ (dist B C = dist C A)

def segment_contains (A C E : Point) : Prop := 
  ∃ λ, 0 ≤ λ ∧ λ ≤ 1 ∧ C.x = A.x + λ * (E.x - A.x) ∧ C.y = A.y + λ * (E.y - A.y)

def same_side (B D A E : Point) : Prop := 
  (B.y - A.y) * (E.x - A.x) = (B.x - A.x) * (E.y - A.y) ∧
  (D.y - A.y) * (E.x - A.x) = (D.x - A.x) * (E.y - A.y)

def circumcircle (A B C O : Point) : Prop := 
  equilateral A B C -- simplification, O is the circumcenter iff A, B, C are equilateral points

def meets_at_second (F O₁ O₂ : Point) : Prop := sorry -- definition placeholder

def intersects (line₁ line₂ : Set Point) (K : Point) : Prop := 
  K ∈ line₁ ∩ line₂

def line (A B: Point) : Set Point := sorry -- definition placeholder

-- Lean statement for the proof problem
theorem triangle_equality_statement
  (A B C D E F K O₁ O₂ : Point)
  (h1 : segment_contains A C E)
  (h2 : same_side B D A E)
  (h3 : circumcircle A B C O₁ ∧ circumcircle C D E O₂) 
  (h4 : meets_at_second F O₁ O₂)
  (h5 : intersects (line A D) (line O₁ O₂) K) :
  dist A K = dist B F :=
sorry

end triangle_equality_statement_l35_35422


namespace smallest_two_digit_number_product_12_l35_35476

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l35_35476


namespace city_mpg_l35_35504

-- Define the conditions
variables {T H C : ℝ}
axiom cond1 : H * T = 560
axiom cond2 : (H - 6) * T = 336

-- The formal proof goal
theorem city_mpg : C = 9 :=
by
  have h1 : H = 560 / T := by sorry
  have h2 : (560 / T - 6) * T = 336 := by sorry
  have h3 : C = H - 6 := by sorry
  have h4 :  C = 9 := by sorry
  exact h4

end city_mpg_l35_35504


namespace heptagon_diagonals_divide_into_triangles_l35_35393

theorem heptagon_diagonals_divide_into_triangles :
  let n := 7 in n - 2 = 5 :=
by
  sorry

end heptagon_diagonals_divide_into_triangles_l35_35393


namespace number_of_correct_operations_l35_35413

-- Define the four assertions as hypotheses
def assert1 (x : ℝ) : Prop := deriv (deriv (λ x, x^2 * real.cos x)) x = -2 * x * real.sin x
def assert2 (x : ℝ) : Prop := deriv (deriv (λ x, 3^x)) x = 3^x * real.log 3
def assert3 (x : ℝ) : Prop := deriv (λ x, real.log x / real.log 10) x = 1 / (x * real.log 10)
def assert4 (x : ℝ) : Prop := deriv (λ x, real.exp x / x) x = (real.exp x + x * real.exp x) / x^2

-- A theorem stating that all the above assertions are false
theorem number_of_correct_operations : (¬∃ x : ℝ, assert1 x) ∧ (¬∃ x : ℝ, assert2 x) ∧ (¬∃ x : ℝ, assert3 x) ∧ (¬∃ x : ℝ, assert4 x) → (∀ (x : ℝ), 0 = 0) :=
by
  intro h
  trivial

end number_of_correct_operations_l35_35413


namespace leaha_can_determine_positions_l35_35294

open Finset

theorem leaha_can_determine_positions (nums : Finset ℕ) (sum_rectangles : Finset (Finset (Fin 8 × Fin 8)) → ℕ)
  (cells : Fin 8 × Fin 8)
  (h1 : nums = {1, 2, 3, ..., 63, 64})
  (h2 : ∀ r1 r2 : Fin 8 × Fin 8, r1 ≠ r2 → sum_rectangles ({r1, r2}) = sum_rectangles ({r1, r2}))
  (h3 : ∃ d, ((1 : ℕ), 64) ∈ d) :
  ∃ f : Fin 8 × Fin 8 → ℕ, bijective f ∧ (∀ i, f i ∈ nums) :=
sorry

end leaha_can_determine_positions_l35_35294


namespace smallest_n_is_12_l35_35459

noncomputable def smallest_n : ℕ :=
  Inf {n : ℕ | ∀ z : ℂ, (z^5 - z^3 + z = 0) → ∃ k : ℕ, z = e^(2 * π * complex.I * k / n) }

theorem smallest_n_is_12 : smallest_n = 12 :=
sorry

end smallest_n_is_12_l35_35459


namespace part1_part2_l35_35019

noncomputable def chi_square_stat (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

def is_related_to_age (a b c d : ℕ) (alpha : ℚ) : Prop :=
  chi_square_stat a b c d > alpha

theorem part1 {a b c d : ℕ} (ha : a = 5) (hb : b = 15) (hc : c = 55) (hd : d = 25)
  (valid_table : (a + c = 60) ∧ (b + d = 40) ∧ (a + b + c + d = 100))
  (chi_crit : 10.828 < chi_square_stat a b c d) :
  is_related_to_age a b c d 10.828 :=
by sorry

noncomputable def expectation_X (prob : ℚ) (n : ℕ) : ℚ :=
  Σ i in finset.range (n + 1), (i : ℚ) * (nat.choose n i) * prob^i * (1 - prob)^(n - i)

theorem part2 (p : ℚ) (X : list ℚ) (expected : ℚ) (hX : X = [0.001, 0.027, 0.243, 0.729])
  (H : expected = expectation_X 0.9 3) :
  X = [0.001, 0.027, 0.243, 0.729] ∧ expected = 2.7 :=
by sorry

end part1_part2_l35_35019


namespace find_other_root_l35_35263

variable {m : ℝ} -- m is a real number
variable (x : ℝ)

theorem find_other_root (h : x^2 + m * x - 5 = 0) (hx1 : x = -1) : x = 5 :=
sorry

end find_other_root_l35_35263


namespace inscribed_semicircle_radius_l35_35728

theorem inscribed_semicircle_radius 
  (PQ QR: ℝ) 
  (hPQ: PQ = 15) 
  (hQR: QR = 8) 
  (angle_right: ∃ P Q R. ∠ Q = π / 2) :
  ∃ r: ℝ, r = 3 := 
begin
  -- Define variables
  let P : point := {x := 0, y := 0}, 
  let Q : point := {x := 15, y := 0}, 
  let R : point := {x := 15, y := 8},
  -- Calculate hypotenuse
  have PR := sqrt (PQ^2 + QR^2), 
  have hPR: PR = 17,  
  -- Calculate area
  have area := 0.5 * (PQ * QR), 
  have h_area : area = 60, 
  -- Calculate semiperimeter
  have semiperimeter := (PQ + QR + PR) / 2, 
  have h_semiperimeter : semiperimeter = 20, 
  -- Calculate radius
  have inradius := area / semiperimeter,
  -- Assert radius is 3
  use 3,
  show 3 = 3,
  sorry
end

end inscribed_semicircle_radius_l35_35728


namespace probability_of_event_correct_l35_35020

noncomputable def probability_event : ℝ := (3 : ℝ) / 8

theorem probability_of_event_correct: 
  (probability (λ (x y z : ℝ), x ∈ Icc (-1:ℝ) 1 ∧ y ∈ Icc (-1:ℝ) 1 ∧ z ∈ Icc (-1:ℝ) 1 ∧ 
                 |x| + |y| + |z| + |x + y + z| = |x + y| + |y + z| + |z + x|)) = probability_event :=
sorry

end probability_of_event_correct_l35_35020


namespace triangle_angles_l35_35157

theorem triangle_angles (A B C : Type*) 
  (area_ABC : ℝ)
  (AH_is_altitude : Prop)
  (M_is_midpoint_BC : Prop)
  (K_is_angle_bisector : Prop)
  (area_AHM : ℝ)
  (area_AKM : ℝ) :
  area_ABC = 1 ∧ AH_is_altitude ∧ M_is_midpoint_BC ∧ 
  K_is_angle_bisector ∧ area_AHM = 1 / 4 ∧ 
  area_AKM = 1 - real.sqrt 3 / 2 → 
  -- Angles A, B, C in degrees
  A = 90 ∧ B = 30 ∧ C = 60 := by
  sorry

end triangle_angles_l35_35157


namespace success_rate_increase_24_percentage_points_l35_35597

-- Defining the conditions
def initial_attempts : ℕ := 20
def initial_successes : ℕ := 8
def next_attempts : ℕ := 30
def next_success_rate : ℚ := 4/5

-- Defining the total successes and attempts based on the conditions
def total_successes : ℕ := initial_successes + (next_success_rate * next_attempts).toNat
def total_attempts : ℕ := initial_attempts + next_attempts

-- Initial and new success rates
def initial_success_rate : ℚ := initial_successes / initial_attempts
def new_success_rate : ℚ := total_successes / total_attempts

-- Increase in success rate
def increase_in_success_rate : ℚ := new_success_rate - initial_success_rate

-- Stating the main theorem to be proved
theorem success_rate_increase_24_percentage_points :
  increase_in_success_rate * 100 = 24 := sorry

end success_rate_increase_24_percentage_points_l35_35597


namespace blue_tiles_in_45th_row_l35_35916

theorem blue_tiles_in_45th_row :
  ∀ (n : ℕ), n = 45 → (∃ r b : ℕ, (r + b = 2 * n - 1) ∧ (r > b) ∧ (r - 1 = b)) → b = 44 :=
by
  -- Skipping the proof with sorry to adhere to instruction
  sorry

end blue_tiles_in_45th_row_l35_35916


namespace calculate_expression_l35_35939

theorem calculate_expression (b : ℝ) (hb : b ≠ 0) : 
  (1 / 25) * b^0 + (1 / (25 * b))^0 - 81^(-1 / 4 : ℝ) - (-27)^(-1 / 3 : ℝ) = 26 / 25 :=
by sorry

end calculate_expression_l35_35939


namespace ball_sequences_l35_35634

theorem ball_sequences (n m : ℕ) (h_n : n = 8) (h_m : m = 5) :
  (Nat.factorial (n + m)) / (Nat.factorial n * Nat.factorial m) = 1287 :=
by
  rw [h_n, h_m]
  exact congr_arg2 (÷) (factorial_add 8 5) (congr_arg2 (*) (Nat.factorial_def 8) (Nat.factorial_def 5))
  -- computation details would go here
  sorry

end ball_sequences_l35_35634


namespace correct_calculation_l35_35861

theorem correct_calculation (x : ℝ) :
  (x / 5 + 16 = 58) → (x / 15 + 74 = 88) :=
by
  sorry

end correct_calculation_l35_35861


namespace twelve_integers_divisible_by_eleven_l35_35644

theorem twelve_integers_divisible_by_eleven (a : Fin 12 → ℤ) : 
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
by
  sorry

end twelve_integers_divisible_by_eleven_l35_35644


namespace max_tulips_count_l35_35445

theorem max_tulips_count : ∃ (r y n : ℕ), 
  n = r + y ∧ 
  n % 2 = 1 ∧ 
  |r - y| = 1 ∧ 
  50 * y + 31 * r ≤ 600 ∧ 
  n = 15 := 
by
  sorry

end max_tulips_count_l35_35445


namespace not_factorial_tails_1991_l35_35954

noncomputable def factorial_trailing_zeros (m : ℕ) : ℕ :=
  (List.range (Nat.log 5 (m + 1))).map (λ k => m / 5 ^ (k + 1)).sum

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, factorial_trailing_zeros m = n

def count_not_factorial_tails (n : ℕ) : ℕ :=
  (List.range n).filter (λ x => ¬is_factorial_tail x).length

theorem not_factorial_tails_1991 : count_not_factorial_tails 1992 = 396 :=
by
  sorry

end not_factorial_tails_1991_l35_35954


namespace max_tulips_l35_35448

theorem max_tulips (r y n : ℕ) (hr : r = y + 1) (hc : 50 * y + 31 * r ≤ 600) :
  r + y = 2 * y + 1 → n = 2 * y + 1 → n = 15 :=
by
  -- Given the constraints, we need to identify the maximum n given the costs and relationships.
  sorry

end max_tulips_l35_35448


namespace winston_cents_left_l35_35498

-- Definitions based on the conditions in the problem
def quarters := 14
def cents_per_quarter := 25
def half_dollar_in_cents := 50

-- Formulation of the problem statement in Lean
theorem winston_cents_left : (quarters * cents_per_quarter) - half_dollar_in_cents = 300 :=
by sorry

end winston_cents_left_l35_35498


namespace range_of_a_l35_35276

open Set

theorem range_of_a (a : ℝ) (h : ∃ x_0 : ℝ, x_0 ∈ Ioo (-1 : ℝ) 1 ∧ f a x_0 = 0) :
  a ∈ Iio (-3) ∪ Ioi 1 := by
  sorry

def f (a x : ℝ) : ℝ := 2 * a * x - a + 3

end range_of_a_l35_35276


namespace no_real_roots_of_quadratic_l35_35832

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

theorem no_real_roots_of_quadratic :
  let a := 2
  let b := -5
  let c := 6
  discriminant a b c < 0 → ¬∃ x : ℝ, 2 * x ^ 2 - 5 * x + 6 = 0 :=
by {
  -- Proof skipped
  sorry
}

end no_real_roots_of_quadratic_l35_35832


namespace coins_problem_l35_35529

theorem coins_problem : ∃ n : ℕ, (n % 8 = 6) ∧ (n % 9 = 7) ∧ (n % 11 = 8) :=
by {
  sorry
}

end coins_problem_l35_35529


namespace piravena_total_distance_l35_35781

-- Define distances in km
def CA : ℝ := 3000
def AB : ℝ := 3250

-- Define BC using the Pythagorean Theorem
def BC : ℝ := real.sqrt (AB ^ 2 - CA ^ 2)

-- Define the total distance of Piravena's trip
def total_distance : ℝ := AB + BC + CA

-- State the theorem
theorem piravena_total_distance : total_distance = 7500 :=
by 
  have h_bc : BC = 1250 := by 
    calc
      BC = real.sqrt (3250 ^ 2 - 3000 ^ 2) : rfl
      ... = real.sqrt (10562500 - 9000000) : by norm_num
      ... = real.sqrt (1562500) : by norm_num
      ... = 1250 : by norm_num
  calc
    total_distance 
      = 3250 + 1250 + 3000 : by rw h_bc
      ... = 7500 : by norm_num

end piravena_total_distance_l35_35781


namespace shaded_area_fraction_l35_35776

theorem shaded_area_fraction (total_grid_squares : ℕ) (number_1_squares : ℕ) (number_9_squares : ℕ) (number_8_squares : ℕ) (partial_squares_1 : ℕ) (partial_squares_2 : ℕ) (partial_squares_3 : ℕ) :
  total_grid_squares = 18 * 8 →
  number_1_squares = 8 →
  number_9_squares = 15 →
  number_8_squares = 16 →
  partial_squares_1 = 6 →
  partial_squares_2 = 6 →
  partial_squares_3 = 8 →
  (2 * (number_1_squares + number_9_squares + number_9_squares + number_8_squares) + (partial_squares_1 + partial_squares_2 + partial_squares_3)) = 2 * (74 : ℕ) →
  (74 / 144 : ℚ) = 37 / 72 :=
by
  intros _ _ _ _ _ _ _ _
  sorry

end shaded_area_fraction_l35_35776


namespace unique_triple_l35_35198

theorem unique_triple (a b p : ℕ) (hp : p.prime) (h_positive : a > 0 ∧ b > 0 ∧ p > 0)
    (h_eqn : (a + b) ^ p = p ^ a + p ^ b) : (a, b, p) = (1, 1, 2) := by
  sorry

end unique_triple_l35_35198


namespace expected_value_biased_die_l35_35881

theorem expected_value_biased_die :
  let P : ℕ → ℝ := λ n, 
    if n = 1 ∨ n = 2 then 1/4 else 
    if n = 3 ∨ n = 4 then 1/6 else 
    if n = 5 ∨ n = 6 then 1/12 else 0,
  let earning : ℕ → ℝ := λ n, 
    if n = 1 ∨ n = 2 then 4 else 
    if n = 3 ∨ n = 4 then -3 else 
    if n = 5 ∨ n = 6 then 0 else 0,
  (P 1 * earning 1 + P 2 * earning 2 +
   P 3 * earning 3 + P 4 * earning 4 +
   P 5 * earning 5 + P 6 * earning 6) = 1 := 
by {
  let probs := [P 1, P 2, P 3, P 4, P 5, P 6],
  let earn := [earning 1, earning 2, earning 3, earning 4, earning 5, earning 6],
  let sum : ℝ := List.sum (List.zipWith (*) probs earn),
  exact (sum = 1),
  sorry
}

end expected_value_biased_die_l35_35881


namespace smallest_two_digit_l35_35465

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l35_35465


namespace find_angles_l35_35714

theorem find_angles : 
  ∃ (a b : ℝ), 
  (b = 4 * a - 30) ∧ ((a + b = 180 ∧ a = 42 ∧ b = 38) ∨ (a = b ∧ a = 10 ∧ b = 10)) :=
begin
  sorry
end

end find_angles_l35_35714


namespace smallest_two_digit_number_product_12_l35_35487

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l35_35487


namespace solve_equation_l35_35359

noncomputable def frac_part (x : ℝ) : ℝ := x - floor x

theorem solve_equation (x : ℝ) (h : ⌊x⌋^4 + (frac_part x)^4 + x^4 = 2048) : x = 2 :=
by
  sorry

end solve_equation_l35_35359


namespace prism_distances_l35_35717

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def dist (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

theorem prism_distances :
  let A := Point3D.mk 0 0 0
  let B := Point3D.mk 1 0 0
  let C := Point3D.mk 0 1 0
  let A' := Point3D.mk 0 0 2
  let C' := Point3D.mk 0 1 2
  let B' := Point3D.mk 1 0 2 in
  dist A C' = Real.sqrt 5 ∧ dist B' C = Real.sqrt 6 :=
by
  sorry

end prism_distances_l35_35717


namespace largest_prime_factor_of_binomial_l35_35086

theorem largest_prime_factor_of_binomial :
  ∃ p : ℕ, p.prime ∧ 10 ≤ p ∧ p < 100 ∧ (∃ k : ℕ, p ^ k ∣ (300.factorial / (150.factorial * 150.factorial))) ∧
  (∀ q : ℕ, q.prime → 10 ≤ q ∧ q < 100 → (∃ k : ℕ, q ^ k ∣ (300.factorial / (150.factorial * 150.factorial))) → q ≤ p) :=
sorry

end largest_prime_factor_of_binomial_l35_35086


namespace largest_circle_area_proof_l35_35545

-- Define the perimeter and area conditions of the rectangle
variables (P : ℝ) (A : ℝ)
variables (x y : ℝ)
variable (r : ℝ)

-- Assume the perimeter equals the length of the string
axiom perimeter_def : P = 2 * x + 2 * y

-- Assume the area of the rectangle is 200
axiom area_def : x * y = A
axiom area_200 : A = 200

-- The perimeter P is used as the circumference of the circle
axiom circumference_def : P = 2 * π * r

-- Define the area of the circle
def circle_area := π * r^2

-- Prove that the area of the largest circle formed from the string, rounded to the nearest whole number, is 255
theorem largest_circle_area_proof (P A x y r : ℝ) 
  (perimeter_def : P = 2 * x + 2 * y)
  (area_def : x * y = A)
  (area_200 : A = 200)
  (circumference_def : P = 2 * π * r)
  (circle_area := π * r^2) :
  Real.round (circle_area r) = 255 := 
sorry

end largest_circle_area_proof_l35_35545


namespace area_of_equilateral_triangle_with_given_conditions_l35_35854

-- Condition definitions
def is_equilateral (A B C : ℝ) : Prop :=
A = B ∧ B = C

def is_altitude (A K B C : ℝ) : Prop :=
A * A + K * K = B * B

-- Main problem statement
theorem area_of_equilateral_triangle_with_given_conditions :
  ∀ (A B C K : ℝ),
    A = 12 ∧ B = 3 ∧ C = 12 ∧ K = 3 → is_equilateral A C B →
    is_altitude (A / 2) K C →
    (1 / 2) * C * (3 * Real.sqrt 3) = 18 * Real.sqrt 3 :=
by
  intros A B C K hA hB hC hK h_equilateral h_altitude
  sorry

end area_of_equilateral_triangle_with_given_conditions_l35_35854


namespace second_divisor_correct_l35_35625

noncomputable def smallest_num: Nat := 1012
def known_divisors := [12, 18, 21, 28]
def lcm_divisors: Nat := 252 -- This is the LCM of 12, 18, 21, and 28.
def result: Nat := 14

theorem second_divisor_correct :
  ∃ (d : Nat), d ≠ 12 ∧ d ≠ 18 ∧ d ≠ 21 ∧ d ≠ 28 ∧ d ≠ 252 ∧ (smallest_num - 4) % d = 0 ∧ d = result :=
by
  sorry

end second_divisor_correct_l35_35625


namespace fifteen_times_number_eq_150_l35_35116

theorem fifteen_times_number_eq_150 (n : ℕ) (h : 15 * n = 150) : n = 10 :=
sorry

end fifteen_times_number_eq_150_l35_35116


namespace length_segment_PQ_l35_35722

noncomputable def curve_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

noncomputable def curve_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ

noncomputable def line_l1 (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ + Real.pi / 3) + 3 * Real.sqrt 3 = 0

noncomputable def line_l2 (θ : ℝ) : Prop :=
  θ = Real.pi / 3

theorem length_segment_PQ :
  (∃ (P Q : ℝ × ℝ), curve_polar P.1 P.2 ∧ curve_polar O1 P.2 ∧ line_l1 Q.1 Q.2 ∧ line_l2 O2 Q.2 ∧ P.2 = Q.2) →
  abs ((1 : ℝ) - (-3 : ℝ)) = 4 :=
by
  intros h
  sorry

end length_segment_PQ_l35_35722


namespace tenth_term_of_sequence_l35_35971

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 0 then 3
else if n = 1 then 4
else 12 / sequence (n - 1)

theorem tenth_term_of_sequence :
  sequence 9 = 4 :=
sorry

end tenth_term_of_sequence_l35_35971


namespace count_red_multiples_of_7_count_multiples_of_7_or_red_min_multiples_of_7_in_80_blacks_l35_35871

-- Definitions using conditions from step a).

def is_red (n : ℕ) := n % 2 = 0
def is_multiple_of_7 (n : ℕ) := n % 7 = 0

-- Lean statements for each of the questions

-- Statement (a)
theorem count_red_multiples_of_7 :
  ∃ (count : ℕ), count = 12 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → is_red n → is_multiple_of_7 n → count = (∑ k in finset.range (180+1), if is_red k ∧ is_multiple_of_7 k then 1 else 0) :=
sorry

-- Statement (b)
theorem count_multiples_of_7_or_red :
  ∃ (count : ℕ), count = 103 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → is_red n ∨ is_multiple_of_7 n → count = (∑ k in finset.range (180+1), if is_red k ∨ is_multiple_of_7 k then 1 else 0) :=
sorry

-- Statement (c)
theorem min_multiples_of_7_in_80_blacks (chosen_cards : finset ℕ) (h_size : chosen_cards.card = 80) :
  ∃ (count : ℕ), count = 3 ∧ count = (∑ k in chosen_cards, if is_multiple_of_7 k then 1 else 0) :=
sorry

end count_red_multiples_of_7_count_multiples_of_7_or_red_min_multiples_of_7_in_80_blacks_l35_35871


namespace reciprocal_of_5_over_7_l35_35045

theorem reciprocal_of_5_over_7 : (5 / 7 : ℚ) * (7 / 5) = 1 := by
  sorry

end reciprocal_of_5_over_7_l35_35045


namespace maximum_marks_l35_35370

theorem maximum_marks (pass_percent : ℚ) (scored_marks : ℕ) (short_fall : ℕ) (required_marks : ℕ) (total_marks : ℕ)
  (H1 : pass_percent = 0.50)
  (H2 : scored_marks = 212)
  (H3 : short_fall = 45)
  (H4 : required_marks = scored_marks + short_fall)
  (H5 : required_marks = pass_percent * total_marks) :
  total_marks = 514 :=
begin
  sorry,
end

end maximum_marks_l35_35370


namespace total_oranges_over_four_days_l35_35008

def jeremy_oranges_monday := 100
def jeremy_oranges_tuesday (B: ℕ) := 3 * jeremy_oranges_monday
def jeremy_oranges_wednesday (B: ℕ) (C: ℕ) := 2 * (jeremy_oranges_monday + B)
def jeremy_oranges_thursday := 70
def brother_oranges_tuesday := 3 * jeremy_oranges_monday - jeremy_oranges_monday -- This is B from Tuesday
def cousin_oranges_wednesday (B: ℕ) (C: ℕ) := 2 * (jeremy_oranges_monday + B) - (jeremy_oranges_monday + B)

theorem total_oranges_over_four_days (B: ℕ) (C: ℕ)
        (B_equals_tuesday: B = brother_oranges_tuesday)
        (J_plus_B_equals_300 : jeremy_oranges_tuesday B = 300)
        (J_plus_B_plus_C_equals_600 : jeremy_oranges_wednesday B C = 600)
        (J_thursday_is_70 : jeremy_oranges_thursday = 70)
        (B_thursday_is_B : B = brother_oranges_tuesday):
    100 + 300 + 600 + 270 = 1270 := by
        sorry

end total_oranges_over_four_days_l35_35008


namespace sum_of_two_smallest_prime_factors_of_450_l35_35489

theorem sum_of_two_smallest_prime_factors_of_450 : 
  let prime_factors := [2, 3, 5] in
  prime_factors.nth 0 + prime_factors.nth 1 = 5 := 
by 
  sorry

end sum_of_two_smallest_prime_factors_of_450_l35_35489


namespace pigeons_percentage_l35_35340

theorem pigeons_percentage (total_birds pigeons sparrows crows doves non_sparrows : ℕ)
  (h_total : total_birds = 100)
  (h_pigeons : pigeons = 40)
  (h_sparrows : sparrows = 20)
  (h_crows : crows = 15)
  (h_doves : doves = 25)
  (h_non_sparrows : non_sparrows = total_birds - sparrows) :
  (pigeons / non_sparrows : ℚ) * 100 = 50 :=
sorry

end pigeons_percentage_l35_35340


namespace value_of_b_l35_35302

variable (a b c : ℕ)
variable (h_a_nonzero : a ≠ 0)
variable (h_a : a < 8)
variable (h_b : b < 8)
variable (h_c : c < 8)
variable (h_square : ∃ k, k^2 = a * 8^3 + 3 * 8^2 + b * 8 + c)

theorem value_of_b : b = 1 :=
by sorry

end value_of_b_l35_35302


namespace magnitude_of_angle_B_l35_35325

-- Define Triangle with sides opposite to angles A, B, C respectively being a, b, c
variable (a b c : ℝ)
variable (A B C : ℝ)
-- Conditions in the problem
variable (h1 : 2 * b * real.cos B - c * real.cos A = a * real.cos C)

-- Lean statement for the problem
theorem magnitude_of_angle_B (h1 : 2 * b * real.cos B - c * real.cos A = a * real.cos C) : B = real.pi / 3 :=
sorry

end magnitude_of_angle_B_l35_35325


namespace red_balls_count_l35_35880

theorem red_balls_count:
  ∀ (R : ℕ), (0 : ℝ) < R ∧
    (probability = (R * (R - 1) / 2) / ((R + 5) * (R + 4) / 2)) ∧
    (probability = (1 / 6)) →
    R = 4 :=
by {
  sorry
}

end red_balls_count_l35_35880


namespace fixed_point_sum_l35_35518

theorem fixed_point_sum (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : (m, n) = (1, a * (1-1) + 2)) : m + n = 4 :=
by {
  sorry
}

end fixed_point_sum_l35_35518


namespace room_length_l35_35401

-- Defining conditions
def room_height : ℝ := 5
def room_width : ℝ := 7
def door_height : ℝ := 3
def door_width : ℝ := 1
def num_doors : ℝ := 2
def window1_height : ℝ := 1.5
def window1_width : ℝ := 2
def window2_height : ℝ := 1.5
def window2_width : ℝ := 1
def num_window2 : ℝ := 2
def paint_cost_per_sq_m : ℝ := 3
def total_paint_cost : ℝ := 474

-- Defining the problem as a statement to prove x (room length) is 10 meters
theorem room_length {x : ℝ} 
  (H1 : total_paint_cost = paint_cost_per_sq_m * ((2 * (x * room_height) + 2 * (room_width * room_height)) - (num_doors * (door_height * door_width) + (window1_height * window1_width) + num_window2 * (window2_height * window2_width)))) 
  : x = 10 :=
by 
  sorry

end room_length_l35_35401


namespace length_AC_l35_35531

-- Definitions based on the conditions
def radius (C : ℝ) : ℝ := C / (2 * Real.pi)
def SA (C : ℝ) : ℝ := radius C
def SC (C : ℝ) : ℝ := radius C

-- Lean 4 statement of the problem
theorem length_AC (C : ℝ) (hC : C = 16 * Real.pi) (θ : ℝ) (hθ : θ = 45) : 
  sqrt (SA C ^ 2 + SC C ^ 2) = 8 * sqrt 2 := 
by 
  sorry

end length_AC_l35_35531


namespace vertex_position_l35_35588

-- Definitions based on the conditions of the problem
def quadratic_function (x : ℝ) : ℝ := 3*x^2 + 9*x + 5

-- Theorem that the vertex of the parabola is at x = -1.5
theorem vertex_position : ∃ x : ℝ, x = -1.5 ∧ ∀ y : ℝ, quadratic_function y ≥ quadratic_function x :=
by
  sorry

end vertex_position_l35_35588


namespace Kenny_played_basketball_for_10_hours_l35_35739

theorem Kenny_played_basketball_for_10_hours
  (played_basketball ran practiced_trumpet : ℕ)
  (H1 : practiced_trumpet = 40)
  (H2 : ran = 2 * played_basketball)
  (H3 : practiced_trumpet = 2 * ran) :
  played_basketball = 10 :=
by
  sorry

end Kenny_played_basketball_for_10_hours_l35_35739


namespace fourth_divisor_of_9600_l35_35856

theorem fourth_divisor_of_9600 (x : ℕ) (h1 : ∀ (d : ℕ), d = 15 ∨ d = 25 ∨ d = 40 → 9600 % d = 0) 
  (h2 : 9600 / Nat.lcm (Nat.lcm 15 25) 40 = x) : x = 16 := by
  sorry

end fourth_divisor_of_9600_l35_35856


namespace segment_C1C2_constant_segment_C1C2_length_constant_l35_35390

open Complex

noncomputable def rotate90 {z : ℂ} (a : ℂ) : ℂ :=
  a + (z - a) * Complex.I

theorem segment_C1C2_constant (a b c  : ℂ) :
  let c1 := rotate90 c a
  let c2 := rotate90 c b in
  (c2 - c1 = (b - a) * (1 - Complex.I)) :=
by {
  let c1 := rotate90 c a,
  let c2 := rotate90 c b,
  calc
    c2 - c1
    = (b + (c - b) * Complex.I) - (a + (c - a) * Complex.I) : by sorry
    ... = (b - a) + ((c - b) * Complex.I - (c - a) * Complex.I) : by sorry
    ... = (b - a) + ((c - b) * Complex.I - (c - a) * Complex.I) : by sorry
    ... = (b - a) * (1 - Complex.I) : by ring,
}

theorem segment_C1C2_length_constant (a b c : ℂ) :
  let c1 := rotate90 c a
  let c2 := rotate90 c b in
  abs (c2 - c1) = Complex.abs (b - a) * sqrt 2 :=
by {
  let c1 := rotate90 c a,
  let c2 := rotate90 c b,
  calc
    abs (c2 - c1)
    = abs ((b - a) * (1 - Complex.I)) : by rw segment_C1C2_constant; rfl
    ... = abs (b - a) * abs (1 - Complex.I) : by rw Complex.abs_mul
    ... = abs (b - a) * sqrt 2 : by rw Complex.abs_one_sub_I,
}

end segment_C1C2_constant_segment_C1C2_length_constant_l35_35390


namespace find_ac_length_l35_35011

-- Define the problem in Lean 4.

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (A B C : Point)
  (isosceles_ab_bc : dist A B = dist B C)

noncomputable def dist (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def angle_p (p1 p2 p3 : Point) : ℝ :=
  let a := dist p2 p3 in
  let b := dist p1 p3 in
  let c := dist p1 p2 in
  real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))

variable (A B C M : Point)
variable (ABC : Triangle)
variable (AM : ℝ)
variable (MB : ℝ)
variable (angle_bmc : ℝ)

-- The given conditions
axiom h1 : AM = 7
axiom h2 : MB = 3
axiom h3 : angle_p B M C = 60 * real.pi / 180  -- converting degrees to radians
axiom h4 : dist A M + dist M C = dist A C      -- M lies on AC

-- The question to prove
theorem find_ac_length : dist A C = 17 :=
by
  sorry

end find_ac_length_l35_35011


namespace simplify_expr_l35_35577

noncomputable def expr : ℝ := Real.sqrt 12 - 3 * Real.sqrt (1 / 3) + Real.sqrt 27 + (Real.pi + 1)^0

theorem simplify_expr : expr = 4 * Real.sqrt 3 + 1 := by
  sorry

end simplify_expr_l35_35577


namespace correct_operation_l35_35105

theorem correct_operation (a b : ℝ) : 2 * a^2 * b - a^2 * b = a^2 * b :=
by
  sorry

end correct_operation_l35_35105


namespace distance_to_school_l35_35733

variable (v d : ℝ) -- typical speed (v) and distance (d)

theorem distance_to_school :
  (30 / 60 : ℝ) = 1 / 2 ∧ -- 30 minutes is 1/2 hour
  (18 / 60 : ℝ) = 3 / 10 ∧ -- 18 minutes is 3/10 hour
  d = v * (1 / 2) ∧ -- distance for typical day
  d = (v + 12) * (3 / 10) -- distance for quieter day
  → d = 9 := sorry

end distance_to_school_l35_35733


namespace no_pairwise_coprime_n_exists_l35_35383

def are_pairwise_coprime (a b c d : ℕ) (n : ℕ) : Prop :=
  Nat.coprime (a + n) (b + n) ∧
  Nat.coprime (a + n) (c + n) ∧
  Nat.coprime (a + n) (d + n) ∧
  Nat.coprime (b + n) (c + n) ∧
  Nat.coprime (b + n) (d + n) ∧
  Nat.coprime (c + n) (d + n)

theorem no_pairwise_coprime_n_exists :
  ¬ ∃ n : ℕ, n > 0 ∧ are_pairwise_coprime 1 2 3 4 n :=
sorry

end no_pairwise_coprime_n_exists_l35_35383


namespace bob_needs_additional_weeks_l35_35932

-- Definitions based on conditions
def weekly_prize : ℕ := 100
def initial_weeks_won : ℕ := 2
def total_prize_won : ℕ := initial_weeks_won * weekly_prize
def puppy_cost : ℕ := 1000
def additional_weeks_needed : ℕ := (puppy_cost - total_prize_won) / weekly_prize

-- Statement of the theorem
theorem bob_needs_additional_weeks : additional_weeks_needed = 8 := by
  -- Proof here
  sorry

end bob_needs_additional_weeks_l35_35932


namespace grid_valid_l35_35120

def is_black_cell (grid : ℕ × ℕ → Prop) (x y : ℕ) : Prop := 
  grid (x, y) = true

def neighbor (x y : ℕ) (dir : ℕ) : ℕ × ℕ :=
  match dir with
  | 0 => (x + 1, y)
  | 1 => (x - 1, y)
  | 2 => (x, y + 1)
  | 3 => (x, y - 1)
  | _ => (x, y)
  
def valid_config (grid : ℕ × ℕ → Prop) : Prop :=
  ∀ x y : ℕ,
  (x < 4 ∧ y < 4) →
  ((is_black_cell grid x y) → (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ ((λ n, ¬ is_black_cell grid (neighbor x y n).1 (neighbor x y n).2) a) ∧ (λ n, ¬ is_black_cell grid (neighbor x y n).1 (neighbor x y n).2) b ∧ (λ n, ¬ is_black_cell grid (neighbor x y n).1 (neighbor x y n).2) c)) ∧
  ((¬ is_black_cell grid x y) → (∃! n, is_black_cell grid (neighbor x y n).1 (neighbor x y n).2)) 

noncomputable def grid : ℕ × ℕ → Prop :=
  λ xy, match xy with
  | (0, 1) | (1, 2) | (2, 3) | (3, 0) => true
  | _ => false

theorem grid_valid : valid_config grid :=
  by sorry

end grid_valid_l35_35120


namespace cost_of_bread_l35_35836

theorem cost_of_bread
  (total_cost groceries : ℕ) 
  (cost_bananas cost_milk cost_apples : ℕ) 
  (h_groceries : total_cost = 42) 
  (h_bananas : cost_bananas = 12)
  (h_milk : cost_milk = 7)
  (h_apples : cost_apples = 14) : 
  let cost_bread := total_cost - (cost_bananas + cost_milk + cost_apples) in
  cost_bread = 9 :=
by
  sorry

end cost_of_bread_l35_35836


namespace hyperbola_asymptotes_l35_35283

def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (y^2 / a^2) - (x^2 / b^2) = 1

def is_circle (a c x y : ℝ) : Prop :=
  x^2 + y^2 - (2 * c / 3) * y + (a^2 / 9) = 0

def is_focus (x y c : ℝ) (is_upper : Bool) : Prop :=
  if is_upper then (x = 0 ∧ y = c) else (x = 0 ∧ y = -c)

def is_tangent_point (D : ℝ × ℝ) (F1 : ℝ × ℝ) (circle : ℝ × ℝ → Prop) : Prop :=
  ∃ (x₀ y₀ : ℝ), circle (x₀, y₀) ∧ tangent_calculation

-- Placeholder - replace with the appropriate condition for tangency
def tangent_calculation : Prop := sorry 

def perpendicular {A B C D : ℝ × ℝ} : Prop :=
  let m₁ := (D.2 - B.2) / (D.1 - B.1)
  let m₂ := (C.2 - B.2) / (C.1 - B.1)
  m₁ * m₂ = -1

theorem hyperbola_asymptotes (a b c : ℝ) (F1 F2 M D : ℝ × ℝ):
    a > 0 →
    b > 0 →
    c > 0 →
    is_hyperbola a b M.1 M.2 →
    is_focus F1.1 F1.2 c true →
    is_focus F2.1 F2.2 c false →
    is_circle a c D.1 D.2 →
    is_tangent_point D F1 (is_circle a c) →
    perpendicular F2 F1 M →
    b = 4*a →
    (D.1 - M.1/2)^2 + ((D.2 - c)/2)^2 = a^2 / 9 →
    ∃ (x y : ℝ), (x = 4 ∧ y = 1) ∨ (x = 1 ∧ y = 4) :=
  sorry

end hyperbola_asymptotes_l35_35283


namespace inequality_solution_l35_35221

theorem inequality_solution :
  { x : ℝ | 0 < x ∧ x ≤ 7/3 ∨ 3 ≤ x } = { x : ℝ | (0 < x ∧ x ≤ 7/3) ∨ 3 ≤ x } :=
sorry

end inequality_solution_l35_35221


namespace shifted_parabola_correct_l35_35712

-- Define original equation of parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2 - 1

-- Define shifted equation of parabola
def shifted_parabola (x : ℝ) : ℝ := 2 * (x + 1)^2 - 1

-- Proof statement: the expression of the new parabola after shifting 1 unit to the left
theorem shifted_parabola_correct :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) :=
by
  -- Proof is omitted, sorry
  sorry

end shifted_parabola_correct_l35_35712


namespace union_of_sets_l35_35684

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

theorem union_of_sets : A ∪ B = {-1, 0, 1, 2} := 
by
  sorry

end union_of_sets_l35_35684


namespace smallest_sum_of_bases_l35_35435

theorem smallest_sum_of_bases :
  ∃ (c d : ℕ), 8 * c + 9 = 9 * d + 8 ∧ c + d = 19 := 
by
  sorry

end smallest_sum_of_bases_l35_35435


namespace largest_sum_ab_bc_cd_da_l35_35053

theorem largest_sum_ab_bc_cd_da (a b c d : ℕ) 
  (h1 : a ∈ {2, 3, 4, 5}) 
  (h2 : b ∈ {2, 3, 4, 5})
  (h3 : c ∈ {2, 3, 4, 5})
  (h4 : d ∈ {2, 3, 4, 5})
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ab + bc + cd + da ≤ 48 := sorry

end largest_sum_ab_bc_cd_da_l35_35053


namespace remainder_3n_2m_l35_35713

theorem remainder_3n_2m (n m : ℤ) (hn : n ≡ 15 [MOD 37]) (hm : m ≡ 21 [MOD 47]) : 
  (3 * n + 2 * m) % 59 = 28 := 
by 
  sorry

end remainder_3n_2m_l35_35713


namespace distance_between_foci_of_hyperbola_is_10_l35_35985

-- Definitions as per the given conditions
def hyperbola_equation : ℝ → ℝ → Prop := 
  λ x y, 9 * x^2 - 18 * x - 16 * y^2 - 32 * y = 144

-- The theorem statement
theorem distance_between_foci_of_hyperbola_is_10 : 
  (∃ a b c : ℝ, 
    (∀ x y : ℝ, hyperbola_equation x y ↔ ((x - a)^2 / 16 - (y - b)^2 / 9 = 1))
    ∧ c^2 = 16 + 9 
    ∧ 2 * c = 10) :=
sorry

end distance_between_foci_of_hyperbola_is_10_l35_35985


namespace circle_equation_l35_35835

theorem circle_equation (h k r : ℝ) (x y : ℝ)
    (hA : h = 2)
    (kA : k = 1)
    (tangent : r = 1) :
    (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2 :=
by
  rw [hA, kA, tangent]
  exact eq.refl ((x - 2) ^ 2 + (y - 1) ^ 2 = 1)

end circle_equation_l35_35835


namespace symmetry_center_of_cubic_l35_35230

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x

theorem symmetry_center_of_cubic :
  ∃ c : ℝ, f c = 1 ∧ f'' c = 0 :=
by {
  let f' := λ x, 3 * x^2 - 6 * x + 3,
  let f'' := λ x, 6 * x - 6,
  have h1 : f'' 1 = 0 := sorry,
  have h2 : f 1 = 1 := sorry,
  use 1,
  exact ⟨h2, h1⟩
}

end symmetry_center_of_cubic_l35_35230


namespace liquid_levels_after_opening_tap_l35_35067

-- Definitions of initial conditions and constants
def height : ℝ := sorry
def H : ℝ := sorry
def K_height : ℝ := 0.2 * H
def initial_level : ℝ := 0.9 * H
def density_water : ℝ := 1000
def density_gasoline : ℝ := 600
def final_level_left : ℝ := 0.69 * H
def final_level_right : ℝ := H

-- Theorem to prove liquid levels after opening the tap
theorem liquid_levels_after_opening_tap
  (h_height : height = H)
  (h_initial_level : initial_level = 0.9 * H)
  (h_density_water : density_water = 1000)
  (h_density_gasoline : density_gasoline = 600)
  (h_final_level_left : final_level_left = 0.69 * H)
  (h_final_level_right : final_level_right = H) :
  final_level_left = 0.69 * H ∧ final_level_right = H :=
sorry

end liquid_levels_after_opening_tap_l35_35067


namespace magnitude_of_z_l35_35888

noncomputable def z : ℂ := (-1/5) + (7/5) * Complex.i

theorem magnitude_of_z : (z - (Complex.i)) * (2 - (Complex.i)) = Complex.i → Complex.abs z = Real.sqrt 2 :=
by
  intro h
  sorry

end magnitude_of_z_l35_35888


namespace exists_player_winning_range_l35_35528

theorem exists_player_winning_range (n k : ℕ) (players : Fin (2*n+1) → ℕ) 
(h1 : ∀ i j, i ≠ j → (players i ≠ players j))
(h2 : ∀ i j, i < j ∨ j < i → (players i < players j ∨ players j < players i))
 (h3 : ∃ k : ℕ, (k = count (λ (i j : Fin (2*n + 1)), i ≠ j ∧ players i < players j ∧ results i j = some false))) : ∃ (player : Fin (2*n+1)), n - nat.sqrt (2*k) ≤ count (λ j, results player j = some true) ∧ count (λ j, results player j = some true) ≤ n + nat.sqrt (2*k) :=
sorry

end exists_player_winning_range_l35_35528


namespace b_geometric_sequence_sum_a_seq_l35_35285

-- Conditions of the problem
def a₁ : ℚ := 3 / 2
def a_seq (a : ℕ → ℚ) := ∀ n : ℕ, a (n + 1) = 3 * a n - 1

-- b_n condition
def b_seq (a b : ℕ → ℚ) := ∀ n : ℕ, b n = a n - 1 / 2

-- Prove that b_n is a geometric sequence
theorem b_geometric_sequence (a b : ℕ → ℚ) (h1 : a 1 = a₁) (h2 : a_seq a) (h3 : b_seq a b) :
  ∃ r : ℚ, ∀ n : ℕ, b (n + 1) = r * b n ∧ b 1 = 1 :=
sorry

-- Given the sum of the first n terms Sn of sequence an
def sum_seq (a : ℕ → ℚ) (S : ℕ → ℚ) := ∀ n : ℕ, S n = (∑ i in finset.range n, a i)

-- Prove the sum of the sequence
theorem sum_a_seq (a : ℕ → ℚ) (S : ℕ → ℚ) (h1 : a 1 = a₁) (h2 : a_seq a) :
  sum_seq a S → ∀ n : ℕ, S n = (3^n + n - 1) / 2 :=
sorry

end b_geometric_sequence_sum_a_seq_l35_35285


namespace katie_total_expenditure_l35_35926

-- Define the conditions
def flower_cost : ℕ := 6
def roses_bought : ℕ := 5
def daisies_bought : ℕ := 5

-- Define the total flowers bought
def total_flowers_bought : ℕ := roses_bought + daisies_bought

-- Calculate the total cost
def total_cost (flower_cost : ℕ) (total_flowers_bought : ℕ) : ℕ :=
  total_flowers_bought * flower_cost

-- Prove that Katie spent 60 dollars
theorem katie_total_expenditure : total_cost flower_cost total_flowers_bought = 60 := sorry

end katie_total_expenditure_l35_35926


namespace smallest_n_for_unity_roots_l35_35457

theorem smallest_n_for_unity_roots (n : ℕ) :
  (∀ x, (x ^ 5 - x ^ 3 + x = 0) → 
       (∃ k : ℕ, x = exp (2 * π * I * k / n))) ↔ n = 12 := 
sorry

end smallest_n_for_unity_roots_l35_35457


namespace area_equality_l35_35785

open Real

-- Definitions of points on the curve and perpendicular projections
variables {x_A x_B : ℝ} (hx_A : x_A > 0) (hx_B : x_B > 0)

-- Points A and B on the curve y = 1/x
def A : ℝ × ℝ := (x_A, 1 / x_A)
def B : ℝ × ℝ := (x_B, 1 / x_B)
-- Foot of the perpendiculars from A and B to the x-axis
def H_A : ℝ × ℝ := (x_A, 0)
def H_B : ℝ × ℝ := (x_B, 0)

theorem area_equality :
  let O : ℝ × ℝ := (0, 0) in
  area_triangle O A H_A + area_triangle O B H_B + comp_area_AB = 
  area_triangle AH_A BH_B + 
  ∫ x in (min x_A x_B)..(max x_A x_B), (λ x, 1 / x) :=
sorry

end area_equality_l35_35785


namespace range_of_a_l35_35676

noncomputable def f (x t : ℝ) : ℝ := (x - t) * |x|

theorem range_of_a (a : ℝ) : (∃ t : ℝ, t ∈ Ioo 0 2 ∧ ∀ x ∈ Icc (-1 : ℝ) 2, f x t > x + a) → a ≤ -1/4 := sorry

end range_of_a_l35_35676


namespace number_of_diamonds_in_F10_l35_35193

def sequence_of_figures (F : ℕ → ℕ) : Prop :=
  F 1 = 4 ∧
  (∀ n ≥ 2, F n = F (n-1) + 4 * (n + 2)) ∧
  F 3 = 28

theorem number_of_diamonds_in_F10 (F : ℕ → ℕ) (h : sequence_of_figures F) : F 10 = 336 :=
by
  sorry

end number_of_diamonds_in_F10_l35_35193


namespace minimum_surface_area_of_sphere_minimum_volume_of_sphere_l35_35156

-- Definition of a right-angled triangle
structure RightAngledTriangle :=
  (x y : ℝ)
  (hypotenuse : ℝ)
  (xy_eq_8 : x * y = 8)
  (hypotenuse_eq : (hypotenuse^2 = x^2 + y^2))

-- Definition of a triangular pyramid
structure TriangularPyramid (base : RightAngledTriangle) :=
  (height : ℝ)
  (volume_eq_4 : (1 / 3) * (1 / 2) * base.x * base.y * height = 4)

-- Definition of Sphere O with vertices of the pyramid on the sphere
structure SphereO (pyramid : TriangularPyramid) :=
  (radius : ℝ)
  (vertex_on_sphere : ∀ (A B C P : ℝ), True) -- Placeholder for vertex condition

-- Projection of P
structure ProjectionPontoBase (pyramid : TriangularPyramid) :=
  (K : ℝ)
  (PK_eq_3 : K = 3)

-- Proof Statements
theorem minimum_surface_area_of_sphere
  (pyramid : TriangularPyramid)
  (sphere : SphereO pyramid)
  (proj : ProjectionPontoBase pyramid)
  (K_coincides_A : proj.K = 0) -- Example condition where K coincides with A
  : 4 * real.pi * (sphere.radius^2) ≥ 25 * real.pi := sorry

theorem minimum_volume_of_sphere
  (pyramid : TriangularPyramid)
  (sphere : SphereO pyramid)
  (proj : ProjectionPontoBase pyramid)
  (K_midpoint_hypotenuse : proj.K = (pyramid.base.hypotenuse / 2)) -- Example condition for midpoint of hypotenuse
  : (4 / 3) * real.pi * (sphere.radius^3) ≥ (2197 * real.pi / 162) := sorry

end minimum_surface_area_of_sphere_minimum_volume_of_sphere_l35_35156


namespace range_of_a_l35_35656

noncomputable def set_A : Set ℝ := { x | (3 * x - 1) / (x - 2) ≤ 1 }
noncomputable def set_B (a : ℝ) : Set ℝ := { x | x^2 - (a + 2) * x + 2 * a < 0 }

theorem range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, x ∈ set_A → x ∈ set_B a) ∧ ¬ (∀ x : ℝ, x ∈ set_B a → x ∈ set_A) →
  a ∈ Iio (-1/2) := 
sorry

end range_of_a_l35_35656


namespace total_distance_traveled_l35_35782

theorem total_distance_traveled : 
  ∀ (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z],
  (XZ Y Z = 4000) → (XY X Y = 4500) → 
  (XYZ_form_right_angled_triangle X Y Z) →
  (total_distance X Y Z = 10562) :=
by
  intro X Y Z hXZ hXY hXYZ
  sorry

end total_distance_traveled_l35_35782


namespace machine_a_sprockets_per_hour_l35_35508

theorem machine_a_sprockets_per_hour (s h : ℝ)
    (H1 : 1.1 * s * h = 550)
    (H2 : s * (h + 10) = 550) : s = 5 := by
  sorry

end machine_a_sprockets_per_hour_l35_35508


namespace largest_prime_factor_of_binomial_l35_35088

theorem largest_prime_factor_of_binomial :
  ∃ p : ℕ, p.prime ∧ 10 ≤ p ∧ p < 100 ∧ (∃ k : ℕ, p ^ k ∣ (300.factorial / (150.factorial * 150.factorial))) ∧
  (∀ q : ℕ, q.prime → 10 ≤ q ∧ q < 100 → (∃ k : ℕ, q ^ k ∣ (300.factorial / (150.factorial * 150.factorial))) → q ≤ p) :=
sorry

end largest_prime_factor_of_binomial_l35_35088


namespace twenty_sided_polygon_distinct_points_l35_35240

open Complex

noncomputable def numberOfDistinctPoints (n : ℕ) (k : ℕ) (m : ℤ) : ℕ :=
  let vertices := (0 : ℕ) :: List.range (n - 1)
  let points := vertices.map (λ i => (exp (I * (2 * ↑i * real.pi / ↑n) ) ) ^ m)
  (points.to_finset.card : ℕ)

theorem twenty_sided_polygon_distinct_points : 
  numberOfDistinctPoints 20 0 1995 = 4 :=
sorry

end twenty_sided_polygon_distinct_points_l35_35240


namespace marble_selection_sum_l35_35691

-- Constants representing the two sets of marbles
def myMarbles : List ℕ := List.range (8 + 1) -- marbles 1 to 8
def mathewsMarbles : List ℕ := List.range (20 + 1) -- marbles 1 to 20

-- Function to calculate the sum of all valid 3-combinations of my marbles
def valid_sums : Finset ℕ :=
  Finset.image List.sum (List.subsetsOfCard 3 myMarbles).toFinset

-- Define counting function for valid combinations
def count_valid_combinations (n : ℕ) : ℕ :=
  ((List.subsetsOfCard 3 myMarbles).filter (λ l, l.sum = n)).length

-- Define the total number of valid combinations
def total_valid_combinations : ℕ := 
  (Finset.range 22).sum count_valid_combinations

-- Theorem stating the question in formal terms
theorem marble_selection_sum :
  total_valid_combinations = ∑ (n in Finset.range 22), count_valid_combinations n := by
sorry

end marble_selection_sum_l35_35691


namespace inequality_solution_l35_35029

theorem inequality_solution (x : ℝ) (h1 : 3x + 2 ≠ 0) (h2 : 3 - 1/(3x + 2) < 5) :
  x ∈ set.Iio (-5/3) ∪ set.Ioi (-2/3) :=
by sorry

end inequality_solution_l35_35029


namespace train_crossing_time_l35_35122

noncomputable def train_length : ℝ := 160  -- length of the train in meters
noncomputable def train_speed_kmh : ℝ := 144  -- speed of the train in km/h

def convert_kmh_to_mps (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)  -- conversion factor from km/h to m/s

theorem train_crossing_time : 
  let speed_mps := convert_kmh_to_mps train_speed_kmh in
  let crossing_time := train_length / speed_mps in
  crossing_time = 4 :=
by
  sorry

end train_crossing_time_l35_35122


namespace isosceles_triangle_properties_l35_35566

/--
  An isosceles triangle has a base of 6 units and legs of 5 units each.
  Prove:
  1. The area of the triangle is 12 square units.
  2. The radius of the inscribed circle is 1.5 units.
-/
theorem isosceles_triangle_properties (base : ℝ) (legs : ℝ) 
  (h_base : base = 6) (h_legs : legs = 5) : 
  ∃ (area : ℝ) (inradius : ℝ), 
  area = 12 ∧ inradius = 1.5 
  :=
by
  sorry

end isosceles_triangle_properties_l35_35566


namespace decreasing_implies_inequality_l35_35196

variable (f : ℝ → ℝ)

theorem decreasing_implies_inequality (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) : f 3 < f 2 ∧ f 2 < f 1 :=
  sorry

end decreasing_implies_inequality_l35_35196


namespace expression_value_l35_35956

theorem expression_value : (100 - (1000 - 300)) - (1000 - (300 - 100)) = -1400 := by
  sorry

end expression_value_l35_35956


namespace total_cost_price_l35_35552

theorem total_cost_price (SP1 SP2 SP3 : ℝ) (P1 P2 P3 : ℝ) 
  (h1 : SP1 = 120) (h2 : SP2 = 150) (h3 : SP3 = 200)
  (h4 : P1 = 0.20) (h5 : P2 = 0.25) (h6 : P3 = 0.10) : (SP1 / (1 + P1) + SP2 / (1 + P2) + SP3 / (1 + P3) = 401.82) :=
by
  sorry

end total_cost_price_l35_35552


namespace general_formula_sum_of_bn_l35_35348

def sequence (n : ℕ) : ℕ → ℕ :=
  λ n, 2 * n + 1

theorem general_formula
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h1 : ∀ n, a (n + 1) - a n = 2)
  (h2 : a 1 = 3) :
  ∀ n, a n = 2 * n + 1 :=
sorry

theorem sum_of_bn (n : ℕ) 
  (a : ℕ → ℕ) 
  (b : ℕ → ℕ) 
  (h1 : ∀ n, a n = 2 * n + 1) :
  ∑ i in finset.range n, b i = n / (2 * (2 * n + 1)) :=
sorry

end general_formula_sum_of_bn_l35_35348


namespace incorrect_option_C_l35_35349

variables {α : Type*} [OrderedRing α]

-- Definitions of the arithmetic sequence and sums
def arithmetic_sequence (a d : α) (n : ℕ) : α := a + n * d

def sum_arithmetic_sequence (a d : α) : ℕ → α
| 0       := 0
| (n + 1) := sum_arithmetic_sequence n + arithmetic_sequence a d n

-- Given conditions
variables {a d : α}
variables {S : ℕ → α}
variable h_sum_def : ∀ n, S (n + 1) = S n + arithmetic_sequence a d n
variable h_S5_lt_S6 : S 5 < S 6
variable h_S6_eq_S7 : S 6 = S 7
variable h_S7_gt_S8 : S 7 > S 8

-- Conclusion to prove
theorem incorrect_option_C : S 9 ≤ S 5 :=
sorry

end incorrect_option_C_l35_35349


namespace least_multiple_of_11_not_lucky_l35_35134

-- Define what it means for a number to be a lucky integer
def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

-- Define what it means for a number to be a multiple of 11
def is_multiple_of_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- State the problem: the least positive multiple of 11 that is not a lucky integer is 132
theorem least_multiple_of_11_not_lucky :
  ∃ n : ℕ, is_multiple_of_11 n ∧ ¬ is_lucky n ∧ n = 132 :=
sorry

end least_multiple_of_11_not_lucky_l35_35134


namespace range_of_a_l35_35677

def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 + 4 * x else 4 * x - x^2

theorem range_of_a (a : ℝ) :
  (f (2 - a^2) > f a) ↔ (-2 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l35_35677


namespace line_intersects_at_least_one_l35_35292

theorem line_intersects_at_least_one
  (a b : Line) (α β : Plane) (l : Line)
  (h1 : a ≠ b)
  (h2 : ¬(a ∥ b))
  (h3 : a ⊆ α)
  (h4 : b ⊆ β)
  (h5 : α ∩ β = l) :
  l Intersects_at_least_one_of (a, b) := sorry

end line_intersects_at_least_one_l35_35292


namespace points_on_curve_l35_35329

theorem points_on_curve (x y : ℝ) :
  (∃ p : ℝ, y = p^2 + (2 * p - 1) * x + 2 * x^2) ↔ y ≥ x^2 - x :=
by
  sorry

end points_on_curve_l35_35329


namespace loss_percentage_l35_35917

theorem loss_percentage
  (CP : ℝ := 1166.67)
  (SP : ℝ)
  (H : SP + 140 = CP + 0.02 * CP) :
  ((CP - SP) / CP) * 100 = 10 := 
by 
  sorry

end loss_percentage_l35_35917


namespace calculate_expression_l35_35576

variables (a b : ℝ)

theorem calculate_expression : -a^2 * 2 * a^4 * b = -2 * (a^6) * b :=
by
  sorry

end calculate_expression_l35_35576


namespace eggs_collection_l35_35181

theorem eggs_collection (b : ℕ) (c : ℕ) (t : ℕ) 
  (h₁ : b = 6) 
  (h₂ : c = 3 * b) 
  (h₃ : t = b - 4) : 
  b + c + t = 26 :=
by
  sorry

end eggs_collection_l35_35181


namespace Theorem3_l35_35679

theorem Theorem3 {f g : ℝ → ℝ} (T1_eq_1 : ∀ x, f (x + 1) = f x)
  (m : ℕ) (h_g_periodic : ∀ x, g (x + 1 / m) = g x) (hm : m > 1) :
  ∃ k : ℕ, k > 0 ∧ (k = 1 ∨ (k ≠ m ∧ ¬(m % k = 0))) ∧ 
    (∀ x, (f x + g x) = (f (x + 1 / k) + g (x + 1 / k))) := 
sorry

end Theorem3_l35_35679


namespace tenth_term_of_sequence_l35_35969

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 0 then 3
else if n = 1 then 4
else 12 / sequence (n - 1)

theorem tenth_term_of_sequence :
  sequence 9 = 4 :=
sorry

end tenth_term_of_sequence_l35_35969


namespace total_egg_collection_l35_35174

theorem total_egg_collection (
  -- Conditions
  (Benjamin_collects : Nat) (h1 : Benjamin_collects = 6) 
  (Carla_collects : Nat) (h2 : Carla_collects = 3 * Benjamin_collects) 
  (Trisha_collects : Nat) (h3 : Trisha_collects = Benjamin_collects - 4)
  ) : 
  -- Question and answer
  (Total_collects : Nat) (h_total : Total_collects = Benjamin_collects + Carla_collects + Trisha_collects) => 
  (Total_collects = 26) := 
  by
  sorry

end total_egg_collection_l35_35174


namespace sum_of_k_l35_35488

theorem sum_of_k (k : ℕ) :
  ((∃ x, x^2 - 4 * x + 3 = 0 ∧ x^2 - 7 * x + k = 0) →
  (k = 6 ∨ k = 12)) →
  (6 + 12 = 18) :=
by sorry

end sum_of_k_l35_35488


namespace smallest_n_is_12_l35_35458

noncomputable def smallest_n : ℕ :=
  Inf {n : ℕ | ∀ z : ℂ, (z^5 - z^3 + z = 0) → ∃ k : ℕ, z = e^(2 * π * complex.I * k / n) }

theorem smallest_n_is_12 : smallest_n = 12 :=
sorry

end smallest_n_is_12_l35_35458


namespace nearest_whole_number_l35_35794

theorem nearest_whole_number (x : ℝ) (h : x = 7263.4987234) : Int.floor (x + 0.5) = 7263 := by
  sorry

end nearest_whole_number_l35_35794


namespace speed_ratio_thirteen_l35_35006

noncomputable section

def speed_ratio (vNikita vCar : ℝ) : ℝ := vCar / vNikita

theorem speed_ratio_thirteen :
  ∀ (vNikita vCar : ℝ),
  (65 * vNikita = 5 * vCar) →
  speed_ratio vNikita vCar = 13 :=
by
  intros vNikita vCar h
  unfold speed_ratio
  sorry

end speed_ratio_thirteen_l35_35006


namespace value_of_x_l35_35269

theorem value_of_x (x c m n : ℝ) (hne: m≠n) (hneq : c ≠ 0) 
  (h1: c = 3) (h2: m = 2) (h3: n = 5)
  (h4: (x + c * m)^2 - (x + c * n)^2 = (m - n)^2) : 
  x = -11 := by
  sorry

end value_of_x_l35_35269


namespace total_egg_collection_l35_35175

theorem total_egg_collection (
  -- Conditions
  (Benjamin_collects : Nat) (h1 : Benjamin_collects = 6) 
  (Carla_collects : Nat) (h2 : Carla_collects = 3 * Benjamin_collects) 
  (Trisha_collects : Nat) (h3 : Trisha_collects = Benjamin_collects - 4)
  ) : 
  -- Question and answer
  (Total_collects : Nat) (h_total : Total_collects = Benjamin_collects + Carla_collects + Trisha_collects) => 
  (Total_collects = 26) := 
  by
  sorry

end total_egg_collection_l35_35175


namespace brown_ball_weight_l35_35385

theorem brown_ball_weight (w_blue : ℝ) (w_total : ℝ) (w_brown : ℝ)
  (h_blue : w_blue = 6) (h_total : w_total = 9.12) :
  w_brown = 3.12 :=
by
  -- substitute the given weights
  have h_calculation : w_brown = w_total - w_blue := sorry
  -- show that 9.12 - 6 = 3.12
  rw [h_total, h_blue] at h_calculation
  exact h_calculation

end brown_ball_weight_l35_35385


namespace equivalent_function_l35_35692

theorem equivalent_function (x : ℝ) (h : x > 1) : 10^(Real.log10 (x - 1)) = ((x - 1) / (Real.sqrt (x - 1)))^2 := 
by sorry

end equivalent_function_l35_35692


namespace tan_300_deg_l35_35961

theorem tan_300_deg : tan (300 * (Real.pi / 180)) = -Real.sqrt 3 :=
by
  -- Definitions and conditions
  have h1 : tan (θ : ℝ) = tan (θ + 2 * Real.pi) := sorry
  have h2 : tan (-θ : ℝ) = -tan θ := sorry
  have h3 : tan (60 * (Real.pi / 180)) = Real.sqrt 3 := sorry
  -- Claim to be proven
  sorry

end tan_300_deg_l35_35961


namespace walk_back_steps_l35_35689

theorem walk_back_steps :
  let prime_numbers := [2, 3, 5, 7, 11, 13, 17, 19, 23]
  let composite_count := 24 - prime_numbers.length
  let steps_forward := prime_numbers.length
  let steps_backward := composite_count * 2
  steps_backward - steps_forward = 21 :=
by
  let prime_numbers := [2, 3, 5, 7, 11, 13, 17, 19, 23]
  let composite_count := 24 - prime_numbers.length
  let steps_forward := prime_numbers.length
  let steps_backward := composite_count * 2
  show steps_backward - steps_forward = 21 from by
    calc
      steps_backward - steps_forward
          = 30 - 9 : by
          have composite_count := 24 - 9
          have steps_backward := composite_count * 2
          exact rfl
      ... = 21 : by rfl

end walk_back_steps_l35_35689


namespace maximum_F_on_1_5_minimum_F_on_1_5_l35_35235

noncomputable def F (x : ℝ) : ℝ := (1 / 3) * x^3 - 2 * x^2 + (7 / 3)

theorem maximum_F_on_1_5 : ∀ (x : ℝ), x ∈ set.Icc (1 : ℝ) (5 : ℝ) → F x ≤ F 1 :=
by
  sorry

theorem minimum_F_on_1_5 : ∀ (x : ℝ), x ∈ set.Icc (1 : ℝ) (5 : ℝ) → F x ≥ F 4 :=
by
  sorry

end maximum_F_on_1_5_minimum_F_on_1_5_l35_35235


namespace toucans_total_l35_35057

theorem toucans_total :
  let L1 := 3.5
  let L2 := 4.25
  let L3 := 2.75
  let J1 := 1.5
  let J2 := 0.6
  let J3 := 1.2
  (L1 + J1) + (L2 + J2) + (L3 + J3) = 13.8 :=
by
  let L1 := 3.5
  let L2 := 4.25
  let L3 := 2.75
  let J1 := 1.5
  let J2 := 0.6
  let J3 := 1.2
  calc
    (L1 + J1) + (L2 + J2) + (L3 + J3) = (3.5 + 1.5) + (4.25 + 0.6) + (2.75 + 1.2)   : by rw [L1, L2, L3, J1, J2, J3]
    ... = 5 + 4.85 + 3.95                                                          : by norm_num
    ... = 13.8                                                                     : by norm_num

end toucans_total_l35_35057


namespace part_a_part_b_part_c_part_d_l35_35018

noncomputable def C0 (a : list ℝ) : ℝ := (list.prod a) ^ (1 / (list.length a))

noncomputable def C1 (a : list ℝ) : ℝ := (list.sum a) / (list.length a)

noncomputable def C_alpha (alpha : ℝ) (a : list ℝ) : ℝ := 
  ((list.sum (a.map (λ x => x ^ alpha))) / (list.length a)) ^ (1 / alpha)

noncomputable def C2 (a : list ℝ) : ℝ := 
  (list.sum (a.map (λ x => x ^ 2)) / (list.length a)) ^ (1 / 2)

theorem part_a (a : list ℝ) : ln (C0 a) = C1 (a.map ln) := 
  sorry

theorem part_b (a : list ℝ) (alpha : ℝ) : 
  (C_alpha alpha (a.map (λ x => x ^ (1 / alpha)))) ^ alpha = C1 a :=
  sorry

theorem part_c (a b : list ℝ) : 
  C1 (list.zip a b).map (λ p => p.1 * p.2) ≤ (C2 a) * (C2 b) :=
  sorry

theorem part_d (a b : list ℝ) : 
  C1 a * C1 b ≥ C_alpha (1 / 2) (list.zip a b).map (λ p => p.1 * p.2) :=
  sorry

end part_a_part_b_part_c_part_d_l35_35018


namespace sum_of_interior_edges_l35_35144

def frame_width : ℝ := 1
def outer_length : ℝ := 5
def frame_area : ℝ := 18
def inner_length1 : ℝ := outer_length - 2 * frame_width

/-- Given conditions and required to prove:
1. The frame is made of one-inch-wide pieces of wood.
2. The area of just the frame is 18 square inches.
3. One of the outer edges of the frame is 5 inches long.
Prove: The sum of the lengths of the four interior edges is 14 inches.
-/
theorem sum_of_interior_edges (inner_length2 : ℝ) 
  (h1 : (outer_length * (inner_length2 + 2) - inner_length1 * inner_length2) = frame_area)
  (h2 : (inner_length2 - 2) / 2 = 1) : 
  inner_length1 + inner_length1 + inner_length2 + inner_length2 = 14 :=
by
  sorry

end sum_of_interior_edges_l35_35144


namespace find_m_l35_35666

theorem find_m (m : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → (x^2 + 2 * m * x + m + 2) ≥ -3) →
  (∃ m, (m = 3 ∨ m = (1 - Real.sqrt 21) / 2)) :=
by {
  intros h,
  sorry
}

end find_m_l35_35666


namespace max_elements_in_set_l35_35906

noncomputable def max_elements : ℕ :=
  sorry

theorem max_elements_in_set (S : Set ℕ) (h1 : (1 : ℕ) ∈ S) (h1001 : 1001 ∈ S)
  (h_distinct : S.ToFinset.val.Nodup) (h_condition : ∀ x y ∈ S, 
  (∃ k : ℤ, k = (S.sum - x - y) / (|S| - 2)))
  : S.card ≤ 28 :=
  sorry

end max_elements_in_set_l35_35906


namespace count_distinct_triangles_l35_35262

theorem count_distinct_triangles : 
  ∃ (S : Finset (Finset (ℕ × ℕ × ℕ))), 
    (∀ (t ∈ S), 
      let ⟨a, b, c⟩ := t in 
      (a ≤ 4 ∧ b ≤ 4 ∧ c ≤ 4) ∧ 
      (a + b > c ∧ b + c > a ∧ c + a > b)) ∧
    S.card = 13 :=
begin
  sorry
end

end count_distinct_triangles_l35_35262


namespace largest_good_number_lt_2012_l35_35619

-- Definitions based on the given conditions
def is_good (n : ℕ) : Prop :=
  ∀ d ∣ n, d.bits.count (λ b, b = tt) ≤ 2

-- The primes in the form of 2^k + 1, less than 2012
def valid_primes := {3, 5, 17, 257}

-- Any good number n < 2012 is in the form c * 2^k, where c ∈ {1, 3, 5, 9, 17, 257}
def good_form (n : ℕ) : Prop :=
  ∃ c k, c ∈ {1, 3, 5, 9, 17, 257} ∧ n = c * 2^k

-- Main theorem stating what we need to prove
theorem largest_good_number_lt_2012 :
  ∃ n, is_good n ∧ good_form n ∧ n < 2012 ∧ ∀ m, is_good m ∧ good_form m ∧ m < 2012 → m ≤ n :=
  ∃ n, n = 1536 :=
  sorry

end largest_good_number_lt_2012_l35_35619


namespace singer_total_hours_l35_35907

theorem singer_total_hours :
  let vocals_1 := 8 * 12
  let vocals_2 := 10 * 9
  let vocals_3 := 6 * 15
  let instruments_1 := 2 * 6
  let instruments_2 := 3 * 4
  let instruments_3 := 1 * 5
  let mixing_1 := 4 * 3
  let mixing_2 := 5 * 2
  let mixing_3 := 3 * 4
  let video_production := 5 * 7
  let marketing := 4 * 10
  vocals_1 + vocals_2 + vocals_3 + instruments_1 + instruments_2 + instruments_3 + mixing_1 + mixing_2 + mixing_3 + video_production + marketing = 414 :=
by
  let vocals := vocals_1 + vocals_2 + vocals_3
  let instruments := instruments_1 + instruments_2 + instruments_3
  let mixing := mixing_1 + mixing_2 + mixing_3
  have h1 : vocals = 276 := by sorry
  have h2 : instruments = 29 := by sorry
  have h3 : mixing = 34 := by sorry
  have h4 : video_production = 35 := by sorry
  have h5 : marketing = 40 := by sorry
  have h6 : vocals + instruments + mixing + video_production + marketing = 414 := by sorry
  exact h6

end singer_total_hours_l35_35907


namespace mary_donated_books_l35_35772

theorem mary_donated_books 
  (s : ℕ) (b_c : ℕ) (b_b : ℕ) (b_y : ℕ) (g_d : ℕ) (g_m : ℕ) (e : ℕ) (s_s : ℕ) 
  (total : ℕ) (out_books : ℕ) (d : ℕ)
  (h1 : s = 72)
  (h2 : b_c = 12)
  (h3 : b_b = 5)
  (h4 : b_y = 2)
  (h5 : g_d = 1)
  (h6 : g_m = 4)
  (h7 : e = 81)
  (h8 : s_s = 3)
  (ht : total = s + b_c + b_b + b_y + g_d + g_m)
  (ho : out_books = total - e)
  (hd : d = out_books - s_s) :
  d = 12 :=
by { sorry }

end mary_donated_books_l35_35772


namespace common_intersection_point_l35_35256

structure Disk (P : Type) :=
(center : P)
(radius : ℝ)
(nonnegative_radius : 0 ≤ radius)

variables {P : Type} [metric_space P]

-- Given conditions
variables (red1 red2 white1 white2 green1 green2 : Disk P)
  (common_point : ∀ (r : Disk P) (w : Disk P) (g : Disk P), (r = red1 ∨ r = red2) → (w = white1 ∨ w = white2) → (g = green1 ∨ g = green2) → ∃ p : P, dist p r.center ≤ r.radius ∧ dist p w.center ≤ w.radius ∧ dist p g.center ≤ g.radius)

theorem common_intersection_point :
  (∃ p : P, dist p red1.center ≤ red1.radius ∧ dist p red2.center ≤ red2.radius) ∨
  (∃ p : P, dist p white1.center ≤ white1.radius ∧ dist p white2.center ≤ white2.radius) ∨
  (∃ p : P, dist p green1.center ≤ green1.radius ∧ dist p green2.center ≤ green2.radius) :=
sorry

end common_intersection_point_l35_35256


namespace find_line_equation_l35_35958

theorem find_line_equation (
    x y : ℝ,
    h₁ : 3 * x + 4 * y - 2 = 0,
    h₂ : 2 * x + y + 2 = 0,
    h₃ : 3 * x - 2 * y + 4 = 0) :
    2 * x - 3 * y - 22 = 0 :=
sorry

end find_line_equation_l35_35958


namespace mass_percentage_H_is_correct_l35_35620

-- Define the given atomic masses and their quantities in the molecule
def atomic_mass_H : ℝ := 1.01
def atomic_mass_C : ℝ := 12.01
def atomic_mass_O : ℝ := 16.00

def num_H_atoms : ℕ := 2
def num_C_atoms : ℕ := 1
def num_O_atoms : ℕ := 3

def mass_percentage_H_in_H2CO3 : ℝ :=
  (num_H_atoms * atomic_mass_H) / 
  ((num_H_atoms * atomic_mass_H) + (num_C_atoms * atomic_mass_C) + (num_O_atoms * atomic_mass_O)) * 100

theorem mass_percentage_H_is_correct : 
  mass_percentage_H_in_H2CO3 = 3.26 := by
  sorry

end mass_percentage_H_is_correct_l35_35620


namespace distance_sum_l35_35658

noncomputable section

open Real

def ellipse : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.1 ^ 2 / 2 + p.2 ^ 2 = 1 }

def line : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.2 = p.1 - 1 }

def left_focus_of_ellipse : ℝ × ℝ := (-1, 0)

def points_of_intersection : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p ∈ ellipse ∧ p ∈ line }

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_sum : 
  (∃ A B : ℝ × ℝ, A ∈ points_of_intersection ∧ B ∈ points_of_intersection
    ∧ distance left_focus_of_ellipse A + distance left_focus_of_ellipse B = (8 * sqrt(2)) / 3) :=
sorry

end distance_sum_l35_35658


namespace function_properties_l35_35537

def f (x : ℝ) : ℝ := sin (2 * x - π / 6)

def smallest_period (T : ℝ) (f : ℝ → ℝ) := ∀ x, f (x + T) = f x ∧ (∀ ε > 0, ε < T → ∃ x, f (x + ε) ≠ f x)
def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) := ∀ x, f (2 * a - x) = f x
def is_monotone_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

theorem function_properties :
  smallest_period π f ∧ is_symmetric_about f (π / 3) ∧ is_monotone_on f (5 * π / 6) π :=
by
  -- The proof goes here
  sorry

end function_properties_l35_35537


namespace equation_of_line_passing_focus_and_parallel_l35_35616

-- Define parabola
def parabola : (ℝ × ℝ) → Prop := λ P, P.2 ^ 2 = 2 * P.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1 / 2, 0)

-- Define the characteristic of parallel lines
def is_parallel (l1 l2 : ℝ × ℝ → Prop) : Prop := ∃ (a b : ℝ), l1 = λ P, a * P.1 - b * P.2 = 0 ∧ l2 = λ P, a * P.1 - b * P.2 = 0

-- The line we are given to be parallel to
def given_line : ℝ × ℝ → Prop := λ P, 3 * P.1 - 2 * P.2 + 5 = 0

-- The line we need to find
def required_line : ℝ × ℝ → Prop := λ P, 6 * P.1 - 4 * P.2 - 3 = 0

theorem equation_of_line_passing_focus_and_parallel :
  (parabola focus) →
  is_parallel required_line given_line →
  required_line focus :=
by {
  -- Proof would go here. The steps would involve verifying the focus lies on the parabola
  -- and confirming that the lines are indeed parallel and the required line passes through the focus.
  sorry
}

end equation_of_line_passing_focus_and_parallel_l35_35616


namespace two_thousand_sixth_digit_of_3_div_7_l35_35044

-- Define the repeating decimal of 3/7
def repeating_decimal_3_div_7 : List ℕ := [4, 2, 8, 5, 7, 1]

-- Function to get the nth digit of the repeating decimal of 3/7
def repeating_decimal_digit (n : ℕ) : ℕ :=
  repeating_decimal_3_div_7[(n - 1) % repeating_decimal_3_div_7.length]

-- Define the mathematical proof problem
theorem two_thousand_sixth_digit_of_3_div_7 : repeating_decimal_digit 2006 = 2 :=
by
  sorry

end two_thousand_sixth_digit_of_3_div_7_l35_35044


namespace distance_from_focus_to_y_axis_l35_35244

theorem distance_from_focus_to_y_axis :
  ∀ (P : ℝ × ℝ), (P.1^2 = 4 * P.2) → (dist P (0, 1) = 3) → (abs P.1 = 2 * real.sqrt 2) := 
by
  intros P hp parabola_eq
  sorry

end distance_from_focus_to_y_axis_l35_35244


namespace smallest_two_digit_product_12_l35_35473

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l35_35473


namespace find_f1_f9_f96_l35_35818

noncomputable def f : ℕ+ → ℕ+
| ⟨k, h⟩ := sorry  -- Function definition has been omitted

theorem find_f1_f9_f96 :
  (strict_mono f) → 
  (∀ k : ℕ+, f (f k) = 3 * k) → 
  f 1 + f 9 + f 96 = 197 :=
by
  intros
  sorry

end find_f1_f9_f96_l35_35818


namespace smallest_two_digit_product_12_l35_35468

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l35_35468


namespace fixed_point_is_5_225_l35_35628

theorem fixed_point_is_5_225 : ∃ a b : ℝ, (∀ k : ℝ, 9 * a^2 + k * a - 5 * k = b) → (a = 5 ∧ b = 225) :=
by
  sorry

end fixed_point_is_5_225_l35_35628


namespace largest_prime_factor_of_binomial_l35_35092

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_prime_factor_of_binomial {p : ℕ} (hp : Nat.Prime p) (hp_range : 10 ≤ p ∧ p < 100) :
  p ∣ binomial 300 150 → p = 97 :=
by
suffices ∀ q : ℕ, Nat.Prime q → 10 ≤ q ∧ q < 100 → q ∣ binomial 300 150 → q ≤ 97
from fun h => le_antisymm (this p hp hp_range h) (le_of_eq (rfl : 97 = 97))
intro q hq hq_range hq_div
sorry

end largest_prime_factor_of_binomial_l35_35092


namespace prob_AC_adjacent_BE_not_adjacent_l35_35426

open Finset
open Perm
open Probability

-- Define the students as a Finset
def students : Finset (Fin 5) := {0, 1, 2, 3, 4}

-- Define that A is 0, B is 1, C is 2, D is 3, E is 4
def A := 0
def B := 1
def C := 2
def D := 3
def E := 4

-- Define the event of A and C being adjacent
def adjacent (x y : Fin 5) (p : List (Fin 5)) : Prop := 
  (List.indexOf x p + 1 = List.indexOf y p) ∨ (List.indexOf y p + 1 = List.indexOf x p)

-- Define the event of B and E not being adjacent
def not_adjacent (x y : Fin 5) (p : List (Fin 5)) : Prop := 
  ¬ adjacent x y p

-- Lean 4 statement: Calculate the probability that A and C are adjacent while B and E are not adjacent
noncomputable def probabilityAC_adjacent_BE_not_adjacent : ℚ :=
  let total_permutations := (univ.perm 5).toFinset.card
  let valid_permutations := (univ.filter (λ p, adjacent A C p ∧ not_adjacent B E p)).perm.toFinset.card
  valid_permutations / total_permutations

theorem prob_AC_adjacent_BE_not_adjacent : probabilityAC_adjacent_BE_not_adjacent = 1/5 :=
  sorry

end prob_AC_adjacent_BE_not_adjacent_l35_35426


namespace chord_perpendicular_exists_l35_35918

-- Definition of the circle and auxiliary circle
def Circle (O : Point) (r : ℝ) : Prop := sorry
def AuxiliaryCircle (O : Point) (R : ℝ) : Prop := sorry

-- Definition of a line in the plane
def Line (l : Set Point) : Prop := sorry

-- Intersection points on the circle
def intersects (c : Circle O r) (l : Line) : Set Point := sorry

theorem chord_perpendicular_exists (O : Point) (r : ℝ) (l : Line)
  (h1 : Circle O r) (h2 : AuxiliaryCircle O (3 * r)) :
  ∃ (A B : Point), (A ≠ B) ∧ (A ∈ intersects (Circle O r) l) ∧ (B ∈ intersects (Circle O r) l) ∧
  is_perpendicular (↔Line_A_B(A,B), l) ∧ (dist A B = dist A B) := sorry

end chord_perpendicular_exists_l35_35918


namespace digit_divisibility_l35_35074

-- Define the sum of digits function
def sum_digits (B : ℕ) : ℕ := 5 + 1 + 4 + B

-- Define the function to check divisibility by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Prove that B makes 514B divisible by 3
theorem digit_divisibility (B : ℕ) (hB : B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  divisible_by_3 (sum_digits B) ↔ B ∈ {2, 5, 8} :=
by
  -- Proof steps go here
  sorry

end digit_divisibility_l35_35074


namespace length_of_AC_is_correct_l35_35015
open_locale real

noncomputable def radius : ℝ := 7
noncomputable def chord_length : ℝ := 8
noncomputable def length_AC (r l : ℝ) : ℝ := sqrt (r ^ 2 + r ^ 2 - 2 * r ^ 2 * (sqrt(1 - ((l^ 2) / (4 * r ^ 2))) / 2))

theorem length_of_AC_is_correct : length_AC radius chord_length = sqrt(98 - 14 * sqrt 33) :=
by
  sorry

end length_of_AC_is_correct_l35_35015


namespace sports_arrangement_l35_35430

theorem sports_arrangement :
  let volleyball_stadiums := 4
  let basketball_stadiums := 4
  let football_stadiums := 4
  let total_arrangements := volleyball_stadiums * basketball_stadiums * football_stadiums
  let invalid_arrangements := 4
  total_arrangements - invalid_arrangements = 60 := by
sor                                                                                                                                                                                                                                                                                    Typography

Printing Options

File

Edit

View

Insert

Format

Tools

Help
Currently:  
English
    Insert:

MathProofProblem
TEXT
 \displaystyle \frac{MathProofProblem}{TEXTieto matches across three different sports, volleyball is planned to be held. Let there be 4 total venues and volleyball can only be played in one Venue each time. We need to find the total numbers of matches that can be arranged such that none of them planned more than two sports at the same venue.}

Fold Unfold

Toolbar
Basic
Math Toolbar
Math Toolbar
and
{
 Retrieve Document
   Please select a file from your device to retrieve its contents and replace the current deck.
Input File
         
Replace current

Generate Problem
 Export deck to a file on your device )
   
Rewindable Proofs
 Proofs can be rewound to observe their intermediate steps. Try evaluating smaller proof terms first.
   

 Submit
For those sequence of proofs which require rewinding, we offer convenient support.

Toggle Main Navigator
Currently showing Math Toolbar
Volleyball Matches 
9 steps
3 wanted
 currently planned for 500 problems

         Retry Scenario
 momenteel gepland voor 500 problemen
 help guide you retrieve existing response 
      and
 correct
  problem 
 Submit ProofAssentoretica) Identify all questions and conditions in the given problem.
MathProofProblem
TEXT
 \displaystyle Proposition(Problem)

Theorem:
Volleyball matches can be arranged in such a way that no more than two sports are allowed in a stadium such that there are 4 choices per stadium allowing.

Plan:
 
Given the total choices, let’s just assume volleyball matches can be arranged in 4 possible Stadiums
Since each sport can only be played one stadium at a time
Therefore we exclude those cases where there match at same stadium possibly

Thereby subtracting those invalid venues
Choose 4 Stadiums protecting no overlapping should take place via the Description. Mathematically they need prove equivalent Problem.
 
Submit Proof
lean
 
Update Problem  
Create Proof
Next
Alignment Problem ⟶ ⟵ Step in Advance

end sports_arrangement_l35_35430


namespace find_values_of_x_and_y_l35_35229

noncomputable def x := 3 * Real.sqrt 2
def y := 3
def equation :=
  ( ( ( 17.28 / x^2 ) * ( y.factorial / 4 ) ) / ( 3.6 * 0.2 ) ) = 2

theorem find_values_of_x_and_y 
  (pos_x : x > 0)
  (pos_y : y > 0)
  (int_y : y ∈ Int)
  (eq : equation) : 
  x = 3 * Real.sqrt 2 ∧ y = 3 :=
by
  sorry

end find_values_of_x_and_y_l35_35229


namespace remainder_abc_div9_l35_35698

theorem remainder_abc_div9 (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9) 
    (h1 : a + 2 * b + 3 * c ≡ 0 [MOD 9]) 
    (h2 : 2 * a + 3 * b + c ≡ 5 [MOD 9]) 
    (h3 : 3 * a + b + 2 * c ≡ 5 [MOD 9]) : 
    (a * b * c) % 9 = 0 := 
sorry

end remainder_abc_div9_l35_35698


namespace hyperbola_eccentricity_correct_l35_35541

-- conditions of the problem
variables {a b : ℝ} (ha : a > 0) (hb : b > 0)
def hyperbola (x y : ℝ) := (x^2) / (a^2) - (y^2) / (b^2) = 1

-- The question and definition of eccentricity
def eccentricity (a b : ℝ) : ℝ := 
  let c := 2 * b in
  c / a

theorem hyperbola_eccentricity_correct :
  eccentricity a b = (2 * Real.sqrt 3) / 3 :=
sorry

end hyperbola_eccentricity_correct_l35_35541


namespace width_of_smallest_room_l35_35821

-- Definitions based on given conditions
def largest_room_width : ℕ := 45
def largest_room_length : ℕ := 30
def smallest_room_length : ℕ := 8
def area_difference : ℕ := 1230

-- Mathematical statement to be proved
theorem width_of_smallest_room :
  let largest_room_area := largest_room_width * largest_room_length,
      smallest_room_area := largest_room_area - area_difference in
  smallest_room_area / smallest_room_length = 15 :=
begin
  sorry
end

end width_of_smallest_room_l35_35821


namespace part1_part2_l35_35878

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2

noncomputable def g (a x : ℝ) : ℝ := Real.exp x * f a x

def is_extreme_point (a x : ℝ) : Prop :=
  fderiv ℝ (λ x, f a x) x = 0

def is_monotonically_decreasing (a : ℝ) (I : Set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → g a x ≥ g a y

theorem part1 (a : ℝ) : 
  is_extreme_point a 2 → a = 1 := 
sorry

theorem part2 (a : ℝ) : 
  is_monotonically_decreasing a (Set.Icc 0 2) → a ≤ 6 / 5 := 
sorry

end part1_part2_l35_35878


namespace hours_per_trainer_l35_35410

-- Define the conditions from part (a)
def number_of_dolphins : ℕ := 4
def hours_per_dolphin : ℕ := 3
def number_of_trainers : ℕ := 2

-- Define the theorem we want to prove using the answer from part (b)
theorem hours_per_trainer : (number_of_dolphins * hours_per_dolphin) / number_of_trainers = 6 :=
by
  -- Proof goes here
  sorry

end hours_per_trainer_l35_35410


namespace largestSixDigitNumber_l35_35857

def isValidSixDigitNumber (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  digits.length = 6 ∧ 
  digits.nodup ∧ 
  (∀ d ∈ digits, d ≠ 0) ∧ 
  digits.sum = 21

theorem largestSixDigitNumber : ∃ n : ℕ, isValidSixDigitNumber n ∧
  ∀ m : ℕ, isValidSixDigitNumber m → n ≥ m :=
sorry

end largestSixDigitNumber_l35_35857


namespace total_games_played_l35_35022

def games_lost : ℕ := 4
def games_won : ℕ := 8

theorem total_games_played : games_lost + games_won = 12 :=
by
  -- Proof is omitted
  sorry

end total_games_played_l35_35022


namespace seq_10_is_4_l35_35976

-- Define the sequence with given properties
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 4
  | (n + 2) => if n % 2 = 0 then 4 else 3

-- Theorem statement: The 10th term of the sequence is 4
theorem seq_10_is_4 : seq 9 = 4 :=
by sorry

end seq_10_is_4_l35_35976


namespace red_exterior_possible_l35_35160

theorem red_exterior_possible (a : ℕ) (h₁ : ∃ n, n = 8 * (1/3) * 6 * a^2)
  (h₂ : ∀ n, n = (2 * a)^2 → 6 * n = 24 * a^2)
  (h₃ : ∃ n, n = 8 * a^2) 
  (h₄ : ∀ n, n = (1/3) * 24 * a^2) :
  ∃ (cube_assembly : cube_assembly a),
  red_exterior cube_assembly = true := 
by
  sorry

end red_exterior_possible_l35_35160


namespace monotonicity_of_g_f_monotonically_increasing_on_R_f_bound_on_interval_l35_35636

namespace LeanMathProblem

-- Define the function f and g as given
def f (a : ℝ) (x : ℝ) : ℝ := exp x - (1 / 2) * x ^ 2 - a * sin x - 1
def g (a : ℝ) (x : ℝ) : ℝ := f a x + f a (-x)

-- Condition: -1 ≤ a ≤ 1
variables (a : ℝ) (ha : -1 ≤ a ∧ a ≤ 1)

-- Statement Ⅰ: Monotonicity of g(x)
theorem monotonicity_of_g :
  (∀ x, g a x ≤ g a 0) ∨ (∀ x, g a x ≥ g a 0) :=
sorry

-- Statement Ⅱ (i): f(x) is monotonically increasing on ℝ
theorem f_monotonically_increasing_on_R :
  (∀ x y, x ≤ y → f a x ≤ f a y) :=
sorry

-- Statement Ⅱ (ii): For x ∈ [-π/3, π/3], |f'(x)| ≤ M implies |f(x)| ≤ M
theorem f_bound_on_interval (M : ℝ) :
  (∀ x ∈ Icc (-π/3) (π/3), abs (derivative (f a) x) ≤ M → abs (f a x) ≤ M) :=
sorry

end LeanMathProblem

end monotonicity_of_g_f_monotonically_increasing_on_R_f_bound_on_interval_l35_35636


namespace range_of_dot_product_equation_of_line_l35_35313

section part1

variable (x0 y0 : ℝ)

def is_on_ellipse (x : ℝ) (y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

def f1 : ℝ × ℝ := (-Real.sqrt 3, 0)
def f2 : ℝ × ℝ := (Real.sqrt 3, 0)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  (b.1 - a.1, b.2 - a.2)

theorem range_of_dot_product (h : is_on_ellipse x0 y0) :
  -2 ≤ dot_product (vector (x0, y0) f1) (vector (x0, y0) f2) ∧
  dot_product (vector (x0, y0) f1) (vector (x0, y0) f2) ≤ 1 :=
sorry

end part1

section part2

variable (k m : ℝ)

def is_right_isosceles_triangle (A B D : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = -1 ∧
  ((D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2)) = 0 ∧
  ((B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2)

def line_eqn (x y : ℝ) : Prop :=
  y = k * x + m

def is_on_line (x y : ℝ) : Prop :=
  line_eqn k m x y

def is_on_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2) = 1

theorem equation_of_line
  (h_triangle : ∃ B D, is_right_isosceles_triangle (0, -1) B D ∧
                       is_on_ellipse B.1 B.2 ∧ is_on_ellipse D.1 D.2 ∧
                       is_on_line B.1 B.2 ∧ is_on_line D.1 D.2) :
  (m = 3 / 5 ∧ (k = Real.sqrt 5 / 5 ∨ k = -Real.sqrt 5 / 5)) :=
sorry

end part2

end range_of_dot_product_equation_of_line_l35_35313


namespace nancy_packs_of_crayons_l35_35372

theorem nancy_packs_of_crayons (total_crayons : ℕ) (crayons_per_pack : ℕ) (h1 : total_crayons = 615) (h2 : crayons_per_pack = 15) : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end nancy_packs_of_crayons_l35_35372


namespace range_of_a_l35_35278

noncomputable def f (x : ℝ) := real.sqrt (x^2 - 1)

def domain_set (x : ℝ) := x ≥ 1 ∨ x ≤ -1

def set_B (a x : ℝ) := 1 < a * x ∧ a * x < 2

theorem range_of_a (a : ℝ) : (∀ x, set_B a x → domain_set x) ↔ a ∈ Icc (-1:ℝ) 1 :=
by
  sorry

end range_of_a_l35_35278


namespace exists_infinite_primes_dividing_x_m_l35_35742

noncomputable theory

def P (n : ℕ) : ℕ := sorry -- Polynomial P(n) with non-negative integer coefficients
def Q (n : ℕ) : ℕ := sorry -- Polynomial Q(n) with non-negative integer coefficients
def x_n (n : ℕ) : ℕ := 2016 ^ (P n) + Q n

def is_squarefree (m : ℕ) : Prop :=
  ∀ p : ℕ, (p : ℕ).prime → p ^ 2 ∣ m → False

theorem exists_infinite_primes_dividing_x_m :
  ∃^∞ p, ∃ m : ℕ, is_squarefree m ∧ p ∣ x_n m := 
sorry

end exists_infinite_primes_dividing_x_m_l35_35742


namespace calculate_expression_l35_35575

theorem calculate_expression :
  (8 ^ (-2 / 3) + log 10 100 - (-7 / 8) ^ 0) = (5 / 4) :=
by
  have h1 : 8 = 2 ^ 3 := by norm_num
  have h2 : log 10 100 = 2 := by norm_num
  have h3 : (-7 / 8) ^ 0 = 1 := by norm_num
  sorry  -- provide the necessary mathematical proof here

end calculate_expression_l35_35575


namespace math_problem_l35_35752

open Classical

theorem math_problem (s x y : ℝ) (h₁ : s > 0) (h₂ : x^2 + y^2 ≠ 0) (h₃ : x * s^2 < y * s^2) :
  ¬(-x^2 < -y^2) ∧ ¬(-x^2 < y^2) ∧ ¬(x^2 < -y^2) ∧ ¬(x^2 > y^2) := by
  sorry

end math_problem_l35_35752


namespace rectangle_area_approx_33p88_l35_35822

-- Define the conditions and the proof problem in Lean 4.
theorem rectangle_area_approx_33p88
  (w : ℝ) -- Let 'w' be the width of the rectangle
  (hl : ∀ w, l = 3 * w + 1 / 2 * w) -- Define the relationship of length to width
  (P : ℝ) -- Perimeter of the rectangle
  (hP : P = 28) -- The given perimeter
  (hP_formula : P = 2 * (3.5 * w) + 2 * w) -- Relationship of perimeter in terms of width
  (hw : w = 28 / 9) -- Solve for width
  (hl_formula : l = 3.5 * w) -- Find the length using width
  (A : ℝ) -- Define the area of the rectangle
  (hA_formula : A = (3.5 * w) * w) -- Relationship of area in terms of width and length
  : A ≈ 33.88 := -- Prove that the area is approximately 33.88 square inches
sorry

end rectangle_area_approx_33p88_l35_35822


namespace largest_prime_factor_of_binomial_l35_35090

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_prime_factor_of_binomial {p : ℕ} (hp : Nat.Prime p) (hp_range : 10 ≤ p ∧ p < 100) :
  p ∣ binomial 300 150 → p = 97 :=
by
suffices ∀ q : ℕ, Nat.Prime q → 10 ≤ q ∧ q < 100 → q ∣ binomial 300 150 → q ≤ 97
from fun h => le_antisymm (this p hp hp_range h) (le_of_eq (rfl : 97 = 97))
intro q hq hq_range hq_div
sorry

end largest_prime_factor_of_binomial_l35_35090


namespace rahim_average_price_per_book_l35_35111

theorem rahim_average_price_per_book :
  let books1 := 65
  let cost1 := 1160 -- Rs.
  let books2 := 50
  let cost2 := 920 -- Rs.
  let total_books := books1 + books2
  let total_cost := cost1 + cost2
  total_books ≠ 0 → 
  total_cost / total_books = 18.09 :=
by
  let books1 := 65
  let cost1 := 1160 -- Rs.
  let books2 := 50
  let cost2 := 920 -- Rs.
  let total_books := books1 + books2
  let total_cost := cost1 + cost2
  have h1 : total_books = 115 := by simp [total_books]
  have h2 : total_cost = 2080 := by simp [total_cost]
  have h3 : total_books ≠ 0 := by norm_num [total_books, h1]
  have h4 : total_cost / total_books = 2080 / 115 := by rw [h1, h2]
  have h5 : 2080 / 115 = 18.09 := by norm_num
  exact h5

end rahim_average_price_per_book_l35_35111


namespace amount_for_gifts_and_charitable_causes_l35_35963

namespace JillExpenses

def net_monthly_salary : ℝ := 3700
def discretionary_income : ℝ := 0.20 * net_monthly_salary -- 1/5 * 3700
def vacation_fund : ℝ := 0.30 * discretionary_income
def savings : ℝ := 0.20 * discretionary_income
def eating_out_and_socializing : ℝ := 0.35 * discretionary_income
def gifts_and_charitable_causes : ℝ := discretionary_income - (vacation_fund + savings + eating_out_and_socializing)

theorem amount_for_gifts_and_charitable_causes : gifts_and_charitable_causes = 111 := sorry

end JillExpenses

end amount_for_gifts_and_charitable_causes_l35_35963


namespace triangle_area_le_half_l35_35650

/-- Given a square with side length n containing (n+1)^2 points,
    none of which are collinear, prove that among any three points 
    chosen from these, the area of the triangle formed does not 
    exceed 1/2. -/
theorem triangle_area_le_half (n : ℕ) (points : Fin (n+1)^2 → ℝ × ℝ)
  (h_points : ∀ (i j k : Fin (n+1)^2), i ≠ j → j ≠ k → i ≠ k →
    ¬ collinear ℝ {points i, points j, points k}) :
  ∃ (a b c : Fin (n+1)^2), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  area_of_triangle (points a) (points b) (points c) ≤ 1 / 2 :=
sorry

end triangle_area_le_half_l35_35650


namespace EllipseEquation_DistanceToDirectrix_l35_35726

open Real

-- Definition of the Ellipse
def ellipse (a b : ℝ) : Prop := 
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

-- Define the midpoint
def midpoint (A C : ℝ × ℝ) : ℝ × ℝ := 
  ( (fst A + fst C) / 2, (snd A + snd C) / 2 )

-- Define the distance function
def distance (D: ℝ × ℝ) (directrix: ℝ) : ℝ :=
  abs (fst D - directrix)

-- The main theorems
theorem EllipseEquation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
                        (A B C F : ℝ × ℝ) (F_left : F = (-1, 0)) 
                        (A_left : A = (-a, 0)) (B_upper: B = (0, b)) (C_lower: C = (0, -b))
                        (BF_slope: 1) 
                        (M: midpoint A C = (-a/2, -b/2)) :
  ellipse 3 (sqrt 8) := 
sorry

theorem DistanceToDirectrix (D : ℝ × ℝ) (a b : ℝ) (a_pos : a = sqrt 2) (b_pos : b = 1)
                            (A B C F: ℝ × ℝ) (F_left : F = (-1, 0))
                            (A_left : A = (-a, 0)) (B_upper : B = (0, b)) (C_lower : C = (0, -b))
                            (BF_slope: 1) 
                            (BF_intersect: ∃ y, (y = 1 + y) ∧ (distance D 2 = 10 / 3)) : 
  distance D 2 = 10 / 3 :=
sorry

end EllipseEquation_DistanceToDirectrix_l35_35726


namespace yellow_tint_percentage_in_new_mixture_l35_35894

-- Definitions based on conditions:
def original_mixture_volume : ℕ := 50
def yellow_tint_percentage : ℝ := 0.25
def added_yellow_tint : ℝ := 10

-- Calculation of yellow tint in the original mixture:
def original_yellow_tint := yellow_tint_percentage * original_mixture_volume

-- New yellow tint amount after addition:
def new_yellow_tint := original_yellow_tint + added_yellow_tint

-- New total mixture volume:
def new_total_volume := original_mixture_volume + added_yellow_tint

-- Percentage calculation:
def yellow_tint_percentage_new := (new_yellow_tint / new_total_volume) * 100

-- The proof problem statement:
theorem yellow_tint_percentage_in_new_mixture :
  yellow_tint_percentage_new = 37.5 := 
sorry

end yellow_tint_percentage_in_new_mixture_l35_35894


namespace total_volume_of_mixture_l35_35001

theorem total_volume_of_mixture 
    (V_A V_B : ℝ)
    (hV_A : V_A = 10)
    (percentage_A : ℝ := 0.20)
    (percentage_B : ℝ := 0.50)
    (final_percentage : ℝ := 0.30) 
    : V_A + V_B = 15 := 
by
  -- Conditions
  have h1 : percentage_A * V_A + percentage_B * V_B = final_percentage * (V_A + V_B),
  { sorry },
  -- Substitution of given V_A
  have h2 : percentage_A * 10 + percentage_B * V_B = final_percentage * (10 + V_B),
  { rw hV_A, exact h1 },
  -- Solving for V_B
  have h3 : 2 + 0.50 * V_B = 3 + 0.30 * V_B,
  { sorry },
  have h4 : 0.20 * V_B = 1,
  { linarith at h3 },
  have h5 : V_B = 5,
  { field_simp at h4, exact h4 },
  -- Calculate total volume
  show V_A + V_B = 10 + 5,
  { rw [hV_A, h5], }

end total_volume_of_mixture_l35_35001


namespace inverse_of_g_at_17_over_40_l35_35699

def g (x : ℝ) : ℝ := (x^3 - 3*x) / 5

theorem inverse_of_g_at_17_over_40 (c : ℝ) : g c = 17 / 40 := sorry

end inverse_of_g_at_17_over_40_l35_35699


namespace log_base_7_cube_root_7_l35_35209

theorem log_base_7_cube_root_7 : log 7 (7 ^ (1 / 3)) = 1 / 3 := 
by sorry

end log_base_7_cube_root_7_l35_35209


namespace smallest_two_digit_product_12_l35_35470

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l35_35470


namespace incorrect_statement_A_l35_35637

variables (α β : Plane) (m n : Line)

-- Statement of the problem with the given conditions
def plane_parallel_correct (hαβ : ∀ m, m ∈ β → m ∩ α = n) : Prop := m ∈ α ∧ α ∩ β = n → m ∥ n

theorem incorrect_statement_A :
  ¬ plane_parallel_correct (λ m h, _) :=
sorry

end incorrect_statement_A_l35_35637


namespace exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared_l35_35962

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared :
  ∃ (n : ℕ), sum_of_digits n = 1000 ∧ sum_of_digits (n ^ 2) = 1000000 := sorry

end exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared_l35_35962


namespace largest_2_digit_prime_factor_of_binom_l35_35077

open Nat

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

/-- Definition of the problem conditions -/
def problem_conditions : Prop :=
  let n := binom 300 150
  ∃ p : ℕ, Prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p ≤ 75 ∨ 3 * p < 300) ∧ p = 97

/-- Statement of the proof problem -/
theorem largest_2_digit_prime_factor_of_binom : problem_conditions := 
  sorry

end largest_2_digit_prime_factor_of_binom_l35_35077


namespace Teresa_age_at_Michiko_birth_l35_35805

-- Definitions of the conditions
def Teresa_age_now : ℕ := 59
def Morio_age_now : ℕ := 71
def Morio_age_at_Michiko_birth : ℕ := 38

-- Prove that Teresa was 26 years old when she gave birth to Michiko.
theorem Teresa_age_at_Michiko_birth : 38 - (71 - 59) = 26 := by
  -- Provide the proof here
  sorry

end Teresa_age_at_Michiko_birth_l35_35805


namespace smallest_positive_period_monotonically_increasing_intervals_symmetry_center_l35_35675

noncomputable def f (x : ℝ) : ℝ := cos x * sin (x + π / 3) - sqrt 3 * cos x ^ 2 + sqrt 3 / 4

-- Prove the smallest positive period of f(x) is π
theorem smallest_positive_period (T : ℝ) : (∀ x : ℝ, f(x) = f(x + T)) ∧ T > 0 → T = π :=
sorry

-- Prove the intervals where f(x) is monotonically increasing
theorem monotonically_increasing_intervals (k : ℤ) : 
  ∃ (a b : ℝ), (a = k * π - π / 12) ∧ (b = k * π + 5 * π / 12) ∧ 
  (∀ x1 x2 : ℝ, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f(x1) < f(x2)) :=
sorry

-- Prove the symmetry center of f(x)
theorem symmetry_center (k : ℤ) : 
  ∃ (a : ℝ), (a = k * π / 2 + π / 6) ∧ ∀ x : ℝ, f(a - x) = f(a + x) :=
sorry

end smallest_positive_period_monotonically_increasing_intervals_symmetry_center_l35_35675


namespace find_CF_and_BC_l35_35344

-- Definitions based on conditions

def Point := {x : ℝ // x ≠ 0} -- Placeholder definition for a point; should be extended properly

-- Known lengths
def AB : ℝ := 6
def CD : ℝ := 10
def AF : ℝ := 7

-- Proving the required lengths
theorem find_CF_and_BC 
  (C D : Point)
  (h1 : ∀ (A F : Point), C.x ≠ F.x ∧ C.x ≠ A.x)
  (h2 : ∀ (A F : Point), (D.x = A.x) ∧ (D.x = F.x) ∧ (C.x ≠ A.x) ∧ (C.x ≠ F.x))
  (B : Point) (h3 : ∃ (B : Point), B.x ≠ C.x)
  (E : Point) (h4 : ∀ (B C D : Point), E.x ≠ D.x) :
  ∃ (CF BC : ℝ), CF = 35 / 3 ∧ BC = 170 / Real.sqrt 611 :=
by
  sorry

end find_CF_and_BC_l35_35344


namespace slope_of_line_is_2_l35_35705

def Point : Type := ℝ × ℝ

def A : Point := (1, 3)
def B : Point := (2, 5)

def line_slope (P Q : Point) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

theorem slope_of_line_is_2 : line_slope A B = 2 :=
by
  sorry

end slope_of_line_is_2_l35_35705


namespace winston_cents_left_l35_35499

-- Definitions based on the conditions in the problem
def quarters := 14
def cents_per_quarter := 25
def half_dollar_in_cents := 50

-- Formulation of the problem statement in Lean
theorem winston_cents_left : (quarters * cents_per_quarter) - half_dollar_in_cents = 300 :=
by sorry

end winston_cents_left_l35_35499


namespace claudia_total_earnings_l35_35190

def cost_per_beginner_class : Int := 15
def cost_per_advanced_class : Int := 20
def num_beginner_kids_saturday : Int := 20
def num_advanced_kids_saturday : Int := 10
def num_sibling_pairs : Int := 5
def sibling_discount : Int := 3

theorem claudia_total_earnings : 
  let beginner_earnings_saturday := num_beginner_kids_saturday * cost_per_beginner_class
  let advanced_earnings_saturday := num_advanced_kids_saturday * cost_per_advanced_class
  let total_earnings_saturday := beginner_earnings_saturday + advanced_earnings_saturday
  
  let num_beginner_kids_sunday := num_beginner_kids_saturday / 2
  let num_advanced_kids_sunday := num_advanced_kids_saturday / 2
  let beginner_earnings_sunday := num_beginner_kids_sunday * cost_per_beginner_class
  let advanced_earnings_sunday := num_advanced_kids_sunday * cost_per_advanced_class
  let total_earnings_sunday := beginner_earnings_sunday + advanced_earnings_sunday

  let total_earnings_no_discount := total_earnings_saturday + total_earnings_sunday

  let total_sibling_discount := num_sibling_pairs * 2 * sibling_discount
  
  let total_earnings := total_earnings_no_discount - total_sibling_discount
  total_earnings = 720 := 
by
  sorry

end claudia_total_earnings_l35_35190


namespace fraction_area_outside_circle_l35_35887

theorem fraction_area_outside_circle (r : ℝ) (h1 : r > 0) :
  let side_length := 2 * r
  let area_square := side_length ^ 2
  let area_circle := π * r ^ 2
  let area_outside := area_square - area_circle
  (area_outside / area_square) = 1 - ↑π / 4 :=
by
  sorry

end fraction_area_outside_circle_l35_35887


namespace train_cross_bridge_time_l35_35154

theorem train_cross_bridge_time (train_length : ℤ) (train_speed_kmhr : ℝ) (total_length : ℤ) : 
  train_length = 130 ∧ train_speed_kmhr = 45 ∧ total_length = 245 → 
  let train_speed_ms := (train_speed_kmhr * 1000) / 3600 in
  let time := (total_length) / train_speed_ms in
  time = 9.8 :=
by
  intros h
  rcases h with ⟨hl, hs, ht⟩
  have train_speed_ms_calc : (train_speed_kmhr * 1000) / 3600 = 25 := by sorry
  have time_calc : (total_length : ℝ) / 25 = 9.8 := by sorry
  rw [train_speed_ms_calc, time_calc]
  sorry

end train_cross_bridge_time_l35_35154


namespace total_eggs_collected_l35_35177

def benjamin_collects : Nat := 6
def carla_collects := 3 * benjamin_collects
def trisha_collects := benjamin_collects - 4

theorem total_eggs_collected :
  benjamin_collects + carla_collects + trisha_collects = 26 := by
  sorry

end total_eggs_collected_l35_35177


namespace isosceles_trapezoid_area_l35_35451

theorem isosceles_trapezoid_area (a b l : ℝ)
  (ha : a = 8) (hb : b = 14) (hl : l = 5) : 
  let h := real.sqrt (l^2 - ((b - a) / 2)^2),
      area := (1 / 2) * (a + b) * h 
  in area = 44 := 
by
  have ha_8 : a = 8 := ha
  have hb_14 : b = 14 := hb
  have hl_5 : l = 5 := hl
  sorry

end isosceles_trapezoid_area_l35_35451


namespace range_of_g_l35_35960

open Real

-- Define the function g(x)
def g (x : ℝ) : ℝ := arcsin x + arccos x - arctan x

-- Statement of the problem in Lean 4
theorem range_of_g : set.Icc (π / 4) (3 * π / 4) = {y : ℝ | ∃ x : ℝ, x ∈ set.Icc (-1 : ℝ) (1 : ℝ) ∧ g x = y} :=
by sorry

end range_of_g_l35_35960


namespace reciprocal_neg_one_thirteen_l35_35415

theorem reciprocal_neg_one_thirteen : -(1:ℝ) / 13⁻¹ = -13 := 
sorry

end reciprocal_neg_one_thirteen_l35_35415


namespace increase_in_rectangle_area_l35_35865

theorem increase_in_rectangle_area (L B : ℝ) :
  let L' := 1.11 * L
  let B' := 1.22 * B
  let original_area := L * B
  let new_area := L' * B'
  let area_increase := new_area - original_area
  let percentage_increase := (area_increase / original_area) * 100
  percentage_increase = 35.42 :=
by
  sorry

end increase_in_rectangle_area_l35_35865


namespace percent_difference_z_w_l35_35305

theorem percent_difference_z_w (w x y z : ℝ)
  (h1 : w = 0.60 * x)
  (h2 : x = 0.60 * y)
  (h3 : z = 0.54 * y) :
  (z - w) / w * 100 = 50 := by
sorry

end percent_difference_z_w_l35_35305


namespace tenth_term_of_sequence_l35_35964

noncomputable def sequence : ℕ → ℚ
| 0       := 3
| 1       := 4
| (n + 2) := 12 / (sequence n)

theorem tenth_term_of_sequence : sequence 9 = 4 :=
by
  sorry

end tenth_term_of_sequence_l35_35964


namespace smallest_m_for_integral_solutions_l35_35858

theorem smallest_m_for_integral_solutions :
  ∃ (m : ℕ), (∀ x : ℤ, 10 * x^2 - (m : ℤ) * x + 630 = 0 → x ∈ ℤ) ∧ m = 160 :=
begin
  sorry

end smallest_m_for_integral_solutions_l35_35858


namespace func_a_odd_func_b_neither_func_c_neither_func_d_even_func_e_neither_func_f_odd_func_g_neither_func_h_odd_func_i_neither_func_j_odd_l35_35330

noncomputable def func_a (x : ℝ) : ℝ := Real.sin x ^ 3 + (Real.cot x) ^ 5
noncomputable def func_b (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (3 * x)
noncomputable def func_c (x : ℝ) : ℝ := (1 - Real.sin x) / (1 + Real.sin x)
noncomputable def func_d (x : ℝ) : ℝ := Real.sin x ^ 4 + x ^ 2 + 1
noncomputable def func_e (x : ℝ) : ℝ := x + Real.sqrt x
noncomputable def func_f (x : ℝ) : ℝ := x * abs x
noncomputable def func_g (x : ℝ) : ℝ := Real.arccos (3 * x)
noncomputable def func_h (x : ℝ) : ℝ := 5 * Real.arctan x
noncomputable def func_i (x : ℝ) : ℝ := -Real.arcctg x
noncomputable def func_j (x : ℝ) : ℝ := (3 ^ x - 1) / (3 ^ x + 1)

theorem func_a_odd : ∀ x, func_a (-x) = -func_a x := sorry
theorem func_b_neither : ¬(∀ x, func_b (-x) = func_b x) ∧ ¬(∀ x, func_b (-x) = -func_b x) := sorry
theorem func_c_neither : ¬(∀ x, func_c (-x) = func_c x) ∧ ¬(∀ x, func_c (-x) = -func_c x) := sorry
theorem func_d_even : ∀ x, func_d (-x) = func_d x := sorry
theorem func_e_neither : ¬(∀ x, func_e (-x) = func_e x) ∧ ¬(∀ x, func_e (-x) = -func_e x) := sorry
theorem func_f_odd : ∀ x, func_f (-x) = -func_f x := sorry
theorem func_g_neither : ¬(∀ x, func_g (-x) = func_g x) ∧ ¬(∀ x, func_g (-x) = -func_g x) := sorry
theorem func_h_odd : ∀ x, func_h (-x) = -func_h x := sorry
theorem func_i_neither : ¬(∀ x, func_i (-x) = func_i x) ∧ ¬(∀ x, func_i (-x) = -func_i x) := sorry
theorem func_j_odd : ∀ x, func_j (-x) = -func_j x := sorry

end func_a_odd_func_b_neither_func_c_neither_func_d_even_func_e_neither_func_f_odd_func_g_neither_func_h_odd_func_i_neither_func_j_odd_l35_35330


namespace correct_answer_l35_35114

-- Definitions of the sets
def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℤ := {-2, 2}

-- The proof statement which we need to prove
theorem correct_answer : M ∩ N = {2} := by
  sorry

end correct_answer_l35_35114


namespace number_of_valid_rods_l35_35919

theorem number_of_valid_rods :
  let rods := {n | 1 ≤ n ∧ n ≤ 50}.erase 10 .erase 20 .erase 30 in
  ∃ count, count = rods.card ∧ count = 56 := by
  sorry

end number_of_valid_rods_l35_35919


namespace ratio_of_sphere_surface_areas_l35_35831

noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := a / 2
noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem ratio_of_sphere_surface_areas (a : ℝ) (h : 0 < a) : 
  (sphere_surface_area (circumscribed_sphere_radius a)) / (sphere_surface_area (inscribed_sphere_radius a)) = 3 :=
by
  sorry

end ratio_of_sphere_surface_areas_l35_35831


namespace intersecting_segments_l35_35374

theorem intersecting_segments (n : ℕ) (segments : fin (n + 1) → set ℝ)
  (common_point : ℝ) (h_segments : ∀ i, common_point ∈ segments i) :
  ∃ (I J : fin (n + 1)), I ≠ J ∧
  let d := real.bsupr (λ x, x ∈ segments I)
  in real.bsupr (λ x, x ∈ (segments I ∩ segments J)) ≥ (n-1) / n * d :=
by
  sorry

end intersecting_segments_l35_35374


namespace profit_high_demand_profit_moderate_demand_profit_low_demand_l35_35557

-- Define the profit calculation based on given conditions
def original_profit (meters : ℕ) (profit_per_meter : ℕ) : ℕ :=
  meters * profit_per_meter

def discount (original_profit : ℕ) (percent : ℕ) : ℕ :=
  (original_profit * percent) / 100

def tax (discounted_profit : ℕ) (percent : ℕ) : ℕ :=
  (discounted_profit * percent) / 100

def final_profit (original_profit : ℕ) (discount_percent : ℕ) (tax_percent : ℕ) : ℕ :=
  let after_discount := original_profit - discount(original_profit, discount_percent)
  in after_discount - tax(after_discount, tax_percent)

-- Conditions
def meters_high := 40
def meters_low := 30
def profit_per_meter := 35
def discount_high := 10
def discount_moderate := 5
def sales_tax := 5

-- Theorem statements for trader's profit in each scenario
theorem profit_high_demand : final_profit (original_profit meters_high profit_per_meter) discount_high sales_tax = 1197 := 
sorry

theorem profit_moderate_demand : final_profit (original_profit meters_high profit_per_meter) discount_moderate sales_tax = 1263.5 := 
sorry

theorem profit_low_demand : final_profit (original_profit meters_low profit_per_meter) 0 sales_tax = 997.5 := 
sorry

end profit_high_demand_profit_moderate_demand_profit_low_demand_l35_35557


namespace smallest_two_digit_product_12_l35_35471

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l35_35471


namespace correct_distribution_l35_35841

def student_distribution : Prop :=
  ∃ d x y: ℕ, 
  d + x + y = 33 ∧ 
  y = 2 * x ∧ 
  d : x = 5 ∧ x : y = 1 : 4 ∧ 
  d = 15 ∧ x = 6 ∧ y = 12

theorem correct_distribution : student_distribution :=
  sorry

end correct_distribution_l35_35841


namespace dice_product_multiple_of_four_probability_l35_35778

theorem dice_product_multiple_of_four_probability:
  let dice_values := {1, 2, 3, 4} 
  in let total_outcomes := 4^4
  in let favorable_outcomes := 
        (208: ℝ)  
  in (favorable_outcomes / total_outcomes) = (13 / 16) := 
by
  sorry

end dice_product_multiple_of_four_probability_l35_35778


namespace correct_relation_is_identity_l35_35564

theorem correct_relation_is_identity : 0 = 0 :=
by {
  -- Skipping proof steps as only statement is required
  sorry
}

end correct_relation_is_identity_l35_35564


namespace winston_cents_left_l35_35496

def initial_amount_quarters (num_quarters : Nat) (value_per_quarter : Nat) : Nat :=
-num_quarters * value_per_quarter

def amount_spent (dollar_spent : Nat) : Nat := 
-dollar_spent * 50

theorem winston_cents_left (quarters : Nat) (value_per_quarter : Nat) (dollar_spent : Nat) : 
quarters = 14 ∧ value_per_quarter = 25 ∧ dollar_spent = 1/2 → 
initial_amount_quarters quarters value_per_quarter - amount_spent 1/2 = 300 :=
by 
intro h
cases h with h1 h_value_per_quarter 
cases h_value_per_quarter with h2 h_dollar_spent 
have h_amount_quarters : initial_amount_quarters 14 25 = 350 := 
by norm_num
have h_amount_spent : amount_spent 1 = 50 := 
by norm_num
rw [h_amount_quarters, h_amount_spent] 
norm_num
sorry

end winston_cents_left_l35_35496


namespace reflect_center_is_image_center_l35_35396

def reflect_over_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

theorem reflect_center_is_image_center : 
  reflect_over_y_eq_neg_x (3, -4) = (4, -3) :=
by
  -- Proof is omitted as per instructions.
  -- This proof would show the reflection of the point (3, -4) over the line y = -x resulting in (4, -3).
  sorry

end reflect_center_is_image_center_l35_35396


namespace avg_variance_stability_excellent_performance_probability_l35_35311

-- Define the scores of players A and B in seven games
def scores_A : List ℕ := [26, 28, 32, 22, 37, 29, 36]
def scores_B : List ℕ := [26, 29, 32, 28, 39, 29, 27]

-- Define the mean and variance calculations
def mean (scores : List ℕ) : ℚ := (scores.sum : ℚ) / scores.length
def variance (scores : List ℕ) : ℚ := 
  (scores.map (λ x => (x - mean scores) ^ 2)).sum / scores.length

theorem avg_variance_stability :
  mean scores_A = 30 ∧ mean scores_B = 30 ∧
  variance scores_A = 174 / 7 ∧ variance scores_B = 116 / 7 ∧
  variance scores_A > variance scores_B := 
by
  sorry

-- Define the probabilities of scoring higher than 30
def probability_excellent (scores : List ℕ) : ℚ := 
  (scores.filter (λ x => x > 30)).length / scores.length

theorem excellent_performance_probability :
  probability_excellent scores_A = 3 / 7 ∧ probability_excellent scores_B = 2 / 7 ∧
  (probability_excellent scores_A * probability_excellent scores_B = 6 / 49) :=
by
  sorry

end avg_variance_stability_excellent_performance_probability_l35_35311


namespace matrix_product_is_zero_l35_35945

-- Define the two matrices
def A (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -c], ![-d, 0, b], ![c, -b, 0]]

def B (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![d^2, b * d, c * d], ![b * d, b^2, b * c], ![c * d, b * c, c^2]]

-- Define the zero matrix
def zero_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 0, 0], ![0, 0, 0]]

-- The theorem to prove
theorem matrix_product_is_zero (b c d : ℝ) : A b c d * B b c d = zero_matrix :=
by sorry

end matrix_product_is_zero_l35_35945


namespace meaningful_expression_l35_35301

theorem meaningful_expression (x : ℝ) : (1 / (x - 2) ≠ 0) ↔ (x ≠ 2) :=
by
  sorry

end meaningful_expression_l35_35301


namespace winston_cents_left_l35_35497

def initial_amount_quarters (num_quarters : Nat) (value_per_quarter : Nat) : Nat :=
-num_quarters * value_per_quarter

def amount_spent (dollar_spent : Nat) : Nat := 
-dollar_spent * 50

theorem winston_cents_left (quarters : Nat) (value_per_quarter : Nat) (dollar_spent : Nat) : 
quarters = 14 ∧ value_per_quarter = 25 ∧ dollar_spent = 1/2 → 
initial_amount_quarters quarters value_per_quarter - amount_spent 1/2 = 300 :=
by 
intro h
cases h with h1 h_value_per_quarter 
cases h_value_per_quarter with h2 h_dollar_spent 
have h_amount_quarters : initial_amount_quarters 14 25 = 350 := 
by norm_num
have h_amount_spent : amount_spent 1 = 50 := 
by norm_num
rw [h_amount_quarters, h_amount_spent] 
norm_num
sorry

end winston_cents_left_l35_35497


namespace range_of_a_l35_35259

theorem range_of_a (f : ℝ → ℝ)
  (h1 : ∀ x < 1, f x = (2 * a - 1) * x + 4 * a)
  (h2 : ∀ x ≥ 1, f x = Real.log x / Real.log a)
  (h_decreasing : ∀ x y, x < y → f x ≥ f y) :
  a ∈ Icc (1/6 : ℝ) (1/2 : ℝ) :=
sorry

end range_of_a_l35_35259


namespace spring_expenses_l35_35043

noncomputable def expense_by_end_of_february : ℝ := 0.6
noncomputable def expense_by_end_of_may : ℝ := 1.8
noncomputable def spending_during_spring_months := expense_by_end_of_may - expense_by_end_of_february

-- Lean statement for the proof problem
theorem spring_expenses : spending_during_spring_months = 1.2 := by
  sorry

end spring_expenses_l35_35043


namespace three_teams_not_competed_l35_35119

theorem three_teams_not_competed :
  ∀ (T : Type) (teams : set T) (R : T → T → Prop),
  -- Number of teams is 518
  (set.finite teams ∧ set.card teams = 518) →
  -- Each team competes with another in each round, 8 rounds completed
  (∀ (t1 t2 : T), R t1 t2 → t1 ∈ teams ∧ t2 ∈ teams) →
  (∀ t : T, ∃ s : set T, set.card s = 8 ∧ (∀ x ∈ s, R t x)) →
  -- Teams that have already competed do not compete again
  (∀ (r1 r2 : T), r1 ≠ r2 → (R r1 r2 → ¬R r2 r1)) →
  -- Conclusion: There exists a set of 3 teams that have not competed amongst each other
  ∃ (s : set T), set.card s = 3 ∧ (∀ t1 t2 ∈ s, ¬ R t1 t2) :=
begin
  sorry
end

end three_teams_not_competed_l35_35119


namespace find_k_l35_35706

theorem find_k (k : ℤ) : (0.0004040404 * 10 ^ k > 1000000) ∧ (0.0004040404 * 10 ^ k < 10000000) ↔ k = 11 :=
by
  sorry

end find_k_l35_35706


namespace value_of_f_two_l35_35258

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_two :
  (∀ x : ℝ, f (1 / x) = 1 / (x + 1)) → f 2 = 2 / 3 := by
  intro h
  -- The proof would go here
  sorry

end value_of_f_two_l35_35258


namespace platform_length_is_240_meters_l35_35558

-- Define the necessary variables and constants.
def train_length : ℝ := 120
def train_speed_kmph : ℝ := 60
def time_to_pass_platform : ℝ := 21.598272138228943

-- Conversion factor from kmph to m/s
def kmph_to_mps : ℝ := 1000 / 3600

-- The speed of the train in m/s
def train_speed_mps : ℝ := train_speed_kmph * kmph_to_mps

-- The total distance covered while passing the platform
def total_distance_covered : ℝ := train_speed_mps * time_to_pass_platform

-- Define the length of the platform
def platform_length : ℝ := total_distance_covered - train_length

-- Prove that the length of the platform is approximately 240 meters
theorem platform_length_is_240_meters : abs (platform_length - 240) < 1e-7 :=
by
  sorry  -- Proof not required

end platform_length_is_240_meters_l35_35558


namespace calculate_extra_fee_l35_35367

-- Conditions
def monthly_charge := 30
def promotional_rate := (1 / 3 : ℝ)
def first_month_charge := promotional_rate * monthly_charge
def num_months_without_extra_fee := 5
def normal_rate_charge := num_months_without_extra_fee * monthly_charge
def expected_charge := first_month_charge + normal_rate_charge
def total_paid := 175

-- Question and proof statement
theorem calculate_extra_fee : total_paid - expected_charge = 15 :=
by
  -- the proof is omitted (sorry)
  sorry

end calculate_extra_fee_l35_35367


namespace bonnie_roark_wire_length_ratio_l35_35935

noncomputable def ratio_of_wire_lengths : ℚ :=
let bonnie_wire_per_piece := 8
let bonnie_pieces := 12
let bonnie_total_wire := bonnie_pieces * bonnie_wire_per_piece

let bonnie_side := bonnie_wire_per_piece
let bonnie_volume := bonnie_side^3

let roark_side := 2
let roark_volume := roark_side^3
let roark_cubes := bonnie_volume / roark_volume

let roark_wire_per_piece := 2
let roark_pieces_per_cube := 12
let roark_wire_per_cube := roark_pieces_per_cube * roark_wire_per_piece
let roark_total_wire := roark_cubes * roark_wire_per_cube

let ratio := bonnie_total_wire / roark_total_wire
ratio 

theorem bonnie_roark_wire_length_ratio :
  ratio_of_wire_lengths = (1 : ℚ) / 16 := 
sorry

end bonnie_roark_wire_length_ratio_l35_35935


namespace largest_n_modulo_seven_l35_35095

theorem largest_n_modulo_seven (n : ℕ) (h₁ : n < 80000) (h₂ : (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 33) % 7 = 0) : n = 79993 :=
begin
  sorry
end

end largest_n_modulo_seven_l35_35095


namespace is_concyclic_bfhc_l35_35753

-- Let's define the problem in Lean 4 terms
variables {A E F G B C D H : Type} [inhabited A] [inhabited E] [inhabited F] [inhabited G] [inhabited B] [inhabited C] [inhabited D] [inhabited H]

-- Define the conditions given in the problem
def is_cyclic_quadrilateral (A E F G : Type) : Prop :=
∃ (circle : Type) (h1 : inscribed A circle) (h2 : inscribed E circle) (h3 : inscribed F circle) (h4 : inscribed G circle), true

def midpoint (D B C : Type) : Prop := (D = (B + C) / 2)

-- Given the conditions
variables (inscribed_quadrilateral : is_cyclic_quadrilateral A E F G)
variables (extension_ae_gf : intersection_point (line A E) (line G F) = B)
variables (extension_ef_ag : intersection_point (line E F) (line A G) = C)
variables (midpoint_d : midpoint D B C)
variables (intersection_ad : intersection_point (line A D) circle = H)

-- To prove
theorem is_concyclic_bfhc : is_cyclic_quadrilateral B F H C :=
sorry

end is_concyclic_bfhc_l35_35753


namespace book_stack_sum_l35_35553

theorem book_stack_sum : 
  let a := 15 -- first term
  let d := -2 -- common difference
  let l := 1 -- last term
  -- n = (l - a) / d + 1
  let n := (l - a) / d + 1
  -- S = n * (a + l) / 2
  let S := n * (a + l) / 2
  S = 64 :=
by
  -- The given conditions
  let a := 15 -- first term
  let d := -2 -- common difference
  let l := 1 -- last term
  -- Calculate the number of terms (n)
  let n := (l - a) / d + 1
  -- Calculate the total sum (S)
  let S := n * (a + l) / 2
  -- Prove the sum is 64
  show S = 64
  sorry

end book_stack_sum_l35_35553


namespace profit_per_box_type_A_and_B_maximize_profit_l35_35125

-- Condition definitions
def total_boxes : ℕ := 600
def profit_type_A : ℕ := 40000
def profit_type_B : ℕ := 160000
def profit_difference : ℕ := 200

-- Question 1: Proving the profit per box for type A and B
theorem profit_per_box_type_A_and_B (x : ℝ) :
  (profit_type_A / x + profit_type_B / (x + profit_difference) = total_boxes)
  → (x = 200) ∧ (x + profit_difference = 400) :=
sorry

-- Condition definitions for question 2
def price_reduction_per_box_A (a : ℕ) : ℕ := 5 * a
def price_increase_per_box_B (a : ℕ) : ℕ := 5 * a

-- Initial number of boxes sold for type A and B
def initial_boxes_sold_A : ℕ := 200
def initial_boxes_sold_B : ℕ := 400

-- General profit function
def profit (a : ℕ) : ℝ :=
  (initial_boxes_sold_A + 2 * a) * (200 - price_reduction_per_box_A a) +
  (initial_boxes_sold_B - 2 * a) * (400 + price_increase_per_box_B a)

-- Question 2: Proving the price reduction and maximum profit
theorem maximize_profit (a : ℕ) :
  ((price_reduction_per_box_A a = 75) ∧ (profit a = 204500)) :=
sorry

end profit_per_box_type_A_and_B_maximize_profit_l35_35125


namespace numbers_from_1_to_16_cannot_be_written_in_circle_but_row_l35_35789

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def adjacent_square_sums (l : List ℕ) : Prop :=
  ∀ (i : ℕ), i < l.length - 1 → is_perfect_square (l.nthLe i sorry + l.nthLe (i + 1) sorry)

def can_be_written_in_a_row (l : List ℕ) : Prop :=
  Set.of l.nodup ∧ adjacent_square_sums l

def can_be_written_in_a_circle (l : List ℕ) : Prop :=
  Set.of l.nodup ∧ adjacent_square_sums l ∧ is_perfect_square (l.head sorry + l.last sorry)

theorem numbers_from_1_to_16_cannot_be_written_in_circle_but_row (l : List ℕ) :
  can_be_written_in_a_row l ∧ ¬ can_be_written_in_a_circle l :=
sorry

end numbers_from_1_to_16_cannot_be_written_in_circle_but_row_l35_35789


namespace simplify_sqrt_expression_l35_35260

theorem simplify_sqrt_expression (x y : ℝ) (h : x * y < 0) : x * real.sqrt (- (y / (x^2))) = real.sqrt (-y) :=
sorry

end simplify_sqrt_expression_l35_35260


namespace sum_of_roots_cubic_l35_35099

theorem sum_of_roots_cubic :
  let a := 3
  let b := 7
  let c := -12
  let d := -4
  let roots_sum := -(b / a)
  roots_sum = -2.33 :=
by
  sorry

end sum_of_roots_cubic_l35_35099


namespace cranberries_left_l35_35170

theorem cranberries_left (total_cranberries : ℕ) (harvested_percent: ℝ) (cranberries_eaten : ℕ) 
  (h1 : total_cranberries = 60000) 
  (h2 : harvested_percent = 0.40) 
  (h3 : cranberries_eaten = 20000) : 
  total_cranberries - (harvested_percent * total_cranberries).to_nat - cranberries_eaten = 16000 := 
by 
  sorry

end cranberries_left_l35_35170


namespace unique_prime_triplets_l35_35613

theorem unique_prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) := 
by
  sorry

end unique_prime_triplets_l35_35613


namespace value_of_f_inv_sum_l35_35768

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f_inv (y : ℝ) : ℝ := sorry

axiom f_inv_is_inverse : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x
axiom f_condition : ∀ x : ℝ, f x + f (-x) = 2

theorem value_of_f_inv_sum (x : ℝ) : f_inv (2008 - x) + f_inv (x - 2006) = 0 :=
sorry

end value_of_f_inv_sum_l35_35768


namespace largest_last_digit_in_string_l35_35814

def is_multiple_of_17_or_23 (n : ℕ) := n % 17 = 0 ∨ n % 23 = 0

noncomputable def max_last_digit (s : String) : ℕ :=
if s.length = 2003 ∧ s.get 0 = '2'
   ∧ (∀ i, i < 2002 → is_multiple_of_17_or_23 (s.get i).toNat * 10 + (s.get (i + 1)).toNat)
then s.get (2003 - 1).toNat
else 0

theorem largest_last_digit_in_string :
  ∃ (s : String), s.length = 2003 ∧ s.get 0 = '2'
  ∧ (∀ i, i < 2002 → is_multiple_of_17_or_23 (s.get i).toNat * 10 + (s.get (i + 1)).toNat)
  ∧ max_last_digit s = 8 :=
sorry

end largest_last_digit_in_string_l35_35814


namespace pyramid_top_circle_multiple_of_4_l35_35899

-- Define the conditions for the pyramid of circles
def valid_initial_distributions (n : ℕ) : Finset (Fin n → ℕ) :=
  (Finset.range 2).pi (Finset.range n)

-- Define the requirement for the top circle to be a multiple of 4
def is_multiple_of_4 (f : Fin 12 → ℕ) : Prop :=
  (∃ (i j : Fin 12), i ≠ j ∧ f i = 2 ∧ f j = 2) ∨ (∀ i, f i = 0)

-- Define the proof of how many valid initial distributions lead to the top circle being a multiple of 4
theorem pyramid_top_circle_multiple_of_4 : (valid_initial_distributions 12).filter is_multiple_of_4).card = 4083 := 
sorry

end pyramid_top_circle_multiple_of_4_l35_35899


namespace final_position_independent_l35_35585

noncomputable
def transformation_terminal (k l k' l' : ℕ) (h1 : k ≥ 1) (h2 : l ≥ 1) (h3 : k' ≥ 1) (h4 : l' ≥ 1) 
  (T : ℕ × ℕ → ℕ × ℕ) : Prop :=
  ∀ a b : ℕ, a ≥ 1 ∧ b ≥ 1 → (T (a, b) = (b, a))

theorem final_position_independent (k l k' l' : ℕ) 
  (h1 : k ≥ 1) (h2 : l ≥ 1) (h3 : k' ≥ 1) (h4 : l' ≥ 1) 
  (T : ℕ × ℕ → ℕ × ℕ) (H : transformation_terminal k l k' l' h1 h2 h3 h4 T) :
  ∀ seq1 seq2 : list (ℕ × ℕ → ℕ × ℕ), 
    seq1.forall T → seq2.forall T → 
    (seq1.last = seq2.last) :=
sorry

end final_position_independent_l35_35585


namespace irrational_gap_gt_inv_4b2_l35_35796

noncomputable def rational_in_interval (a b : ℕ) : Prop := 
  a < b ∧ 0 < a ∧ a / b < 1

theorem irrational_gap_gt_inv_4b2 (a b : ℕ) (h_rational: rational_in_interval a b) : 
  abs ((a : ℝ) / (b : ℝ) - 1 / real.sqrt 2) > 1 / (4 * (b : ℝ) ^ 2) :=
sorry

end irrational_gap_gt_inv_4b2_l35_35796


namespace max_min_value_correct_l35_35343

noncomputable def max_min_value (a b : ℝ) (x y : ℂ) (hx : |x| = a) (hy : |y| = b) : ℝ :=
  let P := complex.abs ((x + y) / (1 + x * conj y)) in
  if (a^2 - 1) * (b^2 - 1) > 0 then (a + b) / (1 + a * b)
  else if (a^2 - 1) * (b^2 - 1) < 0 then |a - b| / |1 - a * b|
  else 1

theorem max_min_value_correct (a b : ℝ) (x y : ℂ) (hx : |x| = a) (hy : |y| = b) :
  max_min_value a b x y hx hy = 
  if (a^2 - 1) * (b^2 - 1) > 0 then (a + b) / (1 + a * b)
  else if (a^2 - 1) * (b^2 - 1) < 0 then |a - b| / |1 - (a * b)|
  else 1 :=
sorry

end max_min_value_correct_l35_35343


namespace total_number_of_bills_received_l35_35070

open Nat

-- Definitions based on the conditions:
def total_withdrawal_amount : ℕ := 600
def bill_denomination : ℕ := 20

-- Mathematically equivalent proof problem
theorem total_number_of_bills_received : (total_withdrawal_amount / bill_denomination) = 30 := 
by
  sorry

end total_number_of_bills_received_l35_35070


namespace mac_expected_loss_l35_35366

noncomputable def value_of_trade_1 : ℝ := 4 * 0.10 + 2 * 0.01 - 0.25
noncomputable def expected_loss_trade_1 : ℝ := 20 * value_of_trade_1 * 0.05

noncomputable def value_of_trade_2 : ℝ := 9 * 0.05 + 0.01 - 0.25
noncomputable def expected_loss_trade_2 : ℝ := 20 * value_of_trade_2 * 0.10

noncomputable def value_of_trade_3 : ℝ := 0.50 + 0.03 - 0.25
noncomputable def expected_loss_trade_3 : ℝ := 20 * value_of_trade_3 * 0.85

noncomputable def total_expected_loss : ℝ :=
  expected_loss_trade_1 + expected_loss_trade_2 + expected_loss_trade_3

theorem mac_expected_loss : total_expected_loss = 5.35 :=
by
  unfold total_expected_loss
  unfold expected_loss_trade_1 expected_loss_trade_2 expected_loss_trade_3
  unfold value_of_trade_1 value_of_trade_2 value_of_trade_3
  -- Detailed computation steps can be added if needed
  have h1 : value_of_trade_1 = 0.17 := by norm_num
  have h2 : expected_loss_trade_1 = 0.17 := by norm_num
  have h3 : value_of_trade_2 = 0.21 := by norm_num
  have h4 : expected_loss_trade_2 = 0.42 := by norm_num
  have h5 : value_of_trade_3 = 0.28 := by norm_num
  have h6 : expected_loss_trade_3 = 4.76 := by norm_num
  have h7 : total_expected_loss = 5.35 := by norm_num
  exact h7

end mac_expected_loss_l35_35366


namespace vivian_songs_each_day_l35_35853

variable (V C : ℕ)
variable (songs_vivian songs_clara : ℕ)
variable (days_in_month weekend_days total_songs weekdays_played : ℕ)

noncomputable def songs_each_day : ℕ := V

axiom clara_plays_fewer_songs : C = V - 2
axiom weekend_days_count : weekend_days = 8
axiom total_songs_in_month : total_songs = 396
axiom days_in_a_month : days_in_month = 30

def weekdays_played_days : ℕ := days_in_month - weekend_days

axiom total_songs_equation : weekdays_played_days * V + weekdays_played_days * (C) = total_songs

theorem vivian_songs_each_day : songs_each_day = 10 :=
by {
  have clara_songs := clara_plays_fewer_songs,
  have weekdays := weekdays_played_days,
  have total_songs_formula := total_songs_equation,
  calc
    songs_each_day
    = V : by rfl
    ... = 10 : by {
      -- Solve the mathematical equations
      sorry
    }
}

end vivian_songs_each_day_l35_35853


namespace complex_pure_imaginary_l35_35707

theorem complex_pure_imaginary (a : ℝ) : (¬ ∃ Re, Im, Re = 0 ∧ (∃ Re, Im, (2 - a - 2 * Im + a) / 2 ≠ 0) ∧ 
                                           ((2 - a) / 2 = 0) ∧ ¬ (-(2 + a) / 2 = 0)) → a = 2 :=
by
  sorry

end complex_pure_imaginary_l35_35707


namespace tenth_term_is_four_l35_35975

noncomputable def a : ℕ → ℝ
| 0     := 3
| 1     := 4
| (n + 1) := 12 / a n

theorem tenth_term_is_four : a 9 = 4 :=
by
  sorry

end tenth_term_is_four_l35_35975


namespace sum_of_all_values_of_z_l35_35758

noncomputable def f (x : ℚ) : ℚ := x^2 - x + 2

theorem sum_of_all_values_of_z :
  (∑ z in {z : ℚ | f(4 * z) = 8}, z) = 1 / 16 :=
sorry

end sum_of_all_values_of_z_l35_35758


namespace max_points_on_circle_at_fixed_distance_l35_35378

open Real EuclideanGeometry

variable {P : Type} [MetricSpace P] [NormedAddCommGroup P] [NormedSpace ℝ P]

def Point := P
def Radius := ℝ
def Distance := λ (p1 p2 : Point), dist p1 p2

noncomputable def circle (center : Point) (r : Radius) := {p : Point | Distance p center = r}

theorem max_points_on_circle_at_fixed_distance
  (Q D_center : Point) (r_d : Radius) (r_q : Radius)
  (h_outside : Distance Q D_center > r_d) :
  ∃ (p1 p2 : Point), p1 ≠ p2 ∧
  p1 ∈ circle D_center r_d ∧ Distance p1 Q = r_q ∧
  p2 ∈ circle D_center r_d ∧ Distance p2 Q = r_q ∧
  ∀ p ∈ circle D_center r_d, Distance p Q = r_q → (p = p1 ∨ p = p2) :=
  sorry

end max_points_on_circle_at_fixed_distance_l35_35378


namespace least_non_lucky_multiple_of_11_l35_35136

/--
A lucky integer is a positive integer which is divisible by the sum of its digits.
Example:
- 18 is a lucky integer because 1 + 8 = 9 and 18 is divisible by 9.
- 20 is not a lucky integer because 2 + 0 = 2 and 20 is not divisible by 2.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_non_lucky_multiple_of_11 : ∃ n, n > 0 ∧ n % 11 = 0 ∧ ¬ is_lucky n ∧ ∀ m, m > 0 → m % 11 = 0 → ¬ is_lucky m → n ≤ m := 
by
  sorry

end least_non_lucky_multiple_of_11_l35_35136


namespace arithmetic_sequence_ratio_l35_35761

noncomputable def A_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def B_n (b e : ℤ) (n : ℕ) : ℤ :=
  n * (2 * b + (n - 1) * e) / 2

theorem arithmetic_sequence_ratio (a d b e : ℤ) :
  (∀ n : ℕ, n ≠ 0 → A_n a d n / B_n b e n = (5 * n - 3) / (n + 9)) →
  (a + 5 * d) / (b + 2 * e) = 26 / 7 :=
by
  sorry

end arithmetic_sequence_ratio_l35_35761


namespace microwave_price_reduction_l35_35139

theorem microwave_price_reduction (M : ℝ) (h : M > 0) : 
  let discount1 := 0.3 
  let discount2 := 0.4 
  let reduced_price1 := M * (1 - discount1)
  let reduced_price2 := reduced_price1 * (1 - discount2)
  let final_reduction := M - reduced_price2
  (final_reduction / M * 100) = 58 :=
by
  let discount1 := 0.3 
  let discount2 := 0.4 
  let reduced_price1 := M * (1 - discount1)
  let reduced_price2 := reduced_price1 * (1 - discount2)
  let final_reduction := M - reduced_price2
  sorry

end microwave_price_reduction_l35_35139


namespace exists_difference_divisible_by_11_l35_35642

theorem exists_difference_divisible_by_11 (a : Fin 12 → ℤ) :
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
  sorry

end exists_difference_divisible_by_11_l35_35642


namespace youngest_person_age_l35_35058

theorem youngest_person_age (total_age_now : ℕ) (total_age_when_born : ℕ) (Y : ℕ) (h1 : total_age_now = 210) (h2 : total_age_when_born = 162) : Y = 48 :=
by
  sorry

end youngest_person_age_l35_35058


namespace cylinder_volume_increase_l35_35036

theorem cylinder_volume_increase (r h : ℝ) :
  let initial_volume := π * r^2 * h in
  let new_volume := π * (2 * r)^2 * (2 * h) in
  new_volume = 8 * initial_volume :=
by
  sorry

end cylinder_volume_increase_l35_35036


namespace solve_g_eq_3_l35_35763

def g (x : ℝ) : ℝ :=
if x < 0 then 3*x + 6 else 2*x - 13

theorem solve_g_eq_3 : {x : ℝ | g(x) = 3} = {-1, 8} :=
  sorry

end solve_g_eq_3_l35_35763


namespace calculate_cells_after_12_days_l35_35162

theorem calculate_cells_after_12_days :
  let initial_cells := 5
  let division_factor := 3
  let days := 12
  let period := 3
  let n := days / period
  initial_cells * division_factor ^ (n - 1) = 135 := by
  sorry

end calculate_cells_after_12_days_l35_35162


namespace quadratic_has_distinct_real_roots_l35_35834

def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

theorem quadratic_has_distinct_real_roots :
  let a := 5
  let b := -2
  let c := -7
  discriminant a b c > 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l35_35834


namespace affine_transformation_circle_rotation_or_reflection_l35_35381

theorem affine_transformation_circle_rotation_or_reflection (L : Affine.Transformation ℝ) (circle : set ℝ) :
  (∀ p ∈ circle, L p ∈ circle) → (L.is_rotation ∨ L.is_reflection) :=
by {
  -- Proof of the theorem will be provided here
  sorry
}

end affine_transformation_circle_rotation_or_reflection_l35_35381


namespace parallelogram_area_l35_35452

theorem parallelogram_area (b h : ℕ) (hb : b = 20) (hh : h = 4) : b * h = 80 :=
by
  rw [hb, hh]
  exact rfl

end parallelogram_area_l35_35452


namespace recurring_decimals_sum_correct_l35_35603

noncomputable def recurring_decimals_sum : ℚ :=
  let x := (2:ℚ) / 3
  let y := (4:ℚ) / 9
  x + y

theorem recurring_decimals_sum_correct :
  recurring_decimals_sum = 10 / 9 := 
  sorry

end recurring_decimals_sum_correct_l35_35603


namespace common_chords_intersect_at_single_point_l35_35289

theorem common_chords_intersect_at_single_point
  (A B C: Point)
  (O1 O2 O3: Circle)
  (h1: O1 ≠ O2)
  (h2: O2 ≠ O3)
  (h3: O1 ≠ O3)
  (hA1 : A_1 ∈ (inter O2 O3))
  (hB1 : B_1 ∈ (inter O1 O3))
  (hC1 : C_1 ∈ (inter O1 O2)) :
  ∃ P: Point, is_common_point O1 A1 C1 ∧ is_common_point O2 B1 C1 ∧ is_common_point O3 A1 B1 := 
sorry

end common_chords_intersect_at_single_point_l35_35289


namespace smallest_two_digit_product_12_l35_35467

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l35_35467


namespace compare_neg_sqrt_l35_35581

theorem compare_neg_sqrt : -real.sqrt 10 < -3 :=
by
  sorry

end compare_neg_sqrt_l35_35581


namespace chicken_price_per_pound_l35_35335

theorem chicken_price_per_pound (beef_pounds chicken_pounds : ℕ) (beef_price chicken_price : ℕ)
    (total_amount : ℕ)
    (h_beef_quantity : beef_pounds = 1000)
    (h_beef_cost : beef_price = 8)
    (h_chicken_quantity : chicken_pounds = 2 * beef_pounds)
    (h_total_price : 1000 * beef_price + chicken_pounds * chicken_price = total_amount)
    (h_total_amount : total_amount = 14000) : chicken_price = 3 :=
by
  sorry

end chicken_price_per_pound_l35_35335


namespace sled_dog_race_l35_35153

theorem sled_dog_race (d t : ℕ) (h1 : d + t = 315) (h2 : (1.2 : ℚ) * d + t = (1 / 2 : ℚ) * (2 * d + 3 * t)) :
  d = 225 ∧ t = 90 :=
sorry

end sled_dog_race_l35_35153


namespace foci_distance_l35_35984

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 - 32 * y = 144

-- State the problem to prove
theorem foci_distance :
  ∀ x y : ℝ, hyperbola_eq x y → distance_between_foci (9 * x^2 - 18 * x - 16 * y^2 - 32 * y = 144) = (sqrt 3425) / 72 :=
by
  sorry

end foci_distance_l35_35984


namespace vet_appointments_l35_35735

theorem vet_appointments (n : ℕ) (cost_per_appointment : ℕ := 400) (insurance_cost : ℕ := 100) 
  (insurance_coverage : ℕ := 80) (total_cost : ℕ := 660) :
  (400 + 100 + 80 * (n - 1) = 660) → n = 3 :=
by {
  intros h,
  sorry
}

end vet_appointments_l35_35735


namespace prime_ge_7_p2_sub1_div_by_30_l35_35199

theorem prime_ge_7_p2_sub1_div_by_30 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 30 ∣ (p^2 - 1) :=
sorry

end prime_ge_7_p2_sub1_div_by_30_l35_35199


namespace divide_students_into_teams_l35_35869

theorem divide_students_into_teams (n_A n_B n_C n_D : ℕ) (h : n_A + n_B + n_C + n_D = 300) :
  ∃ teams : list (list ℕ), teams.length = 100 ∧
    (∀ team ∈ teams, team.length = 3 ∧ 
      (∀ x ∈ team, x = team.head ∨
       (team.head ≠ team[1] ∧ team[1] ≠ team[2] ∧ team[2] ≠ team.head))) :=
sorry

end divide_students_into_teams_l35_35869


namespace largest_two_digit_prime_factor_of_binom_300_150_l35_35084

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

def is_prime (p : ℕ) : Prop :=
  nat.prime p

def is_two_digit (p : ℕ) : Prop :=
  10 ≤ p ∧ p < 100

def less_than_300_divided_by_three (p : ℕ) : Prop :=
  3 * p < 300

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  let n := binomial_coefficient 300 150 in
  ∃ p, is_prime p ∧ is_two_digit p ∧ less_than_300_divided_by_three p ∧
        ∀ q, is_prime q ∧ is_two_digit q ∧ less_than_300_divided_by_three q → q ≤ p ∧ n % p = 0 ∧ p = 97 :=
by
  sorry

end largest_two_digit_prime_factor_of_binom_300_150_l35_35084


namespace exponential_function_solution_l35_35303

theorem exponential_function_solution (a : ℝ) (h₁ : ∀ x : ℝ, a ^ x > 0) :
  (∃ y : ℝ, y = a ^ 2 ∧ y = 4) → a = 2 :=
by
  sorry

end exponential_function_solution_l35_35303


namespace log_base_7_cube_root_7_l35_35210

theorem log_base_7_cube_root_7 : log 7 (7 ^ (1 / 3)) = 1 / 3 := 
by sorry

end log_base_7_cube_root_7_l35_35210


namespace polygon_is_quadrilateral_l35_35140

-- Problem statement in Lean 4
theorem polygon_is_quadrilateral 
  (n : ℕ) 
  (h₁ : (n - 2) * 180 = 360) :
  n = 4 :=
by
  sorry

end polygon_is_quadrilateral_l35_35140


namespace find_S2_l35_35685

-- Define the spheres and their surface areas.
variables {R1 R2 R3 : ℝ} {S1 S2 S3 : ℝ}

-- Assume given conditions.
def sphere_surface_area_relations :=
  (R1 + R3 = 2 * R2) ∧ 
  (S1 = 4 * Real.pi * R1^2) ∧ 
  (S3 = 4 * Real.pi * R3^2) ∧ 
  (S1 = 1) ∧ 
  (S3 = 9)

-- The goal is to prove S2 = 4 given the conditions.
theorem find_S2 : sphere_surface_area_relations → S2 = 4 := 
  by
  sorry

end find_S2_l35_35685


namespace largest_prime_factor_of_binomial_l35_35087

theorem largest_prime_factor_of_binomial :
  ∃ p : ℕ, p.prime ∧ 10 ≤ p ∧ p < 100 ∧ (∃ k : ℕ, p ^ k ∣ (300.factorial / (150.factorial * 150.factorial))) ∧
  (∀ q : ℕ, q.prime → 10 ≤ q ∧ q < 100 → (∃ k : ℕ, q ^ k ∣ (300.factorial / (150.factorial * 150.factorial))) → q ≤ p) :=
sorry

end largest_prime_factor_of_binomial_l35_35087


namespace Angle_AXC_is_106_Angle_ACB_is_48_l35_35049

@[simp]
theorem Angle_AXC_is_106 (A B C D X : Point) 
  (h1 : AD = DC) 
  (h2 : AB = BX) 
  (h3 : ∠B = 32) 
  (h4 : ∠XDC = 52) 
  : ∠AXC = 106 := 
by
  sorry

@[simp]
theorem Angle_ACB_is_48 (A B C D X : Point) 
  (h1 : AD = DC) 
  (h2 : AB = BX) 
  (h3 : ∠B = 32) 
  (h4 : ∠XDC = 52) 
  : ∠ACB = 48 := 
by
  sorry

end Angle_AXC_is_106_Angle_ACB_is_48_l35_35049


namespace find_natural_n_l35_35610

theorem find_natural_n (n : ℕ) :
  (992768 ≤ n ∧ n ≤ 993791) ↔ 
  (∀ k : ℕ, k > 0 → k^2 + (n / k^2) = 1991) := sorry

end find_natural_n_l35_35610


namespace car_traveled_miles_per_gallon_city_l35_35503

noncomputable def miles_per_gallon_city (H C G : ℝ) : Prop :=
  (C = H - 18) ∧ (462 = H * G) ∧ (336 = C * G)

theorem car_traveled_miles_per_gallon_city :
  ∃ H G, miles_per_gallon_city H 48 G :=
by
  sorry

end car_traveled_miles_per_gallon_city_l35_35503


namespace set_representation_l35_35214

def is_nat_star (n : ℕ) : Prop := n > 0
def satisfies_eqn (x y : ℕ) : Prop := y = 6 / (x + 3)

theorem set_representation :
  {p : ℕ × ℕ | is_nat_star p.fst ∧ is_nat_star p.snd ∧ satisfies_eqn p.fst p.snd } = { (3, 1) } :=
by
  sorry

end set_representation_l35_35214


namespace area_transformed_region_l35_35347

theorem area_transformed_region (area_R : ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ)
  (h_area_R : area_R = 15) (h_M : M = !![3, 2; 4, -5]) : 
  let area_R' := Real.abs (Matrix.det M) * area_R in area_R' = 345 := by
  sorry

end area_transformed_region_l35_35347


namespace find_k_l35_35267

theorem find_k (x : ℝ) (k : ℝ) (h : 2 * x - 3 = 3 * x - 2 + k) (h_solution : x = 2) : k = -3 := by
  sorry

end find_k_l35_35267


namespace number_of_non_degenerate_rectangles_excluding_center_l35_35959

/-!
# Problem Statement
We want to find the number of non-degenerate rectangles in a 7x7 grid that do not fully cover the center point (4, 4).
-/

def num_rectangles_excluding_center : Nat :=
  let total_rectangles := (Nat.choose 7 2) * (Nat.choose 7 2)
  let rectangles_including_center := 4 * ((3 * 3 * 3) + (3 * 3))
  total_rectangles - rectangles_including_center

theorem number_of_non_degenerate_rectangles_excluding_center :
  num_rectangles_excluding_center = 297 :=
by
  sorry -- proof goes here

end number_of_non_degenerate_rectangles_excluding_center_l35_35959


namespace cranberries_left_l35_35171

def initial_cranberries : ℕ := 60000
def harvested_by_humans : ℕ := initial_cranberries * 40 / 100
def eaten_by_elk : ℕ := 20000

theorem cranberries_left (c : ℕ) : c = initial_cranberries - harvested_by_humans - eaten_by_elk → c = 16000 := by
  sorry

end cranberries_left_l35_35171


namespace singers_in_fifth_verse_l35_35530

theorem singers_in_fifth_verse (choir : ℕ) (absent : ℕ) (participating : ℕ) 
(half_first_verse : ℕ) (third_second_verse : ℕ) (quarter_third_verse : ℕ) 
(fifth_fourth_verse : ℕ) (late_singers : ℕ) :
  choir = 70 → 
  absent = 10 → 
  participating = choir - absent →
  half_first_verse = participating / 2 → 
  third_second_verse = (participating - half_first_verse) / 3 →
  quarter_third_verse = (participating - half_first_verse - third_second_verse) / 4 →
  fifth_fourth_verse = (participating - half_first_verse - third_second_verse - quarter_third_verse) / 5 →
  late_singers = 5 →
  participating = 60 :=
by sorry

end singers_in_fifth_verse_l35_35530


namespace tile_probability_l35_35432

theorem tile_probability :
  let tiles_A := {n : ℕ | 1 ≤ n ∧ n ≤ 30}
  let tiles_B := {n : ℕ | 10 ≤ n ∧ n ≤ 49}
  let favorable_A := {n : ℕ | 1 ≤ n ∧ n < 20}
  let favorable_B := {n : ℕ | (10 ≤ n ∧ n ≤ 49 ∧ (n % 2 = 1 ∨ 40 < n))}
  let p_A := (favorable_A.card : ℚ) / tiles_A.card
  let p_B := (favorable_B.card : ℚ) / tiles_B.card
  let combined_probability := p_A * p_B
  combined_probability = 19 / 50 :=
by
  sorry

end tile_probability_l35_35432


namespace first_term_of_arithmetic_sequence_l35_35905

theorem first_term_of_arithmetic_sequence (a : ℕ) (median last_term : ℕ) 
  (h_arithmetic_progression : true) (h_median : median = 1010) (h_last_term : last_term = 2015) :
  a = 5 :=
by
  have h1 : 2 * median = 2020 := by sorry
  have h2 : last_term + a = 2020 := by sorry
  have h3 : 2015 + a = 2020 := by sorry
  have h4 : a = 2020 - 2015 := by sorry
  have h5 : a = 5 := by sorry
  exact h5

end first_term_of_arithmetic_sequence_l35_35905


namespace jasmine_spent_l35_35331

theorem jasmine_spent 
  (original_cost : ℝ)
  (discount : ℝ)
  (h_original : original_cost = 35)
  (h_discount : discount = 17) : 
  original_cost - discount = 18 := 
by
  sorry

end jasmine_spent_l35_35331


namespace largest_2_digit_prime_factor_of_binom_l35_35078

open Nat

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

/-- Definition of the problem conditions -/
def problem_conditions : Prop :=
  let n := binom 300 150
  ∃ p : ℕ, Prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p ≤ 75 ∨ 3 * p < 300) ∧ p = 97

/-- Statement of the proof problem -/
theorem largest_2_digit_prime_factor_of_binom : problem_conditions := 
  sorry

end largest_2_digit_prime_factor_of_binom_l35_35078


namespace bethany_total_time_spent_l35_35183

def total_time_spent_on_activities_over_two_weeks
  (horse_riding_hours_mon_wed_fri : ℕ := 1)
  (horse_riding_hours_tue_thu : ℕ := 30)  -- in minutes
  (horse_riding_hours_sat : ℕ := 2)
  (horse_riding_hours_sun : ℕ := 0)
  (swimming_hours_mon : ℕ := 45)  -- in minutes
  (swimming_hours_wed : ℕ := 60)  -- in minutes
  (swimming_hours_fri : ℕ := 30)  -- in minutes
  (piano_hours_tue : ℕ := 30)  -- in minutes
  (piano_hours_thu : ℕ := 60)  -- in minutes
  (piano_hours_sun : ℕ := 90)  -- in minutes
  : ℕ :=
  let total_horse_riding_hours_per_week := horse_riding_hours_mon_wed_fri * 3 +
                                           horse_riding_hours_tue_thu * 2 / 60 +
                                           horse_riding_hours_sat +
                                           horse_riding_hours_sun
  let total_horse_riding_hours := total_horse_riding_hours_per_week * 2
  let total_swimming_hours_per_week := (swimming_hours_mon + swimming_hours_wed + swimming_hours_fri) / 60
  let total_swimming_hours := total_swimming_hours_per_week * 2
  let total_piano_hours_per_week := (piano_hours_tue + piano_hours_thu + piano_hours_sun) / 60
  let total_piano_hours := total_piano_hours_per_week * 2
  total_horse_riding_hours + total_swimming_hours + total_piano_hours

theorem bethany_total_time_spent :
  total_time_spent_on_activities_over_two_weeks = 22.5 :=
by
  sorry

end bethany_total_time_spent_l35_35183


namespace smallest_two_digit_l35_35466

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l35_35466


namespace max_rhombic_tiles_l35_35833

theorem max_rhombic_tiles (n : ℕ) : 
  let small_triangles := n * (n + 1) / 2
  let max_tiles := small_triangles / 2
  in max_tiles = (n^2 - n) / 2 := 
by sorry

end max_rhombic_tiles_l35_35833


namespace proof_problem_l35_35245

-- Definitions related to the problem
variables {A B M C D E F N K : Point}

-- We will assume the existence of the segments and squares mentioned in the problem
def is_square (A B C D : Point) : Prop := 
  ∃ l : ℝ, dist A B = l ∧ dist B C = l ∧ dist C D = l ∧ dist D A = l ∧ 
           dist A C = l * sqrt 2 ∧ dist B D = l * sqrt 2 ∧ 
           (∃ O : Point, is_center O A B C D)

-- Circumscribed circles
def is_circumscribed (A B C D : Point) (N : Point) : Prop :=
  is_square A B C D ∧ N ≠ M ∧ point_on_circle N A B C D

-- Intersect condition
def lines_intersect_at (P Q R : Point) : Prop := 
  incident P Q R

-- Fixed point condition
def passes_through_fixed_point (P Q : Point) : Prop :=
  ∃ fixed : Point, incident P fixed Q

-- Geometric locus condition
def geometric_locus_midline (A B K : Point) : Prop := 
  let L := symmetric_point_over_line A B K in
  midline_geometry A B L

-- Main theorem combining all three parts
theorem proof_problem 
    (h_sq1 : is_square A M C D) 
    (h_sq2 : is_square M B E F) 
    (h_circ1 : is_circumscribed A M C D N)
    (h_circ2 : is_circumscribed M B E F N) :
    (lines_intersect_at A F N ∧ lines_intersect_at B C N) ∧ 
    passes_through_fixed_point M N ∧ 
    geometric_locus_midline A B K :=
sorry

end proof_problem_l35_35245


namespace tenth_term_of_sequence_l35_35967

noncomputable def sequence : ℕ → ℚ
| 0       := 3
| 1       := 4
| (n + 2) := 12 / (sequence n)

theorem tenth_term_of_sequence : sequence 9 = 4 :=
by
  sorry

end tenth_term_of_sequence_l35_35967


namespace hydrogen_burns_oxygen_certain_l35_35875

-- define what it means for a chemical reaction to be well-documented and known to occur
def chemical_reaction (reactants : String) (products : String) : Prop :=
  (reactants = "2H₂ + O₂") ∧ (products = "2H₂O")

-- Event description and classification
def event_is_certain (event : String) : Prop :=
  event = "Hydrogen burns in oxygen to form water"

-- Main statement
theorem hydrogen_burns_oxygen_certain :
  ∀ (reactants products : String), (chemical_reaction reactants products) → event_is_certain "Hydrogen burns in oxygen to form water" :=
by
  intros reactants products h
  have h1 : reactants = "2H₂ + O₂" := h.1
  have h2 : products = "2H₂O" := h.2
  -- proof omitted
  exact sorry

end hydrogen_burns_oxygen_certain_l35_35875


namespace length_BD_eq_sqrt10_l35_35741

noncomputable def AB : ℝ := 10
noncomputable def AC : ℝ := 8
noncomputable def D : ℝ := 9

theorem length_BD_eq_sqrt10 (AB AC D : ℝ) (h1 : AB = 10) (h2 : AC = 8) (h3 : D = 9) : sqrt ((AB - D)^2 + (0 + 3)^2) = sqrt 10 := 
by sorry

end length_BD_eq_sqrt10_l35_35741


namespace largest_two_digit_prime_factor_of_binom_300_150_l35_35083

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

def is_prime (p : ℕ) : Prop :=
  nat.prime p

def is_two_digit (p : ℕ) : Prop :=
  10 ≤ p ∧ p < 100

def less_than_300_divided_by_three (p : ℕ) : Prop :=
  3 * p < 300

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  let n := binomial_coefficient 300 150 in
  ∃ p, is_prime p ∧ is_two_digit p ∧ less_than_300_divided_by_three p ∧
        ∀ q, is_prime q ∧ is_two_digit q ∧ less_than_300_divided_by_three q → q ≤ p ∧ n % p = 0 ∧ p = 97 :=
by
  sorry

end largest_two_digit_prime_factor_of_binom_300_150_l35_35083


namespace count_of_numbers_coprime_to_2_and_3_not_prime_l35_35159

def is_coprime(a b : Nat) : Prop := Nat.gcd a b = 1

def is_not_prime_and_coprime_to_2_and_3 (n : Nat) : Prop :=
  n ≤ 200 ∧ is_coprime n 2 ∧ is_coprime n 3 ∧ ¬ Nat.prime n

theorem count_of_numbers_coprime_to_2_and_3_not_prime :
  Finset.card (Finset.filter is_not_prime_and_coprime_to_2_and_3 (Finset.range 201)) = 23 :=
by
  sorry

end count_of_numbers_coprime_to_2_and_3_not_prime_l35_35159


namespace julia_error_approx_97_percent_l35_35731

theorem julia_error_approx_97_percent (x : ℝ) : 
  abs ((6 * x - x / 6) / (6 * x) * 100 - 97) < 1 :=
by 
  sorry

end julia_error_approx_97_percent_l35_35731


namespace proof_problem_l35_35631

def sigma_except_self (n : ℕ) : ℕ :=
  Nat.divisors (n) |>.filter (fun d => d ≠ n) |>.sum

theorem proof_problem : sigma_except_self (sigma_except_self 10) = 7 := by
  sorry

end proof_problem_l35_35631


namespace sum_n_x_y_l35_35316

def D := (0, 0)
def E := (0, 15)
def F := (20, 0)
def D' := (30, 20)
def E' := (45, 20)
def F' := (30, 0)

def rotation (θ : ℝ) (p q : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  let (qx, qy) := q
  let θ_rad := θ * real.pi / 180
  (
    qx + (px - qx) * real.cos θ_rad - (py - qy) * real.sin θ_rad,
    qy + (px - qx) * real.sin θ_rad + (py - qy) * real.cos θ_rad
  )

theorem sum_n_x_y : ∃ (x y n : ℝ), 
  0 < n ∧ n < 180 ∧
  rotation (-n) D (x, y) = D' ∧
  rotation (-n) E (x, y) = E' ∧
  rotation (-n) F (x, y) = F' ∧
  n + x + y = 40 := by
  sorry

end sum_n_x_y_l35_35316


namespace Teresa_age_at_Michiko_birth_l35_35804

-- Definitions of the conditions
def Teresa_age_now : ℕ := 59
def Morio_age_now : ℕ := 71
def Morio_age_at_Michiko_birth : ℕ := 38

-- Prove that Teresa was 26 years old when she gave birth to Michiko.
theorem Teresa_age_at_Michiko_birth : 38 - (71 - 59) = 26 := by
  -- Provide the proof here
  sorry

end Teresa_age_at_Michiko_birth_l35_35804


namespace min_L_shaped_trominos_l35_35096

-- Definitions based on the conditions
def L_shaped_tromino (grid : Matrix ℕ ℕ ℕ 6 6) : Prop :=
  ∀ (r c : ℕ), 0 ≤ r ∧ r ≤ 5 → 0 ≤ c ∧ c ≤ 5 →
  (grid[r, c] = 1 ∧ grid[r, c + 1] = 1 ∧ grid[r + 1, c] = 1) ∨
  (grid[r, c] = 1 ∧ grid[r + 1, c] = 1 ∧ grid[r + 1, c + 1] = 1) ∨
  (grid[r, c] = 1 ∧ grid[r + 1, c] = 1 ∧ grid[r + 1, c - 1] = 1) ∨
  (grid[r, c] = 1 ∧ grid[r + 1, c] = 1 ∧ grid[r + 1, c + 1] = 1)

-- The question translated into a Lean theorem requiring proof
theorem min_L_shaped_trominos :
  ∀ (grid : Matrix ℕ ℕ ℕ 6 6), (∀ r c, r < 6 → c < 6 → grid[r, c] ∈ {0, 1}) → 
  (∀ r c, grid[r, c] = 1 → L_shaped_tromino grid) →
  (∀ r c, L_shaped_tromino grid → grid[r, c] ≠ L_shaped_tromino grid) →
  6 := sorry

end min_L_shaped_trominos_l35_35096


namespace problem_statement_l35_35239

noncomputable def f : ℝ → ℝ
| x := if x ≥ 6 then x - 4 else f (x + 3)

theorem problem_statement : f 2 = 4 :=
by {
  sorry
}

end problem_statement_l35_35239


namespace part1_part2_l35_35247

noncomputable def S : ℕ+ → ℤ := λ n, 2 * a n - 3 * n.val

theorem part1 (a : ℕ+ → ℤ) (S : ℕ+ → ℤ) (h : ∀ n, S n = 2 * a n - 3 * n.val) :
  a 1 = 3 ∧ a 2 = 9 ∧ a 3 = 21 :=
sorry

theorem part2 (a : ℕ+ → ℤ) (S : ℕ+ → ℤ) 
  (h : ∀ n, S n = 2 * a n - 3 * n.val) :
  ∃ (b : ℕ+ → ℤ), (∀ m n, b (m + n) = b m * b n) ∧
  b 1 = 6 ∧ 
  (∀ n, a n = 3 * (2^n.val - 1)) :=
sorry

end part1_part2_l35_35247


namespace bob_needs_additional_weeks_l35_35933

-- Definitions based on conditions
def weekly_prize : ℕ := 100
def initial_weeks_won : ℕ := 2
def total_prize_won : ℕ := initial_weeks_won * weekly_prize
def puppy_cost : ℕ := 1000
def additional_weeks_needed : ℕ := (puppy_cost - total_prize_won) / weekly_prize

-- Statement of the theorem
theorem bob_needs_additional_weeks : additional_weeks_needed = 8 := by
  -- Proof here
  sorry

end bob_needs_additional_weeks_l35_35933


namespace nagel_point_on_line_mi_l35_35108

/-- Problem statement -/
theorem nagel_point_on_line_mi (ABC : Triangle) (N M I : Point) :
  is_nagel_point ABC N ∧ is_centroid ABC M ∧ is_incenter ABC I → 
  vector_between_points N M = 2 • vector_between_points M I := 
sorry

end nagel_point_on_line_mi_l35_35108


namespace value_of_f_at_31_over_2_l35_35536

noncomputable def f (x: ℝ) : ℝ := 
  if x ∈ set.Icc 0 1 then 
    x * (3 - 2 * x)
  else if x + 1 ∈ set.Icc 0 1 then
    (x + 1) * (3 - 2 * (x + 1))
  else 
    0 -- This is a placeholder, you would need to define f for all x based on the periodicity etc.

theorem value_of_f_at_31_over_2 :
  (∀ x: ℝ, f(-x) = -f(x)) →
  (∀ x: ℝ, f(x+1) = f(-x-1)) →
  f(31/2) = -1 :=
by
  intros odd even
  sorry

end value_of_f_at_31_over_2_l35_35536


namespace area_of_triangle_PQR_l35_35065

theorem area_of_triangle_PQR (P Q R M N G : Point) 
    (hPM : is_median P M R)
    (hQN : is_median Q N P) 
    (hPM_length : length P M = 15)
    (hQN_length : length Q N = 9)
    (h_perpendicular : ∠PMN = 90°):
    area PQR = 90 :=
sorry

end area_of_triangle_PQR_l35_35065


namespace unique_prime_triplets_l35_35614

theorem unique_prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) := 
by
  sorry

end unique_prime_triplets_l35_35614


namespace simplify_frac_l35_35187

variable (m : ℝ)

theorem simplify_frac : m^2 ≠ 9 → (3 / (m^2 - 9) + m / (9 - m^2)) = - (1 / (m + 3)) :=
by
  intro h
  sorry

end simplify_frac_l35_35187


namespace abc_sum_is_22_l35_35346

noncomputable def polynomial_roots (a b c : ℝ) (w : ℂ) : Prop :=
  let r1 := w - complex.I
  let r2 := w - 3 * complex.I
  let r3 := 2 * w + 2 in
  (r1 + r2 + r3) = -a ∧
  (r1 * r2 + r2 * r3 + r3 * r1) = b ∧
  (r1 * r2 * r3) = -c

theorem abc_sum_is_22 (a b c : ℝ) (w : ℂ) (h : polynomial_roots a b c w) :
  a + b + c = 22 :=
sorry

end abc_sum_is_22_l35_35346


namespace conference_attendees_l35_35928

theorem conference_attendees (A : ℝ) (h1 : 0.10 * (1 - 0.10) * A = 0.9 * (P/100) * A) (h2 : 0.8667 * A)
: P = 96.3 := 
by sorry

end conference_attendees_l35_35928


namespace max_tulips_l35_35440

theorem max_tulips (r y : ℕ) (h₁ : r + y = 2 * (y : ℕ) + 1) (h₂ : |r - y| = 1) (h₃ : 50 * y + 31 * r ≤ 600) :
    r + y = 15 :=
sorry

end max_tulips_l35_35440


namespace good_coloring_four_corners_same_l35_35596

def ConditionX (grid : ℕ → ℕ → bool) : Prop := 
  ∀ i j, (i < n - 1) → (j < n- 1) → 
    (grid i j = grid (i+1) j) → 
    (grid i j = grid i (j+1)) → 
    (grid (i+1) j = grid (i+1) (j+1)) → 
    (grid i (j+1) = grid (i+1) (j+1))

def ConditionY (grid : ℕ → ℕ → bool) : Prop := 
  ∀ i j, (1 ≤ i) → (i < n - 1) → (1 ≤ j) → (j < n - 1) → 
    (grid i (j+1) = grid (i+1) j) ∧ 
    (grid i (j-1) = grid (i-1) j) ∧ 
    (grid (i-1) j = grid (i+1) j) ∧ 
    (grid (i-1) j = grid (i+1) (j+1))

theorem good_coloring_four_corners_same (n : ℕ) (h : n ≥ 3) : 
  (∀ grid : ℕ → ℕ → bool, ConditionX grid ∧ ConditionY grid → 
  (grid 0 0 = grid 0 (n-1) ∧ grid 0 0 = grid (n-1) 0 ∧ grid 0 0 = grid (n-1) (n-1))) :=
by
  sorry

end good_coloring_four_corners_same_l35_35596


namespace max_tulips_count_l35_35443

theorem max_tulips_count : ∃ (r y n : ℕ), 
  n = r + y ∧ 
  n % 2 = 1 ∧ 
  |r - y| = 1 ∧ 
  50 * y + 31 * r ≤ 600 ∧ 
  n = 15 := 
by
  sorry

end max_tulips_count_l35_35443


namespace find_point_on_parabola_l35_35898

theorem find_point_on_parabola :
  ∃ (P : ℝ × ℝ), P.1 = 8 * Real.sqrt 6 ∧ P.2 = 48 ∧
  P.2 > 0 ∧ P.1 > 0 ∧
  (∃ (V F : ℝ × ℝ), V = (0, 0) ∧ F = (0, 2) ∧
  let d := λ (a b : ℝ × ℝ), Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) in
  d P F = 50) := sorry

end find_point_on_parabola_l35_35898


namespace solve_inequality_l35_35872

noncomputable def is_valid_solution (x : ℝ) :=
  (sqrt (x^2 + x) - sqrt (4 - 2 * x)) / (2 * x + 5 - 2 * sqrt (x^2 + 5 * x + 6)) ≤ 0

def condition_1 (x : ℝ) := x * (x + 1) ≥ 0
def condition_2 (x : ℝ) := 4 - 2 * x ≥ 0
def condition_3 (x : ℝ) := x^2 + 5 * x + 6 ≥ 0

theorem solve_inequality : 
  { x : ℝ // condition_1 x ∧ condition_2 x ∧ condition_3 x } →
  { x : ℝ // is_valid_solution x } → 
  x ∈ Set.Iic (-4) ∪ Set.Icc (-2) (-1) ∪ Set.Icc 0 1 :=
sorry

end solve_inequality_l35_35872


namespace parabola_equation_l35_35813

-- Definitions for the given conditions
def parabola_vertex_origin (y x : ℝ) : Prop := y = 0 ↔ x = 0
def axis_of_symmetry_x (y x : ℝ) : Prop := (x = -y) ↔ (x = y)
def focus_on_line (y x : ℝ) : Prop := 3 * x - 4 * y - 12 = 0

-- The statement to be proved
theorem parabola_equation :
  ∀ (y x : ℝ),
  (parabola_vertex_origin y x) ∧ (axis_of_symmetry_x y x) ∧ (focus_on_line y x) →
  y^2 = 16 * x :=
by
  intros y x h
  sorry

end parabola_equation_l35_35813


namespace increasing_interval_and_cos_a_l35_35238

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * (Real.sin x + Real.cos x)

theorem increasing_interval_and_cos_a (a : ℝ) (k : ℤ)
  (h₁ : f (a / 2) = 1 + (3 * Real.sqrt 2) / 5)
  (h₂ : (3 * Real.pi / 4) < a ∧ a < (5 * Real.pi / 4)) :
  (∀ x, k * Real.pi - (Real.pi / 8) ≤ x ∧ x ≤ k * Real.pi + (3 * Real.pi / 8) → 
    Real.derivative f x ≥ 0 ) ∧
  Real.cos a = -(7 * Real.sqrt 2) / 10 :=
by
  sorry

end increasing_interval_and_cos_a_l35_35238


namespace range_of_k_l35_35674

noncomputable def f (x t : ℝ) : ℝ := -t * x^2 + 2 * x + 1

theorem range_of_k (t k : ℝ) (h_t : t < 0)
  (h_lipschitz : ∀ (x1 x2 : ℝ), x1 ∈ [-2, 2] → x2 ∈ [-2, 2] → 
   |f x1 t - f x2 t| ≤ k * |x1 - x2|) : k ∈ [-4 * t + 2, + ∞) := 
sorry

end range_of_k_l35_35674


namespace max_tulips_l35_35447

theorem max_tulips (r y n : ℕ) (hr : r = y + 1) (hc : 50 * y + 31 * r ≤ 600) :
  r + y = 2 * y + 1 → n = 2 * y + 1 → n = 15 :=
by
  -- Given the constraints, we need to identify the maximum n given the costs and relationships.
  sorry

end max_tulips_l35_35447


namespace optimal_point_is_circumcenter_l35_35590

variable (A B C P O : Type)
variable (AP BP CP : ℝ)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited O]

-- Define the properties of the triangle and the distance maximization
def is_acute_angle_triangle (A B C : Type) : Prop := sorry
def is_circumcenter (P : Type) (A B C : Type) : Prop := sorry
def max_distance (AP BP CP : ℝ) : ℝ := max AP (max BP CP)

-- Main statement
theorem optimal_point_is_circumcenter
  (abc_acute_right : is_acute_angle_triangle A B C)
  (minimal_point : ∀ P, ∃ O, is_circumcenter O A B C ∧ max_distance AP BP CP = max_distance (dist O A) (dist O B) (dist O C)) :
  ∀ P, is_circumcenter P A B C → max_distance AP BP CP = max_distance (dist P A) (dist P B) (dist P C) := sorry

end optimal_point_is_circumcenter_l35_35590


namespace inequality_solution_l35_35050

theorem inequality_solution (x : ℝ) : 
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1 / 2 < x ∧ x ≤ 1 :=
sorry

end inequality_solution_l35_35050


namespace all_black_after_4_operations_l35_35377

-- Define the initial conditions and the operation rule
def initial_stones (x1 x2 x3 x4 : ℤ) := x1 = 1 ∨ x1 = -1 ∧ x2 = 1 ∨ x2 = -1 ∧ x3 = 1 ∨ x3 = -1 ∧ x4 = 1 ∨ x4 = -1

-- Define the transformation rule
def operation (x y : ℤ) : ℤ :=
if x = y then 1 else -1

-- Define the sequence of operations
noncomputable def sequence_op (x1 x2 x3 x4 : ℤ) : ℤ × ℤ × ℤ × ℤ :=
let a := operation x1 x2,
    b := operation x2 x3,
    c := operation x3 x4,
    d := operation x4 x1,
    e := operation a b,
    f := operation b c,
    g := operation c d,
    h := operation d a,
    i := operation e f,
    j := operation f g,
    k := operation g h,
    l := operation h e in
(operation i j, operation j k, operation k l, operation l i)

-- Proof statement
theorem all_black_after_4_operations (x1 x2 x3 x4 : ℤ) (h : initial_stones x1 x2 x3 x4) :
let (y1, y2, y3, y4) := sequence_op x1 x2 x3 x4 in
y1 = 1 ∧ y2 = 1 ∧ y3 = 1 ∧ y4 = 1 :=
sorry

end all_black_after_4_operations_l35_35377


namespace remaining_insects_l35_35795

-- Definitions based on conditions
def spiders := 3
def ants := 12
def ladybugs := 8
def ladybugs_flew_away := 2

-- Main statement encapsulating the proof problem
theorem remaining_insects (spiders : ℕ) (ants : ℕ) (ladybugs : ℕ) (ladybugs_flew_away : ℕ) :
  spiders = 3 → ants = 12 → ladybugs = 8 → ladybugs_flew_away = 2 →
  (spiders + ants + ladybugs - ladybugs_flew_away) = 21 :=
by
  intros h_spiders h_ants h_ladybugs h_ladybugs_flew_away
  rw [h_spiders, h_ants, h_ladybugs, h_ladybugs_flew_away]
  norm_num
  sorry

end remaining_insects_l35_35795


namespace no_solution_to_inequality_l35_35028

theorem no_solution_to_inequality (x : ℝ) (h : x ≥ -1/4) : ¬(-1 - 1 / (3 * x + 4) < 2) :=
by sorry

end no_solution_to_inequality_l35_35028


namespace tenth_term_of_sequence_l35_35965

noncomputable def sequence : ℕ → ℚ
| 0       := 3
| 1       := 4
| (n + 2) := 12 / (sequence n)

theorem tenth_term_of_sequence : sequence 9 = 4 :=
by
  sorry

end tenth_term_of_sequence_l35_35965


namespace fish_catch_approx_l35_35310

theorem fish_catch_approx (N : ℕ) (hN : N = 1500) (tagged_count : ℕ) (second_catch_tagged : ℕ) 
    (h_tagged_count : tagged_count = 60) (h_second_catch_tagged : second_catch_tagged = 2) :
    ∃ (x : ℕ), x ≈ 50 :=
begin
  sorry
end

end fish_catch_approx_l35_35310


namespace propositions_correct_l35_35816

def f (x : Real) (b c : Real) : Real := x * abs x + b * x + c

-- Define proposition P1: When c = 0, y = f(x) is an odd function.
def P1 (b : Real) : Prop :=
  ∀ x : Real, f x b 0 = - f (-x) b 0

-- Define proposition P2: When b = 0 and c > 0, the equation f(x) = 0 has only one real root.
def P2 (c : Real) : Prop :=
  c > 0 → ∃! x : Real, f x 0 c = 0

-- Define proposition P3: The graph of y = f(x) is symmetric about the point (0, c).
def P3 (b c : Real) : Prop :=
  ∀ x : Real, f x b c = 2 * c - f x b c

-- Define the final theorem statement
theorem propositions_correct (b c : Real) : P1 b ∧ P2 c ∧ P3 b c := sorry

end propositions_correct_l35_35816


namespace ticket_sales_revenue_l35_35433

theorem ticket_sales_revenue :
  let student_ticket_price := 4
  let general_admission_ticket_price := 6
  let total_tickets_sold := 525
  let general_admission_tickets_sold := 388
  let student_tickets_sold := total_tickets_sold - general_admission_tickets_sold
  let money_from_student_tickets := student_tickets_sold * student_ticket_price
  let money_from_general_admission_tickets := general_admission_tickets_sold * general_admission_ticket_price
  let total_money_collected := money_from_student_tickets + money_from_general_admission_tickets
  total_money_collected = 2876 :=
by
  sorry

end ticket_sales_revenue_l35_35433


namespace smaller_than_neg5_l35_35494

theorem smaller_than_neg5 (
  a : ℤ := 1,
  b : ℤ := 0,
  c : ℤ := -4,
  d : ℤ := -6
) : d < -5 :=
by {
  sorry
}

end smaller_than_neg5_l35_35494


namespace final_price_of_shirt_l35_35333

theorem final_price_of_shirt (original_price : ℝ) (first_discount_rate : ℝ) (second_discount_rate : ℝ) : 
  original_price = 32 → 
  first_discount_rate = 0.25 → 
  second_discount_rate = 0.25 → 
  let first_discount := original_price * first_discount_rate,
      price_after_first := original_price - first_discount,
      second_discount := price_after_first * second_discount_rate,
      final_price := price_after_first - second_discount 
  in final_price = 18 :=
by
  intros h_orig h_first_disc_rate h_second_disc_rate
  let first_discount := original_price * first_discount_rate 
  let price_after_first := original_price - first_discount
  let second_discount := price_after_first * second_discount_rate 
  let final_price := price_after_first - second_discount 
  sorry

end final_price_of_shirt_l35_35333


namespace seven_points_not_all_isosceles_l35_35255

theorem seven_points_not_all_isosceles :
  ∀ (points : Finset (EuclideanSpace ℝ (Fin 2))), 
  points.card = 7 → 
  ∃ (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)), 
  {p1, p2, p3} ⊆ points ∧ ¬isosceles p1 p2 p3 := 
by
  sorry

def isosceles (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)) : Prop := 
  (dist p1 p2 = dist p2 p3) ∨ 
  (dist p1 p3 = dist p2 p3) ∨ 
  (dist p1 p2 = dist p1 p3)

end seven_points_not_all_isosceles_l35_35255


namespace problem_conditions_part_one_part_two_l35_35749

noncomputable def a : ℝ := sorry  -- to be defined as one root of the equation
noncomputable def b : ℝ := sorry  -- to be defined as the other root of the equation

theorem problem_conditions : a > b ∧ (a^2 - 6*a + 4 = 0) ∧ (b^2 - 6*b + 4 = 0) := sorry

theorem part_one : a > 0 ∧ b > 0 :=
by
  have h : a > b ∧ (a^2 - 6*a + 4 = 0) ∧ (b^2 - 6*b + 4 = 0) := problem_conditions
  sorry

theorem part_two : (sqrt a - sqrt b) / (sqrt a + sqrt b) = sqrt 5 / 5 :=
by
  have h : a > b ∧ (a^2 - 6*a + 4 = 0) ∧ (b^2 - 6*b + 4 = 0) := problem_conditions
  have pos_a_b : a > 0 ∧ b > 0 := part_one
  sorry

end problem_conditions_part_one_part_two_l35_35749


namespace tenth_term_is_four_l35_35972

noncomputable def a : ℕ → ℝ
| 0     := 3
| 1     := 4
| (n + 1) := 12 / a n

theorem tenth_term_is_four : a 9 = 4 :=
by
  sorry

end tenth_term_is_four_l35_35972


namespace sqrt3_pow_log_sqrt3_8_eq_8_l35_35490

theorem sqrt3_pow_log_sqrt3_8_eq_8 : (Real.sqrt 3) ^ (Real.log 8 / Real.log (Real.sqrt 3)) = 8 :=
by
  sorry

end sqrt3_pow_log_sqrt3_8_eq_8_l35_35490


namespace no_geom_fib_seq_without_sqrt5_l35_35506

theorem no_geom_fib_seq_without_sqrt5 
  (p_arithmetic : Type) 
  (sqrt5_impossible : ¬ ∃ (t : p_arithmetic), t^2 = 5) :
  ¬ ∃ (f : ℕ → p_arithmetic), 
    (∀ n, f(n + 2) = f(n) + f(n + 1)) ∧
    ∃ (q : p_arithmetic), ∀ n, f(n + 1) = q * f(n) := 
sorry

end no_geom_fib_seq_without_sqrt5_l35_35506


namespace badgers_win_at_least_five_games_l35_35035

noncomputable def badgers_probability_at_least_five_wins : ℚ :=
  ∑ k in Finset.range 10, if 5 ≤ k then (Nat.choose 9 k) * (0.5 : ℚ)^9 else 0
  
theorem badgers_win_at_least_five_games :
  badgers_probability_at_least_five_wins = 1 / 2 := 
sorry

end badgers_win_at_least_five_games_l35_35035


namespace find_line_l35_35843

def point_on_line (P : ℝ × ℝ) (m b : ℝ) : Prop :=
  P.2 = m * P.1 + b

def intersection_points_distance (k m b : ℝ) : Prop :=
  |(k^2 - 4*k + 4) - (m*k + b)| = 6

noncomputable def desired_line (m b : ℝ) : Prop :=
  point_on_line (2, 3) m b ∧ ∀ (k : ℝ), intersection_points_distance k m b

theorem find_line : desired_line (-6) 15 := sorry

end find_line_l35_35843


namespace avg_tickets_sold_by_male_l35_35127

theorem avg_tickets_sold_by_male (M F : ℕ) 
  (avg_per_member : ℕ) 
  (avg_female : ℕ) 
  (ratio : ℕ → ℕ → Prop) 
  (h1 : avg_per_member = 66) 
  (h2 : avg_female = 70) 
  (h3 : ratio M F) 
  (h4 : ratio = (λ m f, 2 * m = f)) : 
  66 = avg_per_member :=
by {
  -- Placeholder for proof
  sorry
}

end avg_tickets_sold_by_male_l35_35127


namespace domain_of_f_l35_35810

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then 2*x - x^2
  else if -2 ≤ x ∧ x < 0 then x^2 + 6*x
  else 0

theorem domain_of_f : set_of (λ x, (0 ≤ x ∧ x ≤ 3) ∨ (-2 ≤ x ∧ x < 0)) = set.interval (-2 : ℝ) (3 : ℝ) :=
sorry

end domain_of_f_l35_35810


namespace rectangle_centers_locus_correct_l35_35290

open Set

-- Define the necessary geometric entities and conditions
variables (O1 O2 A B C D : ℝ) (R1 R2 : ℝ)

-- Assuming we have two intersecting circles with the centers and radii defined
-- Points of intersection on the line connecting the two centers
def intersecting_points (O1 O2 : ℝ) := (A, B, C, D)

-- Define midpoints of specific segments
def midpoint (x y: ℝ) : ℝ := (x + y) / 2

-- Define the locus of centers of the rectangles
noncomputable def rectangle_centers_locus (O1 O2 : ℝ) (R1 R2 : ℝ) :=
  let {A, B, C, D} := intersecting_points O1 O2 in
  let midpoint_AB := midpoint A B in
  let midpoint_AD := midpoint A D in
  let midpoint_BC := midpoint B C in
  let midpoint_CD := midpoint C D in
  ((midpoint_AB, midpoint_AD) ∪ (midpoint_BC, midpoint_CD)) \ {midpoint_AB, midpoint_AD, midpoint_BC, midpoint_CD}

-- The main theorem to be proved
theorem rectangle_centers_locus_correct :
  ∀ (O1 O2 : ℝ) (R1 R2 : ℝ),
  rectangle_centers_locus O1 O2 R1 R2 = 
    ((midpoint A B, midpoint A D) ∪ (midpoint B C, midpoint C D)) \ {midpoint A B, midpoint A D, midpoint B C, midpoint C D} := 
  by sorry

end rectangle_centers_locus_correct_l35_35290


namespace odd_function_when_x_neg_l35_35039

noncomputable def f (x : ℝ) : ℝ :=
  if h : x > 0 then -x + 1
  else if x < 0 then -( -x + 1) -- equivalent to -x - 1
  else 0 -- by odd function property, f(0) = 0

theorem odd_function_when_x_neg {x : ℝ} (h : x < 0) :
  f x = -( -x + 1) :=
by
  have h1 : x > 0 ∨ x = 0 := or.inl (neg_pos.mpr h),
  simp [f, h, h1]
  sorry

end odd_function_when_x_neg_l35_35039


namespace sufficient_but_not_necessary_l35_35112

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 0 → x^2 + x > 0) ∧ (∃ y : ℝ, y < -1 ∧ y^2 + y > 0) :=
by
  sorry

end sufficient_but_not_necessary_l35_35112


namespace necessarily_positive_l35_35792

theorem necessarily_positive (x y z : ℝ) (h1 : 0 < x ∧ x < 2) (h2 : -2 < y ∧ y < 0) (h3 : 0 < z ∧ z < 3) : 
  y + 2 * z > 0 := 
sorry

end necessarily_positive_l35_35792


namespace unique_paths_max_bound_l35_35163

theorem unique_paths_max_bound (m n : ℕ) :
  let S := m * n in
  f_{(m, n)} ≤ 2^S :=
sorry

end unique_paths_max_bound_l35_35163


namespace wall_length_l35_35909

theorem wall_length (s : ℕ) (w : ℕ) (a_ratio : ℕ) (A_mirror : ℕ) (A_wall : ℕ) (L : ℕ) 
  (hs : s = 24) (hw : w = 42) (h_ratio : a_ratio = 2) 
  (hA_mirror : A_mirror = s * s) 
  (hA_wall : A_wall = A_mirror * a_ratio) 
  (h_area : A_wall = w * L) : L = 27 :=
  sorry

end wall_length_l35_35909


namespace true_option_is_C_l35_35860

-- Definitions for options A, B, C, D
def option_A (l1 l2 : ℝ → Prop) (h : ∀ x, l1 x ↔ l2 x) : Prop :=
∀ α β, (l1 α ∧ l2 β) → α = β

def option_B (polygon : ℕ → ℝ) : Prop :=
(sum polygon 0 (n - 1)) < (sum (λ i, 180 - polygon i) 0 (n - 1))

def option_C (l1 l2 l3 : ℝ → Prop) : Prop :=
((∀ x, l1 x ↔ l3 x) ∧ (∀ x, l2 x ↔ l3 x)) → (∀ x, l1 x ↔ l2 x)

def option_D (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : Prop :=
a^2 + b^2 = (a + b)^2

-- The main statement to prove
theorem true_option_is_C (l1 l2 l3 : ℝ → Prop) (H1 : ∀ α β, (l1 α ∧ l2 β) → α = β)
  (polygon : ℕ → ℝ) (H2 : (sum polygon 0 (n - 1)) < (sum (λ i, 180 - polygon i) 0 (n - 1)))
  (H3 : (∀ x, l1 x ↔ l3 x) ∧ (∀ x, l2 x ↔ l3 x))
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (H4 : a^2 + b^2 = (a + b)^2) : 
  option_C l1 l2 l3 ∧ ¬option_A l1 l2 H1 ∧ ¬option_B polygon H2 ∧ ¬option_D a b ha hb :=
by {
  sorry
}

end true_option_is_C_l35_35860


namespace range_of_y_l35_35702

theorem range_of_y (y : ℝ) (hy : y > 0) (hceiling_floor : (⌈y⌉₊ : ℝ) * (⌊y⌋₊ : ℝ) = 72) :
  y ∈ set.Ioo (8 : ℝ) 9 :=
sorry

end range_of_y_l35_35702


namespace point_P_inside_circle_l35_35671

theorem point_P_inside_circle
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (e : ℝ)
  (h4 : e = 1 / 2)
  (x1 x2 : ℝ)
  (hx1 : a * x1 ^ 2 + b * x1 - c = 0)
  (hx2 : a * x2 ^ 2 + b * x2 - c = 0) :
  x1 ^ 2 + x2 ^ 2 < 2 :=
by
  sorry

end point_P_inside_circle_l35_35671


namespace correct_statements_count_l35_35201

-- Problem conditions
def prop_1 (x : ℝ) := x^2 - 3 * x + 2 ≥ 0
def neg_prop_1 (x : ℝ) := x^2 - 3 * x + 2 < 0

def prop_2 (P : α → Prop) := ∀ x, P x
def neg_prop_2 (P : α → Prop) := ∃ x, ¬ P x

def nec_suff_3 (p q : Prop) := (¬p → q) → (p → ¬q)

def suff_not_nec_4 (M N : ℝ) (a : ℝ) (h : 1 < a) := 
  (M > N) → (log a M > log a N) ∧ ¬((log a M > log a N) → M > N)

theorem correct_statements_count : 
  (¬ ∃ x, prop_1 x) ∧ 
  (¬ ∀ x, P x → ¬ ∃ x, ¬ P x) ∧ 
  (nec_suff_3 p q) ∧ 
  (suff_not_nec_4 M N a h) → 
  1 = 1 :=
by sorry

end correct_statements_count_l35_35201


namespace mod_computation_l35_35582

theorem mod_computation (a b n : ℕ) (h_modulus : n = 7) (h_a : a = 47) (h_b : b = 28) :
  (a^2023 - b^2023) % n = 5 :=
by
  sorry

end mod_computation_l35_35582


namespace find_ratio_l35_35354

noncomputable theory -- Since we're dealing with complex numbers and division

open Complex

variables (a b z1 z2 : ℂ)

-- The conditions outlined in the problem
def quadratic_roots (a b z1 z2 : ℂ) : Prop :=
  z1 * z1 + a * z1 + b = 0 ∧
  z2 * z2 + a * z2 + b = 0

def right_angle_triangle (z1 z2 : ℂ) : Prop :=
  ∃ i : ℂ, i = Complex.I ∧ z2 = i * z1

-- The theorem we want to prove
theorem find_ratio (a b z1 z2 : ℂ) (h1 : quadratic_roots a b z1 z2) (h2 : right_angle_triangle z1 z2) : 
  a ≠ 0 ∧ b ≠ 0 → (a^2 / b) = 2 :=
sorry

end find_ratio_l35_35354


namespace biquadratic_root_neg_root_l35_35786

theorem biquadratic_root_neg_root (a b c α : ℝ) :
  a * α^4 + b * α^2 + c = 0 → a * (-α)^4 + b * (-α)^2 + c = 0 :=
by 
  intro h
  calc
    a * (-α)^4 + b * (-α)^2 + c
      = a * α^4 + b * α^2 + c : by simp [pow_four, pow_two]
    ... = 0 : h

end biquadratic_root_neg_root_l35_35786


namespace cos_pi_minus_half_alpha_l35_35261

-- Conditions given in the problem
variable (α : ℝ)
variable (hα1 : 0 < α ∧ α < π / 2)
variable (hα2 : Real.sin α = 3 / 5)

-- The proof problem statement
theorem cos_pi_minus_half_alpha (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.sin α = 3 / 5) : 
  Real.cos (π - α / 2) = -3 * Real.sqrt 10 / 10 := 
sorry

end cos_pi_minus_half_alpha_l35_35261


namespace long_sleeve_shirts_l35_35373

variable (short_sleeve long_sleeve : Nat)
variable (total_shirts washed_shirts : Nat)
variable (not_washed_shirts : Nat)

-- Given conditions
axiom h1 : short_sleeve = 9
axiom h2 : total_shirts = 29
axiom h3 : not_washed_shirts = 1
axiom h4 : washed_shirts = total_shirts - not_washed_shirts

-- The question to be proved
theorem long_sleeve_shirts : long_sleeve = washed_shirts - short_sleeve := by
  sorry

end long_sleeve_shirts_l35_35373


namespace no_one_has_card_less_than_03_l35_35337

def jungkooks_card := 0.8
def yoongis_card := 1/2 -- Lean automatically understands this as 0.5
def yoojeongs_card := 0.9

theorem no_one_has_card_less_than_03 :
  (if jungkooks_card < 0.3 then 1 else 0) +
  (if yoongis_card < 0.3 then 1 else 0) +
  (if yoojeongs_card < 0.3 then 1 else 0) = 0 := by
  sorry

end no_one_has_card_less_than_03_l35_35337


namespace ellipse_equation_line_equation_l35_35251

theorem ellipse_equation (a b : ℝ) (h_ab : a > b > 0) (eccentricity : ℝ) (A : ℝ × ℝ) (h_e : eccentricity = 1 / 2) 
    (h_A : A = (1, 3/2)) (h_pass : ((1 : ℝ) / a)^2 + (3/2 / b)^2 = 1) : 
    ∃ (c : ℝ), (a = 2 * c) ∧ (b = Real.sqrt (a^2 - c^2)) ∧ c = 1 ∧ 
    ( ∀ (x y : ℝ), (x / (2 * c))^2 + (y / (Real.sqrt (3 * c^2)))^2 = 1 → 
      (x / 2)^2 + (y / Real.sqrt 3)^2 = 1 ) := 
sorry

theorem line_equation (B D : ℝ × ℝ) (BD DA : ℝ × ℝ) (h_B : ∃ x0 y0, B = (x0, y0)) (h_D : ∃ m, D = (0, m)) 
    (h_BD : BD = ⟨-B.1, D.2 - B.2⟩) (h_DA : DA = ⟨1, 3/2 - D.2⟩) (h_vec : BD = 2 • DA)  
    (h_B_on_ellipse : (B.1 / 2)^2 + (B.2 / Real.sqrt 3)^2 = 1) :
    ∃ m : ℝ, (∀ (x y : ℝ), y = 1 / 2 * x + m → ((B.1 = -2 ) ∧ (D.2 = 1))) := 
sorry

end ellipse_equation_line_equation_l35_35251


namespace special_pair_example_1_special_pair_example_2_special_pair_negation_l35_35007

-- Definition of "special rational number pair"
def is_special_rational_pair (a b : ℚ) : Prop := a + b = a * b - 1

-- Problem (1)
theorem special_pair_example_1 : is_special_rational_pair 5 (3 / 2) :=
  by sorry

-- Problem (2)
theorem special_pair_example_2 (a : ℚ) : is_special_rational_pair a 3 → a = 2 :=
  by sorry

-- Problem (3)
theorem special_pair_negation (m n : ℚ) : is_special_rational_pair m n → ¬ is_special_rational_pair (-n) (-m) :=
  by sorry

end special_pair_example_1_special_pair_example_2_special_pair_negation_l35_35007


namespace problem_solved_by_half_participants_l35_35307

variables (n m : ℕ)
variable (solve : ℕ → ℕ → Prop)  -- solve i j means participant i solved problem j

axiom half_n_problems_solved : ∀ i, i < m → (∃ s, s ≥ n / 2 ∧ ∀ j, j < n → solve i j → j < s)

theorem problem_solved_by_half_participants (h : ∀ i, i < m → (∃ s, s ≥ n / 2 ∧ ∀ j, j < n → solve i j → j < s)) : 
  ∃ j, j < n ∧ (∃ count, count ≥ m / 2 ∧ (∃ i, i < m → solve i j)) :=
  sorry

end problem_solved_by_half_participants_l35_35307


namespace sum_of_factors_of_120_is_37_l35_35827

theorem sum_of_factors_of_120_is_37 :
  ∃ a b c d e : ℤ, (a * b = 120) ∧ (b = a + 1) ∧ (c * d * e = 120) ∧ (d = c + 1) ∧ (e = d + 1) ∧ (a + b + c + d + e = 37) :=
by
  sorry

end sum_of_factors_of_120_is_37_l35_35827


namespace radius_of_inscribed_circle_in_COD_l35_35400

theorem radius_of_inscribed_circle_in_COD
  (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ)
  (H1 : r1 = 6)
  (H2 : r2 = 2)
  (H3 : r3 = 1.5)
  (H4 : 1/r1 + 1/r3 = 1/r2 + 1/r4) :
  r4 = 3 :=
by
  sorry

end radius_of_inscribed_circle_in_COD_l35_35400


namespace minimize_perimeter_of_triangle_l35_35719

-- Define the basic setup and necessary structures for the geometric problem
variables (M N : Type) [plane_angle : real] (alpha : ℝ)
variables (sphere : Type) (O : sphere) -- Center of the sphere
variables (P Q : point) -- Common tangent points on faces
variables (A B C : point) -- Points on sphere's surface, and half-planes

-- Define the conditions
axiom tangency : ∀ (p : point), p ∈ sphere → (p ∈ M) ∨ (p ∈ N)
axiom acute_dihedral_angle : 0 < alpha ∧ alpha < (π / 2)
axiom geometric_reflection : reflects(A, M, A') ∧ reflects(A, N, A'')
axiom minimization : ∀ A, B, C, minimize_perimeter(A, B, C, l)

-- Define the goal
theorem minimize_perimeter_of_triangle : ∃ (A B C : point),
  (A ∈ sphere) ∧
  (B ∈ M) ∧
  (C ∈ N) ∧
  min_perimeter (triangle A B C) :=
by
  -- Here we define an outline of the geometric properties and setup
  sorry

end minimize_perimeter_of_triangle_l35_35719


namespace radius_of_larger_circle_l35_35434

theorem radius_of_larger_circle (AB : ℝ) (r : ℝ) (R : ℝ)
  (H1 : 2 * R = 5 * r) -- radii ratio 2:5 means 2 * R = 5 * r or r = (2/5) * R
  (H2 : AB = 8) -- AB is given as 8
  (H3 : R = 10) -- R is the radius of the larger circle
  : R = 10 := 
begin
  -- also define semantics of the proof using the triangle angles and similarity 
  sorry
end

end radius_of_larger_circle_l35_35434


namespace max_triangle_area_l35_35270

-- Conditions
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1
def right_focus : ℝ × ℝ := (1, 0)
def chord_passes_focus (P Q : ℝ × ℝ) : Prop :=
  let F2 := right_focus in
  ∃ m : ℝ, P.1 = m * P.2 + 1 ∧ Q.1 = m * Q.2 + 1

-- Prove maximum area
theorem max_triangle_area (P Q : ℝ × ℝ) :
  ellipse_eq P.1 P.2 → ellipse_eq Q.1 Q.2 → chord_passes_focus P Q →
  ∃ S : ℝ, S = 3 ∧ 
  (∀ F1 : ℝ × ℝ, F1 = (-1, 0) → ∃ a : ℝ, a = abs (P.2 - Q.2) / (3 * (m^2 + 1) + 1) * 12) :=
sorry

end max_triangle_area_l35_35270


namespace eccentricity_of_ellipse_l35_35653

noncomputable def eccentricity_range (a b : ℝ) (a_gt_b : a > b) (a_b_pos : 0 < b) (AF1_leq_4BF1 : ∀ {F1 F2 : ℝ}, |AF1| ≤ 4 * |BF1|) : set ℝ :=
  { e | e ∈ (set.Ioo (real.sqrt 2 / 2) (real.sqrt 17 / 5)) ∪ {real.sqrt 17 / 5} }

theorem eccentricity_of_ellipse (a b : ℝ) (a_gt_b : a > b) (a_b_pos : 0 < b) (AF1_leq_4BF1 : ∀ {F1 F2 : ℝ}, |AF1| ≤ 4 * |BF1|)
  (e : ℝ) (h : ∃ C : set (ℝ × ℝ), C = {p | ∀ x y : ℝ, (x, y) ∈ C ↔ x^2 / a^2 + y^2 / b^2 = 1}) :
  e ∈ eccentricity_range a b a_gt_b a_b_pos AF1_leq_4BF1 :=
sorry

end eccentricity_of_ellipse_l35_35653


namespace area_of_triangle_f_is_monotonically_increasing_l35_35275

-- Define the function f(x)
def f (x : Real) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos (Real.pi + 2 * x)

-- Define the interval where f(x) is monotonically increasing
def is_monotonically_increasing (x : Real) :=
  ∃ k : ℤ, -Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi

-- Define the sides of triangle and the condition f(C) = 1
variables (a b c : Real) (A B C : Real)
variable h1 : f C = 1
variable h2 : c = Real.sqrt 3
variable h3 : a + b = 2 * Real.sqrt 3

-- Prove the area of the triangle ABC
theorem area_of_triangle (h_cosine_rule : c = Real.sqrt 3)
    (h_angles_sum : A + B + C = Real.pi)
    (h_sides_sum : a + b = 2 * Real.sqrt 3)
    (h_cosine_value : Real.cos C = 0.5)
    : 0.5 * a * b * Real.sin C = 3 * Real.sqrt 3 / 4 :=
  sorry

-- Prove that f(x) is monotonically increasing in the given interval
theorem f_is_monotonically_increasing {x : Real} 
    (h : ∃ k : ℤ, -Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi) 
    : is_monotonically_increasing x :=
  sorry

end area_of_triangle_f_is_monotonically_increasing_l35_35275


namespace distribution_of_balls_l35_35428

-- Definition for the problem conditions
inductive Ball : Type
| one : Ball
| two : Ball
| three : Ball
| four : Ball

inductive Box : Type
| box1 : Box
| box2 : Box
| box3 : Box

-- Function to count the number of ways to distribute the balls according to the conditions
noncomputable def num_ways_to_distribute_balls : Nat := 18

-- Theorem statement
theorem distribution_of_balls :
  num_ways_to_distribute_balls = 18 := by
  sorry

end distribution_of_balls_l35_35428


namespace smallest_k_for_digit_sum_945_l35_35203

theorem smallest_k_for_digit_sum_945 : 
  ∃ k : ℕ, (∀ n, n = 7 * (10^k - 1) / 9 ∧ (∑ d in (List.ofDigits 10 (Nat.digits 10 n)), d) = 945) → k = 312 :=
sorry

end smallest_k_for_digit_sum_945_l35_35203


namespace shaded_area_l35_35318

/-- The area of the shaded region in the grid is 24.5. -/
theorem shaded_area {b1 h1 b2 h2 b3 h3 : ℕ} (hb1 : b1 = 3) (hh1 : h1 = 4)
  (hb2 : b2 = 4) (hh2 : h2 = 5) (hb3 : b3 = 5) (hh3 : h3 = 6)
  (base : ℕ) (height : ℕ) (hbase : base = 15) (hheight : height = 5) :
  b1 * h1 + b2 * h2 + b3 * h3 - (base * height / 2) = 24.5 :=
by
  sorry

end shaded_area_l35_35318


namespace largest_two_digit_prime_factor_of_binom_300_150_l35_35082

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

def is_prime (p : ℕ) : Prop :=
  nat.prime p

def is_two_digit (p : ℕ) : Prop :=
  10 ≤ p ∧ p < 100

def less_than_300_divided_by_three (p : ℕ) : Prop :=
  3 * p < 300

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  let n := binomial_coefficient 300 150 in
  ∃ p, is_prime p ∧ is_two_digit p ∧ less_than_300_divided_by_three p ∧
        ∀ q, is_prime q ∧ is_two_digit q ∧ less_than_300_divided_by_three q → q ≤ p ∧ n % p = 0 ∧ p = 97 :=
by
  sorry

end largest_two_digit_prime_factor_of_binom_300_150_l35_35082


namespace cylindrical_to_rectangular_coordinates_l35_35195

theorem cylindrical_to_rectangular_coordinates (r θ z : ℝ) (h1 : r = 6) (h2 : θ = 5 * Real.pi / 3) (h3 : z = 7) :
    (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, 7) :=
by
  rw [h1, h2, h3]
  -- Using trigonometric identities:
  have hcos : Real.cos (5 * Real.pi / 3) = 1 / 2 := sorry
  have hsin : Real.sin (5 * Real.pi / 3) = -(Real.sqrt 3) / 2 := sorry
  rw [hcos, hsin]
  simp
  sorry

end cylindrical_to_rectangular_coordinates_l35_35195


namespace largest_prime_factor_of_binomial_l35_35094

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_prime_factor_of_binomial {p : ℕ} (hp : Nat.Prime p) (hp_range : 10 ≤ p ∧ p < 100) :
  p ∣ binomial 300 150 → p = 97 :=
by
suffices ∀ q : ℕ, Nat.Prime q → 10 ≤ q ∧ q < 100 → q ∣ binomial 300 150 → q ≤ 97
from fun h => le_antisymm (this p hp hp_range h) (le_of_eq (rfl : 97 = 97))
intro q hq hq_range hq_div
sorry

end largest_prime_factor_of_binomial_l35_35094


namespace trihedral_angle_bisectors_perpendicular_l35_35787

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def bisectors_perpendicular (a0 b0 c0 : V) (m n p : V) :=
  m = a0 + b0 ∧ 
  n = c0 + a0 ∧ 
  p = b0 + c0 ∧ 
  inner_product m n = 0

theorem trihedral_angle_bisectors_perpendicular (a0 b0 c0 : V) (m n p : V)
  (h : bisectors_perpendicular a0 b0 c0 m n p) :
  inner_product m p = 0 ∧ inner_product n p = 0 :=
by sorry

end trihedral_angle_bisectors_perpendicular_l35_35787


namespace average_of_last_5_l35_35839

theorem average_of_last_5 (results : Fin 11 → ℝ) (H1 : (∑ i, results i) / 11 = 20)
  (H2 : (∑ i in Finset.range 5, results i) / 5 = 15) (H3 : results 5 = 35) :
  (∑ i in Finset.Ico 6 11, results i) / 5 = 22 := sorry

end average_of_last_5_l35_35839


namespace rectangle_dimensions_l35_35770

theorem rectangle_dimensions (a b: ℕ) (h1 : 2 * (a + b) = 76) (h2 : 40 + 52 - 76 = 2 * a) :
  set.mem (a, b) [{8, 30}, {30, 8}] :=
by {
  let h3 := by linarith [h2],          -- From h2 we get 2 * a = 16
  let a_eq_8 := by linarith [h3],      -- Solving 2a = 16, we get a = 8
  have b_eq_30 := by linarith [h1, a_eq_8], -- Using 2 * (8 + b) = 76, we get b = 30
  exact or.inl rfl, -- Finally the tuple {8, 30} exists in the set [{8, 30}, {30, 8}]
}

end rectangle_dimensions_l35_35770


namespace negation_prop_l35_35042

variable x : ℝ

theorem negation_prop : ¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0 ↔ ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 :=
by
  sorry

end negation_prop_l35_35042


namespace edge_length_correct_cm_l35_35527

noncomputable def edge_length_of_each_small_cube (total_cubes : ℕ) (box_edge_length : ℝ) : ℝ :=
  let volume_of_box := box_edge_length ^ 3
  let volume_of_each_cube := volume_of_box / total_cubes
  (volume_of_each_cube) ^ (1/3)

theorem edge_length_correct_cm :
  (edge_length_of_each_small_cube 8000 1 * 100) = 5 :=
by
  -- Definitions
  let box_edge_length := 1 -- in meters
  let total_cubes := 8000 

  -- Volume calculations
  let volume_of_box := box_edge_length ^ 3 -- Volume of the box in cubic meters
  let volume_of_each_cube := volume_of_box / total_cubes -- Volume of each smaller cube

  -- Calculation edge length of cube in meters and then convert to cm
  let edge_length_in_m := (volume_of_each_cube) ^ (1 / 3)
  let edge_length_in_cm := edge_length_in_m * 100

  -- Conclude the required result
  exact_mod_cast edge_length_in_cm ≈ 5

end edge_length_correct_cm_l35_35527


namespace number_of_ways_eq_fibonacci_l35_35720

-- Define the Fibonacci sequence recursively
def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Define the function x_n according to the problem
def x_n : ℕ → ℕ
| 1     := 1
| 2     := 1
| (n+1) := if n = 0 then 0 else x_n n + x_n (n-1)

-- Theorem to state that x_n and fibonacci have the same values for all n
theorem number_of_ways_eq_fibonacci (n : ℕ) : x_n n = fibonacci n := 
sorry

end number_of_ways_eq_fibonacci_l35_35720


namespace count_valid_outfits_l35_35690

/-
  I have 7 shirts, 5 pairs of pants, and 7 hats. 
  The pants come in tan, black, blue, gray, and green. 
  The shirts and hats come in those colors plus white and yellow. 
  I refuse to wear an outfit where the shirt, pants, and hat are all from no more than two distinct colors. 

  Prove the number of valid choices for an outfit consisting of one shirt, one pair of pants, and one hat is 151.
-/

theorem count_valid_outfits : 
  let shirts := 7
  let pants := 5
  let hats := 7
  let pants_colors := 5 -- tan, black, blue, gray, green
  let shirts_and_hats_colors := 7 -- tan, black, blue, gray, green, white, yellow
  shirts * pants * hats - (4 + (5.choose 2 * 9)) = 151 := 
by
  let shirts := 7
  let pants := 5
  let hats := 7
  let pants_colors := 5
  let shirts_and_hats_colors := 7
  let total_combinations := shirts * pants * hats
  let invalid_combinations := 4 + (5.choose 2 * 9)
  show total_combinations - invalid_combinations = 151
  sorry

end count_valid_outfits_l35_35690


namespace four_letter_words_with_AE_l35_35295

theorem four_letter_words_with_AE : 
  let letters := {A, B, C, D, E}
  let total_words := 5^4
  let words_without_A := 4^4
  let words_without_E := 4^4
  let words_without_AE := 3^4
  total_words - words_without_A - words_without_E + words_without_AE = 194 :=
by {
  let letters := {A, B, C, D, E}
  let total_words := 5^4
  let words_without_A := 4^4
  let words_without_E := 4^4
  let words_without_AE := 3^4
  sorry
}

end four_letter_words_with_AE_l35_35295


namespace sum_sequence_eq_4024_l35_35682

noncomputable def sequence (n : ℕ) : ℝ := sorry

def C1 : set (ℝ × ℝ) := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 - 4 * p.2 = 0 }

def C2 : ℕ → ℝ → ℝ → set (ℝ × ℝ) :=
  λ n an a2013n, { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 2 * an * p.1 - 2 * a2013n * p.2 = 0 }

def bisects_circumference (C1 C2 : set (ℝ × ℝ)) : Prop := sorry -- Define bisecting property appropriately

theorem sum_sequence_eq_4024 :
  (∀ n ∈ finset.range 1 2013, a_n + a_(2013 - n) = 4) →
  finset.sum (finset.range 1 2013) (λ n, a_n) = 4024 :=
by
  sorry

end sum_sequence_eq_4024_l35_35682


namespace inequality_1_inequality_3_l35_35236

variable (a b : ℝ)
variable (hab : a > b ∧ b ≥ 2)

theorem inequality_1 (hab : a > b ∧ b ≥ 2) : b ^ 2 > 3 * b - a :=
by sorry

theorem inequality_3 (hab : a > b ∧ b ≥ 2) : a * b > a + b :=
by sorry

end inequality_1_inequality_3_l35_35236


namespace minimum_sum_of_distances_l35_35512

-- Define points A, B, C, D and an arbitrary point X.
variables {A B C D X : EuclideanSpace ℝ (Fin 2)}

-- Define the conditions of the problem.
def is_right_trapezoid (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (angle A B C = π / 2) ∧ (angle A D C = π / 2) ∧
  (dist A D = 2 * sqrt 7) ∧ (dist A B = sqrt 21) ∧ (dist B C = 2)

-- Define the problem as a Lean theorem.
theorem minimum_sum_of_distances (h : is_right_trapezoid A B C D):
  (∀ X : EuclideanSpace ℝ (Fin 2), dist X A + dist X B + dist X C + dist X D) ≥ 12 :=
sorry

end minimum_sum_of_distances_l35_35512


namespace number_of_digits_if_million_place_l35_35897

theorem number_of_digits_if_million_place (n : ℕ) (h : n = 1000000) : 7 = 7 := by
  sorry

end number_of_digits_if_million_place_l35_35897


namespace correct_percentage_fruits_in_good_condition_l35_35548

noncomputable def percentage_fruits_in_good_condition
    (total_oranges : ℕ)
    (total_bananas : ℕ)
    (rotten_percentage_oranges : ℝ)
    (rotten_percentage_bananas : ℝ) : ℝ :=
let rotten_oranges := (rotten_percentage_oranges / 100) * total_oranges
let rotten_bananas := (rotten_percentage_bananas / 100) * total_bananas
let good_condition_oranges := total_oranges - rotten_oranges
let good_condition_bananas := total_bananas - rotten_bananas
let total_fruits_in_good_condition := good_condition_oranges + good_condition_bananas
let total_fruits := total_oranges + total_bananas
(total_fruits_in_good_condition / total_fruits) * 100

theorem correct_percentage_fruits_in_good_condition :
  percentage_fruits_in_good_condition 600 400 15 4 = 89.4 := by
  sorry

end correct_percentage_fruits_in_good_condition_l35_35548


namespace digit_count_of_N_l35_35896

theorem digit_count_of_N (N : ℕ) (h_digits : ∀ d ∈ digits N, d = 1 ∨ d = 2)
  (h_len : digits N = 100) (h_even_between_2s : ∀ i j, (i < j ∧ nth_digit N i = 2 ∧ nth_digit N j = 2) → (j - i - 1) % 2 = 0)
  (h_div_by_3 : N % 3 = 0) :
  count_digit N 1 = 98 ∧ count_digit N 2 = 2 :=
begin
  sorry
end

-- This helper definition may be required for digit_list and nth_digit
noncomputable def digits : ℕ → ℕ → List ℕ
  -- Implementation to extract digits, using 
  -- a list to represent the number and nth_digit function
  := sorry

noncomputable def nth_digit : ℕ → ℕ → ℕ
  -- Implementation for extracting the nth digit
  := sorry

-- Function to count specific digit
noncomputable def count_digit : ℕ → ℕ → ℕ
  | n, d := List.count d (digits n)

-- Assume N:
noncomputable def N : ℕ := sorry -- Assume N is given, satisfying the constraints.

end digit_count_of_N_l35_35896


namespace each_trainer_hours_l35_35411

theorem each_trainer_hours (dolphins : ℕ) (hours_per_dolphin : ℕ) (trainers : ℕ) :
  dolphins = 4 →
  hours_per_dolphin = 3 →
  trainers = 2 →
  (dolphins * hours_per_dolphin) / trainers = 6 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end each_trainer_hours_l35_35411


namespace integral_x_plus_one_over_x_l35_35940

theorem integral_x_plus_one_over_x :
  ∫ x in 1..2, x + (1 / x) = (3 / 2) + Real.log 2 :=
by
  sorry

end integral_x_plus_one_over_x_l35_35940


namespace tenth_term_of_sequence_l35_35968

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 0 then 3
else if n = 1 then 4
else 12 / sequence (n - 1)

theorem tenth_term_of_sequence :
  sequence 9 = 4 :=
sorry

end tenth_term_of_sequence_l35_35968


namespace coordinates_of_C_l35_35547

-- Define points A and B
def A : ℝ × ℝ := (2, -2)
def B : ℝ × ℝ := (14, 4)

-- Define the vector from A to B
def AB := (B.1 - A.1, B.2 - A.2)

-- Define the scaling factor
def scale_factor : ℝ := 1 / 2

-- Define the vector BC which is half of AB
def BC := (scale_factor * AB.1, scale_factor * AB.2)

-- Calculate the coordinates of C
def C : ℝ × ℝ := (B.1 + BC.1, B.2 + BC.2)

-- State the theorem
theorem coordinates_of_C : C = (20, 7) := by
  sorry

end coordinates_of_C_l35_35547


namespace probability_of_yellow_light_l35_35569

def time_red : ℕ := 30
def time_green : ℕ := 25
def time_yellow : ℕ := 5
def total_cycle_time : ℕ := time_red + time_green + time_yellow

theorem probability_of_yellow_light :
  (time_yellow : ℚ) / (total_cycle_time : ℚ) = 1 / 12 :=
by
  sorry

end probability_of_yellow_light_l35_35569


namespace number_of_throws_to_return_to_ami_l35_35598

def throw_pattern (n : ℕ) (k : ℕ) (start : ℕ) : ℕ :=
  (start + k - 1) % n + 1

theorem number_of_throws_to_return_to_ami (n : ℕ) (skip : ℕ) :
  ∀ k : ℕ, n = 11 → skip = 4 → 
  let ami_start := 1 in
  let rec throws (curr_girl : ℕ) (count : ℕ) : ℕ :=
      if count ≥ n then count
      else if curr_girl = ami_start ∧ count > 0 then count
      else throws (throw_pattern n skip curr_girl) (count + 1)
  in throws ami_start 0 = 11 := by
  intros _ _ n_eq skip_eq
  rw [n_eq, skip_eq]
  sorry

end number_of_throws_to_return_to_ami_l35_35598


namespace smallest_z_l35_35927

theorem smallest_z (x z : ℝ) (m n : ℤ)
  (h1 : cos x = 1)
  (h2 : cos (x + z) = sqrt 2 / 2) :
  z = π / 4 ∨ z = (7 * π / 4) :=
sorry

end smallest_z_l35_35927


namespace num_integers_condition_l35_35630

theorem num_integers_condition : 
  (∃ (n1 n2 n3 : ℤ), 0 < n1 ∧ n1 < 30 ∧ (∃ k1 : ℤ, (30 - n1) / n1 = k1 ^ 2) ∧
                     0 < n2 ∧ n2 < 30 ∧ (∃ k2 : ℤ, (30 - n2) / n2 = k2 ^ 2) ∧
                     0 < n3 ∧ n3 < 30 ∧ (∃ k3 : ℤ, (30 - n3) / n3 = k3 ^ 2) ∧
                     ∀ n : ℤ, 0 < n ∧ n < 30 ∧ (∃ k : ℤ, (30 - n) / n = k ^ 2) → 
                              (n = n1 ∨ n = n2 ∨ n = n3)) :=
sorry

end num_integers_condition_l35_35630


namespace thabo_paperback_diff_l35_35034

variable (total_books : ℕ) (H_books : ℕ) (P_books : ℕ) (F_books : ℕ)

def thabo_books_conditions :=
  total_books = 160 ∧
  H_books = 25 ∧
  P_books > H_books ∧
  F_books = 2 * P_books ∧
  total_books = F_books + P_books + H_books 

theorem thabo_paperback_diff :
  thabo_books_conditions total_books H_books P_books F_books → 
  (P_books - H_books) = 20 :=
by
  sorry

end thabo_paperback_diff_l35_35034


namespace solve_for_m_l35_35944

theorem solve_for_m (m : ℕ) (h_cond : m > 3) (h_eq : log 10 ((m - 3)! : ℝ) + log 10 ((m - 1)! : ℝ) + 3 = 2 * log 10 (m! : ℝ)) : m = 10 :=
sorry

end solve_for_m_l35_35944


namespace ant_probability_at_C_after_7_moves_l35_35161

-- Definitions of the problem's conditions
def lattice : set (ℤ × ℤ) := { p | true }  -- assume infinite lattice for simplicity

def red_dots : set (ℤ × ℤ) :=
  { (x, y) | x % 2 = 0 ∧ y % 2 = 0 ∨ x % 2 = 1 ∧ y % 2 = 1 }

def A : ℤ × ℤ := (0, 0)
def C : ℤ × ℤ := (0, 2)

-- The ant's movement condition
def is_neighbor (p q : ℤ × ℤ) : Prop :=
  abs (p.1 - q.1) + abs (p.2 - q.2) = 1

-- The ant starts at dot A, moves every minute to a neighboring dot, choosing randomly
noncomputable def ant_path (start : ℤ × ℤ) (steps : ℕ) : ℕ → (ℤ × ℤ)
| 0     := start
| (n+1) := sorry  -- Randomly picks a neighbor, defined noncomputably due to randomness.

-- Define the term in Lean to capture the given problem and conditions:
theorem ant_probability_at_C_after_7_moves :
  ∀ p : fin 9, ∃ q : lattice, ant_path A 7 = C → q = (C) :=
begin
  sorry
end

end ant_probability_at_C_after_7_moves_l35_35161


namespace largest_two_digit_prime_factor_of_binom_300_150_l35_35080

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

def is_prime (p : ℕ) : Prop :=
  nat.prime p

def is_two_digit (p : ℕ) : Prop :=
  10 ≤ p ∧ p < 100

def less_than_300_divided_by_three (p : ℕ) : Prop :=
  3 * p < 300

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  let n := binomial_coefficient 300 150 in
  ∃ p, is_prime p ∧ is_two_digit p ∧ less_than_300_divided_by_three p ∧
        ∀ q, is_prime q ∧ is_two_digit q ∧ less_than_300_divided_by_three q → q ≤ p ∧ n % p = 0 ∧ p = 97 :=
by
  sorry

end largest_two_digit_prime_factor_of_binom_300_150_l35_35080


namespace max_a_avoiding_lattice_points_l35_35890

def is_lattice_point (x y : ℤ) : Prop :=
  true  -- Placeholder for (x, y) being in lattice points.

def passes_through_lattice_point (m : ℚ) (x : ℤ) : Prop :=
  is_lattice_point x (⌊m * x + 2⌋)

theorem max_a_avoiding_lattice_points :
  ∀ {a : ℚ}, (∀ x : ℤ, (0 < x ∧ x ≤ 100) → ¬passes_through_lattice_point ((1 : ℚ) / 2) x ∧ ¬passes_through_lattice_point (a - 1) x) →
  a = 50 / 99 :=
by
  sorry

end max_a_avoiding_lattice_points_l35_35890


namespace trigonometric_identity_proof_l35_35696

theorem trigonometric_identity_proof (α : ℝ) (h : Real.tan α = 3) : (Real.sin (2 * α)) / ((Real.cos α) ^ 2) = 6 :=
by
  sorry

end trigonometric_identity_proof_l35_35696


namespace simplify_expression_eq_l35_35226

theorem simplify_expression_eq (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) :=
by 
  sorry

end simplify_expression_eq_l35_35226


namespace dogs_food_consumption_l35_35064

theorem dogs_food_consumption :
  (let cups_per_meal_momo_fifi := 1.5
   let meals_per_day := 3
   let cups_per_meal_gigi := 2
   let cups_to_pounds := 3
   let daily_food_momo_fifi := cups_per_meal_momo_fifi * meals_per_day * 2
   let daily_food_gigi := cups_per_meal_gigi * meals_per_day
   daily_food_momo_fifi + daily_food_gigi) / cups_to_pounds = 5 :=
by
  sorry

end dogs_food_consumption_l35_35064


namespace nuts_consumed_range_l35_35392

def diet_day_nuts : Nat := 1
def normal_day_nuts : Nat := diet_day_nuts + 2

def total_nuts_consumed (start_with_diet_day : Bool) : Nat :=
  if start_with_diet_day then
    (10 * diet_day_nuts) + (9 * normal_day_nuts)
  else
    (10 * normal_day_nuts) + (9 * diet_day_nuts)

def min_nuts_consumed : Nat :=
  Nat.min (total_nuts_consumed true) (total_nuts_consumed false)

def max_nuts_consumed : Nat :=
  Nat.max (total_nuts_consumed true) (total_nuts_consumed false)

theorem nuts_consumed_range :
  min_nuts_consumed = 37 ∧ max_nuts_consumed = 39 := by
  sorry

end nuts_consumed_range_l35_35392


namespace train_start_time_l35_35849

theorem train_start_time (distance_AB : ℝ) (speed_trainA : ℝ) (speed_trainB : ℝ) (start_time_B : ℝ) (meeting_time : ℝ) : 
  distance_AB = 200 ∧ speed_trainA = 20 ∧ speed_trainB = 25 ∧ start_time_B = 8 ∧ meeting_time = 12 →
  ∃ T : ℝ, T = 7 :=
by
  intros h
  use 7
  sorry

end train_start_time_l35_35849


namespace downhill_distance_approx_correct_l35_35123

noncomputable def downhill_distance_approx (uphill_distance : ℕ) (uphill_speed : ℕ) (downhill_speed : ℕ) (average_speed : ℝ) : ℝ :=
(uphill_distance * downhill_speed * average_speed) /
(downhill_speed * uphill_speed + average_speed * uphill_speed - average_speed * downhill_speed)

theorem downhill_distance_approx_correct :
  downhill_distance_approx 100 30 80 37.89 ≈ 49.96 := 
sorry

end downhill_distance_approx_correct_l35_35123


namespace find_MA_l35_35328

-- Definitions of points, sides, and conditions
variables {A B C M : Type} {AB BC MA R : ℝ}

-- Conditions in Lean 4
def conditions (A B C M : Type) (AB BC R : ℝ) : (AB = 4) ∧ (BC = 6) ∧ (circumradius A B C = 9) ∧ (on_perpendicular_bisector A B C M) ∧ (perpendicular A M A C) := sorry

-- Problem statement in Lean 4
theorem find_MA
  (h : conditions A B C M AB BC R)
  : MA = 6 :=
sorry

end find_MA_l35_35328


namespace sale_in_fifth_month_l35_35538

def sale_first_month : ℝ := 3435
def sale_second_month : ℝ := 3927
def sale_third_month : ℝ := 3855
def sale_fourth_month : ℝ := 4230
def required_avg_sale : ℝ := 3500
def sale_sixth_month : ℝ := 1991

theorem sale_in_fifth_month :
  (sale_first_month + sale_second_month + sale_third_month + sale_fourth_month + s + sale_sixth_month) / 6 = required_avg_sale ->
  s = 3562 :=
by
  sorry

end sale_in_fifth_month_l35_35538


namespace inequality_solution_l35_35953

def op (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

def f (x : ℝ) : ℝ :=
  (op 1 x) * x - 2 * (op 2 x)

theorem inequality_solution :
  ∀ m : ℝ, 0 ≤ m ∧ m ≤ 1 ↔ f (m - 2) ≤ f (2 * m) :=
sorry

end inequality_solution_l35_35953


namespace fourth_individual_is_14_l35_35141

noncomputable def random_selection (table: list (list (string))) (n: ℕ) : ℕ := sorry

theorem fourth_individual_is_14 (table: list (list (string))) 
  (condition1: ∀ i, 1 ≤ i ∧ i ≤ 20) 
  (random_table : table = [["7816", "6572", "0802", "6314", "0702", "4369", "9728", "0198"],
                            ["3204", "9234", "4935", "8200", "3623", "4869", "6938", "7481"]])
  (selection_method : ∀ (t : list (list (string))), ∀ (n1 n2 : string), 
    ∃ (result : list ℕ), (forall s ∈ result, 1 ≤ s ∧ s ≤ 20) ∧
    result.length = 5 ∧ 
    list.nth result (n - 1) = some n1 ∧ list.nth result (n - 2) = some n2) 
  : random_selection table 4 = 14 := sorry

end fourth_individual_is_14_l35_35141


namespace sin_cos_105_eq_l35_35837

theorem sin_cos_105_eq : sin (105 : ℝ) * cos (105 : ℝ) = -1/4 :=
by sorry

end sin_cos_105_eq_l35_35837


namespace shaded_region_area_l35_35048

theorem shaded_region_area (PQ : ℝ) (A : ℕ) (n : ℕ) (congruent_squares : ∀ k, set (ℝ × ℝ)) :
  PQ = 8 → n = 25 → (∀ k, congruent_squares k) → A = 32 → (∃ sq, ∀ (i j : ℕ), sq⟨i, j⟩ ∈ congruent_squares k) :=
by 
  sorry

end shaded_region_area_l35_35048


namespace find_f_neg_a_l35_35709

def f (x : ℝ) : ℝ := x^2 * sin x + 1

theorem find_f_neg_a (a : ℝ) (h : f a = 11) : f (-a) = -9 :=
by
  sorry

end find_f_neg_a_l35_35709


namespace min_distance_sum_l35_35654

-- Definitions of the lines and the parabola
def l1 : (ℝ × ℝ) → Prop := λ P, P.2 = -1
def l2 : (ℝ × ℝ) → Prop := λ P, 3 * P.1 - 4 * P.2 + 19 = 0
def parabola : (ℝ × ℝ) → Prop := λ P, P.1^2 = 4 * P.2

-- Prove the minimum sum of distances
theorem min_distance_sum : 
  (∃ P, parabola P → 
    let d1 := abs (P.2 + 1)
    let d2 := abs (3 * P.1 - 4 * P.2 + 19) / (sqrt (3^2 + (-4)^2))
    d1 + d2 = 3) :=
  sorry

end min_distance_sum_l35_35654


namespace selling_price_of_cricket_bat_l35_35129

variable (profit : ℝ) (profit_percentage : ℝ)
variable (selling_price : ℝ)

theorem selling_price_of_cricket_bat 
  (h1 : profit = 215)
  (h2 : profit_percentage = 33.85826771653544) : 
  selling_price = 849.70 :=
sorry

end selling_price_of_cricket_bat_l35_35129


namespace min_value_y_l35_35621

def y (x : ℝ) : ℝ := (4 / Real.sin x) + Real.sin x

theorem min_value_y : ∃ (c : ℝ), (0 < c ∧ c < real.pi) ∧ 
  ∀ x, (0 < x ∧ x < real.pi) → y(x) ≥ y(c) ∧ y(c) = 5 :=
by
  sorry

end min_value_y_l35_35621


namespace max_value_ta_proof_l35_35746

noncomputable def max_value_ta (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  let E := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 in
  let F := (real.sqrt (a^2 + b^2), 0) in
  let A := (-a, 0) in
  let B := (a, 0) in
  let P := (x, y) in
  let l := fun t => (t, _) in
  let Q := (q_x, q_y) in
  let collinear := (B, P, Q) in -- These points are collinear
  (5 / 4)

theorem max_value_ta_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  max_value_ta a b ha hb = (5 / 4) :=
sorry

end max_value_ta_proof_l35_35746


namespace angle_BDC_18_degrees_l35_35777

def is_right_angled_triangle (A B C : Point) : Prop :=
  ∃ a b c : ℝ, 
    C.angle A B = 90 ∧ 
    a = dist B C ∧ b = dist A C ∧ c = dist A B ∧ 
    a ^ 2 + b ^ 2 = c ^ 2

def H_altitude_condition (A B C H : Point) : Prop :=
  dist H (line_of B C) = dist H A

theorem angle_BDC_18_degrees (A B C D H : Point) 
  (h1: is_right_angled_triangle A B C)
  (h2: D ∈ ray ON A B ∧ dist D C = 2 * dist B C)
  (h3: altitude_from C to line_of A B = H)
  (h4: H_altitude_condition A B C H) : 
  angle D B C = 18 :=
sorry

end angle_BDC_18_degrees_l35_35777


namespace unit_vector_sum_magnitude_l35_35567

variable (a0 b0 : ℝ^3)

def is_unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1

theorem unit_vector_sum_magnitude (h₁ : is_unit_vector a0) (h₂ : is_unit_vector b0) :
  ‖a0‖ + ‖b0‖ = 2 :=
sorry

end unit_vector_sum_magnitude_l35_35567


namespace nested_abs_expression_eval_l35_35185

theorem nested_abs_expression_eval :
  abs (abs (-abs (-2 + 3) - 2) + 3) = 6 := sorry

end nested_abs_expression_eval_l35_35185


namespace expression_evaluation_l35_35942

theorem expression_evaluation : 
  (2^10 * 3^3) / (6 * 2^5) = 144 :=
by 
  sorry

end expression_evaluation_l35_35942


namespace files_remaining_l35_35873

theorem files_remaining 
(h_music_files : ℕ := 16) 
(h_video_files : ℕ := 48) 
(h_files_deleted : ℕ := 30) :
(h_music_files + h_video_files - h_files_deleted = 34) := 
by sorry

end files_remaining_l35_35873


namespace slope_tangent_line_at_origin_l35_35208

open Real

theorem slope_tangent_line_at_origin :
  deriv (λ x : ℝ, exp x) 0 = exp 0 := by
  sorry

end slope_tangent_line_at_origin_l35_35208


namespace total_pizza_eaten_l35_35589

def don_pizzas : ℝ := 80
def daria_pizzas : ℝ := 2.5 * don_pizzas
def total_pizzas : ℝ := don_pizzas + daria_pizzas

theorem total_pizza_eaten : total_pizzas = 280 := by
  sorry

end total_pizza_eaten_l35_35589


namespace circle_k_range_l35_35402

def circle_equation (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

theorem circle_k_range (k : ℝ) (h : ∃ x y, circle_equation k x y) : k > 4 ∨ k < -1 :=
by
  sorry

end circle_k_range_l35_35402


namespace lights_remain_on_l35_35056

-- Define the initial conditions
def total_lights : ℕ := 1000
def total_switches : ℕ := 1000
def switches_pulled : List ℕ := [2, 3, 5]

-- Define a function to count the multiples of a number up to a limit
def count_multiples (n limit : ℕ) : ℕ :=
  limit / n

-- Define a function to count lights controlled by combinations of switches using Inclusion-Exclusion Principle
def inclusion_exclusion (ns : List ℕ) (limit : ℕ) : ℕ :=
  List.sum (ns.toFinset.powerset.toList.filter (λ s => s ≠ ∅)
    .map (λ s, (-1 : ℤ) ^ (s.card + 1) * count_multiples (s.prod id) limit).map (Int.toNat))

-- Define the main problem
theorem lights_remain_on : 
  ∀ (total_lights total_switches : ℕ) (switches_pulled : List ℕ), 
  total_lights = 1000 →
  total_switches = 1000 →
  switches_pulled = [2, 3, 5] →
  (total_lights - inclusion_exclusion switches_pulled total_lights) = 499
:= by
  intros
  sorry  -- Proof is intentionally omitted

end lights_remain_on_l35_35056


namespace perimeter_of_triangle_l35_35830

theorem perimeter_of_triangle (A B C : Point) (h : right_triangle A B C)
  (hAB : distance A B = 8) (hAC : distance A C = 15) : 
  distance A B + distance A C + distance B C = 40 := 
sorry

end perimeter_of_triangle_l35_35830


namespace smallest_n_for_integer_x_l35_35912

theorem smallest_n_for_integer_x (n : ℕ) (x : ℕ) (h_pos_n : 0 < n) (h_cost : 0.0107 * x = n) :
  n = 107 :=
sorry

end smallest_n_for_integer_x_l35_35912


namespace triangle_proof_l35_35326

-- Lean statement for the math proof problem
theorem triangle_proof (A B C D E : Type)
  (AB BC : ℝ) (AC : ℝ) (BD : Set (A -> B -> C -> Type))
  (midpoint : Set (B -> C -> E -> Type))
  (BD_angle_bisector: ∀ P Q R, (BD P Q R) -> (midpoint Q R E))
  (AB_eq : AB = 2)
  (BC_eq : BC = 3)
  (AC_eq : AC = real.sqrt 7) :
  ∃ AE BD, 
    AE = (real.sqrt 13) / 2 ∧ 
    BD = (6 * real.sqrt 3) / 5 := by
  sorry

end triangle_proof_l35_35326


namespace minimum_distance_proof_trajectory_Q_proof_trajectory_general_proof_l35_35664

variables (C : Type*) {x y t k m} [LinearOrderedField t]

-- problem 1
def minimum_distance (M : t × t) (t > 4) : t :=
  if 4 < t ∧ t ≤ 5 then t - 4 
  else if t > 5 then (1 / 5) * (real.sqrt (5 * t^2 - 100)) 
  else 0

theorem minimum_distance_proof (M : t × t) (t : t) (h: t > 4) :
  minimum_distance M t h = if 4 < t ∧ t ≤ 5 then t - 4 
  else if t > 5 then (1 / 5) * (real.sqrt (5 * t^2 - 100)) 
  else 0 := sorry

-- problem 2(i)
variable {Q : t × t}

def trajectory_Q : set (t × t) := { p : t × t | p.2^2 / 25 - p.1^2 / 100 = 1 }

theorem trajectory_Q_proof (k : t) (hk : k ≠ 2 ∧ k ≠ -2) : 
  Q ∈ trajectory_Q := sorry

-- problem 2(ii)
def trajectory_general (a b : t) : set (t × t) := 
  { p : t × t | p.2^2 / ((a^2 + b^2) / a)^2 - p.1^2 / ((a^2 + b^2) / b)^2 = 1 }

theorem trajectory_general_proof (a b : t) (k : t) 
  (hk : k ≠ a / b ∧ k ≠ -a / b) :
  Q ∈ trajectory_general a b := sorry

end minimum_distance_proof_trajectory_Q_proof_trajectory_general_proof_l35_35664


namespace digits_in_product_l35_35686

theorem digits_in_product : 
  let x := 3^7 * 7^5
  ∃ d : ℕ, d = Nat.floor (Real.log10 x) + 1 ∧ d = 8 :=
begin
  sorry
end

end digits_in_product_l35_35686


namespace digit_in_452nd_place_l35_35859

def repeating_sequence : List Nat := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]
def repeat_length : Nat := 18

theorem digit_in_452nd_place :
  (repeating_sequence.get ⟨(452 % repeat_length) - 1, sorry⟩ = 6) :=
sorry

end digit_in_452nd_place_l35_35859


namespace exists_nat_with_sum_property_l35_35593

-- Definition of the sum of the digits of a number
def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem statement
theorem exists_nat_with_sum_property :
  ∃ n : ℕ, digitSum (n + 18) = digitSum n - 18 :=
by
  let n := 982
  have h1 : digitSum 982 = 19 := by sorry
  have h2 : digitSum 1000 = 1 := by sorry
  use n
  rw [show n + 18 = 1000, by norm_num]
  rw [h1, h2]
  norm_num
  sorry

end exists_nat_with_sum_property_l35_35593


namespace shepherd_initial_sheep_l35_35146

def sheep_pass_gate (sheep : ℕ) : ℕ :=
  sheep / 2 + 1

noncomputable def shepherd_sheep (initial_sheep : ℕ) : ℕ :=
  (sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate) initial_sheep

theorem shepherd_initial_sheep (initial_sheep : ℕ) (h : shepherd_sheep initial_sheep = 2) :
  initial_sheep = 2 :=
sorry

end shepherd_initial_sheep_l35_35146


namespace remaining_black_cards_l35_35118

-- Define the conditions of the problem
def total_cards : ℕ := 52
def colors : ℕ := 2
def cards_per_color := total_cards / colors
def black_cards_taken_out : ℕ := 5
def total_black_cards : ℕ := cards_per_color

-- Prove the remaining black cards
theorem remaining_black_cards : total_black_cards - black_cards_taken_out = 21 := 
by
  -- Logic to calculate remaining black cards
  sorry

end remaining_black_cards_l35_35118


namespace tangent_line_inclination_l35_35670

def curve (x : ℝ) : ℝ := (1 / 2) * x^2 - 2

def pointP := (1, -3 / 2 : ℝ)

def derivative_of_curve (x : ℝ) : ℝ := x

def slope_at_P : ℝ := derivative_of_curve 1

def angle_of_inclination (k : ℝ) : ℝ := Real.arctan k * 180 / Real.pi

theorem tangent_line_inclination :
  angle_of_inclination slope_at_P = 45 :=
by
  -- proof goes here
  sorry

end tangent_line_inclination_l35_35670


namespace intervals_of_monotonicity_range_of_values_for_a_l35_35272

noncomputable theory
open Classical
open Real

def f₁ (b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + b * x
def f₁' (b : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + b

def f₂ (a b x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x 

theorem intervals_of_monotonicity (b : ℝ) : 
  (b ≥ 1 → (∀ x, (f₁' b x) ≥ 0)) ∧ 
  (b < 1 → (∀ x, (x < -1 - Real.sqrt(1 - b) ∨ x > -1 + Real.sqrt(1 - b)) ↔ (f₁' b x > 0) ∧ (-1 - Real.sqrt(1 - b) < x ∧ x < -1 + Real.sqrt(1 - b) ↔ f₁' b x < 0)))
  := sorry

theorem range_of_values_for_a (a : ℝ) (h₁ : f₂ a (-a) 1 = 1/3) (h₂ : ∀ x ∈ Ioo (0:ℝ) (1/2:ℝ), (x^2 + 2 * a * x - a ≠ 0)) : 
  a ∈ (-∞, 0] := sorry

end intervals_of_monotonicity_range_of_values_for_a_l35_35272


namespace bob_needs_additional_weeks_l35_35934

-- Definitions based on conditions
def weekly_prize : ℕ := 100
def initial_weeks_won : ℕ := 2
def total_prize_won : ℕ := initial_weeks_won * weekly_prize
def puppy_cost : ℕ := 1000
def additional_weeks_needed : ℕ := (puppy_cost - total_prize_won) / weekly_prize

-- Statement of the theorem
theorem bob_needs_additional_weeks : additional_weeks_needed = 8 := by
  -- Proof here
  sorry

end bob_needs_additional_weeks_l35_35934


namespace perfect_cube_base_l35_35220

theorem perfect_cube_base (b : ℕ) (n : ℕ) (h1 : b ≥ 9)
  (h2 : ∃ k, (toReal (((∑ (i : ℕ) in range (n-1), 10^(i))) * 10^(n-1) * 7 + 8 * 10^(n-1) + ∑ (i : ℕ) in range n, 10^(i)) / 3) = k^3) :
  b = 10 :=
sorry

end perfect_cube_base_l35_35220


namespace A_wins_if_N_is_perfect_square_l35_35920

noncomputable def player_A_can_always_win (N : ℕ) : Prop :=
  ∀ (B_moves : ℕ → ℕ), ∃ (A_moves : ℕ → ℕ), A_moves 0 = N ∧
  (∀ n, B_moves n = 0 ∨ (A_moves n ∣ B_moves (n + 1) ∨ B_moves (n + 1) ∣ A_moves n))

theorem A_wins_if_N_is_perfect_square :
  ∀ N : ℕ, player_A_can_always_win N ↔ ∃ n : ℕ, N = n * n := sorry

end A_wins_if_N_is_perfect_square_l35_35920


namespace hyperbola_eccentricity_l35_35282

theorem hyperbola_eccentricity
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_eq : ∀ x y : ℝ, y = - (a / b) * x → ∀ x, (4, -2) = (x, y)) :
  let e := √(1 + (b^2 / a^2)) in e = √5 :=
by sorry

end hyperbola_eccentricity_l35_35282


namespace volleyball_team_lineup_count_l35_35012

theorem volleyball_team_lineup_count (n : ℕ) (h : n = 10) :
  ∃ k : ℕ, (k = 10 * 9 * 8 * 7 * 6) :=
by {
  existsi 30240,
  rw [mul_assoc 10 9 8, mul_assoc 7 8 6, mul_comm 7 (8 * 6), ← mul_assoc 7 8 6, mul_assoc 9 (10 * 8) 7, ← h],
  exact rfl,
}

end volleyball_team_lineup_count_l35_35012


namespace sum_of_50th_row_l35_35946

-- Define triangular numbers
def T (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of numbers in the nth row
def f (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1 -- T_1 is 1 for the base case
  else 2 * f (n - 1) + n * (n + 1)

-- Prove the sum of the 50th row
theorem sum_of_50th_row : f 50 = 2^50 - 2550 := 
  sorry

end sum_of_50th_row_l35_35946


namespace seq_10_is_4_l35_35979

-- Define the sequence with given properties
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 4
  | (n + 2) => if n % 2 = 0 then 4 else 3

-- Theorem statement: The 10th term of the sequence is 4
theorem seq_10_is_4 : seq 9 = 4 :=
by sorry

end seq_10_is_4_l35_35979


namespace largest_prime_factor_of_binomial_l35_35089

theorem largest_prime_factor_of_binomial :
  ∃ p : ℕ, p.prime ∧ 10 ≤ p ∧ p < 100 ∧ (∃ k : ℕ, p ^ k ∣ (300.factorial / (150.factorial * 150.factorial))) ∧
  (∀ q : ℕ, q.prime → 10 ≤ q ∧ q < 100 → (∃ k : ℕ, q ^ k ∣ (300.factorial / (150.factorial * 150.factorial))) → q ≤ p) :=
sorry

end largest_prime_factor_of_binomial_l35_35089


namespace number_of_triangles_and_edges_l35_35908

/-- Given:
      - A square with 1004 points.
      - 4 points at the vertices of the square.
      - Any 3 points are not collinear.
      - Line segments are drawn to form triangles within the square.
    To Prove:
      - The number of triangles in the triangulated square is 2002.
      - The number of edges is 3005.
-/
theorem number_of_triangles_and_edges
  (square_has_points : ∀ (n : ℕ), n = 1004)
  (vertices_at_points : ∀ v ∈ {0, 1, 2, 3}, is_vertex v)
  (three_points_not_collinear : ∀ {p1 p2 p3 : Point}, ¬collinear p1 p2 p3)
  (lines_drawn_between_points : ∀ {p1 p2 : Point}, LineSegment p1 p2)
  : ∃ T A : ℕ, T = 2002 ∧ A = 3005 :=
  -- Proof will be provided here
sorry

end number_of_triangles_and_edges_l35_35908


namespace locus_of_centers_l35_35397

-- Statement of the problem
theorem locus_of_centers :
  ∀ (a b : ℝ),
    ((∃ r : ℝ, (a^2 + b^2 = (r + 2)^2) ∧ ((a - 1)^2 + b^2 = (3 - r)^2))) ↔ (4 * a^2 + 4 * b^2 - 25 = 0) := by
  sorry

end locus_of_centers_l35_35397


namespace area_ratio_of_triangles_l35_35798

variables {A B C D P Q R S T U V W O : Type}
variables [Square ABCD] (center O)
variables (OnAB P Q) (OnBC R S) (OnCD T U) (OnAD V W)
variables (Isosceles △APW △BRQ △CTS △DVU)
variables (Equilateral △POW △ROQ △TOS △VOU)

theorem area_ratio_of_triangles (T1 : triangle P Q O) (T2 : triangle B R Q) :
  area T1 / area T2 = 1 :=
sorry

end area_ratio_of_triangles_l35_35798


namespace closest_log2_factors_r0_l35_35495

theorem closest_log2_factors_r0 :
  let r0 := 2 ^ (2 ^ 2018)
  let d := Nat.divisors r0 |>.length
  abs <| (Real.log d / Real.log 2).ceil - 2018 < 1 :=
by
  let r0 := 2 ^ (2 ^ 2018)
  let d := Nat.divisors r0 |>.length
  have : Real.log d / Real.log 2 ≈ 2018 := sorry
  exact sorry

end closest_log2_factors_r0_l35_35495


namespace sarah_earns_five_times_more_l35_35387

-- Definitions based on conditions
def connor_hourly_wage : ℝ := 7.20
def sarah_daily_wage : ℝ := 288
def sarah_working_hours : ℕ := 8
def sarah_hourly_wage : ℝ := sarah_daily_wage / sarah_working_hours

-- Theorem statement
theorem sarah_earns_five_times_more :
  sarah_hourly_wage = 5 * connor_hourly_wage :=
by
  sorry

end sarah_earns_five_times_more_l35_35387


namespace equal_segments_BP_CP_l35_35754

variables (A B C L P H : Type) [IsAngleBisector A B C L] [IsAcuteAngleTriangle A B C] [Circumcircle A B C ω]

-- Hypotheses
variable (angle_bisector : IsAngleBisector A B C L)
variable (circumcircle : Circumcircle A B C)
variable (altitude : IsAltitude B H)
variable (intersection : ∃P, (extension B H) ∩ ω = P)
variable (angle_condition : ∠BLA = ∠BAC)

-- Theorem to prove that BP = CP given the conditions
theorem equal_segments_BP_CP  (BP CP : Segment) :
  BP = CP :=
by
  sorry

end equal_segments_BP_CP_l35_35754


namespace smallest_two_digit_number_product_12_l35_35483

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l35_35483


namespace sum_of_special_numbers_l35_35436

theorem sum_of_special_numbers :
  let a := 2 in  -- since 2 is the smallest positive integer with exactly two positive divisors
  let b := 49 in -- since 49 is the largest integer less than 50 with exactly three positive divisors (7^2)
  a + b = 51 :=
by
  let a := 2
  let b := 49
  show a + b = 51
  sorry

end sum_of_special_numbers_l35_35436


namespace arithmetic_sequence_problem_l35_35031

variable {α : Type} [AddCommGroup α] [MulAction ℕ α] [HasSmul ℤ α] [HasSmul ℝ α]

theorem arithmetic_sequence_problem 
  (b : ℕ → ℝ)
  (h1 : ∑ i in finset.range 50 + 1, b (i + 1) = 50)
  (h2 : ∑ i in finset.range 50 + 1, b (i + 51) = 150) :
  (b 2 - b 1) = 1 / 25 :=
sorry

end arithmetic_sequence_problem_l35_35031


namespace log_base_4_of_4096_l35_35600

theorem log_base_4_of_4096 : Real.log 4096 / Real.log 4 = 6 := sorry

end log_base_4_of_4096_l35_35600


namespace sum_of_repeating_decimals_l35_35605

theorem sum_of_repeating_decimals : (0.6666.repeating + 0.4444.repeating : ℝ) = (10 / 9 : ℝ) :=
by
  sorry

end sum_of_repeating_decimals_l35_35605


namespace solve_quadratic_1_solve_quadratic_2_l35_35027

theorem solve_quadratic_1 : ∀ x : ℝ, x^2 + 4 * x - 1 = 0 ↔ x = -2 + Real.sqrt(5) ∨ x = -2 - Real.sqrt(5) := by
  sorry

theorem solve_quadratic_2 : ∀ x : ℝ, (x - 2)^2 - 3 * x * (x - 2) = 0 ↔ x = 2 ∨ x = -1 := by
  sorry

end solve_quadratic_1_solve_quadratic_2_l35_35027


namespace compute_expression_l35_35191

theorem compute_expression : 
  (∏ i in finset.range (23+1) ∧ (1 + 21/i)) / (∏ i in finset.range (21+1) ∧ (1 + 23/i)) = 506 := 
sorry

end compute_expression_l35_35191


namespace minimize_at_five_halves_five_sixths_l35_35228

noncomputable def minimize_expression (x y : ℝ) : ℝ :=
  (y - 1)^2 + (x + y - 3)^2 + (2 * x + y - 6)^2

theorem minimize_at_five_halves_five_sixths (x y : ℝ) :
  minimize_expression x y = 1 / 6 ↔ (x = 5 / 2 ∧ y = 5 / 6) :=
sorry

end minimize_at_five_halves_five_sixths_l35_35228


namespace problem1_problem2_l35_35638

-- First proof problem
theorem problem1 (a b : ℝ) : a^4 + 6 * a^2 * b^2 + b^4 ≥ 4 * a * b * (a^2 + b^2) :=
by sorry

-- Second proof problem
theorem problem2 (a b : ℝ) : ∃ (x : ℝ), 
  (∀ (x : ℝ), |2 * x - a^4 + (1 - 6 * a^2 * b^2 - b^4)| + 2 * |x - (2 * a^3 * b + 2 * a * b^3 - 1)| ≥ 1) ∧
  ∃ (x : ℝ), |2 * x - a^4 + (1 - 6 * a^2 * b^2 - b^4)| + 2 * |x - (2 * a^3 * b + 2 * a * b^3 - 1)| = 1 :=
by sorry

end problem1_problem2_l35_35638


namespace cost_of_lamp_and_flashlight_max_desk_lamps_l35_35793

-- Part 1: Cost of purchasing one desk lamp and one flashlight
theorem cost_of_lamp_and_flashlight (x : ℕ) (desk_lamp_cost flashlight_cost : ℕ) 
        (hx : desk_lamp_cost = x + 20)
        (hdesk : 400 = x / 2 * desk_lamp_cost)
        (hflash : 160 = x * flashlight_cost)
        (hnum : desk_lamp_cost = 2 * flashlight_cost) : 
        desk_lamp_cost = 25 ∧ flashlight_cost = 5 :=
sorry

-- Part 2: Maximum number of desk lamps Rongqing Company can purchase
theorem max_desk_lamps (a : ℕ) (desk_lamp_cost flashlight_cost : ℕ)
        (hc1 : desk_lamp_cost = 25)
        (hc2 : flashlight_cost = 5)
        (free_flashlight : ℕ := a) (required_flashlight : ℕ := 2 * a + 8) 
        (total_cost : ℕ := desk_lamp_cost * a + flashlight_cost * required_flashlight)
        (hcost : total_cost ≤ 670) :
        a ≤ 21 :=
sorry

end cost_of_lamp_and_flashlight_max_desk_lamps_l35_35793


namespace find_units_digit_l35_35431

theorem find_units_digit (A : ℕ) (h : 10 * A + 2 = 20 + A + 9) : A = 3 :=
by
  sorry

end find_units_digit_l35_35431


namespace probability_of_selecting_cubes_l35_35947

theorem probability_of_selecting_cubes :
  let total_cubes := 27;
  let two_painted_faces := 4;
  let no_painted_faces := 8;
  (choose total_cubes 2) > 0 ->
  (two_painted_faces * no_painted_faces) / (choose total_cubes 2) = (32 : ℚ) / 351 := 
by
  sorry

end probability_of_selecting_cubes_l35_35947


namespace find_p_x_l35_35406

noncomputable def p (x : ℝ) := - (12 / 5) * (x^2) + (48 / 5)

-- Definitions required for conditions
def is_quadratic (f : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
def p_of_neg3 : Prop := p (-3) = 12
def asymptotes : Prop := ∀ x, (x = -2 ∨ x = 2) → ¬ ∃ y, 1 / p x = y

theorem find_p_x : is_quadratic p ∧ p_of_neg3 ∧ asymptotes → p = (λ x, - (12 / 5) * (x^2) + (48 / 5)) :=
by
  sorry

end find_p_x_l35_35406


namespace arithmetic_sequence_sum_l35_35662

variable (a : ℕ → ℚ)
variable (m : ℕ)

-- Conditions
axiom a1 : a 1 = 1
axiom a_m : a m = 2
axiom m_pos : m ≥ 3
axiom sum_reciprocal : (∑ i in Finset.range (m - 1), 1 / (a i * a (i + 1))) = 3

-- Proof statement
theorem arithmetic_sequence_sum : (∑ i in Finset.range m, a i) = 21 / 2 := sorry

end arithmetic_sequence_sum_l35_35662


namespace total_cost_price_correct_l35_35549

def SP1 : ℝ := 120
def SP2 : ℝ := 150
def SP3 : ℝ := 200
def profit1 : ℝ := 0.20
def profit2 : ℝ := 0.25
def profit3 : ℝ := 0.10

def CP1 := SP1 / (1 + profit1)
def CP2 := SP2 / (1 + profit2)
def CP3 := SP3 / (1 + profit3)

def total_cost_price := CP1 + CP2 + CP3

theorem total_cost_price_correct : total_cost_price = 401.82 :=
by sorry

end total_cost_price_correct_l35_35549


namespace interest_rate_is_five_percent_l35_35565

variables (P : ℝ) (R : ℝ) (n : ℝ)

def compound_interest (P R : ℝ) (n : ℝ) : ℝ :=
  P * (1 + R / 100) ^ n

theorem interest_rate_is_five_percent
  (h1 : compound_interest P R 2 = 17640)
  (h2 : compound_interest P R 3 = 18522) :
  R = 5 :=
by
  sorry

end interest_rate_is_five_percent_l35_35565


namespace proof_problem_l35_35030

-- Given Conditions
variables (Students : Type) [Fintype Students]
variables (Teams : Type) [Fintype Teams]

-- A team is a set of 4 members
def team (t : Teams) : Finset Students := sorry

-- Condition (i): Any 2 different teams have exactly 2 members in common.
def condition_i : Prop :=
  ∀ (t1 t2 : Teams), t1 ≠ t2 → (team t1 ∩ team t2).card = 2

-- Condition (ii): Each team has exactly 4 members.
def condition_ii : Prop :=
  ∀ (t : Teams), (team t).card = 4

-- Condition (iii): For any 2 students, there is a team that neither of them is part of.
def condition_iii : Prop :=
  ∀ (s1 s2 : Students), ∃ t : Teams, s1 ∉ team t ∧ s2 ∉ team t

-- The theorem to be proved
theorem proof_problem 
  (h1 : condition_i)
  (h2 : condition_ii)
  (h3 : condition_iii) :
  (∀ (s1 s2 : Students), {t : Teams | s1 ∈ team t ∧ s2 ∈ team t}.toFinset.card ≤ 3) ∧
  ∃ (n : ℕ), ∀ (t : Finset Teams), t.card ≤ n :=
begin
  sorry
end

end proof_problem_l35_35030


namespace vector_sum_zero_l35_35755

variables {A B C O : Type}
variables [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup O]
variables [NormedSpace ℝ A] [NormedSpace ℝ B] [NormedSpace ℝ C] [NormedSpace ℝ O]

noncomputable def area (P Q R : O) : ℝ := sorry -- Assuming we have an area function for triangles

variables (P Q R : O)

def α (A B C O : Type) [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup O] 
  [NormedSpace ℝ A] [NormedSpace ℝ B] [NormedSpace ℝ C] [NormedSpace ℝ O] 
  (α : ℝ) := (area O B C) / (area A B C)

def β (A B C O : Type) [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup O] 
  [NormedSpace ℝ A] [NormedSpace ℝ B] [NormedSpace ℝ C] [NormedSpace ℝ O] 
  (β : ℝ) := (area O C A) / (area A B C)

def γ (A B C O : Type) [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup O] 
  [NormedSpace ℝ A] [NormedSpace ℝ B] [NormedSpace ℝ C] [NormedSpace ℝ O] 
  (γ : ℝ) := (area O A B) / (area A B C)

theorem vector_sum_zero {A B C O : Type} [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup O] 
  [NormedSpace ℝ A] [NormedSpace ℝ B] [NormedSpace ℝ C] [NormedSpace ℝ O]
  (v₁ : A) (v₂ : B) (v₃ : C) (v₄ : O)
  (α β γ : ℝ): 
  α v₁ + β v₂ + γ v₃ = 0 ↔ α = (area O B C) / (area A B C) ∧ β = (area O C A) / (area A B C) ∧ γ = (area O A B) / (area A B C) := 
sorry

end vector_sum_zero_l35_35755


namespace decimal_representation_l35_35607

theorem decimal_representation :
  (13 : ℝ) / (2 * 5^8) = 0.00001664 := 
  sorry

end decimal_representation_l35_35607


namespace hours_per_trainer_l35_35409

-- Define the conditions from part (a)
def number_of_dolphins : ℕ := 4
def hours_per_dolphin : ℕ := 3
def number_of_trainers : ℕ := 2

-- Define the theorem we want to prove using the answer from part (b)
theorem hours_per_trainer : (number_of_dolphins * hours_per_dolphin) / number_of_trainers = 6 :=
by
  -- Proof goes here
  sorry

end hours_per_trainer_l35_35409


namespace binomial_trailing_zeros_l35_35102

theorem binomial_trailing_zeros (n k : ℕ) (h : n = 125 ∧ k = 64) :
  ∃ z : ℕ, trailing_zeros (nat.choose 125 64) = 0 :=
by
  -- Placeholder for actual proof steps
  sorry

end binomial_trailing_zeros_l35_35102


namespace det_P_eq_zero_l35_35748

def v : ℝ^3 := ![3, -2, 4]
def P : Matrix (Fin 3) (Fin 3) ℝ := (1 / (3^2 + (-2)^2 + 4^2)) • (v ⬝ (vᵀ))

theorem det_P_eq_zero : det P = 0 := by
  sorry

end det_P_eq_zero_l35_35748


namespace value_of_4_op_3_l35_35994

-- Definitions based on conditions
def op (m n x y : ℝ) : ℝ := ((n ^ 3) / x - m ^ 2) / y

-- Specific values
def m_val : ℝ := 4
def n_val : ℝ := 3
def x_val : ℝ := 2
def y_val : ℝ := 5

-- Proof statement
theorem value_of_4_op_3 : x_val > 0 → y_val ≠ 0 → op m_val n_val x_val y_val = -0.5 :=
by
  intros h_x_pos h_y_ne_zero
  sorry

end value_of_4_op_3_l35_35994


namespace total_animal_sightings_l35_35013

theorem total_animal_sightings 
  (animal_sightings_january animal_sightings_february animal_sightings_march animal_sightings_april animal_sightings_may animal_sightings_june: ℕ)
  (families_january families_february families_march families_april families_may families_june: ℕ)
  (h1: families_january = 100)
  (h2: animal_sightings_january = 26)
  (h3: families_february = families_january + (families_january * 50 / 100))
  (h4: animal_sightings_february = animal_sightings_january * 3)
  (h5: families_march = families_february - (families_february * 20 / 100))
  (h6: animal_sightings_march = animal_sightings_february / 2)
  (h7: families_april = families_march + (families_march * 70 / 100))
  (h8: animal_sightings_april = animal_sightings_march + (animal_sightings_march * 40 / 100))
  (h9: families_may = families_april)
  (h10: animal_sightings_may = animal_sightings_april - (animal_sightings_april * 25 / 100))
  (h11: families_june = families_may + (families_may * 30 / 100))
  (h12: animal_sightings_june = animal_sightings_may) :
  animal_sightings_january + animal_sightings_february + animal_sightings_march + animal_sightings_april + animal_sightings_may + animal_sightings_june = 280 :=
begin
  sorry
end

end total_animal_sightings_l35_35013


namespace rate_per_kg_grapes_is_70_l35_35574

-- Let G be the rate per kg for the grapes
def rate_per_kg_grapes (G : ℕ) := G

-- Bruce purchased 8 kg of grapes at rate G per kg
def grapes_cost (G : ℕ) := 8 * G

-- Bruce purchased 11 kg of mangoes at the rate of 55 per kg
def mangoes_cost := 11 * 55

-- Bruce paid a total of 1165 to the shopkeeper
def total_paid := 1165

-- The problem: Prove that the rate per kg for the grapes is 70
theorem rate_per_kg_grapes_is_70 : rate_per_kg_grapes 70 = 70 ∧ grapes_cost 70 + mangoes_cost = total_paid := by
  sorry

end rate_per_kg_grapes_is_70_l35_35574


namespace escalator_steps_l35_35147

theorem escalator_steps (x y : ℕ) (h : ∀ steps, (steps = 55 → (x + y) * 55 / y = (x + 2 * y) * 60 / (2 * y)) ∧ (steps = 60 → (x + 2 * y) * 60 / (2 * y) = (x + y) * 55 / y)) :
  let E := 66 in
  E = 66 := 
by
  intro steps
  let y := 5*x
  have h : (x + y) * 55 / y = 66 := sorry
  exact h

end escalator_steps_l35_35147


namespace parallelepiped_properties_l35_35811

-- Definitions of the conditions
structure Parallelepiped :=
  (A B C D A1 B1 C1 D1 : Point)
  (perpendicular_edge : is_perpendicular_to A1 A (plane_of A B C D))
  (sphere : Sphere)
  (tangent_edges : sphere_tangent_to_edges sphere B B1 ∧ sphere_tangent_to_edges sphere B1 C1 ∧ sphere_tangent_to_edges sphere C1 C ∧ sphere_tangent_to_edges sphere C B ∧ sphere_tangent_to_edges sphere C1 D1)
  (tangent_point : Point)
  (tangent_property_1 : tangent_to_edge sphere C1 D1 tangent_point)
  (distance_C1K : distance C1 tangent_point = 9)
  (distance_KD1 : distance tangent_point D1 = 4)
  (AD_tangent : sphere_tangent_to_edge sphere A D)

-- Prove the properties
theorem parallelepiped_properties (p : Parallelepiped) :
  (length p.A1 p.A = 18) ∧ (volume_of_parallelepiped p = 3888) ∧ (sphere_radius p.sphere = 3 * sqrt 13) := by
  sorry

end parallelepiped_properties_l35_35811


namespace determine_omega_l35_35819

theorem determine_omega :
  ∃ ω : ℝ, ω > 0 ∧ 
  (∀ x : ℝ, g(x) = sin (ω * x - ω * π / 12)) ∧
  (∀ x : ℝ, x ∈ Icc (π / 6) (π / 3) → strict_mono (g x)) ∧
  (∀ x : ℝ, x ∈ Icc (π / 3) (π / 2) → strict_anti (g x))
  → ω = 2 :=
by
  sorry

end determine_omega_l35_35819


namespace S_value_l35_35955

noncomputable def f (x : ℝ) : ℝ := (4^(x+1)) / (4^x + 2)

def S : ℝ := 
  f (1/10) + f (2/10) + f (3/10) + f (4/10) + f (5/10) +
  f (6/10) + f (7/10) + f (8/10) + f (9/10)

theorem S_value : S = 18 := 
by 
  sorry

end S_value_l35_35955


namespace count_ordered_pairs_l35_35992

theorem count_ordered_pairs : 
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 1 < a + b ∧ a + b < 22) →
  ({p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ 1 < p.1 + p.2 ∧ p.1 + p.2 < 22}.to_finset.card = 210) :=
sorry

end count_ordered_pairs_l35_35992


namespace relation_confidence_l35_35124

-- Definitions of values from the question
def number_of_students := 100
def students_like_outdoor_sports := 60
def students_with_excellent_test_scores := 75
def students_like_outdoor_sports_non_excellent := 10
def students_non_excellent_test_scores := number_of_students - students_with_excellent_test_scores
def students_like_outdoor_sports_excellent := students_like_outdoor_sports - students_like_outdoor_sports_non_excellent
def students_do_not_like_outdoor_sports_excellent := students_with_excellent_test_scores - students_like_outdoor_sports_excellent
def students_do_not_like_outdoor_sports_non_excellent := students_non_excellent_test_scores - students_like_outdoor_sports_non_excellent
def total_students_like_outdoor_sports := students_like_outdoor_sports
def total_students_do_not_like_outdoor_sports := number_of_students - total_students_like_outdoor_sports

-- Completed contingency table values
def a := students_like_outdoor_sports_excellent
def b := students_like_outdoor_sports_non_excellent
def c := students_do_not_like_outdoor_sports_excellent
def d := students_do_not_like_outdoor_sports_non_excellent
def n := number_of_students

-- Formula for K^2
def K_squared (a b c d n : ℝ) : ℝ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- k-value for 95% confidence level
def k_value := 3.841

-- Define the core theorem statement
theorem relation_confidence :
  K_squared a b c d n > k_value :=
by
  -- exact computation is skipped, it involves proving the expression around 5.56
  have K_squared (a b c d n) ≈ 5.56, sorry
  exact sorry

end relation_confidence_l35_35124


namespace distance_from_apex_to_larger_cross_section_l35_35066

noncomputable def area1 : ℝ := 324 * Real.sqrt 2
noncomputable def area2 : ℝ := 648 * Real.sqrt 2
def distance_between_planes : ℝ := 12

theorem distance_from_apex_to_larger_cross_section
  (area1 area2 : ℝ)
  (distance_between_planes : ℝ)
  (h_area1 : area1 = 324 * Real.sqrt 2)
  (h_area2 : area2 = 648 * Real.sqrt 2)
  (h_distance : distance_between_planes = 12) :
  ∃ (H : ℝ), H = 24 + 12 * Real.sqrt 2 :=
by sorry

end distance_from_apex_to_larger_cross_section_l35_35066


namespace task_completion_time_l35_35334

noncomputable def john_days := 20
noncomputable def jane_days := 10
noncomputable def days_jane_indisposed_before_completion := 5

theorem task_completion_time :
  let john_rate := 1 / john_days
  let jane_rate := 1 / jane_days
  let combined_rate := john_rate + jane_rate
  let x := (1 - john_rate * days_jane_indisposed_before_completion) / combined_rate
  let total_days := x + days_jane_indisposed_before_completion in
  total_days = 10 := 
by
  let john_rate := 1 / john_days
  let jane_rate := 1 / jane_days
  let combined_rate := john_rate + jane_rate
  let x := (1 - john_rate * days_jane_indisposed_before_completion) / combined_rate
  have total_days := x + days_jane_indisposed_before_completion
  exact rfl

end task_completion_time_l35_35334


namespace G_at_8_l35_35345

noncomputable def G : ℝ → ℝ := sorry

axiom G_polynomial : ∃ p : polynomial ℝ, ∀ x, G x = p.eval x

axiom G_at_4 : G 4 = 10

axiom G_eq : ∀ (x : ℝ), (x^2 + 3 * x + 2) ≠ 0 → G(2 * x) / G(x + 2) = 4 - (16 * x + 8) / (x^2 + 3 * x + 2)

theorem G_at_8 : G 8 = 140 / 3 := sorry

end G_at_8_l35_35345


namespace categorize_numbers_l35_35608

def numbers : Set (Rat) := {-16, 0.04, 1/2, -2/3, 25, 0, -3.6, -0.3, 4/3}

def is_integer (x : Rat) : Prop := ∃ z : Int, x = z
def is_fraction (x : Rat) : Prop := ∃ (p q : Int), q ≠ 0 ∧ x = p / q
def is_negative (x : Rat) : Prop := x < 0

def integers (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_integer x}
def fractions (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_fraction x}
def negative_rationals (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_fraction x ∧ is_negative x}

theorem categorize_numbers :
  integers numbers = {-16, 25, 0} ∧
  fractions numbers = {0.04, 1/2, -2/33, -3.6, -0.3, 4/3} ∧
  negative_rationals numbers = {-16, -2/3, -3.6, -0.3} :=
  sorry

end categorize_numbers_l35_35608


namespace fridge_magnet_pairs_l35_35562

def is_valid_pair (a b : ℕ) : Prop :=
  (a + b) % 5 = 0

def valid_pairings (S : Finset (ℕ × ℕ)) : Prop :=
  (∀ (p ∈ S), ∃ (a b : ℕ), a ≠ b ∧ p = (a, b) ∧ is_valid_pair a b) ∧
  S.card = 5 ∧
  (∀ (a b : ℕ), a ≠ b → ∃ (p ∈ S), p = (a, b) ∨ p = (b, a))

theorem fridge_magnet_pairs :
  ∃ (S : Finset (ℕ × ℕ)), valid_pairings S ∧ S.card = 5 := by
  sorry

end fridge_magnet_pairs_l35_35562


namespace coefficient_x4_l35_35319

theorem coefficient_x4 (x : ℝ) : 
  let expr := (x + 1) * (x + (1 / (2 * x))) ^ 8 in
  -- (x + 1)(x + 1/(2x))^8 expansion 
  -- coeff of x^4 in expr is 7
  (coeff_of_x4 expr = 7) :=
sorry

end coefficient_x4_l35_35319


namespace tangent_line_at_1_l35_35281

noncomputable def g (x : ℝ) : ℝ := (-x^2 + 5*x - 3) * Real.exp x

def tangent_line_eq (a b : ℝ) (x : ℝ) : ℝ := a * x + b

theorem tangent_line_at_1 :
  let g' := fun x => (e x * (-2 * x + 5 - x^2 + 5 * x - 3))
  ∃ a b : ℝ, ∀ x, tangent_line_eq 4 (3 * e) x = g x - g 1 :=
  begin
    sorry
  end

end tangent_line_at_1_l35_35281


namespace arc_length_formula_l35_35815

section ArcLength

variable (n R : ℝ)

theorem arc_length_formula (h1 : 0 ≤ n) (h2 : 0 ≤ R) :
  L = (n * real.pi * R) / 180 := by
  sorry

end ArcLength

end arc_length_formula_l35_35815


namespace smallest_λ_exists_l35_35227

def satisfies_condition (a : Fin 100 → ℝ) : Prop :=
  (∑ i : Fin 100, a i ^ 2) = 100

theorem smallest_λ_exists :
  ∃ λ : ℝ, (λ = 8) ∧ (∀ (a : Fin 100 → ℝ),
    satisfies_condition a →
    (∑ i : Fin 100, (a i - a (i + 1) % 100 ) ^ 2) ≤ λ * (100 - ∑ i : Fin 100, a i)) :=
sorry

end smallest_λ_exists_l35_35227


namespace hexagon_area_minus_sectors_l35_35901

/-- A regular hexagon has a side length of 8. Congruent arcs with a radius of 4 are drawn with the center at each of the vertices, each arc covering an angle of 90 degrees. Calculate the area of the region inside the hexagon but outside these sectors. -/
theorem hexagon_area_minus_sectors (side_length : ℕ) (arc_radius : ℕ) (arc_angle : ℝ) 
  (h_side_length : side_length = 8) 
  (h_arc_radius : arc_radius = 4) 
  (h_arc_angle : arc_angle = 90) :
  let hex_area := 6 * (sqrt 3 / 4 * side_length ^ 2),
      sector_area := 6 * (arc_angle / 360 * π * arc_radius ^ 2)
  in hex_area - sector_area = 96 * sqrt 3 - 24 * π := 
by
  /- This is where the actual proof would be constructed.
  Since we are not required to provide the proof steps,
  this part will be omitted. -/
  sorry

end hexagon_area_minus_sectors_l35_35901


namespace pickle_slice_ratio_l35_35386

theorem pickle_slice_ratio (S T R : ℕ) (h1 : S = 15) (h2 : R = 24) (h3 : 0.8 * T = R) : T / S = 2 :=
by
  sorry

end pickle_slice_ratio_l35_35386


namespace intersection_of_function_and_inverse_l35_35407

theorem intersection_of_function_and_inverse (c k : ℤ) (f : ℤ → ℤ)
  (hf : ∀ x:ℤ, f x = 4 * x + c) 
  (hf_inv : ∀ y:ℤ, (∃ x:ℤ, f x = y) → (∃ x:ℤ, f y = x))
  (h_intersection : ∀ k:ℤ, f 2 = k ∧ f k = 2 ) 
  : k = 2 :=
sorry

end intersection_of_function_and_inverse_l35_35407


namespace crop_planting_count_l35_35534

/-- A farm's field is a 3x3 grid of 9 smaller square sections. There are three types of crops: 
    tomatoes, cucumbers, and carrots. 
    The farmer does not want tomatoes to be adjacent to cucumbers, 
    and carrots should not be adjacent to tomatoes.
    Prove that there are 175 ways to plant these crops in the grid sections. -/
theorem crop_planting_count : 
  let grid := (fin 3) × (fin 3),
      crops := {tomato, cucumber, carrot : Type}
  in
  let planting := grid → crops in
  let adj (p q : planting) := ∃ i j, 
     (p (i, j) = tomato ∧ (p (i + 1, j) = cucumber ∨ p (i - 1, j) = cucumber ∨ 
                           p (i, j + 1) = cucumber ∨ p (i, j - 1) = cucumber)) ∨
     (p (i, j) = tomato ∧ (p (i + 1, j) = carrot ∨ p (i - 1, j) = carrot ∨ 
                           p (i, j + 1) = carrot ∨ p (i, j - 1) = carrot) ∨
  ∀ p : planting, ¬ adj p q → count_plants p = 175 :=
sorry

end crop_planting_count_l35_35534


namespace imaginary_part_of_z_l35_35669

open Complex

theorem imaginary_part_of_z : 
  let z : ℂ := (2 + I) / ((1 + I) * (1 + I))
  in z.im = -1 :=
by
  let z : ℂ := (2 + I) / ((1 + I) * (1 + I))
  show z.im = -1
  sorry

end imaginary_part_of_z_l35_35669


namespace pq_false_implies_m_range_l35_35587

def p : Prop := ∀ x : ℝ, abs x + x ≥ 0

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0

theorem pq_false_implies_m_range (m : ℝ) :
  (¬ (p ∧ q m)) → -2 < m ∧ m < 2 :=
by
  sorry

end pq_false_implies_m_range_l35_35587


namespace train_speed_is_correct_l35_35559

def length_of_train : ℝ := 200.016 -- meters
def time_to_cross_pole : ℝ := 6 -- seconds

def speed_of_train (length: ℝ) (time: ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

theorem train_speed_is_correct :
  speed_of_train length_of_train time_to_cross_pole = 120.0096 := 
sorry

end train_speed_is_correct_l35_35559


namespace sixth_graders_more_than_seventh_l35_35568

theorem sixth_graders_more_than_seventh (c_pencil : ℕ) (h_cents : c_pencil > 0)
    (h_cond : ∀ n : ℕ, n * c_pencil = 221 ∨ n * c_pencil = 286)
    (h_sixth_graders : 35 > 0) :
    ∃ n6 n7 : ℕ, n6 > n7 ∧ n6 - n7 = 5 :=
by
  sorry

end sixth_graders_more_than_seventh_l35_35568


namespace eggs_collection_l35_35180

theorem eggs_collection (b : ℕ) (c : ℕ) (t : ℕ) 
  (h₁ : b = 6) 
  (h₂ : c = 3 * b) 
  (h₃ : t = b - 4) : 
  b + c + t = 26 :=
by
  sorry

end eggs_collection_l35_35180


namespace infinite_integer_pairs_exist_l35_35025

theorem infinite_integer_pairs_exist (k : ℤ) : 
  ∃ᶠ p : ℤ × ℤ in filter.at_top, 
    let (m, n) := p in 
    (m ≠ 0 ∧ n ≠ 0) ∧ ((m + 1) / n + (n + 1) / m = k) :=
begin
  sorry
end

end infinite_integer_pairs_exist_l35_35025


namespace largest_2_digit_prime_factor_of_binom_l35_35076

open Nat

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

/-- Definition of the problem conditions -/
def problem_conditions : Prop :=
  let n := binom 300 150
  ∃ p : ℕ, Prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p ≤ 75 ∨ 3 * p < 300) ∧ p = 97

/-- Statement of the proof problem -/
theorem largest_2_digit_prime_factor_of_binom : problem_conditions := 
  sorry

end largest_2_digit_prime_factor_of_binom_l35_35076


namespace remainder_when_sum_divided_l35_35951

theorem remainder_when_sum_divided (p q : ℕ) (m n : ℕ) (hp : p = 80 * m + 75) (hq : q = 120 * n + 115) :
  (p + q) % 40 = 30 := 
by sorry

end remainder_when_sum_divided_l35_35951


namespace cranberries_left_l35_35169

theorem cranberries_left (total_cranberries : ℕ) (harvested_percent: ℝ) (cranberries_eaten : ℕ) 
  (h1 : total_cranberries = 60000) 
  (h2 : harvested_percent = 0.40) 
  (h3 : cranberries_eaten = 20000) : 
  total_cranberries - (harvested_percent * total_cranberries).to_nat - cranberries_eaten = 16000 := 
by 
  sorry

end cranberries_left_l35_35169


namespace find_primes_l35_35611

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

/- Define the three conditions -/
def condition1 (p q r : ℕ) : Prop := divides p (1 + q ^ r)
def condition2 (p q r : ℕ) : Prop := divides q (1 + r ^ p)
def condition3 (p q r : ℕ) : Prop := divides r (1 + p ^ q)

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ condition1 p q r ∧ condition2 p q r ∧ condition3 p q r

theorem find_primes (p q r : ℕ) :
  satisfies_conditions p q r ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) :=
by
  sorry

end find_primes_l35_35611


namespace find_big_bonsai_cost_l35_35771

-- Given definitions based on conditions
def small_bonsai_cost : ℕ := 30
def num_small_bonsai_sold : ℕ := 3
def num_big_bonsai_sold : ℕ := 5
def total_earnings : ℕ := 190

-- Define the function to calculate total earnings from bonsai sales
def calculate_total_earnings (big_bonsai_cost: ℕ) : ℕ :=
  (num_small_bonsai_sold * small_bonsai_cost) + (num_big_bonsai_sold * big_bonsai_cost)

-- The theorem state
theorem find_big_bonsai_cost (B : ℕ) : calculate_total_earnings B = total_earnings → B = 20 :=
by
  sorry

end find_big_bonsai_cost_l35_35771


namespace sequence_divisibility_l35_35924

theorem sequence_divisibility (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, 2^n = ∑ d in (Finset.filter (λ d, d ∣ n) (Finset.range (n+1))), a d) : 
  ∀ n : ℕ, n ∣ a n :=
begin
  sorry
end

end sequence_divisibility_l35_35924


namespace Martha_cards_l35_35369

theorem Martha_cards :
  let initial_cards := 76.0
  let given_away_cards := 3.0
  initial_cards - given_away_cards = 73.0 :=
by 
  let initial_cards := 76.0
  let given_away_cards := 3.0
  have h : initial_cards - given_away_cards = 73.0 := by sorry
  exact h

end Martha_cards_l35_35369


namespace min_value_frac_sum_l35_35701

theorem min_value_frac_sum (x : ℝ) (h : 1 ≤ x ∧ x < 2) :
  ∃ y, y = 2 ∧ (∀ z, (1 ≤ z ∧ z < 2) → (frac_sum z z) ≥ y) 

end min_value_frac_sum_l35_35701


namespace buses_dispatched_theorem_l35_35883

-- Define the conditions and parameters
def buses_dispatched (buses: ℕ) (hours: ℕ) : ℕ :=
  buses * hours

-- Define the specific problem
noncomputable def buses_from_6am_to_4pm : ℕ :=
  let buses_per_hour := 5 / 2
  let hours         := 16 - 6
  buses_dispatched (buses_per_hour : ℕ) hours

-- State the theorem that needs to be proven
theorem buses_dispatched_theorem : buses_from_6am_to_4pm = 25 := 
by {
  -- This 'sorry' is a placeholder for the actual proof.
  sorry
}

end buses_dispatched_theorem_l35_35883


namespace derivative_of_exp_neg2x_l35_35708

variable (x : ℝ)

def f (x : ℝ) : ℝ := Real.exp (-2 * x)

theorem derivative_of_exp_neg2x :
  (Real.deriv f) x = -2 * Real.exp (-2 * x) :=
sorry

end derivative_of_exp_neg2x_l35_35708


namespace find_a_b_l35_35715

def system_solution (a b x y : ℝ) : Prop :=
  (b * x - 3 * y = 2) ∧ (a * x + y = 2)

theorem find_a_b : 
  ∃ a b : ℝ, system_solution a b 4 2 ∧ a = 0 ∧ b = 2 :=
begin
  sorry
end

end find_a_b_l35_35715


namespace max_tulips_l35_35442

theorem max_tulips (r y : ℕ) (h₁ : r + y = 2 * (y : ℕ) + 1) (h₂ : |r - y| = 1) (h₃ : 50 * y + 31 * r ≤ 600) :
    r + y = 15 :=
sorry

end max_tulips_l35_35442


namespace cube_root_of_one_eighth_l35_35809

theorem cube_root_of_one_eighth : 
  (∃ y : ℚ, y^3 = 1 / 8 ∧ ∀ (z : ℚ), z^3 = 1 / 8 → z = y) :=
begin
  use 1 / 2,
  split,
  { norm_num },
  { intros z hz,
    exact by linarith [z^3, hz] }
end

end cube_root_of_one_eighth_l35_35809


namespace mean_score_of_all_students_l35_35003

-- Define the conditions as given in the problem
variables (M A : ℝ) (m a : ℝ)
  (hM : M = 90)
  (hA : A = 75)
  (hRatio : m / a = 2 / 5)

-- State the theorem which proves that the mean score of all students is 79
theorem mean_score_of_all_students (hM : M = 90) (hA : A = 75) (hRatio : m / a = 2 / 5) : 
  (36 * a + 75 * a) / ((2 / 5) * a + a) = 79 := 
by
  sorry -- Proof is omitted

end mean_score_of_all_students_l35_35003


namespace props_false_l35_35645

-- Define Proposition 1
def Prop1 : Prop := ∀ (P : Type) [metric_space P] (a b : P),
  (a ≠ b ∧ ∃ c, dist c a = dist c b) → is_plane {c : P | dist c a = dist c b}

-- Define Proposition 2
def Prop2 : Prop := ∀ (P : Type) [metric_space P] (a b c : P),
  is_plane a ∧ is_plane b ∧ is_plane c →
  ∃ d, dist d a = dist d b ∧ dist d c

-- The theorem stating that both Proposition 1 and Proposition 2 are false
theorem props_false : ¬Prop1 ∧ ¬Prop2 :=
by 
  sorry

end props_false_l35_35645


namespace always_positive_sum_l35_35817

def f : ℝ → ℝ := sorry  -- assuming f(x) is provided elsewhere

theorem always_positive_sum (f : ℝ → ℝ)
    (h1 : ∀ x, f x = -f (2 - x))
    (h2 : ∀ x, x < 1 → f (x) < f (x + 1))
    (x1 x2 : ℝ)
    (h3 : x1 + x2 > 2)
    (h4 : (x1 - 1) * (x2 - 1) < 0) :
  f x1 + f x2 > 0 :=
by {
  sorry
}

end always_positive_sum_l35_35817


namespace selling_price_of_cycle_l35_35107

theorem selling_price_of_cycle (cp : ℝ) (loss_percentage : ℝ) (sp : ℝ) : 
  cp = 1400 → loss_percentage = 20 → sp = cp - (loss_percentage / 100) * cp → sp = 1120 :=
by 
  intro h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end selling_price_of_cycle_l35_35107


namespace triangle_minimize_circumradius_correct_l35_35915

noncomputable def triangle_minimize_circumradius (a b : ℕ) (c : ℝ) : Prop :=
  a = 888 ∧ b = 925 ∧ c > 0 → 
  let R := (888 * 925 * c) / (4 * real.sqrt ((1813 + c) / 2 * ((1813 + c) / 2 - 888) * ((1813 + c) / 2 - 925) * ((1813 + c) / 2 - c))) in
  c = 259 ∧ ∀ x > 0, let R' := (888 * 925 * x) / (4 * real.sqrt ((1813 + x) / 2 * ((1813 + x) / 2 - 888) * ((1813 + x) / 2 - 925) * ((1813 + x) / 2 - x))) in
  R ≤ R'

theorem triangle_minimize_circumradius_correct : triangle_minimize_circumradius 888 925 259 :=
by sorry

end triangle_minimize_circumradius_correct_l35_35915


namespace smallest_two_digit_number_product_12_l35_35480

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l35_35480


namespace butterfly_probability_l35_35583

-- Define the vertices of the cube
inductive Vertex
| A | B | C | D | E | F | G | H

open Vertex

-- Define the edges of the cube
def edges : Vertex → List Vertex
| A => [B, D, E]
| B => [A, C, F]
| C => [B, D, G]
| D => [A, C, H]
| E => [A, F, H]
| F => [B, E, G]
| G => [C, F, H]
| H => [D, E, G]

-- Define a function to simulate the butterfly's movement
noncomputable def move : Vertex → ℕ → List (Vertex × ℕ)
| v, 0 => [(v, 0)]
| v, n + 1 =>
  let nextMoves := edges v
  nextMoves.bind (λ v' => move v' n)

-- Define the probability calculation part
noncomputable def probability_of_visiting_all_vertices (n_moves : ℕ) : ℚ :=
  let total_paths := (3 ^ n_moves : ℕ)
  let valid_paths := 27 -- Based on given final solution step
  valid_paths / total_paths

-- Statement of the problem in Lean 4
theorem butterfly_probability :
  probability_of_visiting_all_vertices 11 = 27 / 177147 :=
by
  sorry

end butterfly_probability_l35_35583


namespace total_number_of_bills_received_l35_35071

open Nat

-- Definitions based on the conditions:
def total_withdrawal_amount : ℕ := 600
def bill_denomination : ℕ := 20

-- Mathematically equivalent proof problem
theorem total_number_of_bills_received : (total_withdrawal_amount / bill_denomination) = 30 := 
by
  sorry

end total_number_of_bills_received_l35_35071


namespace calculate_amount_left_l35_35371

def base_income : ℝ := 2000
def bonus_percentage : ℝ := 0.15
def public_transport_percentage : ℝ := 0.05
def rent : ℝ := 500
def utilities : ℝ := 100
def food : ℝ := 300
def miscellaneous_percentage : ℝ := 0.10
def savings_percentage : ℝ := 0.07
def investment_percentage : ℝ := 0.05
def medical_expense : ℝ := 250
def tax_percentage : ℝ := 0.15

def total_income (base_income : ℝ) (bonus_percentage : ℝ) : ℝ :=
  base_income + (bonus_percentage * base_income)

def taxes (base_income : ℝ) (tax_percentage : ℝ) : ℝ :=
  tax_percentage * base_income

def total_fixed_expenses (rent : ℝ) (utilities : ℝ) (food : ℝ) : ℝ :=
  rent + utilities + food

def public_transport_expense (total_income : ℝ) (public_transport_percentage : ℝ) : ℝ :=
  public_transport_percentage * total_income

def miscellaneous_expense (total_income : ℝ) (miscellaneous_percentage : ℝ) : ℝ :=
  miscellaneous_percentage * total_income

def variable_expenses (public_transport_expense : ℝ) (miscellaneous_expense : ℝ) : ℝ :=
  public_transport_expense + miscellaneous_expense

def savings (total_income : ℝ) (savings_percentage : ℝ) : ℝ :=
  savings_percentage * total_income

def investment (total_income : ℝ) (investment_percentage : ℝ) : ℝ :=
  investment_percentage * total_income

def total_savings_investments (savings : ℝ) (investment : ℝ) : ℝ :=
  savings + investment

def total_expenses_contributions 
  (fixed_expenses : ℝ) 
  (variable_expenses : ℝ) 
  (medical_expense : ℝ) 
  (total_savings_investments : ℝ) : ℝ :=
  fixed_expenses + variable_expenses + medical_expense + total_savings_investments

def amount_left (income_after_taxes : ℝ) (total_expenses_contributions : ℝ) : ℝ :=
  income_after_taxes - total_expenses_contributions

theorem calculate_amount_left 
  (base_income : ℝ)
  (bonus_percentage : ℝ)
  (public_transport_percentage : ℝ)
  (rent : ℝ)
  (utilities : ℝ)
  (food : ℝ)
  (miscellaneous_percentage : ℝ)
  (savings_percentage : ℝ)
  (investment_percentage : ℝ)
  (medical_expense : ℝ)
  (tax_percentage : ℝ)
  (total_income : ℝ := total_income base_income bonus_percentage)
  (taxes : ℝ := taxes base_income tax_percentage)
  (income_after_taxes : ℝ := total_income - taxes)
  (fixed_expenses : ℝ := total_fixed_expenses rent utilities food)
  (public_transport_expense : ℝ := public_transport_expense total_income public_transport_percentage)
  (miscellaneous_expense : ℝ := miscellaneous_expense total_income miscellaneous_percentage)
  (variable_expenses : ℝ := variable_expenses public_transport_expense miscellaneous_expense)
  (savings : ℝ := savings total_income savings_percentage)
  (investment : ℝ := investment total_income investment_percentage)
  (total_savings_investments : ℝ := total_savings_investments savings investment)
  (total_expenses_contributions : ℝ := total_expenses_contributions fixed_expenses variable_expenses medical_expense total_savings_investments)
  : amount_left income_after_taxes total_expenses_contributions = 229 := 
sorry

end calculate_amount_left_l35_35371


namespace least_integer_value_x_l35_35454

theorem least_integer_value_x (x : ℤ) (h : |(2 : ℤ) * x + 3| ≤ 12) : x = -7 :=
by
  sorry

end least_integer_value_x_l35_35454


namespace sin_alpha_value_l35_35693

noncomputable def alpha : ℝ := sorry  -- We will define alpha but avoid computing it concretely in Lean

theorem sin_alpha_value (h1 : cos (π - alpha) = 4 / 5) (h2 : π / 2 < alpha ∧ alpha < π) :
  sin alpha = 3 / 5 :=
sorry -- The proof will go here

end sin_alpha_value_l35_35693


namespace sum_arithmetic_sequence_l35_35668

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_arithmetic_sequence {a : ℕ → ℝ} 
  (h_arith : arithmetic_seq a)
  (h1 : a 2^2 + a 7^2 + 2 * a 2 * a 7 = 9)
  (h2 : ∀ n, a n < 0) : 
  S₁₀ = -15 :=
by
  sorry

end sum_arithmetic_sequence_l35_35668


namespace sequence_a_n_l35_35388

theorem sequence_a_n {a : ℕ → ℤ}
  (h1 : a 2 = 5)
  (h2 : a 1 = 1)
  (h3 : ∀ n ≥ 2, a (n+1) - 2 * a n + a (n-1) = 7) :
  a 17 = 905 :=
  sorry

end sequence_a_n_l35_35388


namespace total_eggs_collected_l35_35178

def benjamin_collects : Nat := 6
def carla_collects := 3 * benjamin_collects
def trisha_collects := benjamin_collects - 4

theorem total_eggs_collected :
  benjamin_collects + carla_collects + trisha_collects = 26 := by
  sorry

end total_eggs_collected_l35_35178


namespace equivalent_sets_l35_35047

-- Definitions of the condition and expected result
def condition_set : Set ℕ := { x | x - 3 < 2 }
def expected_set : Set ℕ := {0, 1, 2, 3, 4}

-- Theorem statement
theorem equivalent_sets : condition_set = expected_set := 
by
  sorry

end equivalent_sets_l35_35047


namespace largest_prime_factor_of_binomial_l35_35091

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_prime_factor_of_binomial {p : ℕ} (hp : Nat.Prime p) (hp_range : 10 ≤ p ∧ p < 100) :
  p ∣ binomial 300 150 → p = 97 :=
by
suffices ∀ q : ℕ, Nat.Prime q → 10 ≤ q ∧ q < 100 → q ∣ binomial 300 150 → q ≤ 97
from fun h => le_antisymm (this p hp hp_range h) (le_of_eq (rfl : 97 = 97))
intro q hq hq_range hq_div
sorry

end largest_prime_factor_of_binomial_l35_35091


namespace total_surface_area_l35_35419

-- Define the conditions
variables (a b c : ℝ)

-- Condition 1: sum of lengths of the twelve edges
def sum_of_edges (a b c : ℝ) : Prop := 4 * (a + b + c) = 140

-- Condition 2: distance from one corner to the farthest corner
def diagonal_distance (a b c : ℝ) : Prop := real.sqrt (a^2 + b^2 + c^2) = 21

-- Prove the surface area of the box
theorem total_surface_area (a b c : ℝ) (h1 : sum_of_edges a b c) (h2 : diagonal_distance a b c) : 2 * (a * b + b * c + c * a) = 784 :=
sorry

end total_surface_area_l35_35419


namespace min_value_of_expression_l35_35657

noncomputable def minValue (a : ℝ) : ℝ :=
  1 / (3 - 2 * a) + 2 / (a - 1)

theorem min_value_of_expression : ∀ a : ℝ, 1 < a ∧ a < 3 / 2 → (1 / (3 - 2 * a) + 2 / (a - 1)) ≥ 16 / 9 :=
by
  intro a h
  sorry

end min_value_of_expression_l35_35657


namespace number_of_shaded_cubes_l35_35133

open Nat

-- Define the conditions given in the problem
def isCorner (x y z : ℕ) : Prop :=
  (x = 0 ∨ x = 3) ∧ (y = 0 ∨ y = 3) ∧ (z = 0 ∨ z = 3)

def isCenter (x y z : ℕ) : Prop :=
  x = 1 ∧ y = 1 ∧ z = 1

def isMiddleEdge (x y z : ℕ) : Prop :=
  ((x = 0 ∨ x = 3) ∧ (y = 1 ∨ y = 2) ∧ (z = 1 ∨ z = 2)) ∨ 
  ((x = 1 ∨ x = 2) ∧ (y = 0 ∨ y = 3) ∧ (z = 1 ∨ z = 2)) ∨
  ((x = 1 ∨ x = 2) ∧ (y = 1 ∨ y = 2) ∧ (z = 0 ∨ z = 3))

def isShaded (x y z : ℕ) : Prop :=
  isCorner x y z ∨ isCenter x y z ∨ isMiddleEdge x y z

def countShadedCubes : ℕ :=
  ((List.product (List.product (List.range 4) (List.range 4)) (List.range 4))
   .countp (λ (coords : (ℕ × ℕ) × ℕ) => isShaded coords.fst.fst coords.fst.snd coords.snd))

-- Lean theorem statement proving the total count of shaded smaller cubes
theorem number_of_shaded_cubes : countShadedCubes = 23 := 
by
  -- Here, the proof must be provided
  sorry

end number_of_shaded_cubes_l35_35133


namespace sum_of_distinct_integers_l35_35750

theorem sum_of_distinct_integers
  (a b c d e : ℤ)
  (distinct : list.nodup [a, b, c, d, e])
  (prime_exists : ∃ x ∈ [a, b, c, d, e], nat.prime (int.nat_abs x))
  (prod_eq : (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120) :
  a + b + c + d + e = 34 :=
sorry

end sum_of_distinct_integers_l35_35750


namespace equal_sum_partition_possible_l35_35639

theorem equal_sum_partition_possible (n : ℕ) (a : Fin n → ℕ)
  (h1 : a 0 = 1)
  (h2 : ∀ i : Fin (n - 1), a i < a i.succ ∧ a i.succ ≤ 2 * a i)
  (h_sum_even : (Finset.univ.sum a) % 2 = 0) :
  ∃ (S₁ S₂ : Finset (Fin n)), S₁ ∪ S₂ = Finset.univ ∧ S₁ ∩ S₂ = ∅ ∧ (Finset.sum S₁ a = Finset.sum S₂ a) :=
sorry

end equal_sum_partition_possible_l35_35639


namespace general_term_sequence_sum_first_n_terms_l35_35661

theorem general_term_sequence (a n : ℕ) (S : ℕ → ℕ) (h_seq : ∀ n : ℕ, 0 < n → 2 * S n = (a n + 3) * (a n - 2)) :
  (∀ n : ℕ, 2 ≤ n → a n = n + 2) := 
sorry

theorem sum_first_n_terms (T : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) (h_an : ∀ n : ℕ, 0 < n → a n = n + 2) :
  (∀ n : ℕ, T n = n / (6 * n + 9)) :=
sorry

end general_term_sequence_sum_first_n_terms_l35_35661


namespace top_three_probability_l35_35554

-- Definitions for the real-world problem
def total_ways_to_choose_three_cards : ℕ :=
  52 * 51 * 50

def favorable_ways_to_choose_three_specific_suits : ℕ :=
  13 * 13 * 13 * 6

def probability_top_three_inclusive (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

-- The mathematically equivalent proof problem's Lean statement
theorem top_three_probability:
  probability_top_three_inclusive total_ways_to_choose_three_cards favorable_ways_to_choose_three_specific_suits = 2197 / 22100 :=
by
  sorry

end top_three_probability_l35_35554


namespace distance_between_foci_of_hyperbola_is_10_l35_35986

-- Definitions as per the given conditions
def hyperbola_equation : ℝ → ℝ → Prop := 
  λ x y, 9 * x^2 - 18 * x - 16 * y^2 - 32 * y = 144

-- The theorem statement
theorem distance_between_foci_of_hyperbola_is_10 : 
  (∃ a b c : ℝ, 
    (∀ x y : ℝ, hyperbola_equation x y ↔ ((x - a)^2 / 16 - (y - b)^2 / 9 = 1))
    ∧ c^2 = 16 + 9 
    ∧ 2 * c = 10) :=
sorry

end distance_between_foci_of_hyperbola_is_10_l35_35986


namespace temperature_difference_l35_35002

theorem temperature_difference 
    (foot_temp summit_temp : ℝ)
    (h_foot : foot_temp = 24) 
    (h_summit : summit_temp = -50) :
    foot_temp - summit_temp = 74 := 
by 
    rw [h_foot, h_summit]
    exact rfl

end temperature_difference_l35_35002


namespace problem1_problem2_l35_35358

section Problem1

variable {n : ℕ} (A : fin n.succ → ℕ)

def A_prime (A : fin n.succ → ℕ) (i : fin n.succ) : ℕ :=
if A (if i = 0 then ⟨n, Nat.lt_succ_self n⟩ else ⟨i - 1, Nat.pred_lt (zero_lt_succ i) ⟩) = 
   A (if i = n then 0 else ⟨i + 1, Nat.succ_lt_succ i.is_lt⟩) then 0 else 1

theorem problem1 (h_n : 3 ≤ n) (h : ∀ i : fin n.succ, A i = 0 ∨ A i = 1) :
  (∀ i : fin n.succ, A i + A_prime A i = 1) →
  (A = λ i, 1) ∨ (n % 3 = 0 ∧ (A = λ i, if i % 3 = 0 then 0 else if (i + 1) % 3 = 0 then 0 else 1))
:= sorry

end Problem1

section Problem2

variable {n : ℕ}

noncomputable def u_n (n : ℕ) : ℕ :=
if n % 4 = 0 then 2^(n-2)
else 2^(n-2) - ((Complex.i * Complex.ofReal 4) ^ n - (-Complex.i * Complex.ofReal 4) ^ n) / (Complex.ofReal 4 * Complex.i)

theorem problem2 (h : ∀ i : fin n, A i = 0 ∨ A i = 1) :
  (∃ k : ℕ, n = 4 * k + 1) →
  (∑ i in finset.range n, A i = 3) →
  u_n n
:= sorry

end Problem2

end problem1_problem2_l35_35358


namespace cross_product_result_l35_35224

open Matrix

def a : Fin 3 → ℝ := ![3, 4, -5]

def b : Fin 3 → ℝ := ![2, -1, 4]

def cross_product (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![(a 1) * (b 2) - (a 2) * (b 1),
    (a 2) * (b 0) - (a 0) * (b 2),
    (a 0) * (b 1) - (a 1) * (b 0)]

theorem cross_product_result : cross_product a b = ![11, -22, -11] :=
by
  sorry

end cross_product_result_l35_35224


namespace tanya_body_lotions_l35_35943

variable {F L : ℕ}  -- Number of face moisturizers (F) and body lotions (L) Tanya bought

theorem tanya_body_lotions
  (price_face_moisturizer : ℕ := 50)
  (price_body_lotion : ℕ := 60)
  (num_face_moisturizers : ℕ := 2)
  (total_spent : ℕ := 1020)
  (christy_spending_factor : ℕ := 2)
  (h_together_spent : total_spent = 3 * (num_face_moisturizers * price_face_moisturizer + L * price_body_lotion)) :
  L = 4 :=
by
  sorry

end tanya_body_lotions_l35_35943


namespace distinct_lg_differences_count_l35_35232

noncomputable def num_distinct_lg_differences (s : Finset ℕ) : ℕ :=
  let pairs := (s.product s).filter (λ (ab : ℕ × ℕ), ab.1 ≠ ab.2)
  let ratios := pairs.image (λ (ab : ℕ × ℕ), ab.1.to_rat / ab.2.to_rat)
  ratios.card

theorem distinct_lg_differences_count :
  num_distinct_lg_differences ({1, 2, 3, 4, 5, 6} : Finset ℕ) = 22 :=
sorry

end distinct_lg_differences_count_l35_35232


namespace meal_serving_problem_l35_35775

theorem meal_serving_problem :
  let people := 9
  let choices := 3
  let orders : Fin 3 → Fin 3 := by admit -- beef, chicken, fish
  let serving_ways : (orders → Fin 9) → Bool := by admit
  ∃ f : (Fin 3 → Fin 3), serving_ways f → f '' {i | ∃ k, orders i = k} = 216 := by
sorry

end meal_serving_problem_l35_35775


namespace a_general_term_l35_35248

noncomputable def a : ℕ+ → ℤ
| 1 => 3
| 2 => 5
| 3 => 7
| n+1 => 2 * n + 1

def S (n : ℕ+) : ℤ :=
  2 * n * a (n + 1) - 3 * n^2 - 4 * n

lemma S_3_correct : S 3 = 15 :=
sorry

theorem a_general_term (n : ℕ+) : a n = 2 * n + 1 :=
sorry

end a_general_term_l35_35248


namespace equal_split_l35_35740

theorem equal_split (A B C : ℝ) (h1 : A < B) (h2 : B < C) : 
  (B + C - 2 * A) / 3 = (A + B + C) / 3 - A :=
by
  sorry

end equal_split_l35_35740


namespace weight_of_lightest_bag_l35_35055

theorem weight_of_lightest_bag (x : ℝ) :
  let w1 := 4 * x,
      w2 := 5 * x,
      w3 := 6 * x in
  w3 + w1 = w2 + 45 → w1 = 36 :=
by
  intros _ h,
  sorry

end weight_of_lightest_bag_l35_35055


namespace line_equation_l35_35115

variable (θ : ℝ) (b : ℝ) (y x : ℝ)

-- Conditions: 
-- Slope angle θ = 45°
def slope_angle_condition : θ = 45 := by
  sorry

-- Y-intercept b = 2
def y_intercept_condition : b = 2 := by
  sorry

-- Given these conditions, we want to prove the line equation
theorem line_equation (x : ℝ) (θ : ℝ) (b : ℝ) :
  θ = 45 → b = 2 → y = x + 2 := by
  sorry

end line_equation_l35_35115


namespace problem_l35_35273

def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

noncomputable def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x1 x2 : ℝ, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 < f x2

theorem problem (x : ℝ) :
  (is_increasing_on f 1 (1/(0:ℝ))) ∧ f 4 = 9 / 5 ∧ f 1 = 3 / 2 :=
by
  sorry

end problem_l35_35273


namespace team_size_per_team_l35_35570

theorem team_size_per_team (managers employees teams people_per_team : ℕ) 
  (h1 : managers = 23) 
  (h2 : employees = 7) 
  (h3 : teams = 6) 
  (h4 : people_per_team = (managers + employees) / teams) : 
  people_per_team = 5 :=
by 
  sorry

end team_size_per_team_l35_35570


namespace second_machine_fill_800_cartridges_in_4_minutes_l35_35543

def first_machine_cartridge_rate : ℚ := 200 / 3
def combined_cartridge_rate : ℚ := 800 / 3
def combined_envelope_rate : ℚ := 1000 / 5
def first_machine_envelope_rate : ℚ := 1000 / 20

theorem second_machine_fill_800_cartridges_in_4_minutes :
  ∀ r₂_cartridge_rate: ℚ,
    (first_machine_cartridge_rate + r₂_cartridge_rate = combined_cartridge_rate) →
    (50 + (combined_envelope_rate - first_machine_envelope_rate) = combined_envelope_rate) →
    (800 / r₂_cartridge_rate = 4) :=
by {
  intros r₂_cartridge_rate h1 h2,
  sorry,
}

end second_machine_fill_800_cartridges_in_4_minutes_l35_35543


namespace moles_of_MgCO3_formed_l35_35622

-- Definitions for the chemical substances and their moles
def MgO := "MgO"
def CO2 := "CO2"
def MgCO3 := "MgCO3"

-- Conditions from the chemical equation
def balanced_reaction (r MgO: ℕ) (r CO2: ℕ) : ℕ := r MgO

theorem moles_of_MgCO3_formed :
  ∀ n : ℕ, (balanced_reaction 3 3) = 3 :=
by
  intro n
  sorry

end moles_of_MgCO3_formed_l35_35622


namespace smallest_two_digit_l35_35464

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l35_35464


namespace teresa_age_at_michiko_birth_l35_35803

noncomputable def Teresa_age_now : ℕ := 59
noncomputable def Morio_age_now : ℕ := 71
noncomputable def Morio_age_at_Michiko_birth : ℕ := 38

theorem teresa_age_at_michiko_birth :
  (Teresa_age_now - (Morio_age_now - Morio_age_at_Michiko_birth)) = 26 := 
by
  sorry

end teresa_age_at_michiko_birth_l35_35803


namespace terminating_fraction_count_l35_35991

theorem terminating_fraction_count : 
  let range := {m : ℕ | 1 ≤ m ∧ m ≤ 594}
  (set.count range_subtype, ∃ (m : ℕ) (h : m ∈ range), (∃ k, m = k * 119) ∧ (gcd m 595 = 119)) = 4 :=
begin
  sorry
end

end terminating_fraction_count_l35_35991


namespace summation_problem_l35_35799

theorem summation_problem (n : ℕ) (h : n ≥ 2) :
  (∑ pq in { (p, q) | 0 < p ∧ p < q ∧ q ≤ n ∧ p + q > n ∧ Nat.gcd p q = 1 }.to_finset, 1 / (p * q : ℚ)) = 1 / 2 := 
sorry

end summation_problem_l35_35799


namespace total_egg_collection_l35_35176

theorem total_egg_collection (
  -- Conditions
  (Benjamin_collects : Nat) (h1 : Benjamin_collects = 6) 
  (Carla_collects : Nat) (h2 : Carla_collects = 3 * Benjamin_collects) 
  (Trisha_collects : Nat) (h3 : Trisha_collects = Benjamin_collects - 4)
  ) : 
  -- Question and answer
  (Total_collects : Nat) (h_total : Total_collects = Benjamin_collects + Carla_collects + Trisha_collects) => 
  (Total_collects = 26) := 
  by
  sorry

end total_egg_collection_l35_35176


namespace slope_tangent_line_at_origin_l35_35207

open Real

theorem slope_tangent_line_at_origin :
  deriv (λ x : ℝ, exp x) 0 = exp 0 := by
  sorry

end slope_tangent_line_at_origin_l35_35207


namespace incorrect_optionC_l35_35493

def optionA : Prop := (+(-5) = -5)
def optionB : Prop := (-(-0.5) = 0.5)
def optionC : Prop := (-(+ (1 + 1/2)) = (1 + 1/2)) -- This is the incorrect statement we need to refute
def optionD : Prop := (-|3| = -3)

theorem incorrect_optionC : optionC = False :=
by
  have h : -(+ (1 + 1/2)) = -(1 + 1/2) := by sorry
  exact sorry

#check optionA  -- Verify statement type
#check optionB  -- Verify statement type
#check optionC  -- Verify statement type
#check optionD  -- Verify statement type

end incorrect_optionC_l35_35493


namespace probability_small_decagon_l35_35584

theorem probability_small_decagon :
  let n : ℕ := 10
  let s_L : ℝ := 1
  let s_s : ℝ := 0.5
  let area (n : ℕ) (s : ℝ) : ℝ := (n * s^2 * (Real.cot (Real.pi / n))) / 4
  let A_large := area n s_L
  let A_small := area n s_s
  (A_small / A_large) = (1 / 4) :=
  by
  sorry

end probability_small_decagon_l35_35584


namespace arithmetic_sequence_properties_l35_35250

noncomputable theory

-- Variables and conditions
def arithmetic_seq (a : ℕ → ℚ) := ∃ d, d ≠ 0 ∧ ∀ n, a (n + 1) - a n = d
def sum_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) := ∀ n, S n = n * (a 1 + a n) / 2
def S3_condition (a : ℕ → ℚ) (S : ℕ → ℚ) := S 3 = a 4 + 4
def geometric_seq (a : ℕ → ℚ) := a 1 * a 4 = a 2 ^ 2

-- Problem statement
theorem arithmetic_sequence_properties (a : ℕ → ℚ) (S : ℕ → ℚ) 
    (h_arith : arithmetic_seq a) 
    (h_sum : sum_first_n_terms a S)
    (h_S3 : S3_condition a S)
    (h_geom : geometric_seq a) : 
  (∀ n, a n = 2 * n) ∧ (∀ n, (1 / S n) + (1 / S (n - 1)) + ... + (1 / S 1) = n / (n + 1)) :=
sorry

end arithmetic_sequence_properties_l35_35250


namespace sum_of_sequence_l35_35729

noncomputable def sequence (a : ℕ → ℝ) := ∀ n, 2 ≤ n ∧ n ≤ 100 → a n + 2 * a (102 - n) = 3 * 2^n

theorem sum_of_sequence : 
  ∀ (a : ℕ → ℝ), 
  (a 1 = - 2 ^ 101) ∧ (sequence a) →
  (Finset.sum (Finset.range 100) a = -4) :=
by
  sorry

end sum_of_sequence_l35_35729


namespace least_non_lucky_multiple_of_11_l35_35137

/--
A lucky integer is a positive integer which is divisible by the sum of its digits.
Example:
- 18 is a lucky integer because 1 + 8 = 9 and 18 is divisible by 9.
- 20 is not a lucky integer because 2 + 0 = 2 and 20 is not divisible by 2.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_non_lucky_multiple_of_11 : ∃ n, n > 0 ∧ n % 11 = 0 ∧ ¬ is_lucky n ∧ ∀ m, m > 0 → m % 11 = 0 → ¬ is_lucky m → n ≤ m := 
by
  sorry

end least_non_lucky_multiple_of_11_l35_35137


namespace train_platform_passing_time_l35_35523

theorem train_platform_passing_time :
  ∀ (train_length platform_length : ℕ) (train_tree_time platform_cross_time : ℕ)
    (speed : ℚ),
    train_length = 600 →
    train_tree_time = 60 →
    platform_cross_time = 105 →
    speed = train_length / train_tree_time →
    platform_length = 450 →
    speed = (train_length + platform_length) / platform_cross_time :=
begin
  intros train_length platform_length train_tree_time platform_cross_time speed,
  intros h_train_length h_train_tree_time h_platform_cross_time h_speed h_platform_length,
  sorry
end

end train_platform_passing_time_l35_35523


namespace paula_money_left_l35_35779

theorem paula_money_left : 
  ∀ (starting_amount shirts_price_per_unit pants_price num_shirts pants_price_total), 
  starting_amount = 109 ∧ 
  shirts_price_per_unit = 11 ∧ 
  pants_price = 13 ∧ 
  num_shirts = 2 →
  (starting_amount - (num_shirts * shirts_price_per_unit + pants_price) = 74) :=
by
  intros starting_amount shirts_price_per_unit pants_price num_shirts pants_price_total h
  cases' h with h_starting_amount h_rest1
  cases' h_rest1 with h_shirts_price_per_unit h_rest2
  cases' h_rest2 with h_pants_price h_num_shirts
  sorry

end paula_money_left_l35_35779


namespace saffron_milk_caps_and_milk_caps_in_basket_l35_35840

structure MushroomBasket :=
  (total : ℕ)
  (saffronMilkCapCount : ℕ)
  (milkCapCount : ℕ)
  (TotalMushrooms : total = 30)
  (SaffronMilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 12 → ∃ i ∈ selected, i < saffronMilkCapCount)
  (MilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 20 → ∃ i ∈ selected, i < milkCapCount)

theorem saffron_milk_caps_and_milk_caps_in_basket
  (basket : MushroomBasket)
  (TotalMushrooms : basket.total = 30)
  (SaffronMilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 12 → ∃ i ∈ selected, i < basket.saffronMilkCapCount)
  (MilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 20 → ∃ i ∈ selected, i < basket.milkCapCount) :
  basket.saffronMilkCapCount = 19 ∧ basket.milkCapCount = 11 :=
sorry

end saffron_milk_caps_and_milk_caps_in_basket_l35_35840


namespace max_single_player_salary_l35_35904

theorem max_single_player_salary 
  (num_players : ℕ)
  (min_salary : ℕ)
  (max_total_salary : ℕ) 
  (h1 : num_players = 25) 
  (h2 : min_salary = 18000) 
  (h3 : max_total_salary = 1000000) 
  : ∃ x : ℕ, x = 568000 ∧ num_players * min_salary + (x - min_salary) ≤ max_total_salary := 
by 
  have min_expenditure : ℕ := 24 * 18000
  have remaining_budget := 1000000 - min_expenditure
  use remaining_budget
  simp [remaining_budget, min_expenditure]
  sorry

end max_single_player_salary_l35_35904


namespace length_of_faster_train_l35_35850

-- Definitions for the given conditions
def speed_faster_train_kmh : ℝ := 50
def speed_slower_train_kmh : ℝ := 32
def time_seconds : ℝ := 15

theorem length_of_faster_train : 
  let speed_relative_kmh := speed_faster_train_kmh - speed_slower_train_kmh
  let speed_relative_mps := speed_relative_kmh * (1000 / 3600)
  let length_faster_train := speed_relative_mps * time_seconds
  length_faster_train = 75 := 
by 
  sorry 

end length_of_faster_train_l35_35850


namespace f_28_l35_35647

noncomputable def f1 (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

def fn : ℕ → (ℝ → ℝ)
| 0 => id
| (n+1) => λ x, f1 (fn n x)

theorem f_28 (x : ℝ) : fn 27 (f1 x) = 1 / (1 - x) :=
by
  sorry

end f_28_l35_35647


namespace smallest_two_digit_number_product_12_l35_35481

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l35_35481


namespace polynomial_non_integer_infinite_l35_35743

theorem polynomial_non_integer_infinite 
  (P : Polynomial ℝ)
  (k : ℤ)
  (h : ∃ m : ℤ, P.eval m ∉ ℤ) :
  ∃ m : ℤ, ∀ n : ℤ, P.eval (m + k * n) ∉ ℤ := 
sorry

end polynomial_non_integer_infinite_l35_35743


namespace factorization_of_x4_minus_16_l35_35217

theorem factorization_of_x4_minus_16 :
  (x : ℂ) → (x^4 - 16 = (x - 2) * (x + 2) * (x - 2 * complex.I) * (x + 2 * complex.I)) :=
begin
  sorry 
end

end factorization_of_x4_minus_16_l35_35217


namespace interval_contains_root_l35_35408

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem interval_contains_root :
  f (-1) < 0 → 
  f 0 < 0 → 
  f 1 < 0 → 
  f 2 > 0 → 
  ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  intro h1 h2 h3 h4
  sorry

end interval_contains_root_l35_35408


namespace repeating_decimal_sum_l35_35213

open Real

noncomputable def repeating_decimal_to_fraction (d: ℕ) : ℚ :=
  if d = 3 then 1/3 else if d = 7 then 7/99 else if d = 9 then 1/111 else 0 -- specific case of 3, 7, 9.

theorem repeating_decimal_sum:
  let x := repeating_decimal_to_fraction 3
  let y := repeating_decimal_to_fraction 7
  let z := repeating_decimal_to_fraction 9
  x + y + z = 499 / 1189 :=
by
  sorry -- Proof is omitted

end repeating_decimal_sum_l35_35213


namespace sum_primes_f_equals_802_l35_35990

noncomputable def f : ℕ → ℤ
| n => n^4 - 360 * n^2 + 400

theorem sum_primes_f_equals_802 :
  (∑ n in (Finset.filter (λ n => Nat.Prime (f n)) (Finset.Icc 1 19)), f n) = 802 := by
  sorry

end sum_primes_f_equals_802_l35_35990


namespace find_t_l35_35032

theorem find_t (ξ : ℝ → ℝ) (mean : ℝ) (variance : ℝ)
  (hξ : ∀ x, ξ x = pdf_normal mean variance x)
  (hmean : mean = 2) (hvar : variance = 9) (hprob : ∀ t, P (ξ > t) = P (ξ < t - 2))
  : ∀ t, t = 3 :=
by 
  sorry

end find_t_l35_35032


namespace inequality_proof_l35_35646

theorem inequality_proof (n : ℕ) (h1 : n ≥ 2) (a : ℕ → ℝ) (h2 : ∀ i, 1 < a i) :
  2^(n-1) * (∏ i in Finset.range n, a i) + 1 > ∏ i in Finset.range n, (1 + a i) :=
by
  sorry

end inequality_proof_l35_35646


namespace externally_tangent_circles_proof_l35_35511

noncomputable def externally_tangent_circles (r r' : ℝ) (φ : ℝ) : Prop :=
  (r + r')^2 * Real.sin φ = 4 * (r - r') * Real.sqrt (r * r')

theorem externally_tangent_circles_proof (r r' φ : ℝ) 
  (h1: r > 0) (h2: r' > 0) (h3: φ ≥ 0 ∧ φ ≤ π) : 
  externally_tangent_circles r r' φ :=
sorry

end externally_tangent_circles_proof_l35_35511


namespace solution_to_system_l35_35797

def system_of_equations (x y : ℝ) : Prop := (x^2 - 9 * y^2 = 36) ∧ (3 * x + y = 6)

theorem solution_to_system : 
  {p : ℝ × ℝ | system_of_equations p.1 p.2} = { (12 / 5, -6 / 5), (3, -3) } := 
by sorry

end solution_to_system_l35_35797


namespace initial_balls_in_bag_l35_35308

theorem initial_balls_in_bag (n : ℕ) 
  (h_add_white : ∀ x : ℕ, x = n + 1)
  (h_probability : (5 / 8) = 0.625):
  n = 7 :=
sorry

end initial_balls_in_bag_l35_35308


namespace max_roses_purchase_l35_35021

theorem max_roses_purchase (price_individual : ℝ) (price_dozen : ℝ) (price_two_dozen : ℝ) (budget : ℝ) :
  price_individual = 4.5 →
  price_dozen = 36 →
  price_two_dozen = 50 →
  budget = 680 →
  ∃ n : ℕ, n = 318 ∧ 
  let max_roses := (floor (budget / price_two_dozen) * 24 + 
    floor ((budget - floor (budget / price_two_dozen) * price_two_dozen) / price_individual)) in
  n = max_roses := sorry

end max_roses_purchase_l35_35021


namespace value_of_x_l35_35700

theorem value_of_x : 
  ∀ (x : ℤ), (1 / 4 : ℚ) - (1 / 6) = 4 / x → x = 48 :=
by
  intros x h
  unfold_coes at h
  sorry

end value_of_x_l35_35700


namespace train_cars_l35_35847

theorem train_cars (cars_in_15_seconds : ℕ) (time_seconds : ℕ) (constant_speed : Prop) :
  cars_in_15_seconds = 9 → 
  time_seconds = 210 → 
  constant_speed → 
  let cars_per_second := cars_in_15_seconds / 15 in
  (cars_per_second * time_seconds = 126) :=
by
  intros h1 h2 h3
  simp [h1, h2]
  let cars_per_second := 9 / 15
  have : cars_per_second * 210 = 126 := by
    calc
      (9 / 15) * 210 = 0.6 * 210 : by sorry
      ... = 126 : by sorry
  exact this

end train_cars_l35_35847


namespace ordered_samples_count_words_of_length_n_injective_functions_count_l35_35026

-- Definition of the Pochhammer symbol (falling factorial) in Lean
def pochhammer (N n : ℕ) : ℕ := list.prod (list.range' N n)

-- Part (a) Ordered samples without replacement
theorem ordered_samples_count (N n : ℕ) (hN : N ≥ n) :
  ∃ (A : finset ℕ), A.card = N ∧ finset.card (finset.pi (list.range' N n) (λ _, A)) = pochhammer N n := sorry

-- Part (b) Words of length n from N letters
theorem words_of_length_n (N n : ℕ) (hN : N ≥ n) :
  ∃ (alphabet : finset ℕ), alphabet.card = N ∧ finset.card (finset.pi (list.range' n N) (λ _, alphabet)) = pochhammer N n := sorry

-- Part (c) Injective functions
theorem injective_functions_count (N n : ℕ) (hN : N ≥ n):
  ∃ (X Y : finset ℕ), X.card = n ∧ Y.card = N ∧ finset.card (finset.pi (finset.range n) (λ _, Y)) = pochhammer N n := sorry

end ordered_samples_count_words_of_length_n_injective_functions_count_l35_35026


namespace cranberries_left_in_bog_l35_35167

theorem cranberries_left_in_bog
  (total_cranberries : ℕ)
  (percentage_harvested_by_humans : ℕ)
  (cranberries_eaten_by_elk : ℕ)
  (initial_count : total_cranberries = 60000)
  (harvested_percentage : percentage_harvested_by_humans = 40)
  (eaten_by_elk : cranberries_eaten_by_elk = 20000) :
  total_cranberries - (total_cranberries * percentage_harvested_by_humans / 100) - cranberries_eaten_by_elk = 16000 :=
by
  sorry

end cranberries_left_in_bog_l35_35167


namespace ratio_distance_traveled_by_foot_l35_35151

theorem ratio_distance_traveled_by_foot (D F B C : ℕ) (hD : D = 40) 
(hB : B = D / 2) (hC : C = 10) (hF : F = D - (B + C)) : F / D = 1 / 4 := 
by sorry

end ratio_distance_traveled_by_foot_l35_35151


namespace masks_prepared_by_teacher_l35_35595

theorem masks_prepared_by_teacher Zhao (n t : ℕ)
  (h1 : n / 2 * 5 + n / 2 * 7 + 25 = t)
  (h2 : n / 3 * 10 + 2 * n / 3 * 7 - 35 = t) :
  t = 205 := 
sorry

end masks_prepared_by_teacher_l35_35595


namespace total_cost_price_l35_35551

theorem total_cost_price (SP1 SP2 SP3 : ℝ) (P1 P2 P3 : ℝ) 
  (h1 : SP1 = 120) (h2 : SP2 = 150) (h3 : SP3 = 200)
  (h4 : P1 = 0.20) (h5 : P2 = 0.25) (h6 : P3 = 0.10) : (SP1 / (1 + P1) + SP2 / (1 + P2) + SP3 / (1 + P3) = 401.82) :=
by
  sorry

end total_cost_price_l35_35551


namespace Rs_share_correct_l35_35949

variables (P Q R S : ℝ) -- capitals of partners P, Q, R, S
variables (p q r s : ℝ) -- aliases for the capitals of P, Q, R, S
variables (total_profit : ℝ := 12090) -- total profit at the end of the second year

-- Conditions based on the problem statement
axiom ratio_PQ : 4 * p = 6 * q
axiom ratio_QR : 6 * q = 10 * r
axiom capital_S : s = p + q 

-- Reinvestment and calculation of total profit
def total_capital : ℝ := p + q + r + s

-- Assuming total profit at the end of the year 2 as a given fact
constant total_year2_profit : ℝ := total_profit

-- R's share of the profit
def r_share_of_profit : ℝ := (total_year2_profit * r) / total_capital

-- Proof statement verifying that R's share is Rs 1,295
theorem Rs_share_correct : 
  (by { simp only [ratio_PQ, ratio_QR, capital_S],
        have r_value := r, sorry :  r_value = 6 * (p / 15),
        have t_cap := sorry : total_capital = 56 * (p / 15),
        have R_profit := sorry : r_share_of_profit  = 1295 }) :=
  1295

end Rs_share_correct_l35_35949


namespace smallest_n_l35_35791

theorem smallest_n {x y z : ℝ} (n : ℕ) :
  (∀ x y z, 0 ≤ x ∧ x ≤ n ∧ 0 ≤ y ∧ y ≤ n ∧ 0 ≤ z ∧ z ≤ n ∧ abs (x - y) ≥ 2 ∧ abs (y - z) ≥ 2 ∧ abs (z - x) ≥ 2) →
  (∃ n : ℕ, (n ≥ 12) ∧ ((n - 4 : ℝ)^3 / n^3 > 1 / 2)) :=
sorry

end smallest_n_l35_35791


namespace canoes_more_than_kayaks_l35_35510

noncomputable def canoes_and_kayaks (C K : ℕ) : Prop :=
  (2 * C = 3 * K) ∧ (12 * C + 18 * K = 504) ∧ (C - K = 7)

theorem canoes_more_than_kayaks (C K : ℕ) (h : canoes_and_kayaks C K) : C - K = 7 :=
sorry

end canoes_more_than_kayaks_l35_35510


namespace g_5_fifth_power_is_125_l35_35800

noncomputable def f (x : ℝ) := sorry
noncomputable def g (x : ℝ) := sorry

theorem g_5_fifth_power_is_125 :
  (∀ x : ℝ, x ≥ 1 → f(g(x)) = x^3) →
  (∀ x : ℝ, x ≥ 1 → g(f(x)) = x^5) →
  g(25) = 25 →
  (g(5))^5 = 125 :=
by
  intros h1 h2 h3
  sorry

end g_5_fifth_power_is_125_l35_35800


namespace total_cost_price_correct_l35_35550

def SP1 : ℝ := 120
def SP2 : ℝ := 150
def SP3 : ℝ := 200
def profit1 : ℝ := 0.20
def profit2 : ℝ := 0.25
def profit3 : ℝ := 0.10

def CP1 := SP1 / (1 + profit1)
def CP2 := SP2 / (1 + profit2)
def CP3 := SP3 / (1 + profit3)

def total_cost_price := CP1 + CP2 + CP3

theorem total_cost_price_correct : total_cost_price = 401.82 :=
by sorry

end total_cost_price_correct_l35_35550


namespace range_of_distance_OP_l35_35667

-- Defining the points and the radius
variable {O P : Point} (r : ℝ)
def radius : ℝ := 5
def not_inside_circle (P : Point) : Prop := dist P O ≥ radius

-- Statement to prove the range
theorem range_of_distance_OP (h : not_inside_circle P) : dist P O ≥ radius := by
  sorry

end range_of_distance_OP_l35_35667


namespace find_line_equation_through_ellipse_midpoint_l35_35665

theorem find_line_equation_through_ellipse_midpoint {A B : ℝ × ℝ} 
  (hA : (A.fst^2 / 2) + A.snd^2 = 1) 
  (hB : (B.fst^2 / 2) + B.snd^2 = 1) 
  (h_midpoint : (A.fst + B.fst) / 2 = 1 ∧ (A.snd + B.snd) / 2 = 1 / 2) : 
  ∃ k : ℝ, (k = -1) ∧ (∀ x y : ℝ, (y - 1/2 = k * (x - 1)) → 2*x + 2*y - 3 = 0) :=
sorry

end find_line_equation_through_ellipse_midpoint_l35_35665


namespace smallest_two_digit_number_product_12_l35_35482

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l35_35482


namespace jimmy_shoveled_9_driveways_l35_35332

noncomputable def discounted_price (original_price discount : ℚ) : ℚ :=
  original_price - original_price * discount

noncomputable def total_cost (prices : List ℚ) : ℚ :=
  prices.sum

noncomputable def with_sales_tax (subtotal tax_rate : ℚ) : ℚ :=
  subtotal + subtotal * tax_rate

noncomputable def total_earned (spent fraction : ℚ) : ℚ :=
  spent / fraction

noncomputable def driveways_shoveled (total_earned charge_per_driveway : ℚ) : ℕ :=
  (total_earned / charge_per_driveway).to_nat

theorem jimmy_shoveled_9_driveways :
  let candy_bar_price := 0.75
      lollipop_price := 0.25
      candy_bar_discount := 0.20
      sales_tax_rate := 0.05
      snow_fraction := 1/6
      charge_per_driveway := 1.5
      candy_bars_bought := 2
      lollipops_bought := 4
      discounted_candy_bar_price := discounted_price candy_bar_price candy_bar_discount
      total_candy_bars := total_cost [discounted_candy_bar_price * candy_bars_bought]
      total_lollipops := total_cost [lollipop_price * lollipops_bought]
      subtotal := total_candy_bars + total_lollipops
      total_spent := with_sales_tax subtotal sales_tax_rate
      total_earned := total_earned total_spent snow_fraction
  in driveways_shoveled total_earned charge_per_driveway = 9 :=
by
  sorry

end jimmy_shoveled_9_driveways_l35_35332


namespace arithmetic_seq_a12_l35_35724

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Prove that a_12 = 12 given the conditions
theorem arithmetic_seq_a12 :
  ∃ a₁, (arithmetic_seq a₁ 2 2 = -8) → (arithmetic_seq a₁ 2 12 = 12) :=
by
  sorry

end arithmetic_seq_a12_l35_35724


namespace chessboard_bishops_placement_l35_35721

theorem chessboard_bishops_placement : 
  (∃ (num_ways : ℕ), num_ways = 768) :=
by
  have num_ways := 32 * 24
  existsi num_ways
  -- The correct number of ways to place the bishops
  have h : num_ways = 768 := by decide
  exact h

end chessboard_bishops_placement_l35_35721


namespace closest_fraction_among_given_l35_35563

theorem closest_fraction_among_given (f1 f2 f3 f4 : ℚ) (x : ℚ) :
  f1 = 101 / 5 ∧ f2 = 141 / 7 ∧ f3 = 181 / 9 ∧ f4 = 161 / 8 ∧ x = 20.14 →
  (| (141 / 7) - 20.14 | ≤ | (101 / 5) - 20.14 |) ∧
  (| (141 / 7) - 20.14 | ≤ | (181 / 9) - 20.14 |) ∧
  (| (141 / 7) - 20.14 | ≤ | (161 / 8) - 20.14 |) :=
by
  intros
  exact sorry

end closest_fraction_among_given_l35_35563


namespace trapezoid_lower_side_length_l35_35394

variable (U L : ℝ) (height area : ℝ)

theorem trapezoid_lower_side_length
  (h1 : L = U - 3.4)
  (h2 : height = 5.2)
  (h3 : area = 100.62)
  (h4 : area = (1 / 2) * (U + L) * height) :
  L = 17.65 :=
by
  sorry

end trapezoid_lower_side_length_l35_35394


namespace min_value_of_T_l35_35747

noncomputable def T (x p : ℝ) : ℝ := |x - p| + |x - 15| + |x - (15 + p)|

theorem min_value_of_T (p : ℝ) (hp : 0 < p ∧ p < 15) :
  ∃ x, p ≤ x ∧ x ≤ 15 ∧ T x p = 15 :=
sorry

end min_value_of_T_l35_35747


namespace points_from_set_l35_35996

theorem points_from_set :
  let S := {1, 2, 3, 4, 5}
  in ∃ (S : Finset ℕ), S.card = 5 ∧ ∃ (T : Finset (ℕ × ℕ)), (∀ p ∈ T, p.1 ≠ p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S) ∧ T.card = 25 := 
by sorry

end points_from_set_l35_35996


namespace average_of_four_variables_l35_35694

theorem average_of_four_variables (x y z w : ℝ) (h : (5 / 2) * (x + y + z + w) = 25) :
  (x + y + z + w) / 4 = 2.5 :=
sorry

end average_of_four_variables_l35_35694


namespace remainder_when_sum_divided_by_29_l35_35368

theorem remainder_when_sum_divided_by_29 (c d : ℤ) (k j : ℤ) 
  (hc : c = 52 * k + 48) 
  (hd : d = 87 * j + 82) : 
  (c + d) % 29 = 22 := 
by 
  sorry

end remainder_when_sum_divided_by_29_l35_35368


namespace midpoints_of_segments_not_collinear_l35_35054

variables {V : Type} [add_comm_group V] [vector_space ℝ V]

structure Triangle (V : Type) [add_comm_group V] [vector_space ℝ V] :=
(A B C : V)

structure PointOnSide (V : Type) [add_comm_group V] [vector_space ℝ V] (T : Triangle V) := 
(A1 : V)
(B1 : V)
(C1 : V)
(HA1 : ∃ t ∈ set.Icc (0 : ℝ) 1, A1 = t • (T.B) + (1-t) • (T.C))
(HB1 : ∃ t ∈ set.Icc (0 : ℝ) 1, B1 = t • (T.C) + (1-t) • (T.A))
(HC1 : ∃ t ∈ set.Icc (0 : ℝ) 1, C1 = t • (T.A) + (1-t) • (T.B))

noncomputable def midpoint (P Q : V) : V := (1/2 : ℝ) • (P + Q)

noncomputable def midpoints_not_collinear 
(T : Triangle V) 
(P : PointOnSide V T) : Prop :=
¬ collinear ℝ 
  { midpoint T.A P.A1,
    midpoint T.B P.B1,
    midpoint T.C P.C1 }

theorem midpoints_of_segments_not_collinear (T : Triangle V) (P : PointOnSide V T) : 
  midpoints_not_collinear T P :=
sorry

end midpoints_of_segments_not_collinear_l35_35054


namespace marbles_sum_l35_35215

variable {K M : ℕ}

theorem marbles_sum (hFabian_kyle : 15 = 3 * K) (hFabian_miles : 15 = 5 * M) :
  K + M = 8 :=
by
  sorry

end marbles_sum_l35_35215


namespace quadrilateral_triangle_area_l35_35649

theorem quadrilateral_triangle_area (A B C P₁ P₂ P₃ P₄ : Type*)
  [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C]
  (h_trig: triangle A B C)
  (h_P₁ : P₁ ∈ segment ℝ A B ∨ P₁ ∈ segment ℝ A C ∨ P₁ ∈ segment ℝ B C)
  (h_P₂ : P₂ ∈ segment ℝ A B ∨ P₂ ∈ segment ℝ A C ∨ P₂ ∈ segment ℝ B C)
  (h_P₃ : P₃ ∈ segment ℝ A B ∨ P₃ ∈ segment ℝ A C ∨ P₃ ∈ segment ℝ B C)
  (h_P₄ : P₄ ∈ segment ℝ A B ∨ P₄ ∈ segment ℝ A C ∨ P₄ ∈ segment ℝ B C) :
  ∃ T ∈ {triangle P₁ P₂ P₃, triangle P₁ P₂ P₄, triangle P₁ P₃ P₄, triangle P₂ P₃ P₄}
    area(T) ≤ (1/4 : ℝ) * area(triangle A B C) :=
by
  sorry

end quadrilateral_triangle_area_l35_35649


namespace each_trainer_hours_l35_35412

theorem each_trainer_hours (dolphins : ℕ) (hours_per_dolphin : ℕ) (trainers : ℕ) :
  dolphins = 4 →
  hours_per_dolphin = 3 →
  trainers = 2 →
  (dolphins * hours_per_dolphin) / trainers = 6 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end each_trainer_hours_l35_35412


namespace constant_term_binomial_expansion_l35_35398

theorem constant_term_binomial_expansion :
  let expr := (3 * x - 2 / x)^8 in
  ∃ t: ℤ, t = 112 ∧ ∀ (x: ℝ), -- working with x as reals to allow division
  expr = ∑ k in range(9), (nat.choose 8 k) * (3 * x)^(8 - k) * ((-2 / x)^k) ∧
  (∃ r: ℕ, r ≤ 8 ∧ (8 - 2 * r = 0) ∧
             ((nat.choose 8 r) * (3^(8 - r)) * (-2)^r) = 112) :=
by
  sorry -- Proof to be completed

end constant_term_binomial_expansion_l35_35398


namespace ratio_of_female_democrats_l35_35429

theorem ratio_of_female_democrats (F M : ℕ) (total_participants : ℕ) (total_democrats : ℕ) (female_democrats : ℕ) :
    total_participants = 720 →
    (M / 4 : ℝ) = 120 →
    (total_participants / 3 : ℝ) = 240 →
    female_democrats = 120 →
    total_democrats = 240 →
    M + F = total_participants →
    120 / (F : ℝ) = 1 / 2 :=
by sorry

end ratio_of_female_democrats_l35_35429


namespace probability_slope_le_one_l35_35757

noncomputable def point := (ℝ × ℝ)

def Q_in_unit_square (Q : point) : Prop :=
  0 ≤ Q.1 ∧ Q.1 ≤ 1 ∧ 0 ≤ Q.2 ∧ Q.2 ≤ 1

def slope_le_one (Q : point) : Prop :=
  (Q.2 - (1/4)) / (Q.1 - (3/4)) ≤ 1

theorem probability_slope_le_one :
  ∃ p q : ℕ, Q_in_unit_square Q → slope_le_one Q →
  p.gcd q = 1 ∧ (p + q = 11) :=
sorry

end probability_slope_le_one_l35_35757


namespace circle_radius_l35_35886

theorem circle_radius (x y d : ℝ) (h₁ : x = π * r^2) (h₂ : y = 2 * π * r) (h₃ : d = 2 * r) (h₄ : x + y + d = 164 * π) : r = 10 :=
by sorry

end circle_radius_l35_35886


namespace diagonal_sum_l35_35838

-- We need to setup the problem in Lean using the provided conditions and show the required sum.
theorem diagonal_sum (n : ℕ) (h : n ≥ 4) 
  (a : ℤ) (d : ℤ) (q : ℚ)
  (h24 : (a + 3 * d) * q = 1) 
  (h42 : (a + d) * q ^ 3 = 1 / 8)
  (h43 : (a + 2 * d) * q ^ 3 / q = 3 / 16) :
  ∑ k in finset.range n, ((a + (k : ℤ) * d) * q ^ (k : ℤ)) = 2 - n / 2^n := 
sorry

end diagonal_sum_l35_35838


namespace largest_prime_factor_of_binomial_l35_35085

theorem largest_prime_factor_of_binomial :
  ∃ p : ℕ, p.prime ∧ 10 ≤ p ∧ p < 100 ∧ (∃ k : ℕ, p ^ k ∣ (300.factorial / (150.factorial * 150.factorial))) ∧
  (∀ q : ℕ, q.prime → 10 ≤ q ∧ q < 100 → (∃ k : ℕ, q ^ k ∣ (300.factorial / (150.factorial * 150.factorial))) → q ≤ p) :=
sorry

end largest_prime_factor_of_binomial_l35_35085


namespace squat_percentage_loss_l35_35737

variable (original_squat : ℕ)
variable (original_bench : ℕ)
variable (original_deadlift : ℕ)
variable (lost_deadlift : ℕ)
variable (new_total : ℕ)
variable (unchanged_bench : ℕ)

theorem squat_percentage_loss
  (h1 : original_squat = 700)
  (h2 : original_bench = 400)
  (h3 : original_deadlift = 800)
  (h4 : lost_deadlift = 200)
  (h5 : new_total = 1490)
  (h6 : unchanged_bench = 400) :
  (original_squat - (new_total - (unchanged_bench + (original_deadlift - lost_deadlift)))) * 100 / original_squat = 30 :=
by sorry

end squat_percentage_loss_l35_35737


namespace parity_of_expression_l35_35704

theorem parity_of_expression (n : ℤ) (h : n > 1) : odd (n + (n^2 - 1) ^ ((1 - (-1) ^ n) / 2)) := 
sorry

end parity_of_expression_l35_35704


namespace least_multiple_of_11_not_lucky_l35_35135

-- Define what it means for a number to be a lucky integer
def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

-- Define what it means for a number to be a multiple of 11
def is_multiple_of_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- State the problem: the least positive multiple of 11 that is not a lucky integer is 132
theorem least_multiple_of_11_not_lucky :
  ∃ n : ℕ, is_multiple_of_11 n ∧ ¬ is_lucky n ∧ n = 132 :=
sorry

end least_multiple_of_11_not_lucky_l35_35135


namespace square_divided_into_40_smaller_squares_l35_35862

theorem square_divided_into_40_smaller_squares : ∃ squares : ℕ, squares = 40 :=
by
  sorry

end square_divided_into_40_smaller_squares_l35_35862


namespace series_sum_l35_35751

variable {c d : ℝ}

theorem series_sum (h : ∑' n : ℕ, c / d ^ ((3 : ℝ) ^ n) = 9) :
  ∑' n : ℕ, c / (c + 2 * d) ^ (n + 1) = 9 / 11 :=
by
  -- The code that follows will include the steps and proof to reach the conclusion
  sorry

end series_sum_l35_35751


namespace recurring_decimals_sum_correct_l35_35604

noncomputable def recurring_decimals_sum : ℚ :=
  let x := (2:ℚ) / 3
  let y := (4:ℚ) / 9
  x + y

theorem recurring_decimals_sum_correct :
  recurring_decimals_sum = 10 / 9 := 
  sorry

end recurring_decimals_sum_correct_l35_35604


namespace plane_equation_intercept_l35_35324

theorem plane_equation_intercept (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  ∀ x y z : ℝ, ∃ k : ℝ, k = 1 → (x / a + y / b + z / c) = k :=
by sorry

end plane_equation_intercept_l35_35324


namespace tenth_term_is_four_l35_35974

noncomputable def a : ℕ → ℝ
| 0     := 3
| 1     := 4
| (n + 1) := 12 / a n

theorem tenth_term_is_four : a 9 = 4 :=
by
  sorry

end tenth_term_is_four_l35_35974


namespace common_chords_intersect_at_single_point_l35_35288

theorem common_chords_intersect_at_single_point
  (A B C: Point)
  (O1 O2 O3: Circle)
  (h1: O1 ≠ O2)
  (h2: O2 ≠ O3)
  (h3: O1 ≠ O3)
  (hA1 : A_1 ∈ (inter O2 O3))
  (hB1 : B_1 ∈ (inter O1 O3))
  (hC1 : C_1 ∈ (inter O1 O2)) :
  ∃ P: Point, is_common_point O1 A1 C1 ∧ is_common_point O2 B1 C1 ∧ is_common_point O3 A1 B1 := 
sorry

end common_chords_intersect_at_single_point_l35_35288


namespace approx_values_l35_35937

noncomputable def linear_approx (f f' : ℝ → ℝ) (x0 x1 : ℝ) : ℝ :=
  f x0 + f' x0 * (x1 - x0)

-- Part 1: sqrt[4]{17} ≈ 2.03125
def f1 (x : ℝ) : ℝ := x^(1/4)
def f1' (x : ℝ) : ℝ := (1/4) * x^(-3/4)
def x0_1 := 16 -- Closest easy point
def x1_1 := 17 -- Point of interest
def approx1 := linear_approx f1 f1' x0_1 x1_1 -- Expected to be approximately 2.03125

-- Part 2: arctg(0.98) ≈ 0.7754
def y2 (x : ℝ) : ℝ := Real.arctan x
def y2' (x : ℝ) : ℝ := 1 / (1 + x^2)
def x0_2 := 1 -- Closest easy point
def x1_2 := 0.98 -- Point of interest
def approx2 := linear_approx y2 y2' x0_2 x1_2 -- Expected to be approximately 0.7754

-- Part 3: sin(29°) ≈ 0.4848
def y3 (x : ℝ) : ℝ := Real.sin x
def y3' (x : ℝ) : ℝ := Real.cos x
def deg_to_rad (deg : ℝ) := (Real.pi / 180) * deg
def x0_3 := deg_to_rad 30 -- Closest easy point in radians
def x1_3 := deg_to_rad 29 -- Point of interest in radians
def approx3 := linear_approx y3 y3' x0_3 x1_3 -- Expected to be approximately 0.4848

theorem approx_values :
  approx1 ≈ 2.03125 ∧ approx2 ≈ 0.7754 ∧ approx3 ≈ 0.4848 :=
by
  sorry

end approx_values_l35_35937


namespace angle_BAD_measure_l35_35784

-- Define the setup of the problem
variables {A B C D : Type}
variables [geometry : AffineGeometry A]
include geometry

variables (point : A)
variables (ABC : triangle point B point)
variables {α β γ δ : ℝ}
variables (ABD_angle : α = 15)
variables (DBC_angle : β = 50)

-- Define the theorem to be proved
theorem angle_BAD_measure
  (H1 : extension B C point) -- D is on the extension of side BC beyond C
  (H2 : ABD_angle)
  (H3 : DBC_angle)
  : BAC = 35 :=
sorry

end angle_BAD_measure_l35_35784


namespace ship_blown_distance_l35_35375

theorem ship_blown_distance (time_traveled : ℕ) (speed : ℕ) (fraction_halfway : ℚ) (fraction_after_storm : ℚ) :
  time_traveled = 20 → speed = 30 → fraction_halfway = 1 / 2 → fraction_after_storm = 1 / 3 →
  let distance_traveled := time_traveled * speed in
  let total_destination := distance_traveled / fraction_halfway.to_rat in
  let distance_after_storm := total_destination * fraction_after_storm.to_rat in
  let distance_blown_back := distance_traveled - distance_after_storm in
  distance_blown_back = 200 := 
by 
  intros _
  sorry

end ship_blown_distance_l35_35375


namespace imaginary_part_z_plus_inv_z_l35_35363

theorem imaginary_part_z_plus_inv_z :
  let z := (2 : ℂ) + (1 : ℂ) * Complex.i in
  let w := z + (1 / z) in
  Complex.im w = 4 / 5 :=
by
  let z : ℂ := 2 + Complex.i
  let w : ℂ := z + (1 / z)
  have H : w = 12 / 5 + (4 / 5) * Complex.i := by sorry
  rw [H]
  exact rfl

end imaginary_part_z_plus_inv_z_l35_35363


namespace radius_of_circle_Q_l35_35189

noncomputable def radius_Q (r_p r_s : ℝ) : ℝ := r_s * (r_p + r_p / 2 - r_p * r_p / 4) / (r_p - r_p * r_p / 2)

theorem radius_of_circle_Q :
  let P_radius := 2
  let S_radius := 4
  let Q_radius := radius_Q P_radius S_radius in
  Q_radius = 16 / 9 :=
by
  let P_radius := 2
  let S_radius := 4
  let Q_radius := radius_Q P_radius S_radius
  sorry

end radius_of_circle_Q_l35_35189


namespace incorrect_value_of_quadratic_eval_l35_35142

/-- Given a quadratic polynomial P(x) = ax^2 + bx + c, and the evaluations of this polynomial
at a sequence of integers yielding values 2116, 2209, 2304, 2401, 2496, 2601, 2704, and 2809,
prove that the value 2496 is incorrect if we want to maintain a consistent progression in
second differences -/
theorem incorrect_value_of_quadratic_eval (a b c x : ℤ) :
  let P (x : ℤ) := a * x^2 + b * x + c in
  (P 1 = 2116) ∧ (P 2 = 2209) ∧ (P 3 = 2304) ∧ (P 4 = 2401) ∧ (P 5 = 2496)
  ∧ (P 6 = 2601) ∧ (P 7 = 2704) ∧ (P 8 = 2809) →
  ¬(second_differences_constant (λ n, P (n : ℤ))) :=
by
  sorry

noncomputable def second_differences_constant (f : ℕ → ℤ) : Prop :=
  ∀ n, f (n + 2) - 2 * f (n + 1) + f n = constant_value

variable (constant_value : ℤ)


end incorrect_value_of_quadratic_eval_l35_35142


namespace grid_unattainable_l35_35910

def Grid (n m : ℕ) := Fin n → Fin m → ℕ

def is_valid_digit (x : ℕ) : Prop := x = 0 ∨ x = 1 ∨ x = 2

def valid_grid (g : Grid 100 100) : Prop :=
  ∀ i j, is_valid_digit (g i j)

def count_digits (g : Grid 100 100) (r1 r2 c1 c2 : ℕ) : Fin 101 → ℕ
| k => ∑ i in finRange (r2 - r1 + 1), ∑ j in finRange (c2 - c1 + 1), if g (Fin.ofNat (i + r1)) (Fin.ofNat (j + c1)) = k then 1 else 0

def meets_condition (g : Grid 100 100) : Prop :=
  ∀ i j : ℕ, i + 3 ≤ 100 → j + 4 ≤ 100 →
    let z := count_digits g i (i + 2) j (j + 3) 0 in
    let o := count_digits g i (i + 2) j (j + 3) 1 in
    let t := count_digits g i (i + 2) j (j + 3) 2 in
    z = 3 ∧ o = 4 ∧ t = 5

def meets_condition' (g : Grid 100 100) : Prop :=
  ∀ i j : ℕ, i + 4 ≤ 100 → j + 3 ≤ 100 →
    let z := count_digits g i (i + 3) j (j + 2) 0 in
    let o := count_digits g i (i + 3) j (j + 2) 1 in
    let t := count_digits g i (i + 3) j (j + 2) 2 in
    z = 3 ∧ o = 4 ∧ t = 5

theorem grid_unattainable : ∀ g : Grid 100 100, valid_grid g → meets_condition g ∧ meets_condition' g → False :=
by
  intros
  sorry

end grid_unattainable_l35_35910


namespace problem_l35_35941

noncomputable section
open Complex

theorem problem (a b : ℝ) (i : ℂ) (hi : i = Complex.I) : 
  (⟨(1 - Complex.I) / Real.sqrt 2, 0⟩ ^ 2 = (a : ℂ) + b * Complex.I) → 
  a = 0 :=
by
  sorry

end problem_l35_35941


namespace kyle_money_after_snowboarding_l35_35339

theorem kyle_money_after_snowboarding (dave_has : ℕ) (a : dave_has = 46) : 
  let kyle_before := 3 * dave_has - 12 in
  let kyle_after := kyle_before - kyle_before / 3 in
  kyle_after = 84 :=
by
  have h_dave : dave_has = 46 := a
  let kyle_before := 3 * dave_has - 12
  let kyle_after := kyle_before - kyle_before / 3
  have h_kyle_before : kyle_before = 126 := by norm_num [kyle_before, h_dave]
  have h_kyle_after : kyle_after = 84 := by norm_num [kyle_after, h_kyle_before]
  exact h_kyle_after
sorry

end kyle_money_after_snowboarding_l35_35339


namespace initial_num_families_eq_41_l35_35500

-- Definitions based on the given conditions
def num_families_flew_away : ℕ := 27
def num_families_left : ℕ := 14

-- Statement to prove
theorem initial_num_families_eq_41 : num_families_flew_away + num_families_left = 41 := by
  sorry

end initial_num_families_eq_41_l35_35500


namespace expression_D_divisible_by_9_l35_35921

theorem expression_D_divisible_by_9 (k : ℕ) (hk : k > 0) : 9 ∣ 3 * (2 + 7^k) :=
by
  sorry

end expression_D_divisible_by_9_l35_35921


namespace find_k_l35_35711

-- Define the function y = kx
def linear_function (k x : ℝ) : ℝ := k * x

-- Define the point P(3,1)
def P : ℝ × ℝ := (3, 1)

theorem find_k (k : ℝ) (h : linear_function k 3 = 1) : k = 1 / 3 :=
by
  sorry

end find_k_l35_35711


namespace buses_pass_together_buses_pass_together_times_l35_35571

def lcm (a b : ℕ) : ℕ := sorry  -- assume some lcm definition

theorem buses_pass_together (t₀ : ℕ) (interval₁ interval₂ : ℕ) :
  t₀ = 450 ∧ interval₁ = 15 ∧ interval₂ = 25 →
  t₀ + lcm interval₁ interval₂ = 8 * 60 + 45 :=
sorry

theorem buses_pass_together_times (t₀ : ℕ) (interval₁ interval₂ : ℕ) 
  (midnight : ℕ) :
  t₀ = 450 ∧ interval₁ = 15 ∧ interval₂ = 25 ∧ midnight = 1440 →
  ∃ times : list ℕ,
  times = [525, 600, 675, 750, 825, 900, 975, 1050, 1125, 1200, 1275,
           1350, 1425] :=
sorry

end buses_pass_together_buses_pass_together_times_l35_35571


namespace contradiction_method_l35_35790

theorem contradiction_method (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) → a < 1 :=
sorry

end contradiction_method_l35_35790


namespace revenue_increase_l35_35864

theorem revenue_increase (P Q : ℝ) :
    let R := P * Q
    let P_new := 1.7 * P
    let Q_new := 0.8 * Q
    let R_new := P_new * Q_new
    R_new = 1.36 * R :=
sorry

end revenue_increase_l35_35864


namespace number_of_two_digit_integers_congruent_to_1_mod_5_l35_35298

theorem number_of_two_digit_integers_congruent_to_1_mod_5 : 
  let k_range := finset.Icc 2 19 in
  k_range.card = 18 :=
by
  let k_range := finset.Icc 2 19
  have h_card : k_range.card = 19 - 2 + 1 := finset.card_Icc 2 19
  norm_num at h_card
  exact h_card

end number_of_two_digit_integers_congruent_to_1_mod_5_l35_35298


namespace value_of_b_l35_35806

theorem value_of_b (b : ℝ) (h1 : 1/2 * (b / 3) * b = 6) (h2 : b ≥ 0) : b = 6 := sorry

end value_of_b_l35_35806


namespace crackers_per_sleeve_proof_l35_35580

noncomputable def crackers_per_sleeve
  (sandwich_crackers : ℕ)
  (sandwiches_per_night : ℕ)
  (sleeves_per_box : ℕ)
  (boxes : ℕ)
  (nights : ℕ)
  (total_crackers : ℕ)
  (total_sleeves : ℕ) : ℕ :=
  total_crackers / total_sleeves

theorem crackers_per_sleeve_proof :
  crackers_per_sleeve 2 5 4 5 56 (56 * 10) (5 * 4) = 28 :=
by
  unfold crackers_per_sleeve
  simp
  -- "sorry" is used to indicate the completion of the proof is omitted.
  sorry

end crackers_per_sleeve_proof_l35_35580


namespace dot_product_expression_l35_35697

open Real EuclideanSpace

variables (a b : EuclideanSpace) (θ : ℝ)
hypothesis ha : ∥a∥ = 4
hypothesis hb : ∥b∥ = 5
hypothesis hθ : θ = π / 3 -- angle 60 degrees in radians
hypothesis hab : inner a b = 4 * 5 * cos (π / 3)

theorem dot_product_expression : inner (a + b) (a - b) = 11 :=
by
  -- insert proof here
  sorry

end dot_product_expression_l35_35697


namespace find_x_difference_l35_35532

noncomputable def isosceles_triangle_circumcircle (x: ℝ) : Prop :=
∃ (triangle: Triangle) (circle: Circle),
is_isosceles triangle ∧ 
is_circumscribed circle triangle ∧ 
triangle.base_angle = x

noncomputable def chord_intersects_triangle_probability (x: ℝ) : Prop :=
∃ (circle: Circle),
probability_of_chord_intersecting_triangle circle x = 14/25

theorem find_x_difference :
∀ (x1 x2: ℝ), 
isosceles_triangle_circumcircle x1 ∧ 
isosceles_triangle_circumcircle x2 ∧ 
chord_intersects_triangle_probability x1 ∧ 
chord_intersects_triangle_probability x2 → 
abs(x1 - x2) = 23.6643 :=
begin 
  sorry 
end

end find_x_difference_l35_35532


namespace distance_upstream_l35_35892

variable (v : ℝ) -- speed of the stream in km/h
variable (t : ℝ := 6) -- time of each trip in hours
variable (d_down : ℝ := 24) -- distance for downstream trip in km
variable (u : ℝ := 3) -- speed of man in still water in km/h

/- The distance the man swam upstream -/
theorem distance_upstream : 
  24 = (u + v) * t → 
  ∃ (d_up : ℝ), 
    d_up = (u - v) * t ∧
    d_up = 12 :=
by
  sorry

end distance_upstream_l35_35892


namespace shaded_area_l35_35317

-- Definition for the conditions provided in the problem
def side_length := 6
def area_square := side_length ^ 2
def area_square_unit := area_square * 4

-- The problem and proof statement
theorem shaded_area (sl : ℕ) (asq : ℕ) (nsq : ℕ):
    sl = 6 ∧
    asq = sl ^ 2 ∧
    nsq = asq * 4 →
    nsq - (4 * (sl^2 / 2)) = 72 :=
by
  sorry

end shaded_area_l35_35317
