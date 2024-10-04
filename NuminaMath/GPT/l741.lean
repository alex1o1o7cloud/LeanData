import Mathlib

namespace fewerSevensCanProduce100_l741_741911

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l741_741911


namespace num_twodigit_multiples_of_eight_l741_741690

theorem num_twodigit_multiples_of_eight : 
  let is_twodigit := λ n : ℕ, n >= 10 ∧ n <= 99 in
  let is_multiple_of_8 := λ n : ℕ, n % 8 = 0 in
  (finset.filter (λ n, is_twodigit n ∧ is_multiple_of_8 n) (finset.range 100)).card = 11 := 
by {
  sorry
}

end num_twodigit_multiples_of_eight_l741_741690


namespace permutation_qu_in_equation_l741_741814

theorem permutation_qu_in_equation : 
  ∃ (S : Finset (List Char)), 
    (S.card = 5) ∧ 
    (∀ x ∈ S, x ∈ (Finset.ofList ['e', 'q', 'u', 'a', 't', 'i', 'o', 'n'])) ∧ 
    ('q' ∈ ('q' :: 'u' :: [])) ∧ 
    permutation.length_eq 480 := 
sorry

end permutation_qu_in_equation_l741_741814


namespace inequality_solution_set_l741_741653

theorem inequality_solution_set (a b c : ℝ)
  (h1 : ∀ x, (ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, ¬ (bx + c > 0 ↔ x > 6)) ∧ (∀ x, (cx^2 + bx + a < 0 ↔ -1/3 < x ∧ x < 1/2)) :=
by
  sorry

end inequality_solution_set_l741_741653


namespace Fourier_transform_correct_l741_741579

noncomputable def Fourier_inverse (F : ℂ → ℂ) : ℝ → ℂ :=
  λ x, (1 / real.sqrt (2 * real.pi)) * ∫ p in (-∞), (∞), F p * Complex.exp (-Complex.I * p * x)

def F (p : ℂ) : ℂ := -Complex.I * p * Complex.log ((1 + p^2) / p^2)

def f (x : ℝ) : ℂ := real.sqrt (2 * real.pi) * (Complex.exp (-|x|) * (|x| + 1) - 1) / x^2 * Complex.sign x

theorem Fourier_transform_correct :
  Fourier_inverse F x = f x :=
by
  sorry

end Fourier_transform_correct_l741_741579


namespace triangle_problem_l741_741737

variable {Point Line Triangle : Type}
variable [Geometry Point Line Triangle]

-- Conditions from the problem
variables (A B C D E F G M N : Point)
variables (triangle_ABC : Triangle)
variables (AD BC AB AC : Line)
variables (EF AD_BG DF_CG DE DF : Line)

-- Corresponding conditions
axiom AD_bisects_angle_BAC : bisects_line AD (angle_of_triangle triangle_ABC A B C)
axiom AD_intersects_BC_at_D : intersects AD BC D
axiom DE_bisects_angle_ADB : bisects_line DE (angle_at_point D A B)
axiom DE_intersects_AB_at_E : intersects DE AB E
axiom DF_bisects_angle_ADC : bisects_line DF (angle_at_point D A C)
axiom DF_intersects_AC_at_F : intersects DF AC F
axiom EF_intersects_AD_at_G : intersects EF AD G
axiom BG_intersects_DF_at_M : intersects BG DF M
axiom CG_intersects_DE_at_N : intersects CG DE N

-- Statement to prove
axiom points_collinear : collinear M A N
axiom line_perpendicular : perpendicular (line_through M N) AD

theorem triangle_problem (triangle_ABC : Triangle) (A B C D E F G M N : Point)
  (AD BC AB AC : Line) (EF AD_BG DF_CG DE DF : Line)
  (AD_bisects_angle_BAC : bisects_line AD (angle_of_triangle triangle_ABC A B C))
  (AD_intersects_BC_at_D : intersects AD BC D)
  (DE_bisects_angle_ADB : bisects_line DE (angle_at_point D A B))
  (DE_intersects_AB_at_E : intersects DE AB E)
  (DF_bisects_angle_ADC : bisects_line DF (angle_at_point D A C))
  (DF_intersects_AC_at_F : intersects DF AC F)
  (EF_intersects_AD_at_G : intersects EF AD G)
  (BG_intersects_DF_at_M : intersects BG DF M)
  (CG_intersects_DE_at_N : intersects CG DE N) :
  collinear M A N ∧ perpendicular (line_through M N) AD :=
by 
  sorry

end triangle_problem_l741_741737


namespace bailey_chew_toys_l741_741553

theorem bailey_chew_toys (dog_treats rawhide_bones: ℕ) (cards items_per_card : ℕ)
  (h1 : dog_treats = 8)
  (h2 : rawhide_bones = 10)
  (h3 : cards = 4)
  (h4 : items_per_card = 5) :
  ∃ chew_toys : ℕ, chew_toys = 2 :=
by
  sorry

end bailey_chew_toys_l741_741553


namespace like_terms_sum_l741_741693

theorem like_terms_sum (m n : ℕ) (a b : ℝ) 
  (h₁ : 5 * a^m * b^3 = 5 * a^m * b^3) 
  (h₂ : -4 * a^2 * b^(n-1) = -4 * a^2 * b^(n-1)) 
  (h₃ : m = 2) (h₄ : 3 = n - 1) : m + n = 6 := by
  sorry

end like_terms_sum_l741_741693


namespace count_perfect_fourth_powers_l741_741301

theorem count_perfect_fourth_powers: 
  ∃ n_count: ℕ, n_count = 4 ∧ ∀ n: ℕ, (50 ≤ n^4 ∧ n^4 ≤ 2000) → (n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) :=
by {
  sorry
}

end count_perfect_fourth_powers_l741_741301


namespace triangle_cannot_have_two_right_angles_l741_741410

theorem triangle_cannot_have_two_right_angles (A B C : ℝ) (h : A + B + C = 180) : 
  ¬ (A = 90 ∧ B = 90) :=
by {
  sorry
}

end triangle_cannot_have_two_right_angles_l741_741410


namespace probability_jack_and_jill_chosen_l741_741497

open_locale classical  -- Enable classical logic

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def combination (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

noncomputable def probability_of_jack_and_jill (total_workers workers_chosen : ℕ) : ℚ :=
let total_combinations := combination total_workers workers_chosen in
let favorable_combinations := 1 in
favorable_combinations / total_combinations

theorem probability_jack_and_jill_chosen :
  probability_of_jack_and_jill 4 2 = 1 / 6 :=
by sorry

end probability_jack_and_jill_chosen_l741_741497


namespace minimum_throws_to_ensure_same_sum_twice_l741_741009

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741009


namespace a_div_c_le_n_plus_1_l741_741827

-- Definition of the polynomials and the conditions
variables {R : Type*} [linear_ordered_field R]

-- Define the polynomials f and g
noncomputable def f (a : fin (n + 1) → R) (x : R) : R :=
  ∑ i in finset.range (n + 1), a i * x ^ i

noncomputable def g (c : fin (n + 2) → R) (x : R) : R :=
  ∑ i in finset.range (n + 2), c i * x ^ i

-- Maximum absolute value of coefficients
def max_abs (coeffs : fin (n + k) → R) : R :=
  finset.max' (finset.univ.image (λ i, abs (coeffs i))) sorry

variables (a : fin (n + 1) → R) (c : fin (n + 2) → R) (r : R)

def g_eq_x_minus_r_f (f : R → R) : Prop :=
  ∀ x, g c x = (x - r) * f x

-- The main statement to prove
theorem a_div_c_le_n_plus_1 (h : g_eq_x_minus_r_f (f a)) :
  max_abs a / max_abs c ≤ n + 1 :=
sorry

end a_div_c_le_n_plus_1_l741_741827


namespace hexagon_covering_problem_l741_741215

noncomputable def countRhombusCoverings (hexagon : Type) [Field hexagon] : ℕ := 20

theorem hexagon_covering_problem
  (hexagon : Type) [Field hexagon]
  (rhombus : Type) [Inhabited rhombus]
  (equilateral_triangle : Type) [Inhabited equilateral_triangle]
  (hexagon_composed_of: List (List (equilateral_triangle)) → Prop)
  (hexagon24: hexagon_composed_of (List.replicate 24 (default equilateral_triangle)))
  (rhombus_composed_of: List (equilateral_triangle) → Prop)
  (rhombus2: rhombus_composed_of (List.replicate 2 (default equilateral_triangle)))
  (coverHexagon : hexagon_composed_of (hexagon24) → List (List (rhombus)) → Prop)
  (cover12 : coverHexagon hexagon24 (List.replicate 12 (replicate 2 (default rhombus)))) :
  countRhombusCoverings hexagon = 20 :=
by
  sorry

end hexagon_covering_problem_l741_741215


namespace infinitely_many_fibonacci_divisible_by_n_l741_741626

noncomputable def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem infinitely_many_fibonacci_divisible_by_n (N : ℕ) (hN : 0 < N) :
  ∃ (A : ℕ → ℕ), (∀ n : ℕ, A n = fibonacci n) ∧ (∀ k : ℕ, ∃ m : ℕ, m > k ∧ N ∣ fibonacci m) := 
sorry

end infinitely_many_fibonacci_divisible_by_n_l741_741626


namespace subset_properties_l741_741426

def isLatticePoint (p : ℤ × ℤ × ℤ) : Prop := 
  true

def isNeighbor (p q : ℤ × ℤ × ℤ) : Prop :=
  (p.1 = q.1 ∧ p.2 = q.2 ∧ abs (p.3 - q.3) = 1) ∨
  (p.1 = q.1 ∧ abs (p.2 - q.2) = 1 ∧ p.3 = q.3) ∨
  (abs (p.1 - q.1) = 1 ∧ p.2 = q.2 ∧ p.3 = q.3)

noncomputable def S : set (ℤ × ℤ × ℤ) := 
  {p | 2 * p.1 + 4 * p.2 + 6 * p.3 ≡ 0 [MOD 7]}

theorem subset_properties (p : ℤ × ℤ × ℤ):
  (p ∈ S → ∀ q, isNeighbor p q → q ∉ S) ∧
  (p ∉ S → ∃! q, isNeighbor p q ∧ q ∈ S) :=
by sorry

end subset_properties_l741_741426


namespace noelle_homework_assignments_l741_741790

theorem noelle_homework_assignments : 
  (let 
    points_gr1 := 7 * 1, 
    points_gr2 := 7 * 2, 
    points_gr3 := 7 * 3, 
    points_gr4 := 7 * 4 in 
  points_gr1 + points_gr2 + points_gr3 + points_gr4 = 70) :=
by sorry

end noelle_homework_assignments_l741_741790


namespace more_broken_spiral_shells_l741_741682

theorem more_broken_spiral_shells (perc1_perfect_spiral: ℝ) (perc1_broken_spiral: ℝ)
  (perc2_perfect_spiral: ℝ) (perc2_broken_spiral: ℝ) 
  (perc3_perfect_spiral: ℝ) (perc3_broken_spiral: ℝ)
  (beach1_total_perfect: ℕ) (beach1_total_broken: ℕ)
  (beach2_total_perfect: ℕ) (beach2_total_broken: ℕ)
  (beach3_total_perfect: ℕ) (beach3_total_broken: ℕ)
  (round1_spiral_broken: ℕ) (round2_spiral_broken: ℕ) (round3_spiral_broken: ℕ)
  (round1_spiral_perfect: ℕ) (round2_spiral_perfect: ℕ) (round3_spiral_perfect: ℕ) :
  perc1_perfect_spiral = 0.40 ∧ perc1_broken_spiral = 0.30 ∧
  perc2_perfect_spiral = 0.20 ∧ perc2_broken_spiral = 0.25 ∧
  perc3_perfect_spiral = 0.35 ∧ perc3_broken_spiral = 0.40 ∧
  beach1_total_perfect = 20 ∧ beach1_total_broken = 60 ∧
  beach2_total_perfect = 35 ∧ beach2_total_broken = 95 ∧
  beach3_total_perfect = 25 ∧ beach3_total_broken = 45 ∧
  round1_spiral_broken = int.floor((60 : ℝ) * 0.3) ∧
  round2_spiral_broken = int.floor((95 : ℝ) * 0.25) ∧
  round3_spiral_broken = int.floor((45 : ℝ) * 0.4) ∧
  round1_spiral_perfect = int.floor((20 : ℝ) * 0.4) ∧
  round2_spiral_perfect = int.floor((35 : ℝ) * 0.2) ∧
  round3_spiral_perfect = int.floor((25 : ℝ) * 0.35) →
  (round1_spiral_broken + round2_spiral_broken + round3_spiral_broken) - 
  (round1_spiral_perfect + round2_spiral_perfect + round3_spiral_perfect) = 36 := sorry

end more_broken_spiral_shells_l741_741682


namespace root_in_interval_l741_741453

theorem root_in_interval : 
  (∃ x, (1 < x ∧ x < 2) ∧ (log (2 * x) + x - 2 = 0)) →
  (∃ k : ℤ, k = 2) :=
by
  sorry

end root_in_interval_l741_741453


namespace min_value_sqrt_expr_l741_741759

theorem min_value_sqrt_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  \(\frac{\sqrt{(x^2 + y^2)(4x^2 + y^2)}}{xy}\) ≥ 3 ∧ 
    (\(\frac{\sqrt{(x^2 + y^2)(4x^2 + y^2)}}{xy}\) = 3 ↔ y = x \(\sqrt{2}\)) :=
sorry

end min_value_sqrt_expr_l741_741759


namespace find_sin_phi_l741_741391

variable {u v w : ℝ^3}
variable (h1 : u ≠ 0 ∧ v ≠ 0 ∧ w ≠ 0)
variable (h2 : ¬ ∃ c1 c2, (u = c1 • v ∧ w = c2 • v) ∨ (v = c1 • u ∧ w = c2 • u) ∨ (w = c1 • u ∧ v = c2 • u))
variable (h3 : (u × v) × w = (1 / 4) * ‖v‖ * ‖w‖ • u)

theorem find_sin_phi : let φ := Real.arccos (-(1 / 4)) in Real.sin φ = sqrt (15) / 4 := 
by
  let φ := Real.arccos (-(1 / 4))
  sorry

end find_sin_phi_l741_741391


namespace part1_part2_l741_741671

variable (a : ℝ)
def inequality_1 := ∀ x : ℝ, x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0
def A (a : ℝ) := {x : ℝ | inequality_1 a x}
def B := Ioo (-2 : ℝ) 2

theorem part1 (h : a = 2) : (A 2) ∪ B = Ioo (-2 : ℝ) 3 := by
  sorry

theorem part2 (h2 : A a ∩ B = ∅) : (a ≤ -3 ∨ a ≥ 2) := by
  sorry

end part1_part2_l741_741671


namespace minimum_rolls_for_duplicate_sum_l741_741063

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741063


namespace minimum_value_of_c_l741_741801

theorem minimum_value_of_c {a b c : ℕ} (h1 : a < b) (h2 : b < c)
    (h3 : ∀ y, ∃! x, 2 * x + y = 2003 ∧ y = abs (x - a) + abs (x - b) + abs (x - c)) :
    c = 1002 := sorry

end minimum_value_of_c_l741_741801


namespace b_money_used_for_10_months_l741_741933

theorem b_money_used_for_10_months
  (a_capital_ratio : ℚ)
  (a_time_used : ℕ)
  (b_profit_share : ℚ)
  (h1 : a_capital_ratio = 1 / 4)
  (h2 : a_time_used = 15)
  (h3 : b_profit_share = 2 / 3) :
  ∃ (b_time_used : ℕ), b_time_used = 10 :=
by
  sorry

end b_money_used_for_10_months_l741_741933


namespace minimum_throws_to_ensure_same_sum_twice_l741_741012

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741012


namespace December_revenue_times_average_l741_741701

-- Conditions
variable (D : ℝ) -- Revenue in December
def revenue_Nov := (2 / 5) * D -- Revenue in November
def revenue_Jan := (1 / 3) * revenue_Nov D -- Revenue in January

-- Definition of the average revenue
def average_revenue := (revenue_Nov D + revenue_Jan D) / 2

-- Theorem stating the problem to be proved
theorem December_revenue_times_average :
  D = 5 * average_revenue D :=
by sorry

end December_revenue_times_average_l741_741701


namespace red_blue_seg_eq_l741_741720

-- Define the conditions
def grid := fin 100 × fin 100
def is_red (g : grid → Prop) : Prop := ∀ i j, (∃ f : grid, g f ∧ i≠j ∧ f=i ∧ ∃ (c : fin 2), ∃ (half_red_half_blue : fin 2), f= (i,j) ∧ is_red (half_red_half_blue)  c) = 50
def is_blue (g : grid → Prop) : Prop := ∀ i j, (∃ f : grid, g f ∧ i≠j ∧ f=i ∧ ∃ (c : fin 2), ∃ (half_red_half_blue : fin 2), f= (i,j) ∧ is_blue (half_red_half_blue) c) = 50

-- Define the adjacency condition
def adjacent (a b : grid) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

-- Define the problem statement
theorem red_blue_seg_eq (g : grid → Prop) (hr : is_red g) (hb : is_blue g) :
  (finset.univ.filter (λ a b, g a ∧ g b ∧ is_red g a ∧ is_red g b ∧ adjacent a b)).card =
  (finset.univ.filter (λ a b, g a ∧ g b ∧ is_blue g a ∧ is_blue g b ∧ adjacent a b)).card := sorry

end red_blue_seg_eq_l741_741720


namespace strawberries_in_crate_l741_741460

theorem strawberries_in_crate (total_fruit : ℕ) (fruit_in_crate : total_fruit = 78) (kiwi_fraction : ℚ) (kiwi_fraction_eq : kiwi_fraction = 1/3) :
  let kiwi_count := total_fruit * kiwi_fraction,
      strawberry_count := total_fruit - kiwi_count
  in 
  strawberry_count = 52 :=
by
  sorry

end strawberries_in_crate_l741_741460


namespace min_value_sin6_cos6_l741_741587

open Real

theorem min_value_sin6_cos6 (x : ℝ) : sin x ^ 6 + 2 * cos x ^ 6 ≥ 2 / 3 :=
by
  sorry

end min_value_sin6_cos6_l741_741587


namespace three_operations_final_state_probability_five_red_four_blue_l741_741169

/-- Define the initial state of the urn with two red balls and one blue ball. --/
def initial_urn : (ℕ × ℕ) := (2, 1)

/-- Define the addition of two balls of the same color
    to the urn after drawing a ball of that color. --/
def add_balls (color : Bool) (r b : ℕ) : (ℕ × ℕ) :=
  if color = tt then (r + 2, b) else (r, b + 2)

/-- Define one draw operation which picks a ball and updates the urn state. --/
noncomputable def draw_and_add (r b : ℕ) : Prop :=
  a = (2, 1) ∧ (r + b = 3 ∧ draw_and_add r alive) ∧ (a = count.r)

example (urn : ℕ × ℕ) : Prop :=
  draw_and_add urn = 5 Interview
  
/-- Prove that the final state after three operations is five red balls and four blue balls. --/
theorem three_operations_final_state :
  ∃ (r b : ℕ), (r + b = 9) ∧ (urn = (2, 1) ∧ 5 = blue.r[urn].(χινω)) ∧ (draw_and_add urn.ra alive) :
     pr.r[χινων] := (r,b)

end

/-- Define the probability calculation for a specific sequence of draws. --/
noncomputable def sequence_probability (seq : List Bool) : ℚ :=
sorry

/-- Prove that the probability of having exactly five red balls and four blue balls
    after three operations is 3/10.
--/
theorem probability_five_red_four_blue :
  (∃ (seqs : List (List Bool)), (urn = (2, 1) ∧ sum 3 = blue ∧ 9 = state[seqs].final) ∧ pr (3 / 10) := begin
    sorry
  end) :
routines. urn ba.done.pro
five
nballs)
black balls

end three_operations_final_state_probability_five_red_four_blue_l741_741169


namespace minimum_number_of_good_permutations_l741_741627

def is_good_permutation {n : ℕ} (a : Fin n → ℝ) (b : Fin n → ℝ) : Prop :=
  ∀ k : Fin n, 0 < (Finset.univ.filter (fun i => i < k.val)).sum b

noncomputable def minimum_good_permutations (n : ℕ) (a : Fin n → ℝ) : ℕ :=
  (n - 1)!

theorem minimum_number_of_good_permutations (n : ℕ) (a : Fin n → ℝ) (h₁ : n ≥ 3)
  (h₂ : Function.injective a) (h₃ : 0 < (Finset.univ).sum a) :
  ∃ p : Finset (Fin n → ℝ), (∀ x ∈ p, is_good_permutation a x) ∧ p.card = minimum_good_permutations n a :=
sorry

end minimum_number_of_good_permutations_l741_741627


namespace find_k_l741_741385

def arithmetic_sequence (b : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n m, b (n + 1) = b 1 + n * d

theorem find_k
  (b: ℕ → ℕ)
  (d: ℕ)
  (h_seq: arithmetic_sequence b d)
  (h_1: b 3 + b 6 + b 9 = 21)
  (h_2: b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 + b 10 + b 11 + b 12 = 110)
  (h_k: b (9 + 7) = 16) :
  9 = 9 := 
sorry

end find_k_l741_741385


namespace magazine_purchasing_methods_l741_741128

theorem magazine_purchasing_methods :
  ∃ n : ℕ, n = 266 ∧ 
  (∃ (m2 m1 : Finset Nat), 
    (m2.card = 8 ∧ 
     m1.card = 3 ∧
     ∃ (chosen2 chosen1 : Finset Nat), 
       (chosen2 ⊆ m2 ∧ 
        chosen1 ⊆ m1 ∧ 
        (chosen2.card * 2 + chosen1.card) = 10 ∧ 
        ((chosen2.card = 5 ∧ chosen1.card = 0) ∨ 
         (chosen2.card = 4 ∧ chosen1.card = 2 ∧ chosen2.card.choose 4 * chosen1.card.choose 2 = 210)
        )
       )
    )
  )
: sorry

end magazine_purchasing_methods_l741_741128


namespace min_throws_to_ensure_same_sum_twice_l741_741076

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741076


namespace number_of_valid_sets_eq_3_l741_741446

def valid_sets (a b c : Type) : Set (Set Type) :=
  {P | {a} ⊂ P ∧ P ⊆ {a, b, c}}

theorem number_of_valid_sets_eq_3 (a b c : Type) :
  (valid_sets a b c).card = 3 :=
by
  sorry

end number_of_valid_sets_eq_3_l741_741446


namespace number_of_students_in_class_l741_741833

theorem number_of_students_in_class:
  ∀ (n : ℕ), 
  (∀ (students_age : ℕ), students_age = 22 * n) ∧
  (∀ (teacher_age : ℕ), teacher_age = 46) ∧
  (∀ (new_average : ℕ), new_average = 23) →
  23 = ∀ n, students_age + teacher_age = new_average * (n + 1) → n = 23
:= sorry

end number_of_students_in_class_l741_741833


namespace person_B_processes_components_l741_741799

theorem person_B_processes_components (x : ℕ) (h1 : ∀ x, x > 0 → x + 2 > 0) 
(h2 : ∀ x, x > 0 → (25 / (x + 2)) = (20 / x)) :
  x = 8 := sorry

end person_B_processes_components_l741_741799


namespace seven_expression_one_seven_expression_two_l741_741898

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l741_741898


namespace seven_expression_one_seven_expression_two_l741_741902

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l741_741902


namespace length_MN_eq_semiperimeter_l741_741236

-- Definitions based on the problem conditions
variables (A B C M N : Point) (triangle_ABC : Triangle)
variables (AM AN : Line)
variables [IsPerpendicularTo AM (ExteriorAngleBisector B)] [IsPerpendicularTo AN (ExteriorAngleBisector C)]

-- Theorem statement
theorem length_MN_eq_semiperimeter (hM : Perpendicular AM (bisector (ExteriorAngle B)))
                                    (hN : Perpendicular AN (bisector (ExteriorAngle C)))
                                    (h : AM contained_in triangle_ABC ∧ AN contained_in triangle_ABC) : 
  length (Segment M N) = semiperimeter triangle_ABC :=
sorry

end length_MN_eq_semiperimeter_l741_741236


namespace min_throws_to_ensure_same_sum_twice_l741_741067

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741067


namespace log_base_2_a16_l741_741245

theorem log_base_2_a16 (q : ℝ) (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) (h2 : q = real.cbrt 2) (h3 : a 3 * a 11 = 16) : 
  real.log 2 (a 16) = 5 :=
sorry

end log_base_2_a16_l741_741245


namespace minimum_value_f_l741_741620

def f (x : ℝ) := x + 1 / (x - 2)

theorem minimum_value_f (h : ∀ x > (2 : ℝ), f x ≥ 4) : ∃ x, x > 2 ∧ f x = 4 :=
by
  sorry

end minimum_value_f_l741_741620


namespace graph_shift_l741_741266

theorem graph_shift (f : ℝ → ℝ) (h : f 0 = 2) : f (-1 + 1) = 2 :=
by
  have h1 : f 0 = 2 := h
  sorry

end graph_shift_l741_741266


namespace pencils_count_l741_741110

theorem pencils_count (P L : ℕ) 
  (h1 : P * 6 = L * 5) 
  (h2 : L = P + 7) : 
  L = 42 :=
by
  sorry

end pencils_count_l741_741110


namespace production_rates_are_correct_probability_of_selecting_two_grade_eight_l741_741514

variable (ξ : ℕ) -- Grade coefficient
variable (sample : List ℕ) -- The sample grades

-- Sample data given
def sample_data : List ℕ := [
  3, 5, 3, 3, 8, 5, 5, 6, 3, 4, 
  6, 3, 4, 7, 5, 3, 4, 8, 5, 3, 
  8, 3, 4, 3, 4, 4, 7, 5, 6, 7
]

-- Definition of grade categories
def is_premium (ξ : ℕ) : Prop := ξ ≥ 7
def is_second_grade (ξ : ℕ) : Prop := ξ ≥ 5 ∧ ξ < 7
def is_third_grade (ξ : ℕ) : Prop := ξ ≥ 3 ∧ ξ < 5

-- Prove the production rates for each grade
theorem production_rates_are_correct : 
  (sample.countp is_premium sample_data) / sample.size = 0.2 ∧
  (sample.countp is_second_grade sample_data) / sample.size = 0.3 ∧
  (sample.countp is_third_grade sample_data) / sample.size = 0.5 :=
sorry

-- Definition for the probability calculation
def probability_of_eight_from_premium (sample : List ℕ) : ℚ :=
  (sample.filter is_premium).countp (λ ξ, ξ = 8) choose 2 / 
  (sample.filter is_premium).length choose 2

-- Prove the probability of selecting two grade 8 products from premium
theorem probability_of_selecting_two_grade_eight : 
  probability_of_eight_from_premium sample_data = 1 / 5 :=
sorry

end production_rates_are_correct_probability_of_selecting_two_grade_eight_l741_741514


namespace solve_equations_l741_741420

theorem solve_equations (Ax Axsq Axp1sq C8x : ℕ → ℕ) :
  (∀ x, 3 * Ax x^3 = 2 * Ax (x + 1)^2 + 6 * Ax x^2 ∧ Ax x = x * (x - 1) * (x - 2)) →
  (∀ x, C8x x = nat.choose 8 x ∧ ∀ x, C8x x = C8x (5 * x - 4)) →
  (∃ x, Ax x = 5) ∧ (∃ x, C8x x = 1 ∨ C8x x = 2) :=
by
  sorry

end solve_equations_l741_741420


namespace correct_option_is_D_l741_741925

theorem correct_option_is_D :
  (-3 - (-4) = -7) → (-2 * (-3) = -6) → (6 / (-1/2 + 1/3) = 6) → (-10 / 5 = -2) → Option D is correct.
  by
    intros hA hB hC hD
    sorry

end correct_option_is_D_l741_741925


namespace minimum_throws_to_ensure_same_sum_twice_l741_741005

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741005


namespace solution_l741_741286

noncomputable def problem_statement (φ : ℝ) (k : ℤ) (hφ1 : 0 < φ) (hφ2 : φ < π)
  (h_symmetry : ∀ x : ℝ, (sin (π * x + φ) - 2 * cos (π * x + φ)) = 
    sin (π * (2 * 1 - x) + φ) - 2 * cos (π * (2 * 1 - x) + φ)) : Prop :=
  sin (2 * φ) = -4 / 5

theorem solution (φ : ℝ) (k : ℤ) (hφ1 : 0 < φ) (hφ2 : φ < π) 
  (h_symmetry : ∀ x : ℝ, (sin (π * x + φ) - 2 * cos (π * x + φ)) = 
    sin (π * (2 * 1 - x) + φ) - 2 * cos (π * (2 * 1 - x) + φ)) : 
  problem_statement φ k hφ1 hφ2 h_symmetry := sorry

end solution_l741_741286


namespace minimum_value_expression_l741_741761

theorem minimum_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  3 ≤ (sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x * y) :=
by
  sorry

end minimum_value_expression_l741_741761


namespace minimum_throws_l741_741020

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741020


namespace decreasing_function_range_a_l741_741262

theorem decreasing_function_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (1 / 6) ≤ a ∧ a < 1 / 3 :=
by 
  let f := λ a x, if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x
  sorry

end decreasing_function_range_a_l741_741262


namespace tina_days_to_use_pink_pens_tina_total_pens_l741_741468

-- Definitions based on the problem conditions.
def pink_pens : ℕ := 15
def green_pens : ℕ := pink_pens - 9
def blue_pens : ℕ := green_pens + 3
def total_pink_green := pink_pens + green_pens
def yellow_pens : ℕ := total_pink_green - 5
def pink_pens_per_day := 4

-- Prove the two statements based on the definitions.
theorem tina_days_to_use_pink_pens 
  (h1 : pink_pens = 15)
  (h2 : pink_pens_per_day = 4) :
  4 = 4 :=
by sorry

theorem tina_total_pens 
  (h1 : pink_pens = 15)
  (h2 : green_pens = pink_pens - 9)
  (h3 : blue_pens = green_pens + 3)
  (h4 : yellow_pens = total_pink_green - 5) :
  pink_pens + green_pens + blue_pens + yellow_pens = 46 :=
by sorry

end tina_days_to_use_pink_pens_tina_total_pens_l741_741468


namespace seven_expression_one_seven_expression_two_l741_741899

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l741_741899


namespace minimum_throws_to_ensure_same_sum_twice_l741_741017

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741017


namespace largest_n_for_sum_of_fourth_powers_l741_741220

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

/-- The main theorem to be proven. -/
theorem largest_n_for_sum_of_fourth_powers (n : ℕ) 
    (h : n = 240) : 
    (∀ (p : ℕ), p > 10 → is_prime p → ∃ k : ℕ, (p^4 - 1) * k = 240 * any_sum_of_primes n) := sorry

end largest_n_for_sum_of_fourth_powers_l741_741220


namespace largest_fraction_l741_741633

theorem largest_fraction (a b c d : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < 5) :
  (∀ x y, x ∈ {(a + c) / (b + d), (b + d) / (c + a), (c + a) / (d + b), (d + b) / (a + c), (d + c) / (a + b)} →
          y ∈ {(a + c) / (b + d), (b + d) / (c + a), (c + a) / (d + b), (d + b) / (a + c), (d + c) / (a + b)} →
          y ≤ x → x = (d + c) / (a + b)) := 
begin
  sorry
end

end largest_fraction_l741_741633


namespace normal_distribution_properties_l741_741269

noncomputable def P (X : ℝ → ℝ) : ℝ → ℝ := sorry -- Assume a function for probability

variable (X : ℝ → ℝ)
def f (x : ℝ) : ℝ := P (X ≤ x)

-- Statement: Given that the random variable X follows N(0,1)
-- Prove the following:
theorem normal_distribution_properties (x : ℝ) (hx : x > 0) :
  (f (-x) = 1 - f x) ∧
  (∀ y : ℝ, y > 0 → f y < f (y+1)) ∧
  (P (abs X ≤ x) = 2 * f x - 1) :=
  sorry

end normal_distribution_properties_l741_741269


namespace third_side_length_l741_741318

theorem third_side_length (x : ℝ) (h1 : 2 + 4 > x) (h2 : 4 + x > 2) (h3 : x + 2 > 4) : x = 4 :=
by {
  sorry
}

end third_side_length_l741_741318


namespace expr1_eq_result_expr2_eq_result_l741_741182

-- Define the mathematical constants and operations
noncomputable def expr1 : ℝ :=
  0.027^(1/3) - (-1/7)^(-2) + 2.56^(3/4) - 3^(-1) + (Real.sqrt 2 - 1)^0

noncomputable def expr1_result : ℝ :=
  -(1471 - 48 * Real.sqrt 10) / 30

-- State the proof problem for expression 1
theorem expr1_eq_result : expr1 = expr1_result := 
by sorry

-- Define the mathematical constants and operations for log expression
noncomputable def expr2 : ℝ :=
  (Real.log 8 + Real.log 125 - Real.log 2 - Real.log 5) / (Real.log (Real.sqrt 10) * Real.log 0.1)

-- State the correct result for expr2
noncomputable def expr2_result : ℝ := -4

-- State the proof problem for expression 2
theorem expr2_eq_result : expr2 = expr2_result :=
by sorry

end expr1_eq_result_expr2_eq_result_l741_741182


namespace points_in_triangular_regions_l741_741354

variables {n : ℕ} (P : fin n → point) 
variables (Q : fin (n - 2) → point)
hypothesis (convex_P : is_convex P)
hypothesis (condition : ∀ (i j k : fin n) (h1 : i ≠ j) (h2 : j ≠ k) (h3 : i ≠ k), 
    ∃ l, l < n - 2 ∧ Q l ∈ triangle (P i) (P j) (P k) ∧ 
        (∀ m < n - 2, Q m ∈ triangle (P i) (P j) (P k) → m = l))

theorem points_in_triangular_regions : 
∀ l < n - 2, ∃ (i j k : fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    Q l ∈ triangle (P i) (P j) (P k) := 
by
sorry

end points_in_triangular_regions_l741_741354


namespace fewerSevensCanProduce100_l741_741915

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l741_741915


namespace sum_27_probability_l741_741955

-- Define the range of faces for the first and second dice
def die1_faces := {x : ℕ | 1 ≤ x ∧ x ≤ 19}
def die2_faces := {x : ℕ | (1 ≤ x ∧ x ≤ 7) ∨ (9 ≤ x ∧ x ≤ 21)}

-- Define the number of possible outcomes for the dice pair
def total_outcomes := 20 * 20  -- Since both dice are 20-faced

-- Define a function to count valid pairs (d1, d2) where d1 + d2 = 27
def valid_pairs : Nat :=
  (die1_faces.to_finset.product die2_faces.to_finset).filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 27).card

-- Define the probability of rolling a sum of 27
def probability : ℚ := valid_pairs / total_outcomes

theorem sum_27_probability :
  probability = 3 / 100 :=
by
  sorry

end sum_27_probability_l741_741955


namespace example_of_equal_sigma1_diff_nat_equal_prod_divisors_imp_eq_l741_741934

-- Define sum of positive divisors function
def sigma1 (n : ℕ) : ℕ := ∑ d in (Finset.divisors n), d

-- Define number of positive divisors function
def sigma0 (n : ℕ) : ℕ := (Finset.divisors n).card

-- Define product of positive divisors function
def prod_divisors (n : ℕ) : ℕ := n^(sigma0 n / 2)

-- Problem (a) statement
theorem example_of_equal_sigma1_diff_nat :
  sigma1 14 = sigma1 15 := by
  sorry

-- Problem (b) statement
theorem equal_prod_divisors_imp_eq (n m : ℕ) :
  prod_divisors n = prod_divisors m → n = m := by
  sorry

end example_of_equal_sigma1_diff_nat_equal_prod_divisors_imp_eq_l741_741934


namespace arrangement_count_l741_741466

def male_students : List String := ["A", "B", "C"]
def female_students : List String:= ["D", "E", "F"]

/-- There are three male students and three female students, a total of six students, stand in a row.
    Male student A does not stand at either end, and exactly two of the three female students stand next to each other. -/
def valid_arrangement_count : ℕ := 288

theorem arrangement_count :
  ∃ arr : List String, ∀ arr ∈ List.permutations (male_students ++ female_students),
  (arr.head ≠ "A") ∧ (arr.last ≠ "A") ∧
  (exists_adj_females : ∃ i, 0 ≤ i ∧ i < 5 ∧ (arr.nth i ∈ female_students) ∧ (arr.nth (i + 1) ∈ female_students)) →
  List.countp (λ x, x ∈ female_students) arr = 3 ∧
  List.countp (λ x, x ∈ male_students) arr = 3 :=
  ⟨valid_arrangement_count, sorry⟩

end arrangement_count_l741_741466


namespace find_a_plus_k_l741_741289

theorem find_a_plus_k (a k : ℝ) (f : ℝ → ℝ) (h : f = λ x, (a-1)*x^k) (hc1 : f (real.sqrt 2) = 2) : 
  a + k = 4 :=
sorry

end find_a_plus_k_l741_741289


namespace quadratic_inequality_solution_l741_741450

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - (3 / 8) < 0) ↔ (-3 < k ∧ k < 0) :=
sorry

end quadratic_inequality_solution_l741_741450


namespace B_is_subset_of_A_l741_741944
open Set

def A := {x : ℤ | ∃ n : ℤ, x = 2 * n}
def B := {y : ℤ | ∃ k : ℤ, y = 4 * k}

theorem B_is_subset_of_A : B ⊆ A :=
by sorry

end B_is_subset_of_A_l741_741944


namespace integer_solutions_compare_x_y_l741_741101

-- Part (a)
theorem integer_solutions (m n : ℤ) : 9 * m^2 + 3 * n = n^2 + 8 ↔ 
  (m = 2 ∧ (n = 7 ∨ n = -4)) ∨ 
  (m = -2 ∧ (n = 7 ∨ n = -4)) := 
by 
  sorry

-- Part (b)
theorem compare_x_y (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let x := a^(a + b) + (a + b)^a
  let y := a^a + (a + b)^(a + b)
  in x < y :=
by 
  sorry

end integer_solutions_compare_x_y_l741_741101


namespace reciprocals_expression_eq_zero_l741_741696

theorem reciprocals_expression_eq_zero {m n : ℝ} (h : m * n = 1) : (2 * m - 2 / n) * (1 / m + n) = 0 :=
by
  sorry

end reciprocals_expression_eq_zero_l741_741696


namespace crude_oil_mixture_l741_741152

theorem crude_oil_mixture (x y : ℝ) 
  (h1 : x + y = 50)
  (h2 : 0.25 * x + 0.75 * y = 0.55 * 50) : 
  y = 30 :=
by
  sorry

end crude_oil_mixture_l741_741152


namespace minimum_rolls_for_duplicate_sum_l741_741053

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741053


namespace solve_for_y_l741_741226

theorem solve_for_y :
  (∃ y : ℝ, 10^(2 * y) * 1000^y = 100^6) ↔ ∃ y : ℝ, y = 12 / 5 :=
by
  sorry

end solve_for_y_l741_741226


namespace find_f2023_l741_741144

noncomputable def f : ℕ → ℕ 
| 1 := 2
| 2 := 3
| (n + 3) := f (n + 2) + f (n + 1) + (n + 3)

noncomputable def g : ℕ → ℕ 
| 1 := 1
| 2 := 1
| (n + 3) := g (n + 2) + g (n + 1) + 2

lemma g_f_relation (n : ℕ) : f (n + 1) = g (n + 1) + (n + 1) := by
  sorry

theorem find_f2023 : f 2023 = g 2023 + 2023 :=
by
  exact g_f_relation 2022

end find_f2023_l741_741144


namespace intersection_complement_P_Q_l741_741273

open Set

variables (x : ℝ)

def U : Set ℝ := univ

def P : Set ℝ := {x | x^2 - x - 6 ≥ 0}

def Q : Set ℝ := {x | 2^x ≥ 1}

def C_R_P : Set ℝ := U \ P

theorem intersection_complement_P_Q : C_R_P ∩ Q = {x | 0 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_complement_P_Q_l741_741273


namespace probability_of_three_of_a_kind_after_reroll_l741_741228

theorem probability_of_three_of_a_kind_after_reroll
  (X Y : Fin 6) (h_ne : X ≠ Y) (d5 : Fin 6) (h_d5_ne_X : d5 ≠ X) (h_d5_ne_Y : d5 ≠ Y) :
  let P : ℚ := 1/3 in
  P = 1/3 :=
by
  sorry

end probability_of_three_of_a_kind_after_reroll_l741_741228


namespace vasya_100_using_fewer_sevens_l741_741904

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l741_741904


namespace minimum_throws_to_ensure_same_sum_twice_l741_741019

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741019


namespace notebooks_type_A_count_minimum_profit_m_l741_741797

def total_notebooks := 350
def costA := 12
def costB := 15
def total_cost := 4800

def selling_priceA := 20
def selling_priceB := 25
def discountA := 0.7
def profit_min := 2348

-- Prove the number of type A notebooks is 150
theorem notebooks_type_A_count (x y : ℕ) (h1 : x + y = total_notebooks)
    (h2 : costA * x + costB * y = total_cost) : x = 150 := by
  sorry

-- Prove the minimum value of m is 111 such that profit is not less than 2348
theorem minimum_profit_m (m : ℕ) (profit : ℕ)
    (h : profit = (m * selling_priceA + m * selling_priceB  + (150 - m) * (selling_priceA * discountA).toNat + (200 - m) * costB - total_cost))
    (h_prof : profit >= profit_min) : m >= 111 := by
  sorry

end notebooks_type_A_count_minimum_profit_m_l741_741797


namespace total_strings_needed_l741_741365

def basses := 3
def strings_per_bass := 4
def guitars := 2 * basses
def strings_per_guitar := 6
def eight_string_guitars := guitars - 3
def strings_per_eight_string_guitar := 8

theorem total_strings_needed :
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := by
  sorry

end total_strings_needed_l741_741365


namespace sequence_formula_l741_741674

variable (a : ℕ → ℤ)

-- Define the given sequence terms
def seq := [20, 11, 2, -7] : List ℤ

-- Define the general formula to be proved
def general_formula (n : ℕ) : ℤ := -9n + 29

-- Assert that seq matches the values given by general_formula
theorem sequence_formula :
  (a 0 = 20) ∧ (a 1 = 11) ∧ (a 2 = 2) ∧ (a 3 = -7) →
  (∀ n, n < 4 → a n = general_formula n) → ∀ n, a n = -9n + 29 :=
by
  sorry

end sequence_formula_l741_741674


namespace problem_statement_l741_741564

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := (finset.range n).sum a

def sequence_an (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ n ≥ 2, a n + 2 * (S n) * (S (n-1)) = 0

def sequence_bn (a : ℕ → ℝ) (n : ℕ) : ℕ → ℝ :=
  λ n, if n < 2 then 0 else 2 * (1 - n) * (a n)

noncomputable def partial_sum (a : ℕ → ℝ) : ℕ → ℝ
| 0 := 0
| (n+1) := partial_sum n + a (n+1)

theorem problem_statement (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) :
  (∀ n, S n = partial_sum a n) →
  a 1 = 1 / 2 →
  sequence_an a S →
  (∀ n ≥ 2, b n = sequence_bn a n) →
  ∀ n ≥ 2, ∑ i in finset.range (n - 1), (b (i + 2))^2 < 1 := by
  sorry

end problem_statement_l741_741564


namespace k_times_k_prime_constant_l741_741255

noncomputable def ellipse (x y a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def point_on_ellipse (x y a b : ℝ) (e : ℝ) : Prop := (a > b) ∧ (b > 0) ∧ (y = a * x + 6) ∧ (e = √2 / 2)

theorem k_times_k_prime_constant
    {a b k k' : ℝ}
    (h : a^2 = 8)
    (h1 : b^2 = 4)
    (h2 : a > b)
    (h3 : b > 0)
    (h4 : ∀ x y : ℝ, ellipse x y a b)
    (h5 : |AB| = √(a^2 + b^2 - 2ab cos θ))
    (h6 : |AP| * S2 = |BP| * S1)
    (h7 : S1 ≠ 0 ∧ S2 ≠ 0)
    (P_on_ellipse : point_on_ellipse x y a b (√2 / 2))
    : (k * k') = 1 / 2 := by 
    sorry

end k_times_k_prime_constant_l741_741255


namespace find_a_find_b_find_p_find_k_l741_741700

-- SG. 1
theorem find_a (a : ℝ) : (∃ t : ℝ, 2 * a * t^2 + 12 * t + 9 = 0 /\ ∀ t : ℝ, 2 * a * t^2 + 12 * t + 9 = 0 -> t = -6 / a) -> a = 2 :=
sorry

-- SG. 2
theorem find_b (b a : ℝ) (h1 : a = 2) : (∀ x y : ℝ, a * x + b * y = 1 -> 4 * x + 18 * y = 3) -> b = 9 :=
sorry

-- SG. 3
noncomputable def nth_prime (n : ℕ) : ℕ := Nat.factor (#2 ..)
theorem find_p (p b : ℕ) (h1 : b = 9) : nth_prime b = p -> p = 23 :=
by {intro hp, rw [hp], simp [nth_prime]}

-- SG. 4
theorem find_k (k : ℝ) (θ : ℝ) : (k = (4 * sin θ + 3 * cos θ) / (2 * sin θ - cos θ)) ∧ (tan θ = 3) -> k = 3 :=
sorry

end find_a_find_b_find_p_find_k_l741_741700


namespace insurance_compensation_l741_741114

/-- Given the actual damage amount and the deductible percentage, 
we can compute the amount of insurance compensation. -/
theorem insurance_compensation : 
  ∀ (damage_amount : ℕ) (deductible_percent : ℕ), 
  damage_amount = 300000 → 
  deductible_percent = 1 →
  (damage_amount - (damage_amount * deductible_percent / 100)) = 297000 :=
by
  intros damage_amount deductible_percent h_damage h_deductible
  sorry

end insurance_compensation_l741_741114


namespace math_problem_l741_741642

def A_coords : ℝ × ℝ := (-1, 0)
def B_coords : ℝ × ℝ := (1, 0)
def C_coords : ℝ × ℝ := (0, Real.sqrt 7)

noncomputable def circle_center : ℝ × ℝ := (3, 0)
noncomputable def circle_radius : ℝ := 2 * Real.sqrt 2

-- Point P
variable (P : ℝ × ℝ)

-- Given Condition
def P_condition : Prop :=
  let PA := Real.sqrt ((P.1 + 1)^2 + P.2^2)
  let PB := Real.sqrt ((P.1 - 1)^2 + P.2^2)
  PA = Real.sqrt 2 * PB

-- Statements to verify
def statement_A : Prop :=
  ∃ (P : ℝ × ℝ), P_condition P ∧ P.1 ≠ 3 ∧ P.2 ≠ 0 ∧
  let PC := Real.sqrt ((P.1^2) + (P.2 - Real.sqrt 7)^2)
  PC = 2 * Real.sqrt 2

def statement_B : Prop :=
  ∃ (P : ℝ × ℝ), P_condition P ∧ P.1 ≠ 3 ∧ P.2 ≠ 0 ∧
  let PC := Real.sqrt ((P.1^2) + (P.2 - Real.sqrt 7)^2)
  PC = 2 * Real.sqrt 2

def statement_C : Prop :=
  ∀ (P : ℝ × ℝ), P_condition P → 
  let PA := Real.sqrt ((P.1 + 1)^2 + P.2^2)
  PA ≠ 2 * Real.sqrt 2

def statement_D : Prop :=
  ∃ (P : ℝ × ℝ), P_condition P ∧
  let area := |P.2|
  let PA := Real.sqrt ((P.1 + 1)^2 + P.2^2)
  let PC := Real.sqrt ((P.1^2) + (P.2 - Real.sqrt 7)^2)
  let val := Real.sqrt 2 * PC - PA
  val = Real.sqrt 7

theorem math_problem : statement_A ∧ statement_B ∧ ¬ statement_C ∧ statement_D := sorry

end math_problem_l741_741642


namespace part1_part2_l741_741281

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

theorem part1 (x : ℝ) : f x 1 >= f x 1 := sorry

theorem part2 (a b : ℝ) (h : ∀ x > 0, f x a ≤ b - a) : b / a ≥ 0 := sorry

end part1_part2_l741_741281


namespace books_not_sold_l741_741950

variable {B : ℕ} -- Total number of books

-- Conditions
def two_thirds_books_sold (B : ℕ) : ℕ := (2 * B) / 3
def price_per_book : ℕ := 2
def total_amount_received : ℕ := 144
def remaining_books_sold : ℕ := 0
def two_thirds_by_price (B : ℕ) : ℕ := two_thirds_books_sold B * price_per_book

-- Main statement to prove
theorem books_not_sold (h : two_thirds_by_price B = total_amount_received) : (B / 3) = 36 :=
by
  sorry

end books_not_sold_l741_741950


namespace inversely_proportional_l741_741424

theorem inversely_proportional (x y : ℕ) (c : ℕ) 
  (h1 : x * y = c)
  (hx1 : x = 40) 
  (hy1 : y = 5) 
  (hy2 : y = 10) : x = 20 :=
by
  sorry

end inversely_proportional_l741_741424


namespace randys_trip_length_l741_741809

theorem randys_trip_length
  (trip_length : ℚ)
  (fraction_gravel : trip_length = (1 / 4) * trip_length)
  (middle_miles : 30 = (7 / 12) * trip_length)
  (fraction_dirt : trip_length = (1 / 6) * trip_length) :
  trip_length = 360 / 7 :=
by
  sorry

end randys_trip_length_l741_741809


namespace minimum_throws_for_repeated_sum_l741_741037

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741037


namespace f_one_eq_minus_one_third_f_of_a_f_is_odd_l741_741662

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 1)

theorem f_one_eq_minus_one_third : f 1 = -1/3 := 
by sorry

theorem f_of_a (a : ℝ) : f a = (1 - 2^a) / (2^a + 1) := 
by sorry

theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

end f_one_eq_minus_one_third_f_of_a_f_is_odd_l741_741662


namespace additional_stamps_required_l741_741397

theorem additional_stamps_required (current_stamps : ℕ) (stamps_per_row : ℕ) (desired_stamps : ℕ) : current_stamps = 37 -> stamps_per_row = 8 -> desired_stamps = 40 -> 40 - 37 = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end additional_stamps_required_l741_741397


namespace minimum_throws_l741_741029

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741029


namespace min_value_expression_l741_741221

theorem min_value_expression (x y : ℝ) :
  ∃ (x_min y_min : ℝ), (2 * x_min^2 + 4 * x_min * y_min + 5 * y_min^2 - 8 * x_min - 6 * y_min = -11.25) :=
begin
  use [0.5, -1.5],
  sorry
end

end min_value_expression_l741_741221


namespace minimum_throws_to_ensure_same_sum_twice_l741_741010

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741010


namespace bank_tellers_total_coins_l741_741207

theorem bank_tellers_total_coins (tellers rolls_per_teller coins_per_roll : ℕ) (h1 : tellers = 4) (h2 : rolls_per_teller = 10) (h3 : coins_per_roll = 25) 
    : tellers * (rolls_per_teller * coins_per_roll) = 1000 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end bank_tellers_total_coins_l741_741207


namespace sum_of_distances_l741_741225

-- Define the rectangle vertices and midpoint centers
def A := (0, 0)
def B := (3, 0)
def C := (3, 4)
def D := (0, 4)

def M := ((3+3)/2, (0+4)/2) -- Center of BC
def N := ((0+0)/2, (0+4)/2) -- Center of AD

-- Define distance function between two points
def distance (P Q : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Prove the sum of distances from A to the centers of the opposite sides
theorem sum_of_distances : distance A M + distance A N = real.sqrt 13 + 2 := 
by 
  sorry

end sum_of_distances_l741_741225


namespace area_arccos_cos_eq_l741_741576

noncomputable def area_arccos_cos : ℝ :=
  ∫ x in 0..2 * Real.pi, Real.arccos (Real.cos x)

theorem area_arccos_cos_eq :
  area_arccos_cos = Real.pi ^ 2 :=
by
  sorry

end area_arccos_cos_eq_l741_741576


namespace complex_conjugate_of_i_times_l741_741316

def complex_conjugate (z : ℂ) : ℂ := z.re - z.im * Complex.i

theorem complex_conjugate_of_i_times (i : ℂ) (h_i : i = Complex.I) :
  complex_conjugate (i * (3 - 2 * i)) = 2 - 3 * Complex.I :=
by
  sorry

end complex_conjugate_of_i_times_l741_741316


namespace geometric_sum_eq_five_l741_741815

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (h_geometric : ∀ n, a (n+1) = a n * r)
variable (h_pos : ∀ n, a n > 0)
variable (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

theorem geometric_sum_eq_five : a 3 + a 5 = 5 :=
by
  -- Proof steps should be provided here
  -- sorry

end geometric_sum_eq_five_l741_741815


namespace solve_problem_l741_741656

variable (a b c x : ℝ)

-- Conditions
def condition1 : Prop := ∀ x, ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2

-- Statements to prove
def statementA : Prop := a < 0
def statementB : Prop := a + b + c > 0
def statementD : Prop := ∀ x, (cx^2 + bx + a < 0 ↔ (-1/3 < x ∧ x < 1/2))

theorem solve_problem (h1 : condition1)
  (h2 : statementA)
  (h3 : statementB)
  (h4 : statementD) : a < 0 ∧ a + b + c > 0 ∧ ∀ x, (cx^2 + bx + a < 0 ↔ (-1/3 < x ∧ x < 1/2)) :=
by
  sorry

end solve_problem_l741_741656


namespace min_rolls_to_duplicate_sum_for_four_dice_l741_741000

theorem min_rolls_to_duplicate_sum_for_four_dice : 
    let min_sum := 4 * 1,
    let max_sum := 4 * 6,
    let possible_sums := max_sum - min_sum + 1 in
    possible_sums = 21 → 
    (possible_sums + 1 = 22) := 
by
  intros min_sum max_sum possible_sums h
  have h1 : min_sum = 4 := rfl
  have h2 : max_sum = 24 := rfl
  have h3 : possible_sums = 21 := h
  have h4 : possible_sums + 1 = 22 := calc
    possible_sums + 1 = 21 + 1 : by rw h
    ... = 22 : by rfl
  exact h4

end min_rolls_to_duplicate_sum_for_four_dice_l741_741000


namespace seven_expression_one_seven_expression_two_l741_741901

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l741_741901


namespace horner_method_multiplication_addition_count_l741_741184

def f (x : ℝ) : ℝ :=
  2 * x^6 + 3 * x^5 + 4 * x^4 + 5 * x^3 + 6 * x^2 + 7 * x + 8

def horner_eval (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldr (λ coeff acc, acc * x + coeff) 0

noncomputable def horner_method_multiplications_additions : (ℕ × ℕ) :=
  let multiplications := 6 -- one for each level of the Horner's method
  let additions := 6 -- one for each level of the Horner's method
  (multiplications, additions)

theorem horner_method_multiplication_addition_count :
  horner_method_multiplications_additions = (6, 6) :=
by
  sorry

end horner_method_multiplication_addition_count_l741_741184


namespace main_theorem_l741_741409

def a (n : ℕ) : ℝ := (√3 + 1)^(2 * n) + (√3 - 1)^(2 * n)

-- Condition (0 < (√3 - 1)^(2 * n) < 1) encapsulated in witness property
lemma lemma1 (n : ℕ) : 0 < (√3 - 1)^(2 * n) ∧ (√3 - 1)^(2 * n) < 1 :=
  sorry

-- Recurrence relation given as a definition
lemma recurrence_relation (n : ℕ) (h_pos : 2 ≤ n) : a n = 8 * a (n - 1) - 4 * a (n - 2) :=
  sorry

-- Base cases for the sequence
@[simp]
lemma a_1 : a 1 = 8 := by
  sorry

@[simp]
lemma a_2 : a 2 = 26 := by
  sorry

-- Main theorem to be proved
theorem main_theorem (n : ℕ) :
  ∃ k : ℕ, a n = k ∧ k % 2^(n+1) = 0 ∧ a n = Int.ceil ((√3 + 1)^(2 * n)) :=
  sorry

end main_theorem_l741_741409


namespace length_of_rectangular_garden_l741_741496

theorem length_of_rectangular_garden (P B : ℝ) (h₁ : P = 1200) (h₂ : B = 240) :
  ∃ L : ℝ, P = 2 * (L + B) ∧ L = 360 :=
by
  sorry

end length_of_rectangular_garden_l741_741496


namespace characterization_of_M_l741_741769

noncomputable def M : Set ℂ := {z : ℂ | (z - 1) ^ 2 = Complex.abs (z - 1) ^ 2}

theorem characterization_of_M : M = {z : ℂ | ∃ r : ℝ, z = r} :=
by
  sorry

end characterization_of_M_l741_741769


namespace gf_of_3_l741_741287

def f : ℕ → ℕ
| 1 := 3
| 2 := 4
| 3 := 2
| 4 := 1
| _ := 0  -- For completeness, but not used

def g : ℕ → ℕ
| 1 := 2
| 2 := 1
| 3 := 6
| 4 := 8
| _ := 0  -- For completeness, but not used

theorem gf_of_3 : g (f 3) = 1 :=
by sorry

end gf_of_3_l741_741287


namespace example_one_example_two_l741_741883

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l741_741883


namespace length_of_PC_in_similar_triangles_l741_741718

theorem length_of_PC_in_similar_triangles
  (A B C P : Type)
  (distance : A -> A -> ℝ)
  (hAB : distance A B = 10)
  (hBC : distance B C = 9)
  (hCA : distance C A = 7)
  (sim1 : ∃ k, ∀ (x : A), distance x B / distance x A = k)
  (sim2 : ∃ k, ∀ (x : A), distance x P / distance x A = k) :
  distance P C = 31.5 :=
by
  sorry

end length_of_PC_in_similar_triangles_l741_741718


namespace solution_set_l741_741238

variables (x : ℝ)

def a : ℝ × ℝ := (x, -1)
def b : ℝ × ℝ := (1, 1 / x)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem solution_set (h : dot_product a b ≤ 0) :
  x ≤ -1 ∨ (0 < x ∧ x ≤ 1) := sorry

end solution_set_l741_741238


namespace shared_property_of_shapes_l741_741091

-- Definition of each shape
def is_parallelogram (P : Type) [AddCommGroup P] [Module ℝ P] (a b : P) : Prop :=
  -- Assuming some abstract properties that define a parallelogram.
  sorry

def is_rectangle (R : Type) [AddCommGroup R] [Module ℝ R] (a b : R) [is_parallelogram R a b] : Prop :=
  -- Assuming some abstract properties that define a rectangle.
  sorry

def is_rhombus (H : Type) [AddCommGroup H] [Module ℝ H] (a b : H) [is_parallelogram H a b] : Prop :=
  -- Assuming some abstract properties that define a rhombus.
  sorry

def is_square (S : Type) [AddCommGroup S] [Module ℝ S] (a b : S) [is_rectangle S a b] [is_rhombus S a b] : Prop :=
  -- Assuming some abstract properties that define a square.
  sorry

-- Property to be proven
theorem shared_property_of_shapes
    (P : Type) [AddCommGroup P] [Module ℝ P] (a b : P)
    (hP : is_parallelogram P a b)
    (R : Type) [AddCommGroup R] [Module ℝ R] (c d : R)
    (hR : is_rectangle R c d)
    (H : Type) [AddCommGroup H] [Module ℝ H] (e f : H)
    (hH : is_rhombus H e f)
    (S : Type) [AddCommGroup S] [Module ℝ S] (g k : S)
    (hS : is_square S g k) : 
    (∀ (U : Type) [AddCommGroup U] [Module ℝ U] (x y : U), (is_parallelogram U x y) → opposite_sides_parallel_and_equal U x y) :=
  sorry

end shared_property_of_shapes_l741_741091


namespace ice_cream_ordering_ways_l741_741713

def number_of_cone_choices : ℕ := 2
def number_of_flavor_choices : ℕ := 4

theorem ice_cream_ordering_ways : number_of_cone_choices * number_of_flavor_choices = 8 := by
  sorry

end ice_cream_ordering_ways_l741_741713


namespace socks_picking_problem_l741_741135

theorem socks_picking_problem : 
  ∀ (red green blue yellow : ℕ), 
    red = 50 ∧ green = 40 ∧ blue = 30 ∧ yellow = 20 → 
    ∃ (n : ℕ), n = 33 ∧ 
      (∀ (picked_socks : list ℕ), 
        (picked_socks.length ≥ n → 
          ∃ (pairs : ℕ), pairs ≥ 15 ∧ 
            (∃ color, ∃ m, m ≥ 30 ∧ color ∈ [red, green, blue, yellow] ∧ 
               list.count picked_socks color ≥ 2 * pairs))) :=
begin
  sorry
end

end socks_picking_problem_l741_741135


namespace min_intersection_area_l741_741428

-- Definitions of necessary geometric constructs
structure Triangle (α : Type*) :=
(A B C : α)

def midpoints (α : Type*) [HasMidpoint α] (Δ : Triangle α) : Triangle α :=
{ A := midpoint Δ.B Δ.C,
  B := midpoint Δ.C Δ.A,
  C := midpoint Δ.A Δ.B }

def is_on_segment {α : Type*} [IsSegment α] (P : α) (Q : α) (R : α) : Prop :=
P ∈ segment Q R

-- Define and state the main theorem
theorem min_intersection_area {α : Type*} [MetricSpace α] [IsTriangle α] [HasMidpoint α] [IsSegment α]
  (Δ : Triangle α) (K L M : α)
  (h_area_ABC : area Δ = 1)
  (A1 B1 C1 : α)
  (H_midpoints : A1 = midpoint Δ.B Δ.C ∧ B1 = midpoint Δ.C Δ.A ∧ C1 = midpoint Δ.A Δ.B)
  (H_segments : is_on_segment K Δ.A B1 ∧ is_on_segment L Δ.C A1 ∧ is_on_segment M Δ.B C1) :
  ∃ (inter_area : ℝ), inter_area = area (intersection (Triangle.mk K L M) (Triangle.mk A1 B1 C1)) ∧ inter_area = (1/8) :=
sorry

end min_intersection_area_l741_741428


namespace circumradius_triangle_FKO_l741_741406

-- Define the scenario of the problem
variable {K L M A F O : Type}
variables [inhabited L] [inhabited M] [inhabited A]
variables (d₁ : ∠KLM = 120)
variables (d₂ : A ∈ lineSegment L M)
variables (d₃ : incenter AKL = F)
variables (d₄ : incenter AKM = O)
variables (d₅ : distance A O = 2)
variables (d₆ : distance A F = 7)

-- Define the target theorem to prove based on the given statements
theorem circumradius_triangle_FKO 
  (open_locale euclidean_geometry)
  (open_locale real) :
  circumradius (triangle F K O) = sqrt 159 / 3 :=
sorry

end circumradius_triangle_FKO_l741_741406


namespace annual_interest_rate_approx_l741_741743

noncomputable def find_annual_rate (P A n t : ℝ) : ℝ := 
  let r_approx := (A / P)^(1 / (n * t)) - 1
  2 * r_approx

theorem annual_interest_rate_approx {P A : ℝ} (hP : P = 10000) (hA : A = 10815.83) 
  (hn : n = 2) (ht : t = 2) : 
  find_annual_rate 10000 10815.83 2 2 ≈ 0.0398 :=
by
  unfold find_annual_rate
  rw [hP, hA, hn, ht]
  -- We assume here that the result of the calculations gives approximately 0.0398.
  -- The exact numeric handling part (≈) should be justified by numerical computation tools outside Lean.
  sorry

end annual_interest_rate_approx_l741_741743


namespace min_rolls_to_duplicate_sum_for_four_dice_l741_741003

theorem min_rolls_to_duplicate_sum_for_four_dice : 
    let min_sum := 4 * 1,
    let max_sum := 4 * 6,
    let possible_sums := max_sum - min_sum + 1 in
    possible_sums = 21 → 
    (possible_sums + 1 = 22) := 
by
  intros min_sum max_sum possible_sums h
  have h1 : min_sum = 4 := rfl
  have h2 : max_sum = 24 := rfl
  have h3 : possible_sums = 21 := h
  have h4 : possible_sums + 1 = 22 := calc
    possible_sums + 1 = 21 + 1 : by rw h
    ... = 22 : by rfl
  exact h4

end min_rolls_to_duplicate_sum_for_four_dice_l741_741003


namespace min_magnitude_c_l741_741636

variables {V : Type*} [inner_product_space ℝ V]
variables {a b c : V}

theorem min_magnitude_c (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hab : ⟪a, b⟫ = 0) (hc : ∥c - a - b∥ = 1) : 
  ∥c∥ = real.sqrt 2 - 1 :=
sorry

end min_magnitude_c_l741_741636


namespace closest_log_num_divisors_2014_factorial_eq_439_l741_741763

def num_divisors_factorial (n : ℕ) : ℕ := 
  -- Computes the number of divisors of n!
  ∏ p in (Finset.range (n + 1)).filter Nat.Prime, (Nat.factorial n).factorization p + 1

def closest_integer (x : ℝ) : ℤ := 
  -- Computes the closest integer to the real number x
  if x - x.floor < 0.5 then x.floor else x.ceil

theorem closest_log_num_divisors_2014_factorial_eq_439 : closest_integer (Real.log (num_divisors_factorial 2014)) = 439 :=
by
  sorry -- Proof is left as an exercise

end closest_log_num_divisors_2014_factorial_eq_439_l741_741763


namespace peter_money_left_l741_741404

variable (soda_cost : ℝ) (money_brought : ℝ) (soda_ounces : ℝ)

theorem peter_money_left (h1 : soda_cost = 0.25) (h2 : money_brought = 2) (h3 : soda_ounces = 6) : 
    money_brought - soda_ounces * soda_cost = 0.50 := 
by 
  sorry

end peter_money_left_l741_741404


namespace minimum_rolls_for_duplicate_sum_l741_741051

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741051


namespace minimum_rolls_for_duplicate_sum_l741_741050

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741050


namespace determine_min_bottles_l741_741147

-- Define the capacities and constraints
def mediumBottleCapacity : ℕ := 80
def largeBottleCapacity : ℕ := 1200
def additionalBottles : ℕ := 5

-- Define the minimum number of medium-sized bottles Jasmine needs to buy
def minimumMediumBottles (mediumCapacity largeCapacity extras : ℕ) : ℕ :=
  let requiredBottles := largeCapacity / mediumCapacity
  requiredBottles

theorem determine_min_bottles :
  minimumMediumBottles mediumBottleCapacity largeBottleCapacity additionalBottles = 15 :=
by
  sorry

end determine_min_bottles_l741_741147


namespace ancient_chinese_poem_l741_741731

theorem ancient_chinese_poem (x : ℕ) :
  (7 * x + 7 = 9 * (x - 1)) :=
sorry

end ancient_chinese_poem_l741_741731


namespace JohnNeeds72Strings_l741_741370

def JohnHasToRestring3Basses : Nat := 3
def StringsPerBass : Nat := 4

def TwiceAsManyGuitarsAsBasses : Nat := 2 * JohnHasToRestring3Basses
def StringsPerNormalGuitar : Nat := 6

def ThreeFewerEightStringGuitarsThanNormal : Nat := TwiceAsManyGuitarsAsBasses - 3
def StringsPerEightStringGuitar : Nat := 8

def TotalStringsNeeded : Nat := 
  (JohnHasToRestring3Basses * StringsPerBass) +
  (TwiceAsManyGuitarsAsBasses * StringsPerNormalGuitar) +
  (ThreeFewerEightStringGuitarsThanNormal * StringsPerEightStringGuitar)

theorem JohnNeeds72Strings : TotalStringsNeeded = 72 := by
  calculate
  sorry

end JohnNeeds72Strings_l741_741370


namespace find_radius_of_smaller_sphere_l741_741336

-- Define the radii of the four given spheres
def r1 : ℝ := 2
def r2 : ℝ := 2
def r3 : ℝ := 3
def r4 : ℝ := 3

-- Define the curvature of a sphere given its radius
def curvature (r : ℝ) : ℝ := 1 / r

-- Descartes' Circle Theorem condition
def descartes_condition (k1 k2 k3 k4 ks : ℝ) : Prop :=
  (k1 + k2 + k3 + k4 + ks)^2 = 2 * (k1^2 + k2^2 + k3^2 + k4^2 + ks^2)

-- Define the curvatures using the given radii
def k1 : ℝ := curvature r1
def k2 : ℝ := curvature r2
def k3 : ℝ := curvature r3
def k4 : ℝ := curvature r4

-- The correct value of the curvature for the small sphere
def ks : ℝ := (2 - Real.sqrt 26) / 6

def radius_smaller_sphere_proof : Prop :=
  let r_s := 1 / ks in
  r_s = 6 / (2 - Real.sqrt 26)

-- The final Lean statement to prove that the radius r_s satisfies the conditions
theorem find_radius_of_smaller_sphere :
  descartes_condition k1 k2 k3 k4 ks → radius_smaller_sphere_proof :=
by
  sorry

end find_radius_of_smaller_sphere_l741_741336


namespace binomial_expansion_coefficient_l741_741644

theorem binomial_expansion_coefficient (a : ℝ) (h : (6.choose 3) * a^3 * 8 = 5/2) : a = 1/4 :=
sorry

end binomial_expansion_coefficient_l741_741644


namespace distance_from_B_l741_741971

theorem distance_from_B (A : ℝ) (B_moves_to_diagonal : Prop) (visible_black_area_twice_white : Prop) : 
  A = 18 ∧ B_moves_to_diagonal ∧ visible_black_area_twice_white → 
  ∃ d : ℝ, d = (6 * sqrt 10) / 5 :=
by sorry

end distance_from_B_l741_741971


namespace village_population_500_l741_741324

variable (n : ℝ) -- Define the variable for population increase
variable (initial_population : ℝ) -- Define the variable for the initial population

-- Conditions from the problem
def first_year_increase : Prop := initial_population * (3 : ℝ) = n
def initial_population_def : Prop := initial_population = n / 3
def second_year_increase_def := ((n / 3 + n) * (n / 100 )) = 300

-- Define the final population formula
def population_after_two_years : ℝ := (initial_population + n + 300)

theorem village_population_500 (n : ℝ) (initial_population: ℝ) :
  first_year_increase n initial_population →
  initial_population_def n initial_population →
  second_year_increase_def n →
  population_after_two_years n initial_population = 500 :=
by sorry

#check village_population_500

end village_population_500_l741_741324


namespace vasya_100_using_fewer_sevens_l741_741908

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l741_741908


namespace total_revenue_correct_l741_741865

-- Definitions based on the problem conditions
def price_per_kg_first_week : ℝ := 10
def quantity_sold_first_week : ℝ := 50
def discount_percentage : ℝ := 0.25
def multiplier_next_week : ℝ := 3

-- Derived definitions
def revenue_first_week := quantity_sold_first_week * price_per_kg_first_week
def quantity_sold_second_week := multiplier_next_week * quantity_sold_first_week
def discounted_price_per_kg := price_per_kg_first_week * (1 - discount_percentage)
def revenue_second_week := quantity_sold_second_week * discounted_price_per_kg
def total_revenue := revenue_first_week + revenue_second_week

-- The theorem that needs to be proven
theorem total_revenue_correct : total_revenue = 1625 := 
by
  sorry

end total_revenue_correct_l741_741865


namespace integer_values_satisfying_abs_lt_2pi_l741_741684

theorem integer_values_satisfying_abs_lt_2pi : 
  (finset.Icc (-6 : ℤ) 6).card = 13 :=
by
  sorry

end integer_values_satisfying_abs_lt_2pi_l741_741684


namespace mod_remainder_l741_741483

theorem mod_remainder (n : ℤ) (h : n % 5 = 3) : (4 * n - 5) % 5 = 2 := by
  sorry

end mod_remainder_l741_741483


namespace additional_cars_needed_l741_741191

theorem additional_cars_needed (current_cars : ℕ) (cars_per_row : ℕ) : current_cars = 35 → cars_per_row = 8 → (∃ additional_cars : ℕ, additional_cars = 5 ∧ (current_cars + additional_cars) % cars_per_row = 0) :=
by
  intros h1 h2
  rw [h1, h2]
  use 5
  split
  . rfl
  . norm_num

end additional_cars_needed_l741_741191


namespace min_value_of_fraction_sum_l741_741393

theorem min_value_of_fraction_sum (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x^2 + y^2 + z^2 = 1) :
  (2 * (1/(1-x^2) + 1/(1-y^2) + 1/(1-z^2))) = 3 * Real.sqrt 3 :=
sorry

end min_value_of_fraction_sum_l741_741393


namespace infinite_int_and_nonint_seq_l741_741748

open Nat 

noncomputable def a (m k : ℕ) : ℚ :=
  (2 * k * m).factorial / (3 ^ ((k - 1) * m) : ℚ)

theorem infinite_int_and_nonint_seq
  (m : ℕ) (h : m > 0) :
  (∃ S T : Set ℕ, (∀ k ∈ S, a m k ∈ ℤ) ∧ (∀ k ∈ T, a m k ∉ ℤ) ∧ S.Infinite ∧ T.Infinite) :=
sorry

end infinite_int_and_nonint_seq_l741_741748


namespace rectangle_area_outside_three_circles_l741_741413

theorem rectangle_area_outside_three_circles :
  let FH := 4
    HE := 6
    r_E := 2
    r_F := 1.5
    r_H := 3
    area_rectangle := FH * HE
    pi := Real.pi
    area_quarter_circle_E := (pi * r_E ^ 2) / 4
    area_quarter_circle_F := (pi * r_F ^ 2) / 4
    area_quarter_circle_H := (pi * r_H ^ 2) / 4
    area_outside_circles := area_rectangle - area_quarter_circle_E - area_quarter_circle_F - area_quarter_circle_H
  in  area_outside_circles ≈ 12.0 := sorry

end rectangle_area_outside_three_circles_l741_741413


namespace volume_removal_percentage_l741_741531

def box_dimensions : ℕ × ℕ × ℕ := (20, 15, 10)
def cube_side : ℕ := 4
def cubes_removed : ℕ := 8

noncomputable def original_volume := (20 * 15 * 10 : ℚ)
noncomputable def volume_cube := (4 ^ 3 : ℚ)
noncomputable def total_volume_removed := (8 * volume_cube)

noncomputable def percent_volume_removed := (total_volume_removed / original_volume) * 100

theorem volume_removal_percentage :
  percent_volume_removed ≈ 17.07 := sorry

end volume_removal_percentage_l741_741531


namespace moving_point_trajectory_and_perpendicularity_l741_741526

theorem moving_point_trajectory_and_perpendicularity :
  (∀ (x y: ℝ), (x ≠ -2 ∧ x ≠ 2) → 
    (let slope_PA := y / (x + 2),
         slope_PB := y / (x - 2) in
       slope_PA * slope_PB = -1/3) →
    (∀ (c d : Point), (non_zero_slope ? ? (line_through Q c)) ->
    (line_through Q (-1, 0) intersect curve_E) = {c, d} &
      (x^2 / 4 + 3*y^2 / 4 = 1  → 
        ∀ (a : Point), a = (-2, 0) → 
          let AC := vector a c,
          let AD := vector a d in
            dot_product AC AD = 0)) :=
sorry

end moving_point_trajectory_and_perpendicularity_l741_741526


namespace length_of_path_along_arrows_l741_741416

theorem length_of_path_along_arrows (s : List ℝ) (h : s.sum = 73) :
  (3 * s.sum = 219) :=
by
  sorry

end length_of_path_along_arrows_l741_741416


namespace sum_cubic_zero_l741_741749

-- Definitions for given problem
open_locale big_operators

variables (n : ℕ) (y : ℕ → ℝ)

def conditions (n : ℕ) (y : ℕ → ℝ) : Prop :=
  n ≥ 4 ∧
  ∑ k in finset.range (n + 1), y k = 0 ∧
  ∑ k in finset.range (n + 1), k * y k = 0 ∧
  ∑ k in finset.range (n + 1), k^2 * y k = 0 ∧
  (∀ k, 1 ≤ k ∧ k ≤ n - 3 → y (k + 3) - 3 * y (k + 2) + 3 * y (k + 1) - y k = 0)

theorem sum_cubic_zero (n : ℕ) (y : ℕ → ℝ) :
  conditions n y → 
  ∑ k in finset.range (n + 1), k^3 * y k = 0 :=
begin
  sorry
end

end sum_cubic_zero_l741_741749


namespace agatha_frame_cost_l741_741978

variables (F : ℕ) -- F represents the amount spent on the frame.

-- Define the conditions as hypotheses
def front_wheel_cost : ℕ := 25
def remaining_cost : ℕ := 20
def initial_amount : ℕ := 60

-- Define the total amount spent on non-frame items
def total_other_cost : ℕ := front_wheel_cost + remaining_cost

-- Define the proof problem statement
theorem agatha_frame_cost (h1 : total_other_cost = 25 + 20)
                          (h2 : initial_amount = 60) :
   F = initial_amount - total_other_cost := 
begin
  sorry -- Proof to be provided
end

end agatha_frame_cost_l741_741978


namespace monotonic_decreasing_interval_l741_741445

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < (1 / 2) → (0 < x ∧ x < (1 / 2)) ∧ (f (1 / 2) - f x) > 0 :=
sorry

end monotonic_decreasing_interval_l741_741445


namespace example_one_example_two_l741_741886

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l741_741886


namespace tangent_line_at_origin_extreme_points_count_l741_741284

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 + a * Real.log (x + 1)
def f_prime (x a : ℝ) : ℝ := 2 * x + a / (x + 1)

-- Part 1: Tangent line at the origin when a = -1
theorem tangent_line_at_origin (a : ℝ) (h : a = -1) : 
  f_prime 0 a = -1 → (∀ x, f x a = x^2 + a * Real.log (x + 1)) → 
  (∀ y, y = -x) := 
sorry

-- Part 2: Number of extreme points based on the value of a
theorem extreme_points_count (a : ℝ) (h : a ≠ 0) :
  (a ≥ 1/2 → ¬ ∃ x, ∃ y, ∀ t, f t a = f x a ∧ f t a = f y a) ∧
  (0 < a ∧ a < 1/2 → ∃ x1 x2, x1 ≠ x2 ∧ ∀ t, f t a = f x1 a ∨ f t a = f x2 a) ∧
  (a < 0 → ∃ x, ∀ t, f t a = f x a) :=
sorry

end tangent_line_at_origin_extreme_points_count_l741_741284


namespace patrick_fish_count_l741_741986

variables (p a o : ℕ)
variables (h1 : a = p + 4) (h2 : o = a - 7) (o_val : o = 5)

theorem patrick_fish_count : p = 8 :=
by
  have h3 : a = 12, from sorry
  have h4 : p = 8, from sorry
  exact h4

end patrick_fish_count_l741_741986


namespace linear_function_relationship_price_for_profit_maximize_profit_with_donation_l741_741130

-- Part 1: Linear relationship proof
theorem linear_function_relationship 
  (points : List (ℝ × ℝ)) 
  (h1 : (80, 240) ∈ points)
  (h2 : (90, 220) ∈ points)
  (h3 : ∀ x y, (x, y) ∈ points → y = k * x + b → ∃ k b, y = -2 * x + 400) :
  ∃ k b, ∀ x y, (x, y) ∈ points → y = -2 * x + 400 := 
sorry

-- Part 2: Price for 8000 yuan profit
theorem price_for_profit (x : ℝ) 
  (profit_equation : ℝ → ℝ)
  (h : profit_equation x = (x - 60) * (-2 * x + 400))
  (profit_target : profit_equation x = 8000) :
  x = 100 :=
sorry

-- Part 3: Maximizing profit with donation
theorem maximize_profit_with_donation (x : ℝ)
  (profit_equation_with_donation : ℝ → ℝ)
  (h : profit_equation_with_donation x = (x - 70) * (-2 * x + 400)) :
  ∃ x, ∀ x, profit_equation_with_donation x = -2 * (x - 135)^2 + 8450 :=
sorry

end linear_function_relationship_price_for_profit_maximize_profit_with_donation_l741_741130


namespace katie_baked_5_cookies_l741_741232

theorem katie_baked_5_cookies (cupcakes cookies sold left : ℕ) 
  (h1 : cupcakes = 7) 
  (h2 : sold = 4) 
  (h3 : left = 8) 
  (h4 : cupcakes + cookies = sold + left) : 
  cookies = 5 :=
by sorry

end katie_baked_5_cookies_l741_741232


namespace negation_of_universal_proposition_l741_741703

variable (p : Prop)
variable (x : ℝ)

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - 1 > 0)) ↔ (∃ x : ℝ, x^2 - 1 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l741_741703


namespace system_of_equations_unique_solutions_l741_741189

theorem system_of_equations_unique_solutions :
  ∃ (S : Set (ℝ × ℝ)), (∀ (x y : ℝ), ((x, y) ∈ S ↔ (x = x^2 + y^2 ∧ y = 3 * x^2 * y - y^3)) ∧ S.card = 2) :=
begin
  sorry
end

end system_of_equations_unique_solutions_l741_741189


namespace polygon_sides_l741_741319

theorem polygon_sides (n : ℕ) (h1 : n ≥ 3)
  (h2 : ∃ (theta theta' : ℝ), theta = (n - 2) * 180 / n ∧ theta' = (n + 7) * 180 / (n + 9) ∧ theta' = theta + 9) : n = 15 :=
sorry

end polygon_sides_l741_741319


namespace minimum_throws_l741_741022

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741022


namespace even_number_of_teams_l741_741922

noncomputable theory

variables (N : ℕ) (A : matrix (fin N) (fin N) ℕ)

-- Auxiliary definitions to encode the condition in Lean
def points_conditions (A : matrix (fin N) (fin N) ℕ) : Prop :=
  (∀ i j, i ≠ j → A i j + A j i = 2) ∧
  (∀ i, A i i = 0) ∧
  (∀ (S : finset (fin N)), ∃ i ∈ S, ∑ j in S, A i j % 2 = 1)

-- The main theorem
theorem even_number_of_teams (h : points_conditions N A) : N % 2 = 0 :=
sorry

end even_number_of_teams_l741_741922


namespace AE_length_is_10_angle_AEB_is_105_l741_741727

variables (A B C D P E : Type)
variables [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq P] [DecidableEq E]
variables (length_AD : ℝ) (length_CE : ℝ)
variables (angle_PAD : ℝ) (angle_AEB : ℝ)

-- Conditions
variables
  (h_rect : rectangle A B C D)
  (h_AD_length : length_AD = 5)
  (h_CE_length : length_CE = 5)
  (h_APD_eq : equilateral_triangle A P D)

-- Proof problems
theorem AE_length_is_10 :
  AE.A P D E = 10 :=
sorry

theorem angle_AEB_is_105 :
  angle A E B = 105 :=
sorry

end AE_length_is_10_angle_AEB_is_105_l741_741727


namespace minimum_throws_l741_741030

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741030


namespace measure_angle_BAD_l741_741407

theorem measure_angle_BAD 
  (A B C D : Type) 
  (angle_ABC : ℝ) 
  (angle_ABD : ℝ) 
  (angle_DBC : ℝ) 
  (angle_C : ℝ) 
  (D_on_AC: Prop):
  angle_ABD = 20 ∧ 
  angle_DBC = 30 ∧ 
  angle_C = 120 ∧ 
  angle_ABC < 180 → 
  angle_ABC - angle_ABD - angle_DBC = 90 :=
begin
  sorry
end

end measure_angle_BAD_l741_741407


namespace tens_digit_of_smallest_positive_integer_l741_741478

def is_divisible (n divisor : ℕ) : Prop := ∃ k, n = k * divisor

def lcm_of_three (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

def smallest_positive_integer (a b c : ℕ) : ℕ := lcm_of_three a b c

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_smallest_positive_integer :
  let N := smallest_positive_integer 20 16 2016 in
  is_divisible N 20 → is_divisible N 16 → is_divisible N 2016 → tens_digit N = 8 :=
by
  sorry

end tens_digit_of_smallest_positive_integer_l741_741478


namespace hyperbola_equations_l741_741577

theorem hyperbola_equations :
  ∀ (e : ℝ) (a_ellipse2 : ℝ) (b_ellipse2 : ℝ), 
  e = 2 → 
  (a_ellipse2 = 16 ∧ b_ellipse2 = 9) → 
  ∃ (a_hyperbola : ℝ) (b_hyperbola2_1 b_hyperbola2_2 : ℝ),
  (a_hyperbola = 4 ∧ b_hyperbola2_1 = 48 ∧
  ( ∀ (x y : ℝ),
  (x^2 / a_hyperbola^2 - y^2 / b_hyperbola2_1 = 1) ∨
  (y^2 / (b_ellipse2) - x^2 / b_hyperbola2_2 = 1))) :=
begin
  sorry
end

end hyperbola_equations_l741_741577


namespace win_sector_area_l741_741134

-- Define the radius of the circle
def radius : ℝ := 12

-- Define the probability of winning
def probability_of_winning : ℝ := 1/3

-- Define the area of the entire circle
def circle_area : ℝ := Real.pi * radius^2

-- Define the area of the WIN sector
def win_area : ℝ := circle_area * probability_of_winning

-- The theorem to prove that the area of the WIN sector is 48 * π
theorem win_sector_area : win_area = 48 * Real.pi :=
by
  sorry

end win_sector_area_l741_741134


namespace average_of_new_set_l741_741109

theorem average_of_new_set (a : ℕ → ℝ) (h_avg : (∑ i in Finset.range 7, a i) / 7 = 26) :
  (∑ i in Finset.range 7, 5 * a i) / 7 = 130 := 
by
  sorry

end average_of_new_set_l741_741109


namespace rectangle_length_l741_741442

theorem rectangle_length : 
  ∃ l b : ℝ, 
    (l = 2 * b) ∧ 
    (20 < l ∧ l < 50) ∧ 
    (10 < b ∧ b < 30) ∧ 
    ((l - 5) * (b + 5) = l * b + 75) ∧ 
    (l = 40) :=
sorry

end rectangle_length_l741_741442


namespace inequality_solution_set_l741_741651

theorem inequality_solution_set (a b c : ℝ)
  (h1 : ∀ x, (ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, ¬ (bx + c > 0 ↔ x > 6)) ∧ (∀ x, (cx^2 + bx + a < 0 ↔ -1/3 < x ∧ x < 1/2)) :=
by
  sorry

end inequality_solution_set_l741_741651


namespace fewerSevensCanProduce100_l741_741913

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l741_741913


namespace parallelogram_vector_eq_l741_741726

variable (A B C D O : Type) [AddCommGroup A] [Module ℝ A]
variable (v a b : A)
variables (AC BD AB : A)

-- Conditions
variable (h1 : AC = a)
variable (h2 : BD = b)
variable (h3 : AB = (1/2 : ℝ) • AC - (1/2 : ℝ) • BD)

-- Proof Problem Statement
theorem parallelogram_vector_eq (h4 : AB = (1/2 : ℝ) • a - (1/2 : ℝ) • b) : 
  AB = (1/2 : ℝ) • a - (1/2 : ℝ) • b := 
by
  rw [h4]
  exact h3


end parallelogram_vector_eq_l741_741726


namespace symmetry_axis_shifted_sin_l741_741858

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - (Real.pi / 3))

theorem symmetry_axis_shifted_sin (x : ℝ) :
  ∃ k ∈ Int, x = (k * Real.pi / 2) + (5 * Real.pi / 12) :=
sorry

end symmetry_axis_shifted_sin_l741_741858


namespace min_throws_to_ensure_same_sum_twice_l741_741068

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741068


namespace symmetric_line_eq_l741_741339

theorem symmetric_line_eq (x y: ℝ) :
    (∃ (a b: ℝ), 3 * a - b + 2 = 0 ∧ a = 2 - x ∧ b = 2 - y) → 3 * x - y - 6 = 0 :=
by
    intro h
    sorry

end symmetric_line_eq_l741_741339


namespace arithmetic_sequence_count_l741_741628

theorem arithmetic_sequence_count :
  ∃! (n a d : ℕ), n ≥ 3 ∧ (n * (2 * a + (n - 1) * d) = 2 * 97^2) :=
sorry

end arithmetic_sequence_count_l741_741628


namespace annual_interest_rate_l741_741744

theorem annual_interest_rate (A P : ℝ) (r : ℝ) (n t : ℕ)
  (hP : P = 10000) (hA : A = 10815.83) (hn : n = 2) (ht : t = 2) 
  (hA_eq : A = P * (1 + r / n)^ (n * t)) : 
  r ≈ 0.0398 :=
by
  have h1 : 10815.83 = 10000 * (1 + r / 2) ^ (2 * 2), from hA_eq,
  sorry

end annual_interest_rate_l741_741744


namespace volume_pqrs_l741_741427

noncomputable def tetrahedron_volume (pq pr qr qs ps rs : ℝ) : ℝ :=
(float.sqrt (737)).recip * 24

theorem volume_pqrs (PQ PR QR QS PS : ℝ) (RS : ℝ) :
  PQ = 6 → PR = 4 → QR = 5 → QS = 5 → PS = 4 → RS = (15 / 4) * (Real.sqrt 2) →
  tetrahedron_volume PQ PR QR QS PS RS = 24 / (Real.sqrt 737) :=
by
  intros hPQ hPR hQR hQS hPS hRS
  sorry

end volume_pqrs_l741_741427


namespace determine_t_l741_741792

-- Define Katrina's and Andy's working hours and earnings per hour
def katrina_hours (t : ℤ) : ℤ := t - 4
def katrina_rate (t : ℤ) : ℤ := 3t - 10
def andy_hours (t : ℤ) : ℤ := 3t - 12
def andy_rate (t : ℤ) : ℤ := t - 3

-- Define the earnings for Katrina and Andy
def katrina_earnings (t : ℤ) : ℤ := katrina_hours t * katrina_rate t
def andy_earnings (t : ℤ) : ℤ := andy_hours t * andy_rate t

-- Prove the value of t that satisfies the equality of their earnings
theorem determine_t : ∃ (t : ℤ), katrina_earnings t = andy_earnings t ∧ t = 4 :=
by
  sorry

end determine_t_l741_741792


namespace percentage_female_on_duty_l741_741402

-- Definition of conditions
def on_duty_officers : ℕ := 152
def female_on_duty : ℕ := on_duty_officers / 2
def total_female_officers : ℕ := 400

-- Proof goal
theorem percentage_female_on_duty : (female_on_duty * 100) / total_female_officers = 19 := by
  -- We would complete the proof here
  sorry

end percentage_female_on_duty_l741_741402


namespace employees_count_l741_741429

theorem employees_count (n : ℕ) 
  (avg_salary_without_manager : n * 1500) 
  (avg_salary_with_manager : (n * 1500 + 12000) / (n + 1) = 2000) :
  n = 20 :=
sorry

end employees_count_l741_741429


namespace distinct_complex_numbers_count_l741_741196

theorem distinct_complex_numbers_count (z : ℂ) (h1: abs z = 1) (h2: ∃ n : ℤ, z ^ (8! : ℕ) - z ^ (7! : ℕ) = n) : 
  ∃ n : ℕ, n = 350 :=
by sorry

end distinct_complex_numbers_count_l741_741196


namespace denise_travel_l741_741193

theorem denise_travel (a b c : ℕ) (h₀ : a ≥ 1) (h₁ : a + b + c = 8) (h₂ : 90 * (b - a) % 48 = 0) : a^2 + b^2 + c^2 = 26 :=
sorry

end denise_travel_l741_741193


namespace find_difference_l741_741567

variables (N M_s P_s M m : ℕ)

-- Total number of students
def total_students : ℕ := 2500

-- Range for students studying Mathematics
def studying_mathematics (M_s : ℕ) : Prop :=
  1750 ≤ M_s ∧ M_s ≤ 1875

-- Range for students studying Physics
def studying_physics (P_s : ℕ) : Prop :=
  875 ≤ P_s ∧ P_s ≤ 1125

-- Apply Principle of Inclusion-Exclusion
def min_intersection (M_s P_s : ℕ) : ℕ := M_s + P_s - total_students
def max_intersection (M_s P_s : ℕ) : ℕ := M_s + P_s - total_students

-- Prove M - m = -375
theorem find_difference (M_s P_s : ℕ) (hM : studying_mathematics M_s) (hP : studying_physics P_s) :
  (max_intersection 1750 875) - (min_intersection 1875 1125) = -375 :=
by {
  sorry
}

end find_difference_l741_741567


namespace correct_function_is_f2_l741_741165

def f1 (x : ℝ) : ℝ := -x^2

def f2 (x : ℝ) : ℝ := abs x

def f3 (x : ℝ) : ℝ := -1 / x

def f4 (x : ℝ) : ℝ := log x / log 2

-- Definition of even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Definition of monotonically increasing function
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, a < x1 ∧ x1 < x2 ∧ x2 < b → f x1 ≤ f x2

theorem correct_function_is_f2 :
  is_even f2 ∧ (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < ∞ → f2 x1 ≤ f2 x2) ∧
  (¬ (is_even f1 ∧ (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < ∞ → f1 x1 ≤ f1 x2))) ∧
  (¬ (is_even f3 ∧ (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < ∞ → f3 x1 ≤ f3 x2))) ∧
  (¬ (is_even f4 ∧ (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < ∞ → f4 x1 ≤ f4 x2))) :=
by sorry

end correct_function_is_f2_l741_741165


namespace problem_proof_l741_741663

-- Definitions based on conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

-- Theorem to prove
theorem problem_proof (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end problem_proof_l741_741663


namespace prob_divisible_by_7_or_11_l741_741547

theorem prob_divisible_by_7_or_11 (m n : ℕ) (hm : m = 11) (hn : n = 50) :
  (m + n) = 61 :=
by {
  sorry
}

end prob_divisible_by_7_or_11_l741_741547


namespace find_second_number_l741_741508

theorem find_second_number (X : ℝ) : 
  (0.6 * 50 - 0.3 * X = 27) → X = 10 :=
by
  sorry

end find_second_number_l741_741508


namespace minimum_throws_for_repeated_sum_l741_741040

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741040


namespace true_propositions_l741_741661

variables {Point : Type} {Line Plane : Type}
variables m l n : Line
variables α β : Plane
variable A : Point

-- Conditions for proposition ①
def prop_1_cond1 := m ⊆ α
def prop_1_cond2 := l ⊥ α ∧ A ∉ m

-- Conditions for proposition ②
def prop_2_cond1 := l ∥ α
def prop_2_cond2 := m ∥ β
def prop_2_cond3 := α ∥ β

-- Conditions for proposition ③
def prop_3_cond1 := l ⊆ α
def prop_3_cond2 := m ⊆ α
def prop_3_cond3 := l ∩ m = A
def prop_3_cond4 := l ∥ β
def prop_3_cond5 := m ∥ β

-- Statements of the propositions
def prop_1 := ¬ coplanar l m
def prop_2 := l ∥ m
def prop_3 := α ∥ β

-- Theorem stating the equivalence of problem and correct answers, without proof
theorem true_propositions : 
  (prop_1_cond1 ∧ prop_1_cond2 → prop_1) ∧
  (¬ (prop_2_cond1 ∧ prop_2_cond2 ∧ prop_2_cond3 → prop_2)) ∧
  (prop_3_cond1 ∧ prop_3_cond2 ∧ prop_3_cond3 ∧ prop_3_cond4 ∧ prop_3_cond5 → prop_3) :=
by sorry

end true_propositions_l741_741661


namespace sum_major_axes_l741_741456

theorem sum_major_axes (n : ℕ) (e : ℕ → ℝ) (a : ℕ → ℝ) : 
  (∀ k, 1 ≤ k ∧ k ≤ n → e k = 2^(-k) ∧ a k = 2^(-k)) → 
  (∑ k in finset.range n, 2 * a (k + 1)) = 2 - 2^(1 - n) :=
by
  sorry

end sum_major_axes_l741_741456


namespace relay_team_permutations_l741_741747

theorem relay_team_permutations : 
  let team := ["Jordan", "Bob", "Charlie", "Alice"] in
  (team[3] = "Jordan" ∧ team[1] = "Bob") →
  (list.permutations ["Charlie", "Alice"]).length = 2 :=
by
  sorry

end relay_team_permutations_l741_741747


namespace smallest_number_with_55_divisors_l741_741594

theorem smallest_number_with_55_divisors : ∃ n : ℕ, 
  (number_of_divisors n = 55) ∧ (∀ m : ℕ, number_of_divisors m = 55 → n ≤ m) := 
sorry

end smallest_number_with_55_divisors_l741_741594


namespace cards_cannot_be_gathered_l741_741939

noncomputable def cards_gathered_at_one_person : Prop :=
  let people := 11
  let cards := List.range' 1 people
  let modulo := (n : ℕ) → n % people
  let triangle_no_acute (i : ℕ) (distributed_cards : ℕ → ℕ) : Prop :=
    let i_minus_1 := modulo (i - 1)
    let i_plus_1 := modulo (i + 1)
    ¬ (condition that i-1, i, i+1 do not form an acute triangle) -- More precisely needed here as geometric condition

  ∀ (initial_distribution : ℕ → ℕ),
    (∀ i, initial_distribution (i + 1) = initial_distribution i + 1) →
    (¬ ∃ p, ∀ i, initial_distribution i = p)
    → (under any allowable moves, the above remains true)

theorem cards_cannot_be_gathered : cards_gathered_at_one_person :=
  sorry

end cards_cannot_be_gathered_l741_741939


namespace value_of_a2_b2_c2_l741_741766

noncomputable def nonzero_reals := { x : ℝ // x ≠ 0 }

theorem value_of_a2_b2_c2 (a b c : nonzero_reals) (h1 : (a : ℝ) + (b : ℝ) + (c : ℝ) = 0) 
  (h2 : (a : ℝ)^3 + (b : ℝ)^3 + (c : ℝ)^3 = (a : ℝ)^7 + (b : ℝ)^7 + (c : ℝ)^7) : 
  (a : ℝ)^2 + (b : ℝ)^2 + (c : ℝ)^2 = 6 / 7 :=
by
  sorry

end value_of_a2_b2_c2_l741_741766


namespace reservoir_pre_storms_percentage_l741_741540

theorem reservoir_pre_storms_percentage :
  let stormA := 80
  let stormB := 150
  let stormC := 45
  let total_deposited := stormA + stormB + stormC
  let current_full_percentage := 90
  let original_contents := 400
  let total_current_contents := original_contents + total_deposited
  let total_capacity := total_current_contents * 10 / current_full_percentage
  let pre_storms_percentage := (original_contents / total_capacity) * 100
  pre_storms_percentage ≈ 53.33 := 
by {
  sorry
}

end reservoir_pre_storms_percentage_l741_741540


namespace class_average_l741_741310

theorem class_average (x : ℝ) :
  (0.25 * 80 + 0.5 * x + 0.25 * 90 = 75) → x = 65 := by
  sorry

end class_average_l741_741310


namespace train_cross_time_l741_741353

noncomputable def speed_conversion (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def time_to_cross_pole (length_m speed_kmh : ℝ) : ℝ :=
  length_m / speed_conversion speed_kmh

theorem train_cross_time (length_m : ℝ) (speed_kmh : ℝ) :
  length_m = 225 → speed_kmh = 250 → time_to_cross_pole length_m speed_kmh = 3.24 := by
  intros hlen hspeed
  simp [time_to_cross_pole, speed_conversion, hlen, hspeed]
  sorry

end train_cross_time_l741_741353


namespace calculations_false_l741_741185

def calculations_performed_from_left_to_right :=
  ∀ (ops : List (ℕ → ℕ → ℕ)) (nums : List ℕ),
    ops.length = nums.length - 1 →
    (ops, nums).foldl (λ acc op_num, op_num.1 acc op_num.2) nums.head! = 
    List.foldl (λ acc num, (ops, nums).1 num (ops, nums).2 num) nums.head! = 
    False

theorem calculations_false :
  ¬ calculations_performed_from_left_to_right :=
by
  sorry

end calculations_false_l741_741185


namespace polynomial_perfect_square_value_of_k_l741_741321

noncomputable def is_perfect_square (p : Polynomial ℝ) : Prop :=
  ∃ (q : Polynomial ℝ), p = q^2

theorem polynomial_perfect_square_value_of_k {k : ℝ} :
  is_perfect_square (Polynomial.X^2 - Polynomial.C k * Polynomial.X + Polynomial.C 25) ↔ (k = 10 ∨ k = -10) :=
by
  sorry

end polynomial_perfect_square_value_of_k_l741_741321


namespace seven_expression_one_seven_expression_two_l741_741900

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l741_741900


namespace boundary_length_is_correct_l741_741156

-- Defining the lengths of the sides of the rectangle
def short_side : ℝ := 8
def long_side : ℝ := 12

-- Defining the segments length portions of the sides
def short_segment_length : ℝ := short_side / 3
def long_segment_length : ℝ := long_side / 3

-- Calculation of the boundary length
def straight_segments_length : ℝ := 4 * short_segment_length + 4 * long_segment_length
def quarter_circles_length : ℝ := (4 * (Float.pi * short_segment_length / 2) + 4 * (Float.pi * long_segment_length / 2)) / 2

-- Total length of the boundary in exact form 
def total_boundary_length_exact : ℝ := straight_segments_length + quarter_circles_length

-- Proposition and proof statement
theorem boundary_length_is_correct :
  total_boundary_length_exact = (80 + 12 * Float.pi) / 3 := 
sorry

end boundary_length_is_correct_l741_741156


namespace minimum_value_expression_l741_741760

theorem minimum_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  3 ≤ (sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x * y) :=
by
  sorry

end minimum_value_expression_l741_741760


namespace solution_fraction_replaced_l741_741096

theorem solution_fraction_replaced 
    (V : ℝ) (x : ℝ)
    (h₁ : 0.80 * V - 0.80 * x * V + 0.25 * x * V = 0.35 * V) :
    x = 9 / 11 :=
by 
  have h2 : 0.80 * V - 0.55 * x * V = 0.35 * V :=
    by rw [← add_sub_assoc, ← sub_eq_add_neg, h₁]
  have h3 : 0.80 * V - 0.35 * V = 0.55 * x * V :=
    by rw [sub_add_eq_sub_sub_swap]; exact h2
  have h4 : 0.45 * V = 0.55 * x * V :=
    by norm_num at h3; exact h3
  have h5 : 0.45 = 0.55 * x :=
    by rw [← mul_inv_cancel (ne_of_gt zero_lt_one)] at h4; exact h4
  have h6 : x = 9 / 11 :=
    by rw [eq_div_iff_mul_eq] at h5; norm_num at h5; exact h5
  exact h6

end solution_fraction_replaced_l741_741096


namespace integer_values_sides_triangle_l741_741850

theorem integer_values_sides_triangle (x : ℝ) (hx_pos : x > 0) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) : 
    (∃ (n : ℤ), ∃ (hn : 0 < n) (hn1 : (n : ℝ) = x) (hn2 : 26 ≤ n) (hn3 : n ≤ 54), 
    ∀ (y : ℤ), (26 ≤ y ∧ y ≤ 54) → (∃ (m : ℤ), y = 26 + m ∧ m < 29 ∧ m ≥ 0)) := 
sorry

end integer_values_sides_triangle_l741_741850


namespace perfect_square_substring_exists_l741_741117

theorem perfect_square_substring_exists (A : Fin 16 → Nat) (h : (∀ i, A i < 10) ∧ (A 0 ≠ 0)) :
  ∃ (i j : Fin 16), i < j ∧ (∏ k in Finset.Icc i j, A k) = n^2 
  for some n : ℕ := sorry

end perfect_square_substring_exists_l741_741117


namespace group_capacity_l741_741470

theorem group_capacity (total_students : ℕ) (selected_students : ℕ) (removed_students : ℕ) :
  total_students = 5008 → selected_students = 200 → removed_students = 8 →
  (total_students - removed_students) / selected_students = 25 :=
by
  intros h1 h2 h3
  sorry

end group_capacity_l741_741470


namespace series_converges_to_1_over_128_l741_741188

-- Definitions of the series terms.
def series_term (n : ℕ) : ℝ := (6 * n - 1) / ((4 * n - 2)^2 * (4 * n + 2)^2)

-- The infinite series sum to be proven convergent and evaluating to 1/128.
def series_sum : ℝ := ∑' n, series_term (n + 1)

theorem series_converges_to_1_over_128 : series_sum = 1 / 128 := by
  sorry

end series_converges_to_1_over_128_l741_741188


namespace infinite_series_sum_l741_741559

theorem infinite_series_sum :
  ∑' n : ℕ, (1 / (n.succ * (n.succ + 2))) = 3 / 4 :=
by sorry

end infinite_series_sum_l741_741559


namespace vasya_100_using_fewer_sevens_l741_741906

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l741_741906


namespace probability_sum_div_by_3_l741_741123

/-- 
A bag contains 30 balls that are numbered 1 through 30. Two balls are randomly chosen from the bag. 
This theorem proves that the probability that the sum of the two numbers is divisible by 3 is 1/3.
-/
theorem probability_sum_div_by_3 :
  (∃ s : set ℕ, s = {1, 2, ..., 30} ∧
      (∀ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a ≠ b → 
        (a + b) % 3 = 0 ↔ rational_number = 1/3)) :=
begin
  sorry
end

end probability_sum_div_by_3_l741_741123


namespace area_of_triangle_l741_741282

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x

-- Derivative of the function f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 3

theorem area_of_triangle (a : ℝ) (hx : f' a 1 = -6) : 
  let l := 6 * (1:ℝ) + (-6 : ℝ) * (1:ℝ) = 6 
  in 1 -
  by simp [f, f', hx]; sorry

end area_of_triangle_l741_741282


namespace max_distance_S_origin_l741_741855

noncomputable def z : ℂ := sorry

def P : ℂ := z
def Q : ℂ := (2 + complex.I) * z
def R : ℂ := 3 * conjugate z
def S : ℂ := Q + R - P

axiom abs_z : abs z = 1
axiom not_collinear : ¬(∃ k : ℝ, Q = k • P ∧ R = k • Q)

theorem max_distance_S_origin : dist S 0 = real.sqrt 10 := by
  sorry

end max_distance_S_origin_l741_741855


namespace color_prob_X_equals_3_l741_741558

-- Define the conditions and the problem setup 
def squares := Finset.range 9

def adjacent (m n : ℕ) : Prop :=
  (m = n + 1 ∨ m = n - 1 ∨ m = n + 3 ∨ m = n - 3)

def colorable (seq : List ℕ) : Prop :=
  seq.length > 0 ∧ seq.length ≤ 9 ∧
  ∀ i j, i < j ∧ j < seq.length ∧ adjacent (seq.get! i) (seq.get! j) → False ∧
  seq.get! (seq.length - 1) = 4  -- Square 5 is the 5th square (zero-indexed 4) and must be the last one colored

def is_valid_coloring (seq : List ℕ) : Prop :=
  (⟨seq, sorry⟩ : {l // l.all (λ n, n ∉ {4} ⊗ ∀ m, adjacent n m → m ∈ seq ∨ m = 4)} : Nat → Bool).Util

noncomputable def X : ℕ :=
  let some seq := (List.filter is_valid_coloring sequences).get 4
  seq.length

theorem color_prob_X_equals_3 : probability (X = 3) = 4 / 9 :=
by 
  sorry

end color_prob_X_equals_3_l741_741558


namespace mixture_ratio_l741_741313

theorem mixture_ratio (V : ℝ) (a b c : ℕ)
  (h_pos : V > 0)
  (h_ratio : V = (3/8) * V + (5/11) * V + ((88 - 33 - 40)/88) * V) :
  a = 33 ∧ b = 40 ∧ c = 15 :=
by
  sorry

end mixture_ratio_l741_741313


namespace slope_angle_parallel_to_x_axis_l741_741452

-- Define the concept of a line being parallel to the x-axis
def is_parallel_to_x_axis (l : Line) : Prop :=
  l.slope = 0

-- State the theorem
theorem slope_angle_parallel_to_x_axis (l : Line) (h : is_parallel_to_x_axis l) : slope_angle l = 0 :=
sorry

end slope_angle_parallel_to_x_axis_l741_741452


namespace example_one_example_two_l741_741885

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l741_741885


namespace shooter_variance_calculation_l741_741968

theorem shooter_variance_calculation : 
  let scores := [9.7, 9.9, 10.1, 10.2, 10.1] in
  let mean := (9.7 + 9.9 + 10.1 + 10.2 + 10.1) / 5 in
  let variance := (1 / 5) * ((9.7 - mean)^2 + (9.9 - mean)^2 + (10.1 - mean)^2 + (10.2 - mean)^2 + (10.1 - mean)^2) in
  variance = 0.032 :=
by
  let scores := [9.7, 9.9, 10.1, 10.2, 10.1]
  let mean : ℝ := (9.7 + 9.9 + 10.1 + 10.2 + 10.1) / 5
  let variance : ℝ := (1 / 5) * ((9.7 - mean)^2 + (9.9 - mean)^2 + (10.1 - mean)^2 + (10.2 - mean)^2 + (10.1 - mean)^2)
  show variance = 0.032
  sorry

end shooter_variance_calculation_l741_741968


namespace find_n_l741_741216

theorem find_n : ∃ (n : ℕ), n^2 * nat.factorial n + nat.factorial n = 5040 ∧ n = 7 :=
by
  sorry

end find_n_l741_741216


namespace magnitude_of_vector_sum_l741_741396

variable (m : ℝ)

def a := (4, m)
def b := (1, -2)
def dot_product (v u : ℝ × ℝ) := v.1 * u.1 + v.2 * u.2
def add_vector (v u : ℝ × ℝ) := (v.1 + u.1, v.2 + u.2)
def scalar_mul (c : ℝ) (v : ℝ × ℝ) := (c * v.1, c * v.2)
def magnitude (v : ℝ × ℝ) := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_vector_sum :
  dot_product a b = 0 →
  magnitude (add_vector a (scalar_mul 2 b)) = 2 * Real.sqrt 10 :=
by
  assume h : dot_product a b = 0
  sorry

end magnitude_of_vector_sum_l741_741396


namespace john_total_strings_l741_741366

theorem john_total_strings :
  let basses := 3
  let strings_per_bass := 4
  let guitars := 2 * basses
  let strings_per_guitar := 6
  let eight_string_guitars := guitars - 3
  let strings_per_eight_string_guitar := 8
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := 
by
  let basses := 3
  let strings_per_bass := 4
  let guitars := 2 * basses
  let strings_per_guitar := 6
  let eight_string_guitars := guitars - 3
  let strings_per_eight_string_guitar := 8
  have H_bass := basses * strings_per_bass
  have H_guitar := guitars * strings_per_guitar
  have H_eight_string_guitar := eight_string_guitars * strings_per_eight_string_guitar
  have H_total := H_bass + H_guitar + H_eight_string_guitar
  show H_total = 72 from sorry

end john_total_strings_l741_741366


namespace main_theorem_l741_741118

-- Define points A, B, C, P on the Euclidean plane.
variables {A B C P : Point ℝ}

-- Define the right-angled triangle condition at vertex C
def is_right_angled_triangle (A B C : Point ℝ) : Prop :=
  ∃ (l1 l2 : Line ℝ), l1 ⊥ l2 ∧ A ∈ l1 ∧ B ∈ l1 ∧ C ∈ l1 ∧ C ∈ l2

-- Internal/external point on the hypotenuse or its extension
def on_hypotenuse_or_extension (P A B : Point ℝ) : Prop :=
  collinear A B P

-- Condition for point P being on the hypotenuse A B or its extension
axiom h1 : on_hypotenuse_or_extension P A B

-- Condition for triangle ABC being right-angled at C
axiom h2 : is_right_angled_triangle A B C

-- Distances between points
def dist (a b : Point ℝ) : ℝ := EuclideanDistance.dist a b

-- Main theorem statement
theorem main_theorem (hPAB : on_hypotenuse_or_extension P A B) (hRT : is_right_angled_triangle A B C) :
  (dist P A * dist B C)^2 + (dist P B * dist C A)^2 = (dist P C * dist A B)^2 :=
sorry

end main_theorem_l741_741118


namespace general_term_l741_741248

-- Definitions for the problem
def a (n : ℕ) : ℕ → ℕ := sorry
def S (n : ℕ) : ℕ := (finset.range n).sum a

-- Conditions
axiom h1 : ∀ (n : ℕ), n ≥ 1 → S n + S (n + 1) + S (n + 2) = 6 * n^2 + 9 * n + 7
axiom h2 : a 1 = 1
axiom h3 : a 2 = 5

-- Theorem to prove
theorem general_term : ∀ n ≥ 1, a n = 4 * n - 3 :=
by
  sorry

end general_term_l741_741248


namespace vasya_100_using_fewer_sevens_l741_741905

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l741_741905


namespace radius_of_larger_circle_l741_741831

theorem radius_of_larger_circle
  (r r_s : ℝ)
  (h1 : r_s = 2)
  (h2 : π * r^2 = 4 * π * r_s^2) :
  r = 4 :=
by
  sorry

end radius_of_larger_circle_l741_741831


namespace max_area_ABC_l741_741729

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

def point_A := (0 : ℝ, 1/16 : ℝ)

def parabola_eq (x y : ℝ) : Prop := y = x^2

def tangent_at_parabola (t x y : ℝ) : Prop := y = 2 * t * x - t^2 + t^2

noncomputable def area_triangle_ABC (t : ℝ) : ℝ := 
  (1/2) * (1 / (1 + 16 * t^2)) * sqrt((1 + 4 * t^2)^2 * (-t^4 + 16 * t^2 + 1)) / 16

theorem max_area_ABC :
  ∃ t : ℝ, area_triangle_ABC t = sqrt 65 / 8 :=
sorry

end max_area_ABC_l741_741729


namespace complete_square_solution_l741_741874

theorem complete_square_solution (x : ℝ) (h : x^2 - 4 * x + 2 = 0) : (x - 2)^2 = 2 := 
by sorry

end complete_square_solution_l741_741874


namespace a_let_b_c_d_e_are_positives_arithmetic_mean_is_10_and_median_is_12_l741_741529

theorem a_let_b_c_d_e_are_positives_arithmetic_mean_is_10_and_median_is_12 
  (a b c d e : ℕ) 
  (h1: a ≤ b) 
  (h2: b ≤ c) 
  (h3: c ≤ d)
  (h4: d ≤ e)
  (h5: (a + b + c + d + e) = 50)
  (h6: c = 12) :
  ∃a b d e, e - a = 5 := by
  sorry

end a_let_b_c_d_e_are_positives_arithmetic_mean_is_10_and_median_is_12_l741_741529


namespace find_exponent_l741_741479

theorem find_exponent (n : ℝ) : 10^n = 10^(-2) * real.sqrt(10^(75) / 10^(-4)) → n = 37.5 :=
by
  intro h
  sorry

end find_exponent_l741_741479


namespace additional_life_vests_needed_l741_741542

def num_students : ℕ := 40
def num_instructors : ℕ := 10
def life_vests_on_hand : ℕ := 20
def percent_students_with_vests : ℕ := 20

def total_people : ℕ := num_students + num_instructors
def students_with_vests : ℕ := (percent_students_with_vests * num_students) / 100
def total_vests_available : ℕ := life_vests_on_hand + students_with_vests

theorem additional_life_vests_needed : 
  total_people - total_vests_available = 22 :=
by 
  sorry

end additional_life_vests_needed_l741_741542


namespace y_equals_4_if_abs_diff_eq_l741_741480

theorem y_equals_4_if_abs_diff_eq (y : ℝ) (h : |y - 3| = |y - 5|) : y = 4 :=
sorry

end y_equals_4_if_abs_diff_eq_l741_741480


namespace equation1_solutions_equation2_solutions_l741_741825

-- Declare the first proof problem for equation (1)
theorem equation1_solutions (x : ℝ) : 
  2 * x * (x + 1) = x + 1 ↔ (x = -1 ∨ x = 1 / 2) := 
begin
  -- We state the structure but do not provide the proof
  sorry
end

-- Declare the second proof problem for equation (2)
theorem equation2_solutions (x : ℝ) : 
  2 * x^2 + 3 * x - 5 = 0 ↔ (x = -5 / 2 ∨ x = 1) := 
begin
  -- We state the structure but do not provide the proof
  sorry
end

end equation1_solutions_equation2_solutions_l741_741825


namespace inversion_fixed_point_l741_741938

-- Given definitions and conditions
variable {A O M N : Point}
variable {S : Circle} (hS : center S = O)
variable {l : Line} (hl : A ∈ l)
variable {hlM : M ∈ l} {hlN : N ∈ l}
variable (hM : M ∈ S) (hN : N ∈ S)
variable (hA : O ≠ A)
variable (hO : O ∉ l)
variable (M' N' : Point) -- Symmetric points
variable (hM' : symmetric M' M OA)
variable (hN' : symmetric N' N OA)
variable (A' : Point)
variable (hA' : intersection (line_through M N') (line_through M' N) A')

-- The theorem to prove that A' coincides with the image of A under inversion with respect to S
theorem inversion_fixed_point :
  A' = inversion A S :=
sorry -- Proof goes here

end inversion_fixed_point_l741_741938


namespace simplify_sqrt_of_three_minus_pi_l741_741821

noncomputable def simplify_sqrt_expr (x : ℝ) : ℝ :=
  real.sqrt ((3 - x)^2)

theorem simplify_sqrt_of_three_minus_pi :
  simplify_sqrt_expr real.pi = real.pi - 3 :=
by sorry

end simplify_sqrt_of_three_minus_pi_l741_741821


namespace quadratic_value_of_q_l741_741154

theorem quadratic_value_of_q (p q : ℝ) (h : ∃ x : ℂ, (3 : ℂ) * x^2 + (p : ℂ) * x + (q : ℂ) = 0 ∧ x = 4 + 3 * complex.I) : q = 75 :=
sorry

end quadratic_value_of_q_l741_741154


namespace count_g_iter_three_l741_741750

def g (n : ℕ) : ℕ :=
  if (n % 2 = 1) then
    n ^ 2 + 3
  else
    n / 2

theorem count_g_iter_three :
  (finset.card (finset.filter (λ n : ℕ, ∃ k : ℕ, (Nat.iterate g k n) = 3) (finset.Icc 1 100))) = 1 :=
by sorry

end count_g_iter_three_l741_741750


namespace find_width_of_brick_l741_741469

noncomputable def width_of_brick (W : ℝ) : Prop :=
  let l_wall := 800 -- 8 m to cm
  let h_wall := 600 -- 6 m to cm
  let t_wall := 22.5 -- already in cm
  let volume_wall := l_wall * h_wall * t_wall
  let l_brick := 100
  let h_brick := 6
  let volume_brick := l_brick * W * h_brick
  let num_bricks := 1600
  volume_wall = num_bricks * volume_brick

theorem find_width_of_brick : width_of_brick 11.25 :=
by
  let l_wall := 800 -- 8 m to cm
  let h_wall := 600 -- 6 m to cm
  let t_wall := 22.5 -- already in cm
  let volume_wall := l_wall * h_wall * t_wall
  let l_brick := 100
  let h_brick := 6
  let W := 11.25
  let volume_brick := l_brick * W * h_brick
  let num_bricks := 1600
  show volume_wall = num_bricks * volume_brick
  sorry

end find_width_of_brick_l741_741469


namespace number_of_correct_statements_is_4_l741_741346

def class_set (r : ℤ) : Set ℤ := { n | ∃ k : ℤ, n = 7 * k + r }

def statement_1 := (2016 ∈ class_set 1) = false
def statement_2 := (-3 ∈ class_set 4) = true
def statement_3 := Disjoint (class_set 3) (class_set 6)
def statement_4 := (Set.univ : Set ℤ) = Set.bigUnion (λ r, class_set r)

def same_class (a b : ℤ) : Prop := (a - b) ∈ class_set 0
def statement_5 := ∀ a b : ℤ, same_class a b ↔ (a - b) ∈ class_set 0

def correct_statements := 1 + 1 + 1 + 1

theorem number_of_correct_statements_is_4 :
    (if statement_1 then 1 else 0) + 
    (if statement_2 then 1 else 0) +
    (if statement_3 then 1 else 0) + 
    (if statement_4 then 1 else 0) + 
    (if statement_5 then 1 else 0) = correct_statements := 
sorry

end number_of_correct_statements_is_4_l741_741346


namespace cannot_be_decomposed_l741_741998

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D :=
  ⟨p1.x - p2.x, p1.y - p2.y, p1.z - p2.z⟩

noncomputable def scalar_mul (k : ℝ) (v : Point3D) : Point3D :=
  ⟨k * v.x, k * v.y, k * v.z⟩

noncomputable def vec_add (v1 v2 : Point3D) : Point3D :=
  ⟨v1.x + v2.x, v1.y + v2.y, v1.z + v2.z⟩

theorem cannot_be_decomposed :
  ∀ (K D1 D A1 : Point3D),
  K = ⟨-3, 7, -7⟩ →
  D1 = ⟨-3, 10, -5⟩ →
  D = ⟨-5, 6, -1⟩ →
  A1 = ⟨1, 6, -7⟩ →
  ¬ ∃ k n : ℝ,
    vector_sub K D1 = vec_add (scalar_mul k (vector_sub D D1)) (scalar_mul n (vector_sub A1 D1)) :=
by 
  intros K D1 D A1 hK hD1 hD hA1
  have v_D1K := vector_sub K D1
  have v_D1D := vector_sub D D1
  have v_D1A1 := vector_sub A1 D1
  rw ←hK at v_D1K
  rw ←hD1 at v_D1D
  rw ←hA1 at v_D1A1
  dsimp [vector_sub, scalar_mul, vec_add] at *
  sorry

end cannot_be_decomposed_l741_741998


namespace interval_of_third_bell_l741_741181

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem interval_of_third_bell :
  ∃ x : ℕ, (lcm (lcm 5 (lcm 8 15)) x = 1320) ∧ (x ∣ 1320) ∧ ¬ (x ∣ 120) :=
begin
  sorry
end

end interval_of_third_bell_l741_741181


namespace example_one_example_two_l741_741888

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l741_741888


namespace sum_of_bi_corresponding_to_odd_ai_l741_741536

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem sum_of_bi_corresponding_to_odd_ai (a b : ℕ) :
  ∀ (a_i b_i : ℕ → ℕ) (n : ℕ),
  (a_i 0 = a) →
  (b_i 0 = b) →
  (∀ i, a_i (i + 1) = if is_odd (a_i i) then (a_i i - 1) / 2 else a_i i / 2) →
  (∀ i, b_i (i + 1) = 2 * b_i i) →
  (a_i n = 1) →
  (∑ i in Finset.range (n + 1), if is_odd (a_i i) then b_i i else 0) = a * b :=
by
  sorry

end sum_of_bi_corresponding_to_odd_ai_l741_741536


namespace fedya_incorrect_l741_741355

theorem fedya_incorrect 
  (a b c d : ℕ) 
  (a_ends_in_9 : a % 10 = 9)
  (b_ends_in_7 : b % 10 = 7)
  (c_ends_in_3 : c % 10 = 3)
  (d_is_1 : d = 1) : 
  a ≠ b * c + d :=
by {
  sorry
}

end fedya_incorrect_l741_741355


namespace angle_AEP_eq_112_5_l741_741813

-- Define the problem parameters and conditions
variables (AB BE EP : ℝ) (E F P : Type*) 
variables (h_AB_midpoint_E : midpoint E A B)
variables (h_BE_ratio : BE F E = 1 / 4)
variables (h_semicircles : true) -- Assume semicircles as described
variables (h_EP_equal_areas : divides_area EP (semicircle AB) (semicircle BE))

-- Define the theorem to be proved
theorem angle_AEP_eq_112_5 (h_AB_midpoint_E : midpoint E A B)
                            (h_BE_ratio : BE F E = 1 / 4)
                            (h_semicircles : true)
                            (h_EP_equal_areas : divides_area EP (semicircle AB) (semicircle BE)) :
  angle A E P = 112.5 :=
by sorry

end angle_AEP_eq_112_5_l741_741813


namespace jason_borrowed_amount_l741_741740

theorem jason_borrowed_amount (hours := 25) :
  let cycle_length := 7 
  let earnings_per_cycle := 1 + 2 + 3 + 4 + 5 + 6 + 7 
  let full_cycles := hours / cycle_length 
  let remaining_hours := hours % cycle_length 
  let earnings_from_full_cycles := full_cycles * earnings_per_cycle 
  let earnings_from_remaining_hours := list.sum (list.map (λ h, h) [1,2,3,4,5,6,7].take remaining_hours) 
  total_earnings = earnings_from_full_cycles + earnings_from_remaining_hours := 
by
  sorry

end jason_borrowed_amount_l741_741740


namespace minimum_throws_for_repeated_sum_l741_741049

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741049


namespace nth_term_2011_is_671st_l741_741120

noncomputable def arithmetic_sequence_nth_term (a d n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem nth_term_2011_is_671st :
  let a := 1
  let d := 3
  let n := 671
  nth_term : arithmetic_sequence_nth_term a d n = 2011 :=
begin
  sorry
end

end nth_term_2011_is_671st_l741_741120


namespace james_marbles_left_l741_741359

def marbles_remain (total_marbles : ℕ) (bags : ℕ) (given_away : ℕ) : ℕ :=
  (total_marbles / bags) * (bags - given_away)

theorem james_marbles_left :
  marbles_remain 28 4 1 = 21 := 
by
  sorry

end james_marbles_left_l741_741359


namespace tourist_distribution_correct_l741_741328

variable (X_0 X_1 X_2 X_3 X_4 X_5 : Type)
variable (tourist_count : X_0 → ℕ)
variable (X1_visitor_count X2_visitor_count X3_visitor_count X4_visitor_count X5_visitor_count : ℕ)

-- Given conditions
axiom tourists_total : tourist_count X_0 = 80
axiom totally_visited_X1 : ∀ tourists : X_0, tourist_count X_1 = 40
axiom totally_visited_X2 : ∀ tourists : X_0, tourist_count X_2 = 60
axiom totally_visited_X3 : ∀ tourists : X_0, tourist_count X_3 = 65
axiom totally_visited_X4 : ∀ tourists : X_0, tourist_count X_4 = 70
axiom totally_visited_X5 : ∀ tourists : X_0, tourist_count X_5 = 75

-- Correct answers
def initial_choice_counts (x1 x2 x3 x4 x5: ℕ) : Prop :=
  x1 = 40 ∧ x2 = 20 ∧ x3 = 5 ∧ x4 = 5 ∧ x5 = 5

def forced_visit_pairs : Prop := 
  (∀ i j : ℕ, i = 1 → j = 2) ∧ (∀ i j : ℕ, i = 2 → j = 3) ∧ (∀ i j : ℕ, i = 3 → j = 4) ∧ (∀ i j : ℕ, i = 4 → j = 5)

theorem tourist_distribution_correct : 
  ∃ (x1 x2 x3 x4 x5: ℕ), 
    initial_choice_counts x1 x2 x3 x4 x5 ∧ forced_visit_pairs :=
sorry

end tourist_distribution_correct_l741_741328


namespace area_of_B_l741_741837

variable (d_A : ℝ) (x : ℝ)
def d_B : ℝ := 3 * d_A
def area_of_square_B : ℝ := (d_B / Math.sqrt 2)^2

theorem area_of_B (h : d_A = x) : area_of_square_B d_A = 9 * x^2 / 2 :=
by sorry

end area_of_B_l741_741837


namespace circles_disjoint_l741_741291

-- Definitions of the circles
def circleM (x y : ℝ) : Prop := x^2 + y^2 = 1
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Prove that the circles are disjoint
theorem circles_disjoint : 
  (¬ ∃ (x y : ℝ), circleM x y ∧ circleN x y) :=
by sorry

end circles_disjoint_l741_741291


namespace volume_calculation_l741_741557

-- Define the functions as per the conditions
def f (x : ℝ) := x^2 + 1
def g (x : ℝ) := x

noncomputable def volume_of_rotation (y1 y2 : ℝ) : ℝ :=
  let f_inv (y : ℝ) := real.sqrt (y - 1)
  let g_inv (y : ℝ) := y
  π * ∫ y in y1..y2, (f_inv y)^2 - (g_inv y)^2

theorem volume_calculation : volume_of_rotation 1 2 = (5 * π) / 6 :=
  sorry

end volume_calculation_l741_741557


namespace smallest_nat_with_55_divisors_l741_741596

open BigOperators

theorem smallest_nat_with_55_divisors :
  ∃ (n : ℕ), 
    (∃ (f : ℕ → ℕ) (primes : Finset ℕ),
      (∀ p ∈ primes, Nat.Prime p) ∧ 
      (primes.Sum (λ p => p ^ (f p))) = n ∧
      ((primes.Sum (λ p => f p + 1)) = 55)) ∧
    (∀ m, 
      (∃ (f_m : ℕ → ℕ) (primes_m : Finset ℕ),
        (∀ p ∈ primes_m, Nat.Prime p) ∧ 
        (primes_m.Sum (λ p => p ^ (f_m p))) = m ∧
        ((primes_m.Sum (λ p => f_m p + 1)) = 55)) → 
      n ≤ m) ∧
  n = 3^4 * 2^10 := 
begin
  sorry
end

end smallest_nat_with_55_divisors_l741_741596


namespace non_monotonic_range_k_l741_741279

variable {k : ℝ}

def f (x : ℝ) : ℝ := x^3 - k * x

def f' (x : ℝ) : ℝ := 3 * x^2 - k

theorem non_monotonic_range_k :
  ¬ monotoneOn f (set.Ioo (-3 : ℝ) (-1)) ↔ 3 < k ∧ k < 27 := by
  sorry

end non_monotonic_range_k_l741_741279


namespace rectangular_farm_fencing_cost_correct_l741_741157

noncomputable def rectangular_farm_fencing_cost (A : ℝ) (W : ℝ) (cost_long : ℝ) (cost_short : ℝ) (cost_diag : ℝ) : ℝ :=
  let L := A / W in
  let diagonal := real.sqrt (L^2 + W^2) in
  cost_long * L + cost_short * W + cost_diag * diagonal

theorem rectangular_farm_fencing_cost_correct :
  rectangular_farm_fencing_cost 1200 30 16 14 18 = 1960 :=
by norm_num; sorry

end rectangular_farm_fencing_cost_correct_l741_741157


namespace find_solution_l741_741132

variables (x y : ℝ)

def salt_concentration (x y : ℝ) : Prop := 0.45 * x = 0.15 * (x + y + 1)
def sugar_concentration (x y : ℝ) : Prop := 0.30 * y = 0.05 * (x + y + 1)

theorem find_solution (x y : ℝ) (h1 : salt_concentration x y) (h2 : sugar_concentration x y) : 
  x = 2 / 3 ∧ y = 1 / 3 :=
begin
  sorry
end

end find_solution_l741_741132


namespace john_total_strings_l741_741368

theorem john_total_strings :
  let basses := 3
  let strings_per_bass := 4
  let guitars := 2 * basses
  let strings_per_guitar := 6
  let eight_string_guitars := guitars - 3
  let strings_per_eight_string_guitar := 8
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := 
by
  let basses := 3
  let strings_per_bass := 4
  let guitars := 2 * basses
  let strings_per_guitar := 6
  let eight_string_guitars := guitars - 3
  let strings_per_eight_string_guitar := 8
  have H_bass := basses * strings_per_bass
  have H_guitar := guitars * strings_per_guitar
  have H_eight_string_guitar := eight_string_guitars * strings_per_eight_string_guitar
  have H_total := H_bass + H_guitar + H_eight_string_guitar
  show H_total = 72 from sorry

end john_total_strings_l741_741368


namespace work_completion_l741_741095

theorem work_completion (A_complete_days B_remaining_work_days A_work_days total_work_fraction : ℕ) 
(h1 : A_complete_days = 15) 
(h2 : B_remaining_work_days = 12) 
(h3 : A_work_days = 5) 
(h4 : total_work_fraction = 1) : 
  (B_complete_days : ℕ) := 
  let A_work_fraction := A_work_days / A_complete_days
  let remaining_work_fraction := total_work_fraction - A_work_fraction
  let B_complete_days := B_remaining_work_days / remaining_work_fraction
  B_complete_days = 18 := sorry

end work_completion_l741_741095


namespace solve_equation_l741_741937

theorem solve_equation : ∃! x : ℝ, 2 * sin (π * x / 2) - 2 * cos (π * x / 2) = x^5 + 10 * x - 54 :=
begin
  -- Sorry is added to skip the proof in this step.
  sorry
end

end solve_equation_l741_741937


namespace tangent_problem_max_value_problem_l741_741278

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Tangency Problem
theorem tangent_problem {a : ℝ} :
  (∃ x : ℝ, f(a) x = 3 * x - 1 ∧ (∀ x : ℝ, deriv (f(a)) x = 3) ) → a = -2 :=
  by sorry

-- Maximum Value Problem
theorem max_value_problem {a : ℝ} (h_max : ∀ x ∈ Set.Icc (1:ℝ) (Real.exp 2), f(a) x ≤ 1 - a * Real.exp 1) :
  a = (1 : ℝ) / Real.exp 1 :=
  by sorry

end tangent_problem_max_value_problem_l741_741278


namespace possible_points_P_infinite_l741_741752

noncomputable def P_lies_within_circle (P : ℝ × ℝ) : Prop :=
  let r : ℝ := 2 in
  P.1^2 + P.2^2 <= r^2

noncomputable def squared_distances_sum (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ :=
  let dist_A := (P.1 - A.1)^2 + (P.2 - A.2)^2 in
  let dist_B := (P.1 - B.1)^2 + (P.2 - B.2)^2 in
  dist_A + dist_B

theorem possible_points_P_infinite :
  let A := (-2, 0) in
  let B := (2, 0) in
  ∃ P : (ℝ × ℝ) → Prop, (P_lies_within_circle P ∧ squared_distances_sum P A B = 4.5) = ∞ :=
by
  sorry

end possible_points_P_infinite_l741_741752


namespace number_of_integer_values_of_x_l741_741849

theorem number_of_integer_values_of_x (x : ℕ) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) :
  ∃ n : ℕ, n = 29 ∧ ∀ y : ℕ, (26 ≤ y ∧ y ≤ 54) ↔ true :=
by
  sorry

end number_of_integer_values_of_x_l741_741849


namespace find_C_S_l741_741751

def C (A : Set ℝ) : ℤ := A.count

def A_star_B (A B : Set ℝ) : ℤ := 
  if C A ≥ C B then C A - C B
  else C B - C A

def A : Set ℝ := {1, 2}

def B (a : ℝ) : Set ℝ := { x | (x + a) * (x^3 + a * x^2 + 2 * x) = 0 }

def S : Set ℝ := {a | A_star_B A (B a) = 1 }

theorem find_C_S : C S = 3 :=
sorry

end find_C_S_l741_741751


namespace correlation_statements_l741_741928

def heavy_snow_predicts_harvest_year (heavy_snow benefits_wheat : Prop) : Prop := benefits_wheat → heavy_snow
def great_teachers_produce_students (great_teachers outstanding_students : Prop) : Prop := great_teachers → outstanding_students
def smoking_is_harmful (smoking harmful_to_health : Prop) : Prop := smoking → harmful_to_health
def magpies_call_signifies_joy (magpies_call joy_signified : Prop) : Prop := joy_signified → magpies_call

theorem correlation_statements (heavy_snow benefits_wheat great_teachers outstanding_students smoking harmful_to_health magpies_call joy_signified : Prop)
  (H1 : heavy_snow_predicts_harvest_year heavy_snow benefits_wheat)
  (H2 : great_teachers_produce_students great_teachers outstanding_students)
  (H3 : smoking_is_harmful smoking harmful_to_health) :
  ¬ magpies_call_signifies_joy magpies_call joy_signified := sorry

end correlation_statements_l741_741928


namespace coeff_x3_in_expansion_l741_741432

theorem coeff_x3_in_expansion : 
  ∀ (x y : ℝ), ∃ (c : ℝ), 
  (c = 15) ∧ coeff (expand (λ x y, (x / real.sqrt y - y / real.sqrt x)^6) x 3) = c :=
by sorry

end coeff_x3_in_expansion_l741_741432


namespace interval_monotonic_increase_l741_741845

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem interval_monotonic_increase : 
  {x : ℝ | f'$ f x > 0} = {x : ℝ | x > Real.exp (-1)} := 
by 
  sorry

end interval_monotonic_increase_l741_741845


namespace find_f_neg_half_l741_741666

#check @Function
#check Real.pow

variable {a : ℝ}
variable {f : ℝ → ℝ}

theorem find_f_neg_half (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x, f x = a ^ x) (h4 : f 2 = 81) :
  f (-1/2) = 1/3 :=
by
  sorry

end find_f_neg_half_l741_741666


namespace polynomial_f_mod6_l741_741288

-- Given conditions
variable (n : ℕ)
variable (a : Fin n.succ → ℤ)
def f (x : ℤ) : ℤ := ∑ i in Finset.range (n + 1), a ⟨i, Nat.lt_succ_self _⟩ * x ^ i

-- Main theorem to prove
theorem polynomial_f_mod6 (h2 : 6 ∣ f n a 2) (h3 : 6 ∣ f n a 3) : 6 ∣ f n a 5 := sorry

end polynomial_f_mod6_l741_741288


namespace min_buses_needed_l741_741967

theorem min_buses_needed (x y : ℕ) (h1 : 45 * x + 35 * y ≥ 530) (h2 : y ≥ 3) : x + y = 13 :=
by
  sorry

end min_buses_needed_l741_741967


namespace total_visitors_gorilla_exhibit_is_404_l741_741860

def visitors_per_hour : List ℝ := [ 50, 70, 90, 100, 70, 60, 80, 50 ]
def percentages_per_hour : List ℝ := [ 0.80, 0.75, 0.90, 0.40, 0.85, 0.70, 0.60, 0.80 ]

def visitors_per_gorilla_exhibit (visitors : List ℝ) (percentages : List ℝ) : List ℝ :=
  List.map₂ (fun v p => v * p) visitors percentages

noncomputable def total_visitors_to_gorilla_exhibit : ℝ :=
  (∑ v in visitors_per_gorilla_exhibit visitors_per_hour percentages_per_hour, v).round

theorem total_visitors_gorilla_exhibit_is_404 : total_visitors_to_gorilla_exhibit = 404 := 
  by
  sorry

end total_visitors_gorilla_exhibit_is_404_l741_741860


namespace even_function_period_not_pi_symmetric_about_pi_range_of_f_l741_741664

def f (x : ℝ) : ℝ := abs (Real.sin x) + Real.cos x

theorem even_function : ∀ x : ℝ, f (-x) = f x := by
  sorry

theorem period_not_pi : ∃ x : ℝ, f (x + Real.pi) ≠ f x := by
  sorry

theorem symmetric_about_pi : ∀ x : ℝ, f (Real.pi + x) = f (Real.pi - x) := by
  sorry

theorem range_of_f : ∀ y : ℝ, (y ∈ set.range f) ↔ (y ≥ -1 ∧ y ≤ Real.sqrt 2) := by
  sorry

end even_function_period_not_pi_symmetric_about_pi_range_of_f_l741_741664


namespace find_s_l741_741725

-- Conditions definitions
variables (A B C D O : Type) -- arbitrary points
variables [parallelogram A B C D] -- ABCD forms a parallelogram
variables (φ : Real) -- angle φ for DBA
variables (h_intersect : is_intersection O (diagonal A C) (diagonal B D))
variables (h_angle_proportions : ∀ {x y z O},
  is_angle (x y z) → is_angle (x z y) → 
  x = A ∧ y = B ∧ z = O ∨ x = D ∧ y = B ∧ z = C ∨ (angle x y z) = φ → 
  (angle A C B) = (3 / 2) * φ ∧ (angle D B C) = (3 / 2) * φ)
variables (s : ℝ)
variables (h_s_angle: (angle A C B) = s * (angle A O B))

-- Proof statement
theorem find_s : s = 3 := 
  sorry

end find_s_l741_741725


namespace minimum_rolls_for_duplicate_sum_l741_741060

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741060


namespace find_f_of_2_l741_741263

-- Definitions based on problem conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (x) + 9

-- The main statement to proof that f(2) = 6 under the given conditions
theorem find_f_of_2 (f : ℝ → ℝ)
  (hf : is_odd_function f)
  (hg : ∀ x, g f x = f x + 9)
  (h : g f (-2) = 3) :
  f 2 = 6 := 
sorry

end find_f_of_2_l741_741263


namespace percentage_division_l741_741942

theorem percentage_division :
  let percentage := 208 in
  let base := 100 in
  let number := 1265 in
  let divisor := 6 in
  ((percentage / base : ℝ) * number) / divisor = 437.8666666666667 :=
by {
  -- Proof should go here
  sorry
}

end percentage_division_l741_741942


namespace score_ordering_l741_741931

-- Definitions of the scores
variables (L H M Y : ℕ)

-- The conditions
def condition1 : Prop := L + H = M + Y
def condition2 : Prop := Y + L > M + H
def condition3 : Prop := H > M + L

-- The conclusion we want to establish
def conclusion : Prop := Y > H ∧ H > L ∧ L > M

-- The final proof problem statement in Lean 4
theorem score_ordering (h1 : condition1 L H M Y) (h2 : condition2 L H M Y) (h3 : condition3 L H M Y) : conclusion L H M Y :=
sorry

end score_ordering_l741_741931


namespace length_of_platform_l741_741100

/--
Problem statement:
A train 450 m long running at 108 km/h crosses a platform in 25 seconds.
Prove that the length of the platform is 300 meters.

Given:
- The train is 450 meters long.
- The train's speed is 108 km/h.
- The train crosses the platform in 25 seconds.

To prove:
The length of the platform is 300 meters.
-/
theorem length_of_platform :
  let train_length := 450
  let train_speed := 108 * (1000 / 3600) -- converting km/h to m/s
  let crossing_time := 25
  let total_distance_covered := train_speed * crossing_time
  let platform_length := total_distance_covered - train_length
  platform_length = 300 := by
  sorry

end length_of_platform_l741_741100


namespace minimum_throws_l741_741027

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741027


namespace evaluate_v2_at_x_value_l741_741873

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^6 + 6 * x^4 + 9 * x^2 + 208

-- Define the value of x at which we are evaluating
def x_value : ℝ := -4

-- Define v2
def v2 (x : ℝ) : ℝ := x^2 + 6

-- The proof statement that evaluates v2 at x_value and checks if it equals 22
theorem evaluate_v2_at_x_value : v2 x_value = 22 :=
by
  have h1 : v2 x_value = (-4)^2 + 6 := by rfl
  simp [v2, x_value] at h1
  exact h1


end evaluate_v2_at_x_value_l741_741873


namespace vasya_example_fewer_sevens_l741_741881

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l741_741881


namespace probability_units_digit_even_l741_741141

theorem probability_units_digit_even : 
  let num_digits := 5
  let total_digits := 9 - 0 + 1
  let even_digits := 5
  0 < num_digits ∧ num_digits == 5 ∧ total_digits == 10 ∧ even_digits == 5 ↔ (sorry : (even_digits : ℝ) / total_digits == (1 / 2))

end probability_units_digit_even_l741_741141


namespace solution_set_l741_741574

noncomputable def f (x : ℝ) : ℝ := x^2 - 3*x + 2 - real.sqrt (x + 4)

theorem solution_set :
  {x : ℝ | x^2 - 3*x + 2 - real.sqrt (x + 4) < 0} = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end solution_set_l741_741574


namespace triangle_angle_problem_l741_741539

open EuclideanGeometry

variables {A B C D E : Point}
variables (AB AC BC : LineSegment) (∠ : Angle)
variables (m : Midpoint D AB) (n : ∣ BE = 2 * EC) (∠1 : ∠ ADC) (∠2 : ∠ BAE)

theorem triangle_angle_problem 
  (h1 : is_triangle A B C)
  (h2 : midpoint D AB)
  (h3 : E ∈ line_segment B C)
  (h4 : ∣ line_segment B E = 2 * ∣ line_segment E C)
  (h5 : ∠ A D C = ∠ B A E) :
  ∠ B A C = 30 :=
sorry

end triangle_angle_problem_l741_741539


namespace chess_tournament_l741_741506

theorem chess_tournament (n : ℕ) (players : fin 2n → ℕ) (tournament_score : fin 2n → ℕ) :
  (∀ i, abs (tournament_score i - players i) ≥ n) →
  (∀ i, abs (tournament_score i - players i) = n) :=
by
  sorry

end chess_tournament_l741_741506


namespace min_throws_to_ensure_same_sum_twice_l741_741065

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741065


namespace gcd_seq_coprime_l741_741803

def seq (n : ℕ) : ℕ := 2^(2^n) + 1

theorem gcd_seq_coprime (n k : ℕ) (hnk : n ≠ k) : Nat.gcd (seq n) (seq k) = 1 :=
by
  sorry

end gcd_seq_coprime_l741_741803


namespace find_b_l741_741755

noncomputable def a : ℂ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℂ := sorry

-- Given conditions
axiom sum_eq : a + b + c = 4
axiom prod_pairs_eq : a * b + b * c + c * a = 5
axiom prod_triple_eq : a * b * c = 6

-- Prove that b = 1
theorem find_b : b = 1 :=
by
  -- Proof omitted
  sorry

end find_b_l741_741755


namespace condition_1_valid_for_n_condition_2_valid_for_n_l741_741234

-- Definitions from the conditions
def is_cube_root_of_unity (ω : ℂ) : Prop := ω^3 = 1

def roots_of_polynomial (ω : ℂ) (ω2 : ℂ) : Prop :=
  ω^2 + ω + 1 = 0 ∧ is_cube_root_of_unity ω ∧ is_cube_root_of_unity ω2

-- Problem statements
theorem condition_1_valid_for_n (n : ℕ) (ω : ℂ) (ω2 : ℂ) (h : roots_of_polynomial ω ω2) :
  (x^2 + x + 1) ∣ (x+1)^n - x^n - 1 ↔ ∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k - 1 := sorry

theorem condition_2_valid_for_n (n : ℕ) (ω : ℂ) (ω2 : ℂ) (h : roots_of_polynomial ω ω2) :
  (x^2 + x + 1) ∣ (x+1)^n + x^n + 1 ↔ ∃ k : ℕ, n = 6 * k + 2 ∨ n = 6 * k - 2 := sorry

end condition_1_valid_for_n_condition_2_valid_for_n_l741_741234


namespace arithmetic_sequence_length_l741_741298

theorem arithmetic_sequence_length :
  ∃ n, (2 + (n - 1) * 5 = 3007) ∧ n = 602 :=
by
  use 602
  sorry

end arithmetic_sequence_length_l741_741298


namespace probability_red_joker_is_1_over_54_l741_741148

-- Define the conditions as given in the problem
def total_cards : ℕ := 54
def red_joker_count : ℕ := 1

-- Define the function to calculate the probability
def probability_red_joker_top_card : ℚ := red_joker_count / total_cards

-- Problem: Prove that the probability of drawing the red joker as the top card is 1/54
theorem probability_red_joker_is_1_over_54 :
  probability_red_joker_top_card = 1 / 54 :=
by
  sorry

end probability_red_joker_is_1_over_54_l741_741148


namespace smallest_number_with_55_divisors_l741_741601

theorem smallest_number_with_55_divisors : ∃ (n : ℕ), (∃ (p : ℕ → ℕ) (k : ℕ → ℕ) (m : ℕ), 
  n = ∏ i in finset.range m, (p i)^(k i) ∧ (∀ i j, i ≠ j → nat.prime (p i) → nat.prime (p j) → p i ≠ p j) ∧ 
  (finset.range m).card = m ∧ 
  (∏ i in finset.range m, (k i + 1) = 55)) ∧ 
  n = 3^4 * 2^10 then n = 82944 :=
by
  sorry

end smallest_number_with_55_divisors_l741_741601


namespace minimum_perimeter_l741_741779

def fractional_part (x : ℚ) : ℚ := x - x.floor

-- Define l, m, n being sides of the triangle with l > m > n
variables (l m n : ℤ)

-- Defining conditions as Lean predicates
def triangle_sides (l m n : ℤ) : Prop := l > m ∧ m > n

def fractional_part_condition (l m n : ℤ) : Prop :=
  fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4) ∧
  fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)

-- Prove the minimum perimeter is 3003 given above conditions
theorem minimum_perimeter (l m n : ℤ) :
  triangle_sides l m n →
  fractional_part_condition l m n →
  l + m + n = 3003 :=
by
  intros h_sides h_fractional
  sorry

end minimum_perimeter_l741_741779


namespace skipping_rope_equation_correct_l741_741929

-- Definitions of constraints
variable (x : ℕ) -- Number of skips per minute by Xiao Ji
variable (H1 : 0 < x) -- The number of skips per minute by Xiao Ji is positive
variable (H2 : 100 / x * x = 100) -- Xiao Ji skips exactly 100 times

-- Xiao Fan's conditions
variable (H3 : 100 + 20 = 120) -- Xiao Fan skips 20 more times than Xiao Ji
variable (H4 : x + 30 > 0) -- Xiao Fan skips 30 more times per minute than Xiao Ji

-- Prove the equation is correct
theorem skipping_rope_equation_correct :
  100 / x = 120 / (x + 30) :=
by
  sorry

end skipping_rope_equation_correct_l741_741929


namespace sum_lengths_XYZ_l741_741962

-- Define the length of slanted segments using the Pythagorean theorem.
def diagonal_length (a b : ℝ) : ℝ := real.sqrt (a ^ 2 + b ^ 2)

-- Conditions derived from the problem statement
def straight_segment_length : ℝ := 1
def slanted_segment_length : ℝ := diagonal_length 1 1

-- Segment counts for each letter
def X_segments : ℕ := 2  -- "X" has 2 slanted segments
def Y_segments_straight : ℕ := 3 -- "Y" has 3 straight segments
def Z_segments_straight : ℕ := 2 -- "Z" has 2 straight segments
def Z_segments_slanted : ℕ := 1  -- "Z" has 1 slanted segment

-- Calculate total lengths
def total_straight_segments : ℕ := Y_segments_straight + Z_segments_straight
def total_slanted_segments : ℕ := X_segments + Z_segments_slanted

def total_length : ℝ := total_straight_segments * straight_segment_length 
                      + total_slanted_segments * slanted_segment_length

-- Theorem to be proved
theorem sum_lengths_XYZ : total_length = 5 + 3 * real.sqrt 2 := 
by
  sorry

end sum_lengths_XYZ_l741_741962


namespace james_marbles_left_l741_741360

theorem james_marbles_left (initial_marbles : ℕ) (total_bags : ℕ) (marbles_per_bag : ℕ) (bags_given_away : ℕ) : 
  initial_marbles = 28 → total_bags = 4 → marbles_per_bag = initial_marbles / total_bags → bags_given_away = 1 → 
  initial_marbles - marbles_per_bag * bags_given_away = 21 :=
by
  intros h_initial h_total h_each h_given
  sorry

end james_marbles_left_l741_741360


namespace total_strings_needed_l741_741363

def basses := 3
def strings_per_bass := 4
def guitars := 2 * basses
def strings_per_guitar := 6
def eight_string_guitars := guitars - 3
def strings_per_eight_string_guitar := 8

theorem total_strings_needed :
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := by
  sorry

end total_strings_needed_l741_741363


namespace scoops_per_tub_l741_741471

theorem scoops_per_tub (pieces_per_pan : ℕ) (pans_baked : ℕ) (fraction_eaten_of_second_pan : ℚ) (scoops_per_piece : ℕ) (tubs_eaten : ℕ) :
  pieces_per_pan = 16 ->
  pans_baked = 2 ->
  fraction_eaten_of_second_pan = 0.75 ->
  scoops_per_piece = 2 ->
  tubs_eaten = 6 ->
  (ceil ((pans_baked * pieces_per_pan - (pieces_per_pan - fraction_eaten_of_second_pan * pieces_per_pan)) * scoops_per_piece / tubs_eaten)) = 10 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end scoops_per_tub_l741_741471


namespace more_likely_second_machine_l741_741615

variable (P_B1 : ℝ := 0.8) -- Probability that a part is from the first machine
variable (P_B2 : ℝ := 0.2) -- Probability that a part is from the second machine
variable (P_A_given_B1 : ℝ := 0.01) -- Probability that a part is defective given it is from the first machine
variable (P_A_given_B2 : ℝ := 0.05) -- Probability that a part is defective given it is from the second machine

noncomputable def P_A : ℝ :=
  P_B1 * P_A_given_B1 + P_B2 * P_A_given_B2

noncomputable def P_B1_given_A : ℝ :=
  (P_B1 * P_A_given_B1) / P_A

noncomputable def P_B2_given_A : ℝ :=
  (P_B2 * P_A_given_B2) / P_A

theorem more_likely_second_machine :
  P_B2_given_A > P_B1_given_A :=
by
  sorry

end more_likely_second_machine_l741_741615


namespace annual_interest_rate_approx_l741_741742

noncomputable def find_annual_rate (P A n t : ℝ) : ℝ := 
  let r_approx := (A / P)^(1 / (n * t)) - 1
  2 * r_approx

theorem annual_interest_rate_approx {P A : ℝ} (hP : P = 10000) (hA : A = 10815.83) 
  (hn : n = 2) (ht : t = 2) : 
  find_annual_rate 10000 10815.83 2 2 ≈ 0.0398 :=
by
  unfold find_annual_rate
  rw [hP, hA, hn, ht]
  -- We assume here that the result of the calculations gives approximately 0.0398.
  -- The exact numeric handling part (≈) should be justified by numerical computation tools outside Lean.
  sorry

end annual_interest_rate_approx_l741_741742


namespace product_of_sums_of_four_squares_is_sum_of_four_squares_l741_741449

/-- The product of two numbers, each of which is the sum of four squares,
    is also equal to the sum of four squares according to Euler's identity. -/
theorem product_of_sums_of_four_squares_is_sum_of_four_squares
(p q r s p₁ q₁ r₁ s₁ : ℤ) :
  (p^2 + q^2 + r^2 + s^2) * (p₁^2 + q₁^2 + r₁^2 + s₁^2) =
  ∃ P Q R S : ℤ, P^2 + Q^2 + R^2 + S^2 :=
by
  sorry

end product_of_sums_of_four_squares_is_sum_of_four_squares_l741_741449


namespace problem_statement_l741_741312

theorem problem_statement (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 6 = 104 := by
  sorry

end problem_statement_l741_741312


namespace minimum_throws_l741_741026

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741026


namespace samantha_average_speed_l741_741811

theorem samantha_average_speed :
  let d1 := 50
  let s1 := 15
  let d2 := 20
  let s2 := 25
  let total_distance := d1 + d2
  let time1 := d1 / s1
  let time2 := d2 / s2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed ≈ 16.94 :=
by
  let d1 := 50
  let s1 := 15
  let d2 := 20
  let s2 := 25
  let total_distance := d1 + d2
  let time1 := d1 / s1
  let time2 := d2 / s2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  sorry

end samantha_average_speed_l741_741811


namespace intersection_of_sets_l741_741777

def setA : Set ℝ := { x | x^2 - 4*x + 3 < 0 }
def setB : Set ℝ := { x | 2*x - 3 > 0 }

theorem intersection_of_sets : setA ∩ setB = { x : ℝ | x > 3/2 ∧ x < 3 } :=
  by sorry

end intersection_of_sets_l741_741777


namespace number_of_purchasing_methods_l741_741126

theorem number_of_purchasing_methods : 
  (nat.choose 8 5) + ((nat.choose 8 4) * (nat.choose 3 2)) = 266 :=
by
  sorry

end number_of_purchasing_methods_l741_741126


namespace smallest_number_with_55_divisors_l741_741602

theorem smallest_number_with_55_divisors : ∃ (n : ℕ), (∃ (p : ℕ → ℕ) (k : ℕ → ℕ) (m : ℕ), 
  n = ∏ i in finset.range m, (p i)^(k i) ∧ (∀ i j, i ≠ j → nat.prime (p i) → nat.prime (p j) → p i ≠ p j) ∧ 
  (finset.range m).card = m ∧ 
  (∏ i in finset.range m, (k i + 1) = 55)) ∧ 
  n = 3^4 * 2^10 then n = 82944 :=
by
  sorry

end smallest_number_with_55_divisors_l741_741602


namespace monotonic_increasing_m_ge_neg4_l741_741707

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ a → y > x → f y ≥ f x

def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x - 2

theorem monotonic_increasing_m_ge_neg4 (m : ℝ) :
  is_monotonic_increasing (f m) 2 → m ≥ -4 :=
by
  sorry

end monotonic_increasing_m_ge_neg4_l741_741707


namespace evaluate_expression_l741_741212

def a : ℕ := 3
def b : ℕ := 4

theorem evaluate_expression :
  2 * ((a^b)^a - (b^a)^b) = -32471550 := by
  sorry

end evaluate_expression_l741_741212


namespace oranges_taken_from_basket_l741_741464

-- Define the original number of oranges and the number left after taking some out.
def original_oranges : ℕ := 8
def oranges_left : ℕ := 3

-- Prove that the number of oranges taken from the basket equals 5.
theorem oranges_taken_from_basket : original_oranges - oranges_left = 5 := by
  sorry

end oranges_taken_from_basket_l741_741464


namespace insurance_compensation_correct_l741_741112

def actual_damage : ℝ := 300000
def deductible_percent : ℝ := 0.01
def deductible_amount : ℝ := deductible_percent * actual_damage
def insurance_compensation : ℝ := actual_damage - deductible_amount

theorem insurance_compensation_correct : insurance_compensation = 297000 :=
by
  -- To be proved
  sorry

end insurance_compensation_correct_l741_741112


namespace x_coordinate_of_P_l741_741660

theorem x_coordinate_of_P (x : ℝ) : 
(∃ (P : ℝ × ℝ), P = (x, 4 / x) ∧ (x = 2 ∨ x = -2) ∧ 
(∃ (tangent_slope_at_P : ℝ), tangent_slope_at_P = -4 / x^2) ∧ 
(∃ (slope_at_1_0 : ℝ), slope_at_1_0 = 1 + log 1 ∧ slope_at_1_0 = 1) ∧ 
(tangent_slope_at_P * slope_at_1_0 = -1)) := 
begin
  sorry
end

end x_coordinate_of_P_l741_741660


namespace distinct_distributions_of_balls_l741_741199

theorem distinct_distributions_of_balls :
  ∃ (f : Fin 3 → ℕ), (∀ i, 1 ≤ f i) ∧ (f 0 + f 1 + f 2 = 9) ∧ (f 0 ≠ f 1 ∧ f 1 ≠ f 2 ∧ f 2 ≠ f 0) ∧
    (Finset.univ.filter (λ f, ∀ i, 1 ≤ f i ∧ f 0 + f 1 + f 2 = 9).card = 18) :=
sorry

end distinct_distributions_of_balls_l741_741199


namespace two_cos_30_eq_sqrt_3_l741_741994

open Real

-- Given condition: cos 30 degrees is sqrt(3)/2
def cos_30_eq : cos (π / 6) = sqrt 3 / 2 := 
sorry

-- Goal: to prove that 2 * cos 30 degrees = sqrt(3)
theorem two_cos_30_eq_sqrt_3 : 2 * cos (π / 6) = sqrt 3 :=
by
  rw [cos_30_eq]
  sorry

end two_cos_30_eq_sqrt_3_l741_741994


namespace minimum_throws_l741_741031

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741031


namespace area_difference_of_circles_l741_741431

theorem area_difference_of_circles (circumference_large: ℝ) (half_radius_relation: ℝ → ℝ) (hl: circumference_large = 36) (hr: ∀ R, half_radius_relation R = R / 2) :
  ∃ R r, R = 18 / π ∧ r = 9 / π ∧ (π * R ^ 2 - π * r ^ 2) = 243 / π :=
by 
  sorry

end area_difference_of_circles_l741_741431


namespace minimum_rolls_for_duplicate_sum_l741_741052

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741052


namespace bank_tellers_have_total_coins_l741_741208

theorem bank_tellers_have_total_coins :
  (∃ (num_tellers num_rolls_per_teller coins_per_roll : ℕ)
     (total_coins_one_teller total_coins_all_tellers : ℕ),
    num_tellers = 4 ∧ num_rolls_per_teller = 10 ∧ coins_per_roll = 25 ∧
    total_coins_one_teller = num_rolls_per_teller * coins_per_roll ∧
    total_coins_all_tellers = num_tellers * total_coins_one_teller ∧
    total_coins_all_tellers = 1000) :=
begin
  use [4, 10, 25, 250, 1000],
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  split,
  sorry,
end

end bank_tellers_have_total_coins_l741_741208


namespace ratio_of_areas_l741_741172

variable {M N A B C D : Type}
variables [Circle M] [Circle N]
variable (A : Point) (B : Point) (C : Point) (D : Point)

-- Geometry given that there are intersecting circles at points A and B
variable (h1 : Intersection M N A B)
-- Line passing through point B intersecting the circles at points C and D
variable (h2 : LineThrough B C D)
-- Given angle
variable (h3 : angle A B D = 60)

theorem ratio_of_areas (h1 : Intersection M N A B) (h2 : LineThrough B C D) (h3 : angle A B D = 60) :
  area (quadrilateral A M B N) = (2 / 3 : ℝ) * area (triangle A C D) :=
sorry

end ratio_of_areas_l741_741172


namespace vasya_example_fewer_sevens_l741_741880

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l741_741880


namespace min_throws_to_ensure_same_sum_twice_l741_741075

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741075


namespace prove_f_0_eq_sqrt2_div2_l741_741264

-- Define the function f and its derivative
def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
def f' (ω φ x : ℝ) : ℝ := ω * Real.cos (ω * x + φ)

-- Given conditions
variables (ω φ : ℝ)
variable (h1 : ω > 0)
variable (h2 : f ω φ (-1) > 0)
variable (h3 : f' ω φ (-1) > 0)
variable h4 : ∀ x : ℝ, x = -3 ∨ x = 1 → Real.cos (ω * x + φ) = 0

-- Prove that f(0) = √2/2
theorem prove_f_0_eq_sqrt2_div2 : f ω φ 0 = Real.sqrt 2 / 2 :=
by
  sorry

end prove_f_0_eq_sqrt2_div2_l741_741264


namespace minimum_expression_l741_741257

variable (a b : ℝ)

theorem minimum_expression (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) :
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 3 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x + y = 3 → 
  x = a ∧ y = b  → ∃ m : ℝ, m ≥ 1 ∧ (m = (1/(a+1)) + 1/b))) := sorry

end minimum_expression_l741_741257


namespace distinct_paths_count_l741_741513

structure OctagonalLattice (α : Type) :=
(points : set α)
(start : α)
(finish : α)
(can_traverse : α → α → Prop)
(is_directed : α → α → Prop)
(no_revisit : α → α → Prop)

noncomputable def distinct_pathways_from_A_to_B (lattice : OctagonalLattice ℕ) : ℕ :=
  1728

axiom bug_path_conditions (lattice : OctagonalLattice ℕ) :
  ∃ (paths : set (list ℕ)), 
    ∀ path ∈ paths, 
      path.head = lattice.start ∧ 
      path.last = lattice.finish ∧ 
      (∀ (i j : ℕ), i ≠ j → path.nth i ≠ path.nth j) ∧
      (∀ (i j : ℕ), lattice.can_traverse (path.nth i) (path.nth j)) ∧
      (∀ (i j : ℕ), lattice.is_directed (path.nth i) (path.nth j))

theorem distinct_paths_count (lattice : OctagonalLattice ℕ) : 
  (bug_path_conditions lattice) → distinct_pathways_from_A_to_B lattice = 1728 := by
  sorry

end distinct_paths_count_l741_741513


namespace integer_values_abs_lt_2pi_l741_741687

theorem integer_values_abs_lt_2pi : 
  {x : ℤ | abs x < 2 * Real.pi}.finset.card = 13 := 
by
  sorry

end integer_values_abs_lt_2pi_l741_741687


namespace value_of_x_l741_741843

def f (x : ℝ) : ℝ := 9^x
def g (x : ℝ) : ℝ := log 3 (9 * x)

theorem value_of_x : ∃ x : ℝ, g (f x) = f (g 2) ∧ x = 161 := by 
  -- Definitions
  let f (x : ℝ) : ℝ := 9^x
  let g (x : ℝ) : ℝ := log 3 (9 * x)
  -- Calculate g(f(x))
  have g_f_x : ∀ x, g (f x) = 2 * x + 2 := by sorry
  -- Calculate f(g(2))
  have f_g_2 : f (g 2) = 324 := by sorry
  -- Establish the value of x
  use 161
  split
  -- Show that the equation holds
  · rw [g_f_x, f_g_2]
    sorry
  -- Show that x equals 161
  · refl

end value_of_x_l741_741843


namespace sum_of_reciprocal_sqrt_bound_l741_741804

theorem sum_of_reciprocal_sqrt_bound {n : ℕ} (h : n > 0) : 
  (∑ k in Finset.range n + 1, 1 / (k * Real.sqrt k)) < 3 :=
sorry

end sum_of_reciprocal_sqrt_bound_l741_741804


namespace intersection_A_B_l741_741676

-- Definitions based on conditions
variable (U : Set Int) (A B : Set Int)

#check Set

-- Given conditions
def U_def : Set Int := {-1, 3, 5, 7, 9}
def compl_U_A : Set Int := {-1, 9}
def B_def : Set Int := {3, 7, 9}

-- A is defined as the set difference of U and the complement of A in U
def A_def : Set Int := { x | x ∈ U_def ∧ ¬ (x ∈ compl_U_A) }

-- Theorem stating the intersection of A and B equals {3, 7}
theorem intersection_A_B : A_def ∩ B_def = {3, 7} :=
by
  -- Here would be the proof block, but we add 'sorry' to indicate it is unfinished.
  sorry

end intersection_A_B_l741_741676


namespace anna_integer_is_fourteen_l741_741549

/-- Conditions of the problem -/
def not_multiple_of_three (n : ℕ) : Prop :=
  ¬ (3 ∣ n)

def not_perfect_square (n : ℕ) : Prop :=
  ∀ k, k ^ 2 ≠ n

def sum_of_digits_is_prime (n : ℕ) : Prop :=
  (nat.digits 10 n).sum.isPrime

/-- Main theorem statement -/
theorem anna_integer_is_fourteen (n : ℕ) :
  not_multiple_of_three n ∧ not_perfect_square n ∧ sum_of_digits_is_prime n → n = 14 :=
by
  sorry  -- proof omitted

end anna_integer_is_fourteen_l741_741549


namespace BCM_hens_l741_741137

theorem BCM_hens (total_chickens : ℕ) 
  (percent_BCM : ℝ) 
  (percent_BCM_hens : ℝ) 
  (h1 : total_chickens = 100) 
  (h2 : percent_BCM = 0.20) 
  (h3 : percent_BCM_hens = 0.80)
  : (total_chickens * percent_BCM * percent_BCM_hens).to_nat = 16 := 
by
  sorry

end BCM_hens_l741_741137


namespace polygon_interior_angle_l741_741167

theorem polygon_interior_angle (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → ∀ j, 1 ≤ j ∧ j ≤ n → interior_angle n = 120) : n = 6 :=
sorry

end polygon_interior_angle_l741_741167


namespace extremal_point_a_range_of_a_l741_741386

noncomputable def f (x a : ℝ) : ℝ := (x - a) ^ 2 * Real.log x

theorem extremal_point_a (a : ℝ) (e : ℝ) (he : Real.exp 1 = e) :
  (f e a) = 0 ↔ a = e ∨ a = 3 * e :=
begin
  sorry
end

theorem range_of_a (a e : ℝ) (he : Real.exp 1 = e) :
  (∀ x > 0, x ≤ 3 * e → f x a ≤ 4 * e ^ 2) ↔
    3 * e - 2 * e / Real.sqrt (Real.log (3 * e)) ≤ a ∧ a ≤ 3 * e :=
begin
  sorry
end

end extremal_point_a_range_of_a_l741_741386


namespace percentage_increase_is_20_percent_l741_741417

noncomputable def originalSalary : ℝ := 575 / 1.15
noncomputable def increasedSalary : ℝ := 600
noncomputable def percentageIncreaseTo600 : ℝ := (increasedSalary - originalSalary) / originalSalary * 100

theorem percentage_increase_is_20_percent :
  percentageIncreaseTo600 = 20 := 
by
  sorry -- The proof will go here

end percentage_increase_is_20_percent_l741_741417


namespace atleast_one_alarm_rings_on_time_l741_741781

def probability_alarm_A_rings := 0.80
def probability_alarm_B_rings := 0.90

def probability_atleast_one_rings := 1 - (1 - probability_alarm_A_rings) * (1 - probability_alarm_B_rings)

theorem atleast_one_alarm_rings_on_time :
  probability_atleast_one_rings = 0.98 :=
sorry

end atleast_one_alarm_rings_on_time_l741_741781


namespace value_of_a2_b2_c2_l741_741767

noncomputable def nonzero_reals := { x : ℝ // x ≠ 0 }

theorem value_of_a2_b2_c2 (a b c : nonzero_reals) (h1 : (a : ℝ) + (b : ℝ) + (c : ℝ) = 0) 
  (h2 : (a : ℝ)^3 + (b : ℝ)^3 + (c : ℝ)^3 = (a : ℝ)^7 + (b : ℝ)^7 + (c : ℝ)^7) : 
  (a : ℝ)^2 + (b : ℝ)^2 + (c : ℝ)^2 = 6 / 7 :=
by
  sorry

end value_of_a2_b2_c2_l741_741767


namespace minimum_throws_to_ensure_same_sum_twice_l741_741013

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741013


namespace sum_ratio_arithmetic_seq_l741_741329

def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

def sum_arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ :=
  n * (a + (n - 1) * d / 2)

theorem sum_ratio_arithmetic_seq (a d : ℝ)
  (h : sum_arithmetic_seq a d 2 / sum_arithmetic_seq a d 4 = 1 / 3) :
  sum_arithmetic_seq a d 4 / sum_arithmetic_seq a d 8 = 3 / 10 :=
by
  sorry

end sum_ratio_arithmetic_seq_l741_741329


namespace quadratic_to_vertex_form_l741_741414

theorem quadratic_to_vertex_form :
  ∀ (x : ℝ), (1/2) * x^2 - 2 * x + 1 = (1/2) * (x - 2)^2 - 1 :=
by
  intro x
  -- full proof omitted
  sorry

end quadratic_to_vertex_form_l741_741414


namespace man_owns_fraction_of_business_l741_741525

theorem man_owns_fraction_of_business
  (x : ℚ)
  (H1 : (3 / 4) * (x * 90000) = 45000)
  (H2 : x * 90000 = y) : 
  x = 2 / 3 := 
by
  sorry

end man_owns_fraction_of_business_l741_741525


namespace distribution_schemes_l741_741830

theorem distribution_schemes 
    (total_professors : ℕ)
    (high_schools : Finset ℕ) 
    (A : ℕ) 
    (B : ℕ) 
    (C : ℕ)
    (D : ℕ)
    (cond1 : total_professors = 6) 
    (cond2 : A = 1)
    (cond3 : B ≥ 1)
    (cond4 : C ≥ 1)
    (D' := (total_professors - A - B - C)) 
    (cond5 : D' ≥ 1) : 
    ∃ N : ℕ, N = 900 := by
  sorry

end distribution_schemes_l741_741830


namespace pascals_triangle_number_of_entries_pascals_triangle_binomial_coefficient_l741_741180

theorem pascals_triangle_number_of_entries (n : ℕ) (h : n = 29) :
  (∑ k in Finset.range (n + 1), k + 1) = 465 := by
  sorry

theorem pascals_triangle_binomial_coefficient (n k : ℕ) (h1 : n = 30) (h2 : k = 5) :
  binomial n k = binomial 30 5 := by
  sorry

end pascals_triangle_number_of_entries_pascals_triangle_binomial_coefficient_l741_741180


namespace sin_cubed_decomposition_l741_741195

theorem sin_cubed_decomposition :
  ∃ c d : ℝ, (∀ θ : ℝ, sin θ ^ 3 = c * sin (3 * θ) + d * sin θ) ∧
           c = -1/4 ∧ d = 3/4 := 
begin
  use [-1/4, 3/4],
  split,
  { intro θ,
    calc 
      sin θ ^ 3
          = sin θ ^ 3 : by refl
      ... = 3/4 * sin θ - 1/4 * sin (3 * θ) : sorry },
  split; refl,
end

end sin_cubed_decomposition_l741_741195


namespace range_of_a_l741_741275

noncomputable def curve_points (a : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (a + 2*cos θ, a + 2*sin θ)

def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  p.1^2 + p.2^2

def two_points_distance_two (a : ℝ) : Prop :=
  ∃ θ₁ θ₂,
  θ₁ ≠ θ₂ ∧
  distance_from_origin (curve_points a θ₁) = 4 ∧
  distance_from_origin (curve_points a θ₂) = 4

theorem range_of_a :
  {a : ℝ | two_points_distance_two a} = {a | -2*sqrt 2 < a ∧ a < 0 ∨ 0 < a ∧ a < 2*sqrt 2 } :=
sorry

end range_of_a_l741_741275


namespace find_AB_l741_741408

variable (α : Type)
variable [OrderedRing α]

noncomputable def AB {a b : α} (h₁ : (4:α) = ⟦P⟧)
                     (h₂ : (3:α) = ⟦R⟧)
                     (h₃ : (5:α) = ⟦S⟧)
                     (h₄ : is_right_triangle a b)
                     (h₅ : point_in_right_triangle ⟦P⟧ a b) : α :=
12.25

theorem find_AB (a b : ℕ) (P R S : ℕ)
                (h₁ : 4 = P)
                (h₂ : 3 = R)
                (h₃ : 5 = S)
                (h₄ : a = 3 * b / 4)
                (h₅ : b = 4 * a / 3) :
  AB h₁ h₂ h₃ _ _ = 12.25 := 
by 
  sorry

end find_AB_l741_741408


namespace measure_AB_l741_741730

-- Problem definitions
variable {a b : ℕ}
variable {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

def is_parallel (x y : Set Point) : Prop := 
  -- Assume we have a definition of parallel lines
  sorry

def measure_angle (x y z : Point) : ℝ := 
  -- Assume we have a definition to measure angles
  sorry

noncomputable def measure_segment (x y : Point) : ℝ := 
  -- Assume we have a definition to measure segments
  sorry

-- Given
variables {AB CD AD : Set Point}
variables {B C D : Point}
axiom h1 : is_parallel AB CD
axiom h2 : measure_angle A B C = measure_angle C D D * 3
axiom h3 : measure_segment A D = 2 * a
axiom h4 : measure_segment C D = 3 * b

-- Goal
theorem measure_AB : measure_segment A B = 2 * a + 3 * b :=
by
  sorry

end measure_AB_l741_741730


namespace minimum_throws_to_ensure_same_sum_twice_l741_741011

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741011


namespace Charles_learning_vowels_l741_741609

theorem Charles_learning_vowels :
  ∀ (days_per_alphabet : ℕ) (num_vowels : ℕ), 
  days_per_alphabet = 7 → num_vowels = 5 → 
  (days_per_alphabet * num_vowels) = 35 :=
by
  intros days_per_alphabet num_vowels H_days H_vowels
  rw [H_days, H_vowels]
  exact Nat.mul_comm 7 5 ▸ Nat.mul_self_add 5 30

end Charles_learning_vowels_l741_741609


namespace add_and_subtract_l741_741080

theorem add_and_subtract (a b c : ℝ) (h1 : a = 0.45) (h2 : b = 52.7) (h3 : c = 0.25) : 
  (a + b) - c = 52.9 :=
by 
  sorry

end add_and_subtract_l741_741080


namespace intersection_M_N_l741_741776

-- Given set M defined by the inequality
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Given set N defined by the interval
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The intersection M ∩ N should be equal to the interval [1, 2)
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l741_741776


namespace range_of_m_l741_741258

-- Define the propositions
def decreasing_function_on_interval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x > f y

def solution_set_is_real (m : ℝ) : Prop :=
  ∀ x : ℝ, (x - 1)^2 > m

-- Define the main problem
theorem range_of_m (f : ℝ → ℝ) (p : decreasing_function_on_interval f (Set.Ioi 0)) 
  (q : solution_set_is_real m) (hpq : (p ∨ q) ∧ ¬(p ∧ q)) : 0 ≤ m ∧ m < 0.5 := 
sorry

end range_of_m_l741_741258


namespace total_distance_walked_l741_741789

def distance_to_fountain : ℕ := 30
def number_of_trips : ℕ := 4
def round_trip_distance : ℕ := 2 * distance_to_fountain

theorem total_distance_walked : (number_of_trips * round_trip_distance) = 240 := by
  sorry

end total_distance_walked_l741_741789


namespace fewerSevensCanProduce100_l741_741916

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l741_741916


namespace slices_left_for_lunch_tomorrow_l741_741201

def pizza_slices : ℕ := 12
def lunch_slices : ℕ := pizza_slices / 2
def remaining_after_lunch : ℕ := pizza_slices - lunch_slices
def dinner_slices : ℕ := remaining_after_lunch * 1/3
def slices_left : ℕ := remaining_after_lunch - dinner_slices

theorem slices_left_for_lunch_tomorrow : slices_left = 4 :=
by
  sorry

end slices_left_for_lunch_tomorrow_l741_741201


namespace cherry_orange_punch_ratio_l741_741423

theorem cherry_orange_punch_ratio 
  (C : ℝ)
  (h_condition1 : 4.5 + C + (C - 1.5) = 21) : 
  C / 4.5 = 2 :=
by
  sorry

end cherry_orange_punch_ratio_l741_741423


namespace solve_fx_eq_one_over_four_lt_one_solve_fx_eq_one_over_four_ge_one_solve_fx_le_two_l741_741772

def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else log x / log 4

theorem solve_fx_eq_one_over_four_lt_one :
  ∃ x < 1, f x = 1/4 ↔ x = 2 := 
by
  sorry

theorem solve_fx_eq_one_over_four_ge_one :
  ∃ x ≥ 1, f x = 1/4 ↔ x = real.sqrt 2 := 
by
  sorry

theorem solve_fx_le_two : 
  ∀ x, (f x ≤ 2) ↔ x ∈ Icc (-1 : ℝ) 16 :=
by
  sorry

end solve_fx_eq_one_over_four_lt_one_solve_fx_eq_one_over_four_ge_one_solve_fx_le_two_l741_741772


namespace retailer_loss_l741_741159

def cost_price : ℝ := 225
def overhead_expenses : ℝ := 28
def purchase_tax_rate : ℝ := 0.08
def final_selling_price : ℝ := 300
def sales_tax_rate : ℝ := 0.12

def purchase_tax : ℝ := purchase_tax_rate * cost_price
def total_cost_price : ℝ := cost_price + overhead_expenses + purchase_tax

def selling_price_before_tax : ℝ := final_selling_price / (1 + sales_tax_rate)

def profit : ℝ := selling_price_before_tax - total_cost_price

theorem retailer_loss : profit = -3.14 :=
by
  sorry

end retailer_loss_l741_741159


namespace problem_solution_l741_741121

def matrix_orig : Matrix (Fin 5) (Fin 5) ℕ :=
  ![![1, 2, 3, 4, 5],
    ![11, 12, 13, 14, 15],
    ![21, 22, 23, 24, 25],
    ![31, 32, 33, 34, 35],
    ![41, 42, 43, 44, 45]]

def reverse_row {α : Type} (r : Fin 5 → α) : Fin 5 → α :=
  λ i, r ⟨4 - i.1, by linarith [i.2]⟩

def matrix_mod : Matrix (Fin 5) (Fin 5) ℕ :=
  ![![1, 2, 3, 4, 5],
    reverse_row ![11, 12, 13, 14, 15],
    ![21, 22, 23, 24, 25],
    ![31, 32, 33, 34, 35],
    reverse_row ![41, 42, 43, 44, 45]]

def main_diagonal_sum (m : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  ∑ i, m i i

def anti_diagonal_sum (m : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  ∑ i, m i (⟨4 - i.1, by linarith [i.2]⟩)

theorem problem_solution : abs (main_diagonal_sum matrix_mod - anti_diagonal_sum matrix_mod) = 4 := by
  sorry

end problem_solution_l741_741121


namespace intersection_A_B_l741_741631

open Set

variable (l : ℝ)

def A := {x : ℝ | x > l}
def B := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_A_B (h₁ : l = 1) :
  A l ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l741_741631


namespace fixed_salary_new_scheme_l741_741161

theorem fixed_salary_new_scheme :
  ∀ (F : ℝ), 
    (0.05 * 12000 = 600) →
    (12000 > 4000) →
    (F + 0.025 * (12000 - 4000) = 600 + 600) →
    (F = 400) :=
begin
  intros F commission_scheme sales_value remuneration_scheme,
  sorry
end

end fixed_salary_new_scheme_l741_741161


namespace sin_exponential_solution_count_l741_741590

noncomputable def count_solutions : ℝ :=
  let f := λ x: ℝ, Real.sin x
  let g := λ x: ℝ, (1 / 3)^x
  (0, 50 * Real.pi).bInter (λ x, f x = g x)

theorem sin_exponential_solution_count :
  count_solutions = 50 :=
sorry

end sin_exponential_solution_count_l741_741590


namespace min_rolls_to_duplicate_sum_for_four_dice_l741_741001

theorem min_rolls_to_duplicate_sum_for_four_dice : 
    let min_sum := 4 * 1,
    let max_sum := 4 * 6,
    let possible_sums := max_sum - min_sum + 1 in
    possible_sums = 21 → 
    (possible_sums + 1 = 22) := 
by
  intros min_sum max_sum possible_sums h
  have h1 : min_sum = 4 := rfl
  have h2 : max_sum = 24 := rfl
  have h3 : possible_sums = 21 := h
  have h4 : possible_sums + 1 = 22 := calc
    possible_sums + 1 = 21 + 1 : by rw h
    ... = 22 : by rfl
  exact h4

end min_rolls_to_duplicate_sum_for_four_dice_l741_741001


namespace count_valid_subsets_l741_741252

def is_even (n : ℕ) : Prop := n % 2 = 0

def valid_set (s : Set ℕ) : Prop := s ⊆ {4, 7, 8} ∧ (∀ n ∈ s, is_even n) ↔ s.card ≤ 1

theorem count_valid_subsets : 
  -- Define all the sets that satisfy the conditions
  { M : Set ℕ // M ⊆ {4, 7, 8} ∧ (∀ s : Set ℕ, s ⊆ M ∧ s.card ≤ 1) }.card = 6 := 
sorry

end count_valid_subsets_l741_741252


namespace max_tetrahedron_volume_l741_741335

theorem max_tetrahedron_volume 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (right_triangle : ∃ A B C : Type, 
    ∃ (angle_C : ℝ) (h_angle_C : angle_C = π / 2), 
    ∃ (BC CA : ℝ), BC = a ∧ CA = b) : 
  ∃ V : ℝ, V = (a^2 * b^2) / (6 * (a^(2/3) + b^(2/3))^(3/2)) := 
sorry

end max_tetrahedron_volume_l741_741335


namespace cone_base_circumference_l741_741516

   -- Define the radius of the circle and the angle of the sector.
   def radius : ℝ := 6
   def sector_angle : ℝ := 120
   def full_circle_angle : ℝ := 360

   -- Define the circumference of a circle with given radius.
   def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

   -- Define the proportion of the sector.
   def sector_proportion (sector_angle full_circle_angle : ℝ) : ℝ :=
     sector_angle / full_circle_angle

   -- Prove the circumference of the base of the resulting cone.
   theorem cone_base_circumference (r : ℝ) (sector_angle full_circle_angle : ℝ)
     (sector_proportion : ℝ) :
     sector_proportion * circle_circumference r = 4 * Real.pi :=
   by
     have h : sector_proportion = 1/3 := by sorry
     have h1 : circle_circumference r = 12 * Real.pi := by sorry
     sorry
   
end cone_base_circumference_l741_741516


namespace coloring_process_result_l741_741794

def initial_colored_cells : Nat := 3
def repetitions : Nat := 100
def final_colored_cells : Nat := 20503

theorem coloring_process_result :
  (starting_cells : Nat) (steps : Nat) (final_cells : Nat) (_initial_colored_cells : starting_cells = initial_colored_cells)
  (_steps : steps = repetitions)
  (_final_cells : final_cells = final_colored_cells) :
  (stepsave := starting_cells + steps + steps^2) → (stepsave = final_cells) := sorry

end coloring_process_result_l741_741794


namespace minimum_throws_for_repeated_sum_l741_741044

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741044


namespace sqrt_square_simplify_l741_741823

noncomputable def pi : ℝ := Real.pi

theorem sqrt_square_simplify : sqrt ((3 - pi) ^ 2) = pi - 3 :=
by
  sorry

end sqrt_square_simplify_l741_741823


namespace integer_solution_divisibility_by_primes_l741_741820

theorem integer_solution (k : ℕ) (hk : k > 0) : 
  ∃ n : ℕ, n = 224 * 10 ^ (k + k - 2) + 10 ^ (k + k - 2) - 81 := 
sorry

theorem divisibility_by_primes (k : ℕ) (hk : k > 0) 
  (number := 224 * 10 ^ (k + k - 2) + 10 ^ (k + k - 2) - 81) :
  ∀ p : ℕ, prime p → p ∣ number →
  p = 2 ∨ p = 3 ∨ p = 5 :=
sorry

end integer_solution_divisibility_by_primes_l741_741820


namespace find_lambda_l741_741679

theorem find_lambda (λ : ℝ) (m n : ℝ × ℝ)
    (hm : m = (λ + 1, 1))
    (hn : n = (λ + 2, 2))
    (h_perpendicular : ((λ + 1 + (λ + 2), 1 + 2) • (λ + 1 - (λ + 2), 1 - 2)) = 0) :
    λ = -3 :=
begin
  sorry
end

end find_lambda_l741_741679


namespace cross_covers_two_rectangles_l741_741545

def Chessboard := Fin 8 × Fin 8

def is_cross (center : Chessboard) (point : Chessboard) : Prop :=
  (point.1 = center.1 ∧ (point.2 = center.2 - 1 ∨ point.2 = center.2 + 1)) ∨
  (point.2 = center.2 ∧ (point.1 = center.1 - 1 ∨ point.1 = center.1 + 1)) ∨
  (point = center)

def Rectangle_1x3 (rect : Fin 22) : Chessboard → Prop := sorry -- This represents Alina's rectangles
def Rectangle_1x2 (rect : Fin 22) : Chessboard → Prop := sorry -- This represents Polina's rectangles

theorem cross_covers_two_rectangles :
  ∃ center : Chessboard, 
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_b p)) ∨
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_b p)) ∨
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_b p)) :=
sorry

end cross_covers_two_rectangles_l741_741545


namespace solution_to_problem_l741_741155

noncomputable def problem_statement : Prop :=
  ∃ (w : ℝ) (t : ℝ), 
    (3 * w^2 = 8 * w) ∧ -- Condition related to the rectangle's area and perimeter
    (w > 0) ∧ -- Non-trivial width
    (3 * w > 0) ∧ -- Non-trivial length
    (t > 0) ∧ -- Non-trivial side length of the triangle
    ( (sqrt 3 / 4 * t^2 = 3 / 2 * t) ∧ -- Condition related to the triangle's area and half the perimeter
      (sqrt (w^2 + (3 * w)^2) = 3 / 2 * (sqrt 3 / 2 * t) ) -- Compare diagonal of rectangle with height of triangle
    )

theorem solution_to_problem : problem_statement := 
by sorry

end solution_to_problem_l741_741155


namespace sum_of_m_with_distinct_integer_solutions_eq_0_l741_741082

theorem sum_of_m_with_distinct_integer_solutions_eq_0 :
  (∑ m in {m | ∃ r s : ℤ, r ≠ s ∧ (3 * r^2 - m * r + 12 = 0) ∧ (3 * s^2 - m * s + 12 = 0)}, m) = 0 :=
sorry

end sum_of_m_with_distinct_integer_solutions_eq_0_l741_741082


namespace xyz_inequality_l741_741622

theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) : 
    x^2 + y^2 + z^2 + (real.sqrt 3 / 2) * real.sqrt (x * y * z) ≥ 1 / 2 :=
by
    sorry

end xyz_inequality_l741_741622


namespace yan_distance_ratio_l741_741932

-- Define conditions
variable (x z w: ℝ)  -- x: distance from Yan to his home, z: distance from Yan to the school, w: Yan's walking speed
variable (h1: z / w = x / w + (x + z) / (5 * w))  -- Both choices require the same amount of time

-- The ratio of Yan's distance from his home to his distance from the school is 2/3
theorem yan_distance_ratio :
    x / z = 2 / 3 :=
by
  sorry

end yan_distance_ratio_l741_741932


namespace tangerine_initial_count_l741_741211

theorem tangerine_initial_count 
  (X : ℕ) 
  (h1 : X - 9 + 5 = 20) : 
  X = 24 :=
sorry

end tangerine_initial_count_l741_741211


namespace estimate_num_2016_digit_squares_l741_741210

noncomputable def num_estimate_2016_digit_squares : ℕ := 2016

theorem estimate_num_2016_digit_squares :
  let t1 := (10 ^ (2016 / 2) - 10 ^ (2015 / 2) - 1)
  let t2 := (2017 ^ 10)
  let result := t1 / t2
  t1 > 10 ^ 1000 → 
  result > 10 ^ 900 →
  result == num_estimate_2016_digit_squares :=
by
  intros
  sorry

end estimate_num_2016_digit_squares_l741_741210


namespace interval_of_monotonic_increase_of_f_l741_741695

noncomputable def f (x : ℝ) := x^2 - 2 * x - 4 * Real.log x

theorem interval_of_monotonic_increase_of_f :
  ∃ I : Set ℝ, I = (Set.Ioo 2 ⊤) ∧ ∀ x ∈ I, ∀ y ∈ I, x < y → f x < f y := by
sorry

end interval_of_monotonic_increase_of_f_l741_741695


namespace solve_quadratic1_solve_quadratic2_l741_741421

-- For the first quadratic equation: 3x^2 = 6x
theorem solve_quadratic1 (x : ℝ) (h : 3 * x^2 = 6 * x) : x = 0 ∨ x = 2 :=
sorry

-- For the second quadratic equation: x^2 - 6x + 5 = 0
theorem solve_quadratic2 (x : ℝ) (h : x^2 - 6 * x + 5 = 0) : x = 5 ∨ x = 1 :=
sorry

end solve_quadratic1_solve_quadratic2_l741_741421


namespace minimum_throws_l741_741033

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741033


namespace total_value_is_correct_l741_741963

-- We will define functions that convert base 7 numbers to base 10
def base7_to_base10 (n : Nat) : Nat :=
  let digits := (n.digits 7)
  digits.enum.foldr (λ ⟨i, d⟩ acc => acc + d * 7^i) 0

-- Define the specific numbers in base 7
def silver_value_base7 : Nat := 5326
def gemstone_value_base7 : Nat := 3461
def spice_value_base7 : Nat := 656

-- Define the combined total in base 10
def total_value_base10 : Nat := base7_to_base10 silver_value_base7 + base7_to_base10 gemstone_value_base7 + base7_to_base10 spice_value_base7

theorem total_value_is_correct :
  total_value_base10 = 3485 :=
by
  sorry

end total_value_is_correct_l741_741963


namespace lim_div_f_x_over_x_eq_zero_l741_741501

open Real Filter

noncomputable def continuously_differentiable_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ContinuousAt f a ∧ ∃ f' : ℝ → ℝ, (∀ x, derivWithin f (univ \ {a}) x = f' x) ∧ ContinuousAt f' a

theorem lim_div_f_x_over_x_eq_zero
  (f : ℝ → ℝ)
  (h_cd : ∀ x > 0, continuously_differentiable_at f x)
  (h_lim : Tendsto (fun x => deriv f x) atTop (𝓝 0)) :
  Tendsto (fun x => f x / x) atTop (𝓝 0) :=
by
  sorry

end lim_div_f_x_over_x_eq_zero_l741_741501


namespace lambda_value_range_l741_741527

theorem lambda_value_range (m n λ : ℝ) (h₁ : λ * Real.sqrt (m^2 + n^2) = |m + 5|) (h₂ : ∀ P, ∃ c > 1, ∀ (m n : ℝ), λ * Real.sqrt (m^2 + n^2) = c * |m + 5|):
  0 < λ ∧ λ < 1 :=
sorry

end lambda_value_range_l741_741527


namespace correct_options_l741_741648

theorem correct_options (a b c : ℝ) (h1 : ∀ x : ℝ, (a*x^2 + b*x + c > 0) ↔ (-3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, (b*x + c > 0) ↔ x > 6) = False ∧ (∀ x, (c*x^2 + b*x + a < 0) ↔ (-1/3 < x ∧ x < 1/2)) :=
by 
  sorry

end correct_options_l741_741648


namespace five_digit_units_digit_probability_even_l741_741140

theorem five_digit_units_digit_probability_even : 
  (∃ n : ℕ, 10000 ≤ n ∧ n < 100000) →
  (probability (d : ℕ, 0 ≤ d ∧ d < 10 ∧ ∃ k, n = 10 * k + d ∧ d % 2 = 0) = 1 / 2) := by
  sorry

end five_digit_units_digit_probability_even_l741_741140


namespace john_total_strings_l741_741367

theorem john_total_strings :
  let basses := 3
  let strings_per_bass := 4
  let guitars := 2 * basses
  let strings_per_guitar := 6
  let eight_string_guitars := guitars - 3
  let strings_per_eight_string_guitar := 8
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := 
by
  let basses := 3
  let strings_per_bass := 4
  let guitars := 2 * basses
  let strings_per_guitar := 6
  let eight_string_guitars := guitars - 3
  let strings_per_eight_string_guitar := 8
  have H_bass := basses * strings_per_bass
  have H_guitar := guitars * strings_per_guitar
  have H_eight_string_guitar := eight_string_guitars * strings_per_eight_string_guitar
  have H_total := H_bass + H_guitar + H_eight_string_guitar
  show H_total = 72 from sorry

end john_total_strings_l741_741367


namespace probability_even_sum_between_3_and_15_l741_741829

open Finset

theorem probability_even_sum_between_3_and_15 :
  let S := Icc 3 15
  let pairs := S.product S \ (diagonal S)
  let even_pairs : Finset (ℕ × ℕ) := pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  let probability : ℚ := even_pairs.card / pairs.card 
  probability = 6 / 13 := 
sorry

end probability_even_sum_between_3_and_15_l741_741829


namespace posters_sum_l741_741398

variables (Mario_rate : ℕ) (Mario_hours : ℕ)
           (Samantha_rate_multiplier : ℕ)
           (Samantha_hours : ℕ)
           (Jonathan_rate_multiplier : ℕ)
           (Jonathan_hours : ℕ)

noncomputable def total_posters_by_Mario : ℕ :=
  Mario_rate * Mario_hours

noncomputable def Samantha_rate : ℕ :=
  Samantha_rate_multiplier * Mario_rate

noncomputable def total_posters_by_Samantha : ℕ :=
  Samantha_rate * Samantha_hours / 2 -- Assuming rate multiplier of 1.5 implies half in Lean

noncomputable def Jonathan_rate : ℕ :=
  Jonathan_rate_multiplier * Samantha_rate

noncomputable def total_posters_by_Jonathan : ℕ :=
  Jonathan_rate * Jonathan_hours

noncomputable def total_posters : ℕ :=
  total_posters_by_Mario Mario_rate Mario_hours +
  total_posters_by_Samantha Mario_rate Samantha_rate_multiplier Samantha_hours +
  total_posters_by_Jonathan Mario_rate Samantha_rate_multiplier Jonathan_rate_multiplier Jonathan_hours

theorem posters_sum (Mario_rate : ℕ := 5) (Mario_hours : ℕ := 7)
                    (Samantha_rate_multiplier : ℕ := 3) 
                    (Samantha_hours : ℕ := 9)
                    (Jonathan_rate_multiplier : ℕ := 2) 
                    (Jonathan_hours : ℕ := 6) :
  total_posters Mario_rate Mario_hours 
                Samantha_rate_multiplier Samantha_hours 
                Jonathan_rate_multiplier Jonathan_hours = 192 :=
by sorry

end posters_sum_l741_741398


namespace decimal_to_base2_l741_741190

theorem decimal_to_base2 : ∃ (b : ℕ), (b = 93) → (nat.digits 2 b = [1, 0, 1, 1, 1, 0, 1]) :=
begin
  use 93,
  intro h,
  rw h,
  exact dec_trivial,
end

end decimal_to_base2_l741_741190


namespace calculate_first_interest_rate_l741_741415

theorem calculate_first_interest_rate
  (S : ℕ := 80000)       -- Total amount in Rs
  (P1 : ℕ := 70000)      -- Amount with the first interest rate in Rs
  (P2 : ℕ := 10000)      -- Amount with the 20% interest rate in Rs
  (profit : ℕ := 9000)   -- Total profit at the end of the first year in Rs
  (r : ℕ)                -- First interest rate in percentage
  (h : P1 * r / 100 + P2 * 20 / 100 = profit) : r = 10 := by
  have h1 : P2 * 20 / 100 = 2000 := by
    norm_num
  have h2 : P1 * r / 100 + 2000 = profit := by
    rw [h1]
    exact h
  have h3 : P1 * r / 100 = profit - 2000 := by
    linarith
  have h4 : P1 * r = (profit - 2000) * 100 := by
    rw [mul_comm, nat.div_eq_iff_eq_mul_left (nat.zero_lt_of_add_pos_right (nat.zero_lt_mul (nat.succ_pos _) _))]
    exact h3
  have h5 : r = (profit - 2000) * 100 / P1 := by
    rw [nat.mul_div_cancel (profit - 2000) (nat.pos_of_ne_zero (nat.pos_p1 70000).ne')]
    exact h4
  norm_num at h5
  exact h5
-- sorry

end calculate_first_interest_rate_l741_741415


namespace general_term_sum_first_n_l741_741249

-- Definitions based on the given conditions of the problem.
def a_n (a₁ d n : ℕ) := a₁ + (n - 1) * d
def b_n (a₁ d n : ℕ) := a_n a₁ d n + 2^n

-- Conditions given in the problem.
variables (a₁ d : ℕ)
axiom h1 : a_n a₁ d 3 + a_n a₁ d 8 = 37
axiom h2 : a_n a₁ d 7 = 23

-- Prove the general term of the sequence aₙ
theorem general_term : ∀ n : ℕ, a_n 5 3 n = 3 * n + 2 :=
by
  intros
  sorry

-- Prove the sum of the first n terms of sequence bₙ
def sum_b_n (n : ℕ) :=
  ∑ k in Finset.range n, b_n 5 3 k + ∑ k in Finset.range n, 2^k

theorem sum_first_n : ∀ n : ℕ, sum_b_n n = (n * (7 + 3 * n)) / 2 + 2^(n + 1) - 2 :=
by
  intros
  sorry

end general_term_sum_first_n_l741_741249


namespace total_area_of_tickets_is_3_6_m2_l741_741507

def area_of_one_ticket (side_length_cm : ℕ) : ℕ :=
  side_length_cm * side_length_cm

def total_tickets (people : ℕ) (tickets_per_person : ℕ) : ℕ :=
  people * tickets_per_person

def total_area_cm2 (area_per_ticket_cm2 : ℕ) (number_of_tickets : ℕ) : ℕ :=
  area_per_ticket_cm2 * number_of_tickets

def convert_cm2_to_m2 (area_cm2 : ℕ) : ℚ :=
  (area_cm2 : ℚ) / 10000

theorem total_area_of_tickets_is_3_6_m2 :
  let side_length := 30
  let people := 5
  let tickets_per_person := 8
  let one_ticket_area := area_of_one_ticket side_length
  let number_of_tickets := total_tickets people tickets_per_person
  let total_area_cm2 := total_area_cm2 one_ticket_area number_of_tickets
  let total_area_m2 := convert_cm2_to_m2 total_area_cm2
  total_area_m2 = 3.6 := 
by
  sorry

end total_area_of_tickets_is_3_6_m2_l741_741507


namespace apple_picking_l741_741357

/-- 
Conditions:
1. Each apple pie requires 4 apples.
2. They can make 7 apple pies.
3. 6 of the apples that they picked are not ripe.

Question:
How many apples did they pick in total?
-/
theorem apple_picking (apples_per_pie : ℕ) (number_of_pies : ℕ) (not_ripe_apples : ℕ) (total_picked : ℕ) :
  apples_per_pie = 4 ∧ number_of_pies = 7 ∧ not_ripe_apples = 6 ∧ total_picked = number_of_pies * apples_per_pie + not_ripe_apples → total_picked = 34 := 
begin
  sorry
end

end apple_picking_l741_741357


namespace minimum_rolls_for_duplicate_sum_l741_741059

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741059


namespace matrix_exponentiation_l741_741692

theorem matrix_exponentiation (b m : ℤ)
  (h : (λ (x y z : ℤ), Matrix 3 3 ℤ
    | 0, 0 => 1 | 0, 1 => 3 | 0, 2 => b
    | 1, 0 => 0 | 1, 1 => 1 | 1, 2 => 5
    | 2, 0 => 0 | 2, 1 => 0 | 2, 2 => 1
    end) =
  (Matrix.pow (λ (x y : ℤ), Matrix 3 3 ℤ
    | 0, 0 => 1 | 0, 1 => 3 | 0, 2 => b
    | 1, 0 => 0 | 1, 1 => 1 | 1, 2 => 5
    | 2, 0 => 0 | 2, 1 => 0 | 2, 2 => 1
    end) m) :=
  λ (x y : ℤ),
    Matrix 3 3 ℤ
    | 0, 0 => 1 | 0, 1 => 33 | 0, 2 => 4014
    | 1, 0 => 0 | 1, 1 => 1 | 1, 2 => 65
    | 2, 0 => 0 | 2, 1 => 0 | 2, 2 => 1
    end) :
  b + m = 277 :=
by
  sorry

end matrix_exponentiation_l741_741692


namespace beta_value_lambda_range_l741_741293

-- Given the vectors and conditions
def vector_OA (λ α : ℝ) : ℝ × ℝ := (λ * Real.cos α, λ * Real.sin α)
def vector_OB (β : ℝ) : ℝ × ℝ := (-Real.sin β, Real.cos β)
def vector_OC : ℝ × ℝ := (1, 0)

-- Question 1
theorem beta_value (λ : ℝ) (α β : ℝ) (h1 : λ = 2) (h2 : α = Real.pi / 3) (h3 : 0 < β ∧ β < Real.pi)
  (h4 : let OA := vector_OA λ α in let BC := (1 + (vector_OB β).1, -(vector_OB β).2) in
        OA.fst * BC.fst + OA.snd * BC.snd = 0) : β = Real.pi / 6 := sorry

-- Question 2
theorem lambda_range (λ α β : ℝ)
  (h : (vector_OA λ α).fst - (vector_OB β).fst + (vector_OA λ α).snd - (vector_OB β).snd ≥ 4) :
  λ ≤ -3 ∨ λ ≥ 3 := sorry

end beta_value_lambda_range_l741_741293


namespace candy_cost_l741_741175

theorem candy_cost
    (grape_candies : ℕ)
    (cherry_candies : ℕ)
    (apple_candies : ℕ)
    (total_cost : ℝ)
    (total_candies : ℕ)
    (cost_per_candy : ℝ)
    (h1 : grape_candies = 24)
    (h2 : grape_candies = 3 * cherry_candies)
    (h3 : apple_candies = 2 * grape_candies)
    (h4 : total_cost = 200)
    (h5 : total_candies = cherry_candies + grape_candies + apple_candies)
    (h6 : cost_per_candy = total_cost / total_candies) :
    cost_per_candy = 2.50 :=
by
    sorry

end candy_cost_l741_741175


namespace landscape_breadth_l741_741834

theorem landscape_breadth (L B : ℕ) (h1 : B = 8 * L)
  (h2 : 3200 = 1 / 9 * (L * B))
  (h3 : B * B = 28800) :
  B = 480 := by
  sorry

end landscape_breadth_l741_741834


namespace relationship_among_abc_l741_741618

noncomputable def a : ℝ := 2^0.5
noncomputable def b : ℝ := Real.logBase π 3
noncomputable def c : ℝ := Real.logBase 2 (Real.sin (2 * π / 5))

theorem relationship_among_abc : c < b ∧ b < a :=
by
  sorry

end relationship_among_abc_l741_741618


namespace triangle_area_l741_741239

theorem triangle_area (q : ℝ) (h : 0 < q ∧ q ≤ 10) : 
  let X := (0, 10 : ℝ);
      Y := (3, 10 : ℝ);
      Z := (0, q) in
  let base := (Y.1 - X.1).abs;
      height := (X.2 - Z.2).abs;
      area := (1 / 2) * base * height in
  area = (3 / 2) * (10 - q) :=
by
  sorry

end triangle_area_l741_741239


namespace waiting_probability_no_more_than_10_seconds_l741_741551

def total_cycle_time : ℕ := 30 + 10 + 40
def proceed_during_time : ℕ := 40 -- green time
def yellow_time : ℕ := 10

theorem waiting_probability_no_more_than_10_seconds :
  (proceed_during_time + yellow_time + yellow_time) / total_cycle_time = 3 / 4 := by
  sorry

end waiting_probability_no_more_than_10_seconds_l741_741551


namespace total_area_of_room_l741_741160

theorem total_area_of_room : 
  let length_rect := 8 
  let width_rect := 6 
  let base_triangle := 6 
  let height_triangle := 3 
  let area_rect := length_rect * width_rect 
  let area_triangle := (1 / 2 : ℝ) * base_triangle * height_triangle 
  let total_area := area_rect + area_triangle 
  total_area = 57 := 
by 
  sorry

end total_area_of_room_l741_741160


namespace max_y_midpoint_l741_741344

open Real

-- Definitions based on conditions in the problem
noncomputable def f (x : ℝ) : ℝ := exp x
noncomputable def tangent_line (m : ℝ) : ℝ → ℝ := λ x, exp m * (x - m) + exp m
noncomputable def y_coord_M (m : ℝ) : ℝ := (1 - m) * exp m
noncomputable def perp_line (m : ℝ) : ℝ → ℝ := λ x, -exp (-m) * (x - m) + exp m
noncomputable def y_coord_N (m : ℝ) : ℝ := exp m + m * exp (-m)
noncomputable def y_midpoint (m : ℝ) : ℝ := 0.5 * ((2 - m) * exp m + m * exp (-m))

-- The theorem statement
theorem max_y_midpoint : ∃ m : ℝ, y_midpoint m = 0.5 * (exp 1 + exp (-1)) :=
by
  use 1
  sorry

end max_y_midpoint_l741_741344


namespace correct_equations_choice_A_l741_741438

-- Definitions for each of the equations
def eq1 : Prop := sqrt 2 + sqrt 5 = sqrt 7
def eq2 : Prop := 5 * sqrt a - 3 * sqrt a = 2 * sqrt a
def eq3 : Prop := (sqrt 8 + sqrt 50) / 2 = sqrt 4 + sqrt 25 ∧ sqrt 4 + sqrt 25 = 7
def eq4 : Prop := 2 * sqrt (3 * a) + sqrt (27 * a) = 5 * sqrt (3 * a)

-- The main statement that checks which of the equations are correct
theorem correct_equations (h : eq2 ∧ eq4 ∧ ¬ eq1 ∧ ¬ eq3) : true :=
by trivial

-- Confirm choice A: (2) and (4)
theorem choice_A : correct_equations :=
begin
  sorry
end

end correct_equations_choice_A_l741_741438


namespace example_one_example_two_l741_741884

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l741_741884


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741892

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741892


namespace inequality_solution_set_l741_741652

theorem inequality_solution_set (a b c : ℝ)
  (h1 : ∀ x, (ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, ¬ (bx + c > 0 ↔ x > 6)) ∧ (∀ x, (cx^2 + bx + a < 0 ↔ -1/3 < x ∧ x < 1/2)) :=
by
  sorry

end inequality_solution_set_l741_741652


namespace square_dist_intersection_points_l741_741871

theorem square_dist_intersection_points :
  let center1 := (3 : ℝ, -2 : ℝ)
  let center2 := (3 : ℝ, 6 : ℝ)
  let radius1 := 5
  let radius2 := 3
  let dist_square_intersection_points (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) :=
    if (r1 + r2 < Real.dist c1 c2) then 0
    else
      let y := 3 in
      let x := 3 in
      (x - x) ^ 2 + (y - y) ^ 2
  dist_square_intersection_points center1 center2 radius1 radius2 = 0 :=
by
  let center1 := (3 : ℝ, -2 : ℝ)
  let center2 := (3 : ℝ, 6 : ℝ)
  let radius1 := 5
  let radius2 := 3
  have h1: (center1 = (3, -2)) := rfl
  have h2: (center2 = (3, 6)) := rfl
  have h3: (radius1 = 5) := rfl
  have h4: (radius2 = 3) := rfl
  -- Skip the full proof
  sorry

end square_dist_intersection_points_l741_741871


namespace quadrilateral_RSYP_area_l741_741350

def TriangleArea (a b c : ℝ) : ℝ := √((a+b+c) * (a+b-c) * (a+c-b) * (b+c-a)) / 4

noncomputable def midpoint_area (total_area : ℝ) (ratio1 ratio2 : ℝ) : ℝ :=
  total_area * ratio1 * ratio2 / (ratio1 + ratio2) / (ratio1 + ratio2)

noncomputable def problem (XY XZ area_XYZ : ℝ) : Prop :=
  let XP := XY / 2
  let XQ := XZ / 2
  let P := XP
  let Q := XQ
  let PQ := √(XP^2 + XQ^2) / 2
  let YP := 4 * Q
  let PZ := Q 
  (midpoint_area area_XYZ 1 4 + midpoint_area area_XYZ 1 32 - midpoint_area area_XYZ 3 18 = 182.5)

theorem quadrilateral_RSYP_area :
  problem 80 20 240 :=
by
  sorry

end quadrilateral_RSYP_area_l741_741350


namespace range_of_m_l741_741322

theorem range_of_m (m : ℝ) :
  (¬ ∃ x ∈ set.Icc (real.pi / 2) real.pi, sin x + sqrt 3 * cos x < m) ↔ m ≤ -sqrt 3 :=
by sorry

end range_of_m_l741_741322


namespace min_value_fraction_8_l741_741677

noncomputable def min_value_of_fraction (x y: ℝ) : Prop :=
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  let parallel := (3 * (y - 1)) = (-2) * x
  x > 0 ∧ y > 0 ∧ parallel → (∀ z, z = (3 / x) + (2 / y) → z ≥ 8)

theorem min_value_fraction_8 (x y : ℝ) (h_posx : x > 0) (h_posy : y > 0) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  let parallel := (3 * (y - 1)) = (-2) * x
  parallel → (3 / x) + (2 / y) ≥ 8 :=
by
  sorry

end min_value_fraction_8_l741_741677


namespace log2_n_tournament_l741_741864

theorem log2_n_tournament :
  let games := 435
  let total_outcomes := 2^games
  let distinct_outcomes := Nat.factorial 30
  let probability := distinct_outcomes / total_outcomes
  let n := 2^(games - (Nat.floorLog 2 (Nat.factorial 30)))
  log2 n = 409 :=
by
  intros
  let games := 435
  let total_outcomes := 2^games
  let distinct_outcomes := Nat.factorial 30
  let power_of_2 := Nat.floorLog 2 (Nat.factorial 30)
  let n := 2^(games - power_of_2)
  have log2_n_eq : log2 n = games - power_of_2 := by
    sorry
  exact log2_n_eq

# Reduce verbosity by collapsing steps into one clear statement

end log2_n_tournament_l741_741864


namespace min_value_am_hm_l741_741382

theorem min_value_am_hm (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end min_value_am_hm_l741_741382


namespace debt_payments_l741_741947

theorem debt_payments (n : ℕ) (h_avg : (20 * 410 + (n - 20) * 475) / n = 442.5) : n = 40 :=
sorry

end debt_payments_l741_741947


namespace find_smallest_number_label_1993_l741_741241

theorem find_smallest_number_label_1993:
  ∀ (points labeled: ℕ), points = 2000 → 
  (labeled = list.range 1993) → 
  let pos := (∑ i in list.range 1993, i + 1) % 2000 in
  let n := 1993 in
  ∃ (k: ℕ), ∃ i: ℕ, 
  i * (i + 1) / 2 ≡ 1021 + 2000 * k [MOD 2000] → 
  i = 118 := sorry

end find_smallest_number_label_1993_l741_741241


namespace difference_in_average_speed_l741_741472

theorem difference_in_average_speed 
  (distance : ℕ) 
  (time_diff : ℕ) 
  (speed_B : ℕ) 
  (time_B : ℕ) 
  (time_A : ℕ) 
  (speed_A : ℕ)
  (h1 : distance = 300)
  (h2 : time_diff = 3)
  (h3 : speed_B = 20)
  (h4 : time_B = distance / speed_B)
  (h5 : time_A = time_B - time_diff)
  (h6 : speed_A = distance / time_A) 
  : speed_A - speed_B = 5 := 
sorry

end difference_in_average_speed_l741_741472


namespace table_wobbles_l741_741970

-- Define lengths of the table legs
def leg_lengths : list ℝ := [70, 71, 72.5, 72]

-- Define positions of the legs assuming a square table
def leg_positions : list (ℝ × ℝ) := [(0,0), (1,0), (1,1), (0,1)]

-- Define the endpoints of the legs in 3D space
def endpoints : list (ℝ × ℝ × ℝ) :=
  leg_lengths.zipWith (λ l (x,y), (x, y, l)) leg_positions

-- Statement to prove the table wobbles (i.e., endpoints are not coplanar)
theorem table_wobbles : ¬ ∃ A B C D : ℝ × ℝ × ℝ, 
  A ∈ endpoints ∧ B ∈ endpoints ∧ C ∈ endpoints ∧ D ∈ endpoints ∧ 
  ∀ {x y z : ℝ}, ∃ a b c d : ℝ, a*x + b*y + c*z + d = 0 ∧ 
    a*(A.1) + b*(A.2) + c*(A.3) + d = 0 ∧ 
    a*(B.1) + b*(B.2) + c*(B.3) + d = 0 ∧ 
    a*(C.1) + b*(C.2) + c*(C.3) + d = 0 ∧ 
    a*(D.1) + b*(D.2) + c*(D.3) + d = 0 :=
sorry

end table_wobbles_l741_741970


namespace diff_of_squares_div_l741_741087

theorem diff_of_squares_div (a b : ℤ) (h1 : a = 121) (h2 : b = 112) : 
  (a^2 - b^2) / (a - b) = a + b :=
by
  rw [h1, h2]
  rw [sub_eq_add_neg, add_comm]
  exact sorry

end diff_of_squares_div_l741_741087


namespace expand_polynomial_l741_741571

theorem expand_polynomial (t : ℝ) : (2 * t^3 - 3 * t + 2) * (-3 * t^2 + 3 * t - 5) = 
  -6 * t^5 + 6 * t^4 - t^3 + 3 * t^2 + 21 * t - 10 :=
by
  sorry

end expand_polynomial_l741_741571


namespace trapezoid_exists_parallelogram_not_exists_l741_741356

-- Definitions for the conditions
variables (a b c d : set (ℝ × ℝ × ℝ))

-- Assumption stating the lines are pairwise skew lines
axiom pairwise_skew_lines : 
  ∀ (l m : set (ℝ × ℝ × ℝ)), l ≠ m → 
  (l ⊆ (plane) ∧ m ⊆ (plane) ∧ ∀ (a ∈ l) (b ∈ m), a ≠ b)

-- Proof problem for part (a)
theorem trapezoid_exists :
  ∃ A ∈ a, ∃ B ∈ b, ∃ C ∈ c, ∃ D ∈ d, 
  (same plane : ((convex_hull {A, B, C, D})).plane)

-- Proof problem for part (b)
theorem parallelogram_not_exists :
  ¬ ∃ A ∈ a, ∃ B ∈ b, ∃ C ∈ c, ∃ D ∈ d, 
  (parallelogram : 
    (convex_hull {A, B, C, D}).parallelogram)

end trapezoid_exists_parallelogram_not_exists_l741_741356


namespace sum_floor_diff_le_l741_741768

noncomputable def greatest_int_le (a: ℝ) : ℤ := ⌊a⌋  -- Definition of floor function

theorem sum_floor_diff_le (n : ℕ) (x : ℝ) (hx : 0 < x) :
  (∑ k in finset.range n, 
    (x * (greatest_int_le (k / x)) - (x + 1) * (greatest_int_le (k / (x + 1))))) ≤ n :=
by
  sorry

end sum_floor_diff_le_l741_741768


namespace minimum_throws_l741_741024

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741024


namespace value_of_a_l741_741323

-- Definitions based on conditions
def cond1 (a : ℝ) := |a| - 1 = 0
def cond2 (a : ℝ) := a + 1 ≠ 0

-- The main proof problem
theorem value_of_a (a : ℝ) : (cond1 a ∧ cond2 a) → a = 1 :=
by
  sorry

end value_of_a_l741_741323


namespace solve_for_x_l741_741233

theorem solve_for_x (x : ℝ) : 2^(2*x) * 50^x = 250^3 → x = 1 :=
by
  sorry

end solve_for_x_l741_741233


namespace area_of_triangle_l741_741538

def line1 (x : ℝ) : ℝ := 2 * x + 4
def line2 (x : ℝ) : ℝ := - (1/2) * x + 3
def line3 (x : ℝ) : ℝ := 2

theorem area_of_triangle : 
  let A := (-1, line3 (-1)) in
  let B := (2, line3 2) in
  let C := (-2/5, line1 (-2/5)) in
  let base := dist (A.1, 0) (B.1, 0) in
  let height := dist (0, line3 0) (0, C.2) in
  (1/2:ℝ) * base * height = 2.4 := 
by
  sorry

end area_of_triangle_l741_741538


namespace minimum_throws_to_ensure_same_sum_twice_l741_741016

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741016


namespace smallest_nat_with_55_divisors_l741_741597

open BigOperators

theorem smallest_nat_with_55_divisors :
  ∃ (n : ℕ), 
    (∃ (f : ℕ → ℕ) (primes : Finset ℕ),
      (∀ p ∈ primes, Nat.Prime p) ∧ 
      (primes.Sum (λ p => p ^ (f p))) = n ∧
      ((primes.Sum (λ p => f p + 1)) = 55)) ∧
    (∀ m, 
      (∃ (f_m : ℕ → ℕ) (primes_m : Finset ℕ),
        (∀ p ∈ primes_m, Nat.Prime p) ∧ 
        (primes_m.Sum (λ p => p ^ (f_m p))) = m ∧
        ((primes_m.Sum (λ p => f_m p + 1)) = 55)) → 
      n ≤ m) ∧
  n = 3^4 * 2^10 := 
begin
  sorry
end

end smallest_nat_with_55_divisors_l741_741597


namespace jim_retail_profit_l741_741746

theorem jim_retail_profit :
  let profit_per_tire_repair := 20 - 5 in
  let total_tire_repairs := 300 in
  let profit_per_complex_repair := 300 - 50 in
  let total_complex_repairs := 2 in
  let fixed_expenses := 4000 in
  let total_profit := 3000 in
  let tire_repair_profit := total_tire_repairs * profit_per_tire_repair in
  let complex_repair_profit := total_complex_repairs * profit_per_complex_repair in
  let total_repair_profit := tire_repair_profit + complex_repair_profit in
  let profit_after_expenses := total_repair_profit - fixed_expenses in
  let retail_profit := total_profit - profit_after_expenses in
  retail_profit = 2000 :=
by
  -- placeholder for the proof
  sorry

end jim_retail_profit_l741_741746


namespace proof_equivalence_l741_741294

variable {x y : ℝ}

theorem proof_equivalence (h : x - y = 1) : x^3 - 3 * x * y - y^3 = 1 := by
  sorry

end proof_equivalence_l741_741294


namespace seven_expression_one_seven_expression_two_l741_741903

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l741_741903


namespace inverse_proportion_points_relation_l741_741315

theorem inverse_proportion_points_relation {y1 y2 : ℝ} :
  (∃ k : ℝ, (λ x, k / x) (-2) = y1 ∧ (λ x, k / x) 1 = y2 ∧ (λ x, k / x) 2 = 1) → y1 < 1 ∧ 1 < y2 :=
by {
  intro h,
  obtain ⟨k, hk1, hk2, hk3⟩ := h,
  rw [← hk3, one_div_eq_inv] at hk1 hk2,
  have k_eq : k = 2 := sorry,
  rw k_eq at *,
  rw [inv_mul_cancel_left₀] at hk1 hk2,
  exact sorry
}

end inverse_proportion_points_relation_l741_741315


namespace min_distance_l741_741412

-- Define the paths parameterized by t for Rational Man and Irrational Man
def rationalManPath : ℝ → ℝ × ℝ := λ t, (2 * Real.cos t, 2 * Real.sin t)
def irrationalManPath : ℝ → ℝ × ℝ := λ t, (2 + 3 * Real.cos (t / Real.sqrt 3), 3 * Real.sin (t / Real.sqrt 3))

-- Define the function to calculate the distance between two points (x1, y1) and (x2, y2)
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Prove the smallest possible distance AB between points on Rational Man's and Irrational Man's paths
theorem min_distance :
  ∃ t1 t2 : ℝ, distance (rationalManPath t1) (irrationalManPath t2) = (17 - 6 * Real.sqrt 5) / (3 * Real.sqrt 5) := by
  sorry

end min_distance_l741_741412


namespace fraction_of_solution_replaced_eq_third_l741_741952

-- Define the initial conditions
def container_volume : ℝ := 100
def initial_concentration : ℝ := 0.40
def new_concentration : ℝ := 0.35
def replacement_concentration : ℝ := 0.25

-- Define the quantity of solution replaced
def quantity_replaced (x : ℝ) : ℝ := x

-- Define the amount of solute in the solution before and after replacement
def initial_solute : ℝ := container_volume * initial_concentration
def solute_removed (x : ℝ) : ℝ := quantity_replaced x * initial_concentration
def solute_added (x : ℝ) : ℝ := quantity_replaced x * replacement_concentration

-- Statement to be proved
theorem fraction_of_solution_replaced_eq_third (x : ℝ) (h : 0 ≤ x ∧ x ≤ container_volume) :
  initial_solute - solute_removed x + solute_added x = container_volume * new_concentration →
  x / container_volume = 1 / 3 :=
begin
  sorry
end

end fraction_of_solution_replaced_eq_third_l741_741952


namespace tanya_erasers_l741_741295

theorem tanya_erasers (H R TR T : ℕ) 
  (h1 : H = 2 * R) 
  (h2 : R = TR / 2 - 3) 
  (h3 : H = 4) 
  (h4 : TR = T / 2) : 
  T = 20 := 
by 
  sorry

end tanya_erasers_l741_741295


namespace general_formula_b_sum_T_n_l741_741250

variable {n : ℕ}

-- Definitions from conditions
def S : ℕ → ℤ := λ n, 3 * n^2 + 8 * n
def a (n : ℕ) : ℤ := S n - S (n - 1)
def b (n : ℕ) : ℤ := 3 * n + 1
def c (n : ℕ) : ℤ := ((a n + 1) ^ (n + 1)) / ((b n + 2) ^ n)

-- Definitions to encapsulate the statements to be proven
def arithmetic_sequence (u : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, u (n + 1) = u n + d

theorem general_formula_b :
  (arithmetic_sequence b) ∧ (∀ n, b n = 3 * n + 1) :=
by
  split
  -- Show that b is an arithmetic sequence
  {
    use 3
    intro n
    rw [b, b]
    ring
  }
  -- Show the general formula for b
  intro n
  exact rfl

theorem sum_T_n :
  ∀ n, (Σ i in finset.range n, c i) = 3 * n * 2 ^ (n + 2) :=
by
  intro n
  sorry

end general_formula_b_sum_T_n_l741_741250


namespace denver_wood_used_per_birdhouse_l741_741194

-- Definitions used in the problem
def cost_per_piee_of_wood : ℝ := 1.50
def profit_per_birdhouse : ℝ := 5.50
def price_for_two_birdhouses : ℝ := 32
def num_birdhouses_purchased : ℝ := 2

-- Property to prove
theorem denver_wood_used_per_birdhouse (W : ℝ) 
  (h : num_birdhouses_purchased * (cost_per_piee_of_wood * W + profit_per_birdhouse) = price_for_two_birdhouses) : 
  W = 7 :=
sorry

end denver_wood_used_per_birdhouse_l741_741194


namespace shared_property_l741_741093

-- Definitions of the shapes with their properties
structure Parallelogram (α : Type) [EuclideanSpace α] :=
(opposite_sides_parallel_equal : ∀ {a b : α}, parallel a b ∧ equal_length a b)

structure Rectangle (α : Type) [EuclideanSpace α] extends Parallelogram α :=
(all_right_angles : ∀ a : α, right_angle a)
(equal_length_diagonals : ∀ {a b : α}, equal_length a b)

structure Rhombus (α : Type) [EuclideanSpace α] :=
(all_sides_equal : ∀ a b : α, equal_length a b)
(opposite_sides_parallel : ∀ a b : α, parallel a b)
(perpendicular_diagonals : ∀ {a b : α}, perpendicular a b)

structure Square (α : Type) [EuclideanSpace α] extends Rectangle α, Rhombus α

-- The theorem to be proven
theorem shared_property (α : Type) [EuclideanSpace α] : 
  ∀ (P : Parallelogram α) (R : Rectangle α) (H : Rhombus α) (S : Square α), 
    (P.opposite_sides_parallel_equal ∧ R.opposite_sides_parallel_equal ∧ H.opposite_sides_parallel ∧ S.opposite_sides_parallel) :=
begin
  sorry
end

end shared_property_l741_741093


namespace find_n_for_trailing_zeros_l741_741581

theorem find_n_for_trailing_zeros :
  ∃ n : ℕ, n ≥ 48 ∧ (∑ k in Finset.range (Nat.log 5 n + 1), n / (5 ^ k)) = n - 48 ∧ n = 62 :=
by
  sorry

end find_n_for_trailing_zeros_l741_741581


namespace min_throws_to_ensure_same_sum_twice_l741_741069

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741069


namespace geometric_sequence_arithmetic_sequence_l741_741734

theorem geometric_sequence (n : ℕ) : 
    ∀ a₁ a₄ : ℕ, a₁ = 2 ∧ a₄ = 16 → ∃ (q : ℕ), a_n = 2^(n) :=
by
  sorry

theorem arithmetic_sequence (n : ℕ) : 
    ∀ a₃ a₅ : ℕ, ∀ b₄ b₁ b₁₆ d : ℕ, a₃ = 8 ∧ a₅ = 32 ∧ (b₄ = a₃ ∧ b₁₆ = a₅) → 
    (b₁ + 3 * d = 8 ∧ b₁ + 15 * d = 32) ∧ b₁ = 2 ∧ d = 2 ∧ b_n = 2 * n ∧ ∑ i in range n, b_i = n^2 + n :=
by
  sorry

end geometric_sequence_arithmetic_sequence_l741_741734


namespace expression_divisible_by_7_l741_741486

theorem expression_divisible_by_7 (n : ℕ) (hn : n > 0) :
  7 ∣ (3^(3*n+1) + 5^(3*n+2) + 7^(3*n+3)) :=
sorry

end expression_divisible_by_7_l741_741486


namespace red_gumballs_count_l741_741522

def gumballs_problem (R B G : ℕ) : Prop :=
  B = R / 2 ∧
  G = 4 * B ∧
  R + B + G = 56

theorem red_gumballs_count (R B G : ℕ) (h : gumballs_problem R B G) : R = 16 :=
by
  rcases h with ⟨h1, h2, h3⟩
  sorry

end red_gumballs_count_l741_741522


namespace dice_arithmetic_sequence_probability_l741_741555

theorem dice_arithmetic_sequence_probability :
  ∀ (die_roll : ℕ → fin 6) (seq : ℕ → ℕ),
  (∀ n, 1 ≤ seq n ∧ seq n ≤ 6) →
  (finset.card (finset.filter (λ s : ℕ × ℕ × ℕ, (s.2.1 - s.1) = (s.2.2 - s.2.1)) finset.univ) / 216 = 1 / 12) :=
by
  sorry

end dice_arithmetic_sequence_probability_l741_741555


namespace minimum_throws_l741_741023

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741023


namespace number_of_fractions_is_2_l741_741163

def is_fraction (expr : String) : Bool :=
  expr = "(1/x)" ∨ 
  expr = "(a/(3-2a))"

def expressions : List String :=
  ["(1/x)", "(x^2 + 5x)", "(1/2 * x)", "(a/(3-2a))", "(3.14/pi)"]

theorem number_of_fractions_is_2 :
  (expressions.countp is_fraction) = 2 :=
by
  sorry

end number_of_fractions_is_2_l741_741163


namespace magazine_purchasing_methods_l741_741127

theorem magazine_purchasing_methods :
  ∃ n : ℕ, n = 266 ∧ 
  (∃ (m2 m1 : Finset Nat), 
    (m2.card = 8 ∧ 
     m1.card = 3 ∧
     ∃ (chosen2 chosen1 : Finset Nat), 
       (chosen2 ⊆ m2 ∧ 
        chosen1 ⊆ m1 ∧ 
        (chosen2.card * 2 + chosen1.card) = 10 ∧ 
        ((chosen2.card = 5 ∧ chosen1.card = 0) ∨ 
         (chosen2.card = 4 ∧ chosen1.card = 2 ∧ chosen2.card.choose 4 * chosen1.card.choose 2 = 210)
        )
       )
    )
  )
: sorry

end magazine_purchasing_methods_l741_741127


namespace program_output_is_24_l741_741810

def compute_output (x : ℕ) : ℕ :=
  if x < 3 then 2 * x
  else if x > 3 then x * x - 1
  else 2

theorem program_output_is_24 (x : ℕ) (h : x = 5) : compute_output x = 24 :=
by {
  rw h,
  unfold compute_output,
  simp,
  sorry,
}

end program_output_is_24_l741_741810


namespace droneSystemEquations_l741_741131

-- Definitions based on conditions
def typeADrones (x y : ℕ) : Prop := x = (1/2 : ℝ) * (x + y) + 11
def typeBDrones (x y : ℕ) : Prop := y = (1/3 : ℝ) * (x + y) - 2

-- Theorem statement
theorem droneSystemEquations (x y : ℕ) :
  typeADrones x y ∧ typeBDrones x y ↔
  (x = (1/2 : ℝ) * (x + y) + 11 ∧ y = (1/3 : ℝ) * (x + y) - 2) :=
by sorry

end droneSystemEquations_l741_741131


namespace smallest_nat_with_55_divisors_l741_741598

open BigOperators

theorem smallest_nat_with_55_divisors :
  ∃ (n : ℕ), 
    (∃ (f : ℕ → ℕ) (primes : Finset ℕ),
      (∀ p ∈ primes, Nat.Prime p) ∧ 
      (primes.Sum (λ p => p ^ (f p))) = n ∧
      ((primes.Sum (λ p => f p + 1)) = 55)) ∧
    (∀ m, 
      (∃ (f_m : ℕ → ℕ) (primes_m : Finset ℕ),
        (∀ p ∈ primes_m, Nat.Prime p) ∧ 
        (primes_m.Sum (λ p => p ^ (f_m p))) = m ∧
        ((primes_m.Sum (λ p => f_m p + 1)) = 55)) → 
      n ≤ m) ∧
  n = 3^4 * 2^10 := 
begin
  sorry
end

end smallest_nat_with_55_divisors_l741_741598


namespace calculate_expression_l741_741990

-- Define the conditions
def exp1 : ℤ := (-1)^(53)
def exp2 : ℤ := 2^(2^4 + 5^2 - 4^3)

-- State and skip the proof
theorem calculate_expression :
  exp1 + exp2 = -1 + 1 / (2^23) :=
by sorry

#check calculate_expression

end calculate_expression_l741_741990


namespace burger_cost_proof_l741_741177

variable {burger_cost fries_cost salad_cost total_cost : ℕ}
variable {quantity_of_fries : ℕ}

theorem burger_cost_proof (h_fries_cost : fries_cost = 2)
    (h_salad_cost : salad_cost = 3 * fries_cost)
    (h_quantity_of_fries : quantity_of_fries = 2)
    (h_total_cost : total_cost = 15)
    (h_equation : burger_cost + (quantity_of_fries * fries_cost) + salad_cost = total_cost) :
    burger_cost = 5 :=
by 
  sorry

end burger_cost_proof_l741_741177


namespace zongzi_price_equation_l741_741566

-- Define the variables and conditions
variables (x : ℝ) (total_cost : ℝ)
variables (num_meat_zongzi num_veg_zongzi : ℝ)
variables (price_meat_zongzi price_veg_zongzi : ℝ)

-- Assert the given conditions
def question (x : ℝ) (total_cost : ℝ) (num_meat_zongzi num_veg_zongzi : ℝ) (price_meat_zongzi price_veg_zongzi : ℝ) :=
  num_meat_zongzi = 10 ∧
  num_veg_zongzi = 5 ∧
  price_meat_zongzi = x ∧
  price_veg_zongzi = x - 1 ∧
  total_cost = 70

-- Prove that the correct equation is 10x + 5(x - 1) = 70
theorem zongzi_price_equation (x : ℝ) (total_cost : ℝ) (num_meat_zongzi num_veg_zongzi : ℝ) (price_meat_zongzi price_veg_zongzi : ℝ) :
  question x total_cost num_meat_zongzi price_meat_zongzi price_veg_zongzi →
  10 * price_meat_zongzi + 5 * (price_meat_zongzi - 1) = 70 :=
begin
  sorry
end

end zongzi_price_equation_l741_741566


namespace art_of_war_rice_storage_l741_741941

/-
We are given:
- circumference of a cylindrical base as five zhang and four chi (54 chi)
- height of the cylinder as one zhang and eight chi (18 chi)
- one zhang is ten chi
- π is approximately 3
- the volume of one bushel of rice is approximately 1.62 cubic chi
- formula for the volume of the cylinder (circumference squared, multiplied by the height, divided by 12)
-/

noncomputable def pi_approx : ℝ := 3
noncomputable def chi_per_zhang : ℝ := 10
noncomputable def circumference : ℝ := 54
noncomputable def height : ℝ := 18
noncomputable def volume_per_bushel : ℝ := 1.62

def radius (c : ℝ) (pi : ℝ) : ℝ := c / (2 * pi)

def volume (r : ℝ) (h : ℝ) (pi : ℝ) : ℝ := pi * r^2 * h

def estimated_bushels (V : ℝ) (vpb : ℝ) : ℝ := V / vpb

def celler_estimated_bushels : ℝ :=
  let r := radius circumference pi_approx in
  let V := volume r height pi_approx in
  estimated_bushels V volume_per_bushel

theorem art_of_war_rice_storage :
  celler_estimated_bushels = 2700 := sorry

end art_of_war_rice_storage_l741_741941


namespace magnitude_of_2a_plus_b_l741_741292

variables (a b : ℝ^n)  -- Assume a and b are vectors in n-dimensional real space

-- Definition of the conditions
def magnitude_a : Prop := ∥a∥ = 1
def magnitude_diff_ab : Prop := ∥a - b∥ = sqrt 3
def dot_product_cond : Prop := a ⬝ (a - b) = 0

-- The statement we want to prove
theorem magnitude_of_2a_plus_b (ha : magnitude_a a) (hb : magnitude_diff_ab a b) (hc : dot_product_cond a b) : 
  ∥2 • a + b∥ = 2 * sqrt 3 :=
sorry

end magnitude_of_2a_plus_b_l741_741292


namespace probability_at_least_one_even_l741_741484

noncomputable def choose : ℕ → ℕ → ℕ
| 0, 0 => 1
| n, 0 => 1
| 0, k => 0
| n, k => choose (n-1) (k-1) + choose (n-1) k

/-- 
When randomly selecting two numbers from 1, 2, 3, and 4, 
the probability that at least one of the selected numbers is even is 5/6.
-/
theorem probability_at_least_one_even :
  let total_outcomes := choose 4 2,
      outcomes_all_odd := choose 2 2,
      probability_at_least_one_even := 1 - (outcomes_all_odd / total_outcomes : ℚ)
  in probability_at_least_one_even = 5 / 6 := 
by 
  sorry

end probability_at_least_one_even_l741_741484


namespace infinite_solutions_x2_plus_2y2_eq_z2_no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2_l741_741818

-- Define x, y, z to be natural numbers
def has_infinitely_many_solutions : Prop :=
  ∃ (x y z : ℕ), x^2 + 2 * y^2 = z^2

-- Prove that there are infinitely many such x, y, z
theorem infinite_solutions_x2_plus_2y2_eq_z2 : has_infinitely_many_solutions :=
  sorry

-- Define x, y, z, t to be integers and non-zero
def no_nontrivial_integer_quadruplets : Prop :=
  ∀ (x y z t : ℤ), 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) → 
    ¬((x^2 + 2 * y^2 = z^2) ∧ (2 * x^2 + y^2 = t^2))

-- Prove that no nontrivial integer quadruplets exist
theorem no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2 : no_nontrivial_integer_quadruplets :=
  sorry

end infinite_solutions_x2_plus_2y2_eq_z2_no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2_l741_741818


namespace find_m_t_l741_741754

noncomputable def T := {x : ℝ // x ≠ 0}

def g (f : T → T) : Prop :=
  ∀ (x y : T), x.val + y.val ≠ 0 → f x + f y = 4 * f ⟨(x.val * y.val) / (f ⟨x.val + y.val, by exact add_ne_zero x.property y.property⟩).val, sorry⟩

theorem find_m_t (f : T → T) (h : g f) : 
  (∃ m t : ℕ, m * t = 8) :=
sorry

end find_m_t_l741_741754


namespace train_speed_l741_741494

theorem train_speed
  (length_of_train : ℝ)
  (time_to_cross_pole : ℝ)
  (h1 : length_of_train = 3000)
  (h2 : time_to_cross_pole = 120) :
  length_of_train / time_to_cross_pole = 25 :=
by {
  sorry
}

end train_speed_l741_741494


namespace minimum_throws_to_ensure_same_sum_twice_l741_741006

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741006


namespace appropriate_sampling_method_l741_741518

-- Definitions (conditions from the problem)
def company_has_three_models : Prop := true
def significant_differences_between_models : Prop := true

-- The main theorem (question and correct answer in the solution)
theorem appropriate_sampling_method : 
  company_has_three_models → 
  significant_differences_between_models → 
  (method : Type) → 
  (method = "stratified_sampling") :=
by
  intros _ _
  sorry

end appropriate_sampling_method_l741_741518


namespace annual_interest_rate_l741_741745

theorem annual_interest_rate (A P : ℝ) (r : ℝ) (n t : ℕ)
  (hP : P = 10000) (hA : A = 10815.83) (hn : n = 2) (ht : t = 2) 
  (hA_eq : A = P * (1 + r / n)^ (n * t)) : 
  r ≈ 0.0398 :=
by
  have h1 : 10815.83 = 10000 * (1 + r / 2) ^ (2 * 2), from hA_eq,
  sorry

end annual_interest_rate_l741_741745


namespace probability_htth_l741_741723

def probability_of_sequence_HTTH := (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)

theorem probability_htth : probability_of_sequence_HTTH = 1 / 16 := by
  sorry

end probability_htth_l741_741723


namespace odd_n_sigma_floor_log2_sum_lt_n_squared_over_8_l741_741791

variable (n : ℕ)

noncomputable def σ (k : ℕ) : ℕ := ∑ d in (List.range k.succ).filter (λ x, x ∣ k), d

theorem odd_n_sigma_floor_log2_sum_lt_n_squared_over_8
  (hn_odd : odd n) :
  ∑ i in List.range (n + 1), if odd i then σ i * ⌊ real.log 2 (n / i) ⌋ else 0 < n^2 / 8 :=
by sorry

end odd_n_sigma_floor_log2_sum_lt_n_squared_over_8_l741_741791


namespace find_square_sum_l741_741765

theorem find_square_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 2 / 7 :=
by
  sorry

end find_square_sum_l741_741765


namespace certain_number_is_2_l741_741311

theorem certain_number_is_2 
    (X : ℕ) 
    (Y : ℕ) 
    (h1 : X = 15) 
    (h2 : 0.40 * (X : ℝ) = 0.80 * 5 + (Y : ℝ)) : 
    Y = 2 := 
  sorry

end certain_number_is_2_l741_741311


namespace soldiers_age_26_to_29_participation_l741_741721

theorem soldiers_age_26_to_29_participation 
  (total_soldiers : ℕ)
  (soldiers_18_to_21 : ℕ)
  (soldiers_22_to_25 : ℕ)
  (soldiers_26_to_29 : ℕ)
  (total_spots : ℕ)
  (equal_probability : Prop)
  (h1 : total_soldiers = 45)
  (h2 : soldiers_18_to_21 = 15)
  (h3 : soldiers_22_to_25 = 20)
  (h4 : soldiers_26_to_29 = 10)
  (h5 : total_spots = 9)
  (h6 : equal_probability → (∀ s, s ∈ {soldiers_18_to_21, soldiers_22_to_25, soldiers_26_to_29} → s / total_soldiers = total_spots / total_soldiers)) :
  (soldiers_26_to_29 * (total_spots / total_soldiers) = 2) :=
by
  sorry

end soldiers_age_26_to_29_participation_l741_741721


namespace gear_p_revolutions_per_minute_l741_741186

theorem gear_p_revolutions_per_minute (r : ℝ) 
  (cond2 : ℝ := 40) 
  (cond3 : 1.5 * r + 45 = 1.5 * 40) :
  r = 10 :=
by
  sorry

end gear_p_revolutions_per_minute_l741_741186


namespace minimum_throws_for_repeated_sum_l741_741038

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741038


namespace nonnegative_integers_count_l741_741688

theorem nonnegative_integers_count :
  let form := λ a : Fin 8 → ℤ, ∑ i, a i * (2^i : ℤ)
  (condition : ∀ i, a i ∈ {-1, 0, 1})
  in ∃! n, (0 ≤ n) ∧ (n ≤ form (λ _, 1)) :=
by
  sorry

end nonnegative_integers_count_l741_741688


namespace expected_value_counter_l741_741519

theorem expected_value_counter :
  let E (n : ℕ) : ℚ := 1 - (1 / (2 ^ n))
  let m := 1023
  let n := 1024
  assert : E 10 = (1023 / 1024)
  assert : Nat.gcd m n = 1
  100 * m + n = 103324 :=
by sorry

end expected_value_counter_l741_741519


namespace new_average_l741_741108

variable (numbers : Fin 15 → ℝ)

def average (nums : Fin 15 → ℝ) : ℝ :=
  (∑ i, nums i) / 15

theorem new_average (h : average numbers = 40) : average (λ i, numbers i + 11) = 51 := by
  sorry

end new_average_l741_741108


namespace triangle_ABC_area_l741_741352

theorem triangle_ABC_area (A B C K L : Type) [InnerProductSpace ℝ A] :
  let AB : ℝ := 40
  let BC : ℝ := 26
  let K := midpoint ℝ A B
  let L := midpoint ℝ B C
  let AKLC_cyclic : Kaly ≤ A
  ∃ area : ℝ, area = 1 / 2 * AB * BC * sin (angle A B C) := 264 := 
begin
  sorry,
end

end triangle_ABC_area_l741_741352


namespace vasya_example_fewer_sevens_l741_741876

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l741_741876


namespace work_days_together_l741_741495

variable (d : ℝ) (j : ℝ)

theorem work_days_together (hd : d = 1 / 5) (hj : j = 1 / 9) :
  1 / (d + j) = 45 / 14 := by
  sorry

end work_days_together_l741_741495


namespace minimum_rolls_for_duplicate_sum_l741_741058

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741058


namespace problem1_problem2_l741_741380

def vecA (x : ℝ) : ℝ × ℝ := (1 + Real.cos x, 1 + Real.sin x)
def vecB : ℝ × ℝ := (1, 0)
def vecC : ℝ × ℝ := (1, 2)

theorem problem1 (x : ℝ) : 
  let a_b := (vecA x).fst - vecB.fst, (vecA x).snd - vecB.snd
  let a_c := (vecA x).fst - vecC.fst, (vecA x).snd - vecC.snd
  a_b.1 * a_c.1 + a_b.2 * a_c.2 = 0 := 
sorry

theorem problem2 :
  ∃ (x : ℝ), ∀ (k : ℤ), 
  1 + Real.sqrt(2) = Real.sqrt (3 + 2 * Real.sqrt(2)) ∧ 
  x = 2 * k * Real.pi + Real.pi / 4 := 
sorry

end problem1_problem2_l741_741380


namespace subtraction_of_tenths_l741_741499

theorem subtraction_of_tenths (a b : ℝ) (n : ℕ) (h1 : a = (1 / 10) * 6000) (h2 : b = (1 / 10 / 100) * 6000) : (a - b) = 594 := by
sorry

end subtraction_of_tenths_l741_741499


namespace _l741_741635

noncomputable theorem correct_expression (
  alpha beta : ℝ)
  (h1 : alpha ∈ Ioo 0 (π/2))
  (h2 : beta ∈ Ioo 0 (π/2))
  (h3 : tan alpha = (1 + sin beta) / cos beta)
  : 2 * alpha - beta = π / 2 := sorry

end _l741_741635


namespace least_number_of_square_tiles_l741_741099

theorem least_number_of_square_tiles
  (length_cm : ℕ) (width_cm : ℕ)
  (h1 : length_cm = 816) (h2 : width_cm = 432) :
  ∃ tile_count : ℕ, tile_count = 153 :=
by
  sorry

end least_number_of_square_tiles_l741_741099


namespace pool_one_quarter_capacity_in_six_hours_l741_741111

theorem pool_one_quarter_capacity_in_six_hours (d : ℕ → ℕ) :
  (∀ n : ℕ, d (n + 1) = 2 * d n) → d 8 = 2^8 →
  d 6 = 2^6 :=
by
  intros h1 h2
  sorry

end pool_one_quarter_capacity_in_six_hours_l741_741111


namespace trapezoid_APQC_circumscribed_l741_741349

-- Definitions based on the conditions
variables {A B C P Q R S M K L : Type} [add_comm_group A] [module real A]
variables (PQ RS AC BM : set A)
variables (RPKL MLSC APQC : set A)

-- Conditions
def parallel (PQ AC : set A) : Prop := sorry
def circumscribed (RPKL MLSC APQC : set A) : Prop := sorry

-- The main statement
theorem trapezoid_APQC_circumscribed
  (h_parallel_PQ_RS : PQ ∥ RS)
  (h_parallel_PQ_AC : PQ ∥ AC)
  (h_parallel_RS_AC : RS ∥ AC)
  (h_segment_BM_exists : BM = BM)
  (h_circumscribed_RPKL : circumscribed RPKL)
  (h_circumscribed_MLSC : circumscribed MLSC) :
  circumscribed APQC :=
sorry

end trapezoid_APQC_circumscribed_l741_741349


namespace hyperbola_eqn_l741_741639

theorem hyperbola_eqn (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = sqrt 3 / 2) 
(h4 : ∃ c, c^2 = a^2 + b^2 ∧ c = sqrt 7) :
  (∃ h : a^2 = 4 ∧ b^2 = 3, (∀ x y : ℝ, (x^2 / 4 - y^2 / 3 = 1 ↔ ∃ (ha : x^2 / a^2 - y^2 / b^2 = 1), True))) :=
by 
  sorry

end hyperbola_eqn_l741_741639


namespace mean_of_first_set_is_67_l741_741444

theorem mean_of_first_set_is_67 (x : ℝ) 
  (h : (50 + 62 + 97 + 124 + x) / 5 = 75.6) : 
  (28 + x + 70 + 88 + 104) / 5 = 67 := 
by
  sorry

end mean_of_first_set_is_67_l741_741444


namespace man_speed_is_5_point_98_kmph_l741_741974

/-- 
  Given:
  - Train length: 275 meters
  - Train speed: 60 kmph
  - Time taken to pass a man running in the opposite direction: 15 seconds

  Prove that the speed of the man is 5.98 kmph.
-/

theorem man_speed_is_5_point_98_kmph 
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (time_seconds : ℝ)
  (man_speed_kmph : ℝ)
  (h1 : train_length = 275)
  (h2 : train_speed_kmph = 60)
  (h3 : time_seconds = 15)
  (h4 : man_speed_kmph = 5.98) :
  true :=
by
  -- Convert train speed to m/s
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  -- Calculate relative speed (distance / time)
  let relative_speed_mps := train_length / time_seconds
  -- Calculate man's speed in m/s
  let man_speed_mps := relative_speed_mps - train_speed_mps
  -- Convert man's speed back to kmph
  let man_speed_kmph_calculated := man_speed_mps * 3600 / 1000
  -- Check that man's speed is approximately 5.98 kmph
  have : man_speed_kmph_calculated ≈ man_speed_kmph := sorry
  have : man_speed_kmph = 5.98 := h4
  trivial

end man_speed_is_5_point_98_kmph_l741_741974


namespace minimum_throws_l741_741021

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741021


namespace seven_expression_one_seven_expression_two_l741_741897

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l741_741897


namespace parallelepiped_diagonal_l741_741434

theorem parallelepiped_diagonal 
  (x y z m n p d : ℝ)
  (h1 : x^2 + y^2 = m^2)
  (h2 : x^2 + z^2 = n^2)
  (h3 : y^2 + z^2 = p^2)
  : d = Real.sqrt ((m^2 + n^2 + p^2) / 2) := 
sorry

end parallelepiped_diagonal_l741_741434


namespace percentage_changed_l741_741973

structure SurveyStats where
  totalParents : ℕ
  upgradedPercent : ℝ
  maintainedPercent : ℝ
  downgradedPercent : ℝ

theorem percentage_changed (stats : SurveyStats)
  (h1 : stats.totalParents = 120)
  (h2 : stats.upgradedPercent = 0.30)
  (h3 : stats.maintainedPercent = 0.60)
  (h4 : stats.downgradedPercent = 0.10) :
  let changedPercent := ((stats.upgradedPercent + stats.downgradedPercent) * 100) in
  changedPercent = 40 :=
by 
  sorry

end percentage_changed_l741_741973


namespace total_money_9pennies_4nickels_3dimes_l741_741691

def value_of_pennies (num_pennies : ℕ) : ℝ := num_pennies * 0.01
def value_of_nickels (num_nickels : ℕ) : ℝ := num_nickels * 0.05
def value_of_dimes (num_dimes : ℕ) : ℝ := num_dimes * 0.10

def total_value (pennies nickels dimes : ℕ) : ℝ :=
  value_of_pennies pennies + value_of_nickels nickels + value_of_dimes dimes

theorem total_money_9pennies_4nickels_3dimes :
  total_value 9 4 3 = 0.59 :=
by 
  sorry

end total_money_9pennies_4nickels_3dimes_l741_741691


namespace not_possible_first_case_possible_second_case_l741_741826

-- Define what a strict trapezoid is
structure StrictTrapezoid (V : Type) :=
  (vertices : V × V × V × V)
  (is_strict_trapezoid : (∃ p q : V, the vertices (p, q, p, q)))

-- Define a polyhedron with faces being strict trapezoids
structure Polyhedron (V : Type) :=
  (faces : list (StrictTrapezoid V))
  (convex : Prop)
  (each_edge_base_and_leg : ∀ edge : V × V, (∃ f1 f2 : StrictTrapezoid V, 
      (edge ∈ (f1.vertices)) ∧ (edge ∈ (f2.vertices))
      ∧ (is_base f1 edge) ∧ (is_leg f2 edge)))

-- Define relaxed condition for the edge requirement
structure PolyhedronRelaxed (V : Type) :=
  (faces : list (StrictTrapezoid V))
  (convex : Prop)

-- The first case: proving impossibility under strict conditions
theorem not_possible_first_case (V : Type) [DecidableEq V] :
  ¬(∃ p : Polyhedron V, true) :=
  sorry

-- The second case: proving possibility under relaxed conditions
theorem possible_second_case (V : Type) [DecidableEq V] :
  ∃ p : PolyhedronRelaxed V, true :=
  sorry

end not_possible_first_case_possible_second_case_l741_741826


namespace find_BP_l741_741503

theorem find_BP 
  (A B C D P : Type*) 
  (h_circle : is_on_circle A B C D) 
  (h_intersect : ∃ (u v : Type*), u = AC ∧ v = BD ∧ ∃ P, point_on (AC) P ∧ point_on (BD) P) 
  (h_AP : AP = 9) 
  (h_PC : PC = 2) 
  (h_BD : BD = 10) 
  (h_BP_lt_DP : BP < DP) : 
  BP = 5 - sqrt 7 := 
sorry

end find_BP_l741_741503


namespace int_product_negative_max_negatives_l741_741708

theorem int_product_negative_max_negatives (n : ℤ) (hn : n ≤ 9) (hp : n % 2 = 1) :
  ∃ m : ℤ, n + m = m ∧ m ≥ 0 :=
by
  use 9
  sorry

end int_product_negative_max_negatives_l741_741708


namespace unique_function_l741_741588

def satisfies_inequality (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x + z) + k * f x * f (y * z) ≥ k^2

theorem unique_function (k : ℤ) (h : k > 0) :
  ∃! f : ℝ → ℝ, satisfies_inequality f k :=
by
  sorry

end unique_function_l741_741588


namespace min_throws_to_ensure_same_sum_twice_l741_741077

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741077


namespace magic_square_y_value_l741_741326

theorem magic_square_y_value :
  ∃ y : ℤ, y = 175 ∧ 
  ∀ a b c d e : ℤ, 
    y + 23 + 84 = 3 + a + b ∧
    y + 3 + c = 3 + a + 84 ∧
    y + (y - 81) + e = 84 + b + e →
    y = 175 :=
by
  use 175
  intros a b c d e h1 h2 h3
  sorry

end magic_square_y_value_l741_741326


namespace min_throws_to_ensure_same_sum_twice_l741_741078

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741078


namespace jellybean_removal_l741_741863

theorem jellybean_removal 
    (initial_count : ℕ) 
    (first_removal : ℕ) 
    (added_back : ℕ) 
    (final_count : ℕ)
    (initial_count_eq : initial_count = 37)
    (first_removal_eq : first_removal = 15)
    (added_back_eq : added_back = 5)
    (final_count_eq : final_count = 23) :
    (initial_count - first_removal + added_back - final_count) = 4 :=
by 
    sorry

end jellybean_removal_l741_741863


namespace probability_A_C_winning_l741_741613

-- Definitions based on the conditions given
def students := ["A", "B", "C", "D"]

def isDistictPositions (x y : String) : Prop :=
  x ≠ y

-- Lean statement for the mathematical problem
theorem probability_A_C_winning :
  ∃ (P : ℚ), P = 1/6 :=
by
  sorry

end probability_A_C_winning_l741_741613


namespace integer_values_satisfying_abs_lt_2pi_l741_741685

theorem integer_values_satisfying_abs_lt_2pi : 
  (finset.Icc (-6 : ℤ) 6).card = 13 :=
by
  sorry

end integer_values_satisfying_abs_lt_2pi_l741_741685


namespace tangent_slope_angle_expression_l741_741276

open Real

noncomputable def f (x : ℝ) : ℝ := (2 / 3) * x^3

theorem tangent_slope_angle_expression :
  let α := atan (2 : ℝ)
  in  (sin α ^ 2 - cos α ^ 2) / (2 * sin α * cos α + cos α ^ 2) = 3 / 5 :=
by
  sorry

end tangent_slope_angle_expression_l741_741276


namespace example_one_example_two_l741_741887

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l741_741887


namespace matrix_inverse_correct_l741_741583

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, -2], ![5, 3]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3/22, 1/11], ![-5/22, 2/11]]

theorem matrix_inverse_correct : A⁻¹ = A_inv :=
  by
    sorry

end matrix_inverse_correct_l741_741583


namespace smallest_number_with_55_divisors_l741_741600

theorem smallest_number_with_55_divisors : ∃ (n : ℕ), (∃ (p : ℕ → ℕ) (k : ℕ → ℕ) (m : ℕ), 
  n = ∏ i in finset.range m, (p i)^(k i) ∧ (∀ i j, i ≠ j → nat.prime (p i) → nat.prime (p j) → p i ≠ p j) ∧ 
  (finset.range m).card = m ∧ 
  (∏ i in finset.range m, (k i + 1) = 55)) ∧ 
  n = 3^4 * 2^10 then n = 82944 :=
by
  sorry

end smallest_number_with_55_divisors_l741_741600


namespace donna_pizza_slices_l741_741204

theorem donna_pizza_slices :
  ∀ (total_slices : ℕ) (half_eaten_for_lunch : ℕ) (one_third_eaten_for_dinner : ℕ),
  total_slices = 12 →
  half_eaten_for_lunch = total_slices / 2 →
  one_third_eaten_for_dinner = half_eaten_for_lunch / 3 →
  (half_eaten_for_lunch - one_third_eaten_for_dinner) = 4 :=
by
  intros total_slices half_eaten_for_lunch one_third_eaten_for_dinner
  intros h1 h2 h3
  sorry

end donna_pizza_slices_l741_741204


namespace probability_same_group_l741_741458

/-- There are three interest groups. Each student has an equal chance of joining any group.
    Prove that the probability that two students join the same interest group is 1/3. -/
theorem probability_same_group (n : ℕ) (h : n = 3) :
  (1 / 3 : ℚ) = (3 / (n * n) : ℚ) :=
by
  rw h
  norm_num
  sorry

end probability_same_group_l741_741458


namespace problem1_problem2_problem3_l741_741619

noncomputable theory

-- Define the given function f
def f (x t : ℝ) : ℝ :=
  (sin (2*x - π/4))^2 - 2 * t * sin (2*x - π/4) + t^2 - 6 * t + 1

-- Define g(t) for different ranges of t
def g (t : ℝ) : ℝ :=
  if t < -1/2 then t^2 - 5 * t + 5/4
  else if -1/2 ≤ t ∧ t ≤ 1 then -6 * t + 1
  else t^2 - 8 * t + 2

-- Proof problem 1: f evaluated at x = π/8 and t = 1 equals -4
theorem problem1 : f (π/8) 1 = -4 :=
sorry

-- Proof problem 2: Define g(t) as given
theorem problem2 (t : ℝ) : g t = 
  if t < -1/2 then t^2 - 5 * t + 5/4
  else if -1/2 ≤ t ∧ t ≤ 1 then -6 * t + 1
  else t^2 - 8 * t + 2 :=
sorry

-- Proof problem 3: Range of k for g(t) = kt to have a real root for -1/2 ≤ t ≤ 1
theorem problem3 (k : ℝ) : 
  (∃ t : ℝ, -1/2 ≤ t ∧ t ≤ 1 ∧ g t = k * t) ↔ (k ≤ -8 ∨ k ≥ -5) :=
sorry

end problem1_problem2_problem3_l741_741619


namespace part_a_l741_741102

variables {a b c S α β γ : ℝ}

-- Defining conditions
def condition1 : Prop := S = (1/2) * a * b * (Real.sin γ)
def condition2 : Prop := S = (1/2) * b * c * (Real.sin α)
def condition3 : Prop := S = (1/2) * c * a * (Real.sin β)
def condition4 : Prop := (Real.sin α) * (Real.sin β) * (Real.sin γ) ≤ (Real.sqrt 3 / 2)^3

-- The theorem to prove
theorem part_a (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  S^3 ≤ (Real.sqrt 3 / 4)^3 * (a * b * c)^2 :=
sorry

end part_a_l741_741102


namespace bucket_full_weight_l741_741512

variables (x y p q : Real)

theorem bucket_full_weight (h1 : x + (1 / 4) * y = p)
                           (h2 : x + (3 / 4) * y = q) :
    x + y = 3 * q - p :=
by
  sorry

end bucket_full_weight_l741_741512


namespace minimum_rolls_for_duplicate_sum_l741_741055

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741055


namespace prop_3_prop_4_l741_741757

variable {Line Plane : Type}
variable (perp : Line → Line → Prop)
variable (perp_plane : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line)

-- Proposition (3)
theorem prop_3 (α β γ : Plane) (m : Line) :
  perp_plane α β →
  perp_plane α γ →
  intersect β γ = m →
  perp m (intersect α β) :=
sorry

-- Proposition (4)
theorem prop_4 (α β : Plane) (m n : Line) :
  perp m α →
  perp n β →
  perp m n →
  perp_plane α β :=
sorry

end prop_3_prop_4_l741_741757


namespace probability_units_digit_even_l741_741142

theorem probability_units_digit_even : 
  let num_digits := 5
  let total_digits := 9 - 0 + 1
  let even_digits := 5
  0 < num_digits ∧ num_digits == 5 ∧ total_digits == 10 ∧ even_digits == 5 ↔ (sorry : (even_digits : ℝ) / total_digits == (1 / 2))

end probability_units_digit_even_l741_741142


namespace number_of_purchasing_methods_l741_741125

theorem number_of_purchasing_methods : 
  (nat.choose 8 5) + ((nat.choose 8 4) * (nat.choose 3 2)) = 266 :=
by
  sorry

end number_of_purchasing_methods_l741_741125


namespace line_equation_l741_741578

theorem line_equation (p : ℝ × ℝ) (l : ℝ → ℝ → ℝ) 
  (h_point : p = (1, 2))
  (h_line : ∀ x y, l x y = 2 * x - y - 1) :
  ∃ c : ℝ, (∀ x y, (2 * x - y + c = 0) ↔ (x, y) = p) :=
by {
  let c := 0,
  use c,
  intros x y,
  split,
  {
    intro h,
    rw [h],
    simp,
    exact h_point,
  },
  {
    intro h,
    rw h,
    simp,
  }
}

#check line_equation

end line_equation_l741_741578


namespace minimum_throws_for_repeated_sum_l741_741047

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741047


namespace projection_vector_of_b_onto_a_l741_741711

theorem projection_vector_of_b_onto_a :
  let a := (-4 : ℝ, 3 : ℝ)
  let b := (5 : ℝ, 12 : ℝ)
  let dot_product := a.1 * b.1 + a.2 * b.2
  let a_magnitude_squared := (a.1 * a.1 + a.2 * a.2)
  let projection := (dot_product / a_magnitude_squared) * a
  projection = (-(64 / 25) : ℝ, 48 / 25 : ℝ) :=
by
  sorry

end projection_vector_of_b_onto_a_l741_741711


namespace determine_third_root_exists_l741_741954

variables (a b c d α β: ℝ)

-- Let P(x) be the third-degree polynomial P(x) = ax^3 + bx^2 + cx + d
def polynomial (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Conditions
axiom distinct_roots : α ≠ β
axiom roots_in_unit_interval (x : ℝ) : x ∈ {α, β} → 0 < x ∧ x < 1
axiom polynomial_properties (P : ℝ → ℝ) : P(α) = 0 ∧ P(β) = 0

-- Vieta's formulas
axiom vieta_sum_roots : α + β + γ = -b / a
axiom vieta_prod_two_roots : α * β + β * γ + γ * α = c / a
axiom vieta_prod_all_roots : α * β * γ = -d / a

-- Prove that it is always possible to determine the third root γ
theorem determine_third_root_exists (γ : ℝ) : 
  (polynomial α) = 0 ∧ (polynomial β) = 0 → 
  ∃ γ, γ ≠ α ∧ γ ≠ β ∧ 0 < γ ∧ γ < 1 ∧ (polynomial γ) = 0 :=
sorry

end determine_third_root_exists_l741_741954


namespace find_AM_l741_741467

theorem find_AM (M A B C : Point) (circle : Circle)
  (tangent_line : TangentTo circle M A) 
  (intersecting_line : Intersects circle M B C)
  (hBC : dist B C = 7) (hBM : dist B M = 9) :
  dist A M = 12 ∨ dist A M = 3 * Real.sqrt 2 := 
sorry

end find_AM_l741_741467


namespace problem_a_problem_b_l741_741500

-- Problem (a)
theorem problem_a (n : ℕ) (hn : n > 0) :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

-- Problem (b)
theorem problem_b (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (coprime_ab : Nat.gcd a b = 1)
  (coprime_ac_or_bc : Nat.gcd c a = 1 ∨ Nat.gcd c b = 1) :
  ∃ᶠ x : ℕ in Filter.atTop, ∃ (y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^a + y^b = z^c :=
sorry

end problem_a_problem_b_l741_741500


namespace solve_problem_l741_741655

variable (a b c x : ℝ)

-- Conditions
def condition1 : Prop := ∀ x, ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2

-- Statements to prove
def statementA : Prop := a < 0
def statementB : Prop := a + b + c > 0
def statementD : Prop := ∀ x, (cx^2 + bx + a < 0 ↔ (-1/3 < x ∧ x < 1/2))

theorem solve_problem (h1 : condition1)
  (h2 : statementA)
  (h3 : statementB)
  (h4 : statementD) : a < 0 ∧ a + b + c > 0 ∧ ∀ x, (cx^2 + bx + a < 0 ↔ (-1/3 < x ∧ x < 1/2)) :=
by
  sorry

end solve_problem_l741_741655


namespace smallest_nat_with_55_divisors_l741_741595

open BigOperators

theorem smallest_nat_with_55_divisors :
  ∃ (n : ℕ), 
    (∃ (f : ℕ → ℕ) (primes : Finset ℕ),
      (∀ p ∈ primes, Nat.Prime p) ∧ 
      (primes.Sum (λ p => p ^ (f p))) = n ∧
      ((primes.Sum (λ p => f p + 1)) = 55)) ∧
    (∀ m, 
      (∃ (f_m : ℕ → ℕ) (primes_m : Finset ℕ),
        (∀ p ∈ primes_m, Nat.Prime p) ∧ 
        (primes_m.Sum (λ p => p ^ (f_m p))) = m ∧
        ((primes_m.Sum (λ p => f_m p + 1)) = 55)) → 
      n ≤ m) ∧
  n = 3^4 * 2^10 := 
begin
  sorry
end

end smallest_nat_with_55_divisors_l741_741595


namespace ratio_yuan_david_l741_741490

-- Definitions
def yuan_age (david_age : ℕ) : ℕ := david_age + 7
def ratio (a b : ℕ) : ℚ := a / b

-- Conditions
variable (david_age : ℕ) (h_david : david_age = 7)

-- Proof Statement
theorem ratio_yuan_david : ratio (yuan_age david_age) david_age = 2 :=
by
  sorry

end ratio_yuan_david_l741_741490


namespace boys_girls_relationship_l741_741552

theorem boys_girls_relationship (b g : ℕ) (h1 : b > 0) (h2 : g > 2) (h3 : ∀ n : ℕ, n < b → (n + 1) + 2 ≤ g) (h4 : b + 2 = g) : b = g - 2 := 
by
  sorry

end boys_girls_relationship_l741_741552


namespace ratio_CQ_QA_l741_741348

noncomputable def triangle_ABC (A B C : Type) :=
  ∃ (AB AC BC D N Q : ℝ),
    AB = 13 ∧
    AC = 7 ∧
    (∃ (x : ℝ),
      BC = 20 * x ∧
      BD = 13 * x ∧
      DC = 7 * x) ∧
    N = midpoint A D ∧
    (∃ (CQ QA : ℝ),
      CQ / QA = 7 / 13 ∧
      CQ + QA = QA * (7/13 + 1))

theorem ratio_CQ_QA (A B C : Type) :
  triangle_ABC A B C →
  p = 7 →
  q = 13 →
  p + q = 20 := by
    sorry

end ratio_CQ_QA_l741_741348


namespace bank_tellers_total_coins_l741_741206

theorem bank_tellers_total_coins (tellers rolls_per_teller coins_per_roll : ℕ) (h1 : tellers = 4) (h2 : rolls_per_teller = 10) (h3 : coins_per_roll = 25) 
    : tellers * (rolls_per_teller * coins_per_roll) = 1000 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end bank_tellers_total_coins_l741_741206


namespace plane_intersect_interior_prob_zero_l741_741965

-- Define a regular tetrahedron with 4 vertices
structure RegularTetrahedron :=
  (vertices : Finset ℝ)
  (h_vertices : vertices.card = 4)

-- Defining the selection of 3 vertices randomly from the 4 vertices
def choose_3_vertices (tetrahedron : RegularTetrahedron) : Finset (Finset ℝ) :=
  (tetrahedron.vertices.powerset.filter (λ s, s.card = 3))

-- Define the probability of a plane determined by 3 vertices intersecting the interior
noncomputable def probability_intersect_interior (tetrahedron : RegularTetrahedron) : ℝ :=
  let planes := choose_3_vertices tetrahedron in
  if planes.nonempty then 0 else sorry

-- The main theorem to be proven
theorem plane_intersect_interior_prob_zero (tetrahedron : RegularTetrahedron) :
  probability_intersect_interior tetrahedron = 0 := 
sorry

end plane_intersect_interior_prob_zero_l741_741965


namespace find_a_value_l741_741839

theorem find_a_value:
  ∃ (a : ℝ), (∀ (x : ℝ), x^2 + a*x + 2 = 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^3 + (14 / x2^2) = x2^3 + (14 / x1^2)) → a = 4 :=
begin
  sorry
end

end find_a_value_l741_741839


namespace smallest_multipler_of_21_l741_741756

def g (m : ℕ) : ℕ :=
  if h : ∃ p : ℕ, (∀ q < p, factorial q < m) ∧ factorial p >= m
  then Classical.choose h
  else 0

theorem smallest_multipler_of_21 (m : ℕ) : m = 21 * 23 → g(m) = 23 := by
  sorry

end smallest_multipler_of_21_l741_741756


namespace symmetric_difference_cardinality_l741_741712

variable (x y : Set ℤ)
variable (n_x n_y n_xy : ℕ)

theorem symmetric_difference_cardinality
  (hx : x.card = 14)
  (hy : y.card = 18)
  (hxy : (x ∩ y).card = 6) :
  (x.symmDiff y).card = 20 :=
sorry

end symmetric_difference_cardinality_l741_741712


namespace part1_part2_l741_741616

open Set

variable (U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3})
variable (A : Set ℤ := {1, 2, 3})
variable (B : Set ℤ := {-1, 0, 1})
variable (C : Set ℤ := {-2, 0, 2})

theorem part1 : A ∪ (B ∩ C) = {0, 1, 2, 3} := by
  sorry

theorem part2 : A ∩ Uᶜ ∪ (B ∪ C) = {3} := by
  sorry

end part1_part2_l741_741616


namespace problem_statement_l741_741309

theorem problem_statement (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_statement_l741_741309


namespace min_throws_to_ensure_same_sum_twice_l741_741066

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741066


namespace cone_base_radius_l741_741951

theorem cone_base_radius 
  (central_angle : ℝ)
  (sector_radius : ℝ)
  (circumference : ℝ := 2 * Real.pi * sector_radius * (central_angle / 360)) :
  central_angle = 200 → sector_radius = 2 → circumference = 2 * Real.pi * (10 / 9) → 
  ∃ R : ℝ, R = 10 / 9 :=
by
  intro h1 h2 h3
  use 10 / 9
  apply h3
sorry

end cone_base_radius_l741_741951


namespace sum_of_first_15_terms_l741_741106

-- Define the conditions and the target property.
theorem sum_of_first_15_terms (a d : ℝ) (h : (a + 3 * d) + (a + 11 * d) = 8) :
  let S_15 := 15 / 2 * (2 * a + 14 * d)
  in S_15 = 60 :=
by
  -- The calculations will be filled in Lean 4 proof later
  sorry

-- Prove the required statement
example (a d : ℝ) (h : (a + 3 * d) + (a + 11 * d) = 8) :
  let S_15 := 15 / 2 * (2 * a + 14 * d)
  in S_15 = 60 :=
by
  -- Need to use the theorem defined above
  exact sum_of_first_15_terms a d h

end sum_of_first_15_terms_l741_741106


namespace pi_mode_correct_l741_741491

theorem pi_mode_correct:
  let freq := [8, 8, 12, 11, 10, 8, 9, 8, 12, 14] in
  ∃! d, d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] ∧ 
          freq.nth d = some 14 ∧ 
          ∀ j, j ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → j ≠ d → freq.nth j < some 14 :=
by
  sorry

end pi_mode_correct_l741_741491


namespace insurance_compensation_l741_741115

/-- Given the actual damage amount and the deductible percentage, 
we can compute the amount of insurance compensation. -/
theorem insurance_compensation : 
  ∀ (damage_amount : ℕ) (deductible_percent : ℕ), 
  damage_amount = 300000 → 
  deductible_percent = 1 →
  (damage_amount - (damage_amount * deductible_percent / 100)) = 297000 :=
by
  intros damage_amount deductible_percent h_damage h_deductible
  sorry

end insurance_compensation_l741_741115


namespace minimum_throws_to_ensure_same_sum_twice_l741_741018

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741018


namespace general_term_a_general_term_b_sum_first_n_terms_l741_741675

def a : Nat → Nat
| 0     => 1
| (n+1) => 2 * a n

def b (n : Nat) : Int :=
  3 * (n + 1) - 2

def S (n : Nat) : Int :=
  2^n - (3 * n^2) / 2 + n / 2 - 1

-- We state the theorems with the conditions included.

theorem general_term_a (n : Nat) : a n = 2^(n - 1) := by
  sorry

theorem general_term_b (n : Nat) : b n = 3 * (n + 1) - 2 := by
  sorry

theorem sum_first_n_terms (n : Nat) : 
  (Finset.range n).sum (λ i => a i - b i) = 2^n - (3 * n^2) / 2 + n / 2 - 1 := by
  sorry

end general_term_a_general_term_b_sum_first_n_terms_l741_741675


namespace region_area_calculation_value_of_abc_l741_741435

noncomputable def region_area (radius angle : ℝ) : ℝ :=
  let side_length := 2 * radius * real.sin (angle / 2)
  let triangle_area := 0.5 * side_length * radius
  let octagon_area := 8 * triangle_area
  let circle_area := real.pi * radius ^ 2
  octagon_area - circle_area

theorem region_area_calculation :
  let radius := 5
  let angle := real.pi / 4
  let area := 100 * real.sqrt 2 - 25 * real.pi
  region_area radius angle = area :=
sorry

theorem value_of_abc :
  100 + 2 - 25 = 77 :=
by norm_num

end region_area_calculation_value_of_abc_l741_741435


namespace germination_rate_prob_correct_l741_741440

noncomputable def germination_probability (n : ℕ) (p : ℚ) (k1 k2 : ℕ) : ℚ :=
  let q := 1 - p
  let μ := n * p
  let σ := real.sqrt (n * p * q)
  let x1 := (k1 - μ) / σ
  let x2 := (k2 - μ) / σ
  real.exp (-0.5 * x1^2) / (σ * real.sqrt (2 * real.pi)) - real.exp (-0.5 * x2^2) / (σ * real.sqrt (2 * real.pi))

theorem germination_rate_prob_correct :
  germination_probability 900 0.9 790 830 ≈ 0.9736 := sorry

end germination_rate_prob_correct_l741_741440


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741895

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741895


namespace simplify_and_evaluate_expression_l741_741419

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.pi^0 + 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (2 * x + 2)) = 2 / 3 := by
  sorry

end simplify_and_evaluate_expression_l741_741419


namespace six_integers_satisfy_inequalities_l741_741231

theorem six_integers_satisfy_inequalities :
  let S := {n : ℤ | 2 ≤ n ∧ n ≤ 7} in
  ∀ n ∈ S, (sqrt n : ℝ) ≤ sqrt (5 * ↑n - 8) ∧ sqrt (5 * ↑n - 8) < sqrt (3 * ↑n + 7) :=
by {
  sorry
}

end six_integers_satisfy_inequalities_l741_741231


namespace total_attendance_l741_741465

theorem total_attendance (T_before T_door : ℕ) 
  (h1 : T_before = 475) 
  (h2 : 2 * T_before + 275 * T_door = 1706.25) : 
  T_before + T_door = 750 := 
sorry

end total_attendance_l741_741465


namespace product_of_roots_of_cubic_l741_741561

theorem product_of_roots_of_cubic :
  let p : Polynomial ℝ := Polynomial.C (-35) + Polynomial.X * (Polynomial.C 27 + Polynomial.X * (Polynomial.C (-9) + Polynomial.X)) in
  (p.coeff 0 = -35) →
  (p.coeff 1 = 27) →
  (p.coeff 2 = -9) →
  (p.coeff 3 = 1) →
  (Polynomial.prodRoots p) = 35 :=
by
  sorry

end product_of_roots_of_cubic_l741_741561


namespace yellow_candies_eaten_prob_before_red_find_m_n_sum_l741_741958

open Nat

theorem yellow_candies_eaten_prob_before_red
  (yellow red blue : ℕ) (h₁ : yellow = 2) (h₂ : red = 4) (h₃ : blue = 6) :
  let total := yellow + red in
  let total_arrangements := Nat.choose total yellow in
  let favorable_arrangements := 1 in
  let prob := favorable_arrangements / total_arrangements in
  prob = 1 / 15 :=
by
  sorry

theorem find_m_n_sum (m n : ℕ) (h₁ : m = 1) (h₂ : n = 15) : m + n = 16 :=
by
  rw [h₁, h₂]
  rfl

end yellow_candies_eaten_prob_before_red_find_m_n_sum_l741_741958


namespace shared_property_l741_741092

-- Definitions of the shapes with their properties
structure Parallelogram (α : Type) [EuclideanSpace α] :=
(opposite_sides_parallel_equal : ∀ {a b : α}, parallel a b ∧ equal_length a b)

structure Rectangle (α : Type) [EuclideanSpace α] extends Parallelogram α :=
(all_right_angles : ∀ a : α, right_angle a)
(equal_length_diagonals : ∀ {a b : α}, equal_length a b)

structure Rhombus (α : Type) [EuclideanSpace α] :=
(all_sides_equal : ∀ a b : α, equal_length a b)
(opposite_sides_parallel : ∀ a b : α, parallel a b)
(perpendicular_diagonals : ∀ {a b : α}, perpendicular a b)

structure Square (α : Type) [EuclideanSpace α] extends Rectangle α, Rhombus α

-- The theorem to be proven
theorem shared_property (α : Type) [EuclideanSpace α] : 
  ∀ (P : Parallelogram α) (R : Rectangle α) (H : Rhombus α) (S : Square α), 
    (P.opposite_sides_parallel_equal ∧ R.opposite_sides_parallel_equal ∧ H.opposite_sides_parallel ∧ S.opposite_sides_parallel) :=
begin
  sorry
end

end shared_property_l741_741092


namespace periodic_sequence_l741_741563

noncomputable def sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = (a n - 1) / (a n + 1)

theorem periodic_sequence (a : ℕ → ℚ) (h1 : a 1 ≠ -1) (h2 : a 1 ≠ 0) (h3 : a 1 ≠ 1) :
  sequence a →
  a 5 = a 1 :=
by
  sorry

end periodic_sequence_l741_741563


namespace operations_even_when_a_even_l741_741256

theorem operations_even_when_a_even (a k : ℤ) (h : a = 2 * k) : 
  (∃ b1, a^2 = 2 * b1) ∧
  (∃ b2, 2 * a = 2 * b2) ∧
  (∃ b3, (a % 4 = 0) ∨ (a % 4 = 2)) ∧
  (∃ b5, a^3 = 2 * b5) :=
by
  sorry

end operations_even_when_a_even_l741_741256


namespace difference_between_p_and_s_l741_741984

noncomputable def p := 2 * q
noncomputable def q := r
noncomputable def r := (5 / 36) * 25000
noncomputable def s := 4 * r
noncomputable def t := s / 2
noncomputable def total := 25000

theorem difference_between_p_and_s :
  6944.4444 = s - p := by
  sorry

end difference_between_p_and_s_l741_741984


namespace non_participating_members_l741_741327

noncomputable def members := 35
noncomputable def badminton_players := 15
noncomputable def tennis_players := 18
noncomputable def both_players := 3

theorem non_participating_members : 
  members - (badminton_players + tennis_players - both_players) = 5 := by
  sorry

end non_participating_members_l741_741327


namespace solve_complex_eqn_l741_741198

def complex_eqn (z : ℂ) : Prop :=
  z - 1 = (z + 1) * complex.I

theorem solve_complex_eqn :
  ∃ z : ℂ, complex_eqn z ∧ z = complex.I :=
by {
  sorry
}

end solve_complex_eqn_l741_741198


namespace arithmetic_sequence_length_l741_741300

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ (a d l : ℕ), a = 2 → d = 5 → l = 3007 → l = a + (n-1) * d → n = 602 :=
by
  sorry

end arithmetic_sequence_length_l741_741300


namespace seven_digit_numbers_with_at_least_one_zero_l741_741683

def total_7_digit_numbers : ℕ := 9 * 10^6

def no_zero_7_digit_numbers : ℕ := 9^7

def at_least_one_zero_7_digit_numbers : ℕ :=
  total_7_digit_numbers - no_zero_7_digit_numbers

theorem seven_digit_numbers_with_at_least_one_zero :
  at_least_one_zero_7_digit_numbers = 8_521_704 := by
  sorry

end seven_digit_numbers_with_at_least_one_zero_l741_741683


namespace arithmetic_sequence_length_l741_741297

theorem arithmetic_sequence_length :
  ∃ n, (2 + (n - 1) * 5 = 3007) ∧ n = 602 :=
by
  use 602
  sorry

end arithmetic_sequence_length_l741_741297


namespace order_of_abc_l741_741694

noncomputable def a : ℝ := (0.3)^3
noncomputable def b : ℝ := (3)^3
noncomputable def c : ℝ := Real.log 0.3 / Real.log 3

theorem order_of_abc : b > a ∧ a > c :=
by
  have ha : a = (0.3)^3 := rfl
  have hb : b = (3)^3 := rfl
  have hc : c = Real.log 0.3 / Real.log 3 := rfl
  sorry

end order_of_abc_l741_741694


namespace find_square_sum_l741_741764

theorem find_square_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 2 / 7 :=
by
  sorry

end find_square_sum_l741_741764


namespace find_z_l741_741733

/- Definitions of angles and their relationships -/
def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

/- Given conditions -/
def ABC : ℝ := 75
def BAC : ℝ := 55
def BCA : ℝ := 180 - ABC - BAC  -- This follows from the angle sum property of triangle ABC
def DCE : ℝ := BCA
def CDE : ℝ := 90

/- Prove z given the above conditions -/
theorem find_z : ∃ (z : ℝ), z = 90 - DCE := by
  use 40
  sorry

end find_z_l741_741733


namespace investment_Q_correct_l741_741403

-- Define the investments of P and Q
def investment_P : ℝ := 40000
def investment_Q : ℝ := 60000

-- Define the profit share ratio
def profit_ratio_PQ : ℝ × ℝ := (2, 3)

-- State the theorem to prove
theorem investment_Q_correct :
  (investment_P / investment_Q = (profit_ratio_PQ.1 / profit_ratio_PQ.2)) → 
  investment_Q = 60000 := 
by 
  sorry

end investment_Q_correct_l741_741403


namespace integer_roots_of_polynomial_l741_741219

def polynomial : Polynomial ℤ := Polynomial.C 24 + Polynomial.C (-11) * Polynomial.X + Polynomial.C (-4) * Polynomial.X^2 + Polynomial.X^3

theorem integer_roots_of_polynomial :
  Set { x : ℤ | polynomial.eval x polynomial = 0 } = Set ({-4, 3, 8} : Set ℤ) :=
sorry

end integer_roots_of_polynomial_l741_741219


namespace sum_reciprocal_inequality_l741_741621

theorem sum_reciprocal_inequality (p q a b c d e : ℝ) (hp : 0 < p) (ha : p ≤ a) (hb : p ≤ b) (hc : p ≤ c) (hd : p ≤ d) (he : p ≤ e) (haq : a ≤ q) (hbq : b ≤ q) (hcq : c ≤ q) (hdq : d ≤ q) (heq : e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e) ≤ 25 + 6 * ((Real.sqrt (q / p) - Real.sqrt (p / q)) ^ 2) :=
by sorry

end sum_reciprocal_inequality_l741_741621


namespace pentagon_area_l741_741992

theorem pentagon_area 
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ) (side5 : ℝ)
  (h1 : side1 = 12) (h2 : side2 = 20) (h3 : side3 = 30) (h4 : side4 = 15) (h5 : side5 = 25)
  (right_angle : ∃ (a b : ℝ), a = side1 ∧ b = side5 ∧ a^2 + b^2 = (a + b)^2) : 
  ∃ (area : ℝ), area = 600 := 
  sorry

end pentagon_area_l741_741992


namespace min_rolls_to_duplicate_sum_for_four_dice_l741_741002

theorem min_rolls_to_duplicate_sum_for_four_dice : 
    let min_sum := 4 * 1,
    let max_sum := 4 * 6,
    let possible_sums := max_sum - min_sum + 1 in
    possible_sums = 21 → 
    (possible_sums + 1 = 22) := 
by
  intros min_sum max_sum possible_sums h
  have h1 : min_sum = 4 := rfl
  have h2 : max_sum = 24 := rfl
  have h3 : possible_sums = 21 := h
  have h4 : possible_sums + 1 = 22 := calc
    possible_sums + 1 = 21 + 1 : by rw h
    ... = 22 : by rfl
  exact h4

end min_rolls_to_duplicate_sum_for_four_dice_l741_741002


namespace find_first_two_solutions_l741_741246

theorem find_first_two_solutions :
  ∃ (n1 n2 : ℕ), 
    (n1 ≡ 3 [MOD 7]) ∧ (n1 ≡ 4 [MOD 9]) ∧ 
    (n2 ≡ 3 [MOD 7]) ∧ (n2 ≡ 4 [MOD 9]) ∧ 
    n1 < n2 ∧ 
    n1 = 31 ∧ n2 = 94 := 
by 
  sorry

end find_first_two_solutions_l741_741246


namespace problem_1_problem_2_l741_741395

open Set

def universal_set := ℤ

def A : Set ℤ := {x | x^2 + 2 * x - 15 = 0}
def B (a : ℚ) : Set ℤ := {x | a * x - 1 = 0}

theorem problem_1 :
  A ∩ (universal_set \ B (1 / 5)) = {-5, 3} := sorry

theorem problem_2 :
  {a : ℚ | B a ⊆ A} = {-1 / 5, 1 / 3, 0} := sorry

end problem_1_problem_2_l741_741395


namespace EFGH_is_parallelogram_l741_741771

-- Define the problem
variables {A B C D E F G H : Point}
variables (h1 : is_convex A B C D)
variables (h2 : is_equilateral A B E)
variables (h3 : is_equilateral B C F)
variables (h4 : is_equilateral C D G)
variables (h5 : is_equilateral D A H)
variables (o1 : oriented_outward A B E)
variables (o2 : oriented_outward C D G)
variables (o3 : oriented_inward B C F)
variables (o4 : oriented_inward D A H)

theorem EFGH_is_parallelogram :
  is_parallelogram E F G H :=
sorry

end EFGH_is_parallelogram_l741_741771


namespace cos6_plus_sin6_equal_19_div_64_l741_741381

noncomputable def cos6_plus_sin6 (θ : ℝ) : ℝ :=
  (Real.cos θ) ^ 6 + (Real.sin θ) ^ 6

theorem cos6_plus_sin6_equal_19_div_64 (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) :
  cos6_plus_sin6 θ = 19 / 64 := by
  sorry

end cos6_plus_sin6_equal_19_div_64_l741_741381


namespace minimum_throws_to_ensure_same_sum_twice_l741_741007

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741007


namespace problem_NCMO_12th_l741_741379

noncomputable def floor := Int.floor
noncomputable def sqrt := Real.sqrt
def gcd := Nat.gcd

theorem problem_NCMO_12th :
    (∃^∞ n : ℕ+, gcd n (floor (sqrt 2 * n)) = 1) ∧
    (∃^∞ n : ℕ+, gcd n (floor (sqrt 2 * n)) ≠ 1) := 
sorry

end problem_NCMO_12th_l741_741379


namespace other_employee_number_l741_741977

-- Define the conditions
variables (total_employees : ℕ) (sample_size : ℕ) (e1 e2 e3 : ℕ)

-- Define the systematic sampling interval
def sampling_interval (total : ℕ) (size : ℕ) : ℕ := total / size

-- The Lean statement for the proof problem
theorem other_employee_number
  (h1 : total_employees = 52)
  (h2 : sample_size = 4)
  (h3 : e1 = 6)
  (h4 : e2 = 32)
  (h5 : e3 = 45) :
  ∃ e4 : ℕ, e4 = 19 := 
sorry

end other_employee_number_l741_741977


namespace painter_total_earnings_l741_741533

-- Representing conditions
def south_addresses (n : ℕ) : ℕ := 5 + 6 * (n - 1)
def north_addresses (n : ℕ) : ℕ := 6 + 6 * (n - 1)

-- Definition of cost to paint nth house number based on number of digits
def digits (n : ℕ) : ℕ := if n < 10 then 1 else if n < 100 then 2 else 3
def cost_per_house (addr_seq : ℕ → ℕ) (n : ℕ) : ℕ := digits (addr_seq n)

-- Total houses
def total_houses := 30

-- Cost calculation function
def total_cost : ℕ := (finset.range total_houses).sum (cost_per_house south_addresses) +
                      (finset.range total_houses).sum (cost_per_house north_addresses)

theorem painter_total_earnings : total_cost = 84 := by
  sorry

end painter_total_earnings_l741_741533


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741893

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741893


namespace swim_back_distance_l741_741150

variables (swimming_speed_still_water : ℝ) (water_speed : ℝ) (time_back : ℝ) (distance_back : ℝ)

theorem swim_back_distance :
  swimming_speed_still_water = 12 → 
  water_speed = 10 → 
  time_back = 4 →
  distance_back = (swimming_speed_still_water - water_speed) * time_back →
  distance_back = 8 :=
by
  intros swimming_speed_still_water_eq water_speed_eq time_back_eq distance_back_eq
  have swim_speed : (swimming_speed_still_water - water_speed) = 2 := by sorry
  rw [swim_speed, time_back_eq] at distance_back_eq
  sorry

end swim_back_distance_l741_741150


namespace NumberOfCorrectPropositions_l741_741728

-- Definitions of the conditions
def Prop1 (a b c : Line) : Prop := (angle a c = angle b c) → (a ∥ b)
def Prop2 (a b : Line) (α : Plane) : Prop := (angle a α = angle b α) → (a ∥ b)
def Prop3 (a : Line) (α : Plane) : Prop := 
  (∀ (p1 p2 : Point), p1 ∈ a → p2 ∈ a → distance p1 α = distance p2 α) → (a ∥ α)
def Prop4 (β α : Plane) : Prop :=
  (∀ (p1 p2 p3 : Point), p1 ∈ β → p2 ∈ β → p3 ∈ β → ¬Collinear p1 p2 p3 → 
  distance p1 α = distance p2 α ∧ distance p2 α = distance p3 α) → (α ∥ β)

-- Proving the number of correct propositions is 3
theorem NumberOfCorrectPropositions : (¬ Prop1 ∧ Prop2 ∧ Prop3 ∧ Prop4) → 3 := by
  intros h
  sorry

end NumberOfCorrectPropositions_l741_741728


namespace largest_number_is_n2_l741_741487

noncomputable def n1 : ℝ := 8.23455
noncomputable def n2 : ℝ := 8.23455 + 5 / 10^6 + 5 / 10^7 + 5 / 10^8 -- corresponding to 8.2345555...
noncomputable def n3 : ℝ := 8.23454 + 54 / 10^7 + 54 / 10^9 -- corresponding to 8.234545454...
noncomputable def n4 : ℝ := 8.23453 + 45 / 10^6 + 4 / 10^8 -- corresponding to 8.234534534...
noncomputable def n5 : ℝ := 8.2345  + 2345 / 10^7 + 2345 / 10^11 -- corresponding to 8.234523452...

theorem largest_number_is_n2 : ∃ m : ℝ, m = n2 ∧ m > n1 ∧ m > n3 ∧ m > n4 ∧ m > n5 := 
by {
  use n2,
  split,
  refl,
  split,
  -- n2 > n1
  sorry,
  split,
  -- n2 > n3
  sorry,
  split,
  -- n2 > n4
  sorry,
  -- n2 > n5
  sorry
}

end largest_number_is_n2_l741_741487


namespace find_smallest_x_l741_741224

noncomputable def smallest_pos_real_x : ℝ :=
  55 / 7

theorem find_smallest_x (x : ℝ) (h : x > 0) (hx : ⌊x^2⌋ - x * ⌊x⌋ = 6) : x = smallest_pos_real_x :=
  sorry

end find_smallest_x_l741_741224


namespace ball_hits_ground_l741_741989

-- Define the height equation and conditions
def height (t : ℝ) : ℝ := -16 * t^2 - 20 * t + 200

-- Define the problem of determining when the ball hits the ground
theorem ball_hits_ground : ∃ t : ℝ, height t = 0 ∧ t > 0 ∧ t ≈ 3.144 := by
  /- We need to solve the equation height(t) = 0 and find the positive root approximately equal to 3.144.
     This theorem states that such a t exists. -/
  sorry

end ball_hits_ground_l741_741989


namespace complex_in_third_quadrant_l741_741659

open Complex

theorem complex_in_third_quadrant (a : ℝ) (z : ℂ) (h₁ : z = (2 + complex.I * a) * (a - complex.I))
  (h₂ : z.re < 0) (h₃ : z.im < 0) : -real.sqrt 2 < a ∧ a < 0 :=
by
  sorry

end complex_in_third_quadrant_l741_741659


namespace abc_inequality_l741_741575

theorem abc_inequality {n : ℕ} : 
  (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a + b + c = 1 → 
  abc_inequality := (abc * (a^n + b^n + c^n) ≤ 1 / 3^(n + 2))) ↔ (n = 1 ∨ n = 2) :=
sorry

end abc_inequality_l741_741575


namespace no_real_solution_for_t_l741_741377

open Real

theorem no_real_solution_for_t :
  ∀ t : ℝ, let P := (t - 5, -2) in
           let Q := (-3, t + 4) in
           let midpoint := ((t - 5 - 3) / 2, (-2 + (t + 4)) / 2) in
           ¬ ((dist midpoint P / 2)^2 = t^2 + t - 1) :=
by
  sorry

end no_real_solution_for_t_l741_741377


namespace correct_operation_l741_741926

theorem correct_operation (a b : ℝ) :
  (2 * a^2)^3 ≠ 6 * a^6 ∧ 
  a^8 / a^2 ≠ a^4 ∧ 
  a^3 * a^4 = a^7 ∧ 
  5 * a + 2 * b ≠ 7 * a * b :=
by
  repeat { split }
  sorry
  sorry
  sorry
  sorry

end correct_operation_l741_741926


namespace parabola_vertex_l741_741835

theorem parabola_vertex : ∀ x : ℝ, ∃ h k : ℝ, y = 2 * (x - 3)^2 - 7 ∧ h = 3 ∧ k = -7 :=
by
  -- Define the equation of the parabola
  let y : ℝ := 2 * (x - 3)^2 - 7
  -- State the conditions for the vertex form
  use [3, -7]
  -- Provide the coordinate pair for the vertex
  exact ⟨rfl, rfl⟩
  sorry


end parabola_vertex_l741_741835


namespace student_arrangements_l741_741724

/--
In how many different ways can four students stand in a straight line if two of the
students refuse to stand next to each other?
-/
theorem student_arrangements
  (students : Fin 4 → Type)
  (student1 student2 : Fin 4)
  (h : student1 ≠ student2) :
  let arrangements := {
    l : List (Fin 4) // l.nodup ∧ ¬(l.indexOf student1 + 1 = l.indexOf student2 ∨ l.indexOf student1 - 1 = l.indexOf student2)
  } in
  Fintype.card arrangements = 12 :=
by sorry

end student_arrangements_l741_741724


namespace functions_are_equal_l741_741980

noncomputable def f : ℝ → ℝ := λ x, |x + 1|

noncomputable def g : ℝ → ℝ := λ x, if x ≥ -1 then x + 1 else -1 - x

theorem functions_are_equal : ∀ x : ℝ, f x = g x := by
  sorry

end functions_are_equal_l741_741980


namespace number_is_multiple_of_15_l741_741342

theorem number_is_multiple_of_15
  (W X Y Z D : ℤ)
  (h1 : X - W = 1)
  (h2 : Y - W = 9)
  (h3 : Y - X = 8)
  (h4 : Z - W = 11)
  (h5 : Z - X = 10)
  (h6 : Z - Y = 2)
  (hD : D - X = 5) :
  15 ∣ D :=
by
  sorry -- Proof goes here

end number_is_multiple_of_15_l741_741342


namespace sum_of_any_three_on_line_is_30_l741_741405

/-- Define the list of numbers from 1 to 19 -/
def numbers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- Define the specific sequence found in the solution -/
def arrangement :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 18,
   17, 16, 15, 14, 13, 12, 11]

/-- Define the function to compute the sum of any three numbers on a straight line -/
def sum_on_line (a b c : ℕ) := a + b + c

theorem sum_of_any_three_on_line_is_30 :
  ∀ i j k : ℕ, 
  i ∈ numbers ∧ j ∈ numbers ∧ k ∈ numbers ∧ (i = 10 ∨ j = 10 ∨ k = 10) →
  sum_on_line i j k = 30 :=
by
  sorry

end sum_of_any_three_on_line_is_30_l741_741405


namespace midpoints_of_chords_lie_on_circle_l741_741819

theorem midpoints_of_chords_lie_on_circle {O X : Point} {r : ℝ} (h1 : circle O r) (h2 : ∀ (A B : Point), chord_passes_through_point A B X):
  ∃ (C : Point) (R : ℝ), ∀ (A B : Point), midpoint_of_chord A B = C ∧ lies_on_circle C R  :=
sorry

end midpoints_of_chords_lie_on_circle_l741_741819


namespace minimum_rolls_for_duplicate_sum_l741_741064

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741064


namespace sqrt_square_simplify_l741_741824

noncomputable def pi : ℝ := Real.pi

theorem sqrt_square_simplify : sqrt ((3 - pi) ^ 2) = pi - 3 :=
by
  sorry

end sqrt_square_simplify_l741_741824


namespace Grandfather_age_correct_l741_741094

-- Definitions based on the conditions
def Yuna_age : Nat := 9
def Father_age (Yuna_age : Nat) : Nat := Yuna_age + 27
def Grandfather_age (Father_age : Nat) : Nat := Father_age + 23

-- The theorem stating the problem to prove
theorem Grandfather_age_correct : Grandfather_age (Father_age Yuna_age) = 59 := by
  sorry

end Grandfather_age_correct_l741_741094


namespace binary_to_octal_l741_741565

theorem binary_to_octal (b : ℕ) (h : b = 11010) : 
  nat.to_digits 8 (nat.of_digits 2 [1,1,0,1,0]) = [3,2] := 
by sorry

end binary_to_octal_l741_741565


namespace a_share_of_profit_correct_l741_741722

noncomputable def totalCapital : ℝ := sorry
noncomputable def totalTime : ℝ := sorry
def totalProfit : ℝ := 2300

def investmentRatioA : ℝ := (1 / 6) * (1 / 6)
def investmentRatioB : ℝ := (1 / 3) * (1 / 3)
def investmentRatioC : ℝ := (1 / 2) * 1
def totalInvestmentRatio : ℝ := investmentRatioA + investmentRatioB + investmentRatioC

def aShareOfProfit : ℝ := (investmentRatioA / totalInvestmentRatio) * totalProfit

theorem a_share_of_profit_correct : aShareOfProfit = 100 := 
by
  sorry

end a_share_of_profit_correct_l741_741722


namespace orthocenter_of_triangle_l741_741331

noncomputable theory

open_locale classical

-- Define the points A, B, and C in 3D space
def A : ℝ × ℝ × ℝ := (2, 3, 4)
def B : ℝ × ℝ × ℝ := (6, 4, 2)
def C : ℝ × ℝ × ℝ := (4, 5, 6)

-- Define the orthocenter H we want to prove
def H : ℝ × ℝ × ℝ := (1/2, 8, 1/2)

-- The statement that H is the orthocenter of triangle ABC
theorem orthocenter_of_triangle :
  ∃ H : ℝ × ℝ × ℝ, H = (1/2, 8, 1/2) := sorry

end orthocenter_of_triangle_l741_741331


namespace sum_divisible_by_17_l741_741222

theorem sum_divisible_by_17 :
    (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92 + 93 + 94) % 17 = 0 := 
by 
  sorry

end sum_divisible_by_17_l741_741222


namespace rain_at_least_one_day_l741_741454

noncomputable def prob_rain_saturday : ℚ := 0.3
noncomputable def prob_rain_sunday : ℚ := 0.6
noncomputable def cond_prob_rain_sunday_given_rain_saturday : ℚ := 0.8

theorem rain_at_least_one_day :
  let no_rain_saturday := 1 - prob_rain_saturday,
      no_rain_sunday := 1 - prob_rain_sunday,
      no_rain_saturday_and_sunday :=
        no_rain_saturday * no_rain_sunday,
      no_rain_sunday_given_rain_saturday :=
        1 - cond_prob_rain_sunday_given_rain_saturday,
      no_rain_given_rain_saturday :=
        prob_rain_saturday * no_rain_sunday_given_rain_saturday,
      total_no_rain := no_rain_saturday_and_sunday + no_rain_given_rain_saturday,
      prob_rain_at_least_one_day := 1 - total_no_rain
  in prob_rain_at_least_one_day = 0.66 :=
by
  sorry

end rain_at_least_one_day_l741_741454


namespace length_FI_l741_741869

-- Given conditions for the triangle DEF with side lengths DE, EF, and FD
def triangle_DEF (DE EF FD : ℝ) (D E F : Type) : Prop :=
  DE = 13 ∧ EF = 30 ∧ FD = 23

-- Point G is the intersection of the angle bisector of ∠DEF with EF
def angle_bisector_DEF_intersects_EF (G D E F : Type) : Prop :=
  G ∈ LineSegment E F ∧
  -- Additional properties to specify that G is the bisector needs to be formalized

-- Point H is the intersection of the angle bisector of ∠DEF with the circumcircle of △DEF
def angle_bisector_DEF_intersects_circumcircle (H D E F : Type) : Prop :=
  H ≠ D ∧
  -- Additional properties to specify that H is the intersection with the circumcircle

-- The circumcircle of △DGH intersects line DE at points D and I ≠ D
def circumcircle_DGH_intersects_DE (D G H E I : Type) : Prop :=
  -- Formalize circumcircle intersection condition here

-- Question: Find the length FI
def find_FI (D E F G H I : Type) : ℝ :=
  -- Assume the appropriate geometric properties are formalized here
  let DE : ℝ := 13
  let EF : ℝ := 30
  let FD : ℝ := 23

  -- Placeholder for the actual computation
  -- Using given conditions and geometric properties to find FI
  
  21  -- Assuming the problem's simplified evaluation step results in 21

theorem length_FI {D E F G H I : Type} :
  triangle_DEF 13 30 23 D E F →
  angle_bisector_DEF_intersects_EF G D E F →
  angle_bisector_DEF_intersects_circumcircle H D E F →
  circumcircle_DGH_intersects_DE D G H E I →
  ∃ FI : ℝ, FI = 21 :=
by
  -- The proof is required here. For now, it ends with sorry.

sorry

end length_FI_l741_741869


namespace remainder_of_3042_div_98_l741_741477

theorem remainder_of_3042_div_98 : 3042 % 98 = 4 := 
by
  sorry

end remainder_of_3042_div_98_l741_741477


namespace max_colorful_subsets_l741_741374

-- Define the set S
def S : set (ℕ × ℕ) := { p | p.1 ≥ 1 ∧ p.1 ≤ 100 ∧ p.2 ≥ 1 ∧ p.2 ≤ 100 }

-- Define what it means for a subset T of S to be colorful
def colorful (T : set (ℕ × ℕ)) [decidable_pred T] : Prop :=
  ∃ (p1 p2 p3 p4 ∈ S),
    T = {p1, p2, p3, p4} ∧
    (p1.2 = p2.2 ∧ p3.2 = p4.2 ∧ p1.1 = p3.1 ∧ p2.1 = p4.1) ∧
    (p1 ≠ p3 ∧ p2 ≠ p4 ∧ p1 ≠ p2 ∧ p1 ≠ p4) ∧
    (∀ p ∈ T, ∃ (c : ℕ), c ≥ 1 ∧ c ≤ 4)

-- Define the problem statement
theorem max_colorful_subsets : 
  ∃ (T : set (ℕ × ℕ)) [decidable_pred T], 
    colorful T ∧ 
    ∀ (T' : set (ℕ × ℕ)) [decidable_pred T'], colorful T' → T ⊆ T' → T'.card ≤ 150000 :=
sorry

end max_colorful_subsets_l741_741374


namespace number_of_ordered_quadruples_l741_741589

theorem number_of_ordered_quadruples :
  let quadruples := { (a, b, c, d) : ℝ × ℝ × ℝ × ℝ | a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
                     a^2 + b^2 + c^2 + d^2 = 9 ∧
                     (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81 } 
  in quadruples.fintype.card = 15 := 
sorry

end number_of_ordered_quadruples_l741_741589


namespace expansion_term_x4_largest_binomial_coeff_largest_coefficient_l741_741271

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem expansion_term_x4 (n : ℕ) (r : ℕ) (x : ℕ) :
  (3 * x^2 + 3 * x) ^ n = 992 + 2^n → 
  n = 5 → 
  ∃ (r : ℕ), r = 2 ∧ (binom 5 r * 3^r * x^4) = 90 * x^4 := 
by
  intro h1 h2
  use 2
  split
  case left =>
    rfl
  case right =>
    sorry

theorem largest_binomial_coeff (n : ℕ) (x : ℕ) :
  (3 * x^2 + 3 * x) ^ n = 992 + 2^n → 
  n = 5 → 
  (∃ k, k = 2 ∧ binom 5 k * 3^k * x^4 = 90 * x^4) ∧ 
  (∃ k, k = 3 ∧ binom 5 k * 3^k * x^(13/3) = 270 * x^(13/3)) := 
by
  intro h1 h2
  split
  case left =>
    use 2
    split
    case left => rfl
    case right => sorry
  case right =>
    use 3
    split
    case left => rfl
    case right => sorry
    
theorem largest_coefficient (n : ℕ) (x : ℕ) :
  (3 * x^2 + 3 * x) ^ n = 992 + 2^n → 
  n = 5 → 
  ∃ k, k = 4 ∧ (binom 5 k * 3^k * x^(14/3)) = 405 * x^(14/3) :=
by
  intro h1 h2
  use 4
  split
  case left =>
    rfl
  case right =>
    sorry

end expansion_term_x4_largest_binomial_coeff_largest_coefficient_l741_741271


namespace strawberries_count_l741_741462

theorem strawberries_count (total_fruits : ℕ) (kiwi_fraction : ℚ) (remaining_fraction : ℚ) 
  (H1 : total_fruits = 78) (H2 : kiwi_fraction = 1 / 3) (H3 : remaining_fraction = 2 / 3) : 
  let kiwi_count := kiwi_fraction * total_fruits
      strawberry_count := remaining_fraction * total_fruits in
  strawberry_count = 52 := 
by
  sorry

end strawberries_count_l741_741462


namespace range_of_a_l741_741504

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → (x^2 - 2 * a * x + 2) ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l741_741504


namespace cos_alpha_value_l741_741302

-- Define the problem
theorem cos_alpha_value (α : ℝ) 
  (h : 2 * cos (2 * α) + 9 * sin α = 4) :
  cos α = (sqrt 15 / 4) ∨ cos α = -(sqrt 15 / 4) :=
by
  sorry

end cos_alpha_value_l741_741302


namespace chime_2203_occurs_on_March_19_l741_741949

-- Define the initial conditions: chime patterns
def chimes_at_half_hour : Nat := 1
def chimes_at_hour (h : Nat) : Nat := if h = 12 then 12 else h % 12

-- Define the start time and the question parameters
def start_time_hours : Nat := 10
def start_time_minutes : Nat := 45
def start_day : Nat := 26 -- Assume February 26 as starting point, to facilitate day count accurately
def target_chime : Nat := 2203

-- Define the date calculation function (based on given solution steps)
noncomputable def calculate_chime_date (start_day : Nat) : Nat := sorry

-- The goal is to prove calculate_chime_date with given start conditions equals 19 (March 19th is the 19th day after the base day assumption of March 0)
theorem chime_2203_occurs_on_March_19 :
  calculate_chime_date start_day = 19 :=
sorry

end chime_2203_occurs_on_March_19_l741_741949


namespace minimum_throws_l741_741028

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741028


namespace arithmetic_sequence_length_l741_741299

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ (a d l : ℕ), a = 2 → d = 5 → l = 3007 → l = a + (n-1) * d → n = 602 :=
by
  sorry

end arithmetic_sequence_length_l741_741299


namespace lucas_change_l741_741785

-- Define the initial amount of money Lucas has
def initial_amount : ℕ := 20

-- Define the cost of one avocado
def cost_per_avocado : ℕ := 2

-- Define the number of avocados Lucas buys
def number_of_avocados : ℕ := 3

-- Calculate the total cost of avocados
def total_cost : ℕ := number_of_avocados * cost_per_avocado

-- Calculate the remaining amount of money (change)
def remaining_amount : ℕ := initial_amount - total_cost

-- The proposition to prove: Lucas brings home $14
theorem lucas_change : remaining_amount = 14 := by
  sorry

end lucas_change_l741_741785


namespace determine_H_zero_l741_741852

theorem determine_H_zero (E F G H : ℕ) 
  (h1 : E < 10) (h2 : F < 10) (h3 : G < 10) (h4 : H < 10)
  (add_eq : 10 * E + F + 10 * G + E = 10 * H + E)
  (sub_eq : 10 * E + F - (10 * G + E) = E) : 
  H = 0 :=
sorry

end determine_H_zero_l741_741852


namespace sum_of_digits_base2_310_l741_741085

-- We define what it means to convert a number to binary and sum its digits.
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (Nat.digits 2 n).sum

-- The main statement of the problem.
theorem sum_of_digits_base2_310 :
  sum_of_binary_digits 310 = 5 :=
by
  sorry

end sum_of_digits_base2_310_l741_741085


namespace circle_equation_line_equation_minimum_area_l741_741337

-- Given conditions as definitions
def parametricCircle (t : ℝ) : ℝ × ℝ :=
  (-5 + Real.sqrt 2 * Real.cos t, 3 + Real.sqrt 2 * Real.sin t)

def polarLine (ρ θ : ℝ) : Prop :=
  ρ * Real.cos (θ + Real.pi / 4) = -Real.sqrt 2

def pointA : ℝ × ℝ := (2, Real.pi / 2)
def pointB : ℝ × ℝ := (2, Real.pi)

-- Theorem statements to be proved
theorem circle_equation :
  ∃ x y, parametricCircle t = (x, y) → (x + 5)^2 + (y - 3)^2 = 2 :=
sorry

theorem line_equation (ρ θ : ℝ) :
  polarLine ρ θ → ρ * (Real.cos θ) - ρ * (Real.sin θ) = 2 :=
sorry

theorem minimum_area (P : ℝ → ℝ × ℝ) :
  (∀ t, P t = parametricCircle t) →
  let A := (0, 2)
  let B := (-2, 0)
  ∃ t, ∃ dmin, let d := (|-5 + Real.sqrt 2 * Real.cos t - 3 - Real.sqrt 2 * Real.sin t + 2|) / Real.sqrt 2 in
  dmin = 2 * Real.sqrt 2 ∧ dmin * 2 * Real.sqrt 2 / 2 = 4 :=
sorry

end circle_equation_line_equation_minimum_area_l741_741337


namespace minimum_throws_l741_741025

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741025


namespace number_of_points_l741_741632

def A : Set ℕ := {4}
def B : Set ℕ := {1, 2}
def C : Set ℕ := {1, 3, 5}

theorem number_of_points : (A.card) * (B.card) * (C.card) = 6 :=
by
  sorry

end number_of_points_l741_741632


namespace gcd_180_126_l741_741580

theorem gcd_180_126 : Nat.gcd 180 126 = 18 := by
  sorry

end gcd_180_126_l741_741580


namespace smallest_number_among_list_l741_741546

theorem smallest_number_among_list : ∀ (a b c d : ℝ), a = 0 → b = -1 → c = - (Real.sqrt 2) → d = -2 → d ≤ a ∧ d ≤ b ∧ d ≤ c :=
by
  intros _ _ _ _ h1 h2 h3 h4
  refine ⟨_, _, _⟩
  sorry
  sorry
  sorry

end smallest_number_among_list_l741_741546


namespace max_and_min_l741_741586

open Real

-- Define the function
def y (x : ℝ) : ℝ := 3 * x^4 - 6 * x^2 + 4

-- Define the interval
def a : ℝ := -1
def b : ℝ := 3

theorem max_and_min:
  ∃ (xmin xmax : ℝ), xmin = 1 ∧ xmax = 193 ∧
  (∀ x, a ≤ x ∧ x ≤ b → y(xmin) ≤ y x ∧ y x ≤ y(xmax)) :=
by
  sorry

end max_and_min_l741_741586


namespace discount_on_pony_jeans_l741_741103

theorem discount_on_pony_jeans 
  (F P : ℕ)
  (h1 : F + P = 25)
  (h2 : 5 * F + 4 * P = 100) : P = 25 :=
by
  sorry

end discount_on_pony_jeans_l741_741103


namespace garden_table_bench_ratio_l741_741957

theorem garden_table_bench_ratio:
  ∀ (total_cost table_cost bench_cost : ℕ), 
  bench_cost = 150 →
  total_cost = 450 →
  table_cost + bench_cost = total_cost →
  table_cost = 2 * bench_cost →
  table_cost / bench_cost = 2 := 
by
  intros total_cost table_cost bench_cost h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end garden_table_bench_ratio_l741_741957


namespace correct_value_of_a_l741_741394

namespace ProofProblem

-- Condition 1: Definition of set M
def M : Set ℤ := {x | x^2 ≤ 1}

-- Condition 2: Definition of set N dependent on a parameter a
def N (a : ℤ) : Set ℤ := {a, a * a}

-- Question translated: Correct value of a such that M ∪ N = M
theorem correct_value_of_a (a : ℤ) : (M ∪ N a = M) → a = -1 :=
by
  sorry

end ProofProblem

end correct_value_of_a_l741_741394


namespace number_of_true_propositions_is_three_l741_741164

def proposition1 (x : ℝ) : Prop := (x^2 - 3 * x + 2 = 0) → (x = 1)
def contrapositive1 (x : ℝ) : Prop := (x ≠ 1) → (x^2 - 3 * x + 2 ≠ 0)

def proposition2 (p q : Prop) : Prop := (¬ (p ∨ q)) → (¬ p ∧ ¬ q)

def proposition3 : Prop := ∃ x : ℝ, x^2 + x + 1 < 0
def negation3 : Prop := ∀ x : ℝ, x^2 + x + 1 ≥ 0

def proposition4 (A B : ℝ) [has_lt.lt A B] : Prop := ∀ (A B : ℝ), A < B → sin A < sin B

theorem number_of_true_propositions_is_three :
  ((∀ x, proposition1 x ↔ contrapositive1 x) ∧
   ∀ p q, proposition2 p q ∧
   ¬ proposition3 ∧ negation3 ∧
   ¬ ∀ (A B : ℝ), proposition4 A B) → 3 := 
sorry

end number_of_true_propositions_is_three_l741_741164


namespace quadrilateral_similarity_l741_741243

noncomputable def quadrilateral : Type := sorry
noncomputable def midpoint (A B : quadrilateral) : quadrilateral := sorry
noncomputable def perpendicular_bisector (A B : quadrilateral) : quadrilateral := sorry

def Q1 : quadrilateral := sorry
def Q2 : quadrilateral := 
  let A := midpoint Q1 Q1,
      B := midpoint Q1 Q1,
      C := midpoint Q1 Q1,
      D := midpoint Q1 Q1
  in quadrilateral.mk (perpendicular_bisector A B) (perpendicular_bisector B C) (perpendicular_bisector C D) (perpendicular_bisector D A)

def Q3 : quadrilateral :=
  let A := midpoint Q2 Q2,
      B := midpoint Q2 Q2,
      C := midpoint Q2 Q2,
      D := midpoint Q2 Q2
  in quadrilateral.mk (perpendicular_bisector A B) (perpendicular_bisector B C) (perpendicular_bisector C D) (perpendicular_bisector D A)

def similar (Q1 Q2 : quadrilateral) : Prop := sorry

theorem quadrilateral_similarity : similar Q3 Q1 := sorry

end quadrilateral_similarity_l741_741243


namespace minimum_throws_for_repeated_sum_l741_741042

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741042


namespace div_condition_l741_741629

theorem div_condition
  (a b : ℕ)
  (h₁ : a < 1000)
  (h₂ : b ≠ 0)
  (h₃ : b ∣ a ^ 21)
  (h₄ : b ^ 10 ∣ a ^ 21) :
  b ∣ a ^ 2 :=
sorry

end div_condition_l741_741629


namespace find_ad_value_l741_741270

noncomputable theory

variables {a b c d : ℝ}

def is_geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

def function_maximum (y : ℝ → ℝ) (x b : ℝ) (c : ℝ) : Prop :=
  y x = ln x - x ∧ ∀ x, ∃ b, ∀ x, ((x > 0) → c ≥ y x)

theorem find_ad_value (h_seq : is_geometric_sequence a b c d)
                      (h_func : function_maximum (λ x : ℝ, ln x - x) b c) :
  ad = -1 :=
begin
  sorry
end

end find_ad_value_l741_741270


namespace strawberries_in_crate_l741_741461

theorem strawberries_in_crate (total_fruit : ℕ) (fruit_in_crate : total_fruit = 78) (kiwi_fraction : ℚ) (kiwi_fraction_eq : kiwi_fraction = 1/3) :
  let kiwi_count := total_fruit * kiwi_fraction,
      strawberry_count := total_fruit - kiwi_count
  in 
  strawberry_count = 52 :=
by
  sorry

end strawberries_in_crate_l741_741461


namespace max_distance_is_135sqrt10_l741_741388

noncomputable def max_distance (z : ℂ) (hz : abs z = 3) : ℝ :=
  abs ((2 + 5 * complex.I) * z^4 - z^3)

theorem max_distance_is_135sqrt10 {z : ℂ} (hz : abs z = 3) :
  max_distance z hz = 135 * real.sqrt 10 := by sorry

end max_distance_is_135sqrt10_l741_741388


namespace cheaper_lens_price_l741_741399

theorem cheaper_lens_price (original_price : ℝ) (discount_rate : ℝ) (savings : ℝ) 
  (h₁ : original_price = 300) 
  (h₂ : discount_rate = 0.20) 
  (h₃ : savings = 20) 
  (discounted_price : ℝ) 
  (cheaper_lens_price : ℝ)
  (discount_eq : discounted_price = original_price * (1 - discount_rate))
  (savings_eq : cheaper_lens_price = discounted_price - savings) :
  cheaper_lens_price = 220 := 
by sorry

end cheaper_lens_price_l741_741399


namespace centroid_equidistant_l741_741373

theorem centroid_equidistant {A B C P Q X Y : ℝ} 
  (acute_triangle : ∀ {A B C : ℝ}, is_acute_triangle A B C) 
  (on_segment_BC : P ∈ segment B C ∧ Q ∈ segment B C) 
  (BP_PQ_QC : dist B P = dist P Q ∧ dist P Q = dist Q C)
  (feet_perp_XY : is_perpendicular_foot P AC X ∧ is_perpendicular_foot Q AB Y) :
  equidistant_centroid_lines QX PY :=
sorry

end centroid_equidistant_l741_741373


namespace sqrt_two_irrational_l741_741418

theorem sqrt_two_irrational (p q : ℕ) (h : nat.coprime p q) (hpq : (p:ℝ) / (q:ℝ) = real.sqrt 2) : false :=
by
  sorry

end sqrt_two_irrational_l741_741418


namespace triangle_equality_l741_741116

-- Assuming we have the necessary geometric definitions and axioms from Mathlib.
open EuclideanGeometry

theorem triangle_equality 
  (A B C D : Point)
  (hAD_DC : dist A D = dist D C)
  (hAC_AB : dist A C = dist A B)
  (hCAB_20 : ∠ A C B = 20)
  (hADC_100 : ∠ A D C = 100) : 
  dist A B = dist B C + dist C D :=
by
  sorry

end triangle_equality_l741_741116


namespace unique_solution_complex_eq_l741_741770

open Complex

theorem unique_solution_complex_eq (z ω λ : ℂ) (h : |λ| ≠ 1) :
  z = (conj λ) * ω + conj ω / (1 - |λ| ^ 2) → --prove that this is a solution
  ∀ z' : ℂ, z' - conj (λ * z') = conj ω → z' = (conj λ) * ω + conj ω / (1 - |λ| ^ 2) := -- prove uniqueness
sorry


end unique_solution_complex_eq_l741_741770


namespace square_on_circle_tangent_radius_l741_741872

theorem square_on_circle_tangent_radius
  (A B C D: Point)
  (O: Point)
  (R: ℝ)
  (h₁: square A B C D)
  (h₂: area A B C D = 256)
  (h₃: ∃ P Q: Point, P ∉ interior (circle O R) ∧ Q ∉ interior (circle O R) ∧ side A B ⊆ (circle O R))
  (h₄: tangent L (circle O R) ∧ side C D ⊆ L)
  : R = 10 :=
by 
  sorry

end square_on_circle_tangent_radius_l741_741872


namespace part1_part2_part3_l741_741669

-- For part (1)
theorem part1 (θ : ℝ) (Hθ : θ ∈ set.Ioo 0 real.pi) 
  (Hmono : ∀ x ∈ set.Ici 1, (∂g(x) / ∂x ≥ 0)) : θ = real.pi / 2 :=
  sorry

-- For part (2)
theorem part2 (f g : ℝ → ℝ) (f' : ℝ → ℝ) (Hf_def : ∀ x ∈ set.Icc 1 2, f(x) = g(x) + (2 * x - 1) / (2 * x^2)) 
  (Hf_prime : ∀ x ∈ set.Icc 1 2, f'(x) = derivative (f, x))
  (Hf_mono : ∀ x ∈ set.Icc 1 2, f(x) > f'(x) + 1/2) : true :=
  sorry

-- For part (3)
theorem part3 (k : ℝ) (g : ℝ → ℝ)
  (Hineq : ∀ x > 0, exp x - x - 1 ≥ k * g(x + 1)) : k ≤ 1 :=
  sorry

end part1_part2_part3_l741_741669


namespace friends_can_reach_destinations_l741_741437

/-- The distance between Coco da Selva and Quixajuba is 24 km. 
    The walking speed is 6 km/h and the biking speed is 18 km/h. 
    Show that the friends can proceed to reach their destinations in at most 2 hours 40 minutes, with the bicycle initially in Quixajuba. -/
theorem friends_can_reach_destinations (d q c : ℕ) (vw vb : ℕ) (h1 : d = 24) (h2 : vw = 6) (h3 : vb = 18): 
  (∃ ta tb tc : ℕ, ta ≤ 2 * 60 + 40 ∧ tb ≤ 2 * 60 + 40 ∧ tc ≤ 2 * 60 + 40 ∧ 
     True) :=
sorry

end friends_can_reach_destinations_l741_741437


namespace independence_test_method_l741_741735

noncomputable theory

-- defining the independence test and contour bar chart
def independence_test : Type := sorry -- a placeholder definition
def contour_bar_chart : independence_test := sorry -- specifying contour_bar_chart as a method in independence_test

-- stating the theorem to prove
theorem independence_test_method :
  (∃ m : independence_test, m = contour_bar_chart) :=
sorry

end independence_test_method_l741_741735


namespace mowing_time_tie_l741_741548

theorem mowing_time_tie (x y : ℝ) (hx0 : 0 < x) (hy0 : 0 < y) :
  let a_area := x,
      b_area := x / 2,
      c_area := x / 3,
      a_rate := y,
      b_rate := y / 2,
      c_rate := y / 3,
      a_time := a_area / a_rate,
      b_time := b_area / b_rate,
      c_time := c_area / c_rate in
  a_time = b_time ∧ b_time = c_time :=
by
  -- Proof skipped for this statement
  sorry

end mowing_time_tie_l741_741548


namespace count_valid_numbers_sum_even_valid_numbers_l741_741983

-- Define the necessary conditions
def valid_digits : List ℕ := [1, 2, 3, 4, 5]
def is_four_digit (n : ℕ) : Prop := n / 1000 > 0 ∧ n / 10000 = 0
def no_repeats (n : ℕ) : Prop :=
  let digits := [n % 10, (n / 10) % 10, (n / 100) % 10, n / 1000]
  List.nodup digits
def not_in_place (n : ℕ) (digit : ℕ) (place : ℕ) : Prop :=
  (n / place) % 10 ≠ digit
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the set of valid numbers
def valid_numbers : List ℕ :=
  List.filter (λ n, is_four_digit n ∧ no_repeats n ∧
    not_in_place n 1 100 ∧ not_in_place n 2 10) 
  (List.range 10000)

-- Lean statement to verify the count of such numbers in valid_numbers is 78
theorem count_valid_numbers : List.length valid_numbers = 78 := sorry

-- Lean statement to verify the sum of all even numbers in valid_numbers is 159984
theorem sum_even_valid_numbers : 
  List.sum (List.filter is_even valid_numbers) = 159984 := sorry

end count_valid_numbers_sum_even_valid_numbers_l741_741983


namespace area_of_triangle_XYZ_l741_741975

noncomputable def centroid (p1 p2 p3 : (ℚ × ℚ)) : (ℚ × ℚ) :=
((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

noncomputable def triangle_area (p1 p2 p3 : (ℚ × ℚ)) : ℚ :=
abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p1.2 - p1.2 * p2.1 - p2.2 * p3.1 - p3.2 * p1.1) / 2)

noncomputable def point_A : (ℚ × ℚ) := (5, 12)
noncomputable def point_B : (ℚ × ℚ) := (0, 0)
noncomputable def point_C : (ℚ × ℚ) := (14, 0)

noncomputable def point_X : (ℚ × ℚ) :=
(109 / 13, 60 / 13)
noncomputable def point_Y : (ℚ × ℚ) :=
centroid point_A point_B point_X
noncomputable def point_Z : (ℚ × ℚ) :=
centroid point_B point_C point_Y

theorem area_of_triangle_XYZ : triangle_area point_X point_Y point_Z = 84 / 13 :=
sorry

end area_of_triangle_XYZ_l741_741975


namespace minimum_rolls_for_duplicate_sum_l741_741061

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741061


namespace problem_statement_l741_741308

theorem problem_statement (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_statement_l741_741308


namespace customer_survey_response_l741_741535

theorem customer_survey_response (N : ℕ)
  (avg_income : ℕ → ℕ)
  (avg_all : avg_income N = 45000)
  (avg_top10 : avg_income 10 = 55000)
  (avg_others : avg_income (N - 10) = 42500) :
  N = 50 := 
sorry

end customer_survey_response_l741_741535


namespace range_of_k_l741_741610

theorem range_of_k (k : ℝ) : ((∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0)) :=
sorry

end range_of_k_l741_741610


namespace probability_not_snowing_l741_741856

  -- Define the probability that it will snow tomorrow
  def P_snowing : ℚ := 2 / 5

  -- Define the probability that it will not snow tomorrow
  def P_not_snowing : ℚ := 1 - P_snowing

  -- Theorem stating the required proof
  theorem probability_not_snowing : P_not_snowing = 3 / 5 :=
  by 
    -- Proof would go here
    sorry
  
end probability_not_snowing_l741_741856


namespace equation_of_line_through_P_with_angle_l741_741265

def point (x y : ℝ) := (x, y)

noncomputable def line_equation_through_point_and_inclination 
  (P : ℝ × ℝ) (inclination : ℝ) : Prop :=
  ∃ b : ℝ, P.2 = b ∧ 
  if inclination = 90 then P.1 = b else ∃ m : ℝ, P.2 = m * P.1 + b

theorem equation_of_line_through_P_with_angle :
  line_equation_through_point_and_inclination (3, 4) 90 ↔ (λ x y : ℝ, x = 3) :=
begin
  sorry
end

end equation_of_line_through_P_with_angle_l741_741265


namespace solution_set_of_inequality_l741_741607

theorem solution_set_of_inequality (x : ℝ) : x < (1 / x) ↔ (x < -1 ∨ (0 < x ∧ x < 1)) :=
by
  sorry

end solution_set_of_inequality_l741_741607


namespace problem_statement_l741_741283

noncomputable def f (x m : ℝ) : ℝ := Real.exp x * (Real.log x + (x - m)^2)

noncomputable def f_prime (x m : ℝ) : ℝ := Real.exp x * (1 / x + 2 * (x - m))

theorem problem_statement (m : ℝ) :
  (∀ x > 0, f_prime x m - f x m > 0) → m < Real.sqrt 2 :=
begin
  sorry
end

end problem_statement_l741_741283


namespace vasya_100_using_fewer_sevens_l741_741909

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l741_741909


namespace vasya_example_fewer_sevens_l741_741878

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l741_741878


namespace min_throws_to_ensure_same_sum_twice_l741_741073

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741073


namespace num_600_ray_partitional_but_not_360_ray_partitional_l741_741378

def is_4n_ray_partitional (R : set (ℝ × ℝ)) (n : ℕ) (X : ℝ × ℝ) : Prop :=
  ∃ rays : set (ℝ × ℝ), 
  X ∈ R ∧
  (card rays = 4 * n) ∧
  (∀ (r ∈ rays), is_ray_emanating_from X r) ∧
  divides_into_equal_areas R rays 4*n

def num_600_ray_but_not_360_ray_partitional (R : set (ℝ × ℝ)) : ℕ :=
  let grid_151 := grid_points R (1/151)
  let grid_91 := grid_points R (1/91)
  let intersection := grid_151 ∩ grid_91
  card grid_151 - card intersection

theorem num_600_ray_partitional_but_not_360_ray_partitional (R : set (ℝ × ℝ)) :
  num_600_ray_but_not_360_ray_partitional R = 22800 :=
sorry

end num_600_ray_partitional_but_not_360_ray_partitional_l741_741378


namespace Rick_is_three_times_Sean_l741_741812

-- Definitions and assumptions
def Fritz_money : ℕ := 40
def Sean_money : ℕ := (Fritz_money / 2) + 4
def total_money : ℕ := 96

-- Rick's money can be derived from total_money - Sean_money
def Rick_money : ℕ := total_money - Sean_money

-- Claim to be proven
theorem Rick_is_three_times_Sean : Rick_money = 3 * Sean_money := 
by 
  -- Proof steps would go here
  sorry

end Rick_is_three_times_Sean_l741_741812


namespace minimum_throws_for_repeated_sum_l741_741046

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741046


namespace change_in_spiders_l741_741554

theorem change_in_spiders 
  (x a y b : ℤ) 
  (h1 : x + a = 20) 
  (h2 : y + b = 23) 
  (h3 : x - b = 5) :
  y - a = 8 := 
by
  sorry

end change_in_spiders_l741_741554


namespace definite_quadratic_radical_l741_741979

theorem definite_quadratic_radical : 
  ∀ (x : ℝ), (∃ (a : ℝ), a ≥ 0 ∧ -sqrt 3 = sqrt a) ∧
  (¬ ∃ (a : ℝ), a ≥ 0 ∧ sqrt[3]{3} = sqrt a) ∧
  (¬ (∀ (x : ℝ), x ≥ 0) ∧ sqrt x = sqrt (abs x)) ∧
  (¬ ∃ (a : ℝ), a ≥ 0 ∧ sqrt (-3) = sqrt a) :=
by
  sorry

end definite_quadratic_radical_l741_741979


namespace trigonometric_identity_l741_741304

theorem trigonometric_identity (α : Real) (h : Real.tan (α + Real.pi / 4) = -3) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 9 / 5 :=
sorry

end trigonometric_identity_l741_741304


namespace find_x_values_l741_741223

theorem find_x_values (x : ℝ) (h : x ^ log x / log 10 = x ^ 4 / 1000) : x = 10 ∨ x = 1000 :=
sorry

end find_x_values_l741_741223


namespace cos_arctan_12_5_l741_741560

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem cos_arctan_12_5 : 
  cos (arctan (12 / 5)) = 5 / 13 :=
by
  have h_hyp : hypotenuse 5 12 = 13 := by
    unfold hypotenuse
    rw [Real.sqrt_add, Real.sqrt_eq_rpow, ←Real.sqrt_l_eq_rpow, Real.pow_add]
    norm_num
  have h_tan : tan (arctan (12 / 5)) = 12 / 5 := Real.tan_arctan (show (12 : ℝ) / 5 > 0 by norm_num)
  sorry

end cos_arctan_12_5_l741_741560


namespace sum_of_squares_of_segments_is_diameter_squared_l741_741702

theorem sum_of_squares_of_segments_is_diameter_squared
  {O : Point} {R : ℝ} (circle : Circle O R)
  {A B C D E : Point} 
  (hA : circle.On A) (hB : circle.On B) (hC : circle.On C) (hD : circle.On D)
  (h1 : Chord A B) (h2 : Chord C D)
  (hIntersection : ChordsIntersectRightAngle h1 h2 E) :
  let AE := Segment A E
      BE := Segment B E
      CE := Segment C E
      DE := Segment D E
      AF := Diameter circle
  in (AE.length^2 + BE.length^2 + CE.length^2 + DE.length^2 = AF.length^2) :=
sorry

end sum_of_squares_of_segments_is_diameter_squared_l741_741702


namespace clowns_per_mobile_28_l741_741841

def clowns_in_each_mobile (total_clowns num_mobiles : Nat) (h : total_clowns = 140 ∧ num_mobiles = 5) : Nat :=
  total_clowns / num_mobiles

theorem clowns_per_mobile_28 (total_clowns num_mobiles : Nat) (h : total_clowns = 140 ∧ num_mobiles = 5) :
  clowns_in_each_mobile total_clowns num_mobiles h = 28 :=
by
  sorry

end clowns_per_mobile_28_l741_741841


namespace curve_and_line_polar_coordinates_and_locus_of_Q_l741_741345

/-- Given curve C defined parametrically and line l defined parametrically,
find their respective polar coordinates equations and verify that the locus
of point Q satisfies the given rectangular coordinate equation. -/
theorem curve_and_line_polar_coordinates_and_locus_of_Q :
  (∀ α : ℝ, let x := cos α, y := 2 * sin α in 4 * x^2 + y^2 = 4) ∧
  (∀ t : ℝ, let x := 1 + t, y := 3 - 2 * t in 2 * x + y = 5) ∧
  (∀ θ ρ : ℝ,
    (5 * (4 / (4 * cos θ^2 + sin θ^2)) = 4 * ((5 / (2 * cos θ + sin θ)) * ρ) →
      (4 * (ρ * cos θ)^2 + (ρ * sin θ)^2 = 2 * (ρ * cos θ) + 2 * (ρ * sin θ))) →
    4 * x^2 + y^2 = 2 * x + 2 * y) :=
begin
  sorry
end

end curve_and_line_polar_coordinates_and_locus_of_Q_l741_741345


namespace triangle_is_isosceles_l741_741716

open Real

theorem triangle_is_isosceles 
 {A B C : ℝ} {a b c : ℝ}
  (h1: A + B + C = π)  -- Sum of the interior angles of a triangle
  (h2: a = 2 * sin(A) * c)
  (h3: b = 2 * sin(B) * c)
  (h: a * cos B = b * cos A) :
  A = B :=
by
  sorry

end triangle_is_isosceles_l741_741716


namespace cone_height_is_correct_l741_741133

-- Given conditions
variables (R : ℝ)
def cone_base_radius := (2 * R) / 3
noncomputable def cone_height := sqrt (R^2 - (cone_base_radius R)^2)

-- Theorem statement
theorem cone_height_is_correct : cone_height R = (sqrt 5 * R) / 3 :=
by
  sorry  -- Proof not required here

end cone_height_is_correct_l741_741133


namespace notebooks_type_A_count_minimum_profit_m_l741_741798

def total_notebooks := 350
def costA := 12
def costB := 15
def total_cost := 4800

def selling_priceA := 20
def selling_priceB := 25
def discountA := 0.7
def profit_min := 2348

-- Prove the number of type A notebooks is 150
theorem notebooks_type_A_count (x y : ℕ) (h1 : x + y = total_notebooks)
    (h2 : costA * x + costB * y = total_cost) : x = 150 := by
  sorry

-- Prove the minimum value of m is 111 such that profit is not less than 2348
theorem minimum_profit_m (m : ℕ) (profit : ℕ)
    (h : profit = (m * selling_priceA + m * selling_priceB  + (150 - m) * (selling_priceA * discountA).toNat + (200 - m) * costB - total_cost))
    (h_prof : profit >= profit_min) : m >= 111 := by
  sorry

end notebooks_type_A_count_minimum_profit_m_l741_741798


namespace Dodo_is_sane_l741_741200

-- Declare the names of the characters
inductive Character
| Dodo : Character
| Lori : Character
| Eagle : Character

open Character

-- Definitions of sanity state
def sane (c : Character) : Prop := sorry
def insane (c : Character) : Prop := ¬ sane c

-- Conditions based on the problem statement
axiom Dodo_thinks_Lori_thinks_Eagle_not_sane : (sane Lori → insane Eagle)
axiom Lori_thinks_Dodo_not_sane : insane Dodo
axiom Eagle_thinks_Dodo_sane : sane Dodo

-- Theorem to prove Dodo is sane
theorem Dodo_is_sane : sane Dodo :=
by {
    sorry
}

end Dodo_is_sane_l741_741200


namespace rectangle_perimeter_is_104_l741_741528

noncomputable def perimeter_of_rectangle (b : ℝ) (h1 : b > 0) (h2 : 3 * b * b = 507) : ℝ :=
  2 * (3 * b) + 2 * b

theorem rectangle_perimeter_is_104 {b : ℝ} (h1 : b > 0) (h2 : 3 * b * b = 507) :
  perimeter_of_rectangle b h1 h2 = 104 :=
by
  sorry

end rectangle_perimeter_is_104_l741_741528


namespace corresponding_side_of_larger_triangle_l741_741436

-- Conditions
variables (A1 A2 : ℕ) (s1 s2 : ℕ)
-- A1 is the area of the larger triangle
-- A2 is the area of the smaller triangle
-- s1 is a side of the smaller triangle = 4 feet
-- s2 is the corresponding side of the larger triangle

-- Given conditions as hypotheses
axiom diff_in_areas : A1 - A2 = 32
axiom ratio_of_areas : A1 = 9 * A2
axiom side_of_smaller_triangle : s1 = 4

-- Theorem to prove the corresponding side of the larger triangle
theorem corresponding_side_of_larger_triangle 
  (h1 : A1 - A2 = 32)
  (h2 : A1 = 9 * A2)
  (h3 : s1 = 4) : 
  s2 = 12 :=
sorry

end corresponding_side_of_larger_triangle_l741_741436


namespace daily_harvest_l741_741840

theorem daily_harvest (sacks_per_section : ℕ) (num_sections : ℕ) 
  (h1 : sacks_per_section = 45) (h2 : num_sections = 8) : 
  sacks_per_section * num_sections = 360 :=
by
  sorry

end daily_harvest_l741_741840


namespace largest_prime_divisor_of_213024033_5_l741_741584

def base5_to_decimal (n : Nat) : Nat := 
  2 * 5^8 + 1 * 5^7 + 3 * 5^6 + 0 * 5^5 + 2 * 5^4 + 4 * 5^3 + 0 * 5^2 + 3 * 5^1 + 3 * 5^0

def prime_factors (n : Nat) : List Nat := 
  -- List of prime factors
  [2, 3, 13, 11019]  -- Already known in the problem

def largest_prime_divisor (n : Nat) : Nat :=
  -- Select the maximum from the list of prime factors
  List.maximum (prime_factors n)

theorem largest_prime_divisor_of_213024033_5 : 
  largest_prime_divisor (base5_to_decimal 213024033) = 11019 := 
by 
  -- This is where the actual proof would go
  sorry

end largest_prime_divisor_of_213024033_5_l741_741584


namespace correct_options_l741_741650

theorem correct_options (a b c : ℝ) (h1 : ∀ x : ℝ, (a*x^2 + b*x + c > 0) ↔ (-3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, (b*x + c > 0) ↔ x > 6) = False ∧ (∀ x, (c*x^2 + b*x + a < 0) ↔ (-1/3 < x ∧ x < 1/2)) :=
by 
  sorry

end correct_options_l741_741650


namespace hyperbola_equation_l741_741643

-- Definitions for the conditions
def center_origin : Prop := (0, 0) = (0, 0)

def foci_positions : Prop := (-real.sqrt 5, 0) = (-real.sqrt 5, 0) ∧ (real.sqrt 5, 0) = (real.sqrt 5, 0)

def perpendicular_condition (P : ℝ × ℝ) : Prop := 
  let F1 := (-real.sqrt 5, 0)
  let F2 := (real.sqrt 5, 0)
  let PF1 := real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)
  let PF2 := real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)
  PF1 * PF2 = 2
  
def area_condition (P : ℝ × ℝ) : Prop := 
  let F1 := (-real.sqrt 5, 0)
  let F2 := (real.sqrt 5, 0)
  let area := (1 / 2) * (abs ((P.1 * (F1.2 - F2.2) + F1.1 * (F2.2 - P.2) + F2.1 * (P.2 - F1.2))))
  area = 1

-- Statement to prove
theorem hyperbola_equation : center_origin ∧ foci_positions ∧ ∃ P: ℝ × ℝ, perpendicular_condition P ∧ area_condition P → ∃ a b : ℝ, a = 4 ∧ b = 1 ∧ ∀ (x y : ℝ), (x^2) / a - (y^2) / b = 1 :=
by
  sorry

end hyperbola_equation_l741_741643


namespace tangent_line_at_y_eq_fx_l741_741244

noncomputable def f : ℝ → ℝ :=
  λ x, 2 * f (8 - x) - x^2 + 11 * x - 18

theorem tangent_line_at_y_eq_fx (x : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x, f x = 2 * f (8 - x) - x^2 + 11 * x - 18) →
  let f_deriv := deriv f in
  let slope := f_deriv 4 in
  let y_point := (λ x, x^2 - 7 * x + 2) 4 in
  (slope = 1) ∧ (y_point = -10) →
  let tangent_line := λ (x : ℝ), slope * x + (y_point - slope * 4) in
  tangent_line 0 = -14 := by
  sorry

end tangent_line_at_y_eq_fx_l741_741244


namespace angle_APB_is_106_l741_741343

theorem angle_APB_is_106
  (O1 O2 P A B S R T : Type*) -- defining the points
  (h1 : ∀ (X : Type*), SRT X → affine_space ℝ Type*)
  (h2 : aff.eq_linear_independent S R T) -- SRT being a straight line
  (h3 : circle semicircle_SAR)
  (h4 : circle semicircle_RBT)
  (h5 : (arc semicircle_SAR AS) = 48)
  (h6 : (arc semicircle_RBT BT) = 58)
  (h7 : (arc semicircle_RBT RT) = (arc semicircle_SAR SR) + 10) :
  ∠(PA) (PB) = 106 :=
by
  sorry

end angle_APB_is_106_l741_741343


namespace pascal_triangle_21st_number_l741_741474

theorem pascal_triangle_21st_number 
: (Nat.choose 22 2) = 231 :=
by 
  sorry

end pascal_triangle_21st_number_l741_741474


namespace possible_values_of_k_l741_741375

noncomputable def operation (k x y : ℝ) : ℝ :=
  if h : x + y + k ≠ 0 then x * y / (x + y + k) else 0

theorem possible_values_of_k :
  let roots := {x : ℝ // ∃ r : ℕ → ℝ, r 0 = x ∧ x^4 = 27 * (x^2 + x + 1)}
  ∃ (x1 x2 x3 x4 : ℝ) (k : ℝ),
    x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
    (x1, x2, x3, x4 ∈ roots) ∧
    operation k (operation k (operation k x1 x2) x3) x4 = 1 ∧
    (k = 3 ∨ k = -6) :=
sorry

end possible_values_of_k_l741_741375


namespace bills_are_fake_bart_can_give_exact_amount_l741_741176

-- Problem (a)
theorem bills_are_fake : 
  (∀ x, x = 17 ∨ x = 19 → false) :=
sorry

-- Problem (b)
theorem bart_can_give_exact_amount (n : ℕ) :
  (∀ m, m = 323  → (n ≥ m → ∃ a b : ℕ, n = 17 * a + 19 * b)) :=
sorry

end bills_are_fake_bart_can_give_exact_amount_l741_741176


namespace pizza_consumption_order_l741_741227

theorem pizza_consumption_order :
  let total_slices := 168
  let alex_slices := (1/6) * total_slices
  let beth_slices := (2/7) * total_slices
  let cyril_slices := (1/3) * total_slices
  let eve_slices_initial := (1/8) * total_slices
  let dan_slices_initial := total_slices - (alex_slices + beth_slices + cyril_slices + eve_slices_initial)
  let eve_slices := eve_slices_initial + 2
  let dan_slices := dan_slices_initial - 2
  (cyril_slices > beth_slices ∧ beth_slices > eve_slices ∧ eve_slices > alex_slices ∧ alex_slices > dan_slices) :=
  sorry

end pizza_consumption_order_l741_741227


namespace minimum_throws_to_ensure_same_sum_twice_l741_741008

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741008


namespace max_negative_ints_l741_741314

noncomputable def product4 (w x y z : ℤ) : ℤ := w * x * y * z
noncomputable def product4 (w x y z : ℤ) : ℤ := w * x * y * z

theorem max_negative_ints (a b c d e f g h i j : ℤ) (h1 : a * b + product4 c d e f + product4 g h i j < 0) : 
  let w := 7 in w = 7 :=
by
  -- Proof omitted
  sorry

end max_negative_ints_l741_741314


namespace largest_prime_factor_210_l741_741862

-- Define the prime factors of 210
def prime_factors_210 := {2, 3, 5, 7}

-- State the problem formally
theorem largest_prime_factor_210 : 
  ∃ p ∈ prime_factors_210, (∀ q ∈ prime_factors_210, q ≤ p) ∧ p = 7 :=
by {
  sorry
}

end largest_prime_factor_210_l741_741862


namespace burger_cost_proof_l741_741178

variable {burger_cost fries_cost salad_cost total_cost : ℕ}
variable {quantity_of_fries : ℕ}

theorem burger_cost_proof (h_fries_cost : fries_cost = 2)
    (h_salad_cost : salad_cost = 3 * fries_cost)
    (h_quantity_of_fries : quantity_of_fries = 2)
    (h_total_cost : total_cost = 15)
    (h_equation : burger_cost + (quantity_of_fries * fries_cost) + salad_cost = total_cost) :
    burger_cost = 5 :=
by 
  sorry

end burger_cost_proof_l741_741178


namespace polynomial_irreducible_l741_741392

variable (a : ℕ → ℤ) (n : ℕ)
hypothesis h1 : n ≥ 2
hypothesis h2 : ∀ i j : ℕ, i < n → j < n → i ≠ j → a i ≠ a j

theorem polynomial_irreducible :
  let f : Polynomial ℤ := ∏ i in Finset.range n, (Polynomial.X - Polynomial.C (a i)) - 1
  irreducible f := by
  sorry

end polynomial_irreducible_l741_741392


namespace part1_part2_l741_741502

-- Part (1) Lean 4 statement
theorem part1 {x : ℕ} (h : 0 < x ∧ 4 * (x + 2) < 18 + 2 * x) : x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 :=
sorry

-- Part (2) Lean 4 statement
theorem part2 (x : ℝ) (h1 : 5 * x + 2 ≥ 4 * x + 1) (h2 : (x + 1) / 4 > (x - 3) / 2 + 1) : -1 ≤ x ∧ x < 3 :=
sorry

end part1_part2_l741_741502


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741896

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741896


namespace vasya_100_using_fewer_sevens_l741_741907

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l741_741907


namespace snake_turns_in_green_l741_741510

-- Define the board size and snake pattern
def board_size : Nat := 2018

-- Color pattern definition
inductive Color : Type
| green
| red
| blue

-- Snake structure and properties
structure Snake :=
  (position : Fin board_size × Fin board_size → Color)

-- Condition: Snake color pattern
def color_pattern (n : Nat) : Color :=
  match n % 4 with
  | 0 => Color.green
  | 1 => Color.red
  | 2 => Color.green
  | _ => Color.blue

-- The certain pattern of the snake
def snake : Snake := ⟨λ n, color_pattern (n.1 + n.2)⟩

-- Theorem statement: Snake turns in one of the green cells
theorem snake_turns_in_green
  (h_diagonal_red : ∃ (x y : Fin board_size), 
   snake.position (x, y) = Color.red ∧ 
   ∃ (dx dy : Fin board_size), 
     dx ≠ 0 ∧ dy ≠ 0 ∧ 
     (dx = 1 ∨ dx = -1) ∧ (dy = 1 ∨ dy = -1) ∧ 
     snake.position (x + dx, y + dy) = Color.red) :
  ∃ (x y : Fin board_size), 
    snake.position (x, y) = Color.green ∧ 
    (snake.position (x, y) ≠ snake.position (x + 1, y) ∨
     snake.position (x, y) ≠ snake.position (x, y + 1) ∨
     snake.position (x, y) ≠ snake.position (x - 1, y) ∨
     snake.position (x, y) ≠ snake.position (x, y - 1)) :=
by
  sorry

end snake_turns_in_green_l741_741510


namespace value_of_expression_l741_741698

theorem value_of_expression (x : ℝ) (h : x + 3 = 10) : 5 * x + 15 = 50 := by
  sorry

end value_of_expression_l741_741698


namespace sum_of_odd_subsets_S4_l741_741753

-- Definitions from conditions
def Sn (n : ℕ) : Set ℕ := { i | 1 ≤ i ∧ i ≤ n }
def capacity (X : Set ℕ) : ℕ := if X = ∅ then 0 else X.prod id
def is_odd_subset (X : Set ℕ) : Prop := X.prod id % 2 = 1
def odd_subsets (n : ℕ) : Set (Set ℕ) := { X | X ⊆ Sn n ∧ is_odd_subset X }

-- Statement of the problem
theorem sum_of_odd_subsets_S4 : (∑ X in odd_subsets 4, capacity X) = 7 := 
  sorry

end sum_of_odd_subsets_S4_l741_741753


namespace MrsHiltTravelMiles_l741_741400

theorem MrsHiltTravelMiles
  (one_book_miles : ℕ)
  (finished_books : ℕ)
  (total_miles : ℕ)
  (h1 : one_book_miles = 450)
  (h2 : finished_books = 15)
  (h3 : total_miles = one_book_miles * finished_books) :
  total_miles = 6750 :=
by
  sorry

end MrsHiltTravelMiles_l741_741400


namespace blocks_difference_l741_741411

def blocks_house := 89
def blocks_tower := 63

theorem blocks_difference : (blocks_house - blocks_tower = 26) :=
by sorry

end blocks_difference_l741_741411


namespace arith_seq_proof_gen_formula_T_n_range_l741_741775

-- Definitions of the sequence
def seq (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 6
  else 2 * seq (n - 1) - seq (n - 2) + 2

-- Definitions of the arithmetic sequence
def arith_seq (n : ℕ) : ℕ → ℕ
| 0 => 4
| k + 1 => 2

-- Definitions of T_n
def T_n (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), (1 / (i + 3) * seq (i + 1))

-- Main theorem statements

-- (1) Arithmetic sequence
theorem arith_seq_proof : ∀ n : ℕ, seq (n + 1) - seq n = 4 + 2 * n := sorry

-- (2) General formula for the sequence
theorem gen_formula : ∀ n : ℕ, seq n = n * (n + 1) := sorry

-- (3) Range of T_n
theorem T_n_range : ∀ n : ℕ, (1 / 6 : ℝ) ≤ T_n n ∧ T_n n < (1 / 4 : ℝ) := sorry

end arith_seq_proof_gen_formula_T_n_range_l741_741775


namespace integer_values_abs_lt_2pi_l741_741686

theorem integer_values_abs_lt_2pi : 
  {x : ℤ | abs x < 2 * Real.pi}.finset.card = 13 := 
by
  sorry

end integer_values_abs_lt_2pi_l741_741686


namespace limit_C_prime_at_infinity_l741_741562

def C_prime (n : ℕ) (e R r : ℕ) : ℚ :=
  2 * e * n / (R + 2 * n * r)

theorem limit_C_prime_at_infinity :
  ∀ e R r : ℕ, e = 4 → R = 6 → r = 3 →
  (tendsto (λ n : ℕ, C_prime n e R r) at_top (pure (4/3))) :=
by
  intros e R r he hR hr
  have h1 : C_prime = λ n, 2 * e * n / (R + 2 * n * r), from rfl,
  rw [he, hR, hr],
  sorry

end limit_C_prime_at_infinity_l741_741562


namespace find_c_plus_d_l741_741699

noncomputable def y_satisfies (y : ℝ) : Prop :=
  y^2 - 5 * y + 5 / y + 1 / y^2 = 14

theorem find_c_plus_d :
  ∃ (c d : ℕ), (∃ y : ℝ, y_satisfies y ∧ y = c - real.sqrt d) ∧ c + d = 52 :=
by
  sorry

end find_c_plus_d_l741_741699


namespace angle_AKC_obtuse_l741_741936

def is_angle_bisector (C K : Point ℝ) (A B : Segment ℝ) : Prop :=
  ∃ α β, α + β = π ∧ α = β

axiom triangle_inequality (A B C : Point ℝ) (ACgtBC : AC > BC) : 
  angle CAB < angle CBA 

theorem angle_AKC_obtuse
  (A B C K : Point ℝ)
  (h1 : AC > BC)
  (h2 : is_angle_bisector C K A B) :
  angle AKC > π / 2 :=
sorry

end angle_AKC_obtuse_l741_741936


namespace probability_of_non_defective_product_l741_741153

-- Define the probability of producing a grade B product
def P_B : ℝ := 0.03

-- Define the probability of producing a grade C product
def P_C : ℝ := 0.01

-- Define the probability of producing a non-defective product (grade A)
def P_A : ℝ := 1 - P_B - P_C

-- The theorem to prove: The probability of producing a non-defective product is 0.96
theorem probability_of_non_defective_product : P_A = 0.96 := by
  -- Insert proof here
  sorry

end probability_of_non_defective_product_l741_741153


namespace volume_of_sphere_given_cube_volume_8_l741_741953

-- Given conditions
def cube_volume (a : ℝ) : ℝ := a^3
def cube_side_length := 2 -- Side length in cm since a^3 = 8
def sphere_diameter (a : ℝ) : ℝ := a * Math.sqrt 3
def sphere_radius (a : ℝ) : ℝ := (sphere_diameter a) / 2

-- Make the relevant definitions
def cube_with_volume_has_all_vertices_on_sphere (a : ℝ) (v_c : ℝ) (v_s : ℝ) : Prop :=
  cube_volume a = v_c ∧ sphere_volume (sphere_radius a) = v_s

-- Sphere volume calculation
def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Statement to be proved
theorem volume_of_sphere_given_cube_volume_8 : cube_with_volume_has_all_vertices_on_sphere cube_side_length 8 (4 * Real.sqrt 3 * Real.pi) :=
by
  -- include code to actually check the statement
  sorry

end volume_of_sphere_given_cube_volume_8_l741_741953


namespace inequality_of_nonnegative_f_l741_741383

variables {a b A B : ℝ}

noncomputable def f (θ : ℝ) : ℝ :=
  1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ)

theorem inequality_of_nonnegative_f (h : ∀ θ : ℝ, f θ ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end inequality_of_nonnegative_f_l741_741383


namespace problem_statement_l741_741307

theorem problem_statement (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_statement_l741_741307


namespace largest_common_in_range_l741_741985

theorem largest_common_in_range (a b c d : ℕ) (h1 : ∀ n : ℕ, a + n * b ∈ finset.Ico 1 151) (h2 : ∀ m : ℕ, c + m * d ∈ finset.Ico 1 151) : 
  let seq1 := λ n, 2 + n * 4 in
  let seq2 := λ m, 3 + m * 5 in
  { x | ∃ n, x = seq1 n } ∩ { x | ∃ m, x = seq2 m } ∩ finset.Ico 1 151 = { x | x = 138 } :=
by
  sorry

end largest_common_in_range_l741_741985


namespace min_distance_theorem_l741_741853

noncomputable def min_distance_from_point_on_curve_to_line : ℝ :=
  let curve : ℝ → ℝ := λ x, x^2
  let line : ℝ → ℝ := λ x, 2*x - 2
  let tangent_line : ℝ → ℝ := λ x, 2*x - 1 -- Tangent line to the curve
  let distance_between_parallel_lines : ℝ → ℝ → ℝ :=
    λ y₁ y₂, abs(y₂ - y₁) / sqrt (2^2 + 1^2) -- Distance formula for parallel lines
  in distance_between_parallel_lines (-2) (-1)

theorem min_distance_theorem : min_distance_from_point_on_curve_to_line = sqrt 5 / 5 := sorry

end min_distance_theorem_l741_741853


namespace find_least_value_of_diagonal_l741_741320

def least_diagonal_of_rectangle (perimeter : ℕ) (length_diff : ℕ) : ℝ :=
  let w := (perimeter - 2 * length_diff) / 4
  let l := w + length_diff
  Real.sqrt (l^2 + w^2)

theorem find_least_value_of_diagonal :
  least_diagonal_of_rectangle 30 3 = Real.sqrt 117 :=
by
  sorry

end find_least_value_of_diagonal_l741_741320


namespace relative_error_comparison_l741_741136

def discrepancy_first (error1 length1 : ℝ) : ℝ :=
  (error1 / length1) * 100

def discrepancy_second (error2 length2 : ℝ) : ℝ :=
  (error2 / length2) * 100

theorem relative_error_comparison (err1 len1 err2 len2 : ℝ)
  (h1 : err1 = 0.01) (h2 : len1 = 20) (h3 : err2 = 0.3) (h4 : len2 = 150) :
  discrepancy_second err2 len2 > discrepancy_first err1 len1 :=
by {
  rw [h1, h2, h3, h4],
  dsimp [discrepancy_first, discrepancy_second],
  norm_num,
  sorry
}

end relative_error_comparison_l741_741136


namespace simplify_sqrt_of_three_minus_pi_l741_741822

noncomputable def simplify_sqrt_expr (x : ℝ) : ℝ :=
  real.sqrt ((3 - x)^2)

theorem simplify_sqrt_of_three_minus_pi :
  simplify_sqrt_expr real.pi = real.pi - 3 :=
by sorry

end simplify_sqrt_of_three_minus_pi_l741_741822


namespace medians_concurrent_altitudes_concurrent_angle_bisectors_concurrent_l741_741253

-- Definition of the medians being concurrent
theorem medians_concurrent (a b c : Real) (A B C : Point) :
  is_concurrent (median A B C) :=
sorry

-- Definition of the altitudes being concurrent
theorem altitudes_concurrent (a b c : Real) (A B C : Point) :
  is_concurrent (altitude A B C) :=
sorry

-- Definition of the internal angle bisectors being concurrent
theorem angle_bisectors_concurrent (a b c : Real) (A B C : Point) :
  is_concurrent (angle_bisector A B C) :=
sorry

end medians_concurrent_altitudes_concurrent_angle_bisectors_concurrent_l741_741253


namespace minimum_throws_to_ensure_same_sum_twice_l741_741015

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741015


namespace bucket_weight_full_l741_741921

variable (p q x y : ℝ)

theorem bucket_weight_full (h1 : x + (3 / 4) * y = p)
                           (h2 : x + (1 / 3) * y = q) :
  x + y = (1 / 5) * (8 * p - 3 * q) :=
by
  sorry

end bucket_weight_full_l741_741921


namespace vasya_example_fewer_sevens_l741_741882

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l741_741882


namespace prove_f_pi_over_4_eq_sqrt_3_over_2_l741_741441

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem prove_f_pi_over_4_eq_sqrt_3_over_2
  (ω φ : ℝ)
  (h1 : ω > 0)
  (h2 : 0 < φ ∧ φ < π)
  (h3 : ∃ d : ℝ, d = π / 2 ∧ ∀ x, f ω φ (x + d / ω) = f ω φ x)
  (h4 : Real.tan φ = sqrt 3 / 3)
  : f ω φ (π / 4) = sqrt 3 / 2 :=
sorry

end prove_f_pi_over_4_eq_sqrt_3_over_2_l741_741441


namespace alice_needs_more_life_vests_l741_741543

-- Definitions based on the given conditions
def students : ℕ := 40
def instructors : ℕ := 10
def lifeVestsOnHand : ℕ := 20
def percentWithLifeVests : ℚ := 0.20

-- Statement of the problem
theorem alice_needs_more_life_vests :
  let totalPeople := students + instructors
  let lifeVestsBroughtByStudents := (percentWithLifeVests * students).toNat
  let totalLifeVestsAvailable := lifeVestsOnHand + lifeVestsBroughtByStudents
  totalPeople - totalLifeVestsAvailable = 22 :=
by
  sorry

end alice_needs_more_life_vests_l741_741543


namespace undetermined_people_count_l741_741537

theorem undetermined_people_count (athletes referees people : ℕ)
  (matches_per_referee photos_per_match photos_per_person games : ℕ) :
  athletes = 20 →
  referees = 10 →
  people = athletes + referees →
  matches_per_referee = 20 →
  photos_per_match = 3 →
  games = (athletes * (athletes - 1)) / 2 →
  photos_per_person = 20 →
  2 = (if ∃ x : ℕ, x ∈ {i ∈ finset.range people | (x < athletes ∧ ∀ x' < athletes, ∃ y < referees, y ≠ x')} then 2 else 0) :=
by sorry

end undetermined_people_count_l741_741537


namespace find_k_l741_741171

-- Define the conditions
variables (k : ℝ) -- the variable k
variables (x1 : ℝ) -- x1 coordinate of point A on the graph y = k/x
variable (AREA_ABCD : ℝ := 10) -- the area of the quadrilateral ABCD

-- The statement to be proven
theorem find_k (k : ℝ) (h1 : ∀ x1 : ℝ, (0 < x1 ∧ 2 * abs k = AREA_ABCD → x1 * abs k * 2 = AREA_ABCD)) : k = -5 :=
sorry

end find_k_l741_741171


namespace impossible_to_make_all_positive_l741_741341

def initial_board : Fin 8 × Fin 8 → ℤ
| ⟨7, 1⟩, _ => -1  -- b8 in chess notation translates to (7, 1) in 0-based indexing
| _, _ => 1

def flip_signs_in_row (board : Fin 8 × Fin 8 → ℤ) (i : Fin 8) : Fin 8 × Fin 8 → ℤ :=
  λ ⟨x, y⟩ => if x = i then -board ⟨x, y⟩ else board ⟨x, y⟩

def flip_signs_in_col (board : Fin 8 × Fin 8 → ℤ) (j : Fin 8) : Fin 8 × Fin 8 → ℤ :=
  λ ⟨x, y⟩ => if y = j then -board ⟨x, y⟩ else board ⟨x, y⟩

theorem impossible_to_make_all_positive :
  ¬ ∃ (ops : List (Fin 8 → Fin 8 → ℤ)), 
  let final_board := ops.foldl (λ board op => op board) initial_board in
  (∀ p : Fin 8 × Fin 8, final_board p > 0) :=
begin
  sorry
end

end impossible_to_make_all_positive_l741_741341


namespace thrown_away_oranges_l741_741162

theorem thrown_away_oranges (x : ℕ) (h : 40 - x + 7 = 10) : x = 37 :=
by sorry

end thrown_away_oranges_l741_741162


namespace area_arcsin_cos_eq_pi_sq_l741_741991

noncomputable def area_bounded_by_arcsin_cos : ℝ :=
  ∫ x in 0..2 * π, (Real.arcsin (Real.cos x))

theorem area_arcsin_cos_eq_pi_sq :
  (area_bounded_by_arcsin_cos = π^2) :=
  sorry

end area_arcsin_cos_eq_pi_sq_l741_741991


namespace Tyoma_wins_with_optimal_play_l741_741800

-- The initial number is 123456789 repeated 2015 times
def initial_board := List.replicate 2015 [1, 2, 3, 4, 5, 6, 7, 8, 9].join

-- Define the players winning conditions
def Petya_wins (d : ℕ) : Prop := d = 1 ∨ d = 4 ∨ d = 7
def Vasya_wins (d : ℕ) : Prop := d = 2 ∨ d = 5 ∨ d = 8
def Tyoma_wins (d : ℕ) : Prop := d = 3 ∨ d = 6 ∨ d = 9

-- Define the final winning condition based on the turns and initial board
theorem Tyoma_wins_with_optimal_play
  (initial_sum : ℕ := initial_board.sum)
  (sum_mod_3 : initial_sum % 3 = 0) :
  ∃ d, (d ∈ initial_board) → Tyoma_wins d :=
by { sorry }

end Tyoma_wins_with_optimal_play_l741_741800


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741891

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741891


namespace max_distance_l741_741347

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
(t, sqrt 3 * t + 2)

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
(Real.cos θ, sqrt 3 * Real.sin θ)

def dist_to_line (P : ℝ × ℝ) (A B C : ℝ) : ℝ :=
(abs (A * P.1 + B * P.2 + C)) / sqrt (A^2 + B^2)

theorem max_distance : ∀ θ : ℝ,
  let P := curve_C θ in
  dist_to_line P (sqrt 3) (-1) 2 ≤ (sqrt 6 + 2) / 2 :=
by
  sorry

end max_distance_l741_741347


namespace solution_to_equation_l741_741217

theorem solution_to_equation (x : ℝ) (h : (5 - x / 2)^(1/3) = 2) : x = -6 :=
sorry

end solution_to_equation_l741_741217


namespace find_angles_of_triangle_l741_741969

noncomputable def angles_of_triangle (a S : ℝ) : ℝ × ℝ × ℝ :=
  let alpha := 1 / 2 * Real.arctan (4 * S / a^2)
  (alpha, Real.pi / 2 + alpha, Real.pi / 2 - Real.arctan (4 * S / a^2))

theorem find_angles_of_triangle (a S : ℝ) :
  ∃ α β γ, α + β + γ = Real.pi 
    ∧ (α, β, γ) = angles_of_triangle a S
    ∧ |β - α| = Real.pi / 2 
    ∧ 1 / 2 * a^2 * (Real.sin α * Real.cos (β - α) + Real.sin β * Real.cos (α - β)) = S :=
by
  sorry

end find_angles_of_triangle_l741_741969


namespace triangle_inequality_check_l741_741981

theorem triangle_inequality_check :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ↔
    ((a = 6 ∧ b = 9 ∧ c = 14) ∨ (a = 9 ∧ b = 6 ∧ c = 14) ∨ (a = 6 ∧ b = 14 ∧ c = 9) ∨
     (a = 14 ∧ b = 6 ∧ c = 9) ∨ (a = 9 ∧ b = 14 ∧ c = 6) ∨ (a = 14 ∧ b = 9 ∧ c = 6)) := sorry

end triangle_inequality_check_l741_741981


namespace cos_double_angle_l741_741617

theorem cos_double_angle (α : ℝ) : (tan α = 2) → cos (2 * α) = -3 / 5 := 
by
  sorry

end cos_double_angle_l741_741617


namespace num_two_digit_numbers_A_squared_congruent_one_mod_15_l741_741637

theorem num_two_digit_numbers_A_squared_congruent_one_mod_15 : 
  {A : ℕ // 10 ≤ A ∧ A < 100 ∧ (A^2 % 15 = 1)}.card = 24 :=
by
  sorry

end num_two_digit_numbers_A_squared_congruent_one_mod_15_l741_741637


namespace solve_fraction_zero_l741_741859

theorem solve_fraction_zero (x : ℝ) (h1 : (x^2 - 25) / (x + 5) = 0) (h2 : x ≠ -5) : x = 5 :=
sorry

end solve_fraction_zero_l741_741859


namespace ellipse_parabola_line_existence_l741_741645

theorem ellipse_parabola_line_existence :
  (∃ (a c b : Real), a = 2 ∧ c = sqrt 3 ∧ b^2 = a^2 - c^2 ∧
   ∀ x y, (a ≠ 0 ∧ c ≠ 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧
      (x^2 = 4 * (y - 1))) ∧
   ∃ k, ∀ x y, (y = k * (x + 1)) ∧ 
         (∀ x1 y1 x2 y2, y1 = (1 / 4) * x1^2 ∧ y2 = (1 / 4) * x2^2 ∧
         (1 / 2) * x1 * (1 / 2) * x2 = -1 → x1 * x2 = -4 → k = 1 ∧
         y = x + 1)) :=
sorry

end ellipse_parabola_line_existence_l741_741645


namespace triangle_side_c_l741_741714

noncomputable def angle_B_eq_2A (A B : ℝ) := B = 2 * A
noncomputable def side_a_eq_1 (a : ℝ) := a = 1
noncomputable def side_b_eq_sqrt3 (b : ℝ) := b = Real.sqrt 3

noncomputable def find_side_c (A B C a b c : ℝ) :=
  angle_B_eq_2A A B ∧
  side_a_eq_1 a ∧
  side_b_eq_sqrt3 b →
  c = 2

theorem triangle_side_c (A B C a b c : ℝ) : find_side_c A B C a b c :=
by sorry

end triangle_side_c_l741_741714


namespace problem1_problem2_l741_741996

theorem problem1 : 101 * 99 = 9999 := 
by sorry

theorem problem2 : 32 * 2^2 + 14 * 2^3 + 10 * 2^4 = 400 := 
by sorry

end problem1_problem2_l741_741996


namespace units_digit_of_2_pow_2012_l741_741261

theorem units_digit_of_2_pow_2012 : 
  let units_digit_cycle := [2, 4, 8, 6] in
  (units_digit_cycle[(2012 % 4)]) = 6 := 
by
  sorry

end units_digit_of_2_pow_2012_l741_741261


namespace average_interest_rate_l741_741960

theorem average_interest_rate 
  (total : ℝ)
  (rate1 rate2 yield1 yield2 : ℝ)
  (amount1 amount2 : ℝ)
  (h_total : total = amount1 + amount2)
  (h_rate1 : rate1 = 0.03)
  (h_rate2 : rate2 = 0.07)
  (h_yield_equal : yield1 = yield2)
  (h_yield1 : yield1 = rate1 * amount1)
  (h_yield2 : yield2 = rate2 * amount2) :
  (yield1 + yield2) / total = 0.042 :=
by
  sorry

end average_interest_rate_l741_741960


namespace correct_options_l741_741649

theorem correct_options (a b c : ℝ) (h1 : ∀ x : ℝ, (a*x^2 + b*x + c > 0) ↔ (-3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, (b*x + c > 0) ↔ x > 6) = False ∧ (∀ x, (c*x^2 + b*x + a < 0) ↔ (-1/3 < x ∧ x < 1/2)) :=
by 
  sorry

end correct_options_l741_741649


namespace angle_CBM_eq_30_l741_741351

theorem angle_CBM_eq_30
  (A B C A1 B1 C1 M : Type*)
  (angle_B : ℝ) (angle_B_eq_120 : angle_B = 120)
  (is_angle_bisector_AA1 : Type*)
  (is_angle_bisector_BB1 : Type*)
  (is_angle_bisector_CC1 : Type*)
  (A1B1_intersects_CC1_at_M : Type*) :
  angle_B = 120 → 
  is_angle_bisector_AA1 = AA1 →
  is_angle_bisector_BB1 = BB1 →
  is_angle_bisector_CC1 = CC1 →
  A1B1_intersects_CC1_at_M = M →
  ∠C B M = 30 := 
by
  sorry

end angle_CBM_eq_30_l741_741351


namespace union_eq_301_l741_741943

-- Define the given sets P and Q based on conditions
def P (a : ℝ) : Set ℝ := {3, Real.logBase 2 a}
def Q (a b : ℝ) : Set ℝ := {a, b}

-- Given conditions
variables (a b : ℝ) (h_inter : P a ∩ Q a b = {0})

-- Main proof statement
theorem union_eq_301 (h_inter : P a ∩ Q a b = {0}) :
  P a ∪ Q a b = {3, 0, 1} :=
sorry

end union_eq_301_l741_741943


namespace sadeh_polynomial_exists_l741_741473

-- A polynomial S(x) ∈ ℝ[x] is called sadeh if it is divisible by x but not by x²
def is_sadeh (S : Polynomial ℝ) : Prop :=
  S.coeff 0 = 0 ∧ S.coeff 1 ≠ 0

-- Given P ∈ ℝ[x] and there exists a sadeh polynomial Q such that P(Q(x)) - Q(2x) is divisible by x²,
-- prove that there exists a sadeh polynomial R such that P(R(x)) - R(2x) is divisible by x¹⁴⁰¹
theorem sadeh_polynomial_exists (P : Polynomial ℝ) :
  (∃ Q : Polynomial ℝ, is_sadeh Q ∧ (P.eval₂ Polynomial.C Q - Q.eval (2 : ℝ)).coeff 2 = 0) →
  ∃ R : Polynomial ℝ, is_sadeh R ∧ (P.eval₂ Polynomial.C R - R.eval (2 : ℝ)).coeff 1401 = 0 :=
  sorry

end sadeh_polynomial_exists_l741_741473


namespace find_x_for_h_eq_20_l741_741828

-- Given conditions as definitions in Lean
def h (x : ℝ) : ℝ := 4 * (f⁻¹ x)
def f (x : ℝ) : ℝ := 30 / (x + 2)

-- Lean theorem statement
theorem find_x_for_h_eq_20 (x : ℝ) (h_eq_20 : h x = 20) : x = 30 / 7 := 
by sorry

end find_x_for_h_eq_20_l741_741828


namespace shared_property_of_shapes_l741_741090

-- Definition of each shape
def is_parallelogram (P : Type) [AddCommGroup P] [Module ℝ P] (a b : P) : Prop :=
  -- Assuming some abstract properties that define a parallelogram.
  sorry

def is_rectangle (R : Type) [AddCommGroup R] [Module ℝ R] (a b : R) [is_parallelogram R a b] : Prop :=
  -- Assuming some abstract properties that define a rectangle.
  sorry

def is_rhombus (H : Type) [AddCommGroup H] [Module ℝ H] (a b : H) [is_parallelogram H a b] : Prop :=
  -- Assuming some abstract properties that define a rhombus.
  sorry

def is_square (S : Type) [AddCommGroup S] [Module ℝ S] (a b : S) [is_rectangle S a b] [is_rhombus S a b] : Prop :=
  -- Assuming some abstract properties that define a square.
  sorry

-- Property to be proven
theorem shared_property_of_shapes
    (P : Type) [AddCommGroup P] [Module ℝ P] (a b : P)
    (hP : is_parallelogram P a b)
    (R : Type) [AddCommGroup R] [Module ℝ R] (c d : R)
    (hR : is_rectangle R c d)
    (H : Type) [AddCommGroup H] [Module ℝ H] (e f : H)
    (hH : is_rhombus H e f)
    (S : Type) [AddCommGroup S] [Module ℝ S] (g k : S)
    (hS : is_square S g k) : 
    (∀ (U : Type) [AddCommGroup U] [Module ℝ U] (x y : U), (is_parallelogram U x y) → opposite_sides_parallel_and_equal U x y) :=
  sorry

end shared_property_of_shapes_l741_741090


namespace min_throws_to_ensure_same_sum_twice_l741_741071

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741071


namespace integer_values_sides_triangle_l741_741851

theorem integer_values_sides_triangle (x : ℝ) (hx_pos : x > 0) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) : 
    (∃ (n : ℤ), ∃ (hn : 0 < n) (hn1 : (n : ℝ) = x) (hn2 : 26 ≤ n) (hn3 : n ≤ 54), 
    ∀ (y : ℤ), (26 ≤ y ∧ y ≤ 54) → (∃ (m : ℤ), y = 26 + m ∧ m < 29 ∧ m ≥ 0)) := 
sorry

end integer_values_sides_triangle_l741_741851


namespace range_of_x_for_inequality_l741_741229

variable (f : ℝ → ℝ)

noncomputable def f' (x : ℝ) : ℝ := (deriv f) x

axiom f_defined_on_Reals (x : ℝ) : true
axiom f_derivative_defined (x : ℝ) : has_deriv_at f (f' x) x
axiom f_zero_at_0 : f 0 = 0
axiom f_inequality (x : ℝ) : f x > f' x + 1

theorem range_of_x_for_inequality : {x : ℝ | f x + exp x < 1} = Ioi 0 := 
by
  -- proof goes here
  sorry

end range_of_x_for_inequality_l741_741229


namespace solution_l741_741493

-- Define the conditions as hypotheses
def condition1 (x : ℝ) : Prop :=
  log 4 (4*x - 1)/(x + 1) > 0

def condition2 (x : ℝ) : Prop :=
  log (1/4) (x + 1)/(4*x - 1) > 0

def log_inequality (x : ℝ) : Prop :=
  log 3 (log 4 (4*x - 1)/(x + 1)) - log (1/3) (log (1/4) (x + 1)/(4*x - 1)) < 0

-- Lean statement to prove the equivalent problem
theorem solution (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x > 2/3 := 
sorry

end solution_l741_741493


namespace smallest_number_with_55_divisors_l741_741591

theorem smallest_number_with_55_divisors : ∃ n : ℕ, 
  (number_of_divisors n = 55) ∧ (∀ m : ℕ, number_of_divisors m = 55 → n ≤ m) := 
sorry

end smallest_number_with_55_divisors_l741_741591


namespace min_throws_to_ensure_same_sum_twice_l741_741074

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741074


namespace find_fraction_l741_741481

theorem find_fraction (x : ℝ) (h1 : 7 = (1 / 10) / 100 * 7000) (h2 : x * 7000 - 7 = 700) : x = 707 / 7000 :=
by sorry

end find_fraction_l741_741481


namespace temperature_decrease_2C_l741_741710

variable (increase_3 : ℤ := 3)
variable (decrease_2 : ℤ := -2)

theorem temperature_decrease_2C :
  decrease_2 = -2 :=
by
  -- This is where the proof would go
  sorry

end temperature_decrease_2C_l741_741710


namespace distance_calculation_l741_741741

-- Define the rates and directions of movement
def jay_speed : ℝ := 1.1 / 20
def paul_speed : ℝ := 3.1 / 45
def anne_speed : ℝ := 0.9 / 30
def time_duration : ℝ := 180

-- Calculate distances covered
def jay_distance : ℝ := jay_speed * time_duration
def paul_distance : ℝ := paul_speed * time_duration
def anne_distance : ℝ := anne_speed * time_duration

-- Prove the distances between Jay and Paul, and Anne's distance from their line
theorem distance_calculation :
  jay_distance = 9.9 ∧
  paul_distance = 12.4 ∧
  (jay_distance + paul_distance = 22.3) ∧
  (anne_distance = 5.4) ∧
  (0 = 0) := 
by
  sorry

end distance_calculation_l741_741741


namespace part_a_part_b_l741_741505

open Real

theorem part_a (n : ℕ) : sqrt (↑n + 1) - sqrt (↑n) < 1 / (2 * sqrt (↑n)) ∧ 1 / (2 * sqrt (↑n)) < sqrt (↑n) - sqrt (↑n - 1) :=
by
  sorry

theorem part_b (m : ℕ) : floor (∑ k in finset.range (m ^ 2 + 1), 1 / sqrt k) = 2 * m - 2 ∨ floor (∑ k in finset.range (m ^ 2 + 1), 1 / sqrt k) = 2 * m - 1 :=
by
  sorry

end part_a_part_b_l741_741505


namespace polynomial_remainder_l741_741608

theorem polynomial_remainder (b : ℚ) : 
  ∃ b, (∃ q : polynomial ℚ, 
  polynomial.divByX_addC (3 * (polynomial.X ^ 3) + b * (polynomial.X ^ 2) + 17 * polynomial.X - 53) 3 5 = (q, -7)) :=
begin
  use -4 / 5,
  sorry
end

end polynomial_remainder_l741_741608


namespace Shekar_weighted_average_is_correct_l741_741816

-- Define the scores and weightages
def scores : List ℝ := [76, 65, 82, 67, 75, 89, 71, 78, 63, 55]
def weightages : List ℝ := [0.15, 0.10, 0.15, 0.15, 0.05, 0.05, 0.10, 0.05, 0.10, 0.10]

-- Define a function to calculate the weighted average
def weighted_average (scores : List ℝ) (weights : List ℝ) : ℝ :=
  (List.zipWith (· * ·) scores weights).sum / weights.sum

-- Prove that the calculated weighted average is 71.25
theorem Shekar_weighted_average_is_correct :
  weighted_average scores weightages = 71.25 := 
by
  sorry

end Shekar_weighted_average_is_correct_l741_741816


namespace minimum_rolls_for_duplicate_sum_l741_741062

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741062


namespace min_value_f_x_gt_1_min_value_a_f_x_lt_1_l741_741280

def f (x : ℝ) : ℝ := 4 * x + 1 / (x - 1)

theorem min_value_f_x_gt_1 : 
  (∀ x : ℝ, x > 1 → f x ≥ 8) ∧ (∃ x : ℝ, x > 1 ∧ f x = 8) := 
by 
  sorry

theorem min_value_a_f_x_lt_1 : 
  (∀ x : ℝ, x < 1 → f x ≤ 0) ∧ (∀ a : ℝ, (∀ x : ℝ, x < 1 → f x ≤ a) → a ≥ 0 ∧ (∃ x : ℝ, f x = 0)) := 
by 
  sorry

end min_value_f_x_gt_1_min_value_a_f_x_lt_1_l741_741280


namespace count_valid_digits_l741_741875

theorem count_valid_digits (n : ℕ) (h : n ≥ 2) :
  let digit_options := [1, 2, 3, 4],
  let valid_numbers := {num : ℕ | 
    num.digits 10 ∘ length = n ∧ 
    num.digits 10 ∘ all (∈ digit_options) ∧ 
    (count num.digits 10 1) % 2 = 1 ∧ 
    (count num.digits 10 2) % 2 = 0 ∧ 
    (count num.digits 10 3) > 0
  },
  let total_valid_numbers :=  
    (4^n - 3^n + (-1)^n)/4
  in
    card valid_numbers = total_valid_numbers := sorry

end count_valid_digits_l741_741875


namespace cos_segments_ratio_proof_l741_741836

open Real

noncomputable def cos_segments_ratio := 
  let p := 5
  let q := 26
  ∀ x : ℝ, (cos x = cos 50) → (p, q) = (5, 26)

theorem cos_segments_ratio_proof : cos_segments_ratio :=
by 
  sorry

end cos_segments_ratio_proof_l741_741836


namespace solve_camel_cost_l741_741945

noncomputable def cost_problem : Type := ℝ

def camel_cost (C H O E L B : cost_problem) : Prop :=
  (10 * C = 24 * H) ∧
  (16 * H = 4 * O) ∧
  (6 * O = 4 * E) ∧
  (3 * E = 8 * L) ∧
  (2 * L = 6 * B) ∧
  (14 * B = 204000)

def camel_cost_is_46542_86 (C H O E L B : cost_problem) (h : camel_cost C H O E L B) : Prop :=
  C = 46542.86

theorem solve_camel_cost :
  ∃ (C H O E L B : cost_problem), camel_cost C H O E L B ∧ camel_cost_is_46542_86 C H O E L B :=
sorry

end solve_camel_cost_l741_741945


namespace find_lambda_l741_741242

-- Definitions
def vec_a (λ : ℝ) : ℝ × ℝ := (2, λ)
def vec_b : ℝ × ℝ := (3, 4)

-- Given condition: vectors are perpendicular
def is_perpendicular (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2) = 0

-- Theorem statement
theorem find_lambda (λ : ℝ) : is_perpendicular (vec_a λ) vec_b → λ = -3/2 := by
  sorry

end find_lambda_l741_741242


namespace minimum_throws_to_ensure_same_sum_twice_l741_741014

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l741_741014


namespace find_side_b_l741_741640

variables {A B C a b c x : ℝ}

theorem find_side_b 
  (cos_A : ℝ) (cos_C : ℝ) (a : ℝ) (hcosA : cos_A = 4/5) 
  (hcosC : cos_C = 5/13) (ha : a = 1) : 
  b = 21/13 :=
by
  sorry

end find_side_b_l741_741640


namespace math_problem_l741_741773

open Real

noncomputable def f (x : ℝ) : ℝ := sin (π * x / 2) ^ 2
noncomputable def g (x : ℝ) : ℝ := 1 - abs (x - 1)

theorem math_problem (x : ℝ) (k : ℤ) :
  (∀ x, f (x) = f (2 * k - x)) ∧
  (∀ x, 1 < x ∧ x < 2 → ∃ x1 x2, x1 < x2 ∧ f x1 > f x ∧ f x2 < f x) ∧
  (∀ x, f (x - 1) + g (x - 1) = f (1 - x) + g (1 - x)) ∧
  (∀ x, f (x) ≤ 1 ∧ g (x) ≤ 1 ∧ f 1 + g 1 = 2)
:=
begin
  -- Proof for each statement here
  sorry,
end

end math_problem_l741_741773


namespace melanie_batches_l741_741788

theorem melanie_batches (total_brownies_given: ℕ)
                        (brownies_per_batch: ℕ)
                        (fraction_bake_sale: ℚ)
                        (fraction_container: ℚ)
                        (remaining_brownies_given: ℕ) :
                        brownies_per_batch = 20 →
                        fraction_bake_sale = 3/4 →
                        fraction_container = 3/5 →
                        total_brownies_given = 20 →
                        (remaining_brownies_given / (brownies_per_batch * (1 - fraction_bake_sale) * (1 - fraction_container))) = 10 :=
by
  sorry

end melanie_batches_l741_741788


namespace range_of_a_l741_741260

noncomputable def set_A : Set ℝ := { x | x^2 - 3 * x - 10 < 0 }
noncomputable def set_B : Set ℝ := { x | x^2 + 2 * x - 8 > 0 }
def set_C (a : ℝ) : Set ℝ := { x | 2 * a < x ∧ x < a + 3 }

theorem range_of_a (a : ℝ) :
  (A ∩ B) ∩ set_C a = set_C a → 1 ≤ a := 
sorry

end range_of_a_l741_741260


namespace min_sum_distances_convex_quadrilateral_l741_741923

theorem min_sum_distances_convex_quadrilateral (A B C D P : Point) 
  (h_convex : is_convex_quadrilateral A B C D) 
  (h_inter : is_intersection_point_of_diagonals P A B C D) : 
  minimizes_sum_distances P A B C D := 
sorry

end min_sum_distances_convex_quadrilateral_l741_741923


namespace fewerSevensCanProduce100_l741_741912

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l741_741912


namespace frustum_properties_l741_741268

noncomputable def frustum_slant_height (r1 r2 : ℝ) (A : ℝ) : ℝ :=
  (A / (π * (r1 + r2)))

noncomputable def frustum_volume (r1 r2 h : ℝ) : ℝ :=
  (π * h * (r1^2 + r2^2 + r1 * r2)) / 3

theorem frustum_properties :
  let r1 := 2
  let r2 := 6
  let l := frustum_slant_height r1 r2 ((π * r1^2) + (π * r2^2))
  let h := Real.sqrt (l^2 - (r2 - r1)^2)
  l = 5 ∧ frustum_volume r1 r2 h = 52 * π :=
by
  sorry

end frustum_properties_l741_741268


namespace prove_y_l741_741107

-- Define the conditions
variables (x y : ℤ) -- x and y are integers

-- State the problem conditions
def conditions := (x + y = 270) ∧ (x - y = 200)

-- Define the theorem to prove that y = 35 given the conditions
theorem prove_y : conditions x y → y = 35 :=
by
  sorry

end prove_y_l741_741107


namespace notebook_problem_l741_741796

theorem notebook_problem
    (total_notebooks : ℕ)
    (cost_price_A : ℕ)
    (cost_price_B : ℕ)
    (total_cost_price : ℕ)
    (selling_price_A : ℕ)
    (selling_price_B : ℕ)
    (discount_A : ℕ)
    (profit_condition : ℕ)
    (x y m : ℕ) 
    (h1 : total_notebooks = 350)
    (h2 : cost_price_A = 12)
    (h3 : cost_price_B = 15)
    (h4 : total_cost_price = 4800)
    (h5 : selling_price_A = 20)
    (h6 : selling_price_B = 25)
    (h7 : discount_A = 30)
    (h8 : 12 * x + 15 * y = 4800)
    (h9 : x + y = 350)
    (h10 : selling_price_A * m + selling_price_B * m + (x - m) * selling_price_A * 7 / 10 + (y - m) * cost_price_B - total_cost_price ≥ profit_condition):
    x = 150 ∧ m ≥ 128 :=
by
    sorry

end notebook_problem_l741_741796


namespace x_coordinate_of_P_l741_741732

noncomputable section

open Real

-- Define the standard properties of the parabola and point P
def parabola (p : ℝ) (x y : ℝ) := (y ^ 2 = 4 * x)

def distance (P F : ℝ × ℝ) : ℝ := 
  sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

-- Position of the focus for the given parabola y^2 = 4x; Focus F(1, 0)
def focus : ℝ × ℝ := (1, 0)

-- The given conditions translated into Lean form
def on_parabola (x y : ℝ) := parabola 2 x y ∧ distance (x, y) focus = 5

-- The theorem we need to prove: If point P satisfies these conditions, then its x-coordinate is 4
theorem x_coordinate_of_P (P : ℝ × ℝ) (h : on_parabola P.1 P.2) : P.1 = 4 :=
by
  sorry

end x_coordinate_of_P_l741_741732


namespace problem_solution_l741_741251

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a 1 + (∑ i in finset.range (n - 1), 2^i * a (i + 2)) = n * 2^(n + 1)

theorem problem_solution (a : ℕ → ℕ) (h : sequence a) :
  (∀ n, a n = 2 * n + 2) ∧
  (∀ n, ∑ i in finset.range n, a (i + 1) = n * (n + 3)) ∧
  (∑ i in finset.range 20, abs (a (i + 1) - 10) = 284) :=
by
  sorry

end problem_solution_l741_741251


namespace intersection_problem_l741_741924

noncomputable def curve_intersection_points (a : ℝ) : ℕ :=
  let f := λ x : ℝ, x^4 - (8 * a - 1) * x^2 + (12 * a^2 - 8 * a + 1) - 4 * a^2
  (f.realRoot .connected_COMPONENT { x | x.realRoot f } ∩ { x | x ≠ 0 }).card

theorem intersection_problem (a : ℝ) : 
  (curve_intersection_points a = 3) ↔ (a > 1 / 8) := sorry

end intersection_problem_l741_741924


namespace correct_statement_d_l741_741982

theorem correct_statement_d :
  let StudentSampling (n m : ℕ) :=
        (m ∈ finset.range 1 (n + 1) ∧ (1 ≤ m ∧ m ≤ 50) 
        ∧ ∀ k : ℕ, (m + 50*k) ∈ finset.range 1 (n + 1))
      ∧ ∀ s ∈ finset.range 1 (n + 1), 50 ∣ (s -m)  :=
  let RegressionLinePassThroughCenter (b a x̄ ȳ : ℝ) :=
        ∀ (x : ℝ), ∃ y : ℝ, y = b * x + a ∧ ȳ = b * x̄ + a :=
  let CorrelationAbsLessThanOne (r : ℝ) :=
        (abs (r) ≤ 1)  :=
  let NormalDistributionProperty (μ σ : ℝ) (X : ℝ) :=
        μ = 10 ∧ σ = 0.1 ∧  P (X > 10) = 1/2 :=
        True :=
  StudentSampling 2000 m ∧
  RegressionLinePassThroughCenter b a x̄ ȳ ∧
  CorrelationAbsLessThanOne r ∧
  NormalDistributionProperty 10 0.1 X 10 Σ
  → NormalDistributionProperty 10 0.01 X ∧
     P (X > 10) = 1/2
:= sorry

end correct_statement_d_l741_741982


namespace find_an_expression_l741_741657

def seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, S n = (∑ i in finset.range n, a (i + 1)))
  ∧ (∀ n : ℕ, 2 * S n = a (n + 1))

theorem find_an_expression {a : ℕ → ℕ} {S : ℕ → ℕ} 
  (h : seq a S) (n : ℕ) : a (n + 1) = 2 * 3^(n - 1) :=
begin
  sorry,
end

end find_an_expression_l741_741657


namespace sum_of_tetrahedron_edge_angles_gt_540_l741_741806

theorem sum_of_tetrahedron_edge_angles_gt_540 (A B C D O : Point) (h : point_inside_tetrahedron A B C D O) : 
  ( ∠AOB + ∠AOC + ∠BOC + ∠AOD + ∠BOD + ∠COD ) > 540 := 
sorry

end sum_of_tetrahedron_edge_angles_gt_540_l741_741806


namespace circle_passing_through_points_eq_l741_741930

theorem circle_passing_through_points_eq :
  let A := (-2, 1)
  let B := (9, 3)
  let C := (1, 7)
  let center := (7/2, 2)
  let radius_sq := 125 / 4
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius_sq ↔ 
    (∃ t : ℝ, (x - center.1)^2 + (y - center.2)^2 = t^2) ∧
    ∀ P : ℝ × ℝ, P = A ∨ P = B ∨ P = C → (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius_sq := by sorry

end circle_passing_through_points_eq_l741_741930


namespace compute_expression_at_three_l741_741999

theorem compute_expression_at_three (x : ℤ) (hx : x = 3) : ((x^8 - 32 * x^4 + 256) / (x^4 - 8) = 65) :=
by
  rw hx
  sorry

end compute_expression_at_three_l741_741999


namespace lucas_change_l741_741783

def initialAmount : ℕ := 20
def costPerAvocado : ℕ := 2
def numberOfAvocados : ℕ := 3

def totalCost : ℕ := numberOfAvocados * costPerAvocado
def change : ℕ := initialAmount - totalCost

theorem lucas_change : change = 14 := by
  sorry

end lucas_change_l741_741783


namespace james_marbles_left_l741_741361

theorem james_marbles_left (initial_marbles : ℕ) (total_bags : ℕ) (marbles_per_bag : ℕ) (bags_given_away : ℕ) : 
  initial_marbles = 28 → total_bags = 4 → marbles_per_bag = initial_marbles / total_bags → bags_given_away = 1 → 
  initial_marbles - marbles_per_bag * bags_given_away = 21 :=
by
  intros h_initial h_total h_each h_given
  sorry

end james_marbles_left_l741_741361


namespace orthocenter_of_triangle_ABC_l741_741332

noncomputable def point : Type := (ℝ × ℝ × ℝ)

def A : point := (2, 3, 4)
def B : point := (6, 4, 2)
def C : point := (4, 5, 6)
def orthocenter : point := (5 / 2, 3, 7 / 2)

theorem orthocenter_of_triangle_ABC (H : point) : 
  H = orthocenter ↔ (H = (5 / 2, 3, 7 / 2)) :=
begin
  sorry -- Proof
end

end orthocenter_of_triangle_ABC_l741_741332


namespace min_throws_to_ensure_same_sum_twice_l741_741072

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741072


namespace smallest_number_with_55_divisors_l741_741605

theorem smallest_number_with_55_divisors :
  ∃ n : ℕ, (n = 3^4 * 2^{10}) ∧ (nat.count_divisors n = 55) :=
by
  have n : ℕ := 3^4 * 2^{10}
  exact ⟨n, ⟨rfl, nat.count_divisors_eq_count_divisors 3 4 2 10⟩⟩
  sorry

end smallest_number_with_55_divisors_l741_741605


namespace BE_plus_DE_geq_AC_l741_741623

variables {V : Type*} [add_comm_group V] [module ℝ V]

structure CyclicQuadrilateral (A B C D : V) : Prop :=
(cyclic : ∃ (O : V), ∃ (r : ℝ), (A - O).norm = r ∧ (B - O).norm = r ∧ (C - O).norm = r ∧ (D - O).norm = r)

def midpoint (x y : V) : V := (x + y) / 2

variables (A B C D : V) (E : V)
variables (h_cyclic : CyclicQuadrilateral A B C D)
variables (h_eq_bc_cd : (B - C).norm = (C - D).norm)
variables (h_midpoint : E = midpoint A C)

theorem BE_plus_DE_geq_AC : (B - E).norm + (D - E).norm ≥ (A - C).norm :=
sorry

end BE_plus_DE_geq_AC_l741_741623


namespace total_surface_area_first_rectangular_parallelepiped_equals_22_l741_741187

theorem total_surface_area_first_rectangular_parallelepiped_equals_22
  (x y z : ℝ)
  (h1 : (x + 1) * (y + 1) * (z + 1) = x * y * z + 18)
  (h2 : 2 * ((x + 1) * (y + 1) + (y + 1) * (z + 1) + (z + 1) * (x + 1)) = 2 * (x * y + x * z + y * z) + 30) :
  2 * (x * y + x * z + y * z) = 22 := sorry

end total_surface_area_first_rectangular_parallelepiped_equals_22_l741_741187


namespace problem1_problem2_l741_741254

-- Define the triangle and the condition a + 2a * cos B = c
variable {A B C : ℝ} (a b c : ℝ)
variable (cos_B : ℝ) -- cosine of angle B

-- Condition: a + 2a * cos B = c
variable (h1 : a + 2 * a * cos_B = c)

-- (I) Prove B = 2A
theorem problem1 (h1 : a + 2 * a * cos_B = c) : B = 2 * A :=
sorry

-- Define the acute triangle condition
variable (Acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)

-- Given: c = 2
variable (h2 : c = 2)

-- (II) Determine the range for a if the triangle is acute and c = 2
theorem problem2 (h1 : a + 2 * a * cos_B = 2) (Acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) : 1 < a ∧ a < 2 :=
sorry

end problem1_problem2_l741_741254


namespace smallest_number_with_55_divisors_l741_741599

theorem smallest_number_with_55_divisors : ∃ (n : ℕ), (∃ (p : ℕ → ℕ) (k : ℕ → ℕ) (m : ℕ), 
  n = ∏ i in finset.range m, (p i)^(k i) ∧ (∀ i j, i ≠ j → nat.prime (p i) → nat.prime (p j) → p i ≠ p j) ∧ 
  (finset.range m).card = m ∧ 
  (∏ i in finset.range m, (k i + 1) = 55)) ∧ 
  n = 3^4 * 2^10 then n = 82944 :=
by
  sorry

end smallest_number_with_55_divisors_l741_741599


namespace volume_of_inequality_region_l741_741088

-- Define the inequality condition as a predicate
def region (x y z : ℝ) : Prop :=
  |4 * x - 20| + |3 * y + 9| + |z - 2| ≤ 6

-- Define the volume calculation for the region
def volume_of_region := 36

-- The proof statement
theorem volume_of_inequality_region : 
  (∃ x y z : ℝ, region x y z) → volume_of_region = 36 :=
by
  sorry

end volume_of_inequality_region_l741_741088


namespace smallest_number_with_55_divisors_l741_741604

theorem smallest_number_with_55_divisors :
  ∃ n : ℕ, (n = 3^4 * 2^{10}) ∧ (nat.count_divisors n = 55) :=
by
  have n : ℕ := 3^4 * 2^{10}
  exact ⟨n, ⟨rfl, nat.count_divisors_eq_count_divisors 3 4 2 10⟩⟩
  sorry

end smallest_number_with_55_divisors_l741_741604


namespace smallest_possible_T_l741_741568

theorem smallest_possible_T :
  ∃ b : Fin 100 → ℤ, (∀ i, b i = 1 ∨ b i = -1) ∧ (let T := ∑ i in Finset.range 100, ∑ j in Finset.range i, b i * b j in T = 22) :=
sorry

end smallest_possible_T_l741_741568


namespace shorter_side_is_8_l741_741104

-- Define the conditions
variables {L W : ℝ}
def area_condition : Prop := L * W = 104
def perimeter_condition : Prop := L + W = 21

-- The theorem statement
theorem shorter_side_is_8 (h1 : area_condition) (h2 : perimeter_condition) : min L W = 8 := 
by sorry

end shorter_side_is_8_l741_741104


namespace arithmetic_mean_bound_l741_741940

open_locale big_operators

theorem arithmetic_mean_bound (n : ℕ) (h₁ : 0 < n) :
  1 ≤ (1 / n) * ∑ k in finset.range (n + 1), real.rpow k (1 / k) ∧ 
  (1 / n) * ∑ k in finset.range (n + 1), real.rpow k (1 / k) ≤ 1 + 2 * real.sqrt 2 / real.sqrt n :=
begin
  sorry,
end

end arithmetic_mean_bound_l741_741940


namespace bank_tellers_have_total_coins_l741_741209

theorem bank_tellers_have_total_coins :
  (∃ (num_tellers num_rolls_per_teller coins_per_roll : ℕ)
     (total_coins_one_teller total_coins_all_tellers : ℕ),
    num_tellers = 4 ∧ num_rolls_per_teller = 10 ∧ coins_per_roll = 25 ∧
    total_coins_one_teller = num_rolls_per_teller * coins_per_roll ∧
    total_coins_all_tellers = num_tellers * total_coins_one_teller ∧
    total_coins_all_tellers = 1000) :=
begin
  use [4, 10, 25, 250, 1000],
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  split,
  sorry,
end

end bank_tellers_have_total_coins_l741_741209


namespace lucas_change_l741_741784

-- Define the initial amount of money Lucas has
def initial_amount : ℕ := 20

-- Define the cost of one avocado
def cost_per_avocado : ℕ := 2

-- Define the number of avocados Lucas buys
def number_of_avocados : ℕ := 3

-- Calculate the total cost of avocados
def total_cost : ℕ := number_of_avocados * cost_per_avocado

-- Calculate the remaining amount of money (change)
def remaining_amount : ℕ := initial_amount - total_cost

-- The proposition to prove: Lucas brings home $14
theorem lucas_change : remaining_amount = 14 := by
  sorry

end lucas_change_l741_741784


namespace find_parabola_eq_find_range_of_b_l741_741267

-- Problem 1: Finding the equation of the parabola
theorem find_parabola_eq (p : ℝ) (h1 : p > 0) (x1 x2 y1 y2 : ℝ) 
  (A : (x1 + 4) * 2 = 2 * p * y1) (C : (x2 + 4) * 2 = 2 * p * y2)
  (h3 : x1^2 = 2 * p * y1) (h4 : x2^2 = 2 * p * y2) 
  (h5 : y2 = 4 * y1) :
  x1^2 = 4 * y1 :=
sorry

-- Problem 2: Finding the range of b
theorem find_range_of_b (k : ℝ) (h : k > 0 ∨ k < -4) : 
  ∃ b : ℝ, b = 2 * (k + 1)^2 ∧ b > 2 :=
sorry

end find_parabola_eq_find_range_of_b_l741_741267


namespace find_m_l741_741658

theorem find_m (m : ℝ) (α : ℝ) (h1 : ∃ (P : ℝ × ℝ), P = (-8 * m, -3))
  (h2 : cos α = -4 / 5) : m = 1 / 2 :=
sorry

end find_m_l741_741658


namespace negation_of_proposition_l741_741802

variables (x : ℝ)

def proposition (x : ℝ) : Prop := x > 0 → (x ≠ 2 → (x^3 / (x - 2) > 0))

theorem negation_of_proposition : ∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x ≤ 2 :=
by
  sorry

end negation_of_proposition_l741_741802


namespace km_to_miles_l741_741520

theorem km_to_miles (h : 8 * 5 ≈ 8 * k) : 1.2 * 5 / 8 ≈ 0.75 :=
by
  sorry

end km_to_miles_l741_741520


namespace sum_of_coefficients_l741_741556

noncomputable def poly := 5 * (2 * (x : ℚ) ^ 8 - 3 * x ^ 5 + 9) + 6 * (x ^ 6 + 4 * x ^ 3 - 6)

theorem sum_of_coefficients : (poly.eval 1) = 34 :=
by
  sorry

end sum_of_coefficients_l741_741556


namespace field_dimension_solution_l741_741521

theorem field_dimension_solution (m : ℤ) (H1 : (3 * m + 11) * m = 100) : m = 5 :=
sorry

end field_dimension_solution_l741_741521


namespace intersection_point_l741_741582

theorem intersection_point (x y : ℝ) 
  (h1 : 8 * x - 3 * y = 24) 
  (h2 : 5 * x + 2 * y = 17) : 
  x = 99 / 31 ∧ y = 16 / 31 :=
begin
  sorry
end

end intersection_point_l741_741582


namespace correct_statement_C_l741_741089

theorem correct_statement_C
  (a : ℚ) : a < 0 → |a| = -a := 
by
  sorry

end correct_statement_C_l741_741089


namespace cauchy_schwarz_inequality_l741_741638

theorem cauchy_schwarz_inequality (x y : ℕ → ℝ) (n : ℕ) 
  (hypos : ∀ i, i < n → y i > 0) :
  (∑ i in Finset.range n, y i) * (∑ i in Finset.range n, (x i)^2 / (y i)) ≥ (∑ i in Finset.range n, x i)^2 :=
by
  sorry

end cauchy_schwarz_inequality_l741_741638


namespace determine_top_5_median_required_l741_741719

theorem determine_top_5_median_required (scores : Fin 9 → ℝ) (unique_scores : ∀ (i j : Fin 9), i ≠ j → scores i ≠ scores j) :
  ∃ median,
  (∀ (student_score : ℝ), 
    (student_score > median ↔ ∃ (idx_top : Fin 5), student_score = scores ⟨idx_top.1, sorry⟩)) :=
sorry

end determine_top_5_median_required_l741_741719


namespace find_fraction_l741_741697

theorem find_fraction (F : ℝ) (N : ℝ) (X : ℝ)
  (h1 : 0.85 * F = 36)
  (h2 : N = 70.58823529411765)
  (h3 : F = 42.35294117647059) :
  X * N = 42.35294117647059 → X = 0.6 :=
by
  sorry

end find_fraction_l741_741697


namespace map_area_ratio_l741_741857

theorem map_area_ratio (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  ¬ ((l * w) / ((500 * l) * (500 * w)) = 1 / 500) :=
by
  -- The proof will involve calculations showing the true ratio is 1/250000
  sorry

end map_area_ratio_l741_741857


namespace tangent_line_at_point_no_such_a_exists_l741_741667

noncomputable def f (x : ℝ) : ℝ := x - (1 / x) - Real.log x

theorem tangent_line_at_point :
  let a := 3
  let x1 := 1
  let y1 := f 1
  in  y1 = 0 ∧ (∃ m b, m = (f'(x1)) ∧ b = y1 - m * x1 ∧ x + y - 1 = 0 ) :=
begin
  let a := 3,
  have hx1 := (1 : ℝ),
  have hy1 := f hx1,
  have hy1_eq : f 1 = 0,
  simp [f], sorry
end

theorem no_such_a_exists :
  ∀ (a k : ℝ) (b c : ℝ), 
  let f' (x : ℝ) : ℝ := 1 + (1 / x^2) - (1 / x)
  (f' b = 0) ∧ (f' c = 0) ∧ (k = (f b - f c) / (b - c))
  → ¬ (k + a = 2) :=
begin
  intros a k b c cond,
  let f' (x : ℝ) := 1 + (1 / x^2) - (1 / x),
  sorry
end

end tangent_line_at_point_no_such_a_exists_l741_741667


namespace square_minimum_rotation_l741_741532

theorem square_minimum_rotation (deg : ℝ) : 
  (∃ (k : ℤ) (deg = 90 * k)) → 
  deg = 90 :=
by
  sorry

end square_minimum_rotation_l741_741532


namespace tan_product_ge_n_exp_n_succ_l741_741384

theorem tan_product_ge_n_exp_n_succ
  (n : ℕ)
  (a : Fin (n + 1) → ℝ)
  (h₀ : ∀ i, 0 < a i ∧ a i < π / 2)
  (h₁ : ∑ i, Real.tan (a i - π / 4) ≥ n - 1) :
  (∏ i, Real.tan (a i)) ≥ n^(n + 1) := sorry

end tan_product_ge_n_exp_n_succ_l741_741384


namespace solve_for_x_l741_741920

theorem solve_for_x (x : ℝ) (h : 2 * (1/x + 3/x / 6/x) - 1/x = 1.5) : x = 2 := 
by 
  sorry

end solve_for_x_l741_741920


namespace blue_stamp_price_l741_741786

theorem blue_stamp_price :
  ∀ (red_stamps blue_stamps yellow_stamps : ℕ) (red_price blue_price yellow_price total_earnings : ℝ),
    red_stamps = 20 →
    blue_stamps = 80 →
    yellow_stamps = 7 →
    red_price = 1.1 →
    yellow_price = 2 →
    total_earnings = 100 →
    (red_stamps * red_price + yellow_stamps * yellow_price + blue_stamps * blue_price = total_earnings) →
    blue_price = 0.80 :=
by
  intros red_stamps blue_stamps yellow_stamps red_price blue_price yellow_price total_earnings
  intros h_red_stamps h_blue_stamps h_yellow_stamps h_red_price h_yellow_price h_total_earnings
  intros h_earning_eq
  sorry

end blue_stamp_price_l741_741786


namespace max_area_triangle_ABC_l741_741389

noncomputable def area_triangle_ABC (p : ℝ) (q : ℝ) :=
  1 / 2 * abs ((1 * 4) + (5 * q) + (p * 0) - (0 * 5) - (4 * p) - (q * 1))

theorem max_area_triangle_ABC :
  (∀ p q, q = -p^2 + 8 * p - 12 ∧ 1 ≤ p ∧ p ≤ 5 → area_triangle_ABC p q ≤ 0.5) ∧
  ∃ p q, q = -p^2 + 8 * p - 12 ∧ 1 ≤ p ∧ p ≤ 5 ∧ area_triangle_ABC p q = 0.5 :=
begin
  -- Proof will go here
  sorry
end

end max_area_triangle_ABC_l741_741389


namespace point_not_in_region_l741_741927

theorem point_not_in_region (A B C D : ℝ × ℝ) :
  (A = (0, 0) ∧ 3 * A.1 + 2 * A.2 < 6) ∧
  (B = (1, 1) ∧ 3 * B.1 + 2 * B.2 < 6) ∧
  (C = (0, 2) ∧ 3 * C.1 + 2 * C.2 < 6) ∧
  (D = (2, 0) ∧ ¬ ( 3 * D.1 + 2 * D.2 < 6 )) :=
by {
  sorry
}

end point_not_in_region_l741_741927


namespace compute_100a_plus_b_l741_741334

-- Definitions of the given conditions

variables {A B C I D X : Point}
variables {a b : ℕ}

-- Conditions:
def cyclic_quadrilateral : Prop := 
  ∃ X, ∠XAB = ∠XAC ∧ Incenter I A B C ∧ Projection I D BC ∧
  AI = 25 ∧ ID = 7 ∧ BC = 14

-- Proof that 100a + b = 17524
theorem compute_100a_plus_b (a b : ℕ) (h1 : AI = 25) (h2 : ID = 7) (h3 : BC = 14) (X_def : cyclic_quadrilateral) (rel_prime : nat.coprime a b) :
  XI = a / b → 100 * a + b = 17524 :=
sorry

end compute_100a_plus_b_l741_741334


namespace number_of_terms_in_geometric_sequence_is_8_l741_741624

/-
Given a geometric sequence with the first term being 1 and an even number of terms,
if the sum of the odd terms is 85 and the sum of the even terms is 170, then the number of terms in the sequence is 8.
-/

theorem number_of_terms_in_geometric_sequence_is_8
  (a₁ : ℕ)
  (even_n : ℕ)
  (sum_of_odd_terms : ℕ)
  (sum_of_even_terms : ℕ) :
  a₁ = 1 →
  ∃ n, even_n = 2 * n →
  sum_of_odd_terms = 85 →
  sum_of_even_terms = 170 →
  even_n = 8 :=
by
  intros h₁ ⟨n, hn⟩ h₂ h₃
  sorry

end number_of_terms_in_geometric_sequence_is_8_l741_741624


namespace orthocenter_of_triangle_ABC_l741_741333

noncomputable def point : Type := (ℝ × ℝ × ℝ)

def A : point := (2, 3, 4)
def B : point := (6, 4, 2)
def C : point := (4, 5, 6)
def orthocenter : point := (5 / 2, 3, 7 / 2)

theorem orthocenter_of_triangle_ABC (H : point) : 
  H = orthocenter ↔ (H = (5 / 2, 3, 7 / 2)) :=
begin
  sorry -- Proof
end

end orthocenter_of_triangle_ABC_l741_741333


namespace catfish_weight_l741_741569

-- Define the given conditions
def num_trout : ℕ := 4
def num_catfish : ℕ := 3
def num_blues : ℕ := 5
def weight_trout : ℝ := 2.0
def weight_blues : ℝ := 2.5
def total_weight : ℝ := 25.0

-- Define the problem statement
theorem catfish_weight :
  (num_trout * weight_trout + num_blues * weight_blues + num_catfish * x = total_weight) → 
  (x = total_weight - (num_trout * weight_trout + num_blues * weight_blues)) / num_catfish :=
begin
  sorry
end

end catfish_weight_l741_741569


namespace theo_drinks_8_cups_per_day_l741_741455

/--
Theo, Mason, and Roxy are siblings. 
Mason drinks 7 cups of water every day.
Roxy drinks 9 cups of water every day. 
In one week, the siblings drink 168 cups of water together. 

Prove that Theo drinks 8 cups of water every day.
-/
theorem theo_drinks_8_cups_per_day (T : ℕ) :
  (∀ (d m r : ℕ), 
    (m = 7 ∧ r = 9 ∧ d + m + r = 168) → 
    (T * 7 = d) → T = 8) :=
by
  intros d m r cond1 cond2
  have h1 : d + 49 + 63 = 168 := by sorry
  have h2 : T * 7 = d := cond2
  have goal : T = 8 := by sorry
  exact goal

end theo_drinks_8_cups_per_day_l741_741455


namespace number_of_unripe_integers_l741_741976

/-- A digit that is not prime -/
def non_prime_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 9

/-- A two-digit positive integer is \textit{primeable} if one of its digits can be deleted to produce a prime number -/
def is_primeable (n : ℕ) : Prop :=
  n / 10 ∈ {1, 3, 7, 9} ∧ Prime (n % 10) ∨ Prime (n / 10) ∧ n % 10 ∈ {1, 3, 7, 9}

/-- A two-digit positive integer that is prime, yet not primeable, is \textit{unripe} -/
def is_unripe (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Prime n ∧ ¬is_primeable n

theorem number_of_unripe_integers : ∃ count : ℕ, count = 10 ∧ ∀ n : ℕ, is_unripe n → n ∈ {11, 31, 41, 61, 71, 19, 29, 59, 79, 89} :=
by
  sorry

end number_of_unripe_integers_l741_741976


namespace smallest_number_with_55_divisors_l741_741603

theorem smallest_number_with_55_divisors :
  ∃ n : ℕ, (n = 3^4 * 2^{10}) ∧ (nat.count_divisors n = 55) :=
by
  have n : ℕ := 3^4 * 2^{10}
  exact ⟨n, ⟨rfl, nat.count_divisors_eq_count_divisors 3 4 2 10⟩⟩
  sorry

end smallest_number_with_55_divisors_l741_741603


namespace slices_left_for_lunch_tomorrow_l741_741202

def pizza_slices : ℕ := 12
def lunch_slices : ℕ := pizza_slices / 2
def remaining_after_lunch : ℕ := pizza_slices - lunch_slices
def dinner_slices : ℕ := remaining_after_lunch * 1/3
def slices_left : ℕ := remaining_after_lunch - dinner_slices

theorem slices_left_for_lunch_tomorrow : slices_left = 4 :=
by
  sorry

end slices_left_for_lunch_tomorrow_l741_741202


namespace additional_life_vests_needed_l741_741541

def num_students : ℕ := 40
def num_instructors : ℕ := 10
def life_vests_on_hand : ℕ := 20
def percent_students_with_vests : ℕ := 20

def total_people : ℕ := num_students + num_instructors
def students_with_vests : ℕ := (percent_students_with_vests * num_students) / 100
def total_vests_available : ℕ := life_vests_on_hand + students_with_vests

theorem additional_life_vests_needed : 
  total_people - total_vests_available = 22 :=
by 
  sorry

end additional_life_vests_needed_l741_741541


namespace q_invested_time_l741_741498

/--
The ratio of investments of two partners P and Q is 7:5 and the ratio of their profits is 7:9. 
If P invested the money for 5 months, prove that Q invested the money for 9 months.
-/
theorem q_invested_time
  (x : ℝ)
  (investment_ratio : (7:ℝ * x) / (5 * x) = 7 / 5)
  (profit_ratio : (7:ℝ) / 9 = 7 / 9)
  (p_investment_time : 5:ℝ)
  (investment_time_proportion : (7 * 5) / (5 * t) = 7 / 9) :
  t = 9 := 
by 
  sorry

end q_invested_time_l741_741498


namespace integral_f_l741_741240

def f (x : ℝ) : ℝ := 2 - |x|

theorem integral_f : ∫ x in -1..2, f x = 3.5 :=
by
  sorry

end integral_f_l741_741240


namespace distance_between_points_l741_741736

theorem distance_between_points :
  let A : ℝ × ℝ × ℝ := (1, -2, 3)
  let B : ℝ × ℝ × ℝ := (1, 2, 3)
  let C : ℝ × ℝ × ℝ := (1, 2, -3)
  dist B C = 6 :=
by
  sorry

end distance_between_points_l741_741736


namespace chapters_page_difference_l741_741124

def chapter1_pages : ℕ := 37
def chapter2_pages : ℕ := 80

theorem chapters_page_difference : chapter2_pages - chapter1_pages = 43 := by
  -- Proof goes here
  sorry

end chapters_page_difference_l741_741124


namespace fewerSevensCanProduce100_l741_741917

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l741_741917


namespace factorize_expression_l741_741573

theorem factorize_expression (x : ℝ) : 
  x^4 + 324 = (x^2 - 18 * x + 162) * (x^2 + 18 * x + 162) := 
sorry

end factorize_expression_l741_741573


namespace evaluate_expression_l741_741086

-- Conditions as definitions
def val_1296 : ℕ := 6 ^ 4
def val_4096 : ℕ := 4096
def log_base_6 (x : ℕ) : ℝ := Real.log x / Real.log 6
def exp_log_id (a x : ℕ) : ℝ := Real.exp (log_base_6 x * Real.log a)

-- Main theorem
theorem evaluate_expression : ((val_1296 ^ log_base_6 val_4096) ^ (1 / 4 : ℝ)) = val_4096 := by
  sorry

end evaluate_expression_l741_741086


namespace mayuki_speed_l741_741787

theorem mayuki_speed {r : ℝ} (w : ℝ) (t : ℝ) (s : ℝ) (h₁ : w = 4) (h₂ : t = 24) 
  (h₃ : s = 8 * π) : (s / t) = π / 3 :=
by
  rw [h₂, h₃]
  simp
  norm_num
  rw [two_mul, mul_div_cancel_left]
  norm_num
  exact pi_ne_zero

end mayuki_speed_l741_741787


namespace negation_of_proposition_l741_741672

variable (x : ℝ)
variable (p : Prop)

def proposition : Prop := ∀ x > 0, (x + 1) * Real.exp x > 1

theorem negation_of_proposition : ¬ proposition ↔ ∃ x > 0, (x + 1) * Real.exp x ≤ 1 :=
by
  sorry

end negation_of_proposition_l741_741672


namespace v_closed_under_multiplication_l741_741778

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def is_closed_under_mul (s : set ℕ) : Prop :=
  ∀ a b ∈ s, a * b ∈ s

theorem v_closed_under_multiplication :
  is_closed_under_mul {n | is_cube n} :=
by
  sorry

end v_closed_under_multiplication_l741_741778


namespace fewerSevensCanProduce100_l741_741914

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l741_741914


namespace find_c_l741_741625

structure Point where
  x : ℝ
  y : ℝ

def direction_vector (p1 p2 : Point) : Point :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

theorem find_c (c : ℝ) (p1 p2 : Point) (h1 : p1 = ⟨-3, 1⟩) (h2 : p2 = ⟨0, 4⟩) (hvec : direction_vector p1 p2 = ⟨3, c⟩) :
  c = 3 :=
by
  have h1_vec := direction_vector ⟨-3, 1⟩ ⟨0, 4⟩
  rw [direction_vector] at h1_vec
  cases h1_vec
  have h3 : 3 = 3 := rfl
  have h4 : 3 = c := (by congr; assumption)
  rw [h4]; exact h3
  sorry

end find_c_l741_741625


namespace prop1_prop2_prop4_true_propositions_proved_l741_741277

open Real 
open Set

-- Proposition 1
theorem prop1 (h : ∃ x_0 ∈ Ioo 0 2, 3 ^ x_0 ≤ x_0 ^ 3) : ¬ (∀ x ∈ Ioo 0 2, 3 ^ x > x ^ 3) :=
  sorry

-- Proposition 2
def f1 (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem prop2 : ∀ x : ℝ, f1 (-x) = -(f1 x) :=
  sorry

-- Proposition 4
theorem prop4 {A B : ℝ} (h : A > B) : sin A > sin B :=
  sorry

-- Wrapper theorem for combining the true propositions
theorem true_propositions_proved :
  (∃ x_0 ∈ Ioo 0 2, 3 ^ x_0 ≤ x_0 ^ 3 → ¬ ∀ x ∈ Ioo 0 2, 3 ^ x > x ^ 3) ∧
  (∀ x : ℝ, f1 (-x) = - (f1 x)) ∧
  ∀ A B : ℝ, (A > B → sin A > sin B) :=
by
  exact ⟨prop1, prop2, prop4⟩

end prop1_prop2_prop4_true_propositions_proved_l741_741277


namespace min_value_sqrt_expr_l741_741758

theorem min_value_sqrt_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  \(\frac{\sqrt{(x^2 + y^2)(4x^2 + y^2)}}{xy}\) ≥ 3 ∧ 
    (\(\frac{\sqrt{(x^2 + y^2)(4x^2 + y^2)}}{xy}\) = 3 ↔ y = x \(\sqrt{2}\)) :=
sorry

end min_value_sqrt_expr_l741_741758


namespace sum_of_digits_of_sevens_and_threes_l741_741197

def string_of_n_sevens (n : ℕ) : ℕ :=
  nat.repeat 7 n

def string_of_n_threes (n : ℕ) : ℕ :=
  nat.repeat 3 n

def sum_of_digits (n : ℕ) : ℕ :=
  nat.digits 10 n |> list.sum

theorem sum_of_digits_of_sevens_and_threes : sum_of_digits (string_of_n_sevens 77 * string_of_n_threes 77) = 231 :=
  sorry

end sum_of_digits_of_sevens_and_threes_l741_741197


namespace lucas_change_l741_741782

def initialAmount : ℕ := 20
def costPerAvocado : ℕ := 2
def numberOfAvocados : ℕ := 3

def totalCost : ℕ := numberOfAvocados * costPerAvocado
def change : ℕ := initialAmount - totalCost

theorem lucas_change : change = 14 := by
  sorry

end lucas_change_l741_741782


namespace evaluate_expression_l741_741213

theorem evaluate_expression : 6 - 8 * (5 - 2^3) / 2 = 18 := by
  sorry

end evaluate_expression_l741_741213


namespace heidi_uniform_number_is_19_l741_741614

/-- Four tennis players have two-digit primes as uniform numbers. --/
noncomputable def tennis_players (e f g h : ℕ) : Prop :=
  (e.prime ∧ e ≥ 10 ∧ e ≤ 99) ∧
  (f.prime ∧ f ≥ 10 ∧ f ≤ 99) ∧
  (g.prime ∧ g ≥ 10 ∧ g ≤ 99) ∧
  (h.prime ∧ h ≥ 10 ∧ h ≤ 99) ∧ 
  (f + g + h = 29) ∧
  (e + f = 30) ∧
  (g + f = 28) ∧
  (h + g = 30)

/-- Prove that Heidi's uniform number is 19 given the above conditions. --/
theorem heidi_uniform_number_is_19 : ∃ e f g h : ℕ, tennis_players e f g h ∧ h = 19 :=
by 
  sorry

end heidi_uniform_number_is_19_l741_741614


namespace problem1_problem2_l741_741183

theorem problem1 :
  (1 : ℝ) * (0.001 : ℝ) ^ (-1 / 3) + (27 : ℝ) ^ (2 / 3) + (1 / 4 : ℝ) ^ (-1 / 2) - (1 / 9 : ℝ) ^ (-3 / 2) = -6 :=
by
  sorry

theorem problem2 :
  (1 / 2 : ℝ) * real.log10 25 + real.log10 2 - real.log10 (sqrt 0.1) - (real.logb 2 9) * (real.logb 3 2) = 1 / 2 :=
by
  sorry

end problem1_problem2_l741_741183


namespace binders_per_student_l741_741739

theorem binders_per_student
  (num_students : ℕ)
  (pens_per_student notebooks_per_student binders_per_student highlighters_per_student : ℕ)
  (pen_cost notebook_cost binder_cost highlighter_cost teacher_discount amount_spent : ℝ)
  (h1 : num_students = 30)
  (h2 : pens_per_student = 5)
  (h3 : notebooks_per_student = 3)
  (h4 : highlighters_per_student = 2)
  (h5 : pen_cost = 0.5)
  (h6 : notebook_cost = 1.25)
  (h7 : binder_cost = 4.25)
  (h8 : highlighter_cost = 0.75)
  (h9 : teacher_discount = 100)
  (h10 : amount_spent = 260) :
  binders_per_student = 1 :=
begin
  sorry
end

end binders_per_student_l741_741739


namespace time_away_proof_l741_741961

-- Define necessary constants and functions
def hour_hand_angle (n : ℝ) : ℝ := 180 + n / 2
def minute_hand_angle (n : ℝ) : ℝ := 6 * n
def angle_between_hands (n : ℝ) : ℝ := abs (hour_hand_angle n - minute_hand_angle n)
def time_away (n : ℝ) : ℝ := n
def n₁ : ℝ := 120 / 11
def n₂ : ℝ := 600 / 11

theorem time_away_proof : time_away (n₂ - n₁) = 43.636 :=
  by
  let n_total := n₂ - n₁
  have : n_total = 480 / 11,
    calc n_total = n₂ - n₁ : by sorry
              ... = 480 / 11 : by sorry
  show n_total = 43.636,
    calc n_total = 480 / 11 : by sorry
              ... = 43.636 : by norm_num

end time_away_proof_l741_741961


namespace correct_statement_l741_741488

def is_accurate_to (value : ℝ) (place : ℝ) : Prop :=
  ∃ k : ℤ, value = k * place

def statement_A : Prop := is_accurate_to 51000 0.1
def statement_B : Prop := is_accurate_to 0.02 1
def statement_C : Prop := (2.8 = 2.80)
def statement_D : Prop := is_accurate_to (2.3 * 10^4) 1000

theorem correct_statement : statement_D :=
by
  sorry

end correct_statement_l741_741488


namespace simplify_expression_l741_741997

theorem simplify_expression (x : ℝ) : 2 * x * (x - 4) - (2 * x - 3) * (x + 2) = -9 * x + 6 :=
by
  sorry

end simplify_expression_l741_741997


namespace field_area_l741_741098

theorem field_area (x y : ℕ) (h1 : x + y = 700) (h2 : y - x = (1/5) * ((x + y) / 2)) : x = 315 :=
  sorry

end field_area_l741_741098


namespace intersection_M_N_eq_l741_741290

def M (x : ℝ) : Set ℝ := {y | ∃ x, y = 2^x ∧ x > 0}
def N (x : ℝ) : Set ℝ := {y | ∃ x, y = log 10 x}

theorem intersection_M_N_eq (x : ℝ) : M x ∩ N x = { y | 1 < y ∧ y < ∞ } :=
by sorry

end intersection_M_N_eq_l741_741290


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741894

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741894


namespace segment_sum_condition_l741_741854

def sum_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem segment_sum_condition :
  let vertices := [1, 2, 3, 4, 5, 6, 10, 7, 13, 14, 12, 11, 9, 8],
      segments := [[vertices[0], vertices[1], vertices[5], vertices[7]],  -- Segment 1
                   [vertices[1], vertices[2], vertices[6], vertices[8]],  -- Segment 2
                   [vertices[2], vertices[3], vertices[12], vertices[13]], -- Segment 3
                   [vertices[3], vertices[4], vertices[11], vertices[10]], -- Segment 4
                   [vertices[4], vertices[5], vertices[9], vertices[10]],  -- Segment 5
                   [vertices[6], vertices[7], vertices[11], vertices[9]],  -- Segment 6
                   [vertices[8], vertices[12], vertices[13], vertices[10]]] -- Segment 7
  in ∀ s ∈ segments, (s.sum = 30) :=
by
  sorry

end segment_sum_condition_l741_741854


namespace tan_alpha_beta_l741_741237

theorem tan_alpha_beta (α β : ℝ) 
    (h1 : 3 * Real.tan(α / 2) + Real.tan(α / 2)^2 = 1)
    (h2 : Real.sin β = 3 * Real.sin(2 * α + β)) :
    Real.tan (α + β) = -4 / 3 :=
by
  sorry

end tan_alpha_beta_l741_741237


namespace solve_problem_l741_741654

variable (a b c x : ℝ)

-- Conditions
def condition1 : Prop := ∀ x, ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2

-- Statements to prove
def statementA : Prop := a < 0
def statementB : Prop := a + b + c > 0
def statementD : Prop := ∀ x, (cx^2 + bx + a < 0 ↔ (-1/3 < x ∧ x < 1/2))

theorem solve_problem (h1 : condition1)
  (h2 : statementA)
  (h3 : statementB)
  (h4 : statementD) : a < 0 ∧ a + b + c > 0 ∧ ∀ x, (cx^2 + bx + a < 0 ↔ (-1/3 < x ∧ x < 1/2)) :=
by
  sorry

end solve_problem_l741_741654


namespace sin_inequality_l741_741807

theorem sin_inequality (α : ℝ) (h1 : 0 < α) (h2 : α < π / 6) :
  2 * sin α < sin (3 * α) ∧ sin (3 * α) < 3 * sin α :=
by
  sorry

end sin_inequality_l741_741807


namespace length_calculation_l741_741443

noncomputable def length_of_rectangular_floor (breadth : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) : ℝ :=
  let area := total_cost / rate_per_sq_meter
  let B_sq := area / 3
  let B := real.sqrt B_sq
  let L := 3 * B
  L

theorem length_calculation 
  (breadth : ℝ)
  (total_cost : ℝ := 300)
  (rate_per_sq_meter : ℝ := 5)
  (h1 : total_cost = 300)
  (h2 : rate_per_sq_meter = 5) :
  length_of_rectangular_floor breadth total_cost rate_per_sq_meter ≈ 13.416 :=
by
  sorry

end length_calculation_l741_741443


namespace largest_constant_C_valid_l741_741390

noncomputable def largest_constant_C (n : ℕ) :=
  2 * n / (n - 1)

theorem largest_constant_C_valid {n} (h : 1 < n) {a : Fin n → ℝ}
  (h_nonneg : ∀ i, 0 ≤ a i) :
  (∑ i, a i)^2 ≥ largest_constant_C n * ∑ i j, if i < j then a i * a j else 0 :=
by 
  sorry

end largest_constant_C_valid_l741_741390


namespace find_t_when_perpendicular_l741_741680

variable {t : ℝ}

def vector_m (t : ℝ) : ℝ × ℝ := (t + 1, 1)
def vector_n (t : ℝ) : ℝ × ℝ := (t + 2, 2)
def add_vectors (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def sub_vectors (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

theorem find_t_when_perpendicular : 
  (dot_product (add_vectors (vector_m t) (vector_n t)) (sub_vectors (vector_m t) (vector_n t)) = 0) ↔ t = -3 := by
  sorry

end find_t_when_perpendicular_l741_741680


namespace minimum_throws_for_repeated_sum_l741_741039

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741039


namespace fg_difference_l741_741387

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 4 * x - 1

theorem fg_difference : f (g 3) - g (f 3) = -16 := by
  sorry

end fg_difference_l741_741387


namespace sum_of_m_with_distinct_integer_solutions_eq_0_l741_741081

theorem sum_of_m_with_distinct_integer_solutions_eq_0 :
  (∑ m in {m | ∃ r s : ℤ, r ≠ s ∧ (3 * r^2 - m * r + 12 = 0) ∧ (3 * s^2 - m * s + 12 = 0)}, m) = 0 :=
sorry

end sum_of_m_with_distinct_integer_solutions_eq_0_l741_741081


namespace calculate_speed_l741_741151

variable (time : ℝ) (distance : ℝ)

theorem calculate_speed (h_time : time = 5) (h_distance : distance = 500) : 
  distance / time = 100 := 
by 
  sorry

end calculate_speed_l741_741151


namespace range_of_m_l741_741646

theorem range_of_m (f : ℝ → ℝ) {m : ℝ} (h_dec : ∀ x y, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ x < y → f x ≥ f y)
  (h_ineq : f (m - 1) > f (2 * m - 1)) : 0 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l741_741646


namespace locus_of_vertex_A_90_degrees_l741_741247

-- Define the points and line segment
variables (A B C P : Point) (BC : LineSegment B C)

-- Conditions: A is a point in space and BC is a line segment
axiom point_A : Point
axiom lineSeg_BC : LineSegment B C

-- Defining conditions for the angle and the locus
def right_angle_condition (P : Point) : Prop :=
  ∃ (M : Point), (M ∈ BC) ∧ (dist A P = dist A M) ∧ (angle A P M = π / 2) 

-- Proven statement: The locus of point P is the set of points that lie in the regions defined by the two spheres centered at A with radii dist A B and dist A C, excluding their overlap.
theorem locus_of_vertex_A_90_degrees : 
  ∀ P : Point, (right_angle_condition P) ↔ 
  ((dist A P = dist A B) ∨ (dist A P = dist A C)) ∧ ¬((dist A P = dist A B) ∧ (dist A P = dist A C)) := 
sorry

end locus_of_vertex_A_90_degrees_l741_741247


namespace sequence_mono_increasing_range_of_b_l741_741774

theorem sequence_mono_increasing_range_of_b (b : ℝ) 
  (a_n : ℕ → ℝ) (h : ∀ n : ℕ, a_n n = n^2 + b * n) :
  (∀ n : ℕ, n > 0 → a_n (n + 1) - a_n n > 0) ↔ b > -3 := by
  unfold a_n
  sorry

end sequence_mono_increasing_range_of_b_l741_741774


namespace integer_roots_of_polynomial_l741_741218

def polynomial : Polynomial ℤ := Polynomial.C 24 + Polynomial.C (-11) * Polynomial.X + Polynomial.C (-4) * Polynomial.X^2 + Polynomial.X^3

theorem integer_roots_of_polynomial :
  Set { x : ℤ | polynomial.eval x polynomial = 0 } = Set ({-4, 3, 8} : Set ℤ) :=
sorry

end integer_roots_of_polynomial_l741_741218


namespace area_of_large_square_l741_741550

-- Define the conditions for the exposed areas of the squares
variables (A B C : ℝ) -- Exposed areas of yellow, red, and blue sheets

-- Define the side length of each square
variable s : ℝ

-- Hypotheses based on the conditions
axiom h1 : B = 19
axiom h2 : C = 11
axiom h3 : A = 25

-- The goal is to show that, the area of the larger square box is 64
theorem area_of_large_square (A B C s : ℝ) 
  (h_blue : A = 25) (h_red : B = 19) (h_yellow : C = 11) 
  (h_s : s = 5) : 
  (s + (C / s))^2 = 64 := 
by
  sorry

end area_of_large_square_l741_741550


namespace minimum_throws_for_repeated_sum_l741_741036

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741036


namespace correct_equation_l741_741706

-- Given definition
def z : ℂ := 1 + complex.i

-- The proof problem
theorem correct_equation : z^2 - 2*z + 2 = 0 := by
  sorry

end correct_equation_l741_741706


namespace strawberries_count_l741_741463

theorem strawberries_count (total_fruits : ℕ) (kiwi_fraction : ℚ) (remaining_fraction : ℚ) 
  (H1 : total_fruits = 78) (H2 : kiwi_fraction = 1 / 3) (H3 : remaining_fraction = 2 / 3) : 
  let kiwi_count := kiwi_fraction * total_fruits
      strawberry_count := remaining_fraction * total_fruits in
  strawberry_count = 52 := 
by
  sorry

end strawberries_count_l741_741463


namespace smallest_abcd_value_l741_741485

theorem smallest_abcd_value (A B C D : ℕ) (h1 : A ≠ B) (h2 : 1 ≤ A) (h3 : A ≤ 9) (h4 : 0 ≤ B) 
                            (h5 : B ≤ 9) (h6 : 1 ≤ C) (h7 : C ≤ 9) (h8 : 1 ≤ D) (h9 : D ≤ 9)
                            (h10 : 10 * A * A + A * B = 1000 * A + 100 * B + 10 * C + D)
                            (h11 : A ≠ C) (h12 : A ≠ D) (h13 : B ≠ C) (h14 : B ≠ D) (h15 : C ≠ D) :
  1000 * A + 100 * B + 10 * C + D = 2046 :=
sorry

end smallest_abcd_value_l741_741485


namespace vasya_example_fewer_sevens_l741_741879

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l741_741879


namespace union_sets_l741_741259

def A : Set ℝ := {x | 2 ^ x - 5 ^ x < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (x ^ 2 - x - 2)}

theorem union_sets :
  A ∪ B = {x | x < -1} ∪ {x | x > 0} :=
by
  sorry

end union_sets_l741_741259


namespace number_of_integer_values_of_x_l741_741848

theorem number_of_integer_values_of_x (x : ℕ) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) :
  ∃ n : ℕ, n = 29 ∧ ∀ y : ℕ, (26 ≤ y ∧ y ≤ 54) ↔ true :=
by
  sorry

end number_of_integer_values_of_x_l741_741848


namespace chord_length_l741_741447

theorem chord_length (a b : ℝ) (M : ℝ) (h : M * M = a * b) : ∃ AB : ℝ, AB = 2 * Real.sqrt (a * b) :=
by
  sorry

end chord_length_l741_741447


namespace walnut_trees_count_l741_741459

theorem walnut_trees_count (current_trees new_trees : ℕ)
(assume_init : current_trees = 4)
(assume_new : new_trees = 6) :
current_trees + new_trees = 10 :=
by 
  rw [assume_init, assume_new]
  exact rfl

end walnut_trees_count_l741_741459


namespace sum_of_m_with_distinct_integer_roots_l741_741083

theorem sum_of_m_with_distinct_integer_roots :
  (∑ m in {m | ∃ (r s : ℤ), r ≠ s ∧ (3 * r * r - m * r + 12 = 0) ∧ (3 * s * s - m * s + 12 = 0)}, m) = 0 :=
sorry

end sum_of_m_with_distinct_integer_roots_l741_741083


namespace range_of_a_l741_741317

theorem range_of_a
  (h : ∀ x : ℝ, |x - 1| + |x - 2| > Real.log (a ^ 2) / Real.log 4) :
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2 :=
sorry

end range_of_a_l741_741317


namespace arithmetic_seq_ratio_l741_741168

noncomputable def S (n : ℕ) (a₁ d : ℝ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_seq_ratio (a₁ d : ℝ) (h : (S 6 a₁ d) / (S 3 a₁ d) = 4) :
  (S 9 a₁ d) / (S 6 a₁ d) = 9 / 4 :=
by
  -- definitions for S_6 and S_3
  have S₆ := S 6 a₁ d
  have S₃ := S 3 a₁ d
  -- condition 
  have h₁ : S₆ / S₃ = 4 := h
  -- hypothesis about d
  have d_eq : d = 2 * a₁, sorry
  -- substitution in the target 
  have S₉ := S 9 a₁ d
  -- calculation for S₉ / S₆
  sorry

end arithmetic_seq_ratio_l741_741168


namespace age_difference_is_36_l741_741868

open Nat

theorem age_difference_is_36 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
    (h_eq : (10 * a + b) + 8 = 3 * ((10 * b + a) + 8)) :
    (10 * a + b) - (10 * b + a) = 36 :=
by
  sorry

end age_difference_is_36_l741_741868


namespace minimum_rolls_for_duplicate_sum_l741_741056

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741056


namespace alice_needs_more_life_vests_l741_741544

-- Definitions based on the given conditions
def students : ℕ := 40
def instructors : ℕ := 10
def lifeVestsOnHand : ℕ := 20
def percentWithLifeVests : ℚ := 0.20

-- Statement of the problem
theorem alice_needs_more_life_vests :
  let totalPeople := students + instructors
  let lifeVestsBroughtByStudents := (percentWithLifeVests * students).toNat
  let totalLifeVestsAvailable := lifeVestsOnHand + lifeVestsBroughtByStudents
  totalPeople - totalLifeVestsAvailable = 22 :=
by
  sorry

end alice_needs_more_life_vests_l741_741544


namespace fencing_cost_l741_741846

theorem fencing_cost (w : ℝ) (h : ℝ) (p : ℝ) (cost_per_meter : ℝ) 
  (hw : h = w + 10) (perimeter : p = 220) (cost_rate : cost_per_meter = 6.5) : 
  ((p * cost_per_meter) = 1430) := by 
  sorry

end fencing_cost_l741_741846


namespace quadratic_real_roots_l741_741612

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) :=
by
  sorry

end quadratic_real_roots_l741_741612


namespace polyhedron_faces_l741_741570

theorem polyhedron_faces (V E : ℕ) (F T P : ℕ) (h1 : F = 40) (h2 : V - E + F = 2) (h3 : T + P = 40) 
  (h4 : E = (3 * T + 4 * P) / 2) (h5 : V = (160 - T) / 2 - 38) (h6 : P = 3) (h7 : T = 1) :
  100 * P + 10 * T + V = 351 :=
by
  sorry

end polyhedron_faces_l741_741570


namespace min_rolls_to_duplicate_sum_for_four_dice_l741_741004

theorem min_rolls_to_duplicate_sum_for_four_dice : 
    let min_sum := 4 * 1,
    let max_sum := 4 * 6,
    let possible_sums := max_sum - min_sum + 1 in
    possible_sums = 21 → 
    (possible_sums + 1 = 22) := 
by
  intros min_sum max_sum possible_sums h
  have h1 : min_sum = 4 := rfl
  have h2 : max_sum = 24 := rfl
  have h3 : possible_sums = 21 := h
  have h4 : possible_sums + 1 = 22 := calc
    possible_sums + 1 = 21 + 1 : by rw h
    ... = 22 : by rfl
  exact h4

end min_rolls_to_duplicate_sum_for_four_dice_l741_741004


namespace embankment_building_l741_741738

theorem embankment_building (days : ℕ) (workers_initial : ℕ) (workers_later : ℕ) (embankments : ℕ) :
  workers_initial = 75 → days = 4 → embankments = 2 →
  (∀ r : ℚ, embankments = workers_initial * r * days →
            embankments = workers_later * r * 5) :=
by
  intros h75 hd4 h2 r hr
  sorry

end embankment_building_l741_741738


namespace complex_div_conjugate_l741_741274

theorem complex_div_conjugate (z1 z2 : ℂ) (h1 : z1 = 3 - complex.i) (h2 : z2 = 1 + complex.i) : 
  (conj z1) / z2 = 2 - complex.i :=
by
  sorry

end complex_div_conjugate_l741_741274


namespace minimum_throws_for_repeated_sum_l741_741041

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741041


namespace minimum_throws_for_repeated_sum_l741_741045

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741045


namespace find_m_l741_741678

-- Define the vectors and angle condition
def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b : ℝ × ℝ := (Real.sqrt 3, 1)
def angle_between_a_b (θ : ℝ) : Prop := θ = 30 * Real.pi / 180

-- Main statement to prove
theorem find_m (m : ℝ) (h1 : angle_between_a_b (Real.acos ((vector_a m).fst * vector_b.fst + (vector_a m).snd * vector_b.snd / (Real.sqrt (vector_a m).fst ^ 2 + 9) * 2))) : m = Real.sqrt 3 :=
sorry

end find_m_l741_741678


namespace probability_of_specific_balls_l741_741959

-- Define total number of balls in the jar
def total_balls : ℕ := 5 + 7 + 2 + 3 + 4

-- Define combinations function
def comb (n k : ℕ) : ℕ := n.choose k

-- Define number of ways to choose 1 black, 1 green, and 1 red ball
def favorable_outcomes : ℕ := comb 5 1 * comb 2 1 * comb 4 1

-- Define total number of ways to choose any 3 balls
def total_outcomes : ℕ := comb total_balls 3

-- Define probability of picking one black, one green, and one red ball
def probability := (4 : ℚ) / 133

theorem probability_of_specific_balls :
  5 = 5 ∧ 7 = 7 ∧ 2 = 2 ∧ 3 = 3 ∧ 4 = 4 →
  (favorable_outcomes.to_rat / total_outcomes.to_rat) = probability :=
begin
  sorry
end

end probability_of_specific_balls_l741_741959


namespace five_digit_units_digit_probability_even_l741_741139

theorem five_digit_units_digit_probability_even : 
  (∃ n : ℕ, 10000 ≤ n ∧ n < 100000) →
  (probability (d : ℕ, 0 ≤ d ∧ d < 10 ∧ ∃ k, n = 10 * k + d ∧ d % 2 = 0) = 1 / 2) := by
  sorry

end five_digit_units_digit_probability_even_l741_741139


namespace james_marbles_left_l741_741358

def marbles_remain (total_marbles : ℕ) (bags : ℕ) (given_away : ℕ) : ℕ :=
  (total_marbles / bags) * (bags - given_away)

theorem james_marbles_left :
  marbles_remain 28 4 1 = 21 := 
by
  sorry

end james_marbles_left_l741_741358


namespace donna_pizza_slices_l741_741203

theorem donna_pizza_slices :
  ∀ (total_slices : ℕ) (half_eaten_for_lunch : ℕ) (one_third_eaten_for_dinner : ℕ),
  total_slices = 12 →
  half_eaten_for_lunch = total_slices / 2 →
  one_third_eaten_for_dinner = half_eaten_for_lunch / 3 →
  (half_eaten_for_lunch - one_third_eaten_for_dinner) = 4 :=
by
  intros total_slices half_eaten_for_lunch one_third_eaten_for_dinner
  intros h1 h2 h3
  sorry

end donna_pizza_slices_l741_741203


namespace probability_units_digit_even_l741_741143

theorem probability_units_digit_even : 
  let num_digits := 5
  let total_digits := 9 - 0 + 1
  let even_digits := 5
  0 < num_digits ∧ num_digits == 5 ∧ total_digits == 10 ∧ even_digits == 5 ↔ (sorry : (even_digits : ℝ) / total_digits == (1 / 2))

end probability_units_digit_even_l741_741143


namespace min_throws_to_ensure_same_sum_twice_l741_741070

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741070


namespace five_digit_units_digit_probability_even_l741_741138

theorem five_digit_units_digit_probability_even : 
  (∃ n : ℕ, 10000 ≤ n ∧ n < 100000) →
  (probability (d : ℕ, 0 ≤ d ∧ d < 10 ∧ ∃ k, n = 10 * k + d ∧ d % 2 = 0) = 1 / 2) := by
  sorry

end five_digit_units_digit_probability_even_l741_741138


namespace number_of_valid_integers_1994_digits_l741_741689

theorem number_of_valid_integers_1994_digits :
  ∃ n : ℕ, n = 8 * 3^996 ∧
  ∀ (x : ℕ), 
    (∀ i < 1994, (∀ d, x.digitChar i = d → d ∈ {1, 2, 3, 4, 5})) ∧
    (∀ i < 1993, abs ((x div (10^i)) % 10 - (x div (10^(i+1))) % 10) = 1) →
    count_valid_integers x = n :=
sorry

end number_of_valid_integers_1994_digits_l741_741689


namespace min_grades_required_l741_741296

variable {n : ℕ}
variable {s : ℕ}

theorem min_grades_required (hn : 0 < n) (hs : 4.5 < s / n ∧ s / n < 4.51) :
  n ≥ 51 :=
sorry

end min_grades_required_l741_741296


namespace intervals_of_monotonicity_extremum_values_on_interval_l741_741668

-- Given data and conditions
def f (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 2

-- Theorem statement to prove intervals of monotonicity
theorem intervals_of_monotonicity 
  (a b : ℝ)
  (extremum_cond : f(-1) = 7 ∧ 3*(-1)^2 + 2*a*(-1) + b = 0) :
  (∃ (I1 I2 I3 : set ℝ), 
    I1 = set.Iio (-1) ∧ I2 = set.Ioc (-1) 3 ∧ I3 = set.Ioi 3 ∧ 
    (∀ x ∈ I1, f'(x) > 0) ∧ 
    (∀ x ∈ I2, f'(x) < 0) ∧ 
    (∀ x ∈ I3, f'(x) > 0)) :=
sorry

-- Theorem to prove the extremum values on the interval [-2, 4]
theorem extremum_values_on_interval 
  (a b : ℝ)
  (extremum_cond : f(-1) = 7 ∧ 3*(-1)^2 + 2*a*(-1) + b = 0) :
  (∀ x ∈ set.Icc (-2 : ℝ) 4, f x ≤ 7 ∧ f x ≥ -25) :=
sorry

end intervals_of_monotonicity_extremum_values_on_interval_l741_741668


namespace next_ten_winners_each_receive_160_l741_741122

def total_prize : ℕ := 2400
def first_winner_share : ℚ := 1 / 3 * total_prize
def remaining_after_first : ℚ := total_prize - first_winner_share
def next_ten_winners_share : ℚ := remaining_after_first / 10

theorem next_ten_winners_each_receive_160 :
  next_ten_winners_share = 160 := by
sorry

end next_ten_winners_each_receive_160_l741_741122


namespace coeff_x2_p1_times_p2_l741_741475

noncomputable def p1 : Polynomial ℤ := 3 * X^4 - 2 * X^3 + 4 * X^2 - 3 * X - 1
noncomputable def p2 : Polynomial ℤ := 2 * X^3 - X^2 + 5 * X - 4

theorem coeff_x2_p1_times_p2 :
  (p1 * p2).coeff 2 = -31 := 
sorry

end coeff_x2_p1_times_p2_l741_741475


namespace difference_max_min_coins_l741_741372

def john_owes (c: ℕ) : Prop := c = 60
def coin (n: ℕ) : Prop := n = 5 ∨ n = 20 ∨ n = 50

theorem difference_max_min_coins (h₁ : john_owes 60)
(h₂ : ∀ n : ℕ, coin n -> n = 5 ∨ n = 20 ∨ n = 50) :
∃ min_coins max_coins : ℕ, min_coins = 3 ∧ max_coins = 12 ∧ (max_coins - min_coins) = 9 :=
begin
  sorry
end

end difference_max_min_coins_l741_741372


namespace orthocenter_of_triangle_l741_741330

noncomputable theory

open_locale classical

-- Define the points A, B, and C in 3D space
def A : ℝ × ℝ × ℝ := (2, 3, 4)
def B : ℝ × ℝ × ℝ := (6, 4, 2)
def C : ℝ × ℝ × ℝ := (4, 5, 6)

-- Define the orthocenter H we want to prove
def H : ℝ × ℝ × ℝ := (1/2, 8, 1/2)

-- The statement that H is the orthocenter of triangle ABC
theorem orthocenter_of_triangle :
  ∃ H : ℝ × ℝ × ℝ, H = (1/2, 8, 1/2) := sorry

end orthocenter_of_triangle_l741_741330


namespace stratified_sampling_sample_size_l741_741966

theorem stratified_sampling_sample_size:
  ∀ (M F sampled_female n: ℕ), 
  M = 1200 ∧ F = 1000 ∧ sampled_female = 80 ∧
  (sampled_female:ℚ / n:ℚ) = (F:ℚ / (M + F):ℚ) →
  n = 176 :=
by
  intros M F sampled_female n h
  cases h,
  -- Conditions: M = 1200, F = 1000, sampled_female = 80, and stratified ratio equation
  sorry

end stratified_sampling_sample_size_l741_741966


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741890

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l741_741890


namespace distinct_arrays_for_48_chairs_with_conditions_l741_741517

theorem distinct_arrays_for_48_chairs_with_conditions : 
  ∃ n : ℕ, n = 7 ∧ 
    ∀ (m r c : ℕ), 
      m = 48 ∧ 
      2 ≤ r ∧ 
      2 ≤ c ∧ 
      r * c = m ↔ 
      (∃ (k : ℕ), 
         ((k = (m / r) ∧ r * (m / r) = m) ∨ (k = (m / c) ∧ c * (m / c) = m)) ∧ 
         r * c = m) → 
    n = 7 :=
by
  sorry

end distinct_arrays_for_48_chairs_with_conditions_l741_741517


namespace cupcake_distribution_l741_741457

theorem cupcake_distribution :
  ∃ (cupcakes_per_child : ℕ → ℕ), 
  (cupcakes_per_child 1 = 18) ∧ 
  (cupcakes_per_child 2 = 12) ∧ 
  (cupcakes_per_child 3 = 6) ∧ 
  (∃ ratio : list ℕ, ratio = [3, 2, 1]) ∧
  (∃ total_cupcakes : ℕ, total_cupcakes = 144) ∧
  (∃ total_children : ℕ, total_children = 12) ∧
  (∀ group, 1 ≤ group ∧ group ≤ 3 → cupcakes_per_child group = total_cupcakes * ratio.nth (group - 1) / (ratio.sum * total_children / 3)) :=
sorry

end cupcake_distribution_l741_741457


namespace integer_odd_iff_odd_number_of_odd_digits_l741_741817

theorem integer_odd_iff_odd_number_of_odd_digits
  (r : ℕ) (h_odd : r % 2 = 1) (a : ℕ → ℕ) (n : ℕ) (h_digits : ∀ i, 0 ≤ a i ∧ a i < r) :
  let N := ∑ i in Finset.range (n + 1), a i * r^i
  in (N % 2 = 1 ↔ (∑ i in Finset.range (n + 1), a i % 2) % 2 = 1) :=
by
  sorry

end integer_odd_iff_odd_number_of_odd_digits_l741_741817


namespace fewest_posts_required_l741_741158

theorem fewest_posts_required 
  (length length_side_two : ℕ)
  (interval : ℕ)
  (rock_wall_side : ℕ) 
  (shared_post : ℕ)
  (shared_post_num : ℕ)
  (rock_wall_length: ℕ := 120) 
  (dimension_one : ℕ := 45)
  (dimension_two: ℕ := 75)
  (interval_length : ℕ := 15)
  : length = (rock_wall_side+dimension_one)
  → dimension_two = rock_wall_side 
  → interval = (50 / 5 + 1)
  → shared_post = 3 / interval + 1 
  → shared_post_num = (2 * length) - shared_post 
  → ∀ total_post, total_post = shared_post_num 
  → total_post = 12 := 
begin 
  sorry 
end 

end fewest_posts_required_l741_741158


namespace part_a_midpoint_cycle_part_b_seventh_point_cycle_l741_741524

-- Defining the configuration for part (a)
variable {α : Type} [LinearOrderedField α]
variables {A B C : Point α}
variable {M : Point α} -- M is the midpoint of BC

theorem part_a_midpoint_cycle (hM : Midpoint M B C) :
  let N := intersection (line_thru M parallel AC) AB in
  let P := intersection (line_thru N parallel BC) AC in
  let Q := intersection (line_thru P parallel AB) BC in
  Q = M := 
sorry

-- Defining the configuration for part (b)
variable {M1 : Point α} -- M1 is a point on AB, not midpoint

theorem part_b_seventh_point_cycle (hM1 : OnLine M1 AB) (hM1_not_midpoint: ¬Midpoint M1 A B) :
  let M2 := intersection (line_thru M1 parallel AC) BC in
  let M3 := intersection (line_thru M2 parallel AB) AC in
  let M4 := intersection (line_thru M3 parallel BC) AB in
  let M5 := intersection (line_thru M4 parallel AC) BC in
  let M6 := intersection (line_thru M5 parallel AB) AC in
  let M7 := intersection (line_thru M6 parallel BC) AB in
  M7 = M1 :=
sorry

end part_a_midpoint_cycle_part_b_seventh_point_cycle_l741_741524


namespace find_principal_l741_741534

-- Define the principal P, the rate of interest R, and amounts A1 and A2
variables (P R : ℝ)
constant A1 : ℝ := 1717
constant A2 : ℝ := 1734

-- Define the conditions based on the problem
axiom h1 : A1 = P + P * R * 1 / 100
axiom h2 : A2 = P + P * R * 2 / 100

-- The goal is to prove that P = 1700 under these conditions
theorem find_principal : P = 1700 :=
by 
  sorry

end find_principal_l741_741534


namespace expand_and_simplify_l741_741572

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 4) + 6 = x^2 + x - 6 := by
  sorry

end expand_and_simplify_l741_741572


namespace john_initial_payment_l741_741362

-- Definitions based on the conditions from step a)
def cost_per_soda : ℕ := 2
def num_sodas : ℕ := 3
def change_received : ℕ := 14

-- Problem Statement: Prove that the total amount of money John paid initially is $20
theorem john_initial_payment :
  cost_per_soda * num_sodas + change_received = 20 := 
by
  sorry -- Proof steps are omitted as per instructions

end john_initial_payment_l741_741362


namespace best_comprehensive_survey_l741_741489

/-- Define the type of surveys and their suitability for a comprehensive survey -/
inductive Survey
| satisfaction_of_tourists
| security_check_subway
| types_of_fish
| lightbulb_lifespan

/-- Define the suitability for comprehensive surveys, returning whether a survey is suitable -/
def is_comprehensive (s : Survey) : Prop :=
  match s with
  | Survey.satisfaction_of_tourists => False
  | Survey.security_check_subway => True
  | Survey.types_of_fish => False
  | Survey.lightbulb_lifespan => False

/-- The proof statement that security_check_subway is the only comprehensive survey -/
theorem best_comprehensive_survey : 
  ∀ s : Survey, is_comprehensive s → s = Survey.security_check_subway :=
by
  intros s h
  cases s
  case satisfaction_of_tourists { contradiction }
  case security_check_subway { refl }
  case types_of_fish { contradiction }
  case lightbulb_lifespan { contradiction }

#eval best_comprehensive_survey

end best_comprehensive_survey_l741_741489


namespace insurance_compensation_correct_l741_741113

def actual_damage : ℝ := 300000
def deductible_percent : ℝ := 0.01
def deductible_amount : ℝ := deductible_percent * actual_damage
def insurance_compensation : ℝ := actual_damage - deductible_amount

theorem insurance_compensation_correct : insurance_compensation = 297000 :=
by
  -- To be proved
  sorry

end insurance_compensation_correct_l741_741113


namespace items_in_descending_order_l741_741523

-- Assume we have four real numbers representing the weights of the items.
variables (C S B K : ℝ)

-- The conditions given in the problem.
axiom h1 : S > B
axiom h2 : C + B > S + K
axiom h3 : K + C = S + B

-- Define a predicate to check if the weights are in descending order.
def DescendingOrder (C S B K : ℝ) : Prop :=
  C > S ∧ S > B ∧ B > K

-- The theorem to prove the descending order of weights.
theorem items_in_descending_order : DescendingOrder C S B K :=
sorry

end items_in_descending_order_l741_741523


namespace produce_mask_each_day_donation_packages_l741_741205

-- Define the context and variables
variables (x y m n : ℕ) (P: ℝ) (total_profit: ℝ)

-- Define equations and conditions
-- Conditions given: Total masks production equation, profit equation
def masks_conditions (x y : ℕ) : Prop := x + y = 80 ∧ (0.4 * x + 0.5 * y = 35)

-- Part 1 question
theorem produce_mask_each_day (x y : ℕ) (h: masks_conditions x y) : x = 50 ∧ y = 30 :=
sorry

-- Part 2 conditions: Donation and resulting profit conditions
def donation_conditions (m n : ℕ) : Prop := 
  (35 - 1.2 * m - 3 * n = 2) ∧ n ≤ m / 3

-- Part 2 question
theorem donation_packages (m n : ℕ) (h: donation_conditions m n) :
  (m = 15 ∧ n = 5) ∨ (m = 20 ∧ n = 3) ∨ (m = 25 ∧ n = 1) :=
sorry

end produce_mask_each_day_donation_packages_l741_741205


namespace max_largest_integer_l741_741705

theorem max_largest_integer (A B C D E : ℕ) (h₀ : A ≤ B) (h₁ : B ≤ C) (h₂ : C ≤ D) (h₃ : D ≤ E) 
(h₄ : (A + B + C + D + E) = 225) (h₅ : E - A = 10) : E = 215 :=
sorry

end max_largest_integer_l741_741705


namespace mean_of_xyz_l741_741832

theorem mean_of_xyz (mean7 : ℕ) (mean10 : ℕ) (x y z : ℕ) (h1 : mean7 = 40) (h2 : mean10 = 50) : (x + y + z) / 3 = 220 / 3 :=
by
  have sum7 := 7 * mean7
  have sum10 := 10 * mean10
  have sum_xyz := sum10 - sum7
  have mean_xyz := sum_xyz / 3
  sorry

end mean_of_xyz_l741_741832


namespace satellite_max_height_l741_741530

noncomputable def max_height_reached (g R H : ℝ) : Prop :=
  (1 / 2) * R * H = (1 / 2) * R^2

theorem satellite_max_height
  (g : ℝ := 10)   -- gravitational acceleration
  (R : ℝ := 6400) -- Earth's radius in km
  (H : ℝ := 6400) -- Expected maximum height reached
  (v_I : ℝ := (g * R) ^ (1 / 2)) -- first cosmic velocity
  (γ M : ℝ) -- gravitational constant and Earth's mass
  (h1 : g = γ * M / R^2) -- gravitational relation
  : max_height_reached g R H :=
by
  -- Conditions setup
  have h1' : γ = g * R^2 / M,
    from sorry, -- Derived from the condition g = γ * M / R^2

  -- Substitution and simplification steps to show
  -- (1 / 2) * R * H = (1 / 2) * R^2
  exact sorry


end satellite_max_height_l741_741530


namespace altitudes_bisect_angles_l741_741376

theorem altitudes_bisect_angles (A B C A1 B1 C1 : Point) (hA1 : is_foot_of_altitude A1 A B C) (hB1 : is_foot_of_altitude B1 B A C) (hC1 : is_foot_of_altitude C1 C A B) : 
  bisects_angles (altitude_through A B C) (triangle A1 B1 C1) :=
sorry

end altitudes_bisect_angles_l741_741376


namespace defective_bulbs_produced_l741_741097

theorem defective_bulbs_produced (total_bulbs : ℕ) (defective_rate : ℝ) (h_rate : defective_rate = 0.1) (h_bulbs : total_bulbs = 870) :
    total_bulbs * defective_rate = 87 :=
by
    rw [h_bulbs, h_rate]
    simp
    norm_num
    sorry

end defective_bulbs_produced_l741_741097


namespace JohnNeeds72Strings_l741_741371

def JohnHasToRestring3Basses : Nat := 3
def StringsPerBass : Nat := 4

def TwiceAsManyGuitarsAsBasses : Nat := 2 * JohnHasToRestring3Basses
def StringsPerNormalGuitar : Nat := 6

def ThreeFewerEightStringGuitarsThanNormal : Nat := TwiceAsManyGuitarsAsBasses - 3
def StringsPerEightStringGuitar : Nat := 8

def TotalStringsNeeded : Nat := 
  (JohnHasToRestring3Basses * StringsPerBass) +
  (TwiceAsManyGuitarsAsBasses * StringsPerNormalGuitar) +
  (ThreeFewerEightStringGuitarsThanNormal * StringsPerEightStringGuitar)

theorem JohnNeeds72Strings : TotalStringsNeeded = 72 := by
  calculate
  sorry

end JohnNeeds72Strings_l741_741371


namespace xiaoQianVisited_l741_741482

-- Define the statements made by each person
def xiaoZhaoStatement : Prop := ¬ visited (Xiao Zhao)
def xiaoQianStatement : Prop := visited (Xiao Li)
def xiaoSunStatement : Prop := visited (Xiao Qian)
def xiaoLiStatement : Prop := ¬ visited (Xiao Li)

-- Define the condition that only one person is lying
def onlyOneLied (statements : List Prop) : Prop :=
  (statements.count (λ s => ¬ s)) = 1

-- Main theorem to be proved
theorem xiaoQianVisited :
  onlyOneLied [
    xiaoZhaoStatement = ¬ visited (Xiao Zhao),
    xiaoQianStatement = visited (Xiao Li),
    xiaoSunStatement = visited (Xiao Qian),
    xiaoLiStatement = ¬ visited (Xiao Li)
  ] → visited (Xiao Qian) :=
by
  sorry

end xiaoQianVisited_l741_741482


namespace train_length_is_correct_l741_741847

noncomputable def length_of_train (train_speed_kmph : ℕ) (cross_time_minutes : ℕ) : ℕ := 
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let cross_time_seconds := cross_time_minutes * 60
  let total_distance := train_speed_mps * cross_time_seconds
  total_distance / 2

theorem train_length_is_correct (train_speed_kmph : ℕ) (cross_time_minutes : ℕ) :
  train_speed_kmph = 180 → cross_time_minutes = 1 → length_of_train train_speed_kmph cross_time_minutes = 1500 :=
begin
  intros h1 h2,
  simp [length_of_train, h1, h2],
  norm_num,
end

end train_length_is_correct_l741_741847


namespace restore_original_numbers_l741_741793

-- Definitions from problem conditions
def sequence : list string := ["T", "EL", "EK", "LA", "SS"]
def natural_numbers : list ℕ := [5, 12, 19, 26, 33]

-- Lean theorem statement
theorem restore_original_numbers :
  ∃ (T EL EK LA SS : ℕ) (a d : ℕ),
    T = a ∧ EL = a + d ∧ EK = a + 2 * d ∧ LA = a + 3 * d ∧ SS = a + 4 * d ∧
    a = 5 ∧ d = 7 ∧
    natural_numbers = [T, EL, EK, LA, SS] := by
  sorry

end restore_original_numbers_l741_741793


namespace market_value_of_13_percent_stock_yielding_8_percent_l741_741509

noncomputable def market_value_of_stock (yield rate dividend_per_share : ℝ) : ℝ :=
  (dividend_per_share / yield) * 100

theorem market_value_of_13_percent_stock_yielding_8_percent
  (yield_rate : ℝ) (dividend_per_share : ℝ) (market_value : ℝ)
  (h_yield_rate : yield_rate = 0.08)
  (h_dividend_per_share : dividend_per_share = 13) :
  market_value = 162.50 :=
by
  sorry

end market_value_of_13_percent_stock_yielding_8_percent_l741_741509


namespace exactly_two_students_choose_math_l741_741948

-- Define the properties and conditions.
def students : Type := {A, B, C, D}
def courses : Type := {mathematics, physics, chemistry, biology}

noncomputable def choose_math_course : ℕ :=
  (Nat.choose 4 2) * (3 ^ 2)

-- State the theorem to prove.
theorem exactly_two_students_choose_math :
  choose_math_course = 54 :=
by
  unfold choose_math_course
  sorry

end exactly_two_students_choose_math_l741_741948


namespace triangle_ABC_area_equals_240_l741_741717

variables {A B C D E F : Type*} [EuclideanGeometry]
variables {triangle_area : Set (A × B × C)}
variables (midpoint : ∀ {x y}, ∃ z, z = (x + y) / 2)
variables (point_ratios : ∀ {a c}, ∃ e, ∃ f, (ae_ratio : a / c = 1 / 2) (af_ratio : a / d = 1 / 3))
variables (area_DEF : A × B × C → ℝ)
variables (area_given : area_DEF D E F = 30)

theorem triangle_ABC_area_equals_240 :
  area_given → triangle_area (A, B, C) 240 :=
by 
  sorry

end triangle_ABC_area_equals_240_l741_741717


namespace probability_not_snowing_l741_741448

theorem probability_not_snowing (P_snowing : ℚ) (h : P_snowing = 2/7) :
  (1 - P_snowing) = 5/7 :=
sorry

end probability_not_snowing_l741_741448


namespace solve_logarithmic_system_l741_741492

-- Define the given conditions as hypotheses
variables (a b x y : ℝ)
hypothesis h1 : log a x + log a y = 2
hypothesis h2 : log b x - log b y = 4

-- State the goal to prove
theorem solve_logarithmic_system (a b : ℝ) (h1 : log a x + log a y = 2) (h2 : log b x - log b y = 4) :
  x = a * b^2 ∧ y = a / (b^2) :=
by
  -- The proof is omitted
  sorry

end solve_logarithmic_system_l741_741492


namespace distance_focus_directrix_l741_741838

theorem distance_focus_directrix (y x : ℝ) (h : y^2 = 2 * x) : x = 1 := 
by 
  sorry

end distance_focus_directrix_l741_741838


namespace smallest_number_with_55_divisors_l741_741592

theorem smallest_number_with_55_divisors : ∃ n : ℕ, 
  (number_of_divisors n = 55) ∧ (∀ m : ℕ, number_of_divisors m = 55 → n ≤ m) := 
sorry

end smallest_number_with_55_divisors_l741_741592


namespace triangle_area_is_correct_l741_741988

-- Define the coordinate points and the hyperbola equation
def A : ℝ × ℝ := (-1, 0)

-- Define point B and C to be on the right branch of hyperbola such that triangle ABC is equilateral
variable {B C : ℝ × ℝ}
variable (h1 : B.1^2 - B.2^2 = 1)
variable (h2 : C.1^2 - C.2^2 = 1)
variable (h_eq_triangle : dist A B = dist B C ∧ dist B C = dist C A)

noncomputable def area_triangle : ℝ :=
  let s := dist A B in
  (s^2 * sqrt 3) / 4

theorem triangle_area_is_correct :
  let A : ℝ × ℝ := (-1, 0) in 
  ∀ (B C : ℝ × ℝ), 
  B.1^2 - B.2^2 = 1 → 
  C.1^2 - C.2^2 = 1 → 
  dist A B = dist B C ∧ dist B C = dist C A → 
  area_triangle = 3 * sqrt 3 := sorry

end triangle_area_is_correct_l741_741988


namespace find_m_l741_741641

theorem find_m (x y m : ℤ) (h1 : x = 2) (h2 : y = -3) (h3 : 3 * x - 4 * (m - 1) * y + 30 = 0) : m = -2 :=
by
  sorry

end find_m_l741_741641


namespace polygon_sides_l741_741780

theorem polygon_sides {n k : ℕ} (h1 : k = n * (n - 3) / 2) (h2 : k = 3 * n / 2) : n = 6 :=
by
  sorry

end polygon_sides_l741_741780


namespace part_a_part_b_l741_741611

/-- Definition of the sequence of numbers on the cards -/
def card_numbers (n : ℕ) : ℕ :=
  if n = 0 then 1 else (10^(n + 1) - 1) / 9 * 2 + 1

/-- Part (a) statement: Is it possible to choose at least three cards such that 
the sum of the numbers on them equals a number where all digits except one are twos? -/
theorem part_a : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ card_numbers a + card_numbers b + card_numbers c % 10 = 2 ∧ 
  (∀ d, ∃ k ≤ 1, (card_numbers a + card_numbers b + card_numbers c / (10^d)) % 10 = 2) :=
sorry

/-- Part (b) statement: Suppose several cards were chosen such that the sum of the numbers 
on them equals a number where all digits except one are twos. What could be the digit that is not two? -/
theorem part_b (sum : ℕ) :
  (∀ d, sum / (10^d) % 10 = 2) → ((sum % 10 = 0) ∨ (sum % 10 = 1)) :=
sorry

end part_a_part_b_l741_741611


namespace find_radius_of_circle_l741_741515

noncomputable def radius_of_circle (y x : ℝ) (h : x + 2 * y = 100 * Real.pi) : ℝ :=
  let r := (sqrt (416 - 16) - 4) / 2
  r

theorem find_radius_of_circle (y x : ℝ) (h1 : y = 2 * Real.pi * (radius_of_circle y x h1))
                             (h2 : x = Real.pi * (radius_of_circle y x h1) ^ 2) :
  x + 2 * y = 100 * Real.pi → (radius_of_circle y x h1 = 8 ∨ radius_of_circle y x h1 ≈ 8.198) :=
by
  sorry

end find_radius_of_circle_l741_741515


namespace fe_equals_2_l741_741665

noncomputable def f : ℝ → ℝ
| x := if x < 1 then Real.exp x + 1 else f (Real.log x)

theorem fe_equals_2 : f Real.exp Real.e = 2 :=
by
  sorry

end fe_equals_2_l741_741665


namespace vasya_example_fewer_sevens_l741_741877

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l741_741877


namespace inverse_of_4_l741_741993

theorem inverse_of_4 :
  4^(-1) = 1 / 4 := by
  sorry

end inverse_of_4_l741_741993


namespace range_of_a_l741_741439

theorem range_of_a (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) 
  (h_decreasing : ∀ x y : ℝ, x < y → f a x ≥ f a y) 
  : (1 / 3 ≤ a) ∧ (a < 1) := by 
  sorry

noncomputable def f (a : ℝ) : ℝ → ℝ 
| x if x < 0 := -x + 3 * a
| x := a^x 

end range_of_a_l741_741439


namespace reflection_line_is_x_eq_0_l741_741870

-- Define the points P, Q, R and their reflections P', Q', R'
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (5, 7)
def R : ℝ × ℝ := (-2, 5)

def P' : ℝ × ℝ := (-1, 2)
def Q' : ℝ × ℝ := (-5, 7)
def R' : ℝ × ℝ := (2, 5)

-- Define the equation for the line of reflection M
def line_of_reflection : ℝ → Prop := fun x => x = 0

theorem reflection_line_is_x_eq_0 : ∀ P Q R P' Q' R', 
  P = (1, 2) ∧ Q = (5, 7) ∧ R = (-2, 5) ∧ 
  P' = (-1, 2) ∧ Q' = (-5, 7) ∧ R' = (2, 5) → 
  ∃ M, line_of_reflection M := 
by {
  intros P Q R P' Q' R',
  intro h,
  use 0,
  simp only [line_of_reflection],
  sorry
}

end reflection_line_is_x_eq_0_l741_741870


namespace number_of_divisions_l741_741918

theorem number_of_divisions (n m : ℕ) (h : n * m = 604800) :
  (∃ n m : ℕ, n * m = 604800) ∧ (∀ k (p : k * p = 604800), (n, m) = (k, p) ∨ (n, m) = (p, k))
    → 90 := sorry

end number_of_divisions_l741_741918


namespace example_one_example_two_l741_741889

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l741_741889


namespace sequence_values_l741_741673

-- Define parameters for the sequence
def sequence (a : ℕ → ℕ) := 
  (∀ n : ℕ, n > 0 → a (4 * n - 3) = 1) ∧ 
  (∀ n : ℕ, n > 0 → a (4 * n - 1) = 0) ∧ 
  (∀ n : ℕ, n > 0 → a (2 * n) = a (n))

-- Define the problem statement
theorem sequence_values (a : ℕ → ℕ) (h : sequence a) : 
  a 2009 = 1 ∧ a 2004 = 1 :=
by
  sorry

end sequence_values_l741_741673


namespace bob_km_per_gallon_l741_741179

-- Define the total distance Bob can drive.
def total_distance : ℕ := 100

-- Define the total amount of gas in gallons Bob's car uses.
def total_gas : ℕ := 10

-- Define the expected kilometers per gallon
def expected_km_per_gallon : ℕ := 10

-- Define the statement we want to prove
theorem bob_km_per_gallon : total_distance / total_gas = expected_km_per_gallon :=
by 
  sorry

end bob_km_per_gallon_l741_741179


namespace Maximum_abcd_value_l741_741325

theorem Maximum_abcd_value {a b c d e f : ℝ} (h1: a ≤ 1) (h2: b ≤ 1) (h3: c ≤ 1) (h4: d ≤ 1) (h5: e ≤ 1) (h6: f ≤ 1) (h7: max a (max b (max c (max d (max e f)))) = 1) : 
  ∃ x, x = a * b * c * d ∧ x ≤ 2 - real.sqrt 3 :=
sorry

end Maximum_abcd_value_l741_741325


namespace range_of_a_l741_741303

theorem range_of_a (a : ℝ) (h : Real.sqrt ((2 * a - 1)^2) = 1 - 2 * a) : a ≤ 1 / 2 :=
sorry

end range_of_a_l741_741303


namespace ratio_inscribed_sphere_height_l741_741647

theorem ratio_inscribed_sphere_height (H R : ℝ) (h_tetrahedron : regular_tetrahedron height H inscribed_sphere_radius R) :
  R / H = 1 / 4 :=
sorry

end ratio_inscribed_sphere_height_l741_741647


namespace vasya_100_using_fewer_sevens_l741_741910

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l741_741910


namespace perimeter_independent_of_theta_l741_741987

-- Define the basic properties of the ellipse
def ellipse_foci (a : ℝ) (b : ℝ) := 
  let c := sqrt (a^2 - b^2) in 
  (c, -c)

-- Define the perimeter function of the triangle MNF1 given ellipse properties
noncomputable def perimeter_triangle_MNF1 (a b θ : ℝ) : ℝ := 
  -- Based on geometric properties of the ellipse the perimeter is constant.
  let P := 2 * a in 
  P

-- Lean statement to show perimeter P is independent of θ
theorem perimeter_independent_of_theta (a b θ : ℝ) : 
  ∀ θ1 θ2 : ℝ, perimeter_triangle_MNF1 a b θ1 = perimeter_triangle_MNF1 a b θ2 := 
by
  -- Proof goes here
  sorry

end perimeter_independent_of_theta_l741_741987


namespace minimum_rolls_for_duplicate_sum_l741_741054

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741054


namespace apple_price_l741_741956

theorem apple_price (emmy_money gerry_money total_apples: ℕ) 
  (h1: emmy_money = 200) (h2: gerry_money = 100) 
  (h3: total_apples = 150) (h4: emmy_money + gerry_money = 300) : 
  total_apples * 2 = emmy_money + gerry_money := 
by
  calc
    total_apples * 2 = 150 * 2 : by rw h3
                   ... = 300    : by norm_num
                   ... = emmy_money + gerry_money : by rw [h4];
  
#check apple_price

end apple_price_l741_741956


namespace linear_function_relationship_price_for_profit_maximize_profit_with_donation_l741_741129

-- Part 1: Linear relationship proof
theorem linear_function_relationship 
  (points : List (ℝ × ℝ)) 
  (h1 : (80, 240) ∈ points)
  (h2 : (90, 220) ∈ points)
  (h3 : ∀ x y, (x, y) ∈ points → y = k * x + b → ∃ k b, y = -2 * x + 400) :
  ∃ k b, ∀ x y, (x, y) ∈ points → y = -2 * x + 400 := 
sorry

-- Part 2: Price for 8000 yuan profit
theorem price_for_profit (x : ℝ) 
  (profit_equation : ℝ → ℝ)
  (h : profit_equation x = (x - 60) * (-2 * x + 400))
  (profit_target : profit_equation x = 8000) :
  x = 100 :=
sorry

-- Part 3: Maximizing profit with donation
theorem maximize_profit_with_donation (x : ℝ)
  (profit_equation_with_donation : ℝ → ℝ)
  (h : profit_equation_with_donation x = (x - 70) * (-2 * x + 400)) :
  ∃ x, ∀ x, profit_equation_with_donation x = -2 * (x - 135)^2 + 8450 :=
sorry

end linear_function_relationship_price_for_profit_maximize_profit_with_donation_l741_741129


namespace nth_pattern_equation_l741_741401

theorem nth_pattern_equation (n : ℕ) : 
  (finset.range n).sum (λ k, 8 * (k + 1) - 4) = (2 * n) ^ 2 :=
by
  sorry

end nth_pattern_equation_l741_741401


namespace minimum_throws_for_repeated_sum_l741_741048

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741048


namespace average_last_part_l741_741430

theorem average_last_part (sum_25_results : ℝ)
  (sum_first_12_results : ℝ)
  (thirteenth_result : ℝ) :
  (sum_25_results = 1250 ∧ sum_first_12_results = 168 ∧ thirteenth_result = 878)
  → (13 * (1250 - (168 + 878)) / 13 = 15.69) :=
by
  intro h
  cases h with h_sum_25_results h_rest
  cases h_rest with h_sum_first_12_results h_thirteenth_result
  rw h_sum_25_results at *
  rw h_sum_first_12_results at *
  rw h_thirteenth_result at *
  norm_num
  sorry

end average_last_part_l741_741430


namespace quadratic_inequality_l741_741422

theorem quadratic_inequality : ∀ x : ℝ, -7 * x ^ 2 + 4 * x - 6 < 0 :=
by
  intro x
  have delta : 4 ^ 2 - 4 * (-7) * (-6) = -152 := by norm_num
  have neg_discriminant : -152 < 0 := by norm_num
  have coef : -7 < 0 := by norm_num
  sorry

end quadratic_inequality_l741_741422


namespace triangle_BC_length_l741_741715

theorem triangle_BC_length
  (A B C : Type)
  [InnerProductSpace ℝ A]
  {a b c : A}
  (h1 : ∠A b c = π / 3) -- ∠A corresponds to 60 degrees
  (h2 : ‖b - a‖ = 1) -- AC = 1
  (area : ℝ)
  (h3 : area = sqrt 3 / 2) -- Area = √3 / 2
  : ‖c - b‖ = sqrt 3 := -- BC = √3
sorry

end triangle_BC_length_l741_741715


namespace side_length_of_square_eq_two_l741_741230

theorem side_length_of_square_eq_two
    (a : ℝ)
    (R : ℝ := a * Real.sqrt 2 / 2)
    (semicircle_area : ℝ := (a^2 * Real.pi) / 8)
    (triangle_area : ℝ := a^2 / 4)
    (segment_area : ℝ := (a^2 / 4) * (Real.pi / 2 - 1))
    (half_moon_area : ℝ := (Real.pi / 8) * a^2 - (a^2 / 4) * (Real.pi / 2 - 1)) :
    (∀ k, half_moon_area = 1 ) → a = 2 :=
by {
  intro h,
  sorry
}

end side_length_of_square_eq_two_l741_741230


namespace sum_of_m_with_distinct_integer_roots_l741_741084

theorem sum_of_m_with_distinct_integer_roots :
  (∑ m in {m | ∃ (r s : ℤ), r ≠ s ∧ (3 * r * r - m * r + 12 = 0) ∧ (3 * s * s - m * s + 12 = 0)}, m) = 0 :=
sorry

end sum_of_m_with_distinct_integer_roots_l741_741084


namespace smallest_number_with_55_divisors_l741_741606

theorem smallest_number_with_55_divisors :
  ∃ n : ℕ, (n = 3^4 * 2^{10}) ∧ (nat.count_divisors n = 55) :=
by
  have n : ℕ := 3^4 * 2^{10}
  exact ⟨n, ⟨rfl, nat.count_divisors_eq_count_divisors 3 4 2 10⟩⟩
  sorry

end smallest_number_with_55_divisors_l741_741606


namespace closest_sum_to_six_sevenths_l741_741235

theorem closest_sum_to_six_sevenths :
  (∃ S : set ℚ, 
    S = {1/2, 1/5, 1/6} ∧
    abs (∑ x in S, x - 6/7) ≤ abs (∑ x in {1/2, 1/3, 1/5}, x - 6/7) ∧
    abs (∑ x in S, x - 6/7) ≤ abs (∑ x in {1/2, 1/3, 1/6}, x - 6/7) ∧
    abs (∑ x in S, x - 6/7) ≤ abs (∑ x in {1/2, 1/4, 1/5}, x - 6/7) ∧
    abs (∑ x in S, x - 6/7) ≤ abs (∑ x in {1/2, 1/4, 1/6}, x - 6/7)) :=
sorry

end closest_sum_to_six_sevenths_l741_741235


namespace smallest_number_with_55_divisors_l741_741593

theorem smallest_number_with_55_divisors : ∃ n : ℕ, 
  (number_of_divisors n = 55) ∧ (∀ m : ℕ, number_of_divisors m = 55 → n ≤ m) := 
sorry

end smallest_number_with_55_divisors_l741_741593


namespace probability_of_both_tails_l741_741425

-- Definition of flipping three coins
def flip : Type := {nickel : bool, dime : bool, quarter : bool}

-- Possible outcomes when flipping three coins
def total_outcomes : ℕ := 2 ^ 3

-- Event where both the nickel and the quarter come up tails
def event_tails (f : flip) : Prop :=
  f.nickel = false ∧ f.quarter = false

-- Number of successful outcomes where both the nickel and the quarter are tails
def successful_outcomes : ℕ := 2

-- The probability that both the nickel and the quarter are tails
def probability_event_tails : ℚ :=
  successful_outcomes / total_outcomes

-- Theorem statement
theorem probability_of_both_tails : probability_event_tails = 1 / 4 := sorry

end probability_of_both_tails_l741_741425


namespace compute_expression_l741_741762

theorem compute_expression (x y z : ℝ) (h₀ : x ≠ y) (h₁ : y ≠ z) (h₂ : z ≠ x) (h₃ : x + y + z = 3) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = 9 / (2 * (x^2 + y^2 + z^2)) - 1 / 2 :=
by
  sorry

end compute_expression_l741_741762


namespace exists_set_with_divisibility_property_l741_741805

theorem exists_set_with_divisibility_property (n : ℕ) (hn : n ≥ 2) :
  ∃ S : Finset ℤ, S.card = n ∧ ∀ (a b ∈ S), a ≠ b → (a - b)^2 ∣ (a * b) :=
sorry

end exists_set_with_divisibility_property_l741_741805


namespace find_n_l741_741149

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map (λ c => c.toNat - ('0'.toNat))).sum

-- Main theorem statement
theorem find_n (n : ℕ) (h : 2 * n * sum_of_digits (3 * n) = 2022) : n = 337 :=
by
  sorry

end find_n_l741_741149


namespace playback_methods_proof_l741_741511

/-- A TV station continuously plays 5 advertisements, consisting of 3 different commercial advertisements
and 2 different Olympic promotional advertisements. The requirements are:
  1. The last advertisement must be an Olympic promotional advertisement.
  2. The 2 Olympic promotional advertisements can be played consecutively.
-/
def number_of_playback_methods (commercials olympics: ℕ) (last_ad_olympic: Bool) (olympics_consecutive: Bool) : ℕ :=
  if commercials = 3 ∧ olympics = 2 ∧ last_ad_olympic ∧ olympics_consecutive then 36 else 0

theorem playback_methods_proof :
  number_of_playback_methods 3 2 true true = 36 := by
  sorry

end playback_methods_proof_l741_741511


namespace JohnNeeds72Strings_l741_741369

def JohnHasToRestring3Basses : Nat := 3
def StringsPerBass : Nat := 4

def TwiceAsManyGuitarsAsBasses : Nat := 2 * JohnHasToRestring3Basses
def StringsPerNormalGuitar : Nat := 6

def ThreeFewerEightStringGuitarsThanNormal : Nat := TwiceAsManyGuitarsAsBasses - 3
def StringsPerEightStringGuitar : Nat := 8

def TotalStringsNeeded : Nat := 
  (JohnHasToRestring3Basses * StringsPerBass) +
  (TwiceAsManyGuitarsAsBasses * StringsPerNormalGuitar) +
  (ThreeFewerEightStringGuitarsThanNormal * StringsPerEightStringGuitar)

theorem JohnNeeds72Strings : TotalStringsNeeded = 72 := by
  calculate
  sorry

end JohnNeeds72Strings_l741_741369


namespace edge_length_of_cube_l741_741476

theorem edge_length_of_cube {V_cube V_cuboid : ℝ} (base_area : ℝ) (height : ℝ)
  (h1 : base_area = 10) (h2 : height = 73) (h3 : V_cube = V_cuboid - 1)
  (h4 : V_cuboid = base_area * height) :
  ∃ (a : ℝ), a^3 = V_cube ∧ a = 9 :=
by
  /- The proof is omitted -/
  sorry

end edge_length_of_cube_l741_741476


namespace relationship_y1_y2_l741_741634

theorem relationship_y1_y2 (y1 y2 : ℝ) 
  (h1 : (∃ y1, (y1 = -(-1 : ℝ) + 1)) : y1 = 2) 
  (h2 : (∃ y2, (y2 = -(2 : ℝ) + 1)) : y2 = -1) :
  y1 > y2 := 
  by sorry

end relationship_y1_y2_l741_741634


namespace arithmetic_seq_even_sum_l741_741340

-- Definitions based on conditions in a)
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) := ∀ n, a (n + 1) = a n + d

def sum_to_n (a : ℕ → ℝ) (n : ℕ) := (finset.range n).sum a

-- The main theorem to prove
theorem arithmetic_seq_even_sum (a : ℕ → ℝ) (a_1 : ℝ) (d : ℝ) (S_98 : ℝ) (h_seq : arithmetic_seq a d) 
(h_d : d = 1) (h_sum : sum_to_n a 98 = S_98) (h_S_98 : S_98 = 137) :
  (finset.range 49).sum (λ k, a (2 * (k + 1))) = 93 :=
sorry

end arithmetic_seq_even_sum_l741_741340


namespace becky_packs_lunch_days_l741_741861

-- Definitions of conditions
def school_days := 180
def aliyah_packing_fraction := 1 / 2
def becky_relative_fraction := 1 / 2

-- Derived quantities from conditions
def aliyah_pack_days := school_days * aliyah_packing_fraction
def becky_pack_days := aliyah_pack_days * becky_relative_fraction

-- Statement to prove
theorem becky_packs_lunch_days : becky_pack_days = 45 := by
  sorry

end becky_packs_lunch_days_l741_741861


namespace fraction_power_evaluation_l741_741306

theorem fraction_power_evaluation (x y : ℚ) (h1 : x = 2 / 3) (h2 : y = 3 / 2) : 
  (3 / 4) * x^8 * y^9 = 9 / 8 := 
by
  sorry

end fraction_power_evaluation_l741_741306


namespace part1_part2_l741_741285

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 6) + 2 * (cos x)^2 - 2

theorem part1 : ∀ k : ℤ, ∀ x : ℝ, 
  (-π/3 + k * π) ≤ x ∧ x ≤ (π/6 + k * π) → 
  f' x ≥ 0 := 
sorry

structure TriangleABC where
  A B C a b c : ℝ
  ha : c = sqrt 3
  hb : 2 * sin A = sin B
  hc : 2 * f C = -1

noncomputable def area (t : TriangleABC) : ℝ := 
  1/2 * t.a * t.b * sin t.C

theorem part2 (t : TriangleABC) (h : t.C = π/3 ∧ t.a = 1 ∧ t.b = 2) : 
  t.area = sqrt 3 / 2 :=
sorry

end part1_part2_l741_741285


namespace AnalyzeRelationship_l741_741867

noncomputable def analysisMethod := Prop
constant height_weight_relationship : Prop
constant regression_analysis : analysisMethod

theorem AnalyzeRelationship 
  (h : height_weight_relationship) : 
  regression_analysis = true := 
sorry

end AnalyzeRelationship_l741_741867


namespace conjugate_of_z_is_minus_i_l741_741433

-- Define the complex number z
def z : ℂ := (2 + I) / (1 - 2 * I)

-- Define what we're trying to prove
theorem conjugate_of_z_is_minus_i : complex.conj z = -I := sorry

end conjugate_of_z_is_minus_i_l741_741433


namespace notebook_problem_l741_741795

theorem notebook_problem
    (total_notebooks : ℕ)
    (cost_price_A : ℕ)
    (cost_price_B : ℕ)
    (total_cost_price : ℕ)
    (selling_price_A : ℕ)
    (selling_price_B : ℕ)
    (discount_A : ℕ)
    (profit_condition : ℕ)
    (x y m : ℕ) 
    (h1 : total_notebooks = 350)
    (h2 : cost_price_A = 12)
    (h3 : cost_price_B = 15)
    (h4 : total_cost_price = 4800)
    (h5 : selling_price_A = 20)
    (h6 : selling_price_B = 25)
    (h7 : discount_A = 30)
    (h8 : 12 * x + 15 * y = 4800)
    (h9 : x + y = 350)
    (h10 : selling_price_A * m + selling_price_B * m + (x - m) * selling_price_A * 7 / 10 + (y - m) * cost_price_B - total_cost_price ≥ profit_condition):
    x = 150 ∧ m ≥ 128 :=
by
    sorry

end notebook_problem_l741_741795


namespace protein_in_steak_is_correct_l741_741170

-- Definitions of the conditions
def collagen_protein_per_scoop : ℕ := 18 / 2 -- 9 grams
def protein_powder_per_scoop : ℕ := 21 -- 21 grams

-- Define the total protein consumed
def total_protein (collagen_scoops protein_scoops : ℕ) (protein_from_steak : ℕ) : ℕ :=
  collagen_protein_per_scoop * collagen_scoops + protein_powder_per_scoop * protein_scoops + protein_from_steak

-- Condition in the problem
def total_protein_consumed : ℕ := 86

-- Prove that the protein in the steak is 56 grams
theorem protein_in_steak_is_correct : 
  total_protein 1 1 56 = total_protein_consumed :=
sorry

end protein_in_steak_is_correct_l741_741170


namespace hexadecagon_area_l741_741146

theorem hexadecagon_area (P : ℝ) (n : ℕ) (s t : ℕ) :
  P = 160 → n = 16 → s = 4 → t = 2 →
  let side_length := P / 4 in
  let segment_length := side_length / (s + t * (s - 1)) in
  let square_area := side_length^2 in
  let triangle_area := (1/2) * segment_length^2 in
  let total_triangles_area := 8 * triangle_area in
  let hexadecagon_area := square_area - total_triangles_area in
  hexadecagon_area = 1344 :=
by
  intros P_eq n_eq s_eq t_eq
  let side_length := P / 4
  let segment_length := side_length / (s + t * (s - 1))
  let square_area := side_length^2
  let triangle_area := (1/2) * segment_length^2
  let total_triangles_area := 8 * triangle_area
  let hexadecagon_area := square_area - total_triangles_area
  have h1 : P = 160 := P_eq
  have h2 : n = 16 := n_eq
  have h3 : s = 4 := s_eq
  have h4 : t = 2 := t_eq
  sorry

end hexadecagon_area_l741_741146


namespace minimum_throws_for_repeated_sum_l741_741035

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741035


namespace problem_proof_l741_741192

variable {α : Type*}
noncomputable def op (a b : ℝ) : ℝ := 1/a + 1/b
theorem problem_proof (a b : ℝ) (h : op a (-b) = 2) : (3 * a * b) / (2 * a - 2 * b) = -3/4 :=
by
  sorry

end problem_proof_l741_741192


namespace quadratic_equation_with_given_means_l741_741704

-- Define conditions
def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2
def geometric_mean (a b : ℝ) : ℝ := real.sqrt (a * b)

-- Problem statement
theorem quadratic_equation_with_given_means (a b : ℝ) 
  (h1 : arithmetic_mean a b = 7)
  (h2 : geometric_mean a b = 8) :
  ∃ x : ℝ, polynomial.eval x (polynomial.C 64 + polynomial.X * -14 + polynomial.X^2) = 0 :=
begin
  sorry
end

end quadratic_equation_with_given_means_l741_741704


namespace find_m_decreasing_l741_741842

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem find_m_decreasing :
  ∃ m : ℝ, f m = λ x, (m^2 - m - 1) * x^(m^2 + m - 3) ∧ (∀ (x : ℝ), x > 0 → f (-1) x < f (-1) (x + 1)) ∧ m = -1 :=
sorry

end find_m_decreasing_l741_741842


namespace harry_min_additional_marbles_l741_741681

theorem harry_min_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 45) :
  (∑ k in finset.range (friends + 1), k) - marbles = 33 :=
by 
  sorry

end harry_min_additional_marbles_l741_741681


namespace number_of_people_in_room_l741_741866

-- Definitions based on conditions
def people := ℕ
def chairs := ℕ

-- Three-fifths of people are seated in four-fifths of the chairs.
constant three_fifths_people_seated : ℚ := 3 / 5

-- Four-fifths of the chairs are occupied.
constant four_fifths_chairs_occupied : ℚ := 4 / 5

-- There are 10 empty chairs.
constant empty_chairs : chairs := 10

-- The proof problem statement
theorem number_of_people_in_room :
  ∃ (total_people : people) (total_chairs : chairs),
  (empty_chairs * 5 = total_chairs) ∧
  (total_chairs * four_fifths_chairs_occupied = 40) ∧
  (total_people * three_fifths_people_seated = 40) ∧
  (total_people = 67) :=
begin
  sorry
end

end number_of_people_in_room_l741_741866


namespace lateral_surface_area_l741_741451

theorem lateral_surface_area (a : ℝ) (S : ℝ)
  (h_angle : ∀ P K M : ℝ, angle_between_lateral_face_and_base P K M = 45) :
  S = a^2 * sqrt 2 := 
sorry

end lateral_surface_area_l741_741451


namespace volleyball_teams_l741_741935

theorem volleyball_teams (n : ℕ)
  (h1 : ∀ (a b : ℕ), a ∈ (Finset.range n) → b ∈ (Finset.range (2 * n)) → a ≠ b → (¬(∃ c : ℤ, c < 0)))
  (h2 : ratio n (2 * n) 3 4) :
  n = 5 :=
by sorry

end volleyball_teams_l741_741935


namespace number_of_terms_arithmetic_sequence_l741_741709

theorem number_of_terms_arithmetic_sequence
  (a₁ d n : ℝ)
  (h1 : a₁ + (a₁ + d) + (a₁ + 2 * d) = 34)
  (h2 : (a₁ + (n-3) * d) + (a₁ + (n-2) * d) + (a₁ + (n-1) * d) = 146)
  (h3 : n / 2 * (2 * a₁ + (n-1) * d) = 390) :
  n = 11 :=
by sorry

end number_of_terms_arithmetic_sequence_l741_741709


namespace repeating_decimal_to_fraction_l741_741214

noncomputable def repeating_decimal_solution : ℚ := 7311 / 999

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 7 + 318 / 999) : x = repeating_decimal_solution := 
by
  sorry

end repeating_decimal_to_fraction_l741_741214


namespace probability_calculation_l741_741808

noncomputable def probability_of_event : ℝ :=
  let x_domain := Icc (0 : ℝ) 2 -- interval [0, 2]
  let event := Icc (0 : ℝ) (3 / 2) -- interval that satisfies the event
  (volume event) / (volume x_domain)

theorem probability_calculation :
  probability_of_event = 3 / 4 :=
sorry

end probability_calculation_l741_741808


namespace arrange_grid_l741_741844

-- Define the grid as a 10 × 10 array of integers
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

-- State that we have a 10 x 10 grid with integers 1 to 100
structure grid10x10 :=
(arr : array (fin 10 × fin 10) ℕ)
(h : ∀ i j : fin 10, 1 ≤ arr.read i j ∧ arr.read i j ≤ 100)

-- Define adjacency (sharing an edge)
def adjacent (i j : fin 10 × fin 10) (i' j' : fin 10 × fin 10) : Prop :=
(i = i' ∧ (j.val + 1 = j'.val ∨ j.val = j'.val + 1))
∨ (j = j' ∧ (i.val + 1 = i'.val ∨ i.val = i'.val + 1))

-- Define the condition that sums of adjacent cells are composite
def composite_adjacency (g : grid10x10) : Prop :=
∀ i j i' j', adjacent (i, j) (i', j') → is_composite (g.arr.read i j + g.arr.read i' j')

-- The actual theorem statement
theorem arrange_grid (g : grid10x10) :
  ∃ swaps : list (fin 10 × fin 10 × fin 10 × fin 10),
    swaps.length ≤ 35 ∧ 
    let g' := swaps.foldl (λ g p, let ⟨i, j, i', j'⟩ := p in g.update (i, j) (g.read i' j').update (i', j') (g.read i j)) g in
    composite_adjacency g' :=
sorry

end arrange_grid_l741_741844


namespace find_f_pi_over_2_l741_741272

noncomputable def phi : ℝ := -π / 4
noncomputable def omega : ℝ := 3
noncomputable def f (x : ℝ) : ℝ := Real.sin (omega * x + phi)

theorem find_f_pi_over_2 : f (π / 2) = -Real.sqrt 2 / 2 :=
by
  -- The proof will be inserted here
  sorry

end find_f_pi_over_2_l741_741272


namespace minimum_throws_for_repeated_sum_l741_741043

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l741_741043


namespace platform_length_correct_l741_741145

noncomputable def platform_length (train_speed_kmph : ℝ) (crossing_time_s : ℝ) (train_length_m : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_covered := train_speed_mps * crossing_time_s
  distance_covered - train_length_m

theorem platform_length_correct :
  platform_length 72 26 260.0416 = 259.9584 :=
by
  sorry

end platform_length_correct_l741_741145


namespace infinite_product_base_three_eq_nine_l741_741995

theorem infinite_product_base_three_eq_nine :
  ∏' (n : ℕ) (h : n > 0), (3^n)^(1 / (2^n)) = 9 :=
sorry

end infinite_product_base_three_eq_nine_l741_741995


namespace eccentricity_locus_A_l741_741630

structure Point :=
  (x : ℝ)
  (y : ℝ)

def perimeter (A B C : Point) : ℝ :=
  (real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)) + 
  (real.sqrt ((A.x - C.x)^2 + (A.y - C.y)^2)) + 
  (real.sqrt ((B.x - C.x)^2 + (B.y - C.y)^2))

noncomputable def e := 2 / 3 

theorem eccentricity_locus_A :
  ∃ (A : Point), 
    let B := Point.mk (-2) 0,
        C := Point.mk 2 0 
    in perimeter A B C = 10 → e = 2 / 3 :=
by
  sorry

end eccentricity_locus_A_l741_741630


namespace correct_statements_are_1_and_4_l741_741166

theorem correct_statements_are_1_and_4 :
  (∀ (x y : ℝ), x + y ≠ 3 → x ≠ 2 ∨ y ≠ 1) ∧
  (¬ (¬ (∀ (p q : Prop), (¬ p → ¬ q) → p → q)) ∨
  ∃ a b : ℝ, a > b ∧ ¬ ((1/a) < (1/b))) ∧
  (∃ (x : ℝ), x^2 = 1 → ¬ ∀ (x : ℝ), x^2 ≠ 1) :=
by
  → sorry

end correct_statements_are_1_and_4_l741_741166


namespace count_sets_with_six_and_sum_eighteen_l741_741174

/-- We want to prove the number of sets of three different numbers drawn from {2, 3, 4, 5, 6, 7, 8, 10, 11}
where one of the numbers is 6 and their sum is 18 is 3. -/
theorem count_sets_with_six_and_sum_eighteen :
  let numbers := {2, 3, 4, 5, 6, 7, 8, 10, 11},
      valid_triples := {x : finset ℕ // x ⊂ numbers ∧ 6 ∈ x ∧ x.card = 3 ∧ x.sum = 18} in
  valid_triples.card = 3 :=
by
sorry

end count_sets_with_six_and_sum_eighteen_l741_741174


namespace sum_lines_repetition_l741_741946

theorem sum_lines_repetition (grid : Fin 5 → Fin 5 → {x : ℕ // x = 1 ∨ x = 3 ∨ x = 5 ∨ x = 7}) :
  ∃ (line1 line2 : Fin 20), line1 ≠ line2 ∧ 
    (let sum_line := λ (i : Fin 20), 
      if h : i < 5 then 
        ∑ j : Fin 5, (grid h j).val
      else if h : i < 10 then 
        ∑ j : Fin 5, (grid j (i - 5)).val
      else 
        ∑ ⟨x, y⟩ in (diagonals (i - 10)), (grid x y).val
    in sum_line line1 = sum_line line2) :=
  sorry

end sum_lines_repetition_l741_741946


namespace ratio_BF_FC_l741_741173

variable (a b : ℝ) (A B C D E F O : Point)

variables [rect : Rectangle A B C D]
variables [hE : PointOn E (line_through D C)]
variables [hF : PointOn F (line_through B C)]
variables [hRatio_E : SegmentRatio D E E C (2:3)]
variables [hAF_BE : IntersectLinesPoint (line_through A F) (line_through B E) O]
variables [hRatio_AO_OF : SegmentRatio A O O F (5:2)]

theorem ratio_BF_FC : RatioOfSegments B F F C (2:1) :=
sorry

end ratio_BF_FC_l741_741173


namespace derivative_f_l741_741305

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) * (Real.cos x + Real.sin x)

theorem derivative_f : ∀ x : ℝ, (Real.deriv f) x = -2 * Real.exp (-x) * Real.sin x :=
by
  intro x
  -- Here, sorry is used as a placeholder for the actual proof
  sorry

end derivative_f_l741_741305


namespace percent_profit_l741_741105

namespace ProfitCalculation

-- Define the cost price and selling price of an article
variables {C : ℝ} {S : ℝ}

-- Define the main condition given: cost price of 55 articles equals selling price of 50 articles
def condition_1 : Prop := 55 * C = 50 * S

-- Define the theorem that we want to prove
theorem percent_profit (h : condition_1) : 10 = 100 * (((S - C) / C) / 10) :=
by sorry

end ProfitCalculation

end percent_profit_l741_741105


namespace total_sum_lent_l741_741972

-- Conditions
def interest_equal (x y : ℕ) : Prop :=
  (x * 3 * 8) / 100 = (y * 5 * 3) / 100

def second_sum : ℕ := 1704

-- Assertion
theorem total_sum_lent : ∃ x : ℕ, interest_equal x second_sum ∧ (x + second_sum = 2769) :=
  by
  -- Placeholder proof
  sorry

end total_sum_lent_l741_741972


namespace problem1_problem2_l741_741119

-- Problem (1)
theorem problem1 (a b : ℝ) (i : ℂ) (hi : i = complex.I) (h : (a + i) * (1 + i) = b * i) : a = 1 ∧ b = 2 :=
sorry

-- Problem (2)
theorem problem2 (m : ℝ) (i : ℂ) (hi : i = complex.I) (h : ∃ c : ℝ, (m^2 + m - 2) + (m^2 - 1) * i = c * i) : m = -2 :=
sorry

end problem1_problem2_l741_741119


namespace total_strings_needed_l741_741364

def basses := 3
def strings_per_bass := 4
def guitars := 2 * basses
def strings_per_guitar := 6
def eight_string_guitars := guitars - 3
def strings_per_eight_string_guitar := 8

theorem total_strings_needed :
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := by
  sorry

end total_strings_needed_l741_741364


namespace min_throws_to_ensure_same_sum_twice_l741_741079

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l741_741079


namespace proof_problem_l741_741338

noncomputable def parametric_C1 (m: ℝ) : ℝ × ℝ :=
  (sqrt m + 1 / sqrt m, sqrt m - 1 / sqrt m)

def curve_C2 (theta : ℝ) := 4 * real.sin (theta - real.pi / 6)

def P_cartesian : ℝ × ℝ :=
  (1, real.sqrt 3)

def line_l_parametric (t: ℝ) : ℝ × ℝ :=
  (1 - t / 2, real.sqrt 3 + (real.sqrt 3) * t / 2)

-- Lean 4 Statement
theorem proof_problem :
  (∀ m : ℝ, let ⟨x, y⟩ := parametric_C1 m in x^2 - y^2 = 4) ∧
  (curve_C2 (real.pi / 3) = 2) ∧
  (∀ t : ℝ, let ⟨x, y⟩ := line_l_parametric t in x^2 - y^2 = 4 -> abs t + abs (-8 - t) = 8) :=
by
  sorry

end proof_problem_l741_741338


namespace max_and_min_l741_741585

open Real

-- Define the function
def y (x : ℝ) : ℝ := 3 * x^4 - 6 * x^2 + 4

-- Define the interval
def a : ℝ := -1
def b : ℝ := 3

theorem max_and_min:
  ∃ (xmin xmax : ℝ), xmin = 1 ∧ xmax = 193 ∧
  (∀ x, a ≤ x ∧ x ≤ b → y(xmin) ≤ y x ∧ y x ≤ y(xmax)) :=
by
  sorry

end max_and_min_l741_741585


namespace volume_of_pyramid_l741_741964

noncomputable def pyramid_volume (length width height edge : ℝ) : ℝ :=
  (1 / 3) * (length * width) * height

theorem volume_of_pyramid
  (length width edge : ℝ)
  (h1 : length = 7)
  (h2 : width = 9)
  (h3 : edge = 15)
  (h4 : let diagonal := real.sqrt (length^2 + width^2) in 
    let half_diagonal := diagonal / 2 in 
    let height := real.sqrt (edge^2 - half_diagonal^2) in 
    pyramid_volume length width height edge = 21 * real.sqrt 192.5) :
  pyramid_volume length width height edge = 21 * real.sqrt 192.5 := 
sorry

end volume_of_pyramid_l741_741964


namespace minimum_throws_l741_741034

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741034


namespace minimum_throws_l741_741032

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l741_741032


namespace min_max_terms_seq_l741_741670

theorem min_max_terms_seq :
  (∀ n : ℕ, a_n = (n - 2017.5) / (n - 2016.5)) →
  ∃ min max, (∀ n : ℕ, a_n ≥ min ∧ a_n ≤ max) ∧ min = -1 ∧ max = 3 :=
by
  sorry

end min_max_terms_seq_l741_741670


namespace domain_of_f_l741_741919

noncomputable def f (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_f :
  ∀ x, (6:ℝ)^(625:ℝ) < x → ∃ y, f x = y :=
by
  sorry

end domain_of_f_l741_741919


namespace minimum_rolls_for_duplicate_sum_l741_741057

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l741_741057
