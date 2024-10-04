import Data.List.Basic
import Data.Nat.Basic
import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.LinearEquiv
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.AreaUnderCurve
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.SpecialFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.SimpleGraph.Coloring
import Mathlib.Data.Complex.Roots
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Floor
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle
import Mathlib.Init.Data.Int.Basic
import Mathlib.NumberTheory.Coprime
import Mathlib.Probability.Basic
import Mathlib.SetTheory.Cardinal.Finite
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.RingExp
import data.complex.exponential
import data.real.basic

namespace train_speed_l598_598619

variable (distance : ℝ) (time : ℝ) (speed : ℝ)

theorem train_speed (h1 : distance = 800) (h2 : time = 10) : speed = distance / time → speed = 80 :=
by 
  intros h_speed_eq
  rw [h1, h2] at h_speed_eq
  exact h_speed_eq

end train_speed_l598_598619


namespace chess_or_basketball_students_l598_598775

-- Definitions based on the conditions
def percentage_likes_basketball : ℝ := 0.4
def percentage_likes_chess : ℝ := 0.1
def total_students : ℕ := 250

-- Main statement to prove
theorem chess_or_basketball_students : 
  (percentage_likes_basketball + percentage_likes_chess) * total_students = 125 :=
by
  sorry

end chess_or_basketball_students_l598_598775


namespace constant_term_expansion_l598_598915

theorem constant_term_expansion :
  let f := (x^3 + 2 * x + 7)
  let g := (2 * x^4 + 3 * x^2 + 10)
  ∀ x : ℝ, constant_term (f * g) = 70 :=
by
  sorry

end constant_term_expansion_l598_598915


namespace tennis_tournament_total_rounds_l598_598167

theorem tennis_tournament_total_rounds
  (participants : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (pairs_formation : ℕ → ℕ)
  (single_points_award : ℕ → ℕ)
  (elimination_condition : ℕ → Prop)
  (tournament_continues : ℕ → Prop)
  (progression_condition : ℕ → ℕ → ℕ)
  (group_split : Π (n : ℕ), Π (k : ℕ), (ℕ × ℕ))
  (rounds_needed : ℕ) :
  participants = 1152 →
  points_win = 1 →
  points_loss = 0 →
  pairs_formation participants ≥ 0 →
  single_points_award participants ≥ 0 →
  (∀ p, p > 1 → participants / p > 0 → tournament_continues participants) →
  (∀ m n, progression_condition m n = n - m) →
  (group_split 1152 1024 = (1024, 128)) →
  rounds_needed = 14 :=
by
  sorry

end tennis_tournament_total_rounds_l598_598167


namespace equal_real_imag_part_l598_598369

noncomputable def complex_number_real_imag_equal (a : ℝ) : Prop :=
  let z := (1 + a * complex.i) / (2 - complex.i)
  in (complex.re z) = (complex.im z)

theorem equal_real_imag_part (a : ℝ) (h : complex_number_real_imag_equal a) : 
  a = 1 / 3 :=
sorry

end equal_real_imag_part_l598_598369


namespace remainder_ab_eq_2_l598_598020

variable (n : ℕ) (a b : ℤ)
variable [hn : Fact (n > 0)]
variable [invertiblea : Invertible (a : ZMod n)]
variable [invertibleb : Invertible (b : ZMod n)]

theorem remainder_ab_eq_2 (h : a ≡ 2 * (b⁻¹ : ZMod n) [MOD n]) : (a * b) % n = 2 := by
  sorry

end remainder_ab_eq_2_l598_598020


namespace Q_solution_l598_598431

def Q (x : ℝ) : ℝ :=
  3 - (1/2) * x - 3 * x^2

theorem Q_solution :
  (∀ x, Q x = Q 0 + Q 1 * x + Q 3 * x^2) →
  Q (-2) = 2 →
  Q = λ x, -3 * x^2 - (1/2) * x + 3 :=
by
  intros h1 h2
  have h0 : Q 0 = 3 := sorry
  have h1' : Q 1 = -1/2 := sorry
  have h3 : Q 3 = -3 := sorry
  apply funext
  intro x
  rw [h0, h1', h3]
  funext x
  simp
  rw h1
  sorry

end Q_solution_l598_598431


namespace triangle_circle_square_value_l598_598219

theorem triangle_circle_square_value (Δ : ℝ) (bigcirc : ℝ) (square : ℝ) 
  (h1 : 2 * Δ + 3 * bigcirc + square = 45)
  (h2 : Δ + 5 * bigcirc + 2 * square = 58)
  (h3 : 3 * Δ + bigcirc + 3 * square = 62) :
  Δ + 2 * bigcirc + square = 35 :=
sorry

end triangle_circle_square_value_l598_598219


namespace time_for_train_to_pass_jogger_l598_598177

noncomputable def time_to_pass (s_jogger s_train : ℝ) (d_headstart l_train : ℝ) : ℝ :=
  let speed_jogger := s_jogger * (1000 / 3600)
  let speed_train := s_train * (1000 / 3600)
  let relative_speed := speed_train - speed_jogger
  let total_distance := d_headstart + l_train
  total_distance / relative_speed

theorem time_for_train_to_pass_jogger :
  time_to_pass 12 60 360 180 = 40.48 :=
by
  sorry

end time_for_train_to_pass_jogger_l598_598177


namespace amount_C_l598_598215

theorem amount_C (A B C : ℕ) 
  (h₁ : A + B + C = 900) 
  (h₂ : A + C = 400) 
  (h₃ : B + C = 750) : 
  C = 250 :=
sorry

end amount_C_l598_598215


namespace sum_even_factors_630_l598_598129

theorem sum_even_factors_630 : 
  ∀ (n : ℕ), (n = 630 → ∃ d : ℕ, n = 2 * d) → 
  (∑ x in (606 : Finset ℕ).divisors.filter (λ d, d % 2 = 0), x) = 1248 :=
by
  intro n h1
  obtain ⟨d, hd⟩ := h1 rfl
  sorry

end sum_even_factors_630_l598_598129


namespace range_of_a_l598_598309

variable (a : ℝ)

def p := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0
def r := -1 ≤ a ∧ a ≤ 1 ∨ a > 3

theorem range_of_a
  (h₀ : p a ∨ q a)
  (h₁ : ¬ (p a ∧ q a)) :
  r a :=
sorry

end range_of_a_l598_598309


namespace sum_even_factors_630_l598_598128

theorem sum_even_factors_630 : 
  ∀ (n : ℕ), (n = 630 → ∃ d : ℕ, n = 2 * d) → 
  (∑ x in (606 : Finset ℕ).divisors.filter (λ d, d % 2 = 0), x) = 1248 :=
by
  intro n h1
  obtain ⟨d, hd⟩ := h1 rfl
  sorry

end sum_even_factors_630_l598_598128


namespace trapezoid_volume_proof_l598_598300

-- Given conditions
variables (a b m : ℝ)
-- Assume b > a
axiom h : b > a

-- Volume of the solid of revolution
def trapezoid_volume := (π * m * (a + 2 * b)) / 3

-- Proof statement
theorem trapezoid_volume_proof : 
  V = (π * m * (a + 2 * b)) / 3 :=
sorry

end trapezoid_volume_proof_l598_598300


namespace average_is_5x_minus_10_implies_x_is_50_l598_598076

theorem average_is_5x_minus_10_implies_x_is_50 (x : ℝ) 
  (h : (1 / 3) * ((3 * x + 8) + (7 * x + 3) + (4 * x + 9)) = 5 * x - 10) : 
  x = 50 :=
by
  sorry

end average_is_5x_minus_10_implies_x_is_50_l598_598076


namespace part_to_third_fraction_is_six_five_l598_598453

noncomputable def ratio_of_part_to_third_fraction (P N : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) : ℝ :=
  P / (N / 3)

theorem part_to_third_fraction_is_six_five (P N : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) : ratio_of_part_to_third_fraction P N h1 h2 = 6 / 5 :=
  sorry

end part_to_third_fraction_is_six_five_l598_598453


namespace min_garden_cost_l598_598065

theorem min_garden_cost : 
  let flower_cost (flower : String) : Real :=
    if flower = "Asters" then 1 else
    if flower = "Begonias" then 2 else
    if flower = "Cannas" then 2 else
    if flower = "Dahlias" then 3 else
    if flower = "Easter lilies" then 2.5 else
    0
  let region_area (region : String) : Nat :=
    if region = "Bottom left" then 10 else
    if region = "Top left" then 9 else
    if region = "Bottom right" then 20 else
    if region = "Top middle" then 2 else
    if region = "Top right" then 7 else
    0
  let min_cost : Real :=
    (flower_cost "Dahlias" * region_area "Top middle") + 
    (flower_cost "Easter lilies" * region_area "Top right") + 
    (flower_cost "Cannas" * region_area "Top left") + 
    (flower_cost "Begonias" * region_area "Bottom left") + 
    (flower_cost "Asters" * region_area "Bottom right")
  min_cost = 81.5 :=
by
  sorry

end min_garden_cost_l598_598065


namespace standard_heat_of_formation_Fe2O3_l598_598113

def Q_form_Al2O3 := 1675.5 -- kJ/mol

def Q1 := 854.2 -- kJ

-- Definition of the standard heat of formation of Fe2O3
def Q_form_Fe2O3 := Q_form_Al2O3 - Q1

-- The proof goal
theorem standard_heat_of_formation_Fe2O3 : Q_form_Fe2O3 = 821.3 := by
  sorry

end standard_heat_of_formation_Fe2O3_l598_598113


namespace min_value_of_reciprocals_l598_598422

theorem min_value_of_reciprocals (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 2) :
  (1 / m + 1 / n) = 2 :=
sorry

end min_value_of_reciprocals_l598_598422


namespace daisy_dog_toys_l598_598645

theorem daisy_dog_toys :
  let monday_toys := 5
  let tuesday_left := 3
  let tuesday_bought := 3
  let wednesday_bought := 5 in
  monday_toys + tuesday_bought + wednesday_bought + (tuesday_left + monday_toys) - (tuesday_left + monday_toys) = 13 := by
  sorry

end daisy_dog_toys_l598_598645


namespace daily_construction_areas_minimum_area_A_must_build_l598_598952

-- Definitions based on conditions and questions
variable {area : ℕ}
variable {daily_A : ℕ}
variable {daily_B : ℕ}
variable (h_area : area = 5100)
variable (h_A_B_diff : daily_A = daily_B + 2)
variable (h_A_days : 900 / daily_A = 720 / daily_B)

-- Proof statements for the questions in the problem
theorem daily_construction_areas (daily_B : ℕ) (daily_A : ℕ) :
  daily_B = 8 ∧ daily_A = 10 :=
by sorry

theorem minimum_area_A_must_build (daily_A : ℕ) (daily_B : ℕ) (area_A : ℕ) :
  (area_A ≥ 2 * (5100 - area_A)) → (area_A ≥ 3400) :=
by sorry

end daily_construction_areas_minimum_area_A_must_build_l598_598952


namespace ribbon_left_l598_598001

-- Define the variables
def T : ℕ := 18 -- Total ribbon in yards
def G : ℕ := 6  -- Number of gifts
def P : ℕ := 2  -- Ribbon per gift in yards

-- Statement of the theorem
theorem ribbon_left (T G P : ℕ) : (T - G * P) = 6 :=
by
  -- Add conditions as Lean assumptions
  have hT : T = 18 := sorry
  have hG : G = 6 := sorry
  have hP : P = 2 := sorry
  -- Now prove the final result
  sorry

end ribbon_left_l598_598001


namespace right_triangle_area_l598_598436

theorem right_triangle_area (a b c : ℝ) (p : ℝ) (S : ℝ)
    (h1 : a ≤ b) (h2 : b ≤ c) 
    (h3 : a^2 + b^2 = c^2) 
    (h4 : 2 * p = a + b + c)
    (h5 : S = (1 / 2) * a * b) :
    p * (p - c) = (p - a) * (p - b) ∧ p * (p - c) = S := 
begin
  sorry
end

end right_triangle_area_l598_598436


namespace range_of_m_l598_598589

variable {f : ℝ → ℝ} (m : ℝ)

-- f is a decreasing function defined on (-2, 2)
def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f y ≤ f x

axiom decreasing_f : is_decreasing_on f (Set.Ioo (-2 : ℝ) (2 : ℝ))

-- f(m-1) > f(2m-1)
axiom f_decreasing_ineq : f (m - 1) > f (2 * m - 1)

-- We need to prove 0 < m < 3/2
theorem range_of_m (m : ℝ) (f : ℝ → ℝ) : 0 < m ∧ m < 3 / 2 :=
by
  sorry

end range_of_m_l598_598589


namespace complement_of_A_l598_598337

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set A
def A : Set ℕ := {2, 4, 5}

-- Define the complement of A with respect to U
def CU : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- State the theorem that the complement of A with respect to U is {1, 3, 6, 7}
theorem complement_of_A : CU = {1, 3, 6, 7} := by
  sorry

end complement_of_A_l598_598337


namespace problem_solution_l598_598419

theorem problem_solution (a b : ℝ) (h1 : a^3 - 15 * a^2 + 25 * a - 75 = 0) (h2 : 8 * b^3 - 60 * b^2 - 310 * b + 2675 = 0) :
  a + b = 15 / 2 :=
sorry

end problem_solution_l598_598419


namespace strawberries_left_l598_598837

/-- Mrs. Smith has 3.5 baskets of strawberries, each containing 50 strawberries. She distributes them equally among 24 girls. Prove that the number of strawberries left is 7. -/
theorem strawberries_left (baskets : ℝ) (strawberries_per_basket : ℕ) (girls : ℕ) (total_strawberries : ℕ) 
  (h1 : baskets = 3.5)
  (h2 : strawberries_per_basket = 50)
  (h3 : girls = 24)
  (h4 : total_strawberries = (baskets * strawberries_per_basket)) :
  total_strawberries % girls = 7 :=
by
  rw [h1, h2, h3] at h4
  norm_cast at h4
  have h5 : total_strawberries = 175 := h4
  rw [h5]
  norm_num

end strawberries_left_l598_598837


namespace ratio_of_areas_of_triangles_l598_598549

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l598_598549


namespace quadratic_real_roots_range_l598_598296

theorem quadratic_real_roots_range (k : ℝ) (h : ∀ x : ℝ, (k - 1) * x^2 - 2 * x + 1 = 0) : k ≤ 2 ∧ k ≠ 1 :=
by
  sorry

end quadratic_real_roots_range_l598_598296


namespace trees_died_l598_598524

theorem trees_died 
  (original_trees : ℕ) 
  (cut_trees : ℕ) 
  (remaining_trees : ℕ) 
  (died_trees : ℕ)
  (h1 : original_trees = 86)
  (h2 : cut_trees = 23)
  (h3 : remaining_trees = 48)
  (h4 : original_trees - died_trees - cut_trees = remaining_trees) : 
  died_trees = 15 :=
by
  sorry

end trees_died_l598_598524


namespace good_numbers_100_2010_ex_good_and_not_good_x_y_l598_598610

-- Definition of a good number
def is_good_number (n : ℤ) : Prop := ∃ a b : ℤ, n = a^2 + 161 * b^2

-- (1) Prove 100 and 2010 are good numbers
theorem good_numbers_100_2010 : is_good_number 100 ∧ is_good_number 2010 :=
by sorry

-- (2) Prove there exist positive integers x and y such that x^161 + y^161 is a good number, 
-- but x + y is not a good number
theorem ex_good_and_not_good_x_y : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ is_good_number (x^161 + y^161) ∧ ¬ is_good_number (x + y) :=
by sorry

end good_numbers_100_2010_ex_good_and_not_good_x_y_l598_598610


namespace no_nine_diagonals_intersect_at_one_point_l598_598772

noncomputable def not_nine_diagonals_through_single_point : Prop :=
  ∀ (P : Point) (polygon : Polygon 25), 
  regular_polygon polygon ∧ all_diagonals_drawn polygon →
  ¬ (∃ (diagonals : list (Diagonal polygon)) (h : diagonals.length = 9), 
     ∀ diagonal ∈ diagonals, passes_through_point diagonal P)

-- Add this to skip the proof
theorem no_nine_diagonals_intersect_at_one_point : not_nine_diagonals_through_single_point :=
by
  sorry

end no_nine_diagonals_intersect_at_one_point_l598_598772


namespace perfect_squares_with_specific_digits_count_perfect_squares_less_than_5000_with_digits_4_5_6_l598_598351

theorem perfect_squares_with_specific_digits (n : ℕ) (h : n < 5000) (digit : ℕ) 
    (h_digit : digit = 4 ∨ digit = 5 ∨ digit = 6) : Σ' (s : ℕ), s < 71 ∧ (s * s) % 10 = digit :=
begin
  sorry
end

theorem count_perfect_squares_less_than_5000_with_digits_4_5_6 : 
    (finset.univ.filter (λ n, n < 5000 ∧ ((n % 10 = 4) ∨ (n % 10 = 5) ∨ (n % 10 = 6)))).card = 35 :=
begin
  sorry
end

end perfect_squares_with_specific_digits_count_perfect_squares_less_than_5000_with_digits_4_5_6_l598_598351


namespace complex_parts_l598_598705

open Complex

theorem complex_parts : ∀ (i : ℂ), i.im ≠ 0 → 
  let z := i * (3 - 4 * i)
  real_part z = 4 ∧ imag_part z = 3 :=
by 
  intros i h
  let z := i * (3 - 4 * i)
  have h_real : real_part z = 4 := sorry
  have h_imag : imag_part z = 3 := sorry
  exact ⟨h_real, h_imag⟩

end complex_parts_l598_598705


namespace sum_of_distances_l598_598146

-- Definitions

def is_regular_pentagon (A B C D E : Point) : Prop := 
-- assume is_regular_pentagon definition, since it is not provided in Mathlib, you can mock it

def perpendicular (P Q R S : Point) : Prop := 
-- assume perpendicular definition, since it is not explicitly provided in Mathlib and you can mock it

def center (O : Point) (A B C D E : Point) : Prop := 
-- assume center definition, since it is not explicitly provided in Mathlib and you can mock it

variable (A B C D E O P Q R: Point)
variable (h1 : is_regular_pentagon A B C D E)
variable (h2 : perpendicular A P C D)
variable (h3 : perpendicular A Q B C)
variable (h4 : perpendicular A R D E)
variable (h5 : center O A B C D E)
variable (h6 : dist O P = 2)

-- Proof Statement

theorem sum_of_distances :
  dist A O + dist A Q + dist A R = 8 :=
sorry

end sum_of_distances_l598_598146


namespace correct_operation_l598_598575

theorem correct_operation : 
  ¬(3 * x^2 + 2 * x^2 = 6 * x^4) ∧ 
  ¬((-2 * x^2)^3 = -6 * x^6) ∧ 
  ¬(x^3 * x^2 = x^6) ∧ 
  (-6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y) :=
by
  sorry

end correct_operation_l598_598575


namespace minimum_raft_weight_l598_598202

-- Define the weights of the animals.
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal.
def num_mice : ℕ := 5
def num_moles : ℕ := 3
def num_hamsters : ℕ := 4

-- The function that represents the minimum weight capacity required for the raft.
def minimum_raft_capacity : ℕ := 140

-- Prove that the minimum raft capacity to transport all animals is 140 grams.
theorem minimum_raft_weight :
  (∀ (total_weight : ℕ), 
    total_weight = (num_mice * weight_mouse) + (num_moles * weight_mole) + (num_hamsters * weight_hamster) →
    (exists (raft_capacity : ℕ), 
      raft_capacity = minimum_raft_capacity ∧
      raft_capacity >= 2 * weight_mouse)) :=
begin
  -- Initial state setup and logical structure.
  intros total_weight total_weight_eq,
  use minimum_raft_capacity,
  split,
  { refl },
  { have h1: 2 * weight_mouse = 140,
    { norm_num },
    rw h1,
    exact le_refl _,
  }
end

end minimum_raft_weight_l598_598202


namespace profit_per_box_correct_optimal_price_reduction_and_max_profit_l598_598597

-- Define the conditions
def total_profit (nA nB : ℕ) (pA pB : ℕ) : ℕ := nA * pA + nB * pB

def profit_per_box_B : ℕ := 10
def profit_per_box_A : ℕ := profit_per_box_B + 5

def price_reduction_max_profit (a : ℕ) : ℕ := (15 - a) * (100 + 20 * a)

-- Part 1: Proving the profit per box for type A and B
theorem profit_per_box_correct :
  total_profit 60 40 profit_per_box_A profit_per_box_B = 1300 → profit_per_box_A = 15 ∧ profit_per_box_B = 10 :=
by
  intro h
  dsimp [total_profit, profit_per_box_A, profit_per_box_B] at h
  -- Proof-skipping placeholder
  sorry

-- Part 2: Optimal price reduction and maximum profit
theorem optimal_price_reduction_and_max_profit :
  (∀ a : ℕ, price_reduction_max_profit a ≤ price_reduction_max_profit 5) ∧ price_reduction_max_profit 5 = 2000 :=
by
  intro h
  dsimp [price_reduction_max_profit] at h
  -- Proof-skipping placeholder
  sorry

end profit_per_box_correct_optimal_price_reduction_and_max_profit_l598_598597


namespace raft_minimum_capacity_l598_598195

theorem raft_minimum_capacity 
  (mice : ℕ) (mice_weight : ℕ) 
  (moles : ℕ) (mole_weight : ℕ) 
  (hamsters : ℕ) (hamster_weight : ℕ) 
  (raft_cannot_move_without_rower : Bool)
  (rower_condition : ∀ W, W ≥ 2 * mice_weight) :
  mice = 5 → mice_weight = 70 →
  moles = 3 → mole_weight = 90 →
  hamsters = 4 → hamster_weight = 120 →
  ∃ W, (W = 140) :=
by
  intros mice_eq mice_w_eq moles_eq mole_w_eq hamsters_eq hamster_w_eq
  use 140
  sorry

end raft_minimum_capacity_l598_598195


namespace abs_neg_three_halves_l598_598476

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end abs_neg_three_halves_l598_598476


namespace number_of_girls_more_than_boys_l598_598176

theorem number_of_girls_more_than_boys
    (total_students : ℕ)
    (number_of_boys : ℕ)
    (h1 : total_students = 485)
    (h2 : number_of_boys = 208) :
    total_students - number_of_boys - number_of_boys = 69 :=
by
    sorry

end number_of_girls_more_than_boys_l598_598176


namespace ellipse_AB_values_correct_l598_598780

noncomputable def ellipse_possible_AB_values (Γ : Type) (F A B : Γ) (d1 d2 : ℝ) : Set ℝ :=
  {AB | |FA| = 3 ∧ |FB| = 2 ∧ AB = |AB|}

theorem ellipse_AB_values_correct :
  ellipse_possible_AB_values Γ F A B =
  {5, real.sqrt 7, real.sqrt 17} := 
sorry

end ellipse_AB_values_correct_l598_598780


namespace part1_part2_l598_598285

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp (a * x)) / x

theorem part1 (a : ℝ) : (∀ x ∈ Ioc 0 4, (f a)' x ≤ 0) ↔ a ≤ 1/4 := 
by sorry

theorem part2 (m : ℝ) (hm : 0 < m) : 
  let fm := λ x, (Real.exp x) / x in
  if (0 < m) ∧ (m < 1) then
    Inf (fm '' (set.Icc m (m+2))) = Real.exp 1
  else
    Inf (fm '' (set.Icc m (m+2))) = (Real.exp m) / m := 
by sorry

end part1_part2_l598_598285


namespace endpoint_sum_l598_598511

theorem endpoint_sum
  (x y : ℤ)
  (H_midpoint_x : (x + 15) / 2 = 10)
  (H_midpoint_y : (y - 8) / 2 = -3) :
  x + y = 7 :=
sorry

end endpoint_sum_l598_598511


namespace smallest_even_of_sum_380_l598_598641

theorem smallest_even_of_sum_380 :
  ∃ (a b c d e : ℤ), 
    (a < b ∧ b < c ∧ c < d ∧ d < e) ∧
    (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ d % 2 = 0 ∧ e % 2 = 0) ∧
    (a + b + c + d + e = 380) ∧
    (a + 0 = 72) :=
begin
  sorry
end

end smallest_even_of_sum_380_l598_598641


namespace find_full_pound_price_l598_598187

-- Define the conditions provided in the problem.
def discounted_price_half_pound := 3
def discount_rate := 0.5
def full_pound_price (x : ℝ) := x
def discounted_full_pound_price (x : ℝ) := (discount_rate * full_pound_price x)
def discounted_half_pound_price (x : ℝ) := (discounted_full_pound_price x) / 2

-- State the problem to find the original price of a full pound given the conditions
theorem find_full_pound_price (x : ℝ) (h : discounted_half_pound_price x = discounted_price_half_pound) :
  full_pound_price x = 12 :=
  sorry

end find_full_pound_price_l598_598187


namespace expression_f_domain_f_range_f_l598_598327

noncomputable def g (x : ℝ) : ℝ := sqrt x + 1
noncomputable def h (x : ℝ) : ℝ := 1 / (x + 3)
noncomputable def f (x : ℝ) : ℝ := g x * h x

theorem expression_f (x : ℝ) : f x = (sqrt x + 1) / (x + 3) :=
by {
  unfold f g h,
  field_simp [sqrt x + 1],
}

theorem domain_f (a : ℝ) (ha : 0 < a) : 
  ∀ (x : ℝ), x ∈ Icc 0 a → 
  (f x = (sqrt x + 1) / (x + 3) ∧ 
  0 ≤ x ∧ x ≤ a) :=
by {
  intros x hx,
  simp only [f, g, h],
  split,
  { field_simp [hx], },
  { exact hx, }
}

theorem range_f (x : ℝ) : 
(0 : ℝ) ≤ x ∧ x ≤ (1 / 4) → 
(f x ∈ (set.Icc (1 / 3) (6 / 13))) :=
by {
  intro hx,
  let t := sqrt x + 1,
  have ht : 1 ≤ t ∧ t ≤ 3 / 2,
  { split; 
    { // prove with basic inequalities considering sqrt properties
      sorry, }  
  },
  { // Prove f(x) bounds based on new limits
    sorry }
}

end expression_f_domain_f_range_f_l598_598327


namespace problem1_problem2_l598_598160

-- Problem 1: Prove the simplification of an expression
theorem problem1 (x : ℝ) : (2*x + 1)^2 + x*(x-4) = 5*x^2 + 1 := 
by sorry

-- Problem 2: Prove the solution set for the system of inequalities
theorem problem2 (x : ℝ) (h1 : 3*x - 6 > 0) (h2 : (5 - x) / 2 < 1) : x > 3 := 
by sorry

end problem1_problem2_l598_598160


namespace projection_correct_l598_598294

def vect1 : ℝ × ℝ × ℝ := (2, -4, 1)
def vect2 : ℝ × ℝ × ℝ := (1, -2, 0.5)
def input_vect : ℝ × ℝ × ℝ := (-6, 2, -3)
def output_vect : ℝ × ℝ × ℝ := (-2.2, 4.4, -1.1)

theorem projection_correct :
  let proj := (λ u v : ℝ × ℝ × ℝ, let dot_uv := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 in
                                let dot_uu := u.1 * u.1 + u.2 * u.2 + u.3 * u.3 in
                                (dot_uv / dot_uu) • (u.1, u.2, u.3)) in
  proj vect2 input_vect = output_vect :=
sorry

end projection_correct_l598_598294


namespace diagonals_bisect_and_perpendicular_implies_rhombus_l598_598362

-- Definitions based on the conditions
variable (Q : Type) [quadrilateral Q]
variable (bisect : bisect_diagonals Q) (perpendicular : perpendicular_diagonals Q)

-- Statement to prove
theorem diagonals_bisect_and_perpendicular_implies_rhombus :
  bisect_diagonals Q → perpendicular_diagonals Q → rhombus Q :=
by
  intros
  sorry

end diagonals_bisect_and_perpendicular_implies_rhombus_l598_598362


namespace no_integer_product_1980_no_integer_product_1990_exists_integer_product_2000_l598_598398

/-- 
Prove that no integer whose digits' product equals 1980 exists.
-/
theorem no_integer_product_1980 : ¬ ∃ n : ℤ, (∀ d ∈ digits n, 1 ≤ d ∧ d ≤ 9) ∧ product_of_digits n = 1980 := 
sorry

/-- 
Prove that no integer whose digits' product equals 1990 exists.
-/
theorem no_integer_product_1990 : ¬ ∃ n : ℤ, (∀ d ∈ digits n, 1 ≤ d ∧ d ≤ 9) ∧ product_of_digits n = 1990 :=
sorry

/--
Prove that there exists an integer whose digits' product equals 2000.
-/
theorem exists_integer_product_2000 : ∃ n : ℤ, (∀ d ∈ digits n, 1 ≤ d ∧ d ≤ 9) ∧ product_of_digits n = 2000 :=
sorry

/--
Helper function to extract digits of an integer.
-/
def digits (n : ℤ) : List ℤ := sorry

/--
Helper function to calculate the product of digits of an integer.
-/
def product_of_digits (n : ℤ) : ℤ := (digits n).prod

end no_integer_product_1980_no_integer_product_1990_exists_integer_product_2000_l598_598398


namespace solution_set_l598_598271

theorem solution_set (x : ℝ) (h : x > 0) (h_log : log (1 / 2 * x) ≠ 0) :
  (abs((1 / (log (1 / 2 * x))) + 2) > 3 / 2) ↔ x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 (2 ^ (5 / 7)) ∪ Set.Ioi 4) := 
sorry

end solution_set_l598_598271


namespace tickets_needed_to_ride_l598_598145

noncomputable def tickets_required : Float :=
let ferris_wheel := 3.5
let roller_coaster := 8.0
let bumper_cars := 5.0
let additional_ride_discount := 0.5
let newspaper_coupon := 1.5
let teacher_discount := 2.0

let total_cost_without_discounts := ferris_wheel + roller_coaster + bumper_cars
let total_additional_discounts := additional_ride_discount * 2
let total_coupons_discounts := newspaper_coupon + teacher_discount

let total_cost_with_discounts := total_cost_without_discounts - total_additional_discounts - total_coupons_discounts
total_cost_with_discounts

theorem tickets_needed_to_ride : tickets_required = 12.0 := by
  sorry

end tickets_needed_to_ride_l598_598145


namespace lives_per_remaining_player_l598_598523

theorem lives_per_remaining_player (initial_players : ℕ) 
                                   (players_quit : ℕ)
                                   (total_lives : ℕ) 
                                   (remaining_players : ℕ)
                                   (lives_each_remaining : ℕ)
    (h1 : initial_players = 16)
    (h2 : players_quit = 7)
    (h3 : remaining_players = initial_players - players_quit)
    (h4 : total_lives = 72)
    (h5 : lives_each_remaining = total_lives / remaining_players) :
    lives_each_remaining = 8 :=
begin
    sorry
end

end lives_per_remaining_player_l598_598523


namespace coeff_x2_of_poly_mul_l598_598560

theorem coeff_x2_of_poly_mul :
  let p1 := 2 * X ^ 3 - 4 * X ^ 2 + 3 * X + 2
  let p2 := - X ^ 2 + 3 * X - 5
  coeff (p1 * p2) 2 = 7 := by
  sorry

end coeff_x2_of_poly_mul_l598_598560


namespace complex_fraction_identity_l598_598706

def question : ℂ := 2
def condition_denom : ℂ := 1 - complex.I
def correct_answer : ℂ := 1 + complex.I

theorem complex_fraction_identity (i : ℂ) (hi : i = complex.I) : 
  question / condition_denom = correct_answer := 
by {
  sorry
}

end complex_fraction_identity_l598_598706


namespace sun_salutations_per_year_l598_598073

theorem sun_salutations_per_year :
  let poses_per_day := 5
  let days_per_week := 5
  let weeks_per_year := 52
  poses_per_day * days_per_week * weeks_per_year = 1300 :=
by
  sorry

end sun_salutations_per_year_l598_598073


namespace total_marbles_l598_598445

-- Define the number of marbles Mary has
def marblesMary : Nat := 9 

-- Define the number of marbles Joan has
def marblesJoan : Nat := 3 

-- Theorem to prove the total number of marbles
theorem total_marbles : marblesMary + marblesJoan = 12 := 
by sorry

end total_marbles_l598_598445


namespace library_books_l598_598101

def initial_shelves : ℝ := 25793.5
def books_per_shelf : ℝ := 13.2
def rounded_shelves : ℕ := 25794
def expected_books : ℕ := 340481

theorem library_books : 
    let total_books := rounded_shelves * (books_per_shelf : ℕ) in 
    total_books = expected_books :=
by
    sorry

end library_books_l598_598101


namespace largest_integer_less_than_100_with_remainder_7_divided_9_l598_598268

theorem largest_integer_less_than_100_with_remainder_7_divided_9 :
  ∃ x : ℕ, (∀ m : ℤ, x = 9 * m + 7 → 9 * m + 7 < 100) ∧ x = 97 :=
sorry

end largest_integer_less_than_100_with_remainder_7_divided_9_l598_598268


namespace escalator_length_l598_598627

theorem escalator_length
  (escalator_speed : ℕ)
  (person_speed : ℕ)
  (time_taken : ℕ)
  (combined_speed : ℕ)
  (condition1 : escalator_speed = 12)
  (condition2 : person_speed = 2)
  (condition3 : time_taken = 14)
  (condition4 : combined_speed = escalator_speed + person_speed)
  (condition5 : combined_speed * time_taken = 196) :
  combined_speed * time_taken = 196 := 
by
  -- the proof would go here
  sorry

end escalator_length_l598_598627


namespace triangles_area_sum_correct_l598_598640

noncomputable def cube_edge_length : ℝ := 2

def face_diagonal (e : ℝ) : ℝ := (e^2 + e^2).sqrt
def space_diagonal (e : ℝ) : ℝ := (e^2 + e^2 + e^2).sqrt

def face_triangle_area (e : ℝ) : ℝ := (1 / 2) * e * e
def perpendicular_triangle_area (e : ℝ) : ℝ := (1 / 2) * e * face_diagonal e
def oblique_triangle_area (d : ℝ) : ℝ := (d^2 * (3).sqrt) / 4

def total_area_of_face_triangles : ℝ := 24 * face_triangle_area cube_edge_length
def total_area_of_perpendicular_triangles : ℝ := 24 * perpendicular_triangle_area cube_edge_length
def total_area_of_oblique_triangles : ℝ := 8 * oblique_triangle_area (face_diagonal cube_edge_length)

def total_triangle_area : ℝ :=
  total_area_of_face_triangles +
  total_area_of_perpendicular_triangles +
  total_area_of_oblique_triangles

theorem triangles_area_sum_correct :
  ∃ (a : ℝ) (b c : ℝ), 
  total_triangle_area = a + b.sqrt + c.sqrt ∧ a + b + c = 224 :=
by {
  existsi 48 : ℝ,
  existsi 48*2 : ℝ, -- 128 = 48*2
  existsi 48 : ℝ,
  split,
  { 
    sorry    -- actual step proving the total area (computation)
  },
  { 
    sorry    -- proving the sum a + b + c is 224
  }
}

end triangles_area_sum_correct_l598_598640


namespace smallest_positive_period_maximum_triangle_area_l598_598828

-- Conditions
def f (x : ℝ) : ℝ := 1 - 2 * sin x ^ 2 - cos (2 * x + π / 3)
def length_b := (5 : ℝ)
def angle_B_condition (B : ℝ) : Prop := f (B / 2) = 1

-- Questions stated as hypotheses
theorem smallest_positive_period : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π := sorry

theorem maximum_triangle_area (a b c : ℝ) (A B C : ℝ)
  (h_b : b = length_b)
  (h_B_cond : angle_B_condition B)
  (h_angles : A + B + C = π)  -- Sum of angles in a triangle
  : ∃ S : ℝ, S = (25 * sqrt 3) / 4 := sorry

end smallest_positive_period_maximum_triangle_area_l598_598828


namespace volunteers_distribution_no_one_in_place_A_all_places_have_at_least_one_person_ambulances_distribution_l598_598385

-- Problem statements in Lean 4

-- 1. No one goes to Place A and two volunteers go to each of Place B and Place C.
theorem volunteers_distribution_no_one_in_place_A :
  ∃ (A B C D : Type), (total_permutations (place B 4, place C 4) = 6) := sorry

-- 2. Each place has at least one person
theorem all_places_have_at_least_one_person :
  ∃ (A B C D : Type) (places : list (subtype set)),
    (total_permutations (place A 4, place B 4, place C 4) = 36) := sorry

-- 3. Allocate 20 ambulances to three places with at least one ambulance in each
theorem ambulances_distribution :
  ∃ (places : Type) (ambulances : ℕ), 
    (total_permutations (place A, place B, place C) = 171) := sorry

end volunteers_distribution_no_one_in_place_A_all_places_have_at_least_one_person_ambulances_distribution_l598_598385


namespace magnitude_sum_of_unit_vectors_l598_598417

noncomputable def unit_vector (v : ℝ × ℝ × ℝ) : Prop := v.1^2 + v.2^2 + v.3^2 = 1

theorem magnitude_sum_of_unit_vectors (a b : ℝ × ℝ × ℝ) (h_a : unit_vector a) (h_b : unit_vector b) 
  (angle_ab : real.angle a b = real.angle 60)
  : real.sqrt (real.dot_product (a + b) (a + b)) = real.sqrt 3 :=
sorry

end magnitude_sum_of_unit_vectors_l598_598417


namespace sqrt_a_minus_b_l598_598710

theorem sqrt_a_minus_b (a b : ℝ) (h1 : (5 * a + 2) ^ (1 / 3) = 3) (h2 : b ^ 2 = 16) :
  sqrt (a - b) = 1 ∨ sqrt (a - b) = 3 := by
  sorry

end sqrt_a_minus_b_l598_598710


namespace quadratic_roots_square_l598_598280

theorem quadratic_roots_square (q : ℝ) :
  (∃ a : ℝ, a + a^2 = 12 ∧ q = a * a^2) → (q = 27 ∨ q = -64) :=
by
  sorry

end quadratic_roots_square_l598_598280


namespace exists_positive_integers_l598_598850

theorem exists_positive_integers (k n : ℕ) (hk : 0 < k) (hn : 0 < n) :
  ∃ (m : fin k → ℕ), (∀ i, 0 < m i) ∧ (1 + (2^k - 1) / n : ℚ) = ∏ i, (1 + 1 / (m i) : ℚ) :=
by sorry

end exists_positive_integers_l598_598850


namespace probability_all_balls_same_color_probability_4_white_balls_l598_598317

-- Define initial conditions
def initial_white_balls : ℕ := 6
def initial_yellow_balls : ℕ := 4
def total_initial_balls : ℕ := initial_white_balls + initial_yellow_balls

-- Define the probability calculation for drawing balls as described
noncomputable def draw_probability_same_color_after_4_draws : ℚ :=
  (6 / 10) * (7 / 10) * (8 / 10) * (9 / 10)

noncomputable def draw_probability_4_white_balls_after_4_draws : ℚ :=
  (6 / 10) * (3 / 10) * (4 / 10) * (5 / 10) + 
  3 * ((4 / 10) * (5 / 10) * (4 / 10) * (5 / 10))

-- The theorem we want to prove about the probabilities
theorem probability_all_balls_same_color :
  draw_probability_same_color_after_4_draws = 189 / 625 := by
  sorry

theorem probability_4_white_balls :
  draw_probability_4_white_balls_after_4_draws = 19 / 125 := by
  sorry

end probability_all_balls_same_color_probability_4_white_balls_l598_598317


namespace problem_solution_l598_598142

def is_functional_relationship {X Y : Type} (f : X → Y) : Prop := sorry

def are_correlated (X Y : Type) : Prop := sorry

def taxi_fare_distance : Type := sorry
def house_size_price : Type := sorry
def human_height_weight : Type := sorry
def iron_block_mass : Type := sorry

axiom taxi_fare_is_functional : is_functional_relationship taxi_fare_distance
axiom house_size_is_functional : is_functional_relationship house_size_price
axiom iron_block_is_functional : is_functional_relationship iron_block_mass
axiom human_height_weight_correlated : are_correlated human_height_weight

-- Prove that the correct answer is C: Human height and weight are correlated given the conditions
theorem problem_solution :
  (¬ are_correlated taxi_fare_distance) ∧
  (¬ are_correlated house_size_price) ∧
  (are_correlated human_height_weight) ∧
  (¬ are_correlated iron_block_mass) :=
begin
  sorry
end

end problem_solution_l598_598142


namespace common_tangents_between_circles_l598_598237

def center (a b c: ℝ) := (a, b)
def radius (c: ℝ) := sqrt c

def C₁_center := center (-1) (-4)
def C₁_radius := radius 9
def C₂_center := center 2 2
def C₂_radius := radius 9

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def common_tangent_lines (radius1 radius2 distance : ℝ) : ℕ := 
  if (distance > radius1 + radius2) ∨ (distance < abs (radius1 - radius2)) then 0
  else if (distance = radius1 + radius2) ∨ (distance = abs (radius1 - radius2)) then 1
  else if (radius1 - radius2 < distance) ∧ (distance < radius1 + radius2) then 2
  else 0

theorem common_tangents_between_circles : 
  common_tangent_lines C₁_radius C₂_radius (distance C₁_center C₂_center) = 2 := 
sorry

end common_tangents_between_circles_l598_598237


namespace oliver_earning_correct_l598_598031

open Real

noncomputable def total_weight_two_days_ago : ℝ := 5

noncomputable def total_weight_yesterday : ℝ := total_weight_two_days_ago + 5

noncomputable def total_weight_today : ℝ := 2 * total_weight_yesterday

noncomputable def total_weight_three_days : ℝ := total_weight_two_days_ago + total_weight_yesterday + total_weight_today

noncomputable def earning_per_kilo : ℝ := 2

noncomputable def total_earning : ℝ := total_weight_three_days * earning_per_kilo

theorem oliver_earning_correct : total_earning = 70 := by
  sorry

end oliver_earning_correct_l598_598031


namespace number_of_keepers_l598_598584

theorem number_of_keepers (k : ℕ)
  (hens : ℕ := 50)
  (goats : ℕ := 45)
  (camels : ℕ := 8)
  (hen_feet : ℕ := 2)
  (goat_feet : ℕ := 4)
  (camel_feet : ℕ := 4)
  (keeper_feet : ℕ := 2)
  (feet_more_than_heads : ℕ := 224)
  (total_heads : ℕ := hens + goats + camels + k)
  (total_feet : ℕ := (hens * hen_feet) + (goats * goat_feet) + (camels * camel_feet) + (k * keeper_feet)):
  total_feet = total_heads + feet_more_than_heads → k = 15 :=
by
  sorry

end number_of_keepers_l598_598584


namespace min_cost_for_boxes_l598_598585

def box_volume (l w h : ℕ) : ℕ := l * w * h
def total_boxes_needed (total_volume box_volume : ℕ) : ℕ := (total_volume + box_volume - 1) / box_volume
def total_cost (num_boxes : ℕ) (cost_per_box : ℚ) : ℚ := num_boxes * cost_per_box

theorem min_cost_for_boxes : 
  let l := 20
  let w := 20
  let h := 15
  let cost_per_box := (7 : ℚ) / 10
  let total_volume := 3060000
  let volume_box := box_volume l w h
  let num_boxes_needed := total_boxes_needed total_volume volume_box
  (num_boxes_needed = 510) → 
  (total_cost num_boxes_needed cost_per_box = 357) :=
by
  intros
  sorry

end min_cost_for_boxes_l598_598585


namespace oliver_earnings_l598_598034

-- Define the conditions
def cost_per_kilo : ℝ := 2
def kilos_two_days_ago : ℝ := 5
def kilos_yesterday : ℝ := kilos_two_days_ago + 5
def kilos_today : ℝ := 2 * kilos_yesterday

-- Calculate the total kilos washed over the three days
def total_kilos : ℝ := kilos_two_days_ago + kilos_yesterday + kilos_today

-- Calculate the earnings over the three days
def earnings : ℝ := total_kilos * cost_per_kilo

-- The theorem we want to prove
theorem oliver_earnings : earnings = 70 := by
  sorry

end oliver_earnings_l598_598034


namespace minimum_value_at_x_eq_3_l598_598923

theorem minimum_value_at_x_eq_3 (b : ℝ) : 
  ∃ m : ℝ, (∀ x : ℝ, 3 * x^2 - 18 * x + b ≥ m) ∧ (3 * 3^2 - 18 * 3 + b = m) :=
by
  sorry

end minimum_value_at_x_eq_3_l598_598923


namespace area_CDE_l598_598046

variable (ABC : Type) [Triangle ABC] (A B C D E F : Point ABC)
variable (on_AC : On D AC) (on_BC : On E BC)
variable (intersect_AE_BD : Intersect AE BD F)
variable (area_ABF : Area (Triangle A B F) = 1)
variable (area_ADF : Area (Triangle A D F) = 1/3)
variable (area_BEF : Area (Triangle B E F) = 1/4)

theorem area_CDE
  (ABC : Triangle)
  (A B C D E F : Point ABC)
  (on_AC : On D AC)
  (on_BC : On E BC)
  (intersect_AE_BD : Intersect AE BD F)
  (area_ABF : Area (Triangle A B F) = 1)
  (area_ADF : Area (Triangle A D F) = 1/3)
  (area_BEF : Area (Triangle B E F) = 1/4)
  : Area (Triangle C D E) = 1/4 :=
sorry

end area_CDE_l598_598046


namespace largest_valid_digit_is_9_digit_9_is_valid_l598_598591

-- Define the problem parameters
def digit_in_hundred_million_place := 4
def target_approximation := 5.5e9

-- Define a predicate to check if a number n in the ten million's place can make 54n9607502 approximately 5.5 billion when rounded up
def valid_digit (n : ℕ) : Prop :=
  54 * 10^8 + n * 10^7 + 9607502 >= (target_approximation - 0.5e9)

-- Define the theorem to prove that the largest valid digit is 9
theorem largest_valid_digit_is_9 : ∀ n, valid_digit n → n <= 9 :=
begin
  intro n,
  assume h,
  sorry
end

-- Another theorem to assert that 9 is indeed a valid digit
theorem digit_9_is_valid : valid_digit 9 :=
by
  sorry

end largest_valid_digit_is_9_digit_9_is_valid_l598_598591


namespace downstream_distance_is_36_l598_598961

variables (V_m V_s D : ℝ)

-- Conditions
def man_speed_still_water : Prop := V_m = 15.5
def upstream_distance : Prop := V_up := V_m - V_s → V_up * 2 = 26
def downstream_distance : Prop := V_down := V_m + V_s → V_down * 2 = D

-- Problem statement: Prove that the distance downstream is 36 km.
theorem downstream_distance_is_36 (h1 : man_speed_still_water)
                                  (h2 : upstream_distance)
                                  (h3 : downstream_distance) :
  D = 36 := sorry

end downstream_distance_is_36_l598_598961


namespace find_a_plus_k_l598_598989

noncomputable def ellipse_center (f1 f2 : ℝ × ℝ) : ℝ × ℝ :=
  ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)

noncomputable def ellipse_major_axis (p f1 f2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) + real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)

noncomputable def ellipse_minor_axis (d_major d_foci : ℝ) : ℝ :=
  real.sqrt (d_major^2 - d_foci^2)

theorem find_a_plus_k (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) (h k a b : ℝ)
  (h_foci1 : f1 = (1, 1)) (h_foci2 : f2 = (1, 4)) (h_point : p = (5, 2))
  (h_form : ∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)
  (h_positive_a : 0 < a) (h_positive_b : 0 < b) :
  a + k = (real.sqrt 17 + 2 * real.sqrt 5 + 5) / 2 := sorry

end find_a_plus_k_l598_598989


namespace convex_ineq_sym_jensen_ineq_sym_jensen_ineq_equal_weights_l598_598232

variable {k : Nat}
variable (x : Fin k → ℝ) (f : ℝ → ℝ)
variable (p q : Fin k → ℝ)
variable (hpos_p : ∀ (i : Fin k), 0 < p i) (hpos_q : ∀ (i : Fin k), 0 < q i)
variable (hsum_q : ∑ i, q i = 1)

noncomputable def centroid (x : Fin k → ℝ) (p : Fin k → ℝ) : ℝ :=
  (∑ i, p i * x i) / (∑ i, p i)

theorem convex_ineq :
  f (centroid x p) ≤ (∑ i, (p i * f (x i))) / (∑ i, p i) :=
sorry

theorem sym_jensen_ineq :
  f (∑ i, (q i * x i)) ≤ ∑ i, (q i * f (x i)) :=
sorry

theorem sym_jensen_ineq_equal_weights :
  let q' := λ i, 1 / (↑k : ℝ) in
  f (∑ i, (q' i * x i)) ≤ ∑ i, (q' i * f (x i)) :=
sorry

end convex_ineq_sym_jensen_ineq_sym_jensen_ineq_equal_weights_l598_598232


namespace sum_reciprocals_nonempty_subsets_l598_598984

theorem sum_reciprocals_nonempty_subsets (n : ℕ) :
  ∑ S in (Finset.powerset (Finset.range (n+1)).erase ∅), 
    ∏ x in S, (1 : ℚ) / x = n := 
sorry

end sum_reciprocals_nonempty_subsets_l598_598984


namespace determine_ABC_l598_598831

theorem determine_ABC : 
  ∀ (A B C : ℝ), 
    A = 2 * B - 3 * C ∧ 
    B = 2 * C - 5 ∧ 
    A + B + C = 100 → 
    A = 18.75 ∧ B = 52.5 ∧ C = 28.75 :=
by
  intro A B C h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end determine_ABC_l598_598831


namespace unique_wxyz_solution_l598_598363

theorem unique_wxyz_solution (w x y z : ℕ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : w.factorial = x.factorial + y.factorial + z.factorial) : (w, x, y, z) = (3, 2, 2, 2) :=
by
  sorry

end unique_wxyz_solution_l598_598363


namespace best_years_to_scrap_l598_598455

-- Define the conditions from the problem
def purchase_cost : ℕ := 150000
def annual_cost : ℕ := 15000
def maintenance_initial : ℕ := 3000
def maintenance_difference : ℕ := 3000

-- Define the total_cost function
def total_cost (n : ℕ) : ℕ :=
  purchase_cost + annual_cost * n + (n * (2 * maintenance_initial + (n - 1) * maintenance_difference)) / 2

-- Define the average annual cost function
def average_annual_cost (n : ℕ) : ℕ :=
  total_cost n / n

-- Statement to be proven: the best number of years to minimize average annual cost is 10
theorem best_years_to_scrap : 
  (∀ n : ℕ, average_annual_cost 10 ≤ average_annual_cost n) :=
by
  sorry
  
end best_years_to_scrap_l598_598455


namespace number_of_lockers_is_1676_l598_598507

def cost_per_digit := 0.03
def total_cost := 167.94

def total_digits (n : Nat) : Nat :=
  let digits (x : Nat) := (x.toString.length)
  (List.range (n + 1)).map digits |>.sum

def cost_to_label (n : Nat) : Float :=
  total_digits n * cost_per_digit

theorem number_of_lockers_is_1676 :
  ∃ n : Nat, cost_to_label n = total_cost ∧ n = 1676 :=
by
  sorry

end number_of_lockers_is_1676_l598_598507


namespace raft_minimum_capacity_l598_598203

theorem raft_minimum_capacity (n_mice n_moles n_hamsters : ℕ)
  (weight_mice weight_moles weight_hamsters : ℕ)
  (total_weight : ℕ) :
  n_mice = 5 →
  weight_mice = 70 →
  n_moles = 3 →
  weight_moles = 90 →
  n_hamsters = 4 →
  weight_hamsters = 120 →
  (∀ (total_weight : ℕ), total_weight = n_mice * weight_mice + n_moles * weight_moles + n_hamsters * weight_hamsters) →
  (∃ (min_capacity: ℕ), min_capacity ≥ 140) :=
by
  intros
  sorry

end raft_minimum_capacity_l598_598203


namespace line_passes_through_fixed_point_l598_598287

theorem line_passes_through_fixed_point (m n : ℝ) (h : m + n - 1 = 0) : 
  ∃ (x y : ℝ), (x = 1) ∧ (y = -1) ∧ (m * x + y + n = 0) :=
by {
  use 1, -1,
  split,
  { exact rfl },
  split,
  { exact rfl },
  { calc m * 1 + -1 + n
        = m - 1 + n : by ring
    ... = (m + n) - 1 : by ring
    ... = 0 : by rw h },
  sorry
}

end line_passes_through_fixed_point_l598_598287


namespace count_g_iterations_to_one_l598_598649

def g (n : ℕ) : ℕ :=
if nat.prime n then n ^ 2 + 3 else n / 2

theorem count_g_iterations_to_one : 
  (finset.filter (λ n, ∃ m, g^[m] n = 1) (finset.range 101)).card = 7 := 
sorry

end count_g_iterations_to_one_l598_598649


namespace probability_is_23_over_48_l598_598612

noncomputable def probability_purple_point_conditioned (x y : ℝ) : ℝ :=
  if (0 ≤ x ∧ x ≤ 2) ∧ (0 ≤ y ∧ y ≤ 3) then
    if (x < y ∧ y < 2 * x) then 1 else 0
  else 0

theorem probability_is_23_over_48 :
  ∫ y in 0..3, ∫ x in 0..2, probability_purple_point_conditioned x y
  = (3 - 0) * (2 - 0) * 23 / 48 :=
by
  sorry

end probability_is_23_over_48_l598_598612


namespace colored_balls_trade_l598_598655

theorem colored_balls_trade
    (n : ℕ := 100)
    (girls : Fin n → Fin n → ℕ)
    (total_balls : Fin n → ℕ := λ _ => n * n)
    (colors : Fin n → ℕ := λ _ => n)
    (initial_condition : ∀ i, ∑ j, girls i j = n)
    (distribution_condition : ∀ j, ∑ i, girls i j = n) :
    ∃ (moves : List (Fin n × Fin n × Fin n × Fin n)), 
      (∀ (a b c d : Fin n), (a, b, c, d) ∈ moves → a ≠ c ∧ b ≠ d) ∧
      (∀ i, exists! colors_girl: Fin n → Fin n, 
        (∀ c, colors_girl c ≠ colors_girl c → ∃ j, girls i (colors_girl j) = 1) ∧ 
        (num_exchanges: ∀ a b c d, (a, b, c, d) ∈ moves → some (girls a b) = none → some (girls c d) = none)
        sorry
          
sorry

end colored_balls_trade_l598_598655


namespace raft_min_capacity_l598_598192

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end raft_min_capacity_l598_598192


namespace sum_S30_l598_598709

variable {a : ℕ → ℝ} -- Define the arithmetic sequence

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def S (n : ℕ) : ℝ := (n * (a 0 + a (n-1))) / 2 

axiom S10 : S 10 = 12
axiom S20 : S 20 = 17

theorem sum_S30 : S 30 = 15 :=
by
  sorry

end sum_S30_l598_598709


namespace a_2013_correct_l598_598730

noncomputable def a : ℕ → ℕ
| 1 := 0
| (n + 1) := a n + n

theorem a_2013_correct : a 2013 = 2025078 :=
  sorry

end a_2013_correct_l598_598730


namespace area_of_quadrilateral_EFGH_l598_598853

-- Definitions of the sides and angles in the quadrilateral
variables (EF FG EH HG : ℝ)
variables (EG : ℝ) (EG_pos : EG = 5)
variables (right_angle_F : EF^2 + FG^2 = EG^2)
variables (right_angle_H : EH^2 + HG^2 = EG^2)
variables (different_lengths : ∃ a b : ℝ, a ≠ b ∧ ((EF = a ∧ FG = b) ∨ (EH = b ∧ HG = a)))

-- The theorem that we aim to prove
theorem area_of_quadrilateral_EFGH : EF^2 + FG^2 = EH^2 + HG^2 → EF ∈ {3, 4} → FG ∈ {3, 4} → 
                                      EH ∈ {3, 4} → HG ∈ {3, 4} → 
                                      EF^2 + FG^2 = 25 ∧ EH^2 + HG^2 = 25 → 
                                      (1 / 2) * EF * FG + (1 / 2) * EH * HG = 12 :=
by sorry

end area_of_quadrilateral_EFGH_l598_598853


namespace female_salmon_returned_l598_598253

theorem female_salmon_returned :
  let total_salmon : ℕ := 971639
  let male_salmon : ℕ := 712261
  total_salmon - male_salmon = 259378 :=
by
  let total_salmon := 971639
  let male_salmon := 712261
  calc
    971639 - 712261 = 259378 := by norm_num

end female_salmon_returned_l598_598253


namespace complex_vector_proof_l598_598809

def OA : ℂ := 2 - 3 * complex.I
def OB : ℂ := -3 + 2 * complex.I
def BA : ℂ := 5 - 5 * complex.I

theorem complex_vector_proof : OA - OB = BA :=
by
  sorry

end complex_vector_proof_l598_598809


namespace initial_sticky_keys_l598_598107

theorem initial_sticky_keys (cleaning_time_per_key remaining_keys cleaned_keys final_total_time assignment_time : ℕ)
  (condition1 : assignment_time = 10) 
  (condition2 : remaining_keys = 14) 
  (condition3 : cleaning_time_per_key = 3) 
  (condition4 : cleaned_keys = 1) 
  (condition5 : final_total_time = 52) :
  cleaned_keys + remaining_keys = 15 :=
by 
  -- Simplifying calculations
  have time_to_clean := remaining_keys * cleaning_time_per_key,
  have total_time_needed := time_to_clean + assignment_time,
  -- Using given conditions
  rw [condition1, condition2, condition3, condition5] at *,
  simp [time_to_clean, total_time_needed],
  -- Concluding the proof
  sorry

end initial_sticky_keys_l598_598107


namespace monthly_pool_cost_is_correct_l598_598403

def cost_of_cleaning : ℕ := 150
def tip_percentage : ℕ := 10
def number_of_cleanings_in_a_month : ℕ := 30 / 3
def cost_of_chemicals_per_use : ℕ := 200
def number_of_chemical_uses_in_a_month : ℕ := 2

def monthly_cost_of_pool : ℕ :=
  let cost_per_cleaning := cost_of_cleaning + (cost_of_cleaning * tip_percentage / 100)
  let total_cleaning_cost := number_of_cleanings_in_a_month * cost_per_cleaning
  let total_chemical_cost := number_of_chemical_uses_in_a_month * cost_of_chemicals_per_use
  total_cleaning_cost + total_chemical_cost

theorem monthly_pool_cost_is_correct : monthly_cost_of_pool = 2050 :=
by
  sorry

end monthly_pool_cost_is_correct_l598_598403


namespace captain_smollett_problem_l598_598234

/-- 
Given the captain's age, the number of children he has, and the length of his schooner, 
prove that the unique solution to the product condition is age = 53 years, children = 6, 
and length = 101 feet, under the given constraints.
-/
theorem captain_smollett_problem
  (age children length : ℕ)
  (h1 : age < 100)
  (h2 : children > 3)
  (h3 : age * children * length = 32118) : age = 53 ∧ children = 6 ∧ length = 101 :=
by {
  -- Proof will be filled in later
  sorry
}

end captain_smollett_problem_l598_598234


namespace peanuts_in_box_l598_598378

theorem peanuts_in_box : 
  let initial_peanuts := 4 in 
  let peanuts_added_by_mary := 4 in 
  let peanuts_taken_by_john := 2 in 
  let peanuts_shared_with_friends := 2 in 
  let final_peanuts := initial_peanuts + peanuts_added_by_mary - peanuts_taken_by_john in
  final_peanuts = 6 :=
by
  let initial_peanuts := 4
  let peanuts_added_by_mary := 4
  let peanuts_taken_by_john := 2
  let peanuts_shared_with_friends := 2
  let final_peanuts := initial_peanuts + peanuts_added_by_mary - peanuts_taken_by_john
  exact Eq.refl final_peanuts

end peanuts_in_box_l598_598378


namespace tom_needs_ten_vaccines_l598_598901

-- Define the constants from conditions
def vaccine_cost : ℤ := 45
def doctor_visit_cost : ℤ := 250
def insurance_coverage : ℚ := 0.8
def trip_cost : ℤ := 1200
def total_payment : ℤ := 1340

-- Define helper calculatiors based on conditions and necessary to prove the theorem
def medical_payment := total_payment - trip_cost
def total_medical_bills : ℚ := medical_payment / ((1 - insurance_coverage : ℚ) : ℚ)
def vaccines_cost := total_medical_bills.to_int - doctor_visit_cost
def number_of_vaccines := vaccines_cost / vaccine_cost

-- Theorem statement to prove the number of vaccines is 10
theorem tom_needs_ten_vaccines : number_of_vaccines = 10 := by
  sorry

end tom_needs_ten_vaccines_l598_598901


namespace gcf_lcm_problem_l598_598821

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcf_lcm_problem :
  GCF (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end gcf_lcm_problem_l598_598821


namespace Niraek_donut_holes_covered_at_same_time_l598_598632

theorem Niraek_donut_holes_covered_at_same_time :
  let surface_area (r : ℕ) : ℝ := 4 * real.pi * (r ^ 2)
  let lcm (a b c : ℕ) : ℕ := nat.lcm a (nat.lcm b c)
  let radius_Niraek := 6
  let radius_Theo := 8
  let radius_Akshaj := 10
  let area_Niraek := surface_area radius_Niraek
  let area_Theo := surface_area radius_Theo
  let area_Akshaj := surface_area radius_Akshaj
  let lcm_areas := lcm (floor area_Niraek) (floor area_Theo) (floor area_Akshaj)
  400 = lcm_areas / (floor area_Niraek) :=
by
  sorry

end Niraek_donut_holes_covered_at_same_time_l598_598632


namespace min_value_of_reciprocals_l598_598423

theorem min_value_of_reciprocals (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 2) :
  (1 / m + 1 / n) = 2 :=
sorry

end min_value_of_reciprocals_l598_598423


namespace problem1_problem2_problem3_problem4_l598_598998

-- Problem 1: Verify the arithmetic expression
theorem problem1 : -2.4 + 3.5 - 4.6 + 3.5 = 0 :=
by
  have h : (-2.4 + 3.5) - 4.6 + 3.5 = 1.1 - 4.6 + 3.5 := by norm_num
  have h2 : 1.1 - 4.6 + 3.5 = -3.5 + 3.5 := by norm_num
  have h3 : -3.5 + 3.5 = 0 := by norm_num
  exact h3

-- Problem 2: Verify the arithmetic expression with negative values
theorem problem2 : (-40) - (-28) - (-19) + (-24) = -17 :=
by
  have h : (-40) + 28 - (-19) + (-24) = (-40) + 28 + 19 + (-24) := by norm_num
  have h2 : (-40 + 28) + 19 - 24 = -12 + 19 - 24 := by norm_num
  have h3 : (-12 + 19) - 24 = 7 - 24 := by norm_num
  have h4 : 7 - 24 = -17 := by norm_num
  exact h4

-- Problem 3: Verify the multiplication of fractions
theorem problem3 : (-3) * (5 / 6) * (-4 / 5) * (-1 / 4) = -1 / 2 :=
by
  have h : (-3) * (5 / 6) = -2.5 := by norm_num
  have h2 : -2.5 * (-4 / 5) = 2 := by norm_num
  have h3 : 2 * (-1 / 4) = -0.5 := by norm_num
  exact h3

-- Problem 4: Verify the complex fraction operation
theorem problem4 : (- (5 / 7)) * (- (4 / 3)) / (- (15 / 7)) = -4 / 9 :=
by
  have h1 : - (5 / 7) * - (4 / 3) = 20 / 21 := by norm_num
  have h2 : (20 / 21) / - (15 / 7) = 20 / 21 * 7 / 15 := by
    norm_num
  have h3 : (20 / 21) * (7 / 15) = 4 / 9 := by
    norm_num
  exact h3 

end problem1_problem2_problem3_problem4_l598_598998


namespace triangle_angle_A_is_60_degrees_l598_598338

theorem triangle_angle_A_is_60_degrees
  (a b c : ℚ) 
  (h1 : (a + Real.sqrt 2)^2 = (b + Real.sqrt 2) * (c + Real.sqrt 2)) : 
  ∠A = 60 := 
sorry

end triangle_angle_A_is_60_degrees_l598_598338


namespace semicircle_problem_l598_598787

theorem semicircle_problem (N : ℕ) (r : ℝ) (π : ℝ) (hπ : 0 < π) 
  (h1 : ∀ (r : ℝ), ∃ (A B : ℝ), A = N * (π * r^2 / 2) ∧ B = (π * (N^2 * r^2 / 2) - N * (π * r^2 / 2)) ∧ A / B = 1 / 3) :
  N = 4 :=
by
  sorry

end semicircle_problem_l598_598787


namespace find_k_l598_598812

theorem find_k
  (y : ℝ)
  (h₁ : log 8 4 = y)
  (h₂ : log 2 81 = k * y) :
  k = 6 := by
  sorry

end find_k_l598_598812


namespace probability_different_colors_probability_shorts_different_from_jerseys_l598_598003

theorem probability_different_colors (shorts_colors : Finset ℕ) (jersey_colors : Finset ℕ) 
(hs : shorts_colors = {1, 2, 3}) (hj : jersey_colors = {1, 2, 3, 4}) :
  (Finset.filter (λ (pair : ℕ × ℕ), pair.1 ≠ pair.2) (Finset.product shorts_colors jersey_colors)).card 
  = 9 :=
by 
  sorry

theorem probability_shorts_different_from_jerseys (shorts_colors : Finset ℕ) (jersey_colors : Finset ℕ) 
(hs : shorts_colors = {1, 2, 3}) (hj : jersey_colors = {1, 2, 3, 4}) :
  let total_combinations := (shorts_colors.card * jersey_colors.card)
  in 9 / total_combinations = 3 / 4 :=
by
  let total_combinations := (shorts_colors.card * jersey_colors.card);
  have h_total : total_combinations = 12, by { unfold total_cominations, simp [hs, hj], };
  rw ←h_total;
  have h_non_matching : 9 = (Finset.filter (λ (pair : ℕ × ℕ), pair.1 ≠ pair.2) (Finset.product shorts_colors jersey_colors)).card, by 
    exact probability_different_colors shorts_colors jersey_colors hs hj;
  rw h_non_matching;
  norm_num;
  sorry

end probability_different_colors_probability_shorts_different_from_jerseys_l598_598003


namespace evaluate_area_l598_598909

-- Define the square and triangle vertices
def square_vertices : List (ℝ × ℝ) := [(0, 0), (0, 10), (10, 10), (10, 0)]
def triangle_vertices : List (ℝ × ℝ) := [(5, 10), (0, 20), (10, 20)]

-- Area calculation goal
def area_of_triangle_segment_square_triangle : ℕ := 50

theorem evaluate_area : 
  let A := (0, 0)
  let B := (0, 10)
  let C := (10, 10)
  let F := (0, 20)
  ∃ (area : ℕ), 
  ((intersection segment F A with line AC and FG) ∧ 
  ((segment FG has coordinates of vertices F and G))
  area = area_of_triangle_segment_square_triangle 
:= 
  sorry

end evaluate_area_l598_598909


namespace ratio_of_areas_GHI_to_JKL_l598_598541

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l598_598541


namespace CorrectChoice_l598_598729

open Classical

-- Define the integer n
variable (n : ℤ)

-- Define proposition p: 2n - 1 is always odd
def p : Prop := ∃ k : ℤ, 2 * k + 1 = 2 * n - 1

-- Define proposition q: 2n + 1 is always even
def q : Prop := ∃ k : ℤ, 2 * k = 2 * n + 1

-- The theorem we want to prove
theorem CorrectChoice : (p n ∨ q n) :=
by
  sorry

end CorrectChoice_l598_598729


namespace four_non_intersecting_spheres_block_light_l598_598795

theorem four_non_intersecting_spheres_block_light :
  ∃ (A B C D : ℝ^3), ∃ (s₁ s₂ s₃ s₄ : Set ℝ^3), 
  (∀ (i j : Fin 4), i ≠ j → s₁ ∩ s₂ = ∅) ∧
  ∀ (O : ℝ^3), (∃ (K₁ K₂ K₃ K₄ : ℝ^3), 
  IsTetrahedron O A B C D →
  s₁ = Sphere O K₁ ∧
  s₂ = Sphere O K₂ ∧
  s₃ = Sphere O K₃ ∧
  s₄ = Sphere O K₄ ∧
  BlockLight O {s₁, s₂, s₃, s₄}) :=
sorry

end four_non_intersecting_spheres_block_light_l598_598795


namespace intersection_of_sets_l598_598438

open Set Real

theorem intersection_of_sets :
  let A := {x : ℝ | x^2 - 2*x - 3 < 0}
  let B := {y : ℝ | ∃ (x : ℝ), y = sin x}
  A ∩ B = Ioc (-1) 1 := by
  sorry

end intersection_of_sets_l598_598438


namespace raft_minimum_capacity_l598_598193

theorem raft_minimum_capacity 
  (mice : ℕ) (mice_weight : ℕ) 
  (moles : ℕ) (mole_weight : ℕ) 
  (hamsters : ℕ) (hamster_weight : ℕ) 
  (raft_cannot_move_without_rower : Bool)
  (rower_condition : ∀ W, W ≥ 2 * mice_weight) :
  mice = 5 → mice_weight = 70 →
  moles = 3 → mole_weight = 90 →
  hamsters = 4 → hamster_weight = 120 →
  ∃ W, (W = 140) :=
by
  intros mice_eq mice_w_eq moles_eq mole_w_eq hamsters_eq hamster_w_eq
  use 140
  sorry

end raft_minimum_capacity_l598_598193


namespace exists_coprime_n_non_perfect_power_l598_598019

theorem exists_coprime_n_non_perfect_power (t : ℕ) (ht : t > 0) :
  ∃ (n : ℕ), n > 1 ∧ nat.coprime n t ∧ ∀ k : ℕ, ¬∃ m : ℕ, ∃ a : ℕ, a > 1 ∧ m^a = n^k + t :=
sorry

end exists_coprime_n_non_perfect_power_l598_598019


namespace sum_of_proper_divisors_of_600_not_perfect_square_600_l598_598566

def is_proper_divisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d ≠ n

def sum_proper_divisors (n : ℕ) : ℕ :=
  (∑ d in (finset.range (n + 1)).filter (is_proper_divisor n), d)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem sum_of_proper_divisors_of_600 : sum_proper_divisors 600 = 1260 := sorry

theorem not_perfect_square_600 : ¬is_perfect_square 600 := sorry

end sum_of_proper_divisors_of_600_not_perfect_square_600_l598_598566


namespace pool_maintenance_cost_l598_598401

theorem pool_maintenance_cost 
  {d_cleaning : Nat}
  {cleaning_cost : ℕ}
  {tip_rate : ℝ}
  {d_month : Nat}
  {cleanings_per_month : ℕ}
  {use_chem_freq : ℕ}
  {chem_cost : ℕ}
  {total_cleaning_cost : ℕ}
  {total_chem_cost : ℕ}
  {total_monthly_cost : ℕ} 
  (hc1 : d_cleaning = 3)
  (hc2 : cleaning_cost = 150)
  (hc3 : tip_rate = 0.1)
  (hc4 : d_month = 30)
  (hc5 : cleanings_per_month = d_month / d_cleaning)
  (hc6 : use_chem_freq = 2)
  (hc7 : chem_cost = 200)
  (hc8 : total_cleaning_cost = cleanings_per_month * (cleaning_cost + (cleaning_cost * tip_rate).toNat))
  (hc9 : total_chem_cost = use_chem_freq * chem_cost)
  (hc10 : total_monthly_cost = total_cleaning_cost + total_chem_cost) :
  total_monthly_cost = 2050 :=
by
  sorry

end pool_maintenance_cost_l598_598401


namespace isosceles_triangle_area_l598_598906

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem isosceles_triangle_area :
  let a := 13
  let b := 13
  let c := 24
  triangle_area a b c = 60 := by
  sorry

end isosceles_triangle_area_l598_598906


namespace constant_term_in_expansion_l598_598916

theorem constant_term_in_expansion :
  let p := (x^3 + 2 * x + 7)
  let q := (2 * x^4 + 3 * x^2 + 10)
  (∀ (x : ℝ), (p * q).coeff 0 = 70) :=
by
  sorry

end constant_term_in_expansion_l598_598916


namespace simplify_expression_l598_598014

theorem simplify_expression (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 :=
by
  sorry

end simplify_expression_l598_598014


namespace ratio_of_areas_GHI_to_JKL_l598_598543

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l598_598543


namespace intersection_distance_l598_598509

theorem intersection_distance :
  let y_line := 5
  let parabola (x : ℝ) := 3 * x^2 + 2 * x - 2
  let roots := { x : ℝ // parabola x = y_line }
  let a := 3
  let b := 2
  let c := -7
  let x1 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let distance := Real.abs (x1 - x2)
  let p := 88
  let q := 3
  let dist := (2 * Real.sqrt 22) / 3
  in p - q = 85 := 
by {
  -- We have to prove that p - q equals 85 given the distance calculation is correct.
  have hpq : (2 * Real.sqrt 22) / 3 = dist, from sorry,
  have hdist : dist = (Real.sqrt p) / q, from sorry,
  have hvalue : p - q = 85, from sorry,
  exact hvalue
}

end intersection_distance_l598_598509


namespace isosceles_trapezoid_legs_squared_l598_598411

theorem isosceles_trapezoid_legs_squared
  (A B C D : Type)
  (AB CD AD BC : ℝ)
  (isosceles_trapezoid : AB = 50 ∧ CD = 14 ∧ AD = BC)
  (circle_tangent : ∃ M : ℝ, M = 25 ∧ ∀ x : ℝ, MD = 7 ↔ AD = x ∧ BC = x) :
  AD^2 = 800 := 
by
  sorry

end isosceles_trapezoid_legs_squared_l598_598411


namespace study_hours_correct_l598_598654

-- Definitions based on conditions
def weeks := 15
def study_hours_subj_A_per_week := 3 * 5
def study_hours_subj_B_per_week := 2 * 3
def study_hours_subj_C_per_week := 4 + 3 + 3
def study_hours_subj_D_per_week := (1 * 5) + 5

def total_study_hours_subj_A := study_hours_subj_A_per_week * weeks
def total_study_hours_subj_B := study_hours_subj_B_per_week * weeks
def total_study_hours_subj_C := study_hours_subj_C_per_week * weeks
def total_study_hours_subj_D := study_hours_subj_D_per_week * weeks

def combined_study_hours := total_study_hours_subj_A + total_study_hours_subj_B + total_study_hours_subj_C + total_study_hours_subj_D

theorem study_hours_correct: 
  total_study_hours_subj_A = 225 ∧
  total_study_hours_subj_B = 90 ∧
  total_study_hours_subj_C = 150 ∧
  total_study_hours_subj_D = 150 ∧
  combined_study_hours = 615 ∧
  total_study_hours_subj_A > total_study_hours_subj_B ∧
  ∀ s ∈ ({total_study_hours_subj_B, total_study_hours_subj_C, total_study_hours_subj_D} : set ℕ), total_study_hours_subj_A > s ∧
  total_study_hours_subj_B < total_study_hours_subj_C ∧
  total_study_hours_subj_B < total_study_hours_subj_D := by
  sorry

end study_hours_correct_l598_598654


namespace abs_neg_frac_l598_598482

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end abs_neg_frac_l598_598482


namespace part_I_part_II_l598_598323

def f(x : ℝ) (a : ℝ) : ℝ := log x - (a * (x - 1) / (x + 1))

theorem part_I (a : ℝ) :
  (∀ x > 0, deriv (fun x => f(x, a)) x ≥ 0) → a ≤ 2 :=
by
  sorry

theorem part_II (m n : ℝ) (h₁ : m > n) (h₂ : n > 0) :
  (m - n) / (log m - log n) < (m + n) / 2 :=
by
  sorry

end part_I_part_II_l598_598323


namespace problem_H_J_sum_l598_598012

theorem problem_H_J_sum (H J K L : ℕ) (h_distinct : list.nodup [H, J, K, L]) (h_set : H ∈ [1, 2, 5, 6] ∧ J ∈ [1, 2, 5, 6] ∧ K ∈ [1, 2, 5, 6] ∧ L ∈ [1, 2, 5, 6])
  (h_diff : H ≠ J ∧ H ≠ K ∧ H ≠ L ∧ J ≠ K ∧ J ≠ L ∧ K ≠ L)
  (h_fraction : (H : ℚ) / J - (K : ℚ) / L = 5 / 6) : H + J = 7 :=
sorry

end problem_H_J_sum_l598_598012


namespace set_equality_l598_598158

theorem set_equality (a b : ℝ) (h : {a, b / a, 1} = {a^2, a + b, 0}) : a ^ 2002 + b ^ 2003 = 1 := 
by 
  sorry

end set_equality_l598_598158


namespace algae_coverage_double_l598_598487

theorem algae_coverage_double (algae_cov : ℕ → ℝ) (h1 : ∀ n : ℕ, algae_cov (n + 2) = 2 * algae_cov n)
  (h2 : algae_cov 24 = 1) : algae_cov 18 = 0.125 :=
by
  sorry

end algae_coverage_double_l598_598487


namespace oliver_earnings_l598_598035

-- Define the conditions
def cost_per_kilo : ℝ := 2
def kilos_two_days_ago : ℝ := 5
def kilos_yesterday : ℝ := kilos_two_days_ago + 5
def kilos_today : ℝ := 2 * kilos_yesterday

-- Calculate the total kilos washed over the three days
def total_kilos : ℝ := kilos_two_days_ago + kilos_yesterday + kilos_today

-- Calculate the earnings over the three days
def earnings : ℝ := total_kilos * cost_per_kilo

-- The theorem we want to prove
theorem oliver_earnings : earnings = 70 := by
  sorry

end oliver_earnings_l598_598035


namespace value_of_f_g_8_l598_598359

theorem value_of_f_g_8 :
  ∀ (f g : ℝ → ℝ), 
  (∀ x, g(x) = 3 * x + 7) → 
  (∀ x, f(x) = 5 * x - 9) → 
  f(g(8)) = 146 := 
by 
  intros f g hg hf 
  simp [hg, hf] 
  sorry

end value_of_f_g_8_l598_598359


namespace domain_of_function_l598_598651

theorem domain_of_function :
  ∀ x : ℝ, (x > 0) ∧ (x ≤ 2) ∧ (x ≠ 1) ↔ ∀ x, (∃ y : ℝ, y = (1 / (Real.log x / Real.log 10) + Real.sqrt (2 - x))) :=
by
  sorry

end domain_of_function_l598_598651


namespace running_days_find_running_days_l598_598060

theorem running_days (d : ℕ) :
  (∀ n : ℕ, 5 * n = 35 → n = d) → d = 5 :=
by
  intro h
  specialize h 5
  have : 5 * 5 = 35 := by norm_num
  exact h this

-- Given that Peter runs 5 miles a day (Andrew runs 2 miles + 3 miles more)
-- and both have run a total of 35 miles.
-- We prove that the number of days they have been running is 5.
theorem find_running_days
  (andrew_miles_per_day : ℕ)
  (peter_miles_per_day : ℕ)
  (total_miles : ℕ)
  (days : ℕ) :
  andrew_miles_per_day = 2 ∧ 
  peter_miles_per_day = andrew_miles_per_day + 3 ∧
  total_miles = 35 ∧
  (λ d, 2 * d + 5 * d = total_miles) days →
  days = 5 :=
by
  rintro ⟨haw, hpw, ht, hdays_eq⟩
  have : 7 * days = total_miles,
  { rw [hdays_eq] },
  rw ht at this,
  norm_num at this,
  exact eq.symm (nat.div_eq_of_eq_mul_right (by norm_num : 7 ≠ 0) this)

end running_days_find_running_days_l598_598060


namespace length_in_miles_l598_598970

-- Define the given conditions
def inches_to_feet (inches : ℝ) : ℝ := inches * 1000
def feet_to_miles (feet : ℝ) : ℝ := feet / 5280

-- Given: One inch represents 1,000 feet and one mile is 5,280 feet
-- Line segment in the drawing is 7.5 inches long
def line_segment_inches : ℝ := 7.5

-- Calculate the length in feet
def length_in_feet : ℝ := inches_to_feet line_segment_inches

-- Prove: Length in miles = 125/88
theorem length_in_miles : feet_to_miles length_in_feet = 125 / 88 := by
  sorry

end length_in_miles_l598_598970


namespace blue_ball_higher_than_orange_l598_598907

noncomputable def S : ℝ := ∑' (k : ℕ), 2^(-(k + 1))

lemma sum_S_eq_one : S = 1 :=
by {
  have : S = ∑' (k : ℕ), 2^(-(k + 1)), 
  { exact S },
  calc
    S = ∑' (k : ℕ), 2^(-(k + 1)) : by simp [S]
    ... = (∑' (k : ℕ), (1 / 2) * (2^(-k))) : by simp [pow_add, div_eq_mul_inv, mul_comm]
    ... = (1 / 2) * (∑' (k : ℕ), 2^(-k)) : by { exact tsum_mul_left (1 / 2) (λ k, 2 / (2 ^ k)) }
    ... = (1 / 2) * 1 : by { simp [tsum_geometric, inv_eq_one_div, tsum_geometric'] },
  exact mul_one ((1 / 2) : ℝ)
}

theorem blue_ball_higher_than_orange : 
  (∑' (k : ℕ), (2^-(k + 1) * 2^-(k + 1)) = 1/3) ∧ (S = 1) → 
  (1 - (∑' (k : ℕ), (2^-(k + 1) * 2^-(k + 1))) / 2) = 1/3 := 
by { intros h1 h2, sorry }

end blue_ball_higher_than_orange_l598_598907


namespace length_FQ_is_3_over_2_l598_598178

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def parabola_eq (P : ℝ × ℝ) : Prop := 
  P.2^2 = 4 * P.1

noncomputable def distance (A B : ℝ × ℝ) : ℝ := 
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem length_FQ_is_3_over_2 
  (P Q F : ℝ × ℝ)
  (hF : F = parabola_focus)
  (hParabolaP : parabola_eq P) 
  (hParabolaQ : parabola_eq Q)
  (hLinePF : ∃ m : ℝ, ∃ b : ℝ, ∀ x : ℝ, (m = (P.2 - F.2) / (P.1 - F.1)) ∧ b = F.2 - m * F.1 ∧ Q.2 = m * Q.1 + b)
  (hDistPF : distance P F = 3) :
  distance F Q = 3 / 2 := 
sorry

end length_FQ_is_3_over_2_l598_598178


namespace cevian_concurrence_l598_598063

-- Define the problem setup
variables {α : Type*} [euclidean_space α] (A B C D E F : α)

-- Define conditions for points D, E, F being points of tangency of the incircle with sides BC, CA, AB respectively
variables (hD : tangent_point A B C D)
          (hE : tangent_point B C A E)
          (hF : tangent_point C A B F)

-- The theorem statement
theorem cevian_concurrence 
  (hD : tangent_point A B C D) 
  (hE : tangent_point B C A E) 
  (hF : tangent_point C A B F) :
  concurrent (line_through A D) (line_through B E) (line_through C F) :=
sorry

end cevian_concurrence_l598_598063


namespace three_days_earning_l598_598037

theorem three_days_earning
  (charge : ℤ := 2)
  (day_before_yesterday_wash : ℤ := 5)
  (yesterday_wash : ℤ := day_before_yesterday_wash + 5)
  (today_wash : ℤ := 2 * yesterday_wash)
  (three_days_earning : ℤ := charge * (day_before_yesterday_wash + yesterday_wash + today_wash)) :
  three_days_earning = 70 := 
by
  have h1 : day_before_yesterday_wash = 5 := by rfl
  have h2 : yesterday_wash = day_before_yesterday_wash + 5 := by rfl
  have h3 : today_wash = 2 * yesterday_wash := by rfl
  have h4 : charge * (day_before_yesterday_wash + yesterday_wash + today_wash) = 70 := sorry
  exact h4

end three_days_earning_l598_598037


namespace boys_from_Pine_l598_598801

/-
We need to prove that the number of boys from Pine Middle School is 70
given the following conditions:
1. There were 150 students in total.
2. 90 were boys and 60 were girls.
3. 50 students were from Maple Middle School.
4. 100 students were from Pine Middle School.
5. 30 of the girls were from Maple Middle School.
-/
theorem boys_from_Pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_girls : ℕ)
  (h_total : total_students = 150) (h_boys : total_boys = 90)
  (h_girls : total_girls = 60) (h_maple : maple_students = 50)
  (h_pine : pine_students = 100) (h_maple_girls : maple_girls = 30) :
  total_boys - maple_students + maple_girls = 70 :=
by
  sorry

end boys_from_Pine_l598_598801


namespace cube_root_neg_eight_l598_598498

theorem cube_root_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by {
  sorry
}

end cube_root_neg_eight_l598_598498


namespace intersect_sets_l598_598336

def setA : Set ℝ := { x | log 2 (x - 1) < 0 }
def setB : Set ℝ := { x | x ≤ 3 }

theorem intersect_sets (x : ℝ) : x ∈ (setA ∩ setB) ↔ (1 < x ∧ x < 2) :=
by
  sorry

end intersect_sets_l598_598336


namespace least_coins_l598_598991

theorem least_coins (n : ℕ) (h₁ : n = 22) 
  (h₂ : ∀ (k : ℕ), n = k + 6 ∨ n = k + 18 ∨ n = k - 12) : 
  ∃ m : ℕ, m ≡ 4 [MOD 6] ∧ (∀ k, n = k + 6 ∨ n = k + 18 ∨ n = k - 12 → m ≤ k) :=
begin
  sorry,
end

end least_coins_l598_598991


namespace haley_total_expenditure_l598_598740

-- Definition of conditions
def ticket_cost : ℕ := 4
def tickets_bought_for_self_and_friends : ℕ := 3
def tickets_bought_for_others : ℕ := 5
def total_tickets : ℕ := tickets_bought_for_self_and_friends + tickets_bought_for_others

-- Proof statement
theorem haley_total_expenditure : total_tickets * ticket_cost = 32 := by
  sorry

end haley_total_expenditure_l598_598740


namespace matt_math_homework_percentage_l598_598446

variable (totalTime : ℕ) (sciencePercentage : ℕ) (otherTime : ℕ)

theorem matt_math_homework_percentage 
  (h1 : totalTime = 150) 
  (h2 : sciencePercentage = 40) 
  (h3 : otherTime = 45) :
  let mathTime := totalTime - (totalTime * sciencePercentage / 100) - otherTime in
  let mathPercentage := (mathTime * 100) / totalTime in
  mathPercentage = 30 :=
by
  sorry

end matt_math_homework_percentage_l598_598446


namespace sum_first_50_terms_l598_598722

def f : ℤ → ℚ :=
  λ x, if x ≤ 7 then 2 * x - 10 else 1 / f (x - 2)

def a (n : ℕ) : ℚ := f n

-- A definition to represent the sum of first n terms of a sequence
def sum_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum a

theorem sum_first_50_terms :
  sum_sequence a 50 = 225 / 4 :=
by 
  sorry

end sum_first_50_terms_l598_598722


namespace smallest_positive_integer_y_l598_598921

def is_multiple_of (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n

theorem smallest_positive_integer_y :
  ∃ y : ℕ, y > 0 ∧ (is_multiple_of ((3 * y)^2 + 3 * 43 * (3 * y) + 43^2) 53) ∧ (∀ z : ℕ, z > 0 ∧ (is_multiple_of ((3 * z)^2 + 3*43*(3*z) + 43^2) 53) → y ≤ z) :=
begin
  use 21,
  split,
  {
    -- Here we show that 'y' is a positive integer.
    norm_num,
  },
  split,
  {
    -- Here we show that 'y' satisfies the condition of the given problem.
    use 1, -- Showing 21 satisfies the multiple relationship.
    norm_num,
  },
  {
    -- Here we prove that 'y' is the smallest positive integer satisfying the condition.
    intros z hz_mul, -- Assume z is another positive integer satisfying the conditions.
    cases hz_mul with k hk,
    norm_num,
  },
  sorry -- proof not provided
end

end smallest_positive_integer_y_l598_598921


namespace product_ab_l598_598881

theorem product_ab : 
  ∃ a b : ℂ, (a = 2 - 2*Complex.I) ∧ (b = 2 + 2*Complex.I) ∧ (∀ z : ℂ, a * z + b * Complex.conj z = 16) ∧ (a * b = 8) :=
  by
  let a := (2 - 2 * Complex.I)
  let b := (2 + 2 * Complex.I)
  have H1 : a * b = 8 := by sorry
  use [a, b]
  simp [H1]
  sorry

end product_ab_l598_598881


namespace AG_perpendicular_GF_l598_598789

open EuclideanGeometry

noncomputable theory

variables {A B C E D F G : Point}
variables {O P : Circle}

-- Definitions based on the conditions
axiom triangle_ABC_right : RightTriangle A B C
axiom angle_BAC_90 : ∠ B A C = 90º
axiom E_on_AB : LiesOn E AB
axiom D_on_AC : LiesOn D AC
axiom BD_CE_intersect_F : Intersects BD CE F
axiom circumcircle_ABC : Circumcircle O A B C
axiom circumcircle_AED : Circumcircle P A E D
axiom O_P_intersect_G : Intersects O P G

-- The theorem to prove
theorem AG_perpendicular_GF : Perpendicular (Line.through A G) (Line.through G F) := by
  sorry

end AG_perpendicular_GF_l598_598789


namespace total_animals_sighted_l598_598257

theorem total_animals_sighted (lions_saturday elephants_saturday buffaloes_sunday leopards_sunday rhinos_monday warthogs_monday : ℕ)
(hlions_saturday : lions_saturday = 3)
(helephants_saturday : elephants_saturday = 2)
(hbuffaloes_sunday : buffaloes_sunday = 2)
(hleopards_sunday : leopards_sunday = 5)
(hrhinos_monday : rhinos_monday = 5)
(hwarthogs_monday : warthogs_monday = 3) :
  lions_saturday + elephants_saturday + buffaloes_sunday + leopards_sunday + rhinos_monday + warthogs_monday = 20 :=
by
  -- This is where the proof will be, but we are skipping the proof here.
  sorry

end total_animals_sighted_l598_598257


namespace min_ships_needed_l598_598602

theorem min_ships_needed (passenger_count : ℕ) (ship_capacity : ℕ) (h_passenger_count : passenger_count = 792) (h_ship_capacity : ship_capacity = 55) : (passenger_count + ship_capacity - 1) / ship_capacity = 15 :=
by
  rw [h_passenger_count, h_ship_capacity]
  sorry

end min_ships_needed_l598_598602


namespace ratio_calculation_l598_598333

theorem ratio_calculation (A B C : ℚ)
  (h_ratio : (A / B = 3 / 2) ∧ (B / C = 2 / 5)) :
  (4 * A + 3 * B) / (5 * C - 2 * B) = 15 / 23 := by
  sorry

end ratio_calculation_l598_598333


namespace range_of_mn_l598_598331

-- Geometry of the given parabola
def parabola (x y : ℝ) := y^2 = 8 * x 

-- Coordinates of the focus
def focus := (2, 0 : ℝ × ℝ)

-- Slopes and points for intersections
def line (k x y : ℝ) := y = k * x - 2 * k

-- Prove the given range for the product of distances
theorem range_of_mn :
  ∀ A B : ℝ × ℝ, 
    (parabola A.1 A.2) → 
    (parabola B.1 B.2) → 
    A ≠ B →
    (∃ k : ℝ, line k A.1 A.2 ∧ line k B.1 B.2) → 
    let m := (dist A focus) in
    let n := (dist B focus) in
    16 ≤ m * n :=
by
  sorry

end range_of_mn_l598_598331


namespace half_dollars_on_Sunday_l598_598857

-- Define the conditions from the problem
def half_dollars_on_Saturday : ℕ := 17
def total_money_received : ℝ := 11.5
def value_per_half_dollar : ℝ := 0.5

-- Calculating the total number of half-dollars received
def total_half_dollars_received : ℕ := (total_money_received / value_per_half_dollar).toNat

-- Lean statement proving Sandy got 6 half-dollars on Sunday
theorem half_dollars_on_Sunday : total_half_dollars_received - half_dollars_on_Saturday = 6 := 
by
  -- skipping proof
  sorry

end half_dollars_on_Sunday_l598_598857


namespace maximum_value_of_linear_expression_l598_598098

theorem maximum_value_of_linear_expression (m n : ℕ) (h_sum : (m*(m + 1) + n^2 = 1987)) : 3 * m + 4 * n ≤ 221 :=
sorry

end maximum_value_of_linear_expression_l598_598098


namespace total_capacity_of_bowl_l598_598008

theorem total_capacity_of_bowl (L C : ℕ) (h1 : L / C = 3 / 5) (h2 : C = L + 18) : L + C = 72 := 
by
  sorry

end total_capacity_of_bowl_l598_598008


namespace ratio_of_a_to_c_l598_598513

variables {a b c d : ℚ}

def a_to_b := (a / b = 5 / 4)
def c_to_d := (c / d = 7 / 3)
def d_to_b := (d / b = 1 / 5)

theorem ratio_of_a_to_c (h1 : a_to_b) (h2 : c_to_d) (h3 : d_to_b) : 
  a / c = 75 / 28 := by
  sorry

end ratio_of_a_to_c_l598_598513


namespace ratio_of_areas_l598_598539

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l598_598539


namespace min_deletions_to_avoid_prime_sums_l598_598087

def set_natural_numbers : Finset ℕ := Finset.range 51 \ {0}

def is_prime_sum (a b : ℕ) : Prop := Nat.Prime (a + b)

def min_deletions (s : Finset ℕ) : ℕ := s.card

theorem min_deletions_to_avoid_prime_sums :
  ∃ (deleted : Finset ℕ) (remaining : Finset ℕ),
    remaining = set_natural_numbers \ deleted ∧ min_deletions deleted ≥ 25 ∧ 
    ∀ (a b : ℕ), a ∈ remaining → b ∈ remaining → a ≠ b → ¬ is_prime_sum a b :=
begin
  sorry
end

end min_deletions_to_avoid_prime_sums_l598_598087


namespace max_bats_purchasable_l598_598975

variable (B C : ℝ)

def budget : ℝ := 210
def cost_eq1 : Prop := 2 * B + 4 * C = 200
def cost_eq2 : Prop := 0.9 * B + 5.7 * C = 220
def cost_per_pair_with_tax : ℝ := (B + C) * 1.07

theorem max_bats_purchasable (h1 : cost_eq1 B C) (h2 : cost_eq2 B C) : (budget / cost_per_pair_with_tax B C).floor = 2 := by
  sorry

end max_bats_purchasable_l598_598975


namespace fraction_expression_evaluation_l598_598568

theorem fraction_expression_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/4) = 1 := 
by
  sorry

end fraction_expression_evaluation_l598_598568


namespace find_m_no_solution_l598_598762

-- Define the condition that the equation has no solution
def no_solution (m : ℤ) : Prop :=
  ∀ x : ℤ, (x + m)/(4 - x^2) + x / (x - 2) ≠ 1

-- State the proof problem in Lean 4
theorem find_m_no_solution : ∀ m : ℤ, no_solution m → (m = 2 ∨ m = 6) :=
by
  sorry

end find_m_no_solution_l598_598762


namespace valid_N_form_l598_598664

def is_valid_N_conditions (N : ℕ) : Prop :=
  -- Requirement 1: Only two of the digits of N are distinct from 0, and one of them is 3.
  (N.digits.count 0 = N.digits.length - 2) ∧ (3 ∈ N.digits) ∧ (∀ d ∈ N.digits, d = 0 ∨ d = 3 ∨ N.digits.count d = 1) ∧ 
  -- Requirement 2: N is a perfect square.
  ∃ k : ℕ, k * k = N

theorem valid_N_form (N : ℕ) : is_valid_N_conditions N ↔ ∃ n : ℕ, N = 36 * 100^n :=
sorry

end valid_N_form_l598_598664


namespace geom_progression_lines_common_point_l598_598623

theorem geom_progression_lines_common_point
  (a c b : ℝ) (r : ℝ)
  (h_geom_prog : c = a * r ∧ b = a * r^2) :
  ∃ (P : ℝ × ℝ), ∀ (a c b : ℝ), c = a * r ∧ b = a * r^2 → (P = (0, 0) ∧ a ≠ 0) :=
by
  sorry

end geom_progression_lines_common_point_l598_598623


namespace solve_equation_l598_598068

-- Define the equation to be solved
def equation (x : ℝ) : Prop := (x + 2)^4 + (x - 4)^4 = 272

-- State the theorem we want to prove
theorem solve_equation : ∃ x : ℝ, equation x :=
  sorry

end solve_equation_l598_598068


namespace ratio_of_areas_l598_598534

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l598_598534


namespace circle_equation_l598_598289

theorem circle_equation (C : ℝ → ℝ → Prop)
  (h₁ : C 1 0)
  (h₂ : C 0 (Real.sqrt 3))
  (h₃ : C (-3) 0) :
  ∃ D E F : ℝ, (∀ x y, C x y ↔ x^2 + y^2 + D * x + E * y + F = 0) ∧ D = 2 ∧ E = 0 ∧ F = -3 := 
by
  sorry

end circle_equation_l598_598289


namespace chessboard_probability_l598_598843

theorem chessboard_probability :
  ∀ (total_squares perimeter_squares : ℕ),
  total_squares = 100 →
  perimeter_squares = 36 →
  (total_squares - perimeter_squares) / total_squares = 16 / 25 := by
  intros total_squares perimeter_squares h_total h_perim
  rw [h_total, h_perim]
  norm_num
  sorry

end chessboard_probability_l598_598843


namespace smallest_gcd_bc_l598_598750

theorem smallest_gcd_bc (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (gcd_ab : Nat.gcd a b = 168) (gcd_ac : Nat.gcd a c = 693) : Nat.gcd b c = 21 := 
sorry

end smallest_gcd_bc_l598_598750


namespace simplify_expression_correct_l598_598465

def simplify_expression : ℚ :=
  15 * (7 / 10) * (1 / 9)

theorem simplify_expression_correct : simplify_expression = 7 / 6 :=
by
  unfold simplify_expression
  sorry

end simplify_expression_correct_l598_598465


namespace tommy_profit_l598_598954

def crate_weight : ℕ := 20
def number_of_crates : ℕ := 3
def cost_of_crates : ℕ := 330
def price_per_kg : ℕ := 6
def rotten_tomatoes : ℕ := 3

theorem tommy_profit :
  let total_weight := crate_weight * number_of_crates in
  let sellable_weight := total_weight - rotten_tomatoes in
  let earnings := sellable_weight * price_per_kg in
  let profit := earnings - cost_of_crates in
  profit = 12 :=
by
  -- proof goes here
  sorry

end tommy_profit_l598_598954


namespace dance_arrangement_possible_l598_598251

variables {B G : Type}
variables (boys : Fin 10 → B) (girls : Fin 10 → G)

structure GirlAttrs :=
(beauty : ℕ) 
(intelligence : ℕ)

variables (beauty intelligence : G → ℕ)
variables (initial_pairing second_pairing : Fin 10 → Fin 10)

def valid_initial_pairing : Prop :=
  ∀ i : Fin 10, initial_pairing i = i

def valid_second_pairing : Prop :=
  ∀ i : Fin 10, 
    (i < 9 → beauty (girls (second_pairing i)) > beauty (girls (initial_pairing i)) ∧ intelligence (girls (second_pairing i)) > intelligence (girls (initial_pairing i))) ∧
    (i = 9 → second_pairing i = 10 - 1) ∧ 
    (i = 10 - 1 → second_pairing i = 0)

def ratio_greater_beauty_intelligence : Prop :=
  (∑ i in Finset.range 9, if beauty (girls (second_pairing i)) > beauty (girls (initial_pairing i)) ∧ intelligence (girls (second_pairing i)) > intelligence (girls (initial_pairing i)) then 1 else 0) ≥ (8 * 1)

theorem dance_arrangement_possible (boys : Fin 10 → B) (girls : Fin 10 → G) 
  [∀ i : Fin 10, valid_initial_pairing initial_pairing] 
  [∀ i : Fin 10, valid_second_pairing second_pairing] :
  ratio_greater_beauty_intelligence beauty intelligence initial_pairing second_pairing :=
sorry

end dance_arrangement_possible_l598_598251


namespace ratio_of_areas_GHI_to_JKL_l598_598544

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l598_598544


namespace length_segment_AB_l598_598785

-- Define the parametric equations of line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + t, 2 - 2 * t)

-- Define the Cartesian equation for the circle C based on the polar equation
def circle_C : set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 = 1}

-- Define the standard form of the line l
def line_standard_form : set (ℝ × ℝ) :=
  {p | 2 * p.1 + p.2 - 4 = 0}

-- Theorem stating the length of the segment AB
theorem length_segment_AB : 
  ∃ A B : ℝ × ℝ, A ∈ line_standard_form ∧ A ∈ circle_C ∧ B ∈ line_standard_form ∧ B ∈ circle_C ∧ dist A B = (2 * sqrt 5) / 5 :=
sorry

end length_segment_AB_l598_598785


namespace remainder_5_pow_100_mod_18_l598_598943

theorem remainder_5_pow_100_mod_18 : (5 ^ 100) % 18 = 13 := 
by
  -- We will skip the proof since only the statement is required.
  sorry

end remainder_5_pow_100_mod_18_l598_598943


namespace sum_x_bounds_l598_598818

theorem sum_x_bounds (n : ℕ) (x : fin n → ℝ) (h_nonneg : ∀ i, 0 ≤ x i)
  (h_eq : (∑ i, (x i)^2) + 2 * ∑ k in finset.range n, ∑ j in finset.range n, (if h : k < j then real.sqrt (k / j:ℝ) * x k * x j else 0) = 1) :
  1 ≤ ∑ i, x i ∧ ∑ i, x i ≤ real.sqrt n :=
by sorry

end sum_x_bounds_l598_598818


namespace three_days_earning_l598_598036

theorem three_days_earning
  (charge : ℤ := 2)
  (day_before_yesterday_wash : ℤ := 5)
  (yesterday_wash : ℤ := day_before_yesterday_wash + 5)
  (today_wash : ℤ := 2 * yesterday_wash)
  (three_days_earning : ℤ := charge * (day_before_yesterday_wash + yesterday_wash + today_wash)) :
  three_days_earning = 70 := 
by
  have h1 : day_before_yesterday_wash = 5 := by rfl
  have h2 : yesterday_wash = day_before_yesterday_wash + 5 := by rfl
  have h3 : today_wash = 2 * yesterday_wash := by rfl
  have h4 : charge * (day_before_yesterday_wash + yesterday_wash + today_wash) = 70 := sorry
  exact h4

end three_days_earning_l598_598036


namespace ratio_of_areas_of_triangles_l598_598546

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l598_598546


namespace constant_term_expansion_l598_598914

theorem constant_term_expansion :
  let f := (x^3 + 2 * x + 7)
  let g := (2 * x^4 + 3 * x^2 + 10)
  ∀ x : ℝ, constant_term (f * g) = 70 :=
by
  sorry

end constant_term_expansion_l598_598914


namespace vitamin_C_in_apple_juice_l598_598452

theorem vitamin_C_in_apple_juice (A O : ℝ) 
  (h₁ : A + O = 185) 
  (h₂ : 2 * A + 3 * O = 452) :
  A = 103 :=
sorry

end vitamin_C_in_apple_juice_l598_598452


namespace find_a_n_find_b_n_find_T_n_l598_598697

-- definitions of sequences and common ratios
variable (a_n b_n : ℕ → ℕ)
variable (S_n T_n : ℕ → ℕ)
variable (q : ℝ)
variable (n : ℕ)

-- conditions
axiom a1 : a_n 1 = 1
axiom S3 : S_n 3 = 9
axiom b1 : b_n 1 = 1
axiom b3 : b_n 3 = 20
axiom q_pos : q > 0
axiom geo_seq : (∀ n, b_n n / a_n n = q ^ (n - 1))

-- goals to prove
theorem find_a_n : ∀ n, a_n n = 2 * n - 1 := 
by sorry

theorem find_b_n : ∀ n, b_n n = (2 * n - 1) * 2 ^ (n - 1) := 
by sorry

theorem find_T_n : ∀ n, T_n n = (2 * n - 3) * 2 ^ n + 3 :=
by sorry

end find_a_n_find_b_n_find_T_n_l598_598697


namespace minimum_raft_weight_l598_598200

-- Define the weights of the animals.
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal.
def num_mice : ℕ := 5
def num_moles : ℕ := 3
def num_hamsters : ℕ := 4

-- The function that represents the minimum weight capacity required for the raft.
def minimum_raft_capacity : ℕ := 140

-- Prove that the minimum raft capacity to transport all animals is 140 grams.
theorem minimum_raft_weight :
  (∀ (total_weight : ℕ), 
    total_weight = (num_mice * weight_mouse) + (num_moles * weight_mole) + (num_hamsters * weight_hamster) →
    (exists (raft_capacity : ℕ), 
      raft_capacity = minimum_raft_capacity ∧
      raft_capacity >= 2 * weight_mouse)) :=
begin
  -- Initial state setup and logical structure.
  intros total_weight total_weight_eq,
  use minimum_raft_capacity,
  split,
  { refl },
  { have h1: 2 * weight_mouse = 140,
    { norm_num },
    rw h1,
    exact le_refl _,
  }
end

end minimum_raft_weight_l598_598200


namespace Friday_exams_left_l598_598447

-- Define initial conditions
def total_exams : ℕ := 200
def Monday_percentage : ℚ := 0.40
def Tuesday_percentage : ℚ := 0.50
def Wednesday_percentage : ℚ := 0.60
def Thursday_percentage : ℚ := 0.30

-- Translate conditions into actual calculations
def Monday_graded := (Monday_percentage * total_exams : ℕ)
def remaining_after_Monday := total_exams - Monday_graded

def Tuesday_graded := (Tuesday_percentage * remaining_after_Monday : ℕ)
def remaining_after_Tuesday := remaining_after_Monday - Tuesday_graded

def Wednesday_graded := (Wednesday_percentage * remaining_after_Tuesday : ℕ)
def remaining_after_Wednesday := remaining_after_Tuesday - Wednesday_graded

def Thursday_graded := (Thursday_percentage * remaining_after_Wednesday : ℚ).toNat
def remaining_after_Thursday := remaining_after_Wednesday - Thursday_graded

-- Theorem to prove the number of exams left on Friday
theorem Friday_exams_left : remaining_after_Thursday = 17 := by
  sorry

end Friday_exams_left_l598_598447


namespace min_binary_questions_to_determine_number_l598_598839

theorem min_binary_questions_to_determine_number (x : ℕ) (h : 10 ≤ x ∧ x ≤ 19) : 
  ∃ (n : ℕ), n = 3 := 
sorry

end min_binary_questions_to_determine_number_l598_598839


namespace women_tea_problem_l598_598935

theorem women_tea_problem : 
  (∀ t : ℝ, ∀ w : ℝ, ∀ m : ℝ, w = 1.5 → m = 1.5 → 1.5 * t = w * (m / 1.5)) →
  (∀ t_9 : ℝ, ∀ m_3 : ℝ, 9 = 6 * 1.5 → m_3 = 2 * 1.5 → 9 * (m_3 / 1.5) = 18) :=
by
  intros h t w m hw hm ht
  sorry

end women_tea_problem_l598_598935


namespace marked_cells_range_l598_598842

-- Define the properties of the 10x10 grid
structure Grid :=
  (cells : Fin 10 × Fin 10 → Bool)

-- Define the property that each 3x3 square must contain exactly one marked cell
def marked_property (g : Grid) : Prop :=
  ∀ i j, (i < 3) → (j < 3) → 
    (Finset.card (Finset.filter id (Finset.image (λ n : Fin 10 × Fin 10, g.cells n) 
      { a | a.1 ∈ (3 * i) + (0 : Fin 3) ∧ a.2 ∈ (3 * j) + (0 : Fin 3) })) = 1)

-- Propose the theorem regarding the number of marked cells ranging from 9 to 16
theorem marked_cells_range {g : Grid} (h : marked_property g) : 
  ∃ n, 9 ≤ n ∧ n ≤ 16 ∧ Finset.card (Finset.filter g.cells Finset.univ) = n :=
sorry

end marked_cells_range_l598_598842


namespace inequality_proof_l598_598939

noncomputable theory

open Real

theorem inequality_proof (n : ℕ) (x : Fin (n+2) → ℝ)
  (h_pos : ∀ i, 0 < x i) 
  (h_prod : (∏ i : Fin (n+2), x i) = 1) :
  (∑ i : Fin (n+2), n^(1 / x i)) ≥ (∑ i : Fin (n+2), n^(x i^(1 / n))) :=
by
  sorry

end inequality_proof_l598_598939


namespace largest_integer_of_five_consecutive_l598_598264

def is_non_prime (n : ℕ) : Prop := ¬ (nat.prime n)

def five_consecutive_integers_meet_conditions (a : ℕ) : Prop :=
  a > 25 ∧ a < 50 ∧ 
  is_non_prime a ∧ 
  is_non_prime (a + 1) ∧ 
  is_non_prime (a + 2) ∧ 
  is_non_prime (a + 3) ∧ 
  is_non_prime (a + 4) ∧ 
  (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) % 10 = 0

theorem largest_integer_of_five_consecutive :
  ∃ (a : ℕ), five_consecutive_integers_meet_conditions a ∧ (a + 4 = 36) :=
begin
  sorry
end

end largest_integer_of_five_consecutive_l598_598264


namespace three_character_license_plates_l598_598746

theorem three_character_license_plates :
  let consonants := 20
  let vowels := 6
  (consonants * consonants * vowels = 2400) :=
by
  sorry

end three_character_license_plates_l598_598746


namespace unique_area_not_determined_l598_598621

def is_perpendicular (v₁ v₂ : ℕ → ℝ) : Prop :=
  v₁.dot_product v₂ = 0

def equilateral (P : ℕ → ℝ → Prop) (a : ℝ) : Prop :=
  ∀ (i : ℕ), ∃ (j : ℕ), P i (j * a)

-- Function to determine possible configurations.
noncomputable def polygon_configurations : (Fin 20) → ℝ × ℝ := sorry

theorem unique_area_not_determined (a : ℝ) :
  (∀ i : Fin 20, equilateral polygon_configurations a) ∧
  (∀ i : Fin 20, is_perpendicular (polygon_configurations i) (polygon_configurations ((i + 1) % 20))) →
  ¬ ∃ A : ℝ, ∀ i : Fin 20, polygon_configurations a = A :=
by {
  sorry
}

end unique_area_not_determined_l598_598621


namespace chord_bisected_line_eq_l598_598077

theorem chord_bisected_line_eq (x y : ℝ) (hx1 : x^2 + 4 * y^2 = 36) (hx2 : (4, 2) = ((x1 + x2) / 2, (y1 + y2) / 2)) :
  x + 2 * y - 8 = 0 :=
sorry

end chord_bisected_line_eq_l598_598077


namespace distribution_of_cousins_l598_598025

theorem distribution_of_cousins : 
  let num_of_cousins := 5 
  let num_of_rooms := 5
  let possible_distributions := 
    [(5,0,0,0,0), (4,1,0,0,0), (3,2,0,0,0), (3,1,1,0,0), (2,2,1,0,0), (2,1,1,1,0), (1,1,1,1,1)]
  let ways_of_distribution (dist : ℕ × ℕ × ℕ × ℕ × ℕ) : ℕ :=
    match dist with
    | (5,0,0,0,0)   => 1
    | (4,1,0,0,0)   => 5
    | (3,2,0,0,0)   => 10
    | (3,1,1,0,0)   => 30
    | (2,2,1,0,0)   => 30
    | (2,1,1,1,0)   => 60
    | (1,1,1,1,1)   => 1
    | _             => 0
  in
  (possible_distributions.map ways_of_distribution).sum = 137 := 
by sorry

end distribution_of_cousins_l598_598025


namespace similar_triangles_THN_and_PBC_l598_598408

section SimilarTriangles

variables {P A B C X Y Z O H N T : Type} 
  [EuclideanGeometry P A B C X Y Z O H N T]

-- Given conditions in triangle ABC
def P_insides_ABC (P A B C : Point) : Prop := 
  ∠P A C = ∠P C B ∧ 
  exists X Y Z, 
    Perpendicular (P, X) (B, C) ∧ 
    Perpendicular (P, Y) (C, A) ∧ 
    Perpendicular (P, Z) (A, B)

-- O is the circumcenter of triangle XYZ
def O_circumcenter_of_XYZ (O X Y Z : Point) : Prop := 
  Circumcenter O (Triangle.mk X Y Z)

-- H is the foot of the altitude from B to AC
def H_foot_of_altitude_B_to_AC (H B A C : Point) : Prop := 
  FootOfAltitude H B A C

-- N is the midpoint of AC
def N_midpoint_of_AC (N A C : Point) : Prop := 
  Midpoint N A C

-- TYPO is a parallelogram
def TYPO_is_parallelogram (T Y P O : Point) : Prop := 
  Parallelogram T Y P O

-- Problem statement: Showing the similarity of triangles THN and PBC
theorem similar_triangles_THN_and_PBC 
  {P A B C X Y Z O H N T : Point}
  (hP : P_insides_ABC P A B C)
  (hO : O_circumcenter_of_XYZ O X Y Z)
  (hH : H_foot_of_altitude_B_to_AC H B A C)
  (hN : N_midpoint_of_AC N A C)
  (hT : TYPO_is_parallelogram T Y P O) :
  Similar (Triangle.mk T H N) (Triangle.mk P B C) :=
sorry

end SimilarTriangles

end similar_triangles_THN_and_PBC_l598_598408


namespace magic_to_multiplicative_magic_l598_598342

def isMagicSquare (square : Array (Array Nat)) (sum_val : Nat) : Prop :=
  (∀ i, (Array.foldl (λ acc x => acc + x) 0 square[i]) = sum_val) ∧  -- rows
  (∀ j, (Array.foldl (λ acc x => acc + x[j]) 0 square) = sum_val) ∧  -- columns
  ((Array.foldl (λ acc i => acc + square[i][i]) 0 (Array.range 3)) = sum_val) ∧
  ((Array.foldl (λ acc i => acc + square[i][2 - i]) 0 (Array.range 3)) = sum_val)

def isMultiplicativeMagicSquare (square : Array (Array Nat)) (prod_val : Nat) : Prop :=
  (∀ i, (Array.foldl (λ acc x => acc * x) 1 square[i]) = prod_val) ∧  -- rows
  (∀ j, (Array.foldl (λ acc x => acc * x[j]) 1 square) = prod_val) ∧  -- columns
  ((Array.foldl (λ acc i => acc * square[i][i]) 1 (Array.range 3)) = prod_val) ∧
  ((Array.foldl (λ acc i => acc * square[i][2 - i]) 1 (Array.range 3)) = prod_val)

theorem magic_to_multiplicative_magic :
  ∃ (rearranged : Array (Array Nat)),
    isMagicSquare #[#[27, 20, 25], #[22, 24, 26], #[23, 28, 21]] 72 ∧
    isMultiplicativeMagicSquare rearranged 7488 :=
by
  sorry

end magic_to_multiplicative_magic_l598_598342


namespace find_g_inv_84_l598_598755

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem find_g_inv_84 : g 3 = 84 → ∃ x, g x = 84 ∧ x = 3 :=
by
  sorry

end find_g_inv_84_l598_598755


namespace count_perfect_squares_ending_4_5_6_l598_598346

theorem count_perfect_squares_ending_4_5_6 : 
  ∃ n, n = 36 ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 70 ∧ (let d := k % 10 in d = 2 ∨ d = 8 ∨ d = 5 ∨ d = 4 ∨ d = 6) → k^2 < 5000) := 
sorry

end count_perfect_squares_ending_4_5_6_l598_598346


namespace g_expression_l598_598942

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := sorry

theorem g_expression :
  (∀ x : ℝ, g (x + 2) = f x) → ∀ x : ℝ, g x = 2 * x - 1 :=
by
  sorry

end g_expression_l598_598942


namespace polynomial_remainder_l598_598712

theorem polynomial_remainder (a b c d : ℝ)
    (h1 : (a + b + c + d = 1))
    (h2 : (8a + 4b + 2c + d = 3)) :
    ∃ m n : ℝ, (m = 2) ∧ (n = -1) ∧ ∀ x : ℝ, (ax^3 + bx^2 + cx + d) = (x-1)*(x-2)*(ax + b - a) + (2x - 1) :=
by {
  sorry
}

end polynomial_remainder_l598_598712


namespace sum_of_other_endpoint_l598_598050

theorem sum_of_other_endpoint (x y : ℝ) (h1 : (6 + x) / 2 = 3) (h2 : (-2 + y) / 2 = 5) : x + y = 12 := 
by {
  sorry
}

end sum_of_other_endpoint_l598_598050


namespace other_endpoint_coordinates_sum_l598_598055

noncomputable def other_endpoint_sum (x1 y1 x2 y2 xm ym : ℝ) : ℝ :=
  let x := 2 * xm - x1
  let y := 2 * ym - y1
  x + y

theorem other_endpoint_coordinates_sum :
  (other_endpoint_sum 6 (-2) 0 12 3 5) = 12 := by
  sorry

end other_endpoint_coordinates_sum_l598_598055


namespace symmetric_circle_equation_l598_598880

theorem symmetric_circle_equation :
  ∀ (x y : ℝ), (x + 2) ^ 2 + y ^ 2 = 5 → (x - 2) ^ 2 + y ^ 2 = 5 :=
by 
  sorry

end symmetric_circle_equation_l598_598880


namespace john_made_47000_l598_598400

variable (original_cost discount_rate prize money_kept discounted_cost : ℝ)

def original_cost := 20000

def discount_rate := 0.20

def prize := 70000

def money_kept := prize * 0.90

def discounted_cost := original_cost - (discount_rate * original_cost)

theorem john_made_47000 :
  money_kept - discounted_cost = 47000 :=
by
  sorry

end john_made_47000_l598_598400


namespace mandy_brother_age_ratio_l598_598443

theorem mandy_brother_age_ratio :
  ∀ (M B S : ℤ) (x : ℚ), 
  M = 3 →
  B = x * M →
  S = B - 5 →
  M - S = 4 →
  x = 4 / 3 :=
by 
  intros M B S x hM hB hS hDiff
  rw [hM, hS, hB]
  sorry

end mandy_brother_age_ratio_l598_598443


namespace raft_minimum_capacity_l598_598205

theorem raft_minimum_capacity (n_mice n_moles n_hamsters : ℕ)
  (weight_mice weight_moles weight_hamsters : ℕ)
  (total_weight : ℕ) :
  n_mice = 5 →
  weight_mice = 70 →
  n_moles = 3 →
  weight_moles = 90 →
  n_hamsters = 4 →
  weight_hamsters = 120 →
  (∀ (total_weight : ℕ), total_weight = n_mice * weight_mice + n_moles * weight_moles + n_hamsters * weight_hamsters) →
  (∃ (min_capacity: ℕ), min_capacity ≥ 140) :=
by
  intros
  sorry

end raft_minimum_capacity_l598_598205


namespace minimum_score_118_l598_598630

noncomputable def minimum_score (μ σ : ℝ) (p : ℝ) : ℝ :=
  sorry

theorem minimum_score_118 :
  minimum_score 98 10 (9100 / 400000) = 118 :=
by sorry

end minimum_score_118_l598_598630


namespace unique_function_exists_sum_f_l598_598825

def f (n : ℕ) : ℕ := sorry

theorem unique_function_exists (h : ∀ m n : ℕ, f(m + f(n)) = n + f(m + 95)) :
  ∃! f : ℕ → ℕ, ∀ m n : ℕ, f(m + f(n)) = n + f(m + 95) :=
begin
  sorry
end

theorem sum_f (h : ∀ m n : ℕ, f(m + f(n)) = n + f(m + 95)) :
  (finset.range 19).sum (λ k, f(k + 1)) = 1995 :=
begin
  sorry
end

end unique_function_exists_sum_f_l598_598825


namespace range_of_a_l598_598883

noncomputable def f (x a : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (3 * x₁^2 + 6 * a * x₁ + 3 * (a + 2) = 0) ∧ 
   (3 * x₂^2 + 6 * a * x₂ + 3 * (a + 2) = 0)) → a ∈ set.Iio (-1) ∪ set.Ioi 2 :=
by
  sorry

end range_of_a_l598_598883


namespace boys_in_class_l598_598444

theorem boys_in_class (total_students : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ)
    (h_ratio : ratio_girls = 3) (h_ratio_boys : ratio_boys = 4)
    (h_total_students : total_students = 35) :
    ∃ boys, boys = 20 :=
by
  let k := total_students / (ratio_girls + ratio_boys)
  have hk : k = 5 := by sorry
  let boys := ratio_boys * k
  have h_boys : boys = 20 := by sorry
  exact ⟨boys, h_boys⟩

end boys_in_class_l598_598444


namespace max_area_of_triangle_OAB_l598_598318

noncomputable def max_area_OAB : ℝ :=
  let k := by sorry -- This represents the condition that line l has a variable slope k
  let d := 1 / Real.sqrt (k^2 + 1)
  let t := k^2 + 1
  Real.sqrt (4 * t - 1) / t

theorem max_area_of_triangle_OAB (x y : ℝ) (l: ℝ → ℝ) (P : ℝ × ℝ) (hP : P = (0, 1)) 
  (h1 : ∀ x y, x^2 + y^2 = 4)
  (h2 : ∀ x, l x = k * x + 1)
  (h3 : ∀ k : ℝ, maximized t = 1) :
  max_area_OAB = Real.sqrt 3 :=
by
  sorry

end max_area_of_triangle_OAB_l598_598318


namespace shiela_used_seven_colors_l598_598220

theorem shiela_used_seven_colors (total_blocks : ℕ) (blocks_per_color : ℕ) 
    (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) : 
    total_blocks / blocks_per_color = 7 :=
by
  sorry

end shiela_used_seven_colors_l598_598220


namespace F_101_F_unbounded_l598_598334

def F : ℕ → ℝ
| 1       := 3
| (n + 2) := (2 * F (n + 1) + 3) / 2

theorem F_101 : F 101 = 153 := 
sorry

theorem F_unbounded : ∀ M : ℝ, ∃ N : ℕ, ∀ n : ℕ, n ≥ N → F n > M :=
sorry

end F_101_F_unbounded_l598_598334


namespace rate_of_simple_interest_is_correct_l598_598997

-- Define the given conditions
def principal : ℝ := 25000
def amount_after_12_years : ℝ := 35500
def time_years : ℝ := 12

-- Define the simple interest formula
def simple_interest (P R T : ℝ) := P * R * T / 100

-- Define the interest calculation
def interest := amount_after_12_years - principal

-- State the main proposition
theorem rate_of_simple_interest_is_correct :
  ∃ R : ℝ, simple_interest principal R time_years = interest ∧ R = 3.5 :=
by
  use 3.5
  slog_sorry

end rate_of_simple_interest_is_correct_l598_598997


namespace sum_first_2018_terms_l598_598335

-- Define the sequence according to the given problem conditions
def sequence : ℕ → ℝ
| 0 => 1 / 2
| (n + 1) => 1 / 2 + sqrt (sequence n - (sequence n) ^ 2)

-- Theorem statement for the sum of the first 2018 terms
theorem sum_first_2018_terms : 
  (Finset.range 2018).sum sequence = 3027 / 2 :=
  sorry

end sum_first_2018_terms_l598_598335


namespace dvds_bought_online_l598_598023

theorem dvds_bought_online (total_dvds : ℕ) (store_dvds : ℕ) (online_dvds : ℕ) :
  total_dvds = 10 → store_dvds = 8 → online_dvds = total_dvds - store_dvds → online_dvds = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dvds_bought_online_l598_598023


namespace raft_min_capacity_l598_598191

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end raft_min_capacity_l598_598191


namespace increasing_iff_derivative_positive_l598_598765

theorem increasing_iff_derivative_positive {f : ℝ → ℝ} (h : ∀ x, differentiable_at ℝ f x) :
  (∀ x, f'(x) > 0 → ∀ x, f x is_strictly_increasing) :=
sorry

end increasing_iff_derivative_positive_l598_598765


namespace DaisyDogToys_l598_598644

-- Defining the conditions as variables and premises
variables (MondayToys TuesdayLeftToys BoughtTuesdayToys WednesdayBoughtToys : ℕ)
variables (LostToysMondayTuesday : ℕ)

-- The given conditions from the problem
def condition1 := MondayToys = 5
def condition2 := TuesdayLeftToys = 3
def condition3 := BoughtTuesdayToys = 3
def condition4 := WednesdayBoughtToys = 5
def condition5 := LostToysMondayTuesday = MondayToys - TuesdayLeftToys

-- The target problem statement
theorem DaisyDogToys :
  MondayToys = 5 →
  TuesdayLeftToys = 3 →
  BoughtTuesdayToys = 3 →
  WednesdayBoughtToys = 5 →
  LostToysMondayTuesday = MondayToys - TuesdayLeftToys →
  MondayToys + LostToysMondayTuesday + BoughtTuesdayToys + WednesdayBoughtToys = 15 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact add_assoc (5 + 2) 3 5 ▸ rfl

end DaisyDogToys_l598_598644


namespace length_of_field_l598_598495

-- Define the known conditions
def width := 50
def total_distance_run := 1800
def num_laps := 6

-- Define the problem statement
theorem length_of_field :
  ∃ L : ℕ, 6 * (2 * (L + width)) = total_distance_run ∧ L = 100 :=
by
  sorry

end length_of_field_l598_598495


namespace least_area_of_figure_l598_598803

theorem least_area_of_figure (c : ℝ) (hc : c > 1) : 
  ∃ A : ℝ, A = (4 / 3) * (c - 1)^(3 / 2) :=
by
  sorry

end least_area_of_figure_l598_598803


namespace pages_revised_only_once_l598_598093

theorem pages_revised_only_once 
  (total_pages : ℕ)
  (cost_per_page_first_time : ℝ)
  (cost_per_page_revised : ℝ)
  (revised_twice_pages : ℕ)
  (total_cost : ℝ)
  (pages_revised_only_once : ℕ) :
  total_pages = 100 →
  cost_per_page_first_time = 10 →
  cost_per_page_revised = 5 →
  revised_twice_pages = 30 →
  total_cost = 1400 →
  10 * (total_pages - pages_revised_only_once - revised_twice_pages) + 
  15 * pages_revised_only_once + 
  20 * revised_twice_pages = total_cost →
  pages_revised_only_once = 20 :=
by
  intros 
  sorry

end pages_revised_only_once_l598_598093


namespace truck_loading_time_l598_598622

theorem truck_loading_time :
  let worker1_rate := (1:ℝ) / 6
  let worker2_rate := (1:ℝ) / 5
  let combined_rate := worker1_rate + worker2_rate
  (combined_rate != 0) → 
  (1 / combined_rate = (30:ℝ) / 11) :=
by
  sorry

end truck_loading_time_l598_598622


namespace three_letter_initials_l598_598743

theorem three_letter_initials : 
  let letters := { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I' }
  in finset.card (finset.unordered_triples_valued letters) = 504 :=
sorry

end three_letter_initials_l598_598743


namespace parallel_conditions_l598_598988

variables {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

def is_zero_vector (v : V) : Prop := ∥v∥ = 0

def is_parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b

theorem parallel_conditions :
  is_parallel a b ↔ is_zero_vector a ∨ is_zero_vector b ∨ a = -2 • b :=
by
  sorry

end parallel_conditions_l598_598988


namespace vector_on_line_l598_598240

theorem vector_on_line (t : ℝ) (x y : ℝ) : 
  (x = 3 * t + 1) → (y = 2 * t + 3) → 
  ∃ t, (∃ x y, (x = 3 * t + 1) ∧ (y = 2 * t + 3) ∧ (x = 23 / 2) ∧ (y = 10)) :=
  by
  sorry

end vector_on_line_l598_598240


namespace happy_valley_farm_arrangements_l598_598473

theorem happy_valley_farm_arrangements :
  let chickens := 5
      dogs := 3
      cats := 4
      total_animals := chickens + dogs + cats
      arrangements := Nat.factorial 3 * Nat.factorial chickens * Nat.factorial dogs * Nat.factorial cats
  in
  total_animals = 12 ∧ arrangements = 103680 := 
by
  let chickens := 5
      dogs := 3
      cats := 4
      total_animals := chickens + dogs + cats
      arrangements := Nat.factorial 3 * Nat.factorial chickens * Nat.factorial dogs * Nat.factorial cats
  have h1: total_animals = 12, from rfl
  have h2: arrangements = 103680, from sorry
  exact ⟨h1, h2⟩

end happy_valley_farm_arrangements_l598_598473


namespace tammy_loops_l598_598873

/-- Tammy's weekly running schedule -/
def weekly_running_distance : ℕ := 3500

/-- Number of days Tammy runs in a week (excluding Sunday) -/
def running_days : ℕ := 6

/-- Average distance Tammy runs per day -/
def average_distance_per_day : ℝ := weekly_running_distance / running_days

/-- Distance Tammy runs on weekdays (10% more than average) -/
def weekday_distance : ℝ := average_distance_per_day * 1.10

/-- Distance Tammy runs on weekends (20% less than average) -/
def weekend_distance : ℝ := average_distance_per_day * 0.80

/-- Length of the school track (in meters) -/
def school_track_length : ℝ := 50

/-- Length of the public track (in meters) -/
def public_track_length : ℝ := 100

/-- Number of loops Tammy should run on the school track on weekdays -/
def loops_on_weekdays : ℕ := (weekday_distance / school_track_length).ceil.toNat

/-- Number of loops Tammy should run on the public track on weekends -/
def loops_on_weekends : ℕ := (weekend_distance / public_track_length).ceil.toNat

theorem tammy_loops :
  loops_on_weekdays = 13 ∧ loops_on_weekends = 5 := by
  sorry

end tammy_loops_l598_598873


namespace hyperbola_eccentricity_l598_598502

theorem hyperbola_eccentricity :
  (∃ a b c : ℝ, a = 3 ∧ b = 4 ∧ c = 5 ∧ c * c = a * a + b * b) →
  eccentricity (hyperbola 9 16) = 5 / 3 :=
sorry

end hyperbola_eccentricity_l598_598502


namespace daisy_dog_toys_l598_598646

theorem daisy_dog_toys :
  let monday_toys := 5
  let tuesday_left := 3
  let tuesday_bought := 3
  let wednesday_bought := 5 in
  monday_toys + tuesday_bought + wednesday_bought + (tuesday_left + monday_toys) - (tuesday_left + monday_toys) = 13 := by
  sorry

end daisy_dog_toys_l598_598646


namespace shells_total_l598_598468

theorem shells_total (a s v : ℕ) 
  (h1 : s = v + 16) 
  (h2 : v = a - 5) 
  (h3 : a = 20) : 
  s + v + a = 66 := 
by
  sorry

end shells_total_l598_598468


namespace ratio_of_areas_of_triangles_l598_598547

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l598_598547


namespace arithmetic_sequence_has_11_terms_l598_598303

theorem arithmetic_sequence_has_11_terms
  (a1 d : ℝ)
  (h_sum_first_four : 4 * a1 + 6 * d = 26)
  (h_sum_last_four : ∀ n, 4 * a1 + (4 * n - 10) * d = 110)
  (h_total_sum : ∃ n, n / 2 * (2 * a1 + (n - 1) * d) = 187) :
  ∃ n : ℝ, n = 11 := by
  sorry

end arithmetic_sequence_has_11_terms_l598_598303


namespace binary_subtraction_to_decimal_l598_598557

theorem binary_subtraction_to_decimal :
  (511 - 63 = 448) :=
by
  sorry

end binary_subtraction_to_decimal_l598_598557


namespace ribbon_left_l598_598002

-- Define the variables
def T : ℕ := 18 -- Total ribbon in yards
def G : ℕ := 6  -- Number of gifts
def P : ℕ := 2  -- Ribbon per gift in yards

-- Statement of the theorem
theorem ribbon_left (T G P : ℕ) : (T - G * P) = 6 :=
by
  -- Add conditions as Lean assumptions
  have hT : T = 18 := sorry
  have hG : G = 6 := sorry
  have hP : P = 2 := sorry
  -- Now prove the final result
  sorry

end ribbon_left_l598_598002


namespace jeans_cost_proof_l598_598996

def cheaper_jeans_cost (coat_price: Float) (backpack_price: Float) (shoes_price: Float) (subtotal: Float) (difference: Float): Float :=
  let known_items_cost := coat_price + backpack_price + shoes_price
  let jeans_total_cost := subtotal - known_items_cost
  let x := (jeans_total_cost - difference) / 2
  x

def more_expensive_jeans_cost (cheaper_price : Float) (difference: Float): Float :=
  cheaper_price + difference

theorem jeans_cost_proof : ∀ (coat_price backpack_price shoes_price subtotal difference : Float),
  coat_price = 45 →
  backpack_price = 25 →
  shoes_price = 30 →
  subtotal = 139 →
  difference = 15 →
  cheaper_jeans_cost coat_price backpack_price shoes_price subtotal difference = 12 ∧
  more_expensive_jeans_cost (cheaper_jeans_cost coat_price backpack_price shoes_price subtotal difference) difference = 27 :=
by
  intros coat_price backpack_price shoes_price subtotal difference
  intros h1 h2 h3 h4 h5
  sorry

end jeans_cost_proof_l598_598996


namespace Polly_lunch_time_l598_598454

-- Define the conditions
def breakfast_time_per_day := 20
def total_days_in_week := 7
def dinner_time_4_days := 10
def remaining_days_in_week := 3
def remaining_dinner_time_per_day := 30
def total_cooking_time := 305

-- Define the total time Polly spends cooking breakfast in a week
def total_breakfast_time := breakfast_time_per_day * total_days_in_week

-- Define the total time Polly spends cooking dinner in a week
def total_dinner_time := (dinner_time_4_days * 4) + (remaining_dinner_time_per_day * remaining_days_in_week)

-- Define the time Polly spends cooking lunch in a week
def lunch_time := total_cooking_time - (total_breakfast_time + total_dinner_time)

-- The theorem to prove Polly's lunch time
theorem Polly_lunch_time : lunch_time = 35 :=
by
  sorry

end Polly_lunch_time_l598_598454


namespace ferris_wheel_problem_l598_598944

def radius := 30
def period := 120

def height (t : ℝ) := radius * Real.cos (t * Real.pi / (period / 2)) + radius

theorem ferris_wheel_problem :
  ∃ t₁ t₂: ℝ, 
  height t₁ = 55 ∧ height (t₁ + t₂) = 60 ∧ 
  t₁ = 98 ∧ t₂ = 22 :=
by
  sorry

end ferris_wheel_problem_l598_598944


namespace smallest_sum_97_l598_598512

theorem smallest_sum_97 (X Y Z W : ℕ) 
  (h1 : X + Y + Z = 3)
  (h2 : 4 * Z = 7 * Y)
  (h3 : 16 ∣ Y) : 
  X + Y + Z + W = 97 :=
by
  sorry

end smallest_sum_97_l598_598512


namespace weight_problem_l598_598230

variable (M T : ℕ)

theorem weight_problem
  (h1 : 220 = 3 * M + 10)
  (h2 : T = 2 * M)
  (h3 : 2 * T = 220) :
  M = 70 ∧ T = 140 :=
by
  sorry

end weight_problem_l598_598230


namespace total_houses_l598_598375

theorem total_houses (houses_one_side : ℕ) (houses_other_side : ℕ) (h1 : houses_one_side = 40) (h2 : houses_other_side = 3 * houses_one_side) : houses_one_side + houses_other_side = 160 :=
by sorry

end total_houses_l598_598375


namespace ratio_of_areas_GHI_to_JKL_l598_598540

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l598_598540


namespace product_is_102_l598_598792

open Classical

variables {X Y Z X' Y' Z' P : Type}

-- Define the conditions
def on_sides_of_triangle (X' Y' Z' : Type) (YZ XZ XY : Type) : Prop := 
  ∃ (YZ XZ XY : Type), True 

def lines_concur (A B C A' B' C' P : Type) : Prop := 
  ∃ (P : Type), True

def given_sum (XP X' XP' YP Y' YP' ZP Z' PZ : Type) (a b c : ℝ) : Prop :=
  a + b + c = 100

-- Starts the theorem statement
theorem product_is_102 (X Y Z X' Y' Z' P : Type) 
  (h1 : on_sides_of_triangle X' Y' Z' YZ XZ XY)
  (h2 : lines_concur X Y Z X' Y' Z' P)
  (h3 : given_sum XP PX' YP PY' ZP PZ 100) :
  (XP / PX') * (YP / PY') * (ZP / PZ') = 102 :=
by
  sorry

end product_is_102_l598_598792


namespace partitions_le_factorial_l598_598955

def decomposition (α : Type) (S : set α) := { T : set (set α) // ∀ A ∈ T, ∀ B ∈ T, A ≠ B → A ∩ B = ∅ }

def partition_function (n : ℕ) : ℕ := sorry

theorem partitions_le_factorial {n : ℕ} (h : n ≥ 1) : partition_function n ≤ nat.factorial n := sorry

end partitions_le_factorial_l598_598955


namespace tangent_line_eq_monotonic_intervals_extreme_values_inequality_holds_l598_598726

-- Definition of the function f
def f (x : ℝ) (k : ℝ) := x^3 + k * Real.log x

-- First derivative of f
def f_prime (x : ℝ) (k : ℝ) := 3 * x^2 + k / x

-- Definition of the function g
def g (x : ℝ) := f x 6 - f_prime x 6 + 9 / x

-- Statement of the proof problem
theorem tangent_line_eq (k : ℝ) : k = 6 → (∀ x : ℝ, x = 1 → ∃ m b : ℝ, m = 9 ∧ b = -8 ∧ (∀ y : ℝ, y = f x k → y = m * x + b)) :=
by
  sorry

theorem monotonic_intervals_extreme_values : (∀ x : ℝ, ((0 < x ∧ x < 1) → (g x) < 1) ∧ ((1 < x) → (g x) > 1) ∧ (g 1 = 1)) :=
by
  sorry

theorem inequality_holds (k : ℝ) (x₁ x₂ : ℝ) : (k ≥ -3) → (1 ≤ x₁ ∧ 1 ≤ x₂) → (x₁ > x₂) → ((f_prime x₁ k + f_prime x₂ k) / 2 > (f x₁ k - f x₂ k) / (x₁ - x₂)) :=
by
  sorry

end tangent_line_eq_monotonic_intervals_extreme_values_inequality_holds_l598_598726


namespace range_of_a_no_good_points_l598_598673

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 1

theorem range_of_a_no_good_points (a : ℝ) : (∃ x₀ : ℝ, f x₀ a = x₀) ↔ a ∈ Ioc (-1/2 : ℝ) (3/2 : ℝ) := by
sorry

end range_of_a_no_good_points_l598_598673


namespace triangle_lengths_l598_598469

theorem triangle_lengths (A B C P D E F : Point) (h₁ : inside_triangle P A B C)
  (h₂ : intersection AP BC D) (h₃ : intersection BP CA E) (h₄ : intersection CP AB F)
  (h5 : angle APB = 120 ∧ angle BPC = 120 ∧ angle CPA = 120)
  (h6 : length PD = 1/4) (h7 : length PE = 1/5) (h8 : length PF = 1/7) :
  length AP + length BP + length CP = 19/12 := 
  sorry

end triangle_lengths_l598_598469


namespace arithmetic_sequence_general_term_and_sum_l598_598388

theorem arithmetic_sequence_general_term_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 2 = 2) →
  (a 4 = 4) →
  (∀ n, a n = n) →
  (∀ n, b n = 2 ^ (a n)) →
  (∀ n, S n = 2 * (2 ^ n - 1)) :=
by
  intros h1 h2 h3 h4
  -- Proof part is skipped
  sorry

end arithmetic_sequence_general_term_and_sum_l598_598388


namespace distance_from_P_to_left_focus_of_ellipse_is_4_l598_598320

-- Define the ellipse \( C_1 \)
def ellipse (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 5 = 1

-- Define the hyperbola \( C_2 \)
def hyperbola (x y : ℝ) : Prop := (x^2) - (y^2) / 3 = 1

-- Define the point P as point of intersection in the first quadrant
def point_P (x y : ℝ) : Prop := ellipse x y ∧ hyperbola x y ∧ x > 0 ∧ y > 0

-- Define the distance from P to the left focus of the ellipse
def distance_to_left_focus_of_ellipse (x y : ℝ) : ℝ := 
  let a := 3 in
  let c := 1 in
  let b := √5 in
  let f₁ := (-c, 0) in
  let distance := √((x + c)^2 + y^2) in
  distance

theorem distance_from_P_to_left_focus_of_ellipse_is_4 :
  ∃ x y : ℝ, point_P x y → distance_to_left_focus_of_ellipse x y = 4 := 
sorry

end distance_from_P_to_left_focus_of_ellipse_is_4_l598_598320


namespace range_of_b_l598_598367

theorem range_of_b (b : ℤ) : 
  (∃ x1 x2 : ℤ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ x1 - b > 0 ∧ x2 - b > 0 ∧ (∀ x : ℤ, x < 0 ∧ x - b > 0 → (x = x1 ∨ x = x2))) ↔ (-3 ≤ b ∧ b < -2) :=
by sorry

end range_of_b_l598_598367


namespace beetles_initial_positions_l598_598624

noncomputable def min_distance_le (polygon : Polygon) : Prop :=
  ∀ (posA posB posC posD : polygon.vertices),
    min_distance_fly posC posD ≤ min_distance_beetle posA posB

theorem beetles_initial_positions (polygon : Polygon) :
  ∃ (posA posB : polygon.vertices),
    min_distance_le polygon :=
begin
  -- proof goes here
  sorry
end

end beetles_initial_positions_l598_598624


namespace limit_expression_l598_598635

theorem limit_expression :
  (tendsto (λx : ℝ, (2 - exp (x^2)) ^ (1 / (1 - cos (π * x)))) (nhds 0) (nhds (exp (-2 / (π^2))))) :=
sorry

end limit_expression_l598_598635


namespace rational_solution_exists_l598_598642

theorem rational_solution_exists (x y z : ℚ) (t : ℝ) (ht : t = real.cbrt 2) (hne : x + y * t + z * t^2 ≠ 0) :
  ∃ (u v w : ℚ), (x + y * t + z * t^2) * (u + v * t + w * t^2) = 1 :=
begin
  sorry
end

end rational_solution_exists_l598_598642


namespace max_volume_prism_l598_598379

theorem max_volume_prism (a b h : ℝ) (θ : ℝ) 
    (H1 : a > 0) (H2 : b > 0) (H3 : h > 0) (H4 : 0 < θ ∧ θ < π)
    (sum_areas_eq : a * h + b * h + 1 / 2 * a * b * Real.sin θ = 24) : 
    let V := 1 / 2 * a * b * h * Real.sin θ in V ≤ 16 :=
by
  sorry

end max_volume_prism_l598_598379


namespace abs_neg_three_halves_l598_598478

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end abs_neg_three_halves_l598_598478


namespace john_tour_days_l598_598798

noncomputable def numberOfDaysInTourProgram (d e : ℕ) : Prop :=
  d * e = 800 ∧ (d + 7) * (e - 5) = 800

theorem john_tour_days :
  ∃ (d e : ℕ), numberOfDaysInTourProgram d e ∧ d = 28 :=
by
  sorry

end john_tour_days_l598_598798


namespace simplify_exponents_l598_598572

theorem simplify_exponents : (10^0.5) * (10^0.3) * (10^0.2) * (10^0.1) * (10^0.9) = 100 := 
by 
  sorry

end simplify_exponents_l598_598572


namespace insurance_covers_80_percent_of_medical_bills_l598_598528

theorem insurance_covers_80_percent_of_medical_bills 
    (vaccine_cost : ℕ) (num_vaccines : ℕ) (doctor_visit_cost trip_cost : ℕ) (amount_tom_pays : ℕ) 
    (total_cost := num_vaccines * vaccine_cost + doctor_visit_cost) 
    (total_trip_cost := trip_cost + total_cost)
    (insurance_coverage := total_trip_cost - amount_tom_pays)
    (percent_covered := (insurance_coverage * 100) / total_cost) :
    vaccine_cost = 45 → num_vaccines = 10 → doctor_visit_cost = 250 → trip_cost = 1200 → amount_tom_pays = 1340 →
    percent_covered = 80 := 
by
  sorry

end insurance_covers_80_percent_of_medical_bills_l598_598528


namespace patricias_hair_length_after_donation_l598_598846

def patricias_final_hair_length 
  (initial_length : ℕ) 
  (growth : ℕ) 
  (donation : ℕ) : ℕ :=
  initial_length + growth - donation

theorem patricias_hair_length_after_donation 
  (h_initial : ℕ := 14) 
  (h_growth : ℕ := 21) 
  (h_donation : ℕ := 23) :
  patricias_final_hhair_length h_initial h_growth h_donation = 12 :=
by
  unfold patricias_final_hair_length
  rw [Nat.add_sub_assoc]
  norm_num
  sorry

end patricias_hair_length_after_donation_l598_598846


namespace root_magnitude_bound_l598_598016

open Complex

theorem root_magnitude_bound (n : ℕ) (hn : n ≥ 1) (a : Fin n → ℂ) 
  (z : Fin n → ℂ) (hz : ∀ i, eval₂ id z i = 0) :
  let A := Finset.max' (Finset.univ.image (λ i, abs (a i))) (by apply Finset.nonempty_univ) in
  ∀ j : Fin n, abs (z j) ≤ 1 + A :=
sorry

end root_magnitude_bound_l598_598016


namespace cannot_determine_right_triangle_l598_598716

/-- Proof that the condition \(a^2 = 5\), \(b^2 = 12\), \(c^2 = 13\) cannot determine that \(\triangle ABC\) is a right triangle. -/
theorem cannot_determine_right_triangle (a b c : ℝ) (ha : a^2 = 5) (hb : b^2 = 12) (hc : c^2 = 13) : 
  ¬(a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := 
by
  sorry

end cannot_determine_right_triangle_l598_598716


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l598_598869

-- Problem 1
theorem solve_quadratic_1 (x : ℝ) : (x - 1) ^ 2 - 4 = 0 ↔ (x = -1 ∨ x = 3) :=
by
  sorry

-- Problem 2
theorem solve_quadratic_2 (x : ℝ) : (2 * x - 1) * (x + 3) = 4 ↔ (x = -7 / 2 ∨ x = 1) :=
by
  sorry

-- Problem 3
theorem solve_quadratic_3 (x : ℝ) : 2 * x ^ 2 - 5 * x + 2 = 0 ↔ (x = 2 ∨ x = 1 / 2) :=
by
  sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l598_598869


namespace company_employee_count_l598_598173

theorem company_employee_count : 
  ∃ E : ℕ, 
    (0.6 * E) = (0.2 * E + 40) ∧
    E = 100 :=
by
  sorry

end company_employee_count_l598_598173


namespace front_view_length_l598_598179

theorem front_view_length
  (a b c : ℝ)
  (hab : a = 5)
  (hbc : b = sqrt 34)
  (hc : c = 5 * sqrt 2) :
  (a^2 + b^2 + (sqrt 41)^2 = c^2) :=
by
  sorry

end front_view_length_l598_598179


namespace least_five_digit_integer_l598_598119

theorem least_five_digit_integer (n : ℕ) :
  (∃ n, 10000 ≤ n ∧ n < 100000 ∧ (∀ d, d ∈ (List.digits n) → d ≠ 0 ∧ d ≠ 5) ∧
  (∀ d, d ∈ (List.digits n) → (d ≠ 0 ∧ d ≠ 5 → n % d = 0)) ∧ List.distinct (List.digits n)
  ∧ (∀ m, 10000 ≤ m ∧ m < 100000 ∧ (∀ d, d ∈ (List.digits m) → d ≠ 0 ∧ d ≠ 5) ∧
  (∀ d, d ∈ (List.digits m) → (d ≠ 0 ∧ d ≠ 5 → m % d = 0)) ∧ List.distinct (List.digits m) → n ≤ m)) → n = 12376 :=
begin
  sorry,
end

end least_five_digit_integer_l598_598119


namespace sine_inequality_l598_598663

theorem sine_inequality (x y : ℝ) (hx : -π/2 ≤ x ∧ x ≤ π/2) (hy : 0 ≤ y ∧ y ≤ π/2) :
  sin (x + y) ≤ sin x + sin y :=
sorry

end sine_inequality_l598_598663


namespace interest_related_to_gender_l598_598949

variable (a b c d n : ℕ)
variable (K_squared critical_value : ℚ)

-- Given values
def a : ℕ := 50
def b : ℕ := 10
def c : ℕ := 30
def d : ℕ := 20
def n : ℕ := 110

-- Chi-squared formula
def K_squared : ℚ := (n * ((a * d - b * c) ^ 2)) / (↑((a + b) * (c + d) * (a + c) * (b + d)))

-- Critical value at 1% significance level
def critical_value : ℚ := 6.635

-- Theorem to prove that K_squared > critical_value
theorem interest_related_to_gender : K_squared > critical_value := by
  sorry

end interest_related_to_gender_l598_598949


namespace find_x_in_terms_of_a_b_l598_598424

variable (a b x : ℝ)
variable (ha : a > 0) (hb : b > 0) (hx : x > 0) (r : ℝ)
variable (h1 : r = (4 * a)^(3 * b))
variable (h2 : r = a ^ b * x ^ b)

theorem find_x_in_terms_of_a_b 
  (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h1 : (4 * a)^(3 * b) = r)
  (h2 : r = a^b * x^b) :
  x = 64 * a^2 :=
by
  sorry

end find_x_in_terms_of_a_b_l598_598424


namespace minimum_raft_weight_l598_598198

-- Define the weights of the animals.
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal.
def num_mice : ℕ := 5
def num_moles : ℕ := 3
def num_hamsters : ℕ := 4

-- The function that represents the minimum weight capacity required for the raft.
def minimum_raft_capacity : ℕ := 140

-- Prove that the minimum raft capacity to transport all animals is 140 grams.
theorem minimum_raft_weight :
  (∀ (total_weight : ℕ), 
    total_weight = (num_mice * weight_mouse) + (num_moles * weight_mole) + (num_hamsters * weight_hamster) →
    (exists (raft_capacity : ℕ), 
      raft_capacity = minimum_raft_capacity ∧
      raft_capacity >= 2 * weight_mouse)) :=
begin
  -- Initial state setup and logical structure.
  intros total_weight total_weight_eq,
  use minimum_raft_capacity,
  split,
  { refl },
  { have h1: 2 * weight_mouse = 140,
    { norm_num },
    rw h1,
    exact le_refl _,
  }
end

end minimum_raft_weight_l598_598198


namespace min_distance_PQ_l598_598441

-- Define the vertices of the tetrahedron
def A : ℝ^3 := ⟨0, 0, 0⟩
def B : ℝ^3 := ⟨1, 0, 0⟩
def C : ℝ^3 := ⟨1/2, (Real.sqrt 3)/2, 0⟩
def D : ℝ^3 := ⟨1/2, (Real.sqrt 3)/6, (Real.sqrt 6)/3⟩

-- Define the points P and Q on edges AB and CD respectively
def P (t : ℝ) (h : 0 ≤ t ∧ t ≤ 1) : ℝ^3 := t • A + (1 - t) • B
def Q (s : ℝ) (h : 0 ≤ s ∧ s ≤ 1) : ℝ^3 := s • C + (1 - s) • D

-- Define the distance function between P and Q
def distance (P Q : ℝ^3) : ℝ := Real.sqrt ((P - Q).sum λ x, x^2)

-- Calculate the minimum distance PQ
theorem min_distance_PQ : ∀ (t s : ℝ) (ht : 0 ≤ t ∧ t ≤ 1) (hs : 0 ≤ s ∧ s ≤ 1), 
  distance (P t ht) (Q s hs) = (Real.sqrt 2) / 2 := sorry

end min_distance_PQ_l598_598441


namespace find_T_l598_598519

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
variables (T : V → V) (v w : V) (a b : ℝ)
variables (v1 v2 v3: V) (t1 t2 t3: V)

-- Conditions
axiom T_add (a b : ℝ) (v w : V) : T (a • v + b • w) = a • T v + b • T w
axiom T_cross (v w : V) : T (v ⬝ w) = T v ⬝ T w
axiom T_v1 : T (⟨5, 5, 1⟩: V) = (⟨2, -1, 7⟩: V)
axiom T_v2 : T (⟨-5, 1, 5⟩: V) = (⟨2, 7, -1⟩: V)

-- Theorem to be proved
theorem find_T : T (⟨1, 6, 10⟩: V) = (⟨8/3, 4/3, 10/3⟩: V) :=
sorry

end find_T_l598_598519


namespace total_area_rectangle_l598_598183

theorem total_area_rectangle (BF CF : ℕ) (A1 A2 x : ℕ) (h1 : BF = 3 * CF) (h2 : A1 = 3 * A2) (h3 : 2 * x = 96) (h4 : 48 = x) (h5 : A1 = 3 * 48) (h6 : A2 = 48) : A1 + A2 = 192 :=
  by sorry

end total_area_rectangle_l598_598183


namespace width_is_70_l598_598887

noncomputable def width_of_field (w : ℝ) : Prop :=
  let l := (7 / 5) * w in
  2 * l + 2 * w = 336

theorem width_is_70 : ∃ w : ℝ, width_of_field w ∧ w = 70 :=
by
  use 70
  unfold width_of_field
  sorry

end width_is_70_l598_598887


namespace raft_min_capacity_l598_598189

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end raft_min_capacity_l598_598189


namespace length_CD_l598_598514

noncomputable def volume_cylinder_with_hemispheres (r h : ℝ) : ℝ :=
  let π := Real.pi
  let hemisphere_volume := (2 * (1/2) * (4/3) * π * r ^ 3)
  let cylinder_volume := π * r^2 * h
  in hemisphere_volume + cylinder_volume

theorem length_CD (r h : ℝ) (volume : ℝ) : 
  r = 4 → volume = 448 * Real.pi → volume_cylinder_with_hemispheres r h = volume → h = 68 / 3 :=
by
  intros
  sorry

end length_CD_l598_598514


namespace sum_of_even_factors_of_630_l598_598126

noncomputable def sum_of_positive_even_factors (n : Nat) : Nat :=
  ∑ i in (Finset.filter (λ d, d % 2 = 0) (Finset.divisors n)), i

theorem sum_of_even_factors_of_630 :
  (sum_of_positive_even_factors 630) = 1248 := by
  sorry

end sum_of_even_factors_of_630_l598_598126


namespace range_of_function_l598_598924

noncomputable def func (x : ℝ) : ℝ :=
  3 - Real.sin x - 2 * (Real.cos x) ^ 2

theorem range_of_function :
  (range (func) ∩ Set.Icc (Real.sin (Real.pi / 6)) (Real.sin (7 * Real.pi / 6))) = Set.Icc (7 / 8 : ℝ) 2 :=
by
  sorry

end range_of_function_l598_598924


namespace simplify_expression_l598_598161

theorem simplify_expression :
  (Real.sin (Real.pi / 6) + (1 / 2) - 2007^0 + abs (-2) = 2) :=
by
  sorry

end simplify_expression_l598_598161


namespace find_k_l598_598708

open Real

variables (k : ℝ) (P A B : ℝ × ℝ)
variables (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop)

def is_point_on_line (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop := l P
def is_tangent_to_circle (P A : ℝ × ℝ) (C : ℝ × ℝ → Prop) : Prop := 
  let radius : ℝ := 1
  let center : (ℝ × ℝ) := (0, 1)
  let PA_dist := dist P A
  PA_dist = dist center A ∧ dist center A = radius ∧ line_through P A C

theorem find_k 
  (h1 : k > 0)
  (h2 : is_point_on_line P (λ (p : ℝ × ℝ), k * p.1 + p.2 + 4 = 0))
  (h3 : is_tangent_to_circle P A C ∧ is_tangent_to_circle P B C)
  (h4 : ∃ PA AC PB BC, 
    PA ≠ PB ∧ CA = 1 ∧ 
    (min (PA * 1 + PB * CA) = 2))
  : k = 2 :=
sorry

end find_k_l598_598708


namespace twist_45_eq_530_twist_between_1978_2010_l598_598448

def is_twist (n : ℕ) : Prop :=
  (∃ k : ℕ, 2 + (3 * k) = n) ∨ (∃ k : ℕ, (k * (k + 1) / 2 + 1) = n)

def exp_twist_pos (n : ℕ) : ℕ :=
  if odd n then ((n + 1) / 2) ^ 2 + 1
  else (1 + n / 2) * (n / 2) + 1

theorem twist_45_eq_530 : exp_twist_pos 45 = 530 := sorry

theorem twist_between_1978_2010 : ∃ (n : ℕ), 1978 ≤ n ∧ n ≤ 2010 ∧ is_twist n ∧ n = 1981 := sorry

end twist_45_eq_530_twist_between_1978_2010_l598_598448


namespace find_f_l598_598432

def is_quadratic_poly (f : ℝ → ℝ → ℝ) : Prop := 
  ∃ a b c d e : ℝ, ∀ x y, f(x, y) = a * x^2 + b * y^2 + c * x * y + d * x + e * y

axiom f : ℝ → ℝ → ℝ

axiom f_is_quadratic : is_quadratic_poly f

axiom condition1 : f 1 2 = 2

axiom condition2 : ∀ x y, y * f x (f x y) = x * f (f x y) y ∧ y * f x (f x y) = (f x y)^2

theorem find_f : f = λ x y, x * y := 
by
  sorry

end find_f_l598_598432


namespace sufficient_but_not_necessary_condition_l598_598588

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x - 2

theorem sufficient_but_not_necessary_condition 
  (a : ℝ) 
  (h : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) : 
  a ≤ 0 :=
sorry

end sufficient_but_not_necessary_condition_l598_598588


namespace num_real_a_with_int_roots_l598_598678

theorem num_real_a_with_int_roots :
  (∃ n : ℕ, n = 15 ∧ ∀ a : ℝ, (∃ r s : ℤ, (r + s = -a) ∧ (r * s = 12 * a) → true)) :=
sorry

end num_real_a_with_int_roots_l598_598678


namespace find_f_two_sevenths_l598_598505

noncomputable def f (x : ℝ) : ℝ := 
  if x = 0 then 0 else
    if x = 1 then 1 else 
      if x = 1/3 then 1/2 else 
        if x = 2/3 then 1/2 else
          if x = 3/7 then 1/2 else
            if x = 1/7 then 1/4 else
              if x = 6/7 then 3/4 else 0 -- Dummy value, real function definition skipped for problem statement

theorem find_f_two_sevenths (x : ℝ) :
  (f(0) = 0) ∧ 
  (∀ x y : ℝ, 0 ≤ x → x < y → y ≤ 1 → f(x) ≤ f(y)) ∧ 
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → f(1 - x) = 1 - f(x)) ∧ 
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → f(x / 3) = f(x) / 2) →
  f(2 / 7) = 3 / 8 :=
by
  intros,
  sorry

end find_f_two_sevenths_l598_598505


namespace complex_square_l598_598634

theorem complex_square (i : ℂ) (h : i^2 = -1) : (1 - i)^2 = -2 * i :=
by
  sorry

end complex_square_l598_598634


namespace part1_f1_part1_f_1div4_part2_range_x_l598_598819

variable {α : Type*} [LinearOrder α]

-- Assume we have a decreasing function f defined on (0, +∞)
variable (f : α → α)
variable (h_decreasing : ∀ x y : α, 0 < x ∧ 0 < y ∧ y < x → f(x) < f(y))
variable (h_f2 : f(2) = 1)
variable (h_property : ∀ x y : α, 0 < x ∧ 0 < y → f (x / y) = f x - f y)

-- We want to prove the given statements

-- Part (1): Prove f(1) = 0
theorem part1_f1 : f(1) = 0 := sorry

-- Part (1): Prove f(1/4) = -2
theorem part1_f_1div4 : f(1 / 4) = -2 := sorry

-- Part (2): Prove range of x where f(3^x) + f(3^(x - 2)) < 3 is x > log_3 4
theorem part2_range_x (x : α) (h_ineq : f (3 ^ x) + f (3 ^ (x - 2)) < 3) : x > Real.log 4 / Real.log 3 := sorry

end part1_f1_part1_f_1div4_part2_range_x_l598_598819


namespace g_at_91_l598_598245

def g : ℤ → ℤ
| n :=
  if n ≥ 2000 then n - 4
  else g (g (n + 7))

theorem g_at_91 : g 91 = 1997 := by
  sorry

end g_at_91_l598_598245


namespace employees_cycle_l598_598601

theorem employees_cycle (total_employees : ℕ) (drivers_percentage walkers_percentage cyclers_percentage: ℕ) (walk_cycle_ratio_walk walk_cycle_ratio_cycle: ℕ)
    (h_total : total_employees = 500)
    (h_drivers_perc : drivers_percentage = 35)
    (h_transit_perc : walkers_percentage = 25)
    (h_walkers_cyclers_ratio_walk : walk_cycle_ratio_walk = 3)
    (h_walkers_cyclers_ratio_cycle : walk_cycle_ratio_cycle = 7) :
    cyclers_percentage = 140 :=
by
  sorry

end employees_cycle_l598_598601


namespace almond_butter_cookie_cost_diff_l598_598840

-- Definitions based on conditions
def pb_cost : ℝ := 3        -- Cost of a jar of peanut butter
def ab_cost : ℝ := 3 * pb_cost                     -- Cost of a jar of almond butter
def pb_cost_per_batch : ℝ := pb_cost / 2           -- Cost of peanut butter per batch
def ab_cost_per_batch : ℝ := ab_cost / 2           -- Cost of almond butter per batch
def sugar_cost_diff : ℝ := 0.5                     -- Additional cost of organic sugar per cup
def total_cost_diff : ℝ := ab_cost_per_batch - pb_cost_per_batch + sugar_cost_diff

-- Theorem to prove that the additional cost per batch is $3.50
theorem almond_butter_cookie_cost_diff : total_cost_diff = 3.50 :=
by
  sorry

end almond_butter_cookie_cost_diff_l598_598840


namespace total_animals_seen_l598_598259

theorem total_animals_seen (lions_sat : ℕ) (elephants_sat : ℕ) 
                           (buffaloes_sun : ℕ) (leopards_sun : ℕ)
                           (rhinos_mon : ℕ) (warthogs_mon : ℕ) 
                           (h_sat : lions_sat = 3 ∧ elephants_sat = 2)
                           (h_sun : buffaloes_sun = 2 ∧ leopards_sun = 5)
                           (h_mon : rhinos_mon = 5 ∧ warthogs_mon = 3) :
  lions_sat + elephants_sat + buffaloes_sun + leopards_sun + rhinos_mon + warthogs_mon = 20 := by
  sorry

end total_animals_seen_l598_598259


namespace mass_of_curve_l598_598999

open Real

theorem mass_of_curve :
  let y := λ x : ℝ, x^3 / 3
  let rho := λ x : ℝ, 1 + x^2
  let integrand := λ x : ℝ, (1 + x^2) * sqrt (1 + (x^2)^2)
  ∫ x in 0..0.1, integrand x = 0.099985655 := 
by
  sorry

end mass_of_curve_l598_598999


namespace inequality_holds_for_all_xyz_in_unit_interval_l598_598864

theorem inequality_holds_for_all_xyz_in_unit_interval :
  ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) → 
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z)) :=
by
  intros x y z hx hy hz
  sorry

end inequality_holds_for_all_xyz_in_unit_interval_l598_598864


namespace part1_part2_1_part2_2_l598_598064

theorem part1 (n : ℚ) :
  (2 / 2 + n / 5 = (2 + n) / 7) → n = -25 / 2 :=
by sorry

theorem part2_1 (m n : ℚ) :
  (m / 2 + n / 5 = (m + n) / 7) → m = -4 / 25 * n :=
by sorry

theorem part2_2 (m n: ℚ) :
  (m = -4 / 25 * n) → (25 * m + n = 6) → (m = 8 / 25 ∧ n = -2) :=
by sorry

end part1_part2_1_part2_2_l598_598064


namespace find_angle_C_max_area_of_triangle_l598_598302

variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides of the triangle
variables {R : ℝ} -- Circumradius

-- Condition 1: \(2\sqrt{2}(\sin^2 A - \sin^2 C) = (a - b) \sin B\)
def condition1 : Prop := 2 * real.sqrt 2 * (real.sin A ^ 2 - real.sin C ^ 2) = (a - b) * real.sin B

-- Condition 2: \(R = \sqrt{2}\)
def condition2 : Prop := R = real.sqrt 2

-- Question 1
theorem find_angle_C (h1 : condition1) (h2 : condition2) : C = 60 * (real.pi / 180) :=
sorry

-- Question 2
theorem max_area_of_triangle (h1 : condition1) (h2 : condition2) (h3 : C = 60 * (real.pi / 180)) : 
  ∃ S, S = (3 * real.sqrt 3) / 2 :=
sorry

end find_angle_C_max_area_of_triangle_l598_598302


namespace magnitude_n_eq_sqrt_5_l598_598339

variable (x : ℝ)

def m := (1 : ℝ, 2 : ℝ)
def n := (x, 1 : ℝ)

theorem magnitude_n_eq_sqrt_5 (h : 1 * x + 2 * 1 = 0) : ∥n x∥ = real.sqrt 5 := by
  sorry

end magnitude_n_eq_sqrt_5_l598_598339


namespace find_width_of_lawn_l598_598613

-- Definitions based on the conditions
def length_lawn : ℝ := 80
def road_width : ℝ := 10
def cost_per_sqm : ℝ := 4
def total_cost : ℝ := 5200

-- Definition of the total area of the roads
def total_area_roads (w : ℝ) : ℝ := (10 * w) + (10 * length_lawn) - (road_width * road_width)

-- The total area based on cost
def total_area_from_cost : ℝ := total_cost / cost_per_sqm

-- The main theorem statement
theorem find_width_of_lawn (w : ℝ) (h : total_area_roads w = total_area_from_cost) : w = 60 :=
by
  sorry

end find_width_of_lawn_l598_598613


namespace sum_of_altitudes_less_than_sum_of_sides_l598_598066

theorem sum_of_altitudes_less_than_sum_of_sides 
  (a b c h_a h_b h_c K : ℝ) 
  (triangle_area : K = (1/2) * a * h_a)
  (h_a_def : h_a = 2 * K / a) 
  (h_b_def : h_b = 2 * K / b)
  (h_c_def : h_c = 2 * K / c) : 
  h_a + h_b + h_c < a + b + c := by
  sorry

end sum_of_altitudes_less_than_sum_of_sides_l598_598066


namespace transform_grid_l598_598383

theorem transform_grid (a b : Fin 24 → Fin 24 → ℤ)
  (h_initial : ∀ i j : Fin 24, a i j = 1 ∨ a i j = -1)
  (h_target : ∀ i j : Fin 24, b i j = 1 ∨ b i j = -1) :
  ∃ moves : List (Fin 24 × Fin 24),
  ∀ n m : Fin 24, (apply_moves a moves) n m = b n m :=
sorry

def apply_moves (g : Fin 24 → Fin 24 → ℤ) (moves : List (Fin 24 × Fin 24)) : Fin 24 → Fin 24 → ℤ :=
  moves.foldl (λ grid (i_j : Fin 24 × Fin 24), flip_signs grid i_j.1 i_j.2) g

def flip_signs (g : Fin 24 → Fin 24 → ℤ) (i j : Fin 24) : Fin 24 → Fin 24 → ℤ :=
  λ n m, if n = i ∨ m = j then -g n m else g n m

end transform_grid_l598_598383


namespace andrea_avg_km_per_day_l598_598225

theorem andrea_avg_km_per_day
  (total_distance : ℕ := 168)
  (total_days : ℕ := 6)
  (completed_fraction : ℚ := 3/7)
  (completed_days : ℕ := 3) :
  (total_distance * (1 - completed_fraction)) / (total_days - completed_days) = 32 := 
sorry

end andrea_avg_km_per_day_l598_598225


namespace bug_can_reach_96_points_l598_598594

def point := (ℝ × ℝ)
def A : point := (-4, 3)
def B : point := (4, -3)
def forbidden_rectangle : set point := {p | (p.1 = -1 ∨ p.1 = 1) ∧ (p.2 ≥ -1 ∧ p.2 ≤ 1)}

noncomputable def path_length (p1 p2 : point) : ℝ :=
  (abs (p2.1 - p1.1)) + (abs (p2.2 - p1.2))

def is_valid_point (p : point) : Prop :=
  ¬(p ∈ forbidden_rectangle) ∧ (path_length A p + path_length p B ≤ 24)

def valid_integer_points : finset (ℤ × ℤ) :=
  {p | p.1 ∈ (finset.range 9).map (λ x, x - 4) ∧ p.2 ∈ (finset.range 11).map (λ y, y - 5) ∧ is_valid_point (p.1, p.2)}

theorem bug_can_reach_96_points :
  valid_integer_points.card = 96 := sorry

end bug_can_reach_96_points_l598_598594


namespace club_must_have_ten_members_l598_598172

-- Definitions for the conditions
def club_has_five_committees : Prop := True
def each_member_joins_exactly_three_different_committees : Prop := True
def each_trio_of_committees_has_exactly_one_member_in_common : Prop := True

-- The theorem stating the conclusion
theorem club_must_have_ten_members 
  (h1 : club_has_five_committees)
  (h2 : each_member_joins_exactly_three_different_committees)
  (h3 : each_trio_of_committees_has_exactly_one_member_in_common) : 
  ∃ (n : ℕ), n = 10 :=
begin
  sorry
end

end club_must_have_ten_members_l598_598172


namespace student_correct_sums_l598_598617

theorem student_correct_sums (x wrong total : ℕ) (h1 : wrong = 2 * x) (h2 : total = x + wrong) (h3 : total = 54) : x = 18 :=
by
  sorry

end student_correct_sums_l598_598617


namespace vans_needed_l598_598648

-- Definitions of conditions
def students : Nat := 2
def adults : Nat := 6
def capacity_per_van : Nat := 4

-- Main theorem to prove
theorem vans_needed : (students + adults) / capacity_per_van = 2 := by
  sorry

end vans_needed_l598_598648


namespace find_area_of_triangle_l598_598882

variable {a b c A C : ℝ}

noncomputable def area (a b c : ℝ) : ℝ :=
  sqrt (1/4 * (a^2 * c^2 - ( (a^2 + c^2 - b^2) / 2 ) ^ 2))

theorem find_area_of_triangle 
  (h1 : a^2 * sin C = 4 * sin A)
  (h2 : (a + c)^2 = 12 + b^2)
  : area a b c = sqrt 3 :=
sorry

end find_area_of_triangle_l598_598882


namespace card_trick_part_a_l598_598551

def deck : Type := Fin 52
def permutations : Type := Fin 120

theorem card_trick_part_a (n : ℕ) (cards : Finset deck) (perm_num : permutations) :
  cards.cardinality = 5 →
  ∃ (face_down : deck),  ∀ (a b c d : deck),
    a ∈ cards →
    b ∈ cards →
    c ∈ cards →
    d ∈ cards → 
    a ≠ b → b ≠ c → c ≠ d → d ≠ a → 
    (∃ p : permutations, true)  → -- representing the prearranged permutation system 
    true := -- representing the decoding process
sorry

end card_trick_part_a_l598_598551


namespace prob_x_plus_y_lt_4_in_square_l598_598181

theorem prob_x_plus_y_lt_4_in_square :
  let square_area := 9 in
  let triangle_area := 2 in
  let desired_area := square_area - triangle_area in
  let probability := (desired_area : ℝ) / square_area in
  probability = (7 : ℝ) / 9 := by
  let square_area := 9
  let triangle_area := 2
  let desired_area := square_area - triangle_area
  let probability := (desired_area : ℝ) / square_area
  show probability = (7 : ℝ) / 9
  sorry

end prob_x_plus_y_lt_4_in_square_l598_598181


namespace count_perfect_squares_ending_4_5_6_l598_598348

theorem count_perfect_squares_ending_4_5_6 : 
  ∃ n, n = 36 ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 70 ∧ (let d := k % 10 in d = 2 ∨ d = 8 ∨ d = 5 ∨ d = 4 ∨ d = 6) → k^2 < 5000) := 
sorry

end count_perfect_squares_ending_4_5_6_l598_598348


namespace sum_of_first_cards_l598_598059

variables (a b c d : ℕ)

theorem sum_of_first_cards (a b c d : ℕ) : 
  ∃ x, x = b * (c + 1) + d - a :=
by
  sorry

end sum_of_first_cards_l598_598059


namespace monotonicity_intervals_condition_for_fx_ge_one_l598_598324

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * Real.log x

theorem monotonicity_intervals (x : ℝ) (hx : 0 < x) :
  (∀ a : ℝ, f x 3 = x - 3 * Real.log x → ((f' x > 0 ↔ 3 < x) ∧ (f' x < 0 ↔ 0 < x ∧ x < 3))) :=
by sorry

theorem condition_for_fx_ge_one (a : ℝ) :
  (∀ x : ℝ, 0 < x → f x a ≥ 1 ↔ a = 1) :=
by sorry

end monotonicity_intervals_condition_for_fx_ge_one_l598_598324


namespace compute_fraction_sum_l598_598074

variables {a b c : ℝ}

def condition1 : Prop :=
  (ac / (a + b) + ba / (b + c) + cb / (c + a) = -7)

def condition2 : Prop :=
  (bc / (a + b) + ca / (b + c) + ab / (c + a) = 8)

theorem compute_fraction_sum (h1 : condition1) (h2 : condition2) :
  (b / (a + b) + c / (b + c) + a / (c + a) = 9) :=
sorry

end compute_fraction_sum_l598_598074


namespace percentage_chemical_a_in_x_l598_598616

theorem percentage_chemical_a_in_x (A : ℝ) 
  (hx_a : chemical_percentage x a = A)
  (hx_b : chemical_percentage x b = 90)
  (hy_a : chemical_percentage y a = 20)
  (hy_b: chemical_percentage y b = 80)
  (m_a : chemical_percentage (0.8 * x + 0.2 * y) a = 12)
  (mix_x : mixture_percentage x (0.8 * x + 0.2 * y) = 80) :
  A = 10 := 
sorry

end percentage_chemical_a_in_x_l598_598616


namespace total_houses_l598_598374

theorem total_houses (houses_one_side : ℕ) (houses_other_side : ℕ) (h1 : houses_one_side = 40) (h2 : houses_other_side = 3 * houses_one_side) : houses_one_side + houses_other_side = 160 :=
by sorry

end total_houses_l598_598374


namespace euclid_can_draw_circle_centered_at_A_through_B_l598_598260

-- Axiomatizing the cyclos abilities
axiom draw_circle_three_points : ∀ (A B C : Point), ¬Collinear A B C → exists_circle_through A B C
axiom draw_circle_diameter : ∀ (A B : Point), exists_circle_diameter A B
axiom mark_intersection : ∀ (C1 C2 : Circle), ∃ (I : Point), is_intersection C1 C2 I
axiom mark_point_on_circle : ∀ (C : Circle) (A : Point), is_on_circle C A → ∃ (B : Point), is_on_circle C B

-- Given proposition
theorem euclid_can_draw_circle_centered_at_A_through_B (A B : Point) :
  ∃ (C : Circle), center C = A ∧ passes_through C B :=
sorry

end euclid_can_draw_circle_centered_at_A_through_B_l598_598260


namespace isosceles_triangle_condition_l598_598391

-- Define the coordinates of points A, B, and the condition for point P
def A : Point := ⟨-2, 0⟩
def B : Point := ⟨2, 0⟩

noncomputable def distance (P Q : Point) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def isosceles_triangle (P₁ P₂ P₃ : Point) : Prop :=
  (distance P₁ P₂ = distance P₁ P₃ ∨ distance P₁ P₂ = distance P₂ P₃ ∨ distance P₁ P₃ = distance P₂ P₃)

-- The main statement to prove
theorem isosceles_triangle_condition :
  let points_P := {P : Point | distance P A = sqrt 3 * distance P B} in
  (finset.filter (λ P, isosceles_triangle P A B)
                  (finset.univ.filter (λ P : Point, P ∈ points_P))).card = 4 :=
sorry

end isosceles_triangle_condition_l598_598391


namespace ratio_areas_top_to_side_l598_598244

theorem ratio_areas_top_to_side (l w h : ℝ) (h1 : w * h = (1 / 2) * l * w) (h2 : l * w * h = 192) (h3 : l * h ≈ 32) :
  l * w / (l * h) = 3 / 2 :=
sorry

end ratio_areas_top_to_side_l598_598244


namespace cylinder_height_l598_598081

variable (r h : ℝ) (SA : ℝ)

theorem cylinder_height (h : ℝ) (r : ℝ) (SA : ℝ) (h_eq : h = 2) (r_eq : r = 3) (SA_eq : SA = 30 * Real.pi) :
  SA = 2 * Real.pi * r ^ 2 + 2 * Real.pi * r * h → h = 2 :=
by
  intros
  sorry

end cylinder_height_l598_598081


namespace ceil_sqrt_fraction_eq_neg2_l598_598658

theorem ceil_sqrt_fraction_eq_neg2 :
  (Int.ceil (-Real.sqrt (36 / 9))) = -2 :=
by
  sorry

end ceil_sqrt_fraction_eq_neg2_l598_598658


namespace problem1_problem2_l598_598689

def f (x : ℝ) := |x - 1| + |x + 2|

def T (a : ℝ) := -Real.sqrt 3 < a ∧ a < Real.sqrt 3

theorem problem1 (a : ℝ) : (∀ x : ℝ, f x > a^2) ↔ T a :=
by
  sorry

theorem problem2 (m n : ℝ) (h1 : T m) (h2 : T n) : Real.sqrt 3 * |m + n| < |m * n + 3| :=
by
  sorry

end problem1_problem2_l598_598689


namespace polynomials_with_same_roots_are_equal_l598_598415

theorem polynomials_with_same_roots_are_equal
  (P Q : ℝ[X]) 
  (hP_nonconst : ¬ is_constant P)
  (hQ_nonconst : ¬ is_constant Q)
  (hPQ_same_roots : ∀ x, P.is_root x ↔ Q.is_root x)
  (hP1Q1_same_roots : ∀ x, (P - C 1).is_root x ↔ (Q - C 1).is_root x) :
  P = Q := 
sorry

end polynomials_with_same_roots_are_equal_l598_598415


namespace triangle_vector_min_l598_598394

theorem triangle_vector_min (ABC : Triangle) (E : ABC.AC) (hAC : ∃ AE, 4 * AE = ABC.AC) (P : ABC.BE) 
    (hAP : ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ ABC.AP = m * ABC.AB + n * ABC.AC) : 
    (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ (1 / m + 1 / n = (1 / (1 / 3)) + 1 / (1 / 6))) → 
    (∥⟨m, n⟩∥ = (√5) / 6) :=
by
    sorry

end triangle_vector_min_l598_598394


namespace least_n_divisible_by_15_l598_598668

theorem least_n_divisible_by_15 {a : ℕ → ℕ} (hpos : ∀ i, 1 ≤ a i) :
  ∃ n, (∀ a, 15 ∣ (finset.range 15).prod a * finset.sum (finset.range 15) (λ i, (a i)^n)) ∧ 
  ∀ m, (∀ a, 15 ∣ (finset.range 15).prod a * finset.sum (finset.range 15) (λ i, (a i)^m)) → n ≤ m :=
begin
  sorry -- proof is not required
end

end least_n_divisible_by_15_l598_598668


namespace determine_country_l598_598844

structure Person (Country : Type) :=
  (honest : Country → Prop)
  (liar   : Country → Prop)

def Country := {A : Prop // ∀ (c : A), true} ∪ {Y : Prop // ∀ (c : Y), false}

theorem determine_country (person : Person Country) (c : Country) :
  (∀ (p : person), person.honest p) ∨ (∀ (p : person), person.liar p) →
  (c ∈ Country) →
  ∃ (question : String), (question = "Are you a local inhabitant of this country?") →
  True :=
by
  sorry

end determine_country_l598_598844


namespace eventually_non_multiples_of_5_l598_598972

def sequence_condition (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if a n % 5 = 0 then a n / 5 else Nat.floor (Real.sqrt 5 * (a n))

theorem eventually_non_multiples_of_5 (a : ℕ → ℕ) (h0 : 0 < a 0)
  (h1 : ∀ n, a (n + 1) = sequence_condition a n) :
  ∃ N, ∀ n, N ≤ n → a n % 5 ≠ 0 :=
by sorry

end eventually_non_multiples_of_5_l598_598972


namespace inequality_satisfied_by_zero_l598_598925

theorem inequality_satisfied_by_zero : ∀ x : ℕ, x = 0 → x + 1 < 2 := 
by
  intro x hx
  rw hx
  norm_num

end inequality_satisfied_by_zero_l598_598925


namespace find_x_l598_598826

noncomputable theory
open Classical

def f (x y : ℕ) : ℕ := (x - y)! % x

theorem find_x :
  (∀ (x y : ℕ), f x y = 0 ↔ x ∣ (x - y)!) → (∃ (x : ℕ), (∀ (y : ℕ), f x y = 0 → y ≤ 40) ∧ x = 41) :=
by
  sorry

end find_x_l598_598826


namespace sum_even_factors_630_l598_598131

theorem sum_even_factors_630 : 
  (∑ n in (finset.filter (λ n, even n) (divisors 630)), n) = 1248 := 
sorry

end sum_even_factors_630_l598_598131


namespace polynomial_divisible_by_five_l598_598822

open Polynomial

theorem polynomial_divisible_by_five
  (a b c d m : ℤ)
  (h1 : (a * m^3 + b * m^2 + c * m + d) % 5 = 0)
  (h2 : d % 5 ≠ 0) :
  ∃ (n : ℤ), (d * n^3 + c * n^2 + b * n + a) % 5 = 0 := 
  sorry

end polynomial_divisible_by_five_l598_598822


namespace no_nine_digit_number_l598_598397

theorem no_nine_digit_number (f : Fin 9 → Fin 9) :
  (∀ i : Fin 8, (∃! j : Fin 9, f j = ⟨i, by simp [Fin.size 8]⟩) ∧ (∃! k : Fin 9, f k = ⟨i + 1, by simp [Fin.size 8]⟩ ∧ abs (k - j) % 2 = 1)) -> False := 
sorry

end no_nine_digit_number_l598_598397


namespace exists_valid_sequence_from_1_to_100_l598_598993

def valid_sequence (s : List ℕ) : Prop :=
  s.Nodup ∧ ∀ (n : ℕ), n ∈ s ↔ 1 ≤ n ∧ n ≤ 100 ∧ 
    ∀ (i : ℕ), i < s.length - 1 → (s[i + 1] = s[i] + 2) ∨ (s[i + 1] = s[i] - 2) ∨ 
                                        (s[i + 1] = s[i] + 5) ∨ (s[i + 1] = s[i] - 5)

theorem exists_valid_sequence_from_1_to_100 :
  ∃ (s : List ℕ), valid_sequence s :=
by
  sorry

end exists_valid_sequence_from_1_to_100_l598_598993


namespace distance_between_M_and_N_l598_598011

-- Define the points given the conditions
def Point := ℝ × ℝ × ℝ

def A : Point := (0, 0, 0)
def B : Point := (0, 2, 0)
def C : Point := (3, 2, 0)
def D : Point := (3, 0, 0)

def A' : Point := (0, 0, 12)
def B' : Point := (0, 2, 18)
def C' : Point := (3, 2, 12)
def D' : Point := (3, 0, 18)

-- Define the midpoints M and N
def midpoint (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def M : Point := midpoint A' C'
def N : Point := midpoint B' D'

-- Distance function in 3D space
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

-- The theorem we need to prove
theorem distance_between_M_and_N : distance M N = 6 :=
by
  sorry

end distance_between_M_and_N_l598_598011


namespace min_capacity_for_raft_l598_598212

-- Define the weights of the animals
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal
def number_mice : ℕ := 5
def number_moles : ℕ := 3
def number_hamsters : ℕ := 4

-- Define the minimum weight capacity for the raft
def min_weight_capacity : ℕ := 140

-- Prove that the minimum weight capacity the raft must have to transport all animals is 140 grams.
theorem min_capacity_for_raft :
  (weight_mouse * 2 ≤ min_weight_capacity) ∧ 
  (∀ trip_weight, trip_weight ≥ min_weight_capacity → 
    (trip_weight = weight_mouse * 2 ∨ trip_weight = weight_mole * 2 ∨ trip_weight = weight_hamster * 2)) :=
by 
  sorry

end min_capacity_for_raft_l598_598212


namespace hyperbola_from_ellipse_l598_598306

theorem hyperbola_from_ellipse (x y : ℝ) :
  (x^2 / 4 + y^2 / 2 = 1) →
  ((∃ x₀ : ℝ, ∃ y₀ : ℝ, x₀^2 / 2 - y₀^2 / 2 = 1) ∧ 
   (∀ p q : ℝ, x = sqrt(2) ∧ y = 0)) :=
by
  sorry

end hyperbola_from_ellipse_l598_598306


namespace sum_of_divisors_117_l598_598565

-- Defining the conditions in Lean
def n : ℕ := 117
def is_factorization : n = 3^2 * 13 := by rfl

-- The sum-of-divisors function can be defined based on the problem
def sum_of_divisors (n : ℕ) : ℕ :=
  (1 + 3 + 3^2) * (1 + 13)

-- Assertion of the correct answer
theorem sum_of_divisors_117 : sum_of_divisors n = 182 := by
  sorry

end sum_of_divisors_117_l598_598565


namespace fluid_ounce_ml_equivalence_l598_598973

theorem fluid_ounce_ml_equivalence : 
  (packets : ℕ) → (ml_per_packet : ℕ) → (ounces : ℕ) → 
  (total_ml = packets * ml_per_packet) → 
  (total_ml = ounces * ml_per_ounce) → 
  packets = 150 → ml_per_packet = 250 → ounces = 1250 → 
  ml_per_ounce = 30 :=
by intros packets ml_per_packet ounces total_ml_eq total_ml_eq_ounces
   packets_eq ml_per_packet_eq ounces_eq;
   rw [packets_eq, ml_per_packet_eq] at total_ml_eq;
   have total_ml_150_250 : total_ml = 150 * 250 := by rw [packets_eq, ml_per_packet_eq];
   rw packets_eq at total_ml_eq;
   have total_ml_37500 : total_ml = 37500 := by rw packets_eq;
   rw ounces_eq at total_ml_eq_ounces;
   have ml_per_ounce_calc : 37500 = 1250 * ml_per_ounce := by rw [ounces_eq, packets_eq];
   sorry

end fluid_ounce_ml_equivalence_l598_598973


namespace ValleyFalcons_all_items_l598_598275

noncomputable def num_fans_receiving_all_items (capacity : ℕ) (tshirt_interval : ℕ) 
  (cap_interval : ℕ) (wristband_interval : ℕ) : ℕ :=
  (capacity / Nat.lcm (Nat.lcm tshirt_interval cap_interval) wristband_interval)

theorem ValleyFalcons_all_items:
  num_fans_receiving_all_items 3000 50 25 60 = 10 :=
by
  -- This is where the mathematical proof would go
  sorry

end ValleyFalcons_all_items_l598_598275


namespace min_value_ratio_l598_598429

open Real -- Use the real numbers from Lean's mathematical library

/--
Given a triangle \( \triangle ABC \) and any point \( P \) inside the triangle,
with the sides opposite to angles \( A \), \( B \), and \( C \) being \( a \), 
\( b \), and \( c \) respectively, and the area of the triangle \( S \),
prove that the minimum value of \( \frac{a \cdot PA + b \cdot PB + c \cdot PC}{S} \)
is \( 4 \).
-/
theorem min_value_ratio (a b c PA PB PC S : ℝ) (P_inside_triangle : P ∈ interior (triangle ABC)) :
  ∃ P : Point, ∀ P_inside_triangle P, (a * PA + b * PB + c * PC) / S = 4 :=
begin
  sorry -- Proof goes here
end

end min_value_ratio_l598_598429


namespace exists_participant_with_no_more_than_4_known_l598_598609

noncomputable def participants : Type := Fin 20

def knows (a b : participants) : Prop := sorry -- To be defined as to meet the problem conditions.

theorem exists_participant_with_no_more_than_4_known :
  (Finset.univ.card : ℕ) = 20 → 
  (Finset.card (Finset.univ.filter (λ (p : participants × participants), knows p.fst p.snd))) = 49 →
  ∃ p : participants, (Finset.univ.filter (λ q, knows p q)).card ≤ 4 :=
by
  intros h_card h_pairs
  -- proof steps will go here
  sorry

end exists_participant_with_no_more_than_4_known_l598_598609


namespace abs_neg_three_halves_l598_598477

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end abs_neg_three_halves_l598_598477


namespace inequality_holds_for_all_xyz_in_unit_interval_l598_598863

theorem inequality_holds_for_all_xyz_in_unit_interval :
  ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) → 
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z)) :=
by
  intros x y z hx hy hz
  sorry

end inequality_holds_for_all_xyz_in_unit_interval_l598_598863


namespace each_person_gets_equal_share_l598_598956

-- Definitions based on the conditions
def number_of_friends: Nat := 4
def initial_chicken_wings: Nat := 9
def additional_chicken_wings: Nat := 7

-- The proof statement
theorem each_person_gets_equal_share (total_chicken_wings := initial_chicken_wings + additional_chicken_wings) : 
       total_chicken_wings / number_of_friends = 4 := 
by 
  sorry

end each_person_gets_equal_share_l598_598956


namespace car_late_time_l598_598144

def car_late_minutes (d : ℕ) (v1 v2 : ℕ) (t1 t2 : ℕ): ℕ :=
  ((d / v2) * 60) - ((d / v1) * 60)

theorem car_late_time (d : ℕ) (v1 v2 : ℕ) (t1 t2 : ℕ) (h1 : d = 70) (h2 : v1 = 40) (h3 : v2 = 35) (h4 : t1 = 1.75) (h5 : t2 = 2) :
  car_late_minutes d v1 v2 t1 t2 = 15 :=
by
  sorry

end car_late_time_l598_598144


namespace labeling_possible_l598_598412

structure Graph :=
(vertices : Type)
(edges : vertices → vertices → Bool)

def connected (G : Graph) := 
  ∀ (u v : G.vertices), ∃ (path : List G.vertices), (path.head? = some u ∧ path.getLast? = some v ∧ ∀ i, path.nth i ≠ none → G.edges (path.nth_override i).get (path.nth_override (i + 1)).get = true)

def valid_labeling (G : Graph) (label : G.edges → ℕ) := 
  ∀ v, ∃ (edges_incident : List (Σ' u, G.edges v u)), 1 < edges_incident.length → edges_incident.Pairwise (λ a b, Nat.gcd (label a.2) (label b.2) = 1)

theorem labeling_possible (G : Graph) (k : ℕ) (hG : connected G) (hE : ∃ e : G.edges, true) :
  ∃ (label : G.edges → ℕ), valid_labeling G label :=
sorry

end labeling_possible_l598_598412


namespace volume_tetrahedron_EFGH_l598_598653

def Point := (ℝ × ℝ × ℝ)

noncomputable def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def tetrahedron (E G H F : Point) : Prop :=
  distance E G = 4 ∧ distance E H = 5 ∧ distance E F = 2 ∧
  distance G H = real.sqrt 29 ∧ distance G F = 5 ∧ distance H F = real.sqrt 41

theorem volume_tetrahedron_EFGH (E G H F : Point) (h : tetrahedron E G H F) : 
  ∃ V, V = 20 / 3 :=
sorry

end volume_tetrahedron_EFGH_l598_598653


namespace sam_after_joan_took_marbles_l598_598856

theorem sam_after_joan_took_marbles
  (original_yellow : ℕ)
  (marbles_taken_by_joan : ℕ)
  (remaining_yellow : ℕ)
  (h1 : original_yellow = 86)
  (h2 : marbles_taken_by_joan = 25)
  (h3 : remaining_yellow = original_yellow - marbles_taken_by_joan) :
  remaining_yellow = 61 :=
by
  sorry

end sam_after_joan_took_marbles_l598_598856


namespace number_of_correct_statements_l598_598986

theorem number_of_correct_statements:
  let m := 1 in -- placeholder m since it should hold for any non-zero m
  let a := 2 in -- example value to hold
  let b := 3 in -- example value to hold
  let x := 4 in -- example calculation value
  let y := -0.1 * x + 1 in
  let k := 1.0 in -- positive correlation factor
  let z := k * y in
  let correct1 := am2_lt_bm2 (h: am^2 < bm^2) : a < b := 
    by { sorry }
  let correct2 := y_pos_corr_z_neg_corr_x (h: positive_correlation y z): neg_corr x y z := by { sorry }

  (correct1 ∧ correct2 ∧ ¬ incorrect3 ∧ ¬ incorrect4) = 2 :=
sorry
  ) :=
by {
  exact nat.succ (nat.succ 0) -- representing 2
}

end number_of_correct_statements_l598_598986


namespace tenth_term_of_arithmetic_sequence_l598_598381

theorem tenth_term_of_arithmetic_sequence 
  (a d : ℤ)
  (h1 : a + 2 * d = 14)
  (h2 : a + 5 * d = 32) : 
  (a + 9 * d = 56) ∧ (d = 6) := 
by
  sorry

end tenth_term_of_arithmetic_sequence_l598_598381


namespace probability_prime_gt_3_l598_598471

def is_prime (n : ℕ) : Prop := nat.prime n

def numbers : list ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def primes_greater_than_3 : list ℕ := list.filter (λ n, is_prime n ∧ n > 3) numbers

def total_balls : ℕ := list.length numbers

def eligible_balls : ℕ := list.length primes_greater_than_3

theorem probability_prime_gt_3 :
  let prob := eligible_balls.to_rat / total_balls.to_rat in
  prob = (3 : ℚ) / 10 :=
sorry

end probability_prime_gt_3_l598_598471


namespace dice_prob_not_one_l598_598136

theorem dice_prob_not_one : 
  let outcomes := [1, 2, 3, 4, 5, 6]
  let prob_not_1 := 5 / 6
  let total_outcomes := 6
  let number_of_dice := 4
  let prob := prob_not_1 ^ number_of_dice 
  prob = 625 / 1296 :=
by
  sorry

end dice_prob_not_one_l598_598136


namespace two_hundredth_digit_of_fraction_l598_598558

theorem two_hundredth_digit_of_fraction (h1 : (17 : ℚ) / 70 = (1 / 10) * (17 / 7))
    (h2 : ∃ r : ℝ, (((17 : ℚ) / 7) : ℝ) = r ∧ r = 2 + (0.428571).over) :
    (∃ d : ℕ, d = 2 ∧ ∀ n : ℕ, n = 200 → Digit_At n (17 / 70) d) := 
by
  sorry

end two_hundredth_digit_of_fraction_l598_598558


namespace asymptote_equations_line_through_P_midpoint_constant_l598_598608

noncomputable section

variable {P : ℝ × ℝ} (x0 y0 : ℝ)
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

theorem asymptote_equations : ∀ (x y : ℝ), hyperbola_equation x y -> (y = 2*x ∨ y = -2*x) :=
by
  intros x y h
  sorry

theorem line_through_P (P : ℝ × ℝ) (x0 : ℝ) : P = (x0, 2) → x0^2 = 2 → ∃ k b, y = k*x + b ∧ k = 2*sqrt(2) ∧ b = -2 :=
by
  intros h1 h2
  sorry

theorem midpoint_constant (P A B : ℝ × ℝ) (x0 y0 : ℝ) :
  hyperbola_equation x0 y0 →
  P = (x0, y0) →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∃ k : ℝ, k = 5 :=
by
  intros h1 h2 h3
  sorry

end asymptote_equations_line_through_P_midpoint_constant_l598_598608


namespace basketball_team_selection_l598_598058

theorem basketball_team_selection :
  let total_lineups := Nat.choose 16 6
  let three_quadruplet_violations := Nat.choose 4 3 * Nat.choose 12 3
  let four_quadruplet_violations := Nat.choose 4 4 * Nat.choose 12 2
  let total_violations := three_quadruplet_violations + four_quadruplet_violations
  let valid_lineups := total_lineups - total_violations
  valid_lineups = 7062 :=
by
  let total_lineups := Nat.choose 16 6
  let three_quadruplet_violations := Nat.choose 4 3 * Nat.choose 12 3
  let four_quadruplet_violations := Nat.choose 4 4 * Nat.choose 12 2
  let total_violations := three_quadruplet_violations + four_quadruplet_violations
  let valid_lineups := total_lineups - total_violations
  have h1 : total_lineups = 8008 := by sorry
  have h2 : three_quadruplet_violations = 880 := by sorry
  have h3 : four_quadruplet_violations = 66 := by sorry
  have h4 : total_violations = 946 := by sorry
  have h5 : valid_lineups = 7062 := by 
    rw [h1, h4]
    exact Nat.sub_eq_of_eq_add h4.symm
  exact h5

end basketball_team_selection_l598_598058


namespace only_set_C_is_pythagorean_triple_l598_598139

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem only_set_C_is_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 7 ∧
  ¬ is_pythagorean_triple 15 20 25 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 1 3 5 :=
by {
  -- Proof goes here
  sorry
}

end only_set_C_is_pythagorean_triple_l598_598139


namespace race_distance_l598_598771

-- Definitions for the conditions
def A_time : ℕ := 20
def B_time : ℕ := 25
def A_beats_B_by : ℕ := 14

-- Definition of the function to calculate whether the total distance D is correct
def total_distance : ℕ := 56

-- The theorem statement without proof
theorem race_distance (D : ℕ) (A_time B_time A_beats_B_by : ℕ)
  (hA : A_time = 20)
  (hB : B_time = 25)
  (hAB : A_beats_B_by = 14)
  (h_eq : (D / A_time) * B_time = D + A_beats_B_by) : 
  D = total_distance :=
sorry

end race_distance_l598_598771


namespace problem_equivalence_l598_598892

def a : ℤ := 2014
def b : ℤ := 2013
def c : ℤ := 2015

theorem problem_equivalence : a^2 - b * c = 1 :=
by
  have h : b = a - 1 := by sorry
  have h2 : c = a + 1 := by sorry
  rw [h, h2]
  calc
    a^2 - (a - 1) * (a + 1) = a^2 - (a^2 - 1) : by sorry
    ... = 1 : by sorry

end problem_equivalence_l598_598892


namespace area_CDE_l598_598044

variables (A B C D E F : Type)
variables [geometry_path A B C D E F]

-- Definitions for describing the problem
def is_on_line (X Y Z : Type) [segment X Y] : Prop := Z ∈ segment X Y
def area (T : Type) : ℝ := sorry -- some way to compute the area

-- Conditions
variables (hD_on_AC : is_on_line A C D)
variables (hE_on_BC : is_on_line B C E)
variables (hF_intersect : intersect_line A E B D = F)
variables (h_area_ABF : area (triangle A B F) = 1)
variables (h_area_ADF : area (triangle A D F) = 1/3)
variables (h_area_BEF : area (triangle B E F) = 1/4)

-- The goal
theorem area_CDE :
  area (triangle C D E) = 1/4 :=
begin
  sorry,
end

end area_CDE_l598_598044


namespace monotonic_decreasing_interval_l598_598083

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 + x + 1) * Real.exp x

-- Define the derivative of f(x)
def f_prime (x : ℝ) : ℝ := (x^2 + 3 * x + 2) * Real.exp x

-- State the theorem we want to prove
theorem monotonic_decreasing_interval : 
  ∀ x ∈ set.Ioo (-2 : ℝ) (-1 : ℝ), f_prime x < 0 :=
by 
  intros x hx
  sorry

end monotonic_decreasing_interval_l598_598083


namespace product_of_all_possible_x_product_of_x_values_l598_598358

theorem product_of_all_possible_x (x : ℝ) (h : abs (15 / x - 2) = 3) :
  x = 3 ∨ x = -15 :=
sorry

theorem product_of_x_values (h : ∃ x : ℝ, abs (15 / x - 2) = 3) :
  (∀ x : ℝ, (abs (15 / x - 2) = 3 → (x = 3 ∨ x = -15))) →
  let x1 := 3
  let x2 := -15
  x1 * x2 = -45 :=
by
  intros h1 h2
  have prod := h1 3 (by simp [abs, h2])
  have prod := h1 (-15) (by simp [abs, h2])
  rw [prod]
  simp
  -- Final substitution and calculation to show the product
  sorry

end product_of_all_possible_x_product_of_x_values_l598_598358


namespace abs_neg_frac_l598_598481

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end abs_neg_frac_l598_598481


namespace percentage_increase_to_original_price_l598_598757

theorem percentage_increase_to_original_price (x : ℝ) (h₁ : x ≠ 0) :
  ∃ y : ℝ, 0 < y ∧ (0.8 * x) * (1 + y) = x := 
begin
  use 0.25,
  split,
  { linarith, },
  { field_simp [h₁], norm_num, }
end

end percentage_increase_to_original_price_l598_598757


namespace abs_neg_three_halves_l598_598483

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end abs_neg_three_halves_l598_598483


namespace tangent_line_at_point_monotonic_and_extreme_values_inequality_holds_l598_598725

noncomputable theory

-- Define the function f based on the parameter k
def f (k x : ℝ) : ℝ := x ^ 3 + k * log x
-- Define the derivative f'
def f' (k x : ℝ) : ℝ := 3 * x ^ 2 + k / x

-- Problem (I) (i): Tangent line for k = 6 at the point (1, f(6, 1))
theorem tangent_line_at_point : (∀ x, f 6 x = x^3 + 6 * log x) → 
  (∀ x, f' 6 x = 3 * x ^ 2 + 6 / x) →
  9 * (1:ℝ) - (f 6 1) - 8 = 0 :=
  sorry

-- Problem (I) (ii): Monotonic intervals and extreme values of g when k = 6
def g (x : ℝ) : ℝ := f 6 x - f' 6 x + 9 / x

theorem monotonic_and_extreme_values : 
  ∀ x, g x = x ^ 3 - 3 * x ^ 2 + 6 * log x + 3 / x →
  (∀ x, 0 < x ∧ x < 1 → deriv g x < 0) ∧ 
  (∀ x, x > 1 → deriv g x > 0) ∧ 
  g 1 = 1 :=
  sorry

-- Problem (II): Inequality for k ≥ -3 and x1, x2 ∈ [1, +∞) with x1 > x2
theorem inequality_holds (k : ℝ) (x1 x2 : ℝ) (h1 : k ≥ -3) (h2 : 1 ≤ x1) (h3 : 1 ≤ x2) (h4 : x1 > x2) : 
  (f' k x1 + f' k x2) / 2 > (f k x1 - f k x2) / (x1 - x2) :=
  sorry

end tangent_line_at_point_monotonic_and_extreme_values_inequality_holds_l598_598725


namespace wendy_full_face_time_l598_598114

theorem wendy_full_face_time (a b c : ℕ) (H1 : a = 5) (H2 : b = 5) (H3 : c = 30) : a * b + c = 55 := 
by
  rw [H1, H2, H3]
  norm_num
  sorry

end wendy_full_face_time_l598_598114


namespace determine_remainder_l598_598133

-- Define the sequence and its sum
def geom_series_sum_mod (a r n m : ℕ) : ℕ := 
  ((r^(n+1) - 1) / (r - 1)) % m

-- Define the specific geometric series and modulo
theorem determine_remainder :
  geom_series_sum_mod 1 11 1800 500 = 1 :=
by
  -- Using geom_series_sum_mod to define the series
  let S := geom_series_sum_mod 1 11 1800 500
  -- Remainder when the series is divided by 500
  show S = 1
  sorry

end determine_remainder_l598_598133


namespace interior_angles_sum_l598_598517

def sum_of_interior_angles (sides : ℕ) : ℕ :=
  180 * (sides - 2)

theorem interior_angles_sum (n : ℕ) (h : sum_of_interior_angles n = 1800) :
  sum_of_interior_angles (n + 4) = 2520 :=
sorry

end interior_angles_sum_l598_598517


namespace triangle_construction_exists_l598_598241

-- Given two positive real numbers a and b with a < b
variables (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_lt : a < b)

-- Define an angle relationship where the angle opposite one side is three times the angle opposite the other side
def valid_triangle_with_angle_relationship (α β : ℝ) :=
  (β = 3 * α) ∧
  (a * (sin β) = b * (sin α))

-- Prove the existence of such a triangle given sides a and b
theorem triangle_construction_exists :
  ∃ (α β : ℝ), valid_triangle_with_angle_relationship a b α β :=
sorry

end triangle_construction_exists_l598_598241


namespace ashley_family_spending_l598_598629

theorem ashley_family_spending:
  let child_ticket := 4.25
  let adult_ticket := child_ticket + 3.50
  let senior_ticket := adult_ticket - 1.75
  let morning_discount := 0.10
  let total_morning_tickets := 2 * adult_ticket + 4 * child_ticket + senior_ticket
  let morning_tickets_after_discount := total_morning_tickets * (1 - morning_discount)
  let buy_2_get_1_free_discount := child_ticket
  let discount_for_5_or_more := 4.00
  let total_tickets_after_vouchers := morning_tickets_after_discount - buy_2_get_1_free_discount - discount_for_5_or_more
  let popcorn := 5.25
  let soda := 3.50
  let candy := 4.00
  let concession_total := 3 * popcorn + 2 * soda + candy
  let concession_discount := concession_total * 0.10
  let concession_after_discount := concession_total - concession_discount
  let final_total := total_tickets_after_vouchers + concession_after_discount
  final_total = 50.47 := by
  sorry

end ashley_family_spending_l598_598629


namespace total_weight_of_peppers_l598_598741

theorem total_weight_of_peppers :
  let green_pepper_weight : ℝ := 0.33
      red_pepper_weight : ℝ := 0.33
  in green_pepper_weight + red_pepper_weight = 0.66 := by
  sorry

end total_weight_of_peppers_l598_598741


namespace longest_side_is_5_l598_598980

-- Define the coordinates of the vertices
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 5)
def C : ℝ × ℝ := (5, 1)

-- Define the distance function between two points
def dist (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

-- Define the distances between the vertices of the triangle
def AB := dist A B
def AC := dist A C
def BC := dist B C

-- Define the length of the longest side of the triangle
def longest_side := max AB (max AC BC)

-- State the theorem to be proved
theorem longest_side_is_5 : longest_side = 5 :=
by
  sorry

end longest_side_is_5_l598_598980


namespace truck_capacity_cost_function_minimum_cost_l598_598466

theorem truck_capacity :
  ∃ (m n : ℕ),
    3 * m + 4 * n = 27 ∧ 
    4 * m + 5 * n = 35 ∧
    m = 5 ∧ 
    n = 3 :=
by {
  sorry
}

theorem cost_function (a : ℕ) (h : a ≤ 5) :
  ∃ (w : ℕ),
    w = 50 * a + 2250 :=
by {
  sorry
}

theorem minimum_cost :
  ∃ (w : ℕ),
    w = 2250 ∧ 
    ∀ (a : ℕ), a ≤ 5 → (50 * a + 2250) ≥ 2250 :=
by {
  sorry
}

end truck_capacity_cost_function_minimum_cost_l598_598466


namespace number_of_cats_l598_598891

/-- 
The ratio of cats to dogs at the pet store is 3:4. 
There are 20 dogs.
-/
theorem number_of_cats (h_ratio : 3 / 4) (dogs : ℕ) (h_dogs : dogs = 20) : 
cats = 3 * (dogs / 4) :=
by
let cats := 3 * (dogs / 4)
have equiv : 3 / 4 = 3 * (1 / 4), from sorry
have calc1 : dogs / 4 = 5, from sorry
have calc2 : 3 * 5 = 15, from sorry
show cats = 15, from sorry

end number_of_cats_l598_598891


namespace cumulative_percentage_decrease_l598_598976

theorem cumulative_percentage_decrease :
  let original_price := 100
  let first_reduction := original_price * 0.85
  let second_reduction := first_reduction * 0.90
  let third_reduction := second_reduction * 0.95
  let fourth_reduction := third_reduction * 0.80
  let final_price := fourth_reduction
  (original_price - final_price) / original_price * 100 = 41.86 := by
  sorry

end cumulative_percentage_decrease_l598_598976


namespace percent_geese_among_non_swans_l598_598007

theorem percent_geese_among_non_swans 
(h_total_birds : 120 = 100 * (1 + 0.20)) 
(h_distribution : ∀ (geese swans herons ducks : ℕ), 
  geese = 0.40 * 120 ∧ swans = 0.20 * 120 ∧ herons = 0.20 * 120 ∧ ducks = 0.20 * 120) : 
  0.50 = (48 / (120 - 24)) :=
by
  intro geese swans herons ducks
  have h1 : geese = 0.40 * 120 := sorry
  have h2 : swans = 0.20 * 120 := sorry
  have h3 : herons = 0.20 * 120 := sorry
  have h4 : ducks = 0.20 * 120 := sorry
  have h5 : 120 - 24 = 96 := sorry
  have h6 : 48 / 96 = 0.50 := sorry
  exact h6


end percent_geese_among_non_swans_l598_598007


namespace identity_eq_coefficients_l598_598761

theorem identity_eq_coefficients (a b c d : ℝ) :
  (∀ x : ℝ, a * x + b = c * x + d) ↔ (a = c ∧ b = d) :=
by
  sorry

end identity_eq_coefficients_l598_598761


namespace raft_minimum_capacity_l598_598194

theorem raft_minimum_capacity 
  (mice : ℕ) (mice_weight : ℕ) 
  (moles : ℕ) (mole_weight : ℕ) 
  (hamsters : ℕ) (hamster_weight : ℕ) 
  (raft_cannot_move_without_rower : Bool)
  (rower_condition : ∀ W, W ≥ 2 * mice_weight) :
  mice = 5 → mice_weight = 70 →
  moles = 3 → mole_weight = 90 →
  hamsters = 4 → hamster_weight = 120 →
  ∃ W, (W = 140) :=
by
  intros mice_eq mice_w_eq moles_eq mole_w_eq hamsters_eq hamster_w_eq
  use 140
  sorry

end raft_minimum_capacity_l598_598194


namespace sets_equal_l598_598749

-- Definitions of sets based on conditions
def M1 := {(3, 2)}
def N1 := {(2, 3)}

def M2 := {3, 2}
def N2 := {2, 3}

def M3 := {(1, 2)}
def N3 := {1, 2}

-- The theorem to prove that M2 and N2 are equal sets.
theorem sets_equal : M2 = N2 :=
by {
  sorry
}

end sets_equal_l598_598749


namespace mutually_exclusive_but_not_complementary_l598_598682

-- Define the type representing balls color
inductive BallColor
| Red
| White

-- Define the pouch containing balls
def pouch := [BallColor.Red, BallColor.Red, BallColor.White, BallColor.White]

-- Define the events:
-- Event A: exactly one white ball drawn
def eventA (draw: List BallColor) : Prop := (draw.count BallColor.White = 1)

-- Event B: exactly two white balls drawn
def eventB (draw: List BallColor) : Prop := (draw.count BallColor.White = 2)

-- Define the draw where two balls are drawn at random
def draw (pouch: List BallColor) : List (List BallColor) :=
(pouch.combinations 2)

-- Prove that eventA and eventB are mutually exclusive but not complementary
theorem mutually_exclusive_but_not_complementary :
  ∀ d ∈ draw pouch, eventA d → ¬eventB d ∧ ¬(eventA d ∨ eventB d ↔ (d.count BallColor.White = 0 ∨ d.count BallColor.Red = 0)) :=
by
  -- Proof omitted
  sorry

end mutually_exclusive_but_not_complementary_l598_598682


namespace solution_set_l598_598015

variable {f : ℝ → ℝ}

def differentiable (f : ℝ → ℝ) : Prop := sorry

def condition (x : ℝ) (f : ℝ → ℝ) : Prop := f x + x * (λ x', deriv f x') x > 0

theorem solution_set (h_diff : differentiable f)
    (h_cond : ∀ x, condition x f) :
  {x | 1 ≤ x ∧ x < 2} = {x | f' (sqrt (x + 1)) > (sqrt (x - 1)) * f (sqrt (x^2 - 1))} :=
sorry

end solution_set_l598_598015


namespace non_negative_transformation_l598_598463

-- Define the transformation operation on a triplet (x, y, z).
def transform (x y z : Int) : Int × Int × Int := (x + y, -y, z + y)

-- Prove that starting from a circle of numbers with positive sum, we can end up with all non-negative numbers.
theorem non_negative_transformation (n : ℕ) (circle : Fin n → Int)
  (h_pos_sum : (Finset.univ.sum circle : Int) > 0) :
  ∃ (circle' : Fin n → Int), 
    (∀ i, circle' i ≥ 0) ∧
    (∀ i j k : Fin n, transform (circle i) (circle j) (circle k) ≠ circle') := 
sorry

end non_negative_transformation_l598_598463


namespace length_of_MN_l598_598297

variable {Point : Type}
variable {AB CD K L M N : Point}
variable [rectangle : Rectangle AB CD]
variable [circle : Circle AB CD]
variable [intersects_ab : circle.Intersects AB K L]
variable [intersects_cd : circle.Intersects CD M N]

-- Given conditions
variable (AK : ℝ) (KL : ℝ) (DN : ℝ)
variable (AK_cond : AK = 10)
variable (KL_cond : KL = 17)
variable (DN_cond : DN = 7)

-- Conclusion
theorem length_of_MN : MN = 23 :=
  sorry

end length_of_MN_l598_598297


namespace minimize_f_l598_598563

def f (x : ℝ) : ℝ := 5 * x^2 - 30 * x + 2000

theorem minimize_f : ∃ x : ℝ, ∀ y : ℝ, f (x) ≤ f (y) ∧ f (x) = 1955 :=
begin
  use 3,
  split,
  {
    intro y,
    calc f y = 5 * y^2 - 30 * y + 2000 : by rfl
    ... = 5 * (y - 3)^2 + 1955 : by sorry, -- Step to show the function equivalence
    ... ≥ 1955 : by nlinarith,
  },
  {
    calc f 3 = 5 * 3^2 - 30 * 3 + 2000 : by rfl
    ... = 1955 : by norm_num,
  }
end

end minimize_f_l598_598563


namespace find_a_l598_598719

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b * x^2 + x

theorem find_a (a b : ℝ) (h1 : Deriv (f a b) 1 = 0) (h2 : Deriv (f a b) 2 = 0) : a = -2/3 :=
by
  sorry

end find_a_l598_598719


namespace find_modular_inverse_l598_598662

theorem find_modular_inverse :
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 228 ∧ 3 * x ≡ 1 [MOD 229] := by
  use 153
  split
  · norm_num
  split
  · norm_num
  · norm_num
  sorry

end find_modular_inverse_l598_598662


namespace six_digit_number_contains_7_l598_598832

theorem six_digit_number_contains_7
  (a b k : ℤ)
  (h1 : 100 ≤ 7 * a + k ∧ 7 * a + k < 1000)
  (h2 : 100 ≤ 7 * b + k ∧ 7 * b + k < 1000) :
  7 ∣ (1000 * (7 * a + k) + (7 * b + k)) :=
by
  sorry

end six_digit_number_contains_7_l598_598832


namespace raft_minimum_capacity_l598_598204

theorem raft_minimum_capacity (n_mice n_moles n_hamsters : ℕ)
  (weight_mice weight_moles weight_hamsters : ℕ)
  (total_weight : ℕ) :
  n_mice = 5 →
  weight_mice = 70 →
  n_moles = 3 →
  weight_moles = 90 →
  n_hamsters = 4 →
  weight_hamsters = 120 →
  (∀ (total_weight : ℕ), total_weight = n_mice * weight_mice + n_moles * weight_moles + n_hamsters * weight_hamsters) →
  (∃ (min_capacity: ℕ), min_capacity ≥ 140) :=
by
  intros
  sorry

end raft_minimum_capacity_l598_598204


namespace sum_of_digits_of_n_l598_598816

theorem sum_of_digits_of_n : 
  ∃ n : ℕ, n > 1500 ∧ 
    (Nat.gcd 40 (n + 105) = 10) ∧ 
    (Nat.gcd (n + 40) 105 = 35) ∧ 
    (Nat.digits 10 n).sum = 8 :=
by 
  sorry

end sum_of_digits_of_n_l598_598816


namespace find_f_5_l598_598437

def f (x : ℝ) : ℝ :=
  if x < 0 then
    2 * x - 1
  else if x < 4 then
    3 * x + 2
  else
    8 - 2 * x

theorem find_f_5 : f 5 = -2 := 
by
  unfold f
  split_ifs
  · sorry -- x < 0 case
  · sorry -- 0 ≤ x < 4 case
  · rfl -- x ≥ 4 case

end find_f_5_l598_598437


namespace pool_maintenance_cost_l598_598402

theorem pool_maintenance_cost 
  {d_cleaning : Nat}
  {cleaning_cost : ℕ}
  {tip_rate : ℝ}
  {d_month : Nat}
  {cleanings_per_month : ℕ}
  {use_chem_freq : ℕ}
  {chem_cost : ℕ}
  {total_cleaning_cost : ℕ}
  {total_chem_cost : ℕ}
  {total_monthly_cost : ℕ} 
  (hc1 : d_cleaning = 3)
  (hc2 : cleaning_cost = 150)
  (hc3 : tip_rate = 0.1)
  (hc4 : d_month = 30)
  (hc5 : cleanings_per_month = d_month / d_cleaning)
  (hc6 : use_chem_freq = 2)
  (hc7 : chem_cost = 200)
  (hc8 : total_cleaning_cost = cleanings_per_month * (cleaning_cost + (cleaning_cost * tip_rate).toNat))
  (hc9 : total_chem_cost = use_chem_freq * chem_cost)
  (hc10 : total_monthly_cost = total_cleaning_cost + total_chem_cost) :
  total_monthly_cost = 2050 :=
by
  sorry

end pool_maintenance_cost_l598_598402


namespace geometric_sequence_common_ratio_l598_598878

theorem geometric_sequence_common_ratio
  (q a_1 : ℝ)
  (h1: a_1 * q = 1)
  (h2: a_1 + a_1 * q^2 = -2) :
  q = -1 :=
by
  sorry

end geometric_sequence_common_ratio_l598_598878


namespace find_y_and_z_l598_598674

noncomputable def mode (l : List ℕ) : ℕ :=
  l.max' (by sorry) -- assuming non empty list

def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem find_y_and_z :
  ∃ (y z : ℕ), 
  y = 49 ∧ z = 21 ∧ 
  [45, 76, y, y, z, z].length = 6 ∧ 
  mode [45, 76, y, y, z, z] = z ∧ 
  mean [45, 76, y, y, z, z] = 2 * z ∧ 
  0 < y ∧ y ≤ 150 ∧ 
  0 < z ∧ z ≤ 150 := 
begin
  use [49, 21],
  split, 
  { exact rfl },
  split, 
  { exact rfl },
  split,
  { exact rfl },
  split,
  { refl },
  { sorry } -- Proving mode and mean relationship formally
end

end find_y_and_z_l598_598674


namespace math_word_value_is_10_l598_598508

def letter_value : ℕ → ℤ
| 1  := 1
| 2  := 2
| 3  := 1
| 4  := 0
| 5  := -1
| 6  := -2
| 7  := -1
| 8  := 0
| 9  := 1
| 0  := 2
| _  := 0  -- this covers cases beyond the intended 10, though unnecessary here

def letter_pos (c : Char) : Option ℕ :=
  -- Calculates the 1-based position of the letter in the alphabet
  let n := c.toNat - 'a'.toNat + 1 in
  if 'a' ≤ c ∧ c ≤ 'z' then some n else none

def word_value (s : String) : ℤ :=
  s.toList.foldl (λ acc c => 
    match letter_pos c with
    | some n => acc + letter_value (n % 10)
    | none => acc
  ) 0

#eval word_value "mathematics"  -- Evaluates to 10

theorem math_word_value_is_10 : word_value "mathematics" = 10 := by
  -- The proof will be skipped for now.
  sorry

end math_word_value_is_10_l598_598508


namespace radius_of_circle_from_polar_l598_598728

theorem radius_of_circle_from_polar (ρ θ : ℝ) : 
    (ρ^2 + 2 * real.sqrt 2 * ρ * real.sin (θ - real.pi / 4) - 4 = 0) → 
    ∃ r, r = real.sqrt 6 :=
by
  sorry

end radius_of_circle_from_polar_l598_598728


namespace parity_invariant_impossible_l598_598288

open Nat

-- Define the problem conditions
def initialPiles : List ℕ := (List.range 2018).map (λ i => prime (i+1))

-- Define the allowed operations
inductive Operation
| split (i : ℕ) (a b : ℕ) : Operation
| merge (i j : ℕ) (k : ℕ) : Operation

-- Define the target state
def targetPiles : List ℕ := List.replicate 2018 2018

-- Define the main theorem statement
theorem parity_invariant_impossible :
  ¬∃ ops : List Operation, (apply_operations initialPiles ops = targetPiles) := sorry

end parity_invariant_impossible_l598_598288


namespace angle_terminal_side_l598_598266

noncomputable def rad_to_deg (r : ℝ) : ℝ := r * (180 / Real.pi)

theorem angle_terminal_side :
  ∃ k : ℤ, rad_to_deg (π / 12) + 360 * k = 375 :=
sorry

end angle_terminal_side_l598_598266


namespace people_owning_only_cats_and_dogs_l598_598105

theorem people_owning_only_cats_and_dogs 
  (total_people : ℕ) 
  (only_dogs : ℕ) 
  (only_cats : ℕ) 
  (cats_dogs_snakes : ℕ) 
  (total_snakes : ℕ) 
  (only_cats_and_dogs : ℕ) 
  (h1 : total_people = 89) 
  (h2 : only_dogs = 15) 
  (h3 : only_cats = 10) 
  (h4 : cats_dogs_snakes = 3) 
  (h5 : total_snakes = 59) 
  (h6 : total_people = only_dogs + only_cats + only_cats_and_dogs + cats_dogs_snakes + (total_snakes - cats_dogs_snakes)) : 
  only_cats_and_dogs = 5 := 
by 
  sorry

end people_owning_only_cats_and_dogs_l598_598105


namespace lincoln_high_fraction_of_girls_l598_598631

noncomputable def fraction_of_girls_in_science_fair (total_girls total_boys : ℕ) (frac_girls_participated frac_boys_participated : ℚ) : ℚ :=
  let participating_girls := frac_girls_participated * total_girls
  let participating_boys := frac_boys_participated * total_boys
  participating_girls / (participating_girls + participating_boys)

theorem lincoln_high_fraction_of_girls 
  (total_girls : ℕ) (total_boys : ℕ)
  (frac_girls_participated : ℚ) (frac_boys_participated : ℚ)
  (h1 : total_girls = 150) (h2 : total_boys = 100)
  (h3 : frac_girls_participated = 4/5) (h4 : frac_boys_participated = 3/4) :
  fraction_of_girls_in_science_fair total_girls total_boys frac_girls_participated frac_boys_participated = 8/13 := 
by
  sorry

end lincoln_high_fraction_of_girls_l598_598631


namespace ratio_of_areas_l598_598536

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l598_598536


namespace a_8_is_194_l598_598885

noncomputable def a : ℕ → ℤ
| 1       := 120
| 2       := 194 - 120    -- Notice we need some initial value for a(2) which is an integer
| (n + 2) := a (n + 1) + a n

theorem a_8_is_194 : a 8 = 194 := by
  -- compute the value of a_8 using the definition of the sequence
  have h1 : a 2 = 74 := by sorry   -- This opens space for proper initial calculation or assumption for a2
  have h2 : a 3 = 194 := by sorry -- Similarly, computes subsequent terms correctly
  sorry

end a_8_is_194_l598_598885


namespace exists_n_l598_598247

-- Definitions.
def f (x : ℝ) : ℝ :=
  if x < 1/2 then x + 1/2 else x^2

variables (a b : ℝ) (a_seq b_seq : ℕ → ℝ)
  (h0 : 0 < a)
  (h1 : a < b)
  (h2 : b < 1)

-- Sequences definitions.
def a_seq_condition (n : ℕ) : Prop :=
  a_seq 0 = a ∧ ∀ n > 0, a_seq n = f (a_seq (n - 1))

def b_seq_condition (n : ℕ) : Prop :=
  b_seq 0 = b ∧ ∀ n > 0, b_seq n = f (b_seq (n - 1))

-- Main theorem statement.
theorem exists_n (n : ℕ) :
  a_seq_condition a_seq n ∧ b_seq_condition b_seq n →
  ∃ n > 0, (a_seq n - a_seq (n - 1)) * (b_seq n - b_seq (n - 1)) < 0 :=
sorry

end exists_n_l598_598247


namespace solve_inequality_l598_598370

theorem solve_inequality (a b : ℝ) (h : ∀ x, (x > 1 ∧ x < 2) ↔ (x - a) * (x - b) < 0) : a + b = 3 :=
sorry

end solve_inequality_l598_598370


namespace bill_weight_training_l598_598228

theorem bill_weight_training (jugs : ℕ) (gallons_per_jug : ℝ) (percent_filled : ℝ) (density : ℝ) 
  (h_jugs : jugs = 2)
  (h_gallons_per_jug : gallons_per_jug = 2)
  (h_percent_filled : percent_filled = 0.70)
  (h_density : density = 5) :
  jugs * gallons_per_jug * percent_filled * density = 14 := 
by
  subst h_jugs
  subst h_gallons_per_jug
  subst h_percent_filled
  subst h_density
  norm_num
  done

end bill_weight_training_l598_598228


namespace least_number_of_coins_l598_598135

theorem least_number_of_coins : ∃ (n : ℕ), 
  (n % 6 = 3) ∧ 
  (n % 4 = 1) ∧ 
  (n % 7 = 2) ∧ 
  (∀ m : ℕ, (m % 6 = 3) ∧ (m % 4 = 1) ∧ (m % 7 = 2) → n ≤ m) :=
by
  exists 9
  simp
  sorry

end least_number_of_coins_l598_598135


namespace sum_of_interior_diagonals_l598_598968

theorem sum_of_interior_diagonals (a b c : ℝ)
  (h₁ : 2 * (a * b + b * c + c * a) = 166)
  (h₂ : a + b + c = 16) :
  4 * Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2) = 12 * Real.sqrt 10 :=
by
  sorry

end sum_of_interior_diagonals_l598_598968


namespace distance_from_apex_l598_598908

theorem distance_from_apex (a₁ a₂ : ℝ) (d : ℝ)
  (ha₁ : a₁ = 150 * Real.sqrt 3)
  (ha₂ : a₂ = 300 * Real.sqrt 3)
  (hd : d = 10) :
  ∃ h : ℝ, h = 10 * Real.sqrt 2 :=
by
  sorry

end distance_from_apex_l598_598908


namespace abs_neg_three_halves_l598_598486

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end abs_neg_three_halves_l598_598486


namespace sufficient_but_not_necessary_l598_598587

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1 → 1/x < 1) ∧ ¬(1/x < 1 → x > 1) :=
by
  split
  . intro hx
    sorry -- Proof that x > 1 implies 1/x < 1
  . intro h
    sorry -- Proof that 1/x < 1 does not necessarily imply x > 1

end sufficient_but_not_necessary_l598_598587


namespace exists_m_inequality_l598_598738

theorem exists_m_inequality (a b : ℝ) (h : a > b) : ∃ m : ℝ, m < 0 ∧ a * m < b * m :=
by
  sorry

end exists_m_inequality_l598_598738


namespace workers_task_solution_l598_598111

-- Defining the variables for the number of days worked by A and B
variables (x y : ℕ)

-- Defining the total earnings for A and B
def total_earnings_A := 30
def total_earnings_B := 14

-- Condition: B worked 3 days less than A
def condition1 := y = x - 3

-- Daily wages of A and B
def daily_wage_A := total_earnings_A / x
def daily_wage_B := total_earnings_B / y

-- New scenario conditions
def new_days_A := x - 2
def new_days_B := y + 5

-- New total earnings in the scenario where they work changed days
def new_earnings_A := new_days_A * daily_wage_A
def new_earnings_B := new_days_B * daily_wage_B

-- Final proof to show the number of days worked and daily wages satisfying the conditions
theorem workers_task_solution 
  (h1 : y = x - 3)
  (h2 : new_earnings_A = new_earnings_B) 
  (hx : x = 10)
  (hy : y = 7) 
  (wageA : daily_wage_A = 3) 
  (wageB : daily_wage_B = 2) : 
  x = 10 ∧ y = 7 ∧ daily_wage_A = 3 ∧ daily_wage_B = 2 :=
by {
  sorry  -- Proof is skipped as instructed
}

end workers_task_solution_l598_598111


namespace departure_sequences_count_l598_598182

noncomputable def total_departure_sequences (trains: Finset ℕ) (A B : ℕ) 
  (h : A ∈ trains ∧ B ∈ trains ∧ trains.card = 6) 
  (hAB : ∀ g1 g2 : Finset ℕ, g1 ∪ g2 = trains ∧ g1.card = 3 ∧ g2.card = 3 → ¬(A ∈ g1 ∧ B ∈ g1 ∨ A ∈ g2 ∧ B ∈ g2)) 
  : ℕ := 6 * 6 * 6

-- The main theorem statement: given the conditions, prove the total number of different sequences is 216
theorem departure_sequences_count (trains: Finset ℕ) (A B : ℕ)
  (h : A ∈ trains ∧ B ∈ trains ∧ trains.card = 6)
  (hAB : ∀ g1 g2 : Finset ℕ, g1 ∪ g2 = trains ∧ g1.card = 3 ∧ g2.card = 3 → ¬(A ∈ g1 ∧ B ∈ g1 ∨ A ∈ g2 ∧ B ∈ g2)) 
  : total_departure_sequences trains A B h hAB = 216 := 
by 
  sorry

end departure_sequences_count_l598_598182


namespace estimate_initial_probability_calculate_additional_white_balls_l598_598382

-- Definitions for problem conditions
def total_balls : ℕ := 60
def initial_probability : ℚ := 0.25
def desired_probability : ℚ := 2/5

-- Expected results
def initial_estimated_probability : ℚ := 0.25
def additional_white_balls_needed : ℕ := 15

-- Math proof problem statement in Lean 4
theorem estimate_initial_probability :
  initial_probability = initial_estimated_probability :=
by
  sorry

theorem calculate_additional_white_balls (total_balls initial_white_balls additional_white_balls : ℕ) 
  (initial_probability desired_probability : ℚ) :
  initial_white_balls = total_balls * initial_probability →
  initial_white_balls = 15 →
  (λ x, (initial_white_balls + x) / (total_balls + x)) additional_white_balls =
  desired_probability → 
  additional_white_balls_needed = additional_white_balls :=
by
  sorry

end estimate_initial_probability_calculate_additional_white_balls_l598_598382


namespace round_trip_percentage_l598_598153

variable (P R : ℝ)
variable (h1 : 0.30 * P = 0.40 * R)
variable (h2 : P ≠ 0)

theorem round_trip_percentage : R / P = 0.75 :=
by
  have h : 0.30 * P = 0.40 * R, from h1
  have hP : P ≠ 0, from h2
  sorry -- Skip the proof.

end round_trip_percentage_l598_598153


namespace inequality_problem_l598_598017

variable {n : ℕ} (a : Fin n → ℝ)

theorem inequality_problem (h1 : ∀ k : Fin n, 0 < a k)
  (h2 : ¬∀ (i j : Fin n), i ≠ j → a i = a j)
  (h3 : ∑ k, (a k) ^ (-2 * n) = 1) :
  (∑ k, (a k) ^ (2 * n)) - (n^2 * ∑ (i j : Fin n) (h : i.val < j.val), ((a i / a j) - (a j / a i)) ^ 2) > n^2 :=
by
  sorry

end inequality_problem_l598_598017


namespace round_24_7394_to_nearest_hundredth_l598_598855

theorem round_24_7394_to_nearest_hundredth :
  (Real.round_to_precision 2 24.7394 = 24.74) :=
sorry

end round_24_7394_to_nearest_hundredth_l598_598855


namespace covered_exactly_twice_l598_598041

-- Definition of the grid and folding process
def grid := fin 5 × fin 7
def folding_function := (coords: grid) --> coords' : grid -- placeholder for the actual folding function

theorem covered_exactly_twice : 
  (* Assuming a fold function that transforms coordinates appropriately *)
  ∑ i j, if (number_of_overlaps ((i, j) : grid)) = 2 then 1 else 0 = 9 :=
begin
  sorry
end

end covered_exactly_twice_l598_598041


namespace balance_blue_balls_l598_598040

noncomputable def weight_balance (G B Y W : ℝ) : ℝ :=
  3 * G + 3 * Y + 5 * W

theorem balance_blue_balls (G B Y W : ℝ)
  (hG : G = 2 * B)
  (hY : Y = 2 * B)
  (hW : W = (5 / 3) * B) :
  weight_balance G B Y W = (61 / 3) * B :=
by
  sorry

end balance_blue_balls_l598_598040


namespace integral_calculation_l598_598250

theorem integral_calculation :
  ∫ x in 0..1, (x^2 + Real.exp x - 1/3) = Real.exp 1 - 1 :=
by
  sorry

end integral_calculation_l598_598250


namespace find_A_find_area_area_correct_l598_598301

-- Define the general context and conditions.
variable (a b c : ℝ) (A B C : ℝ)

-- Define constants.
def pi_div_six : ℝ := Real.pi / 6
def pi_div_three : ℝ := Real.pi / 3
def pi_div_four : ℝ := Real.pi / 4
def five_pi_div_twelve : ℝ := 5 * Real.pi / 12

-- Condition 1: The given trigonometric equation.
def condition1 : Prop := 2 * a * Real.sin (C + pi_div_six) = b + c

-- Proving part 1: Finding angle A given the condition.
theorem find_A (h : condition1 a b c A B C) : A = pi_div_three := sorry

-- Given additional conditions for part 2.
variable (hB : B = pi_div_four) (h_b_a : b - a = Real.sqrt 2 - Real.sqrt 3)

-- Calculate side lengths based on given conditions.
def calc_a : ℝ := (Real.sqrt 2 - Real.sqrt 3) / (Real.csc pi_div_four - 1)
def calc_b : ℝ := calc_a * Real.csc pi_div_four

-- Proving part 2: Finding the area of the triangle.
theorem find_area (h : condition1 a b c A B C) (hA : A = pi_div_three) (hB : B = pi_div_four) (h_b_a : b - a = Real.sqrt 2 - Real.sqrt 3)
  : Real := 
  let a := calc_a in
  let b := calc_b in
  0.5 * a * b * Real.sin five_pi_div_twelve

theorem area_correct (h : condition1 a b c A B C) (hA : A = pi_div_three) (hB : B = pi_div_four) (h_b_a : b - a = Real.sqrt 2 - Real.sqrt 3)
  : find_area a b c A B C h hA hB h_b_a = (3 + Real.sqrt 3) / 4 := sorry

end find_A_find_area_area_correct_l598_598301


namespace counting_perfect_squares_l598_598344

theorem counting_perfect_squares :
  let count := (finset.range 71).filter (λ n, n * n < 5000 ∧ ((n * n % 10 = 4) ∨ 
                                                               (n * n % 10 = 5) ∨ 
                                                               (n * n % 10 = 6))).card in
  count = 35 :=
sorry

end counting_perfect_squares_l598_598344


namespace enclosedArea_l598_598559

theorem enclosedArea (x y : ℝ) :
  (x^2 + y^2 = 2 * (|x| + |y|)) → (area {p : ℝ × ℝ | p.1^2 + p.2^2 = 2 * (|p.1| + |p.2|)}) = 2 * real.pi :=
by
  sorry

end enclosedArea_l598_598559


namespace sum_of_num_denom_l598_598927

theorem sum_of_num_denom (x : ℚ) (hx : x = 0.24 * (1 / (1 - 10^(-2)))) :
  (x.num + x.denom) = 41 := sorry

end sum_of_num_denom_l598_598927


namespace tennis_tournament_rounds_needed_l598_598164

theorem tennis_tournament_rounds_needed (n : ℕ) (total_participants : ℕ) (win_points loss_points : ℕ) (get_point_no_pair : ℕ) (elimination_loss : ℕ) :
  total_participants = 1152 →
  win_points = 1 →
  loss_points = 0 →
  get_point_no_pair = 1 →
  elimination_loss = 2 →
  n = 14 :=
by
  sorry

end tennis_tournament_rounds_needed_l598_598164


namespace wendy_full_face_time_l598_598115

theorem wendy_full_face_time (a b c : ℕ) (H1 : a = 5) (H2 : b = 5) (H3 : c = 30) : a * b + c = 55 := 
by
  rw [H1, H2, H3]
  norm_num
  sorry

end wendy_full_face_time_l598_598115


namespace count_valid_n_l598_598434

def is_even (r : ℕ) : Prop := r % 2 = 0

def satisfies_condition (n q r : ℕ) : Prop :=
  n = q * 100 + r ∧ (101 ≤ q ∧ q ≤ 999) ∧ is_even r ∧ (q + r) % 11 = 0

theorem count_valid_n :
  let valid_n_count := { n : ℕ | ∃ q r : ℕ, satisfies_condition n q r }.to_finset.card
  valid_n_count = 4495 := by
  sorry

end count_valid_n_l598_598434


namespace sum_even_factors_630_l598_598130

theorem sum_even_factors_630 : 
  (∑ n in (finset.filter (λ n, even n) (divisors 630)), n) = 1248 := 
sorry

end sum_even_factors_630_l598_598130


namespace prime_sum_as_product_l598_598456

theorem prime_sum_as_product (p : ℕ → ℕ) (h₁ : ∀ n, p n ∈ prime) (h₂ : ∀ n, p n < p (n + 1)) (n : ℕ) (hn : n ≥ 2) :
  ∃ A B : ℕ, A ≥ 2 ∧ B ≥ 2 ∧ p n + p (n + 1) = 2 * A * B := 
sorry

end prime_sum_as_product_l598_598456


namespace find_T4_l598_598316

-- We translate the conditions and the problem to Lean's syntax and logic

-- Assume a_n is a geometric sequence with first term a_1 and common ratio q
variable (a1 q : ℝ)

-- Given conditions
def a2 := a1 * q
def a3 := a1 * q^2
def a4 := a1 * q^3
def a7 := a1 * q^6
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)
def T (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), S i

-- Conditions from the problem
axiom condition1 : a2 * a3 = 2 * a1
axiom condition2 : (a4 + 2 * a7) / 2 = 5 / 4

-- Statement to prove
theorem find_T4 : T 4 = X :=
by
  sorry

end find_T4_l598_598316


namespace probability_neither_red_nor_purple_l598_598931

theorem probability_neither_red_nor_purple (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) : 
  total_balls = 60 →
  white_balls = 22 →
  green_balls = 18 →
  yellow_balls = 2 →
  red_balls = 15 →
  purple_balls = 3 →
  (total_balls - red_balls - purple_balls : ℚ) / total_balls = 7 / 10 :=
by
  sorry

end probability_neither_red_nor_purple_l598_598931


namespace area_of_sin3x_on_0_to_2pi_div_3_l598_598876

theorem area_of_sin3x_on_0_to_2pi_div_3:
  (∀ (n : ℕ) (hn : n > 0), ∫ x in 0..(π / n), sin (n * x) = 2 / n) →
  ∫ x in 0..(2 * π / 3), sin (3 * x) = 4 / 3 :=
by
  sorry

end area_of_sin3x_on_0_to_2pi_div_3_l598_598876


namespace wendy_full_face_time_l598_598116

-- Define the constants based on the conditions
def num_products := 5
def wait_time := 5
def makeup_time := 30

-- Calculate the total time to put on "full face"
def total_time (products : ℕ) (wait_time : ℕ) (makeup_time : ℕ) : ℕ :=
  (products - 1) * wait_time + makeup_time

-- The theorem stating that Wendy's full face routine takes 50 minutes
theorem wendy_full_face_time : total_time num_products wait_time makeup_time = 50 :=
by {
  -- the proof would be provided here, for now we use sorry
  sorry
}

end wendy_full_face_time_l598_598116


namespace algebraic_simplification_l598_598580

variables (a b : ℝ)

theorem algebraic_simplification (h : a > b ∧ b > 0) : 
  ((a + b) / ((Real.sqrt a - Real.sqrt b)^2)) * 
  (((3 * a * b - b * Real.sqrt (a * b) + a * Real.sqrt (a * b) - 3 * b^2) / 
    (1/2 * Real.sqrt (1/4 * ((a / b + b / a)^2) - 1)) + 
   (4 * a * b * Real.sqrt a + 9 * a * b * Real.sqrt b - 9 * b^2 * Real.sqrt a) / 
   (3/2 * Real.sqrt b - 2 * Real.sqrt a))) 
  = -2 * b * (a + 3 * Real.sqrt (a * b)) :=
sorry

end algebraic_simplification_l598_598580


namespace max_sum_of_squares_l598_598898

theorem max_sum_of_squares {x : Fin 10 → ℝ}
  (hmin : ∀ i, 2 ≤ x i)
  (hmax : ∀ i, x i ≤ 10)
  (hsum : (∑ i, x i) = 70) :
  (∑ i, (x i) ^ 2) ≤ 628 :=
sorry

end max_sum_of_squares_l598_598898


namespace min_n_for_x7_term_l598_598707

theorem min_n_for_x7_term (n : ℕ) (h : 0 < n) :
  (∃ r, binomial n r * x^(2 * (n - r)) * (1/x^3)^r = x^7) ↔ n = 6 := 
sorry

end min_n_for_x7_term_l598_598707


namespace min_capacity_for_raft_l598_598210

-- Define the weights of the animals
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal
def number_mice : ℕ := 5
def number_moles : ℕ := 3
def number_hamsters : ℕ := 4

-- Define the minimum weight capacity for the raft
def min_weight_capacity : ℕ := 140

-- Prove that the minimum weight capacity the raft must have to transport all animals is 140 grams.
theorem min_capacity_for_raft :
  (weight_mouse * 2 ≤ min_weight_capacity) ∧ 
  (∀ trip_weight, trip_weight ≥ min_weight_capacity → 
    (trip_weight = weight_mouse * 2 ∨ trip_weight = weight_mole * 2 ∨ trip_weight = weight_hamster * 2)) :=
by 
  sorry

end min_capacity_for_raft_l598_598210


namespace sum_of_numbers_l598_598799

theorem sum_of_numbers (x y : ℕ) (hx : 100 ≤ x ∧ x < 1000) (hy : 1000 ≤ y ∧ y < 10000) (h : 10000 * x + y = 12 * x * y) :
  x + y = 1083 :=
sorry

end sum_of_numbers_l598_598799


namespace table_area_l598_598527

theorem table_area (A : ℝ) 
  (combined_area : ℝ)
  (coverage_percentage : ℝ)
  (area_two_layers : ℝ)
  (area_three_layers : ℝ)
  (combined_area_eq : combined_area = 220)
  (coverage_percentage_eq : coverage_percentage = 0.80 * A)
  (area_two_layers_eq : area_two_layers = 24)
  (area_three_layers_eq : area_three_layers = 28) :
  A = 275 :=
by
  -- Assumptions and derivations can be filled in.
  sorry

end table_area_l598_598527


namespace sum_of_positive_integers_eq_32_l598_598500

noncomputable def sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : ℕ :=
  x + y

theorem sum_of_positive_integers_eq_32 (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : sum_of_integers x y h1 h2 = 32 :=
  sorry

end sum_of_positive_integers_eq_32_l598_598500


namespace non_congruent_triangle_count_l598_598744

-- Define the points in the grid.
def points : List (ℝ × ℝ) := [(0,0), (0.5,0), (1,0), (1.5,0), 
                               (0,0.5), (0.5,0.5), (1,0.5), (1.5,0.5)]

-- Define what it means for a triangle to be non-congruent.
def non_congruent (a b c : ℝ × ℝ) (d e f : ℝ × ℝ) : Prop :=
  ¬ ((a, b, c) = (d, e, f) ∨ -- Identical configuration
     (a, b, c) = (d, f, e) ∨ -- Reflection across some axis
     (a, b, c) = (e, d, f) ∨ -- Rotation by 180°
     (a, b, c) = (e, f, d))  -- Any other congruent move

-- Definition to count non-congruent triangles from the given points.
def count_non_congruent_triangles (pts : List (ℝ × ℝ)) : Nat :=
  let triangles := List.bind pts (λ a, 
    List.bind pts (λ b, 
      List.bind pts (λ c, 
        if a ≠ b ∧ b ≠ c ∧ a ≠ c then [(a, b, c)] else [])))
  ((triangles.map (λ (t : ℝ × ℝ × ℝ), 
    (∀ (u : ℝ × ℝ × ℝ), u ∈ triangles → non_congruent t u))).length)

-- The theorem stating that the number of non-congruent triangles is 3.
theorem non_congruent_triangle_count : count_non_congruent_triangles points = 3 := 
by
  sorry

end non_congruent_triangle_count_l598_598744


namespace a_n_lt_n_sq_l598_598420

noncomputable def a : ℕ → ℕ
-- We need to define a^ ⟨sequence here satisfying the conditions,
-- but since the specific form of a is not given, we'll assume a noncomputable definition
-- which would be provided proof or defined outside of this snippet

axiom a_mono_increasing : ∀ n : ℕ, a n ≤ a (n+1)
axiom a_in_nat : ∀ n : ℕ, a n ∈ ℕ
axiom a_condition : ∀ x : ℕ, ∃ i j : ℕ, (i ≠ j ∧ x = a i + a j) ∨ x = a i

theorem a_n_lt_n_sq : ∀ n : ℕ, a n < n^2 :=
by
  intro n
  sorry

end a_n_lt_n_sq_l598_598420


namespace trapezoid_area_proof_l598_598951

noncomputable def trapezoid_area (AK DK CD : ℝ) : ℝ :=
  let AB : ℝ := AK + DK
  let base1 : ℝ := AK + DK + AB + CD
  let height : ℝ := 2 * math.sqrt (AK * DK)
  0.5 * (base1 + CD) * height

theorem trapezoid_area_proof : 
  trapezoid_area 16 4 6 = 432 := 
by
  -- Explanation of given conditions 
  -- and direct calculation follows:
  sorry

end trapezoid_area_proof_l598_598951


namespace ratio_of_40_to_8_l598_598120

theorem ratio_of_40_to_8 : 40 / 8 = 5 := 
by
  sorry

end ratio_of_40_to_8_l598_598120


namespace raft_min_capacity_l598_598190

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end raft_min_capacity_l598_598190


namespace max_abs_a_geq_max_abs_b_l598_598936

variable (n : ℕ)
variable (a : ℕ → ℝ)
variable (b : ℕ × ℝ)

noncomputable def sum_a_zero : Prop :=
  (∑ i in finset.range n, a i) = 0

noncomputable def b_def (i : ℕ) : ℕ → ℝ :=
  ∑ j in finset.range (i + 1), a j

noncomputable def b_condition (i j : ℕ) : Prop :=
  i < j → (b i) * (a j - a (i + 1)) ≥ 0

theorem max_abs_a_geq_max_abs_b
  (sum_a_zero : sum_a_zero a n)
  (b_def : ∀ (i : ℕ), b i = b_def a n i)
  (b_condition : ∀ (i j : ℕ), b_condition n b i j):
  (finset.max (finset.range n) (λ i, |a i|)) ≥ (finset.max (finset.range n) (λ i, |b i|)) :=
  sorry

end max_abs_a_geq_max_abs_b_l598_598936


namespace investor_total_profit_l598_598990

theorem investor_total_profit 
  (total_investment : ℝ)
  (fund_one_investment : ℝ)
  (fund_one_profit_rate : ℝ)
  (fund_two_profit_rate : ℝ)
  (total_investment = 1900)
  (fund_one_investment = 1700)
  (fund_one_profit_rate = 0.09)
  (fund_two_profit_rate = 0.02)
  (fund_two_investment : ℝ)
  (fund_two_investment = total_investment - fund_one_investment)
  (profit_one : ℝ)
  (profit_one = fund_one_investment * fund_one_profit_rate)
  (profit_two : ℝ)
  (profit_two = fund_two_investment * fund_two_profit_rate) :
  profit_one + profit_two = 157 :=
sorry

end investor_total_profit_l598_598990


namespace chess_tournament_wins_minus_losses_l598_598598

theorem chess_tournament_wins_minus_losses (W L D : ℕ) (h1 : W + L + D = 20) (h2 : W + 0.5 * D = 12.5) : W - L = 5 :=
by
  sorry

end chess_tournament_wins_minus_losses_l598_598598


namespace tan_of_angle_in_second_quadrant_l598_598310

theorem tan_of_angle_in_second_quadrant (α : ℝ) (hα1 : π / 2 < α ∧ α < π) (hα2 : Real.cos (π / 2 - α) = 4 / 5) : Real.tan α = -4 / 3 :=
by
  sorry

end tan_of_angle_in_second_quadrant_l598_598310


namespace inclination_angle_is_45_degrees_l598_598097

open Real

-- Define the coordinates of the points.
def point1 : ℝ × ℝ := (0, 0)
def point2 : ℝ × ℝ := (-1, -1)

-- Define the slope m of the line passing through the points.
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the inclination angle using the arctangent function.
def inclination_angle (m : ℝ) : ℝ := arctan m

-- The main theorem to prove that the inclination angle of the line 
-- passing through the given points is 45 degrees.
theorem inclination_angle_is_45_degrees :
  inclination_angle (slope point1 point2) = π / 4 :=
by
  -- Provide the proof here.
  sorry

end inclination_angle_is_45_degrees_l598_598097


namespace algebraic_expression_evaluation_l598_598718

theorem algebraic_expression_evaluation (x m : ℝ) (h1 : 5 * (2 - 1) + 3 * m * 2 = -7) (h2 : m = -2) :
  5 * (x - 1) + 3 * m * x = -1 ↔ x = -4 :=
by
  sorry

end algebraic_expression_evaluation_l598_598718


namespace Andrew_is_19_l598_598992

-- Define individuals and their relationships
def Andrew_age (Bella_age : ℕ) : ℕ := Bella_age - 5
def Bella_age (Carlos_age : ℕ) : ℕ := Carlos_age + 4
def Carlos_age : ℕ := 20

-- Formulate the problem statement
theorem Andrew_is_19 : Andrew_age (Bella_age Carlos_age) = 19 :=
by
  sorry

end Andrew_is_19_l598_598992


namespace sum_of_other_endpoint_coordinates_l598_598047

theorem sum_of_other_endpoint_coordinates 
  (A B O : ℝ × ℝ)
  (hA : A = (6, -2)) 
  (hO : O = (3, 5)) 
  (midpoint_formula : (A.1 + B.1) / 2 = O.1 ∧ (A.2 + B.2) / 2 = O.2):
  (B.1 + B.2) = 12 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l598_598047


namespace lcm_inequality_l598_598409

theorem lcm_inequality
  (a b c d e : ℤ)
  (h1 : 1 ≤ a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : d < e) :
  (1 : ℚ) / Int.lcm a b + (1 : ℚ) / Int.lcm b c + 
  (1 : ℚ) / Int.lcm c d + (1 : ℚ) / Int.lcm d e ≤ (15 : ℚ) / 16 := by
  sorry

end lcm_inequality_l598_598409


namespace cannot_achieve_80_cents_l598_598671

def is_possible_value (n : ℕ) : Prop :=
  ∃ (n_nickels n_dimes n_quarters n_half_dollars : ℕ), 
    n_nickels + n_dimes + n_quarters + n_half_dollars = 5 ∧
    5 * n_nickels + 10 * n_dimes + 25 * n_quarters + 50 * n_half_dollars = n

theorem cannot_achieve_80_cents : ¬ is_possible_value 80 :=
by sorry

end cannot_achieve_80_cents_l598_598671


namespace prove_proposition_correct_l598_598021

variables (m n : Line) (α β γ : Plane)

def proposition (m n : Line) (α β γ : Plane) : Prop :=
  (α ∥ β) ∧ (β ∥ γ) ∧ (m ⟂ α) → (m ⟂ γ)

theorem prove_proposition_correct :
  proposition m n α β γ :=
by
  sorry

end prove_proposition_correct_l598_598021


namespace part_one_part_two_l598_598829

open Real

-- Part (1)
theorem part_one (x : ℝ) : (∀ x, abs (x - 1) + abs (x + 1) ≥ 3) → x ≤ -1.5 ∨ x ≥ 1.5 := by
  sorry

-- Part (2)
theorem part_two (a : ℝ) : (∀ x, abs (x - 1) + abs (x - a) ≥ 2) → (a = 3 ∨ a = -1) := by
  sorry

end part_one_part_two_l598_598829


namespace heights_of_parallelogram_l598_598884

-- Define a parallelogram in Lean
structure Parallelogram (A B C D : Type) :=
  (is_parallelogram : -- Conditions to specify the properties of a parallelogram
    ∀ h1 h2 : Type, (h1 = h2) → (true))

-- The problem states we need to prove properties about the heights of a parallelogram
theorem heights_of_parallelogram (A B C D : Type) [Parallelogram A B C D]:
  ∀ h_opposite h_adjacent : Type,
    (h_opposite = h_opposite) →
    (h_adjacent ≠ h_adjacent) :=
by
  sorry

end heights_of_parallelogram_l598_598884


namespace isosceles_triangle_area_l598_598904

theorem isosceles_triangle_area {A B C : Type} 
  (h_isosceles : ΔABC.IsIsosceles)
  (h_sides : ℓ(A, B) = 13 ∧ ℓ(B, C) = 24 ∧ ℓ(C, A) = 13) :
  ΔABC.area = 60 :=
by
  sorry

end isosceles_triangle_area_l598_598904


namespace coffee_blend_l598_598221

variable (pA pB : ℝ) (cA cB : ℝ) (total_cost : ℝ) 

theorem coffee_blend (hA : pA = 4.60) 
                     (hB : pB = 5.95) 
                     (h_ratio : cB = 2 * cA) 
                     (h_total : 4.60 * cA + 5.95 * cB = 511.50) : 
                     cA = 31 := 
by
  sorry

end coffee_blend_l598_598221


namespace sum_of_B_values_l598_598963

def last_three_digits (n : ℕ) : ℕ := n % 1000

def divisible_by_12 (n : ℕ) : Prop := n % 12 = 0

theorem sum_of_B_values (∀ B : ℕ, B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → 
  divisible_by_12 (last_three_digits (99 * 10^7 + B * 10^4 + 466176)) = 
  (B = 2 ∨ B = 5 ∨ B = 8)) :
  2 + 5 + 8 = 15 :=
by
  -- Here you normally provide the proof, but we'll use sorry as we are only setting up the problem
  sorry

end sum_of_B_values_l598_598963


namespace mod_inverse_9_mod_23_l598_598269

theorem mod_inverse_9_mod_23 : ∃ (a : ℤ), 0 ≤ a ∧ a < 23 ∧ (9 * a) % 23 = 1 :=
by
  use 18
  sorry

end mod_inverse_9_mod_23_l598_598269


namespace find_possible_values_of_alpha_l598_598416

noncomputable def possible_values_of_alpha (α : ℂ) : Prop :=
α ≠ 1 ∧ abs (α^2 - 1) = 3 * abs (α - 1) ∧ abs (α^4 - 1) = 5 * abs (α - 1)

theorem find_possible_values_of_alpha : ∃ α : ℂ, possible_values_of_alpha α :=
begin
  sorry
end

end find_possible_values_of_alpha_l598_598416


namespace triangle_area_l598_598089

theorem triangle_area (P : ℝ) (r : ℝ) (s : ℝ) (A : ℝ) :
  P = 42 → r = 5 → s = P / 2 → A = r * s → A = 105 :=
by
  intro hP hr hs hA
  sorry

end triangle_area_l598_598089


namespace placement_of_balls_l598_598102

noncomputable def number_of_arrangements (balls boxes : ℕ) : ℕ :=
  if H : balls <= boxes then Nat.factorial boxes / Nat.factorial (boxes - balls)
  else 0

theorem placement_of_balls :
  number_of_arrangements 3 5 = 60 :=
by
  -- Verification steps showing that 3 <= 5 and calculating A_5^3
  dsimp [number_of_arrangements]
  rw [if_pos (by decide)]
  -- factorial 5 = 120, factorial (5 - 3) = 2, 120 / 2 = 60
  norm_num

end placement_of_balls_l598_598102


namespace probability_of_odd_number_l598_598974

theorem probability_of_odd_number (wedge1 wedge2 wedge3 wedge4 wedge5 : ℝ)
  (h_wedge1_split : wedge1/3 = wedge2) 
  (h_wedge2_twice_wedge1 : wedge2 = 2 * (wedge1/3))
  (h_wedge3 : wedge3 = 1/4)
  (h_wedge5 : wedge5 = 1/4)
  (h_total : wedge1/3 + wedge2 + wedge3 + wedge4 + wedge5 = 1) :
  wedge1/3 + wedge3 + wedge5 = 7 / 12 :=
by
  sorry

end probability_of_odd_number_l598_598974


namespace counting_perfect_squares_l598_598345

theorem counting_perfect_squares :
  let count := (finset.range 71).filter (λ n, n * n < 5000 ∧ ((n * n % 10 = 4) ∨ 
                                                               (n * n % 10 = 5) ∨ 
                                                               (n * n % 10 = 6))).card in
  count = 35 :=
sorry

end counting_perfect_squares_l598_598345


namespace total_grocery_cost_in_usd_l598_598442

theorem total_grocery_cost_in_usd :
  let
    cookies_usd := 12 * 2.50
    cereals_usd := 5 * 3.40

    noodles_eur := 16 * 1.80
    euros_to_usd := (1 / 0.85)
    noodles_usd := noodles_eur * euros_to_usd

    soup_gbp := 28 * 1.20
    gbp_to_usd := (1 / 0.75)
    soup_usd := soup_gbp * gbp_to_usd

    crackers_eur := 45 * 1.10
    crackers_usd := crackers_eur * euros_to_usd

    total_cost_usd := cookies_usd + cereals_usd + noodles_usd + soup_usd + crackers_usd
  in
    total_cost_usd = 183.92 :=
by
  sorry

end total_grocery_cost_in_usd_l598_598442


namespace ship_speeds_l598_598110

theorem ship_speeds (x : ℝ) 
  (h1 : (2 * x) ^ 2 + (2 * (x + 3)) ^ 2 = 174 ^ 2) :
  x = 60 ∧ x + 3 = 63 :=
by
  sorry

end ship_speeds_l598_598110


namespace sum_of_even_factors_of_630_l598_598125

noncomputable def sum_of_positive_even_factors (n : Nat) : Nat :=
  ∑ i in (Finset.filter (λ d, d % 2 = 0) (Finset.divisors n)), i

theorem sum_of_even_factors_of_630 :
  (sum_of_positive_even_factors 630) = 1248 := by
  sorry

end sum_of_even_factors_of_630_l598_598125


namespace sports_club_membership_l598_598774

theorem sports_club_membership (B T Both Neither : ℕ) (hB : B = 17) (hT : T = 19) (hBoth : Both = 11) (hNeither : Neither = 2) :
  B + T - Both + Neither = 27 := by
  sorry

end sports_club_membership_l598_598774


namespace possible_face_value_totals_cube_l598_598888

theorem possible_face_value_totals_cube (S : Int) :
  ∃ (f : Fin 8 → ℤ), (∀ i, f i = 1 ∨ f i = -1) ∧
  let face_product (a b c d : Fin 8) : ℤ :=
    f a * f b * f c * f d in
  let face_values := [
    face_product (0, 1, 2, 3),
    face_product (4, 5, 6, 7),
    face_product (0, 1, 4, 5),
    face_product (2, 3, 6, 7),
    face_product (0, 2, 4, 6),
    face_product (1, 3, 5, 7)
  ] in
  S = face_values.sum →
  S ∈ {14, 6, 2, -2, -6, -10} :=
sorry

end possible_face_value_totals_cube_l598_598888


namespace impossible_to_make_more_than_half_million_moves_l598_598449

theorem impossible_to_make_more_than_half_million_moves :
  let n : Nat := 1000 in
  let total_moves := (n - 1) * n / 2 in
  total_moves < 500000 :=
begin
  -- Insert the proof here
  sorry
end

end impossible_to_make_more_than_half_million_moves_l598_598449


namespace find_values_of_a1_l598_598147

theorem find_values_of_a1 (a1 d : ℤ) (S : ℤ) :
  (∀ d, ∃ a1, (∃ S, S = 5 * a1 + 10 * d ∧
      (a1 + 5 * d) * (a1 + 10 * d) > S + 15 ∧
      (a1 + 8 * d) * (a1 + 7 * d) < S + 39)) →
  a1 ∈ {-9, -8, -7, -6, -4, -3, -2, -1} :=
by sorry

end find_values_of_a1_l598_598147


namespace locus_of_centers_is_spherical_triangle_l598_598170

noncomputable def locus_of_circle_centers (R : ℝ) : set (ℝ × ℝ × ℝ) :=
  {p | let x := p.1, y := p.2, z := p.3 in 
       (x = R ∨ x = -R) ∧ (y = R ∨ y = -R) ∧ (z = R ∨ z = -R) ∧ 
       x^2 + y^2 + z^2 = 2 * R^2 ∧ 
       x >= 0 ∧ y >= 0 ∧ z >= 0}

theorem locus_of_centers_is_spherical_triangle (R : ℝ) :
  ∀ p : ℝ × ℝ × ℝ, p ∈ locus_of_circle_centers R ↔ 
    (p.1 = R ∨ p.2 = R ∨ p.3 = R) ∧ p.1^2 + p.2^2 + p.3^2 = 2 * R^2 ∧ 
    0 ≤ p.1 ∧ p.1 ≤ R ∧ 0 ≤ p.2 ∧ p.2 ≤ R ∧ 0 ≤ p.3 ∧ p.3 ≤ R := 
by
  sorry

end locus_of_centers_is_spherical_triangle_l598_598170


namespace other_endpoint_coordinates_sum_l598_598053

noncomputable def other_endpoint_sum (x1 y1 x2 y2 xm ym : ℝ) : ℝ :=
  let x := 2 * xm - x1
  let y := 2 * ym - y1
  x + y

theorem other_endpoint_coordinates_sum :
  (other_endpoint_sum 6 (-2) 0 12 3 5) = 12 := by
  sorry

end other_endpoint_coordinates_sum_l598_598053


namespace find_g_inv_84_l598_598754

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem find_g_inv_84 : g 3 = 84 → ∃ x, g x = 84 ∧ x = 3 :=
by
  sorry

end find_g_inv_84_l598_598754


namespace area_CDE_l598_598045

variable (ABC : Type) [Triangle ABC] (A B C D E F : Point ABC)
variable (on_AC : On D AC) (on_BC : On E BC)
variable (intersect_AE_BD : Intersect AE BD F)
variable (area_ABF : Area (Triangle A B F) = 1)
variable (area_ADF : Area (Triangle A D F) = 1/3)
variable (area_BEF : Area (Triangle B E F) = 1/4)

theorem area_CDE
  (ABC : Triangle)
  (A B C D E F : Point ABC)
  (on_AC : On D AC)
  (on_BC : On E BC)
  (intersect_AE_BD : Intersect AE BD F)
  (area_ABF : Area (Triangle A B F) = 1)
  (area_ADF : Area (Triangle A D F) = 1/3)
  (area_BEF : Area (Triangle B E F) = 1/4)
  : Area (Triangle C D E) = 1/4 :=
sorry

end area_CDE_l598_598045


namespace cost_price_computer_table_l598_598088

theorem cost_price_computer_table (S : ℝ) (C : ℝ) (h1 : S = C * 1.15) (h2 : S = 5750) : C = 5000 :=
by
  sorry

end cost_price_computer_table_l598_598088


namespace Gargamel_bought_tires_l598_598683

def original_price_per_tire := 84
def sale_price_per_tire := 75
def total_savings := 36
def discount_per_tire := original_price_per_tire - sale_price_per_tire
def num_tires (total_savings : ℕ) (discount_per_tire : ℕ) := total_savings / discount_per_tire

theorem Gargamel_bought_tires :
  num_tires total_savings discount_per_tire = 4 :=
by
  sorry

end Gargamel_bought_tires_l598_598683


namespace sum_of_product_of_red_balls_l598_598592

theorem sum_of_product_of_red_balls :
  let balls := (List.range 999).map (fun n => 1 / (n + 2)) in
  let red_ball_products (r : List ℚ) := r.product in
  let even_combinations := List.filter (fun r => (r.length % 2 = 0) ∧ r.length > 0) (List.powerset balls) in
  let S := even_combinations.map red_ball_products in
  S.sum = 498501 / 2000 :=
by
  sorry

end sum_of_product_of_red_balls_l598_598592


namespace problem1_problem2_l598_598162

-- Problem 1:
theorem problem1 (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 :=
sorry

-- Problem 2:
theorem problem2 (α : ℝ) : 
  (Real.tan (2 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-α + Real.pi) * Real.sin (-Real.pi + α)) = 1 :=
sorry

end problem1_problem2_l598_598162


namespace cats_kill_rats_constant_time_l598_598751

-- Define the conditions and the conclusion in Lean

theorem cats_kill_rats_constant_time (n : ℕ) : 
  (∀ t : ℕ, 3 cats_time 3 rats_time t) ∧ (100 cats_time 100 rats_time 3) → 
  (n cats_time n rats_time 3) :=
sorry

-- Helper definitions
def cats_time (c t : ℕ) : Type := sorry -- This will represent 'c' cats taking 't' time to kill 'r' rats.
def rats_time (r t : ℕ) : Type := sorry -- This will represent 'r' rats killed in 't' time by 'c' cats.

end cats_kill_rats_constant_time_l598_598751


namespace area_of_triangle_segments_l598_598427

-- We define the basic context of the problem first
variables {A B C M : Point}
variables {a d : ℝ}

-- Given conditions
axiom is_equilateral (ABC : Triangle) (a : ℝ) : IsEquilateral ABC a
axiom distance_center (ABC : Triangle) (M : Point) (d : ℝ) : Distance (Center ABC) M = d

-- Definition of area
noncomputable def area (ABC : Triangle) : ℝ :=
  (sqrt 3) / 12 * |(side_length ABC)^2 - 3 * (distance_center ABC M)^2|

-- Problem statement, we need to prove this area equals given formula
theorem area_of_triangle_segments (ABC : Triangle) (M : Point) (a d : ℝ)
  (h_eq : is_equilateral ABC a) (h_dist : distance_center ABC M d) :
  let S := area (triangle_with_sides (distance MA) (distance MB) (distance MC)) in
  S = (sqrt 3) / 12 * |a^2 - 3*d^2| :=
sorry

end area_of_triangle_segments_l598_598427


namespace prism_volume_l598_598493

theorem prism_volume (a b : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : b < 2 * a) : 
  volume = (1/8) * a * b * sqrt(12 * a^2 - 3 * b^2) := sorry

end prism_volume_l598_598493


namespace locus_of_right_triangle_vertex_l598_598893

theorem locus_of_right_triangle_vertex (O : Point) (r : ℝ) (A B C : Point) (h₁ : ∀ B C : Point, dist B O = r ∧ dist C O = r) (h₂ : right_triangle A B C) : 
  ∃ R : ℝ, ∀ A : Point, ∃ I : Point, midpoint B C I ∧ dist O I = R → dist O A = R :=
begin
  sorry
end

end locus_of_right_triangle_vertex_l598_598893


namespace probability_f_times_f_zero_l598_598723

-- Define the function f
def f (x : ℕ) : ℝ := Real.sin (Real.pi * x / 6)

-- Define the set M
def M := {0, 1, 2, 3, 4, 5, 6, 7, 8}

-- Define the condition where f(m) * f(n) = 0
def f_times_f_zero (m n : ℕ) : Prop := f m * f n = 0

-- Define the total pairs
def total_pairs : Finset (ℕ × ℕ) := Finset.filter (λ p, p.1 ≠ p.2) (Finset.product M M)

-- Define the pairs where f(m) * f(n) = 0
def valid_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ p, f_times_f_zero p.1 p.2) total_pairs

-- Define the probability
noncomputable def probability : ℝ :=
  (Finset.card valid_pairs : ℝ) / (Finset.card total_pairs : ℝ)

-- Assert the probability is 5/12
theorem probability_f_times_f_zero :
  probability = 5 / 12 :=
sorry

end probability_f_times_f_zero_l598_598723


namespace area_CDE_l598_598043

variables (A B C D E F : Type)
variables [geometry_path A B C D E F]

-- Definitions for describing the problem
def is_on_line (X Y Z : Type) [segment X Y] : Prop := Z ∈ segment X Y
def area (T : Type) : ℝ := sorry -- some way to compute the area

-- Conditions
variables (hD_on_AC : is_on_line A C D)
variables (hE_on_BC : is_on_line B C E)
variables (hF_intersect : intersect_line A E B D = F)
variables (h_area_ABF : area (triangle A B F) = 1)
variables (h_area_ADF : area (triangle A D F) = 1/3)
variables (h_area_BEF : area (triangle B E F) = 1/4)

-- The goal
theorem area_CDE :
  area (triangle C D E) = 1/4 :=
begin
  sorry,
end

end area_CDE_l598_598043


namespace quadrilateral_parallelogram_l598_598852

open innerProductGeometry

/-- If the sum of the distances from any point P inside a convex quadrilateral ABCD to its sides 
AB, BC, CD, and DA is constant, then ABCD is a parallelogram. -/
theorem quadrilateral_parallelogram (ABCD : ConvexQuadrilateral) 
  (h : ∀ P ∈ interior ABCD, distance_sum_constant P ABCD) :
  isParallelogram ABCD :=
  sorry

end quadrilateral_parallelogram_l598_598852


namespace max_value_constraints_l598_598766

noncomputable def maximum_value (x y : ℝ) : ℝ := y / (x + 1)

theorem max_value_constraints (x y : ℝ) 
  (h1 : x - y ≤ 2) 
  (h2 : x + 2 * y ≥ 7) 
  (h3 : y ≤ 3) : 
  ∃ (z : ℝ), z = 1 ∧ ∀ (w : ℝ), (w = y / (x + 1)) → w ≤ z := 
begin
  sorry
end

end max_value_constraints_l598_598766


namespace total_animals_seen_l598_598258

theorem total_animals_seen (lions_sat : ℕ) (elephants_sat : ℕ) 
                           (buffaloes_sun : ℕ) (leopards_sun : ℕ)
                           (rhinos_mon : ℕ) (warthogs_mon : ℕ) 
                           (h_sat : lions_sat = 3 ∧ elephants_sat = 2)
                           (h_sun : buffaloes_sun = 2 ∧ leopards_sun = 5)
                           (h_mon : rhinos_mon = 5 ∧ warthogs_mon = 3) :
  lions_sat + elephants_sat + buffaloes_sun + leopards_sun + rhinos_mon + warthogs_mon = 20 := by
  sorry

end total_animals_seen_l598_598258


namespace distance_F_F_l598_598529

-- Define the coordinates of points F and F'
def F := (-4 : ℝ, 3 : ℝ)
def F' := (-4 : ℝ, -3 : ℝ)

-- Define a function for the distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the theorem to be proven
theorem distance_F_F' : distance F F' = 6 := by
  sorry

end distance_F_F_l598_598529


namespace find_set_for_congruent_product_l598_598859

theorem find_set_for_congruent_product (n : ℕ) (hn : n ≥ 2) :
  ∃ (A : Fin n → ℕ), (∀ i, (2 ≤ A i)) ∧ (∀ k, 1 ≤ k ∧ k ≤ n → 
  ((∏ (j : Fin n) in (Finset.univ.erase (Fin.kth n k).val), A j) % A (Fin.kth n k).val = 1 
    ∨ 
   (∏ (j : Fin n) in (Finset.univ.erase (Fin.kth n k).val), A j) % A (Fin.kth n k).val = -1)) :=
sorry

end find_set_for_congruent_product_l598_598859


namespace locus_of_M_full_rotation_locus_of_M_chord_slide_l598_598900

open EuclideanGeometry

noncomputable def point_locus_full_rotation (O A B C D M : Point) (h : Circle O B) :=
  (∃ (h₁ : Circle O A) (h₂ : Chord O C D), M ∈ line_of_points (h₁.intersection_line h₂)) →
  (∃ (locus : Circle O M), ∀ (θ : ℝ), rotation θ (diameter AB) → M ∈ locus.circumference)

noncomputable def point_locus_chord_slide (O A B C D M : Point) (h : Circle O B) :=
  (∃ (h₁ : Circle O A), ∀ (E F : Point), (chord O E F) → (∃ (line_1 : Line O A C) (line_2 : Line O B D),  M ∈ line_1 ∩ line_2)) →
  (∃ (locus : Circle O M), ∀ (E F : Point), (chord O E F) → M ∈ locus.circumference)

variable {O A B C D M : Point}

theorem locus_of_M_full_rotation (h₁ : Circle O A) (h₂ : Circle O B) (h₃ : Chord O C D) :
  point_locus_full_rotation O A B C D M h₁ :=
by {
  sorry
}

theorem locus_of_M_chord_slide (h₁ : Circle O A) (h₂ : Circle O B) :
  point_locus_chord_slide O A B C D M h₁ :=
by {
  sorry
}

end locus_of_M_full_rotation_locus_of_M_chord_slide_l598_598900


namespace imaginary_part_eq_neg2_l598_598291

-- Define a complex number z
variable (z : ℂ)

-- Define the condition given that the conjugate of z is z + 4i
def condition : Prop := z.conj = z + 4 * complex.I

-- State the theorem that the imaginary part of z is -2
theorem imaginary_part_eq_neg2 (z : ℂ) (h : condition z) : z.im = -2 :=
by
  sorry

end imaginary_part_eq_neg2_l598_598291


namespace trig_identity_l598_598704

open Real

theorem trig_identity (α : ℝ) (h_tan : tan α = 2) (h_quad : 0 < α ∧ α < π / 2) :
  sin (2 * α) + cos α = (4 + sqrt 5) / 5 :=
sorry

end trig_identity_l598_598704


namespace largest_n_all_truthful_l598_598099

-- Define the problem conditions and the final result
theorem largest_n_all_truthful (n : ℕ) (hn : n ≤ 99) :
  (∀ (initial_states : Fin n → ℤ), ∃ t : ℕ, ∀ i, reaches_state t (initial_states i) 1) → n = 64 :=
sorry

-- Auxiliary function to model the state transition
def reaches_state (t : ℕ) (x_i : ℤ) (target : ℤ) : Prop :=
  -- Definition of state reachability goes here, to be defined based on problem
  sorry

end largest_n_all_truthful_l598_598099


namespace brother_to_madeline_ratio_l598_598836

theorem brother_to_madeline_ratio (M B T : ℕ) (hM : M = 48) (hT : T = 72) (hSum : M + B = T) : B / M = 1 / 2 := by
  sorry

end brother_to_madeline_ratio_l598_598836


namespace fraction_simplification_l598_598866

def numerator : Int := 5^4 + 5^2 + 5
def denominator : Int := 5^3 - 2 * 5

theorem fraction_simplification :
  (numerator : ℚ) / (denominator : ℚ) = 27 + (14 / 23) := by
  sorry

end fraction_simplification_l598_598866


namespace monthly_pool_cost_is_correct_l598_598404

def cost_of_cleaning : ℕ := 150
def tip_percentage : ℕ := 10
def number_of_cleanings_in_a_month : ℕ := 30 / 3
def cost_of_chemicals_per_use : ℕ := 200
def number_of_chemical_uses_in_a_month : ℕ := 2

def monthly_cost_of_pool : ℕ :=
  let cost_per_cleaning := cost_of_cleaning + (cost_of_cleaning * tip_percentage / 100)
  let total_cleaning_cost := number_of_cleanings_in_a_month * cost_per_cleaning
  let total_chemical_cost := number_of_chemical_uses_in_a_month * cost_of_chemicals_per_use
  total_cleaning_cost + total_chemical_cost

theorem monthly_pool_cost_is_correct : monthly_cost_of_pool = 2050 :=
by
  sorry

end monthly_pool_cost_is_correct_l598_598404


namespace expression_value_l598_598922

theorem expression_value :
  (35 + 12) ^ 2 - (12 ^ 2 + 35 ^ 2 - 2 * 12 * 35) = 1680 :=
by
  sorry

end expression_value_l598_598922


namespace birth_rate_calculation_l598_598499

theorem birth_rate_calculation (D : ℕ) (G : ℕ) (P : ℕ) (NetGrowth : ℕ) (B : ℕ) (h1 : D = 16) (h2 : G = 12) (h3 : P = 3000) (h4 : NetGrowth = G * P / 100) (h5 : NetGrowth = B - D) : B = 52 := by
  sorry

end birth_rate_calculation_l598_598499


namespace area_of_T_l598_598813

def omega : ℂ := -1 / 2 + (1 / 2) * complex.I * real.sqrt 3

def T (a b c : ℝ) : ℂ := a + b * omega + c * (conj omega)

theorem area_of_T : 
  (∀ a b c, 0 ≤ a ∧ a ≤ 0.5 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 → T a b c ∈ set.univ ℂ) →
  ∀ (A : ℝ), (A = real.sqrt 3 / 4 * 3) := 
sorry

end area_of_T_l598_598813


namespace find_g_inv_84_l598_598753

def g (x : ℝ) : ℝ := 3 * x ^ 3 + 3

theorem find_g_inv_84 (x : ℝ) (h : g x = 84) : x = 3 :=
by 
  unfold g at h
  -- Begin proof steps here, but we will use sorry to denote placeholder 

  sorry

end find_g_inv_84_l598_598753


namespace sun_salutations_per_year_l598_598070

theorem sun_salutations_per_year :
  (∀ S : Nat, S = 5) ∧
  (∀ W : Nat, W = 5) ∧
  (∀ Y : Nat, Y = 52) →
  ∃ T : Nat, T = 1300 :=
by 
  sorry

end sun_salutations_per_year_l598_598070


namespace frankie_candies_l598_598680

theorem frankie_candies (M D F : ℕ) (h1 : M = 92) (h2 : D = 18) (h3 : F = M - D) : F = 74 :=
by
  sorry

end frankie_candies_l598_598680


namespace quadratic_polynomial_l598_598669

theorem quadratic_polynomial (q : ℝ → ℝ)
  (h1 : q (-2) = 0)
  (h2 : q (3) = 0)
  (h3 : q (1) = -24) :
  q = (λ x, 4 * x^2 - 4 * x - 24) :=
by
  sorry

end quadratic_polynomial_l598_598669


namespace sophia_age_eight_years_later_l598_598783

variables (J S I So L O E : ℕ)

def conditions :=
  J = 40 ∧
  S = J + 4 ∧
  I = S - 3 ∧
  So = 2 * L ∧
  L = J - 5 ∧
  O = I ∧
  E = O / 2 ∧
  (J + 6 + S + 6 + I + 6 + So + 6 + L + 6 + O + 6 + E + 6 = 495) ∧
  (J + 2 + S + 2 + I + 2 = 150)

theorem sophia_age_eight_years_later (h : conditions) : So + 8 = 78 :=
sorry

end sophia_age_eight_years_later_l598_598783


namespace solve_fractional_equation_1_solve_fractional_equation_2_l598_598870

-- Proof Problem 1
theorem solve_fractional_equation_1 (x : ℝ) (h : 6 * x - 2 ≠ 0) :
  (3 / 2 - 1 / (3 * x - 1) = 5 / (6 * x - 2)) ↔ (x = 10 / 9) :=
sorry

-- Proof Problem 2
theorem solve_fractional_equation_2 (x : ℝ) (h1 : 3 * x - 6 ≠ 0) :
  (5 * x - 4) / (x - 2) = (4 * x + 10) / (3 * x - 6) - 1 → false :=
sorry

end solve_fractional_equation_1_solve_fractional_equation_2_l598_598870


namespace problem_correct_statements_l598_598246

def T (a b x y : ℚ) : ℚ := a * x * y + b * x - 4

theorem problem_correct_statements (a b : ℚ) (h₁ : T a b 2 1 = 2) (h₂ : T a b (-1) 2 = -8) :
  (a = 1 ∧ b = 2) ∧
  (∀ m n : ℚ, T 1 2 m n = 0 ∧ n ≠ -2 → m = 4 / (n + 2)) ∧
  ¬ (∃ m n : ℤ, T 1 2 m n = 0 ∧ n ≠ -2 ∧ m + n = 3) ∧
  (∀ k x y : ℚ, T 1 2 (k * x) y = T 1 2 (k * x) y → y = -2) ∧
  (∀ k x y : ℚ, x ≠ y → T 1 2 (k * x) y = T 1 2 (k * y) x → k = 0) :=
by
  sorry

end problem_correct_statements_l598_598246


namespace steven_name_day_44_l598_598137

def W (n : ℕ) : ℕ :=
  2 * (n / 2) + 4 * ((n - 1) / 2)

theorem steven_name_day_44 : ∃ n : ℕ, W n = 44 :=
  by 
  existsi 16
  sorry

end steven_name_day_44_l598_598137


namespace population_change_factors_l598_598494

theorem population_change_factors (natural_growth migration mortality_rate natural_increase birth_rate : Prop) :
  (population_change : Prop) :=
  population_change ↔ natural_growth ∧ migration
  sorry

end population_change_factors_l598_598494


namespace neg_p_sufficient_not_necessary_neg_q_l598_598817

variable (x : ℝ)

def p : Prop := x < -1 ∨ x > 1
def q : Prop := x < -2 ∨ x > 1

theorem neg_p_sufficient_not_necessary_neg_q : (¬ p) → (¬ q) :=
by
  sorry

end neg_p_sufficient_not_necessary_neg_q_l598_598817


namespace simplify_expression_l598_598867

theorem simplify_expression (x : ℤ) (h1 : 2 * (x - 1) < x + 1) (h2 : 5 * x + 3 ≥ 2 * x) :
  (x = 2) → (2 / (x^2 + x) / (1 - (x - 1) / (x^2 - 1)) = 1 / 2) :=
by
  sorry

end simplify_expression_l598_598867


namespace a_perfect_square_l598_598802

theorem a_perfect_square (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_div : 2 * a * b ∣ a^2 + b^2 - a) : ∃ k : ℕ, a = k^2 := 
sorry

end a_perfect_square_l598_598802


namespace sum_even_coefficients_l598_598684

theorem sum_even_coefficients (n : ℕ) (a : ℕ → ℤ) :
  (1 + x + x^2)^n = ∑ i in (Finset.range (2 * n + 1)), a i * x^i →
  (∑ i in (Finset.range (2 * n + 1)).filter even, a i) = (3^n + 1) / 2 :=
by sorry

end sum_even_coefficients_l598_598684


namespace apples_in_pile_l598_598104

theorem apples_in_pile (initial_apples added_apples : ℕ) (h₀ : initial_apples = 8) (h₁ : added_apples = 5) : initial_apples + added_apples = 13 :=
by {
  rw [h₀, h₁],
  norm_num,
  sorry,
}

end apples_in_pile_l598_598104


namespace ratio_of_areas_l598_598532

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l598_598532


namespace c_value_of_parabola_l598_598964

theorem c_value_of_parabola : 
  (∀ b c : ℝ, (1^2 + b * 1 + c = 6) ∧ (5^2 + b * 5 + c = 10)) → (c = 10) :=
by
  assume h,
  sorry

end c_value_of_parabola_l598_598964


namespace distance_between_intersections_is_sqrt3_l598_598784

noncomputable def intersection_distance : ℝ :=
  let C1_polar := (θ : ℝ) → θ = (2 * Real.pi / 3)
  let C2_standard := (x y : ℝ) → (x + Real.sqrt 3)^2 + (y + 2)^2 = 1
  let C3 := (θ : ℝ) → θ = (Real.pi / 3) 
  let C3_cartesian := (x y : ℝ) → y = Real.sqrt 3 * x
  let center := (-Real.sqrt 3, -2)
  let dist_to_C3 := abs (-3 + 2) / 2
  2 * Real.sqrt (1 - (dist_to_C3)^2)

theorem distance_between_intersections_is_sqrt3:
  intersection_distance = Real.sqrt 3 := by
  sorry

end distance_between_intersections_is_sqrt3_l598_598784


namespace frank_maze_time_l598_598679

theorem frank_maze_time 
    (n mazes : ℕ)
    (avg_time_per_maze completed_time total_allowable_time remaining_maze_time extra_time_inside current_time : ℕ) 
    (h1 : mazes = 5)
    (h2 : avg_time_per_maze = 60)
    (h3 : completed_time = 200)
    (h4 : total_allowable_time = mazes * avg_time_per_maze)
    (h5 : total_allowable_time = 300)
    (h6 : remaining_maze_time = total_allowable_time - completed_time) 
    (h7 : extra_time_inside = 55)
    (h8 : current_time + extra_time_inside ≤ remaining_maze_time) :
  current_time = 45 :=
by
  sorry

end frank_maze_time_l598_598679


namespace basketball_team_total_points_l598_598159

theorem basketball_team_total_points :
  ∀ (points : ℕ → ℕ), 
    points 1 = 7 ∧ points 2 = 8 ∧ points 3 = 2 ∧ points 4 = 11 ∧ 
    points 5 = 6 ∧ points 6 = 12 ∧ points 7 = 1 ∧ points 8 = 7 →
    (Σ i in finrange 8, points (i + 1)) = 54 :=
by
  intro points
  assume h : points 1 = 7 ∧ points 2 = 8 ∧ points 3 = 2 ∧ points 4 = 11 ∧ 
             points 5 = 6 ∧ points 6 = 12 ∧ points 7 = 1 ∧ points 8 = 7
  sorry

end basketball_team_total_points_l598_598159


namespace minimize_sum_of_distances_l598_598396

open_locale real

-- Define the circle and points A, B
variables {O A B M : Point}
variables (circle : Circle O r)
variables (N : Point) -- Midpoint of AB

-- Define the conditions: OA = OB and M is on the circle
def is_midpoint (A B N : Point) : Prop :=
  dist A N = dist B N ∧ 2 * dist A N = dist A B

def point_on_circle (circle : Circle O r) (M : Point) : Prop :=
  dist O M = r

def minimizes_distance_sum (M A B : Point) : Prop :=
  ∀ M' : Point, M' ∈ circle → (dist M A + dist M B) ≤ (dist M' A + dist M' B)

-- The statement to prove
theorem minimize_sum_of_distances
  (h1 : dist O A = dist O B)
  (h2 : is_midpoint A B N)
  (M_in_circle : point_on_circle circle M)
  (N_midpoint : N = midpoint A B) :
  minimizes_distance_sum M A B :=
sorry

end minimize_sum_of_distances_l598_598396


namespace diplomats_spoke_both_percentage_l598_598450

theorem diplomats_spoke_both_percentage (T F H B : ℕ) 
  (hT : T = 120)
  (hF : F = 20)
  (h_non_Hindi : T - H = 32)
  (h_neither : 0.20 * T = 24)
  : (B / T) * 100 = 10 := 
sorry

end diplomats_spoke_both_percentage_l598_598450


namespace max_expression_value_l598_598503

theorem max_expression_value : 
  ∃ a b c d e f : ℕ, 1 ≤ a ∧ a ≤ 6 ∧
                   1 ≤ b ∧ b ≤ 6 ∧
                   1 ≤ c ∧ c ≤ 6 ∧
                   1 ≤ d ∧ d ≤ 6 ∧
                   1 ≤ e ∧ e ≤ 6 ∧
                   1 ≤ f ∧ f ≤ 6 ∧
                   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                   d ≠ e ∧ d ≠ f ∧
                   e ≠ f ∧
                   (f * (a * d + b * c) / (b * d * e) = 14) :=
sorry

end max_expression_value_l598_598503


namespace problem_l598_598685

theorem problem :
  ∃ (z : ℂ), (1 - complex.I)^2 = (1 + complex.I) * z ∧ complex.conj z = -1 + complex.I :=
by
  sorry

end problem_l598_598685


namespace digit_in_tens_place_of_smallest_even_number_l598_598112

   theorem digit_in_tens_place_of_smallest_even_number :
     ∃ n : ℕ, (∀ d, d ∈ [1, 3, 5, 6, 8] → n % 10 ∈ {6, 8}) ∧ 
              (∀ k ∈ [1, 3, 5, 6, 8], n / 10^k % 10 = k) ∧ 
              (∀ m : ℕ, (∀ d, d ∈ [1, 3, 5, 6, 8] → m % 10 ∈ {6, 8}) → 
                       (∀ k ∈ [1, 3, 5, 6, 8], m / 10^k % 10 = k) → 
                       n ≤ m) ∧ 
              ((n / 10 % 10) = 8) :=
   sorry
   
end digit_in_tens_place_of_smallest_even_number_l598_598112


namespace sum_even_factors_630_l598_598132

theorem sum_even_factors_630 : 
  (∑ n in (finset.filter (λ n, even n) (divisors 630)), n) = 1248 := 
sorry

end sum_even_factors_630_l598_598132


namespace bob_orders_muffins_per_day_l598_598229

theorem bob_orders_muffins_per_day (x : ℕ) 
  (h1 : ∀ (x : ℕ), 0.75 * 7 * x = 63) : x = 12 :=
by {
  sorry
}

end bob_orders_muffins_per_day_l598_598229


namespace area_triangle_XYZ_eq_one_fourth_l598_598982

variables (A B C D X Y Z : ℝ^2)
variables (a b : ℝ^3)
variables (h k : ℝ)

def midpoint (P Q : ℝ^2) : ℝ^2 := (P + Q) / 2

-- Given conditions as hypotheses
hypothesis (convex_ABCD : convex_hull ℝ ({A, B, C, D} : set (ℝ^2))) 
hypothesis (area_ABCD_eq_one : (area ℝ (λ i, [A, B, C, D] i) = 1))
hypothesis (lines_intersection_AD_BC : X = line_intersection (through ℝ A D) (through ℝ B C))
hypothesis (midpoints_Y_Z : Y = midpoint A C ∧ Z = midpoint B D)

-- The theorem to prove
theorem area_triangle_XYZ_eq_one_fourth :
  (area ℝ (λ i, [X, Y, Z] i) = 1 / 4) :=
sorry

end area_triangle_XYZ_eq_one_fourth_l598_598982


namespace complex_right_triangle_l598_598820

open Complex

theorem complex_right_triangle {z1 z2 a b : ℂ}
  (h1 : z2 = I * z1)
  (h2 : z1 + z2 = -a)
  (h3 : z1 * z2 = b) :
  a^2 / b = 2 :=
by sorry

end complex_right_triangle_l598_598820


namespace banker_speed_ratio_l598_598946

noncomputable def required_ratio {V_b V_c : ℝ} (Th : 55) (T5 : 10) : Prop :=
  (V_b * 60 = V_c * 5) → (V_c = 12 * V_b)

theorem banker_speed_ratio : ∀ (V_b V_c : ℝ), required_ratio 55 10 → V_c = 12 * V_b :=
by {
  intros V_b V_c h,
  have ratio := h,
  sorry, -- Proof omitted
}

end banker_speed_ratio_l598_598946


namespace problem1_problem2_problem3_problem4_l598_598140

-- Problem 1: Prove that the line y = ax - 2a + 4 passes through the point (2, 4) for any a in ℝ
theorem problem1 (a : ℝ) : ∃ (x y : ℝ), (x, y) = (2, 4) ∧ y = a * x - 2 * a + 4 :=
by
  use 2
  use 4
  split
  . rfl
  calc
    4 = a * 2 - 2 * a + 4 : by linarith

-- Problem 2: Prove that the y-intercept of the line y + 1 = 3x is not 1
theorem problem2 : ¬∃ y : ℝ, y = 1 ∧ y + 1 = 3 * 0 :=
by
  rintro ⟨y, hy, _⟩
  linarith

-- Problem 3: Prove that the slope of the line x + √3y + 1 = 0 is not such that it makes an angle of 120° with the positive direction of the x-axis
theorem problem3 : ¬(∃ (α : ℝ), α = 120 ∧ let m := -1 / (Real.sqrt 3) in tan α = m) :=
by
  intro h
  rcases h with ⟨α, ha, integration⟩
  rw [ha] at integration
  sorry  -- complete proof calculation using trigonometric properties

-- Problem 4: Prove that the equation of the line passing through the point (-2, 3) and perpendicular to the line x - 2y + 3 = 0 is 2x + y + 1 = 0
theorem problem4 (x y: ℝ) : (x, y) = (-2, 3) ∧ (∃ (m : ℝ), m = -2 ∧ y - 3 = m * (x + 2)) → 2 * x + y + 1 = 0 :=
by
  intro ⟨hx, hz⟩
  rcases hz with ⟨m, hm, h⟩
  rw [hm] at h
  calc
    2 * -2 + 3 + 1 = 0 : by linarith  -- since LHS simplifies to 0, RHS should be 0

end problem1_problem2_problem3_problem4_l598_598140


namespace mean_median_mode_equal_l598_598553

theorem mean_median_mode_equal :
  let data := [5, 0, 0, 1, 1, 2, 2, 2, 5]
  let mean := data.sum / data.length
  let median := data.nth (data.length / 2)
  let mode := data.mode
  mean = median ∧ median = mode :=
by
  sorry

end mean_median_mode_equal_l598_598553


namespace symmetric_point_in_first_quadrant_l598_598387

structure Point :=
  (x : ℝ)
  (y : ℝ)

def reflect_across_y_axis (P : Point) : Point :=
  { x := -P.x, y := P.y }

def quadrant (P : Point) : String :=
  if P.x > 0 ∧ P.y > 0 then "First quadrant"
  else if P.x < 0 ∧ P.y > 0 then "Second quadrant"
  else if P.x < 0 ∧ P.y < 0 then "Third quadrant"
  else if P.x > 0 ∧ P.y < 0 then "Fourth quadrant"
  else "On axis"

theorem symmetric_point_in_first_quadrant : 
  quadrant (reflect_across_y_axis {x := -3, y := 1}) = "First quadrant" :=
by
  sorry

end symmetric_point_in_first_quadrant_l598_598387


namespace convex_and_concave_is_affine_l598_598860

-- Definitions needed for the problem
def isConvex (I : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ (x y ∈ I) (λ ∈ Set.Icc 0 1), f (λ * x + (1 - λ) * y) ≤ λ * f x + (1 - λ) * f y

def isConcave (I : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ (x y ∈ I) (λ ∈ Set.Icc 0 1), f (λ * x + (1 - λ) * y) ≥ λ * f x + (1 - λ) * f y

def isAffine (I : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x ∈ I, f x = a * x + b

-- The theorem statement
theorem convex_and_concave_is_affine (I : Set ℝ) (f : ℝ → ℝ) (hI : ∃ a b c d : ℝ, I = Set.Icc a b ∧ c ≤ d) :
  isConvex I f → isConcave I f → isAffine I f :=
by
  intros hConvex hConcave
  sorry

end convex_and_concave_is_affine_l598_598860


namespace find_x_l598_598421

noncomputable def h (x : ℝ) : ℝ := (2 * x^2 + 3 * x + 1)^(1 / 3) / 5^(1/3)

theorem find_x (x : ℝ) :
  h (3 * x) = 3 * h x ↔ x = -1 + (10^(1/2)) / 3 ∨ x = -1 - (10^(1/2)) / 3 := by
  sorry

end find_x_l598_598421


namespace sum_of_other_endpoint_l598_598051

theorem sum_of_other_endpoint (x y : ℝ) (h1 : (6 + x) / 2 = 3) (h2 : (-2 + y) / 2 = 5) : x + y = 12 := 
by {
  sorry
}

end sum_of_other_endpoint_l598_598051


namespace ratio_of_areas_of_triangles_l598_598548

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l598_598548


namespace monotonic_decreasing_interval_l598_598085

noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 1) * Real.exp x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (-2 < x ∧ x < -1) -> (f'(x) < 0) :=
sorry

end monotonic_decreasing_interval_l598_598085


namespace light_distance_in_50_years_correct_l598_598080

variables (distance_per_year : ℕ)
def light_travel_distance_in_50_years (d : ℕ) : ℕ := d * 50

theorem light_distance_in_50_years_correct (h : distance_per_year = 5870 * 10^9) :
  light_travel_distance_in_50_years distance_per_year = 2935 * 10^11 :=
by {
  -- Start with the equation for distance in one year
  rw h,
  -- Express in scientific notation
  have h2 : 5870 * 10^9 = 587 * 10^10 := by norm_num,
  rw h2,
  -- Calculate the distance for 50 years
  have h3 : (587 * 10^10) * 50 = 2935 * 10^11 := by norm_num,
  exact h3,
}

end light_distance_in_50_years_correct_l598_598080


namespace number_of_real_a_l598_598676

open Int

-- Define the quadratic equation with integer roots
def quadratic_eq_with_integer_roots (a : ℝ) : Prop :=
  ∃ (r s : ℤ), r + s = -a ∧ r * s = 12 * a

-- Prove there are exactly 9 values of a such that the quadratic equation has only integer roots
theorem number_of_real_a (n : ℕ) : n = 9 ↔ ∃ (as : Finset ℝ), as.card = n ∧ ∀ a ∈ as, quadratic_eq_with_integer_roots a :=
by
  -- We can skip the proof with "sorry"
  sorry

end number_of_real_a_l598_598676


namespace coffee_price_ratio_l598_598026

-- Define original prices of coffee A and B
def original_price_A : ℝ := 50
def original_price_B : ℝ := 40

-- Define the price increase rate for coffee A and the decrease rate for coffee B
def price_increase_rate_A : ℝ := 0.10
def price_decrease_rate_B : ℝ := -0.15

-- Define the new prices of coffee A and B after adjustments
def new_price_A := original_price_A * (1 + price_increase_rate_A)
def new_price_B := original_price_B * (1 + price_decrease_rate_B)

-- Define the function for the price of mixed coffee
def mixed_price (x y : ℝ) := (50 * x + 40 * y) / (x + y)
def new_mixed_price (x y : ℝ) := (55 * x + 34 * y) / (x + y)

-- Define the goal: proving the ratio x : y = 6 : 5
theorem coffee_price_ratio (x y : ℝ) (h : mixed_price x y = new_mixed_price x y) : x / y = 6 / 5 :=
by
  sorry

end coffee_price_ratio_l598_598026


namespace C1_cartesian_eq_C2_cartesian_eq_distance_MN_range_l598_598386

-- Definition of curve C1
def curve_C1 (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, Real.sin φ)

-- Definition of curve C2
def curve_C2 (r θ : ℝ) : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)

-- 1. Prove Cartesian equation for C1
theorem C1_cartesian_eq : ∀ x y, (∃ φ, x = 2 * Real.cos φ ∧ y = Real.sin φ) ↔ (x^2 / 4 + y^2 = 1) :=
by
  sorry

-- 2. Prove Cartesian equation for C2
theorem C2_cartesian_eq : ∀ x y, (∃ θ, ∃ r, r = 3 ∧ θ = Float.pi / 2 ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) ↔ (x^2 + (y - 3)^2 = 1) :=
by
  sorry

-- 3. Prove range of distance MN
theorem distance_MN_range : ∀ M N,
  (∃ φ, M = (2 * Real.cos φ, Real.sin φ) ∧ ∃ θ, θ = Float.pi / 2 ∧ N = (3 * Real.cos θ, 3 * Real.sin θ) ∧ sqrt ((2 * Real.cos φ - 3 * Real.cos θ) ^ 2 + (Real.sin φ - 3 * Real.sin θ) ^ 2) ≥ 1 
   ∧ sqrt ((2 * Real.cos φ - 3 * Real.cos θ) ^ 2 + (Real.sin φ - 3 * Real.sin θ) ^ 2) ≤ 5) :=
by
  sorry

end C1_cartesian_eq_C2_cartesian_eq_distance_MN_range_l598_598386


namespace points_on_same_circle_l598_598282

variables {A B C H3 A1 B1: Point}

-- Conditions:
variable (triangle_ABC : ABC)
variable (height_CH3 : CH_3)
variable (perpendiculars_H3A1_H3B1 : H_3A_1 ∧ H_3B_1)
variable (right_angles_A1C_B1C : ∠H3A1C = 90 ∧ ∠H3B1C = 90)

-- Proof Goal:
theorem points_on_same_circle :
  cyclic_quadrilateral A B A1 B1 :=
sorry

end points_on_same_circle_l598_598282


namespace first_digit_base9_650_l598_598918

theorem first_digit_base9_650 : ∃ d : ℕ, 
  d = 8 ∧ (∃ k : ℕ, 650 = d * 9^2 + k ∧ k < 9^2) :=
by {
  sorry
}

end first_digit_base9_650_l598_598918


namespace simplify_sum_of_polynomials_l598_598464

-- Definitions of the given polynomials
def P (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 8 * x + 15
def Q (x : ℝ) : ℝ := -5 * x^4 - 2 * x^3 + 3 * x^2 + 8 * x + 9

-- Statement to prove that the sum of P and Q equals the simplified polynomial
theorem simplify_sum_of_polynomials (x : ℝ) : 
  P x + Q x = 2 * x^5 - 8 * x^4 - x^3 + 8 * x^2 + 24 := 
sorry

end simplify_sum_of_polynomials_l598_598464


namespace find_phi_l598_598428

variable (a : ℝ) {D1 D2 : Point}

def eq_dist (D1 D2 : Point) (plane : Plane) : Prop := 
  abs (dist_to_plane D1 plane) = abs (dist_to_plane D2 plane)

theorem find_phi (h1 : ∀ (line: Line), ∃ plane, angle plane ABC_plane = φ ∧ intersects plane line = D1)
                (h2 : ∀ (line: Line), ∃ plane, angle plane ABC_plane = 2 * φ ∧ intersects plane line = D2)
                (h3 : eq_dist D1 D2 ABC_plane) :
  φ = 30 :=
by
  -- the proof steps would go here
  sorry

end find_phi_l598_598428


namespace inequality_proof_l598_598862

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (x / (y + z + 1)) + (y / (z + x + 1)) + (z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) :=
sorry

end inequality_proof_l598_598862


namespace count_valid_triples_l598_598747

noncomputable def num_valid_triples : ℕ :=
  let lattice_points := { p : ℕ × ℕ // p.1 ≤ 4 ∧ p.2 ≤ 4 }
  let triples := { t : lattice_points × lattice_points × lattice_points // t.1 ≠ t.2 ∧ t.1 ≠ t.3 ∧ t.2 ≠ t.3 }

  let is_valid (t : triples) : Prop :=
    let p1 := t.1.1
    let p2 := t.1.2
    let p3 := t.1.3
    let area := | (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) |
    (2 * area) % 5 = 0

  (fintype.card { t : triples // is_valid t })

theorem count_valid_triples : num_valid_triples = 300 := 
  sorry

end count_valid_triples_l598_598747


namespace find_a2_b2_l598_598687

theorem find_a2_b2 (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 16 = 10 * a * b) : a^2 + b^2 = 8 :=
by
  sorry

end find_a2_b2_l598_598687


namespace relay_race_team_members_l598_598940

theorem relay_race_team_members (n : ℕ) (d : ℕ) (h1 : n = 5) (h2 : d = 150) : d / n = 30 := 
by {
  -- Place the conditions here as hypotheses
  sorry
}

end relay_race_team_members_l598_598940


namespace students_identified_chess_or_basketball_l598_598777

theorem students_identified_chess_or_basketball (total_students : ℕ) (p_basketball : ℝ) (p_chess : ℝ) (p_soccer : ℝ) :
    total_students = 250 → 
    p_basketball = 0.4 → 
    p_chess = 0.1 →
    p_soccer = 0.28 → 
    (p_basketball * total_students + p_chess * total_students) = 125 :=
begin 
  intros h1 h2 h3 h4,
  sorry
end

end students_identified_chess_or_basketball_l598_598777


namespace seashells_total_l598_598460

theorem seashells_total :
  let sally := 9.5
  let tom := 7.2
  let jessica := 5.3
  let alex := 12.8
  sally + tom + jessica + alex = 34.8 :=
by
  sorry

end seashells_total_l598_598460


namespace intersection_of_lines_l598_598561

theorem intersection_of_lines :
  ∃ (x y : ℚ), y = -3 * x + 1 ∧ y + 1 = 7 * x ∧ x = 1/5 ∧ y = 2/5 :=
by
  use 1/5, 2/5
  simp
  split
  sorry

end intersection_of_lines_l598_598561


namespace quadratic_solution_l598_598516

theorem quadratic_solution (x : ℝ) :
  (x^2 + 2 * x = 0) ↔ (x = 0 ∨ x = -2) :=
by
  sorry

end quadratic_solution_l598_598516


namespace orthographic_projection_cube_sphere_l598_598626

-- Define the condition: orthographic projections for cube, sphere, and cone
def orthographic_projections (shape : Type) (view : Type) :=
  Π (s : shape), view

-- Define specific views for each shape, assume visual representations as types
inductive Shape
| cube
| sphere
| cone

inductive View
| square
| circle
| isosceles_triangle

-- The orthographic projections for each shape
def cube_projection (view : View) : Prop :=
  match view with
  | View.square => true
  | _ => false

def sphere_projection (view : View) : Prop :=
  match view with
  | View.circle => true
  | _ => false

def cone_projection (view : View) : Type :=
  match view with
  | View.isosceles_triangle => Unit
  | View.circle => Unit
  | _ => false

-- The theorem to be proved
theorem orthographic_projection_cube_sphere :
    (∀ (v : View), cube_projection v = true) ∧
    (∀ (v : View), sphere_projection v = true) ∧
    (∃ (v1 v2 : View), cone_projection v1 ≠ cone_projection v2) :=
by
  sorry

end orthographic_projection_cube_sphere_l598_598626


namespace train_speed_l598_598618

theorem train_speed (length : ℝ) (time : ℝ) (length_is_125 : length = 125) (time_is_7.5 : time = 7.5) :
  (length / time) * (3600 / 1000) = 60 :=
by
  sorry

end train_speed_l598_598618


namespace part1_fixed_point_part2_range_of_a_l598_598274

-- Part 1
theorem part1_fixed_point :
  let f := (λ x : ℝ, Real.log 2 ((2 : ℝ)^(x - 1) + 1/4))
  in f (-1) = -1 := 
sorry

-- Part 2
theorem part2_range_of_a :
  let f := (λ x a : ℝ, Real.log 2 (a * 4^(x - 1/2) - (a - 1) * 2^(x - 1) + a / 2 + 1 / 4))
  in ∀ a ∈ set.Ioo (1/2:ℝ) (Real.sqrt 3 / 3), ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ ∀ x, f x a = x ↔ x = x1 ∨ x = x2 :=
sorry

end part1_fixed_point_part2_range_of_a_l598_598274


namespace point_circle_relationship_l598_598277

theorem point_circle_relationship (a : ℝ) : 
  let P := (a, 2 - a) in 
  (P.1^2 + P.2^2 ≥ 1) :=
by
  sorry

end point_circle_relationship_l598_598277


namespace systematic_sampling_probability_l598_598554

-- Define the parameters
def total_parts : Nat := 120
def sample_size : Nat := 20

-- Define the probability function for systematic sampling
def probability (total parts sample_size) : Rat :=
  (sample_size : Rat) / (total_parts : Rat)

-- The theorem we want to prove
theorem systematic_sampling_probability :
  probability total_parts sample_size = 1 / 6 :=
by
  sorry

end systematic_sampling_probability_l598_598554


namespace number_of_real_a_l598_598675

open Int

-- Define the quadratic equation with integer roots
def quadratic_eq_with_integer_roots (a : ℝ) : Prop :=
  ∃ (r s : ℤ), r + s = -a ∧ r * s = 12 * a

-- Prove there are exactly 9 values of a such that the quadratic equation has only integer roots
theorem number_of_real_a (n : ℕ) : n = 9 ↔ ∃ (as : Finset ℝ), as.card = n ∧ ∀ a ∈ as, quadratic_eq_with_integer_roots a :=
by
  -- We can skip the proof with "sorry"
  sorry

end number_of_real_a_l598_598675


namespace equal_discriminants_of_monic_quadratic_trinomials_l598_598430

-- Given conditions
variables {α : Type*} [LinearOrderedField α] {P Q : α → α}
variables {a1 a2 b1 b2 : α}
variable (hP_monic : isMonic P)
variable (hQ_monic : isMonic Q)
variable (hP_roots : roots P = [a1, a2])
variable (hQ_roots : roots Q = [b1, b2])
variable (h_distinct_roots_P : a1 ≠ a2)
variable (h_distinct_roots_Q : b1 ≠ b2)
variable (h_condition : Q a1 + Q a2 = P b1 + P b2)

-- Define the statement to be proved
theorem equal_discriminants_of_monic_quadratic_trinomials :
  discrim P = discrim Q :=
sorry

end equal_discriminants_of_monic_quadratic_trinomials_l598_598430


namespace cos_gamma_eq_l598_598413

-- Define the conditions and the problem setup
variables (x y z : ℝ)
variables (α β γ : ℝ)
variables (positive_coords : x > 0 ∧ y > 0 ∧ z > 0)
variables (hα : cos α = 1 / 3)
variables (hβ : cos β = 1 / 5)
variables (hα_geom : cos α = x / sqrt (x^2 + y^2 + z^2))
variables (hβ_geom : cos β = y / sqrt (x^2 + y^2 + z^2))
variables (hγ_geom : cos γ = z / sqrt (x^2 + y^2 + z^2))

-- The theorem statement you want to prove
theorem cos_gamma_eq : cos γ = sqrt 191 / 15 :=
sorry

end cos_gamma_eq_l598_598413


namespace min_value_expr_l598_598426

noncomputable theory
open Real

theorem min_value_expr (y : ℝ) (hy : 0 < y) : ∃ m, m = 3 * y^4 + 4 * y^(-3) ∧ m = 7 :=
by sorry

end min_value_expr_l598_598426


namespace analogical_reasoning_l598_598926

-- Definitions based on conditions
def line_tangent_to_circle_perpendicular (circle_center: Point) (tangent_point: Point) (tangent_line: Line) : Prop :=
  tangent (circle_center, tangent_point) ∧ perpendicular (line_from_points circle_center tangent_point, tangent_line)

def plane_tangent_to_sphere_perpendicular (sphere_center: Point) (tangent_point: Point) (tangent_plane: Plane) : Prop :=
  tangent (sphere_center, tangent_point) ∧ perpendicular (line_from_points sphere_center tangent_point, tangent_plane)

-- Proof problem
theorem analogical_reasoning (circle_center sphere_center tangent_point: Point)
  (tangent_line: Line) (tangent_plane: Plane) :
  line_tangent_to_circle_perpendicular circle_center tangent_point tangent_line →
  plane_tangent_to_sphere_perpendicular sphere_center tangent_point tangent_plane →
  reasoning_used = analogical_reasoning :=
by
  sorry

end analogical_reasoning_l598_598926


namespace integer_polynomial_at_2005_l598_598009

theorem integer_polynomial_at_2005 (P : ℤ[X]) 
  (h1 : P.eval 1997 = 0) 
  (h2 : P.eval 2010 = 0) 
  (h3 : abs (P.eval 2005) < 10) : 
  P.eval 2005 = 0 := 
sorry

end integer_polynomial_at_2005_l598_598009


namespace area_of_square_diagonal_length_2_l598_598758

theorem area_of_square_diagonal_length_2 (d : ℝ) (h : d = 2) : ∃ A, A = 2 :=
by
  let s := d / Real.sqrt 2
  have : s = Real.sqrt 2, sorry
  let A := s ^ 2
  use A
  have : A = 2, sorry
  exact this

end area_of_square_diagonal_length_2_l598_598758


namespace russian_tennis_players_probability_l598_598472

theorem russian_tennis_players_probability :
  let total_players := 10
  let russian_players := 4
  let target_probability := (1 : ℚ) / 21
  probability_all_russian_pair :=
    (3 : ℚ) / 9 * (1 : ℚ) / 7 = target_probability :=
by
  sorry

end russian_tennis_players_probability_l598_598472


namespace smallest_k_for_good_coloring_l598_598435

def good_coloring (E : set (circle)) (n : ℕ) (k : ℕ) : Prop := 
  ∃ (A B ∈ E), A ≠ B ∧ has_n_points_in_arc_interior E A B n

theorem smallest_k_for_good_coloring (n : ℕ) (E : set (circle)) (hE : E.card = 2 * n - 1) 
  (hn : n ≥ 3) :
  ∃ k, (∀ (B ⊆ E), B.card = k → good_coloring E n k) ∧
  k = if ∃ e, n = 3 * e + 2 then n - 1 else n := 
sorry

end smallest_k_for_good_coloring_l598_598435


namespace line_plane_relationships_l598_598759

open Set

variable {α : Type*} {l m : α → Prop}

def is_perpendicular_to (x y : α → Prop) : Prop := sorry
def is_parallel_to (x y : α → Prop) : Prop := sorry

theorem line_plane_relationships {l : α → Prop} {α : Set (α → Prop)} :
  (∀ m, is_perpendicular_to m l → ¬ is_parallel_to m α) ∧
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α) :=
by
  sorry

end line_plane_relationships_l598_598759


namespace raft_minimum_capacity_l598_598197

theorem raft_minimum_capacity 
  (mice : ℕ) (mice_weight : ℕ) 
  (moles : ℕ) (mole_weight : ℕ) 
  (hamsters : ℕ) (hamster_weight : ℕ) 
  (raft_cannot_move_without_rower : Bool)
  (rower_condition : ∀ W, W ≥ 2 * mice_weight) :
  mice = 5 → mice_weight = 70 →
  moles = 3 → mole_weight = 90 →
  hamsters = 4 → hamster_weight = 120 →
  ∃ W, (W = 140) :=
by
  intros mice_eq mice_w_eq moles_eq mole_w_eq hamsters_eq hamster_w_eq
  use 140
  sorry

end raft_minimum_capacity_l598_598197


namespace parabola_equation_max_distance_to_line_l598_598357

noncomputable theory

section dec
  variables {p : ℝ} (M : ℝ × ℝ) (l : ℝ → ℝ)

  -- Conditions
  def parabola (x y : ℝ) : Prop := y^2 = 2 * p * x
  def point_on_parabola (M : ℝ × ℝ) : Prop := parabola (M.1) (M.2)
  def distance_to_y_axis (M : ℝ × ℝ) : ℝ := M.1
  def distance_to_point (M : ℝ × ℝ) (P : ℝ × ℝ) : ℝ := real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  def condition_1 (M : ℝ × ℝ) : Prop := distance_to_point M (1, 0) = distance_to_y_axis M + 1

  -- Prove that the parabola equation is y^2 = 4x when p = 2
  theorem parabola_equation :
    p > 0 → 
    point_on_parabola (p / 2, p) →
    condition_1 (p / 2, p) → 
    p = 2 → 
    ∀ x y, parabola x y ↔ y^2 = 4 * x :=
  by
    intros hp hM hc hp_eq
    sorry

  -- Assuming the intersection of line l and the condition on M and the circle through A, B, and M
  def line (m t : ℝ) : ℝ → ℝ := fun y => m * y + t
  def distance_from_line (P : ℝ × ℝ) (m t : ℝ) : ℝ := (abs (P.1 - (line m t P.2))) / sqrt (1 + m^2)

  -- Prove that the maximum distance from (1,0) to line l is 2 sqrt 5
  theorem max_distance_to_line :
    p = 2 →
    ∃ (m t : ℝ), (t = 2 * m + 5 ∨ t = -2 * m + 1) →
    distance_from_line (1, 0) m t = 2 * sqrt 5 :=
  by
    intros hp_eq
    sorry
end dec

end parabola_equation_max_distance_to_line_l598_598357


namespace speed_of_train_from_A_l598_598620

variable (v : ℝ)
variable (dAB : ℝ) (vB : ℝ) (tA : ℝ) (tB : ℝ)

theorem speed_of_train_from_A :
  dAB = 465 →
  vB = 75 →
  tA = 4 →
  tB = 3 →
  4 * v + 3 * vB = dAB →
  v = 60 :=
by
  intros h1 h2 h3 h4 h5
  have h : 4 * v + 3 * 75 = 465 := h5
  simp at h
  sorry

end speed_of_train_from_A_l598_598620


namespace yule_log_surface_area_increase_l598_598945

theorem yule_log_surface_area_increase :
  let h := 10
  let d := 5
  let r := d / 2
  let n := 9
  let initial_surface_area := 2 * Real.pi * r * h + 2 * Real.pi * r^2
  let slice_height := h / n
  let slice_surface_area := 2 * Real.pi * r * slice_height + 2 * Real.pi * r^2
  let total_surface_area_slices := n * slice_surface_area
  let delta_surface_area := total_surface_area_slices - initial_surface_area
  delta_surface_area = 100 * Real.pi :=
by
  sorry

end yule_log_surface_area_increase_l598_598945


namespace city_partition_l598_598157

def City := Type
variable {α β : Type*}

-- Define a typeclass for representing the airline routes
class Routes (α : Type*) :=
  (common_endpoint : α → α → Prop)

-- The main theorem statement
theorem city_partition (k : ℕ) (c : set City) 
  (r : ∀ (c1 c2 : City) (a : ℕ), a < k → c1 ∈ c ∧ c2 ∈ c → Prop)
  (h : ∀ (a : ℕ) (h1 : a < k), ∃ (v : City), 
    ∀ (c1 c2 : City), c1 ≠ v ∧ c2 ≠ v → ¬ r c1 c2 a h1) : 
  ∃ (partition : finset (finset City)), partition.card = k + 2 ∧ 
  (∀ group ∈ partition, ∀ c1 c2 ∈ group, c1 ≠ c2 → ¬ ∃ a h1, r c1 c2 a h1) :=
sorry

end city_partition_l598_598157


namespace clock_hands_overlap_rightangle_straightangle_l598_598353

theorem clock_hands_overlap_rightangle_straightangle :
  ∀ (t : ℝ) (h : 0 ≤ t ∧ t ≤ 12),
  let overlap_count := 11,
      right_angle_count := 22,
      straight_angle_count := 11
  in (
    -- Clock Mechanics: frequency of overlaps
    (number of overlaps in 12 hours = overlap_count) ∧
    -- Clock Mechanics: frequency of right angles
    (number of right angles in 12 hours = right_angle_count) ∧
    -- Clock Mechanics: frequency of straight angles
    (number of straight angles in 12 hours = straight_angle_count)
  ) := 
sorry

end clock_hands_overlap_rightangle_straightangle_l598_598353


namespace find_b_l598_598506

theorem find_b
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (1/12) * x^2 + a * x + b)
  (A C: ℝ × ℝ)
  (hA : A = (x1, 0))
  (hC : C = (x2, 0))
  (T : ℝ × ℝ)
  (hT : T = (3, 3))
  (h_TA : dist (3, 3) (x1, 0) = dist (3, 3) (0, b))
  (h_TB : dist (3, 3) (0, b) = dist (3, 3) (x2, 0))
  (vietas : x1 * x2 = 12 * b)
  : b = -6 := 
sorry

end find_b_l598_598506


namespace Steven_more_than_Jill_l598_598797

variable (Jill Jake Steven : ℕ)

def Jill_peaches : Jill = 87 := by sorry
def Jake_peaches_more : Jake = Jill + 13 := by sorry
def Steven_peaches_more : Steven = Jake + 5 := by sorry

theorem Steven_more_than_Jill : Steven - Jill = 18 := by
  -- Proof steps to be filled
  sorry

end Steven_more_than_Jill_l598_598797


namespace parabola_ratio_l598_598959

noncomputable def AF_over_BF (p : ℝ) (h_p : p > 0) : ℝ :=
  let AF := 4 * p
  let x := (4 / 7) * p -- derived from solving the equation in the solution
  AF / x

theorem parabola_ratio (p : ℝ) (h_p : p > 0) : AF_over_BF p h_p = 7 :=
  sorry

end parabola_ratio_l598_598959


namespace regular_polygon_sides_l598_598213

noncomputable def triangle_ratios := (P Q R : ℝ) (h1 : P : Q : R = 1 : 2 : 4)
noncomputable def inscribed_circle := (circle : Type) (T : Triangle circle) (P Q : Points circle)

theorem regular_polygon_sides (P Q R : ℝ) (m : ℕ) (h : (1 : 2 : 4) = (P : Q : R))
  (H : inscribed_circle circle T P Q R) : m = 7 :=
sorry

end regular_polygon_sides_l598_598213


namespace bus_trip_product_l598_598462
open Nat

theorem bus_trip_product (n k : ℕ) (h1 : 3 < n) (h2: ∑ i in finset.range(n), odd ((2*k)-1)) (h3: ∑ i in finset.range(n), meet_cond i k) (h4 : n * (n - 1) * ((2 * k) - 1) = 600) :
  n * k = 52 ∨ n * k = 40 :=
sorry

end bus_trip_product_l598_598462


namespace measure_of_angle_C_l598_598834

theorem measure_of_angle_C (m l : ℝ) (angle_A angle_B angle_D angle_C : ℝ)
  (h_parallel : l = m)
  (h_angle_A : angle_A = 130)
  (h_angle_B : angle_B = 140)
  (h_angle_D : angle_D = 100) :
  angle_C = 90 :=
by
  sorry

end measure_of_angle_C_l598_598834


namespace ticket_window_time_correct_l598_598005

noncomputable def time_to_ticket_window
  (initial_distance_yards : ℕ)
  (distance_moved_feet : ℕ)
  (time_spent_mins : ℕ)
  (rate_feet_per_min : ℚ)
: ℚ :=
let total_initial_distance_feet := initial_distance_yards * 3,
    remaining_distance_feet := total_initial_distance_feet - distance_moved_feet in
remaining_distance_feet / rate_feet_per_min

theorem ticket_window_time_correct :
  let initial_distance_yards := 100 in
  let distance_moved_feet := 90 in
  let time_spent_mins := 40 in
  let rate_feet_per_min := (90 : ℚ) / 40 in
  time_to_ticket_window initial_distance_yards distance_moved_feet time_spent_mins rate_feet_per_min = 93.333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333... := 
by
  sorry

end ticket_window_time_correct_l598_598005


namespace monotonic_decreasing_interval_l598_598086

noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 1) * Real.exp x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (-2 < x ∧ x < -1) -> (f'(x) < 0) :=
sorry

end monotonic_decreasing_interval_l598_598086


namespace coffee_consumption_total_l598_598983

variables (x : ℝ)

-- Conditions
def alice_coffee := x
def bob_coffee := 1.25 * x
def alice_drinks := 3 / 4 * x
def bob_drinks := 0.9375 * x
def alice_gives_bob := 1 / 8 * x + 1

-- Hypothesis: They both drank the same total amount
def equal_consumption : Prop :=
  alice_drinks - (1 / 8 * x + 1) = bob_drinks + (1 / 8 * x + 1)

-- The goal is to show that the total coffee they drank together is 9 ounces
theorem coffee_consumption_total : equal_consumption x → alice_coffee + bob_coffee = 9 :=
sorry

end coffee_consumption_total_l598_598983


namespace cube_tower_remainder_mod_1000_l598_598600

def cubes := { n : ℕ | 1 ≤ n ∧ n ≤ 8 }

def valid_tower (tower : List ℕ) : Prop :=
  ∀ (i : ℕ), i < tower.length - 1 → (tower.nth i).get_or_else 0 ≤ (tower.nth (i+1)).get_or_else 0 + 3

def T := { t : List ℕ | valid_tower t ∧ t.to_finset = cubes }

noncomputable def number_of_towers : ℕ := T.card

theorem cube_tower_remainder_mod_1000 : number_of_towers % 1000 = 288 := 
  sorry

end cube_tower_remainder_mod_1000_l598_598600


namespace three_days_earning_l598_598038

theorem three_days_earning
  (charge : ℤ := 2)
  (day_before_yesterday_wash : ℤ := 5)
  (yesterday_wash : ℤ := day_before_yesterday_wash + 5)
  (today_wash : ℤ := 2 * yesterday_wash)
  (three_days_earning : ℤ := charge * (day_before_yesterday_wash + yesterday_wash + today_wash)) :
  three_days_earning = 70 := 
by
  have h1 : day_before_yesterday_wash = 5 := by rfl
  have h2 : yesterday_wash = day_before_yesterday_wash + 5 := by rfl
  have h3 : today_wash = 2 * yesterday_wash := by rfl
  have h4 : charge * (day_before_yesterday_wash + yesterday_wash + today_wash) = 70 := sorry
  exact h4

end three_days_earning_l598_598038


namespace find_f_5_l598_598366

def f : ℤ → ℤ
| x := if x ≤ 0 then 1 - x else f (x - 3)

theorem find_f_5 : f 5 = 2 :=
  by
    unfold f
    sorry

end find_f_5_l598_598366


namespace number_of_agents_l598_598235

theorem number_of_agents (jan_claims missy_claims : ℕ) (john_claims : ℕ) :
  (jan_claims = 20) →
  (john_claims = jan_claims + jan_claims * 30 / 100) →
  (missy_claims = john_claims + 15) →
  (missy_claims = 41) →
  ∃ n, n = 3 :=
by
  intros h_jan h_john h_missy h_missy_given
  have h_john_correct : john_claims = 26 := by sorry
  have h_missy_correct : missy_claims = 41 := by sorry
  use 3
  exact h_john_correct

end number_of_agents_l598_598235


namespace total_houses_is_160_l598_598377

namespace MariamNeighborhood

-- Define the given conditions as variables in Lean.
def houses_on_one_side : ℕ := 40
def multiplier : ℕ := 3

-- Define the number of houses on the other side of the road.
def houses_on_other_side : ℕ := multiplier * houses_on_one_side

-- Define the total number of houses in Mariam's neighborhood.
def total_houses : ℕ := houses_on_one_side + houses_on_other_side

-- Prove that the total number of houses is 160.
theorem total_houses_is_160 : total_houses = 160 :=
by
  -- Placeholder for proof
  sorry

end MariamNeighborhood

end total_houses_is_160_l598_598377


namespace smallest_X_l598_598810

noncomputable def T : ℕ := 1110
noncomputable def X : ℕ := T / 6

theorem smallest_X (hT_digits : (∀ d ∈ T.digits 10, d = 0 ∨ d = 1))
  (hT_positive : T > 0)
  (hT_div_6 : T % 6 = 0) :
  X = 185 := by
  sorry

end smallest_X_l598_598810


namespace hoseok_divides_number_l598_598570

theorem hoseok_divides_number (x : ℕ) (h : x / 6 = 11) : x = 66 := by
  sorry

end hoseok_divides_number_l598_598570


namespace multiple_of_k4_with_4_digits_at_most_l598_598851

theorem multiple_of_k4_with_4_digits_at_most (k : ℤ) (hk : k > 1) :
  ∃ m : ℤ, m % (k^4) = 0 ∧ (nat.digits 10 m.nat_abs).to_finset.card ≤ 4 := sorry

end multiple_of_k4_with_4_digits_at_most_l598_598851


namespace remainder_is_correct_l598_598270

def P (x : ℝ) : ℝ := x^6 + 2 * x^5 - 3 * x^4 + x^2 - 8
def D (x : ℝ) : ℝ := x^2 - 1

theorem remainder_is_correct : 
  ∃ q : ℝ → ℝ, ∀ x : ℝ, P x = D x * q x + (2.5 * x - 9.5) :=
by
  sorry

end remainder_is_correct_l598_598270


namespace min_max_difference_l598_598425

noncomputable def ratio (x y z : ℝ) : ℝ := |x + y + z| / (|x| + |y| + |z|)

theorem min_max_difference (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : 
  let m := 0 in
  let M := 1 in
  M - m = 1 :=
by
  sorry

end min_max_difference_l598_598425


namespace log_base_one_six_value_l598_598094

noncomputable def sequence (a : ℕ+ → ℤ) : Prop :=
∀ n : ℕ+, a (n + 1) = 3 + (a n)

theorem log_base_one_six_value :
  ∃ a : ℕ+ → ℤ,
    sequence a ∧
    a 2 + a 4 + a 6 = 9 ∧
    Real.logBase (1/6) (a 5 + a 7 + a 9) = -2 :=
sorry

end log_base_one_six_value_l598_598094


namespace exists_quadrilateral_divided_into_5_equal_triangles_l598_598930

theorem exists_quadrilateral_divided_into_5_equal_triangles :
  ∃ (Q : Type) [IsQuadrilateral Q], ∃ (T : Finset (Triangle Q)), T.card = 5 ∧ 
  (∀ t ∈ T, triangle_area t = (quadrilateral_area Q) / 5) := 
sorry

end exists_quadrilateral_divided_into_5_equal_triangles_l598_598930


namespace cos_C_correct_l598_598392

noncomputable def cos_C (A B C : Type) [InnerProductSpace ℝ A] [DecidableEq A] 
  (AB BC : ℝ) (angleB : Real) 
  (h1 : angleB = Real.pi / 3) 
  (h2 : AB = 8) 
  (h3 : BC = 14) : ℝ :=
  let AC2 := AB^2 + BC^2 - 2 * AB * BC * Real.cos angleB in
  let AC := Real.sqrt AC2 in
  let cosC := (BC^2 - AB^2 - AC2) / (-2 * AB * AC) in
  cosC

theorem cos_C_correct (A B C : Type) [InnerProductSpace ℝ A] [DecidableEq A] 
  (AB BC : ℝ) (angleB : Real) 
  (h1 : angleB = Real.pi / 3) 
  (h2 : AB = 8) 
  (h3 : BC = 14) : cos_C A B C AB BC angleB h1 h2 h3 = 1 / Real.sqrt 148 :=
by
  sorry

end cos_C_correct_l598_598392


namespace students_identified_chess_or_basketball_l598_598778

theorem students_identified_chess_or_basketball (total_students : ℕ) (p_basketball : ℝ) (p_chess : ℝ) (p_soccer : ℝ) :
    total_students = 250 → 
    p_basketball = 0.4 → 
    p_chess = 0.1 →
    p_soccer = 0.28 → 
    (p_basketball * total_students + p_chess * total_students) = 125 :=
begin 
  intros h1 h2 h3 h4,
  sorry
end

end students_identified_chess_or_basketball_l598_598778


namespace ceil_sqrt_fraction_eq_neg2_l598_598659

theorem ceil_sqrt_fraction_eq_neg2 :
  (Int.ceil (-Real.sqrt (36 / 9))) = -2 :=
by
  sorry

end ceil_sqrt_fraction_eq_neg2_l598_598659


namespace solution_inequality_l598_598390

noncomputable def proof_inequality (k b n : ℝ) : Prop :=
  n > 2 → (∀ x : ℝ, (k-2) * x + b > 0 ↔ x < 1)

theorem solution_inequality (k b n : ℝ) (h1 : k ≠ 0) (h2 : y = k * -1 + b = n) (h3 : y = k * 1 + b = 2) 
  (h4 : n > 2) : proof_inequality k b n :=
begin
  sorry
end

end solution_inequality_l598_598390


namespace donation_percentage_correct_l598_598965

def distributed_to_children (income : ℝ) : ℝ := 0.20 * 3 * income
def deposited_to_wife (income : ℝ) : ℝ := 0.30 * income
def total_distribution_to_family (income : ℝ) : ℝ := distributed_to_children income + deposited_to_wife income
def remaining_after_distribution (income : ℝ) : ℝ := income - total_distribution_to_family income
def donated_to_orphan_house (remaining : ℝ) (final_amount : ℝ) : ℝ := remaining - final_amount
def donation_percentage (remaining : ℝ) (donated : ℝ) : ℝ := (donated / remaining) * 100

theorem donation_percentage_correct :
  ∀ income final_amount : ℝ,
  income = 1000000 →
  final_amount = 50000 →
  donation_percentage (remaining_after_distribution income) (donated_to_orphan_house (remaining_after_distribution income) final_amount) = 50 :=
by
  intros income final_amount hincome hfinal_amount
  rw [hincome, hfinal_amount]
  -- the line below shows it takes $1,000,000 of income
  -- with final amount after donation as $50,000
  -- ensuring the proof strategy for remaining and donated values
  sorry

end donation_percentage_correct_l598_598965


namespace planes_perpendicular_l598_598361

-- Define the normal vectors of the planes
def n_alpha : ℝ × ℝ × ℝ := (1, 2, 0)
def n_beta : ℝ × ℝ × ℝ := (2, -1, 0)

-- Dot product of two 3D vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Prove that the dot product of the normal vectors of the planes is 0, implying they are perpendicular
theorem planes_perpendicular : dot_product n_alpha n_beta = 0 :=
  sorry

end planes_perpendicular_l598_598361


namespace line_circle_intersection_l598_598788

noncomputable def line_parametric (a t : ℝ) : ℝ × ℝ := (a + real.sqrt 3 * t, t)

def circle_cartesian (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

def distance_to_line (a x y : ℝ) : ℝ := real.abs (2 - a) / real.sqrt (1 + 3)

theorem line_circle_intersection (a : ℝ) :
  (∃ t, let ⟨x, y⟩ := line_parametric a t in circle_cartesian x y) ↔ -2 ≤ a ∧ a ≤ 6 :=
by
  sorry

end line_circle_intersection_l598_598788


namespace max_n_value_l598_598279

theorem max_n_value (a : ℕ → ℤ) (n : ℕ) (h1 : a 1 = 1) (h2 : a n = 2000) (h3 : ∀ i : ℕ, 2 ≤ i → i ≤ n → a i - a (i - 1) ∈ {-3, 5}): n ≤ 1996 :=
sorry

end max_n_value_l598_598279


namespace AP_is_angle_bisector_l598_598226

theorem AP_is_angle_bisector 
  (A B C D E F P Q: Type) 
  [plane A] [plane B] [plane C] [plane D] [plane E] [plane F]
  [plane P] [plane Q]
  (h₁: altitude A B D) (h₂: altitude B C E) (h₃: altitude C A F)
  (h₄: on_line P D F) (h₅: on_line Q E F)
  (h₆: angle_eq (P, A, Q) (D, A, C)):
  is_angle_bisector A P F Q :=
by
  sorry

end AP_is_angle_bisector_l598_598226


namespace motorist_spent_amount_l598_598185

noncomputable def original_price := 6.222222222222222
def reduction_rate := 0.10
def increased_amount := 5
def final_amount := 28

theorem motorist_spent_amount : 
  let P := original_price in
  let R := P * (1 - reduction_rate) in
  let M := increased_amount * P * (1 - reduction_rate) in
  M = final_amount :=
by
  sorry

end motorist_spent_amount_l598_598185


namespace grid_distribution_same_l598_598028

noncomputable def grid_distribute (n : ℕ) (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, 0 < i → i < n-1 → 0 < j → j < n-1 →
    grid i j = (grid (i-1) j + grid (i+1) j + grid i (j-1) + grid i (j+1)) / 4

theorem grid_distribution_same (grid : ℕ → ℕ → ℕ) (h : ∀ i j, grid_distribute 2014 grid) :
  ∀ i j, grid i j = grid 0 0 :=
by
  sorry

end grid_distribution_same_l598_598028


namespace max_negative_coeffs_P_squared_l598_598010

theorem max_negative_coeffs_P_squared (n : ℤ) (P : ℝ[X]) (h : degree P = n) (hn : n ≥ 2) : 
  ∃ m : ℤ, m = (2 * n - 2) ∧ max_neg_coefs (P * P) = m := 
by sorry

def max_neg_coefs (p : ℝ[X]) : ℤ := sorry

end max_negative_coeffs_P_squared_l598_598010


namespace max_height_table_l598_598393

noncomputable def area_of_triangle (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem max_height_table (PQ QR RP : ℕ) (h : ℝ) :
  PQ = 26 → QR = 28 → RP = 32 →
  let area := area_of_triangle PQ QR RP in
  let h_p := 2 * area / QR in
  let h_r := 2 * area / PQ in
  let max_possible_height := h_p * h_r / (h_p + h_r) →
  h = max_possible_height →
  h = 450 * Real.sqrt 1001 / 29 :=
by sorry

end max_height_table_l598_598393


namespace triangular_pyramid_volume_l598_598214

-- Definitions based on conditions
variables (a b c : ℝ)
axiom ab_condition : a * b = 6
axiom bc_condition : b * c = 8
axiom ac_condition : a * c = 12

-- The final statement to prove
theorem triangular_pyramid_volume :
  (∃ (a b c : ℝ), a * b = 6 ∧ b * c = 8 ∧ a * c = 12) → (1 / 6) * (a * b * c) = 4 :=
  by
    intro h
    cases h with a h1
    cases h1 with b h2
    cases h2 with c h3
    cases h3
    use a, b, c
    sorry -- skip the proof

end triangular_pyramid_volume_l598_598214


namespace largest_number_l598_598224

def base9_to_dec (n : ℕ) : ℕ :=
  8 * 9^1 + 5 * 9^0

def base6_to_dec (n : ℕ) : ℕ :=
  2 * 6^2 + 1 * 6^1 + 0 * 6^0

def base4_to_dec (n : ℕ) : ℕ :=
  1 * 4^3 + 0 * 4^2 + 0 * 4^1 + 0 * 4^0

def base2_to_dec (n : ℕ) : ℕ :=
  1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem largest_number :
  base6_to_dec 210 > base9_to_dec 85 ∧
  base6_to_dec 210 > base4_to_dec 1000 ∧
  base6_to_dec 210 > base2_to_dec 111111 :=
begin
  -- Proof goes here
  sorry
end

end largest_number_l598_598224


namespace smallest_number_is_a_l598_598222

def smallest_number_among_options : ℤ :=
  let a: ℤ := -3
  let b: ℤ := 0
  let c: ℤ := -(-1)
  let d: ℤ := (-1)^2
  min a (min b (min c d))

theorem smallest_number_is_a : smallest_number_among_options = -3 :=
  by
    sorry

end smallest_number_is_a_l598_598222


namespace solve_for_x_l598_598355

-- Define the operation
def triangle (a b : ℝ) : ℝ := 2 * a - b

-- Define the necessary conditions and the goal
theorem solve_for_x :
  (∀ (a b : ℝ), triangle a b = 2 * a - b) →
  (∃ x : ℝ, triangle x (triangle 1 3) = 2) →
  ∃ x : ℝ, x = 1 / 2 :=
by 
  intros h_main h_eqn
  -- We can skip the proof part as requested.
  sorry

end solve_for_x_l598_598355


namespace NinatsHighSchoolHas4000Students_l598_598029

-- Definitions
variable (N M : ℕ)

-- Conditions as hypotheses
hypothesis h1 : N = 5 * M
hypothesis h2 : N + M = 4800
hypothesis h3 : (N - 200) + (M + 200) = 2 * (M + 200)

-- Proof statement
theorem NinatsHighSchoolHas4000Students (h1 : N = 5 * M) (h2 : N + M = 4800) (h3 : (N - 200) + (M + 200) = 2 * (M + 200)) : N = 4000 :=
sorry

end NinatsHighSchoolHas4000Students_l598_598029


namespace ratio_of_areas_l598_598531

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l598_598531


namespace find_a2_b2_c2_l598_598639

-- Define the conditions as described in the problem statement
def diameter (d : ℝ) := 1
def num_circles (n : ℕ) := 9

-- Define the line equation parameters
def line_slope (m : ℝ) := 4

-- Define the resulting coefficients and their GCD condition
def gcd_condition (a b c : ℕ) : Prop := Int.gcd (Int.gcd a b) c = 1

-- Now, we need to state the theorem to prove
theorem find_a2_b2_c2
  (a b c : ℕ)
  (h_circles : num_circles 9)
  (h_diameter : ∀ n, n < h_circles → diameter 1 = 1)
  (h_slope : line_slope 4)
  (h_line : a * 1 = b * (3 * 1 + c))
  (h_gcd : gcd_condition a b c) :
  a^2 + b^2 + c^2 = 65 := 
sorry

end find_a2_b2_c2_l598_598639


namespace degrees_to_radians_15_l598_598660

theorem degrees_to_radians_15 (pi_eq_deg: Real = 180) : 15 * (pi / 180) = pi / 12 := 
by
  sorry

end degrees_to_radians_15_l598_598660


namespace num_real_a_with_int_roots_l598_598677

theorem num_real_a_with_int_roots :
  (∃ n : ℕ, n = 15 ∧ ∀ a : ℝ, (∃ r s : ℤ, (r + s = -a) ∧ (r * s = 12 * a) → true)) :=
sorry

end num_real_a_with_int_roots_l598_598677


namespace general_term_b_l598_598695

noncomputable def S (n : ℕ) : ℚ := sorry -- Define the sum of the first n terms sequence S_n
noncomputable def a (n : ℕ) : ℚ := sorry -- Define the sequence a_n
noncomputable def b (n : ℕ) : ℤ := Int.log 3 (|a n|) -- Define the sequence b_n using log base 3

-- Theorem stating the general formula for the sequence b_n
theorem general_term_b (n : ℕ) (h : 0 < n) :
  b n = -n :=
sorry -- We skip the proof, focusing on statement declaration

end general_term_b_l598_598695


namespace octagon_diagonals_l598_598151

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l598_598151


namespace ratio_of_areas_of_triangles_l598_598545

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l598_598545


namespace standard_deviation_transformed_data_l598_598733

variable {n : ℕ} {x : Fin n → ℝ}
variable (s² : ℝ) (h_var : s² = 4)

theorem standard_deviation_transformed_data :
  let transformed : Fin n → ℝ := fun i ↦ -3 * x i + 5
  let variance_transformed : ℝ := 9 * s²
  let stddev_transformed : ℝ := Real.sqrt variance_transformed
  stddev_transformed = 6 :=
by
  intro transformed variance_transformed stddev_transformed
  -- Proof omitted
  sorry

end standard_deviation_transformed_data_l598_598733


namespace price_of_pen_l598_598929

theorem price_of_pen (price_pen : ℚ) (price_notebook : ℚ) :
  (price_pen + 3 * price_notebook = 36.45) →
  (price_notebook = 15 / 4 * price_pen) →
  price_pen = 3 :=
by
  intros h1 h2
  sorry

end price_of_pen_l598_598929


namespace initial_investment_amount_l598_598978

theorem initial_investment_amount :
  ∃ P : ℝ, P ≈ 7837.94 ∧ (P * (1 + 0.03) * (1 + 0.04) * (1 + 0.05) * (1 + 0.06) * (1 + 0.07) = 10000) :=
begin
  use 7837.94,
  split,
  { sorry }, -- Here, ~ would be rigorously defined or approximated
  { sorry }
end

end initial_investment_amount_l598_598978


namespace double_mean_value_range_l598_598650

def is_double_mean_value_function (f : ℝ → ℝ) (a b x1 x2 : ℝ) : Prop :=
  (a < x1 ∧ x1 < x2 ∧ x2 < b) ∧
  (f'' x1 = (f b - f a) / (b - a) ∧ f'' x2 = (f b - f a) / (b - a))

theorem double_mean_value_range (m : ℝ) : 
  is_double_mean_value_function (λ x : ℝ, (1/3) * x^3 - (m/2) * x^2) 0 2 x1 x2 →
  (4 / 3 < m ∧ m < 8 / 3) := 
sorry

end double_mean_value_range_l598_598650


namespace sun_salutations_per_year_l598_598072

theorem sun_salutations_per_year :
  let poses_per_day := 5
  let days_per_week := 5
  let weeks_per_year := 52
  poses_per_day * days_per_week * weeks_per_year = 1300 :=
by
  sorry

end sun_salutations_per_year_l598_598072


namespace solution_set_inequality_l598_598830

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃a b⦄, a ∈ s → b ∈ s → a ≤ b → f a ≤ f b

def f_increasing_on_pos : Prop := is_increasing_on f (Set.Ioi 0)

def f_at_one_zero : Prop := f 1 = 0

theorem solution_set_inequality : 
    is_odd f →
    f_increasing_on_pos →
    f_at_one_zero →
    {x : ℝ | x * (f x - f (-x)) < 0} = {x : ℝ | -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1} :=
sorry

end solution_set_inequality_l598_598830


namespace number_of_toys_is_correct_l598_598578

-- Define the quantities and their relationships
def initial_amount : ℝ := 57
def game_cost : ℝ := 27
def sales_tax_rate : ℝ := 0.08
def toy_cost : ℝ := 6

def sales_tax : ℝ := game_cost * sales_tax_rate
def total_game_cost : ℝ := game_cost + sales_tax
def remaining_money : ℝ := initial_amount - total_game_cost
def number_of_toys : ℕ := (remaining_money / toy_cost).toNat

-- Prove the number of toys Will can buy is 4
theorem number_of_toys_is_correct : number_of_toys = 4 :=
by 
  -- Lean will throw an error without this line, but we leave this since our goal was statement only
  sorry

end number_of_toys_is_correct_l598_598578


namespace ellipse_equation_fixed_point_exists_l598_598304

variables (a b : ℝ) (P : ℝ × ℝ)
variable  (k : ℝ)

-- Conditions
def is_ellipse (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def focal_length := 1
def point_P : Prop := (P = (1, 3/2))

-- Proof Problem (1): Prove the equation of the ellipse given the conditions.
theorem ellipse_equation :
  a > b ∧ b > 0 ∧ is_ellipse 1 (3 / 2) → (a^2 = 4 ∧ b^2 = 3) :=
begin
    sorry
end

-- Proof Problem (2): Prove existence of fixed point T on the x-axis
def focus_right := (focal_length, 0)
def is_fixed_point (T : ℝ × ℝ) : Prop := 
  ∀ (l : ℝ → ℝ), l 0 = focus_right.1 ∧ l focus_right.2 = k * focus_right.2 → 
  ∃ T_x, T = (T_x, 0) ∧ T_x = 4 ∧ ∀ A B : ℝ × ℝ, 
  (is_ellipse A.1 A.2 ∧ is_ellipse B.1 B.2 ∧ A ≠ B) → 
  |(A.1 - focus_right.1) * (B.1 - T_x)| = |(B.1 - focus_right.1) * (A.1 - T_x)|

theorem fixed_point_exists :
  k ≠ 0 ∧ a > b ∧ b > 0 ∧ is_ellipse 1 (3 / 2) ∧ focal_length = 1 →
  ∃ T, T = (4, 0) ∧ is_fixed_point T :=
begin
    sorry
end

end ellipse_equation_fixed_point_exists_l598_598304


namespace Reuschle_theorem_l598_598848

theorem Reuschle_theorem (A B C A1 B1 C1 A2 B2 C2 : Point) (circumcircle_A1B1C1 : Circle) 
  (h_concur_AA1_BB1_CC1 : Concurrent cevians (A, A1) (B, B1) (C, C1))
  (h_A2 : second_intersection (circumcircle_A1B1C1) (line_segment A B) = A2)
  (h_B2 : second_intersection (circumcircle_A1B1C1) (line_segment B C) = B2)
  (h_C2 : second_intersection (circumcircle_A1B1C1) (line_segment C A) = C2) :
  Concurrent cevians (A, A2) (B, B2) (C, C2) :=
sorry

end Reuschle_theorem_l598_598848


namespace highest_power_of_2_l598_598555

theorem highest_power_of_2 (a b c d e f g k l m : ℕ) (h_perm : {a, b, c, d, e, f, g, k, l, m} = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :
  ∃ p : ℕ, (2^p ∣ a^b * b^c * c^d * d^e * e^f * f^g * g^k * k^l * l^m * m^a) ∧ p = 69 :=
  by sorry

end highest_power_of_2_l598_598555


namespace decreasing_function_in_interval_l598_598814

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x + Real.pi / 4)

theorem decreasing_function_in_interval (ω : ℝ) (h_omega_pos : ω > 0) (h_period : Real.pi / 3 < 2 * Real.pi / (2 * ω) ∧ 2 * Real.pi / (2 * ω) < Real.pi / 2)
    (h_symmetry : 2 * ω * 3 * Real.pi / 4 + Real.pi / 4 = (4:ℤ) * Real.pi) :
    ∀ x : ℝ, Real.pi / 6 < x ∧ x < Real.pi / 4 → f ω x < f ω (x + Real.pi / 100) :=
by
    intro x h_interval
    have ω_value : ω = 5 / 2 := sorry
    exact sorry

end decreasing_function_in_interval_l598_598814


namespace sum_less_than_four_l598_598824

noncomputable def a_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n - 1), (n : ℝ) / (n - k) * (1 / (2 ^ (k - 1)))

theorem sum_less_than_four (n : ℕ) (h : 2 ≤ n) :
  a_n n < 4 := 
sorry

end sum_less_than_four_l598_598824


namespace answer_to_eighth_group_l598_598307

noncomputable def fifth_group_sample : ℕ := 22

def sampling_interval : ℕ := 5

def group_size : ℕ := 5

def group_start (n : ℕ) : ℕ := (n - 1) * group_size + 1

def fifth_group_start : ℕ := group_start 5

def number_in_group (group_start : ℕ) (position : ℕ) : ℕ := group_start + position - 1

def position_in_fifth_group : ℕ := fifth_group_sample - fifth_group_start + 1

def eighth_group_start : ℕ := group_start 8

def expected_eighth_group_sample : ℕ := number_in_group eighth_group_start position_in_fifth_group

theorem answer_to_eighth_group:
expected_eighth_group_sample = 37 := sorry

end answer_to_eighth_group_l598_598307


namespace sequence_formula_correct_l598_598790

-- Define the sequence S_n
def S (n : ℕ) : ℤ := n^2 - 2

-- Define the general term of the sequence a_n
def a (n : ℕ) : ℤ :=
  if n = 1 then -1 else 2 * n - 1

-- Theorem to prove that for the given S_n, the defined a_n is correct
theorem sequence_formula_correct (n : ℕ) (h : n > 0) : 
  a n = if n = 1 then -1 else S n - S (n - 1) :=
by sorry

end sequence_formula_correct_l598_598790


namespace find_angle_C_find_side_c_l598_598767

def condition1 (A B : ℝ) : Prop :=
  4 * sin A * sin B - 4 * cos ((A - B) / 2)^2 = sqrt 2 - 2

def condition2 (a A B : ℝ) : Prop :=
  a * sin B / sin A = 4

def condition3 (a b c : ℝ) : Prop :=
  (1 / 2) * a * b * sin (π / 4) = 8

theorem find_angle_C (A B C : ℝ) (h1 : condition1 A B) : C = π / 4 :=
by sorry

theorem find_side_c (A B a b c : ℝ) (h1 : condition1 A B) (h2 : condition2 a A B) (h3 : condition3 a b c) : c = 4 :=
by sorry

end find_angle_C_find_side_c_l598_598767


namespace find_z_l598_598418

def bowtie (a b : ℝ) : ℝ :=
  a + real.sqrt (b + real.sqrt (b + real.sqrt (b + ...)))

theorem find_z (z : ℝ) (h : bowtie 5 z = 12) : z = 42 :=
sorry

end find_z_l598_598418


namespace calls_from_at_least_one_cousin_l598_598243

/-- Daniel has four cousins who call him regularly:
 - One calls every 2 days,
 - One calls every 3 days,
 - One calls every 4 days,
 - One calls every 6 days.
 All four cousins called him on December 31 of a leap year.

 We need to prove that the total number of days in the following year on which Daniel receives calls from at least one cousin is 244. -/
theorem calls_from_at_least_one_cousin :
  ∃ days : ℕ, days = 244 ∧
  (days = ( ⌊366 / 2⌋ + ⌊366 / 3⌋ + ⌊366 / 4⌋ + ⌊366 / 6⌋
          - (2 * ⌊366 / 6⌋ + ⌊366 / 4⌋ + ⌊366 / 12⌋)
          + ⌊366 / 12⌋)) :=
by
  sorry

end calls_from_at_least_one_cousin_l598_598243


namespace raft_min_capacity_l598_598188

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end raft_min_capacity_l598_598188


namespace solve_equation_l598_598263

theorem solve_equation (x : ℝ) : (∛(5 - x / 3) = 2) → (x = 9) := 
by 
  sorry

end solve_equation_l598_598263


namespace angle_BCQ_eq_angle_BAC_l598_598227

-- Define the geometrical constructs and conditions
variables (A B C M D E P Q : Type*)
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited M] [Inhabited D] [Inhabited E] [Inhabited P] [Inhabited Q]

-- Conditions
axiom A_eq_middle_DE : A = midpoint D E
axiom MD_parallel_AB : parallel MD AB
axiom M_eq_middle_AC : M = midpoint A C
axiom circle_ABC_E_has_P : intersects (circumcircle A B E) AC P
axiom circle_ADP_has_Q_on_DM : intersects (circumcircle A D P) extension_DM Q

-- To Prove
theorem angle_BCQ_eq_angle_BAC : ∀ (A B C Q : Type*)
    (A_eq_middle_DE : A = midpoint D E)
    (MD_parallel_AB : parallel MD AB)
    (M_eq_middle_AC : M = midpoint A C)
    (circle_ABC_E_has_P : intersects (circumcircle A B E) AC P)
    (circle_ADP_has_Q_on_DM : intersects (circumcircle A D P) extension_DM Q),
    angle B C Q = angle B A C := 
begin
  -- Proof goes here, currently skipped
  sorry
end

end angle_BCQ_eq_angle_BAC_l598_598227


namespace sun_salutations_per_year_l598_598071

theorem sun_salutations_per_year :
  (∀ S : Nat, S = 5) ∧
  (∀ W : Nat, W = 5) ∧
  (∀ Y : Nat, Y = 52) →
  ∃ T : Nat, T = 1300 :=
by 
  sorry

end sun_salutations_per_year_l598_598071


namespace find_line_eq_of_given_conditions_l598_598692

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * y + 5 = 0
def line_perpendicular (a b : ℝ) : Prop := a + b + 1 = 0
def is_center (x y : ℝ) : Prop := (x, y) = (0, 3)
def is_eq_of_line (x y : ℝ) : Prop := x - y + 3 = 0

theorem find_line_eq_of_given_conditions (x y : ℝ) (h1 : circle_eq x y) (h2 : line_perpendicular x y) (h3 : is_center x y) : is_eq_of_line x y :=
by
  sorry

end find_line_eq_of_given_conditions_l598_598692


namespace sufficiency_but_not_necessity_l598_598155

theorem sufficiency_but_not_necessity (a b : ℝ) :
  (a = 0 → a * b = 0) ∧ (a * b = 0 → a = 0) → False :=
by
   -- Proof is skipped
   sorry

end sufficiency_but_not_necessity_l598_598155


namespace maximum_pencils_l598_598184

-- Define the problem conditions
def red_pencil_cost := 27
def blue_pencil_cost := 23
def max_total_cost := 940
def max_diff := 10

-- Define the main theorem
theorem maximum_pencils (x y : ℕ) 
  (h1 : red_pencil_cost * x + blue_pencil_cost * y ≤ max_total_cost)
  (h2 : y - x ≤ max_diff)
  (hx_min : ∀ z : ℕ, z < x → red_pencil_cost * z + blue_pencil_cost * (z + max_diff) > max_total_cost):
  x = 14 ∧ y = 24 ∧ x + y = 38 := 
  sorry

end maximum_pencils_l598_598184


namespace ball_hits_ground_in_2_72_seconds_l598_598633

noncomputable def ball_hits_ground_time : ℝ :=
  let h : ℝ → ℝ := λ t, -16 * t^2 - 30 * t + 200
  in 2.72

theorem ball_hits_ground_in_2_72_seconds :
  let h : ℝ → ℝ := λ t, -16 * t^2 - 30 * t + 200
  in h 2.72 = 0 :=
by
  let h : ℝ → ℝ := λ t, -16 * t^2 - 30 * t + 200
  sorry

end ball_hits_ground_in_2_72_seconds_l598_598633


namespace volume_of_cylinder_unfold_l598_598315

theorem volume_of_cylinder_unfold (h r : ℝ) (h_height : h = 4) (h_circ : 2 * real.pi * r = 4) :
  (real.pi * r^2 * h = 16 / real.pi) :=
by
  sorry

end volume_of_cylinder_unfold_l598_598315


namespace increasing_functions_count_l598_598223

noncomputable def f1 : ℝ → ℝ := λ x, x⁻¹
noncomputable def f2 : ℝ → ℝ := λ x, x^(1/2)
noncomputable def f3 : ℝ → ℝ := λ x, x
noncomputable def f4 : ℝ → ℝ := λ x, x^2
noncomputable def f5 : ℝ → ℝ := λ x, x^3

-- Define a predicate to check if a function is increasing over its domain
def is_increasing (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x < y → f x < f y

-- Define the theorem statement
theorem increasing_functions_count :
  (if is_increasing (f1) then 1 else 0) +
  (if is_increasing (f2) then 1 else 0) +
  (if is_increasing (f3) then 1 else 0) +
  (if is_increasing (f4) then 1 else 0) +
  (if is_increasing (f5) then 1 else 0) = 3 :=
sorry

end increasing_functions_count_l598_598223


namespace cars_transfer_equation_l598_598470

theorem cars_transfer_equation (x : ℕ) : 100 - x = 68 + x :=
sorry

end cars_transfer_equation_l598_598470


namespace sum_of_rectangle_areas_l598_598652

theorem sum_of_rectangle_areas (a b : ℕ) (ha : a ≥ 1) (hb : b ≥ 1) (h : a * b = 3 * (2 * a + 2 * b)) : 
  (Set.range (λ (ab : ℕ × ℕ), (ab.1 * ab.2)) : Set ℕ).sum = 942 :=
sorry

end sum_of_rectangle_areas_l598_598652


namespace hypotenuse_length_l598_598773

noncomputable theory
open Real

theorem hypotenuse_length (A B C D E : Point) 
  (x : ℝ) 
  (hABC : right_triangle A B C)
  (hD : midpoint D B C) 
  (hE : trisection E B C) 
  (hAD : distance A D = Real.cos x) 
  (hAE : distance A E = Real.sin x) 
  (hx : 0 < x ∧ x < π / 2) : 
  distance B C = Real.sqrt (18 / 13) := 
sorry

end hypotenuse_length_l598_598773


namespace parabola_coordinates_and_area_l598_598329

theorem parabola_coordinates_and_area
  (A B C : ℝ × ℝ)
  (hA : A = (2, 0))
  (hB : B = (3, 0))
  (hC : C = (5 / 2, 1 / 4))
  (h_vertex : ∀ x y, y = -x^2 + 5 * x - 6 → 
                   ((x, y) = A ∨ (x, y) = B ∨ (x, y) = C)) :
  A = (2, 0) ∧ B = (3, 0) ∧ C = (5 / 2, 1 / 4)
  ∧ (1 / 2 * (3 - 2) * (1 / 4) = 1 / 8) := 
by
  sorry

end parabola_coordinates_and_area_l598_598329


namespace alice_grades_l598_598027

theorem alice_grades (papers_per_8_hours : ℕ) (time_in_hours_1 : ℕ) (time_in_hours_2 : ℕ) (papers_per_hour : ℕ) (total_papers : ℕ) (rate_calculation : papers_per_hour = papers_per_8_hours / time_in_hours_1) (calculation_correct : total_papers = papers_per_hour * time_in_hours_2) : 
  papers_per_8_hours = 296 → time_in_hours_1 = 8 → time_in_hours_2 = 11 → total_papers = 407 :=
by
  intros h1 h2 h3
  rw [h1, h2] at rate_calculation
  rw rate_calculation at calculation_correct
  simp at calculation_correct
  exact calculation_correct

end alice_grades_l598_598027


namespace even_function_derivative_at_zero_l598_598763

variable (f : ℝ → ℝ)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_diff : Differentiable ℝ f)

theorem even_function_derivative_at_zero : deriv f 0 = 0 :=
by 
  -- proof omitted
  sorry

end even_function_derivative_at_zero_l598_598763


namespace standard_polar_representation_l598_598782

theorem standard_polar_representation {r θ : ℝ} (hr : r < 0) (hθ : θ = 5 * Real.pi / 6) :
  ∃ (r' θ' : ℝ), r' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * Real.pi ∧ (r', θ') = (5, 11 * Real.pi / 6) := 
by {
  sorry
}

end standard_polar_representation_l598_598782


namespace area_enclosed_l598_598967

theorem area_enclosed : 
  let AC := 40
  let AE := 24
  let AB := AC / 3
  let AF := AE / 2
  let rectangle_area := AC * AE
  let semicircle_radius := AC / 2
  let semicircle_area := (Real.pi * semicircle_radius ^ 2) / 2
  let total_area := rectangle_area + semicircle_area
  let triangle1_area := (AB * AF) / 2
  let triangle2_area := (AB * AF) / 2
  let quadrilateral_area := triangle1_area + triangle2_area
  let final_area := total_area - quadrilateral_area
  in final_area = 800 + 200 * Real.pi := by
  {
    have hAC : AC = 40 := rfl,
    have hAE : AE = 24 := rfl,
    have hAB : AB = 40 / 3 := rfl,
    have hAF : AF = 24 / 2 := rfl,
    have hRectangle : rectangle_area = 40 * 24 := rfl,
    have hSemiradius : semicircle_radius = 40 / 2 := rfl,
    have hSemicircle : semicircle_area = (Real.pi * (20^2)) / 2 := rfl,
    have hTotal : total_area = 960 + 200 * Real.pi := by rw [hRectangle, hSemicircle],
    have hTriangle1 : triangle1_area = (40 / 3 * 12) / 2 := rfl,
    have hTriangle2 : triangle2_area = (40 / 3 * 12) / 2 := rfl,
    have hQuadrilateral : quadrilateral_area = 80 + 80 := rfl,
    have hFinal : final_area = 960 + 200 * Real.pi - 160 := by rw [hTotal, hQuadrilateral],
    simp at hFinal,
    exact hFinal
  }

end area_enclosed_l598_598967


namespace largest_square_area_l598_598056

def original_square_side_length : ℝ := 5
def cut_square_side_length : ℝ := 1

def remaining_space_diagonal : ℝ := original_square_side_length - 2 * cut_square_side_length
-- Diagonal space available for the largest inscribed square
def effective_diagonal_space : ℝ := remaining_space_diagonal + 2 * cut_square_side_length

theorem largest_square_area :
  let diagonal := (effective_diagonal_space / Real.sqrt 2) in
  let side_length := diagonal / Real.sqrt 2 in
  side_length ^ 2 = 12.5 :=
by
  sorry

end largest_square_area_l598_598056


namespace cannot_determine_right_triangle_l598_598717

/-- Proof that the condition \(a^2 = 5\), \(b^2 = 12\), \(c^2 = 13\) cannot determine that \(\triangle ABC\) is a right triangle. -/
theorem cannot_determine_right_triangle (a b c : ℝ) (ha : a^2 = 5) (hb : b^2 = 12) (hc : c^2 = 13) : 
  ¬(a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := 
by
  sorry

end cannot_determine_right_triangle_l598_598717


namespace product_of_areas_eq_l598_598966

-- Definitions for distances
variables {n : ℕ} (P : Point) (h a : Fin n → ℝ)

-- Theorem statement
theorem product_of_areas_eq (n_even : Even n) (h_pos : ∀ i, 0 < h i) (a_pos : ∀ i, 0 < a i) :
  let red_indices := Finset.filter (λ i, i.1 % 2 = 1) (Finset.range n)
      blue_indices := Finset.filter (λ i, i.1 % 2 = 0) (Finset.range n) in
  (∏ i in red_indices, (1 / 2) * a i * h i) = (∏ i in blue_indices, (1 / 2) * a i * h i) :=
sorry

end product_of_areas_eq_l598_598966


namespace counting_perfect_squares_l598_598343

theorem counting_perfect_squares :
  let count := (finset.range 71).filter (λ n, n * n < 5000 ∧ ((n * n % 10 = 4) ∨ 
                                                               (n * n % 10 = 5) ∨ 
                                                               (n * n % 10 = 6))).card in
  count = 35 :=
sorry

end counting_perfect_squares_l598_598343


namespace semicircle_radius_l598_598994

theorem semicircle_radius (AC BC : ℝ) (hAC : AC = 12) (hBC : BC = 5) (angleC : ∠ABC = 90) :
    ∃ r : ℝ, r = 10 / 3 :=
by
  sorry

end semicircle_radius_l598_598994


namespace find_b_l598_598756

variable (p q r b : ℤ)

-- Conditions
def condition1 : Prop := p - q = 2
def condition2 : Prop := p - r = 1

-- The main statement to prove
def problem_statement : Prop :=
  b = (r - q) * ((p - q)^2 + (p - q) * (p - r) + (p - r)^2) → b = 7

theorem find_b (h1 : condition1 p q) (h2 : condition2 p r) (h3 : problem_statement p q r b) : b = 7 :=
sorry

end find_b_l598_598756


namespace other_endpoint_coordinates_sum_l598_598054

noncomputable def other_endpoint_sum (x1 y1 x2 y2 xm ym : ℝ) : ℝ :=
  let x := 2 * xm - x1
  let y := 2 * ym - y1
  x + y

theorem other_endpoint_coordinates_sum :
  (other_endpoint_sum 6 (-2) 0 12 3 5) = 12 := by
  sorry

end other_endpoint_coordinates_sum_l598_598054


namespace power_sum_integer_l598_598796

theorem power_sum_integer (x : ℝ) (hx : x ≠ 0) (ha : x + 1/x ∈ ℤ) (n : ℕ) : x^n + 1/(x^n) ∈ ℤ :=
by
  sorry

end power_sum_integer_l598_598796


namespace four_digit_permutations_l598_598742

theorem four_digit_permutations : 
  let digits := [2, 0, 2, 5],
      perms := (Multiset.cons 2 (Multiset.cons 0 (Multiset.cons 2 (Multiset.cons 5 Multiset.nil)))),
      all_perms := perms.pperm,
      valid_perms := all_perms.filter (λ l, l.head ≠ 0)
  in valid_perms.card = 9 := sorry

end four_digit_permutations_l598_598742


namespace f_of_f_inv_e_eq_inv_e_l598_598688

noncomputable def f : ℝ → ℝ := λ x =>
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_of_f_inv_e_eq_inv_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end f_of_f_inv_e_eq_inv_e_l598_598688


namespace increasing_interval_proof_l598_598082

open Real

-- Given that the graph of f(x) is symmetric to the graph of 3^x with respect to y = x.
-- We are to prove that the increasing interval of f(6x - x^2) is (0, 3).

/--
f is a function such that f(x) is symmetric to 3^x with respect to y = x.
-/
def f (x : ℝ) : ℝ := log x / log 3

/--
The interval where 6x - x^2 is positive.
-/
def valid_domain : set ℝ := {x | 0 < x ∧ x < 6}

/--
The interval where f(6x - x^2) is increasing.
-/
def increasing_interval : set ℝ := {x | 0 < x ∧ x < 3}

/--
Proof that the increasing interval of f(6x - x^2) is (0, 3).
-/
theorem increasing_interval_proof : ∀ x, x ∈ valid_domain → f (6 * x - x^2) ∈ increasing_interval :=
by
  sorry

end increasing_interval_proof_l598_598082


namespace optionA_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_not_right_triangle_l598_598715

-- Four conditions for the triangle ABC
axiom condA : ∀ (A B C : ℝ), A + B = C
axiom condB : ∀ (A B C : ℝ), 2 * A = B ∧ 3 * A = C
axiom condC : ∀ (a b c : ℝ), a^2 = b^2 - c^2
axiom condD : ∀ (a b c : ℝ), a^2 = 5 ∧ b^2 = 12 ∧ c^2 = 13

-- The angles and sides of triangle ABC
variable (A B C : ℝ)
variable (a b c : ℝ)

-- The proof
theorem optionA_right_triangle : condA A B C → A + B = 90 := by 
  sorry
theorem optionB_right_triangle : condB A B C → C = 90 := by 
  sorry
theorem optionC_right_triangle : condC a b c → a^2 + c^2 = b^2 := by 
  sorry
theorem optionD_not_right_triangle : condD a b c → ¬(a^2 + b^2 = c^2) := by 
  sorry

end optionA_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_not_right_triangle_l598_598715


namespace tennis_tournament_total_rounds_l598_598166

theorem tennis_tournament_total_rounds
  (participants : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (pairs_formation : ℕ → ℕ)
  (single_points_award : ℕ → ℕ)
  (elimination_condition : ℕ → Prop)
  (tournament_continues : ℕ → Prop)
  (progression_condition : ℕ → ℕ → ℕ)
  (group_split : Π (n : ℕ), Π (k : ℕ), (ℕ × ℕ))
  (rounds_needed : ℕ) :
  participants = 1152 →
  points_win = 1 →
  points_loss = 0 →
  pairs_formation participants ≥ 0 →
  single_points_award participants ≥ 0 →
  (∀ p, p > 1 → participants / p > 0 → tournament_continues participants) →
  (∀ m n, progression_condition m n = n - m) →
  (group_split 1152 1024 = (1024, 128)) →
  rounds_needed = 14 :=
by
  sorry

end tennis_tournament_total_rounds_l598_598166


namespace range_of_m_l598_598694

variable {m x x1 x2 y1 y2 : ℝ}

noncomputable def linear_function (m x : ℝ) : ℝ := (m - 2) * x + (2 + m)

theorem range_of_m (h1 : x1 < x2) (h2 : y1 = linear_function m x1) (h3 : y2 = linear_function m x2) (h4 : y1 > y2) : m < 2 :=
by
  sorry

end range_of_m_l598_598694


namespace isosceles_triangle_area_l598_598903

theorem isosceles_triangle_area {A B C : Type} 
  (h_isosceles : ΔABC.IsIsosceles)
  (h_sides : ℓ(A, B) = 13 ∧ ℓ(B, C) = 24 ∧ ℓ(C, A) = 13) :
  ΔABC.area = 60 :=
by
  sorry

end isosceles_triangle_area_l598_598903


namespace pos_sol_eq_one_l598_598433

theorem pos_sol_eq_one (n : ℕ) (hn : 1 < n) :
  ∀ x : ℝ, 0 < x → (x ^ n - n * x + n - 1 = 0) → x = 1 := by
  -- The proof goes here
  sorry

end pos_sol_eq_one_l598_598433


namespace polynomial_divisible_by_five_l598_598823

open Polynomial

theorem polynomial_divisible_by_five
  (a b c d m : ℤ)
  (h1 : (a * m^3 + b * m^2 + c * m + d) % 5 = 0)
  (h2 : d % 5 ≠ 0) :
  ∃ (n : ℤ), (d * n^3 + c * n^2 + b * n + a) % 5 = 0 := 
  sorry

end polynomial_divisible_by_five_l598_598823


namespace log_property_l598_598313

theorem log_property (x : ℝ) (h₁ : Real.log x > 0) (h₂ : x > 1) : x > Real.exp 1 := by 
  sorry

end log_property_l598_598313


namespace sample_size_calculation_l598_598947

theorem sample_size_calculation : 
  ∀ (high_school_students junior_high_school_students sampled_high_school_students n : ℕ), 
  high_school_students = 3500 →
  junior_high_school_students = 1500 →
  sampled_high_school_students = 70 →
  n = (3500 + 1500) * 70 / 3500 →
  n = 100 :=
by
  intros high_school_students junior_high_school_students sampled_high_school_students n
  intros h1 h2 h3 h4
  sorry

end sample_size_calculation_l598_598947


namespace number_of_adults_attending_concert_l598_598490

-- We have to define the constants and conditions first.
variable (A C : ℕ)
variable (h1 : A + C = 578)
variable (h2 : 2 * A + 3 / 2 * C = 985)

-- Now we state the theorem that given these conditions, A is equal to 236.

theorem number_of_adults_attending_concert : A = 236 :=
by sorry

end number_of_adults_attending_concert_l598_598490


namespace same_terminal_side_eq_l598_598249

theorem same_terminal_side_eq (α : ℝ) : 
    (∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 3) ↔ α = 5 * Real.pi / 3 :=
by sorry

end same_terminal_side_eq_l598_598249


namespace mb_range_l598_598958
-- Define the slope m and y-intercept b
def m : ℚ := 2 / 3
def b : ℚ := -1 / 2

-- Define the product mb
def mb : ℚ := m * b

-- Prove the range of mb
theorem mb_range : -1 < mb ∧ mb < 0 := by
  unfold mb
  sorry

end mb_range_l598_598958


namespace find_cone_circumference_l598_598186

noncomputable def cone_circumference 
(V : ℝ) (h : ℝ) (π : ℝ) (radius circumference : ℝ) : Prop :=
V = 24 * π ∧ h = 6 ∧ radius = 2 * real.sqrt 3 ∧ circumference = 4 * real.sqrt 3 * π

theorem find_cone_circumference :
  ∀ (V h π radius circumference : ℝ), 
    V = 24 * π → h = 6 →
    radius = (2 : ℝ) * real.sqrt 3 →
    circumference = (4 : ℝ) * real.sqrt 3 * π →
    cone_circumference V h π radius circumference :=
begin
  intros V h π radius circumference V_eq h_eq radius_eq circumference_eq,
  rw V_eq, rw h_eq, rw radius_eq, rw circumference_eq,
  sorry
end

end find_cone_circumference_l598_598186


namespace analogies_correct_l598_598138

-- Definition of conditions
def cond_A (a b : ℕ) : Prop := (a * 3 = b * 3) → (a = b)
def cond_B (a b c : ℕ) : Prop := ((a + b) * c = a * c + b * c) → ((a * b) * c = (a * c) * (b * c))
def cond_C (a b c : ℕ) : Prop := (c ≠ 0) → ((a + b) * c = a * c + b * c) → ((a + b) / c = a / c + b / c)
def cond_D (a b n : ℕ) : Prop := ((a * b)^n = a^n * b^n) → ((a + b)^n = a^n + b^n)

-- Definition of the analogies correctness
def correct_option : Prop := cond_C

-- Theorem statement
theorem analogies_correct : correct_option :=
by sorry

end analogies_correct_l598_598138


namespace choose_president_and_secretary_l598_598057

theorem choose_president_and_secretary (total_members boys girls : ℕ) (h_total : total_members = 30) (h_boys : boys = 18) (h_girls : girls = 12) : 
  (boys * girls = 216) :=
by
  sorry

end choose_president_and_secretary_l598_598057


namespace min_capacity_for_raft_l598_598208

-- Define the weights of the animals
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal
def number_mice : ℕ := 5
def number_moles : ℕ := 3
def number_hamsters : ℕ := 4

-- Define the minimum weight capacity for the raft
def min_weight_capacity : ℕ := 140

-- Prove that the minimum weight capacity the raft must have to transport all animals is 140 grams.
theorem min_capacity_for_raft :
  (weight_mouse * 2 ≤ min_weight_capacity) ∧ 
  (∀ trip_weight, trip_weight ≥ min_weight_capacity → 
    (trip_weight = weight_mouse * 2 ∨ trip_weight = weight_mole * 2 ∨ trip_weight = weight_hamster * 2)) :=
by 
  sorry

end min_capacity_for_raft_l598_598208


namespace probability_sum_18_is_correct_l598_598504

/-- The faces of a dodecahedral die are labeled with digits from 1 to 12. -/
def faces : Finset ℕ := Finset.range' 1 12

/-- Total number of outcomes when rolling two dodecahedral dice. -/
def total_outcomes : ℕ := Finset.card (Finset.product faces faces)

/-- The pairs of values on the two dice that sum to 18. -/
def favorable_outcomes : Finset (ℕ × ℕ) := 
  Finset.filter (λ (p : ℕ × ℕ), p.1 + p.2 = 18) (Finset.product faces faces)

/-- The probability of rolling a sum of 18 with two dodecahedral dice. -/
def probability_of_sum_18 : ℚ := 
  (Finset.card favorable_outcomes : ℚ) / (Finset.card (Finset.product faces faces) : ℚ)

theorem probability_sum_18_is_correct : probability_of_sum_18 = 7 / 144 := 
by 
  sorry

end probability_sum_18_is_correct_l598_598504


namespace moon_speed_km_per_hour_l598_598937

theorem moon_speed_km_per_hour : 
  ∀ (speed_per_sec : ℝ) (num_sec_in_hour : ℝ), speed_per_sec = 1.04 → num_sec_in_hour = 3600 → 
  speed_per_sec * num_sec_in_hour = 3744 :=
by
  intros speed_per_sec num_sec_in_hour h_speed h_sec
  rw [h_speed, h_sec]
  norm_num
  sorry

end moon_speed_km_per_hour_l598_598937


namespace abs_neg_three_halves_l598_598484

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end abs_neg_three_halves_l598_598484


namespace arctan_sum_identity_l598_598239

theorem arctan_sum_identity (n : ℕ) :
  (∑ k in Finset.range n, Real.arctan (1 / (2 * (k + 1)^2))) = Real.arctan (n / (n + 1)) := by
  sorry

end arctan_sum_identity_l598_598239


namespace total_animals_sighted_l598_598256

theorem total_animals_sighted (lions_saturday elephants_saturday buffaloes_sunday leopards_sunday rhinos_monday warthogs_monday : ℕ)
(hlions_saturday : lions_saturday = 3)
(helephants_saturday : elephants_saturday = 2)
(hbuffaloes_sunday : buffaloes_sunday = 2)
(hleopards_sunday : leopards_sunday = 5)
(hrhinos_monday : rhinos_monday = 5)
(hwarthogs_monday : warthogs_monday = 3) :
  lions_saturday + elephants_saturday + buffaloes_sunday + leopards_sunday + rhinos_monday + warthogs_monday = 20 :=
by
  -- This is where the proof will be, but we are skipping the proof here.
  sorry

end total_animals_sighted_l598_598256


namespace first_player_wins_l598_598522

theorem first_player_wins (n : ℕ) : 
  n = 10000000 → 
  ∃ f : ℕ → ℕ, 
  (∀ m, m ≤ n → ∃ P : ℕ, Prime P ∧ ∃ k : ℕ, m = P^k) →
  (∀ t, ∃ m, m ≤ n ∧ m % 6 = 0) →
  true := 
by 
  sorry

end first_player_wins_l598_598522


namespace solve_complex_z_l598_598078

theorem solve_complex_z (z : ℂ) (h : z - 2 * complex.I = 3 + 7 * complex.I) : z = 3 + 9 * complex.I := by
  sorry

end solve_complex_z_l598_598078


namespace main_theorem_l598_598407

variable {A : Type*} [MetricSpace A] [NormedGroup A]

structure Triangle (A : Type*) [MetricSpace A] where
  A1 A2 A3 : A

noncomputable def circumcenter {A : Type*} [MetricSpace A] [NormedGroup A]
  (T : Triangle A) : A := sorry

noncomputable def circumradius {A : Type*} [MetricSpace A] [NormedGroup A]
  (T : Triangle A) : ℝ := sorry

variable {Q : Type*} [MetricSpace Q] [NormedGroup Q]
variable (A1 A2 A3 A4 : Q)
variable (O1 O2 O3 O4 : Q)
variable (r1 r2 r3 r4 : ℝ)

-- Assume each O_i and r_i are circumcentre and circumradius
axiom AO1 : O1 = circumcenter {A1 := A2, A2 := A3, A3 := A4}
axiom AO2 : O2 = circumcenter {A1 := A1, A2 := A3, A3 := A4}
axiom AO3 : O3 = circumcenter {A1 := A1, A2 := A2, A3 := A4}
axiom AO4 : O4 = circumcenter {A1 := A1, A2 := A2, A3 := A3}
axiom AR1 : r1 = circumradius {A1 := A2, A2 := A3, A3 := A4}
axiom AR2 : r2 = circumradius {A1 := A1, A2 := A3, A3 := A4}
axiom AR3 : r3 = circumradius {A1 := A1, A2 := A2, A3 := A4}
axiom AR4 : r4 = circumradius {A1 := A1, A2 := A2, A3 := A3}

-- Distance function (assuming a metric space)
noncomputable def distance (x y : Q) : ℝ := sorry

-- Main statement
theorem main_theorem :
  (1 / (distance O1 A1 ^ 2 - r1 ^ 2)) +
  (1 / (distance O2 A2 ^ 2 - r2 ^ 2)) +
  (1 / (distance O3 A3 ^ 2 - r3 ^ 2)) +
  (1 / (distance O4 A4 ^ 2 - r4 ^ 2)) = 0 := sorry

end main_theorem_l598_598407


namespace correct_statement_l598_598928

-- Definitions for the question
def monomial1 (x y z : ℕ) := x^3 * y * z^4
def degree1 := 3 + 1 + 4

def monomial2 (a b : ℕ) := -(π * a^2 * b^3) / 2
def degree2 := 2 + 3

def polynomial1 (a b : ℕ) := 2 * a^2 * b - a * b - 1
def degree3 := 3

def binomial1 (x y : ℕ) := x^2 * y + 1
def degree4 := 2 + 1

-- Theorem to verify
theorem correct_statement :
  x^2 * y + 1 = 3 :=
sorry

end correct_statement_l598_598928


namespace ab_cd_congruence_l598_598410

theorem ab_cd_congruence (p : ℕ) (hp : fact (nat.prime p)) (hp7 : p > 7) 
  (A : finset (zmod p)) (hA1 : A ⊆ finset.range p) (hA2 : (A.card : ℕ) ≥ (p - 1) / 2) :
  ∀ r : zmod p, ∃ a b c d ∈ A, a * b - c * d = r :=
by sorry

end ab_cd_congruence_l598_598410


namespace cubic_polynomial_r_value_l598_598603

theorem cubic_polynomial_r_value :
  ∃ (r : ℕ → ℚ), (∀ n ∈ {1, 2, 3, 4}, r n = 1 / n^3) ∧ r 5 = -1 / 12 :=
begin
  sorry
end

end cubic_polynomial_r_value_l598_598603


namespace smallest_positive_integer_a_l598_598360

theorem smallest_positive_integer_a (a : ℕ) (hpos : a > 0) :
  (∃ k, 5880 * a = k ^ 2) → a = 15 := 
by
  sorry

end smallest_positive_integer_a_l598_598360


namespace perfect_squares_with_specific_digits_count_perfect_squares_less_than_5000_with_digits_4_5_6_l598_598350

theorem perfect_squares_with_specific_digits (n : ℕ) (h : n < 5000) (digit : ℕ) 
    (h_digit : digit = 4 ∨ digit = 5 ∨ digit = 6) : Σ' (s : ℕ), s < 71 ∧ (s * s) % 10 = digit :=
begin
  sorry
end

theorem count_perfect_squares_less_than_5000_with_digits_4_5_6 : 
    (finset.univ.filter (λ n, n < 5000 ∧ ((n % 10 = 4) ∨ (n % 10 = 5) ∨ (n % 10 = 6)))).card = 35 :=
begin
  sorry
end

end perfect_squares_with_specific_digits_count_perfect_squares_less_than_5000_with_digits_4_5_6_l598_598350


namespace remaining_ribbon_l598_598000

-- Definitions of the conditions
def total_ribbon : ℕ := 18
def gifts : ℕ := 6
def ribbon_per_gift : ℕ := 2

-- The statement to prove the remaining ribbon
theorem remaining_ribbon 
  (initial_ribbon : ℕ) (num_gifts : ℕ) (ribbon_each_gift : ℕ) 
  (H1 : initial_ribbon = total_ribbon) 
  (H2 : num_gifts = gifts) 
  (H3 : ribbon_each_gift = ribbon_per_gift) : 
  initial_ribbon - (ribbon_each_gift * num_gifts) = 6 := 
  by 
    simp [H1, H2, H3, total_ribbon, gifts, ribbon_per_gift]
    linarith
    sorry 

end remaining_ribbon_l598_598000


namespace monotonic_decreasing_interval_l598_598084

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 + x + 1) * Real.exp x

-- Define the derivative of f(x)
def f_prime (x : ℝ) : ℝ := (x^2 + 3 * x + 2) * Real.exp x

-- State the theorem we want to prove
theorem monotonic_decreasing_interval : 
  ∀ x ∈ set.Ioo (-2 : ℝ) (-1 : ℝ), f_prime x < 0 :=
by 
  intros x hx
  sorry

end monotonic_decreasing_interval_l598_598084


namespace part1_part2_l598_598696

-- Defining the sequence and sum.

variable {a : ℝ}
variable {a_n : ℕ → ℝ} -- sequence a_n
variable {S_n : ℕ → ℝ} -- sum S_n

-- Given conditions
variable h_1 : a_n 0 = a -- first term is a
variable h_2 : (n : ℕ) → n ≥ 2 → a_n n ≠ 0 -- a_n ≠ 0 for n ≥ 2
variable h_3 : S_n n = a_n n + S_n (n-1)

-- Required theorem statements

-- (1) If the sequence {a_n} is an arithmetic sequence, then a = 3
theorem part1 (h_arithmetic : ∀ (n:ℕ), a_n (n+1) - a_n n = a_n 2 - a_n 1) :
  a = 3 := sorry

-- (2) The set of values M for a such that the sequence {a_n} is increasing is (1, ∞)
theorem part2 (h_increasing : ∀ n ≥ 2, a_n (n+1) > a_n n) :
  { a : ℝ | ∀ n, a_n (n+1) > a_n n } = { a : ℝ | a > 1 } := sorry

end part1_part2_l598_598696


namespace distribution_ways_l598_598474

open Finset

-- Define the condition that each box must have at least as many balls as its number
def valid_distribution (x y z : ℕ) : Prop :=
  (x >= 1) ∧ (y >= 2) ∧ (z >= 3) ∧ (x + y + z = 9)

-- The main theorem stating the number of valid ways to distribute the balls
theorem distribution_ways : 
  ∃! (d : ℕ × ℕ × ℕ), valid_distribution d.1 d.2 d.3 ∧ 
    number_of_valid_ways d.1 d.2 d.3 = 10 := 
sorry

end distribution_ways_l598_598474


namespace base_of_numbering_system_l598_598845

-- Definitions based on conditions
def num_children := 100
def num_boys := 24
def num_girls := 32

-- Problem statement: Prove the base of numbering system used is 6
theorem base_of_numbering_system (n: ℕ) (h: n ≠ 0):
    n^2 = (2 * n + 4) + (3 * n + 2) → n = 6 := 
  by
    sorry

end base_of_numbering_system_l598_598845


namespace abs_neg_frac_l598_598480

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end abs_neg_frac_l598_598480


namespace plumber_assignment_l598_598143

theorem plumber_assignment :
  let plumbers := 5
  let areas := 3
  let total_plans := 150
  (∃ (grouping : list (list ℕ)), 
    (length grouping = 3 ∧ ∀ g ∈ grouping, (1 ≤ length g ∧ length g ≤ 3) ∧ list.sum (list.map length grouping) = 5 ∧ 
    list.distinct grouping)) → 
  ∃ (plans : ℕ), plans = total_plans := 
by
  sorry

end plumber_assignment_l598_598143


namespace sum_even_factors_630_l598_598127

theorem sum_even_factors_630 : 
  ∀ (n : ℕ), (n = 630 → ∃ d : ℕ, n = 2 * d) → 
  (∑ x in (606 : Finset ℕ).divisors.filter (λ d, d % 2 = 0), x) = 1248 :=
by
  intro n h1
  obtain ⟨d, hd⟩ := h1 rfl
  sorry

end sum_even_factors_630_l598_598127


namespace max_non_attacking_kings_l598_598562

-- Define the chessboard size
def ChessboardSize := 8

-- Define what it means for a king to attack another
def king_attacks (x1 y1 x2 y2 : ℕ) : Prop :=
  abs (x1 - x2) ≤ 1 ∧ abs (y1 - y2) ≤ 1

-- Define the main theorem statement
theorem max_non_attacking_kings : 
  ∃ (k : fin ChessboardSize → fin ChessboardSize → Prop), 
  (∀ i j m n, k i j → k m n → (i ≠ m ∨ j ≠ n) ∧ ¬king_attacks i j m n) ∧ 
  (∃! a, (∑ i j, if k i j then 1 else 0) = a) ∧ a = 16 := 
sorry

end max_non_attacking_kings_l598_598562


namespace inequality_sqrt_sum_le_sqrt_sum_sq_l598_598457

theorem inequality_sqrt_sum_le_sqrt_sum_sq (a b c : ℝ) 
  (ha : -1 ≤ a ∧ a ≤ 1) (hb : -1 ≤ b ∧ b ≤ 1) (hc : -1 ≤ c ∧ c ≤ 1) :
  sqrt (1 - a^2) + sqrt (1 - b^2) + sqrt (1 - c^2) ≤ sqrt (9 - (a + b + c)^2) :=
by
  sorry

end inequality_sqrt_sum_le_sqrt_sum_sq_l598_598457


namespace most_suitable_for_comprehensive_survey_l598_598141

-- Definitions of the survey options
inductive SurveyOption
| A
| B
| C
| D

-- Condition definitions based on the problem statement
def comprehensive_survey (option : SurveyOption) : Prop :=
  option = SurveyOption.B

-- The theorem stating that the most suitable survey is option B
theorem most_suitable_for_comprehensive_survey : ∀ (option : SurveyOption), comprehensive_survey option ↔ option = SurveyOption.B :=
by
  intro option
  sorry

end most_suitable_for_comprehensive_survey_l598_598141


namespace raft_minimum_capacity_l598_598206

theorem raft_minimum_capacity (n_mice n_moles n_hamsters : ℕ)
  (weight_mice weight_moles weight_hamsters : ℕ)
  (total_weight : ℕ) :
  n_mice = 5 →
  weight_mice = 70 →
  n_moles = 3 →
  weight_moles = 90 →
  n_hamsters = 4 →
  weight_hamsters = 120 →
  (∀ (total_weight : ℕ), total_weight = n_mice * weight_mice + n_moles * weight_moles + n_hamsters * weight_hamsters) →
  (∃ (min_capacity: ℕ), min_capacity ≥ 140) :=
by
  intros
  sorry

end raft_minimum_capacity_l598_598206


namespace odometer_problem_l598_598647

theorem odometer_problem :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a + b + c ≤ 10 ∧ (11 * c - 10 * a - b) % 6 = 0 ∧ a^2 + b^2 + c^2 = 54 :=
by
  sorry

end odometer_problem_l598_598647


namespace find_intersections_l598_598254

-- Definition of the parametric curve C_1
def parametric_curve_C1 (α : ℝ) : ℝ × ℝ :=
  (2 + cos α, 2 + sin α)

-- Definition of the line C_2 in Cartesian coordinates
def line_C2 (x : ℝ) : ℝ :=
  sqrt 3 * x

-- Polar coordinate system setup with O as the pole and positive x-axis as the polar axis
def intersection_points (ρ1 ρ2 : ℝ) (OA OB : ℝ) (θ : ℝ) : Prop :=
  θ = π / 3 ∧
  (ρ1^2 - (2 * sqrt 3 + 2) * ρ1 + 7 = 0) ∧
  (ρ2^2 - (2 * sqrt 3 + 2) * ρ2 + 7 = 0) ∧
  (OA = ρ1) ∧
  (OB = ρ2)

theorem find_intersections :
  ∃ OA OB ρ1 ρ2,
    intersection_points ρ1 ρ2 OA OB (π / 3) →
    (1 / OA + 1 / OB) = (2 * sqrt 3 + 2) / 7 :=
sorry

end find_intersections_l598_598254


namespace parametric_to_polar_l598_598510

-- Define the parametric equations
def parametric_eq (x y t : ℝ) : Prop :=
  x = 1 + sqrt 3 * t ∧ y = sqrt 3 - t

-- Define the expected polar form equation
def polar_eq (ρ θ : ℝ) : Prop :=
  ρ * sin (θ + π / 6) = 2

-- The theorem stating that given the parametric equations, the polar form is as stated.
theorem parametric_to_polar (ρ θ t : ℝ) :
  ∃ x y : ℝ, parametric_eq x y t → polar_eq ρ θ :=
sorry

end parametric_to_polar_l598_598510


namespace Gretel_makes_more_than_Hansel_l598_598341

-- Defining initial salaries and raises
def initial_salary_Hansel := 30000
def raise_Hansel := 0.10
def initial_salary_Gretel := 30000
def raise_Gretel := 0.15
def initial_salary_Rapunzel := 40000
def raise_Rapunzel := 0.08
def initial_salary_Rumpelstiltskin := 35000
def raise_Rumpelstiltskin := 0.12

-- Defining new salaries after the raise
def new_salary (initial_salary : ℕ) (raise : ℝ) : ℝ := initial_salary * (1 + raise)

def new_salary_Hansel := new_salary initial_salary_Hansel raise_Hansel
def new_salary_Gretel := new_salary initial_salary_Gretel raise_Gretel
def new_salary_Rapunzel := new_salary initial_salary_Rapunzel raise_Rapunzel
def new_salary_Rumpelstiltskin := new_salary initial_salary_Rumpelstiltskin raise_Rumpelstiltskin

-- Proving Gretel makes $1,500 more than Hansel after the raise
theorem Gretel_makes_more_than_Hansel : 
  new_salary_Gretel - new_salary_Hansel = (1500 : ℝ) := by
  sorry

end Gretel_makes_more_than_Hansel_l598_598341


namespace cara_constant_speed_l598_598501

noncomputable def cara_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

theorem cara_constant_speed
  ( distance : ℕ := 120 )
  ( dan_speed : ℕ := 40 )
  ( dan_time_offset : ℕ := 1 ) :
  cara_speed distance (3 + dan_time_offset) = 30 := 
by
  -- skip proof
  sorry

end cara_constant_speed_l598_598501


namespace log_eq_root_product_theorem_l598_598371

noncomputable def log_eq_root_product : Prop :=
  ∀ (x1 x2 : ℝ), (∃ x, (log x)^2 + (log 2 + log 3) * log x + log 2 * log 3 = 0) → x1 * x2 = 1/6

theorem log_eq_root_product_theorem : log_eq_root_product :=
sorry

end log_eq_root_product_theorem_l598_598371


namespace sin_angle_BAD_equal_zero_l598_598459

theorem sin_angle_BAD_equal_zero
  (A B C D : Type) [RightTriangle A B C]
  (h_AB : length A B = 2)
  (h_BC : length B C = 2)
  (h_perim_ACD : perimeter A C D = (1 / 2) * perimeter A B C)
  (h_right_angle_C : right_angle ∠C)
  (h_angle_isosceles_right : isosceles_right_triangle A B C)
  : sin (2 * ∠BAD) = 0 :=
  sorry

end sin_angle_BAD_equal_zero_l598_598459


namespace mary_added_peanuts_l598_598103

-- Defining the initial number of peanuts
def initial_peanuts : ℕ := 4

-- Defining the final number of peanuts
def total_peanuts : ℕ := 10

-- Defining the number of peanuts added by Mary
def peanuts_added : ℕ := total_peanuts - initial_peanuts

-- The proof problem is to show that Mary added 6 peanuts
theorem mary_added_peanuts : peanuts_added = 6 :=
by
  -- We leave the proof part as a sorry as per instruction
  sorry

end mary_added_peanuts_l598_598103


namespace cone_volume_proof_l598_598368

noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_proof :
  (cone_volume 1 (Real.sqrt 3)) = (Real.sqrt 3 / 3) * Real.pi :=
by
  sorry

end cone_volume_proof_l598_598368


namespace isosceles_triangle_area_l598_598905

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem isosceles_triangle_area :
  let a := 13
  let b := 13
  let c := 24
  triangle_area a b c = 60 := by
  sorry

end isosceles_triangle_area_l598_598905


namespace sum_of_even_squares_mod_11_l598_598122

theorem sum_of_even_squares_mod_11 :
  (∑ k in finset.filter (λ x : ℕ, x % 2 = 0) (finset.range 21), (k^2 : ℤ) % 11) % 11 = 0 := sorry

end sum_of_even_squares_mod_11_l598_598122


namespace find_symmetric_sequence_l598_598292

noncomputable def symmetric_sequence (b : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → b k = b (n - k + 1)

noncomputable def arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d, b 2 = b 1 + d ∧ b 3 = b 2 + d ∧ b 4 = b 3 + d

theorem find_symmetric_sequence :
  ∃ b : ℕ → ℤ, symmetric_sequence b 7 ∧ arithmetic_sequence b ∧ b 1 = 2 ∧ b 2 + b 4 = 16 ∧
  (b 1 = 2 ∧ b 2 = 5 ∧ b 3 = 8 ∧ b 4 = 11 ∧ b 5 = 8 ∧ b 6 = 5 ∧ b 7 = 2) :=
by {
  sorry
}

end find_symmetric_sequence_l598_598292


namespace circle_equation_with_diameter_AB_l598_598701

-- Definition of the given points A and B
def A : ℝ × ℝ := (3, -2)
def B : ℝ × ℝ := (-5, 4)

-- The statement of the theorem: 
-- The equation of the circle with AB as its diameter is (x + 1)^2 + (y - 1)^2 = 25
theorem circle_equation_with_diameter_AB :
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    let eq := λ x y, (x - C.1)^2 + (y - C.2)^2 = r^2 in
    C = (-1, 1) ∧ r = 5 ∧ eq x y = (x + 1)^2 + (y - 1)^2 := 
begin
  sorry,
end

end circle_equation_with_diameter_AB_l598_598701


namespace arithmetic_mean_of_4_and_16_l598_598312

theorem arithmetic_mean_of_4_and_16 : 
  let m := (4 + 16) / 2 
  in m = 10 := 
by
  let m := (4 + 16) / 2;
  show m = 10 from sorry

end arithmetic_mean_of_4_and_16_l598_598312


namespace simplify_fraction_l598_598865

-- Define the fractions and the product
def fraction1 : ℚ := 18 / 11
def fraction2 : ℚ := -42 / 45
def product : ℚ := 15 * fraction1 * fraction2

-- State the theorem to prove the correctness of the simplification
theorem simplify_fraction : product = -23 + 1 / 11 :=
by
  -- Adding this as a placeholder. The proof would go here.
  sorry

end simplify_fraction_l598_598865


namespace intersection_value_unique_l598_598520

theorem intersection_value_unique (x : ℝ) :
  (∃ y : ℝ, y = 8 / (x^2 + 4) ∧ x + y = 2) → x = 0 :=
by
  sorry

end intersection_value_unique_l598_598520


namespace class_B_has_21_fewer_than_class_A_l598_598216

/--
Given:
1. The school teaches 80 students in three classes.
2. 40% of the students are in class A.
3. There are 37 students in class C.

Prove:
Class B has 21 fewer students than Class A.
-/
theorem class_B_has_21_fewer_than_class_A :
  let TotalStudents := 80
  let PercentageStudentsInA := 40
  let StudentsInC := 37
  (0.40 * TotalStudents).toInt - (TotalStudents - ((0.40 * TotalStudents).toInt + StudentsInC)) = 21 :=
by
  sorry

end class_B_has_21_fewer_than_class_A_l598_598216


namespace false_inverse_proposition_l598_598886

theorem false_inverse_proposition (a b : ℝ) : (a^2 = b^2) → (a = b ∨ a = -b) := sorry

end false_inverse_proposition_l598_598886


namespace arithmetic_sequence_sum_l598_598779

open Nat

theorem arithmetic_sequence_sum (m n : Nat) (d : ℤ) (a_1 : ℤ)
    (hnm : n ≠ m)
    (hSn : (n * (2 * a_1 + (n - 1) * d) / 2) = n / m)
    (hSm : (m * (2 * a_1 + (m - 1) * d) / 2) = m / n) :
  ((m + n) * (2 * a_1 + (m + n - 1) * d) / 2) > 4 := by
  sorry

end arithmetic_sequence_sum_l598_598779


namespace natural_number_1981_l598_598962

theorem natural_number_1981 (x : ℕ) 
  (h1 : ∃ a : ℕ, x - 45 = a^2)
  (h2 : ∃ b : ℕ, x + 44 = b^2) :
  x = 1981 :=
sorry

end natural_number_1981_l598_598962


namespace angle_of_inclination_of_tangent_line_l598_598414

theorem angle_of_inclination_of_tangent_line (x : ℝ) (hP : x ≠ 0) :
    let y := sqrt x * (x + 1)
    let y' := (3*x + 1) / (2 * sqrt x)
    let theta := atan y'
    0 ≤ theta ∧ theta < π → (θ : ℝ) ∈ set.Ico (π / 3) (π / 2) := 
by
    intros
    sorry

end angle_of_inclination_of_tangent_line_l598_598414


namespace maximum_M_value_max_value_is_7_l598_598672

def J_k (k : ℕ) (h : k > 0) : ℕ :=
  let zeros := List.replicate k 0
  (2 * 10^(k + 2)) + 64

def M (k : ℕ) (h : k > 0) : ℕ := 
  Nat.factorization (J_k k h) 2

theorem maximum_M_value : ∀ (k : ℕ) (h : k > 0), M k h ≤ 7 := 
by {
  -- sorry will be replaced with actual proof
  sorry 
}

theorem max_value_is_7 : ∀ (k : ℕ) (h : k > 0), ∃ k₀, k₀ > 0 ∧ M k₀ h = 7 := 
by {
  -- sorry will be replaced with actual proof
  sorry 
}

end maximum_M_value_max_value_is_7_l598_598672


namespace find_percentage_ryegrass_in_seed_mixture_X_l598_598615

open Real

noncomputable def percentage_ryegrass_in_seed_mixture_X (R : ℝ) : Prop := 
  let proportion_X : ℝ := 2 / 3
  let percentage_Y_ryegrass : ℝ := 25 / 100
  let proportion_Y : ℝ := 1 / 3
  let final_percentage_ryegrass : ℝ := 35 / 100
  final_percentage_ryegrass = (R / 100 * proportion_X) + (percentage_Y_ryegrass * proportion_Y)

/-
  Given the conditions:
  - Seed mixture Y is 25 percent ryegrass.
  - A mixture of seed mixtures X (66.67% of the mixture) and Y (33.33% of the mixture) contains 35 percent ryegrass.

  Prove:
  The percentage of ryegrass in seed mixture X is 40%.
-/
theorem find_percentage_ryegrass_in_seed_mixture_X : 
  percentage_ryegrass_in_seed_mixture_X 40 := 
  sorry

end find_percentage_ryegrass_in_seed_mixture_X_l598_598615


namespace number_of_boys_in_second_class_l598_598841

def boys_in_first_class : ℕ := 28
def portion_of_second_class (b2 : ℕ) : ℚ := 7 / 8 * b2

theorem number_of_boys_in_second_class (b2 : ℕ) (h : portion_of_second_class b2 = boys_in_first_class) : b2 = 32 :=
by 
  sorry

end number_of_boys_in_second_class_l598_598841


namespace optionA_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_not_right_triangle_l598_598714

-- Four conditions for the triangle ABC
axiom condA : ∀ (A B C : ℝ), A + B = C
axiom condB : ∀ (A B C : ℝ), 2 * A = B ∧ 3 * A = C
axiom condC : ∀ (a b c : ℝ), a^2 = b^2 - c^2
axiom condD : ∀ (a b c : ℝ), a^2 = 5 ∧ b^2 = 12 ∧ c^2 = 13

-- The angles and sides of triangle ABC
variable (A B C : ℝ)
variable (a b c : ℝ)

-- The proof
theorem optionA_right_triangle : condA A B C → A + B = 90 := by 
  sorry
theorem optionB_right_triangle : condB A B C → C = 90 := by 
  sorry
theorem optionC_right_triangle : condC a b c → a^2 + c^2 = b^2 := by 
  sorry
theorem optionD_not_right_triangle : condD a b c → ¬(a^2 + b^2 = c^2) := by 
  sorry

end optionA_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_not_right_triangle_l598_598714


namespace tv_price_increase_percentage_l598_598091

theorem tv_price_increase_percentage (P Q : ℝ) (x : ℝ) :
  (P * (1 + x / 100) * Q * 0.8 = P * Q * 1.28) → x = 60 :=
by sorry

end tv_price_increase_percentage_l598_598091


namespace sin_theta_eq_cos_diff_pi_over_3_theta_eq_l598_598703

noncomputable def cos_theta : ℝ := -3/5
noncomputable def theta : ℝ := sorry  -- Placeholder for θ in (π/2, π)

-- We now state the theorems to be proven based on the given conditions.
theorem sin_theta_eq (h_cos : cos θ = cos_theta) (h_range : θ ∈ set.Ioc (π/2) π) : sin θ = 4/5 :=
by sorry

theorem cos_diff_pi_over_3_theta_eq (h_cos : cos θ = cos_theta) (h_range : θ ∈ set.Ioc (π/2) π) : 
  cos (π/3 - θ) = (4 * real.sqrt 3 - 3) / 10 :=
by sorry

end sin_theta_eq_cos_diff_pi_over_3_theta_eq_l598_598703


namespace valid_pairs_l598_598835

theorem valid_pairs :
    ∃ (a b : ℕ), 
    (a = 41 ∧ b = 271 ∨ a = 164 ∧ b = 271 ∨ a = 82 ∧ b = 542 ∨ a = 123 ∧ b = 813) ∧ 
    a ≤ b ∧ 
    (100 ≤ a + b ∧ a + b ≤ 999 ∧ 
     (∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ a + b = d1 * 100 + d2 * 10 + d3 ∧ d2 - d1 = d3 - d2)) ∧ 
    (10000 ≤ a * b ∧ a * b ≤ 99999 ∧ 
    ∃ (d : ℕ), a * b = d * 11111) := 
by 
  use (41, 271) 
  use (164, 271) 
  use (82, 542) 
  use (123, 813)
  sorry

end valid_pairs_l598_598835


namespace solve_for_x_l598_598868

theorem solve_for_x : 
  let x := (√(8^2 + 15^2)) / (√(49 + 36))
  in x = (17 * √85) / 85 :=
by
  sorry

end solve_for_x_l598_598868


namespace find_vector_l598_598328

-- Define matrix A
def A : Matrix (Fin 2) (Fin 2) ℚ := ![![2, 0], ![1, 1]]

-- Define vector beta
def β : Fin 2 → ℚ := ![1, 2]

-- Define vector α
def α : Fin 2 → ℚ := ![1/4, 5/4]

-- Define matrix square
def A_sq : Matrix (Fin 2) (Fin 2) ℚ := A.mul A

theorem find_vector : A_sq.mul_vec α = β := by
  sorry

end find_vector_l598_598328


namespace sum_of_other_endpoint_coordinates_l598_598049

theorem sum_of_other_endpoint_coordinates 
  (A B O : ℝ × ℝ)
  (hA : A = (6, -2)) 
  (hO : O = (3, 5)) 
  (midpoint_formula : (A.1 + B.1) / 2 = O.1 ∧ (A.2 + B.2) / 2 = O.2):
  (B.1 + B.2) = 12 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l598_598049


namespace sheila_attendance_l598_598858

noncomputable def sheila_attending_prob : ℝ :=
  let prob_rain := 0.5
  let prob_go_if_rain := 0.4
  let prob_go_if_sunny := 0.9
  let prob_finish_homework := 0.7
  let prob_rain_and_attend := prob_rain * prob_go_if_rain * prob_finish_homework
  let prob_sunny_and_attend := (1 - prob_rain) * prob_go_if_sunny * prob_finish_homework
  prob_rain_and_attend + prob_sunny_and_attend

theorem sheila_attendance :
  sheila_attending_prob = 0.455 :=
by
  unfold sheila_attending_prob
  rw [← mul_assoc, ← mul_assoc]
  norm_num
  sorry

end sheila_attendance_l598_598858


namespace find_b_l598_598693

open Matrix

def direction_vector (p1 p2 : Matrix (Fin 2) (Fin 1) ℝ) : Matrix (Fin 2) (Fin 1) ℝ :=
  p2 - p1

def scaled_vector (v : Matrix (Fin 2) (Fin 1) ℝ) (c : ℝ) : Matrix (Fin 2) (Fin 1) ℝ :=
  c • v

theorem find_b (p1 p2 : Matrix (Fin 2) (Fin 1) ℝ)
  (h1 : p1 = ![-3; 4])
  (h2 : p2 = ![2; -1])
  (v := direction_vector p1 p2)
  (scaled_v := scaled_vector v (1 / 5))
  (target_v := ![scaled_v[0 0], scaled_v[1 0]]) :
  target_v = ![1, -1] →
  (b : ℝ) (h : target_v = ![b, -1]) := rfl

end find_b_l598_598693


namespace olivia_race_time_l598_598255

variable (O E : ℕ)

theorem olivia_race_time (h1 : O + E = 112) (h2 : E = O - 4) : O = 58 :=
sorry

end olivia_race_time_l598_598255


namespace max_value_max_value_achieved_l598_598314

variable (n : ℕ)
variable (a b : Fin n → ℝ)

theorem max_value 
  (h0 : n ≥ 5) 
  (h1 : ∀ i, 0 ≤ a i) 
  (h2 : ∀ i, 0 ≤ b i) 
  (h3 : ∑ i, (a i)^2 = 1)
  (h4 : ∑ i, b i = 1) :
  (∑ i, (a i)^(1+b i)) ≤ Real.sqrt (n-1) :=
sorry

theorem max_value_achieved 
  (h0 : n ≥ 5) 
  (h1 : ∀ i, 0 ≤ a i) 
  (h2 : ∀ i, 0 ≤ b i) 
  (h3 : ∑ i, (a i)^2 = 1)
  (h4 : ∑ i, b i = 1) :
  ∃ (a' b' : Fin n → ℝ), a' = λ i, if i < n-1 then 1 / Real.sqrt (n-1) else 0 
  ∧ b' = λ i, if i = n-1 then 1 else 0 
  ∧ (∑ i, (a' i)^(1+b' i)) = Real.sqrt (n-1) :=
sorry

end max_value_max_value_achieved_l598_598314


namespace radius_of_circle_l598_598332

-- Define the circle in polar coordinates equation
def polar_circle_eq := ∀ (ρ θ : ℝ), ρ^2 + 2 * real.sqrt 2 * ρ * real.sin (θ - real.pi / 4) - 4 = 0

-- Define the radius of circle C
def radius := real.sqrt 6

-- The main theorem state that the radius of the circle defined by polar equation is sqrt(6)
theorem radius_of_circle : 
  (∃ (ρ θ : ℝ), polar_circle_eq ρ θ) → (∃ (r : ℝ), r = radius) :=
sorry

end radius_of_circle_l598_598332


namespace blue_fractions_denominators_too_large_l598_598938

theorem blue_fractions_denominators_too_large
  (a1 a2 a3 a4 a5 : ℚ)
  (q1 q2 q3 q4 q5 : ℤ)
  (h1 : a1 = ⟨p1, q1⟩) (h2 : a2 = ⟨p2, q2⟩)
  (h3 : a3 = ⟨p3, q3⟩) (h4 : a4 = ⟨p4, q4⟩)
  (h5 : a5 = ⟨p5, q5⟩)
  (h_q1 : odd q1 ∧ q1 > 10^10)
  (h_q2 : odd q2 ∧ q2 > 10^10)
  (h_q3 : odd q3 ∧ q3 > 10^10)
  (h_q4 : odd q4 ∧ q4 > 10^10)
  (h_q5 : odd q5 ∧ q5 > 10^10) : 
  (¬ (∀ i ∈ {1, 2, 3, 4, 5}, (let s :=
      match i with
      | 1 => a1 + a2
      | 2 => a2 + a3
      | 3 => a3 + a4
      | 4 => a4 + a5
      | 5 => a5 + a1
      end in s.denom < 100))) := 
sorry

end blue_fractions_denominators_too_large_l598_598938


namespace incenter_circumcenter_distance_l598_598969

-- Define the context of the problem
variables {A B C : Type*} [euclidean_space A]
variables (A B C : A) -- Vertices of the triangle
variables (AB AC BC : ℝ) -- Side lengths of the triangle
variables [fact (AB = 8)] [fact (AC = 15)] [fact (BC = 17)]
variables {I O : A} -- Incenter and circumcenter

-- Define the distance function
noncomputable def distance (x y : A) : ℝ := sorry

-- The theorem stating the desired result
theorem incenter_circumcenter_distance :
  AB = 8 → AC = 15 → BC = 17 →
  let I := incenter A B C in
  let O := circumcenter A B C in
  distance I O = (√85) / 2 :=
sorry

end incenter_circumcenter_distance_l598_598969


namespace greatest_A_l598_598018

def f (r₂ r₃ x : ℝ) : ℝ :=
  x^2 - r₂ * x + r₃

def sequence_g (r₂ r₃ : ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := f r₂ r₃ (sequence_g r₂ r₃ n)

lemma condition1 (r₂ r₃ : ℝ) (i : ℕ) (h : i ≤ 2011) :
  sequence_g r₂ r₃ (2 * i) < sequence_g r₂ r₃ (2 * i + 1) ∧
  sequence_g r₂ r₃ (2 * i + 1) > sequence_g r₂ r₃ (2 * i + 2) :=
sorry

lemma condition2 (r₂ r₃ : ℝ) :
  ∃ j : ℕ, ∀ i > j, sequence_g r₂ r₃ (i + 1) > sequence_g r₂ r₃ i :=
sorry

lemma condition3 (r₂ r₃ : ℝ) :
  ∀ M : ℝ, ∃ n : ℕ, sequence_g r₂ r₃ n > M :=
sorry

noncomputable def find_A : ℝ :=
  2

theorem greatest_A (r₂ r₃ : ℝ) :
  (condition1 r₂ r₃) →
  (condition2 r₂ r₃) →
  (condition3 r₂ r₃) →
  2 ≤ |r₂| :=
sorry

end greatest_A_l598_598018


namespace ratio_of_areas_GHI_to_JKL_l598_598542

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l598_598542


namespace max_projection_sum_l598_598169

-- Define the given conditions
def edge_length : ℝ := 2

def projection_front_view (length : ℝ) : Prop := length = edge_length
def projection_side_view (length : ℝ) : Prop := ∃ a : ℝ, a = length
def projection_top_view (length : ℝ) : Prop := ∃ b : ℝ, b = length

-- State the theorem
theorem max_projection_sum (a b : ℝ) (ha : projection_side_view a) (hb : projection_top_view b) :
  a + b ≤ 4 := sorry

end max_projection_sum_l598_598169


namespace perfect_squares_with_specific_digits_count_perfect_squares_less_than_5000_with_digits_4_5_6_l598_598349

theorem perfect_squares_with_specific_digits (n : ℕ) (h : n < 5000) (digit : ℕ) 
    (h_digit : digit = 4 ∨ digit = 5 ∨ digit = 6) : Σ' (s : ℕ), s < 71 ∧ (s * s) % 10 = digit :=
begin
  sorry
end

theorem count_perfect_squares_less_than_5000_with_digits_4_5_6 : 
    (finset.univ.filter (λ n, n < 5000 ∧ ((n % 10 = 4) ∨ (n % 10 = 5) ∨ (n % 10 = 6)))).card = 35 :=
begin
  sorry
end

end perfect_squares_with_specific_digits_count_perfect_squares_less_than_5000_with_digits_4_5_6_l598_598349


namespace tennis_tournament_rounds_needed_l598_598165

theorem tennis_tournament_rounds_needed (n : ℕ) (total_participants : ℕ) (win_points loss_points : ℕ) (get_point_no_pair : ℕ) (elimination_loss : ℕ) :
  total_participants = 1152 →
  win_points = 1 →
  loss_points = 0 →
  get_point_no_pair = 1 →
  elimination_loss = 2 →
  n = 14 :=
by
  sorry

end tennis_tournament_rounds_needed_l598_598165


namespace smallest_term_is_8th_l598_598299
  
def sequence (a₀ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₀ + n * d
def sumSequence (a₀ : ℤ) (d : ℤ) (n : ℕ) : ℤ := ∑ i in Finset.range n, sequence a₀ d i

theorem smallest_term_is_8th :
  ∀ n : ℕ, ∑ i in Finset.range (n+1), sequence (-15) 2 i = (n - 8)^2 - 64 →
  ∃ n0 : ℕ, ∀ n₁ : ℕ, 
  ∑ i in Finset.range (n₀ + 1), sequence (-15) 2 i ≥ 
  ∑ i in Finset.range (n₁ + 1), sequence (-15) 2 i :=
begin
  sorry
end

end smallest_term_is_8th_l598_598299


namespace lcm_165_396_l598_598919

-- Given conditions
def factor_165 : multiset ℕ := {3, 5, 11}
def factor_396 : multiset ℕ := {2, 2, 3, 3, 11}

-- Definition to calculate LCM given factorizations
def lcm_of_multisets (s₁ s₂ : multiset ℕ) : ℕ :=
  multiset.fold (λ a b, a * (b : ℕ)) 1 (multiset.union s₁ s₂)

-- Theorem to prove
theorem lcm_165_396 : lcm_of_multisets factor_165 factor_396 = 1980 :=
by
  sorry  -- proof goes here but is not required for this task

end lcm_165_396_l598_598919


namespace largest_prime_to_test_l598_598108

theorem largest_prime_to_test (n : ℕ) (h1 : 1100 ≤ n) (h2 : n ≤ 1150) : 
  ∃ p, prime p ∧ p = 31 ∧ p ≤ Int.floor (Real.sqrt 1150) :=
by
  sorry

end largest_prime_to_test_l598_598108


namespace monotonicity_inequality_range_l598_598326

-- Definitions for monotonicity
def f (a x : ℝ) := a * exp x - x
def f' (a x : ℝ) := a * exp x - 1

-- Monotonicity (Part I)
theorem monotonicity (a : ℝ) :
  (∀ x : ℝ, f' a x < 0) ↔ a ≤ 0 ∧
  (a > 0 → (∀ x : ℝ, x < real.log a → f' a x < 0) ∧ (∀ x : ℝ, x > real.log a → f' a x > 0)) :=
sorry

-- Definitions for inequality conditions (Part II)
def g (x : ℝ) := (1 + x * exp x) / exp (2 * x)
def h (a x : ℝ) := a * exp x - x - exp (-x)

-- Maximum value function g(x) on interval [1, 2]
def max_g (g : ℝ → ℝ) (a b : ℝ) := max (g a) (g b)

-- Range of 'a' (Part II)
theorem inequality_range (a : ℝ) (x : ℝ) (h₁ : 1 ≤ x) (h₂ : x ≤ 2) :
  (f a x ≥ exp (-x)) ↔ a ≥ g 1 :=
sorry

end monotonicity_inequality_range_l598_598326


namespace part_a_l598_598933

theorem part_a (n : ℕ) (h_n : n ≥ 3) (x : Fin n → ℝ) (hx : ∀ i j : Fin n, i ≠ j → x i ≠ x j) (hx_pos : ∀ i : Fin n, 0 < x i) :
  ∃ (i j : Fin n), i ≠ j ∧ 0 < (x i - x j) / (1 + (x i) * (x j)) ∧ (x i - x j) / (1 + (x i) * (x j)) < Real.tan (π / (2 * (n - 1))) :=
by
  sorry

end part_a_l598_598933


namespace sin_cot_identity_l598_598062

theorem sin_cot_identity (n : ℕ) (x : ℝ) (h : ∀ k ≤ n, sin (2^k * x) ≠ 0) : 
  (List.sum (List.map (λ k, 1 / sin (2^k * x)) (List.range (n + 1)))) = 
  cot x - cot (2^n * x) := by
  sorry

end sin_cot_identity_l598_598062


namespace x_intercept_of_rotated_line_l598_598833

noncomputable def line_l := { p : ℝ × ℝ | 2 * p.1 - 7 * p.2 + 35 = 0 }
def point_P := (10, 10 : ℝ)
def angle := real.pi / 6  -- pi / 6 radians is 30 degrees

def rotate (l1 : set (ℝ × ℝ)) (θ : ℝ) (p : ℝ × ℝ) : set (ℝ × ℝ) := sorry  -- Define the rotation

def line_k := rotate line_l angle point_P

noncomputable def x_intercept (l : set (ℝ × ℝ)) : ℝ := sorry  -- Define a function to find x-intercept

theorem x_intercept_of_rotated_line :
  x_intercept line_k = c := sorry

end x_intercept_of_rotated_line_l598_598833


namespace sufficient_but_not_necessary_condition_l598_598686

-- The conditions of the problem
variables (a b : ℝ)

-- The proposition to be proved
theorem sufficient_but_not_necessary_condition (h : a + b = 1) : 4 * a * b ≤ 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l598_598686


namespace tangent_line_eq_function_monotonicity_log_inequality_l598_598322

/-- Part 1 - Equation of the tangent line to the graph of the function at x = 0 for a = 2-/
theorem tangent_line_eq (f : ℝ → ℝ) (a : ℝ) (h : a = 2) (f_def : ∀ x, f x = ln(x + 1) + (a * x) / (x + 1)) :
    (tangent_line_eq := 3) :=
sorry

/-- Part 2 - Monotonicity of the function depending on 'a' -/
theorem function_monotonicity (f : ℝ → ℝ) (a : ℝ) (h_a_ge_0 : a ≥ 0) (h_a_lt_0 : a < 0) 
    (f_def : ∀ x, f x = ln(x + 1) + (a * x) / (x + 1)) :
    ((∀ x, x > -1 → f' x > 0) ∨ ((∀ x ∈ (-1, -1 - a), f' x < 0) ∧ (∀ x > -1 - a, f' x > 0))) :=
sorry

/-- Part 3 - Prove the given inequality -/
theorem log_inequality (n : ℕ) (h : 0 < n) :
    ln (1 + (1 : ℝ) / n) > (1 : ℝ) / n - (1 : ℝ) / (n ^ 2) :=
sorry

end tangent_line_eq_function_monotonicity_log_inequality_l598_598322


namespace area_ratio_of_rotated_triangles_l598_598061

/-- 
Let O be a point on side AC of triangle ABC such that CO / CA = 2 / 3. When triangle ABC is rotated by a certain angle around point O, vertex B moves to vertex C, and vertex A moves to point D, which lies on side AB. Prove that the ratio of the areas of triangles BOD and ABC is 1 / 6. 
-/
theorem area_ratio_of_rotated_triangles
  (A B C D O : Point)
  (h1 : O ∈ Segment A C)
  (h2 : ratio (length (Segment C O)) (length (Segment C A)) = 2 / 3)
  (h3 : rotates_by_angle_around A B C D O)
  (h4 : D ∈ Segment A B)
  (h5 : midpoint D A B) :
  area (Triangle B O D) / area (Triangle A B C) = 1 / 6 :=
sorry

end area_ratio_of_rotated_triangles_l598_598061


namespace greatest_value_x_is_correct_l598_598666

noncomputable def greatest_value_x : ℝ :=
-8 + Real.sqrt 6

theorem greatest_value_x_is_correct :
  ∀ x : ℝ, (x ≠ 9) → ((x^2 - x - 90) / (x - 9) = 2 / (x + 6)) → x ≤ greatest_value_x :=
by
  sorry

end greatest_value_x_is_correct_l598_598666


namespace final_price_is_correct_l598_598596

-- Define the original price
variable (a : ℝ)

-- Define the conditions
def first_reduction (a : ℝ) : ℝ := a * 0.9
def second_reduction (a : ℝ) : ℝ := (first_reduction a) * 0.9
def final_increase (a : ℝ) : ℝ := (second_reduction a) * 1.2

-- The theorem to prove that the final price is 0.972a
theorem final_price_is_correct : final_increase a = 0.972 * a :=
by
  unfold first_reduction
  unfold second_reduction
  unfold final_increase
  sorry

end final_price_is_correct_l598_598596


namespace abs_neg_three_halves_l598_598485

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end abs_neg_three_halves_l598_598485


namespace tommy_profit_l598_598953

def crate_weight : ℕ := 20
def number_of_crates : ℕ := 3
def cost_of_crates : ℕ := 330
def price_per_kg : ℕ := 6
def rotten_tomatoes : ℕ := 3

theorem tommy_profit :
  let total_weight := crate_weight * number_of_crates in
  let sellable_weight := total_weight - rotten_tomatoes in
  let earnings := sellable_weight * price_per_kg in
  let profit := earnings - cost_of_crates in
  profit = 12 :=
by
  -- proof goes here
  sorry

end tommy_profit_l598_598953


namespace eighth_term_of_arithmetic_sequence_l598_598720

theorem eighth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℤ),
  (a 1 = 11) →
  (a 2 = 8) →
  (a 3 = 5) →
  (∃ (d : ℤ), ∀ n, a (n + 1) = a n + d) →
  a 8 = -10 :=
by
  intros a h1 h2 h3 arith
  sorry

end eighth_term_of_arithmetic_sequence_l598_598720


namespace find_n_eq_6_l598_598262

theorem find_n_eq_6 (n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) : 2^n + n^2 + 25 = p^3 → n = 6 := by
  sorry

end find_n_eq_6_l598_598262


namespace suitcase_lock_combinations_l598_598977

def is_valid_combination (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧ 
  d1 + d2 + d3 + d4 ≤ 10 ∧
  List.all [d1, d2, d3, d4] (λ d, d ∈ Finset.range 10)

def count_valid_combinations : ℕ :=
  Finset.card (Finset.filter (λ d, is_valid_combination d.1.1 d.1.2.1 d.1.2.2.1 d.1.2.2.2) 
  (Finset.product (Finset.product (Finset.product Finset.range 10 Finset.range 10) Finset.range 10) Finset.range 10))

theorem suitcase_lock_combinations : count_valid_combinations = C := 
  sorry

end suitcase_lock_combinations_l598_598977


namespace find_x_l598_598121

theorem find_x
  (x : ℤ)
  (h1 : 71 * x % 9 = 8) :
  x = 1 :=
sorry

end find_x_l598_598121


namespace number_of_hens_is_50_l598_598769

def number_goats : ℕ := 45
def number_camels : ℕ := 8
def number_keepers : ℕ := 15
def extra_feet : ℕ := 224

def total_heads (number_hens number_goats number_camels number_keepers : ℕ) : ℕ :=
  number_hens + number_goats + number_camels + number_keepers

def total_feet (number_hens number_goats number_camels number_keepers : ℕ) : ℕ :=
  2 * number_hens + 4 * number_goats + 4 * number_camels + 2 * number_keepers

theorem number_of_hens_is_50 (H : ℕ) :
  total_feet H number_goats number_camels number_keepers = (total_heads H number_goats number_camels number_keepers) + extra_feet → H = 50 :=
sorry

end number_of_hens_is_50_l598_598769


namespace sum_of_other_endpoint_l598_598052

theorem sum_of_other_endpoint (x y : ℝ) (h1 : (6 + x) / 2 = 3) (h2 : (-2 + y) / 2 = 5) : x + y = 12 := 
by {
  sorry
}

end sum_of_other_endpoint_l598_598052


namespace fraction_of_second_year_given_not_third_year_l598_598380

theorem fraction_of_second_year_given_not_third_year (total_students : ℕ) 
  (third_year_students : ℕ) (second_year_students : ℕ) :
  third_year_students = total_students * 30 / 100 →
  second_year_students = total_students * 10 / 100 →
  ↑second_year_students / (total_students - third_year_students) = (1 : ℚ) / 7 :=
by
  -- Proof omitted
  sorry

end fraction_of_second_year_given_not_third_year_l598_598380


namespace Vikas_submitted_6_questions_l598_598854

theorem Vikas_submitted_6_questions (R V A : ℕ) (h1 : 7 * V = 3 * R) (h2 : 2 * V = 3 * A) (h3 : R + V + A = 24) : V = 6 :=
by
  sorry

end Vikas_submitted_6_questions_l598_598854


namespace convex_quadrilateral_count_l598_598109

theorem convex_quadrilateral_count (P : Finset (EuclideanSpace R ℝ)) (h : P.card = 12)
  (circ : ∃ (c : EuclideanSpace R ℝ) (r : ℝ), ∀ p ∈ P, ∃ θ : ℝ, p = c + r • (cos θ, sin θ))
  (no_three_collinear: ∀ (A B C : EuclideanSpace R ℝ), A ∈ P → B ∈ P → C ∈ P → ¬ Collinear R {A, B, C}) :
  ∃ Q : ℕ, Q = 495 := by
  have P_exists : P.card = 12 := h
  sorry

end convex_quadrilateral_count_l598_598109


namespace problem_statement_l598_598440

noncomputable def f (x : ℝ) : ℝ := x - Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log x / x

theorem problem_statement (x : ℝ) (h : 0 < x ∧ x ≤ Real.exp 1) : 
  f x > g x + 1/2 :=
sorry

end problem_statement_l598_598440


namespace expand_x_plus_3y_squared_expand_2x_plus_3y_squared_expand_m3_plus_n5_squared_expand_5x_minus_3y_squared_expand_3m5_minus_4n2_squared_l598_598149

-- Proof for (x + 3y)^2 = x^2 + 6xy + 9y^2
theorem expand_x_plus_3y_squared (x y : ℝ) : 
  (x + 3 * y) ^ 2 = x ^ 2 + 6 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (2x + 3y)^2 = 4x^2 + 12xy + 9y^2
theorem expand_2x_plus_3y_squared (x y : ℝ) : 
  (2 * x + 3 * y) ^ 2 = 4 * x ^ 2 + 12 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (m^3 + n^5)^2 = m^6 + 2m^3n^5 + n^10
theorem expand_m3_plus_n5_squared (m n : ℝ) : 
  (m ^ 3 + n ^ 5) ^ 2 = m ^ 6 + 2 * m ^ 3 * n ^ 5 + n ^ 10 := 
  sorry

-- Proof for (5x - 3y)^2 = 25x^2 - 30xy + 9y^2
theorem expand_5x_minus_3y_squared (x y : ℝ) : 
  (5 * x - 3 * y) ^ 2 = 25 * x ^ 2 - 30 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (3m^5 - 4n^2)^2 = 9m^10 - 24m^5n^2 + 16n^4
theorem expand_3m5_minus_4n2_squared (m n : ℝ) : 
  (3 * m ^ 5 - 4 * n ^ 2) ^ 2 = 9 * m ^ 10 - 24 * m ^ 5 * n ^ 2 + 16 * n ^ 4 := 
  sorry

end expand_x_plus_3y_squared_expand_2x_plus_3y_squared_expand_m3_plus_n5_squared_expand_5x_minus_3y_squared_expand_3m5_minus_4n2_squared_l598_598149


namespace trig_identity_l598_598286

variable (m : ℝ) (α : ℝ)

def point_M_on_terminal_side_of_angle (m : ℝ) (α : ℝ) : Prop :=
  m < 0 ∧ (∃ t : ℝ, t = α ∧ t = real.arctan (-2))

theorem trig_identity (h1 : point_M_on_terminal_side_of_angle m α) :
  (1 / (2 * real.sin α * real.cos α + real.cos α ^ 2) = -5 / 3) :=
sorry

end trig_identity_l598_598286


namespace probability_of_point_in_heart_shape_l598_598995

-- Define the regions as per the given conditions
def area_of_heart_shape_region (A : Set (ℝ × ℝ)) : ℝ := 3 * Real.pi

def area_of_rectangle_Omega (Ω : Set (ℝ × ℝ)) : ℝ := 4 * (Real.pi + 1)

def probability (heart_area : ℝ) (rectangle_area : ℝ) : ℝ := heart_area / rectangle_area

-- Main theorem statement
theorem probability_of_point_in_heart_shape :
  probability (area_of_heart_shape_region {(x, y) | (x <= 0 ∧ (x^2 + y^2 = 2 * |y|)) ∨ (0 <= x ∧ x <= Real.pi ∧ |y| = Real.cos x + 1)})
               (area_of_rectangle_Omega {(x, y) | -1 <= x ∧ x <= Real.pi ∧ -2 <= y ∧ y <= 2}) = 
  3 * Real.pi / (4 * (Real.pi + 1)) := by
sorry

end probability_of_point_in_heart_shape_l598_598995


namespace car_travel_time_l598_598595

theorem car_travel_time:
  ∀ (distance : ℝ) (speed : ℝ), distance = 715 → speed = 65 → (distance / speed) = 11 :=
by
  intros distance speed h_distance h_speed
  rw [h_distance, h_speed]
  norm_num
  sorry

end car_travel_time_l598_598595


namespace brianna_remaining_money_l598_598231

theorem brianna_remaining_money (m b : ℝ) 
  (h1 : (1 / 3) * m = (1 / 2) * b) 
  (h2 : b = (2 / 3) * m) :
  (m - ((5 / 6) * m)) = (1 / 6) * m :=
by { sorry }

end brianna_remaining_money_l598_598231


namespace length_of_bridge_l598_598979

/-- A train that is 357 meters long is running at a speed of 42 km/hour. 
    It takes 42.34285714285714 seconds to pass a bridge. 
    Prove that the length of the bridge is 136.7142857142857 meters. -/
theorem length_of_bridge : 
  let train_length := 357 -- meters
  let speed_kmh := 42 -- km/hour
  let passing_time := 42.34285714285714 -- seconds
  let speed_mps := 42 * (1000 / 3600) -- meters/second
  let total_distance := speed_mps * passing_time -- meters
  let bridge_length := total_distance - train_length -- meters
  bridge_length = 136.7142857142857 :=
by
  sorry

end length_of_bridge_l598_598979


namespace triangle_angles_l598_598042

-- Definitions of points and tangency
variables {A B C D E : Type} [MetricSpace A] [Line A] [Triangle A B C]

-- Conditions provided in the problem
def condition_1 (B C : Type) (A B C D : A) : Prop :=
  extension_point B C D ∧ tangent_line A D (circumcircle A B C)

def condition_2 (A C D : A) (E : Type) : Prop :=
  intersects (line A C) (circumcircle (triangle A B D)) E ∧ ratio A C E = 1/2

def condition_3 (A D E : A) : Prop :=
  tangent_to_circumcircle angle_bisector (angle ADE)

-- Proof statement
theorem triangle_angles :
  ∀ (A B C D E : Type) [Condition_1 A B C D] [Condition_2 A C D E] [Condition_3 A D E] ,
  triangle_angle A B C = (30, 60, 90) :=
by sorry

end triangle_angles_l598_598042


namespace DaisyDogToys_l598_598643

-- Defining the conditions as variables and premises
variables (MondayToys TuesdayLeftToys BoughtTuesdayToys WednesdayBoughtToys : ℕ)
variables (LostToysMondayTuesday : ℕ)

-- The given conditions from the problem
def condition1 := MondayToys = 5
def condition2 := TuesdayLeftToys = 3
def condition3 := BoughtTuesdayToys = 3
def condition4 := WednesdayBoughtToys = 5
def condition5 := LostToysMondayTuesday = MondayToys - TuesdayLeftToys

-- The target problem statement
theorem DaisyDogToys :
  MondayToys = 5 →
  TuesdayLeftToys = 3 →
  BoughtTuesdayToys = 3 →
  WednesdayBoughtToys = 5 →
  LostToysMondayTuesday = MondayToys - TuesdayLeftToys →
  MondayToys + LostToysMondayTuesday + BoughtTuesdayToys + WednesdayBoughtToys = 15 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact add_assoc (5 + 2) 3 5 ▸ rfl

end DaisyDogToys_l598_598643


namespace probability_even_product_l598_598899

-- Define the conditions for the spinners
def SpinnerA : Set ℕ := {1, 2, 3, 4, 5}
def SpinnerB : Set ℕ := {1, 2, 3, 4}

-- Equally likely probabilities
def probability_spinner_a (n : ℕ) : ℚ := if n ∈ SpinnerA then 1/5 else 0
def probability_spinner_b (n : ℕ) : ℚ := if n ∈ SpinnerB then 1/4 else 0

-- Define the statement to prove
theorem probability_even_product : (1 - (3/10)) = (7/10) := by
  sorry

end probability_even_product_l598_598899


namespace max_value_l598_598311

-- Define the vector types
structure Vector2 where
  x : ℝ
  y : ℝ

-- Define the properties given in the problem
def a_is_unit_vector (a : Vector2) : Prop :=
  a.x^2 + a.y^2 = 1

def a_plus_b (a b : Vector2) : Prop :=
  a.x + b.x = 3 ∧ a.y + b.y = 4

-- Define dot product for the vectors
def dot_product (a b : Vector2) : ℝ :=
  a.x * b.x + a.y * b.y

-- The theorem statement
theorem max_value (a b : Vector2) (h1 : a_is_unit_vector a) (h2 : a_plus_b a b) :
  ∃ m, m = 5 ∧ ∀ c : ℝ, |1 + dot_product a b| ≤ m :=
  sorry

end max_value_l598_598311


namespace sum_of_other_endpoint_coordinates_l598_598048

theorem sum_of_other_endpoint_coordinates 
  (A B O : ℝ × ℝ)
  (hA : A = (6, -2)) 
  (hO : O = (3, 5)) 
  (midpoint_formula : (A.1 + B.1) / 2 = O.1 ∧ (A.2 + B.2) / 2 = O.2):
  (B.1 + B.2) = 12 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l598_598048


namespace max_sum_of_factors_l598_598890

theorem max_sum_of_factors (p q : ℕ) (hpq : p * q = 100) : p + q ≤ 101 :=
sorry

end max_sum_of_factors_l598_598890


namespace avg_speed_BC_60_mph_l598_598148

theorem avg_speed_BC_60_mph 
  (d_AB : ℕ) (d_BC : ℕ) (avg_speed_total : ℚ) (time_ratio : ℚ) (t_AB : ℕ) :
  d_AB = 120 ∧ d_BC = 60 ∧ avg_speed_total = 45 ∧ time_ratio = 3 ∧
  t_AB = 3 → (d_BC / (t_AB / time_ratio) = 60) :=
by
  sorry

end avg_speed_BC_60_mph_l598_598148


namespace number_of_students_registered_l598_598875

theorem number_of_students_registered 
  (P : ℕ) 
  (H1 : 1.80 * P + 30 = 156) : 
  156 = 156 :=
by
  sorry

end number_of_students_registered_l598_598875


namespace TrianglePAreaIs22PercentLess_TriangleRAreaIs31_25PercentLess_l598_598550

variables {B_Q H_Q : ℝ}

def area (base height : ℝ) : ℝ := (base * height) / 2

def B_P := 1.30 * B_Q
def H_P := 0.60 * H_Q
def B_R := 0.55 * B_Q
def H_R := 1.25 * H_Q

noncomputable def A_Q := area B_Q H_Q
noncomputable def A_P := area B_P H_P
noncomputable def A_R := area B_R H_R

theorem TrianglePAreaIs22PercentLess :
  A_P = 0.78 * A_Q :=
begin
  sorry
end

theorem TriangleRAreaIs31_25PercentLess :
  A_R = 0.6875 * A_Q :=
begin
  sorry
end

end TrianglePAreaIs22PercentLess_TriangleRAreaIs31_25PercentLess_l598_598550


namespace basketball_player_score_prob_l598_598168

variable (p : ℚ) (n : ℕ)

def independent_shots (prob : ℚ) (shots : ℕ) (score_prob : ℚ) : Prop :=
  ∀ (p : ℚ), score_prob = prob^(shots)

theorem basketball_player_score_prob
  (prob_shooting : ℚ) (score_target : ℚ) (shots : ℕ)
  (h_prob : prob_shooting = 0.7)
  (h_shots : shots = 3)
  (h_target : score_target = 0.343) :
  independent_shots prob_shooting shots score_target :=
begin
  assume prob,
  have h : score_target = prob_shooting ^ shots,
  {
    rw [h_prob, h_shots],
    norm_num,
  },
  exact h_target.symm ▸ h,
end

end basketball_player_score_prob_l598_598168


namespace intersection_count_l598_598793

noncomputable def f (x : ℝ) : ℝ := 3 * log x
noncomputable def g (x : ℝ) : ℝ := log (3 * (x - 1))

theorem intersection_count :
  ∃! x, f x = g x :=
sorry

end intersection_count_l598_598793


namespace soccer_points_l598_598874

def total_points (wins draws losses : ℕ) (points_per_win points_per_draw points_per_loss : ℕ) : ℕ :=
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss

theorem soccer_points : total_points 14 4 2 3 1 0 = 46 :=
by
  sorry

end soccer_points_l598_598874


namespace cannot_remove_all_pieces_l598_598606

-- Define the grid and pieces
def grid_size := (22, 23)
def initial_pieces := 22 * 23
def black_cells := 11 * 23
def white_cells := 11 * 23
def initial_black_pieces := black_cells
def initial_white_pieces := white_cells

-- Moves and removals preserving parity of piece counts
def is_adjacent (x1 y1 x2 y2 : Nat) :=
  (x1 = x2 ∧ abs (y1 - y2) = 1) ∨
  (y1 = y2 ∧ abs (x1 - x2) = 1)

theorem cannot_remove_all_pieces :
  ∀ (move : (Nat × Nat) → (Nat × Nat) → (Nat × Nat) → (Nat × Nat)),
  ∀ (remove : (Nat × Nat) → Nat → Nat),
  initial_black_pieces % 2 = 1 →
  initial_white_pieces % 2 = 1 →
  ∀ (steps : List (((Nat × Nat) → (Nat × Nat)) × (Nat × Nat) → Nat)),
  ¬(initial_black_pieces = 0 ∧ initial_white_pieces = 0) :=
by
  sorry

end cannot_remove_all_pieces_l598_598606


namespace tangent_line_eq_monotonic_intervals_extreme_values_inequality_holds_l598_598727

-- Definition of the function f
def f (x : ℝ) (k : ℝ) := x^3 + k * Real.log x

-- First derivative of f
def f_prime (x : ℝ) (k : ℝ) := 3 * x^2 + k / x

-- Definition of the function g
def g (x : ℝ) := f x 6 - f_prime x 6 + 9 / x

-- Statement of the proof problem
theorem tangent_line_eq (k : ℝ) : k = 6 → (∀ x : ℝ, x = 1 → ∃ m b : ℝ, m = 9 ∧ b = -8 ∧ (∀ y : ℝ, y = f x k → y = m * x + b)) :=
by
  sorry

theorem monotonic_intervals_extreme_values : (∀ x : ℝ, ((0 < x ∧ x < 1) → (g x) < 1) ∧ ((1 < x) → (g x) > 1) ∧ (g 1 = 1)) :=
by
  sorry

theorem inequality_holds (k : ℝ) (x₁ x₂ : ℝ) : (k ≥ -3) → (1 ≤ x₁ ∧ 1 ≤ x₂) → (x₁ > x₂) → ((f_prime x₁ k + f_prime x₂ k) / 2 > (f x₁ k - f x₂ k) / (x₁ - x₂)) :=
by
  sorry

end tangent_line_eq_monotonic_intervals_extreme_values_inequality_holds_l598_598727


namespace dessert_menu_count_l598_598175

def options := {'cake, 'pie, 'ice_cream, 'pudding, 'cookies}

def valid_dessert_menu (menu : List String) : Bool :=
  (menu.length = 10) ∧
  (menu.nth 2 = some 'cake) ∧
  (menu.nth 3 = some 'ice_cream) ∧
  ∀ i, i < 9 → menu.nth i ≠ menu.nth (i + 1)

theorem dessert_menu_count :
  {menu : List String // valid_dessert_menu menu} = 5 * 4^8 := 
sorry

end dessert_menu_count_l598_598175


namespace conjugate_of_z_l598_598365

def z : ℂ := (2 + complex.i) / complex.i
def z_conjugate : ℂ := complex.conj z

theorem conjugate_of_z :
  z_conjugate = 1 + 2 * complex.i :=
sorry

end conjugate_of_z_l598_598365


namespace length_of_FQ_l598_598872

theorem length_of_FQ (DE DF : ℝ) (h1 : DE = 7) (h2 : DF = Real.sqrt 85) :
  ∃ (EF FQ : ℝ), EF = FQ ∧ FQ = 6 :=
by
  have EF : ℝ := Real.sqrt (DF^2 - DE^2)
  use EF, EF
  split
  · reflexivity
  · rw [h1, h2]
    calc 
      EF = Real.sqrt (85 - 49) : by rw [Real.sqrt_eq_rpow, sqr_sqrt (by norm_num)]
      ... = 6 : by norm_num
    rfl

end length_of_FQ_l598_598872


namespace equilateral_triangle_l598_598356

theorem equilateral_triangle (a b c : ℝ) (A B C : ℝ) :
  (a + b + c) * (b + c - a) = 3 * b * c ∧ sin A = 2 * sin B * cos C → 
  ∃ (t : Type), t = "equilateral triangle" := 
by
  sorry

end equilateral_triangle_l598_598356


namespace M_subset_N_l598_598731

def M : Set ℚ := { x | ∃ k : ℤ, x = k / 2 + 1 / 4 }
def N : Set ℚ := { x | ∃ k : ℤ, x = k / 4 + 1 / 2 }

theorem M_subset_N : M ⊆ N :=
sorry

end M_subset_N_l598_598731


namespace minimum_daily_revenue_l598_598950

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := 5 + 5 / x
def g (x : ℝ) : ℝ := 20 * x + 500
def y (x : ℝ) : ℝ := 100 * (100 * x + 2500 / x + 2600)

-- Proof that y(x) is minimized on the 5th day and the minimum value is 360000 yuan
theorem minimum_daily_revenue :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 15 → y(x) ≥ 360000) ∧ y 5 = 360000 := 
by
  sorry  -- proof not given

end minimum_daily_revenue_l598_598950


namespace sum_last_three_coeff_l598_598569

theorem sum_last_three_coeff (a : ℕ) : 
  (last_three_coeff_sum : (1 + 1/a)^8) = 37 := 
by
  sorry

end sum_last_three_coeff_l598_598569


namespace stable_number_divisible_by_11_l598_598217

/-- Definition of a stable number as a three-digit number (cen, ten, uni) where
    each digit is non-zero, and the sum of any two digits is greater than the remaining digit.
-/
def is_stable_number (cen ten uni : ℕ) : Prop :=
cen ≠ 0 ∧ ten ≠ 0 ∧ uni ≠ 0 ∧
(cen + ten > uni) ∧ (cen + uni > ten) ∧ (ten + uni > cen)

/-- Function F defined for a stable number (cen ten uni). -/
def F (cen ten uni : ℕ) : ℕ := 10 * ten + cen + uni

/-- Function Q defined for a stable number (cen ten uni). -/
def Q (cen ten uni : ℕ) : ℕ := 10 * cen + ten + uni

/-- Statement to prove: Given a stable number s = 100a + 101b + 30 where 1 ≤ a ≤ 5 and 1 ≤ b ≤ 4,
    the expression 5 * F(s) + 2 * Q(s) is divisible by 11.
-/
theorem stable_number_divisible_by_11 (a b cen ten uni : ℕ)
  (h_a : 1 ≤ a ∧ a ≤ 5)
  (h_b : 1 ≤ b ∧ b ≤ 4)
  (h_s : 100 * a + 101 * b + 30 = 100 * cen + 10 * ten + uni)
  (h_stable : is_stable_number cen ten uni) :
  (5 * F cen ten uni + 2 * Q cen ten uni) % 11 = 0 :=
sorry

end stable_number_divisible_by_11_l598_598217


namespace double_sum_geometric_series_l598_598638

theorem double_sum_geometric_series :
  (∑ j : ℕ, ∑ k : ℕ, 2^(-(k + j + (k + j)^3))) = 4 / 3 :=
by
  sorry

end double_sum_geometric_series_l598_598638


namespace dice_probability_l598_598281

theorem dice_probability : 
  let outcomes := {1, 2, 3, 4, 5, 6} in
  let a_possible := outcomes ∧ 
  let b_possible := outcomes ∧ 
  let c_possible := outcomes ∧ 
  let d_possible := outcomes in
  ∀ (a b c d ∈ outcomes), 
  (a ≠ 6 ∧ b ≠ 6 ∧ c ≠ 6 ∧ d ≠ 6) → 
  (a-6) * (b-6) * (c-6) * (d-6) ≠ 0 → 
  (5^4 / 6^4 = 625 / 1296) :=
by sorry

end dice_probability_l598_598281


namespace abs_neg_frac_l598_598479

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end abs_neg_frac_l598_598479


namespace cone_base_circumference_l598_598599

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (C : ℝ) : 
  r = 5 → θ = 300 → C = (θ / 360) * (2 * Real.pi * r) → C = (25 / 3) * Real.pi :=
by
  sorry

end cone_base_circumference_l598_598599


namespace polyhedron_value_l598_598174

theorem polyhedron_value (T H V E : ℕ) (h t : ℕ) 
  (F : ℕ) (h_eq : h = 10) (t_eq : t = 10)
  (F_eq : F = 20)
  (edges_eq : E = (3 * t + 6 * h) / 2)
  (vertices_eq : V = E - F + 2)
  (T_value : T = 2) (H_value : H = 2) :
  100 * H + 10 * T + V = 227 := by
  sorry

end polyhedron_value_l598_598174


namespace area_triangle_ABC_l598_598713

theorem area_triangle_ABC
  {A B C : ℝ} 
  {a b c : ℝ} 
  (h1 : ∃ (A1 B1 C1 : ℝ), sin A = cos A1 ∧ sin B = cos B1 ∧ sin C = cos C1)
  (h2 : a = 2 * Real.sqrt 5)
  (h3 : b = 2 * Real.sqrt 2)
  (h4 : A > π / 2)
  (h5 : cos_rule : ∀ (c : ℝ), c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * cos A) :
  (1 / 2) * b * c * sin A = 2 :=
by 
  sorry

end area_triangle_ABC_l598_598713


namespace PM_eq_PN_l598_598911

noncomputable theory

open Real EuclideanGeometry

structure TwoSpheresSetup (O₁ O₂ : Point) (P A C B D M N : Point) :=
  (externally_tangent_at_P : ∃ r₁ r₂ : ℝ, 0 < r₁ ∧ 0 < r₂ ∧ Sphere O₁ r₁ ∩ Sphere O₂ r₂ = {P})
  (A_on_sphere₁ : ∃ r₁ : ℝ, A ∈ Sphere O₁ r₁)
  (C_on_sphere₁ : ∃ r₁ : ℝ, C ∈ Sphere O₁ r₁)
  (B_on_sphere₂ : ∃ r₂ : ℝ, B ∈ Sphere O₂ r₂)
  (D_on_sphere₂ : ∃ r₂ : ℝ, D ∈ Sphere O₂ r₂)
  (M_projection : ∃ K : Point, midpoint A C = K ∧ M ∈ line_through O₁ O₂)
  (N_projection : ∃ L : Point, midpoint B D = L ∧ N ∈ line_through O₁ O₂)

theorem PM_eq_PN {O₁ O₂ P A C B D M N : Point} (setup : TwoSpheresSetup O₁ O₂ P A C B D M N) :
  dist P M = dist P N :=
sorry

end PM_eq_PN_l598_598911


namespace tangent_line_at_point_monotonic_and_extreme_values_inequality_holds_l598_598724

noncomputable theory

-- Define the function f based on the parameter k
def f (k x : ℝ) : ℝ := x ^ 3 + k * log x
-- Define the derivative f'
def f' (k x : ℝ) : ℝ := 3 * x ^ 2 + k / x

-- Problem (I) (i): Tangent line for k = 6 at the point (1, f(6, 1))
theorem tangent_line_at_point : (∀ x, f 6 x = x^3 + 6 * log x) → 
  (∀ x, f' 6 x = 3 * x ^ 2 + 6 / x) →
  9 * (1:ℝ) - (f 6 1) - 8 = 0 :=
  sorry

-- Problem (I) (ii): Monotonic intervals and extreme values of g when k = 6
def g (x : ℝ) : ℝ := f 6 x - f' 6 x + 9 / x

theorem monotonic_and_extreme_values : 
  ∀ x, g x = x ^ 3 - 3 * x ^ 2 + 6 * log x + 3 / x →
  (∀ x, 0 < x ∧ x < 1 → deriv g x < 0) ∧ 
  (∀ x, x > 1 → deriv g x > 0) ∧ 
  g 1 = 1 :=
  sorry

-- Problem (II): Inequality for k ≥ -3 and x1, x2 ∈ [1, +∞) with x1 > x2
theorem inequality_holds (k : ℝ) (x1 x2 : ℝ) (h1 : k ≥ -3) (h2 : 1 ≤ x1) (h3 : 1 ≤ x2) (h4 : x1 > x2) : 
  (f' k x1 + f' k x2) / 2 > (f k x1 - f k x2) / (x1 - x2) :=
  sorry

end tangent_line_at_point_monotonic_and_extreme_values_inequality_holds_l598_598724


namespace distance_BM_l598_598791

/-- Define the points A, B, and M -/
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := -1, y := 2, z := -3 }
def B : Point := { x := -1, y := 0, z := 2 }
def M : Point := { x := 1, y := 2, z := -3 }

/-- Definition of distance between two points -/
def distance (P Q : Point) : ℝ :=
  real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2 + (Q.z - P.z)^2)

/-- Prove that the distance between B and M is 3 -/
theorem distance_BM : distance B M = 3 := by
  sorry

end distance_BM_l598_598791


namespace find_ages_l598_598373

-- Define the conditions
axiom cond1 (a b : ℚ) : a + 40 = 3 * (b - 50)
axiom cond2 (a c : ℚ) : c = 1 / 2 * a
axiom cond3 (a b : ℚ) : a = b + 5

-- Define the theorem to prove
theorem find_ages (a b c : ℚ) (h1 : cond1 a b) (h2 : cond2 a c) (h3 : cond3 a b) :
  a = 102.5 ∧ b = 97.5 ∧ c = 51.25 :=
sorry

end find_ages_l598_598373


namespace max_cubes_fit_l598_598920

-- Define the conditions
def box_volume (length : ℕ) (width : ℕ) (height : ℕ) : ℕ := length * width * height
def cube_volume : ℕ := 27
def total_cubes (V_box : ℕ) (V_cube : ℕ) : ℕ := V_box / V_cube

-- Statement of the problem
theorem max_cubes_fit (length width height : ℕ) (V_box : ℕ) (V_cube q : ℕ) :
  length = 8 → width = 9 → height = 12 → V_box = box_volume length width height →
  V_cube = cube_volume → q = total_cubes V_box V_cube → q = 32 :=
by sorry

end max_cubes_fit_l598_598920


namespace James_age_is_11_l598_598405

-- Define the ages of Julio and James.
def Julio_age := 36

-- The age condition in 14 years.
def Julio_age_in_14_years := Julio_age + 14

-- James' age in 14 years and the relation as per the condition.
def James_age_in_14_years (J : ℕ) := J + 14

-- The main proof statement.
theorem James_age_is_11 (J : ℕ) 
  (h1 : Julio_age_in_14_years = 2 * James_age_in_14_years J) : J = 11 :=
by
  sorry

end James_age_is_11_l598_598405


namespace cupcakes_per_child_l598_598896

theorem cupcakes_per_child (total_cupcakes children : ℕ) (h1 : total_cupcakes = 96) (h2 : children = 8) : total_cupcakes / children = 12 :=
by
  sorry

end cupcakes_per_child_l598_598896


namespace inequality_holds_l598_598711

variable {f : ℝ → ℝ}

-- Conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_monotonic_on_nonneg_interval (f : ℝ → ℝ) : Prop := ∀ x y, (0 ≤ x ∧ x < y ∧ y < 8) → f y ≤ f x

axiom condition1 : is_even f
axiom condition2 : is_monotonic_on_nonneg_interval f
axiom condition3 : f (-3) < f 2

-- The statement to be proven
theorem inequality_holds : f 5 < f (-3) ∧ f (-3) < f (-1) :=
by
  sorry

end inequality_holds_l598_598711


namespace problem_statement_l598_598590

theorem problem_statement : 
  10 - 1.05 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.93 :=
by sorry

end problem_statement_l598_598590


namespace projection_correct_l598_598690

variables (a b : Vector ℝ) (θ : ℝ)
def length_a_eq_one : ‖a‖ = 1 := sorry
def length_b_eq_two : ‖b‖ = 2 := sorry
def angle_theta_eq_sixty : θ = real.pi / 3 := sorry

noncomputable def projection_of_b_onto_a : ℝ :=
‖b‖ * real.cos θ

theorem projection_correct :
  length_a_eq_one a →
  length_b_eq_two b →
  angle_theta_eq_sixty θ →
  projection_of_b_onto_a a b θ = 1 := by
  intros
  sorry

end projection_correct_l598_598690


namespace prism_correct_statement_l598_598497

def is_prism (P : Type) := sorry -- Define what it means for P to be a prism

-- Defining what it means for a face to be parallel and for edges to be parallel
def parallel_faces (F1 F2 : Type) := sorry
def parallel_edges (E1 E2 : Type) := sorry

-- Axiom: The two base faces in a prism are parallel.
axiom base_faces_parallel {P : Type} [is_prism P] (B1 B2 : Type) : parallel_faces B1 B2

-- Axiom: Each lateral edge in a prism is parallel to each other.
axiom lateral_edges_parallel {P : Type} [is_prism P] (E1 E2 : Type) : parallel_edges E1 E2

-- Defining the statement D
def statement_D (P : Type) [is_prism P] (B1 B2 : Type) (E1 E2 : Type) : Prop :=
  parallel_faces B1 B2 ∧ parallel_edges E1 E2

-- The theorem stating that statement D is true for prisms
theorem prism_correct_statement {P : Type} [is_prism P] (B1 B2 : Type) (E1 E2 : Type) : 
  statement_D P B1 B2 E1 E2 :=
by
  constructor
  { apply base_faces_parallel }
  { apply lateral_edges_parallel }

end prism_correct_statement_l598_598497


namespace total_houses_is_160_l598_598376

namespace MariamNeighborhood

-- Define the given conditions as variables in Lean.
def houses_on_one_side : ℕ := 40
def multiplier : ℕ := 3

-- Define the number of houses on the other side of the road.
def houses_on_other_side : ℕ := multiplier * houses_on_one_side

-- Define the total number of houses in Mariam's neighborhood.
def total_houses : ℕ := houses_on_one_side + houses_on_other_side

-- Prove that the total number of houses is 160.
theorem total_houses_is_160 : total_houses = 160 :=
by
  -- Placeholder for proof
  sorry

end MariamNeighborhood

end total_houses_is_160_l598_598376


namespace family_reunion_attendance_l598_598948

theorem family_reunion_attendance 
  (box_contains : ℕ := 10) 
  (box_costs : ℕ := 2) 
  (consume_per_person : ℕ := 2) 
  (family_members : ℕ := 6) 
  (pay_per_member : ℕ := 4) 
  (total_payment : ℕ := family_members * pay_per_member) : 
  ∃ (P : ℕ), P = 60 :=
by 
  let boxes_needed := (2 * 60) / box_contains
  let total_cost := boxes_needed * box_costs
  have h1 : total_cost = total_payment,
  sorry
  existsi 60
  exact rfl

end family_reunion_attendance_l598_598948


namespace distinct_pairs_l598_598637

noncomputable def first_three_digits (n : ℕ) : ℕ :=
  n / 10^(nat.log10 n - 2)

theorem distinct_pairs :
  (∃ (n : ℕ), 4495 = (set.univ.filter (λ x : ℕ, x > 10^10)).count (λ x, 
    let d1 := first_three_digits x,
        d2 := first_three_digits (x^4)
    in (d1, d2))) :=
sorry

end distinct_pairs_l598_598637


namespace treadmill_time_saved_correct_l598_598340

noncomputable def treadmill_time_saved : ℕ :=
  let t_monday := 3 / 6
  let t_tuesday := 2 / 5
  let t_wednesday := 4 / 4
  let t_thursday := 1 / 2
  let t_total := t_monday + t_tuesday + t_wednesday + t_thursday
  let t_hypothetical := 10 / 5
  let time_saved := (t_total - t_hypothetical) * 60
  time_saved.toNat

theorem treadmill_time_saved_correct : treadmill_time_saved = 30 := sorry

end treadmill_time_saved_correct_l598_598340


namespace hyperbola_eq_l598_598691

theorem hyperbola_eq {a b : ℝ} (h : a > b ∧ b > 0)
  (asymptote_slope : ∃ l : ℝ , ∀ x y, y = -2*x - 10 → y = l * x + c)
  (focus_on_l : ∃ f : Point2D, ∀ x f_x, f_y = 0)
  (hy1 : ∃ a, a = 5 )
  (hy2 : ∃ b, b = 4) :

  (\frac{x^2}{5} - \frac{y^2}{20} = 1):

Proof
sorry

end hyperbola_eq_l598_598691


namespace count_winning_positions_l598_598552

-- Define the game conditions
def is_losing_position (N : ℕ) : Prop :=
  N % 3 = 0

def has_winning_strategy (N : ℕ) : Prop :=
  ¬ is_losing_position N

-- Statement that counts the number of N with a winning strategy between 1 and 2019
theorem count_winning_positions : 
  (Finset.filter has_winning_strategy (Finset.range 2020)).card = 1346 := 
sorry

end count_winning_positions_l598_598552


namespace sum_of_even_factors_of_630_l598_598124

noncomputable def sum_of_positive_even_factors (n : Nat) : Nat :=
  ∑ i in (Finset.filter (λ d, d % 2 = 0) (Finset.divisors n)), i

theorem sum_of_even_factors_of_630 :
  (sum_of_positive_even_factors 630) = 1248 := by
  sorry

end sum_of_even_factors_of_630_l598_598124


namespace no_two_tetrahedra_of_volume_half_in_sphere_radius_one_l598_598794

theorem no_two_tetrahedra_of_volume_half_in_sphere_radius_one :
  ¬ ∃ (T₁ T₂ : Tetrahedron), T₁.volume = 1 / 2 ∧ T₂.volume = 1 / 2 ∧ 
  (∀ (p : Point), (p ∈ T₁ → dist (Point.center_of_sphere 1) p ≤ 1) ∧ 
  (p ∈ T₂ → dist (Point.center_of_sphere 1) p ≤ 1)) ∧ 
  ¬ ∃ (p : Point), p ∈ T₁ ∧ p ∈ T₂ :=
by
  sorry

end no_two_tetrahedra_of_volume_half_in_sphere_radius_one_l598_598794


namespace range_of_a_l598_598439

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then 2^x else abs (Real.log x / Real.log 2)

theorem range_of_a :
  {a : ℝ | (f a) < (1/2)} = {a : ℝ | (a < -1) ∨ ((sqrt 2 / 2) < a ∧ a < sqrt 2)} :=
by
  sorry

end range_of_a_l598_598439


namespace s_plough_time_l598_598154

theorem s_plough_time (r_s_combined_time : ℝ) (r_time : ℝ) (t_time : ℝ) (s_time : ℝ) :
  r_s_combined_time = 10 → r_time = 15 → t_time = 20 → s_time = 30 :=
by
  sorry

end s_plough_time_l598_598154


namespace sufficient_to_know_unitary_sum_ineq_deg_bound_l598_598808

noncomputable theory

namespace RealPolyRoots

-- Define the set E
def is_in_E (P : Polynomial ℝ) : Prop :=
  (∀ x : ℝ, Polynomial.eval x P = 0 → P.coeffs.all (λ c, c ∈ {-1, 0, 1})) ∧
  (∀ x : ℝ, Polynomial.eval x P = 0 → ∃ a : ℝ, x = a)

-- Question 1: Sufficient to know unitary polynomials in E that do not vanish at 0.
theorem sufficient_to_know_unitary (P : Polynomial ℝ) (hPE : is_in_E P) :
  ∀ (Q : Polynomial ℝ), Q.Monic ∧ Q ≠ 0 → is_in_E Q :=
sorry

-- Question 2: Proving the inequality
theorem sum_ineq (n : ℕ) (a : Fin n → ℝ) (ha : ∀ i, 0 < a i) :
  (∑ i j, a i / a j) ≥ n ^ 2 :=
sorry

-- Question 3: Bounding the degree of polynomials in E to 3
theorem deg_bound (P : Polynomial ℝ) (hPE : is_in_E P) (hMonic : P.Monic) (hNonZero : P.eval 0 ≠ 0) :
  P.degree ≤ 3 :=
sorry

end RealPolyRoots

end sufficient_to_know_unitary_sum_ineq_deg_bound_l598_598808


namespace profit_percentage_l598_598581

-- Define the selling price
def selling_price : ℝ := 900

-- Define the profit
def profit : ℝ := 100

-- Define the cost price as selling price minus profit
def cost_price : ℝ := selling_price - profit

-- Statement of the profit percentage calculation
theorem profit_percentage : (profit / cost_price) * 100 = 12.5 := by
  sorry

end profit_percentage_l598_598581


namespace regular_pentagon_construction_l598_598467

theorem regular_pentagon_construction (f : ℝ) 
    (A B C D E M : ℝ) 
    (equilateral_triangle : (AB = 2 * f ∧ BC = 2 * f ∧ CA = 2 * f)) 
    (circumscribed_circle: (∀ i j, dist i j = dist i j)) 
    (midpoints: (A = midpoint(B, C) ∧ B = midpoint(C, A)))
    (intersection: (ray_intersection(A, B, circumscribed_circle) = M)) 
    (triangle_equal_lengths : (AD = BD = AM)) 
    (triangle_lateral_lengths : (BC = CD = DE = EA = f)) : 
    is_regular_pentagon (pentagon A B C D E) := 
begin 
    sorry 
end

end regular_pentagon_construction_l598_598467


namespace no_combination_of_three_coins_sums_to_52_cents_l598_598526

def is_valid_coin (c : ℕ) : Prop :=
  c = 5 ∨ c = 10 ∨ c = 25 ∨ c = 50 ∨ c = 100

theorem no_combination_of_three_coins_sums_to_52_cents :
  ¬ ∃ a b c : ℕ, is_valid_coin a ∧ is_valid_coin b ∧ is_valid_coin c ∧ a + b + c = 52 :=
by 
  sorry

end no_combination_of_three_coins_sums_to_52_cents_l598_598526


namespace rhombus_other_diagonal_length_l598_598489

theorem rhombus_other_diagonal_length (area_square : ℝ) (side_length_square : ℝ) (d1_rhombus : ℝ) (d2_expected: ℝ) 
  (h1 : area_square = side_length_square^2) 
  (h2 : side_length_square = 8) 
  (h3 : d1_rhombus = 16) 
  (h4 : (d1_rhombus * d2_expected) / 2 = area_square) :
  d2_expected = 8 := 
by
  sorry

end rhombus_other_diagonal_length_l598_598489


namespace minimum_raft_weight_l598_598201

-- Define the weights of the animals.
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal.
def num_mice : ℕ := 5
def num_moles : ℕ := 3
def num_hamsters : ℕ := 4

-- The function that represents the minimum weight capacity required for the raft.
def minimum_raft_capacity : ℕ := 140

-- Prove that the minimum raft capacity to transport all animals is 140 grams.
theorem minimum_raft_weight :
  (∀ (total_weight : ℕ), 
    total_weight = (num_mice * weight_mouse) + (num_moles * weight_mole) + (num_hamsters * weight_hamster) →
    (exists (raft_capacity : ℕ), 
      raft_capacity = minimum_raft_capacity ∧
      raft_capacity >= 2 * weight_mouse)) :=
begin
  -- Initial state setup and logical structure.
  intros total_weight total_weight_eq,
  use minimum_raft_capacity,
  split,
  { refl },
  { have h1: 2 * weight_mouse = 140,
    { norm_num },
    rw h1,
    exact le_refl _,
  }
end

end minimum_raft_weight_l598_598201


namespace max_value_at_x0_l598_598764

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem max_value_at_x0 {x0 : ℝ} (h : ∃ x0, ∀ x, f x ≤ f x0) : 
  f x0 = x0 :=
sorry

end max_value_at_x0_l598_598764


namespace central_checker_exists_l598_598039

theorem central_checker_exists:
  let n := 25
  let center := (13, 13)
  let board := set.range (λ (i j : ℕ), (i, j)) 25
  let symmetric_with_respect_to_main_diagonals := 
    ∀ (i j : ℕ), (i, j) ∈ board → (25 - i + 1, 25 - j + 1) ∈ board
  let checkers := finset.univ.filter₂ (λ i j, (i, j) ∈ set.range (λ (i j : ℕ), (i, j)) 25) (finset.range n) (finset.range n)
  (checkers.card = 25)
  (∀ (i j : ℕ), checkers.contains (i, j) ↔ checkers.contains (n + 1 - i, n + 1 - j)) →
  25 % 2 = 1 →
  (∀ (i j : ℕ), checkers.card % 2 = 1) →
  ∃ (i j : ℕ), center = (i, j) :=
begin
  assume (h_conditions : symmetric_with_respect_to_main_diagonals checkers),
  apply sorry
end

end central_checker_exists_l598_598039


namespace locus_center_M_exists_circle_origin_l598_598734

-- Definitions/Conditions
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 18
def C₂ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 2
def circle_M_center (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Theorem Statements
theorem locus_center_M : ∀ (x y : ℝ), (C₁ x y) → (C₂ x y) → (circle_M_center x y) :=
by
  intros x y hC₁ hC₂
  sorry -- Proof goes here

theorem exists_circle_origin : ∃ (r : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = r^2) → (∃ (Mx My : ℝ), circle_M_center Mx My ∧ Mx ≠ My ∧ dot_prod (x, y) (Mx, My) = 0) :=
by
  use (2 * real.sqrt(6) / 3)
  intros x y hcircle
  sorry -- Proof goes here

end locus_center_M_exists_circle_origin_l598_598734


namespace hyperbola_equation_l598_598760

theorem hyperbola_equation (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_distance : 3 = b * (sqrt (a^2 + b^2)) / (sqrt (a^2 + b^2))) (h_eccentricity : (a^2 + b^2) / a^2 = 4) :
  (∃ a b : ℝ, (a > 0 ∧ b > 0 ∧ 3 = b ∧ (a = sqrt 3) ∧ ∀ x y : ℝ, (x^2 / 3 - y^2 / 9 = 1))) :=
by {
  sorry
}

end hyperbola_equation_l598_598760


namespace triangle_MXY_circumscribed_l598_598847

noncomputable def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

structure Triangle :=
  (A B C : Point)

structure Point :=
  (x : ℝ)
  (y : ℝ)

def circumcircle (Δ : Triangle) : set Point := sorry -- Assume definition of circumcircle

def is_midpoint (M P Q : Point) : Prop :=
  M = midpoint P Q

def is_inscribed (P Q R : Point) (C : set Point) : Prop :=
  P ∈ C ∧ Q ∈ C ∧ R ∈ C

theorem triangle_MXY_circumscribed (A B C E M X Y : Point) (Δ : Triangle) :
  is_midpoint M B C ∧
  is_midpoint X B E ∧
  is_midpoint Y C E ∧
  Δ = ⟨A, B, C⟩ →
  is_inscribed M X Y (circumcircle Δ) := sorry

end triangle_MXY_circumscribed_l598_598847


namespace first_player_win_l598_598525

def victory_condition (state : Fin (21 × 1) → Option (Fin 4) pos) : Prop :=
  state ⟨19, 0⟩ = some 0 ∧ state ⟨19, 0⟩ = some 1 ∧ state ⟨19, 0⟩ = some 2 ∧ state ⟨19, 0⟩ = some 3

def move_valid (state : Fin (21 × 1) → Option (Fin 4) pos) (move : Fin 4 × ℕ) : Prop :=
  ∀ (i j ∈ Fin 4), i < j → state ⟨i.snd, 0⟩ + move.snd < state ⟨j.snd, 0⟩

noncomputable def initial_state : Fin (21 × 1) → Option (Fin 4) pos := sorry

noncomputable def first_move_strategy (initial_state : Fin (21 × 1) → Option (Fin 4) pos) : Fin 4 × ℕ :=
  ⟨0, 2⟩

theorem first_player_win : ∀ initial_state,
  move_valid initial_state (first_move_strategy initial_state) →
  victory_condition (some (0 + 2)) := sorry

end first_player_win_l598_598525


namespace ratio_of_adult_to_kid_charge_l598_598800

variable (A : ℝ)  -- Charge for adults

-- Conditions
def kids_charge : ℝ := 3
def num_kids_per_day : ℝ := 8
def num_adults_per_day : ℝ := 10
def weekly_earnings : ℝ := 588
def days_per_week : ℝ := 7

-- Hypothesis for the relationship between charges and total weekly earnings
def total_weekly_earnings_eq : Prop :=
  days_per_week * (num_kids_per_day * kids_charge + num_adults_per_day * A) = weekly_earnings

-- Statement to be proved
theorem ratio_of_adult_to_kid_charge (h : total_weekly_earnings_eq A) : (A / kids_charge) = 2 := 
by 
  sorry

end ratio_of_adult_to_kid_charge_l598_598800


namespace simson_lines_intersect_at_one_point_l598_598092

/--
Given a cyclic quadrilateral ABCD inscribed in a circle, 
the Simson lines of points A, B, C, and D with respect to 
the triangles BCD, CDA, DAB, and ABC respectively, 
intersect at one point.
-/
theorem simson_lines_intersect_at_one_point
  (A B C D O H : Point)
  (h_cycle: is_cyclic_quadrilateral O A B C D)
  (l_a l_b l_c l_d: Line)
  (h_la: is_simson_line A B C D O l_a)
  (h_lb: is_simson_line B C D A O l_b)
  (h_lc: is_simson_line C D A B O l_c)
  (h_ld: is_simson_line D A B C O l_d)
  : intersects_at_one_point l_a l_b l_c l_d H := 
sorry

end simson_lines_intersect_at_one_point_l598_598092


namespace more_trees_died_than_survived_l598_598739

def haley_trees : ℕ := 14
def died_in_typhoon : ℕ := 9
def survived_trees := haley_trees - died_in_typhoon

theorem more_trees_died_than_survived : (died_in_typhoon - survived_trees) = 4 := by
  -- proof goes here
  sorry

end more_trees_died_than_survived_l598_598739


namespace total_age_is_47_l598_598957

-- Define the ages of B and conditions
def B : ℕ := 18
def A : ℕ := B + 2
def C : ℕ := B / 2

-- Prove the total age of A, B, and C
theorem total_age_is_47 : A + B + C = 47 :=
by
  sorry

end total_age_is_47_l598_598957


namespace cube_root_of_8_is_2_l598_598079

theorem cube_root_of_8_is_2 : ∃ x : ℝ, x ^ 3 = 8 ∧ x = 2 :=
by
  have h : (2 : ℝ) ^ 3 = 8 := by norm_num
  exact ⟨2, h, rfl⟩

end cube_root_of_8_is_2_l598_598079


namespace find_point_l598_598295

variables {x y : ℝ}

def mapping (x y: ℝ) := (x + y, 2 * x - y)

theorem find_point (h : mapping x y = (5, 1)) : x = 2 ∧ y = 3 :=
by
  unfold mapping at h
  cases h with h1 h2
  have hx : x + y = 5 := h1
  have hy : 2 * x - y = 1 := h2
  have hx' : y = 5 - x, from eq.symm hx
  rw [hx'] at hy
  have hxy' : 2 * x - (5 - x) = 1 := hy
  have hxy'' : 3 * x - 5 = 1 := hxy'
  have hxy''' : 3 * x = 6 := by linarith
  have hx' : x = 2 := by linarith
  have hy' : y = 3 := by linarith
  exact ⟨hx', hy'⟩

end find_point_l598_598295


namespace solution_exists_l598_598628

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m ∈ finset.Icc 2 (n - 1), m ∣ n → m = n

def problem_statement : Prop :=
∃ S : finset ℤ, S.card = 12 ∧
  (finset.filter is_prime S).card = 6 ∧
  (finset.filter (λ n, n % 2 ≠ 0) S).card = 9 ∧
  (finset.filter (λ n, n ≥ 0) S).card = 10 ∧
  (finset.filter (λ n, n > 10) S).card = 7

theorem solution_exists : problem_statement :=
sorry

end solution_exists_l598_598628


namespace obtuse_triangle_area_side_l598_598913

theorem obtuse_triangle_area_side (a b : ℝ) (C : ℝ) 
  (h1 : a = 8) 
  (h2 : C = 150 * (π / 180)) -- converting degrees to radians
  (h3 : 1 / 2 * a * b * Real.sin C = 24) : 
  b = 12 :=
by sorry

end obtuse_triangle_area_side_l598_598913


namespace range_of_x_for_inequality_l598_598698

variable {f : ℝ → ℝ}

-- Conditions from the problem
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def increasing_on_nonnegative (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 ≤ x → x < y → f x < f y

-- Statement to prove
theorem range_of_x_for_inequality
  (h_even : is_even f)
  (h_increasing : increasing_on_nonnegative f)
  (h_derivative_positive : ∀ x : ℝ, 0 ≤ x → 0 < f' x)
  : {x : ℝ | f (x^2 - 2 * x) < f x} = {x : ℝ | 1 < x ∧ x < 3} :=
by sorry

end range_of_x_for_inequality_l598_598698


namespace max_imag_part_of_roots_l598_598985

noncomputable def polynomial (z : ℂ) : ℂ := z^12 - z^9 + z^6 - z^3 + 1

theorem max_imag_part_of_roots :
  ∃ (z : ℂ), polynomial z = 0 ∧ ∀ w, polynomial w = 0 → (z.im ≤ w.im) := sorry

end max_imag_part_of_roots_l598_598985


namespace coprime_tiling_impossible_l598_598700

theorem coprime_tiling_impossible {n m : ℕ} (coprime : Nat.gcd n m = 1) :
  ¬ (can_tile (n, n) ∧ can_tile (m, m)) :=
sorry

-- Definitions of can_tile and the specific polyominos would be necessary 
-- to make this complete, here's a rough sketch of what they might look like:

def L_shape_polyomino : Polyomino := sorry
def another_polyomino : Polyomino := sorry

def can_tile (board_dim : ℕ × ℕ) : Prop := 
  ∃ tiling : Board_tiling L_shape_polyomino another_polyomino board_dim, tiling.valid

structure Board_tiling (L another : Polyomino) (dims : ℕ × ℕ) :=
  (valid : True) -- A proper definition describing the tiling condition.

end coprime_tiling_impossible_l598_598700


namespace sequence_a4_value_l598_598894

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * a n + 1

theorem sequence_a4_value : a 3 = 15 :=
by
  sorry

end sequence_a4_value_l598_598894


namespace card_A_ge_card_B_plus_card_C_l598_598910

open Finset

variables (n k : ℕ) (hn : 0 < k ∧ k < n) 
variables (a : Finₓ n → ℝ)
noncomputable def A : Finset (Finₓ n) := {i | a i > a (i - k) ∧ a i > a (i - 1) ∧ a i > a (i + 1) ∧ a i > a (i + k) ∨ a i < a (i - k) ∧ a i < a (i - 1) ∧ a i < a (i + 1) ∧ a i < a (i + k)}
noncomputable def B : Finset (Finₓ n) := {i | a i > a (i - k) ∧ a i > a (i + k) ∧ a i < a (i - 1) ∧ a i < a (i + 1)}
noncomputable def C : Finset (Finₓ n) := {i | a i > a (i - 1) ∧ a i > a (i + 1) ∧ a i < a (i - k) ∧ a i < a (i + k)}

theorem card_A_ge_card_B_plus_card_C
  (ha_distinct : function.injective a)
  : A n k a ⊇ A B C := 
begin
  rw Finset.subset_def, simp only [mem_def,A_def,B_def,C_def,forall_prop_of_true,univ_subset_iff,uniserial.to_Fintype_simp],
  sorry
end

end card_A_ge_card_B_plus_card_C_l598_598910


namespace evaluate_division_l598_598583

theorem evaluate_division : 64 / 0.08 = 800 := by
  sorry

end evaluate_division_l598_598583


namespace wendy_full_face_time_l598_598117

-- Define the constants based on the conditions
def num_products := 5
def wait_time := 5
def makeup_time := 30

-- Calculate the total time to put on "full face"
def total_time (products : ℕ) (wait_time : ℕ) (makeup_time : ℕ) : ℕ :=
  (products - 1) * wait_time + makeup_time

-- The theorem stating that Wendy's full face routine takes 50 minutes
theorem wendy_full_face_time : total_time num_products wait_time makeup_time = 50 :=
by {
  -- the proof would be provided here, for now we use sorry
  sorry
}

end wendy_full_face_time_l598_598117


namespace ratio_of_areas_l598_598533

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l598_598533


namespace tg_sum_inequality_l598_598582

variable {α β γ : ℝ}

-- Defining acute angles
def is_acute_triangle (α β γ : ℝ) : Prop :=
  α < π / 2 ∧ β < π / 2 ∧ γ < π / 2

-- Tangent of angles condition we want to prove
theorem tg_sum_inequality (h : is_acute_triangle α β γ) :
  Real.tan α + Real.tan β + Real.tan γ ≥ 3 * Real.sqrt 3 :=
sorry

end tg_sum_inequality_l598_598582


namespace sum_of_ages_26_l598_598004

-- Define an age predicate to manage the three ages
def is_sum_of_ages (kiana twin : ℕ) : Prop :=
  kiana < twin ∧ twin * twin * kiana = 180 ∧ (kiana + twin + twin = 26)

theorem sum_of_ages_26 : 
  ∃ (kiana twin : ℕ), is_sum_of_ages kiana twin :=
by 
  sorry

end sum_of_ages_26_l598_598004


namespace find_angle_FEA_l598_598807

-- Assume we have a square ABCD and specific points E and F as described.
variables (A B C D E F : Type) [HasAngle (E F A : Type)]

-- Definition of the square ABCD and the point E such that EB = AB.
def is_square (A B C D : Type) : Prop :=
  -- Definition assuming the properties of a square
  sorry

def point_on_segment (B D E : Type) : Prop :=
  sorry

def EB_eq_AB (E B A : Type) : Prop :=
  sorry

def point_F_intersection (C E A D F : Type) : Prop :=
  sorry

-- The proof obligation to show that the angle FEA = 45 degrees.
theorem find_angle_FEA (h1 : is_square A B C D)
                      (h2 : point_on_segment B D E)
                      (h3 : EB_eq_AB E B A)
                      (h4 : point_F_intersection C E A D F):
  angle E F A = 45 :=
  sorry

end find_angle_FEA_l598_598807


namespace marble_probability_l598_598252

theorem marble_probability (x y r_x r_y b_x b_y p q : ℕ)
  (h1 : x + y = 30)
  (h2 : x = r_x + b_x)
  (h3 : y = r_y + b_y)
  (h4 : (r_x : ℚ) / x * (r_y : ℚ) / y = 2 / 3)
  (hpq_coprime : Nat.coprime p q)
  (h_eq : (b_x : ℚ) / x * (b_y : ℚ) / y = p / q) :
  p + q = 7 :=
sorry

end marble_probability_l598_598252


namespace find_b_l598_598319

def complexProblem (b : ℝ) : Bool :=
  let z := (1 + Complex.i) / (1 - Complex.i) + (1 / 2) * b
  z.re = z.im

theorem find_b : ∃ b : ℝ, complexProblem b ∧ b = 2 := by
  sorry

end find_b_l598_598319


namespace jessica_final_balance_l598_598218

variable {original_balance current_balance final_balance withdrawal1 withdrawal2 deposit1 deposit2 : ℝ}

theorem jessica_final_balance:
  (2 / 5) * original_balance = 200 → 
  current_balance = original_balance - 200 → 
  withdrawal1 = (1 / 3) * current_balance → 
  current_balance - withdrawal1 = current_balance - (1 / 3 * current_balance) → 
  deposit1 = (1 / 5) * (current_balance - (1 / 3 * current_balance)) → 
  final_balance = (current_balance - (1 / 3 * current_balance)) + deposit1 → 
  deposit2 / 7 * 3 = final_balance - (current_balance - (1 / 3 * current_balance) + deposit1) → 
  (final_balance + deposit2) = 420 :=
sorry

end jessica_final_balance_l598_598218


namespace problem1_problem2_l598_598163

-- Problem 1 statement in Lean 4
noncomputable def partitioned_into_three (A1 A2 A3 : set ℕ) : Prop :=
(∀ n ≥ 1, n ∈ A1 ∨ n ∈ A2 ∨ n ∈ A3) ∧
(∀ (A : set ℕ), A = A1 ∨ A = A2 ∨ A = A3 → 
    ∀ n ≥ 15, (∃ x y : ℕ, x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ x + y = n))

theorem problem1 : ∃ (A1 A2 A3 : set ℕ), partitioned_into_three A1 A2 A3 :=
sorry

-- Problem 2 statement in Lean 4
noncomputable def partitioned_into_four (A1 A2 A3 A4 : set ℕ) : Prop :=
∀ n ≥ 1, n ∈ A1 ∨ n ∈ A2 ∨ n ∈ A3 ∨ n ∈ A4

theorem problem2 : ∀ (A1 A2 A3 A4 : set ℕ), partitioned_into_four A1 A2 A3 A4 →
(∃ n ≥ 15, ¬(∃ x y : ℕ, x ∈ (A1 ∪ A2 ∪ A3 ∪ A4) ∧ y ∈ (A1 ∪ A2 ∪ A3 ∪ A4) ∧ x ≠ y ∧ x + y = n)) :=
sorry

end problem1_problem2_l598_598163


namespace smallest_angle_WYZ_l598_598811

-- Define the given angle measures.
def angle_XYZ : ℝ := 40
def angle_XYW : ℝ := 15

-- The theorem statement proving the smallest possible degree measure for ∠WYZ
theorem smallest_angle_WYZ : angle_XYZ - angle_XYW = 25 :=
by
  -- Add the proof here
  sorry

end smallest_angle_WYZ_l598_598811


namespace parabola_translation_l598_598902

theorem parabola_translation :
  (∀ x : ℝ, y = x^2 → y' = (x - 1)^2 + 3) :=
sorry

end parabola_translation_l598_598902


namespace find_difference_l598_598152

variables (a b c : ℝ)

theorem find_difference (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 50) : c - a = 10 := by
  sorry

end find_difference_l598_598152


namespace coins_problem_l598_598748

theorem coins_problem
  (N Q : ℕ)
  (h1 : N + Q = 21)
  (h2 : 0.05 * N + 0.25 * Q = 3.65) :
  Q = 13 :=
by
  -- To be proved
  sorry

end coins_problem_l598_598748


namespace next_term_geometric_sequence_l598_598564

theorem next_term_geometric_sequence (x : ℝ) :
  ∃ t : ℝ, (4, 12 * x^2, 36 * x^4, 108 * x^6, t) is_geom_seq → t = 324 * x^8 :=
by { sorry }

def is_geom_seq (seq : ℕ → ℝ) : Prop :=
  ∀ n, seq (n + 1) = seq n * 3 * x ^ 2

end next_term_geometric_sequence_l598_598564


namespace only_D_is_odd_and_decreasing_l598_598987

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x: ℝ, f(-x) = -f(x)
def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop := ∀ x y: ℝ, x ∈ I → y ∈ I → x < y → f(x) > f(y)

def f_A (x : ℝ) : ℝ := -1 / x
def f_B (x : ℝ) : ℝ := x
def f_C (x : ℝ) : ℝ := Real.log (|x - 1|)
def f_D (x : ℝ) : ℝ := -Real.sin x

theorem only_D_is_odd_and_decreasing :
  is_odd_function f_D ∧ is_decreasing_on f_D (set.Ioo 0 1) ∧
  (¬is_odd_function f_A ∨ ¬is_decreasing_on f_A (set.Ioo 0 1)) ∧
  (¬is_odd_function f_B ∨ ¬is_decreasing_on f_B (set.Ioo 0 1)) ∧
  (¬is_odd_function f_C ∨ ¬is_decreasing_on f_C (set.Ioo 0 1)) :=
by
  sorry

end only_D_is_odd_and_decreasing_l598_598987


namespace number_of_terms_l598_598895

noncomputable def Sn (n : ℕ) : ℝ := sorry

def an_arithmetic_seq (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

theorem number_of_terms {a : ℕ → ℝ}
  (h_arith : an_arithmetic_seq a)
  (cond1 : a 1 + a 2 + a 3 + a 4 = 1)
  (cond2 : a 5 + a 6 + a 7 + a 8 = 2)
  (cond3 : Sn = 15) :
  ∃ n, n = 16 :=
sorry

end number_of_terms_l598_598895


namespace line_through_point_P_l598_598290

theorem line_through_point_P {C P : ℝ × ℝ} (x y : ℝ) :
  let circle_C := {c : ℝ × ℝ | (fst c + 2)^2 + (snd c)^2 = 4} in
  C = (-2, 0) →
  P = (-1, 1) →
  (x - y + 2 = 0) →
  dist (-2, 0) (2, 3) = 5 →
  is_tangent circle_C {c : ℝ × ℝ | (fst c - 2)^2 + (snd c - 3)^2 = 9} →
  ∃ l : ℝ → ℝ, l (-1) = 1 ∧ (forall t : ℝ, (t, l t) ∈ circle_C) ∧ minimizes_angle l P (-2, 0) (-1, 1) → 
  (forall x y, x + y = 0) :=
by
  intros circle_C C P hC hP h_intersect h_dist h_tangent,
  sorry

end line_through_point_P_l598_598290


namespace chess_or_basketball_students_l598_598776

-- Definitions based on the conditions
def percentage_likes_basketball : ℝ := 0.4
def percentage_likes_chess : ℝ := 0.1
def total_students : ℕ := 250

-- Main statement to prove
theorem chess_or_basketball_students : 
  (percentage_likes_basketball + percentage_likes_chess) * total_students = 125 :=
by
  sorry

end chess_or_basketball_students_l598_598776


namespace evalCeilingOfNegativeSqrt_l598_598657

noncomputable def ceiling_of_negative_sqrt : ℤ :=
  Int.ceil (-(Real.sqrt (36 / 9)))

theorem evalCeilingOfNegativeSqrt : ceiling_of_negative_sqrt = -2 := by
  sorry

end evalCeilingOfNegativeSqrt_l598_598657


namespace number_of_possible_ordered_pairs_l598_598095

theorem number_of_possible_ordered_pairs :
  ∃ (b s : ℕ), (b > 0) → (s > 0) →
  (∀ n : ℕ, ∃ (log_sum : ℤ),
    (n = 15) →
    (log_sum = 3010) →
    (∑ i in finset.range n, log 10 (b * s^i)) = log_sum) →
  ∃ (count : ℕ), count = 10 :=
begin
  sorry
end

end number_of_possible_ordered_pairs_l598_598095


namespace max_distance_C_to_l_l598_598786

noncomputable def curve_C (α : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos α, Real.sin α)

noncomputable def line_l (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ + Real.sin θ) = 4

theorem max_distance_C_to_l :
  let C := { p : ℝ × ℝ | ∃ α : ℝ, p = (√3 * Real.cos α, Real.sin α) }
  let l := { p : ℝ × ℝ | ∃ ρ θ : ℝ, (ρ * (Real.cos θ + Real.sin θ) = 4) ∧ p = (ρ * Real.cos θ, ρ * Real.sin θ) }
  ∀ α, let A := (√3 * Real.cos α, Real.sin α) in A ∈ C →
  ∃ p : ℝ × ℝ, p ∈ l ∧
  Real.dist A p = 3 :=
by
  sorry

end max_distance_C_to_l_l598_598786


namespace sum_of_interior_angles_l598_598521

theorem sum_of_interior_angles (n : ℕ) (h : n ≥ 3) : (n-2) * 180 = sum_of_interior_angles n := 
sorry

end sum_of_interior_angles_l598_598521


namespace ratio_of_areas_l598_598535

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l598_598535


namespace number_of_matches_in_round_robin_l598_598614

theorem number_of_matches_in_round_robin (n : ℕ) (h : n = 10) :
  (n * (n - 1)) / 2 = 45 :=
by
  rw h
  norm_num

end number_of_matches_in_round_robin_l598_598614


namespace exists_max_f_value_l598_598276

theorem exists_max_f_value : 
  ∃ x0 y0 : ℝ, 
    0 < x0 ∧ 0 < y0 ∧ x0 = 1/real.sqrt 2 ∧ y0 = 1/real.sqrt 2 ∧ 
    (∀ x y : ℝ, 0 < x → 0 < y → min x (y / (x^2 + y^2)) ≤ 1/real.sqrt 2) :=
  sorry

end exists_max_f_value_l598_598276


namespace parallelepiped_base_sides_lengths_l598_598611

theorem parallelepiped_base_sides_lengths
  (D1 D2 : ℝ) (angle : ℝ) (hD1 : D1 = 20) (hD2 : D2 = 8) (hangle : angle = (real.pi / 3)) :
  ∃ (a b : ℝ), (a = 2 * real.sqrt 5) ∧ (b = real.sqrt 30) :=
by
  sorry

end parallelepiped_base_sides_lengths_l598_598611


namespace least_positive_difference_l598_598461

noncomputable def sequenceA : List ℕ := [3, 9, 27, 81, 243]
noncomputable def sequenceB : List ℕ := [10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 205, 220, 235, 250, 265, 280, 295, 310, 325, 340, 355, 370, 385, 400, 415, 430, 445, 460, 475, 490]

theorem least_positive_difference :
  ∃ (a ∈ sequenceA) (b ∈ sequenceB), |a - b| = 1 := by
  sorry

end least_positive_difference_l598_598461


namespace proof_problem_l598_598325

noncomputable def f (x a : ℝ) : ℝ := 
  if |x| ≤ 1 then Real.logBase 2 (x + a)
  else -10 / (|x| + 3)

theorem proof_problem (a : ℝ) (h₀ : f 0 a = 2) : 4 + f (-2) a = 2 := 
by
  sorry

end proof_problem_l598_598325


namespace donny_cost_of_apples_l598_598636

def cost_of_apples (small_cost medium_cost big_cost : ℝ) (n_small n_medium n_big : ℕ) : ℝ := 
  n_small * small_cost + n_medium * medium_cost + n_big * big_cost

theorem donny_cost_of_apples :
  cost_of_apples 1.5 2 3 6 6 8 = 45 :=
by
  sorry

end donny_cost_of_apples_l598_598636


namespace Uncle_Bob_more_candy_bars_l598_598681

theorem Uncle_Bob_more_candy_bars (f j : ℕ) (hb : 0.4 * j = 120) (h_total : Fred + Uncle_Bob = 30) (h_j : j = 10 * (Fred + Uncle_Bob)) (Fred_eq : Fred = 12) : Uncle_Bob - Fred = 6 :=
by
  sorry

end Uncle_Bob_more_candy_bars_l598_598681


namespace min_capacity_for_raft_l598_598209

-- Define the weights of the animals
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal
def number_mice : ℕ := 5
def number_moles : ℕ := 3
def number_hamsters : ℕ := 4

-- Define the minimum weight capacity for the raft
def min_weight_capacity : ℕ := 140

-- Prove that the minimum weight capacity the raft must have to transport all animals is 140 grams.
theorem min_capacity_for_raft :
  (weight_mouse * 2 ≤ min_weight_capacity) ∧ 
  (∀ trip_weight, trip_weight ≥ min_weight_capacity → 
    (trip_weight = weight_mouse * 2 ∨ trip_weight = weight_mole * 2 ∨ trip_weight = weight_hamster * 2)) :=
by 
  sorry

end min_capacity_for_raft_l598_598209


namespace solve_system_of_equations_l598_598871

theorem solve_system_of_equations (x y : ℚ) :
  (10 / (2 * x + 3 * y - 29) + 9 / (7 * x - 8 * y + 24) = 8) ∧ 
  ((2 * x + 3 * y - 29) / 2 = (7 * x - 8 * y) / 3 + 8) →
  x = 5 ∧ y = 7 :=
begin
  sorry
end

end solve_system_of_equations_l598_598871


namespace find_g_inv_84_l598_598752

def g (x : ℝ) : ℝ := 3 * x ^ 3 + 3

theorem find_g_inv_84 (x : ℝ) (h : g x = 84) : x = 3 :=
by 
  unfold g at h
  -- Begin proof steps here, but we will use sorry to denote placeholder 

  sorry

end find_g_inv_84_l598_598752


namespace isosceles_right_triangle_leg_length_l598_598090

theorem isosceles_right_triangle_leg_length (hypotenuse : ℝ) (h_hypotenuse : hypotenuse = 8.485281374238571) :
  (∃ (a : ℝ), abs(a - 6) < 1e-1 ∧ abs(hypotenuse - a * Real.sqrt 2) < 1e-6) :=
by { sorry }

end isosceles_right_triangle_leg_length_l598_598090


namespace dogwood_trees_final_count_l598_598897

theorem dogwood_trees_final_count (initial_trees workers_A workers_B workers_C workers_D workers_E losses_C losses_D : ℕ) :
  initial_trees = 34 →
  workers_A = 12 →
  workers_B = 10 →
  workers_C = 15 →
  workers_D = 8 →
  workers_E = 4 →
  losses_C = 2 →
  losses_D = 1 →
  initial_trees + (workers_A + workers_B + workers_C + workers_D + workers_E - (losses_C + losses_D)) = 80 :=
by
  intros initial_trees_eq workers_A_eq workers_B_eq workers_C_eq workers_D_eq workers_E_eq losses_C_eq losses_D_eq
  rw [initial_trees_eq, workers_A_eq, workers_B_eq, workers_C_eq, workers_D_eq, workers_E_eq, losses_C_eq, losses_D_eq ]
  norm_num
  sorry

end dogwood_trees_final_count_l598_598897


namespace probability_three_digit_div_by_3_l598_598283

-- define the set of possible digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

-- count the total number of three-digit numbers without repeated digits
def total_num_three_digit_numbers : ℕ :=
  9 * 9 * 8 -- (first digit can't be 0, so 9 options, then 9 remaining, then 8 remaining)

-- define the groups based on remainder when divided by 3
def group_0 : List ℕ := [0, 3, 6, 9]
def group_1 : List ℕ := [1, 4, 7]
def group_2 : List ℕ := [2, 5, 8]

-- calculate the number of such multiples of 3
def num_multiples_of_3 : ℕ :=
  -- permutations within each group: A_3^3 for non-zero groups and A_4^3 for group with zero
  List.permutations([0, 3, 6, 9]).length + 
  List.permutations([1, 4, 7]).length +
  List.permutations([2, 5, 8]).length + 
  -- selecting one from each group and then permuting
  List.permutations([1, 2, 3]).length +
  List.permutations([1, 2, 6]).length +
  List.permutations([1, 2, 9]).length +
  List.permutations([1, 5, 3]).length +
  List.permutations([1, 5, 6]).length +
  List.permutations([1, 5, 9]).length +
  List.permutations([4, 2, 3]).length +
  List.permutations([4, 2, 6]).length +
  List.permutations([4, 2, 9]).length +
  List.permutations([4, 5, 3]).length +
  List.permutations([4, 5, 6]).length +
  List.permutations([4, 5, 9]).length +
  List.permutations([7, 2, 3]).length +
  List.permutations([7, 2, 6]).length +
  List.permutations([7, 2, 9]).length +
  List.permutations([7, 5, 3]).length +
  List.permutations([7, 5, 6]).length +
  List.permutations([7, 5, 9]).length

-- define the probability as a fraction using results above
def probability_divisible_by_3 : ℚ :=
  num_multiples_of_3 / total_num_three_digit_numbers

theorem probability_three_digit_div_by_3 : 
  probability_divisible_by_3 = 19 / 54 :=
sorry

end probability_three_digit_div_by_3_l598_598283


namespace mean_greater_than_median_by_six_l598_598150

theorem mean_greater_than_median_by_six (x : ℕ) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 37)) / 5
  let median := x + 4
  mean - median = 6 :=
by
  sorry

end mean_greater_than_median_by_six_l598_598150


namespace students_below_50_is_60_l598_598971

-- Conditions
noncomputable def total_students := 600
noncomputable def score_segments : List (ℕ × ℕ × ℝ) := [
  (50, 60, 0.15),
  (60, 70, 0.15),
  (70, 80, 0.3),
  (80, 90, 0.25),
  (90, 100, 0.05)
]

def frequency_below_50 := 
  1 - (List.sum (score_segments.map Prod.snd.snd.snd)) 

noncomputable def students_below_50 := 
  total_students * frequency_below_50

theorem students_below_50_is_60 : students_below_50 = 60 := by
  -- We have already defined conditions
  sorry

end students_below_50_is_60_l598_598971


namespace initial_deposit_l598_598605

theorem initial_deposit (x : ℝ) 
  (h1 : x - (1 / 4) * x - (4 / 9) * ((3 / 4) * x) - 640 = (3 / 20) * x) 
  : x = 2400 := 
by 
  sorry

end initial_deposit_l598_598605


namespace raft_minimum_capacity_l598_598196

theorem raft_minimum_capacity 
  (mice : ℕ) (mice_weight : ℕ) 
  (moles : ℕ) (mole_weight : ℕ) 
  (hamsters : ℕ) (hamster_weight : ℕ) 
  (raft_cannot_move_without_rower : Bool)
  (rower_condition : ∀ W, W ≥ 2 * mice_weight) :
  mice = 5 → mice_weight = 70 →
  moles = 3 → mole_weight = 90 →
  hamsters = 4 → hamster_weight = 120 →
  ∃ W, (W = 140) :=
by
  intros mice_eq mice_w_eq moles_eq mole_w_eq hamsters_eq hamster_w_eq
  use 140
  sorry

end raft_minimum_capacity_l598_598196


namespace percentage_of_y_l598_598372

theorem percentage_of_y (y : ℝ) (h : y > 0) : (9 * y) / 20 + (3 * y) / 10 = 0.75 * y :=
by
  sorry

end percentage_of_y_l598_598372


namespace ratio_of_areas_l598_598530

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l598_598530


namespace sin_A_of_triangle_area_and_geom_mean_l598_598364

theorem sin_A_of_triangle_area_and_geom_mean 
  (AB AC : ℝ) 
  (area : ℝ)
  (h_area : area = 50)
  (geom_mean : ℝ)
  (h_geom_mean : geom_mean = 10)
  (h_geom_mean_eq : (AB * AC) = geom_mean^2) 
  : (sin (acos ((AB^2 + AC^2 - (2 * AB * AC)) / (2 * AB * AC)))) = 1 := 
by
  sorry

end sin_A_of_triangle_area_and_geom_mean_l598_598364


namespace angle_ACD_is_90_l598_598389

noncomputable def quadrilateral.ABCD (A B C D : Point) : Prop :=
  convex_quadrilateral ABCD ∧
  ∃ E, is_midpoint E A D ∧ bisects_angle E B ∠C = ∠A + ∠D

theorem angle_ACD_is_90 
  (A B C D : Point) 
  (h : quadrilateral.ABCD A B C D) : 
  ∠ACD = 90° :=
sorry

end angle_ACD_is_90_l598_598389


namespace total_cents_correct_l598_598406

-- Define the amounts in cents for each person
def Lance_cents : ℕ := 70
def Margaret_cents : ℕ := 75
def Guy_cents : ℕ := 60
def Bill_cents : ℕ := 60

-- Define the conversion rates
def pound_to_dollar : ℝ := 1.4
def yen_to_dollar : ℝ := 0.009
def franc_to_dollar : ℝ := 1.1

-- Define the amounts in their local currencies
def Alex_pounds : ℕ := 2
def Fiona_yen : ℕ := 150
def Kevin_francs : ℕ := 5

-- Convert the local currencies to dollars
def Alex_dollars : ℝ := Alex_pounds * pound_to_dollar
def Fiona_dollars : ℝ := Fiona_yen * yen_to_dollar
def Kevin_dollars : ℝ := Kevin_francs * franc_to_dollar

-- Convert dollars to cents
def Alex_cents : ℕ := (Alex_dollars * 100).toNat
def Fiona_cents : ℕ := (Fiona_dollars * 100).toNat
def Kevin_cents : ℕ := (Kevin_dollars * 100).toNat

-- Total cents
def total_cents : ℕ :=
  Lance_cents + Margaret_cents + Guy_cents + Bill_cents + Alex_cents + Fiona_cents + Kevin_cents

theorem total_cents_correct :
  total_cents = 1230 := by
  sorry

end total_cents_correct_l598_598406


namespace complement_union_l598_598732

open Set

def U : Set ℕ := {x | -1 ≤ x ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_union (x : ℕ) : x ∈ U → x ∉ (A ∪ B) → x = 2 ∨ x = 4 :=
by
  intro hx hAUB
  have hU : U = {1, 2, 3, 4, 5} := by 
    ext y
    simp [U]
    cases y <;> finish
  rw [hU, union_def, mem_union, mem_compl_iff, mem_set_of_eq, not_or_distrib] at hx hAUB
  cases hx
  { exfalso; exact not_mem_of_mem_compl hAUB hx.left }
  exact ⟨hx.right.left, hx.right.right⟩

end complement_union_l598_598732


namespace number_of_distinct_scores_l598_598593

-- Definitions for the problem conditions
def free_throw_value : ℕ := 1
def two_pointer_value : ℕ := 2
def three_pointer_value : ℕ := 3

def total_shots : ℕ := 7

-- Statement to prove the equivalence
theorem number_of_distinct_scores : 
  (set.range (λ (p : ℕ × ℕ × ℕ), (p.1 * free_throw_value + p.2 * two_pointer_value + p.2 * three_pointer_value)) ∩ (set.Icc 0 total_shots) = set.range _)) = 15 := 
sorry

end number_of_distinct_scores_l598_598593


namespace units_digit_of_calculation_l598_598567

-- Base definitions for units digits of given numbers
def units_digit (n : ℕ) : ℕ := n % 10

-- Main statement to prove
theorem units_digit_of_calculation : 
  units_digit ((25 ^ 3 + 17 ^ 3) * 12 ^ 2) = 2 :=
by
  -- This is where the proof would go, but it's omitted as requested
  sorry

end units_digit_of_calculation_l598_598567


namespace find_unknown_number_l598_598515

def unknown_number (x : ℝ) : Prop :=
  (0.5^3) - (0.1^3 / 0.5^2) + x + (0.1^2) = 0.4

theorem find_unknown_number : ∃ (x : ℝ), unknown_number x ∧ x = 0.269 :=
by
  sorry

end find_unknown_number_l598_598515


namespace max_candies_one_student_l598_598770

theorem max_candies_one_student (n : ℕ) (mean_candies : ℕ) 
  (total_students : ℕ) (min_candies_each : ℕ) 
  (h_n : n = 25) 
  (h_mean : mean_candies = 6) 
  (h_total_students : total_students = 25) 
  (h_min_candies_each : min_candies_each = 2) 
  (h_mean_calc : total_students * mean_candies = n * mean_candies) :
  ∃ (max_candies : ℕ), max_candies = 102 :=
by
  use 102
  sorry

end max_candies_one_student_l598_598770


namespace percentage_of_value_l598_598118

theorem percentage_of_value (x y : ℝ) (h : y = 0.765) : x * y = 984.495 :=
by
  sorry

# Check this out for the specific example:
def specific_example : Prop := percentage_of_value 1287 0.765 rfl

end percentage_of_value_l598_598118


namespace pentagon_volume_ratio_l598_598735

noncomputable def pentagon_ratio (a b : ℝ) : ℝ := 
  ( (3 * Real.sqrt 5 + 5) / (6 * Real.sqrt 5 + 12) ) ^ (1/3 : ℝ)

theorem pentagon_volume_ratio (a b : ℝ) 
  (h : volume_by_side_rotation a = volume_by_diagonal_rotation b) : 
  a/b = pentagon_ratio a b := 
by 
  sorry

end pentagon_volume_ratio_l598_598735


namespace line_equations_through_P_tangent_lines_l598_598941

-- Define the straight lines
def l1 : ℝ → ℝ := λ x, 2 * x
def l2 : ℝ → ℝ := λ y, 3 - y

-- Define intersection point P
def P : ℝ × ℝ := (1, 2)

-- Define circle C
def C (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 8 * y + 19 = 0

-- Define the conditions for the tangent problem
def tangent_conditions (x y : ℝ) (P : ℝ × ℝ) : Prop :=
  (x = -4) ∧ (y = 5) ∧ C x y

-- Define the problem statement for line equations
theorem line_equations_through_P (x y : ℝ) : 
  (y = 2 * x) → 
  (x + y = 3) →
  (dist (0, 0) (1, 2) = 1) →
  (x = 1 ∨ 3 * x - 4 * y + 5 = 0) :=
by sorry

-- Define the problem statement for tangent lines
theorem tangent_lines (x y : ℝ) : 
  (x^2 + y^2 + 4 * x - 8 * y + 19 = 0) → 
  ((x, y) = (-4, 5)) →
  (y = 5 ∨ 4 * x + 3 * y + 1 = 0) :=
by sorry

end line_equations_through_P_tangent_lines_l598_598941


namespace ratio_of_areas_l598_598537

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l598_598537


namespace sum_inequality_l598_598806

theorem sum_inequality (n : ℕ) (x : ℕ → ℝ) (h₁ : 0 < n) (h₂ : ∀ i, 1 ≤ i → i ≤ n → 0 ≤ x i) (h₃ : (finset.range n).sum (λ i, x (i + 1)) = 1) :
  (finset.range n).sum (λ i, (x (i + 1)) * (1 - (x (i + 1)))^2) ≤ (1 - 1/n)^2 :=
begin
  sorry
end

end sum_inequality_l598_598806


namespace evalCeilingOfNegativeSqrt_l598_598656

noncomputable def ceiling_of_negative_sqrt : ℤ :=
  Int.ceil (-(Real.sqrt (36 / 9)))

theorem evalCeilingOfNegativeSqrt : ceiling_of_negative_sqrt = -2 := by
  sorry

end evalCeilingOfNegativeSqrt_l598_598656


namespace equilateral_iff_area_relation_l598_598488

section
variables {ABC : Type*}
variables (A B C A1 B1 C1 : ABC) -- Points in triangle ABC
variables (altitudes : set (ABC × ABC)) -- Set of altitudes
variables (area : ABC → ℝ) -- Function giving area of triangle

noncomputable def is_equilateral (triangle : set ABC) : Prop :=
  ∀ x y z, x ∈ triangle → y ∈ triangle → z ∈ triangle → dist x y = dist y z

noncomputable def altitudes_of_triangle (triangle : set ABC) : Prop :=
  { (A, A1), (B, B1), (C, C1) } ⊆ altitudes

noncomputable def area_of_triangle (triangle : set ABC) (t : ℝ) : Prop :=
  area triangle = t

theorem equilateral_iff_area_relation (triangle : set ABC) (t : ℝ) :
  altitudes_of_triangle triangle →
  area_of_triangle triangle t →
  (dist A A1 * dist A B + dist B B1 * dist B C + dist C C1 * dist C A = 6 * t ↔ is_equilateral triangle) :=
by
  intro h1 h2
  sorry
end

end equilateral_iff_area_relation_l598_598488


namespace find_eccentricity_l598_598702

def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

def eccentricity (a c : ℝ) : ℝ := c / a

theorem find_eccentricity (a b c : ℝ) (F1 F2 O P : ℝ × ℝ) 
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (F1_eq : F1 = (-c, 0))
  (F2_eq : F2 = (c, 0))
  (O_eq : O = (0, 0))
  (h_on_hyperbola : ∃ x y, (hyperbola x y a b) ∧ (P = (x, y)) ∧ (0 < x))
  (h_OR_perpendicular : (P.1 - F2.1) = 0 )
  (h_distance : dist F1 F2 = dist P F2) :
  eccentricity a c = 1 + Real.sqrt 2 := 
by 
  sorry

end find_eccentricity_l598_598702


namespace length_of_tunnel_l598_598518

/-- A basic definition representing the speed of the train in miles per minute. -/
def train_speed : ℝ := 1 

/-- A basic definition representing the length of the train in miles. -/
def train_length : ℝ := 1

/-- A basic definition representing the time in minutes for the train to exit the tunnel after the front entered. -/
def exit_time : ℝ := 4

/-- A basic definition representing the distance traveled by the train in miles in exit_time minutes. -/
def distance_travelled : ℝ := train_speed * exit_time

/-- The length of the tunnel is the total distance travelled minus the length of the train. -/
theorem length_of_tunnel : distance_travelled - train_length = 3 := 
by
  have h : distance_travelled = 4 := rfl
  have h2 : train_length = 1 := rfl
  calc
    distance_travelled - train_length = 4 - 1 : by rw [h, h2]
    ... = 3 : by norm_num

end length_of_tunnel_l598_598518


namespace mean_rest_scores_l598_598171

theorem mean_rest_scores (n : ℕ) (h : 15 < n) 
  (overall_mean : ℝ := 10)
  (mean_of_fifteen : ℝ := 12)
  (total_score : ℝ := n * overall_mean): 
  (180 + p * (n - 15) = total_score) →
  p = (10 * n - 180) / (n - 15) :=
sorry

end mean_rest_scores_l598_598171


namespace find_ordered_pair_l598_598013

theorem find_ordered_pair (a b : ℝ) 
  (h1 : (a + 3 * complex.i) + (b + 7 * complex.i) = 10 + 10 * complex.i)
  (h2 : (a + 3 * complex.i) * (b + 7 * complex.i) = 70 + 16 * complex.i) :
  (a, b) = (-3.5, 13.5) :=
by 
  sorry

end find_ordered_pair_l598_598013


namespace number_of_integers_with_square_fraction_l598_598278

theorem number_of_integers_with_square_fraction : 
  ∃! (S : Finset ℤ), (∀ (n : ℤ), n ∈ S ↔ ∃ (k : ℤ), (n = 15 * k^2) ∨ (15 - n = k^2)) ∧ S.card = 2 := 
sorry

end number_of_integers_with_square_fraction_l598_598278


namespace exists_polynomial_divisibility_l598_598308

theorem exists_polynomial_divisibility (a b : ℕ) (hb : b > 1) (h : a ≥ 2 * b) :
  ∃ P : polynomial ℕ, degree P > 0 ∧ (∀ x ∈ P.coeffs, x < b) ∧ ((P.eval a) % (P.eval b) = 0) :=
sorry

end exists_polynomial_divisibility_l598_598308


namespace cone_vertex_angle_l598_598265

theorem cone_vertex_angle (P A B C : Type) [IsCone P A B C] 
  (h_perpendicular : ∀ x y z, IsGeneratrix x y z →
    (Perp x y ∧ Perp y z ∧ Perp z x)) :
  vertex_angle P A B C = 2 * real.arcsin (real.sqrt 6 / 3) :=
sorry

end cone_vertex_angle_l598_598265


namespace maria_fourth_test_score_l598_598024

theorem maria_fourth_test_score :
  ∀ (x : ℕ), (80 + 70 + 90 + x) / 4 = 85 → x = 100 :=
by
  intros x h,
  have h1 : 4 * 85 = 340 := rfl,
  have h2 : 80 + 70 + 90 = 240 := rfl,
  have h3 : 340 - 240 = 100 := rfl,
  sorry

end maria_fourth_test_score_l598_598024


namespace projection_of_perpendicular_vectors_is_zero_l598_598737

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (3, 4)
def vec_b : ℝ × ℝ := (8, -6)

-- Define the condition that vectors a and b are perpendicular
def are_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Define the dot product of two vectors
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

-- Define the projection formula
def projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_ab := dot_product a b in
  let mag_b2 := (magnitude b)^2 in
  (dot_ab / mag_b2) * b.1, (dot_ab / mag_b2) * b.2

-- State the problem in Lean 4
theorem projection_of_perpendicular_vectors_is_zero :
  are_perpendicular vec_a vec_b →
  projection vec_a vec_b = (0, 0) :=
by
  sorry

end projection_of_perpendicular_vectors_is_zero_l598_598737


namespace find_f_minus_1_l598_598721

noncomputable def piecewiseFunction (x : ℝ) (a b : ℝ) : ℝ :=
if x >= 0 then sqrt x + 3 else a * x + b

def conditions (a b : ℝ) :=
a > 0 ∧ b ≤ 3 ∧ sqrt a + 3 = 4 ∧ a * b + b = -4

theorem find_f_minus_1 (a b : ℝ) (h : conditions a b) : 
  piecewiseFunction (-1) a b = -3 := 
sorry

end find_f_minus_1_l598_598721


namespace arithmetic_sequence_x_l598_598273

theorem arithmetic_sequence_x (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ d : ℝ, fractional_part x + d = floor x ∧ floor x + d = x ∧ x + d = x + fractional_part x) : x = 1.5 :=
sorry

end arithmetic_sequence_x_l598_598273


namespace cos_angle_ACB_l598_598981

-- Definitions of given conditions
variables {α : Type*} [metric_space α] [normed_space ℝ α]

-- AB is the diameter of a circle
def is_diameter (A B : α) (c : α) : Prop :=
∀ (P : α), dist P A = dist P B

-- C is a point not on the line AB
def not_on_line (A B C : α) : Prop :=
¬ collinear ℝ ({A, B, C} : set α)

-- The line AC intersects the circle again at X
def intersects_again (A C X : α) : Prop :=
∀ (P : α), P ∈ line ℝ A C → C ≠ P → P = X

-- The line BC intersects the circle again at Y
def intersects_again_BC (B C Y : α) : Prop :=
∀ (P : α), P ∈ line ℝ B C → C ≠ P → P = Y

-- Main theorem
theorem cos_angle_ACB {A B C X Y : α}
  (h1 : is_diameter A B (circle_center))
  (h2 : not_on_line A B C)
  (h3 : intersects_again A C X)
  (h4 : intersects_again_BC B C Y)
  : cos (angle A C B) ^ 2 = (dist C X / dist C A) * (dist C Y / dist C B) :=
sorry

end cos_angle_ACB_l598_598981


namespace correct_operation_l598_598576

theorem correct_operation : 
  ¬(3 * x^2 + 2 * x^2 = 6 * x^4) ∧ 
  ¬((-2 * x^2)^3 = -6 * x^6) ∧ 
  ¬(x^3 * x^2 = x^6) ∧ 
  (-6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y) :=
by
  sorry

end correct_operation_l598_598576


namespace find_circle_center_radius_l598_598665

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y - 4 = 0

theorem find_circle_center_radius :
  (∃ (a b r : ℝ), ∀ x y : ℝ, circle_equation x y ↔ (x + 1) ^ 2 + (y - 2) ^ 2 = r ^ 2) :=
by {
  use [-1, 2, 3],
  intros x y,
  split; intro h,
  {
    sorry
  },
  {
    sorry
  }
}

end find_circle_center_radius_l598_598665


namespace concurrency_of_lines_l598_598305

open set real

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, -1)
def line_l1 (x y : ℝ) : Prop := x = -2
def line_l2 (x y : ℝ) : Prop := y = -1
def point_P (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ ellipse x y
def line_l3 (x y : ℝ) (x0 y0 : ℝ) (h : point_P x0 y0) : Prop := ∃ m b : ℝ, y = m * x + b ∧ y0 = m * x0 + b ∧ ∀ x' y' : ℝ, (y' = m * x' + b → ¬ ellipse x' y')
def point_C (x y : ℝ) : Prop := line_l1 x y ∧ line_l2 x y
def point_D (x y : ℝ) (x0 y0 : ℝ) (h : point_P x0 y0) : Prop := line_l2 x y ∧ line_l3 x y x0 y0 h
def point_E (x y : ℝ) (x0 y0 : ℝ) (h : point_P x0 y0) : Prop := line_l1 x y ∧ line_l3 x y x0 y0 h
def line_AD (x y : ℝ) (x0 y0 : ℝ) (h : point_P x0 y0) : Prop := ∃ m b : ℝ, y = m * x + b ∧ y = m * x + b ∧ y0 = m * (-2) + b
def line_BE (x y : ℝ) (x0 y0 : ℝ) (h : point_P x0 y0) : Prop := ∃ m b : ℝ, y = m * x + b ∧ y = m * x + b ∧ y = m * 0 + b
def line_CP (x y : ℝ) (x0 y0 : ℝ) (h : point_P x0 y0) : Prop := ∃ m b : ℝ, y = m * x + b ∧ y = m * x + b ∧ y = m * x0 + b

theorem concurrency_of_lines (x y x0 y0 : ℝ) (hP : point_P x0 y0)
    (C : point_C x y) (D : point_D x y x0 y0 hP) (E : point_E x y x0 y0 hP) :
    ∃ (d : ℝ × ℝ), line_AD d.1 d.2 x0 y0 hP ∧ line_BE d.1 d.2 x0 y0 hP ∧ line_CP d.1 d.2 x0 y0 hP :=
sorry

end concurrency_of_lines_l598_598305


namespace final_notebooks_l598_598625

def initial_notebooks : ℕ := 10
def ordered_notebooks : ℕ := 6
def lost_notebooks : ℕ := 2

theorem final_notebooks : initial_notebooks + ordered_notebooks - lost_notebooks = 14 :=
by
  sorry

end final_notebooks_l598_598625


namespace calculate_possible_values_of_g_l598_598815

def y_k (k : ℕ) : ℤ :=
  if (k % 2 = 1) then 1 else -1

def T_n (n : ℕ) : ℤ :=
  (Finset.range (n)).sum (λ k => y_k (k + 1))

def g (n : ℕ) : ℚ :=
  T_n n / n

theorem calculate_possible_values_of_g (n : ℕ) : 
  { g n | n > 0 } = if n % 2 = 0 then {0} else {0, 1 / n} :=
by
  sorry

end calculate_possible_values_of_g_l598_598815


namespace find_CQ_over_AD_find_length_AD_l598_598781

-- Define the parallelogram ABCD with given conditions
structure parallelogram (A B C D E P F Q : Type) :=
(angle_BAD_is_acute : ∀ (A B D : Type), angle A B D < 90)
(AD_lt_AB : ∀ (A D B : Type), length A D < length A B)
(bisector_intersects_CD : ∀ (A B D E C : Type), bisector (angle A B D) intersects C D at E)
(perpendicular_from_D_to_AE : ∀ (D A E P B F : Type), perpendicular (D to A E) intersects A E at P and A B at F)
(perpendicular_from_E_to_AE : ∀ (E A E Q B C : Type), perpendicular (E to A E) intersects B C at Q)
(PQ_parallel_AB : ∀ (P Q A B : Type), parallel P Q to A B)
(length_AB : ∀ (A B : Type), length A B = 20)

-- Define the mathematical equivalencies to be proven
theorem find_CQ_over_AD {A B C D E P F Q : Type} (h : parallelogram A B C D E P F Q) : 
  (length C Q / length A D) = 1 / 2 := 
sorry

theorem find_length_AD {A B C D E P F Q : Type} (h : parallelogram A B C D E P F Q) : 
  length A D = 40 / 3 :=
sorry

end find_CQ_over_AD_find_length_AD_l598_598781


namespace circular_fib_correct_l598_598022

def circular_fib (n : ℕ) : list ℕ :=
  ((λ f : ℕ → ℕ, fix f) (λ g n, if n = 0 then 0 else if n = 1 then 1 else g (n - 1) + g (n - 2))) '' (list.range n)

noncomputable def correct_fib : ℕ → list ℕ
| 2 := [0, 1]
| 3 := [0, 1, 1]
| 4 := [0, 1, 1, 2]
| 5 := [0, 1, 1, 2, 3]
| 6 := [0, 1, 1, 2, 3, 5]
| 7 := [0, 1, 1, 2, 3, 5, 8]
| 8 := [0, 1, 1, 2, 3, 5, 8, 13]
| 9 := [0, 1, 1, 2, 3, 5, 8, 13, 21]
| 10 := [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
| 11 := [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
| _ := []

theorem circular_fib_correct (m : ℕ) (h : m ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}):
  circular_fib m = correct_fib m :=
sorry

end circular_fib_correct_l598_598022


namespace bead_bracelet_problem_l598_598384

-- Define the condition Bead A and Bead B are always next to each other
def adjacent (A B : ℕ) (l : List ℕ) : Prop :=
  ∃ (l1 l2 : List ℕ), l = l1 ++ A :: B :: l2 ∨ l = l1 ++ B :: A :: l2

-- Define the context and translate the problem
def bracelet_arrangements (n : ℕ) : ℕ :=
  if n = 8 then 720 else 0

theorem bead_bracelet_problem : bracelet_arrangements 8 = 720 :=
by {
  -- Place proof here
  sorry 
}

end bead_bracelet_problem_l598_598384


namespace petya_always_wins_if_odd_l598_598100

noncomputable def petya_wins_if_odd (n: ℕ) (h: n ≥ 5) : Prop :=
  n % 2 = 1

theorem petya_always_wins_if_odd (n: ℕ) (h: n ≥ 5) :
  petya_wins_if_odd n h :=
begin
  sorry
end

end petya_always_wins_if_odd_l598_598100


namespace oliver_earnings_l598_598033

-- Define the conditions
def cost_per_kilo : ℝ := 2
def kilos_two_days_ago : ℝ := 5
def kilos_yesterday : ℝ := kilos_two_days_ago + 5
def kilos_today : ℝ := 2 * kilos_yesterday

-- Calculate the total kilos washed over the three days
def total_kilos : ℝ := kilos_two_days_ago + kilos_yesterday + kilos_today

-- Calculate the earnings over the three days
def earnings : ℝ := total_kilos * cost_per_kilo

-- The theorem we want to prove
theorem oliver_earnings : earnings = 70 := by
  sorry

end oliver_earnings_l598_598033


namespace parabola_equation_l598_598180

theorem parabola_equation (x y : ℝ) 
  (h_vertex : ∀ x y, (x, y) = (0, 0) → y^2 = 2 * p * x)
  (h_center_C : ∀ x y, (x - 1)^2 + (y + √2)^2 = 0 → (x, y) = (1, -√2)) 
  (h_axis_perp : ∀ y x, (x=0) ∧ (y≠0) → True)
  (point_on_parabola : (1, -√2)): 
  y^2 = 2*x := sorry

end parabola_equation_l598_598180


namespace perfect_square_digit_sum_l598_598586

-- Definition of the digits conditions
def digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

-- The problem setup
theorem perfect_square_digit_sum :
  ∃ A B : ℕ, digit A ∧ digit B ∧ (∃ n : ℕ, 15 * 10000 + A * 1000 + B * 100 + 9 = n * n) ∧ (A + B = 3) :=
begin
  use 1,
  use 2,
  split,
  { rw digit, exact ⟨nat.zero_le _, le_refl _⟩ },
  split,
  { rw digit, exact ⟨nat.zero_le _, le_refl _⟩ },
  split,
  { use 123,
    ring_nf,
    norm_num },
  { norm_num }
end

end perfect_square_digit_sum_l598_598586


namespace correct_statements_l598_598293

theorem correct_statements (f : ℝ → ℝ)
  (h_add : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
  (h_pos : ∀ x : ℝ, x > 0 → f (x) > 0) :
  (f 0 ≠ 1) ∧
  (∀ x : ℝ, f (-x) = -f (x)) ∧
  ¬ (∀ x : ℝ, |f (x)| = |f (-x)|) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f (x₁) < f (x₂)) ∧
  ¬ (∀ x : ℝ, f (x) + 1 < f (x + 1)) :=
by
  sorry

end correct_statements_l598_598293


namespace sum_of_real_solutions_l598_598272

noncomputable def equation (x : ℝ) : Prop := 
  Real.sqrt x + Real.sqrt (9 / x) + 2 * Real.sqrt (x + 9 / x) = 8

theorem sum_of_real_solutions : 
  ∑ x in {x : ℝ | equation x}, x = 40.96 := 
sorry

end sum_of_real_solutions_l598_598272


namespace point_in_third_quadrant_l598_598889

-- Define the complex number.z
def z : ℂ := -1 / 2 + (complex.I) * (real.sqrt 3 / 2)

-- State the proposition to prove
theorem point_in_third_quadrant : (z^2).re < 0 ∧ (z^2).im < 0 :=
sorry

end point_in_third_quadrant_l598_598889


namespace smallest_n_terminating_decimal_contains_9_distinct_l598_598123

noncomputable def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ n.digits 10

noncomputable def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  ∃ x ∈ digits, ∃ y ∈ digits, x ≠ y

theorem smallest_n_terminating_decimal_contains_9_distinct :
  ∃ n : ℕ, 
    (∃ a b : ℕ, n = 2^a * 5^b) ∧ 
    contains_digit n 9 ∧ 
    has_distinct_digits n ∧ 
    ∀ m : ℕ, 
      (∃ a b : ℕ, m = 2^a * 5^b) ∧ 
      contains_digit m 9 ∧ 
      has_distinct_digits m → 
      n ≤ m :=
  sorry

end smallest_n_terminating_decimal_contains_9_distinct_l598_598123


namespace cereal_sugar_ratio_l598_598236

variables (A B C : ℕ)
variables (sugarA sugarB sugarC : ℚ)

-- Assume the sugar content percentages as conditions
def sugar_content_A: sugarA = 12 / 100
def sugar_content_B: sugarB = 3 / 100
def sugar_content_C: sugarC = 7 / 100

-- Assume the ratio to be proved is 2:13:13
def ratio_A: ℚ := 2 / (2 + 13 + 13)
def ratio_B: ℚ := 13 / (2 + 13 + 13)
def ratio_C: ℚ := 13 / (2 + 13 + 13)

theorem cereal_sugar_ratio (A B C : ℕ)
  (h1 : sugarA = 12 / 100)
  (h2 : sugarB = 3 / 100)
  (h3 : sugarC = 7 / 100)
  (h4 : ratio_A * A + ratio_B * B + ratio_C * C = A + B + C)
  (h5 : sugarA * A + sugarB * B + sugarC * C = 5.5 / 100 * (A + B + C)) :
  (A : ℚ) / B = 2 / 13 ∧ (C : ℚ) / B = 1 :=
sorry 

end cereal_sugar_ratio_l598_598236


namespace sum_of_interior_angles_l598_598879

theorem sum_of_interior_angles (n : ℕ) (h : 180 * (n - 2) = 1800) : 180 * ((n - 3) - 2) = 1260 :=
by
  sorry

end sum_of_interior_angles_l598_598879


namespace simplify_fraction_l598_598067

theorem simplify_fraction (n : ℕ) : 
  (3 ^ (n + 3) - 3 * (3 ^ n)) / (3 * 3 ^ (n + 2)) = 8 / 9 :=
by sorry

end simplify_fraction_l598_598067


namespace oliver_earning_correct_l598_598030

open Real

noncomputable def total_weight_two_days_ago : ℝ := 5

noncomputable def total_weight_yesterday : ℝ := total_weight_two_days_ago + 5

noncomputable def total_weight_today : ℝ := 2 * total_weight_yesterday

noncomputable def total_weight_three_days : ℝ := total_weight_two_days_ago + total_weight_yesterday + total_weight_today

noncomputable def earning_per_kilo : ℝ := 2

noncomputable def total_earning : ℝ := total_weight_three_days * earning_per_kilo

theorem oliver_earning_correct : total_earning = 70 := by
  sorry

end oliver_earning_correct_l598_598030


namespace maximum_value_condition_abc_sum_l598_598075

noncomputable def find_maximum_value (x y : ℝ) (h_x_pos : 0 < x) (h_y_pos : 0 < y) (h_constraint : x^2 - x * y + y^2 = 8) :
  ℝ :=
max (x^2 + x * y + y^2)

theorem maximum_value_condition : ∃ x y : ℝ, 
  0 < x ∧ 
  0 < y ∧ 
  x^2 - x * y + y^2 = 8 ∧ 
  find_maximum_value x y 0_lt_one 0_lt_one sorry = 24 :=
sorry

theorem abc_sum : 
  let a := 24
  let b := 0
  let c := 1
  let d := 1 in
  a + b + c + d = 26 :=
by decide

end maximum_value_condition_abc_sum_l598_598075


namespace exists_polynomial_h_l598_598804

variable {R : Type} [CommRing R] [IsDomain R] [CharZero R]

noncomputable def f (x : R) : ℝ := sorry -- define the polynomial f(x) here
noncomputable def g (x : R) : ℝ := sorry -- define the polynomial g(x) here

theorem exists_polynomial_h (m n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) (h_mn : m + n > 0)
  (h_fg_squares : ∀ x : ℝ, (∃ k : ℤ, f x = k^2) ↔ (∃ l : ℤ, g x = l^2)) :
  ∃ h : ℝ → ℝ, ∀ x : ℝ, f x * g x = (h x)^2 :=
sorry

end exists_polynomial_h_l598_598804


namespace problem1_problem2_problem3_problem4_l598_598574

theorem problem1 (h : Real.cos 75 * Real.sin 75 = 1 / 2) : False :=
by
  sorry

theorem problem2 : (1 + Real.tan 15) / (1 - Real.tan 15) = Real.sqrt 3 :=
by
  sorry

theorem problem3 : Real.tan 20 + Real.tan 25 + Real.tan 20 * Real.tan 25 = 1 :=
by
  sorry

theorem problem4 (θ : Real) (h1 : Real.sin (2 * θ) ≠ 0) : (1 / Real.tan θ - 1 / Real.tan (2 * θ) = 1 / Real.sin (2 * θ)) :=
by
  sorry

end problem1_problem2_problem3_problem4_l598_598574


namespace obtuse_triangle_from_conditions_l598_598395

variable {A B C a b c : ℝ}
variable {sin cos tan : ℝ → ℝ}
-- Define the dot product
def dot_product (x y : ℝ) : ℝ := x * y

-- Assume the given conditions
variable (h1 : dot_product (A - B) (B - C) < 0)
variable (h2 : (a - b) / (c + b) = (sin C) / (sin A + sin B))
variable (h3 : tan A + tan B + tan C < 0)

-- Define what it means for a triangle to be obtuse
def is_obtuse_triangle : Prop :=
  A > π / 2 ∨ B > π / 2 ∨ C > π / 2

-- Problem statement to be proved
theorem obtuse_triangle_from_conditions :
  is_obtuse_triangle :=
by
  sorry

end obtuse_triangle_from_conditions_l598_598395


namespace b_capital_amount_l598_598932

noncomputable theory

-- Define the conditions
def a_capital := 15000
def total_profit := 9600
def managing_fee := 0.10 * total_profit  -- 10% of profit
def a_total_received := 4200
def remaining_profit := total_profit - managing_fee
def a_profit_share := a_total_received - managing_fee

-- The formalization of the problem and proof statement:
theorem b_capital_amount (x : ℝ) :
  (15000 / x) = (a_profit_share / (remaining_profit - a_profit_share)) → x = 25000 :=
by
  -- Definitions to match the conditions
  let a_capital := 15000
  let total_profit := 9600
  let managing_fee := 0.10 * total_profit  -- 10% of profit
  let a_total_received := 4200
  let remaining_profit := total_profit - managing_fee
  let a_profit_share := a_total_received - managing_fee
  have h₁ : managing_fee = 960 := sorry  -- from condition 1
  have h₂ : remaining_profit = 8640 := sorry  -- from condition 2
  have h₃ : a_profit_share = 3240 := sorry  -- from condition 3
  sorry  -- Proof follows here

end b_capital_amount_l598_598932


namespace raft_minimum_capacity_l598_598207

theorem raft_minimum_capacity (n_mice n_moles n_hamsters : ℕ)
  (weight_mice weight_moles weight_hamsters : ℕ)
  (total_weight : ℕ) :
  n_mice = 5 →
  weight_mice = 70 →
  n_moles = 3 →
  weight_moles = 90 →
  n_hamsters = 4 →
  weight_hamsters = 120 →
  (∀ (total_weight : ℕ), total_weight = n_mice * weight_mice + n_moles * weight_moles + n_hamsters * weight_hamsters) →
  (∃ (min_capacity: ℕ), min_capacity ≥ 140) :=
by
  intros
  sorry

end raft_minimum_capacity_l598_598207


namespace original_num_cookies_l598_598006

-- Definitions
variables {C' : ℕ} --number of cookies left
variables {B' : ℕ} --number of brownies left
variables {B : ℕ} -- number of brownies originally baked

-- Conditions
def original_num_brownies_equal_32 : Prop := B = 32
def total_money_from_sales (C' B' : ℕ) : Prop := C' + 1.5 * B' = 99

-- Proof problem
theorem original_num_cookies (assume_no_cookies_eaten : C = C') 
  (original_num_brownies_equal_32 : original_num_brownies_equal_32) 
  (total_money_from_sales_C_B : total_money_from_sales C' 32):
  C = 51 :=
begin
  sorry  -- Proof to be provided
end

end original_num_cookies_l598_598006


namespace abs_neg_three_halves_l598_598475

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end abs_neg_three_halves_l598_598475


namespace speed_of_goods_train_l598_598960

/-- Define the given conditions -/
def man's_train_speed := 45 -- speed in km/h
def goods_train_length := 340 -- length in meters
def time_to_pass := 8 -- time in seconds

/-- Define the formula for relative speed in m/s and its conversion to km/h -/
def relative_speed_mps := goods_train_length / time_to_pass -- speed in m/s
def relative_speed_kmph := relative_speed_mps * 3.6 -- conversion factor from m/s to km/h

/-- Define the statement to prove the speed of the goods train -/
theorem speed_of_goods_train : ∀ (V_g : ℝ), V_g = 108 ↔ relative_speed_kmph = man's_train_speed + V_g :=
by
  sorry

end speed_of_goods_train_l598_598960


namespace largest_angle_consecutive_even_pentagon_l598_598096

theorem largest_angle_consecutive_even_pentagon :
  ∀ (n : ℕ), (2 * n + (2 * n + 2) + (2 * n + 4) + (2 * n + 6) + (2 * n + 8) = 540) →
  (2 * n + 8 = 112) :=
by
  intros n h
  sorry

end largest_angle_consecutive_even_pentagon_l598_598096


namespace solve_inequality_l598_598069

theorem solve_inequality (x : ℝ) (n : ℤ) :
  x > 2 / 3 ∧ x ≠ 1 →
  (log x (3 * x - 2))^2 - 4 * (sin (π * x) - 1) ≤ 0 ↔
  (x ∈ Set.Ioo (2 / 3 : ℝ) 1) ∨ (x ∈ Set.Ioo 1 2) ∨ (∃ n : ℤ, x = 2 * n + 0.5) :=
sorry

end solve_inequality_l598_598069


namespace ceil_minus_eq_zero_l598_598556

theorem ceil_minus_eq_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ - x = 0 :=
sorry

end ceil_minus_eq_zero_l598_598556


namespace socks_pairs_different_colors_l598_598354

theorem socks_pairs_different_colors :
  let white := 5
      brown := 5
      blue := 4
      red := 2
  in
  (white * brown) + (white * blue) + (white * red) + (brown * blue) + (brown * red) + (blue * red) = 93 := by
  sorry

end socks_pairs_different_colors_l598_598354


namespace area_enclosed_by_graph_l598_598912

-- We define the given condition as a predicate on real numbers.
def is_condition (x y : ℝ) : Prop := abs (3 * x) + abs (4 * y) = 12

-- We state the theorem to prove the area enclosed by the graph is 24.
theorem area_enclosed_by_graph : (area (set_of (λ (p : ℝ × ℝ), is_condition p.1 p.2))) = 24 :=
sorry

end area_enclosed_by_graph_l598_598912


namespace average_sales_l598_598233

theorem average_sales (jan feb mar apr : ℝ) (h_jan : jan = 100) (h_feb : feb = 60) (h_mar : mar = 40) (h_apr : apr = 120) : 
  (jan + feb + mar + apr) / 4 = 80 :=
by {
  sorry
}

end average_sales_l598_598233


namespace parabola_vertex_coordinates_l598_598496

theorem parabola_vertex_coordinates (h k : ℝ) : 
  (∀ x : ℝ, y = -3 * (x - 1)^2 + 4) → (h = 1 ∧ k = 4) :=
begin
  sorry
end

end parabola_vertex_coordinates_l598_598496


namespace ratio_of_areas_l598_598538

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l598_598538


namespace root_bound_l598_598805

theorem root_bound (n : ℕ) (a : ℕ → ℝ)
  (h_n : 2 ≤ n)
  (h_real_coeffs : ∀ i, a i ∈ ℝ)
  (h_real_roots : ∀ r, is_root ((λ x, x^n + a (n-2) * x^(n-2) + a (n-3) * x^(n-3) + ... + a 1 * x + a 0) r) → r ∈ ℝ) :
  ∀ r, is_root ((λ x, x^n + a (n-2) * x^(n-2) + a (n-3) * x^(n-3) + ... + a 1 * x + a 0) r) → |r| ≤ sqrt (2 * (1 - n) / n * a (n-2)) :=
  sorry

end root_bound_l598_598805


namespace parallel_trans_l598_598849

variables {l1 l2 l3 : Type} [parallel_relation : ParallelRelation Type]

-- Defining parallel relation
def parallel (x y : Type) [ParallelRelation Type] : Prop := sorry

-- Hypothesis
hypothesis (h1 : parallel l1 l2)
hypothesis (h2 : parallel l2 l3)

-- Goal
theorem parallel_trans (h1 : parallel l1 l2) (h2 : parallel l2 l3) : parallel l1 l3 := by
  sorry

end parallel_trans_l598_598849


namespace wrongly_read_number_l598_598877

theorem wrongly_read_number (initial_avg correct_avg n wrong_correct_sum : ℝ) : 
  initial_avg = 23 ∧ correct_avg = 24 ∧ n = 10 ∧ wrong_correct_sum = 36
  → ∃ (X : ℝ), 36 - X = 10 ∧ X = 26 :=
by
  intro h
  sorry

end wrongly_read_number_l598_598877


namespace find_n_l598_598667

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -2345 [MOD 13] ∧ n = 8 :=
by
  sorry

end find_n_l598_598667


namespace quiz_scores_dropped_students_l598_598492

theorem quiz_scores_dropped_students (T S : ℝ) :
  T = 30 * 60.25 →
  T - S = 26 * 63.75 →
  S = 150 :=
by
  intros hT h_rem
  -- Additional steps would be implemented here.
  sorry

end quiz_scores_dropped_students_l598_598492


namespace polygon_parallel_edges_l598_598298

theorem polygon_parallel_edges (n : ℕ) (h : n > 2) :
  (∃ i j, i ≠ j ∧ (i + 1) % n = (j + 1) % n) ↔ (∃ k, n = 2 * k) :=
  sorry

end polygon_parallel_edges_l598_598298


namespace farm_owns_more_horses_l598_598451

noncomputable def farm_transaction (x : ℕ) :=
  let initial_horses := 4 * x
  let initial_cows := x
  let sheep := 3 * initial_horses / 2
  let horses_after_transaction := initial_horses - 15
  let cows_after_transaction := initial_cows + 30
  let ratio_condition := (horses_after_transaction * 3 = cows_after_transaction * 7)
  let total_money := 4500
  let earn := 15 * 300
  let spend := 30 * 150
  let budget_condition := (earn = total_money) ∧ (spend = total_money)
  if ratio_condition && budget_condition then
    horses_after_transaction - cows_after_transaction
  else sorry

theorem farm_owns_more_horses (x : ℕ) (h : x = 51) : farm_transaction x = 108 :=
by simp [farm_transaction, h]; sorry

end farm_owns_more_horses_l598_598451


namespace unique_seating_and_new_neighbors_l598_598607

def alternating_sum (i : ℕ) : ℕ :=
  (List.range (i + 1)).map (λ j => (-1)^(j + 1) * j).sum

theorem unique_seating_and_new_neighbors :
  ∀ (k : ℕ) (h₀ : k < 60) (i : ℕ) (h₁ : i < 60),
  let seat := λ k i => (k + alternating_sum i) % 60 in
  -- Ensure no vacationer sits in the same place twice
  (∀ m n, (m < 60) → (n < 4 * 15) → seat m n ≠ seat m (n + 1)) ∧
  -- Ensure each has a new neighbor to their right
  (∀ m n p, (m < 60) → (n < 4 * 15) → (p < 60) → seat m n ≠ seat p n) :=
by
  sorry

end unique_seating_and_new_neighbors_l598_598607


namespace bellas_score_l598_598838

-- Definitions from the problem conditions
def n : Nat := 17
def x : Nat := 75
def new_n : Nat := n + 1
def y : Nat := 76

-- Assertion that Bella's score is 93
theorem bellas_score : (new_n * y) - (n * x) = 93 :=
by
  -- This is where the proof would go
  sorry

end bellas_score_l598_598838


namespace fish_weight_l598_598573

variables (W G T : ℕ)

-- Define the known conditions
axiom tail_weight : W = 1
axiom head_weight : G = W + T / 2
axiom torso_weight : T = G + W

-- Define the proof statement
theorem fish_weight : W + G + T = 8 :=
by
  sorry

end fish_weight_l598_598573


namespace min_capacity_for_raft_l598_598211

-- Define the weights of the animals
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal
def number_mice : ℕ := 5
def number_moles : ℕ := 3
def number_hamsters : ℕ := 4

-- Define the minimum weight capacity for the raft
def min_weight_capacity : ℕ := 140

-- Prove that the minimum weight capacity the raft must have to transport all animals is 140 grams.
theorem min_capacity_for_raft :
  (weight_mouse * 2 ≤ min_weight_capacity) ∧ 
  (∀ trip_weight, trip_weight ≥ min_weight_capacity → 
    (trip_weight = weight_mouse * 2 ∨ trip_weight = weight_mole * 2 ∨ trip_weight = weight_hamster * 2)) :=
by 
  sorry

end min_capacity_for_raft_l598_598211


namespace incenters_form_rectangle_l598_598458

-- Definitions for geometric objects and properties
variables {A B C D : Point}
variables {I_A I_B I_C I_D : Point}
variable circ : Circle

-- Assumptions about the geometric configuration
axiom ABCD_inscribed : circ.inscribedQuadrilateral A B C D
axiom incenter_BCD : incenter_of_triangle I_A B C D
axiom incenter_CDA : incenter_of_triangle I_B C D A
axiom incenter_DAB : incenter_of_triangle I_C D A B
axiom incenter_ABC : incenter_of_triangle I_D A B C

-- Goal statement: Prove that the quadrilateral formed by the incenters is a rectangle
theorem incenters_form_rectangle : is_rectangle I_A I_B I_C I_D :=
sorry

end incenters_form_rectangle_l598_598458


namespace consecutive_odds_coprime_l598_598934

theorem consecutive_odds_coprime (a : ℤ) : Nat.coprime a (a + 2) :=
sorry

end consecutive_odds_coprime_l598_598934


namespace nth_74th_number_l598_598827

namespace PositiveIntegersMod8Set

def is_in_set_s (x : ℕ) : Prop := ∃ k : ℕ, x = 8 * k + 5

def nth_number_in_set_s {n : ℕ} (nth_num : ℕ) : Prop :=
  nat.find (λ m, is_in_set_s m) n = nth_num

theorem nth_74th_number :
  nth_number_in_set_s 597 74 :=
sorry

end PositiveIntegersMod8Set

end nth_74th_number_l598_598827


namespace marbles_leftover_l598_598571

theorem marbles_leftover (r p : ℤ) (hr : r % 8 = 5) (hp : p % 8 = 6) : (r + p) % 8 = 3 := by
  sorry

end marbles_leftover_l598_598571


namespace decomposition_of_x_l598_598579

-- Definitions derived from the conditions
def x : ℝ × ℝ × ℝ := (11, 5, -3)
def p : ℝ × ℝ × ℝ := (1, 0, 2)
def q : ℝ × ℝ × ℝ := (-1, 0, 1)
def r : ℝ × ℝ × ℝ := (2, 5, -3)

-- Theorem statement proving the decomposition
theorem decomposition_of_x : x = (3 : ℝ) • p + (-6 : ℝ) • q + (1 : ℝ) • r := by
  sorry

end decomposition_of_x_l598_598579


namespace sin_cos_cubic_l598_598399

theorem sin_cos_cubic (α n : ℝ) (h : sin α - cos α = n) : 
  sin α ^ 3 - cos α ^ 3 = (3 * n - n ^ 3) / 2 :=
by
  sorry

end sin_cos_cubic_l598_598399


namespace count_perfect_squares_ending_4_5_6_l598_598347

theorem count_perfect_squares_ending_4_5_6 : 
  ∃ n, n = 36 ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 70 ∧ (let d := k % 10 in d = 2 ∨ d = 8 ∨ d = 5 ∨ d = 4 ∨ d = 6) → k^2 < 5000) := 
sorry

end count_perfect_squares_ending_4_5_6_l598_598347


namespace setA_not_right_triangle_l598_598577

-- Define the sets of numbers
def setA := (Real.sqrt 3, 2, Real.sqrt 5)
def setB := (3, 4, 5)
def setC := (0.6, 0.8, 1)
def setD := (130, 120, 50)

-- Define condition under which a set cannot form a right triangle 
def notRightTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 ≠ c^2

-- Prove that set A cannot be the lengths of the sides of a right triangle
theorem setA_not_right_triangle : notRightTriangle (Real.sqrt 3) 2 (Real.sqrt 5) :=
by sorry

end setA_not_right_triangle_l598_598577


namespace factor_polynomial_l598_598661

theorem factor_polynomial :
  ∀ u : ℝ, (u^4 - 81 * u^2 + 144) = (u^2 - 72) * (u - 3) * (u + 3) :=
by
  intro u
  -- Establish the polynomial and its factorization in Lean
  have h : u^4 - 81 * u^2 + 144 = (u^2 - 72) * (u - 3) * (u + 3) := sorry
  exact h

end factor_polynomial_l598_598661


namespace constant_term_in_expansion_l598_598917

theorem constant_term_in_expansion :
  let p := (x^3 + 2 * x + 7)
  let q := (2 * x^4 + 3 * x^2 + 10)
  (∀ (x : ℝ), (p * q).coeff 0 = 70) :=
by
  sorry

end constant_term_in_expansion_l598_598917


namespace remainder_zero_l598_598670

noncomputable def remainder_when_divided (f g : Polynomial ℤ) : Polynomial ℤ :=
f % g

theorem remainder_zero : 
  remainder_when_divided (X^2023 + X) (X^6 - X^4 + X^2 - 1) = 0 := 
by 
  sorry

end remainder_zero_l598_598670


namespace find_x_l598_598491

theorem find_x :
  let avg1 := (20 + 40 + 60) / 3 in
  let avg2 := (10 + 80 + x) / 3 in
  (avg1 = avg2 + 5) →
  x = 15 :=
by
  sorry

end find_x_l598_598491


namespace area_difference_8_7_area_difference_9_8_l598_598238

-- Define the side lengths of the tablets
def side_length_7 : ℕ := 7
def side_length_8 : ℕ := 8
def side_length_9 : ℕ := 9

-- Define the areas of the tablets
def area_7 := side_length_7 * side_length_7
def area_8 := side_length_8 * side_length_8
def area_9 := side_length_9 * side_length_9

-- Prove the differences in area
theorem area_difference_8_7 : area_8 - area_7 = 15 := by sorry
theorem area_difference_9_8 : area_9 - area_8 = 17 := by sorry

end area_difference_8_7_area_difference_9_8_l598_598238


namespace numSolutions_eq_dep_on_a_l598_598352

noncomputable def numberOfSolutions (a : ℝ) : ℕ :=
if 0 < a ∧ a < real.exp (-real.exp 1) then 3
else if 1 < a ∧ a < real.exp (1 / real.exp 1) then 2
else if a = real.exp (1 / real.exp 1) ∨ (real.exp (-real.exp 1) ≤ a ∧ a < 1) then 1
else if a > real.exp (1 / real.exp 1) then 0
else 0

theorem numSolutions_eq_dep_on_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  numberOfSolutions a = 
    if 0 < a ∧ a < real.exp (-real.exp 1) then 3
    else if 1 < a ∧ a < real.exp (1 / real.exp 1) then 2
    else if a = real.exp (1 / real.exp 1) ∨ (real.exp (-real.exp 1) ≤ a ∧ a < 1) then 1
    else if a > real.exp (1 / real.exp 1) then 0
    else 0 :=
sorry

end numSolutions_eq_dep_on_a_l598_598352


namespace inequality_proof_l598_598861

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (x / (y + z + 1)) + (y / (z + x + 1)) + (z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) :=
sorry

end inequality_proof_l598_598861


namespace two_five_digit_numbers_sum_condition_l598_598261

theorem two_five_digit_numbers_sum_condition :
  let digits := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] in
  let sum_to_99999 (x y : ℕ) := x + y = 99999 in
  let valid_five_digits (n : ℕ) := ∃ a b c d e, 
    n = a * 10^4 + b * 10^3 + c * 10^2 + d * 10 + e ∧
    a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits in
  let count_valid_pairs (f : ℕ → ℕ → Prop) (valid : ℕ → Prop) := 
    (univ.filter (λ p : ℕ × ℕ, f p.1 p.2 ∧ valid p.1 ∧ valid p.2)).card / 2 in
  count_valid_pairs sum_to_99999 valid_five_digits = 768 :=
sorry

end two_five_digit_numbers_sum_condition_l598_598261


namespace collinear_X_Y_Z_l598_598699

variable (P Q R S : Type)
variable [line_segment P Q] -- Assuming P Q represents points on line segment?

theorem collinear_X_Y_Z
  (ABCD : Type) -- Trapezoid
  (BC AD : Type) -- Bases
  (omega : Type) -- Circle
  (B C : Type) -- Points on the circle
  (AB XD : Type) -- Side and diagonal intersected by the circle
  (X Y : Type) -- Points of intersection with the circle
  (Z : Type) -- Tangent intersection point
  (H1 : isosceles_trapezoid ABCD)
  (H2 : passes_through B C omega)
  (H3 : intersects_at AB X omega)
  (H4 : intersects_at BD Y omega)
  (H5 : tangent_at C omega = Z AD)
  : collinear X Y Z := 
sorry

end collinear_X_Y_Z_l598_598699


namespace original_number_l598_598106

theorem original_number (x : ℤ) (h : x / 2 = 9) : x = 18 := by
  sorry

end original_number_l598_598106


namespace cos_pi_over_3_plus_2alpha_l598_598284

theorem cos_pi_over_3_plus_2alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_pi_over_3_plus_2alpha_l598_598284


namespace find_expression_l598_598321

theorem find_expression (x y : ℝ) (h1 : 4 * x + y = 17) (h2 : x + 4 * y = 23) :
  17 * x^2 + 34 * x * y + 17 * y^2 = 818 :=
by
  sorry

end find_expression_l598_598321


namespace minimum_raft_weight_l598_598199

-- Define the weights of the animals.
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal.
def num_mice : ℕ := 5
def num_moles : ℕ := 3
def num_hamsters : ℕ := 4

-- The function that represents the minimum weight capacity required for the raft.
def minimum_raft_capacity : ℕ := 140

-- Prove that the minimum raft capacity to transport all animals is 140 grams.
theorem minimum_raft_weight :
  (∀ (total_weight : ℕ), 
    total_weight = (num_mice * weight_mouse) + (num_moles * weight_mole) + (num_hamsters * weight_hamster) →
    (exists (raft_capacity : ℕ), 
      raft_capacity = minimum_raft_capacity ∧
      raft_capacity >= 2 * weight_mouse)) :=
begin
  -- Initial state setup and logical structure.
  intros total_weight total_weight_eq,
  use minimum_raft_capacity,
  split,
  { refl },
  { have h1: 2 * weight_mouse = 140,
    { norm_num },
    rw h1,
    exact le_refl _,
  }
end

end minimum_raft_weight_l598_598199


namespace MrKozelGarden_l598_598768

theorem MrKozelGarden :
  ∀ (x y : ℕ), 
  (y = 3 * x + 1) ∧ (y = 4 * (x - 1)) → (x = 5 ∧ y = 16) := 
by
  intros x y h
  sorry

end MrKozelGarden_l598_598768


namespace distance_between_planes_l598_598267

noncomputable def plane1 := {p : ℝ × ℝ × ℝ | 3 * p.1 + p.2 - 4 * p.3 + 3 = 0}
noncomputable def plane2 := {p : ℝ × ℝ × ℝ | 6 * p.1 + 2 * p.2 - 8 * p.3 + 6 = 0}

theorem distance_between_planes : 
  ∀ p₁ p₂ : set (ℝ × ℝ × ℝ), 
  p₁ = plane1 → p₂ = plane2 → 
  ∃ d : ℝ, d = 0 := 
begin
  intros _ _ h₁ h₂,
  rw [h₁, h₂],
  use 0,
  sorry,
end

end distance_between_planes_l598_598267


namespace divisor_of_3135_modulo_7_is_391_l598_598134

theorem divisor_of_3135_modulo_7_is_391 :
  ∃ d : ℕ, d = 391 ∧ (55 * 57) % d = 7 := by
  use 391
  have h : 55 * 57 = 3135 := by norm_num
  rw h
  norm_num
  sorry

end divisor_of_3135_modulo_7_is_391_l598_598134


namespace angle_between_vectors_l598_598736

open Real -- to deal with real numbers, trigonometric functions, etc.

theorem angle_between_vectors (α β : ℝ) (hαβ : 0 < α ∧ α < β ∧ β < π) :
  let a := (Real.cos α, Real.sin α)
  let b := (Real.cos β, Real.sin β)
  let sum := (a.1 + b.1, a.2 + b.2)
  let diff := (a.1 - b.1, a.2 - b.2)
  (sum.1 * diff.1 + sum.2 * diff.2 = 0) → angle sum diff = π / 2 :=
by
  assume h
  sorry

end angle_between_vectors_l598_598736


namespace lucky_license_plates_count_l598_598156

open Finset

def num_lucky_license_plates : ℕ :=
  let letters := {'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'U', 'X'}
  let consonants := {'B', 'K', 'M', 'H', 'P', 'C', 'T', 'X'}
  let odd_digits := {1, 3, 5, 7, 9}
  let even_digits := {0, 2, 4, 6, 8}
  let num_letters := 12
  let num_consonants := 8
  let num_odd_digits := 5
  let num_even_digits := 5
  let num_digits := 10
  num_letters * num_odd_digits * num_digits * num_even_digits * num_consonants * num_letters

theorem lucky_license_plates_count :
  num_lucky_license_plates = 288000 := by
  sorry

end lucky_license_plates_count_l598_598156


namespace parabola_directrix_l598_598330

theorem parabola_directrix {p m : ℝ} 
  (h_parabola : (1 : ℝ) ^ 2 = 2 * p * 1)
  (h_distance : (5 : ℝ) = real.sqrt ((1 : ℝ) ^ 2 + (m - 0) ^ 2)): 
  (p = 8) ∧ (x = -4) :=
begin
  sorry
end

end parabola_directrix_l598_598330


namespace sphere_radius_l598_598604

theorem sphere_radius (R r : ℝ) (h1 : R > 0) (h2 : r > 0) :
  ∃ x : ℝ, x = ( -r + real.sqrt (6 * R^2 - 3 * r^2 ) ) / 2 :=
by
  sorry

end sphere_radius_l598_598604


namespace no_real_solution_l598_598248

theorem no_real_solution : ∀ x : ℝ, ¬ ((2*x - 3*x + 7)^2 + 4 = -|2*x|) :=
by
  intro x
  have h1 : (2*x - 3*x + 7)^2 + 4 ≥ 4 := by
    sorry
  have h2 : -|2*x| ≤ 0 := by
    sorry
  -- The main contradiction follows from comparing h1 and h2
  sorry

end no_real_solution_l598_598248


namespace difference_of_squares_count_l598_598745

theorem difference_of_squares_count :
  let count_diff_squares (n : ℕ) := 
    (n / 2) + (n / 4) 
  in 
  count_diff_squares 1200 = 900 :=
by
  sorry

end difference_of_squares_count_l598_598745


namespace coprime_in_bases_l598_598242

noncomputable def gcd : ℕ → ℕ → ℕ
| a, 0 => a
| a, b => gcd b (a % b)

lemma coprime_35_58 : gcd 35 58 = 1 :=
by
  have h1 : gcd 58 35 = gcd 35 23 := by simp [gcd]
  have h2 : gcd 35 23 = gcd 23 12 := by simp [gcd]
  have h3 : gcd 23 12 = gcd 12 11 := by simp [gcd]
  have h4 : gcd 12 11 = gcd 11 1 := by simp [gcd]
  have h5 : gcd 11 1 = 1 := by simp [gcd]
  simp [gcd, h1, h2, h3, h4, h5]

theorem coprime_in_bases : ∀ b : ℕ, b > 8 → gcd 35 58 = 1 :=
by
  intros b hb
  exact coprime_35_58

end coprime_in_bases_l598_598242


namespace oliver_earning_correct_l598_598032

open Real

noncomputable def total_weight_two_days_ago : ℝ := 5

noncomputable def total_weight_yesterday : ℝ := total_weight_two_days_ago + 5

noncomputable def total_weight_today : ℝ := 2 * total_weight_yesterday

noncomputable def total_weight_three_days : ℝ := total_weight_two_days_ago + total_weight_yesterday + total_weight_today

noncomputable def earning_per_kilo : ℝ := 2

noncomputable def total_earning : ℝ := total_weight_three_days * earning_per_kilo

theorem oliver_earning_correct : total_earning = 70 := by
  sorry

end oliver_earning_correct_l598_598032
