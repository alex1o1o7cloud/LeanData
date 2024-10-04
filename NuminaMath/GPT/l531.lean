import Lean
import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Logarithm.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Field
import Mathlib.Data.Finset
import Mathlib.Data.Fintype
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Rank
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Logic.Basic
import Mathlib.Probability
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Tactic.Linarith

namespace tangent_line_at_1_eqn_l531_531736

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / x

theorem tangent_line_at_1_eqn : 
  ∃ (m b : ℝ), (m = Real.exp 1 - 1) ∧ (b = 2) ∧ 
               (∀ y, ∀ x, y - (Real.exp 1 + 1) = (Real.exp 1 - 1) * (x - 1) ↔ y = (Real.exp 1 - 1) * x + 2) :=
begin
  sorry
end

end tangent_line_at_1_eqn_l531_531736


namespace no_same_distribution_of_silver_as_gold_l531_531726

theorem no_same_distribution_of_silver_as_gold (n m : ℕ) 
  (hn : n ≡ 5 [MOD 10]) 
  (hm : m = 2 * n) 
  : ∀ (f : Fin 10 → ℕ), (∀ i j : Fin 10, i ≠ j → ¬ (f i - f j ≡ 0 [MOD 10])) 
  → ∀ (g : Fin 10 → ℕ), ¬ (∀ i j : Fin 10, i ≠ j → ¬ (g i - g j ≡ 0 [MOD 10])) :=
sorry

end no_same_distribution_of_silver_as_gold_l531_531726


namespace reflection_A_l531_531243

noncomputable theory
open scoped Classical

-- Given definitions from the conditions
variables {A B C D K E T S X : Type}
variables [acute_triangle ABC] [circumcircle ω ABC]
variables (D_perpendicular : ∀ {A B C}, perpendicular A B C D)
variables (D_on_BC : ∀ {A B C}, D ∈ segment B C)
variables (K_on_ω : ∀ {A B C}, K ∈ ω)
variables (circle_A : ∀ {A D}, tangent_circumcircle A D)
variables (E_on_ω : ∀ {A D}, E ∈ ω)
variables (E_on_circle_A : ∀ {A D}, E ∈ circle_A)
variables (T_on_BC : ∀ {A E}, T ∈ intersection_line A E B C)
variables (S_on_ω : ∀ {T K}, S ∈ intersection_line T K ω)
variables (X_on_ω : ∀ {S D}, X ∈ intersection_line S D ω)

-- The proof statement
theorem reflection_A {A B C D K E T S X}
    (h1 : acute_triangle ABC) (h2 : circumcircle ω ABC)
    (h3 : perpendicular A B C D) (h4 : D ∈ segment B C)
    (h5 : K ∈ ω) (h6 : circle_tangent A D BC)
    (h7 : E ∈ ω) (h8 : E ∈ circle A D) 
    (h9 : T ∈ intersection_line A E B C) 
    (h10 : S ∈ intersection_line T K ω)
    (h11 : X ∈ intersection_line S D ω) :
    reflection_of A X (perpendicular_bisector B C) :=
by sorry

end reflection_A_l531_531243


namespace find_angle_C_range_sinA_sinC_l531_531229

variable {A B C a b c : ℝ}

theorem find_angle_C 
    (h1 : a = 2) 
    (h2 : b = 2 * Real.sqrt 3) 
    (h3 : Real.cos B = -1 / 2) 
    (h4 : A + B + C = Real.pi) 
    (h5 : 0 < B) 
    (h6 : B < Real.pi) :
    C = Real.pi / 6 := 
sorry

theorem range_sinA_sinC 
    (h1 : a = 2) 
    (h2 : b = 2 * Real.sqrt 3) 
    (h3 : Real.cos B = -1 / 2) 
    (h4 : A + B + C = Real.pi) 
    (h5 : 0 < B) 
    (h6 : B < Real.pi) :
    ∀ s : ℝ, (sin A * sin C = s) → (0 < s ∧ s ≤ 1 / 4) := 
sorry

end find_angle_C_range_sinA_sinC_l531_531229


namespace probability_right_triangle_is_1_over_36_l531_531342

noncomputable def dice_faces := {1, 2, 3, 4, 5, 6}
noncomputable def sample_space := dice_faces × dice_faces × dice_faces

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ b * b + c * c = a * a ∨ c * c + a * a = b * b

-- Define the event of getting (a, b, c) that forms a right-angled triangle
def successful_outcomes : Finset (ℕ × ℕ × ℕ) :=
  (∅ : Finset (ℕ × ℕ × ℕ)).insert (3, 4, 5)
                               .insert (3, 5, 4)
                               .insert (4, 3, 5)
                               .insert (4, 5, 3)
                               .insert (5, 3, 4)
                               .insert (5, 4, 3)

theorem probability_right_triangle_is_1_over_36 :
  (successful_outcomes.card : ℚ) / (sample_space.card : ℚ) = 1 / 36 :=
by
  sorry

end probability_right_triangle_is_1_over_36_l531_531342


namespace price_increase_percentage_l531_531374

theorem price_increase_percentage (P : ℝ) :
  let P1 := P * 0.75 in
  let P2 := P1 * 0.90 in
  let increase_needed := P - P2 in
  ((increase_needed / P2) * 100) ≈ 48.15 := 
  sorry

end price_increase_percentage_l531_531374


namespace find_circle_and_line_l531_531574

open Real  -- Opening the Real namespace for real number operations.

noncomputable def circle_center_and_line (a b: ℝ) (x_r y_r: ℝ) : Prop :=
  -- Condition: Point A is on the circle
  ((2 - a) ^ 2 + (-1 + 2 * a) ^ 2 = 2) ∧
  -- Condition: Circle is tangent to the line
  (abs (a + 2 * a + 1) / sqrt 2 = sqrt 2) ∧
  -- Condition: Center lies on the line y = -2x
  (b = -2 * a) ∧
  -- Circle's equation
  ((x_r - a) ^ 2 + (y_r - b) ^ 2 = 2) ∧
  -- Line passing through origin and associated condition on length of intercepted chord
  ∃ (k : ℝ), (abs (k + 2) / sqrt (1 + k^2) = 1 ∨ (2 = 0 ∧ x_r = 0) ∧ y_r = k * x_r)

theorem find_circle_and_line :
  circle_center_and_line 1 (-2) (1 : ℝ) (-2 : ℝ) :=
by sorry  -- Proof is skipped with 'sorry'.

end find_circle_and_line_l531_531574


namespace minimum_area_l531_531916

noncomputable def parabola := {p : ℝ × ℝ // p.snd ^ 2 = 4 * p.fst}

def is_focus (F : ℝ × ℝ) : Prop :=
  F = (1, 0)

def is_chord (A C : parabola) (F : ℝ × ℝ) : Prop :=
  (F.snd) * (C.val.fst - A.val.fst) + (F.fst) * (C.val.snd - A.val.snd) = 0

def perpendicular_chords (A B C D : parabola) (F : ℝ × ℝ) : Prop :=
  is_chord A C F ∧ is_chord B D F ∧ (A.val - C.val).fst * (B.val - D.val).fst + (A.val - C.val).snd * (B.val - D.val).snd = 0

def quadrilateral_area (A B C D : parabola) : ℝ :=
  let ab := ((B.val.snd - A.val.snd) * (B.val.snd - A.val.snd) + (B.val.fst - A.val.fst) * (B.val.fst - A.val.fst))^.sqrt
  let cd := ((D.val.snd - C.val.snd) * (D.val.snd - C.val.snd) + (D.val.fst - C.val.fst) * (D.val.fst - C.val.fst))^.sqrt
  1/2 * ab * cd

theorem minimum_area (A B C D : parabola) (F : ℝ × ℝ) (hF : is_focus F)
  (h : perpendicular_chords A B C D F) :
  quadrilateral_area A B C D = 32 :=
sorry

end minimum_area_l531_531916


namespace test_max_marks_l531_531007

theorem test_max_marks (M : ℝ) (pass_percentage : ℝ) :
  pass_percentage = 0.30 →
  (80 + 10) / pass_percentage = M →
  M = 300 :=
by
  intros h1 h2
  rw [h1] at h2
  exact h2

end test_max_marks_l531_531007


namespace mean_and_mode_of_data_l531_531653

open BigOperators

def data : List ℕ := [7, 5, 6, 8, 7, 9]

lemma mean_of_data : (data.sum / data.length) = 7 := 
by {
  -- sum of the data is (7 + 5 + 6 + 8 + 7 + 9) = 42
  -- length of the data is 6
  -- mean is 42 / 6 = 7
  sorry
}

lemma mode_of_data : (∃ n, n ∈ data ∧ (data.count n = data.maximum.data.count n)) ∧ (data.count 7 = 2) := 
by {
  -- 7 appears twice which is the most frequent 
  sorry
}

# Theorem combining both the mean and mode lemmas
theorem mean_and_mode_of_data : (data.sum / data.length) = 7 ∧ (∃ n, n ∈ data ∧ (data.count n = data.maximum.data.count n)) ∧ (data.count 7 = 2) := 
by {
  split,
  exact mean_of_data,
  exact mode_of_data
}

end mean_and_mode_of_data_l531_531653


namespace shaded_regions_area_sum_l531_531091

theorem shaded_regions_area_sum (side_len : ℚ) (radius : ℚ) (a b c : ℤ) :
  side_len = 16 → radius = side_len / 2 →
  a = (64 / 3) ∧ b = 32 ∧ c = 3 →
  (∃ x : ℤ, x = a + b + c ∧ x = 99) :=
by
  intros hside_len hradius h_constituents
  sorry

end shaded_regions_area_sum_l531_531091


namespace sum_of_divisions_is_187_l531_531742

theorem sum_of_divisions_is_187 :
  ∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    {d1, d2, d3, d4, d5, d6, d7, d8, d9} = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    {62 / d1, 63 / d2, 64 / d3, 65 / d4, 66 / d5, 67 / d6, 68 / d7, 69 / d8, 70 / d9}.Sum = 187 :=
by
  sorry

end sum_of_divisions_is_187_l531_531742


namespace angle_AQB_obtuse_probability_correct_l531_531703

-- Define the vertices of the pentagon
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (3, 1)
def C : ℝ × ℝ := (5, 1)
def D : ℝ × ℝ := (5, 6)
def E : ℝ × ℝ := (-1, 6)

-- Define the probability calculation
noncomputable def probability_obtuse_AQB : ℝ :=
  let area_pentagon := 23
  let radius_semi := Real.sqrt 5
  let area_semi := (radius_semi^2 * Real.pi) / 2
  (area_semi / area_pentagon: ℝ)

-- The theorem statement
theorem angle_AQB_obtuse_probability_correct :
  probability_obtuse_AQB = 5 * Real.pi / 46 :=
sorry

end angle_AQB_obtuse_probability_correct_l531_531703


namespace remaining_meals_to_distribute_l531_531524

theorem remaining_meals_to_distribute :
  let initial_meals := 113
  let sole_mart_meals := 50
  let green_garden_meals := 25
  let colt_additional_meals := 30
  let curt_additional_meals := 40
  let given_away_meals := 85
  let total_meals := initial_meals + sole_mart_meals + green_garden_meals
  let total_meals_after_cooking := total_meals + colt_additional_meals + curt_additional_meals
  total_meals_after_cooking - given_away_meals = 173 :=
by
  let initial_meals := 113
  let sole_mart_meals := 50
  let green_garden_meals := 25
  let colt_additional_meals := 30
  let curt_additional_meals := 40
  let given_away_meals := 85
  let total_meals := initial_meals + sole_mart_meals + green_garden_meals
  let total_meals_after_cooking := total_meals + colt_additional_meals + curt_additional_meals
  show total_meals_after_cooking - given_away_meals = 173 from sorry

end remaining_meals_to_distribute_l531_531524


namespace identify_heaviest_and_lightest_coin_l531_531808

theorem identify_heaviest_and_lightest_coin :
  ∀ (coins : Fin 10 → ℕ), 
  (∀ i j, i ≠ j → coins i ≠ coins j) → 
  ∃ (seq : List (Fin 10 × Fin 10)), 
  seq.length = 13 ∧ 
  (∀ (i j : Fin 10), (i, j) ∈ seq → 
    (coins i < coins j ∨ coins i > coins j)) ∧ 
  (∃ (heaviest lightest : Fin 10),
    (∀ coin, coins coin ≤ coins heaviest) ∧ (∀ coin, coins coin ≥ coins lightest)) :=
by
  intros coins h_coins
  exists [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), -- initial pairs
          (0, 2), (2, 4), (4, 6), (6, 8),         -- heaviest coin comparisons
          (1, 3), (3, 5), (5, 7), (7, 9)]         -- lightest coin comparisons
  constructor
  . -- length check
    rfl
  . constructor
    . -- all comparisons
      intros i j h_pair
      cases h_pair; simp; solve_by_elim
    . -- finding heaviest and lightest coins
      exists 8, 9
      constructor
      . -- all coins are less than or equal to the heaviest
        sorry
      . -- all coins are greater than or equal to the lightest
        sorry

end identify_heaviest_and_lightest_coin_l531_531808


namespace sum_complex_l531_531107

-- Define the given complex numbers
def z1 : ℂ := ⟨2, 5⟩
def z2 : ℂ := ⟨3, -7⟩

-- State the theorem to prove the sum
theorem sum_complex : z1 + z2 = ⟨5, -2⟩ :=
by
  sorry

end sum_complex_l531_531107


namespace maximum_constant_C_l531_531910

noncomputable def interesting_sequence (z : ℕ → ℂ) : Prop :=
  | z 1 = 1
  | ∀ n : ℕ, n > 0 → 4 * (z (n+1))^2 + 2 * z n * z (n+1) + (z n)^2 = 0

theorem maximum_constant_C :
  ∃ C : ℝ, (∀ (z : ℕ → ℂ), interesting_sequence z → ∀ (m : ℕ), m > 0 → 
    |∑ i in finset.range m, z (i + 1)| ≥ C) ∧ C = real.sqrt 3 / 3 :=
sorry

end maximum_constant_C_l531_531910


namespace Yangyang_helps_Mom_for_4_days_l531_531671

/-
Problem:
In two warehouses, Warehouse A and Warehouse B, the amounts of rice are the same.
If Dad, Mom, and Yangyang work alone, they can each finish transporting all the rice from one warehouse in 10 days, 12 days, and 15 days, respectively.
Dad and Mom start transporting rice from Warehouse A and Warehouse B at the same time.
Yangyang first helps Mom and then Dad, and they finish transporting the rice from both warehouses at the same time.
How many days did Yangyang help Mom transport rice?

Conditions:
- rate_dad = 1 / 10 (units/day)
- rate_mom = 1 / 12 (units/day)
- rate_yangyang = 1 / 15 (units/day)
- They finish both warehouses at the same time.
- Dad and Mom start at the same time.
- Yangyang helps Mom first, then Dad.

Conclusion:
Yangyang helped Mom for exactly 4 days.
-/

def rate_dad : ℝ := 1 / 10
def rate_mom : ℝ := 1 / 12
def rate_yangyang : ℝ := 1 / 15

theorem Yangyang_helps_Mom_for_4_days :
  let combined_rate := rate_dad + rate_mom + rate_yangyang in
  ∀ (days : ℝ), days = 4 → 
     (combined_rate * days = 1) →
     (rate_dad * days + rate_yangyang * days = 1 - rate_mom * days) :=
sorry

end Yangyang_helps_Mom_for_4_days_l531_531671


namespace concentration_proof_l531_531420

noncomputable def newConcentration (vol1 vol2 vol3 : ℝ) (perc1 perc2 perc3 : ℝ) (totalVol : ℝ) (finalVol : ℝ) :=
  (vol1 * perc1 + vol2 * perc2 + vol3 * perc3) / finalVol

theorem concentration_proof : 
  newConcentration 2 6 4 0.2 0.55 0.35 (12 : ℝ) (15 : ℝ) = 0.34 := 
by 
  sorry

end concentration_proof_l531_531420


namespace count_divisible_by_16_l531_531191

theorem count_divisible_by_16 : 
  (∃ (a b c : ℕ), a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                  b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                  c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                  (10000 * a + 1000 * b + 160 + c) % 16 = 0) ↔ 
  (S := 9 * 10) := sorry

end count_divisible_by_16_l531_531191


namespace maximum_weight_truck_can_carry_l531_531878

-- Definitions for the conditions.
def weight_boxes : Nat := 100 * 100
def weight_crates : Nat := 10 * 60
def weight_sacks : Nat := 50 * 50
def weight_additional_bags : Nat := 10 * 40

-- Summing up all the weights.
def total_weight : Nat :=
  weight_boxes + weight_crates + weight_sacks + weight_additional_bags

-- The theorem stating the maximum weight.
theorem maximum_weight_truck_can_carry : total_weight = 13500 := by
  sorry

end maximum_weight_truck_can_carry_l531_531878


namespace least_number_to_add_l531_531829

theorem least_number_to_add 
  (a b c d e : ℕ)
  (h7 : a = 7)
  (h11 : b = 11)
  (h13 : c = 13)
  (h17 : d = 17)
  (h19 : e = 19) :
  let lcm := Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e)))
  in lcm = 323323 → 
     let remainder := 625573 % lcm
     in 21073 = lcm - remainder := 
by {
  sorry
}

end least_number_to_add_l531_531829


namespace cube_surface_area_difference_l531_531861

theorem cube_surface_area_difference :
  let large_cube_volume := 8
  let small_cube_volume := 1
  let num_small_cubes := 8
  let large_cube_side := (large_cube_volume : ℝ) ^ (1 / 3)
  let small_cube_side := (small_cube_volume : ℝ) ^ (1 / 3)
  let large_cube_surface_area := 6 * (large_cube_side ^ 2)
  let small_cube_surface_area := 6 * (small_cube_side ^ 2)
  let total_small_cubes_surface_area := num_small_cubes * small_cube_surface_area
  total_small_cubes_surface_area - large_cube_surface_area = 24 :=
by
  sorry

end cube_surface_area_difference_l531_531861


namespace table_area_l531_531343

/-- Given the combined area of three table runners is 224 square inches, 
     overlapping the runners to cover 80% of a table results in exactly 24 square inches being covered by 
     two layers, and the area covered by three layers is 30 square inches,
     prove that the area of the table is 175 square inches. -/
theorem table_area (A : ℝ) (S T H : ℝ) (h1 : S + 2 * T + 3 * H = 224)
   (h2 : 0.80 * A = S + T + H) (h3 : T = 24) (h4 : H = 30) : A = 175 := 
sorry

end table_area_l531_531343


namespace units_digit_27_mul_46_l531_531557

theorem units_digit_27_mul_46 : (27 * 46) % 10 = 2 :=
by 
  -- Definition of units digit
  have def_units_digit :=  (n : ℕ) => n % 10

  -- Step 1: units digit of 27 is 7
  have units_digit_27 : 27 % 10 = 7 := by norm_num
  
  -- Step 2: units digit of 46 is 6
  have units_digit_46 : 46 % 10 = 6 := by norm_num

  -- Step 3: multiple the units digits
  have step3 : 7 * 6 = 42 := by norm_num

  -- Step 4: Find the units digit of 42
  have units_digit_42 : 42 % 10 = 2 := by norm_num

  exact units_digit_42

end units_digit_27_mul_46_l531_531557


namespace least_number_to_subtract_l531_531932

theorem least_number_to_subtract (x : ℕ) (h : x = 1234567890) : ∃ n, x - n = 5 := 
  sorry

end least_number_to_subtract_l531_531932


namespace scientific_notation_of_coronavirus_diameter_l531_531733

theorem scientific_notation_of_coronavirus_diameter : 
  (0.00000011 : ℝ) = 1.1 * 10^(-7) :=
  sorry

end scientific_notation_of_coronavirus_diameter_l531_531733


namespace chord_length_perpendicular_l531_531836

theorem chord_length_perpendicular 
  (R a b : ℝ)  
  (h1 : a + b = R)
  (h2 : (1 / 2) * Real.pi * R^2 - (1 / 2) * Real.pi * (a^2 + b^2) = 10 * Real.pi) :
  2 * Real.sqrt 10 = 6.32 :=
by 
  sorry

end chord_length_perpendicular_l531_531836


namespace pencil_count_l531_531798

-- Lean statement to encapsulate the problem
theorem pencil_count (initial_pencils : ℕ) 
                     (nancy_added : ℕ) 
                     (steven_added : ℕ) 
                     (maria_contributed : ℕ) 
                     (kim_took : ℕ) 
                     (george_removed : ℕ) : 
                     initial_pencils = 200 →
                     nancy_added = 375 →
                     steven_added = 150 →
                     maria_contributed = 250 →
                     kim_took = 85 →
                     george_removed = 60 →
                     initial_pencils + 
                     nancy_added + 
                     steven_added + 
                     maria_contributed - 
                     kim_took - 
                     george_removed = 830 :=
by
  intros h_init h_nancy h_steven h_maria h_kim h_george
  rw [h_init, h_nancy, h_steven, h_maria, h_kim, h_george]
  -- Starting from the initial number of pencils and adding/removing step by step
  calc 
    (200 : ℕ) 
    + 375 
    + 150 
    + 250 
    - 85 
    - 60 
    = 575       : by norm_num
    ... = 725   : by norm_num
    ... = 975   : by norm_num
    ... = 890   : by norm_num
    ... = 830   : by norm_num
  -- sorry is not needed here since the calc steps complete the proof

end pencil_count_l531_531798


namespace area_of_larger_figure_excluding_XYZ_l531_531218

-- Definition of the problem conditions and parameters
def side_length1 : ℝ := 2
def side_length2 : ℝ := 1
def angle_FAB_BCD : ℝ := 60
def num_large_triangles : ℕ := 4

-- Area calculation for one equilateral triangle with given side length
def equilateral_triangle_area (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

-- Area of the larger figure excluding the smaller equilateral triangle
theorem area_of_larger_figure_excluding_XYZ :
  let A_large := equilateral_triangle_area side_length1,
      A_small := equilateral_triangle_area side_length2,
      T_large := num_large_triangles * A_large,
      T := T_large - A_small
  in T = (15 * sqrt 3) / 4 :=
by
  sorry

end area_of_larger_figure_excluding_XYZ_l531_531218


namespace dice_diff_by_three_probability_l531_531437

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l531_531437


namespace percentage_l_75_m_l531_531201

theorem percentage_l_75_m
  (j k l m : ℝ)
  (x : ℝ)
  (h1 : 1.25 * j = 0.25 * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : (x / 100) * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 175 :=
by
  sorry

end percentage_l_75_m_l531_531201


namespace parabola_focus_distance_l531_531184

noncomputable def parabola_directrix_focus_distance (x y : ℝ) : Prop :=
  let directrix : ℝ := -1
  let c := 1 / 4
  let ordinateA := 4
  let focus_distance := |ordinateA - directrix| in
  x^2 = 16 → focus_distance = 5

theorem parabola_focus_distance (x y : ℝ) (h1 : y = 1 / 4 * x^2) (h2 : y = 4) : parabola_directrix_focus_distance x y :=
by
  sorry

end parabola_focus_distance_l531_531184


namespace num_excluding_n_is_20_l531_531396

-- Define the conditions and question in Lean
theorem num_excluding_n_is_20 (L : List ℝ) (n : ℝ) (hL_len : L.Length = 21)
  (hn_avg : n = 5 * (L.filter (≠ n)).sum / 20)
  (hn_sum : n = 0.2 * L.sum)
  : L.Length - 1 = 20 := 
begin
  sorry
end

end num_excluding_n_is_20_l531_531396


namespace part1_coordinates_of_P_if_AB_perp_PB_part2_coordinates_of_P_area_ABP_10_l531_531660

-- Part (Ⅰ)
theorem part1_coordinates_of_P_if_AB_perp_PB :
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (P = (7, 0)) :=
by
  sorry

-- Part (Ⅱ)
theorem part2_coordinates_of_P_area_ABP_10 :
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (P = (9, 0) ∨ P = (-11, 0)) :=
by
  sorry

end part1_coordinates_of_P_if_AB_perp_PB_part2_coordinates_of_P_area_ABP_10_l531_531660


namespace available_floor_space_equals_110_sqft_l531_531029

-- Definitions for the conditions
def tile_side_in_feet : ℝ := 0.5
def width_main_section_tiles : ℕ := 15
def length_main_section_tiles : ℕ := 25
def width_alcove_tiles : ℕ := 10
def depth_alcove_tiles : ℕ := 8
def width_pillar_tiles : ℕ := 3
def length_pillar_tiles : ℕ := 5

-- Conversion of tiles to feet
def width_main_section_feet : ℝ := width_main_section_tiles * tile_side_in_feet
def length_main_section_feet : ℝ := length_main_section_tiles * tile_side_in_feet
def width_alcove_feet : ℝ := width_alcove_tiles * tile_side_in_feet
def depth_alcove_feet : ℝ := depth_alcove_tiles * tile_side_in_feet
def width_pillar_feet : ℝ := width_pillar_tiles * tile_side_in_feet
def length_pillar_feet : ℝ := length_pillar_tiles * tile_side_in_feet

-- Area calculations
def area_main_section : ℝ := width_main_section_feet * length_main_section_feet
def area_alcove : ℝ := width_alcove_feet * depth_alcove_feet
def total_area : ℝ := area_main_section + area_alcove
def area_pillar : ℝ := width_pillar_feet * length_pillar_feet
def available_floor_space : ℝ := total_area - area_pillar

-- Proof statement
theorem available_floor_space_equals_110_sqft 
  (h1 : width_main_section_feet = width_main_section_tiles * tile_side_in_feet)
  (h2 : length_main_section_feet = length_main_section_tiles * tile_side_in_feet)
  (h3 : width_alcove_feet = width_alcove_tiles * tile_side_in_feet)
  (h4 : depth_alcove_feet = depth_alcove_tiles * tile_side_in_feet)
  (h5 : width_pillar_feet = width_pillar_tiles * tile_side_in_feet)
  (h6 : length_pillar_feet = length_pillar_tiles * tile_side_in_feet) 
  (h7 : area_main_section = width_main_section_feet * length_main_section_feet)
  (h8 : area_alcove = width_alcove_feet * depth_alcove_feet)
  (h9 : total_area = area_main_section + area_alcove)
  (h10 : area_pillar = width_pillar_feet * length_pillar_feet)
  (h11 : available_floor_space = total_area - area_pillar) : 
  available_floor_space = 110 := 
by 
  sorry

end available_floor_space_equals_110_sqft_l531_531029


namespace evaluate_expression_l531_531914

def g (x : ℝ) : ℝ := x^2 + 3 * Real.sqrt x

theorem evaluate_expression :
  3 * g 3 - g 9 = -63 + 9 * Real.sqrt 3 :=
by
  -- proof steps go here
  sorry

end evaluate_expression_l531_531914


namespace difference_sqrt_cube_approx_l531_531120

-- Define the conditions as variables
def x := 0.4 * 60
def y := Real.sqrt x
def z := 25 ^ 3
def w := (4 / 5) * z

-- State the theorem we are going to prove
theorem difference_sqrt_cube_approx :
  y - w ≈ -12495.10102 :=
by
  sorry

end difference_sqrt_cube_approx_l531_531120


namespace num_sheets_in_stack_l531_531048

-- Definitions coming directly from the conditions
def thickness_ream := 4 -- cm
def num_sheets_ream := 400
def height_stack := 10 -- cm

-- The final proof statement
theorem num_sheets_in_stack : (height_stack / (thickness_ream / num_sheets_ream)) = 1000 :=
by
  sorry

end num_sheets_in_stack_l531_531048


namespace intersection_eq_l531_531262

def A := {x : ℝ | |x| = x}
def B := {x : ℝ | x^2 + x ≥ 0}

theorem intersection_eq : A ∩ B = {x : ℝ | 0 ≤ x} := by
  sorry

end intersection_eq_l531_531262


namespace count_five_digit_multiples_of_seven_l531_531193

theorem count_five_digit_multiples_of_seven : 
  (set_of (λ n, (10000 ≤ n ∧ n ≤ 99999) ∧ n % 7 = 0)).card = 12858 :=
sorry

end count_five_digit_multiples_of_seven_l531_531193


namespace probability_of_difference_three_l531_531494

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l531_531494


namespace complement_U_A_eq_two_l531_531022

open Set

universe u

def U : Set ℕ := { x | x ≥ 2 }
def A : Set ℕ := { x | x^2 ≥ 5 }
def comp_U_A : Set ℕ := U \ A

theorem complement_U_A_eq_two : comp_U_A = {2} :=
by 
  sorry

end complement_U_A_eq_two_l531_531022


namespace poly_expansion_sum_l531_531632

theorem poly_expansion_sum (A B C D E : ℤ) (x : ℤ):
  (x + 3) * (4 * x^3 - 2 * x^2 + 3 * x - 1) = A * x^4 + B * x^3 + C * x^2 + D * x + E → 
  A + B + C + D + E = 16 :=
by
  sorry

end poly_expansion_sum_l531_531632


namespace find_a_l531_531613

noncomputable def angle := 30 * Real.pi / 180 -- In radians

noncomputable def tan_angle : ℝ := Real.tan angle

theorem find_a (a : ℝ) (h1 : tan_angle = 1 / Real.sqrt 3) : 
  x - a * y + 3 = 0 → a = Real.sqrt 3 :=
by
  sorry

end find_a_l531_531613


namespace number_of_spheres_in_cylinder_l531_531038

-- Define the diameter and height of the cylinder, and the diameter of the sphere
def cylinder_diameter : ℝ := 82
def cylinder_height : ℝ := 225
def sphere_diameter : ℝ := 38

-- Calculate the radius of the cylinder and the sphere
def cylinder_radius : ℝ := cylinder_diameter / 2
def sphere_radius : ℝ := sphere_diameter / 2

-- The main statement to prove
theorem number_of_spheres_in_cylinder :
  ∃ n : ℕ, n = 21 ∧
    cylinder_diameter = 82 ∧
    cylinder_height = 225 ∧
    sphere_diameter = 38 :=
sorry

end number_of_spheres_in_cylinder_l531_531038


namespace probability_of_differ_by_three_l531_531502

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l531_531502


namespace part1_part2_l531_531144

variable (a : ℝ) 

def P : Prop := ∀ x ∈ set.Icc 1 2, x^2 - a ≥ 0
def Q : Prop := (∃ (x y : ℝ), x^2 / (a + 1) + y^2 / (a - 2) = 1)

theorem part1 (hQ : Q) : a ∈ set.Ioo (-1 : ℝ) 2 :=
by sorry

theorem part2 (h_or : P ∨ Q) (h_and : ¬(P ∧ Q)) : a ∈ set.union (set.Ioo 1 2) (set.Iic (-1)) :=
by sorry

end part1_part2_l531_531144


namespace correct_proposition_l531_531606

noncomputable def f (x a : ℝ) := log (x^2 + a * x - a - 1)

theorem correct_proposition (a : ℝ) :
  let Prop1 := ∀ x, x^2 + a * x - a - 1 < 0 → ∃ min, f x a = min
  let Prop2 := (a = 0 → range (λ x, f x 0) = set.univ)
  let Prop3 := (∀ x ∈ set.Icc (-∞ : ℝ) 2, monotone_decreasing (λ x, f x a)) → a ≤ -4
  Prop2 := true :=
sorry

end correct_proposition_l531_531606


namespace problem1_problem2_l531_531387

-- Define the binomial coefficient C(2n, n)
noncomputable def binom_coeff (n : ℕ) : ℕ := (Nat.factorial (2 * n)) / ((Nat.factorial n) * (Nat.factorial n))

-- Define the prime counting function π(x)
noncomputable def prime_count (x : ℝ) : ℕ := Finset.card (Finset.filter Prime (Finset.Icc 2 (Nat.floor x)))

-- Define mathematical problem as a statement in Lean 4
theorem problem1 (n : ℕ) (h1 : n > 0) : 
  (∃ p, n < p ∧ p ≤ 2 * n ∧ p ∣ binom_coeff n) ∧ binom_coeff n < 2^(2*n) := by
  sorry

theorem problem2 (n : ℕ) (h1 : n ≥ 3) : 
  prime_count (2 * ↑n) < prime_count (↑n) + (2 * ↑n) / Real.log2 ↑n ∧ 
  prime_count (2^n) < (2^(n+1) * Real.log2 (n-1)) / n ∧ 
  (∀ (x : ℝ), x ≥ 8 → prime_count x < (4 * x * Real.log2 (Real.log2 x)) / Real.log2 x) := by
  sorry

end problem1_problem2_l531_531387


namespace probability_of_differ_by_three_l531_531501

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l531_531501


namespace color_8_cells_black_on_4x4_chessboard_l531_531103

theorem color_8_cells_black_on_4x4_chessboard :
  ∃ (ways : ℕ), ways = 90 ∧ 
    ∃ (board : matrix (fin 4) (fin 4) bool), 
      (∀ i, (finset.filter id (finset.univ.map (function.eval i))).card = 2) ∧
      (∀ j, (finset.filter id (finset.univ.map (λ i, board i j))).card = 2) :=
sorry

end color_8_cells_black_on_4x4_chessboard_l531_531103


namespace extreme_point_f_min_value_g_l531_531997

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Problem 1: Extreme points and value of f
theorem extreme_point_f : 
  ∃ x, f x = -1 / Real.exp 1 ∧ x = 1 / Real.exp 1 := sorry

noncomputable def g (x a : ℝ) : ℝ := x * Real.log x - a * (x - 1)

-- Problem 2: Minimum value of g on [1, e]
theorem min_value_g (a : ℝ) :
  ∀ x ∈ Set.Icc 1 Real.exp 1, 
  (a < 1 → g 1 a = 0) ∧
  (1 ≤ a ∧ a ≤ 2 → ∃ x, x = Real.exp (a - 1) ∧ g x a = a - Real.exp (a - 1)) ∧
  (a > 2 → g (Real.exp 1) a = (1 - a) * Real.exp 1 + a) := sorry

end extreme_point_f_min_value_g_l531_531997


namespace probability_pink_correct_l531_531377

def total_flowers_a : ℕ := 6 + 3
def total_flowers_b : ℕ := 2 + 7

def pink_flowers_a : ℕ := 3
def pink_flowers_b : ℕ := 7

def probability_pink_a : ℚ := pink_flowers_a / total_flowers_a
def probability_pink_b : ℚ := pink_flowers_b / total_flowers_b

def probability_pink : ℚ := (probability_pink_a + probability_pink_b) / 2

theorem probability_pink_correct : probability_pink = 5 / 9 := by
  sorry

end probability_pink_correct_l531_531377


namespace sqrt_condition_then_square_l531_531199

theorem sqrt_condition_then_square (x : ℝ) (h : sqrt (x + 3) = 3) : (x + 3)^2 = 81 :=
sorry

end sqrt_condition_then_square_l531_531199


namespace elvins_fixed_charge_l531_531110

theorem elvins_fixed_charge (F C : ℝ) 
  (h1 : F + C = 40) 
  (h2 : F + 2 * C = 76) : F = 4 := 
by 
  sorry

end elvins_fixed_charge_l531_531110


namespace main_theorem_l531_531823

open Classical

noncomputable def problem_statement 
  (O1 O2 A1 A2 B1 B2 : Type) 
  (line_s : O1 → O2 → Type) 
  (h1 : ∀ {O1 O2 A1 A2}, ¬ (O1 = O2) ∧ ¬ (A1 = A2) ∧ O1 ≠ O2)
  (h2 : ∀ {O1 O2 A1 A2}, s A1 A2) 
  (h3 : B1 ∈ circle O1 ∧ B2 ∈ circle O2) 
  (h4 : line O1 O2 ∩ circle O1 = {B1} ∧ line O1 O2 ∩ circle O2 = {B2}) 
  : Prop := 
  ∀l1 l2, (line A1 B1 = l1) → (line A2 B2 = l2) → is_perpendicular l1 l2

theorem main_theorem 
  (O1 O2 A1 A2 B1 B2 : Point) 
  (line_s : Line) 
  (h1 : ¬ O1.is_intersecting O2 ∧ 
        ¬ O1.is_intersecting A2 ∧ 
        O2.is_tangent line_s ∧ 
        A1.is_on_line line_s ∧ 
        A2.is_on_line line_s)
  (h2 : O1.is_on_same_side line_s ∧ 
        O2.is_on_same_side line_s)
  (h3 : B1.is_on_circle O1 ∧ 
        B2.is_on_circle O2)
  (h4 : O1O2.intersect_circle O1 = {B1} ∧ 
        O1O2.intersect_circle O2 = {B2}) 
  : problem_statement O1 O2 A1 A2 B1 B2 line_s h1 h2 h3 h4 := 
sorry

end main_theorem_l531_531823


namespace total_race_distance_l531_531283

theorem total_race_distance :
  let sadie_time := 2
  let sadie_speed := 3
  let ariana_time := 0.5
  let ariana_speed := 6
  let total_time := 4.5
  let sarah_speed := 4
  let sarah_time := total_time - sadie_time - ariana_time
  let sadie_distance := sadie_speed * sadie_time
  let ariana_distance := ariana_speed * ariana_time
  let sarah_distance := sarah_speed * sarah_time
  let total_distance := sadie_distance + ariana_distance + sarah_distance
  total_distance = 17 :=
by
  sorry

end total_race_distance_l531_531283


namespace professor_has_to_grade_405_more_problems_l531_531418

theorem professor_has_to_grade_405_more_problems
  (problems_per_paper : ℕ)
  (total_papers : ℕ)
  (graded_papers : ℕ)
  (remaining_papers := total_papers - graded_papers)
  (p : ℕ := remaining_papers * problems_per_paper) :
  problems_per_paper = 15 ∧ total_papers = 45 ∧ graded_papers = 18 → p = 405 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end professor_has_to_grade_405_more_problems_l531_531418


namespace sum_of_factors_l531_531113

theorem sum_of_factors (x y : ℕ) :
  let exp := (27 * x ^ 6 - 512 * y ^ 6)
  let factor1 := (3 * x ^ 2 - 8 * y ^ 2)
  let factor2 := (3 * x ^ 2 + 8 * y ^ 2)
  let factor3 := (9 * x ^ 4 - 24 * x ^ 2 * y ^ 2 + 64 * y ^ 4)
  let sum := 3 + (-8) + 3 + 8 + 9 + (-24) + 64
  (factor1 * factor2 * factor3 = exp) ∧ (sum = 55) := 
by
  sorry

end sum_of_factors_l531_531113


namespace find_C_coordinates_l531_531233

variables {A B M L C : ℝ × ℝ}

def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def on_line_bisector (L B : ℝ × ℝ) : Prop :=
  B.1 = 6  -- Vertical line through B

theorem find_C_coordinates
  (A := (2, 8))
  (M := (4, 11))
  (L := (6, 6))
  (hM : is_midpoint M A B)
  (hL : on_line_bisector L B) :
  C = (6, 14) :=
sorry

end find_C_coordinates_l531_531233


namespace no_right_triangle_set_B_l531_531885

theorem no_right_triangle_set_B : ¬ ∃ a b c : ℝ, (a = 4 ∧ b = 5 ∧ c = 6) ∧ (a^2 + b^2 = c^2) := 
by 
  intro h
  obtain ⟨a, b, c, ⟨ha, hb, hc⟩, h_pythagorean⟩ := h
  rw [←ha, ←hb, ←hc] at h_pythagorean
  have h1 : 4^2 + 5^2 = 6^2 := h_pythagorean
  norm_num at h1
  exact h1
sorry

end no_right_triangle_set_B_l531_531885


namespace dice_diff_by_three_probability_l531_531433

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l531_531433


namespace carson_circles_theorem_l531_531900

-- Define the dimensions of the warehouse
def warehouse_length : ℕ := 600
def warehouse_width : ℕ := 400

-- Define the perimeter calculation
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- Define the distance Carson walked
def distance_walked : ℕ := 16000

-- Define the number of circles Carson skipped
def circles_skipped : ℕ := 2

-- Define the expected number of circles Carson was supposed to circle
def expected_circles :=
  let actual_circles := distance_walked / (perimeter warehouse_length warehouse_width)
  actual_circles + circles_skipped

-- The theorem we want to prove
theorem carson_circles_theorem : expected_circles = 10 := by
  sorry

end carson_circles_theorem_l531_531900


namespace base7_addition_example_l531_531129

theorem base7_addition_example : 
  let a := 2 * 7^2 + 4 * 7^1 + 5 * 7^0,
      b := 5 * 7^2 + 4 * 7^1 + 3 * 7^0,
      result := 1 * 7^3 + 1 * 7^2 + 2 * 7^1 + 1 * 7^0 in
  a + b = result := by
sorry

end base7_addition_example_l531_531129


namespace sasha_remainder_l531_531786

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l531_531786


namespace geometric_sequence_properties_l531_531594

variable (a : ℕ → ℝ)
variable (q : ℝ)

noncomputable def geometric_series (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_properties
  (hgeo : geometric_series a q)
  (hpos : ∀ n, 0 < a n)
  (ha1 : a 1 = 3)
  (hsum : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
begin
  sorry
end

end geometric_sequence_properties_l531_531594


namespace probability_of_difference_three_l531_531489

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l531_531489


namespace trig_sum_roots_l531_531198

theorem trig_sum_roots {θ a : Real} (hroots : ∀ x, x^2 - a * x + a = 0 → x = Real.sin θ ∨ x = Real.cos θ) :
  Real.cos (θ - 3 * Real.pi / 2) + Real.sin (3 * Real.pi / 2 + θ) = Real.sqrt 2 - 1 :=
by
  sorry

end trig_sum_roots_l531_531198


namespace area_ratio_XYP_XZP_l531_531231

theorem area_ratio_XYP_XZP (X Y Z P : Type) [normed_field X] [normed_space X Y] [normed_space X Z] :
  ∀ (XY XZ YZ XP : ℝ),
  XY = 20 → XZ = 30 → YZ = 25 →
  ∃ (YP ZP : ℝ), XP = YP + ZP ∧ (YP / ZP = XY / XZ) →
  (YP / ZP = (2 / 3)) →
  ∀ (area_XYP area_XZP : ℝ), (area_XYP / area_XZP = 2 / 3) :=
by
  intros XY XZ YZ XP hXY hXZ hYZ YP ZP hYPZP_ratio hYPZP_fraction area_XYP area_XZP
  -- skipping the proof here
  sorry

end area_ratio_XYP_XZP_l531_531231


namespace savings_percentage_proof_l531_531241

variables (S : ℝ) (last_year_savings_rate this_year_savings_rate increase_rate : ℝ)
variable (salary_last_year : ℝ)
variable (salary_this_year : ℝ)
variable (savings_last_year : ℝ)
variable (savings_this_year : ℝ)

-- Conditions
axiom h1 : last_year_savings_rate = 0.06
axiom h2 : increase_rate = 0.10
axiom h3 : this_year_savings_rate = 0.09

def salary_last_year := S
def salary_this_year := (1 + increase_rate) * S
def savings_last_year := last_year_savings_rate * S
def savings_this_year := this_year_savings_rate * salary_this_year

theorem savings_percentage_proof
  (S : ℝ)
  (last_year_savings_rate this_year_savings_rate increase_rate : ℝ)
  (h1 : last_year_savings_rate = 0.06)
  (h2 : increase_rate = 0.10)
  (h3 : this_year_savings_rate = 0.09)
  : (savings_this_year S last_year_savings_rate this_year_savings_rate increase_rate) / 
    (savings_last_year S last_year_savings_rate) * 100 = 165 :=
by
  sorry

end savings_percentage_proof_l531_531241


namespace samanta_s_eggs_left_l531_531717

def total_eggs : ℕ := 30
def cost_per_crate_dollars : ℕ := 5
def cost_per_crate_cents : ℕ := cost_per_crate_dollars * 100
def sell_price_per_egg_cents : ℕ := 20

theorem samanta_s_eggs_left
  (total_eggs : ℕ) (cost_per_crate_dollars : ℕ) (sell_price_per_egg_cents : ℕ) 
  (cost_per_crate_cents = cost_per_crate_dollars * 100) : 
  total_eggs - (cost_per_crate_cents / sell_price_per_egg_cents) = 5 :=
by sorry

end samanta_s_eggs_left_l531_531717


namespace biodiversity_of_kerbin_l531_531822

theorem biodiversity_of_kerbin (n : ℕ) : ∀ (existing_species extinct_species : ℕ),
  existing_species = (2 : ℕ) ^ n →
  extinct_species = (2 : ℕ) ^ n - 1 →
  (2 * existing_species - 2) / (extinct_species) = 2 := by {
  intros existing_species extinct_species h1 h2,
  sorry
}

end biodiversity_of_kerbin_l531_531822


namespace find_integers_solution_l531_531118

open Nat Real

theorem find_integers_solution :
  {x y z : ℕ // x > 0 ∧ y > 0 ∧ z > 0 ∧ (x, y, z) ∈ {(2, 4, 15), (2, 5, 9), (2, 6, 7), (3, 3, 8), (3, 4, 5)}}
  ↔ ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (1 + (1/x)) * (1 + (1/y)) * (1 + (1/z)) = 2 := 
  sorry

end find_integers_solution_l531_531118


namespace min_sin4_cos4_l531_531933

theorem min_sin4_cos4 : ∃ x : ℝ, sin^2 x + cos^2 x = 1 ∧ sin^4 x + cos^4 x = 1 / 2:= by
  sorry

end min_sin4_cos4_l531_531933


namespace imaginary_root_eq_four_l531_531200

theorem imaginary_root_eq_four (z : ℂ) (h1 : |z| = 2) (h2 : is_root (λ x : ℂ, x^2 + 2*x + 4) z) :
  ∃ p : ℝ, p = 4 :=
begin
  sorry
end

end imaginary_root_eq_four_l531_531200


namespace quiz_score_difference_l531_531271

theorem quiz_score_difference :
  let percentage_70 := 0.10
  let percentage_80 := 0.35
  let percentage_90 := 0.30
  let percentage_100 := 0.25
  let mean_score := (percentage_70 * 70) + (percentage_80 * 80) + (percentage_90 * 90) + (percentage_100 * 100)
  let median_score := 90
  mean_score = 87 → median_score - mean_score = 3 :=
by
  sorry

end quiz_score_difference_l531_531271


namespace quadrilateral_is_trapezoid_cos_angle_DAB_find_t_value_l531_531175

noncomputable def A := (1 : ℝ, 0 : ℝ)
noncomputable def B := (4 : ℝ, 3 : ℝ)
noncomputable def C := (2 : ℝ, 4 : ℝ)
noncomputable def D := (0 : ℝ, 2 : ℝ)

def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def norm (u : ℝ × ℝ) : ℝ := real.sqrt (u.1^2 + u.2^2)

theorem quadrilateral_is_trapezoid :
  vector A B = (3, 3) ∧ vector D C = (2, 2) ∧ vector A B = (3/2 : ℝ) • vector D C :=
sorry

theorem cos_angle_DAB :
  let AD := vector A D in let AB := vector A B in
  cos_angle AD AB = (real.sqrt 10) / 10 :=
sorry

theorem find_t_value :
  let AB := vector A B in let OC := vector (0, 0) C in
  ∃ t : ℝ, (AB - t • OC).1 * OC.1 + (AB - t • OC).2 * OC.2 = 0 ∧ t = 9 / 10 :=
sorry

end quadrilateral_is_trapezoid_cos_angle_DAB_find_t_value_l531_531175


namespace range_of_g_l531_531888

-- We start with given condition definitions directly from the problem.
def g (x : ℝ) : ℝ := (Real.cos x) ^ 4 + (Real.sin x) ^ 2

-- The proof problem statement confirming the range of g(x)
theorem range_of_g : ∀ x : ℝ, g x ∈ set.Icc (3/4) 1 :=
by
  sorry -- proof is not required

end range_of_g_l531_531888


namespace simplify_fraction_l531_531296

theorem simplify_fraction :
  ∀ (x y : ℝ), x = 1 → y = 1 → (x^3 + y^3) / (x + y) = 1 :=
by
  intros x y hx hy
  rw [hx, hy]
  simp
  sorry

end simplify_fraction_l531_531296


namespace additional_cars_needed_to_make_multiple_of_8_l531_531919

theorem additional_cars_needed_to_make_multiple_of_8 (current_cars : ℕ) (rows_of_cars : ℕ) (next_multiple : ℕ)
  (h1 : current_cars = 37)
  (h2 : rows_of_cars = 8)
  (h3 : next_multiple = 40)
  (h4 : next_multiple ≥ current_cars)
  (h5 : next_multiple % rows_of_cars = 0) :
  (next_multiple - current_cars) = 3 :=
by { sorry }

end additional_cars_needed_to_make_multiple_of_8_l531_531919


namespace compare_fractions_l531_531100

theorem compare_fractions : (6/29 : ℚ) < (8/25 : ℚ) ∧ (8/25 : ℚ) < (11/31 : ℚ):=
by
  have h1 : (6/29 : ℚ) < (8/25 : ℚ) := sorry
  have h2 : (8/25 : ℚ) < (11/31 : ℚ) := sorry
  exact ⟨h1, h2⟩

end compare_fractions_l531_531100


namespace rectangle_area_percentage_increase_l531_531009

theorem rectangle_area_percentage_increase
  (L W : ℝ) -- Original length and width of the rectangle
  (L_new : L_new = 2 * L) -- New length of the rectangle
  (W_new : W_new = 2 * W) -- New width of the rectangle
  : (4 * L * W - L * W) / (L * W) * 100 = 300 := 
by
  sorry

end rectangle_area_percentage_increase_l531_531009


namespace projectile_area_enclosed_l531_531865

-- Define the parametric equations of motion for the projectile
def x (v t θ : ℝ) : ℝ := v * t * Real.cos θ
def y (v t θ g : ℝ) : ℝ := v * t * Real.sin θ - (1/2) * g * t^2

-- Define the time at the peak of the projectile's motion
def t_peak (v θ g : ℝ) : ℝ := v * Real.sin θ / g

-- Define the maximum x and y coordinates at the peak
def x_max (v θ g : ℝ) : ℝ := (v^2 / g) * Real.sin θ * Real.cos θ
def y_max (v θ g : ℝ) : ℝ := (v^2 / (2 * g)) * (Real.sin θ)^2

-- Define the area enclosed by the curve
def area (v g : ℝ) : ℝ := (1/2) * Real.pi * (v^2 / (2 * g)) * (v^2 / (4 * g))

-- Theorem to prove the enclosed area
theorem projectile_area_enclosed (v g : ℝ) : area v g = (Real.pi / 16) * (v^4 / g^2) := by
  sorry

end projectile_area_enclosed_l531_531865


namespace num_valid_colorings_l531_531055

-- Define the dimensions and the constraints
def board_dimension : ℕ := 8
def small_square : ℕ := 1
def large_square : ℕ := 2
def total_small_squares : ℕ := board_dimension * board_dimension / (small_square * small_square)

-- Definition for a valid 2x2 square
def valid_2x2_square (squares : Matrix (Fin 8) (Fin 8) Bool) (i j : Fin 8) : Prop :=
  (squares i j) + (squares i.succ j) + (squares i j.succ) + (squares i.succ j.succ) = 2

-- Definition for a valid board
def valid_board (squares : Matrix (Fin 8) (Fin 8) Bool) : Prop :=
  ∀ i j : Fin 7, valid_2x2_square squares i j

-- Theorem statement: the number of valid colorings is 65534
theorem num_valid_colorings : ∃ (squares : Matrix (Fin 8) (Fin 8) Bool), valid_board squares ∧ (count_valid_boards squares = 65534) :=
sorry

end num_valid_colorings_l531_531055


namespace rearrangementCount_l531_531215

-- Define the sequence as Finsets to leverage Lean's combinatorial libraries
def sequence := {1, 2, 3, 4, 5, 6}

-- Define the condition that numbers 5 and 6 must be together in any permutation
def consecutive56 (l : List Nat) : Prop :=
  ∃ a b c, l = a ++ [5, 6] ++ b ++ c ∨
           l = a ++ [6, 5] ++ b ++ c

-- Define the condition that no three consecutive terms are either increasing or decreasing
def noThreeConsecIncrDec (l : List Nat) : Prop :=
  ∀ i, i + 2 < l.length →
  ¬ (l.nthLe i (by sorry) < l.nthLe (i + 1) (by sorry) < l.nthLe (i + 2) (by sorry)) ∧
  ¬ (l.nthLe i (by sorry) > l.nthLe (i + 1) (by sorry) > l.nthLe (i + 2) (by sorry))

noncomputable def countValidPermutations : Nat :=
  (Finset.permList sequence.toList).filter (λ l, consecutive56 l ∧ noThreeConsecIncrDec l).card

theorem rearrangementCount : countValidPermutations = 20 := by sorry

end rearrangementCount_l531_531215


namespace units_digit_27_mul_46_l531_531558

theorem units_digit_27_mul_46 : (27 * 46) % 10 = 2 :=
by 
  -- Definition of units digit
  have def_units_digit :=  (n : ℕ) => n % 10

  -- Step 1: units digit of 27 is 7
  have units_digit_27 : 27 % 10 = 7 := by norm_num
  
  -- Step 2: units digit of 46 is 6
  have units_digit_46 : 46 % 10 = 6 := by norm_num

  -- Step 3: multiple the units digits
  have step3 : 7 * 6 = 42 := by norm_num

  -- Step 4: Find the units digit of 42
  have units_digit_42 : 42 % 10 = 2 := by norm_num

  exact units_digit_42

end units_digit_27_mul_46_l531_531558


namespace complex_number_in_second_quadrant_l531_531638

theorem complex_number_in_second_quadrant
    (A B : ℝ)
    (h0 : 0 < A) (h1 : A < π / 2)
    (h2 : 0 < B) (h3 : B < π / 2)
    (h4 : A + B = π / 2) :
    (cos B - sin A) < 0 ∧ (sin B  - cos A) > 0 :=
by
    sorry

end complex_number_in_second_quadrant_l531_531638


namespace distance_between_points_l531_531310

theorem distance_between_points (x : ℝ) (h : x > 0) : 
  (real.sqrt (x^2 + 2^2 + (3 - 3)^2) = real.sqrt 5) → x = 1 := 
by
  intros h_dist
  sorry

end distance_between_points_l531_531310


namespace positive_number_y_l531_531410

theorem positive_number_y (y : ℕ) (h1 : y > 0) (h2 : y^2 / 100 = 9) : y = 30 :=
by
  sorry

end positive_number_y_l531_531410


namespace smallest_positive_period_monotonic_increase_interval_find_a_l531_531178

noncomputable def f (x a : ℝ) : ℝ := sqrt 3 * sin x * cos x + cos x ^ 2 + a

theorem smallest_positive_period (a : ℝ) : 
  ∀ x, f x a = f (x + π) a :=
sorry

theorem monotonic_increase_interval (a : ℝ) (k : ℤ) : 
  ∀ x, k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 →
  ∀ y, k * π - π / 3 ≤ y ∧ y ≤ k * π + π / 6 → 
  x < y → f x a < f y a :=
sorry

theorem find_a (h_sum : (max f (- π / 6) (π / 3)) + (min f (- π / 6) (π / 3)) = 1) :
  a = - 1 / 4 :=
sorry

end smallest_positive_period_monotonic_increase_interval_find_a_l531_531178


namespace necessary_but_not_sufficient_condition_l531_531021

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ( (2*x - 1)*x = 0 → x = 0 ) ∧ ( x = 0 → (2*x - 1)*x = 0 ) :=
by
  sorry

end necessary_but_not_sufficient_condition_l531_531021


namespace product_modulo_7_l531_531084

theorem product_modulo_7 : 
  (2007 % 7 = 4) ∧ (2008 % 7 = 5) ∧ (2009 % 7 = 6) ∧ (2010 % 7 = 0) →
  (2007 * 2008 * 2009 * 2010) % 7 = 0 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end product_modulo_7_l531_531084


namespace CartesianEquationOfCurve_sum_PA_PB_l531_531615

-- Definitions for conditions
def polarEquation (p θ : ℝ) : Prop := p * sin θ ^ 2 = 4 * cos θ

def parametricLine (x y t : ℝ) : Prop :=
  x = 1 + (2 / sqrt 5) * t ∧ y = 1 + (1 / sqrt 5) * t

def pointP (x y : ℝ) : Prop := x = 1 ∧ y = 1

-- Problems to prove
theorem CartesianEquationOfCurve (p θ x y : ℝ) 
  (hEq : polarEquation p θ) 
  (hx : x = p * cos θ) 
  (hy : y = p * sin θ) : y ^ 2 = 4 * x := 
by sorry

theorem sum_PA_PB (t : ℝ) 
  (l : ∀ t, parametricLine (1 + (1 / sqrt 5) * t) (1 + (2 / sqrt 5) * t) t) 
  (pP : pointP 1 1) 
  (A B : ℝ) 
  (hPoly : t ^ 2 - 6 * sqrt 5 * t - 15 = 0) : 
  abs(A - 1) + abs(B - 1) = 4 * sqrt 15 :=
by sorry

end CartesianEquationOfCurve_sum_PA_PB_l531_531615


namespace find_unique_n_l531_531947

def P (n : ℕ) : ℕ := 
  Nat.primeFactors n |>.maximum'.get? 0

theorem find_unique_n :
  ∃! (n : ℕ), 1 < n ∧ P n = Nat.sqrt n ∧ P (n + 48) = Nat.sqrt (n + 48) := 
sorry

end find_unique_n_l531_531947


namespace find_principal_l531_531360

theorem find_principal
  (R : ℝ) (T : ℕ) (interest_less_than_principal : ℝ) : 
  R = 0.05 → 
  T = 10 → 
  interest_less_than_principal = 3100 → 
  ∃ P : ℝ, P - ((P * R * T): ℝ) = P - interest_less_than_principal ∧ P = 6200 :=
by
  sorry

end find_principal_l531_531360


namespace minimum_pipe_length_l531_531043

theorem minimum_pipe_length 
  (M S : ℝ × ℝ) 
  (horiz_dist : abs (M.1 - S.1) = 160)
  (vert_dist : abs (M.2 - S.2) = 120) :
  dist M S = 200 :=
by {
  sorry
}

end minimum_pipe_length_l531_531043


namespace series_sum_proof_l531_531915

noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, if n % 3 = 0 then 1 / (27 ^ (n / 3)) * (5 / 9) else 0

theorem series_sum_proof : infinite_series_sum = 15 / 26 :=
  sorry

end series_sum_proof_l531_531915


namespace max_distance_unit_circle_l531_531981

theorem max_distance_unit_circle (z : ℂ) (hz : |z| = 1) : ∃ w : ℂ, |z - w| = 6 :=
begin
  use 3 - 4 * complex.I,
  suffices h_distance : ∀ z : ℂ, |z| = 1 → |z - (3 - 4 * complex.I)| ≤ 6,
  { exact ⟨3 - 4 * complex.I, h_distance z hz⟩ },
  sorry -- detailed distance proof skipped
end

end max_distance_unit_circle_l531_531981


namespace no_rational_solution_l531_531587

theorem no_rational_solution 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 :=
by sorry

end no_rational_solution_l531_531587


namespace find_f_1_0_plus_f_2_0_general_form_F_l531_531580

variable {F : ℝ → ℝ → ℝ}

-- Conditions
axiom cond1 : ∀ a, F a a = a
axiom cond2 : ∀ (k a b : ℝ), F (k * a) (k * b) = k * F a b
axiom cond3 : ∀ (a1 a2 b1 b2 : ℝ), F (a1 + a2) (b1 + b2) = F a1 b1 + F a2 b2
axiom cond4 : ∀ (a b : ℝ), F a b = F b ((a + b) / 2)

-- Proof problem
theorem find_f_1_0_plus_f_2_0 : F 1 0 + F 2 0 = 0 :=
sorry

theorem general_form_F : ∀ (x y : ℝ), F x y = y :=
sorry

end find_f_1_0_plus_f_2_0_general_form_F_l531_531580


namespace eggs_left_after_capital_recovered_l531_531712

-- Conditions as definitions
def eggs_in_crate := 30
def crate_cost_dollars := 5
def price_per_egg_cents := 20

-- The amount of cents in a dollar
def cents_per_dollar := 100

-- Total cost in cents
def crate_cost_cents := crate_cost_dollars * cents_per_dollar

-- The number of eggs needed to recover the capital
def eggs_to_recover_capital := crate_cost_cents / price_per_egg_cents

-- The number of eggs left
def eggs_left := eggs_in_crate - eggs_to_recover_capital

-- The theorem stating the problem
theorem eggs_left_after_capital_recovered : eggs_left = 5 :=
by
  sorry

end eggs_left_after_capital_recovered_l531_531712


namespace surface_area_of_cube_l531_531309

theorem surface_area_of_cube (d : ℝ) (h : d = 8 * real.sqrt 3) : 
  let a := d / real.sqrt 3 in
  let surface_area := 6 * a^2 in
  surface_area = 384 :=
by
  let a := d / real.sqrt 3
  let surface_area := 6 * a^2
  sorry

end surface_area_of_cube_l531_531309


namespace condition_neither_sufficient_nor_necessary_l531_531388
-- Import necessary library

-- Define the function and conditions
def f (x a : ℝ) : ℝ := x^2 + a * x + 1

-- State the proof problem
theorem condition_neither_sufficient_nor_necessary :
  ∀ a : ℝ, (∀ x : ℝ, f x a = 0 -> x = 1/2) ↔ a^2 - 4 = 0 ∧ a ≤ -2 := sorry

end condition_neither_sufficient_nor_necessary_l531_531388


namespace empty_cistern_time_l531_531857

variable (t_fill : ℝ) (t_empty₁ : ℝ) (t_empty₂ : ℝ) (t_empty₃ : ℝ)

theorem empty_cistern_time
  (h_fill : t_fill = 3.5)
  (h_empty₁ : t_empty₁ = 14)
  (h_empty₂ : t_empty₂ = 16)
  (h_empty₃ : t_empty₃ = 18) :
  1008 / (1/t_empty₁ + 1/t_empty₂ + 1/t_empty₃) = 1.31979 := by
  sorry

end empty_cistern_time_l531_531857


namespace kite_fraction_to_BD_l531_531908

noncomputable def kiteFieldFraction (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
    (AB BC AD DC : ℝ) (angleA angleC : ℝ) :=
  AB = 100 ∧ BC = 100 ∧ AD = 150 ∧ DC = 150 ∧ angleA = π/2 ∧ angleC = π/2 →
  1

-- Define the statement for the problem
theorem kite_fraction_to_BD (A B C D : Point) :
  let AB := dist A B
  let BC := dist B C
  let AD := dist A D
  let DC := dist D C
  let angleA := ∠ A B D
  let angleC := ∠ C D B
  (AB = 100 ∧ BC = 100 ∧ AD = 150 ∧ DC = 150 ∧ angleA = π/2 ∧ angleC = π/2) →
  kiteFieldFraction A B C D AB BC AD DC angleA angleC = 1 := sorry

end kite_fraction_to_BD_l531_531908


namespace one_fifth_of_8_point_5_l531_531929

def one_fifth_of (x : ℝ) : ℝ := x / 5

theorem one_fifth_of_8_point_5 : one_fifth_of 8.5 = (17 / 10 : ℝ) :=
by
  sorry

end one_fifth_of_8_point_5_l531_531929


namespace second_derivative_at_pi_over_4_l531_531608

noncomputable def f (x : ℝ) : ℝ := x * sin x

theorem second_derivative_at_pi_over_4 :
  (deriv (deriv f)) (π / 4) = sqrt 2 - (sqrt 2 * π / 8) :=
by
  sorry

end second_derivative_at_pi_over_4_l531_531608


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531441

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531441


namespace problem_a_problem_b_l531_531389

-- Problem a
theorem problem_a (p q : ℕ) (h1 : ∃ n : ℤ, 2 * p - q = n^2) (h2 : ∃ m : ℤ, 2 * p + q = m^2) : ∃ k : ℤ, q = 2 * k :=
sorry

-- Problem b
theorem problem_b (m : ℕ) (h1 : ∃ n : ℕ, 2 * m - 4030 = n^2) (h2 : ∃ k : ℕ, 2 * m + 4030 = k^2) : (m = 2593 ∨ m = 12097 ∨ m = 81217 ∨ m = 2030113) :=
sorry

end problem_a_problem_b_l531_531389


namespace impossible_grid_configuration_l531_531673

theorem impossible_grid_configuration :
  ¬ ∃ (grid : Fin 100 × Fin 100 → Fin 3), 
  (∀ i j, 
    ∃ r c : Fin (100-2) × Fin (100-3), 
    (∀ a b, 0 ≤ a ∧ a < 3 → 0 ≤ b ∧ b < 4 →
     grid (r + ⟨a, h1⟩) (c + ⟨b, h2⟩) ∈ {0, 1, 2} ∧
     ∑ (x : Fin 3), (grid (r + ⟨x / 4, h3⟩) (c + ⟨x % 4, h4⟩) = 0) = 3 ∧
     ∑ (x : Fin 3), (grid (r + ⟨x / 4, h3⟩) (c + ⟨x % 4, h4⟩) = 1) = 4 ∧
     ∑ (x : Fin 3), (grid (r + ⟨x / 4, h3⟩) (c + ⟨x % 4, h4⟩) = 2) = 5))
  := by
  sorry

end impossible_grid_configuration_l531_531673


namespace FranSpeedToMatchJoannDistance_l531_531239

-- Define the conditions
def JoannsSpeed : ℝ := 14
def JoannsTime : ℝ := 4
def FransTime : ℝ := 2

-- Define the theorem stating the problem and required proof
theorem FranSpeedToMatchJoannDistance : 
  let JoannsDistance := JoannsSpeed * JoannsTime in 
  let FransSpeed : ℝ := JoannsDistance / FransTime in
  FransSpeed = 28 := 
by 
  sorry

end FranSpeedToMatchJoannDistance_l531_531239


namespace solution_ineq_l531_531599

noncomputable def f : ℝ → ℝ := sorry

theorem solution_ineq :
  (∀ x, f (-x) = f x) -- f(x) is an even function
  ∧ (∀ x₁ x₂ ∈ set.Iic 0, x₁ < x₂ → (f x₂ - f x₁) / (x₂ - x₁) > 0) -- f(x) is monotonically increasing on (-∞, 0]
  ∧ (f 6 = 1) -- f(6) = 1
  → {x : ℝ | f (x^2 - x) > 1} = set.Ioo (-2) 3 :=
sorry

end solution_ineq_l531_531599


namespace integral_cos_2x_eq_half_l531_531025

theorem integral_cos_2x_eq_half :
  ∫ x in (0:ℝ)..(Real.pi / 4), Real.cos (2 * x) = 1 / 2 := by
sorry

end integral_cos_2x_eq_half_l531_531025


namespace area_at_stage_4_l531_531639

-- Define the initial side length and the rule for successive squares
def initial_side_length : ℕ := 4
def side_length (n : ℕ) : ℕ := initial_side_length * 2 ^ (n - 1)

-- Compute the area given the side length
def area (side_length : ℕ) : ℕ := side_length * side_length

-- Compute the total area at a given stage
def total_area (stage : ℕ) : ℕ :=
  (List.range stage).map (λ n => area (side_length (n + 1))).sum

-- The theorem to prove
theorem area_at_stage_4 : total_area 4 = 1360 := 
by
  sorry

end area_at_stage_4_l531_531639


namespace probability_of_diff_3_is_1_over_9_l531_531423

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l531_531423


namespace calculate_p_l531_531174

variable (m n : ℤ) (p : ℤ)

theorem calculate_p (h1 : 3 * m - 2 * n = -2) (h2 : p = 3 * (m + 405) - 2 * (n - 405)) : p = 2023 := 
  sorry

end calculate_p_l531_531174


namespace exists_infinitely_many_n_l531_531390

noncomputable def squareFreeSeq : ℕ → ℕ := sorry  -- Assuming the existence of such a sequence.

theorem exists_infinitely_many_n (a : ℕ → ℕ) (h_square_free : ∀ n, is_squarefree (a n)) :
  ∃∞ n, a (n + 1) - a n = 2020 :=
sorry

end exists_infinitely_many_n_l531_531390


namespace compound_interest_correct_l531_531642

-- Definitions and conditions
def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

def compound_amount (P r n t : ℝ) : ℝ := P * (1 + r / n)^(n * t)

theorem compound_interest_correct :
  let P := 50 / (0.05 + 0.06) in
  let A1 := compound_amount P 0.05 4 1 in
  let A2 := compound_amount A1 0.06 4 1 in
  A2 - P = 52.73 := 
by sorry

end compound_interest_correct_l531_531642


namespace line_AB_parallel_xOz_l531_531228

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (a b : Point3D) : Point3D :=
  { x := b.x - a.x, y := b.y - a.y, z := b.z - a.z }

def dot_product (v₁ v₂ : Point3D) : ℝ :=
  v₁.x * v₂.x + v₁.y * v₂.y + v₁.z * v₂.z

def is_parallel_to_xOz (v : Point3D) : Prop :=
  dot_product v {x := 0, y := 1, z := 0} = 0

theorem line_AB_parallel_xOz :
  let A := { x := 1, y := 3, z := 0 }
  let B := { x := 0, y := 3, z := -1 }
  let AB := vector_sub A B
  is_parallel_to_xOz AB :=
by
  let A := { x := 1, y := 3, z := 0 }
  let B := { x := 0, y := 3, z := -1 }
  let AB := vector_sub A B
  change is_parallel_to_xOz AB
  sorry

end line_AB_parallel_xOz_l531_531228


namespace quadrilateral_with_equal_angles_not_rectangle_l531_531002

/--
A quadrilateral with all interior angles equal is not necessarily a rectangle.
-/
theorem quadrilateral_with_equal_angles_not_rectangle :
  ∃ (Q : Type) [quadrilateral Q], (∀ θ ∈ interior_angles Q, θ = 90) → ¬ is_rectangle Q :=
by sorry

end quadrilateral_with_equal_angles_not_rectangle_l531_531002


namespace days_x_worked_l531_531378

def x_work_rate (W : ℝ) : ℝ := W / 40
def y_work_rate (W : ℝ) : ℝ := W / 20
def work_in_days (days : ℝ) (rate : ℝ) : ℝ := days * rate

theorem days_x_worked (W : ℝ) (d : ℝ) : 
  work_in_days d (x_work_rate W) + work_in_days 16 (y_work_rate W) = W → d = 8 := 
by
  sorry

end days_x_worked_l531_531378


namespace fifth_group_pythagorean_triples_l531_531636

theorem fifth_group_pythagorean_triples :
  ∃ (a b c : ℕ), (a, b, c) = (11, 60, 61) ∧ a^2 + b^2 = c^2 :=
by
  use 11, 60, 61
  sorry

end fifth_group_pythagorean_triples_l531_531636


namespace jerry_bought_one_pound_of_pasta_sauce_l531_531675

-- Definitions of the given conditions
def cost_mustard_oil_per_liter : ℕ := 13
def liters_mustard_oil : ℕ := 2
def cost_pasta_per_pound : ℕ := 4
def pounds_pasta : ℕ := 3
def cost_pasta_sauce_per_pound : ℕ := 5
def leftover_amount : ℕ := 7
def initial_amount : ℕ := 50

-- The goal to prove
theorem jerry_bought_one_pound_of_pasta_sauce :
  (initial_amount - leftover_amount - liters_mustard_oil * cost_mustard_oil_per_liter 
  - pounds_pasta * cost_pasta_per_pound) / cost_pasta_sauce_per_pound = 1 :=
by
  sorry

end jerry_bought_one_pound_of_pasta_sauce_l531_531675


namespace arrange_plants_together_l531_531510

open Nat

theorem arrange_plants_together (basil tomato : ℕ) :
  basil = 6 → tomato = 3 → (∃ ways : ℕ, ways = factorial (basil + 1) * factorial tomato ∧ ways = 30240) :=
by 
  intros hB hT 
  use (factorial 7 * factorial 3) 
  split
  { simp [hB, hT] } 
  { norm_num }
  done

end arrange_plants_together_l531_531510


namespace expression_evaluation_l531_531941

theorem expression_evaluation (a : ℕ) (h : a = 1580) : 
  2 * a - ((2 * a - 3) / (a + 1) - (a + 1) / (2 - 2 * a) - (a^2 + 3) / 2) * ((a^3 + 1) / (a^2 - a)) + 2 / a = 2 := 
sorry

end expression_evaluation_l531_531941


namespace all_selected_prob_l531_531700

def probability_of_selection (P_ram P_ravi P_raj : ℚ) : ℚ :=
  P_ram * P_ravi * P_raj

theorem all_selected_prob :
  let P_ram := 2/7
  let P_ravi := 1/5
  let P_raj := 3/8
  probability_of_selection P_ram P_ravi P_raj = 3/140 := by
  sorry

end all_selected_prob_l531_531700


namespace second_yellow_probability_l531_531076

-- Define the conditions in Lean
def BagA : Type := {marble : Int // marble ≥ 0}
def BagB : Type := {marble : Int // marble ≥ 0}
def BagC : Type := {marble : Int // marble ≥ 0}
def BagD : Type := {marble : Int // marble ≥ 0}

noncomputable def marbles_in_A := 4 + 5 + 2
noncomputable def marbles_in_B := 7 + 5
noncomputable def marbles_in_C := 3 + 7
noncomputable def marbles_in_D := 8 + 2

-- Probabilities of drawing specific colors from Bag A
noncomputable def prob_white_A := 4 / 11
noncomputable def prob_black_A := 5 / 11
noncomputable def prob_red_A := 2 / 11

-- Probabilities of drawing a yellow marble from Bags B, C and D
noncomputable def prob_yellow_B := 7 / 12
noncomputable def prob_yellow_C := 3 / 10
noncomputable def prob_yellow_D := 8 / 10

-- Expected probability that the second marble is yellow
noncomputable def prob_second_yellow : ℚ :=
  (prob_white_A * prob_yellow_B) + (prob_black_A * prob_yellow_C) + (prob_red_A * prob_yellow_D)

/-- Prove that the total probability the second marble drawn is yellow is 163/330. -/
theorem second_yellow_probability :
  prob_second_yellow = 163 / 330 := sorry

end second_yellow_probability_l531_531076


namespace right_triangle_angle_bisector_circumcircle_l531_531577

universe u
variables {α : Type u} [EuclideanGeometry α]
variables {A B C K L : α}

open EuclideanGeometry

theorem right_triangle_angle_bisector_circumcircle 
  (hABC: right_triangle A B C)
  (hACB: ∠ACB = 90°)
  (hBK_bisector: angle_bisector B K C A)
  (hCircumcircle: cyclic_quad A K B L)
  (hLC_on_BC: L ∈ line_through B C) :
  ((dist C B) + (dist C L) = dist A B) :=
sorry

end right_triangle_angle_bisector_circumcircle_l531_531577


namespace imaginary_part_of_square_sub_one_l531_531252

noncomputable def imaginary_unit : ℂ :=
{ re := 0, im := 1 }

theorem imaginary_part_of_square_sub_one : 
  ∀ (i : ℂ), i = imaginary_unit → (1 - i)^2.im = -2 :=
by
  intros i hi
  rw hi
  sorry

end imaginary_part_of_square_sub_one_l531_531252


namespace symmetric_point_l531_531307

theorem symmetric_point : ∃ (x0 y0 : ℝ), 
  (x0 = -6 ∧ y0 = -3) ∧ 
  (∃ (m1 m2 : ℝ), 
    m1 = -1 ∧ 
    m2 = (y0 - 2) / (x0 + 1) ∧ 
    m1 * m2 = -1) ∧ 
  (∃ (x_mid y_mid : ℝ), 
    x_mid = (x0 - 1) / 2 ∧ 
    y_mid = (y0 + 2) / 2 ∧ 
    x_mid + y_mid + 4 = 0) := 
sorry

end symmetric_point_l531_531307


namespace inscribed_circle_quadrilateral_AMKP_l531_531152

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def altitude_foot (B : Point) (A C : Line) : Point := sorry
noncomputable def incircle_touch_point (incircle : Circle) (A C : Line) : Point := sorry
noncomputable def parallel_line_intersect_segment (K : Point) (MH : Line) (MN : Segment) : Point := sorry

variables {A B C M N H K P : Point}
variables {ABC : Triangle}
variables {incircle : Circle}

-- Assuming the given conditions
axiom h_triangle_acute (ABC : Triangle (A, B, C)) (h_acute : acute_angle ABC) : Prop
axiom h_side_inequality (h_ABC : Triangle A B C) (h_side : AB < BC) : Prop
axiom h_midpoints (A B C M N: Point) (h_midpoints_M : midpoint A B = M) (h_midpoints_N : midpoint A C = N) : Prop
axiom h_altitude_foot (A B C H: Point) (h_altitude : altitude_foot B (line A C) = H) : Prop
axiom h_incircle_touch (incircle : Circle) (A C K: Point) (h_touch : incircle_touch_point incircle (line A C) = K) : Prop
axiom h_parallel_line (K: Point) (line_MH: Line) (segment_MN: Segment) (P: Point) (h_intersect : parallel_line_intersect_segment K line_MH segment_MN = P) : Prop

theorem inscribed_circle_quadrilateral_AMKP :
  ∀ (ABC : Triangle) (incircle : Circle),
    (h_triangle_acute ABC) ∧ (h_side_inequality ABC) ∧
    (h_midpoints ABC) ∧ (h_altitude_foot ABC) ∧
    (h_incircle_touch incircle) ∧ (h_parallel_line incircle) →
    ∃ (circle : Circle), is_inscribed_circle circle (Quadrilateral A M K P) := 
  sorry

end inscribed_circle_quadrilateral_AMKP_l531_531152


namespace value_of_a_l531_531135

noncomputable def M : set ℝ → ℝ 
| s := sup (sin '' s)

theorem value_of_a (a : ℝ) (h : M (set.Icc 0 a) = 2 * M (set.Icc a (2 * a))):
 a = (5 * Real.pi) / 6 ∨ a = (13 * Real.pi) / 12 :=
begin
  sorry
end

end value_of_a_l531_531135


namespace P_symmetry_l531_531873

noncomputable def P : ℕ → (ℝ → ℝ → ℝ → ℝ)
| 0 := λ x y z, 1
| (m+1) := λ x y z, (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem P_symmetry (m : ℕ) (x y z : ℝ) :
  P m x y z = P m y x z ∧ P m x y z = P m x z y ∧ P m x y z = P m z y x :=
sorry

end P_symmetry_l531_531873


namespace minimum_travel_time_l531_531109

theorem minimum_travel_time (a b d t : ℝ) 
  (h1 : ∀ t, Dolly_speed = 6 ∧ Molly_speed = 6 ∧ Polly_speed = 6)
  (h2 : Motorcycle_speed = 90)
  (h3 : Total_distance = 135)
  (h4 : Motorcycle_capacity ≤ 2)
  (h5 : Motorbike_cannot_drive_by_itself : ∀ t, 0 ≤ t)
  (distance_relation1 : 7 * a = d)
  (distance_relation2 : 7 * b = d)
  (distance_sum : 9 * a = 135)
  (dolly_time : dolly_total_distance = a + 7 * a + a ∧ dolly_total_time = 3.83) :
  t < 3.9 := by
  sorry

end minimum_travel_time_l531_531109


namespace passing_percentage_correct_l531_531212

-- Define the conditions
def max_marks : ℕ := 500
def candidate_marks : ℕ := 180
def fail_by : ℕ := 45

-- Define the passing_marks based on given conditions
def passing_marks : ℕ := candidate_marks + fail_by

-- Theorem to prove: the passing percentage is 45%
theorem passing_percentage_correct : 
  (passing_marks / max_marks) * 100 = 45 := 
sorry

end passing_percentage_correct_l531_531212


namespace profit_division_ratio_l531_531704

-- Definitions of the given conditions
def praveen_initial_capital : ℝ := 3780
def praveen_duration_months : ℝ := 12
def hari_initial_capital : ℝ := 9720
def hari_duration_months : ℝ := 7

-- Stating the problem: Proving the profit division ratio
theorem profit_division_ratio 
  (p_initial : ℝ := praveen_initial_capital)
  (p_months : ℝ := praveen_duration_months)
  (h_initial : ℝ := hari_initial_capital)
  (h_months : ℝ := hari_duration_months) :
  (p_initial * p_months) / real.gcd (p_initial * p_months) (h_initial * h_months) : (h_initial * h_months) / real.gcd (p_initial * p_months) (h_initial * h_months) = 2 / 3 :=
by
  sorry

end profit_division_ratio_l531_531704


namespace probability_of_difference_three_l531_531486

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l531_531486


namespace arith_seq_100th_term_l531_531154

noncomputable def arithSeq (a : ℤ) (n : ℕ) : ℤ :=
  a - 1 + (n - 1) * ((a + 1) - (a - 1))

theorem arith_seq_100th_term (a : ℤ) : arithSeq a 100 = 197 := by
  sorry

end arith_seq_100th_term_l531_531154


namespace dice_rolls_diff_by_3_probability_l531_531451

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l531_531451


namespace albums_either_but_not_both_l531_531068

-- Definition of the problem conditions
def shared_albums : Nat := 11
def andrew_total_albums : Nat := 20
def bob_exclusive_albums : Nat := 8

-- Calculate Andrew's exclusive albums
def andrew_exclusive_albums : Nat := andrew_total_albums - shared_albums

-- Question: Prove the total number of albums in either Andrew's or Bob's collection but not both is 17
theorem albums_either_but_not_both : 
  andrew_exclusive_albums + bob_exclusive_albums = 17 := 
by
  sorry

end albums_either_but_not_both_l531_531068


namespace B_days_to_finish_work_alone_l531_531851

-- Define the conditions
def A_work_rate_per_day : ℝ := 1 / 10
def total_wages : ℝ := 3300
def A_wages : ℝ := 1980

-- Define the problem
theorem B_days_to_finish_work_alone (B_work_rate_per_day : ℝ) (B_days : ℝ) : B_days = 15 :=
  let B_work_rate := 1 / B_days in
  let combined_work_rate := A_work_rate_per_day + B_work_rate in
  let A_work_ratio := A_work_rate_per_day / combined_work_rate in
  let A_wages_ratio := A_wages / total_wages in
  have : A_work_ratio = A_wages_ratio := by {
    sorry
  },
  sorry

end B_days_to_finish_work_alone_l531_531851


namespace chord_through_midpoint_l531_531149

theorem chord_through_midpoint (P : Point) (A B : Point) (x1 y1 x2 y2 : ℝ)
  (parabola_def : ∀ (x y : ℝ), y^2 = 4 * x) 
  (midpoint_def : P = ⟨(x1 + x2) / 2, (y1 + y2) / 2⟩) 
  (P_coords : P = ⟨2, 1⟩) 
  (A_coords : A = ⟨x1, y1⟩) 
  (B_coords : B = ⟨x2, y2⟩) :
  equation_of_line_through_points A B = "2x - y - 3 = 0" :=
sorry

end chord_through_midpoint_l531_531149


namespace limit_bounds_Cn_l531_531946

theorem limit_bounds_Cn :
  (∀ n : ℕ, n ≥ 2 → (2 ^ (0.1887 * n^2) ≤ C(n) ∧ C(n) ≤ 2 ^ (0.6571 * n^2))) →
  (∀ n : ℕ, n ≥ 2 → (0.1887 ≤ (Real.log (C(n)) / n^2) / Real.log 2 ∧ (Real.log (C(n)) / n^2) / Real.log 2 ≤ 0.6571)) :=
by
  -- proof here
  sorry

end limit_bounds_Cn_l531_531946


namespace binomial_510_510_l531_531905

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem binomial_510_510 : binomial 510 510 = 1 :=
  by
    -- Skip the proof with sorry
    sorry

end binomial_510_510_l531_531905


namespace find_a_value_of_a_l531_531627

noncomputable def possible_value_of_a : ℕ → Prop :=
  λ a, a > 0 ∧ let r1 := 2, r2 := 8, d := Real.sqrt (16 + a^2) in
    6 < d ∧ d < 10

theorem find_a_value_of_a :
  ∃ (a : ℕ), possible_value_of_a a :=
begin
  use 5,
  unfold possible_value_of_a,
  split,
  { exact Nat.zero_lt_succ _ },
  { dsimp,
    split,
    { norm_num },
    { norm_num } }
end

end find_a_value_of_a_l531_531627


namespace remainingMealsForChildren_l531_531042

noncomputable def totalMealsForAdults : ℕ := 70
noncomputable def totalMealsForChildren : ℕ := 90
noncomputable def initialAdultsFed : ℕ := 42
noncomputable def remainingMealsForAdults : ℕ := totalMealsForAdults - initialAdultsFed
noncomputable def consumptionRatio : ℚ := 7 / 9

theorem remainingMealsForChildren :
  (remainingMealsForAdults : ℚ) * (consumptionRatio.denom : ℚ) / (consumptionRatio.num : ℚ) = 36 :=
by
  sorry

end remainingMealsForChildren_l531_531042


namespace identify_heaviest_and_lightest_l531_531812

theorem identify_heaviest_and_lightest (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ weighings : Fin 13 → (Fin 10 × Fin 10),
  (let outcomes := fun w ℕ => ite (coins (weighings w).fst > coins (weighings w).snd) (weighings w).fst (weighings w).snd,
  max_coin := nat.rec_on 12 (outcomes 0) (λ n max_n, if coins (outcomes (succ n)) > coins max_n then outcomes (succ n) else max_n),
  min_coin := nat.rec_on 12 (outcomes 0) (λ n min_n, if coins (outcomes (succ n)) < coins min_n then outcomes (succ n) else min_n))
  (∃ max_c : Fin 10, ∃ min_c : Fin 10, max_c ≠ min_c ∧ max_c = Some max_coin ∧ min_c = Some min_coin) :=
sorry

end identify_heaviest_and_lightest_l531_531812


namespace max_bag_weight_l531_531265

-- Let's define the conditions first
def green_beans_weight := 4
def milk_weight := 6
def carrots_weight := 2 * green_beans_weight
def additional_capacity := 2

-- The total weight of groceries
def total_groceries_weight := green_beans_weight + milk_weight + carrots_weight

-- The maximum weight the bag can hold is the total weight of groceries plus the additional capacity
theorem max_bag_weight : (total_groceries_weight + additional_capacity) = 20 := by
  sorry

end max_bag_weight_l531_531265


namespace sasha_remainder_l531_531785

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l531_531785


namespace probability_diff_by_3_l531_531484

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l531_531484


namespace exists_k_le_n_l531_531018

theorem exists_k_le_n 
  {n : ℕ} 
  (h : n > 0) 
  (a : Fin n → ℝ) : 
  ∃ k : Fin n, 
  ∀ i : Fin (k + 1), 
    (1 / (i + 1 + 1).val) * (List.range (i + 1 + 1).val).map (λ j => a (Fin.mk (k.val - j) sorry)).sum ≤ 
    (List.range n.val).map (λ j => a (Fin.mk j sorry)).sum / n :=
by
  sorry

end exists_k_le_n_l531_531018


namespace probability_differ_by_three_is_one_sixth_l531_531459

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l531_531459


namespace minutes_after_2017_is_0554_l531_531830

theorem minutes_after_2017_is_0554 :
  let initial_time := (20, 17) -- time in hours and minutes
  let total_minutes := 2017
  let hours_passed := total_minutes / 60
  let minutes_passed := total_minutes % 60
  let days_passed := hours_passed / 24
  let remaining_hours := hours_passed % 24
  let resulting_hours := (initial_time.fst + remaining_hours) % 24
  let resulting_minutes := initial_time.snd + minutes_passed
  let final_hours := if resulting_minutes >= 60 then resulting_hours + 1 else resulting_hours
  let final_minutes := if resulting_minutes >= 60 then resulting_minutes - 60 else resulting_minutes
  final_hours % 24 = 5 ∧ final_minutes = 54 := by
  sorry

end minutes_after_2017_is_0554_l531_531830


namespace same_day_ticket_price_l531_531061

theorem same_day_ticket_price (adv_ticket_price : ℝ) (total_tickets : ℕ) (total_receipts : ℝ) (adv_tickets_sold : ℕ) 
  (h1 : adv_ticket_price = 20) (h2 : total_tickets = 60) (h3 : total_receipts = 1600) (h4 : adv_tickets_sold = 20) :
  ∃ (x : ℝ), 40 * x = 1200 ∧ x = 30 :=
by
  -- Let x be the cost of same-day tickets
  let x := 30
  have h5 : 40 * x = 1200 := by
    -- just write x explicitly to avoid noncomputable issue
    norm_num
  use x
  exact ⟨h5, rfl⟩

end same_day_ticket_price_l531_531061


namespace sasha_remainder_l531_531766

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l531_531766


namespace angles_in_second_quadrant_l531_531505

open Real

theorem angles_in_second_quadrant:
  ∀ (θ: ℝ) (k: ℤ), θ = 160 ∨ θ = 480 ∨ θ = -960 →
    ∃ k: ℤ, 2 * k * π + π / 2 < θ ∧ θ < 2 * k * π + π 
:= by
  intros θ k h
  cases h
  case or.inl { 
    -- proof for 160°
    use 0
    sorry
  }
  case or.inr h₁ {
    cases h₁
    case or.inl {
      -- proof for 480°
      use 1
      sorry
    }
    case or.inr {
      -- proof for -960°
      use -3
      sorry
    }
  }

end angles_in_second_quadrant_l531_531505


namespace maximize_sum_of_arithmetic_sequence_l531_531583

theorem maximize_sum_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_common_diff_negative : d < 0)
  (h_modulus : |a 2 + 2 * d| = |a 2 + 8 * d|):
  ∃ n, (n = 5 ∨ n = 6) ∧ ∀ m, (m ≤ n → sum (range m) a ≤ sum (range n) a) := 
sorry

end maximize_sum_of_arithmetic_sequence_l531_531583


namespace miles_remaining_l531_531843

theorem miles_remaining (total_miles driven_miles : ℕ) (h1 : total_miles = 1200) (h2 : driven_miles = 768) :
    total_miles - driven_miles = 432 := by
  sorry

end miles_remaining_l531_531843


namespace distance_from_origin_l531_531211

theorem distance_from_origin (x y z : ℤ) (h : x = -20 ∧ y = 21 ∧ z = 15) : 
  real.sqrt (x^2 + y^2 + z^2) = real.sqrt 1066 :=
by
  obtain ⟨hx, hy, hz⟩ := h
  rw [hx, hy, hz]
  simp
  sorry

end distance_from_origin_l531_531211


namespace problem_statement_l531_531955

open Set

variable (U M N : Set ℤ)

def Complement (A B : Set ℤ) : Set ℤ := { x | x ∈ A ∧ x ∉ B }

theorem problem_statement : 
  U = {0, -1, -2, -3, -4} → 
  M = {0, -1, -2} → 
  N = {0, -3, -4} → 
  (Complement U M ∩ N) = {-3, -4} :=
by
  intros hU hM hN
  rw [hU, hM, hN]
  sorry

end problem_statement_l531_531955


namespace samantha_eggs_left_l531_531716

variables (initial_eggs : ℕ) (total_cost price_per_egg : ℝ)

-- Conditions
def samantha_initial_eggs : initial_eggs = 30 := sorry
def samantha_total_cost : total_cost = 5 := sorry
def samantha_price_per_egg : price_per_egg = 0.20 := sorry

-- Theorem to prove:
theorem samantha_eggs_left : 
  initial_eggs - (total_cost / price_per_egg) = 5 := 
  by
  rw [samantha_initial_eggs, samantha_total_cost, samantha_price_per_egg]
  -- Completing the arithmetic proof
  rw [Nat.cast_sub (by norm_num), Nat.cast_div (by norm_num), Nat.cast_mul (by norm_num)]
  norm_num
  sorry

end samantha_eggs_left_l531_531716


namespace pies_sold_each_day_l531_531868

theorem pies_sold_each_day (total_pies: ℕ) (days_in_week: ℕ) 
  (h1: total_pies = 56) (h2: days_in_week = 7) : 
  total_pies / days_in_week = 8 :=
by
  sorry

end pies_sold_each_day_l531_531868


namespace carol_invitations_l531_531899

-- Definitions: each package has 3 invitations, Carol bought 2 packs, and Carol needs 3 extra invitations.
def invitations_per_pack : ℕ := 3
def packs_bought : ℕ := 2
def extra_invitations : ℕ := 3

-- Total number of invitations Carol will have
def total_invitations : ℕ := (packs_bought * invitations_per_pack) + extra_invitations

-- Statement to prove: Carol wants to invite 9 friends.
theorem carol_invitations : total_invitations = 9 := by
  sorry  -- Proof omitted

end carol_invitations_l531_531899


namespace identify_heaviest_and_lightest_l531_531814

theorem identify_heaviest_and_lightest (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ weighings : Fin 13 → (Fin 10 × Fin 10),
  (let outcomes := fun w ℕ => ite (coins (weighings w).fst > coins (weighings w).snd) (weighings w).fst (weighings w).snd,
  max_coin := nat.rec_on 12 (outcomes 0) (λ n max_n, if coins (outcomes (succ n)) > coins max_n then outcomes (succ n) else max_n),
  min_coin := nat.rec_on 12 (outcomes 0) (λ n min_n, if coins (outcomes (succ n)) < coins min_n then outcomes (succ n) else min_n))
  (∃ max_c : Fin 10, ∃ min_c : Fin 10, max_c ≠ min_c ∧ max_c = Some max_coin ∧ min_c = Some min_coin) :=
sorry

end identify_heaviest_and_lightest_l531_531814


namespace polynomial_divisibility_P_l531_531019

noncomputable def P (ϕ x : ℂ) (n : ℕ) : ℂ :=
  (Real.cos ϕ + x * Real.sin ϕ)^n - Real.cos (n * ϕ) - x * Real.sin (n * ϕ)

theorem polynomial_divisibility_P (ϕ : ℂ) (n : ℕ) :
  P ϕ Complex.I n = 0 ∧ P ϕ (-Complex.I) n = 0 → ∃ R (x : ℂ), P ϕ x n = (x^2 + 1) * R :=
sorry

end polynomial_divisibility_P_l531_531019


namespace average_of_solutions_eq_one_l531_531617

theorem average_of_solutions_eq_one (c : ℝ) (h : ∃ x1 x2 : ℝ, 3 * x1^2 - 6 * x1 + c = 0 ∧ 3 * x2^2 - 6 * x2 + c = 0 ∧ x1 ≠ x2) : 
  (r1 r2 : ℝ) (hr1 : 3 * r1^2 - 6 * r1 + c = 0) (hr2 : 3 * r2^2 - 6 * r2 + c = 0) -> 
  1 = (r1 + r2) / 2 := 
by 
  sorry

end average_of_solutions_eq_one_l531_531617


namespace circle_equation_determine_a_l531_531217

noncomputable theory

open Real

def point := (ℝ × ℝ)

def circle (D E F : ℝ) := ∀ p : point, (p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0)

def distance (p1 p2 : point) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def orthogonal (p1 p2 p3 : point) : Prop := (p2.1 - p1.1) * (p3.1 - p1.1) + (p2.2 - p1.2) * (p3.2 - p1.2) = 0

theorem circle_equation
  (P Q R : point)
  (C : D E F : ℝ)
  (h1 : P = (3 + 2*sqrt 2, 0))
  (h2 : Q = (3 - 2*sqrt 2, 0))
  (h3 : R = (0, 1))
  (h_circle : circle D E F)
  :
  circle D E F :=
by
  sorry

theorem determine_a
  (a : ℝ)
  (A B O : point)
  (circle_eq : x^2 + y^2 - 6 * x - 2 * y + 1 = 0)
  (line_eq : A.1 - A.2 + a = 0 ∧ B.1 - B.2 + a = 0)
  (orth : orthogonal O A B)
  :
  a = 1 ∨ a = -5 :=
by
  sorry

end circle_equation_determine_a_l531_531217


namespace exists_sequence_occuring_M_times_l531_531155

theorem exists_sequence_occuring_M_times
  (n : ℕ) (hn : n > 5)
  (strip : List ℕ) (hstrip : ∀ i, strip.nth i = ((i < n) : ℤ))
  (M : ℕ)
  (hmax_seq : ∀ i, (strip.slice i (i+n)).count 0 = n - 2 ∧ (strip.slice i (i+n)).count 1 = 2 → M := strip.nth_le i)
  (hmin_seq : ∀ i, (strip.slice i (i+n)).count 0 = n - 2 ∧ (strip.slice i (i+n)).count 1 = 2 := 0)
  :
  ∃ seq, (strip.indexes seq).length = M := sorry

end exists_sequence_occuring_M_times_l531_531155


namespace total_race_distance_l531_531284

theorem total_race_distance :
  let sadie_time := 2
  let sadie_speed := 3
  let ariana_time := 0.5
  let ariana_speed := 6
  let total_time := 4.5
  let sarah_speed := 4
  let sarah_time := total_time - sadie_time - ariana_time
  let sadie_distance := sadie_speed * sadie_time
  let ariana_distance := ariana_speed * ariana_time
  let sarah_distance := sarah_speed * sarah_time
  let total_distance := sadie_distance + ariana_distance + sarah_distance
  total_distance = 17 :=
by
  sorry

end total_race_distance_l531_531284


namespace combined_area_is_correct_l531_531628

def tract1_length := 300
def tract1_width  := 500
def tract2_length := 250
def tract2_width  := 630
def tract3_length := 350
def tract3_width  := 450
def tract4_length := 275
def tract4_width  := 600
def tract5_length := 325
def tract5_width  := 520

def area (length width : ℕ) : ℕ := length * width

theorem combined_area_is_correct :
  area tract1_length tract1_width +
  area tract2_length tract2_width +
  area tract3_length tract3_width +
  area tract4_length tract4_width +
  area tract5_length tract5_width = 799000 :=
by
  sorry

end combined_area_is_correct_l531_531628


namespace area_ratio_XYP_XZP_l531_531230

theorem area_ratio_XYP_XZP (X Y Z P : Type) [normed_field X] [normed_space X Y] [normed_space X Z] :
  ∀ (XY XZ YZ XP : ℝ),
  XY = 20 → XZ = 30 → YZ = 25 →
  ∃ (YP ZP : ℝ), XP = YP + ZP ∧ (YP / ZP = XY / XZ) →
  (YP / ZP = (2 / 3)) →
  ∀ (area_XYP area_XZP : ℝ), (area_XYP / area_XZP = 2 / 3) :=
by
  intros XY XZ YZ XP hXY hXZ hYZ YP ZP hYPZP_ratio hYPZP_fraction area_XYP area_XZP
  -- skipping the proof here
  sorry

end area_ratio_XYP_XZP_l531_531230


namespace probability_of_diff_3_is_1_over_9_l531_531430

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l531_531430


namespace commission_percentage_proof_l531_531051

-- Let's define the problem conditions in Lean

-- Condition 1: Commission on first Rs. 10,000
def commission_first_10000 (sales : ℕ) : ℕ :=
  if sales ≤ 10000 then
    5 * sales / 100
  else
    500

-- Condition 2: Amount remitted to company after commission
def amount_remitted (total_sales : ℕ) (commission : ℕ) : ℕ :=
  total_sales - commission

-- Condition 3: Function to calculate commission on exceeding amount
def commission_exceeding (sales : ℕ) (x : ℕ) : ℕ :=
  x * sales / 100

-- The main hypothesis as per the given problem
def correct_commission_percentage (total_sales : ℕ) (remitted : ℕ) (x : ℕ) :=
  commission_first_10000 10000 + commission_exceeding (total_sales - 10000) x
  = total_sales - remitted

-- Problem statement to prove the percentage of commission on exceeding Rs. 10,000 is 4%
theorem commission_percentage_proof : correct_commission_percentage 32500 31100 4 := 
  by sorry

end commission_percentage_proof_l531_531051


namespace exists_diagonal_with_triangle_area_leq_sixth_l531_531706

-- Define the problem
theorem exists_diagonal_with_triangle_area_leq_sixth {H : hexagon} (h_convex: convex H) :
  ∃ d : diagonal H, let T := triangle_cut_off_by_diagonal H d in area T ≤ (area H) / 6 := 
sorry

end exists_diagonal_with_triangle_area_leq_sixth_l531_531706


namespace calculate_selling_price_l531_531720

-- Define the conditions
def purchase_price : ℝ := 900
def repair_cost : ℝ := 300
def gain_percentage : ℝ := 0.10

-- Define the total cost
def total_cost : ℝ := purchase_price + repair_cost

-- Define the gain
def gain : ℝ := gain_percentage * total_cost

-- Define the selling price
def selling_price : ℝ := total_cost + gain

-- The theorem to prove
theorem calculate_selling_price : selling_price = 1320 := by
  sorry

end calculate_selling_price_l531_531720


namespace integer_solution_exists_l531_531277

theorem integer_solution_exists (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1)/(m * m - n) :=
by {
  cases' (d : ℤ) with
  | basic d =>  -- handle general integer case
    let m := d + 2
    let n := d^2 + 3*d + 3
    use m, n
    sorry,  -- Proof goes here
  | d_eq_minus_one =>  -- handle case where d = -1
    use 1, 0
    sorry  -- Proof goes here
}

end integer_solution_exists_l531_531277


namespace sasha_remainder_20_l531_531779

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l531_531779


namespace max_sqrt_distance_l531_531974

theorem max_sqrt_distance (x y : ℝ) 
  (h : x^2 + y^2 - 4 * x - 4 * y + 6 = 0) : 
  ∃ z, z = 3 * Real.sqrt 2 ∧ ∀ w, w = Real.sqrt (x^2 + y^2) → w ≤ z :=
sorry

end max_sqrt_distance_l531_531974


namespace systematic_sampling_five_from_fifty_l531_531137

theorem systematic_sampling_five_from_fifty :
  ∃ (start interval : ℕ) (selected : List ℕ),
    start = 3 ∧
    interval = 10 ∧
    (∀ i, i < 5 → selected.get i = some (start + i * interval)) ∧
    selected = [3, 13, 23, 33, 43] :=
by
  sorry

end systematic_sampling_five_from_fifty_l531_531137


namespace area_of_triangle_OAB_eq_9_over_4_l531_531247

noncomputable def parabola_focus : Point := ⟨3/4, 0⟩

def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 3 * x

def line_equation (x y : ℝ) : Prop :=
  x = sqrt 3 * y + 3 / 4

def intersection_points : set (ℝ × ℝ) :=
  {p | parabola_equation p.1 p.2 ∧ line_equation p.1 p.2}

def O : Point := ⟨0, 0⟩

def area_of_triangle_OAB : ℝ :=
  -- S_△OAB = 3/8 * sqrt ((y1 + y2)^2 + 9)
  let y1 := some_intersection_point_of conjunction_of parabola_equation_and line_equation.1 in
  let y2 := some_intersection_point_of conjunction_of parabola_equation_and line_equation.2 in
  (3 / 8) * sqrt ((3 * sqrt 3)^2 + 9)

theorem area_of_triangle_OAB_eq_9_over_4 : area_of_triangle_OAB = 9 / 4 := sorry

end area_of_triangle_OAB_eq_9_over_4_l531_531247


namespace parallel_line_eq_l531_531183

def parallel_lines_distance_equation (m a b d : ℝ) (y₁ y₂ : ℝ → ℝ) : Prop :=
  y₁ x ≡ m * x + a ∧ y₂ x ≡ m * x + b ∧ |b - a| = d * sqrt (m^2 + 1)

theorem parallel_line_eq {
  (m : ℝ) 
  (c₁ : ℝ) 
  (c_diff : ℝ)
  (d : ℝ) 
  (y : ℝ → ℝ)
  }:
  (m = 3/2) →
  (c₁ = 12) →
  (d = 8) →
  (c_diff = 4 * real.sqrt 13) →
  parallel_lines_distance_equation m c₁ (c₁ + c_diff) d y (yλ x : ℝ, m*x + (c₁ + c_diff)) ∧
  parallel_lines_distance_equation m c₁ (c₁ - c_diff) d y (λ x : ℝ, m*x + (c₁ - c_diff))

end parallel_line_eq_l531_531183


namespace part1_values_of_n_eq_Pn_l531_531562

noncomputable def digits_product (n : ℕ) : ℕ :=
  if n = 0 then 0 else n.to_digits.reduce (λ x y => x * y)

theorem part1_values_of_n_eq_Pn :
  {n : ℕ | n > 0 ∧ n = digits_product n} = {1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end part1_values_of_n_eq_Pn_l531_531562


namespace transformed_roots_polynomial_l531_531695

theorem transformed_roots_polynomial :
  let r1 r2 r3 : ℝ in
  (∀ x : ℝ, (x^3 - x^2 + 3 * x - 7 = 0) ↔ (x = r1 ∨ x = r2 ∨ x = r3)) →
  (∀ y : ℝ, (y^3 - 3 * y^2 + 27 * y - 189 = 0) ↔ (y = 3 * r1 ∨ y = 3 * r2 ∨ y = 3 * r3)) :=
by
  sorry

end transformed_roots_polynomial_l531_531695


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531448

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531448


namespace sasha_remainder_is_20_l531_531790

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l531_531790


namespace angle_between_O₁N_O₂B_is_90_degrees_l531_531522

-- Mathematical setup:
variables {P : Type} [metric_space P] [normed_add_torsor ℝ P]
-- Definitions:
variables {O₁ O₂ B K L A C N : P}
variables (ω₁ ω₂: set P)

-- Conditions:
def circles_intersection : Prop :=
  (ω₁ ∩ ω₂).Nonempty ∧ B ∈ ω₁ ∧ B ∈ ω₂

def extension_intersects (O₁ O₂ B K L : P) [metric_space P] :=
  ∃ t₁ t₂: ℝ, O₂ + t₁ • (B - O₂) ∈ ω₁ ∧ O₂ + t₁ • (B - O₂) = K ∧
               O₁ + t₂ • (B - O₁) ∈ ω₂ ∧ O₁ + t₂ • (B - O₁) = L

def line_through_parallel (B K L A C : P) : Prop :=
  ∃ line : set P, B ∈ line ∧ (∀ x y ∈ line, (K - L) ∥ (y - x)) ∧
  A ∈ ω₁ ∩ line ∧ C ∈ ω₂ ∩ line

def rays_intersection (A K C L N : P) : Prop :=
  (ray A K ∩ ray C L).Nonempty ∧ N ∈ ray A K ∧ N ∈ ray C L

-- Proof Problem:
theorem angle_between_O₁N_O₂B_is_90_degrees
  (h₁ : circles_intersection ω₁ ω₂ B)
  (h₂ : extension_intersects O₁ O₂ B K L)
  (h₃ : line_through_parallel B K L A C)
  (h₄ : rays_intersection A K C L N) :
  angle_between (line O₁ N) (line O₂ B) = 90ᵒ :=
sorry

end angle_between_O₁N_O₂B_is_90_degrees_l531_531522


namespace exists_sequence_of_permutations_with_divisibility_condition_l531_531258

open Nat -- Open the namespace for natural numbers

def isPermutation (a : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, ∃! n : ℕ, a n = m

def S (k : ℕ) (P : ℕ → ℕ) : ℕ := ∑ j in range k, P j

theorem exists_sequence_of_permutations_with_divisibility_condition :
  ∃ (P : ℕ → ℕ → ℕ), (∀ i : ℕ, isPermutation (P i)) ∧
  (∀ k i₁ i₂ : ℕ, 1 ≤ i₁ → i₁ < i₂ → S k (P i₁) ∣ S k (P i₂)) :=
sorry

end exists_sequence_of_permutations_with_divisibility_condition_l531_531258


namespace find_ABC_l531_531724

noncomputable section

def valid_digit (x : ℕ) : Prop :=
  x ≠ 0 ∧ x < 7

def distinct (x y z : ℕ) : Prop :=
  x ≠ y ∧ y ≠ z ∧ x ≠ z

def base7_value (d1 d2 : ℕ) : ℕ :=
  d1 * 7 + d2

theorem find_ABC (A B C : ℕ) (hA : valid_digit A) (hB : valid_digit B) (hC : valid_digit C)
  (h_distinct : distinct A B C)
  (h_eq1 : base7_value A B + C = C * 7)
  (h_eq2 : base7_value A B + base7_value B A = C * 7 + C) : A = 3 ∧ B = 2 ∧ C = 5 :=
begin
  sorry
end

end find_ABC_l531_531724


namespace coordinates_of_point_on_terminal_side_l531_531153

noncomputable def point_coordinates
  (α : ℝ) (r : ℝ) (P : EuclideanSpace ℝ (Fin 2))
  (hP : P ≠ 0) (hr : ∥P∥ = r) : Prop :=
  P = EuclideanSpace.constr ω [r * real.cos α, r * real.sin α]

-- The theorem statement:
theorem coordinates_of_point_on_terminal_side (α : ℝ) (r : ℝ) (P : EuclideanSpace ℝ (Fin 2))
  (hP : P ≠ 0) (hr : ∥P∥ = r) : point_coordinates α r P hP hr :=
begin
  sorry, -- Proof is omitted
end

end coordinates_of_point_on_terminal_side_l531_531153


namespace greatest_divisor_less_than_30_l531_531355

theorem greatest_divisor_less_than_30 :
  (∃ d, d ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} ∧ ∀ m, m ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} → m ≤ d) → 
  18 ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} :=
by
  sorry

end greatest_divisor_less_than_30_l531_531355


namespace c_share_correct_l531_531008

noncomputable def a_oxen_months := 10 * 7
noncomputable def b_oxen_months := 12 * 5
noncomputable def c_oxen_months := 15 * 3
noncomputable def d_oxen_months := 18 * 6
noncomputable def e_oxen_months := 20 * 4
noncomputable def f_oxen_months := 25 * 2

noncomputable def total_oxen_months : ℕ := a_oxen_months + b_oxen_months + c_oxen_months + d_oxen_months + e_oxen_months + f_oxen_months
noncomputable def total_rent : ℝ := 750
noncomputable def cost_per_oxen_month : ℝ := total_rent / total_oxen_months.toReal

noncomputable def c_share : ℝ := c_oxen_months.toReal * cost_per_oxen_month

theorem c_share_correct : c_share ≈ 81.75 := 
by sorry

end c_share_correct_l531_531008


namespace max_disjoint_sets_l531_531547

theorem max_disjoint_sets : 
  ∃ (I: set (ℤ × ℤ)) (hI: ∀ (a b c d: ℤ), (a, b) ∈ I → (c, d) ∈ I → (S a b ∩ S c d = ∅ ↔ (a, b) = (c, d))), 
  finset.card (finset.image id (set.to_finset I)) ≤ 2 :=
by
  sorry

-- Define the set S_{a,b} 
def S (a b : ℤ) : set ℤ := {n | ∃ k, n = k^2 + a * k + b }

-- Prove that the maximum number of pairwise disjoint sets of the form S_{a,b} is 2

end max_disjoint_sets_l531_531547


namespace determinant_of_matrix_l531_531090

def matrix3x3 : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![3, 1, -2], 
    ![8, 5, -4], 
    ![3, 3, 6]]

theorem determinant_of_matrix : matrix.det matrix3x3 = 48 := by
  sorry

end determinant_of_matrix_l531_531090


namespace find_other_integer_l531_531266

theorem find_other_integer (x y : ℤ) (h1 : 3 * x + 4 * y = 135) (h2 : x = 15 ∨ y = 15) : x = 25 ∨ y = 25 :=
by 
  cases h2 with hx15 hy15;
  { subst hx15,
    have hyilv : 4 * y = 90, from (by linarith [h1] : 4 * y = 90),
    have yint : y = 22.5, exact (by norm_cast at *,
                                rw hyilv,
                                norm_num),
    contradiction },
  { subst hy15,
    have hxilv : 3 * x = 75, from (by linarith [h1] : 3 * x = 75),
    have xint : x = 25, exact (by norm_cast at *,
                              rw hxilv,
                              norm_num),
    exact or.inl xint }

end find_other_integer_l531_531266


namespace calculate_error_percentage_l531_531047

theorem calculate_error_percentage (x : ℝ) (hx : x > 0) (x_eq_9 : x = 9) :
  (abs ((x * (x - 8)) / (8 * x)) * 100) = 12.5 := by
  sorry

end calculate_error_percentage_l531_531047


namespace white_cells_remainder_one_l531_531682

theorem white_cells_remainder_one (n : ℕ) (h : n > 1) :
  (exists (I : ℕ),
    (∀ rook_moves : list (ℤ × ℤ), -- list of rook moves
      (∀ move ∈ rook_moves, -- each move adheres to the n steps rule
        (abs move.fst = n ∨ move.fst = 0) ∧ (abs move.snd = n ∨ move.snd = 0)) ∧
      (path_closed rook_moves) ∧ -- path returns to starting cell
      (cells_painted_black rook_moves)) →
    (I % n = 1)) :=
begin
  sorry
end

end white_cells_remainder_one_l531_531682


namespace angle_RQT_square_equilateral_l531_531415

theorem angle_RQT_square_equilateral (P Q R S T : Type)
  (is_square : ∀ a b c d : P, (a = Q) → (b = R) → (c = S) → (d = P) → (1 : a = b) → (1 : b = c) → (1 : c = d) → (1 : d = a) → (angle a b c = 90) → (angle b c d = 90) → (angle c d a = 90) → (angle d a b = 90))
  (is_equilateral : ∀ a b c : P, (a = P) → (b = R) → (c = T) → (1 : a = b) → (1 : b = c) → (angle a b c = 60) → (angle b c a = 60) → (angle c a b = 60))
  : angle R Q T = 135 := 
sorry

end angle_RQT_square_equilateral_l531_531415


namespace class_championship_probability_l531_531209

theorem class_championship_probability (prob_win : ℕ → ℕ → ℚ) (c : ℕ) 
  (h1 : ∀ g, ∀ i ∈ {1, 2, 3}, prob_win g i = 1 / 3) :
  prob_win c 1 * prob_win c 2 * prob_win c 3 = 1 / 27 :=
by
  sorry

end class_championship_probability_l531_531209


namespace geometric_sequence_general_term_sum_of_bn_l531_531600

theorem geometric_sequence_general_term {a : ℕ → ℕ} (h_geom : ∀ n ≥ 1, a (n + 1) = 2 * a n) (h_a1 : a 1 = 1) : ∀ n ≥ 1, a n = 2^(n - 1) := 
by sorry

theorem sum_of_bn {a b : ℕ → ℕ} (h_geom : ∀ n ≥ 1, a (n + 1) = 2 * a n) (h_a1 : a 1 = 1) (h_b : ∀ n ≥ 1, b n = n * a n) : ∀ n, (∑ i in finset.range n, b (i + 1)) = 1 + (n - 1) * 2^n := 
by sorry

end geometric_sequence_general_term_sum_of_bn_l531_531600


namespace speed_of_current_l531_531862

-- Definitions
def downstream_speed (m current : ℝ) := m + current
def upstream_speed (m current : ℝ) := m - current

-- Theorem
theorem speed_of_current 
  (m : ℝ) (current : ℝ) 
  (h1 : downstream_speed m current = 20) 
  (h2 : upstream_speed m current = 14) : 
  current = 3 :=
by
  -- proof goes here
  sorry

end speed_of_current_l531_531862


namespace paintable_area_correct_l531_531077

-- Define the dimensions of the bedroom walls and unpainted areas
def length1 := 14
def width1 := 12
def height := 9
def unpainted_area_per_bedroom := 70
def number_of_bedrooms := 4

-- Calculate the paintable wall area
def area_per_wall (length : ℕ) (height : ℕ) : ℕ :=
  length * height

def total_paintable_area_per_bedroom : ℕ :=
  2 * area_per_wall length1 height + 2 * area_per_wall width1 height - unpainted_area_per_bedroom

def total_paintable_area : ℕ :=
  number_of_bedrooms * total_paintable_area_per_bedroom

theorem paintable_area_correct :
  total_paintable_area = 1592 := by
  -- Proof omitted
  sorry

end paintable_area_correct_l531_531077


namespace number_of_questions_in_test_l531_531332

theorem number_of_questions_in_test (x : ℕ) (sections questions_correct : ℕ)
  (h_sections : sections = 5)
  (h_questions_correct : questions_correct = 32)
  (h_percentage : 0.70 < (questions_correct : ℚ) / x ∧ (questions_correct : ℚ) / x < 0.77) 
  (h_multiple_of_sections : x % sections = 0) : 
  x = 45 :=
sorry

end number_of_questions_in_test_l531_531332


namespace value_of_ratio_l531_531980

theorem value_of_ratio (x y : ℝ)
    (hx : x > 0)
    (hy : y > 0)
    (h : 2 * x + 3 * y = 8) :
    (2 / x + 3 / y) = 25 / 8 := 
by
  sorry

end value_of_ratio_l531_531980


namespace largest_study_only_Biology_l531_531504

-- Let's define the total number of students
def total_students : ℕ := 500

-- Define the given conditions
def S : ℕ := 65 * total_students / 100
def M : ℕ := 55 * total_students / 100
def B : ℕ := 50 * total_students / 100
def P : ℕ := 15 * total_students / 100

def MS : ℕ := 35 * total_students / 100
def MB : ℕ := 25 * total_students / 100
def BS : ℕ := 20 * total_students / 100
def MSB : ℕ := 10 * total_students / 100

-- Required to prove that the largest number of students who study only Biology is 75
theorem largest_study_only_Biology : 
  (B - MB - BS + MSB) = 75 :=
by 
  sorry

end largest_study_only_Biology_l531_531504


namespace geometric_sequence_l531_531167

theorem geometric_sequence (q : ℝ) (a : ℕ → ℝ) (h1 : q > 0) (h2 : a 2 = 1)
  (h3 : a 2 * a 10 = 2 * (a 5)^2) : ∀ n, a n = 2^((n-2:ℝ)/2) := by
  sorry

end geometric_sequence_l531_531167


namespace arrangement_correct_l531_531846

def A := 4
def B := 1
def C := 2
def D := 5
def E := 6
def F := 3

def sum1 := A + B + C
def sum2 := A + D + F
def sum3 := B + E + D
def sum4 := C + F + E
def sum5 := A + E + F
def sum6 := B + D + C
def sum7 := B + C + F

theorem arrangement_correct :
  sum1 = 15 ∧ sum2 = 15 ∧ sum3 = 15 ∧ sum4 = 15 ∧ sum5 = 15 ∧ sum6 = 15 ∧ sum7 = 15 := 
by
  unfold sum1 sum2 sum3 sum4 sum5 sum6 sum7 
  unfold A B C D E F
  sorry

end arrangement_correct_l531_531846


namespace find_principal_l531_531010

-- Conditions translated to Lean definitions
def amount : ℝ := 1344
def annual_rate : ℝ := 0.05
def time : ℝ := 12 / 5
def principal : ℝ := 1200

-- Proof problem statement
theorem find_principal 
  (A : ℝ := amount)
  (r : ℝ := annual_rate)
  (t : ℝ := time)
  (P : ℝ := principal) :
  A = P * (1 + r * t) :=
  by 
    sorry

end find_principal_l531_531010


namespace distance_O_to_line_AB_is_correct_l531_531670

-- Define polar coordinates for points A and B.
structure PolarCoord where
  r : ℝ
  θ : ℝ

def O : PolarCoord := {r := 0, θ := 0}
def A : PolarCoord := {r := 2 * real.sqrt 3, θ := real.pi / 6}
def B : PolarCoord := {r := 3, θ := 2 * real.pi / 3}

-- Define a function to find the distance from the origin O to line AB given polar coordinates of A and B.
noncomputable def distance_from_origin_to_line (O A B : PolarCoord) : ℝ := sorry

-- State the theorem that the distance from the origin to the line passing through points A and B is 6 * sqrt(7) / 7.
theorem distance_O_to_line_AB_is_correct :
  distance_from_origin_to_line O A B = 6 * real.sqrt 7 / 7 := sorry

end distance_O_to_line_AB_is_correct_l531_531670


namespace proof_equivalence_l531_531607

def f (a : ℝ) (x : ℝ) : ℝ := log a (sqrt (x ^ 2 + 1) + x) + (1 / (a ^ x - 1)) + 1

variable α : ℝ

theorem proof_equivalence (a : ℝ) (ha : 0 < a) (ha_ne_one : a ≠ 1)
    (h : f a (sin (π / 6 - α)) = 1 / 3) :
    f a (cos (α - 2 * π / 3)) = 2 / 3 :=
sorry

end proof_equivalence_l531_531607


namespace value_of_M_l531_531754

theorem value_of_M : 
  ∃ (M : ℚ), 
  let a₁₁ := 25,
      a₄₁ := 16, a₅₁ := 20,
      a₄₂ := -20 in
  ∀ a b c d : ℚ,
  (20 - 16 = a) ∧
  (16 - a = b) ∧
  (b - a = c) ∧
  (a₁₁ + -5 * (-17 / 3) = d) ∧
  ([-20 - d] / 4 = -115 / 6) ∧
  (d - 115 / 6 = M) 
  → M = 37.5 := 
by
  sorry

end value_of_M_l531_531754


namespace count_five_digit_numbers_div_by_16_l531_531190

theorem count_five_digit_numbers_div_by_16 :
  let digits := {d : ℕ | d < 10}
  ∃ count : ℕ, count = 90 ∧ 
    ∀ a b c : ℕ, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 1 ≤ a →
      (nat.gcd (a * 10000 + b * 1000 + 160 + c) 16 = 16) → count = 90 :=
by
  let digits := {d : ℕ | d < 10}
  existsi 90
  split
  { sorry }
  { intros a b c ha hb hc ha_pos hdiv
    sorry }

end count_five_digit_numbers_div_by_16_l531_531190


namespace union_A_B_equals_C_l531_531621

-- Define Set A
def A : Set ℝ := {x : ℝ | 3 - 2 * x > 0}

-- Define Set B
def B : Set ℝ := {x : ℝ | x^2 ≤ 4}

-- Define the target set C which is supposed to be A ∪ B
def C : Set ℝ := {x : ℝ | x ≤ 2}

theorem union_A_B_equals_C : A ∪ B = C := by 
  -- Proof is omitted here
  sorry

end union_A_B_equals_C_l531_531621


namespace lilith_needs_to_find_l531_531263

def num_water_bottles := 60
def num_energy_bars := 48
def original_price_water := 2.0
def original_price_energy := 3.0
def market_price_water := 1.85
def market_price_energy := 2.75
def discount := 0.10

def original_total_money := num_water_bottles * original_price_water + num_energy_bars * original_price_energy
def discounted_price (market_price : ℝ) := market_price * (1 - discount)

def total_discounted_money := num_water_bottles * discounted_price market_price_water + num_energy_bars * discounted_price market_price_energy
def money_needed_to_find := original_total_money - total_discounted_money

theorem lilith_needs_to_find : money_needed_to_find = 45.30 :=
  sorry

end lilith_needs_to_find_l531_531263


namespace union_of_sets_l531_531160

def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 5, 7}
def union_result : Set ℕ := {1, 3, 5, 7}

theorem union_of_sets : A ∪ B = union_result := by
  sorry

end union_of_sets_l531_531160


namespace correct_weights_l531_531702

def weight (item : String) : Nat :=
  match item with
  | "Banana" => 140
  | "Pear" => 120
  | "Melon" => 1500
  | "Tomato" => 150
  | "Apple" => 170
  | _ => 0

theorem correct_weights :
  weight "Banana" = 140 ∧
  weight "Pear" = 120 ∧
  weight "Melon" = 1500 ∧
  weight "Tomato" = 150 ∧
  weight "Apple" = 170 ∧
  (weight "Melon" > weight "Pear") ∧
  (weight "Melon" < weight "Tomato") :=
by
  sorry

end correct_weights_l531_531702


namespace tangent_length_from_P_to_circle_l531_531951

noncomputable def point (x y : ℝ) := (x, y)

-- Definitions
def P := point 2 3
def C := point 1 1
def circle (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 1

-- Theorem statement
theorem tangent_length_from_P_to_circle :
  ∀ (P C : ℝ × ℝ), P = (2, 3) → C = (1, 1) → ((P.1 - C.1)^2 + (P.2 - C.2)^2 - 1).sqrt = 2 :=
by
  intros P C hP hC
  rw [hP, hC]
  sorry

end tangent_length_from_P_to_circle_l531_531951


namespace probability_of_differ_by_three_l531_531495

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l531_531495


namespace shelves_needed_l531_531383

theorem shelves_needed (initial_stock sold stock_per_shelf : ℕ) (h₁ : initial_stock = 40) 
  (h₂ : sold = 20) (h₃ : stock_per_shelf = 4) : (initial_stock - sold) / stock_per_shelf = 5 :=
by
    -- Initial stock
    have h_stock : initial_stock = 40 := h₁
    -- Sold amount
    have h_sold : sold = 20 := h₂
    -- Books per shelf
    have h_shelf : stock_per_shelf = 4 := h₃
    -- Remaining stock
    have h_remaining : initial_stock - sold = 20 := by
        rw [h_stock, h_sold]
        exact rfl
    -- Number of shelves
    have h_shelves : 20 / stock_per_shelf = 5 := by
        rw h_shelf
        exact rfl
    -- Prove the statement
    rw [h_remaining, h_shelves]
    exact rfl

end shelves_needed_l531_531383


namespace first_number_is_48_l531_531840

-- Definitions of the conditions
def ratio (A B : ℕ) := 8 * B = 9 * A
def lcm (A B : ℕ) := Nat.lcm A B = 432

-- The statement to prove
theorem first_number_is_48 (A B : ℕ) (h_ratio : ratio A B) (h_lcm : lcm A B) : A = 48 :=
by
  sorry

end first_number_is_48_l531_531840


namespace find_m_l531_531187

def A := {x : ℝ | x^2 - 3 * x + 2 = 0}
def C (m : ℝ) := {x : ℝ | x^2 - m * x + 2 = 0}

theorem find_m (m : ℝ) (h : A ∩ C m = C m) : 
  m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) :=
by sorry

end find_m_l531_531187


namespace collinear_PQR_concyclic_UROV_l531_531324

-- Definitions for conditions
def cyclic_quad (A B C D O : Point) : Prop :=
  -- Quadrilateral ABCD is inscribed in a circle centered at O
  inscribed_in_circle A B C D O

def AC_intersection_BD (A B C D P : Point) : Prop :=
  -- P is the intersection of AC and BD
  intersect AC BD P

def AB_intersection_CD (A B C D Q : Point) : Prop :=
  -- Q is the intersection of AB and CD
  intersect AB CD Q

def second_intersection {A B C D P R : Point} : Prop :=
  -- R is the second intersection point of the circumcircles of triangles ABP and CDP
  circumcircle_intersection_second ABP CDP R

def circumcenter (X Y Z U : Point) : Prop :=
  -- U is the circumcenter of triangle XYZ
  is_circumcenter X Y Z U

-- The first theorem: P, Q, and R are collinear
theorem collinear_PQR (A B C D O P Q R : Point) 
  (ABC_inscribed : cyclic_quad A B C D O)
  (P_is_intersection : AC_intersection_BD A B C D P)
  (Q_is_intersection : AB_intersection_CD A B C D Q)
  (R_is_second_intersection : second_intersection A B C D P R) :
  collinear P Q R := 
sorry

-- The second theorem: U, R, O, V are concyclic
theorem concyclic_UROV (A B C D O P Q R U V : Point)
  (ABC_inscribed : cyclic_quad A B C D O)
  (P_is_intersection : AC_intersection_BD A B C D P)
  (Q_is_intersection : AB_intersection_CD A B C D Q)
  (R_is_second_intersection : second_intersection A B C D P R)
  (U_is_circumcenter_ABP : circumcenter A B P U)
  (V_is_circumcenter_CDP : circumcenter C D P V):
  concyclic U R O V :=
sorry

end collinear_PQR_concyclic_UROV_l531_531324


namespace identify_heaviest_and_lightest_coin_l531_531809

theorem identify_heaviest_and_lightest_coin :
  ∀ (coins : Fin 10 → ℕ), 
  (∀ i j, i ≠ j → coins i ≠ coins j) → 
  ∃ (seq : List (Fin 10 × Fin 10)), 
  seq.length = 13 ∧ 
  (∀ (i j : Fin 10), (i, j) ∈ seq → 
    (coins i < coins j ∨ coins i > coins j)) ∧ 
  (∃ (heaviest lightest : Fin 10),
    (∀ coin, coins coin ≤ coins heaviest) ∧ (∀ coin, coins coin ≥ coins lightest)) :=
by
  intros coins h_coins
  exists [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), -- initial pairs
          (0, 2), (2, 4), (4, 6), (6, 8),         -- heaviest coin comparisons
          (1, 3), (3, 5), (5, 7), (7, 9)]         -- lightest coin comparisons
  constructor
  . -- length check
    rfl
  . constructor
    . -- all comparisons
      intros i j h_pair
      cases h_pair; simp; solve_by_elim
    . -- finding heaviest and lightest coins
      exists 8, 9
      constructor
      . -- all coins are less than or equal to the heaviest
        sorry
      . -- all coins are greater than or equal to the lightest
        sorry

end identify_heaviest_and_lightest_coin_l531_531809


namespace average_expenditure_decrease_l531_531340

theorem average_expenditure_decrease:
  let original_students := 35
  let new_students := 42
  let increase_in_expenses := 42
  let original_total_expenditure := 420
  let new_total_expenditure := original_total_expenditure + increase_in_expenses
  let original_average := original_total_expenditure / original_students
  let new_average := new_total_expenditure / new_students
  in original_average - new_average = 1 := by
sorry

end average_expenditure_decrease_l531_531340


namespace zero_a_if_square_every_n_l531_531073

theorem zero_a_if_square_every_n (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 := 
sorry

end zero_a_if_square_every_n_l531_531073


namespace simplify_tan_cot_fraction_l531_531289

theorem simplify_tan_cot_fraction :
  let tan45 := 1
  let cot45 := 1
  (tan45^3 + cot45^3) / (tan45 + cot45) = 1 := by
    sorry

end simplify_tan_cot_fraction_l531_531289


namespace inequality_relationship_l531_531988

noncomputable def even_function_periodic_decreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧
  (∀ x, f (x + 2) = f x) ∧
  (∀ x1 x2, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 1 → f x1 > f x2)

theorem inequality_relationship (f : ℝ → ℝ) (h : even_function_periodic_decreasing f) : 
  f (-1) < f (2.5) ∧ f (2.5) < f 0 :=
by 
  sorry

end inequality_relationship_l531_531988


namespace indeterminate_relation_between_l1_and_l4_l531_531658

theorem indeterminate_relation_between_l1_and_l4
  (l1 l2 l3 l4 : Type) [linear_space l1] [linear_space l2] [linear_space l3] [linear_space l4]
  (perpendicular : l1 ⊥ l2) (parallel : l2 ∥ l3) (perpendicular_2 : l3 ⊥ l4) :
  ¬ (l1 ∥ l4) ∧ ¬ (l1 ⊥ l4) :=
  sorry

end indeterminate_relation_between_l1_and_l4_l531_531658


namespace distance_between_trees_l531_531652

theorem distance_between_trees 
  (rows columns : ℕ)
  (boundary_distance garden_length d : ℝ)
  (h_rows : rows = 10)
  (h_columns : columns = 12)
  (h_boundary_distance : boundary_distance = 5)
  (h_garden_length : garden_length = 32) :
  (9 * d + 2 * boundary_distance = garden_length) → 
  d = 22 / 9 := 
by 
  intros h_eq
  sorry

end distance_between_trees_l531_531652


namespace intersection_A_B_l531_531589

open Set

-- Define set A and set B based on the conditions
def A : Set ℝ := {x | abs (x - 3) ≤ 1}
def B : Set ℝ := {x | x^2 - 5 * x + 4 ≥ 0}

-- Theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {4} :=
by sorry

end intersection_A_B_l531_531589


namespace sum_b_1000_correct_l531_531095

noncomputable def b : ℕ → ℕ 
| 1 := 2
| 2 := 2
| 3 := 2
| n := if n > 3 then (let p := b (n - 1);
                           q := b (n - 2) * b (n - 3);
                           if 9 * p^2 / 4 - q - 1 < 0 then 0 -- No real roots
                           else if 9 * p^2 / 4 - q - 1 = 0 then 2 -- Two real roots
                           else 4) -- Four real roots
         else 0

def sum_b (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem sum_b_1000_correct : sum_b 1000 = -- correct computed value goes here
sorry

end sum_b_1000_correct_l531_531095


namespace largest_integer_k_for_distinct_real_roots_l531_531993

theorem largest_integer_k_for_distinct_real_roots :
  ∀ (k : ℤ), ((k < 3) ∧ (k ≠ 2)) → k ≤ 1 :=
by
  intros k h_conditions,
  let a := k - 2,
  let b := -4,
  let c := 4,
  have discriminant_pos : 48 - 16 * k > 0,
  {
    -- The proof for 48 - 16k > 0 which implies k < 3 is assumed from the conditions
    sorry,
  },
  have k_not_2 : k ≠ 2,
  {
    -- The proof for k ≠ 2 is assumed from the conditions
    sorry,
  },
  -- Since k < 3 and k ≠ 2, the largest integer satisfying this is 1
  sorry

end largest_integer_k_for_distinct_real_roots_l531_531993


namespace number_of_digits_in_sum_is_4_l531_531104

theorem number_of_digits_in_sum_is_4 (C D : ℕ) (hC : 1 ≤ C ∧ C ≤ 9) (hD : 1 ≤ D ∧ D ≤ 9) :
  (let sum := 6543 + C * 100 + 75 + D * 10 + 6 in
   1000 ≤ sum ∧ sum < 10000) :=
by
  sorry

end number_of_digits_in_sum_is_4_l531_531104


namespace manuscript_page_count_l531_531748

-- Define the main statement
theorem manuscript_page_count
  (P : ℕ)
  (cost_per_page : ℕ := 10)
  (rev1_pages : ℕ := 30)
  (rev2_pages : ℕ := 20)
  (total_cost : ℕ := 1350)
  (cost_rev1 : ℕ := 15)
  (cost_rev2 : ℕ := 20) 
  (remaining_pages_cost : ℕ := 10 * (P - (rev1_pages + rev2_pages))) :
  (remaining_pages_cost + rev1_pages * cost_rev1 + rev2_pages * cost_rev2 = total_cost)
  → P = 100 :=
by
  sorry

end manuscript_page_count_l531_531748


namespace patrick_savings_ratio_l531_531275

theorem patrick_savings_ratio (S : ℕ) (bike_cost : ℕ) (lent_amt : ℕ) (remaining_amt : ℕ)
  (h1 : bike_cost = 150)
  (h2 : lent_amt = 50)
  (h3 : remaining_amt = 25)
  (h4 : S = remaining_amt + lent_amt) :
  (S / bike_cost : ℚ) = 1 / 2 := 
sorry

end patrick_savings_ratio_l531_531275


namespace book_spending_fraction_l531_531236

-- Define the conditions
def earnings_per_week := 10
def num_weeks := 4
def total_savings := earnings_per_week * num_weeks
def money_spent_video_game := total_savings / 2
def remaining_money_after_video_game := total_savings - money_spent_video_game
def remaining_money_after_book := 15
def money_spent_book := remaining_money_after_video_game - remaining_money_after_book

-- Prove the fraction spent on the book is 1/4
theorem book_spending_fraction :
  (money_spent_book / remaining_money_after_video_game) = 1 / 4 :=
by
  sorry

end book_spending_fraction_l531_531236


namespace ways_to_assign_roles_is_24_l531_531585

-- Define the four members
inductive Member
| Alice | Bob | Carol | Dave

-- Define the four roles
inductive Role
| President | Secretary | Treasurer | VicePresident

-- Define the number of members and roles (both are set to 4)
def numMembers : Nat := 4
def numRoles : Nat := 4

-- Define the problem statement in Lean
def numWaysToAssignRoles : Nat :=
  numMembers.factorial

theorem ways_to_assign_roles_is_24 :
  numWaysToAssignRoles = 24 :=
by
  unfold numWaysToAssignRoles
  unfold numMembers
  apply Nat.factorial_def
  sorry

end ways_to_assign_roles_is_24_l531_531585


namespace krishan_money_l531_531375

theorem krishan_money (R G K : ℕ) (hR : R = 637) (hRG : R * 17 = G * 7) (hGK : G * 17 = K * 7) : K = 3774 :=
by {
  sorry -- Proof not required as per the instructions
}

end krishan_money_l531_531375


namespace trapezoid_bases_l531_531150

def is_right_trapezoid (c d : ℝ) (d_gt_c : d > c) : Prop :=
  ∀ (a b : ℝ),
    parallel_lines_divide_trapezoid c d a b ∧
    inscribed_circle_possible c d a b

def bases_of_trapezoid (c d : ℝ) (d_gt_c : d > c) : (ℝ × ℝ) :=
  ( (d - (Real.sqrt (d^2 - c^2))) / 2, 
    (d + (Real.sqrt (d^2 - c^2))) / 2 )

theorem trapezoid_bases (c d : ℝ) (d_gt_c : d > c) :
  is_right_trapezoid c d d_gt_c →
  bases_of_trapezoid c d d_gt_c =
  ( (d - (Real.sqrt (d^2 - c^2))) / 2, 
    (d + (Real.sqrt (d^2 - c^2))) / 2 ) :=
by {
  intro h,
  sorry
}

end trapezoid_bases_l531_531150


namespace probability_diff_by_three_l531_531474

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l531_531474


namespace always_composite_for_x64_l531_531689

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

theorem always_composite_for_x64 (n : ℕ) : is_composite (n^4 + 64) :=
by
  sorry

end always_composite_for_x64_l531_531689


namespace product_of_binaries_l531_531934

-- Step a) Define the binary numbers as Lean 4 terms.
def bin_11011 : ℕ := 0b11011
def bin_111 : ℕ := 0b111
def bin_101 : ℕ := 0b101

-- Step c) Define the goal to be proven.
theorem product_of_binaries :
  bin_11011 * bin_111 * bin_101 = 0b1110110001 :=
by
  -- proof goes here
  sorry

end product_of_binaries_l531_531934


namespace proof_final_value_l531_531844

-- Define percentages as fractions
def percentage (p : ℝ) (x : ℝ) : ℝ := (p / 100) * x

-- Define the given conditions
def condition_1 : ℝ := percentage 47 1442 -- 47% of 1442

def condition_2 : ℝ := percentage 36 1412 -- 36% of 1412

-- Final question
def final_value : ℝ := (condition_1 - condition_2) + 63

-- Proof statement
theorem proof_final_value : final_value = 232.42 :=
by
  sorry

end proof_final_value_l531_531844


namespace num_different_results_l531_531207

theorem num_different_results (n : ℕ) (f : ℕ → ℤ) :
  (∀ k, k > 0 → f k = -(2^k))
  → (n = 10)
  → (∑ k in finset.range (n + 1), (if k = 0 then 1 else f k)) = 1024 := sorry

end num_different_results_l531_531207


namespace find_z_l531_531691

theorem find_z (x y k : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (h : 1/x + 1/y = k) :
  ∃ z : ℝ, 1/z = k ∧ z = xy/(x + y) :=
by {
  sorry
}

end find_z_l531_531691


namespace f_neg_4_equals_3_l531_531169

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Define the function based on the given conditions
def f (x : ℝ) : ℝ :=
  if h : x > 0 then -x + 1 else sorry  -- We'll skip the definition for x ≤ 0

-- State that f(-4) = 3 given the conditions
theorem f_neg_4_equals_3 (h_odd : is_odd_function f) (h_pos : ∀ x : ℝ, x > 0 → f x = -x + 1) :
  f (-4) = 3 :=
sorry

end f_neg_4_equals_3_l531_531169


namespace find_k_l531_531659

-- Define the conditions
def parabola (k : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + k

-- Theorem statement
theorem find_k (k : ℝ) : (∀ x : ℝ, parabola k x = 0 → x = -1) → k = 1 :=
by
  sorry

end find_k_l531_531659


namespace simplify_tan_cot_fraction_l531_531291

theorem simplify_tan_cot_fraction :
  let tan45 := 1
  let cot45 := 1
  (tan45^3 + cot45^3) / (tan45 + cot45) = 1 := by
    sorry

end simplify_tan_cot_fraction_l531_531291


namespace new_person_weight_l531_531014

theorem new_person_weight (w : ℝ) (avg_increase : ℝ) (replaced_person_weight : ℝ) (num_people : ℕ) 
(H1 : avg_increase = 4.8) (H2 : replaced_person_weight = 62) (H3 : num_people = 12) : 
w = 119.6 :=
by
  -- We could provide the intermediate steps as definitions here but for the theorem statement, we just present the goal.
  sorry

end new_person_weight_l531_531014


namespace find_point_P_on_line_segment_l531_531588

-- Define points P1 and P2
structure Point where
  x : ℝ
  y : ℝ

def P1 : Point := { x := 0, y := 2 }
def P2 : Point := { x := 3, y := 0 }

-- Define vector calculations and conditions
def vecP1P (P : Point) : Point := { x := P.x, y := P.y - P1.y }
def vecPP2 (P : Point) : Point := { x := P2.x - P.x, y := -P.y }

-- Proof statement
theorem find_point_P_on_line_segment (P : Point) (h : vecP1P P = { x := 2 * vecPP2 P.x, y := 2 * vecPP2 P.y }) :
  P = { x := 2, y := 2 / 3 } :=
sorry

end find_point_P_on_line_segment_l531_531588


namespace range_of_f_l531_531533

def f (x : ℝ) : ℝ := (x^2 - 2*x) / (x^2 + 2*x + 2)

theorem range_of_f : set.Icc (2 - Real.sqrt 5) (2 + Real.sqrt 5) = set.range f :=
by
  sorry

end range_of_f_l531_531533


namespace probability_differ_by_three_is_one_sixth_l531_531462

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l531_531462


namespace samanta_s_eggs_left_l531_531718

def total_eggs : ℕ := 30
def cost_per_crate_dollars : ℕ := 5
def cost_per_crate_cents : ℕ := cost_per_crate_dollars * 100
def sell_price_per_egg_cents : ℕ := 20

theorem samanta_s_eggs_left
  (total_eggs : ℕ) (cost_per_crate_dollars : ℕ) (sell_price_per_egg_cents : ℕ) 
  (cost_per_crate_cents = cost_per_crate_dollars * 100) : 
  total_eggs - (cost_per_crate_cents / sell_price_per_egg_cents) = 5 :=
by sorry

end samanta_s_eggs_left_l531_531718


namespace old_toilet_water_per_flush_correct_l531_531677

noncomputable def old_toilet_water_per_flush (water_saved : ℕ) (flushes_per_day : ℕ) (days_in_june : ℕ) (reduction_percentage : ℚ) : ℚ :=
  let total_flushes := flushes_per_day * days_in_june
  let water_saved_per_flush := water_saved / total_flushes
  let reduction_factor := reduction_percentage
  let original_water_per_flush := water_saved_per_flush / (1 - reduction_factor)
  original_water_per_flush

theorem old_toilet_water_per_flush_correct :
  old_toilet_water_per_flush 1800 15 30 (80 / 100) = 5 := by
  sorry

end old_toilet_water_per_flush_correct_l531_531677


namespace find_principal_l531_531831

-- Define the given conditions
def R : ℝ := 12 / 100
def T : ℝ := 4
def difference : ℝ := 93.51936000000069

-- Define the simple interest formula
def simple_interest (P : ℝ) : ℝ := P * R * T

-- Define the compound interest formula
def compound_interest (P : ℝ) : ℝ := P * ((1 + R) ^ T - 1)

-- Define the equation to solve for principal P
def principal (P : ℝ) : Prop := compound_interest P - simple_interest P = difference

-- Statement to prove
theorem find_principal : ∃ P, principal P := sorry

end find_principal_l531_531831


namespace sasha_remainder_l531_531767

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l531_531767


namespace probability_of_diff_3_is_1_over_9_l531_531425

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l531_531425


namespace machine_X_takes_longer_l531_531282

variables (W : ℝ) -- Non-negative and non-zero widget quantity

-- Conditions
def rate_machine_X : ℝ := W / 6
def rate_machine_Y : ℝ := W / 4
def combined_rate : ℝ := (5 * W) / 12

-- Time taken by each machine to produce W widgets
def time_machine_X : ℝ := W / rate_machine_X
def time_machine_Y : ℝ := W / rate_machine_Y

-- Difference in time
def D : ℝ := time_machine_X - time_machine_Y

theorem machine_X_takes_longer :
  time_machine_X = 6 ∧ -- Time for machine X to produce W widgets
  time_machine_Y = 4 ∧ -- Time for machine Y to produce W widgets
  D = 2 :=            -- Machine X takes 2 days longer than machine Y
by
  -- Proof steps would go here; for now, assume to validate the definitions and statements.
  sorry

end machine_X_takes_longer_l531_531282


namespace root_expression_value_l531_531635

theorem root_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : 2021 - 2 * a^2 - 2 * a = 2019 := 
by sorry

end root_expression_value_l531_531635


namespace count_divisible_by_16_l531_531192

theorem count_divisible_by_16 : 
  (∃ (a b c : ℕ), a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                  b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                  c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                  (10000 * a + 1000 * b + 160 + c) % 16 = 0) ↔ 
  (S := 9 * 10) := sorry

end count_divisible_by_16_l531_531192


namespace probA_probB_probC_probD_l531_531620

-- Definition for Problem A
def seqA (a : ℕ → ℕ) (n : ℕ) : Prop :=
match n with
| 0       => a 0 = 2
| n + 1   => a (n + 1) = a n + n + 1

-- Proof of a₃ = 7 given seqA
theorem probA : ∀ a, seqA a 0 → seqA a 1 → seqA a 2 → a 3 = 7 :=
by
  intros a h0 h1 h2
  sorry

-- Definition for Problem B
def seqB (a : ℕ → ℕ) (n : ℕ) : Prop :=
match n with
| 0       => a 0 = 1
| n + 1   => a (n + 1) = 3 * a n + 2

-- Proof of a₄ = 53 given seqB
theorem probB : ∀ a, seqB a 0 → seqB a 1 → seqB a 2 → seqB a 3 → a 4 = 53 :=
by
  intros a h0 h1 h2 h3
  sorry

-- Definition for Problem C
def seqC (S : ℕ → ℚ) : Prop :=
S = λ n, 3^n + 1/2

-- Proof that seqC is not geometric
theorem probC (S : ℕ → ℚ) (h : seqC S) : ¬ geometric_sequence (λ i, S (i + 1) - S i) :=
by
  intro h
  unfold geometric_sequence at h
  sorry

-- Definition for Problem D
def seqD (a : ℕ → ℚ) (n : ℕ) : Prop :=
match n with
| 0       => a 0 = 1
| n + 1   => a (n + 1) = 2 * a n / (2 + a n)

-- Proof of a₅ ≠ 1/5 given seqD
theorem probD : ∀ a, seqD a 0 → seqD a 1 → seqD a 2 → seqD a 3 → seqD a 4 → a 5 ≠ 1/5 :=
by
  intros a h0 h1 h2 h3 h4
  sorry

end probA_probB_probC_probD_l531_531620


namespace log_inequality_l531_531979

theorem log_inequality (n : ℕ) (h1 : n > 1) : 
  (1 : ℝ) / (n : ℝ) > Real.log ((n + 1 : ℝ) / n) ∧ 
  Real.log ((n + 1 : ℝ) / n) > (1 : ℝ) / (n + 1) := 
by
  sorry

end log_inequality_l531_531979


namespace triangle_altitude_length_l531_531579

-- Given conditions
variables (r : ℝ) (h : ℝ)
def square_side : ℝ := 4 * r
def square_area : ℝ := (square_side r) ^ 2
def diagonal_length : ℝ := square_side r * real.sqrt 2
def triangle_area : ℝ := (1 / 2) * (diagonal_length r) * h

-- The target equation to prove
theorem triangle_altitude_length (r : ℝ) :
  (2 * square_area r = triangle_area r h) → h = 8 * r * real.sqrt 2 :=
begin
  -- Proof omitted
  sorry
end

end triangle_altitude_length_l531_531579


namespace trig_identity_l531_531975

variable {α : ℝ}

theorem trig_identity (h1 : sin α - cos α = 1 / 2) (h2 : 0 < α ∧ α < π) : 
  sin α + cos α = sqrt 7 / 2 := 
sorry

end trig_identity_l531_531975


namespace part1_part2_l531_531146

-- Define the circle and point
def circle (x y : ℝ) := (x - 1)^2 + y^2 = 6
def point_P := (2, 2)

-- Part Ⅰ
theorem part1 :
  (P_m : (ℝ × ℝ)) (P_m = P_m) → (∀ A B : ℝ × ℝ, circle (A.1) (A.2) → circle (B.1) (B.2) → (A.1 + B.1)/2 = 2 ∧ (A.2 + B.2)/2 = 2) → ∃ l : ℝ → ℝ, ∀ x y : ℝ, l x y = x + 2*y - 6 :=
sorry

-- Part Ⅱ
theorem part2 :
  (∀ A B : ℝ × ℝ, (circle (A.1) (A.2)) ∧ (circle (B.1) (B.2)) ∧ ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * √5)^2)) → ∃ l : ℝ → ℝ, (∀ x y : ℝ, l x y = (x - 2)) ∨ (l x y = (3*x - 4*y + 2)) :=
sorry

end part1_part2_l531_531146


namespace number_of_real_solutions_eq_60_l531_531548

theorem number_of_real_solutions_eq_60 : 
  (∃ x : ℝ, -50 ≤ x ∧ x ≤ 50 ∧ (x / 50) = Real.cos (2 * x)) ↔ 60 := 
sorry

end number_of_real_solutions_eq_60_l531_531548


namespace value_of_M_l531_531751

theorem value_of_M :
  let row_seq := [25, 25 + (8 - 25) / 3, 25 + 2 * (8 - 25) / 3, 8, 8 + (8 - 25) / 3, 8 + 2 * (8 - 25) / 3, -9]
  let col_seq1 := [25, 25 - 4, 25 - 8]
  let col_seq2 := [16, 20, 20 + 4]
  let col_seq3 := [-9, -9 - 11/4, -9 - 2 * 11/4, -20]
  let M := -9 - (-11/4)
  M = -6.25 :=
by
  sorry

end value_of_M_l531_531751


namespace ethyl_bought_l531_531112

variable (x : ℝ) (y : ℝ)
variable (h1 : 0.25 * x = y)
variable (h2 : x + y = 10)

theorem ethyl_bought (h : 0.25 * x = y) (h2 : x + y = 10) : y = 2 :=
by 
  have h3 : 1.25 * x = 10 := calc
    1.25 * x = (1 + 0.25) * x : by ring
          ... = x + 0.25 * x : by ring
          ... = x + y       : by rw h
          ... = 10          : by exact h2
  have h4: x = 10 / 1.25 := eq_div_of_mul_eq (by norm_num) h3
  have h5 : x = 8 := by norm_num at h4
  have h6 : y = 0.25 * 8 := by rw [h5, h]
  norm_num at h6
  exact h6

end ethyl_bought_l531_531112


namespace total_trees_after_planting_l531_531336

-- Definitions based on conditions
def initial_trees : ℕ := 34
def trees_to_plant : ℕ := 49

-- Statement to prove the total number of trees after planting
theorem total_trees_after_planting : initial_trees + trees_to_plant = 83 := 
by 
  sorry

end total_trees_after_planting_l531_531336


namespace Bill_tossed_objects_l531_531078

theorem Bill_tossed_objects (Ted_sticks Ted_rocks Bill_sticks Bill_rocks : ℕ)
  (h1 : Bill_sticks = Ted_sticks + 6)
  (h2 : Ted_rocks = 2 * Bill_rocks)
  (h3 : Ted_sticks = 10)
  (h4 : Ted_rocks = 10) :
  Bill_sticks + Bill_rocks = 21 :=
by
  sorry

end Bill_tossed_objects_l531_531078


namespace missy_yells_at_stubborn_dog_l531_531699

theorem missy_yells_at_stubborn_dog {yell_obedient : ℕ} {total_yells : ℕ} (h1 : yell_obedient = 12) (h2 : total_yells = 60) :
  ∃ (x : ℕ), 12 + 12 * x = total_yells ∧ x = 4 :=
by
  use 4
  split
  · rw [h1, h2]
    norm_num
  · norm_num

end missy_yells_at_stubborn_dog_l531_531699


namespace chairs_left_after_selling_l531_531401

-- Definitions based on conditions
def chairs_before_selling : ℕ := 15
def difference_after_selling : ℕ := 12

-- Theorem statement based on the question
theorem chairs_left_after_selling : (chairs_before_selling - 3 = difference_after_selling) → (chairs_before_selling - difference_after_selling = 3) := by
  intro h
  sorry

end chairs_left_after_selling_l531_531401


namespace area_enclosed_by_3x2_l531_531727

theorem area_enclosed_by_3x2 (a b : ℝ) (h₀ : a = 0) (h₁ : b = 1) :
  ∫ (x : ℝ) in a..b, 3 * x^2 = 1 :=
by 
  rw [h₀, h₁]
  sorry

end area_enclosed_by_3x2_l531_531727


namespace prime_pairs_divisibility_l531_531531

theorem prime_pairs_divisibility:
  ∀ (p q : ℕ), (Nat.Prime p ∧ Nat.Prime q ∧ p ≤ q ∧ p * q ∣ ((5 ^ p - 2 ^ p) * (7 ^ q - 2 ^ q))) ↔ 
                (p = 3 ∧ q = 5) ∨ 
                (p = 3 ∧ q = 3) ∨ 
                (p = 5 ∧ q = 37) ∨ 
                (p = 5 ∧ q = 83) := by
  sorry

end prime_pairs_divisibility_l531_531531


namespace sasha_remainder_20_l531_531777

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l531_531777


namespace probability_of_diff_3_is_1_over_9_l531_531427

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l531_531427


namespace jack_weight_52_l531_531235

theorem jack_weight_52 (Sam Jack : ℕ) (h1 : Sam + Jack = 96) (h2 : Jack = Sam + 8) : Jack = 52 := 
by
  sorry

end jack_weight_52_l531_531235


namespace findSt_correct_l531_531567

noncomputable def findSt : Prop :=
  ∃ (s t : ℂ) (z1 z2 z3 : ℂ), 
    t ∈ ℝ ∧ 
    arg s = π / 6 ∧
    (∀ x : ℂ, x^3 + x * t + s = 0) ∧
    (abs (z2 - z1) = √3 ∧ abs (z3 - z2) = √3 ∧ abs (z1 - z3) = √3) ∧
    (s = √3 / 2 + (1 / 2) * Complex.I) ∧
    (t = 0)

theorem findSt_correct : findSt := sorry

end findSt_correct_l531_531567


namespace sasha_remainder_l531_531773

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l531_531773


namespace seq_formula_l531_531957

noncomputable def f (a x : ℝ) : ℝ := (a * x) / (a + x)

def a_seq (a : ℝ) [ha : 0 < a] : ℕ → ℝ
| 0     := 1
| (n+1) := f a (a_seq a n)

theorem seq_formula (a : ℝ) [ha : 0 < a] (n : ℕ) : 
  a_seq a (n + 1) = a / (n + a) := by
  sorry

end seq_formula_l531_531957


namespace sally_picked_peaches_l531_531287

theorem sally_picked_peaches (original_peaches total_peaches picked_peaches : ℕ)
  (h_orig : original_peaches = 13)
  (h_total : total_peaches = 55)
  (h_picked : picked_peaches = total_peaches - original_peaches) :
  picked_peaches = 42 :=
by
  sorry

end sally_picked_peaches_l531_531287


namespace complex_subtraction_l531_531827

theorem complex_subtraction (a b : ℂ) (h₁ : a = 4 - 2 * complex.I) (h₂ : b = 3 + 2 * complex.I) : 
  a - 2 * b = -2 - 6 * complex.I :=
by {
  rw [h₁, h₂],
  ring,
}

end complex_subtraction_l531_531827


namespace min_value_of_inverse_sum_l531_531576

theorem min_value_of_inverse_sum {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : (1/x) + (1/y) ≥ 4 :=
by
  sorry

end min_value_of_inverse_sum_l531_531576


namespace sum_S_r_is_zero_matrix_l531_531244

open Matrix

variables {n r : ℕ}

def S_r (n r : ℕ) [fact (1 ≤ r)] [fact (r ≤ n)] : set (matrix (fin n) (fin n) (zmod 2)) :=
{A | rank A = r}

theorem sum_S_r_is_zero_matrix (n r : ℕ) [fact (2 ≤ n)] [fact (1 ≤ r)] [fact (r ≤ n)]
    : ∑ X in (S_r n r), X = 0 := 
by
  sorry

end sum_S_r_is_zero_matrix_l531_531244


namespace probability_diff_by_3_l531_531481

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l531_531481


namespace total_units_in_building_l531_531392

theorem total_units_in_building (x y : ℕ) (cost_1_bedroom cost_2_bedroom total_cost : ℕ)
  (h1 : cost_1_bedroom = 360) (h2 : cost_2_bedroom = 450)
  (h3 : total_cost = 4950) (h4 : y = 7) (h5 : total_cost = cost_1_bedroom * x + cost_2_bedroom * y) :
  x + y = 12 :=
sorry

end total_units_in_building_l531_531392


namespace range_of_m_l531_531619

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, (m + 1) * x^2 ≥ 0) : m > -1 :=
by
  sorry

end range_of_m_l531_531619


namespace dice_rolls_diff_by_3_probability_l531_531456

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l531_531456


namespace range_of_x_l531_531096

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x :
  {x : ℝ | odot x (x - 2) < 0} = {x : ℝ | -2 < x ∧ x < 1} := 
by sorry

end range_of_x_l531_531096


namespace find_cost_price_profit_maximization_overall_profit_l531_531855

-- definitions for part 1
def cost_price_B(B: ℝ) : Prop := let pB := 20 in B = pB
def cost_price_A(A: ℝ) : Prop := let pA := 50 in A = pA
def cost_price_relation(A B: ℝ) : Prop := A = B + 30

-- definitions for part 2
def max_profitA(x: ℝ) (profit: ℝ) : Prop :=
  if 50 ≤ x ∧ x ≤ 60 then profit = (x - 50) * 100
  else if 60 < x ∧ x ≤ 80 then profit = -5 * x^2 + 650 * x - 20000
  else False

def max_profit_overall(profit: ℝ) : Prop :=
  profit = 2480

-- the statements to be proved
theorem find_cost_price(A B: ℝ) (hA : cost_price_A A) (hB: cost_price_B B) : A = 50 ∧ B = 20 := sorry

theorem profit_maximization(x p: ℝ) (h: max_profitA x p) : (x = 65 ∧ p = 1125) := sorry

theorem overall_profit(x: ℝ) (p: ℝ) (hA: cost_price_A A) (hB: cost_price_B B) (h_overall: max_profit_overall p) : p = 2480 := sorry

end find_cost_price_profit_maximization_overall_profit_l531_531855


namespace number_of_questions_in_test_l531_531334

variable (n : ℕ) -- the total number of questions
variable (correct_answers : ℕ) -- the number of correct answers
variable (sections : ℕ) -- number of sections in the test
variable (questions_per_section : ℕ) -- number of questions per section
variable (percentage_correct : ℚ) -- percentage of correct answers

-- Given conditions
def conditions := 
  correct_answers = 32 ∧ 
  sections = 5 ∧ 
  questions_per_section * sections = n ∧ 
  (70 : ℚ) < percentage_correct ∧ 
  percentage_correct < 77 ∧ 
  percentage_correct * n = 3200

-- The main statement to prove
theorem number_of_questions_in_test : conditions n correct_answers sections questions_per_section percentage_correct → 
  n = 45 :=
by
  sorry

end number_of_questions_in_test_l531_531334


namespace simplify_fraction_l531_531295

theorem simplify_fraction :
  ∀ (x y : ℝ), x = 1 → y = 1 → (x^3 + y^3) / (x + y) = 1 :=
by
  intros x y hx hy
  rw [hx, hy]
  simp
  sorry

end simplify_fraction_l531_531295


namespace sum_g_equals_zero_l531_531694

def g (x : ℝ) : ℝ := x^2 * (1 - x)^2

theorem sum_g_equals_zero :
  ∑ k in finset.range(2020).map (λ k, k + 1), (-1)^(k + 1) * g (k / 2021) = 0 := by
  sorry

end sum_g_equals_zero_l531_531694


namespace largest_possible_cos_x_l531_531697

theorem largest_possible_cos_x (x y z : ℝ) (hx : sin x = cos y)
(hy : sin y = sec z) (hz : sin z = cos x) : cos x ≤ 1 :=
sorry

end largest_possible_cos_x_l531_531697


namespace allocation_of_teaching_positions_l531_531023

theorem allocation_of_teaching_positions :
  ∃ n : ℕ, n = 15 ∧
  (∃ (count : list (ℕ × ℕ × ℕ)), 
    count.length = 5 ∧
    (count = [(6,1,1), (5,2,1), (4,3,1), (4,2,2), (3,3,2)] ∧ 
    ∀ (x : ℕ × ℕ × ℕ) (h : x ∈ count), 
      (x = (6,1,1) → 1 = 1) ∧ 
      (x = (5,2,1) → 4 = 4) ∧ 
      (x = (4,3,1) → 4 = 4) ∧ 
      (x = (4,2,2) → 3 = 3) ∧ 
      (x = (3,3,2) → 3 = 3))) :=
by
  have pos := [(6,1,1), (5,2,1), (4,3,1), (4,2,2), (3,3,2)]
  have len_5 := pos.length = 5
  have all_satisfied := ∀ (x : ℕ × ℕ × ℕ) (h : x ∈ pos), 
    (x = (6,1,1) → 1 = 1) ∧ 
    (x = (5,2,1) → 4 = 4) ∧ 
    (x = (4,3,1) → 4 = 4) ∧ 
    (x = (4,2,2) → 3 = 3) ∧ 
    (x = (3,3,2) → 3 = 3)
  existsi 15
  existsi pos
  split
  exact len_5
  split
  exact rfl
  sorry

end allocation_of_teaching_positions_l531_531023


namespace train_length_is_correct_l531_531368

-- Define the given conditions
def speed_km_hr := 40
def time_sec := 17.1
def conversion_factor := 5 / 18

-- Calculate the speed in m/s
def speed_m_per_s := speed_km_hr * conversion_factor

-- Calculate the length of the train
def train_length := speed_m_per_s * time_sec

theorem train_length_is_correct : train_length = 190 :=
by
  -- Prove the statement here (proof omitted)
  sorry

end train_length_is_correct_l531_531368


namespace chord_length_perpendicular_l531_531837

theorem chord_length_perpendicular 
  (R a b : ℝ)  
  (h1 : a + b = R)
  (h2 : (1 / 2) * Real.pi * R^2 - (1 / 2) * Real.pi * (a^2 + b^2) = 10 * Real.pi) :
  2 * Real.sqrt 10 = 6.32 :=
by 
  sorry

end chord_length_perpendicular_l531_531837


namespace diagonal_angle_with_plane_l531_531744

theorem diagonal_angle_with_plane (α : ℝ) {a : ℝ} 
  (h_square: a > 0)
  (θ : ℝ := Real.arcsin ((Real.sin α) / Real.sqrt 2)): 
  ∃ (β : ℝ), β = θ :=
sorry

end diagonal_angle_with_plane_l531_531744


namespace sasha_remainder_is_20_l531_531789

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l531_531789


namespace arithmetic_sequence_sum_l531_531750

theorem arithmetic_sequence_sum : 
  ∃ x y, (∃ d, 
  d = 12 - 5 ∧ 
  19 + d = x ∧ 
  x + d = y ∧ 
  y + d = 40 ∧ 
  x + y = 59) :=
by {
  sorry
}

end arithmetic_sequence_sum_l531_531750


namespace squirrel_travel_distance_l531_531875

theorem squirrel_travel_distance
  (height: ℝ)
  (circumference: ℝ)
  (vertical_rise: ℝ)
  (num_circuits: ℝ):
  height = 25 →
  circumference = 3 →
  vertical_rise = 5 →
  num_circuits = height / vertical_rise →
  (num_circuits * circumference) ^ 2 + height ^ 2 = 850 :=
by
  sorry

end squirrel_travel_distance_l531_531875


namespace area_of_quadrilateral_MNPQ_equal_l531_531985

noncomputable def log_base_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

def f (a b x : ℝ) : ℝ := log_base_half ((a*x + 1) / (x + b))

def are_points_on_graph (a b : ℝ) : Prop :=
  f a b 3 = -1 ∧ f a b (5/3) = -2

def are_vectors_opposite (M N P Q : ℝ × ℝ) : Prop :=
  (N.fst - M.fst = Q.fst - P.fst) ∧ (N.snd - M.snd = Q.snd - P.snd)

def area_parallelogram (M N : ℝ × ℝ) : ℝ :=
  let OM : ℝ × ℝ := (M.fst, M.snd)
  let ON : ℝ × ℝ := (N.fst, N.snd)
  let dot_product := OM.fst * ON.fst + OM.snd * ON.snd
  let magnitude_OM := Math.sqrt (OM.fst^2 + OM.snd^2)
  let magnitude_ON := Math.sqrt (ON.fst^2 + ON.snd^2)
  let cos_theta := dot_product / (magnitude_OM * magnitude_ON)
  let sin_theta := Math.sqrt (1 - cos_theta^2)
  4 * (1/2 * magnitude_OM * magnitude_ON * sin_theta)

theorem area_of_quadrilateral_MNPQ_equal (a b : ℝ) (M N P Q : ℝ × ℝ)
(h1 : are_points_on_graph a b)
(h2 : are_vectors_opposite M N P Q)
(hM : M = (3, -1))
(hN : N = (5/3, -2)) :
  area_parallelogram M N = 26 / 3 := 
sorry

end area_of_quadrilateral_MNPQ_equal_l531_531985


namespace total_cost_of_stamps_is_correct_l531_531882

-- Define the costs of each type of stamp
def cost_of_stamp_A : ℕ := 34 -- cost in cents
def cost_of_stamp_B : ℕ := 52 -- cost in cents
def cost_of_stamp_C : ℕ := 73 -- cost in cents

-- Define the number of stamps Alice needs to buy
def num_stamp_A : ℕ := 4
def num_stamp_B : ℕ := 6
def num_stamp_C : ℕ := 2

-- Define the expected total cost in dollars
def expected_total_cost : ℝ := 5.94

-- State the theorem about the total cost
theorem total_cost_of_stamps_is_correct :
  ((num_stamp_A * cost_of_stamp_A) + (num_stamp_B * cost_of_stamp_B) + (num_stamp_C * cost_of_stamp_C)) / 100 = expected_total_cost :=
by
  sorry

end total_cost_of_stamps_is_correct_l531_531882


namespace cone_sand_weight_l531_531864

noncomputable def cone_weight (diameter : ℝ) (height : ℝ) (density : ℝ) : ℝ :=
  let r := diameter / 2
  let v := (1 / 3) * π * (r ^ 2) * height
  density * v

theorem cone_sand_weight : cone_weight 12 9.6 100 = 11520 * π := 
by
  -- proof omitted
  sorry

end cone_sand_weight_l531_531864


namespace constant_term_binomial_expansion_l531_531305

theorem constant_term_binomial_expansion : 
  (∀ x : ℝ, x ≠ 0 → ( ( ∑ r in finset.range 11, (binom 10 r) * (-1:ℝ)^(10-r) * x^(20 - (5*r)/2))
    .filter (λ t, t = (0 : ℝ))).sum = 45) := 
by
  sorry

end constant_term_binomial_expansion_l531_531305


namespace probability_of_difference_three_l531_531491

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l531_531491


namespace simplify_tan_cot_expression_l531_531293

theorem simplify_tan_cot_expression
  (h1 : Real.tan (Real.pi / 4) = 1)
  (h2 : Real.cot (Real.pi / 4) = 1) :
  (Real.tan (Real.pi / 4))^3 + (Real.cot (Real.pi / 4))^3 = 1 := by
  sorry

end simplify_tan_cot_expression_l531_531293


namespace airport_distance_l531_531094

theorem airport_distance (d t : ℝ) (h1 : d = 45 * (t + 0.75))
                         (h2 : d - 45 = 65 * (t - 1.25)) :
  d = 241.875 :=
by
  sorry

end airport_distance_l531_531094


namespace oreo_shop_total_ways_l531_531507

def oreo_shop_ways : ℕ :=
  let flavors_oreos := 6
  let flavors_milk := 3
  let total_items := 3
  let total_choices := flavors_oreos + flavors_milk
  let no_item_more_than := 2
  
  if total_items > total_choices then 0 else
    -- Total ways of purchasing 3 items collectively given conditions
    ∑ i in [0, 1, 2, 3], 
      (if i ≤ total_items then 
        (Nat.choose total_choices i) * (Nat.choose total_choices (total_items - i))
      else 0)
  
theorem oreo_shop_total_ways : oreo_shop_ways = 708 := by
  sorry

end oreo_shop_total_ways_l531_531507


namespace mole_fraction_partial_pressure_of_N2O5_l531_531967

variables 
  (N2O5 O2 N2 : Type)
  (n_N2O5 n_total : ℝ) -- number of moles of N2O5 and total moles 
  (P : ℝ)       -- total pressure
  (X_N2O5 : ℝ)  -- mole fraction of N2O5
  (P_N2O5 : ℝ)  -- partial pressure of N2O5

theorem mole_fraction_partial_pressure_of_N2O5 
  (h1 : X_N2O5 = n_N2O5 / n_total)
  (h2 : P_N2O5 = X_N2O5 * P) : 
  (X_N2O5 = n_N2O5 / n_total) ∧ (P_N2O5 = n_N2O5 / n_total * P) :=
by {
  split,
  { exact h1 },
  { rw [h1],
    exact h2 },
}

end mole_fraction_partial_pressure_of_N2O5_l531_531967


namespace runner_speed_ratio_l531_531349

theorem runner_speed_ratio (d s u v_f v_s : ℝ) (hs : s ≠ 0) (hu : u ≠ 0)
  (H1 : (v_f + v_s) * s = d) (H2 : (v_f - v_s) * u = v_s * u) :
  v_f / v_s = 2 :=
by
  sorry

end runner_speed_ratio_l531_531349


namespace product_of_solutions_with_positive_real_parts_l531_531204

open Complex

/-- Prove that the product of the solutions with positive real parts of the equation x^8 = -256 is 8. -/
theorem product_of_solutions_with_positive_real_parts :
  ∀ x : ℂ, x^8 = -256 → re x > 0 → 
  ∃ y : ℂ, re y > 0 ∧ ∏ (z : ℂ) in {z | z^8 = -256 ∧ re z > 0}.to_finset = 8 := 
sorry

end product_of_solutions_with_positive_real_parts_l531_531204


namespace solve_log_equation_l531_531005

theorem solve_log_equation (k : ℤ) (x : ℝ) :
  log (sin x) / log (cos x) - 2 * log (cos x) / log (sin x) + 1 = 0 →
  ∃ n : ℤ, x = π / 4 + n * π :=
sorry

end solve_log_equation_l531_531005


namespace taxi_fare_proportionality_l531_531880

theorem taxi_fare_proportionality (h : (∃ k, ∀ d: ℕ, d > 0 → fare d = k * d) ∧ fare 80 = 200) : fare 120 = 300 :=
sorry

end taxi_fare_proportionality_l531_531880


namespace closest_to_fraction_l531_531111

theorem closest_to_fraction (n d : ℝ) (h_n : n = 510) (h_d : d = 0.125) :
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 5000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 6000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 7000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 8000 :=
by
  sorry

end closest_to_fraction_l531_531111


namespace min_value_f_l531_531961

theorem min_value_f
  (a b c : ℝ)
  (α β γ : ℤ)
  (hα : α = 1 ∨ α = -1)
  (hβ : β = 1 ∨ β = -1)
  (hγ : γ = 1 ∨ γ = -1)
  (h : a * α + b * β + c * γ = 0) :
  (∃ f_min : ℝ, f_min = ( ((a ^ 3 + b ^ 3 + c ^ 3) / (a * b * c)) ^ 2) ∧ f_min = 9) :=
sorry

end min_value_f_l531_531961


namespace probability_of_difference_three_l531_531492

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l531_531492


namespace true_propositions_are_l531_531994

-- Define the propositions as separate statements
def proposition1 (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x - 1) = f(x + 1) → ¬ (∀ x, f (x - 1) = f (2 - x))

def proposition2 : Prop :=
  ∃ p q : ℝ × ℝ, p = (2, 1) ∧ q = (0, 3) ∧
  let midpoint := ((fst p + fst q) / 2, (snd p + snd q) / 2) in
  let line := (x : ℝ) * (1 : ℝ) - (y : ℝ) + (1 : ℝ) = 0 in
  line (fst midpoint)

def proposition3 : Prop :=
  ∃ y x b a : ℝ, y = b * x + a

def proposition4 (x : ℝ) : Prop :=
  let f := (sin (x^2 + 1)) in
  ¬ (∀ x, f (-x) = -f (x))

-- The main theorem to state which propositions are true
theorem true_propositions_are :
  proposition2 ∧ proposition3 ∧ ¬ proposition1 ∧ ¬ proposition4 :=
by
  -- Proof omitted
  sorry

end true_propositions_are_l531_531994


namespace units_digit_27_mul_46_l531_531556

theorem units_digit_27_mul_46 : (27 * 46) % 10 = 2 :=
by 
  -- Definition of units digit
  have def_units_digit :=  (n : ℕ) => n % 10

  -- Step 1: units digit of 27 is 7
  have units_digit_27 : 27 % 10 = 7 := by norm_num
  
  -- Step 2: units digit of 46 is 6
  have units_digit_46 : 46 % 10 = 6 := by norm_num

  -- Step 3: multiple the units digits
  have step3 : 7 * 6 = 42 := by norm_num

  -- Step 4: Find the units digit of 42
  have units_digit_42 : 42 % 10 = 2 := by norm_num

  exact units_digit_42

end units_digit_27_mul_46_l531_531556


namespace probability_of_difference_three_l531_531490

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l531_531490


namespace distance_inequality_l531_531685

open BigOperators

noncomputable def point : Type := ℝ × ℝ

noncomputable def dist (p q : point) : ℝ := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def angle (A B C : point) : ℝ :=
let v1 := (B.1 - A.1, B.2 - A.2) in
let v2 := (C.1 - A.1, C.2 - A.2) in
real.acos ((v1.1 * v2.1 + v1.2 * v2.2) / (dist A B * dist A C))

variables (A B C M P : point)
variable (h1 : angle M A B = 120)
variable (h2 : angle M B C = 120)
variable (h3 : angle M C A = 120)

theorem distance_inequality : dist P A + dist P B + dist P C ≥ dist M A + dist M B + dist M C :=
sorry

end distance_inequality_l531_531685


namespace simplest_square_root_l531_531364

theorem simplest_square_root :
  ∀ (a b c d : Real),
  a = Real.sqrt 0.2 →
  b = Real.sqrt (1 / 2) →
  c = Real.sqrt 6 →
  d = Real.sqrt 12 →
  c = Real.sqrt 6 :=
by
  intros a b c d ha hb hc hd
  simp [ha, hb, hc, hd]
  sorry

end simplest_square_root_l531_531364


namespace probability_differ_by_three_is_one_sixth_l531_531465

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l531_531465


namespace eggs_in_box_l531_531799

theorem eggs_in_box (initial_count : ℝ) (added_count : ℝ) (total_count : ℝ) 
  (h_initial : initial_count = 47.0) 
  (h_added : added_count = 5.0) : total_count = 52.0 :=
by 
  sorry

end eggs_in_box_l531_531799


namespace max_min_values_of_f_l531_531741

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_min_values_of_f :
  (∀ x : ℝ, f x ≤ 2) ∧ (∃ x : ℝ, f x = 2) ∧
  (∀ x : ℝ, -2 ≤ f x) ∧ (∃ x : ℝ, f x = -2) :=
by 
  sorry

end max_min_values_of_f_l531_531741


namespace find_z_coordinate_of_point_l531_531403

noncomputable def line_passing_through_points : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → ℝ → (ℝ × ℝ × ℝ)
| (x1, y1, z1), (x2, y2, z2) :=
  λ t, ⟨x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1)⟩

theorem find_z_coordinate_of_point :
  ∃ z, (line_passing_through_points (1, 4, 3) (4, 3, 0) 2).fst = 7 ∧
       (line_passing_through_points (1, 4, 3) (4, 3, 0) 2).snd.snd.fst = z :=
begin
  use -3,
  unfold line_passing_through_points,
  simp,
  sorry
end

end find_z_coordinate_of_point_l531_531403


namespace device_failure_probability_l531_531854

noncomputable def probability_fail_device (p1 p2 p3 : ℝ) (p_one p_two p_three : ℝ) : ℝ :=
  0.006 * p3 + 0.092 * p_two + 0.398 * p_one

theorem device_failure_probability
  (p1 p2 p3 : ℝ) (p_one p_two p_three : ℝ)
  (h1 : p1 = 0.1)
  (h2 : p2 = 0.2)
  (h3 : p3 = 0.3)
  (h4 : p_one = 0.25)
  (h5 : p_two = 0.6)
  (h6 : p_three = 0.9) :
  probability_fail_device p1 p2 p3 p_one p_two p_three = 0.1601 :=
by
  sorry

end device_failure_probability_l531_531854


namespace geometric_triangle_sine_l531_531321

theorem geometric_triangle_sine (m n k : ℤ) (a r : ℝ) (h1 : a > 0) (h2 : r > 0) (h3 : m + √n = 1 - √5) (h4 : k = 2) (h5 : ¬ (∃ p : ℕ, prime p ∧ p^2 ∣ k)) :
  m + n + k = 6 :=
sorry

end geometric_triangle_sine_l531_531321


namespace cos_double_angle_l531_531173

theorem cos_double_angle (α : ℝ) (P : ℝ × ℝ)
  (h1 : P = (-3/5, 4/5))
  (h2 : ∀ θ, θ = α → θ = real.atan2 P.2 P.1) :
  real.cos (2 * α) = -7 / 25 :=
by
  -- placeholder for proof
  sorry

end cos_double_angle_l531_531173


namespace required_oranges_for_juice_l531_531849

theorem required_oranges_for_juice (oranges quarts : ℚ) (h : oranges = 36 ∧ quarts = 48) :
  ∃ x, ((oranges / quarts) = (x / 6) ∧ x = 4.5) := 
by sorry

end required_oranges_for_juice_l531_531849


namespace initial_state1_cannot_lead_to_final_state_initial_state2_can_lead_to_final_state_l531_531367

-- Definitions for the problem conditions
def transformation (x y z : ℕ) : ℕ × ℕ := (y, z, x + y - 1)

-- Initial and final states
def initial_state1 := (2, 2, 2)
def initial_state2 := (3, 3, 3)
def final_state := (17, 1967, 1983)

-- Proof problems
theorem initial_state1_cannot_lead_to_final_state : 
  ¬ ∃ (n: ℕ), ((transformation^ n) initial_state1) = final_state :=
sorry

theorem initial_state2_can_lead_to_final_state : 
  ∃ (n: ℕ), ((transformation^ n) initial_state2) = final_state :=
sorry

end initial_state1_cannot_lead_to_final_state_initial_state2_can_lead_to_final_state_l531_531367


namespace equilateral_triangle_in_ellipse_side_length_squared_l531_531066

noncomputable def square_of_side_length : ℚ :=
  1475 / 196

theorem equilateral_triangle_in_ellipse_side_length_squared
  (A B C : ℝ × ℝ)
  (ellipse : ∀ (x y : ℝ), 9 * x ^ 2 + 25 * y ^ 2 = 225)
  (vertex_A : A = (5/3, 0))
  (altitude_on_x_axis : B.2 = - C.2)
  (equilateral : ∀ P Q R : ℝ × ℝ, 
    (P = A ∨ P = B ∨ P = C) →
    (Q = A ∨ Q = B ∨ Q = C) →
    (R = A ∨ R = B ∨ R = C) →
    P ≠ Q → len P Q = len Q R ∧ len Q R = len R P) :
  len A B ^ 2 = square_of_side_length :=
by
  sorry

-- Helper function to compute the length between two points
noncomputable def len (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

end equilateral_triangle_in_ellipse_side_length_squared_l531_531066


namespace intersection_A_B_l531_531260

-- Define sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

-- The theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 3} :=
by
  sorry -- proof is skipped as instructed

end intersection_A_B_l531_531260


namespace geometric_sequence_product_condition_l531_531654

theorem geometric_sequence_product_condition (a : ℕ → ℝ) (q : ℝ) (m : ℕ) :
  (∀ n, a n = q ^ (n - 1)) →
  a 2 = 4 → a 4 = 16 →
  (∏ i in finset.range (m+1), a (i+1)) = (a (m+1))^2 →
  m = 4 :=
by
  sorry

end geometric_sequence_product_condition_l531_531654


namespace probability_of_diff_3_is_1_over_9_l531_531426

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l531_531426


namespace part1_f_0_part1_f_odd_part2_f_decreasing_part3_f_range_l531_531964

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (x y : ℝ) : f(x + y) = f(x) + f(y)
axiom f_neg_pos (x : ℝ) (hx : 0 < x) : f(x) < 0
axiom f_neg_one : f(-1) = 2

theorem part1_f_0 : f(0) = 0 :=
sorry

theorem part1_f_odd : ∀ x : ℝ, f(-x) = -f(x) :=
sorry

theorem part2_f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁) > f(x₂) :=
sorry

theorem part3_f_range : set.range f ∩ set.Icc (-2 : ℝ) 4 = set.Icc (-8 : ℝ) 4 :=
sorry

end part1_f_0_part1_f_odd_part2_f_decreasing_part3_f_range_l531_531964


namespace area_of_hexagon_correct_l531_531245

open Real

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def area_of_hexagon_ABQCDP (A B C D P Q : ℝ × ℝ) : ℝ :=
let AB := distance (A.1) (A.2) (B.1) (B.2),
    CD := distance (C.1) (C.2) (D.1) (D.2),
    height := (P.2 - A.2) in
(AB + CD) * height / 2 
- (1 / 2) * (distance (A.1) (A.2) (B.1) (B.2)) * height 
- (1 / 2) * (distance (B.1) (B.2) (C.1) (C.2)) * height

theorem area_of_hexagon_correct :
  let A := (0, 0),
      B := (15, 0),
      D := (5, 2 * sqrt 14),
      C := (20, 2 * sqrt 14),
      P := (5, 2 * sqrt 14),
      Q := (7, 2 * sqrt 14)
  in area_of_hexagon_ABQCDP A B C D P Q = 28 * sqrt 14 :=
by
  sorry

end area_of_hexagon_correct_l531_531245


namespace polynomial_multiplication_l531_531326

theorem polynomial_multiplication (x a : ℝ) : (x - a) * (x^2 + a * x + a^2) = x^3 - a^3 :=
by
  sorry

end polynomial_multiplication_l531_531326


namespace problem1_problem2_l531_531186

noncomputable def A : set ℝ := set.Ico 0 (1 / 2)
noncomputable def B : set ℝ := set.Icc (1 / 2) 1

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ A then x + (1 / 2) else 2 * (1 - x)

theorem problem1 : f (f (1 / 3)) = 1 / 3 :=
by
  sorry

theorem problem2 (x₀ : ℝ) (hx₀ : x₀ ∈ A) (hffx₀ : f (f x₀) ∈ A) :
  1 / 4 < x₀ ∧ x₀ < 1 / 2 :=
by
  sorry

end problem1_problem2_l531_531186


namespace curve_touches_limit_sum_S_l531_531850

def f_n (a_n : ℝ) (n : ℕ) (x : ℝ) : ℝ := a_n * (x - n) * (n + 1 - x)

def g (x : ℝ) : ℝ := real.exp (-x)

noncomputable def a_n (n : ℕ) : ℝ := real.exp (-(n + (real.sqrt 5 - 1) / 2)) / (real.sqrt 5 - 2)

theorem curve_touches (n : ℕ) (x : ℝ) :
  f_n (a_n n) n x = g x ∧ deriv (f_n (a_n n) n) x = deriv g x :=
sorry

def S_0 : ℝ := ∫ x in 0..1, min (f_n (a_n 0) 0 x) (g x)

noncomputable def S_n (n : ℕ) : ℝ :=
  ∫ x in (n:ℝ)..(n+1:ℝ), min (min (f_n (a_n (n-1)) (n-1) x) (f_n (a_n n) n x)) (g x)

noncomputable def sum_S (n : ℕ) : ℝ :=
  S_0 + (∑ i in finset.range (n+1), S_n i)

theorem limit_sum_S :
  tendsto sum_S at_top (𝓝 (1 - real.exp ((3 - real.sqrt 5) / 2) / (6 * (real.sqrt 5 - 2) * (real.exp 1 - 1)))) :=
sorry

end curve_touches_limit_sum_S_l531_531850


namespace max_min_A_l531_531257

open Complex

noncomputable def A (z : ℂ) : ℝ :=
  (z.re : ℝ) * ((abs (z - I))^2 - 1)

theorem max_min_A :
  let A := A
  in ∀ z : ℂ, abs (z - I) ≤ 1 →
  ∃ (max min : ℝ), 
    max = (2 * real.sqrt 3 / 9) ∧ min = (-2 * real.sqrt 3 / 9) ∧
    (∀ z : ℂ, abs (z - I) ≤ 1 → A z ≤ max) ∧
    (∀ z : ℂ, abs (z - I) ≤ 1 → A z ≥ min) :=
by
  sorry

end max_min_A_l531_531257


namespace split_cape_town_coins_l531_531397

noncomputable def is_cape_town_coin (n : ℕ) : Prop :=
  ∃ m > 0, n = 1 / m

noncomputable def total_value_le (coins : list ℝ) (value : ℝ) : Prop :=
  (coins.sum ≤ value)

noncomputable def can_split_into_groups (coins : list ℝ) (max_groups : ℕ) (max_value_per_group : ℝ) : Prop :=
  ∃ groups : list (list ℝ),
    groups.length ≤ max_groups ∧
    (∀ g ∈ groups, g.sum ≤ max_value_per_group)

theorem split_cape_town_coins (coins : list ℝ) (h_ctc : ∀ c ∈ coins, ∃ n > 0, c = 1 / n)
  (h_value : coins.sum ≤ 99.5) : can_split_into_groups coins 100 1 :=
sorry

end split_cape_town_coins_l531_531397


namespace hotdogs_sold_l531_531041

-- Definitions of initial and remaining hotdogs
def initial : ℕ := 99
def remaining : ℕ := 97

-- The statement that needs to be proven
theorem hotdogs_sold : initial - remaining = 2 :=
by
  sorry

end hotdogs_sold_l531_531041


namespace sasha_remainder_l531_531761

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l531_531761


namespace part1_f1_is_zero_part1_f_is_increasing_part2_solve_inequality_l531_531986

-- Definitions
variable {f : ℝ → ℝ} 
variable (xy_relation : ∀ x y, f(x * y) = f(x) + f(y)) 
variable (domain_pos : ∀ x, 0 < x → x ∈ (0, +∞))
variable (f_pos : ∀ x, 1 < x → 0 < f(x))
variable (f_two : f 2 = 1)

-- Proof statements
theorem part1_f1_is_zero : f 1 = 0 := 
by 
  sorry

theorem part1_f_is_increasing : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 > x2 → f x1 > f x2 := 
by 
  sorry

theorem part2_solve_inequality : ∀ x, f (-x) + f (3 - x) ≥ -2 ↔ x ≤ (3 - (√10))/2 := 
by 
  sorry

end part1_f1_is_zero_part1_f_is_increasing_part2_solve_inequality_l531_531986


namespace cos_theta_value_l531_531172

-- Define the point P
def P : ℝ × ℝ := (-1, -Real.sqrt 3)

-- Define the distance r from the origin to point P
noncomputable def r (P : ℝ × ℝ) : ℝ :=
  Real.sqrt (P.1 ^ 2 + P.2 ^ 2)

-- Define cosine function for the given point P
noncomputable def cos_θ (P : ℝ × ℝ) : ℝ :=
  P.1 / r P

-- Define the mathematical problem to be proved
theorem cos_theta_value (P : ℝ × ℝ) (hP : P = (-1, -Real.sqrt 3)) :
  cos_θ P = -1 / 2 :=
by
  sorry

end cos_theta_value_l531_531172


namespace ravon_has_card_4_l531_531708

-- Defining the structure with the players' scores and available cards
structure CardGame :=
  (cards : finset ℕ) -- set of cards
  (scores : ℕ → ℕ) -- scores for each player ID

-- Constants representing the players
def Ravon := 0
def Oscar := 1
def Aditi := 2
def Tyrone := 3
def Kim := 4

-- Initializing the card game with scores and available cards
def cardGame : CardGame :=
  { cards := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    scores := λ n, match n with
                    | 0 => 11
                    | 1 => 4
                    | 2 => 7
                    | 3 => 16
                    | 4 => 17
                    | _ => 0
                  end
  }

-- Main goal statement
theorem ravon_has_card_4 : 
  ∃ (card1 card2 : ℕ), card1 ≠ card2 ∧ card1 ∈ cardGame.cards ∧ card2 ∈ cardGame.cards ∧ card1 + card2 = cardGame.scores Ravon ∧ (card1 = 4 ∨ card2 = 4) :=
  by
  sorry

end ravon_has_card_4_l531_531708


namespace boris_number_of_bowls_l531_531896

-- Definitions from the conditions
def total_candies : ℕ := 100
def daughter_eats : ℕ := 8
def candies_per_bowl_after_removal : ℕ := 20
def candies_removed_per_bowl : ℕ := 3

-- Derived definitions
def remaining_candies : ℕ := total_candies - daughter_eats
def candies_per_bowl_orig : ℕ := candies_per_bowl_after_removal + candies_removed_per_bowl

-- Statement to prove
theorem boris_number_of_bowls : remaining_candies / candies_per_bowl_orig = 4 :=
by sorry

end boris_number_of_bowls_l531_531896


namespace most_suitable_survey_l531_531366

-- Define the options as a type
inductive SurveyOption
| A -- Understanding the crash resistance of a batch of cars
| B -- Surveying the awareness of the "one helmet, one belt" traffic regulations among citizens in our city
| C -- Surveying the service life of light bulbs produced by a factory
| D -- Surveying the quality of components of the latest stealth fighter in our country

-- Define a function determining the most suitable for a comprehensive survey
def mostSuitableForCensus : SurveyOption :=
  SurveyOption.D

-- Theorem statement that Option D is the most suitable for a comprehensive survey
theorem most_suitable_survey :
  mostSuitableForCensus = SurveyOption.D :=
  sorry

end most_suitable_survey_l531_531366


namespace selection_schemes_correct_l531_531138

-- Define the problem parameters
def number_of_selection_schemes (persons : ℕ) (cities : ℕ) (persons_cannot_visit : ℕ) : ℕ :=
  let choices_for_paris := persons - persons_cannot_visit
  let remaining_people := persons - 1
  choices_for_paris * remaining_people * (remaining_people - 1) * (remaining_people - 2)

-- Define the example constants
def total_people : ℕ := 6
def total_cities : ℕ := 4
def cannot_visit_paris : ℕ := 2

-- The statement to be proved
theorem selection_schemes_correct : 
  number_of_selection_schemes total_people total_cities cannot_visit_paris = 240 := by
  sorry

end selection_schemes_correct_l531_531138


namespace dice_diff_by_three_probability_l531_531434

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l531_531434


namespace necessary_not_sufficient_to_form_triangle_l531_531004

-- Define the vectors and the condition
variables (a b c : ℝ × ℝ)

-- Define the condition that these vectors form a closed loop (triangle)
def forms_closed_loop (a b c : ℝ × ℝ) : Prop :=
  a + b + c = (0, 0)

-- Prove that the condition is necessary but not sufficient
theorem necessary_not_sufficient_to_form_triangle :
  forms_closed_loop a b c → ∃ (x : ℝ × ℝ), a ≠ x ∧ b ≠ -2 * x ∧ c ≠ x :=
sorry

end necessary_not_sufficient_to_form_triangle_l531_531004


namespace correct_statements_genetic_engineering_l531_531281

theorem correct_statements_genetic_engineering :
  let s1 := "The enzymes used in recombinant DNA technology are restriction enzymes, DNA ligase, and vectors."
  let s2 := "Genetic engineering is the design and construction at the molecular level of DNA."
  let s3 := "The cutting site of restriction enzymes must be the base sequence GAATTC."
  let s4 := "As long as the target gene enters the receptor cell, it can successfully achieve expression."
  let s5 := "All restriction enzymes can only recognize the same specific nucleotide sequence."
  let s6 := "Genetic engineering can transplant excellent traits from one organism to another."
  let s7 := "Plasmids are the only vectors used to carry target genes in genetic engineering."
  let correct := [s2, s6]
in correct.length = 2 := 
by {
  let correct_statements := [s2, s6],
  have h : correct_statements.length = 2 := by rfl,
  exact h,
  sorry
}

end correct_statements_genetic_engineering_l531_531281


namespace exists_exactly_four_separators_l531_531856

/-- A circle is called a separator for a set of five points in a plane if it passes through three of these points, contains a fourth point inside, and the fifth point is outside the circle. -/
structure separator (A B C D E : Point) (c : Circle) : Prop :=
  (passes_through_three_points : c.passes_through A ∧ c.passes_through B ∧ c.passes_through C)
  (contains_fourth_point_inside : c.contains_point_inside D)
  (fifth_point_outside : c.contains_point_outside E)

/-- Given five points in a plane with no three collinear and no four concyclic, there exist exactly four separators. -/
theorem exists_exactly_four_separators (A B C D E : Point) 
  (h1 : no_three_collinear {A, B, C, D, E}) 
  (h2 : no_four_concyclic {A, B, C, D, E}) :
  ∃ separators : finset (Circle), separator_count separators 4 :=
sorry

end exists_exactly_four_separators_l531_531856


namespace probability_differ_by_three_is_one_sixth_l531_531460

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l531_531460


namespace slope_angle_range_l531_531645

-- Given conditions expressed as Lean definitions
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 18

def line_eq (a b x y : ℝ) : Prop := a * x + b * y = 0

def distance_point_to_line (a b x y d : ℝ) : Prop := 
  abs (a * x + b * y) / sqrt (a^2 + b^2) = d

-- Given conditions specified in Lean
constant a b : ℝ
constant α : ℝ
constant slope_ranges : ∀ m : ℝ, 2 - sqrt 3 ≤ m ∧ m ≤ 2 + sqrt 3 → 
                                  ∀ α : ℝ, tan α = m → 
                                  π / 12 ≤ α ∧ α ≤ 5 * π / 12

-- Main theorem to prove
theorem slope_angle_range (h_circle : ∀ x y, circle_eq x y) 
                          (h_distance : ∃ x y, circle_eq x y ∧ distance_point_to_line a b x y (2 * sqrt 2)) 
                          (h_inequality : (a / b)^2 + 4 * (a / b) + 1 ≤ 0) : 
                          π / 12 ≤ α ∧ α ≤ 5 * π / 12 := 
begin
  sorry
end

end slope_angle_range_l531_531645


namespace f_eq_zero_all_l531_531928

-- Define function f from positive natural numbers to natural numbers
def f : ℕ+ → ℕ

-- Given properties of the function f
axiom f_mul_add (x y : ℕ+) : f (x * y) = f x + f y
axiom f_30_zero : f 30 = 0
axiom f_units_digit_7 (x : ℕ+) : (x % 10 = 7) → f x = 0

-- Prove that f(n) = 0 for all positive natural numbers n
theorem f_eq_zero_all (n : ℕ+) : f n = 0 := sorry

end f_eq_zero_all_l531_531928


namespace base4_addition_l531_531527

def base4_sum : ℕ → ℕ → ℕ → ℕ
| a b c := sorry

theorem base4_addition (a b c : ℕ) (sum : ℕ) : base4_sum 232 121 313 = 1332 :=
sorry

end base4_addition_l531_531527


namespace angle_BEC_is_110_degrees_l531_531902

/-- Given: 
1. Triangle ABC with an inscribed circle Ω.
2. ∠BAC = 40°
3. Point D is inside angle BAC and is the A-excenter (intersection of exterior bisectors 
   of angles B and C).
4. Tangent from D touches Ω at E.

To Prove: ∠BEC = 110° --/
theorem angle_BEC_is_110_degrees
  (A B C D E : Point)
  (Omega : Circle)
  (BAC : angle A B C)
  (h1: inscribed_circle Omega (triangle A B C))
  (h2 : BAC.angle = 40)
  (h3 : is_A_excenter D (triangle A B C))
  (h4 : tangent D Omega E) :
  angle B E C = 110 := 
sorry

end angle_BEC_is_110_degrees_l531_531902


namespace candies_per_friend_l531_531901

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (friends : ℕ) 
  (h_initial : initial_candies = 10)
  (h_additional : additional_candies = 4)
  (h_friends : friends = 7) : initial_candies + additional_candies = 14 ∧ 14 / friends = 2 :=
by
  sorry

end candies_per_friend_l531_531901


namespace sum_of_three_different_squares_l531_531214

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def existing_list (ns : List Nat) : Prop :=
  ∀ n ∈ ns, is_perfect_square n

theorem sum_of_three_different_squares (a b c : Nat) :
  existing_list [a, b, c] →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 128 →
  false :=
by
  intros
  sorry

end sum_of_three_different_squares_l531_531214


namespace evaluate_expression_l531_531540

theorem evaluate_expression : 6^4 - 4 * 6^3 + 6^2 - 2 * 6 + 1 = 457 := 
  by
    sorry

end evaluate_expression_l531_531540


namespace mn_parallel_bc_l531_531072

open EuclideanGeometry

theorem mn_parallel_bc (A B C D F E G H N M : Point)
  [Circumcircle : Circumcircle ABC O]
  (D_angle_bisector : OnAngleBisector D A B C)
  (BD_intersect_AC_at_F : IntersectionPoint BD AC F)
  (CD_intersect_AB_at_E : IntersectionPoint CD AB E)
  (EF_intersect_circumcircle_GH : EF.IntersectsCircumcircleAt G H E)
  (G_is_on_GF : OnSegment G F E)
  (GD_intersect_circumcircle_N : GD.IntersectsCircumcircleAt N)
  (HD_intersect_circumcircle_M : HD.IntersectsCircumcircleAt M):
  Parallel MN BC := by
  sorry

end mn_parallel_bc_l531_531072


namespace correlation_relationship_of_PhenomenonA_l531_531000

-- Define the conditions as propositions
def PhenomenonA : Prop := ∀ (family_income consumption : ℝ), family_income > 0 → consumption > 0
def PhenomenonB : Prop := ∀ (r : ℝ), r > 0 → (circle_area r = π * r * r)
def PhenomenonC : Prop := ∀ (volume temperature : ℝ), temperature > 0 → volume > 0
def PhenomenonD : Prop := ∀ (price quantity sales_revenue : ℝ), price * quantity = sales_revenue

-- Define the correlation relationship property
def is_correlation (P : Prop) : Prop := P -- You would typically define what makes a correlation relationship precisely.

-- State the theorem to prove
theorem correlation_relationship_of_PhenomenonA (P_corr : is_correlation PhenomenonA) : 
  ∃ P : Prop, P = PhenomenonA ∧ is_correlation P := 
by 
  sorry

end correlation_relationship_of_PhenomenonA_l531_531000


namespace solve_for_x_l531_531845

theorem solve_for_x (x : ℝ) (h : 3 * x = 16 - x + 4) : x = 5 := 
by
  sorry

end solve_for_x_l531_531845


namespace yellow_crayons_count_l531_531797

def red_crayons := 14
def blue_crayons := red_crayons + 5
def yellow_crayons := 2 * blue_crayons - 6

theorem yellow_crayons_count : yellow_crayons = 32 := by
  sorry

end yellow_crayons_count_l531_531797


namespace probability_even_sum_is_half_l531_531890

-- Let us define the conditions in Lean
def first_wheel_probs : fin 6 → ℚ
| ⟨0, _⟩ := 1 / 6
| ⟨1, _⟩ := 1 / 6
| ⟨2, _⟩ := 1 / 6
| ⟨3, _⟩ := 1 / 6
| ⟨4, _⟩ := 1 / 6
| ⟨5, _⟩ := 1 / 6

def second_wheel_probs : fin 5 → ℚ
| ⟨0, _⟩ := 2 / 10
| ⟨1, _⟩ := 2 / 10
| ⟨2, _⟩ := 2 / 10
| ⟨3, _⟩ := 2 / 10
| ⟨4, _⟩ := 2 / 10

-- Define helper functions to determine if numbers are even/odd
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the event of getting even sum
def even_sum_event : Prop :=
  ∑ i in fin 6, ∑ j in fin 5, ite (is_even (i + j)) 
    (first_wheel_probs i * second_wheel_probs j) 
    0 = 1/2

theorem probability_even_sum_is_half : even_sum_event := 
by sorry

end probability_even_sum_is_half_l531_531890


namespace count_odd_f_m_l531_531254

def f (n : ℕ) : ℕ :=
  if n < 2 then 0 else
  if n < 4 then 1 else
  if n < 16 then 2 else
  if n < 256 then 3 else
  if n < 65536 then 4 else
  if n < 2^32 then 5 else 6

theorem count_odd_f_m :
  let count_odd_f := (λ m, 1 < m ∧ m < 2008 ∧ (f m) % 2 = 1)
  ∃ n, n = (∑ m in Ico 2 2008, if (count_odd_f m) then 1 else 0) ∧ n = 242 :=
by
  sorry

end count_odd_f_m_l531_531254


namespace sum_of_infinite_series_l531_531128

theorem sum_of_infinite_series :
  let series := (λ n : ℕ, (1 / (n * (n + 1) : ℝ)) - (1 / ((n + 1) * (n + 2) : ℝ)))
  ∑' n, series (n + 1) = 1 / 2 :=
by
  sorry

end sum_of_infinite_series_l531_531128


namespace difference_of_a_and_e_l531_531943

-- Define the problem
theorem difference_of_a_and_e (a b c d e : ℕ) 
  (h1 : a * b + a + b = 182) 
  (h2 : b * c + b + c = 306) 
  (h3 : c * d + c + d = 210) 
  (h4 : d * e + d + e = 156) 
  (h5 : a * b * c * d * e = 10 !) 
  : a - e = -154 :=
sorry

end difference_of_a_and_e_l531_531943


namespace find_sin_theta_l531_531687

noncomputable def direction_vector : ℝ^3 := ⟨4, 5, 7⟩
noncomputable def normal_vector : ℝ^3 := ⟨6, 3, -2⟩

def dot_product (v1 v2 : ℝ^3) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def norm (v : ℝ^3) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def cos_theta (d n : ℝ^3) : ℝ :=
  dot_product d n / (norm d * norm n)

def sin_theta (d n : ℝ^3) : ℝ :=
  real.sqrt (1 - cos_theta d n ^ 2)

theorem find_sin_theta :
  sin_theta direction_vector normal_vector = 25 / (21 * real.sqrt 10) :=
sorry

end find_sin_theta_l531_531687


namespace angle_and_area_of_triangle_l531_531232

variable {A B C : ℝ}
variable {a b c S : ℝ}

-- Conditions
def triangle_sides (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ (a b c : ℝ) (A B C : ℝ), A + B + C = π ∧ a = 5 ∧ S = (√3 / 2) * (b * c * cos A)

-- Part I: Determine the magnitude of angle A
def angle_A_eq_pi_over_3 (A : ℝ) : Prop := -- Angle A equals π/3
  A = π / 3

-- Part II: Find the area of the triangle with given values of sides and angles
def triangle_area (b c S : ℝ) : Prop :=
  let sin_A := sin (π / 3) in
  let bc := 6 in
  S = (1 / 2) * b * c * sin_A

theorem angle_and_area_of_triangle :
  triangle_sides a b c A B C S ∧ (b + c = 5) ∧ (a = √7) → angle_A_eq_pi_over_3 A ∧ triangle_area b c S := by
  sorry

end angle_and_area_of_triangle_l531_531232


namespace equal_tuesdays_thursdays_in_30_day_month_l531_531863

theorem equal_tuesdays_thursdays_in_30_day_month : 
  ∃ (days : Finset ℕ), days.card = 4 ∧ 
  ∀ d ∈ days, 
  let tuesdays := (0..4).count (λ i, (d + i * 7 + 2) % 7 = 2) in -- count Tuesdays
  let thursdays := (0..4).count (λ i, (d + i * 7 + 2) % 7 = 4) in -- count Thursdays
  tuesdays = thursdays := 
by
  sorry

end equal_tuesdays_thursdays_in_30_day_month_l531_531863


namespace chord_length_l531_531838

theorem chord_length (R a b : ℝ) (hR : a + b = R) (hab : a * b = 10) 
    (h_nonneg : 0 ≤ R ∧ 0 ≤ a ∧ 0 ≤ b) : ∃ L : ℝ, L = 2 * Real.sqrt 10 :=
by
  sorry

end chord_length_l531_531838


namespace length_of_hall_is_36_l531_531402

/-- Conditions -/
def breadth_of_hall : ℝ := 15
def stone_length_dm : ℝ := 4
def stone_width_dm : ℝ := 5
def number_of_stones : ℕ := 2700

/-- Conversion from decimeters to meters -/
def dm_to_m (d : ℝ) : ℝ := d * 0.1

/-- Calculation of stone dimensions in meters -/
def stone_length_m : ℝ := dm_to_m stone_length_dm
def stone_width_m : ℝ := dm_to_m stone_width_dm

/-- Calculation of area of one stone -/
def stone_area_m2 : ℝ := stone_length_m * stone_width_m

/-- Total area to be paved -/
def total_area_to_be_paved : ℝ := number_of_stones * stone_area_m2

/-- Length of the hall -/
def length_of_hall : ℝ := total_area_to_be_paved / breadth_of_hall

/-- The proof statement: the length of the hall is 36 meters -/
theorem length_of_hall_is_36 : length_of_hall = 36 :=
by sorry

end length_of_hall_is_36_l531_531402


namespace vector_representation_l531_531684

-- Define vectors as elements in a vector space over reals
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (A B C D : V)

-- Given condition: \(\overrightarrow{BC} = 3\overrightarrow{CD}\)
def given_condition : Prop := (B - C) = 3 • (C - D)

-- The statement to prove
theorem vector_representation :
  given_condition A B C D →
  (D - A) = - (1/3 : ℝ) • (B - A) + (4/3 : ℝ) • (C - A) :=
by
  intro h
  sorry

end vector_representation_l531_531684


namespace sum_of_number_and_square_eq_132_l531_531011

theorem sum_of_number_and_square_eq_132 (x : ℝ) (h : x + x^2 = 132) : x = 11 ∨ x = -12 :=
by
  sorry

end sum_of_number_and_square_eq_132_l531_531011


namespace compare_neg_fractions_l531_531525

theorem compare_neg_fractions : (-3 / 5) < (-1 / 3) := 
by {
  sorry
}

end compare_neg_fractions_l531_531525


namespace m_range_l531_531921

noncomputable def otimes (a b : ℝ) : ℝ := 
if a > b then a else b

theorem m_range (m : ℝ) : (otimes (2 * m - 5) 3 = 3) ↔ (m ≤ 4) := by
  sorry

end m_range_l531_531921


namespace num_valid_subsets_l531_531755

def isRelativelyPrime (a b : ℕ) : Prop := Nat.gcd a b = 1

def pairwiseRelativelyPrime (T : Finset ℕ) : Prop :=
  T.nonempty ∧ ∀ a b, a ∈ T → b ∈ T → a ≠ b → isRelativelyPrime a b

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

def validSubsetsCount : ℕ :=
  (S.powerset.filter pairwiseRelativelyPrime).card

theorem num_valid_subsets : validSubsetsCount = 27 := 
by 
  sorry

end num_valid_subsets_l531_531755


namespace simplify_expression_l531_531833

theorem simplify_expression (c d : ℕ) (hc : c = 2) (hd : d = 250) : 
  (\sqrt[4]{2^5 \cdot 5^3} = c * \sqrt[4]{d} → c + d = 252) :=
by
  -- Declare the values of c and d from the given problem
  rw [hc, hd]
  -- Use the simplified values to prove the result
  sorry

end simplify_expression_l531_531833


namespace lowest_common_denominator_l531_531315

theorem lowest_common_denominator (a b c : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : c = 18) : Nat.lcm (Nat.lcm a b) c = 36 :=
by
  -- Introducing the given conditions
  rw [h1, h2, h3]
  -- Compute the LCM of the provided values
  sorry

end lowest_common_denominator_l531_531315


namespace area_of_fountain_base_l531_531413

-- Definitions of the given conditions
def plank_AB : ℝ := 24
def plank_DC : ℝ := 14
def point_D_fraction : ℝ := 1 / 3
def AD : ℝ := point_D_fraction * plank_AB
def DC : ℝ := plank_DC

-- Given the Pythagorean relationship in triangle ADC
def radius_squared : ℝ := AD^2 + DC^2

-- Final statement to prove the area of the circular base of the fountain
theorem area_of_fountain_base : 
  area (circle_radius := real.sqrt radius_squared) = 260 * real.pi :=
by
  sorry

end area_of_fountain_base_l531_531413


namespace solve_for_q_l531_531723

theorem solve_for_q (k l q : ℕ) (h1 : (2 : ℚ) / 3 = k / 45) (h2 : (2 : ℚ) / 3 = (k + l) / 75) (h3 : (2 : ℚ) / 3 = (q - l) / 105) : q = 90 :=
sorry

end solve_for_q_l531_531723


namespace base_sequence_count_l531_531339

theorem base_sequence_count :
  let A := 1
  let C := 2
  let G := 3
  (nat.choose 6 A) * (nat.choose 5 C) * (nat.choose 3 G) = 60 :=
by
  sorry

end base_sequence_count_l531_531339


namespace original_order_amount_l531_531050

-- Definitions based on the conditions
def service_charge := 0.04
def total_amount := 468

-- Definition of the original amount condition
def original_amount_condition (x : ℝ) := x + service_charge * x = total_amount

-- The theorem that we need to prove
theorem original_order_amount (x : ℝ) (h : original_amount_condition x) : x = 450 :=
sorry

end original_order_amount_l531_531050


namespace rod_mass_equilibrium_l531_531329

variable (g : ℝ) (m1 : ℝ) (l : ℝ) (S : ℝ)

-- Given conditions
axiom m1_value : m1 = 1
axiom l_value  : l = 0.5
axiom S_value  : S = 0.1

-- The goal is to find m2 such that the equilibrium condition holds
theorem rod_mass_equilibrium (m2 : ℝ) :
  (m1 * S = m2 * l) → m2 = 0.2 :=
by
  sorry

end rod_mass_equilibrium_l531_531329


namespace principal_amount_l531_531125

theorem principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) : 
  A = 1120 → r = 0.05 → t = 6 → P = 1120 / (1 + 0.05 * 6) :=
by
  intros h1 h2 h3
  sorry

end principal_amount_l531_531125


namespace magnitude_of_complex_number_l531_531538

theorem magnitude_of_complex_number : 
  ∀ (a b : ℤ), (a = 3 ∧ b = -10) → (|complex.mk a b| = real.sqrt (a^2 + b^2)) := 
by
  intros a b hb
  cases hb
  rw [hb_left, hb_right]
  exact sorry

end magnitude_of_complex_number_l531_531538


namespace number_of_rhombuses_l531_531669

theorem number_of_rhombuses (grid: Fin 5 × Fin 5 → bool) (h_eq : ∀ i j, grid (i, j) = true) : 
  ∃ n : ℕ, n = 30 := 
sorry

end number_of_rhombuses_l531_531669


namespace student_assignment_count_l531_531568

theorem student_assignment_count :
  let classes := ["A", "B", "C"]
  let students := ["A", "B", "C", "D"]
  (∃ (assignments : students → classes), 
    -- condition 1: Each class must have at least one student
    (∀ cls ∈ classes, ∃ stu ∈ assignments, assignments stu = cls) ∧ 
    -- condition 2: Student A cannot be assigned to class A
    assignments "A" ≠ "A"
  ) →
  -- Expected number of valid assignments
  card { assignments // ∀ cls ∈ classes, ∃ stu ∈ assignments, assignments stu = cls ∧ assignments "A" ≠ "A" } = 24 :=
sorry

end student_assignment_count_l531_531568


namespace bottles_per_case_l531_531876

/-- The number of bottles per case -/
def number_of_bottles_per_case (cases_april cases_may total_bottles : ℕ) : ℕ :=
  (cases_april + cases_may) / total_bottles

theorem bottles_per_case :
  ∀ (x : ℕ),
    (20 * x + 30 * x = 1000 ) → x = 20 := 
by
  intro x
  assume h : 20 * x + 30 * x = 1000
  sorry

end bottles_per_case_l531_531876


namespace probability_of_A_l531_531746

theorem probability_of_A (P : Set α → ℝ) (A B : Set α) :
  P B = 0.40 →
  P (A ∩ B) = 0.15 →
  P Aᶜ ∩ Bᶜ = 0.50 →
  P A = 0.25 :=
by
  intros h1 h2 h3
  sorry

end probability_of_A_l531_531746


namespace shortest_distance_to_line_l531_531940

noncomputable def f (x : ℝ) : ℝ := 1 / x

def line (x y : ℝ) : Prop := y = -x - 1

theorem shortest_distance_to_line : 
  ∃ p : ℝ × ℝ, (p.1 = -1 ∨ p.1 = 1) ∧ (f p.1 = p.2) ∧ 
  (∀ x y, (x, y) = p → line x y → (abs (-x - y - 1)) / (sqrt 2) = (sqrt 2) / 2) :=
sorry

end shortest_distance_to_line_l531_531940


namespace parabola_expression_correct_area_triangle_ABM_correct_l531_531170

-- Given conditions
def pointA : ℝ × ℝ := (-1, 0)
def pointB : ℝ × ℝ := (3, 0)
def pointC : ℝ × ℝ := (0, 3)

-- Analytical expression of the parabola as y = -x^2 + 2x + 3
def parabola_eqn (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Definition of the vertex M of the parabola (derived from calculations)
def vertexM : ℝ × ℝ := (1, 4)

-- Calculation of distance AB
def distance_AB : ℝ := 4

-- Calculation of area of triangle ABM
def triangle_area_ABM : ℝ := 8

theorem parabola_expression_correct :
  (∀ x y, (y = parabola_eqn x ↔ (parabola_eqn x = y))) ∧
  (parabola_eqn pointC.1 = pointC.2) :=
by
  sorry

theorem area_triangle_ABM_correct :
  (1 / 2 * distance_AB * vertexM.2 = 8) :=
by
  sorry

end parabola_expression_correct_area_triangle_ABM_correct_l531_531170


namespace math_problem_l531_531603

theorem math_problem
  (x y z : ℤ)
  (hz : z ≠ 0)
  (eq1 : 2 * x - 3 * y - z = 0)
  (eq2 : x + 3 * y - 14 * z = 0) :
  (x^2 - x * y) / (y^2 + 2 * z^2) = 10 / 11 := 
by 
  sorry

end math_problem_l531_531603


namespace strips_overlap_area_l531_531867

theorem strips_overlap_area :
  ∀ (length_left length_right area_only_left area_only_right : ℕ) (S : ℚ),
    length_left = 9 →
    length_right = 7 →
    area_only_left = 27 →
    area_only_right = 18 →
    (area_only_left + S) / (area_only_right + S) = 9 / 7 →
    S = 13.5 :=
by
  intros length_left length_right area_only_left area_only_right S
  intro h1 h2 h3 h4 h5
  sorry

end strips_overlap_area_l531_531867


namespace range_of_a_l531_531250

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (2^x - 2^(-x))
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * (2^x + 2^(-x))

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * f x + g (2 * x) ≥ 0) ↔ a ≥ -17 / 6 :=
by
  sorry

end range_of_a_l531_531250


namespace identify_heaviest_and_lightest_in_13_weighings_l531_531804

-- Definitions based on the conditions
def coins := Finset ℕ
def weighs_with_balance_scale (c1 c2: coins) : Prop := true  -- Placeholder for weighing functionality

/-- There are 10 coins, each with a distinct weight. -/
def ten_distinct_coins (coin_set : coins) : Prop :=
  coin_set.card = 10 ∧ (∀ c1 c2 ∈ coin_set, c1 ≠ c2 → weighs_with_balance_scale c1 c2)

-- Theorem statement
theorem identify_heaviest_and_lightest_in_13_weighings 
  (coin_set : coins)
  (hc: ten_distinct_coins coin_set):
  ∃ (heaviest lightest : coins), 
    weighs_with_balance_scale heaviest coin_set ∧ weighs_with_balance_scale coin_set lightest ∧ 
    -- Assuming weighs_with_balance_scale keeps track of number of weighings
    weights_used coin_set = 13 :=
sorry

end identify_heaviest_and_lightest_in_13_weighings_l531_531804


namespace cars_on_weekend_each_day_l531_531819

theorem cars_on_weekend_each_day (cars_mask_Tuesday : ℕ) (cars_Tuesday : cars_mask_Tuesday = 25)
    (cars_mask_Monday : ℕ) (cars_Monday : cars_mask_Monday = cars_mask_Tuesday - (20 * cars_mask_Tuesday / 100))
    (cars_mask_Wednesday : ℕ) (cars_Wednesday : cars_mask_Wednesday = cars_mask_Monday + 2)
    (cars_Thursday : ℕ) (cars_Thursday := 10)
    (cars_Friday : ℕ) (cars_Friday := 10)
    (total_weekly_cars : ℕ) (total_weekly_cars = 97)
    (total_weekend_cars := total_weekly_cars - (cars_mask_Monday + cars_mask_Tuesday + cars_mask_Wednesday + cars_Thursday + cars_Friday))
    (equal_weekend_cars : ℕ) (equal_weekend_cars := total_weekend_cars / 2) :
equal_weekend_cars = 5 :=
sorry

end cars_on_weekend_each_day_l531_531819


namespace exists_k_in_octahedron_l531_531665

noncomputable def rational_points_lemma (a b c : ℚ) (h_sum : a + b + c ∉ ℤ) : 
  ∃ k : ℕ, 
  (ka, kb, kc are not all integers) ∧ 
  1 < frac (ka) + frac (kb) + frac (kc) < 2 := 
sorry

theorem exists_k_in_octahedron (x0 y0 z0 : ℚ)
  (h_not_on_planes : ∀ n : ℤ, x0 + y0 + z0 ≠ n ∧ x0 - y0 + z0 ≠ n ∧ x0 + y0 - z0 ≠ n ∧ x0 - y0 - z0 ≠ n) :
  ∃ k : ℕ, 
  ∃ T : set (ℝ × ℝ × ℝ) ∧ T ⊆ (lambda (x, y, z), (x0, y0, z0)), 
  x0 + y0 + z0 not ∈ T := 
begin
  -- Application of rational_points_lemma
  obtain ⟨k, h_not_all_int, h_in_sum⟩ := rational_points_lemma 
    (y0 + z0 - x0) 
    (x0 - y0 + z0) 
    (x0 + y0 - z0) 
    _,
  
  -- Use of k to build required octahedron
  have h_k_not_all_int := h_not_all_int,
  have h_frac_sum := h_in_sum,

  -- Constructing the octahedron from k
  use k,
  sorry
end

end exists_k_in_octahedron_l531_531665


namespace staffing_battle_station_l531_531898

-- Define the conditions as constants
constant total_applicants : ℕ := 30
constant unsuitable_ratio : ℚ := 1 / 3
constant suitable_applicants : ℕ := total_applicants * (1 - unsuitable_ratio.toNat)
constant radio_specialist_experience : ℕ := 5
constant roles : ℕ := 5

-- The problem is to prove the number of ways to staff the battle station is 292,320
theorem staffing_battle_station :
  let assistant_engineer_candidates := suitable_applicants - 1,
      weapons_maintenance_candidates := assistant_engineer_candidates - 1,
      field_technician_candidates := weapons_maintenance_candidates - 1,
      communications_officer_candidates := field_technician_candidates - 1,
      radio_specialist_ways := radio_specialist_experience in
  radio_specialist_ways *
  assistant_engineer_candidates *
  weapons_maintenance_candidates *
  field_technician_candidates *
  communications_officer_candidates = 292320 :=
by
  let assistant_engineer_candidates := suitable_applicants - 1
  let weapons_maintenance_candidates := assistant_engineer_candidates - 1
  let field_technician_candidates := weapons_maintenance_candidates - 1
  let communications_officer_candidates := field_technician_candidates - 1
  let radio_specialist_ways := radio_specialist_experience
  have calculation :
    radio_specialist_ways *
    assistant_engineer_candidates *
    weapons_maintenance_candidates *
    field_technician_candidates *
    communications_officer_candidates = 292320
  := sorry,
  exact calculation

end staffing_battle_station_l531_531898


namespace length_segment_AB_max_length_major_axis_l531_531148

-- Definitions and conditions as given in the problem
def line_eq (x : ℝ) : ℝ := -x + 1
def eclipse_eq (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity : ℝ := Real.sqrt 2 / 2
def focal_distance (c : ℝ) : ℝ := 2 * c
def perpendicular_vectors (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0
def eccentricity_interval : Set ℝ := {e | 1/2 ≤ e ∧ e ≤ Real.sqrt 2 / 2}

-- Theorem statements
theorem length_segment_AB (a b c x1 y1 x2 y2 : ℝ)
  (h1 : a = Real.sqrt 2)
  (h2 : c = 1)
  (h3 : b = Real.sqrt (a^2 - c^2))
  (h4 : x1 = 4/3)
  (h5 : y1 = -1/3)
  (h6 : x2 = 0)
  (h7 : y2 = 1)
  : Real.dist (x1, y1) (x2, y2) = 4/3 * Real.sqrt 2 := 
sorry

theorem max_length_major_axis (a b e : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : focal_distance 1 = 2)
  (h4 : ∀ (x1 y1 x2 y2 : ℝ), perpendicular_vectors x1 y1 x2 y2)
  (h5 : e ∈ eccentricity_interval)
  (h6 : ∀ (a e : ℝ), a^2 + (a^2 - e^2 * a^2) > 1)
  : 2 * a ≤ Real.sqrt 6 := 
sorry

end length_segment_AB_max_length_major_axis_l531_531148


namespace identify_heaviest_and_lightest_13_weighings_l531_531817

theorem identify_heaviest_and_lightest_13_weighings (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ f : (Fin 13 → ((Fin 10) × (Fin 10) × ℝ)), true :=
by
  sorry

end identify_heaviest_and_lightest_13_weighings_l531_531817


namespace sin_cos_identity_l531_531196

theorem sin_cos_identity (a b : ℝ) (θ : ℝ)
  (h : (\sin θ)^6 / a + (\cos θ)^6 / b = 1 / (a + b)) :
  (\sin θ)^12 / (a^5) + (\cos θ)^12 / (b^5) = 1 / (a + b)^5 :=
by
  sorry

end sin_cos_identity_l531_531196


namespace no_simultaneous_inequalities_l531_531693

theorem no_simultaneous_inequalities (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
by
  sorry

end no_simultaneous_inequalities_l531_531693


namespace find_number_l531_531012

theorem find_number (x : ℝ) (h : 0.65 * x = 0.8 * x - 21) : x = 140 := by
  sorry

end find_number_l531_531012


namespace num_lines_passing_through_p_forming_30_degree_angle_l531_531409

theorem num_lines_passing_through_p_forming_30_degree_angle (P : Point) (a : Line) (h : P ∉ a) : Infinite (SetOf L, where L passes_through P and angle_between L a = 30°) :=
sorry

end num_lines_passing_through_p_forming_30_degree_angle_l531_531409


namespace sasha_remainder_l531_531757

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l531_531757


namespace sufficient_but_not_necessary_condition_l531_531950

theorem sufficient_but_not_necessary_condition (a b : ℝ) : b < a ∧ a < 0 → (1 / b > 1 / a) ∧ ¬ (1 / b > 1 / a → b < a ∧ a < 0) :=
by
  intro h
  cases h with hba h0
  split
  -- first part
  { linarith [inv_lt_inv_of_neg hba h0] }
  -- counterexample for the converse
  {
    intro hab
    have contra : b > 0 ∧ a < 0 := sorry /-Provide a suitable counterexample like the one in the solution -/
    exact contra
  }

end sufficient_but_not_necessary_condition_l531_531950


namespace cube_passage_possible_l531_531672

theorem cube_passage_possible (a : ℝ) : 
  let b := (a * (real.sqrt 2)) / (real.sqrt 3) in
  let s := (2 * a * (real.sqrt 2)) / 3 in 
  (s ≥ a) ∨ (2 * (real.sqrt 2) > 3) → True := 
by 
  let b := (a * (real.sqrt 2)) / (real.sqrt 3)
  let s := (2 * a * (real.sqrt 2)) / 3
  intro h
  exact trivial

end cube_passage_possible_l531_531672


namespace sasha_remainder_is_20_l531_531788

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l531_531788


namespace who_received_card_5_l531_531210

def players := ["Alice", "Bob", "Charlie", "Dana", "Eve"]
def scores := [("Alice", 15), ("Bob", 5), ("Charlie", 19), ("Dana", 12), ("Eve", 19)]

theorem who_received_card_5 (cards : list (string × ℕ)) :
  (∀ p1 p2, p1 ≠ p2 → (p1, 5) ∈ cards → (p2, 5) ∉ cards) →
  ("Alice", 15) ∈ scores →
  ("Bob", 5) ∈ scores →
  ("Charlie", 19) ∈ scores →
  ("Dana", 12) ∈ scores →
  ("Eve", 19) ∈ scores →
  ∃ (p : string), (p, 5) ∈ cards ∧ p = "Alice" :=
by
  intro non_dup_cards Alice_score Bob_score Charlie_score Dana_score Eve_score
  sorry

end who_received_card_5_l531_531210


namespace open_feasible_region_has_no_maximum_value_l531_531604

noncomputable def z (x y : ℝ) : ℝ := 3 * x + 5 * y

theorem open_feasible_region_has_no_maximum_value
  (x y : ℝ)
  (h1 : 6 * x + 3 * y < 15)
  (h2 : y ≤ x + 1)
  (h3 : x - 5 * y ≤ 3)
  : ∃ inf, ∀ M, ¬ (M ∈ upper_bound (set_of (λ z, ∃ (x y : ℝ), z = 3 * x + 5 * y ∧ 6 * x + 3 * y < 15 ∧ y ≤ x + 1 ∧ x - 5 * y ≤ 3))) :=
sorry

end open_feasible_region_has_no_maximum_value_l531_531604


namespace number_of_solution_pairs_l531_531105

theorem number_of_solution_pairs (x y : ℕ) (h : 4 * x + 7 * y = 1003) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 4 * x + 7 * y = 1003) ∧
  ∃! n : ℕ, n = 36 := 
begin
  sorry
end

end number_of_solution_pairs_l531_531105


namespace trigonometric_inequalities_l531_531563

theorem trigonometric_inequalities (φ : ℝ) (h : 0 < φ ∧ φ < π / 2) :
  sin (cos φ) < cos φ ∧ cos φ < cos (sin φ) := 
sorry

end trigonometric_inequalities_l531_531563


namespace problem_distance_between_Howard_and_Rineville_l531_531350

noncomputable def distance_between_cities := 
  ∀ (speed_Howard speed_Rineville : ℕ) (time_to_meet : ℚ),
  speed_Howard = 30 → 
  speed_Rineville = 45 → 
  time_to_meet = 4 / 3 →
  speed_Howard * time_to_meet + speed_Rineville * time_to_meet = 100

theorem problem_distance_between_Howard_and_Rineville :
  distance_between_cities 30 45 (4 / 3) := 
by
  sorry

end problem_distance_between_Howard_and_Rineville_l531_531350


namespace function_properties_l531_531953

noncomputable def f (x : ℝ) : ℝ := (1 + Real.sec x) * (Real.cot (x / 2)) ^ 2

theorem function_properties : (∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f x ≥ 8) ∧ ¬(∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), ∃ M, f x ≤ M) :=
by sorry

end function_properties_l531_531953


namespace arc_length_60_degrees_radius_1_l531_531650

def radius : ℝ := 1
def angle_in_degrees : ℝ := 60
def angle_in_radians : ℝ := angle_in_degrees * (Real.pi / 180)
def arc_length (r : ℝ) (θ : ℝ) : ℝ := r * θ

theorem arc_length_60_degrees_radius_1 : arc_length radius angle_in_radians = Real.pi / 3 :=
by
  have angle_in_rads_simp : angle_in_radians = Real.pi / 3 := by
    unfold angle_in_radians
    simp [angle_in_degrees, Real.pi]
  rw [angle_in_rads_simp]
  simp [arc_length, radius, angle_in_radians]
  sorry

end arc_length_60_degrees_radius_1_l531_531650


namespace units_digit_27_mul_46_l531_531555

theorem units_digit_27_mul_46 : (27 * 46) % 10 = 2 :=
by 
  -- Definition of units digit
  have def_units_digit :=  (n : ℕ) => n % 10

  -- Step 1: units digit of 27 is 7
  have units_digit_27 : 27 % 10 = 7 := by norm_num
  
  -- Step 2: units digit of 46 is 6
  have units_digit_46 : 46 % 10 = 6 := by norm_num

  -- Step 3: multiple the units digits
  have step3 : 7 * 6 = 42 := by norm_num

  -- Step 4: Find the units digit of 42
  have units_digit_42 : 42 % 10 = 2 := by norm_num

  exact units_digit_42

end units_digit_27_mul_46_l531_531555


namespace distance_between_after_given_time_l531_531195

-- To avoid concerns regarding speed conversion during the proof steps, we can include it directly in the conditions.
variables {total_distance : ℝ} {hyosung_speed_mpm : ℝ} {mimi_speed_kph : ℝ} {minutes_passed : ℕ}

def mimi_speed_mpm := mimi_speed_kph / 60

theorem distance_between_after_given_time
  (h1 : total_distance = 2.5) 
  (h2 : hyosung_speed_mpm = 0.08)
  (h3 : mimi_speed_kph = 2.4)
  (h4 : minutes_passed = 15) : 
  total_distance - ((hyosung_speed_mpm + mimi_speed_mpm) * minutes_passed) = 0.7 :=
by 
  have hs := h2,
  have ms_cvt := h3 / 60,
  have combined_speed := hs + ms_cvt,
  have distance_covered := combined_speed * h4,
  have remaining_distance := h1 - distance_covered,
  show remaining_distance = 0.7,
  calc remaining_distance = h1 - distance_covered : by sorry
                 ... = 0.7 : by sorry

end distance_between_after_given_time_l531_531195


namespace least_number_divisible_by_6_has_remainder_4_is_40_l531_531356

-- Define the least number N which leaves a remainder of 4 when divided by 6
theorem least_number_divisible_by_6_has_remainder_4_is_40 :
  ∃ (N : ℕ), (∀ (k : ℕ), N = 6 * k + 4) ∧ N = 40 := by
  sorry

end least_number_divisible_by_6_has_remainder_4_is_40_l531_531356


namespace sasha_remainder_20_l531_531780

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l531_531780


namespace find_rate_percent_l531_531017

-- Define the conditions
def principal : ℝ := 1200
def time : ℝ := 4
def simple_interest : ℝ := 400

-- Define the rate that we need to prove
def rate : ℝ := 8.3333  -- approximately

-- Formalize the proof problem in Lean 4
theorem find_rate_percent
  (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ)
  (hP : P = principal) (hT : T = time) (hSI : SI = simple_interest) :
  SI = (P * R * T) / 100 → R = rate :=
by
  intros h
  sorry

end find_rate_percent_l531_531017


namespace chord_length_l531_531839

theorem chord_length (R a b : ℝ) (hR : a + b = R) (hab : a * b = 10) 
    (h_nonneg : 0 ≤ R ∧ 0 ≤ a ∧ 0 ≤ b) : ∃ L : ℝ, L = 2 * Real.sqrt 10 :=
by
  sorry

end chord_length_l531_531839


namespace dice_diff_by_three_probability_l531_531432

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l531_531432


namespace range_of_a_l531_531643

variable {a : ℝ}

def condition1 : Prop := ∀ x : ℝ, 0 < x → x + 4 / x ≥ a
def condition2 : Prop := ∃ x : ℝ, x^2 + 2 * x + a = 0

theorem range_of_a (h1 : condition1) (h2 : condition2) : a ≤ 1 := 
sorry

end range_of_a_l531_531643


namespace sasha_remainder_l531_531781

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l531_531781


namespace number_of_arrangements_is_1200_l531_531288

def number_of_arrangements (A B C : Type) := 
  λ (duty_schedule : Fin 7 → Option Student), 
  (∀ i, A (duty_schedule 0) = false) ∧
  (∃ j, (duty_schedule j) = B ∧ (duty_schedule (j + 1 mod 7) = C ∨ duty_schedule (j - 1 % 7) = C)) ∧
  (∃ k ≠ 0, duty_schedule k = A)


theorem number_of_arrangements_is_1200 {A B C : Type} : 
  ∃ (duty_schedule : Fin 7 → Option Student), 
    number_of_arrangements A B C duty_schedule = 1200 :=
by
  sorry

end number_of_arrangements_is_1200_l531_531288


namespace amount_per_delivered_bowl_l531_531347

-- Definitions of conditions
def total_bowls : ℕ := 638
def lost_bowls : ℕ := 12
def broken_bowls : ℕ := 15
def delivered_safely := total_bowls - lost_bowls - broken_bowls
def fee : ℝ := 100
def total_payment : ℝ := 1825

-- Proposition to prove
theorem amount_per_delivered_bowl : 
  let x : ℝ := (total_payment - fee) / delivered_safely in
  x = 2.82 :=
by
  sorry

end amount_per_delivered_bowl_l531_531347


namespace construct_triangle_with_conditions_l531_531528

noncomputable def circle_with_radius (R : ℝ) : Type :=
{ S : Type // ∃ O : Type, ∀ P ∈ S, dist P O = R }

def exists_triangle_ABC (R : ℝ) (bisector : ℝ → Prop) : Prop :=
∃ (A B C : ℝ),
    (∀ (S : circle_with_radius R),
      ∃ O : Type, ∃ (L : Type), ∃ (K : Type),
      let circumcircle_center := O in
      let A := A in
      let L := L in
      let K := K in
      let points_on_S := λ P, dist P O = R ∧ collinear O L P ∧ angle A L = 90 in
      (dist S A = 0) ∧
      (AK = bisector angle A) ∧
      let B := B in
      let C := C in
      let perpendicular := λ l, ∀ P, collinear O L P → ∃ Q1, dist Q1 O = dist Q1 L ∧ ∃ Q2, dist Q2 O = dist Q2 L in
      perpendicular K ∧
      ((angle B C) = 90))

-- statement of the problem
theorem construct_triangle_with_conditions (R : ℝ) (bisector : ℝ → Prop) :
    exists_triangle_ABC R bisector := sorry

end construct_triangle_with_conditions_l531_531528


namespace length_of_BD_l531_531274

theorem length_of_BD (BC AC AD : ℝ) (hBC : BC = 3) (hAC : AC = 4) (hAD : AD = 5)
  (hABC_right : BC^2 + AC^2 = (real.sqrt (AC^2 + BC^2))^2)
  (hABD_right : AD^2 + BD^2 = AB^2) :
  ∃ BD : ℝ, BD = 5 :=
by {
  have hAB : real.sqrt (AC^2 + BC^2) = 5 := by sorry,
  have hABD_iso : AD = (real.sqrt (AC^2 + BC^2)) := by sorry,
  use 5,
  exact sorry,
}

end length_of_BD_l531_531274


namespace calculate_100a_plus_b_l531_531692

noncomputable def cube_expected_perimeter (t : ℝ) : ℝ :=
  let P := -- Definition of random point in [0,1]^3 (not explicitly needed for the final proof structure)
  let plane := λ (P : ℝ × ℝ × ℝ), -- Plane parallel to x + y + z = 0 through P (not explicitly needed for the final proof structure)
  t^2

theorem calculate_100a_plus_b (t : ℝ) (a b : ℕ) (h1 : t = (11 * Real.sqrt 2) / 4)
  (h2 : a = 121) (h3 : b = 8) (h4 : Nat.gcd a b = 1) :
  100 * a + b = 12108 := 
  by
    sorry

end calculate_100a_plus_b_l531_531692


namespace local_maximum_at_negative_one_l531_531165

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * a * x + 2

theorem local_maximum_at_negative_one
  (a : ℝ)
  (h_min : ∀ f' : ℝ → ℝ, (f' = λ x, 3 * x^2 - 3 * a) → f' 1 = 0) :
  f a (-1) = 4 :=
by
  -- Definitions and hypotheses are in place; proof is omitted.
  sorry

end local_maximum_at_negative_one_l531_531165


namespace tom_bought_pieces_l531_531344

/-
  Tom already had 2 pieces of candy.
  His friend gave him 7 more pieces of candy.
  Tom now has 19 pieces of candy.
  How many pieces did Tom buy?
-/

theorem tom_bought_pieces :
  let initial_pieces : Nat := 2
  let given_pieces : Nat := 7
  let total_pieces : Nat := 19
  in initial_pieces + given_pieces + a = total_pieces → a = 10 :=
by
  sorry

end tom_bought_pieces_l531_531344


namespace second_number_exists_l531_531338

theorem second_number_exists (S : ℕ) : 
  (∃ (l : ℤ), S = 4 * l + 4) ∧ ∃ (n : ℕ), S = n :=
begin
  sorry
end

end second_number_exists_l531_531338


namespace symmetrical_point_l531_531728

structure Point :=
  (x : ℝ)
  (y : ℝ)

def reflect_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetrical_point (M : Point) (hM : M = {x := 3, y := -4}) : reflect_x_axis M = {x := 3, y := 4} :=
  by
  sorry

end symmetrical_point_l531_531728


namespace dots_per_ladybug_l531_531067

-- Define the conditions as variables
variables (m t : ℕ) (total_dots : ℕ) (d : ℕ)

-- Setting actual values for the variables based on the given conditions
def m_val : ℕ := 8
def t_val : ℕ := 5
def total_dots_val : ℕ := 78

-- Defining the total number of ladybugs and the average dots per ladybug
def total_ladybugs : ℕ := m_val + t_val

-- To prove: Each ladybug has 6 dots on average
theorem dots_per_ladybug : total_dots_val / total_ladybugs = 6 :=
by
  have m := m_val
  have t := t_val
  have total_dots := total_dots_val
  have d := 6
  sorry

end dots_per_ladybug_l531_531067


namespace find_f2_f2_prime_l531_531737

theorem find_f2_f2_prime (f : ℝ → ℝ) (h_tangent : ∀ x, 2 * x + (f x) - 3 = 0) :
  f(2) + f'(2) = -3 := 
by sorry

end find_f2_f2_prime_l531_531737


namespace ratio_of_playground_to_landscape_l531_531320

-- Conditions
def length_of_landscape : ℝ := 120
def breadth_of_landscape : ℝ := length_of_landscape / 4
def total_landscape_area : ℝ := length_of_landscape * breadth_of_landscape
def playground_area : ℝ := 1200

-- Theorem statement proving the desired ratio
theorem ratio_of_playground_to_landscape :
  playground_area / total_landscape_area = 1 / 3 := by
  -- proof omitted
  sorry

end ratio_of_playground_to_landscape_l531_531320


namespace simplify_fraction_l531_531297

theorem simplify_fraction :
  ∀ (x y : ℝ), x = 1 → y = 1 → (x^3 + y^3) / (x + y) = 1 :=
by
  intros x y hx hy
  rw [hx, hy]
  simp
  sorry

end simplify_fraction_l531_531297


namespace determine_digit_square_l531_531532

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_palindrome (n : ℕ) : Prop :=
  let d1 := (n / 100000) % 10
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  d1 = d6 ∧ d2 = d5 ∧ d3 = d4

def is_multiple_of_6 (n : ℕ) : Prop := is_even (n % 10) ∧ is_divisible_by_3 (List.sum (Nat.digits 10 n))

theorem determine_digit_square :
  ∃ (square : ℕ),
  (is_palindrome (53700000 + square * 10 + 735) ∧ is_multiple_of_6 (53700000 + square * 10 + 735)) ∧ square = 6 := by
  sorry

end determine_digit_square_l531_531532


namespace correct_spatial_relationships_l531_531065

theorem correct_spatial_relationships :
  (∀ (a b : Type) (α β γ : a)
     (H1 : a ∥ α) (H2 : b ∥ α) (H3 : a ∥ β) (H4 : b ∥ β) (H5 : a ⊥ b),
    α ∥ β) :=
begin
  sorry
end

end correct_spatial_relationships_l531_531065


namespace distance_between_points_on_hyperbola_asymptote_l531_531404

theorem distance_between_points_on_hyperbola_asymptote :
  let hyperbola := ∀ x y : ℝ, x^2 - y^2 / 3 = 1 in
  let asymptote1 := ∀ x : ℝ, y = sqrt 3 * x in
  let asymptote2 := ∀ x : ℝ, y = -sqrt 3 * x in
  let right_focus := (2, 0) in
  let line_through_focus := ∀ y : ℝ, x = 2 in
  let A := (2, 2 * sqrt 3) in
  let B := (2, -2 * sqrt 3) in
  ∀ A B : (ℝ × ℝ), abs (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) = 4 * sqrt 3 :=
sorry

end distance_between_points_on_hyperbola_asymptote_l531_531404


namespace condition_purely_imaginary_l531_531384

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem condition_purely_imaginary (m : ℝ) :
  (is_purely_imaginary ((1 + m * Complex.i) * (1 + Complex.i))) ↔ (m = 1) := by
  sorry

end condition_purely_imaginary_l531_531384


namespace serviced_car_speed_l531_531393

theorem serviced_car_speed :
  let unserviced_speed := 55
  let time_unserviced := 6
  let time_serviced := 3
  let V_s := (unserviced_speed * time_unserviced) / time_serviced
  V_s = 110 := by
  let unserviced_speed := 55
  let time_unserviced := 6
  let time_serviced := 3
  let V_s := (unserviced_speed * time_unserviced) / time_serviced
  show V_s = 110
  calc
    V_s = (unserviced_speed * time_unserviced) / time_serviced : by rfl
       ... = (55 * 6) / 3                           : by rfl
       ... = 330 / 3                                : by rfl
       ... = 110                                    : by arithmetic

end serviced_car_speed_l531_531393


namespace edward_initial_money_l531_531536

theorem edward_initial_money (cars qty : Nat) (car_cost race_track_cost left_money initial_money : ℝ) 
    (h1 : cars = 4) 
    (h2 : car_cost = 0.95) 
    (h3 : race_track_cost = 6.00)
    (h4 : left_money = 8.00)
    (h5 : initial_money = (cars * car_cost) + race_track_cost + left_money) :
  initial_money = 17.80 := sorry

end edward_initial_money_l531_531536


namespace marble_carving_percentage_l531_531416

theorem marble_carving_percentage :
  ∃ x : ℝ, 
  let orig_weight : ℝ := 180,
      first_week_remain : ℝ := (72/100) * orig_weight,
      final_statue_weight : ℝ := 85.0176 in
  129.6 * ((100 - x) / 100) * 0.8 = final_statue_weight ∧ 
  x = 17.99 :=
sorry

end marble_carving_percentage_l531_531416


namespace polynomial_not_factored_l531_531255

theorem polynomial_not_factored {n : ℕ} (hn : n > 1) :
    ¬ ∃ g h : ℤ[X], (degree g ≥ 1) ∧ (degree h ≥ 1) ∧ f = g * h :=
by
  let f := (λ x : ℤ[X], X^n + 5 * X^(n-1) + 3)
  sorry

end polynomial_not_factored_l531_531255


namespace contrapositive_sin_l531_531834

theorem contrapositive_sin (x y : ℝ) : (¬ (sin x = sin y) → ¬ (x = y)) :=
by {
  sorry
}

end contrapositive_sin_l531_531834


namespace initial_short_trees_in_the_park_l531_531337

def InitialShortTrees (FinalShortTrees ShortTreesToBePlanted : Nat) : Nat := 
  FinalShortTrees - ShortTreesToBePlanted

theorem initial_short_trees_in_the_park : InitialShortTrees 12 9 = 3 := by
  simp [InitialShortTrees]
  sorry

end initial_short_trees_in_the_park_l531_531337


namespace sasha_remainder_l531_531760

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l531_531760


namespace eggs_left_after_capital_recovered_l531_531711

-- Conditions as definitions
def eggs_in_crate := 30
def crate_cost_dollars := 5
def price_per_egg_cents := 20

-- The amount of cents in a dollar
def cents_per_dollar := 100

-- Total cost in cents
def crate_cost_cents := crate_cost_dollars * cents_per_dollar

-- The number of eggs needed to recover the capital
def eggs_to_recover_capital := crate_cost_cents / price_per_egg_cents

-- The number of eggs left
def eggs_left := eggs_in_crate - eggs_to_recover_capital

-- The theorem stating the problem
theorem eggs_left_after_capital_recovered : eggs_left = 5 :=
by
  sorry

end eggs_left_after_capital_recovered_l531_531711


namespace shoes_produced_min_pairs_for_profit_l531_531989

-- given conditions
def production_cost (n : ℕ) : ℕ := 4000 + 50 * n

-- Question (1)
theorem shoes_produced (C : ℕ) (h : C = 36000) : ∃ n : ℕ, production_cost n = C :=
by sorry

-- given conditions for part (2)
def selling_price (price_per_pair : ℕ) (n : ℕ) : ℕ := price_per_pair * n
def profit (price_per_pair : ℕ) (n : ℕ) : ℕ := selling_price price_per_pair n - production_cost n

-- Question (2)
theorem min_pairs_for_profit (price_per_pair profit_goal : ℕ) (h : price_per_pair = 90) (h1 : profit_goal = 8500) :
  ∃ n : ℕ, profit price_per_pair n ≥ profit_goal :=
by sorry

end shoes_produced_min_pairs_for_profit_l531_531989


namespace simplify_tan_cot_expression_l531_531292

theorem simplify_tan_cot_expression
  (h1 : Real.tan (Real.pi / 4) = 1)
  (h2 : Real.cot (Real.pi / 4) = 1) :
  (Real.tan (Real.pi / 4))^3 + (Real.cot (Real.pi / 4))^3 = 1 := by
  sorry

end simplify_tan_cot_expression_l531_531292


namespace intersection_point_lines_distance_point_to_line_l531_531069

-- Problem 1
theorem intersection_point_lines :
  ∃ (x y : ℝ), (x - y + 2 = 0) ∧ (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 1) :=
sorry

-- Problem 2
theorem distance_point_to_line :
  ∀ (x y : ℝ), (x = 1) ∧ (y = -2) → ∃ d : ℝ, d = 3 ∧ (d = abs (3 * x + 4 * y - 10) / (Real.sqrt (3^2 + 4^2))) :=
sorry

end intersection_point_lines_distance_point_to_line_l531_531069


namespace value_of_M_l531_531752

theorem value_of_M :
  let row_seq := [25, 25 + (8 - 25) / 3, 25 + 2 * (8 - 25) / 3, 8, 8 + (8 - 25) / 3, 8 + 2 * (8 - 25) / 3, -9]
  let col_seq1 := [25, 25 - 4, 25 - 8]
  let col_seq2 := [16, 20, 20 + 4]
  let col_seq3 := [-9, -9 - 11/4, -9 - 2 * 11/4, -20]
  let M := -9 - (-11/4)
  M = -6.25 :=
by
  sorry

end value_of_M_l531_531752


namespace minimum_value_is_1297_l531_531142

noncomputable def find_minimum_value (a b c n : ℕ) : ℕ :=
  if (a + b ≠ b + c) ∧ (b + c ≠ c + a) ∧ (a + b ≠ c + a) ∧
     ((a + b = n^2 ∧ b + c = (n + 1)^2 ∧ c + a = (n + 2)^2) ∨
      (a + b = (n + 1)^2 ∧ b + c = (n + 2)^2 ∧ c + a = n^2) ∨
      (a + b = (n + 2)^2 ∧ b + c = n^2 ∧ c + a = (n + 1)^2)) then
    a^2 + b^2 + c^2
  else
    0

theorem minimum_value_is_1297 (a b c n : ℕ) :
  a ≠ b → b ≠ c → c ≠ a → (∃ a b c n, (a + b = n^2 ∧ b + c = (n + 1)^2 ∧ c + a = (n + 2)^2) ∨
                                  (a + b = (n + 1)^2 ∧ b + c = (n + 2)^2 ∧ c + a = n^2) ∨
                                  (a + b = (n + 2)^2 ∧ b + c = n^2 ∧ c + a = (n + 1)^2)) →
  (∃ a b c, a^2 + b^2 + c^2 = 1297) :=
by sorry

end minimum_value_is_1297_l531_531142


namespace regular_star_intersections_l531_531698

theorem regular_star_intersections (n k : ℕ) (h_coprime : Nat.gcd k n = 1) (h_n_ge_5 : n ≥ 5) (h_k_lt_half_n : k < n / 2) : 
  (n = 2018) → (k = 25) → 
  n * (k - 1) = 48432 := 
by 
  intros h_n h_k 
  rw [h_n, h_k] 
  norm_num 
  sorry

end regular_star_intersections_l531_531698


namespace problem_2018_CCA_bonanza_LR_2_1_l531_531242

/--
Let \( S \) be the set of the first 2018 positive integers,
and let \( T \) be the set of all distinct numbers of the form \( ab \),
where \( a \) and \( b \) are distinct members of \( S \).
Prove that the 2018th smallest member of \( T \) is \( 6 \).
-/
theorem problem_2018_CCA_bonanza_LR_2_1 :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 2018}
  let T := {ab | ∃ a b : ℕ, a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ ab = a * b}
  (set.sort (· < ·) T).nth 2017 = some 6 :=
by
  sorry

end problem_2018_CCA_bonanza_LR_2_1_l531_531242


namespace product_of_roots_of_t_squared_equals_49_l531_531938

theorem product_of_roots_of_t_squared_equals_49 : 
  ∃ t : ℝ, (t^2 = 49) ∧ (t = 7 ∨ t = -7) ∧ (t * (7 + -7)) = -49 := 
by
  sorry

end product_of_roots_of_t_squared_equals_49_l531_531938


namespace ellipse_eccentricity_l531_531597

theorem ellipse_eccentricity (a b m c : Real) (h₁ : a > b) (h₂ : b > 0)
  (P F₁ F₂ : Real × Real) (hp : P ∈ {(x, y) | x^2 / a^2 + y^2 / b^2 = 1})
  (h₃ : dist P F₁ = 2 * dist P F₂) (h₄ : dist P F₁ ^ 2 + dist P F₂ ^ 2 = dist F₁ F₂ ^ 2) :
  c = (sqrt 5 / 2) * m → e = c / a → e = (sqrt 5) / 3 :=
by
  sorry

end ellipse_eccentricity_l531_531597


namespace b_2020_eq_4037_p_plus_q_eq_4038_l531_531053

noncomputable def b : ℕ → ℚ
| 1     := 2
| 2     := 7/2
| (n+3) := (b (n + 1) * b (n + 2)) / (2 * b (n + 1) - b (n + 2))

theorem b_2020_eq_4037 : b 2020 = 4037 := by sorry

theorem p_plus_q_eq_4038 (p q : ℕ) (h1 : p = 4037) (h2 : q = 1) : p + q = 4038 :=
by sorry

end b_2020_eq_4037_p_plus_q_eq_4038_l531_531053


namespace minimum_average_cost_l531_531395

noncomputable def average_cost (x : ℝ) : ℝ :=
  let y := (x^2) / 10 - 30 * x + 4000
  y / x

theorem minimum_average_cost : 
  ∃ (x : ℝ), 150 ≤ x ∧ x ≤ 250 ∧ (∀ (x' : ℝ), 150 ≤ x' ∧ x' ≤ 250 → average_cost x ≤ average_cost x') ∧ average_cost x = 10 := 
by
  sorry

end minimum_average_cost_l531_531395


namespace rice_grains_12th_minus_first_10_l531_531408

theorem rice_grains_12th_minus_first_10 :
    let grains_on_square (k : ℕ) : ℕ := 2 ^ k
    let sum_first_10_squares := (List.range 10).map (grains_on_square ∘ Nat.succ).sum
    grains_on_square 12 - sum_first_10_squares = 2050 :=
by
  sorry

end rice_grains_12th_minus_first_10_l531_531408


namespace not_unique_equilateral_by_one_angle_and_opposite_side_l531_531835

-- Definitions related to triangles
structure Triangle :=
  (a b c : ℝ) -- sides
  (alpha beta gamma : ℝ) -- angles

-- Definition of triangle types
def isIsosceles (t : Triangle) : Prop :=
  (t.a = t.b ∨ t.b = t.c ∨ t.a = t.c)

def isRight (t : Triangle) : Prop :=
  (t.alpha = 90 ∨ t.beta = 90 ∨ t.gamma = 90)

def isEquilateral (t : Triangle) : Prop :=
  (t.a = t.b ∧ t.b = t.c ∧ t.alpha = 60 ∧ t.beta = 60 ∧ t.gamma = 60)

def isScalene (t : Triangle) : Prop :=
  (t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c)

-- Proof that having one angle and the side opposite it does not determine an equilateral triangle.
theorem not_unique_equilateral_by_one_angle_and_opposite_side :
  ¬ ∀ (t1 t2 : Triangle), (isEquilateral t1 ∧ isEquilateral t2 →
    t1.alpha = t2.alpha ∧ t1.a = t2.a →
    t1 = t2) := sorry

end not_unique_equilateral_by_one_angle_and_opposite_side_l531_531835


namespace sum_of_digits_large_sum_l531_531092

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

noncomputable def large_sum : ℕ := 9 * (List.range 500).sum (λ k, k + 1)

theorem sum_of_digits_large_sum :
  sum_of_digits (large_sum - 500) = 1126750 :=
by {
  sorry,
}

end sum_of_digits_large_sum_l531_531092


namespace sum_of_integer_values_of_a_satisfying_conditions_l531_531644

theorem sum_of_integer_values_of_a_satisfying_conditions :
  (∑ a in { a | ∀ x y : ℤ, (x + 3 ≤ 8) ∧ (2 * x ≥ a + 2) ∧
                    [ (y = (a - 1) / 2) ∧ (y ≥ 0) ] }, 
       a) = 4 :=
by
  sorry

end sum_of_integer_values_of_a_satisfying_conditions_l531_531644


namespace difference_between_mean_and_median_l531_531273

def percentage_students (p70 p80 p90 p100 : ℝ) : Prop :=
  p70 + p80 + p90 + p100 = 1

def median_score (p70 p80 p90 p100 : ℝ) (s70 s80 s90 s100 : ℕ) : ℕ :=
  if (p70 + p80) < 0.5 then s100
  else if (p70 < 0.5) then s90 
  else if (p70 > 0.5) then s70
  else s80

def mean_score (p70 p80 p90 p100 : ℝ) (s70 s80 s90 s100 : ℕ) : ℝ :=
  p70 * s70 + p80 * s80 + p90 * s90 + p100 * s100

theorem difference_between_mean_and_median
  (p70 p80 p90 p100 : ℝ)
  (h_sum : percentage_students p70 p80 p90 p100)
  (s70 s80 s90 s100 : ℕ) :
  median_score p70 p80 p90 p100 s70 s80 s90 s100 - mean_score p70 p80 p90 p100 s70 s80 s90 s100 = 3 :=
  sorry

end difference_between_mean_and_median_l531_531273


namespace sasha_remainder_l531_531784

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l531_531784


namespace find_line_equation_l531_531931

def is_line_through_point (A B C : ℝ) (P : ℝ × ℝ) : Prop :=
  A * P.1 + B * P.2 + C = 0

def is_equidistant_from_points (A B C : ℝ) (M N : ℝ × ℝ) : Prop :=
  let dist (A B C : ℝ) (P : ℝ × ℝ) : ℝ := abs (A * P.1 + B * P.2 + C) / sqrt (A^2 + B^2)
  dist A B C M = dist A B C N

theorem find_line_equation 
  (P : ℝ × ℝ) (M N : ℝ × ℝ) : 
  P = (1, 2) → M = (2, 3) → N = (4, -5) →
  (∃ (A B C : ℝ), is_line_through_point A B C P ∧ 
                   is_equidistant_from_points A B C M N ∧ 
                   ((A, B, C) = (4, 1, -6) ∨ (A, B, C) = (3, 2, -7))) :=
by
  intro hP hM hN
  existsi [4, 1, -6]
  existsi [3, 2, -7]
  split
  { sorry }
  split
  { sorry }
  left -- or choose the other solution using right
  { sorry }


end find_line_equation_l531_531931


namespace kevin_kangaroo_hops_l531_531240

theorem kevin_kangaroo_hops :
  let distance (n : ℕ) : ℚ :=
    if n = 0 then 2 else (3/4)^n * 2 in
  (∑ i in Finset.range 6, (1/4) * distance i) = 1321 / 1024 :=
by
  sorry

end kevin_kangaroo_hops_l531_531240


namespace percentage_increase_third_year_l531_531952

theorem percentage_increase_third_year
  (initial_price : ℝ)
  (price_2007 : ℝ := initial_price * (1 + 20 / 100))
  (price_2008 : ℝ := price_2007 * (1 - 25 / 100))
  (price_end_third_year : ℝ := initial_price * (108 / 100)) :
  ((price_end_third_year - price_2008) / price_2008) * 100 = 20 :=
by
  sorry

end percentage_increase_third_year_l531_531952


namespace probability_diff_by_three_l531_531476

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l531_531476


namespace sqrt_expression_form_l531_531205

theorem sqrt_expression_form (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : c = 24):
  (∃ k, (√8 + 1/√8 + √9 + 1/√9) = (a * √8 + b * √9) / c) →
  (a + b + c = 158) :=
by
  sorry

end sqrt_expression_form_l531_531205


namespace price_increase_to_restore_original_l531_531860

-- Define the initial conditions explicitly
def initial_price : ℝ := 100
def initial_reduction : ℝ := 0.15
def special_sale_reduction : ℝ := 0.30
def tax_rate : ℝ := 0.10

-- Define the transformed prices after each step
def reduced_price_after_initial : ℝ := initial_price * (1 - initial_reduction)
def reduced_price_after_sale : ℝ := reduced_price_after_initial * (1 - special_sale_reduction)
def price_after_tax : ℝ := reduced_price_after_sale * (1 + tax_rate)

-- Define the calculated values necessary for the final part of the problem
def increase_needed : ℝ := initial_price - price_after_tax
def percentage_increase : ℝ := (increase_needed / price_after_tax) * 100

-- The Lean 4 statement proving the needed percentage increase
theorem price_increase_to_restore_original :
  abs (percentage_increase - 52.78) < 0.01 := by
    sorry

end price_increase_to_restore_original_l531_531860


namespace trajectory_P_min_AB_l531_531227

-- Define points and conditions
def point := ℝ × ℝ

def F : point := (√2, 0)
def line_x_eq_2sqrt2 (P : point) := P.1 = 2 * √2

def distance (P Q : point) : ℝ := sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
def distance_point_to_line (P : point) (line : point → Prop) : ℝ :=
  abs (P.1 - 2 * √2)

def condition1 (P : point) : Prop :=
  (distance P F) / (distance_point_to_line P line_x_eq_2sqrt2) = (√2) / 2

noncomputable def ellipse_C (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 2 = 1

-- Trajectory equation of point P given the condition1
theorem trajectory_P (P : point) (h : condition1 P) : ellipse_C P.1 P.2 :=
  sorry

-- Functions to represent perpendicular vectors and their dot product
def dot_product (v1 v2 : point) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

def perpendicular (O A B : point) : Prop := dot_product (A.1, A.2) (B.1, B.2) = 0

-- Given points A and B, OA and OB perpendicular and B is on the ellipse
def point_A (t : ℝ) : point := (t, 2)
def point_B (x0 y0 : ℝ) : point := (x0, y0)

def condition2 (t x0 y0 : ℝ) : Prop :=
  perpendicular (0, 0) (t, 2) (x0, y0) ∧ ellipse_C x0 y0

-- Find the minimum length of |AB|
theorem min_AB (t x0 y0 : ℝ) (h : condition2 t x0 y0) : ∃ (min_length : ℝ), min_length = 2 * √2 :=
  sorry

end trajectory_P_min_AB_l531_531227


namespace gas_cost_correct_l531_531657

def gas_cost (a b c x : ℝ) : ℝ :=
  if h : 0 < x ∧ x <= 310 then a * x
  else if h : 310 < x ∧ x <= 520 then 310 * a + b * (x - 310)
  else 310 * a + 210 * b + c * (x - 520)

theorem gas_cost_correct (x : ℝ) (hx_pos : 0 < x) : 
  gas_cost 3 3.3 4.2 x = 
    if h : x <= 310 then 3 * x
    else if h : x <= 520 then 310 * 3 + 3.3 * (x - 310)
    else 310 * 3 + 210 * 3.3 + 4.2 * (x - 520) :=
by 
  sorry

end gas_cost_correct_l531_531657


namespace sasha_remainder_l531_531764

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l531_531764


namespace cyclic_quadrilateral_area_l531_531276

variable (a b c d R : ℝ)
noncomputable def p : ℝ := (a + b + c + d) / 2
noncomputable def Brahmagupta_area : ℝ := Real.sqrt ((p a b c d - a) * (p a b c d - b) * (p a b c d - c) * (p a b c d - d))

theorem cyclic_quadrilateral_area :
  Brahmagupta_area a b c d = Real.sqrt ((a * b + c * d) * (a * d + b * c) * (a * c + b * d)) / (4 * R) := sorry

end cyclic_quadrilateral_area_l531_531276


namespace problem_proof_l531_531171

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem problem_proof :
  ∃ a b x y : ℝ, a > b ∧ b > 0 ∧ (1, 3 / 2) ∈ { (x, y) | ellipse_equation a b x y } ∧
  (∃ c : ℝ, c = a / 2 ∧ ellipse_equation 2 (sqrt 3) x y ∧
  ∀ M A B : (ℝ × ℝ),
  M = (2, 0) ∧ A ≠ M ∧ B ≠ M ∧ ellipse_equation 2 (sqrt 3) (prod.fst A) (prod.snd A) ∧ 
  ellipse_equation 2 (sqrt 3) (prod.fst B) (prod.snd B) ∧
  (prod.snd A / (prod.fst A - 2)) * (prod.snd B / (prod.fst B - 2)) = 1 / 4 →
  ∃ k m : ℝ, y = k * x + m ∧ m = 4 * k ∧ (prod.fst A = -4) ∧ (prod.snd A = 0)) :=
begin
  sorry
end

end problem_proof_l531_531171


namespace sum_of_three_squares_l531_531707

theorem sum_of_three_squares (n : ℤ) : ∃ a b c : ℤ, 3 * (n - 1)^2 + 8 = a^2 + b^2 + c^2 :=
by
  let N := 3 * (n - 1)^2 + 8
  use [n - 3, n - 1, n + 1]
  simp
  sorry

end sum_of_three_squares_l531_531707


namespace rectangular_prism_total_count_l531_531412

-- Define the dimensions of the rectangular prism
def length : ℕ := 4
def width : ℕ := 3
def height : ℕ := 5

-- Define the total count of edges, corners, and faces
def total_count : ℕ := 12 + 8 + 6

-- The proof statement that the total count is 26
theorem rectangular_prism_total_count : total_count = 26 :=
by
  sorry

end rectangular_prism_total_count_l531_531412


namespace batsman_average_after_17_matches_l531_531853

theorem batsman_average_after_17_matches (A : ℕ) (h : (17 * (A + 3) = 16 * A + 87)) : A + 3 = 39 := by
  sorry

end batsman_average_after_17_matches_l531_531853


namespace range_of_m_l531_531945

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x - 2 < 0) → -8 < m ∧ m ≤ 0 :=
sorry

end range_of_m_l531_531945


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531445

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531445


namespace diagonals_of_square_equal_l531_531730

-- Definitions from conditions:
def is_rhombus (quadrilateral : Type) : Prop := sorry -- A placeholder definition for rhombus
def is_square (quadrilateral : Type) : Prop := sorry -- A placeholder definition for square
def diagonals_equal (quadrilateral : Type) : Prop := sorry -- A placeholder definition for diagonals equality

-- Assumption that should be incorrect
axiom rhombus_diagonals_equal (R : Type) (hR : is_rhombus R) : diagonals_equal R := sorry

-- Assumption
axiom square_is_rhombus (S : Type) (hS : is_square S) : is_rhombus S := sorry

-- Goal: Show that the diagonals of a square are equal based on the assumptions
theorem diagonals_of_square_equal (S : Type) (hS : is_square S) : diagonals_equal S :=
by 
  have hR : is_rhombus S := square_is_rhombus S hS
  exact rhombus_diagonals_equal S hR

end diagonals_of_square_equal_l531_531730


namespace bob_twice_alice_l531_531881

open ProbabilityTheory

theorem bob_twice_alice (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1000) (hb : 0 ≤ b ∧ b ≤ 3000) :
  P (λ (x : ℝ × ℝ), x.2 ≥ 2 * x.1) = 1 / 2 := by
sorry

end bob_twice_alice_l531_531881


namespace arithmetic_geometric_sum_term_l531_531970

-- Given conditions as definitions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), d ≠ 0 ∧ ∀ n, a n = a1 + d * (n - 1)

def geometric_triplet (a1 a2 a4 : ℝ) : Prop :=
  (a1 + a2) * (a1 + a4) = (a2 - a1) * (a4 + a2)

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a (i + 1)

-- Problem statement without proof
theorem arithmetic_geometric_sum_term (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :
  arithmetic_sequence a ∧ geometric_triplet a1 (a 2) (a 4) ∧ S a 5 = 30 →
  (∀ n, a n = 2 * n) ∧ (∑ n in finset.range 20, 1 / (a n * a (n + 1)) = 5 / 21) :=
by
  sorry

end arithmetic_geometric_sum_term_l531_531970


namespace joan_seashells_left_l531_531676

theorem joan_seashells_left (original_seashells : ℕ) (given_seashells : ℕ) (seashells_left : ℕ)
  (h1 : original_seashells = 70) (h2 : given_seashells = 43) : seashells_left = 27 :=
by
  sorry

end joan_seashells_left_l531_531676


namespace joker_probability_l531_531044

-- Definition of the problem parameters according to the conditions
def total_cards := 54
def jokers := 2

-- Calculate the probability
def probability (favorable : Nat) (total : Nat) : ℚ :=
  favorable / total

-- State the theorem that we want to prove
theorem joker_probability : probability jokers total_cards = 1 / 27 := by
  sorry

end joker_probability_l531_531044


namespace magnitude_of_complex_number_l531_531537

theorem magnitude_of_complex_number : 
  ∀ (a b : ℤ), (a = 3 ∧ b = -10) → (|complex.mk a b| = real.sqrt (a^2 + b^2)) := 
by
  intros a b hb
  cases hb
  rw [hb_left, hb_right]
  exact sorry

end magnitude_of_complex_number_l531_531537


namespace solve_x_l531_531559

theorem solve_x :
  ∃ x : ℝ, 2.5 * ( ( x * 0.48 * 2.50 ) / ( 0.12 * 0.09 * 0.5 ) ) = 2000.0000000000002 ∧ x = 3.6 :=
by sorry

end solve_x_l531_531559


namespace blood_drops_per_liter_l531_531045

def mosquito_drops : ℕ := 20
def fatal_blood_loss_liters : ℕ := 3
def mosquitoes_to_kill : ℕ := 750

theorem blood_drops_per_liter (D : ℕ) (total_drops : ℕ) : 
  (total_drops = mosquitoes_to_kill * mosquito_drops) → 
  (fatal_blood_loss_liters * D = total_drops) → 
  D = 5000 := 
  by 
    intros h1 h2
    sorry

end blood_drops_per_liter_l531_531045


namespace eggs_left_after_capital_recovered_l531_531713

-- Conditions as definitions
def eggs_in_crate := 30
def crate_cost_dollars := 5
def price_per_egg_cents := 20

-- The amount of cents in a dollar
def cents_per_dollar := 100

-- Total cost in cents
def crate_cost_cents := crate_cost_dollars * cents_per_dollar

-- The number of eggs needed to recover the capital
def eggs_to_recover_capital := crate_cost_cents / price_per_egg_cents

-- The number of eggs left
def eggs_left := eggs_in_crate - eggs_to_recover_capital

-- The theorem stating the problem
theorem eggs_left_after_capital_recovered : eggs_left = 5 :=
by
  sorry

end eggs_left_after_capital_recovered_l531_531713


namespace sasha_remainder_is_20_l531_531787

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l531_531787


namespace selling_price_of_article_l531_531405

-- Define the conditions
def CP : ℝ := 20
def gain_percent : ℝ := 75

-- Define the gain amount based on the given conditions
def gain_amount : ℝ := (gain_percent / 100) * CP

-- Define the selling price based on the gain amount and the cost price
def SP : ℝ := CP + gain_amount

-- State the theorem
theorem selling_price_of_article : SP = 35 := by
  -- Sorry to skip the proof
  sorry

end selling_price_of_article_l531_531405


namespace number_of_valid_sets_l531_531151

-- Define the subset condition: M is a subset of {-1, 0, 2} and contains exactly two elements
def valid_set (M : Set ℤ) : Prop :=
  M ⊆ {-1, 0, 2} ∧ M.size = 2

-- Define the problem in Lean: Prove that there are exactly 3 such sets
theorem number_of_valid_sets : 
  Finset.card (Finset.filter valid_set (Finset.powerset { -1, 0, 2 }.to_finset)) = 3 := 
sorry

end number_of_valid_sets_l531_531151


namespace trig_expression_value_l531_531598

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 2) : 
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := 
by
  sorry

end trig_expression_value_l531_531598


namespace prove_height_ratio_oil_barrel_l531_531399

noncomputable def height_ratio_oil_barrel (h R: ℝ) : ℝ :=
  let area_horizontal := (1/4 * π * R^2 - 1/2 * R^2);
  let volume_horizontal: ℝ := area_horizontal * h;
  let volume_vertical (x: ℝ) : ℝ := π * R^2 * x;
  volume_horizontal = volume_vertical ((π - 2) / (4 * π) * h)

theorem prove_height_ratio_oil_barrel (h R: ℝ) :
  (∀ a : ℝ, a = height_ratio_oil_barrel h R) → 
  ∃ (ratio: ℝ), ratio = ((1 / 4) - (1 / (2 * π))) :=
begin 
  intro hR,
  use ((1 / 4) - (1 / (2 * π))),
  sorry
end

end prove_height_ratio_oil_barrel_l531_531399


namespace probability_differ_by_three_is_one_sixth_l531_531463

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l531_531463


namespace notebook_cost_l531_531264

-- Define the conditions
def cost_book := 16
def cost_binder := 2
def num_binders := 3
def num_notebooks := 6
def total_cost := 28

-- Define the target statement we need to prove
theorem notebook_cost :
  let cost_of_binders := num_binders * cost_binder in
  let cost_non_notebooks := cost_book + cost_of_binders in
  let cost_of_notebooks := total_cost - cost_non_notebooks in
  (cost_of_notebooks / num_notebooks) = 1 := 
by sorry

end notebook_cost_l531_531264


namespace smallest_value_l531_531740

theorem smallest_value (a b c d: ℝ)
  (h₁ : P(-1) = 4)  -- P(-1) = 1 - a + b - c + d
  (h₂ : d > 5)  -- d > 5
  (h₃ : P(1) > 1.5)  -- P(1) = 1 + a + b + c + d
  (h₄ : -a > 4) :  -- sum of real zeros > 4
  ∃ p, p < 6/5 ∧ ∀ q, q ≠ p → q ≠ p := 
by
  -- Proving here that the product of the non-real zeros is the smallest value
  sorry

end smallest_value_l531_531740


namespace exists_a_even_functions_l531_531612

def f (x a : ℝ) : ℝ := x^2 + (Real.pi - a) * x
def g (x a : ℝ) : ℝ := Real.cos (2 * x + a)

def is_even (h : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, h (-x) = h x

theorem exists_a_even_functions : ∃ a : ℝ, is_even (f · a) ∧ is_even (g · a) :=
by
  sorry

end exists_a_even_functions_l531_531612


namespace max_difference_of_mean_505_l531_531253

theorem max_difference_of_mean_505 (x y : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : 100 ≤ y ∧ y ≤ 999) (h3 : (x + y) / 2 = 505) : 
  x - y ≤ 810 :=
sorry

end max_difference_of_mean_505_l531_531253


namespace mult_mod_7_zero_l531_531088

theorem mult_mod_7_zero :
  (2007 ≡ 5 [MOD 7]) →
  (2008 ≡ 6 [MOD 7]) →
  (2009 ≡ 0 [MOD 7]) →
  (2010 ≡ 1 [MOD 7]) →
  (2007 * 2008 * 2009 * 2010 ≡ 0 [MOD 7]) :=
by
  intros h1 h2 h3 h4
  sorry

end mult_mod_7_zero_l531_531088


namespace probability_of_vowel_initials_l531_531647

/-
In Mrs. Vale's class, there are 18 students. Each student's initials are unique and of the form YY, where both the first and last initials are the same. The available initials are from a subset of letters {B, C, D, F, G, H, J, K, L, M, N, P, Q, R, S, T, V, Y}. Given that the vowel list only includes the letter "Y," prove that the probability of picking a student with initials from the vowel list is 1/18.
-/
theorem probability_of_vowel_initials :
  let initials := {B, C, D, F, G, H, J, K, L, M, N, P, Q, R, S, T, V, Y}
  let vowels := {Y}
  (∀ student | student ∈ initials, ∃ initial | initials.contains initial = true) → 
  (∀ student | student ∈ initials, (student = Y) → true) →
  (vowels.size / initials.size = 1 / 18) :=
by
  sorry

end probability_of_vowel_initials_l531_531647


namespace radius_of_sphere_through_A_and_B_l531_531749

theorem radius_of_sphere_through_A_and_B
  (A B : Point)
  (AB : Real)
  (angle_intersection : Real)
  (ratio_division : Real) :
  AB = 8 →
  angle_intersection = 30 →
  ratio_division = 1 / 3 →
  ∃ (R : Real), R = 2 * Real.sqrt 7 :=
by
  intro hAB hangle hratio
  use 2 * Real.sqrt 7
  sorry

end radius_of_sphere_through_A_and_B_l531_531749


namespace sin_cos_identity_l531_531976

theorem sin_cos_identity (θ : ℝ) (h : Real.tan (θ + (Real.pi / 4)) = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = -7/5 := 
by 
  sorry

end sin_cos_identity_l531_531976


namespace yellow_crayons_count_l531_531796

def red_crayons := 14
def blue_crayons := red_crayons + 5
def yellow_crayons := 2 * blue_crayons - 6

theorem yellow_crayons_count : yellow_crayons = 32 := by
  sorry

end yellow_crayons_count_l531_531796


namespace find_a_value_of_a_l531_531626

noncomputable def possible_value_of_a : ℕ → Prop :=
  λ a, a > 0 ∧ let r1 := 2, r2 := 8, d := Real.sqrt (16 + a^2) in
    6 < d ∧ d < 10

theorem find_a_value_of_a :
  ∃ (a : ℕ), possible_value_of_a a :=
begin
  use 5,
  unfold possible_value_of_a,
  split,
  { exact Nat.zero_lt_succ _ },
  { dsimp,
    split,
    { norm_num },
    { norm_num } }
end

end find_a_value_of_a_l531_531626


namespace rational_inequality_solution_l531_531328

theorem rational_inequality_solution {x : ℝ} : (4 / (x + 1) ≤ 1) → (x ∈ Set.Iic (-1) ∪ Set.Ici 3) :=
by 
  sorry

end rational_inequality_solution_l531_531328


namespace simplest_square_root_l531_531365

theorem simplest_square_root :
  ∀ (a b c d : Real),
  a = Real.sqrt 0.2 →
  b = Real.sqrt (1 / 2) →
  c = Real.sqrt 6 →
  d = Real.sqrt 12 →
  c = Real.sqrt 6 :=
by
  intros a b c d ha hb hc hd
  simp [ha, hb, hc, hd]
  sorry

end simplest_square_root_l531_531365


namespace range_e1_e2_l531_531971

theorem range_e1_e2 (a1 a2 c x y : ℝ) 
  (h1 : x + y = 2 * a1)
  (h2 : x - y = 2 * a2)
  (h3 : 2 * c = y)
  (e₁ e₂ : ℝ) 
  (h_e₁ : e₁ = c / a₁)
  (h_e₂ : e₂ = c / a₂) : 
  (1 / 3 < e₁ * e₂) ∧ (e₁ * e₂ < ∞) :=
by
  sorry

end range_e1_e2_l531_531971


namespace probability_4a_5b_units_digit_9_l531_531942

open finset

def units_digit (n : ℕ) : ℕ := n % 10

def probability_units_digit_is_9 : ℚ := 
  let a_set := finset.range 101 \ {0}
  let b_set := finset.range 101 \ {0}
  let outcomes := (a_set.product b_set).filter (λ p, units_digit (4^p.1 + 5^p.2) = 9)
  (outcomes.card : ℚ) / (a_set.card * b_set.card)

theorem probability_4a_5b_units_digit_9 : probability_units_digit_is_9 = 1 / 2 :=
by sorry

end probability_4a_5b_units_digit_9_l531_531942


namespace volume_triangular_pyramid_l531_531133

variables (OA OB OC : ℝ^3)
variables (hOA : ‖OA‖ = 5) (hOB : ‖OB‖ = 2) (hOC : ‖OC‖ = 6)
variables (hOAdotOB : dot_product OA OB = 0) (hOAdotOC : dot_product OA OC = 0) (hOBdotOC : dot_product OB OC = 8)

theorem volume_triangular_pyramid :
  volume_of_pyramid OA OB OC = (10 * real.sqrt 5) / 3 := 
sorry

end volume_triangular_pyramid_l531_531133


namespace correct_limiting_reagent_and_yield_l531_531566

noncomputable def balanced_reaction_theoretical_yield : Prop :=
  let Fe2O3_initial : ℕ := 4
  let CaCO3_initial : ℕ := 10
  let moles_Fe2O3_needed_for_CaCO3 := Fe2O3_initial * (6 / 2)
  let limiting_reagent := if CaCO3_initial < moles_Fe2O3_needed_for_CaCO3 then true else false
  let theoretical_yield := (CaCO3_initial * (3 / 6))
  limiting_reagent = true ∧ theoretical_yield = 5

theorem correct_limiting_reagent_and_yield : balanced_reaction_theoretical_yield :=
by
  sorry

end correct_limiting_reagent_and_yield_l531_531566


namespace max_vertex_sum_l531_531565

theorem max_vertex_sum
  (a U : ℤ)
  (hU : U ≠ 0)
  (hA : 0 = a * 0 * (0 - 3 * U))
  (hB : 0 = a * (3 * U) * ((3 * U) - 3 * U))
  (hC : 12 = a * (3 * U - 1) * ((3 * U - 1) - 3 * U))
  : ∃ N : ℝ, N = (3 * U) / 2 - (9 * a * U^2) / 4 ∧ N ≤ 17.75 :=
by sorry

end max_vertex_sum_l531_531565


namespace solve_system_l531_531127

theorem solve_system :
  ∃ (x y : ℚ), 3 * x + y = 2 ∧ 2 * x - y = 8 ∧ x = 26 / 9 ∧ y = -20 / 3 :=
by {
  use [26 / 9, -20 / 3],
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
}

end solve_system_l531_531127


namespace dishonest_dealer_profit_percentages_l531_531040

theorem dishonest_dealer_profit_percentages :
  let weightA := 800 / 1000
  let weightB := 850 / 1000
  let weightC := 780 / 1000
  let profit_percent (actual_weight : ℚ) : ℚ := ((1 - actual_weight) / actual_weight) * 100
  profit_percent weightA = 25 ∧
  profit_percent weightB ≈ 17.65 ∧
  profit_percent weightC ≈ 28.21 :=
by
  sorry

end dishonest_dealer_profit_percentages_l531_531040


namespace average_salary_non_officers_l531_531213

theorem average_salary_non_officers
  (average_salary_all : ℝ)
  (average_salary_officers : ℝ)
  (number_officers : ℕ)
  (number_non_officers : ℕ)
  (total_employees : ℕ)
  (total_salary_all : ℝ)
  (h1 : average_salary_all = 120)
  (h2 : average_salary_officers = 440)
  (h3 : number_officers = 15)
  (h4 : number_non_officers = 480)
  (h5 : total_employees = 495)
  (h6 : total_salary_all = 59400) :
  let X := (total_salary_all - (number_officers * average_salary_officers)) / number_non_officers in
  X = 110 :=
by
  sorry

end average_salary_non_officers_l531_531213


namespace max_area_equilateral_triangle_in_rectangle_l531_531756

/-- Maximum possible area of an equilateral triangle inscribed in a rectangle with sides 12 and 13 --/
theorem max_area_equilateral_triangle_in_rectangle :
  ∃ A B C : ℂ, 
    A = 0 ∧
    B.re = 12 ∧
    0 ≤ B.im ∧ B.im ≤ 13 ∧
    C.im = 13 ∧
    C.re ≤ 12 ∧
    ∃ (area : ℝ), 
      area = (205 * Real.sqrt 3 - 468) ∧
      area ≥ ∀ (x y z : ℂ),
        (x = 0 ∧ y.re = 12 ∧ 0 ≤ y.im ∧ y.im ≤ 13 ∧ z.im = 13 ∧ z.re ≤ 12 →
        let base := Complex.abs (y - x),
            height := Complex.abs (z - y) in
          area / 2 * base * height) := sorry

end max_area_equilateral_triangle_in_rectangle_l531_531756


namespace choose_copresidents_l531_531033

theorem choose_copresidents (total_members : ℕ) (departments : ℕ) (members_per_department : ℕ) 
    (h1 : total_members = 24) (h2 : departments = 4) (h3 : members_per_department = 6) :
    ∃ ways : ℕ, ways = 54 :=
by
  sorry

end choose_copresidents_l531_531033


namespace sasha_remainder_l531_531762

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l531_531762


namespace problem_statement_l531_531982

-- Definitions of vectors as per the problem conditions.
def n1 := (1 : ℝ, -2 : ℝ, -1 / 2 : ℝ)
def n2 := (-1 : ℝ, 0 : ℝ, -2 : ℝ)
def a := (1 : ℝ, 0 : ℝ, 2 : ℝ)
def b := (0 : ℝ, 1 : ℝ, -2 : ℝ)

-- Statement to prove the correct options B and C.
theorem problem_statement : 
  (dot_product n1 n2 = 0) ∧ (¬is_scalar_multiple a b) :=
by
  sorry

end problem_statement_l531_531982


namespace product_of_g_l531_531248

def U : set (ℕ × ℕ) := { (x, y) | x ∈ {0, 1, 2, 3, 4, 5} ∧ y ∈ {0, 1, 2, 3, 4} }

structure triangle :=
(vertex1 : ℕ × ℕ)
(vertex2 : ℕ × ℕ)
(vertex3 : ℕ × ℕ)
(right_angle_at : ℕ × ℕ)

def V : set triangle :=
  { t | t.vertex1 ∈ U ∧ t.vertex2 ∈ U ∧ t.vertex3 ∈ U ∧
        (t.right_angle_at = t.vertex1 ∨ t.right_angle_at = t.vertex2 ∨ t.right_angle_at = t.vertex3) }

noncomputable def g (t : triangle) : ℝ :=
  if t.right_angle_at = t.vertex1 then
    let v2 := t.vertex2 in
    let v3 := t.vertex3 in
    ((v3.2 - v2.2) : ℝ) / ((v3.1 - v2.1) : ℝ)
  else if t.right_angle_at = t.vertex2 then
    let v1 := t.vertex1 in
    let v3 := t.vertex3 in
    ((v3.2 - v1.2) : ℝ) / ((v3.1 - v1.1) : ℝ)
  else
    let v1 := t.vertex1 in
    let v2 := t.vertex2 in
    ((v2.2 - v1.2) : ℝ) / ((v2.1 - v1.1) : ℝ)

theorem product_of_g (h : ∀ t ∈ V, g t = 1) : (∏ t in finset.univ, g t) = 1 := 
by {
  sorry
}


end product_of_g_l531_531248


namespace probability_diff_by_3_l531_531479

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l531_531479


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531443

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531443


namespace shape_is_plane_l531_531134

-- Define the spherical coordinates and the constant k
structure SphericalCoord where
  ρ : ℝ  -- Radial distance
  θ : ℝ  -- Polar angle
  φ : ℝ  -- Azimuthal angle

-- The constant k
def k : ℝ := some_constant_value -- Placeholder for the constant k

-- The given conditions and the proof goal
def problem_statement (s : SphericalCoord) : Prop :=
  s.θ = k

theorem shape_is_plane (s : SphericalCoord) (h : problem_statement s) : 
  ∃ a b c d : ℝ, a * s.ρ * sin s.φ * cos k + b * s.ρ * sin s.φ * sin k + c * s.ρ * cos s.φ + d = 0 :=
sorry

end shape_is_plane_l531_531134


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531444

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531444


namespace area_parallelogram_l531_531081

variables (p q : ℝ^3)
variable a : ℝ^3 := p - 3 * q
variable b : ℝ^3 := p + 2 * q

-- Conditions
axiom norm_p : ‖p‖ = (1 / 5 : ℝ)
axiom norm_q : ‖q‖ = 1
axiom angle_pq : real.angle p q = (real.pi / 2)

-- Theorem to Prove
theorem area_parallelogram : ‖a × b‖ = 1 := 
by sorry

end area_parallelogram_l531_531081


namespace exists_zero_in_interval_l531_531177

def f (x : ℝ) : ℝ := (6 / x) - Real.logBase 2 x

theorem exists_zero_in_interval :
  ∃ x ∈ Ioo (2 : ℝ) 4, f x = 0 :=
begin
  sorry
end

end exists_zero_in_interval_l531_531177


namespace max_telephones_batch_l531_531279

theorem max_telephones_batch (n : ℕ) (hq1 : ℕ) (hq2 : ℕ) (tg1 : ℕ) (tg2 : ℕ)
  (hq1_apply : hq1 = 49) (tg1_apply : tg1 = 50)
  (hq2_apply : hq2 = 7) (tg2_apply : tg2 = 8)
  (quality_rate : (hq1 + x * hq2) / (tg1 + x * tg2) ≥ 0.9) :
  n ≤ 210 :=
begin
  -- Proof goes here
  sorry
end

end max_telephones_batch_l531_531279


namespace probability_of_differ_by_three_l531_531498

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l531_531498


namespace area_of_tangent_segments_l531_531962

-- Definitions: Circle radius, line segment length, tangent property, and area of region
axiom circle_radius : ℝ
axiom line_segment_length : ℝ
axiom is_tangent_midpoint (line_segment : ℝ → ℝ) (circle_center : ℝ × ℝ) : Prop
axiom region_area (segments_configuration : set (ℝ → ℝ)) : ℝ

-- Conditions
def circle_radius_eq_two : circle_radius = 2 := sorry
def line_segment_length_eq_two : line_segment_length = 2 := sorry
def tangency_condition (line_segment : ℝ → ℝ) : is_tangent_midpoint line_segment (0, 0) :=
  sorry

-- The main statement to be proved
theorem area_of_tangent_segments : region_area {line_segment | is_tangent_midpoint line_segment (0, 0)} = π := sorry

end area_of_tangent_segments_l531_531962


namespace probability_of_differ_by_three_l531_531500

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l531_531500


namespace find_y_value_l531_531132

theorem find_y_value : 
  (15^2 * 8^3) / y = 450 → y = 256 :=
by
  sorry

end find_y_value_l531_531132


namespace part_a_part_b_part_c_l531_531690

-- Define S(n) as the largest integer such that for every positive integer k with k ≤ S(n),
-- n^2 can be expressed as the sum of k positive square numbers.
def S (n : ℕ) : ℕ := sorry

-- Theorem (a): For every n ≥ 4, S(n) ≤ n^2 - 14.
theorem part_a (n : ℕ) (h : n ≥ 4) : S(n) ≤ n^2 - 14 := sorry

-- Theorem (b): There exists a positive integer n such that S(n) = n^2 - 14.
theorem part_b : ∃ n : ℕ, S(n) = n^2 - 14 := sorry

-- Theorem (c): There are infinitely many positive integers n such that S(n) = n^2 - 14.
theorem part_c : ∀ n : ℕ, ∃ m : ℕ, S(m * n) = (m * n)^2 - 14 := sorry

end part_a_part_b_part_c_l531_531690


namespace number_of_questions_in_test_l531_531333

variable (n : ℕ) -- the total number of questions
variable (correct_answers : ℕ) -- the number of correct answers
variable (sections : ℕ) -- number of sections in the test
variable (questions_per_section : ℕ) -- number of questions per section
variable (percentage_correct : ℚ) -- percentage of correct answers

-- Given conditions
def conditions := 
  correct_answers = 32 ∧ 
  sections = 5 ∧ 
  questions_per_section * sections = n ∧ 
  (70 : ℚ) < percentage_correct ∧ 
  percentage_correct < 77 ∧ 
  percentage_correct * n = 3200

-- The main statement to prove
theorem number_of_questions_in_test : conditions n correct_answers sections questions_per_section percentage_correct → 
  n = 45 :=
by
  sorry

end number_of_questions_in_test_l531_531333


namespace pies_sold_each_day_l531_531871

theorem pies_sold_each_day (total_pies : ℕ) (days_in_week : ℕ) (h1 : total_pies = 56) (h2 : days_in_week = 7) :
  (total_pies / days_in_week = 8) :=
by
exact sorry

end pies_sold_each_day_l531_531871


namespace sasha_remainder_l531_531770

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l531_531770


namespace recurrent_nested_radicals_l531_531925

theorem recurrent_nested_radicals:
  (let x := sqrt 18 + sqrt (18 + sqrt (18 + sqrt (18 + ...)))) 
  in x = (1 + sqrt 73) / 2 :=
by
  sorry

end recurrent_nested_radicals_l531_531925


namespace length_of_FD_is_20_over_9_l531_531911

/-- 
Given a square piece of paper ABCD with each side measuring 8 cm, 
corner C is folded to point E located one-third the distance from A to D along AD.
When folded, GF represents the crease, and point F falls on CD. 
The length of FD is 20/9 cm. 
-/
theorem length_of_FD_is_20_over_9 : 
  ∀ (A B C D E F G : ℝ) (side_length : ℝ),
  side_length = 8 →
  E = (2 * side_length) / 3 →
  let FD = 20 / 9 in 
  FD = 20 / 9 :=
by
  intros A B C D E F G side_length h_len h_E FD h_FD
  sorry

end length_of_FD_is_20_over_9_l531_531911


namespace sasha_remainder_l531_531774

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l531_531774


namespace sasha_remainder_l531_531783

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l531_531783


namespace limit_special_number_probability_l531_531519

noncomputable def special_number_probability_limit : ℕ → ℝ :=
  λ n, p_n

def is_special (number : list ℕ) (n : ℕ) : Prop :=
  ∃ (A B : list ℕ), A.length = n ∧ B.length = n ∧
  (A ++ B) = number ∧ (A.sum = B.sum)

theorem limit_special_number_probability : 
  tendsto special_number_probability_limit atTop (𝓝 (1 / 2)) := 
sorry

end limit_special_number_probability_l531_531519


namespace sum_of_reciprocals_of_roots_l531_531249

-- Define the problem
variables (a b c d e : ℝ)
variables (z : ℂ)
def polynomial := z^5 + (a : ℂ) * z^4 + (b : ℂ) * z^3 + (c : ℂ) * z^2 + (d : ℂ) * z + (e : ℂ)

-- Assume all roots lie on the circle of radius 2
noncomputable def roots : fin 5 → ℂ := sorry

-- Given condition
axiom roots_on_circle : ∀ i, abs (roots i) = 2

-- Vieta's Formula assumption for the sum of the roots
axiom sum_of_roots : ∑ i, roots i = -a

-- Main theorem statement
theorem sum_of_reciprocals_of_roots : (∑ i, 1 / (roots i)) = - (a / 4) :=
by sorry

end sum_of_reciprocals_of_roots_l531_531249


namespace dice_diff_by_three_probability_l531_531440

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l531_531440


namespace domain_of_f2x_l531_531168

theorem domain_of_f2x (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = f x) : 
  ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, f y = f (2 * x) :=
by
  sorry

end domain_of_f2x_l531_531168


namespace sqrt_sqrt_nine_l531_531358

theorem sqrt_sqrt_nine : Real.sqrt (Real.sqrt 9) = 3 := by
  have h : Real.sqrt 9 = 3 := sorry
  rw [h]
  exact h

end sqrt_sqrt_nine_l531_531358


namespace polynomial_property_l531_531688

noncomputable def f (x : ℝ) : ℝ := x^4 - 4*x^2 + 6 * real.sqrt (x^2 - 2)

theorem polynomial_property {x : ℝ} (h : f (x^2 + 2) = x^4 + 4*x^2 + 6*x) : 
  f (x^2 - 2) = x^4 - 4*x^2 + 6 * real.sqrt (x^2 - 2) := by
  sorry

end polynomial_property_l531_531688


namespace monotonic_sufficient_has_max_min_not_necessary_monotonic_for_max_min_l531_531848

open Set

variable {α : Type*} [LinearOrder α] {β : Type*} [LinearOrder β]

def is_monotonic_on (f : α → β) (s : Set α) : Prop :=
  (∀ x y, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y) ∨
  (∀ x y, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x)

theorem monotonic_sufficient_has_max_min {f : α → β} {a b : α} (h : is_monotonic_on f (Icc a b)) :
  (∃ x ∈ Icc a b, ∀ y ∈ Icc a b, f y ≤ f x) ∧ (∃ x ∈ Icc a b, ∀ y ∈ Icc a b, f x ≤ f y) :=
sorry

theorem not_necessary_monotonic_for_max_min {a b : ℝ} (h : ∃ (f : ℝ → ℝ), 
  (∃ x ∈ Icc a b, ∀ y ∈ Icc a b, f y ≤ f x) ∧ (∃ x ∈ Icc a b, ∀ y ∈ Icc a b, f x ≤ f y) ∧ 
  ¬ is_monotonic_on f (Icc a b)) : true :=
  trivial

end monotonic_sufficient_has_max_min_not_necessary_monotonic_for_max_min_l531_531848


namespace product_conjugate_eq_five_l531_531164

-- Definitions
def z : ℂ := 1 + 2 * complex.I
def z_conj := conj z

-- Theorem
theorem product_conjugate_eq_five : z * z_conj = 5 := by
  sorry

end product_conjugate_eq_five_l531_531164


namespace find_num_candies_bought_l531_531237

-- Conditions
def cost_per_candy := 80
def sell_price_per_candy := 100
def num_sold := 48
def profit := 800

-- Question equivalence
theorem find_num_candies_bought (x : ℕ) 
  (hc : cost_per_candy = 80)
  (hs : sell_price_per_candy = 100)
  (hn : num_sold = 48)
  (hp : profit = 800) :
  48 * 100 - 80 * x = 800 → x = 50 :=
  by
  sorry

end find_num_candies_bought_l531_531237


namespace find_PQ_l531_531115

-- Definition of a 30-60-90 triangle
structure Triangle :=
(vertices : Fin 3 → Point)
(angle_30 : ((vertices 1).angle (vertices 0) (vertices 2) = 30))
(angle_60 : ((vertices 1).angle (vertices 0) (vertices 1) = 60))
(length_PR : (vertices 0).distance (vertices 2) = 6 * Real.sqrt 3)

-- Define the points P, Q, R
def P : Point := ⟨0, 0⟩
def Q : Point := ⟨sqrt 3, 0⟩
def R : Point := ⟨0, 1⟩

-- Define the triangle PQR
def PQR : Triangle := {
  vertices := ![P, Q, R],
  angle_30 := sorry, -- This is where we would prove the right angles and so forth
  angle_60 := sorry, -- This is where we would prove the right angles and so forth
  length_PR := sorry -- Assert the length PR = 6sqrt(3)
}

-- Theorem to prove PQ length
theorem find_PQ (t : Triangle) (h : t = PQR) : (P.distance Q = 6 * Real.sqrt 3) := 
sorry 

end find_PQ_l531_531115


namespace scout_troop_profit_l531_531872

theorem scout_troop_profit (bars_purchased bars_sold : ℕ) (purchase_rate sale_rate : ℚ) 
  (h1 : bars_purchased = 1200) 
  (h2 : purchase_rate = 1 / 3)
  (h3 : sale_rate = 3 / 5) :
  let cost := bars_purchased * purchase_rate,
      revenue := bars_purchased * sale_rate,
      profit := revenue - cost
  in profit = 320 := 
by {
  -- Definitions used in the proof
  have cost_def : cost = bars_purchased * purchase_rate := rfl,
  have revenue_def : revenue = bars_purchased * sale_rate := rfl,
  have profit_def : profit = revenue - cost := rfl,
  
  -- Proof skipped
  sorry
}

end scout_troop_profit_l531_531872


namespace interval_of_increase_log_base_one_third_l531_531317

/-- The function y = log_(1/3)(-3 + 4 * x - x^2) is increasing on the interval (2, 3). -/
theorem interval_of_increase_log_base_one_third :
  ∀ x : ℝ, 2 < x ∧ x < 3 → ((log (λ (a : ℝ), (1 / 3)) (-3 + 4 * x - x^2)) < (log (λ (a : ℝ), (1 / 3)) (-3 + 4 * (x + ε) - (x + ε)^2))) :=
begin
  sorry,
end

end interval_of_increase_log_base_one_third_l531_531317


namespace coefficient_x2y4_in_expansion_l531_531832

theorem coefficient_x2y4_in_expansion :
  (binomial 6 4) = 15 :=
by
  sorry

end coefficient_x2y4_in_expansion_l531_531832


namespace max_value_of_f_period_of_f_monotonic_intervals_of_f_l531_531188

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (2 * cos (x / 2), tan (x / 2 + π / 4))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (sqrt 2 * sin (x / 2 + π / 4), tan (x / 2 - π / 4))

noncomputable def f (x : ℝ) : ℝ :=
  let (a1, a2) := a x
  let (b1, b2) := b x
  a1 * b1 + a2 * b2

theorem max_value_of_f :
  ∃ x, (0 ≤ x ∧ x ≤ π / 2) → f x = sqrt 2 := sorry

theorem period_of_f :
  ∀ x, f (x + 2 * π) = f x := sorry

theorem monotonic_intervals_of_f :
  (∀ x, (0 ≤ x ∧ x ≤ π / 4) → f (x + ε) > f x) ∧ 
  (∀ x, (π / 4 ≤ x ∧ x ≤ π / 2) → f (x + ε) < f x) := sorry

end max_value_of_f_period_of_f_monotonic_intervals_of_f_l531_531188


namespace fraction_of_70cm_ropes_l531_531301

theorem fraction_of_70cm_ropes (R : ℕ) (avg_all : ℚ) (avg_70 : ℚ) (avg_85 : ℚ) (total_len : R * avg_all = 480) 
  (total_ropes : R = 6) : 
  ∃ f : ℚ, f = 1 / 3 ∧ f * R * avg_70 + (R - f * R) * avg_85 = R * avg_all :=
by
  sorry

end fraction_of_70cm_ropes_l531_531301


namespace ten_integers_disjoint_subsets_same_sum_l531_531825

theorem ten_integers_disjoint_subsets_same_sum (S : Finset ℕ) (h : S.card = 10) (h_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ A B : Finset ℕ, A ≠ B ∧ A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end ten_integers_disjoint_subsets_same_sum_l531_531825


namespace avg_of_four_num_prob_correct_l531_531139

noncomputable def avg_of_four_num_prob_eq_five : Prop :=
  let numbers := {1, 3, 4, 6, 7, 9}
  let all_combinations := {s : set ℕ | s ⊆ numbers ∧ s.card = 4}
  let target_combinations := {s ∈ all_combinations | (s.sum id) / 4 = 5}
  (target_combinations.card : ℚ) / (all_combinations.card : ℚ) = 1 / 5

theorem avg_of_four_num_prob_correct : avg_of_four_num_prob_eq_five := sorry

end avg_of_four_num_prob_correct_l531_531139


namespace tangent_line_eq_l531_531996

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * (x - 1/x) - 2 * Real.log x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 2 + 2 / x^2 - 2 / x

-- Define the point-slope form of the tangent line at the point (1, f(1))
theorem tangent_line_eq (x y : ℝ) : f(1) = 0 ∧ f'(1) = 2 → 2 * x - y - 2 = 0 := 
by
  intro h
  have h1 : f(1) = 0 := h.1
  have h2 : f'(1) = 2 := h.2
  sorry

end tangent_line_eq_l531_531996


namespace identify_heaviest_and_lightest_in_13_weighings_l531_531805

-- Definitions based on the conditions
def coins := Finset ℕ
def weighs_with_balance_scale (c1 c2: coins) : Prop := true  -- Placeholder for weighing functionality

/-- There are 10 coins, each with a distinct weight. -/
def ten_distinct_coins (coin_set : coins) : Prop :=
  coin_set.card = 10 ∧ (∀ c1 c2 ∈ coin_set, c1 ≠ c2 → weighs_with_balance_scale c1 c2)

-- Theorem statement
theorem identify_heaviest_and_lightest_in_13_weighings 
  (coin_set : coins)
  (hc: ten_distinct_coins coin_set):
  ∃ (heaviest lightest : coins), 
    weighs_with_balance_scale heaviest coin_set ∧ weighs_with_balance_scale coin_set lightest ∧ 
    -- Assuming weighs_with_balance_scale keeps track of number of weighings
    weights_used coin_set = 13 :=
sorry

end identify_heaviest_and_lightest_in_13_weighings_l531_531805


namespace samanta_s_eggs_left_l531_531719

def total_eggs : ℕ := 30
def cost_per_crate_dollars : ℕ := 5
def cost_per_crate_cents : ℕ := cost_per_crate_dollars * 100
def sell_price_per_egg_cents : ℕ := 20

theorem samanta_s_eggs_left
  (total_eggs : ℕ) (cost_per_crate_dollars : ℕ) (sell_price_per_egg_cents : ℕ) 
  (cost_per_crate_cents = cost_per_crate_dollars * 100) : 
  total_eggs - (cost_per_crate_cents / sell_price_per_egg_cents) = 5 :=
by sorry

end samanta_s_eggs_left_l531_531719


namespace count_equilateral_triangles_l531_531316

theorem count_equilateral_triangles : 
  let lines := λ k : ℤ, (λ (x : ℝ), x * sqrt 3 + 3 * k, λ (x : ℝ), -x * sqrt 3 + 3 * k, λ (x : ℝ), k)
  in ∑ k in (-15 : ℤ)..15, 
    ∑ (x y : ℝ) [∃ t, t ∈ (lines k)], 
      (∃ t', t' ∈ (lines k) ∧ 
        dist t t' = 3 / sqrt 3 ∧ 
        is_triangle t t' ) = 4920 :=
begin
  sorry
end

end count_equilateral_triangles_l531_531316


namespace find_matrix_N_l531_531123

def MatrixN (N : Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  (∀ u : ℝ × ℝ × ℝ, N.mul_vec (λi, match i with
    | 0 => u.1
    | 1 => u.2
    | 2 => u.3
  end) = (λi, match i with
    | 0 => 7 * u.1
    | 1 => 7 * u.2
    | 2 => 7 * u.3
  end)) ∧
  (∀ t : ℝ, N.mul_vec (λi, match i with
    | 2 => t
    | _ => 0
  end) = (λi, match i with
    | 2 => -3 * t
    | _ => 0
  end))

theorem find_matrix_N : ∃ N : Matrix (Fin 3) (Fin 3) ℝ, MatrixN N ∧ N = ![![7, 0, 0], ![0, 7, 0], ![0, 0, -3]] :=
sorry

end find_matrix_N_l531_531123


namespace algebraic_expression_correct_l531_531300

theorem algebraic_expression_correct (x y : ℝ) :
  (x - y)^2 - (x^2 - y^2) = (x - y)^2 - (x^2 - y^2) :=
by
  sorry

end algebraic_expression_correct_l531_531300


namespace function_is_odd_function_l531_531965

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

theorem function_is_odd_function (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f(x) - 2 * f(1 / x) = 3 * x) :
  is_odd_function f :=
by
  sorry

end function_is_odd_function_l531_531965


namespace find_range_t_l531_531995

noncomputable def f (x t : ℝ) : ℝ :=
  if x < t then -6 + Real.exp (x - 1) else x^2 - 4 * x

theorem find_range_t (f : ℝ → ℝ → ℝ)
  (h : ∀ t : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ t = x₁ - 6 ∧ f x₂ t = x₂ - 6 ∧ f x₃ t = x₃ - 6)) :
  ∀ t : ℝ, 1 < t ∧ t ≤ 2 := sorry

end find_range_t_l531_531995


namespace probability_of_diff_3_is_1_over_9_l531_531428

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l531_531428


namespace Alfonso_daily_earnings_l531_531421

-- Define the conditions given in the problem
def helmet_cost : ℕ := 340
def current_savings : ℕ := 40
def days_per_week : ℕ := 5
def weeks_to_work : ℕ := 10

-- Define the question as a property to prove
def daily_earnings : ℕ := 6

-- Prove that the daily earnings are $6 given the conditions
theorem Alfonso_daily_earnings :
  (helmet_cost - current_savings) / (days_per_week * weeks_to_work) = daily_earnings :=
by
  sorry

end Alfonso_daily_earnings_l531_531421


namespace quadrilateral_diagonal_length_l531_531037

theorem quadrilateral_diagonal_length (A B C D : ℝ) (AB AC CD: ℝ) 
    (h_area : 32 = 1 / 2 * AB * CD) 
    (h_sum : AB + AC + CD = 16) : 
    BD = 8 * sqrt 2 := sorry

end quadrilateral_diagonal_length_l531_531037


namespace circles_common_tangents_l531_531625

theorem circles_common_tangents (a : ℕ) (h : a ∈ Set.Set.of List.range (a + 1)) :
  (x: ℝ) (y: ℝ) , x^2 + y^2 = 4 ∧ (x-4)^2 + (y + a) ^ 2 = 64 ∧  6 < Real.sqrt (16 + a^2) ∧ Real.sqrt (16 + a^2) < 10 := by
sorrry

end circles_common_tangents_l531_531625


namespace cos_of_right_angle_D_l531_531216

noncomputable def triangle_def := (D E F : ℝ)

/-- Given a right triangle DEF, with DE=12, EF=5, and ∠D=90°, then cos(D) = 0 -/
theorem cos_of_right_angle_D {D E F : ℝ} (h1 : E ≠ D) (h2 : F ≠ E) (h3 : F ≠ D) 
  (right_triangle : triangle_def D E F) (DE := 12) (EF := 5) (angle_D := 90) : cos D = 0 :=
sorry

end cos_of_right_angle_D_l531_531216


namespace distance_EF_is_6_sqrt_2_l531_531739

noncomputable def point (x y : ℝ) := (x, y)

noncomputable def AB : ℝ := 6
noncomputable def BC : ℝ := 12

noncomputable def A := point (-9) 0
noncomputable def B := point (-3) 0
noncomputable def C := point 3 0

noncomputable def D := point (-3) 6

noncomputable def E := point (-9) 6 -- Assumed coordinates here align with conditions
noncomputable def F := point (3) 0

theorem distance_EF_is_6_sqrt_2 :
  let EF := dist E F in
  EF = 6 * Real.sqrt 2 := by
  sorry

end distance_EF_is_6_sqrt_2_l531_531739


namespace circles_intersect_l531_531745

def circle1 := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}
def circle2 := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 2}

def center1 : ℝ × ℝ := (1, 0)
def center2 : ℝ × ℝ := (0, 1)

def radius1 : ℝ := 1
def radius2 : ℝ := Real.sqrt 2

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circles_intersect : distance center1 center2 < radius1 + radius2 ∧ distance center1 center2 > Real.abs (radius1 - radius2) :=
by
  sorry

end circles_intersect_l531_531745


namespace cos_neg_1500_eq_half_l531_531541

theorem cos_neg_1500_eq_half : Real.cos (-1500 * Real.pi / 180) = 1/2 := by
  sorry

end cos_neg_1500_eq_half_l531_531541


namespace eccentricity_of_conic_section_l531_531261

-- Definition of the conditions
variables (F1 F2 P : ℝ → ℝ) (m : ℝ)
def dist (F P : ℝ → ℝ) : ℝ := sqrt ((F 0 - P 0) ^ 2 + (F 1 - P 1) ^ 2)

-- Hypotheses
axiom h1 : dist P F1 = 4 * m
axiom h2 : dist F1 F2 = 3 * m
axiom h3 : dist P F2 = 2 * m

-- The proof goal
theorem eccentricity_of_conic_section (h1 : dist P F1 = 4 * m) (h2 : dist F1 F2 = 3 * m) (h3 : dist P F2 = 2 * m) :
  ∃ e : ℝ, e = 1 / 2 ∨ e = 3 / 2 :=
begin
  sorry
end

end eccentricity_of_conic_section_l531_531261


namespace domain_log_function_l531_531312

theorem domain_log_function :
  {x : ℝ | 1 - 2^x > 0} = {x : ℝ | x < 0} :=
by
  sorry

end domain_log_function_l531_531312


namespace number_of_routes_l531_531032

-- Define the conditions of the problem
def moves_right : Nat := 8
def moves_up : Nat := 5
def total_moves : Nat := moves_right + moves_up

-- Define binomial coefficient function
def binomial (n k : Nat) : Nat :=
  Math.lib.binomial n k

-- Lean Statement to prove the number of routes equals 12870
theorem number_of_routes : binomial total_moves moves_up = 12870 :=
  sorry

end number_of_routes_l531_531032


namespace probability_of_diff_3_is_1_over_9_l531_531431

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l531_531431


namespace find_x_l531_531361

theorem find_x (x : ℝ) (h : (3 * x) / 4 = 24) : x = 32 :=
by
  sorry

end find_x_l531_531361


namespace triangle_area_inscribed_in_circle_l531_531419

theorem triangle_area_inscribed_in_circle (R : ℝ) 
    (h_pos : R > 0) 
    (h_ratio : ∃ (x : ℝ)(hx : x > 0), 2*x + 5*x + 17*x = 2*π) :
  (∃ (area : ℝ), area = (R^2 / 4)) :=
by
  sorry

end triangle_area_inscribed_in_circle_l531_531419


namespace x_intercepts_of_curve_l531_531629

theorem x_intercepts_of_curve : 
  ∃ y1 y2 y3 : ℝ, (0 = y1^3 - 4*y1^2 + 3*y1 + 2) ∧ 
                (0 = y2^3 - 4*y2^2 + 3*y2 + 2) ∧ 
                (0 = y3^3 - 4*y3^2 + 3*y3 + 2) := 
by {
    -- Declare variables
    let y1 := 1,
    let y2 := (3 + Real.sqrt 17) / 2,
    let y3 := (3 - (Real.sqrt 17)) / 2,
    use [y1, y2, y3],
    split,
    -- Proof of y1
    {
        norm_num,
        sorry -- your detailed proof here (not required as per the instructions)
    },
    split,
    -- Proof of y2
    {
        ring, 
        sorry -- your detailed proof here (not required as per the instructions)
    },
    -- Proof of y3
    {
        ring,
        sorry -- your detailed proof here (not required as per the instructions)
    }
}

end x_intercepts_of_curve_l531_531629


namespace larger_of_two_numbers_l531_531015

theorem larger_of_two_numbers (H : Nat := 15) (f1 : Nat := 11) (f2 : Nat := 15) :
  let lcm := H * f1 * f2;
  ∃ (A B : Nat), A = H * f1 ∧ B = H * f2 ∧ A ≤ B := by
  sorry

end larger_of_two_numbers_l531_531015


namespace identify_heaviest_and_lightest_13_weighings_l531_531816

theorem identify_heaviest_and_lightest_13_weighings (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ f : (Fin 13 → ((Fin 10) × (Fin 10) × ℝ)), true :=
by
  sorry

end identify_heaviest_and_lightest_13_weighings_l531_531816


namespace point_reflection_first_quadrant_l531_531664

noncomputable def magnitude (z : Complex) : ℝ :=
  Complex.abs z

noncomputable def z (w : Complex) : Complex :=
  w / (1 + Complex.i)

theorem point_reflection_first_quadrant :
  ∃ z : Complex, z * (1 + Complex.i) = magnitude (1 + Complex.sqrt 3 * Complex.i) ∧
  let conj_z := Complex.conj z in
  conj_z.re > 0 ∧ conj_z.im > 0 :=
by
  sorry

end point_reflection_first_quadrant_l531_531664


namespace cost_of_bench_eq_150_l531_531859

theorem cost_of_bench_eq_150 (B : ℕ) (h : B + 2 * B = 450) : B = 150 :=
sorry

end cost_of_bench_eq_150_l531_531859


namespace ball_height_after_third_bounce_l531_531391

/-- The problem conditions: initial height of the ball (h₀) and the bounce fraction (b). -/
variables (h₀ : ℝ) (b : ℝ)

/-- The height of the ball after n bounces. -/
def height_after_bounces (n : ℕ) : ℝ :=
  h₀ * (b ^ n)

/-- The specific problem statement: Proving the height after the third bounce. -/
theorem ball_height_after_third_bounce : 
  ∀ h₀ b, h₀ = 25 → b = 2 / 3 → height_after_bounces h₀ b 3 = 25 * 8 / 27 :=
by 
  intros h₀ b h₀_val b_val
  sorry

end ball_height_after_third_bounce_l531_531391


namespace probability_of_juliet_supporter_resides_in_capulet_l531_531027

variable (P : ℕ) -- Let P be the total population of Venezia.

def montague_population := (6 / 8 : ℚ) * P
def capulet_population := P - montague_population

def romeo_supporters_montague := (8 / 10 : ℚ) * montague_population
def juliet_supporters_montague := montague_population - romeo_supporters_montague

def juliet_supporters_capulet := (7 / 10 : ℚ) * capulet_population

def total_juliet_supporters := juliet_supporters_montague + juliet_supporters_capulet

def probability_rhs := juliet_supporters_capulet / total_juliet_supporters
def probability_rhs_percent := probability_rhs * 100

theorem probability_of_juliet_supporter_resides_in_capulet (P_pos : 0 < P):
  probability_rhs_percent = 54 := 
sorry

end probability_of_juliet_supporter_resides_in_capulet_l531_531027


namespace coeff_x2_in_expansion_l531_531223

noncomputable def poly := (x - 1) * (1 / x^2022 + sqrt x + 1)^8

theorem coeff_x2_in_expansion : polynomial.coeff (polynomial.expand ℤ poly) 2 = -42 := 
sorry

end coeff_x2_in_expansion_l531_531223


namespace number_of_valid_pairs_l531_531630

-- Define the set of integers from 1 to 20
def I := {n : ℕ | n ∈ finset.Icc 1 20} 

-- Define the even and odd subsets
def evens := {n ∈ I | n % 2 = 0}
def odds := {n ∈ I | n % 2 = 1}

-- Define the condition for pairs (m, n) where m < n and m + n is even
def valid_pairs (m n : ℕ) := m < n ∧ (m + n) % 2 = 0

theorem number_of_valid_pairs :
  (finset.card {p : ℕ × ℕ | p.1 ∈ I ∧ p.2 ∈ I ∧ valid_pairs p.1 p.2} = 90) :=
sorry

end number_of_valid_pairs_l531_531630


namespace every_player_five_coins_l531_531560

noncomputable def players : List String := ["Abby", "Bernardo", "Carl", "Debra", "Eliza"]

def initial_coins : String → ℕ
| _ => 5 

def rounds : ℕ := 4

def ball_outcomes : List String := ["green", "red", "yellow", "white", "white"]

def transaction_rule (draws: List (String × String)) : String → ℕ
| player => 
  if ∃ draw, player = draw.1 ∧ draw.2 = "green" then
    -1
  else if ∃ draw, player = draw.1 ∧ draw.2 = "red" then
    1
  else 
    0

def final_coins (draws: List (List (String × String))) : String → ℕ
| player => initial_coins player + draws.sumBy (transaction_rule · player)

theorem every_player_five_coins :
  ∀ (draws: List (List (String × String))),
  (∀ player ∈ players, final_coins draws player = 5) →
  (∃ p: ℝ, p = (1 / 81)) :=
begin
  sorry
end

end every_player_five_coins_l531_531560


namespace intersection_A_B_l531_531185

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | abs (x - 2) ≥ 1}
def answer : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_A_B :
  A ∩ B = answer :=
sorry

end intersection_A_B_l531_531185


namespace cynthia_water_balloons_l531_531529

variable (Cynthia Randy Janice : ℕ)

theorem cynthia_water_balloons :
  ( Randy = Janice / 2 ) →
  ( Cynthia = 4 * Randy ) →
  ( Janice = 6 ) →
  ( Cynthia = 12 ) :=
by
  intros h1 h2 h3
  rw [h3, ←h1] at h2
  simp at h2
  exact h2

end cynthia_water_balloons_l531_531529


namespace length_of_the_train_l531_531057

noncomputable def length_of_train (s1 s2 : ℝ) (t1 t2 : ℕ) : ℝ :=
  (s1 * t1 + s2 * t2) / 2

theorem length_of_the_train :
  ∀ (s1 s2 : ℝ) (t1 t2 : ℕ), s1 = 25 → t1 = 8 → s2 = 100 / 3 → t2 = 6 → length_of_train s1 s2 t1 t2 = 200 :=
by
  intros s1 s2 t1 t2 hs1 ht1 hs2 ht2
  rw [hs1, ht1, hs2, ht2]
  simp [length_of_train]
  norm_num

end length_of_the_train_l531_531057


namespace monotonic_decreasing_interval_l531_531610

def f (x : ℝ) : ℝ := sin x ^ 2 + sqrt 3 * sin x * cos x

theorem monotonic_decreasing_interval : 
  ∀ x y : ℝ, (π / 3 < x) → (x < 5 * π / 6) → (x < y) → (y < 5 * π / 6) → f y < f x :=
by
  sorry

end monotonic_decreasing_interval_l531_531610


namespace identify_heaviest_and_lightest_in_13_weighings_l531_531803

-- Definitions based on the conditions
def coins := Finset ℕ
def weighs_with_balance_scale (c1 c2: coins) : Prop := true  -- Placeholder for weighing functionality

/-- There are 10 coins, each with a distinct weight. -/
def ten_distinct_coins (coin_set : coins) : Prop :=
  coin_set.card = 10 ∧ (∀ c1 c2 ∈ coin_set, c1 ≠ c2 → weighs_with_balance_scale c1 c2)

-- Theorem statement
theorem identify_heaviest_and_lightest_in_13_weighings 
  (coin_set : coins)
  (hc: ten_distinct_coins coin_set):
  ∃ (heaviest lightest : coins), 
    weighs_with_balance_scale heaviest coin_set ∧ weighs_with_balance_scale coin_set lightest ∧ 
    -- Assuming weighs_with_balance_scale keeps track of number of weighings
    weights_used coin_set = 13 :=
sorry

end identify_heaviest_and_lightest_in_13_weighings_l531_531803


namespace find_sin_and_tan_l531_531956

variable (θ : ℝ)

axiom condition_cos : cos(θ - π / 4) = 1 / 3
axiom condition_range : π / 2 < θ ∧ θ < π

theorem find_sin_and_tan (h1 : cos(θ - π / 4) = 1 / 3) (h2 : π / 2 < θ ∧ θ < π) :
  sin θ = sqrt(23 / 72) ∧ tan θ = - sqrt(14) / 2 := by
  sorry

end find_sin_and_tan_l531_531956


namespace range_of_b_l531_531611

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := log x / log 10

-- Main theorem statement
theorem range_of_b (a b : ℝ) (h : f a = g b) : 1 ≤ b :=
by
  sorry

end range_of_b_l531_531611


namespace arithmetic_geo_sum_l531_531581

theorem arithmetic_geo_sum (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a (n + 1) = a n + d) →
  (d = 2) →
  (a 3) ^ 2 = (a 1) * (a 4) →
  (a 2 + a 3 = -10) := 
by
  intros h_arith h_d h_geo
  sorry

end arithmetic_geo_sum_l531_531581


namespace Annie_runs_on_Saturday_l531_531509

-- Define days of the week
inductive day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open day

-- Define Annie's activities on each day
def swims : day → Prop
def plays_basketball : day → Prop
def plays_golf : day → Prop
def runs : day → Prop
def plays_tennis : day → Prop

-- Statement of the conditions
axiom Annie_swims_non_consecutive :
  ∃ d1 d2 d3 : day,
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧
    ((d1 = Monday ∨ d1 = Wednesday ∨ d1 = Friday ∨ d1 = Saturday ∨ d1 = Sunday) ∧
     (d2 ≠ Tuesday ∧ d3 ≠ Tuesday)) ∧
    ((d1 = Monday ∨ d1 = Wednesday ∨ d1 = Friday ∨ d1 = Sunday) ∧
     (d2 ≠ Thursday ∧ d3 ≠ Thursday))

axiom plays_basketball_on_Tuesday :
  plays_basketball Tuesday

axiom plays_golf_on_Thursday :
  plays_golf Thursday

axiom no_tennis_after_swim_or_run :
  ∀ d : day,
    (∃ d' : day, swims d' ∧ d = succ d') ∨
    (∃ d' : day, runs d' ∧ d = succ d') → ¬ plays_tennis d

-- The goal
theorem Annie_runs_on_Saturday  :
  runs Saturday :=
by
  sorry

end Annie_runs_on_Saturday_l531_531509


namespace handshake_count_l531_531512

def num_players_on_team := 6
def num_teams := 2
def num_players := num_players_on_team * num_teams
def num_referees := 3

theorem handshake_count : 
  (num_players_on_team * (num_teams - 1) * num_players_on_team) + (num_players * num_referees) = 72 :=
by
  -- handshakes between teams
  have h1 : num_players_on_team * num_teams * (num_teams - 1) / num_teams = num_players_on_team * (num_teams - 1) * num_players_on_team,
  -- handshakes between players and referees
  have h2 : num_players * num_referees = 12 * 3,
  -- total handshakes
  have h3 : (6 * 6 + 12 * 3 = 36 + 36 = 72),
  
  sorry

end handshake_count_l531_531512


namespace probability_diff_by_3_l531_531483

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l531_531483


namespace area_BCD_l531_531667

-- Definitions
def area_of_triangle (base height : ℝ) : ℝ := 0.5 * base * height

-- Given the area of triangle ABC
def area_ABC : ℝ := 27

-- Base of triangle ABC
def base_AC : ℝ := 6

-- Solving for height h from area_ABC = 0.5 * base_AC * h
def height_BC : ℝ := (2 * area_ABC) / base_AC

-- Base of triangle BCD
def base_CD : ℝ := 26

-- Proving the area of triangle BCD
theorem area_BCD : area_of_triangle base_CD height_BC = 117 :=
by sorry

end area_BCD_l531_531667


namespace cyclic_quadrilateral_property_l531_531506

-- Definitions for the configurations and conditions
structure Quadrilateral (P : Type) :=
(A B C D : P)
(inscribed : Circle P)
(no_parallel_sides : ¬(parallel (line_through A B) (line_through C D)) ∧ ¬(parallel (line_through A D) (line_through B C)))

structure PointConfiguration (P : Type) :=
(quadrilateral : Quadrilateral P)
(E F : P)  -- Points of intersection
(intersection1 : collinear [quadrilateral.A, quadrilateral.B, E] ∧ collinear [quadrilateral.C, quadrilateral.D, E])
(intersection2 : collinear [quadrilateral.A, quadrilateral.D, F] ∧ collinear [quadrilateral.B, quadrilateral.C, F])
(equal_segments : dist quadrilateral.A E = dist quadrilateral.C F)

-- Total distance between points
def dist {P : Type} [metric_space P] (p₁ p₂ : P) : ℝ := sorry

-- Main lean statement to prove DE = DF
theorem cyclic_quadrilateral_property {P : Type} [metric_space P] (conf: PointConfiguration P) :
  dist conf.quadrilateral.D conf.E = dist conf.quadrilateral.D conf.F := sorry

end cyclic_quadrilateral_property_l531_531506


namespace ratio_BE_EC_l531_531206

open_locale big_operators

-- Definitions for the problem conditions
variable {A B C F D G E : Point}
variable {ratio_AC : ℕ} -- Ratio in which F divides AC
variable {ratio_BD_DC : ℕ} -- Ratio BD/DC
variable {midpoint_G_BD : Prop} -- G is the midpoint of BD
variable {intersection_E_AG_BC : Prop} -- E is the intersection of AG and BC

-- Translation of the given conditions
def divides_in_ratio (a b n m : Point) (r s : ℕ) : Prop := 
  ∃ λ : ℝ, λ = r / (r + s) ∧ b = (1 - λ) • a + λ • n

-- Special conditions according to the given problem
def special_conditions (A B C F D G E : Point) (r1 r2 : ℕ) : Prop :=
  divides_in_ratio A C F 1 3 ∧
  divides_in_ratio B C D 3 2 ∧
  midpoint G B D ∧
  incidence E A G ∧
  incidence E B C

-- Main theorem statement
theorem ratio_BE_EC (A B C F D G E : Point) :
  special_conditions A B C F D G E 1 3 → divides_in_ratio E B C 2 5 :=
sorry

end ratio_BE_EC_l531_531206


namespace relay_race_total_distance_l531_531285

theorem relay_race_total_distance
  (Sadie_speed : ℝ) (Sadie_time : ℝ) (Ariana_speed : ℝ) (Ariana_time : ℝ) (Sarah_speed : ℝ) (total_race_time : ℝ)
  (h1 : Sadie_speed = 3) (h2 : Sadie_time = 2)
  (h3 : Ariana_speed = 6) (h4 : Ariana_time = 0.5)
  (h5 : Sarah_speed = 4) (h6 : total_race_time = 4.5) :
  (Sadie_speed * Sadie_time + Ariana_speed * Ariana_time + Sarah_speed * (total_race_time - (Sadie_time + Ariana_time))) = 17 :=
by
  sorry

end relay_race_total_distance_l531_531285


namespace simplify_trig_identity_l531_531298

theorem simplify_trig_identity (α : ℝ) :
  (Real.cos (Real.pi / 3 + α) + Real.sin (Real.pi / 6 + α)) = Real.cos α :=
by
  sorry

end simplify_trig_identity_l531_531298


namespace range_of_omega_l531_531572

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ (a b : ℝ), a ≠ b ∧ 0 ≤ a ∧ a ≤ π/2 ∧ 0 ≤ b ∧ b ≤ π/2 ∧ f ω a + f ω b = 4) ↔ 5 ≤ ω ∧ ω < 9 :=
sorry

end range_of_omega_l531_531572


namespace find_difference_condition_l531_531640

variable (a b c : ℝ)

theorem find_difference_condition (h1 : (a + b) / 2 = 40) (h2 : (b + c) / 2 = 60) : c - a = 40 := by
  sorry

end find_difference_condition_l531_531640


namespace number_of_dwarves_is_12_l531_531220

variable (n m k : ℕ)  -- Variables for number of elves, fairies, and dwarves respectively.

-- Conditions as constraints.
axiom elves_friends : ∀ e : ℕ, e < n → friends_with_fairies e (m - 3)
axiom fairies_friends : ∀ f : ℕ, f < m → friends_with_elves f (2 * (m - 3))
axiom elves_dwarves : ∀ e : ℕ, e < n → friends_with_dwarves e 3
axiom fairies_dwarves : ∀ f : ℕ, f < m → friends_with_all_dwarves f
axiom dwarves_friends : ∀ d : ℕ, d < k → friends_with_half_elf_fairy d ((n + m) / 2)

-- Prove that the number of dwarves is 12.
theorem number_of_dwarves_is_12 : k = 12 :=
by {
  sorry
}

end number_of_dwarves_is_12_l531_531220


namespace probability_differ_by_three_is_one_sixth_l531_531464

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l531_531464


namespace ramu_spent_on_repairs_l531_531280

def profit_percent (purchase_price sell_price repairs : ℝ) : ℝ :=
  ((sell_price - (purchase_price + repairs)) / (purchase_price + repairs)) * 100

theorem ramu_spent_on_repairs :
  ∃ (repairs : ℝ), 
  let purchase_price := 42000
  let sell_price := 64900
  let expected_profit_percent := 20.185185185185187
  profit_percent purchase_price sell_price repairs = expected_profit_percent ∧
  repairs = 11990 :=
by
  sorry

end ramu_spent_on_repairs_l531_531280


namespace wire_length_given_area_equivalence_l531_531059

noncomputable def L : ℝ := 49
def s : ℝ := 7
def L1 := (4 / 7) * L
def L2 := (3 / 7) * L

def area_square (s : ℝ) := s^2
def area_octagon (L2 : ℝ) := 2 * (1 + Real.sqrt 2) * (L2 / 8)^2

theorem wire_length_given_area_equivalence :
  (4 / 7) * L = 4 * s ∧ s = 7 ∧
  area_square s = area_octagon L2 :=
by
  sorry

end wire_length_given_area_equivalence_l531_531059


namespace ephraim_necklaces_production_total_l531_531924

def production_sunday (machine1 machine2 machine3 machine4: ℕ) :=
  machine1 = 45 ∧
  machine2 = nat.floor(45 * 2.4) ∧
  machine3 = nat.floor((45 + nat.floor(45 * 2.4)) * 1.5) ∧
  machine4 = nat.floor(nat.floor((45 + nat.floor(45 * 2.4)) * 1.5) * 0.8)

def production_monday (machine1_s machine2_s machine3_s machine4_s machine1 machine2 machine3 machine4: ℕ) :=
  machine1 = nat.floor(machine1_s * 1.3) ∧
  machine2 = nat.floor(machine2_s / 2) ∧
  machine3 = machine3_s ∧
  machine4 = machine4_s * 2

def production_tuesday (machine1_m machine2_s machine3_s machine4_s machine1 machine2 machine3 machine4: ℕ) :=
  machine1 = machine1_m ∧
  machine2 = machine2_s ∧
  machine3 = nat.floor(machine3_s * 1.2) ∧
  machine4 = nat.floor(machine4_s * 1.2)

def total_production (machine1_s machine2_s machine3_s machine4_s machine1_m machine2_m machine3_m machine4_m machine1_t machine2_t machine3_t machine4_t: ℕ) :=
  machine1_s + machine2_s + machine3_s + machine4_s +
  machine1_m + machine2_m + machine3_m + machine4_m +
  machine1_t + machine2_t + machine3_t + machine4_t

theorem ephraim_necklaces_production_total :
  ∃ (machine1_s machine2_s machine3_s machine4_s machine1_m machine2_m machine3_m machine4_m machine1_t machine2_t machine3_t machine4_t : ℕ),
    production_sunday machine1_s machine2_s machine3_s machine4_s ∧
    production_monday machine1_s machine2_s machine3_s machine4_s machine1_m machine2_m machine3_m machine4_m ∧
    production_tuesday machine1_m machine2_s machine3_s machine4_s machine1_t machine2_t machine3_t machine4_t ∧
    total_production machine1_s machine2_s machine3_s machine4_s machine1_m machine2_m machine3_m machine4_m machine1_t machine2_t machine3_t machine4_t = 1942 :=
by
  sorry

end ephraim_necklaces_production_total_l531_531924


namespace conditional_probability_B_given_A_l531_531034

-- Definitions of events in the problem statement
def event_A := {s : Finset (Fin 3) // s.card > 0} -- At least one occurrence of tails
def event_B := {s : Finset (Fin 3) // s.filter (λ x => x = 0).card = 1} -- Exactly one occurrence of heads

-- Conditional probability calculation
theorem conditional_probability_B_given_A :
  (|event_B ∩ event_A| : ℝ) / |event_A| = 3 / 7 := sorry

end conditional_probability_B_given_A_l531_531034


namespace quiz_score_difference_l531_531270

theorem quiz_score_difference :
  let percentage_70 := 0.10
  let percentage_80 := 0.35
  let percentage_90 := 0.30
  let percentage_100 := 0.25
  let mean_score := (percentage_70 * 70) + (percentage_80 * 80) + (percentage_90 * 90) + (percentage_100 * 100)
  let median_score := 90
  mean_score = 87 → median_score - mean_score = 3 :=
by
  sorry

end quiz_score_difference_l531_531270


namespace dice_diff_by_three_probability_l531_531439

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l531_531439


namespace product_modulo_7_l531_531086

theorem product_modulo_7 : 
  (2007 % 7 = 4) ∧ (2008 % 7 = 5) ∧ (2009 % 7 = 6) ∧ (2010 % 7 = 0) →
  (2007 * 2008 * 2009 * 2010) % 7 = 0 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end product_modulo_7_l531_531086


namespace max_regions_with_6_chords_l531_531534

-- Definition stating the number of regions created by k chords
def regions_by_chords (k : ℕ) : ℕ :=
  1 + (k * (k + 1)) / 2

-- Lean statement for the proof problem
theorem max_regions_with_6_chords : regions_by_chords 6 = 22 :=
  by sorry

end max_regions_with_6_chords_l531_531534


namespace probability_differ_by_three_is_one_sixth_l531_531461

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l531_531461


namespace probability_of_difference_three_l531_531487

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l531_531487


namespace eggs_per_box_l531_531026

-- Conditions
def num_eggs : ℝ := 3.0
def num_boxes : ℝ := 2.0

-- Theorem statement
theorem eggs_per_box (h1 : num_eggs = 3.0) (h2 : num_boxes = 2.0) : (num_eggs / num_boxes = 1.5) :=
sorry

end eggs_per_box_l531_531026


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531449

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531449


namespace smallest_four_digit_number_with_given_properties_l531_531828

-- Define predicates for each condition
def different_digits (n: Nat) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  List.Nodup digits

def no_repeated_digits (n: Nat) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  ∀ x ∈ digits, List.count digits x = 1

def has_digit_5 (n: Nat) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  5 ∈ digits

def divisible_by_each_digit (n: Nat) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  ∀ d ∈ digits, d ≠ 0 → n % d = 0

-- Define the final theorem statement
theorem smallest_four_digit_number_with_given_properties : 
  ∃ n : Nat, n >= 1000 ∧ n < 10000 ∧
             different_digits n ∧
             no_repeated_digits n ∧
             has_digit_5 n ∧
             divisible_by_each_digit n ∧
             n = 1524 :=
sorry

end smallest_four_digit_number_with_given_properties_l531_531828


namespace problem_proof_l531_531596

noncomputable def f : ℝ → ℝ := sorry

theorem problem_proof (h1 : ∀ x : ℝ, f (-x) = f x)
    (h2 : ∀ x y : ℝ, x < y ∧ y ≤ -1 → f x < f y) : 
    f 2 < f (-3 / 2) ∧ f (-3 / 2) < f (-1) :=
by
  sorry

end problem_proof_l531_531596


namespace solve_trig_eqn_l531_531299

theorem solve_trig_eqn (x : ℝ) : (tan (6 * x) * cos (2 * x) - sin (2 * x) - 2 * sin (4 * x) = 0) 
    ∧ (cos (6 * x) ≠ 0) ↔ 
    (∃ (k : ℤ), x = (k * Real.pi) / 4 ∧ (6 * k * Real.pi / 4) % Real.pi ≠ Real.pi / 2) 
    ∨ (∃ (n : ℤ), x = (n * Real.pi) / 3 + Real.pi / 18 ∨ x = (n * Real.pi) / 3 - Real.pi / 18) :=
sorry

end solve_trig_eqn_l531_531299


namespace chemical_x_percentage_l531_531637

-- Define the initial volume of the mixture
def initial_volume : ℕ := 80

-- Define the percentage of chemical x in the initial mixture
def percentage_x_initial : ℚ := 0.30

-- Define the volume of chemical x added to the mixture
def added_volume_x : ℕ := 20

-- Define the calculation of the amount of chemical x in the initial mixture
def initial_amount_x : ℚ := percentage_x_initial * initial_volume

-- Define the calculation of the total amount of chemical x after adding more
def total_amount_x : ℚ := initial_amount_x + added_volume_x

-- Define the calculation of the total volume after adding 20 liters of chemical x
def total_volume : ℚ := initial_volume + added_volume_x

-- Define the percentage of chemical x in the final mixture
def percentage_x_final : ℚ := (total_amount_x / total_volume) * 100

-- The proof goal
theorem chemical_x_percentage : percentage_x_final = 44 := 
by
  sorry

end chemical_x_percentage_l531_531637


namespace sasha_remainder_l531_531759

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l531_531759


namespace point_N_inside_triangle_ABM_l531_531696

structure Quadrilateral (A B C D : Type) :=
  (BC_parallel_AD : B ≠ D → C ≠ A → ∃ M, ∃ P, ∃ Q, ∃ N,
    M ≠ (CD:Type) → 
    P ≠ (MA:Type) →
    Q ≠ (MB:Type) → 
    ∃ (DP_line : (D P : Type) → N ∈ DP_line) 
      (CQ_line : (C Q : Type) → N ∈ CQ_line),
    N ∈ triangle AB M )
 
theorem point_N_inside_triangle_ABM :
  ∀ (A B C D : Type), Quadrilateral A B C D → 
  ∃ (N:Type), 
  (BC_parallel_AD A B C D ⟨N⟩ (triangle AB M) → 
    N ∈ triangle AB M) :=
by
  sorry

end point_N_inside_triangle_ABM_l531_531696


namespace mult_mod_7_zero_l531_531087

theorem mult_mod_7_zero :
  (2007 ≡ 5 [MOD 7]) →
  (2008 ≡ 6 [MOD 7]) →
  (2009 ≡ 0 [MOD 7]) →
  (2010 ≡ 1 [MOD 7]) →
  (2007 * 2008 * 2009 * 2010 ≡ 0 [MOD 7]) :=
by
  intros h1 h2 h3 h4
  sorry

end mult_mod_7_zero_l531_531087


namespace tempo_original_value_l531_531417

noncomputable def original_value (V : ℝ) : Prop :=
  let I := (5/7) * V in
  let P := (3/100) * I in
  P = 300 → V = 14000

theorem tempo_original_value (V : ℝ) : original_value V :=
by
  -- Proof to be provided later
  sorry

end tempo_original_value_l531_531417


namespace probability_diff_by_three_l531_531469

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l531_531469


namespace cos_subtract_double_alpha_equals_neg_five_div_nine_l531_531633

variable (α : ℝ)

-- Given condition: cos(π/2 - α) = √2/3
axiom cos_half_pi_sub_alpha_eq_sqrt2_div3 : cos (π / 2 - α) = (√2) / 3

-- Theorem to be proven: cos(π - 2α) = -5/9
theorem cos_subtract_double_alpha_equals_neg_five_div_nine (α : ℝ) : 
  cos (π - 2 * α) = - (5 / 9) :=
sorry

end cos_subtract_double_alpha_equals_neg_five_div_nine_l531_531633


namespace angle_bisector_length_l531_531318

open Real -- Use Real numbers and operations

theorem angle_bisector_length (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  ∃ x, x = (a * b * sqrt 2) / (a + b) :=
by
  use (a * b * sqrt 2) / (a + b)
  sorry

end angle_bisector_length_l531_531318


namespace min_S_in_grid_l531_531648

def valid_grid (grid : Fin 10 × Fin 10 → Fin 100) (S : ℕ) : Prop :=
  ∀ i j, 
    (i < 9 → grid (i, j) + grid (i + 1, j) ≤ S) ∧
    (j < 9 → grid (i, j) + grid (i, j + 1) ≤ S)

theorem min_S_in_grid : ∃ grid : Fin 10 × Fin 10 → Fin 100, ∃ S : ℕ, valid_grid grid S ∧ 
  (∀ (other_S : ℕ), valid_grid grid other_S → S ≤ other_S) ∧ S = 106 :=
sorry

end min_S_in_grid_l531_531648


namespace trig_identity_eval_l531_531003

theorem trig_identity_eval (h1 : cos 67 * cos 7 - cos 83 * cos 23 = sin 16)
                           (h2 : cos (128 - 90) * cos 68 - cos 38 * cos 22 = -1/2) :
    (cos 67 * cos 7 - cos 83 * cos 23) / (cos 128 * cos 68 - cos 38 * cos 22) - tan 164 = 0 :=
by
  sorry

end trig_identity_eval_l531_531003


namespace pies_sold_each_day_l531_531870

theorem pies_sold_each_day (total_pies : ℕ) (days_in_week : ℕ) (h1 : total_pies = 56) (h2 : days_in_week = 7) :
  (total_pies / days_in_week = 8) :=
by
exact sorry

end pies_sold_each_day_l531_531870


namespace cost_of_bananas_l531_531346

theorem cost_of_bananas (A B : ℝ) (n : ℝ) (Tcost: ℝ) (Acost: ℝ): 
  (A * n + B = Tcost) → (A * (1 / 2 * n) + B = Acost) → (Tcost = 7) → (Acost = 5) → B = 3 :=
by
  intros hTony hArnold hTcost hAcost
  sorry

end cost_of_bananas_l531_531346


namespace binomial_510_510_l531_531904

theorem binomial_510_510 : binomial 510 510 = 1 :=
by sorry

end binomial_510_510_l531_531904


namespace probability_of_even_sum_l531_531820

noncomputable def probability_even_sum : ℚ := sorry

theorem probability_of_even_sum :
  let S := set.range 1 101 in
  let evens := {n ∣ n ∈ S ∧ n % 2 = 0} in
  let odds := {n ∣ n ∈ S ∧ n % 2 = 1} in
  ∑ x y z in S, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x + y + z) % 2 = 0 
    / ∑ x y z in S, x ≠ y ∧ y ≠ z ∧ x ≠ z = 0.5 :=
sorry

end probability_of_even_sum_l531_531820


namespace total_shells_correct_l531_531020

def morning_shells : ℕ := 292
def afternoon_shells : ℕ := 324

theorem total_shells_correct : morning_shells + afternoon_shells = 616 := by
  sorry

end total_shells_correct_l531_531020


namespace centroid_line_area_ratio_centroid_line_parallel_side_area_ratio_l531_531705

variable {A B C S : Type} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry S]

theorem centroid_line_area_ratio (A B C S : Point)
  (hS_centroid : is_centroid A B C S)
  (l : Line)
  (h_line_passing_through_S : passes_through S l) :
  4/5 ≤ area_ratio_divided_by_line A B C S l ∧ area_ratio_divided_by_line A B C S l ≤ 5/4 :=
  sorry

theorem centroid_line_parallel_side_area_ratio (A B C S : Point)
  (hS_centroid : is_centroid A B C S)
  (l : Line)
  (h_line_passing_through_S : passes_through S l)
  (h_line_parallel_to_side : is_parallel l (line_through A B) ∨ is_parallel l (line_through B C) ∨ is_parallel l (line_through C A)) :
  area_ratio_divided_by_line A B C S l = 4/5 :=
  sorry

end centroid_line_area_ratio_centroid_line_parallel_side_area_ratio_l531_531705


namespace unique_rectangle_property_l531_531064

-- Define the properties of rectangles and rhombuses
structure Rectangle where
  opposite_sides_parallel : ∀ (a b : ℝ), a = b
  all_right_angles : true
  diagonals_bisect_and_equal : true

structure Rhombus where
  all_sides_equal : ∀ (a b : ℝ), a = b
  opposite_angles_equal : true
  diagonals_bisect_and_perpendicular : true

theorem unique_rectangle_property (R : Rectangle) (Rh : Rhombus) :
  R.diagonals_bisect_and_equal ∧ ¬ Rh.diagonals_bisect_and_equal :=
sorry

end unique_rectangle_property_l531_531064


namespace first_and_third_non_integer_chord_lengths_l531_531097

def is_integer_line_segment (A B : ℝ × ℝ) : Prop :=
  ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2).sqrt ∈ set_of (λ x, ∃ (n : ℕ), x = n)

def is_non_integer_chord (A B A' B' : ℝ × ℝ) : Prop :=
  let SAB := (A.1 * B.2 - A.2 * B.1) / 2
  let SA'B' := (A'.1 * B'.2 - A'.2 * B'.1) / 2
  in (((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2).sqrt = ((A'.1 - B'.1) ^ 2 + (A'.2 - B'.2) ^ 2).sqrt) 
  ∧ (SAB ≠ SA'B') 
  ∧ is_integer_line_segment A B 
  ∧ is_integer_line_segment A' B'

theorem first_and_third_non_integer_chord_lengths : 
  ∃ A B A' B' AA' BA' O, 
    let lengths := [5, 10, 13, 15, 20].sort (≤) in 
    is_non_integer_chord A B A' B' 
    ∧ is_non_integer_chord AA' BA' O
    ∧ lengths.head = 5 
    ∧ lengths.tail.tail.head = 13 :=
by sorry

end first_and_third_non_integer_chord_lengths_l531_531097


namespace find_common_difference_l531_531663

variable {a : ℕ → ℤ}  -- Define a sequence indexed by natural numbers, returning integers
variable (d : ℤ)  -- Define the common difference as an integer

-- The conditions: sequence is arithmetic, a_2 = 14, a_5 = 5
axiom arithmetic_sequence (n : ℕ) : a n = a 0 + n * d
axiom a_2_eq_14 : a 2 = 14
axiom a_5_eq_5 : a 5 = 5

-- The proof statement
theorem find_common_difference : d = -3 :=
by sorry

end find_common_difference_l531_531663


namespace number_of_factors_of_x2y3z4_l531_531341

-- Given three distinct natural numbers each having exactly three natural-number factors
variable (x y z : ℕ)

-- Condition that each of these numbers has exactly three natural-number factors
def has_exactly_three_factors (n : ℕ) : Prop :=
  ∃ (p : ℕ), Nat.Prime p ∧ n = p^2

theorem number_of_factors_of_x2y3z4 
  (h₁ : has_exactly_three_factors x)
  (h₂ : has_exactly_three_factors y)
  (h₃ : has_exactly_three_factors z)
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) :
  Nat.factors (x^2 * y^3 * z^4).length = 315 :=
sorry

end number_of_factors_of_x2y3z4_l531_531341


namespace local_maximum_at_negative_one_l531_531166

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * a * x + 2

theorem local_maximum_at_negative_one
  (a : ℝ)
  (h_min : ∀ f' : ℝ → ℝ, (f' = λ x, 3 * x^2 - 3 * a) → f' 1 = 0) :
  f a (-1) = 4 :=
by
  -- Definitions and hypotheses are in place; proof is omitted.
  sorry

end local_maximum_at_negative_one_l531_531166


namespace triangle_area_approx_l531_531575

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let C := Real.arccos cos_C
  let sin_C := Real.sin C
  (1 / 2) * a * b * sin_C

theorem triangle_area_approx : 
  triangle_area 27 31 15 ≈ 204.5 := 
by sorry

end triangle_area_approx_l531_531575


namespace construct_triangle_l531_531917

variables {l : line} {A1 B1 : point}

-- Conditions
def line_l (l : line) (A B : point) : Prop := A ∈ l ∧ B ∈ l
def feet_of_altitudes (A1 B1 : point) (BC AC : line) : Prop :=
  ∃ A B C : point, foot_of_altitude A BC = A1 ∧ foot_of_altitude B AC = B1

-- Proven statement
theorem construct_triangle (A B C : point) :
  (line_l l A B) ∧ (feet_of_altitudes A1 B1 (line_through B C) (line_through A C)) →
  ∃ A B C : point, triangle A B C :=
sorry

end construct_triangle_l531_531917


namespace smallest_share_arith_seq_l531_531302

theorem smallest_share_arith_seq (a1 d : ℚ) (h1 : 5 * a1 + 10 * d = 100) (h2 : (3 * a1 + 9 * d) * (1 / 7) = 2 * a1 + d) : a1 = 5 / 3 :=
by
  sorry

end smallest_share_arith_seq_l531_531302


namespace pies_sold_each_day_l531_531869

theorem pies_sold_each_day (total_pies: ℕ) (days_in_week: ℕ) 
  (h1: total_pies = 56) (h2: days_in_week = 7) : 
  total_pies / days_in_week = 8 :=
by
  sorry

end pies_sold_each_day_l531_531869


namespace number_of_women_l531_531891

-- Defining the conditions
def total_employees (E : ℕ) : Prop := 0.25 * (0.40 * E) = 8

-- Statement of the problem
theorem number_of_women (E : ℕ) (h : total_employees E) : 0.60 * E = 48 :=
by {
  sorry
}

end number_of_women_l531_531891


namespace arithmetic_geometric_l531_531590

noncomputable theory

def a : ℕ → ℤ
def b : ℕ → ℕ
def T : ℕ → ℤ

constants (a_1 d : ℤ)
constants (b_1 : ℕ)

-- Define the general form of the arithmetic sequence a_n
def arithmetic_sequence(a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + d * (n - 1)

-- Define the general form of the geometric sequence b_n
def geometric_sequence(b_1 : ℕ) (n : ℕ) : ℕ := b_1 * 2^(n - 1)


axiom a₂ : arithmetic_sequence a_1 d 2 = 4
axiom S_b5 : 62 = b_1 * (1 - 2^5) / (1 - 2)
axiom sum_a₂a₅ : 17 = arithmetic_sequence a_1 d 2 + arithmetic_sequence a_1 d 5
axiom product_a₃a₄ : 70 = arithmetic_sequence a_1 d 3 * arithmetic_sequence a_1 d 4

axiom general_arithmetic : a = λ n : ℕ, 3 * n - 2
axiom general_geometric : b = λ n : ℕ, 2 ^ n

-- Define the sequence a_{2n}b_{2n-1}
noncomputable def seq_a2nb2n1 (n : ℕ) : ℤ :=
  let a_2n := 3 * 2 * n - 2
  let b_2n_1 := 2^(2 * n - 1)
  a_2n * b_2n_1

axiom general_sum_T : T = λ n : ℕ, ((3 * n - 2) / 3) * 4^(n + 1) + 8 / 3

-- The theorem we want to prove
theorem arithmetic_geometric (a_1 d : ℤ) (b_1 : ℕ) :
  a = λ n : ℕ, 3 * n - 2 ∧
  b = λ n : ℕ, 2 ^ n ∧
  T = λ n : ℕ, ((3 * n - 2) / 3) * 4^(n + 1) + 8 / 3 :=
sorry

end arithmetic_geometric_l531_531590


namespace final_value_of_A_l531_531060

theorem final_value_of_A (A : ℤ) (h₁ : A = 15) (h₂ : A = -A + 5) : A = -10 := 
by 
  sorry

end final_value_of_A_l531_531060


namespace total_canoes_built_by_end_of_june_l531_531515

noncomputable def g (n : Nat) : Nat :=
  match n with
  | 0 => 7
  | k + 1 => 3 * g k

def total_canoes (end_month : Nat) : Nat :=
  (List.range (end_month + 1)).foldr (fun n acc => g n + acc) 0

theorem total_canoes_built_by_end_of_june : total_canoes 5 = 2548 := 
by
  -- Proof can be added here
  sorry

end total_canoes_built_by_end_of_june_l531_531515


namespace sin_cos_fraction_identity_l531_531197

variables {θ a b : ℝ}

theorem sin_cos_fraction_identity (h : (sin θ)^6 / a + (cos θ)^6 / b = 1 / (a + b)) : 
  (sin θ)^12 / a^5 + (cos θ)^12 / b^5 = 1 / (a + b)^5 :=
by {
  sorry
}

end sin_cos_fraction_identity_l531_531197


namespace binomial_510_510_l531_531906

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem binomial_510_510 : binomial 510 510 = 1 :=
  by
    -- Skip the proof with sorry
    sorry

end binomial_510_510_l531_531906


namespace arithmetic_sequence_a3_eq_four_l531_531219

noncomputable theory

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

-- Given conditions
variable (a : ℕ → ℝ)
variable (h1 : is_arithmetic_sequence a)
variable (h2 : a 0 + a 1 + a 2 + a 3 + a 4 = 20)

theorem arithmetic_sequence_a3_eq_four :
  a 2 = 4 :=
sorry

end arithmetic_sequence_a3_eq_four_l531_531219


namespace min_sum_of_labels_on_9x9_chessboard_l531_531907

-- Define the problem conditions and the proof statement in Lean 4
theorem min_sum_of_labels_on_9x9_chessboard :
  (∃ (r : Fin 9 → Fin 9), 
     (∀ i j, i ≠ j → r i ≠ r j) ∧ 
     (∑ i, 1 / ((r i).val + i.val + 2)) = (27 / 29)) := 
sorry

end min_sum_of_labels_on_9x9_chessboard_l531_531907


namespace probability_differ_by_three_is_one_sixth_l531_531466

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l531_531466


namespace probability_diff_by_three_l531_531470

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l531_531470


namespace negation_of_exists_statement_l531_531323

theorem negation_of_exists_statement :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by
  sorry

end negation_of_exists_statement_l531_531323


namespace imaginary_part_of_i_mul_root_l531_531203

theorem imaginary_part_of_i_mul_root
  (z : ℂ) (hz : z^2 - 4 * z + 5 = 0) : (i * z).im = 2 := 
sorry

end imaginary_part_of_i_mul_root_l531_531203


namespace lateral_surface_area_cut_off_l531_531046

theorem lateral_surface_area_cut_off {a b c d : ℝ} (h₁ : a = 4) (h₂ : b = 25) 
(h₃ : c = (2/5 : ℝ)) (h₄ : d = 2 * (4 / 25) * b) : 
4 + 10 + (1/4 * b) = 20.25 :=
by
  sorry

end lateral_surface_area_cut_off_l531_531046


namespace probability_of_differ_by_three_l531_531497

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l531_531497


namespace range_of_f_min_distance_f_smallest_period_l531_531586

-- Part I
theorem range_of_f (α : ℝ) (hα : α = 1) (ϕ : ℝ → ℝ) (hϕ : ϕ = λ x, 3^x - 1) : 
  Set.Icc (-1/3 : ℝ) (⊤ : ℝ) = { y | ∃ x, y = ϕ(x) * ϕ(x + α) } :=
sorry

-- Part II
theorem min_distance_f (α : ℝ) (hα : α = π / 2) (ϕ : ℝ → ℝ) (hϕ : ϕ = cos) :
  (∀ x : ℝ, ∃ x1 x2 : ℝ, ϕ(x1) * ϕ(x1 + α) ≤ ϕ(x) * ϕ(x + α) ∧ ϕ(x) * ϕ(x + α) ≤ ϕ(x2) * ϕ(x2 + α)) →
  Inf { y : ℝ | ∃ x1 x2 : ℝ, y = |x1 - x2| } = π / 2 :=
sorry

-- Part III
theorem smallest_period (A ω : ℝ) (hA : A > 0) (hω : ω > 0) (ϕ : ℝ → ℝ) (hϕ : ϕ = λ x, A * sin (ω * x)) :
  Inf { p : ℝ | p > 0 ∧ ∀ x, ϕ (x + p) = ϕ x } = π / ω :=
sorry

end range_of_f_min_distance_f_smallest_period_l531_531586


namespace number_of_pickup_trucks_l531_531948

theorem number_of_pickup_trucks 
  (cars : ℕ) (bicycles : ℕ) (tricycles : ℕ) (total_tires : ℕ)
  (tires_per_car : ℕ) (tires_per_bicycle : ℕ) (tires_per_tricycle : ℕ) (tires_per_pickup : ℕ) :
  cars = 15 →
  bicycles = 3 →
  tricycles = 1 →
  total_tires = 101 →
  tires_per_car = 4 →
  tires_per_bicycle = 2 →
  tires_per_tricycle = 3 →
  tires_per_pickup = 4 →
  ((total_tires - (cars * tires_per_car + bicycles * tires_per_bicycle + tricycles * tires_per_tricycle)) / tires_per_pickup) = 8 :=
by
  sorry

end number_of_pickup_trucks_l531_531948


namespace cost_price_of_apple_l531_531370

variable (CP SP: ℝ)
variable (loss: ℝ)
variable (h1: SP = 18)
variable (h2: loss = CP / 6)
variable (h3: SP = CP - loss)

theorem cost_price_of_apple : CP = 21.6 :=
by
  sorry

end cost_price_of_apple_l531_531370


namespace range_of_a_l531_531584

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2^x - 4 else 2^(-x) - 4

theorem range_of_a (a : ℝ) : f(a - 2) > 0 ↔ a ∈ set.Iio 0 ∪ set.Ioi 4 :=
by {
  sorry
}

end range_of_a_l531_531584


namespace probability_diff_by_three_l531_531472

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l531_531472


namespace samantha_eggs_left_l531_531714

variables (initial_eggs : ℕ) (total_cost price_per_egg : ℝ)

-- Conditions
def samantha_initial_eggs : initial_eggs = 30 := sorry
def samantha_total_cost : total_cost = 5 := sorry
def samantha_price_per_egg : price_per_egg = 0.20 := sorry

-- Theorem to prove:
theorem samantha_eggs_left : 
  initial_eggs - (total_cost / price_per_egg) = 5 := 
  by
  rw [samantha_initial_eggs, samantha_total_cost, samantha_price_per_egg]
  -- Completing the arithmetic proof
  rw [Nat.cast_sub (by norm_num), Nat.cast_div (by norm_num), Nat.cast_mul (by norm_num)]
  norm_num
  sorry

end samantha_eggs_left_l531_531714


namespace arrangements_eq_2_pow_l531_531071

def a : ℕ → ℕ 
| 1 := 1
| (n + 1) := 2 * a n

theorem arrangements_eq_2_pow (n : ℕ) : a (n + 1) = 2^n := by
  induction n with
  | zero => 
    simp [a]
  | succ n ih =>
    simp [a, ih]
    sorry

end arrangements_eq_2_pow_l531_531071


namespace find_m_l531_531251

def fractional_part (x : ℝ) := x - real.floor x

def g (x : ℝ) := abs (3 * fractional_part x - 1.5)

theorem find_m : ∃ m : ℕ, (∀ x : ℝ, (mgx : ℝ := m * g (x * g x), mgx = x)) → m = 19 :=
by
  sorry

end find_m_l531_531251


namespace identify_heaviest_and_lightest_13_weighings_l531_531815

theorem identify_heaviest_and_lightest_13_weighings (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ f : (Fin 13 → ((Fin 10) × (Fin 10) × ℝ)), true :=
by
  sorry

end identify_heaviest_and_lightest_13_weighings_l531_531815


namespace find_a4_l531_531330

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1) / 2) * d

theorem find_a4 (a d : ℤ)
    (h₁ : sum_first_n_terms a d 5 = 15)
    (h₂ : sum_first_n_terms a d 9 = 63) :
  arithmetic_sequence a d 4 = 5 :=
sorry

end find_a4_l531_531330


namespace probability_of_both_red_is_one_sixth_l531_531028

noncomputable def probability_both_red (red blue green : ℕ) (balls_picked : ℕ) : ℚ :=
  if balls_picked = 2 ∧ red = 4 ∧ blue = 3 ∧ green = 2 then (4 / 9) * (3 / 8) else 0

theorem probability_of_both_red_is_one_sixth :
  probability_both_red 4 3 2 2 = 1 / 6 :=
by
  unfold probability_both_red
  split_ifs
  · sorry
  · contradiction

end probability_of_both_red_is_one_sixth_l531_531028


namespace problem_xyz_inequality_l531_531269

theorem problem_xyz_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h_eq : x^2 + y^2 + z^2 + x * y * z = 4) :
  x * y * z ≤ x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ x * y * z + 2 :=
by 
  sorry

end problem_xyz_inequality_l531_531269


namespace probability_diff_by_3_l531_531485

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l531_531485


namespace sasha_remainder_20_l531_531776

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l531_531776


namespace identify_heaviest_and_lightest_coin_l531_531810

theorem identify_heaviest_and_lightest_coin :
  ∀ (coins : Fin 10 → ℕ), 
  (∀ i j, i ≠ j → coins i ≠ coins j) → 
  ∃ (seq : List (Fin 10 × Fin 10)), 
  seq.length = 13 ∧ 
  (∀ (i j : Fin 10), (i, j) ∈ seq → 
    (coins i < coins j ∨ coins i > coins j)) ∧ 
  (∃ (heaviest lightest : Fin 10),
    (∀ coin, coins coin ≤ coins heaviest) ∧ (∀ coin, coins coin ≥ coins lightest)) :=
by
  intros coins h_coins
  exists [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), -- initial pairs
          (0, 2), (2, 4), (4, 6), (6, 8),         -- heaviest coin comparisons
          (1, 3), (3, 5), (5, 7), (7, 9)]         -- lightest coin comparisons
  constructor
  . -- length check
    rfl
  . constructor
    . -- all comparisons
      intros i j h_pair
      cases h_pair; simp; solve_by_elim
    . -- finding heaviest and lightest coins
      exists 8, 9
      constructor
      . -- all coins are less than or equal to the heaviest
        sorry
      . -- all coins are greater than or equal to the lightest
        sorry

end identify_heaviest_and_lightest_coin_l531_531810


namespace sum_inequality_l531_531944

theorem sum_inequality (a b : ℕ → ℝ) (n : ℕ) (hapos : ∀ k : ℕ, k < n → 0 < a k) (hbpos : ∀ k : ℕ, k < n → 0 < b k) :
  let A := ∑ k in Finset.range n, a k
  let B := ∑ k in Finset.range n, b k
  (∑ k in Finset.range n, (a k * b k) / (a k + b k)) ≤ (A * B) / (A + B) :=
by
  sorry

end sum_inequality_l531_531944


namespace find_matrix_A_l531_531545

open Matrix

variables (A : Matrix (Fin 2) (Fin 2) ℝ) 

def cond1 : Prop := (A.mul_vec ![4, 0]) = ![12, 0]
def cond2 : Prop := (A.mul_vec ![2, -3]) = ![6, 9]

theorem find_matrix_A (h1 : cond1 A) (h2 : cond2 A) :
  A = ![![3, 0], ![0, -3]] :=
sorry

end find_matrix_A_l531_531545


namespace ground_beef_lean_beef_difference_l531_531114

theorem ground_beef_lean_beef_difference (x y z : ℕ) 
  (h1 : x + y + z = 20) 
  (h2 : y + 2 * z = 18) :
  x - z = 2 :=
sorry

end ground_beef_lean_beef_difference_l531_531114


namespace working_days_in_specific_month_l531_531542

axiom months : Type
axiom days : Type

-- Statements for days in a month
axiom num_days_in_month : months → ℕ
axiom starts_on_saturday : months → Prop

-- Definitions of holidays and working days
def is_saturday (d : ℕ) : Prop := d % 7 = 0
def is_sunday (d : ℕ) : Prop := d % 7 = 1

-- Every second Saturday is a holiday
def is_holiday (d : ℕ) : Prop :=
  is_saturday d ∧ (d / 7) % 2 = 1

-- Sundays are not considered holidays by given condition
def is_working_day (d : ℕ) : Prop :=
  ¬is_holiday d ∧ ¬is_sunday d

-- Number of working days in a given month
noncomputable def num_working_days (m : months) : ℕ :=
  if starts_on_saturday m then
    (finset.range (num_days_in_month m)).filter is_working_day |>.card
  else
    0

-- Specific month in question
axiom specific_month : months
axiom specific_month_conditions :
  num_days_in_month specific_month = 30 ∧ starts_on_saturday specific_month

-- Theorem statement
theorem working_days_in_specific_month : num_working_days specific_month = 28 :=
by
  have cond := specific_month_conditions
  sorry

end working_days_in_specific_month_l531_531542


namespace number_of_yellow_crayons_l531_531794

theorem number_of_yellow_crayons : 
  ∃ (R B Y : ℕ), 
  R = 14 ∧ 
  B = R + 5 ∧ 
  Y = 2 * B - 6 ∧ 
  Y = 32 :=
by
  sorry

end number_of_yellow_crayons_l531_531794


namespace sqrt_sin_cos_not_both_rational_l531_531520

theorem sqrt_sin_cos_not_both_rational (theta : ℝ) (h1 : 0 < theta) (h2 : theta < real.pi / 2) :
  ¬(∃ a b c d : ℤ, b ≠ 0 ∧ d ≠ 0 ∧ (real.sqrt (real.sin theta) = a / b) ∧ (real.sqrt (real.cos theta) = c / d)) := 
sorry

end sqrt_sin_cos_not_both_rational_l531_531520


namespace find_speed_of_goods_train_l531_531006

noncomputable def speed_of_goods_train (v_man : ℝ) (t_pass : ℝ) (d_goods : ℝ) : ℝ := 
  let v_man_mps := v_man * (1000 / 3600)
  let v_relative := d_goods / t_pass
  let v_goods_mps := v_relative - v_man_mps
  v_goods_mps * (3600 / 1000)

theorem find_speed_of_goods_train :
  speed_of_goods_train 45 8 340 = 108 :=
by sorry

end find_speed_of_goods_train_l531_531006


namespace find_k_l531_531116

theorem find_k (a b c k : ℝ) 
  (h : ∀ x : ℝ, 
    (a * x^2 + b * x + c + b * x^2 + a * x - 7 + k * x^2 + c * x + 3) / (x^2 - 2 * x - 5) = (x^2 - 2*x - 5)) :
  k = 2 :=
by
  sorry

end find_k_l531_531116


namespace newtons_method_coincide_with_convergents_l531_531380

noncomputable def alpha (p q : ℝ) : ℝ :=
p - real.recurring (λ x, q / (p - x))

noncomputable def beta (p q : ℝ) : ℝ :=
q / (p - real.recurring (λ x, q / (p - x)))

def newtons_method (p q : ℝ) (x_n : ℕ → ℝ) : ℕ → ℝ
| 0     := x_n 0
| (n+1) := x_n n - (x_n n * x_n n - p * x_n n + q) / (2 * x_n n - p)

theorem newtons_method_coincide_with_convergents (p q : ℝ) (x_0 : ℝ) (x_n : ℕ → ℝ) :
  (x_0 = alpha p q ∨ x_0 = beta p q) →
  (∀ n, newtons_method p q x_n n = real.recurring_convergent n (newtons_method p q x_n)) :=
sorry

end newtons_method_coincide_with_convergents_l531_531380


namespace simplify_tan_cot_fraction_l531_531290

theorem simplify_tan_cot_fraction :
  let tan45 := 1
  let cot45 := 1
  (tan45^3 + cot45^3) / (tan45 + cot45) = 1 := by
    sorry

end simplify_tan_cot_fraction_l531_531290


namespace general_term_of_arithmetic_sequence_l531_531582

theorem general_term_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a3 : a 3 = -2)
  (h_a7 : a 7 = -10) :
  ∀ n : ℕ, a n = 4 - 2 * n :=
sorry

end general_term_of_arithmetic_sequence_l531_531582


namespace sqrt_11_custom_op_l531_531841

noncomputable def sqrt := Real.sqrt

def custom_op (x y : Real) := (x + y) ^ 2 - (x - y) ^ 2

theorem sqrt_11_custom_op : custom_op (sqrt 11) (sqrt 11) = 44 :=
by
  sorry

end sqrt_11_custom_op_l531_531841


namespace solve_system_l531_531382

theorem solve_system :
  ∃ x y z : ℝ, 
    (∀ n : ℕ, 
      (choose n 0 * x + choose n 1 * y + choose n 2 * z = choose n 3) ∧
      (choose (n+1) 0 * x + choose (n+1) 1 * y + choose (n+1) 2 * z = choose (n+1) 3) ∧
      (choose (n+2) 0 * x + choose (n+2) 1 * y + choose (n+2) 2 * z = choose (n+2) 3)
    ) → 
    (∃ x' y' z' : ℝ, 
      x' = x ∧ 
      y' = -choose (n+1) 2 ∧ 
      z' = choose n 1) :=
begin
  sorry
end

end solve_system_l531_531382


namespace equal_angles_ADV_BDU_l531_531662

-- Definitions for the geometric elements and conditions
variables (A B C D P U V : Type) [PlaneGeometry A B C D P U V]
variable {ABC_acute : is_acute_triangle A B C}
variable {foot_of_altitude : is_foot_of_altitude C D}
variable {P_on_CD : is_internal_point_on_segment P C D}
variable {intersect_U : is_intersection_point U (line A P) (line B C)}
variable {intersect_V : is_intersection_point V (line B P) (line A C)}

-- The theorem to prove
theorem equal_angles_ADV_BDU : angle A D V = angle B D U :=
sorry

end equal_angles_ADV_BDU_l531_531662


namespace binary_arithmetic_l531_531526

theorem binary_arithmetic :
  (nat.ofDigits 2 [1, 1, 1, 0, 1] + nat.ofDigits 2 [1, 1, 0, 1] - nat.ofDigits 2 [1, 0, 1, 1, 0] + nat.ofDigits 2 [1, 0, 1, 1]) = nat.ofDigits 2 [1, 1, 0, 1, 1] :=
by
  sorry

end binary_arithmetic_l531_531526


namespace units_digit_27_mul_46_l531_531553

-- Define the function to calculate the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement: The units digit of 27 * 46 is 2
theorem units_digit_27_mul_46 : units_digit (27 * 46) = 2 :=
  sorry

end units_digit_27_mul_46_l531_531553


namespace relay_race_total_distance_l531_531286

theorem relay_race_total_distance
  (Sadie_speed : ℝ) (Sadie_time : ℝ) (Ariana_speed : ℝ) (Ariana_time : ℝ) (Sarah_speed : ℝ) (total_race_time : ℝ)
  (h1 : Sadie_speed = 3) (h2 : Sadie_time = 2)
  (h3 : Ariana_speed = 6) (h4 : Ariana_time = 0.5)
  (h5 : Sarah_speed = 4) (h6 : total_race_time = 4.5) :
  (Sadie_speed * Sadie_time + Ariana_speed * Ariana_time + Sarah_speed * (total_race_time - (Sadie_time + Ariana_time))) = 17 :=
by
  sorry

end relay_race_total_distance_l531_531286


namespace identify_heaviest_and_lightest_l531_531811

theorem identify_heaviest_and_lightest (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ weighings : Fin 13 → (Fin 10 × Fin 10),
  (let outcomes := fun w ℕ => ite (coins (weighings w).fst > coins (weighings w).snd) (weighings w).fst (weighings w).snd,
  max_coin := nat.rec_on 12 (outcomes 0) (λ n max_n, if coins (outcomes (succ n)) > coins max_n then outcomes (succ n) else max_n),
  min_coin := nat.rec_on 12 (outcomes 0) (λ n min_n, if coins (outcomes (succ n)) < coins min_n then outcomes (succ n) else min_n))
  (∃ max_c : Fin 10, ∃ min_c : Fin 10, max_c ≠ min_c ∧ max_c = Some max_coin ∧ min_c = Some min_coin) :=
sorry

end identify_heaviest_and_lightest_l531_531811


namespace range_x2_y2_l531_531614

noncomputable def intersect_points (k : ℝ) : ℝ × ℝ :=
  let xA := 1 / Real.sqrt k
  let yA := Real.sqrt k
  let xB := -1 / Real.sqrt k
  let yB := -Real.sqrt k
  ((xA, yA), (xB, yB))

theorem range_x2_y2 (k : ℝ) :
  let P := (3, 4)
  let A := intersect_points k
  (∀ x : ℝ, ∀ y : ℝ, let PA := (A.1.1 - x, A.1.2 - y)
                     let PB := (A.2.1 - x, A.2.2 - y)
                     ‖(PA.1, PA.2) + (PB.1, PB.2)‖ = 2) →
  16 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 36 :=
sorry

end range_x2_y2_l531_531614


namespace coefficient_x3_expansion_l531_531222

open Nat Binomial

theorem coefficient_x3_expansion :
  (∑ k in range 6, (binomial 5 k) * (0 ^ (3 - k)) - ∑ k in range 7, (binomial 6 k) * (0 ^ (3 - k))) = -5 :=
by
  sorry

end coefficient_x3_expansion_l531_531222


namespace coordinates_of_P_range_of_b_solution_of_inequality_l531_531641

-- Part 1
theorem coordinates_of_P (a b x y : ℝ) (h₁: 2*a - b - 4 = x + y) (h₂: b - 4 = x - y) (ha: a = 1) (hb: b = 2) :
    x = -3 ∧ y = -1 := sorry

-- Part 2
theorem range_of_b (a b x y : ℤ) (h₁: x = a - 4) (h₂: y = a - b) (h₃: x < 0) (h₄: y > 0) 
    (h_int_count: ∃ n : ℕ, n = 4 ∧ card ({a | -1 ≤ a ∧ a < 4} ∩ ℤ) = n) :
    -1 ≤ b ∧ b < 0 := sorry

-- Part 3
theorem solution_of_inequality (a b t x y : ℝ) (h₁: b = 3/2 * a) (h₂: yz + x + 4 = 0) (hz: z = 2) :
    (a > 0 → t > 3/2) ∧ (a < 0 → t < 3/2) := sorry

end coordinates_of_P_range_of_b_solution_of_inequality_l531_531641


namespace binomial_510_510_l531_531903

theorem binomial_510_510 : binomial 510 510 = 1 :=
by sorry

end binomial_510_510_l531_531903


namespace arithmetic_sequence_twentieth_term_l531_531912

theorem arithmetic_sequence_twentieth_term :
  ∀ (a_1 d : ℕ), a_1 = 3 → d = 4 → (a_1 + (20 - 1) * d) = 79 := by
  intros a_1 d h1 h2
  rw [h1, h2]
  simp
  sorry

end arithmetic_sequence_twentieth_term_l531_531912


namespace probability_of_difference_three_l531_531488

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l531_531488


namespace scientific_notation_of_coronavirus_diameter_l531_531734

theorem scientific_notation_of_coronavirus_diameter : 
  (0.00000011 : ℝ) = 1.1 * 10^(-7) :=
  sorry

end scientific_notation_of_coronavirus_diameter_l531_531734


namespace arithmetic_mean_is_correct_l531_531516

-- Define the numbers
def num1 : ℕ := 18
def num2 : ℕ := 27
def num3 : ℕ := 45

-- Define the number of terms
def n : ℕ := 3

-- Define the sum of the numbers
def total_sum : ℕ := num1 + num2 + num3

-- Define the arithmetic mean
def arithmetic_mean : ℕ := total_sum / n

-- Theorem stating that the arithmetic mean of the numbers is 30
theorem arithmetic_mean_is_correct : arithmetic_mean = 30 := by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l531_531516


namespace common_point_ln_ln_eq_neg_one_l531_531143

noncomputable def common_point_ln_ln (a : ℝ) (h : 1 < a) : (y : ℝ) :=
begin
  sorry
end

theorem common_point_ln_ln_eq_neg_one (a : ℝ) (h : 1 < a) 
  (h_common : ∃ (x : ℝ), (a ^ x = x) ∧ (log a x = x)) : 
  common_point_ln_ln a h = -1 :=
begin
  sorry
end

end common_point_ln_ln_eq_neg_one_l531_531143


namespace sum_of_angles_l531_531130

open Real

theorem sum_of_angles (A : Real) (hA1: 0 ≤ A ∧ A ≤ 360)
  (h : ∀ y ∈ Icc 0 360, sin y ^ 4 - cos y ^ 4 = 1 / cos y + 1 / sin y → y = 45 ∨ y = 225) :
  A = 270 :=
sorry

end sum_of_angles_l531_531130


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531446

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531446


namespace sam_final_investment_l531_531710

-- Definitions based on conditions
def initial_investment : ℝ := 10000
def first_interest_rate : ℝ := 0.20
def years_first_period : ℕ := 3
def triple_amount : ℕ := 3
def second_interest_rate : ℝ := 0.15
def years_second_period : ℕ := 1

-- Lean function to accumulate investment with compound interest
def compound_interest (P r: ℝ) (n: ℕ) : ℝ := P * (1 + r) ^ n

-- Sam's investment calculations
def amount_after_3_years : ℝ := compound_interest initial_investment first_interest_rate years_first_period
def new_investment : ℝ := triple_amount * amount_after_3_years
def final_amount : ℝ := compound_interest new_investment second_interest_rate years_second_period

-- Proof goal (statement with the proof skipped)
theorem sam_final_investment : final_amount = 59616 := by
  sorry

end sam_final_investment_l531_531710


namespace flea_returns_to_0_l531_531858

noncomputable def flea_return_probability (p : ℝ) : ℝ :=
if p = 1 then 0 else 1

theorem flea_returns_to_0 (p : ℝ) : 
  flea_return_probability p = (if p = 1 then 0 else 1) :=
by
  sorry

end flea_returns_to_0_l531_531858


namespace average_speed_second_day_l531_531079

theorem average_speed_second_day
  (t v : ℤ)
  (h1 : 2 * t + 2 = 18)
  (h2 : (v + 5) * (t + 2) + v * t = 680) :
  v = 35 :=
by
  sorry

end average_speed_second_day_l531_531079


namespace mult_mod_7_zero_l531_531089

theorem mult_mod_7_zero :
  (2007 ≡ 5 [MOD 7]) →
  (2008 ≡ 6 [MOD 7]) →
  (2009 ≡ 0 [MOD 7]) →
  (2010 ≡ 1 [MOD 7]) →
  (2007 * 2008 * 2009 * 2010 ≡ 0 [MOD 7]) :=
by
  intros h1 h2 h3 h4
  sorry

end mult_mod_7_zero_l531_531089


namespace sasha_remainder_20_l531_531778

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l531_531778


namespace value_of_M_l531_531753

theorem value_of_M : 
  ∃ (M : ℚ), 
  let a₁₁ := 25,
      a₄₁ := 16, a₅₁ := 20,
      a₄₂ := -20 in
  ∀ a b c d : ℚ,
  (20 - 16 = a) ∧
  (16 - a = b) ∧
  (b - a = c) ∧
  (a₁₁ + -5 * (-17 / 3) = d) ∧
  ([-20 - d] / 4 = -115 / 6) ∧
  (d - 115 / 6 = M) 
  → M = 37.5 := 
by
  sorry

end value_of_M_l531_531753


namespace probability_of_diff_3_is_1_over_9_l531_531424

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l531_531424


namespace number_of_green_peaches_l531_531801

theorem number_of_green_peaches:
  ∀ (total_peaches red_peaches green_peaches : ℕ), 
  (total_peaches = 10) ∧ (red_peaches = 7) → (green_peaches = total_peaches - red_peaches) → green_peaches = 3 :=
by
  intros total_peaches red_peaches green_peaches h_total h_green
  cases h_total
  cases h_green
  simp
  sorry

end number_of_green_peaches_l531_531801


namespace correct_arrangements_count_l531_531800

noncomputable def arrangements_with_restriction := 
  let total_positions := 5
  let positions_for_A := 3
  let arrangements_of_remaining := Nat.fact 4
  positions_for_A * arrangements_of_remaining

theorem correct_arrangements_count : arrangements_with_restriction = 72 := by
  sorry

end correct_arrangements_count_l531_531800


namespace prop_or_lean_statement_l531_531578

theorem prop_or_lean_statement (p : 3 > 4) (q : 3 < 4) : p ∨ q := 
by
  -- Placeholder for the proof
  sorry

end prop_or_lean_statement_l531_531578


namespace total_weight_of_sand_l531_531802

theorem total_weight_of_sand (n : ℕ) : 
  ∃ total_weight, total_weight = (n - 1) * 65 + 42 :=
by
  use (n - 1) * 65 + 42
  sorry

end total_weight_of_sand_l531_531802


namespace man_son_work_together_in_4_days_l531_531406

-- Definitions of the work rates
def man_work_rate : ℝ := 1 / 5
def son_work_rate : ℝ := 1 / 20

-- The combined work rate of man and his son
def combined_work_rate : ℝ := man_work_rate + son_work_rate

-- The total time to complete the work together
def total_time_to_complete_work : ℝ := 1 / combined_work_rate

-- The theorem we want to prove
theorem man_son_work_together_in_4_days : total_time_to_complete_work = 4 := by
  -- Proof steps would be here, but we add sorry for now
  sorry

end man_son_work_together_in_4_days_l531_531406


namespace positive_difference_equal_4750_l531_531238

-- Define the sum of the first 100 positive integers
def jo_sum : ℕ := (100 * 101) / 2

-- Function to round integers to the nearest multiple of 5
def round_to_nearest_five (n : ℕ) : ℕ :=
  if n % 5 < 3 then (n / 5) * 5 else (n / 5 + 1) * 5

-- Define Luke's sum by rounding each integer from 1 to 100 and then summing
def luke_sum : ℕ :=
  (Finset.range 100).sum (λ n, round_to_nearest_five (n + 1))

-- The target theorem to prove the positive difference is 4750
theorem positive_difference_equal_4750 : abs (jo_sum - luke_sum) = 4750 :=
by
  sorry

end positive_difference_equal_4750_l531_531238


namespace sum_of_final_sequence_digits_l531_531062

theorem sum_of_final_sequence_digits :
  let initial_sequence := (List.repeat ['1', '2', '3', '4', '5', '6'] 2000).join
  let sequence_after_first_erasure := initial_sequence.eraseEveryNth 5
  let final_sequence := sequence_after_first_erasure.eraseEveryNth 7
  let digit3031 := final_sequence.getDigit 3031
  let digit3032 := final_sequence.getDigit 3032
  let digit3033 := final_sequence.getDigit 3033
  digit3031.digitToInt + digit3032.digitToInt + digit3033.digitToInt = 9 :=
  sorry

end sum_of_final_sequence_digits_l531_531062


namespace sum_of_non_palindrome_four_step_integers_l531_531949

-- Definition of a helper function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr.toList in s = s.reverse

-- Definition of a helper function to reverse the digits of a number
def reverse_number (n : ℕ) : ℕ :=
  let s := n.repr.toList.reverse in s.asString.toNat

-- Definition of the reverse-and-add process
def reverse_and_add (n : ℕ) : ℕ :=
  n + reverse_number n

-- Definition of the iterative process of taking steps to become a palindrome
def steps_to_palindrome (n : ℕ) : ℕ :=
  if is_palindrome n then 0
  else if is_palindrome (reverse_and_add n) then 1
  else if is_palindrome (reverse_and_add (reverse_and_add n)) then 2
  else if is_palindrome (reverse_and_add (reverse_and_add (reverse_and_add n))) then 3
  else if is_palindrome (reverse_and_add (reverse_and_add (reverse_and_add (reverse_and_add n)))) then 4
  else 5  -- if it needs more steps, it's more than 4

-- Definition of the main proof statement
theorem sum_of_non_palindrome_four_step_integers :
  ∑ n in (Finset.range (200 - 100 + 1)).map (λ x, x + 100) | non_palindrome n ∧ steps_to_palindrome n = 4, n = 1191 :=
sorry

end sum_of_non_palindrome_four_step_integers_l531_531949


namespace sasha_remainder_l531_531772

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l531_531772


namespace at_least_one_ge_two_l531_531591

theorem at_least_one_ge_two (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  a + 1 / b ≥ 2 ∨ b + 1 / c ≥ 2 ∨ c + 1 / a ≥ 2 := 
sorry

end at_least_one_ge_two_l531_531591


namespace find_f_log3_9_l531_531592

open Real

theorem find_f_log3_9
  (f : ℝ → ℝ)
  (h_inc : ∀ x y : ℝ, x < y → f x < f y)
  (h_fn_eq : ∀ x : ℝ, f (f x - 3^x) = 4) :
  f (log 3 9) = 10 := 
sorry

end find_f_log3_9_l531_531592


namespace magnitude_of_complex_number_l531_531539

theorem magnitude_of_complex_number : 
  ∀ (a b : ℤ), (a = 3 ∧ b = -10) → (|complex.mk a b| = real.sqrt (a^2 + b^2)) := 
by
  intros a b hb
  cases hb
  rw [hb_left, hb_right]
  exact sorry

end magnitude_of_complex_number_l531_531539


namespace amelia_wins_probability_is_expected_l531_531884

noncomputable def amelia_wins_probability : ℚ :=
  let P_am_head := (1:ℚ) / 4
  let P_bl_head := (1:ℚ) / 3
  let P_am_tail := 1 - P_am_head
  let P_bl_tail := 1 - P_bl_head
  let P_both_tails := P_am_tail * P_bl_tail
  let P_not_both_tails := 1 - P_both_tails
  let sum_geom_prob (n : ℕ) : ℚ := if n = 0 then 1 else sum (λ k, (P_both_tails^k)) (fin (n+1))
  P_am_head * sum_geom_prob 4

theorem amelia_wins_probability_is_expected : amelia_wins_probability = (15:ℚ) / (32:ℚ) :=
by sorry

end amelia_wins_probability_is_expected_l531_531884


namespace constant_term_binomial_expansion_l531_531306

theorem constant_term_binomial_expansion :
  let T (r : ℕ) := (Nat.choose 6 r) * (-2)^r * (x^((6 - 3*r)/2))
  x ≠ 0 →
  ∃ r : ℕ, r = 2 ∧ T r = 60 :=
by
  let T (r : ℕ) := Nat.choose 6 r * (-2)^r * x^((6 - 3*r)/2)
  have x_ne_0 : x ≠ 0, sorry
  have r := 2
  use r
  have T_r : T r = 60, sorry
  exact ⟨r, rfl, T_r⟩

end constant_term_binomial_expansion_l531_531306


namespace interest_calculated_months_l531_531549

-- Definitions of the conditions
def simple_interest := 400
def principal := 10000
def annual_rate := 0.04
def T_years := simple_interest / (principal * annual_rate)

-- The proof statement (targeting number of months)
theorem interest_calculated_months : (T_years * 12) = 12 :=
by
  -- Proof is omitted
  sorry

end interest_calculated_months_l531_531549


namespace analytic_expression_of_f_range_of_k_equations_of_axis_of_symmetry_l531_531179

noncomputable def f (x : ℝ) : ℝ := 2 * real.sin ((2 * real.pi / 3) * x + real.pi / 6)

-- The first problem statement
theorem analytic_expression_of_f :
  ∃ A ω φ, (A > 0) ∧ (ω > 0) ∧ (φ ∈ Ioo 0 (real.pi / 2)) ∧
  (f 0 = 1) ∧
  (∀ x₀ > 0, f (x₀ - 3 / 2) = 2) ∧
  (∀ x₀ > 0, f x₀ = -2) ∧
  (f = λ x, 2 * real.sin ((2 * real.pi / 3) * x + real.pi / 6)) := 
sorry

-- The second problem statement
theorem range_of_k (k : ℝ) :
  (∃ x1 x2 ∈ Icc 0 (3 / 2), f x1 = (k + 1) / 2 ∧ f x2 = (k + 1) / 2 ∧ x1 ≠ x2) ↔
  1 ≤ k ∧ k < 3 := 
sorry

-- The third problem statement
theorem equations_of_axis_of_symmetry :
  ∃ x1 x2, (x1 = 7 / 2 ∨ x1 = 5) ∧ (x2 = 7 / 2 ∨ x2 = 5) ∧
  (x1 ≠ x2) ∧
  ∀ x ∈ Icc (13 / 4) (23 / 4), (f x = f ((2:x1 + x2) / 2 - x)) ∧
  f (x1) = f (x2) := 
sorry

end analytic_expression_of_f_range_of_k_equations_of_axis_of_symmetry_l531_531179


namespace scuba_diver_descent_rate_l531_531052

theorem scuba_diver_descent_rate :
  ∀ (depth time : ℕ), depth = 3600 ∧ time = 60 → (depth / time = 60) :=
by
  intro depth time h
  cases h
  rw [h_left, h_right]
  exact Nat.div_eq_of_lt_of_dvd (by decide) (by decide)

end scuba_diver_descent_rate_l531_531052


namespace dice_rolls_diff_by_3_probability_l531_531453

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l531_531453


namespace circle_radius_twice_value_l531_531523

theorem circle_radius_twice_value (r_x r_y v : ℝ) (h1 : π * r_x^2 = π * r_y^2)
  (h2 : 2 * π * r_x = 12 * π) (h3 : r_y = 2 * v) : v = 3 := by
  sorry

end circle_radius_twice_value_l531_531523


namespace probability_of_difference_three_l531_531493

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l531_531493


namespace units_digit_of_p_is_6_l531_531983

-- Given conditions
variable (p : ℕ)
variable (h1 : p % 2 = 0)                -- p is a positive even integer
variable (h2 : (p^3 % 10) - (p^2 % 10) = 0)  -- The units digit of p^3 minus the units digit of p^2 is 0
variable (h3 : (p + 2) % 10 = 8)         -- The units digit of p + 2 is 8

-- Prove the units digit of p is 6
theorem units_digit_of_p_is_6 : p % 10 = 6 :=
sorry

end units_digit_of_p_is_6_l531_531983


namespace solutions_tan_cot_eq_l531_531106

noncomputable def numberOfSolutions : ℕ :=
  ∑ (theta : ℝ) in ({0 < theta ∧ theta < 2 * π} : set ℝ), 
  ite ((tan (3 * π * cos theta)) = (cot (4 * π * sin theta))) 1 0

theorem solutions_tan_cot_eq : numberOfSolutions = 18 := 
  sorry

end solutions_tan_cot_eq_l531_531106


namespace balls_into_boxes_l531_531793

theorem balls_into_boxes (r n : ℕ) (h : r ≥ n) : 
  let f := ∑ k in Finset.range (n + 1), (-1:ℤ)^k * (Nat.choose n k) * (n - k)^r in
  let S := f / Nat.factorial n in
  S = (1 / (Nat.factorial n:ℚ) * ∑ k in Finset.range (n + 1), (-1:ℤ)^k * (Nat.choose n k) * (n - k)^r) := 
by {
  sorry
}

end balls_into_boxes_l531_531793


namespace ten_a_plus_b_l531_531618

section
variable (f g h : ℝ → ℝ)
variable (a b : ℕ)

-- Definition of f(x)
def f (x : ℝ) : ℝ := (x - 2) * (x - 4) / 3

-- Definition of g(x) as -f(x)
def g (x : ℝ) : ℝ := -f x

-- Definition of h(x) as f(-x)
def h (x : ℝ) : ℝ := f (-x)

-- Number of intersection points
def a : ℕ := 2
def b : ℕ := 1

-- Proof statement
theorem ten_a_plus_b : 10 * a + b = 21 := by
  sorry
end

end ten_a_plus_b_l531_531618


namespace angle_of_inclination_of_line_l531_531930

theorem angle_of_inclination_of_line :
  ∀ (α : ℝ), (tan α = -√3) → (0 ≤ α ∧ α < real.pi) → α = (2 * real.pi / 3) :=
by
  sorry

end angle_of_inclination_of_line_l531_531930


namespace find_max_m_solve_inequality_l531_531998

def f (x m : ℝ) := real.sqrt (abs (x + 1) + abs (x - 3) - m)

theorem find_max_m (m : ℝ) : (∀ x : ℝ, f x m ≠ real.sqrt (-1 - 1)) → m ≤ 4 := by
  sorry

theorem solve_inequality (n x : ℝ) (h : n = 4) :
  (abs (x - 3) - 2 * x ≤ 2 * n - 4) →
  (x ≥ -1/3) := by
  sorry

end find_max_m_solve_inequality_l531_531998


namespace factor_expression_l531_531083

variables (b : ℝ)

theorem factor_expression :
  (8 * b ^ 3 + 45 * b ^ 2 - 10) - (-12 * b ^ 3 + 5 * b ^ 2 - 10) = 20 * b ^ 2 * (b + 2) :=
by
  sorry

end factor_expression_l531_531083


namespace dice_rolls_diff_by_3_probability_l531_531452

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l531_531452


namespace smallest_period_find_a_l531_531140

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + a

theorem smallest_period (a : ℝ) : 
  ∃ T > 0, ∀ x, f x a = f (x + T) a ∧ (∀ T' > 0, (∀ x, f x a = f (x + T') a) → T ≤ T') :=
by
  sorry

theorem find_a :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) ∧ (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = 4) ∧ a = 1 :=
by
  sorry

end smallest_period_find_a_l531_531140


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531447

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531447


namespace angle_ABC_twice_angle_FCB_l531_531226

-- Definition of the theorem with all necessary geometrical constructs and assumptions

theorem angle_ABC_twice_angle_FCB
  (O : Circle)
  (A B C P K D E F : Point)
  (h_insc : inscribed_in O (triangle A B C))
  (h_P_arc_BC : on_arc P B C O)
  (h_K_on_AP : on_segment K A P)
  (h_bisects : angle_bisector_segment B K (angle A B C))
  (Omega : Circle)
  (h_Omega : passes_through Omega [K, P, C])
  (h_D_on_AC : on_segment D A C)
  (h_E_on_Omega : on_circle E Omega)
  (h_BD_intersects_E : BD_intersects_circle_at E Omega)
  (h_PE_extends_F : line_segment_extends_to PE F A B) :
  angle A B C = 2 * angle F C B := 
sorry

end angle_ABC_twice_angle_FCB_l531_531226


namespace find_A_l531_531954

variable (x ω φ b A : ℝ)

-- Given conditions
axiom cos_squared_eq : 2 * (Real.cos (x + Real.sin (2 * x)))^2 = A * Real.sin (ω * x + φ) + b
axiom A_gt_zero : A > 0

-- Lean 4 statement to prove
theorem find_A : A = Real.sqrt 2 :=
by
  sorry

end find_A_l531_531954


namespace interval_of_decrease_l531_531101

noncomputable def f (x : ℝ) : ℝ := (1/3)^(x^2 - 6*x + 5)

theorem interval_of_decrease : ∀ x, x ∈ Set.Ici 3 ↔ ∀ y : ℝ, f' y < 0 :=
by
  sorry

end interval_of_decrease_l531_531101


namespace dice_rolls_diff_by_3_probability_l531_531450

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l531_531450


namespace polynomial_remainder_correct_l531_531939

noncomputable def polynomial_remainder : ℤ[X] :=
  (X^4 + 3 * X^2 - 2) % (X^2 - 4 * X + 3)

theorem polynomial_remainder_correct :
  polynomial_remainder = 88 * X - 59 :=
by
  sorry

end polynomial_remainder_correct_l531_531939


namespace find_b_l531_531569

theorem find_b (b : ℝ) (h : Function.Bijective (λ x : ℝ, 2^x + b)) (hinv : (h.1 5) = 2) : b = 1 := by
  -- The proof would go here.
  sorry

end find_b_l531_531569


namespace similar_triangle_side_sum_l531_531058

theorem similar_triangle_side_sum 
  (a b c : ℕ) 
  (h1 : a = 8) 
  (h2 : b = 10) 
  (h3 : c = 12) 
  (perimeter_similar : ℕ)
  (h4 : perimeter_similar = 180) :
  let x := perimeter_similar / (a + b + c) in
  let side1_similar := a * x in
  let side2_similar := b * x in
  side1_similar + side2_similar = 108 := 
by
  sorry

end similar_triangle_side_sum_l531_531058


namespace spinsters_count_l531_531016

theorem spinsters_count (S C : ℕ) (h1 : S / C = 2 / 9) (h2 : C = S + 42) : S = 12 := by
  sorry

end spinsters_count_l531_531016


namespace find_line_equation_l531_531122

theorem find_line_equation : 
  ∃ c : ℝ, (∀ x y : ℝ, 2*x + 4*y + c = 0 ↔ x + 2*y - 8 = 0) ∧ (2*2 + 4*3 + c = 0) :=
sorry

end find_line_equation_l531_531122


namespace probability_diff_by_3_l531_531478

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l531_531478


namespace find_m_l531_531959

theorem find_m {a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} m : ℝ} 
    (h1 : m > 0)
    (h2 : (1 + m * x) ^ 10 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_{10} * x^{10})
    (h3 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} = 1024)
    (h4 : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} = 1023) : 
    m = 1 :=
by
  sorry

end find_m_l531_531959


namespace ratio_norm_eq_sqrt3_over_2_l531_531156

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (h₁ : a ≠ 0) (h₂ : b ≠ 0)
variables (h₃ : ‖a + b‖ = ‖a - 2 • b‖)
variables (h₄ : (proj a b) = (2 / 3) • a)

theorem ratio_norm_eq_sqrt3_over_2 
  (a b : EuclideanSpace ℝ (Fin 3)) 
  (h₁ : a ≠ 0) 
  (h₂ : b ≠ 0) 
  (h₃ : ‖a + b‖ = ‖a - 2 • b‖) 
  (h₄ : Proj a b = (2 / 3) • a) : 
  ‖a‖ / ‖b‖ = (Real.sqrt 3) / 2 :=
  sorry

end ratio_norm_eq_sqrt3_over_2_l531_531156


namespace quadratic_solution_l531_531202

theorem quadratic_solution (m n : ℝ) (h1 : m ≠ 0) (h2 : m * 1^2 + n * 1 - 1 = 0) : m + n = 1 :=
sorry

end quadratic_solution_l531_531202


namespace intersection_point_of_symmetric_lines_P_is_circumcenter_of_A1B1C1_l531_531889

variables {A B C O A' B' C' A1 B1 C1 P : Type*}
variables [is_circumcenter O A B C] [is_symmetric O A'] [is_symmetric O B'] [is_symmetric O C']
variables [is_midpoint A1 B C] [is_midpoint B1 C A] [is_midpoint C1 A B]
variables [xLine : ∀ (X'1 X'2 : Type*), Type*]

noncomputable def AA' := xLine A A'
noncomputable def BB' := xLine B B'
noncomputable def CC' := xLine C C'

theorem intersection_point_of_symmetric_lines :
  ∃ (P : Type*), ∃ A' B' C' : Type*, 
    (P ∈ AA' ∧ P ∈ BB' ∧ P ∈ CC') :=
sorry

theorem P_is_circumcenter_of_A1B1C1 : 
  is_circumcenter P A1 B1 C1 :=
sorry

end intersection_point_of_symmetric_lines_P_is_circumcenter_of_A1B1C1_l531_531889


namespace arcade_ticket_problem_l531_531511

-- Define all the conditions given in the problem
def initial_tickets : Nat := 13
def used_tickets : Nat := 8
def more_tickets_for_clothes : Nat := 10
def tickets_for_toys : Nat := 8
def tickets_for_clothes := tickets_for_toys + more_tickets_for_clothes

-- The proof statement (goal)
theorem arcade_ticket_problem : tickets_for_clothes = 18 := by
  -- This is where the proof would go
  sorry

end arcade_ticket_problem_l531_531511


namespace possible_values_of_n_l531_531990

open Nat

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c (n : ℕ) : ℕ := a (b n)

noncomputable def T (n : ℕ) : ℕ := (Finset.range n).sum (λ i => c (i + 1))

theorem possible_values_of_n (n : ℕ) :
  T n < 2021 → n = 8 ∨ n = 9 := by
  sorry

end possible_values_of_n_l531_531990


namespace box_width_is_target_width_l531_531049

-- Defining the conditions
def cube_volume : ℝ := 27
def box_length : ℝ := 8
def box_height : ℝ := 12
def max_cubes : ℕ := 24

-- Defining the target width we want to prove
def target_width : ℝ := 6.75

-- The proof statement
theorem box_width_is_target_width :
  ∃ w : ℝ,
  (∀ v : ℝ, (v = max_cubes * cube_volume) →
   ∀ l : ℝ, (l = box_length) →
   ∀ h : ℝ, (h = box_height) →
   v = l * w * h) →
   w = target_width :=
by
  sorry

end box_width_is_target_width_l531_531049


namespace xy_equation_result_l531_531709

theorem xy_equation_result (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -5) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -10.528 :=
by
  sorry

end xy_equation_result_l531_531709


namespace cube_reciprocal_identity_l531_531960

theorem cube_reciprocal_identity (x : ℝ) (h : x + x⁻¹ = 3) : x^3 + x⁻³ = 18 := 
by 
  sorry

end cube_reciprocal_identity_l531_531960


namespace mine_avoiding_path_count_l531_531381

open Finset

-- Definition of lattice points
def LatticePoint := ℤ × ℤ

-- Definition of a set of lattice points (M)
def M : Finset LatticePoint := sorry

-- Definition of n as a positive integer
def n : ℕ := sorry

-- Definition of a mine-avoiding path
def is_mine_avoiding_path (path : List LatticePoint) : Prop :=
  path.head! = (0, 0) ∧ List.last path (0, 0).snd = n ∧ 
  (∀ p ∈ path, p ∉ M) ∧ (∀ i < path.length - 1, (path.get i).fst + (path.get i).snd + 1 = (path.get i + 1).fst + (path.get i + 1).snd)

-- Predicate: There exists a mine-avoiding path
def exists_mine_avoiding_path : Prop :=
  ∃ path : List LatticePoint, is_mine_avoiding_path path

-- Theorem stating the problem
theorem mine_avoiding_path_count :
  exists_mine_avoiding_path → ∃ k ≥ 2^(n - M.card), k = nat.card {path // is_mine_avoiding_path path} :=
sorry

end mine_avoiding_path_count_l531_531381


namespace triangle_inequality_sqrt_equality_condition_l531_531631

theorem triangle_inequality_sqrt 
  {a b c : ℝ} 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) 
  ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) := 
sorry

theorem equality_condition 
  {a b c : ℝ} 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) 
  = Real.sqrt a + Real.sqrt b + Real.sqrt c) → 
  (a = b ∧ b = c) := 
sorry

end triangle_inequality_sqrt_equality_condition_l531_531631


namespace identify_heaviest_and_lightest_coin_l531_531807

theorem identify_heaviest_and_lightest_coin :
  ∀ (coins : Fin 10 → ℕ), 
  (∀ i j, i ≠ j → coins i ≠ coins j) → 
  ∃ (seq : List (Fin 10 × Fin 10)), 
  seq.length = 13 ∧ 
  (∀ (i j : Fin 10), (i, j) ∈ seq → 
    (coins i < coins j ∨ coins i > coins j)) ∧ 
  (∃ (heaviest lightest : Fin 10),
    (∀ coin, coins coin ≤ coins heaviest) ∧ (∀ coin, coins coin ≥ coins lightest)) :=
by
  intros coins h_coins
  exists [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), -- initial pairs
          (0, 2), (2, 4), (4, 6), (6, 8),         -- heaviest coin comparisons
          (1, 3), (3, 5), (5, 7), (7, 9)]         -- lightest coin comparisons
  constructor
  . -- length check
    rfl
  . constructor
    . -- all comparisons
      intros i j h_pair
      cases h_pair; simp; solve_by_elim
    . -- finding heaviest and lightest coins
      exists 8, 9
      constructor
      . -- all coins are less than or equal to the heaviest
        sorry
      . -- all coins are greater than or equal to the lightest
        sorry

end identify_heaviest_and_lightest_coin_l531_531807


namespace area_shaded_region_l531_531668

-- Given conditions:
def radius_small : ℝ := 50
def chord_length : ℝ := 120

-- Definition of the radius of the larger circle using the Pythagorean theorem.
noncomputable def radius_large : ℝ := real.sqrt (radius_small ^ 2 + (chord_length / 2) ^ 2)

-- Definition of the area of the circles.
noncomputable def area_small_circle : ℝ := real.pi * (radius_small ^ 2)
noncomputable def area_large_circle : ℝ := real.pi * (radius_large ^ 2)

-- Problem statement
theorem area_shaded_region : area_large_circle - area_small_circle = 3600 * real.pi :=
by
  sorry

end area_shaded_region_l531_531668


namespace sum_first_five_terms_l531_531145

-- Define the geometric sequence
noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^n

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n
  else a1 * (1 - q^(n + 1)) / (1 - q)

-- Given conditions
def a1 : ℝ := 1
def q : ℝ := 2
def n : ℕ := 5

-- The theorem to be proven
theorem sum_first_five_terms : sum_geometric_sequence a1 q (n-1) = 31 := by
  sorry

end sum_first_five_terms_l531_531145


namespace students_in_exams_l531_531655

theorem students_in_exams (T_Math T_Science T_English : ℕ)
  (h_Math : 0.65 * T_Math = 546)
  (h_Science : 0.58 * T_Science = 458)
  (h_English : 0.62 * T_English = 490) :
  T_Math = 840 ∧ T_Science = 790 ∧ T_English = 790 :=
by
  sorry

end students_in_exams_l531_531655


namespace grace_pennies_l531_531039

theorem grace_pennies :
  let dime_value := 10
  let coin_value := 5
  let dimes := 10
  let coins := 10
  dimes * dime_value + coins * coin_value = 150 :=
by
  let dime_value := 10
  let coin_value := 5
  let dimes := 10
  let coins := 10
  sorry

end grace_pennies_l531_531039


namespace simplify_expression_l531_531721

-- Define the algebraic expression
def algebraic_expr (x : ℚ) : ℚ := (3 / (x - 1) - x - 1) * (x - 1) / (x^2 - 4 * x + 4)

theorem simplify_expression : algebraic_expr 0 = 1 :=
by
  -- The proof is skipped using sorry
  sorry

end simplify_expression_l531_531721


namespace literature_club_students_neither_english_nor_french_l531_531701

theorem literature_club_students_neither_english_nor_french
  (total_students english_students french_students both_students : ℕ)
  (h1 : total_students = 120)
  (h2 : english_students = 72)
  (h3 : french_students = 52)
  (h4 : both_students = 12) :
  (total_students - ((english_students - both_students) + (french_students - both_students) + both_students) = 8) :=
by
  sorry

end literature_club_students_neither_english_nor_french_l531_531701


namespace convex_polygon_parts_with_different_areas_l531_531398

theorem convex_polygon_parts_with_different_areas (p : ℕ) (h : p ≥ 5) (P : Type) [polygon P] [convex P] [has_sides P p] :
  ∃ parts : list (set P), ∃ part1 part2 ∈ parts, part1 ≠ part2 ∧ area part1 ≠ area part2 :=
by
  sorry

end convex_polygon_parts_with_different_areas_l531_531398


namespace standard_equation_of_ellipse_distance_from_O_to_line_constant_l531_531176

-- Define the ellipse and related variables
noncomputable def a : ℝ := 2
noncomputable def b : ℝ := sqrt 3
def ellipse (x y : ℝ) : Prop := (x^2)/(a^2) + (y^2)/(b^2) = 1

-- Condition through the origin with slope 1
def line_through_origin_with_slope (x y : ℝ) : Prop := y = x

-- Define the perimeter and area conditions
def perimeter_of_quadrilateral : ℝ := 8
def area_of_quadrilateral : ℝ := 4 * sqrt 21 / 7

-- Define the solution parts
theorem standard_equation_of_ellipse :
  (∀ x y : ℝ, ellipse x y ↔ (x^2) / 4 + (y^2) / 3 = 1) := by
sorry

theorem distance_from_O_to_line_constant (m n : ℝ) (x y : ℝ) :
  (∀ x y : ℝ, line_through_origin_with_slope x y → ellipse x y) → 
  (let n := 2 * sqrt 21 * sqrt (m^2 + 1) / 7 in 
  (∀ x y : ℝ, y = m * x + n ∨ x = x) → 
  (∀ x y : ℝ, dist O line = 2 * sqrt 21 / 7)) := by
sorry

end standard_equation_of_ellipse_distance_from_O_to_line_constant_l531_531176


namespace median_of_vision_data_is_4_6_l531_531926

def student_vision_data := [(4.0, 1), (4.1, 2), (4.2, 6), (4.3, 3), (4.4, 3), 
                            (4.5, 4), (4.6, 1), (4.7, 2), (4.8, 5), (4.9, 7), (5.0, 5)]

def total_students := 39

def median_position := 20

noncomputable def student_visions_sorted := 
  student_vision_data.flat_map (fun (vision, num) => list.replicate num vision)

def median_vision (visions : list ℝ) : ℝ :=
  if h : 1 ≤ median_position ∧ median_position ≤ visions.length then 
    visions.nth_le (median_position - 1) h 
  else 
    0

theorem median_of_vision_data_is_4_6 : median_vision (student_visions_sorted) = 4.6 :=
sorry

end median_of_vision_data_is_4_6_l531_531926


namespace remainder_of_99_times_101_divided_by_9_is_0_l531_531357

theorem remainder_of_99_times_101_divided_by_9_is_0 : (99 * 101) % 9 = 0 :=
by
  sorry

end remainder_of_99_times_101_divided_by_9_is_0_l531_531357


namespace area_of_PQRS_l531_531656

theorem area_of_PQRS (PQ QR RS SP : ℝ) (angle_RSP : ℝ) 
  (hPQ : PQ = 9) (hQR : QR = 5) (hRS : RS = 13) (hSP : SP = 13) (hAngle : angle_RSP = 60) :
  ∃ (a b c : ℝ), a + b + c = 40.67 ∧ 
                  (∀ k, a = k^2 → k = 0 ∧ k = 1) ∧ 
                  (∀ k, c = k^2 → k = 0 ∧ k = 1) ∧
                  ∃ x y, x^2 = a ∧ y^2 = c ∧ 
                           (area_PQRS hPQ hQR hRS hSP hAngle = x + b * y) := 
sorry

end area_of_PQRS_l531_531656


namespace sasha_remainder_is_20_l531_531792

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l531_531792


namespace pure_imaginary_complex_number_l531_531303

theorem pure_imaginary_complex_number (x : ℝ) (z : ℂ) (h1 : z = (x^2 - 1) + (x + 1) * complex.I)
  (h2 : ∃ b : ℝ, z = b * complex.I) :
  x = 1 ∨ x = -1 :=
by
  sorry

end pure_imaginary_complex_number_l531_531303


namespace sasha_remainder_l531_531765

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l531_531765


namespace angle_AHB_l531_531883

open Real EuclideanGeometry

theorem angle_AHB (A B C D E H : Point)
  (h1 : Altitude A D)
  (h2 : Altitude B E)
  (h3 : ∠BCA = 80)
  (h4 : ∠BAC = 30)
  (h5 : distance A B = distance B C) :
  ∠AHB = 100 :=
sorry

end angle_AHB_l531_531883


namespace problem1_problem2_even_problem2_odd_l531_531386

-- Problem (1)
theorem problem1 (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : sin α = 4 / 5) :
  (sin (2 * π - α) * tan (π + α) * cos (-π + α)) / (sin (π / 2 - α) * cos (π / 2 + α)) = 4 / 3 :=
by sorry

-- Problem (2)
theorem problem2_even (n k : ℤ) (α : ℝ) (hn : n = 2 * k) :
  (sin (α + n * π) + sin (α - n * π)) / (sin (α + n * π) * cos (α - n * π)) = 2 / cos α :=
by sorry

theorem problem2_odd (n k : ℤ) (α : ℝ) (hn : n = 2 * k + 1) :
  (sin (α + n * π) + sin (α - n * π)) / (sin (α + n * π) * cos (α - n * π)) = -2 / cos α :=
by sorry

end problem1_problem2_even_problem2_odd_l531_531386


namespace product_of_roots_of_t_squared_equals_49_l531_531937

theorem product_of_roots_of_t_squared_equals_49 : 
  ∃ t : ℝ, (t^2 = 49) ∧ (t = 7 ∨ t = -7) ∧ (t * (7 + -7)) = -49 := 
by
  sorry

end product_of_roots_of_t_squared_equals_49_l531_531937


namespace probability_diff_by_3_l531_531480

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l531_531480


namespace part_I_extreme_points_part_II_range_of_a_l531_531180

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + Real.log (x + 1)

-- Specify the conditions and goals for part I
theorem part_I_extreme_points (x : ℝ) :
  let a := 2
  (f a x).deriv = 0 → (x = - Real.sqrt 2 / 2 ∨ x = Real.sqrt 2 / 2) := sorry

-- Specify the conditions and goals for part II
theorem part_II_range_of_a (x a : ℝ) :
  (∃ x ∈ Ioo 0 1, deriv (f a) x > x) → a ≤ 1 := sorry

end part_I_extreme_points_part_II_range_of_a_l531_531180


namespace max_divisors_in_products_l531_531886

theorem max_divisors_in_products :
  let products := [20, 38, 54, 68, 80, 90, 98]
  in ∃ n ∈ products, (∀ m ∈ products, (number_of_divisors n ≥ number_of_divisors m)) ∧ n = 90 :=
by
  sorry

def number_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

end max_divisors_in_products_l531_531886


namespace find_students_in_third_set_l531_531824

theorem find_students_in_third_set (x : ℕ) (h1 : ∀ a ∈ S_1, passed a) (h2 : ∀ b ∈ S_2, (p b) = 0.9 * 50) 
  (h3 : ∀ c ∈ S_3, (p c) = 0.8 * x) (h4 : ∑ a in (students S_1 ∪ students S_2 ∪ students S_3), passed a / (total_students S_1 + total_students S_2 + total_students S_3) = 88.66666666666667 / 100) : 
  x = 60 := 
sorry

end find_students_in_third_set_l531_531824


namespace area_ratio_l531_531743

theorem area_ratio
  (a b c : ℕ)
  (h1 : 2 * (a + c) = 2 * 2 * (b + c))
  (h2 : a = 2 * b)
  (h3 : c = c) :
  (a * c) = 2 * (b * c) :=
by
  sorry

end area_ratio_l531_531743


namespace find_x_eq_2_l531_531407

theorem find_x_eq_2 (x : ℕ) (h : 7899665 - 36 * x = 7899593) : x = 2 := 
by
  sorry

end find_x_eq_2_l531_531407


namespace sum_infinite_geometric_series_l531_531518

theorem sum_infinite_geometric_series (a r : ℚ) (h : a = 1) (h2 : r = 1/4) : 
  (∀ S, S = a / (1 - r) → S = 4 / 3) :=
by
  intros S hS
  rw [h, h2] at hS
  simp [hS]
  sorry

end sum_infinite_geometric_series_l531_531518


namespace all_Mems_not_Zeiges_no_Mem_is_Zeige_l531_531622

variable (Mem Enform Zeige : Type)

axiom all_Mems_are_Enforms : ∀ x : Mem, Enform x
axiom no_Enforms_are_Zeiges : ∀ y : Enform, ¬ Zeige y

theorem all_Mems_not_Zeiges : ∀ x : Mem, ¬ Zeige x :=
by
  intro x
  have h1 : Enform x := all_Mems_are_Enforms x
  have h2 : ¬ Zeige (Enform x) := no_Enforms_are_Zeiges (Enform x)
  exact h2 

theorem no_Mem_is_Zeige : ∀ x : Mem, ¬ Zeige x :=
all_Mems_not_Zeiges Mem Enform Zeige

end all_Mems_not_Zeiges_no_Mem_is_Zeige_l531_531622


namespace factorize_binomial_square_l531_531543

theorem factorize_binomial_square (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 :=
by
  sorry

end factorize_binomial_square_l531_531543


namespace k_solvable_implies_k_plus_3_solvable_l531_531159

theorem k_solvable_implies_k_plus_3_solvable 
  (n k : ℕ) 
  (a : ℕ → ℕ)
  (ha1 : (∑ i in finset.range k, 1 / (a i) : ℝ) = 1)
  (ha2 : ∑ i in finset.range k, a i = n) :
  ∃ b : ℕ → ℕ, 
    (∑ i in finset.range (k + 3), 1 / (b i) : ℝ) = 1 
    ∧ (∑ i in finset.range (k + 3), b i = 42 * n + 12) := 
sorry

end k_solvable_implies_k_plus_3_solvable_l531_531159


namespace dice_rolls_diff_by_3_probability_l531_531458

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l531_531458


namespace construct_triangle_from_bisectors_l531_531093

theorem construct_triangle_from_bisectors
  (M N P : Point)
  (circumcircle : Circle)
  (H : are_angle_bisector_intersections M N P (triangle_circumcircle circumcircle)) :
  ∃ (A B C : Point), 
    is_triangle A B C ∧
    are_altitude_intersections_with_circumcircle_extended_triangle A B C M N P (circumcircle) :=
by
  sorry

end construct_triangle_from_bisectors_l531_531093


namespace boat_speed_5_kmh_l531_531030

noncomputable def boat_speed_in_still_water (V_s : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
  (d / t) - V_s

theorem boat_speed_5_kmh :
  boat_speed_in_still_water 5 10 100 = 5 :=
by
  sorry

end boat_speed_5_kmh_l531_531030


namespace average_of_first_15_even_numbers_is_16_l531_531013

-- Define the sum of the first 15 even numbers
def sum_first_15_even_numbers : ℕ :=
  2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30

-- Define the average of the first 15 even numbers
def average_of_first_15_even_numbers : ℕ :=
  sum_first_15_even_numbers / 15

-- Prove that the average is equal to 16
theorem average_of_first_15_even_numbers_is_16 : average_of_first_15_even_numbers = 16 :=
by
  -- Sorry placeholder for the proof
  sorry

end average_of_first_15_even_numbers_is_16_l531_531013


namespace inequality_proof_l531_531978

noncomputable def a : Real := (1 / 3) ^ Real.pi
noncomputable def b : Real := (1 / 3) ^ (1 / 2 : Real)
noncomputable def c : Real := Real.pi ^ (1 / 2 : Real)

theorem inequality_proof : a < b ∧ b < c :=
by
  -- Proof will be provided here
  sorry

end inequality_proof_l531_531978


namespace units_digit_27_mul_46_l531_531552

-- Define the function to calculate the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement: The units digit of 27 * 46 is 2
theorem units_digit_27_mul_46 : units_digit (27 * 46) = 2 :=
  sorry

end units_digit_27_mul_46_l531_531552


namespace incorrect_temperature_relation_l531_531325

theorem incorrect_temperature_relation 
  (t : ℕ → ℤ)
  (h_values : list ℕ)
  (t_values : list ℤ)
  (table : ∀ (n : ℕ), n < 6 → (h_values.nth n, t_values.nth n) = (some n, some (20 - 6 * n))) :
  ¬(∀ (n : ℕ), n < 6 → t n = 20 - 5 * n) :=
by {
  -- Ensure h_values and t_values match the given altitude and temperature pairs
  have altitudes_match : h_values = [0, 1, 2, 3, 4, 5] := by rfl,
  have temperatures_match : t_values = [20, 14, 8, 2, -4, -10] := by rfl,
  sorry
}

end incorrect_temperature_relation_l531_531325


namespace find_number_l531_531887

def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def first_digit_is_three (n : ℕ) : Prop :=
  n / 1000 = 3

def last_digit_is_five (n : ℕ) : Prop :=
  n % 10 = 5

theorem find_number :
  ∃ (x : ℕ), four_digit_number (x^2) ∧ first_digit_is_three (x^2) ∧ last_digit_is_five (x^2) ∧ x = 55 :=
sorry

end find_number_l531_531887


namespace midpoint_correct_l531_531157

-- Define the points A and B
def A : ℝ × ℝ × ℝ := (3, 2, 1)
def B : ℝ × ℝ × ℝ := (1, -2, 5)

-- Define the theorem stating the midpoint of segment AB is (2, 0, 3)
theorem midpoint_correct :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)
  in M = (2, 0, 3) :=
by
  -- Skip the proof
  sorry

end midpoint_correct_l531_531157


namespace car_travel_hours_l531_531075

example (speed : ℕ) (miles_per_gallon : ℕ) (tank_capacity : ℕ) (fraction_used : ℚ) : ℕ :=
  let gasoline_used := fraction_used * tank_capacity
  let distance_traveled := gasoline_used * miles_per_gallon
  let travel_hours := distance_traveled / speed
  travel_hours

-- Given values
variable (speed := 40) (miles_per_gallon := 40) (tank_capacity := 12) (fraction_used : ℚ := 0.4166666666666667)

theorem car_travel_hours : example speed miles_per_gallon tank_capacity fraction_used = 5 :=
by
  sorry

end car_travel_hours_l531_531075


namespace remove_terms_l531_531923

-- Define the fractions
def f1 := 1 / 3
def f2 := 1 / 6
def f3 := 1 / 9
def f4 := 1 / 12
def f5 := 1 / 15
def f6 := 1 / 18

-- Define the total sum
def total_sum := f1 + f2 + f3 + f4 + f5 + f6

-- Define the target sum after removal
def target_sum := 2 / 3

-- Define the condition to be proven
theorem remove_terms {x y : Real} (h1 : (x = f4) ∧ (y = f5)) : 
  total_sum - (x + y) = target_sum := by
  sorry

end remove_terms_l531_531923


namespace divisible_by_12_for_all_integral_n_l531_531102

theorem divisible_by_12_for_all_integral_n (n : ℤ) : 12 ∣ (2 * n ^ 3 - 2 * n) :=
sorry

end divisible_by_12_for_all_integral_n_l531_531102


namespace sasha_remainder_20_l531_531775

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l531_531775


namespace sum_of_intersection_coordinates_l531_531895

noncomputable def h : ℝ → ℝ := sorry

theorem sum_of_intersection_coordinates : 
  (∃ a b : ℝ, h a = h (a + 2) ∧ h 1 = 3 ∧ h (-1) = 3 ∧ a = -1 ∧ b = 3) → -1 + 3 = 2 :=
by
  intro h_assumptions
  sorry

end sum_of_intersection_coordinates_l531_531895


namespace max_profit_at_800_l531_531400

open Nat

def P (x : ℕ) : ℝ :=
  if h : 0 < x ∧ x ≤ 100 then 80
  else if h : 100 < x ∧ x ≤ 1000 then 82 - 0.02 * x
  else 0

def f (x : ℕ) : ℝ :=
  if h : 0 < x ∧ x ≤ 100 then 30 * x
  else if h : 100 < x ∧ x ≤ 1000 then 32 * x - 0.02 * x^2
  else 0

theorem max_profit_at_800 :
  ∀ x : ℕ, f x ≤ 12800 ∧ f 800 = 12800 :=
sorry

end max_profit_at_800_l531_531400


namespace units_digit_27_mul_46_l531_531554

-- Define the function to calculate the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement: The units digit of 27 * 46 is 2
theorem units_digit_27_mul_46 : units_digit (27 * 46) = 2 :=
  sorry

end units_digit_27_mul_46_l531_531554


namespace dice_diff_by_three_probability_l531_531435

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l531_531435


namespace number_of_questions_in_test_l531_531331

theorem number_of_questions_in_test (x : ℕ) (sections questions_correct : ℕ)
  (h_sections : sections = 5)
  (h_questions_correct : questions_correct = 32)
  (h_percentage : 0.70 < (questions_correct : ℚ) / x ∧ (questions_correct : ℚ) / x < 0.77) 
  (h_multiple_of_sections : x % sections = 0) : 
  x = 45 :=
sorry

end number_of_questions_in_test_l531_531331


namespace polynomial_root_conditions_l531_531725

theorem polynomial_root_conditions (a b : ℤ) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
  (h₃ : ∃ r s : ℤ, (x^3 + a * x^2 + b * x + 16 * a) = (x - r)^2 * (x - s) ∧ r * r * s = -16 * a ∧ (r^2 + 2 * r * s) = b) : 
  |a * b| = 5832 :=
sorry

end polynomial_root_conditions_l531_531725


namespace range_of_omega_l531_531573

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ (a b : ℝ), a ≠ b ∧ 0 ≤ a ∧ a ≤ π/2 ∧ 0 ≤ b ∧ b ≤ π/2 ∧ f ω a + f ω b = 4) ↔ 5 ≤ ω ∧ ω < 9 :=
sorry

end range_of_omega_l531_531573


namespace domain_of_sqrt_function_l531_531225

theorem domain_of_sqrt_function : ∀ (x : ℝ), (∃ y : ℝ, y = sqrt (2 * x - 1)) ↔ (x ≥ (1/2)) :=
by sorry

end domain_of_sqrt_function_l531_531225


namespace concyclic_AH_PQ_l531_531681

variables (ABC : Triangle)
variables (O H A P Q A' : Point)
variables (circumcenter : IsCircumcenter O ABC)
variables (orthocenter : IsOrthocenter H ABC)
variables (reflection : ReflectsPoint A P OH)
variables (concurrent : Concur AQ OH BC)
variables (parallelogram : IsParallelogram ABA'C)

theorem concyclic_AH_PQ :
  Concyclic A' H P Q :=
sorry

end concyclic_AH_PQ_l531_531681


namespace fraction_of_married_men_l531_531892

theorem fraction_of_married_men (prob_single_woman : ℚ) (H : prob_single_woman = 3 / 7) :
  ∃ (fraction_married_men : ℚ), fraction_married_men = 4 / 11 :=
by
  -- Further proof steps would go here if required
  sorry

end fraction_of_married_men_l531_531892


namespace sasha_remainder_l531_531771

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l531_531771


namespace josette_additional_cost_l531_531679

def small_bottle_cost_eur : ℝ := 1.50
def large_bottle_cost_eur : ℝ := 2.40
def exchange_rate : ℝ := 1.20
def discount_10_percent : ℝ := 0.10
def discount_15_percent : ℝ := 0.15

def initial_small_bottles : ℕ := 3
def initial_large_bottles : ℕ := 2

def initial_total_cost_eur : ℝ :=
  (small_bottle_cost_eur * initial_small_bottles) +
  (large_bottle_cost_eur * initial_large_bottles)

def discounted_cost_eur_10 : ℝ :=
  initial_total_cost_eur * (1 - discount_10_percent)

def additional_bottle_cost_eur : ℝ := small_bottle_cost_eur

def new_total_cost_eur : ℝ :=
  initial_total_cost_eur + additional_bottle_cost_eur

def discounted_cost_eur_15 : ℝ :=
  new_total_cost_eur * (1 - discount_15_percent)

def cost_usd (eur_amount : ℝ) : ℝ :=
  eur_amount * exchange_rate

def discounted_cost_usd_10 : ℝ := cost_usd discounted_cost_eur_10
def discounted_cost_usd_15 : ℝ := cost_usd discounted_cost_eur_15

def additional_cost_usd : ℝ :=
  discounted_cost_usd_15 - discounted_cost_usd_10

theorem josette_additional_cost :
  additional_cost_usd = 0.972 :=
by 
  sorry

end josette_additional_cost_l531_531679


namespace probability_diff_by_3_l531_531482

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l531_531482


namespace sasha_remainder_l531_531763

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l531_531763


namespace proof_tan_alpha_proof_g_alpha_proof_sum_beta_gamma_l531_531162

variables 
  (α β γ m : Real)
  (h1 : 3 * Real.pi / 4 < α ∧ α < Real.pi)
  (h2 : tan α + 1 / tan α = -10 / 3)
  (h3 : β > 0 ∧ β < Real.pi / 2)
  (h4 : γ > 0 ∧ γ < Real.pi / 2)
  (h5 : tan γ = sqrt 3 * (m - 3 * tan α))
  (h6 : sqrt 3 * (tan γ * tan β + m) + tan β = 0)

theorem proof_tan_alpha : tan α = -1 / 3 :=
sorry

noncomputable def g (α : Real) : Real :=
  (sin (Real.pi + α) + 4 * cos (2 * Real.pi - α)) / (sin (Real.pi / 2 - α) - 4 * sin (-α))

theorem proof_g_alpha : g α = -13 :=
sorry

theorem proof_sum_beta_gamma : β + γ = Real.pi / 3 :=
sorry

end proof_tan_alpha_proof_g_alpha_proof_sum_beta_gamma_l531_531162


namespace range_of_a_l531_531601

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 :=
sorry

end range_of_a_l531_531601


namespace cylinder_volume_ratio_l531_531918

noncomputable def volume_cylinder (r h : ℝ) := π * r^2 * h

theorem cylinder_volume_ratio (r h : ℝ) (h𝓪 : r ≠ 0) (h𝓫 : h ≠ 0)
  (height_eq : hₐ = hₑ) (radius_eq : rₐ = rₑ)
  (volume_relation : volume_cylinder rₐ hₐ = 3 * volume_cylinder rₑ hₑ) :
  ∃ N : ℝ, volume_cylinder rₐ hₐ = N * π * r^3 ∧ N = (1 / 3) :=
by
  use 1 / 3
  simp [volume_cylinder, height_eq, radius_eq] at *
  sorry

end cylinder_volume_ratio_l531_531918


namespace biker_bob_rides_west_distance_l531_531894

theorem biker_bob_rides_west_distance :
  ∃ x : ℝ, (20 * 20 + (x - 4) * (x - 4) = 20.396078054371138 * 20.396078054371138) ∧ x = 8 := 
by 
  -- Define the conditions
  let d_north : ℝ := 20
  let d_direct : ℝ := 20.396078054371138

  -- Existential quantifier for x
  existsi (8 : ℝ)

  -- Apply the conditions
  show 20 * 20 + (8 - 4) * (8 - 4) = d_direct * d_direct, by sorry
  show 8 = 8, by rfl

end biker_bob_rides_west_distance_l531_531894


namespace parabola_focus_distance_l531_531616

-- defining the problem in Lean
theorem parabola_focus_distance
  (A : ℝ × ℝ)
  (hA : A.2^2 = 4 * A.1)
  (h_distance : |A.1| = 3)
  (F : ℝ × ℝ)
  (hF : F = (1, 0)) :
  |(A.1 - F.1)^2 + (A.2 - F.2)^2| = 4 := 
sorry

end parabola_focus_distance_l531_531616


namespace probability_of_differ_by_three_l531_531496

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l531_531496


namespace find_radius_of_sector_l531_531376

noncomputable def radius_of_sector (P : ℝ) (θ : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem find_radius_of_sector :
  radius_of_sector 144 180 = 144 / (Real.pi + 2) :=
by
  unfold radius_of_sector
  sorry

end find_radius_of_sector_l531_531376


namespace interval_length_difference_l531_531141

variable {x a b : ℝ}

def length_of_interval (x1 x2 : ℝ) : ℝ := x2 - x1

theorem interval_length_difference (h1 : x1 < x2)
                                    (h_dom : ∀ x, a ≤ x ∧ x ≤ b → 2^real.abs x ∈ Set.Icc 1 2)
                                    (h_range : ∀ y, y ∈ Set.Icc 1 2 → ∃ x, a ≤ x ∧ x ≤ b ∧ 2^real.abs x = y)
                                    (h_len_max : length_of_interval (-1 : ℝ) (1)) 
                                    (h_len_min : length_of_interval (0 : ℝ) (1)) :
                                    ∀ a b, 1 = 2 - 1 :=
by
  intros a b h1 h_dom h_range h_len_max h_len_min
  sorry

end interval_length_difference_l531_531141


namespace gcf_lcm_sum_l531_531683

open Nat

/-- The greatest common factor (GCF) of 8, 12, and 24 -/
def gcf (a b c : ℕ) : ℕ := gcd a (gcd b c)

/-- The least common multiple (LCM) of 8, 12, and 24 -/
def lcm (a b c : ℕ) : ℕ := let lcm_ab := lcm a b in lcm lcm_ab c

theorem gcf_lcm_sum :
  let A := gcf 8 12 24
  let B := lcm 8 12 24
  A + B = 28 :=
by sorry

end gcf_lcm_sum_l531_531683


namespace intervals_of_monotonicity_and_extrema_f_g_inequality_sum_greater_than_4_l531_531605

noncomputable def f (x : ℝ) := (x - 1) / real.exp x
noncomputable def g (x : ℝ) := f (4 - x)

theorem intervals_of_monotonicity_and_extrema :
  (∀ x : ℝ, x < 2 → f.deriv x > 0) ∧
  (∀ x : ℝ, x > 2 → f.deriv x < 0) ∧
  (∀ x : ℝ, x = 2 → f.deriv x = 0) ∧
  (f 2 = 1 / real.exp 2) :=
sorry

theorem f_g_inequality (x : ℝ) (h : x > 2) : f x > g x :=
sorry

theorem sum_greater_than_4 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f x1 = f x2) : x1 + x2 > 4 :=
sorry

end intervals_of_monotonicity_and_extrema_f_g_inequality_sum_greater_than_4_l531_531605


namespace binary_to_base4_representation_l531_531352

def binary_to_base4 (n : ℕ) : ℕ :=
  -- Assuming implementation that converts binary number n to its base 4 representation 
  sorry

theorem binary_to_base4_representation :
  binary_to_base4 0b10110110010 = 23122 :=
by sorry

end binary_to_base4_representation_l531_531352


namespace units_digit_27_mul_46_l531_531551

-- Define the function to calculate the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement: The units digit of 27 * 46 is 2
theorem units_digit_27_mul_46 : units_digit (27 * 46) = 2 :=
  sorry

end units_digit_27_mul_46_l531_531551


namespace find_n_variations_equation_l531_531564

theorem find_n_variations_equation : 
  ∃ (n : ℕ), n^3 = n * (n - 1) * (n - 2) + 225 ∧ n = 9 :=
begin
  sorry
end

end find_n_variations_equation_l531_531564


namespace radius_large_circle_l531_531221

theorem radius_large_circle {A B C : Type}
  (side1 : real) (side2 : real) (side3 : real)
  (h_triangle : side1 = 6 ∧ side2 = 8 ∧ side3 = 10):
  ∃ (r : real), r = 144 / 23 :=
by
  sorry

end radius_large_circle_l531_531221


namespace part_1_part_2a_part_2b_l531_531602

variable {a b : ℕ → ℝ}
variable {S T : ℕ → ℝ}
variable {a1 a2 : ℕ → ℝ}
variable {b1 b2 : ℕ → ℝ}
variable {q : ℝ}

-- Definitions given in the problem
def arithmetic_seq (a : ℕ → ℝ) := ∀ n, a 1 = -1
def geometric_seq (b : ℕ → ℝ) := ∀ n, b 1 = 1
def sum_arith_seq (S : ℕ → ℝ) := ∀ n, S (n + 1) = S n + a (n + 1)
def sum_geom_seq (T : ℕ → ℝ) := ∀ n, T n = ∑ i in range n, b (i + 1)

-- Conditions in the problem
def condition_1 := a 1 + b 1 = 5
def condition_2 := a 2 + b 2 = 2
def condition_3 := T 3 = 21

-- Goals to prove
theorem part_1 (h1 : arithmetic_seq a) (h2 : geometric_seq b) (h3 : sum_geom_seq T) (cond1 : condition_1) (cond2 : condition_2) :
  (b n = 2 ^ (n - 1)) :=
sorry

theorem part_2a (h1 : arithmetic_seq a) (h2 : geometric_seq b) (h3 : sum_geom_seq T) (h4 : sum_arith_seq S) (cond3 : condition_3) :
  q = 4 → S 3 = -6 :=
sorry

theorem part_2b (h1 : arithmetic_seq a) (h2 : geometric_seq b) (h3 : sum_geom_seq T) (h4 : sum_arith_seq S) (cond3 : condition_3) :
  q = -5 → S 3 = 21 :=
sorry

end part_1_part_2a_part_2b_l531_531602


namespace measure_of_angle_A_l531_531208

-- Define the conditions as assumptions
variable (B : Real) (angle1 angle2 A : Real)
-- Angle B is 120 degrees
axiom h1 : B = 120
-- One of the angles formed by the dividing line is 50 degrees
axiom h2 : angle1 = 50
-- Angles formed sum up to 180 degrees as they are supplementary
axiom h3 : angle2 = 180 - angle1
-- Vertical angles are equal
axiom h4 : A = angle2

theorem measure_of_angle_A (B angle1 angle2 A : Real) 
    (h1 : B = 120) (h2 : angle1 = 50) (h3 : angle2 = 180 - angle1) (h4 : A = angle2) : A = 130 := 
by
    sorry

end measure_of_angle_A_l531_531208


namespace sedrich_more_jelly_beans_l531_531268

-- Define the given conditions
def napoleon_jelly_beans : ℕ := 17
def mikey_jelly_beans : ℕ := 19
def sedrich_jelly_beans (x : ℕ) : ℕ := napoleon_jelly_beans + x

-- Define the main theorem to be proved
theorem sedrich_more_jelly_beans (x : ℕ) :
  2 * (napoleon_jelly_beans + sedrich_jelly_beans x) = 4 * mikey_jelly_beans → x = 4 :=
by
  -- Proving the theorem
  sorry

end sedrich_more_jelly_beans_l531_531268


namespace coeff_x3_in_product_l531_531546

open Polynomial

noncomputable def p : Polynomial ℤ := 3 * X^3 + 2 * X^2 + 5 * X + 3
noncomputable def q : Polynomial ℤ := 4 * X^3 + 5 * X^2 + 6 * X + 8

theorem coeff_x3_in_product :
  (p * q).coeff 3 = 61 :=
by sorry

end coeff_x3_in_product_l531_531546


namespace star_perimeter_l531_531246

theorem star_perimeter (R : ℝ) {A B C D E : Point}
  (h_inscribed : InCircle {A, B, C, D, E})
  (h_ap : sides_are_arithmetic_progression A B C D E)
  (h_perimeter : perimeter A B C D E = 1) :
  perimeter_star A B C D E = 5 * R * Real.sin (36 * Real.pi / 180) :=
sorry

end star_perimeter_l531_531246


namespace largest_angle_in_triangles_l531_531362

theorem largest_angle_in_triangles (angles : Fin 9 → ℝ) (d : ℝ)
  (h_seq : ∀ i j, i < j → angles j - angles i = d * (j - i))
  (h_angle_42 : ∃ i, angles i = 42) :
  set.mem (78 : ℝ) (set.range (largest angles)) ∨
  set.mem (84 : ℝ) (set.range (largest angles)) ∨
  set.mem (96 : ℝ) (set.range (largest angles)) :=
begin
  sorry
end

def largest (angles : Fin 9 → ℝ) : ℝ :=
  finset.max' (finset.univ.image angles) (finset.univ_nonempty.image angles)

end largest_angle_in_triangles_l531_531362


namespace samantha_eggs_left_l531_531715

variables (initial_eggs : ℕ) (total_cost price_per_egg : ℝ)

-- Conditions
def samantha_initial_eggs : initial_eggs = 30 := sorry
def samantha_total_cost : total_cost = 5 := sorry
def samantha_price_per_egg : price_per_egg = 0.20 := sorry

-- Theorem to prove:
theorem samantha_eggs_left : 
  initial_eggs - (total_cost / price_per_egg) = 5 := 
  by
  rw [samantha_initial_eggs, samantha_total_cost, samantha_price_per_egg]
  -- Completing the arithmetic proof
  rw [Nat.cast_sub (by norm_num), Nat.cast_div (by norm_num), Nat.cast_mul (by norm_num)]
  norm_num
  sorry

end samantha_eggs_left_l531_531715


namespace probability_of_differ_by_three_l531_531503

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l531_531503


namespace product_eq_one_of_log_abs_eq_l531_531958

theorem product_eq_one_of_log_abs_eq {a b : ℝ} (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a ≠ b) (h₄ : |log a| = |log b|) : a * b = 1 :=
by
  sorry

end product_eq_one_of_log_abs_eq_l531_531958


namespace find_a_b_l531_531973

section
variables (a b : ℝ)

-- Define the initial point M and the transformed point N
def M : ℝ × ℝ := (3, -1)
def N : ℝ × ℝ := (3, 5)

-- Define the rotation matrix for a 90-degree counterclockwise rotation
def rotation_matrix : matrix (fin 2) (fin 2) ℝ :=
  !![ 0, -1;
      1,  0]

-- Define the transformation matrix A
def matrix_A (a b : ℝ) : matrix (fin 2) (fin 2) ℝ :=
  !![ a, 0;
      2, b]

-- Define the combined transformation: rotation followed by matrix A
def combined_transform (m : matrix (fin 2) (fin 2) ℝ) (M : ℝ × ℝ) : ℝ × ℝ :=
  (m.mul_vec (λ i, ![M.1, M.2]) 0, m.mul_vec (λ i, ![M.1, M.2]) 1)

-- The theorem to prove
theorem find_a_b (hab : combined_transform (matrix_A a b ⬝ rotation_matrix) M = N) : a = 3 ∧ b = 1 :=
by sorry
end

end find_a_b_l531_531973


namespace selling_price_correct_l531_531054

def cost_price : ℝ := 975
def profit_percentage : ℝ := 20 / 100
def profit_amount := profit_percentage * cost_price
def expected_selling_price := cost_price + profit_amount

theorem selling_price_correct :
  expected_selling_price = 1170 := 
by sorry

end selling_price_correct_l531_531054


namespace probability_diff_by_three_l531_531471

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l531_531471


namespace identify_heaviest_and_lightest_13_weighings_l531_531818

theorem identify_heaviest_and_lightest_13_weighings (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ f : (Fin 13 → ((Fin 10) × (Fin 10) × ℝ)), true :=
by
  sorry

end identify_heaviest_and_lightest_13_weighings_l531_531818


namespace perpendicular_points_l531_531136

/-- Given points lying on specific lines and having specific ratios, prove that FH is perpendicular to EG -/
theorem perpendicular_points
  (A B C D E F G H : Type)
  [HasOrder A B E]
  [HasOrder C D G]
  (AE_EB_eq_AF_FB : AE / EB = AF / FB)
  (AF_FB_eq_DG_GC : AF / FB = DG / GC)
  (DG_GC_eq_DH_HC : DG / GC = DH / HC)
  (DH_HC_eq_AD_BC : DH / HC = AD / BC) :
  FH ⊥ EG :=
sorry

end perpendicular_points_l531_531136


namespace product_modulo_7_l531_531085

theorem product_modulo_7 : 
  (2007 % 7 = 4) ∧ (2008 % 7 = 5) ∧ (2009 % 7 = 6) ∧ (2010 % 7 = 0) →
  (2007 * 2008 * 2009 * 2010) % 7 = 0 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end product_modulo_7_l531_531085


namespace number_of_paths_A_to_D_l531_531561

-- Definition of conditions
def ways_A_to_B : Nat := 2
def ways_B_to_C : Nat := 2
def ways_C_to_D : Nat := 2
def direct_A_to_D : Nat := 1

-- Theorem statement for the total number of paths from A to D
theorem number_of_paths_A_to_D : ways_A_to_B * ways_B_to_C * ways_C_to_D + direct_A_to_D = 9 := by
  sorry

end number_of_paths_A_to_D_l531_531561


namespace sasha_remainder_is_20_l531_531791

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l531_531791


namespace odd_or_even_property_l531_531181

def f (a b x : ℝ) : ℝ := 2 / (a^x - 1) + b

theorem odd_or_even_property (a b : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  (∀ x : ℝ, x ≠ 0 → f a b (-x) + f a b x = 0 ↔ b = 1) ∧
  (¬(∀ x : ℝ, x ≠ 0 → f a b (-x) = f a b x)) :=
sorry

end odd_or_even_property_l531_531181


namespace profit_share_difference_l531_531411

theorem profit_share_difference (P : ℝ) (hP : P = 1000) 
  (rX rY : ℝ) (hRatio : rX / rY = (1/2) / (1/3)) : 
  let total_parts := (1/2) + (1/3)
  let value_per_part := P / total_parts
  let x_share := (1/2) * value_per_part
  let y_share := (1/3) * value_per_part
  x_share - y_share = 200 := by 
  sorry

end profit_share_difference_l531_531411


namespace least_possible_value_of_z_minus_x_l531_531646

theorem least_possible_value_of_z_minus_x 
  (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : y - x > 5) : 
  z - x = 9 :=
sorry

end least_possible_value_of_z_minus_x_l531_531646


namespace find_divisor_l531_531369

-- Defining the conditions
def dividend : ℕ := 181
def quotient : ℕ := 9
def remainder : ℕ := 1

-- The statement to prove
theorem find_divisor : ∃ (d : ℕ), dividend = (d * quotient) + remainder ∧ d = 20 := by
  sorry

end find_divisor_l531_531369


namespace difference_between_mean_and_median_l531_531272

def percentage_students (p70 p80 p90 p100 : ℝ) : Prop :=
  p70 + p80 + p90 + p100 = 1

def median_score (p70 p80 p90 p100 : ℝ) (s70 s80 s90 s100 : ℕ) : ℕ :=
  if (p70 + p80) < 0.5 then s100
  else if (p70 < 0.5) then s90 
  else if (p70 > 0.5) then s70
  else s80

def mean_score (p70 p80 p90 p100 : ℝ) (s70 s80 s90 s100 : ℕ) : ℝ :=
  p70 * s70 + p80 * s80 + p90 * s90 + p100 * s100

theorem difference_between_mean_and_median
  (p70 p80 p90 p100 : ℝ)
  (h_sum : percentage_students p70 p80 p90 p100)
  (s70 s80 s90 s100 : ℕ) :
  median_score p70 p80 p90 p100 s70 s80 s90 s100 - mean_score p70 p80 p90 p100 s70 s80 s90 s100 = 3 :=
  sorry

end difference_between_mean_and_median_l531_531272


namespace monotonic_has_at_most_one_solution_l531_531987

def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y ∨ f y ≤ f x

theorem monotonic_has_at_most_one_solution (f : ℝ → ℝ) (c : ℝ) 
  (hf : monotonic f) : ∃! x : ℝ, f x = c :=
sorry

end monotonic_has_at_most_one_solution_l531_531987


namespace contradictory_statement_of_p_l531_531922

-- Given proposition p
def p : Prop := ∀ (x : ℝ), x + 3 ≥ 0 → x ≥ -3

-- Contradictory statement of p
noncomputable def contradictory_p : Prop := ∀ (x : ℝ), x + 3 < 0 → x < -3

-- Proof statement
theorem contradictory_statement_of_p : contradictory_p :=
sorry

end contradictory_statement_of_p_l531_531922


namespace probability_of_diff_3_is_1_over_9_l531_531429

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l531_531429


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531442

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l531_531442


namespace min_distance_MN_l531_531984

-- Define the line 3x + 4y - 2 = 0
def line (M : ℝ × ℝ) : Prop := 3 * M.1 + 4 * M.2 - 2 = 0

-- Define the circle (x + 1)^2 + (y + 1)^2 = 1
def circle (N : ℝ × ℝ) : Prop := (N.1 + 1) ^ 2 + (N.2 + 1) ^ 2 = 1

-- Define the distance function from a point to a line
def distance_point_line (P : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  abs (A * P.1 + B * P.2 + C) / real.sqrt (A ^ 2 + B ^ 2)

-- Define the distance between two points
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Define the center of the circle
def center : ℝ × ℝ := (-1, -1)

-- Define the minimum value calculation
def min_MN : ℝ :=
  distance_point_line center 3 4 (-2) - 1

-- The theorem asserting the minimum distance
theorem min_distance_MN :
  ∃ M N : ℝ × ℝ, line M ∧ circle N ∧ distance M N = 4 / 5 :=
by {
  -- skip the proof
  sorry
}

end min_distance_MN_l531_531984


namespace area_of_inscribed_hexagon_in_square_is_27sqrt3_l531_531031

noncomputable def side_length_of_triangle : ℝ := 6
noncomputable def radius_of_circle (a : ℝ) : ℝ := (a * Real.sqrt 2) / 2
noncomputable def side_length_of_square (r : ℝ) : ℝ := 2 * r
noncomputable def side_length_of_hexagon_in_square (s : ℝ) : ℝ := s / (Real.sqrt 2)
noncomputable def area_of_hexagon (side_hexagon : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * side_hexagon^2

theorem area_of_inscribed_hexagon_in_square_is_27sqrt3 :
  ∀ (a r s side_hex : ℝ), 
    a = side_length_of_triangle →
    r = radius_of_circle a →
    s = side_length_of_square r →
    side_hex = side_length_of_hexagon_in_square s →
    area_of_hexagon side_hex = 27 * Real.sqrt 3 :=
by
  intros a r s side_hex h_a h_r h_s h_side_hex
  sorry

end area_of_inscribed_hexagon_in_square_is_27sqrt3_l531_531031


namespace bronson_cost_per_bushel_is_12_l531_531080

noncomputable def cost_per_bushel 
  (sale_price_per_apple : ℝ := 0.40)
  (apples_per_bushel : ℕ := 48)
  (profit_from_100_apples : ℝ := 15)
  (number_of_apples_sold : ℕ := 100) 
  : ℝ :=
  let revenue := number_of_apples_sold * sale_price_per_apple
  let cost := revenue - profit_from_100_apples
  let number_of_bushels := (number_of_apples_sold : ℝ) / apples_per_bushel
  cost / number_of_bushels

theorem bronson_cost_per_bushel_is_12 :
  cost_per_bushel = 12 :=
by
  sorry

end bronson_cost_per_bushel_is_12_l531_531080


namespace max_value_y_l531_531256

theorem max_value_y : ∀ x : ℝ, -2 * x^2 + 8 ≤ 8 :=
by
  intro x
  calc
    -2 * x^2 + 8 ≤ 8 : sorry

end max_value_y_l531_531256


namespace depth_finite_x_mod_6_x_difference_mod_2016_l531_531098

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def depth (N : ℕ) : ℕ :=
  if N < 10 then 0 else 1 + depth (sum_of_digits N)

def x (n : ℕ) : ℕ :=
  if n = 1 then 10 else if n = 2 then 19 else if n = 3 then 199 else
  1 * 10 ^ (n - 1) + (List.replicate (n - 1) 9 |>.foldl (λ acc d, acc * 10 + d) 0)

theorem depth_finite (N : ℕ) (h : 0 < N) :
  ∃ k, depth N = k ∧ k < 10 :=
by sorry

theorem x_mod_6 (h : 5776 = 5776) : x 5776 % 6 = 4 :=
by sorry

theorem x_difference_mod_2016 (h1 : 5776 = 5776) (h2 : 5708 = 5708) :
  (x 5776 - x 5708) % 2016 = 0 :=
by sorry

end depth_finite_x_mod_6_x_difference_mod_2016_l531_531098


namespace identify_heaviest_and_lightest_l531_531813

theorem identify_heaviest_and_lightest (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ weighings : Fin 13 → (Fin 10 × Fin 10),
  (let outcomes := fun w ℕ => ite (coins (weighings w).fst > coins (weighings w).snd) (weighings w).fst (weighings w).snd,
  max_coin := nat.rec_on 12 (outcomes 0) (λ n max_n, if coins (outcomes (succ n)) > coins max_n then outcomes (succ n) else max_n),
  min_coin := nat.rec_on 12 (outcomes 0) (λ n min_n, if coins (outcomes (succ n)) < coins min_n then outcomes (succ n) else min_n))
  (∃ max_c : Fin 10, ∃ min_c : Fin 10, max_c ≠ min_c ∧ max_c = Some max_coin ∧ min_c = Some min_coin) :=
sorry

end identify_heaviest_and_lightest_l531_531813


namespace simplify_tan_cot_expression_l531_531294

theorem simplify_tan_cot_expression
  (h1 : Real.tan (Real.pi / 4) = 1)
  (h2 : Real.cot (Real.pi / 4) = 1) :
  (Real.tan (Real.pi / 4))^3 + (Real.cot (Real.pi / 4))^3 = 1 := by
  sorry

end simplify_tan_cot_expression_l531_531294


namespace sasha_remainder_l531_531758

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l531_531758


namespace number_of_zeros_of_derivative_f_ge_a_two_minus_ln_a_l531_531999

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * Real.log x

theorem number_of_zeros_of_derivative (a : ℝ) :
  if a = 0 then 0 else if a > 0 then 1 else 0 :=
sorry

theorem f_ge_a_two_minus_ln_a (a x : ℝ) (h : a > 0) (hx : x > 0) :
  f x a ≥ a * (2 - Real.log a) :=
sorry

end number_of_zeros_of_derivative_f_ge_a_two_minus_ln_a_l531_531999


namespace function_classification_l531_531379

theorem function_classification (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x - y) = f(x) * f(y)) →
  (∀ x : ℝ, f(x) = 0) ∨ (∀ x : ℝ, f(x) = 1) :=
by
  -- we are skipping the proof here
  sorry

end function_classification_l531_531379


namespace trapezoid_ratio_l531_531729

theorem trapezoid_ratio 
  (ABCD : Prop)  -- Declaring ABCD as a proposition representing trapezoid
  (BC AD : ℝ)    -- Declaring BC and AD as real numbers representing bases
  (h : ℝ)        -- Declaring h as real number representing height of triangles and trapezoid
  (S_BCD : ℝ)    -- Declaring S_BCD as real number representing area of ΔBCD
  (S_ABD : ℝ)    -- Declaring S_ABD as real number representing area of ΔABD
  (ratio_diag : S_BCD / S_ABD = 3 / 7) -- Given the diagonal divides the area ratio of the triangles
  (ratio_base : BC / AD = 3 / 7) -- Resulting ratio of bases from area condition
  (AK KD : ℝ) -- Declaring AK and KD as real numbers representing segments on AD
  (CK // AB : Prop) -- Declaring CK parallel to AB

  : ((AK / (KD = AK + KD = AD - BC)) = (3 / 4)) -- Intermediate segment ratios
    → (AK / KD = 3 / 4) -- Final segment ratio
    -- Prove the area ratio between ABCK and CKD
    → (3 / 2)
    -- Thus
    = (3 / 2) :=
by
  sorry -- Skipping proof

end trapezoid_ratio_l531_531729


namespace tech_investment_is_correct_future_investment_value_l531_531345

noncomputable def investment_in_tech := 
  let total_investment : ℝ := 250000
  let ratio : ℝ := 6
  total_investment * ratio / (1 + ratio)

theorem tech_investment_is_correct :
  investment_in_tech = 214286 := by
  let total_investment : ℝ := 250000
  let r := total_investment / 7
  let tech_investment := 6 * r
  have h : r = total_investment / 7 := rfl
  have h_tech : tech_investment = 6 * (total_investment / 7) := rfl
  rw [←h_tech]
  have h2 : 6 * (total_investment / 7) = 214286 := sorry
  rw [h2]

def future_value (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem future_investment_value :
  future_value 250000 0.05 3 = 289406 := by
  let P : ℝ := 250000
  let r : ℝ := 0.05
  let n : ℕ := 3
  have h : future_value P r n = P * (1 + r)^n := rfl
  have h_value : P * (1 + r)^n = 289406 := sorry
  rw [h_value]

end tech_investment_is_correct_future_investment_value_l531_531345


namespace circle_tangent_line_l531_531121

theorem circle_tangent_line 
    (center : ℝ × ℝ) (line_eq : ℝ → ℝ → ℝ) 
    (tangent_eq : ℝ) :
    center = (-1, 1) →
    line_eq 1 (-1)= 0 →
    tangent_eq = 2 :=
  let h := -1;
  let k := 1;
  let radius := Real.sqrt 2;
  sorry

end circle_tangent_line_l531_531121


namespace chewbacca_gum_packs_l531_531521

theorem chewbacca_gum_packs (x : ℕ) :
  (30 - 2 * x) * (40 + 4 * x) = 1200 → x = 5 :=
by
  -- This is where the proof would go. We'll leave it as sorry for now.
  sorry

end chewbacca_gum_packs_l531_531521


namespace monotonic_increasing_intervals_l531_531322

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 2*x + 1)
noncomputable def f' (x : ℝ) : ℝ := Real.exp x * (x^2 + 4*x + 3)

theorem monotonic_increasing_intervals :
  ∀ x, f' x > 0 ↔ (x < -3 ∨ x > -1) :=
by
  intro x
  -- proof omitted
  sorry

end monotonic_increasing_intervals_l531_531322


namespace exists_periodic_M_l531_531530

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
(a 1).Prime ∧ (a 2).Prime ∧ (a 1 < a 2) ∧
∀ n, if (∃ m, a n + a (n + 1) = 2^m) then a (n + 2) = 2
      else a (n + 2) = Nat.minPrimeFactor (a n + a (n + 1))

theorem exists_periodic_M (p q : ℕ) (hp : p.Prime) (hq : q.Prime) (hpq : p < q) :
  ∃ M, ∀ n > M, ∀ (a : ℕ → ℕ), sequence a → a (n + 1) = a n :=
begin
  sorry
end

end exists_periodic_M_l531_531530


namespace sasha_remainder_l531_531769

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l531_531769


namespace combined_length_cannot_be_determined_l531_531874

-- Define the lengths of the legs of the triangle
def a : ℕ := 10
def b : ℕ := 24

-- Define the hypotenuse calculated using the Pythagorean theorem
def hypotenuse (a b : ℕ) : ℕ := Nat.sqrt(a * a + b * b)

theorem combined_length_cannot_be_determined 
  (three_segments_form_part_of_border : Bool) : 
  three_segments_form_part_of_border → 
  (∃ (length : ℕ), length = a + b ∨ length < a + b ∨ length > a + b) → 
  False :=
by
  sorry

end combined_length_cannot_be_determined_l531_531874


namespace first_player_winning_strategy_l531_531394

def unique_digit_sets (deck : Finset (Finset ℕ)) : Prop :=
  ∀ card ∈ deck, ∀ card' ∈ deck, card ≠ card' → card ≠ card'

def symmetric_difference (A B : Finset ℕ) : Finset ℕ :=
  A \ B ∪ B \ A

noncomputable def even_digit_count (cards : Finset (Finset ℕ)) : Prop :=
  ∀ d : ℕ, (∀ card ∈ cards, d ∈ card → ∑ c in cards, (if d ∈ c then 1 else 0) % 2 = 0)

theorem first_player_winning_strategy (deck : Finset (Finset ℕ)) (empty_card : Finset ℕ) :
  unique_digit_sets deck →
  empty_card = ∅ →
  (∀ card ∈ deck, ∃ card' ∈ deck, card ≠ card' → symmetric_difference card card' = empty_card) →
  ∃ move : Finset ℕ, move ∈ deck ∧ move ≠ empty_card ∧
    (∀ turns : ℕ, turns < deck.card →
      (∃ p1_cards p2_cards : Finset (Finset ℕ), p1_cards ∪ p2_cards = deck ∧ 
        (even_digit_count p1_cards → ¬ even_digit_count p2_cards) ∨ ¬ even_digit_count p1_cards ∧ even_digit_count p2_cards)) :=
sorry

end first_player_winning_strategy_l531_531394


namespace identify_heaviest_and_lightest_in_13_weighings_l531_531806

-- Definitions based on the conditions
def coins := Finset ℕ
def weighs_with_balance_scale (c1 c2: coins) : Prop := true  -- Placeholder for weighing functionality

/-- There are 10 coins, each with a distinct weight. -/
def ten_distinct_coins (coin_set : coins) : Prop :=
  coin_set.card = 10 ∧ (∀ c1 c2 ∈ coin_set, c1 ≠ c2 → weighs_with_balance_scale c1 c2)

-- Theorem statement
theorem identify_heaviest_and_lightest_in_13_weighings 
  (coin_set : coins)
  (hc: ten_distinct_coins coin_set):
  ∃ (heaviest lightest : coins), 
    weighs_with_balance_scale heaviest coin_set ∧ weighs_with_balance_scale coin_set lightest ∧ 
    -- Assuming weighs_with_balance_scale keeps track of number of weighings
    weights_used coin_set = 13 :=
sorry

end identify_heaviest_and_lightest_in_13_weighings_l531_531806


namespace largest_result_l531_531314

theorem largest_result (a b c : ℕ) (h1 : a = 0 / 100) (h2 : b = 0 * 100) (h3 : c = 100 - 0) : 
  c > a ∧ c > b :=
by
  sorry

end largest_result_l531_531314


namespace three_integers_same_parity_l531_531070

theorem three_integers_same_parity (a b c : ℤ) : 
  (∃ i j, i ≠ j ∧ (i = a ∨ i = b ∨ i = c) ∧ (j = a ∨ j = b ∨ j = c) ∧ (i % 2 = j % 2)) :=
by
  sorry

end three_integers_same_parity_l531_531070


namespace inequality_holds_for_real_numbers_l531_531024

theorem inequality_holds_for_real_numbers (a1 a2 a3 a4 : ℝ) (h1 : 1 < a1) 
  (h2 : 1 < a2) (h3 : 1 < a3) (h4 : 1 < a4) : 
  8 * (a1 * a2 * a3 * a4 + 1) ≥ (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) :=
by sorry

end inequality_holds_for_real_numbers_l531_531024


namespace problem1_problem2_l531_531991

-- Definition of the universal set and sets A and B
def universal_set := set ℝ
def A : set ℝ := { x | (1 - 2 * x) / (x - 3) ≥ 0 }
def B (a : ℝ) : set ℝ := { x | x ^ 2 + a ≤ 0 }

-- Complement of set A
def CR_A : set ℝ := { x | x ≥ 3 ∨ x < 1 / 2 }

-- 1. Proving A ∪ B when a = -4
theorem problem1 : A ∪ (B (-4)) = { x | -2 ≤ x ∧ x < 3 } :=
sorry

-- 2. Proving the range of a such that (CR_A) ∩ B = B
theorem problem2 (a : ℝ) : ((CR_A) ∩ (B a) = B a) ↔ a > -1 / 4 :=
sorry

end problem1_problem2_l531_531991


namespace normal_curve_properties_l531_531308

noncomputable def normal_density (x μ σ : ℝ) : ℝ :=
  (1 / (Real.sqrt (2 * Real.pi) * σ)) * Real.exp (-(x - μ) ^ 2 / (2 * σ ^ 2))

theorem normal_curve_properties (μ σ : ℝ) (h₁: σ > 0) :
    (∀ x, normal_density (μ + α) μ σ = normal_density (μ - α) μ σ) ∧
    (∀ x, normal_density x μ σ > 0) :=
by
  sorry

end normal_curve_properties_l531_531308


namespace quadratic_roots_distinct_real_l531_531327

theorem quadratic_roots_distinct_real :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₁ - 2 * x₁ - 6 = 0 ∧ x₂ * x₂ - 2 * x₂ - 6 = 0) :=
by 
  let a := 1
  let b := -2
  let c := -6
  let Δ := b * b - 4 * a * c
  have hΔ : Δ = 28 := by norm_num [a, b, c]
  have hΔ_pos : Δ > 0 := by linarith [hΔ]
  use quadratic_formula a b c hΔ_pos
  sorry

end quadratic_roots_distinct_real_l531_531327


namespace sum_of_divisors_of_91_l531_531517

theorem sum_of_divisors_of_91 : ∑ d in (finset.filter (λ x, 91 % x = 0) (finset.Icc 1 91)), d = 112 :=
by 
  -- This line essentially states the problem as described: sum of all divisors of 91 equals 112
  sorry

end sum_of_divisors_of_91_l531_531517


namespace count_digit_difference_l531_531535

theorem count_digit_difference :
  let pages := [1:ℕ]++(list.range 434).erase_nth 0,            -- Page numbers 1 to 435
      num_to_digits (n : ℕ) : list ℕ := if n < 10  then [0,0,n] else if n < 100 then [0, n / 10, n % 10] else [n / 100, (n % 100) / 10, n % 10],
      digit_count (d : ℕ) (l : list ℕ) : ℕ := (l.filter (λ x, x = d)).length,
      page_digits : list ℕ := pages.bind num_to_digits,
      count_3s : ℕ := digit_count 3 page_digits,
      count_5s : ℕ := digit_count 5 page_digits
  in count_3s - count_5s = 98 :=
by
  sorry

end count_digit_difference_l531_531535


namespace circles_common_tangents_l531_531624

theorem circles_common_tangents (a : ℕ) (h : a ∈ Set.Set.of List.range (a + 1)) :
  (x: ℝ) (y: ℝ) , x^2 + y^2 = 4 ∧ (x-4)^2 + (y + a) ^ 2 = 64 ∧  6 < Real.sqrt (16 + a^2) ∧ Real.sqrt (16 + a^2) < 10 := by
sorrry

end circles_common_tangents_l531_531624


namespace product_of_solutions_l531_531126

theorem product_of_solutions :
  let y := (∃ y : ℝ, |5 * y + 10| = 50) in
  let sol1 := 8 in
  let sol2 := -12 in
  sol1 * sol2 = -96 :=
by
  sorry

end product_of_solutions_l531_531126


namespace bike_sharing_problem_l531_531514

def combinations (n k : ℕ) : ℕ := (Nat.choose n k)

theorem bike_sharing_problem:
  let total_bikes := 10
  let blue_bikes := 4
  let yellow_bikes := 6
  let inspected_bikes := 4
  let way_two_blue := combinations blue_bikes 2 * combinations yellow_bikes 2
  let way_three_blue := combinations blue_bikes 3 * combinations yellow_bikes 1
  let way_four_blue := combinations blue_bikes 4
  way_two_blue + way_three_blue + way_four_blue = 115 :=
by
  sorry

end bike_sharing_problem_l531_531514


namespace probability_of_differ_by_three_l531_531499

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l531_531499


namespace time_to_fill_pool_l531_531877

theorem time_to_fill_pool :
  let R1 := 1
  let R2 := 1 / 2
  let R3 := 1 / 3
  let R4 := 1 / 4
  let R_total := R1 + R2 + R3 + R4
  let T := 1 / R_total
  T = 12 / 25 := 
by
  sorry

end time_to_fill_pool_l531_531877


namespace largest_integer_k_for_distinct_real_roots_l531_531992

theorem largest_integer_k_for_distinct_real_roots :
  ∀ (k : ℤ), ((k < 3) ∧ (k ≠ 2)) → k ≤ 1 :=
by
  intros k h_conditions,
  let a := k - 2,
  let b := -4,
  let c := 4,
  have discriminant_pos : 48 - 16 * k > 0,
  {
    -- The proof for 48 - 16k > 0 which implies k < 3 is assumed from the conditions
    sorry,
  },
  have k_not_2 : k ≠ 2,
  {
    -- The proof for k ≠ 2 is assumed from the conditions
    sorry,
  },
  -- Since k < 3 and k ≠ 2, the largest integer satisfying this is 1
  sorry

end largest_integer_k_for_distinct_real_roots_l531_531992


namespace mark_second_part_playtime_l531_531649

theorem mark_second_part_playtime (total_time initial_time sideline_time : ℕ) 
  (h1 : total_time = 90) (h2 : initial_time = 20) (h3 : sideline_time = 35) :
  total_time - initial_time - sideline_time = 35 :=
sorry

end mark_second_part_playtime_l531_531649


namespace area_bounded_by_curves_l531_531082

open Real

theorem area_bounded_by_curves :
  let f (y : ℝ) := sqrt (exp y - 1)
  let a : ℝ := ln 2
  let lower_limit : ℝ := 0
  let upper_limit : ℝ := 1
  ∫ y in lower_limit..upper_limit, (a - ln (f y^2 + 1)) = 2 - (π / 2) :=
begin
  let f (y : ℝ) := sqrt (exp y - 1),
  let a : ℝ := ln 2,
  let lower_limit : ℝ := 0,
  let upper_limit : ℝ := 1,
  sorry
end

end area_bounded_by_curves_l531_531082


namespace sasha_remainder_l531_531782

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l531_531782


namespace dice_rolls_diff_by_3_probability_l531_531457

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l531_531457


namespace f_expression_lambda_value_l531_531966

-- Define the function f and the conditions
def f (x : ℝ) (a : ℝ) : ℝ := 2 * (x + 1) + 3 * a
def a := 1
axiom f_a_7 : f a a = 7

-- Translate the conditions to Lean assertions
theorem f_expression : f a a = 7 -> ∃ a : ℝ, f a = 2 * a + 5 :=
begin
  sorry -- Proof goes here
end

-- Define the function g and its properties
def g (x : ℝ) (λ : ℝ) : ℝ := x * f x a + λ * f x a + x

axiom g_max_value : ∀ (x : ℝ), 0 <= x -> x <= 2 -> g x (-2) <= 2

theorem lambda_value : g_max_value -> ∃ λ : ℝ, λ = -2 :=
begin
  sorry -- Proof goes here
end

end f_expression_lambda_value_l531_531966


namespace problem_is_perfect_square_l531_531363

theorem problem_is_perfect_square :
  ∃ n : ℕ, n = 23 ∧ ∃ k : ℕ, (k * k = (23! * 24!) / 2) :=
by
  sorry

end problem_is_perfect_square_l531_531363


namespace english_marks_l531_531920

-- Definitions based on the conditions given in the problem
def marks_math : ℕ := 65
def marks_physics : ℕ := 82
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def average_marks : ℕ := 70
def num_subjects : ℕ := 5

-- The statement to be proved
theorem english_marks :
  let total_marks := average_marks * num_subjects,
      known_marks := marks_math + marks_physics + marks_chemistry + marks_biology,
      E := total_marks - known_marks
  in E = 51 :=
by
  sorry

end english_marks_l531_531920


namespace area_of_triangle_ABC_eq_l531_531674

noncomputable def area_of_triangle_ABC : ℝ :=
  let O := (0, 0) in
  let A := (1, 0) in
  let B := sorry  -- We would need coordinates for B and C satisfying the conditions
  let C := sorry
  let D := (B.1 - C.1, B.2 - C.2) in  -- reflection of C across B
  have AB_EQ_AC : dist A B = dist A C,
    from sorry,
  have DO_SQRT3 : dist D O = sqrt 3,
    from sorry,
  let BC := dist B C in
  let h := sqrt (1 - (BC^2) / 4) in  -- height from A to BC bar
  have AREA : (1/2) * BC * h = (sqrt 2 + 1) / 2 ∨ (1/2) * BC * h = (sqrt 2 - 1) / 2,
    from sorry,
  (sqrt 2 + 1) / 2  -- We arbitrarily pick one of the correct answers as a representative

theorem area_of_triangle_ABC_eq :
  area_of_triangle_ABC = (sqrt 2 + 1) / 2 ∨ area_of_triangle_ABC = (sqrt 2 - 1) / 2 := sorry

end area_of_triangle_ABC_eq_l531_531674


namespace perpendicular_vectors_implies_alpha_minimum_magnitude_l531_531163

open Real  -- Open the real number namespace

-- Definitions of vectors
def m (α : ℝ) : ℝ × ℝ := (cos α, sin α)
def n : ℝ × ℝ := (sqrt 3, -1)

-- Condition that α is in the interval (0, π)
def α_in_interval (α : ℝ) : Prop := 0 < α ∧ α < π

-- Dot product of m and n equals zero implies α = π/3
theorem perpendicular_vectors_implies_alpha (α : ℝ) (h1 : 0 < α) (h2 : α < π)
  (h3 : (m α).fst * n.fst + (m α).snd * n.snd = 0) : α = π / 3 :=
  sorry

-- Minimum value of the magnitude of the sum of vectors m and n
theorem minimum_magnitude (α : ℝ) (h1 : 0 < α) (h2 : α < π) : 
  ∃ α_min : ℝ, α_min = 1 ∧ α_min = |sqrt ((cos α + sqrt 3)^2 + (sin α - 1)^2)| :=
  sorry

end perpendicular_vectors_implies_alpha_minimum_magnitude_l531_531163


namespace breakfast_plate_contains_2_eggs_l531_531074

-- Define the conditions
def breakfast_plate := Nat
def num_customers := 14
def num_bacon_strips := 56

-- Define the bacon strips per plate
def bacon_strips_per_plate (num_bacon_strips num_customers : Nat) : Nat :=
  num_bacon_strips / num_customers

-- Define the number of eggs per plate given twice as many bacon strips as eggs
def eggs_per_plate (bacon_strips_per_plate : Nat) : Nat :=
  bacon_strips_per_plate / 2

-- The main theorem we need to prove
theorem breakfast_plate_contains_2_eggs :
  eggs_per_plate (bacon_strips_per_plate 56 14) = 2 :=
by
  sorry

end breakfast_plate_contains_2_eggs_l531_531074


namespace probability_diff_by_three_l531_531468

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l531_531468


namespace dice_rolls_diff_by_3_probability_l531_531454

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l531_531454


namespace inequality_D_holds_l531_531595

theorem inequality_D_holds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := 
sorry

end inequality_D_holds_l531_531595


namespace jennifer_money_left_l531_531842

theorem jennifer_money_left (initial_amount : ℕ) (sandwich_fraction museum_ticket_fraction book_fraction : ℚ) 
  (h_initial : initial_amount = 90) 
  (h_sandwich : sandwich_fraction = 1/5)
  (h_museum_ticket : museum_ticket_fraction = 1/6)
  (h_book : book_fraction = 1/2) : 
  initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_ticket_fraction + initial_amount * book_fraction) = 12 :=
by
  sorry

end jennifer_money_left_l531_531842


namespace hypotenuse_length_l531_531909

theorem hypotenuse_length (a b : ℝ) (h : ℝ) :
  (b^2 + (a^2 / 4) = 50) →
  (a^2 + (b^2 / 4) = 36) →
  h = ∥sqrt(4 * (a^2 + b^2))∥ ∧ h = sqrt(275.2) := 
by
  intro h_eq1 h_eq2
  sorry

end hypotenuse_length_l531_531909


namespace omega_range_l531_531570

namespace Problem

def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) (hω : ω > 0) : 
  (∃ (a b : ℝ), a ∈ set.Icc 0 (Real.pi / 2) ∧ b ∈ set.Icc 0 (Real.pi / 2) ∧ a ≠ b ∧ f ω a + f ω b = 4) ↔ 
  5 ≤ ω ∧ ω < 9 :=
begin
  sorry
end

end Problem

end omega_range_l531_531570


namespace symmetric_line_equation_l531_531313

theorem symmetric_line_equation :
  (∃ line : ℝ → ℝ, ∀ x y, x + 2 * y - 3 = 0 → line 1 = 1 ∧ (∃ b, line 0 = b → x - 2 * y + 1 = 0)) :=
sorry

end symmetric_line_equation_l531_531313


namespace tens_digit_of_8_pow_2048_l531_531359

theorem tens_digit_of_8_pow_2048 : (8^2048 % 100) / 10 = 8 := 
by
  sorry

end tens_digit_of_8_pow_2048_l531_531359


namespace line_circle_intersection_l531_531304

theorem line_circle_intersection (k : ℝ) :
  (k > - (√3) / 3) ↔ ¬ (∀ x y, x ∈ ℝ ∧ y ∈ ℝ → (y = k * (x + 1)) ∧ (x - 1)^2 + y^2 = 1) :=
sorry

end line_circle_intersection_l531_531304


namespace find_y_l531_531651

theorem find_y (y : ℝ) (α : ℝ) (hα : α = 140) (h_sum : y + y + α = 360) : y = 110 :=
begin
  sorry
end

end find_y_l531_531651


namespace count_integers_satisfying_conditions_l531_531099

theorem count_integers_satisfying_conditions :
  (∃ (s : Finset ℤ), s.card = 3 ∧
  ∀ x : ℤ, x ∈ s ↔ (-5 ≤ x ∧ x ≤ -3)) :=
by {
  sorry
}

end count_integers_satisfying_conditions_l531_531099


namespace sum_of_vertices_l531_531544

def six_numbers := {1, 3, 5, 7, 9, 11}

noncomputable def sum_on_each_side := 17

theorem sum_of_vertices :
  ∃ (A B C : ℕ), A ∈ six_numbers ∧ B ∈ six_numbers ∧ C ∈ six_numbers ∧
  A + B + C = 15 ∧
  ∀ x y z : ℕ, (x + y + z = sum_on_each_side) ∧ (x ∈ six_numbers) ∧ (y ∈ six_numbers) ∧ (z ∈ six_numbers) :=
sorry

end sum_of_vertices_l531_531544


namespace dividend_is_correct_l531_531354

-- Define the conditions
def Divisor : Nat := 20
def Quotient : Nat := 8
def Remainder : Nat := 6

-- Define the dividend calculation
def Dividend : Nat := (Divisor * Quotient) + Remainder

-- Theorem stating the equality
theorem dividend_is_correct : Dividend = 166 :=
by
  sorry

end dividend_is_correct_l531_531354


namespace no_such_number_exists_l531_531117

theorem no_such_number_exists :
  ¬ ∃ n : ℕ, 529 < n ∧ n < 538 ∧ 16 ∣ n :=
by sorry

end no_such_number_exists_l531_531117


namespace new_median_is_eight_l531_531035

/-
  We need to formalize the mathematical problem as a theorem in Lean.
  Given:
  - A collection of seven positive integers has a mean of 5.7,
  - A unique mode of 4,
  - and a median of 6,
  If a 9 and a 10 are added to the collection, then the new median is 8.
-/

theorem new_median_is_eight :
  ∃ (s : Multiset ℕ),
    s.card = 7 ∧
    (s.sum : ℝ) / 7 = 5.7 ∧
    (∀ n ∈ s, n > 0) ∧
    (∀ a b ∈ s, a = b → a = 4 ∨ b = 4 → a = 4) ∧
    (s.nth_le 3 (by linarith) = 6) →
    let s' := s ∪ {9, 10} in
    (s'.card = 9 ∧ s'.nth_le 4 (by linarith) = 8) :=
sorry

end new_median_is_eight_l531_531035


namespace circle_intersects_axes_l531_531661

def center_P : ℝ × ℝ := (-3, 4)
def radius (r : ℝ) : Prop := r > 4 ∧ r ≠ 5

theorem circle_intersects_axes (r : ℝ) :
  radius r → ∃ points : set (ℝ × ℝ), points.card = 4 ∧ ∀ p ∈ points, (p.1 = 0 ∨ p.2 = 0) :=
sorry

end circle_intersects_axes_l531_531661


namespace product_of_values_of_t_squared_eq_49_l531_531936

theorem product_of_values_of_t_squared_eq_49 :
  (∀ t, t^2 = 49 → t = 7 ∨ t = -7) →
  (7 * -7 = -49) :=
by
  intros h
  sorry

end product_of_values_of_t_squared_eq_49_l531_531936


namespace green_space_is_18000_l531_531335

-- Define constants and conditions
def total_area := 24000
def occupied_percentage := 0.25

-- Define the green space calculation
def green_space_area := total_area * (1 - occupied_percentage)

-- The theorem to prove the correct answer
theorem green_space_is_18000 :
  green_space_area = 18000 := 
by
  sorry

end green_space_is_18000_l531_531335


namespace sum_of_rational_coefficients_is_neg27_l531_531224

noncomputable def binomial_coefficient (n k : ℕ) : ℤ :=
  if k ≤ n then nat.choose n k else 0

def general_term (r : ℕ) : ℤ :=
  (-1) ^ r * binomial_coefficient 8 r

def is_rational_term (r : ℕ) : Prop :=
  ∃ m : ℤ, 8 - (4 * r) / 3 = m

def sum_of_rational_coefficients : ℤ :=
  finset.sum (finset.filter is_rational_term (finset.range 9)) general_term

theorem sum_of_rational_coefficients_is_neg27 : sum_of_rational_coefficients = -27 :=
  sorry

end sum_of_rational_coefficients_is_neg27_l531_531224


namespace sufficient_but_not_necessary_l531_531385

-- Define the conditions
def abs_value_condition (x : ℝ) : Prop := |x| < 2
def quadratic_condition (x : ℝ) : Prop := x^2 - x - 6 < 0

-- Theorem statement
theorem sufficient_but_not_necessary : (∀ x : ℝ, abs_value_condition x → quadratic_condition x) ∧ ¬ (∀ x : ℝ, quadratic_condition x → abs_value_condition x) :=
by
  sorry

end sufficient_but_not_necessary_l531_531385


namespace angle_MON_constant_l531_531161

noncomputable def y_squared_eq_2x (p: ℝ) (y x: ℝ) := y^2 = 2 * p * x

theorem angle_MON_constant (k: ℝ) (y1 y2 xM yM xN yN : ℝ)
    (hE : y_squared_eq_2x 1 2 2)
    (h_line_l : ∀ x, y1 y2 k xM yM xN yN = k * (x - 2))
    (h_intersection : y1 + y2 = 2 / k ∧ y1 * y2 = -4)
    (h_M : yM = (2 * y1 - 4) / (y1 + 2))
    (h_N : yN = (2 * y2 - 4) / (y2 + 2))
    (h_OM_ON : -2 * -2 + yM * yN = 0) :
  ∃ θ:ℝ, θ = Real.pi / 2 :=
sorry

end angle_MON_constant_l531_531161


namespace probability_differ_by_three_is_one_sixth_l531_531467

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l531_531467


namespace probability_diff_by_three_l531_531473

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l531_531473


namespace range_of_dot_product_l531_531158

-- Define the given points O, A, and B
def O := (0 : ℝ, 0 : ℝ)
def A := (2 : ℝ, 0 : ℝ)
def B := (1 : ℝ, -2 * Real.sqrt 3 : ℝ)

-- Define the moving point P on the ellipse
def P (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sin θ)

-- Define the dot product of vectors OP and BA
def dot_product_OP_BA (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi) : ℝ :=
  let P_θ := P θ hθ in
  let OP := (P_θ.1, P_θ.2) in
  let BA := (A.1 - B.1, A.2 - B.2) in
  OP.1 * BA.1 + OP.2 * BA.2

theorem range_of_dot_product :
  ∀ (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi), 
  -2 ≤ dot_product_OP_BA θ hθ ∧ dot_product_OP_BA θ hθ ≤ 4 := 
  sorry

end range_of_dot_product_l531_531158


namespace smallest_n_for_A_pow_n_to_identity_l531_531550

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  [real.sqrt 2 / 2, real.sqrt 2 / 2],
  [-real.sqrt 2 / 2, real.sqrt 2 / 2]
]

theorem smallest_n_for_A_pow_n_to_identity :
  ∃ n : ℕ, 0 < n ∧ A^n = 1 ∧ ∀ m : ℕ, 0 < m → A^m = (1 : Matrix (Fin 2) (Fin 2) ℝ) → n ≤ m :=
  sorry

end smallest_n_for_A_pow_n_to_identity_l531_531550


namespace dice_rolls_diff_by_3_probability_l531_531455

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l531_531455


namespace count_five_digit_numbers_div_by_16_l531_531189

theorem count_five_digit_numbers_div_by_16 :
  let digits := {d : ℕ | d < 10}
  ∃ count : ℕ, count = 90 ∧ 
    ∀ a b c : ℕ, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 1 ≤ a →
      (nat.gcd (a * 10000 + b * 1000 + 160 + c) 16 = 16) → count = 90 :=
by
  let digits := {d : ℕ | d < 10}
  existsi 90
  split
  { sorry }
  { intros a b c ha hb hc ha_pos hdiv
    sorry }

end count_five_digit_numbers_div_by_16_l531_531189


namespace consecutive_page_sum_l531_531747

theorem consecutive_page_sum (n : ℕ) (h : n * (n + 1) = 2156) : n + (n + 1) = 93 :=
sorry

end consecutive_page_sum_l531_531747


namespace probability_diff_by_three_l531_531475

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l531_531475


namespace octagon_area_half_l531_531968

theorem octagon_area_half (parallelogram : ℝ) (h_parallelogram : parallelogram = 1) : 
  (octagon_area : ℝ) =
  1 / 2 := 
  sorry

end octagon_area_half_l531_531968


namespace boris_mountain_shorter_l531_531194

variable (B x : ℕ)
variable (elevation_hugo : ℕ := 10_000)
variable (condition1 : elevation_hugo = B + x)
variable (condition2 : 3 * elevation_hugo = 4 * B)

theorem boris_mountain_shorter : x = 2500 :=
by
  sorry

end boris_mountain_shorter_l531_531194


namespace product_of_fractions_l531_531897

theorem product_of_fractions : 
  ∀ (a b c d : ℚ), a = 2 / 3 → b = 1 + 4 / 9 → c = 13 / 9 → d = 26 / 27 →
  (a * c) = d := 
begin
  sorry
end

end product_of_fractions_l531_531897


namespace angle_LOQ_degree_measure_l531_531353

noncomputable theory
open_locale classical

variable (L M N O P Q : Type) [regular_hexagon L M N O P Q]

theorem angle_LOQ_degree_measure :
  ∃ (deg_LOQ : ℝ), deg_LOQ = 30 :=
sorry

end angle_LOQ_degree_measure_l531_531353


namespace smallest_m_n_sum_l531_531311

theorem smallest_m_n_sum (m n : ℕ) (h_m : 1 < m) (h_pos : 0 < n) 
  (interval_length : (m^2 - 1) / (m * n) = 1 / 4033) : 
  m + n = 48421 :=
sorry

end smallest_m_n_sum_l531_531311


namespace range_of_a_l531_531609

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a / 2) * x^2 + (2 * a - 1) * x - 2 * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc 1 Real.exp 1, ∃ b ∈ set.Ici (2 : ℝ), f a x ≥ b) ↔ (a ≥ 6 / 5) := 
by
  sorry

end range_of_a_l531_531609


namespace cone_volume_is_24_l531_531036

def base_area : ℝ := 18
def height : ℝ := 4

def cone_volume (B h : ℝ) : ℝ := (1 / 3) * B * h

theorem cone_volume_is_24 : cone_volume base_area height = 24 := by
  sorry

end cone_volume_is_24_l531_531036


namespace task_assignment_ways_l531_531623

-- Definitions based on conditions given in the problem
def total_people : ℕ := 10
def people_selected : ℕ := 4
def task_A_people : ℕ := 2
def task_B_people : ℕ := 1
def task_C_people : ℕ := 1

-- Lean statement to assert the number of ways to achieve the assignment is 2520
theorem task_assignment_ways :
  (Nat.choose total_people people_selected) * 
  (Nat.choose people_selected task_A_people) *
  (Nat.fact (task_B_people + task_C_people)) = 2520 :=
by  -- Begin proof block
  sorry  -- Placeholder for the proof

end task_assignment_ways_l531_531623


namespace trapezoid_area_l531_531119

-- Define the lengths of the bases and legs
def base1 := 18
def base2 := 13
def leg1 := 3
def leg2 := 4

-- Define the calculated height and area for the trapezoid
def height := 2.4
def area := 37.2

-- State the theorem about the area of the trapezoid
theorem trapezoid_area : (1 / 2) * (base1 + base2) * height = area :=
by 
  sorry

end trapezoid_area_l531_531119


namespace position_12340_is_10_l531_531063

noncomputable def position_of_12340 : Nat := 
  let digits := [0, 1, 2, 3, 4]
  let five_digit_numbers := (digits.perm.filter (λ perm, perm.head ≠ 0)).map (λ perm, (perm.head * 10000) + (perm.tail.head! * 1000) + (perm.tail.tail.head! * 100) + (perm.tail.tail.tail.head! * 10) + (perm.tail.tail.tail.tail.head!))
  let sorted_numbers := five_digit_numbers.qsort (≤)
  list.index_of 12340 sorted_numbers + 1

theorem position_12340_is_10 : position_of_12340 = 10 := 
sorry

end position_12340_is_10_l531_531063


namespace perimeter_greater_than_4R_l531_531278

theorem perimeter_greater_than_4R (T : Type) [triangle T] (acute : is_acute T) (P : perimeter T) (R : circumradius T) : P > 4 * R := 
begin
  sorry
end

end perimeter_greater_than_4R_l531_531278


namespace hypotenuse_of_isosceles_right_triangle_l531_531056

theorem hypotenuse_of_isosceles_right_triangle
  (side_length_of_square : ℝ)
  (area_of_square : ℝ)
  (number_of_triangles : ℕ)
  (sum_of_triangle_areas : ℝ)
  (single_triangle_area : ℝ)
  (leg_length_of_triangle : ℝ)
  (hypotenuse_length : ℝ) 
  (h_side_length_of_square : side_length_of_square = 2)
  (h_area_of_square : area_of_square = side_length_of_square ^ 2)
  (h_number_of_triangles : number_of_triangles = 4)
  (h_sum_of_triangle_areas : sum_of_triangle_areas = area_of_square)
  (h_single_triangle_area : single_triangle_area = sum_of_triangle_areas / number_of_triangles)
  (h_leg_length_of_triangle : 2 * single_triangle_area = leg_length_of_triangle ^ 2)
  (h_hypotenuse_length : hypotenuse_length = real.sqrt (2 * leg_length_of_triangle ^ 2))
  : hypotenuse_length = 2 := 
sorry

end hypotenuse_of_isosceles_right_triangle_l531_531056


namespace area_union_arcs_subtract_triangle_l531_531234

/-- Inside a circle with center O and radius 2, chord AB divides the circle into two semicircles.
    Point P is on the semicircle such that OP is ⊥ to AB. Extend AP and BP to points Q and R
    such that AQB and BRP are semicircles with diameter AB, outside the original circle. Both arcs QR 
    and semicircle BRP have centers on P. Prove that the area of the union of the areas enclosed by arcs 
    AQB, BRP, and QR minus the area of triangle ABP is 22π - 8.
-/
theorem area_union_arcs_subtract_triangle :
  ∀ (O A B P Q R : Point) 
    (hO : distance O A = 2) 
    (hO_eq_OP : distance O P = 2)
    (hAB_eq_4 : distance A B = 4)
    (h_perp : is_perpendicular (line O P) (line A B))
    (h_arc_AQB : is_semi_circle (arc A Q B))
    (h_arc_BRP : is_semi_circle (arc B R P))
    (h_arc_QR : is_arc (arc Q R) (center P)),
  union_area ({arc A Q B, arc B R P, arc Q R}) - area_triangle (triangle A B P) = 22 * Real.pi - 8 := 
by 
sorry

end area_union_arcs_subtract_triangle_l531_531234


namespace product_of_values_of_t_squared_eq_49_l531_531935

theorem product_of_values_of_t_squared_eq_49 :
  (∀ t, t^2 = 49 → t = 7 ∨ t = -7) →
  (7 * -7 = -49) :=
by
  intros h
  sorry

end product_of_values_of_t_squared_eq_49_l531_531935


namespace find_a_l531_531963

noncomputable def f (a x : ℝ) : ℝ := 2^x / (2^x + a * x)

variables (a p q : ℝ)

theorem find_a
  (h1 : f a p = 6 / 5)
  (h2 : f a q = -1 / 5)
  (h3 : 2^(p + q) = 16 * p * q)
  (h4 : a > 0) :
  a = 4 :=
  sorry

end find_a_l531_531963


namespace triangle_area_l531_531373

-- Given conditions as definitions
def perimeter : ℝ := 28
def inradius : ℝ := 2.5
def semiperimeter : ℝ := perimeter / 2

-- Statement of the proof
theorem triangle_area : (inradius * semiperimeter = 35) :=
by
  sorry

end triangle_area_l531_531373


namespace coronavirus_diameter_scientific_notation_l531_531731

theorem coronavirus_diameter_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), a = 1.1 ∧ n = -7 ∧ 0.00000011 = a * 10^n := by
sorry

end coronavirus_diameter_scientific_notation_l531_531731


namespace range_of_a_l531_531847

theorem range_of_a (a : ℝ) (h : ∀ θ ∈ Ico 0 (π / 2), 
  √2 * (2 * a + 3) * Real.cos (θ - π / 4) + 6 / (Real.sin θ + Real.cos θ) - 2 * (2 * Real.sin θ * Real.cos θ) < 3 * a + 6) : 
  a > 3 := 
sorry

end range_of_a_l531_531847


namespace sasha_remainder_l531_531768

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l531_531768


namespace calories_per_cookie_l531_531852

theorem calories_per_cookie :
  ∀ (cookies_per_bag bags_per_box total_calories total_number_cookies : ℕ),
  cookies_per_bag = 20 →
  bags_per_box = 4 →
  total_calories = 1600 →
  total_number_cookies = cookies_per_bag * bags_per_box →
  (total_calories / total_number_cookies) = 20 :=
by sorry

end calories_per_cookie_l531_531852


namespace average_marks_of_all_students_l531_531371

theorem average_marks_of_all_students :
  let students1 := 20
  let avg_marks1 := 40
  let students2 := 50
  let avg_marks2 := 60
  let total_marks1 := students1 * avg_marks1
  let total_marks2 := students2 * avg_marks2
  let total_marks := total_marks1 + total_marks2
  let total_students := students1 + students2
  let overall_average_marks := total_marks.toFloat / total_students.toFloat
  in overall_average_marks ≈ 54.29 := 
by
  sorry

end average_marks_of_all_students_l531_531371


namespace number_of_minimally_intersecting_triples_mod_1000_l531_531969

def set_of_natural_numbers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def minimally_intersecting_triple (A B C : Set ℕ) : Prop :=
  A ∩ B ∩ set_of_natural_numbers ≠ ∅ ∧ A ∩ C ∩ set_of_natural_numbers ≠ ∅ ∧ 
  B ∩ C ∩ set_of_natural_numbers ≠ ∅ ∧ 
  ¬∃ x, x ∈ A ∩ B ∩ C ∧ x ∈ set_of_natural_numbers ∧
  (A ∩ B ≠ ∅ ∧ B ∩ C ≠ ∅ ∧ C ∩ A ≠ ∅) ∧
  (A.card = 4 ∧ B.card = 4 ∧ C.card = 4)

theorem number_of_minimally_intersecting_triples_mod_1000 : 
  (∑ (A B C : Set ℕ) in set_of_natural_numbers, if minimally_intersecting_triple A B C then 1 else 0) % 1000 = 360 :=
sorry

end number_of_minimally_intersecting_triples_mod_1000_l531_531969


namespace domain_of_function_l531_531735

noncomputable def domain_function (x : ℝ) : Set ℝ := { y | y = (sqrt (6 - x)) / (abs x - 4)}

theorem domain_of_function :
  {x : ℝ | (6 - x ≥ 0) ∧ (abs x - 4 ≠ 0)} = {x : ℝ | x ∈ (-∞, -4) ∪ (-4, 4) ∪ (4, 6]} :=
by {
  sorry
}

end domain_of_function_l531_531735


namespace number_of_valid_conclusions_is_three_l531_531593

variable {f : ℝ → ℝ}
axiom functional_eq : ∀ x y : ℝ, f(x) * f(y) = f(x + y - 1)
axiom gt1_condition : ∀ x : ℝ, x > 1 → f(x) > 1

noncomputable def number_of_correct_conclusions : ℕ :=
  if h1 : f(1) = 1 then
    let h2 := false, -- The symmetry conclusion has been proved incorrect.
        h3 := ∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁) < f(x₂), -- f is monotonically increasing.
        h4 := ∀ x : ℝ, x < 1 → 0 < f(x) ∧ f(x) < 1 -- 0 < f(x) < 1 when x < 1
    in if h3 ∧ h4 ∧ h1 then 3 else sorry
  else sorry

-- The theorem we aim to state and prove.
theorem number_of_valid_conclusions_is_three : number_of_correct_conclusions = 3 := by
  sorry

end number_of_valid_conclusions_is_three_l531_531593


namespace small_bottle_count_is_747_l531_531678

theorem small_bottle_count_is_747 :
  ∃ (x : ℕ), 
    let large_bottles := 1325 in
    let large_price := 1.89 in
    let small_price := 1.38 in
    let avg_price := 1.7057 in
    let total_cost_large := large_bottles * large_price in
    let total_cost_small := x * small_price in
    let total_bottles := large_bottles + x in
    let avg_price_calc := (total_cost_large + total_cost_small) / total_bottles in
    avg_price_calc = avg_price ∧
    x = 747 := sorry

end small_bottle_count_is_747_l531_531678


namespace problem_omega_pow_l531_531686

noncomputable def omega : ℂ := Complex.I -- Define a non-real root for x² = 1; an example choice could be i, the imaginary unit.

theorem problem_omega_pow :
  omega^2 = 1 → 
  (1 - omega + omega^2)^6 + (1 + omega - omega^2)^6 = 730 := 
by
  intro h1
  -- proof steps omitted
  sorry

end problem_omega_pow_l531_531686


namespace base_prime_representation_450_l531_531351

-- Define prime factorization property for number 450
def prime_factorization_450 := (450 = 2^1 * 3^2 * 5^2)

-- Define base prime representation concept
def base_prime_representation (n : ℕ) : ℕ := 
  if n = 450 then 122 else 0

-- Prove that the base prime representation of 450 is 122
theorem base_prime_representation_450 : 
  prime_factorization_450 →
  base_prime_representation 450 = 122 :=
by
  intros
  sorry

end base_prime_representation_450_l531_531351


namespace number_of_students_suggested_bacon_l531_531893

theorem number_of_students_suggested_bacon :
  ∀ (mashed_potatoes bacon : ℕ),
  mashed_potatoes = 457 →
  mashed_potatoes = bacon + 63 →
  bacon = 394 :=
by
  intros mashed_potatoes bacon h1 h2
  calc
    bacon = mashed_potatoes - 63 : by sorry
    ...   = 457 - 63             : by { exact h1.symm ▸ rfl }
    ...   = 394                  : by norm_num

end number_of_students_suggested_bacon_l531_531893


namespace a_range_l531_531182

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2
def g (x a : ℝ) : ℝ := 2 * x + a

theorem a_range :
  (∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (1 / 2) 2 ∧ x2 ∈ Set.Icc (1 / 2) 2 ∧ f x1 = g x2 a) ↔ -5 ≤ a ∧ a ≤ 0 := 
by 
  sorry

end a_range_l531_531182


namespace correctStatementIsD_l531_531001

-- Condition definitions
def isMonomial (expr : Expr) : Prop :=
  -- A monomial is defined as an algebraic expression with only one term
  containsOnlyOneTerm expr

-- Question as a proof goal
theorem correctStatementIsD (x y : ℝ) :
  isMonomial (expr := xy / 2) :=
sorry

end correctStatementIsD_l531_531001


namespace omega_range_l531_531571

namespace Problem

def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) (hω : ω > 0) : 
  (∃ (a b : ℝ), a ∈ set.Icc 0 (Real.pi / 2) ∧ b ∈ set.Icc 0 (Real.pi / 2) ∧ a ≠ b ∧ f ω a + f ω b = 4) ↔ 
  5 ≤ ω ∧ ω < 9 :=
begin
  sorry
end

end Problem

end omega_range_l531_531571


namespace chocolate_bars_produced_per_minute_l531_531821

theorem chocolate_bars_produced_per_minute
  (sugar_per_bar : ℝ)
  (total_sugar : ℝ)
  (time_in_minutes : ℝ) 
  (bars_per_min : ℝ) :
  sugar_per_bar = 1.5 →
  total_sugar = 108 →
  time_in_minutes = 2 →
  bars_per_min = 36 :=
sorry

end chocolate_bars_produced_per_minute_l531_531821


namespace min_distance_PQ_l531_531147

-- Define the parameters and conditions
def p : ℝ := 2
def F : ℝ × ℝ := (1, 0)
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def directrix (x : ℝ) : Prop := x = -1

-- Define the points M and N on the parabola
def M (y1 : ℝ) : ℝ × ℝ := (y1^2 / 4, y1)
def N (y2 : ℝ) : ℝ × ℝ := (y2^2 / 4, y2)

-- Define the points P and Q where MO and NO intersect the directrix
def P (y1 : ℝ) : ℝ × ℝ := (-1, -4 / y1)
def Q (y2 : ℝ) : ℝ × ℝ := (-1, -4 / y2)

-- Helper to compute |PQ|
noncomputable def distance_PQ (y1 y2 : ℝ) : ℝ :=
  4 * real.sqrt ((y1 + y2)^2 / (y1 * y2)^2 + 1)

-- Prove the minimum value of |PQ| is 4
theorem min_distance_PQ (y1 y2 : ℝ) (h : y1 + y2 = 4 * 0 ∧ y1 * y2 = -4) : distance_PQ y1 y2 = 4 := by
  sorry

end min_distance_PQ_l531_531147


namespace simplify_sqrt_square_l531_531722

theorem simplify_sqrt_square (h : Real.sqrt 7 < 3) : Real.sqrt ((Real.sqrt 7 - 3)^2) = 3 - Real.sqrt 7 :=
by
  sorry

end simplify_sqrt_square_l531_531722


namespace field_area_proof_l531_531866

-- Define the length of the uncovered side
def L : ℕ := 20

-- Define the total amount of fencing used for the other three sides
def total_fence : ℕ := 26

-- Define the field area function
def field_area (length width : ℕ) : ℕ := length * width

-- Statement: Prove that the area of the field is 60 square feet
theorem field_area_proof : 
  ∃ W : ℕ, (2 * W + L = total_fence) ∧ (field_area L W = 60) :=
  sorry

end field_area_proof_l531_531866


namespace determine_k_a_l531_531108

theorem determine_k_a (k a : ℝ) (h : k - a ≠ 0) : (k = 0 ∧ a = 1 / 2) ↔ 
  (∀ x : ℝ, (x + 2) / (kx - ax - 1) = x → x = -2) :=
by
  sorry

end determine_k_a_l531_531108


namespace division_of_decimals_l531_531826

theorem division_of_decimals : (0.5 : ℝ) / (0.025 : ℝ) = 20 := 
sorry

end division_of_decimals_l531_531826


namespace animal_costs_l531_531738

theorem animal_costs :
  ∃ (C G S P : ℕ),
      C + G + S + P = 1325 ∧
      G + S + P = 425 ∧
      C + S + P = 1225 ∧
      G + P = 275 ∧
      C = 900 ∧
      G = 100 ∧
      S = 150 ∧
      P = 175 :=
by
  sorry

end animal_costs_l531_531738


namespace P_at_7_l531_531259

noncomputable def P (x : ℝ) : ℝ :=
  (3 * x ^ 5 - 45 * x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x + d) *
  (4 * x ^ 5 - 100 * x ^ 4 + e * x ^ 3 + f * x ^ 2 + g * x + h)

theorem P_at_7 
  (a b c d e f g h : ℝ)
  (h_roots : ∀ x : ℂ, (x = 1 ∨ x = 2 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 4 ∨ x = 5 ∨ x = 5 ∨ x = 5)
    → x = (x - 1) * (x - 2) * (x - 2) * (x - 3) * (x - 4) * (x - 4) * (x - 5) * (x - 5) * (x - 5)) :
  P 7 = 172800 :=
by
  sorry

end P_at_7_l531_531259


namespace kristin_runs_n_times_faster_l531_531680

theorem kristin_runs_n_times_faster (D K S : ℝ) (n : ℝ) 
  (h1 : K = n * S) 
  (h2 : 12 * D / K = 4 * D / S) : 
  n = 3 :=
by
  sorry

end kristin_runs_n_times_faster_l531_531680


namespace probability_diff_by_3_l531_531477

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l531_531477


namespace soda_cost_is_20_l531_531422

noncomputable def cost_of_soda (b s : ℕ) : Prop :=
  4 * b + 3 * s = 500 ∧ 3 * b + 2 * s = 370

theorem soda_cost_is_20 {b s : ℕ} (h : cost_of_soda b s) : s = 20 :=
  by sorry

end soda_cost_is_20_l531_531422


namespace find_length_DC_l531_531666

noncomputable def length_DC (AB BC AD : ℕ) (BD : ℕ) (h1 : AB = 52) (h2 : BC = 21) (h3 : AD = 48) (h4 : AB^2 = AD^2 + BD^2) (h5 : BD^2 = 20^2) : ℕ :=
  let DC := 29
  DC

theorem find_length_DC (AB BC AD : ℕ) (BD : ℕ) (h1 : AB = 52) (h2 : BC = 21) (h3 : AD = 48) (h4 : AB^2 = AD^2 + BD^2) (h5 : BD^2 = 20^2) (h6 : 20^2 + BC^2 = DC^2) : length_DC AB BC AD BD h1 h2 h3 h4 h5 = 29 :=
  by
  sorry

end find_length_DC_l531_531666


namespace area_of_rectangle_is_270_l531_531319

noncomputable def side_of_square := Real.sqrt 2025

noncomputable def radius_of_circle := side_of_square

noncomputable def length_of_rectangle := (2/5 : ℝ) * radius_of_circle

noncomputable def initial_breadth_of_rectangle := (1/2 : ℝ) * length_of_rectangle + 5

noncomputable def breadth_of_rectangle := if (length_of_rectangle + initial_breadth_of_rectangle) % 3 = 0 
                                          then initial_breadth_of_rectangle 
                                          else initial_breadth_of_rectangle + 1

noncomputable def area_of_rectangle := length_of_rectangle * breadth_of_rectangle

theorem area_of_rectangle_is_270 :
  area_of_rectangle = 270 := by
  sorry

end area_of_rectangle_is_270_l531_531319


namespace number_of_yellow_crayons_l531_531795

theorem number_of_yellow_crayons : 
  ∃ (R B Y : ℕ), 
  R = 14 ∧ 
  B = R + 5 ∧ 
  Y = 2 * B - 6 ∧ 
  Y = 32 :=
by
  sorry

end number_of_yellow_crayons_l531_531795


namespace max_value_3_range_of_a_l531_531977

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + b|
noncomputable def g (x a b : ℝ) : ℝ := -x^2 - ax - b

theorem max_value_3 (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : ∀ x, f x a b ≤ 3) : a + b = 3 :=
sorry

theorem range_of_a (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 3) 
(h₄ : ∀ x, x ≥ a → g x a b < f x a b) : 1 / 2 < a ∧ a < 3 :=
sorry

end max_value_3_range_of_a_l531_531977


namespace unit_prices_purchase_plans_exchange_methods_l531_531513

theorem unit_prices (x r : ℝ) (hx : r = 2 * x) 
  (h_eq : (40/(2*r)) + 4 = 30/x) : 
  x = 2.5 ∧ r = 5 := sorry

theorem purchase_plans (x r : ℝ) (a b : ℕ)
  (hx : x = 2.5) (hr : r = 5) (h_eq : x * a + r * b = 200)
  (h_ge_20 : 20 ≤ a ∧ 20 ≤ b) (h_mult_10 : a % 10 = 0) :
  (a, b) = (20, 30) ∨ (a, b) = (30, 25) ∨ (a, b) = (40, 20) := sorry

theorem exchange_methods (a b t m : ℕ) 
  (hx : x = 2.5) (hr : r = 5) 
  (h_leq : 1 < m ∧ m < 10) 
  (h_eq : a + 2 * t = b + (m - t))
  (h_planA : (a = 20 ∧ b = 30) ∨ (a = 30 ∧ b = 25) ∨ (a = 40 ∧ b = 20)) :
  (m = 5 ∧ t = 5 ∧ b = 30) ∨
  (m = 8 ∧ t = 6 ∧ b = 25) ∨
  (m = 5 ∧ t = 0 ∧ b = 25) ∨
  (m = 8 ∧ t = 1 ∧ b = 20) := sorry

end unit_prices_purchase_plans_exchange_methods_l531_531513


namespace school_class_schedules_l531_531414

theorem school_class_schedules :
  let subjects := {Chinese, Mathematics, English, Physics, Chemistry, PE}
  (∀ (sched : Fin 6 → subjects), 
    (sched 0 ≠ PE ∧ sched 3 ≠ Mathematics)) →
  Fintype.card {sched // sched 0 ≠ PE ∧ sched 3 ≠ Mathematics} = 504 := by
  sorry

end school_class_schedules_l531_531414


namespace arithmetic_sequence_twentieth_term_l531_531913

theorem arithmetic_sequence_twentieth_term :
  ∀ (a_1 d : ℕ), a_1 = 3 → d = 4 → (a_1 + (20 - 1) * d) = 79 := by
  intros a_1 d h1 h2
  rw [h1, h2]
  simp
  sorry

end arithmetic_sequence_twentieth_term_l531_531913


namespace max_distance_circle_to_line_l531_531124

theorem max_distance_circle_to_line :
  let C := (-1, -1 : ℝ)
  let r := 4
  let circle := λ (x y : ℝ), (x + 1)^2 + (y + 1)^2 = 16
  let line := λ (x y : ℝ), 3 * x - 4 * y - 2 = 0
  ∃ d : ℝ, d = abs ((3 * (-1) - 4 * (-1) - 2) / real.sqrt (3^2 + (-4)^2)) ∧
            d + r = 21 / 5 :=
sorry

end max_distance_circle_to_line_l531_531124


namespace dice_diff_by_three_probability_l531_531438

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l531_531438


namespace problem_3_at_7_hash_4_l531_531634

def oper_at (a b : ℕ) : ℚ := (a * b) / (a + b)
def oper_hash (c d : ℚ) : ℚ := c + d

theorem problem_3_at_7_hash_4 :
  oper_hash (oper_at 3 7) 4 = 61 / 10 := by
  sorry

end problem_3_at_7_hash_4_l531_531634


namespace coronavirus_diameter_scientific_notation_l531_531732

theorem coronavirus_diameter_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), a = 1.1 ∧ n = -7 ∧ 0.00000011 = a * 10^n := by
sorry

end coronavirus_diameter_scientific_notation_l531_531732


namespace A_salary_less_than_B_by_20_percent_l531_531879

theorem A_salary_less_than_B_by_20_percent (A B : ℝ) (h1 : B = 1.25 * A) : 
  (B - A) / B * 100 = 20 :=
by
  sorry

end A_salary_less_than_B_by_20_percent_l531_531879


namespace front_view_correct_l531_531927

def column1 := [4, 2]
def column2 := [1, 2, 2]
def column3 := [5, 1, 1]
def column4 := [3, 2]
def column5 := [5, 3]

def columns := [column1, column2, column3, column4, column5]

def max_heights (cols : List (List ℕ)) : List ℕ :=
  cols.map List.maximum

theorem front_view_correct :
  max_heights columns = [4, 2, 5, 3, 5] :=
  by
    sorry

end front_view_correct_l531_531927


namespace lambda_range_if_u_distributed_in_all_quadrants_l531_531972

noncomputable def f (x : ℝ) (λ : ℝ) : ℝ := x^2 - λ * x + 2 * λ
noncomputable def g (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def u (x : ℝ) (λ : ℝ) : ℝ := f x λ * g x

theorem lambda_range_if_u_distributed_in_all_quadrants :
  (∀ x : ℝ, (x^2 - λ * x + 2 * λ) * Real.log (x + 1) < 0)
  ↔ λ ∈ set.Ioo (-1 / 3 : ℝ) 0 := sorry

end lambda_range_if_u_distributed_in_all_quadrants_l531_531972


namespace overall_average_score_l531_531267

theorem overall_average_score
  (mean_morning mean_evening : ℕ)
  (ratio_morning_evening : ℚ) 
  (h1 : mean_morning = 90)
  (h2 : mean_evening = 80)
  (h3 : ratio_morning_evening = 4 / 5) : 
  ∃ overall_mean : ℚ, overall_mean = 84 :=
by
  sorry

end overall_average_score_l531_531267


namespace rulers_length_13_l531_531348

theorem rulers_length_13 :
  ∀ (ruler1 cm ruler2). (ruler1.length = 10 ∧ ruler2.length = 10)
    ∧ (3 ∈ ruler1.marks ∧ 4 ∈ ruler2.marks ∧ ruler1.marks(3) = ruler2.marks(4))
    → (L = 13) :=
by
  sorry

end rulers_length_13_l531_531348


namespace Bia_fraction_smallest_largest_quantity_total_sandwiches_l531_531508

/-- Ana, Bia, Cátia, Diana, and Elaine work as street vendors selling sandwiches.
  They took an identical number of sandwiches daily.
  Mr. Manoel left a note asking each to take 1/5 of the sandwiches.
  Ana took 1/5 of the sandwiches first.
  Bia, thinking she was first, took 1/5 of the remaining sandwiches.
  Cátia, Diana, and Elaine divided the remaining sandwiches equally.
  Prove that Bia took 4/25 of the total sandwiches. -/
theorem Bia_fraction (total : ℕ) (ha : ℕ) (hb : ℕ) (hc : ℕ) (hd : ℕ) (he : ℕ) (h : total ≥ 20) :
  (Bia : ℚ) = 4 / 25 := by
  sorry

/-- Ana, Bia, Cátia, Diana, and Elaine work as street vendors selling sandwiches.
  They took an identical number of sandwiches daily.
  Mr. Manoel left a note asking each to take 1/5 of the sandwiches.
  Ana took 1/5 of the sandwiches first.
  Bia, thinking she was first, took 1/5 of the remaining sandwiches.
  Cátia, Diana, and Elaine divided the remaining sandwiches equally.
  Prove that Bia took the smallest quantity of 12/75 sandwiches
  and Cátia, Diana, and Elaine took the largest quantity of 16/75 sandwiches. -/
theorem smallest_largest_quantity (total : ℕ) (ha : ℕ) (hb : ℕ) (hc : ℕ) (hd : ℕ) (he : ℕ) (h : total ≥ 20) :
  (Bia : ℚ) = 12 / 75 ∧ (Catia : ℚ) = (Diana : ℚ) = (Elaine : ℚ) = 16 / 75 := by
  sorry

/-- Ana, Bia, Cátia, Diana, and Elaine work as street vendors selling sandwiches.
  They took an identical number of sandwiches daily.
  Mr. Manoel left a note asking each to take 1/5 of the sandwiches.
  Ana took 1/5 of the sandwiches first.
  Bia, thinking she was first, took 1/5 of the remaining sandwiches.
  Cátia, Diana, and Elaine divided the remaining sandwiches equally.
  By the end of the division, none of the vendors had more than 20 sandwiches.
  Prove that Mr. Manoel left 75 sandwiches for them. -/
theorem total_sandwiches (total : ℕ) (h : total ≤ 75) :
  total = 75 := by
  sorry

end Bia_fraction_smallest_largest_quantity_total_sandwiches_l531_531508


namespace cost_price_computer_table_l531_531372

theorem cost_price_computer_table :
  ∃ CP : ℝ, CP * 1.25 = 5600 ∧ CP = 4480 :=
by
  sorry

end cost_price_computer_table_l531_531372


namespace dice_diff_by_three_probability_l531_531436

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l531_531436


namespace sum_of_real_solutions_l531_531131

noncomputable def equation (x : ℝ) : Prop :=
  real.sqrt x - real.sqrt (9 / x) + real.sqrt (x + (9 / x)) = 8

theorem sum_of_real_solutions : 
  (∑ x in {x : ℝ | equation x}, x) = 29 / 8 :=
by
  -- Proof to be filled in later
  sorry

end sum_of_real_solutions_l531_531131
