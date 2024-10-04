import Lean
import Mathlib
import Mathlib.Algebra.Combinatorics.Permutations
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Geometry
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.ExpLog
import Mathlib.Algebra.Ring.Defs
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.Integral
import Mathlib.Analysis.SpecialFunctions.Logarithm.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.SimpleGraph.Coloring
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith

namespace largest_prime_divisor_of_2014_l597_597971

-- Define the conditions and the problem
def divides (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem largest_prime_divisor_of_2014 :
  ∀ p : ℕ, (prime p ∧ divides p 2014) → (p ≤ 53) :=
by
  sorry

end largest_prime_divisor_of_2014_l597_597971


namespace repeat_block_of_7_div_13_l597_597631

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l597_597631


namespace intersection_A_B_l597_597393

open Set

-- Define sets A and B with given conditions
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | ∃ a ∈ A, x = 3 * a}

-- Prove the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {0, 3} := 
by
  sorry

end intersection_A_B_l597_597393


namespace angle_ABC_is_77_degrees_l597_597512

open Real EuclideanGeometry

noncomputable def A : ℝ × ℝ × ℝ := (-3, 0, 5)
noncomputable def B : ℝ × ℝ × ℝ := (-4, -2, 1)
noncomputable def C : ℝ × ℝ × ℝ := (-5, -2, 2)

theorem angle_ABC_is_77_degrees :
  ∠ B A C = 77 :=
sorry

end angle_ABC_is_77_degrees_l597_597512


namespace find_dividend_l597_597741

def dividend (divisor quotient remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem find_dividend (divisor quotient remainder : ℕ) (h_divisor : divisor = 16) (h_quotient : quotient = 8) (h_remainder : remainder = 4) :
  dividend divisor quotient remainder = 132 :=
by
  sorry

end find_dividend_l597_597741


namespace find_A_l597_597509

-- Definitions and conditions
def f (A B : ℝ) (x : ℝ) : ℝ := A * x - 3 * B^2 
def g (B C : ℝ) (x : ℝ) : ℝ := B * x + C

theorem find_A (A B C : ℝ) (hB : B ≠ 0) (hBC : B + C ≠ 0) :
  f A B (g B C 1) = 0 → A = (3 * B^2) / (B + C) :=
by
  -- Introduction of the hypotheses
  intro h
  sorry

end find_A_l597_597509


namespace colored_points_exist_l597_597242

universe u

def Point := ℝ × ℝ

variable (color : Point → Bool) -- true for red, false for blue

theorem colored_points_exist (x : ℝ) (h : x > 0) :
  ∃ (p1 p2 : Point), color p1 = color p2 ∧ dist p1 p2 = x :=
by
  sorry

end colored_points_exist_l597_597242


namespace optimal_selling_price_l597_597271

theorem optimal_selling_price : 
  ∀ (purchase_price initial_price units_sold price_increment decreasing_units x : ℕ),
  purchase_price = 40 → 
  initial_price = 50 → 
  units_sold = 500 → 
  price_increment = 1 → 
  decreasing_units = 10 → 
  0 ≤ x ∧ x ≤ 50 → 
  let profit (x : ℕ) := (units_sold - decreasing_units * x) * (initial_price + x) - (units_sold - decreasing_units * x) * purchase_price in
  ∃ x : ℕ, (x = 20) ∧ (initial_price + x = 70 ∧ profit x = (-10 * (x - 20) ^ 2 + 9000)) :=
by
  intros _ _ _ _ _ x _ _ _ _ hx _ _ _ _ _ _
  use 20
  split
  · rfl
  · split
    · rfl
    · sorry

end optimal_selling_price_l597_597271


namespace james_total_money_l597_597132

section
-- Conditions
def number_of_bills : ℕ := 3
def value_of_each_bill : ℕ := 20
def initial_wallet_amount : ℕ := 75

-- Question:
-- What is the total amount of money James has now?
def total_value_of_bills : ℕ := number_of_bills * value_of_each_bill
def total_money_now : ℕ := initial_wallet_amount + total_value_of_bills

-- Theorem stating that he has $135 now.
theorem james_total_money : total_money_now = 135 := 
  by
    sorry
end

end james_total_money_l597_597132


namespace inverse_of_x_squared_minus_x_l597_597097

noncomputable def x : ℂ := (2 - complex.I * real.sqrt 3) / 3

theorem inverse_of_x_squared_minus_x :
  (1 / (x ^ 2 - x)) = (-45 / 28) + (9 * complex.I * real.sqrt 3 / 28) :=
by
  -- The actual proof should follow here.
  sorry

end inverse_of_x_squared_minus_x_l597_597097


namespace repeating_block_length_7_div_13_l597_597598

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l597_597598


namespace ellipse_foci_y_axis_l597_597465

-- Given the equation of the ellipse x^2 + k * y^2 = 2 with foci on the y-axis,
-- prove that the range of k such that the ellipse is oriented with foci on the y-axis is (0, 1).
theorem ellipse_foci_y_axis (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  ∃ (a b : ℝ), a^2 + b^2 = 2 ∧ a > 0 ∧ b > 0 ∧ b / a = k ∧ x^2 + k * y^2 = 2 :=
sorry

end ellipse_foci_y_axis_l597_597465


namespace find_y_value_l597_597765

theorem find_y_value : (12 ^ 2 * 6 ^ 4) / 432 = 432 := by
  sorry

end find_y_value_l597_597765


namespace interval_length_sum_l597_597867

theorem interval_length_sum (x : ℝ) : 
  (0 < x ∧ x < 2) ∧ (Real.sin x > 1/2) → 
  (\let I := setOf (λ x, (Real.sin x > 1/2) ∧ (0 < x ∧ x < 2)) in
   (∃ (a b : ℝ), a < b ∧ I = λ x, a < x ∧ x < b ∧ (Real.sin x > 1/2))) 
   ∧ abs (((5*Real.pi/6) - (Real.pi/6)) - 2.09) < 0.02 :=
by
  sorry

end interval_length_sum_l597_597867


namespace identify_cutting_lines_l597_597193

/-- Definition of a cutting line given a point and a distance. -/
def cutting_line (M : (ℝ × ℝ)) (d : ℝ) (line : ℝ → ℝ) : Prop :=
  ∃ (x : ℝ), (x - M.1) ^ 2 + (line x - M.2) ^ 2 = d ^ 2

/-- Specific problem instance for identifying cutting lines among given options. -/
theorem identify_cutting_lines :
  ∀ M : ℝ × ℝ, M = (5, 0) →
  ((cutting_line M 4 (λ x, 2)) ∧ (cutting_line M 4 (λ x, 4 / 3 * x))) ∧
  ¬ (cutting_line M 4 (λ x, x + 1)) ∧
  ¬ (cutting_line M 4 (λ x, 2 * x + 1)) :=
by
  intro M hM
  split
  { split
    { -- prove y = 2 is a cutting_line
      sorry
    }
    { -- prove y = (4 / 3)x is a cutting_line
      sorry
    }
  }
  { split
    { -- prove y = x + 1 is NOT a cutting_line
      sorry
    }
    { -- prove y = 2x + 1 is NOT a cutting_line
      sorry
    }
  }

end identify_cutting_lines_l597_597193


namespace abs_sum_ineq_l597_597968

theorem abs_sum_ineq (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 2| ≥ 1) : a ∈ Iic 1 ∪ Ici 3 :=
sorry

end abs_sum_ineq_l597_597968


namespace problem_1_problem_2_l597_597910

noncomputable theory

-- Define the function f(x)
def f (a x : ℝ) : ℝ := Real.exp x - a * x

-- Define the interval (-e, -1)
def interval_1 (x : ℝ) : Prop := (-Real.exp 1 < x) ∧ (x < -1)

-- Define the function F(x)
def F (a x : ℝ) : ℝ := f a x - (Real.exp x - 2 * a * x + 2 * Real.log x + a)

-- Define the interval (0, 1/2)
def interval_2 (x : ℝ) : Prop := (0 < x) ∧ (x < 1/2)

-- First problem: show that f(x) is decreasing on (-e, -1) iff a > 1/e
theorem problem_1 (a : ℝ) : (∀ x, interval_1 x → f' a x < 0) ↔ a > (1 / Real.exp 1) :=
sorry

-- Second problem: find maximum value of a such that F(x) has no zero points in (0, 1/2)
theorem problem_2 (a : ℝ) : (∀ x, interval_2 x → F a x ≠ 0) → a ≤ 4 * Real.log 2 :=
sorry

end problem_1_problem_2_l597_597910


namespace rectangles_in_5x5_grid_l597_597731

theorem rectangles_in_5x5_grid : 
  let grid_rows := 5
  let grid_cols := 5
  -- A function that calculates the number of rectangles in an n x m grid
  num_rectangles_in_grid grid_rows grid_cols = 225 :=
  sorry

end rectangles_in_5x5_grid_l597_597731


namespace prime_factors_count_l597_597083

theorem prime_factors_count :
  (∃ a b c d : ℕ, a = 95 ∧ b = 97 ∧ c = 99 ∧ d = 101 ∧
   ∃ p q r s : ℕ, 95 = 5 * 19 ∧ prime 97 ∧ 99 = 3^2 * 11 ∧ prime 101) →
  ∃ n : ℕ, n = 6 :=
by sorry

end prime_factors_count_l597_597083


namespace product_sequence_eq_l597_597827

theorem product_sequence_eq : (∏ k in Finset.range 501, (4 * (k + 1)) / ((4 * (k + 1)) + 4)) = (1 : ℚ) / 502 := 
sorry

end product_sequence_eq_l597_597827


namespace proof1_proof2_l597_597027

variable (a : ℝ) (m n : ℝ)
axiom am_eq_two : a^m = 2
axiom an_eq_three : a^n = 3

theorem proof1 : a^(4 * m + 3 * n) = 432 := by
  sorry

theorem proof2 : a^(5 * m - 2 * n) = 32 / 9 := by
  sorry

end proof1_proof2_l597_597027


namespace license_plates_count_l597_597233

open Finset

def valid_license_plates : Finset (List Char) := 
  let alphabet : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N']
  let alphabetSet := (alphabet.toFinset : Finset Char)
  let firstLetterChoices := (['B', 'C'].toFinset : Finset Char)
  let lastLetterChoices := ('N' : Finset Char)
  let excludedLetters := (['B', 'C', 'M', 'N'].toFinset : Finset Char)
  let usableLetters := alphabetSet \ excludedLetters
  let fValidPlates (freeLetters : Finset Char) : Finset (List Char) → Finset (List Char) :=
    λ acc, acc.bind fun prefix =>
      freeLetters.toFinset.bind (λ l => singleton (prefix ++ [l]))

  (firstLetterChoices.product ((usableLetters.toFinset.powerset 4).bind fValidPlates)).bind
    fun prefix => (<'N'>.product (singleton (prefix.toList))).image
      fun p => p.1 ++ p.2

theorem license_plates_count : valid_license_plates.card = 15840 := by
  sorry

end license_plates_count_l597_597233


namespace prob_3_tails_in_8_flips_l597_597020

def unfair_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

def probability_of_3_tails : ℚ :=
  unfair_coin_probability 8 3 (2/3)

theorem prob_3_tails_in_8_flips :
  probability_of_3_tails = 448 / 6561 :=
by
  sorry

end prob_3_tails_in_8_flips_l597_597020


namespace repeating_block_length_7_div_13_l597_597619

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597619


namespace line_intersects_x_axis_l597_597333

theorem line_intersects_x_axis : ∃ x, 5 * 0 - 7 * x = 14 ∧ (x, 0) = (-2, 0) :=
by {
  use -2,
  split,
  { simp, },
  { refl, },
  sorry
}

end line_intersects_x_axis_l597_597333


namespace hyperbola_eq_sum_mn_constant_area_triangle_range_l597_597441

theorem hyperbola_eq (M : ℝ × ℝ) (F : ℝ × ℝ) (hM : M = (3, real.sqrt 2)) (hF : F = (2, 0)):
  ∃ a b : ℝ, a = real.sqrt 3 ∧ b = 1 ∧
  ∀ x y : ℝ, C x y = (x^2) / (a^2) - (y^2) / (b^2) = 1 → 
  x = 3 ∧ y = real.sqrt 2 → 
  C x y = 1 := sorry

theorem sum_mn_constant (l : ℝ → ℝ × ℝ) (A B P : ℝ × ℝ) (m n : ℝ)
  (hA : l A.1 = A ∧ A ∈ hyperbola) (hB : l B.1 = B ∧ B ∈ hyperbola)
  (hP : l P.1 = P ∧ ∃ x, P = (0, x))
  (hm : ∃ m, PA = m * AF) (hn : ∃ n, PB = n * BF) :
  m + n = 6 := sorry

theorem area_triangle_range (Q : ℝ × ℝ) (Q_eq : Q = (-P.1, -P.2)) (A B P : ℝ × ℝ)
  (area_triangle : ℝ)
  (hA : l A.1 = A ∧ A ∈ hyperbola) (hB : l B.1 = B ∧ B ∈ hyperbola) :
  area_triangle > 4 * real.sqrt 3 / 3 := sorry

end hyperbola_eq_sum_mn_constant_area_triangle_range_l597_597441


namespace repeating_decimal_block_length_l597_597591

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l597_597591


namespace fixed_point_coordinates_l597_597882

theorem fixed_point_coordinates (k : ℝ) (M : ℝ × ℝ) (h : ∀ k : ℝ, M.2 - 2 = k * (M.1 + 1)) :
  M = (-1, 2) :=
sorry

end fixed_point_coordinates_l597_597882


namespace solve_for_x_l597_597563

theorem solve_for_x : ∃ x : ℝ, 4 * x + 6 * x = 360 - 9 * (x - 4) ∧ x = 396 / 19 :=
by
  sorry

end solve_for_x_l597_597563


namespace fresh_water_needed_l597_597559

noncomputable def mass_of_seawater : ℝ := 30
noncomputable def initial_salt_concentration : ℝ := 0.05
noncomputable def desired_salt_concentration : ℝ := 0.015

theorem fresh_water_needed :
  ∃ (fresh_water_mass : ℝ), 
    fresh_water_mass = 70 ∧ 
    (mass_of_seawater * initial_salt_concentration) / (mass_of_seawater + fresh_water_mass) = desired_salt_concentration :=
by
  sorry

end fresh_water_needed_l597_597559


namespace value_of_B_l597_597809

noncomputable def polynomial_has_positive_integer_roots (A B C D : ℤ) : Prop :=
  ∃ (roots : list ℤ), roots.all (λ x, x > 0) ∧ (roots.sum = 12) ∧
    (z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36 =
    roots.prod (λ r, (z - r)))

theorem value_of_B (A C D : ℤ) (h : polynomial_has_positive_integer_roots A B C D) : B = -162 :=
sorry

end value_of_B_l597_597809


namespace correct_propositions_l597_597164

-- Definitions of relations between lines and planes
variable {Line : Type}
variable {Plane : Type}

-- Definition of relationships
variable (parallel_lines : Line → Line → Prop)
variable (parallel_plane_with_plane : Plane → Plane → Prop)
variable (parallel_line_with_plane : Line → Plane → Prop)
variable (perpendicular_plane_with_plane : Plane → Plane → Prop)
variable (perpendicular_line_with_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (distinct_lines : Line → Line → Prop)
variable (distinct_planes : Plane → Plane → Prop)

-- The main theorem we are proving with the given conditions
theorem correct_propositions (m n : Line) (α β γ : Plane)
  (hmn : distinct_lines m n) (hαβ : distinct_planes α β) (hαγ : distinct_planes α γ)
  (hβγ : distinct_planes β γ) :
  -- Statement 1
  (parallel_plane_with_plane α β → parallel_plane_with_plane α γ → parallel_plane_with_plane β γ) ∧
  -- Statement 3
  (perpendicular_line_with_plane m α → parallel_line_with_plane m β → perpendicular_plane_with_plane α β) :=
by
  sorry

end correct_propositions_l597_597164


namespace monotonic_increase_interval_l597_597930

open Set

variable {f : ℝ → ℝ}

theorem monotonic_increase_interval (h_deriv : ∀ x, deriv f x = (x - 3) * (x + 1)^2) :
  Icc 3 (⊤ : ℝ) ⊆ {x : ℝ | ∃ I, I ⊆ Ioi x ∧ MonotoneOn f I} :=
by
  sorry

end monotonic_increase_interval_l597_597930


namespace part_I_part_II_l597_597933

-- Define the parametric equations for the curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (1 + sqrt 3 * cos θ, sqrt 3 * sin θ)

-- Define the line equation
def line_l (m θ : ℝ) : ℝ := sqrt 3 * m

-- Prove that the line is tangent to the curve when m = 3 
theorem part_I (m θ : ℝ) (p ∈ curve_C θ) :
  m = 3 →
  let (x, y) := p in
  (x - 1) ^ 2 + y ^ 2 = 3 → 
  y + sqrt 3 * x = sqrt 3 * m →
  sqrt 3 * m = 3 → 
  sqrt 3 = (sqrt 3)  := sorry

-- Prove the given range of m
theorem part_II (m θ : ℝ) (p ∈ curve_C θ) :
  (∃ p, p = curve_C θ ∧ sqrt ((fst p - 1) ^ 2 + (snd p) ^ 2) = sqrt 3 / 2) →
  abs (sqrt 3 - m * sqrt 3) ≤ (sqrt 3 + sqrt 3 / 2) →
  -2 ≤ m ∧ m ≤ 4 := sorry

end part_I_part_II_l597_597933


namespace parallel_vectors_x_value_l597_597449

noncomputable def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem parallel_vectors_x_value :
  vectors_parallel (1, -2) (x, 1) → x = -1 / 2 :=
by
  sorry

end parallel_vectors_x_value_l597_597449


namespace sequence_eventually_one_or_three_l597_597848

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 3

def sequence (m : ℕ) : ℕ → ℕ
| 0       := m
| (n + 1) := f (sequence n)

theorem sequence_eventually_one_or_three (m : ℕ) (hm : m > 0) :
  ∃ n, sequence m n = 1 ∨ sequence m n = 3 :=
  sorry

end sequence_eventually_one_or_three_l597_597848


namespace sum_of_digits_10_38_minus_85_l597_597197

theorem sum_of_digits_10_38_minus_85 : 
  ∑ c in (Nat.digits 10 (10^38 - 85)).to_finset, c = 330 := 
sorry

end sum_of_digits_10_38_minus_85_l597_597197


namespace largest_possible_value_of_abs_z_l597_597507

theorem largest_possible_value_of_abs_z 
  (a b c z : ℂ) 
  (ha : |a| = 1) 
  (hb : |b| = 1) 
  (hc : |c| = 1) 
  (h_eq : a * z ^ 2 + 2 * b * z + c = 0)
: |z| ≤ 1 + Real.sqrt 2 := 
sorry

end largest_possible_value_of_abs_z_l597_597507


namespace sam_wins_l597_597557

variable (p : ℚ) -- p is the probability that Sam wins
variable (phit : ℚ) -- probability of hitting the target in one shot
variable (pmiss : ℚ) -- probability of missing the target in one shot

-- Define the problem and set up the conditions
def conditions : Prop := phit = 2 / 5 ∧ pmiss = 3 / 5

-- Define the equation derived from the problem
def equation (p : ℚ) (phit : ℚ) (pmiss : ℚ) : Prop :=
  p = phit + (pmiss * pmiss * p)

-- State the theorem that Sam wins with probability 5/8
theorem sam_wins (h : conditions phit pmiss) : 
  equation p phit pmiss → p = 5 / 8 :=
by
  intros
  sorry

end sam_wins_l597_597557


namespace relationship_abc_l597_597093

noncomputable def a := (1 / 3 : ℝ) ^ (2 / 3)
noncomputable def b := (2 / 3 : ℝ) ^ (1 / 3)
noncomputable def c := Real.logb (1/2) (1/3)

theorem relationship_abc : c > b ∧ b > a :=
by
  sorry

end relationship_abc_l597_597093


namespace y_intercept_is_3_l597_597072

-- Define the points P and Q
structure Point where
  x : ℤ
  y : ℤ

def P : Point := { x := 2, y := -1 }
def Q : Point := { x := -2, y := 7 }

-- Define the slope of the line passing through P and Q
def slope (P Q : Point) : ℚ :=
  (Q.y - P.y : ℚ) / (Q.x - P.x : ℚ)

-- Define the y-intercept of the line passing through P and Q
def y_intercept (P Q : Point) : ℚ :=
  let m := slope P Q
  let b := P.y - m * P.x
  b

-- The theorem to prove: the y-intercept of the line passing through P and Q is 3
theorem y_intercept_is_3 : y_intercept P Q = 3 := by
  sorry

end y_intercept_is_3_l597_597072


namespace compute_usage_difference_l597_597278

theorem compute_usage_difference
  (usage_last_week : ℕ)
  (usage_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : usage_last_week = 91)
  (h2 : usage_per_day = 8)
  (h3 : days_in_week = 7) :
  (usage_last_week - usage_per_day * days_in_week) = 35 :=
  sorry

end compute_usage_difference_l597_597278


namespace trailing_zeros_factorial_fraction_l597_597111

theorem trailing_zeros_factorial_fraction:
  let n := 26!
  let d := 35 ^ 3
  (multiplicity 10 (n / d)) = 3 :=
sorry

end trailing_zeros_factorial_fraction_l597_597111


namespace quad_eq_double_root_m_value_l597_597013

theorem quad_eq_double_root_m_value (m : ℝ) : 
  (∀ x : ℝ, x^2 + 6 * x + m = 0) → m = 9 := 
by 
  sorry

end quad_eq_double_root_m_value_l597_597013


namespace parallel_lines_a_eq_1_l597_597467

theorem parallel_lines_a_eq_1 (a : ℝ) : 
  (ax - 2y + 2 = 0) ∧ (x + (a - 3)y + 1 = 0) → a = 1 :=
by 
  sorry

end parallel_lines_a_eq_1_l597_597467


namespace james_total_money_l597_597139

theorem james_total_money :
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  total_money = 135 := by
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  exact 135

end james_total_money_l597_597139


namespace trigonometric_identity_l597_597534

theorem trigonometric_identity 
  (α β : ℝ) : 
  (cos α)^2 + (cos β)^2 - 2 * (cos α) * (cos β) * (cos (α + β)) = 
  (sin α)^2 + (sin β)^2 + 2 * (sin α) * (sin β) * (sin (α + β)) :=
sorry

end trigonometric_identity_l597_597534


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_l597_597821

theorem problem_1 (x y z : ℝ) (h : z = (x + y) / 2) : z = (x + y) / 2 :=
sorry

theorem problem_2 (x y w : ℝ) (h1 : w = x + y) : w = x + y :=
sorry

theorem problem_3 (x w y : ℝ) (h1 : w = x + y) (h2 : y = w - x) : y = w - x :=
sorry

theorem problem_4 (x z v : ℝ) (h1 : z = (x + y) / 2) (h2 : v = 2 * z) : v = 2 * (x + (x + y) / 2) :=
sorry

theorem problem_5 (x z u : ℝ) (h : u = - (x + z) / 5) : x + z + 5 * u = 0 :=
sorry

theorem problem_6 (y z t : ℝ) (h : t = (6 + y + z) / 2) : t = (6 + y + z) / 2 :=
sorry

theorem problem_7 (y z s : ℝ) (h : y + z + 4 * s - 10 = 0) : y + z + 4 * s - 10 = 0 :=
sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_l597_597821


namespace repeating_decimal_block_length_l597_597586

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l597_597586


namespace max_m_eq_4_inequality_a_b_c_l597_597435

noncomputable def f (x : ℝ) : ℝ :=
  |x - 3| + |x + 2|

theorem max_m_eq_4 (m : ℝ) (h : ∀ x : ℝ, f x ≥ |m + 1|) : m ≤ 4 ∧ m ≥ -6 :=
  sorry

theorem inequality_a_b_c (a b c : ℝ) (h : a + 2 * b + c = 4) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
  sorry

end max_m_eq_4_inequality_a_b_c_l597_597435


namespace earth_surface_area_scientific_notation_l597_597658

theorem earth_surface_area_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 780000000 = a * 10^n ∧ a = 7.8 ∧ n = 8 :=
by
  sorry

end earth_surface_area_scientific_notation_l597_597658


namespace geologists_can_reach_station_l597_597259

open Real

def motorcycle_speed  := 50 -- km/h
def pedestrian_speed := 5 -- km/h
def distance_to_station := 60 -- km
def total_time := 3 -- hours
def max_riders := 2

theorem geologists_can_reach_station :
  ∃ (time1 time2 time3 : ℝ), 
    (0 ≤ time1 ∧ 0 ≤ time2 ∧ 0 ≤ time3) ∧ 
    (time1 + time2 + time3 ≤ total_time) ∧
    (motorcycle_speed * time1 + pedestrian_speed * (total_time - (time1 + time2 + time3)) ≥ distance_to_station) :=
begin
  sorry
end

end geologists_can_reach_station_l597_597259


namespace repeating_block_length_of_7_div_13_is_6_l597_597604

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l597_597604


namespace correct_average_l597_597218

theorem correct_average :
  let avg_incorrect := 15
  let num_numbers := 20
  let read_incorrect1 := 42
  let read_correct1 := 52
  let read_incorrect2 := 68
  let read_correct2 := 78
  let read_incorrect3 := 85
  let read_correct3 := 95
  let incorrect_sum := avg_incorrect * num_numbers
  let diff1 := read_correct1 - read_incorrect1
  let diff2 := read_correct2 - read_incorrect2
  let diff3 := read_correct3 - read_incorrect3
  let total_diff := diff1 + diff2 + diff3
  let correct_sum := incorrect_sum + total_diff
  let correct_avg := correct_sum / num_numbers
  correct_avg = 16.5 :=
by
  sorry

end correct_average_l597_597218


namespace sequence_non_periodic_l597_597317

/-- A helper function to compute the sum of the digits of a number n -/
def digitSum (n : ℕ) : ℕ :=
  let digits := List.ofFn (λ i, (n / 10^i) % 10) (n.digits.size)
  digits.sum

/-- Defining the sequence based on the sum of digits being even or odd -/
def sequence (k : ℕ) : ℕ :=
  if digitSum k % 2 = 0 then 0 else 1

/-- The main theorem stating the sequence is non-periodic -/
theorem sequence_non_periodic : ¬(∃ d m : ℕ, ∀ n : ℕ, (n ≥ m → sequence n = sequence (n + d))) :=
sorry

end sequence_non_periodic_l597_597317


namespace JordanRectangleWidth_l597_597763

/-- Given that Carol's rectangle measures 15 inches by 24 inches,
and Jordan's rectangle is 8 inches long with equal area as Carol's rectangle,
prove that Jordan's rectangle is 45 inches wide. -/
theorem JordanRectangleWidth :
  ∃ W : ℝ, (15 * 24 = 8 * W) → W = 45 := by
  sorry

end JordanRectangleWidth_l597_597763


namespace maximum_distance_and_coordinates_l597_597204

noncomputable def point_P (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.sqrt 3 * Real.cos θ, 2 * Real.sin θ)

def curve_C1 (P : ℝ × ℝ) : Prop :=
  (P.1 ^ 2 / 12 + P.2 ^ 2 / 4 = 1)

def line_C2 (t : ℝ) : ℝ × ℝ :=
  (3 + (Real.sqrt 3 / 2) * t, Real.sqrt 3 - (1 / 2) * t)

def line_C2_standard (x y : ℝ) : Prop :=
  (x + Real.sqrt 3 * y - 6 = 0)

def distance_between_point_line (P : ℝ × ℝ) : ℝ :=
  Real.abs ((Real.sqrt 6 * (Real.sin (P.2 + Real.pi / 4))) - 3)

theorem maximum_distance_and_coordinates :
  ∃ (θ : ℝ) (d_max : ℝ) (P_max : ℝ × ℝ),
    curve_C1 (point_P θ) ∧
    distance_between_point_line (point_P θ) = d_max ∧
    d_max = Real.sqrt 6 + 3 ∧
    P_max = (-Real.sqrt 6, -Real.sqrt 2) :=
sorry

end maximum_distance_and_coordinates_l597_597204


namespace partA_density_partA_characteristic_partB_characteristic_partC_distribution_l597_597290

-- Definitions and main statements of the proof problems

def f (x : ℝ) : ℝ := (1 - Real.cos x) / (π * x^2)
def varphi (t : ℝ) : ℝ := if |t| ≤ 1 then 1 - |t| else 0

theorem partA_density (x : ℝ) :
  f x = (1 - Real.cos x) / (π * x^2) = (1 / (2 * π)) * ((Real.sin(x / 2) / (x / 2)) ^ 2) :=
sorry

theorem partA_characteristic :
  (∀ t, varphi t = (if |t| ≤ 1 then 1 - |t| else 0)) ∧ 
  (is_density f) := sorry

def f2 (x : ℝ) : ℝ := 1 / (π * Real.cosh x)
def g (x : ℝ) : ℝ := 1 / (2 * (Real.cosh x) ^ 2)
def varphi2 (t : ℝ) : ℝ := 1 / Real.cosh (π * t / 2)
def phi (t : ℝ) : ℝ := (π * t) / (2 * Real.sinh(π * t / 2))

theorem partB_characteristic :
  (varphi2 = λ t, 1 / Real.cosh (π * t / 2)) ∧ 
  (phi = λ t, (π * t) / (2 * Real.sinh (π * t / 2))) := sorry

inductive Distribution
| exponential : Distribution
| cauchy : Distribution
| uniform : Distribution
| trigonometric_derived : Distribution

def characteristic_func (t : ℝ) (d : Distribution) : ℝ :=
match d with
| Distribution.exponential => 1 / (1 - Complex.I * t)
| Distribution.cauchy => 1 / (1 + t^2)
| Distribution.uniform => Real.cos t
| Distribution.trigonometric_derived => 1 / (2 - Real.cos t)

theorem partC_distribution :
  (characteristic_func t Distribution.exponential = (1 / (1- Complex.I * t))) ∧ 
  (characteristic_func t Distribution.cauchy = (1 / (1 + t^2)))  ∧ 
  (characteristic_func t Distribution.uniform = (Real.cos t)) ∧ 
  (characteristic_func t Distribution.trigonometric_derived = (1 / (2 - Real.cos t))) := sorry

end partA_density_partA_characteristic_partB_characteristic_partC_distribution_l597_597290


namespace part1_part2_l597_597029

variables {a m n : ℝ}

theorem part1 (h1 : a^m = 2) (h2 : a^n = 3) : a^(4*m + 3*n) = 432 :=
by sorry

theorem part2 (h1 : a^m = 2) (h2 : a^n = 3) : a^(5*m - 2*n) = 32 / 9 :=
by sorry

end part1_part2_l597_597029


namespace distinct_primes_count_l597_597080

theorem distinct_primes_count : 
  ∃ n, 
  (95 = 5 * 19) ∧ 
  (Prime 97) ∧ 
  (99 = 3^2 * 11) ∧ 
  (Prime 101) ∧ 
  (n = 6 ∧ (∀ primes, primes ∈ [5, 19, 97, 3, 11, 101] → primes ∈ PrimeFactors (95 * 97 * 99 * 101))) := 
sorry

end distinct_primes_count_l597_597080


namespace sequence_no_limit_l597_597881

noncomputable def sequence_limit (x : ℕ → ℝ) (a : ℝ) : Prop :=
    ∀ ε > 0, ∃ N, ∀ n > N, abs (x n - a) < ε

theorem sequence_no_limit (x : ℕ → ℝ) (a : ℝ) (ε : ℝ) (k : ℕ) :
    (ε > 0) ∧ (∀ n, n > k → abs (x n - a) ≥ ε) → ¬ sequence_limit x a :=
by
  sorry

end sequence_no_limit_l597_597881


namespace complement_intersection_l597_597447

noncomputable def U : Set Real := Set.univ
noncomputable def M : Set Real := { x : Real | Real.log x < 0 }
noncomputable def N : Set Real := { x : Real | (1 / 2) ^ x ≥ Real.sqrt (1 / 2) }

theorem complement_intersection (U M N : Set Real) : 
  (Set.compl M ∩ N) = Set.Iic 0 :=
by
  sorry

end complement_intersection_l597_597447


namespace repeat_block_of_7_div_13_l597_597632

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l597_597632


namespace james_total_money_l597_597129

theorem james_total_money (bills : ℕ) (value_per_bill : ℕ) (initial_money : ℕ) : 
  bills = 3 → value_per_bill = 20 → initial_money = 75 → initial_money + (bills * value_per_bill) = 135 :=
by
  intros hb hv hi
  rw [hb, hv, hi]
  -- Algebraic simplification
  sorry

end james_total_money_l597_597129


namespace find_y_l597_597961

theorem find_y
  (x y : ℝ)
  (h1 : x - y = 10)
  (h2 : x + y = 8) : y = -1 :=
by
  sorry

end find_y_l597_597961


namespace red_peaches_each_basket_l597_597256

variable (TotalGreenPeachesInABasket : Nat) (TotalPeachesInABasket : Nat)

theorem red_peaches_each_basket (h1 : TotalPeachesInABasket = 10) (h2 : TotalGreenPeachesInABasket = 3) :
  (TotalPeachesInABasket - TotalGreenPeachesInABasket) = 7 := by
  sorry

end red_peaches_each_basket_l597_597256


namespace greg_charges_per_dog_l597_597450

theorem greg_charges_per_dog (charge_per_dog : ℝ) (time_rate : ℝ) (minutes_dog1 : ℕ) (minutes_dog2 : ℕ) 
  (dogs_dog2 : ℕ) (minutes_dog3 : ℕ) (dogs_dog3 : ℕ) (total_earnings : ℝ) 
  (h1 : time_rate = 1) (h2 : minutes_dog1 = 10) (h3 : minutes_dog2 = 7) (h4 : dogs_dog2 = 2)
  (h5 : minutes_dog3 = 9) (h6 : dogs_dog3 = 3) (h7 : total_earnings = 171) :
  charge_per_dog = 20 :=
by
  have eq1 : total_earnings = charge_per_dog + minutes_dog1 * time_rate + dogs_dog2 * charge_per_dog + dogs_dog2 * minutes_dog2 * time_rate + dogs_dog3 * charge_per_dog + dogs_dog3 * minutes_dog3 * time_rate,
    admit
  have eq2 : total_earnings = charge_per_dog + 10 + 2 * charge_per_dog + 2 * 7 + 3 * charge_per_dog + 3 * 9,
    admit
  have eq3 : total_earnings = 6 * charge_per_dog + 51,
    admit
  have final_eq : 171 = 6 * charge_per_dog + 51,
    admit
  have simplify_eq : 6 * charge_per_dog = 120,
    admit
  have final_charge : charge_per_dog = 20,
    admit
  exact final_charge

end greg_charges_per_dog_l597_597450


namespace find_t_l597_597190

theorem find_t (t : ℚ) : 
  ((t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 1) + 5) → t = 5 / 3 :=
by
  intro h
  sorry

end find_t_l597_597190


namespace gumballs_each_shared_equally_l597_597142

def initial_gumballs_joanna : ℕ := 40
def initial_gumballs_jacques : ℕ := 60
def multiplier : ℕ := 4

def purchased_gumballs (initial : ℕ) (multiplier : ℕ) : ℕ :=
  initial * multiplier

def total_gumballs (initial : ℕ) (purchased : ℕ) : ℕ :=
  initial + purchased

def total_combined_gumballs (total1 : ℕ) (total2 : ℕ) : ℕ :=
  total1 + total2

def shared_equally (total : ℕ) : ℕ :=
  total / 2

theorem gumballs_each_shared_equally :
  let joanna_initial := initial_gumballs_joanna,
      jacques_initial := initial_gumballs_jacques,
      joanna_purchased := purchased_gumballs joanna_initial multiplier,
      jacques_purchased := purchased_gumballs jacques_initial multiplier,
      joanna_total := total_gumballs joanna_initial joanna_purchased,
      jacques_total := total_gumballs jacques_initial jacques_purchased,
      combined_total := total_combined_gumballs joanna_total jacques_total in
  shared_equally combined_total = 250 :=
by
  sorry

end gumballs_each_shared_equally_l597_597142


namespace value_of_x_plus_y_squared_l597_597169

variable (x y : ℝ)

def condition1 : Prop := x * (x + y) = 40
def condition2 : Prop := y * (x + y) = 90
def condition3 : Prop := x - y = 5

theorem value_of_x_plus_y_squared (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : (x + y) ^ 2 = 130 :=
by
  sorry

end value_of_x_plus_y_squared_l597_597169


namespace pyramid_section_ratios_l597_597976

-- Definitions based on the conditions
def HexagonalPyramid (S A B C D E F : Type) : Prop := sorry
def DividesDiagonally (A D : Type) (points : List Type) : Prop := sorry
def SectionsParallel (S A B : Type) (points : List Type) : Prop := sorry

-- Main theorem statement proving the ratio of areas
theorem pyramid_section_ratios (S A B C D E F : Type) 
  (points : List Type)
  (h1 : HexagonalPyramid S A B C D E F)
  (h2 : DividesDiagonally A D points)
  (h3 : SectionsParallel S A B points) :
  let ratio := (25:20:9) in 
  ratio = (25:20:9) :=
sorry

end pyramid_section_ratios_l597_597976


namespace exterior_angle_of_regular_octagon_l597_597108

theorem exterior_angle_of_regular_octagon : 
  (∑ i in (Finset.range 8), (angle i)) = 1080 ∧
  (∀ i ∈ (Finset.range 8), (angle i) = 135 ) →
  (∃ e, e = 45∧(∀ i ∈ (Finset.range 8), e = 180 - (angle i))) :=
by
  sorry

end exterior_angle_of_regular_octagon_l597_597108


namespace sam_wins_probability_l597_597545

theorem sam_wins_probability : 
  let hit_prob := (2 : ℚ) / 5
      miss_prob := (3 : ℚ) / 5
      p := hit_prob + (miss_prob * miss_prob) * p
  in p = 5 / 8 := 
by
  -- Proof goes here
  sorry

end sam_wins_probability_l597_597545


namespace avg_growth_rate_first_brand_eq_l597_597325

noncomputable def avg_growth_rate_first_brand : ℝ :=
  let t := 5.647
  let first_brand_households_2001 := 4.9
  let second_brand_households_2001 := 2.5
  let second_brand_growth_rate := 0.7
  let equalization_time := t
  (second_brand_households_2001 + second_brand_growth_rate * equalization_time - first_brand_households_2001) / equalization_time

theorem avg_growth_rate_first_brand_eq :
  avg_growth_rate_first_brand = 0.275 := by
  sorry

end avg_growth_rate_first_brand_eq_l597_597325


namespace cylinder_ratio_max_volume_l597_597302

theorem cylinder_ratio_max_volume (r h : ℝ) (h_nonneg : 0 ≤ h)
  (surface_area_eq : 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 12 * Real.pi) :
  let V := λ r h, Real.pi * r^2 * h in
  let h_val := 6 - r^2 in
  (∀ r, 0 < r → h = h_val → 
   (∀ r, V r h ≤ V sqrt(2) h_val) → 
   r / h = 1 / 2) := 
sorry

end cylinder_ratio_max_volume_l597_597302


namespace tan_x_plus_tan_y_l597_597092

theorem tan_x_plus_tan_y (x y : ℝ) 
  (h1 : sin x + sin y = 3 / 5) 
  (h2 : cos x + cos y = 4 / 5) : 
  tan x + tan y = 57 / 26 := 
by
  sorry

end tan_x_plus_tan_y_l597_597092


namespace player1_points_after_13_rotations_l597_597692

theorem player1_points_after_13_rotations :
  let sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
  let player_points (player : Nat) (rotations : Nat) :=
      rotations • (λ i, sector_points[(i + player) % 16])
  player_points 5 13 = 72 ∧ player_points 9 13 = 84 → player_points 1 13 = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597692


namespace largest_percentage_increase_between_l597_597331

theorem largest_percentage_increase_between (n2003 n2004 n2005 n2006 n2007 n2008 : ℕ) 
    (h1: n2003 = 50) (h2: n2004 = 55) (h3: n2005 = 60) (h4: n2006 = 65) (h5: n2007 = 75) (h6: n2008 = 80) :
  (2006, 2007) = 
    let perc_increase_2003_2004 := (n2004 - n2003) * 100 / n2003 in
    let perc_increase_2004_2005 := (n2005 - n2004) * 100 / n2004 in
    let perc_increase_2005_2006 := (n2006 - n2005) * 100 / n2005 in
    let perc_increase_2006_2007 := (n2007 - n2006) * 100 / n2006 in
    let perc_increase_2007_2008 := (n2008 - n2007) * 100 / n2007 in
    if perc_increase_2006_2007 > perc_increase_2003_2004 &&
       perc_increase_2006_2007 > perc_increase_2004_2005 &&
       perc_increase_2006_2007 > perc_increase_2005_2006 &&
       perc_increase_2006_2007 > perc_increase_2007_2008
    then (2006, 2007)
    else sorry :=
by
  sorry

end largest_percentage_increase_between_l597_597331


namespace asymptotic_lines_of_hyperbola_l597_597573

-- Define the given equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := 3 * x^2 - y^2 = 3

-- Define what it means for y to be equal to ±sqrt(3)x
def asymptotic_eq (x y : ℝ) : Prop := y = sqrt 3 * x ∨ y = -sqrt 3 * x

-- State the theorem to prove
theorem asymptotic_lines_of_hyperbola (x y : ℝ) (h : hyperbola_eq x y) : asymptotic_eq x y := 
sorry

end asymptotic_lines_of_hyperbola_l597_597573


namespace centroid_path_area_correct_l597_597526

noncomputable def centroid_path_area (AB : ℝ) (A B C : ℝ × ℝ) (O : ℝ × ℝ) : ℝ :=
  let R := AB / 2
  let radius_of_path := R / 3
  let area := Real.pi * radius_of_path ^ 2
  area

theorem centroid_path_area_correct (AB : ℝ) (A B C : ℝ × ℝ)
  (hAB : AB = 32)
  (hAB_diameter : (∃ O : ℝ × ℝ, dist O A = dist O B ∧ dist A B = 2 * dist O A))
  (hC_circle : ∃ O : ℝ × ℝ, dist O C = AB / 2 ∧ C ≠ A ∧ C ≠ B):
  centroid_path_area AB A B C (0, 0) = (256 / 9) * Real.pi := by
  sorry

end centroid_path_area_correct_l597_597526


namespace categorize_numbers_l597_597373

-- Define the set of numbers provided
def numbers : Set ℝ := {10, -2.5, 0.8, 0, -real.pi, 11, -9, -4.2, -2}

-- Define the integer set expected
def integer_set : Set ℝ := {10, 0, 11, -9, -2}

-- Define the negative number set expected
def negative_set : Set ℝ := {-2.5, -real.pi, -9, -4.2, -2}

-- The theorem that needs to be proved
theorem categorize_numbers :
  (∀ x ∈ numbers, (x ∈ integer_set ↔ x ∈ Set.Ici (-9) ∧ x ∈ Int))
  ∧ 
  (∀ x ∈ numbers, (x ∈ negative_set ↔ x < 0)) :=
sorry

end categorize_numbers_l597_597373


namespace find_n_l597_597416

-- Define the polynomial function
def polynomial (n : ℤ) : ℤ :=
  n^4 + 2 * n^3 + 6 * n^2 + 12 * n + 25

-- Define the condition that n is a positive integer
def is_positive_integer (n : ℤ) : Prop :=
  n > 0

-- Define the condition that polynomial is a perfect square
def is_perfect_square (k : ℤ) : Prop :=
  ∃ m : ℤ, m^2 = k

-- The theorem we need to prove
theorem find_n (n : ℤ) (h1 : is_positive_integer n) (h2 : is_perfect_square (polynomial n)) : n = 8 :=
sorry

end find_n_l597_597416


namespace cevian_product_one_l597_597524

theorem cevian_product_one
  (A B C C1 A1 B1 : Point)
  (hC1 : SegmentIntersect AB e C1)
  (hA1 : SegmentIntersect BC e A1)
  (hB1 : SegmentIntersect CA e B1) :
  (AC_1 / BC_1) * (BA_1 / CA_1) * (CB_1 / AB_1) = 1 :=
sorry

end cevian_product_one_l597_597524


namespace ratio_S6_S3_l597_597407

theorem ratio_S6_S3 (a : ℝ) (q : ℝ) (h : a + 8 * a * q^3 = 0) : 
  (a * (1 - q^6) / (1 - q)) / (a * (1 - q^3) / (1 - q)) = 9 / 8 :=
by
  sorry

end ratio_S6_S3_l597_597407


namespace max_magnitude_vector_c_l597_597289

noncomputable def vector_a : ℝ × ℝ := (1, 0)
noncomputable def vector_b : ℝ × ℝ := (0, 1)

def satisfies_condition (c : ℝ × ℝ) : Prop :=
  let (m, n) := c in
  (m + 1) * m + n * (n + 1) = 0

def magnitude (v : ℝ × ℝ) : ℝ :=
  let (x, y) := v in
  real.sqrt (x * x + y * y)

theorem max_magnitude_vector_c : ∃ c : ℝ × ℝ, satisfies_condition c ∧ magnitude c = real.sqrt 2 := sorry

end max_magnitude_vector_c_l597_597289


namespace divides_sum_if_divides_polynomial_l597_597175

theorem divides_sum_if_divides_polynomial (x y : ℕ) : 
  x^2 ∣ x^2 + x * y + x + y → x^2 ∣ x + y :=
by
  sorry

end divides_sum_if_divides_polynomial_l597_597175


namespace repeating_block_length_of_7_div_13_is_6_l597_597605

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l597_597605


namespace tangent_line_eqn_monotonic_intervals_and_extrema_l597_597438

noncomputable def f (a x : ℝ) := x - a * Real.log x

theorem tangent_line_eqn {a : ℝ} (h : a = 2) : 
  let f := f a in 
  let tangent_line_eqn := (λ x y : ℝ, x + y - 2 = 0) in
  tangent_line_eqn 1 (f 1) :=
by sorry

theorem monotonic_intervals_and_extrema (a : ℝ) :
    (a ≤ 0 → ∀ x : ℝ, x > 0 → ((f a).deriv x > 0)) ∧ 
    (a > 0 → let f' := (f a).deriv in
              (∃ x : ℝ, x = a ∧ (f' x = 0)) ∧ 
              (∀ x : ℝ, (0 < x ∧ x < a) → f' x < 0) ∧ 
              (∀ x : ℝ, (x > a) → f' x > 0) ∧ 
              (∃ y : ℝ, x = a → f a x = y ∧ y = a - a * Real.log a)) :=
by sorry

end tangent_line_eqn_monotonic_intervals_and_extrema_l597_597438


namespace correct_calculation_l597_597272

theorem correct_calculation : sqrt 27 / sqrt 3 = 3 :=
by
  sorry

end correct_calculation_l597_597272


namespace sum_of_solutions_eq_16_l597_597385

theorem sum_of_solutions_eq_16 :
  (∑ x in ({x : ℝ | (x - 8) ^ 2 = 49}).toFinset, x) = 16 :=
by
  sorry

end sum_of_solutions_eq_16_l597_597385


namespace player_1_points_after_13_rotations_l597_597679

-- Add necessary definitions and state the problem in Lean
def sectors : Fin 16 → ℕ
| ⟨0, _⟩ := 0
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 2
| ⟨3, _⟩ := 3
| ⟨4, _⟩ := 4
| ⟨5, _⟩ := 5
| ⟨6, _⟩ := 6
| ⟨7, _⟩ := 7
| ⟨8, _⟩ := 8
| ⟨9, _⟩ := 7
| ⟨10, _⟩ := 6
| ⟨11, _⟩ := 5
| ⟨12, _⟩ := 4
| ⟨13, _⟩ := 3
| ⟨14, _⟩ := 2
| ⟨15, _⟩ := 1

def points_earned (player_offset : Fin 16) (rotations : ℕ) : ℕ :=
List.sum (List.map sectors
  (List.map (λ n => (Fin.add (Fin.ofNat n) player_offset)) (List.range rotations)))

theorem player_1_points_after_13_rotations 
  (p5_points : points_earned ⟨5, by decide⟩ 13 = 72)
  (p9_points : points_earned ⟨9, by decide⟩ 13 = 84) :
  points_earned ⟨1, by decide⟩ 13 = 20 := 
sorry

end player_1_points_after_13_rotations_l597_597679


namespace tangent_line_at_1_strictly_decreasing_on_R_l597_597439

-- Define the function
def f (a x : ℝ) : ℝ := -x^3 + a*x^2 + (1 - 2*a)*x + a

-- First part: equation of the tangent line at x = 1
theorem tangent_line_at_1 (a : ℝ) :
  let f' (a x : ℝ) : ℝ := -3*x^2 + 2*a*x + (1 - 2*a)
  (f' a 1) = -2 ∧ (f a 1) = 0 →
  ∃ b : ℝ, ∀ x y: ℝ, y - (f a 1) = -2*(x - 1) ↔ 2*x + y - 2 = 0 :=
by {
  intros,
  existsi 0,
  intros,
  sorry
}

-- Second part: strictly decreasing function on ℝ
theorem strictly_decreasing_on_R (a : ℝ) :
  ∀ x : ℝ, (let f' := -3*x^2 + 2*a*x + (1 - 2*a) 
              in f' < 0) ↔ a ∈ set.Ioo (3 - real.sqrt 6) (3 + real.sqrt 6) :=
by {
  intros,
  sorry
}

end tangent_line_at_1_strictly_decreasing_on_R_l597_597439


namespace value_of_a_l597_597458

theorem value_of_a (a : ℚ) (h : 2 * a + a / 2 = 9 / 2) : a = 9 / 5 :=
by
  sorry

end value_of_a_l597_597458


namespace question1_question2_l597_597890

section Problem

variable (x a b m : ℝ)

def A : Set ℝ := { x | x^2 + a * x + b <= 0 }
def B : Set ℝ := { x | x^2 - 2 * m * x + m^2 - 4 < 0 }
def neg_q : Set ℝ := { x | x >= m + 2 ∨ x <= m - 2 }

-- The first part: Prove that a + b = -7
theorem question1 (hA : A = { x | -1 <= x ∧ x <= 4 }) : a + b = -7 :=
sorry

-- The second part: Prove the range of m
theorem question2 (hA : A = { x | -1 <= x ∧ x <= 4 }) 
                  (necessary_condition : ∀ x, x ∈ A → x ∈ neg_q) : m ≤ -3 ∨ m ≥ 6 :=
sorry

end Problem

end question1_question2_l597_597890


namespace gunther_initial_free_time_l597_597940

def total_cleaning_time (C : ℕ) : ℝ :=
  (45 + 60 + 30 + 5 * C) / 60

def final_free_time (initial_free_time : ℝ) (C : ℕ) : ℝ :=
  initial_free_time - total_cleaning_time C

theorem gunther_initial_free_time (C : ℕ) : initial_free_time >= 2.75 :=
  let initial_free_time := total_cleaning_time 0 + 0.5
  sorry

end gunther_initial_free_time_l597_597940


namespace range_of_m_l597_597061

/-- Define the function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := abs ((4 / x) - a * x)

/-- Prove the required range for m -/
theorem range_of_m :
  ∀ m : ℝ, (∀ a > 0, ∃ x₀ ∈ set.Icc (1:ℝ) 4, f a x₀ ≥ m) ↔ m ≤ 3 := 
by 
  sorry

end range_of_m_l597_597061


namespace concorrence_of_cevins_l597_597650

-- Definitions for points and lines
noncomputable def A1 : Point := sorry -- Point on line BC
noncomputable def B1 : Point := sorry -- Point on line CA
noncomputable def C1 : Point := sorry -- Point on line AB
noncomputable def A2 : Point := sorry -- Point on line BC
noncomputable def B2 : Point := sorry -- Point on line CA
noncomputable def C2 : Point := sorry -- Point on line AB

theorem concorrence_of_cevins :
  ∀ {A B C A1 B1 C1 A2 B2 C2 : Point},
    (dist A B / dist B C = dist A1 C / dist B A1) →
    (dist B C / dist C A = dist B1 A / dist C B1) →
    (dist C A / dist A B = dist C1 B / dist A C1) →
    concurrent A A2 B B2 C C2 :=
begin
  intros,
  sorry
end

end concorrence_of_cevins_l597_597650


namespace color_distribution_exists_l597_597343

-- Define the problem structure
def color := fin 4 -- 4 colors

-- Define the 100x100 table with 4 colors
def table := fin 100 → fin 100 → color

-- Problem statement
theorem color_distribution_exists (T : table)
    (h_row : ∀ r : fin 100, (finset.univ.filter (λ c, T r c = 0)).card = 25 ∧ 
                            (finset.univ.filter (λ c, T r c = 1)).card = 25 ∧
                            (finset.univ.filter (λ c, T r c = 2)).card = 25 ∧
                            (finset.univ.filter (λ c, T r c = 3)).card = 25)
    (h_col : ∀ c : fin 100, (finset.univ.filter (λ r, T r c = 0)).card = 25 ∧ 
                            (finset.univ.filter (λ r, T r c = 1)).card = 25 ∧
                            (finset.univ.filter (λ r, T r c = 2)).card = 25 ∧
                            (finset.univ.filter (λ r, T r c = 3)).card = 25) :
    ∃ r1 r2 c1 c2 : fin 100, r1 ≠ r2 ∧ c1 ≠ c2 ∧ 
                      T r1 c1 ≠ T r1 c2 ∧ 
                      T r1 c1 ≠ T r2 c1 ∧ 
                      T r1 c2 ≠ T r2 c2 ∧ 
                      T r2 c1 ≠ T r2 c2 :=
begin
    sorry -- Proof goes here
end

end color_distribution_exists_l597_597343


namespace doubling_profit_condition_l597_597299

-- Definitions
def purchase_price : ℝ := 210
def initial_selling_price : ℝ := 270
def initial_items_sold : ℝ := 30
def profit_per_item (selling_price : ℝ) : ℝ := selling_price - purchase_price
def daily_profit (selling_price : ℝ) (items_sold : ℝ) : ℝ := profit_per_item selling_price * items_sold
def increase_in_items_sold_per_yuan (reduction : ℝ) : ℝ := 3 * reduction

-- Condition: Initial daily profit
def initial_daily_profit : ℝ := daily_profit initial_selling_price initial_items_sold

-- Proof problem
theorem doubling_profit_condition (reduction : ℝ) :
  daily_profit (initial_selling_price - reduction) (initial_items_sold + increase_in_items_sold_per_yuan reduction) = 2 * initial_daily_profit :=
sorry

end doubling_profit_condition_l597_597299


namespace circumcircle_diameter_correct_l597_597469

open Triangle
open Real

def circumcircle_diameter (a : ℝ) (B : Real.Angle) (S : ℝ) : ℝ :=
  if a = 1 ∧ B = Real.Angle.pi / 4 ∧ S = 2 then 5 * Real.sqrt 2 else 0

theorem circumcircle_diameter_correct :
  ∀ (a : ℝ) (B : Real.Angle) (S : ℝ), a = 1 → B = Real.Angle.pi / 4 → S = 2 →
    circumcircle_diameter a B S = 5 * Real.sqrt 2 :=
begin
  intros a B S ha hB hS,
  unfold circumcircle_diameter,
  rw [if_pos],
  { refl },
  { exact ⟨ha, ⟨hB, hS⟩⟩ }
end

end circumcircle_diameter_correct_l597_597469


namespace work_done_l597_597297

noncomputable def F (x : ℝ) := 3 * x^2 - 2 * x + 3

theorem work_done : (∫ x in 1..5, F x) = 112 := by
  sorry

end work_done_l597_597297


namespace sam_wins_probability_l597_597552

theorem sam_wins_probability (hitting_probability missing_probability : ℚ)
    (hit_prob : hitting_probability = 2/5)
    (miss_prob : missing_probability = 3/5) : 
    let p := hitting_probability / (1 - missing_probability ^ 2)
    p = 5 / 8 :=
by
    sorry

end sam_wins_probability_l597_597552


namespace eccentricity_of_ellipse_l597_597050

theorem eccentricity_of_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) 
  (h_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → ∃ c, x = c ∧ y = c) :
  let e := (Real.sqrt 5 - 1) / 2 in ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → 
  ⁇ sorry :=
begin
  sorry
end

end eccentricity_of_ellipse_l597_597050


namespace proof_problem_l597_597434

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (2 * k - 1) * x + k
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := Real.log (x + k) / Real.log 2

theorem proof_problem :
  (∀ x k : ℝ, f k x = k * x^2 + (2 * k - 1) * x + k) → 
  (∀ x k : ℝ, g k x = Real.log (x + k) / Real.log 2) →
  (∀ k : ℝ, f k 0 = 7 → 
    (∃ m : ℝ, (∀ x : ℝ, x ∈ set.Ici 9 → has_min_on (g k) (set.Ici 9) → m = 4) ∧
      (0 < g k 1 ∧ g k 1 ≤ 5 ∧ 
      (∀ x : ℝ, x ∈ set.Icc 0 2 → has_min_on (f k) (set.Icc 0 2) → f k x ≥ 4) → 
        (k ∈ set.Icc (2/3) 31)))) :=
by 
  sorry

end proof_problem_l597_597434


namespace first_player_has_winning_strategy_l597_597016

-- Define the initial heap sizes and rules of the game.
def initial_heaps : List Nat := [38, 45, 61, 70]

-- Define a function that checks using the rules whether the first player has a winning strategy given the initial heap sizes.
def first_player_wins : Bool :=
  -- placeholder for the actual winning strategy check logic
  sorry

-- Theorem statement referring to the equivalency proof problem where player one is established to have the winning strategy.
theorem first_player_has_winning_strategy : first_player_wins = true :=
  sorry

end first_player_has_winning_strategy_l597_597016


namespace wendy_bouquets_l597_597267

-- Defining the initial number of flowers, the number that wilted, and the number per bouquet
def initial_flowers : ℕ := 45
def wilted_flowers : ℕ := 35
def flowers_per_bouquet : ℕ := 5

-- Calculating the remaining flowers and the number of bouquets
def remaining_flowers := initial_flowers - wilted_flowers
def number_of_bouquets := remaining_flowers / flowers_per_bouquet

-- Proving the main statement: Wendy could still make 2 bouquets
theorem wendy_bouquets : 
  remaining_flowers = initial_flowers - wilted_flowers → 
  number_of_bouquets = 2 :=
by {
  intros h,
  unfold remaining_flowers number_of_bouquets,
  rw h,
  norm_num,
  sorry,
}

end wendy_bouquets_l597_597267


namespace difference_of_two_numbers_l597_597657

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 15) (h2 : x^2 - y^2 = 150) : x - y = 10 :=
by
  sorry

end difference_of_two_numbers_l597_597657


namespace megan_popsicles_consumed_l597_597528

noncomputable def popsicles_consumed_in_time_period (time: ℕ) (interval: ℕ) : ℕ :=
  (time / interval)

theorem megan_popsicles_consumed:
  popsicles_consumed_in_time_period 315 30 = 10 :=
by
  sorry

end megan_popsicles_consumed_l597_597528


namespace repeating_block_length_7_div_13_l597_597628

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l597_597628


namespace maximize_integral_k_l597_597151

theorem maximize_integral_k (f : ℝ → ℝ) (k : ℝ) 
  (h_cont : Continuous f)
  (h_eq : ∀ x, f x = 1 + k * ∫ t in -π/2 .. π/2, f t * Real.sin (x - t)) :
  k = 2 / π :=
sorry

end maximize_integral_k_l597_597151


namespace sequences_are_coprime_l597_597248

/-- Define the sequences (a_n) and (b_n) by:
a_1 = b_1 = 1,
a_{n+1} = a_n + b_n,
b_{n+1} = a_n * b_n
for n = 1, 2, ...
-/
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := a n + b n

/-- Define the sequence b_n -/
def b : ℕ → ℕ
| 0       := 1
| (n + 1) := a n * b n

theorem sequences_are_coprime :
  ∀ i j : ℕ, i ≠ j → Nat.coprime (a i) (a j) :=
by
  sorry

end sequences_are_coprime_l597_597248


namespace fastest_bricklayer_time_l597_597879

theorem fastest_bricklayer_time (r : ℕ → ℝ) (h : ∀ i, r i ≤ 1/36 ∧ r 0 + r 1 + r 2 + r 3 + r 4 = 1/3) : ∃ n, n = 270 :=
begin
  sorry
end

end fastest_bricklayer_time_l597_597879


namespace sum_of_variables_is_seven_l597_597443

-- Given matrices expressed as condition
def matrix1 (a b c d : ℝ) : Matrix 3 3 ℝ := 
  ![[2 * a, 2, 3 * b], 
    [1, 3, 2], 
    [d, 4, c]]

def matrix2 (e f g h : ℝ) : Matrix 3 3 ℝ := 
  ![[-10, e, -15], 
    [f, -20, g], 
    [3, h, 5]]

-- This is the hypothesis: matrices are inverses
axiom matrices_are_inverses (a b c d e f g h : ℝ) :
  (matrix1 a b c d) ⬝ (matrix2 e f g h) = 1

-- This is the theorem we need to prove
theorem sum_of_variables_is_seven (a b c d e f g h : ℝ) :
  (matrix1 a b c d) ⬝ (matrix2 e f g h) = (1 : Matrix 3 3 ℝ) → 
  a + b + c + d + e + f + g + h = 7 :=
by
  sorry  -- Proof is omitted as per instructions

end sum_of_variables_is_seven_l597_597443


namespace mean_is_six_greater_than_median_l597_597008

theorem mean_is_six_greater_than_median (x a : ℕ) 
  (h1 : (x + a) + (x + 4) + (x + 7) + (x + 37) + x == 5 * (x + 10)) :
  a = 2 :=
by
  -- proof goes here
  sorry

end mean_is_six_greater_than_median_l597_597008


namespace replace_asterisks_l597_597767

theorem replace_asterisks (x : ℕ) (h : (x / 20) * (x / 180) = 1) : x = 60 := by
  sorry

end replace_asterisks_l597_597767


namespace find_c_l597_597931

theorem find_c (c : ℝ) : (∀ x : ℝ, -2 < x ∧ x < 1 → x^2 + x - c < 0) → c = 2 :=
by
  intros h
  -- Sorry to skip the proof
  sorry

end find_c_l597_597931


namespace johns_apartment_number_l597_597995

theorem johns_apartment_number (car_reg : Nat) (apartment_num : Nat) 
  (h_car_reg_sum : car_reg = 834205) 
  (h_car_digits : (8 + 3 + 4 + 2 + 0 + 5 = 22)) 
  (h_apartment_digits : ∃ (d1 d2 d3 : Nat), d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ d1 + d2 + d3 = 22) :
  apartment_num = 985 :=
by
  sorry

end johns_apartment_number_l597_597995


namespace find_n_l597_597239

theorem find_n :
  ∃ n : ℕ, n > 0 ∧ n^6 = 734851474594578436096 :=
begin
  use 3004,
  split,
  { exact nat.succ_pos 3003 },
  { norm_num }
end

end find_n_l597_597239


namespace max_candies_consumed_l597_597494

/-- The maximum number of candies Karlson could have consumed in 50 minutes,
starting with 50 ones on the board, given that each minute he erases two randomly chosen 
numbers and writes their sum on the board while consuming candies equal to the product 
of the two erased numbers, is exactly 1225. -/
theorem max_candies_consumed : 
  ∀ (initial_numbers : List ℕ), 
    (initial_numbers.length = 50 ∧ All initial_numbers (λ x, x = 1)) →
      (∃ candies_consumed : ℕ, candies_consumed = 1225) := 
by
  intros initial_numbers h_initial
  existsi 1225
  sorry

end max_candies_consumed_l597_597494


namespace jake_needs_total_hours_to_pay_off_debts_l597_597123

-- Define the conditions for the debts and payments
variable (debtA debtB debtC : ℝ)
variable (paymentA paymentB paymentC : ℝ)
variable (task1P task2P task3P task4P task5P task6P : ℝ)
variable (task2Payoff task4Payoff task6Payoff : ℝ)

-- Assume provided values
noncomputable def total_hours_needed : ℝ :=
  let remainingA := debtA - paymentA
  let remainingB := debtB - paymentB
  let remainingC := debtC - paymentC
  let hoursTask1 := (remainingA - task2Payoff) / task1P
  let hoursTask2 := task2Payoff / task2P
  let hoursTask3 := (remainingB - task4Payoff) / task3P
  let hoursTask4 := task4Payoff / task4P
  let hoursTask5 := (remainingC - task6Payoff) / task5P
  let hoursTask6 := task6Payoff / task6P
  hoursTask1 + hoursTask2 + hoursTask3 + hoursTask4 + hoursTask5 + hoursTask6

-- Given our specific problem conditions
theorem jake_needs_total_hours_to_pay_off_debts :
  total_hours_needed 150 200 250 60 80 100 15 12 20 10 25 30 30 40 60 = 20.1 :=
by
  sorry

end jake_needs_total_hours_to_pay_off_debts_l597_597123


namespace student_correct_answers_l597_597762

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 79) : C = 93 :=
by
  sorry

end student_correct_answers_l597_597762


namespace total_amount_spent_l597_597757

variable (your_spending : ℝ) (friend_spending : ℝ)
variable (h1 : friend_spending = your_spending + 3) (h2 : friend_spending = 10)

theorem total_amount_spent : your_spending + friend_spending = 17 :=
by sorry

end total_amount_spent_l597_597757


namespace solve_fraction_equation_l597_597206

theorem solve_fraction_equation :
  ∀ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) → x = -2 / 11 :=
by
  intro x
  intro h
  sorry

end solve_fraction_equation_l597_597206


namespace factorial_trailing_zeros_l597_597340

theorem factorial_trailing_zeros (n : ℕ) : 
  let num_factors_of_5 := ∑ k in (Finset.range (n.log 5 + 1)), n / 5^k in
  num_factors_of_5 = 502 :=
by
  sorry

end factorial_trailing_zeros_l597_597340


namespace player_1_points_after_13_rotations_l597_597677

-- Add necessary definitions and state the problem in Lean
def sectors : Fin 16 → ℕ
| ⟨0, _⟩ := 0
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 2
| ⟨3, _⟩ := 3
| ⟨4, _⟩ := 4
| ⟨5, _⟩ := 5
| ⟨6, _⟩ := 6
| ⟨7, _⟩ := 7
| ⟨8, _⟩ := 8
| ⟨9, _⟩ := 7
| ⟨10, _⟩ := 6
| ⟨11, _⟩ := 5
| ⟨12, _⟩ := 4
| ⟨13, _⟩ := 3
| ⟨14, _⟩ := 2
| ⟨15, _⟩ := 1

def points_earned (player_offset : Fin 16) (rotations : ℕ) : ℕ :=
List.sum (List.map sectors
  (List.map (λ n => (Fin.add (Fin.ofNat n) player_offset)) (List.range rotations)))

theorem player_1_points_after_13_rotations 
  (p5_points : points_earned ⟨5, by decide⟩ 13 = 72)
  (p9_points : points_earned ⟨9, by decide⟩ 13 = 84) :
  points_earned ⟨1, by decide⟩ 13 = 20 := 
sorry

end player_1_points_after_13_rotations_l597_597677


namespace polynomial_transformation_l597_597162

-- Given the conditions of the polynomial function g and the provided transformation
-- We aim to prove the equivalence in a mathematically formal way using Lean

theorem polynomial_transformation (g : ℝ → ℝ) (h : ∀ x : ℝ, g (x^2 + 2) = x^4 + 5 * x^2 + 1) :
  ∀ x : ℝ, g (x^2 - 2) = x^4 - 3 * x^2 - 3 :=
by
  intro x
  sorry

end polynomial_transformation_l597_597162


namespace product_remainder_mod_5_l597_597874

theorem product_remainder_mod_5 :
  (1024 * 1455 * 1776 * 2018 * 2222) % 5 = 0 := 
sorry

end product_remainder_mod_5_l597_597874


namespace course_selection_ways_l597_597777

theorem course_selection_ways :
  let num_type_A := 4 in
  let num_type_B := 2 in
  let num_courses_selected := 3 in
  (num_type_A ∈ ℕ ∧ num_type_B ∈ ℕ ∧ num_courses_selected ∈ ℕ) →
  (num_type_A = 4 ∧ num_type_B = 2 ∧ num_courses_selected = 3) →
  let total_ways :=
    (nat.choose num_type_A 1 * nat.choose num_type_B 2) +
    (nat.choose num_type_A 2 * nat.choose num_type_B 1) in
  total_ways = 16 :=
by
  intros
  sorry

end course_selection_ways_l597_597777


namespace minimum_value_f_l597_597168

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ t ≥ 0, ∀ (x y : ℝ), 0 < x → 0 < y → f x y ≥ t ∧ t = 4 * Real.sqrt 2 :=
sorry

end minimum_value_f_l597_597168


namespace find_cos_2beta_l597_597055

noncomputable def alpha := sorry -- Placeholders for the acute angle α
noncomputable def beta := sorry -- Placeholders for the acute angle β

axiom alpha_acute : 0 < alpha ∧ alpha < π / 2
axiom beta_acute : 0 < beta ∧ beta < π / 2
axiom tan_alpha : Real.tan alpha = 7
axiom sin_alpha_minus_beta : Real.sin (alpha - beta) = sqrt 10 / 10

theorem find_cos_2beta : Real.cos (2 * beta) = -3 / 5 :=
by
  -- Proof goes here
  sorry

end find_cos_2beta_l597_597055


namespace conic_section_propositions_l597_597812

theorem conic_section_propositions
  (A B : ℝ × ℝ)
  (P : ℝ × ℝ)
  (dAB : ℝ)
  (h1 : ∀ A B P, dist P A + dist P B = 8 → ¬is_ellipse A B P)
  (h2 : ∀ A B P, dist P A = 10 - dist P B ∧ dist A B = 8 → ∃ c a, a = 5 ∧ c = 4 ∧ dist P A ≤ a + c)
  (h3 : ∀ A B P, dist P A - dist P B = 6 → ¬is_hyperbola A B P)
  (h4 : (∃ x y, x^2 / 16 - y^2 / 10 = 1) ∧ (∃ x y, x^2 / 30 + y^2 / 4 = 1) → has_same_foci (16, 10) (30, 4)) :
  ∃ (true_props : list (ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop)),
    true_props = [h2, h4] := 
  sorry

end conic_section_propositions_l597_597812


namespace repeating_block_length_7_div_13_l597_597612

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597612


namespace mathematics_value_correct_l597_597649

def letter_value (pos : ℕ) : ℤ :=
  match pos % 13 with
  | 0 => -3
  | 1 => -3
  | 2 => -2
  | 3 => -1
  | 4 => 0
  | 5 => 1
  | 6 => 2
  | 7 => 3
  | 8 => 2
  | 9 => 1
  | 10 => 0
  | 11 => -1
  | 12 => -2
  | _ => 0  -- This case is actually impossible since pos % 13 is always in {0, ..., 12}

def calculate_sum : ℤ :=
  let letters := "mathematics".toList.map (λ c => c.toNat - 'a'.toNat + 1)
  letters.sum (λ pos => letter_value pos)

theorem mathematics_value_correct : calculate_sum = -1 := sorry

end mathematics_value_correct_l597_597649


namespace find_f_0_f_1_l597_597781

noncomputable def f : ℤ → ℤ := sorry

theorem find_f_0_f_1 :
  (∀ x : ℤ, f(x + 5) - f(x) = 10 * x + 25) →
  (∀ x : ℤ, f(x^3 - 1) = (f(x) - x)^3 + x^3 - 3) →
  (f(0) = -1 ∧ f(1) = 2) :=
by
  intros h1 h2
  sorry

end find_f_0_f_1_l597_597781


namespace find_a_find_b_estimate_probability_l597_597804

noncomputable def frequency_to_count (n : ℕ) (f : ℚ) : ℚ :=
  n * f

theorem find_a (n : ℕ) (f : ℚ) (h1 : n = 20) (h2 : f = 0.7) : frequency_to_count n f = 14 :=
by
  rw [h1, h2]
  norm_num

theorem find_b (count : ℕ) (n : ℕ) (h1 : count = 88) (h2 : n = 160) : (count : ℚ) / n = 0.55 :=
by
  rw [h1, h2]
  norm_num

def stabilized_frequency : ℚ := 0.55

theorem estimate_probability : stabilized_frequency = 0.55 :=
by
  refl

end find_a_find_b_estimate_probability_l597_597804


namespace min_value_of_M_l597_597523

noncomputable def min_max_sum (a b c d e : ℕ) (h : a + b + c + d + e = 2345) : ℕ :=
  Nat.max (Nat.max (a + b) (Nat.max (b + c) (Nat.max (c + d) (d + e))))

theorem min_value_of_M {a b c d e : ℕ} (h : a + b + c + d + e = 2345) :
  min_max_sum a b c d e h = 782 := 
sorry

end min_value_of_M_l597_597523


namespace unit_digit_pow_l597_597445

theorem unit_digit_pow :
  (x : ℝ) (a : ℝ) 
  (h : x = ( (√((a - 2) * (|a| - 1)) + √((a - 2) * (1 - |a|))) / (1 + 1 / (1 - a)) + (5 * a + 1) / (1 - a) ) ^ 1988)
  (cond : ((a - 2) * (|a| - 1)) = 0) :
  x % 10 = 6 := 
sorry

end unit_digit_pow_l597_597445


namespace arc_length_of_quarter_circle_l597_597575

theorem arc_length_of_quarter_circle (C : ℝ) (α : ℝ) (hC : C = 72) (hα : α = 90) : ∃ L : ℝ, L = 18 :=
by
  use 18
  sorry

end arc_length_of_quarter_circle_l597_597575


namespace angle_bisector_l597_597885

theorem angle_bisector (A E F G B C : Point)
  (k : Circle)
  (h_tangent1 : Tangent (A, E) k)
  (h_tangent2 : Tangent (A, F) k)
  (h_midpoint : Midpoint G E F)
  (h_intersection : LineThrough ABC k) :
  AngleBisector (LineThrough E F) (Angle B G C) :=
  sorry

end angle_bisector_l597_597885


namespace series_150_result_l597_597181

noncomputable def series (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1).filter (λ k, k > 0), (-1)^k * ((k^3 + k^2 + k + 1) / (k.factorial : ℚ))

theorem series_150_result :
  ∃ d e f : ℕ, (series 150 = d / e.factorial - f) ∧ d + e + f = 45305 :=
by
  use 45154
  use 150
  use 1
  -- Here we need to prove series 150 = 45154 / 150! - 1
  -- and 45154 + 150 + 1 = 45305
  sorry  -- Proof is omitted as per the instructions.

end series_150_result_l597_597181


namespace range_of_g_l597_597760

noncomputable def g (x : ℝ) : ℝ := (x^3 - 2 * x) / (x^2 - 2 * x + 2)

theorem range_of_g : set.range g = set.univ :=
by
  sorry

end range_of_g_l597_597760


namespace onion_to_carrot_ratio_l597_597182

theorem onion_to_carrot_ratio (p c o g : ℕ) (h1 : 6 * p = c) (h2 : c = o) (h3 : g = 1 / 3 * o) (h4 : p = 2) (h5 : g = 8) : o / c = 1 / 1 :=
by
  sorry

end onion_to_carrot_ratio_l597_597182


namespace distance_from_center_of_circle_to_MK_l597_597778

theorem distance_from_center_of_circle_to_MK :
  ∀ (A B C M K O : Type) [metric_space A]
  [metric_space B] [metric_space C] [metric_space M]
  [metric_space K] [metric_space O],
  let hypotenuse := dist A B,
      radius := dist O A in
  (circle_centered_at O).passes_through A ∧
  (circle_centered_at O).passes_through B ∧
  (circle_centered_at O).intersects_legs_of_triangle M K A B C →
  dist O (line.mk M K) = hypotenuse / 2 :=
by
  intro A B C M K O h_A_B_O h_circle_passes_through M K h_circle_intersects_legs,
  sorry

end distance_from_center_of_circle_to_MK_l597_597778


namespace tony_total_winning_l597_597712

theorem tony_total_winning : 
  ∀ (num_tickets num_winning_numbers_per_ticket winnings_per_number : ℕ),
  num_tickets = 3 → 
  num_winning_numbers_per_ticket = 5 →
  winnings_per_number = 20 →
  num_tickets * num_winning_numbers_per_ticket * winnings_per_number = 300 :=
by {
  intros num_tickets num_winning_numbers_per_ticket winnings_per_number h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
}

end tony_total_winning_l597_597712


namespace repeat_block_of_7_div_13_l597_597636

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l597_597636


namespace cos_alpha_minus_beta_is_23_over_27_l597_597394

noncomputable def cos_diff_alpha_beta 
  (α β : ℝ)
  (cos_alpha : ℝ)
  (cos_alpha_plus_beta : ℝ)
  (h_alpha : 0 < α ∧ α < π / 2)
  (h_beta : 0 < β ∧ β < π / 2)
  (h_cos_alpha : cos α = 1 / 3)
  (h_cos_alpha_plus_beta : cos (α + β) = -1 / 3)
  : ℝ :=
  cos (α - β)

theorem cos_alpha_minus_beta_is_23_over_27 
  (α β : ℝ)
  (cos_alpha : ℝ)
  (cos_alpha_plus_beta : ℝ)
  (h_alpha : 0 < α ∧ α < π / 2)
  (h_beta : 0 < β ∧ β < π / 2)
  (h_cos_alpha : cos α = 1 / 3)
  (h_cos_alpha_plus_beta : cos (α + β) = -1 / 3) :
  cos_diff_alpha_beta α β cos_alpha cos_alpha_plus_beta h_alpha h_beta h_cos_alpha h_cos_alpha_plus_beta = 23 / 27 :=
sorry

end cos_alpha_minus_beta_is_23_over_27_l597_597394


namespace product_xyz_equals_one_l597_597892

theorem product_xyz_equals_one (x y z : ℝ) (h1 : x + (1/y) = 2) (h2 : y + (1/z) = 2) : x * y * z = 1 := 
by
  sorry

end product_xyz_equals_one_l597_597892


namespace find_y_l597_597963

theorem find_y (x y : ℤ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 :=
sorry

end find_y_l597_597963


namespace sufficient_not_necessary_condition_l597_597159

variable (a b c : ℝ)

-- Define the condition that the sequence forms a geometric sequence
def geometric_sequence (a1 a2 a3 a4 a5 : ℝ) :=
  ∃ q : ℝ, q ≠ 0 ∧ a1 * q = a2 ∧ a2 * q = a3 ∧ a3 * q = a4 ∧ a4 * q = a5

-- Lean statement proving the problem
theorem sufficient_not_necessary_condition :
  (geometric_sequence 1 a b c 16) → (b = 4) ∧ ¬ (b = 4 → geometric_sequence 1 a b c 16) :=
sorry

end sufficient_not_necessary_condition_l597_597159


namespace angle_C_in_triangle_l597_597121

theorem angle_C_in_triangle 
  (A B C : ℝ) 
  (h₁ : |2 * Real.sin A - 1| + |Real.sqrt 2 / 2 - Real.cos B| = 0) 
  (h₂ : A + B + C = 180) 
  (h₃ : 0 < A ∧ A < 180) 
  (h₄ : 0 < B ∧ B < 180) 
  (h₅ : 0 < C ∧ C < 180) 
  : C = 105 :=
begin
  sorry
end

end angle_C_in_triangle_l597_597121


namespace students_distribution_l597_597795

theorem students_distribution {students : ℕ} (h_students : students = 5) :
  ∃ (ways : ℕ), ways = 150 ∧ 
    (∃ (c1 c2 : ℕ), c1 = (nat.choose 5 2) * (nat.choose 3 2) * (algebra.factorial 3 / algebra.factorial 2) ∧
                    c2 = (nat.choose 5 3) * (algebra.factorial 3) ∧ 
                    ways = c1 + c2) :=
  sorry

end students_distribution_l597_597795


namespace player_1_points_l597_597668

-- Definition: point distribution on the table.
noncomputable def sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

-- Conditions
axiom player_5_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(5 + i) % 16]) = 72
axiom player_9_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(9 + i) % 16]) = 84

-- Question translated to proof statement:
theorem player_1_points (rotations : ℕ) (p5_points : ℕ) (p9_points : ℕ) :
  rotations = 13 → p5_points = 72 → p9_points = 84 →
  ∑ i in finset.range rotations, (sector_points[(1 + i) % 16]) = 20 :=
by
  sorry

end player_1_points_l597_597668


namespace total_amount_paid_l597_597319

theorem total_amount_paid {p_coat p_hat : ℝ} (d_coat d_hat coupon sales_tax : ℝ) :
  p_coat = 150 →
  p_hat = 70 →
  d_coat = 0.25 →
  d_hat = 0.10 →
  coupon = 10 →
  sales_tax = 0.10 →
  let coat_after_discount := p_coat * (1 - d_coat),
      hat_after_discount := p_hat * (1 - d_hat),
      combined_discounted_price := coat_after_discount + hat_after_discount,
      price_after_coupon := combined_discounted_price - coupon,
      total_price := price_after_coupon * (1 + sales_tax)
  in total_price = 182.05 :=
begin
  intros,
  sorry
end

end total_amount_paid_l597_597319


namespace center_of_circle_folds_l597_597779

theorem center_of_circle_folds (P : Type) [Inhabited P] [MetricSpace P] (C : Set P) :
  IsCircle C →
  (∀ p ∈ C, ∃ q ∈ C, p ≠ q → Folded p q ∧ Intersect p q = Center C) →
  ∃ n : ℕ, n = 2 := 
by
  sorry

end center_of_circle_folds_l597_597779


namespace player1_points_after_13_rotations_l597_597697

theorem player1_points_after_13_rotations :
  let sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
  let player_points (player : Nat) (rotations : Nat) :=
      rotations • (λ i, sector_points[(i + player) % 16])
  player_points 5 13 = 72 ∧ player_points 9 13 = 84 → player_points 1 13 = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597697


namespace find_total_votes_l597_597529

-- Definitions based on conditions
variable (score : ℕ)
variable (total_votes : ℕ)
variable (likes_ratio : ℝ)
variable (initial_score : ℕ)

-- Conditions from the problem
def conditions : Prop :=
  (initial_score = 0) ∧          -- Initial score is 0
  (score = 120) ∧                -- Given score is 120
  (likes_ratio = 0.75) ∧         -- 75% of votes are likes
  (score = (likes_ratio * total_votes - (1 - likes_ratio) * total_votes)) -- Score calculation expression

-- Statement of the proof problem
theorem find_total_votes (h : conditions) : total_votes = 240 :=
by
  sorry

end find_total_votes_l597_597529


namespace find_parabola_directrix_l597_597379

def parabola_directrix (a b c : ℝ) : ℝ :=
  let h := -b / (2 * a),
      k := c - (b^2 / (4 * a)),
      directrix := k - (1 / (4 * a))
  in directrix

theorem find_parabola_directrix :
  parabola_directrix (-3) 6 (-7) = -47 / 12 :=
by
  sorry

end find_parabola_directrix_l597_597379


namespace shaded_area_ratio_l597_597116

noncomputable def ratio_of_shaded_area_to_circle_area (AB r : ℝ) : ℝ :=
  let AC := r
  let CB := 2 * r
  let radius_semicircle_AB := 3 * r / 2
  let area_semicircle_AB := (1 / 2) * (Real.pi * (radius_semicircle_AB ^ 2))
  let radius_semicircle_AC := r / 2
  let area_semicircle_AC := (1 / 2) * (Real.pi * (radius_semicircle_AC ^ 2))
  let radius_semicircle_CB := r
  let area_semicircle_CB := (1 / 2) * (Real.pi * (radius_semicircle_CB ^ 2))
  let total_area_semicircles := area_semicircle_AB + area_semicircle_AC + area_semicircle_CB
  let non_overlapping_area_semicircle_AB := area_semicircle_AB - (area_semicircle_AC + area_semicircle_CB)
  let shaded_area := non_overlapping_area_semicircle_AB
  let area_circle_CD := Real.pi * (r ^ 2)
  shaded_area / area_circle_CD

theorem shaded_area_ratio (AB r : ℝ) : ratio_of_shaded_area_to_circle_area AB r = 1 / 4 :=
by
  sorry

end shaded_area_ratio_l597_597116


namespace max_value_of_n_possible_stack_on_a1_l597_597533

/-- Maximum value of n checkers such that any initial stack on h8 and order can be rearranged into any stack on a1 --/
theorem max_value_of_n_possible_stack_on_a1 (n : ℕ) :
  (∀ n, ∃ a b : ℕ, a + b = 14 ∧ (14.choose 7 = 3432)) → n ≤ 6 :=
sorry

end max_value_of_n_possible_stack_on_a1_l597_597533


namespace derivative_of_ln_2x_l597_597062

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x)

theorem derivative_of_ln_2x (x : ℝ) : deriv f x = 1 / x :=
  sorry

end derivative_of_ln_2x_l597_597062


namespace cos_angle_sum_l597_597957

theorem cos_angle_sum (θ : ℝ) (hcos : cos θ = -12 / 13) (hθ : θ ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) :
  cos (θ + Real.pi / 4) = -7 * Real.sqrt 2 / 26 :=
by
  sorry

end cos_angle_sum_l597_597957


namespace raisins_in_boxes_l597_597334

theorem raisins_in_boxes :
  ∃ x : ℕ, 72 + 74 + 3 * x = 437 ∧ x = 97 :=
by
  existsi 97
  split
  · rw [←add_assoc, add_comm 146, add_assoc]; exact rfl
  · exact rfl

end raisins_in_boxes_l597_597334


namespace AB_length_l597_597986

theorem AB_length (A B C M E D : Point) (AE EC BD AB : ℝ)
    (h1 : M = midpoint B C)
    (h2 : A = constructPoint)
    (h3 : reflectOverLine A M B = B')
    (h4 : reflectOverLine A M C = C')
    (h5 : sameLength A E (8 : ℝ))
    (h6 : sameLength E C (16 : ℝ))
    (h7 : sameLength B D (12 : ℝ))
    :
    sameLength A B (4 * √ 15) := sorry

end AB_length_l597_597986


namespace least_integer_square_eq_56_more_than_quadruple_value_l597_597747

theorem least_integer_square_eq_56_more_than_quadruple_value :
  ∃ x : ℤ, x^2 - 4 * x - 56 = 0 ∧ ∀ y : ℤ, y^2 - 4 * y - 56 = 0 → y ≥ x := 
begin
  existsi -7,
  split,
  {
    -- show -7^2 - 4 * (-7) - 56 = 0
    norm_num,
  },
  {
    -- show this is the minimum solution
    rintro y ⟨hy⟩,
    contrapose! hy,
    exact lt_of_lt_of_le hy (int.neg_of_sub_ge_zero ⟨Sub.int_to_int⟩),
  }
end

end least_integer_square_eq_56_more_than_quadruple_value_l597_597747


namespace intersection_sets_l597_597048

open Set

theorem intersection_sets :
  let A := {2, 3, 4}
  let B := {1, 2, 3}
  A ∩ B = {2, 3} :=
by
  sorry

end intersection_sets_l597_597048


namespace tank_capacity_equality_l597_597210

theorem tank_capacity_equality :
  let π := Real.pi in
  let C_A := 8 in
  let h_A := 10 in
  let C_B := 10 in
  let h_B := 8 in
  let occupied_fraction := 0.2 in
  let r_A := C_A / (2 * π) in
  let V_A := π * r_A^2 * h_A in
  let r_B := C_B / (2 * π) in
  let V_B := π * r_B^2 * h_B in
  let reduced_V_B := (1 - occupied_fraction) * V_B in
  V_A = reduced_V_B :=
by
  sorry

end tank_capacity_equality_l597_597210


namespace find_highest_score_l597_597284

theorem find_highest_score (average innings : ℕ) (avg_excl_two innings_excl_two H L : ℕ)
  (diff_high_low total_runs total_excl_two : ℕ)
  (h1 : diff_high_low = 150)
  (h2 : total_runs = average * innings)
  (h3 : total_excl_two = avg_excl_two * innings_excl_two)
  (h4 : total_runs - total_excl_two = H + L)
  (h5 : H - L = diff_high_low)
  (h6 : average = 62)
  (h7 : innings = 46)
  (h8 : avg_excl_two = 58)
  (h9 : innings_excl_two = 44)
  (h10 : total_runs = 2844)
  (h11 : total_excl_two = 2552) :
  H = 221 :=
by
  sorry

end find_highest_score_l597_597284


namespace parallelepiped_is_cube_l597_597639

theorem parallelepiped_is_cube
  {a b c : ℝ}
  (h1 : ∀ (a b c : ℝ), (a ^ 2 + b ^ 2 + c ^ 2 = 0) → a = b ∧ b = c ∧ a = c) :
  (∀ (a b c : ℝ), (a = b ∧ b = c ∧ a = c) → (parallelepiped a b c).is_cube := sorry

end parallelepiped_is_cube_l597_597639


namespace three_digit_numbers_count_l597_597360

theorem three_digit_numbers_count : 
  let digits := {0, 1, 2, 3}
  let valid_numbers := { x // ∃ a b c : ℕ, x = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits }
  valid_numbers.finite ∧ valid_numbers.to_finset.card = 48 := 
by 
  have H1 : (3 * 4 * 4) = 48 := by norm_num
  exact sorry

end three_digit_numbers_count_l597_597360


namespace smallest_solution_correct_l597_597384

noncomputable def smallest_solution (x : ℝ) : ℝ :=
if (⌊ x^2 ⌋ - ⌊ x ⌋^2 = 17) then x else 0

theorem smallest_solution_correct :
  smallest_solution (7 * Real.sqrt 2) = 7 * Real.sqrt 2 :=
by sorry

end smallest_solution_correct_l597_597384


namespace square_of_85_l597_597840

theorem square_of_85 : 85 ^ 2 = 7225 := by
  have h : (80 + 5) ^ 2 = 80 ^ 2 + 2 * 80 * 5 + 5 ^ 2 := by
    rw [add_sq]
    rfl
  calc
    85 ^ 2   = (80 + 5) ^ 2           := by rw [add_comm 80 5]
    ...      = 80 ^ 2 + 2 * 80 * 5 + 5 ^ 2 := by rw [h]
    ...      = 6400 + 800 + 25       := by norm_num
    ...      = 7225                  := by norm_num

end square_of_85_l597_597840


namespace quadratic_has_equal_roots_l597_597011

theorem quadratic_has_equal_roots :
  ∀ m : ℝ, (∀ x : ℝ, x^2 + 6 * x + m = 0 → x = -3) ↔ m = 9 := 
by
  intro m
  constructor
  {
    intro h
    have : (6:ℝ) ^ 2 - 4 * 1 * m = 0,
      from by simp [(pow_two 6), h.eq_c],
    simp [six_pow_two, neg_eq_zero] at this,
    linarith
  }
  {
    intro h
    simp [h],
    exact fun x _ => rfl
  }

end quadratic_has_equal_roots_l597_597011


namespace sam_wins_l597_597554

variable (p : ℚ) -- p is the probability that Sam wins
variable (phit : ℚ) -- probability of hitting the target in one shot
variable (pmiss : ℚ) -- probability of missing the target in one shot

-- Define the problem and set up the conditions
def conditions : Prop := phit = 2 / 5 ∧ pmiss = 3 / 5

-- Define the equation derived from the problem
def equation (p : ℚ) (phit : ℚ) (pmiss : ℚ) : Prop :=
  p = phit + (pmiss * pmiss * p)

-- State the theorem that Sam wins with probability 5/8
theorem sam_wins (h : conditions phit pmiss) : 
  equation p phit pmiss → p = 5 / 8 :=
by
  intros
  sorry

end sam_wins_l597_597554


namespace find_angle_RST_l597_597886

variable (PQ ST : Line)
variable (Q R S T P : Point)
variable (x : ℝ)
variable (angle_QRP angle_QRS : ℝ)

-- Conditions
def Parallel : Prop := PQ ∥ ST
def angle_QRP_def : Prop := angle Q R P = x
def angle_QRS_def : Prop := angle Q R S = 2 * x

theorem find_angle_RST
  (h1 : Parallel PQ ST)
  (h2 : angle_QRP_def angle Q R P x)
  (h3 : angle_QRS_def angle Q R S (2 * x))
  : angle R S T = 120 := 
sorry

end find_angle_RST_l597_597886


namespace repeating_block_length_7_div_13_l597_597579

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597579


namespace intersection_with_unit_circle_l597_597102

theorem intersection_with_unit_circle (α : ℝ) : 
    let x := Real.cos (α - Real.pi / 2)
    let y := Real.sin (α - Real.pi / 2)
    (x, y) = (Real.sin α, -Real.cos α) :=
by
  sorry

end intersection_with_unit_circle_l597_597102


namespace expression_undefined_expression_defined_otherwise_l597_597365

def rational_expression (x : ℝ) : ℝ :=
  (3 * x^3 - 14 * x + 8) / ((x + 1) * (x - 2))

theorem expression_undefined (x : ℝ) :
  x = -1 ∨ x = 2 → ¬ (∃ y, y = rational_expression x) :=
by
  intro hx
  cases hx with hx1 hx2
  { use (-1), rw [rational_expression], rw hx1, linarith }
  { use (2), rw [rational_expression], rw hx2, linarith }
  sorry

-- For other x values, the expression evaluates to a rational function.
theorem expression_defined_otherwise (x : ℝ) (h : x ≠ -1 ∧ x ≠ 2) :
  ∃ y, y = rational_expression x :=
by
  use (rational_expression x)
  exact ⟨rfl, h⟩
  sorry

end expression_undefined_expression_defined_otherwise_l597_597365


namespace valid_three_digit_numbers_l597_597943

theorem valid_three_digit_numbers : 
  let possible_digits_hundreds := {2, 3, 4, 6, 8},
      possible_digits_others := {0, 2, 3, 4, 6, 8} in
  (possible_digits_hundreds.card * possible_digits_others.card * possible_digits_others.card) = 180 := 
by
  sorry

end valid_three_digit_numbers_l597_597943


namespace ratio_of_areas_l597_597990

variables (P Q R S T : Type) [EuclideanGeometry P Q R S T]

-- Given conditions
def length_PQ : ℝ := 10
def length_RS : ℝ := 23

-- Definitions for the areas of triangle TPQ and trapezoid PQRS
def area_triangle_TPQ : ℝ := sorry -- needs area computation logic
def area_trapezoid_PQRS : ℝ := sorry -- needs area computation logic

-- Hypothesis that needs to be proved
theorem ratio_of_areas (hPQ : length_PQ = 10) (hRS : length_RS = 23) :
  area_triangle_TPQ / area_trapezoid_PQRS = 100 / 429 :=
sorry

end ratio_of_areas_l597_597990


namespace distance_from_point_to_plane_l597_597378

/-- 
  Given four points in 3-dimensional space: 
  M1 (1, -1, 2), 
  M2 (2, 1, 2), 
  M3 (1, 1, 4), 
  M0 (-3, 2, 7), 
  prove that the distance from the point M0 to the plane passing through M1, M2, and M3 is √6.
-/
theorem distance_from_point_to_plane :
  let M0 := (λ (i : Fin 3), ![-3, 2, 7] (i))
  let M1 := (λ (i : Fin 3), ![1, -1, 2] (i))
  let M2 := (λ (i : Fin 3), ![2, 1, 2] (i))
  let M3 := (λ (i : Fin 3), ![1, 1, 4] (i))
  let A := 2
  let B := -1
  let C := 1
  let D := -5
  ∃ (x y z : ℝ), 
    M1 = (x, y, z) ∧ 
    2 * x - y + z - 5 = 0 ∧
    M2 = (x, y, z) ∧ 
    2 * x - y + z - 5 = 0 ∧ 
    M3 = (x, y, z) ∧ 
    2 * x - y + z - 5 = 0 ∧ 
    (√((A^2) + (B^2) + (C^2))) = √6 :=
by
  sorry

end distance_from_point_to_plane_l597_597378


namespace number_of_rectangles_in_5x5_grid_l597_597725

theorem number_of_rectangles_in_5x5_grid : 
  (∑ i in Finset.range 6, i^3) = 225 := 
by 
  sorry

end number_of_rectangles_in_5x5_grid_l597_597725


namespace gumballs_each_shared_equally_l597_597143

def initial_gumballs_joanna : ℕ := 40
def initial_gumballs_jacques : ℕ := 60
def multiplier : ℕ := 4

def purchased_gumballs (initial : ℕ) (multiplier : ℕ) : ℕ :=
  initial * multiplier

def total_gumballs (initial : ℕ) (purchased : ℕ) : ℕ :=
  initial + purchased

def total_combined_gumballs (total1 : ℕ) (total2 : ℕ) : ℕ :=
  total1 + total2

def shared_equally (total : ℕ) : ℕ :=
  total / 2

theorem gumballs_each_shared_equally :
  let joanna_initial := initial_gumballs_joanna,
      jacques_initial := initial_gumballs_jacques,
      joanna_purchased := purchased_gumballs joanna_initial multiplier,
      jacques_purchased := purchased_gumballs jacques_initial multiplier,
      joanna_total := total_gumballs joanna_initial joanna_purchased,
      jacques_total := total_gumballs jacques_initial jacques_purchased,
      combined_total := total_combined_gumballs joanna_total jacques_total in
  shared_equally combined_total = 250 :=
by
  sorry

end gumballs_each_shared_equally_l597_597143


namespace common_ratio_l597_597426

theorem common_ratio (a1 a2 a3 : ℚ) (S3 q : ℚ)
  (h1 : a3 = 3 / 2)
  (h2 : S3 = 9 / 2)
  (h3 : a1 + a2 + a3 = S3)
  (h4 : a1 = a3 / q^2)
  (h5 : a2 = a3 / q):
  q = 1 ∨ q = -1/2 :=
by sorry

end common_ratio_l597_597426


namespace range_of_a_l597_597964

open Real

theorem range_of_a (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∀ x : ℝ, 0 < x ∧ x < 1 / 2 → 9^x < log a x) : 2^(-1/3) ≤ a ∧ a < 1 := 
sorry

end range_of_a_l597_597964


namespace sin_780_eq_sqrt3_div_2_l597_597346

theorem sin_780_eq_sqrt3_div_2 :
  Real.sin (780 * Real.pi / 180) = (Real.sqrt 3) / 2 :=
by
  sorry

end sin_780_eq_sqrt3_div_2_l597_597346


namespace fill_in_the_blank_with_flowchart_l597_597238

def methods_to_describe_algorithm := ["Natural language", "Flowchart", "Pseudocode"]

theorem fill_in_the_blank_with_flowchart : 
  methods_to_describe_algorithm[1] = "Flowchart" :=
sorry

end fill_in_the_blank_with_flowchart_l597_597238


namespace new_mean_of_remaining_numbers_l597_597039

theorem new_mean_of_remaining_numbers 
  (numbers : Fin 60 → ℝ) 
  (h_mean : (∑ i, numbers i) / 60 = 47)
  (h_removed : numbers 0 = 50 ∧ numbers 1 = 60) :
  ((∑ i in Finset.erase (Finset.erase Finset.univ 0) 1, numbers i) / 58) = 47 :=
by
  sorry  -- This is a placeholder for the proof.

end new_mean_of_remaining_numbers_l597_597039


namespace larger_number_is_437_l597_597285

-- Definitions from the conditions
def hcf : ℕ := 23
def factor1 : ℕ := 13
def factor2 : ℕ := 19

-- The larger number should be the product of H.C.F and the larger factor.
theorem larger_number_is_437 : hcf * factor2 = 437 := by
  sorry

end larger_number_is_437_l597_597285


namespace distance_apart_after_two_hours_l597_597807

theorem distance_apart_after_two_hours :
  (Jay_walk_rate : ℝ) = 1 / 20 →
  (Paul_jog_rate : ℝ) = 3 / 40 →
  (time_duration : ℝ) = 2 * 60 →
  (distance_apart : ℝ) = 15 :=
by
  sorry

end distance_apart_after_two_hours_l597_597807


namespace points_player_1_after_13_rotations_l597_597683

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end points_player_1_after_13_rotations_l597_597683


namespace reflection_center_is_correct_l597_597220

-- Definition of what it means to reflect a point over the line y = -x
def reflect_over_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in
  (-y, -x)

-- The given original center is (9, -4)
def original_center : ℝ × ℝ := (9, -4)

-- The expected new center after reflection should be (4, -9)
def new_center : ℝ × ℝ := (4, -9)

-- Statement that reflects original_center over the line y = -x
-- should yield new_center
theorem reflection_center_is_correct :
  reflect_over_y_eq_neg_x original_center = new_center :=
sorry

end reflection_center_is_correct_l597_597220


namespace rational_reciprocal_sum_of_cube_roots_l597_597520

theorem rational_reciprocal_sum_of_cube_roots
  (p q r : ℚ)
  (h : ∃ a b c : ℚ, 
        a = real.cbrt (p^2 * q) ∧ 
        b = real.cbrt (q^2 * r) ∧ 
        c = real.cbrt (r^2 * p) ∧ 
        a + b + c ∈ ℚ) :
  ∃ x : ℚ, 
  x = 1 / real.cbrt (p^2 * q) + 1 / real.cbrt (q^2 * r) + 1 / real.cbrt (r^2 * p) :=
sorry

end rational_reciprocal_sum_of_cube_roots_l597_597520


namespace player1_points_after_13_rotations_l597_597700

theorem player1_points_after_13_rotations :
  ∃ (player1_points : ℕ), 
    (∀ (i : ℕ),  (i = 5 → player1_points = 72) ∧ (i = 9 → player1_points = 84)) → 
    player1_points = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597700


namespace proof_f_g_l597_597053

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 1
def g (x : ℝ) : ℝ := 2*x + 3

theorem proof_f_g (x : ℝ) : f (g 2) - g (f 2) = 258 :=
by
  sorry

end proof_f_g_l597_597053


namespace endpoints_of_diameters_form_trapezoid_l597_597078

theorem endpoints_of_diameters_form_trapezoid
  (C₁ C₂ : Circle)
  (h_tangent : ExternallyTangent C₁ C₂)
  (d₁ : Diameter C₁)
  (d₂ : Diameter C₂)
  (h_tangent_d₁ : Tangent (line_through d₁) C₂)
  (h_tangent_d₂ : Tangent (line_through d₂) C₁) :
  Trapezoid (endpoints d₁) (endpoints d₂) :=
by
  sorry

end endpoints_of_diameters_form_trapezoid_l597_597078


namespace product_of_fractions_l597_597831

theorem product_of_fractions :
  ∏ k in finset.range 501, (4 + 4 * k : ℕ) / (8 + 4 * k) = 1 / 502 := 
by sorry

end product_of_fractions_l597_597831


namespace cars_equilibrium_l597_597665

variable (days : ℕ) -- number of days after which we need the condition to hold
variable (carsA_init carsB_init carsA_to_B carsB_to_A : ℕ) -- initial conditions and parameters

theorem cars_equilibrium :
  let cars_total := 192 + 48
  let carsA := carsA_init + (carsB_to_A - carsA_to_B) * days
  let carsB := carsB_init + (carsA_to_B - carsB_to_A) * days
  carsA_init = 192 -> carsB_init = 48 ->
  carsA_to_B = 21 -> carsB_to_A = 24 ->
  cars_total = 192 + 48 ->
  days = 6 ->
  cars_total = carsA + carsB -> carsA = 7 * carsB :=
by
  intros
  sorry

end cars_equilibrium_l597_597665


namespace negation_of_p_l597_597073

theorem negation_of_p :
  (¬ (∀ x : ℝ, x^3 + 2 < 0)) = ∃ x : ℝ, x^3 + 2 ≥ 0 := 
  by sorry

end negation_of_p_l597_597073


namespace infinitely_many_divisible_by_1986_l597_597521

theorem infinitely_many_divisible_by_1986 :
  ∃ᶠ n in at_top, 1986 ∣ (nat.rec_on n 39 (λ n u, (nat.rec_on n 45 (λ n v, v^2 - u)))) :=
sorry

end infinitely_many_divisible_by_1986_l597_597521


namespace john_trip_total_time_l597_597499

theorem john_trip_total_time :
  let t1 := 2
  let t2 := 3 * t1
  let t3 := 4 * t2
  let t4 := 5 * t3
  let t5 := 6 * t4
  t1 + t2 + t3 + t4 + t5 = 872 :=
by
  let t1 := 2
  let t2 := 3 * t1
  let t3 := 4 * t2
  let t4 := 5 * t3
  let t5 := 6 * t4
  have h1: t1 + t2 + t3 + t4 + t5 = 2 + (3 * 2) + (4 * (3 * 2)) + (5 * (4 * (3 * 2))) + (6 * (5 * (4 * (3 * 2)))) := by
    sorry
  have h2: 2 + 6 + 24 + 120 + 720 = 872 := by
    sorry
  exact h2

end john_trip_total_time_l597_597499


namespace min_value_expr_l597_597888

variable (a b : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = 1)

theorem min_value_expr : (1 / (2 * a)) + (1 / (2 * b)) + (8 / (a + b)) ≥ 4 :=
by
  sorry

end min_value_expr_l597_597888


namespace areas_of_circles_are_equal_l597_597344

-- Define the circumferences and radii conditions
def circle_x_circumference : ℝ := 10 * Real.pi
def half_radius_y : ℝ := 2.5

-- Define the theorem to prove the relationship between the areas of circles x and y
theorem areas_of_circles_are_equal (C_x : ℝ) (C_x_eq : C_x = circle_x_circumference)
    (half_r_y : ℝ) (half_r_y_eq : half_r_y = half_radius_y) :
    let r_x := C_x / (2 * Real.pi)
    let r_y := 2 * half_r_y
    let A_x := Real.pi * r_x^2
    let A_y := Real.pi * r_y^2
    A_x = A_y :=
by
  sorry

end areas_of_circles_are_equal_l597_597344


namespace quadratic_a_plus_b2_l597_597003

theorem quadratic_a_plus_b2 (a b : ℂ) (h : ∀ x, 5 * x^2 + 4 * x + 20 = 0 ↔ (x = a + b * complex.I ∨ x = a - b * complex.I)) : 
  a + b^2 = 86 / 25 :=
sorry

end quadratic_a_plus_b2_l597_597003


namespace factorial_inequality_l597_597172

theorem factorial_inequality (a b n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < n) (h4 : n.factorial = a.factorial * b.factorial) :
  a + b < n + 2 * Real.log n / Real.log 2 + 4 :=
by
  sorry

end factorial_inequality_l597_597172


namespace count_rectangles_in_5x5_grid_l597_597732

theorem count_rectangles_in_5x5_grid :
  let n := 5 in
  ∑ x in finset.range n.succ, (n - x) * (n - x) * (n - x) = 225 :=
by {
  simp only [finset.range, finset.sum],
  sorry
}

end count_rectangles_in_5x5_grid_l597_597732


namespace polynomials_with_real_roots_characterization_l597_597863

noncomputable def polynomials_with_real_roots : List (Polynomial ℝ) :=
  [1, -1, X + 1, X - 1, X^2 + X - 1, X^2 - X - 1, X^3 + X^2 - X - 1, X^3 - X^2 - X + 1]

theorem polynomials_with_real_roots_characterization :
  ∀ P : Polynomial ℝ,
    (∀ i : ℕ, i < P.natDegree → (P.coeff i = 1 ∨ P.coeff i = -1)) →
    (∀ r : ℝ, Polynomial.aeval r P = 0 → ∃ r_i : List ℝ, (Polynomial.eval r_i P = 0 ∧
    (r_i.all (λ x, P.eval x = 0)))) →
    P ∈ polynomials_with_real_roots :=
by
  sorry

end polynomials_with_real_roots_characterization_l597_597863


namespace divisors_larger_than_9_factorial_l597_597087

theorem divisors_larger_than_9_factorial (n : ℕ) :
  (∃ k : ℕ, k = 9 ∧ (number_of_divisors_of_10_factorial_greater_than_9_factorial = k)) :=
begin
  sorry
end

def number_of_divisors_of_10_factorial_greater_than_9_factorial : ℕ :=
  (10.fact.divisors.filter (λ d, d > 9.fact)).length

end divisors_larger_than_9_factorial_l597_597087


namespace infinite_alternating_parity_l597_597536

theorem infinite_alternating_parity (m : ℕ) : ∃ᶠ n in at_top, 
  ∀ i < m, ((5^n / 10^i) % 2) ≠ (((5^n / 10^(i+1)) % 10) % 2) :=
sorry

end infinite_alternating_parity_l597_597536


namespace exists_equal_values_at_0_1_apart_l597_597518

theorem exists_equal_values_at_0_1_apart (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h0 : f 0 = 0) (h1 : f 1 = 0) :
  ∃ x y ∈ Set.Icc 0 1, |x - y| = 0.1 ∧ f x = f y := by
sorry

end exists_equal_values_at_0_1_apart_l597_597518


namespace exists_acute_triangle_l597_597280

-- Given five segment lengths and it being possible to form a triangle with any three of them
variables {a1 a2 a3 a4 a5 : ℝ}

-- The condition that any three of these segments can form a triangle
def triangle_inequality (x y z : ℝ) : Prop := (x + y > z) ∧ (x + z > y) ∧ (y + z > x)

axiom segments_can_form_triangle :
  triangle_inequality a1 a2 a3 ∧
  triangle_inequality a1 a2 a4 ∧
  triangle_inequality a1 a2 a5 ∧
  triangle_inequality a1 a3 a4 ∧
  triangle_inequality a1 a3 a5 ∧
  triangle_inequality a1 a4 a5 ∧
  triangle_inequality a2 a3 a4 ∧
  triangle_inequality a2 a3 a5 ∧
  triangle_inequality a2 a4 a5 ∧
  triangle_inequality a3 a4 a5

-- Define what it means to have all angles acute in a triangle
def triangle_all_acute (x y z : ℝ) : Prop :=
  (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2)

-- The theorem stating that at least one of the triangles has all angles acute
theorem exists_acute_triangle :
  ∃ (i j k : ℝ), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ triangle_inequality i j k ∧ triangle_all_acute i j k :=
by
  sorry

end exists_acute_triangle_l597_597280


namespace curve_parametrization_l597_597301

-- Define the parametrization of the curve
def parametrization (t : ℝ) : ℝ × ℝ :=
  (3 * Real.cos t + 2 * Real.sin t, 5 * Real.sin t)

-- Define the form to be expressed
def general_form (A B C x y : ℝ) : ℝ :=
  A * x^2 + B * x * y + C * y^2

theorem curve_parametrization :
  ∃ (A B C : ℝ),
    (∀ t : ℝ,
      let (x, y) := parametrization t
      in general_form A B C x y = 1) ∧ 
    A = (1 / 9) ∧
    B = - (4 / 45) ∧
    C = (16 / 225) := 
  sorry

end curve_parametrization_l597_597301


namespace people_sharing_bill_l597_597661

theorem people_sharing_bill (total_bill : ℝ) (tip_percent : ℝ) (share_per_person : ℝ) (n : ℝ) :
  total_bill = 211.00 →
  tip_percent = 0.15 →
  share_per_person = 26.96 →
  abs (n - 9) < 1 :=
by
  intros h1 h2 h3
  sorry

end people_sharing_bill_l597_597661


namespace average_price_per_book_l597_597540

theorem average_price_per_book (books1 books2 totalBooks : ℕ) (price1 price2 totalPrice : ℝ) : 
  books1 = 55 → books2 = 60 → totalBooks = books1 + books2 → price1 = 1500 → price2 = 340 → totalPrice = price1 + price2 → 
  totalPrice / totalBooks = 16 :=
  by
    intros h1 h2 h3 h4 h5 h6
    rw [h1, h2, h4, h5, h3, h6]
    exact sorry

end average_price_per_book_l597_597540


namespace sqrt_meaningful_iff_range_l597_597949

variable (x : ℝ)

theorem sqrt_meaningful_iff_range :
  (∃ (v : ℝ), v = sqrt (x - 1)) ↔ x ≥ 1 := sorry

end sqrt_meaningful_iff_range_l597_597949


namespace second_graders_cost_correct_l597_597213

noncomputable def number_of_second_graders_wearing_blue_shirts
  (kindergarteners : ℕ) (cost_per_orange_shirt : ℝ)
  (first_graders : ℕ) (cost_per_yellow_shirt : ℝ)
  (third_graders : ℕ) (cost_per_green_shirt : ℝ)
  (second_graders_total_shirts_cost : ℝ)
  (total_amount_spent : ℝ) : ℕ :=
  (total_amount_spent - ((kindergarteners : ℝ) * cost_per_orange_shirt +
    (first_graders : ℝ) * cost_per_yellow_shirt +
    (third_graders : ℝ) * cost_per_green_shirt))
  / second_graders_total_shirts_cost

theorem second_graders_cost_correct :
  number_of_second_graders_wearing_blue_shirts 101 5.80 113 5.00 108 5.25 5.60 2317 = 107 :=
by
  sorry

end second_graders_cost_correct_l597_597213


namespace M_leq_N_l597_597895

theorem M_leq_N (n : ℕ) (a b : Fin n.succ → ℝ) (λ : ℝ)
  (ha : ∀ i j, i ≤ j → a i ≥ a j)
  (hb : ∀ i j, i ≤ j → b i ≥ b j)
  (hλ : 0 ≤ λ ∧ λ ≤ 2) :
  let M := ∑ i in Finset.range n,
    Real.sqrt (a (Fin.succ i) ^ 2 + b (Fin.succ i) ^ 2 - λ * a (Fin.succ i) * b (Fin.succ i)),
      N := ∑ i in Finset.range n,
    Real.sqrt (a (Fin.succ i) ^ 2 + b (Fin.succ i) ^ 2 - λ * a (Fin.succ (i + 1)) * b (Fin.succ i))
  in M ≤ N := sorry

end M_leq_N_l597_597895


namespace player_1_points_after_13_rotations_l597_597686

theorem player_1_points_after_13_rotations :
  ∀ (table : ℕ → ℕ) (n : ℕ) (points : List ℕ),
    (∀ i, table i+16 = table i) →
    (table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]) →
    (points.length = 16) →
    (points.get 5 = 72) →
    (points.get 9 = 84) →
    (n = 13) →
    ((points.sum i₁, 0, 13) table ((stats : List ℕ) (i : fin 16) =>
      List.sum (List.take stats.toList) i.val + 
      List.sum (List.drop stats.toList i.val i.val + 2 * n) table) = points.sum table) →
    points.get 1 = 20 :=
by
  intros
  sorry

end player_1_points_after_13_rotations_l597_597686


namespace black_area_after_transformations_l597_597793

theorem black_area_after_transformations :
  let initial_fraction : ℝ := 1
  let transformation_factor : ℝ := 3 / 4
  let number_of_transformations : ℕ := 5
  let final_fraction : ℝ := transformation_factor ^ number_of_transformations
  final_fraction = 243 / 1024 :=
by
  -- Proof omitted
  sorry

end black_area_after_transformations_l597_597793


namespace repeating_block_length_7_div_13_l597_597617

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597617


namespace distance_and_avg_velocity_l597_597249

noncomputable def velocity (t : ℝ) : ℝ := (1 / 2) * t^2

theorem distance_and_avg_velocity :
  let S := ∫ t in (0 : ℝ)..(12 : ℝ), velocity t
  ∧ let t := 12
  ∧ let v_cp := S / t in
  S = 288 ∧ v_cp = 24 :=
by
  sorry

end distance_and_avg_velocity_l597_597249


namespace repeating_decimal_block_length_l597_597585

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l597_597585


namespace fib_rect_decomp_l597_597009

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

-- Define the Fibonacci rectangle
structure fibonacci_rectangle (n : ℕ) :=
(length : ℕ := fib (n + 1)) -- Length of the rectangle (F_{n+1})
(width : ℕ := fib n) -- Width of the rectangle (F_n)

-- Proposition stating the decomposition property 
theorem fib_rect_decomp (n : ℕ) : 
    ∃ squares : list (ℕ × ℕ), 
    (∀ s ∈ squares, s.1 = s.2) ∧ -- each element in the list is a square
    (length squares = n) ∧ -- there are exactly n squares
    (∀ x ∈ squares, list.count squares x <= 2) ∧ -- no more than 2 identical squares
    (sum (list.map (λ s, s.1 * s.2) squares) = fib (n+1) * fib n) := sorry

end fib_rect_decomp_l597_597009


namespace Sam_wins_probability_l597_597549

-- Define the basic probabilities
def prob_hit : ℚ := 2 / 5
def prob_miss : ℚ := 3 / 5

-- Define the desired probability that Sam wins
noncomputable def p : ℚ := 5 / 8

-- The mathematical problem statement in Lean
theorem Sam_wins_probability :
  p = prob_hit + (prob_miss * prob_miss * p) := 
sorry

end Sam_wins_probability_l597_597549


namespace infinite_m_lt_m1_infinite_m_gt_m1_l597_597006

open Nat

def f (n : ℕ) : ℚ :=
  (1 : ℚ) / n * ∑ k in Finset.range n, ⌊(n : ℚ) / (k + 1)⌋

theorem infinite_m_lt_m1 (m : ℕ) : ∃ᶠ n in atTop, f(n) < f(n + 1) := sorry

theorem infinite_m_gt_m1 (m : ℕ) : ∃ᶠ n in atTop, f(n) > f(n + 1) := sorry

end infinite_m_lt_m1_infinite_m_gt_m1_l597_597006


namespace inscribed_triangle_center_on_square_side_l597_597797

-- Define the semicircle with center O and diameter 2R
variables (O : Point) (R : ℝ)
def semicircle : Set Point := { P | P.distance O ≤ R ∧ P.y ≥ 0 }

-- Define the inscribed square with side length a, where diagonal is 2R
variables (a : ℝ) (square : Set Point)
def inscribed_square : Prop := ∀ P ∈ square, semicircle O R P ∧ a * sqrt 2 = 2 * R 

-- Define the inscribed right triangle with base 2R and height h, area equals the square's area
variables (triangle : Set Point) (h : ℝ)
def inscribed_right_triangle : Prop := 
  ∀ P ∈ triangle, semicircle O R P ∧ P.y = 0 ∧ 
  0.5 * 2 * R * h = a^2 

-- Define the circle inscribed in the triangle
variables (I : Point) (r : ℝ)
def inscribed_circle : Prop := 
  (∀ P ∈ triangle, I.distance P = r) ∧ I.distance O < R

-- The proof goal
theorem inscribed_triangle_center_on_square_side :
  inscribed_square O R a square ∧ inscribed_right_triangle O R a triangle h ∧ inscribed_circle O R I r triangle →
  ∃ P ∈ square, I = P :=
sorry

end inscribed_triangle_center_on_square_side_l597_597797


namespace David_fewer_crunches_l597_597363

-- Definitions as per conditions.
def Zachary_crunches := 62
def David_crunches := 45

-- Proof statement for how many fewer crunches David did compared to Zachary.
theorem David_fewer_crunches : Zachary_crunches - David_crunches = 17 := by
  -- Proof details would go here, but we skip them with 'sorry'.
  sorry

end David_fewer_crunches_l597_597363


namespace math_problem_l597_597887

theorem math_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a + 2) * (b + 2) = 18) :
  (∀ x, (x = 3 / (a + 2) + 3 / (b + 2)) → x ≥ Real.sqrt 2) ∧
  ¬(∃ y, (y = a * b) ∧ y ≤ 11 - 6 * Real.sqrt 2) ∧
  (∀ z, (z = 2 * a + b) → z ≥ 6) ∧
  (∀ w, (w = (a + 1) * b) → w ≤ 8) :=
sorry

end math_problem_l597_597887


namespace find_c_l597_597876

theorem find_c (c : ℝ) : c = 3 → ∃ x : ℝ, (3 * x + 5 = 2) ∧ (c * x + 4 = 1) :=
by
  intro hc
  rw [hc]
  use -1
  split
  { 
    -- 3 * (-1) + 5 = 2
    linarith 
  }
  { 
    -- 3 * (-1) + 4 = 1
    linarith 
  }

end find_c_l597_597876


namespace sarah_initial_followers_l597_597209

noncomputable def initial_followers_sarah (followers_3_weeks_later : ℕ) (gained_week1 : ℕ) 
  (gained_week2 : ℕ) (gained_week3 : ℕ) : ℕ :=
  followers_3_weeks_later - (gained_week1 + gained_week2 + gained_week3)

theorem sarah_initial_followers : 
  let gained_week1 := 90 in
  let gained_week2 := 30 in
  let gained_week3 := 10 in
  let followers_3_weeks_later := 180 in
  initial_followers_sarah followers_3_weeks_later gained_week1 gained_week2 gained_week3 = 50 :=
by
  unfold initial_followers_sarah
  rfl

end sarah_initial_followers_l597_597209


namespace sequence_a_correct_T_n_correct_l597_597913

def sequence_a (n : ℕ) : ℕ := 3 * n - 1

def b_n (n : ℕ) : ℤ := 20 - 3 * n

noncomputable def T_n (n : ℕ) : ℚ :=
if n <= 6 then -3 / 2 * n^2 + 37 / 2 * n
else 3 / 2 * n^2 - 37 / 2 * n + 114

theorem sequence_a_correct (n : ℕ) (h_pos : 0 < n) 
  (h_sn_gt_1 : ∀ n : ℕ, 0 < n -> S_n n > 1)
  (h_relation : ∀ n : ℕ, 0 < n -> 6 * S_n n = (a_n n + 1) * (a_n n + 2)) :
  sequence_a n = 3 * n - 1 := sorry

theorem T_n_correct (n : ℕ) :
  T_n n = (if n <= 6 then -3 / 2 * n^2 + 37 / 2 * n
           else 3 / 2 * n^2 - 37 / 2 * n + 114) := sorry

end sequence_a_correct_T_n_correct_l597_597913


namespace geometric_sequence_common_ratio_l597_597423

noncomputable def common_ratio_q : ℚ :=
  let a : ℕ → ℚ := λ n, a₁ * q^n in
  have h1 : a₁ > 0, from sorry,
  have h2 : (∀ n : ℕ, a n < a (n + 1)), from sorry, -- Increasing sequence condition
  have h3 : 2 * (a 4 + a 6) = 5 * a 5, from sorry, -- Given condition 2(a_4 + a_6) = 5a_5
  let q := RootsOfQuadratic 2 (-5) 2 in
  classical.some (classical.some_spec (exists_unique_of_leh1 q h1 h2 h3))

theorem geometric_sequence_common_ratio (a₁ : ℚ) (h1 : a₁ > 0)
(h2 : ∀ n : ℕ, (a₁ * q^n) < (a₁ * q^(n+1)))
(h3 : 2 * ((a₁ * q^3) + (a₁ * q^5)) = 5 * (a₁ * q^4)) :
  common_ratio_q = 2 :=
sorry

end geometric_sequence_common_ratio_l597_597423


namespace Sam_wins_probability_l597_597546

-- Define the basic probabilities
def prob_hit : ℚ := 2 / 5
def prob_miss : ℚ := 3 / 5

-- Define the desired probability that Sam wins
noncomputable def p : ℚ := 5 / 8

-- The mathematical problem statement in Lean
theorem Sam_wins_probability :
  p = prob_hit + (prob_miss * prob_miss * p) := 
sorry

end Sam_wins_probability_l597_597546


namespace sum_of_lattice_points_l597_597982

def lattice_points_on_segment (x1 y1 x2 y2 : ℤ) : ℕ :=
  have : x2 - x1 = y2 - y1 + 3 := by sorry
  sorry

def f (n : ℕ) : ℕ := lattice_points_on_segment 0 0 n (n + 3)

theorem sum_of_lattice_points :
  ∑ n in Finset.range 100, f (n + 1) = 66 :=
by sorry

end sum_of_lattice_points_l597_597982


namespace no_cubes_between_squares_l597_597837

theorem no_cubes_between_squares :
  ∀ (n a b : ℕ), 0 < n → 0 < a → 0 < b → n^2 < a^3 → a^3 < b^3 → b^3 < (n + 1)^2 → false :=
begin
  intros n a b hn ha hb h1 h2 h3,
  sorry,
end

end no_cubes_between_squares_l597_597837


namespace eighth_term_is_84_l597_597074

-- Definition of the nth term in the sequence
def nth_term (n : ℕ) : ℕ := (3 * n * (n - 1)) / 2

-- Proof that the 8th term in the sequence is 84
theorem eighth_term_is_84 : nth_term 8 = 84 := by
  -- Definitions as given in the conditions
  have term_8_definition : nth_term 8 = (3 * 8 * (8 - 1)) / 2 := rfl
  -- Calculate it
  calc nth_term 8 
      = (3 * 8 * (8 - 1)) / 2 : by rw [term_8_definition]
  ... = 84 : by norm_num

end eighth_term_is_84_l597_597074


namespace sqrt_meaningful_iff_range_l597_597948

variable (x : ℝ)

theorem sqrt_meaningful_iff_range :
  (∃ (v : ℝ), v = sqrt (x - 1)) ↔ x ≥ 1 := sorry

end sqrt_meaningful_iff_range_l597_597948


namespace resistance_of_one_rod_l597_597258

section RodResistance

variables (R_0 R : ℝ)

-- Given: the resistance of the entire construction is 8 Ω
def entire_construction_resistance : Prop := R = 8

-- Given: formula for the equivalent resistance
def equivalent_resistance_formula : Prop := R = 4 / 10 * R_0

-- To prove: the resistance of one rod is 20 Ω
theorem resistance_of_one_rod 
  (h1 : entire_construction_resistance R)
  (h2 : equivalent_resistance_formula R_0 R) :
  R_0 = 20 :=
sorry

end RodResistance

end resistance_of_one_rod_l597_597258


namespace range_of_a_for_exactly_two_roots_l597_597064

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 / 4 * x + 1)
  else Real.log x

theorem range_of_a_for_exactly_two_roots :
  ∀ a : ℝ, (∃! x₁ x₂ : ℝ, f x₁ = a * x₁ ∧ f x₂ = a * x₂ ∧ x₁ ≠ x₂) ↔ a ∈ Icc (1 / 4) (1 / Real.exp 1) :=
by
  sorry

end range_of_a_for_exactly_two_roots_l597_597064


namespace repeating_block_length_7_div_13_l597_597596

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l597_597596


namespace Lennon_total_reimbursement_l597_597148

def mileage_reimbursement (industrial_weekday: ℕ → ℕ) (commercial_weekday: ℕ → ℕ) (weekend: ℕ → ℕ) : ℕ :=
  let industrial_rate : ℕ := 36
  let commercial_weekday_rate : ℕ := 42
  let weekend_rate : ℕ := 45
  (industrial_weekday 1 * industrial_rate + commercial_weekday 1 * commercial_weekday_rate)    -- Monday
  + (industrial_weekday 2 * industrial_rate + commercial_weekday 2 * commercial_weekday_rate + commercial_weekday 3 * commercial_weekday_rate)  -- Tuesday
  + (industrial_weekday 3 * industrial_rate + commercial_weekday 3 * commercial_weekday_rate)    -- Wednesday
  + (commercial_weekday 4 * commercial_weekday_rate + commercial_weekday 5 * commercial_weekday_rate)  -- Thursday
  + (industrial_weekday 5 * industrial_rate + commercial_weekday 6 * commercial_weekday_rate + industrial_weekday 6 * industrial_rate)    -- Friday
  + (weekend 1 * weekend_rate)                                       -- Saturday

def monday_industrial_miles : ℕ := 10
def monday_commercial_miles : ℕ := 8

def tuesday_industrial_miles : ℕ := 12
def tuesday_commercial_miles_1 : ℕ := 9
def tuesday_commercial_miles_2 : ℕ := 5

def wednesday_industrial_miles : ℕ := 15
def wednesday_commercial_miles : ℕ := 5

def thursday_commercial_miles_1 : ℕ := 10
def thursday_commercial_miles_2 : ℕ := 10

def friday_industrial_miles_1 : ℕ := 5
def friday_commercial_miles : ℕ := 8
def friday_industrial_miles_2 : ℕ := 3

def saturday_commercial_miles : ℕ := 12

def reimbursement_total :=
  mileage_reimbursement
    (fun day => if day = 1 then monday_industrial_miles else if day = 2 then tuesday_industrial_miles else if day = 3 then wednesday_industrial_miles else if day = 5 then friday_industrial_miles_1 + friday_industrial_miles_2 else 0)
    (fun day => if day = 1 then monday_commercial_miles else if day = 2 then tuesday_commercial_miles_1 + tuesday_commercial_miles_2 else if day = 3 then wednesday_commercial_miles else if day = 4 then thursday_commercial_miles_1 + thursday_commercial_miles_2 else if day = 6 then friday_commercial_miles else 0)
    (fun day => if day = 1 then saturday_commercial_miles else 0)

theorem Lennon_total_reimbursement : reimbursement_total = 4470 := 
by sorry

end Lennon_total_reimbursement_l597_597148


namespace parabola_ellipse_focus_l597_597466

-- Define the equations for the parabola and the ellipse
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def ellipse (x y : ℝ) := x^2 / 5 + y^2 = 1

-- Define the condition that the focus of the parabola coincides with the right focus of the ellipse
def parabola_focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)
def ellipse_right_focus : (ℝ × ℝ) := (2, 0)

-- The theorem: If the focus of the parabola coincides with the right focus of the ellipse, then p = 4
theorem parabola_ellipse_focus (p : ℝ) 
  (h_parabola : ∀ x y, parabola p x y)
  (h_ellipse : ∀ x y, ellipse x y)
  (h_focus : parabola_focus p = ellipse_right_focus) : p = 4 :=
by {
  sorry
}

end parabola_ellipse_focus_l597_597466


namespace target_percentage_of_water_l597_597313

theorem target_percentage_of_water (initial_volume : ℝ) (initial_percentage : ℝ) (added_water : ℝ) (target_percentage : ℝ) :
  initial_volume = 125 →
  initial_percentage = 20 →
  added_water = 8.333333333333334 →
  target_percentage = 25 →
  let initial_water := initial_volume * (initial_percentage / 100) in
  let new_water := initial_water + added_water in
  let new_volume := initial_volume + added_water in
  (new_water / new_volume) * 100 = target_percentage :=
by
  intros h_initial_vol h_initial_perc h_added_water h_target_perc
  simp [h_initial_vol, h_initial_perc, h_added_water, h_target_perc]
  let initial_water := 125 * (20 / 100)
  let new_water := initial_water + 8.333333333333334
  let new_volume := 125 + 8.333333333333334
  have h_initial_water : initial_water = 25 := rfl
  have h_new_water : new_water = 33.333333333333334 := by
    simp_all
  have h_new_volume : new_volume = 133.33333333333334 := rfl
  have h_target : (new_water / new_volume) * 100 = 25 := by
    simp [h_new_water, h_new_volume]
  exact h_target

end target_percentage_of_water_l597_597313


namespace find_y_l597_597960

theorem find_y
  (x y : ℝ)
  (h1 : x - y = 10)
  (h2 : x + y = 8) : y = -1 :=
by
  sorry

end find_y_l597_597960


namespace lambda_range_l597_597939

open Real

def vector_dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem lambda_range (λ : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, λ)
  (vector_dot_product a b > 0) ↔ λ ∈ SetOf (λ, (λ < 1 ∧ λ ≠ -4) ∨ λ ∈ Ioo (-∞) -↑4) := sorry

end lambda_range_l597_597939


namespace james_total_money_l597_597131

theorem james_total_money (bills : ℕ) (value_per_bill : ℕ) (initial_money : ℕ) : 
  bills = 3 → value_per_bill = 20 → initial_money = 75 → initial_money + (bills * value_per_bill) = 135 :=
by
  intros hb hv hi
  rw [hb, hv, hi]
  -- Algebraic simplification
  sorry

end james_total_money_l597_597131


namespace find_initial_amount_l597_597947

noncomputable def initial_amount (diff : ℝ) : ℝ :=
  diff / (1.4641 - 1.44)

theorem find_initial_amount
  (diff : ℝ)
  (h : diff = 964.0000000000146) :
  initial_amount diff = 40000 :=
by
  -- the steps to prove this can be added here later
  sorry

end find_initial_amount_l597_597947


namespace minimum_rubles_to_reverse_chips_l597_597369

theorem minimum_rubles_to_reverse_chips (n : ℕ) (h : n = 100)
  (adjacent_cost : ℕ → ℕ → ℕ)
  (free_cost : ℕ → ℕ → Prop)
  (reverse_cost : ℕ) :
  (∀ i j, i + 1 = j → adjacent_cost i j = 1) →
  (∀ i j, i + 5 = j → free_cost i j) →
  reverse_cost = 61 :=
by
  sorry

end minimum_rubles_to_reverse_chips_l597_597369


namespace palindrome_divisibility_probability_l597_597790

theorem palindrome_divisibility_probability :
  let palindromes := {n : ℕ | n >= 100 ∧ n < 1000 ∧ (∃ a b, ∃ (h₁ : a ≠ 0), 10 ≤ n ∧ n = 101 * a + 10 * b)};
  let count_palindromes := (palindromes.toFinset.card : ℕ);
  let divisible_by_five := {n ∈ palindromes | n % 5 = 0};
  let count_divisible_by_five := (divisible_by_five.toFinset.card : ℕ);
  ∑ palindromes) :=
begin
  let palindromes : set ℕ := {n ∈ Ico 100 1000 | ∃ a b,  a ≠ 0 ∧ n = 101 * a + 10 * b},
  let divisible_by_five : set ℕ := {n ∈ palindromes | n % 5 = 0},
  have h1 : (palindromes.toFinset.card : ℕ) = 90, sorry,
  have h2 : (divisible_by_five.toFinset.card : ℕ) = 10, sorry,
  exact (h2 : ℝ) / (h1 : ℝ) = 1 / 9,
end

end palindrome_divisibility_probability_l597_597790


namespace function_properties_l597_597811

theorem function_properties (x : ℝ) :
  let f := fun x => -3^(abs x)
  let g := fun x => log ((0.5 : ℝ)) (abs x)
  (∀ x, f (-x) = f x) ∧ (∀ x < 0, f x < f (x + 1))
  → (∀ x, g (-x) = g x) ∧ (∀ x < 0, g x < g (x + 1)) :=
by
  let f := fun x => -3^(abs x)
  let g := fun x => log ((0.5 : ℝ)) (abs x)
  intro h
  cases h with hf_hmon hf_hinc
  split
  {
    -- Proof that g is even
    sorry
  }
  {
    -- Proof that g is increasing on (-∞, 0)
    sorry
  }

end function_properties_l597_597811


namespace triangle_area_ratio_ext_l597_597991

theorem triangle_area_ratio_ext :
  ∀ (a b c : ℝ), 
  -- triangle side lengths
  let S_ABC := abs (0.5 * a * b * sin (arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) in
  let S_AAA'' := S_ABC * (1 + (a * (a + b + c)) / (b * c)) in
  let S_BBB'' := S_ABC * (1 + (b * (a + b + c)) / (a * c)) in
  let S_CCC'' := S_ABC * (1 + (c * (a + b + c)) / (a * b)) in
  let total_area_ratio := (S_AAA'' + S_BBB'' + S_CCC'') / S_ABC in
  -- The required ratio simplifies to 13
  total_area_ratio = 13 :=
by 
  intros a b c
  let S_ABC := abs (0.5 * a * b * sin (arccos ((a^2 + b^2 - c^2) / (2 * a * b))))
  let S_AAA'' := S_ABC * (1 + (a * (a + b + c)) / (b * c))
  let S_BBB'' := S_ABC * (1 + (b * (a + b + c)) / (a * c))
  let S_CCC'' := S_ABC * (1 + (c * (a + b + c)) / (a * b))
  let total_area_ratio := (S_AAA'' + S_BBB'' + S_CCC'') / S_ABC
  have : total_area_ratio = 3 + (a + b + c) * ((a / (b * c) + b / (a * c) + c / (a * b)) + 1 
  rw [total_area_ratio] 
  -- Use of AM-GM inequality here
  have h_am_gm : (a / (b * c) + b / (a * c) + c / (a * b)) * (a * b * c) = 3 
  rw [h_am_gm]
  have eq1 : total_area_ratio = 4 + 3
  sorry 

end triangle_area_ratio_ext_l597_597991


namespace repeating_block_length_7_div_13_l597_597601

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l597_597601


namespace equidistant_from_center_l597_597153

structure EquilateralTriangle (ABC : Type) :=
  (A B C : ABC)
  (equilateral : ∀ (a b c : ℝ), a = b ∧ b = c)

noncomputable def circumscribing_circle_center (ABC : EquilateralTriangle) : ABC := sorry

noncomputable def endpoint_of_diameter (O A : ABC) : ABC := sorry

theorem equidistant_from_center (ABC : EquilateralTriangle) (O : circumscribing_circle_center ABC) (A D P Q : ABC) (arc1 arc2 : set ABC) (h1 : P ∈ arc1) (h2 : Q ∈ arc2) (h3 : set.center arc1 = A) (h4 : set.center arc2 = D) (h5 : (metric.dist P B) = (metric.dist Q B)) : 
  metric.dist P O = metric.dist Q O := 
begin
  sorry,
end

end equidistant_from_center_l597_597153


namespace purely_imaginary_sol_l597_597906

theorem purely_imaginary_sol (x : ℝ) 
  (h1 : (x^2 - 1) = 0)
  (h_imag : (x^2 + 3 * x + 2) ≠ 0) :
  x = 1 :=
sorry

end purely_imaginary_sol_l597_597906


namespace exists_n_l597_597769

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0.5 then x + 0.5 else x * x

def a_seq (a : ℝ) : ℕ → ℝ
| 0     => a
| (n+1) => f (a_seq n)

def b_seq (b : ℝ) : ℕ → ℝ
| 0     => b
| (n+1) => f (b_seq n)

theorem exists_n (a b : ℝ) (h : 0 < a ∧ a < b ∧ b < 1) :
  ∃ n, (a_seq a n - a_seq a (n-1)) * (b_seq b n - b_seq b (n-1)) < 0 :=
by
  sorry

end exists_n_l597_597769


namespace player1_points_after_13_rotations_l597_597703

theorem player1_points_after_13_rotations :
  ∃ (player1_points : ℕ), 
    (∀ (i : ℕ),  (i = 5 → player1_points = 72) ∧ (i = 9 → player1_points = 84)) → 
    player1_points = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597703


namespace mean_age_of_children_l597_597571

theorem mean_age_of_children :
  let ages := [6, 6, 6, 6, 8, 8, 16] in
  let total_age := 6 + 6 + 6 + 6 + 8 + 8 + 16 in
  let number_of_children := 7 in
  (total_age / number_of_children : ℝ) = 8 := sorry

end mean_age_of_children_l597_597571


namespace years_between_2000_and_3000_with_palindrome_and_prime_factors_l597_597456

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_two_digit_prime_palindrome (n : ℕ) : Prop :=
  n = 11 -- 11 is the only two-digit prime palindrome

def factors_as_two_digit_prime_palindrome_product (n : ℕ) : Prop :=
  ∃ (p q : ℕ), is_two_digit_prime_palindrome p ∧ is_two_digit_prime_palindrome q ∧ n = p * q

theorem years_between_2000_and_3000_with_palindrome_and_prime_factors :
  {y | 2000 ≤ y ∧ y < 3000 ∧ is_palindrome y ∧ factors_as_two_digit_prime_palindrome_product y}.card = 0 :=
by
  sorry

end years_between_2000_and_3000_with_palindrome_and_prime_factors_l597_597456


namespace increase_interval_l597_597651

-- Definitions of the function and conditions
def f (x : ℝ) : ℝ := log (x^2 - 4 * x + 3)

-- Condition: The domain where the function is defined
def domain (x : ℝ) : Prop := x < 1 ∨ x > 3

-- Condition: Monotonicity of inner function t = x^2 - 4x + 3
def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b → f a < f b

def t (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Proof statement indicating that f(x) is increasing on (3, +∞)
theorem increase_interval (x : ℝ) (h1 : x > 3) (h2 : domain x) :
  is_increasing f 3 x :=
sorry

end increase_interval_l597_597651


namespace fraction_simplification_l597_597954

variable (a b x : ℝ)
variable (h1 : x = a / b)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : a = b * x ^ 2)

theorem fraction_simplification : (a + b) / (a - b) = (x ^ 2 + 1) / (x ^ 2 - 1) := by
  sorry

end fraction_simplification_l597_597954


namespace large_island_graphstums_odd_l597_597214

-- Definitions corresponding to conditions
def small_island_graphstums : ℕ := 6

def is_rectangular_shape (n : ℕ) : Prop := 
  ∃ a b : ℕ, a * b = n

def forms_closed_path (n : ℕ) : Prop := 
  (∀ (g : ℕ), g < n, ∃ (diagonal_road : ℕ), ...)
  -- This part symbolically represents the closed path formed by roads in each graphstum.

-- Main theorem statement
theorem large_island_graphstums_odd :
  ∃ n : ℕ, (n % 2 = 1 ∧ n = 9) ∧ is_rectangular_shape n ∧ forms_closed_path n := 
sorry

end large_island_graphstums_odd_l597_597214


namespace james_total_money_l597_597134

section
-- Conditions
def number_of_bills : ℕ := 3
def value_of_each_bill : ℕ := 20
def initial_wallet_amount : ℕ := 75

-- Question:
-- What is the total amount of money James has now?
def total_value_of_bills : ℕ := number_of_bills * value_of_each_bill
def total_money_now : ℕ := initial_wallet_amount + total_value_of_bills

-- Theorem stating that he has $135 now.
theorem james_total_money : total_money_now = 135 := 
  by
    sorry
end

end james_total_money_l597_597134


namespace sum_largest_smallest_three_digit_l597_597758

theorem sum_largest_smallest_three_digit (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = 0) (h2 : d2 = 2) (h3 : d3 = 4) (h4 : d4 = 6) :
  let largest_number := max (max (100 * d4 + 10 * d3 + d2) (100 * d4 + 10 * d2 + d3)) (max (100 * d3 + 10 * d4 + d2) (100 * d3 + 10 * d2 + d4))
      smallest_number := min (min (100 * d2 + 10 * d1 + d3) (100 * d3 + 10 * d1 + d2)) (min (100 * d2 + 10 * d3 + d1) (100 * d3 + 10 * d2 + d1)) in
  largest_number + smallest_number = 846 := 
by 
  sorry

end sum_largest_smallest_three_digit_l597_597758


namespace jason_reroll_probability_l597_597994

theorem jason_reroll_probability :
  let dice_sides := 6
  let rolls := fin 3 → fin dice_sides
  let sum := (r : rolls) → fin dice_sides + fin dice_sides + fin dice_sides
  let optimized_play := True -- Assume Jason always plays optimally
  let win_condition := sum = 10
  let reroll_two_probability := 5 / 12

  -- Define the game logic and probabilities here
in sorry

end jason_reroll_probability_l597_597994


namespace count_rectangles_in_5x5_grid_l597_597734

theorem count_rectangles_in_5x5_grid :
  let n := 5 in
  ∑ x in finset.range n.succ, (n - x) * (n - x) * (n - x) = 225 :=
by {
  simp only [finset.range, finset.sum],
  sorry
}

end count_rectangles_in_5x5_grid_l597_597734


namespace points_player_1_after_13_rotations_l597_597681

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end points_player_1_after_13_rotations_l597_597681


namespace ratio_of_running_to_swimming_l597_597710

variable (Speed_swimming Time_swimming Distance_total Speed_factor : ℕ)

theorem ratio_of_running_to_swimming :
  let Distance_swimming := Speed_swimming * Time_swimming
  let Distance_running := Distance_total - Distance_swimming
  let Speed_running := Speed_factor * Speed_swimming
  let Time_running := Distance_running / Speed_running
  (Distance_total = 12) ∧
  (Speed_swimming = 2) ∧
  (Time_swimming = 2) ∧
  (Speed_factor = 4) →
  (Time_running : ℕ) / Time_swimming = 1 / 2 :=
by
  intros
  sorry

end ratio_of_running_to_swimming_l597_597710


namespace proof1_proof2_l597_597026

variable (a : ℝ) (m n : ℝ)
axiom am_eq_two : a^m = 2
axiom an_eq_three : a^n = 3

theorem proof1 : a^(4 * m + 3 * n) = 432 := by
  sorry

theorem proof2 : a^(5 * m - 2 * n) = 32 / 9 := by
  sorry

end proof1_proof2_l597_597026


namespace sum_of_first_11_terms_is_minus_66_l597_597110

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d 

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a n + a 1)) / 2

theorem sum_of_first_11_terms_is_minus_66 
  (a : ℕ → ℤ) 
  (h_seq : arithmetic_sequence a)
  (h_roots : ∃ a2 a10, (a2 = a 2 ∧ a10 = a 10) ∧ (a2 + a10 = -12) ∧ (a2 * a10 = -8)) 
  : sum_of_first_n_terms a 11 = -66 :=
by
  sorry

end sum_of_first_11_terms_is_minus_66_l597_597110


namespace distinct_primes_count_l597_597081

theorem distinct_primes_count : 
  ∃ n, 
  (95 = 5 * 19) ∧ 
  (Prime 97) ∧ 
  (99 = 3^2 * 11) ∧ 
  (Prime 101) ∧ 
  (n = 6 ∧ (∀ primes, primes ∈ [5, 19, 97, 3, 11, 101] → primes ∈ PrimeFactors (95 * 97 * 99 * 101))) := 
sorry

end distinct_primes_count_l597_597081


namespace correct_statements_are_three_l597_597429

-- Define the conditions as propositions
def condition1 : Prop := ∀ (a b c : ℝ), 
  (∃ (A B C : ℝ -> ℝ),  A ≠ B ∧ B ≠ C ∧ A ≠ C) → 
  (∃ (A B C : Type), true)

def condition2 : Prop := ∀ (a b c : ℝ), 
  (∀ (A B C : ℝ -> ℝ), (A a ≠ B a) -> 
  (∃ (S : Type), ∃ (x y : ℝ), S = (x, y)))

def condition3 : Prop := ∀ (a b c : ℝ), 
  (∀ (A B C : ℝ -> ℝ), A ≠ B → B ≠ C → A ≠ C → 
  (∃ (H : Type), H = (a, b) ∧ H = (b, c)))

def condition4 : Prop := ∀ (a b c : ℝ), 
  (∀ (A B C : ℝ -> ℝ), true → 
  (∀ P Q R : Type, true))

def condition5 : Prop := ∀ (a b c : ℝ), 
  (∀ (A B C : ℝ -> ℝ), true → 
  (∃ (O : Type), O = (a, b, c)))

-- Define which statements are correct
def correct_statements : ℕ := 3

-- Define a theorem that counts the correct statements
theorem correct_statements_are_three :
  (condition3 ∧ condition4 ∧ condition5) ∧ ¬condition1 ∧ ¬condition2 →
  correct_statements = 3 := 
by
  sorry

end correct_statements_are_three_l597_597429


namespace triangle_area_unchanged_l597_597966

theorem triangle_area_unchanged (a h : ℝ) (S : ℝ) (hS : S = (1/2) * a * h) : 
  let a' := a / 3 
  let h' := 3 * h in 
  (1/2) * a' * h' = S := by
  sorry

end triangle_area_unchanged_l597_597966


namespace symmetry_proof_l597_597113

-- Define the coordinates of point A
def A : ℝ × ℝ := (-1, 8)

-- Define the reflection property across the y-axis
def is_reflection_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

-- Define the point B which we need to prove
def B : ℝ × ℝ := (1, 8)

-- The proof statement
theorem symmetry_proof :
  is_reflection_y_axis A B :=
by
  sorry

end symmetry_proof_l597_597113


namespace black_rectangle_ways_l597_597737

theorem black_rectangle_ways : ∑ a in Finset.range 5, ∑ b in Finset.range 5, (5 - a) * (5 - b) = 225 := sorry

end black_rectangle_ways_l597_597737


namespace repeat_block_of_7_div_13_l597_597634

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l597_597634


namespace repeating_decimal_block_length_l597_597593

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l597_597593


namespace count_special_sequences_l597_597455

theorem count_special_sequences : 
  let length := 15 in 
  let total := 268 in
  ∃ (count : ℕ), 
    (∀ (seq : Fin length → Fin 2), 
       (allZerosConsecutive : (seq[length-1] = 0) ∨ (∃ i : Fin length, 
          seq[i] = 1 ∧ (∀ j : Fin length, j < i → seq[j] = 0) ∧ 
          (∀ j : Fin length, j > i → seq[j] = 1))) ∧ 
       (allOnesConsecutive : (seq[length-1] = 1) ∨ (∃ i : Fin length, 
          seq[i] = 0 ∧ (∀ j : Fin length, j < i → seq[j] = 1) ∧ 
          (∀ j : Fin length, j > i → seq[j] = 0))) ∧ 
       ((allZerosConsecutive ∨ allOnesConsecutive) → seq = seq ∨ both) ∧ 
    count = total :=
  sorry

end count_special_sequences_l597_597455


namespace circle_symmetric_line_l597_597967

theorem circle_symmetric_line (m : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 + 2*x - 4*y = 0) → (3*x + y + m = 0)) →
  m = 1 :=
by
  intro h
  sorry

end circle_symmetric_line_l597_597967


namespace right_triangle_hypotenuse_l597_597316

noncomputable def hypotenuse_length (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem right_triangle_hypotenuse :
  ∃ (x y h : ℝ), 
    (1/3) * Real.pi * y^2 * x = 1280 * Real.pi ∧
    (1/3) * Real.pi * x^2 * y = 450 * Real.pi ∧
    h = hypotenuse_length x y ∧
    h ≈ 23.53 :=
by
  sorry

end right_triangle_hypotenuse_l597_597316


namespace repeating_block_length_7_div_13_l597_597597

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l597_597597


namespace repeating_decimal_block_length_l597_597588

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l597_597588


namespace triangle_disjoint_probability_l597_597100

theorem triangle_disjoint_probability :
  (∀ (P : Fin 6 → Prop), 
    let A := {x : Fin 6 // P x}
    let B := {x : Fin 6 // ¬ P x}
    (∃ (f : Fin 3 → A) (g : Fin 3 → B), 
      Set.pairwise_disjoint (Set.range f) ∧ Set.pairwise_disjoint (Set.range g)) →
      (Probability := (3/10) : ℚ)) := 
by sorry

end triangle_disjoint_probability_l597_597100


namespace player_1_points_l597_597673

-- Definition: point distribution on the table.
noncomputable def sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

-- Conditions
axiom player_5_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(5 + i) % 16]) = 72
axiom player_9_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(9 + i) % 16]) = 84

-- Question translated to proof statement:
theorem player_1_points (rotations : ℕ) (p5_points : ℕ) (p9_points : ℕ) :
  rotations = 13 → p5_points = 72 → p9_points = 84 →
  ∑ i in finset.range rotations, (sector_points[(1 + i) % 16]) = 20 :=
by
  sorry

end player_1_points_l597_597673


namespace coin_toss_probability_l597_597263

open BigOperators

theorem coin_toss_probability (n : ℕ) :
  2^(-2 * n) * ∑ k in Finset.range (n + 1), (Nat.choose n k)^2 = 2^(-2 * n) * Nat.choose (2 * n) n := by
sorry

end coin_toss_probability_l597_597263


namespace find_a_b_find_range_m_l597_597068

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x ^ 2 - b * log x

theorem find_a_b (a b : ℝ) (h1 : f 1 a b = 1) (h2 : deriv (λ x, f x a b) 1 = 0) : 
a = 1 ∧ b = 2 := 
sorry

noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) (m : ℝ) : ℝ := f x a b - x^2 + m*(x-1)

theorem find_range_m (m : ℝ) :
  (∃ (a = 1) (b = 2), (∀ x ∈ set.Ioc 0 1, g x 1 2 m ≥ 0) ↔ m ∈ set.Iic (2 : ℝ)) := 
sorry

end find_a_b_find_range_m_l597_597068


namespace player_1_points_after_13_rotations_l597_597676

-- Add necessary definitions and state the problem in Lean
def sectors : Fin 16 → ℕ
| ⟨0, _⟩ := 0
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 2
| ⟨3, _⟩ := 3
| ⟨4, _⟩ := 4
| ⟨5, _⟩ := 5
| ⟨6, _⟩ := 6
| ⟨7, _⟩ := 7
| ⟨8, _⟩ := 8
| ⟨9, _⟩ := 7
| ⟨10, _⟩ := 6
| ⟨11, _⟩ := 5
| ⟨12, _⟩ := 4
| ⟨13, _⟩ := 3
| ⟨14, _⟩ := 2
| ⟨15, _⟩ := 1

def points_earned (player_offset : Fin 16) (rotations : ℕ) : ℕ :=
List.sum (List.map sectors
  (List.map (λ n => (Fin.add (Fin.ofNat n) player_offset)) (List.range rotations)))

theorem player_1_points_after_13_rotations 
  (p5_points : points_earned ⟨5, by decide⟩ 13 = 72)
  (p9_points : points_earned ⟨9, by decide⟩ 13 = 84) :
  points_earned ⟨1, by decide⟩ 13 = 20 := 
sorry

end player_1_points_after_13_rotations_l597_597676


namespace weeks_to_cover_expenses_is_three_l597_597853

-- Let us define the conditions.
def earnings_per_weekday_hour := 20 
def teaching_earnings_per_week := 100
def monthly_expenses := 1200

def weekday_babysitting_hours := 3
def weekday_babysitting_days := 5
def min_saturday_hours := 2
def max_saturday_hours := 5

-- Define the uniformly distributed average hours for Saturday Babysitting
def average_saturday_hours := (min_saturday_hours + max_saturday_hours) / 2

-- Calculate weekly earnings from babysitting and teaching
def weekly_earnings := 
  (weekday_babysitting_days * weekday_babysitting_hours * earnings_per_weekday_hour) + 
  (average_saturday_hours * earnings_per_weekday_hour) + 
  teaching_earnings_per_week

-- Number of weeks it takes to cover monthly expenses on average
def weeks_to_cover_expenses := (monthly_expenses : ℝ) / weekly_earnings

-- We need to prove that the number of weeks required is exactly 3 when rounding up.
theorem weeks_to_cover_expenses_is_three : 
  ceiling weeks_to_cover_expenses = 3 := 
  by 
    sorry

end weeks_to_cover_expenses_is_three_l597_597853


namespace range_of_real_roots_l597_597430

theorem range_of_real_roots (a : ℝ) :
  (∃ x : ℝ, x^2 + 4*a*x - 4*a + 3 = 0) ∨
  (∃ x : ℝ, x^2 + (a-1)*x + a^2 = 0) ∨
  (∃ x : ℝ, x^2 + 2*a*x - 2*a = 0) ↔
  a >= -1 ∨ a <= -3/2 :=
  sorry

end range_of_real_roots_l597_597430


namespace number_of_solutions_l597_597872

def complex_satisfying_condition (z : ℂ) : Prop :=
  abs z = 1 ∧ (z^nat.factorial 7 - z^nat.factorial 6).im = 0

theorem number_of_solutions : 
  {z : ℂ | complex_satisfying_condition z}.to_finset.card = 2800 := 
sorry

end number_of_solutions_l597_597872


namespace parabola_transformation_l597_597708

theorem parabola_transformation :
  ∀ (x : ℝ), (∃ (h : ℝ → ℝ), (∀ x, h (x - 4) = 2 * x^2 - 1) ∧ (h x = 2 * (x - 4)^2 - 1)) :=
begin
  intros x,
  use (λ x, 2 * x^2),
  split,
  { intros x,
    simp, -- simplification step to show transformation
    sorry
  },
  { sorry
  }
end

end parabola_transformation_l597_597708


namespace driving_time_per_trip_l597_597207

-- Define the conditions
def filling_time_per_trip : ℕ := 15
def number_of_trips : ℕ := 6
def total_moving_hours : ℕ := 7
def total_moving_time : ℕ := total_moving_hours * 60

-- Define the problem
theorem driving_time_per_trip :
  (total_moving_time - (filling_time_per_trip * number_of_trips)) / number_of_trips = 55 :=
by
  sorry

end driving_time_per_trip_l597_597207


namespace raisins_in_boxes_l597_597335

theorem raisins_in_boxes :
  ∃ x : ℕ, 72 + 74 + 3 * x = 437 ∧ x = 97 :=
by
  existsi 97
  split
  · rw [←add_assoc, add_comm 146, add_assoc]; exact rfl
  · exact rfl

end raisins_in_boxes_l597_597335


namespace slope_of_line_l597_597424

theorem slope_of_line (a : ℝ) (m n : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f : ℝ → ℝ := λ x, a^(x-2) + 1) :
  f m = n →
  P : ℝ × ℝ := (m, n) →
  Q : ℝ × ℝ := (m-1, n) →
  (∃ k : ℝ, (∀ x y, y - n = k * (x - (m-1)) → ∃ z, (x + 1)^2 + (y - 1)^2 = 9) ∧ 
               (chord_length = 3 * sqrt 2)) →
  k = -1 ∨ k = -7 :=
sorry

end slope_of_line_l597_597424


namespace custom_op_evaluation_l597_597096

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_evaluation : (custom_op 9 6) - (custom_op 6 9) = -12 := by
  sorry

end custom_op_evaluation_l597_597096


namespace product_sequence_equals_l597_597834

-- Define the form of each fraction in the sequence
def frac (k : ℕ) : ℚ := (4 * k : ℚ) / (4 * k + 4)

-- Define the product of the sequence from k=1 to k=501
def productSequence : ℚ :=
  (finset.range 501).prod (λ k => frac (k + 1))

-- The theorem that the product equals 1/502
theorem product_sequence_equals : productSequence = 1 / 502 := by
  sorry

end product_sequence_equals_l597_597834


namespace num_participants_k1989_num_participants_k1988_l597_597806

theorem num_participants_k1989 (n k : ℕ) (A : ℕ) (p : ℕ) 
  (hp : p > 1) (h1 : k = 1989) 
  (h2 : A * p^(n-1) ≤ 2 * k * (n-1)) 
  (h3 : A + A * p + A * p^2 + ... + A * p^(n-2) ≥ k * (n-1) * (n-2)) : 
  n = 2 := 
sorry

theorem num_participants_k1988 (n k : ℕ) (A : ℕ) (p : ℕ) 
  (hp : p > 1) (h1 : k = 1988) 
  (h2 : A * p^(n-1) ≤ 2 * k * (n-1)) 
  (h3 : A + A * p + A * p^2 + ... + A * p^(n-2) ≥ k * (n-1) * (n-2)) : 
  n = 2 ∨ n = 3 := 
sorry

end num_participants_k1989_num_participants_k1988_l597_597806


namespace interval_length_sum_l597_597866

theorem interval_length_sum (x : ℝ) : 
  (0 < x ∧ x < 2) ∧ (Real.sin x > 1/2) → 
  (\let I := setOf (λ x, (Real.sin x > 1/2) ∧ (0 < x ∧ x < 2)) in
   (∃ (a b : ℝ), a < b ∧ I = λ x, a < x ∧ x < b ∧ (Real.sin x > 1/2))) 
   ∧ abs (((5*Real.pi/6) - (Real.pi/6)) - 2.09) < 0.02 :=
by
  sorry

end interval_length_sum_l597_597866


namespace part1_part2_l597_597069

noncomputable def f (x : ℝ) : ℝ := |x| + |x + 1|

theorem part1 (x : ℝ) : f(x) > 3 ↔ (x > 1 ∨ x < -2) :=
sorry

noncomputable def f_piecewise (x : ℝ) : ℝ :=
if x >= 0 then 2*x + 1
else if -1 < x then 1
else -2*x - 1

theorem part2 (m : ℝ) : ∀ x : ℝ, (m^2 + 3*m + 2*f_piecewise(x) >= 0) ↔ (m >= -1 ∨ m <= -2) :=
sorry

end part1_part2_l597_597069


namespace parabola_focus_directrix_distance_l597_597428

def parabola_distance_to_directrix : ℕ :=
  let p := 2 in
  let focus := (p / 2, 0) in
  let directrix := - (p / 2) in
  let distance := abs (focus.1 - directrix) in
  distance

theorem parabola_focus_directrix_distance :
  parabola_distance_to_directrix = 2 :=
by
  sorry

end parabola_focus_directrix_distance_l597_597428


namespace fraction_representation_of_2_375_l597_597740

theorem fraction_representation_of_2_375 : 2.375 = 19 / 8 := by
  sorry

end fraction_representation_of_2_375_l597_597740


namespace total_legs_at_pet_shop_l597_597818

theorem total_legs_at_pet_shop : 
  let birds := 3 in
  let dogs := 5 in
  let snakes := 4 in
  let spiders := 1 in
  let legs_bird := 2 in
  let legs_dog := 4 in
  let legs_snake := 0 in
  let legs_spider := 8 in
  (birds * legs_bird + dogs * legs_dog + snakes * legs_snake + spiders * legs_spider) = 34 :=
by 
  -- Proof will be here
  sorry

end total_legs_at_pet_shop_l597_597818


namespace distance_from_A_to_C_l597_597202

noncomputable def sam_speed : ℕ := 50 -- Sam's speed in meters per minute
noncomputable def sam_time : ℕ := 20  -- Sam's travel time in minutes
def distance_BC : ℕ := 400 -- Distance from point B to point C in meters
def total_distance : ℕ := sam_speed * sam_time -- Total distance from A to B

theorem distance_from_A_to_C : total_distance - distance_BC = 600 := by
  sorry

end distance_from_A_to_C_l597_597202


namespace find_dimes_spent_l597_597532

theorem find_dimes_spent (total_spent ice_cream_spent baseball_card_spent : ℝ) (value_of_dime : ℝ) (number_of_dimes : ℕ)
  (h1 : total_spent = 1.22) 
  (h2 : ice_cream_spent = 0.02) 
  (h3 : baseball_card_spent = total_spent - ice_cream_spent) 
  (h4 : value_of_dime = 0.10) 
  (h5 : number_of_dimes = (baseball_card_spent / value_of_dime).to_nat) : 
  number_of_dimes = 12 := 
by {
  sorry
}

end find_dimes_spent_l597_597532


namespace baxter_purchases_pounds_over_minimum_l597_597332

theorem baxter_purchases_pounds_over_minimum :
  ∃ (pounds_over_min: ℕ),
    (∀ (total_spent tax discount cost_per_pound min_pounds pounds_purchased: ℝ),
      total_spent = 120.96 →
      tax = 0.08 →
      discount = 0.10 →
      cost_per_pound = 3 →
      min_pounds = 15 →
      pounds_purchased ≥ 25 →
      total_spent = (pounds_purchased * cost_per_pound * (1 - discount)) * (1 + tax) →
      pounds_over_min = pounds_purchased - min_pounds) →
  pounds_over_min = 26 :=
begin
  sorry -- Proof not required
end

end baxter_purchases_pounds_over_minimum_l597_597332


namespace raisins_in_other_three_boxes_l597_597337

-- Definitions of the known quantities
def total_raisins : ℕ := 437
def box1_raisins : ℕ := 72
def box2_raisins : ℕ := 74

-- The goal is to prove that each of the other three boxes has 97 raisins
theorem raisins_in_other_three_boxes :
  total_raisins - (box1_raisins + box2_raisins) = 3 * 97 :=
by
  sorry

end raisins_in_other_three_boxes_l597_597337


namespace number_of_factors_of_M_l597_597084

def M : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_factors_of_M : Nat.factors M = 120 := sorry

end number_of_factors_of_M_l597_597084


namespace repeating_block_length_7_div_13_l597_597578

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597578


namespace time_to_cross_signal_pole_is_approximately_18_l597_597775

noncomputable def length_of_train := 300 -- meters
noncomputable def length_of_platform := 150.00000000000006 -- meters
noncomputable def time_to_cross_platform := 27 -- seconds

noncomputable def speed_of_train : ℝ := 
  (length_of_train + length_of_platform) / time_to_cross_platform

noncomputable def time_to_cross_signal_pole : ℝ := 
  length_of_train / speed_of_train

theorem time_to_cross_signal_pole_is_approximately_18 : 
  abs (time_to_cross_signal_pole - 18) < 0.0000000001 :=
by sorry

end time_to_cross_signal_pole_is_approximately_18_l597_597775


namespace value_of_fraction_l597_597511

theorem value_of_fraction {n : ℕ} (h : n = 3 ^ 4053) : n / (3 ^ 2) = 3 ^ 4051 :=
by
  rw [h]
  rw [pow_sub (3 : ℕ) _ _ (le_of_lt two_lt_4053)]
  sorry

end value_of_fraction_l597_597511


namespace total_profit_is_60000_l597_597282

-- Definitions for initial investments and c's share of the profit
variable (a_investment : ℝ := 45000)
variable (b_investment : ℝ := 63000)
variable (c_investment : ℝ := 72000)
variable (c_share : ℝ := 24000)

-- Define the greatest common divisor
def gcd (x y z : ℝ) : ℝ := 9000  -- Given that gcd(45000, 63000, 72000) = 9000

-- Define the simplified ratio
def ratio : ℝ × ℝ × ℝ := (a_investment / gcd a_investment b_investment c_investment,
                         b_investment / gcd a_investment b_investment c_investment,
                         c_investment / gcd a_investment b_investment c_investment)

-- Calculate the total profit based on the given share and ratio
def total_profit : ℝ :=
  let one_part := c_share / (c_investment / gcd a_investment b_investment c_investment) in
  let total_parts := (a_investment / gcd a_investment b_investment c_investment +
                      b_investment / gcd a_investment b_investment c_investment +
                      c_investment / gcd a_investment b_investment c_investment) in
  total_parts * one_part

-- Problem statement: Prove the total profit equals $60,000
theorem total_profit_is_60000 : total_profit = 60000 := by
  -- Proof goes here
  sorry

end total_profit_is_60000_l597_597282


namespace polynomial_root_triples_l597_597878

theorem polynomial_root_triples (a b c : ℝ) :
  (∀ x : ℝ, x > 0 → (x^4 + a * x^3 + b * x^2 + c * x + b = 0)) ↔ (a, b, c) = (-21, 112, -204) ∨ (a, b, c) = (-12, 48, -80) :=
by
  sorry

end polynomial_root_triples_l597_597878


namespace line_l2_equation_min_area_triangle_EPQ_l597_597981

theorem line_l2_equation (t PQ : ℝ) (h_t : t > 0) (h_PQ : PQ = 6) :
  (l_2 = {p | p.y = 0}) ∨ (l_2 = {p | 4 * p.x - 3 * p.y - 1 = 0}) :=
sorry

theorem min_area_triangle_EPQ (t : ℝ) (t_pos_int : t ∈ ℕ ∧ t > 0) 
  (AM_le_2BM : ∀ M, AM ≤ 2 * BM) :
  ∃ t_min : ℕ, t_min = t ∧ area_EPQ = (sqrt 15) / 2 :=
sorry

end line_l2_equation_min_area_triangle_EPQ_l597_597981


namespace ln_x2_gt_2_sub_ln_x1_l597_597440

open Real

theorem ln_x2_gt_2_sub_ln_x1 
  (k : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ < x₂) 
  (h₄ : ln x₁ - k * x₁ = 0) (h₅ : ln x₂ - k * x₂ = 0) : 
  ln x₂ > 2 - ln x₁ :=
sorry

end ln_x2_gt_2_sub_ln_x1_l597_597440


namespace symmetric_about_x_eq_1_l597_597842

def f (x : ℝ) : ℝ := |⌊x + 1⌋| - |⌊2 - x⌋|

theorem symmetric_about_x_eq_1 : ∀ x : ℝ, f(x) = f(2 - x) := by 
  -- proof would go here
  sorry

end symmetric_about_x_eq_1_l597_597842


namespace compute_value_l597_597841

theorem compute_value : ((-120) - (-60)) / (-30) = 2 := 
by 
  sorry

end compute_value_l597_597841


namespace largest_common_divisor_408_340_is_68_l597_597745

theorem largest_common_divisor_408_340_is_68 :
  let factors_408 := [1, 2, 3, 4, 6, 8, 12, 17, 24, 34, 51, 68, 102, 136, 204, 408]
  let factors_340 := [1, 2, 4, 5, 10, 17, 20, 34, 68, 85, 170, 340]
  ∀ d ∈ factors_408, d ∈ factors_340 → ∀ (e ∈ factors_408), (e ∈ factors_340) → d ≤ e :=
  68 := by sorry

end largest_common_divisor_408_340_is_68_l597_597745


namespace sqrt_meaningful_range_l597_597950

theorem sqrt_meaningful_range {x : ℝ} (h : x - 1 ≥ 0) : x ≥ 1 :=
sorry

end sqrt_meaningful_range_l597_597950


namespace beavers_swimming_correct_l597_597292

variable (initial_beavers remaining_beavers beavers_swimming : ℕ)

def beavers_problem : Prop :=
  initial_beavers = 2 ∧
  remaining_beavers = 1 ∧
  beavers_swimming = initial_beavers - remaining_beavers

theorem beavers_swimming_correct :
  beavers_problem initial_beavers remaining_beavers beavers_swimming → beavers_swimming = 1 :=
by
  sorry

end beavers_swimming_correct_l597_597292


namespace son_age_is_18_l597_597310

theorem son_age_is_18
  (S F : ℕ)
  (h1 : F = S + 20)
  (h2 : F + 2 = 2 * (S + 2)) :
  S = 18 :=
by sorry

end son_age_is_18_l597_597310


namespace train_length_proof_l597_597776

noncomputable def length_of_first_train (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (5 / 18) -- convert to m/s
  let total_distance := relative_speed * time
  total_distance - length2

theorem train_length_proof (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) :
  speed1 = 120 →
  speed2 = 80 →
  time = 9 →
  length2 = 270.04 →
  length_of_first_train speed1 speed2 time length2 = 230 :=
by
  intros h1 h2 h3 h4
  -- Use the defined function and simplify
  rw [h1, h2, h3, h4]
  simp [length_of_first_train]
  sorry

end train_length_proof_l597_597776


namespace solve_eqn_l597_597864

theorem solve_eqn {x : ℝ} : x^4 + (3 - x)^4 = 130 ↔ x = 0 ∨ x = 3 :=
by
  sorry

end solve_eqn_l597_597864


namespace general_formulas_sum_of_c_n_l597_597117

variable {n : ℕ}

def a_1 : ℝ := 3
def q : ℝ := 3
def b_1 : ℝ := a_1

def a_n (n : ℕ) : ℝ := (q)^n
def b_n (n : ℕ) : ℝ := 2 * n + 1
def c_n (n : ℕ) : ℝ := a_n n * b_n n

def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, c_n (i + 1)

theorem general_formulas :
  a_n 1 = 3 ∧ b_n 1 = 3 ∧ b_n 4 = a_n 2 ∧ b_n 13 = a_n 3 :=
by
  -- Proof steps go here
  sorry

theorem sum_of_c_n (n : ℕ) : S_n n = n * (q)^(n + 1) :=
by
  -- Proof steps go here
  sorry

end general_formulas_sum_of_c_n_l597_597117


namespace find_q_l597_597115

theorem find_q (A B C Q : Point) (xA yA xB yB xC yC xQ yQ : ℝ)
  (area_ABC : ℝ) : 
  A = (0, 12) → B = (15, 0) → C = (0, q) → Q = (3, 12) → area_ABC = 36 → 
  yC = q → 
  q = 9 := 
by 
  intros hA hB hC hQ hAreaABC h_yC
  sorry

end find_q_l597_597115


namespace sum_of_cubes_decomposition_l597_597645

theorem sum_of_cubes_decomposition :
  ∃ a b c d e : ℤ, (∀ x : ℤ, 1728 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ (a + b + c + d + e = 132) :=
by
  sorry

end sum_of_cubes_decomposition_l597_597645


namespace max_diff_lengths_and_possible_counts_l597_597899

/-- Given a triangle touching side BC at point C',
    consider the lengths of the 12 segments which are heights of triangles A'B'C', A'_aB'_aC'_a,
    A'_bB'_bC'_b, and A'_cB'_cC'_c.

    Prove that the maximum number of different lengths among these segments is 6,
    and identify all possible numbers of different lengths. -/
theorem max_diff_lengths_and_possible_counts
  (ABC : Triangle) (C' : Point)
  (A'B'C' A'_aB'_aC'_a A'_bB'_bC'_b A'_cB'_cC'_c : Triangle) :
  max_different_lengths {length_segment | is_height_segment length_segment (ABC, C', A'B'C', A'_aB'_aC'_a, A'_bB'_bC'_b, A'_cB'_cC'_c)} 6
  ∧ possible_different_lengths {length_segment | is_height_segment length_segment (ABC, C', A'B'C', A'_aB'_aC'_a, A'_bB'_bC'_b, A'_cB'_cC'_c)} = [6, 5, 4, 3, 2] :=
sorry

end max_diff_lengths_and_possible_counts_l597_597899


namespace relay_team_order_count_l597_597144

-- Definitions based on conditions
def six_team_members_including_jordan := 6
def members_except_jordan := 5

-- Define the problem statement as a theorem
theorem relay_team_order_count (fixed_last_lap : jordan_runs_last := true) : 
  (∏ i in (finset.range members_except_jordan).map (λ i, i + 1), i) = 120 :=
by
  -- This is a proof outline placeholder
  sorry

end relay_team_order_count_l597_597144


namespace robert_initial_balls_l597_597199

def initial_balls (R : ℕ) : Prop :=
  let balls_given := 40 / 2 in
  let balls_now := 45 in
  R + balls_given = balls_now

theorem robert_initial_balls (R : ℕ) (h : initial_balls R) : R = 25 :=
by
  sorry

end robert_initial_balls_l597_597199


namespace at_most_70_percent_triangles_are_acute_l597_597723

variables (P : Fin 100 → Type) [inhabited P] [fintype (Fin 100)]
  (h_no_three_collinear : ∀ (A B C : Fin 100), ¬collinear (P A) (P B) (P C))

theorem at_most_70_percent_triangles_are_acute
  (h_no_three_collinear : ∀ (A B C : Fin 100), ¬collinear (P A) (P B) (P C)) :
  ∃ (acute_percentage : ℝ), acute_percentage ≤ 0.7 :=
by
  sorry

end at_most_70_percent_triangles_are_acute_l597_597723


namespace multiples_of_3_ending_number_l597_597454

theorem multiples_of_3_ending_number :
  ∃ n, ∃ k, k = 93 ∧ (∀ m, 81 + 3 * m = n → 0 ≤ m ∧ m < k) ∧ n = 357 := 
by
  sorry

end multiples_of_3_ending_number_l597_597454


namespace midpoint_translation_l597_597560

-- Definitions of endpoints of segment s1
def s1_start : ℝ × ℝ := (5, 2)
def s1_end : ℝ × ℝ := (-9, 6)

-- Definition of the translation vector for segment s2
def translation_vector : ℝ × ℝ := (-3, 4)

-- Statement to prove the midpoint of segment s2 after translation
def midpoint_s2 : Prop :=
  let midpoint_s1 := ((s1_start.1 + s1_end.1) / 2, (s1_start.2 + s1_end.2) / 2) in
  let translated_midpoint := (midpoint_s1.1 + translation_vector.1, midpoint_s1.2 + translation_vector.2) in
  translated_midpoint = (-5, 8)

theorem midpoint_translation : midpoint_s2 := by
  sorry

end midpoint_translation_l597_597560


namespace triangle_shape_l597_597094

theorem triangle_shape (a b A B C : ℝ) 
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (h : a * cos (π - A) + b * sin (π / 2 + B) = 0) :
  (A = B ∨ A + B = π / 2) :=
begin
  sorry
end

end triangle_shape_l597_597094


namespace joan_spent_on_trucks_l597_597497

-- Define constants for the costs
def cost_cars : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def total_toys : ℝ := 25.62
def cost_trucks : ℝ := 25.62 - (14.88 + 4.88)

-- Statement to prove
theorem joan_spent_on_trucks : cost_trucks = 5.86 := by
  sorry

end joan_spent_on_trucks_l597_597497


namespace find_t_l597_597021

open Real

variables (t : ℝ)

def vector_oa : ℝ × ℝ := (2, 3)
def vector_ob : ℝ × ℝ := (3, t)

def vector_ab (t : ℝ) : ℝ × ℝ :=
  let (x1, y1) := vector_oa 
  let (x2, y2) := vector_ob t
  (x2 - x1, y2 - y1)
  
def magnitude (v : ℝ × ℝ) : ℝ :=
  let (x, y) := v
  sqrt (x^2 + y^2)

theorem find_t (t : ℝ) : magnitude (vector_ab t) = 1 → t = 3 := 
sorry

end find_t_l597_597021


namespace coefficient_x2_f_of_f_l597_597433

def f (x : ℝ) : ℝ :=
if x ≥ 1 then x^6 else -2 * x - 1

theorem coefficient_x2_f_of_f (x : ℝ) (h : x ≤ -1) : 
  (f (f x)).coeff 2 = 60 :=
sorry

end coefficient_x2_f_of_f_l597_597433


namespace sum_indices_equal_elements_l597_597506

noncomputable def a_seq : ℕ → ℝ 
| 1 := 0.101
| n := if n % 2 = 1 then (0.1 + 10^(-(n + 1)))^(a_seq (n - 1)) else (0.1 + 10^(-(n + 2)))^(a_seq (n - 1))

noncomputable def decreasing_rearrangement (seq: ℕ → ℝ) (n: ℕ) : (ℕ → ℝ) :=
fun i => seq (n - i + 1)

def target_sum (n : ℕ) :=
  (n / 2) * (2 + n)

theorem sum_indices_equal_elements :
  (∑ i in (finset.filter (λ x, a_seq x = decreasing_rearrangement a_seq 1011 x) (finset.range 1012)), i) = 255255 :=
by
  sorry

end sum_indices_equal_elements_l597_597506


namespace shaded_region_l597_597330

theorem shaded_region (z : ℂ) :
  (|z| ≤ 1 ∧ z.im ≥ 1 / 2) ↔ (∃ r θ : ℝ, 0 ≤ r ∧ r ≤ 1 ∧ π / 6 ≤ θ ∧ θ ≤ 5 * π / 6 ∧ z = r * exp(θ * complex.I)) :=
by
  sorry

end shaded_region_l597_597330


namespace l1_parallel_l2_l1_perpendicular_l2_l597_597420

section
variables {m : ℝ}

-- Definitions of slopes
def slope_l1 (m : ℝ) : ℝ := (m - 1) / (-1 - m)
def slope_l2 : ℝ := (2 - 0) / (1 - (-5))

-- 1. If l1 is parallel to l2, prove m = 1/2
theorem l1_parallel_l2 (hl1_parallel : slope_l1 m = slope_l2) : m = 1 / 2 :=
by 
  sorry

-- 2. If l1 is perpendicular to l2, prove m = -2
theorem l1_perpendicular_l2 (hl1_perpendicular : slope_l1 m * slope_l2 = -1) : m = -2 :=
by 
  sorry

end

end l1_parallel_l2_l1_perpendicular_l2_l597_597420


namespace james_total_money_l597_597135

section
-- Conditions
def number_of_bills : ℕ := 3
def value_of_each_bill : ℕ := 20
def initial_wallet_amount : ℕ := 75

-- Question:
-- What is the total amount of money James has now?
def total_value_of_bills : ℕ := number_of_bills * value_of_each_bill
def total_money_now : ℕ := initial_wallet_amount + total_value_of_bills

-- Theorem stating that he has $135 now.
theorem james_total_money : total_money_now = 135 := 
  by
    sorry
end

end james_total_money_l597_597135


namespace measure_BAO_l597_597988

-- Definitions for the given problem
def in_circle (C D : Point) (O : Point) (r : ℝ) : Bool := true -- O is the center of semicircle and CD is the diameter
def on_extension (p1 p2 : Line) (A : Point) : Bool := true -- A lies on the extension past p2
def on_semicircle (E : Point) (O : Point) (r : ℝ) : Bool := true -- E lies on the semicircle with center O and radius r
def intersection (α β : Line) (p : Point) : Bool := true -- B is a point of intersection

constant Point : Type
constant Line : Type

open Real

-- Setting up geometry
variables (O C D A E B : Point)
variables (r : ℝ)
variables (l1 l2 : Line)

noncomputable def measure_angle (P Q R : Point) : ℝ := sorry -- Measure of angle ∠PQR
def distance (P Q : Point) : ℝ := sorry -- Distance between points P and Q

-- Given Conditions
axiom h1 : in_circle C D O r
axiom h2 : on_extension l1 l2 A
axiom h3 : on_semicircle E O r
axiom h4 : intersection l1 l2 B
axiom h5 : distance A B = distance O D
axiom h6 : measure_angle E O D = 60

-- To Prove
theorem measure_BAO : measure_angle B A O = 15 := 
by sorry

end measure_BAO_l597_597988


namespace find_sqrt_abc_sum_l597_597160

theorem find_sqrt_abc_sum (a b c : ℝ) (h1 : b + c = 20) (h2 : c + a = 22) (h3 : a + b = 24) :
    Real.sqrt (a * b * c * (a + b + c)) = 206.1 := by
  sorry

end find_sqrt_abc_sum_l597_597160


namespace part1_part2_l597_597924

theorem part1 (a : ℝ) (x : ℝ) (h : a ≠ 0) :
    (|x - a| + |x + a + (1 / a)|) ≥ 2 * Real.sqrt 2 :=
sorry

theorem part2 (a : ℝ) (h : a ≠ 0) (h₁ : |2 - a| + |2 + a + 1 / a| ≤ 3) :
    a ∈ Set.Icc (-1 : ℝ) (-1/2) ∪ Set.Ico (1/2 : ℝ) 2 :=
sorry

end part1_part2_l597_597924


namespace repeating_decimal_block_length_l597_597590

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l597_597590


namespace sam_wins_l597_597555

variable (p : ℚ) -- p is the probability that Sam wins
variable (phit : ℚ) -- probability of hitting the target in one shot
variable (pmiss : ℚ) -- probability of missing the target in one shot

-- Define the problem and set up the conditions
def conditions : Prop := phit = 2 / 5 ∧ pmiss = 3 / 5

-- Define the equation derived from the problem
def equation (p : ℚ) (phit : ℚ) (pmiss : ℚ) : Prop :=
  p = phit + (pmiss * pmiss * p)

-- State the theorem that Sam wins with probability 5/8
theorem sam_wins (h : conditions phit pmiss) : 
  equation p phit pmiss → p = 5 / 8 :=
by
  intros
  sorry

end sam_wins_l597_597555


namespace balls_drawn_l597_597291

theorem balls_drawn (n m k l : ℕ) (h₁ : n = 18) (h₂ : m = 18) (h₃ : k = 17) (h₄ : l = 16) :
  n * m * k * l = 87984 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end balls_drawn_l597_597291


namespace constant_first_digit_sqrt_l597_597037

theorem constant_first_digit_sqrt (m : ℕ) (hm : 0 < m) : 
  ∃ n₀ : ℕ, ∀ n : ℕ, n > n₀ → (Float.fract (Float.sqrt (n^2 + 817 * n + m)) * 10).floor = 0 :=
by
  sorry

end constant_first_digit_sqrt_l597_597037


namespace intersection_sets_l597_597101

theorem intersection_sets (A B : set ℕ) (hA : A = {x ∈ ℕ | -2 < (x : ℤ) ∧ (x : ℤ) < 1}) (hB : B = {-2, -1, 0, 1}) :
  A ∩ B = {0} :=
by
  sorry

end intersection_sets_l597_597101


namespace sam_wins_probability_l597_597543

theorem sam_wins_probability : 
  let hit_prob := (2 : ℚ) / 5
      miss_prob := (3 : ℚ) / 5
      p := hit_prob + (miss_prob * miss_prob) * p
  in p = 5 / 8 := 
by
  -- Proof goes here
  sorry

end sam_wins_probability_l597_597543


namespace number_of_false_statements_l597_597704

noncomputable def line := ℝ × ℝ -> ℝ 

def intersect_pairwise (a b c : line) : Prop :=
  ∃ p1 p2 p3 : ℝ × ℝ, 
    (a p1.1 p1.2 = b p1.1 p1.2) ∧
    (a p2.1 p2.2 = c p2.1 p2.2) ∧
    (b p3.1 p3.2 = c p3.1 p3.2) ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

def statement1 (a b c : line) : Prop :=
  ∀ β : line, (∀ p : ℝ × ℝ, (a p.1 p.2 = b p.1 p.2 → β p.1 p.2)) → (∀ p : ℝ × ℝ, c p.1 p.2 → β p.1 p.2)

def statement2 (a b c : line) : Prop :=
  ∀ d : line, (∀ p : ℝ × ℝ, d p.1 p.2 = a p.1 p.2 = b p.1 p.2) → (∀ p : ℝ × ℝ, c p.1 p.2 = d p.1 p.2)

def statement3 (a b c : line) : Prop :=
  ∀ γ : line, (∀ p i : ℝ × ℝ, (i = intersect_pairwise a b c) → (γ p.1 p.2) = ((γ i.1 i.2) → (γ = a ∧ γ = b ∧ γ = c)))

def problem (a b c : line) : Prop :=
  intersect_pairwise a b c →
  (¬(statement1 a b c) → statement1 a b c = false) ∧
  (¬(statement2 a b c) → statement2 a b c = false) ∧
  statement3 a b c → (statement3 a b c = false) ∧ (¬(statement3 a b c) → statement3 a b c = false)

theorem number_of_false_statements (a b c : line) : Prop :=
  problem a b c → (statement1 a b c ∧ statement2 a b c ∧ ¬statement3 a b c) → 1 = 1

end number_of_false_statements_l597_597704


namespace find_A_l597_597789

theorem find_A (A : ℕ) (B : ℕ) (h₀ : 0 ≤ B) (h₁ : B ≤ 999) :
  1000 * A + B = (A * (A + 1)) / 2 → A = 1999 := sorry

end find_A_l597_597789


namespace slope_of_line_through_ellipse_l597_597901

theorem slope_of_line_through_ellipse 
    (C : set (ℝ × ℝ))
    (l : ℝ → ℝ)
    (M : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (k : ℝ) :
  (C = { p | ∃ x y : ℝ, p = (x, y) ∧ (x^2 / 4 + y^2 / 3 = 1) }) →
  (l = λ x, k * x + 1) →
  (M = (0, 1)) →
  (∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ y1 > 0 ∧ ((x1 - 0, y1 - 1) = (1/2) * (0 - x2, 1 - y2))) →
  (∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ (x1, y1) = (x1, k * x1 + 1) ∧ (x2, y2) = (x2, k * x2 + 1) ∧ (x1^2 / 4 + (k * x1 + 1)^2 / 3 = 1) ∧ (x2^2 / 4 + (k * x2 + 1)^2 / 3 = 1)) →
  k = 1/2 ∨ k = -1/2 :=
by
  sorry

end slope_of_line_through_ellipse_l597_597901


namespace product_of_elements_in_M_is_neg_one_l597_597036

theorem product_of_elements_in_M_is_neg_one (M : Set ℝ) (h_nonempty : M.nonempty) 
  (h_condition : ∀ x ∈ M, (1 - x) ≠ 0 → (1 / (1 - x)) ∈ M) 
  (h_four : 4 ∈ M) : 
  ∃ l : List ℝ, (∀ x ∈ l, x ∈ M) ∧ (l.product = -1) :=
sorry

end product_of_elements_in_M_is_neg_one_l597_597036


namespace divisibility_by_six_l597_597237

theorem divisibility_by_six (n : ℤ) : 6 ∣ (n^3 - n) := 
sorry

end divisibility_by_six_l597_597237


namespace no_valid_2011_matrix_l597_597770

def valid_matrix (A : ℕ → ℕ → ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ 2011 →
    (∀ k, 1 ≤ k ∧ k ≤ 4021 →
      (∃ j, 1 ≤ j ∧ j ≤ 2011 ∧ A i j = k) ∨ (∃ j, 1 ≤ j ∧ j ≤ 2011 ∧ A j i = k))

theorem no_valid_2011_matrix :
  ¬ ∃ A : ℕ → ℕ → ℕ, (∀ i j, 1 ≤ i ∧ i ≤ 2011 ∧ 1 ≤ j ∧ j ≤ 2011 → 1 ≤ A i j ∧ A i j ≤ 4021) ∧ valid_matrix A :=
by
  sorry

end no_valid_2011_matrix_l597_597770


namespace cos_theta_add_pi_over_4_l597_597958

theorem cos_theta_add_pi_over_4 :
  ∀ θ : ℝ, (cos θ = -12 / 13) → (π < θ ∧ θ < 3 * π / 2) → cos (θ + π / 4) = -7 * Real.sqrt 2 / 26 := 
by
  sorry

end cos_theta_add_pi_over_4_l597_597958


namespace triangle_ADC_properties_l597_597122

-- Definitions of the conditions
def angle_bisector (A B C D : Point) (b : Angle) : Prop := 
  ∠(A, D, C) = b / 2 ∧ ∠(A, D, B) = b / 2

def right_triangle (A B C : Point) : Prop := 
  ∠(A, B, C) = 90

def lengths (A B C : Point) (AB BC AC : ℝ) : Prop := 
  dist A B = AB ∧ dist B C = BC ∧ dist A C = AC

axiom triangle_ABC (A B C : Point) : Point
axiom D [Classical] : Point 
axiom A B C D : Point

-- Given conditions
axiom ABC_right : right_triangle A B C
axiom AD_bisector : angle_bisector A B C D ∠A B C
axiom AB_len : dist A B = 80
axiom BC_len : ∃ x : ℝ, dist B C = x
axiom AC_len : ∃ x : ℝ, dist A C = 2 * x - 8

-- The proof statement
theorem triangle_ADC_properties :
  ∃ (x : ℝ), 
  area (triangle A D C) = 520 ∧ dist B D = 230 / 13 := by
  sorry

end triangle_ADC_properties_l597_597122


namespace area_BCO_l597_597501

-- Definitions of geometrical objects
structure Trapezoid where
  A B C D O : Type
  is_trapezoid : true -- Placeholder for actual trapezoid properties

structure Point where
  x y : ℝ

-- Definitions of areas
def area (T : Type) : ℝ := sorry

-- Conditions
axiom ABC (ABCD : Trapezoid) : Type
axiom ACD (ABCD : Trapezoid) : Type
axiom ABC_area (ABCD : Trapezoid) (ABC : Type) : area ABC = 150
axiom ACD_area (ABCD : Trapezoid) (ACD : Type) : area ACD = 120

-- Theorem to be proved
theorem area_BCO (ABCD : Trapezoid) (ABC : Type) (ACD : Type) : area (ABC ∩ ACD) = 200 / 3 := by
  sorry

end area_BCO_l597_597501


namespace betty_cookies_brownies_l597_597338

def initial_cookies : ℕ := 60
def initial_brownies : ℕ := 10

def cookies_eaten (day: ℕ) : ℕ :=
  match day with
  | 1 => 2
  | 2 => 4
  | 3 => 3
  | 4 => 5
  | 5 => 4
  | 6 => 3
  | 7 => 2
  | _ => 0

def brownies_eaten (day: ℕ) : ℕ :=
  match day with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 1
  | 5 => 3
  | 6 => 2
  | 7 => 1
  | _ => 0

def total_eaten {f : ℕ → ℕ} : ℕ :=
  (List.range 7).map (λ d => f (d+1)).sum

theorem betty_cookies_brownies :
  let cookies_left := initial_cookies - total_eaten cookies_eaten in
  let brownies_left := initial_brownies - total_eaten brownies_eaten in
  cookies_left = 37 ∧ brownies_left = 0 ∧ (cookies_left - brownies_left = 37) :=
by
  sorry

end betty_cookies_brownies_l597_597338


namespace find_analytical_expression_of_f_l597_597925

-- Define the function f satisfying the condition
def f (x : ℝ) : ℝ := sorry

-- Lean 4 theorem statement
theorem find_analytical_expression_of_f :
  (∀ x : ℝ, f (x + 1) = x^2 + 2*x + 2) → (∀ x : ℝ, f x = x^2 + 1) :=
by
  -- The initial f definition and theorem statement are created
  -- The proof is omitted since the focus is on translating the problem
  sorry

end find_analytical_expression_of_f_l597_597925


namespace sum_equidistant_terms_l597_597983

def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n m : ℕ, (n < m) → a (n+1) - a n = a (m+1) - a m

variable {a : ℕ → ℤ}

theorem sum_equidistant_terms (h_seq : is_arithmetic_sequence a)
  (h_4 : a 4 = 5) : a 3 + a 5 = 10 :=
sorry

end sum_equidistant_terms_l597_597983


namespace min_score_to_increase_avg_l597_597997

theorem min_score_to_increase_avg :
  let scores := [88, 92, 75, 83, 90]
  let current_avg := (scores.sum.toFloat / scores.length)
  let target_avg := current_avg + 5
  let n := scores.length + 1
  ∃ x : Float, (scores.sum.toFloat + x) / n = target_avg ∧ x ≥ 116 :=
by
  let scores := [88, 92, 75, 83, 90]
  let current_avg := (scores.sum.toFloat / scores.length)
  let target_avg := current_avg + 5
  let n := scores.length + 1
  use 116.0
  split
  · -- show (scores.sum.toFloat + 116.0) / n = target_avg
    sorry
  · -- show 116.0 >= 116
    exact le_refl _

end min_score_to_increase_avg_l597_597997


namespace james_total_money_l597_597133

section
-- Conditions
def number_of_bills : ℕ := 3
def value_of_each_bill : ℕ := 20
def initial_wallet_amount : ℕ := 75

-- Question:
-- What is the total amount of money James has now?
def total_value_of_bills : ℕ := number_of_bills * value_of_each_bill
def total_money_now : ℕ := initial_wallet_amount + total_value_of_bills

-- Theorem stating that he has $135 now.
theorem james_total_money : total_money_now = 135 := 
  by
    sorry
end

end james_total_money_l597_597133


namespace sum_of_integers_990_l597_597240

theorem sum_of_integers_990 :
  ∃ (n m : ℕ), (n * (n + 1) = 990 ∧ (m - 1) * m * (m + 1) = 990 ∧ (n + n + 1 + m - 1 + m + m + 1 = 90)) :=
sorry

end sum_of_integers_990_l597_597240


namespace twelfth_equation_l597_597187

theorem twelfth_equation : (14 : ℤ)^2 - (12 : ℤ)^2 = 4 * 13 := by
  sorry

end twelfth_equation_l597_597187


namespace solution_l597_597457

theorem solution (x : ℝ) (h : 10 ^ (Real.log10 9) = 8 * x + 5) : x = 1 / 2 := sorry

end solution_l597_597457


namespace expression_evaluation_l597_597835

-- Define the conditions as Lean definitions
def sin_60_deg : Real := Real.sin (Real.pi / 3) -- sin(60 degrees) in radians
def reciprocal_half : Real := (1 / 2)⁻¹
def sqrt_12 : Real := Real.sqrt 12
def abs_neg_3 : Real := Real.abs (-3)

-- State the proof goal
theorem expression_evaluation :
  4 * sin_60_deg + reciprocal_half - sqrt_12 + abs_neg_3 = 5 := by
  -- Define the specific known values for the conditions
  have sin60_simplified : sin_60_deg = (Real.sqrt 3) / 2 := by sorry
  have reciprocal_half_simplified : reciprocal_half = 2 := by sorry
  have sqrt12_simplified : sqrt_12 = 2 * Real.sqrt 3 := by sorry
  have abs_neg3_simplified : abs_neg_3 = 3 := by sorry
  -- Substitute and simplify
  sorry

end expression_evaluation_l597_597835


namespace solve_for_s_l597_597564

theorem solve_for_s : ∃ s, (∃ x, 4 * x^2 - 8 * x - 320 = 0) ∧ s = 81 :=
by {
  -- Sorry is used to skip the actual proof.
  sorry
}

end solve_for_s_l597_597564


namespace determine_m_l597_597934

theorem determine_m :
  ∃ m : ℚ, 
    let P : ℕ → ℚ := λ k, m * (2 / 3) ^ k in 
      (P 1 + P 2 + P 3 = 1) ∧ (m = 27 / 38) :=
begin
  use 27 / 38,
  simp only [pow_one, pow_two],
  let P := λ k, (27 / 38) * (2 / 3) ^ k,
  split,
  { 
    calc P 1 + P 2 + P 3
        = (27 / 38) * (2 / 3) + (27 / 38) * (4 / 9) + (27 / 38) * (8 / 27) : by simp [P]
    ... = (27 / 38) * ((2 / 3) + (4 / 9) + (8 / 27)) : by ring
    ... = (27 / 38) * (38 / 27) : by norm_num
    ... = 1 : by norm_num
  },
  {
    refl,
  }
end

end determine_m_l597_597934


namespace ways_to_divide_60480_l597_597791

def number_of_divisors (n : ℕ) : ℕ :=
  (n.divisors.filter (λ x, x > 0)).card

theorem ways_to_divide_60480 :
  let T := 60480 in number_of_divisors T = 96 :=
by
  let T := 60480
  sorry

end ways_to_divide_60480_l597_597791


namespace james_total_money_l597_597138

theorem james_total_money :
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  total_money = 135 := by
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  exact 135

end james_total_money_l597_597138


namespace number_of_students_in_third_row_l597_597326

/-
After the teacher moved Vovochka from the first row to the second,
Vanechka from the second row to the third, and Masha from the third row to the first,
the average age in the first row increased by one week,
the average age in the second row increased by two weeks,
the average age in the third row decreased by four weeks,
and there are 12 students each in the first and second rows.
Prove that the number of students in the third row is 9.
-/

theorem number_of_students_in_third_row
  (students_first_row : ℕ)
  (students_second_row : ℕ)
  (average_increase_first_row : ℤ)
  (average_increase_second_row : ℤ)
  (average_decrease_third_row : ℤ)
  (students_third_row : ℕ) :
  students_first_row = 12 →
  students_second_row = 12 →
  average_increase_first_row = 1 →
  average_increase_second_row = 2 →
  average_decrease_third_row = -4 →
  12 + (12 * 2) + (students_third_row * (-4)) = 0 →
  students_third_row = 9 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  have : 36 - 4 * students_third_row = 0 := h6
  linarith

end number_of_students_in_third_row_l597_597326


namespace imag_part_of_z_l597_597921

theorem imag_part_of_z : ∃ (z : ℂ), z = (1 + complex.I)^2 ∧ complex.im z = 2 :=
by
  have h : (1 + complex.I)^2 = 2 * complex.I :=
    by sorry
  use (1 + complex.I)^2
  split
  . exact rfl
  . rw h
    rw complex.im
    exact rfl

end imag_part_of_z_l597_597921


namespace geometric_sequence_sum_l597_597401

theorem geometric_sequence_sum (a : ℕ → ℝ) (S₄ : ℝ) (S₈ : ℝ) (r : ℝ) 
    (h1 : r = 2) 
    (h2 : S₄ = a 0 + a 0 * r + a 0 * r^2 + a 0 * r^3)
    (h3 : S₄ = 1) 
    (h4 : S₈ = a 0 + a 0 * r + a 0 * r^2 + a 0 * r^3 + a 0 * r^4 + a 0 * r^5 + a 0 * r^6 + a 0 * r^7) :
    S₈ = 17 := by
  sorry

end geometric_sequence_sum_l597_597401


namespace derivative_at_one_l597_597063

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem derivative_at_one : deriv f 1 = 2 :=
by sorry

end derivative_at_one_l597_597063


namespace sin_780_eq_sqrt3_div_2_l597_597354

theorem sin_780_eq_sqrt3_div_2 : Real.sin (780 * Real.pi / 180) = Math.sqrt 3 / 2 := by
  sorry

end sin_780_eq_sqrt3_div_2_l597_597354


namespace pizza_pasta_cost_difference_l597_597215

variable (x y z : ℝ)
variable (A1 : 2 * x + 3 * y + 4 * z = 53)
variable (A2 : 5 * x + 6 * y + 7 * z = 107)

theorem pizza_pasta_cost_difference :
  x - z = 1 :=
by
  sorry

end pizza_pasta_cost_difference_l597_597215


namespace walking_speed_is_6_kmh_l597_597786

-- Define the given conditions as def
def time_in_minutes : ℕ := 15
def length_in_meters : ℕ := 1500

-- Convert conditions to the appropriate units for the problem
def time_in_hours : ℝ := time_in_minutes / 60.0
def length_in_kilometers : ℝ := length_in_meters / 1000.0

-- Lean 4 theorem statement to prove the man's walking speed
theorem walking_speed_is_6_kmh (time_in_hours length_in_kilometers : ℝ) : time_in_hours = 0.25 → length_in_kilometers = 1.5 → (length_in_kilometers / time_in_hours) = 6 :=
by
  intros h_time h_length
  rw [h_time, h_length]
  norm_num
  sorry

end walking_speed_is_6_kmh_l597_597786


namespace triangle_area_example_l597_597323

structure Point where
  x : ℤ
  y : ℤ

def triangle_area (A B C : Point) : ℚ :=
  (1 / 2 : ℚ) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem triangle_area_example :
  let A := ⟨2, 3⟩
      B := ⟨2, -4⟩
      C := ⟨9, 3⟩
  triangle_area A B C = 24.5 := by
  sorry

end triangle_area_example_l597_597323


namespace find_a4_l597_597114

-- Define the sequence
noncomputable def a : ℕ → ℝ := sorry

-- Define the initial term a1 and common difference d
noncomputable def a1 : ℝ := sorry
noncomputable def d : ℝ := sorry

-- The conditions from the problem
def condition1 : Prop := a 2 + a 6 = 10 * Real.sqrt 3
def condition2 : Prop := a 3 + a 7 = 14 * Real.sqrt 3

-- Using the conditions to prove a4
theorem find_a4 (h1 : condition1) (h2 : condition2) : a 4 = 5 * Real.sqrt 3 :=
by
  sorry

end find_a4_l597_597114


namespace function_is_decreasing_on_R_l597_597229

def is_decreasing (a : ℝ) : Prop := a - 1 < 0

theorem function_is_decreasing_on_R (a : ℝ) : (1 < a ∧ a < 2) ↔ is_decreasing a :=
by
  sorry

end function_is_decreasing_on_R_l597_597229


namespace program_output_when_N_is_6_l597_597196

theorem program_output_when_N_is_6 : 
  ∃ S : ℕ, (∃ (N : ℕ) (I : ℕ), N = 6 ∧ I = 1 ∧ S = 1 ∧ 
  (∀ I, (I ≤ N) → (∃ S, S = (Nat.factorial N)))) ∧ S = 720 :=
begin
  sorry,
end

end program_output_when_N_is_6_l597_597196


namespace negation_of_p_range_of_m_if_p_false_l597_597935

open Real

noncomputable def neg_p (m : ℝ) : Prop :=
∀ x : ℝ, x^2 - m*x - m > 0

theorem negation_of_p (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m*x - m ≤ 0) ↔ neg_p m := 
by sorry

theorem range_of_m_if_p_false : 
  (∀ m : ℝ, neg_p m → (-4 < m ∧ m < 0)) :=
by sorry

end negation_of_p_range_of_m_if_p_false_l597_597935


namespace tangent_line_equation_at_1_l597_597644

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

noncomputable def f_prime (x : ℝ) : ℝ := 2*x - 1/(x^2)

theorem tangent_line_equation_at_1 : 
  let p := (1 : ℝ, f 1 : ℝ) in
  let m := f_prime 1 in
  m = 1 ∧ p = (1, 2) ∧ ∀ (x y : ℝ), y - 2 = m * (x - 1) ↔ x - y + 1 = 0 :=
by
  sorry

end tangent_line_equation_at_1_l597_597644


namespace part1_x_values_part2_abscissa_xP_part2_OM_ON_constant_l597_597403

variable {b m n : ℝ}

def quadratic_eq (x : ℝ) := x^2 + b * x + b - 1
def line_eq (m : ℝ) : ℝ → ℝ := λ x, m * x + m
def point_E := (4 : ℝ, 2 : ℝ)
def point_M := (0 : ℝ, m)
def point_N := (0 : ℝ, n)

-- 1. Prove the values for x when y = 0 for the given quadratic equation
theorem part1_x_values (b : ℝ) :
  (quadratic_eq (-1) = 0) ∧ (quadratic_eq (1 - b) = 0) := by
  sorry

-- 2①. Prove the abscissa x_P of point P when b < 2
theorem part2_abscissa_xP (m b : ℝ) (h1 : b < 2) :
  let x_P := m - b + 1 in
  quadratic_eq x_P = line_eq m x_P := by
  sorry

-- 2②. Prove OM * ON is a constant value when b = -3
theorem part2_OM_ON_constant (b : ℝ) (h1 : b = -3) :
  (quadratic_eq (-1) = 0) ∧ (quadratic_eq 4 = 0) →
  OM * ON = 2 := by
  let OM := point_M
  let ON := point_N
  sorry

end part1_x_values_part2_abscissa_xP_part2_OM_ON_constant_l597_597403


namespace pentagon_inequality_l597_597514

theorem pentagon_inequality 
  (a b c d : ℝ) 
  (h1 : ∀ (ABCDE : Set ℝ) , IsConvex ABCDE) 
  (h2 : CircleInscribed ABCDE 1) 
  (h3 : ∀ (AB BC CD DE AE : ℝ), AB = a ∧ BC = b ∧ CD = c ∧ DE = d ∧ AE = 2) :
  a^2 + b^2 + c^2 + d^2 + abc + bcd < 4 :=
sorry

end pentagon_inequality_l597_597514


namespace value_of_f_10_l597_597099

variable {x y : ℝ}
variable f : ℝ → ℝ
variable h1 : f x = 2 * x ^ 2 + y
variable h2 : f 2 = 30

-- The theorem to prove
theorem value_of_f_10 : f 10 = 222 :=
by
  -- proof goes here
  sorry

end value_of_f_10_l597_597099


namespace percentage_supports_policy_l597_597799

theorem percentage_supports_policy (men women : ℕ) (men_favor women_favor : ℝ) (total_population : ℕ) (total_supporters : ℕ) (percentage_supporters : ℝ)
  (h1 : men = 200) 
  (h2 : women = 800)
  (h3 : men_favor = 0.70)
  (h4 : women_favor = 0.75)
  (h5 : total_population = men + women)
  (h6 : total_supporters = (men_favor * men) + (women_favor * women))
  (h7 : percentage_supporters = (total_supporters / total_population) * 100) :
  percentage_supporters = 74 := 
by
  sorry

end percentage_supports_policy_l597_597799


namespace count_numbers_with_D_eq_3_l597_597846

def D (n : ℕ) : ℕ :=
  (n.bits.enumFrom 0).zip (n.bits.enumFrom 0).tail.filter (λ (pair : ℕ × ℕ), pair.fst ≠ pair.snd).length

def count_valid_numbers : ℕ :=
  (list.range (63 + 1)).filter (λ n, D n = 3).length

theorem count_numbers_with_D_eq_3 :
  count_valid_numbers = 18 := sorry

end count_numbers_with_D_eq_3_l597_597846


namespace frog_escape_probability_l597_597476

def P : ℕ → ℚ
def P_recurrence (N : ℕ) (h : 0 < N ∧ N < 10) : 
  P N = (N / 10) * P (N - 1) + (1 - (N / 10)) * P (N + 1) := 
sorry

def P_boundary_0 : 
  P 0 = 0 := 
sorry 

def P_boundary_10 : 
  P 10 = 1 := 
sorry 

theorem frog_escape_probability : 
  P 1 = 63 / 146 := 
sorry

end frog_escape_probability_l597_597476


namespace min_n_for_two_max_values_l597_597230

theorem min_n_for_two_max_values :
  ∃ n : ℕ, (∀ x ∈ (Icc (0 : ℝ) ↑n),  y = (sin (π * x / 3)) ∧ ∃ x1 x2, x1 ≠ x2 ∧ y x1 =  y x2 = 1) ∧ y 8 = 1  :=
  sorry

end min_n_for_two_max_values_l597_597230


namespace domain_of_sqrt_fn_l597_597643

theorem domain_of_sqrt_fn : {x : ℝ | -2 ≤ x ∧ x ≤ 2} = {x : ℝ | 4 - x^2 ≥ 0} := 
by sorry

end domain_of_sqrt_fn_l597_597643


namespace player1_points_after_13_rotations_l597_597699

theorem player1_points_after_13_rotations :
  ∃ (player1_points : ℕ), 
    (∀ (i : ℕ),  (i = 5 → player1_points = 72) ∧ (i = 9 → player1_points = 84)) → 
    player1_points = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597699


namespace find_t_l597_597022

theorem find_t (t : ℚ) (a b : ℚ × ℚ) 
  (ha : a = (3, -4)) 
  (hb : b = (2, t)) 
  (proj_ab : (3 * 2 + -4 * t) / real.sqrt (3^2 + (-4)^2) = -3) : 
  t = 21 / 4 :=
sorry

end find_t_l597_597022


namespace repeating_decimal_block_length_l597_597592

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l597_597592


namespace inverse_function_eval_l597_597663

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h : Function.Bijective g)
variable (table_data : ∀ x, x ∈ {2, 4, 6, 7, 9, 11} → g x = 7 ∨ g x = 2 ∨ g x = 9 ∨ g x = 6 ∨ g x = 11 ∨ g x = 4)

theorem inverse_function_eval :
  g_inv (g_inv 9 + g_inv 2) / g_inv 11 = 11 :=
sorry

end inverse_function_eval_l597_597663


namespace cosine_of_angle_between_planes_l597_597400

-- Define coordinates for the vertices of the cube
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def D : ℝ × ℝ × ℝ := (0, 1, 0)
def A1 : ℝ × ℝ × ℝ := (0, 0, 1)
def B1 : ℝ × ℝ × ℝ := (1, 0, 1)
def D1 : ℝ × ℝ × ℝ := (0, 1, 1)

-- Define vectors in the planes AB1D1 and A1BD
def AB1 : ℝ × ℝ × ℝ := (B1.1 - A.1, B1.2 - A.2, B1.3 - A.3)
def AD1 : ℝ × ℝ × ℝ := (D1.1 - A.1, D1.2 - A.2, D1.3 - A.3)
def A1B : ℝ × ℝ × ℝ := (B.1 - A1.1, B.2 - A1.2, B.3 - A1.3)
def A1D : ℝ × ℝ × ℝ := (D.1 - A1.1, D.2 - A1.2, D.3 - A1.3)

-- Define cross products to find normal vectors
def normal1 : ℝ × ℝ × ℝ :=
  (AB1.2 * AD1.3 - AB1.3 * AD1.2, 
   AB1.3 * AD1.1 - AB1.1 * AD1.3, 
   AB1.1 * AD1.2 - AB1.2 * AD1.1)

def normal2 : ℝ × ℝ × ℝ :=
  (A1B.2 * A1D.3 - A1B.3 * A1D.2, 
   A1B.3 * A1D.1 - A1B.1 * A1D.3, 
   A1B.1 * A1D.2 - A1B.2 * A1D.1)

-- Function to calculate dot product of two 3D vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Function to calculate magnitude of a 3D vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

-- Main theorem to prove the cosine of the angle between the planes
theorem cosine_of_angle_between_planes : 
  dot_product normal1 normal2 / (magnitude normal1 * magnitude normal2) = 1 / 3 :=
by
  sorry

end cosine_of_angle_between_planes_l597_597400


namespace area_of_square_perimeter_40_l597_597824

noncomputable def side_length (P : ℝ) : ℝ := P / 4
noncomputable def area (s : ℝ) : ℝ := s * s

theorem area_of_square_perimeter_40 (P : ℝ) (hP : P = 40) : area (side_length P) = 100 :=
by
  -- Definitions for side length and area based on the provided perimeter
  let s := side_length P
  have hs : s = 10 := by
    rw [side_length, hP]
    norm_num

  show area s = 100
  simp [area, hs]
  norm_num

-- skip the proof
sorry

end area_of_square_perimeter_40_l597_597824


namespace sum_of_lengths_eq_sum_of_areas_eq_alpha_value_eq_l597_597018

-- Define the conditions
variables (A B1 B2 B3 : Point) (α a : ℝ)

-- Define the proof problem
theorem sum_of_lengths_eq :
  ∀ (n : ℕ), (finset.sum (finset.range n) (λ i, a * (cos α)^(i - 1) * sin α)) = a * cot (α / 2) := sorry

theorem sum_of_areas_eq :
  ∑' (n : ℕ), (1 / 2) * a^2 * (cos α)^((2 * n) + 1) * sin α = (1 / 2) * a^2 * cot α := sorry

theorem alpha_value_eq :
  (1 / 2) * a^2 * cot (α) = a^2 → α = Real.arccot 2 := sorry

end sum_of_lengths_eq_sum_of_areas_eq_alpha_value_eq_l597_597018


namespace total_journey_time_correct_l597_597474

def boat_speed_in_still_water : ℝ := 6 -- km/hr
def distance_upstream : ℝ := 64 -- km
def distance_downstream : ℝ := 64 -- km
def stop_time : ℝ := 0.5 -- hours per stop

-- Define conditions for different sections
def first_section_distance : ℝ := 20 -- km
def first_section_current_speed : ℝ := 2 -- km/hr
def first_section_cross_current : ℝ := 0.5 -- km/hr

def second_section_distance : ℝ := 24 -- km
def second_section_current_speed : ℝ := 3 -- km/hr
def second_section_cross_current : ℝ := 1 -- km/hr

def third_section_distance : ℝ := 20 -- km
def third_section_current_speed : ℝ := 1.5 -- km/hr
def third_section_cross_current : ℝ := 0 -- km/hr

-- Define number of stops
def upstream_stops : ℕ := 3
def downstream_stops : ℕ := 4

-- Define the effective speeds and travel times
def effective_speed_upstream (boat_speed current_speed : ℝ) : ℝ :=
  boat_speed - current_speed

def effective_speed_downstream (boat_speed current_speed : ℝ) : ℝ :=
  boat_speed + current_speed

def travel_time (distance speed : ℝ) : ℝ :=
  distance / speed

-- Define the total journey time
noncomputable def total_journey_time : ℝ :=
  let upstream_time := (travel_time first_section_distance (effective_speed_upstream boat_speed_in_still_water first_section_current_speed)) +
                      (travel_time second_section_distance (effective_speed_upstream boat_speed_in_still_water second_section_current_speed)) +
                      (travel_time third_section_distance (effective_speed_upstream boat_speed_in_still_water third_section_current_speed))
  let downstream_time := (travel_time first_section_distance (effective_speed_downstream boat_speed_in_still_water first_section_current_speed)) +
                        (travel_time second_section_distance (effective_speed_downstream boat_speed_in_still_water second_section_current_speed)) +
                        (travel_time third_section_distance (effective_speed_downstream boat_speed_in_still_water third_section_current_speed))
  let stops_time := (upstream_stops * stop_time) + (downstream_stops * stop_time)
  upstream_time + downstream_time + stops_time

theorem total_journey_time_correct :
  total_journey_time = 28.78 :=
sorry

end total_journey_time_correct_l597_597474


namespace cube_root_of_division_as_fraction_l597_597860

theorem cube_root_of_division_as_fraction : 
  ∃ (x : ℚ), x = real.cbrt (4 / (27 / 2)) ∧ x = 2 / 3 := 
by
  use 2 / 3
  split
  sorry  -- Proof to show that real.cbrt(4 / (27 / 2)) = 2 / 3

end cube_root_of_division_as_fraction_l597_597860


namespace expected_value_of_winnings_l597_597780

/-- A fair 6-sided die is rolled. If the roll is even, then you win the amount of dollars 
equal to the square of the number you roll. If the roll is odd, you win nothing. 
Prove that the expected value of your winnings is 28/3 dollars. -/
theorem expected_value_of_winnings : 
  (1 / 6) * (2^2 + 4^2 + 6^2) = 28 / 3 := by
sorry

end expected_value_of_winnings_l597_597780


namespace speed_goods_train_l597_597306

def length_train : ℝ := 50
def length_platform : ℝ := 250
def time_crossing : ℝ := 15

/-- The speed of the goods train in km/hr given the length of the train, the length of the platform, and the time to cross the platform. -/
theorem speed_goods_train :
  (length_train + length_platform) / time_crossing * 3.6 = 72 :=
by
  sorry

end speed_goods_train_l597_597306


namespace max_min_difference_l597_597025

noncomputable def a : ℝ := - (2023 + Real.pi) ^ 0
noncomputable def b : ℝ := (-10) ^ (-1)
noncomputable def c : ℝ := (-1 / 3) ^ 2
noncomputable def d : ℝ := (1 / 2) ^ (-3)

theorem max_min_difference :
  let max_val := max (max a b) (max c d)
  let min_val := min (min a b) (min c d)
  max_val - min_val = 9 :=
by
  sorry

end max_min_difference_l597_597025


namespace conjugate_coordinate_l597_597985

theorem conjugate_coordinate {z : ℂ} (hz : z = 10 * complex.I / (3 + complex.I)) :
  complex.conj z = 1 - 3 * complex.I :=
by 
  simp at hz
  rw hz
  simp
  sorry

end conjugate_coordinate_l597_597985


namespace repeating_block_length_of_7_div_13_is_6_l597_597607

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l597_597607


namespace sin_780_eq_sqrt3_over_2_l597_597352

theorem sin_780_eq_sqrt3_over_2 :
  sin (780 : ℝ) = (Real.sqrt 3 / 2) :=
by
  sorry

end sin_780_eq_sqrt3_over_2_l597_597352


namespace triangle_median_identity_l597_597392

-- Define the triangle with medians intersecting at point O and the distances A, B, C to O
variable (A B C O : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
variable [HasDist A ℝ] [HasDist B ℝ] [HasDist C ℝ] [HasDist O ℝ]
variable (mediansIntersectAtCentroid : Medians A B C O)

-- Prove the equality
theorem triangle_median_identity (A B C O : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
[HasDist A ℝ] [HasDist B ℝ] [HasDist C ℝ] [HasDist O ℝ]
(mediansIntersectAtCentroid : Medians A B C O) :
  dist2 A B + dist2 B C + dist2 C A = 3 * (dist2 O A + dist2 O B + dist2 O C) :=
sorry

end triangle_median_identity_l597_597392


namespace school_activity_problem_l597_597654

-- Definitions required for the first part of the problem
def teachers_students_count (x y : Nat) : Prop :=
  (38 * x + 6 = y) ∧ (40 * x - 6 = y)

-- Definition to specify the total number of people
def total_people (x y : Nat) : Prop :=
  x + y = 240

-- Define the rental cost condition
def rental_cost_condition : Prop :=
  (∀ x y : Nat, 4 * 400 + 2 * 280 = 2160 ∨ 5 * 400 + 1 * 280 = 2160) ∧ (120 * x + 1680 ≤ 2300)

-- The main theorem combining all problem conditions and the correct answers.
theorem school_activity_problem : 
  ∃ (x y : Nat), 
    (teachers_students_count x y) → 
    (total_people x y) → 
    (rental_cost_condition)
:=
by {
  use (6, 234),
  have h1 : teachers_students_count 6 234 := by {
    split,
    { calc 38 * 6 + 6 = 234 : by norm_num },
    { calc 40 * 6 - 6 = 234 : by norm_num }
  },
  have h2 : total_people 6 234 := by {
    calc 6 + 234 = 240 : by norm_num
  },
  have h3 : rental_cost_condition := by {
    split,
    { intros x y,
      left,
      calc 4 * 400 + 2 * 280 = 2160 : by norm_num,
      right,
      calc 5 * 400 + 1 * 280 = 2160 : by norm_num
    },
    calc 120 * 6 + 1680 ≤ 2300 : by norm_num
  },
  exact ⟨h1, h2, h3⟩
}

end school_activity_problem_l597_597654


namespace combined_sale_price_correct_l597_597531

def price_after_discount (price : ℝ) (discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate)

def final_price (initial_price discount1 discount2 : ℝ) : ℝ :=
  price_after_discount (price_after_discount initial_price discount1) discount2

def total_price (tv_regular_price tv_discount1 tv_discount2 sound_system_price sound_system_discount tax_rate : ℝ) : ℝ :=
  let tv_sale_price := final_price tv_regular_price tv_discount1 tv_discount2
  let sound_system_sale_price := price_after_discount sound_system_price sound_system_discount
  let pre_tax_total := tv_sale_price + sound_system_sale_price
  pre_tax_total * (1 + tax_rate)

theorem combined_sale_price_correct :
  total_price 600 0.10 0.15 400 0.20 0.08 = 841.32 :=
by
  sorry

end combined_sale_price_correct_l597_597531


namespace find_last_two_digits_l597_597720

variables {z a r m l : ℕ}
variables (ZARAZA ALMAZ : ℕ)
variables (digits : char → ℕ)

-- Each character represents a unique digit
axiom zaraza_unique_digits : function.injective digits
axiom almakza_unique_digits : function.injective digits

-- The numbers
def ZARAZA := 100000 * digits 'z' + 10000 * digits 'a' + 1000 * digits 'r' + 100 * digits 'a' + 10 * digits 'z' + digits 'a'
def ALMAZ := 10000 * digits 'a' + 1000 * digits 'l' + 100 * digits 'm' + 10 * digits 'a' + digits 'z'

-- Divisibility constraints
axiom zaraza_div_by_4 : ZARAZA % 4 = 0
axiom almaz_div_by_28 : ALMAZ % 28 = 0

-- Proof Goal
theorem find_last_two_digits :
  (ZARAZA + ALMAZ) % 100 = 32 := 
sorry

end find_last_two_digits_l597_597720


namespace tangency_points_coplanar_l597_597513

open Set

variables {A B C D R S T U O : Point}

-- Definitions of the points and sphere tangency conditions
def distinct_points (A B C D : Point) : Prop :=
  ¬ ∃ P : Plane, A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ D ∈ P

def tangent_to_sphere (O : Point) (sphere_radius : ℝ) (tangent_point : Point) (point_1 point_2 : Point) : Prop :=
  dist O tangent_point = sphere_radius ∧
  (tangent_point - (point_1 - point_2) * (dist tangent_point point_1 + dist tangent_point point_2) / 
  (dist point_1 point_2 + dist point_2 point_1)) = tangent_point

def coplanar_tangency_points (R S T U : Point) : Prop :=
  ∃ plane : Plane, R ∈ plane ∧ S ∈ plane ∧ T ∈ plane ∧ U ∈ plane

theorem tangency_points_coplanar 
  (A B C D : Point) 
  (h : distinct_points A B C D) 
  (O : Point) 
  (sphere_radius : ℝ) 
  (R S T U : Point)
  (tangent_r : tangent_to_sphere O sphere_radius R A B)
  (tangent_s : tangent_to_sphere O sphere_radius S B C)
  (tangent_t : tangent_to_sphere O sphere_radius T C D)
  (tangent_u : tangent_to_sphere O sphere_radius U D A) :
  coplanar_tangency_points R S T U :=
by sorry

end tangency_points_coplanar_l597_597513


namespace yella_computer_usage_difference_l597_597276

-- Define the given conditions
def last_week_usage : ℕ := 91
def this_week_daily_usage : ℕ := 8
def days_in_week : ℕ := 7

-- Compute this week's total usage
def this_week_total_usage := this_week_daily_usage * days_in_week

-- Statement to prove
theorem yella_computer_usage_difference :
  last_week_usage - this_week_total_usage = 35 := 
by
  -- The proof will be filled in here
  sorry

end yella_computer_usage_difference_l597_597276


namespace area_of_right_triangle_is_54_l597_597870

noncomputable def right_triangle_area (hypotenuse : ℝ) (side1 : ℝ) (area : ℝ) : Prop :=
∃ (side2 : ℝ), (side1^2 + side2^2 = hypotenuse^2) ∧ (area = 1 / 2 * side1 * side2)

theorem area_of_right_triangle_is_54 :
  right_triangle_area 15 12 54 :=
begin
  sorry
end

end area_of_right_triangle_is_54_l597_597870


namespace original_strength_of_class_l597_597217

-- Define the conditions
variables {x : ℕ} -- original strength of the class
variables (total_original_age : ℕ) (total_new_students_age : ℕ) (total_new_class_age : ℕ)

def average_age_before := 40
def number_of_new_students := 15
def average_age_new_students := 32
def new_average_age := 36

-- Define the equations based on the conditions
def initial_total_age_eq := total_original_age = average_age_before * x
def new_students_total_age_eq := total_new_students_age = number_of_new_students * average_age_new_students
def new_class_total_age_eq := total_new_class_age = new_average_age * (x + number_of_new_students)

def total_age_consistency := total_original_age + total_new_students_age = total_new_class_age

-- Lean statement to prove the final conclusion
theorem original_strength_of_class : 
  ∀ (x : ℕ) 
    (total_original_age total_new_students_age total_new_class_age : ℕ),
  initial_total_age_eq x total_original_age →
  new_students_total_age_eq total_new_students_age →
  new_class_total_age_eq total_new_class_age →
  total_age_consistency total_original_age total_new_students_age total_new_class_age →
  x = 15 :=
by
  intros,
  sorry

end original_strength_of_class_l597_597217


namespace player1_points_after_13_rotations_l597_597702

theorem player1_points_after_13_rotations :
  ∃ (player1_points : ℕ), 
    (∀ (i : ℕ),  (i = 5 → player1_points = 72) ∧ (i = 9 → player1_points = 84)) → 
    player1_points = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597702


namespace smallest_positive_period_and_intervals_range_of_f_on_interval_l597_597066

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * sin (x + (π / 6))

-- Proof problem (1): Proving the period and monotonically increasing intervals
theorem smallest_positive_period_and_intervals (k : ℤ) :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ x : ℝ, x ∈ set.Icc (k * π - π / 12) (k * π + 5 * π / 12) → 
   ∀ y : ℝ, y ∈ set.Icc (k * π - π / 12) (k * π + 5 * π / 12) → x ≤ y → f x ≤ f y) := 
  sorry

-- Proof problem (2): Proving the range of f(x) when x is in [0, π/2]
theorem range_of_f_on_interval :
  set.Icc (0 : ℝ) (1 + sqrt 3 / 2) = 
  (set.image f (set.Icc 0 (π / 2))) :=
  sorry

end smallest_positive_period_and_intervals_range_of_f_on_interval_l597_597066


namespace M_salary_percentage_l597_597718

noncomputable def P : ℝ :=
  let total_salary := 605
  let n_salary := 275
  let m_salary (P : ℝ) := (P / 100) * n_salary
  have h1 : m_salary P + n_salary = total_salary := by sorry
  (330 / 275) * 100

theorem M_salary_percentage :
  let total_salary := 605
  let n_salary := 275
  let percentage := P
  let m_salary (P : ℝ) := (P / 100) * n_salary
  m_salary percentage + n_salary = total_salary →
  percentage = 120 :=
by
  intro h
  rw [<-h at ⊢]
  sorry

end M_salary_percentage_l597_597718


namespace sufficient_but_not_necessary_condition_l597_597077

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 5) :=
by
  sorry

end sufficient_but_not_necessary_condition_l597_597077


namespace calculate_rate_of_interest_l597_597800

theorem calculate_rate_of_interest:
  let SI := 4016.25 in
  let P := 6178.846153846154 in
  let T := 5 in
  let R := (SI * 100) / (P * T) in
  R = 13 := by sorry

end calculate_rate_of_interest_l597_597800


namespace repeating_block_length_7_div_13_l597_597626

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l597_597626


namespace repeating_block_length_7_div_13_l597_597627

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l597_597627


namespace centroid_to_hypotenuse_distance_l597_597109

theorem centroid_to_hypotenuse_distance (S : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
    distance_centroid_hypotenuse S α = (1 / 3) * real.sqrt (S * real.sin (2 * α)) :=
sorry

end centroid_to_hypotenuse_distance_l597_597109


namespace trig_expr_value_l597_597041

theorem trig_expr_value (α : ℝ) (P : ℝ × ℝ) (hP : P = (-4, 3)) :
  (cos (α - π / 2) / sin (5 * π / 2 + α) * sin (α - 2 * π) * cos (2 * π - α) = 9 / 25) :=
by
  -- proof steps would go here, skipped
  sorry

end trig_expr_value_l597_597041


namespace tangents_sum_l597_597462

open Real

theorem tangents_sum
    (x y : ℝ)
    (h1 : tan x + tan y = 18)
    (h2 : cot x + cot y = 24) :
    tan (x + y) = 72 := sorry

end tangents_sum_l597_597462


namespace points_player_1_after_13_rotations_l597_597682

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end points_player_1_after_13_rotations_l597_597682


namespace train_pass_time_l597_597321

-- Definitions based on conditions from part (a)
def length_of_train : ℝ := 110  -- in meters
def speed_of_train_kmh : ℝ := 80  -- in km/h
def speed_of_man_kmh : ℝ := 8  -- in km/h
def speed_of_train_ms : ℝ := speed_of_train_kmh * 1000 / 3600  -- converting to m/s
def speed_of_man_ms : ℝ := speed_of_man_kmh * 1000 / 3600  -- converting to m/s
def relative_speed : ℝ := speed_of_train_ms + speed_of_man_ms  -- relative speed in m/s

-- Theorem based on part (c) to be proven
theorem train_pass_time : length_of_train / relative_speed = 4.5 := by
  sorry

end train_pass_time_l597_597321


namespace always_positive_expression_l597_597919

variable (x a b : ℝ)

theorem always_positive_expression (h : ∀ x, (x - a)^2 + b > 0) : b > 0 :=
sorry

end always_positive_expression_l597_597919


namespace best_fit_model_l597_597975

theorem best_fit_model 
  (R2_model1 R2_model2 R2_model3 R2_model4 : ℝ)
  (h1 : R2_model1 = 0.976)
  (h2 : R2_model2 = 0.776)
  (h3 : R2_model3 = 0.076)
  (h4 : R2_model4 = 0.351) : 
  (R2_model1 > R2_model2) ∧ (R2_model1 > R2_model3) ∧ (R2_model1 > R2_model4) :=
by
  sorry

end best_fit_model_l597_597975


namespace inner_cone_volume_l597_597261

theorem inner_cone_volume {R α : ℝ} 
  (hα_pos : 0 < α) 
  (hα_le_pi : α ≤ π / 2) 
  (hR_pos : 0 < R)
  (S1 : ℝ) (S2 : ℝ)
  (hS1 : S1 = π * R^2 * (1 + cos (π / 2 - α) / sin α))
  (hS2 : S2 = π * (R * cos (π / 4 - α / 2)) ^ 2 / sin α)
  (hS2_halfS1 : S2 = (1 / 2) * S1):
  let r := R * cos (π / 4 - α / 2),
      h := r * cot α,
      V := (1 / 3) * π * r^2 * h in
  V = (1 / 3) * π * R^3 * cos^3 (π / 4 - α / 2) * cot α :=
by
  intro r h V
  sorry

end inner_cone_volume_l597_597261


namespace marco_paint_fraction_l597_597463

theorem marco_paint_fraction (W : ℝ) (M : ℝ) (minutes_paint : ℝ) (fraction_paint : ℝ) :
  M = 60 ∧ W = 1 ∧ minutes_paint = 12 ∧ fraction_paint = 1/5 → 
  (minutes_paint / M) * W = fraction_paint := 
by
  sorry

end marco_paint_fraction_l597_597463


namespace range_of_a_l597_597912

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + 4^x * a > 0) ↔ a > -3/4 := 
sorry

end range_of_a_l597_597912


namespace james_total_money_l597_597130

theorem james_total_money (bills : ℕ) (value_per_bill : ℕ) (initial_money : ℕ) : 
  bills = 3 → value_per_bill = 20 → initial_money = 75 → initial_money + (bills * value_per_bill) = 135 :=
by
  intros hb hv hi
  rw [hb, hv, hi]
  -- Algebraic simplification
  sorry

end james_total_money_l597_597130


namespace minimum_value_of_quadratic_expression_l597_597395

theorem minimum_value_of_quadratic_expression (x y z : ℝ)
  (h : x + y + z = 2) : 
  x^2 + 2 * y^2 + z^2 ≥ 4 / 3 :=
sorry

end minimum_value_of_quadratic_expression_l597_597395


namespace triangle_side_length_difference_l597_597802

theorem triangle_side_length_difference :
  (∃ x : ℤ, 3 ≤ x ∧ x ≤ 17 ∧ ∀ a b c : ℤ, x + 8 > 10 ∧ x + 10 > 8 ∧ 8 + 10 > x) →
  (17 - 3 = 14) :=
by
  intros
  sorry

end triangle_side_length_difference_l597_597802


namespace minimize_total_cost_l597_597709

-- Define the conditions and the problem
def water_tank_volume : ℝ := 4800
def water_tank_depth : ℝ := 3
def bottom_cost_per_sqm : ℝ := 150
def wall_cost_per_sqm : ℝ := 120

-- Calculate the area of the bottom of the tank
def bottom_area : ℝ := water_tank_volume / water_tank_depth

-- Total cost function
def total_cost (x : ℝ) : ℝ :=
  let bottom_cost := bottom_cost_per_sqm * bottom_area
  let wall_cost := wall_cost_per_sqm * 2 * water_tank_depth * (x + bottom_area / x)
  bottom_cost + wall_cost

-- Proof problem statement
theorem minimize_total_cost : ∃ x : ℝ, x > 0 ∧ total_cost x = 297600 :=
by
  sorry

end minimize_total_cost_l597_597709


namespace flag_ratio_proof_l597_597646

def flag_ratio_problem (k y x : ℝ) (h_flag_ratio : 3 * k = 5 * y) (h_area_ratio : 15 * k^2 = 4 * x * y) : Prop :=
  y / x = 4 / 15

theorem flag_ratio_proof (k y x : ℝ) (h_flag_ratio : 3 * k = 5 * y) (h_area_ratio : 15 * k^2 = 4 * x * y) : 
  flag_ratio_problem k y x h_flag_ratio h_area_ratio :=
begin
  sorry
end

end flag_ratio_proof_l597_597646


namespace base_five_product_l597_597268

def base_five_to_decimal (x : List ℕ) : ℕ :=
  x.foldr (λ (b acc : ℕ) => 5 * acc + b) 0

def decimal_to_base_five (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else List.unfoldr (λ m => if m = 0 then none else some (m % 5, m / 5)) n

theorem base_five_product (x y : List ℕ) :
  x = [2, 0, 3] → y = [1, 4] → decimal_to_base_five (base_five_to_decimal x * base_five_to_decimal y) = [3, 4, 0, 2] :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end base_five_product_l597_597268


namespace find_f_of_condition_l597_597889

variable (f : ℝ → ℝ)

theorem find_f_of_condition (h : ∀ x, f(x + 2) = x^2 - 4x) : ∀ x, f(x) = x^2 - 8x + 12 :=
by {
  sorry
}

end find_f_of_condition_l597_597889


namespace find_combination_l597_597322

noncomputable def is_valid_combination (L E K A : ℕ) : Prop :=
  L < 9 ∧ E < 9 ∧ K < 9 ∧ A < 9 ∧ -- ensure the digits are in base 9
  L ≠ E ∧ L ≠ K ∧ L ≠ A ∧ K ≠ E ∧ K ≠ A ∧ E ≠ A ∧ -- no repeated digits
  let SUM1 := L + 9 * (A + 9 * (K + 9 * E)),
      SUM2 := K + 9 * (A + 9 * (L + 9 * E)),
      SUM3 := L + 9 * (E + 9 * (A + 9 * K)),
      SUM := K + 9 * (L + 9 * (A + 9 * E))
  in (SUM1 + SUM2 + SUM3) % 9 = SUM % 9 -- representing the modulo 9 addition ensuring valid cryptarithm solution.

theorem find_combination :
  ∃ (L E K A : ℕ), is_valid_combination L E K A ∧ (L, E, K, A) = (0, 8, 4, 3) :=
sorry

end find_combination_l597_597322


namespace player_1_points_after_13_rotations_l597_597675

-- Add necessary definitions and state the problem in Lean
def sectors : Fin 16 → ℕ
| ⟨0, _⟩ := 0
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 2
| ⟨3, _⟩ := 3
| ⟨4, _⟩ := 4
| ⟨5, _⟩ := 5
| ⟨6, _⟩ := 6
| ⟨7, _⟩ := 7
| ⟨8, _⟩ := 8
| ⟨9, _⟩ := 7
| ⟨10, _⟩ := 6
| ⟨11, _⟩ := 5
| ⟨12, _⟩ := 4
| ⟨13, _⟩ := 3
| ⟨14, _⟩ := 2
| ⟨15, _⟩ := 1

def points_earned (player_offset : Fin 16) (rotations : ℕ) : ℕ :=
List.sum (List.map sectors
  (List.map (λ n => (Fin.add (Fin.ofNat n) player_offset)) (List.range rotations)))

theorem player_1_points_after_13_rotations 
  (p5_points : points_earned ⟨5, by decide⟩ 13 = 72)
  (p9_points : points_earned ⟨9, by decide⟩ 13 = 84) :
  points_earned ⟨1, by decide⟩ 13 = 20 := 
sorry

end player_1_points_after_13_rotations_l597_597675


namespace quadratic_has_equal_roots_l597_597012

theorem quadratic_has_equal_roots :
  ∀ m : ℝ, (∀ x : ℝ, x^2 + 6 * x + m = 0 → x = -3) ↔ m = 9 := 
by
  intro m
  constructor
  {
    intro h
    have : (6:ℝ) ^ 2 - 4 * 1 * m = 0,
      from by simp [(pow_two 6), h.eq_c],
    simp [six_pow_two, neg_eq_zero] at this,
    linarith
  }
  {
    intro h
    simp [h],
    exact fun x _ => rfl
  }

end quadratic_has_equal_roots_l597_597012


namespace valid_codes_count_correct_l597_597856

def valid_codes_count : ℕ :=
  let digits := {0, 1, 2, 3, 4}
  let pairs := [(1, 2), (2, 4)]
  let remaining_digits (a b : ℕ) := (digits.erase a).erase b
  let num_permutations (s : Finset ℕ) := s.toFinset.perm.card
  pairs.length * num_permutations (remaining_digits 1 2)

theorem valid_codes_count_correct : valid_codes_count = 12 :=
by
  have h_perm : ∀ (a b : ℕ), 
  let rem_digits := (digits.erase a).erase b in
  rem_digits.toFinset.perm.card = 6 := sorry
  simp [valid_codes_count, remaining_digits, pairs, h_perm]
  done

end valid_codes_count_correct_l597_597856


namespace fans_stayed_until_end_l597_597250

-- Define the initial conditions
def total_seats : ℕ := 60000
def percentage_sold : ℚ := 0.75
def fans_stayed_home : ℕ := 5000
def percentage_left_early : ℚ := 0.10

-- Translate the problem into a proof statement
theorem fans_stayed_until_end : 
  let seats_sold := total_seats * percentage_sold.toNat in
  let fans_attended := seats_sold - fans_stayed_home in
  let fans_left_early := fans_attended * percentage_left_early.toNat in
  fans_attended - fans_left_early = 36000 :=
by 
  sorry

end fans_stayed_until_end_l597_597250


namespace david_work_days_l597_597764

theorem david_work_days
  (D A : ℝ)
  (h1 : D + A = 1 / 12)
  (h2 : 8 * (D + A) = 2 / 3)
  (h3 : D * 8 = 1 / 3) :
  D = 1 / 24 :=
by
  rw [← h1] at h2
  rw [← h3] at h3
  sorry 

end david_work_days_l597_597764


namespace inequality_solution_l597_597865

theorem inequality_solution (a : ℝ) :
  (4:ℝ)^(x^2) + 2 * (2*a + 1) * (2:ℝ)^(x^2) + 4*a^2 - 3 > 0 ↔
  a < (-1:ℝ) ∨ a ≥ (real.sqrt 3 / 2) :=
sorry

end inequality_solution_l597_597865


namespace unique_triple_solution_l597_597375

open Real

theorem unique_triple_solution
  (x y z : ℝ)
  (x_pos : 0 < x)
  (y_pos : 0 < y)
  (z_pos : 0 < z)
  (hx : 2 * x * sqrt (x + 1) - y * (y + 1) = 1)
  (hy : 2 * y * sqrt (y + 1) - z * (z + 1) = 1)
  (hz : 2 * z * sqrt (z + 1) - x * (x + 1) = 1) :
  (x, y, z) = (1 + sqrt 5) / 2 :=
sorry  -- The proof is not required as per the instructions.

end unique_triple_solution_l597_597375


namespace animal_enclosure_assignments_l597_597234

-- Definitions for the enclosures and conditions
def Enclosures : Type := Fin 5 -- There are 5 enclosures

inductive Animal
| giraffe | monkey | rhino | lion | seal

def enclosureSideCount : Enclosures → Nat
def poolInEnclosure : Enclosures → Bool

def borders : Enclosures → Enclosures → Bool
-- Definitions for the problem's conditions
@[simp] axiom giraffe_has_five_sides (e : Enclosures) : enclosureSideCount e = 5 ↔ e = 3
@[simp] axiom monkey_does_not_border (m r g : Enclosures) : 
  (m = 1 ∧ r = 5 ∧ g = 3) → ¬ (borders m r ∨ borders m g)
@[simp] axiom lion_same_sides_as_monkey (m l : Enclosures) : 
  (m = 1 ∧ l = 2) → enclosureSideCount m = enclosureSideCount l
@[simp] axiom pool_in_seal_enclosure (s : Enclosures) : poolInEnclosure s ↔ s = 4

-- The main proof statement to be proved

theorem animal_enclosure_assignments :
  (∃ (g m r l s : Enclosures),
    (g = 3 ∧ m = 1 ∧ r = 5 ∧ l = 2 ∧ s = 4) ∧ 
    giraffe_has_five_sides g ∧ 
    monkey_does_not_border m r g ∧ 
    lion_same_sides_as_monkey m l ∧
    pool_in_seal_enclosure s) :=
begin
  sorry
end

end animal_enclosure_assignments_l597_597234


namespace dandelion_dog_puffs_l597_597836

theorem dandelion_dog_puffs :
  let original_puffs := 40
  let mom_puffs := 3
  let sister_puffs := 3
  let grandmother_puffs := 5
  let friends := 3
  let puffs_per_friend := 9
  original_puffs - (mom_puffs + sister_puffs + grandmother_puffs + friends * puffs_per_friend) = 2 :=
by
  sorry

end dandelion_dog_puffs_l597_597836


namespace interval_length_l597_597869

theorem interval_length (x : ℝ) :
  (1/x > 1/2) ∧ (Real.sin x > 1/2) → (2 - Real.pi / 6 = 1.48) :=
by
  sorry

end interval_length_l597_597869


namespace books_remaining_correct_l597_597666

-- Define the total number of books and the number of books read
def total_books : ℕ := 32
def books_read : ℕ := 17

-- Define the number of books remaining to be read
def books_remaining : ℕ := total_books - books_read

-- Prove that the number of books remaining to be read is 15
theorem books_remaining_correct : books_remaining = 15 := by
  sorry

end books_remaining_correct_l597_597666


namespace gumballs_result_l597_597141

def gumballs_after_sharing_equally (initial_joanna : ℕ) (initial_jacques : ℕ) (multiplier : ℕ) : ℕ :=
  let joanna_total := initial_joanna + initial_joanna * multiplier
  let jacques_total := initial_jacques + initial_jacques * multiplier
  (joanna_total + jacques_total) / 2

theorem gumballs_result :
  gumballs_after_sharing_equally 40 60 4 = 250 :=
by
  sorry

end gumballs_result_l597_597141


namespace rook_traversal_possible_l597_597404

theorem rook_traversal_possible (m n : ℕ) (rook_moves : (ℕ × ℕ) → (ℕ × ℕ) → Prop) :
  (∀ p q, rook_moves p q → (p.1 = q.1 ∧ p.2 = q.2 + 1 ∨ p.1 = q.1 + 1 ∧ p.2 = q.2) ∨ 
                        (p.1 = q.1 ∧ p.2 = q.2 - 1 ∨ p.1 = q.1 - 1 ∧ p.2 = q.2)) →
  (∃ path : list (ℕ × ℕ), path.head = (0, 0) ∧ path.last = (0, 0) ∧
                           (∀ p ∈ path, ∃ q ∈ path, rook_moves p q) ∧ 
                           (path.nodup) ∧ (path.length = m * n)) ↔ 
  (even (m * n)) :=
sorry

end rook_traversal_possible_l597_597404


namespace min_value_a_l597_597023

-- Definitions for conditions from part a)
variable (a : ℝ) (x : ℝ)

-- The given conditions
axiom a_gt_one : a > 1
axiom x_in_domain : x ∈ Icc (1 / 3 : ℝ) (Real.Infty)

-- Additional inequality condition
axiom inequality_condition : ∀ x ∈ Icc (1 / 3 : ℝ) (Real.Infty), (1 / (3 * x) - 2 * x + Real.log (3 * x) ≤ 1 / (a * Real.exp (2 * x)) + Real.log a)

-- The proof problem to show: the minimum value of 'a' is 3 / (2 * Real.exp 1)
theorem min_value_a : a = 3 / (2 * Real.exp 1) :=
sorry

end min_value_a_l597_597023


namespace smallest_multiple_sum_l597_597508

theorem smallest_multiple_sum :
  let c := 10,
  let d := 105
  in c + d = 115 := 
by
  rfl

end smallest_multiple_sum_l597_597508


namespace U_mod_500_eq_375_l597_597157

-- Condition definitions
def remainders_mod_500 : Finset ℕ :=
  (Finset.range 500).filter (λ x, ∃ n : ℕ, x = (3^n % 500))

def U : ℕ :=
  remainders_mod_500.sum id

-- Proof statement
theorem U_mod_500_eq_375 : U % 500 = 375 := 
sorry

end U_mod_500_eq_375_l597_597157


namespace divisible_by_72_l597_597411

theorem divisible_by_72 (a b : ℕ) (h1 : 0 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10) :
  (b = 2 ∧ a = 3) → (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) % 72 = 0 :=
by
  sorry

end divisible_by_72_l597_597411


namespace chord_probability_concentric_circles_l597_597717

noncomputable def chord_intersects_inner_circle_probability : ℝ :=
  sorry

theorem chord_probability_concentric_circles :
  let r₁ := 2
  let r₂ := 3
  ∀ (P₁ P₂ : ℝ × ℝ),
    dist P₁ (0, 0) = r₂ ∧ dist P₂ (0, 0) = r₂ →
    chord_intersects_inner_circle_probability = 0.148 :=
  sorry

end chord_probability_concentric_circles_l597_597717


namespace derivative_at_2_l597_597929

def f (x : ℝ) : ℝ := x^2 - x

theorem derivative_at_2 : (derivative f 2) = 3 :=
by
  sorry

end derivative_at_2_l597_597929


namespace count_real_root_poly_eq_l597_597358

def hasRealRoots (b c d : ℤ) : Prop :=
  b^2 - 4 * (c + d) ≥ 0

def polynomialCoeffSet : Finset ℤ := {1, 2, 3, 4}

theorem count_real_root_poly_eq :
  {x | ∃ (b ∈ polynomialCoeffSet) (c ∈ polynomialCoeffSet) (d ∈ polynomialCoeffSet), hasRealRoots b c d}.card = 19 :=
sorry

end count_real_root_poly_eq_l597_597358


namespace player1_points_after_13_rotations_l597_597696

theorem player1_points_after_13_rotations :
  let sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
  let player_points (player : Nat) (rotations : Nat) :=
      rotations • (λ i, sector_points[(i + player) % 16])
  player_points 5 13 = 72 ∧ player_points 9 13 = 84 → player_points 1 13 = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597696


namespace coastal_village_population_l597_597246

variable (N : ℕ) (k : ℕ) (parts_for_males : ℕ) (total_males : ℕ)

theorem coastal_village_population 
  (h_total_population : N = 540)
  (h_division : k = 4)
  (h_parts_for_males : parts_for_males = 2)
  (h_total_males : total_males = (N / k) * parts_for_males) :
  total_males = 270 := 
by
  sorry

end coastal_village_population_l597_597246


namespace total_money_l597_597124

def JamesPocketBills : Nat := 3
def BillValue : Nat := 20
def WalletMoney : Nat := 75

theorem total_money (JamesPocketBills BillValue WalletMoney : Nat) : 
  (JamesPocketBills * BillValue + WalletMoney) = 135 :=
by
  sorry

end total_money_l597_597124


namespace trains_passing_time_correct_l597_597787

noncomputable def trains_passing_time 
  (v_m : ℝ) (v_g : ℝ) (L : ℝ) (km_to_m_per_s : ℝ) (conversion_factor : ℝ) : ℝ :=
  let v_r := v_m + v_g
  let v_r_m_s := v_r * km_to_m_per_s / conversion_factor
  L / v_r_m_s

theorem trains_passing_time_correct :
  trains_passing_time 64 20 420 1000 3600 ≈ 18 :=
by
  sorry

end trains_passing_time_correct_l597_597787


namespace intersection_complement_l597_597178

def M : set ℝ := {x | -2 < x ∧ x < 3}
def N : set ℝ := {x | 2^(x + 1) ≤ 1}
def complement_N : set ℝ := {x | ¬ (2^(x + 1) ≤ 1)}

theorem intersection_complement :
  M ∩ complement_N = {x | -1 < x ∧ x < 3} :=
by
  sorry

end intersection_complement_l597_597178


namespace cad_to_jpy_l597_597296

theorem cad_to_jpy (h : 2000 / 18 =  y / 5) : y = 556 := 
by 
  sorry

end cad_to_jpy_l597_597296


namespace repeating_block_length_7_div_13_l597_597625

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l597_597625


namespace AG_length_l597_597488

theorem AG_length {A B C D E G : Point} 
  (h_right_angle : is_right_angle A B C)
  (h_AB : dist A B = 3)
  (h_AC : dist A C = 3 * (sqrt 3))
  (h_intersects : intersects AD BE G) 
  : dist A G = (15 * (sqrt 3)) / 16 :=
sorry

end AG_length_l597_597488


namespace simplify_quotient_P_Q_l597_597515

theorem simplify_quotient_P_Q :
  let P : Polynomial ℝ := Polynomial.C 24 + Polynomial.X * (Polynomial.C (-14 + Polynomial.C (-13) * Polynomial.X + Polynomial.C 2 * Polynomial.X ^ 3 + Polynomial.X ^ 4))
  let r1 r2 r3 r4 : ℝ := 
  (P.natDegree = 4 ∧ 
  P.coeff 4 = 1 ∧ 
  P.coeff 3 = 2 ∧ 
  P.coeff 2 = -13 ∧ 
  P.coeff 1 = -14 ∧ 
  P.coeff 0 = 24 ∧
  (roots P).toFinset = {r1, r2, r3, r4})
  let Q : Polynomial ℝ := (X - Polynomial.C (r1 ^ 2)) * (X - Polynomial.C (r2 ^ 2)) * (X - Polynomial.C (r3 ^ 2)) * (X - Polynomial.C (r4 ^ 2))
  let R : Polynomial ℝ := (X - r1) * (X - r2) * (X - r3) * (X - r4)
  let S := Q.eval (X^2)
  (
    P ≠ 0 ∧ 
    Q.coeff 4 = 1 → 
  (R ≠ 0 → 
    S / R = Polynomial.C 24 + Polynomial.X * (Polynomial.C 14 + Polynomial.C (-13) * Polynomial.X + Polynomial.C (-2) * Polynomial.X ^ 3 + Polynomial.X ^ 4))
  sorry

end simplify_quotient_P_Q_l597_597515


namespace total_charge_for_first_4_minutes_under_plan_A_is_0_60_l597_597300

def planA_charges (X : ℝ) (minutes : ℕ) : ℝ :=
  if minutes <= 4 then X
  else X + (minutes - 4) * 0.06

def planB_charges (minutes : ℕ) : ℝ :=
  minutes * 0.08

theorem total_charge_for_first_4_minutes_under_plan_A_is_0_60
  (X : ℝ)
  (h : planA_charges X 18 = planB_charges 18) :
  X = 0.60 :=
by
  sorry

end total_charge_for_first_4_minutes_under_plan_A_is_0_60_l597_597300


namespace northern_village_population_l597_597485

theorem northern_village_population
    (x : ℕ) -- Northern village population
    (western_village_population : ℕ := 400)
    (southern_village_population : ℕ := 200)
    (total_conscripted : ℕ := 60)
    (northern_village_conscripted : ℕ := 10)
    (h : (northern_village_conscripted : ℚ) / total_conscripted = (x : ℚ) / (x + western_village_population + southern_village_population)) : 
    x = 120 :=
    sorry

end northern_village_population_l597_597485


namespace cone_volume_calculation_l597_597399

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

theorem cone_volume_calculation :
  let slant_height := 2 in
  let unfolded_surface_arc_length := 2 * π in
  let base_circumference := 2 * π in
  let radius := 1 in  -- Derived from base_circumference
  let height := real.sqrt (slant_height^2 - radius^2) in
  cone_volume radius height = √3 / 3 * π :=
by
  let slant_height := 2
  let unfolded_surface_arc_length := 2 * π
  let base_circumference := 2 * π
  let radius := 1
  let height := real.sqrt (slant_height^2 - radius^2)
  show cone_volume radius height = √3 / 3 * π from sorry

end cone_volume_calculation_l597_597399


namespace linear_coefficient_l597_597120

theorem linear_coefficient (a b c : ℤ) (h : a = 1 ∧ b = -2 ∧ c = -1) :
    b = -2 := 
by
  -- Use the given hypothesis directly
  exact h.2.1

end linear_coefficient_l597_597120


namespace prove_angle_A_prove_range_cosB_cosC_l597_597979

-- Assuming that angles are in radians and triangle is acute.
variables {A B C : ℝ}

-- Condition of the problem
axiom cond1 : sqrt 3 * sin ((B + C) / 2) - cos A = 1
axiom cond2 : A = π / 3
axiom acute_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2

noncomputable def measure_angle_A : Prop :=
  A = π / 3

noncomputable def range_cosB_cosC : Set ℝ :=
  {x | ∃ B C, cond1 ∧ cond2 ∧ acute_triangle ∧ (x = cos B + cos C)}

theorem prove_angle_A : measure_angle_A :=
by 
  exact cond2

theorem prove_range_cosB_cosC : 
  range_cosB_cosC = {y | sqrt 3 / 2 < y ∧ y ≤ 1} :=
sorry

end prove_angle_A_prove_range_cosB_cosC_l597_597979


namespace find_f_10_l597_597056

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) (hx : x ≠ 0) : f x = f (1 / x) * Real.log x + 10

theorem find_f_10 : f 10 = 10 :=
by
  sorry

end find_f_10_l597_597056


namespace cos_theta_add_pi_over_4_l597_597959

theorem cos_theta_add_pi_over_4 :
  ∀ θ : ℝ, (cos θ = -12 / 13) → (π < θ ∧ θ < 3 * π / 2) → cos (θ + π / 4) = -7 * Real.sqrt 2 / 26 := 
by
  sorry

end cos_theta_add_pi_over_4_l597_597959


namespace year_2015_common_and_feb_days_l597_597294

theorem year_2015_common_and_feb_days : 
  ∀ (year : ℕ), year = 2015 → (¬ year % 4 = 0) → February_days year = 28 :=
by
  intro year h_year h_not_leap
  have h1 : year = 2015 := h_year
  have h2 : ¬ year % 4 = 0 := h_not_leap
  have h_common : year % 4 ≠ 0 := by simp [h2]
  have h_days : February_days year = 28 := 
  sorry
end

def February_days (year: ℕ) : ℕ := 
  if year % 4 = 0 then 29 
  else 28

end year_2015_common_and_feb_days_l597_597294


namespace odd_function_g_find_f_neg2_l597_597432

variable {R : Type*} [LinearOrderedField R]

def f (x m n : R) : R :=
  (2^x - 2^(-x)) * m + (x^3 + x) * n + x^2 - 1

def g (x m n : R) : R := 
  f x m n - x^2 + 1

theorem odd_function_g (m n : R) : 
  ∀ x : R, g (-x) m n = -g x m n := by
  sorry

theorem find_f_neg2 (m n : R) (h : f 2 m n = 8) :
  f (-2) m n = -2 := by
  have odd_g := odd_function_g m n
  sorry

end odd_function_g_find_f_neg2_l597_597432


namespace simplify_expression_l597_597270

theorem simplify_expression : 4 * (8 - 2 + 3) - 7 = 29 := 
by {
  sorry
}

end simplify_expression_l597_597270


namespace φ_monotonic_intervals_f_g_comparison_l597_597043

-- Definition of functions
def f (x : ℝ) := 3 * Real.exp x + x^2
def g (x : ℝ) := 9 * x - 1
def φ (x : ℝ) := x * Real.exp x + 4 * x - f x

-- Statement for monotonicity intervals of φ
theorem φ_monotonic_intervals :
  (∀ x, x ∈ Ioc (-∞) (Real.log 2) → φ' x > 0) ∧
  (∀ x, x ∈ Ioc 2 ⊤ → φ' x > 0) ∧
  (∀ x, x ∈ Ioo (Real.log 2) 2 → φ' x < 0) := sorry

-- Statement comparing f and g
theorem f_g_comparison : ∀ x, f x > g x := sorry

end φ_monotonic_intervals_f_g_comparison_l597_597043


namespace exceed_1000_cents_l597_597530

def total_amount (n : ℕ) : ℕ :=
  3 * (3 ^ n - 1) / (3 - 1)

theorem exceed_1000_cents : 
  ∃ n : ℕ, total_amount n ≥ 1000 ∧ (n + 7) % 7 = 6 := 
by
  sorry

end exceed_1000_cents_l597_597530


namespace sequence_general_term_l597_597247

-- Define the sequence
def sequence (n : ℕ) : ℝ :=
  match n with
  | 0 => sqrt 2
  | 1 => sqrt 5
  | 2 => sqrt 8
  | 3 => sqrt 11
  | _ => sqrt (3 * n - 1) -- This general form is assumed it's correct from n>=4

-- Theorem to state the mathematical problem
theorem sequence_general_term (n : ℕ) : 
  sequence n = sqrt (3 * n - 1) :=
sorry

end sequence_general_term_l597_597247


namespace clothing_prices_l597_597221

theorem clothing_prices (m : ℕ) (x y : ℕ) 
  (cost_A cost_B cost_C sell_A sell_C : ℕ)
  (h1 : cost_A = 20)
  (h2 : cost_B = 30)
  (h3 : cost_C = 40)
  (h4 : sell_A = 24)
  (sell_B : ℕ equals m)
  (h5 : sell_C = 52)
  (h_profit : 4 * x = (m - 30) * y = 3 * (x + y))
  (h_volume : x + y = 4 * ⟨1/4⟩ * (x + y)) 
  : m = 42 :=
by
  sorry

end clothing_prices_l597_597221


namespace repeat_block_of_7_div_13_l597_597637

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l597_597637


namespace quadrilateral_perimeter_sum_pqr_l597_597177

structure Point where
  x : ℝ
  y : ℝ

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

noncomputable def sum_p_q_r (E F G H : Point) : ℝ :=
  let EF := distance E F
  let FG := distance F G
  let GH := distance G H
  let HE := distance H E
  let perimeter := EF + FG + GH + HE
  if (perimeter = 10 + real.sqrt 13 + real.sqrt 85) then 2 else 0

theorem quadrilateral_perimeter_sum_pqr :
  let E := Point.mk 1 2
  let F := Point.mk 4 6
  let G := Point.mk 8 3
  let H := Point.mk 10 0
  sum_p_q_r E F G H = 2 :=
by
  sorry

end quadrilateral_perimeter_sum_pqr_l597_597177


namespace find_x_satisfying_floor_eq_l597_597862

theorem find_x_satisfying_floor_eq (x : ℝ) (hx: ⌊x⌋ * x = 152) : x = 38 / 3 :=
sorry

end find_x_satisfying_floor_eq_l597_597862


namespace tony_total_winning_l597_597711

theorem tony_total_winning : 
  ∀ (num_tickets num_winning_numbers_per_ticket winnings_per_number : ℕ),
  num_tickets = 3 → 
  num_winning_numbers_per_ticket = 5 →
  winnings_per_number = 20 →
  num_tickets * num_winning_numbers_per_ticket * winnings_per_number = 300 :=
by {
  intros num_tickets num_winning_numbers_per_ticket winnings_per_number h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
}

end tony_total_winning_l597_597711


namespace interest_rate_is_25_percent_l597_597192

variables (P : ℝ) (r : ℝ)

-- Conditions:
def condition1 : Prop := P * (1 + 4 * r) = 400
def condition2 : Prop := P * (1 + 6 * r) = 500

-- The rate of interest
def rate_of_interest : ℝ := 0.25

-- Proof problem statement
theorem interest_rate_is_25_percent (h1 : condition1) (h2 : condition2) : r = rate_of_interest :=
sorry

end interest_rate_is_25_percent_l597_597192


namespace repeating_block_length_7_div_13_l597_597623

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l597_597623


namespace sum_of_complex_terms_l597_597415

-- Define the imaginary unit i and the sequence
def i := Complex.I
def seq := [1, 2, 3, 4, 5, 6, 7, 8]
def term (n : ℕ) := seq[n] * i^(n + 1)

-- Define the sum of the sequence
def sum_seq := term 0 + term 1 + term 2 + term 3 + term 4 + term 5 + term 6 + term 7

-- The theorem to prove
theorem sum_of_complex_terms : sum_seq = 4 - 4 * i := 
    sorry

end sum_of_complex_terms_l597_597415


namespace repeating_block_length_7_div_13_l597_597577

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597577


namespace number_of_pairs_without_zero_digits_l597_597873

def no_zero_digit (n : ℕ) : Prop :=
  ∀ d in (n.digits 10), d ≠ 0

theorem number_of_pairs_without_zero_digits :
  (∃! (a b : ℕ), a + b = 2000 ∧ a > 0 ∧ b > 0 ∧ no_zero_digit a ∧ no_zero_digit b) → 
  (card {p : ℕ × ℕ | p.1 + p.2 = 2000 ∧ no_zero_digit p.1 ∧ no_zero_digit p.2} = 1458) :=
sorry

end number_of_pairs_without_zero_digits_l597_597873


namespace XB_parallel_YC_l597_597033

open EuclideanGeometry

-- Define the given circle and its elements
variables {A B C X Y : Point}
variables {circle : Circle}

axiom diameter_AB : diameter circle A B
axiom point_on_diameter : C ∈ segment A B
axiom points_on_circle : X ∈ circle ∧ Y ∈ circle
axiom symmetric_points : symmetric_about_diameter circle A B X Y

-- Define that YC is perpendicular to XA
axiom perpendicular_lines : Perp (line_through Y C) (line_through X A)

-- The theorem we need to prove
theorem XB_parallel_YC : (parallel (line_through X B) (line_through Y C)) :=
sorry

end XB_parallel_YC_l597_597033


namespace distance_to_asymptote_l597_597642

theorem distance_to_asymptote :
  let hyperbola := { P : ℝ × ℝ | P.1^2 - (P.2^2 / 9) = 1 }
  let asymptote := { P : ℝ × ℝ | 3 * P.1 + P.2 = 0 }
  let point_M := (-1, 0 : ℝ)
  let distance (P L : ℝ × ℝ) := abs (3 * P.1 + P.2) / real.sqrt (3^2 + 1^1)
  distance point_M (0, 0) = 3 * real.sqrt 10 / 10 :=
sorry

end distance_to_asymptote_l597_597642


namespace find_distance_between_P_and_Q_l597_597792

-- Definitions and setup
variable (A B C D : ℝ^3) -- Vertices of the regular tetrahedron
variable (P Q : ℝ^3)

-- Conditions
def regular_tetrahedron (A B C D : ℝ^3) : Prop :=
  (dist A B = 1) ∧ (dist A C = 1) ∧ (dist A D = 1) ∧
  (dist B C = 1) ∧ (dist B D = 1) ∧ (dist C D = 1)

def point_on_edge (P : ℝ^3) (A C : ℝ^3) : Prop :=
  P = (2/3) • A + (1/3) • C

def point_on_edge' (Q : ℝ^3) (B D : ℝ^3) : Prop :=
  Q = (2/3) • B + (1/3) • D

-- Problem statement
theorem find_distance_between_P_and_Q 
  (A B C D : ℝ^3)
  (h_tetrahedron : regular_tetrahedron A B C D)
  (P_on_edge : point_on_edge P A C)
  (Q_on_edge : point_on_edge' Q B D) :
  dist P Q = 2 / 3 :=
  sorry

end find_distance_between_P_and_Q_l597_597792


namespace proof_sum_of_ab_l597_597993

theorem proof_sum_of_ab :
  ∃ (a b : ℕ), a ≤ b ∧ 0 < a ∧ 0 < b ∧ a ^ 2 + b ^ 2 + 8 * a * b = 2010 ∧ a + b = 42 :=
sorry

end proof_sum_of_ab_l597_597993


namespace blue_whale_tongue_weight_l597_597236

theorem blue_whale_tongue_weight :
  let weight_in_kg := 2700
  let kg_to_pounds := 2.20462
  let pounds_to_ton := 1 / 2000
  let weight_in_pounds := weight_in_kg * kg_to_pounds
  let weight_in_tons := weight_in_pounds * pounds_to_ton
  weight_in_tons ≈ 2.976237 := by
  sorry

end blue_whale_tongue_weight_l597_597236


namespace find_sphere_radius_l597_597308

noncomputable def radius_of_sphere : ℝ :=
let height_of_stick := 1.5
let shadow_length_of_stick := 3.0
let shadow_length_of_sphere := 15.0
tan(Real.angle_of (height_of_stick / shadow_length_of_stick)) = tan(Real.angle_of (radius_of_sphere / shadow_length_of_sphere))

theorem find_sphere_radius :
    radius_of_sphere = 7.5 :=
by sorry

end find_sphere_radius_l597_597308


namespace xiaoqiang_average_score_l597_597756

theorem xiaoqiang_average_score
    (x : ℕ)
    (prev_avg : ℝ)
    (next_score : ℝ)
    (target_avg : ℝ)
    (h_prev_avg : prev_avg = 84)
    (h_next_score : next_score = 100)
    (h_target_avg : target_avg = 86) :
    (86 * x - (84 * (x - 1)) = 100) → x = 8 := 
by
  intros h_eq
  sorry

end xiaoqiang_average_score_l597_597756


namespace parallelogram_area_l597_597339

open Real

variables {V : Type*} [InnerProductSpace ℝ V]

def p : V := sorry
def q : V := sorry
def a : V := 2 • p - 3 • q
def b : V := 5 • p + q

theorem parallelogram_area :
  ‖p‖ = 2 →
  ‖q‖ = 3 →
  real.angle p q = π / 2 →
  ‖ a × b ‖ = 102 :=
sorry

end parallelogram_area_l597_597339


namespace michael_exceeds_suresh_by_36_5_l597_597201

noncomputable def shares_total : ℝ := 730
noncomputable def punith_ratio_to_michael : ℝ := 3 / 4
noncomputable def michael_ratio_to_suresh : ℝ := 3.5 / 3

theorem michael_exceeds_suresh_by_36_5 :
  ∃ P M S : ℝ, P + M + S = shares_total
  ∧ (P / M = punith_ratio_to_michael)
  ∧ (M / S = michael_ratio_to_suresh)
  ∧ (M - S = 36.5) :=
by
  sorry

end michael_exceeds_suresh_by_36_5_l597_597201


namespace extreme_values_curve_intersections_l597_597880

noncomputable def f (x : ℝ) (p : ℝ) := ∫ t in -1..x, p - log(1 + abs t)

theorem extreme_values (p : ℝ) (hp : 0 < p) : 
  ∃ x1 x2 : ℝ, (x1 = exp p - 1 ∧ x2 = -(exp p - 1)) ∧ 
  (∃ x_extreme1 x_extreme2, x_extreme1 = x1 ∧ x_extreme2 = x2) :=
sorry

theorem curve_intersections (p : ℝ) : 
  ln (2 * ln 2) < p ∧ p < 2 * ln 2 - 1 → 
  ∃ x1 x2 : ℝ, (0 < x1 ∧ 0 < x2) ∧ (∫ t in -1..x1, p - log(1 + abs t) = 0 ∧ ∫ t in -1..x2, p - log(1 + abs t) = 0) :=
sorry

end extreme_values_curve_intersections_l597_597880


namespace player_1_points_after_13_rotations_l597_597688

theorem player_1_points_after_13_rotations :
  ∀ (table : ℕ → ℕ) (n : ℕ) (points : List ℕ),
    (∀ i, table i+16 = table i) →
    (table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]) →
    (points.length = 16) →
    (points.get 5 = 72) →
    (points.get 9 = 84) →
    (n = 13) →
    ((points.sum i₁, 0, 13) table ((stats : List ℕ) (i : fin 16) =>
      List.sum (List.take stats.toList) i.val + 
      List.sum (List.drop stats.toList i.val i.val + 2 * n) table) = points.sum table) →
    points.get 1 = 20 :=
by
  intros
  sorry

end player_1_points_after_13_rotations_l597_597688


namespace dan_cookies_proof_l597_597815

variables (rect_area : ℕ) (circle_area : ℝ)
def art_cookies_area := rect_area * 18 = 432
def dan_cookies_area := circle_area = 9 * Real.pi

-- The number of Dan's cookies per batch
def dan_cookies_count := (432 / (9 * Real.pi)).round = 15

theorem dan_cookies_proof
  (h1 : rect_area = 24)
  (h2 : circle_area = 9 * Real.pi)
  (h3 : art_cookies_area 24 = 432)
  (h4 : dan_cookies_area (9 * Real.pi)) :
  dan_cookies_count :=
by sorry

end dan_cookies_proof_l597_597815


namespace AP_eq_PL_l597_597281

-- Define the context: points A, B, C, L, K, and P, and their geometric relations
variables (A B C L K P : Type)
variables [Point A] [Point B] [Point C] [Point L] [Point K] [Point P]

-- Define the line segments and angle bisectors
variables (AL : Line A L) (BK : Line B K) (KL : Line K L)
variables (angleBisectorA : AngleBisector A L B P) (angleBisectorB : AngleBisector B K C P)

-- Define the conditions: angle bisectors and lengths
variables (h1 : is_angle_bisector A B C AL)
variables (h2 : K ∈ AC ∧ CK = CL)
variables (h3 : ∃ P, P ∈ intersect_lines KL BK)

-- The theorem statement
theorem AP_eq_PL (AL : AngleBisector A L B P) (h4 : P ∈ intersect_lines KL BK) : 
  distance A P = distance P L :=
sorry

end AP_eq_PL_l597_597281


namespace nancy_gives_marilyn_bottle_caps_l597_597184

def nancy_gave (initial : ℝ) (final : ℝ) : ℝ :=
  final - initial

theorem nancy_gives_marilyn_bottle_caps :
  nancy_gave 51.0 87.0 = 36.0 :=
by
  unfold nancy_gave
  exact rfl

end nancy_gives_marilyn_bottle_caps_l597_597184


namespace determine_p_and_q_l597_597460

theorem determine_p_and_q (x p q : ℝ) : 
  (x + 4) * (x - 1) = x^2 + p * x + q → (p = 3 ∧ q = -4) := 
by 
  sorry

end determine_p_and_q_l597_597460


namespace largest_measureable_quantity_is_1_l597_597773

theorem largest_measureable_quantity_is_1 : 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd 496 403) 713) 824) 1171 = 1 :=
  sorry

end largest_measureable_quantity_is_1_l597_597773


namespace sum_after_removal_equals_target_l597_597275

theorem sum_after_removal_equals_target :
  let initial_sum := (1 / 3) + (1 / 5) + (1 / 7) + (1 / 9) + (1 / 11) + (1 / 13)
  let target_sum := 3 / 2
  let terms_to_remove := [1 / 5, 1 / 13]
  let remaining_sum := initial_sum - (terms_to_remove.sum)
  in remaining_sum = target_sum :=
by
  sorry

end sum_after_removal_equals_target_l597_597275


namespace change_combinations_50_cents_l597_597090

-- Define the conditions for creating 50 cents using standard coins
def ways_to_make_change (pennies nickels dimes : ℕ) : ℕ :=
  pennies + 5 * nickels + 10 * dimes

theorem change_combinations_50_cents : 
  ∃ num_ways, 
    num_ways = 28 ∧
    ∀ (pennies nickels dimes : ℕ), 
      pennies + 5 * nickels + 10 * dimes = 50 → 
      -- Exclude using only a single half-dollar
      ¬(num_ways = if (pennies = 0 ∧ nickels = 0 ∧ dimes = 0) then 1 else 28) := 
sorry

end change_combinations_50_cents_l597_597090


namespace problem_B_union_C_U_A_l597_597503

open Set

/-- Set up the universe U and the sets A and B -/
def U : Set ℕ := {a, b, c, d, e}
def A : Set ℕ := {b, c, d}
def B : Set ℕ := {b, e}

/-- Define the complement of A with respect to U -/
def C_U_A : Set ℕ := U \ A

/-- Prove that the union of B and the complement of A with respect to U is {a, b, e} -/
theorem problem_B_union_C_U_A : B ∪ C_U_A = {a, b, e} :=
  by
    sorry

end problem_B_union_C_U_A_l597_597503


namespace greatest_abs_solution_l597_597852

theorem greatest_abs_solution :
  (∃ x : ℝ, x^2 + 18 * x + 81 = 0 ∧ ∀ y : ℝ, y^2 + 18 * y + 81 = 0 → |x| ≥ |y| ∧ |x| = 9) :=
sorry

end greatest_abs_solution_l597_597852


namespace reconstructed_pentagon_proof_l597_597402

noncomputable def prove_pentagon (A A' A'' B' C' D' E' : ℝ → ℝ) : Prop := 
  ∃ (p q r s t u : ℝ), 
    A = p * A' + q * A'' + r * B' + s * C' + t * D' + u * E' ∧
    p = 1/31 ∧ q = 2/31 ∧ r = 4/31 ∧ s = 8/31 ∧ t = 16/31 ∧ u = 0

theorem reconstructed_pentagon_proof : 
  ∀ (A A' A'' B' C' D' E' : ℝ → ℝ),
    prove_pentagon A A' A'' B' C' D' E' := 
  by 
    sorry

end reconstructed_pentagon_proof_l597_597402


namespace new_average_weight_l597_597219

def average_weight (A B C D E : ℝ) : Prop :=
  (A + B + C) / 3 = 70 ∧
  (A + B + C + D) / 4 = 70 ∧
  E = D + 3 ∧
  A = 81

theorem new_average_weight (A B C D E : ℝ) (h: average_weight A B C D E) : 
  (B + C + D + E) / 4 = 68 :=
by
  sorry

end new_average_weight_l597_597219


namespace max_blocks_that_fit_l597_597746

noncomputable def box_volume : ℕ :=
  3 * 4 * 2

noncomputable def block_volume : ℕ :=
  2 * 1 * 2

noncomputable def max_blocks (box_volume : ℕ) (block_volume : ℕ) : ℕ :=
  box_volume / block_volume

theorem max_blocks_that_fit : max_blocks box_volume block_volume = 6 :=
by
  sorry

end max_blocks_that_fit_l597_597746


namespace intersection_point_ratio_l597_597482

variables (E F G H Q R S : Type)
variable [Parallelogram E F G H]
variables [Colinear E F Q] [Colinear E H R]
variables [IntersectionPoint S E G Q R]

def ratio_EQ_EF : ℝ := 1 / 4
def ratio_ER_EH : ℝ := 1 / 9

-- The proof statement to be provided
theorem intersection_point_ratio (EG ES : ℝ) (h1 : EQ / EF = ratio_EQ_EF) (h2 : ER / EH = ratio_ER_EH) :
  EG / ES = 36 := sorry

end intersection_point_ratio_l597_597482


namespace min_rounds_for_expected_value_l597_597854

theorem min_rounds_for_expected_value 
  (p1 p2 : ℝ) (h0 : 0 ≤ p1 ∧ p1 ≤ 1) (h1 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h2 : p1 + p2 = 3 / 2)
  (indep : true) -- Assuming independence implicitly
  (X : ℕ → ℕ) (n : ℕ)
  (E_X_eq_24 : (n : ℕ) * (3 * p1 * p2 * (1 - p1 * p2)) = 24) :
  n = 32 := 
sorry

end min_rounds_for_expected_value_l597_597854


namespace player_1_points_after_13_rotations_l597_597689

theorem player_1_points_after_13_rotations :
  ∀ (table : ℕ → ℕ) (n : ℕ) (points : List ℕ),
    (∀ i, table i+16 = table i) →
    (table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]) →
    (points.length = 16) →
    (points.get 5 = 72) →
    (points.get 9 = 84) →
    (n = 13) →
    ((points.sum i₁, 0, 13) table ((stats : List ℕ) (i : fin 16) =>
      List.sum (List.take stats.toList) i.val + 
      List.sum (List.drop stats.toList i.val i.val + 2 * n) table) = points.sum table) →
    points.get 1 = 20 :=
by
  intros
  sorry

end player_1_points_after_13_rotations_l597_597689


namespace minimum_value_f_l597_597167

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ t ≥ 0, ∀ (x y : ℝ), 0 < x → 0 < y → f x y ≥ t ∧ t = 4 * Real.sqrt 2 :=
sorry

end minimum_value_f_l597_597167


namespace avg_height_is_28_l597_597796

-- Define the height relationship between trees
def height_relation (a b : ℕ) := a = 2 * b ∨ a = b / 2

-- Given tree heights (partial information)
def height_tree_2 := 14
def height_tree_5 := 20

-- Define the tree heights variables
variables (height_tree_1 height_tree_3 height_tree_4 height_tree_6 : ℕ)

-- Conditions based on the given data and height relations
axiom h1 : height_relation height_tree_1 height_tree_2
axiom h2 : height_relation height_tree_2 height_tree_3
axiom h3 : height_relation height_tree_3 height_tree_4
axiom h4 : height_relation height_tree_4 height_tree_5
axiom h5 : height_relation height_tree_5 height_tree_6

-- Compute total and average height
def total_height := height_tree_1 + height_tree_2 + height_tree_3 + height_tree_4 + height_tree_5 + height_tree_6
def average_height := total_height / 6

-- Prove the average height is 28 meters
theorem avg_height_is_28 : average_height = 28 := by
  sorry

end avg_height_is_28_l597_597796


namespace function_inequality_l597_597647

theorem function_inequality {f : ℝ → ℝ} (h1 : ∀ x ∈ Icc (0 : ℝ) 1, True)
  (h2 : f 0 = f 1)
  (h3 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → x1 ∈ Icc (0 : ℝ) 1 → x2 ∈ Icc (0 : ℝ) 1 → |f x2 - f x1| < |x2 - x1|) :
  ∀ (x1 x2 : ℝ), x1 ∈ Icc (0 : ℝ) 1 → x2 ∈ Icc (0 : ℝ) 1 → |f x2 - f x1| < 1 / 2 :=
by
  sorry

end function_inequality_l597_597647


namespace real_part_of_zi_add_i_l597_597398

noncomputable def z : ℂ := 1 / (1 - complex.I)^2

theorem real_part_of_zi_add_i : (z * complex.I + complex.I).re = -1 / 2 :=
by
  sorry

end real_part_of_zi_add_i_l597_597398


namespace induction_sum_lemma_l597_597264

theorem induction_sum_lemma (n : ℕ) (hn : n ≥ 1) :
  (∑ i in Finset.range(2 * n), (-1) ^ i * (1 / (i + 1))) =
  (∑ j in Finset.range(n + 1, 2 * n + 1), 1 / j) := sorry

end induction_sum_lemma_l597_597264


namespace volume_prism_result_l597_597040

noncomputable def volume_prism (A B C S : ℝ) (SA : ℝ) (angle_HAB_C : ℝ) (projection_H_orthocenter : Prop) : ℝ :=
  if h₁ : equilateral_triangle A B C ∧
          orthocenter_proj_H_SBC A B C S ∧
          dihedral_angle_H_A_B_C A B C S = angle_HAB_C ∧
          SA = 2 then
    3 / 4
  else
    0

theorem volume_prism_result :
  volume_prism (A B C S : ℝ) (SA : ℝ) (angle_HAB_C : ℝ) (projection_H_orthocenter : Prop) = 3 / 4 := by
  sorry

end volume_prism_result_l597_597040


namespace least_k_l597_597561

noncomputable def u : ℕ → ℝ
| 0       := 1 / 8
| (k + 1) := 2 * u k - 2 * (u k)^2

def L : ℝ := 1 / 2

theorem least_k (k : ℕ) (h : ∀ n ≤ k, (|u n - L| ≤ 1 / 2^10)) : 
  k = 4 := 
sorry

end least_k_l597_597561


namespace repeating_block_length_7_div_13_l597_597621

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l597_597621


namespace problem_solution_l597_597253

noncomputable def x := 91143
noncomputable def y := 82574

def is_sum_173717 (x y : ℕ) : Prop :=
  x + y = 173717

def is_four_digit_difference (x y : ℕ) : Prop :=
  let d := |x - y|
  1000 ≤ d ∧ d < 10000

def no_single_digit_prime_factors (n : ℕ) : Prop :=
  ∀ p, Prime p → p < 10 → ¬ p ∣ n

def is_divisible_1558 (n : ℕ) : Prop :=
  1558 ∣ n

theorem problem_solution :
  is_sum_173717 x y ∧
  is_four_digit_difference x y ∧
  no_single_digit_prime_factors (|x - y|) ∧
  (is_divisible_1558 x ∨ is_divisible_1558 y) :=
by
  sorry

end problem_solution_l597_597253


namespace sin_780_eq_sqrt3_over_2_l597_597349

theorem sin_780_eq_sqrt3_over_2 :
  sin (780 : ℝ) = (Real.sqrt 3 / 2) :=
by
  sorry

end sin_780_eq_sqrt3_over_2_l597_597349


namespace smallest_integer_n_l597_597510

def smallest_n (m : ℤ) (n : ℕ) (r : ℝ) : Prop :=
  m = (n : ℝ + r)^4 ∧ r < 1/10000

theorem smallest_integer_n {m : ℤ} {n : ℕ} {r : ℝ} :
  (∀ r, (smallest_n m n r) → n = 14) :=
by
  sorry

end smallest_integer_n_l597_597510


namespace exists_k_leq_n_l597_597047

theorem exists_k_leq_n 
  (n : ℕ) (n_gt_two : n > 2) 
  (x : ℕ → ℝ) 
  (h_sum_gt_one : abs (∑ i in Finset.range n, x i) > 1) 
  (h_abs_leq_one : ∀ i < n, abs (x i) ≤ 1) :
  ∃ k : ℕ, k < n ∧ abs ((∑ i in Finset.range k, x i) - (∑ i in Finset.range (n - k), x (k + i))) ≤ 1 := 
sorry

end exists_k_leq_n_l597_597047


namespace imaginary_condition_l597_597024

theorem imaginary_condition (a b : ℝ) : (b ≠ 0) ↔ (a + b * complex.I).im ≠ 0 :=
by
  sorry

end imaginary_condition_l597_597024


namespace repeating_block_length_7_div_13_l597_597613

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597613


namespace smallest_unwritable_number_l597_597383

theorem smallest_unwritable_number :
  ∀ a b c d : ℕ, 11 ≠ (2^a - 2^b) / (2^c - 2^d) := sorry

end smallest_unwritable_number_l597_597383


namespace find_specified_time_l597_597984

theorem find_specified_time (distance : ℕ) (slow_time fast_time : ℕ → ℕ) (fast_is_double : ∀ x, fast_time x = 2 * slow_time x)
  (distance_value : distance = 900) (slow_time_eq : ∀ x, slow_time x = x + 1) (fast_time_eq : ∀ x, fast_time x = x - 3) :
  2 * (distance / (slow_time x)) = distance / (fast_time x) :=
by
  intros
  rw [distance_value, slow_time_eq, fast_time_eq]
  sorry

end find_specified_time_l597_597984


namespace bruce_initial_money_l597_597820

-- Definitions of the conditions
def cost_crayons : ℕ := 5 * 5
def cost_books : ℕ := 10 * 5
def cost_calculators : ℕ := 3 * 5
def total_spent : ℕ := cost_crayons + cost_books + cost_calculators
def cost_bags : ℕ := 11 * 10
def initial_money : ℕ := total_spent + cost_bags

-- Theorem statement
theorem bruce_initial_money :
  initial_money = 200 := by
  sorry

end bruce_initial_money_l597_597820


namespace car_mileage_l597_597298

/-- If a car needs 3.5 gallons of gasoline to travel 140 kilometers, it gets 40 kilometers per gallon. -/
theorem car_mileage (gallons_used : ℝ) (distance_traveled : ℝ) 
  (h : gallons_used = 3.5 ∧ distance_traveled = 140) : 
  distance_traveled / gallons_used = 40 :=
by
  sorry

end car_mileage_l597_597298


namespace num_undefined_values_l597_597010

theorem num_undefined_values : 
  let f (x : ℝ) := (x^2 + 2*x - 3) * (x - 3) * (x + 4)
  ∃ S : Finset ℝ, (∀ x : ℝ, f x = 0 ↔ x ∈ S) ∧ S.card = 4 :=
by
  let f (x : ℝ) := (x^2 + 2*x - 3) * (x - 3) * (x + 4)
  use {1, -3, 3, -4}
  split
  · intro x
    simp [f, mul_eq_zero, add_eq_zero_iff_eq_neg, sub_eq_zero, Finset.mem_insert, Finset.mem_singleton]
    tauto
  · rw Finset.card_insert_add_card {−3, 3, -4} 1
    simp
    sorry

end num_undefined_values_l597_597010


namespace repeating_block_length_7_div_13_l597_597616

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597616


namespace vector_coordinates_l597_597915

theorem vector_coordinates (b : ℝ × ℝ)
  (a : ℝ × ℝ := (Real.sqrt 3, 1))
  (angle : ℝ := 2 * Real.pi / 3)
  (norm_b : ℝ := 1)
  (dot_product_eq : (a.fst * b.fst + a.snd * b.snd = -1))
  (norm_b_eq : (b.fst ^ 2 + b.snd ^ 2 = 1)) :
  b = (0, -1) ∨ b = (-Real.sqrt 3 / 2, 1 / 2) :=
sorry

end vector_coordinates_l597_597915


namespace repeating_block_length_7_div_13_l597_597580

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597580


namespace relationship_a_b_c_d_l597_597091

theorem relationship_a_b_c_d 
  (a b c d : ℤ)
  (h : (a + b + 1) * (d + a + 2) = (c + d + 1) * (b + c + 2)) : 
  a + b + c + d = -2 := 
sorry

end relationship_a_b_c_d_l597_597091


namespace geometric_seq_b_sum_formula_T_l597_597287

-- Define the sequence \{x_n\}
def f (x : ℝ) : ℝ := ((x^3 + 3*x) / (3*x^2 + 1))

def seq_x : ℕ → ℝ
| 0       := 2
| (n + 1) := f (seq_x n)

-- Define the sequence \{b_n\}
def b (n : ℕ) : ℝ := Real.logBase 3 (((seq_x (n + 1)) - 1) / ((seq_x (n + 1)) + 1))

-- Define and prove that \{b_n\} is a geometric sequence
theorem geometric_seq_b :
  ∀ n, b (n + 1) = 3 * b n ∧ b 0 = -3 := sorry

-- Define the sequence \{c_n\}
def c (n : ℕ) : ℝ := -n * b n

-- Define the sum of the first n terms of \{c_n\}
def T (n : ℕ) : ℝ := ∑ k in Finset.range n, c (k + 1)

-- Prove the formula for the sum T_n
theorem sum_formula_T :
  ∀ n, T n = (2 * n - 1) * 3^n + 1 / 4 := sorry

end geometric_seq_b_sum_formula_T_l597_597287


namespace remaining_budget_l597_597451

def charge_cost : ℝ := 3.5
def num_charges : ℝ := 4
def total_budget : ℝ := 20

theorem remaining_budget : total_budget - (num_charges * charge_cost) = 6 := 
by 
  sorry

end remaining_budget_l597_597451


namespace inequality_holds_equality_condition_l597_597535

theorem inequality_holds (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  ( (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ) / ( (a - b)^2 + (b - c)^2 + (c - a)^2 ) ≥ 1 / 2 :=
sorry

theorem equality_condition (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  ( (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ) / ( (a - b)^2 + (b - c)^2 + (c - a)^2 ) = 1 / 2 ↔ 
  ((a = 0 ∧ b = 0 ∧ 0 < c) ∨ (a = 0 ∧ c = 0 ∧ 0 < b) ∨ (b = 0 ∧ c = 0 ∧ 0 < a)) :=
sorry

end inequality_holds_equality_condition_l597_597535


namespace distance_to_fourth_side_l597_597406

-- Let s be the side length of the square.
variable (s : ℝ) (d1 d2 d3 d4 : ℝ)

-- The given conditions:
axiom h1 : d1 = 4
axiom h2 : d2 = 7
axiom h3 : d3 = 13
axiom h4 : d1 + d2 + d3 + d4 = s
axiom h5 : 0 < d4

-- The statement to prove:
theorem distance_to_fourth_side : d4 = 10 ∨ d4 = 16 :=
by
  sorry

end distance_to_fourth_side_l597_597406


namespace min_keychains_to_reach_profit_l597_597312

theorem min_keychains_to_reach_profit :
  let cost_per_keychain := 0.15
  let sell_price_per_keychain := 0.45
  let total_keychains := 1200
  let target_profit := 180
  let total_cost := total_keychains * cost_per_keychain
  let total_revenue := total_cost + target_profit
  let min_keychains_to_sell := total_revenue / sell_price_per_keychain
  min_keychains_to_sell = 800 := 
by
  sorry

end min_keychains_to_reach_profit_l597_597312


namespace black_rectangle_ways_l597_597736

theorem black_rectangle_ways : ∑ a in Finset.range 5, ∑ b in Finset.range 5, (5 - a) * (5 - b) = 225 := sorry

end black_rectangle_ways_l597_597736


namespace always_possible_to_cut_1x1_without_holes_l597_597973

theorem always_possible_to_cut_1x1_without_holes (holes : Finset (ℝ × ℝ)) (h : holes.card = 15) :
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 ∧ ∀ (hx ∈ holes), ¬ (x ≤ hx.1 ∧ hx.1 < x + 1 ∧ y ≤ hx.2 ∧ hx.2 < y + 1) :=
by
  sorry

end always_possible_to_cut_1x1_without_holes_l597_597973


namespace inclination_angle_l597_597235

open Real

-- Define the line equation.
def line_eq (x y : ℝ) : Prop := x + y - sqrt 3 = 0

-- Define the slope of the line.
def slope (x y : ℝ) : ℝ := -1

-- Define the inclination angle theta.
def theta : ℝ := 135

-- The theorem that states the inclination angle of the line.
theorem inclination_angle (x y : ℝ) (h : line_eq x y) : tan (theta * π / 180) = slope x y :=
by 
  -- Proof is not needed, so we use sorry.
  sorry

end inclination_angle_l597_597235


namespace largest_common_divisor_l597_597742

theorem largest_common_divisor (h408 : ∀ d, Nat.dvd d 408 → d ∈ [1, 2, 3, 4, 6, 8, 12, 17, 24, 34, 51, 68, 102, 136, 204, 408])
                               (h340 : ∀ d, Nat.dvd d 340 → d ∈ [1, 2, 4, 5, 10, 17, 20, 34, 68, 85, 170, 340]) :
  ∃ d, Nat.dvd d 408 ∧ Nat.dvd d 340 ∧ d = 68 := by
  sorry

end largest_common_divisor_l597_597742


namespace range_of_m_l597_597969

theorem range_of_m (m : ℝ) (x y : ℝ) :
  (∃ (x y : ℝ), sin x = m * sin y ^ 3 ∧ cos x = m * cos y ^ 3) ↔ (1 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l597_597969


namespace count_angles_l597_597851

open Real

noncomputable def isGeometricSequence (a b c : ℝ) : Prop :=
(a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a / b = b / c ∨ b / a = a / c ∨ c / a = a / b)

theorem count_angles (h1 : ∀ θ : ℝ, 0 < θ ∧ θ < 2 * π → (sin θ * cos θ = tan θ) ∨ (sin θ ^ 3 = cos θ ^ 2)) :
  ∃ n : ℕ, 
    (∀ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧ (θ % (π/2) ≠ 0) → isGeometricSequence (sin θ) (cos θ) (tan θ) ) → 
    n = 6 := 
sorry

end count_angles_l597_597851


namespace matinee_receipts_l597_597005

theorem matinee_receipts :
  let child_ticket_cost := 4.50
  let adult_ticket_cost := 6.75
  let num_children := 48
  let num_adults := num_children - 20
  total_receipts = num_children * child_ticket_cost + num_adults * adult_ticket_cost :=
by 
  sorry

end matinee_receipts_l597_597005


namespace solve_sum_problem_l597_597656

def solve_problem (x : ℤ) : Prop :=
  x^2 = 210 + x

theorem solve_sum_problem :
  (∑ x in {x : ℤ | solve_problem x}.to_finset, x) = 1 :=
sorry

end solve_sum_problem_l597_597656


namespace pond_volume_extraction_l597_597766

theorem pond_volume_extraction :
  ∀ (L W H : ℕ), L = 28 → W = 10 → H = 5 → L * W * H = 1400 :=
by
  intros L W H hL hW hH
  rw [hL, hW, hH]
  norm_num
  sorry

end pond_volume_extraction_l597_597766


namespace repeating_block_length_7_div_13_l597_597584

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597584


namespace boiling_point_in_fahrenheit_l597_597265

-- Define the conditions
def celsius_to_fahrenheit (C : ℝ) : ℝ := (C * (9 / 5)) + 32

theorem boiling_point_in_fahrenheit :
  celsius_to_fahrenheit 100 = 212 := by
  sorry

end boiling_point_in_fahrenheit_l597_597265


namespace bisect_segment_KN_l597_597150

structure QuadrilateralCirumscribedOnCircle (A B C D I M N K : Point) (w : Circle) : Prop :=
  (circumscribed : ω.circumscribed_by A B C D)
  (center_at_I : ω.center = I)
  (angle_sum_lt_pi : ∠BAD + ∠ADC < π)
  (tangency_M : ω.tangency_with AB M)
  (tangency_N : ω.tangency_with CD N)
  (K_on_MN : K ∈ Line.seg M N)
  (AK_eq_AM : dist A K = dist A M)

theorem bisect_segment_KN (A B C D I M N K : Point) (ω : Circle)
  (h : QuadrilateralCirumscribedOnCircle A B C D I M N K ω) : 
  bisects (Line.seg I D) (Line.seg K N) :=
sorry

end bisect_segment_KN_l597_597150


namespace problem_proof_l597_597418

-- Given conditions
def quadratic_function (b : ℝ) (x : ℝ) : ℝ :=
  -x^2 + b * x + 5

def passes_through_M (b : ℝ) : Prop :=
  quadratic_function b (-4) = 5

def axis_of_symmetry (b : ℝ) : ℝ :=
  -b / (2 * (-1))

def intersection_points (b : ℝ) : set (ℝ × ℝ) := 
  {p | quadratic_function b (p.1) = 0}

-- Proof problem
theorem problem_proof (b : ℝ) 
  (h : passes_through_M b) : 
  b = -4 ∧ axis_of_symmetry b = -2 ∧ 
  intersection_points b = {(-5, 0), (1, 0)} :=
begin
  sorry
end

end problem_proof_l597_597418


namespace interval_length_l597_597868

theorem interval_length (x : ℝ) :
  (1/x > 1/2) ∧ (Real.sin x > 1/2) → (2 - Real.pi / 6 = 1.48) :=
by
  sorry

end interval_length_l597_597868


namespace limsup_inequality_specific_sequence_l597_597562

noncomputable theory

open Real

theorem limsup_inequality (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  limsup (λ n, n * ((a (n + 1) + 1) / a n - 1)) ≥ 1 := sorry

theorem specific_sequence:
  lim (λ n, n * ((n.log + 1) / log (n + 1) - 1)) atTop = 1 := sorry

end limsup_inequality_specific_sequence_l597_597562


namespace g_x1_g_x2_l597_597422

-- Defining the odd function y = f(x+1)
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-(x+1)) = -f(x+1)

-- Defining symmetry with respect to y = x
def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = g(x) ↔ f(x) = g(x)

-- Main statement
theorem g_x1_g_x2 (f g : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (h1 : odd_function f)
  (h2 : symmetric_wrt_y_eq_x f g) 
  (h3 : x₁ + x₂ = 0) : 
  g(x₁) + g(x₂) = 2 := 
by 
  sorry

end g_x1_g_x2_l597_597422


namespace cone_volume_correct_l597_597917

noncomputable def cone_volume (r l : ℝ) : ℝ :=
  let h := Real.sqrt (l^2 - r^2) in
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_correct : 
  cone_volume 3 5 = 12 * Real.pi :=
by 
  sorry

end cone_volume_correct_l597_597917


namespace repeating_block_length_of_7_div_13_is_6_l597_597606

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l597_597606


namespace repeating_block_length_7_div_13_l597_597595

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l597_597595


namespace circles_cover_parallelogram_l597_597241

theorem circles_cover_parallelogram (a A : ℝ) (ABCD : Type) [parallelogram ABCD]
  (AB AD : ℝ) (angle_BAD : ℝ)
  (h_AB : AB = a)
  (h_AD : AD = 1)
  (h_angle_BAD : angle_BAD = A)
  (h_acute_triangle_ABD : ∀ A B D : Type, acute_triangle A B D):
  (∀ (c : ℝ), (a ≤ cos A + c * sin A)) ↔ (a ≤ cos A + sqrt 3 * sin A) :=
sorry

end circles_cover_parallelogram_l597_597241


namespace symmetric_line_equation_l597_597223

-- Define the given lines
def original_line (x y : ℝ) : Prop := y = 2 * x + 1
def line_of_symmetry (x y : ℝ) : Prop := y + 2 = 0

-- Define the problem statement as a theorem
theorem symmetric_line_equation :
  ∀ (x y : ℝ), line_of_symmetry x y → (original_line x (2 * (-2 - y) + 1)) ↔ (2 * x + y + 5 = 0) := 
sorry

end symmetric_line_equation_l597_597223


namespace sum_of_squares_of_rates_l597_597855

theorem sum_of_squares_of_rates (c j s : ℕ) (cond1 : 3 * c + 2 * j + 2 * s = 80) (cond2 : 2 * j + 2 * s + 4 * c = 104) : 
  c^2 + j^2 + s^2 = 592 :=
sorry

end sum_of_squares_of_rates_l597_597855


namespace solution_set_inequality_l597_597655

theorem solution_set_inequality (x : ℝ) : (x + 1) * (2 - x) < 0 ↔ x > 2 ∨ x < -1 :=
sorry

end solution_set_inequality_l597_597655


namespace exists_constants_for_function_bounded_l597_597173

noncomputable def f : ℝ+ → ℝ+ := sorry

axiom f_triangle_property (a b c : ℝ+) (h : a + b > c ∧ b + c > a ∧ c + a > b) :
  f(a) + f(b) > f(c) ∧ f(b) + f(c) > f(a) ∧ f(c) + f(a) > f(b)

theorem exists_constants_for_function_bounded (f : ℝ+ → ℝ+) 
  (h_f : ∀ a b c : ℝ+, (a + b > c ∧ b + c > a ∧ c + a > b) → (f(a) + f(b) > f(c) ∧ f(b) + f(c) > f(a) ∧ f(c) + f(a) > f(b))) :
  ∃ (A B : ℝ), (0 < A) ∧ (0 < B) ∧ (∀ x, 0 < x → f(x) ≤ A * x + B) :=
begin
  sorry
end

end exists_constants_for_function_bounded_l597_597173


namespace factorizing_definition_l597_597715

-- Definition of factorizing a polynomial
def factorize_polynomial (p : Polynomial ℤ) : Bool :=
  ∃ (factors : List (Polynomial ℤ)), p = factors.foldl (*) 1

-- Theorem to prove the definition matches the given condition
theorem factorizing_definition (p : Polynomial ℤ) :
  factorize_polynomial p ↔ p = ∏ factors, factors :=
sorry

end factorizing_definition_l597_597715


namespace volume_ratio_central_XT_l597_597772

-- Defining the geometrical properties and conditions of the problem
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)
  (length_AB_CD_eq : AB = 12)
  (length_BC_DA_eq : BC = 16)

structure Pyramid :=
  (base : Rectangle)
  (height : ℝ)
  (height_eq : height = 24)

structure PlaneParallelCut :=
  (original_pyramid : Pyramid)
  (frustum_volume_ratio : ℝ)
  (volume_ratio_eq : frustum_volume_ratio = 8)

structure CircumsphereCenter :=
  (frustum : PlaneParallelCut)
  (X : ℝ × ℝ × ℝ)
  (T : ℝ × ℝ × ℝ)

noncomputable def XT_distance : CircumsphereCenter → ℝ
| ⟨_, X, T⟩ := Real.sqrt ((X.1 - T.1) ^ 2 + (X.2 - T.2) ^ 2 + (X.3 - T.3) ^ 2)

theorem volume_ratio_central_XT (r : CircumsphereCenter) (m n : ℕ) :
  let XT := XT_distance r in
  m.gcd n = 1 →
  XT = (m : ℝ) / (n : ℝ) →
  m + n = 177 :=
sorry

end volume_ratio_central_XT_l597_597772


namespace last_two_digits_sum_is_32_l597_597722

-- Definitions for digit representation
variables (z a r l m : ℕ)

-- Numbers definitions
def ZARAZA := z * 10^5 + a * 10^4 + r * 10^3 + a * 10^2 + z * 10 + a
def ALMAZ := a * 10^4 + l * 10^3 + m * 10^2 + a * 10 + z

-- Condition that ZARAZA is divisible by 4
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Condition that ALMAZ is divisible by 28
def divisible_by_28 (n : ℕ) : Prop := n % 28 = 0

-- The theorem to prove
theorem last_two_digits_sum_is_32
  (hz4 : divisible_by_4 (ZARAZA z a r))
  (ha28 : divisible_by_28 (ALMAZ a l m z))
  : (ZARAZA z a r + ALMAZ a l m z) % 100 = 32 :=
by sorry

end last_two_digits_sum_is_32_l597_597722


namespace player_1_points_l597_597671

-- Definition: point distribution on the table.
noncomputable def sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

-- Conditions
axiom player_5_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(5 + i) % 16]) = 72
axiom player_9_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(9 + i) % 16]) = 84

-- Question translated to proof statement:
theorem player_1_points (rotations : ℕ) (p5_points : ℕ) (p9_points : ℕ) :
  rotations = 13 → p5_points = 72 → p9_points = 84 →
  ∑ i in finset.range rotations, (sector_points[(1 + i) % 16]) = 20 :=
by
  sorry

end player_1_points_l597_597671


namespace moving_circle_trajectory_l597_597448

-- Define the two given circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- The theorem statement
theorem moving_circle_trajectory :
  (∀ x y : ℝ, (exists r : ℝ, r > 0 ∧ ∃ M : ℝ × ℝ, 
  (C₁ M.1 M.2 ∧ ((M.1 - 4)^2 + M.2^2 = (13 - r)^2) ∧
  C₂ M.1 M.2 ∧ ((M.1 + 4)^2 + M.2^2 = (r + 3)^2)) ∧
  ((x = M.1) ∧ (y = M.2))) ↔ (x^2 / 64 + y^2 / 48 = 1)) := sorry

end moving_circle_trajectory_l597_597448


namespace smallest_number_of_cookies_proof_l597_597186

def satisfies_conditions (a : ℕ) : Prop :=
  (a % 6 = 5) ∧ (a % 8 = 6) ∧ (a % 10 = 9) ∧ (∃ n : ℕ, a = n * n)

def smallest_number_of_cookies : ℕ :=
  2549

theorem smallest_number_of_cookies_proof :
  satisfies_conditions smallest_number_of_cookies :=
by
  sorry

end smallest_number_of_cookies_proof_l597_597186


namespace find_zero_sum_set_l597_597519

theorem find_zero_sum_set (n : ℕ)
  (table : fin (2^n) → fin n → ℤ)
  (h1 : ∀ (i : fin (2^n)) (j : fin n), table i j = 1 ∨ table i j = -1 ∨ table i j = 0) :
  ∃ (subset : fin (2^n) → Bool), (∀ j, ∑ i in finset.univ.filter subset, table i j = 0) :=
by
  sorry

end find_zero_sum_set_l597_597519


namespace points_player_1_after_13_rotations_l597_597680

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end points_player_1_after_13_rotations_l597_597680


namespace problem1_problem2_l597_597035

variable {R : Type*} [OrderedField R]

variable (f : R → R)

-- Define conditions as hypotheses
axiom f_domain (x : R) : f x ≠ 0
axiom f_equation (x y : R) : f (x + y) + f (x - y) = 2 * f x * f y

-- Problem Statement 1: Prove that f(0) = 1
theorem problem1 : f 0 = 1 := by sorry

-- Problem Statement 2: Prove that f(x) is an even function
theorem problem2 : ∀ x : R, f (-x) = f x := by sorry

end problem1_problem2_l597_597035


namespace player_1_points_after_13_rotations_l597_597687

theorem player_1_points_after_13_rotations :
  ∀ (table : ℕ → ℕ) (n : ℕ) (points : List ℕ),
    (∀ i, table i+16 = table i) →
    (table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]) →
    (points.length = 16) →
    (points.get 5 = 72) →
    (points.get 9 = 84) →
    (n = 13) →
    ((points.sum i₁, 0, 13) table ((stats : List ℕ) (i : fin 16) =>
      List.sum (List.take stats.toList) i.val + 
      List.sum (List.drop stats.toList i.val i.val + 2 * n) table) = points.sum table) →
    points.get 1 = 20 :=
by
  intros
  sorry

end player_1_points_after_13_rotations_l597_597687


namespace max_k_value_l597_597171

open Finset

theorem max_k_value : 
  ∃ k (A : Fin k (Finset (Fin 10))), 
    (∀ i, card (A i) = 5) ∧ 
    (∀ i j, i ≠ j → card ((A i) ∩ (A j)) ≤ 2) ∧ 
    k = 6 :=
by
  sorry

end max_k_value_l597_597171


namespace sam_wins_l597_597556

variable (p : ℚ) -- p is the probability that Sam wins
variable (phit : ℚ) -- probability of hitting the target in one shot
variable (pmiss : ℚ) -- probability of missing the target in one shot

-- Define the problem and set up the conditions
def conditions : Prop := phit = 2 / 5 ∧ pmiss = 3 / 5

-- Define the equation derived from the problem
def equation (p : ℚ) (phit : ℚ) (pmiss : ℚ) : Prop :=
  p = phit + (pmiss * pmiss * p)

-- State the theorem that Sam wins with probability 5/8
theorem sam_wins (h : conditions phit pmiss) : 
  equation p phit pmiss → p = 5 / 8 :=
by
  intros
  sorry

end sam_wins_l597_597556


namespace min_students_in_class_correct_l597_597105

noncomputable def min_students_in_class : ℕ :=
  let b := 4
  let g := 3
  b + g

theorem min_students_in_class_correct :
  ∃ b g : ℕ, (1 / 2 : ℚ) * b = (2 / 3 : ℚ) * g ∧ (b + g = min_students_in_class) :=
by
  use 4, 3
  split
  · norm_num
  · norm_num

end min_students_in_class_correct_l597_597105


namespace sequence_term_1000_l597_597978

theorem sequence_term_1000 (a : ℕ → ℤ) 
  (h1 : a 1 = 2010) 
  (h2 : a 2 = 2011) 
  (h3 : ∀ n, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 1000 = 2676 :=
sorry

end sequence_term_1000_l597_597978


namespace exists_same_colored_points_at_distance_l597_597245

open Classical

noncomputable def point_colored_red_or_blue (p : ℝ × ℝ) : Prop :=
  p = ⟨x, y⟩ ∨ p = ⟨u, v⟩ -- Representing colors by point labels (x, y) for red and (u, v) for blue

theorem exists_same_colored_points_at_distance (x : ℝ) (hx : x > 0) :
  ∃ p q : ℝ × ℝ, (point_colored_red_or_blue p ∧ point_colored_red_or_blue q) ∧ dist p q = x :=
by
  sorry

end exists_same_colored_points_at_distance_l597_597245


namespace length_XY_l597_597487

-- Define the problem conditions
variables (O A B Y X : Point)
variable (r : ℝ)
variable (angleAOB : Angle)
variable (OAsy : r = 10)
variable (OBsy : Line O A B)
variable (OYsy : Perpendicular O Y X B)

axiom angle_condition : angleAOB = 90
axiom radius_condition : OA ^ 2 + OB ^ 2 = 100
axiom midpoint_condition : OX = 5 √ 2

-- Prove the question equals the answer given the conditions
theorem length_XY :
  (OY - OX = 10 - 5 * sqrt 2)
:= sorry

end length_XY_l597_597487


namespace equivalent_expression_l597_597810

theorem equivalent_expression (x y : ℝ) : (-x + 2y) * (-x - 2y) = x^2 - 4y^2 := by
  -- Adding the proof step to prevent Lean from throwing an error.
  sorry

end equivalent_expression_l597_597810


namespace sam_wins_probability_l597_597544

theorem sam_wins_probability : 
  let hit_prob := (2 : ℚ) / 5
      miss_prob := (3 : ℚ) / 5
      p := hit_prob + (miss_prob * miss_prob) * p
  in p = 5 / 8 := 
by
  -- Proof goes here
  sorry

end sam_wins_probability_l597_597544


namespace find_m_l597_597444

variable (m : ℝ)

def p := (m ≤ -2 ∨ m ≥ 2)

def q := (1 < m ∧ m < 3)

theorem find_m (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ q) : 1 < m ∧ m < 2 :=
  sorry

end find_m_l597_597444


namespace num_true_propositions_l597_597902

variables {l m : Type} [Line l] [Line m]
variables {α β : Type} [Plane α] [Plane β]

-- Assuming these propositions
def proposition1 := (l ∥ α) ∧ (m ∥ α) → (l ∥ m) = False
def proposition2 := (α ∥ l) ∧ (β ∥ l) → (α ∥ β) = False
def proposition3 := (α ⊥ l) ∧ (β ⊥ l) → (α ∥ β) = True
def proposition4 := (l ⊥ α) ∧ (m ⊥ α) → (l ∥ m) = True

-- Finally, we want to prove the number of true propositions is 2.
theorem num_true_propositions : 
  (if proposition1 then 1 else 0) + 
  (if proposition2 then 1 else 0) + 
  (if proposition3 then 1 else 0) + 
  (if proposition4 then 1 else 0) = 2 := 
sorry

end num_true_propositions_l597_597902


namespace unique_solution_is_2_or_minus_2_l597_597965

theorem unique_solution_is_2_or_minus_2 (a : ℝ) :
  (∃ x : ℝ, ∀ y : ℝ, (y^2 + a * y + 1 = 0 ↔ y = x)) → (a = 2 ∨ a = -2) :=
by sorry

end unique_solution_is_2_or_minus_2_l597_597965


namespace equivalent_problem_l597_597376

def divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def sum_of_digits (n : ℕ) : ℕ := 
  let digits := (nat.to_digits 10 n).map (λ x => x) in 
  digits.sum

def product_of_digits (n : ℕ) : ℕ :=
  let digits := (nat.to_digits 10 n).map (λ x => x) in
  digits.prod

theorem equivalent_problem
  (n : ℕ) : divisible_by_8 n ∧ sum_of_digits n = 7 ∧ product_of_digits n = 6 ↔ (n = 1312 ∨ n = 3112) :=
sorry

end equivalent_problem_l597_597376


namespace maximum_value_F_l597_597067

noncomputable def f (x : Real) : Real := Real.sin x + Real.cos x

noncomputable def f' (x : Real) : Real := Real.cos x - Real.sin x

noncomputable def F (x : Real) : Real := f x * f' x + (f x) ^ 2

theorem maximum_value_F : ∃ x : Real, F x = 1 + Real.sqrt 2 :=
by
  -- The proof steps are to be added here.
  sorry

end maximum_value_F_l597_597067


namespace total_legs_at_pet_shop_l597_597819

theorem total_legs_at_pet_shop : 
  let birds := 3 in
  let dogs := 5 in
  let snakes := 4 in
  let spiders := 1 in
  let legs_bird := 2 in
  let legs_dog := 4 in
  let legs_snake := 0 in
  let legs_spider := 8 in
  (birds * legs_bird + dogs * legs_dog + snakes * legs_snake + spiders * legs_spider) = 34 :=
by 
  -- Proof will be here
  sorry

end total_legs_at_pet_shop_l597_597819


namespace teachers_no_conditions_percentage_l597_597801

theorem teachers_no_conditions_percentage :
  let total_teachers := 150
  let high_blood_pressure := 90
  let heart_trouble := 60
  let both_hbp_ht := 30
  let diabetes := 10
  let both_diabetes_ht := 5
  let both_diabetes_hbp := 8
  let all_three := 3

  let only_hbp := high_blood_pressure - both_hbp_ht - both_diabetes_hbp - all_three
  let only_ht := heart_trouble - both_hbp_ht - both_diabetes_ht - all_three
  let only_diabetes := diabetes - both_diabetes_hbp - both_diabetes_ht - all_three
  let both_hbp_ht_only := both_hbp_ht - all_three
  let both_hbp_diabetes_only := both_diabetes_hbp - all_three
  let both_ht_diabetes_only := both_diabetes_ht - all_three
  let any_condition := only_hbp + only_ht + only_diabetes + both_hbp_ht_only + both_hbp_diabetes_only + both_ht_diabetes_only + all_three
  let no_conditions := total_teachers - any_condition

  (no_conditions / total_teachers * 100) = 28 :=
by
  sorry

end teachers_no_conditions_percentage_l597_597801


namespace probability_odd_product_l597_597569

def range_integers : Finset ℤ := Finset.range 13 + 4

def odd_integers (s : Finset ℤ) : Finset ℤ :=
  s.filter (λ n, n % 2 = 1)

def num_ways_choose_three (n : ℕ) : ℕ :=
  nat.choose n 3

theorem probability_odd_product :
  let total_integers := range_integers.card
  let total_ways := num_ways_choose_three total_integers
  let odd_int_set := odd_integers range_integers
  let odd_ways := num_ways_choose_three odd_int_set.card
  total_ways ≠ 0 →
  odd_ways / total_ways = 10 / 143 :=
by
  sorry

end probability_odd_product_l597_597569


namespace extreme_points_values_l597_597052

noncomputable def a : ℝ := sorry
def f (x : ℝ) : ℝ := x * (Real.log x - a * x)
def f' (x : ℝ) := Real.log x + 1 - 2 * a * x
def g (x : ℝ) := Real.log x + 1 - 2 * a * x
def g' (x : ℝ) := (1 - 2 * a * x) / x

theorem extreme_points_values (h1: a > 0) (h2: a < 1 / 2)
    (hx1: ∃ x1 x2, x1 < x2 ∧ f' x1 = 0 ∧ f' x2 = 0) :
    ∃ x1 x2, x1 < x2 ∧ f x1 < 0 ∧ f x2 > -1 / 2 :=
sorry

end extreme_points_values_l597_597052


namespace primary_school_capacity_l597_597479

variable (x : ℝ)

/-- In a town, there are four primary schools. Two of them can teach 400 students at a time, 
and the other two can teach a certain number of students at a time. These four primary schools 
can teach a total of 1480 students at a time. -/
theorem primary_school_capacity 
  (h1 : 2 * 400 + 2 * x = 1480) : 
  x = 340 :=
sorry

end primary_school_capacity_l597_597479


namespace hexagonal_frustum_implies_hexagonal_pyramid_l597_597782

theorem hexagonal_frustum_implies_hexagonal_pyramid (G : Type) [geometric_body G] (h : cuts_parallel_to_base G = hexagonal_frustum) : 
  G = hexagonal_pyramid :=
begin
  sorry
end

end hexagonal_frustum_implies_hexagonal_pyramid_l597_597782


namespace polynomial_relation_l597_597409

variables {a b c : ℝ}

theorem polynomial_relation
  (h1: a ≠ 0) (h2: b ≠ 0) (h3: c ≠ 0) (h4: a + b + c = 0) :
  ((a^7 + b^7 + c^7)^2) / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49 / 60 :=
sorry

end polynomial_relation_l597_597409


namespace corveus_sleep_hours_l597_597361

-- Definition of the recommended hours of sleep per day
def recommended_sleep_per_day : ℕ := 6

-- Definition of the hours of sleep Corveus lacks per week
def lacking_sleep_per_week : ℕ := 14

-- Definition of days in a week
def days_in_week : ℕ := 7

-- Prove that Corveus sleeps 4 hours per day given the conditions
theorem corveus_sleep_hours :
  (recommended_sleep_per_day * days_in_week - lacking_sleep_per_week) / days_in_week = 4 :=
by
  -- The proof steps would go here
  sorry

end corveus_sleep_hours_l597_597361


namespace find_x_l597_597660

theorem find_x (α : ℝ) (x : ℝ) (h1 : sin(α) = 4 / 5) (h2 : sqrt (x ^ 2 + 16) ≠ 0) : (x = 3) ∨ (x = -3) :=
by
  have h3 : 4 / sqrt (x ^ 2 + 16) = 4 / 5 := h1
  have h4 : sqrt (x ^ 2 + 16) = 5 := by linarith
  have h5 : x ^ 2 + 16 = 25 := by linarith
  have h6 : x ^ 2 = 9 := by linarith
  exact or.inl (eq_of_sq_eq_sq _ h6).left sorry

end find_x_l597_597660


namespace repeat_block_of_7_div_13_l597_597633

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l597_597633


namespace ce_length_l597_597152

theorem ce_length (A B C D E F : Type) 
  (AB CD BF AE : ℝ) 
  (hAB : AB = 5) 
  (hCD : CD = 10) 
  (hBF : BF = 3) 
  (hAE : AE = 6) 
  (line_AE : set ℝ) 
  (hD : D ∈ line_AE)
  (hCD_perp_AE : CD ⊥ AE)
  (line_CE : set ℝ)
  (hB : B ∈ line_CE)
  (hAB_perp_CE : AB ⊥ CE)
  (hBF_perp_AE : BF ⊥ AE) :
    ∃ CE : ℝ, CE = 12 := 
sorry

end ce_length_l597_597152


namespace product_sequence_eq_l597_597826

theorem product_sequence_eq : (∏ k in Finset.range 501, (4 * (k + 1)) / ((4 * (k + 1)) + 4)) = (1 : ℚ) / 502 := 
sorry

end product_sequence_eq_l597_597826


namespace prime_factors_count_l597_597082

theorem prime_factors_count :
  (∃ a b c d : ℕ, a = 95 ∧ b = 97 ∧ c = 99 ∧ d = 101 ∧
   ∃ p q r s : ℕ, 95 = 5 * 19 ∧ prime 97 ∧ 99 = 3^2 * 11 ∧ prime 101) →
  ∃ n : ℕ, n = 6 :=
by sorry

end prime_factors_count_l597_597082


namespace player_1_points_l597_597670

-- Definition: point distribution on the table.
noncomputable def sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

-- Conditions
axiom player_5_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(5 + i) % 16]) = 72
axiom player_9_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(9 + i) % 16]) = 84

-- Question translated to proof statement:
theorem player_1_points (rotations : ℕ) (p5_points : ℕ) (p9_points : ℕ) :
  rotations = 13 → p5_points = 72 → p9_points = 84 →
  ∑ i in finset.range rotations, (sector_points[(1 + i) % 16]) = 20 :=
by
  sorry

end player_1_points_l597_597670


namespace player_1_points_after_13_rotations_l597_597678

-- Add necessary definitions and state the problem in Lean
def sectors : Fin 16 → ℕ
| ⟨0, _⟩ := 0
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 2
| ⟨3, _⟩ := 3
| ⟨4, _⟩ := 4
| ⟨5, _⟩ := 5
| ⟨6, _⟩ := 6
| ⟨7, _⟩ := 7
| ⟨8, _⟩ := 8
| ⟨9, _⟩ := 7
| ⟨10, _⟩ := 6
| ⟨11, _⟩ := 5
| ⟨12, _⟩ := 4
| ⟨13, _⟩ := 3
| ⟨14, _⟩ := 2
| ⟨15, _⟩ := 1

def points_earned (player_offset : Fin 16) (rotations : ℕ) : ℕ :=
List.sum (List.map sectors
  (List.map (λ n => (Fin.add (Fin.ofNat n) player_offset)) (List.range rotations)))

theorem player_1_points_after_13_rotations 
  (p5_points : points_earned ⟨5, by decide⟩ 13 = 72)
  (p9_points : points_earned ⟨9, by decide⟩ 13 = 84) :
  points_earned ⟨1, by decide⟩ 13 = 20 := 
sorry

end player_1_points_after_13_rotations_l597_597678


namespace cousins_arrangement_ways_l597_597996

theorem cousins_arrangement_ways : 
  let cousins := 5 
  let rooms := 4
  let arrangements := [(5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0), (2,1,1,1)]
  let ways := λ (distribution : (ℕ × ℕ × ℕ × ℕ)), match distribution with
    | (5,0,0,0) => 1
    | (4,1,0,0) => nat.choose 5 1
    | (3,2,0,0) => nat.choose 5 2
    | (3,1,1,0) => nat.choose 5 3
    | (2,2,1,0) => (nat.choose 5 2) * (nat.choose 3 2) / (nat.choose 4 2)
    | (2,1,1,1) => nat.choose 5 2
    | _ => 0 -- Default case which should ideally never occur given valid distributions
  in
  (arrangements.map ways).sum = 51 := sorry

end cousins_arrangement_ways_l597_597996


namespace particle_intersects_sphere_l597_597480

-- Define the starting and ending points
def start := (1 : ℝ, 2, 3)
def end := (-2 : ℝ, -2, -4)

-- Define the center of the sphere and its radius
def center := (1 : ℝ, 1, 1)
def radius := 2

-- Function to express the parametrized line
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 - 3 * t, 2 - 4 * t, 3 - 7 * t)

-- Function to express the distance
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem particle_intersects_sphere :
  ∃ t1 t2 : ℝ, distance (line t1) (line t2) = 2 * real.sqrt 197 / 37 :=
by
  sorry

end particle_intersects_sphere_l597_597480


namespace min_guards_proof_l597_597706

def shift := ℕ

constant day_shift : shift := 1
constant night_shift : shift := 1.5
constant continuous_shift : shift := 2.5

def min_guards_needed (day_shift night_shift continuous_shift : shift) : ℕ :=
  if day_shift = 1 ∧ night_shift = 1.5 ∧ continuous_shift = 2.5 then 4 else 0

theorem min_guards_proof : min_guards_needed day_shift night_shift continuous_shift = 4 :=
sorry

end min_guards_proof_l597_597706


namespace sufficient_but_not_necessary_l597_597952

theorem sufficient_but_not_necessary (a : ℝ) : (a = 2 → |a| = 2) ∧ (¬ (|a| = 2 → a = 2)) := by
  sorry

end sufficient_but_not_necessary_l597_597952


namespace num_solutions_l597_597089

theorem num_solutions (S P : Finset ℕ) 
  (hS : S = Finset.Icc 1 144) 
  (hP : P = Finset.image (λ k, k * k) (Finset.Icc 1 12)) : 
  S.card - P.card = 132 :=
by
  -- hS and hP are the conditions given in the problem
  -- We need only the statement
  sorry

end num_solutions_l597_597089


namespace find_original_number_l597_597527

-- Let x be the original number
def maria_operations (x : ℤ) : Prop :=
  (3 * (x - 3) + 3) / 3 = 10

theorem find_original_number (x : ℤ) (h : maria_operations x) : x = 12 :=
by
  sorry

end find_original_number_l597_597527


namespace sin_identity_l597_597537

theorem sin_identity (a b c : ℝ) : 
  (sin (a - b) / (sin a * sin b)) + (sin (b - c) / (sin b * sin c)) + (sin (c - a) / (sin c * sin a)) = 0 :=
by
  sorry

end sin_identity_l597_597537


namespace number_of_pairs_x_y_l597_597942

theorem number_of_pairs_x_y (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - y^2 = 85) : 
    (1 : ℕ) + (1 : ℕ) = 2 := 
by 
  sorry

end number_of_pairs_x_y_l597_597942


namespace area_enclosed_by_curve_l597_597216

theorem area_enclosed_by_curve :
  ∫ (x : ℝ) in 1..2, (x^2 - 1) = 4 / 3 :=
by
  sorry

end area_enclosed_by_curve_l597_597216


namespace calc_cos2α_l597_597905

-- Define the necessary constants and variables
variable (α : ℝ)

-- Specify the given condition as a hypothesis
theorem calc_cos2α (h : sin (α - (3 / 2) * real.pi) = 3 / 5) : cos (2 * α) = -7 / 25 :=
sorry

end calc_cos2α_l597_597905


namespace find_y_l597_597962

theorem find_y (x y : ℤ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 :=
sorry

end find_y_l597_597962


namespace find_piles_l597_597185

theorem find_piles :
  ∃ N : ℕ, 
  (1000 < N ∧ N < 2000) ∧ 
  (N % 2 = 1) ∧ (N % 3 = 1) ∧ (N % 4 = 1) ∧ 
  (N % 5 = 1) ∧ (N % 6 = 1) ∧ (N % 7 = 1) ∧ (N % 8 = 1) ∧ 
  (∃ p : ℕ, p = 41 ∧ p > 1 ∧ p < N ∧ N % p = 0) :=
sorry

end find_piles_l597_597185


namespace max_remaining_area_l597_597038

-- Define the properties of the right triangle and the circle
variable (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Hypotenuse BC is 2π
def hypotenuse_eq : x^2 + y^2 = (2 * Real.pi)^2 := sorry

-- Radius r of circle radius centered at A tangent to BC
def radius_eq (r : ℝ) : r = (x * y) / (2 * Real.pi) := sorry

-- Area of triangle ABC
def area_triangle (xy : ℝ) : ℝ := 0.5 * xy

-- Area covered by the circle
def area_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

-- Remaining area S
def remaining_area (xy r : ℝ) : ℝ := area_triangle xy - area_circle r

-- Prove the maximum remaining area
theorem max_remaining_area : ∃ (x y : ℝ) (r : ℝ), remaining_area (x * y) r = Real.pi :=
by
  -- Note: proof skipped, just providing statement
  sorry

end max_remaining_area_l597_597038


namespace inverse_proportion_point_l597_597932

theorem inverse_proportion_point (
  k : ℝ,
  hk : k ≠ 0,
  h_inc : ∀ x y : ℝ, (y = k / x) → x^2 < y^2
) :
  (-2, 3) ∈ { (x, y) | y = k / x } ∧ k < 0 :=
by
  sorry

end inverse_proportion_point_l597_597932


namespace oranges_per_sack_l597_597941

theorem oranges_per_sack (harvested_sacks discarded_sacks oranges_per_day : ℕ) 
  (h1 : harvested_sacks = 76) 
  (h2 : discarded_sacks = 64) 
  (h3 : oranges_per_day = 600) :
  oranges_per_day / (harvested_sacks - discarded_sacks) = 50 :=
by
  rw [h1, h2, h3]
  simp
  sorry

end oranges_per_sack_l597_597941


namespace determine_fathers_l597_597771

def Isabelle_age := Nat
def Jean_age := Nat
def Eve_age := Nat
def Teresa_father := String
def Marie_preference := String
def Catherine_father := String
def Anna_relationship := String
def Jean_statement := Nat
def François_statement := Nat

axiom Isabelle_statement : Isabelle_age = Jean_age + 3
axiom Teresa_statement : Teresa_father = "Jacques"
axiom Eve_statement : Eve_age = Isabelle_age + 2
axiom Marie_statement : Marie_preference = "cousin"
axiom Catherine_statement : Catherine_father = "Pierre"
axiom Anna_statement : Anna_relationship = "sons of Uncle Jacques"
axiom Jean_statement : Jean_statement < 4
axiom François_statement : ∀ (P: Prop), P → ¬P

theorem determine_fathers:
  (Pierre_children = ["Isabelle", "Catherine", "Anna"]) ∧
  (Jacques_children = ["Teresa", "Yves", "Jean"]) ∧
  (Paul_children = ["François", "Marie"]) :=
sorry

end determine_fathers_l597_597771


namespace ratio_red_green_socks_l597_597183

variables (g y : ℝ)
def original_cost (g y : ℝ) : ℝ := 15 * y + g * y
def interchanged_cost (g y : ℝ) : ℝ := 3 * g * y + 5 * y

theorem ratio_red_green_socks :
  let g := (22 / 1.2) in
  (original_cost g y, interchanged_cost g y, 1.8) →
  g = 18.33 → 
  5 / g = 5 / 18 :=
by sorry

end ratio_red_green_socks_l597_597183


namespace cos_angle_sum_l597_597956

theorem cos_angle_sum (θ : ℝ) (hcos : cos θ = -12 / 13) (hθ : θ ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) :
  cos (θ + Real.pi / 4) = -7 * Real.sqrt 2 / 26 :=
by
  sorry

end cos_angle_sum_l597_597956


namespace transformed_circle_eq_l597_597427

theorem transformed_circle_eq (x y : ℝ) (h : x^2 + y^2 = 1) : x^2 + 9 * (y / 3)^2 = 1 := by
  sorry

end transformed_circle_eq_l597_597427


namespace good_bad_numbers_l597_597938

theorem good_bad_numbers
  (r s : ℕ) (h_rel_prime : Nat.gcd r s = 1) :
  let c := r * s - r - s in
  (∀ k : ℤ, (∃ (m n : ℕ), k = m * r + n * s) ↔ ¬(∃ (m n : ℕ), (c - k) = m * r + n * s)) ∧
  ((r - 1) * (s - 1)) / 2 = (c - 1) / 2 :=
by
  sorry

end good_bad_numbers_l597_597938


namespace log_inverse_base_change_l597_597461

theorem log_inverse_base_change (x : ℝ) 
  (h : log 16 (x - 3) = (1 / 2)) : 
  (1 / log x 2) = (1 / log 7 2) :=
sorry

end log_inverse_base_change_l597_597461


namespace number_of_choices_l597_597774

theorem number_of_choices (students : ℕ) (lectures : ℕ) (h_students : students = 5) (h_lectures : lectures = 4) :
  (lectures ^ students) = 4 ^ 5 :=
by
  rw [h_students, h_lectures]
  rfl

end number_of_choices_l597_597774


namespace circle_center_polar_coords_l597_597118

noncomputable def polar_center (ρ θ : ℝ) : (ℝ × ℝ) :=
  (-1, 0)

theorem circle_center_polar_coords : 
  ∀ ρ θ : ℝ, ρ = -2 * Real.cos θ → polar_center ρ θ = (1, π) :=
by
  intro ρ θ h
  sorry

end circle_center_polar_coords_l597_597118


namespace angle_ADE_60_l597_597104

-- Defining the geometric context
variables {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]

-- Conditions given in the problem
variables (triangle_ABC : Triangle A B C)
variables (D_on_AC : LiesOn D (LineSeg A C))
variables (E_on_AB : LiesOn E (LineSeg A B))
variables (BD_eq_DC : SegmentEq (LineSeg B D) (LineSeg D C))
variables (BE_eq_EA : SegmentEq (LineSeg B E) (LineSeg E A))
variables (angle_BCD : Angle (LineSeg B C) (LineSeg C D) = 60)

-- Question: what is the angle measure of ADE
theorem angle_ADE_60 : Angle (LineSeg A D) (LineSeg D E) = 60 := sorry

end angle_ADE_60_l597_597104


namespace board_division_condition_l597_597015

open Nat

theorem board_division_condition (n : ℕ) : 
  (∃ k : ℕ, n = 4 * k) ↔ 
  (∃ v h : ℕ, v = h ∧ (2 * v + 2 * h = n * n ∧ n % 2 = 0)) := 
sorry

end board_division_condition_l597_597015


namespace repeating_block_length_7_div_13_l597_597615

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597615


namespace range_of_f_l597_597653

noncomputable def f (x : ℝ) : ℝ := sqrt (-x^2 + 4 * x + 2)

theorem range_of_f :
  ∀ y : ℝ, y ∈ set.range f ↔ 0 ≤ y ∧ y ≤ sqrt 6 :=
by sorry

end range_of_f_l597_597653


namespace smallest_prime_with_reversed_composite_l597_597875

-- Define what it means for a number to be prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define what it means for a number to be composite.
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

-- Define a function to reverse the digits of a number.
def reverse_digits (n : ℕ) : ℕ :=
  n.toString.reverse.toNat

-- Define the proof problem: proving 103 meets the conditions.
theorem smallest_prime_with_reversed_composite :
  ∃ p : ℕ, 100 ≤ p ∧ p < 1000 ∧ is_prime p ∧ is_composite (reverse_digits p) ∧
  (∀ q : ℕ, 100 ≤ q ∧ q < p ∧ is_prime q → is_composite (reverse_digits q) → false) ∧ p = 103 :=
by
  sorry

end smallest_prime_with_reversed_composite_l597_597875


namespace ratio_dog_to_hamster_l597_597707

noncomputable def dog_lifespan : ℝ := 10
noncomputable def hamster_lifespan : ℝ := 2.5

theorem ratio_dog_to_hamster : dog_lifespan / hamster_lifespan = 4 :=
by
  sorry

end ratio_dog_to_hamster_l597_597707


namespace overall_loss_percentage_is_8point67_l597_597318

-- Define the cost prices of the items.
def cp_radio := 1800
def cp_mobile := 4200
def cp_camera := 7500

-- Define the selling prices of the items.
def sp_radio := 1430
def sp_mobile := 3800
def sp_camera := 7100

-- Define the total cost price.
def total_cp := cp_radio + cp_mobile + cp_camera

-- Define the total selling price.
def total_sp := sp_radio + sp_mobile + sp_camera

-- Define the total loss.
def total_loss := total_cp - total_sp

-- Define the loss percentage.
def loss_percentage := (total_loss.to_float / total_cp.to_float) * 100

-- Prove that the overall loss percentage is 8.67.
theorem overall_loss_percentage_is_8point67 :
  loss_percentage = 8.67 := by
  sorry

end overall_loss_percentage_is_8point67_l597_597318


namespace player_1_points_l597_597672

-- Definition: point distribution on the table.
noncomputable def sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

-- Conditions
axiom player_5_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(5 + i) % 16]) = 72
axiom player_9_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(9 + i) % 16]) = 84

-- Question translated to proof statement:
theorem player_1_points (rotations : ℕ) (p5_points : ℕ) (p9_points : ℕ) :
  rotations = 13 → p5_points = 72 → p9_points = 84 →
  ∑ i in finset.range rotations, (sector_points[(1 + i) % 16]) = 20 :=
by
  sorry

end player_1_points_l597_597672


namespace solve_for_A_l597_597382

theorem solve_for_A (A : ℚ) : 80 - (5 - (6 + A * (7 - 8 - 5))) = 89 → A = -4/3 :=
by
  sorry

end solve_for_A_l597_597382


namespace probability_two_dice_show_1_l597_597861

noncomputable def prob_exactly_two_ones : Real :=
  (Nat.choose 15 2) * (1 / 6) ^ 2 * (5 / 6) ^ 13

theorem probability_two_dice_show_1 :
  (Float.round (10^3 * prob_exactly_two_ones) / 10^3) = 0.196 :=
by
  sorry

end probability_two_dice_show_1_l597_597861


namespace number_of_correct_statements_is_1_l597_597328

theorem number_of_correct_statements_is_1 :
  (∀ (α β : inscribed_angle), same_or_congruent_circles α β → α = β → arcs_equal α β ↔ false)
  ∧ (∀ (A B C : Point), ¬collinear A B C → ∃ (O : Circle), A ∈ O ∧ B ∈ O ∧ C ∈ O ↔ false)
  ∧ (∀ (T : Triangle), is_isosceles_right T → circumcenter T ∈ angle_bisector (vertex_angle T) ↔ true)
  ∧ (∀ (T : Triangle), is_equilateral T → 
                      ∀ (I : Point), I = incenter T → 
                                     ∀ (V : Vertex), distance I V = distance I (next_vertex T V)) 
  → 1 := 
begin
  sorry
end

end number_of_correct_statements_is_1_l597_597328


namespace find_x_l597_597808

-- Definitions of variables and conditions
variables (A V R x : ℕ)
variables (h1 : A - V = -2 * x)
variables (h2 : V - A = 4 * x - 30)
variables (h3 : A + V + R = 120)

-- Lean statement for the proof problem
theorem find_x : x = 15 :=
by
  sorry

end find_x_l597_597808


namespace value_of_3m_2n_l597_597970

section ProofProblem

variable (m n : ℤ)
-- Condition that x-3 is a factor of 3x^3 - mx + n
def factor1 : Prop := (3 * 3^3 - m * 3 + n = 0)
-- Condition that x+4 is a factor of 3x^3 - mx + n
def factor2 : Prop := (3 * (-4)^3 - m * (-4) + n = 0)

theorem value_of_3m_2n (h₁ : factor1 m n) (h₂ : factor2 m n) : abs (3 * m - 2 * n) = 45 := by
  sorry

end ProofProblem

end value_of_3m_2n_l597_597970


namespace minimum_value_is_8_l597_597166

noncomputable def minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_is_8 :
  ∃ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), minimum_value x y hx hy = 8 :=
by
  sorry

end minimum_value_is_8_l597_597166


namespace tan_beta_is_three_l597_597908

theorem tan_beta_is_three (α β : ℝ)
  (hβ1 : 0 < β ∧ β < π / 2) 
  (hα1 : cos α = 1 / real.sqrt 5 ∧ sin α = 2 / real.sqrt 5)
  (h_sin_alpha_beta : real.sin (α + β) = real.sqrt 2 / 2) : 
  real.tan β = 3 :=
by {
  sorry
}

end tan_beta_is_three_l597_597908


namespace regular_octagon_ratio_l597_597107

noncomputable def ratio_of_side_to_diagonal (s d : ℝ) : ℝ := s / d

theorem regular_octagon_ratio (s d : ℝ) (h1 : 0 < s) (h2 : d = s * real.sqrt (2 - real.sqrt 2)) :
  ratio_of_side_to_diagonal s d = 1 / real.sqrt (2 - real.sqrt 2) :=
by
  -- Proof goes here
  sorry

end regular_octagon_ratio_l597_597107


namespace honzik_birthday_l597_597452

theorem honzik_birthday :
  ∀ (start_date : ℕ) (end_date : ℕ) (total_pages : ℕ) (pages_read : ℕ) (extra_pages_per_day : ℕ),
      start_date = 24 ∧ end_date = 31 ∧ total_pages = 3 * pages_read ∧ pages_read = 78 ∧ extra_pages_per_day = 4 →
      (∃ (birthday_date : ℕ), birthday_date = end_date + 1 + (total_pages - pages_read) / (pages_read / (start_date + 7 + end_date) + extra_pages_per_day) - 1 ∧ birthday_date = 26) := 
by
  intro start_date end_date total_pages pages_read extra_pages_per_day
  intro hypothesis
  cases hypothesis with hs1 hypothesis
  cases hypothesis with he1 hypothesis
  cases hypothesis with ht1 hypothesis
  cases hypothesis with hp1 he2

  -- Define the conditions as mentioned
  have hs : start_date = 24 := hs1
  have he : end_date = 31 := he1
  have ht : total_pages = 3 * pages_read := ht1
  have hp : pages_read = 78 := hp1
  have hexp : extra_pages_per_day = 4 := he2 

  -- Set dates and steps
  let reading_days := 8 + 31
  let pages_per_day := pages_read / reading_days
  let new_rate := pages_per_day + extra_pages_per_day
  let remaining_pages := total_pages - pages_read
  let days_to_finish := remaining_pages / new_rate
  let birthday := end_date + 1 + days_to_finish - 1

  use birthday
  split
  · refl -- Birthday should be February 26 given all the calculations
  · refl -- This is our final conclusion: Honzík's birthday is February 26
  sorry -- proof will be completed here

end honzik_birthday_l597_597452


namespace track_length_l597_597495

variable {x : ℕ}

-- Conditions
def runs_distance_jacob (x : ℕ) := 120
def runs_distance_liz (x : ℕ) := (x / 2 - 120)

def runs_second_meeting_jacob (x : ℕ) := x + 120 -- Jacob's total distance by second meeting
def runs_second_meeting_liz (x : ℕ) := (x / 2 + 60) -- Liz's total distance by second meeting

-- The relationship is simplified into the final correct answer
theorem track_length (h1 : 120 / (x / 2 - 120) = (x / 2 + 60) / 180) :
  x = 340 := 
sorry

end track_length_l597_597495


namespace IncorrectOption_l597_597489

namespace Experiment

def OptionA : Prop := 
  ∃ method : String, method = "sampling detection"

def OptionB : Prop := 
  ¬(∃ experiment : String, experiment = "does not need a control group, nor repeated experiments")

def OptionC : Prop := 
  ∃ action : String, action = "test tube should be gently shaken"

def OptionD : Prop := 
  ∃ condition : String, condition = "field of view should not be too bright"

theorem IncorrectOption : OptionB :=
  sorry

end Experiment

end IncorrectOption_l597_597489


namespace Sam_wins_probability_l597_597547

-- Define the basic probabilities
def prob_hit : ℚ := 2 / 5
def prob_miss : ℚ := 3 / 5

-- Define the desired probability that Sam wins
noncomputable def p : ℚ := 5 / 8

-- The mathematical problem statement in Lean
theorem Sam_wins_probability :
  p = prob_hit + (prob_miss * prob_miss * p) := 
sorry

end Sam_wins_probability_l597_597547


namespace correct_statements_2_and_5_l597_597273

theorem correct_statements_2_and_5 :
  let is_equal_vectors (v1 v2 : Vector) : Prop := v1.magnitude = v2.magnitude ∧ v1.direction = v2.direction
  let length_zero_vector_is_zero : Prop := ∀ (v : Vector), v = 0 → v.length = 0
  let collinear_vectors_not_same_straight_line : Prop :=
      ∀ (v1 v2 : Vector), v1.collinear v2 → v1.direction = v2.direction ∨ v1.direction = -v2.direction
  let zero_vector_direction_arbitrary : Prop := ∀ (v : Vector), v = 0 → v.direction = arbitrary
  let collinear_vectors_not_equal : Prop := ∀ (v1 v2 : Vector), v1.collinear v2 → ¬ (is_equal_vectors v1 v2)
  let parallel_vectors_same_opposite_direction : Prop :=
      ∀ (v1 v2 : Vector), v1.parallel v2 → v1.direction = v2.direction ∨ v1.direction = -v2.direction
  (length_zero_vector_is_zero ∧ collinear_vectors_not_equal) :=
by {
  sorry
}

end correct_statements_2_and_5_l597_597273


namespace find_b_l597_597464

noncomputable def complex_b_value (i : ℂ) (b : ℝ) : Prop :=
(1 + b * i) * i = 1 + i

theorem find_b (i : ℂ) (b : ℝ) (hi : i^2 = -1) (h : complex_b_value i b) : b = -1 :=
by {
  sorry
}

end find_b_l597_597464


namespace Sn_value_l597_597490

def seq := ℕ → ℕ
def a : seq
| 1       := 1
| (n + 1) := 2 * (a n)

def S (a : seq) (n : ℕ) : ℕ :=
∑ i in range (2 * n), if even i then (-1)^i * (a (i + 1))^2 else 0

theorem Sn_value (n : ℕ) : S a n = (1/5) * (1 - 2^(4*n)) := 
by sorry

end Sn_value_l597_597490


namespace minimum_perimeter_triangle_through_point_l597_597269

theorem minimum_perimeter_triangle_through_point :
  ∃ (a b : ℝ), 
    a = (2 * b) / (b + 1) ∧
    let AB := b * real.sqrt (1 + 4 / (b + 1) ^ 2) in
    (O : ℝ × ℝ) = (0, 0) ∧
    (A : ℝ × ℝ) = (0, a) ∧
    (B : ℝ × ℝ) = (b, 0) ∧
    ∃ (min_perimeter : ℝ), 
      min_perimeter = a + b + AB ∧
      min_perimeter = 3 + 2 * real.sqrt 2 + real.sqrt 3 + real.sqrt 6 :=
sorry

end minimum_perimeter_triangle_through_point_l597_597269


namespace min_flash_drives_needed_l597_597838

theorem min_flash_drives_needed (total_files : ℕ) (capacity_per_drive : ℝ)  
  (num_files_0_9 : ℕ) (size_0_9 : ℝ) 
  (num_files_0_8 : ℕ) (size_0_8 : ℝ) 
  (size_0_6 : ℝ) 
  (remaining_files : ℕ) :
  total_files = 40 →
  capacity_per_drive = 2.88 →
  num_files_0_9 = 5 →
  size_0_9 = 0.9 →
  num_files_0_8 = 18 →
  size_0_8 = 0.8 →
  remaining_files = total_files - (num_files_0_9 + num_files_0_8) →
  size_0_6 = 0.6 →
  (num_files_0_9 * size_0_9 + num_files_0_8 * size_0_8 + remaining_files * size_0_6) / capacity_per_drive ≤ 13 :=
by {
  sorry
}

end min_flash_drives_needed_l597_597838


namespace min_value_and_period_of_f_find_a_and_b_l597_597065

def f (x : ℝ) : ℝ := (√3 * sin x * cos x) - (cos x)^2 - (1/2)

theorem min_value_and_period_of_f : 
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f(x + p) = f x) ∧ 
  (∀ x : ℝ, f x ≥ -2) ∧ 
  (∃ x : ℝ, f x = -2) :=
by
  sorry

variable {A B C : ℝ}
variables {a b c : ℝ} (h1 : c = 3) (h2 : f C = 0)
variables {m n : ℝ × ℝ} (hm : m = ⟨1, sin A⟩) (hn : n = ⟨2, sin B⟩) 
(h_collinear : ∃ k : ℝ, n = k • m)

theorem find_a_and_b : 
  ∃ (a b : ℝ), a = √3 ∧ b = 2 * √3 :=
by
  sorry

end min_value_and_period_of_f_find_a_and_b_l597_597065


namespace repeating_block_length_of_7_div_13_is_6_l597_597603

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l597_597603


namespace sum_smallest_angles_l597_597843

noncomputable def Q (x : ℂ) : ℂ := (∑ i in finset.range 20, x ^ i) ^ 2 - x ^ 19

theorem sum_smallest_angles : 
  let roots := {z : ℂ | z ≠ 1 ∧ (z ^ 19 = 1 ∨ z ^ 21 = 1) ∧ ∃ r a, z = r * (complex.exp (2 * real.pi * complex.I * a)) ∧ 0 < a ∧ a < 1} in
  let angles := finset.sort (≤) ((roots.image (λ z, complex.arg z / (2 * real.pi)))).to_finset.to_list in
  ∑ i in finset.range 5, angles[i] = 183 / 399 :=
sorry

end sum_smallest_angles_l597_597843


namespace number_of_rectangles_in_5x5_grid_l597_597724

theorem number_of_rectangles_in_5x5_grid : 
  (∑ i in Finset.range 6, i^3) = 225 := 
by 
  sorry

end number_of_rectangles_in_5x5_grid_l597_597724


namespace choose_courses_l597_597794

theorem choose_courses :
  let A := 3
  let B := 4
  let total := 3
  (∃ (A_chosen B_chosen : ℕ), 
    (1 ≤ A_chosen ∧ A_chosen ≤ A) ∧ (1 ≤ B_chosen ∧ B_chosen ≤ B) ∧ A_chosen + B_chosen = total) →
  (nat.choose A 2 * nat.choose B 1 + nat.choose A 1 * nat.choose B 2 = 30) :=
by {
  sorry
}

end choose_courses_l597_597794


namespace bn_geometric_sequence_an_general_formula_l597_597491

-- Define the sequence a_n
def a : ℕ → ℝ
| 0       := 0 -- Typically in Lean we start sequences from 0, not 1
| 1       := 1
| 2       := 2
| (n + 1) := (1 + q) * a n - q * a (n - 1)
  where q : ℝ := sorry -- q should be given beforehand as a non-zero real number

-- Define the subsequence b_n
def b : ℕ → ℝ
| 1       := a 2 - a 1
| (n + 1) := a (n + 2) - a (n + 1)

theorem bn_geometric_sequence (n : ℕ) (h : n ≥ 1) : 
  ∃ r : ℝ, b (n + 1) = r * b n :=
sorry

theorem an_general_formula (n : ℕ) : 
  a n = if q ≠ 1 then 1 + (1 - q ^ (n - 1)) / (1 - q) else n :=
sorry

end bn_geometric_sequence_an_general_formula_l597_597491


namespace cube_octahedron_surface_area_ratio_l597_597034

-- Define the data needed
def cube_side_length : ℝ := 2

def cube_surface_area (s : ℝ) : ℝ := 6 * s^2
def octahedron_surface_area (a : ℝ) : ℝ := 2 * sqrt 3 * a^2

-- The proof problem statement
theorem cube_octahedron_surface_area_ratio 
  (s : ℝ) (a : ℝ) (h_s : s = 2) (h_a : a = sqrt 2) :
  cube_surface_area s / octahedron_surface_area a = 2 * sqrt 3 :=
by
  simp [cube_surface_area, octahedron_surface_area, h_s, h_a]
  sorry

end cube_octahedron_surface_area_ratio_l597_597034


namespace son_age_is_14_l597_597558

-- Definition of Sandra's age and the condition about the ages 3 years ago.
def Sandra_age : ℕ := 36
def son_age_3_years_ago (son_age_now : ℕ) : ℕ := son_age_now - 3 
def Sandra_age_3_years_ago := 36 - 3
def condition_3_years_ago (son_age_now : ℕ) : Prop := Sandra_age_3_years_ago = 3 * (son_age_3_years_ago son_age_now)

-- The goal: proving Sandra's son's age is 14
theorem son_age_is_14 (son_age_now : ℕ) (h : condition_3_years_ago son_age_now) : son_age_now = 14 :=
by {
  sorry
}

end son_age_is_14_l597_597558


namespace new_container_volume_l597_597305

def volume_of_cube (s : ℝ) : ℝ := s^3

theorem new_container_volume (s : ℝ) (h : volume_of_cube s = 4) : 
  volume_of_cube (2 * s) * volume_of_cube (3 * s) * volume_of_cube (4 * s) = 96 :=
by
  sorry

end new_container_volume_l597_597305


namespace earliest_time_meet_l597_597813

open Nat

def lap_time_anna := 5
def lap_time_bob := 8
def lap_time_carol := 10

def lcm_lap_times : ℕ :=
  Nat.lcm lap_time_anna (Nat.lcm lap_time_bob lap_time_carol)

theorem earliest_time_meet : lcm_lap_times = 40 := by
  sorry

end earliest_time_meet_l597_597813


namespace solution_mod_5_l597_597380

theorem solution_mod_5 (a : ℤ) : 
  (a^3 + 3 * a + 1) % 5 = 0 ↔ (a % 5 = 1 ∨ a % 5 = 2) := 
by
  sorry

end solution_mod_5_l597_597380


namespace sin_780_eq_sqrt3_over_2_l597_597350

theorem sin_780_eq_sqrt3_over_2 :
  sin (780 : ℝ) = (Real.sqrt 3 / 2) :=
by
  sorry

end sin_780_eq_sqrt3_over_2_l597_597350


namespace six_q_equals_neg_nine_l597_597505

noncomputable def q : ℝ := sorry -- You might want to define q within your proof if needed.

def a_n (n : ℕ) : ℝ := sorry -- You might want to define the general term of the sequence a_n.

def b_n (n : ℕ) : ℝ := a_n n + 1

-- Assume the sequence b_n has four consecutive terms in the set {-53, -23, 19, 37, 82}.
axiom b_n_has_terms : ∃ n m o p : ℕ, 
  {b_n n, b_n m, b_n o, b_n p} ⊆ {-53, -23, 19, 37, 82} ∧
  n < m ∧ m < o ∧ o < p

-- Assume the geometric sequence a_n with common ratio q where |q| > 1.
axiom geometric_sequence : ∃ q : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n * q ∧ |q| > 1

-- The goal is to prove that 6q = -9.
theorem six_q_equals_neg_nine : 6 * q = -9 :=
sorry

end six_q_equals_neg_nine_l597_597505


namespace evaluate_f_f_of_zero_l597_597436

def f (x : Real) : Real :=
  if x > 0 then 3 * x^2 - 4 else if x = 0 then Real.pi else 0

theorem evaluate_f_f_of_zero :
  f (f (0)) = 3 * Real.pi^2 - 4 :=
by
  sorry

end evaluate_f_f_of_zero_l597_597436


namespace find_lambda_l597_597412

variables {a b : ℝ} (lambda : ℝ)

-- Conditions
def orthogonal (x y : ℝ) : Prop := x * y = 0
def magnitude_a : ℝ := 2
def magnitude_b : ℝ := 3
def is_perpendicular (x y : ℝ) : Prop := x * y = 0

-- Proof statement
theorem find_lambda (h₁ : orthogonal a b)
  (h₂ : magnitude_a = 2)
  (h₃ : magnitude_b = 3)
  (h₄ : is_perpendicular (3 * a + 2 * b) (lambda * a - b)) :
  lambda = 3 / 2 :=
sorry

end find_lambda_l597_597412


namespace smallest_m_n_sum_l597_597222

noncomputable def f (m n : ℕ) (x : ℝ) : ℝ := Real.arcsin (Real.log (n * x) / Real.log m)

theorem smallest_m_n_sum 
  (m n : ℕ) 
  (h_m1 : 1 < m) 
  (h_mn_closure : ∀ x, -1 ≤ Real.log (n * x) / Real.log m ∧ Real.log (n * x) / Real.log m ≤ 1) 
  (h_length : (m ^ 2 - 1) / (m * n) = 1 / 2021) : 
  m + n = 86259 := by
sorry

end smallest_m_n_sum_l597_597222


namespace vector_decomposition_l597_597753

theorem vector_decomposition (α β γ : ℝ) :
  let x : Fin 3 → ℝ := ![2, 7, 5]
  let p : Fin 3 → ℝ := ![1, 0, 1]
  let q : Fin 3 → ℝ := ![1, -2, 0]
  let r : Fin 3 → ℝ := ![0, 3, 1]
  x = α • p + β • q + γ • r :=
  α = 4 ∧ β = -2 ∧ γ = 1 :=
by
  sorry

end vector_decomposition_l597_597753


namespace player1_points_after_13_rotations_l597_597701

theorem player1_points_after_13_rotations :
  ∃ (player1_points : ℕ), 
    (∀ (i : ℕ),  (i = 5 → player1_points = 72) ∧ (i = 9 → player1_points = 84)) → 
    player1_points = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597701


namespace player1_points_after_13_rotations_l597_597695

theorem player1_points_after_13_rotations :
  let sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
  let player_points (player : Nat) (rotations : Nat) :=
      rotations • (λ i, sector_points[(i + player) % 16])
  player_points 5 13 = 72 ∧ player_points 9 13 = 84 → player_points 1 13 = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597695


namespace proof_problem_l597_597471

variable (balls : Finset ℕ) (blackBalls whiteBalls : Finset ℕ)
variable (drawnBalls : Finset ℕ)

/-- There are 6 black balls numbered 1 to 6. -/
def initialBlackBalls : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- There are 4 white balls numbered 7 to 10. -/
def initialWhiteBalls : Finset ℕ := {7, 8, 9, 10}

/-- The total balls (black + white). -/
def totalBalls : Finset ℕ := initialBlackBalls ∪ initialWhiteBalls

/-- The hypergeometric distribution condition for black balls. -/
def hypergeometricBlack : Prop :=
  true  -- placeholder: black balls follow hypergeometric distribution

/-- The probability of drawing 2 white balls is not 1/14. -/
def probDraw2White : Prop :=
  (3 / 7) ≠ (1 / 14)

/-- The probability of the maximum total score (8 points) is 1/14. -/
def probMaxScore : Prop :=
  (15 / 210) = (1 / 14)

/-- Main theorem combining the above conditions for the problem. -/
theorem proof_problem : hypergeometricBlack ∧ probMaxScore :=
by
  unfold hypergeometricBlack
  unfold probMaxScore
  sorry

end proof_problem_l597_597471


namespace uncle_gave_13_l597_597362

-- Define all the given constants based on the conditions.
def J := 7    -- cost of the jump rope
def B := 12   -- cost of the board game
def P := 4    -- cost of the playground ball
def S := 6    -- savings from Dalton's allowance
def N := 4    -- additional amount needed

-- Derived quantities
def total_cost := J + B + P

-- Statement: to prove Dalton's uncle gave him $13.
theorem uncle_gave_13 : (total_cost - N) - S = 13 := by
  sorry

end uncle_gave_13_l597_597362


namespace fraction_power_l597_597822

theorem fraction_power : (2 / 5 : ℚ) ^ 3 = 8 / 125 := by
  sorry

end fraction_power_l597_597822


namespace no_valid_placement_6x6_l597_597992

theorem no_valid_placement_6x6 :
  ¬ (∃ (f : Fin 6 → Fin 6 → ℤ),
      (∀ i j, f i j ≠ f i.succ.pred j) ∧
      (∀ i j : Fin 2, (f i 0 + f i 1 + f i 2 + f i 3 + f i 4 = 2022 ∨ 
                        f i 0 + f i 1 + f i 2 + f i 3 + f i 4 = 2023) ∧ 
                       (f 0 j + f 1 j + f 2 j + f 3 j + f 4 j = 2022 ∨ 
                        f 0 j + f 1 j + f 2 j + f 3 j + f 4 j = 2023))) :=
sorry

end no_valid_placement_6x6_l597_597992


namespace repeating_decimal_block_length_l597_597587

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l597_597587


namespace yella_computer_usage_difference_l597_597277

-- Define the given conditions
def last_week_usage : ℕ := 91
def this_week_daily_usage : ℕ := 8
def days_in_week : ℕ := 7

-- Compute this week's total usage
def this_week_total_usage := this_week_daily_usage * days_in_week

-- Statement to prove
theorem yella_computer_usage_difference :
  last_week_usage - this_week_total_usage = 35 := 
by
  -- The proof will be filled in here
  sorry

end yella_computer_usage_difference_l597_597277


namespace solve_custom_eq_l597_597364

-- Define the custom operation a * b = ab + a + b, we will use ∗ instead of * to avoid confusion with multiplication

def custom_op (a b : Nat) : Nat := a * b + a + b

-- State the problem in Lean 4
theorem solve_custom_eq (x : Nat) : custom_op 3 x = 27 → x = 6 :=
by
  sorry

end solve_custom_eq_l597_597364


namespace geometric_sequence_proof_l597_597987

variables (a : Nat → ℝ)
variables (a1 a3 a4 a6 q : ℝ)

-- Conditions given in the problem
def condition1 : Prop := a 1 + a 3 = 10
def condition2 : Prop := a 4 + a 6 = 5 / 4

-- General term we want to prove
def general_term (n : ℕ) : ℝ := 2 ^ (4 - n)

-- Sum of the first four terms we want to prove
def S_4 : ℝ := (2 ^ 3) + (2 ^ 2) + (2 ^ 1) + (2 ^ 0)

-- Theorem to prove both results given the conditions
theorem geometric_sequence_proof (h1 : condition1) (h2 : condition2) : 
  (∀ n, a n = general_term n) ∧ ∑ i in Finset.range 4, a (i + 1) = S_4 :=
begin
  sorry
end

end geometric_sequence_proof_l597_597987


namespace socks_thrown_away_l597_597188

theorem socks_thrown_away 
  (initial_socks new_socks current_socks : ℕ) 
  (h1 : initial_socks = 11) 
  (h2 : new_socks = 26) 
  (h3 : current_socks = 33) : 
  initial_socks + new_socks - current_socks = 4 :=
by {
  sorry
}

end socks_thrown_away_l597_597188


namespace min_dist_circle_parabola_l597_597045

noncomputable def minimum_distance_between_circle_and_parabola : ℝ :=
  let circle_center : ℝ × ℝ := (4, 0) in
  let radius : ℝ := 1 in
  let parabola (x : ℝ) : set (ℝ × ℝ) := {p | p.1 = x ∧ p.2^2 = 4 * x} in
  let d (p1 p2: ℝ × ℝ) : ℝ := (real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)) in
  let min_distance (p : ℝ × ℝ) : Prop := ∀ q ∈ parabola p.1, d p circle_center = d p q - radius in 
  3

theorem min_dist_circle_parabola : minimum_distance_between_circle_and_parabola = 3 := by sorry

end min_dist_circle_parabola_l597_597045


namespace ratio_of_girls_to_boys_l597_597470

variables (g b : ℕ)

theorem ratio_of_girls_to_boys (h₁ : b = g - 6) (h₂ : g + b = 36) :
  (g / gcd g b) / (b / gcd g b) = 7 / 5 :=
by
  sorry

end ratio_of_girls_to_boys_l597_597470


namespace birth_year_l597_597783

theorem birth_year (x : ℤ) (h : 1850 < x^2 - 10 - x ∧ 1849 ≤ x^2 - 10 - x ∧ x^2 - 10 - x ≤ 1880) : 
x^2 - 10 - x ≠ 1849 ∧ x^2 - 10 - x ≠ 1855 ∧ x^2 - 10 - x ≠ 1862 ∧ x^2 - 10 - x ≠ 1871 ∧ x^2 - 10 - x ≠ 1880 := 
sorry

end birth_year_l597_597783


namespace markov_chain_bound_l597_597504

variables (r : ℕ)
variables (p : Fin r → ℕ → ℝ)
variables (p_tilde : Fin r → ℕ → ℝ)
variables (P : Fin r → Fin r → ℝ)
variables (ε : ℝ)

theorem markov_chain_bound (hP : ∀ i j, P i j ≥ ε)
  (hε : ε > 0)
  (hp_sum : ∀ n, ∑ i, p i n = 1)
  (hp_tilde_sum : ∀ n, ∑ i, p_tilde i n = 1):
  ∀ n, (∑ i, abs (p_tilde i n - p i n)) ≤ 2 * (1 - r * ε)^n := 
by 
  sorry

end markov_chain_bound_l597_597504


namespace rectangles_in_5x5_grid_l597_597728

theorem rectangles_in_5x5_grid : 
  let grid_rows := 5
  let grid_cols := 5
  -- A function that calculates the number of rectangles in an n x m grid
  num_rectangles_in_grid grid_rows grid_cols = 225 :=
  sorry

end rectangles_in_5x5_grid_l597_597728


namespace sigma_le_3n_sigma_gt_100n_sufficiently_large_l597_597342

-- Define σ(n) as the sum of divisors of n
def sigma (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).sum id

-- Statement for Part (a)
theorem sigma_le_3n (n : ℕ) (h : n > 0 ∧ n ≤ 320) : sigma(n) ≤ 3 * n :=
  sorry

-- Statement for Part (b)
theorem sigma_gt_100n_sufficiently_large (n : ℕ) (h : n > 0) : ∃ (k : ℕ), n > k → sigma(n) > 100 * n :=
  sorry

end sigma_le_3n_sigma_gt_100n_sufficiently_large_l597_597342


namespace eval_sequence_difference_l597_597858

theorem eval_sequence_difference :
  (∑ i in Finset.range (3001 - 2901) 2901) - (∑ j in Finset.range (351 - 251) 251) = 265000 := 
sorry

end eval_sequence_difference_l597_597858


namespace min_value_of_f_l597_597228

noncomputable def f (x : ℝ) : ℝ := Real.sin(2 * x + Real.pi / 6)
def m := -1 / 2

theorem min_value_of_f : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = m :=
  sorry

end min_value_of_f_l597_597228


namespace intersection_of_M_and_N_l597_597937

-- Define the sets M and N
def M := {-1, 1 : Int}
def N := {x : Int | -1 < x ∧ x < 4}

-- The theorem to prove
theorem intersection_of_M_and_N : (M ∩ N) = {1} := by
  sorry

end intersection_of_M_and_N_l597_597937


namespace proof_problem_l597_597405

def seq_a (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = 2 * a n + n

def seq_geo (a : ℕ → ℤ) : Prop :=
  ∃ c r : ℤ, r ≠ 0 ∧ ∀ n : ℕ, n > 0 → a n = c * r ^ n

def S_formula (S : ℕ → ℤ) (n : ℕ) : ℤ :=
  match n with
  | 0 => 0
  | n + 1 => 2 * (S n) + (n + 1)

def T (b : ℕ → ℤ) : ℕ → ℚ
| 0 => 0
| n + 1 => T n + 1 / (b (n + 1) * b (n + 3))

def b_n (a : ℕ → ℤ) (n : ℕ) : ℕ :=
  (Int.log2 (-a n + 1)).toNat

theorem proof_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℤ) :
  seq_a a S ∧ seq_geo (λ n => a n - 1) ∧
  (∀ n, a n = -2^n + 1) ∧ 
  (∀ n : ℕ, b n = (Int.log2 (-a n + 1)).toNat) →
  (∀ n, T (λ n => b_n a n) n = ((3:ℤ)/4 - (2 * (n + 1) + 3) / (2 * ((n + 1) * (n + 2)):ℚ))) :=
by
  intro h
  sorry

end proof_problem_l597_597405


namespace problem_1_solution_problem_2_solution_l597_597059

open Complex

noncomputable def problem_1 (z1 z2 : ℂ) : ℂ :=
  (z2 / z1)

theorem problem_1_solution (z1 z2 : ℂ) (P1 : z1 = 1 - I) (P2 : z2 = 4 + 6 * I) :
  (problem_1 z1 z2) = -1 + 5 * I :=
  by rw [problem_1, P1, P2]; sorry

noncomputable def problem_2 (b : ℝ) (z1 : ℂ) : ℝ :=
  abs (1 + b * I)

theorem problem_2_solution (b : ℝ) (z1 : ℂ) (P1 : z1 = 1 - I) (P3 : 1 + b * I + z1 ∈ ℝ) :
  (problem_2 b z1) = real.sqrt 2 :=
  by
    have h1 : (1 + b * I + (1 - I)) ∈ ℝ := P3
    sorry

end problem_1_solution_problem_2_solution_l597_597059


namespace min_value_PA_PB_l597_597112

theorem min_value_PA_PB (P : (ℝ × ℝ)) (α : ℝ)
  (hP : P = (1, 2))
  (polar_eq : ∀ θ : ℝ, ∃ ρ : ℝ, ρ = 6 * sin θ) :
  (∃ l : ℝ → ℝ × ℝ, ∀ t : ℝ, l t = (1 + t * cos α, 2 + t * sin α)) →
  ∃ A B : ℝ × ℝ,
    (A, B ∈ {p : ℝ × ℝ | ∃ θ : ℝ, (polar_eq θ).1 = p} ∧
     ∃ t1 t2 : ℝ, l t1 = A ∧ l t2 = B) →
    (1 / (abs (dist P A)) + 1 / (abs (dist P B)) = 2 * sqrt 7 / 7) :=
by
  sorry

end min_value_PA_PB_l597_597112


namespace cos_alpha_value_l597_597051

theorem cos_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos (α + π / 4) = 4 / 5) :
  Real.cos α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end cos_alpha_value_l597_597051


namespace find_f_when_x_neg_l597_597058

variable {ℝ : Type*} [LinearOrderedField ℝ]

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → f (x) = 2^x + x + 1

theorem find_f_when_x_neg (f : ℝ → ℝ)
  (h1 : is_odd_function f)
  (h2 : given_function f) :
  ∀ x : ℝ, x < 0 → f (x) = -2^(-x) + x - 1 :=
sorry

end find_f_when_x_neg_l597_597058


namespace right_triangle_sqrt_l597_597752

noncomputable def sqrt_2 := Real.sqrt 2
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_5 := Real.sqrt 5

theorem right_triangle_sqrt: 
  (sqrt_2 ^ 2 + sqrt_3 ^ 2 = sqrt_5 ^ 2) :=
by
  sorry

end right_triangle_sqrt_l597_597752


namespace number_of_students_l597_597311

def total_students (a b : ℕ) : ℕ :=
  a + b

variables (a b : ℕ)

theorem number_of_students (h : 48 * a + 45 * b = 972) : total_students a b = 21 :=
by
  sorry

end number_of_students_l597_597311


namespace sin_780_eq_sqrt3_div_2_l597_597356

theorem sin_780_eq_sqrt3_div_2 : Real.sin (780 * Real.pi / 180) = Math.sqrt 3 / 2 := by
  sorry

end sin_780_eq_sqrt3_div_2_l597_597356


namespace length_of_other_diagonal_l597_597640

theorem length_of_other_diagonal (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = 15) (h2 : A = 150) : d2 = 20 :=
by
  sorry

end length_of_other_diagonal_l597_597640


namespace downstream_speed_l597_597784

-- Define the given conditions
def V_m : ℝ := 40 -- speed of the man in still water in kmph
def V_up : ℝ := 32 -- speed of the man upstream in kmph

-- Question to be proved as a statement
theorem downstream_speed : 
  ∃ (V_c V_down : ℝ), V_c = V_m - V_up ∧ V_down = V_m + V_c ∧ V_down = 48 :=
by
  -- Provide statement without proof as specified
  sorry

end downstream_speed_l597_597784


namespace min_value_proof_l597_597176

def complex_min_value (x y : ℝ) : Prop := 
  let z := complex.mk x y in
  abs (z - complex.mk 3 2) = 7 →
  (abs (z - complex.mk 2 (-1)))^2 + (abs (z - complex.mk 11 5))^2 = 554

theorem min_value_proof (x y : ℝ) : complex_min_value x y :=
begin
  let z := complex.mk x y,
  assume h : abs (z - complex.mk 3 2) = 7,
  have := (complex.abs_squared_eq_re_add_im _) sorry,
  sorry -- Proof would go here, skipped as per instructions.
end

end min_value_proof_l597_597176


namespace residue_mod_1024_l597_597156

-- Define the series T
def series_T : ℤ := (Finset.range 2048).sum (λ n, if Even n then -(n : ℤ) else n)

-- Define the modulo value
def modulo_val : ℤ := 1024

-- Statement of the problem
theorem residue_mod_1024 : (series_T % modulo_val = 0) :=
begin
  sorry
end

end residue_mod_1024_l597_597156


namespace quad_eq_double_root_m_value_l597_597014

theorem quad_eq_double_root_m_value (m : ℝ) : 
  (∀ x : ℝ, x^2 + 6 * x + m = 0) → m = 9 := 
by 
  sorry

end quad_eq_double_root_m_value_l597_597014


namespace arrangement_count_l597_597315

theorem arrangement_count (n : ℕ) (h : n = 2018) : 
  ∃ c : ℕ, c = 2 * nat.factorial 1008 ∧ 
  ∀ (P : fin n → ℕ),
    (∀ i : fin n, P (i + 1) + P i = P (i + 1009) + P (i + 1010)) ∧
    (∀ rotation : fin n → fin n, 
      (∀ i : fin n, P i = P (rotation i))) → 
      (∃! count : ℕ, count = c) :=
begin
  sorry
end

end arrangement_count_l597_597315


namespace positive_x_solution_l597_597850

theorem positive_x_solution (x : ℝ) (h : x > 0) (cond : (sqrt (16 * x)) * (sqrt (25 * x)) * (sqrt (5 * x)) * (sqrt (20 * x)) = 40) : 
  x = 1 / (sqrt 5) :=
sorry

end positive_x_solution_l597_597850


namespace james_total_money_l597_597137

theorem james_total_money :
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  total_money = 135 := by
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  exact 135

end james_total_money_l597_597137


namespace cover_vs_nonoverlap_l597_597163

noncomputable def m (M : Type*) [convex M] : ℕ := sorry -- Defining m as the minimum number of circles needed to cover M
noncomputable def n (M : Type*) [convex M] : ℕ := sorry -- Defining n as the maximum number of non-overlapping circles in M

theorem cover_vs_nonoverlap (M : Type*) [convex M] : m M ≤ n M := by 
  sorry

end cover_vs_nonoverlap_l597_597163


namespace volume_of_rotational_ellipsoid_l597_597341

-- Lean Statement

theorem volume_of_rotational_ellipsoid (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ V : ℝ, V = (4 / 3) * π * a^2 * b) :=
by
  have h : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 -> False := sorry
  use (4 / 3) * π * a^2 * b
  -- Proof omitted
  sorry

end volume_of_rotational_ellipsoid_l597_597341


namespace lower_right_is_4_l597_597705

-- Definition of initial grid setup
def initial_grid : List (List (Option ℕ)) :=
  [[some 1, none,    some 2, some 3, none],
   [some 2, some 3,  none,   none,   some 1],
   [none,   some 1,  none,   some 5, none],
   [none,   none,    none,   none,   none],
   [none,   none,    some 4, none,   none]]

-- Function to check whether grid is valid (each number 1-5 appears once per row and once per column)
def is_valid_grid (grid : List (List (Option ℕ))) : Prop := 
  ∀ (i j k : ℕ), i < 5 → j < 5 → k < 5 →
    (grid[i].nth j = some (k + 1) → ∀ l, l ≠ j → grid[i].nth l ≠ some (k + 1)) ∧ 
    (grid[j].nth i = some (k + 1) → ∀ l, l ≠ j → grid[l].nth i ≠ some (k + 1))

-- Final proof problem: given initial grid conditions, the number at the lower right corner is 4
theorem lower_right_is_4 : 
  (∃ grid : List (List (Option ℕ)), 
    grid = initial_grid ∧ is_valid_grid grid ∧ grid[4][4] = some 4) :=
  sorry

end lower_right_is_4_l597_597705


namespace repeating_block_length_7_div_13_l597_597594

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l597_597594


namespace sum_of_sequence_l597_597907

-- Declare the initial conditions a_n > 0, a_n^2 + 3a_n = 6S_n + 4
variables (S : ℕ → ℝ) (a : ℕ → ℝ) 

-- Main theorem that encapsulates the two parts of the given math problem.
theorem sum_of_sequence (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (a n)^2 + 3 * (a n) = 6 * S n + 4) :
  (∀ n, a n = 3 * n + 1) ∧
  (∀ n, T n = (λ b_n : ℕ → ℝ, b_n n = 3 / ((a n) * (a (n + 1))) ∧
    ∑ i in finset.range n, b_n i = 1 / 4 - 1 / (3 * n + 4))) :=
by
  sorry

end sum_of_sequence_l597_597907


namespace sum_powers_of_5_mod_7_l597_597749

theorem sum_powers_of_5_mod_7 :
  ( (Finset.range 11).sum (λ n, 5 ^ n) ) % 7 = 4 :=
by
  sorry

end sum_powers_of_5_mod_7_l597_597749


namespace max_f_convex_ngons_l597_597883

noncomputable def f (P1 P2 : Polygon) : ℕ :=
-- placeholder for the definition of f
sorry

theorem max_f_convex_ngons (n : ℕ) (h : n ≥ 4) : 
  ∀ (P1 P2 : Polygon), 
    (convex P1 ∧ vertices_count P1 = n) →
    (convex P2 ∧ vertices_count P2 = n) →
    (vertices_mutually_distinct P1 P2) →
    max_f_convex P1 P2 = (4 * n / 3) :=
sorry

#check max_f_convex_ngons

end max_f_convex_ngons_l597_597883


namespace minimum_cost_to_reverse_chips_order_l597_597370

theorem minimum_cost_to_reverse_chips_order : 
  ∀ (n : ℕ) (chips : Fin n → ℕ), 
    (∀ i : ℕ, i < n → chips i = i) →
    (∀ i j : ℕ, i < j ∧ j = i + 1 → 1) →
    (∀ i j : ℕ, j = i + 5 → 0) →
    n = 100 → 
    reverse_cost chips = 61 := 
by 
  intros n chips hchip_order hswap_cost1 hswap_cost2 hn 
  sorry

end minimum_cost_to_reverse_chips_order_l597_597370


namespace triangle_interior_angle_at_least_one_leq_60_l597_597539

theorem triangle_interior_angle_at_least_one_leq_60 {α β γ : ℝ} :
  α + β + γ = 180 →
  (α > 60 ∧ β > 60 ∧ γ > 60) → false :=
by
  intro hsum hgt
  have hα : α > 60 := hgt.1
  have hβ : β > 60 := hgt.2.1
  have hγ : γ > 60 := hgt.2.2
  have h_total: α + β + γ > 60 + 60 + 60 := add_lt_add (add_lt_add hα hβ) hγ
  linarith

end triangle_interior_angle_at_least_one_leq_60_l597_597539


namespace find_M_P_on_axes_l597_597079

variable {a : ℝ} (h : a > 0)

def A := (a, a)

def is_equilateral_triangle (A M P : ℝ × ℝ) : Prop :=
  let AM := (M.1 - A.1)^2 + (M.2 - A.2)^2
  let AP := (P.1 - A.1)^2 + (P.2 - A.2)^2
  let MP := (P.1 - M.1)^2 + (P.2 - M.2)^2
  AM = AP ∧ AP = MP

theorem find_M_P_on_axes 
  (M : ℝ × ℝ) (P : ℝ × ℝ)
  (hm : M.2 = 0) (hp : P.1 = 0)
  (h_equil : is_equilateral_triangle A M P) :
  M = (a * (Real.sqrt 3 - 1), 0) ∧ P = (0, a * (Real.sqrt 3 - 1)) := 
sorry

end find_M_P_on_axes_l597_597079


namespace find_g_one_l597_597414

variable {α : Type} [AddGroup α]

def is_odd (f : α → α) : Prop :=
∀ x, f (-x) = - f x

def is_even (g : α → α) : Prop :=
∀ x, g (-x) = g x

theorem find_g_one
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 := by
  sorry

end find_g_one_l597_597414


namespace largest_number_among_options_l597_597750

theorem largest_number_among_options :
  let A := 8.12366
  let B := 8.1236666666666 -- Repeating decimal 8.123\overline{6}
  let C := 8.1236363636363 -- Repeating decimal 8.12\overline{36}
  let D := 8.1236236236236 -- Repeating decimal 8.1\overline{236}
  let E := 8.1236123612361 -- Repeating decimal 8.\overline{1236}
  B > A ∧ B > C ∧ B > D ∧ B > E :=
by
  let A := 8.12366
  let B := 8.12366666666666
  let C := 8.12363636363636
  let D := 8.12362362362362
  let E := 8.12361236123612
  sorry

end largest_number_among_options_l597_597750


namespace player_1_points_l597_597669

-- Definition: point distribution on the table.
noncomputable def sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

-- Conditions
axiom player_5_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(5 + i) % 16]) = 72
axiom player_9_points (rotations : ℕ) : rotations = 13 → ∑ i in finset.range rotations, (sector_points[(9 + i) % 16]) = 84

-- Question translated to proof statement:
theorem player_1_points (rotations : ℕ) (p5_points : ℕ) (p9_points : ℕ) :
  rotations = 13 → p5_points = 72 → p9_points = 84 →
  ∑ i in finset.range rotations, (sector_points[(1 + i) % 16]) = 20 :=
by
  sorry

end player_1_points_l597_597669


namespace cosine_of_acute_angle_l597_597377

theorem cosine_of_acute_angle (α x : ℝ) (h1 : 0 < α ∧ α < π/2) : 
    (cos (4 * α) = 7 / 18) :=
sorry

end cosine_of_acute_angle_l597_597377


namespace unique_random_event_l597_597327

-- Definitions based on the conditions
def event_A : Prop := ∀ (balls : Set String), ("red" ∈ balls) → ("white" ∈ balls ∨ "black" ∈ balls) → False

def event_B : Prop := ∀ (temp : ℕ), temp = 100 → boils (Water temp)

def event_C : Prop := ∃ (time : ℕ), ¬predictable (red_light time)

def event_D : Prop := ∀ (d1 d2 : ℕ), 1 ≤ d1 ∧ d1 ≤ 6 → 1 ≤ d2 ∧ d2 ≤ 6 → sum_dice d1 d2 = 1 → False

-- Theorem stating that event C is the only random event
theorem unique_random_event : event_C ∧ ¬event_A ∧ ¬event_B ∧ ¬event_D := 
  by sorry

end unique_random_event_l597_597327


namespace num_two_digit_numbers_with_four_factors_l597_597944

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_exactly_four_factors (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ (n = p^3 ∨ ∃ q : ℕ, is_prime q ∧ p ≠ q ∧ n = p * q)

theorem num_two_digit_numbers_with_four_factors : 
  { n : ℕ | is_two_digit n ∧ has_exactly_four_factors n }.to_finset.card = 31 := 
sorry

end num_two_digit_numbers_with_four_factors_l597_597944


namespace finite_integer_solutions_l597_597194

theorem finite_integer_solutions (n : ℕ) : 
  ∃ (S : Finset (ℤ × ℤ)), ∀ (x y : ℤ), (x^3 + y^3 = n) → (x, y) ∈ S := 
sorry

end finite_integer_solutions_l597_597194


namespace last_two_digits_sum_is_32_l597_597721

-- Definitions for digit representation
variables (z a r l m : ℕ)

-- Numbers definitions
def ZARAZA := z * 10^5 + a * 10^4 + r * 10^3 + a * 10^2 + z * 10 + a
def ALMAZ := a * 10^4 + l * 10^3 + m * 10^2 + a * 10 + z

-- Condition that ZARAZA is divisible by 4
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Condition that ALMAZ is divisible by 28
def divisible_by_28 (n : ℕ) : Prop := n % 28 = 0

-- The theorem to prove
theorem last_two_digits_sum_is_32
  (hz4 : divisible_by_4 (ZARAZA z a r))
  (ha28 : divisible_by_28 (ALMAZ a l m z))
  : (ZARAZA z a r + ALMAZ a l m z) % 100 = 32 :=
by sorry

end last_two_digits_sum_is_32_l597_597721


namespace hyperbola_solution_l597_597914

def hyperbola_asymptote_passing_point (a : ℝ) : Prop :=
    let asymptote_line := λ x: ℝ, (1 / a) * x
    ∃ x y : ℝ, asymptote_line x = y ∧ x = real.sqrt 2 ∧ y = 1

def hyperbola_foci (a : ℝ) : Prop :=
    let b := real.sqrt 2
    let c := real.sqrt (a^2 + b^2)
    c = real.sqrt 6

theorem hyperbola_solution (a : ℝ) (h_asymptote : hyperbola_asymptote_passing_point a) :
    hyperbola_foci a :=
by
    -- A formal proof would go here
    sorry

end hyperbola_solution_l597_597914


namespace black_rectangle_ways_l597_597738

theorem black_rectangle_ways : ∑ a in Finset.range 5, ∑ b in Finset.range 5, (5 - a) * (5 - b) = 225 := sorry

end black_rectangle_ways_l597_597738


namespace line_passes_through_fixed_point_l597_597274

theorem line_passes_through_fixed_point (k : ℝ) :
  ∃ (x y : ℝ), k * x - y - 3 * k + 3 = 0 ∧ x = 3 ∧ y = 3 :=
by
  use 3, 3
  split
  sorry

end line_passes_through_fixed_point_l597_597274


namespace raisins_in_other_three_boxes_l597_597336

-- Definitions of the known quantities
def total_raisins : ℕ := 437
def box1_raisins : ℕ := 72
def box2_raisins : ℕ := 74

-- The goal is to prove that each of the other three boxes has 97 raisins
theorem raisins_in_other_three_boxes :
  total_raisins - (box1_raisins + box2_raisins) = 3 * 97 :=
by
  sorry

end raisins_in_other_three_boxes_l597_597336


namespace product_of_fractions_l597_597830

theorem product_of_fractions :
  ∏ k in finset.range 501, (4 + 4 * k : ℕ) / (8 + 4 * k) = 1 / 502 := 
by sorry

end product_of_fractions_l597_597830


namespace math_problem_l597_597900

variable {a b : ℕ → ℕ}

-- Condition 1: a_n is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m

-- Condition 2: 2a₂ - a₇² + 2a₁₂ = 0
def satisfies_equation (a : ℕ → ℕ) : Prop :=
  2 * a 2 - (a 7)^2 + 2 * a 12 = 0

-- Condition 3: b_n is a geometric sequence
def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, b (n + m) = b n * b m

-- Condition 4: b₇ = a₇
def b7_eq_a7 (a b : ℕ → ℕ) : Prop :=
  b 7 = a 7

-- To prove: b₅ * b₉ = 16
theorem math_problem (a b : ℕ → ℕ)
  (h₁ : is_arithmetic_sequence a)
  (h₂ : satisfies_equation a)
  (h₃ : is_geometric_sequence b)
  (h₄ : b7_eq_a7 a b) :
  b 5 * b 9 = 16 :=
sorry

end math_problem_l597_597900


namespace problem_b_50_l597_597359

def seq (b : ℕ → ℕ) : Prop :=
  b 1 = 3 ∧ ∀ n ≥ 1, b (n + 1) = b n + 3 * n

theorem problem_b_50 (b : ℕ → ℕ) (h : seq b) : b 50 = 3678 := 
sorry

end problem_b_50_l597_597359


namespace equation_solution_exists_l597_597566

theorem equation_solution_exists (n : ℤ) :
  ∃ (x y : ℤ), 2^(x - y) / y - 3 / 2 * y = 1 ∧
               y = (2^(2 * n + 1) - 2) / 3 ∧ 
               x = y * (2 * n + 1) :=
by
  sorry

end equation_solution_exists_l597_597566


namespace repeating_block_length_7_div_13_l597_597624

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l597_597624


namespace axis_of_symmetry_increasing_intervals_minimum_value_l597_597927

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sin x * cos x - 1 / 2

-- 1. Prove the axis of symmetry
theorem axis_of_symmetry (k : ℤ) : ∃ k : ℤ, f (π / 8 + k * π / 2) = f (π / 8) := 
sorry

-- 2. Prove the intervals where f(x) is increasing
theorem increasing_intervals (k : ℤ) : 
  ∀ x, (k * π - 3 * π / 8 ≤ x ∧ x ≤ k * π + π / 8) → (f' x > 0) :=
sorry

-- 3. Prove the minimum value of f(x) in [0, π/2]
theorem minimum_value : ∃ x ∈ (set.Icc 0 (π/2)), f(π / 2) = -1 / 2 :=
sorry

end axis_of_symmetry_increasing_intervals_minimum_value_l597_597927


namespace minimum_value_expression_l597_597468

theorem minimum_value_expression : ∃ N : ℕ, N = 70 ∧ (∃ f: List (ℤ × ℤ), (∏ (a, b) in f, a / b = 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) := sorry

end minimum_value_expression_l597_597468


namespace distinct_integers_in_intersection_l597_597205

def SetA := {n : ℤ | 4 ≤ n ∧ n ≤ 15}
def SetB := {n : ℤ | 6 ≤ n ∧ n ≤ 20}
def Intersection := SetA ∩ SetB

theorem distinct_integers_in_intersection : (Intersection.to_finset.card = 10) :=
by {
  sorry
}

end distinct_integers_in_intersection_l597_597205


namespace points_player_1_after_13_rotations_l597_597685

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end points_player_1_after_13_rotations_l597_597685


namespace superhero_vs_combined_distance_l597_597320

theorem superhero_vs_combined_distance :
  ∀ (distance_superhero : ℝ) (time_superhero : ℝ) (speed_villain : ℝ) (speed_antihero : ℝ),
  distance_superhero = 50 →
  time_superhero = 12 / 60 →
  speed_villain = 150 →
  speed_antihero = 180 →
  (distance_superhero / (time_superhero / 60)  - (speed_villain  + speed_antihero )) = -80 :=
by
  intros
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end superhero_vs_combined_distance_l597_597320


namespace valid_4_word_sentences_l597_597212

def gnollish_words := ["splargh", "glumph", "amr", "zorg"]

def is_valid_sentence (sentence : List String) : Bool :=
  ∀ (i : Nat), i < sentence.length - 1 → 
    (sentence[i] = "splargh" → sentence[i + 1] ≠ "glumph") ∧
    (sentence[i] = "amr" → sentence[i + 1] ≠ "zorg")

def count_valid_sentences (n : Nat) : Nat :=
  (List.replicateM n gnollish_words).countp is_valid_sentence

theorem valid_4_word_sentences : count_valid_sentences 4 = 193 := by
  sorry

end valid_4_word_sentences_l597_597212


namespace tangent_lines_chord_length_correct_l597_597896

noncomputable def circle_c_eq (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 4

-- Defining the equations for the tangent lines as proofs
theorem tangent_lines (x y : ℝ) :
  circle_c_eq 3 (-2) → (x = 3 ∨ 3 * x + 4 * y - 1 = 0) := 
sorry

-- Defining the line equation
def line_l_eq (x y : ℝ) : Prop :=
  2 * x - y + 2 = 0

-- Defining the center and radius of the circle
def circle_center := (1 : ℝ, 2 : ℝ)
def circle_radius := (2 : ℝ)

-- Calculating the distance from the center to the line
noncomputable def distance_from_center_to_line : ℝ :=
  (abs (2 * 1 - 1 * 2 + 2)) / (sqrt (2^2 + (-1)^2))

-- Verifying chord length
theorem chord_length_correct : distance_from_center_to_line = 2 / sqrt 5 →
  2 * sqrt (circle_radius^2 - distance_from_center_to_line^2) = 8 * sqrt 5 / 5 :=
sorry

end tangent_lines_chord_length_correct_l597_597896


namespace product_sequence_equals_l597_597832

-- Define the form of each fraction in the sequence
def frac (k : ℕ) : ℚ := (4 * k : ℚ) / (4 * k + 4)

-- Define the product of the sequence from k=1 to k=501
def productSequence : ℚ :=
  (finset.range 501).prod (λ k => frac (k + 1))

-- The theorem that the product equals 1/502
theorem product_sequence_equals : productSequence = 1 / 502 := by
  sorry

end product_sequence_equals_l597_597832


namespace largest_common_divisor_l597_597743

theorem largest_common_divisor (h408 : ∀ d, Nat.dvd d 408 → d ∈ [1, 2, 3, 4, 6, 8, 12, 17, 24, 34, 51, 68, 102, 136, 204, 408])
                               (h340 : ∀ d, Nat.dvd d 340 → d ∈ [1, 2, 4, 5, 10, 17, 20, 34, 68, 85, 170, 340]) :
  ∃ d, Nat.dvd d 408 ∧ Nat.dvd d 340 ∧ d = 68 := by
  sorry

end largest_common_divisor_l597_597743


namespace problem1_problem2_problem3_l597_597911

variable (a b : ℝ)
variable (h_pos_a : a > 0)
variable (h_pos_b : b > 0)
variable (h_cond1 : a ≥ (1 / a) + (2 / b))
variable (h_cond2 : b ≥ (3 / a) + (2 / b))

/-- Statement 1: Prove that a + b ≥ 4 under the given conditions. -/
theorem problem1 : (a + b) ≥ 4 := 
by 
  sorry

/-- Statement 2: Prove that a^2 + b^2 ≥ 3 + 2√6 under the given conditions. -/
theorem problem2 : (a^2 + b^2) ≥ (3 + 2 * Real.sqrt 6) := 
by 
  sorry

/-- Statement 3: Prove that (1/a) + (1/b) < 1 + (√2/2) under the given conditions. -/
theorem problem3 : (1 / a) + (1 / b) < 1 + (Real.sqrt 2 / 2) := 
by 
  sorry

end problem1_problem2_problem3_l597_597911


namespace tessa_needs_more_apples_l597_597211

def initial_apples : ℝ := 4.75
def received_apples : ℝ := 5.5
def required_apples : ℝ := 12.25

theorem tessa_needs_more_apples : 
  initial_apples + received_apples < required_apples → 
  required_apples - (initial_apples + received_apples) = 2 := 
by 
  intro h
  simp [initial_apples, received_apples, required_apples] at h
  simp [initial_apples, received_apples, required_apples]
  contradiction
  sorry

end tessa_needs_more_apples_l597_597211


namespace intersection_of_line_and_circle_shortest_chord_line_equation_l597_597920

noncomputable def circle_equation (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 25

noncomputable def line_equation (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem intersection_of_line_and_circle (m : ℝ) :
  ∃ x y : ℝ, line_equation m x y ∧ circle_equation x y := sorry

theorem shortest_chord_line_equation :
  ∀ m : ℝ, 
  (∀ x y : ℝ, line_equation m x y → circle_equation x y)
  → (line_equation 0 2 (-1) = 5) := sorry

end intersection_of_line_and_circle_shortest_chord_line_equation_l597_597920


namespace points_away_10_cm_from_square_vertices_l597_597798

theorem points_away_10_cm_from_square_vertices :
  let square_side := 10.0 -- side length of the square
  let distance := 10.0 -- distance from vertices
  ∃ (n : ℕ), n = 12
  ∧ (∀ (square : ℝ × ℝ → Prop), -- a square is defined by four vertices (a function that takes vertices and returns a boolean)
    (∀ (x y : ℝ × ℝ),
      square x
      ∧ square y
      ∧ (real.dist x y = square_side)
      ∧ (∀ (p : ℝ × ℝ),
        (real.dist p x = distance)
        ∧ (real.dist p y = distance)))
    ↔ (n = 12)) :=
by
  sorry

end points_away_10_cm_from_square_vertices_l597_597798


namespace multiplication_letters_correct_l597_597493

theorem multiplication_letters_correct :
  ∃ (a b c d e : ℕ), a = 3 ∧ b = 0 ∧ c = 7 ∧ d = 2 ∧ e = 9 ∧ (abba a b * cdea c d e a = 21900879) :=
by
  let abba (a b : Nat) := a * 1000 + b * 100 + b * 10 + a
  let cdea (c d e a : Nat) := c * 1000 + d * 100 + e * 10 + a
  exists 3 0 7 2 9
  split; exact rfl 
  split; exact rfl 
  split; exact rfl 
  split; exact rfl 
  split; exact rfl 
  compute_rhs 
  disk_math
  sorry

end multiplication_letters_correct_l597_597493


namespace stream_speed_l597_597283

variable (v : ℝ)

def effective_speed_downstream (v : ℝ) : ℝ := 7.5 + v
def effective_speed_upstream (v : ℝ) : ℝ := 7.5 - v 

theorem stream_speed : (7.5 - v) / (7.5 + v) = 1 / 2 → v = 2.5 :=
by
  intro h
  -- Proof will be resolved here
  sorry

end stream_speed_l597_597283


namespace wage_increase_percentage_l597_597805

theorem wage_increase_percentage (new_wage old_wage : ℝ) (h1 : new_wage = 35) (h2 : old_wage = 25) : 
  ((new_wage - old_wage) / old_wage) * 100 = 40 := 
by
  sorry

end wage_increase_percentage_l597_597805


namespace frog_escape_probability_l597_597478

noncomputable def P : ℕ → ℚ
| 0     := 0
| 10    := 1
| (n+1) := if h : n < 9 then (n + 1) / 10 * P n + (1 - (n + 1) / 10) * P (n + 2) 
          else 0 -- this case is only to cover all possible inputs

theorem frog_escape_probability : P 1 = 63 / 146 :=
by 
  sorry

end frog_escape_probability_l597_597478


namespace general_term_a_n_sum_b_n_l597_597916

open Nat Real

-- Define the arithmetic sequence a_n
def a_n (n : ℕ) : ℕ := 2 + (n - 1) * 2

-- The given conditions for a_n
axiom a1 : a_n 1 = 2
axiom sum_a1_a2_a3 : a_n 1 + a_n 2 + a_n 3 = 12

-- Define the sequence b_n related to a_n
def b_n (n : ℕ) : ℕ := 3 ^ ((a_n n) / 2)

-- The proof statement for the general term of a_n
theorem general_term_a_n : ∀ n : ℕ, a_n n = 2 * n :=
by
  sorry

-- The proof statement for the sum of the first n terms of b_n
theorem sum_b_n (n : ℕ) : ∀ n, (finset.range n).sum b_n = (3 ^ (n + 1) - 3) / 2 :=
by
  sorry

end general_term_a_n_sum_b_n_l597_597916


namespace dot_product_implies_collinear_l597_597049

variables {α : Type*} [inner_product_space ℝ α]

def collinear (a b : α) : Prop :=
∃ k : ℝ, a = k • b

theorem dot_product_implies_collinear (a b : α) 
  (h : abs (⟪a, b⟫) = ∥a∥ * ∥b∥) : collinear a b :=
sorry

end dot_product_implies_collinear_l597_597049


namespace range_of_m_l597_597904

theorem range_of_m (m : ℝ) (hm : m > 0) :
  (∀ x, (x^2 + 1) * (x^2 - 8 * x - 20) ≤ 0 → (x^2 - 2 * x + (1 - m^2)) ≤ 0) →
  m ≥ 9 := by
  sorry

end range_of_m_l597_597904


namespace compare_terms_l597_597516

def is_solution (x y : ℝ) : Prop := 
  ∃ (a b : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ x = a + b * (Real.sqrt 5) ∧ y = c + d * (Real.sqrt 5)

theorem compare_terms 
  (a b c d : ℝ) 
  (ha : a ≥ 0) 
  (hb : b ≥ 0) 
  (hc : c ≥ 0) 
  (hd : d ≥ 0) 
  (hineq : a + b * Real.sqrt 5 < c + d * Real.sqrt 5) : 
  a < c ∧ b < d := 
sorry

end compare_terms_l597_597516


namespace sin_780_eq_sqrt3_div_2_l597_597348

theorem sin_780_eq_sqrt3_div_2 :
  Real.sin (780 * Real.pi / 180) = (Real.sqrt 3) / 2 :=
by
  sorry

end sin_780_eq_sqrt3_div_2_l597_597348


namespace house_purchase_decision_l597_597147
-- Lean Code Statement

theorem house_purchase_decision (P Q : Prop) : (P ↔ "only two people came to look at the house") ∧ (Q ↔ "neither of whom wanted to buy it") →
  Q :=
by
  intro h
  have h1 := h.1
  have h2 := h.2
  sorry

end house_purchase_decision_l597_597147


namespace maximum_value_f_x_l597_597891

theorem maximum_value_f_x (x : ℝ) (h : x < 3) : 
  ∃ y : ℝ, ∀ z : ℝ, (z < 3) → (y ≥ f z) ∧ (y = -1) :=
sorry

end maximum_value_f_x_l597_597891


namespace arithmetic_sequence_40th_term_difference_l597_597329

theorem arithmetic_sequence_40th_term_difference :
  ∀ (a : ℕ → ℕ) (d : ℕ) (sn : ℕ),
  (∀ n, 20 ≤ a n ∧ a n ≤ 80) →
  (sn = 150) →
  (∑ i in range sn, a i = 9000) →
  let L := a 39 - 111 * d / 149 in
  let G := a 39 + 111 * d / 149 in
  (G - L) = 4440 / 149 :=
by 
  sorry

end arithmetic_sequence_40th_term_difference_l597_597329


namespace player1_points_after_13_rotations_l597_597698

theorem player1_points_after_13_rotations :
  ∃ (player1_points : ℕ), 
    (∀ (i : ℕ),  (i = 5 → player1_points = 72) ∧ (i = 9 → player1_points = 84)) → 
    player1_points = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597698


namespace cylinder_volume_l597_597421

theorem cylinder_volume (r l : ℝ) (h1 : r = 1) (h2 : l = 2 * r) : 
  ∃ V : ℝ, V = 2 * Real.pi := 
by 
  sorry

end cylinder_volume_l597_597421


namespace point_A_on_line_GH_l597_597574

open_locale classical

noncomputable def isosceles_triangle (A B C : Point) : Prop :=
  ∃ ω : Circle, IsoscelesTriangle A B C ∧ is_inscribed ω (Triangle.mk A B C) ∧ ω.touches BC = D

noncomputable def point_on_extension (A B K : Point) : Prop :=
  ∃ λ : ℝ, K = A + λ • (A - B)

noncomputable def extend_point (A B K : Point) (length : ℝ) : Point :=
  A + length • (A - B)

noncomputable def is_tangent_at (ω : Circle) (BC : Line) (D : Point) : Prop :=
  ω.is_tangent BC D

-- Defining main theorem
theorem point_A_on_line_GH (A B C D K L G H : Point)
  (ω : Circle) (h_iso : isosceles_triangle A B C)
  (h_tangent : ω.is_tangent BC D)
  (h_AK_BL : extend_point A B K = extend_point B A L)
  (h_KG : ω.second_intersection (Line.mk K D) = G) 
  (h_LH : ω.second_intersection (Line.mk L D) = H) :
  lies_on A (Line.mk G H) :=
sorry

end point_A_on_line_GH_l597_597574


namespace colored_points_exist_l597_597243

universe u

def Point := ℝ × ℝ

variable (color : Point → Bool) -- true for red, false for blue

theorem colored_points_exist (x : ℝ) (h : x > 0) :
  ∃ (p1 p2 : Point), color p1 = color p2 ∧ dist p1 p2 = x :=
by
  sorry

end colored_points_exist_l597_597243


namespace remaining_ribbon_l597_597999

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

end remaining_ribbon_l597_597999


namespace bulbs_in_bathroom_and_kitchen_l597_597203

theorem bulbs_in_bathroom_and_kitchen
  (bedroom_bulbs : Nat)
  (basement_bulbs : Nat)
  (garage_bulbs : Nat)
  (bulbs_per_pack : Nat)
  (packs_needed : Nat)
  (total_bulbs : Nat)
  (H1 : bedroom_bulbs = 2)
  (H2 : basement_bulbs = 4)
  (H3 : garage_bulbs = basement_bulbs / 2)
  (H4 : bulbs_per_pack = 2)
  (H5 : packs_needed = 6)
  (H6 : total_bulbs = packs_needed * bulbs_per_pack) :
  (total_bulbs - (bedroom_bulbs + basement_bulbs + garage_bulbs) = 4) :=
by
  sorry

end bulbs_in_bathroom_and_kitchen_l597_597203


namespace equation_no_solution_l597_597565

theorem equation_no_solution (x : ℝ) :
    (\sqrt (\sin x ^ 2 + 2) + 2 ^ x) / (\sqrt (\sin x ^ 2 + 2) + 2 ^ (x + 1)) +
    (\sqrt ((Real.log x) ^ 2 + 3) + 3 ^ x) / (\sqrt ((Real.log x) ^ 2 + 3) + 3 ^ (x + 1)) +
    (\sqrt (Real.exp x + 6) + 6 ^ x) / (\sqrt (Real.exp x + 6) + 6 ^ (x + 1)) > 1 := 
by
  sorry

end equation_no_solution_l597_597565


namespace player_1_points_after_13_rotations_l597_597690

theorem player_1_points_after_13_rotations :
  ∀ (table : ℕ → ℕ) (n : ℕ) (points : List ℕ),
    (∀ i, table i+16 = table i) →
    (table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]) →
    (points.length = 16) →
    (points.get 5 = 72) →
    (points.get 9 = 84) →
    (n = 13) →
    ((points.sum i₁, 0, 13) table ((stats : List ℕ) (i : fin 16) =>
      List.sum (List.take stats.toList) i.val + 
      List.sum (List.drop stats.toList i.val i.val + 2 * n) table) = points.sum table) →
    points.get 1 = 20 :=
by
  intros
  sorry

end player_1_points_after_13_rotations_l597_597690


namespace total_money_l597_597125

def JamesPocketBills : Nat := 3
def BillValue : Nat := 20
def WalletMoney : Nat := 75

theorem total_money (JamesPocketBills BillValue WalletMoney : Nat) : 
  (JamesPocketBills * BillValue + WalletMoney) = 135 :=
by
  sorry

end total_money_l597_597125


namespace quadratic_polynomial_integers_l597_597293

theorem quadratic_polynomial_integers (a b : ℝ) (h_a_prog : is_arith_prog (P (-1)^2) (P 0^2) (P 1^2)) :
  ∃ a_int b_int : ℤ, ↑a_int = a ∧ ↑b_int = b :=
begin
  sorry
end

noncomputable def P (x : ℝ) : ℝ := x^2 + a * x + b

def is_arith_prog (A B C : ℝ) : Prop :=
  2 * B = A + C

end quadratic_polynomial_integers_l597_597293


namespace eqDotProdDA_BE_l597_597980

-- Definitions of vectors and points
structure Point3D (α : Type) := (x y z : α)
structure Vec3D (α : Type) := (i j k : α)

open Point3D
open Vec3D

-- Dot product operation definition
def dotProd (v1 v2 : Vec3D ℝ) : ℝ :=
  v1.i * v2.i + v1.j * v2.j + v1.k * v2.k

-- Given conditions
def isEquilateralTriangle (A B C : Point3D ℝ) (side : ℝ) : Prop :=
  (A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2 = side^2 ∧
  (B.x - C.x)^2 + (B.y - C.y)^2 + (B.z - C.z)^2 = side^2 ∧
  (C.x - A.x)^2 + (C.y - A.y)^2 + (C.z - A.z)^2 = side^2

def midPoint (A B : Point3D ℝ) : Point3D ℝ :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2⟩

def isMidPointD (B C D : Point3D ℝ) : Prop :=
  D = midPoint B C

def AEHalfEC (A E C : Point3D ℝ) : Prop :=
  ∃ k : ℝ, k = 1 / 3 ∧ 
  E = ⟨A.x + k * (C.x - A.x), A.y + k * (C.y - A.y), A.z + k * (C.z - A.z)⟩

-- The main theorem to prove
theorem eqDotProdDA_BE {A B C D E : Point3D ℝ} (side : ℝ)
  (h1 : isEquilateralTriangle A B C side)
  (h2 : side = 2)
  (h3 : isMidPointD B C D)
  (h4 : AEHalfEC A E C) :
  dotProd (Vec3D.mk (D.x - A.x) (D.y - A.y) (D.z - A.z))
          (Vec3D.mk (B.x - E.x) (B.y - E.y) (B.z - E.z)) = 2 :=
by
  sorry

end eqDotProdDA_BE_l597_597980


namespace students_surveyed_l597_597761

theorem students_surveyed (S : ℕ) (h1 : 0.86 * S = 86 / 100 * S) (h2 : 0.14 * S = 70) : S = 500 := by
  -- formal math proof goes here
  sorry

end students_surveyed_l597_597761


namespace find_f_l597_597054

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (h : ∀ x, x ≠ -1 → f ((1-x) / (1+x)) = (1 - x^2) / (1 + x^2)) 
               (hx : x ≠ -1) :
  f x = 2 * x / (1 + x^2) :=
sorry

end find_f_l597_597054


namespace range_of_a_l597_597417

theorem range_of_a (a : ℝ) : (∀ x : ℝ, -1 ≤ x → x ≤ 1 → x^3 - a * x + 1 ≥ 0) → (0 ≤ a ∧ a ≤ (3 * Real.cbrt 2) / 2) :=
by
  intro h
  sorry

end range_of_a_l597_597417


namespace music_library_avg_disk_space_per_hour_l597_597303

theorem music_library_avg_disk_space_per_hour 
  (days_of_music: ℕ) (total_space_MB: ℕ) (hours_in_day: ℕ) 
  (h1: days_of_music = 15) 
  (h2: total_space_MB = 18000) 
  (h3: hours_in_day = 24) : 
  (total_space_MB / (days_of_music * hours_in_day)) = 50 := 
by
  sorry

end music_library_avg_disk_space_per_hour_l597_597303


namespace parallel_segments_l597_597262

variables {S S1 S2 : Circle} {A1 A2 C B1 B2 : Point}

-- Conditions
axiom equal_circles : S1 ≈ S2
axiom touch_internally_S : touch_internally S1 S A1 ∧ touch_internally S2 S A2
axiom C_on_S : on_circle C S
axiom CA1_intersects_S1 : intersects_at (segment C A1) S1 B1
axiom CA2_intersects_S2 : intersects_at (segment C A2) S2 B2

-- Prove that A1A2 is parallel to B1B2
theorem parallel_segments : segment A1 A2 ∥ segment B1 B2 :=
sorry

end parallel_segments_l597_597262


namespace number_of_integers_in_x_l597_597103

variable {α : Type} [Fintype α] [DecidableEq α] -- Assure we are dealing with finite sets with decidable equality

def symmetric_difference (x y : Set α) : Set α := (x \ y) ∪ (y \ x)

theorem number_of_integers_in_x
  (x y : Set ℤ)
  (hYCardinality : Fintype.card y = 18)
  (hXYIntersectionCardinality : Fintype.card (x ∩ y) = 6)
  (hSymmetricDifferenceCardinality : Fintype.card (symmetric_difference x y) = 14)
  : Fintype.card x = 8 :=
sorry

end number_of_integers_in_x_l597_597103


namespace color_cities_graph_l597_597472

-- Define the structure and conditions of the graph
def cities_graph (V : Type) [fintype V] (E : SimpleGraph V) (N : ℕ) :=
  ∀ v : V, ∃ (U : finset (finset V)),
  (∀ u ∈ U, u.card % 2 = 1 ∧ u ∖ {v} ⊆ E.neighbor_set v) ∧ U.card ≤ N

/-
  The main theorem statement
  Given a graph representing the cities and roads with the specified conditions,
  it is possible to color the graph using 2N + 2 colors such that no two adjacent vertices share the same color.
-/
theorem color_cities_graph {V : Type} [fintype V] (E : SimpleGraph V) (N : ℕ)
  (H : cities_graph V E N) : 
  ∃ C : V → fin (2 * N + 2),
  ∀ v w, E.Adj v w → C v ≠ C w :=
sorry

end color_cities_graph_l597_597472


namespace initial_women_count_l597_597974

-- Let x be the initial number of women.
-- Let y be the initial number of men.

theorem initial_women_count (x y : ℕ) (h1 : y = 2 * (x - 15)) (h2 : (y - 45) * 5 = (x - 15)) :
  x = 40 :=
by
  -- sorry to skip the proof
  sorry

end initial_women_count_l597_597974


namespace sam_wins_probability_l597_597551

theorem sam_wins_probability (hitting_probability missing_probability : ℚ)
    (hit_prob : hitting_probability = 2/5)
    (miss_prob : missing_probability = 3/5) : 
    let p := hitting_probability / (1 - missing_probability ^ 2)
    p = 5 / 8 :=
by
    sorry

end sam_wins_probability_l597_597551


namespace repeating_block_length_7_div_13_l597_597629

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l597_597629


namespace relatively_prime_pairs_in_subset_l597_597396

theorem relatively_prime_pairs_in_subset (A : Finset ℕ) (hA_card : A.card = 21) (h_cond : ∀ x ∈ A, 1 ≤ x ∧ x ≤ 40) : 
  ∃ x y ∈ A, Nat.gcd x y = 1 :=
by
  sorry

end relatively_prime_pairs_in_subset_l597_597396


namespace repeating_decimal_block_length_l597_597589

theorem repeating_decimal_block_length (n d : ℕ) (h : d ≠ 0) (hd : repeating_decimal n d) :  
  block_length n d = 6 :=
by
  sorry

end repeating_decimal_block_length_l597_597589


namespace count_distinct_circles_l597_597154

theorem count_distinct_circles (S : Type) [square S] : 
  ∃ (n : ℕ), n = 3 ∧
  (∀ (circle : Type) [diameter_endpoints : ∀ c, set (vertex S) → Prop],
     (diameter_based_circle : ∀ (c₁ c₂ : circle), diameter_endpoints c₁ = diameter_endpoints c₂ → c₁ = c₂) ∧
     (inscribed_circle : ∃ c, ∀ s, s ∈ S → distance (center c) (boundary s) = 1 / 2 * side_length S)) := sorry

end count_distinct_circles_l597_597154


namespace repeating_block_length_7_div_13_l597_597600

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l597_597600


namespace part1_part2_l597_597028

variables {a m n : ℝ}

theorem part1 (h1 : a^m = 2) (h2 : a^n = 3) : a^(4*m + 3*n) = 432 :=
by sorry

theorem part2 (h1 : a^m = 2) (h2 : a^n = 3) : a^(5*m - 2*n) = 32 / 9 :=
by sorry

end part1_part2_l597_597028


namespace number_of_rectangles_in_5x5_grid_l597_597726

theorem number_of_rectangles_in_5x5_grid : 
  (∑ i in Finset.range 6, i^3) = 225 := 
by 
  sorry

end number_of_rectangles_in_5x5_grid_l597_597726


namespace repeating_block_length_7_div_13_l597_597599

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l597_597599


namespace find_p_l597_597897

theorem find_p (p : ℝ) (α β : ℝ)
  (h_eq : ∀ x, Polynomial.eval x (Polynomial.C p + Polynomial.X * Polynomial.C p + Polynomial.X^2) = 0)
  (h_real_roots : ∃ α β : ℝ, h_eq α = 0 ∧ h_eq β = 0)
  (h_sum_squares : α^2 + β^2 = 3) : p = -1 :=
sorry

end find_p_l597_597897


namespace tank_depth_is_5_l597_597266

open Real

structure Tank :=
  (length : ℝ)
  (width : ℝ)
  (fill_rate : ℝ)
  (time_to_fill : ℝ)

def depth_of_tank (t : Tank) : ℝ :=
  (t.fill_rate * t.time_to_fill) / (t.length * t.width)

theorem tank_depth_is_5 :
  ∀ (t : Tank), 
  t.length = 10 → 
  t.width = 6 → 
  t.fill_rate = 5 → 
  t.time_to_fill = 60 → 
  depth_of_tank t = 5 :=
by
  intros t h_length h_width h_rate h_time
  unfold depth_of_tank
  rw [h_length, h_width, h_rate, h_time]
  unfold Real
  rw [mul_assoc, mul_comm 60 5, mul_assoc 10, mul_div_cancel, div_self]
  all_goals
    apply ne_of_gt
    iterate 2 apply zero_lt_mul_of_pos_left
    repeat apply zero_lt_bit1
    repeat apply zero_lt_of_lt_one
    exact zero_lt_one
  apply ne_of_gt
  exact zero_lt_bit1
  exact zero_lt_one

# this final theorem proves that the depth of the tank is 5 feet under the given conditions

end tank_depth_is_5_l597_597266


namespace black_rectangle_ways_l597_597739

theorem black_rectangle_ways : ∑ a in Finset.range 5, ∑ b in Finset.range 5, (5 - a) * (5 - b) = 225 := sorry

end black_rectangle_ways_l597_597739


namespace maxim_can_write_perfect_square_l597_597314

-- Define that a sequence of numbers follows the allowed operations
def valid_sequence (s : List ℕ) : Prop :=
  ∀ i < s.length - 1, (∃ d, d ∣ s[i] ∧ s[i+1] = s[i] + d) ∧ s[i] ≠ s[i+1]

-- Define a predicate for perfect squares
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Main theorem: Starting with any positive integer, Maxim can eventually write a perfect square on the board
theorem maxim_can_write_perfect_square (X : ℕ) (hX: X > 0) :
  ∃ s : List ℕ, valid_sequence s ∧ s.head = X ∧ is_perfect_square s.last :=
by
  sorry

end maxim_can_write_perfect_square_l597_597314


namespace eighteen_power_m_plus_n_l597_597568

theorem eighteen_power_m_plus_n (m n : ℤ) (R S : ℤ)
  (hR : R = 2^m) (hS : S = 3^n) :
  18^(m + n) = R^(m + n) * S^(2 * (m + n)) :=
sorry

end eighteen_power_m_plus_n_l597_597568


namespace compare_sqrt_l597_597839

theorem compare_sqrt (h1 : (sqrt 3)^2 = 3) (h2 : (sqrt 2)^2 = 2) : 2 * sqrt 3 < 3 * sqrt 2 := 
by
  have t1 : (2 * sqrt 3)^2 = 4 * 3 := by { calc (2 * sqrt 3)^2 = 2^2 * (sqrt 3)^2 : by ring 
                                          ... = 4 * 3 : by rw [h1] }
  have t2 : (3 * sqrt 2)^2 = 9 * 2 := by { calc (3 * sqrt 2)^2 = 3^2 * (sqrt 2)^2 : by ring 
                                          ... = 9 * 2 : by rw [h2] }
  rw [t1, t2]
  have comp : 12 < 18 := by norm_num
  apply lt_of_pow_two_lt_pow_two;
  assumption

end compare_sqrt_l597_597839


namespace find_zero_velocity_times_l597_597060

noncomputable def s (t : ℝ) : ℝ := (1 / 3) * t^3 - 4 * t^2 + 12 * t

theorem find_zero_velocity_times :
  ∀ t : ℝ, (derivative (s t)) = 0 ↔ t = 2 ∨ t = 6 := by
  sorry

end find_zero_velocity_times_l597_597060


namespace find_root_power_117_l597_597098

noncomputable def problem (a b c : ℝ) (x1 x2 : ℝ) :=
  (3 * a - b) / c * x1^2 + c * (3 * a + b) / (3 * a - b) = 0 ∧
  (3 * a - b) / c * x2^2 + c * (3 * a + b) / (3 * a - b) = 0 ∧
  x1 + x2 = 0

theorem find_root_power_117 (a b c : ℝ) (x1 x2 : ℝ) (h : problem a b c x1 x2) : 
  x1 ^ 117 + x2 ^ 117 = 0 :=
sorry

end find_root_power_117_l597_597098


namespace sam_wins_probability_l597_597550

theorem sam_wins_probability (hitting_probability missing_probability : ℚ)
    (hit_prob : hitting_probability = 2/5)
    (miss_prob : missing_probability = 3/5) : 
    let p := hitting_probability / (1 - missing_probability ^ 2)
    p = 5 / 8 :=
by
    sorry

end sam_wins_probability_l597_597550


namespace sum_of_8th_and_10th_terms_arithmetic_sequence_l597_597570

theorem sum_of_8th_and_10th_terms_arithmetic_sequence (a d : ℤ)
  (h1 : a + 3 * d = 25) (h2 : a + 5 * d = 61) :
  (a + 7 * d) + (a + 9 * d) = 230 := 
sorry

end sum_of_8th_and_10th_terms_arithmetic_sequence_l597_597570


namespace determine_a2016_l597_597446

noncomputable def a_n (n : ℕ) : ℤ := sorry
noncomputable def S_n (n : ℕ) : ℤ := sorry

axiom S1 : S_n 1 = 6
axiom S2 : S_n 2 = 4
axiom S_pos (n : ℕ) : S_n n > 0
axiom geom_progression (n : ℕ) : (S_n (2 * n - 1))^2 = S_n (2 * n) * S_n (2 * n + 2)
axiom arith_progression (n : ℕ) : 2 * S_n (2 * n + 2) = S_n (2 * n - 1) + S_n (2 * n + 1)

theorem determine_a2016 : a_n 2016 = -1009 :=
by sorry

end determine_a2016_l597_597446


namespace donut_holes_count_l597_597659

theorem donut_holes_count :
  ∀ (number_of_students number_of_mini_cupcakes desserts_per_student : ℕ),
  number_of_students = 13 →
  number_of_mini_cupcakes = 14 →
  desserts_per_student = 2 →
  (number_of_students * desserts_per_student - number_of_mini_cupcakes) = 12 :=
by
  intros number_of_students number_of_mini_cupcakes desserts_per_student h1 h2 h3
  simp [h1, h2, h3]
  exact sorry

#print axioms donut_holes_count -- this ensures that all the assumptions were utilized properly.

end donut_holes_count_l597_597659


namespace no_solution_l597_597174

def g (n : ℕ) (x : ℝ) : ℝ := (Real.sin x)^n + (Real.cos x)^n

theorem no_solution (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  10 * g 8 x - 6 * g 10 x ≠ 2 * g 2 x := by
sorry

end no_solution_l597_597174


namespace diameter_le_h_diameter_eq_h_DeltaBigon_l597_597538

-- Define a Pi-curve and a Delta-bigon in geometrical terms
-- Assuming some properties geometrically (not explicitly detailed in the problem)

noncomputable theory

-- Let Point and Curve be base types
constant Point : Type 
constant Curve : Type

-- Assume PiCurve and DeltaBigon are specific types of curves with given properties
constant PiCurve : Curve → Prop
constant DeltaBigon : Curve → Prop

-- Define that K is a Pi-curve with height h (a positive number)
variables (K : Curve) (h : ℝ)
variable (h_pos : 0 < h)

-- Assume that K is a Pi-curve and h is its height
axiom K_is_PiCurve : PiCurve K
axiom height_K : ∀ p q : Point, K.contains p → K.contains q → dist p q ≤ h

-- Lean statement to prove the main question

theorem diameter_le_h : ∀ p q : Point, K.contains p → K.contains q → dist p q ≤ h := 
sorry -- Proof not provided

theorem diameter_eq_h_DeltaBigon : (∃ p q : Point, K.contains p ∧ K.contains q ∧ dist p q = h) → DeltaBigon K := 
sorry -- Proof not provided

end diameter_le_h_diameter_eq_h_DeltaBigon_l597_597538


namespace find_k_l597_597007

def special_operation (n : ℕ) : ℕ :=
  ∏ i in finset.range (n), (i+1)^(n-i)

theorem find_k (k : ℕ) : 
  special_operation 7 * special_operation 9 = special_operation 5 * special_operation k :=
  by
    have h7 : special_operation 7 = 1^7 * 2^6 * 3^5 * 4^4 * 5^3 * 6^2 * 7^1 := sorry,
    have h9 : special_operation 9 = 1^9 * 2^8 * 3^7 * 4^6 * 5^5 * 6^4 * 7^3 * 8^2 * 9^1 := sorry,
    have h5 : special_operation 5 = 1^5 * 2^4 * 3^3 * 4^2 * 5^1 := sorry,
    have h_k : special_operation k = 1^k * 2^(k-1) * 3^(k-2) * 4^(k-3) * 5^(k-4) * 6^(k-5) * 7^(k-6) * 8^(k-7) * 9^(k-8) * 10^(k-9) := sorry,
    -- Prove that for k = 10, the equation holds
    sorry

-- The value of k should be:
example : find_k 10 = rfl := rfl

end find_k_l597_597007


namespace find_Sn_l597_597158

noncomputable def a : ℕ → ℝ
| 1     := 1
| (n+1) := 1 + 1 / a n + Real.log (a n)

noncomputable def S (n : ℕ) : ℤ :=
(∑ i in Finset.range n, Nat.floor (a (i + 1)))

theorem find_Sn (n : ℕ) : S n = 2 * n - 1 :=
sorry

end find_Sn_l597_597158


namespace first_bakery_sacks_per_week_l597_597814

theorem first_bakery_sacks_per_week (x : ℕ) 
    (H1 : 4 * x + 4 * 4 + 4 * 12 = 72) : x = 2 :=
by 
  -- we will provide the proof here if needed
  sorry

end first_bakery_sacks_per_week_l597_597814


namespace line_equation_with_slope_45_l597_597419

theorem line_equation_with_slope_45° (x y : ℝ) :
  (∃ l : ℝ → ℝ, (∀ x, l x = x + 1) ∧ l 0 = 1) → (x - y + 1 = 0) := by
  sorry

end line_equation_with_slope_45_l597_597419


namespace exists_real_x_alpha_fractional_part_l597_597170

theorem exists_real_x_alpha_fractional_part (α : ℝ) (hα1 : 0 < α) (hα2 : α < 1) :
  ∃ (x : ℝ), (0 < x) ∧ (x < 1) ∧ ∀ n : ℕ, 0 < n → α^n < (n * x - Real.floor (n * x)) :=
by
  sorry

end exists_real_x_alpha_fractional_part_l597_597170


namespace tennis_players_win_cycle_l597_597004

theorem tennis_players_win_cycle 
  (n : ℕ) (h_n : n > 6)
  (win_records : ∀ (p q : ℕ) (h1 : p ≤ n) (h2 : q ≤ n), (∃ r : ℕ, r ≤ n ∧ r ≠ p ∧ r ≠ q ∧ (r, p) ∈ G.E ∧ (r, q) ∈ G.E))
  (k l : ℕ) (h_kl_ineq : 2^(2^k + 1) - 1 > n) (h_l_bounds : 1 < l ∧ l < 2 * k + 1):
  ∃ (A : fin n → ℕ), (∀ i, (A (i + 1) % l, A i % l) ∈ G.E) :=
sorry

end tennis_players_win_cycle_l597_597004


namespace probability_B_winning_is_448_3375_participant_A_higher_probability_of_winning_l597_597189

noncomputable def probability_B_wins_third_round : ℚ :=
  let p_a_correct := 3/5
  let p_b_correct := 2/3
  let event_prob (n m : ℕ) : ℚ :=
    if m = 0 then (choose 3 n) * (p_b_correct ^ n) * (1 - p_b_correct) * (1 - p_a_correct) ^ 3
    else if n = 3 then (p_b_correct ^ 3) * (choose 3 m) * (p_a_correct ^ m)
    else 0 -- Non-covered cases are 0 (By given problem constraints)
  event_prob 2 0 + event_prob 3 0 + event_prob 3 1

theorem probability_B_winning_is_448_3375 :
  probability_B_wins_third_round = 448 / 3375 :=
by sorry

noncomputable def expected_additional_time (p : ℚ) (n : ℕ) : ℚ :=
  20 * n * (1 - p)

theorem participant_A_higher_probability_of_winning :
  let recite_diff := 30
  let p_a_correct := 3/5
  let p_b_correct := 2/3
  let a_incorrect := expected_additional_time (2/5) 9
  let b_incorrect := expected_additional_time (1/3) 9
  recite_diff + a_incorrect < b_incorrect :=
by sorry

end probability_B_winning_is_448_3375_participant_A_higher_probability_of_winning_l597_597189


namespace union_of_sets_l597_597076

open Set

theorem union_of_sets (M N : Set ℝ) (hM : M = {x | -3 < x ∧ x < 1}) (hN : N = {x | x ≤ -3}) :
  M ∪ N = {x | x < 1} := by
  sorry

end union_of_sets_l597_597076


namespace total_money_l597_597127

def JamesPocketBills : Nat := 3
def BillValue : Nat := 20
def WalletMoney : Nat := 75

theorem total_money (JamesPocketBills BillValue WalletMoney : Nat) : 
  (JamesPocketBills * BillValue + WalletMoney) = 135 :=
by
  sorry

end total_money_l597_597127


namespace product_sequence_equals_l597_597833

-- Define the form of each fraction in the sequence
def frac (k : ℕ) : ℚ := (4 * k : ℚ) / (4 * k + 4)

-- Define the product of the sequence from k=1 to k=501
def productSequence : ℚ :=
  (finset.range 501).prod (λ k => frac (k + 1))

-- The theorem that the product equals 1/502
theorem product_sequence_equals : productSequence = 1 / 502 := by
  sorry

end product_sequence_equals_l597_597833


namespace player_1_points_after_13_rotations_l597_597691

theorem player_1_points_after_13_rotations :
  ∀ (table : ℕ → ℕ) (n : ℕ) (points : List ℕ),
    (∀ i, table i+16 = table i) →
    (table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]) →
    (points.length = 16) →
    (points.get 5 = 72) →
    (points.get 9 = 84) →
    (n = 13) →
    ((points.sum i₁, 0, 13) table ((stats : List ℕ) (i : fin 16) =>
      List.sum (List.take stats.toList) i.val + 
      List.sum (List.drop stats.toList i.val i.val + 2 * n) table) = points.sum table) →
    points.get 1 = 20 :=
by
  intros
  sorry

end player_1_points_after_13_rotations_l597_597691


namespace measure_angle_DAB_l597_597483

-- Given a regular hexagon ABCDEF
variables (A B C D E F : Type) [RegularHexagon A B C D E F]

-- Premises
axiom interior_angles : ∀ (a b c d e f : Type), a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f → angle (a, b, c) = 120

-- Definitions related to the problem
def angle_DAB (A B D : Type) : ℝ := -- The measure of angle DAB

theorem measure_angle_DAB (A B C D E F : Type)
  [RegularHexagon A B C D E F]
  (h : ∀ (a b c d e f : Type), a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f) :
  angle_DAB A B D = 30 :=
sorry

end measure_angle_DAB_l597_597483


namespace pet_shop_legs_l597_597816

theorem pet_shop_legs :
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  birds * bird_legs + dogs * dog_legs + snakes * snake_legs + spiders * spider_legs = 34 := 
by
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  sorry

end pet_shop_legs_l597_597816


namespace repeat_block_of_7_div_13_l597_597635

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l597_597635


namespace triangle_BT_eq_2PT_l597_597149

/-- Let ABC be a triangle with AB = AC and ∠BAC = 40°. Points S and T lie on the sides
    AB and BC, such that ∠BAT = ∠BCS = 10°. Lines AT and CS meet at P. Prove that BT = 2PT. -/
theorem triangle_BT_eq_2PT 
  (A B C S T P : Point)
  (h1 : AB = AC)
  (h2 : angle BAC = 40)
  (h3 : S ∈ line AB)
  (h4 : T ∈ line BC)
  (h5 : angle BAT = 10)
  (h6 : angle BCS = 10)
  (h7 : line AT = line_intersection AT CS)
  :
  length (segment BT) = 2 * length (segment PT) :=
sorry

end triangle_BT_eq_2PT_l597_597149


namespace sam_wins_probability_l597_597542

theorem sam_wins_probability : 
  let hit_prob := (2 : ℚ) / 5
      miss_prob := (3 : ℚ) / 5
      p := hit_prob + (miss_prob * miss_prob) * p
  in p = 5 / 8 := 
by
  -- Proof goes here
  sorry

end sam_wins_probability_l597_597542


namespace solution_set_of_inequality1_solution_set_of_inequality2_l597_597000

-- First inequality problem
theorem solution_set_of_inequality1 :
  {x : ℝ | x^2 + 3*x + 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
sorry

-- Second inequality problem
theorem solution_set_of_inequality2 :
  {x : ℝ | -3*x^2 + 2*x + 2 < 0} =
  {x : ℝ | x ∈ Set.Iio ((1 - Real.sqrt 7) / 3) ∪ Set.Ioi ((1 + Real.sqrt 7) / 3)} :=
sorry

end solution_set_of_inequality1_solution_set_of_inequality2_l597_597000


namespace relationship_between_abc_l597_597161

theorem relationship_between_abc :
  let a := 2 ^ 0.3
  let b := 0.3 ^ 2
  let c := log 2 0.3
  a > b ∧ b > c :=
by
  sorry

end relationship_between_abc_l597_597161


namespace plane_divides_segment_DG_l597_597191

theorem plane_divides_segment_DG (K L M D A B C G : Point) (DK DA DL DB DM DC : ℝ) :
  DK = (1 / 2) * DA → DL = (2 / 5) * DB → DM = (3 / 4) * DC → is_centroid G A B C →
  divides_segment_in_ratio (plane K L M) D G (18 / 17) :=
by
  sorry

end plane_divides_segment_DG_l597_597191


namespace fifth_layer_has_61_dots_l597_597260

-- Conditions:
-- 1. The first hexagon (first layer) has only 1 dot.
-- 2. Each new layer forms a hexagon with each side containing an increasing number of dots equal to the number of the layer.

def dots_in_layer (n : ℕ) : ℕ :=
  if n = 1 then 1 else 6 * (n - 1)

noncomputable def total_dots (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, dots_in_layer (k + 1))

theorem fifth_layer_has_61_dots : total_dots 5 = 61 := by
  sorry

end fifth_layer_has_61_dots_l597_597260


namespace correct_addition_result_l597_597755

-- Definitions corresponding to the conditions
def mistaken_addend := 240
def correct_addend := 420
def incorrect_sum := 390

-- The proof statement
theorem correct_addition_result : 
  (incorrect_sum - mistaken_addend + correct_addend) = 570 :=
by
  sorry

end correct_addition_result_l597_597755


namespace equilateral_triangles_count_l597_597232

theorem equilateral_triangles_count :
  let k_values := (-5:ℤ) to 5
  ∃ (arrangement : set (ℝ × ℝ → ℝ)),
  (∀ k ∈ k_values, (λ x:ℝ, x) ∈ arrangement) ∧
  (∀ k ∈ k_values, (λ x:ℝ, abs.sqrt(3) * x + k) ∈ arrangement) ∧
  (∀ k ∈ k_values, (λ x:ℝ, - abs.sqrt(3) * x + k) ∈ arrangement) ∧
  (∃ (equilateral_triangles: set (ℝ × ℝ)),
  (∀ triangle ∈ equilateral_triangles, triangle.has_side_length (1 / abs.sqrt(3))) ∧
  card(equilateral_triangles) = 630)
  :=
by sorry

end equilateral_triangles_count_l597_597232


namespace range_of_m_l597_597046

open Real

variable {f : ℝ → ℝ}
variable m : ℝ
variable a : ℝ

-- Define proposition p
def p : Prop := ∀ x, x > -7 → f x = 2 / (x - m) → ∃ c, (x - c) + ((f x) - (f c)) < 0

-- Define proposition q
def q : Prop := ∀ a ∈ Icc (-1 : ℝ) 1, m^2 + 5 * m - 3 ≥ sqrt (a^2 + 8)

-- Logical condition
theorem range_of_m (h : ¬p ∧ q) : (-7 < m ∧ m ≤ -6) ∨ (1 ≤ m) :=
sorry

end range_of_m_l597_597046


namespace equiangular_polygon_is_regular_hexagon_l597_597304

def equiangular (polygon : Type) := ∀ (angles : list ℝ), angles = (list.repeat (angles.head) (list.length angles))

inductive Polygon 
| Rectangle 
| RegularHexagon 
| Rhombus 
| EquilateralTriangle 
| IsoscelesTrapezoid 

open Polygon 

theorem equiangular_polygon_is_regular_hexagon (p : Polygon) : 
  equiangular p → p = RegularHexagon :=
by 
  sorry

end equiangular_polygon_is_regular_hexagon_l597_597304


namespace bisection_third_iteration_l597_597119

noncomputable def bisection_method_third_iteration_interval 
(interval_0 : Set ℝ)
(midpoint : ℝ → ℝ → ℝ)
(where : ℝ → ℝ → Bool)
(f : ℝ → ℝ) : Set ℝ :=
  if where (midpoint (-2) 4) (f (midpoint (-2) 4))
  then
    let interval_1 := if where (midpoint (-2) 1) (f (midpoint (-2) 1)) then set.Icc (-2) (-1 / 2) else set.Icc (-1 / 2) 1
    in
    if where (midpoint (-2) (-1 / 2)) (f (midpoint (-2) (-1 / 2))) then set.Icc (-2) (-5 / 4) else set.Icc (-5 / 4) 1
  else
    let interval_1 := if where (midpoint 1 4) (f (midpoint 1 4)) then set.Icc 1 (5 / 2) else set.Icc (5 / 2) 4
    in
    if where (midpoint 1 (5 / 2)) (f (midpoint 1 (5 / 2))) then set.Icc 1 (3 / 4) else set.Icc (3 / 4) 4

theorem bisection_third_iteration :
  bisection_method_third_iteration_interval (set.Icc (-2) 4)
    (fun a b => (a + b) / 2)
    (fun mid _ => (0 : Bool)) -- Placeholder condition function
    (fun x => (0 : ℝ)) = set.Icc (-1 / 2) 1 :=
by
  sorry

end bisection_third_iteration_l597_597119


namespace initial_values_l597_597032

def recursive_sequence (x₀ : ℝ) (n : ℕ) : ℝ :=
if 2 * (recursive_sequence x₀ (n - 1)) < 1 then
  2 * (recursive_sequence x₀ (n - 1))
else
  2 * (recursive_sequence x₀ (n - 1)) - 1

noncomputable def initial_values_count : ℕ :=
64

theorem initial_values (x₀ : ℝ) (h : 0 ≤ x₀ ∧ x₀ < 1) :
  (∃ (n : ℕ), n > 0 ∧ x₀ = recursive_sequence x₀ 6) →
  x₀ = initial_values_count :=
sorry

end initial_values_l597_597032


namespace evaluate_expression_l597_597877

theorem evaluate_expression : 
  abs (abs (-abs (3 - 5) + 2) - 4) = 4 :=
by
  sorry

end evaluate_expression_l597_597877


namespace cos_pi_minus_alpha_l597_597413

theorem cos_pi_minus_alpha (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : 0 < α ∧ α < π / 2) :
  Real.cos (π - α) = -12 / 13 :=
sorry

end cos_pi_minus_alpha_l597_597413


namespace arithmetic_sequence_nth_term_l597_597226

noncomputable def nth_arithmetic_term (a : ℤ) (n : ℕ) : ℤ :=
  let a1 := a - 1
  let a2 := a + 1
  let a3 := 2 * a + 3
  if 2 * (a + 1) = (a - 1) + (2 * a + 3) then
    -1 + (n - 1) * 2
  else
    sorry

theorem arithmetic_sequence_nth_term (a : ℤ) (n : ℕ) (h : 2 * (a + 1) = (a - 1) + (2 * a + 3)) :
  nth_arithmetic_term a n = 2 * (n : ℤ) - 3 :=
by
  sorry

end arithmetic_sequence_nth_term_l597_597226


namespace find_mn_l597_597366

noncomputable def line1_satisfies_conditions (m n : ℝ) : Prop :=
  (∃ y_intercept : ℝ, (-m * 0 + n / 2 * y_intercept - 1 = 0 ∧ y_intercept = -1)) ∧
  (∃ (θ1 θ2 : ℝ), (θ1 = Real.arctan (sqrt 3) ∧ θ2 = 2 * θ1 ∧ m = Real.tan θ2 ∧ θ1 = Real.pi / 3))

theorem find_mn : line1_satisfies_conditions (-sqrt 3) (-2) :=
by
  sorry

end find_mn_l597_597366


namespace find_m_plus_n_l597_597502

def triangle_sides := ∀ (A B C P : Point), (A.distance B = 10) ∧ (B.distance C = 10) ∧ (A.distance C = 12)
def point_on_side_bc := ∀ (P : Point), P ∈ interior(B, C)
def perpendiculars := ∀ (P : Point) (AB AC : Segment), (X : Point), (Y : Point), X = feet_perpendicular(P, AB) ∧ Y = feet_perpendicular(P, AC)
def minimum_px2_py2 := ∃ m n : ℕ, gcd m n = 1 ∧ PX^2 + PY^2 = m / n

theorem find_m_plus_n :
  ∀ (A B C P : Point) (PX PY : ℝ) (m n : ℕ),
  triangle_sides A B C P →
  point_on_side_bc P →
  perpendiculars P (A.distance B) (A.distance C) →
  (PX^2 + PY^2 = 1875 / 61) →
  (m = 1875) ∧ (n = 61) →
  (m + n = 1936) :=
begin
  sorry,
end

end find_m_plus_n_l597_597502


namespace triangle_inequality_criterion_l597_597751

theorem triangle_inequality_criterion :
  (let A := (3, 4, 9) in (A.1 + A.2 > A.3) ∧ (A.1 + A.3 > A.2) ∧ (A.2 + A.3 > A.1)) = false ∧
  (let B := (8, 7, 15) in (B.1 + B.2 > B.3) ∧ (B.1 + B.3 > B.2) ∧ (B.2 + B.3 > B.1)) = false ∧
  (let C := (13, 12, 20) in (C.1 + C.2 > C.3) ∧ (C.1 + C.3 > C.2) ∧ (C.2 + C.3 > C.1)) = true ∧
  (let D := (5, 5, 11) in (D.1 + D.2 > D.3) ∧ (D.1 + D.3 > D.2) ∧ (D.2 + D.3 > D.1)) = false :=
by sorry

end triangle_inequality_criterion_l597_597751


namespace range_eq_domain_l597_597070

def f (x : ℝ) : ℝ := |x - 2| - 2

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem range_eq_domain : (Set.range f) = M :=
by
  sorry

end range_eq_domain_l597_597070


namespace abs_sum_less_than_two_l597_597893

theorem abs_sum_less_than_two (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) : |a + b| + |a - b| < 2 := 
sorry

end abs_sum_less_than_two_l597_597893


namespace sin_780_eq_sqrt3_div_2_l597_597355

theorem sin_780_eq_sqrt3_div_2 : Real.sin (780 * Real.pi / 180) = Math.sqrt 3 / 2 := by
  sorry

end sin_780_eq_sqrt3_div_2_l597_597355


namespace trajectory_midpoint_AE_l597_597918

theorem trajectory_midpoint_AE :
  ∀ {x y k : ℝ},
    ((∃ k, k > 0 ∧ (∃ A B E F : ℝ, 
      ∃ xy : ℝ, xy = (x,y) ∧
      (2 + has_sqrt.sqrt (k - 16) = E) ∧
      (4 + has_sqrt.sqrt (k - 4) = A) ∧ 
      (∃ AE : ℝ, 2 + has_sqrt.sqrt(k-4) = AE) ∧
      (x = 1 + (has_sqrt.sqrt(k-16) / 2)) ∧ 
      (y = 2 + (has_sqrt.sqrt(k-4) / 2)))) → 
      (x > 1) ∧ (y > 2 + has_sqrt.sqrt 3) →
      (y - 2)^2 - (x - 1)^2 = 3)
by sorry

end trajectory_midpoint_AE_l597_597918


namespace number_of_rectangles_in_5x5_grid_l597_597727

theorem number_of_rectangles_in_5x5_grid : 
  (∑ i in Finset.range 6, i^3) = 225 := 
by 
  sorry

end number_of_rectangles_in_5x5_grid_l597_597727


namespace probability_of_even_product_is_13_div_18_l597_597017

open_locale classical

-- Define the set of 9 pieces of paper labeled 1 to 9
def paper_set : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the event that the product of two numbers is even
def even_product_event (x y : ℕ) : Prop := (x * y) % 2 = 0

-- Define the probability of drawing two papers and their product being even
noncomputable def probability_even_product : ℚ :=
probs.pairwise_event_probability paper_set even_product_event

-- The goal is to prove that this probability is 13/18
theorem probability_of_even_product_is_13_div_18 :
  probability_even_product = 13 / 18 :=
sorry

end probability_of_even_product_is_13_div_18_l597_597017


namespace problem_water_percentage_l597_597785

noncomputable def percentage_water_in_mixture 
  (volA volB volC volD : ℕ) 
  (pctA pctB pctC pctD : ℝ) : ℝ :=
  let total_volume := volA + volB + volC + volD
  let total_solution := volA * pctA + volB * pctB + volC * pctC + volD * pctD
  let total_water := total_volume - total_solution
  (total_water / total_volume) * 100

theorem problem_water_percentage :
  percentage_water_in_mixture 100 90 60 50 0.25 0.3 0.4 0.2 = 71.33 :=
by
  -- proof goes here
  sorry

end problem_water_percentage_l597_597785


namespace find_2011_otimes_2011_l597_597884

-- Define the operation ⊗
def otimes (a b : ℝ) : ℝ 

-- Define the properties of the operation
axiom axiom1 : ∀ (a b n : ℝ), otimes a b = n → otimes (a + 1) b = n + 1
axiom axiom2 : ∀ (a b n : ℝ), otimes a b = n → otimes a (b + 1) = n - 2

-- Given initial condition
axiom initial_condition : otimes 1 1 = 2

-- The statement to be proven
theorem find_2011_otimes_2011 : otimes 2011 2011 = -2008 :=
sorry

end find_2011_otimes_2011_l597_597884


namespace count_rectangles_in_5x5_grid_l597_597733

theorem count_rectangles_in_5x5_grid :
  let n := 5 in
  ∑ x in finset.range n.succ, (n - x) * (n - x) * (n - x) = 225 :=
by {
  simp only [finset.range, finset.sum],
  sorry
}

end count_rectangles_in_5x5_grid_l597_597733


namespace player1_points_after_13_rotations_l597_597694

theorem player1_points_after_13_rotations :
  let sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
  let player_points (player : Nat) (rotations : Nat) :=
      rotations • (λ i, sector_points[(i + player) % 16])
  player_points 5 13 = 72 ∧ player_points 9 13 = 84 → player_points 1 13 = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597694


namespace ellipse_eccentricity_l597_597410

theorem ellipse_eccentricity (O F A B P M E : Point)
  (a b c : ℝ) (k : ℝ) (C : Ellipse)
  (h1 : O = ⟨0,0⟩)
  (h2 : F = ⟨-c,0⟩)
  (h3 : A = ⟨-a,0⟩)
  (h4 : B = ⟨a,0⟩)
  (h5 : C = Ellipse.mk ⟨O, a, b⟩)
  (h6 : c ≠ 0) (h7 : a ≠ 0) (h8 : b ≠ 0) (h9 : a > b) (h10 : b > 0)
  (h11 : P = ⟨-c, b * sqrt (1 - (c^2 / a^2))⟩ ∨ P = ⟨-c, - (b * sqrt (1 - (c^2 / a^2)))⟩)
  (h12 : PF ⟂ Line.x_axis)
  (h13 : Line.through A P M)
  (h14 : Line.through A E)
  (h15 : E.y = k * a)
  (h16 : Line.through B M H)
  (h17 : Midpoint O E H) :
  eccentricity C = 1 / 3 :=
sorry

end ellipse_eccentricity_l597_597410


namespace colleen_paid_more_than_joy_l597_597145

def price_policy (n : ℕ) : ℕ :=
  if n < 20 then 4 else if n < 40 then 35 / 10 else 3

def total_cost (quantities : List ℕ) : ℕ :=
  quantities.map (λ q, q * price_policy q).sum

def joy_purchases := [10, 15, 5]
def colleen_purchases := [25, 25]

theorem colleen_paid_more_than_joy :
  total_cost colleen_purchases - total_cost joy_purchases = 55 :=
by
  sorry

end colleen_paid_more_than_joy_l597_597145


namespace sin_780_eq_sqrt3_over_2_l597_597351

theorem sin_780_eq_sqrt3_over_2 :
  sin (780 : ℝ) = (Real.sqrt 3 / 2) :=
by
  sorry

end sin_780_eq_sqrt3_over_2_l597_597351


namespace sum_ratio_arithmetic_sequence_l597_597155

theorem sum_ratio_arithmetic_sequence (a₁ d : ℚ) (h : d ≠ 0) 
  (S : ℕ → ℚ)
  (h_sum : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2)
  (h_ratio : S 3 / S 6 = 1 / 3) :
  S 6 / S 12 = 3 / 10 :=
by
  sorry

end sum_ratio_arithmetic_sequence_l597_597155


namespace cost_price_of_watch_l597_597803

theorem cost_price_of_watch (C : ℝ) 
  (h1 : ∃ (SP1 SP2 : ℝ), SP1 = 0.54 * C ∧ SP2 = 1.04 * C ∧ SP2 = SP1 + 140) : 
  C = 280 :=
by
  obtain ⟨SP1, SP2, H1, H2, H3⟩ := h1
  sorry

end cost_price_of_watch_l597_597803


namespace oxygen_part_weight_l597_597381

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O
def given_molecular_weight : ℝ := 108

theorem oxygen_part_weight : molecular_weight_N2O = 44.02 → atomic_weight_O = 16.00 := by
  sorry

end oxygen_part_weight_l597_597381


namespace volume_ratio_of_cubes_l597_597748

theorem volume_ratio_of_cubes :
  (4^3 / 10^3 : ℚ) = 8 / 125 := by
  sorry

end volume_ratio_of_cubes_l597_597748


namespace minimum_rubles_to_reverse_chips_l597_597368

theorem minimum_rubles_to_reverse_chips (n : ℕ) (h : n = 100)
  (adjacent_cost : ℕ → ℕ → ℕ)
  (free_cost : ℕ → ℕ → Prop)
  (reverse_cost : ℕ) :
  (∀ i j, i + 1 = j → adjacent_cost i j = 1) →
  (∀ i j, i + 5 = j → free_cost i j) →
  reverse_cost = 61 :=
by
  sorry

end minimum_rubles_to_reverse_chips_l597_597368


namespace problem1_problem2_problem3a_problem3b_l597_597928

noncomputable def f (x a : ℝ) : ℝ := Real.log x + x^2 - 2 * a * x + a^2

theorem problem1 : f 1 0 = 1 :=
by sorry

theorem problem2 (a : ℝ) (h : ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → (1/x + 2*x - 2*a) > 0) :
  a < 3/2 :=
by sorry

theorem problem3a (a : ℝ) (h : a ≤ Real.sqrt 2) : ¬∃ x : ℝ, Deriv f x a = 0 :=
by sorry

theorem problem3b (a : ℝ) (h : a > Real.sqrt 2) :
  ∃ x₁ x₂ : ℝ, x₁ = (a - Real.sqrt (a^2 - 2)) / 2 ∧ x₂ = (a + Real.sqrt (a^2 - 2)) / 2 ∧
  Deriv f x₁ a = 0 ∧ Deriv f x₂ a = 0 :=
by sorry

end problem1_problem2_problem3a_problem3b_l597_597928


namespace sin_780_eq_sqrt3_div_2_l597_597345

theorem sin_780_eq_sqrt3_div_2 :
  Real.sin (780 * Real.pi / 180) = (Real.sqrt 3) / 2 :=
by
  sorry

end sin_780_eq_sqrt3_div_2_l597_597345


namespace repeating_block_length_7_div_13_l597_597620

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597620


namespace true_proposition_l597_597903

-- Define the propositions
def p : Prop := ∀ x : ℝ, f x = abs (cos x) ∧ ¬ (exists T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x ∧ T < 2 * π))
def q : Prop := ∃ x : ℝ, 2^x > 3^x

-- Main theorem to be proved
theorem true_proposition :
  ¬p ∧ q → (p ∨ q) :=
by
  intro h,
  cases h with hp hq,
  exact Or.inr hq

end true_proposition_l597_597903


namespace probability_symmetric_sum_l597_597225

-- Define the conditions of the problem
def isStandardDie (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

def sum_of_dice_is (dice : List ℕ) (target : ℕ) : Prop := 
  dice.length = 8 ∧ 
  (∀ d ∈ dice, isStandardDie d) ∧ 
  list.sum dice = target

-- State the proof problem
theorem probability_symmetric_sum (dice : List ℕ) :
  sum_of_dice_is dice 15 → sum_of_dice_is dice 41 :=
by sorry

end probability_symmetric_sum_l597_597225


namespace repeating_block_length_7_div_13_l597_597576

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597576


namespace repeating_block_length_7_div_13_l597_597622

theorem repeating_block_length_7_div_13 : 
  let d := decimalExpansion 7 13 
  in minimalRepeatingBlockLength d = 6 :=
sorry

end repeating_block_length_7_div_13_l597_597622


namespace arithmetic_seq_a1_eq_4_l597_597525

theorem arithmetic_seq_a1_eq_4 (a_n : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n m : ℕ, a_n (m+1) = a_n m + d)
  (h_d_ne_zero : d ≠ 0)
  (h_a5_eq_a1_squared : a_n 5 = (a_n 1)^2)
  (h_geom_mean_cond : sqrt ((a_n 1) * (a_n 21)) = a_n 5) :
  a_n 1 = 4 :=
sorry

end arithmetic_seq_a1_eq_4_l597_597525


namespace find_actual_weights_l597_597257

noncomputable def melon_weight : ℝ := 4.5
noncomputable def watermelon_weight : ℝ := 3.5
noncomputable def scale_error : ℝ := 0.5

def weight_bounds (actual_weight measured_weight error_margin : ℝ) :=
  (measured_weight - error_margin ≤ actual_weight) ∧ (actual_weight ≤ measured_weight + error_margin)

theorem find_actual_weights (x y : ℝ) 
  (melon_measured : x = 4)
  (watermelon_measured : y = 3)
  (combined_measured : x + y = 8.5)
  (hx : weight_bounds melon_weight x scale_error)
  (hy : weight_bounds watermelon_weight y scale_error)
  (h_combined : weight_bounds (melon_weight + watermelon_weight) (x + y) (2 * scale_error)) :
  x = melon_weight ∧ y = watermelon_weight := 
sorry

end find_actual_weights_l597_597257


namespace coordinates_of_P_l597_597492

theorem coordinates_of_P (A B : ℝ × ℝ × ℝ) (m : ℝ) :
  A = (1, 0, 2) ∧ B = (1, -3, 1) ∧ (0, 0, m) = (0, 0, -3) :=
by 
  sorry

end coordinates_of_P_l597_597492


namespace distance_from_M_to_focus_left_l597_597442

-- Define the hyperbola as a function
def hyperbola (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1

-- Define the x-coordinate of point M
def x_M : ℝ := 4

-- Define the y-coordinate of point M
def y_M : ℝ := 4 * Real.sqrt 7 / 3

-- Define point M
def point_M : ℝ × ℝ := (4, y_M)

-- Define the left focus of the hyperbola
def focus_left : ℝ × ℝ := (-5, 0)

-- Distance formula
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

-- Theorem stating the distance from point M to the left focus
theorem distance_from_M_to_focus_left :
  hyperbola point_M.1 point_M.2 →
  distance point_M focus_left = 29 / 3 :=
by
  intro h
  -- Proof omitted
  sorry

end distance_from_M_to_focus_left_l597_597442


namespace degree_of_f_plus_cg_l597_597844

def f (x : ℝ) : ℝ := 2 - 15 * x + 4 * x^2 - 5 * x^3 + 6 * x^4
def g (x : ℝ) : ℝ := 5 - 3 * x - 7 * x^3 + 10 * x^4

theorem degree_of_f_plus_cg (c : ℝ) 
  (h_c : c = -3 / 5) : 
  ∃ h : ℝ → ℝ, (∀ x, f x + c * g x = h x) ∧ degree h = 3 := 
by
  sorry

end degree_of_f_plus_cg_l597_597844


namespace find_a_l597_597847

def E (a b c : ℤ) : ℤ := a * b * b + c

theorem find_a (a : ℤ) : E a 3 1 = E a 5 11 → a = -5 / 8 := 
by sorry

end find_a_l597_597847


namespace TwentyFifthMultipleOfFour_l597_597255

theorem TwentyFifthMultipleOfFour (n : ℕ) (h : ∀ k, 0 <= k ∧ k <= 24 → n = 16 + 4 * k) : n = 112 :=
by
  sorry

end TwentyFifthMultipleOfFour_l597_597255


namespace minimum_cost_to_reverse_chips_order_l597_597371

theorem minimum_cost_to_reverse_chips_order : 
  ∀ (n : ℕ) (chips : Fin n → ℕ), 
    (∀ i : ℕ, i < n → chips i = i) →
    (∀ i j : ℕ, i < j ∧ j = i + 1 → 1) →
    (∀ i j : ℕ, j = i + 5 → 0) →
    n = 100 → 
    reverse_cost chips = 61 := 
by 
  intros n chips hchip_order hswap_cost1 hswap_cost2 hn 
  sorry

end minimum_cost_to_reverse_chips_order_l597_597371


namespace length_of_bridge_is_correct_l597_597286

def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (crossing_time_s : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem length_of_bridge_is_correct : 
  length_of_bridge 170 45 30 = 205 :=
by
  -- we state the translation and prove here (proof omitted, just the structure is present)
  sorry

end length_of_bridge_is_correct_l597_597286


namespace eunsung_sungmin_menu_cases_l597_597667

theorem eunsung_sungmin_menu_cases :
  let kinds_of_chicken := 4
  let kinds_of_pizza := 3
  let same_chicken_different_pizza :=
    kinds_of_chicken * (kinds_of_pizza * (kinds_of_pizza - 1))
  let same_pizza_different_chicken :=
    kinds_of_pizza * (kinds_of_chicken * (kinds_of_chicken - 1))
  same_chicken_different_pizza + same_pizza_different_chicken = 60 :=
by
  sorry

end eunsung_sungmin_menu_cases_l597_597667


namespace positive_difference_x_coordinates_l597_597019

noncomputable def slope (x1 y1 x2 y2 : ℝ) := (y2 - y1) / (x2 - x1)

noncomputable def y_intercept (x y m : ℝ) := y - m * x

noncomputable def line_eq (m b x : ℝ) := m * x + b

noncomputable def x_at_y (y m b : ℝ) := (y - b) / m

theorem positive_difference_x_coordinates :
  let p_m := slope 0 3 2 0,
      p_b := y_intercept 0 3 p_m,
      q_m := slope 0 7 5 0,
      q_b := y_intercept 0 7 q_m,
      p_x := x_at_y 10 p_m p_b,
      q_x := x_at_y 10 q_m q_b in
  abs ((p_x : ℝ) - q_x) = 13 / 7 :=
by
  /- Proof goes here -/
  sorry

end positive_difference_x_coordinates_l597_597019


namespace sam_wins_probability_l597_597553

theorem sam_wins_probability (hitting_probability missing_probability : ℚ)
    (hit_prob : hitting_probability = 2/5)
    (miss_prob : missing_probability = 3/5) : 
    let p := hitting_probability / (1 - missing_probability ^ 2)
    p = 5 / 8 :=
by
    sorry

end sam_wins_probability_l597_597553


namespace sqrt_meaningful_range_l597_597951

theorem sqrt_meaningful_range {x : ℝ} (h : x - 1 ≥ 0) : x ≥ 1 :=
sorry

end sqrt_meaningful_range_l597_597951


namespace bridge_extension_length_l597_597224

theorem bridge_extension_length (width_of_river length_of_existing_bridge additional_length_needed : ℕ)
  (h1 : width_of_river = 487)
  (h2 : length_of_existing_bridge = 295)
  (h3 : additional_length_needed = width_of_river - length_of_existing_bridge) :
  additional_length_needed = 192 :=
by {
  -- The steps of the proof would go here, but we use sorry for now.
  sorry
}

end bridge_extension_length_l597_597224


namespace choose_roles_ways_l597_597408

theorem choose_roles_ways :
  let members := ["Alice", "Bob", "Carol", "Dan", "Eve"]
  let president := "Alice"
  let remaining_members := ["Bob", "Carol", "Dan", "Eve"]
  let roles := ["vice_president", "secretary", "treasurer"]
  let no_multiple_jobs := ∀ (x : String), x ∈ remaining_members → x ∉ roles ->
    (∃(role_assigned : String → String), function.injective role_assigned)
  in 4 * 3 * 2 = 24 := by
  sorry

end choose_roles_ways_l597_597408


namespace diana_weekly_earnings_l597_597367

-- Define the hours worked each day
def hours_monday := 10
def hours_tuesday := 15
def hours_wednesday := 10
def hours_thursday := 15
def hours_friday := 10

-- Define the hourly wage
def hourly_wage := 30

-- Define the total weekly earnings calculation and the proof for it
theorem diana_weekly_earnings : 
    let total_hours := hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday in
    let weekly_earnings := total_hours * hourly_wage in
    weekly_earnings = 1800 := 
by
  let total_hours := hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday
  let weekly_earnings := total_hours * hourly_wage
  have h1 : total_hours = 60 := by simp [hours_monday, hours_tuesday, hours_wednesday, hours_thursday, hours_friday]
  have h2 : weekly_earnings = 60 * 30 := by simp [h1, hourly_wage]
  have h3 : 60 * 30 = 1800 := by norm_num
  exact h3

end diana_weekly_earnings_l597_597367


namespace circleC₁_equation_circleC₂_equation_l597_597484

-- Define the points M, N, Q
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-1, 1⟩
def N : Point := ⟨0, 2⟩
def Q : Point := ⟨2, 0⟩

-- Define a circle by its center and radius squared
structure Circle where
  center : Point
  radius_squared : ℝ

-- Question 1: Prove the standard equation of circle C₁ that passes through M, N, and Q
def C₁ : Circle := ⟨⟨1/2, 1/2⟩, 5/2⟩

theorem circleC₁_equation :
  ∀ (p : Point), p = M ∨ p = N ∨ p = Q → (p.x - C₁.center.x)^2 + (p.y - C₁.center.y)^2 = C₁.radius_squared :=
sorry

-- Question 2: Prove the standard equation of circle C₂ symmetrical to C₁ about the line MN
def C₂ : Circle := ⟨⟨-3/2, 5/2⟩, 5/2⟩

theorem circleC₂_equation :
  C₂.center.x = -3/2 ∧ C₂.center.y = 5/2 ∧
  C₂.radius_squared = 5/2 ∧
  ∃ (oc : Circle), oc = C₁ ∧
  (C₂.center.x - oc.center.x)^2 + (C₂.center.y - oc.center.y)^2 = (C₂.radius_squared) :=
sorry

end circleC₁_equation_circleC₂_equation_l597_597484


namespace tony_lottery_winning_l597_597713

theorem tony_lottery_winning
  (tickets : ℕ) (winning_numbers : ℕ) (worth_per_number : ℕ) (identical_numbers : Prop)
  (h_tickets : tickets = 3) (h_winning_numbers : winning_numbers = 5) (h_worth_per_number : worth_per_number = 20)
  (h_identical_numbers : identical_numbers) :
  (tickets * (winning_numbers * worth_per_number) = 300) :=
by
  sorry

end tony_lottery_winning_l597_597713


namespace minimum_value_is_8_l597_597165

noncomputable def minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_is_8 :
  ∃ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), minimum_value x y hx hy = 8 :=
by
  sorry

end minimum_value_is_8_l597_597165


namespace no_cycle_in_present_graph_l597_597768

theorem no_cycle_in_present_graph (n : ℕ) (hn : 2 ≤ n) :
  ∀ (G : fin n → fin n → Prop),
    (∀ a b : fin n, G a b ↔ a ≠ b ∧ (a : ℕ) * (b : ℕ - 1) % n = 0) →
    ¬ ∃ (cycle : list (fin n)), cycle.chain' G cycle ∧ cycle.head = cycle.last :=
begin
  sorry
end

end no_cycle_in_present_graph_l597_597768


namespace min_value_of_frac_sum_l597_597030

theorem min_value_of_frac_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 2) :
  (1 / a + 2 / b) = 9 / 2 :=
sorry

end min_value_of_frac_sum_l597_597030


namespace negation_of_universal_proposition_l597_597754

theorem negation_of_universal_proposition :
  ¬ (∀ (m : ℝ), ∃ (x : ℝ), x^2 + x + m = 0) ↔ ∃ (m : ℝ), ¬ ∃ (x : ℝ), x^2 + x + m = 0 :=
by sorry

end negation_of_universal_proposition_l597_597754


namespace repeating_block_length_of_7_div_13_is_6_l597_597611

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l597_597611


namespace find_sin_A_l597_597972

noncomputable def triangle_condition {a b c S : ℝ} (h_S : S = a^2 - b^2 - c^2 + 2 * b * c) : Prop :=
  ∃ A : ℝ, (A > 0) ∧ (A < π) ∧ (sin A = 8 / 17)

theorem find_sin_A (a b c S : ℝ) (h_S : S = a^2 - b^2 - c^2 + 2 * b * c) :
  triangle_condition h_S :=
sorry

end find_sin_A_l597_597972


namespace Sam_wins_probability_l597_597548

-- Define the basic probabilities
def prob_hit : ℚ := 2 / 5
def prob_miss : ℚ := 3 / 5

-- Define the desired probability that Sam wins
noncomputable def p : ℚ := 5 / 8

-- The mathematical problem statement in Lean
theorem Sam_wins_probability :
  p = prob_hit + (prob_miss * prob_miss * p) := 
sorry

end Sam_wins_probability_l597_597548


namespace divisors_of_10_factorial_larger_than_9_factorial_l597_597085

theorem divisors_of_10_factorial_larger_than_9_factorial :
  ∃ n, n = 9 ∧ (∀ d, d ∣ (Nat.factorial 10) → d > (Nat.factorial 9) → d > (Nat.factorial 1) → n = 9) :=
sorry

end divisors_of_10_factorial_larger_than_9_factorial_l597_597085


namespace repeating_block_length_of_7_div_13_is_6_l597_597610

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l597_597610


namespace right_triangle_integer_segments_l597_597198

theorem right_triangle_integer_segments (DE EF : ℤ) (h1 : DE = 24) (h2 : EF = 25) :
  ∃ (n : ℕ), 
  n = 14 ∧
  ∀ (X : ℝ), 
  X ∈ set.Icc 0 1 → ∃ (EX : ℤ), by {
    let D := (0 : ℝ),
    let E := (0, DE : ℝ),
    let F := (EF, 0 : ℝ),
    let DF := real.sqrt ((DE * DE + EF * EF : ℝ)),
    
    let P := (EF * (1 - X) / DF, DE * X / DF : ℝ), 
    let EP := real.sqrt ((E.1 - P.1)^2 + (E.2 - P.2)^2),
    
    EX = ⌊ EP ⌋ ∧ 18 ⩽ EX ∧ EX ⩽ 25
  } := sorry

end right_triangle_integer_segments_l597_597198


namespace largest_common_divisor_408_340_is_68_l597_597744

theorem largest_common_divisor_408_340_is_68 :
  let factors_408 := [1, 2, 3, 4, 6, 8, 12, 17, 24, 34, 51, 68, 102, 136, 204, 408]
  let factors_340 := [1, 2, 4, 5, 10, 17, 20, 34, 68, 85, 170, 340]
  ∀ d ∈ factors_408, d ∈ factors_340 → ∀ (e ∈ factors_408), (e ∈ factors_340) → d ≤ e :=
  68 := by sorry

end largest_common_divisor_408_340_is_68_l597_597744


namespace repeating_block_length_of_7_div_13_is_6_l597_597609

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l597_597609


namespace sum_of_solutions_l597_597388

theorem sum_of_solutions (x : ℝ) (h₁ : (x - 8)^2 = 49) :
  x = 15 ∨ x = 1 → ∃ s, s = 16 :=
by
  intro h₂
  cases h₂ with h₃ h₄
  { use 16
    sorry }
  { use 16
    sorry }

end sum_of_solutions_l597_597388


namespace equal_area_division_l597_597106

open_locale classical

variables {A B C D X Y V Z S T P : Type*}
variables [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ D] 
variables [affine_space ℝ X] [affine_space ℝ Y] [affine_space ℝ V] [affine_space ℝ Z]
variables [affine_space ℝ S] [affine_space ℝ T] [affine_space ℝ P]
variables [convex_quadrilateral ABCD: Type*]

/-- A theorem that states: In a convex quadrilateral, the segments connecting the intersection
points of these parallel lines with the midpoints of the sides divide the quadrilateral into equal-area parts. --/
theorem equal_area_division (h_mid_AC : S = midpoint ℝ A C)
                           (h_mid_BD : T = midpoint ℝ B D)
                           (h_mid_AB : X = midpoint ℝ A B)
                           (h_mid_BC : Y = midpoint ℝ B C)
                           (h_mid_CD : V = midpoint ℝ C D)
                           (h_mid_DA : Z = midpoint ℝ D A)
                           (h_parallel_ST_AC : line_parallel S T (A - C))
                           (h_parallel_TD_BD : line_parallel T D (B - C))
                           (h_quad_convex : convex_quadrilateral ABCD) :
                             area (triangle A X P) = (1/4) * (area (quadrilateral A B C D))
                             ∧ area (triangle B Y P) = (1/4) * (area (quadrilateral A B C D))
                             ∧ area (triangle C V P) = (1/4) * (area (quadrilateral A B C D))
                             ∧ area (triangle D Z P) = (1/4) * (area (quadrilateral A B C D)) :=
begin
  sorry -- proof goes here
end

end equal_area_division_l597_597106


namespace count_rectangles_in_5x5_grid_l597_597735

theorem count_rectangles_in_5x5_grid :
  let n := 5 in
  ∑ x in finset.range n.succ, (n - x) * (n - x) * (n - x) = 225 :=
by {
  simp only [finset.range, finset.sum],
  sorry
}

end count_rectangles_in_5x5_grid_l597_597735


namespace A_intersect_B_eq_l597_597936

def A (x : ℝ) : Prop := x > 0
def B (x : ℝ) : Prop := x ≤ 1
def A_cap_B (x : ℝ) : Prop := x ∈ {y | A y} ∧ x ∈ {y | B y}

theorem A_intersect_B_eq (x : ℝ) : (A_cap_B x) ↔ (x ∈ Set.Ioc 0 1) :=
by
  sorry

end A_intersect_B_eq_l597_597936


namespace S7_eq_49_l597_597898

-- Define the sequence and sum conditions
variables (a b : ℝ)
def S (n : ℕ) : ℝ := a * n^2 + b * n
def a_n (n : ℕ) : ℝ := S n - (if n = 0 then 0 else S (n - 1))

-- Given conditions
axiom a2_eq_3 : a_n a b 2 = 3
axiom a6_eq_11 : a_n a b 6 = 11

-- Proof of main statement
theorem S7_eq_49 : S a b 7 = 49 :=
sorry

end S7_eq_49_l597_597898


namespace area_of_T_l597_597522

open Complex Real

noncomputable def omega := -1 / 2 + (1 / 2) * Complex.I * Real.sqrt 3
noncomputable def omega2 := -1 / 2 - (1 / 2) * Complex.I * Real.sqrt 3

def inT (z : ℂ) (a b c : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 2 ∧
  0 ≤ b ∧ b ≤ 1 ∧
  0 ≤ c ∧ c ≤ 1 ∧
  z = a + b * omega + c * omega2

theorem area_of_T : ∃ A : ℝ, A = 2 * Real.sqrt 3 :=
sorry

end area_of_T_l597_597522


namespace angle_sum_is_111_degrees_l597_597823

noncomputable def cis (θ : ℝ) := complex.exp (complex.I * θ * real.pi / 180)

theorem angle_sum_is_111_degrees :
  ∃ r > 0, r * cis 111 = cis 65 + cis 73 + cis 81 + cis 89 + cis 97 + cis 105 + cis 113 + cis 121 + cis 129 + cis 137 + cis 145 + cis 153 + cis 157 :=
sorry

end angle_sum_is_111_degrees_l597_597823


namespace exists_same_colored_points_at_distance_l597_597244

open Classical

noncomputable def point_colored_red_or_blue (p : ℝ × ℝ) : Prop :=
  p = ⟨x, y⟩ ∨ p = ⟨u, v⟩ -- Representing colors by point labels (x, y) for red and (u, v) for blue

theorem exists_same_colored_points_at_distance (x : ℝ) (hx : x > 0) :
  ∃ p q : ℝ × ℝ, (point_colored_red_or_blue p ∧ point_colored_red_or_blue q) ∧ dist p q = x :=
by
  sorry

end exists_same_colored_points_at_distance_l597_597244


namespace repeat_block_of_7_div_13_l597_597630

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l597_597630


namespace question_one_question_two_l597_597031

variable (x m : ℝ)

-- Conditions
def p : Prop := -2 ≤ x ∧ x ≤ 6
def q : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Question 1
theorem question_one (h₁ : m > 0) (h₂ : p → q) : m ≥ 4 :=
  sorry

-- Question 2
theorem question_two (h₁ : m = 5) (h₂ : p ∨ q) (h₃ : ¬(p ∧ q)) : 
  x ∈ Set.Ico (-3 : ℝ) (-2) ∪ Set.Ioo (6 : ℝ) (7) :=
  sorry

end question_one_question_two_l597_597031


namespace sum_of_zeros_l597_597849

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then
  if x < 1 then log (x + 1) / log 2
  else |x - 3| - 1
else
  -f (-x)

def F (x a : ℝ) : ℝ := f x - a

theorem sum_of_zeros (a : ℝ) (h : -1 < a ∧ a < 0) :
  ∃ x₁ x₂ x₃ x₄ x₅ : ℝ,
    F x₁ a = 0 ∧ F x₂ a = 0 ∧ F x₃ a = 0 ∧ F x₄ a = 0 ∧ F x₅ a = 0 ∧
    x₁ + x₂ + x₃ + x₄ + x₅ = 1 - 2^(-a) := sorry

end sum_of_zeros_l597_597849


namespace find_smaller_number_l597_597641

noncomputable def smaller_number (x y : ℝ) := y

theorem find_smaller_number 
  (x y : ℝ) 
  (h1 : x - y = 9) 
  (h2 : x + y = 46) :
  smaller_number x y = 18.5 :=
sorry

end find_smaller_number_l597_597641


namespace james_total_money_l597_597136

theorem james_total_money :
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  total_money = 135 := by
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  exact 135

end james_total_money_l597_597136


namespace smallest_positive_period_f_pi_max_min_f_in_interval_l597_597923

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x - sin x ^ 2

theorem smallest_positive_period_f_pi :
  ∀ T > 0, (∀ x, f (x + T) = f x) → T = π :=
sorry

theorem max_min_f_in_interval :
  ∃ (x_max x_min : ℝ), x_max ∈ Icc 0 (π / 2) ∧ x_min ∈ Icc 0 (π / 2) ∧
  (∀ x ∈ Icc 0 (π / 2), f x ≤ f x_max) ∧ f x_max = 1 / 2 ∧
  (∀ x ∈ Icc 0 (π / 2), f x_min ≤ f x) ∧ f x_min = -1 :=
sorry

end smallest_positive_period_f_pi_max_min_f_in_interval_l597_597923


namespace divisors_of_10_factorial_larger_than_9_factorial_l597_597086

theorem divisors_of_10_factorial_larger_than_9_factorial :
  ∃ n, n = 9 ∧ (∀ d, d ∣ (Nat.factorial 10) → d > (Nat.factorial 9) → d > (Nat.factorial 1) → n = 9) :=
sorry

end divisors_of_10_factorial_larger_than_9_factorial_l597_597086


namespace repeating_block_length_7_div_13_l597_597618

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597618


namespace player1_points_after_13_rotations_l597_597693

theorem player1_points_after_13_rotations :
  let sector_points := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
  let player_points (player : Nat) (rotations : Nat) :=
      rotations • (λ i, sector_points[(i + player) % 16])
  player_points 5 13 = 72 ∧ player_points 9 13 = 84 → player_points 1 13 = 20 :=
by
  sorry

end player1_points_after_13_rotations_l597_597693


namespace triangle_isosceles_DE_FG_sum_l597_597716

theorem triangle_isosceles_DE_FG_sum (AB AC BC : ℝ) (E G : ℝ) (D F : ℝ)
  (h_iso : AB = 2 ∧ AC = 2 ∧ BC = 1) 
  (h_parallel_DE_FG : DE ∥ BC ∧ FG ∥ BC)
  (h_perimeters : ∀ (x s u y : ℝ), 
    x = AD ∧ s = DF ∧ u = DG ∧ y = AG →
    perimeter ADE = perimeter DFGE ∧ perimeter DFGE = perimeter FBCG) :
  DE + FG = 2 / 3 := 
  sorry

end triangle_isosceles_DE_FG_sum_l597_597716


namespace circle_sum_zero_l597_597945

variable {α : Type*} [AddGroup α] 

theorem circle_sum_zero  (a1 a2 a3 a4 a5 a6 : α) 
  (h1 : a1 + a2 = 0) 
  (h2 : 3 * a1 + a2 + a3 = 0) 
  (h3 : 2 * a2 + 2 * a3 + a4 = 0) 
  (h4 : a3 + 2 * a4 + 2 * a5 = 0) 
  (h5 : a4 + a5 + 3 * a6 = 0) 
  (h6 : a5 + a6 = 0) 
  : a1 = 0 ∧ a2 = 0 ∧ a3 = 0 ∧ a4 = 0 ∧ a5 = 0 ∧ a6 = 0 := 
sorry

end circle_sum_zero_l597_597945


namespace solve_system_of_equations_l597_597567

variable {x : Fin 15 → ℤ}

theorem solve_system_of_equations (h : ∀ i : Fin 15, 1 - x i * x ((i + 1) % 15) = 0) :
  (∀ i : Fin 15, x i = 1) ∨ (∀ i : Fin 15, x i = -1) :=
by
  -- Here we put the proof, but it's omitted for now.
  sorry

end solve_system_of_equations_l597_597567


namespace satisfied_customers_percentage_is_correct_l597_597295

/- Define the problem conditions -/
def total_customers : ℕ := 300
def less_than_100_pct : ℝ := 0.60
def less_than_100_satisfied_pct : ℝ := 0.55
def at_least_100_not_satisfied_pct : ℝ := 0.07

/- Define the number of customers -/
def less_than_100_customers := total_customers * less_than_100_pct
def less_than_100_satisfied_customers := less_than_100_customers * less_than_100_satisfied_pct
def at_least_100_customers := total_customers * (1 - less_than_100_pct)
def at_least_100_not_satisfied_customers := total_customers * at_least_100_not_satisfied_pct
def at_least_100_satisfied_customers := at_least_100_customers - at_least_100_not_satisfied_customers

/- Define the total number of satisfied customers -/
def total_satisfied_customers := less_than_100_satisfied_customers + at_least_100_satisfied_customers

/- Define the percentage of satisfied customers -/
def satisfied_customers_pct := total_satisfied_customers / total_customers * 100

/- The theorem to prove -/
theorem satisfied_customers_percentage_is_correct : satisfied_customers_pct = 66 := 
  by
  sorry

end satisfied_customers_percentage_is_correct_l597_597295


namespace frog_escape_probability_l597_597475

def P : ℕ → ℚ
def P_recurrence (N : ℕ) (h : 0 < N ∧ N < 10) : 
  P N = (N / 10) * P (N - 1) + (1 - (N / 10)) * P (N + 1) := 
sorry

def P_boundary_0 : 
  P 0 = 0 := 
sorry 

def P_boundary_10 : 
  P 10 = 1 := 
sorry 

theorem frog_escape_probability : 
  P 1 = 63 / 146 := 
sorry

end frog_escape_probability_l597_597475


namespace hyperbola_asymptotes_l597_597071

theorem hyperbola_asymptotes (m : ℝ) :
  (m = -1 / 3) →
  (mx^2 + y^2 = 1) →
  (x^2 + y^2 / 5 = 1) →
  ∃ a b : ℝ, (a = 1 / sqrt 3) ∧ (b = 1) ∧ (∀ x y : ℝ, y = ± (sqrt 3 / 3) * x) :=
by
  intros hm he1 he2
  sorry

end hyperbola_asymptotes_l597_597071


namespace find_last_two_digits_l597_597719

variables {z a r m l : ℕ}
variables (ZARAZA ALMAZ : ℕ)
variables (digits : char → ℕ)

-- Each character represents a unique digit
axiom zaraza_unique_digits : function.injective digits
axiom almakza_unique_digits : function.injective digits

-- The numbers
def ZARAZA := 100000 * digits 'z' + 10000 * digits 'a' + 1000 * digits 'r' + 100 * digits 'a' + 10 * digits 'z' + digits 'a'
def ALMAZ := 10000 * digits 'a' + 1000 * digits 'l' + 100 * digits 'm' + 10 * digits 'a' + digits 'z'

-- Divisibility constraints
axiom zaraza_div_by_4 : ZARAZA % 4 = 0
axiom almaz_div_by_28 : ALMAZ % 28 = 0

-- Proof Goal
theorem find_last_two_digits :
  (ZARAZA + ALMAZ) % 100 = 32 := 
sorry

end find_last_two_digits_l597_597719


namespace equal_saturdays_and_sundays_l597_597788

theorem equal_saturdays_and_sundays (start_day : ℕ) (h : start_day < 7) :
  ∃! d, (d < 7 ∧ ((d + 2) % 7 = 0 → (d = 5))) :=
by
  sorry

end equal_saturdays_and_sundays_l597_597788


namespace find_other_number_l597_597648

open Nat

noncomputable def the_other_number (a gcd lcm : ℕ) : ℕ :=
  lcm * gcd / a

theorem find_other_number
  (a b gcd lcm : ℕ)
  (ha : a = 240)
  (hg : gcd = 12)
  (hl : lcm = 2520)
  (h_eq : gcd * lcm = a * b) : b = 126 :=
by
  rw [ha, hg, hl] at h_eq
  calc 
    b = (gcd * lcm) / a := by sorry -- this part can be expanded with proper proof
    ... = (12 * 2520) / 240 := by rw [ha, hg, hl]
    ... = 126 := by sorry -- calculation details omitted

end find_other_number_l597_597648


namespace repeating_block_length_7_div_13_l597_597581

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597581


namespace find_f_f_neg_e_l597_597231

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then Real.exp x else Real.log (|x|)

theorem find_f_f_neg_e : f (f (-Real.exp 1)) = Real.exp 1 := by
  sorry

end find_f_f_neg_e_l597_597231


namespace sum_of_solutions_eq_16_l597_597386

theorem sum_of_solutions_eq_16 :
  (∑ x in ({x : ℝ | (x - 8) ^ 2 = 49}).toFinset, x) = 16 :=
by
  sorry

end sum_of_solutions_eq_16_l597_597386


namespace express_y_in_terms_of_x_and_p_l597_597208

theorem express_y_in_terms_of_x_and_p (x p : ℚ) (h : x = (1 + p / 100) * (1 / y)) : 
  y = (100 + p) / (100 * x) := 
sorry

end express_y_in_terms_of_x_and_p_l597_597208


namespace N_R_perpendicular_QM_l597_597572

variables {P Q R K M N : Type} [EuclideanGeometry.geom]
variables (QK_angle_bisector : is_angle_bisector P Q R Q K)
variables (QK_circumcircle : is_circumcircle P Q R (QK_point M))
variables (PKM_circumcircle : is_circumcircle_ext P K M P Q N)

theorem N_R_perpendicular_QM 
  (h1 : QK_angle_bisector)
  (h2 : QK_circumcircle)
  (h3 : PKM_circumcircle) :
  is_perpendicular (line N R) (line Q M) :=
sorry

end N_R_perpendicular_QM_l597_597572


namespace rectangles_in_5x5_grid_l597_597729

theorem rectangles_in_5x5_grid : 
  let grid_rows := 5
  let grid_cols := 5
  -- A function that calculates the number of rectangles in an n x m grid
  num_rectangles_in_grid grid_rows grid_cols = 225 :=
  sorry

end rectangles_in_5x5_grid_l597_597729


namespace hacker_cannot_change_grades_l597_597324

theorem hacker_cannot_change_grades :
  ¬ ∃ n1 n2 n3 n4 : ℤ,
    2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
    -n1 + 2 * n2 + n3 - 2 * n4 = -27 := by
  sorry

end hacker_cannot_change_grades_l597_597324


namespace amount_of_money_l597_597486

theorem amount_of_money (x y : ℝ) 
  (h1 : x + 1/2 * y = 50) 
  (h2 : 2/3 * x + y = 50) : 
  (x + 1/2 * y = 50) ∧ (2/3 * x + y = 50) :=
by
  exact ⟨h1, h2⟩ 

end amount_of_money_l597_597486


namespace complex_modulus_l597_597953

theorem complex_modulus (x y : ℝ) (h : i * (x + y * i) = 3 + 4 * i) :
  complex.abs (x + y * i) = 5 :=
by
  sorry

end complex_modulus_l597_597953


namespace eval_powers_of_i_l597_597857

theorem eval_powers_of_i :
  let i := Complex.I in
  i^1234 + i^1235 + i^1236 + i^1237 = 0 :=
by
  let i := Complex.I
  sorry

end eval_powers_of_i_l597_597857


namespace rotate_isosceles_trapezoid_l597_597200

def isosceles_trapezoid (a b h : ℝ) : Prop :=
  b > a ∧ h > 0

def rotating_plane_figure (figure : ℝ → ℝ → ℝ → Prop) (a b h : ℝ) : list Prop :=
  figure a b h → [cone (right_triangle b h), cone (right_triangle b h), cylinder (rectangle a h)]

theorem rotate_isosceles_trapezoid (a b h : ℝ) (h_ : isosceles_trapezoid a b h) :
  rotating_plane_figure isosceles_trapezoid a b h = [cone (right_triangle b h), cone (right_triangle b h), cylinder (rectangle a h)] :=
by
  sorry

end rotate_isosceles_trapezoid_l597_597200


namespace complement_union_l597_597180

def universal_set : Set ℝ := { x : ℝ | true }
def M : Set ℝ := { x : ℝ | x ≤ 0 }
def N : Set ℝ := { x : ℝ | x > 2 }

theorem complement_union (x : ℝ) :
  x ∈ compl (M ∪ N) ↔ (0 < x ∧ x ≤ 2) := by
  sorry

end complement_union_l597_597180


namespace find_x_l597_597459

def G (a b c d e : ℕ) : ℕ := a^b + c * d - e

theorem find_x :
  ∃ x : ℝ, G 3 x 5 12 10 = 500 ∧ abs (x - 6) < 1 :=
by
  -- skipping the actual proof for purposes of this example
  sorry

end find_x_l597_597459


namespace part1_solution_l597_597288

noncomputable def part1 (m : ℝ) : Prop :=
  let z := (m^2 - 8m + 15) + (m^2 - 4m + 3) * complex.i in
  z.re = 0 ∧ z.im ≠ 0

theorem part1_solution : part1 5 :=
by
  sorry

end part1_solution_l597_597288


namespace product_of_fractions_l597_597829

theorem product_of_fractions :
  ∏ k in finset.range 501, (4 + 4 * k : ℕ) / (8 + 4 * k) = 1 / 502 := 
by sorry

end product_of_fractions_l597_597829


namespace part1_part2_l597_597926

noncomputable def f (x : Real) : Real := sqrt 3 * sin (2 * x) + cos (2 * x)

theorem part1 (x : Real) (h : 0 ≤ x ∧ x ≤ π / 2) :
  x ∈ Icc 0 (π / 6) ↔ monotone_on f (Icc 0 (π / 6)) :=
sorry

noncomputable def g (x : Real) : Real := (1 / 2) * (f x)^2 - f (x + π / 4) - 1

theorem part2 (x : Real) (h : -π / 6 ≤ x ∧ x ≤ π / 3) :
  ∃ y ∈ Set.Icc (-3 : Real) (3 / 2), y = g x :=
sorry

end part1_part2_l597_597926


namespace normal_line_eq_l597_597871

variable {x : ℝ}

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem normal_line_eq (x_0 : ℝ) (h : x_0 = 1) :
  ∃ y_0 : ℝ, y_0 = f x_0 ∧ 
  ∀ x y : ℝ, y = -(x - 1) + y_0 ↔ f 1 = 0 ∧ y = -x + 1 :=
by
  sorry

end normal_line_eq_l597_597871


namespace complex_expression_approx_l597_597389

noncomputable def complex_expression : ℝ := 
  4 * (3.6 * ((7 / 3) * (3 / 8)^2 * 2.50) / (4.5 * 10^(-2) * ((9 / 5)^(3 / 2) * (2 / 3)^(-1 / 2)) * 0.5)) 
  + 2^(1 / 3) * (3 - sqrt(4 + (2 / 7)^3))

theorem complex_expression_approx : abs (complex_expression - 153.48127) < 0.0001 :=
sorry

end complex_expression_approx_l597_597389


namespace frog_escape_probability_l597_597477

noncomputable def P : ℕ → ℚ
| 0     := 0
| 10    := 1
| (n+1) := if h : n < 9 then (n + 1) / 10 * P n + (1 - (n + 1) / 10) * P (n + 2) 
          else 0 -- this case is only to cover all possible inputs

theorem frog_escape_probability : P 1 = 63 / 146 :=
by 
  sorry

end frog_escape_probability_l597_597477


namespace product_permutation_formula_l597_597397

theorem product_permutation_formula (n : ℕ) (h : n < 55) : 
  (finset.range 15).prod (λ k, (55 - n) + k) = (finset.range 15).prod (λ k, (69 - n) - k) :=
by
  sorry

end product_permutation_formula_l597_597397


namespace total_money_l597_597126

def JamesPocketBills : Nat := 3
def BillValue : Nat := 20
def WalletMoney : Nat := 75

theorem total_money (JamesPocketBills BillValue WalletMoney : Nat) : 
  (JamesPocketBills * BillValue + WalletMoney) = 135 :=
by
  sorry

end total_money_l597_597126


namespace algebra_expression_never_zero_l597_597391

theorem algebra_expression_never_zero (x : ℝ) : (1 : ℝ) / (x - 1) ≠ 0 :=
sorry

end algebra_expression_never_zero_l597_597391


namespace solve_for_m_l597_597227

def f (x : ℝ) (m : ℝ) := x^3 - m * x + 3

def f_prime (x : ℝ) (m : ℝ) := 3 * x^2 - m

theorem solve_for_m (m : ℝ) : f_prime 1 m = 0 → m = 3 :=
by
  sorry

end solve_for_m_l597_597227


namespace jori_remaining_water_l597_597998

-- Having the necessary libraries for arithmetic and fractions.

-- Definitions directly from the conditions in a).
def initial_water_quantity : ℚ := 4
def used_water_quantity : ℚ := 9 / 4 -- Converted 2 1/4 to an improper fraction

-- The statement proving the remaining quantity of water is 1 3/4 gallons.
theorem jori_remaining_water : initial_water_quantity - used_water_quantity = 7 / 4 := by
  sorry

end jori_remaining_water_l597_597998


namespace repeating_block_length_of_7_div_13_is_6_l597_597608

theorem repeating_block_length_of_7_div_13_is_6:
  ∀ (n d : ℕ), n = 7 → d = 13 → (∀ r : ℕ, r ∈ [7, 9, 12, 3, 4, 11, 1, 10, 5, 6, 8, 2]) → 
  (∀ k : ℕ, (k < 6) → 
    let ⟨q, r⟩ := digits_of_division (7 : ℤ) (13 : ℤ) in 
    repeat_block_length (q, r) = 6) := 
by 
  sorry

end repeating_block_length_of_7_div_13_is_6_l597_597608


namespace compute_usage_difference_l597_597279

theorem compute_usage_difference
  (usage_last_week : ℕ)
  (usage_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : usage_last_week = 91)
  (h2 : usage_per_day = 8)
  (h3 : days_in_week = 7) :
  (usage_last_week - usage_per_day * days_in_week) = 35 :=
  sorry

end compute_usage_difference_l597_597279


namespace product_of_three_numbers_l597_597252

-- Define the problem conditions as variables and assumptions
variables (a b c : ℚ)
axiom h1 : a + b + c = 30
axiom h2 : a = 3 * (b + c)
axiom h3 : b = 6 * c

-- State the theorem to be proven
theorem product_of_three_numbers : a * b * c = 10125 / 14 :=
by
  sorry

end product_of_three_numbers_l597_597252


namespace total_books_l597_597496

-- Define the number of books per shelf
def books_per_shelf : ℕ := 45

-- Define the number of shelves
def number_of_shelves : ℕ := 7

-- Problem Statement: Formalize the proof of the total number of books Jason has
theorem total_books (b_per_shelf : ℕ) (num_shelves : ℕ) : b_per_shelf = 45 ∧ num_shelves = 7 → b_per_shelf * num_shelves = 315 :=
by
  intro h
  cases h with h1 h2
  rw [h1, h2]
  exact rfl

end total_books_l597_597496


namespace sum_of_30_consecutive_even_integers_l597_597251

theorem sum_of_30_consecutive_even_integers (x : ℤ) (sum_eq : (∑ i in finset.range 30, (x + 2 * i)) = 12000) :
  x + 58 = 429 :=
sorry

end sum_of_30_consecutive_even_integers_l597_597251


namespace divisors_larger_than_9_factorial_l597_597088

theorem divisors_larger_than_9_factorial (n : ℕ) :
  (∃ k : ℕ, k = 9 ∧ (number_of_divisors_of_10_factorial_greater_than_9_factorial = k)) :=
begin
  sorry
end

def number_of_divisors_of_10_factorial_greater_than_9_factorial : ℕ :=
  (10.fact.divisors.filter (λ d, d > 9.fact)).length

end divisors_larger_than_9_factorial_l597_597088


namespace problem_l597_597390

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable [Differentiable ℝ f]

theorem problem 
  (h1 : ∀ x : ℝ, (1 - x) / (f' x) ≥ 0)
  (h2 : ∀ x : ℝ, HasDerivAt f (f' x) x) :
  f 0 + f 2 < 2 * f 1 := by
    sorry

end problem_l597_597390


namespace least_alpha_prime_l597_597662

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_distinct_prime (α β : ℕ) : Prop :=
  α ≠ β ∧ is_prime α ∧ is_prime β

theorem least_alpha_prime (α : ℕ) :
  is_distinct_prime α (180 - 2 * α) → α ≥ 41 :=
sorry

end least_alpha_prime_l597_597662


namespace player_1_points_after_13_rotations_l597_597674

-- Add necessary definitions and state the problem in Lean
def sectors : Fin 16 → ℕ
| ⟨0, _⟩ := 0
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 2
| ⟨3, _⟩ := 3
| ⟨4, _⟩ := 4
| ⟨5, _⟩ := 5
| ⟨6, _⟩ := 6
| ⟨7, _⟩ := 7
| ⟨8, _⟩ := 8
| ⟨9, _⟩ := 7
| ⟨10, _⟩ := 6
| ⟨11, _⟩ := 5
| ⟨12, _⟩ := 4
| ⟨13, _⟩ := 3
| ⟨14, _⟩ := 2
| ⟨15, _⟩ := 1

def points_earned (player_offset : Fin 16) (rotations : ℕ) : ℕ :=
List.sum (List.map sectors
  (List.map (λ n => (Fin.add (Fin.ofNat n) player_offset)) (List.range rotations)))

theorem player_1_points_after_13_rotations 
  (p5_points : points_earned ⟨5, by decide⟩ 13 = 72)
  (p9_points : points_earned ⟨9, by decide⟩ 13 = 84) :
  points_earned ⟨1, by decide⟩ 13 = 20 := 
sorry

end player_1_points_after_13_rotations_l597_597674


namespace find_intersections_l597_597044

noncomputable def intersection_points (α : ℝ) (t θ : ℝ) : set (ℝ × ℝ) :=
  {p | ∃ t θ : ℝ, p = ((1 + t * Real.cos α), t * Real.sin α) ∧
                   p = (Real.cos θ, Real.sin θ)}

theorem find_intersections :
  let α := Real.pi / 3 in
  intersection_points α t θ = {(1, 0), (1 / 2, - Real.sqrt 3 / 2)} :=
by
  sorry

end find_intersections_l597_597044


namespace pet_shop_legs_l597_597817

theorem pet_shop_legs :
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  birds * bird_legs + dogs * dog_legs + snakes * snake_legs + spiders * spider_legs = 34 := 
by
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  sorry

end pet_shop_legs_l597_597817


namespace repeat_block_of_7_div_13_l597_597638

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l597_597638


namespace altitude_extension_becomes_median_l597_597195

theorem altitude_extension_becomes_median 
  (A B C A' B' : Type)
  [add_comm_group A] [module ℝ A]
  [add_comm_group B] [module ℝ B]
  [add_comm_group C] [module ℝ C]
  [affine_space C A]
  [add_comm_group A'] [module ℝ A']
  [add_comm_group B'] [module ℝ B']
  [add_comm_group C'] [module ℝ C']
  [affine_space C' A']
  (ABC : triangle A B C)
  (hABC : ABC.angles.is_right (ABC.angle B C))
  (reflection : external_angle_bisector_reflection A B C A' B')
  (CD : altitude_from_right_angle C ABC)
  (hCD_extends : extends_and_meets C D A' B' E)
  : is_median (C'A B' E) :=
sorry

end altitude_extension_becomes_median_l597_597195


namespace repeating_block_length_7_div_13_l597_597602

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l597_597602


namespace num_divisible_1_to_60_l597_597946

theorem num_divisible_1_to_60 : 
  let divisible_by (n : ℕ) (d : ℕ) := d > 0 ∧ n % d = 0
  let count_divisibles (d : ℕ) := finset.card (finset.filter (λ n, divisible_by n d) (finset.range 61))
  let n_3 := count_divisibles 3
  let n_5 := count_divisibles 5
  let n_7 := count_divisibles 7
  let n_15 := count_divisibles 15
  let n_21 := count_divisibles 21
  let n_35 := count_divisibles 35
  let n_105 := count_divisibles 105
  1 <= 60 →
  (n_3 + n_5 + n_7 - (n_15 + n_21 + n_35) = 33) :=
by
  sorry

end num_divisible_1_to_60_l597_597946


namespace find_m_value_l597_597922

noncomputable def fx (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem find_m_value (m : ℝ) : (∀ x > 0, fx m x > fx m 0) → m = 2 := by
  sorry

end find_m_value_l597_597922


namespace domain_A_domain_B_intersection_A_B_union_A_CUB_l597_597437

variable {x : ℝ}

def A := {x | x > 3 / 2}

def B := {x | 1 < x ∧ x ≤ 3}

def C_UB := {x | x ≤ 1 ∨ x > 3}

theorem domain_A : A = {x | x > 3 / 2} := 
by simp; sorry

theorem domain_B : B = {x | 1 < x ∧ x ≤ 3} := 
by simp; sorry

theorem intersection_A_B : A ∩ B = {x | 3 / 2 < x ∧ x ≤ 3} := 
by simp; sorry

theorem union_A_CUB : A ∪ C_UB = {x | x ≤ 1 ∨ x > 3 / 2} := 
by simp; sorry

end domain_A_domain_B_intersection_A_B_union_A_CUB_l597_597437


namespace gabor_can_cross_l597_597977

open Real

-- Definitions based on conditions
def river_width : ℝ := 100
def total_island_perimeter : ℝ := 800
def banks_parallel : Prop := true

theorem gabor_can_cross (w : ℝ) (p : ℝ) (bp : Prop) : 
  w = river_width → 
  p = total_island_perimeter → 
  bp = banks_parallel → 
  ∃ d : ℝ, d ≤ 300 := 
by
  sorry

end gabor_can_cross_l597_597977


namespace problem_solution_l597_597042

noncomputable def F(x : ℝ) (f : ℝ → ℝ) : ℝ := f(x) + f(-x)

theorem problem_solution (f : ℝ → ℝ) (h_even_f : ∀ x, f(x) = f(-x)) (h_even_f2 : ∀ x, f(x+2) = f(-(x+2))) (h_f1 : f(1) = π/3) : 
  F 3 f = 2 * π / 3 := by
  sorry

end problem_solution_l597_597042


namespace julie_hours_per_week_l597_597146

theorem julie_hours_per_week :
  let summer_hours_per_week := 60,
      summer_weeks := 10,
      summer_earnings := 8000,
      school_year_weeks := 50,
      target_earnings := 10000 in
  (summer_earnings / (summer_hours_per_week * summer_weeks)) = 40 / 3 →
  (target_earnings / (40 / 3)) = 750 →
  (750 / school_year_weeks) = 15 :=
by
  intros h1 h2,
  exact div_eq_of_eq_mul_right (of_rat (show (50:ℝ) ≠ 0, by norm_num)) (by linarith [h1, h2])

end julie_hours_per_week_l597_597146


namespace sam_age_five_years_ago_l597_597498

variable (S : ℕ) -- Sam's current age

-- Conditions
def John_current_age := 3 * S
def John_age_in_15_years := 3 * S + 15
def Sam_age_in_15_years := S + 15
def Ted_current_age := S - 5
def Ted_age_in_15_years := S - 5 + 15

-- Given John is 3 times as old as Sam and in 15 years, John will be twice as old as Sam.
def john_condition_1 : Prop := John_age_in_15_years = 2 * Sam_age_in_15_years

-- Given Ted is 5 years younger than Sam and in 15 years, Ted will be three-fourths the age of Sam.
def ted_condition_1 : Prop := Ted_age_in_15_years = (3/4) * Sam_age_in_15_years

-- Prove that Sam was 10 years old five years ago.
theorem sam_age_five_years_ago (h₁ : john_condition_1 S) (h₂ : ted_condition_1 S) : S - 5 = 10 :=
  sorry

end sam_age_five_years_ago_l597_597498


namespace knicks_knocks_equivalence_l597_597955

theorem knicks_knocks_equivalence 
  (knicks knacks knocks : Type)
  (h1 : 9 * knicks = 3 * knacks)
  (h2 : 4 * knacks = 5 * knocks)
  : (80 * knocks = 192 * knicks) := 
by 
  sorry

end knicks_knocks_equivalence_l597_597955


namespace inequality_of_cos_sin_powers_l597_597894

open Real

theorem inequality_of_cos_sin_powers (α : ℝ) (hα1 : π / 4 < α)
                                      (hα2 : α < π / 2) :
  let a := (cos α) ^ (cos α)
  let b := (sin α) ^ (cos α)
  let c := (cos α) ^ (sin α)
  in c < a ∧ a < b :=
by {
  sorry
}

end inequality_of_cos_sin_powers_l597_597894


namespace largest_divisor_l597_597095

theorem largest_divisor (x : ℤ) (hx : x % 2 = 1) : 180 ∣ (15 * x + 3) * (15 * x + 9) * (10 * x + 5) := 
by
  sorry

end largest_divisor_l597_597095


namespace number_of_subsets_of_P_is_4_l597_597075

-- Define sets M and N
def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}

-- Define P as the intersection of M and N
def P : Set ℕ := M ∩ N

-- Theorem statement
theorem number_of_subsets_of_P_is_4 : (Finset.powerset P.to_finset).card = 4 := by
  sorry

end number_of_subsets_of_P_is_4_l597_597075


namespace seven_divisors_of_123_456_l597_597453

def is_divisor (n k : ℕ) : Prop := k % n = 0

theorem seven_divisors_of_123_456 : 
  (finset.filter (λ n => is_divisor n 123456) (finset.range 10)).card = 7 := by
  sorry

end seven_divisors_of_123_456_l597_597453


namespace cube_tetrahedron_volume_ratio_l597_597357

theorem cube_tetrahedron_volume_ratio :
  let s := 2
  let v1 := (0, 0, 0)
  let v2 := (2, 2, 0)
  let v3 := (2, 0, 2)
  let v4 := (0, 2, 2)
  let a := Real.sqrt 8 -- Side length of the tetrahedron
  let volume_tetra := (a^3 * Real.sqrt 2) / 12
  let volume_cube := s^3
  volume_cube / volume_tetra = 6 * Real.sqrt 2 := 
by
  -- Proof content skipped
  intros
  sorry

end cube_tetrahedron_volume_ratio_l597_597357


namespace circumradius_excircle_tangent_points_l597_597500

theorem circumradius_excircle_tangent_points
  (A B C : Point)
  (A' B' C' : Point)
  (R R' r h_a h_b h_c : ℝ)
  (circumradius_ABC : Circumradius ΔABC R)
  (inradius_ABC : Inradius ΔABC r)
  (altitudes_ABC : Altitudes ΔABC h_a h_b h_c)
  (tangency_conditions : TangencyPoints ΔA'B'C' ΔABC)
  (circumradius_A'B'C' : Circumradius ΔA'B'C' R')
  :
  R' = (1 / (2 * r)) * sqrt (2 * R * (2 * R - h_a) * (2 * R - h_b) * (2 * R - h_c)) := sorry

end circumradius_excircle_tangent_points_l597_597500


namespace DavidCrunchesLessThanZachary_l597_597759

-- Definitions based on conditions
def ZacharyPushUps : ℕ := 44
def ZacharyCrunches : ℕ := 17
def DavidPushUps : ℕ := ZacharyPushUps + 29
def DavidCrunches : ℕ := 4

-- Problem statement we need to prove:
theorem DavidCrunchesLessThanZachary : DavidCrunches = ZacharyCrunches - 13 :=
by
  -- Proof will go here
  sorry

end DavidCrunchesLessThanZachary_l597_597759


namespace repeating_block_length_7_div_13_l597_597614

theorem repeating_block_length_7_div_13 : 
  let d := 7 / 13 in repeating_block_length d = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597614


namespace rect_eq_curve_C_general_form_line_chord_length_AB_l597_597989

theorem rect_eq_curve_C (θ ρ : ℝ) (h1 : ρ = 2 * (Math.cos θ) / (Math.sin θ)^2) :
    ∃ (x y : ℝ), y^2 = 2 * x :=
sorry

theorem general_form_line (t : ℝ) :
    ∃ (x y : ℝ), (x = 1 + t) ∧ (y = t - 3) ∧ (x - y - 4 = 0) :=
sorry

theorem chord_length_AB (t1 t2 : ℝ) (h1 : t1 + t2 = 8) (h2 : t1 * t2 = 7) :
    |√2 * (t1 - t2)| = 6 * √2 :=
sorry

end rect_eq_curve_C_general_form_line_chord_length_AB_l597_597989


namespace minimal_slipper_pairs_l597_597859

noncomputable theory

def chews_left_s (n : ℕ) : ℕ :=
  -- Assuming this function calculates the number of left slippers chewed in n days
  sorry

def chews_right_s (n : ℕ) : ℕ :=
  -- Assuming this function calculates the number of right slippers chewed in n days
  sorry

def probability_distrib (n k : ℕ) : ℚ :=
  finset.sum (finset.range (k + 1)).filter (fun i => i ≥ (n - k)) (λ i, ((nat.choose n i) : ℚ) * (0.5 ^ n))

theorem minimal_slipper_pairs :
  ∃ k : ℕ, k = 5 ∧ probability_distrib 7 k ≥ 0.8 :=
begin
  use 5,
  split,
  { refl },  -- Proving k = 5
  { sorry }  -- Sketch: Calculate and sum up the probabilities based on binomial distribution.
end

end minimal_slipper_pairs_l597_597859


namespace james_total_money_l597_597128

theorem james_total_money (bills : ℕ) (value_per_bill : ℕ) (initial_money : ℕ) : 
  bills = 3 → value_per_bill = 20 → initial_money = 75 → initial_money + (bills * value_per_bill) = 135 :=
by
  intros hb hv hi
  rw [hb, hv, hi]
  -- Algebraic simplification
  sorry

end james_total_money_l597_597128


namespace min_degree_implies_K6_l597_597254

theorem min_degree_implies_K6 
  (G : SimpleGraph (Fin 1991)) 
  (h1 : ∀ (v : Fin 1991), G.degree v ≥ 1593) :
  ∃ (H : SimpleGraph (Fin 1991)), H.isClique 6 ∧ H ≤ G :=
sorry

end min_degree_implies_K6_l597_597254


namespace sum_of_solutions_l597_597387

theorem sum_of_solutions (x : ℝ) (h₁ : (x - 8)^2 = 49) :
  x = 15 ∨ x = 1 → ∃ s, s = 16 :=
by
  intro h₂
  cases h₂ with h₃ h₄
  { use 16
    sorry }
  { use 16
    sorry }

end sum_of_solutions_l597_597387


namespace solution_set_fractional_inequality_l597_597001

theorem solution_set_fractional_inequality :
  {x : ℝ | -1 < x ∧ x ≤ 2} = {x : ℝ | (2 - x) / (x + 1) ≥ 0} :=
begin
  sorry
end

end solution_set_fractional_inequality_l597_597001


namespace trajectory_equation_range_of_AB_ST_l597_597309

-- Define the given conditions
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

def on_perpendicular_line_to_x_axis (M N : ℝ × ℝ) : Prop := 
  M.fst = N.fst ∧ N.snd = 0

def point_P (N M P : ℝ × ℝ) : Prop := 
  P.fst = N.fst + (M.fst - N.fst) * (√3 / 2) ∧
  P.snd = N.snd + (M.snd - N.snd) * (√3 / 2)

-- Define the trajectory of point P
def trajectory_E (x y : ℝ) : Prop := 
  (x^2) / 4 + (y^2) / 3 = 1

-- The mathematical problem statements in Lean
theorem trajectory_equation (M N P : ℝ × ℝ) 
  (hM : circle_C M.fst M.snd)
  (hN : on_perpendicular_line_to_x_axis M N)
  (hP : point_P N M P) :
  trajectory_E P.fst P.snd := sorry

theorem range_of_AB_ST (Q A B S T : ℝ × ℝ)
  (hQ : Q = (0, 1))
  (hl_intersects_E: ∃ k : ℝ, ∀ x : ℝ, A = (x, k * x + 1) ∧ B = (x, k * x + 1))
  (hl_intersects_C: ∃ (x y : ℝ), (S = (x, y) ∨ T = (x, y)) ∧ circle_C x y) :
  8 * sqrt 2 ≤ abs (dist A B * dist S T) ∧ 
  abs (dist A B * dist S T) < 8 * sqrt 3 := sorry

end trajectory_equation_range_of_AB_ST_l597_597309


namespace sin_780_eq_sqrt3_div_2_l597_597347

theorem sin_780_eq_sqrt3_div_2 :
  Real.sin (780 * Real.pi / 180) = (Real.sqrt 3) / 2 :=
by
  sorry

end sin_780_eq_sqrt3_div_2_l597_597347


namespace tony_lottery_winning_l597_597714

theorem tony_lottery_winning
  (tickets : ℕ) (winning_numbers : ℕ) (worth_per_number : ℕ) (identical_numbers : Prop)
  (h_tickets : tickets = 3) (h_winning_numbers : winning_numbers = 5) (h_worth_per_number : worth_per_number = 20)
  (h_identical_numbers : identical_numbers) :
  (tickets * (winning_numbers * worth_per_number) = 300) :=
by
  sorry

end tony_lottery_winning_l597_597714


namespace limit_of_sin_ratio_l597_597825

open Real

theorem limit_of_sin_ratio : 
  tendsto (fun x => (sin (x - 1) / (x - 1)) ^ (sin (x - 1) / (x - 1 - sin (x - 1)))) (𝓝 1) (𝓝 (real.exp (-1))) := 
  sorry

end limit_of_sin_ratio_l597_597825


namespace valid_numbers_l597_597374

def is_valid_digit (d : ℕ) : Prop :=
  d = 3 ∨ d = 7

def is_multiple_of (n : ℕ) (m : ℕ) : Prop :=
  n % m = 0

def is_valid_number (n : ℕ) : Prop :=
  ∀ (i : ℕ), i < 7 → is_valid_digit ((n / (10 ^ i)) % 10)

def seven_digit_numbers : set ℕ :=
  {n | 10^6 ≤ n ∧ n < 10^7}

def valid_seven_digit_numbers : set ℕ :=
  {n | n ∈ seven_digit_numbers ∧ is_valid_number n ∧ is_multiple_of n 21}

theorem valid_numbers :
  valid_seven_digit_numbers =
  {3373377, 7373373, 7733733, 3733737, 7337337, 3777333} :=
by sorry

end valid_numbers_l597_597374


namespace product_sequence_eq_l597_597828

theorem product_sequence_eq : (∏ k in Finset.range 501, (4 * (k + 1)) / ((4 * (k + 1)) + 4)) = (1 : ℚ) / 502 := 
sorry

end product_sequence_eq_l597_597828


namespace updated_average_weight_l597_597473

def average (weights : List ℕ) : ℚ :=
  (weights.sum : ℚ) / weights.length

theorem updated_average_weight :
  let first_weights := [50, _, _, _, 70]
  let first_weights_corrected := [55, _, _, _, 75]
  let nine_weights := first_weights_corrected.append [_, _, _, _]
  let twelve_weights := nine_weights.append [_, _, _]
  let twelve_weights_corrected := (nine_weights.append [87, _, _])
  average first_weights = 60 → 
  average nine_weights = 63 → 
  average twelve_weights = 64 →
  average twelve_weights_corrected = 64.42 := by
sorry

end updated_average_weight_l597_597473


namespace repeating_block_length_7_div_13_l597_597582

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597582


namespace maximum_value_of_k_l597_597057

noncomputable def problem_statement (k : ℤ) (x : ℤ) : Prop :=
  k x - 5 = 2021 x + 2 k ∧ x > 0

theorem maximum_value_of_k (k_max : ℤ) :
  (∃ k x, problem_statement k x) → k_max = 6068 :=
by
  sorry

end maximum_value_of_k_l597_597057


namespace chickens_count_l597_597664

def total_animals := 13
def total_legs := 44
def legs_per_chicken := 2
def legs_per_buffalo := 4

theorem chickens_count : 
  (∃ c b : ℕ, c + b = total_animals ∧ legs_per_chicken * c + legs_per_buffalo * b = total_legs ∧ c = 4) :=
by
  sorry

end chickens_count_l597_597664


namespace minimum_value_squared_sum_minimum_value_squared_sum_equality_l597_597517

theorem minimum_value_squared_sum (a b c t : ℝ) (h : a + b + c = t) : 
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

theorem minimum_value_squared_sum_equality (a b c t : ℝ) (h : a + b + c = t) 
  (ha : a = t / 3) (hb : b = t / 3) (hc : c = t / 3) : 
  a^2 + b^2 + c^2 = t^2 / 3 := by
  sorry

end minimum_value_squared_sum_minimum_value_squared_sum_equality_l597_597517


namespace rectangles_in_5x5_grid_l597_597730

theorem rectangles_in_5x5_grid : 
  let grid_rows := 5
  let grid_cols := 5
  -- A function that calculates the number of rectangles in an n x m grid
  num_rectangles_in_grid grid_rows grid_cols = 225 :=
  sorry

end rectangles_in_5x5_grid_l597_597730


namespace number_of_false_propositions_l597_597431

theorem number_of_false_propositions 
  (P1 : ∀ a b : ℝ, a ≥ b → b > -1 → (a / (1 + a)) ≥ (b / (1 + b)))
  (P2 : ∀ m n : ℤ, m > 0 → n > 0 → m ≤ n → real.sqrt (m * (n - m)) ≤ n / 2)
  (P3 : ∀ a b x1 y1 : ℝ, (a - x1)^2 + (b - y1)^2 = 1 → x1^2 + y1^2 = 9 → ¬((a - 0)^2 + (b - 0)^2 = (3 + 1)^2))
  : (P1, P2, P3)False
    → \sum (\cond, P1, P2) == 2 and P3== 1 False:= sorry

end number_of_false_propositions_l597_597431


namespace find_first_number_l597_597002

def first_number_in_expression (x : ℝ) : Prop :=
  x + 2 * (8 - 3) = 24.16

theorem find_first_number :
  ∃ x : ℝ, first_number_in_expression x ∧ x = 14.16 :=
by
  existsi 14.16
  unfold first_number_in_expression
  norm_num
  sorry

end find_first_number_l597_597002


namespace cos_alpha_third_quadrant_l597_597909

theorem cos_alpha_third_quadrant (α : ℝ) (h1 : Real.sin α = -5 / 13) (h2 : Real.tan α > 0) : Real.cos α = -12 / 13 := 
sorry

end cos_alpha_third_quadrant_l597_597909


namespace gumballs_result_l597_597140

def gumballs_after_sharing_equally (initial_joanna : ℕ) (initial_jacques : ℕ) (multiplier : ℕ) : ℕ :=
  let joanna_total := initial_joanna + initial_joanna * multiplier
  let jacques_total := initial_jacques + initial_jacques * multiplier
  (joanna_total + jacques_total) / 2

theorem gumballs_result :
  gumballs_after_sharing_equally 40 60 4 = 250 :=
by
  sorry

end gumballs_result_l597_597140


namespace increasing_function_l597_597652

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_function (a : ℝ) (h : a ≥ 1) : 
  ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
by 
  sorry

end increasing_function_l597_597652


namespace rotated_curve_equation_is_correct_l597_597541

open Real

def rotation_matrix_45 : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.of ![![cos (π / 4), -sin (π / 4)], ![sin (π / 4), cos (π / 4)]]

def initial_curve (x y : ℝ) : Prop :=
  x + y^2 = 1

def transformed_curve (x' y' : ℝ) : Prop :=
  x'^2 + y'^2 - 2*x'*y' + sqrt 2 * x' + sqrt 2 * y' - 2 = 0

theorem rotated_curve_equation_is_correct :
  ∀ x' y', transformed_curve x' y' ↔
  initial_curve (rotation_matrix_45⁻¹.mul_vec (Vector.of (x', y'))).head (rotation_matrix_45⁻¹.mul_vec (Vector.of (x', y'))).tail.head :=
by
  sorry

end rotated_curve_equation_is_correct_l597_597541


namespace sum_perimeter_zero_l597_597481

-- Definitions for the conditions
def is_grid (grid : ℕ → ℕ → ℤ) (n : ℕ) := ∀ (i j : ℕ), i < n ∧ j < n

def sum_2x2_zero (grid : ℕ → ℕ → ℤ) :=
  ∀ (i j : ℕ), i ≤ 5 ∧ j ≤ 5 → 
  grid i j + grid (i+1) j + grid i (j+1) + grid (i+1) (j+1) = 0

def sum_3x3_zero (grid : ℕ → ℕ → ℤ) :=
  ∀ (i j : ℕ), i ≤ 4 ∧ j ≤ 4 → 
  grid i j + grid (i+1) j + grid (i+2) j +
  grid i (j+1) + grid (i+1) (j+1) + grid (i+2) (j+1) +
  grid i (j+2) + grid (i+1) (j+2) + grid (i+2) (j+2) = 0

def perimeter_sum (grid : ℕ → ℕ → ℤ) :=
  (∑ i in finset.range 7, grid i 0) +
  (∑ i in finset.range 7, grid i 6) +
  (∑ j in (finset.range 5), grid 0 (j+1)) + 
  (∑ j in (finset.range 5), grid 6 (j+1))

-- The theorem to prove
theorem sum_perimeter_zero (grid : ℕ → ℕ → ℤ) 
  (h_is_grid : is_grid grid 7)
  (h_sum_2x2_zero : sum_2x2_zero grid)
  (h_sum_3x3_zero : sum_3x3_zero grid) :
  perimeter_sum grid = 0 := 
by
  sorry

end sum_perimeter_zero_l597_597481


namespace perpendicular_lines_a_value_l597_597425

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (a-2)*x + a*y = 1 ↔ 2*x + 3*y = 5) → a = 4/5 := by
sorry

end perpendicular_lines_a_value_l597_597425


namespace points_player_1_after_13_rotations_l597_597684

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end points_player_1_after_13_rotations_l597_597684


namespace binary_to_decimal_l597_597845

theorem binary_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 13) :=
by
  sorry

end binary_to_decimal_l597_597845


namespace repeating_block_length_7_div_13_l597_597583

theorem repeating_block_length_7_div_13 : 
  ∀ (d : ℚ), d = 7 / 13 → (∃ n : ℕ, d = (0 + '0' * 10⁻¹ + '5' * 10⁻² + '3' * 10⁻³ + '8' * 10⁻⁴ + '4' * 10⁻⁵ + '6' * 10⁻⁶ + ('1' * 10⁻⁷ + '5' * 10⁻⁸ + '3' * 10⁻⁹ + '8' * 10⁻¹⁰ + '4' * 10⁻¹¹ + '6' * 10⁻¹²))^n) -> n = 6 := 
by
  sorry

end repeating_block_length_7_div_13_l597_597583


namespace find_2nd_month_sales_l597_597307

def sales_of_1st_month : ℝ := 2500
def sales_of_3rd_month : ℝ := 9855
def sales_of_4th_month : ℝ := 7230
def sales_of_5th_month : ℝ := 7000
def sales_of_6th_month : ℝ := 11915
def average_sales : ℝ := 7500
def months : ℕ := 6
def total_required_sales : ℝ := average_sales * months
def total_known_sales : ℝ := sales_of_1st_month + sales_of_3rd_month + sales_of_4th_month + sales_of_5th_month + sales_of_6th_month

theorem find_2nd_month_sales : 
  ∃ (sales_of_2nd_month : ℝ), total_required_sales = sales_of_1st_month + sales_of_2nd_month + sales_of_3rd_month + sales_of_4th_month + sales_of_5th_month + sales_of_6th_month ∧ sales_of_2nd_month = 10500 := by
  sorry

end find_2nd_month_sales_l597_597307


namespace part1_part2_part3_l597_597179

open Set

-- Define the sets A and B and the universal set
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def U : Set ℝ := univ  -- Universal set R

theorem part1 : A ∩ B = { x | 3 ≤ x ∧ x < 7 } :=
by { sorry }

theorem part2 : U \ A = { x | x < 3 ∨ x ≥ 7 } :=
by { sorry }

theorem part3 : U \ (A ∪ B) = { x | x ≤ 2 ∨ x ≥ 10 } :=
by { sorry }

end part1_part2_part3_l597_597179


namespace integral_sqrt_minus_x_l597_597372

theorem integral_sqrt_minus_x (a b : ℝ) : (a = 0) → (b = 2) → 
  (∫ x in a..b, (sqrt (4 - x^2) - x)) = (Real.pi - 2) :=
begin
  intros ha hb,
  rw [ha, hb],
  sorry
end

end integral_sqrt_minus_x_l597_597372


namespace sin_780_eq_sqrt3_div_2_l597_597353

theorem sin_780_eq_sqrt3_div_2 : Real.sin (780 * Real.pi / 180) = Math.sqrt 3 / 2 := by
  sorry

end sin_780_eq_sqrt3_div_2_l597_597353
