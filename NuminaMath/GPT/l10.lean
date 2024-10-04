import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.QuadraticDiscriminantAnalysis
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Ring.Quot
import Mathlib.Analysis.Convex.Function
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Time.Clock
import Mathlib.Geometry.Euclidean.Angle
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Geometry.Euclidean.Triangle.Altitude
import Mathlib.Geometry.Euclidean.Triangle.Basic
import Mathlib.Geometry.Euclidean.Triangle.ExCircle
import Mathlib.Geometry.Euclidean.Triangle.InCircle
import Mathlib.LinearAlgebra.AffineSpace.Independent
import Mathlib.MeasureTheory.Measure.Space
import Mathlib.NumberTheory.GCD
import Mathlib.Tactic
import Mathlib.Tactic.Induction
import Mathlib.Tactic.Linarith
import Mathlib.Topology.EuclideanSkeleton
import Real

namespace percentage_of_rotten_bananas_l10_10493

-- Define the initial conditions and the question as a Lean theorem statement
theorem percentage_of_rotten_bananas (oranges bananas : ℕ) (perc_rot_oranges perc_good_fruits : ℝ) 
  (total_fruits good_fruits good_oranges good_bananas rotten_bananas perc_rot_bananas : ℝ) :
  oranges = 600 →
  bananas = 400 →
  perc_rot_oranges = 0.15 →
  perc_good_fruits = 0.886 →
  total_fruits = (oranges + bananas) →
  good_fruits = (perc_good_fruits * total_fruits) →
  good_oranges = ((1 - perc_rot_oranges) * oranges) →
  good_bananas = (good_fruits - good_oranges) →
  rotten_bananas = (bananas - good_bananas) →
  perc_rot_bananas = ((rotten_bananas / bananas) * 100) →
  perc_rot_bananas = 6 :=
by
  intros; sorry

end percentage_of_rotten_bananas_l10_10493


namespace longer_to_shorter_ratio_is_two_l10_10579

noncomputable def ratio_of_longer_to_shorter_side (
  inner_square_side : ℝ,
  outer_square_side : ℝ,
  inner_square_area : ℝ := inner_square_side ^ 2,
  outer_square_area : ℝ := outer_square_side ^ 2,
  outer_to_inner_area_ratio : ℝ := outer_square_area / inner_square_area,
  rectangle_short_side : ℝ,
  rectangle_long_side : ℝ
) : Prop :=
  let inner_square_to_outer_square_side_ratio := outer_square_side / inner_square_side in
  let condition1 := rectangle_short_side = inner_square_side in
  let condition2 := rectangle_long_side = outer_square_side - inner_square_side in
  outer_to_inner_area_ratio = 9 → inner_square_to_outer_square_side_ratio = 3 → 
  (rectangle_long_side / rectangle_short_side) = 2

-- shorthand theorem statement to enforce the proof problem
theorem longer_to_shorter_ratio_is_two
  (inner_square_side : ℝ)
  (outer_square_side : ℝ)
  (rectangle_short_side : ℝ)
  (rectangle_long_side : ℝ)
  (h1 : 9 = (outer_square_side ^ 2) / (inner_square_side ^ 2))
  (h2 : 3 = outer_square_side / inner_square_side)
  (h3 : rectangle_short_side = inner_square_side)
  (h4 : rectangle_long_side = outer_square_side - inner_square_side)
: rectangle_long_side / rectangle_short_side = 2 :=
by sorry

end longer_to_shorter_ratio_is_two_l10_10579


namespace final_speed_of_ball_l10_10850

/--
 A small rubber ball moves horizontally between two vertical walls. One wall is fixed, and the other wall moves away from it at a constant speed u.
 The ball's collisions are perfectly elastic. The initial speed of the ball is v₀. Prove that after 10 collisions with the moving wall, the ball's speed is 17 cm/s.
-/
theorem final_speed_of_ball
    (u : ℝ) (v₀ : ℝ) (n : ℕ)
    (u_val : u = 100) (v₀_val : v₀ = 2017) (n_val : n = 10) :
    v₀ - 2 * u * n = 17 := 
    by
    rw [u_val, v₀_val, n_val]
    sorry

end final_speed_of_ball_l10_10850


namespace imaginary_part_of_z_l10_10621

-- Define the complex number z
noncomputable def z : ℂ := (1 / (1 + Complex.i)) + (Complex.i ^ 3)

-- Define the imaginary part function
def imaginary_part (z : ℂ) : ℝ := z.im

-- State the theorem to be proven
theorem imaginary_part_of_z :
  imaginary_part z = -3 / 2 :=
by sorry

end imaginary_part_of_z_l10_10621


namespace smallest_angle_in_trapezoid_l10_10264

theorem smallest_angle_in_trapezoid 
  (a b d e : ℝ) 
  (ht1 : a + b = 180) 
  (ht2 : b + e = 140) 
  (ht3 : a + (a + d) + b + (b + e) = 360) 
  (hlargest : max (max a (a + d)) (max b (b + e)) = 140) : 
  min a (min (a + d) (min b (b + e))) = 20 :=
begin
  sorry
end

end smallest_angle_in_trapezoid_l10_10264


namespace is_composite_1010_pattern_l10_10738

theorem is_composite_1010_pattern (k : ℕ) (h : k ≥ 2) : (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (1010^k + 101 = a * b)) :=
  sorry

end is_composite_1010_pattern_l10_10738


namespace correct_proposition_l10_10267

-- Declare the space, lines, and planes
variables (V : Type*) [inner_product_space ℝ V]
variables (l₁ l₂ : submodule ℝ V) (P : affine_subspace ℝ V)

-- Condition definitions
def condition_1 := ∀ (P : affine_subspace ℝ V), (P ⊓ l₁).direction ≤ (P ⊓ l₂).direction
def condition_2 := ∀ (P : affine_subspace ℝ V), (direction P).perpendicular.l₁ ∧ (direction P).perpendicular.l₂ → l₁ = l₂
def condition_3 := ∀ (l : submodule ℝ V), (affine_subspace ℝ inf).direction ≤ direction P₁ ∧ (affine_subspace ℝ inf).direction ≤ direction P₂
def condition_4 := ∀ (P₃ : affine_subspace ℝ V), (direction P₃).perpendicular.P₁ ∧ (direction P₃).perpendicular.P₂ → direction P₁ = direction P₂

-- Proof problem statement
theorem correct_proposition (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) :
  (∀ (P : affine_subspace ℝ V), (direction P).perpendicular l₁ ∧ (direction P).perpendicular l₂ → l₁ = l₂) :=
sorry

end correct_proposition_l10_10267


namespace TommysFirstHousePrice_l10_10795

theorem TommysFirstHousePrice :
  ∃ (P : ℝ), 1.25 * P = 0.25 * 500000 ∧ P = 100000 :=
by
  use 100000
  split
  · norm_num
  · norm_num
  sorry

end TommysFirstHousePrice_l10_10795


namespace count_f_n_prime_l10_10704

open Nat

/-- Sum of positive divisors of n -/
def f (n : ℕ) : ℕ :=
  (divisors n).sum

/-- Predicate to check if a number is prime -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

/-- Main statement to be proven -/
theorem count_f_n_prime : ( Finset.filter (λ n, is_prime (f n)) (Finset.range 51) ).card = 5 :=
by
  sorry

end count_f_n_prime_l10_10704


namespace function_decreasing_l10_10180

variable {α β : Type*}
variables {a b : α} (f : α → β)
noncomputable def is_decreasing (f : α → β) [has_lt α] [has_le β] := 
  ∀ x₁ x₂, x₁ ∈ Ioo a b → x₂ ∈ Ioo a b → (x₁ < x₂) → (f x₁ > f x₂)

theorem function_decreasing {a b : ℝ} (f : ℝ → ℝ)
  (h : ∀ x₁ x₂ : ℝ, x₁ ∈ set.Ioo a b → x₂ ∈ set.Ioo a b → (x₁ - x₂) * (f x₁ - f x₂) < 0) :
  is_decreasing f :=
begin
  sorry
end

end function_decreasing_l10_10180


namespace dot_path_length_cube_l10_10062

-- Define the edge length
def edge_length : ℝ := 2

-- Initial dot position distance to one of the vertices (center of the top face)
def dot_to_vertex_distance : ℝ := sqrt 7

-- The length of the path followed by the dot after one full cycle
def total_path_length : ℝ := 2 * π * sqrt 7

-- Theorem to prove
theorem dot_path_length_cube :
  total_path_length = 2 * π * sqrt 7 := by
sorry

end dot_path_length_cube_l10_10062


namespace prove_rectangular_selection_l10_10563

def number_of_ways_to_choose_rectangular_region (horizontals verticals : ℕ) : ℕ :=
  (Finset.choose horizontals 2) * (Finset.choose verticals 2)

theorem prove_rectangular_selection :
  number_of_ways_to_choose_rectangular_region 5 5 = 100 :=
by
  sorry

end prove_rectangular_selection_l10_10563


namespace consecutive_integers_squares_sum_l10_10403

theorem consecutive_integers_squares_sum (n : ℤ) :
  (n + 1)^2 + (n + 2)^2 = (n - 2)^2 + (n - 1)^2 + n^2 ↔ (n = 12 ∨ n = 0) := by
  sorry

example : 
  (n = 12) → (n - 2, n - 1, n, n + 1, n + 2) = (10, 11, 12, 13, 14) := by
  intro h
  simp [h]
  sorry

end consecutive_integers_squares_sum_l10_10403


namespace lines_intersection_l10_10843

theorem lines_intersection :
  ∃ (t u : ℚ), 
    (∃ (x y : ℚ),
    (x = 2 - t ∧ y = 3 + 4 * t) ∧ 
    (x = -1 + 3 * u ∧ y = 6 + 5 * u) ∧ 
    (x = 28 / 17 ∧ y = 75 / 17)) := sorry

end lines_intersection_l10_10843


namespace number_of_integer_pairs_satisfying_equation_count_integer_pairs_satisfying_equation_l10_10644

theorem number_of_integer_pairs_satisfying_equation (m n : ℤ) :
  (m + n = m * n) ↔ ((m = 2 ∧ n = 2) ∨ (m = 0 ∧ n = 0)) :=
by sorry

theorem count_integer_pairs_satisfying_equation :
  {pair : (ℤ × ℤ) | pair.1 + pair.2 = pair.1 * pair.2}.to_finset.card = 2 :=
by sorry

end number_of_integer_pairs_satisfying_equation_count_integer_pairs_satisfying_equation_l10_10644


namespace two_lt_xn_yn_lt_three_l10_10602

noncomputable def x : ℕ → ℝ
| 0     := sqrt 3
| (n+1) := x n + sqrt (1 + (x n)^2)

noncomputable def y : ℕ → ℝ
| 0     := sqrt 3
| (n+1) := y n / (1 + sqrt (1 + (y n)^2))

theorem two_lt_xn_yn_lt_three (n : ℕ) : 2 < (x n) * (y n) ∧ (x n) * (y n) < 3 :=
sorry

end two_lt_xn_yn_lt_three_l10_10602


namespace Chandler_more_rolls_needed_l10_10941

theorem Chandler_more_rolls_needed :
  let total_goal := 12
  let sold_to_grandmother := 3
  let sold_to_uncle := 4
  let sold_to_neighbor := 3
  let total_sold := sold_to_grandmother + sold_to_uncle + sold_to_neighbor
  total_goal - total_sold = 2 :=
by
  sorry

end Chandler_more_rolls_needed_l10_10941


namespace snails_trails_divide_torus_l10_10011

theorem snails_trails_divide_torus {T : Type} [Torus T] (outer_equator_trail : Trail T) (helical_trail : Trail T) :
  divides_surface outer_equator_trail helical_trail 3 := 
sorry

end snails_trails_divide_torus_l10_10011


namespace max_perfect_squares_l10_10754

theorem max_perfect_squares (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (∀ x ∈ {a * a, a * (a + 2), b * b, b * (b + 2), a * b, (a + 2) * b, a * (b + 2), (a + 2) * (b + 2)}.filter (λ x, ∃ k, x = k * k), 
   x ≤ 2) := sorry

end max_perfect_squares_l10_10754


namespace cuboid_diagonal_cubes_l10_10840

def num_cubes_intersecting_diagonal (a b c : ℕ) : ℕ :=
  a + b + c - 2

theorem cuboid_diagonal_cubes :
  num_cubes_intersecting_diagonal 77 81 100 = 256 :=
by
  sorry

end cuboid_diagonal_cubes_l10_10840


namespace initial_balance_l10_10884

-- Define the conditions given in the problem
def transferred_percent_of_balance (X : ℝ) : ℝ := 0.15 * X
def balance_after_transfer (X : ℝ) : ℝ := 0.85 * X
def final_balance_after_refund (X : ℝ) (refund : ℝ) : ℝ := 0.85 * X + refund

-- Define the given values
def refund : ℝ := 450
def final_balance : ℝ := 30000

-- The theorem statement to prove the initial balance
theorem initial_balance (X : ℝ) (h : final_balance_after_refund X refund = final_balance) : 
  X = 34564.71 :=
by
  sorry

end initial_balance_l10_10884


namespace greatest_integer_less_than_l10_10803

theorem greatest_integer_less_than: ∀ (x : ℝ), x = -15 / 4 → ∃ (y : ℤ), y < x ∧ ∀ (z : ℤ), z < x → z ≤ y :=
begin
  intro x,
  intro hx,
  use -4,
  split,
  { rw hx, norm_num },
  { intros z hz, have := int.floor_le hz, simp at this, exact this },
end

end greatest_integer_less_than_l10_10803


namespace find_positive_real_numbers_l10_10912

open Real

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16

theorem find_positive_real_numbers (x : ℝ) (hx : x > 0) :
  satisfies_inequality x ↔ 15 * x^2 + 32 * x - 256 = 0 :=
sorry

end find_positive_real_numbers_l10_10912


namespace rectangle_enclosure_l10_10550
open BigOperators

theorem rectangle_enclosure (n m : ℕ) (hn : n = 5) (hm : m = 5) : 
  (∑ i in finset.range n, ∑ j in finset.range i, 1) * 
  (∑ k in finset.range m, ∑ l in finset.range k, 1) = 100 := by
  sorry

end rectangle_enclosure_l10_10550


namespace hexagon_coloring_l10_10772

noncomputable def numHexagonColorings : ℕ := 2

theorem hexagon_coloring :
  ∀ (G : Hexagon),
  (G.color = green ∧
  ∀ (H : Hexagon), (adjacent G H → H.color ≠ G.color) ∧
  (H.color = red ∨ H.color = yellow ∨ H.color = green) ∧
  ∀ (H1 H2 : Hexagon), adjacent H1 H2 → H1.color ≠ H2.color) →
  numHexagonColorings = 2 :=
by
  sorry

end hexagon_coloring_l10_10772


namespace circulation_value_l10_10874

def vector_field (x y : ℝ) : ℝ × ℝ :=
  ( sqrt (1 + x^2 + y^2)
  , y * (x * y + log (x + sqrt (1 + x^2 + y^2))))

def curve (R : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = R^2

noncomputable def circulation (R : ℝ) : ℝ :=
  ∮ (λ (x y : ℝ), if curve R x y then vector_field x y else (0, 0))

theorem circulation_value (R : ℝ) :
  circulation R = π * R^4 / 4 :=
sorry

end circulation_value_l10_10874


namespace vasily_salary_is_higher_l10_10407

variables (total_students graduated_students fyodor_initial_salary fyodor_yearly_increase failed_salary
           fraction_60k fraction_80k fraction_not_in_profession others_salary years : ℝ)

-- Constants
def total_students : ℝ := 300
def graduated_students : ℝ := 270
def fyodor_initial_salary : ℝ := 25000
def fyodor_yearly_increase : ℝ := 3000
def failed_salary : ℝ := 25000
def fraction_60k : ℝ := 1 / 5
def fraction_80k : ℝ := 1 / 10
def fraction_not_in_profession : ℝ := 1 / 20
def others_salary : ℝ := 40000
def years : ℝ := 4

-- Fyodor's salary after 4 years
def fyodor_salary : ℝ := fyodor_initial_salary + (years * fyodor_yearly_increase)

-- Vasily's expected salary
def p_graduate : ℝ := graduated_students / total_students
def p_not_graduate : ℝ := 1 - p_graduate

def p_60k : ℝ := fraction_60k
def p_80k : ℝ := fraction_80k
def p_not_in_profession : ℝ := fraction_not_in_profession
def p_others_salary : ℝ := 1 - p_60k - p_80k - p_not_in_profession

def expected_salary_graduate : ℝ :=
  (p_60k * 60000) + (p_80k * 80000) + (p_not_in_profession * failed_salary) + (p_others_salary * others_salary)

def expected_salary : ℝ := (p_graduate * expected_salary_graduate) + (p_not_graduate * failed_salary)

-- Expected salary values
def vasily_expected_salary_after_4_years : ℝ := expected_salary

-- Comparisons
def salary_difference : ℝ := vasily_expected_salary_after_4_years - fyodor_salary

theorem vasily_salary_is_higher :
    vasily_expected_salary_after_4_years = 45025 ∧ fyodor_salary = 37000 ∧ salary_difference = 8025 :=
  by
    sorry

end vasily_salary_is_higher_l10_10407


namespace goods_train_length_is_350_l10_10810

noncomputable def length_of_train (v_kmph : ℕ) (t_sec : ℕ) (platform_length : ℕ) : ℕ :=
  let v_mps := (v_kmph * 1000) / 3600
  let distance_covered := v_mps * t_sec
  distance_covered - platform_length

theorem goods_train_length_is_350 :
  length_of_train 72 30 250 = 350 :=
by
  -- Definitions
  let v_kmph : ℕ := 72
  let t_sec : ℕ := 30
  let platform_length : ℕ := 250
  
  -- Derived values
  let v_mps := (v_kmph * 1000) / 3600
  let distance_covered := v_mps * t_sec
  let length_of_train := distance_covered - platform_length
  
  -- Conclusion
  have h1 : v_mps = 20 := by sorry
  have h2 : distance_covered = 600 := by sorry
  have h3 : length_of_train = 350 := by sorry
  show length_of_train 72 30 250 = 350 from h3

end goods_train_length_is_350_l10_10810


namespace sequence_properties_l10_10714

-- Define conditions
variables (x y k : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) (k_pos : 0 < k) (k_ne_1 : k ≠ 1)
  (x_ne_y : x ≠ y)

-- Define the sequences
noncomputable def A : ℕ → ℝ
| 1       := (kx + y) / (k + 1)
| (n + 1) := (A n + H n) / 2

noncomputable def G : ℕ → ℝ
| 1       := Real.root (k + 1) (x^k * y)
| (n + 1) := Real.sqrt (A n * H n)

noncomputable def H : ℕ → ℝ
| 1       := (k + 1) * x * y / (kx + y)
| (n + 1) := 2 / (1 / A n + 1 / H n)

-- Define the proof problem
theorem sequence_properties : 1 + 16 + 256 = 273 :=
begin
  -- Proof will go here
  sorry
end

end sequence_properties_l10_10714


namespace find_f_of_2_l10_10952

theorem find_f_of_2 : ∃ (f : ℤ → ℤ), (∀ x : ℤ, f (x+1) = x^2 - 1) ∧ f 2 = 0 :=
by
  sorry

end find_f_of_2_l10_10952


namespace bill_pastry_combination_l10_10514

/--
Bill is to purchase exactly eight pastries from a shop with five kinds of pastries.
He must buy at least one of each kind, and he dislikes one type so he will buy only one of that type.
Prove the number of combinations that satisfy these requirements is 20.
-/
theorem bill_pastry_combination : 
  let n := 8
  let k := 5
  let disliked := 1
  n = 8 ∧ k = 5 ∧ disliked = 1 ∧ 
  (∀ (n k disliked : ℕ), n = 8 → k = 5 → disliked = 1 → true) →
  ∑ x in finset.Ico 0 (n+1), if n - disliked + k - 1 = k - 1 then 20 else 0 = 20 := 
by 
  sorry

end bill_pastry_combination_l10_10514


namespace brad_ate_six_halves_l10_10229

theorem brad_ate_six_halves (total_cookies : ℕ) (total_halves : ℕ) (greg_ate : ℕ) (halves_left : ℕ) (halves_brad_ate : ℕ) 
  (h1 : total_cookies = 14)
  (h2 : total_halves = total_cookies * 2)
  (h3 : greg_ate = 4)
  (h4 : halves_left = 18)
  (h5 : total_halves - greg_ate - halves_brad_ate = halves_left) :
  halves_brad_ate = 6 :=
by
  sorry

end brad_ate_six_halves_l10_10229


namespace three_digit_palindrome_sum_digits_l10_10071

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem three_digit_palindrome_sum_digits (x : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999)
  (h2 : is_palindrome x)
  (h3 : is_palindrome (x + 10))
  (h4 : 1000 ≤ x + 10 ∧ x + 10 ≤ 1031) :
  sum_of_digits x = 19 :=
by
  sorry

end three_digit_palindrome_sum_digits_l10_10071


namespace range_of_f_l10_10928

noncomputable def f (x : ℝ) : ℝ :=
  ( (real.pi / 2 - real.arcsin (x / 2)) ^ 2 ) + real.pi * real.arccos (x / 2)
  - ( real.arccos (x / 2) ^ 2 ) + (real.pi ^ 2 / 6) * (x ^ 2 + 3 * x + 2)
  + real.cos (real.pi * x / 4)

theorem range_of_f : set.Icc (5 / 6 * real.pi ^ 2 - 1) (2 * real.pi ^ 2 + 1) =
  set.image f (set.Icc (-2 : ℝ) (2 : ℝ)) :=
sorry

end range_of_f_l10_10928


namespace angle_B_is_60_l10_10056

-- Definitions according to conditions
variables {A B C A₀ C₀ : Point}
variable ω : Circle

-- Given conditions
axiom circumcircle_ABC : ω.circumscribed_around_triangle A B C
axiom midpoint_arc_BC : A₀.is_midpoint_of_arc B C ω
axiom midpoint_arc_AB : C₀.is_midpoint_of_arc A B ω
axiom tangent_segment : tangent (line_segment A₀ C₀) (inscribed_circle ABC)

-- The theorem to prove
theorem angle_B_is_60 :
  angle B = 60 :=
sorry

end angle_B_is_60_l10_10056


namespace surface_area_of_equal_volume_cube_l10_10073

def vol_rect_prism (l w h : ℝ) : ℝ := l * w * h
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

theorem surface_area_of_equal_volume_cube :
  (vol_rect_prism 5 5 45 = surface_area_cube 10.5) :=
by
  sorry

end surface_area_of_equal_volume_cube_l10_10073


namespace invertible_modulo_10_sum_prod_inverse_zero_l10_10001

theorem invertible_modulo_10_sum_prod_inverse_zero (a b c d : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h4 : d < 10)
    (h5 : Nat.gcd a 10 = 1) (h6 : Nat.gcd b 10 = 1) (h7 : Nat.gcd c 10 = 1) (h8 : Nat.gcd d 10 = 1) :
    ((a + b + c + d) * (Nat.invmod (a * b * c * d) 10)) % 10 = 0 :=
by sorry

end invertible_modulo_10_sum_prod_inverse_zero_l10_10001


namespace exists_divisible_by_power_of_two_with_digits_l10_10300

theorem exists_divisible_by_power_of_two_with_digits (a b : ℕ) (h_odd : a % 2 = 1) (h_even : b % 2 = 0) :
  ∀ n : ℕ, 0 < n → ∃ x : ℕ, (x % 2^n = 0) ∧ (∀ d : ℕ, d ∈ x.digits 10 → d = a ∨ d = b) :=
begin
  sorry
end

end exists_divisible_by_power_of_two_with_digits_l10_10300


namespace balloons_remaining_each_friend_l10_10399

def initial_balloons : ℕ := 250
def number_of_friends : ℕ := 5
def balloons_taken_back : ℕ := 11

theorem balloons_remaining_each_friend :
  (initial_balloons / number_of_friends) - balloons_taken_back = 39 :=
by
  sorry

end balloons_remaining_each_friend_l10_10399


namespace correct_statements_l10_10209

variable (a b c : ℝ)
variable h1 : a ≠ 0
variable f : ℝ → ℝ := λ x, a * x^2 + b * x + c
variable h2 : ∀ x, f x ≠ x

theorem correct_statements :
  (∀ x, f[f x] ≠ x) ∧
  (a > 0 → ∀ x, f[f x] > x) ∧
  (a + b + c = 0 → ∀ x, f[f x] < x) :=
sorry

end correct_statements_l10_10209


namespace correct_voronoi_partitioning_l10_10823
noncomputable theory

def is_closest_point (p x1 x2 : ℝ × ℝ) : Prop :=
  (dist p x1 < dist p x2)

def voronoi_diagram (pts : list (ℝ × ℝ)) (rect_bl rect_tr : ℝ × ℝ) : Prop :=
  ∀ p : ℝ × ℝ,
  (p.1 >= rect_bl.1) ∧ (p.1 <= rect_tr.1) ∧ (p.2 >= rect_bl.2) ∧ (p.2 <= rect_tr.2) →
  ∃! x : ℝ × ℝ, x ∈ pts ∧ (is_closest_point p x <$> (pts.filter (≠ x)))

theorem correct_voronoi_partitioning :
  voronoi_diagram [(0, 0), (2, 0), (4, 0), (0, 3), (4, 3), (0, 6), (2, 6), (4, 6)] (0, 0) (4, 6) :=
by
  sorry

end correct_voronoi_partitioning_l10_10823


namespace largest_multiple_of_7_less_than_neg85_l10_10421

theorem largest_multiple_of_7_less_than_neg85 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n < -85 ∧ n = -91 :=
by
  sorry

end largest_multiple_of_7_less_than_neg85_l10_10421


namespace fourth_term_geom_progression_l10_10889

theorem fourth_term_geom_progression : 
  ∀ (a b c : ℝ), 
    a = 4^(1/2) → 
    b = 4^(1/3) → 
    c = 4^(1/6) → 
    ∃ d : ℝ, d = 1 ∧ b / a = c / b ∧ c / b = 4^(1/6) / 4^(1/3) :=
by
  sorry

end fourth_term_geom_progression_l10_10889


namespace expression_eq_l10_10302

variable {α β γ δ p q : ℝ}

-- Conditions from the problem
def roots_eq1 (α β p : ℝ) : Prop := ∀ x : ℝ, (x - α) * (x - β) = x^2 + p*x - 1
def roots_eq2 (γ δ q : ℝ) : Prop := ∀ x : ℝ, (x - γ) * (x - δ) = x^2 + q*x + 1

-- The proof statement where the expression is equated to p^2 - q^2
theorem expression_eq (h1: roots_eq1 α β p) (h2: roots_eq2 γ δ q) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = p^2 - q^2 := sorry

end expression_eq_l10_10302


namespace compound_interest_years_l10_10391

theorem compound_interest_years
  (P₁ P₂ : ℝ) (R₁ R₂ : ℝ) (T₁ : ℝ) (SI CI : ℝ)
  (h₁ : SI = (P₁ * R₁ * T₁) / 100)
  (h₂ : CI = (P₂ * ((1 + R₂ / 100) ^ 2 - 1)))
  (h₃ : SI * 2 = CI)
  :  CI = 840 :=
by
  have h₄ : P₁ = 1750 := sorry
  have h₅ : R₁ = 8 := sorry
  have h₆ : T₁ = 3 := sorry
  have h₇ : P₂ = 4000 := sorry
  have h₈ : R₂ = 10 := sorry
  show CI = 840, from sorry

end compound_interest_years_l10_10391


namespace chord_intercepted_length_l10_10103

-- Define the circle equation and line equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0
def line_eq (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the function for the length of the chord
def chord_length : ℝ := 2 * Real.sqrt 3

-- The proof problem statement
theorem chord_intercepted_length :
  ∀ (x y : ℝ), line_eq x y → circle_eq x y → chord_length = 2 * Real.sqrt 3 := by
  intros x y hx hy
  sorry

end chord_intercepted_length_l10_10103


namespace length_DI_l10_10394

variables (A B C D I : ℝ×ℝ)
variables (AB AC : ℝ) (angleBAC : ℝ) (radius : ℝ)

def is_incenter (I : ℝ×ℝ) (A B C : ℝ×ℝ) := 
  -- Definition of incenter would go here
  sorry

def is_midpoint (D : ℝ×ℝ) (B C : ℝ×ℝ) :=
  -- Definition of midpoint would go here
  sorry

def dist (P Q : ℝ×ℝ) : ℝ := 
  ( (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 ) ^ (1/2)

theorem length_DI :
  let A := (0, 0) in
  let B := (6, 0) in
  let C := (0, 6) in
  let D := (3 * real.sqrt 2, 3 * real.sqrt 2) in
  let I := (6 - 3 * real.sqrt 2, 6 - 3 * real.sqrt 2) in
  AB = AC ∧ AB = 6 ∧ AC = 6 ∧
  angleBAC = 90 ∧
  radius = 3 ∧
  is_midpoint D B C ∧
  is_incenter I A B C →
  dist D I = 6 * real.sqrt 6 - 6 * real.sqrt 3 :=
by 
  intros h,
  sorry

end length_DI_l10_10394


namespace correct_statement_c_l10_10444

-- Definitions
variables {Point : Type*} {Line Plane : Type*}
variables (l m : Line) (α β : Plane)

-- Conditions
def parallel_planes (α β : Plane) : Prop := sorry  -- α ∥ β
def perpendicular_line_plane (l : Line) (α : Plane) : Prop := sorry  -- l ⊥ α
def line_in_plane (l : Line) (α : Plane) : Prop := sorry  -- l ⊂ α
def line_perpendicular (l m : Line) : Prop := sorry  -- l ⊥ m

-- Theorem to be proven
theorem correct_statement_c 
  (α β : Plane) (l : Line)
  (h_parallel : parallel_planes α β)
  (h_perpendicular : perpendicular_line_plane l α) :
  ∀ (m : Line), line_in_plane m β → line_perpendicular m l := 
sorry

end correct_statement_c_l10_10444


namespace bottles_produced_l10_10034

/-- 
14 machines produce 2520 bottles in 4 minutes, given that 6 machines produce 270 bottles per minute. 
-/
theorem bottles_produced (rate_6_machines : Nat) (bottles_per_minute : Nat) 
  (rate_one_machine : Nat) (rate_14_machines : Nat) (total_production : Nat) : 
  rate_6_machines = 6 ∧ bottles_per_minute = 270 ∧ rate_one_machine = bottles_per_minute / rate_6_machines 
  ∧ rate_14_machines = 14 * rate_one_machine ∧ total_production = rate_14_machines * 4 → 
  total_production = 2520 :=
sorry

end bottles_produced_l10_10034


namespace probability_neither_l10_10458

-- Defining the total number of buyers
def total_buyers : ℕ := 100

-- Defining the number of buyers who purchase cake mix
def buyers_cake_mix : ℕ := 50

-- Defining the number of buyers who purchase muffin mix
def buyers_muffin_mix : ℕ := 40

-- Defining the number of buyers who purchase both cake mix and muffin mix
def buyers_both : ℕ := 18

-- Defining the theorem to prove
theorem probability_neither : 
  (total_buyers - (buyers_cake_mix + buyers_muffin_mix - buyers_both)) / total_buyers.to_real = 0.28 :=
by
  sorry

end probability_neither_l10_10458


namespace find_all_primes_l10_10136

theorem find_all_primes (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r):
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ↔ 
  (p^2q + q^2p) / (p^3 - pq + q^3) = r :=
by sorry

end find_all_primes_l10_10136


namespace graph_of_f_passes_through_fixed_point_l10_10973

-- Given conditions
variables {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)

-- Inverse function definition
def f_inv (x : ℝ) : ℝ := 2 + log a (1 - x)

-- Definition of the original function as the inverse of f_inv
def f (y : ℝ) : ℝ := sorry -- In its place, we assume it exists as the inverse of f_inv

-- The point that is proposed to be on the graph of y = f(x)
def fixed_point : ℝ × ℝ := (2, 0)

theorem graph_of_f_passes_through_fixed_point : f (2 : ℝ) = 0 := sorry

end graph_of_f_passes_through_fixed_point_l10_10973


namespace solve_for_x_l10_10160

theorem solve_for_x (x : ℝ) (h : sqrt (5 * x + 9) = 11) : x = 112 / 5 :=
by {
  -- Here you can assume the proof steps would go.
  sorry
}

end solve_for_x_l10_10160


namespace isosceles_triangle_of_dot_product_zero_l10_10197

theorem isosceles_triangle_of_dot_product_zero 
  {A B C : Type*} [inner_product_space ℝ A]
  (AB AC : A) (h : \overrightarrow{BC} = \overrightarrow{AC} - \overrightarrow{AB})
  (dot_product_zero : \overrightarrow{BC} \cdot (\overrightarrow{AB} + \overrightarrow{AC}) = 0) :
  | \overrightarrow{AC} | = | \overrightarrow{AB} | :=
sorry

end isosceles_triangle_of_dot_product_zero_l10_10197


namespace ratio_of_fallen_cakes_is_one_half_l10_10881

noncomputable def ratio_fallen_to_total (total_cakes fallen_cakes pick_up destroyed_cakes : ℕ) :=
  fallen_cakes / total_cakes

theorem ratio_of_fallen_cakes_is_one_half :
  ∀ (total_cakes fallen_cakes pick_up destroyed_cakes : ℕ),
    total_cakes = 12 →
    pick_up = fallen_cakes / 2 →
    pick_up = destroyed_cakes →
    destroyed_cakes = 3 →
    ratio_fallen_to_total total_cakes fallen_cakes pick_up destroyed_cakes = 1 / 2 :=
by
  intros total_cakes fallen_cakes pick_up destroyed_cakes h1 h2 h3 h4
  rw [h1, h4, ratio_fallen_to_total]
  -- proof goes here
  sorry

end ratio_of_fallen_cakes_is_one_half_l10_10881


namespace ratio_of_triangle_areas_l10_10794

theorem ratio_of_triangle_areas (n : ℝ) 
  (h1 : ∀ (Δ : ℝ → ℝ → ℝ), Δ (a * c) = 1 * a + 1 * c /\ a = 2 * n /\ c = 1/2 * n -> (Δ (h1 + h2) = n))
  (h2 : 0 < n) : 
  ∃ k, k = 1/(4*n) :=
by
  sorry

end ratio_of_triangle_areas_l10_10794


namespace periodic_odd_function_example_l10_10529

open Real

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem periodic_odd_function_example (f : ℝ → ℝ) 
  (h_odd : odd f) 
  (h_periodic : periodic f 2) : 
  f 1 + f 4 + f 7 = 0 := 
sorry

end periodic_odd_function_example_l10_10529


namespace christian_sue_need_more_money_l10_10883

-- Definitions based on the given conditions
def bottle_cost : ℕ := 50
def christian_initial : ℕ := 5
def sue_initial : ℕ := 7
def christian_mowing_rate : ℕ := 5
def christian_mowing_count : ℕ := 4
def sue_walking_rate : ℕ := 2
def sue_walking_count : ℕ := 6

-- Prove that Christian and Sue will need 6 more dollars to buy the bottle of perfume
theorem christian_sue_need_more_money :
  let christian_earning := christian_mowing_rate * christian_mowing_count
  let christian_total := christian_initial + christian_earning
  let sue_earning := sue_walking_rate * sue_walking_count
  let sue_total := sue_initial + sue_earning
  let total_money := christian_total + sue_total
  50 - total_money = 6 :=
by
  sorry

end christian_sue_need_more_money_l10_10883


namespace fraction_of_air_conditioned_rooms_rented_l10_10733

variable (R : ℚ)
variable (h1 : R > 0)
variable (rented_rooms : ℚ := (3/4) * R)
variable (air_conditioned_rooms : ℚ := (3/5) * R)
variable (not_rented_rooms : ℚ := (1/4) * R)
variable (air_conditioned_not_rented_rooms : ℚ := (4/5) * not_rented_rooms)
variable (air_conditioned_rented_rooms : ℚ := air_conditioned_rooms - air_conditioned_not_rented_rooms)
variable (fraction_air_conditioned_rented : ℚ := air_conditioned_rented_rooms / air_conditioned_rooms)

theorem fraction_of_air_conditioned_rooms_rented :
  fraction_air_conditioned_rented = (2/3) := by
  sorry

end fraction_of_air_conditioned_rooms_rented_l10_10733


namespace mom_t_shirts_total_l10_10450

-- Definitions based on the conditions provided in the problem
def packages : ℕ := 71
def t_shirts_per_package : ℕ := 6

-- The statement to prove that the total number of white t-shirts is 426
theorem mom_t_shirts_total : packages * t_shirts_per_package = 426 := by sorry

end mom_t_shirts_total_l10_10450


namespace megan_average_speed_l10_10533

theorem megan_average_speed :
  ∃ s : ℕ, s = 100 / 3 ∧ ∃ (o₁ o₂ : ℕ), o₁ = 27472 ∧ o₂ = 27572 ∧ o₂ - o₁ = 100 :=
by
  sorry

end megan_average_speed_l10_10533


namespace cube_line_segment_length_l10_10284

theorem cube_line_segment_length (a : ℝ) (h : a > 0) :
  let midpoints_distance := (ℝ.sqrt 2 / 2) * a
  in ∀ (sphere_radius := a / 2), (ℝ.sqrt ((sphere_radius^2) - (midpoints_distance / 2)^2)) =
    ℝ.sqrt 2 / 2 * a :=
by sorry

end cube_line_segment_length_l10_10284


namespace christopher_age_l10_10584

variable (C G F : ℕ)

theorem christopher_age (h1 : G = C + 8) (h2 : F = C - 2) (h3 : C + G + F = 60) : C = 18 := by
  sorry

end christopher_age_l10_10584


namespace prove_rectangular_selection_l10_10564

def number_of_ways_to_choose_rectangular_region (horizontals verticals : ℕ) : ℕ :=
  (Finset.choose horizontals 2) * (Finset.choose verticals 2)

theorem prove_rectangular_selection :
  number_of_ways_to_choose_rectangular_region 5 5 = 100 :=
by
  sorry

end prove_rectangular_selection_l10_10564


namespace solve_equation1_solve_equation2_solve_equation3_l10_10358

-- For equation x^2 + 2x = 5
theorem solve_equation1 (x : ℝ) : x^2 + 2 * x = 5 ↔ (x = -1 + Real.sqrt 6) ∨ (x = -1 - Real.sqrt 6) :=
sorry

-- For equation x^2 - 2x - 1 = 0
theorem solve_equation2 (x : ℝ) : x^2 - 2 * x - 1 = 0 ↔ (x = 1 + Real.sqrt 2) ∨ (x = 1 - Real.sqrt 2) :=
sorry

-- For equation 2x^2 + 3x - 5 = 0
theorem solve_equation3 (x : ℝ) : 2 * x^2 + 3 * x - 5 = 0 ↔ (x = -5 / 2) ∨ (x = 1) :=
sorry

end solve_equation1_solve_equation2_solve_equation3_l10_10358


namespace find_range_of_m_l10_10215

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 / 2 then x + m
  else x - 1 + 1 / (x - 1) + 1

def is_triangle_function (f : ℝ → ℝ) (A : set ℝ) : Prop :=
  ∀ a b c ∈ A, ∃ x y z, (x, y, z) = (f a, f b, f c) ∧ x + y > z ∧ x + z > y ∧ y + z > x

theorem find_range_of_m (A : set ℝ) :
  (∀ x ∈ A, 0 ≤ x ∧ x ≤ 3 → f x ∈ [f 0, f 3]) →
  is_triangle_function (f x m) A →
  ∃ m : ℝ, m ∈ (7 / 4 : ℝ, 9 / 2 : ℝ) :=
sorry

end find_range_of_m_l10_10215


namespace walter_exceptional_days_l10_10124

theorem walter_exceptional_days :
  ∃ (w b : ℕ), 
  b + w = 10 ∧ 
  3 * b + 5 * w = 36 ∧ 
  w = 3 :=
by
  sorry

end walter_exceptional_days_l10_10124


namespace factorize_4a2_minus_9_factorize_2x2y_minus_8xy_plus_8y_l10_10905

-- Factorization of 4a^2 - 9 as (2a + 3)(2a - 3)
theorem factorize_4a2_minus_9 (a : ℝ) : 4 * a^2 - 9 = (2 * a + 3) * (2 * a - 3) :=
by 
  sorry

-- Factorization of 2x^2 y - 8xy + 8y as 2y(x-2)^2
theorem factorize_2x2y_minus_8xy_plus_8y (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2) ^ 2 :=
by 
  sorry

end factorize_4a2_minus_9_factorize_2x2y_minus_8xy_plus_8y_l10_10905


namespace evaluate_f_f_l10_10990

def f (x : ℝ) : ℝ :=
  if x > 0 then log 2 x
  else 3^x + 1

theorem evaluate_f_f (h : f (f (1 / 4)) = 10 / 9) : f (f (1 / 4)) = 10 / 9 :=
  by
    sorry

end evaluate_f_f_l10_10990


namespace inequality_proof_l10_10967

variables (x y : ℝ) (n : ℕ)

theorem inequality_proof (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  (x^n / (x + y^3) + y^n / (x^3 + y)) ≥ (2^(4-n) / 5) := by
  sorry

end inequality_proof_l10_10967


namespace oz_lost_words_count_l10_10814
-- We import the necessary library.

-- Define the context.
def total_letters := 69
def forbidden_letter := 7

-- Define function to calculate lost words when a specific letter is forbidden.
def lost_words (total_letters : ℕ) (forbidden_letter : ℕ) : ℕ :=
  let one_letter_lost := 1
  let two_letter_lost := 2 * (total_letters - 1)
  one_letter_lost + two_letter_lost

-- State the theorem.
theorem oz_lost_words_count :
  lost_words total_letters forbidden_letter = 139 :=
by
  sorry

end oz_lost_words_count_l10_10814


namespace smallest_n_integer_S_n_l10_10896

-- Definitions for K and conditions
def K : ℚ := (∑ i in Finset.range 10 \ {0}, (1 : ℚ) / i)

def S_n (n : ℕ) : ℚ := (2 * n * 10^(n-1)) * K + 1

-- The proof goal
theorem smallest_n_integer_S_n : ∃ n: ℕ, S_n n ∈ ℤ ∧ ∀ k: ℕ, k < n → S_n k ∉ ℤ :=
by sorry

end smallest_n_integer_S_n_l10_10896


namespace construct_orthocenter_l10_10094

-- Define the problem setup
variables {A B C O D G H : Point}
variables (circ_circle : Circle)

-- Conditions 
def acute_non_isosceles_triangle (A B C : Point) : Prop := 
  ∃ α β γ : Angle, 
    α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ 0 < α < π/2 ∧ 0 < β < π/2 ∧ 0 < γ < π/2

def midpoint (D : Point) (A B : Point) : Prop :=
  dist A D = dist D B ∧ A, D, B collinear

def circumscribed_circle (circ_circle : Circle) (A B C O : Point) : Prop :=
  circ_circle.center = O ∧ A, B, C on circ_circle

-- Prove the existence of orthocenter H at the intersection of lines AG and AD
theorem construct_orthocenter 
  (acute : acute_non_isosceles_triangle A B C)
  (circ : circumscribed_circle circ_circle A B C O)
  (midpt : midpoint D A B)
  : ∃ H : Point, intersection (line_from_to A G) (line_from_to A D) H := 
sorry -- Proof goes here

end construct_orthocenter_l10_10094


namespace taxi_speed_l10_10032

theorem taxi_speed (v : ℕ) (h₁ : v > 30) (h₂ : ∃ t₁ t₂ : ℕ, t₁ = 3 ∧ t₂ = 3 ∧ 
                    v * t₁ = (v - 30) * (t₁ + t₂)) : 
                    v = 60 :=
by
  sorry

end taxi_speed_l10_10032


namespace number_of_terms_in_simplified_expression_l10_10891

theorem number_of_terms_in_simplified_expression :
  let x y z : ℝ
  let monomials := { (a, b, c) ∈ ℕ × ℕ × ℕ | even a ∧ a + b + c = 2010 }
  (monomials.size) = 1006^2 := sorry

end number_of_terms_in_simplified_expression_l10_10891


namespace min_real_root_neg_p_l10_10983

theorem min_real_root_neg_p (p : ℝ) (h : ∃ r1 r2, (x - 19) * (x - 83) = p) : ∃ r, r = -19 :=
sorry

end min_real_root_neg_p_l10_10983


namespace range_of_ratio_theorem_l10_10600

noncomputable def range_of_ratio (a b : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧
  5 - 3 * a ≤ b ∧ b ≤ 4 - a ∧
  Real.log b ≥ a →
  e ≤ b / a ∧ b / a ≤ 7

theorem range_of_ratio_theorem (a b : ℝ) : range_of_ratio a b := 
by sorry

end range_of_ratio_theorem_l10_10600


namespace find_x_l10_10438

theorem find_x : 
  (5 * 12 / (180 / 3) = 1) → (∃ x : ℕ, 1 + x = 81 ∧ x = 80) :=
by
  sorry

end find_x_l10_10438


namespace chandler_needs_to_sell_more_rolls_l10_10938

/-- Chandler's wrapping paper selling condition. -/
def chandler_needs_to_sell : ℕ := 12

def sold_to_grandmother : ℕ := 3
def sold_to_uncle : ℕ := 4
def sold_to_neighbor : ℕ := 3

def total_sold : ℕ := sold_to_grandmother + sold_to_uncle + sold_to_neighbor

theorem chandler_needs_to_sell_more_rolls : chandler_needs_to_sell - total_sold = 2 :=
by
  sorry

end chandler_needs_to_sell_more_rolls_l10_10938


namespace cube_face_ratio_l10_10858

/-- Given a wooden rectangular prism with dimensions 4 by 5 by 6, painted green and cut into 1 by 1 by 1 cubes,
    the ratio of the number of cubes with exactly two green faces to the number of cubes with three green faces is 9:2. -/
theorem cube_face_ratio : 
  let rect_prism := {length := 4, width := 5, height := 6}  
  let total_cubes := 4 * 5 * 6  
  let two_faces_edge_count := 4 * (4 - 2) + 4 * (5 - 2) + 4 * (6 - 2)
  let three_faces_corner_count := 8  
  let ratio := two_faces_edge_count / three_faces_corner_count  
  ratio = 9 := 
by
  let rect_prism := {length := 4, width := 5, height := 6}
  let total_cubes := 4 * 5 * 6
  let two_faces_edge_count := 4 * (4 - 2) + 4 * (5 - 2) + 4 * (6 - 2)
  let three_faces_corner_count := 8
  let ratio := two_faces_edge_count / three_faces_corner_count
  have h : ratio = 9 := sorry
  exact h

end cube_face_ratio_l10_10858


namespace angle_A_calculation_side_a_calculation_l10_10282

theorem angle_A_calculation :
  ∀ (A B C a b c : ℝ), 
  (0 < A ∧ A < π) → 
  cos A = 1 / 2 → 
  A = π / 3 := 
by
  intros
  sorry

theorem side_a_calculation :
  ∀ (A B C a b c : ℝ), 
  (0 < A ∧ A < π) → 
  cos A = 1 / 2 → 
  b = 2 → 
  c = 3 → 
  a = sqrt 7 := 
by
  intros
  sorry

end angle_A_calculation_side_a_calculation_l10_10282


namespace percentage_of_students_with_same_grade_l10_10674

def num_students : ℕ := 40

def student_grades : matrix (fin 4) (fin 4) ℕ :=
  ![![4, 3, 2, 1],
    ![2, 7, 3, 1],
    ![2, 4, 6, 2],
    ![0, 1, 1, 3]]

def same_grade_students : ℕ :=
  student_grades[0, 0] + student_grades[1, 1] +
  student_grades[2, 2] + student_grades[3, 3]

def percentage_same_grade_students : ℕ :=
  (same_grade_students * 100) / num_students

theorem percentage_of_students_with_same_grade :
  percentage_same_grade_students = 50 := by
  sorry

end percentage_of_students_with_same_grade_l10_10674


namespace expansion_properties_l10_10624

noncomputable def exp_poly (x : ℝ) : ℝ := (x + 1 / (2 * sqrt x))^8

theorem expansion_properties (x : ℝ) (h : x > 0) :
  (n : ℕ) ∧ (coeff : ℝ) ∧ (n = 8) ∧ (coeff = 7/16) :=
begin
  let r := 6,
  have h_coeff : coeff = (nat.choose 8 r) * (1 / 2)^r := by sorry, 
  have h_power : 8 - (3/2) * r = -1 := by sorry,
  exact ⟨8, (nat.choose 8 6) * (1 / 2)^6, rfl, by norm_num; exact rfl⟩
end

end expansion_properties_l10_10624


namespace cos_alpha_values_l10_10652

theorem cos_alpha_values (α : ℝ) (h : Real.sin (π + α) = -3 / 5) :
  Real.cos α = 4 / 5 ∨ Real.cos α = -4 / 5 := 
sorry

end cos_alpha_values_l10_10652


namespace find_positive_real_numbers_l10_10914

open Real

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16

theorem find_positive_real_numbers (x : ℝ) (hx : x > 0) :
  satisfies_inequality x ↔ 15 * x^2 + 32 * x - 256 = 0 :=
sorry

end find_positive_real_numbers_l10_10914


namespace polynomial_roots_l10_10218

-- Lean 4 statement encapsulating the problem
theorem polynomial_roots :
  ∀ (a b c : ℝ), (a = -3 ∧ b = 1 ∧ c = -39) ↔ 
  (2 - 3i : ℂ) is_root_of (λ x, x^3 + (a:ℝ)x^2 + b*x - c) ∧
  (2 + 3i : ℂ) is_root_of (λ x, x^3 + (a:ℝ)x^2 + b*x - c) ∧
  (2:ℝ) is_root_of (λ x, x^3 + (a:ℝ)x^2 + b*x - c) ∧
  (a = -3 ∧ b = 1 ∧ c = -39) := 
begin 
  sorry -- Proof is not needed as per the instructions
end

end polynomial_roots_l10_10218


namespace cone_tetrahedron_edge_length_l10_10766

theorem cone_tetrahedron_edge_length :
  ∃ a : ℝ, 
    (∀ (cone_vertex cone_base_center : ℝ) (generatrix_length : ℝ), 
      cone_vertex = 1 ∧ generatrix_length = 1) → 
    (a ≈ 0.4768) :=
sorry

end cone_tetrahedron_edge_length_l10_10766


namespace proposition_analysis_l10_10737

theorem proposition_analysis :
  let prop1 := ∀ (a b c d : ℝ), a ≠ b → c ≠ d → a = c → b = d → (a + b = 180) → (c + d = 180)
  let prop2 := ∀ (l1 l2 t : Set ℝ), l1 ∥ l2 → relation (l1, t) = relation (l2, t)
  let prop3 := ∀ (a b : ℝ), a * b = 1 → (a = 1) ∨ (b = 1)
  let prop4 := ∀ (x : ℝ), x^2 = 4 → x = 2
  (prop1 = true) ∧ (prop2 = false) ∧ (prop3 = false) ∧ (prop4 = false) :=
by
  sorry

end proposition_analysis_l10_10737


namespace square_area_l10_10851

theorem square_area (side_length : ℝ) (h : side_length = 11) : side_length * side_length = 121 := 
by 
  simp [h]
  sorry

end square_area_l10_10851


namespace B_pow_four_l10_10315

open Matrix

variables {R : Type*} [CommRing R]

def B : Matrix (Fin 2) (Fin 2) R := ![![0, -1], ![-1, 0]]

def v : Fin 2 → R := ![7, 3]

theorem B_pow_four (hv : (B * (λ _, v)) = (λ _, ![-7, -3])) : 
  (B^4 * (λ _, v)) = (λ _, v) :=
sorry

end B_pow_four_l10_10315


namespace binom_50_2_eq_1225_l10_10108

theorem binom_50_2_eq_1225 : ∀ (n r : ℕ), n = 50 → r = 2 → Nat.choose n r = 1225 :=
by
  intros n r hn hr
  rw [hn, hr]
  iterate { rw Nat.choose }
  sorry

end binom_50_2_eq_1225_l10_10108


namespace project_funds_quadruple_l10_10838

noncomputable def time_to_quadruple_funds (initial_funds : ℕ) (annual_profit_rate : ℚ) (annual_withdrawal : ℕ) (final_multiplier : ℚ) : ℕ :=
  let diminisher := annual_withdrawal * 1 * annual_profit_rate^initial_funds
  Nat.find $ λ n => (annual_profit_rate+1)^n * initial_funds - diminisher*n ≥ final_multiplier * initial_funds

theorem project_funds_quadruple (initial_funds annual_withdrawal : ℕ) (annual_profit_rate final_multiplier : ℚ) (n : ℕ) (data : ℕ) :
    initial_funds = 1000 → annual_profit_rate = 0.25 → annual_withdrawal = 100 → final_multiplier = 4 → 
    data = 7 → time_to_quadruple_funds initial_funds annual_profit_rate annual_withdrawal final_multiplier = data :=
by
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end project_funds_quadruple_l10_10838


namespace remainder_sum_first_150_l10_10434

-- Definitions based on the conditions
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Lean statement equivalent to the mathematical problem
theorem remainder_sum_first_150 :
  (sum_first_n 150) % 11250 = 75 :=
by 
sorry

end remainder_sum_first_150_l10_10434


namespace parabola_line_intersection_l10_10632

variables {p x₀ : ℝ} (F : ℝ → ℝ → Prop) (M : ℝ → ℝ → Prop)
variables {A B C D : ℝ × ℝ} (l l' : (ℝ × ℝ) → (ℝ × ℝ) → Prop)

def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0
def point_on_parabola (x₀ : ℝ) : (ℝ × ℝ) := (x₀, 4)

def focus_of_parabola (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def distance (P Q : ℝ × ℝ) : ℝ := sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def line (m : ℝ) (P Q : ℝ × ℝ) : Prop := ∃ b, ∀ x y, (x, y) = P ∨ (x, y) = Q → x = m * y + b

def perpendicular_bisector (P Q : ℝ × ℝ) (l' : (ℝ × ℝ) → (ℝ × ℝ) → Prop) : Prop :=
∃ mid : ℝ × ℝ, mid = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) ∧ ∀ x y, l' (x, y) mid → x = -y

def orthogonal_vectors (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Main statement
theorem parabola_line_intersection {p x₀ : ℝ} (hx₀ : parabola p x₀ 4)
  (h_dist : distance (point_on_parabola x₀) (focus_of_parabola p) = (5 / 4) * x₀)
  (h_line : ∃ m, line m (focus_of_parabola p) (x₀, 4))
  (h_perp_bisect : line m (focus_of_parabola p) (x₀, 4) ∧ perpendicular_bisector (A, B) l' ∧ perpendicular_bisector (C, D) l')
  (h_orth : orthogonal_vectors (A, C) (A, D)) :
  m = 1 ∨ m = -1 :=
sorry

end parabola_line_intersection_l10_10632


namespace total_allocation_is_1800_l10_10681

-- Definitions from conditions.
def part_value (amount_food : ℕ) (ratio_food : ℕ) : ℕ :=
  amount_food / ratio_food

def total_parts (ratio_household : ℕ) (ratio_food : ℕ) (ratio_misc : ℕ) : ℕ :=
  ratio_household + ratio_food + ratio_misc

def total_amount (part_value : ℕ) (total_parts : ℕ) : ℕ :=
  part_value * total_parts

-- Given conditions
def ratio_household := 5
def ratio_food := 4
def ratio_misc := 1
def amount_food := 720

-- Prove the total allocation
theorem total_allocation_is_1800 
  (amount_food : ℕ := 720) 
  (ratio_household : ℕ := 5) 
  (ratio_food : ℕ := 4) 
  (ratio_misc : ℕ := 1) : 
  total_amount (part_value amount_food ratio_food) (total_parts ratio_household ratio_food ratio_misc) = 1800 :=
by
  sorry

end total_allocation_is_1800_l10_10681


namespace net_deflection_l10_10820

theorem net_deflection(
  (m₁ : ℝ) (m₁_val : m₁ = 78.75),
  (x₁ : ℝ) (x₁_val : x₁ = 1),
  (h₁ : ℝ) (h₁_val : h₁ = 15),
  (m₂ : ℝ) (m₂_val : m₂ = 45),
  (h₂ : ℝ) (h₂_val : h₂ = 29)
) : ∃ x₂ : ℝ, x₂ ≈ 1.04 :=
by {
  sorry
}

end net_deflection_l10_10820


namespace paul_prays_more_than_bruce_l10_10343

-- Conditions as definitions in Lean 4
def prayers_per_day_paul := 20
def prayers_per_sunday_paul := 2 * prayers_per_day_paul
def prayers_per_day_bruce := prayers_per_day_paul / 2
def prayers_per_sunday_bruce := 2 * prayers_per_sunday_paul

def weekly_prayers_paul := 6 * prayers_per_day_paul + prayers_per_sunday_paul
def weekly_prayers_bruce := 6 * prayers_per_day_bruce + prayers_per_sunday_bruce

-- Statement of the proof problem
theorem paul_prays_more_than_bruce :
  (weekly_prayers_paul - weekly_prayers_bruce) = 20 := by
  sorry

end paul_prays_more_than_bruce_l10_10343


namespace ten_points_in_ring_l10_10259

theorem ten_points_in_ring (r : ℝ) (points : set (ℝ × ℝ)) (h : ∀ p ∈ points, p.1^2 + p.2^2 ≤ r^2) 
    (rad_16 : r = 16) (num_points : points.card = 650) : 
    ∃ (c : ℝ × ℝ), 2 ≤ dist c 0 ∧ dist c 0 ≤ 3 ∧ (points.filter (λ p, 2 ≤ dist p c ∧ dist p c ≤ 3)).card ≥ 10 :=
by 
    sorry

end ten_points_in_ring_l10_10259


namespace part1_result_part2_result_l10_10224

-- Part 1
theorem part1_result (a b : ℕ) (h_a : a = 1) (h_b : b = 2) : 
  let c := a * b + a + b in
  let c' := b * c + b + c in
  c' = 17 :=
by {
  subst h_a,
  subst h_b,
  let c := 1 * 2 + 1 + 2,
  let c' := 2 * c + 2 + c,
  have : c = 5, by norm_num,
  rw this,
  have : c' = 17, by norm_num,
  exact this
}

-- Part 2
theorem part2_result (p q : ℕ) (h_p : p > q) (h_q : q > 0)
  (final_result : (q + 1) ^ 8 * (p + 1) ^ 5 - 1 = ((q + 1) ^ 8) * ((p + 1) ^ 5) - 1) :
  8 + 5 = 13 :=
by {
  norm_num
}

end part1_result_part2_result_l10_10224


namespace rectangle_count_l10_10561

theorem rectangle_count (h_lines v_lines : Finset ℕ) (h_card : h_lines.card = 5) (v_card : v_lines.card = 5) :
  ∃ (n : ℕ), n = (h_lines.choose 2).card * (v_lines.choose 2).card ∧ n = 100 :=
by
  sorry 

end rectangle_count_l10_10561


namespace triangle_structure_twelve_rows_l10_10498

theorem triangle_structure_twelve_rows :
  let rods n := 3 * n * (n + 1) / 2
  let connectors n := n * (n + 1) / 2
  rods 12 + connectors 13 = 325 :=
by
  let rods n := 3 * n * (n + 1) / 2
  let connectors n := n * (n + 1) / 2
  sorry

end triangle_structure_twelve_rows_l10_10498


namespace problem1_problem2_l10_10827

section Problem1

theorem problem1 :
  2^(-1) + (-2024)^0 - Real.sqrt ((-1 / 2) ^ 2) = 1 := by
  sorry

end Problem1

section Problem2

theorem problem2 : (104 * 96) = 9984 := by
  let a := 100
  let b := 4
  have h1 : 104 = a + b := by
    rw [←nat.add_assoc, add_comm]
  have h2 : 96 = a - b := by
    sorry
  calc
  104 * 96 = (a + b) * (a - b) : by
    rw [h1, h2]
  ... = a^2 - b^2 : by
    apply_nat_pow
    rw [mul_sub, add_sub_add_right_eq_sub, pow_two, pow_two, sub_sub]
  ... = 100^2 - 4^2 : by
    rfl
  ... = 10000 - 16 : by
    norm_num
  ... = 9984 : by
    norm_num

end Problem2

end problem1_problem2_l10_10827


namespace sticks_per_pack_is_3_l10_10088

theorem sticks_per_pack_is_3 :
  (∀ (packs_per_carton cartons_per_box : ℕ), packs_per_carton = 5 ∧ cartons_per_box = 4) →
  (∀ (sticks_total_boxes sticks_total : ℕ), sticks_total_boxes = 8 ∧ sticks_total = 480) →
  (∃ (sticks_per_pack : ℕ), sticks_per_pack = 3) :=
by
  intros packs_per_carton cartons_per_box h1
  intros sticks_total_boxes sticks_total h2
  rw h2.1 at h2,
  rw h2.2 at h2,
  rw h1.1 at h1,
  rw h1.2 at h1,
  have h3: 60 = sticks_total / sticks_total_boxes, from divide_eq_of_eq_mul (symm (calc
    sticks_total = 480 : by rfl
    480 = 8 * 60 : by {
      norm_num }
    )),
  have h4: 15 = 60 / cartons_per_box, from divide_eq_of_eq_mul (symm (calc
    60 = 4 * 15 : by {
      norm_num,
    })),
  have h5: 3 = 15 / packs_per_carton, from divide_eq_of_eq_mul (symm (calc
    15 = 5 * 3 : by {
      norm_num,
    })),
  use 3,
  norm_num,
  exact h5,
sorry

end sticks_per_pack_is_3_l10_10088


namespace r_at_6_l10_10311

-- Define the monic quintic polynomial r(x) with given conditions
def r (x : ℝ) : ℝ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + x^2 + 2 

-- Given conditions
axiom r_1 : r 1 = 3
axiom r_2 : r 2 = 7
axiom r_3 : r 3 = 13
axiom r_4 : r 4 = 21
axiom r_5 : r 5 = 31

-- Proof goal
theorem r_at_6 : r 6 = 158 :=
by
  sorry

end r_at_6_l10_10311


namespace right_triangle_area_l10_10813

variable {AB BC AC : ℕ}

theorem right_triangle_area : ∀ (AB BC AC : ℕ), (AC = 50) → (AB + BC = 70) → (AB^2 + BC^2 = AC^2) → (1 / 2) * AB * BC = 300 :=
by
  intros AB BC AC h1 h2 h3
  -- Proof steps will be added here
  sorry

end right_triangle_area_l10_10813


namespace unfolded_side_view_of_cone_is_sector_l10_10395

theorem unfolded_side_view_of_cone_is_sector 
  (shape : Type)
  (curved_side : shape)
  (straight_side1 : shape)
  (straight_side2 : shape) 
  (condition1 : ∃ (s : shape), s = curved_side) 
  (condition2 : ∃ (s1 s2 : shape), s1 = straight_side1 ∧ s2 = straight_side2)
  : shape = sector :=
sorry

end unfolded_side_view_of_cone_is_sector_l10_10395


namespace probability_all_digits_different_l10_10509

def integers_in_range := {x : ℕ | 100 ≤ x ∧ x ≤ 999}

def count_elements (s : set ℕ) : ℕ :=
  fintype.card {x // x ∈ s}

def has_same_digits (n : ℕ) : Prop :=
  let d0 := n % 10 in
  let d1 := (n / 10) % 10 in
  let d2 := (n / 100) % 10 in
  (d0 = d1) ∨ (d1 = d2) ∨ (d0 = d2)

theorem probability_all_digits_different :
  ∀ x ∈ integers_in_range, 
  (∃ n, n = fintype.card {k // k ∈ integers_in_range ∧ ¬ has_same_digits k}) →
  n.to_float / (count_elements integers_in_range).to_float = 0.99 :=
by 
  sorry

end probability_all_digits_different_l10_10509


namespace max_number_of_terms_equal_l10_10130

noncomputable def maxSummands (n : ℕ) : ℕ :=
  ⌊((Real.sqrt (8 * n + 1) - 1) / 2)⌋₊

theorem max_number_of_terms_equal (n : ℕ) (h : n > 2) :
  maxSummands n = ⌊((Real.sqrt (8 * n + 1) - 1) / 2)⌋₊ :=
by
  sorry

end max_number_of_terms_equal_l10_10130


namespace count_positive_integers_l10_10234

open BigOperators

noncomputable def num_valid_n : ℕ :=
  let fact_7 := 7.fact
  let fact_14 := 14.fact in
  have lcm_gcd_condition : ∀ (n : ℕ), 
    n % 7 = 0 → 
    Nat.lcm fact_7 n = 7 * Nat.gcd fact_14 n, 
    from sorry,
  have count_valid_n : count {n : ℕ | n % 7 = 0 ∧ Nat.lcm fact_7 n = 7 * Nat.gcd fact_14 n} = 192, 
    from sorry,
  192

theorem count_positive_integers (h : true) : num_valid_n = 192 :=
by {
  exact rfl
}

end count_positive_integers_l10_10234


namespace standard_deviation_of_applicants_l10_10765

theorem standard_deviation_of_applicants (σ : ℕ) 
  (h1 : ∃ avg : ℕ, avg = 30)
  (h2 : ∃ n : ℕ, n = 17)
  (h3 : ∃ range_count : ℕ, range_count = (30 + σ) - (30 - σ) + 1) :
  σ = 8 :=
by
  sorry

end standard_deviation_of_applicants_l10_10765


namespace total_tiles_count_l10_10022

theorem total_tiles_count (n total_tiles: ℕ) 
  (h1: total_tiles - n^2 = 36) 
  (h2: total_tiles - (n + 1)^2 = 3) : total_tiles = 292 :=
by {
  sorry
}

end total_tiles_count_l10_10022


namespace unit_digit_expression_l10_10788

theorem unit_digit_expression : 
  let expr := 3 * (2^2 + 1) * (2^4 + 1) * ... * (2^32 + 1) + 1 in 
  expr % 10 = 6 := 
by
  let expr := 3 * (2^2 + 1) * (2^4 + 1) * ... * (2^32 + 1) + 1
  have expr := 3 * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) + 1
  sorry

end unit_digit_expression_l10_10788


namespace smallest_positive_solution_tan_sec_5x_l10_10143

theorem smallest_positive_solution_tan_sec_5x:
  ∃ x : ℝ, 0 < x ∧ tan (4 * x) + tan (5 * x) = sec (5 * x) ∧ x = π / 18 := 
sorry

end smallest_positive_solution_tan_sec_5x_l10_10143


namespace min_parts_square_divided_into_acute_triangles_l10_10651

theorem min_parts_square_divided_into_acute_triangles :
  ∀ (S : Set (Set Point)), is_square S →
  (∀ T ∈ S, is_triangle T ∧ is_acute_triangle T) →
  ∃ (n : ℕ), (card S = n) ∧ (n ≥ 8) :=
by
  sorry

end min_parts_square_divided_into_acute_triangles_l10_10651


namespace river_water_volume_l10_10487

/-- Given conditions:
- depth of the river is 2 meters
- width of the river is 45 meters
- flow rate of the river is 2 kilometers per hour

Prove that the volume of water running into the sea per minute is approximately 2999.7 cubic meters.
-/
theorem river_water_volume (depth width flow_rate_in_kmph : ℝ) (h_depth : depth = 2)
  (h_width : width = 45) (h_flow_rate_in_kmph : flow_rate_in_kmph = 2) :
  let area := depth * width in
  let flow_rate_in_m_per_min := flow_rate_in_kmph * 1000 / 60 in
  let volume_per_min := area * flow_rate_in_m_per_min in
  volume_per_min ≈ 2999.7 :=
by
  sorry

end river_water_volume_l10_10487


namespace quadratic_even_min_solve_inequality_max_value_k_l10_10219

section
variables {f g h : ℝ → ℝ}

def quadratic_fn (m n : ℝ) := λ x : ℝ, x^2 + m * x + n  -- Definition of a quadratic function

-- 1. Prove that for the quadratic function f(x) = x^2 + mx + n which is even and has a minimum value of 1, it follows that f(x) = x^2 + 1
theorem quadratic_even_min (m n : ℝ) (h_even : ∀ x, f (-x) = f x) (h_min : ∀ x, f x ≥ 1) :
  f = quadratic_fn 0 1 := sorry

-- 2. Given f(x) = x^2 + 1, prove the set {x | g(2^x) > 2^x} = {x | x < (1 / 2) * log 2 5} where g(x) = (6x / f(x))
theorem solve_inequality (h_f : f = quadratic_fn 0 1) :
  {x | g (2 ^ x) > 2 ^ x} = {x | x < (1 / 2) * log 2 5} := sorry

-- 3. For h(x) = |f(x)| over x ∈ [-1, 1] and f(x) = x^2 + mx + n, prove that the maximum value k such that M ≥ k is k = 1 / 2
theorem max_value_k (m n : ℝ) :
  (∀ x ∈ Icc (-1 : ℝ) 1, |f x| ≤ k) → (∃ k, k = 1 / 2) := sorry
end

end quadratic_even_min_solve_inequality_max_value_k_l10_10219


namespace final_stamp_collection_l10_10865

section StampCollection

structure Collection :=
  (nature : ℕ)
  (architecture : ℕ)
  (animals : ℕ)
  (vehicles : ℕ)
  (famous_people : ℕ)

def initial_collections : Collection := {
  nature := 10, architecture := 15, animals := 12, vehicles := 6, famous_people := 4
}

-- define transactions as functions that take a collection and return a modified collection
def transaction1 (c : Collection) : Collection :=
  { c with nature := c.nature + 4, architecture := c.architecture + 5, animals := c.animals + 5, vehicles := c.vehicles + 2, famous_people := c.famous_people + 1 }

def transaction2 (c : Collection) : Collection := 
  { c with nature := c.nature + 2, animals := c.animals - 1 }

def transaction3 (c : Collection) : Collection := 
  { c with animals := c.animals - 5, architecture := c.architecture + 3 }

def transaction4 (c : Collection) : Collection :=
  { c with animals := c.animals - 4, nature := c.nature + 7 }

def transaction7 (c : Collection) : Collection :=
  { c with vehicles := c.vehicles - 2, nature := c.nature + 5 }

def transaction8 (c : Collection) : Collection :=
  { c with vehicles := c.vehicles + 3, famous_people := c.famous_people - 3 }

def final_collection (c : Collection) : Collection :=
  transaction8 (transaction7 (transaction4 (transaction3 (transaction2 (transaction1 c)))))

theorem final_stamp_collection :
  final_collection initial_collections = { nature := 28, architecture := 23, animals := 7, vehicles := 9, famous_people := 2 } :=
by
  -- skip the proof
  sorry

end StampCollection

end final_stamp_collection_l10_10865


namespace convert_degrees_to_radians_l10_10894

theorem convert_degrees_to_radians : 
  (-390) * (Real.pi / 180) = - (13 * Real.pi / 6) := 
by 
  sorry

end convert_degrees_to_radians_l10_10894


namespace chemistry_marks_more_than_physics_l10_10786

theorem chemistry_marks_more_than_physics (M P C x : ℕ) 
  (h1 : M + P = 32) 
  (h2 : (M + C) / 2 = 26) 
  (h3 : C = P + x) : 
  x = 20 := 
by
  sorry

end chemistry_marks_more_than_physics_l10_10786


namespace f_25_div_11_lt_0_l10_10712

open scoped Classical

noncomputable def f (x : ℚ) : ℝ := sorry

axiom H1 : ∀ a b : ℚ, 0 < a → 0 < b → f(a * b) = f(a) + f(b)
axiom H2 : ∀ p : ℕ, Nat.Prime p → f(p) = (p : ℚ)

theorem f_25_div_11_lt_0 : f(25 / 11) < 0 :=
sorry

end f_25_div_11_lt_0_l10_10712


namespace indigo_restaurant_average_rating_l10_10762

theorem indigo_restaurant_average_rating :
  let n_5stars := 6
  let n_4stars := 7
  let n_3stars := 4
  let n_2stars := 1
  let total_reviews := 18
  let total_stars := n_5stars * 5 + n_4stars * 4 + n_3stars * 3 + n_2stars * 2
  (total_stars / total_reviews : ℝ) = 4 :=
by
  sorry

end indigo_restaurant_average_rating_l10_10762


namespace percent_gain_on_transaction_l10_10064

theorem percent_gain_on_transaction :
  ∀ (x : ℝ), (850 : ℝ) * x + (50 : ℝ) * (1.10 * ((850 : ℝ) * x / 800)) = 850 * x * (1 + 0.06875) := 
by
  intro x
  sorry

end percent_gain_on_transaction_l10_10064


namespace integer_solutions_2_pow_n_minus_1_eq_x_pow_m_l10_10908

open Nat

theorem integer_solutions_2_pow_n_minus_1_eq_x_pow_m (x : ℤ) (m n : ℕ) : 
  (∃ x, (n = 1 ∧ (even m ∧ x = 1) ∨ (odd m ∧ x = 1))
  ∨ (m = 1 ∧ x = 2^n - 1) 
  ∨ (n > 1 ∧ m > 1 ∧ False)).to_bool ↔
  (2^n - 1 = x^m) := sorry

end integer_solutions_2_pow_n_minus_1_eq_x_pow_m_l10_10908


namespace right_triangle_construction_condition_l10_10113

theorem right_triangle_construction_condition (A B C : Point) (b d : ℝ) :
  AC = b → AC + BC - AB = d → b > d :=
by
  intro h1 h2
  sorry

end right_triangle_construction_condition_l10_10113


namespace number_of_sides_l10_10661

theorem number_of_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
by {
  sorry
}

end number_of_sides_l10_10661


namespace value_of_k_l10_10006

theorem value_of_k (k : ℝ) : 
  (∃ P Q R : ℝ × ℝ, P = (5, 12) ∧ Q = (0, k) ∧ dist (0, 0) P = dist (0, 0) Q + 5) → 
  k = 8 := 
by
  sorry

end value_of_k_l10_10006


namespace min_value_of_sum_of_reciprocals_l10_10171

theorem min_value_of_sum_of_reciprocals 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : Real.log (1 / a + 1 / b) / Real.log 4 = Real.log (1 / Real.sqrt (a * b)) / Real.log 2) : 
  1 / a + 1 / b ≥ 4 := 
by 
  sorry

end min_value_of_sum_of_reciprocals_l10_10171


namespace calculate_sum_l10_10464

def edge_length : ℝ := 3
def PM : ℝ := 1
def MS : ℝ := 2

def volume_of_piece : ℝ := (3 * (27 / 10))
def icing_area_triangle : ℝ := (27 / 10)
def icing_area_vertical : ℝ := (3 * 3)
def icing_area_total : ℝ := (icing_area_triangle + icing_area_vertical)

def total_sum : ℝ := (volume_of_piece + icing_area_total)

theorem calculate_sum :
  total_sum = 19.8 :=
by
  sorry

end calculate_sum_l10_10464


namespace large_ball_radius_l10_10053

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem large_ball_radius :
  let small_ball_radius := 0.5
      num_small_balls := 12
      total_volume := num_small_balls * volume_of_sphere small_ball_radius
      large_ball_radius := (total_volume * 3 / (4 * Real.pi)) ^ (1 / 3)
  in large_ball_radius = (3 / 2) ^ (1 / 3) :=
by
  sorry

end large_ball_radius_l10_10053


namespace days_without_leak_l10_10083

-- Definitions based on the problem conditions
variables {C V : ℝ}
def condition_1 := C = 60 * (V + 10)
def condition_2 := C = 48 * (V + 20)

-- Statement to be proven
theorem days_without_leak (C V : ℝ) (h1 : C = 60 * (V + 10)) (h2 : C = 48 * (V + 20)) : C / V = 80 :=
begin
  sorry
end

end days_without_leak_l10_10083


namespace average_star_rating_is_four_l10_10757

-- Define the conditions
def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

-- Define total star points as per the conditions
def total_star_points : ℕ := (5 * five_star_reviews) + (4 * four_star_reviews) + (3 * three_star_reviews) + (2 * two_star_reviews)

-- Define the average rating calculation
def average_rating : ℚ := total_star_points / total_reviews

theorem average_star_rating_is_four : average_rating = 4 := 
by {
  -- Placeholder for the proof
  sorry
}

end average_star_rating_is_four_l10_10757


namespace count_groups_U_AB_l10_10992

open Finset

theorem count_groups_U_AB :
  let U := ({1, 2, 3, 4, 5} : Finset ℕ) in
  let mutually_exclusive (A B : Finset ℕ) : Prop := A ∩ B = ∅ ∧ A.nonempty ∧ B.nonempty in
  let groups_U (A B : Finset ℕ) := mutually_exclusive A B ∧ U(A, B) ≠ U(B, A) in
  (card {p : Finset ℕ × Finset ℕ // groups_U p.1 p.2}) = 180 :=
by {sorry}

end count_groups_U_AB_l10_10992


namespace cindy_correct_result_l10_10106

theorem cindy_correct_result (x : ℕ) : 
  (x - 7) / 5 = 51 → ((x - 5) / 7 = 36) :=
by
  assume h: (x - 7) / 5 = 51
  sorry

end cindy_correct_result_l10_10106


namespace perpendicular_bisector_of_AB_l10_10923

noncomputable def circle : set (ℝ × ℝ) :=
{ p | p.1^2 + p.2^2 + 4 * p.2 = 0 }

noncomputable def line : set (ℝ × ℝ) :=
{ p | 3 * p.1 + 4 * p.2 + 2 = 0 }

noncomputable def center_of_circle : ℝ × ℝ :=
(0, -2)

theorem perpendicular_bisector_of_AB :
  let bisector_eq := λ x y : ℝ, 4 * x - 3 * y - 6 = 0 in
  ∃ x y : ℝ, bisector_eq x y ∧
    ∀ p ∈ circle, ∀ q ∈ line, 3 * x + 4 * y + 2 = 0 ∧ p.1^2 + p.2^2 + 4 * p.2 = 0 ∧
      line = line ∧
      circle = circle → 4 * (0:ℝ) - 3 * (-2:ℝ) - 6 = 0 :=
by
  sorry

end perpendicular_bisector_of_AB_l10_10923


namespace smallest_positive_solution_tan_sec_eq_l10_10154

theorem smallest_positive_solution_tan_sec_eq 
  (x : ℝ) 
  (hx : x > 0)
  (hx_rad : ∃ y : ℝ, x = y * real.pi) 
  (h_eq : real.tan (4 * x) + real.tan (5 * x) = real.sec (5 * x)) :
  x = real.pi / 18 :=
sorry

end smallest_positive_solution_tan_sec_eq_l10_10154


namespace rectangle_count_l10_10559

theorem rectangle_count (h_lines v_lines : Finset ℕ) (h_card : h_lines.card = 5) (v_card : v_lines.card = 5) :
  ∃ (n : ℕ), n = (h_lines.choose 2).card * (v_lines.choose 2).card ∧ n = 100 :=
by
  sorry 

end rectangle_count_l10_10559


namespace basketball_competition_l10_10122

theorem basketball_competition:
  (∃ x : ℕ, (0 ≤ x) ∧ (x ≤ 12) ∧ (3 * x - (12 - x) ≥ 28)) := by
  sorry

end basketball_competition_l10_10122


namespace display_exceed_250_on_third_press_l10_10833

theorem display_exceed_250_on_third_press (x : ℕ) : 
  x = (iterate (λ n, n^2 + 3) 3 1) → x ≥ 250 := 
by 
  sorry

end display_exceed_250_on_third_press_l10_10833


namespace rectangle_enclosure_l10_10547
open BigOperators

theorem rectangle_enclosure (n m : ℕ) (hn : n = 5) (hm : m = 5) : 
  (∑ i in finset.range n, ∑ j in finset.range i, 1) * 
  (∑ k in finset.range m, ∑ l in finset.range k, 1) = 100 := by
  sorry

end rectangle_enclosure_l10_10547


namespace largest_multiple_of_seven_smaller_than_neg_85_l10_10429

theorem largest_multiple_of_seven_smaller_than_neg_85 
  : ∃ k : ℤ, (k * 7 < -85) ∧ (∀ m : ℤ, (m * 7 < -85) → (m * 7 ≤ k * 7)) ∧ (k = -13) 
  := sorry

end largest_multiple_of_seven_smaller_than_neg_85_l10_10429


namespace lines_perpendicular_l10_10318

theorem lines_perpendicular (p b : ℝ) (hp : 0 < p):
  let line_eq := λ x : ℝ, sqrt(3)*x + b,
  let parab_eq := λ x y : ℝ, y^2 = 2*p*x,
  let points_intersect := ∃ A B C D : ℝ × ℝ, 
    (parab_eq A.1 A.2) ∧ (line_eq A.1 = A.2) ∧
    (parab_eq B.1 B.2) ∧ (line_eq B.1 = B.2) ∧ 
    (parab_eq C.1 C.2) ∧ (parab_eq D.1 D.2),
  ∃ angle : ℝ, points_intersect → angle = 90 :=
begin
  sorry
end

end lines_perpendicular_l10_10318


namespace sin_cos_identity_l10_10104

theorem sin_cos_identity :
  sin (21 * π / 180) * cos (9 * π / 180) + sin (69 * π / 180) * sin (9 * π / 180) = 1 / 2 := 
by
  sorry

end sin_cos_identity_l10_10104


namespace even_function_a_neg_one_l10_10654

theorem even_function_a_neg_one {f : ℝ → ℝ} {a : ℝ} 
  (h₁ : ∀ x, f(x) = (x - 1) * (x - a))
  (h₂ : ∀ x, f(x) = f(-x)) : a = -1 :=
by
  sorry

end even_function_a_neg_one_l10_10654


namespace find_a_l10_10985

theorem find_a
  (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = 3 * Real.sin (2 * x - Real.pi / 3))
  (a : ℝ)
  (h₂ : 0 < a)
  (h₃ : a < Real.pi / 2)
  (h₄ : ∀ x, f (x + a) = f (-x + a)) :
  a = 5 * Real.pi / 12 :=
sorry

end find_a_l10_10985


namespace tan_value_area_of_triangle_l10_10685

-- Define the properties of the triangle and the given conditions
variables {A B C a b c : ℝ}
axiom h1 : A = Real.pi / 4
axiom h2 : c / b = 3 * Real.sqrt 2 / 7

noncomputable def tan_C := (3 : ℝ) / 4

theorem tan_value 
  (A = Real.pi / 4) 
  (c / b = 3 * Real.sqrt 2 / 7) :
  Real.tan C = (3 : ℝ) / 4 :=
sorry

axiom h3 : a = 5

noncomputable def area_ABC := (21 : ℝ) / 2

theorem area_of_triangle 
  (a = 5)
  (A = Real.pi / 4) 
  (c / b = 3 * Real.sqrt 2 / 7) :
  1/2 * b * c * Real.sin A = (21 : ℝ) / 2 :=
sorry

end tan_value_area_of_triangle_l10_10685


namespace drawings_in_five_pages_l10_10691

theorem drawings_in_five_pages :
  let a₁ := 5
  let a₂ := 2 * a₁
  let a₃ := 2 * a₂
  let a₄ := 2 * a₃
  let a₅ := 2 * a₄
  a₁ + a₂ + a₃ + a₄ + a₅ = 155 :=
by
  let a₁ := 5
  let a₂ := 2 * a₁
  let a₃ := 2 * a₂
  let a₄ := 2 * a₃
  let a₅ := 2 * a₄
  sorry

end drawings_in_five_pages_l10_10691


namespace non_negative_solutions_l10_10385

theorem non_negative_solutions (x : ℕ) (h : 1 + x ≥ 2 * x - 1) : x = 0 ∨ x = 1 ∨ x = 2 := 
by {
  sorry
}

end non_negative_solutions_l10_10385


namespace rational_cos_identity_l10_10194

theorem rational_cos_identity (a : ℚ) (h1 : 0 < a ∧ a < 1) 
  (h2 : Real.cos (3 * Real.pi * a) + 2 * Real.cos (2 * Real.pi * a) = 0) : 
  a = ²3 := 
sorry

end rational_cos_identity_l10_10194


namespace Panikovsky_share_l10_10170

theorem Panikovsky_share :
  ∀ (horns hooves weight : ℕ) 
    (k δ : ℝ),
    horns = 17 →
    hooves = 2 →
    weight = 1 →
    (∀ h, h = k + δ) →
    (∀ wt, wt = k + 2 * δ) →
    (20 * k + 19 * δ) / 2 = 10 * k + 9.5 * δ →
    9 * k + 7.5 * δ = (9 * (k + δ) + 2 * k) →
    ∃ (Panikov_hearts Panikov_hooves : ℕ), 
    Panikov_hearts = 9 ∧ Panikov_hooves = 2 := 
by
  intros
  sorry

end Panikovsky_share_l10_10170


namespace floor_length_l10_10484

theorem floor_length (width length : ℕ) 
  (cost_per_square total_cost : ℕ)
  (square_side : ℕ)
  (h1 : width = 64) 
  (h2 : square_side = 8)
  (h3 : cost_per_square = 24)
  (h4 : total_cost = 576) 
  : length = 24 :=
by
  -- Placeholder for the proof, using sorry
  sorry

end floor_length_l10_10484


namespace chandler_needs_to_sell_more_rolls_l10_10939

/-- Chandler's wrapping paper selling condition. -/
def chandler_needs_to_sell : ℕ := 12

def sold_to_grandmother : ℕ := 3
def sold_to_uncle : ℕ := 4
def sold_to_neighbor : ℕ := 3

def total_sold : ℕ := sold_to_grandmother + sold_to_uncle + sold_to_neighbor

theorem chandler_needs_to_sell_more_rolls : chandler_needs_to_sell - total_sold = 2 :=
by
  sorry

end chandler_needs_to_sell_more_rolls_l10_10939


namespace number_of_valid_three_digit_even_numbers_l10_10997

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l10_10997


namespace eggs_per_basket_l10_10290

theorem eggs_per_basket (r y n : ℕ) (h_r : r = 30) (h_y : y = 42) 
  (h_same : ∃ k₁ k₂, r = k₁ * n ∧ y = k₂ * n) (h_geq : n ≥ 5)
  (h_div_r : r % n = 0) (h_div_y : y % n = 0) : n = 6 :=
begin
  sorry
end

end eggs_per_basket_l10_10290


namespace find_hyperbola_equation_l10_10376

-- Defining the focal length condition
def focal_length (a b : ℝ) : Prop := 2 * (sqrt (a^2 + b^2)) = 4

-- Defining the tangent condition
def tangent_to_circle (a b : ℝ) : Prop := ∀ x y : ℝ, (x + 2)^2 + y^2 = 1 → (x * b + y * a = 0) ∧ (x * b - y * a = 0)

-- Given values
def a : ℝ := sqrt 3
def b : ℝ := 1

-- The final equation we need to prove
def equation_of_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 3) - y^2 = 1

theorem find_hyperbola_equation : 
  focal_length a b ∧ tangent_to_circle a b → equation_of_hyperbola x y := 
by
  sorry

end find_hyperbola_equation_l10_10376


namespace winning_percentage_in_August_l10_10667

def total_matches : ℕ := 120
def total_wins_after_streak : ℕ := 75
def winning_average_after_streak : ℝ := 52 / 100
def matches_before_August (T : ℝ) : Prop := 
  (total_wins_after_streak:ℝ) = winning_average_after_streak * T
def wins_in_August (W : ℕ) (T : ℝ) : Prop := 
  W = total_wins_after_streak - (T - total_matches)

theorem winning_percentage_in_August 
  (T : ℝ) (W : ℕ)
  (hT : matches_before_August T)
  (hW : wins_in_August W T) :
  (W : ℝ) / total_matches * 100 = 42.5 :=
by
  sorry

end winning_percentage_in_August_l10_10667


namespace symmetric_center_cos_translation_l10_10378

theorem symmetric_center_cos_translation (x : ℝ) :
  ∃ k : ℤ, y = cos (2*x - π/4) ∧ x = k*π/2 + 3*π/8 :=
begin
  sorry
end

end symmetric_center_cos_translation_l10_10378


namespace ratio_SP2_to_SP1_l10_10507

noncomputable theory
open Real

def CP := 100.0
def SP1 := CP * 1.32
def SP2 := CP * 0.88

theorem ratio_SP2_to_SP1 : SP2 / SP1 = 2 / 3 :=
by
  let cp_val : ℝ := CP
  sorry

end ratio_SP2_to_SP1_l10_10507


namespace leona_earnings_5_hour_shift_l10_10393

theorem leona_earnings_5_hour_shift :
  ∀ (r : ℝ), (24.75 = 3 * r) ∧ (49.50 = 6 * r) → (5 * r = 41.25) :=
by
  intro r
  intro h
  cases h with h1 h2
  sorry

end leona_earnings_5_hour_shift_l10_10393


namespace minimum_stamps_to_make_47_cents_l10_10869

theorem minimum_stamps_to_make_47_cents (c f : ℕ) (h : 5 * c + 7 * f = 47) : c + f = 7 :=
sorry

end minimum_stamps_to_make_47_cents_l10_10869


namespace carson_gold_stars_l10_10520

theorem carson_gold_stars (gold_stars_yesterday gold_stars_today : ℕ) (h1 : gold_stars_yesterday = 6) (h2 : gold_stars_today = 9) : 
  gold_stars_yesterday + gold_stars_today = 15 := 
by
  sorry

end carson_gold_stars_l10_10520


namespace difference_of_roots_l10_10892

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 4 * x - 12 = 0

-- Define the difference of the roots
def roots_diff (a b c : ℝ) : ℝ := 
  let r1 := (b + sqrt (b^2 - 4 * a * c)) / (2 * a)
  let r2 := (b - sqrt (b^2 - 4 * a * c)) / (2 * a)
  r1 - r2

-- The statement to prove
theorem difference_of_roots : roots_diff 1 (-4) (-12) = 8 := by
  sorry

end difference_of_roots_l10_10892


namespace parallel_vectors_l10_10641

theorem parallel_vectors (λ μ : ℝ) :
  let a := (λ + 1, 0, 2 * λ)
  let b := (6, 2 * μ - 1, 2)
  λ = 1/5 ∧ μ = 1/2 → a.1 * b.2 = a.3 * b.1 ∧ a.3 * (2 * μ - 1) = 0 :=
by
  intros a b
  sorry

end parallel_vectors_l10_10641


namespace vasily_salary_is_higher_l10_10408

variables (total_students graduated_students fyodor_initial_salary fyodor_yearly_increase failed_salary
           fraction_60k fraction_80k fraction_not_in_profession others_salary years : ℝ)

-- Constants
def total_students : ℝ := 300
def graduated_students : ℝ := 270
def fyodor_initial_salary : ℝ := 25000
def fyodor_yearly_increase : ℝ := 3000
def failed_salary : ℝ := 25000
def fraction_60k : ℝ := 1 / 5
def fraction_80k : ℝ := 1 / 10
def fraction_not_in_profession : ℝ := 1 / 20
def others_salary : ℝ := 40000
def years : ℝ := 4

-- Fyodor's salary after 4 years
def fyodor_salary : ℝ := fyodor_initial_salary + (years * fyodor_yearly_increase)

-- Vasily's expected salary
def p_graduate : ℝ := graduated_students / total_students
def p_not_graduate : ℝ := 1 - p_graduate

def p_60k : ℝ := fraction_60k
def p_80k : ℝ := fraction_80k
def p_not_in_profession : ℝ := fraction_not_in_profession
def p_others_salary : ℝ := 1 - p_60k - p_80k - p_not_in_profession

def expected_salary_graduate : ℝ :=
  (p_60k * 60000) + (p_80k * 80000) + (p_not_in_profession * failed_salary) + (p_others_salary * others_salary)

def expected_salary : ℝ := (p_graduate * expected_salary_graduate) + (p_not_graduate * failed_salary)

-- Expected salary values
def vasily_expected_salary_after_4_years : ℝ := expected_salary

-- Comparisons
def salary_difference : ℝ := vasily_expected_salary_after_4_years - fyodor_salary

theorem vasily_salary_is_higher :
    vasily_expected_salary_after_4_years = 45025 ∧ fyodor_salary = 37000 ∧ salary_difference = 8025 :=
  by
    sorry

end vasily_salary_is_higher_l10_10408


namespace inclination_of_vertical_line_is_90_l10_10251

-- Definition of the problem and the associated constants
def line : ℝ → Prop := fun x => x = 1

-- The angle of inclination is defined for the line x = 1
noncomputable def inclination_angle (l : ℝ → Prop) : ℝ :=
  if ∃ x, l x ∧ x = 1 then 90 else 0

-- Problem statement
theorem inclination_of_vertical_line_is_90 :
  inclination_angle line = 90 :=
by
  sorry

end inclination_of_vertical_line_is_90_l10_10251


namespace danny_no_wrappers_found_l10_10528

theorem danny_no_wrappers_found (bottle_caps_park 💧nat) (total_wrappers 💧nat) (total_bottle_caps 💧nat) (extra_wrappers 💧nat) : 
  bottle_caps_park = 15 →
  total_wrappers = 67 →
  total_bottle_caps = 35 →
  extra_wrappers = 32 →
  total_wrappers - (total_bottle_caps + extra_wrappers) = 0 :=
by
  sorry

end danny_no_wrappers_found_l10_10528


namespace sum_positive_integers_l10_10934

theorem sum_positive_integers (N : ℕ) (h : ∀ (n : ℕ), (1.5 * n - 6 < 7.5) → n ≤ N) : 
  (∀ (n : ℕ), n ∈ {i | i < 9}) → ∑ i in {i | i < 9}, i = 36 := 
by
  sorry

end sum_positive_integers_l10_10934


namespace smallest_n_digit_count_l10_10314

def number_of_digits (n : ℕ) : ℕ :=
  nat.log10 n + 1
  
theorem smallest_n_digit_count
  (n : ℕ)
  (h_div : n % 30 = 0)
  (h_cube : ∃ k : ℕ, n^2 = k^3)
  (h_square : ∃ m : ℕ, n^3 = m^2)
  (h_min : ∀ m < n, (m % 30 = 0) → (∃ k, m^2 = k^3) → (∃ l, m^3 = l^2) → false) :
  number_of_digits n = 3 :=
sorry

end smallest_n_digit_count_l10_10314


namespace cos_angle_subtraction_l10_10606

theorem cos_angle_subtraction (A B : ℝ) : 
  (sin A + sin B = 1.5) ∧ (cos A + cos B = 1) → cos (A - B) = 0.625 := by
  sorry

end cos_angle_subtraction_l10_10606


namespace distinct_lightbulb_configurations_l10_10863

theorem distinct_lightbulb_configurations : 
  let m := 20
  let n := 16
  let switches := m + n 
  2 ^ (switches - 1) = 2 ^ 35 :=
by
  let m := 20
  let n := 16
  let switches := m + n 
  have : switches = 36 := by rw [switches, Nat.add_comm, Nat.add_eq_add_right_iff.mpr (eq.refl m), Nat.add_eq_right_cancel_iff.mpr (eq.refl n)]
  exact pow_eq_pow (switches - 1) 35

end distinct_lightbulb_configurations_l10_10863


namespace average_star_rating_l10_10761

/-- Define specific constants for the problem. --/
def reviews_5_star := 6
def reviews_4_star := 7
def reviews_3_star := 4
def reviews_2_star := 1
def total_reviews := 18

/-- Calculate the total stars given the number of each type of review. --/
def total_stars : ℕ := 
  (reviews_5_star * 5) + 
  (reviews_4_star * 4) + 
  (reviews_3_star * 3) + 
  (reviews_2_star * 2)

/-- Prove that the average star rating is 4. --/
theorem average_star_rating : total_stars / total_reviews = 4 := by 
  sorry

end average_star_rating_l10_10761


namespace cube_vertex_product_bounds_l10_10503

theorem cube_vertex_product_bounds :
  let faces := {5, 6, 7, 8, 9, 10}
      (exists (a ∈ faces) (b ∈ faces \ {a}) (c ∈ faces \ {a, b}) (d ∈ faces \ {a, b, c}) 
              (e ∈ faces \ {a, b, c, d}) (f ∈ faces \ {a, b, c, d, e})), 
  a + b = 15 ∧ c + d = 15 ∧ e + f = 15 
  → (a + b) * (c + d) * (e + f) = 3375 ∧ 
  (∃ other arrangement, (a' + b') * (c' + d') * (e' + f') = 3135) :=
  sorry

end cube_vertex_product_bounds_l10_10503


namespace measure_angle_APB_l10_10461

noncomputable def circle_radius_unit (P : Type*) [metric_space P] : Prop :=
∃ c : circle P, c.radius = 1

variables {P : Type*} [metric_space P]

-- Assume the existence of points A, B, C, D, C', D' within the circle c with unit radius
variables {A B C D C' D' : P}

-- Assume chord AB is divided into three equal parts at points C and D
def chord_division (A B C D : P) : Prop :=
dist A C = dist C D ∧ dist C D = dist D B

-- Assume the arc AB is divided into three equal parts at points C' and D'
def arc_division (A B C' D' : P) : Prop :=
dist A C' = dist C' D' ∧ dist C' D' = dist D' B

-- Point P is the intersection of two lines connecting these divisions
def intersection (A B C D C' D' P : P) : Prop :=
∃ P : P, line_through A C ∩ line_through D B = {P}

theorem measure_angle_APB 
  (h_unit_radius : circle_radius_unit P)
  (h_chord_div : chord_division A B C D)
  (h_arc_div : arc_division A B C' D')
  (h_intersection : intersection A B C D C' D' P) 
  : angle A P B = 20 :=
sorry

end measure_angle_APB_l10_10461


namespace EllipseEquation_l10_10186

theorem EllipseEquation
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (f1 : ℝ × ℝ := (-1, 0))
  (f2 : ℝ × ℝ := (1, 0))
  (P : ℝ × ℝ := (4/3, 1/3))
  (h4 : (P.1 ^ 2) / a^2 + (P.2 ^ 2) / b^2 = 1)
  (h5 : dist P f1 + dist P f2 = 2 * a)
  (h6 : (f1.1 - f2.1)^2 + (f1.2 - f2.2)^2 = 4 * 1^2) :
  (a = sqrt 2) ∧ (b = 1) → 
  (∀ x y : ℝ, (x^2 / 2 + y^2 = 1)) := 
sorry

end EllipseEquation_l10_10186


namespace average_bmi_is_correct_l10_10743

-- Define Rachel's parameters
def rachel_weight : ℕ := 75
def rachel_height : ℕ := 60  -- in inches

-- Define Jimmy's parameters based on the conditions
def jimmy_weight : ℕ := rachel_weight + 6
def jimmy_height : ℕ := rachel_height + 3

-- Define Adam's parameters based on the conditions
def adam_weight : ℕ := rachel_weight - 15
def adam_height : ℕ := rachel_height - 2

-- Define the BMI formula
def bmi (weight : ℕ) (height : ℕ) : ℚ := (weight * 703 : ℚ) / (height * height)

-- Rachel's, Jimmy's, and Adam's BMIs
def rachel_bmi : ℚ := bmi rachel_weight rachel_height
def jimmy_bmi : ℚ := bmi jimmy_weight jimmy_height
def adam_bmi : ℚ := bmi adam_weight adam_height

-- Proving the average BMI
theorem average_bmi_is_correct : 
  (rachel_bmi + jimmy_bmi + adam_bmi) / 3 = 13.85 := 
by
  sorry

end average_bmi_is_correct_l10_10743


namespace inequality_trig_l10_10601

theorem inequality_trig 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < (π / 2)) 
  (hy : 0 < y ∧ y < (π / 2)) 
  (hz : 0 < z ∧ z < (π / 2)) :
  (π / 2) + 2 * (Real.sin x) * (Real.cos y) + 2 * (Real.sin y) * (Real.cos z) > 
  (Real.sin (2 * x)) + (Real.sin (2 * y)) + (Real.sin (2 * z)) :=
by
  sorry  -- The proof is omitted

end inequality_trig_l10_10601


namespace largest_multiple_of_7_less_than_neg85_l10_10420

theorem largest_multiple_of_7_less_than_neg85 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n < -85 ∧ n = -91 :=
by
  sorry

end largest_multiple_of_7_less_than_neg85_l10_10420


namespace smallest_positive_solution_l10_10151

theorem smallest_positive_solution :
  ∃ x > 0, tan (4 * x) + tan (5 * x) = sec (5 * x) ∧ x = π / 26 := 
begin
  use π / 26,
  split,
  { exact real.pi_pos.trans (by norm_num), },
  split,
  { sorry, },
  { refl, }
end

end smallest_positive_solution_l10_10151


namespace cost_of_5kg_l10_10456

def cost_of_seeds (x : ℕ) : ℕ :=
  if x ≤ 2 then 5 * x else 4 * x + 2

theorem cost_of_5kg : cost_of_seeds 5 = 22 := by
  sorry

end cost_of_5kg_l10_10456


namespace jason_total_points_l10_10692

def total_points 
  (initial_seashells : ℕ) (initial_starfish : ℕ)
  (points_per_seashell : ℕ) (points_per_starfish : ℕ)
  (given_seashells_tim : ℕ) (given_seashells_lily : ℕ)
  (found_seashells : ℕ) (lost_seashells : ℕ) : ℕ :=
  let initial_points := initial_seashells * points_per_seashell + initial_starfish * points_per_starfish
  let points_given_away := (given_seashells_tim + given_seashells_lily) * points_per_seashell
  let net_found_lost_points := found_seashells * points_per_seashell - lost_seashells * points_per_seashell
  in initial_points - points_given_away + net_found_lost_points

theorem jason_total_points :
  total_points 49 48 2 3 13 7 15 5 = 222 :=
by simp [total_points]; norm_num; sorry

end jason_total_points_l10_10692


namespace difference_between_radii_l10_10783

-- Defining the areas of two concentric circles
def area_smaller_circle (r : ℝ) : ℝ := π * r ^ 2
def area_larger_circle (R : ℝ) : ℝ := π * R ^ 2

-- Defining the given condition: the ratio of the areas is 1:4
def ratio_of_areas (R r : ℝ) : Prop := (area_larger_circle R) / (area_smaller_circle r) = 4

-- Stating the proof goal: given the ratio of the areas, prove the difference in radii is r.
theorem difference_between_radii (R r : ℝ) (h : ratio_of_areas R r) : R - r = r :=
by
  sorry  -- Skipping the proof as per instruction

end difference_between_radii_l10_10783


namespace sequence_has_both_max_and_min_l10_10636

noncomputable def a_n (n : ℕ) : ℝ :=
  (n + 1) * ((-10 / 11) ^ n)

theorem sequence_has_both_max_and_min :
  ∃ (max min : ℝ) (N M : ℕ), 
    (∀ n : ℕ, a_n n ≤ max) ∧ (∀ n : ℕ, min ≤ a_n n) ∧ 
    (a_n N = max) ∧ (a_n M = min) := 
sorry

end sequence_has_both_max_and_min_l10_10636


namespace problem_check_l10_10957

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 4 = 4 ∧ ∀ n : ℕ, a (n + 1) ≠ 0 → a n * a (n + 1) = 2 ^ n

theorem problem_check (a : ℕ → ℝ) (h : sequence a) :
  a 1 = 1 ∧
  ¬(∀ n : ℕ, a n < a (n + 1)) ∧
  (∑ i in Finset.range 2023, a (i + 1)) = 2 ^ 1013 - 3 ∧
  (∑ i in Finset.range 2023, 1 / a (i + 1)) < 3 :=
by
  sorry

end problem_check_l10_10957


namespace number_of_sides_l10_10660

theorem number_of_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
by {
  sorry
}

end number_of_sides_l10_10660


namespace install_time_per_window_l10_10845

theorem install_time_per_window :
  ∀ (total_windows installed_windows remaining_hours : ℕ), 
  total_windows = 9 → 
  installed_windows = 6 → 
  remaining_hours = 18 → 
  (remaining_hours / (total_windows - installed_windows)) = 6 := 
by {
  intros total_windows installed_windows remaining_hours,
  sorry
}

end install_time_per_window_l10_10845


namespace neither_drinkers_eq_nine_l10_10098

-- Define the number of businessmen at the conference
def total_businessmen : Nat := 30

-- Define the number of businessmen who drank coffee
def coffee_drinkers : Nat := 15

-- Define the number of businessmen who drank tea
def tea_drinkers : Nat := 13

-- Define the number of businessmen who drank both coffee and tea
def both_drinkers : Nat := 7

-- Prove the number of businessmen who drank neither coffee nor tea
theorem neither_drinkers_eq_nine : 
  total_businessmen - ((coffee_drinkers + tea_drinkers) - both_drinkers) = 9 := 
by
  sorry

end neither_drinkers_eq_nine_l10_10098


namespace problem_equiv_l10_10505

noncomputable def plane : Type := sorry
noncomputable def line : Type := sorry

def is_parallel (a b : plane) : Prop := sorry
def is_perpendicular (a b : plane) : Prop := sorry
def parallel_to_plane (L : line) (α : plane) : Prop := sorry
def skew_lines (L m : line) : Prop := sorry
def three_non_collinear_points {α : plane} : Prop := sorry
def equidistant_from_plane (pts : set plane) (β : plane) : Prop := sorry

-- Define the planes and lines, and then state the conditions
variables (α β γ : plane) (L m : line) 

-- Theorem to prove
theorem problem_equiv :
  (is_parallel α β) ↔ 
  (skew_lines L m ∧ parallel_to_plane L α ∧ parallel_to_plane m α ∧ 
    parallel_to_plane L β ∧ parallel_to_plane m β) :=
sorry

end problem_equiv_l10_10505


namespace integer_part_sum_seq_l10_10849

noncomputable def a : ℕ → ℚ
| 0     := 1/4
| (n+1) := a n ^ 2 + a n

theorem integer_part_sum_seq : 
  (⌊∑ n in Finset.range 2012, 1 / (a n + 1)⌋ : ℤ) = 4 :=
by
  sorry

end integer_part_sum_seq_l10_10849


namespace largest_multiple_of_seven_smaller_than_neg_85_l10_10430

theorem largest_multiple_of_seven_smaller_than_neg_85 
  : ∃ k : ℤ, (k * 7 < -85) ∧ (∀ m : ℤ, (m * 7 < -85) → (m * 7 ≤ k * 7)) ∧ (k = -13) 
  := sorry

end largest_multiple_of_seven_smaller_than_neg_85_l10_10430


namespace weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900_l10_10396

noncomputable def Wa : ℕ := 
  let volume_a := (3/5) * 4
  let volume_b := (2/5) * 4
  let weight_b := 700
  let total_weight := 3280
  (total_weight - (weight_b * volume_b)) / volume_a

theorem weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900 :
  Wa = 900 := 
by
  sorry

end weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900_l10_10396


namespace count_visible_factor_numbers_l10_10479

def is_visible_factor_number (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∀ d in digits, d = 0 ∨ n % d = 0

theorem count_visible_factor_numbers : 
  ∃ n, n = 50 ∧ n = ((100 : ℕ) to 199).count is_visible_factor_number :=
by
  sorry

end count_visible_factor_numbers_l10_10479


namespace sum_positive_integers_l10_10933

theorem sum_positive_integers (N : ℕ) (h : ∀ (n : ℕ), (1.5 * n - 6 < 7.5) → n ≤ N) : 
  (∀ (n : ℕ), n ∈ {i | i < 9}) → ∑ i in {i | i < 9}, i = 36 := 
by
  sorry

end sum_positive_integers_l10_10933


namespace place_pieces_4x4_chessboard_l10_10499

def is_valid_position (pos1 pos2 : ℕ × ℕ) : Prop :=
  pos1.1 ≠ pos2.1 ∧ pos1.2 ≠ pos2.2 ∧ abs (pos1.1 - pos2.1) ≠ abs (pos1.2 - pos2.2)

theorem place_pieces_4x4_chessboard : 
  let pos1 := (1, 1)
  let pos2 := (2, 3)
  let pos3 := (3, 2)
  let pos4 := (4, 4)
  is_valid_position pos1 pos2 ∧
  is_valid_position pos1 pos3 ∧
  is_valid_position pos1 pos4 ∧
  is_valid_position pos2 pos3 ∧
  is_valid_position pos2 pos4 ∧
  is_valid_position pos3 pos4
  :=
by
  sorry

end place_pieces_4x4_chessboard_l10_10499


namespace find_area_of_triangle_ABC_l10_10619

noncomputable def area_of_triangle_ABC (x y : ℝ) (h k : ℝ) : ℝ :=
sorry  -- Placeholder for actual area calculation.

theorem find_area_of_triangle_ABC :
  let x := 1
  let y := -2
  let h := 1
  let k := -2
  let A := (3 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 0 : ℝ)
  let C := (0 : ℝ, -4 : ℝ)
  let area := area_of_triangle_ABC x y h k
  in area = sqrt 3 := sorry

end find_area_of_triangle_ABC_l10_10619


namespace perp_BCR_l10_10663

variable {A B C I P Q R : Point}
variable {γ : Circle}
variable {▱A : IsoscelesTriangle A B C} -- Isosceles triangle with AB = AC
variable {▱I : Incenter A B C I}        -- I is the incenter of triangle ABC
variable {⊙A : Circle A B}              -- Circle centered at A with radius AB
variable {⊙I : Circle I B}              -- Circle centered at I with radius IB
variable {γ : Circle B I}               -- Circle passing through B and I
variable {P Q : Point}                  -- Points of intersection

-- Additional geometric constraints based on the problem statement
def conditions : Prop :=
  γ.intersect_circle ⊙A P ∧ P ≠ B ∧
  γ.intersect_circle ⊙I Q ∧ Q ≠ B ∧
  Line IP ∩ Line BQ = R

theorem perp_BCR (cond : conditions) : ⟂ (Line B R) (Line C R) :=
  sorry

end perp_BCR_l10_10663


namespace balls_distribution_l10_10642

theorem balls_distribution : 
  ∃ (ways : ℕ), ways = 15 ∧ (∀ (distribution : Fin 3 → ℕ), (∀ i, 2 ≤ distribution i) → sum distribution = 10) :=
sorry

end balls_distribution_l10_10642


namespace best_constant_l10_10697

variable {n : ℕ}
variable (x : Fin n → ℝ)
variable {C : ℕ → ℝ}

theorem best_constant (h1 : 2 ≤ n)
                      (h2 : ∀ i, 0 < x i ∧ x i < 1)
                      (h3 : ∀ i j, 1 ≤ j → j < i → (1 - x i) * (1 - x j) ≥ 1 / 4) :
  ∑ i in Finset.range n, x i ≥ (2 / (3 * (n - 1))) * 
    ∑ i in Finset.range n \ Finset.singleton 0, 
        ∑ j in Finset.range i \ Finset.singleton 0, 
            (2 * x i * x j + real.sqrt (x i * x j)) :=
sorry

end best_constant_l10_10697


namespace num_lines_equal_intercepts_through_point_2_3_l10_10233

theorem num_lines_equal_intercepts_through_point_2_3 : 
  (∃ l₁ l₂ : ℝ → ℝ, 
    ∃ b : ℝ, 
    (l₁ = λ x, b - x ∧ l₂ = λ x, x - b) ∧ 
    l₁ 2 = 3 ∧ l₂ 2 = 3) → 2 := 
by sorry

end num_lines_equal_intercepts_through_point_2_3_l10_10233


namespace rectangle_enclosure_l10_10551
open BigOperators

theorem rectangle_enclosure (n m : ℕ) (hn : n = 5) (hm : m = 5) : 
  (∑ i in finset.range n, ∑ j in finset.range i, 1) * 
  (∑ k in finset.range m, ∑ l in finset.range k, 1) = 100 := by
  sorry

end rectangle_enclosure_l10_10551


namespace least_h_is_18_l10_10492
open Nat

def sequence (a : ℕ → ℤ) : Prop :=
  a 0 = 0 ∧
  a 1 = 3 ∧
  ∀ n ≥ 2, a n = 8 * a (n - 1) + 9 * a (n - 2) + 16

noncomputable def least_pos_integer_h (a : ℕ → ℤ) : ℕ :=
  Nat.find (λ h, ∀ n, 1999 ∣ (a (n + h) - a n))

theorem least_h_is_18 :
  ∀ a : ℕ → ℤ, sequence a → least_pos_integer_h a = 18 :=
by
  intros
  sorry

end least_h_is_18_l10_10492


namespace vasily_salary_correct_l10_10409

noncomputable def expected_salary_vasily : ℝ :=
  let salary_if_not_graduate : ℝ := 25000 in
  let salary_if_graduate : ℝ := 0.2 * 60000 + 0.1 * 80000 + 0.05 * 25000 + 0.65 * 40000 in
  let prob_graduate : ℝ := 0.9 in
  let prob_not_graduate : ℝ := 0.1 in
  prob_graduate * salary_if_graduate + prob_not_graduate * salary_if_not_graduate

theorem vasily_salary_correct : expected_salary_vasily = 45025 :=
by
  -- Skipping actual proof for brevity, we indicate its existence with sorry
  sorry

end vasily_salary_correct_l10_10409


namespace range_of_a_proof_l10_10045

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

theorem range_of_a_proof (a : ℝ) : range_of_a a ↔ 0 ≤ a ∧ a < 4 :=
by
  sorry

end range_of_a_proof_l10_10045


namespace angle_B_equiv_60_l10_10281

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A

theorem angle_B_equiv_60 
  (a b c A B C : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : A < π)
  (h6 : 0 < B) (h7 : B < π)
  (h8 : 0 < C) (h9 : C < π)
  (h_triangle : A + B + C = π)
  (h_arith : triangle_condition a b c A B C) : 
  B = π / 3 :=
by
  sorry

end angle_B_equiv_60_l10_10281


namespace length_of_bridge_l10_10448

-- Definitions based on the problem conditions
def train_length := 160  -- in meters
def train_speed_kmph := 45  -- in km/hr
def crossing_time := 30  -- in seconds

-- Conversion factor from km/hr to m/s
def kmph_to_mps (speed_kmph: ℝ) : ℝ := speed_kmph * 1000 / 3600

-- Speed in m/s
def train_speed_mps := kmph_to_mps train_speed_kmph

-- Total distance covered in crossing_time
def total_distance := train_speed_mps * crossing_time

-- Proof statement
theorem length_of_bridge :
  total_distance - train_length = 215 :=
by
  sorry

end length_of_bridge_l10_10448


namespace count_integer_roots_of_10000_eq_3_l10_10582

theorem count_integer_roots_of_10000_eq_3 : 
  {n : ℕ | ((10000 : ℕ) ^ (1 / n : ℝ)).isInt}.finite ∧ 
  (Finset.card {n : ℕ | ((10000 : ℕ) ^ (1 / n : ℝ)).isInt} = 3) :=
sorry

end count_integer_roots_of_10000_eq_3_l10_10582


namespace problem_solution_l10_10627

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f x + 2 * Real.cos x ^ 2

theorem problem_solution :
  (∀ x, (∃ ω > 0, ∃ φ, |φ| < Real.pi / 2 ∧ Real.sin (ω * x - φ) = 0 ∧ 2 * ω = Real.pi)) →
  (∀ x, f x = Real.sin (2 * x - Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), (g x ≤ 2 ∧ g x ≥ 1 / 2)) :=
by
  sorry

end problem_solution_l10_10627


namespace find_natural_number_unique_l10_10135

theorem find_natural_number_unique : ∃! n : ℕ, (n^3.digits ++ n^4.digits).perm (List.range 10) ∧ (n^3.digits ++ n^4.digits).nodup := 
by
  sorry

end find_natural_number_unique_l10_10135


namespace integer_valued_polynomial_l10_10351

def f (x : ℤ) : ℚ := (1/5) * x^5 + (1/2) * x^4 + (1/3) * x^3 - (1/30) * x

theorem integer_valued_polynomial : ∀ x : ℤ, ∃ z : ℤ, f(x) = z :=
sorry

end integer_valued_polynomial_l10_10351


namespace tan_C_over_tan_A_max_tan_B_l10_10305

theorem tan_C_over_tan_A {A B C : ℝ} {a b c : ℝ} (h : a^2 + 2 * b^2 = c^2) :
  let tan_A := Real.tan A
  let tan_C := Real.tan C
  (Real.tan C / Real.tan A) = -3 :=
sorry

theorem max_tan_B {A B C : ℝ} {a b c : ℝ} (h : a^2 + 2 * b^2 = c^2) :
  let B := Real.arctan (Real.tan B)
  ∃ (x : ℝ), x = Real.tan B ∧ ∀ y, y = Real.tan B → y ≤ (Real.sqrt 3) / 3 :=
sorry

end tan_C_over_tan_A_max_tan_B_l10_10305


namespace num_ways_to_form_rectangle_l10_10567

theorem num_ways_to_form_rectangle (n : ℕ) (h : n = 5) :
  (nat.choose n 2) * (nat.choose n 2) = 100 :=
by {
  rw h,
  exact nat.choose_five_two_mul_five_two 100
}

lemma nat.choose_five_two_mul_five_two :
  ((5.choose 2) * (5.choose 2) = 100) :=
by norm_num

end num_ways_to_form_rectangle_l10_10567


namespace sum_geometric_series_l10_10936

noncomputable def S (r : ℝ) : ℝ :=
  12 / (1 - r)

theorem sum_geometric_series (a : ℝ) (h1 : -1 < a) (h2 : a < 1) (h3 : S a * S (-a) = 2016) :
  S a + S (-a) = 336 :=
by
  sorry

end sum_geometric_series_l10_10936


namespace no_partition_with_continuous_function_l10_10880

theorem no_partition_with_continuous_function :
  ¬ ∃ (A B : set ℝ) (f : ℝ → ℝ),
    (A ∪ B = set.Icc 0 1) ∧ (A ∩ B = ∅) ∧
    continuous_on f (set.Icc 0 1) ∧
    (∀ a ∈ A, f a ∈ B) ∧
    (∀ b ∈ B, f b ∈ A) :=
by
  sorry

end no_partition_with_continuous_function_l10_10880


namespace equation_has_100_solutions_l10_10926

noncomputable theory

open Real

def num_solutions : ℝ :=
  { x : ℝ | 0 ≤ x ∧ x ≤ 100 * π ∧ cos (π / 2 + x) = (1 / 2) ^ x }.toFinset.card

theorem equation_has_100_solutions :
    num_solutions = 100 :=
sorry

end equation_has_100_solutions_l10_10926


namespace smallest_positive_solution_tan_sec_eq_l10_10152

theorem smallest_positive_solution_tan_sec_eq 
  (x : ℝ) 
  (hx : x > 0)
  (hx_rad : ∃ y : ℝ, x = y * real.pi) 
  (h_eq : real.tan (4 * x) + real.tan (5 * x) = real.sec (5 * x)) :
  x = real.pi / 18 :=
sorry

end smallest_positive_solution_tan_sec_eq_l10_10152


namespace bahman_max_area_l10_10100

-- Define the length of each fence
def s := 10

-- Define the formula for the area of the trapezoid with a given height 'h'
def area (h : ℝ) : ℝ := s * h

-- Define the height 'h' derived from the maximal condition
def max_height : ℝ := (s * Real.sqrt 3) / 2

-- The highest enclosed area using the fences and garden wall
def max_area : ℝ := area max_height

-- Define the problem: Prove the calculated maximum area matches the given answer
theorem bahman_max_area : max_area = 50 + 25 * Real.sqrt 3 := by
  sorry

end bahman_max_area_l10_10100


namespace log_diff_log_example_l10_10872

theorem log_diff :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → log 10 a - log 10 b = log 10 (a / b) :=
begin
  -- Proof not required
  sorry
end

theorem log_example :
  log 10 80 - log 10 16 = log 10 5 :=
begin
  have h := log_diff 80 16 (by norm_num) (by norm_num),
  rw div_eq_mul_inv at h,
  norm_num at h,
  exact h,
end

end log_diff_log_example_l10_10872


namespace dan_final_marbles_correct_l10_10114

variable (initial_marbles : ℕ) (percent_to_mary : ℚ) (fraction_to_peter : ℚ) (cousin_gift : ℕ) (percent_from_alice : ℚ) (alice_collection : ℕ)

def dan_final_marbles (initial_marbles : ℕ) (percent_to_mary : ℚ) (fraction_to_peter : ℚ) (cousin_gift : ℕ) (percent_from_alice : ℚ) (alice_collection : ℕ) : ℕ :=
  let marbles_after_mary := initial_marbles - (initial_marbles * percent_to_mary).to_nat in
  let marbles_after_peter := marbles_after_mary - (marbles_after_mary * fraction_to_peter).to_nat in
  let marbles_after_cousin := marbles_after_peter + cousin_gift in
  let marbles_from_alice := (alice_collection * percent_from_alice).to_nat in
  marbles_after_cousin + marbles_from_alice

open scoped Qq

theorem dan_final_marbles_correct :
  dan_final_marbles 250 (15 / 100) (1 / 5) 35 (8 / 100) 180 = 220 :=
by
  sorry

end dan_final_marbles_correct_l10_10114


namespace withdrawn_players_match_count_l10_10670

theorem withdrawn_players_match_count:
  (n : ℕ) (total_matches : ℕ) (withdrawn_players : ℕ) (withdrawn_matches : ℕ) (r : ℕ) (n = 13 ∧ total_matches = 50 ∧ withdrawn_players = 3 ∧ withdrawn_matches = 2 ∧
    (withdrawn_players * withdrawn_matches - r = 6) ∧ (50 = ∑ i in finset.range (n - withdrawn_players).card, i / 2 + (withdrawn_matches * withdrawn_players - r)))
  → r = 1 :=
by
  sorry

end withdrawn_players_match_count_l10_10670


namespace table_tennis_probability_l10_10677

-- Define the given conditions
def prob_A_wins_set : ℚ := 2 / 3
def prob_B_wins_set : ℚ := 1 / 3
def best_of_five_sets := 5
def needed_wins_for_A := 3
def needed_losses_for_A := 2

-- Define the problem to prove
theorem table_tennis_probability :
  ((prob_A_wins_set ^ 2) * prob_B_wins_set * prob_A_wins_set) = 8 / 27 :=
by
  sorry

end table_tennis_probability_l10_10677


namespace toy_cost_price_l10_10478

theorem toy_cost_price (C : ℕ) (h : 18 * C + 3 * C = 25200) : C = 1200 := by
  -- The proof is not required
  sorry

end toy_cost_price_l10_10478


namespace problem1_problem2_l10_10878

-- Problem 1 statement
theorem problem1 : -1^(2023) - real.cbrt (-27) + |real.sqrt 3 - 2| = -2 - real.sqrt 3 := by
  sorry

-- Problem 2 statement
variables {x y : ℝ}

theorem problem2 : (2 * x * y)^3 / (4 * x^2 * y) * (1 / 4 * x * y) = (1 / 2) * x^2 * y^3 := by
  sorry

end problem1_problem2_l10_10878


namespace book_pages_l10_10515

theorem book_pages (n days_n : ℕ) (first_day_pages break_days : ℕ) (common_difference total_pages_read : ℕ) (portion_of_book : ℚ) :
    n = 14 → days_n = 12 → first_day_pages = 10 → break_days = 2 → common_difference = 2 →
    total_pages_read = 252 → portion_of_book = 3/4 →
    (total_pages_read : ℚ) * (4/3) = 336 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end book_pages_l10_10515


namespace vector_magnitude_perpendicular_vectors_l10_10227

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions from the conditions
variables (a b : V)
variable (k : ℝ)
variable (angle_ab : real.angle)
variable (norm_a : ∥a∥ = 1)
variable (norm_b : ∥b∥ = 2)
variable (angle_ab_120 : angle_ab = real.angle.of_deg 120)

/- Prove $|\overrightarrow{a} - 2\overrightarrow{b}| = \sqrt{21}$ -/
theorem vector_magnitude (h : real.angle.to_cos angle_ab = -1 / 2) : ∥a - 2 • b∥ = real.sqrt 21 :=
sorry

/- Prove $(\overrightarrow{a} + 2\overrightarrow{b}) \perp (k\overrightarrow{a} - \overrightarrow{b}) \implies k = -7$ -/
theorem perpendicular_vectors (h : ⟪a + 2 • b, k • a - b⟫ = 0) : k = -7 :=
sorry

end vector_magnitude_perpendicular_vectors_l10_10227


namespace cosine_angle_ab_l10_10994

noncomputable def vec_a : ℝ × ℝ := (1, 1)
noncomputable def vec_b : ℝ × ℝ := (-1, 2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def cosine_of_angle (v w : ℝ × ℝ) : ℝ :=
  (dot_product v w) / (magnitude v * magnitude w)

theorem cosine_angle_ab : cosine_of_angle vec_a vec_b = real.sqrt 10 / 10 := sorry

end cosine_angle_ab_l10_10994


namespace solve_quad_eq_l10_10392

theorem solve_quad_eq (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  intro h
  sorry

end solve_quad_eq_l10_10392


namespace solution_set_intersection_l10_10616

theorem solution_set_intersection (a b : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, x^2 + x - 6 < 0 ↔ -3 < x ∧ x < 2) →
  (∀ x : ℝ, x^2 + a * x + b < 0 ↔ (-1 < x ∧ x < 2)) →
  a + b = -3 :=
by 
  sorry

end solution_set_intersection_l10_10616


namespace find_monthly_salary_l10_10068

-- Definitions based on the conditions
def initial_saving_rate : ℝ := 0.25
def initial_expense_rate : ℝ := 1 - initial_saving_rate
def expense_increase_rate : ℝ := 1.25
def final_saving : ℝ := 300

-- Theorem: Prove the man's monthly salary
theorem find_monthly_salary (S : ℝ) (h1 : initial_saving_rate = 0.25)
  (h2 : initial_expense_rate = 0.75) (h3 : expense_increase_rate = 1.25)
  (h4 : final_saving = 300) : S = 4800 :=
by
  sorry

end find_monthly_salary_l10_10068


namespace real_solutions_l10_10921

noncomputable def RealSolutions (a b c x : ℝ) :=
  sqrt (a + b * x) + sqrt (b + c * x) + sqrt (c + a * x) = sqrt (b - a * x) + sqrt (c - b * x) + sqrt (a - c * x)

theorem real_solutions (a b c : ℝ) (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 → ∀ x : ℝ, RealSolutions a b c x ↔ x = 0) ∧ (a = 0 ∧ b = 0 ∧ c = 0 → ∀ x : ℝ, RealSolutions a b c x ↔ true) :=
by
  sorry

end real_solutions_l10_10921


namespace three_numbers_diff_l10_10447

theorem three_numbers_diff (k : ℕ) (a b c : ℕ) 
  (H1 : a ∈ {6 * k + 1, 6 * k + 2, 6 * k + 3, 6 * k + 4, 6 * k + 5, 6 * k + 6}) 
  (H2 : b ∈ {6 * k + 1, 6 * k + 2, 6 * k + 3, 6 * k + 4, 6 * k + 5, 6 * k + 6}) 
  (H3 : c ∈ {6 * k + 1, 6 * k + 2, 6 * k + 3, 6 * k + 4, 6 * k + 5, 6 * k + 6}) 
  (H4 : a ≠ b) (H5 : a ≠ c) (H6 : b ≠ c) :
  ∃ x y, (x ≠ y ∧ x ∈ {a, b, c} ∧ y ∈ {a, b, c}) ∧ (x - y = 1 ∨ x - y = 4 ∨ x - y = 5) :=
sorry

end three_numbers_diff_l10_10447


namespace log_sum_geom_seq_l10_10260

noncomputable def a_n : ℕ → ℝ := sorry
def r : ℝ := sorry

theorem log_sum_geom_seq (h1 : ∀ n, a_n n > 0) (h2 : a_n 5 * a_n 6 = 81) :
  (∑ i in Finset.range 10, Real.log (a_n (i + 1)) / Real.log 3) = 20 :=
sorry

end log_sum_geom_seq_l10_10260


namespace cranberry_initial_count_l10_10513

noncomputable def initial_cranberries (C : ℕ) : Prop :=
  let harvested := 0.40 * C
  let remaining_after_human_harvest := C - harvested
  let remaining_after_elk := remaining_after_human_harvest - 20000
  remaining_after_elk = 16000

theorem cranberry_initial_count {C : ℕ} (h : initial_cranberries C) : C = 60000 :=
  by sorry

end cranberry_initial_count_l10_10513


namespace fisherman_total_fish_l10_10472

theorem fisherman_total_fish :
  let bass : Nat := 32
  let trout : Nat := bass / 4
  let blue_gill : Nat := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  show bass + trout + blue_gill = 104
  sorry

end fisherman_total_fish_l10_10472


namespace surface_area_of_cylinder_with_square_cross_section_l10_10202

theorem surface_area_of_cylinder_with_square_cross_section
  (side_length : ℝ) (h1 : side_length = 2) : 
  (2 * Real.pi * 2 + 2 * Real.pi * 1^2) = 6 * Real.pi :=
by
  rw [←h1]
  sorry

end surface_area_of_cylinder_with_square_cross_section_l10_10202


namespace problem_statement_l10_10611

theorem problem_statement (f : ℝ → ℝ) (hf_odd : ∀ x, f (-x) = - f x)
  (hf_deriv : ∀ x < 0, 2 * f x + x * deriv f x < 0) :
  f 1 < 2016 * f (Real.sqrt 2016) ∧ 2016 * f (Real.sqrt 2016) < 2017 * f (Real.sqrt 2017) := 
  sorry

end problem_statement_l10_10611


namespace probability_all_flip_same_times_l10_10332

theorem probability_all_flip_same_times :
  (let P_oliver (n : ℕ) := (2/3)^(n-1) * (1/3))
  (let P_jayden (n : ℕ) := (3/4)^(n-1) * (1/4))
  (let P_mia (n : ℕ) := (4/5)^(n-1) * (1/5))
  let combined_probability (n : ℕ) := P_oliver n * P_jayden n * P_mia n
  let total_probability := ∑' n : ℕ, if n = 0 then 0 else combined_probability n
  total_probability = 1 / 36 :=
by
  -- Placeholder for proof
  sorry

end probability_all_flip_same_times_l10_10332


namespace value_of_a_51_l10_10276

noncomputable def a : ℕ → ℕ
| 0       := 1
| (n + 1) := a n + 2

theorem value_of_a_51 : a 51 = 101 := 
by sorry

end value_of_a_51_l10_10276


namespace limit_problem_l10_10518

theorem limit_problem :
  (Real.log (343 / 9)) = (lim (λ x : ℝ, (7^(3 * x) - 3^(2 * x)) / (Real.tan x + x^3)) (nhdsWithin 0 (set.univ))) :=
sorry

end limit_problem_l10_10518


namespace midpoint_triang_cong_l10_10592

theorem midpoint_triang_cong {A B C D P Q : Point} 
  (HmidA : midpoint A D) (HmidB : midpoint B C)
  (HmidP : midpoint P Q) (HmidQ : midpoint Q P)
  (Hpar1 : segment A D ∥ segment B C) (Hequal1 : segment A D = segment B C)
  (Hpar2 : segment D Q ∥ segment B P) (Hequal2 : segment D Q = segment B P)
  (Hpar3 : segment A Q ∥ segment C P) (Hequal3 : segment A Q = segment C P) :
  triangle B C P ≅ triangle A D Q :=
by 
  sorry

end midpoint_triang_cong_l10_10592


namespace distance_between_A_and_B_l10_10866

noncomputable def billy_and_bobby_meeting (d : ℝ) : Prop :=
  ∃ x, (3 / (x - 3) = (x - 10) / (2 * x - 10) ∨ 3 / (x - 3) = (x + 10) / (2 * x - 10)) ∧ d = x
  
theorem distance_between_A_and_B : ∃ d, billy_and_bobby_meeting d ∧ d = 15 :=
by
  use 15
  split
  sorry

end distance_between_A_and_B_l10_10866


namespace min_stamps_l10_10867

theorem min_stamps (x y : ℕ) (h : 5 * x + 7 * y = 47) : x + y ≥ 7 :=
by 
  have h₀ : ∃ x y : ℕ, 5 * x + 7 * y = 47 := sorry,
  have min_value := minstamps h₀,
  exact min_value

end min_stamps_l10_10867


namespace part1_part2_l10_10977

-- Define the sequence a_n and the sum of the first n terms S_n
def a (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => n + 1 -- by induction we get a_n = n

def S (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2 -- Sum of the first n natural numbers

-- Given conditions and correct answer
theorem part1 (n : ℕ) : 2 * S n = (n + 1) * a n := by
  sorry

-- Define T_n as the sum of the first n terms of the sequence {log_2(a_{n+1} / a_n)}
def log2 (x : ℕ) : ℝ := Real.log x / Real.log 2

def T (n : ℕ) : ℝ :=
  ∑ i in range(n), log2 ((i + 2) / (i + 1)) -- telescoping sum log_2(n + 1)

-- Given conditions and correct answer
theorem part2 : ∃ (k : ℕ), k ≤ 1023 ∧ T k ≤ 10 := by
  have k_bound : ∀ k, T k = log2 (k + 1) := by
    sorry
  -- Showing that T_k ≤ 10
  use 1023
  have : log2 (1023 + 1) = 10 := by
    sorry
  sorry

end part1_part2_l10_10977


namespace trapezoid_longer_parallel_side_l10_10852

theorem trapezoid_longer_parallel_side (s x : ℝ) (h1 : s > 0) (h2 : (∃ P Q R S O : ℝ, 
(split_square_into_pentagon_and_trapezoids s O P Q R S ∧ ∀ (t1 t2 t3 : ℝ), congruent_trapezoids t1 t2 t3 O P Q R S))) :
  (x = s / 2) :=
sorry


end trapezoid_longer_parallel_side_l10_10852


namespace distance_O_M_is_15_l10_10769

-- Define the necessary points and their properties
structure Point :=
(x : ℝ)
(y : ℝ)

def O := Point.mk 0 0
def M := Point.mk (7 + 5) 9  -- Distance derived: Horizontal 7 + 5, Vertical 14 - 5

noncomputable def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

-- Define a theorem to prove that the distance OM is 15
theorem distance_O_M_is_15 : distance O M = 15 := by
  sorry

end distance_O_M_is_15_l10_10769


namespace infinite_n_with_exactly_i_cubable_l10_10347

def is_sum_of_three_pos_int_cubes (x : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ x = a^3 + b^3 + c^3

theorem infinite_n_with_exactly_i_cubable (i : ℕ)
  (h_i : i = 1 ∨ i = 2 ∨ i = 3) :
  ∃ᶠ n in at_top, (finset.filter is_sum_of_three_pos_int_cubes (finset.range 29)).card = i :=
begin
  sorry
end

end infinite_n_with_exactly_i_cubable_l10_10347


namespace train_crossing_time_l10_10853

theorem train_crossing_time :
  ∀ (lengthTrain speedTrain lengthBridge : ℕ),
    lengthTrain = 150 →
    speedTrain = 45 →
    lengthBridge = 225 →
    let totalDistance := lengthTrain + lengthBridge in
    let speedTrainMS := (speedTrain * 1000) / 3600 in
    let time := totalDistance / speedTrainMS in
    time = 30 := 
by
  intros lengthTrain speedTrain lengthBridge h1 h2 h3
  let totalDistance := lengthTrain + lengthBridge
  have h4: totalDistance = 375 := by rw [h1, h3]; simp
  let speedTrainMS := (speedTrain * 1000) / 3600
  have h5: speedTrainMS = 12.5 := by rw [h2]; norm_num1
  let time := totalDistance / speedTrainMS
  have h6: time = 30 := by rw [h4, h5]; norm_num1
  exact h6

end train_crossing_time_l10_10853


namespace sequence_an_l10_10183

theorem sequence_an (a : ℕ → ℝ) (h₁ : a 1 = 1/2)
  (h_recurrence : ∀ n : ℕ, a (n + 1) ^ 2 = 2 * a n ^ 2 / (1 + 2 * a n ^ 2))
  : ∀ n : ℕ, a n = sqrt (2^(n-2) / (1 + 2^(n-1))) := sorry

end sequence_an_l10_10183


namespace line_symmetric_y_axis_eqn_l10_10374

theorem line_symmetric_y_axis_eqn (x y : ℝ) : 
  (∀ x y : ℝ, x - y + 1 = 0 → x + y - 1 = 0) := 
sorry

end line_symmetric_y_axis_eqn_l10_10374


namespace range_of_inverse_power_l10_10362

open Set

theorem range_of_inverse_power (k : ℝ) (hk : k > 0) :
  range (λ x : ℝ, x ∈ Icc 1 (⊤ : ℝ) → x ^ (-k)) = Ioc 0 1 :=
sorry

end range_of_inverse_power_l10_10362


namespace reckha_valid_codes_l10_10728

def valid_code_count_filtered (f : ℕ → ℕ → ℕ → Bool) : ℕ :=
  let codes := (List.range 10).bind (λ d1 => (List.range 10).bind (λ d2 => (List.range 10).map (λ d3 => (d1, d2, d3))))
  codes.count (λ code => f code.fst code.snd.fst code.snd.snd)

def is_valid_code (code fixed_position transpositions : ℕ × ℕ × ℕ) (a b c : ℕ) : Bool :=
  -- Not the original code
  (a, b, c) ≠ fixed_position ∧
  -- Not matching two or more positions with the original code
  (if (a, b, c) = fixed_position then false else
    (((a = fixed_position.1).toNat + (b = fixed_position.2.1).toNat + (c = fixed_position.2.2).toNat) < 2)) ∧
  -- Not a transposition of the original code
  (transpositions.all (λ (t1, t2, t3), (a, b, c) ≠ (t1, t2, t3)))

def fixed_code : ℕ × ℕ × ℕ := (0, 4, 5)
def transpositions : List (ℕ × ℕ × ℕ) := [(4, 0, 5), (5, 0, 4), (0, 5, 4)]

theorem reckha_valid_codes : valid_code_count_filtered (is_valid_code fixed_code transpositions) = 970 := by
  sorry

end reckha_valid_codes_l10_10728


namespace least_not_factor_of_30fact_and_composite_l10_10020

theorem least_not_factor_of_30fact_and_composite : 
  ∃ n : ℕ, n > 30 ∧ ¬(n ∣ 30!) ∧ ¬(Prime n) ∧ (∀ m : ℕ, m > 30 ∧ ¬(m ∣ 30!) ∧ ¬(Prime m) → n ≤ m) ↔ n = 961 :=
sorry

end least_not_factor_of_30fact_and_composite_l10_10020


namespace fisherman_total_fish_l10_10468

theorem fisherman_total_fish :
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  sorry

end fisherman_total_fish_l10_10468


namespace bonus_percentage_correct_l10_10669

/-
Tom serves 10 customers per hour and works for 8 hours, earning 16 bonus points.
We need to find the percentage of bonus points per customer served.
-/

def customers_per_hour : ℕ := 10
def hours_worked : ℕ := 8
def total_bonus_points : ℕ := 16

def total_customers_served : ℕ := customers_per_hour * hours_worked
def bonus_percentage : ℕ := (total_bonus_points * 100) / total_customers_served

theorem bonus_percentage_correct : bonus_percentage = 20 := by
  sorry

end bonus_percentage_correct_l10_10669


namespace polygon_sides_l10_10656

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_sides_l10_10656


namespace num_values_with_prime_sum_divisors_l10_10702

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem num_values_with_prime_sum_divisors :
  (Finset.filter (λ n, is_prime (sum_divisors n)) (Finset.range 51)).card = 5 :=
by sorry

end num_values_with_prime_sum_divisors_l10_10702


namespace range_of_m_l10_10250

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + x^2 - 5 * x + c

theorem range_of_m (c : ℝ) : ∀ m : ℝ,
  (∀ x ∈ Ioo m (m + 1), deriv f x ≤ 0) ↔ m ∈ set.Icc (1 / 2) 1 :=
by
  sorry

end range_of_m_l10_10250


namespace count_f_n_prime_l10_10703

open Nat

/-- Sum of positive divisors of n -/
def f (n : ℕ) : ℕ :=
  (divisors n).sum

/-- Predicate to check if a number is prime -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

/-- Main statement to be proven -/
theorem count_f_n_prime : ( Finset.filter (λ n, is_prime (f n)) (Finset.range 51) ).card = 5 :=
by
  sorry

end count_f_n_prime_l10_10703


namespace candy_probability_l10_10842

theorem candy_probability (red_candies blue_candies : ℕ) 
  (terry_picks mary_picks : {n // n ≤ red_candies + blue_candies}) 
  (h_total : red_candies = 10 ∧ blue_candies = 10)
  (h_terry: terry_picks.val = 2 ∧ mary_picks.val = 2)
  : (m + n = 441) :=
by {
  -- Definitions for the conditions given
  have h_jar : red_candies + blue_candies = 20, 
    from by {simp [h_total]},
  
  -- Initial assumption around the combination
  have terry_red_comb : ℚ := (10 / 20) * (9 / 19),
  have mary_red_comb  : ℚ := (8 / 18) * (7 / 17),
  have both_red_comb  : ℚ := terry_red_comb * mary_red_comb,
  have both_blue_comb : ℚ := both_red_comb, -- Both probabilities are equivalent
  
  -- Calculation for different color combination
  have terry_diff_comb : ℚ := (10 / 20) * (10 / 19),
  have mary_diff_comb  : ℚ := (9 / 18) * (9 / 17),
  have both_diff_comb  : ℚ := terry_diff_comb * mary_diff_comb,
  
  -- Summing up all possibilities
  have total_prob := 2 * both_red_comb + both_diff_comb,

  -- Ensuring that the fraction representation is in its simplest form
  have simplest_form := (118 : ℚ)/(323 : ℚ),
  have h1 : total_prob = simplest_form :=
    sorry, -- Proof is skipped
  
  exact 118 + 323
}

end candy_probability_l10_10842


namespace sin_120_eq_sqrt3_div_2_l10_10876

theorem sin_120_eq_sqrt3_div_2 :
  let deg := Float.pi / 180
  sin (120 * deg) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l10_10876


namespace average_star_rating_l10_10760

/-- Define specific constants for the problem. --/
def reviews_5_star := 6
def reviews_4_star := 7
def reviews_3_star := 4
def reviews_2_star := 1
def total_reviews := 18

/-- Calculate the total stars given the number of each type of review. --/
def total_stars : ℕ := 
  (reviews_5_star * 5) + 
  (reviews_4_star * 4) + 
  (reviews_3_star * 3) + 
  (reviews_2_star * 2)

/-- Prove that the average star rating is 4. --/
theorem average_star_rating : total_stars / total_reviews = 4 := by 
  sorry

end average_star_rating_l10_10760


namespace math_proof_problem_l10_10172

variable {𝕜 : Type*}
variable {x : 𝕜} {a b : 𝕜}
variable {f : 𝕜 → 𝕜}
variable {x_n : ℕ → 𝕜}

-- Condition (1): Definition of function f
def f_def (x : 𝕜) (f : 𝕜 → 𝕜) (a b : 𝕜) : Prop := 
f x = (b * x + 1) / (a * x + 1)^2 ∧ a > 0 ∧ x ≠ -1 / a

-- Condition (2): Given values for f(1) and f(-2)
def f_conditions (f : 𝕜 → 𝕜) : Prop :=
f 1 = 1 / 4 ∧ f (-2) = 1

-- Conjecture for the general term
def sequence_general_term (x_n : ℕ → 𝕜) : Prop :=
∀ (n : ℕ), x_n n = (n + 2) / (2 * n + 2)

-- Main theorem statement, encompassing all requirements
theorem math_proof_problem :
(f_def x f a b) ∧
(f_conditions f) ∧ 
(sequence_general_term x_n) :=
by
  sorry

end math_proof_problem_l10_10172


namespace proof_problem_l10_10047

noncomputable def initialEfficiencyOfOneMan : ℕ := sorry
noncomputable def initialEfficiencyOfOneWoman : ℕ := sorry
noncomputable def totalWork : ℕ := sorry

-- Condition (1): 10 men and 15 women together can complete the work in 6 days.
def condition1 := 10 * initialEfficiencyOfOneMan + 15 * initialEfficiencyOfOneWoman = totalWork / 6

-- Condition (2): The efficiency of men to complete the work decreases by 5% every day.
-- This condition is not directly measurable to our proof but noted as additional info.

-- Condition (3): The efficiency of women to complete the work increases by 3% every day.
-- This condition is not directly measurable to our proof but noted as additional info.

-- Condition (4): It takes 100 days for one man alone to complete the same work at his initial efficiency.
def condition4 := initialEfficiencyOfOneMan = totalWork / 100

-- Define the days required for one woman alone to complete the work at her initial efficiency.
noncomputable def daysForWomanToCompleteWork : ℕ := 225

-- Mathematically equivalent proof problem
theorem proof_problem : 
  condition1 ∧ condition4 → (totalWork / daysForWomanToCompleteWork = initialEfficiencyOfOneWoman) :=
by
  sorry

end proof_problem_l10_10047


namespace primes_subset_all_primes_l10_10298

theorem primes_subset_all_primes (P : Set ℕ) (M : Set ℕ) [∀ p, Prime p → p ∈ P] 
  (hM : M ⊆ P ∧ M ≠ ∅) 
  (h : ∀ (S : Finset ℕ), S ≠ ∅ → (∀ p ∈ S, p ∈ M) → ∀ q, Prime q → q ∣ (S.prod id + 1) → q ∈ M) :
  M = P :=
sorry

end primes_subset_all_primes_l10_10298


namespace gcd_lcm_sum_l10_10698

open Int

def gcd (a b : ℕ) : ℕ := if h : b = 0 then a else gcd b (a % b)
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

def gcd_three (a b c : ℕ) : ℕ := gcd a (gcd b c)
def lcm_three (a b c : ℕ) : ℕ := lcm a (lcm b c)

theorem gcd_lcm_sum (a b c : ℕ) (C D : ℕ):
  gcd_three a b c = C → lcm_three a b c = D → C + D = 78 :=
sorry

#eval gcd 18 24  -- Just to test the GCD function
#eval lcm 18 24  -- Just to test the LCM function
#eval gcd_three 18 24 36  -- To verify it matches the conditions
#eval lcm_three 18 24 36  -- To verify it matches the conditions

#check gcd_lcm_sum 18 24 36 6 72

end gcd_lcm_sum_l10_10698


namespace sum_cotangents_equal_l10_10352

theorem sum_cotangents_equal (a b c S m_a m_b m_c S' : ℝ) (cot_A cot_B cot_C cot_A' cot_B' cot_C' : ℝ)
  (h1 : cot_A + cot_B + cot_C = (a^2 + b^2 + c^2) / (4 * S))
  (h2 : m_a^2 + m_b^2 + m_c^2 = 3 * (a^2 + b^2 + c^2) / 4)
  (h3 : S' = 3 * S / 4)
  (h4 : cot_A' + cot_B' + cot_C' = (m_a^2 + m_b^2 + m_c^2) / (4 * S')) :
  cot_A + cot_B + cot_C = cot_A' + cot_B' + cot_C' :=
by
  -- Proof is needed, but omitted here
  sorry

end sum_cotangents_equal_l10_10352


namespace circle_tangent_radius_values_l10_10588

theorem circle_tangent_radius_values :
  ∃ r : ℝ, r > 0 ∧ ((r = sqrt 3 / 2) ∨ (r = 3 * sqrt 3 / 2)) :=
sorry

end circle_tangent_radius_values_l10_10588


namespace smoothie_size_l10_10859

theorem smoothie_size (packets_per_smoothie : ℝ := 180 / 150)
  (size_per_packet : ℝ := 3)
  (water_per_packet : ℝ := 15)
  (smoothies : ℝ := 150)
  (total_packets : ℝ := 180) :
  (packets_per_smoothie * size_per_packet + packets_per_smoothie * water_per_packet = 21.6) :=
by 
  have packets_per_smoothie_calc : packets_per_smoothie = 1.2 := by sorry
  calc
    packets_per_smoothie * size_per_packet + packets_per_smoothie * water_per_packet
    = 1.2 * 3 + 1.2 * 15 : by rw packets_per_smoothie_calc
    ... = 3.6 + 18 : by sorry
    ... = 21.6 : by sorry

end smoothie_size_l10_10859


namespace intersection_point_C1_C2_minimum_distance_AB_l10_10275

-- Define the curves C1, C2, and C3

def C1 (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α ^ 2)

def C2 (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - π / 4) = -Real.sqrt 2 / 2

def C3 (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- The first theorem: proving the intersection point of C1 and C2
theorem intersection_point_C1_C2 :
  ∃ α ρ θ, C1 α = (-1, 0) ∧ (C2 ρ θ) ∧
    (Real.cos α ≠ 2) := by
  sorry

-- The second theorem: proving the minimum distance |AB|
theorem minimum_distance_AB :
  let A := {a | ∃ ρ θ, C2 ρ θ ∧ a = (ρ * Real.cos θ, ρ * Real.sin θ)},
  let B := {b | ∃ ρ θ, C3 ρ θ ∧ b = (ρ * Real.cos θ, ρ * Real.sin θ)},
  (∃ a ∈ A, b ∈ B, ∀ (d : ℝ), d ≥ Real.sqrt 2 - 1 → d = |a - b|) := by
  sorry

end intersection_point_C1_C2_minimum_distance_AB_l10_10275


namespace largest_multiple_of_7_smaller_than_neg_85_l10_10424

theorem largest_multiple_of_7_smaller_than_neg_85 :
  ∃ k : ℤ, 7 * k < -85 ∧ (∀ m : ℤ, 7 * m < -85 → 7 * m ≤ 7 * k) ∧ 7 * k = -91 :=
by
  simp only [exists_prop, and.assoc],
  sorry

end largest_multiple_of_7_smaller_than_neg_85_l10_10424


namespace largest_prime_factor_of_1769_l10_10019

theorem largest_prime_factor_of_1769 :
  ∃ p ∈ ({x | nat.prime x ∧ (x = 3 ∨ x = 19 ∨ x = 31)} : set ℕ),
    ∀ q ∈ ({x | nat.prime x ∧ x ∣ 1769} : set ℕ), q ≤ p :=
begin
  sorry
end

end largest_prime_factor_of_1769_l10_10019


namespace sin_minus_cos_eq_sqrt3_div2_l10_10190

theorem sin_minus_cos_eq_sqrt3_div2
  (α : ℝ) 
  (h_range : (Real.pi / 4) < α ∧ α < (Real.pi / 2))
  (h_sincos : Real.sin α * Real.cos α = 1 / 8) :
  Real.sin α - Real.cos α = Real.sqrt 3 / 2 :=
by
  sorry

end sin_minus_cos_eq_sqrt3_div2_l10_10190


namespace conjugate_of_z_l10_10620

open Complex

theorem conjugate_of_z 
  (z : ℂ) 
  (h : (1 + I) * z = 2 * I) : 
  conj z = 1 - I := 
  sorry

end conjugate_of_z_l10_10620


namespace smallest_positive_solution_l10_10145

theorem smallest_positive_solution (x : ℝ) (h : tan (4 * x) + tan (5 * x) = sec (5 * x)) : x = Real.pi / 18 :=
sorry

end smallest_positive_solution_l10_10145


namespace grid_area_with_circles_l10_10256

theorem grid_area_with_circles :
  let small_square_side := 3 in
  let grid_side := 4 * small_square_side in
  let total_area_grid := grid_side * grid_side in
  let radius_large_circle := grid_side / 2 in
  let area_large_circle := Real.pi * (radius_large_circle ^ 2) in
  let radius_small_circle := 1.5 in
  let area_small_circle := Real.pi * (radius_small_circle ^ 2) in
  let total_area_small_circles := 3 * area_small_circle in
  let visible_shaded_area := total_area_grid - area_large_circle - total_area_small_circles in
  let A := 144 in
  let B := 42.75 in
  A + B = 186.75 :=
by {
  -- Definitions from the conditions
  let small_square_side := 3,
  let grid_side := 4 * small_square_side,
  let total_area_grid := grid_side * grid_side,
  let radius_large_circle := grid_side / 2,
  let area_large_circle := Real.pi * (radius_large_circle ^ 2),
  let radius_small_circle := 1.5,
  let area_small_circle := Real.pi * (radius_small_circle ^ 2),
  let total_area_small_circles := 3 * area_small_circle,
  let visible_shaded_area := total_area_grid - area_large_circle - total_area_small_circles,
  -- Sum of A and B
  let A := 144,
  let B := 42.75,
  -- Expected result
  have : A + B = 186.75, sorry,
  exact this
}

end grid_area_with_circles_l10_10256


namespace possible_values_of_m_l10_10346

-- Proposition: for all real values of m, if for all real x, x^2 + 2x + 2 - m >= 0 holds, then m must be one of -1, 0, or 1

theorem possible_values_of_m (m : ℝ) 
  (h : ∀ (x : ℝ), x^2 + 2 * x + 2 - m ≥ 0) : m = -1 ∨ m = 0 ∨ m = 1 :=
sorry

end possible_values_of_m_l10_10346


namespace candle_remaining_length_l10_10834

-- Define the initial length of the candle and the burn rate
def initial_length : ℝ := 20
def burn_rate : ℝ := 5

-- Define the remaining length function
def remaining_length (t : ℝ) : ℝ := initial_length - burn_rate * t

-- Prove the relationship between time and remaining length for the given range of time
theorem candle_remaining_length (t : ℝ) (ht: 0 ≤ t ∧ t ≤ 4) : remaining_length t = 20 - 5 * t :=
by
  dsimp [remaining_length]
  sorry

end candle_remaining_length_l10_10834


namespace minimum_value_expression_l10_10587

theorem minimum_value_expression
  (x y t z : ℝ)
  (h1 : x + 4 * y = 4)
  (h2 : y > 0)
  (h3 : 0 < t)
  (h4 : t < z) :
  ( ∀ x y t z, (x + 4 * y = 4) ∧ (y > 0) ∧ (0 < t) ∧ (t < z) -> 
    (4 * z ^ 2 / |x| + |x * z ^ 2| / y + 12 / (t * (z - t)))) ≥ 24 := 
sorry

end minimum_value_expression_l10_10587


namespace count_square_free_integers_l10_10232

def square_free_in_range_2_to_199 : Nat :=
  91

theorem count_square_free_integers :
  ∃ n : Nat, n = 91 ∧
  ∀ m : Nat, 2 ≤ m ∧ m < 200 →
  (∀ k : Nat, k^2 ∣ m → k^2 = 1) :=
by
  -- The proof will be filled here
  sorry

end count_square_free_integers_l10_10232


namespace swap_rows_and_columns_l10_10016

open Matrix

noncomputable def swap_plus_minus (n : ℕ) (T : Matrix (Fin n) (Fin n) ℤ) : Prop :=
  (∀ i, Finset.card { j | T i j = 1 } = 1 ∧ Finset.card { j | T i j = -1 } = 1) ∧
  (∀ j, Finset.card { i | T i j = 1 } = 1 ∧ Finset.card { i | T i j = -1 } = 1)

theorem swap_rows_and_columns (n : ℕ) (T : Matrix (Fin n) (Fin n) ℤ) :
  swap_plus_minus n T →
  ∃ σ π : (Fin n → Fin n),
  ∀ i j, T (σ i) (π j) = -T i j :=
sorry

end swap_rows_and_columns_l10_10016


namespace find_positive_integer_l10_10911

theorem find_positive_integer (n : ℕ) (h1 : -46 ≤ 2023 / (46 - n)) (h2 : 2023 / (46 - n) ≤ 46 - n) : n = 90 :=
sorry

end find_positive_integer_l10_10911


namespace vector_subtraction_result_l10_10950

open Matrix

/-- Define the vectors a and b -/
noncomputable def a : Fin 3 → ℤ := ![-5, 3, 2]
noncomputable def b : Fin 3 → ℤ := ![2, -1, 4]

/-- Main theorem stating the expected result of the vector subtraction -/
theorem vector_subtraction_result : 
  let c := finVecSub (finVecAddConst a (-5)) (finScalarMul b 5)
  c = ![-15, 8, -18] :=
by
  sorry

end vector_subtraction_result_l10_10950


namespace epicenter_distance_l10_10123

noncomputable def distance_from_epicenter (v1 v2 Δt: ℝ) : ℝ :=
  Δt / ((1 / v2) - (1 / v1))

theorem epicenter_distance : 
  distance_from_epicenter 5.94 3.87 11.5 = 128 := 
by
  -- The proof will use calculations shown in the solution.
  sorry

end epicenter_distance_l10_10123


namespace rectangle_enclosed_by_four_lines_l10_10556

theorem rectangle_enclosed_by_four_lines : 
  let h_lines := 5
  let v_lines := 5
  (choose h_lines 2) * (choose v_lines 2) = 100 :=
by {
  sorry
}

end rectangle_enclosed_by_four_lines_l10_10556


namespace distance_with_father_l10_10694

variable (total_distance driven_with_mother driven_with_father: ℝ)

theorem distance_with_father :
  total_distance = 0.67 ∧ driven_with_mother = 0.17 → driven_with_father = 0.50 := 
by
  sorry

end distance_with_father_l10_10694


namespace chips_probability_l10_10050

theorem chips_probability :
  let total_chips := 14
  let tan_chips := 4
  let pink_chips := 3
  let violet_chips := 5
  let green_chips := 2
  let total_draws := total_chips!
  let tan_arrangements := tan_chips!
  let pink_arrangements := pink_chips!
  let violet_arrangements := violet_chips!
  let group_arrangements := 6
  let num_valid_draws := (tan_arrangements * pink_arrangements * violet_arrangements * group_arrangements)
  num_valid_draws.toNat / total_draws.toNat = 1440 / total_draws.toNat := 
sorry

end chips_probability_l10_10050


namespace fraction_equality_l10_10238

-- Defining the main problem statement
theorem fraction_equality (x y z : ℚ) (k : ℚ) 
  (h1 : x = 3 * k) (h2 : y = 5 * k) (h3 : z = 7 * k) :
  (y + z) / (3 * x - y) = 3 :=
by
  sorry

end fraction_equality_l10_10238


namespace find_point_with_given_volumes_l10_10277

open_locale affine
open_locale big_operators

variables {V : Type*} [normed_add_comm_group V] [normed_space ℝ V] [finite_dimensional ℝ V]
variables {P : Type*} [affine_space V P]

def volumes (α β γ δ : ℝ) (A B C D P : P) : Prop :=
  ∃ (Vol_PBCD Vol_PCDA Vol_PDAB Vol_PABC : ℝ), 
  Vol_PBCD / Vol_PCDA = α / β ∧
  Vol_PBCD / Vol_PDAB = α / γ ∧
  Vol_PBCD / Vol_PABC = α / δ ∧
  Vol_PCDA / Vol_PDAB = β / γ ∧
  Vol_PCDA / Vol_PABC = β / δ ∧
  Vol_PDAB / Vol_PABC = γ / δ
  
theorem find_point_with_given_volumes 
  {A B C D : P} (α β γ δ : ℝ) (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) (hδ : 0 < δ) :
  ∃ P : P, volumes α β γ δ A B C D P := sorry

end find_point_with_given_volumes_l10_10277


namespace rectangles_in_grid_squares_in_grid_l10_10036

theorem rectangles_in_grid (h_lines : ℕ) (v_lines : ℕ) : h_lines = 31 → v_lines = 31 → 
  (∃ rect_count : ℕ, rect_count = 216225) :=
by
  intros h_lines_eq v_lines_eq
  sorry

theorem squares_in_grid (n : ℕ) : n = 31 → (∃ square_count : ℕ, square_count = 6975) :=
by
  intros n_eq
  sorry

end rectangles_in_grid_squares_in_grid_l10_10036


namespace evaluate_expression_find_cos_beta_l10_10830

theorem evaluate_expression :
  log 5 + log 2 + ((3/5: ℝ)^0) + real.log (real.exp (1 / 2)) = 5 / 2 :=
  sorry

theorem find_cos_beta (α β : ℝ)
  (h1 : cos α = (2 * real.sqrt 2) / 3)
  (h2 : sin (α + β) = 1 / 3)
  (hα : 0 < α ∧ α < real.pi / 2)
  (hβ : real.pi / 2 < β ∧ β < real.pi) :
  cos β = -13 / 12 :=
  sorry

end evaluate_expression_find_cos_beta_l10_10830


namespace rectangle_enclosure_l10_10549
open BigOperators

theorem rectangle_enclosure (n m : ℕ) (hn : n = 5) (hm : m = 5) : 
  (∑ i in finset.range n, ∑ j in finset.range i, 1) * 
  (∑ k in finset.range m, ∑ l in finset.range k, 1) = 100 := by
  sorry

end rectangle_enclosure_l10_10549


namespace find_cosA_find_c_l10_10279

noncomputable def triangle_ABC (a b : ℝ) (B A : ℝ) (cosA c : ℝ) :=
  a = 3 ∧ b = 2 * Real.sqrt 6 ∧ B = 2 * A ∧ cosA = Real.sqrt 6 / 3 ∧
  √(b^2 + c^2 - 2 * b * c * cosA) = a

theorem find_cosA : ∀ (a b B A cosA : ℝ), triangle_ABC a b B A cosA 5 → cosA = Real.sqrt 6 / 3 :=
  by
    intros a b B A cosA h
    cases h with _ h1
    cases h1 with _ h2
    cases h2 with _ h3
    cases h3 with h_cond _
    exact h_cond

theorem find_c : ∀ (a b B A cosA c : ℝ), triangle_ABC a b B A cosA c → c = 5 :=
  by
    intros a b B A cosA c h
    cases h with _ h1
    cases h1 with _ h2
    cases h2 with _ h3
    cases h3 with _ h_cond
    exact h_cond

# To prove the conditions that validate the solution, one should complete the proof details, which are represented by 'sorry' in this context.

end find_cosA_find_c_l10_10279


namespace min_value_of_expr_l10_10137

def expr (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10

theorem min_value_of_expr : ∃ x y : ℝ, expr x y = -2 / 3 :=
by
  sorry

end min_value_of_expr_l10_10137


namespace part_one_option_one_cost_part_one_option_two_cost_part_two_more_cost_effective_part_three_purchase_plan_l10_10835

/-- Define the cost functions for the given problem conditions --/

def tea_set_price : ℕ := 200
def tea_cup_price : ℕ := 20
def discount : ℝ := 0.05
def num_tea_sets : ℕ := 30

/-- Calculate cost in option one: 30 tea sets and x tea cups --/
def option_one_cost (x : ℕ) : ℕ :=
  let free_cups := num_tea_sets
  let paid_cups := x - free_cups
  (num_tea_sets * tea_set_price) + (paid_cups * tea_cup_price)

/-- Calculate cost in option two: 30 tea sets and x tea cups --/
def option_two_cost (x : ℕ) : ℕ :=
  let total_sets_cost := num_tea_sets * tea_set_price
  let total_cups_cost := x * tea_cup_price
  (total_sets_cost + total_cups_cost) * (1 - discount)

theorem part_one_option_one_cost (x : ℕ) (h : x > 30):
  option_one_cost x = 20 * x + 5400 := sorry

theorem part_one_option_two_cost (x : ℕ) (h : x > 30):
  option_two_cost x = 19 * x + 5700 := sorry

theorem part_two_more_cost_effective (x : ℕ) (h : x = 50):
  option_one_cost x < option_two_cost x := sorry

theorem part_three_purchase_plan (x : ℕ) (h : x = 50):
  let total_budget : ℕ := 6380
  (option_one_cost x <= total_budget ∨
  (30 * tea_set_price + 30 * tea_cup_price + 20 * tea_cup_price <= total_budget)) := sorry

end part_one_option_one_cost_part_one_option_two_cost_part_two_more_cost_effective_part_three_purchase_plan_l10_10835


namespace Zilla_spends_rent_is_133_l10_10446

-- Conditions
def ZillaEarnings : Type := ℝ
def ZillaSpendsOnRent (E : ZillaEarnings) : ℝ := 0.07 * E
def ZillaSpendsOnExpenses (E : ZillaEarnings) : ℝ := 0.5 * E
def ZillaSaves (E : ZillaEarnings) : ℝ := E - 0.07 * E - 0.5 * E

-- Given condition for savings
axiom savings_condition (E : ZillaEarnings) : ZillaSaves E = 817

-- Theorem to prove
theorem Zilla_spends_rent_is_133 :
  ∃ E : ZillaEarnings, ZillaSpendsOnRent E = 133 := by
  sorry

end Zilla_spends_rent_is_133_l10_10446


namespace smallest_positive_angle_l10_10117

theorem smallest_positive_angle (φ : ℝ) (h1: cos (10 * (Real.pi / 180)) = sin (30 * (Real.pi / 180)) + sin φ) :
  φ = 70 * (Real.pi / 180) :=
by
  sorry

end smallest_positive_angle_l10_10117


namespace vector_magnitude_proof_l10_10201

noncomputable def magnitude_with_scale (a b : ℝ) (angle_ab : ℝ) :=
  real.sqrt (a ^ 2 + (2 * b) ^ 2 + 2 * a * (2 * b) * real.cos angle_ab)

noncomputable def vec1 := 2
noncomputable def vec2 := 1
noncomputable def angle := real.pi / 3 -- angle in radians as 60 degrees = π/3 radians

theorem vector_magnitude_proof : magnitude_with_scale vec1 vec2 angle = 2 * real.sqrt 3 := 
by
  sorry

end vector_magnitude_proof_l10_10201


namespace vertical_line_intersect_parabola_ex1_l10_10087

theorem vertical_line_intersect_parabola_ex1 (m : ℝ) (h : ∀ y : ℝ, (-4 * y^2 + 2*y + 3 = m) → false) :
  m = 13 / 4 :=
sorry

end vertical_line_intersect_parabola_ex1_l10_10087


namespace first_player_not_lose_first_player_not_win_l10_10825

def first_player_strategy_exists (board : array (array char)) : Prop :=
  ∃ strategy, ∀ K H : ℕ, K - H ≥ 0

def first_player_no_winning_strategy (board : array (array char)) : Prop :=
  ∀ strategy, ¬ (∃ K H : ℕ, K - H > 0)

-- Define the conditions
structure Board :=
  (size : ℕ)
  (content : matrix (fin size) (fin size) char)

-- assert conditions applied on the board
noncomputable def TicTacToeBoard := Board.mk 10 (λ _ _, ' ')

-- Equivalent Lean statements
theorem first_player_not_lose : first_player_strategy_exists TicTacToeBoard :=
by
  sorry

theorem first_player_not_win : first_player_no_winning_strategy TicTacToeBoard :=
by
  sorry

end first_player_not_lose_first_player_not_win_l10_10825


namespace fraction_of_income_from_tips_is_7_over_11_l10_10856

variable (S : ℝ)

-- The tips were 7/4 of the salary
def T : ℝ := (7/4) * S

-- The total income
def I : ℝ := S + T

-- The fraction of income that came from tips
def fraction_of_income_from_tips : ℝ := T / I

theorem fraction_of_income_from_tips_is_7_over_11 
  (hT : T = (7/4) * S)
  (hI : I = S + T) :
  fraction_of_income_from_tips = 7 / 11 :=
by
  sorry

end fraction_of_income_from_tips_is_7_over_11_l10_10856


namespace quadratic_roots_l10_10373

theorem quadratic_roots {α p q : ℝ} (hα : 0 < α ∧ α ≤ 1) (hroots : ∃ x : ℝ, x^2 + p * x + q = 0) :
  ∃ x : ℝ, α * x^2 + p * x + q = 0 :=
by sorry

end quadratic_roots_l10_10373


namespace jerry_age_l10_10325

theorem jerry_age
  (M J : ℕ)
  (h1 : M = 2 * J + 5)
  (h2 : M = 21) :
  J = 8 :=
by
  sorry

end jerry_age_l10_10325


namespace rectangle_count_l10_10560

theorem rectangle_count (h_lines v_lines : Finset ℕ) (h_card : h_lines.card = 5) (v_card : v_lines.card = 5) :
  ∃ (n : ℕ), n = (h_lines.choose 2).card * (v_lines.choose 2).card ∧ n = 100 :=
by
  sorry 

end rectangle_count_l10_10560


namespace markup_percentage_is_ten_l10_10775

theorem markup_percentage_is_ten (S C : ℝ)
  (h1 : S - C = 0.0909090909090909 * S) :
  (S - C) / C * 100 = 10 :=
by
  sorry

end markup_percentage_is_ten_l10_10775


namespace inequality_proof_l10_10216

theorem inequality_proof (a b : ℝ) (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (sum_eq_one : a + b = 1) :
  (1 / 2 : ℝ) ≤ (a^3 + b^3) / (a^2 + b^2) ∧ (a^3 + b^3) / (a^2 + b^2) ≤ 1 := by
  -- Provided conditions
  have h1 : 0 ≤ a := non_neg_a,
  have h2 : 0 ≤ b := non_neg_b,
  have h3 : a + b = 1 := sum_eq_one,
  
  -- Introducing the inequality to be proven
  sorry

end inequality_proof_l10_10216


namespace polynomial_solution_characterization_l10_10910

theorem polynomial_solution_characterization (P : ℝ → ℝ → ℝ) (h : ∀ x y z : ℝ, P x (2 * y * z) + P y (2 * z * x) + P z (2 * x * y) = P (x + y + z) (x * y + y * z + z * x)) :
  ∃ (a b : ℝ), ∀ x y : ℝ, P x y = a * x + b * (x^2 + 2 * y) :=
sorry

end polynomial_solution_characterization_l10_10910


namespace greatest_value_of_x_is_20_l10_10366

noncomputable def greatest_multiple_of_4 (x : ℕ) : Prop :=
  (x % 4 = 0 ∧ x^2 < 500 ∧ ∀ y : ℕ, (y % 4 = 0 ∧ y^2 < 500) → y ≤ x)

theorem greatest_value_of_x_is_20 : greatest_multiple_of_4 20 :=
  by 
  sorry

end greatest_value_of_x_is_20_l10_10366


namespace sufficient_condition_not_necessary_condition_x1_plus_x2_eq_0_sufficient_not_necessary_l10_10195

-- Define what it means for a function to be odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- State the sufficient condition
theorem sufficient_condition 
  (f : ℝ → ℝ) (h₁ : is_odd_function f) (x₁ x₂ : ℝ) (h₂ : x₁ + x₂ = 0) :
  f(x₁) + f(x₂) = 0 := 
by 
  sorry

-- State the necessity failure
theorem not_necessary_condition 
  (f : ℝ → ℝ) (h₁ : is_odd_function f) :
  ¬∀ x₁ x₂ : ℝ, (f(x₁) + f(x₂) = 0) →  (x₁ + x₂ = 0) :=
by 
  sorry

-- Combine both theorems to describe the condition
theorem x1_plus_x2_eq_0_sufficient_not_necessary
  (f : ℝ → ℝ) (h₁ : is_odd_function f) :
  (∀ x₁ x₂ : ℝ, (x₁ + x₂ = 0) → (f(x₁) + f(x₂) = 0)) ∧ 
  ¬ (∀ x₁ x₂ : ℝ, (f(x₁) + f(x₂) = 0) →  (x₁ + x₂ = 0)) :=
by 
  sorry

end sufficient_condition_not_necessary_condition_x1_plus_x2_eq_0_sufficient_not_necessary_l10_10195


namespace find_p_plus_q_l10_10364

theorem find_p_plus_q (x p q : ℝ) (h1 : Real.sec x + Real.tan x = 15/4)
  (h2 : Real.csc x + Real.cot x = p/q) (h3 : Nat.gcd p.nat_abs q.nat_abs = 1) : p + q = 390 := 
sorry

end find_p_plus_q_l10_10364


namespace cross_product_distributive_l10_10240

variables (a b c : ℝ × ℝ × ℝ)
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem cross_product_distributive
  (h1 : cross_product a b = (3, -2, 5))
  (h2 : c = (1, 0, -1)) :
  cross_product a (5 • b + 2 • c) = (19, -8, 19) :=
sorry

end cross_product_distributive_l10_10240


namespace Gretchen_performance_Mitzi_performance_Beth_performance_Lucy_performance_l10_10230

namespace BowlingPerformance

def max_score : ℝ := 300

def gret : ℝ := 120.5
def mitzi : ℝ := 113.2
def beth : ℝ := 85.8
def lucy : ℝ := 101.6

def performance_percentage (score : ℝ) : ℝ :=
  (score / max_score) * 100

theorem Gretchen_performance : performance_percentage gret = 40.17 := 
by
  simp [gret, performance_percentage, max_score]
  sorry

theorem Mitzi_performance : performance_percentage mitzi = 37.73 := 
by
  simp [mitzi, performance_percentage, max_score]
  sorry

theorem Beth_performance : performance_percentage beth = 28.60 := 
by
  simp [beth, performance_percentage, max_score]
  sorry

theorem Lucy_performance : performance_percentage lucy = 33.87 := 
by
  simp [lucy, performance_percentage, max_score]
  sorry

end BowlingPerformance

end Gretchen_performance_Mitzi_performance_Beth_performance_Lucy_performance_l10_10230


namespace remainder_sum_first_150_div_11250_l10_10437

theorem remainder_sum_first_150_div_11250 : 
  let S := 150 * 151 / 2 
  in S % 11250 = 75 := 
by
  let S := 11325
  have hSum : S = 11325 := by rfl
  show S % 11250 = 75
  sorry

end remainder_sum_first_150_div_11250_l10_10437


namespace count_valid_numbers_l10_10643

def digit_set : Finset ℕ := {3, 4, 6, 7, 8, 9}

def is_valid_digit (n : ℕ) : Prop := n ∈ digit_set

def is_valid_number (n : ℕ) : Prop :=
  300 ≤ n ∧ n < 800 ∧ n % 2 = 0 ∧ (n.digits 10).Nodup ∧ (∀ d ∈ n.digits 10, d ∈ digit_set)

theorem count_valid_numbers : Finset.card ((Finset.Icc 300 800).filter is_valid_number) = 30 :=
sorry

end count_valid_numbers_l10_10643


namespace son_present_age_l10_10844

-- Definitions
variables (S M : ℕ)
-- Conditions
def age_diff : Prop := M = S + 22
def future_age_condition : Prop := M + 2 = 2 * (S + 2)

-- Theorem statement with proof placeholder
theorem son_present_age (H1 : age_diff S M) (H2 : future_age_condition S M) : S = 20 :=
by sorry

end son_present_age_l10_10844


namespace fisherman_catch_total_l10_10469

theorem fisherman_catch_total :
  let bass := 32
  let trout := bass / 4
  let blue_gill := bass * 2
in bass + trout + blue_gill = 104 := by
  sorry

end fisherman_catch_total_l10_10469


namespace Victor_Total_Money_l10_10411

-- Definitions for the conditions
def originalAmount : Nat := 10
def allowance : Nat := 8

-- The proof problem statement
theorem Victor_Total_Money : originalAmount + allowance = 18 := by
  sorry

end Victor_Total_Money_l10_10411


namespace three_digit_sum_to_25_l10_10164

theorem three_digit_sum_to_25 : 
  {n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ (∃ (a b c : ℕ), a + b + c = 25 ∧ 100 * a + 10 * b + c = n ∧ a ≠ 0)}.card = 6 :=
by
  sorry

end three_digit_sum_to_25_l10_10164


namespace smallest_positive_solution_l10_10144

theorem smallest_positive_solution (x : ℝ) (h : tan (4 * x) + tan (5 * x) = sec (5 * x)) : x = Real.pi / 18 :=
sorry

end smallest_positive_solution_l10_10144


namespace range_of_a_l10_10716

theorem range_of_a (a : ℝ) (x : ℝ) :
  (x^2 - 4 * a * x + 3 * a^2 < 0 → (x^2 - x - 6 ≤ 0 ∨ x^2 + 2 * x - 8 > 0)) → a < 0 → 
  (a ≤ -4 ∨ -2 / 3 ≤ a ∧ a < 0) :=
by
  sorry

end range_of_a_l10_10716


namespace original_number_is_correct_l10_10480

theorem original_number_is_correct (x : ℝ) (h : 10 * x = x + 34.65) : x = 3.85 :=
sorry

end original_number_is_correct_l10_10480


namespace find_manager_salary_l10_10815

-- Definitions based on conditions in the problem
def average_salary : ℕ := 2400
def num_of_employees : ℕ := 24
def increase_in_average_salary : ℕ := 100

-- Derived definitions from given conditions
def manager_salary : ℕ := 4900

-- Theorem: Prove the manager's salary given the conditions
theorem find_manager_salary (average_salary : ℕ) (num_of_employees : ℕ) (increase_in_average_salary : ℕ) : ℕ :=
(average_salary * (num_of_employees + 1) + increase_in_average_salary * (num_of_employees + 1) - (average_salary * num_of_employees)) sorry


end find_manager_salary_l10_10815


namespace symmetry_with_respect_to_origin_l10_10901

def f (x : ℝ) : ℝ := x - 2 / x

theorem symmetry_with_respect_to_origin (x : ℝ) : f (-x) = -f x :=
by 
  -- Proof steps would go here
  sorry

end symmetry_with_respect_to_origin_l10_10901


namespace smallest_integer_N_with_property_l10_10543

theorem smallest_integer_N_with_property :
  ∃ (N : ℕ), N = 392 ∧
  ((∃ (k : ℕ), k ∈ {N, N+1, N+2, N+3} ∧ 2^3 ∣ k) ∧
   (∃ (k : ℕ), k ∈ {N, N+1, N+2, N+3} ∧ 3^3 ∣ k) ∧
   (∃ (k : ℕ), k ∈ {N, N+1, N+2, N+3} ∧ 5^2 ∣ k) ∧
   (∃ (k : ℕ), k ∈ {N, N+1, N+2, N+3} ∧ 7^2 ∣ k)) :=
begin
  sorry
end

end smallest_integer_N_with_property_l10_10543


namespace coordinates_of_B_l10_10949
open Real

-- Define the conditions given in the problem
def A : ℝ × ℝ := (1, 6)
def d : ℝ := 4

-- Define the properties of the solution given the conditions
theorem coordinates_of_B (B : ℝ × ℝ) :
  (B = (-3, 6) ∨ B = (5, 6)) ↔
  (B.2 = A.2 ∧ (B.1 = A.1 - d ∨ B.1 = A.1 + d)) :=
by
  sorry

end coordinates_of_B_l10_10949


namespace dianas_roll_beats_apollos_max_l10_10120

/-- Define that Diana rolls a six-sided die -/
def DianaRolls := Fin 6

/-- Define that Apollo rolls two six-sided dice -/
def ApolloRolls := (Fin 6 × Fin 6)

/-- The probability that Diana's single die roll is higher than the maximum of Apollo's two rolls -/
def DianaBeatsApollo :=
  let outcomes : List (Fin 6 × Fin 6) := List.diag [0, 1, 2, 3, 4, 5]
  let diana_probability := 1 // 6
  let apollo_probabilities := [
    (1 / 36 : ℚ), (3 / 36 : ℚ),
    (5 / 36 : ℚ), (5 / 36 : ℚ),
    (7 / 36 : ℚ), (11 / 36 : ℚ)
  ]
  let beat_probability :=
    diana_probability * (apollo_probabilities.sum)
  beat_probability

/-- The proof that Diana's single die roll results in a higher number than the maximum of Apollo's two rolls
    has the probability 95/216 -/
theorem dianas_roll_beats_apollos_max :
  DianaBeatsApollo = (95 / 216 : ℚ) := sorry

end dianas_roll_beats_apollos_max_l10_10120


namespace find_a_plus_c_find_angle_B_l10_10280

variables (a b c : ℝ) (A B C : ℝ)

-- Condition: Sides a, b, c are opposite to angles A, B, C respectively in triangle ABC
-- Condition and given equation: b * cos C = (2 * a + c) * cos (π - B)
def triangle_condition (a b c A B C : ℝ) : Prop :=
  b * Real.cos C = (2 * a + c) * Real.cos (Real.pi - B)

-- Given b = √13, S = (3√3)/4, prove that a + c = 4
theorem find_a_plus_c (a b c A B C : ℝ)
  (h1 : b = Real.sqrt 13)
  (h2 : 0 < B ∧ B < Real.pi)
  (h3 : ∀ (a b c A B C : ℝ), triangle_condition a b c A B C)
  (h4 : S := 3 * Real.sqrt 3 / 4) :
  a + c = 4 :=
by
  sorry

-- Prove that B = 2π/3 under the given conditions
theorem find_angle_B (a b c A B C : ℝ)
  (h1 : triangle_condition a b c A B C)
  (h2 : ∀ {A B C : ℝ}, b * Real.sin C = ((2 * a + c) * Real.sin (Real.pi - B))) :
  B = 2 * Real.pi / 3 :=
by
  sorry

end find_a_plus_c_find_angle_B_l10_10280


namespace min_colored_cells_l10_10432

-- Defining the problem statement
theorem min_colored_cells (m n : ℕ) (h_m : m = 3) (h_n : n = 2016)
  (adjacent_colored : ∀ i j : ℕ, (i < m ∧ j < n) → ∃ i' j', (abs (i - i') ≤ 1 ∧ abs (j - j') ≤ 1) ∧ (i' < m ∧ j' < n) ∧ (i' ≠ i ∨ j' ≠ j)) :
  ∃ num_colored : ℕ, num_colored = 2016 :=
by
  sorry

end min_colored_cells_l10_10432


namespace three_consecutive_arithmetic_l10_10935

def seq (n : ℕ) : ℝ := 
  if n % 2 = 1 then (n : ℝ)
  else 2 * 3^(n / 2 - 1)

theorem three_consecutive_arithmetic (m : ℕ) (h_m : seq m + seq (m+2) = 2 * seq (m+1)) : m = 1 :=
  sorry

end three_consecutive_arithmetic_l10_10935


namespace trapezoid_EFGH_area_l10_10900

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def distance (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

noncomputable def trapezoid_area (A B C D : Point) : ℝ :=
  let height := distance A C
  let base1 := distance A B
  let base2 := distance C D
  (base1 + base2) / 2 * height

theorem trapezoid_EFGH_area :
  let E := Point.mk 0 0
  let F := Point.mk 0 (-3)
  let G := Point.mk 6 0
  let H := Point.mk 6 8
  trapezoid_area E F G H = 33 :=
by 
  let E := Point.mk 0 0
  let F := Point.mk 0 (-3)
  let G := Point.mk 6 0
  let H := Point.mk 6 8
  show trapezoid_area E F G H = 33 
  sorry

end trapezoid_EFGH_area_l10_10900


namespace train_length_is_300_l10_10084

def speed_km_per_hr : ℝ := 90
def time_sec : ℝ := 12
def conversion_factor : ℝ := 5 / 18

theorem train_length_is_300 :
  (speed_km_per_hr * (conversion_factor * time_sec) = 300) :=
by
  sorry

end train_length_is_300_l10_10084


namespace rational_inequality_solution_set_l10_10930

theorem rational_inequality_solution_set :
  {x : ℝ | (x^2 + 2 * x + 2) / (x + 2) > 1} = {x : ℝ | (-2 < x ∧ x < -1) ∨ (0 < x ∧ x)} :=
by
  sorry

end rational_inequality_solution_set_l10_10930


namespace percentage_of_orange_and_watermelon_juice_l10_10841

-- Define the total volume of the drink
def total_volume := 150

-- Define the volume of grape juice in the drink
def grape_juice_volume := 45

-- Define the percentage calculation for grape juice
def grape_juice_percentage := (grape_juice_volume / total_volume) * 100

-- Define the remaining percentage that is made of orange and watermelon juices
def remaining_percentage := 100 - grape_juice_percentage

-- Define the percentage of orange and watermelon juice being the same
def orange_and_watermelon_percentage := remaining_percentage / 2

theorem percentage_of_orange_and_watermelon_juice : 
  orange_and_watermelon_percentage = 35 :=
by
  -- The proof steps would go here
  sorry

end percentage_of_orange_and_watermelon_juice_l10_10841


namespace indigo_restaurant_average_rating_l10_10763

theorem indigo_restaurant_average_rating :
  let n_5stars := 6
  let n_4stars := 7
  let n_3stars := 4
  let n_2stars := 1
  let total_reviews := 18
  let total_stars := n_5stars * 5 + n_4stars * 4 + n_3stars * 3 + n_2stars * 2
  (total_stars / total_reviews : ℝ) = 4 :=
by
  sorry

end indigo_restaurant_average_rating_l10_10763


namespace not_neighboring_root_equation_x2_x_2_neighboring_root_equation_k_values_l10_10116

def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁ * x₁ + b * x₁ + c = 0 ∧ a * x₂ * x₂ + b * x₂ + c = 0 
  ∧ (x₁ - x₂ = 1 ∨ x₂ - x₁ = 1)

theorem not_neighboring_root_equation_x2_x_2 : 
  ¬ is_neighboring_root_equation 1 1 (-2) :=
sorry

theorem neighboring_root_equation_k_values (k : ℝ) : 
  is_neighboring_root_equation 1 (-(k-3)) (-3*k) ↔ k = -2 ∨ k = -4 :=
sorry

end not_neighboring_root_equation_x2_x_2_neighboring_root_equation_k_values_l10_10116


namespace sector_area_ratio_l10_10339

theorem sector_area_ratio (A B C D O : Point)
  (hAOC : Angle A O C = 40)
  (hDOB : Angle D O B = 60)
  (hAOB : Angle A O B = 180)
  (h_same_side : SameSide A B C D) :
  sector_area_ratio O C D = 2 / 9 := 
by
  sorry

end sector_area_ratio_l10_10339


namespace magnitude_of_z_to_the_sixth_l10_10540

variable z : ℂ 
variable h : z = 2 + 2 * Real.sqrt 2 * Complex.I

theorem magnitude_of_z_to_the_sixth :
  |z^6| = 1728 :=
by
  sorry

end magnitude_of_z_to_the_sixth_l10_10540


namespace solve_inequality_1_range_of_m_l10_10214

noncomputable def f (x : ℝ) : ℝ := abs (x - 1)
noncomputable def g (x m : ℝ) : ℝ := -abs (x + 3) + m

theorem solve_inequality_1 : {x : ℝ | f x + x^2 - 1 > 0} = {x : ℝ | x > 1 ∨ x < 0} := sorry

theorem range_of_m (m : ℝ) (h : m > 4) : ∃ x : ℝ, f x < g x m := sorry

end solve_inequality_1_range_of_m_l10_10214


namespace angle_2016_in_third_quadrant_l10_10808

def quadrant (θ : ℤ) : ℤ :=
  let angle := θ % 360
  if 0 ≤ angle ∧ angle < 90 then 1
  else if 90 ≤ angle ∧ angle < 180 then 2
  else if 180 ≤ angle ∧ angle < 270 then 3
  else 4

theorem angle_2016_in_third_quadrant : 
  quadrant 2016 = 3 := 
by
  sorry

end angle_2016_in_third_quadrant_l10_10808


namespace min_distance_l10_10489

theorem min_distance (W : ℝ) (b : ℝ) (n : ℕ) (H_W : W = 42) (H_b : b = 3) (H_n : n = 8) : 
  ∃ d : ℝ, d = 2 ∧ (W - n * b = 9 * d) := 
by 
  -- Here should go the proof
  sorry

end min_distance_l10_10489


namespace percentage_to_decimal_l10_10033

theorem percentage_to_decimal (p : ℝ) (h : p = 3) : p / 100 = 0.03 := by
  rw h
  norm_num
  sorry

end percentage_to_decimal_l10_10033


namespace smaller_part_area_l10_10030

theorem smaller_part_area (x y : ℝ) (h1 : x + y = 500) (h2 : y - x = (1 / 5) * ((x + y) / 2)) : x = 225 :=
by
  sorry

end smaller_part_area_l10_10030


namespace sum_common_divisors_120_45_l10_10545

-- Define the relevant properties of 120 and 45
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

def divisors (n : ℕ) : set ℕ :=
{d | is_divisor d n}

def common_divisors (a b : ℕ) : set ℕ := 
{d | d ∈ divisors a ∧ d ∈ divisors b}

def sum_common_divisors (a b : ℕ) : ℕ := 
(common_divisors a b).to_finset.sum id

theorem sum_common_divisors_120_45 : sum_common_divisors 120 45 = 24 := 
by
simp only [sum_common_divisors, common_divisors, divisors]
sorry

end sum_common_divisors_120_45_l10_10545


namespace probability_not_in_same_group_l10_10490

theorem probability_not_in_same_group (n_groups : ℕ) (h : n_groups = 3) :
  let total_events := n_groups * n_groups,
      diff_groups_events := n_groups * (n_groups - 1)
  in
  (diff_groups_events : ℝ) / (total_events : ℝ) = (2 / 3 : ℝ) :=
by
  sorry

end probability_not_in_same_group_l10_10490


namespace range_s_l10_10433

theorem range_s (x : ℝ) : 
  let s := λ x, (1 / (1 - Real.cos x) ^ 2) in 
  ∃ (y : ℝ), y > 0 ∧ (y = s (x)) ↔ 1 / 4 < y ∧ y < ∞ :=
sorry

end range_s_l10_10433


namespace f_when_x_lt_4_l10_10618

noncomputable def f : ℝ → ℝ := sorry

theorem f_when_x_lt_4 (x : ℝ) (h1 : ∀ y : ℝ, y > 4 → f y = 2^(y-1)) (h2 : ∀ y : ℝ, f (4-y) = f (4+y)) (hx : x < 4) : f x = 2^(7-x) :=
by
  sorry

end f_when_x_lt_4_l10_10618


namespace collinear_points_vector_sum_coordinates_l10_10312

section collinearity_and_vector_sum

variables {a b : ℝ} (z : ℂ) (h_imaginary : z = a + b * complex.i) 
variables (h_conjugate : complex.conj z = a - b * complex.i)
variables (h_reciprocal : z⁻¹ = (a - b * complex.i) / (a^2 + b^2))

-- Statement (1): Prove that points O, B, C are collinear
theorem collinear_points 
  (h_vector_B : ∀ (a b : ℝ), complex.conj z = complex.mk a (-b))
  (h_vector_C : ∀ (a b : ℝ), z⁻¹ = complex.mk (a / (a^2 + b^2)) (-b / (a^2 + b^2))) 
  (a b : ℝ) :
  ∃ k : ℝ, (a, -b) = k • (a / (a^2 + b^2), -b / (a^2 + b^2)) :=
sorry

-- Statement (2): Find the coordinates of the vector OA + OC given z³ = 1
theorem vector_sum_coordinates
  (h_cubed : z^3 = 1) 
  (h_a : a = -1/2)
  (h_b_squared : b^2 = 3/4) :
  (2 * -1/2, 0) = (-1, 0) :=
sorry

end collinearity_and_vector_sum

end collinear_points_vector_sum_coordinates_l10_10312


namespace line_passes_through_vertex_twice_l10_10943

theorem line_passes_through_vertex_twice :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ (∀ a, (y = 2 * x + a ∧ ∃ (x y : ℝ), y = x^2 + 2 * a^2) ↔ a = a₁ ∨ a = a₂) :=
by
  sorry

end line_passes_through_vertex_twice_l10_10943


namespace functional_eq_zero_l10_10907

theorem functional_eq_zero (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f(x ^ 2022 + y) = f(x ^ 1747 + 2 * y) + f(x ^ 42)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_eq_zero_l10_10907


namespace acute_triangle_tangent_condition_l10_10283

theorem acute_triangle_tangent_condition (A B C : ℝ) (h_sum : A + B + C = π) :
  (tan B * tan C > 1 ↔ (A < π / 2 ∧ B < π / 2 ∧ C < π / 2)) :=
by sorry

end acute_triangle_tangent_condition_l10_10283


namespace combinedAgeIn5Years_l10_10322

variable (Amy Mark Emily : ℕ)

-- Conditions
def amyAge : ℕ := 15
def markAge : ℕ := amyAge + 7
def emilyAge : ℕ := 2 * amyAge

-- Proposition to be proved
theorem combinedAgeIn5Years :
  Amy = amyAge →
  Mark = markAge →
  Emily = emilyAge →
  (Amy + 5) + (Mark + 5) + (Emily + 5) = 82 :=
by
  intros hAmy hMark hEmily
  sorry

end combinedAgeIn5Years_l10_10322


namespace quadratic_roots_properties_quadratic_roots_max_min_l10_10207

theorem quadratic_roots_properties (k : ℝ) (h : 2 ≤ k ∧ k ≤ 8)
  (x1 x2 : ℝ) (h_roots : x1 + x2 = 2 * (k - 1) ∧ x1 * x2 = 2 * k^2 - 12 * k + 17) :
  (x1^2 + x2^2) = 16 * k - 30 :=
sorry

theorem quadratic_roots_max_min :
  (∀ k ∈ { k : ℝ | 2 ≤ k ∧ k ≤ 8 }, 
    ∃ (x1 x2 : ℝ), 
      (x1 + x2 = 2 * (k - 1) ∧ x1 * x2 = 2 * k^2 - 12 * k + 17) 
      ∧ (x1^2 + x2^2) = (if k = 8 then 98 else if k = 2 then 2 else 16 * k - 30)) :=
sorry

end quadratic_roots_properties_quadratic_roots_max_min_l10_10207


namespace angle_AFE_is_165_degrees_l10_10272

theorem angle_AFE_is_165_degrees (A B C D E F : Point)
  (square_ABCD : square A B C D) 
  (condition_E : opposite_half_plane C D E A) 
  (angle_CDE_120 : angle C D E = 120) 
  (F_on_AD : on_line F A D) 
  (DE_EQ_DF : distance D E = distance D F) : 
  angle A F E = 165 :=
sorry

end angle_AFE_is_165_degrees_l10_10272


namespace area_ratio_of_isosceles_triangle_l10_10700

variable (x : ℝ)
variable (hx : 0 < x)

def isosceles_triangle (AB AC : ℝ) (BC : ℝ) : Prop :=
  AB = AC ∧ AB = 2 * x ∧ BC = x

def extend_side (B_length AB_length : ℝ) : Prop :=
  B_length = 2 * AB_length

def ratio_of_areas (area_AB'B'C' area_ABC : ℝ) : Prop :=
  area_AB'B'C' / area_ABC = 9

theorem area_ratio_of_isosceles_triangle
  (AB AC BC : ℝ) (BB' B'C' area_ABC area_AB'B'C' : ℝ)
  (h_isosceles : isosceles_triangle x AB AC BC)
  (h_extend_A : extend_side BB' AB)
  (h_extend_C : extend_side B'C' AC) :
  ratio_of_areas area_AB'B'C' area_ABC := by
  sorry

end area_ratio_of_isosceles_triangle_l10_10700


namespace children_ticket_price_l10_10860

theorem children_ticket_price
  (C : ℝ)
  (adult_ticket_price : ℝ)
  (total_payment : ℝ)
  (total_tickets : ℕ)
  (children_tickets : ℕ)
  (H1 : adult_ticket_price = 8)
  (H2 : total_payment = 201)
  (H3 : total_tickets = 33)
  (H4 : children_tickets = 21)
  : C = 5 :=
by
  sorry

end children_ticket_price_l10_10860


namespace probability_all_digits_different_l10_10510

def integers_in_range := {x : ℕ | 100 ≤ x ∧ x ≤ 999}

def count_elements (s : set ℕ) : ℕ :=
  fintype.card {x // x ∈ s}

def has_same_digits (n : ℕ) : Prop :=
  let d0 := n % 10 in
  let d1 := (n / 10) % 10 in
  let d2 := (n / 100) % 10 in
  (d0 = d1) ∨ (d1 = d2) ∨ (d0 = d2)

theorem probability_all_digits_different :
  ∀ x ∈ integers_in_range, 
  (∃ n, n = fintype.card {k // k ∈ integers_in_range ∧ ¬ has_same_digits k}) →
  n.to_float / (count_elements integers_in_range).to_float = 0.99 :=
by 
  sorry

end probability_all_digits_different_l10_10510


namespace f_neg_a_l10_10377

variable {R : Type*} [RealRing R]

def f (x : R) : R := x^3 + x + 1

theorem f_neg_a (a : R) (h : f a = 2) : f (-a) = 0 := by
  sorry

end f_neg_a_l10_10377


namespace equilateral_triangle_points_distance_l10_10304

theorem equilateral_triangle_points_distance
  (A B C : ℝ × ℝ)
  (h_equilateral : dist A B = 2 * real.sqrt 3 ∧ dist B C = 2 * real.sqrt 3 ∧ dist C A = 2 * real.sqrt 3)
  (points : fin 11 → ℝ × ℝ)
  (h_points_in_triangle : ∀ i, ∃ (α β γ : ℝ), 0 ≤ α ∧ 0 ≤ β ∧ 0 ≤ γ ∧ α + β + γ = 1 ∧ points i = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)) :
  (∃ (i j : fin 11), i ≠ j ∧ dist (points i) (points j) ≤ 1) ∧
  (∃ (i j : fin 11), i ≠ j ∧ dist (points i) (points j) < 1) :=
sorry

end equilateral_triangle_points_distance_l10_10304


namespace mycroft_wins_higher_probability_l10_10355

theorem mycroft_wins_higher_probability :
  (prob_fair_coin_sequence_appears_first "HTT") > (prob_fair_coin_sequence_appears_first "TTT") := sorry

end mycroft_wins_higher_probability_l10_10355


namespace problem_statement_l10_10035

def non_intersecting_chords (k : ℕ) : Prop :=
  ∃ (chords : list (ℕ × ℕ)), 
    chords.length = 2 * k ∧
    ∀ (ch : ℕ × ℕ) ∈ chords, 
      let (a, b) := ch in
      |a - b| ≤ 3 * k - 1 ∧ 
      all_chords_non_intersecting (chords : list (ℕ × ℕ))

def minimal_bound (k : ℕ) : Prop :=
  ∀ (chords : list (ℕ × ℕ)), 
    chords.length = 2 * k ∧
    all_chords_non_intersecting (chords) →
    ∃ (ch : ℕ × ℕ) ∈ chords, 
      let (a, b) := ch in
      |a - b| ≥ 3 * k - 1

theorem problem_statement (k : ℕ) : non_intersecting_chords k ∧ minimal_bound k :=
by
  sorry

-- Helper function to determine if all chords are non-intersecting.
def all_chords_non_intersecting (chords : list (ℕ × ℕ)) : Prop :=
  ∀ (ch1 ch2 : ℕ × ℕ), 
    ch1 ≠ ch2 ∧ ch1 ∈ chords ∧ ch2 ∈ chords → 
    ¬ intersecting (ch1, ch2)

-- Helper function to determine if two chords intersect.
def intersecting (chords : (ℕ × ℕ) × (ℕ × ℕ)) : Prop :=
  let ((a1, b1), (a2, b2)) := chords in
  (a1 < a2 ∧ a2 < b1 ∧ b1 < b2) ∨
  (a2 < a1 ∧ a1 < b2 ∧ b2 < b1)


end problem_statement_l10_10035


namespace quadratic_m_ge_neg2_l10_10655

-- Define the quadratic equation and condition for real roots
def quadratic_has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, (x + 2) ^ 2 = m + 2

-- The theorem to prove
theorem quadratic_m_ge_neg2 (m : ℝ) (h : quadratic_has_real_roots m) : m ≥ -2 :=
by {
  sorry
}

end quadratic_m_ge_neg2_l10_10655


namespace mixed_operations_with_rationals_l10_10888

theorem mixed_operations_with_rationals :
  let a := 1 / 4
  let b := 1 / 2
  let c := 2 / 3
  (a - b + c) * (-12) = -8 :=
by
  sorry

end mixed_operations_with_rationals_l10_10888


namespace joe_bought_plants_l10_10292

def cost_of_oranges (n : ℕ) : ℝ := n * 4.50
def cost_of_juices (n : ℕ) : ℝ := n * 0.50
def cost_of_honey (n : ℕ) : ℝ := n * 5.0
def cost_of_plants (n : ℕ) : ℝ := (n / 2.0) * 18.0

theorem joe_bought_plants :
  ∀ (plants : ℕ), 
    cost_of_oranges 3 + cost_of_juices 7 + cost_of_honey 3 + cost_of_plants plants = 68.0 → plants = 4 :=
by
  intros plants h
  have h₁ : cost_of_oranges 3 = 13.5 := rfl
  have h₂ : cost_of_juices 7 = 3.5 := rfl
  have h₃ : cost_of_honey 3 = 15 := rfl
  have h₄ : 13.5 + 3.5 + 15 = 32 := by norm_num
  have h₅ : cost_of_plants plants = 36.0 := by linarith [h, h₄]
  have h₆ : (plants / 2 : ℝ) * 18.0 = 36 := h₅
  have h₇ : (plants / 2 : ℝ) = 2 := eq_of_mul_eq_mul_right (by norm_num) h₆
  have h₈ : plants = 4 := by linarith [h₇]
  exact h₈

end joe_bought_plants_l10_10292


namespace james_profit_calculation_l10_10286

theorem james_profit_calculation :
  ∀ (num_tickets : ℕ) (percentage_1_dollar : ℝ) (percentage_3_dollar : ℝ) (percentage_4_dollar : ℝ)
    (cost_1_dollar : ℝ) (cost_3_dollar : ℝ) (cost_4_dollar : ℝ)
    (odds_1_dollar : ℝ) (odds_3_dollar : ℝ) (odds_4_dollar : ℝ)
    (win_rate : ℝ) (grand_prize : ℝ) (average_other_winnings : ℝ) (tax_rate : ℝ),
    num_tickets = 200 →
    percentage_1_dollar = 0.50 →
    percentage_3_dollar = 0.30 →
    percentage_4_dollar = 0.20 →
    cost_1_dollar = 1 →
    cost_3_dollar = 3 →
    cost_4_dollar = 4 →
    odds_1_dollar = 1 / 30 →
    odds_3_dollar = 1 / 20 →
    odds_4_dollar = 1 / 10 →
    win_rate = 0.80 →
    grand_prize = 5000 →
    average_other_winnings = 15 →
    tax_rate = 0.10 →
    (let num_1_dollar_tickets := num_tickets * percentage_1_dollar in
     let num_3_dollar_tickets := num_tickets * percentage_3_dollar in
     let num_4_dollar_tickets := num_tickets * percentage_4_dollar in
     let total_cost := num_1_dollar_tickets * cost_1_dollar + num_3_dollar_tickets * cost_3_dollar + num_4_dollar_tickets * cost_4_dollar in
     let winners_1_dollar := num_1_dollar_tickets / odds_1_dollar in
     let winners_3_dollar := num_3_dollar_tickets / odds_3_dollar in
     let winners_4_dollar := num_4_dollar_tickets / odds_4_dollar in
     let total_winners := winners_1_dollar + winners_3_dollar + winners_4_dollar in
     let win_5_dollars := win_rate * total_winners in
     let win_grand_prize := 1 in
     let win_other := total_winners - win_5_dollars - win_grand_prize in
     let total_winnings_before_tax := win_5_dollars * 5 + win_grand_prize * grand_prize + win_other * average_other_winnings in
     let tax_on_winnings := tax_rate * total_winnings_before_tax in
     let total_winnings_after_tax := total_winnings_before_tax - tax_on_winnings in
     let profit := total_winnings_after_tax - total_cost in
     profit = 4109.5) :=
sorry

end james_profit_calculation_l10_10286


namespace Mary_books_count_l10_10324

theorem Mary_books_count :
  let initial_books := 10
  let after_first_return_checkout := initial_books - 5 + 6
  let after_second_return_checkout := after_first_return_checkout - 3 + 4
  let after_third_return_checkout := after_second_return_checkout - 2 + 9
  let final_books := after_third_return_checkout - 5 + 8
  final_books = 22 :=
by
  let initial_books := 10
  let after_first_return_checkout := initial_books - 5 + 6
  let after_second_return_checkout := after_first_return_checkout - 3 + 4
  let after_third_return_checkout := after_second_return_checkout - 2 + 9
  let final_books := after_third_return_checkout - 5 + 8
  show final_books = 22 from sorry

end Mary_books_count_l10_10324


namespace find_original_price_l10_10798

-- Definitions based on the conditions
def original_price_increased (x : ℝ) : ℝ := 1.25 * x
def loan_payment (total_cost : ℝ) : ℝ := 0.75 * total_cost
def own_funds (total_cost : ℝ) : ℝ := 0.25 * total_cost

-- Condition values
def new_home_cost : ℝ := 500000
def loan_amount := loan_payment new_home_cost
def funds_paid := own_funds new_home_cost

-- Proof statement
theorem find_original_price : 
  ∃ x : ℝ, original_price_increased x = funds_paid ↔ x = 100000 :=
by
  -- Placeholder for actual proof
  sorry

end find_original_price_l10_10798


namespace triangle_angles_divide_circle_l10_10340

theorem triangle_angles_divide_circle (A B C : Point) (circle : Circle) 
  (hABC : divides_circle A B C 3 5 7) : 
  angle A B C = 36 ∧ angle B A C = 60 ∧ angle C A B = 84 :=
sorry

end triangle_angles_divide_circle_l10_10340


namespace collinear_points_in_triangle_l10_10709

noncomputable section

open EuclideanGeometry

variables (A B C D J M : Point)

-- Assume the triangle and the relevant points in the plane
variable (hABC : Triangle A B C)

-- D is the point of tangency of the incircle with side BC
variable (hD : inCircleTangentPoint D (side B C) (inCircle A B C))

-- J is the center of the excircle opposite vertex A
variable (hJ : J = exCircleCenterOpposite A B C)

-- M is the midpoint of the altitude from vertex A
variable (hM : isMidpoint M (altitudeFromVertex A B C))

-- Prove D, M, and J are collinear
theorem collinear_points_in_triangle : collinear {D, M, J} :=
  sorry

end collinear_points_in_triangle_l10_10709


namespace range_of_m_l10_10586

noncomputable def condition_p (x : ℝ) : Prop :=
  |2 * x + 1| ≤ 3

noncomputable def condition_q (x m : ℝ) : Prop :=
  x^2 - 2 * x + 1 - m^2 ≤ 0

theorem range_of_m (h : ∀ x, condition_p x → condition_q x m) (h2 : m > 0) : 
  3 ≤ m :=
begin
  sorry
end

end range_of_m_l10_10586


namespace factorial_arithmetic_l10_10044

theorem factorial_arithmetic :
  (fact 12 / fact 3) * (5^3 - 3 * 7^2) = -14515200 := by
  sorry

end factorial_arithmetic_l10_10044


namespace Tom_age_ratio_l10_10003

theorem Tom_age_ratio (T N : ℕ) (h1 : T = T) 
    (h2 : (∑ i in (finset.range 4), (T/4)) = T) 
    (h3 : T - N = 3 * (T - 4 * N)) : T / N = 11 / 2 :=
by {
  sorry
}

end Tom_age_ratio_l10_10003


namespace sum_q_eq_336_l10_10178

noncomputable
def q : ℝ → ℝ := sorry -- define the cubic polynomial q(x)

variables {x : ℝ}

-- Given conditions on q(x)
axiom hq_3 : q 3 = 2
axiom hq_8 : q 8 = 20
axiom hq_16 : q 16 = 12
axiom hq_21 : q 21 = 30

-- Property to be proven
theorem sum_q_eq_336 : (Finset.sum (Finset.range 21) (λ i, q (i + 2))) = 336 :=
sorry

end sum_q_eq_336_l10_10178


namespace fisherman_total_fish_l10_10466

theorem fisherman_total_fish :
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  sorry

end fisherman_total_fish_l10_10466


namespace lcm_multiplied_by_2_is_72x_l10_10460

-- Define the denominators
def denom1 (x : ℕ) := 4 * x
def denom2 (x : ℕ) := 6 * x
def denom3 (x : ℕ) := 9 * x

-- Define the least common multiple of three natural numbers
def lcm_three (a b c : ℕ) := Nat.lcm a (Nat.lcm b c)

-- Define the multiplication by 2
def multiply_by_2 (n : ℕ) := 2 * n

-- Define the final result
def final_result (x : ℕ) := 72 * x

-- The proof statement
theorem lcm_multiplied_by_2_is_72x (x : ℕ): 
  multiply_by_2 (lcm_three (denom1 x) (denom2 x) (denom3 x)) = final_result x := 
by
  sorry

end lcm_multiplied_by_2_is_72x_l10_10460


namespace largest_a1_l10_10954

theorem largest_a1
  (a : ℕ+ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_eq : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h_initial : a 1 = a 10) :
  ∃ (max_a1 : ℝ), max_a1 = 16 ∧ ∀ x, x = a 1 → x ≤ 16 :=
by
  sorry

end largest_a1_l10_10954


namespace impossible_ticket_payment_change_l10_10790

/-- Prove that it is impossible for 40 passengers with 49 coins of
denominations 10, 15, and 20 euros to each pay a bus ticket of 5 euros
and receive the appropriate change -/
theorem impossible_ticket_payment_change (
  num_passengers : ℕ := 40,
  total_coins : ℕ := 49,
  coin_denominations : set ℕ := {10, 15, 20},
  ticket_price : ℕ := 5
) :
  ∀ (coins : list ℕ), (list.length coins = total_coins ∧ coins.all (λ coin, coin ∈ coin_denominations)) →
  ¬ (∃ (payments : list ℕ), list.length payments = num_passengers ∧ list.sum payments = num_passengers * ticket_price ∧
  ∀ (payment : ℕ), payment ∈ payments → (∃ (change_coins : list ℕ), list.sum change_coins = payment - ticket_price ∧ 
    list.all change_coins (λ coin, coin ∈ coin_denominations))) :=
by sorry

end impossible_ticket_payment_change_l10_10790


namespace total_driving_routes_l10_10099

def num_starting_points : ℕ := 4
def num_destinations : ℕ := 3

theorem total_driving_routes (h1 : ¬(num_starting_points = 0)) (h2 : ¬(num_destinations = 0)) : 
  num_starting_points * num_destinations = 12 :=
by
  sorry

end total_driving_routes_l10_10099


namespace tim_took_rulers_l10_10398

theorem tim_took_rulers (initial_rulers : ℕ) (remaining_rulers : ℕ) (rulers_taken : ℕ) :
  initial_rulers = 46 → remaining_rulers = 21 → rulers_taken = initial_rulers - remaining_rulers → rulers_taken = 25 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end tim_took_rulers_l10_10398


namespace four_digit_palindromic_squares_count_l10_10231

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def ends_with (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d

theorem four_digit_palindromic_squares_count :
  ∃ n1 n2 n3, is_palindrome n1 ∧ is_square n1 ∧ 1000 ≤ n1 ∧ n1 ≤ 9999 ∧ (ends_with n1 0 ∨ ends_with n1 4 ∨ ends_with n1 6) ∧
              is_palindrome n2 ∧ is_square n2 ∧ 1000 ≤ n2 ∧ n2 ≤ 9999 ∧ (ends_with n2 0 ∨ ends_with n2 4 ∨ ends_with n2 6) ∧
              is_palindrome n3 ∧ is_square n3 ∧ 1000 ≤ n3 ∧ n3 ≤ 9999 ∧ (ends_with n3 0 ∨ ends_with n3 4 ∨ ends_with n3 6) ∧
              n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3 ∧
              ∀ m, is_palindrome m ∧ is_square m ∧ 1000 ≤ m ∧ m ≤ 9999 ∧ (ends_with m 0 ∨ ends_with m 4 ∨ ends_with m 6) → m = n1 ∨ m = n2 ∨ m = n3 :=
by sorry

end four_digit_palindromic_squares_count_l10_10231


namespace fisherman_total_fish_l10_10467

theorem fisherman_total_fish :
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  sorry

end fisherman_total_fish_l10_10467


namespace polygon_intersection_area_at_least_one_l10_10753

theorem polygon_intersection_area_at_least_one (n : ℕ) (hn : n ≥ 2) 
(s : ℝ) (hs : s = (n - 1) ^ 2) 
(S : ℝ) (hS : S = n * (n - 1) ^ 2 / 2) 
(polygons : finset (set ℝ)) (hpolygons : polygons.card = n) 
(areas : polygons → ℝ) (hareas : ∀ p ∈ polygons, areas p = s) :
  ∃ (p1 p2 : set ℝ), p1 ∈ polygons ∧ p2 ∈ polygons ∧ p1 ≠ p2 ∧ (set.inter p1 p2).measure ≥ 1 := 
sorry

end polygon_intersection_area_at_least_one_l10_10753


namespace andy_diana_weight_l10_10511

theorem andy_diana_weight :
  ∀ (a b c d : ℝ),
  a + b = 300 →
  b + c = 280 →
  c + d = 310 →
  a + d = 330 := by
  intros a b c d h₁ h₂ h₃
  -- Proof goes here
  sorry

end andy_diana_weight_l10_10511


namespace equal_focal_lengths_l10_10770

theorem equal_focal_lengths (k : ℝ) : 
  let a₁_squared := 25
  let b₁_squared := 9
  let a₂_squared := 25 - k
  let b₂_squared := 9 - k
  sqrt (a₁_squared - b₁_squared) = sqrt (a₂_squared - b₂_squared) := by
  let c₁ := sqrt (25 - 9)
  let c₂ := sqrt ((25 - k) - (9 - k))
  have : c₁ = 4 := by sorry
  have : c₂ = 4 := by sorry
  show 2 * c₁ = 2 * c₂, by sorry

end equal_focal_lengths_l10_10770


namespace correct_number_of_conclusions_l10_10628

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x - x * Real.cos x

-- Problem statement
theorem correct_number_of_conclusions : 
  (∀ x ∈ Icc 0 Real.pi, f x ≥ 0) ∧ 
  (∀ α β, 0 < α → α < β → β < Real.pi → α * Real.sin β > β * Real.sin α) ∧
  (∀ x ∈ Ioo 0 (Real.pi / 2), ∃ (n m : ℝ), n < Real.sin x / x ∧ Real.sin x / x < m → m - n = 1 - (2 / Real.pi)) ∧ 
  (∀ k ∈ Icc 0 1, ∀ x_i ∈ Ioo 0 (2 * Real.pi), ∃ n ∈ ({0, 1, 2, 3} : Set ℕ), Real.abs (Real.sin x_i) / x_i = k) → 
  (∃ correct_conclusions : ℕ, correct_conclusions = 3) :=
by
  sorry

end correct_number_of_conclusions_l10_10628


namespace second_part_amount_l10_10246

theorem second_part_amount (total_amount : ℝ) 
  (ratio1 ratio2 ratio3 ratio4 : ℝ) 
  (h0 : ratio1 = 3 / 4) 
  (h1 : ratio2 = 1 / 3) 
  (h2 : ratio3 = 5 / 6) 
  (h3 : ratio4 = 4 / 5) 
  (total : total_amount = 3249) 
  : let common_denominator := (lcm 4 3 6 5 : ℝ),
    total_ratio := (3 * (common_denominator / 4) + 1 * (common_denominator / 3) + 5 * (common_denominator / 6) + 4 * (common_denominator / 5)),
    one_part_value := total_amount / total_ratio,
    second_part_value := one_part_value * (1 * (common_denominator / 3))
  in second_part_value ≈ 399.13 := 
by 
  sorry

end second_part_amount_l10_10246


namespace largest_multiple_of_15_less_than_neg_150_l10_10018

theorem largest_multiple_of_15_less_than_neg_150 : ∃ m : ℤ, m % 15 = 0 ∧ m < -150 ∧ (∀ n : ℤ, n % 15 = 0 ∧ n < -150 → n ≤ m) ∧ m = -165 := sorry

end largest_multiple_of_15_less_than_neg_150_l10_10018


namespace correct_relations_count_l10_10960

variable (R : Set ℝ) (Q : Set ℝ) (N : Set ℕ) (e : ℝ) (n : ℕ)

def sqrt_3_in_R : Prop := (sqrt 3) ∈ R
def point_2_not_in_Q : Prop := 0.2 ∉ Q
def abs_neg3_in_N : Prop := abs (-3) ∈ N
def zero_not_in_empty : Prop := 0 ∉ ∅

theorem correct_relations_count : sqrt_3_in_R R ∧ point_2_not_in_Q Q ∧ abs_neg3_in_N N ∧ zero_not_in_empty e → 3 = 3 :=
by
  intros h
  sorry

end correct_relations_count_l10_10960


namespace question1_question2_l10_10175

noncomputable def minimum_value (x y : ℝ) : ℝ := (1 / x) + (1 / y)

theorem question1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) : 
  minimum_value x y = 2 :=
sorry

theorem question2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) :
  (x + 1) * (y + 1) ≠ 5 :=
sorry

end question1_question2_l10_10175


namespace nadine_hosing_time_l10_10328

theorem nadine_hosing_time (shampoos : ℕ) (time_per_shampoo : ℕ) (total_cleaning_time : ℕ) 
  (h1 : shampoos = 3) (h2 : time_per_shampoo = 15) (h3 : total_cleaning_time = 55) : 
  ∃ t : ℕ, t = total_cleaning_time - shampoos * time_per_shampoo ∧ t = 10 := 
by
  sorry

end nadine_hosing_time_l10_10328


namespace value_of_a_l10_10316

def A (a : ℝ) : Set ℝ := {4, 2, a^2}
def B (a : ℝ) : Set ℝ := {1, a}
def P (a : ℝ) : Prop := ({4, 2, a^2} ∩ {1, a}) = {1}

theorem value_of_a (a : ℝ) (h : P a) : a = -1 :=
sorry

end value_of_a_l10_10316


namespace rectangle_enclosed_by_four_lines_l10_10553

theorem rectangle_enclosed_by_four_lines : 
  let h_lines := 5
  let v_lines := 5
  (choose h_lines 2) * (choose v_lines 2) = 100 :=
by {
  sorry
}

end rectangle_enclosed_by_four_lines_l10_10553


namespace florist_total_roses_l10_10475

-- Define the known quantities
def originalRoses : ℝ := 37.0
def firstPick : ℝ := 16.0
def secondPick : ℝ := 19.0

-- The theorem stating the total number of roses
theorem florist_total_roses : originalRoses + firstPick + secondPick = 72.0 :=
  sorry

end florist_total_roses_l10_10475


namespace smallest_self_sum_number_222_l10_10125

def letter_value (ch : Char) : Nat :=
  match ch with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5
  | 'F' => 6 | 'G' => 7 | 'H' => 8 | 'I' => 9 | 'J' => 10
  | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14 | 'O' => 15
  | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25
  | 'Z' => 26 | _ => 0

def word_value (s : String) : Nat :=
  s.fold 0 (fun acc ch => acc + letter_value ch)

-- Definitions for the written-out name of 222 in French:
def deux := "DEUX"
def cents := "CENTS"
def vingt := "VINGT"

def written_out_222 := deux ++ cents ++ vingt ++ deux

theorem smallest_self_sum_number_222 : word_value written_out_222 = 222 :=
by 
  -- The proof steps are omitted as they are not required in the task.
  sorry

end smallest_self_sum_number_222_l10_10125


namespace rotated_vector_is_correct_l10_10789

open Real

/-- The vector rotated 90 degrees about the origin, passing through the y-axis -/
theorem rotated_vector_is_correct :
  let v := (⟨2, 1, 1⟩ : ℝ × ℝ × ℝ)
  let resulting_vector := (⟨sqrt (6 / 11), -3 * sqrt (6 / 11), sqrt (6 / 11)⟩ : ℝ × ℝ × ℝ)
  -- Condition: v rotated 90 degrees
  -- Condition: resulting vector passes through y-axis
  (v.1 * resulting_vector.1 + v.2 * resulting_vector.2 + v.3 * resulting_vector.3 = 0) ∧ -- orthogonality
  (resulting_vector.1 ^ 2 + resulting_vector.2 ^ 2 + resulting_vector.3 ^ 2 = 6) -- magnitude preservation
  := 
by sorry

end rotated_vector_is_correct_l10_10789


namespace A_oplus_B_l10_10576

def set_diff (M N : Set ℝ) : Set ℝ :=
  {x | x ∈ M ∧ x ∉ N}

def set_symm_diff (M N : Set ℝ) : Set ℝ :=
  set_diff M N ∪ set_diff N M

def set_A : Set ℝ :=
  {x | x ≥ -9/4}

def set_B : Set ℝ :=
  {x | x < 0}

theorem A_oplus_B : set_symm_diff set_A set_B = (set_Iio (-9/4)) ∪ (set_Ici 0) :=
  sorry

end A_oplus_B_l10_10576


namespace non_dth_power_exists_l10_10301

theorem non_dth_power_exists 
  (d : ℕ) (ε : ℝ) (hε : ε > 0) :
  ∀ p : ℕ, p.prime → (∃ k : ℕ, k < p ∧ ∃ q : ℕ, p^q > p ∧ ¬ is_dth_power_mod (q : ℤ)) :=
begin
  sorry
end

end non_dth_power_exists_l10_10301


namespace equivalence_of_functions_l10_10092

theorem equivalence_of_functions :
  (∀ x : ℝ, |x| = real.sqrt (x^2)) :=
begin
  sorry
end

end equivalence_of_functions_l10_10092


namespace tangent_line_equation_l10_10375

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x^3

theorem tangent_line_equation : 
  let f' (x : ℝ) : ℝ := (x * Real.log x - x^3)' in
  let slope := f' 1 in
  let point := (1, f 1) in
  slope = -2 ∧ point = (1, -1) → ∀ (x y : ℝ), 
  y = slope * (x - 1) + (f 1) ↔ 2 * x + y - 1 = 0 :=
by
  sorry

end tangent_line_equation_l10_10375


namespace log_sum_evaluation_l10_10128

theorem log_sum_evaluation :
  log 10 50 + (log 10 45 / log 10 5) = 4 + log 10 (9 / 5) :=
by
  sorry

end log_sum_evaluation_l10_10128


namespace find_positive_real_numbers_l10_10913

open Real

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16

theorem find_positive_real_numbers (x : ℝ) (hx : x > 0) :
  satisfies_inequality x ↔ 15 * x^2 + 32 * x - 256 = 0 :=
sorry

end find_positive_real_numbers_l10_10913


namespace tan_arccot_l10_10109

theorem tan_arccot (adj opp : ℝ) (h_adj : adj = 3) (h_opp : opp = 5) :
  Real.tan (Real.arccot (adj / opp)) = 5 / 3 :=
by
  -- Conditions from the problem:
  have h_adj_cot : Real.cot (Real.arccot (adj / opp)) = adj / opp := Real.cot_arccot _
  
  -- Substitute the given values:
  rw [h_adj, h_opp] at *
  
  -- This needs to be proven explicitly following the steps in the solution:
  -- Placeholder for the actual proof steps
  sorry

end tan_arccot_l10_10109


namespace perimeter_of_new_rectangle_l10_10494

def square_side : ℝ := 8
def rectangle_length : ℝ := 16
def rectangle_breadth : ℝ := 6
def new_rectangle_perimeter : ℝ := 65.34

theorem perimeter_of_new_rectangle :
  let square_area := square_side * square_side
  let rect_area := rectangle_length * rectangle_breadth
  let total_area := square_area + rect_area
  let new_rectangle_breadth := rectangle_breadth
  let new_rectangle_length := total_area / new_rectangle_breadth
  let perimeter := 2 * (new_rectangle_length + new_rectangle_breadth)
in
  perimeter = new_rectangle_perimeter := by
  sorry

end perimeter_of_new_rectangle_l10_10494


namespace cards_three_digit_combinations_l10_10014

theorem cards_three_digit_combinations : 
  ∃ (numbers : Finset ℕ), 
    (∀ x ∈ numbers, ∃ a b c, x = 100 * a + 10 * b + c ∧ 
                              Finset.card numbers = 6 ∧ 
                              Finset.card (Finset.mk ([a, b, c].toFinset.toList) sorry) = 3 ∧ 
                              a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
                              a ∈ ({2, 5, 7} : Finset ℕ) ∧ 
                              b ∈ ({2, 5, 7} : Finset ℕ) ∧ 
                              c ∈ ({2, 5, 7} : Finset ℕ)) := 
sorry

end cards_three_digit_combinations_l10_10014


namespace raisin_cost_fraction_l10_10105

theorem raisin_cost_fraction
  (R : ℚ) -- cost of a pound of raisins in dollars
  (cost_of_nuts : ℚ)
  (total_cost_raisins : ℚ)
  (total_cost_nuts : ℚ) :
  cost_of_nuts = 3 * R →
  total_cost_raisins = 5 * R →
  total_cost_nuts = 4 * cost_of_nuts →
  (total_cost_raisins / (total_cost_raisins + total_cost_nuts)) = 5 / 17 :=
by
  sorry

end raisin_cost_fraction_l10_10105


namespace distinct_angles_in_12_sided_polygon_l10_10341

theorem distinct_angles_in_12_sided_polygon :
  ∃ (angles : Finset ℝ), angles.card = 6 ∧
  (∀ a ∈ angles, ∃ (α β γ : ℝ), is_regular_polygon α ∧ is_regular_polygon β ∧
    vertices_do_not_coincide α β ∧ no_shared_symmetry_axes α β ∧ is_formed_by_consecutive_marked_points α β γ ∧
    exhibit_angle_values γ a) := 
  sorry

def is_regular_polygon (α : ℝ) : Prop :=
  ∀ (n : ℕ) (h : n > 2), ∃ (vertices : List ℝ), (vertices.length = n ∧
  (∀ i, (vertices.nth i = some (α + (i : ℝ)*(2*π / n)))))

def vertices_do_not_coincide (α β : ℝ) : Prop :=
  ∀ (i j : ℕ), (α + i*(2*π/5) % (2*π) ≠ β + j*(2*π/7) % (2*π))

def no_shared_symmetry_axes (α β : ℝ) : Prop :=
  ∀ (i j : ℕ), ¬is_symmetry_axis α β i j

def is_formed_by_consecutive_marked_points (α β γ : ℝ) : Prop :=
  ∃ (vertices : List ℝ), (vertices.length = 12 ∧
  (∀ i, (vertices.nth i = if i%5 < 5 then some (α + (i : ℝ)*(2*π / 5)) else some (β + ((i-5) : ℝ)*(2*π / 7)))))

def exhibit_angle_values (γ a : ℝ) : Prop :=
  ∃ (angles : Finset ℝ), angles.card = 12 ∧ a ∈ angles

end distinct_angles_in_12_sided_polygon_l10_10341


namespace find_x_satisfying_inequality_l10_10920

open Real

theorem find_x_satisfying_inequality :
  ∀ x : ℝ, 0 < x → (x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16 ↔ x = 4) :=
by
  sorry

end find_x_satisfying_inequality_l10_10920


namespace product_of_square_roots_l10_10102
-- Importing the necessary Lean library

-- Declare the mathematical problem in Lean 4
theorem product_of_square_roots (x : ℝ) (hx : 0 ≤ x) :
  Real.sqrt (40 * x) * Real.sqrt (5 * x) * Real.sqrt (18 * x) = 60 * x * Real.sqrt (3 * x) :=
by
  sorry

end product_of_square_roots_l10_10102


namespace Martin_correct_answers_l10_10338

theorem Martin_correct_answers (C K M : ℕ) 
  (h1 : C = 35)
  (h2 : K = C + 8)
  (h3 : M = K - 3) : 
  M = 40 :=
by
  sorry

end Martin_correct_answers_l10_10338


namespace SK_parallel_BC_l10_10787

open EuclideanGeometry

-- Definitions of the points and conditions
variables {A B C K L M P S : Point}
variables (ω : Circle)
variables (HABC : inscribed_triangle ω A B C)
variables (HK_on_AB : on_line_segment K A B)
variables (HL_on_BC : on_line_segment L B C)
variables (HM_on_CA : on_line_segment M C A)
variables (HCondition : collinear_points C M L ∧ collinear_points A M B ∧ CM * CL = AM * BL)
variables (HLK_intersects_AC_at_P : intersection_of_rays LK AC P)
variables (HCommon_Chord : common_chord_circle ω (circumcircle K M P) S)
variables (HS_meets_AM : on_line_segment S A M)

-- Statement to be proven: SK ∥ BC
theorem SK_parallel_BC 
  (HABC : inscribed_triangle ω A B C) 
  (HK_on_AB : on_line_segment K A B) 
  (HL_on_BC : on_line_segment L B C) 
  (HM_on_CA : on_line_segment M C A) 
  (HCondition : collinear_points C M L ∧ collinear_points A M B ∧ CM * CL = AM * BL) 
  (HLK_intersects_AC_at_P : intersection_of_rays LK AC P) 
  (HCommon_Chord : common_chord_circle ω (circumcircle K M P) S) 
  (HS_meets_AM : on_line_segment S A M) : 
  Parallel SK BC :=
sorry

end SK_parallel_BC_l10_10787


namespace max_hands_involved_in_dance_l10_10323

/-- A dance "Pyramid" involves Martians. Each Martian has at most 3 hands, and no more than 7 Martians can participate. Each hand of a Martian holds exactly one hand of another Martian. Prove that the maximum number of hands that can be involved in the dance is 20. --/
theorem max_hands_involved_in_dance : 
  ∀ (martians : ℕ) (hands_per_martian : ℕ), 
  martians ≤ 7 → 
  (∀ m, m < martians → hands_per_martian m ≤ 3) → 
  (∀ m, m < martians → hands_per_martian m % 2 = 0) → 
  (∀ m n, m < martians → n < martians → m ≠ n → equal_num_of_hands (hands_per_martian m) (hands_per_martian n)) → 
  ∑ m in finset.range martians, hands_per_martian m = 20 :=
by
  sorry

end max_hands_involved_in_dance_l10_10323


namespace k_satisfies_triangle_condition_l10_10541

theorem k_satisfies_triangle_condition (k : ℤ) 
  (hk_pos : 0 < k) (a b c : ℝ) (ha_pos : 0 < a) 
  (hb_pos : 0 < b) (hc_pos : 0 < c) 
  (h_ineq : (k : ℝ) * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : k = 6 → 
  (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end k_satisfies_triangle_condition_l10_10541


namespace find_k_percent_l10_10167

theorem find_k_percent (k : ℝ) : 0.2 * 30 = 6 → (k / 100) * 25 = 6 → k = 24 := by
  intros h1 h2
  sorry

end find_k_percent_l10_10167


namespace largest_multiple_of_7_smaller_than_neg_85_l10_10426

theorem largest_multiple_of_7_smaller_than_neg_85 :
  ∃ k : ℤ, 7 * k < -85 ∧ (∀ m : ℤ, 7 * m < -85 → 7 * m ≤ 7 * k) ∧ 7 * k = -91 :=
by
  simp only [exists_prop, and.assoc],
  sorry

end largest_multiple_of_7_smaller_than_neg_85_l10_10426


namespace fisherman_total_fish_l10_10474

theorem fisherman_total_fish :
  let bass : Nat := 32
  let trout : Nat := bass / 4
  let blue_gill : Nat := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  show bass + trout + blue_gill = 104
  sorry

end fisherman_total_fish_l10_10474


namespace number_of_valid_three_digit_even_numbers_l10_10999

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l10_10999


namespace find_sum_of_angles_l10_10736

-- Define the problem's conditions
variables (A B Q D C : Type) -- Points lie on the circle.
variables (arc_BQ arc_QD : ℝ) -- measures of arcs BQ and QD

-- Assume measures of arcs are given as follows
def arc_measures := (arc_BQ = 60) ∧ (arc_QD = 24)

-- Define the statement to prove
theorem find_sum_of_angles (arc_measures : arc_measures) :
  (∃ P Q : ℝ, P + Q = 42) :=
by 
  sorry

end find_sum_of_angles_l10_10736


namespace circumcircle_eq_of_triangle_ABC_l10_10222

noncomputable def circumcircle_equation (A B C : ℝ × ℝ) : String := sorry

theorem circumcircle_eq_of_triangle_ABC :
  circumcircle_equation (4, 1) (-6, 3) (3, 0) = "x^2 + y^2 + x - 9y - 12 = 0" :=
sorry

end circumcircle_eq_of_triangle_ABC_l10_10222


namespace find_a_l10_10609

theorem find_a (a b c : ℕ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) (h_positive_c : 0 < c) (h_eq : (18 ^ a) * (9 ^ (3 * a - 1)) * (c ^ a) = (2 ^ 7) * (3 ^ b)) : a = 7 := by
  sorry

end find_a_l10_10609


namespace max_value_l10_10226

open Real

/-- Given vectors a, b, and c, and real numbers m and n such that m * a + n * b = c,
prove that the maximum value for (m - 3)^2 + n^2 is 16. --/
theorem max_value
  (α : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ)
  (m n : ℝ)
  (ha : a = (1, 1))
  (hb : b = (1, -1))
  (hc : c = (sqrt 2 * cos α, sqrt 2 * sin α))
  (h : m * a.1 + n * b.1 = c.1 ∧ m * a.2 + n * b.2 = c.2) :
  (m - 3)^2 + n^2 ≤ 16 :=
by
  sorry

end max_value_l10_10226


namespace minimum_distance_l10_10530

-- Define the problem conditions
def first_quadrant_condition (x y : ℝ) : Prop := (8 * x + 15 * y = 120) ∧ (x > 0) ∧ (y > 0)

-- Define the minimum value expression
def minimum_value (x y : ℝ) : ℝ := sqrt (x ^ 2 + y ^ 2)

-- Main theorem statement 
theorem minimum_distance (x y : ℝ) (h : first_quadrant_condition x y) :
  minimum_value x y = 120 / 17 := 
sorry

end minimum_distance_l10_10530


namespace initial_percentage_decrease_l10_10779

theorem initial_percentage_decrease (P : ℝ) (x : ℝ) :
  let after_decrease := P * (1 - x / 100),
      after_increase := after_decrease * 1.10 in
  after_increase = P * 1.12 → x = 20 / 11 :=
by
  sorry

end initial_percentage_decrease_l10_10779


namespace remainder_sum_first_150_div_11250_l10_10436

theorem remainder_sum_first_150_div_11250 : 
  let S := 150 * 151 / 2 
  in S % 11250 = 75 := 
by
  let S := 11325
  have hSum : S = 11325 := by rfl
  show S % 11250 = 75
  sorry

end remainder_sum_first_150_div_11250_l10_10436


namespace largest_multiple_of_7_smaller_than_negative_85_l10_10419

theorem largest_multiple_of_7_smaller_than_negative_85 :
  ∃ (n : ℤ), (∃ (k : ℤ), n = 7 * k) ∧ n < -85 ∧ ∀ (m : ℤ), (∃ (k : ℤ), m = 7 * k) ∧ m < -85 → m ≤ n := 
by
  use -91
  split
  { use -13
    norm_num }
  split
  { exact dec_trivial }
  { intros m hm
    cases hm with k hk
    cases hk with hk1 hk2
    have hk3 : k < -12 := by linarith
    have hk4 : k ≤ -13 := int.floor_le $ hk3
    linarith }


end largest_multiple_of_7_smaller_than_negative_85_l10_10419


namespace triangle_vector_sum_l10_10687

/-- In triangle ABC, D is the midpoint of AC. 
    BE = 2 * ED. 
    AE = x * AB + y * AC. 
    Prove that x + y = 2 / 3. -/
theorem triangle_vector_sum 
  (A B C D E : Type)
  [AddCommGroup A] [Module ℝ A]
  (AC AB : A)
  (D_midpoint : 2 • D = AC)
  (BE_eq_2ED : BE = 2 • ED)
  (AE_eq : AE = x • AB + y • AC) :
  x + y = 2 / 3 :=
sorry

end triangle_vector_sum_l10_10687


namespace expected_attempts_for_10_l10_10822

noncomputable def expected_attempts (n : Nat) : ℕ :=
  (n * (n + 3)) / 4 - Nat.harmonic n

theorem expected_attempts_for_10 : expected_attempts 10 ≈ 29.62 :=
by
  sorry

end expected_attempts_for_10_l10_10822


namespace maximum_z_at_sqrt2_0_l10_10752

def satisfies_inequalities (x y : ℝ) : Prop :=
  x + y - real.sqrt 2 ≤ 0 ∧ x - y + real.sqrt 2 ≥ 0 ∧ y ≥ 0

def z (x y : ℝ) := 2 * x - y

theorem maximum_z_at_sqrt2_0 :
  ∃ (x y : ℝ), satisfies_inequalities x y ∧ z x y = 2 * real.sqrt 2 :=
by
  existsi real.sqrt 2
  existsi 0
  apply and.intro
  { exact and.intro
    { linarith [real.sqrt_nonneg 2] }
    { linarith [real.sqrt_nonneg 2] } }
  { exact le_refl 0 }
  linarith

end maximum_z_at_sqrt2_0_l10_10752


namespace sum_A_B_l10_10115

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

def A : ℝ := ∑ n in Finset.range 2015, f (n + 1)

def B : ℝ := ∑ n in Finset.range 2015, f (1 / (n + 1))

theorem sum_A_B : A + B = 2015 := 
by
  sorry

end sum_A_B_l10_10115


namespace number_of_valid_three_digit_even_numbers_l10_10996

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l10_10996


namespace prove_rectangular_selection_l10_10562

def number_of_ways_to_choose_rectangular_region (horizontals verticals : ℕ) : ℕ :=
  (Finset.choose horizontals 2) * (Finset.choose verticals 2)

theorem prove_rectangular_selection :
  number_of_ways_to_choose_rectangular_region 5 5 = 100 :=
by
  sorry

end prove_rectangular_selection_l10_10562


namespace min_elements_in_set_l10_10937

theorem min_elements_in_set 
  (A : Type*) [Fintype A]
  (f : ℕ → A)
  (h : ∀ (i j : ℕ), Prime (|i - j|) → f i ≠ f j) :
  4 ≤ Fintype.card A :=
sorry

end min_elements_in_set_l10_10937


namespace infinite_losing_configurations_l10_10525

-- Define game conditions
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def valid_move (x : ℕ) : Prop :=
  is_square x ∧ x > 0

-- Define the sets L (first player loses) and W (first player wins)
inductive losing (L : set ℕ) : ℕ → Prop
| base : ∀ {n}, (∀ k, valid_move k → n < k) → losing n
| step : ∀ {n m}, ¬ (n < m ∧ losing m ∧ valid_move (n - m)) → losing n

def winning (L : set ℕ) (n : ℕ) : Prop := ¬ losing L n

-- Statement of the problem
theorem infinite_losing_configurations (L : set ℕ) :
  (∀ n ∈ L, ∃ k ∈ L, n - k ∈ L ∧ valid_move (n - k)) →
  infinite L :=
sorry

end infinite_losing_configurations_l10_10525


namespace correct_statements_count_l10_10386

theorem correct_statements_count :
  (∀ x : ℝ, x^2 ≠ -8) ∧
  (∀ x : ℝ, (x^2 = 25 → x = 5 ∨ x = -5)) ∧
  (∀ x : ℝ, (x^3 = x ↔ x = 0 ∨ x = 1)) ∧
  (∃ x : ℝ, x^2 = 16 ∧ x = 4) →
  3 = (
    (∀ x : ℝ, x^2 ≠ -8) +
    (∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5)) +
    (¬(∀ x : ℝ, (x^3 = x ↔ x = 0 ∨ x = 1))) +
    (∃ x : ℝ, x^2 = 16 ∧ x = 4)
  ) :=
by {
  sorry -- Proof will be done here
}

end correct_statements_count_l10_10386


namespace num_divisors_g_2010_l10_10947

noncomputable def g (n : ℕ) : ℕ :=
  2^n * 3^n

theorem num_divisors_g_2010 : ∀ n, (g n) = 2^n * 3^n → 
  number_of_divisors (g 2010) = 4044121 :=
by
  sorry

end num_divisors_g_2010_l10_10947


namespace min_stamps_l10_10868

theorem min_stamps (x y : ℕ) (h : 5 * x + 7 * y = 47) : x + y ≥ 7 :=
by 
  have h₀ : ∃ x y : ℕ, 5 * x + 7 * y = 47 := sorry,
  have min_value := minstamps h₀,
  exact min_value

end min_stamps_l10_10868


namespace probability_of_special_integer_l10_10096

theorem probability_of_special_integer :
  let total_integers : ℕ := 9000 in
  let even_units := {0, 2, 4, 6, 8} in
  let valid_integer (n : ℕ) : Prop := 
    n >= 1000 ∧ n <= 9999 ∧ 
    (n % 2 = 0) ∧ 
    let digits := List.ofFn (λ i => (n / (10 ^ i)) % 10) (range 4) in
    digits.nodup ∧ 
    (digits.nthLe 3 (by simp)) > (digits.nthLe 2 (by simp)) in

  finite (λ n, valid_integer n) →
  ∃ (p : ℚ),
    p = 1/40 ∨ p = 3/40 ∨ p = 7/50 ∨ p = 11/75 ∨ p = 9/50 ∧
    let favorable_count := { n | valid_integer n }.card in
    p = favorable_count / total_integers := 
by 
  sorry

end probability_of_special_integer_l10_10096


namespace f_odd_function_f_max_min_l10_10179

variable {f : ℝ → ℝ}

-- Conditions of the problem
axiom additivity : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom negative_on_positives : ∀ x : ℝ, x > 0 → f(x) < 0
axiom f_one : f(1) = -2

-- Proof that f is odd
theorem f_odd_function : ∀ x : ℝ, f(-x) = -f(x) := sorry

-- Find the maximum and minimum values of f on the interval [-3, 3]
theorem f_max_min : ∃ xmax xmin : ℝ, (xmax = f(-3) ∧ xmax = 6) ∧ (xmin = f(3) ∧ xmin = -6) := sorry

end f_odd_function_f_max_min_l10_10179


namespace max_value_PQ_MN_theorem_l10_10524

-- Given conditions
variable {p : ℝ} (hp : p > 0)

-- Definition of the maximum value problem
def max_value_PQ_MN (PQ MN : ℝ) : ℝ :=
  PQ / MN

-- The proof statement
theorem max_value_PQ_MN_theorem (PQ MN : ℝ) (h : MN > 0) :
  max_value_PQ_MN PQ MN = sqrt 2 / 2 :=
sorry

end max_value_PQ_MN_theorem_l10_10524


namespace determine_phi_for_even_function_l10_10379

theorem determine_phi_for_even_function (φ : ℝ) (k : ℤ)
  (h1 : ∃ f : ℝ → ℝ, f = λ x, Real.sin (2 * x + φ))
  (h2 : ∃ g : ℝ → ℝ, g = λ x, Real.sin (2 * (x + π / 6) + φ))
  (h3 : ∀ x, Real.sin (2 * (x + π / 6) + φ) = Real.sin (2 * (-x + π / 6) + φ)) :
  φ = π / 6 :=
by
  sorry

end determine_phi_for_even_function_l10_10379


namespace proportion_of_r_after_3_minutes_l10_10819

variable (Tank : Type)
variable [Inhabited Tank] -- Assume Tank is non-empty

def rate (time: ℕ) : ℚ := 1 / time

def amount_filled (time: ℚ) (rate: ℚ) : ℚ := time * rate

theorem proportion_of_r_after_3_minutes :
  let rate_a := rate 20
  let rate_b := rate 20
  let rate_c := rate 30
  let amount_a := amount_filled 3 rate_a
  let amount_b := amount_filled 3 rate_b
  let amount_c := amount_filled 3 rate_c
  let total_filled := amount_a + amount_b + amount_c
  (1 / 10) / total_filled = 1 / 4 :=
by
  simp [rate, amount_filled, rat_divide, mul_inv_eq]
  sorry

end proportion_of_r_after_3_minutes_l10_10819


namespace general_term_Tn_inequality_l10_10617

-- Conditions
variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)
variable (S : ℕ → α) -- Sum of first n terms of sequence a_n
variable (b : ℕ → α) -- b_n = a_n / a_{n+1}
variable (T : ℕ → α) -- Sum of first n terms of sequence b_n

-- The sum of the first n terms of the sequence {a_n} is S_n and it satisfies S_n = 2a_n - n
axiom sum_condition (n : ℕ) : S n = 2 * a n - n

-- b_n = a_n / a_{n+1}
axiom b_definition (n : ℕ) : b n = a n / a (n + 1)

-- T_n is the sum of the first n terms of the sequence {b_n}
axiom T_definition (n : ℕ) : T n = ∑ i in range n, b i

-- Prove that a_n = 2^n - 1
theorem general_term (n : ℕ) : a n = 2^n - 1 :=
  sorry

-- Prove that -1/3 < T_n - n/2 < 0
theorem Tn_inequality (n : ℕ) : -1/3 < T n - n/2 ∧ T n - n/2 < 0 :=
  sorry

end general_term_Tn_inequality_l10_10617


namespace ahmed_orange_trees_l10_10090

variables (A O X : ℕ)

theorem ahmed_orange_trees :
  (A = 1) →
  (O = 2) →
  (X + 4 = O + A + 9) →
  X = 8 :=
by
  intros ha ho h
  have ha' := ha
  have ho' := ho
  rw ha' at h
  rw ho' at h
  linarith

end

end ahmed_orange_trees_l10_10090


namespace find_a_l10_10979

theorem find_a (k a : ℚ) (hk : 4 * k = 60) (ha : 15 * a - 5 = 60) : a = 13 / 3 :=
by
  sorry

end find_a_l10_10979


namespace lattice_triangle_area_l10_10854

/-- 
Given a triangle with vertices at lattice points and containing no other lattice points, 
the area of the triangle is 1/2.
-/
theorem lattice_triangle_area (p1 p2 p3 : ℤ × ℤ) 
  (h1 : lattice_triangle p1 p2 p3) 
  (h2 : no_interior_lattice_points p1 p2 p3) 
  (h3 : boundary_lattice_points_count p1 p2 p3 = 3) :
  triangle_area p1 p2 p3 = 1 / 2 := 
sorry

/-- 
A triangle is defined by lattice points. 
-/
def lattice_triangle (p1 p2 p3 : ℤ × ℤ) : Prop := 
(p1.1 % 1 = 0) ∧ (p1.2 % 1 = 0) ∧ 
(p2.1 % 1 = 0) ∧ (p2.2 % 1 = 0) ∧ 
(p3.1 % 1 = 0) ∧ (p3.2 % 1 = 0)

/-- 
A triangle contains no interior lattice points.
-/
def no_interior_lattice_points (p1 p2 p3 : ℤ × ℤ) : Prop := 
num_interior_lattice_points p1 p2 p3 = 0

/-- 
Given a triangle, returns the number of lattice points on the boundary.
-/
def boundary_lattice_points_count (p1 p2 p3 : ℤ × ℤ) : ℕ := 
(num_lattice_points_boundary p1 p2 + 
num_lattice_points_boundary p2 p3 + 
num_lattice_points_boundary p3 p1 - 3)

/-- 
Given the vertices of the triangle, calculates its area.
-/
def triangle_area (p1 p2 p3 : ℤ × ℤ) : ℚ := 
1 / 2 * abs (p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p1.2 - p1.2 * p2.1 - p2.2 * p3.1 - p3.2 * p1.1)


end lattice_triangle_area_l10_10854


namespace valid_outfits_count_l10_10237

noncomputable def number_of_valid_outfits (shirt_count: ℕ) (pant_colors: List String) (hat_count: ℕ) : ℕ :=
  let total_combinations := shirt_count * (pant_colors.length) * hat_count
  let matching_outfits := List.length (List.filter (λ c => c ∈ pant_colors) ["tan", "black", "blue", "gray"])
  total_combinations - matching_outfits

theorem valid_outfits_count :
    number_of_valid_outfits 8 ["tan", "black", "blue", "gray"] 8 = 252 := by
  sorry

end valid_outfits_count_l10_10237


namespace find_m_n_sum_l10_10404

-- Define the conditions
noncomputable def total_ways : ℕ := 12.factorial / (4.factorial * 4.factorial * 4.factorial)
noncomputable def undesirable_ways_one_country : ℕ := 3 * 12 * (8.factorial / (4.factorial * 4.factorial))
noncomputable def undesirable_ways_two_countries : ℕ := 3 * 12
noncomputable def undesirable_ways_all_three_countries : ℕ := 2 * 12

-- Application of Inclusion-Exclusion Principle
noncomputable def undesirable_ways : ℕ := undesirable_ways_one_country - undesirable_ways_two_countries + undesirable_ways_all_three_countries

-- Calculation of desirable arrangements
noncomputable def desirable_ways := total_ways - undesirable_ways

-- Probability simplification and correctness check
noncomputable def m : ℕ := 36
noncomputable def n : ℕ := 385

lemma probability_correct : (desirable_ways : ℚ) / (total_ways : ℚ) = (m : ℚ) / (n : ℚ) := sorry

-- Final goal
theorem find_m_n_sum : m + n = 421 :=
by
    exact rfl

end find_m_n_sum_l10_10404


namespace angle_between_PN_and_OM_l10_10270

noncomputable def calculate_angle_between_lines (O P N M : ℝ × ℝ × ℝ) : ℝ :=
  let v₁ := (N.1 - P.1, N.2 - P.2, N.3 - P.3)
  let v₂ := (M.1 - O.1, M.2 - O.2, M.3 - O.3)
  let dot_product := v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3
  let magnitude_v₁ := Math.sqrt (v₁.1 ^ 2 + v₁.2 ^ 2 + v₁.3 ^ 2)
  let magnitude_v₂ := Math.sqrt (v₂.1 ^ 2 + v₂.2 ^ 2 + v₂.3 ^ 2)
  let cos_theta := dot_product / (magnitude_v₁ * magnitude_v₂)
  Real.arccos cos_theta

theorem angle_between_PN_and_OM : 
  let P := (0, 0, 1 : ℝ)
  let N := (1, 1, 0 : ℝ)
  let M := (1, 2, 1 : ℝ)
  let O := (1, 1, 2 : ℝ)
  calculate_angle_between_lines O P N M = Real.arccos (Real.sqrt 6 / 3) :=
by
  sorry

end angle_between_PN_and_OM_l10_10270


namespace problem_equivalent_l10_10710

theorem problem_equivalent {
  (g : ℝ → ℝ) (T : set ℝ) :
  (∀ x : ℝ, 0 ≤ x → g x = (3 * x + 4) / (x + 3)) →
  (T = {y | ∃ x : ℝ, 0 ≤ x ∧ y = g x}) →
  (3 = Sup T) ∧ (3 ∉ T) ∧ (4 / 3 = Inf T) ∧ (4 / 3 ∈ T) :=
by
  intro h_g h_T
  sorry

end problem_equivalent_l10_10710


namespace city_population_ratio_l10_10523

variable (Z : ℕ) (s : ℕ) (r : ℕ)
variable (h1 : s * 6 = 12)

theorem city_population_ratio (H : s * 6 = 12) : s = 2 :=
by
  rw [Nat.mul_eq_mul_left_iff] at H
  cases H
  . contradiction
  . exact H


end city_population_ratio_l10_10523


namespace final_position_total_fuel_consumed_l10_10055

-- Defining a list of distances traveled.
def distances : List Int := [+22, -3, +4, -2, -8, +17, -2, -3, +12, +7, -5]

-- The fuel consumption rate per kilometer.
def fuelConsumptionRate : Float := 0.2

-- The first part of the proof: Proving the final position.
theorem final_position : List.sum distances = 39 :=
by
  sorry

-- The second part of the proof: Proving the total fuel consumed.
theorem total_fuel_consumed : (List.sum (List.map abs distances)) * fuelConsumptionRate = 17 :=
by
  sorry

end final_position_total_fuel_consumed_l10_10055


namespace tan_xi_in_right_triangle_l10_10262

theorem tan_xi_in_right_triangle (α ξ : ℝ) (tan_of_alpha : tan α = real.root 4 3) :
  tan ξ = (real.root 4 3 / (1 - real.sqrt 3) - real.root 8 3) / (1 + (real.root 4 3 / (1 - real.sqrt 3)) * real.root 8 3) :=
by
  sorry

end tan_xi_in_right_triangle_l10_10262


namespace largest_multiple_of_7_less_than_neg85_l10_10422

theorem largest_multiple_of_7_less_than_neg85 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n < -85 ∧ n = -91 :=
by
  sorry

end largest_multiple_of_7_less_than_neg85_l10_10422


namespace find_coordinates_A_l10_10963

-- Define the point A
structure Point where
  x : ℝ
  y : ℝ

def PointA (a : ℝ) : Point :=
  { x := 3 * a + 2, y := 2 * a - 4 }

-- Define the conditions
def condition1 (a : ℝ) := (PointA a).y = 4

def condition2 (a : ℝ) := |(PointA a).x| = |(PointA a).y|

-- The coordinates solutions to be proven
def valid_coordinates (p : Point) : Prop :=
  p = { x := 14, y := 4 } ∨
  p = { x := -16, y := -16 } ∨
  p = { x := 3.2, y := -3.2 }

-- Main theorem to prove
theorem find_coordinates_A (a : ℝ) :
  (condition1 a ∨ condition2 a) → valid_coordinates (PointA a) :=
by
  sorry

end find_coordinates_A_l10_10963


namespace find_length_BC_l10_10671

variables (circle: Type) [metric_space circle] [normed_group circle]
variables
  (O A B C D : circle)
  (r : ℝ) -- radius of the circle
  (hO: is_center O r)
  (hAD: is_diameter A D O r)
  (hABC: is_chord A B C D O r)
  (hBO: dist B O = 5)
  (h_angle: ∠ A B O = 60)

theorem find_length_BC :
  dist B C = 5 :=
sorry

end find_length_BC_l10_10671


namespace midpoint_sum_l10_10806

theorem midpoint_sum (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -4) (hy2 : y2 = -7) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 1 :=
by
  rw [hx1, hy1, hx2, hy2]
  norm_num

end midpoint_sum_l10_10806


namespace max_value_dn_l10_10527

def a (n : ℕ) : ℕ := 100 + 2 * n * n
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_value_dn : ∀ n : ℕ, d n ≤ 2 :=
by
  -- Definitions used in proof
  intro n
  let an := a n
  let an1 := a (n + 1)
  have h : d n = Nat.gcd an an1, from rfl
  -- Simplifications and Euclidean steps go here (not shown in Lean statement)
  -- Maximum value assertion
  sorry

end max_value_dn_l10_10527


namespace simplify_trig_expression_l10_10356

theorem simplify_trig_expression :
  (sin (20 * Real.pi / 180) + sin (30 * Real.pi / 180) + sin (40 * Real.pi / 180) + sin (50 * Real.pi / 180) + sin (60 * Real.pi / 180) + sin (70 * Real.pi / 180) + sin (80 * Real.pi / 180) + sin (90 * Real.pi / 180)) /
  (cos (15 * Real.pi / 180) * cos (30 * Real.pi / 180) * cos (45 * Real.pi / 180)) = 8 * Real.sqrt 2 := 
by
  -- sorry to skip the proof
  sorry

end simplify_trig_expression_l10_10356


namespace sufficient_remedy_l10_10826

-- Definitions based on conditions
def aspirin_relieves_headache : Prop := true
def aspirin_relieves_knee_rheumatism : Prop := true
def aspirin_causes_heart_pain : Prop := true
def aspirin_causes_stomach_pain : Prop := true

def homeopathic_relieves_heart_issues : Prop := true
def homeopathic_relieves_stomach_issues : Prop := true
def homeopathic_causes_hip_rheumatism : Prop := true

def antibiotics_cure_migraines : Prop := true
def antibiotics_cure_heart_pain : Prop := true
def antibiotics_cause_stomach_pain : Prop := true
def antibiotics_cause_knee_pain : Prop := true
def antibiotics_cause_itching : Prop := true

def cortisone_relieves_itching : Prop := true
def cortisone_relieves_knee_rheumatism : Prop := true
def cortisone_exacerbates_hip_rheumatism : Prop := true

def warm_compress_relieves_itching : Prop := true
def warm_compress_relieves_stomach_pain : Prop := true

def severe_headache_morning : Prop := true
def impaired_ability_to_think : Prop := severe_headache_morning

-- Statement of the proof problem
theorem sufficient_remedy :
  (aspirin_relieves_headache ∧ antibiotics_cure_heart_pain ∧ warm_compress_relieves_itching ∧ warm_compress_relieves_stomach_pain) →
  (impaired_ability_to_think → true) :=
by
  sorry

end sufficient_remedy_l10_10826


namespace largest_multiple_of_7_smaller_than_negative_85_l10_10417

theorem largest_multiple_of_7_smaller_than_negative_85 :
  ∃ (n : ℤ), (∃ (k : ℤ), n = 7 * k) ∧ n < -85 ∧ ∀ (m : ℤ), (∃ (k : ℤ), m = 7 * k) ∧ m < -85 → m ≤ n := 
by
  use -91
  split
  { use -13
    norm_num }
  split
  { exact dec_trivial }
  { intros m hm
    cases hm with k hk
    cases hk with hk1 hk2
    have hk3 : k < -12 := by linarith
    have hk4 : k ≤ -13 := int.floor_le $ hk3
    linarith }


end largest_multiple_of_7_smaller_than_negative_85_l10_10417


namespace min_f_n_f_n_plus_one_l10_10982

noncomputable def min (a b : ℝ) : ℝ := if a ≤ b then a else b

def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem min_f_n_f_n_plus_one {α β : ℝ} (p q : ℝ) (n : ℤ)
  (h1 : n < α) (h2 : α < β) (h3 : β < n + 1)
  (h4 : α + β = -p) (h5 : α * β = q) :
  min (f n p q) (f (n + 1) p q) < 1 / 4 :=
sorry

end min_f_n_f_n_plus_one_l10_10982


namespace appropriate_survey_method_l10_10864

def survey_method_suitability (method : String) (context : String) : Prop :=
  match context, method with
  | "daily floating population of our city", "sampling survey" => true
  | "security checks before passengers board an airplane", "comprehensive survey" => true
  | "killing radius of a batch of shells", "sampling survey" => true
  | "math scores of Class 1 in Grade 7 of a certain school", "census method" => true
  | _, _ => false

theorem appropriate_survey_method :
  survey_method_suitability "census method" "daily floating population of our city" = false ∧
  survey_method_suitability "comprehensive survey" "security checks before passengers board an airplane" = false ∧
  survey_method_suitability "sampling survey" "killing radius of a batch of shells" = false ∧
  survey_method_suitability "census method" "math scores of Class 1 in Grade 7 of a certain school" = true :=
by
  sorry

end appropriate_survey_method_l10_10864


namespace initial_bushes_l10_10791

theorem initial_bushes (b : ℕ) (h1 : b + 4 = 6) : b = 2 :=
by {
  sorry
}

end initial_bushes_l10_10791


namespace even_function_inequality_l10_10042

variable {α : Type*} [LinearOrderedField α]

def is_even_function (f : α → α) : Prop := ∀ x, f x = f (-x)

-- The hypothesis and the assertion in Lean
theorem even_function_inequality
  (f : α → α)
  (h_even : is_even_function f)
  (h3_gt_1 : f 3 > f 1)
  : f (-1) < f 3 :=
sorry

end even_function_inequality_l10_10042


namespace maximum_x_plus_7y_exists_Q_locus_l10_10269

noncomputable def Q_locus (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem maximum_x_plus_7y (M : ℝ × ℝ) (h : Q_locus M.fst M.snd) : 
  ∃ max_value, max_value = 18 :=
  sorry

theorem exists_Q_locus (x y : ℝ) : 
  (∃ (Q : ℝ × ℝ), Q_locus Q.fst Q.snd) :=
  sorry

end maximum_x_plus_7y_exists_Q_locus_l10_10269


namespace length_of_chord_AB_l10_10886

/-- 
Given two circles intersecting at points A and B:
  Circle C1 with equation x^2 + y^2 + 2x - 12 = 0
  Circle C2 with equation x^2 + y^2 + 4x - 4y = 0
Prove that the length of chord AB is 4√2.
-/
theorem length_of_chord_AB (A B : ℝ × ℝ) 
  (hC1 : ∀ (P : ℝ × ℝ), (P = A ∨ P = B) → P.1^2 + P.2^2 + 2 * P.1 - 12 = 0)
  (hC2 : ∀ (P : ℝ × ℝ), (P = A ∨ P = B) → P.1^2 + P.2^2 + 4 * P.1 - 4 * P.2 = 0) :
  dist A B = 4 * real.sqrt 2 :=
sorry

end length_of_chord_AB_l10_10886


namespace min_value_log_sum_l10_10189

noncomputable theory
open_locale classical

theorem min_value_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : log x + log y = 1) :
  ∃ x y, x > 0 ∧ y > 0 ∧ log x + log y = 1 ∧ (2 / x + 5 / y) = 2 :=
begin
  use [2, 5],
  split,
  { norm_num }, 
  split,
  { norm_num },
  split,
  { exact dec_trivial },
  { norm_num }
end

end min_value_log_sum_l10_10189


namespace car_acceleration_l10_10508

open Real

theorem car_acceleration
  (d : ℝ := 5280) -- total distance the car travels in feet
  (t_f : ℝ := 60) -- total time in seconds
  (v_max : ℝ := 132) -- maximum speed in feet per second
  (a_min : ℝ := 6.6) -- minimum required acceleration or deceleration in feet per second squared
  (d_eq : d = 5280) -- distance condition
  (t_f_eq : t_f = 60) -- time condition
  (v_max_eq : v_max = 132) -- maximum speed condition
  (a_min_eq : a_min = 6.6) -- acceleration/deceleration condition) 
  (d_reach : ∃ t₀, t₀ < t_f ∧ (∀ t, 0 ≤ t ∧ t < t_f → 0 ≤ a (t)) ∧ (d (t₀) = 0) ∧ (d(t_f)=d)): 
  (∃ t, 0 ≤ t ∧ t ≤ t_f ∧ (abs (a t) ≥ a_min)) :=
  sorry

end car_acceleration_l10_10508


namespace polygon_sides_l10_10657

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_sides_l10_10657


namespace eq1_solution_eq2_solution_l10_10748

theorem eq1_solution (x : ℝ) : (x - 1)^2 - 1 = 15 ↔ x = 5 ∨ x = -3 := by sorry

theorem eq2_solution (x : ℝ) : (1 / 3) * (x + 3)^3 - 9 = 0 ↔ x = 0 := by sorry

end eq1_solution_eq2_solution_l10_10748


namespace distance_travelled_l10_10031

-- Given conditions
def speed := 20 -- speed in km/hr
def time := 2.5 -- time in hr

-- Prove that the distance traveled is 50 km
theorem distance_travelled : speed * time = 50 := 
by 
  sorry

end distance_travelled_l10_10031


namespace max_value_PF1_PF2_l10_10604

-- Define the point P on the ellipse.
structure PointOnEllipse (a b : ℝ) :=
  (x y : ℝ)
  (on_ellipse : (x ^ 2) / a ^ 2 + (y ^ 2) / b ^ 2 = 1)

-- Define the variables specific to this problem.
def a : ℝ := 5
def b : ℝ := 3
def PF1 (x0 : ℝ) : ℝ := 5 + (4 / 5) * x0
def PF2 (x0 : ℝ) : ℝ := 5 - (4 / 5) * x0

-- State the main theorem.
theorem max_value_PF1_PF2 (P : PointOnEllipse a b) :
  (PF1 P.x) * (PF2 P.x) ≤ 25 :=
sorry

end max_value_PF1_PF2_l10_10604


namespace units_digit_of_m_sq_plus_2_pow_m_is_3_l10_10707

theorem units_digit_of_m_sq_plus_2_pow_m_is_3 :
  let m := 2017^2 + 2^2017 in
  (m^2 + 2^m) % 10 = 3 :=
by
  sorry

end units_digit_of_m_sq_plus_2_pow_m_is_3_l10_10707


namespace max_distance_of_z_l10_10248

noncomputable def complex_max_distance (z : ℂ) : ℝ :=
  if h : |z + 1 - complex.i| = 2 then
    let M := 2 - complex.i in
    let C := -1 + complex.i in
    abs (M - C) + 2
  else
    0

theorem max_distance_of_z (z : ℂ) (h : |z + 1 - complex.i| = 2) : |z - 2 + complex.i| = sqrt 13 + 2 :=
by
  -- Proof to be provided
  sorry

end max_distance_of_z_l10_10248


namespace angle_bisector_midpoint_l10_10732

open EuclideanGeometry

variables {A B C D K M : Point} {BK : LineSegment}

-- Setup for the problem conditions
definition rectangle (ABCD : Type) [h : Rectangular ABCD] : Prop :=
  ∃ (A B C D : Point),
    right_angle ∠BAC ∧ right_angle ∠ABC ∧
    line_segment (A, B) ≅ line_segment (C, D) ∧
    line_segment (B, C) ≅ line_segment (D, A)

definition point_on_extension (D C K : Point) (pDK : line_segment D K) : Prop :=
  ∃ t > 1, K = D + t * (C - D)

definition equal_segments (BD DK : line_segment) : Prop :=
  length BD = length DK

-- Problem statement to be proved in Lean 4
theorem angle_bisector_midpoint (ABCD : Type) [h : Rectangular ABCD] (K : Point) (MDK : line_segment D K) (Line_segment BD D K M B K: LineSegment) : 
  rectangle ABCD →
  point_on_extension D C K MDK →
  equal_segments BD DK →
  angle_bisector (∠BAC) (line_segment (B, M))
  :=
-- Question theorem statement
begin
  sorry -- Proof is omitted.
end

end angle_bisector_midpoint_l10_10732


namespace infinitely_many_bad_numbers_l10_10740

theorem infinitely_many_bad_numbers : ∀ (P : ℕ → Prop), 
  (∀ n, P n ↔ ∃ a b, n = a + b ∧ (∀ p ∣ a, p < 1394) ∧ (∀ p ∣ b, p < 1394)) →
  ¬ ∃ N, ∀ n ≥ N, ¬P n :=
begin
  sorry
end

end infinitely_many_bad_numbers_l10_10740


namespace geometry_problem_l10_10731

variables (A B C D E : ℂ) [is_on_unit_circle A] [is_on_unit_circle B] [is_on_unit_circle C] [is_on_unit_circle D] [is_on_unit_circle E]

def is_on_unit_circle (z : ℂ) : Prop := abs z = 1

def angle_eq (a b c : ℂ) (θ : ℝ) : Prop := arg (b - a) = arg (c - b) + θ

theorem geometry_problem
  (h1 : angle_eq A B E (π/4))
  (h2 : angle_eq B E C (π/4))
  (h3 : angle_eq E C D (π/4)) :
  abs (B - A)^2 + abs (E - C)^2 = abs (E - B)^2 + abs (D - C)^2 :=
by sorry

end geometry_problem_l10_10731


namespace collinear_points_pentagon_l10_10750

variables {A B C D E F G H J K : Type} 
variables [affine_space A] [affine_space B] [affine_space C] [affine_space D] [affine_space E]

-- Definitions for the initial pentagon and points
variables {is_regular_pentagon : A → B → C → D → E → Prop}
variables {FD_perpendicular_DC : F → D → C → Prop}
variables {FB_perpendicular_BD : F → B → D → Prop}
variables {GD_perpendicular_DE : G → D → E → Prop}
variables {GA_perpendicular_AD : G → A → D → Prop}
variables {HA_perpendicular_AC : H → A → C → Prop}
variables {HB_perpendicular_BE : H → B → E → Prop}

variables {JH_parallel_AB : J → H → A → B → Prop}
variables {HK_parallel_AB : H → K → A → B → Prop}

variables {JA_perpendicular_CD : J → A → C → D → Prop}
variables {KB_perpendicular_DE : K → B → D → E → Prop}

-- The statement we want to prove
theorem collinear_points_pentagon 
  (A B C D E F G H J K : Type)
  [is_regular_pentagon A B C D E] 
  [FD_perpendicular_DC F D C] 
  [FB_perpendicular_BD F B D] 
  [GD_perpendicular_DE G D E] 
  [GA_perpendicular_AD G A D] 
  [HA_perpendicular_AC H A C] 
  [HB_perpendicular_BE H B E] 
  [JH_parallel_AB J H A B] 
  [HK_parallel_AB H K A B] 
  [JA_perpendicular_CD J A C D] 
  [KB_perpendicular_DE K B D E] 
  : collinear F G H J K :=
sorry

end collinear_points_pentagon_l10_10750


namespace find_ffive_l10_10590

-- Define the function f and the given conditions
variable (f : ℝ → ℝ)

-- Condition 1: The function satisfies f(x+2) = 1 / f(x) for any real number x
axiom f_periodic : ∀ x : ℝ, f(x + 2) = 1 / f(x)

-- Condition 2: Initial value of the function at x = 1
axiom f_at_one : f(1) = -5

-- Theorem to prove f(f(5)) = -1/5
theorem find_ffive : f(f(5)) = -(1 / 5) := sorry

end find_ffive_l10_10590


namespace integer_modulo_solution_l10_10017

theorem integer_modulo_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 137 ∧ 12345 ≡ n [ZMOD 137] ∧ n = 15 :=
sorry

end integer_modulo_solution_l10_10017


namespace part1_part2_l10_10638

def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}
def C : Set ℝ := {x | -1 < x ∧ x < 4}

theorem part1 : A ∩ (B 3)ᶜ = Set.Icc 3 5 := by
  sorry

theorem part2 : A ∩ B m = C → m = 8 := by
  sorry

end part1_part2_l10_10638


namespace rectangle_perimeter_l10_10927

theorem rectangle_perimeter (a b : ℕ) : 
  (2 * a + b = 6 ∨ a + 2 * b = 6 ∨ 2 * a + b = 9 ∨ a + 2 * b = 9) → 
  2 * a + 2 * b = 10 :=
by 
  sorry

end rectangle_perimeter_l10_10927


namespace circle_distance_k_l10_10005

theorem circle_distance_k (h1 : (5:ℝ)^2 + 12^2 = 25 + 144)
                          (h2 : ∃ k : ℝ, (0, k) ∈ circle 0 8)
                          (h3 : distance (origin: ℝ×ℝ) (5, 12) = 13)
                          (h4 : distance (origin: ℝ×ℝ) (0, k) + 5 = 13):
    k = 8 := 
by
  sorry

end circle_distance_k_l10_10005


namespace probability_top_card_is_king_l10_10082

open Real

def numKings (deck : List (String × String)) : ℕ :=
  deck.countp (λ c, c.1 = "King")

def totalCards (deck : List (String × String)) : ℕ :=
  deck.length

def probabilityTopKing (deck : List (String × String)) : ℚ :=
  numKings deck / totalCards deck

theorem probability_top_card_is_king :
  let deck := List.product ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"] ["spades", "hearts", "diamonds", "clubs"]
  probabilityTopKing deck = (1 : ℚ) / 13 :=
by
  sorry

end probability_top_card_is_king_l10_10082


namespace normal_curve_highest_point_l10_10205

noncomputable def highest_point_of_normal_curve : Prop :=
  ∀ (f : ℝ → ℝ), (∀ x, f(x) = (1 / (√(2 * π))) * exp (-(x^2) / 2)) →
  (∃ x, (∫ y in 0.2..+∞, f y = 0.5) ∧ (∀ y, f y ≤ f x) ∧ x = 0.2)

theorem normal_curve_highest_point : highest_point_of_normal_curve :=
  sorry

end normal_curve_highest_point_l10_10205


namespace derivative_of_exp_sin_l10_10648

theorem derivative_of_exp_sin (x : ℝ) : 
  (deriv (fun x => Real.exp x * Real.sin x)) x = Real.exp x * Real.sin x + Real.exp x * Real.cos x :=
sorry

end derivative_of_exp_sin_l10_10648


namespace fisherman_catch_total_l10_10470

theorem fisherman_catch_total :
  let bass := 32
  let trout := bass / 4
  let blue_gill := bass * 2
in bass + trout + blue_gill = 104 := by
  sorry

end fisherman_catch_total_l10_10470


namespace simple_interest_diff_l10_10390

theorem simple_interest_diff (P R T : ℝ) (hP : P = 3000) (hR : R = 4) (hT : T = 5) :
  let SI := (P * R * T) / 100 in
  P - SI = 2400 :=
by
  sorry

end simple_interest_diff_l10_10390


namespace number_of_positive_divisors_l10_10899

theorem number_of_positive_divisors (N : ℕ) (hN : N = 2^4 * 3^3 * 5^2 * 7) : 
  (finset.range(5).card) * (finset.range(4).card) * (finset.range(3).card) * (finset.range(2).card) = 120 :=
by
  -- Proof goes here
  sorry

end number_of_positive_divisors_l10_10899


namespace _l10_10917

noncomputable theorem unique_solution_x : (∃ x : ℝ, 0 < x ∧ x \sqrt(16 - x) + \sqrt(16 * x - x^3) ≥ 16) :=
  sorry

end _l10_10917


namespace log_expression_value_l10_10532

noncomputable def sqrt3PlusSqrt5 := Real.sqrt (3 + Real.sqrt 5)
noncomputable def sqrt3MinusSqrt5 := Real.sqrt (3 - Real.sqrt 5)
noncomputable def x := sqrt3PlusSqrt5 + sqrt3MinusSqrt5

theorem log_expression_value : Real.log10 x = 1 / 2 := by
  sorry

end log_expression_value_l10_10532


namespace equal_share_payments_l10_10690

theorem equal_share_payments (j n : ℝ) 
  (jack_payment : ℝ := 80) 
  (emma_payment : ℝ := 150) 
  (noah_payment : ℝ := 120)
  (liam_payment : ℝ := 200) 
  (total_cost := jack_payment + emma_payment + noah_payment + liam_payment) 
  (individual_share := total_cost / 4) 
  (jack_due := individual_share - jack_payment) 
  (emma_due := emma_payment - individual_share) 
  (noah_due := individual_share - noah_payment) 
  (liam_due := liam_payment - individual_share) 
  (j := jack_due) 
  (n := noah_due) : 
  j - n = 40 := 
by 
  sorry

end equal_share_payments_l10_10690


namespace EQ_value_l10_10268

theorem EQ_value (EFGH : Type) (Point : EFGH → EFGH → Prop)
  (Q_in_EH : ∀ E H : EFGH, ∃ Q : EFGH, Q ∈ EH ∧ EQ > HQ)
  (circumcenter : (EFGH → EFGH → EFGH → Type) → (EFGH → Type))
  (EF_len : ∀ E F : EFGH, len E F = 10)
  (angle_R1QR2 : ∀ R1 Q R2 : EFGH, ∠R1QR2 = 150) :
  ∃ x y : ℕ, EQ = √x + √y := by
  sorry

end EQ_value_l10_10268


namespace triangle_problem_l10_10665

open Real

noncomputable def triangle_side_b (a A B : ℝ) : ℝ :=
  a / (sin A) * (sin B)

theorem triangle_problem (h₁ : a = 1) (h₂ : A = π / 6) (h₃ : B = π / 3) : triangle_side_b a A B = √3 :=
by
  -- We skip the proof for the statement
  sorry

end triangle_problem_l10_10665


namespace distinct_solutions_subtract_eight_l10_10310

noncomputable def f (x : ℝ) : ℝ := (6 * x - 18) / (x^2 + 2 * x - 15)
noncomputable def equation := ∀ x, f x = x + 3

noncomputable def r_solutions (r s : ℝ) := (r > s) ∧ (f r = r + 3) ∧ (f s = s + 3)

theorem distinct_solutions_subtract_eight
  (r s : ℝ) (h : r_solutions r s) : r - s = 8 :=
sorry

end distinct_solutions_subtract_eight_l10_10310


namespace find_other_number_l10_10816

theorem find_other_number 
  {A B : ℕ} 
  (h_A : A = 24)
  (h_hcf : Nat.gcd A B = 14)
  (h_lcm : Nat.lcm A B = 312) :
  B = 182 :=
by
  -- Proof skipped
  sorry

end find_other_number_l10_10816


namespace eval_sqrt_fractions_l10_10538

theorem eval_sqrt_fractions :
  sqrt (1/25 + 1/36 + 1/49) = sqrt 1111 / 112 :=
by
  sorry

end eval_sqrt_fractions_l10_10538


namespace indigo_restaurant_average_rating_l10_10764

theorem indigo_restaurant_average_rating :
  let n_5stars := 6
  let n_4stars := 7
  let n_3stars := 4
  let n_2stars := 1
  let total_reviews := 18
  let total_stars := n_5stars * 5 + n_4stars * 4 + n_3stars * 3 + n_2stars * 2
  (total_stars / total_reviews : ℝ) = 4 :=
by
  sorry

end indigo_restaurant_average_rating_l10_10764


namespace total_units_is_531_l10_10846

noncomputable def total_units_in_development : Nat :=
  let regular_odd_floors_units := 13 * 14
  let regular_even_floors_units := 12 * 12
  let regular_units := regular_odd_floors_units + regular_even_floors_units
  
  let average_luxury_units_per_floor := (6 + 10) / 2
  let luxury_units := 20 * average_luxury_units_per_floor
  
  let penthouse_units := 10 * 2
  
  let uncounted_units := 4 + 6
  
  let residential_units := regular_units + luxury_units + penthouse_units + uncounted_units
  
  let commercial_units := 3 * 5
  
  residential_units + commercial_units

theorem total_units_is_531
  (regular_floors_odd_units : 14)
  (regular_floors_even_units : 12)
  (luxury_floors_min_units : 6)
  (luxury_floors_max_units : 10)
  (penthouse_floors_units : 2)
  (commercial_floors_units : 5)
  (uncounted_floor1_units : 4)
  (uncounted_floor2_units : 6) :
  total_units_in_development = 531 := by
  sorry

end total_units_is_531_l10_10846


namespace circle_distance_k_l10_10004

theorem circle_distance_k (h1 : (5:ℝ)^2 + 12^2 = 25 + 144)
                          (h2 : ∃ k : ℝ, (0, k) ∈ circle 0 8)
                          (h3 : distance (origin: ℝ×ℝ) (5, 12) = 13)
                          (h4 : distance (origin: ℝ×ℝ) (0, k) + 5 = 13):
    k = 8 := 
by
  sorry

end circle_distance_k_l10_10004


namespace inequality_proof_l10_10176

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (1 + a + a * b)) + (b / (1 + b + b * c)) + (c / (1 + c + c * a)) ≤ 1 :=
by
  sorry

end inequality_proof_l10_10176


namespace second_test_point_4292_l10_10802

theorem second_test_point_4292 (a b : ℝ) (h_start : a = 2000) (h_end : b = 8000) : 
  let l := b - a in 
  let x1 := a + 0.618 * l in 
  x2 = b - (x1 - a) :=
begin
  have h_l : l = 6000,
  { rw [h_start, h_end],
    norm_num, },
  have h_x1 : x1 = 5708,
  { rw [h_l, h_start],
    norm_num, 
    ring, },
  have h_x2 : x2 = 4292,
  { rw [h_x1, h_start, h_end],
    norm_num,
    ring, },
  exact h_x2,
end

end second_test_point_4292_l10_10802


namespace gcd_15_n_eq_3_count_l10_10163

open Nat

theorem gcd_15_n_eq_3_count :
  (Finset.card {n ∈ (Finset.range 101) | n > 0 ∧ gcd 15 n = 3}) = 27 :=
by
  sorry

end gcd_15_n_eq_3_count_l10_10163


namespace quadratic_solution_l10_10156

noncomputable def solve_quadratic (a b c : ℕ) : ℕ :=
  let m := 9 in
  let n := 1 in
  let p := 10 in
  m + n + p

theorem quadratic_solution :
  ∃ (m n p : ℕ), (m = 9) ∧ (n = 1) ∧ (p = 10) ∧ solve_quadratic 5 (-9) 4 = 20 :=
by {
  use [9, 1, 10],
  split; try {refl},
  split; try {refl},
  exact rfl,
  sorry
}

end quadratic_solution_l10_10156


namespace final_amount_is_19_75_l10_10483

noncomputable def finalAmountAfterBets (initialAmount : ℝ) (bets : List (ℝ → ℝ)) : ℝ :=
  bets.foldl (λ amount bet => bet amount) initialAmount

def oneThirdWin (amount : ℝ) : ℝ := (4 / 3) * amount
def oneThirdLoss (amount : ℝ) : ℝ := (2 / 3) * amount
def twoThirdsWin (amount : ℝ) : ℝ := (5 / 3) * amount
def twoThirdsLoss (amount : ℝ) : ℝ := (1 / 3) * amount

theorem final_amount_is_19_75 :
  finalAmountAfterBets 100 [oneThirdWin, oneThirdWin, twoThirdsLoss, twoThirdsLoss] = 1600 / 81 :=
sorry

end final_amount_is_19_75_l10_10483


namespace concentration_of_spirit_in_vessel_a_l10_10767

theorem concentration_of_spirit_in_vessel_a :
  ∀ (x : ℝ), 
    (∀ (v1 v2 v3 : ℝ), v1 * (x / 100) + v2 * (30 / 100) + v3 * (10 / 100) = 15 * (26 / 100) →
      v1 + v2 + v3 = 15 →
      v1 = 4 → v2 = 5 → v3 = 6 →
      x = 45) :=
by
  intros x v1 v2 v3 h h_volume h_v1 h_v2 h_v3
  sorry

end concentration_of_spirit_in_vessel_a_l10_10767


namespace number_of_valid_three_digit_even_numbers_l10_10998

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l10_10998


namespace find_a_l10_10112

noncomputable def g (x : ℝ) := 5 * x - 7

theorem find_a (a : ℝ) (h : g a = 0) : a = 7 / 5 :=
sorry

end find_a_l10_10112


namespace sum_of_fractions_l10_10875

theorem sum_of_fractions :
  (3 / 50) + (5 / 500) + (7 / 5000) = 0.0714 :=
by
  sorry

end sum_of_fractions_l10_10875


namespace cubic_inequality_solution_l10_10119

theorem cubic_inequality_solution (x : ℝ) (h : 0 ≤ x) : 
  x^3 - 9*x^2 - 16*x > 0 ↔ 16 < x := 
by 
  sorry

end cubic_inequality_solution_l10_10119


namespace f_composition_l10_10719

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^(1 - x) else 1 - Real.logb 2 x

theorem f_composition : f (f -1) = -1 := by
  sorry

end f_composition_l10_10719


namespace volume_of_sphere_l10_10184

noncomputable def radius_from_tangent (tan_angle : ℝ) : ℝ :=
if tan_angle = -√3 then 2 else 0 -- derived R = 2 based on tan(∠ACB) = -√3

theorem volume_of_sphere
  (h1 : ∃ (O : Type) (A B C : O), 0 < 1) -- Existence of sphere and points
  (h2 : ∀ (R : ℝ), 0 < R → (R/2)^2 + (3 / (2 * (√3 / 2)))^2 = R^2) -- Geometry condition
  (h3 : AB = 3)
  (h4 : tan ∠ACB = -√3) :
  volume_of_sphere = (4/3) * real.pi * 2^3 :=
by
  -- We'll assume that the exact steps have been proven and construct the
  -- proof goal directly with the expected final result.
  sorry

end volume_of_sphere_l10_10184


namespace square_of_binomial_formula_correct_l10_10807

theorem square_of_binomial_formula_correct :
  ((-x - y) * (-x + y)) = (-x + y) * (-x - y) :=
begin
  sorry
end

end square_of_binomial_formula_correct_l10_10807


namespace circle_radius_7_5_l10_10603

-- Define the square PQRS with side length 10 feet
def side_length : ℝ := 10
def P : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (side_length, 0)
def R : ℝ × ℝ := (side_length, side_length)
def S : ℝ × ℝ := (0, side_length)
def diagonal_length : ℝ := real.sqrt (side_length^2 + side_length^2)

-- Define the center O of the circle, the radius of which we will find
variable (O : ℝ × ℝ)

-- Define the condition that the circle is tangent to side QS and passes through P and R
def circle_passes_through_P_R_and_tangent_to_QS (r : ℝ) : Prop :=
  dist O P = r ∧ dist O R = r ∧ O.2 = side_length - r

theorem circle_radius_7_5 :
  ∃ r : ℝ, circle_passes_through_P_R_and_tangent_to_QS O r ∧ r = 7.5 :=
begin
  sorry
end

end circle_radius_7_5_l10_10603


namespace regular_polygon_l10_10091

theorem regular_polygon
  (n : ℕ) (n_gt_2 : 2 < n)
  (vertices : Fin n → Point)
  (O : Point)
  (equal_interior_angles : ∀ i : Fin n, ∠(vertices i) = ∠(vertices 0))
  (equal_side_angles : ∀ i j : Fin n, ∠(vertices i O vertices (i + 1 % n))  = ∠(vertices 0 O vertices 1)) :
  ∀ i j : Fin n, dist (vertices i) (vertices (i + 1 % n)) = dist (vertices j) (vertices (j + 1 % n)) :=
by
  sorry

end regular_polygon_l10_10091


namespace area_of_right_triangle_l10_10368

-- Definition of a right triangle with specific angles and altitude
def right_triangle (a b c : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ), 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  ∠ A B C = π / 2 ∧ ∠ B A C = π / 3 ∧ ∠ A C B = π / 6 ∧ 
  line_through A C = line_through B (foot_of_perpendicular B (line_through A C)) ∧ 
  dist B (foot_of_perpendicular B (line_through A C)) = 5

-- Statement of the proof problem
theorem area_of_right_triangle :
  ∀ (a b c : ℝ), right_triangle a b c → 
  area (triangle a b c) = 25 * real.sqrt 3 :=
by
  intros a b c h
  sorry

end area_of_right_triangle_l10_10368


namespace graph_point_sum_l10_10975

theorem graph_point_sum (g : ℝ → ℝ) (h : g 3 = 10) :
  let y := 4 * g (3 * 1) + 6 / 3 in
  1 + y = 49 / 3 :=
by
  -- proof goes here
  sorry

end graph_point_sum_l10_10975


namespace min_attempts_to_pair_keys_suitcases_l10_10173

theorem min_attempts_to_pair_keys_suitcases (n : ℕ) : ∃ p : ℕ, (∀ (keyOpen : Fin n → Fin n), ∃ f : (Fin n × Fin n) → Bool, ∀ (i j : Fin n), i ≠ j → (keyOpen i = j ↔ f (i, j) = tt)) ∧ p = Nat.choose n 2 := by
  sorry

end min_attempts_to_pair_keys_suitcases_l10_10173


namespace general_formulas_and_no_k_in_01_l10_10971

-- Definitions for sequences
def a (n : ℕ) : ℕ := 2^(4 - n)
def b (n : ℕ) : ℕ := n^2 - 7 * n + 14

-- Hypothesis that the first three terms of the sequences are equal
axiom first_three_terms_equal :
  a 1 = b 1 ∧ a 2 = b 2 ∧ a 3 = b 3

-- Hypothesis for the given sequence equation
axiom sequence_equation (n : ℕ) (h : n > 0) :
  ∑ i in Finset.range n, 2^i * a i.succ = 8 * n

-- Hypothesis for the arithmetic property of the sequence {b_{n+1} - b_n}
axiom arithmetic_b_seq (n : ℕ) :
  ((b (n + 1) - b n) = (b 2 - b 1) + n * 2)

-- Problem statement
theorem general_formulas_and_no_k_in_01 :
  (∀ n : ℕ, 0 < n → a n = 2^(4 - n) ∧ b n = n^2 - 7 * n + 14) ∧
  ¬(∃ k : ℕ, 0 < k ∧ b k - a k ∈ set.Ioo 0 1) :=
by
  sorry

end general_formulas_and_no_k_in_01_l10_10971


namespace sandra_savings_l10_10354

theorem sandra_savings :
  let num_notepads := 8
  let original_price_per_notepad := 3.75
  let discount_rate := 0.25
  let discount_per_notepad := original_price_per_notepad * discount_rate
  let discounted_price_per_notepad := original_price_per_notepad - discount_per_notepad
  let total_cost_without_discount := num_notepads * original_price_per_notepad
  let total_cost_with_discount := num_notepads * discounted_price_per_notepad
  let total_savings := total_cost_without_discount - total_cost_with_discount
  total_savings = 7.50 :=
sorry

end sandra_savings_l10_10354


namespace smallest_positive_solution_l10_10150

theorem smallest_positive_solution :
  ∃ x > 0, tan (4 * x) + tan (5 * x) = sec (5 * x) ∧ x = π / 26 := 
begin
  use π / 26,
  split,
  { exact real.pi_pos.trans (by norm_num), },
  split,
  { sorry, },
  { refl, }
end

end smallest_positive_solution_l10_10150


namespace sacks_discarded_per_day_l10_10401

theorem sacks_discarded_per_day 
  (h : harvest : ℕ) -- 74 sacks per day
  (d : days : ℕ) -- 51 days
  (f : final : ℕ) -- 153 final sacks 
  (h_eq : harvest = 74)
  (d_eq : days = 51)
  (f_eq : final = 153) :
  ∃ discard_per_day : ℕ, discard_per_day = 71 := 
by 
  let total_harvest := harvest * days
  let total_discarded := total_harvest - final
  let discard_per_day := total_discarded / days
  use discard_per_day
  have h_total_harvest : total_harvest = 74 * 51 := by rw [h_eq, d_eq, nat.mul_comm]
  have h_total_discarded : total_discarded = 74 * 51 - 153 := by rw [h_total_harvest, f_eq]
  have h_discard_per_day : discard_per_day = (74 * 51 - 153) / 51 := by rw [←h_total_discarded, d_eq]
  have discard_correct : discard_per_day = 71 := 
    by norm_num at h_discard_per_day
  exact discard_correct

end sacks_discarded_per_day_l10_10401


namespace train_pass_time_l10_10812

-- Define the conditions
def train_length : ℝ := 275
def train_speed_kmph : ℝ := 60
def man_speed_kmph : ℝ := 6
def kmph_to_mps_factor : ℝ := 5/18

-- Define the proof problem
theorem train_pass_time :
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph in
  let relative_speed_mps := relative_speed_kmph * kmph_to_mps_factor in
  let time_to_pass := train_length / relative_speed_mps in
  |round(time_to_pass) - 15| < 1 :=
by
  sorry

end train_pass_time_l10_10812


namespace martin_correct_answers_l10_10336

theorem martin_correct_answers : 
  ∀ (Campbell_correct Kelsey_correct Martin_correct : ℕ), 
  Campbell_correct = 35 →
  Kelsey_correct = Campbell_correct + 8 →
  Martin_correct = Kelsey_correct - 3 →
  Martin_correct = 40 := 
by
  intros Campbell_correct Kelsey_correct Martin_correct h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  rw [h3]
  rfl

end martin_correct_answers_l10_10336


namespace zeros_of_function_l10_10608

noncomputable def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 2 then 1 - |2 * x - 3| else
  if x ≥ 2 then (1 / 2) * f (1 / 2 * x) else 0

theorem zeros_of_function : (set.Icc 1 2015).count (λ x, (2 * x * f x - 3) = 0) = 11 :=
sorry

end zeros_of_function_l10_10608


namespace rectangle_enclosed_by_four_lines_l10_10552

theorem rectangle_enclosed_by_four_lines : 
  let h_lines := 5
  let v_lines := 5
  (choose h_lines 2) * (choose v_lines 2) = 100 :=
by {
  sorry
}

end rectangle_enclosed_by_four_lines_l10_10552


namespace fraction_BC_AD_l10_10678

-- Defining points and segments
variables (A B C D : Point)
variable (len : Point → Point → ℝ) -- length function

-- Conditions
axiom AB_eq_3BD : len A B = 3 * len B D
axiom AC_eq_7CD : len A C = 7 * len C D
axiom B_mid_AD : 2 * len A B = len A D

-- Theorem: Proving the fraction of BC relative to AD is 2/3
theorem fraction_BC_AD : (len B C) / (len A D) = 2 / 3 :=
sorry

end fraction_BC_AD_l10_10678


namespace ellipse_equation_fixed_point_M1N_l10_10206

open Real

-- Define the initial conditions of the ellipsoid problem
variable (a : ℝ) (h_a : 1 < a)
def ellipse_eq : Prop := (a ^ 2) = 2

theorem ellipse_equation (c : ℝ) (h_1: a^2 = 1 + c^2) (h_2 : 1 / c + 1 / a = c / (a * (a - c))) : 
    ∀ x y : ℝ, x^2 / 2 + y^2 = 1 :=
by
  sorry

theorem fixed_point_M1N (t : ℝ) (t_nonzero : t ≠ 0) :
    ∀ M N F : ℝ × ℝ, 
    F = (1, 0) →
    ∃ M_1 : ℝ × ℝ, M_1.snd + M_1.fst = 2 → 
    M_1N M F = (3 / 2, 0) :=
by
  sorry

end ellipse_equation_fixed_point_M1N_l10_10206


namespace abs_sum_a_b_eq_l10_10966

theorem abs_sum_a_b_eq (a b : ℝ) (h : sqrt (2 * a + 6) + abs (b - sqrt 2) = 0) : abs (a + b) = 3 - sqrt 2 :=
sorry

end abs_sum_a_b_eq_l10_10966


namespace volume_cone_div_pi_l10_10839

-- Define that a cone is formed from a 240-degree sector of a circle with radius 18
def radius_circle : ℝ := 18
def angle_sector : ℝ := 240

-- Define the volume formula and conditions for the cone
theorem volume_cone_div_pi :
  let r := 12 in       -- derived from the arc length calculation
  let h := 6 * Real.sqrt 5 in -- height from the Pythagorean theorem
  let V := (1 / 3) * Real.pi * r^2 * h in
  V / Real.pi = 864 * Real.sqrt 5 :=
by
  sorry

end volume_cone_div_pi_l10_10839


namespace one_perpendicular_line_infinite_perpendicular_planes_infinite_parallel_lines_one_parallel_plane_l10_10121

-- Definitions:
-- A point outside a plane and drawing some lines and planes based on that.
def PointOutOfPlane (P : Type) (Plane : Type) := ∃ (p : P) (pl : Plane), p ∉ pl

-- The theorems to be proven. Note: no proof required.
theorem one_perpendicular_line {P Plane : Type} (h : PointOutOfPlane P Plane) : ∃! (l : P → P), ∃ (pl : Plane), (l ≠ pl) := sorry

theorem infinite_perpendicular_planes {P Plane : Type} (h : PointOutOfPlane P Plane) : ∃ (S : Type), (infinite (S → Plane)) := sorry

theorem infinite_parallel_lines {P : Type} {l1 l2 : P → P} (h : l1 ≠ l2): infinite (P) := sorry

theorem one_parallel_plane {X Plane : Type} (h : X ∈ Plane) : ∃! (pl : Plane), (Plane ≠ pl) := sorry

end one_perpendicular_line_infinite_perpendicular_planes_infinite_parallel_lines_one_parallel_plane_l10_10121


namespace range_of_a_undetermined_l10_10981

theorem range_of_a_undetermined (a : ℝ) : ¬(∃ r : ℝ, ∀ a, 15 + a*i > 14 → r) := 
begin
  sorry
end

end range_of_a_undetermined_l10_10981


namespace votes_for_winning_candidate_l10_10818

theorem votes_for_winning_candidate (V : ℝ) (h1 : ∃ V, V > 0)
    (h2 : (0.65 * V) = (0.35 * V + 300)) :
    0.65 * V = 650 :=
by
  -- Assuming the total votes is given by V and solving the equation
  have V_eq : V = 1000 := by
    linarith
  -- Substituting back to find the total votes for the winning candidate
  have votes_winner : 0.65 * 1000 = 650 := by
    -- Simple arithmetic
    norm_num
  exact votes_winner

end votes_for_winning_candidate_l10_10818


namespace largest_multiple_of_7_smaller_than_neg_85_l10_10425

theorem largest_multiple_of_7_smaller_than_neg_85 :
  ∃ k : ℤ, 7 * k < -85 ∧ (∀ m : ℤ, 7 * m < -85 → 7 * m ≤ 7 * k) ∧ 7 * k = -91 :=
by
  simp only [exists_prop, and.assoc],
  sorry

end largest_multiple_of_7_smaller_than_neg_85_l10_10425


namespace probability_full_house_after_rerolling_l10_10161

theorem probability_full_house_after_rerolling
  (a b c : ℕ)
  (h0 : a ≠ b)
  (h1 : c ≠ a)
  (h2 : c ≠ b) :
  (2 / 6 : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end probability_full_house_after_rerolling_l10_10161


namespace Taps_fill_bucket_time_l10_10755

theorem Taps_fill_bucket_time :
  ∀ (bucket_volume : ℝ) (rate_A rate_B time_B : ℝ),
    bucket_volume = 36 ∧
    rate_A = 3 ∧
    time_B = 20 ∧
    rate_B = (1/3 * bucket_volume) / time_B →
    bucket_volume / (rate_A + rate_B) = 10 :=
by
  intros bucket_volume rate_A rate_B time_B h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h4
  rw [h1, h2, h3]
  have h_rate_B : rate_B = 12/20 := by
    rw h4
  rw [h_rate_B]
  have h_combined : 3 + 0.6 = 3.6 := by norm_num
  rw [h_combined]
  norm_num

end Taps_fill_bucket_time_l10_10755


namespace polynomial_approximation_l10_10744

open Real

theorem polynomial_approximation {m n : ℕ} (hm : m > 0) (hn : n > 0) :
  ∃ (p : Polynomial ℤ), ∃ I : Ioo (1 / n - 1 / (2 * n)) (1 / n + 1 / (2 * n)),
  ∀ x ∈ I, abs (p.eval x - m / n: ℝ) ≤ 1 / n^2 := by
  sorry

end polynomial_approximation_l10_10744


namespace compute_100M_plus_N_l10_10162

-- Definition of f(S)
def f (S : Set ℤ) : ℕ :=
  { k // 0 ≤ k ∧ k < 2019 ∧ ∃ (s1 s2 : ℤ), s1 ∈ S ∧ s2 ∈ S ∧ s1 - s2 = k }.toFinset.card

-- Definition of x_m
def x_m (m : ℕ) : ℕ :=
  Inf { f (S_i) | (S : ℕ → Set ℤ) // ∀ (i : ℕ), 1 ≤ i ∧ i ≤ m → S i ≠ ∅ ∧ ∀ i j, i ≠ j → S i ∩ S j = ∅ }

-- Definition of M
def M : ℕ :=
  Inf {x_m (m) | m : ℕ }

-- Definition of N
def N : ℕ :=
  {m | x_m (m) = M}.card

-- The final proof
theorem compute_100M_plus_N : 100 * M + N = 202576 :=
  by
  sorry

end compute_100M_plus_N_l10_10162


namespace pentagon_angle_equality_l10_10951

variables {α : Real} -- Since we are dealing with real numbers
variables (A B C D E : Point) -- Assuming a type Point for the vertices of the pentagon

-- Definition of angles
variables (angle_BAE angle_BCD angle_CDE : ℝ)

-- Definition of side lengths
variables (BC CD DE : ℝ)

-- Given conditions
def given_conditions :=  
  angle_BAE = 3 * α ∧
  BC = CD ∧
  CD = DE ∧
  angle_BCD = 180 - 2 * α ∧
  angle_CDE = 180 - 2 * α

-- The theorem to be proved
theorem pentagon_angle_equality (h : given_conditions α A B C D E angle_BAE angle_BCD angle_CDE BC CD DE) :
  ∃ β : ℝ, (angle BAC = β ∧ angle CAD = β ∧ angle DAE = β) :=
sorry

end pentagon_angle_equality_l10_10951


namespace smallest_positive_solution_tan_sec_eq_l10_10153

theorem smallest_positive_solution_tan_sec_eq 
  (x : ℝ) 
  (hx : x > 0)
  (hx_rad : ∃ y : ℝ, x = y * real.pi) 
  (h_eq : real.tan (4 * x) + real.tan (5 * x) = real.sec (5 * x)) :
  x = real.pi / 18 :=
sorry

end smallest_positive_solution_tan_sec_eq_l10_10153


namespace girl_scouts_with_signed_permission_slips_l10_10065

theorem girl_scouts_with_signed_permission_slips (
  (total_scouts : ℕ) 
  (signed_permission_slips : ℕ)
  (boy_scouts : ℕ)
  (boy_scouts_with_signed_slips : ℕ)
  (girl_scouts_with_signed_slips : ℕ)
  (H1 : signed_permission_slips = 80 * total_scouts / 100)
  (H2 : boy_scouts = 40 * total_scouts / 100)
  (H3 : boy_scouts_with_signed_slips = 75 * boy_scouts / 100)
  (H4 : girl_scouts_with_signed_slips = signed_permission_slips - boy_scouts_with_signed_slips)
  (H5 : total_scouts = 100)
  
  ) : ((girl_scouts_with_signed_slips * 100 / (total_scouts - boy_scouts)) = 83) := 
by
  sorry

end girl_scouts_with_signed_permission_slips_l10_10065


namespace area_of_quadrilateral_ABCD_l10_10683

theorem area_of_quadrilateral_ABCD
  (ABE_right : ∠AEB = 90)
  (BEC_right : ∠BEC = 90)
  (CDE_right : ∠CED = 90)
  (AEB_45_deg : ∠AEB = 45)
  (BEC_60_deg : ∠BEC = 60)
  (CED_45_deg : ∠CED = 45)
  (AE_30 : AE = 30) :
  let AB := 15 * Real.sqrt 2,
      BE := 15 * Real.sqrt 2,
      BC := 15 * Real.sqrt 6,
      CE := 7.5 * Real.sqrt 2,
      CD := 7.5 * Real.sqrt 2,
      DE := 7.5 * Real.sqrt 2 in
  0.5 * AB * BE + 0.5 * BC * CE + 0.5 * CD * DE = 281.25 + 225 * Real.sqrt 3 :=
by
  sorry

end area_of_quadrilateral_ABCD_l10_10683


namespace ceil_sum_example_l10_10536

theorem ceil_sum_example : 
  (⌈Real.sqrt (25 / 9)⌉ + ⌈25 / 9⌉ + ⌈(25 / 9) ^ 2⌉) = 13 :=
by 
suffices h1: (⌈Real.sqrt (25 / 9)⌉ + ⌈25 / 9⌉ + ⌈(25 / 9) ^ 2⌉ = 2 + 3 + 8), by rwa [h1]
-- prove summands separately
have h2: ⌈Real.sqrt (25 / 9)⌉ = 2, sorry
have h3: ⌈25 / 9⌉ = 3, sorry
have h4: ⌈(25 / 9) ^ 2⌉ = 8, sorry 
-- summing
ring

end ceil_sum_example_l10_10536


namespace sufficient_but_not_necessary_condition_l10_10649

theorem sufficient_but_not_necessary_condition (k : ℝ) : 
  (k > 3) ↔ (∃ x y : ℝ, (x^2 / (k-3) - y^2 / (k+3) = 1)) ∧ (|k| > 3) :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_l10_10649


namespace nine_digit_repetition_divisible_l10_10070

theorem nine_digit_repetition_divisible (a b c : ℕ) :
  let n := 1001001 * (100*a + 10*b + c) in
    (100001000 * a + 10001000 * b + 1001000 * c + 
     1000100 * a + 100010 * b + 1001 * c) = n → n % 1001001 = 0 :=
by sorry

end nine_digit_repetition_divisible_l10_10070


namespace compute_fraction_power_l10_10107

theorem compute_fraction_power :
  8 * (1 / 4) ^ 4 = 1 / 32 := 
by
  sorry

end compute_fraction_power_l10_10107


namespace cube_covered_with_parallelograms_are_squares_and_rectangles_l10_10357

theorem cube_covered_with_parallelograms_are_squares_and_rectangles
  (edge_length : ℝ) (n : ℕ) (area : ℝ)
  (h1 : edge_length = 1)
  (h2 : n = 6)
  (h3 : ∀ (i : ℕ), i < n → (parallelogram_area i = 1))
  (parallelogram_area : ℕ → ℝ) :
  ∀ (i : ℕ), i < n → (is_square (parallelogram i) ∧ is_rectangle (parallelogram i)) :=
by
  sorry

-- Definitions for the sake of completeness in Lean
def parallelogram (i : ℕ) : Type := sorry
def is_square (p : Type) : Prop := sorry
def is_rectangle (p : Type) : Prop := sorry

end cube_covered_with_parallelograms_are_squares_and_rectangles_l10_10357


namespace alfred_gain_percent_l10_10502

theorem alfred_gain_percent (P : ℝ) (R : ℝ) (S : ℝ) (H1 : P = 4700) (H2 : R = 800) (H3 : S = 6000) : 
  (S - (P + R)) / (P + R) * 100 = 9.09 := 
by
  rw [H1, H2, H3]
  norm_num
  sorry

end alfred_gain_percent_l10_10502


namespace average_increase_in_food_expenditure_l10_10861

theorem average_increase_in_food_expenditure (x : ℝ) : 
  let y := 0.254 * x + 0.321 in
  let y' := 0.254 * (x + 1) + 0.321 in
  y' - y = 0.254 :=
by {
  sorry
}

end average_increase_in_food_expenditure_l10_10861


namespace gcd_of_polynomial_l10_10607

theorem gcd_of_polynomial (b : ℕ) (hb : b % 780 = 0) : Nat.gcd (5 * b^3 + 2 * b^2 + 6 * b + 65) b = 65 := by
  sorry

end gcd_of_polynomial_l10_10607


namespace monotonic_decreasing_interval_l10_10383

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

-- Define the proof that the interval (-1/3, 1) is where f is monotonic decreasing
theorem monotonic_decreasing_interval : 
  ∀ x : ℝ, -1/3 < x ∧ x < 1 → f'(x) < 0 :=
by
  -- Here we would provide the detailed mathematical proof if required
  sorry

end monotonic_decreasing_interval_l10_10383


namespace incircle_touches_vertex_l10_10613

open Real

-- Definitions for hyperbola and triangle
variables {F1 F2 M N P : Point}

-- vertices and foci of the hyperbola
axiom hyperbola (F1 F2 M N P : Point) : 
  is_vertex M ∧ is_vertex N ∧ is_focus F1 ∧ is_focus F2 ∧ on_hyperbola P M N F1 F2

-- Statement of the theorem
theorem incircle_touches_vertex : 
  ∀ (P : Point), on_hyperbola P M N F1 F2 →
  touches_incircle (P, F1, F2) (F1, F2) = M ∨ touches_incircle (P, F1, F2) (F1, F2) = N :=
sorry

end incircle_touches_vertex_l10_10613


namespace min_additional_weight_l10_10342

theorem min_additional_weight (std_weight max_weight : ℕ) (h1 : std_weight = 100) (h2 : max_weight = 210) 
  (h3 : ∀ w, w > std_weight → w ≤ 2 * (std_weight + 5)) : 5 ∈ x :=
by
  sorry

end min_additional_weight_l10_10342


namespace ambulance_reachable_area_l10_10261

theorem ambulance_reachable_area :
  let travel_time_minutes := 8
  let travel_time_hours := (travel_time_minutes : ℝ) / 60
  let speed_on_road := 60 -- speed in miles per hour
  let speed_off_road := 10 -- speed in miles per hour
  let distance_on_road := speed_on_road * travel_time_hours
  distance_on_road = 8 → -- this verifies the distance covered on road
  let area := (2 * distance_on_road) ^ 2
  area = 256 := sorry

end ambulance_reachable_area_l10_10261


namespace wizard_digits_and_gps_l10_10037

theorem wizard_digits_and_gps :
  ∃ (Б А Н К С Д : Nat),
    5 * (1000*Б + 100*А + 10*Н + К) = 6 * (100*С + 10*А + Д) ∧
    Б = 1 ∧ А = 0 ∧ Н = 8 ∧ К = 6 ∧ С = 9 ∧ Д = 5 ∧
    let gps := (С * 6 + 1, С * 6, 6 * 6, Д, Б, (К / 2), Н - 1),
    gps = (55.5430, 5317) := by
    sorry

end wizard_digits_and_gps_l10_10037


namespace community_service_arrangements_l10_10581

noncomputable def total_arrangements : ℕ :=
  let case1 := Nat.choose 6 3
  let case2 := 2 * Nat.choose 6 2
  let case3 := case2
  case1 + case2 + case3

theorem community_service_arrangements :
  total_arrangements = 80 :=
by
  sorry

end community_service_arrangements_l10_10581


namespace john_fraction_given_to_mother_l10_10293

theorem john_fraction_given_to_mother (x : ℚ) :
  (x + 3/10) * 200 + 65 = 200 → x = 3/8 :=
by
  intro h
  -200x - 60 = -75
  x = 3 / 8
  sorry

end john_fraction_given_to_mother_l10_10293


namespace distinct_exponentiations_are_four_l10_10890

def power (a b : ℕ) : ℕ := a^b

def expr1 := power 3 (power 3 (power 3 3))
def expr2 := power 3 (power (power 3 3) 3)
def expr3 := power (power (power 3 3) 3) 3
def expr4 := power (power 3 (power 3 3)) 3
def expr5 := power (power 3 3) (power 3 3)

theorem distinct_exponentiations_are_four : 
  (expr1 ≠ expr2 ∧ expr1 ≠ expr3 ∧ expr1 ≠ expr4 ∧ expr1 ≠ expr5 ∧
   expr2 ≠ expr3 ∧ expr2 ≠ expr4 ∧ expr2 ≠ expr5 ∧
   expr3 ≠ expr4 ∧ expr3 ≠ expr5 ∧
   expr4 ≠ expr5) :=
sorry

end distinct_exponentiations_are_four_l10_10890


namespace compare_expressions_l10_10887

-- Define the theorem statement
theorem compare_expressions (x : ℝ) : (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry -- The proof is omitted.

end compare_expressions_l10_10887


namespace cone_base_radius_l10_10074

-- Definitions
def sector_radius : ℝ := 24
def sector_area : ℝ := 120 * Real.pi

-- Theorem
theorem cone_base_radius :
  let r := sector_area / (sector_radius * Real.pi) in r = 5 := by
  sorry

end cone_base_radius_l10_10074


namespace number_of_sides_l10_10659

theorem number_of_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
by {
  sorry
}

end number_of_sides_l10_10659


namespace find_x_find_y_l10_10653

-- Definitions and conditions
def hexagon_area (x : ℝ) : ℝ := 6 * (sqrt 3 / 4 * x^2)
axiom hexagon_area_condition : hexagon_area x = 54 * sqrt 3

def ac_length (y : ℝ) : ℝ := y * sqrt 3
axiom ac_length_is_y_sqrt3 : AC = ac_length y

-- Proof statements
theorem find_x (x : ℝ) : x = 6 :=
by {
  have hex_area: hexagon_area x = 54 * sqrt 3 := hexagon_area_condition,
  sorry
}

theorem find_y (y : ℝ) : y = 6 :=
by {
  have length_ac: ac_length y = 6 * sqrt 3 := ac_length_is_y_sqrt3,
  sorry
}

end find_x_find_y_l10_10653


namespace triangle_inequality_tg_half_angle_triangle_cosine_rule_with_cotangent_triangle_sine_half_angle_inequality_triangle_sum_of_cotangents_l10_10741

-- Problem a)
theorem triangle_inequality_tg_half_angle (A a ha : Real) (S : Real) :
  (Real.tan (A / 2)) ≤ (a / (2 * ha)) := sorry

-- Problem b)
theorem triangle_cosine_rule_with_cotangent (a b c S : Real) (A : ℝ) :
  a^2 = b^2 + c^2 - 4 * S * (Real.cot A) := sorry

-- Problem c)
theorem triangle_sine_half_angle_inequality (A a b c : Real) :
  (Real.sin (A / 2)) ≤ (a / (2 * (Real.sqrt (b * c)))) := sorry

-- Problem d)
theorem triangle_sum_of_cotangents (a b c S : Real) (A B C : Real) :
  (Real.cot A) + (Real.cot B) + (Real.cot C) = (a^2 + b^2 + c^2) / (4 * S) := sorry

end triangle_inequality_tg_half_angle_triangle_cosine_rule_with_cotangent_triangle_sine_half_angle_inequality_triangle_sum_of_cotangents_l10_10741


namespace f_increasing_interval_l10_10384

-- Define the function
def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

-- Define the interval
def interval := Set.Ioo (Real.pi) (3 * Real.pi)

-- Define the monotonic increasing interval
def increasing_interval := Set.Ioo ((3/2) * Real.pi) ((5/2) * Real.pi)

-- Define the derivative of the function
def f_prime (x : ℝ) : ℝ := x * Real.cos x + Real.sin x

theorem f_increasing_interval :
  ∀ x ∈ interval, f_prime x > 0 ↔ x ∈ increasing_interval :=
by
  sorry

end f_increasing_interval_l10_10384


namespace solution_set_l10_10544

theorem solution_set (x : ℝ) :
  Abs.abs (x + 3) - Abs.abs (2 * x - 1) < x / 2 + 1 →
  x < -2 / 5 ∨ x > 2 :=
by 
  sorry

end solution_set_l10_10544


namespace square_prism_surface_area_eq_volume_l10_10589

theorem square_prism_surface_area_eq_volume :
  ∃ (a b : ℕ), (a > 0) ∧ (2 * a^2 + 4 * a * b = a^2 * b)
  ↔ (a = 12 ∧ b = 3) ∨ (a = 8 ∧ b = 4) ∨ (a = 6 ∧ b = 6) ∨ (a = 5 ∧ b = 10) :=
by
  sorry

end square_prism_surface_area_eq_volume_l10_10589


namespace largest_multiple_of_seven_smaller_than_neg_85_l10_10428

theorem largest_multiple_of_seven_smaller_than_neg_85 
  : ∃ k : ℤ, (k * 7 < -85) ∧ (∀ m : ℤ, (m * 7 < -85) → (m * 7 ≤ k * 7)) ∧ (k = -13) 
  := sorry

end largest_multiple_of_seven_smaller_than_neg_85_l10_10428


namespace total_runs_opponents_correct_l10_10051

-- Define the scoring conditions
def team_scores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
def lost_games_scores : List ℕ := [3, 5, 7, 9, 11, 13]
def won_games_scores : List ℕ := [2, 4, 6, 8, 10, 12]

-- Define the total runs scored by opponents in lost games
def total_runs_lost_games : ℕ := (lost_games_scores.map (λ x => x + 1)).sum

-- Define the total runs scored by opponents in won games
def total_runs_won_games : ℕ := (won_games_scores.map (λ x => x / 2)).sum

-- Total runs scored by opponents (given)
def total_runs_opponents : ℕ := total_runs_lost_games + total_runs_won_games

-- The theorem to prove
theorem total_runs_opponents_correct : total_runs_opponents = 75 := by
  -- Proof goes here
  sorry

end total_runs_opponents_correct_l10_10051


namespace line_through_vertex_has_two_a_values_l10_10944

-- Definitions for the line and parabola as conditions
def line_eq (a x : ℝ) : ℝ := 2 * x + a
def parabola_eq (a x : ℝ) : ℝ := x^2 + 2 * a^2

-- The proof problem
theorem line_through_vertex_has_two_a_values :
  (∃ a1 a2 : ℝ, (a1 ≠ a2) ∧ (line_eq a1 0 = parabola_eq a1 0) ∧ (line_eq a2 0 = parabola_eq a2 0)) ∧
  (∀ a : ℝ, line_eq a 0 = parabola_eq a 0 → (a = 0 ∨ a = 1/2)) :=
sorry

end line_through_vertex_has_two_a_values_l10_10944


namespace max_min_f_in_interval_l10_10777

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 9 * x^2 - 2

theorem max_min_f_in_interval : 
  ∃ (max min : ℝ), max = 50 ∧ min = -2 ∧ 
  ∀ x ∈ set.Icc (-4 : ℝ) 2, (f x < max ∨ f x = max) ∧ (f x > min ∨ f x = min) :=
by
  sorry

end max_min_f_in_interval_l10_10777


namespace distance_between_x_intercepts_is_correct_l10_10066

-- Define the slopes of the two lines and the intersection point
def slope1 : ℝ := 4
def slope2 : ℝ := -3
def intersection_point : ℝ × ℝ := (8, 20)

-- Define the equations of the lines using point-slope form
def line1 (x : ℝ) : ℝ := slope1 * (x - (intersection_point.1)) + intersection_point.2
def line2 (x : ℝ) : ℝ := slope2 * (x - (intersection_point.1)) + intersection_point.2

-- Calculate the x-intercepts
def x_intercept1 : ℝ := (12 : ℝ) / slope1
def x_intercept2 : ℝ := (44 : ℝ) / -slope2

-- Define the distance between the x-intercepts
def distance_between_x_intercepts : ℝ := abs (x_intercept1 - x_intercept2)

-- Prove that the distance is 35 / 3
theorem distance_between_x_intercepts_is_correct :
  distance_between_x_intercepts = (35 : ℝ) / 3 :=
by
  sorry

end distance_between_x_intercepts_is_correct_l10_10066


namespace find_x_satisfying_inequality_l10_10918

open Real

theorem find_x_satisfying_inequality :
  ∀ x : ℝ, 0 < x → (x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16 ↔ x = 4) :=
by
  sorry

end find_x_satisfying_inequality_l10_10918


namespace inverse_of_f_l10_10415

def f (x : ℝ) : ℝ := 7 - 3 * x + 1

def g (x : ℝ) : ℝ := (8 - x) / 3

theorem inverse_of_f : (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x) :=
by
  sorry

end inverse_of_f_l10_10415


namespace largest_8_12_double_l10_10879

noncomputable def is_8_12_double (N : ℕ) : Prop :=
  let digits_in_base_8 := N.digits 8
  let base_12_value := digits_in_base_8.foldr (λ d acc, d + 12 * acc) 0
  base_12_value = 2 * N

theorem largest_8_12_double :
  ∃ N : ℕ, is_8_12_double N ∧ (∀ M : ℕ, is_8_12_double M → M ≤ 4032) ∧ N = 4032 :=
by
  -- The proof would go here.
  sorry

end largest_8_12_double_l10_10879


namespace find_m_and_other_root_l10_10623

namespace quadratic

variable {m : ℝ}

def quadratic_eq (x : ℝ) :=
  x^2 + 2 * x + m - 2

theorem find_m_and_other_root (h : quadratic_eq (-3) = 0) :
  m = -1 ∧ (∃ x₂, quadratic_eq x₂ = 0 ∧ x₂ ≠ -3 ∧ x₂ = 1) :=
by
  sorry

end quadratic

end find_m_and_other_root_l10_10623


namespace round_balloons_burst_l10_10288

theorem round_balloons_burst :
  let round_balloons := 5 * 20
  let long_balloons := 4 * 30
  let total_balloons := round_balloons + long_balloons
  let balloons_left := 215
  ((total_balloons - balloons_left) = 5) :=
by 
  sorry

end round_balloons_burst_l10_10288


namespace joao_chocolates_l10_10693

theorem joao_chocolates (n : ℕ) (hn1 : 30 < n) (hn2 : n < 100) (h1 : n % 7 = 1) (h2 : n % 10 = 2) : n = 92 :=
sorry

end joao_chocolates_l10_10693


namespace jerky_batch_size_l10_10412

theorem jerky_batch_size
  (total_order_bags : ℕ)
  (initial_bags : ℕ)
  (days_to_fulfill : ℕ)
  (remaining_bags : ℕ := total_order_bags - initial_bags)
  (production_per_day : ℕ := remaining_bags / days_to_fulfill) :
  total_order_bags = 60 →
  initial_bags = 20 →
  days_to_fulfill = 4 →
  production_per_day = 10 :=
by
  intros
  sorry

end jerky_batch_size_l10_10412


namespace exists_k_not_divisible_l10_10349

theorem exists_k_not_divisible (a b c n : ℤ) (hn : n ≥ 3) :
  ∃ k : ℤ, ¬(n ∣ (k + a)) ∧ ¬(n ∣ (k + b)) ∧ ¬(n ∣ (k + c)) :=
sorry

end exists_k_not_divisible_l10_10349


namespace proof_problem_l10_10241

theorem proof_problem (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 + 2 * a * b = 64 :=
sorry

end proof_problem_l10_10241


namespace derivative_at_zero_l10_10821

def f (x : ℝ) : ℝ :=
if x ≠ 0 then 3^(x^2 * sin(2 / x)) - 1 + 2 * x else 0

theorem derivative_at_zero : (deriv f 0) = -2 :=
by
  sorry

end derivative_at_zero_l10_10821


namespace circle_circumference_l10_10405

/-- Two circles with equal radii intersect such that the area of the shaded region equals the sum of 
the areas of the two unshaded regions, with the area of the shaded region being 216π. Prove that the 
circumference of each circle is 36π. -/
theorem circle_circumference 
  (r : ℝ)
  (A_shaded : ℝ)
  (h_condition : A_shaded = 216 * real.pi)
  (h_geometry : 2 * ((π * r ^ 2 - A_shaded / 2)) = A_shaded) :
  2 * real.pi * r = 36 * real.pi :=
sorry

end circle_circumference_l10_10405


namespace find_possible_n_l10_10984

theorem find_possible_n (n : ℕ) (a b : ℕ) :
  (∀ k, (k = 3 ∨ k = 4) → binomial n k ≤ binomial n 3 ∧ binomial n k ≤ binomial n 4) →
  n = 5 ∨ n = 6 ∨ n = 7 :=
sorry

end find_possible_n_l10_10984


namespace sufficient_wire_length_l10_10799

theorem sufficient_wire_length (A B : ℝ) (l : ℝ) (h : A - B = l):
  let trees := true in /* placeholder for the cylindrical trees environment */
  trees → 1.6 * l ≥ l :=
by sorry

end sufficient_wire_length_l10_10799


namespace find_original_price_l10_10797

-- Definitions based on the conditions
def original_price_increased (x : ℝ) : ℝ := 1.25 * x
def loan_payment (total_cost : ℝ) : ℝ := 0.75 * total_cost
def own_funds (total_cost : ℝ) : ℝ := 0.25 * total_cost

-- Condition values
def new_home_cost : ℝ := 500000
def loan_amount := loan_payment new_home_cost
def funds_paid := own_funds new_home_cost

-- Proof statement
theorem find_original_price : 
  ∃ x : ℝ, original_price_increased x = funds_paid ↔ x = 100000 :=
by
  -- Placeholder for actual proof
  sorry

end find_original_price_l10_10797


namespace range_of_a_l10_10242

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : f = λ x, x^2 - a * x + 1) :
  (∃ x, f x < 0) → (a > 2 ∨ a < -2) :=
by sorry

end range_of_a_l10_10242


namespace quadratic_equation_distinct_real_roots_and_values_of_m_l10_10634

noncomputable theory -- only if necessary

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (m - 1) ^ 2 + 8

-- Define the condition given in the problem
def roots_condition (m x1 x2 : ℝ) : Prop :=
  x1 ^ 2 + x2 ^ 2 - x1 * x2 = 7

-- Prove the quadratic equation has distinct real roots and identify possible values of m
theorem quadratic_equation_distinct_real_roots_and_values_of_m (m : ℝ) (x1 x2 : ℝ) :
  (discriminant m > 0) ∧ roots_condition m x1 x2 → m = 1 ∨ m = 2 :=
sorry

end quadratic_equation_distinct_real_roots_and_values_of_m_l10_10634


namespace divisors_count_l10_10708

noncomputable def m : ℕ := 2^42 * 3^26 * 5^12

theorem divisors_count : 
  let m2 := m^2,
      num_divisors_m2 := (84 + 1) * (52 + 1) * (24 + 1),
      num_divisors_m2_less := (num_divisors_m2 - 1) / 2,
      num_divisors_m := (42 + 1) * (26 + 1) * (12 + 1),
      num_divisors_m2_less_not_m := num_divisors_m2_less - num_divisors_m
  in num_divisors_m2_less_not_m = 38818 := by
  sorry

end divisors_count_l10_10708


namespace part1_part2_l10_10217

-- Definition for the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Part 1: Proving the quadratic equation always has two real roots
theorem part1 (k : ℝ) : 
  discriminant 1 (-(k + 4)) (4 * k) ≥ 0 := 
begin
  sorry -- We are stating the problem, proof is not required.
end

-- Part 2: Finding the value of k given the condition on the roots
theorem part2 (k x1 x2 : ℝ) (h1 : x1 + x2 = k + 4) (h2 : x1 * x2 = 4 * k) (h3 : 1/x1 + 1/x2 = 3/4) : 
  k = 2 := 
begin
  sorry -- We are stating the problem, proof is not required.
end

end part1_part2_l10_10217


namespace amount_paid_to_Y_l10_10008

theorem amount_paid_to_Y 
(h1 : ∀ X Y : ℝ, X + Y = 550)
(h2 : ∀ X Y : ℝ, X = 1.2 * Y) :
  ∃ Y : ℝ, Y = 250 :=
begin
 sorry,
end

end amount_paid_to_Y_l10_10008


namespace email_sequence_correct_l10_10126

theorem email_sequence_correct :
    ∀ (a b c d e f : Prop),
    (a → (e → (b → (c → (d → f))))) :=
by 
  sorry

end email_sequence_correct_l10_10126


namespace solution_set_of_inequality_l10_10784

theorem solution_set_of_inequality :
  {x : ℝ | |x^2 - 2| < 2} = {x : ℝ | (x > -2 ∧ x < 0) ∨ (x > 0 ∧ x < 2)} :=
by
  sorry

end solution_set_of_inequality_l10_10784


namespace expenditure_ratio_l10_10477

/-- A man saves 35% of his income in the first year. -/
def saving_rate_first_year : ℝ := 0.35

/-- His income increases by 35% in the second year. -/
def income_increase_rate : ℝ := 0.35

/-- His savings increase by 100% in the second year. -/
def savings_increase_rate : ℝ := 1.0

theorem expenditure_ratio
  (I : ℝ)  -- first year income
  (S1 : ℝ := saving_rate_first_year * I)  -- first year saving
  (E1 : ℝ := I - S1)  -- first year expenditure
  (I2 : ℝ := I + income_increase_rate * I)  -- second year income
  (S2 : ℝ := 2 * S1)  -- second year saving (increases by 100%)
  (E2 : ℝ := I2 - S2)  -- second year expenditure
  :
  (E1 + E2) / E1 = 2
  :=
  sorry

end expenditure_ratio_l10_10477


namespace widget_production_difference_l10_10895

variable (t : ℕ)
variable (w : ℕ)
variable (h : w = 2 * t)

theorem widget_production_difference :
  let monday_production := w * t
  let tuesday_production := (w + 5) * (t - 1)
  monday_production - tuesday_production = -3 * t + 5 := 
by
  sorry

end widget_production_difference_l10_10895


namespace smallest_positive_solution_tan_sec_eq_l10_10155

theorem smallest_positive_solution_tan_sec_eq 
  (x : ℝ) 
  (hx : x > 0)
  (hx_rad : ∃ y : ℝ, x = y * real.pi) 
  (h_eq : real.tan (4 * x) + real.tan (5 * x) = real.sec (5 * x)) :
  x = real.pi / 18 :=
sorry

end smallest_positive_solution_tan_sec_eq_l10_10155


namespace gcd_a_b_eq_one_l10_10414

def a : ℕ := 130^2 + 240^2 + 350^2
def b : ℕ := 131^2 + 241^2 + 349^2

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 := by
  sorry

end gcd_a_b_eq_one_l10_10414


namespace relationship_f_l10_10625

def f (x : ℝ) : ℝ :=
  Real.sqrt (1 - (x - 1)^2)

def F (x : ℝ) : ℝ :=
  f x / x

theorem relationship_f (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 < 1) :
  F x1 > F x2 :=
by
  sorry

end relationship_f_l10_10625


namespace ceil_of_calc_expr_l10_10127

noncomputable def calc_expr : ℝ :=
  4 * (7 - (2 / 3))

theorem ceil_of_calc_expr : Real.ceil calc_expr = 26 := by
  sorry

end ceil_of_calc_expr_l10_10127


namespace ratio_of_areas_l10_10041

-- Definitions of given conditions
variables (TRN HAM : Triangle)
variables (T R N H A M : Point)
variables (T_is_centroid : IsCentroid T HAM)
variables (TRN_is_equilateral : IsEquilateral TRN)
variables (HAM_is_equilateral : IsEquilateral HAM)
variables (R_on_ray_TA : OnRay R T A)
variables (cong_triangles : Congruent TRN HAM)

-- The main theorem statement: ratio of areas inside and outside
theorem ratio_of_areas (h1 : TRN_is_equilateral) (h2 : HAM_is_equilateral) 
(h3 : T_is_centroid) (h4 : R_on_ray_TA) (h5 : cong_triangles) :
    ratio_of_areas_inside_outside TRN HAM 1 5 :=
sorry

end ratio_of_areas_l10_10041


namespace solution_is_correct_l10_10749

noncomputable def satisfies_inequality (x y : ℝ) : Prop := 
  x + 3 * y + 14 ≤ 0

noncomputable def satisfies_equation (x y : ℝ) : Prop := 
  x^4 + 2 * x^2 * y^2 + y^4 + 64 - 20 * x^2 - 20 * y^2 = 8 * x * y

theorem solution_is_correct : satisfies_inequality (-2) (-4) ∧ satisfies_equation (-2) (-4) :=
  by sorry

end solution_is_correct_l10_10749


namespace positive_difference_of_perimeters_is_zero_l10_10406

theorem positive_difference_of_perimeters_is_zero :
  ∀ (F T : set (ℕ × ℕ)),
  (∃ (f1 f2 f3 f4 f5 : ℕ × ℕ),
      F = {f1, f2, f3, f4, f5} ∧
      f1 = (0, 0) ∧ f2 = (0, 1) ∧ f3 = (0, 2) ∧ f4 = (1, 1) ∧ f5 = (-1, 1)) ∧
  (∃ (t1 t2 t3 t4 t5 : ℕ × ℕ),
      T = {t1, t2, t3, t4, t5} ∧
      t1 = (0, 0) ∧ t2 = (0, 1) ∧ t3 = (0, 2) ∧ t4 = (1, 1) ∧ t5 = (2, 1)) →
  (let perimeter (shape : set (ℕ × ℕ)) : ℕ :=
    4 * shape.card - 2 * (shape.card - 1) in
   abs (perimeter F - perimeter T) = 0) :=
begin
  sorry
end

end positive_difference_of_perimeters_is_zero_l10_10406


namespace exists_negative_column_product_l10_10049

theorem exists_negative_column_product (a : Fin 5 → Fin 5 → ℝ)
  (row_neg_product : ∀ i : Fin 5, (∏ j : Fin 5, a i j) < 0) :
  ∃ j : Fin 5, (∏ i : Fin 5, a i j) < 0 :=
sorry

end exists_negative_column_product_l10_10049


namespace circle_radius_is_three_l10_10271

noncomputable def circle_radius (XP XQ XPQ: ℝ) : ℝ :=
  XP / 2

theorem circle_radius_is_three
  (XP : ℝ)
  (XQ : ℝ)
  (PQ : ℝ)
  (tangent_XP : XP = 6)
  (tangent_XQ : XQ = 12)
  (PQ_eq : PQ = XP + XQ) :
  circle_radius XP XQ PQ = 3 :=
begin
  sorry
end

end circle_radius_is_three_l10_10271


namespace cone_sphere_ratio_l10_10485

theorem cone_sphere_ratio (r h : ℝ) (π_pos : 0 < π) (r_pos : 0 < r) :
  (1/3) * π * r^2 * h = (1/3) * (4/3) * π * r^3 → h / r = 4/3 :=
by
  sorry

end cone_sphere_ratio_l10_10485


namespace arith_seq_s14_gt_0_l10_10188

variable {S : ℕ → ℝ} -- S_n is the sum of the first n terms of an arithmetic sequence
variable {a : ℕ → ℝ} -- a_n is the nth term of the arithmetic sequence
variable {d : ℝ} -- d is the common difference of the arithmetic sequence

-- Conditions
variable (a_7_lt_0 : a 7 < 0)
variable (a_5_plus_a_10_gt_0 : a 5 + a 10 > 0)

-- Assertion
theorem arith_seq_s14_gt_0 (a_7_lt_0 : a 7 < 0) (a_5_plus_a_10_gt_0 : a 5 + a 10 > 0) : S 14 > 0 := by
  sorry

end arith_seq_s14_gt_0_l10_10188


namespace cos_at_min_distance_l10_10198

noncomputable def cosAtMinimumDistance (t : ℝ) (ht : t < 0) : ℝ :=
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  if distance = Real.sqrt 5 then
    x / distance
  else
    0 -- some default value given the condition distance is not sqrt(5), which is impossible in this context

theorem cos_at_min_distance (t : ℝ) (ht : t < 0) :
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  distance = Real.sqrt 5 → cosAtMinimumDistance t ht = - 2 * Real.sqrt 5 / 5 :=
by
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  sorry

end cos_at_min_distance_l10_10198


namespace intersection_point_l10_10296

noncomputable def f (x : ℝ) := (x^2 - 8 * x + 7) / (2 * x - 6)

noncomputable def g (a b c : ℝ) (x : ℝ) := (a * x^2 + b * x + c) / (x - 3)

theorem intersection_point (a b c : ℝ) :
  (∀ x, 2 * x - 6 = 0 <-> x ≠ 3) →
  ∃ (k : ℝ), (g a b c x = -2 * x - 4 + k / (x - 3)) →
  (f x = g a b c x) ∧ x ≠ -3 → x = 1 ∧ f 1 = 0 :=
by
  intros
  sorry

end intersection_point_l10_10296


namespace christian_sue_need_more_money_l10_10882

-- Definitions based on the given conditions
def bottle_cost : ℕ := 50
def christian_initial : ℕ := 5
def sue_initial : ℕ := 7
def christian_mowing_rate : ℕ := 5
def christian_mowing_count : ℕ := 4
def sue_walking_rate : ℕ := 2
def sue_walking_count : ℕ := 6

-- Prove that Christian and Sue will need 6 more dollars to buy the bottle of perfume
theorem christian_sue_need_more_money :
  let christian_earning := christian_mowing_rate * christian_mowing_count
  let christian_total := christian_initial + christian_earning
  let sue_earning := sue_walking_rate * sue_walking_count
  let sue_total := sue_initial + sue_earning
  let total_money := christian_total + sue_total
  50 - total_money = 6 :=
by
  sorry

end christian_sue_need_more_money_l10_10882


namespace largest_multiple_of_7_smaller_than_negative_85_l10_10418

theorem largest_multiple_of_7_smaller_than_negative_85 :
  ∃ (n : ℤ), (∃ (k : ℤ), n = 7 * k) ∧ n < -85 ∧ ∀ (m : ℤ), (∃ (k : ℤ), m = 7 * k) ∧ m < -85 → m ≤ n := 
by
  use -91
  split
  { use -13
    norm_num }
  split
  { exact dec_trivial }
  { intros m hm
    cases hm with k hk
    cases hk with hk1 hk2
    have hk3 : k < -12 := by linarith
    have hk4 : k ≤ -13 := int.floor_le $ hk3
    linarith }


end largest_multiple_of_7_smaller_than_negative_85_l10_10418


namespace ratio_of_inscribed_squares_l10_10495

theorem ratio_of_inscribed_squares (a b : ℝ) 
    (h1 : ∀ (a b : ℝ), is_inscribed_square_side_right_vertex (triangle right_triangle (5 : ℝ) (12 : ℝ) (13 : ℝ)) a) 
    (h2 : ∀ (a b : ℝ), is_inscribed_square_side_hypotenuse (triangle right_triangle (5 : ℝ) (12 : ℝ) (13 : ℝ)) b)  :
  a / b = 39 / 51 := 
sorry

end ratio_of_inscribed_squares_l10_10495


namespace radioactive_decay_minimum_years_l10_10072

noncomputable def min_years (a : ℝ) (n : ℕ) : Prop :=
  (a * (1 - 3 / 4) ^ n ≤ a * 1 / 100)

theorem radioactive_decay_minimum_years (a : ℝ) (h : 0 < a) : ∃ n : ℕ, min_years a n ∧ n = 4 :=
by {
  sorry
}

end radioactive_decay_minimum_years_l10_10072


namespace true_propositions_count_l10_10208

theorem true_propositions_count : 
  let proposition1 := ¬ (∃ x : ℝ, x^2 + 1 > 3 * x) = ∀ x : ℝ, x^2 + 1 < 3 * x 
  let proposition2 := ∀ (p q : Prop), ¬(p ∨ q) ↔ (¬p ∧ ¬q)
  let proposition3 := ∀ a : ℝ, (a > 2 → a > 5) ∧ ¬(a > 5 → a > 2)
  let proposition4 := (∀ (x y : ℝ), (x * y = 0 → x = 0 ∧ y = 0) ↔ (x = 0 ∧ y = 0 → x * y = 0))
  
  (proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4) = 1 := 
sorry

end true_propositions_count_l10_10208


namespace sin_cos_from_tan_l10_10610

variable {α : Real} (hα : α > 0) -- Assume α is an acute angle

theorem sin_cos_from_tan (h : Real.tan α = 2) : 
  Real.sin α = 2 / Real.sqrt 5 ∧ Real.cos α = 1 / Real.sqrt 5 := 
by sorry

end sin_cos_from_tan_l10_10610


namespace other_train_length_l10_10801

noncomputable def relative_speed (speed1 speed2 : ℝ) : ℝ :=
  speed1 + speed2

noncomputable def speed_in_km_per_sec (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr / 3600

noncomputable def total_distance_crossed (relative_speed : ℝ) (time_sec : ℕ) : ℝ :=
  relative_speed * (time_sec : ℝ)

noncomputable def length_of_other_train (total_distance length_of_first_train : ℝ) : ℝ :=
  total_distance - length_of_first_train

theorem other_train_length :
  let speed1 := 210
  let speed2 := 90
  let length_of_first_train := 0.9
  let time_taken := 24
  let relative_speed_km_per_hr := relative_speed speed1 speed2
  let relative_speed_km_per_sec := speed_in_km_per_sec relative_speed_km_per_hr
  let total_distance := total_distance_crossed relative_speed_km_per_sec time_taken
  length_of_other_train total_distance length_of_first_train = 1.1 := 
by
  sorry

end other_train_length_l10_10801


namespace projection_of_a_on_b_l10_10962

-- Define vectors a and b
variables (a b : ℝ × ℝ)

-- Given conditions
def non_zero_vectors : Prop := a ≠ (0, 0) ∧ b ≠ (0, 0)
def vector_b : b = (Math.sqrt 3, 1)
def angle_ab : real.angle (a, b) = real.pi / 3
def orthogonal_condition : innerProductSpace ℝ ℝ (a - b) a = 0

-- Proof statement
theorem projection_of_a_on_b (h1 : non_zero_vectors a b) (h2 : vector_b b) (h3 : angle_ab a b) (h4 : orthogonal_condition a b) :
  projectionVector a b = (1/4) • b :=
sorry

end projection_of_a_on_b_l10_10962


namespace angle_C_range_f_l10_10664

theorem angle_C (a b c : ℝ) (A B C : ℝ) (h : (2 * a - b) / c = (Real.cos B) / (Real.cos C)) : C = π / 3 :=
sorry

theorem range_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) : 
let f : ℝ → ℝ := λ x, 2 * Real.sin x * Real.cos x * Real.cos (π / 3) + 2 * Real.sin x ^ 2 * Real.sin (π / 3) - sqrt 3 / 2 in
- sqrt 3 / 2 ≤ f x ∧ f x ≤ 1 :=
sorry

end angle_C_range_f_l10_10664


namespace infinite_real_solution_count_l10_10713

theorem infinite_real_solution_count 
: ∃ (n : ℕ), n = ∞ ∧ ∀ p : ℝ, (let A := 1; let B := -2 * p; let C := p^2 in (B^2 - 4 * A * C) = 0) := 
by 
  sorry

end infinite_real_solution_count_l10_10713


namespace gcd_14m_21n_126_l10_10650

theorem gcd_14m_21n_126 {m n : ℕ} (hm_pos : 0 < m) (hn_pos : 0 < n) (h_gcd : Nat.gcd m n = 18) : 
  Nat.gcd (14 * m) (21 * n) = 126 :=
by
  sorry

end gcd_14m_21n_126_l10_10650


namespace line_slope_l10_10319

-- Definitions for the given conditions
def point (x y : ℝ) := (x, y)
def line := set (ℝ × ℝ)

-- Definition of our specific points
def p1 := point 0 (-2)
def p2 := point 5 3

-- Definition of slope as a function between two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Statement of the problem: prove that the slope between p1 and p2 is equal to 1
theorem line_slope : slope p1 p2 = 1 := by
  sorry

end line_slope_l10_10319


namespace domain_f_l10_10372

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.log_base (3/4) (2 * x - 1))

theorem domain_f : (set.Ioo (1/2 : ℝ) 1) = {x | f x = Real.sqrt (Real.log_base (3/4) (2 * x - 1)) ∧ 0 < 2 * x - 1 ∧ 2 * x - 1 ≤ 1 } :=
by
  sorry

end domain_f_l10_10372


namespace compare_final_values_l10_10344

noncomputable def final_value_Almond (initial: ℝ): ℝ := (initial * 1.15) * 0.85
noncomputable def final_value_Bean (initial: ℝ): ℝ := (initial * 0.80) * 1.20
noncomputable def final_value_Carrot (initial: ℝ): ℝ := (initial * 1.10) * 0.90

theorem compare_final_values (initial: ℝ) (h_positive: 0 < initial):
  final_value_Almond initial < final_value_Bean initial ∧ 
  final_value_Bean initial < final_value_Carrot initial := by
  sorry

end compare_final_values_l10_10344


namespace second_card_is_three_l10_10948

theorem second_card_is_three (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
                             (h_sum : a + b + c + d = 30)
                             (h_increasing : a < b ∧ b < c ∧ c < d)
                             (h_dennis : ∀ x y z, x = a → (y ≠ b ∨ z ≠ c ∨ d ≠ 30 - a - y - z))
                             (h_mandy : ∀ x y z, x = b → (y ≠ a ∨ z ≠ c ∨ d ≠ 30 - x - y - z))
                             (h_sandy : ∀ x y z, x = c → (y ≠ a ∨ z ≠ b ∨ d ≠ 30 - x - y - z))
                             (h_randy : ∀ x y z, x = d → (y ≠ a ∨ z ≠ b ∨ c ≠ 30 - x - y - z)) :
  b = 3 := 
sorry

end second_card_is_three_l10_10948


namespace maximum_value_of_f_l10_10043

noncomputable def f (a x : ℝ) : ℝ := (1 + x) ^ a - a * x

theorem maximum_value_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  ∃ x : ℝ, x > -1 ∧ ∀ y : ℝ, y > -1 → f a y ≤ f a x ∧ f a x = 1 :=
by {
  sorry
}

end maximum_value_of_f_l10_10043


namespace find_y_l10_10361

theorem find_y (y : ℝ) (h : sqrt (3 + sqrt (4 * y - 5)) = sqrt 8) : y = 7.5 :=
sorry

end find_y_l10_10361


namespace min_distance_from_start_to_finish_l10_10273

-- Definition of points in the grid
structure Point :=
  (x y : ℝ)

-- Starting and finishing points in the grid
def Start : Point := ⟨0, 0⟩
def Finish : Point := ⟨2, 2⟩

-- Definition of the distance function
def distance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 |>.sqrt

-- The main theorem stating that the minimum distance is 2 + 2√2
theorem min_distance_from_start_to_finish : 
  distance Start Finish = 2 + 2 * Real.sqrt 2 :=
sorry

end min_distance_from_start_to_finish_l10_10273


namespace find_x_plus_one_over_x_l10_10968

variable (x : ℝ)

theorem find_x_plus_one_over_x
  (h1 : x^3 + (1/x)^3 = 110)
  (h2 : (x + 1/x)^2 - 2*x - 2*(1/x) = 38) :
  x + 1/x = 5 :=
sorry

end find_x_plus_one_over_x_l10_10968


namespace ball_distribution_unique_boxes_l10_10734

-- Define the problem parameters and conditions
def numBalls : Nat := 8
def numBoxes : Nat := 3
def minBallsPerBox (boxes : Fin numBoxes → Nat) : Prop :=
  (∀ i, boxes i > 0) ∧ ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ boxes i ≠ boxes j ∧ boxes j ≠ boxes k ∧ boxes k ≠ boxes i

-- Define the main theorem to prove the equivalent problem
theorem ball_distribution_unique_boxes :
  ∃ f : (Fin numBalls) → (Fin numBoxes), 
    (∀ i j, i ≠ j → f i ≠ f j) ∧ minBallsPerBox (λ i, (Finset.filter (λ x, f x = i.val) (Finset.univ)).card) ∧ 
    ∑ i in Finset.univ, (Finset.filter (λ x, f x = i.val) (Finset.univ)).card = numBalls ∧
    (Finset.card (Finset.image (λ b : Finset (Fin numBalls), b.card) (Finset.univ.image 
      (λ f : (Fin numBalls) → (Fin numBoxes), 
        (Finset.image f (Finset.univ)))))).card = 2688 := sorry

end ball_distribution_unique_boxes_l10_10734


namespace num_ways_to_form_rectangle_l10_10570

theorem num_ways_to_form_rectangle (n : ℕ) (h : n = 5) :
  (nat.choose n 2) * (nat.choose n 2) = 100 :=
by {
  rw h,
  exact nat.choose_five_two_mul_five_two 100
}

lemma nat.choose_five_two_mul_five_two :
  ((5.choose 2) * (5.choose 2) = 100) :=
by norm_num

end num_ways_to_form_rectangle_l10_10570


namespace sheet_sums_l10_10313

def is_interesting_sequence (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (a i = i ∨ a i = i + 1)

def is_even_sum (n : ℕ) (a : ℕ → ℕ) : Prop :=
  (∑ i in Finset.range n, a (i + 1)) % 2 = 0

def A_n (n : ℕ) : ℕ := 
  ∏ i in Finset.range n, if (i + 1) % 2 = 0 then (i + 2) else (i + 1)

def B_n (n : ℕ) : ℕ := 
  ∏ i in Finset.range n, if (i + 1) % 2 = 0 then (i + 1) else (i + 2)

theorem sheet_sums (n : ℕ) : 
  (n % 4 = 0 ∨ n % 4 = 1) → (A_n n - B_n n = 1) ∧ 
  (¬ (n % 4 = 0 ∨ n % 4 = 1)) → (B_n n - A_n n = 1) :=
by
  sorry

end sheet_sums_l10_10313


namespace part_I_general_formula_part_II_sum_of_b_n_l10_10955

-- Define the conditions
def sequence_a (n : ℕ) : ℤ := 2 * n + 1
def S (n : ℕ) : ℤ := n^2 + 2 * n
def P (n : ℕ) : ℤ × ℤ := (n, S n)
def k (n : ℕ) : ℤ := 2 * n + 2

-- Define b_n and T_n based on the problem
def b (n : ℕ) : ℚ := 1 / ((sequence_a n) * (k n + 1))
def T (n : ℕ) : ℚ := (1 / 2) * ((1 / 3) - (1 / (2 * n + 3)))

-- Prove the general formula for the sequence a_n
theorem part_I_general_formula (n : ℕ) : sequence_a n = 2 * n + 1 :=
  sorry

-- Prove the sum of the first n terms of b_n, T_n
theorem part_II_sum_of_b_n (n : ℕ) : (Finset.sum (Finset.range n) (λ i, b (i + 1))) = n / (6 * n + 9) :=
  sorry

end part_I_general_formula_part_II_sum_of_b_n_l10_10955


namespace log_sqrt6_of_216_sqrt6_l10_10537

theorem log_sqrt6_of_216_sqrt6 : log (real.sqrt 6) (216 * real.sqrt 6) = 7 :=
by
  sorry

end log_sqrt6_of_216_sqrt6_l10_10537


namespace eigenvector_exists_real_iff_l10_10573

noncomputable def characteristic_poly (x m λ : ℝ) : ℝ :=
    λ^2 - (x + 3) * λ + (m^2 - m - 6)

theorem eigenvector_exists_real_iff (x : ℝ) (m : ℝ) : 
    ∃ v : ℝ × ℝ, v ≠ (0, 0) ∧ (v.1, v.2) is an eigenvector of the matrix !![x, 2 + m; 3 - m, 3]
    ↔ -2 ≤ m ∧ m ≤ 3 :=
sorry

end eigenvector_exists_real_iff_l10_10573


namespace min_value_of_t_sum_of_sequence_b_l10_10978

variable (n : ℕ)

-- Definitions for the geometric sequence
variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions given in the problem
def geometric_sequence : Prop :=
  ∀ n, a n > 0 ∧ a 2 = 2/3 ∧ (a 3 * a 4 = 2 * a 6)

-- Sum of the first n terms of the geometric sequence
def sum_of_sequence (n : ℕ) : ℝ := S n

-- Definition b_n
def b (n : ℕ) : ℝ := n / a n

-- Sum of the first n terms of {b_n}
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b i

theorem min_value_of_t (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geo: geometric_sequence a)
  (h_sum: ∀ n, S n = ∑ i in finset.range n, a i) : 
  ∀ n, S n < 3 := by sorry

theorem sum_of_sequence_b (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geo: geometric_sequence a)
  (h_sum: ∀ n, S n = ∑ i in finset.range n, a i) :
  T n = (2 * n - 1) * 3 ^ n + 1 / 8 := by sorry

end min_value_of_t_sum_of_sequence_b_l10_10978


namespace prob_two_five_l10_10593

-- Conditions given in the problem
variable (X : ℝ) (μ σ : ℝ)

axiom normal_dist : X ~ Normal(μ, σ^2)
axiom prob_gt_5 : ∃ p : ℝ, p = 0.2 ∧ ∀ X, P(X > 5) = p
axiom prob_lt_neg1 : ∃ p : ℝ, p = 0.2 ∧ ∀ X, P(X < -1) = p

-- The proof goal
theorem prob_two_five :
  P(2 < X < 5) = 0.6 :=
sorry

end prob_two_five_l10_10593


namespace concyclic_if_and_only_if_angle_bac_90_l10_10299

theorem concyclic_if_and_only_if_angle_bac_90
  {A B C D K L : Point}
  (ABC_scalene : ¬(A = B) ∧ ¬(B = C) ∧ ¬(A = C))
  (BC_largest : ∀ X Y Z : Point, distance X Y ≤ distance B C)
  (D_perpendicular : D ∈ line B C ∧ angle A D B = π / 2)
  (K_on_AB : K ∈ line A B)
  (L_on_AC : L ∈ line A C)
  (D_midpoint_KL : distance K D = distance D L) :
  (concyclic B K C L) ↔ (angle A B C = π / 2) :=
by sorry

end concyclic_if_and_only_if_angle_bac_90_l10_10299


namespace cos_210_eq_neg_sqrt3_div_2_l10_10519

theorem cos_210_eq_neg_sqrt3_div_2 :
  let θ := 30
  let cos30 := (Real.sqrt 3) / 2
  cos (180 + θ) = -cos30 → 
  cos 210 = - (Real.sqrt 3) / 2 :=
by
  intro θ cos30 h
  rw [←h]
  sorry

end cos_210_eq_neg_sqrt3_div_2_l10_10519


namespace domain_of_f_f_at_neg2_f_at_6_l10_10210

def f (x : ℝ) : ℝ := (8 / (x - 2)) + Real.sqrt (x + 3)

def domain (x : ℝ) : Prop := (x ≥ -3 ∧ x < 2) ∨ (x > 2)

theorem domain_of_f : ∀ x, domain x ↔ (x ≠ 2 ∧ x ≥ -3) :=
by sorry

theorem f_at_neg2 : f (-2) = -1 :=
by sorry

theorem f_at_6 : f 6 = 5 :=
by sorry

end domain_of_f_f_at_neg2_f_at_6_l10_10210


namespace volume_of_tetrahedron_height_of_tetrahedron_l10_10877

-- Define the vertex coordinates
def A1 := (1, 0, 2)
def A2 := (1, 2, -1)
def A3 := (2, -2, 1)
def A4 := (2, 1, 0)

-- Function to compute the volume of the tetrahedron
noncomputable def tetrahedron_volume : ℚ :=
  1/6 * |det3
    ⟨A2.1 - A1.1, A2.2 - A1.2, A2.3 - A1.3⟩
    ⟨A3.1 - A1.1, A3.2 - A1.2, A3.3 - A1.3⟩
    ⟨A4.1 - A1.1, A4.2 - A1.2, A4.3 - A1.3⟩|

-- Function to compute the height of the tetrahedron
noncomputable def tetrahedron_height : ℚ :=
  let base_area := 1/2 * |cross_product
    ⟨A2.1 - A1.1, A2.2 - A1.2, A2.3 - A1.3⟩
    ⟨A3.1 - A1.1, A3.2 - A1.2, A3.3 - A1.3⟩|
  in (3 * tetrahedron_volume) / base_area

-- Theorem statements
theorem volume_of_tetrahedron : tetrahedron_volume = 7/6 := by
  sorry -- proof not required

theorem height_of_tetrahedron : tetrahedron_height = real.sqrt (7 / 11) := by
  sorry -- proof not required

end volume_of_tetrahedron_height_of_tetrahedron_l10_10877


namespace initial_dogs_count_is_36_l10_10793

-- Conditions
def initial_cats := 29
def adopted_dogs := 20
def additional_cats := 12
def total_pets := 57

-- Calculate total cats
def total_cats := initial_cats + additional_cats

-- Calculate initial dogs
def initial_dogs (initial_dogs : ℕ) : Prop :=
(initial_dogs - adopted_dogs) + total_cats = total_pets

-- Prove that initial dogs (D) is 36
theorem initial_dogs_count_is_36 : initial_dogs 36 :=
by
-- Here should contain the proof which is omitted
sorry

end initial_dogs_count_is_36_l10_10793


namespace number_of_satisfied_sets_l10_10637

-- Defining the conditions as hypotheses
def A (a : Fin 1000 -> Nat) : Prop :=
  ∀ i j : Fin 1000, i + j < 1000 -> i + j ∈ A -> a i + a j ∈ A

theorem number_of_satisfied_sets : 
  ∃ A, (∀ (i j : Nat), (1 ≤ i ∧ i ≤ 1000) → (1 ≤ j ∧ j ≤ 1000) → (i + j ∈ A → a i + a j ∈ A)) ∧
       (∀ (n : Fin 1000), a n < a (n + 1) ∧ a (999) ≤ 2017) ∧
       (∑ k in range(0, 17).sum (λ _, binom 17 _)) = 2^17
:=
  sorry

end number_of_satisfied_sets_l10_10637


namespace part1_part2_l10_10987

def f (a : ℝ) (x : ℝ) : ℝ := a * |x - 2| + x
def g (x : ℝ) : ℝ := |x - 2| - |2 * x - 3| + x

theorem part1 (a : ℝ) : (∀ x, f a x ≤ f a 2) ↔ a ≤ -1 :=
by sorry

theorem part2 (x : ℝ) : f 1 x < |2 * x - 3| ↔ x > 0.5 :=
by sorry

end part1_part2_l10_10987


namespace new_ratio_of_boarders_to_day_scholars_l10_10388

theorem new_ratio_of_boarders_to_day_scholars
  (B_initial D_initial : ℕ)
  (B_initial_eq : B_initial = 560)
  (ratio_initial : B_initial / D_initial = 7 / 16)
  (new_boarders : ℕ)
  (new_boarders_eq : new_boarders = 80)
  (B_new : ℕ)
  (B_new_eq : B_new = B_initial + new_boarders)
  (D_new : ℕ)
  (D_new_eq : D_new = D_initial) :
  B_new / D_new = 1 / 2 :=
by
  sorry

end new_ratio_of_boarders_to_day_scholars_l10_10388


namespace marie_finishes_third_task_at_3_30_PM_l10_10724

open Time

def doesThreeEqualTasksInARow (start first second third : TimeClock) : Prop :=
  ∃ d : TimeClock, -- duration of one task
  first = start + d ∧
  second = first + d ∧
  third = second + d

theorem marie_finishes_third_task_at_3_30_PM :
  ∀ start first second third : TimeClock,
  start.hours = 13 ∧ start.minutes = 0 ∧ -- 1:00 PM
  second.hours = 14 ∧ second.minutes = 40 ∧ -- 2:40 PM
  doesThreeEqualTasksInARow start first second third →
  third.hours = 15 ∧ third.minutes = 30 := -- 3:30 PM
by
  intros start first second third start_cond second_cond equal_tasks_cond
  sorry

end marie_finishes_third_task_at_3_30_PM_l10_10724


namespace sum_of_first_2018_b_l10_10964

-- Define sequences a_n and b_n where a_n * b_n = 1 and a_n = n^2 + 3n + 2
def a (n : ℕ) : ℝ := n^2 + 3 * n + 2
def b (n : ℕ) : ℝ := 1 / a n

-- Define the sum of the first 2018 terms of the sequence b_n
def sum_b_first_2018 : ℝ := (Finset.range 2018).sum (λ n, b (n + 1))

-- Prove the sum of first 2018 terms of b_n is 1009/2020
theorem sum_of_first_2018_b (h₁ : ∀ n : ℕ, a n * b n = 1) : sum_b_first_2018 = 1009 / 2020 :=
  sorry

end sum_of_first_2018_b_l10_10964


namespace sin_sum_diff_to_product_l10_10132

theorem sin_sum_diff_to_product (a b : ℝ) : 
  sin (2 * a + b) - sin (2 * a - b) = 2 * cos (2 * a) * sin b := 
sorry

end sin_sum_diff_to_product_l10_10132


namespace sum_of_two_lowest_scores_l10_10747

theorem sum_of_two_lowest_scores (scores : List ℝ) (h_len : scores.length = 6) 
  (h_mean : (scores.sum) / 6 = 85) 
  (h_median : scores.sorted (≤) nth 2 = 86 ∧ scores.sorted (≤) nth 3 = 86) 
  (h_mode : scores.count 88 > scores.count any other score) : 
  scores.sorted (≤).head + scores.sorted (≤).head!.tail.head = 162 := 
by 
  sorry

end sum_of_two_lowest_scores_l10_10747


namespace game_C_higher_prob_l10_10052

-- Defining the probabilities
def prob_heads : ℝ := 3 / 5
def prob_tails : ℝ := 2 / 5

-- Probability of winning Game C
def prob_winning_game_C : ℝ :=
  prob_heads^3 + prob_tails^3

-- Probability of winning Game D
def prob_winning_game_D : ℝ :=
  (prob_heads^2 * prob_tails) + (prob_tails^2 * prob_heads)

-- Main theorem
theorem game_C_higher_prob :
  prob_winning_game_C - prob_winning_game_D = 1 / 25 := by
  sorry

end game_C_higher_prob_l10_10052


namespace last_years_rate_per_mile_l10_10695

-- Definitions from the conditions
variables (m : ℕ) (x : ℕ)

-- Condition 1: This year, walkers earn $2.75 per mile
def amount_per_mile_this_year : ℝ := 2.75

-- Condition 2: Last year's winner collected $44
def last_years_total_amount : ℕ := 44

-- Condition 3: Elroy will walk 5 more miles than last year's winner
def elroy_walks_more_miles (m : ℕ) : ℕ := m + 5

-- The main goal is to prove that last year's rate per mile was $4 given the conditions
theorem last_years_rate_per_mile (h1 : last_years_total_amount = m * x)
  (h2 : last_years_total_amount = (elroy_walks_more_miles m) * amount_per_mile_this_year) :
  x = 4 :=
by {
  sorry
}

end last_years_rate_per_mile_l10_10695


namespace range_and_minimum_value_l10_10630

-- Define the function f(x) and the condition for its domain
def f (x : ℝ) (m : ℝ) : ℝ := real.sqrt (abs (x + 1) + abs (x - 3) - m)
def domain_condition (m : ℝ) : Prop := ∀ (x : ℝ), abs (x + 1) + abs (x - 3) - m ≥ 0

-- Define the equation involving a and b and the condition for their positivity
def equation (a b : ℝ) : Prop := (2 / (3 * a + b)) + (1 / (a + 2 * b)) = 4
def positivity_condition (a b : ℝ) : Prop := a > 0 ∧ b > 0

-- Define the final statement to be proven in Lean
theorem range_and_minimum_value (m a b : ℝ) :
  (domain_condition m → m ≤ 4) ∧
  (positivity_condition a b → equation a b → 7 * a + 4 * b = 9 / 4) :=
by  
  sorry

end range_and_minimum_value_l10_10630


namespace fisherman_catch_total_l10_10471

theorem fisherman_catch_total :
  let bass := 32
  let trout := bass / 4
  let blue_gill := bass * 2
in bass + trout + blue_gill = 104 := by
  sorry

end fisherman_catch_total_l10_10471


namespace smallest_positive_solution_l10_10147

theorem smallest_positive_solution (x : ℝ) (h : tan (4 * x) + tan (5 * x) = sec (5 * x)) : x = Real.pi / 18 :=
sorry

end smallest_positive_solution_l10_10147


namespace exists_nk_for_sum_eq_2017_l10_10745

-- Definition of the sequence and the sum S(n, k)
def S (a : ℕ → ℤ) (n k : ℕ) : ℤ :=
  ∑ i in Finset.range (n+1), ∏ j in Finset.range (k+1), a (i + j)

-- The conjecture/proof statement
theorem exists_nk_for_sum_eq_2017 (a : ℕ → ℤ) (h : ∀ i : ℕ, a i = 1 ∨ a i = -1) :
  ∃ n k : ℕ, |S a n k| = 2017 :=
begin
  sorry,
end

end exists_nk_for_sum_eq_2017_l10_10745


namespace radius_of_smallest_sphere_centered_at_origin_l10_10535

noncomputable def smallest_enclosing_sphere_radius (r : ℝ) : ℝ := 
  r * sqrt 3 + r

theorem radius_of_smallest_sphere_centered_at_origin 
  (radius : ℝ) 
  (r_eq : radius = 2) : 
  smallest_enclosing_sphere_radius (2 * radius) = 2 * sqrt 3 + 2 := 
by
  sorry

end radius_of_smallest_sphere_centered_at_origin_l10_10535


namespace distinct_lines_eq_seven_l10_10585

theorem distinct_lines_eq_seven (a b : ℕ) (ha : a ∈ {1, 2, 3}) (hb : b ∈ {1, 2, 3}) :
  ∃ n, n = 7 → (nlines : ax + by = @line a b = 0) → n := sorry

end distinct_lines_eq_seven_l10_10585


namespace area_of_triangle_ABC_l10_10727

theorem area_of_triangle_ABC 
  (A B C D E G : Type) 
  (BD CE : ℝ)
  (angle_G : ℝ)
  (h1 : BD = 10)
  (h2 : CE = 15)
  (h3 : angle_G = 30) :
  let area_ABC := 
        6 * (1 / 2 * (2/3 * BD) * (1/3 * CE) * Real.sin (angle_G / 180 * Real.pi / 6)) in
      area_ABC = 50 := by
{
    sorry
}

end area_of_triangle_ABC_l10_10727


namespace TommysFirstHousePrice_l10_10796

theorem TommysFirstHousePrice :
  ∃ (P : ℝ), 1.25 * P = 0.25 * 500000 ∧ P = 100000 :=
by
  use 100000
  split
  · norm_num
  · norm_num
  sorry

end TommysFirstHousePrice_l10_10796


namespace apex_angle_of_cone_l10_10800

theorem apex_angle_of_cone (r d : ℝ) (α : ℝ) (h₁ : r = 12) (h₂ : d = 13) :
  2 * real.arccot (4 / 3) = α ∨ 2 * real.arccot (3) = α :=
sorry

end apex_angle_of_cone_l10_10800


namespace polygon_sides_l10_10658

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_sides_l10_10658


namespace _l10_10916

noncomputable theorem unique_solution_x : (∃ x : ℝ, 0 < x ∧ x \sqrt(16 - x) + \sqrt(16 * x - x^3) ≥ 16) :=
  sorry

end _l10_10916


namespace solve_inequality_l10_10922

theorem solve_inequality (x : ℚ) :
  (3 / 20 : ℚ) + |x - (9 / 40 : ℚ)| + |x + (1 / 8 : ℚ)| < (1 / 2 : ℚ) ↔ 
  x ∈ set.Ioo (-(3 / 40 : ℚ)) (11 / 40 : ℚ) :=
sorry

end solve_inequality_l10_10922


namespace probability_award_winning_work_l10_10462

def probability_of_selecting_an_award_winning_work (elderly_proportion : ℝ) 
     (middle_aged_proportion : ℝ) (children_proportion : ℝ) 
     (elderly_win_prob : ℝ) (middle_aged_win_prob : ℝ) 
     (children_win_prob : ℝ) : ℝ :=
  elderly_proportion * elderly_win_prob + 
  middle_aged_proportion * middle_aged_win_prob + 
  children_proportion * children_win_prob

theorem probability_award_winning_work:
  let elderly_proportion := 3 / 5
  let middle_aged_proportion := 1 / 5
  let children_proportion := 1 / 5
  let elderly_win_prob := 0.6
  let middle_aged_win_prob := 0.2
  let children_win_prob := 0.1
in
probability_of_selecting_an_award_winning_work 
  elderly_proportion middle_aged_proportion children_proportion 
  elderly_win_prob middle_aged_win_prob children_win_prob = 0.42 :=
by
  sorry

end probability_award_winning_work_l10_10462


namespace complex_pure_imaginary_l10_10247

theorem complex_pure_imaginary (a : ℝ) (i : ℂ) (h_imaginary_unit : i * i = -1) :
  (∀ (z : ℂ), z = (a + (3 : ℂ) * i) / (1 + (2 : ℂ) * i) →
  (∃ (bi : ℂ), z = 0 + bi * i ∧ bi ≠ 0) → a = -6) :=
begin
  sorry,
end

end complex_pure_imaginary_l10_10247


namespace num_values_with_prime_sum_divisors_l10_10701

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem num_values_with_prime_sum_divisors :
  (Finset.filter (λ n, is_prime (sum_divisors n)) (Finset.range 51)).card = 5 :=
by sorry

end num_values_with_prime_sum_divisors_l10_10701


namespace no_obtuse_triangle_probability_l10_10169

noncomputable def probability_no_obtuse_triangle (points : Finset (ℝ × ℝ)) : ℝ :=
  -- probability value using the given conditions
  if points.card = 4 then 3/32 else 0

theorem no_obtuse_triangle_probability :
  ∀ (points : Finset (ℝ × ℝ)),
  (∀ (A B C : ℝ × ℝ), 
    A ∈ points → B ∈ points → C ∈ points → 
    ((angle A O B) < π / 2 ∧ (angle B O C) < π / 2 ∧ (angle C O A) < π / 2)) →
  probability_no_obtuse_triangle points = 3 / 32 :=
by
  -- We are skipping the detailed proof here
  sorry

end no_obtuse_triangle_probability_l10_10169


namespace solve_for_x_l10_10159

theorem solve_for_x (x : ℝ) (h : sqrt (5 * x + 9) = 11) : x = 112 / 5 :=
by {
  -- Here you can assume the proof steps would go.
  sorry
}

end solve_for_x_l10_10159


namespace trigonometric_identity_l10_10711

def d := Real.pi / 7

theorem trigonometric_identity : 
  3 * Real.sin (2 * d) * Real.sin (4 * d) * Real.sin (6 * d) * Real.sin (8 * d) * Real.sin (10 * d) / 
  (Real.sin d * Real.sin (2 * d) * Real.sin (3 * d) * Real.sin (4 * d) * Real.sin (5 * d)) = 3 :=
by
  sorry

end trigonometric_identity_l10_10711


namespace average_income_A_B_l10_10370

theorem average_income_A_B (A B C : ℝ)
  (h1 : (B + C) / 2 = 5250)
  (h2 : (A + C) / 2 = 4200)
  (h3 : A = 3000) : (A + B) / 2 = 4050 :=
by
  sorry

end average_income_A_B_l10_10370


namespace assignment_plans_count_correct_l10_10679

open Finset
open Fintype

variables (roles : Finset (role)) [Fintype role] 
          (volunteers : Finset (volunteer)) [Fintype volunteer]

def role := {translation, tour_guiding, etiquette, driving}
def volunteer := {Zhang, Liu, Li, Song, Wang}

def valid_volunteer_role (v : volunteer) : Finset role :=
  match v with
  | Zhang => {translation, tour_guiding}
  | _ => role

noncomputable def count_assignment_plans : ℕ :=
  ((volunteers.powerset.filter (λ s, s.card = 4)).toFinset.card) *
  ((powerset (roles ∪ valid_volunteer_role(translation) ∪ valid_volunteer_role(tour_guiding))).toFinset.card)

theorem assignment_plans_count_correct : count_assignment_plans role volunteer = 36 :=
by sorry

end assignment_plans_count_correct_l10_10679


namespace functional_equation_solution_l10_10129

noncomputable def my_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x + y) = f(x) + f(y)

theorem functional_equation_solution (f : ℝ → ℝ) (a : ℝ) :
  my_function f ↔ ∃ a : ℝ, ∀ x : ℝ, f(x) = a * x :=
by
  sorry

end functional_equation_solution_l10_10129


namespace sum_of_positive_integers_l10_10932

theorem sum_of_positive_integers (n : ℕ) (h : 1.5 * n - 6 < 7.5) (h_pos : n > 0) :
  n < 9 → ∑ k in finset.range 9, k = 36 :=
by
  sorry

end sum_of_positive_integers_l10_10932


namespace FashionDesignNotInServiceAreas_l10_10023

-- Define the service areas of Digital China
def ServiceAreas (x : String) : Prop :=
  x = "Understanding the situation of soil and water loss in the Yangtze River Basin" ∨
  x = "Understanding stock market trends" ∨
  x = "Wanted criminals"

-- Prove that "Fashion design" is not in the service areas of Digital China
theorem FashionDesignNotInServiceAreas : ¬ ServiceAreas "Fashion design" :=
sorry

end FashionDesignNotInServiceAreas_l10_10023


namespace finish_time_diff_l10_10320

-- Define the speeds of participants and race distance.
def malcolm_speed : ℕ := 5
def joshua_speed : ℕ := 7
def ellie_speed : ℕ := 6
def race_distance : ℕ := 15

-- Define the times to finish the race for each participant.
def time_malcolm() : ℕ := malcolm_speed * race_distance
def time_joshua() : ℕ := joshua_speed * race_distance
def time_ellie() : ℕ := ellie_speed * race_distance

-- Prove the time differences.
theorem finish_time_diff :
  time_joshua() - time_malcolm() = 30 ∧
  time_ellie() - time_malcolm() = 15 :=
by sorry

end finish_time_diff_l10_10320


namespace vector_combination_l10_10640

-- Definitions for vectors a, b, and c with the conditions provided
def a : ℝ × ℝ × ℝ := (-1, 3, 2)
def b : ℝ × ℝ × ℝ := (4, -6, 2)
def c (t : ℝ) : ℝ × ℝ × ℝ := (-3, 12, t)

-- The statement we want to prove
theorem vector_combination (t m n : ℝ)
  (h : c t = m • a + n • b) :
  t = 11 ∧ m + n = 11 / 2 :=
by
  sorry

end vector_combination_l10_10640


namespace ratio_of_numbers_l10_10771

theorem ratio_of_numbers
  (greater less : ℕ)
  (h1 : greater = 64)
  (h2 : less = 32)
  (h3 : greater + less = 96)
  (h4 : ∃ k : ℕ, greater = k * less) :
  greater / less = 2 := by
  sorry

end ratio_of_numbers_l10_10771


namespace distinct_cubes_l10_10061

-- Define the total number of ways to arrange 4 white and 4 blue unit cubes in a 2x2x2 cube
def totalWays : ℕ := Nat.choose 8 4

-- Define the rotation group for a cube (simplified)
-- Notice: this won't define rotations explicitly, but is used to state the problem clearly.
-- For simplicity, let |G| denote the size of the group of rotations of the cube, which is known to be 24.
def G_size : ℕ := 24

-- Using Burnside's Lemma, we want to prove the number of distinct arrangements up to rotation
theorem distinct_cubes : totalWays / G_size = 7 := by
  -- We denote the total number of fixed points by each group element as totalFixedPoints
  -- For the sake of the proof problem statement, we will only reference the division result.
  -- We assume the correctness of the fixed points calculation via exploratory steps.
  let totalFixedPoints : ℕ := 168
  show totalFixedPoints / G_size = 7 from sorry

end distinct_cubes_l10_10061


namespace remainder_sum_first_150_l10_10435

-- Definitions based on the conditions
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Lean statement equivalent to the mathematical problem
theorem remainder_sum_first_150 :
  (sum_first_n 150) % 11250 = 75 :=
by 
sorry

end remainder_sum_first_150_l10_10435


namespace sum_of_terms_l10_10976

noncomputable def arithmetic_sequence : Type :=
  {a : ℕ → ℤ // ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d}

theorem sum_of_terms (a : arithmetic_sequence) (h1 : a.val 1 + a.val 3 = 2) (h2 : a.val 3 + a.val 5 = 4) :
  a.val 5 + a.val 7 = 6 :=
by
  sorry

end sum_of_terms_l10_10976


namespace blueprint_conversion_proof_l10_10455

-- Let inch_to_feet be the conversion factor from blueprint inches to actual feet.
def inch_to_feet : ℝ := 500

-- Let line_segment_inch be the length of the line segment on the blueprint in inches.
def line_segment_inch : ℝ := 6.5

-- Then, line_segment_feet is the actual length of the line segment in feet.
def line_segment_feet : ℝ := line_segment_inch * inch_to_feet

-- Theorem statement to prove
theorem blueprint_conversion_proof : line_segment_feet = 3250 := by
  -- Proof goes here
  sorry

end blueprint_conversion_proof_l10_10455


namespace f_is_defined_correctly_f_is_odd_l10_10972

-- Defining f(x) according to the given conditions
def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x + x - 2
  else if x = 0 then 0
  else -2^(-x) + x + 2

-- Proving the properties
theorem f_is_defined_correctly (x : ℝ) :
  (f(x) = 
    if x > 0 then 2^x + x - 2 
    else if x = 0 then 0
    else -2^(-x) + x + 2) :=
by
  unfold f
  split_ifs
  repeat {sorry}

-- Proving that f(x) is an odd function
theorem f_is_odd (x : ℝ) :
  f(-x) = -f(x) :=
by
  unfold f
  split_ifs
  repeat {sorry}

end f_is_defined_correctly_f_is_odd_l10_10972


namespace compound_interest_years_l10_10778

theorem compound_interest_years (PV FV : ℝ) (r : ℝ) (n : ℕ) :
  PV = 781.25 →
  FV = 845 →
  r = 0.04 →
  PV / FV = (1 + r) ^ (-n) →
  n = 2 :=
sorry

end compound_interest_years_l10_10778


namespace odd_numbers_count_even_numbers_count_greater_than_3125_count_l10_10013

-- Define the set of digits
def digits := {0, 1, 2, 3, 4, 5}

-- Define permutation function A(n, k)
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Prove the number of four-digit odd numbers
theorem odd_numbers_count : 
  (A 3 1) * (A 4 1) * (A 4 2) = A 3 1 * A 4 1 * A 4 2 :=
by 
sorry

-- Prove the number of four-digit even numbers
theorem even_numbers_count : 
  (A 5 3) + (A 2 1) * (A 4 1) * (A 4 2) = A 5 3 + A 2 1 * A 4 1 * A 4 2 :=
by 
sorry

-- Prove the number of four-digit numbers greater than 3125
theorem greater_than_3125_count :
  2 * (A 5 3) + 3 * (A 4 2) + 2 * (A 3 1) = 2 * A 5 3 + 3 * A 4 2 + 2 * A 3 1 := 
by 
sorry

end odd_numbers_count_even_numbers_count_greater_than_3125_count_l10_10013


namespace find_f_log_value_l10_10203

noncomputable def f (x : ℝ) : ℝ := 
if -1 ≤ x ∧ x ≤ 0 then 3 * x + (4 / 9) else 
if x + 1 = x - 1 then f (x - 1) else 
if x + 2 = x then f (x - 2) else 0 -- A generic fallback value, to meet Lean definitions.

theorem find_f_log_value (h : ∀ x, f x = f (-x)) (h_per : ∀ x, f (x + 2) = f x) : 
  f (log (5 : ℝ) / log (1 / 3 : ℝ) ) = 1 := 
by
  sorry

end find_f_log_value_l10_10203


namespace distance_A1_to_plane_BDFE_angle_A1D_plane_BDFE_l10_10046

-- Definitions based on the given problem's conditions
structure Cube (V : Type) [MetricSpace V] :=
  (A B C D A1 B1 C1 D1 : V)
  (edge_length : ℝ)
  (is_vertex : A1 ∈ {A, B, ...}) -- Set of all vertices
  (A1_edge_length : λ x, x ∈ {A, B, C, D, A1, B1, C1, D1} → dist A1 x = edge_length)
  (midpoints : E F : V)
  (E_midpoint : E = midpoint B1 C1)
  (F_midpoint : F = midpoint C1 D1)

-- Problem statement (1): Distance from A1 to the plane BDFE
theorem distance_A1_to_plane_BDFE (V : Type) [MetricSpace V] (cube : Cube V) : 
  let plane_BDFE := plane {cube.B, cube.D, cube.F, cube.E} in
  dist_point_to_plane cube.A1 plane_BDFE = 1 := 
sorry

-- Problem statement (2): Angle between line A1D and plane BDFE
theorem angle_A1D_plane_BDFE (V : Type) [MetricSpace V] (cube : Cube V) :
  let line_A1D := line_through cube.A1 cube.D in
  let plane_BDFE := plane {cube.B, cube.D, cube.F, cube.E} in
  angle line_A1D plane_BDFE = 45 := 
sorry

end distance_A1_to_plane_BDFE_angle_A1D_plane_BDFE_l10_10046


namespace highest_prob_red_ball_l10_10666

-- Definitions
def total_red_balls : ℕ := 5
def total_white_balls : ℕ := 12
def total_balls : ℕ := total_red_balls + total_white_balls

-- Condition that neither bag is empty
def neither_bag_empty (r1 w1 r2 w2 : ℕ) : Prop :=
  (r1 + w1 > 0) ∧ (r2 + w2 > 0)

-- Define the probability of drawing a red ball from a bag
def prob_red (r w : ℕ) : ℚ :=
  if (r + w) = 0 then 0 else r / (r + w)

-- Define the overall probability if choosing either bag with equal probability
def overall_prob_red (r1 w1 r2 w2 : ℕ) : ℚ :=
  (prob_red r1 w1 + prob_red r2 w2) / 2

-- Problem statement to be proved
theorem highest_prob_red_ball :
  ∃ (r1 w1 r2 w2 : ℕ),
    neither_bag_empty r1 w1 r2 w2 ∧
    r1 + r2 = total_red_balls ∧
    w1 + w2 = total_white_balls ∧
    (overall_prob_red r1 w1 r2 w2 = 0.625) :=
sorry

end highest_prob_red_ball_l10_10666


namespace sqrt_pow_simplification_l10_10134

theorem sqrt_pow_simplification : (4thRoot ((sqrt 5)^5))^2 = 5 * (4thRoot 5) := 
by
  sorry

end sqrt_pow_simplification_l10_10134


namespace length_of_other_train_l10_10809

-- Definitions based on conditions
def length_first_train : ℝ := 280
def speed_first_train_km_h : ℝ := 120
def speed_second_train_km_h : ℝ := 80
def time_to_cross : ℝ := 9

-- Convert speeds from km/hr to m/s
def speed_first_train_m_s : ℝ := speed_first_train_km_h * 1000 / 3600
def speed_second_train_m_s : ℝ := speed_second_train_km_h * 1000 / 3600

-- Relative speed when running in opposite directions
def relative_speed_m_s : ℝ := speed_first_train_m_s + speed_second_train_m_s

-- Total distance covered = Relative speed * Time
def total_distance_covered : ℝ := relative_speed_m_s * time_to_cross

-- Proof statement
theorem length_of_other_train : ∃ L : ℝ, length_first_train + L = total_distance_covered ∧ L = 219.95 :=
  by
    sorry

end length_of_other_train_l10_10809


namespace river_width_proof_l10_10488

-- Define the constants given in the problem
constant depth : ℝ := 3
constant flow_rate_kmph : ℝ := 2
constant volume_per_minute : ℝ := 3200

-- Conversion factor from kmph to m/min
def flow_rate_m_per_min (flow_rate_kmph : ℝ) : ℝ :=
  (flow_rate_kmph * 1000) / 60

-- The condition that matches the problem statement.
def width_of_river (volume_per_minute depth flow_rate_kmph : ℝ) : ℝ :=
  volume_per_minute / (depth * flow_rate_m_per_min flow_rate_kmph)

-- The theorem to be proved
theorem river_width_proof :
  width_of_river volume_per_minute depth flow_rate_kmph = 32 :=
by
  sorry

end river_width_proof_l10_10488


namespace smallest_positive_solution_l10_10146

theorem smallest_positive_solution (x : ℝ) (h : tan (4 * x) + tan (5 * x) = sec (5 * x)) : x = Real.pi / 18 :=
sorry

end smallest_positive_solution_l10_10146


namespace max_sum_arithmetic_sequence_l10_10591

theorem max_sum_arithmetic_sequence (n : ℕ) (M : ℝ) (hM : 0 < M) 
  (a : ℕ → ℝ) (h_arith_seq : ∀ k, a (k + 1) - a k = a 1 - a 0) 
  (h_constraint : a 1 ^ 2 + a (n + 1) ^ 2 ≤ M) :
  ∃ S, S = (n + 1) * (Real.sqrt (10 * M)) / 2 :=
sorry

end max_sum_arithmetic_sequence_l10_10591


namespace density_of_second_part_l10_10097

theorem density_of_second_part (ρ₁ : ℝ) (V₁ V : ℝ) (m₁ m : ℝ) (h₁ : ρ₁ = 2700) (h₂ : V₁ = 0.25 * V) (h₃ : m₁ = 0.4 * m) :
  (0.6 * m) / (0.75 * V) = 2160 :=
by
  --- Proof omitted
  sorry

end density_of_second_part_l10_10097


namespace average_after_22nd_inning_eq_60_5_l10_10029

variable (A : ℝ) -- Let A be the average before the 22nd inning

-- Conditions
def total_runs_up_to_21_innings : ℝ := 21 * A
def runs_in_22nd_inning : ℝ := 134
def new_average_after_22nd_inning : ℝ := A + 3.5
def total_runs_after_22_innings : ℝ := 22 * new_average_after_22nd_inning

-- Proposition to prove
theorem average_after_22nd_inning_eq_60_5 (h : total_runs_up_to_21_innings + runs_in_22nd_inning = total_runs_after_22_innings) : A + 3.5 = 60.5 :=
by
  sorry

end average_after_22nd_inning_eq_60_5_l10_10029


namespace distinct_constructions_l10_10058

-- Definitions for the size of the cube and colors of the units
def cube_size : ℕ := 2
def white_units : ℕ := 4
def blue_units : ℕ := 4

-- Definition of the number of distinct ways to construct the cube
def distinct_ways_to_construct_cube : ℕ := 7

-- Theorem stating the number of distinct ways to construct the cube
theorem distinct_constructions (cs : ℕ) (w_units : ℕ) (b_units : ℕ) :
  cs = cube_size → w_units = white_units → b_units = blue_units → distinct_ways_to_construct_cube = 7 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end distinct_constructions_l10_10058


namespace distinct_cubes_l10_10060

-- Define the total number of ways to arrange 4 white and 4 blue unit cubes in a 2x2x2 cube
def totalWays : ℕ := Nat.choose 8 4

-- Define the rotation group for a cube (simplified)
-- Notice: this won't define rotations explicitly, but is used to state the problem clearly.
-- For simplicity, let |G| denote the size of the group of rotations of the cube, which is known to be 24.
def G_size : ℕ := 24

-- Using Burnside's Lemma, we want to prove the number of distinct arrangements up to rotation
theorem distinct_cubes : totalWays / G_size = 7 := by
  -- We denote the total number of fixed points by each group element as totalFixedPoints
  -- For the sake of the proof problem statement, we will only reference the division result.
  -- We assume the correctness of the fixed points calculation via exploratory steps.
  let totalFixedPoints : ℕ := 168
  show totalFixedPoints / G_size = 7 from sorry

end distinct_cubes_l10_10060


namespace natural_number_with_divisors_and_conditions_l10_10069

theorem natural_number_with_divisors_and_conditions :
  ∃ (N A B C : ℕ),
    (∃ (m n : ℕ), N = m^2 * n^2 ∧ m ≠ n) ∧
    N = 441 ∧
    A + B + C = 79 ∧
    A * A = B * C ∧
    (A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
    A ∣ N ∧ B ∣ N ∧ C ∣ N ∧
    ∀ d, d ∣ N → (d = 1 ∨ d = m ∨ d = n ∨ d = m * n ∨ d = m^2 ∨ d = n^2 ∨ 
                   d = m^2 * n ∨ d = m * n^2 ∨ d = m^2 * n^2)) :=
begin
  -- Proof goes here
  sorry
end

end natural_number_with_divisors_and_conditions_l10_10069


namespace ratio_of_profits_l10_10782

theorem ratio_of_profits (x : ℝ) : 
  let investment_ratio_pq := (7 : ℝ) / 5
      time_period_p := 5
      time_period_q := 10.999999999999998
      product_p := 7 * x * time_period_p
      product_q := 5 * x * time_period_q
      profit_ratio := product_p / product_q in
  profit_ratio = 7 / 11 := by
  -- Definitions for convenience
  let investment_ratio_pq := (7 : ℝ) / 5
  let time_period_p := 5
  let time_period_q := 10.999999999999998
  let product_p := 7 * x * time_period_p
  let product_q := 5 * x * time_period_q
  let profit_ratio := product_p / product_q
  
  -- Calculate profit ratios
  have h1 : product_p = 35 * x := by
    unfold product_p
    ring
    
  have h2 : product_q = 55 * x := by
    unfold product_q
    ring

  have h3 : profit_ratio = 35 * x / (55 * x) := by
    unfold profit_ratio
    rw [h1, h2]
    
  have h4 : profit_ratio = 35 / 55 := by
    simp [mul_div_cancel_left] using ne_of_gt (show 0 < x by sorry)

  exact calc
    profit_ratio = 35 / 55 : h4
               ... = 7 / 11 : by norm_num

end ratio_of_profits_l10_10782


namespace approx_value_of_1_05_pow_6_l10_10517

theorem approx_value_of_1_05_pow_6 : abs (1.05^6 - 1.34) < 0.01 := by
  sorry

end approx_value_of_1_05_pow_6_l10_10517


namespace isosceles_trapezoid_of_convex_quadrilateral_with_incircle_l10_10672

theorem isosceles_trapezoid_of_convex_quadrilateral_with_incircle
  {A B C D I : Point}
  (h_convex : convex A B C D)
  (incircle : incircle ABCD I)
  (h_condition : (dist A I + dist D I)^2 + (dist B I + dist C I)^2 = (dist A B + dist C D)^2) :
  isosceles_trapezoid A B C D :=
sorry

end isosceles_trapezoid_of_convex_quadrilateral_with_incircle_l10_10672


namespace monotonic_increase_interval_l10_10380

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_increase_interval : ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < (Real.log x) / x :=
by sorry

end monotonic_increase_interval_l10_10380


namespace num_sequences_l10_10594

theorem num_sequences (a : ℕ → ℤ) (h1 : a 1 = 0) (h100 : a 100 = 475)
  (h_diff : ∀ k, 1 ≤ k ∧ k < 100 → |a (k + 1) - a k| = 5) :
  (Nat.choose 99 2) = 4851 := 
by
  sorry

end num_sequences_l10_10594


namespace square_diagonal_length_inscribed_circle_area_l10_10080

noncomputable def side_length : ℝ := 40 * Real.sqrt 3

theorem square_diagonal_length (s : ℝ) (h : s = side_length) : ∃ d : ℝ, d = 40 * Real.sqrt 6 :=
by {
    use 40 * Real.sqrt 6,
    sorry
}

noncomputable def radius_length (s : ℝ) : ℝ := s / 2

theorem inscribed_circle_area (s : ℝ) (h : s = side_length) : ∃ A : ℝ, A = 1200 * Real.pi :=
by {
    let r := radius_length s,
    have hr : r = 20 * Real.sqrt 3 := by sorry,
    use Real.pi * r^2,
    rw hr,
    norm_num,
    sorry
}

end square_diagonal_length_inscribed_circle_area_l10_10080


namespace ellipse_equation_is_correct_line_equation_is_correct_l10_10597

-- Given conditions
variable (a b e x y : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (ab_order : b < a)
variable (minor_axis_half_major_axis : 2 * a * (1 / 2) = 2 * b)
variable (right_focus_shortest_distance : a - e = 2 - Real.sqrt 3)
variable (ellipse_equation : a^2 = b^2 + e^2)
variable (m : ℝ)
variable (area_triangle_AOB_is_1 : 1 = 1)

-- Part (I) Prove the equation of ellipse C
theorem ellipse_equation_is_correct :
  (∀ x y : ℝ, (x^2 / 4 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
sorry

-- Part (II) Prove the equation of line l
theorem line_equation_is_correct :
  (∀ x y : ℝ, (y = x + m) ↔ ((y = x + (Real.sqrt 10 / 2)) ∨ (y = x - (Real.sqrt 10 / 2)))) :=
sorry

end ellipse_equation_is_correct_line_equation_is_correct_l10_10597


namespace base_not_divisible_by_5_l10_10168

def is_not_divisible_by_5 (c : ℤ) : Prop :=
  ¬(∃ k : ℤ, c = 5 * k)

def check_not_divisible_by_5 (b : ℤ) : Prop :=
  is_not_divisible_by_5 (3 * b^3 - 3 * b^2 - b)

theorem base_not_divisible_by_5 :
  check_not_divisible_by_5 6 ∧ check_not_divisible_by_5 8 :=
by 
  sorry

end base_not_divisible_by_5_l10_10168


namespace problem_1_problem_2_l10_10039

-- Problem 1: Proof for A >= B
theorem problem_1 (a b c : ℝ) : 
  let A := a^2 + b^2 + c^2 + 14 
  let B := 2*a + 4*b + 6*c 
  A >= B :=
by 
  sorry

-- Problem 2: Simplifying the complex expression.
theorem problem_2 : 
  abs ((4 / 9) ^ (-1 / 2) - log 10 5) 
  + sqrt (log 10 2 ^ 2 - log 10 4 + 1)
  - 3 ^ (1 - log 3 2)
  + log 2 7 * log 7 3 * log 3 8 
  = 3 :=
by 
  sorry

end problem_1_problem_2_l10_10039


namespace perimeter_of_shaded_region_l10_10274

def is_center_of_circle (O : Point) (R : Point) (S : Point) (r : ℝ) : Prop :=
  dist O R = r ∧ dist O S = r

def arc_length (r : ℝ) (angle_fraction : ℝ) := 2 * r * Real.pi * angle_fraction

def total_perimeter (r : ℝ) (angle_fraction : ℝ) :=
  2 * r + arc_length r angle_fraction

theorem perimeter_of_shaded_region
  (O R S : Point)
  (r : ℝ)
  (angle_fraction : ℝ)
  (center_circle : is_center_of_circle O R S r)
  (angle_fraction_eq : angle_fraction = 5 / 6) :
  total_perimeter r angle_fraction = 16 + (40 / 3) * Real.pi :=
by
  cases center_circle with h1 h2
  have hr : r = 8 := by sorry
  rw [hr, angle_fraction_eq]
  have h2r : 2 * r = 16 := by sorry
  have arc_length_eq : arc_length r angle_fraction = (40 / 3) * Real.pi := by sorry
  rw [h2r, arc_length_eq]
  rfl

end perimeter_of_shaded_region_l10_10274


namespace conic_section_is_ellipse_l10_10439

-- Define the points and distance condition
def point₁ : (ℝ × ℝ) := (2, 4)
def point₂ : (ℝ × ℝ) := (8, -1)
def total_distance (x y : ℝ) : ℝ :=
  Real.sqrt ((x - point₁.1)^2 + (y - point₁.2)^2) + Real.sqrt ((x - point₂.1)^2 + (y - point₂.2)^2)

-- Define the condition that the total distance is constant
def distance_condition (x y : ℝ) : Prop := total_distance x y = 15

-- Problem statement: prove the conic section described is an ellipse
theorem conic_section_is_ellipse : ∀ x y : ℝ, distance_condition x y → (∃ r1 r2 : ℝ, r1^2 + r2^2 < 15^2) := 
by
  sorry

end conic_section_is_ellipse_l10_10439


namespace _l10_10915

noncomputable theorem unique_solution_x : (∃ x : ℝ, 0 < x ∧ x \sqrt(16 - x) + \sqrt(16 * x - x^3) ≥ 16) :=
  sorry

end _l10_10915


namespace expression_equivalence_l10_10441

theorem expression_equivalence (a b : ℝ) : 2 * a * b - a^2 - b^2 = -((a - b)^2) :=
by {
  sorry
}

end expression_equivalence_l10_10441


namespace remainder_when_divided_l10_10575

theorem remainder_when_divided (P D Q R D' Q' R' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  P % (D * D') = R + R' * D :=
by
  sorry

end remainder_when_divided_l10_10575


namespace part1_part2_l10_10991

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2
noncomputable def g (x : ℝ) : ℝ := Real.log (3 * x + 1) / Real.log 2
noncomputable def h (x : ℝ) : ℝ := (3 * x + 1) / (x + 1)

theorem part1 (x : ℝ) (hx : x ≥ 0) : g(x) ≥ f(x) := by
  sorry

theorem part2 (x : ℝ) (hx : x ∈ Ici 0) : (g(x) - f(x)) ∈ Icc 0 (Real.log 3 / Real.log 2) := by
  sorry

end part1_part2_l10_10991


namespace number_of_propositions_is_two_l10_10506

def is_proposition (s : String) : Prop :=
  -- just a dummy definition for illustration; exact conditions identified in solutions
  (s = "The empty set is a proper subset of any set.") ∨
  (s = "A natural number is an even number.")

def statements : List String := [
  "The empty set is a proper subset of any set.",
  "x^2 - 3x - 4 ≥ 0.",
  "Are two lines perpendicular to the same line necessarily parallel?",
  "A natural number is an even number."
]

theorem number_of_propositions_is_two : (statements.filter is_proposition).length = 2 := 
  by sorry

end number_of_propositions_is_two_l10_10506


namespace sequence_sum_13_l10_10958

open Nat

theorem sequence_sum_13 (a : ℕ → ℝ) (k : ℝ) 
  (h_seq : ∀ n : ℕ, a (n + 1) = a n + k)
  (h_eqn : 3 * (a 3 + a 5) + 2 * (a 7 + a 10 + a 13) = 24) :
  (∑ i in range 13, a (i + 1)) = 26 := by
  sorry

end sequence_sum_13_l10_10958


namespace initial_students_count_eq_16_l10_10673

variable (n T : ℕ)
variable (h1 : (T:ℝ) / n = 62.5)
variable (h2 : ((T - 70):ℝ) / (n - 1) = 62.0)

theorem initial_students_count_eq_16 :
  n = 16 :=
by
  sorry

end initial_students_count_eq_16_l10_10673


namespace evaluate_composition_l10_10647

def g (x : ℝ) : ℝ := 3 * x^2 + 5
def h (x : ℝ) : ℝ := -5 * x^3 + 2

theorem evaluate_composition : g(h(2)) = 4337 :=
by 
  sorry

end evaluate_composition_l10_10647


namespace rectangle_count_l10_10557

theorem rectangle_count (h_lines v_lines : Finset ℕ) (h_card : h_lines.card = 5) (v_card : v_lines.card = 5) :
  ∃ (n : ℕ), n = (h_lines.choose 2).card * (v_lines.choose 2).card ∧ n = 100 :=
by
  sorry 

end rectangle_count_l10_10557


namespace largest_of_consecutive_even_numbers_l10_10252

theorem largest_of_consecutive_even_numbers (x : ℤ) 
  (h : 3 * x + 6 = 1.6 * (x + 2)) : (x + 4) = 6 :=
sorry

end largest_of_consecutive_even_numbers_l10_10252


namespace cuckoo_clock_chimes_l10_10063

theorem cuckoo_clock_chimes (initial_hour : ℕ) (duration_hours : ℕ) : 
  initial_hour = 9 → duration_hours = 7 → 
  (∑ i in Finset.range (duration_hours + 1), let h := (initial_hour + i) % 12 in if h = 0 then 12 else h) = 43 := 
by
  intros h_initial h_duration
  rw [h_initial, h_duration]
  -- define the hours cuckoo sounds
  let fk := (λ i, let h := (9 + i) % 12 in if h = 0 then 12 else h)
  -- hours: 9 => 10, 11, 12, 1, 2, 3, 4
  have : ∑ i in Finset.range 8, fk i = 10 + 11 + 12 + 1 + 2 + 3 + 4,
  calc ∑ i in Finset.range 8, fk i
      = fk 0 + fk 1 + fk 2 + fk 3 + fk 4 + fk 5 + fk 6 + fk 7 : sorry
  ... = 9 + 10 + 11 + 12 + 1 + 2 + 3 + 4 : sorry
  ... = 43 : sorry
  exact this

end cuckoo_clock_chimes_l10_10063


namespace martin_correct_answers_l10_10335

theorem martin_correct_answers : 
  ∀ (Campbell_correct Kelsey_correct Martin_correct : ℕ), 
  Campbell_correct = 35 →
  Kelsey_correct = Campbell_correct + 8 →
  Martin_correct = Kelsey_correct - 3 →
  Martin_correct = 40 := 
by
  intros Campbell_correct Kelsey_correct Martin_correct h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  rw [h3]
  rfl

end martin_correct_answers_l10_10335


namespace identify_letter_X_l10_10334

-- Define the conditions
def date_behind_D (z : ℕ) : ℕ := z
def date_behind_E (z : ℕ) : ℕ := z + 1
def date_behind_F (z : ℕ) : ℕ := z + 14

-- Define the sum condition
def sum_date_E_F (z : ℕ) : ℕ := date_behind_E z + date_behind_F z

-- Define the target date behind another letter
def target_date_behind_another_letter (z : ℕ) : ℕ := z + 15

-- Theorem statement
theorem identify_letter_X (z : ℕ) :
  ∃ (x : Char), sum_date_E_F z = date_behind_D z + target_date_behind_another_letter z → x = 'X' :=
by
  -- The actual proof would go here; we'll defer it for now
  sorry

end identify_letter_X_l10_10334


namespace incorrect_conclusion_l10_10193

open Real

theorem incorrect_conclusion (a b : ℝ) (h1 : b < a) (h2 : a < 0) : ¬ ( (1/2)^b < (1/2)^a ) :=
by
  sorry

end incorrect_conclusion_l10_10193


namespace bus_speed_l10_10131

theorem bus_speed (S : ℝ) (h1 : 36 = S * (2 / 3)) : S = 54 :=
by
sorry

end bus_speed_l10_10131


namespace rectangle_enclosure_l10_10548
open BigOperators

theorem rectangle_enclosure (n m : ℕ) (hn : n = 5) (hm : m = 5) : 
  (∑ i in finset.range n, ∑ j in finset.range i, 1) * 
  (∑ k in finset.range m, ∑ l in finset.range k, 1) = 100 := by
  sorry

end rectangle_enclosure_l10_10548


namespace reduction_percentage_is_60_l10_10780

def original_price : ℝ := 500
def reduction_amount : ℝ := 300
def percent_reduction : ℝ := (reduction_amount / original_price) * 100

theorem reduction_percentage_is_60 :
  percent_reduction = 60 := by
  sorry

end reduction_percentage_is_60_l10_10780


namespace imaginary_part_of_complex_number_l10_10774

theorem imaginary_part_of_complex_number (z : ℂ) (h : z = (2 + complex.I) / (1 + 3 * complex.I)) : z.im = -1/2 := 
sorry

end imaginary_part_of_complex_number_l10_10774


namespace find_x_satisfying_inequality_l10_10919

open Real

theorem find_x_satisfying_inequality :
  ∀ x : ℝ, 0 < x → (x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16 ↔ x = 4) :=
by
  sorry

end find_x_satisfying_inequality_l10_10919


namespace rectangular_coordinates_of_transformed_point_l10_10953

theorem rectangular_coordinates_of_transformed_point
  (r α β : ℝ)
  (h1 : 3 = r * sin β * cos α)
  (h2 : -2 = r * sin β * sin α)
  (h3 : 5 = r * cos β) :
  ∀ (r α β : ℝ),
    (r * sin β * cos (α + real.pi), r * sin β * sin (α + real.pi), r * cos β) =
    (-3, 2, 5) :=
by
  intros,
  have hcos : cos (α + real.pi) = -cos α := by sorry,
  have hsin : sin (α + real.pi) = -sin α := by sorry,
  rw [hcos, hsin],
  simp [h1, h2, h3],
  sorry

end rectangular_coordinates_of_transformed_point_l10_10953


namespace eccentricity_hyperbola_l10_10187

noncomputable def hyperbola_foci := { F1 : ℝ × ℝ // F1 ≠ (0, 0)}  -- Ensuring that F1 is not at the origin
noncomputable def hyperbola := { F2 : ℝ × ℝ // true }  -- F2 can be any point, simplifying the statement

def perpendicular_condition (A B : ℝ × ℝ) (line_through_F2_perpendicular_to_x : ℝ × ℝ → Prop) : Prop :=
  ∀ F2, line_through_F2_perpendicular_to_x F2 → A = (F2.1, -B.2) 

def point_C_on_y_axis (F1 F2 B C : ℝ × ℝ) : Prop :=
  C.1 = 0 ∧ C.2 = -((F1.2 - B.2) / 2)

def perpendicular_vectors (A C B F1 : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - F1.1) + (A.2 - C.2) * (B.2 - F1.2) = 0

noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

theorem eccentricity_hyperbola (F1 F2 A B C : ℝ × ℝ) 
  (hF : hyperbola_foci)
  (heqA : A = (F2.1, (F2.2 * b ^ 2) / a))
  (heqB : B = (F2.1, -(F2.2 * b ^ 2) / a))
  (h_perpendicular : perpendicular_condition A B (λ F2, line_through_F2_perpendicular_to_x F2))
  (h_C : point_C_on_y_axis F1 F2 B C)
  (h_perp : perpendicular_vectors A C B F1)
  (h_ecc : eccentricity c a = sqrt 3) :
  eccentricity c a = sqrt 3 := sorry

end eccentricity_hyperbola_l10_10187


namespace max_g_in_interval_range_a_for_h_zeros_l10_10317

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x)

-- Define the function g when a = -2
def g (x : ℝ) : ℝ := x^2 * f (-2) x

-- Define the function h
def h (a : ℝ) (x : ℝ) : ℝ := x^2 / f a x - 1

-- Proof requirements
theorem max_g_in_interval : 
  ∀ x : ℝ, (0 < x) → (g x ≤ Real.exp (-2)) :=
sorry

theorem range_a_for_h_zeros (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, (0 < x₁) ∧ (x₁ < 16) ∧ (0 < x₂) ∧ (x₂ < 16) ∧ h a x₁ = 0 ∧ h a x₂ = 0) → 
  (Real.log 2 / 2 < a ∧ a < 2 / Real.exp 1) :=
sorry

end max_g_in_interval_range_a_for_h_zeros_l10_10317


namespace number_of_packs_of_cake_l10_10723

-- Define the total number of packs of groceries
def total_packs : ℕ := 14

-- Define the number of packs of cookies
def packs_of_cookies : ℕ := 2

-- Define the number of packs of cake as total packs minus packs of cookies
def packs_of_cake : ℕ := total_packs - packs_of_cookies

theorem number_of_packs_of_cake :
  packs_of_cake = 12 := by
  -- Placeholder for the proof
  sorry

end number_of_packs_of_cake_l10_10723


namespace top_card_king_or_queen_prob_l10_10496

theorem top_card_king_or_queen_prob : 
  ∀ (deck : Finset (Finset (Fin 13 × Fin 4))),
    deck.card = 52 →
    (∀ suit : Fin 4, ∃ ranks : Finset (Fin 13), ranks.card = 13) →
    (∃ top_card : Fin 13 × Fin 4,
      top_card ∈ deck ∧
      (top_card.fst = 0 ∨ top_card.fst = 1) → 
      Rational.mk 8 52 = Rational.mk 2 13) := 
  sorry

end top_card_king_or_queen_prob_l10_10496


namespace one_of_sum_of_others_l10_10739

theorem one_of_sum_of_others (a b c : ℝ) 
  (cond1 : |a - b| ≥ |c|)
  (cond2 : |b - c| ≥ |a|)
  (cond3 : |c - a| ≥ |b|) :
  (a = b + c) ∨ (b = c + a) ∨ (c = a + b) :=
by
  sorry

end one_of_sum_of_others_l10_10739


namespace correct_statements_l10_10605

variables (α β γ : Type) [Plane α] [Plane β] [Plane γ]
variables (m n : Type) [Line m] [Line n]

-- The definition of the statements to evaluate
def statement1 := (line_parallel_plane m α) ∧ (plane_intersection α β = n) → (line_parallel_line m n)
def statement2 := (plane_parallel_plane α β) ∧ (plane_parallel_plane β γ) → (plane_parallel_plane α γ)
def statement3 := (line_perpendicular_plane m α) ∧ (line_perpendicular_plane n β) ∧ (line_perpendicular_line m n) → (plane_perpendicular_plane α β)
def statement4 := (plane_perpendicular_plane α β) ∧ (line_in_plane m β) → (line_perpendicular_plane m α)
def statement5 := (plane_perpendicular_plane α β) ∧ (line_perpendicular_plane m β) ∧ ¬(line_in_plane m α) → (line_parallel_plane m α)

-- Proof problem to verify the correct statements
theorem correct_statements :
  (statement2 α β γ) ∧ (statement3 α β m n) ∧ (statement5 α β m) ∧ ¬(statement1 α β m n) ∧ ¬(statement4 α β m) :=
by sorry

end correct_statements_l10_10605


namespace quadratic_complete_square_l10_10662

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 - 7 * x + 6) = (x - 7 / 2) ^ 2 - 25 / 4 :=
by
  sorry

end quadratic_complete_square_l10_10662


namespace matrix_result_l10_10196

variable (M : Matrix (Fin 2) (Fin 2) ℝ)

def condition1 : M.mulVec (λ i, if i = 0 then 1 else 0) = (λ i, if i = 0 then 1 else 0) := 
  by sorry

def condition2 : M.mulVec (λ i, if i = 0 then 1 else 1) = (λ i, if i = 0 then 2 else 2) :=
  by sorry

theorem matrix_result : 
  (condition1 M) → (condition2 M) → M.mulVec ((λ i, if i = 0 then 1 else -1)) = (λ i, if i = 0 then -2 else -4) := 
  by sorry

end matrix_result_l10_10196


namespace min_distance_circle_line_l10_10925

theorem min_distance_circle_line :
  (∀ x y : ℝ, x^2 + y^2 = 1 → ∃ d ≥ 0, d = 4 ∧ (∃ x₀ y₀ : ℝ, (x₀^2 + y₀^2 = 1) ∧ 3*x₀ + 4*y₀ - 25 = 0 ∧ abs(x - x₀) + abs(y - y₀) = d)) :=
by 
  sorry

end min_distance_circle_line_l10_10925


namespace fisherman_total_fish_l10_10473

theorem fisherman_total_fish :
  let bass : Nat := 32
  let trout : Nat := bass / 4
  let blue_gill : Nat := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  show bass + trout + blue_gill = 104
  sorry

end fisherman_total_fish_l10_10473


namespace line_equation_is_correct_l10_10768

def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

theorem line_equation_is_correct (x y t : ℝ)
  (h1: x = 3 * t + 6)
  (h2: y = 5 * t - 7) :
  y = (5 / 3) * x - 17 :=
sorry

end line_equation_is_correct_l10_10768


namespace area_of_tangential_quadrilateral_l10_10785

-- Define the semiperimeter and area for the quadrilateral
noncomputable def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

noncomputable def area (s r : ℝ) : ℝ := s * r

-- Given: sum of opposite sides and radius of inscribed circle
axiom sum_of_opposite_sides {a b c d : ℝ} : a + c = 20 ∧ b + d = 20
axiom radius_of_inscribed_circle : ℝ := 4

-- Prove: The area of the tangential quadrilateral
theorem area_of_tangential_quadrilateral {a b c d : ℝ} 
    (h : a + c = 20 ∧ b + d = 20) (r : ℝ) (hr : r = 4) : 
    area (semiperimeter a b c d) r = 80 := 
by 
  calc
    area (semiperimeter a b c d) r = semiperimeter a b c d * r : rfl
    ... = 20 * 4 : sorry -- here you can detail the steps from the problem
    ... = 80 : by norm_num

end area_of_tangential_quadrilateral_l10_10785


namespace option_A_correct_option_C_correct_option_D_correct_l10_10443

-- Option A Proof
theorem option_A_correct {n : ℕ} (hX : X ∼ binomial n (1/3)) (hE : E(3 * X + 1) = 6) : n = 5 :=
by sorry

-- Option C Proof
theorem option_C_correct {Ω : Type*} [probability_space Ω] (A B : event Ω) 
  (h_PA : 0 < P(A)) (h_PB : 0 < P(B)) (h_cond : P(B | A) = P(B)) :
  P(A | B) = P(A) :=
by sorry

-- Option D Proof
theorem option_D_correct {Ω : Type*} [probability_space Ω] (A B : event Ω)
  (h_P_not_A : P(Aᶜ) = 1/2) (h_P_not_B_given_A : P(Bᶜ | A) = 2/3) (h_P_B_given_not_A : P(B | Aᶜ) = 1/4) :
  P(Bᶜ) = 17/24 :=
by sorry

end option_A_correct_option_C_correct_option_D_correct_l10_10443


namespace maximum_value_of_a_l10_10166

theorem maximum_value_of_a :
  (∀ x : ℝ, |x - 2| + |x - 8| ≥ a) → a ≤ 6 :=
by
  sorry

end maximum_value_of_a_l10_10166


namespace count_relatively_prime_to_36_in_range_l10_10873

theorem count_relatively_prime_to_36_in_range :
  let range := {n : ℕ | 14 ≤ n ∧ n < 80}
  ∃ count, (count = range.filter (λ n, Nat.gcd n 36 = 1)).card ∧ count = 35 :=
by
  sorry

end count_relatively_prime_to_36_in_range_l10_10873


namespace triangular_grid_burning_l10_10855

def O : Prop := sorry
def A : Prop := sorry
def B : Prop := sorry
def C : Prop := sorry
def D : Prop := sorry
def E : Prop := sorry
def F : Prop := sorry
def segment (x y : Prop) : Prop := sorry
def burns_in (segment : Prop) (minutes : ℕ) : Prop := sorry

theorem triangular_grid_burning : 
  (burns_in (segment B A) 5) ∧ (burns_in (segment F A) 5) :=
by sorry

end triangular_grid_burning_l10_10855


namespace reciprocals_of_roots_l10_10706

variable (a b c k : ℝ)

theorem reciprocals_of_roots (kr ks : ℝ) (h_eq : a * kr^2 + k * c * kr + b = 0) (h_eq2 : a * ks^2 + k * c * ks + b = 0) :
  (1 / (kr^2)) + (1 / (ks^2)) = (k^2 * c^2 - 2 * a * b) / (b^2) :=
by
  sorry

end reciprocals_of_roots_l10_10706


namespace china_land_area_scientific_l10_10381

-- Define the condition
def landAreaChina: ℝ := 9600000

-- Define the scientific notation representation function
def scientificNotation (n : ℝ) : ℝ × ℤ := 
if n = 9600000 then (9.6, 6) 
else (0, 0) -- default value for other cases

-- The theorem statement
theorem china_land_area_scientific : scientificNotation landAreaChina = (9.6, 6) := 
sorry

end china_land_area_scientific_l10_10381


namespace sequence_properties_l10_10970

def quasi_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a n = d

def given_sequence (a : ℕ → ℝ) (a1 : ℝ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, 0 < n → a n + a (n + 1) = 2 * n

theorem sequence_properties (a : ℕ → ℝ) (a1 : ℝ) (h : given_sequence a a1) :
  quasi_arithmetic_sequence a 2 ∧
  (∀ n : ℕ, a n =
    if odd n 
    then n + a1 - 1
    else n - a1) ∧ 
  (∑ i in range 20, a (i + 1)) = 200 :=
sorry

end sequence_properties_l10_10970


namespace function_monotonically_increasing_on_interval_l10_10577

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem function_monotonically_increasing_on_interval (e : ℝ) (h_e_pos : 0 < e) (h_ln_e_pos : 0 < Real.log e) :
  ∀ x : ℝ, e < x → 0 < Real.log x - 1 := 
sorry

end function_monotonically_increasing_on_interval_l10_10577


namespace extend_probability_measure_l10_10303

open MeasureTheory

variables {Ω : Type*} {ℱ : measurable_space Ω} {𝒫 : measure Ω} {C : set Ω}

/-- 
Assume ℱ is a measurable space on Ω, 𝒫 is a probability measure on ℱ, and 
C is a subset of Ω that does not belong to ℱ. 
--/
def extend_measure (ℱ : measurable_space Ω) (𝒫 : measure Ω) (C : set Ω) 
  (h𝒞 : ¬ measurable_set C) : Prop :=
  ∃ (𝒫' : measure Ω), 
  𝒫'.to_outer_measure.caratheodory = measurable_space.generate_from (ℱ.measurable_set' ∪ {C}) ∧
  ∀ (E : set Ω), 
  measurable_set E → 
  ℱ.measurable_set' E → 
  𝒫' E = 𝒫 E ∧
  -- Ensure countable additivity for the extended measure 𝒫'
  (∀ (A : ℕ → set Ω), 
  pairwise (disjoint on A) → 
  (∀ n, measurable_set (A n)) → 
  𝒫'.to_outer_measure (⋃ n, A n) = ∑' n, 𝒫'.to_outer_measure (A n))

noncomputable def extend_P {𝒫 : measure Ω} {ℱ : measurable_space Ω} {C : set Ω} 
  (h𝒞 : ¬ measurable_set C) : Prop :=
  extend_measure ℱ 𝒫 C h𝒞

theorem extend_probability_measure 
  (𝒫 : measure Ω) (ℱ : measurable_space Ω) (C : set Ω) 
  (hℱ : ℱ.measurable_set' = ℱ) 
  (h𝒞 : ¬ measurable_set C) : extend_P h𝒞 :=
sorry

end extend_probability_measure_l10_10303


namespace I_II_III_l10_10626

-- Define the functions and conditions
def f (x : ℝ) : ℝ := -x^3 + x^2

def g (a : ℝ) (x : ℝ) : ℝ := if x > 0 then begin
  let gx := g a x,
  let gea := a,
  some gx
else
  0
end

-- Problem (I)
def h (x : ℝ) : ℝ := (Real.exp (1 - x)) * f x

theorem I (a : ℝ) :
  tangentLine h (1, h 1) = λ x, -x + 1 := sorry

-- Problem (II)
theorem II (a x : ℝ) (hx : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_g : g a x ≥ -x^2 + (a + 2) * x) :
  a ≤ (Real.exp 2 - 2 * Real.exp 1) / (Real.exp 1 - 1) := sorry
  
-- Problem (III)
def F (a : ℝ) (x : ℝ) : ℝ := if x < 1 then f x else g a x

theorem III (a : ℝ) :
  (∀ (P Q : ℝ × ℝ), P.1 ≤ -1 → (F a P.1) = P.2 → (F a Q.1) = Q.2 → 
  (P.1 + Q.1) / 2 = 0 → (P.1 * Q.1) + (P.2 * Q.2) < 0) → 
  a ≤ 0 := sorry

end I_II_III_l10_10626


namespace min_stamps_to_make_60_l10_10871

theorem min_stamps_to_make_60 :
  ∃ c f : ℕ, 5 * c + 6 * f = 60 ∧ c + f = 10 :=
by { use [0, 10], split; simp }

end min_stamps_to_make_60_l10_10871


namespace angle_bisector_theorem_l10_10350

theorem angle_bisector_theorem
  (A B C D : Type)
  [linear_ordered_field B]
  [linear_ordered_field C]
  [linear_ordered_field D]
  (triangle_ABC : triangle B)
  (angle_bisector_ADC : is_angle_bisector A B C D) :
  (BD / DC) = (AB / AC) := 
sorry

end angle_bisector_theorem_l10_10350


namespace division_result_l10_10516

-- Define the arithmetic expression
def arithmetic_expression : ℕ := (20 + 15 * 3) - 10

-- Define the main problem
def problem : Prop := 250 / arithmetic_expression = 250 / 55

-- The theorem statement that needs to be proved
theorem division_result : problem := by
    sorry

end division_result_l10_10516


namespace average_speed_is_24_miles_per_hr_l10_10476

-- Define the conditions
variables (D : ℝ)

-- The speed from home to the office
def speed_to_office : ℝ := 20

-- The speed from the office back to home
def speed_to_home : ℝ := 30

-- Total distance and total time
def total_distance : ℝ := 2 * D
def total_time : ℝ := D / speed_to_office + D / speed_to_home

-- The average speed of the round trip
def average_speed : ℝ := total_distance / total_time

-- The proof statement
theorem average_speed_is_24_miles_per_hr : average_speed D = 24 :=
by
  -- skip the proof
  sorry

end average_speed_is_24_miles_per_hr_l10_10476


namespace percentage_increase_by_l10_10294

theorem percentage_increase_by (M F : ℕ) (h1 : F = 10) (h2 : M + F = 23) :
  ((M - F) * 100 / F : ℤ) = 30 :=
by
  have hM : M = 13 :=
    by
      rw [h1] at h2
      exact nat.add_left_cancel h2
  rw [hM, h1, nat.cast_sub, nat.cast_mul, nat.cast_div, nat.cast_add, nat.cast_mul, nat.cast_one, nat.cast_zero]
  norm_num

-- to acknowledge the non-computability of division over integers, the nat.cast_* operations convert to ℤ
-- the term norm_num helps simplifying numeric calculations to match Lean’s expectations
  sorry

end percentage_increase_by_l10_10294


namespace find_expression_value_l10_10969

noncomputable theory

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions
axiom geom_seq_pos : ∀ n, 0 < a n
axiom arith_seq : 3 * a 1 + 2 * a 2 = a 3

-- Definitions derived from the given problem
def geom_seq : Prop :=
∀ n, a (n+1) = a 1 * q ^ n

-- Proof problem
theorem find_expression_value
  (hq : ∃ q > 0, geom_seq)
  (h_arith : arith_seq) :
  (a 20 + a 19) / (a 18 + a 17) = 9 := by
  sorry

end find_expression_value_l10_10969


namespace Max_students_count_l10_10726

variables (M J : ℕ)

theorem Max_students_count :
  (M = 2 * J + 100) → 
  (M + J = 5400) → 
  M = 3632 := 
by 
  intros h1 h2
  sorry

end Max_students_count_l10_10726


namespace correlation_coefficient_of_line_l10_10263

noncomputable def correlation_coefficient (data : List (ℝ × ℝ)) : ℝ := sorry

theorem correlation_coefficient_of_line (n : ℕ) (x y : Fin n → ℝ)
  (h₁ : n ≥ 2)
  (h₂ : ¬ ∀ i, x i = x 0)
  (h₃ : ∀ i, y i = 1 / 2 * x i + 1) :
  correlation_coefficient (List.ofFn (λ i => (x i, y i))) = 1 := by
  sorry

end correlation_coefficient_of_line_l10_10263


namespace tangent_line_to_circle_l10_10974

theorem tangent_line_to_circle (c : ℝ) : 
  ((∃! p : ℝ × ℝ, 
    p.1 - p.2 + c = 0 ∧ (p.1 - 1) ^ 2 + p.2 ^ 2 = 2) ↔ c = -1 + ℝ.sqrt 2) := 
by {
  sorry
}

end tangent_line_to_circle_l10_10974


namespace reducibility_of_polynomial_l10_10111

open Polynomial

noncomputable def is_prime (n : ℕ) : Prop := ¬ (∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0)

theorem reducibility_of_polynomial (n : ℕ) (P : Polynomial (ZMod 2)) :
  (P = ∑ i in Finset.range (n + 1), monomial i 1) →
  (¬ is_prime (n + 1) → irreducible P → false) ∧
  (is_prime (n + 1) → irreducible P) := by
  sorry

end reducibility_of_polynomial_l10_10111


namespace total_crayons_l10_10903

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) (h1 : crayons_per_child = 5) (h2 : num_children = 10) : crayons_per_child * num_children = 50 :=
by
  rw [h1, h2]
  exact rfl

end total_crayons_l10_10903


namespace wizard_digits_and_gps_l10_10038

theorem wizard_digits_and_gps :
  ∃ (Б А Н К С Д : Nat),
    5 * (1000*Б + 100*А + 10*Н + К) = 6 * (100*С + 10*А + Д) ∧
    Б = 1 ∧ А = 0 ∧ Н = 8 ∧ К = 6 ∧ С = 9 ∧ Д = 5 ∧
    let gps := (С * 6 + 1, С * 6, 6 * 6, Д, Б, (К / 2), Н - 1),
    gps = (55.5430, 5317) := by
    sorry

end wizard_digits_and_gps_l10_10038


namespace goat_region_area_lower_bound_l10_10133

open Set

theorem goat_region_area_lower_bound
  {k n : ℕ}
  (Q : Fin k → E) -- E represents the Euclidean space
  (ℓ : Fin k → ℝ)
  (P : Fin n → E)
  (h_reach : ∀ i, ∃ j, ∥P i - Q j∥ ≤ ℓ j) -- Goat can reach all points P_i
  : ∃ conv_hull_area : ℝ,
      conv_hull_area = convexHull ℝ (range P).area :=
sorry

end goat_region_area_lower_bound_l10_10133


namespace prob_min_X_Y_eq_l10_10781

def prob_min_eq {α : Type*} [Fintype α] (X Y : α → ℕ) (k : ℕ) : ℝ := 
  ∑ ω in univ.filter (λ ω, min (X ω) (Y ω) = k), 1

def prob_X_eq {α : Type*} [Fintype α] (X : α → ℕ) (k : ℕ) : ℝ := 
  ∑ ω in univ.filter (λ ω, X ω = k), 1

def prob_Y_eq {α : Type*} [Fintype α] (Y : α → ℕ) (k : ℕ) : ℝ := 
  ∑ ω in univ.filter (λ ω, Y ω = k), 1

def prob_max_eq {α : Type*} [Fintype α] (X Y : α → ℕ) (k : ℕ) : ℝ := 
  ∑ ω in univ.filter (λ ω, max (X ω) (Y ω) = k), 1

theorem prob_min_X_Y_eq 
  {α : Type*} [Fintype α] 
  (X Y : α → ℕ) (k : ℕ) :
  prob_min_eq X Y k = prob_X_eq X k + prob_Y_eq Y k - prob_max_eq X Y k := 
by
  sorry

end prob_min_X_Y_eq_l10_10781


namespace circle_geometry_proof_l10_10278

theorem circle_geometry_proof 
  (O : Point) (A B C D E P : Point)
  (r : ℝ)
  (hO : ∃ (c : Circle), c.center = O ∧ c.radius = r)
  (h1 : AB ⊥ BC)
  (h2 : Collinear A D O E)
  (h3 : AP = AD)
  (h4 : AB = 4r) :

  (AP^2 = PB * AB) :=
  sorry

end circle_geometry_proof_l10_10278


namespace combined_length_of_snakes_is_90_l10_10522

def length_of_snakes_in_inches : ℕ :=
  let snake1 := 2 * 12 in -- 2 feet to inches
  let snake2 := 16 in     -- 16 inches
  let snake3 := 10 in     -- 10 inches
  let snake4 := 50 / 2.54 in -- 50 centimeters to inches
  let snake5 := 0.5 * 39.37 in -- 0.5 meters to inches
  round (snake1 + snake2 + snake3 + snake4 + snake5) -- rounding step included

theorem combined_length_of_snakes_is_90 : length_of_snakes_in_inches = 90 := sorry

end combined_length_of_snakes_is_90_l10_10522


namespace line_passes_through_vertex_twice_l10_10942

theorem line_passes_through_vertex_twice :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ (∀ a, (y = 2 * x + a ∧ ∃ (x y : ℝ), y = x^2 + 2 * a^2) ↔ a = a₁ ∨ a = a₂) :=
by
  sorry

end line_passes_through_vertex_twice_l10_10942


namespace diaries_ratio_l10_10730

variable (initial_diaries : ℕ)
variable (final_diaries : ℕ)
variable (lost_fraction : ℚ)
variable (bought_diaries : ℕ)

theorem diaries_ratio 
  (h1 : initial_diaries = 8)
  (h2 : final_diaries = 18)
  (h3 : lost_fraction = 1 / 4)
  (h4 : ∃ x : ℕ, (initial_diaries + x - lost_fraction * (initial_diaries + x) = final_diaries) ∧ x = 16) :
  (16 / initial_diaries : ℚ) = 2 := 
by
  sorry

end diaries_ratio_l10_10730


namespace rectangle_enclosed_by_four_lines_l10_10554

theorem rectangle_enclosed_by_four_lines : 
  let h_lines := 5
  let v_lines := 5
  (choose h_lines 2) * (choose v_lines 2) = 100 :=
by {
  sorry
}

end rectangle_enclosed_by_four_lines_l10_10554


namespace find_density_of_gold_l10_10321

theorem find_density_of_gold
  (side_length : ℝ)
  (gold_cost_per_gram : ℝ)
  (sale_factor : ℝ)
  (profit : ℝ)
  (density_of_gold : ℝ) :
  side_length = 6 →
  gold_cost_per_gram = 60 →
  sale_factor = 1.5 →
  profit = 123120 →
  density_of_gold = 19 :=
sorry

end find_density_of_gold_l10_10321


namespace angle_theta_l10_10010

-- Define our non-coplanar lines and fixed point
variables {O : Type} {a b : Type}

-- Define the non-coplanar condition and the fixed point
def non_coplanar (a b : Type) : Prop := sorry -- This is a placeholder

-- Define the angle formed by the lines
def angle (x y : Type) : ℝ := sorry -- This is a placeholder to represent the angle between two lines

-- Define the conditions
def condition1 := non_coplanar a b
def condition2 := ∃ (O : Type) (l1 l2 l3 : Type),
  angle l1 a = 60 ∧ angle l1 b = 60 ∧
  angle l2 a = 60 ∧ angle l2 b = 60 ∧
  angle l3 a = 60 ∧ angle l3 b = 60

-- Define the main theorem to prove
theorem angle_theta (a b : Type) : condition1 ∧ condition2 → angle a b = 60 :=
by
  sorry  -- Placeholder for the proof

end angle_theta_l10_10010


namespace Martin_correct_answers_l10_10337

theorem Martin_correct_answers (C K M : ℕ) 
  (h1 : C = 35)
  (h2 : K = C + 8)
  (h3 : M = K - 3) : 
  M = 40 :=
by
  sorry

end Martin_correct_answers_l10_10337


namespace angles_MAB_NAC_l10_10028

/-- Given equal chords AB and AC, and a tangent MAN, with arc BC's measure (excluding point A) being 200 degrees,
prove that the angles MAB and NAC are either 40 degrees or 140 degrees. -/
theorem angles_MAB_NAC (AB AC : ℝ) (tangent_MAN : Prop)
    (arc_BC_measure : ∀ A : ℝ , A = 200) : 
    ∃ θ : ℝ, (θ = 40 ∨ θ = 140) :=
by
  sorry

end angles_MAB_NAC_l10_10028


namespace point_on_line_AC_l10_10199

variables (A B C P : Type) [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup P]
variables [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ P]
variables (a b c p : A) (λ μ : ℝ)

theorem point_on_line_AC (h : p = λ • (b - a) + μ • (c - a)) : 
  ∃ µ' : ℝ, p = a + µ' • (c - a) :=
sorry

end point_on_line_AC_l10_10199


namespace students_in_all_three_activities_l10_10000

variables (TotalStudents MeditationStudents ChessStudents SculptureStudents StudentsTwoActivities StudentsThreeActivities : ℕ)

noncomputable def students_all_three : ℕ := 
  let a := TotalStudents - MeditationStudents - ChessStudents - SculptureStudents + StudentsTwoActivities + StudentsThreeActivities in
  let b := StudentsTwoActivities - StudentsThreeActivities in 
  a - b - StudentsThreeActivities

theorem students_in_all_three_activities : 
  TotalStudents = 25 ∧ 
  MeditationStudents = 15 ∧ 
  ChessStudents = 18 ∧ 
  SculptureStudents = 11 ∧ 
  StudentsTwoActivities = 6 ∧ 
  4 ≤ StudentsTwoActivities - StudentsThreeActivities ∧
  students_all_three 25 15 18 11 6 (students_all_three 25 15 18 11 6 0) = 7 :=
by
  sorry

end students_in_all_three_activities_l10_10000


namespace curved_surface_area_of_cone_l10_10817

noncomputable def CSA (r l : ℝ) : ℝ := Real.pi * r * l

theorem curved_surface_area_of_cone :
  let r := 7
  let l := 22
  CSA r l ≈ 483.48 := 
by
  let r := 7
  let l := 22
  have CSA_val : CSA r l = 154 * Real.pi := by
    sorry
  have CSA_numeric : 154 * Real.pi ≈ 483.48 := by
    sorry
  exact CSA_numeric

end curved_surface_area_of_cone_l10_10817


namespace sum_of_possible_values_of_g_zero_l10_10705

noncomputable def g (x : ℝ) : ℝ := sorry
def non_constant_polynomial (g : ℝ → ℝ) : Prop := degree g > 0

theorem sum_of_possible_values_of_g_zero (g : ℝ → ℝ) (h_poly : non_constant_polynomial g)
  (h_eq : ∀ x : ℝ, x ≠ 0 → g(x - 1) + g(x) + g(x + 1) = (g x)^2 / x^2) :
  g 0 = 0 :=
sorry

end sum_of_possible_values_of_g_zero_l10_10705


namespace area_triangle_PVQ_l10_10266

theorem area_triangle_PVQ (PQ QR RT US : ℝ) (hPQ : PQ = 8) (hQR : QR = 4) 
    (hRT : RT = 2) (hUS : US = 3) :
    let RS := PQ in
    let TU := RS - (RT + US) in
    let ratio := TU / PQ in
    let height_PQRS := QR in
    let height_PVQ := ratio * height_PQRS in
    let area_PVQ := 1 / 2 * PQ * height_PVQ in
  area_PVQ = 128 / 3 :=
by
  let RS := PQ
  let TU := RS - (RT + US)
  let ratio := TU / PQ
  let height_PQRS := QR
  let height_PVQ := ratio * height_PQRS
  let area_PVQ := 1 / 2 * PQ * height_PVQ
  sorry

end area_triangle_PVQ_l10_10266


namespace smallest_positive_solution_l10_10148

theorem smallest_positive_solution :
  ∃ x > 0, tan (4 * x) + tan (5 * x) = sec (5 * x) ∧ x = π / 26 := 
begin
  use π / 26,
  split,
  { exact real.pi_pos.trans (by norm_num), },
  split,
  { sorry, },
  { refl, }
end

end smallest_positive_solution_l10_10148


namespace count_whole_numbers_between_7_div_4_and_3_pi_l10_10646

theorem count_whole_numbers_between_7_div_4_and_3_pi : 
  let lower_bound := 7 / 4
      upper_bound := 3 * Real.pi
      count := (⌊upper_bound⌋ - ⌈lower_bound⌉ + 1)
  in count = 8 := 
by
  let lower_bound := 7 / 4
  let upper_bound := 3 * Real.pi
  let count := (⌊upper_bound⌋ - ⌈lower_bound⌉ + 1)
  have h1 : lower_bound = 1.75 := by linarith
  have h2 : upper_bound ≈ 9.42477 := by sorry
  exact count

end count_whole_numbers_between_7_div_4_and_3_pi_l10_10646


namespace smallest_positive_solution_tan_sec_5x_l10_10142

theorem smallest_positive_solution_tan_sec_5x:
  ∃ x : ℝ, 0 < x ∧ tan (4 * x) + tan (5 * x) = sec (5 * x) ∧ x = π / 18 := 
sorry

end smallest_positive_solution_tan_sec_5x_l10_10142


namespace license_plates_possible_l10_10078

open Function Nat

theorem license_plates_possible :
  let characters := ['B', 'C', 'D', '1', '2', '2', '5']
  let license_plate_length := 4
  let plate_count_with_two_twos := (choose 4 2) * (choose 5 2 * 2!)
  let plate_count_with_one_two := (choose 4 1) * (choose 5 3 * 3!)
  let plate_count_with_no_twos := (choose 5 4) * 4!
  let plate_count_with_three_twos := (choose 4 3) * (choose 4 1)
  plate_count_with_two_twos + plate_count_with_one_two + plate_count_with_no_twos + plate_count_with_three_twos = 496 := 
  sorry

end license_plates_possible_l10_10078


namespace jean_stuffies_fraction_l10_10289

theorem jean_stuffies_fraction :
  ∀ (total_stuffies : ℕ) (janet_stuffies : ℕ) (fraction_given_to_janet : ℚ), 
  total_stuffies = 60 → 
  janet_stuffies = 10 → 
  fraction_given_to_janet = 1 / 4 → 
  (janet_stuffies * (1 / (fraction_given_to_janet)) ≠ 0) →
  let given_away := janet_stuffies * (1 / (fraction_given_to_janet)),
      kept := total_stuffies - given_away in
  (kept / total_stuffies) = 1 / 3 :=
begin
  assume (total_stuffies janet_stuffies : ℕ) (fraction_given_to_janet : ℚ),
  assume h_total : total_stuffies = 60,
  assume h_janet : janet_stuffies = 10,
  assume h_fraction : fraction_given_to_janet = 1 / 4,
  assume h_nonzero : janet_stuffies * (1 / (fraction_given_to_janet)) ≠ 0,
  let given_away := janet_stuffies * (1 / (fraction_given_to_janet)),
  let kept := total_stuffies - given_away,
  have h_given_away : given_away = 40,
  { calc
      given_away
          = janet_stuffies * 4 : by rw [h_fraction, ←mul_div_assoc, one_div, mul_one_div_cancel 4 (by norm_num : 4 ≠ 0)]
      ... = 10 * 4             : by rw h_janet
      ... = 40                 : by norm_num },
  rw [h_total, h_given_away] at *,
  have h_kept : kept = 20,
  { calc
      kept
          = total_stuffies - given_away : rfl
      ... = 60 - 40                    : by rw [h_total, h_given_away]
      ... = 20                         : by norm_num },
  have h_fraction_kept : (kept / total_stuffies) = (1 : ℚ) / 3,
  { rw [h_kept, h_total], norm_num },
  exact h_fraction_kept
end

end jean_stuffies_fraction_l10_10289


namespace arithmetic_sequence_solutions_l10_10185

noncomputable def arithmetic_sequence_problems (a_3 a_7 a_4 a_6 : ℤ) (d a_n S_n : ℤ → ℤ) : Prop :=
  (a_3 * a_7 = -16) ∧ (a_4 + a_6 = 0) ∧
  ((∀ n: ℕ, a_n n = 10 - 2 * n) ∨ (∀ n: ℕ, a_n n = 2 * n - 10)) ∧
  ((S_n = λ n, n * (n - 9)) ∨ (S_n = λ n, -n * (n - 9)))

theorem arithmetic_sequence_solutions (a_3 a_7 a_4 a_6 : ℤ) (d a_n S_n : ℤ → ℤ) :
  (arithmetic_sequence_problems a_3 a_7 a_4 a_6 d a_n S_n) :=
begin
  sorry
end

end arithmetic_sequence_solutions_l10_10185


namespace chuck_total_time_on_trip_l10_10885

def distance_into_country : ℝ := 28.8
def rate_out : ℝ := 16
def rate_back : ℝ := 24

theorem chuck_total_time_on_trip : (distance_into_country / rate_out) + (distance_into_country / rate_back) = 3 := 
by sorry

end chuck_total_time_on_trip_l10_10885


namespace average_star_rating_is_four_l10_10756

-- Define the conditions
def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

-- Define total star points as per the conditions
def total_star_points : ℕ := (5 * five_star_reviews) + (4 * four_star_reviews) + (3 * three_star_reviews) + (2 * two_star_reviews)

-- Define the average rating calculation
def average_rating : ℚ := total_star_points / total_reviews

theorem average_star_rating_is_four : average_rating = 4 := 
by {
  -- Placeholder for the proof
  sorry
}

end average_star_rating_is_four_l10_10756


namespace general_term_arithmetic_sequence_sum_of_b_n_sequence_l10_10596

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b_n (n : ℕ) : ℚ := 1 / ((2 * n - 1) * (2 * n + 1))

noncomputable def T_n (n : ℕ) : ℚ := ∑ i in finset.range n, b_n i

theorem general_term_arithmetic_sequence (n : ℕ) :
  ∀ n : ℕ, a_n n = 2 * n - 1 := 
by sorry

theorem sum_of_b_n_sequence (n : ℕ) :
  T_n n = n / (2 * n + 1) :=
by sorry

end general_term_arithmetic_sequence_sum_of_b_n_sequence_l10_10596


namespace louis_age_l10_10291

variable (L J M : ℕ) -- L for Louis, J for Jerica, and M for Matilda

theorem louis_age : 
  (M = 35) ∧ (M = J + 7) ∧ (J = 2 * L) → L = 14 := 
by 
  intro h 
  sorry

end louis_age_l10_10291


namespace line_through_vertex_has_two_a_values_l10_10945

-- Definitions for the line and parabola as conditions
def line_eq (a x : ℝ) : ℝ := 2 * x + a
def parabola_eq (a x : ℝ) : ℝ := x^2 + 2 * a^2

-- The proof problem
theorem line_through_vertex_has_two_a_values :
  (∃ a1 a2 : ℝ, (a1 ≠ a2) ∧ (line_eq a1 0 = parabola_eq a1 0) ∧ (line_eq a2 0 = parabola_eq a2 0)) ∧
  (∀ a : ℝ, line_eq a 0 = parabola_eq a 0 → (a = 0 ∨ a = 1/2)) :=
sorry

end line_through_vertex_has_two_a_values_l10_10945


namespace range_of_x_on_obtuse_angle_on_ellipse_l10_10622

theorem range_of_x_on_obtuse_angle_on_ellipse :
  ∀ (x y : ℝ),
    (x^2 / 4 + y^2 = 1) →
    ((x + real.sqrt 3)^2 + y^2 + (x - real.sqrt 3)^2 + y^2 < 12) →
    abs x < 2 * real.sqrt (2 / 3) :=
by
  sorry

end range_of_x_on_obtuse_angle_on_ellipse_l10_10622


namespace find_distance_CD_l10_10387

noncomputable def distance_CD : ℝ :=
  let C : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (3, 6)
  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)

theorem find_distance_CD :
  ∀ (C D : ℝ × ℝ), 
  (C = (0, 0) ∧ D = (3, 6)) ∧ 
  (∃ x y : ℝ, (y^2 = 12 * x ∧ (x^2 + y^2 - 4 * x - 6 * y = 0))) → 
  distance_CD = 3 * Real.sqrt 5 :=
by
  sorry

end find_distance_CD_l10_10387


namespace even_sum_subsets_count_l10_10236

def numbers : List ℤ := [45, 68, 72, 85, 101, 144, 157, 172]

def is_even (n : ℤ) : Prop := n % 2 = 0

def even_sum_subsets : List (List ℤ) :=
  (numbers.combinations 4).filter (λ lst, is_even (lst.sum))

theorem even_sum_subsets_count : even_sum_subsets.length = 37 := 
  sorry

end even_sum_subsets_count_l10_10236


namespace cos_double_angle_l10_10639

theorem cos_double_angle (α : ℝ) (h : ∥((Real.cos α), (1 / 2))∥ = (Real.sqrt 2) / 2) : Real.cos (2 * α) = -1 / 2 :=
by 
  sorry

end cos_double_angle_l10_10639


namespace unique_function_satisfying_equation_l10_10906

theorem unique_function_satisfying_equation :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x^2 + f y) = y + f x^2) → ∀ x : ℝ, f x = x :=
by
  intro f h
  sorry

end unique_function_satisfying_equation_l10_10906


namespace average_star_rating_is_four_l10_10758

-- Define the conditions
def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

-- Define total star points as per the conditions
def total_star_points : ℕ := (5 * five_star_reviews) + (4 * four_star_reviews) + (3 * three_star_reviews) + (2 * two_star_reviews)

-- Define the average rating calculation
def average_rating : ℚ := total_star_points / total_reviews

theorem average_star_rating_is_four : average_rating = 4 := 
by {
  -- Placeholder for the proof
  sorry
}

end average_star_rating_is_four_l10_10758


namespace find_k_find_m_range_l10_10307

-- Define the function f
def f (x : ℝ) : ℝ := Real.log (5 - x)

-- Problem 1: Prove that k = -1 under given conditions
theorem find_k (k : ℝ) (h : 10 ^ f k = 10 ^ f 2 * 10 ^ f 3) : k = -1 :=
sorry

-- Problem 2: Prove the range of m under given conditions
theorem find_m_range (m : ℝ) (h1 : f (2 * m - 1) < f (m + 1)) 
  (h2 : 5 - (2 * m - 1) > 0) (h3 : 5 - (m + 1) > 0) : 2 < m ∧ m < 3 :=
sorry

end find_k_find_m_range_l10_10307


namespace price_after_discount_eq_cost_price_l10_10367

theorem price_after_discount_eq_cost_price (m : Real) :
  let selling_price_before_discount := 1.25 * m
  let price_after_discount := 0.80 * selling_price_before_discount
  price_after_discount = m :=
by
  let selling_price_before_discount := 1.25 * m
  let price_after_discount := 0.80 * selling_price_before_discount
  sorry

end price_after_discount_eq_cost_price_l10_10367


namespace average_of_integers_between_l10_10413

theorem average_of_integers_between (M : ℤ) (h1 : 21 < M) (h2 : M < 28) :
  (M = 22 ∨ M = 23 ∨ M = 24 ∨ M = 25 ∨ M = 26 ∨ M = 27) →
  (∑ m in {22, 23, 24, 25, 26, 27}.to_finset, m) / 6 = 24.5 :=
by
  intro h
  rw [Finset.sum]
  sorry

end average_of_integers_between_l10_10413


namespace line_l_through_A_and_chord_C1_length_is_2sqrt3_exists_point_P_on_line_l_eq_tangents_to_C1_C2_l10_10993

noncomputable def circle_C1 (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 4
noncomputable def circle_C2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1
noncomputable def point_A : ℝ × ℝ := (-2, 3)
noncomputable def line_l1 (x y : ℝ) : Prop := x = -2
noncomputable def line_l2 (x y : ℝ) : Prop := 3 * x - 4 * y + 18 = 0
noncomputable def point_P1 : ℝ × ℝ := (-2, 7)
noncomputable def point_P2 : ℝ × ℝ := (-6 / 11, 45 / 11)

theorem line_l_through_A_and_chord_C1_length_is_2sqrt3 :
  (∀ x y : ℝ, line_l1 x y ↔ l_cutting_chord_of_length_2sqrt3 x y) ∨
  (∀ x y : ℝ, line_l2 x y ↔ l_cutting_chord_of_length_2sqrt3 x y) :=
sorry

theorem exists_point_P_on_line_l_eq_tangents_to_C1_C2 :
  (∀ x y : ℝ, line_l1 x y ↔ |tangent_length_C1 x y| = |tangent_length_C2 x y| ->
    ((x, y) = point_P1 ∨ (x, y) = point_P2)) ∧
  (∀ x y : ℝ, line_l2 x y ↔ |tangent_length_C1 x y| = |tangent_length_C2 x y| ->
    ((x, y) = point_P1 ∨ (x, y) = point_P2)) :=
sorry

end line_l_through_A_and_chord_C1_length_is_2sqrt3_exists_point_P_on_line_l_eq_tangents_to_C1_C2_l10_10993


namespace unique_line_parallel_through_point_l10_10181

variable {α : Type} [plane α] (l : line) (P : point)

theorem unique_line_parallel_through_point (h_parallel_l_α: l ∥ α) (h_P_α: P ∈ α) : 
  ∃! m : line, (m ∥ l) ∧ (P ∈ m) ∧ (m ⊆ α) := 
sorry

end unique_line_parallel_through_point_l10_10181


namespace hyperbola_asymptote_eq_l10_10635

def sequence_an (n : ℕ) : ℚ := 1 / (n * (n + 1))

def sum_up_to (f : ℕ → ℚ) (m : ℕ) : ℚ :=
  (Finset.range m).sum f

theorem hyperbola_asymptote_eq (m : ℕ) (h1 : sum_up_to sequence_an m = 9 / 10)
  (x y : ℝ) :
  (m = 9) →
  (∀ x y, (x^2 / 10 - y^2 / 9 = 1 → ((y = 3 * x / √10 → x = 0)) ∨ (y = -3 * x / √10 → x = 0))) :=
by 
  sorry

end hyperbola_asymptote_eq_l10_10635


namespace right_angled_triangle_exists_l10_10285

theorem right_angled_triangle_exists (m : ℤ) :
  (∃ x y : ℤ, m * x ^ 2 - 2 * x - m + 1 = 0 ∧ m * y ^ 2 - 2 * y - m + 1 = 0 ∧
  (m = 1 ∧ (x = 2 ∨ x = 0) ∧ (y = 2 ∨ y = 0) ∧ ((x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0)))) ↔ (∃ a b c : ℝ, a = 2 ∧ b = 2 ∧ c = real.sqrt 8) :=
sorry

end right_angled_triangle_exists_l10_10285


namespace angle_bac_is_60degrees_l10_10297

-- Define the given conditions
variables {A B C O : Point}
variable {l : Line}
variable (M : Point) -- Midpoint of segment BC
variable (X : Point) -- Midpoint of segment AO

-- Define the necessary conditions and setup
def is_circumcenter (O : Point) (A B C : Point) : Prop := 
  ∀ P : Point, (dist P O = dist A O ∧ dist P O = dist B O ∧ dist P O = dist C O) 

def is_angle_bisector_perpendicular (l : Line) (A B C : Point) : Prop :=
  ¬ ∃ P : Point, is_midpoint P B C ∧ 
                 P ∈ l ∧ 
                 is_perpendicular l (angle_bisector A B C)

-- Define point membership and midpoint condition
def midpoint_on_line (X l : Point) : Prop := X ∈ l
def is_midpoint (M A B : Point) : Prop := dist M A = dist M B

theorem angle_bac_is_60degrees 
  (O : Point)
  (l : Line) 
  (M : Point) (X : Point)
  (h1 : is_circumcenter O A B C)
  (h2 : is_angle_bisector_perpendicular l A B C)
  (h3 : is_midpoint M B C)
  (h4 : midpoint_on_line X l) 
  (h5 : is_midpoint X A O) 
  : ∠ BAC = 60 :=
sorry

end angle_bac_is_60degrees_l10_10297


namespace solve_system_eq_l10_10824

theorem solve_system_eq (x1 x2 x3 x4 x5 : ℝ) :
  (x3 + x4 + x5)^5 = 3 * x1 ∧
  (x4 + x5 + x1)^5 = 3 * x2 ∧
  (x5 + x1 + x2)^5 = 3 * x3 ∧
  (x1 + x2 + x3)^5 = 3 * x4 ∧
  (x2 + x3 + x4)^5 = 3 * x5 →
  (x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨
  (x1 = 1/3 ∧ x2 = 1/3 ∧ x3 = 1/3 ∧ x4 = 1/3 ∧ x5 = 1/3) ∨
  (x1 = -1/3 ∧ x2 = -1/3 ∧ x3 = -1/3 ∧ x4 = -1/3 ∧ x5 = -1/3) :=
by
  sorry

end solve_system_eq_l10_10824


namespace probability_of_different_colors_is_correct_l10_10257

def chips_color_count : Type := 
  { blue : Nat // blue = 7 } ∧ 
  { red : Nat // red = 5 } ∧ 
  { yellow : Nat // yellow = 4 } ∧ 
  { green : Nat // green = 3 }

def total_chips (c : chips_color_count) : Nat :=
  c.blue + c.red + c.yellow + c.green

noncomputable def probability_different_colors (c : chips_color_count) : ℚ :=
  let total := total_chips c
  let prob_blue_non_blue := (c.blue / total) * ((c.red + c.yellow + c.green) / total)
  let prob_red_non_red := (c.red / total) * ((c.blue + c.yellow + c.green) / total)
  let prob_yellow_non_yellow := (c.yellow / total) * ((c.blue + c.red + c.green) / total)
  let prob_green_non_green := (c.green / total) * ((c.blue + c.red + c.yellow) / total)
  prob_blue_non_blue + prob_red_non_red + prob_yellow_non_yellow + prob_green_non_green

theorem probability_of_different_colors_is_correct (c : chips_color_count) : 
  probability_different_colors c = 262 / 361 := by
  sorry

end probability_of_different_colors_is_correct_l10_10257


namespace Phi_at_0_eq_half_l10_10440

noncomputable def standard_normal_curve (x : ℝ) : ℝ :=
  (1 / (Real.sqrt (2 * Real.pi))) * Real.exp (- (x^2) / 2)

def Phi (x0 : ℝ) : ℝ :=
  ∫ x in Set.Iic x0, standard_normal_curve x dx

theorem Phi_at_0_eq_half : Phi 0 = 0.5 := by
  sorry

end Phi_at_0_eq_half_l10_10440


namespace sum_of_possible_values_is_correct_l10_10075

def possible_n_values_sum : ℕ :=
  let set := {5, 8, 12, 14} in
  let n := λ n, n ∉ set ∧ 
    (let new_set := {5, 8, 12, 14, n}.toList.qsort (· < ·) in
     let median := new_set.get! 2 in
     let mean := (new_set.get! 0 + new_set.get! 1 + new_set.get! 2 + new_set.get! 3 + new_set.get! 4) / 5 in
     median = mean) in
  (if n 1 then 1 else 0) + (if n 21 then 21 else 0) + (if n (39 / 4) then 39 / 4 else 0)

theorem sum_of_possible_values_is_correct : possible_n_values_sum = 31.75 := 
  sorry

end sum_of_possible_values_is_correct_l10_10075


namespace min_value_distance_sum_l10_10200

noncomputable def ellipse : set (ℝ × ℝ) := {p | p.1^2 + 2 * p.2^2 = 2}
def f1 : ℝ × ℝ := (-1, 0)
def f2 : ℝ × ℝ := (1, 0)

theorem min_value_distance_sum (P : ℝ × ℝ) (h : P ∈ ellipse) :
  let d1 := (P.1 - f1.1)^2 + (P.2 - f1.2)^2
      d2 := (P.1 - f2.1)^2 + (P.2 - f2.2)^2
  in (Real.sqrt d1 + Real.sqrt d2) = 2 := sorry

end min_value_distance_sum_l10_10200


namespace sum_of_integer_solutions_l10_10359

def sqrt (x : ℝ) := Real.sqrt x

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  sqrt (10 * x - 21) - sqrt (5 * x ^ 2 - 21 * x + 21) ≥ 5 * x ^ 2 - 31 * x + 42

theorem sum_of_integer_solutions :
  (∑ x in {x : ℤ | (x : ℝ) ≥ (21 + Real.sqrt 21) / 10 ∧ satisfies_inequality (x : ℝ)}, x) = 7 :=
by
  sorry

end sum_of_integer_solutions_l10_10359


namespace reflection_line_sum_l10_10773

theorem reflection_line_sum (m b : ℝ) :
  let original := (-3 : ℝ, -1 : ℝ)
  let image := (5 : ℝ, 3 : ℝ)
  let midpoint := ((original.fst + image.fst) / 2, (original.snd + image.snd) / 2) in
  let assumed_line : ℝ → ℝ := λ x, m * x + b in
  assumed_line midpoint.fst = midpoint.snd → 
  m * 2 * midpoint.fst + 2 * b = m * original.fst + original.snd →
  m + b = 1 := 
by
  sorry

end reflection_line_sum_l10_10773


namespace train_speed_correct_l10_10453

def train_length : ℝ := 500
def crossing_time : ℝ := 29.997600191984642
def man_speed_km_hr : ℝ := 3
def man_speed_m_s : ℝ := man_speed_km_hr * 1000 / 3600
def relative_train_speed : ℝ := train_length / crossing_time
def train_speed_m_s : ℝ := relative_train_speed + man_speed_m_s
def train_speed_km_hr : ℝ := train_speed_m_s * 3.6

theorem train_speed_correct : train_speed_km_hr = 63 := by
  sorry

end train_speed_correct_l10_10453


namespace locus_of_intersection_l10_10792

section LocusIntersection

variable {r c : ℝ} (O A : ℝ × ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ)

-- assume O is the origin, and O is center of circle with radius r
-- A is on the diameter
-- M and N are symmetric points on the circle

def circleO (P : ℝ × ℝ) : Prop := (P.1 ^ 2 + P.2 ^ 2 = r ^ 2)

def symmetryPoints (A M N : ℝ × ℝ) : Prop := (M.1 = N.1) ∧ (M.2 = -N.2)

def locusC (P : ℝ × ℝ) : Prop := (P.1 ^ 2 + P.2 ^ 2 - c * P.1 = 0)

-- Main theorem statement
theorem locus_of_intersection : 
  ∀ (O : ℝ × ℝ) (A M N : ℝ × ℝ), 
  circleO A ∧ circleO M ∧ circleO N ∧ symmetryPoints A M N ∧ A.1 = c ∧ A.2 = 0 
  → locusC (O.1 / c * r ^ 2, O.2 / c * r ^ 2) :=
by
  sorry

end locus_of_intersection_l10_10792


namespace cards_given_to_Jeff_l10_10329

-- Definitions according to the conditions
def initial_cards : Nat := 304
def remaining_cards : Nat := 276

-- The proof problem
theorem cards_given_to_Jeff : initial_cards - remaining_cards = 28 :=
by
  sorry

end cards_given_to_Jeff_l10_10329


namespace jack_handing_in_amount_l10_10689

/-- Jack's register amounts -/
def amount_handing_in (b100 b50 b20 b10 b5 b1 leave_in_till : ℕ) : ℕ :=
  let total_notes := (b100 * 100) + (b50 * 50) + (b20 * 20) + (b10 * 10) + (b5 * 5) + b1
  total_notes - leave_in_till

/-- The conditions given in the problem -/
def jacks_notes := { b100 := 2, b50 := 1, b20 := 5, b10 := 3, b5 := 7, b1 := 27, leave_in_till := 300 }

/-- Proof statement for the amount of money Jack will be handing in -/
theorem jack_handing_in_amount :
  amount_handing_in jacks_notes.b100 jacks_notes.b50 jacks_notes.b20 jacks_notes.b10 jacks_notes.b5 jacks_notes.b1 jacks_notes.leave_in_till = 142 := by
  sorry

end jack_handing_in_amount_l10_10689


namespace probability_xi_3_l10_10614

variable (pass_rate : ℚ) (fail_rate : ℚ) (ξ : ℕ → ℚ)

-- Conditions
def pass_rate_condition : pass_rate = 3 / 4 := sorry
def fail_rate_condition : fail_rate = 1 / 4 := sorry

-- Event probability function (ξ(n) represents probability that the first qualified product is detected on the nth test)
def ξ_condition : ξ 3 = (fail_rate ^ 2) * pass_rate := sorry

-- Theorem statement for the problem
theorem probability_xi_3 : P(ξ 3) = 3 / 64 := by
  rw [ξ_condition]
  rw [pass_rate_condition]
  rw [fail_rate_condition]
  norm_num
  sorry

end probability_xi_3_l10_10614


namespace rectangular_solid_volume_l10_10546

theorem rectangular_solid_volume
  (a b c : ℝ)
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : a * c = 6)
  (h4 : b = 2 * a) :
  a * b * c = 12 := 
by
  sorry

end rectangular_solid_volume_l10_10546


namespace good_subset_inequality_l10_10595

-- Define what it means to be a good subset
def good_subset (X : set ℕ) (n : ℕ) : Prop :=
  ∀ a b ∈ X, (a + b) % 2 = 0 → (a + b) / 2 ∈ X

-- Define the function A(n) which counts the number of good subsets of {1, 2, ..., n}
def A (n : ℕ) : ℕ := -- Function to compute the number of good subsets
  sorry

-- State the main theorem
theorem good_subset_inequality : A(100) + A(98) ≥ 2 * A(99) + 6 :=
sorry

end good_subset_inequality_l10_10595


namespace general_formula_minimum_n_l10_10959

-- Definitions based on given conditions
def arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d
def sum_arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- Conditions of the problem
def a2 : ℤ := -5
def S5 : ℤ := -20

-- Proving the general formula of the sequence
theorem general_formula :
  ∃ a₁ d, arith_seq a₁ d 2 = a2 ∧ sum_arith_seq a₁ d 5 = S5 ∧ (∀ n, arith_seq a₁ d n = n - 7) :=
by
  sorry

-- Proving the minimum value of n for which Sn > an
theorem minimum_n :
  ∃ n : ℕ, (n > 14) ∧ sum_arith_seq (-6) 1 n > arith_seq (-6) 1 n :=
by
  sorry

end general_formula_minimum_n_l10_10959


namespace sum_alternating_g_eq_zero_l10_10308

noncomputable def g (x : ℝ) : ℝ := x^3 * (1 - x)^3

theorem sum_alternating_g_eq_zero :
  ∑ k in Finset.range 2021 \ {0}, (-1 : ℝ)^(k + 1) * g (k / 2021) = 0 :=
sorry

end sum_alternating_g_eq_zero_l10_10308


namespace asparagus_price_l10_10583

theorem asparagus_price (A : ℝ) 
  (h_asparagus : 60 * A) 
  (h_grapes : 40 * 2.5 = 100) 
  (h_apples : 700 * 0.5 = 350)
  (h_total : 60 * A + 100 + 350 = 630) : A = 3 := 
by 
  sorry

end asparagus_price_l10_10583


namespace worker_assignment_l10_10089

theorem worker_assignment (x : ℕ) (y : ℕ) 
  (h1 : x + y = 90)
  (h2 : 2 * 15 * x = 3 * 8 * y) : 
  (x = 40 ∧ y = 50) := by
  sorry

end worker_assignment_l10_10089


namespace arithmetic_sequence_sum_formula_l10_10956

noncomputable def S (n : ℕ) := -a_n - (1/2)^(n - 1)

theorem arithmetic_sequence (h1: ∀ n : ℕ, n > 0 →  S (n) = -a_n - (1/2)^(n - 1)) : 
  ∀ n : ℕ, n > 0 → 2^n * S n = -1 + (n - 1) * (-1) :=
sorry

theorem sum_formula (h1: ∀ n : ℕ, n > 0 →  S (n) = -a_n - (1/2)^(n - 1)) : 
  ∀ n: ℕ, n > 0 → (∑ i in range n, S i) = (n + 2)/(2^n) - 2 :=
sorry

end arithmetic_sequence_sum_formula_l10_10956


namespace max_selling_price_selling_price_for_target1_impossibility_of_target2_l10_10836

-- Condition Definitions
def purchase_price : ℝ := 30
def initial_selling_price : ℝ := 40
def initial_sales_volume : ℝ := 280
def price_increase_step : ℝ := 2
def sales_volume_decrease_step : ℝ := 20
def min_sales_volume : ℝ := 130
def sales_profit_target1 : ℝ := 3120
def sales_profit_target2 : ℝ := 3700

-- Sales volume as a function of selling price
def sales_volume (x : ℝ) : ℝ :=
  initial_sales_volume - ((x - initial_selling_price) / price_increase_step) * sales_volume_decrease_step

-- Sales profit as a function of selling price
def sales_profit (x : ℝ) : ℝ :=
  (x - purchase_price) * sales_volume x

-- Maximum selling price per backpack, given the sales volume constraint
theorem max_selling_price :
  ∀ x : ℝ, sales_volume x ≥ min_sales_volume → x ≤ 54 :=
sorry

-- Selling price for a sales profit of 3120 yuan, under the condition in the first part
theorem selling_price_for_target1 :
  ∀ x : ℝ, x ≤ 54 → sales_profit x = sales_profit_target1 → x = 42 :=
sorry

-- Impossibility of achieving a sales profit of 3700 yuan
theorem impossibility_of_target2 :
  ∀ x : ℝ, sales_profit x = sales_profit_target2 → false :=
sorry

end max_selling_price_selling_price_for_target1_impossibility_of_target2_l10_10836


namespace double_mean_value_function_correct_range_l10_10898

-- Define the function f
def f (x a : ℝ) : ℝ := x^3 - x^2 + a + 1

-- Define the derivative of the function f
def f_prime (x : ℝ) : ℝ := 3 * x^2 - 2 * x

-- Define the condition for a function to be a double mean value function on [0, a]
def is_double_mean_value_function (a : ℝ) : Prop :=
  ∃ x1 x2 ∈ Ioo 0 a, f_prime x1 = f_prime x2 ∧ f_prime x1 = (f a - f 0) / a

-- Define the correct range for the parameter a
def correct_range (a : ℝ) : Prop := 1 / 2 < a ∧ a < 1

-- State the theorem
theorem double_mean_value_function_correct_range (a : ℝ) :
  is_double_mean_value_function a ↔ correct_range a := sorry

end double_mean_value_function_correct_range_l10_10898


namespace complement_and_intersection_l10_10721

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {-2, -1, 0}
def B : Set ℤ := {0, 1, 2}

theorem complement_and_intersection :
  ((U \ A) ∩ B) = {1, 2} := 
by
  sorry

end complement_and_intersection_l10_10721


namespace prove_quadrilateral_identity_l10_10742

-- Define the geometric conditions
variables (a b c d : ℝ)

-- Hypotheses
axiom quadrilateral_inscribed_in_semicircle : ∀
  (ABCD: ℝ) (a b c d: ℝ), 
  (d / 2) ^ 2 = (a^2 + b^2 + c^2) / ((2: ℤ)) 

-- The theorem we want to prove
theorem prove_quadrilateral_identity
  (a b c d : ℝ)
  (h1 : quadrilateral_inscribed_in_semicircle a b c d) : d^3 - (a^2 + b^2 + c^2) * d - 2 * a * b * c = 0 :=
by
  sorry

end prove_quadrilateral_identity_l10_10742


namespace find_piece_area_l10_10847

-- Given
def rectangle_area : ℝ := 1000

-- Definitions
def AB := 2 * a
def BC := 2 * b

-- Conditions
def condition_1 : ab = 250 := by sorry  -- ab = 250

-- Correct Answer
def result : ℝ := 125 / 3

-- Proof problem
theorem find_piece_area :
  (piece_area rectangle_area) = result := by sorry

end find_piece_area_l10_10847


namespace parabola_equation_and_m_value_l10_10182

-- Definitions based on the conditions
def is_parabola_with_vertex_origin_axis_x (P : ℝ → ℝ → Prop) : Prop :=
  ∃ (p : ℝ), p > 0 ∧ ∀ (x y : ℝ), P x y ↔ y^2 = -2 * p * x

def is_point_on_parabola (P : ℝ → ℝ → Prop) (M : ℝ × ℝ) : Prop :=
  let (x, y) := M in P x y

def distance (P : ℝ × ℝ) (Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Mathematically equivalent proof problem statement
theorem parabola_equation_and_m_value
  (P : ℝ → ℝ → Prop)
  (H1 : is_parabola_with_vertex_origin_axis_x P)
  (M : ℝ × ℝ)
  (H2 : M = (-3, M.2))
  (H3 : distance M (-H1.some_spec.some / 2, 0) = 5)
  : ∃ (m : ℝ), m = 2 * real.sqrt 6 ∨ m = -2 * real.sqrt 6 ∧ P (-3) m ∧ ∀ x y, P x y → y^2 = -8 * x := 
sorry

end parabola_equation_and_m_value_l10_10182


namespace sum_log_geom_seq_l10_10684

noncomputable theory

theorem sum_log_geom_seq (a : ℕ → ℝ) (lg : ℝ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 4 = 2) (h3 : a 5 = 4) : 
  (∑ i in finset.range 8, lg (a i)) = 12 * (lg 2) :=
by
  sorry

end sum_log_geom_seq_l10_10684


namespace problem_l10_10243

theorem problem (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 6 = 976 :=
by
  sorry

end problem_l10_10243


namespace different_values_count_l10_10138

theorem different_values_count (i : ℕ) (h : 1 ≤ i ∧ i ≤ 2015) : 
  ∃ l : Finset ℕ, (∀ j ∈ l, ∃ i : ℕ, (1 ≤ i ∧ i ≤ 2015) ∧ j = (i^2 / 2015)) ∧
  l.card = 2016 := 
sorry

end different_values_count_l10_10138


namespace total_surface_area_of_rearranged_cube_l10_10481

theorem total_surface_area_of_rearranged_cube 
  (h1: 0 < (1: ℝ))
  (cut1: ℝ)
  (cut2: ℝ)
  (height_piece1: ℝ := 1 / 4)
  (height_piece2: ℝ := 1 / 6) 
  (height_piece3: ℝ := 1 - ((1 / 4) + (1 / 6))) :

  cut1 = height_piece1 ∧ cut2 = height_piece2
    ∧ height_piece3 = (1 - ((1 / 4) + (1 / 6))) ∧ 

  -- Ensure the cuts produce valid piece heights
  0 < height_piece1 ∧ 0 < height_piece2 ∧ 0 < height_piece3 ∧ 

  -- Sum of heights equals 1
  (height_piece1 + height_piece2 + height_piece3 = 1) →

  -- Calculate the total surface area
  let top_bottom_area := 2 in
  let side_area := 2 in
  let front_back_area := 2 in

  -- Sum of individual surface areas
  (top_bottom_area + side_area + front_back_area) = 6 :=
sorry

end total_surface_area_of_rearranged_cube_l10_10481


namespace pyramid_edge_length_l10_10079

theorem pyramid_edge_length
  (height pyramid : ℝ)
  (radius sphere : ℝ)
  (pyramid_tangent_sphere : ∀ (x : ℝ), True) -- placeholder condition for tangency
  (sphere_contacts_base : ∀ (x : ℝ), x = radius sphere) -- placeholder condition for contact with base
  (height pyramid = 9) 
  (radius sphere = 3) :
  let s := 4.5 in
  True := sorry

end pyramid_edge_length_l10_10079


namespace chips_drawn_consecutively_l10_10454
noncomputable def probability_of_drawing_specific_order : ℚ :=
  3! * 5! * 2! * 3! * 12 / 11!

theorem chips_drawn_consecutively :
  probability_of_drawing_specific_order = 1 / 385 := by
  sorry

end chips_drawn_consecutively_l10_10454


namespace wheel_rotation_distance_l10_10857

-- Define the conditions given in the problem
def diameter_wheel : ℝ := 28
def pi_approx : ℝ := 22 / 7 
def circumference_wheel : ℝ := diameter_wheel * pi_approx
--  A correct assumption to compute the cycloid length
-- def length_of_cycloid := circumference_wheel + 24

-- Theorem stating the mathematically equivalent problem
theorem wheel_rotation_distance
  (d_wheel : ℝ)
  (pi_approx : ℝ)
  (circumference_wheel : ℝ)
  (diameter_wheel_length : ℝ)
  : circumference_wheel = 2 * pi_approx * diameter_wheel_length :=
begin
  sorry
end

end wheel_rotation_distance_l10_10857


namespace range_of_t_l10_10365

noncomputable def f (x : ℝ) : ℝ := sorry

theorem range_of_t 
  (H1 : ∀ x ∈ set.Icc (-1 : ℝ) 1, ∀ m ∈ set.Icc (-1 : ℝ) 1, f x ≤ t^2 - 2 * m * t + 1)
  (H2 : ∀ x y, x < y → f x < f y)
  (H3 : ∀ x, f (-x) = -f x)
  (H4 : f (-1) = -1)
  (t : ℝ) :
  t ≥ 2 ∨ t ≤ -2 ∨ t = 0 :=
sorry

end range_of_t_l10_10365


namespace area_of_triangle_ABC_l10_10254

theorem area_of_triangle_ABC (A B C D E : Point) (L : ℝ) 
  (H1 : ∡ A B C = 90)
  (H2 : is_angle_bisector B A D)
  (H3 : is_angle_bisector B C E)
  (H4 : incenter_triangles_intersection A B C D E)
  (H5 : area_triangle B D E = L) : 
  area_triangle A B C = 2 * L := 
sorry

end area_of_triangle_ABC_l10_10254


namespace smallest_C_for_coin_split_l10_10110

theorem smallest_C_for_coin_split :
  ∀ (a : Fin 100 → ℝ),
  (∀ i, 0 < a i ∧ a i ≤ 1) →
  (∑ i, a i = 50) →
  ∃ C > 0,
  (C = (50 / 51)) ∧ (∃ (s1 s2 : Finset (Fin 100)),
   s1.card = 50 ∧
   s2.card = 50 ∧
   s1 ∪ s2 = Finset.univ ∧
   s1 ∩ s2 = ∅ ∧
   |(∑ i in s1, a i) - (∑ i in s2, a i)| ≤ C) :=
sorry

end smallest_C_for_coin_split_l10_10110


namespace unique_n_degree_polynomial_exists_l10_10348

theorem unique_n_degree_polynomial_exists (n : ℕ) (h : n > 0) :
  ∃! (f : Polynomial ℝ), Polynomial.degree f = n ∧
    f.eval 0 = 1 ∧
    ∀ x : ℝ, (x + 1) * (f.eval x)^2 - 1 = -((x + 1) * (f.eval (-x))^2 - 1) := 
sorry

end unique_n_degree_polynomial_exists_l10_10348


namespace sum_of_positive_integers_l10_10931

theorem sum_of_positive_integers (n : ℕ) (h : 1.5 * n - 6 < 7.5) (h_pos : n > 0) :
  n < 9 → ∑ k in finset.range 9, k = 36 :=
by
  sorry

end sum_of_positive_integers_l10_10931


namespace remaining_volume_l10_10057

noncomputable def side_length : ℝ := 6
noncomputable def radius : ℝ := 3
noncomputable def height : ℝ := 6

noncomputable def volume_of_cube (side: ℝ) : ℝ :=
  side ^ 3

noncomputable def volume_of_cylinder (r h: ℝ) : ℝ :=
  Real.pi * (r ^ 2) * h

theorem remaining_volume : 
  volume_of_cube side_length - volume_of_cylinder radius height = 216 - 54 * Real.pi := by
  sorry

end remaining_volume_l10_10057


namespace spot_reach_area_l10_10360

-- Given conditions as definitions
def side_length : ℝ := 1
def tether_length : ℝ := 3

-- Define a proof problem that states Spot can reach a total area outside the doghouse of 7π square yards
theorem spot_reach_area : 
  (area_sector_270_r3 + 2 * area_sector_45_r1 = 7 * π) :=
by sorry

-- Helper definitions for the sectors
def area_sector_270_r3 : ℝ := π * (tether_length)^2 * (270 / 360)
def area_sector_45_r1 : ℝ := π * (side_length)^2 * (45 / 360)

end spot_reach_area_l10_10360


namespace acute_angle_at_5_25_l10_10021

-- Define the positions
def hour_angle_at_5_25 : ℝ := 5 * 30 + (25 / 60) * 30
def minute_angle_at_5_25 : ℝ := 25 * 6

-- Prove the acute angle formed is 12.5 degrees
theorem acute_angle_at_5_25 : |hour_angle_at_5_25 - minute_angle_at_5_25| = 12.5 :=
by
  have ha : hour_angle_at_5_25 = 162.5 := by linarith
  have ma : minute_angle_at_5_25 = 150 := by linarith
  rw [ha, ma]
  calc
    abs (162.5 - 150) = abs 12.5 := by simp
    ... = 12.5 := by norm_num

end acute_angle_at_5_25_l10_10021


namespace probability_AIO26_l10_10682

noncomputable def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}
noncomputable def non_vowels : Finset Char := {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z', 'W'}
noncomputable def even_digits : Finset Char := {'0', '2', '4', '6', '8'}

theorem probability_AIO26 :
  let total_plates := 5 * 4 * 23 * 22 * 5 in
  let favorable_outcomes := 1 in
  (favorable_outcomes: ℚ) / total_plates = (1: ℚ) / 50600 :=
by
  sorry

end probability_AIO26_l10_10682


namespace chord_length_eq_sqrt_two_l10_10371

-- Define the circle equation and the line equation.
def circle (x y : ℝ) := (x + 2)^2 + (y - 2)^2 = 1
def line (x y : ℝ) := x - y + 3 = 0

-- Define a theorem to prove the length of the chord cut by the line from the circle.
theorem chord_length_eq_sqrt_two :
  ∃ l : ℝ, (∀ (x y : ℝ), line x y → circle x y) ∧ l = sqrt 2 :=
by
  sorry -- Proof can be provided later.

end chord_length_eq_sqrt_two_l10_10371


namespace Chandler_more_rolls_needed_l10_10940

theorem Chandler_more_rolls_needed :
  let total_goal := 12
  let sold_to_grandmother := 3
  let sold_to_uncle := 4
  let sold_to_neighbor := 3
  let total_sold := sold_to_grandmother + sold_to_uncle + sold_to_neighbor
  total_goal - total_sold = 2 :=
by
  sorry

end Chandler_more_rolls_needed_l10_10940


namespace Charan_work_rate_l10_10862

-- Definitions based on conditions
def Ajay_Balu_work_rate : ℝ := 1 / 12
def Balu_Charan_work_rate : ℝ := 1 / 16

axiom Ajay_Balu : Ajay + Balu = Ajay_Balu_work_rate
axiom Balu_Charan : Balu + Charan = Balu_Charan_work_rate

axiom Work_done : 5 * Ajay + 7 * Balu + 13 * Charan = 1

-- Theorem to prove
theorem Charan_work_rate :
  Charan = 1 / 24 :=
sorry

end Charan_work_rate_l10_10862


namespace count_four_digit_numbers_divisible_by_5_and_3_l10_10645

theorem count_four_digit_numbers_divisible_by_5_and_3 : 
  { n : ℕ // 1000 ≤ n ∧ n ≤ 9999 ∧ (n % 5 = 0) ∧ (n % 3 = 0)}.card = 600 :=
sorry

end count_four_digit_numbers_divisible_by_5_and_3_l10_10645


namespace find_first_number_l10_10542

theorem find_first_number (x : ℤ) (k : ℤ) :
  (29 > 0) ∧ (x % 29 = 8) ∧ (1490 % 29 = 11) → x = 29 * k + 8 :=
by
  intros h
  sorry

end find_first_number_l10_10542


namespace lines_divide_circle_four_arcs_l10_10961

theorem lines_divide_circle_four_arcs (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → (y = x + a ∨ y = x + b)) → 
  ((abs a / (real.sqrt 2)) = (real.sqrt 2 / 2)) ∧ 
  ((abs b / (real.sqrt 2)) = (real.sqrt 2 / 2)) →
  a^2 + b^2 = 2 :=
by
  intro h₁ h₂ -- Introduce the hypotheses
  sorry -- Proof is omitted

end lines_divide_circle_four_arcs_l10_10961


namespace correct_statements_l10_10897

variable (f : ℝ → ℝ)

-- Given conditions
axiom h1 : ∀ x, f(x + 2) + f(x) = 0
axiom h2 : ∀ x, f(x + 1) = -f(-x - 1)

-- Statements to prove
lemma statement_2 : ∀ x, f(x + 1) = -f(-x + 1) → 
  ∀ x, f(x + 1) = f(1 - x) :=
sorry

lemma statement_3 : ∀ x, f(x + 2) + f(x) = 0 → 
  ∀ x, f(x + 3) = f(-x + 1) :=
sorry

lemma statement_5 : f(1) = 0 ∧ ∀ x, f(x + 2) + f(x) = 0  → 
  f(2013) = 0 :=
sorry

-- Combining the conclusions to match the statements.
theorem correct_statements : 
  (∀ x, f(x + 1) = -f(-x - 1) → ∀ x, f(x + 1) = f(1 - x)) ∧ 
  (∀ x, f(x + 2) + f(x) = 0 → ∀ x, f(x + 3) = f(-x + 1)) ∧ 
  (f(1) = 0 ∧ ∀ x, f(x + 2) + f(x) = 0 → f(2013) = 0) :=
⟨statement_2 f, statement_3 f, statement_5 f⟩

end correct_statements_l10_10897


namespace product_equals_eight_l10_10715

theorem product_equals_eight {x : ℝ} (hx : x ≠ 0) : x * (8/x) = 8 := by
  have h_div : 8 / x * x = 8 := by
    rewrite [mul_div_cancel' 8 hx]
  sorry

end product_equals_eight_l10_10715


namespace mushrooms_eaten_l10_10580

def original_amount : ℕ := 15
def leftover_amount : ℕ := 7
def eaten_amount : ℕ := 15 - 7 -- This can be written directly since it's trivial

theorem mushrooms_eaten : original_amount - leftover_amount = 8 :=
by
  unfold original_amount
  unfold leftover_amount
  simp  -- Simplify the expression to get the answer 8
  exact rfl -- Reflects that both sides are equal

end mushrooms_eaten_l10_10580


namespace third_term_coefficient_binomial_expansion_l10_10331

theorem third_term_coefficient_binomial_expansion :
  binomial_coefficient 10 2 = 45 :=
begin
  sorry,
end

end third_term_coefficient_binomial_expansion_l10_10331


namespace any_nat_as_quotient_of_fashionable_l10_10722

/-- Defining what it means for a number to be fashionable --/
def is_fashionable (n : ℕ) : Prop :=
  ∃ k : ℕ, n /= 0 ∧ String.isInfix "2016" (Nat.digits 10 n).reverse.mkString

/-- The main theorem to prove --/
theorem any_nat_as_quotient_of_fashionable (N : ℕ) :
  ∃ (A D : ℕ), is_fashionable A ∧ is_fashionable D ∧ N = A / D :=
begin
  sorry
end

end any_nat_as_quotient_of_fashionable_l10_10722


namespace right_triangle_side_lengths_l10_10598

theorem right_triangle_side_lengths (a S : ℝ) (b c : ℝ)
  (h1 : S = b + c)
  (h2 : c^2 = a^2 + b^2) :
  b = (S^2 - a^2) / (2 * S) ∧ c = (S^2 + a^2) / (2 * S) :=
by
  sorry

end right_triangle_side_lengths_l10_10598


namespace arithmetic_sequence_problem_l10_10191

variable (a : ℕ → ℤ)
variable (a1 d : ℤ)

-- Definitions based on the problem condition
def is_arithmetic_sequence :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The actual Lean 4 statement
theorem arithmetic_sequence_problem
  (h_arithmetic : is_arithmetic_sequence a a1 d)
  (h_sum : a 4 + a 7 + a 10 = 30) :
  a 3 - 2 * a 5 = -10 := 
  sorry

end arithmetic_sequence_problem_l10_10191


namespace minimum_boys_needed_l10_10101

theorem minimum_boys_needed (k n m : ℕ) (hn : n > 0) (hm : m > 0) (h : 100 * n + m * k = 10 * k) : n + m = 6 :=
by
  sorry

end minimum_boys_needed_l10_10101


namespace inequality_holds_l10_10244

theorem inequality_holds (x : ℝ) (hx : 0 ≤ x) : cos x ≥ 1 - (1/2) * x^2 :=
sorry

end inequality_holds_l10_10244


namespace prove_rectangular_selection_l10_10566

def number_of_ways_to_choose_rectangular_region (horizontals verticals : ℕ) : ℕ :=
  (Finset.choose horizontals 2) * (Finset.choose verticals 2)

theorem prove_rectangular_selection :
  number_of_ways_to_choose_rectangular_region 5 5 = 100 :=
by
  sorry

end prove_rectangular_selection_l10_10566


namespace rectangle_enclosed_by_four_lines_l10_10555

theorem rectangle_enclosed_by_four_lines : 
  let h_lines := 5
  let v_lines := 5
  (choose h_lines 2) * (choose v_lines 2) = 100 :=
by {
  sorry
}

end rectangle_enclosed_by_four_lines_l10_10555


namespace largest_multiple_of_7_smaller_than_negative_85_l10_10416

theorem largest_multiple_of_7_smaller_than_negative_85 :
  ∃ (n : ℤ), (∃ (k : ℤ), n = 7 * k) ∧ n < -85 ∧ ∀ (m : ℤ), (∃ (k : ℤ), m = 7 * k) ∧ m < -85 → m ≤ n := 
by
  use -91
  split
  { use -13
    norm_num }
  split
  { exact dec_trivial }
  { intros m hm
    cases hm with k hk
    cases hk with hk1 hk2
    have hk3 : k < -12 := by linarith
    have hk4 : k ≤ -13 := int.floor_le $ hk3
    linarith }


end largest_multiple_of_7_smaller_than_negative_85_l10_10416


namespace part1_part2_l10_10688

variable {A B C a b c : Real}

-- Define the triangle ABC with side lengths a, b, c opposite to angles A, B, and C respectively.
axiom triangle_ABC (ha : a > 0) (hb : b > 0) (hc : c > 0) : a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

-- Condition 1: Given equation.
axiom given_eq : 
  1 / 2 * sin(2 * B) * cos(C) + cos(B)^2 * sin(C) - sin(A / 2) * cos(A / 2) = 0

-- Proof that B = π / 3.
theorem part1 : B = π / 3 :=
by sorry

-- Circumradius R = √3
axiom circumradius : Real
noncomputable def R := Real.sqrt 3
axiom circumradius_given : circumradius = R

-- Point D is the midpoint of AC
axiom midpoint_D : D = (a + c) / 2

-- Given B = π / 3 and proving BD equality.
theorem part2 {D BD : Real} (hB : B = π / 3) : 
  BD = (Real.sqrt (2 * a^2 + 2 * c^2 - 9)) / 2 :=
by sorry

end part1_part2_l10_10688


namespace junk_items_count_l10_10725

variable (total_items : ℕ)
variable (useful_percentage : ℚ := 0.20)
variable (heirloom_percentage : ℚ := 0.10)
variable (junk_percentage : ℚ := 0.70)
variable (useful_items : ℕ := 8)

theorem junk_items_count (huseful : useful_percentage * total_items = useful_items) : 
  junk_percentage * total_items = 28 :=
by
  sorry

end junk_items_count_l10_10725


namespace intersection_M_N_l10_10220

noncomputable def M : set (ℝ × ℝ) := { p | (p.1^2) / 9 + (p.2^2) / 4 = 1 }
noncomputable def N : set (ℝ × ℝ) := { p | p.1 / 3 + p.2 / 2 = 1 }

theorem intersection_M_N :
  M ∩ N = {(3 : ℝ, 0 : ℝ), (0 : ℝ, 2 : ℝ)} :=
sorry

end intersection_M_N_l10_10220


namespace train_pass_jogger_time_l10_10811

theorem train_pass_jogger_time :
  let speed_jogger := 9 * 1000 / (60 * 60) in
  let distance_ahead := 240 in
  let train_length := 150 in
  let speed_train := 45 * 1000 / (60 * 60) in
  let relative_speed := speed_train - speed_jogger in
  let total_distance := distance_ahead + train_length in
  (total_distance / relative_speed) = 39 := by
  sorry

end train_pass_jogger_time_l10_10811


namespace abs_a_eq_2_abs_b_eq_5_abs_a_add_b_eq_4_abs_a_sub_b_l10_10245

variable (a b : ℂ) -- Using complex numbers to include all possible values

theorem abs_a_eq_2 (h1 : |a| = 2) : true :=
  by { sorry }

theorem abs_b_eq_5 (h2 : |b| = 5) : true :=
  by { sorry }

theorem abs_a_add_b_eq_4 (h3 : |a + b| = 4) : true :=
  by { sorry }

theorem abs_a_sub_b (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a + b| = 4) : |a - b| = 42 :=
  by { sorry }

end abs_a_eq_2_abs_b_eq_5_abs_a_add_b_eq_4_abs_a_sub_b_l10_10245


namespace number_of_black_cells_in_4x4_square_with_given_subsquare_counts_l10_10668

theorem number_of_black_cells_in_4x4_square_with_given_subsquare_counts :
  ∃ n : ℕ, (∃ sub_square_counts : vector ℕ 9,
      sub_square_counts = ⟨[0, 2, 2, 3, 3, 4, 4, 4, 4], by simp⟩) ∧
      n = 11 :=
by
  sorry

end number_of_black_cells_in_4x4_square_with_given_subsquare_counts_l10_10668


namespace smallest_positive_solution_tan_sec_5x_l10_10141

theorem smallest_positive_solution_tan_sec_5x:
  ∃ x : ℝ, 0 < x ∧ tan (4 * x) + tan (5 * x) = sec (5 * x) ∧ x = π / 18 := 
sorry

end smallest_positive_solution_tan_sec_5x_l10_10141


namespace range_of_ab_l10_10223

-- Given two positive numbers a and b such that ab = a + b + 3, we need to prove ab ≥ 9.

theorem range_of_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = a + b + 3) : 9 ≤ a * b :=
by
  sorry

end range_of_ab_l10_10223


namespace speed_ratio_l10_10093

noncomputable def k_value {u v x y : ℝ} (h_uv : u > 0) (h_v : v > 0) (h_x : x > 0) (h_y : y > 0) 
  (h_ratio : u / v = ((x + y) / (u - v)) / ((x + y) / (u + v))) : ℝ :=
  1 + Real.sqrt 2

theorem speed_ratio (u v x y : ℝ) (h_uv : u > 0) (h_v : v > 0) (h_x : x > 0) (h_y : y > 0) 
  (h_ratio : u / v = ((x + y) / (u - v)) / ((x + y) / (u + v))) : 
  u / v = k_value h_uv h_v h_x h_y h_ratio :=
sorry

end speed_ratio_l10_10093


namespace sum_of_arithmetic_sequence_b_n_is_arithmetic_sequence_l10_10615

-- Define the arithmetic sequence and relevant entities
variable (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (c : ℝ)

-- Define the initial conditions of the sequence and auxiliary sums
axiom h1 : ∀ n, a n = 1 + (n-1) * 4
axiom h2 : S = λ n, n * a 1 + (n * (n-1)) * (a 2 - a 1) / 2

-- Define the sequence {b_n} based on S_n and c
def b_n (n : ℕ) : ℝ := (S n) / (n + c)

-- Prove that S_n = 2n^2 - n
theorem sum_of_arithmetic_sequence : S = λ n, 2 * (n:ℝ)^2 - n := by
  sorry

-- Prove that {b_n} is an arithmetic sequence when c = -1/2
theorem b_n_is_arithmetic_sequence : (∀ n, b_n n - b_n (n-1) = (b_n 1 - b_n 0)) :=
  by
  let c := -1/2
  have b : ∀ n, b_n n = (2 * (n:ℝ))
  from
    sorry
  show ∀ n, (b_n (n + 1) - b_n n = b_n 1 - b_n 0) by
    sorry

end sum_of_arithmetic_sequence_b_n_is_arithmetic_sequence_l10_10615


namespace range_of_f_l10_10212

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 
  (sqrt 3) * sin (2 * x - φ) - cos (2 * x - φ)

theorem range_of_f :
  ∀ (φ : ℝ) (x : ℝ), 
  (|φ| < (π / 2)) → 
  (x ≥ - (π / 6) ∧ x ≤ (π / 3)) → 
  (f x φ ≥ -2 ∧ f x φ ≤ 1) := 
sorry

end range_of_f_l10_10212


namespace vasily_salary_correct_l10_10410

noncomputable def expected_salary_vasily : ℝ :=
  let salary_if_not_graduate : ℝ := 25000 in
  let salary_if_graduate : ℝ := 0.2 * 60000 + 0.1 * 80000 + 0.05 * 25000 + 0.65 * 40000 in
  let prob_graduate : ℝ := 0.9 in
  let prob_not_graduate : ℝ := 0.1 in
  prob_graduate * salary_if_graduate + prob_not_graduate * salary_if_not_graduate

theorem vasily_salary_correct : expected_salary_vasily = 45025 :=
by
  -- Skipping actual proof for brevity, we indicate its existence with sorry
  sorry

end vasily_salary_correct_l10_10410


namespace part1_part2_l10_10452

-- Part (1)
theorem part1 : |-2| + (1 + real.sqrt 3) ^ 0 - real.sqrt 9 = 0 :=
by
  sorry

-- Part (2)
theorem part2 {x : ℝ} : (2 * x + 1 > 3 * (x - 1)) ∧ (x + (x - 1) / 3 < 1) → x < 1 :=
by
  sorry

end part1_part2_l10_10452


namespace range_of_g_l10_10531

noncomputable def g (t : ℝ) : ℝ := (t^2 + 2 * t) / (t^2 - 1)

theorem range_of_g : set.range (λ t : {t : ℝ // t ≠ 1 ∧ t ≠ -1}, g t.val) = 
  { y : ℝ | y ≤ 1/2 } ∪ { y : ℝ | y ≥ 1 } :=
by
  sorry

end range_of_g_l10_10531


namespace alex_sandwiches_l10_10500

theorem alex_sandwiches : (nat.choose 8 2) * (nat.choose 7 1) = 196 :=
by
  sorry

end alex_sandwiches_l10_10500


namespace balloons_remaining_each_friend_l10_10400

def initial_balloons : ℕ := 250
def number_of_friends : ℕ := 5
def balloons_taken_back : ℕ := 11

theorem balloons_remaining_each_friend :
  (initial_balloons / number_of_friends) - balloons_taken_back = 39 :=
by
  sorry

end balloons_remaining_each_friend_l10_10400


namespace range_of_m_l10_10631

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 0 → 9^x - m * 3^x + m + 1 > 0) → m < 2 + 2 * Real.sqrt 2 :=
sorry

end range_of_m_l10_10631


namespace maximal_nice_tuple_sum_l10_10696

noncomputable def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

def isNice (n : ℕ) (a : Fin n → ℕ) : Prop :=
  gcd' (Finset.univ.image a) = 1 ∧ ∀ i : Fin n, a i ∣ a ((i : ℕ - 1) % n) + a ((i : ℕ + 1) % n)

theorem maximal_nice_tuple_sum (n : ℕ) (a : Fin n → ℕ) (h : isNice n a) :
  (Finset.univ.sum a) = fibonacci (n + 2) - 2 := 
sorry

end maximal_nice_tuple_sum_l10_10696


namespace find_valid_pairs_l10_10909

-- Define natural number range constraint
def valid_range (a b : ℕ) : Prop := 1 < a ∧ a ≤ 100 ∧ 1 < b ∧ b ≤ 100

-- Define equivalence to a natural number
def log_sum_is_nat (a b : ℕ) : Prop := ∃ k : ℕ, (Real.log 10 / Real.log a) + (Real.log 10 / Real.log b) = k

-- Define the ordered pairs that satisfy the given conditions
def valid_pairs : list (ℕ × ℕ) :=
  [(2, 5), (5, 2), (4, 25), (25, 4), (10, 10), (20, 5), 
   (5, 20), (10, 100), (100, 10), (20, 50), (50, 20), 
   (25, 40), (40, 25), (100, 100)]

-- Problem statement
theorem find_valid_pairs :
  ∀ (a b : ℕ), (a, b) ∈ valid_pairs → valid_range a b → log_sum_is_nat a b :=
by sorry

end find_valid_pairs_l10_10909


namespace find_x_l10_10157

theorem find_x (x : ℝ) (h : sqrt (5 * x + 9) = 11) : x = 112 / 5 :=
sorry

end find_x_l10_10157


namespace parallel_vectors_sin_cos_identity_l10_10995

theorem parallel_vectors_sin_cos_identity (x : ℝ) (h : sin x - 2 * cos x = 0) :
  2 * sin (x + π / 4) / (sin x - cos x) = 3 * real.sqrt 2 :=
by
  sorry

end parallel_vectors_sin_cos_identity_l10_10995


namespace grain_remaining_l10_10077

def originalGrain : ℕ := 50870
def spilledGrain : ℕ := 49952
def remainingGrain : ℕ := 918

theorem grain_remaining : originalGrain - spilledGrain = remainingGrain := by
  -- calculations are omitted in the theorem statement
  sorry

end grain_remaining_l10_10077


namespace f_neg_2016_eq_neg_2018_l10_10986

noncomputable def f : ℝ → ℝ
| x => if x > 0 then real.log x / real.log 2 + 2017 else - f (x + 2)

/-- Theorem statement: Given the function f as defined, f(-2016) = -2018. -/
theorem f_neg_2016_eq_neg_2018 : f (-2016) = -2018 :=
by {
    sorry
}

end f_neg_2016_eq_neg_2018_l10_10986


namespace smallest_positive_solution_l10_10149

theorem smallest_positive_solution :
  ∃ x > 0, tan (4 * x) + tan (5 * x) = sec (5 * x) ∧ x = π / 26 := 
begin
  use π / 26,
  split,
  { exact real.pi_pos.trans (by norm_num), },
  split,
  { sorry, },
  { refl, }
end

end smallest_positive_solution_l10_10149


namespace number_of_people_l10_10572

variable (N F : ℕ)
variable (h1 : F = 8)
variable (h2 : (N - F) / N - F / N = 0.36)

theorem number_of_people (N F : ℕ) (h1 : F = 8) (h2 : (N - F) / N - F / N = 0.36) : N = 25 := 
by
  sorry

end number_of_people_l10_10572


namespace mass_percentage_of_K_in_KBrO3_is_23_41_l10_10924

-- Defining the atomic masses of K, Br, and O
def atomic_mass_K : ℝ := 39.10
def atomic_mass_Br : ℝ := 79.90
def atomic_mass_O : ℝ := 16.00

-- Defining the molar mass of KBrO3
def molar_mass_KBrO3 : ℝ := (atomic_mass_K + atomic_mass_Br + 3 * atomic_mass_O)

-- Defining the mass of K in KBrO3
def mass_of_K_in_KBrO3 : ℝ := atomic_mass_K

-- Defining the mass percentage of K in KBrO3
def mass_percentage_of_K : ℝ := (mass_of_K_in_KBrO3 / molar_mass_KBrO3) * 100

-- The theorem to be proved
theorem mass_percentage_of_K_in_KBrO3_is_23_41 :
  mass_percentage_of_K = 23.41 :=
by
  sorry

end mass_percentage_of_K_in_KBrO3_is_23_41_l10_10924


namespace largest_multiple_of_seven_smaller_than_neg_85_l10_10431

theorem largest_multiple_of_seven_smaller_than_neg_85 
  : ∃ k : ℤ, (k * 7 < -85) ∧ (∀ m : ℤ, (m * 7 < -85) → (m * 7 ≤ k * 7)) ∧ (k = -13) 
  := sorry

end largest_multiple_of_seven_smaller_than_neg_85_l10_10431


namespace problem1_problem2_problem3_l10_10015

-- Prove \(2x = 4\) is a "difference solution equation"
theorem problem1 (x : ℝ) : (2 * x = 4) → x = 4 - 2 :=
by
  sorry

-- Given \(4x = ab + a\) is a "difference solution equation", prove \(3(ab + a) = 16\)
theorem problem2 (x ab a : ℝ) : (4 * x = ab + a) → 3 * (ab + a) = 16 :=
by
  sorry

-- Given \(4x = mn + m\) and \(-2x = mn + n\) are both "difference solution equations", prove \(3(mn + m) - 9(mn + n)^2 = 0\)
theorem problem3 (x mn m n : ℝ) :
  (4 * x = mn + m) ∧ (-2 * x = mn + n) → 3 * (mn + m) - 9 * (mn + n)^2 = 0 :=
by
  sorry

end problem1_problem2_problem3_l10_10015


namespace parabola_intersection_length_l10_10633

theorem parabola_intersection_length :
  let x := 4 * t^2 
  let y := 4 * t 
  (line_passing_through_focus (focus := (1, 0))) ∩ (parabola (equation := y^2 = 4 * x)) → length_AB = 8 
:= sorry

end parabola_intersection_length_l10_10633


namespace evaluate_expression_l10_10965

theorem evaluate_expression (a b c d m : ℤ) (h1 : a = -b) (h2 : c * d = 1) (h3 : |m| = 2) :
  3 * (a + b - 1) + (-c * d)^2023 - 2 * m = -8 ∨ 3 * (a + b - 1) + (-c * d)^2023 - 2 * m = 0 :=
by {
  sorry
}

end evaluate_expression_l10_10965


namespace correct_choice_d_l10_10192

variable {a1 a2 a3 a4 a5 : ℝ}
variable {q : ℝ}

-- Definitions of the conditions
def geo_seq (a1 a2 a3 a4 a5 : ℝ) (q : ℝ) : Prop :=
  a2 = a1 * q ∧ a3 = a1 * q^2 ∧ a4 = a1 * q^3 ∧ a5 = a1 * q^4

def condition (a1 a2 a5 : ℝ) (q : ℝ) : Prop :=
  geo_seq a1 a2 a3 a4 a5 q ∧ a2 * a5 < 0

-- The goal to prove
theorem correct_choice_d 
  (a1 a2 a3 a4 a5 : ℝ) 
  (q : ℝ) 
  (h : condition a1 a2 a5 q) : 
  a1 * a2 * a3 * a4 > 0 :=
sorry

end correct_choice_d_l10_10192


namespace sufficient_but_not_necessary_condition_for_even_function_l10_10213

/--
Given the function f(x) = x^2 + a(b+1)x + a + b (a, b ∈ ℝ),
the statement "a = 0" is a sufficient but not necessary condition for "f(x) is an even function".
-/
theorem sufficient_but_not_necessary_condition_for_even_function :
  ∀ (a b : ℝ), (∀ x : ℝ, f(x) = x^2 + a * (b + 1) * x + a + b) →
  ((∀ x : ℝ, f x = f (-x) → a = 0) ∧
    (¬(∀ a : ℝ, (∀ x : ℝ, f x = f (-x)) → a = 0))) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_even_function_l10_10213


namespace find_x_l10_10048

theorem find_x (x : ℝ) (h : 0.40 * x = (1/3) * x + 110) : x = 1650 :=
sorry

end find_x_l10_10048


namespace gcd_product_l10_10512

theorem gcd_product (n : ℕ) (h : n > 0) : 
  ∏ d in {d | ∃ k, d = gcd (7 * k + 6) k }, d = 36 := 
sorry

end gcd_product_l10_10512


namespace sum_four_product_l10_10330

theorem sum_four_product (n : ℕ) (h : n > 0) :
  (∑ k in Finset.range n, k * (k + 1) * (k + 2) * (k + 3)) = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) / 5 :=
by
  induction n with k hk
  . -- Base case goes here
    sorry
  . -- Inductive step goes here
    sorry

end sum_four_product_l10_10330


namespace no_greatest_and_least_l10_10776

noncomputable def isDedekindCut (M N : Set ℚ) : Prop :=
  (M ∪ N = Set.univ) ∧
  (M ∩ N = ∅) ∧
  (∀ m ∈ M, ∀ n ∈ N, m < n)

theorem no_greatest_and_least (M N : Set ℚ) (hM : ∃ x ∈ M, ∀ y ∈ M, y ≤ x) (hN : ∃ z ∈ N, ∀ w ∈ N, z ≤ w) :
  ¬ isDedekindCut M N :=
begin
  sorry
end

end no_greatest_and_least_l10_10776


namespace num_ways_to_form_rectangle_l10_10568

theorem num_ways_to_form_rectangle (n : ℕ) (h : n = 5) :
  (nat.choose n 2) * (nat.choose n 2) = 100 :=
by {
  rw h,
  exact nat.choose_five_two_mul_five_two 100
}

lemma nat.choose_five_two_mul_five_two :
  ((5.choose 2) * (5.choose 2) = 100) :=
by norm_num

end num_ways_to_form_rectangle_l10_10568


namespace total_number_of_coins_l10_10482

theorem total_number_of_coins (n : ℕ) (h : 0.01 * n + 0.05 * n + 0.10 * n + 0.25 * n + 0.50 * n = 2.73) :
  5 * n = 15 :=
by
  sorry

end total_number_of_coins_l10_10482


namespace yolanda_left_30_minutes_before_l10_10445

-- Given: Yolanda rides at a constant speed of 20 mph.
constant velocity_yolanda : ℝ := 20

-- Given: Yolanda's husband leaves 15 minutes after Yolanda, and takes 15 minutes to catch her.
constant time_after_yolanda_husband_leaves : ℝ := 0.25 -- 15 minutes is 0.25 hours

-- Given: Yolanda's husband rides at a constant speed of 40 mph.
constant velocity_husband : ℝ := 40

-- Problem: Prove Yolanda left for work 30 minutes before her husband caught up to her.
theorem yolanda_left_30_minutes_before : 
    let time_husband_to_catch : ℝ := 0.25,
        distance_husband_traveled := velocity_husband * time_husband_to_catch,
        time_yolanda_to_travel := distance_husband_traveled / velocity_yolanda in
    time_yolanda_to_travel = 0.5 :=
by
  -- time_husband_to_catch = 0.25 hours
  let time_husband_to_catch := 0.25
  -- distance_husband_traveled = 40 mph * 0.25 hours = 10 miles
  let distance_husband_traveled := velocity_husband * time_husband_to_catch
  -- time_yolanda_to_travel = 10 miles / 20 mph = 0.5 hours (30 minutes)
  let time_yolanda_to_travel := distance_husband_traveled / velocity_yolanda
  show time_yolanda_to_travel = 0.5 from 
    sorry

end yolanda_left_30_minutes_before_l10_10445


namespace max_largest_element_of_seven_numbers_l10_10067

theorem max_largest_element_of_seven_numbers (lst : List ℕ) (h_len : lst.length = 7) (h_pos : ∀ n ∈ lst, 0 < n)
  (h_median : lst.nth_le 3 (by linarith) = 4) (h_mean : (lst.sum : ℚ) / 7 = 13) :
  lst.maximum = some 82 :=
sorry

end max_largest_element_of_seven_numbers_l10_10067


namespace smallest_rel_prime_120_l10_10929

theorem smallest_rel_prime_120 : ∃ (x : ℕ), x > 1 ∧ Nat.gcd x 120 = 1 ∧ ∀ y, y > 1 ∧ Nat.gcd y 120 = 1 → x ≤ y :=
by
  use 7
  sorry

end smallest_rel_prime_120_l10_10929


namespace proof_problem1_proof_problem2_proof_problem3_l10_10040

noncomputable def problem1 (x : ℝ) : Prop :=
  9.9 + x = -18

noncomputable def solution1 : ℝ := -27.9

theorem proof_problem1 : problem1 solution1 :=
by
  unfold problem1
  simp [solution1]

noncomputable def problem2 (x : ℝ) : Prop :=
  x - 8.8 = -8.8

noncomputable def solution2 : ℝ := 0

theorem proof_problem2 : problem2 solution2 :=
by
  unfold problem2
  simp [solution2]

noncomputable def problem3 (x : ℝ) : Prop :=
  -3/4 + x = -1/4

noncomputable def solution3 : ℝ := 1/2

theorem proof_problem3 : problem3 solution3 :=
by
  unfold problem3
  simp [solution3]

end proof_problem1_proof_problem2_proof_problem3_l10_10040


namespace min_real_roots_l10_10309

theorem min_real_roots (g : Polynomial ℝ) (h_deg : g.degree = 2010)
    (h_real_coeffs : ∀ c, g.coeff c ∈ ℝ)
    (root_magnitudes : Finset ℝ := (Finset.image (λ z : ℂ, z.abs) (g.root_set ℂ)))
    (h_distinct_magnitudes : root_magnitudes.card = 1008) :
    ∃ r_roots : Finset ℝ, r_roots.card = 6 ∧ ∀ x ∈ r_roots, IsRoot g x := 
sorry

end min_real_roots_l10_10309


namespace car_speed_correct_l10_10054

noncomputable theory

def car_speed (distance_gasoline : ℝ) (distance_diesel : ℝ) (time_hours : ℝ) : ℝ :=
  (distance_gasoline + distance_diesel) / time_hours

def convert_gallons_to_liters (gallons : ℝ) : ℝ :=
  gallons * 3.78541

def convert_kilometers_to_miles (kilometers : ℝ) : ℝ :=
  kilometers / 1.60934

def travel_distance (liters_used : ℝ) (distance_per_liter : ℝ) : ℝ :=
  liters_used * distance_per_liter

theorem car_speed_correct :
  let gasoline_used_gallons := 3.9
  let diesel_used_gallons := 2.45
  let time_hours := 5.7
  let distance_per_liter_gasoline := 40
  let distance_per_liter_diesel := 55 in
  let gasoline_used_liters := convert_gallons_to_liters gasoline_used_gallons in
  let diesel_used_liters := convert_gallons_to_liters diesel_used_gallons in
  let distance_gasoline := travel_distance gasoline_used_liters distance_per_liter_gasoline in
  let distance_diesel := travel_distance diesel_used_liters distance_per_liter_diesel in
  let total_distance_km := distance_gasoline + distance_diesel in
  let total_distance_miles := convert_kilometers_to_miles total_distance_km in
  car_speed distance_gasoline distance_diesel time_hours ≈ 119.95 := sorry

end car_speed_correct_l10_10054


namespace construct_triangle_if_acute_l10_10893

noncomputable theory
open_locale classical

variables (A B C X Y Z : ℝ)
variables (triangle : ℝ → ℝ → ℝ × ℝ × ℝ)    -- Function to construct triangle vertices A, B, C

-- Defining the centers of squares
def is_center_of_square (P : ℝ) (Q : ℝ) : Prop :=
  -- Assume a function that checks if a point is the center of the square constructed outward on a given line segment 
  true  

def is_perpendicular (P Q : ℝ) : Prop :=
  -- Assume a function that checks if two segments are perpendicular
  true

def is_equal_length (P Q : ℝ) : Prop :=
  -- Assume a function that checks if two segments are equal in length
  true

def is_acute_angled (triangle : ℝ → ℝ → ℝ × ℝ × ℝ) (X Y Z : ℝ) : Prop :=
  -- Assume a function that checks if the triangle XYZ is acute-angled
  true

theorem construct_triangle_if_acute (XY_length CZ_perpendicular length_equal acute : Prop)
  (hyp_1 : is_center_of_square X Y) (hyp_2 : is_center_of_square Y Z) (hyp_3 : is_center_of_square Z X)
  (hyp_4 : is_perpendicular X Y) (hyp_5 : is_perpendicular Y Z) (hyp_6 : is_equal_length X Y)
  (hyp_7 : is_acute_angled triangle X Y Z) :
  ∃ A B C : ℝ, triangle A B C := 
begin
  sorry
end

end construct_triangle_if_acute_l10_10893


namespace pyramid_volume_l10_10486

theorem pyramid_volume (total_surface_area : ℝ) (area_ratio : ℝ) (base_area : ℝ) (slant_height : ℝ) (vertical_height : ℝ) (volume : ℝ) :
  total_surface_area = 720 ∧ area_ratio = (1 / 3) ∧ 
  base_area = 324 ∧ slant_height = 12 ∧ vertical_height = 3 * real.sqrt 7 →
  volume = 108 * real.sqrt 7 :=
by
  sorry

end pyramid_volume_l10_10486


namespace largest_multiple_of_7_less_than_neg85_l10_10423

theorem largest_multiple_of_7_less_than_neg85 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n < -85 ∧ n = -91 :=
by
  sorry

end largest_multiple_of_7_less_than_neg85_l10_10423


namespace hyperbola_has_eccentricity_l10_10249

noncomputable def hyperbola_eccentricity (b d c e : ℝ) : Prop :=
  (d = 2) → 
  (x^2 - (y^2 / b^2) = 1) → 
  (c = sqrt (1 + b^2)) → 
  (d = abs ((0 + b * c) / sqrt (1 + b^2))) → 
  (b = 2) → 
  (c = sqrt (1 + 4)) → 
  (e = c / 1) → 
  e = sqrt 5

-- Definition of the values based on the problem
def c := sqrt (1 + 4)
def e := sqrt 5

-- Statement of the proof problem
theorem hyperbola_has_eccentricity :
  hyperbola_eccentricity 2 2 c e :=
by 
  sorry

end hyperbola_has_eccentricity_l10_10249


namespace prove_rectangular_selection_l10_10565

def number_of_ways_to_choose_rectangular_region (horizontals verticals : ℕ) : ℕ :=
  (Finset.choose horizontals 2) * (Finset.choose verticals 2)

theorem prove_rectangular_selection :
  number_of_ways_to_choose_rectangular_region 5 5 = 100 :=
by
  sorry

end prove_rectangular_selection_l10_10565


namespace max_distinct_prime_factors_m_l10_10751

theorem max_distinct_prime_factors_m
  (m n : ℕ)
  (hm : 0 < m)
  (hn : 0 < n)
  (gcd_five_primes : (nat.gcd m n).factorization.support.card = 5)
  (lcm_thirty_primes : (nat.lcm m n).factorization.support.card = 30)
  (m_fewer_primes : m.factorization.support.card < n.factorization.support.card)
  : m.factorization.support.card ≤ 17 := 
sorry

end max_distinct_prime_factors_m_l10_10751


namespace num_ways_to_form_rectangle_l10_10571

theorem num_ways_to_form_rectangle (n : ℕ) (h : n = 5) :
  (nat.choose n 2) * (nat.choose n 2) = 100 :=
by {
  rw h,
  exact nat.choose_five_two_mul_five_two 100
}

lemma nat.choose_five_two_mul_five_two :
  ((5.choose 2) * (5.choose 2) = 100) :=
by norm_num

end num_ways_to_form_rectangle_l10_10571


namespace smallest_x_value_min_smallest_x_value_l10_10804

noncomputable def smallest_x_not_defined : ℝ := ( 47 - (Real.sqrt 2041) ) / 12

theorem smallest_x_value :
  ∀ x : ℝ, (6 * x^2 - 47 * x + 7 = 0) → x = smallest_x_not_defined ∨ (x = (47 + (Real.sqrt 2041)) / 12) :=
sorry

theorem min_smallest_x_value :
  smallest_x_not_defined < (47 + (Real.sqrt 2041)) / 12 :=
sorry

end smallest_x_value_min_smallest_x_value_l10_10804


namespace cone_cut_distance_l10_10463

theorem cone_cut_distance 
  (R : ℝ) 
  (h : R > 0) :
  ∃ x : ℝ, x = (R / 2) * real.sqrt (10 * (5 + real.sqrt 5)) ∧ 
  (∀ (surface_area : ℝ → ℝ),
    surface_area (2 * R) = surface_area (x) + surface_area (2 * R - x)) :=
begin
  sorry
end

end cone_cut_distance_l10_10463


namespace case1_equiv_case2_equiv_determine_case_l10_10012

theorem case1_equiv (a c x : ℝ) (hc : c ≠ 0) (hx : x ≠ 0) : 
  ((x + a) / (x + c) = a / c) ↔ (a = c) :=
by sorry

theorem case2_equiv (b d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) : 
  (b / d = b / d) :=
by sorry

theorem determine_case (a b c d x : ℝ) (hc : c ≠ 0) (hx : x ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0) :
  ¬((x + a) / (x + c) = a / c) ∧ (b / d = b / d) :=
by sorry

end case1_equiv_case2_equiv_determine_case_l10_10012


namespace average_star_rating_l10_10759

/-- Define specific constants for the problem. --/
def reviews_5_star := 6
def reviews_4_star := 7
def reviews_3_star := 4
def reviews_2_star := 1
def total_reviews := 18

/-- Calculate the total stars given the number of each type of review. --/
def total_stars : ℕ := 
  (reviews_5_star * 5) + 
  (reviews_4_star * 4) + 
  (reviews_3_star * 3) + 
  (reviews_2_star * 2)

/-- Prove that the average star rating is 4. --/
theorem average_star_rating : total_stars / total_reviews = 4 := by 
  sorry

end average_star_rating_l10_10759


namespace jonathan_needs_more_money_l10_10295

def cost_of_dictionary : ℕ := 11
def cost_of_dinosaur_book : ℕ := 19
def cost_of_childrens_cookbook : ℕ := 7
def jonathan_savings : ℕ := 8

def total_cost : ℕ := cost_of_dictionary + cost_of_dinosaur_book + cost_of_childrens_cookbook
def amount_needed : ℕ := total_cost - jonathan_savings

theorem jonathan_needs_more_money : amount_needed = 29 :=
by
  unfold cost_of_dictionary
  unfold cost_of_dinosaur_book
  unfold cost_of_childrens_cookbook
  unfold jonathan_savings
  unfold total_cost
  unfold amount_needed
  sorry

end jonathan_needs_more_money_l10_10295


namespace find_square_value_l10_10402

theorem find_square_value :
  ∃ x : ℕ, 60 + x * 5 = 500 ∧ x = 88 :=
by
  use 88
  split
  · calc
      60 + 88 * 5 = 60 + 440  := by rfl
      ...          = 500      := by rfl
  · rfl

end find_square_value_l10_10402


namespace final_bicycle_price_is_225_l10_10491

noncomputable def final_selling_price (cp_A : ℝ) (profit_A : ℝ) (profit_B : ℝ) : ℝ :=
  let sp_B := cp_A * (1 + profit_A / 100)
  let sp_C := sp_B * (1 + profit_B / 100)
  sp_C

theorem final_bicycle_price_is_225 :
  final_selling_price 114.94 35 45 = 224.99505 :=
by
  sorry

end final_bicycle_price_is_225_l10_10491


namespace peter_remaining_money_l10_10345

def initial_amount : Float := 500.0 
def sales_tax : Float := 0.05
def discount : Float := 0.10

def calculate_cost_with_tax (price_per_kilo: Float) (quantity: Float) (tax_rate: Float) : Float :=
  quantity * price_per_kilo * (1 + tax_rate)

def calculate_cost_with_discount (price_per_kilo: Float) (quantity: Float) (discount_rate: Float) : Float :=
  quantity * price_per_kilo * (1 - discount_rate)

def total_first_trip : Float :=
  calculate_cost_with_tax 2.0 6 sales_tax +
  calculate_cost_with_tax 3.0 9 sales_tax +
  calculate_cost_with_tax 4.0 5 sales_tax +
  calculate_cost_with_tax 5.0 3 sales_tax +
  calculate_cost_with_tax 3.50 2 sales_tax +
  calculate_cost_with_tax 4.25 7 sales_tax +
  calculate_cost_with_tax 6.0 4 sales_tax +
  calculate_cost_with_tax 5.50 8 sales_tax

def total_second_trip : Float :=
  calculate_cost_with_discount 1.50 2 discount +
  calculate_cost_with_discount 2.75 5 discount

def remaining_money (initial: Float) (first_trip: Float) (second_trip: Float) : Float :=
  initial - first_trip - second_trip

theorem peter_remaining_money : remaining_money initial_amount total_first_trip total_second_trip = 297.24 := 
  by
    -- Proof omitted
    sorry

end peter_remaining_money_l10_10345


namespace fare_midpoint_to_b_l10_10497

-- Define the conditions
def initial_fare : ℕ := 5
def initial_distance : ℕ := 2
def additional_fare_per_km : ℕ := 2
def total_fare : ℕ := 35
def walked_distance_meters : ℕ := 800

-- Define the correct answer
def fare_from_midpoint_to_b : ℕ := 19

-- Prove that the fare from the midpoint between A and B to B is 19 yuan
theorem fare_midpoint_to_b (y : ℝ) (h1 : 16.8 < y ∧ y ≤ 17) : 
  let half_distance := y / 2
  let total_taxi_distance := half_distance - 2
  let total_additional_fare := ⌈total_taxi_distance⌉ * additional_fare_per_km
  initial_fare + total_additional_fare = fare_from_midpoint_to_b := 
by
  sorry

end fare_midpoint_to_b_l10_10497


namespace minimum_bailing_rate_l10_10025

theorem minimum_bailing_rate (distance_to_shore : ℝ) (row_speed : ℝ) (leak_rate : ℝ) (max_water_intake : ℝ)
  (time_to_shore : ℝ := distance_to_shore / row_speed * 60) (total_water_intake : ℝ := time_to_shore * leak_rate) :
  distance_to_shore = 1.5 → row_speed = 3 → leak_rate = 10 → max_water_intake = 40 →
  ∃ (bail_rate : ℝ), bail_rate ≥ 9 :=
by
  sorry

end minimum_bailing_rate_l10_10025


namespace odd_divisors_under_50_l10_10235

theorem odd_divisors_under_50 :
  { n : ℕ | n < 50 ∧ n > 0 ∧ (∃ k : ℕ, k^2 = n) }.to_finset.card = 7 := by
  sorry

end odd_divisors_under_50_l10_10235


namespace star_vertex_angle_l10_10539

theorem star_vertex_angle (n : ℕ) (h : 3 ≤ n) : 
  let internal_angle := (n - 2) * 180 / n in
  let external_angle := 180 - internal_angle in
  let star_vertex_angle := 180 - 2 * external_angle in
  star_vertex_angle = (n - 4) * 180 / n :=
by
  have : internal_angle = (n - 2) * 180 / n := sorry
  have : external_angle = 180 - internal_angle := sorry
  have : star_vertex_angle = 180 - 2 * external_angle := sorry
  show star_vertex_angle = (n - 4) * 180 / n from sorry

end star_vertex_angle_l10_10539


namespace alexander_total_spending_l10_10501

def apples_price := 5 * 1
def oranges_price := 2 * 2
def bananas_price := 3 * 0.5
def grapes_price := 4
def total_fruits_cost := apples_price + oranges_price + bananas_price + grapes_price

def fruit_discount := 0.10 * total_fruits_cost
def total_after_fruit_discount := total_fruits_cost - fruit_discount

def overall_discount := 0.20 * total_after_fruit_discount
def total_after_overall_discount := total_after_fruit_discount - overall_discount

def sales_tax_rate := 0.08
def sales_tax := total_after_overall_discount * sales_tax_rate

def final_total := total_after_overall_discount + sales_tax

theorem alexander_total_spending : final_total = 11.28 := by
  sorry

end alexander_total_spending_l10_10501


namespace f_of_f_minus1_eq_2_l10_10989

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2
  else 3 * x - 1

-- Prove that f[f(-1)] = 2
theorem f_of_f_minus1_eq_2 : f (f (-1)) = 2 :=
by sorry

end f_of_f_minus1_eq_2_l10_10989


namespace bacteria_growth_l10_10902

-- Defining the function for bacteria growth
def bacteria_count (t : ℕ) (initial_count : ℕ) (division_time : ℕ) : ℕ :=
  initial_count * 2 ^ (t / division_time)

-- The initial conditions given in the problem
def initial_bacteria : ℕ := 1
def division_interval : ℕ := 10
def total_time : ℕ := 2 * 60

-- Stating the hypothesis and the goal
theorem bacteria_growth : bacteria_count total_time initial_bacteria division_interval = 2 ^ 12 :=
by
  -- Proof would go here
  sorry

end bacteria_growth_l10_10902


namespace team_with_3_internists_and_2_surgeons_team_includes_both_internists_and_surgeons_team_includes_at_least_one_chief_team_includes_both_chief_and_surgeons_l10_10504

-- Definitions to represent the given conditions
def internists : ℕ := 6
def surgeons : ℕ := 4
def chief_internist : ℕ := 1
def chief_surgeon : ℕ := 1

-- Statements for the problem parts
theorem team_with_3_internists_and_2_surgeons :
  (nat.choose internists 3) * (nat.choose surgeons 2) = 120 := by
sorry

theorem team_includes_both_internists_and_surgeons :
  (nat.choose (internists + surgeons) 5) - (nat.choose internists 5) - (nat.choose surgeons 5) = 246 := by
sorry

theorem team_includes_at_least_one_chief :
  (nat.choose (internists + surgeons) 5) - (nat.choose (internists - chief_internist + surgeons - chief_surgeon) 5) = 196 := by
sorry

theorem team_includes_both_chief_and_surgeons :
  let total_teams_with_at_least_one_chief := (nat.choose (internists + surgeons) 5) - (nat.choose (internists - chief_internist + surgeons - chief_surgeon) 5) in
  let teams_with_chefs_and_no_surgeons := 2 * (nat.choose internists 4) in
  let teams_with_both_chiefs (++additional_teams)
  total_teams_with_at_least_one_chief - teams_with_chefs_and_no_surgeons + ((nat.choose surgeons 3) * chief_surgeon) + (nat.choose internists 3) = 191 := by
sorry

end team_with_3_internists_and_2_surgeons_team_includes_both_internists_and_surgeons_team_includes_at_least_one_chief_team_includes_both_chief_and_surgeons_l10_10504


namespace shortest_fence_length_l10_10465

-- We define the conditions given in the problem.
def triangle_side_length : ℕ := 50
def number_of_dotted_lines : ℕ := 13

-- We need to prove that the shortest total length of the fences required to protect all the cabbage from goats equals 650 meters.
theorem shortest_fence_length : number_of_dotted_lines * triangle_side_length = 650 :=
by
  -- The proof steps are omitted as per instructions.
  sorry

end shortest_fence_length_l10_10465


namespace total_money_shared_l10_10333

-- Let us define the conditions
def ratio (a b c : ℕ) : Prop := ∃ k : ℕ, (2 * k = a) ∧ (3 * k = b) ∧ (8 * k = c)

def olivia_share := 30

-- Our goal is to prove the total amount of money shared
theorem total_money_shared (a b c : ℕ) (h_ratio : ratio a b c) (h_olivia : a = olivia_share) :
    a + b + c = 195 :=
by
  sorry

end total_money_shared_l10_10333


namespace distance_O_to_AK_is_19_sqrt_26_l10_10085

noncomputable def distance_from_point_to_line
  (O K L M N A B : Point)
  (KL MN : Line)
  (circle : O ∈ Circle)
  (h1 : K L M N are in circle)
  (h2 : is_trapezoid K L M N ∧ KL.parallel MN)
  (h3 : KL.length = 8)
  (h4 : MN.length = 2)
  (h5 : angle N K L = 45)
  (h6 : M A is a chord of circle)
  (h7 : intersection_point (M A) (K L) = B)
  (h8 : K B = 3) :
  ℝ :=
    19 / sqrt 26

theorem distance_O_to_AK_is_19_sqrt_26
  (O K L M N A B : Point)
  (KL MN : Line)
  (circle : O ∈ Circle)
  (h1 : K L M N are in circle)
  (h2 : is_trapezoid K L M N ∧ KL.parallel MN)
  (h3 : KL.length = 8)
  (h4 : MN.length = 2)
  (h5 : angle N K L = 45)
  (h6 : M A is a chord of circle)
  (h7 : intersection_point (M A) (K L) = B)
  (h8 : K B = 3) :
  distance_from_point_to_line O K L M N A B KL MN circle h1 h2 h3 h4 h5 h6 h7 h8 = 19 / sqrt 26 :=
  sorry 

end distance_O_to_AK_is_19_sqrt_26_l10_10085


namespace problem_statement_l10_10612

variable (f : ℝ → ℝ)
hypothesis h1 : ∀ x : ℝ, deriv f x > f x

theorem problem_statement : f 2 > (Real.exp 2) * f 0 := by
  sorry

end problem_statement_l10_10612


namespace increased_area_percentage_l10_10837

-- Define the problem conditions
def original_radius (r : ℝ) := r
def increased_radius (r : ℝ) := 1.5 * r
def original_area (r : ℝ) := Real.pi * r ^ 2
def increased_area (r : ℝ) := Real.pi * (1.5 * r) ^ 2

-- The proof statement
theorem increased_area_percentage (r : ℝ) : 
  (increased_area r - original_area r) / original_area r * 100 = 125 :=
by
  sorry

end increased_area_percentage_l10_10837


namespace apples_selection_probability_l10_10287

theorem apples_selection_probability :
  let total_apples := 10
  let red_apples := 5
  let green_apples := 3
  let yellow_apples := 2
  let chosen_apples := 3
  let ways_to_choose := Nat.choose total_apples chosen_apples
  let ways_to_choose_2_green := Nat.choose green_apples 2
  let ways_to_choose_1_yellow := Nat.choose yellow_apples 1
  let successful_ways := ways_to_choose_2_green * ways_to_choose_1_yellow
  let probability := (successful_ways : ℚ) / (ways_to_choose : ℚ)
  in total_apples = 10 ∧ red_apples = 5 ∧ green_apples = 3 ∧ yellow_apples = 2 ∧ chosen_apples = 3 ∧
     ways_to_choose = 120 ∧ successful_ways = 6 ∧ probability = (1 / 20 : ℚ) :=
by
  sorry

end apples_selection_probability_l10_10287


namespace arithmetic_mean_eq_one_l10_10174

theorem arithmetic_mean_eq_one 
  (x a b : ℝ) 
  (hx : x ≠ 0) 
  (hb : b ≠ 0) : 
  (1 / 2 * ((x + a + b) / x + (x - a - b) / x)) = 1 := by
  sorry

end arithmetic_mean_eq_one_l10_10174


namespace cube_diagonal_length_l10_10980

theorem cube_diagonal_length (V : ℝ) (hV : V = 36 * Real.pi) : 
  let r := ((3 * V) / (4 * Real.pi))^(1/3) in
  let s := 2 * r in
  let d := Real.sqrt (3 * s^2) in
  d = 6 * Real.sqrt 3 := by
  sorry

end cube_diagonal_length_l10_10980


namespace find_tan_A_and_sin_A_l10_10686

def triangle_ABC (A B C : Type) [EuclideanGeometry.Triangle A B C] : Prop :=
  ∃ {a b c : ℝ}, ∡ A B = 90 ∧ AC = 13 ∧ AB = 5

theorem find_tan_A_and_sin_A (A B C : Type) [EuclideanGeometry.Triangle A B C] (h : triangle_ABC A B C) :
  ∃ p q : ℚ, p = 12 / 5 ∧ q = 12 / 13 :=
by
  sorry

end find_tan_A_and_sin_A_l10_10686


namespace least_faces_combined_l10_10009

variables (a b : ℕ)
def is_fair_die (n : ℕ) : Prop := n >= 6 ∧ n > 1
def prob_sum_n (n : ℕ) (a b : ℕ) : ℚ :=
  match n with
  | 9  => 8 / (a * b)
  | 11 => 10 / (a * b)
  | _  => 0

theorem least_faces_combined :
  is_fair_die a → is_fair_die b →
  (2 / 3) * prob_sum_n 11 a b = prob_sum_n 9 a b →
  prob_sum_n 11 a b = 1 / 9 →
  a + b = 19 :=
by intro hfa hfb hprob11 hprob9
have hab : a * b = 90, from sorry,
sorry

end least_faces_combined_l10_10009


namespace correct_operation_l10_10024

theorem correct_operation : (a : ℕ) →
  (a^2 * a^3 = a^5) ∧
  (2 * a + 4 ≠ 6 * a) ∧
  ((2 * a)^2 ≠ 2 * a^2) ∧
  (a^3 / a^3 ≠ a) := sorry

end correct_operation_l10_10024


namespace tangent_parallel_x_axis_monotonically_increasing_intervals_l10_10629

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := m * x^3 + n * x^2

theorem tangent_parallel_x_axis (m n : ℝ) (h : m ≠ 0) (h_tangent : 3 * m * (2:ℝ)^2 + 2 * n * (2:ℝ) = 0) :
  n = -3 * m :=
by
  sorry

theorem monotonically_increasing_intervals (m : ℝ) (h : m ≠ 0) : 
  (∀ x : ℝ, 3 * m * x * (x - (2 : ℝ)) > 0 ↔ 
    if m > 0 then x < 0 ∨ 2 < x else 0 < x ∧ x < 2) :=
by
  sorry

end tangent_parallel_x_axis_monotonically_increasing_intervals_l10_10629


namespace absolute_sum_of_roots_l10_10165

theorem absolute_sum_of_roots (d e f n : ℤ) (h1 : d + e + f = 0) (h2 : d * e + e * f + f * d = -2023) : |d| + |e| + |f| = 98 := 
sorry

end absolute_sum_of_roots_l10_10165


namespace simplify_complex_l10_10746

def question : ℂ := (4 + 3 * complex.i)^2

def correct_answer : ℂ := 7 + 24 * complex.i

theorem simplify_complex (h : complex.i^2 = -1) : question = correct_answer := by
  sorry

end simplify_complex_l10_10746


namespace rectangle_count_l10_10558

theorem rectangle_count (h_lines v_lines : Finset ℕ) (h_card : h_lines.card = 5) (v_card : v_lines.card = 5) :
  ∃ (n : ℕ), n = (h_lines.choose 2).card * (v_lines.choose 2).card ∧ n = 100 :=
by
  sorry 

end rectangle_count_l10_10558


namespace min_value_M_l10_10389

-- Define the sequence a_n
def a (n : ℕ) : ℚ := 1 / ((2 * n + 1) * (2 * n + 3))

-- Define the sum of the first n terms of the sequence a_n
def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k, a k)

-- State the theorem
theorem min_value_M : ∀ (n : ℕ), S n < (1 / 6) := by
  sorry

end min_value_M_l10_10389


namespace percentage_downpayment_l10_10326

variables (salary yearly_savings house_cost total_savings years : ℝ)

-- Given conditions
def given_conditions : Prop :=
  salary = 150000 ∧
  yearly_savings = 0.10 * salary ∧
  house_cost = 450000 ∧
  years = 6 ∧
  total_savings = yearly_savings * years

-- Proving the percentage of house cost for downpayment
theorem percentage_downpayment (p : ℝ) :
  given_conditions →
  p = (total_savings / house_cost) * 100 :=
by
  sorry

end percentage_downpayment_l10_10326


namespace line_passes_through_vertex_of_parabola_l10_10946

theorem line_passes_through_vertex_of_parabola :
  {a : ℝ | ∃ (x : ℝ), (2 * x + a = x^2 + 2 * a^2)}.finite ∧ 
  {a : ℝ | ∃ (x : ℝ), (2 * x + a = x^2 + 2 * a^2)}.toFinset.card = 2 :=
by
  sorry

end line_passes_through_vertex_of_parabola_l10_10946


namespace no_ways_to_write_56_as_sum_of_two_primes_l10_10265

open Nat

/--
In how many ways can 56 be written as the sum of two primes?

Conditions:
1. All primes larger than 2 are odd.
2. The only even prime is 2.
3. 56 must be written as the sum of two primes.

Conclusion:
The number of ways is 0.
-/
theorem no_ways_to_write_56_as_sum_of_two_primes :
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ 56 = p + q) → False :=
by
  sorry

end no_ways_to_write_56_as_sum_of_two_primes_l10_10265


namespace dot_product_of_vectors_l10_10253

variables {V : Type*} [inner_product_space ℝ V]

theorem dot_product_of_vectors (a b : V) 
  (h1 : ∥a + b∥ = real.sqrt 2) 
  (h2 : ∥a - b∥ = real.sqrt 6) : 
  ⟪a, b⟫ = -1 := 
begin
  sorry
end

end dot_product_of_vectors_l10_10253


namespace value_of_m_satisfies_condition_l10_10239

-- Define the condition for m
def is_proper_linear_equation (m : ℝ) := (m - 1) ≠ 0

-- Provide the value of m that satisfies the condition
theorem value_of_m_satisfies_condition : is_proper_linear_equation 2 :=
by {
  have h : 2 - 1 = 1,
  { norm_num, },
  show (2 - 1) ≠ 0,
  { rw h, norm_num, }
}

end value_of_m_satisfies_condition_l10_10239


namespace compute_c_plus_d_l10_10306

variable {c d : ℝ}

-- Define the given polynomial equations
def poly_c (c : ℝ) := c^3 - 21*c^2 + 28*c - 70
def poly_d (d : ℝ) := 10*d^3 - 75*d^2 - 350*d + 3225

theorem compute_c_plus_d (hc : poly_c c = 0) (hd : poly_d d = 0) : c + d = 21 / 2 := sorry

end compute_c_plus_d_l10_10306


namespace triangle_angles_and_centers_l10_10086

theorem triangle_angles_and_centers :
  ∀ (A B C : Type) [triangle A B C]
    (∠A : angle A = 58) (∠B : angle B = 59)
    (I : incenter A B C)
    (O : circumcenter A B C),
  180 - (∠A + ∠B) = 63 :=
by sorry

end triangle_angles_and_centers_l10_10086


namespace calculate_max_income_l10_10675

variables 
  (total_lunch_pasta : ℕ) (total_lunch_chicken : ℕ) (total_lunch_fish : ℕ)
  (sold_lunch_pasta : ℕ) (sold_lunch_chicken : ℕ) (sold_lunch_fish : ℕ)
  (dinner_pasta : ℕ) (dinner_chicken : ℕ) (dinner_fish : ℕ)
  (price_pasta : ℝ) (price_chicken : ℝ) (price_fish : ℝ)
  (discount : ℝ)
  (max_income : ℝ)

def unsold_lunch_pasta := total_lunch_pasta - sold_lunch_pasta
def unsold_lunch_chicken := total_lunch_chicken - sold_lunch_chicken
def unsold_lunch_fish := total_lunch_fish - sold_lunch_fish

def discounted_price (price : ℝ) := price * (1 - discount)

def income_lunch (sold : ℕ) (price : ℝ) := sold * price
def income_dinner (fresh : ℕ) (price : ℝ) := fresh * price
def income_unsold (unsold : ℕ) (price : ℝ) := unsold * discounted_price price

theorem calculate_max_income 
  (h_pasta_total : total_lunch_pasta = 8) (h_chicken_total : total_lunch_chicken = 5) (h_fish_total : total_lunch_fish = 4)
  (h_pasta_sold : sold_lunch_pasta = 6) (h_chicken_sold : sold_lunch_chicken = 3) (h_fish_sold : sold_lunch_fish = 3)
  (h_dinner_pasta : dinner_pasta = 2) (h_dinner_chicken : dinner_chicken = 2) (h_dinner_fish : dinner_fish = 1)
  (h_price_pasta: price_pasta = 12) (h_price_chicken: price_chicken = 15) (h_price_fish: price_fish = 18)
  (h_discount: discount = 0.10) 
  : max_income = 136.80 :=
  sorry

end calculate_max_income_l10_10675


namespace intersection_of_sets_l10_10718

theorem intersection_of_sets (A B : set ℝ) (hA : A = {x | -1 ≤ x ∧ x ≤ 2}) (hB : B = {x | 0 ≤ x ∧ x ≤ 4}) :
  A ∩ B = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_of_sets_l10_10718


namespace zoe_average_mpg_is_25_4_l10_10027

def initial_odometer := 33570
def first_refill_odometer := 33960
def second_refill_odometer := 34585
def first_refill_gallons := 15
def second_refill_gallons := 25

theorem zoe_average_mpg_is_25_4 :
  let total_distance := second_refill_odometer - initial_odometer in
  let total_gallons := first_refill_gallons + second_refill_gallons in
  let average_mpg := total_distance / total_gallons in
  average_mpg = 25.4 :=
by
  sorry

end zoe_average_mpg_is_25_4_l10_10027


namespace minimum_stamps_to_make_47_cents_l10_10870

theorem minimum_stamps_to_make_47_cents (c f : ℕ) (h : 5 * c + 7 * f = 47) : c + f = 7 :=
sorry

end minimum_stamps_to_make_47_cents_l10_10870


namespace dot_product_magnitude_diff_range_f_l10_10228

open Real

variables (x : ℝ) (hx : x ∈ Icc (π / 6) (2 * π / 3))
def a : ℝ × ℝ := (cos (3 * x / 2), sin (3 * x / 2))
def b : ℝ × ℝ := (cos (x / 2), -sin (x / 2))

theorem dot_product : (a x).1 * (b x).1 + (a x).2 * (b x).2 = cos (2 * x) := sorry

theorem magnitude_diff : sqrt ((a x).1 - (b x).1)^2 + ((a x).2 - (b x).2)^2 = 2 * sin x := sorry

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - sqrt ((a x).1 - (b x).1)^2 + ((a x).2 - (b x).2)^2

theorem range_f : set.range (f ) = set.Icc (-3) (-1 / 2) := sorry

end dot_product_magnitude_diff_range_f_l10_10228


namespace largest_n_with_triangle_property_l10_10076

theorem largest_n_with_triangle_property :
  ∃ n, n = 258 ∧
  (∀ (S : Finset ℕ), 10 ≤ S.card → 
   (∀ a b c ∈ S, a + b > c ∧ a + c > b ∧ b + c > a)) :=
by
  sorry

end largest_n_with_triangle_property_l10_10076


namespace find_m_n_compute_R_squared_l10_10457

-- Define data points
def x_values : List ℝ := [3, 4, 5, 6, 7]
def y_values : List ℝ := [0.57, 0.53, 0.44, 0.36, 0.30]

-- Define mean of x and y
def mean (l : List ℝ) : ℝ := l.sum / l.length
def x_bar : ℝ := mean x_values
def y_bar : ℝ := mean y_values

-- Given linear regression equation
def a_hat : ℝ := y_bar + 0.07 * x_bar
def y_hat (x : ℝ) : ℝ := -0.07 * x + a_hat

-- Residual definition
def residuals : List ℝ := List.zipWith (λ y y_hat => y - y_hat) y_values (x_values.map y_hat)
def e_3 : ℝ := residuals.getD 2 0
def e_4 : ℝ := residuals.getD 3 0

-- Given condition for sum of squares
def sum_of_squares_total : ℝ := 0.051

-- Compute sum of squares of residuals
def sum_of_squares_residuals : ℝ := residuals.sumBy (λ e => e * e)

-- Correlation coefficient R^2
def R_squared : ℝ := 1 - (sum_of_squares_residuals / sum_of_squares_total)

-- Proof statements
theorem find_m_n : e_3 = 0 ∧ e_4 = -0.01 := by
  sorry

theorem compute_R_squared : R_squared ≈ 0.99 := by
  sorry

end find_m_n_compute_R_squared_l10_10457


namespace number_of_students_l10_10449

theorem number_of_students 
  (initial_students : ℕ) 
  (students_left : ℕ) 
  (students_added : ℕ)
  (h_initial : initial_students = 8) 
  (h_left : students_left = 5) 
  (h_added : students_added = 8) : 
  initial_students - students_left + students_added = 11 :=
by
  rw [h_initial, h_left, h_added]
  exact (8 - 5 + 8 : ℕ)
  exact 11
  sorry  -- complete this with explicit calculation, if necessary

end number_of_students_l10_10449


namespace squares_form_acute_triangle_l10_10081

theorem squares_form_acute_triangle (a b c x y z d : ℝ)
    (h_triangle : ∀ x y z : ℝ, (x > 0 ∧ y > 0 ∧ z > 0) → (x + y > z) ∧ (x + z > y) ∧ (y + z > x))
    (h_acute : ∀ x y z : ℝ, (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2))
    (h_inscribed_squares : x = a ^ 2 * b * c / (d * a + b * c) ∧
                           y = b ^ 2 * a * c / (d * b + a * c) ∧
                           z = c ^ 2 * a * b / (d * c + a * b)) :
    (x + y > z) ∧ (x + z > y) ∧ (y + z > x) ∧
    (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2) :=
sorry

end squares_form_acute_triangle_l10_10081


namespace symmetric_point_first_quadrant_l10_10599

theorem symmetric_point_first_quadrant (a : ℝ) :
  let P := (2 * a + 1, a - 1)
  in (-2 * a - 1 > 0) ∧ (-a + 1 > 0) → a < -1 / 2 :=
by
  intro P hP
  sorry

end symmetric_point_first_quadrant_l10_10599


namespace exponential_inequality_l10_10717

theorem exponential_inequality (a b c : ℝ) (h : a^2 + 2 * b^2 + 3 * c^2 = 3 / 2) : 
  3^a + 9^b + 27^c ≥ 1 :=
sorry

end exponential_inequality_l10_10717


namespace number_of_subsets_l10_10720

open Set

-- Define the sets A and B
def A : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}
def B : Set ℕ := {1, 2, 3, 4}

-- Define the problem:
theorem number_of_subsets (C : Set ℕ) (hC : C ⊆ A) :
  C ∩ B ≠ ∅ → (Finset.card { C | C ⊆ A ∧ C ∩ B ≠ ∅ } = 960) := by
  sorry

end number_of_subsets_l10_10720


namespace num_ways_to_form_rectangle_l10_10569

theorem num_ways_to_form_rectangle (n : ℕ) (h : n = 5) :
  (nat.choose n 2) * (nat.choose n 2) = 100 :=
by {
  rw h,
  exact nat.choose_five_two_mul_five_two 100
}

lemma nat.choose_five_two_mul_five_two :
  ((5.choose 2) * (5.choose 2) = 100) :=
by norm_num

end num_ways_to_form_rectangle_l10_10569


namespace distinct_constructions_l10_10059

-- Definitions for the size of the cube and colors of the units
def cube_size : ℕ := 2
def white_units : ℕ := 4
def blue_units : ℕ := 4

-- Definition of the number of distinct ways to construct the cube
def distinct_ways_to_construct_cube : ℕ := 7

-- Theorem stating the number of distinct ways to construct the cube
theorem distinct_constructions (cs : ℕ) (w_units : ℕ) (b_units : ℕ) :
  cs = cube_size → w_units = white_units → b_units = blue_units → distinct_ways_to_construct_cube = 7 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end distinct_constructions_l10_10059


namespace part1_monotonic_increase_interval_part2_cos_alpha_l10_10211

noncomputable def f (x : ℝ) : ℝ :=
  (1/2) * ((Real.sin x + Real.cos x) * (Real.sin x - Real.cos x)) + (Real.sqrt 3) * (Real.sin x * Real.cos x)

theorem part1_monotonic_increase_interval (k : ℤ) :
  ∀ x, x ∈ Set.Icc (-Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi) → 
  MonotoneOn f (Set.Icc (-Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi)) := sorry

theorem part2_cos_alpha (α : ℝ) :
  f((α / 2) + (Real.pi / 4)) = Real.sqrt 3 / 3 ∧ -Real.pi / 2 < α ∧ α < 0 →
  Real.cos α = (3 + Real.sqrt 6) / 6 := sorry

end part1_monotonic_increase_interval_part2_cos_alpha_l10_10211


namespace complement_union_complement_intersection_l10_10829

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem complement_union (A B : Set ℝ) :
  (A ∪ B)ᶜ = { x : ℝ | x ≤ 2 ∨ x ≥ 10 } :=
by
  sorry

theorem complement_intersection (A B : Set ℝ) :
  (Aᶜ ∩ B) = { x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } :=
by
  sorry

end complement_union_complement_intersection_l10_10829


namespace maximum_prism_volume_l10_10848

noncomputable def cone_base_radius : ℝ := 2
noncomputable def slant_height : ℝ := 4
noncomputable def cone_height : ℝ := 2 * Real.sqrt 3
noncomputable def volume_function (a : ℝ) : ℝ := -Real.sqrt(6) / 2 * a^3 + 2 * Real.sqrt(3) * a^2

theorem maximum_prism_volume :
  ∃ (a : ℝ), 0 < a ∧ a < 2 ∧ volume_function a = 64 * Real.sqrt 3 / 27 :=
sorry

end maximum_prism_volume_l10_10848


namespace boat_speed_in_still_water_l10_10832

theorem boat_speed_in_still_water
  (W : ℝ) (T : ℝ) (D : ℝ)
  (hW : W = 400) (hT : T = 50) (hD : D = 300) :
  let Vb := W / T
  let Vc := D / T
  let Vs := Real.sqrt (Vb^2 + Vc^2)
  in Vs = 10 := by
  sorry

end boat_speed_in_still_water_l10_10832


namespace value_of_k_l10_10007

theorem value_of_k (k : ℝ) : 
  (∃ P Q R : ℝ × ℝ, P = (5, 12) ∧ Q = (0, k) ∧ dist (0, 0) P = dist (0, 0) Q + 5) → 
  k = 8 := 
by
  sorry

end value_of_k_l10_10007


namespace log_sqrt_five_l10_10904

/-- Given the logarithm identity and basic property, prove the evaluation of log_5(√5). -/
theorem log_sqrt_five :
  (∀ (b : ℝ) (a x : ℝ), x ≥ 0 → log b (a ^ x) = x * log b a) →
  log 5 5 = 1 →
  log 5 (sqrt 5) = 1 / 2 :=
begin
  intros log_identity log_base,
  sorry
end

end log_sqrt_five_l10_10904


namespace tiles_finite_initial_segment_l10_10363

theorem tiles_finite_initial_segment (S : ℕ → Prop) (hTiling : ∀ n : ℕ, ∃ m : ℕ, m ≥ n ∧ S m) :
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → S n :=
by
  sorry

end tiles_finite_initial_segment_l10_10363


namespace count_valid_sets_l10_10139

theorem count_valid_sets : 
  {A : Set ℕ // {1} ⊆ A ∧ A ⊂ {1, 2, 3}}.toFinset.card = 3 := 
sorry

end count_valid_sets_l10_10139


namespace range_of_x_l10_10988

def f (x : ℝ) : ℝ := abs (x - 2)

theorem range_of_x (a b x : ℝ) (a_nonzero : a ≠ 0) (ab_real : a ∈ Set.univ ∧ b ∈ Set.univ) : 
  (|a + b| + |a - b| ≥ |a| • f x) ↔ (0 ≤ x ∧ x ≤ 4) :=
sorry

end range_of_x_l10_10988


namespace find_cost_price_l10_10095

def cost_price (SP : ℝ) (profit_percentage : ℝ) : ℝ := 
  SP / (1 + profit_percentage)

theorem find_cost_price (SP : ℝ) (profit_percentage : ℝ) (CP : ℝ) 
  (h1 : SP = 364) 
  (h2 : profit_percentage = 0.30) 
  (h3 : CP = cost_price SP profit_percentage) : 
  CP = 280 := sorry

end find_cost_price_l10_10095


namespace find_n_l10_10204

noncomputable def parabola_intersections (n : ℝ) :=
  let A := (-1 - Real.sqrt (n + 1), 0),
      B := (-1 + Real.sqrt (n + 1), 0),
      C := (1 - Real.sqrt (n + 1), 0),
      D := (1 + Real.sqrt (n + 1), 0)
  in (A, B, C, D)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem find_n: ∃ n : ℝ, n > 0 ∧
  let (A, B, C, D) := parabola_intersections n
  in distance A D = 2 * distance B C ∧ n = 8 :=
by
  sorry

end find_n_l10_10204


namespace smallest_positive_solution_tan_sec_5x_l10_10140

theorem smallest_positive_solution_tan_sec_5x:
  ∃ x : ℝ, 0 < x ∧ tan (4 * x) + tan (5 * x) = sec (5 * x) ∧ x = π / 18 := 
sorry

end smallest_positive_solution_tan_sec_5x_l10_10140


namespace minimum_distance_to_line_l10_10382

-- Given a line equation and defining lattice points as points with integer coordinates
def is_lattice_point (p : ℤ × ℤ) := true

-- Define the line equation in standard form
def line_in_standard_form (x y : ℤ) : Prop := 25 * x - 15 * y + 12 = 0

-- Define the distance formula according to the distance from a point to a line
def distance_to_line (p : ℤ × ℤ) : ℝ := 
  let (x0, y0) := p in |(25 * x0 - 15 * y0 + 12 : ℤ)|
  / (5 * Real.sqrt 34)

-- State the main theorem
theorem minimum_distance_to_line {p : ℤ × ℤ} (hp : is_lattice_point p) : distance_to_line p = Real.sqrt 34 / 85 :=
  sorry

end minimum_distance_to_line_l10_10382


namespace find_x_l10_10158

theorem find_x (x : ℝ) (h : sqrt (5 * x + 9) = 11) : x = 112 / 5 :=
sorry

end find_x_l10_10158


namespace not_like_terms_D_l10_10442

def like_terms (x y : Expr) : Prop :=
  match x, y with
  | Expr.const n1 _, Expr.const n2 _ => n1 = n2
  | Expr.mul (Expr.var v1) (Expr.var v2), Expr.mul (Expr.var v3) (Expr.var v4) =>
      v1 = v3 ∧ v2 = v4
  | Expr.mul (Expr.const c1) (Expr.mul (Expr.var v1) (Expr.var v2)), Expr.mul (Expr.const c2) (Expr.mul (Expr.var v3) (Expr.var v4)) =>
      c1 = c2 ∧ v1 = v3 ∧ v2 = v4
  | _, _ => false

theorem not_like_terms_D : ¬like_terms (Expr.mul (Expr.pow (Expr.var 0 "a") 2) (Expr.pow (Expr.var 1 "b") 3))
                                    (Expr.mul (Expr.pow (Expr.var 0 "a") 3) (Expr.pow (Expr.var 1 "b") 2)) :=
by sorry

end not_like_terms_D_l10_10442


namespace limit_example_l10_10451

theorem limit_example : 
  Real.lim (λ n : ℝ, (2 * n ^ 2 + 5) / (n ^ 2 - 3 * n)) (Filter.atTop) = 2 :=
by
  sorry

end limit_example_l10_10451


namespace greatest_five_digit_product_72_sum_digits_l10_10699

theorem greatest_five_digit_product_72_sum_digits :
  ∃ M : ℕ, (9999 < M ∧ M < 100000) ∧ (∏ d in (M.to_digits 10), d = 72) ∧ (M.to_digits 10).sum = 20 :=
begin
  sorry
end

end greatest_five_digit_product_72_sum_digits_l10_10699


namespace total_pencils_l10_10521

theorem total_pencils (pencils_per_box : ℕ) (friends : ℕ) (total_pencils : ℕ) : 
  pencils_per_box = 7 ∧ friends = 5 → total_pencils = pencils_per_box + friends * pencils_per_box → total_pencils = 42 :=
by
  intros h1 h2
  sorry

end total_pencils_l10_10521


namespace basketball_game_free_throws_l10_10258

theorem basketball_game_free_throws 
  (x : ℤ)  -- number of 3-point shots
  (t : ℤ)  -- total shots (inside 2-point shots + 3-point shots + free throws)
  (p : ℤ)  -- total points
  (h1 : t = 32)  -- total number of shots made is 32
  (h2 : p = 65)  -- total points scored is 65
  (h3 : ∃ x : ℤ, (4 * x + 3))  -- number of 2-point shots is 3 more than 4 times the number of 3-point shots
  : 1 * (t - x - (4 * x + 3)) = 4 :=  -- the number of free throws is 4
by 
  sorry

end basketball_game_free_throws_l10_10258


namespace Will_first_load_clothes_l10_10026

theorem Will_first_load_clothes : 
  ∀ (total_pieces rest_per_load loads rest_clothes first_load_pieces : ℕ),
  total_pieces = 59 →
  rest_per_load = 3 →
  loads = 9 →
  rest_clothes = rest_per_load * loads →
  first_load_pieces = total_pieces - rest_clothes →
  first_load_pieces = 32 :=
by
  intros total_pieces rest_per_load loads rest_clothes first_load_pieces
  assume h1 : total_pieces = 59
  assume h2 : rest_per_load = 3
  assume h3 : loads = 9
  assume h4 : rest_clothes = rest_per_load * loads
  assume h5 : first_load_pieces = total_pieces - rest_clothes
  sorry

end Will_first_load_clothes_l10_10026


namespace polygon_area_odd_l10_10177

-- Define the given polygon with integer vertices, edges parallel to axes, and odd edge lengths
def polygon_P : Type := 
  { points : List (ℤ × ℤ) // points.length = 101 ∧
    (∀ i < 100, (points.nth_le i i.2).fst = (points.nth_le (i + 1) ((nat.lt_succ_self _).trans_lt (nat.lt_of_lt_of_le i.property (nat.le_of_lt_succ points.length)))).fst ∨
                (points.nth_le (i + 1) ((nat.lt_succ_self _).trans_lt (nat.lt_of_lt_of_le i.property (nat.le_of_lt_succ points.length)))).snd = (points.nth_le (i)).snd) ∧
    (∀ i < 100, nat.gcd ((points.nth_le (i) i.2).fst - (points.nth_le (i + 1) ((nat.lt_succ_self _).trans_lt (nat.lt_of_lt_of_le i.property (nat.le_of_lt_succ points.length)))).fst) 2 ≠ 0 ∧
                nat.gcd ((points.nth_le (i) i.2).snd - (points.nth_le (i + 1) ((nat.lt_succ_self _).trans_lt (nat.lt_of_lt_of_le i.property (nat.le_of_lt_succ points.length)))).snd) 2 ≠ 0) }

-- Prove the area of the polygon is odd
theorem polygon_area_odd (P : polygon_P) : 
  ∃ S : ℤ, (area_of_polygon P = S) ∧ S % 2 = 1 := sorry

-- The function to calculate the area could be expressed using the given vertices
noncomputable def area_of_polygon (P : polygon_P) : ℤ :=
  1 / 2 * (P.points.enum_pairs.map 
    (λ ((i, (x₁, y₁)), (_, (x₂, y₂))), x₂ * y₁ - x₁ * y₂)).sum

end polygon_area_odd_l10_10177


namespace atomic_weight_of_element_l10_10459

theorem atomic_weight_of_element (molecular_weight_oxide : ℕ) (atomic_weight_calcium : ℕ) 
  (h_molecular: molecular_weight_oxide = 56) (h_calcium: atomic_weight_calcium = 40) : 
  (molecular_weight_oxide - atomic_weight_calcium) = 16 :=
by
  rw [h_molecular, h_calcium]
  norm_num

end atomic_weight_of_element_l10_10459


namespace living_room_area_is_60_l10_10831

-- Define the conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width
def coverage_fraction : ℝ := 0.60

-- Define the target area of the living room floor
def target_living_room_area (A : ℝ) : Prop :=
  coverage_fraction * A = carpet_area

-- State the Theorem
theorem living_room_area_is_60 (A : ℝ) (h : target_living_room_area A) : A = 60 := by
  -- Proof omitted
  sorry

end living_room_area_is_60_l10_10831


namespace mrs_sheridan_initial_cats_l10_10327

def cats_initial (cats_given_away : ℕ) (cats_left : ℕ) : ℕ :=
  cats_given_away + cats_left

theorem mrs_sheridan_initial_cats : cats_initial 14 3 = 17 :=
by
  sorry

end mrs_sheridan_initial_cats_l10_10327


namespace triangle_similarity_condition_l10_10255

-- Define the given conditions for the triangle
variables {A B C D : Type}
variables [HasAngle A B C]
variables (AB AC BD BC CD : ℝ)

-- Assume specific values for the angles and sides
variables (angle_A : ∠A = 36)
variables (side_AB_eq_side_AC : AB = AC)

-- Statement of the proof problem
theorem triangle_similarity_condition :
  (side_AB_eq_side_AC : AB = AC) →
  (angle_A : ∠A = 36) →
  (AB / BD = CD / BC) :=
sorry

end triangle_similarity_condition_l10_10255


namespace number_of_correct_conclusions_is_4_l10_10118

noncomputable theory

open Real

def correct_conclusion_1 : Prop := ∀ (x : ℝ), deriv (λ y, cos y) x = -sin x
def correct_conclusion_2 : Prop := ∀ (x ≠ 0 : ℝ), deriv (λ y, -1 / sqrt y) x = 1 / (2 * x * sqrt x)
def correct_conclusion_3 : Prop := ∀ (x > 0 : ℝ), deriv (λ y, 1 / (y ^ 2)) 3 = -2 / 27
def correct_conclusion_4 : Prop := deriv (λ y, (3 : ℝ)) = 0

def all_conclusions_correct : Prop := correct_conclusion_1 ∧ correct_conclusion_2 ∧ correct_conclusion_3 ∧ correct_conclusion_4

theorem number_of_correct_conclusions_is_4 : all_conclusions_correct → 4 = 4 := by
  sorry

end number_of_correct_conclusions_is_4_l10_10118


namespace min_value_a_b_l10_10225

theorem min_value_a_b (x y a b : ℝ) (h1 : 2 * x - y + 2 ≥ 0) (h2 : 8 * x - y - 4 ≤ 0) 
  (h3 : x ≥ 0) (h4 : y ≥ 0) (h5 : a > 0) (h6 : b > 0) (h7 : a * x + y = 8) : 
  a + b ≥ 4 :=
sorry

end min_value_a_b_l10_10225


namespace max_candies_in_26_minutes_l10_10397

theorem max_candies_in_26_minutes :
  ∃ (candies_eaten : ℕ), candies_eaten = 325 ∧ 
  (∃ (initial_nums : list ℕ) 
      (steps : ℕ) 
      (rule : ℕ → ℕ → ℕ → bool), 
      initial_nums.length = 26 ∧ 
      (∀ t, t < 26 → ∀ (x y : ℕ), x ≠ y → 
        (x ∈ initial_nums ∧ y ∈ initial_nums → rule t x y)) →
      candies_eaten = Σ' t < 26, Σ' x y : ℕ, x ≠ y ∧ rule t x y → x * y) :=
sorry

end max_candies_in_26_minutes_l10_10397


namespace find_value_of_x_l10_10828

theorem find_value_of_x (w : ℕ) (x y z : ℕ) (h₁ : x = y / 3) (h₂ : y = z / 6) (h₃ : z = 2 * w) (hw : w = 45) : x = 5 :=
by
  sorry

end find_value_of_x_l10_10828


namespace sum_of_tangents_l10_10526

noncomputable def function_f (x : ℝ) : ℝ :=
  max (max (4 * x + 20) (-x + 2)) (5 * x - 3)

theorem sum_of_tangents (q : ℝ → ℝ) (a b c : ℝ) (h1 : ∀ x, q x - (4 * x + 20) = q x - function_f x)
  (h2 : ∀ x, q x - (-x + 2) = q x - function_f x)
  (h3 : ∀ x, q x - (5 * x - 3) = q x - function_f x) :
  a + b + c = -83 / 10 :=
sorry

end sum_of_tangents_l10_10526


namespace polygon_with_one_diagonal_has_four_edges_l10_10002

theorem polygon_with_one_diagonal_has_four_edges (P : Type) [fintype P] [decidable_eq P] 
  (h : ∀ v : P, 1 = finset.card {d ∈ finset.univ.filter (λ w, ¬adjacent w v)}) : 
  fintype.card (finset.univ) = 4 :=
sorry

end polygon_with_one_diagonal_has_four_edges_l10_10002


namespace sum_divisors_121_l10_10805

theorem sum_divisors_121 : (∑ d in {1, 11, 121}, d) = 133 := by
  have h : 121 = 11 ^ 2 := by linarith
  have divisors : Finset ℕ := {1, 11, 121}
  have H : ∀ d ∈ divisors, d ∣ 121 := by
    intros d hd
    finset_cases hd <;> simp [h]
  have all_divisors : ∀ d, d ∣ 121 → d ∈ divisors := by
    intro d hd
    finset_cases hd <;> simp
  have sum_eq : (∑ d in divisors, d) = 1 + 11 + 121 := by
    finset_cases hd <;> simp
  calc
  (∑ d in {1, 11, 121}, d) = 1 + 11 + 121 := by simp [sum_eq]
                        ... = 133 := by norm_num
  sorry

end sum_divisors_121_l10_10805


namespace height_of_triangle_l10_10369

theorem height_of_triangle (base height area : ℝ) (h1 : base = 6) (h2 : area = 24) (h3 : area = 1 / 2 * base * height) : height = 8 :=
by sorry

end height_of_triangle_l10_10369


namespace chess_tournament_l10_10676

theorem chess_tournament (d : ℕ) (h1 : ∀ x : ℕ, d = 1) : 
  let boys := 5 * d in
  let total_players := d + boys in
  ∀ (x : ℕ), 2 * x + x = 6 * d * (6 * d - 1) → total_players = 6 :=
by {
  intros,
  let games_played := 2 * (total_players * (total_players - 1)) / 2,
  let total_points := games_played,
  let boys_points := 2 * x,
  let girls_points := x,
  have h2 : total_points = boys_points + girls_points := by sorry,
  have h3 : boys_points = 2 * girls_points := by sorry,
  have h4 : 3 * girls_points = total_points := by sorry,
  have h5 : girls_points = 2 * d * (6 * d - 1) := by sorry,
  have h6 : d^2 + 9 * d ≤ 2 * d * (6 * d - 1) := by sorry,
  have h7 : d ≤ 1 := by linarith,
  have h8 : d = 1 := by sorry,
  rw h8 at *,
  sorry
}

end chess_tournament_l10_10676


namespace blades_indicate_lines_move_to_form_plane_l10_10680

theorem blades_indicate_lines_move_to_form_plane (H : "The rapidly spinning blades of a fan appear as a complete plane") : "lines move to form a plane" :=
sorry

end blades_indicate_lines_move_to_form_plane_l10_10680


namespace probability_of_even_product_l10_10534

def spinnerA := {2, 4, 5, 7, 9}
def spinnerB := {1, 2, 3, 4, 5, 6}
def prob_even (s₁ s₂ : finset ℕ) : ℚ :=
  1 - ((finset.filter (λ x : ℕ, x % 2 = 1) s₁).card * (finset.filter (λ y : ℕ, y % 2 = 1) s₂).card : ℚ) / 
        (s₁.card * s₂.card)

theorem probability_of_even_product :
  prob_even spinnerA spinnerB = 7 / 10 :=
by
  sorry

end probability_of_even_product_l10_10534


namespace calories_in_lemonade_l10_10729

theorem calories_in_lemonade (grams_lemon_juice grams_sugar grams_water : ℕ)
    (calories_per_100g_lemon_juice calories_per_100g_sugar calories_per_100g_water : ℕ)
    (grams_lemon_juice = 150)
    (grams_sugar = 200)
    (grams_water = 300)
    (calories_per_100g_lemon_juice = 30)
    (calories_per_100g_sugar = 400)
    (calories_per_100g_water = 0)
    : (calories_in_250g_lemonade : ℕ) = 325 :=
by
  sorry

end calories_in_lemonade_l10_10729


namespace largest_multiple_of_7_smaller_than_neg_85_l10_10427

theorem largest_multiple_of_7_smaller_than_neg_85 :
  ∃ k : ℤ, 7 * k < -85 ∧ (∀ m : ℤ, 7 * m < -85 → 7 * m ≤ 7 * k) ∧ 7 * k = -91 :=
by
  simp only [exists_prop, and.assoc],
  sorry

end largest_multiple_of_7_smaller_than_neg_85_l10_10427


namespace find_m_and_n_l10_10221

theorem find_m_and_n (x y m n : ℝ) 
  (h1 : 5 * x - 2 * y = 3) 
  (h2 : m * x + 5 * y = 4) 
  (h3 : x - 4 * y = -3) 
  (h4 : 5 * x + n * y = 1) :
  m = -1 ∧ n = -4 :=
by
  sorry

end find_m_and_n_l10_10221


namespace pentagon_shaded_area_l10_10735

-- Define a regular pentagon structure
structure Pentagon :=
(center : Point)
(vertices : Fin 5 → Point)
(midpoints : Fin 5 → Point) -- midpoint of each side

-- Define function to calculate area
def area (p : Pentagon) : ℝ := sorry -- define pentagon area calculation

def shaded_area_fraction (p : Pentagon) : ℝ :=
let total_area := area p in
let triangle_area := total_area / 5 in
let shaded_area := total_area - 2 * (triangle_area / 2) in
shaded_area / total_area

theorem pentagon_shaded_area (p : Pentagon) (h1 : ∀ i, dist p.center (p.vertices i) = dist p.center (p.vertices 0)) 
                             (h2 : ∀ i, midpoint (p.vertices i) (p.vertices ((i+1) % 5)) = p.midpoints i) :
  shaded_area_fraction p = 4 / 5 :=
by sorry

end pentagon_shaded_area_l10_10735


namespace remaining_slices_l10_10353

def initial_slices : (nat × nat) := (16, 12)

def after_friday : (nat × nat) := (14, 10)

def after_saturday_morning : (nat × nat) :=
    let pies := 14 * 3 / 4 in (pies, 10)

def after_saturday_afternoon : (nat × nat) :=
    let cakes := 10 * 7 / 10 in (after_saturday_morning.1, cakes)

def after_sunday_morning : (nat × nat) :=
    let pies := after_saturday_morning.1 - 4 in (pies, after_saturday_afternoon.2)

def after_sunday_afternoon : (nat × nat) :=
    let cakes := after_saturday_afternoon.2 * 4 / 5 in (after_sunday_morning.1, cakes)

def after_sunday_evening : (nat × nat) :=
    let pies := after_sunday_afternoon.1 - 2 in
    let cakes := after_sunday_afternoon.2 - 2 in (pies, cakes)

theorem remaining_slices :
    after_sunday_evening = (4, 3) :=
    sorry

end remaining_slices_l10_10353


namespace digit_A_value_l10_10574

theorem digit_A_value :
  ∃ (A : ℕ), A < 10 ∧ (45 % A = 0) ∧ (172 * 10 + A * 10 + 6) % 8 = 0 ∧
    ∀ (B : ℕ), B < 10 ∧ (45 % B = 0) ∧ (172 * 10 + B * 10 + 6) % 8 = 0 → B = A := sorry

end digit_A_value_l10_10574


namespace quadratic_distinct_roots_l10_10578

theorem quadratic_distinct_roots (m : ℝ) : (x^2 - 6*x + m = 0) ∧ (9 > m) → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x^2 - 6*x + m).has_root x1 ∧ (x^2 - 6*x + m).has_root x2) :=
by
  intros h
  sorry

end quadratic_distinct_roots_l10_10578
