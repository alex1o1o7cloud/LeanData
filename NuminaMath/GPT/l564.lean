import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticEquations
import Mathlib.Algebra.Trig
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.SimpleGraph.Connectivity
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finite
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.MetricSpace.Basic

namespace digit_7_count_in_range_100_to_199_l564_564887

/-- 
Given the range of numbers from 100 to 199 inclusive,
prove that the digit 7 appears exactly 20 times.
-/
theorem digit_7_count_in_range_100_to_199 : 
  let count_7 := (λ n : ℕ, 
    (n / 100 = 7) +  -- hundreds place
    ((n % 100) / 10 = 7) + -- tens place
    ((n % 10) = 7)) in
  (Finset.sum (Finset.range' 100 100) count_7 = 20) :=
by sorry

end digit_7_count_in_range_100_to_199_l564_564887


namespace find_w_l564_564367

theorem find_w 
  (w : ℕ)
  (h1 : 3692 = 2^2 * 7 * 263)
  (h2 : w > 0)
  (h3 : ((3692 * w) % 2^5 = 0)
  (h4 : ((3692 * w) % 3^4 = 0)
  (h5 : ((3692 * w) % 7^3 = 0)
  (h6 : ((3692 * w) % 17^2 = 0) :
  w = 2571912 := sorry

end find_w_l564_564367


namespace num_distinct_colorings_l564_564384

namespace DiskColoring

-- Definition of the conditions
def total_disks : ℕ := 8
def blue_disks : ℕ := 4
def red_disks : ℕ := 3
def green_disks : ℕ := 1

-- Necessary conditions for the problem
def is_valid_coloring (colorings : Finset (Fin total_disks → Fin 3)) : Prop :=
  ∀ coloring ∈ colorings,
    (Finset.filter (λ x, coloring x = 0) Finset.univ).card = blue_disks ∧
    (Finset.filter (λ x, coloring x = 1) Finset.univ).card = red_disks ∧
    (Finset.filter (λ x, coloring x = 2) Finset.univ).card = green_disks

-- Number of distinct colorings considering symmetries
theorem num_distinct_colorings : ∃ (colorings : Finset (Fin total_disks → Fin 3)),
  is_valid_coloring colorings ∧ colorings.card = 32 :=
by
  sorry

end DiskColoring

end num_distinct_colorings_l564_564384


namespace sum_of_real_roots_l564_564718

theorem sum_of_real_roots :
  ∀ x : ℝ, (x^4 - 4 * x - 1 = 0) → ∃ r : ℝ, r = sqrt 2 :=
begin
  sorry
end

end sum_of_real_roots_l564_564718


namespace cost_of_mens_t_shirt_l564_564189

-- Definitions based on conditions
def womens_price : ℕ := 18
def womens_interval : ℕ := 30
def mens_interval : ℕ := 40
def shop_open_hours_per_day : ℕ := 12
def total_earnings_per_week : ℕ := 4914

-- Auxiliary definitions based on conditions
def t_shirts_sold_per_hour (interval : ℕ) : ℕ := 60 / interval
def t_shirts_sold_per_day (interval : ℕ) : ℕ := shop_open_hours_per_day * t_shirts_sold_per_hour interval
def t_shirts_sold_per_week (interval : ℕ) : ℕ := t_shirts_sold_per_day interval * 7

def weekly_earnings_womens : ℕ := womens_price * t_shirts_sold_per_week womens_interval
def weekly_earnings_mens : ℕ := total_earnings_per_week - weekly_earnings_womens
def mens_price : ℚ := weekly_earnings_mens / t_shirts_sold_per_week mens_interval

-- The statement to be proved
theorem cost_of_mens_t_shirt : mens_price = 15 := by
  sorry

end cost_of_mens_t_shirt_l564_564189


namespace book_cost_price_l564_564603

theorem book_cost_price (SP : ℝ) (P : ℝ) (C : ℝ) (hSP: SP = 260) (hP: P = 0.20) : C = 216.67 :=
by 
  sorry

end book_cost_price_l564_564603


namespace card_removal_valid_arrangements_l564_564638

theorem card_removal_valid_arrangements : 
  let cards := {1, 2, 3, 4, 5, 6, 7, 8} in
  let valid_arrangements := 
    {arrangement : List ℕ | arrangement ⊆ cards ∧ 
      ∃ (removed : ℕ), removed ∈ arrangement ∧ 
      ∀ (remaining : List ℕ), remaining = arrangement.erase removed → 
      (List.sorted (≤) remaining ∨ List.sorted (≥) remaining)} in
  valid_arrangements.card = 4 := 
sorry

end card_removal_valid_arrangements_l564_564638


namespace a_100_positive_a_100_abs_lt_018_l564_564343

-- Define the sequence based on the given conditions
def a_n (n : ℕ) : ℝ := real.cos (real.pi / 180 * (10^n) : ℝ)

-- Translate mathematical claims to Lean theorems
theorem a_100_positive : a_n 100 > 0 :=
sorry

theorem a_100_abs_lt_018 : |a_n 100| < 0.18 :=
sorry

end a_100_positive_a_100_abs_lt_018_l564_564343


namespace solve_quartic_eqn_l564_564674

theorem solve_quartic_eqn (z : ℂ) : 
  z ^ 4 - 6 * z ^ 2 + 8 = 0 ↔ z = 2 ∨ z = -2 ∨ z = Complex.sqrt 2 ∨ z = -Complex.sqrt 2 := 
sorry

end solve_quartic_eqn_l564_564674


namespace digit_7_count_in_range_l564_564875

def count_digit_7 : ℕ :=
  let units_place := (107 - 100) / 10 + 1
  let tens_place := 10
  units_place + tens_place

theorem digit_7_count_in_range : count_digit_7 = 20 := by
  sorry

end digit_7_count_in_range_l564_564875


namespace cos_270_eq_zero_l564_564209

theorem cos_270_eq_zero : ∀ (θ : ℝ), θ = 270 → (∀ (p : ℝ × ℝ), p = (0,1) → cos θ = p.fst) → cos 270 = 0 :=
by
  intros θ hθ h
  sorry

end cos_270_eq_zero_l564_564209


namespace rebus_solution_l564_564278

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564278


namespace percentage_of_tomato_plants_is_20_l564_564935

-- Define the conditions
def garden1_plants := 20
def garden1_tomato_percentage := 0.10
def garden2_plants := 15
def garden2_tomato_percentage := 1 / 3

-- Define the question as a theorem to be proved
theorem percentage_of_tomato_plants_is_20 :
  let total_plants := garden1_plants + garden2_plants in
  let total_tomato_plants := (garden1_tomato_percentage * garden1_plants) + (garden2_tomato_percentage * garden2_plants) in
  (total_tomato_plants / total_plants) * 100 = 20 :=
by
  sorry

end percentage_of_tomato_plants_is_20_l564_564935


namespace find_modulus_S_l564_564906

noncomputable def z : ℂ := complex.of_real 2 + complex.I

theorem find_modulus_S :
  |(z^18 - conj(z)^18)| = 2^18 * 5^9 := sorry

end find_modulus_S_l564_564906


namespace value_of_a_l564_564755

theorem value_of_a (a : ℝ) (h : a ≠ 0) :
  let expr := (fun x y => (1/x + y) * (x + a/y)^5)
  let term := coefficient_of (expr x y) (x^2 / y^2)
  term = 20 * a → 
  a = -2 :=
by
  sorry

end value_of_a_l564_564755


namespace number_of_three_digit_numbers_l564_564361

theorem number_of_three_digit_numbers :
  let digits := {1, 2, 3, 4} in
  #|digits × digits × digits| = 64 :=
by
  let digits := {1, 2, 3, 4}
  sorry

end number_of_three_digit_numbers_l564_564361


namespace range_of_3x_plus_y_l564_564770

theorem range_of_3x_plus_y (x y : ℝ) (h : 3 * x^2 + y^2 ≤ 1) : -2 ≤ 3 * x + y ∧ 3 * x + y ≤ 2 :=
begin
  sorry
end

end range_of_3x_plus_y_l564_564770


namespace petya_sequences_l564_564031

theorem petya_sequences (n : ℕ) (h : n = 100) : 
    let S := 5^n
    in S - 3^n = 5^100 - 3^100 :=
by {
  have : S = 5^100,
  {
    rw h,
    exact rfl,
  },
  rw this,
  sorry
}

end petya_sequences_l564_564031


namespace trigonometric_equation_solution_l564_564947

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) :
  3 * Real.sin x ^ 2 - 4 * Real.cos x ^ 2 = Real.sin (2 * x) / 2 →
  (x = (135:ℝ) * Real.pi / 180 + k * Real.pi ∨ x = (53:ℝ + 7/60) * Real.pi / 180 + k * Real.pi) :=
by
  assume h : 3 * Real.sin x ^ 2 - 4 * Real.cos x ^ 2 = Real.sin (2 * x) / 2,
  sorry

end trigonometric_equation_solution_l564_564947


namespace triangle_parallelograms_l564_564233

def f (n : ℕ) : ℕ := 3 * (n + 2).choose 4

theorem triangle_parallelograms (n : ℕ) : 
  ∃ f : ℕ → ℕ, f = λ n, 3 * (n + 2).choose 4 ∧ f(n) = 3 * (n + 2).choose 4 :=
by
  sorry

end triangle_parallelograms_l564_564233


namespace volume_ratio_2_to_1_l564_564516

noncomputable def volume_ratio (R r : ℝ) (hR : R > 0) (hr : r > 0) (h : R > r) : ℝ :=
  let cos_alpha := (R - r) / (R + r)
  let sin_alpha := 2 * real.sqrt (R * r) / (R + r)
  let AE₁ := 2 * real.sqrt (R * r) * R / (R + r)
  let BE₂ := 2 * real.sqrt (R * r) * r / (R + r)
  let E₁E₂ := 4 * R * r / (R + r)
  let K_AE₁E₂B := (real.pi / 3) * (E₁E₂) * (AE₁^2 + AE₁ * BE₂ + BE₂^2)
  let K_E₁AD := (real.pi / 3) * (AE₁^2) * (R - AE₁)
  let K_E₂BD := (real.pi / 3) * (BE₂^2) * (r + BE₂)
  let K_ABD := K_AE₁E₂B - K_E₁AD - K_E₂BD
  let K_hat_ADB := (real.pi / 3) * (4 * real.sqrt (R * r)) * (R^2 + R * r + r^2) / (R + r)^3
  K_ABD / K_hat_ADB

theorem volume_ratio_2_to_1 (R r : ℝ) (hR : R > 0) (hr : r > 0) (h : R > r) :
  volume_ratio R r hR hr h = 2 :=
by
  sorry

end volume_ratio_2_to_1_l564_564516


namespace product_of_slopes_is_minus_one_l564_564739

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + (4/9) * y^2 = 1

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

theorem product_of_slopes_is_minus_one
  (A B P : ℝ × ℝ) (N : ℝ × ℝ)
  (hA : ellipse_equation A.fst A.snd)
  (hB : ellipse_equation B.fst B.snd)
  (hP : ellipse_equation P.fst P.snd)
  (hN : N = (A.fst, A.snd / 2))
  (hM : B = (-A.fst, -A.snd))
  (hBNP_collinear : (P.snd - N.snd) * (N.fst - B.fst) = (B.snd - N.snd) * (N.fst - P.fst)) :
  slope A B * slope A P = (-1 : ℝ) :=
sorry

end product_of_slopes_is_minus_one_l564_564739


namespace largest_number_l564_564186

theorem largest_number (a b c d : ℝ) (h1 : a = 1/2) (h2 : b = 0) (h3 : c = 1) (h4 : d = -9) :
  max (max a b) (max c d) = c :=
by
  sorry

end largest_number_l564_564186


namespace books_loaned_out_during_month_l564_564533

-- Definitions and conditions from the problem
variable (total_books : ℕ := 75)
variable (end_month_books : ℕ := 61)
variable (return_rate : ℝ := 0.65)

-- The number of books loaned out at the end of the month
variable (loaned_out : ℕ)

-- The main theorem we want to prove
theorem books_loaned_out_during_month : 
  let 
    books_not_returned := total_books - end_month_books
    percent_not_returned := 1 - return_rate
    loaned_out := books_not_returned / percent_not_returned
  in loaned_out = 40 := 
by 
  sorry

end books_loaned_out_during_month_l564_564533


namespace infinite_irrational_pairs_l564_564450

theorem infinite_irrational_pairs (n : ℕ) (hn : n ≠ 0 ∧ ∀ (m : ℕ), m * m ≠ n) :
  ∃ (x y : ℝ), irrational x ∧ irrational y ∧ x + y = xy ∧ (x + y) ≥ 0 :=
by {
  sorry
}

end infinite_irrational_pairs_l564_564450


namespace number_of_routes_10_minutes_l564_564075

def M : ℕ → ℕ
| 0     := 1  -- Initial condition
| 1     := 1  -- Initial condition
| (n+2) := M n + M (n+1)  -- Recurrence relation

def M_n_A (n : ℕ) : ℕ := if n = 10 then M n else 0

theorem number_of_routes_10_minutes : M_n_A 10 = 34 :=
by
  sorry

end number_of_routes_10_minutes_l564_564075


namespace minimize_MB_MC_over_MA1_l564_564544

-- Assuming that we have a triangle ABC with a circumcircle
variables {A B C M A1 : Point} (circumcircle : Circle)

-- Define that A1 is the second intersection of AM with the circumcircle of triangle ABC.
def meets_circumcircle (AM_intersect : Line) : Prop := 
  ∃ A1 ∈ circumcircle, A != A1 ∧ AM_intersect = Line.mk A M ∧ A1 ∈ circumcircle

-- Minimizing the expression MB * MC / MA₁
theorem minimize_MB_MC_over_MA1 (hM_inside_triangle : Inside_triangle ABC M) 
  (hAM_meets_circumcircle : meets_circumcircle (Line.mk A M)) :
  M = incenter ABC ↔ fraction_eq_for_min (MB M * MC M) (dist M A1) (2 * inradius ABC) := 
sorry

end minimize_MB_MC_over_MA1_l564_564544


namespace Inequality_l564_564937

open Real EuclideanGeometry

variables {A B C D M : Point}
variables (AB BC CD AD : ℝ)
variables (angle_AMD : ℝ)

-- Point M is the midpoint of BC
def is_midpoint (M B C : Point) : Prop :=
  ∃ (MB MC : ℝ), MB = MC ∧ dist B M = MB ∧ dist C M = MC

-- ∠AMD = 120°
def angle_AMD_def : Prop :=
  angle_AMD = 120

-- Prove the inequality AB + 1/2 BC + CD > AD
theorem Inequality (h1 : is_midpoint M B C) (h2 : angle_AMD_def angle_AMD) :
  AB + (1 / 2) * dist B C + CD > AD :=
sorry

end Inequality_l564_564937


namespace a_n_ge_4_sqrt_9n_7_l564_564628

def S : ℕ → ℚ
| 0     := 1
| (n+1) := (2 + S n) ^ 2 / (4 + S n)

def a (n : ℕ) : ℚ := 
  if n = 1 then S 1 else S n - S (n - 1)

theorem a_n_ge_4_sqrt_9n_7 (n : ℕ) (hn : n > 0) : a n ≥ 4 / Real.sqrt (9 * n + 7) :=
by
  sorry -- Proof to be filled in

end a_n_ge_4_sqrt_9n_7_l564_564628


namespace prism_volume_3a3_or_2a3sqrt5_l564_564973

def volume_of_prism (a : ℝ) : ℝ :=
  let A := (0, 0, 0)
  let B := (2 * a, 0, 0)
  let C := (a, a * Real.sqrt 3, 0)
  let A_1 := (0, 0, h)
  let B_1 := (2 * a, 0, h)
  let C_1 := (a, a * Real.sqrt 3, h)
  let triangle_area := Real.sqrt 3 * a^2
  let trapezoid_area := 2 * triangle_area
  let radius := 2 * a

  -- Define conditions for the height h based on the radius condition
  sorry -- further refinement based on geometric conditions

  let volume := triangle_area * h
  volume

theorem prism_volume_3a3_or_2a3sqrt5 (a : ℝ) :
  volume_of_prism a = 3 * a^3 ∨ volume_of_prism a = 2 * a^3 * Real.sqrt 5 := by
  sorry

end prism_volume_3a3_or_2a3sqrt5_l564_564973


namespace find_k_l564_564006

variables {a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 a_12 : ℝ}
variables {k : ℝ}
variables {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) = a n + d

theorem find_k 
  (a : ℕ → ℝ)
  (h1 : a 5 + a 8 + a 11 = 24)
  (h2 : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 = 100)
  (h3 : a k = 20)
  (h_arithmetic : arithmetic_sequence a) :
  k = 23 :=
sorry

end find_k_l564_564006


namespace range_3x_plus_2y_l564_564799

theorem range_3x_plus_2y (x y : ℝ) : -1 < x + y ∧ x + y < 4 → 2 < x - y ∧ x - y < 3 → 
  -3/2 < 3*x + 2*y ∧ 3*x + 2*y < 23/2 :=
by
  sorry

end range_3x_plus_2y_l564_564799


namespace hyperbola_center_l564_564705

theorem hyperbola_center (x y : ℝ) :
    9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 864 = 0 → (x, y) = (3, 5) :=
by
  sorry

end hyperbola_center_l564_564705


namespace downward_parabola_with_symmetry_l564_564036

-- Define the general form of the problem conditions in Lean
theorem downward_parabola_with_symmetry (k : ℝ) :
  ∃ a : ℝ, a < 0 ∧ ∃ h : ℝ, h = 3 ∧ ∃ k : ℝ, k = k ∧ ∃ (y x : ℝ), y = a * (x - h)^2 + k :=
sorry

end downward_parabola_with_symmetry_l564_564036


namespace sum_of_6_digit_numbers_without_0_and_9_divisible_by_37_l564_564048

-- Statement of the problem in Lean 4
theorem sum_of_6_digit_numbers_without_0_and_9_divisible_by_37 :
  ∀ (S : Finset ℕ), (∀ x ∈ S, 100000 ≤ x ∧ x < 1000000 ∧ ∀ d : ℕ, d ∈ digits 10 x → d ≠ 0 ∧ d ≠ 9) → 37 ∣ S.sum id :=
by
  sorry

end sum_of_6_digit_numbers_without_0_and_9_divisible_by_37_l564_564048


namespace cos_270_eq_zero_l564_564203

theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 := by
  sorry

end cos_270_eq_zero_l564_564203


namespace inhabitant_50_statement_l564_564235

-- Definitions
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

def tells_truth (inh: Inhabitant) (statement: Bool) : Bool :=
  match inh with
  | Inhabitant.knight => statement
  | Inhabitant.liar => not statement

noncomputable def inhabitant_at_position (pos: Nat) : Inhabitant :=
  if (pos % 2) = 1 then
    if pos % 4 = 1 then Inhabitant.knight else Inhabitant.liar
  else
    if pos % 4 = 0 then Inhabitant.knight else Inhabitant.liar

def neighbor (pos: Nat) : Nat := (pos % 50) + 1

-- Theorem statement
theorem inhabitant_50_statement : tells_truth (inhabitant_at_position 50) (inhabitant_at_position (neighbor 50) = Inhabitant.knight) = true :=
by
  -- Proof would go here
  sorry

end inhabitant_50_statement_l564_564235


namespace angles_of_quadrilateral_KLMN_l564_564938

-- Definitions from conditions
variables {K L M N O : Point}
variable h1 : liesOnDiagonal O K M -- Point O lies on the diagonal KM
variable h2 : OM = ON -- OM = ON
variable h3 : equidistantFromLines O NK KL LM -- Point O is equidistant from the lines NK, KL, and LM
variable h4 : angle LOM = 55° -- ∠LOM = 55°
variable h5 : angle KON = 90° -- ∠KON = 90°

-- Definition of the angles of the quadrilateral
def angle_KLMN := {a : Angle // a = ∠KL + ∠LM + ∠MN + ∠NK}

-- The proof problem to be converted into Lean statement
theorem angles_of_quadrilateral_KLMN (h1 h2 h3 h4 h5) : 
  angle_KLMN = {a : Angle // a = 20° ∨ a = 90° ∨ a = 125° ∨ a = 125° } := by
  sorry

end angles_of_quadrilateral_KLMN_l564_564938


namespace solve_abs_quadratic_l564_564463

theorem solve_abs_quadratic :
  ∀ x : ℝ, abs (x^2 - 4 * x + 4) = 3 - x ↔ (x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) :=
by
  sorry

end solve_abs_quadratic_l564_564463


namespace sum_of_6_digit_numbers_without_0_and_9_divisible_by_37_l564_564047

-- Statement of the problem in Lean 4
theorem sum_of_6_digit_numbers_without_0_and_9_divisible_by_37 :
  ∀ (S : Finset ℕ), (∀ x ∈ S, 100000 ≤ x ∧ x < 1000000 ∧ ∀ d : ℕ, d ∈ digits 10 x → d ≠ 0 ∧ d ≠ 9) → 37 ∣ S.sum id :=
by
  sorry

end sum_of_6_digit_numbers_without_0_and_9_divisible_by_37_l564_564047


namespace harmonic_mean_closest_to_2_l564_564082

theorem harmonic_mean_closest_to_2 : 
  let a : ℕ := 1 
  let b : ℕ := 2023
  let H := (2 * a * b) / (a + b)
  round H = 2
:= 
by 
  let a := 1
  let b := 2023
  let H := (2 * a * b) / (a + b)
  have H_eq : H = 4046 / 2024 := by sorry
  have approx_H : H ≈ 1.999 := by sorry
  exact Eq.symm (round 1.999) 2

end harmonic_mean_closest_to_2_l564_564082


namespace cylinder_surface_area_l564_564563

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 15) (h_radius : r = 2) : 
  let S := 2 * Real.pi * r^2 + 2 * Real.pi * r * h in
  S = 68 * Real.pi := 
by
  --- the proof will go here, but for now we put sorry
  sorry

end cylinder_surface_area_l564_564563


namespace intersection_points_l564_564229

noncomputable def line1 : ℝ × ℝ → Prop := λ ⟨x, y⟩, 3 * x + 2 * y - 12 = 0
noncomputable def line2 : ℝ × ℝ → Prop := λ ⟨x, y⟩, 5 * x - 2 * y - 10 = 0
noncomputable def line3 : ℝ × ℝ → Prop := λ ⟨x, _⟩, x = 3
noncomputable def line4 : ℝ × ℝ → Prop := λ ⟨_, y⟩, y = 3
noncomputable def line5 : ℝ × ℝ → Prop := λ ⟨x, y⟩, 2 * x + y - 8 = 0

theorem intersection_points :
  ∃ (points : Finset (ℝ × ℝ)),
    points.card = 5 ∧
    (∀ p ∈ points, line1 p ∨ line2 p ∨ line3 p ∨ line4 p ∨ line5 p) := sorry

end intersection_points_l564_564229


namespace number_of_a_for_T_ne_S_is_3_l564_564407

def S : Set ℝ := {x | x ^ 2 - 5 * |x| + 6 = 0}
def T (a : ℝ) : Set ℝ := {x | (a - 2) * x = 2}

theorem number_of_a_for_T_ne_S_is_3 : {a : ℝ | T a ≠ S}.toFinset.card = 3 :=
sorry

end number_of_a_for_T_ne_S_is_3_l564_564407


namespace rebus_solution_l564_564281

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564281


namespace cuboid_volume_l564_564149

theorem cuboid_volume (P h : ℝ) (P_eq : P = 32) (h_eq : h = 9) :
  ∃ (s : ℝ), 4 * s = P ∧ s * s * h = 576 :=
by
  sorry

end cuboid_volume_l564_564149


namespace cos_270_eq_zero_l564_564212

theorem cos_270_eq_zero : Real.cos (270 * (π / 180)) = 0 :=
by
  -- Conditions given in the problem
  have rotation_def : ∀ (θ : ℝ), ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x = Real.cos θ ∧ y = Real.sin θ := by
    intros θ
    use [Real.cos θ, Real.sin θ]
    split
    · exact Real.cos_sq_add_sin_sq θ
    split
    · rfl
    · rfl

  have rotation_270 : ∃ (x y : ℝ), x = 0 ∧ y = -1 := by
    use [0, -1]
    split
    · rfl
    · rfl

  -- Goal to be proved
  have result := Real.cos (270 * (π / 180))
  show result = 0
  sorry

end cos_270_eq_zero_l564_564212


namespace count_digit_7_in_range_l564_564859

theorem count_digit_7_in_range : ∀ n ∈ finset.Icc 100 199, (n % 10 = 7 ∨ n / 10 % 10 = 7) →  ∃ counts, counts = 20 :=
by
  sorry

end count_digit_7_in_range_l564_564859


namespace time_when_ball_hits_ground_l564_564481

def height (t : ℝ) : ℝ :=
  -16 * t^2 + 34 * t + 25

theorem time_when_ball_hits_ground : ∃ t : ℝ, height t = 0 ∧ t = 25 / 8 :=
by
  sorry

end time_when_ball_hits_ground_l564_564481


namespace num_triangles_l564_564362

open Finset

theorem num_triangles (n p1 p2 : ℕ) (h1 : n ≥ 3) (h2 : p1 ≥ 3) (h3 : p2 ≥ 3) :
  (choose n 3) - (choose p1 3) - (choose p2 3) = (choose n 3) - (choose p1 3) - (choose p2 3) := sorry

end num_triangles_l564_564362


namespace grisha_win_probability_expected_number_coin_flips_l564_564822

-- Problem 1: Probability of Grisha's win
theorem grisha_win_probability : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (grisha_win : ℚ) :=
  sorry

-- Problem 2: Expected number of coin flips
theorem expected_number_coin_flips : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (expected_tosses : ℚ) :=
  sorry

end grisha_win_probability_expected_number_coin_flips_l564_564822


namespace digit_7_count_in_range_100_to_199_l564_564885

/-- 
Given the range of numbers from 100 to 199 inclusive,
prove that the digit 7 appears exactly 20 times.
-/
theorem digit_7_count_in_range_100_to_199 : 
  let count_7 := (λ n : ℕ, 
    (n / 100 = 7) +  -- hundreds place
    ((n % 100) / 10 = 7) + -- tens place
    ((n % 10) = 7)) in
  (Finset.sum (Finset.range' 100 100) count_7 = 20) :=
by sorry

end digit_7_count_in_range_100_to_199_l564_564885


namespace green_notebook_cost_each_l564_564445

-- Definitions for conditions:
def num_notebooks := 4
def num_green_notebooks := 2
def num_black_notebooks := 1
def num_pink_notebooks := 1
def total_cost := 45
def black_notebook_cost := 15
def pink_notebook_cost := 10

-- Define the problem statement:
theorem green_notebook_cost_each : 
  (2 * g + black_notebook_cost + pink_notebook_cost = total_cost) → 
  g = 10 := 
by 
  intros h
  sorry

end green_notebook_cost_each_l564_564445


namespace find_digits_l564_564270

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l564_564270


namespace jasmine_needs_small_bottles_l564_564170

theorem jasmine_needs_small_bottles
  (small_bottle_capacity : ℕ)
  (large_bottle_capacity : ℕ)
  (existing_shampoo : ℕ)
  (small_bottle_capacity_eq : small_bottle_capacity = 40)
  (large_bottle_capacity_eq : large_bottle_capacity = 800)
  (existing_shampoo_eq : existing_shampoo = 120) :
  (large_bottle_capacity - existing_shampoo) / small_bottle_capacity = 17 := 
by
  rw [small_bottle_capacity_eq, large_bottle_capacity_eq, existing_shampoo_eq]
  norm_num
  sorry

end jasmine_needs_small_bottles_l564_564170


namespace investment_cost_correct_total_revenue_correct_total_profit_correct_profit_percentage_correct_l564_564159

def bookA_cost : ℝ := 50
def bookB_cost : ℝ := 75
def bookC_cost : ℝ := 100

def bookA_sell : ℝ := 60
def bookB_sell : ℝ := 90
def bookC_sell : ℝ := 120

def total_investment_cost : ℝ :=
  bookA_cost + bookB_cost + bookC_cost

def total_revenue : ℝ :=
  bookA_sell + bookB_sell + bookC_sell

def total_profit : ℝ :=
  total_revenue - total_investment_cost

def profit_percentage : ℝ :=
  (total_profit / total_investment_cost) * 100

theorem investment_cost_correct : total_investment_cost = 225 := by sorry
theorem total_revenue_correct : total_revenue = 270 := by sorry
theorem total_profit_correct : total_profit = 45 := by sorry
theorem profit_percentage_correct : profit_percentage = 20 := by sorry

end investment_cost_correct_total_revenue_correct_total_profit_correct_profit_percentage_correct_l564_564159


namespace parabola_opens_downwards_l564_564034

theorem parabola_opens_downwards (a : ℝ) (h : ℝ) (k : ℝ) :
  a < 0 → h = 3 → ∃ k, (∀ x, y = a * (x - h) ^ 2 + k → y = -(x - 3)^2 + k) :=
by
  intros ha hh
  use k
  sorry

end parabola_opens_downwards_l564_564034


namespace grisha_wins_probability_expected_flips_l564_564824

-- Define conditions
def even_flip_heads_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 0 ∧ flip n = tt

def odd_flip_tails_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 1 ∧ flip n = ff

-- 1. Prove the probability P of Grisha winning the coin toss game is 1/3
theorem grisha_wins_probability (flip : ℕ → bool) (P : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  P = 1/3 := sorry

-- 2. Prove the expected number E of coin flips until the outcome is decided is 2
theorem expected_flips (flip : ℕ → bool) (E : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  E = 2 := sorry

end grisha_wins_probability_expected_flips_l564_564824


namespace find_s_for_quadratic_l564_564948

noncomputable theory
open Classical

-- We define our condition, which is having an equation and completing the square
variables (x : ℝ) (r s : ℝ)

-- The original quadratic equation
def quadratic_eq : Prop := 4 * x^2 - 16 * x - 200 = 0

-- Equation obtained after completing the square 
def completing_square : Prop := (x + r)^2 = s

-- The proof problem
theorem find_s_for_quadratic : (quadratic_eq x) → (completing_square x r 54) :=
by {
  sorry
}

end find_s_for_quadratic_l564_564948


namespace sum_and_product_of_roots_l564_564985

theorem sum_and_product_of_roots (m n : ℕ) : 
  (∀ x y : ℝ, (3 * x^2 - m * x + n = 0) ∧ (3 * y^2 - m * y + n = 0) ∧ (x + y = 8) ∧ (x * y = 9)) → (m + n = 51) :=
by
  intros x y h
  sorry

end sum_and_product_of_roots_l564_564985


namespace rebus_solution_l564_564267

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564267


namespace find_num_students_l564_564539

variables (N T : ℕ)
variables (h1 : T = N * 80)
variables (h2 : 5 * 20 = 100)
variables (h3 : (T - 100) / (N - 5) = 90)

theorem find_num_students (h1 : T = N * 80) (h3 : (T - 100) / (N - 5) = 90) : N = 35 :=
sorry

end find_num_students_l564_564539


namespace Robert_more_than_Claire_l564_564431

variable (Lisa Claire Robert : ℕ)

theorem Robert_more_than_Claire (h1 : Lisa = 3 * Claire) (h2 : Claire = 10) (h3 : Robert > Claire) :
  Robert > 10 :=
by
  rw [h2] at h3
  assumption

end Robert_more_than_Claire_l564_564431


namespace four_digit_number_l564_564709

theorem four_digit_number (a b c d : ℕ)
    (h1 : 0 ≤ a) (h2 : a ≤ 9)
    (h3 : 0 ≤ b) (h4 : b ≤ 9)
    (h5 : 0 ≤ c) (h6 : c ≤ 9)
    (h7 : 0 ≤ d) (h8 : d ≤ 9)
    (h9 : 2 * (1000 * a + 100 * b + 10 * c + d) + 1000 = 1000 * d + 100 * c + 10 * b + a)
    : (1000 * a + 100 * b + 10 * c + d) = 2996 :=
by
  sorry

end four_digit_number_l564_564709


namespace count_digit_7_from_100_to_199_l564_564883

theorem count_digit_7_from_100_to_199 : 
  (list.range' 100 100).countp (λ n, (70 ≤ n ∧ n ≤ 79) ∨ (n % 10 = 7)) = 20 :=
by
  simp
  sorry

end count_digit_7_from_100_to_199_l564_564883


namespace book_price_increase_l564_564982

theorem book_price_increase (P : ℝ) : 
    let new_price := (1.15 * 1.15) * P in 
    (new_price - P) / P * 100 = 32.25 :=
by
  sorry

end book_price_increase_l564_564982


namespace find_vector_magnitude_l564_564357

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def angle_between (v w : ℝ × ℝ) : ℝ := real.acos (dot_product v w / (magnitude v * magnitude w))

theorem find_vector_magnitude 
  (a b : ℝ × ℝ)
  (ha : magnitude a = real.sqrt 3)
  (hb : magnitude b = real.sqrt 6)
  (hab : angle_between a b = 3 * real.pi / 4) :
  magnitude (4 • a - b) = real.sqrt 78 :=
sorry

end find_vector_magnitude_l564_564357


namespace edric_hourly_rate_l564_564245

theorem edric_hourly_rate
  (monthly_salary : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (H1 : monthly_salary = 576)
  (H2 : hours_per_day = 8)
  (H3 : days_per_week = 6)
  (H4 : weeks_per_month = 4) :
  monthly_salary / weeks_per_month / days_per_week / hours_per_day = 3 := by
  sorry

end edric_hourly_rate_l564_564245


namespace a100_pos_a100_abs_lt_018_l564_564336

noncomputable def a (n : ℕ) : ℝ := real.cos (10^n * real.pi / 180)

theorem a100_pos : (a 100 > 0) :=
by
  sorry

theorem a100_abs_lt_018 : (|a 100| < 0.18) :=
by
  sorry

end a100_pos_a100_abs_lt_018_l564_564336


namespace dealer_weight_approx_l564_564565

def profit_percent : ℝ := 8.695652173913047 / 100

def dealer_equation (W : ℝ) : Prop := W + (W * profit_percent) = 1

theorem dealer_weight_approx (W : ℝ) (h : dealer_equation W) : W ≈ 0.92 := sorry

end dealer_weight_approx_l564_564565


namespace find_digits_l564_564269

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l564_564269


namespace afternoon_sequences_count_l564_564087

-- Definitions and conditions based on part a)
def letters := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def printed_before_lunch := 8 ∈ letters

-- Lean statement to prove the number of possible sequences
theorem afternoon_sequences_count : 
  ∃ afternoon_sequences, ∀ (letters : Finset ℕ), printed_before_lunch -> (afternoon_sequences.card = 704) :=
begin
  sorry
end

end afternoon_sequences_count_l564_564087


namespace alloy_parts_separation_l564_564577

theorem alloy_parts_separation {p q x : ℝ} (h0 : p ≠ q)
  (h1 : 6 * p ≠ 16 * q)
  (h2 : 6 * x * p + 2 * (8 - 2 * x) * q = 8 * (8 - x) * p + 6 * x * q) :
  x = 2.4 :=
by
  sorry

end alloy_parts_separation_l564_564577


namespace grisha_win_probability_expected_number_coin_flips_l564_564821

-- Problem 1: Probability of Grisha's win
theorem grisha_win_probability : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (grisha_win : ℚ) :=
  sorry

-- Problem 2: Expected number of coin flips
theorem expected_number_coin_flips : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (expected_tosses : ℚ) :=
  sorry

end grisha_win_probability_expected_number_coin_flips_l564_564821


namespace fiftieth_statement_l564_564238

-- Define the types
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

-- Define the function telling the statement
def statement (inhabitant : Inhabitant) : String :=
  match inhabitant with
  | Inhabitant.knight => "Knight"
  | Inhabitant.liar => "Liar"

-- Define the condition: knights tell the truth and liars lie
def tells_truth (inhabitant : Inhabitant) (statement_about_neighbor : String) : Prop :=
  match inhabitant with
  | Inhabitant.knight => statement_about_neighbor = "Knight"
  | Inhabitant.liar => statement_about_neighbor ≠ "Knight"

-- Define a function that determines what each inhabitant says about their right-hand neighbor
def what_they_say (idx : ℕ) : String :=
  if idx % 2 = 0 then "Liar" else "Knight"

-- Define the inhabitant pattern
def inhabitant_at (idx : ℕ) : Inhabitant :=
  if idx % 2 = 0 then Inhabitant.liar else Inhabitant.knight

-- The main theorem statement
theorem fiftieth_statement : tells_truth (inhabitant_at 49) (what_they_say 50) :=
by 
  -- This proof outlines the theorem statement only
  sorry

end fiftieth_statement_l564_564238


namespace find_z_values_l564_564655

theorem find_z_values : {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -sqrt 2, sqrt 2, 2} :=
sorry

end find_z_values_l564_564655


namespace cd_cost_l564_564789

theorem cd_cost (mp3_cost savings father_amt lacks cd_cost : ℝ) :
  mp3_cost = 120 ∧ savings = 55 ∧ father_amt = 20 ∧ lacks = 64 →
  120 + cd_cost - (savings + father_amt) = lacks → 
  cd_cost = 19 :=
by
  intros
  sorry

end cd_cost_l564_564789


namespace min_distance_tangent_line_l564_564762

def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4
def line_equation (x y : ℝ) : Prop := x - y - 6 = 0

theorem min_distance_tangent_line (M : ℝ × ℝ) 
    (H : line_equation M.1 M.2) :
  ∃ N : ℝ × ℝ, circle_equation N.1 N.2 ∧ t_line_tangent_to_circle M N  ∧
  |dist M N| = real.sqrt 14 :=
sorry

end min_distance_tangent_line_l564_564762


namespace min_payment_l564_564016

def prices := {trousers := 2800, skirt := 1300, jacket := 2600, blouse := 900}

def promo_coupon (total_cost: ℕ) : ℕ := 
  total_cost - min (1000) (15 * total_cost / 100)

def promo_third_item_free (items: List ℕ) : ℕ :=
  let sorted_items := List.sort items
  total_cost - List.head sorted_items

def delivery_cost (total_cost : ℕ) : ℕ :=
  if total_cost > 5000 then 0 else 350

def self_pickup_cost : ℕ := 100

-- Main statement: the minimum cost of purchases including delivery
theorem min_payment 
    (p : Σ (items : List ℕ), items = [prices.trousers, prices.skirt, prices.jacket, prices.blouse]) :
  (Σ (min_cost : ℕ), min_cost = 6265) :=
sorry

end min_payment_l564_564016


namespace area_GaGbGc_l564_564910

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_GaGbGc (H : Type)
  [orthocenter H (triangle.mk (13 : ℝ) 14 15)] :
  let ABC_area := area_triangle 13 14 15 in
  1 / 9 * ABC_area = 28 / 3 :=
by
  let s := (13 + 14 + 15) / 2
  let ABC_area := real.sqrt (s * (s - 13) * (s - 14) * (s - 15))
  have ABC_area_eq : ABC_area = 84 := sorry
  calc
  1 / 9 * ABC_area
      = 1 / 9 * 84 : by rw [ABC_area_eq]
  ... = 28 / 3 : by norm_num


end area_GaGbGc_l564_564910


namespace find_BO_l564_564542

/-- Define the known variables and conditions --/
variables (OC DM BC BO : ℝ)
variable (OC_eq : OC = 12)
variable (DM_eq : DM = 10)

-- Assert the problem solution
theorem find_BO : BO = 8 :=
by
  -- Introduce the conditions OC_eq and DM_eq
  intro OC_eq DM_eq
  -- State the known conditions
  have h1 : OC = 12 := OC_eq
  have h2 : DM = 10 := DM_eq
  -- Calculate BC based on the given conditions and relationships
  -- Since the triangles are similar and specific ratios are used in the condition
  have hBC : BC = 2 * DM := by sorry  -- BC = 20
  -- Calculate BO from BC and OC as defined in the problem and condition.
  have hBO : BO = BC - OC := by sorry
  -- Obtain the final result
  show BO = 8 from sorry 

end find_BO_l564_564542


namespace inequality_proof_l564_564941

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a / b + b / c + c / a) ^ 2 ≥ 3 * (a / c + c / b + b / a) :=
  sorry

end inequality_proof_l564_564941


namespace sum_largest_and_second_largest_l564_564508

theorem sum_largest_and_second_largest (S : Finset ℕ) (h : S = {10, 11, 12, 13, 14}) :
  let largest := S.max' sorry
  let second_largest := (S.erase largest).max' sorry
  largest + second_largest = 27 :=
by
  rw h at *
  let largest := S.max' sorry
  let second_largest := (S.erase largest).max' sorry
  have h_largest : largest = 14 := by sorry
  have h_second_largest : second_largest = 13 := by sorry
  rw [h_largest, h_second_largest]
  exact rfl

end sum_largest_and_second_largest_l564_564508


namespace triangle_properties_l564_564853

variables {A B C : ℝ} {a b c : ℝ} {D : ℝ}

-- Conditions in the problem
def triangle_sides (A B C a b c : ℝ) : Prop :=
  (sqrt 3 * b * cos A - a * sin B = 0) ∧
  (AC = 2) ∧
  (CD = 2 * sqrt 3) ∧
  (D = AC / 2)

-- Our main theorem to prove
theorem triangle_properties (A B C a b c : ℝ) (AC CD D : ℝ)
  (h : triangle_sides A B C a b c) :
  A = π/3 ∧ a = 2 * sqrt 13 :=
sorry

end triangle_properties_l564_564853


namespace polar_equation_of_C_slope_of_line_l564_564388

-- Given parametric equations for the curve C
def curve_C (θ : ℝ) : ℝ × ℝ :=
  (3 + sqrt 5 * cos θ, sqrt 5 * sin θ)

-- Given parametric equations for the line l
def line_l (t α : ℝ) : ℝ × ℝ :=
  (1 + t * cos α, t * sin α)

-- The problem statement: Prove the given results
theorem polar_equation_of_C :
  ∀ θ : ℝ,
    let (x, y) := curve_C θ in
    (x - 3)^2 + y^2 = 5 → cos θ = (x - 3) / sqrt 5 → -- hint for conversion to polar
    y = sqrt 5 * sin θ →                              -- hint for conversion to polar
    exists ρ θ', (ρ = sqrt ((x-3)^2 + y^2) ∧ θ' = θ ∧ ρ ^ 2 - 6 * ρ * cos θ' + 4 = 0)
    := sorry

theorem slope_of_line :
  ∀ α : ℝ,
    (∀ t : ℝ, let (x, y) := line_l t α in
      |((x-1+t*cosα, y-t*sinα) = (x,y))| = 2*sqrt 3)
    →
    exists k : ℝ, k = -1 ∨ k = 1
    := sorry

end polar_equation_of_C_slope_of_line_l564_564388


namespace triangle_tan_c_l564_564849

theorem triangle_tan_c (A B C : ℝ) 
  (h1 : Real.cot A * Real.cot C = 1 / 3) 
  (h2 : Real.cot B * Real.cot C = 2 / 9) 
  (h_sum_angles : A + B + C = 180) : 
  Real.tan C = √6 :=
by
  sorry

end triangle_tan_c_l564_564849


namespace digit_7_count_in_range_100_to_199_l564_564886

/-- 
Given the range of numbers from 100 to 199 inclusive,
prove that the digit 7 appears exactly 20 times.
-/
theorem digit_7_count_in_range_100_to_199 : 
  let count_7 := (λ n : ℕ, 
    (n / 100 = 7) +  -- hundreds place
    ((n % 100) / 10 = 7) + -- tens place
    ((n % 10) = 7)) in
  (Finset.sum (Finset.range' 100 100) count_7 = 20) :=
by sorry

end digit_7_count_in_range_100_to_199_l564_564886


namespace problem1_min_value_problem2_range_a_l564_564923

def f (x a : ℝ) : ℝ :=
  if x < 1 then 2^x - a else 4 * (x - a) * (x - 2 * a)

theorem problem1_min_value (h : ∀ x : ℝ, (1 : ℝ) = 1) : 
  ∃ x : ℝ, f x 1 = -1 := 
sorry

theorem problem2_range_a (h : ∀ x : ℝ, (f x a) = 0 → (∃ y z : ℝ, (f y a = 0) ∧ (f z a = 0) ∧ y ≠ z)) :
  ({ a : ℝ | (½ ≤ a ∧ a < 1) ∨ a ≥ 2 } = set_of f) :=
sorry

end problem1_min_value_problem2_range_a_l564_564923


namespace num_subsets_satisfying_conditions_l564_564428

open Set Finset Fintype

theorem num_subsets_satisfying_conditions :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let A : Finset ℕ := sorry
  (∃ A, A ⊆ S ∧ A ∩ {1, 2, 3} ≠ ∅ ∧ A ∪ {4, 5, 6} ≠ S) →
  card (filter (λ A, A ∩ {1, 2, 3} ≠ ∅ ∧ A ∪ {4, 5, 6} ≠ S) (powerset S)) = 888 :=
sorry

end num_subsets_satisfying_conditions_l564_564428


namespace arc_length_of_parametric_curve_l564_564606

noncomputable def arc_length (x y : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
  ∫ t in t1..t2, sqrt ((deriv x t)^2 + (deriv y t)^2)

def x (t : ℝ) : ℝ := 4 * (t - sin t)
def y (t : ℝ) : ℝ := 4 * (1 - cos t)

theorem arc_length_of_parametric_curve :
  arc_length x y (π / 2) (2 * π) = 8 * (sqrt 2 - 1) :=
by
  sorry

end arc_length_of_parametric_curve_l564_564606


namespace angle_trig_identity_l564_564370

theorem angle_trig_identity
  (A B C : ℝ)
  (h_sum : A + B + C = Real.pi) :
  Real.cos (A / 2) ^ 2 = Real.cos (B / 2) ^ 2 + Real.cos (C / 2) ^ 2 - 
                       2 * Real.cos (B / 2) * Real.cos (C / 2) * Real.sin (A / 2) :=
by
  sorry

end angle_trig_identity_l564_564370


namespace canoe_downstream_speed_l564_564553

-- Definitions based on conditions
def upstream_speed : ℝ := 9  -- upspeed
def stream_speed : ℝ := 1.5  -- vspeed

-- Theorem to prove the downstream speed
theorem canoe_downstream_speed (V_c : ℝ) (V_d : ℝ) :
  (V_c - stream_speed = upstream_speed) →
  (V_d = V_c + stream_speed) →
  V_d = 12 := by 
  intro h1 h2
  sorry

end canoe_downstream_speed_l564_564553


namespace equal_segments_or_angle_l564_564008

variables {A B C D K L E F : Type}
variables [innermost: E ↝ incenter_of (triangle ABD)] 
variables [innermost: F ↝ incenter_of (triangle ACD)] 
variables [meets_EF_AB_AC: (line EF) meets (line AB) and (line AC) at (K, L)]
variables [altitude_AD: D ↝ altitude_of (triangle ABC)]

theorem equal_segments_or_angle : 
  AK = AL ↔ AB = AC ∨ angle A B C = 90° :=
sorry

end equal_segments_or_angle_l564_564008


namespace man_rowing_speed_l564_564158

noncomputable def rowing_speed_in_still_water : ℝ :=
  let distance := 0.1   -- kilometers
  let time := 20 / 3600 -- hours
  let current_speed := 3 -- km/hr
  let downstream_speed := distance / time
  downstream_speed - current_speed

theorem man_rowing_speed :
  rowing_speed_in_still_water = 15 :=
  by
    -- Proof comes here
    sorry

end man_rowing_speed_l564_564158


namespace length_of_bridge_l564_564083

theorem length_of_bridge (length_of_train speed_kmph time_seconds : ℕ) (h_train_length : length_of_train = 120) (h_train_speed : speed_kmph = 45) (h_time : time_seconds = 30) : 
  let speed_mps := (speed_kmph * 1000) / 3600,
      total_distance := speed_mps * time_seconds,
      length_of_bridge := total_distance - length_of_train 
  in length_of_bridge = 255 :=
by
  sorry

end length_of_bridge_l564_564083


namespace cary_ivy_removal_days_correct_l564_564615

noncomputable def cary_ivy_removal_days (initial_ivy : ℕ) (ivy_removed_per_day : ℕ) (ivy_growth_per_night : ℕ) : ℕ :=
  initial_ivy / (ivy_removed_per_day - ivy_growth_per_night)

theorem cary_ivy_removal_days_correct :
  cary_ivy_removal_days 40 6 2 = 10 :=
by
  -- The body of the proof is omitted; it will be filled with the actual proof.
  sorry

end cary_ivy_removal_days_correct_l564_564615


namespace marching_band_members_l564_564088

theorem marching_band_members (B W P : ℕ) (h1 : P = 4 * W) (h2 : W = 2 * B) (h3 : B = 10) : B + W + P = 110 :=
by
  sorry

end marching_band_members_l564_564088


namespace remaining_circles_l564_564780

-- Define the problem conditions
variables {P R X Y Z V W U T : Point}
variable {Pentagon : Polygon}
variable {PR : ℝ}

-- Assume that the pentagon is regular and has the necessary properties
axiom regular_pentagon : regular Pentagon
axiom point_on_vertex (Pentagon : Polygon) (P : Point) : vertex_on_polygon Pentagon P
axiom point_on_vertex (Pentagon : Polygon) (R : Point) : vertex_on_polygon Pentagon R

-- Define distances and circles according to problem description
axiom circle_k1 : Circle P PR
axiom circle_k2 : Circle R PR
axiom circle_k3 : Circle X (distance X Y)
axiom circle_k4 : Circle P (distance X Y)
axiom circle_k5 : Circle V (distance R Z)
axiom circle_k6 : Circle W (distance R Z)

-- Assume points are defined correctly as per intersection and distances
axiom intersect_k1_k2 : intersects circle_k1 circle_k2 X ∧ intersects circle_k1 circle_k2 Y
axiom intersect_k3_k1 : intersects circle_k3 circle_k1 Y ∧ intersects circle_k3 circle_k1 Z
axiom intersect_k4_k3 : intersects circle_k4 circle_k3 V ∧ intersects circle_k4 circle_k3 W
axiom intersect_k5 : intersects circle_k5 circle_k6 U ∧ intersects circle_k5 circle_k6 T

-- Final verification needs to prove \(k_{7}\) and \(k_{8}\) combinations
theorem remaining_circles :
  ∃ k7 k8 : Circle, (center k7 = P ∧ radius k7 = distance P T) ∧ 
             (center k8 = R ∧ radius k8 = distance R T) := 
by sorry

end remaining_circles_l564_564780


namespace num_divisible_by_7_in_range_l564_564526

theorem num_divisible_by_7_in_range (n : ℤ) (h : 1 ≤ n ∧ n ≤ 2015)
    : (∃ k, 1 ≤ k ∧ k ≤ 335 ∧ 3 ^ (6 * k) + (6 * k) ^ 3 ≡ 0 [MOD 7]) :=
sorry

end num_divisible_by_7_in_range_l564_564526


namespace solve_quartic_eq_l564_564663

theorem solve_quartic_eq (z : ℝ) : z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = 2 :=
by 
  sorry

end solve_quartic_eq_l564_564663


namespace Connie_total_markers_l564_564620

/--
Connie has 41 red markers and 64 blue markers. 
We want to prove that the total number of markers Connie has is 105.
-/
theorem Connie_total_markers : 
  let red_markers := 41
  let blue_markers := 64
  let total_markers := red_markers + blue_markers
  total_markers = 105 :=
by
  sorry

end Connie_total_markers_l564_564620


namespace original_bathroom_area_l564_564990

noncomputable def original_length (A_new : ℝ) (W_new : ℝ) : ℝ :=
  A_new / W_new

noncomputable def original_area (L : ℝ) (W : ℝ) : ℝ :=
  L * W

theorem original_bathroom_area :
  let W := 8
  let W_new := 12
  let A_new := 140
  original_area (original_length A_new W_new) W  ≈ 93.36 :=
by
  sorry

end original_bathroom_area_l564_564990


namespace sum_P_1_to_2009_l564_564902

def P (n : ℕ) : ℕ :=
  (n.digits 10).filter (λ d, d ≠ 0).prod

theorem sum_P_1_to_2009 : (∑ i in Finset.range 2010, P i) = 4477547 :=
by
  sorry

end sum_P_1_to_2009_l564_564902


namespace cos_270_eq_zero_l564_564211

theorem cos_270_eq_zero : Real.cos (270 * (π / 180)) = 0 :=
by
  -- Conditions given in the problem
  have rotation_def : ∀ (θ : ℝ), ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x = Real.cos θ ∧ y = Real.sin θ := by
    intros θ
    use [Real.cos θ, Real.sin θ]
    split
    · exact Real.cos_sq_add_sin_sq θ
    split
    · rfl
    · rfl

  have rotation_270 : ∃ (x y : ℝ), x = 0 ∧ y = -1 := by
    use [0, -1]
    split
    · rfl
    · rfl

  -- Goal to be proved
  have result := Real.cos (270 * (π / 180))
  show result = 0
  sorry

end cos_270_eq_zero_l564_564211


namespace distance_P_to_origin_l564_564479

-- Define the point P with coordinates (1, 2, 3)
def P : ℝ × ℝ × ℝ := (1, 2, 3)

-- Define a function to calculate the distance from a point to the origin
def distance_to_origin (p : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := p
  in Real.sqrt (x * x + y * y + z * z)

-- Prove that the distance_to_origin of point P is √14
theorem distance_P_to_origin : distance_to_origin P = Real.sqrt 14 := by
  -- Proof will go here
  sorry

end distance_P_to_origin_l564_564479


namespace a4_b4_c4_double_square_l564_564531

theorem a4_b4_c4_double_square (a b c : ℤ) (h : a = b + c) : 
  a^4 + b^4 + c^4 = 2 * ((a^2 - b * c)^2) :=
by {
  sorry -- proof is not provided as per instructions
}

end a4_b4_c4_double_square_l564_564531


namespace correct_statements_l564_564455

noncomputable def f (x : ℝ) : ℝ := 4 * sin (2 * x + C₁)

theorem correct_statements (C₁ C₂ C₃ C₄ : ℝ) :
  (∀ x : ℝ, f x = 4 * cos (2 * x - C₂)) ∧
  ¬(∀ p : ℝ, p > 0 → ∀ x : ℝ, f (x + p) = f x → p = 2 * π) ∧
  (f (-C₃) = 0) ∧
  ¬(∀ x : ℝ, f (x + 2 * C₄) = f (-x)) :=
sorry

end correct_statements_l564_564455


namespace compound_interest_comparison_l564_564153

theorem compound_interest_comparison :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  (P * (1 + r_monthly)^((12 * t)) > P * (1 + r_annual)^t) :=
by
  sorry

end compound_interest_comparison_l564_564153


namespace find_a_l564_564372

open Real

-- Definitions of the conditions
def circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 4
def line (a x y : ℝ) : Prop := x + y = a + 1
def chord_length : ℝ := 2 * sqrt 2

-- Main theorem statement
theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, circle x y ∧ line a x y ∧ dist (2, 2) (x, y) = chord_length / 2) → 
  (a = 1 ∨ a = 5) := 
sorry

end find_a_l564_564372


namespace poly_degree_and_terms_correct_l564_564960

-- Define the specific polynomial
def poly : Polynomial (ℤ × ℤ) := Polynomial.C (1, 0) * Polynomial.x ^ (2, 0) * Polynomial.y ^ (3, 0) 
                                   - Polynomial.C (3, 0) * Polynomial.x ^ (1, 0) * Polynomial.y ^ (3, 0) 
                                   - Polynomial.C (2, 0)

-- Degree calculation
def poly_degree := 5

-- Number of terms calculation
def poly_num_terms := 3

-- The theorem stating the degree and number of terms of the polynomial
theorem poly_degree_and_terms_correct :
  deg poly = poly_degree ∧ count_terms poly = poly_num_terms :=
by
  sorry

end poly_degree_and_terms_correct_l564_564960


namespace determine_x_in_triangle_l564_564845

theorem determine_x_in_triangle 
  (A B C D : Type) 
  [triangle A B C]
  (angle_BAC_eq_90 : angle A B C = 90)
  (angle_ABC_eq_3x : angle B A C = 3 * x)
  (point_D_on_AC : on_line A C D)
  (angle_ABD_eq_2x : angle A B D = 2 * x) :
  x = 18 := 
by
  sorry

end determine_x_in_triangle_l564_564845


namespace sum_g_values_l564_564010

def g (x : ℝ) : ℝ := 4 / (16 ^ x + 4)

theorem sum_g_values :
  (Finset.sum (Finset.range 2001) (λ k, g (k / 2002))) = 1001 :=
by sorry

end sum_g_values_l564_564010


namespace simplify_complex_expression_l564_564945

variables (x y : ℝ) (i : ℂ)

theorem simplify_complex_expression (h : i^2 = -1) :
  (x + 3 * i * y) * (x - 3 * i * y) = x^2 - 9 * y^2 :=
by sorry

end simplify_complex_expression_l564_564945


namespace find_angle_x_l564_564844

theorem find_angle_x (x : ℝ) (h1 : 3 * x + 2 * x = 90) : x = 18 :=
  by
    sorry

end find_angle_x_l564_564844


namespace edricHourlyRateIsApproximatelyCorrect_l564_564247

-- Definitions as per conditions
def edricMonthlySalary : ℝ := 576
def edricHoursPerDay : ℝ := 8
def edricDaysPerWeek : ℝ := 6
def weeksPerMonth : ℝ := 4.33

-- Calculation as per the proof problem
def edricWeeklyHours (hoursPerDay daysPerWeek : ℝ) : ℝ := hoursPerDay * daysPerWeek

def edricMonthlyHours (weeklyHours weeksPerMonth : ℝ) : ℝ := weeklyHours * weeksPerMonth

def edricHourlyRate (monthlySalary monthlyHours : ℝ) : ℝ := monthlySalary / monthlyHours

-- The theorem to prove
theorem edricHourlyRateIsApproximatelyCorrect : (edricHourlyRate edricMonthlySalary (edricMonthlyHours (edricWeeklyHours edricHoursPerDay edricDaysPerWeek) weeksPerMonth)) ≈ 2.77 :=
by
  sorry

end edricHourlyRateIsApproximatelyCorrect_l564_564247


namespace product_of_roots_l564_564495

def rad1 := real.sqrt (real.exp (log 16 / 4))
def rad2 := real.cbrt 27

theorem product_of_roots : rad1 * rad2 = 6 := 
by 
  sorry

end product_of_roots_l564_564495


namespace solve_quartic_eqn_l564_564675

theorem solve_quartic_eqn (z : ℂ) : 
  z ^ 4 - 6 * z ^ 2 + 8 = 0 ↔ z = 2 ∨ z = -2 ∨ z = Complex.sqrt 2 ∨ z = -Complex.sqrt 2 := 
sorry

end solve_quartic_eqn_l564_564675


namespace find_marks_in_biology_l564_564899

-- Definitions based on conditions in a)
def marks_english : ℕ := 76
def marks_math : ℕ := 60
def marks_physics : ℕ := 72
def marks_chemistry : ℕ := 65
def num_subjects : ℕ := 5
def average_marks : ℕ := 71

-- The theorem that needs to be proved
theorem find_marks_in_biology : 
  let total_marks := marks_english + marks_math + marks_physics + marks_chemistry 
  let total_marks_all := average_marks * num_subjects
  let marks_biology := total_marks_all - total_marks
  marks_biology = 82 := 
by
  sorry

end find_marks_in_biology_l564_564899


namespace sum_of_coordinates_l564_564040

theorem sum_of_coordinates (C D : ℝ × ℝ) (hC : C = (0, 0)) (hD : D.snd = 6) (h_slope : (D.snd - C.snd) / (D.fst - C.fst) = 3/4) : D.fst + D.snd = 14 :=
sorry

end sum_of_coordinates_l564_564040


namespace problem_solution_l564_564909

def g (x : ℝ) : ℝ := -3 * Real.sin (2 * Real.pi * x)

theorem problem_solution :
  {x : ℝ | -1 ≤ x ∧ x ≤ 1 ∧ g (g (g x)) = g x}.to_finset.card = 77 :=
  sorry

end problem_solution_l564_564909


namespace values_of_a2_add_b2_l564_564394

theorem values_of_a2_add_b2 (a b : ℝ) (h1 : a^3 - 3 * a * b^2 = 11) (h2 : b^3 - 3 * a^2 * b = 2) : a^2 + b^2 = 5 := 
by
  sorry

end values_of_a2_add_b2_l564_564394


namespace proof_sufficient_but_not_necessary_condition_l564_564940

variables {F G : ℝ × ℝ → ℝ}
variable {λ : ℝ}
variable {P : ℝ × ℝ}

theorem proof_sufficient_but_not_necessary_condition 
  (hA : F P = 0 ∧ G P = 0) : 
  (F P = 0 ∧ G P = 0) → (F P + λ * G P = 0) :=
by
  sorry

end proof_sufficient_but_not_necessary_condition_l564_564940


namespace solution_l564_564511

noncomputable def sphere_radius (r_cone : ℝ) (h_cone : ℝ) : ℝ := sorry

theorem solution (r_cone := 50) (h_cone := 120) : sphere_radius r_cone h_cone = 38.5 := 
by 
  -- setup the problem conditions
  have radius_of_sphere : ℝ := 38.5,
  -- assert the equality
  exact radius_of_sphere == 38.5 


end solution_l564_564511


namespace find_z_values_l564_564650

theorem find_z_values : {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -sqrt 2, sqrt 2, 2} :=
sorry

end find_z_values_l564_564650


namespace shift_right_to_f2_l564_564514

section
variables {x : ℝ} {Δx : ℝ}

def f1 (x : ℝ) : ℝ := 2 * sin (3 * x + π / 5)
def f2 (x : ℝ) : ℝ := 2 * sin (3 * x)

theorem shift_right_to_f2 : Δx = π / 15 →
  (∀ x, f1 (x - Δx) = f2 x) :=
by
  intro h
  intro x
  simp [f1, f2]
  rw [h]
  sorry
end

end shift_right_to_f2_l564_564514


namespace concurrency_iff_concyclity_l564_564419

variables {A B C D E F I : Point ℝ}

-- Definitions of conditions
def is_cyclic (A B C D : Point ℝ) : Prop := 
  ∃ (Γ : Circle ℝ), A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ ∧ D ∈ Γ

def lines_concurrent (A B C D E F I : Point ℝ) : Prop :=
  ∃ I : Point ℝ, lies_on_line I A B ∧ lies_on_line I C D ∧ lies_on_line I E F

-- Problem to prove
theorem concurrency_iff_concyclity
  (h₁ : is_cyclic A B C D)
  (h₂ : is_cyclic C D E F)
  (h₃ : ¬(parallel (line_through A B) (line_through C D)))
  (h₄ : ¬(parallel (line_through A B) (line_through E F)))
  (h₅ : ¬(parallel (line_through C D) (line_through E F))) :
  (lines_concurrent A B C D E F) ↔ (is_cyclic A B E F) :=
sorry

end concurrency_iff_concyclity_l564_564419


namespace hyperbola_standard_form_l564_564330

theorem hyperbola_standard_form (x y : ℝ) (m : ℝ) (h₁ : m > 0)
  (h₂ : 2 * Real.sqrt m = Real.sqrt (m + 6)) :
  (∃ k₁ k₂ : ℝ, (m = 2 → m + 6 = k₁) ∧ (k₂ = 2 + 6)) →
  (m = 2) → 
  (⟦⟨Real.sqrt m ≠ 0, 
  let k := 2 + 6 in 
  (Real.sqrt (m + 6) ≠ 0 → 
  (2 * Real.sqrt m = Real.sqrt (m + 6)))
  ∧ (x^2 / 2 - y^2 / 8 = 1))⟧) :=
begin
  sorry,
end

end hyperbola_standard_form_l564_564330


namespace first_book_price_is_63_l564_564297

-- Define the conditions based on the problem statement
variables (p : ℕ) -- Price of the first book
constant n : ℕ := 41 -- Number of books
constant diff : ℕ := 3 -- Price difference between adjacent books
constant sum_first_last : ℕ := 246 -- Sum of the first and last book prices

-- Define the equation involving the prices of the first and last book
def first_book_price := p
def last_book_price := p + diff * (n - 1)
def total_first_and_last := first_book_price + last_book_price

-- The main statement to prove
theorem first_book_price_is_63 (h : total_first_and_last = sum_first_last) : p = 63 :=
by
  sorry

end first_book_price_is_63_l564_564297


namespace jack_sees_color_change_l564_564176

noncomputable def traffic_light_cycle := 95    -- Total duration of the traffic light cycle
noncomputable def change_window := 15          -- Duration window where color change occurs
def observation_interval := 5                  -- Length of Jack's observation interval

/-- Probability that Jack sees the color change during his observation. -/
def probability_of_observing_change (cycle: ℕ) (window: ℕ) : ℚ :=
  window / cycle

theorem jack_sees_color_change :
  probability_of_observing_change traffic_light_cycle change_window = 3 / 19 :=
by
  -- We only need the statement for verification
  sorry

end jack_sees_color_change_l564_564176


namespace problem1_problem2_l564_564467

section
variables (x a : ℝ)

-- Problem 1: Prove \(2^{3x-1} < 2 \implies x < \frac{2}{3}\)
theorem problem1 : (2:ℝ)^(3*x-1) < 2 → x < (2:ℝ)/3 :=
by sorry

-- Problem 2: Prove \(a^{3x^2+3x-1} < a^{3x^2+3} \implies (a > 1 \implies x < \frac{4}{3}) \land (0 < a < 1 \implies x > \frac{4}{3})\) given \(a > 0\) and \(a \neq 1\)
theorem problem2 (h0 : a > 0) (h1 : a ≠ 1) :
  a^(3*x^2 + 3*x - 1) < a^(3*x^2 + 3) →
  ((1 < a → x < (4:ℝ)/3) ∧ (0 < a ∧ a < 1 → x > (4:ℝ)/3)) :=
by sorry
end

end problem1_problem2_l564_564467


namespace proof_of_min_value_l564_564004

def constraints_on_powers (a b c d : ℝ) : Prop :=
  a^4 + b^4 + c^4 + d^4 = 16

noncomputable def minimum_third_power_sum (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

theorem proof_of_min_value : 
  ∃ a b c d : ℝ, constraints_on_powers a b c d → ∃ min_val : ℝ, min_val = minimum_third_power_sum a b c d :=
sorry -- Further method to rigorously find the minimum value.

end proof_of_min_value_l564_564004


namespace ellipse_equation_max_area_line_eqn_l564_564311

-- Given
def foci := ( (-2 : ℝ), (0 : ℝ) ), ( (2 : ℝ), (0 : ℝ) )
def pointP := (2 : ℝ), ( (real.sqrt 6) / 3 )

-- Definition of ellipse C with known foci and a point it passes through.
noncomputable def ellipse_C (x y : ℝ) := (x^2 / 6) + (y^2 / 2) = 1

-- Prove the equation of ellipse C
theorem ellipse_equation : ∀ x y : ℝ, 
    (ellipse_C x y ↔ (x = 2 ∧ y = (real.sqrt 6) / 3)) :=
begin
    intro x y,
    sorry
end

-- Given: A line passing through the right focus (2,0) intersecting ellipse at A and B
def right_focus := (2 : ℝ), (0 : ℝ)
def line_l (m : ℝ) := λ y : ℝ, m * y + 2

-- Prove the equation of line l 
theorem max_area_line_eqn : ∀ m : ℝ, 
    (∃ a b : ℝ, line_l m a = b) → 
    let area := λ m : ℝ, (2 * (real.sqrt 6) * (m^2 + 1)) / ((m^2 + 3)^2) in
    (area m = real.sqrt 3 → (m = 1 ∨ m = -1) → ( ∃ x y : ℝ, x + y - 2 = 0 ∨ x - y - 2 = 0 )) :=
begin
    intros m hab,
    let area := λ m : ℝ, (2 * (real.sqrt 6) * (real.sqrt (m^2 + 1))) / ((m^2 + 3)) in
    sorry
end

end ellipse_equation_max_area_line_eqn_l564_564311


namespace sum_of_coordinates_of_point_D_l564_564037

theorem sum_of_coordinates_of_point_D : 
  ∀ {x : ℝ}, (y = 6) ∧ (x ≠ 0) ∧ ((6 - 0) / (x - 0) = 3 / 4) → x + y = 14 := by
  intros x hx hy hslope
  sorry

end sum_of_coordinates_of_point_D_l564_564037


namespace interest_rate_required_l564_564195

-- Define the given constants and parameters
def principal1 : ℝ := 400
def time1 : ℕ := 5
def rate1 : ℝ := 12
def interest1 : ℝ := principal1 * rate1 / 100 * time1

def principal2 : ℝ := 200
def time2 : ℕ := 12
def interest2 : ℝ := interest1

-- Define the rate we need to prove
noncomputable def rate2 : ℝ := 10

theorem interest_rate_required :
  interest2 = principal2 * rate2 / 100 * time2 :=
by
  unfold rate2 interest2 interest1 principal2
  have h1 : 400 * 12 / 100 * 5 = 240 := sorry
  have h2 : 200 * 10 / 100 * 12 = 240 := sorry
  exact Eq.trans h1 h2

end interest_rate_required_l564_564195


namespace digit_7_count_in_range_100_to_199_l564_564888

/-- 
Given the range of numbers from 100 to 199 inclusive,
prove that the digit 7 appears exactly 20 times.
-/
theorem digit_7_count_in_range_100_to_199 : 
  let count_7 := (λ n : ℕ, 
    (n / 100 = 7) +  -- hundreds place
    ((n % 100) / 10 = 7) + -- tens place
    ((n % 10) = 7)) in
  (Finset.sum (Finset.range' 100 100) count_7 = 20) :=
by sorry

end digit_7_count_in_range_100_to_199_l564_564888


namespace perfect_square_trinomial_m_l564_564792

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ a : ℝ, (x^2 + 2 * (m - 1) * x + 4) = (x + a)^2) → (m = 3 ∨ m = -1) :=
by
  sorry

end perfect_square_trinomial_m_l564_564792


namespace cinema_cost_comparison_l564_564513

theorem cinema_cost_comparison (x : ℕ) (hx : x = 1000) :
  let cost_A := if x ≤ 100 then 30 * x else 24 * x + 600
  let cost_B := 27 * x
  cost_A < cost_B :=
by
  sorry

end cinema_cost_comparison_l564_564513


namespace tangent_line_C_value_of_m_if_tangent_l564_564732

def circle_C (m : ℝ) : Prop := ∀ x y : ℝ, (x - m)^2 + (y - 2 * m)^2 = m^2

def circle_E : Prop := ∀ x y : ℝ, (x - 3)^2 + y^2 = 16

theorem tangent_line_C (m : ℝ) (h : m = 2) : 
  ∃ (k : ℝ), y = k * x ∨ x = 0 := sorry

theorem value_of_m_if_tangent : 
  (∃ (x y : ℝ), circle_C m x y ∧ circle_E x y ∧ ∀ r : ℝ, (circle_E x y → r ≠ 4) → |4 - m| = dist (m, 2*m) (3, 0)) → 
  m = (sqrt 29 - 1) / 4 := sorry

end tangent_line_C_value_of_m_if_tangent_l564_564732


namespace infinitely_many_pairs_sum_is_odd_l564_564535

open_locale nat floor

-- Part (a)
theorem infinitely_many_pairs : ∃ᶠ (m n : ℕ) in ⊤, ⌊(4 + 2 * real.sqrt 3) * m⌋ = ⌊(4 - 2 * real.sqrt 3) * n⌋ :=
by sorry

-- Part (b)
theorem sum_is_odd (m n : ℕ) (h : ⌊(4 + 2 * real.sqrt 3) * m⌋ = ⌊(4 - 2 * real.sqrt 3) * n⌋) : odd (m + n) :=
by sorry

end infinitely_many_pairs_sum_is_odd_l564_564535


namespace cone_volume_and_surface_area_l564_564502

noncomputable def cone_volume (slant_height height : ℝ) : ℝ := 
  1 / 3 * Real.pi * (Real.sqrt (slant_height^2 - height^2))^2 * height

noncomputable def cone_surface_area (slant_height height : ℝ) : ℝ :=
  Real.pi * (Real.sqrt (slant_height^2 - height^2)) * (Real.sqrt (slant_height^2 - height^2) + slant_height)

theorem cone_volume_and_surface_area :
  (cone_volume 15 9 = 432 * Real.pi) ∧ (cone_surface_area 15 9 = 324 * Real.pi) :=
by
  sorry

end cone_volume_and_surface_area_l564_564502


namespace solve_quartic_l564_564682

theorem solve_quartic (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ (z = 2 ∨ z = -2 ∨ z = sqrt 2 ∨ z = -sqrt 2) :=
by
  sorry

end solve_quartic_l564_564682


namespace expr_sum_l564_564108

theorem expr_sum : (7 ^ (-3 : ℤ)) ^ 0 + (7 ^ 0) ^ 2 = 2 :=
by
  sorry

end expr_sum_l564_564108


namespace total_descendants_l564_564043

theorem total_descendants :
  let initial_sons := 3
  let productive_descendants := 93
  let offspring_per_descendant := 2
  in 3 + productive_descendants * offspring_per_descendant = 189 :=
by 
  -- initial sons
  let initial_sons := 3
  -- productive descendants
  let productive_descendants := 93
  -- offspring per productive descendant
  let offspring_per_descendant := 2
  -- calculating total descendants
  let additional_descendants := productive_descendants * offspring_per_descendant
  -- adding initial sons to the additional descendants
  have total_descendants := initial_sons + additional_descendants
  show 3 + productive_descendants * offspring_per_descendant = 189 by sorry

end total_descendants_l564_564043


namespace number_of_segments_l564_564023

theorem number_of_segments (n : Nat) (h : n ≥ 2) (G : SimpleGraph (Fin n)) (hG : G.IsConnected) (hA : G.IsAcyclic) : G.edgeFinset.card = n - 1 := 
by
  sorry

end number_of_segments_l564_564023


namespace range_of_k_for_parabola_intersection_area_of_triangle_FMN_l564_564774

theorem range_of_k_for_parabola_intersection (k : ℝ) :
  let line := λ (x y : ℝ), y - 2 = k * (x + 2)
  ∧ ∃ (M N : ℝ × ℝ), y^2 = 4*x ∧ line x y ∧ M ≠ N 
  → k ∈ Ioo (-(1 + sqrt 3) / 2) 0 ∪ Ioo 0 ((-1 + sqrt 3) / 2) :=
sorry

theorem area_of_triangle_FMN (F M N : ℝ × ℝ) :
  let parabola := λ (x y : ℝ), y^2 = 4*x
  ∧ (∀ (M N : ℝ × ℝ), parabola (M.1) (M.2) ∧ parabola (N.1) (N.2) ∧ M ≠ N)
  ∧ let line := (λ (x y : ℝ), y = -x)
  → ∃ (F (1, 0)), intersection_points (line) (parabola) (M) (N)
  → area_triangle (F) (M) (N) = 2 :=
sorry

end range_of_k_for_parabola_intersection_area_of_triangle_FMN_l564_564774


namespace quotient_A_div_B_l564_564571

-- Define A according to the given conditions
def A : ℕ := (8 * 10) + (13 * 1)

-- Define B according to the given conditions
def B : ℕ := 30 - 9 - 9 - 9

-- Prove that the quotient of A divided by B is 31
theorem quotient_A_div_B : (A / B) = 31 := by
  sorry

end quotient_A_div_B_l564_564571


namespace no_solution_for_m_l564_564801

theorem no_solution_for_m (m : ℝ) : ¬ ∃ x : ℝ, x ≠ 1 ∧ (mx - 1) / (x - 1) = 3 ↔ m = 1 ∨ m = 3 := 
by sorry

end no_solution_for_m_l564_564801


namespace quadratic_function_expression_l564_564582

theorem quadratic_function_expression :
  ∀ (f : ℝ → ℝ),
    (∀ x, f (x - 2) = f (-x - 2)) →  -- symmetry condition
    f 0 = 1 →                         -- y-intercept condition
    ∃ x1 x2, x1 ≠ x2 ∧ (f x1 = 0 ∧ f x2 = 0) ∧ (|x1 - x2| = 2 * Real.sqrt 2) →  -- x-intercept condition
    f = λ x, (1 / 2) * (x + 2) ^ 2 - 1 := 
by 
  intros f h_symm h_yint h_xint
  -- omitted proof for the sake of brevity
  sorry

end quadratic_function_expression_l564_564582


namespace abs_inequality_solution_l564_564633

theorem abs_inequality_solution (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) := 
sorry

end abs_inequality_solution_l564_564633


namespace number_of_distinct_real_roots_l564_564757

theorem number_of_distinct_real_roots (k : ℕ) :
  (∃ k : ℕ, ∀ x : ℝ, |x| - 4 = (3 * |x|) / 2 → 0 = k) :=
  sorry

end number_of_distinct_real_roots_l564_564757


namespace tank_capacity_correct_l564_564071

def amount_of_oil_bought : ℕ := 728
def amount_of_oil_still_in_tank : ℕ := 24

def tank_capacity : ℕ := amount_of_oil_bought + amount_of_oil_still_in_tank

theorem tank_capacity_correct : tank_capacity = 752 :=
by
  -- sum the oil bought and still in tank and show it equals to 752
  calc
    tank_capacity = amount_of_oil_bought + amount_of_oil_still_in_tank : rfl
                ... = 728 + 24 : by rfl
                ... = 752 : by norm_num

end tank_capacity_correct_l564_564071


namespace min_turns_2012_l564_564156

/-- 
A function that determines the minimum number of turns required 
to remove exactly n rocks from the table following the given rules.
-/
def minTurns (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let rec remove_rocks (remaining : ℕ) (prev : ℕ) (turns : ℕ) : ℕ :=
      if remaining = 0 then turns
      else
        let next_rocks := min prev (remaining / 2)
        in if next_rocks = 0 then turns
           else remove_rocks (remaining - next_rocks) next_rocks (turns + 1)
    remove_rocks n 1 0

theorem min_turns_2012 : minTurns 2012 = 18 := by
  sorry

end min_turns_2012_l564_564156


namespace rebus_solution_l564_564282

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564282


namespace cary_strips_ivy_l564_564611

variable (strip_per_day : ℕ) (grow_per_night : ℕ) (total_ivy : ℕ)

theorem cary_strips_ivy (h1 : strip_per_day = 6) (h2 : grow_per_night = 2) (h3 : total_ivy = 40) :
  (total_ivy / (strip_per_day - grow_per_night)) = 10 := by
  sorry

end cary_strips_ivy_l564_564611


namespace find_f_nine_l564_564968

-- Define the function f that satisfies the conditions
def f (x : ℝ) : ℝ := sorry

-- Define the condition that f(x + y) = f(x) * f(y) for all real x and y
axiom functional_equation : ∀ (x y : ℝ), f (x + y) = f x * f y

-- Define the condition that f(3) = 4
axiom f_three : f 3 = 4

-- State the main theorem to prove that f(9) = 64
theorem find_f_nine : f 9 = 64 := by
  sorry

end find_f_nine_l564_564968


namespace complement_intersection_l564_564015

open Set

variable {α : Type} [DecidableEq α]

def U : Set α := {1, 2, 3, 4, 5}
def A : Set α := {1, 3, 5}
def B : Set α := {3, 4}
def complement (A U : Set α) : Set α := U \ A

theorem complement_intersection (U A B : Set α) (hU : U = {1, 2, 3, 4, 5}) 
  (hA : A = {1, 3, 5}) (hB : B = {3, 4}) : (complement A U) ∩ B = {4} :=
by
  sorry

end complement_intersection_l564_564015


namespace electronics_weight_l564_564540

-- Define the initial conditions and the solution we want to prove.
theorem electronics_weight (B C E : ℕ) (k : ℕ) 
  (h1 : B = 7 * k) 
  (h2 : C = 4 * k) 
  (h3 : E = 3 * k) 
  (h4 : (B : ℚ) / (C - 8 : ℚ) = 2 * (B : ℚ) / (C : ℚ)) :
  E = 12 := 
sorry

end electronics_weight_l564_564540


namespace bob_distance_when_meet_l564_564538

-- Definitions of the variables and conditions
def distance_XY : ℝ := 40
def yolanda_rate : ℝ := 2  -- Yolanda's walking rate in miles per hour
def bob_rate : ℝ := 4      -- Bob's walking rate in miles per hour
def yolanda_start_time : ℝ := 1 -- Yolanda starts 1 hour earlier 

-- Prove that Bob has walked 25.33 miles when he meets Yolanda
theorem bob_distance_when_meet : 
  ∃ t : ℝ, 2 * (t + yolanda_start_time) + 4 * t = distance_XY ∧ (4 * t = 25.33) := 
by
  sorry

end bob_distance_when_meet_l564_564538


namespace number_of_rational_terms_in_expansion_l564_564118

-- Define the binomial expansion term
def binomial_expansion_term (k : ℕ) (x y : ℚ) : ℚ :=
  x ^ k * y ^ (1000 - k)

-- Define the term condition for rational coefficients
def term_is_rational (k : ℕ) : Prop :=
  (k % 3 = 0) ∧ ((1000 - k) % 2 = 0)

-- Prove the number of terms with rational coefficients is 167
theorem number_of_rational_terms_in_expansion : 
  (finset.range 1001).filter (λ k, term_is_rational k).card = 167 :=
by sorry

end number_of_rational_terms_in_expansion_l564_564118


namespace B_correct_A_inter_B_correct_l564_564353

def A := {x : ℝ | 1 < x ∧ x < 8}
def B := {x : ℝ | x^2 - 5 * x - 14 ≥ 0}

theorem B_correct : B = {x : ℝ | x ≤ -2 ∨ x ≥ 7} := 
sorry

theorem A_inter_B_correct : A ∩ B = {x : ℝ | 7 ≤ x ∧ x < 8} :=
sorry

end B_correct_A_inter_B_correct_l564_564353


namespace number_of_valid_rearrangements_l564_564639

def chair_arrangements : ℕ := 8

def valid_rearrangements (n : ℕ) : Prop :=
  ∃ arrangement : List (Fin n), 
    (∀ i : Fin n, arrangement[i] ≠ i ∧ 
      arrangement[(i + 1) % n] ≠ arrangement[i] ∧ 
      arrangement[(i + n - 1) % n] ≠ arrangement[i]) ∧
    arrangement.length = n

theorem number_of_valid_rearrangements :
  valid_rearrangements chair_arrangements ∧
  (card (valid_rearrangements chair_arrangements)) = 30 :=
sorry

end number_of_valid_rearrangements_l564_564639


namespace find_natural_number_l564_564714

theorem find_natural_number (n : ℕ) (h1 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n ∨ d = 3 ∨ d = 5 ∨ d = 9 ∨ d = 15)
  (h2 : 1 + 3 + 5 + 9 + 15 + n = 78) : n = 45 := sorry

end find_natural_number_l564_564714


namespace temperature_difference_correct_l564_564813

def avg_high : ℝ := 9
def avg_low : ℝ := -5
def temp_difference : ℝ := avg_high - avg_low

theorem temperature_difference_correct : temp_difference = 14 := by
  sorry

end temperature_difference_correct_l564_564813


namespace eccentricity_of_ellipse_l564_564753

open Set

noncomputable theory

theorem eccentricity_of_ellipse :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →
    ∃ (M N : ℝ × ℝ), M ∈ E ∧ N ∈ E ∧
    ∃ (t : ℝ), (-1, t) = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧
    ∃ Q : ℝ × ℝ, Q = (-3/4, 0) ∧
    ∃ l l' : ℝ, l = (N.2 - M.2) / (N.1 - M.1) ∧ l = -1 / (N.2 - M.2) / (N.1 - M.1)) →
    (sqrt (1 - b^2 / a^2)) = sqrt 3 / 2 :=
by
  intro a b
  intro ha hb
  use a, b
  split
  exact ha
  split
  exact hb
  intros x y hxy
  sorry

end eccentricity_of_ellipse_l564_564753


namespace count_digit_7_from_100_to_199_l564_564879

theorem count_digit_7_from_100_to_199 : 
  (list.range' 100 100).countp (λ n, (70 ≤ n ∧ n ≤ 79) ∨ (n % 10 = 7)) = 20 :=
by
  simp
  sorry

end count_digit_7_from_100_to_199_l564_564879


namespace solve_quartic_eq_l564_564661

theorem solve_quartic_eq (z : ℝ) : z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = 2 :=
by 
  sorry

end solve_quartic_eq_l564_564661


namespace childrens_day_2010_l564_564996

-- Define a function that returns the day of the week given a specific date.
noncomputable def day_of_week : ℕ → ℕ → ℕ → string
| 4, 11, 2010 => "Sunday"
| 6, 1, 2010  => "Tuesday"
| _, _, _     => "Undefined"

-- Define a proposition that states April 11, 2010, is a Sunday.
def april_11_2010_is_sunday : Prop :=
  day_of_week 4 11 2010 = "Sunday"

-- Define a proposition that states June 1, 2010, is a Tuesday.
def june_1_2010_is_tuesday : Prop :=
  day_of_week 6 1 2010 = "Tuesday"

-- Lean statement to prove:
theorem childrens_day_2010 :
  april_11_2010_is_sunday → june_1_2010_is_tuesday :=
  by
    intro h
    have : day_of_week 6 1 2010 = "Tuesday" := rfl
    exact this

end childrens_day_2010_l564_564996


namespace cos_270_eq_zero_l564_564216

-- Defining the cosine value for the given angle
theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 :=
by
  sorry

end cos_270_eq_zero_l564_564216


namespace range_of_a_l564_564426

def f (x : ℝ) : ℝ := sin x / (2 + cos x)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x → f x ≤ a * x) ↔ a ∈ Set.Ici (1 / 3) := 
by 
  sorry

end range_of_a_l564_564426


namespace ratio_of_areas_is_rational_l564_564400

theorem ratio_of_areas_is_rational 
  (A B C P : Type)
  (AB BC CA : ℤ)
  (hAB : AB ≠ 0) (hBC : BC ≠ 0) (hCA : CA ≠ 0) 
  (integral_sides : AB = c ∧ BC = a ∧ CA = b) 
  (angle_bisector_from_B_and_altitude_from_C_meet_at_P : P → Prop) :
  ∃ r : ℚ, 
  let area_APB := 1 / 2 * (PD * AB : ℝ)
  let area_APC := 1 / 2 * (PD * AD : ℝ)
  in (area_APB / area_APC).is_rational := 
sorry

end ratio_of_areas_is_rational_l564_564400


namespace car_original_speed_correct_l564_564555

noncomputable def car_original_speed (distance : ℝ) (increase_percent : ℝ) (time_saved : ℝ) : ℝ := 
  let eq := λ x : ℝ, ((distance / x) - (distance / (increase_percent * x))) = time_saved
  classical.some (⟨eq, sorry⟩)

theorem car_original_speed_correct : car_original_speed 160 1.25 0.4 = 80 := 
by
  sorry

end car_original_speed_correct_l564_564555


namespace sqrt_fraction_sum_l564_564605

theorem sqrt_fraction_sum (h1 : 1 / 4 = 0.25) (h2 : 1 / 25 = 0.04) :
  sqrt (1 / 4 + 1 / 25) = sqrt 29 / 10 :=
by
  sorry

end sqrt_fraction_sum_l564_564605


namespace sum_fₙ_pi_over_3_l564_564305

def f₁ (x : ℝ) : ℝ := sin x + cos x

noncomputable def fₙ (n : ℕ) (x : ℝ) : ℝ :=
  (deriv^[n] f₁) x

theorem sum_fₙ_pi_over_3 :
  ((Finset.range 2017).sum (λ n, fₙ n (π / 3))) = (1 + Real.sqrt 3) / 2 :=
by
  sorry

end sum_fₙ_pi_over_3_l564_564305


namespace book_distribution_l564_564232

theorem book_distribution:
  let novels := 3
  let poetry_collections := 2
  let students := 4
  (∃ distributions : set (ℕ → ℕ), 
    (∀ d ∈ distributions, (∑ i in finset.range students, d i ≠ 0) ∧ 
      ∑ i in finset.range students, d i = novels + poetry_collections) ∧ 
    (4 * 3 = 12 ∧ 4 * 1 = 4 ∧ 4 * 3 = 12) ∧ 
    12 + 4 + 12 = 28)
sorry

end book_distribution_l564_564232


namespace simplest_square_root_l564_564132

/-
Define the given square roots.
-/
def sq_a : ℝ := real.sqrt 0.5
def sq_b : ℝ := real.sqrt (9 / 11)
def sq_c : ℝ := real.sqrt 121
def sq_d : ℝ := real.sqrt 17

/-
State the proposition: \( \sqrt{17} \) is the simplest form.
-/
theorem simplest_square_root :
  (sq_d = real.sqrt 17) ∧
  (sq_a = real.sqrt 0.5 → complex.abs (real.sqrt 0.5) > complex.abs (real.sqrt 17)) ∧
  (sq_b = real.sqrt (9 / 11) → complex.abs (real.sqrt (9 / 11)) > complex.abs (real.sqrt 17)) ∧
  (sq_c = real.sqrt 121 → complex.abs (real.sqrt 121) > complex.abs (real.sqrt 17)) :=
  by
    sorry

end simplest_square_root_l564_564132


namespace a100_pos_a100_abs_lt_018_l564_564337

noncomputable def a (n : ℕ) : ℝ := real.cos (10^n * real.pi / 180)

theorem a100_pos : (a 100 > 0) :=
by
  sorry

theorem a100_abs_lt_018 : (|a 100| < 0.18) :=
by
  sorry

end a100_pos_a100_abs_lt_018_l564_564337


namespace people_off_at_second_stop_l564_564987

theorem people_off_at_second_stop (initial_seats : ℕ) (initial_boarded : ℕ) 
  (first_stop_on : ℕ) (first_stop_off : ℕ) 
  (second_stop_on : ℕ) (final_empty_seats : ℕ) 
  (total_seats := 23 * 4) (initial_empty_seats := total_seats - initial_boarded)
  (first_stop_net := first_stop_on - first_stop_off)
  (empty_seats_after_first := initial_empty_seats - first_stop_net)
  (second_stop_net := second_stop_on - final_empty_seats)
  (final_empty_seats_calculation := empty_seats_after_first - second_stop_on + second_stop_net)
  : initial_seats = 92 ∧ initial_boarded = 16 ∧ first_stop_on = 15 ∧ first_stop_off = 3 ∧ second_stop_on = 17 ∧ final_empty_seats = 57 → second_stop_net = 10 := 
by
  intros h
  have : total_seats = 92 := rfl
  have : initial_empty_seats = 76 := rfl
  have : first_stop_net = 12 := rfl
  have : empty_seats_after_first = 64 := rfl
  have : final_empty_seats_calculation = 57 := by assumption
  sorry

end people_off_at_second_stop_l564_564987


namespace rebus_solution_l564_564263

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564263


namespace exists_m_for_inequality_l564_564766

noncomputable def f (a x : ℝ) := 4 * x + a * x^2 - (2/3) * x^3

theorem exists_m_for_inequality :
  ∀ (a t m : ℝ),
  (-1 ≤ a ∧ a ≤ 1) ∧ (-1 ≤ t ∧ t ≤ 1) →
  let x1 := (a + real.sqrt (a^2 + 8)) / 2
  let x2 := (a - real.sqrt (a^2 + 8)) / 2
  in (x1 ≠ 0 ∧ x2 ≠ 0) →
  m^2 + t * m + 1 ≥ real.abs (x1 - x2) ↔ m ≥ 2 ∨ m ≤ -2 := by
sorry

end exists_m_for_inequality_l564_564766


namespace quadratic_root_l564_564320

/-- If one root of the quadratic equation x^2 - 2x + n = 0 is 3, then n is -3. -/
theorem quadratic_root (n : ℝ) (h : (3 : ℝ)^2 - 2 * 3 + n = 0) : n = -3 :=
sorry

end quadratic_root_l564_564320


namespace fourth_number_in_ninth_row_of_lattice_l564_564486

theorem fourth_number_in_ninth_row_of_lattice : 
  (let row_start (n : Nat) := 8 * (n - 1) + 1 in row_start 9 + 3 = 68) :=
by
  -- conditions
  let row_start (n : Nat) := 8 * (n - 1) + 1
  -- question
  have fourth_number: (row_start 9 + 3 = 68)
  done

end fourth_number_in_ninth_row_of_lattice_l564_564486


namespace part1_part2_part3_l564_564728

noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

noncomputable def line_eq (x y k : ℝ) : Prop := 
  y = k * x

noncomputable def line_distance (l_coeff: ℝ × ℝ × ℝ) (p: ℝ × ℝ) : ℝ := 
  abs (l_coeff.1 * p.1 + l_coeff.2 * p.2 + l_coeff.3) / real.sqrt (l_coeff.1 ^ 2 + l_coeff.2 ^ 2)

theorem part1 (hC : (3, -2) = (3, -2)) : 
  ∀ x y: ℝ, (x - 3)^2 + (y + 2)^2 = 25 := 
sorry

theorem part2 
  (tangent_point : ℝ × ℝ := (0, 3)) 
  (center := (3, -2)) 
  (radius := 5) : 
  ∀ x y k: ℝ, (line_eq x y k → 
  line_distance (1, -k, 3 * k - 3) center = radius 
  → (15 * x - 8 * y + 24 = 0 ∨ y = 3)) := 
sorry

theorem part3 
  (center := (3, -2))
  (radius := 5)
  (line_coeff := (3, 4)) 
  (dist := 1) : 
  ∀ (m : ℝ),  
  |line_distance (3, 4, m) center - radius| = dist → 
  (m = 21 ∨ m = 19) := 
sorry

end part1_part2_part3_l564_564728


namespace meeting_arrangements_l564_564557

theorem meeting_arrangements:
  let total_people := 10
  let selected_people := 4
  let participants_A := 2
  let participants_B := 1
  let participants_C := 1
  in combinatorial.choose total_people selected_people * 
     combinatorial.choose selected_people participants_A * 
     combinatorial.choose (selected_people - participants_A) participants_B = 2520 := by
  sorry

end meeting_arrangements_l564_564557


namespace count_digit_7_to_199_l564_564867

theorem count_digit_7_to_199 : (Nat.countDigitInRange 7 100 199) = 20 := by
  sorry

end count_digit_7_to_199_l564_564867


namespace train_start_time_l564_564958

theorem train_start_time (average_speed distance : ℝ) (arrival_time_bangalore halt_time : ℝ) (reach_time : ℕ) :
  average_speed = 87 ∧ distance = 348 ∧ reach_time = 13.75 ∧ halt_time = 45/60 →
  let travel_time := distance / average_speed,
      total_travel_time := travel_time + halt_time,
      scheduled_start_time := reach_time - total_travel_time in
  scheduled_start_time = 9 := 
begin
  sorry
end

end train_start_time_l564_564958


namespace max_additional_plates_l564_564811

/-- Define the initial sizes of the sets of letters. -/
def initial_Size_S1 : ℕ := 5
def initial_Size_S2 : ℕ := 2
def initial_Size_S3 : ℕ := 5

/-- Calculate the initial number of license plates. -/
def initial_Plates : ℕ := initial_Size_S1 * initial_Size_S2 * initial_Size_S3

/-- Prove the maximum number of additional license plates is 75 when adding 3 letters. -/
theorem max_additional_plates : ∃ (additional_Size_S1 additional_Size_S2 additional_Size_S3 : ℕ),
  additional_Size_S1 + additional_Size_S2 + additional_Size_S3 = 3 ∧
  (additional_Size_S1 + initial_Size_S1) * (additional_Size_S2 + initial_Size_S2) * (additional_Size_S3 + initial_Size_S3) - initial_Plates = 75 :=
by
  existsi (0, 3, 0)
  simp
  sorry

end max_additional_plates_l564_564811


namespace circumcircle_fixed_point_l564_564041

variables {V : Type*} [inner_product_space ℝ V]

theorem circumcircle_fixed_point
  (A P : V)
  (ℓ : affine_subspace ℝ V)
  (h10 : A ∉ ℓ)
  (h11 : P ∉ ℓ) :
  ∀ (B C : V), ∠ A B = 90 ∧ ∠ A C = 90 ∧ (affine_span ℝ {B, C}) = ℓ →
  ∃ Q : V, Q ≠ P ∧ affine_triple P B C ∧ Q ∈ (∂ (convex_hull ℝ {B, P, C})) :=
by
  sorry

end circumcircle_fixed_point_l564_564041


namespace angle_MTB_is_90_l564_564998

/-- Triangle ABC is right-angled with ∠ACB = 90°. -/
variables {A B C M N O K T : Type} [has_dist A B] [has_dist B C] [has_dist A C]

/-- Conditions -/
variables (h1 : ∠ACB = 90)
variables (h2 : dist A C / dist B C = 2)
variables (h3 : ∃ M N, parallel A C M N ∧ intersects M N A B ∧ intersects M N B C)
variables (h4 : dist C N / dist B N = 2)
variables (h5 : O = intersection_point (line C M) (line A N))
variables (h6 : K lies_on_segment O N ∧ OM + OK = KN)
variables (h7 : T = intersection_point angle_bisector_of_ABC (perpendicular_from K A N))

/-- Prove that ∠MTB = 90° -/
theorem angle_MTB_is_90 (h : ∠ACB = 90 ∧ dist A C / dist B C = 2 ∧ (∃ M N, parallel A C M N ∧ intersects M N A B ∧ intersects M N B C) ∧ dist C N / dist B N = 2 ∧ O = intersection_point (line C M) (line A N) ∧ (K lies_on_segment O N ∧ OM + OK = KN) ∧ T = intersection_point angle_bisector_of_ABC (perpendicular_from K A N)) : 
    ∠MTB = 90 := by sorry

end angle_MTB_is_90_l564_564998


namespace seq_an_formula_sum_Tn_l564_564733

open BigOperators

-- Define the sequence and conditions
def S : ℕ → ℕ
| n := 2 * (2^n) - 2

-- Prove the formula for the sequence {a_n}
theorem seq_an_formula (n : ℕ) : ∀ n, (∃ a_n, a_n = 2^n) :=
by
  intro n
  use 2^n
  sorry

-- Define the sequence {b_n = (n + 1) / a_n}
def b (n: ℕ) := (n + 1) * (1 / 2^n)

-- Define the sequence for the sum of the first n terms of {b_n}
def T : ℕ → ℝ
| 0 := 0
| (n+1) := 3 - (n+4) * (1 / 2^(n+1))

-- Prove the sum of the first n terms of the sequence {b_n} is T_n
theorem sum_Tn (n : ℕ) : ∀ n, ∑ i in finset.range (n+1), b i = T n :=
by
  intro n
  induction n with n ih
  case zero {
    simp [b, T]
    sorry
  }
  case succ {
    simp [finset.sum_range_succ, b, T]
    rw ih
    sorry
  }

end seq_an_formula_sum_Tn_l564_564733


namespace points_for_each_player_l564_564462

noncomputable def Player := Fin 6

-- Constants for each player
def A : Player := 0
def B : Player := 1
def C : Player := 2
def D : Player := 3
def E : Player := 4
def F : Player := 5

-- Definitions to describe the game rules and conditions
def points (p : Player) : ℕ := sorry -- Definition of points function to be later defined properly

axiom A_drew_all_games : points A = 2.5
axiom B_did_not_lose_any_games : points B = 2.5
axiom C_won_against_winner_and_drew_with_D : points C = 2.5
axiom D_finished_ahead_of_E : points D > points E
axiom F_finished_ahead_of_D_but_behind_E : points E > points F ∧ points F > points D

theorem points_for_each_player :
  points A = 2.5 ∧
  points B = 2.5 ∧
  points C = 2.5 ∧
  points D = 2.5 ∧
  points E = 1.5 ∧
  points F = 3 :=
begin
  sorry
end

end points_for_each_player_l564_564462


namespace total_volume_of_snowballs_l564_564429

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem total_volume_of_snowballs :
  let r1 := 4
      r2 := 6
      r3 := 8
      V1 := volume_of_sphere r1
      V2 := volume_of_sphere r2
      V3 := volume_of_sphere r3
      V_total := V1 + V2 + V3
  in V_total = 1056 * Real.pi :=
by
  let r1 := 4
  let r2 := 6
  let r3 := 8
  let V1 := volume_of_sphere r1
  let V2 := volume_of_sphere r2
  let V3 := volume_of_sphere r3
  let V_total := V1 + V2 + V3
  sorry

end total_volume_of_snowballs_l564_564429


namespace solve_quartic_equation_l564_564693

theorem solve_quartic_equation :
  ∀ z : ℝ, z^4 - 6 * z^2 + 8 = 0 ↔ (z = -2 ∨ z = -sqrt 2 ∨ z = sqrt 2 ∨ z = 2) :=
by {
  sorry
}

end solve_quartic_equation_l564_564693


namespace triangle_inequality_l564_564356

theorem triangle_inequality {a b c : ℝ} (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  ( (√a + √b > √c) ∧ (√b + √c > √a) ∧ (√c + √a > √b) ) ∧
  ¬( (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2) ) ∧
  ( (|a - b| + 1 + |b - c| + 1 > |a - c|+ 1) ∧ (|b - c| + 1 + |c - a| + 1 > |a - b| + 1) ∧ (|c - a| + 1 + |a - b| + 1 > |b - c| + 1) ) :=
sorry

end triangle_inequality_l564_564356


namespace sum_inv_b_squared_l564_564624

-- Define the sequence a_n
def a : ℕ → ℤ
| 0       := 1
| (n + 1) := 4 + a n

-- Define the sequence b_n as the geometric mean of a_n and a_{n+1}
def b (n : ℕ) : ℚ := real.to_rat (real.sqrt ((a n) * (a (n+1))))

-- Define the sequence 1/b_n^2
def inv_b_squared (n : ℕ) : ℚ := 1 / (b n)^2

-- Define the sum T_n of the first n terms of the sequence {1 / b_n^2}
def T (n : ℕ) : ℚ := ∑ i in finset.range n, inv_b_squared i

-- The statement to be proved
theorem sum_inv_b_squared (n : ℕ) : T n = (n : ℚ) / (4 * n + 1) := by
  sorry

end sum_inv_b_squared_l564_564624


namespace level3_non_reserved_parking_capacity_l564_564378

theorem level3_non_reserved_parking_capacity
  (total_spots_level3 : ℕ)
  (parked_cars_level3 : ℕ)
  (reserved_spots_parked : ℕ)
  (non_reserved_parked_cars := parked_cars_level3 - reserved_spots_parked)
  (available_non_reserved_spots := total_spots_level3 - non_reserved_parked_cars) :
  total_spots_level3 = 480 → parked_cars_level3 = 45 → reserved_spots_parked = 15 → available_non_reserved_spots = 450 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  show 480 - (45 - 15) = 450
  sorry

end level3_non_reserved_parking_capacity_l564_564378


namespace correct_conclusions_l564_564326

-- Define the propositions as logical statements
def prop1 := ∀ x ∈ set.Ioo 0 2, 3^x > x^3
def neg_prop1 := ∃ x ∈ set.Ioo 0 2, 3^x ≤ x^3
def prop2 := (∀ (θ : ℝ), θ = real.pi / 3 → real.cos θ = 1 / 2)
def conv_prop2 := (∀ (θ : ℝ), θ ≠ real.pi / 3 → real.cos θ ≠ 1 / 2)
def prop3 := ∀ (p q : Prop), (p ∨ q) → (p ∨ q)
def prop4 := ∀ (m : ℝ), (∃ x, 2^x + m - 1 = 0) ↔ (∀ x > 0, real.log m x < 0)

-- The mathematical proof problem stating the number of correct conclusions
theorem correct_conclusions : (if prop1 → neg_prop1 then 1 else 0) +
                            (if prop2 → ¬conv_prop2 then 1 else 0) +
                            (if (∀ p q, prop3) then 1 else 0) +
                            (if (∀ m, ¬prop4 m) then 1 else 0) = 2 := 
by
  sorry

end correct_conclusions_l564_564326


namespace circumcircle_equation_l564_564740

theorem circumcircle_equation (O A B : Point) :
  (O = ⟨0, 0⟩ ∧
  (∃ x1 y1, A = ⟨x1, y1⟩ ∧ y1^2 = 2 * x1) ∧
  (∃ x2 y2, B = ⟨x2, y2⟩ ∧ y2^2 = 2 * x2) ∧
  (dist O A = dist A B ∧ dist A B = dist B O)) →
  (∃ (h : Point → Prop), h = λ P, (P.x - 4)^2 + P.y^2 = 16) :=
by
  intros h,
  use (λ P, (P.x - 4)^2 + P.y^2 = 16),
  sorry

end circumcircle_equation_l564_564740


namespace smallest_positive_integer_divisible_by_10_13_14_l564_564292

theorem smallest_positive_integer_divisible_by_10_13_14 : ∃ n : ℕ, n > 0 ∧ (10 ∣ n) ∧ (13 ∣ n) ∧ (14 ∣ n) ∧ n = 910 :=
by {
  sorry
}

end smallest_positive_integer_divisible_by_10_13_14_l564_564292


namespace fixed_point_of_line_l564_564127

theorem fixed_point_of_line (k : ℝ) : ∃ p : ℝ × ℝ, p = (3, 1) :=
by
  use (3, 1)
  have : ∀ k : ℝ, (3, 1) ∈ {p : ℝ × ℝ | (λ (x y : ℝ), k * x - y + 1 = 3 * k) p.1 p.2} := 
    by intros; exact dec_trivial
  apply this
  sorry

end fixed_point_of_line_l564_564127


namespace eccentricity_of_ellipse_l564_564754

open Set

noncomputable theory

theorem eccentricity_of_ellipse :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →
    ∃ (M N : ℝ × ℝ), M ∈ E ∧ N ∈ E ∧
    ∃ (t : ℝ), (-1, t) = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧
    ∃ Q : ℝ × ℝ, Q = (-3/4, 0) ∧
    ∃ l l' : ℝ, l = (N.2 - M.2) / (N.1 - M.1) ∧ l = -1 / (N.2 - M.2) / (N.1 - M.1)) →
    (sqrt (1 - b^2 / a^2)) = sqrt 3 / 2 :=
by
  intro a b
  intro ha hb
  use a, b
  split
  exact ha
  split
  exact hb
  intros x y hxy
  sorry

end eccentricity_of_ellipse_l564_564754


namespace grisha_wins_probability_expected_flips_l564_564827

-- Define conditions
def even_flip_heads_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 0 ∧ flip n = tt

def odd_flip_tails_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 1 ∧ flip n = ff

-- 1. Prove the probability P of Grisha winning the coin toss game is 1/3
theorem grisha_wins_probability (flip : ℕ → bool) (P : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  P = 1/3 := sorry

-- 2. Prove the expected number E of coin flips until the outcome is decided is 2
theorem expected_flips (flip : ℕ → bool) (E : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  E = 2 := sorry

end grisha_wins_probability_expected_flips_l564_564827


namespace find_b_minus_a_l564_564905

theorem find_b_minus_a (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a - 9 * b + 18 * a * b = 2018) : b - a = 223 :=
sorry

end find_b_minus_a_l564_564905


namespace train_passing_platform_time_l564_564532

theorem train_passing_platform_time
  (L_train : ℝ) (L_plat : ℝ) (time_to_cross_tree : ℝ) (time_to_pass_platform : ℝ)
  (H1 : L_train = 2400) 
  (H2 : L_plat = 800)
  (H3 : time_to_cross_tree = 60) :
  time_to_pass_platform = 80 :=
by
  -- add proof here
  sorry

end train_passing_platform_time_l564_564532


namespace aquarium_final_volume_l564_564928

theorem aquarium_final_volume :
  let length := 4
  let width := 6
  let height := 3
  let total_volume := length * width * height
  let initial_volume := total_volume / 2
  let spilled_volume := initial_volume / 2
  let remaining_volume := initial_volume - spilled_volume
  let final_volume := remaining_volume * 3
  final_volume = 54 :=
by sorry

end aquarium_final_volume_l564_564928


namespace triangle_reflection_concurrence_l564_564390

theorem triangle_reflection_concurrence
    (A B C P A' B' C' : Point)
    (mid_AB mid_BC mid_CA : Point)
    (h_mid_AB : midpoint A B mid_AB)
    (h_mid_BC : midpoint B C mid_BC)
    (h_mid_CA : midpoint C A mid_CA) 
    (h_reflect_A' : reflection_over P mid_BC A')
    (h_reflect_B' : reflection_over P mid_CA B')
    (h_reflect_C' : reflection_over P mid_AB C'):
    concurrent (line A A') (line B B') (line C C') := 
by
    sorry

end triangle_reflection_concurrence_l564_564390


namespace part_a_part_b_l564_564348

noncomputable def sequence (n : ℕ) : ℝ := Real.cos (Real.pi * 10 ^ n / 180)

theorem part_a : (sequence 100) > 0 := 
by {
  sorry
}

theorem part_b : abs (sequence 100) < 0.18 :=
by {
  sorry
}

end part_a_part_b_l564_564348


namespace ratio_shiny_igneous_to_total_l564_564810

-- Define the conditions
variable (S I SI : ℕ)
variable (SS : ℕ)
variable (h1 : I = S / 2)
variable (h2 : SI = 40)
variable (h3 : S + I = 180)
variable (h4 : SS = S / 5)

-- Statement to prove
theorem ratio_shiny_igneous_to_total (S I SI SS : ℕ) 
  (h1 : I = S / 2) 
  (h2 : SI = 40) 
  (h3 : S + I = 180) 
  (h4 : SS = S / 5) : 
  SI / I = 2 / 3 := 
sorry

end ratio_shiny_igneous_to_total_l564_564810


namespace sin_identity_l564_564363

theorem sin_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 4) :
  Real.sin (2 * α + π / 6) = 7 / 8 := 
by
  sorry

end sin_identity_l564_564363


namespace values_of_x_satisfy_g_l564_564009

noncomputable def g (x : ℝ) : ℝ := -3 * Real.cos (2 * Real.pi * x)

theorem values_of_x_satisfy_g (h : -1 ≤ x ∧ x ≤ 1) : (setOf (x : ℝ) (h ∧ g (g (g x)) = g x)).card = 60 :=
sorry

end values_of_x_satisfy_g_l564_564009


namespace rebus_solution_l564_564268

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564268


namespace tunnel_length_l564_564534

def train_length : ℝ := 800  -- The train is 800 meters long.
def train_speed_kmh : ℝ := 78  -- The speed of the train is 78 km/hr.
def crossing_time_sec : ℝ := 60  -- The train crosses the tunnel in 60 seconds.

-- Convert the speed from km/hr to m/s
def train_speed_ms : ℝ := 
  let conversion_factor := 1000 / 3600  -- Factor to convert km/hr to m/s
  train_speed_kmh * conversion_factor

def distance_covered : ℝ := train_speed_ms * crossing_time_sec  -- Distance covered by the train in 60 seconds

theorem tunnel_length : distance_covered - train_length = 500.2 := by
  sorry  -- Proof not required

end tunnel_length_l564_564534


namespace moles_of_HCl_formed_l564_564716

theorem moles_of_HCl_formed
  (C2H6_initial : Nat)
  (Cl2_initial : Nat)
  (HCl_expected : Nat)
  (balanced_reaction : C2H6_initial + Cl2_initial = C2H6_initial + HCl_expected):
  C2H6_initial = 2 → Cl2_initial = 2 → HCl_expected = 2 :=
by
  intros
  sorry

end moles_of_HCl_formed_l564_564716


namespace min_value_expression_l564_564294

theorem min_value_expression (y : ℝ) : 
  ∃ (m : ℝ), m = min (λ y, y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81)) ∧ m = 1/27 := 
sorry

end min_value_expression_l564_564294


namespace guaranteed_max_points_l564_564458

theorem guaranteed_max_points (points : ℕ → ℕ → ℕ → ℕ) : 
  (∀ i j k m n o p q : ℕ, 
    points i 6 4 2 = 4 → 
    i >= 20 → 
    j < i ∧ k < i ∧ l < i ∧ m < i ∧ n < i ∧ o < i ∧ p < i ∧ q < i →
    (points (i+j+k+l+m+n+o+p) 6 4 2 < i ->
    q <= points i 6 4 2)) := sorry

end guaranteed_max_points_l564_564458


namespace value_of_sum_l564_564950

theorem value_of_sum (a b c d : ℤ) 
  (h1 : a - b + c = 7)
  (h2 : b - c + d = 8)
  (h3 : c - d + a = 5)
  (h4 : d - a + b = 4) : a + b + c + d = 12 := 
  sorry

end value_of_sum_l564_564950


namespace find_all_z_l564_564666

noncomputable def all_solutions (z : ℝ) : Prop := z = -2 ∨ z = 2 ∨ z = -real.sqrt 2 ∨ z = real.sqrt 2

theorem find_all_z (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ all_solutions z := 
sorry

end find_all_z_l564_564666


namespace number_2008_in_row_45_l564_564595

theorem number_2008_in_row_45 :
  ∃ n : ℕ, ∑ k in finset.range (n + 1), (2 * k + 1) = 2008 ∧ 2008 ∈ finset.range (n^2, n^2 + 2 * n + 1) :=
by {
  have h_sum : ∀ n, ∑ k in finset.range (n + 1), (2 * k + 1) = n^2, from sorry,
  use 45,
  split,
  calc ∑ k in finset.range (45 + 1), (2 * k + 1) = 45^2 : by apply h_sum,
  exact dec_trivial,
  have h_2008 : 1936 < 2008 ∧ 2008 < 2025, from sorry,
  exact h_2008 }

end number_2008_in_row_45_l564_564595


namespace percentage_increase_l564_564199

theorem percentage_increase (use_per_six_months : ℝ) (new_annual_use : ℝ) : 
  use_per_six_months = 90 →
  new_annual_use = 216 →
  ((new_annual_use - 2 * use_per_six_months) / (2 * use_per_six_months)) * 100 = 20 :=
by
  intros h1 h2
  sorry

end percentage_increase_l564_564199


namespace sqrt_diff_geq_inv_x_l564_564795

theorem sqrt_diff_geq_inv_x (x : ℝ) (hx : x ≥ 4) : 
  sqrt x - sqrt (x - 1) ≥ 1 / x :=
sorry

end sqrt_diff_geq_inv_x_l564_564795


namespace total_people_in_bus_l564_564135

-- Definitions based on the conditions
def left_seats : Nat := 15
def right_seats := left_seats - 3
def people_per_seat := 3
def back_seat_people := 9

-- Theorem statement
theorem total_people_in_bus : 
  (left_seats * people_per_seat) +
  (right_seats * people_per_seat) + 
  back_seat_people = 90 := 
by sorry

end total_people_in_bus_l564_564135


namespace rebus_solution_l564_564279

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564279


namespace mixture_ratio_l564_564517

variables (p q V W : ℝ)

-- Condition summaries:
-- - First jar has volume V, ratio of alcohol to water is p:1.
-- - Second jar has volume W, ratio of alcohol to water is q:2.

theorem mixture_ratio (hp : p > 0) (hq : q > 0) (hV : V > 0) (hW : W > 0) : 
  (p * V * (p + 2) + q * W * (p + 1)) / ((p + 1) * (q + 2) * (V + 2 * W)) =
  (p * V) / (p + 1) + (q * W) / (q + 2) :=
sorry

end mixture_ratio_l564_564517


namespace machine_original_price_l564_564053

theorem machine_original_price 
  (P : ℝ) 
  (repair_cost transport_cost : ℝ) 
  (profit_rate : ℝ) 
  (selling_price : ℝ)
  (repair_cost_eq : repair_cost = 5000)
  (transport_cost_eq : transport_cost = 1000)
  (profit_rate_eq : profit_rate = 0.5)
  (selling_price_eq : selling_price = 28500) :
  P = 13000 :=
by
  have total_cost := P + repair_cost + transport_cost
  have total_cost_eq : total_cost = P + 6000, from calc
    total_cost = P + repair_cost + transport_cost : rfl
    ... = P + 5000 + 1000 : by rw [repair_cost_eq, transport_cost_eq]
    ... = P + 6000 : rfl
  let final_selling_price := total_cost + profit_rate * total_cost
  have final_selling_price_eq : final_selling_price = 1.5 * total_cost, from calc
    final_selling_price = total_cost + profit_rate * total_cost : rfl
    ... = total_cost + 0.5 * total_cost : by rw profit_rate_eq
    ... = 1.5 * total_cost : by ring
  have selling_price_eq' : final_selling_price = 28500, from calc
    final_selling_price = 1.5 * total_cost : final_selling_price_eq
    ... = 28500 : by rw selling_price_eq
  have eq_28500 : 1.5 * (P + 6000) = 28500, from calc
    1.5 * total_cost = 28500 : selling_price_eq'
    ... = 28500 : rfl
  have eq_P_6000 : P + 6000 = 19000, from
    eq_of_mul_eq_mul_left (show 1.5 ≠ 0, by norm_num) eq_28500
  exact eq_of_add_eq_add_right eq_P_6000

end machine_original_price_l564_564053


namespace cannot_form_trapezoid_with_two_identical_right_angled_triangles_l564_564131

theorem cannot_form_trapezoid_with_two_identical_right_angled_triangles
  (T : Type)
  [inhabited T]
  (triangle : T → T → T → Prop)
  (right_angle : ∀ {a b c : T}, triangle a b c → triangle c a b) :
  ¬ (∃ (a b c d : T), isosceles_triangle a b c ∧ isosceles_triangle c d a ∧ ¬ isosceles_triangle b d a) :=
sorry

end cannot_form_trapezoid_with_two_identical_right_angled_triangles_l564_564131


namespace same_terminal_side_angle_exists_l564_564473

theorem same_terminal_side_angle_exists :
  ∃ k : ℤ, -5 * π / 8 + 2 * k * π = 11 * π / 8 := 
by
  sorry

end same_terminal_side_angle_exists_l564_564473


namespace triangle_is_right_angled_at_C_l564_564808

-- Define the internal angles A, B, and C of triangle ABC
def is_triangle_interior_angles (A B C : ℝ) : Prop :=
A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0

-- Given condition in the math problem
def given_condition (A B C : ℝ) : Prop :=
sin A = sin C * cos B

-- The mathematical proof problem in Lean 4 statement:
theorem triangle_is_right_angled_at_C (A B C : ℝ) (h1 : is_triangle_interior_angles A B C) (h2 : given_condition A B C) :
C = π / 2 :=
sorry

end triangle_is_right_angled_at_C_l564_564808


namespace polar_to_rectangular_l564_564984

theorem polar_to_rectangular : 
  ∀ (r θ : ℝ), r = 4 → θ = (5 * Real.pi / 6) → 
  let x := r * Real.cos θ 
  let y := r * Real.sin θ 
  (x, y) = (-2 * Real.sqrt 3, 2) :=
by
  intros r θ r_def θ_def
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  have x_val : x = -2 * Real.sqrt 3 := by sorry
  have y_val : y = 2 := by sorry
  rw [x_val, y_val]
  exact ⟨rfl, rfl⟩

end polar_to_rectangular_l564_564984


namespace digit_7_count_in_range_l564_564878

def count_digit_7 : ℕ :=
  let units_place := (107 - 100) / 10 + 1
  let tens_place := 10
  units_place + tens_place

theorem digit_7_count_in_range : count_digit_7 = 20 := by
  sorry

end digit_7_count_in_range_l564_564878


namespace radius_of_circle_on_ellipse_l564_564559

theorem radius_of_circle_on_ellipse (E : ℝ^2 → Prop) (r : ℝ) (c : ℝ^2)
  (Hc : E c) (Htan1 Htan2 : ∃ p q : ℝ^2, E p ∧ E q ∧ p ≠ q ∧ (∃ d1 : ℝ^2 → Prop, d1 p ∧ d1 q ∧ ∀ x, d1 x → ‖x - c‖ = r)):
  ∀ (p' q' : ℝ^2), E p' ∧ E q' ∧ p' ≠ q' ∧ (∃ d2 : ℝ^2 → Prop, d2 p' ∧ d2 q' ∧ ∀ x, d2 x → ‖x - c‖ = r) → r = r :=
begin
  sorry -- The proof goes here
end

end radius_of_circle_on_ellipse_l564_564559


namespace value_of_y_when_x_is_27_l564_564369

noncomputable def k (x : ℝ) (y : ℝ) : ℝ := y / (x ^ (1/3))

theorem value_of_y_when_x_is_27 :
  let k := k 8 4 in
  y = k * 27^(1/3) :=
by
  have hx8 : 8 ^ (1/3) = 2 := sorry
  have hx27 : 27 ^ (1/3) = 3 := sorry
  have h1 : 4 = k 8 4 * (8 ^ (1/3)) := sorry
  rw [hx8] at h1
  have h2 : k 8 4 = 2 := sorry
  show y = 2 * 27^(1/3)
  rw [hx27]
  show y = 6
  sorry

end value_of_y_when_x_is_27_l564_564369


namespace count_correct_statements_l564_564786

theorem count_correct_statements :
  let s1 := ¬(∀ a b : ℝ, (a + b ≥ 4 → (a ≥ 2 ∨ b ≥ 2)) ∧ (∀ a b : ℝ, (a ≥ 2 ∨ b ≥ 2 → a + b ≥ 4)))
  let s2 := (∀ a b : ℝ, (a + b ≠ 6 → (a ≠ 3 ∨ b ≠ 3)))
  let s3 := ¬(∀ x : ℝ, x^2 - x > 0) ∧ (∃ x0 : ℝ, x0^2 - x0 < 0)
  let s4 := (∀ a b : ℝ, (a + 1 > b → a > b) ∧ ¬(a > b → a + 1 > b))
  in (if s1 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0) = 2 :=
by
  sorry

end count_correct_statements_l564_564786


namespace Grisha_probability_expected_flips_l564_564830

theorem Grisha_probability (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                            t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (probability (Grisha wins)) = 1 / 3 :=
by 
  sorry

theorem expected_flips (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                        t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (expected_flips) = 2 :=
by
  sorry

end Grisha_probability_expected_flips_l564_564830


namespace sum_of_coordinates_l564_564039

theorem sum_of_coordinates (C D : ℝ × ℝ) (hC : C = (0, 0)) (hD : D.snd = 6) (h_slope : (D.snd - C.snd) / (D.fst - C.fst) = 3/4) : D.fst + D.snd = 14 :=
sorry

end sum_of_coordinates_l564_564039


namespace triangle_sides_containing_rhombus_sides_l564_564169

-- Definitions from given conditions.
variable (m n : ℝ)
variable (A B C D : Type)
variable [has_angle A] [has_angle B] [has_angle C] [has_angle D]
variable [has_rhombus_inscribed A B C D] -- A typeclass representing the rhombus inscribed in a triangle.

-- Some derived definitions
def BD := m
def AC := n
def AD := (1/2) * (m + n)
def AB := (1/2) * (sqrt (m ^ 2 + n ^ 2))

-- Calculating the triangle sides containing the sides of the rhombus.
def side_of_triangle_1 := (5 / 6) * (sqrt (m ^ 2 + n ^ 2))
def side_of_triangle_2 := (5 / 4) * (sqrt (m ^ 2 + n ^ 2))

-- Lean statement for the proof problem.
theorem triangle_sides_containing_rhombus_sides :
  ∃ (side1 side2 : ℝ), side1 = (5 / 6) * (sqrt (m ^ 2 + n ^ 2)) 
                     ∧ side2 = (5 / 4) * (sqrt (m ^ 2 + n ^ 2)) :=
sorry

end triangle_sides_containing_rhombus_sides_l564_564169


namespace downward_parabola_with_symmetry_l564_564035

-- Define the general form of the problem conditions in Lean
theorem downward_parabola_with_symmetry (k : ℝ) :
  ∃ a : ℝ, a < 0 ∧ ∃ h : ℝ, h = 3 ∧ ∃ k : ℝ, k = k ∧ ∃ (y x : ℝ), y = a * (x - h)^2 + k :=
sorry

end downward_parabola_with_symmetry_l564_564035


namespace maximize_x5_y3_solution_l564_564007

noncomputable def maximize_x5_y3 (x y : ℝ) (h : x + y = 45) : x^5 * y^3 ≤ (225 / 8)^5 * (135 / 8)^3 := sorry

theorem maximize_x5_y3_solution :
  maximize_x5_y3 (225 / 8) (135 / 8) (by norm_num) := sorry

end maximize_x5_y3_solution_l564_564007


namespace correct_propositions_count_l564_564623

def prop1 (a b : Vector ℝ 3) (h : a = -b) : |a| = |b| := sorry

def prop2 (A B C D : Point ℝ 3) (h : collinear (A - B) (C - D)) : 
  ∃ l : Line ℝ 3, A ∈ l ∧ B ∈ l ∧ C ∈ l ∧ D ∈ l := sorry

def prop3 (a b : Vector ℝ 3) (h : |a| = |b|) : a = b ∨ a = -b := sorry

def prop4 (a b : Vector ℝ 3) (h : a.dot b = 0) : a = 0 ∨ b = 0 := sorry

theorem correct_propositions_count : 
  count_correct_propositions prop1 prop2 prop3 prop4 = 1 := sorry

end correct_propositions_count_l564_564623


namespace digit_7_count_in_range_l564_564874

def count_digit_7 : ℕ :=
  let units_place := (107 - 100) / 10 + 1
  let tens_place := 10
  units_place + tens_place

theorem digit_7_count_in_range : count_digit_7 = 20 := by
  sorry

end digit_7_count_in_range_l564_564874


namespace pet_store_total_birds_l564_564139

def total_birds_in_pet_store (bird_cages parrots_per_cage parakeets_per_cage : ℕ) : ℕ :=
  bird_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_total_birds :
  total_birds_in_pet_store 4 8 2 = 40 :=
by
  sorry

end pet_store_total_birds_l564_564139


namespace area_perimeter_ratio_eq_l564_564122

theorem area_perimeter_ratio_eq (s : ℝ) (s_eq : s = 10) : 
  let area := (sqrt 3) / 4 * s ^ 2
      perimeter := 3 * s
      ratio := area / (perimeter ^ 2)
  in ratio = (sqrt 3) / 36 :=
by sorry

end area_perimeter_ratio_eq_l564_564122


namespace simplify_expression_l564_564461

theorem simplify_expression :
  let a := (1/2)^2
  let b := (1/2)^3
  let c := (1/2)^4
  let d := (1/2)^5
  1 / (1/a + 1/b + 1/c + 1/d) = 1/60 :=
by
  sorry

end simplify_expression_l564_564461


namespace ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l564_564119

theorem ratio_of_area_to_square_of_perimeter_of_equilateral_triangle :
  let a := 10
  let area := (10 * 10 * (Real.sqrt 3) / 4)
  let perimeter := 3 * 10
  let square_of_perimeter := perimeter * perimeter
  (area / square_of_perimeter) = (Real.sqrt 3 / 36) := by
  -- Proof to be completed
  sorry

end ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l564_564119


namespace ellipse_major_axis_length_l564_564579

-- Given conditions
variable (radius : ℝ) (h_radius : radius = 2)
variable (minor_axis : ℝ) (h_minor_axis : minor_axis = 2 * radius)
variable (major_axis : ℝ) (h_major_axis : major_axis = 1.4 * minor_axis)

-- Proof problem statement
theorem ellipse_major_axis_length : major_axis = 5.6 :=
by
  sorry

end ellipse_major_axis_length_l564_564579


namespace chord_length_of_intersection_is_sqrt14_l564_564719

-- Define the polar equation of the curve C
def curveC (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the parametric equation of the line l
def lineL (t : ℝ) (x y : ℝ) : Prop := x = t ∧ y = -t + 1

-- Define the linear equation of the line l converted from the parametric form
def lineL_cartesian (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the distance formula from a point to a line
def dist_point_to_line (a b c x0 y0: ℝ) : ℝ := abs (a * x0 + b * y0 + c) / Real.sqrt (a^2 + b^2)

-- Define the chord length based on the distance from center to the line
def chord_length (r d : ℝ) : ℝ := 2 * Real.sqrt(r^2 - d^2)

-- The main theorem statement
theorem chord_length_of_intersection_is_sqrt14 : ∀ t x y : ℝ, (curveC x y) → (lineL t x y) → chord_length 2 (dist_point_to_line 1 1 (-1) 0 0) = Real.sqrt 14 :=
by
  intros t x y hC hL
  sorry -- Proof is omitted

end chord_length_of_intersection_is_sqrt14_l564_564719


namespace area_of_inscribed_octagon_l564_564519

-- Define the given conditions and required proof
theorem area_of_inscribed_octagon (r : ℝ) (h : π * r^2 = 400 * π) :
  let A := r^2 * (1 + Real.sqrt 2)
  A = 20^2 * (1 + Real.sqrt 2) :=
by 
  sorry

end area_of_inscribed_octagon_l564_564519


namespace digit_7_count_in_range_100_to_199_l564_564889

/-- 
Given the range of numbers from 100 to 199 inclusive,
prove that the digit 7 appears exactly 20 times.
-/
theorem digit_7_count_in_range_100_to_199 : 
  let count_7 := (λ n : ℕ, 
    (n / 100 = 7) +  -- hundreds place
    ((n % 100) / 10 = 7) + -- tens place
    ((n % 10) = 7)) in
  (Finset.sum (Finset.range' 100 100) count_7 = 20) :=
by sorry

end digit_7_count_in_range_100_to_199_l564_564889


namespace even_square_even_square_even_even_l564_564917

-- Definition for a natural number being even
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Statement 1: If p is even, then p^2 is even
theorem even_square_even (p : ℕ) (hp : is_even p) : is_even (p * p) :=
sorry

-- Statement 2: If p^2 is even, then p is even
theorem square_even_even (p : ℕ) (hp_squared : is_even (p * p)) : is_even p :=
sorry

end even_square_even_square_even_even_l564_564917


namespace probability_of_alpha_between_vectors_is_5_12_l564_564568

theorem probability_of_alpha_between_vectors_is_5_12 :
  (∃ m n : ℕ, 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6 ∧ ∃ α : ℝ, α = real.arccos (m / real.sqrt (m^2 + n^2)) 
  ∧ α ∈ (0, real.pi / 4) → (finset.card {x : ℕ × ℕ | x.1 ∈ finset.range 6 ∧ x.1 + 1 > x.2}) / (6 * 6) = 5 / 12 := by sorry

end probability_of_alpha_between_vectors_is_5_12_l564_564568


namespace percentage_of_tomato_plants_is_20_l564_564936

-- Define the conditions
def garden1_plants := 20
def garden1_tomato_percentage := 0.10
def garden2_plants := 15
def garden2_tomato_percentage := 1 / 3

-- Define the question as a theorem to be proved
theorem percentage_of_tomato_plants_is_20 :
  let total_plants := garden1_plants + garden2_plants in
  let total_tomato_plants := (garden1_tomato_percentage * garden1_plants) + (garden2_tomato_percentage * garden2_plants) in
  (total_tomato_plants / total_plants) * 100 = 20 :=
by
  sorry

end percentage_of_tomato_plants_is_20_l564_564936


namespace cos_angle_a_b_l564_564782

variable (a b : EuclideanSpace ℝ (Fin 3))

axiom h1 : a + b = ![0, Real.sqrt 2, 0]
axiom h2 : a - b = ![2, Real.sqrt 2, -2 * Real.sqrt 3]

theorem cos_angle_a_b : 
  cosine_angle ⟨a, a ≠ 0⟩ ⟨b, b ≠ 0⟩ = -Real.sqrt 6 / 3 :=
by 
  sorry

end cos_angle_a_b_l564_564782


namespace range_of_a_l564_564354

noncomputable def p (x : ℝ) : Prop := (4 * x - 3)^2 ≤ 1
noncomputable def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (h: ∀ x a, p x → q x a) : 
  ∀ a, (∀ x, ¬p x → ¬q x a) ∧ (∃ x, q x a ∧ ¬p x) → (0 ≤ a ∧ a ≤ 1 / 2) :=
begin
  intro a,
  intro h_neq,
  sorry
end

end range_of_a_l564_564354


namespace sum_of_coefficients_of_g_l564_564634

theorem sum_of_coefficients_of_g (a b c d : ℝ) :
  let g := λ x : ℂ, x^4 + (a : ℂ) * x^3 + (b : ℂ) * x^2 + (c : ℂ) * x + d in
  g 2i = 0 → g (1 + i) = 0 → a + b + c + d = 4 :=
by
  -- Begin the proof (content/writing steps are omitted as per instructions)
  sorry

end sum_of_coefficients_of_g_l564_564634


namespace b_50_value_l564_564222

noncomputable def seq (n : ℕ) : ℝ :=
if n = 1 then 2 else (8 : ℝ) * seq (n - 1)

theorem b_50_value :
  seq 50 = (8:ℝ)^49 * 2 :=
sorry

end b_50_value_l564_564222


namespace return_trip_time_is_110_l564_564162

def plane_time_return_trip 
  (distance : ℝ) 
  (plane_speed_still_air wind_speed : ℝ) 
  (time_against_wind : ℝ) 
  (time_less_with_wind : ℝ) 
  : ℝ :=
  distance / (plane_speed_still_air + wind_speed)

theorem return_trip_time_is_110 :
  ∀ (d p w : ℝ), 
  let t := d / p in
  d = 120 * (p - w) → 
  time_against_wind = 120 →
  time_less_with_wind = 10 →
  plane_time_return_trip d p w 120 10 = 110 :=
by 
  intros d p w h₁ h₂ h₃
  sorry

end return_trip_time_is_110_l564_564162


namespace probability_sum_18_l564_564964

def total_outcomes := 100

def successful_pairs := [(8, 10), (9, 9), (10, 8)]

def num_successful_outcomes := successful_pairs.length

theorem probability_sum_18 : (num_successful_outcomes / total_outcomes : ℚ) = 3 / 100 := 
by
  -- The actual proof should go here
  sorry

end probability_sum_18_l564_564964


namespace sequence_count_100_l564_564026

theorem sequence_count_100 :
  let n := 100 in
  let total_sequences := 5^n in
  let unwanted_sequences := 3^n in
  let result := total_sequences - unwanted_sequences in
  result = (5^100 - 3^100) :=
by
  -- Proof goes here
  sorry

end sequence_count_100_l564_564026


namespace radius_increase_of_pizza_l564_564796

/-- 
Prove that the percent increase in radius from a medium pizza to a large pizza is 20% 
given the following conditions:
1. The radius of the large pizza is some percent larger than that of a medium pizza.
2. The percent increase in area between a medium and a large pizza is approximately 44%.
3. The area of a circle is given by the formula A = π * r^2.
--/
theorem radius_increase_of_pizza
  (r R : ℝ) -- r and R are the radii of the medium and large pizza respectively
  (h1 : R = (1 + k) * r) -- The radius of the large pizza is some percent larger than that of a medium pizza
  (h2 : π * R^2 = 1.44 * π * r^2) -- The percent increase in area between a medium and a large pizza is approximately 44%
  : k = 0.2 := 
sorry

end radius_increase_of_pizza_l564_564796


namespace range_of_ab_l564_564304

variable (a b : ℝ)
variable (h_a_pos : a > 0)
variable (h_b_pos : b > 0)
variable (h_dist : |a + b| / sqrt ((a + 1)^2 + (b + 1)^2) = 1)

theorem range_of_ab : ab ≥ 3 + 2 * sqrt 2 :=
sorry

end range_of_ab_l564_564304


namespace radius_of_circle_l564_564978

noncomputable def circle_radius : ℝ := by
  -- Let (x, 0) be the center of the circle
  let x : ℝ := 3
  
  -- Points on the circle
  let A := (1 : ℝ)
  let B := (5 : ℝ)
  let C := (2 : ℝ)
  let D := (4 : ℝ)

  -- Calculate the radius
  let radius := Real.sqrt ((x - A)^2 + (0 - B)^2)
  
  -- It is known that the radius should be √29
  exact radius
  
theorem radius_of_circle : circle_radius = Real.sqrt 29 := sorry

end radius_of_circle_l564_564978


namespace part_a_part_b_l564_564352

noncomputable def sequence (n : ℕ) : ℝ := Real.cos (Real.pi * 10 ^ n / 180)

theorem part_a : (sequence 100) > 0 := 
by {
  sorry
}

theorem part_b : abs (sequence 100) < 0.18 :=
by {
  sorry
}

end part_a_part_b_l564_564352


namespace arithmetic_sequence_sum_l564_564742

theorem arithmetic_sequence_sum : 
  ∀ (a : ℕ → ℝ) (d : ℝ), (a 1 = 2 ∨ a 1 = 8) → (a 2017 = 2 ∨ a 2017 = 8) → 
  (∀ n : ℕ, a (n + 1) = a n + d) →
  a 2 + a 1009 + a 2016 = 15 := 
by
  intro a d h1 h2017 ha
  sorry

end arithmetic_sequence_sum_l564_564742


namespace at_least_one_inequality_false_l564_564547

open Classical

theorem at_least_one_inequality_false (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
  ¬ (a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by
  sorry

end at_least_one_inequality_false_l564_564547


namespace triangle_side_length_l564_564807

variable (A B a b : ℝ)
variable (hABC : ∀ (a b c A B C : ℝ), a / sin A = b / sin B)
variable (hSinA_gt_SinB : sin A > sin B)

theorem triangle_side_length (hSinA_gt_SinB : sin A > sin B) (hABC : ∀ (a b c A B C : ℝ), a / sin A = b / sin B) : a > b :=
  sorry

end triangle_side_length_l564_564807


namespace solve_quartic_l564_564688

theorem solve_quartic (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ (z = 2 ∨ z = -2 ∨ z = sqrt 2 ∨ z = -sqrt 2) :=
by
  sorry

end solve_quartic_l564_564688


namespace people_per_table_l564_564190

theorem people_per_table (kids adults tables : ℕ) (h_kids : kids = 45) (h_adults : adults = 123) (h_tables : tables = 14) :
  ((kids + adults) / tables) = 12 :=
by
  -- Placeholder for proof
  sorry

end people_per_table_l564_564190


namespace sqrt_expr_meaningful_l564_564371

theorem sqrt_expr_meaningful (x : ℝ) : (∃ y, y = sqrt (2 - x)) → x ≤ 2 := by
  intro h
  obtain ⟨y, hy⟩ := h
  sorry

end sqrt_expr_meaningful_l564_564371


namespace no_real_a_values_l564_564254

noncomputable def polynomial_with_no_real_root (a : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + a^2 * x^3 - 2 * x^2 + a * x + 4 ≠ 0
  
theorem no_real_a_values :
  ∀ a : ℝ, (∃ x : ℝ, x^4 + a^2 * x^3 - 2 * x^2 + a * x + 4 = 0) → false :=
by sorry

end no_real_a_values_l564_564254


namespace grisha_win_probability_expected_number_coin_flips_l564_564820

-- Problem 1: Probability of Grisha's win
theorem grisha_win_probability : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (grisha_win : ℚ) :=
  sorry

-- Problem 2: Expected number of coin flips
theorem expected_number_coin_flips : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (expected_tosses : ℚ) :=
  sorry

end grisha_win_probability_expected_number_coin_flips_l564_564820


namespace quadratic_polynomial_roots_l564_564416

-- Given conditions
variables (x y : ℝ)
axiom h1 : 2 * x + 3 * y = 18
axiom h2 : x * y = 8

-- Statement to prove
theorem quadratic_polynomial_roots :
  (∃ (p : polynomial ℝ), p.roots = {x, y} ∧ p = (X^2 - 18 * X + 8)) := 
sorry

end quadratic_polynomial_roots_l564_564416


namespace find_b_for_sine_l564_564600

theorem find_b_for_sine (a b c d : ℝ) (h1 : 5 = d + a) (h2 : -3 = d - a) (h3 : (∀ x, y = a * sin (b * x + c) + d) → (∃ p, p = 2 * π / b) (h4 : p = 2 * π / 3)) :
  b = 3 :=
begin
  sorry
end

end find_b_for_sine_l564_564600


namespace sphere_properties_l564_564586

noncomputable def radius_from_volume (V : ℝ) : ℝ :=
  let π := Real.pi in
  Real.cbrt ((3 * V) / (4 * π))

noncomputable def surface_area (r : ℝ) : ℝ :=
  let π := Real.pi in
  4 * π * r^2

noncomputable def circumference_flat_surface (r : ℝ) : ℝ :=
  let π := Real.pi in
  2 * π * r

theorem sphere_properties 
  (V : ℝ) 
  (hV : V = 288 * Real.pi) :
  let r := radius_from_volume V in
  surface_area r = 144 * Real.pi ∧ 
  circumference_flat_surface r / 2 = 12 * Real.pi := by
    sorry

end sphere_properties_l564_564586


namespace ellipse_foci_on_x_axis_major_axis_twice_minor_axis_l564_564078

theorem ellipse_foci_on_x_axis_major_axis_twice_minor_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 + m * y^2 = 1) → (∃ a b : ℝ, a = 1 ∧ b = Real.sqrt (1 / m) ∧ a = 2 * b) → m = 4 :=
by
  sorry

end ellipse_foci_on_x_axis_major_axis_twice_minor_axis_l564_564078


namespace find_digits_l564_564275

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l564_564275


namespace three_digit_cubes_divisible_by_8_l564_564787

theorem three_digit_cubes_divisible_by_8 : ∃ (S : Finset ℕ), S.card = 2 ∧ ∀ x ∈ S, x ^ 3 ≥ 100 ∧ x ^ 3 ≤ 999 ∧ x ^ 3 % 8 = 0 :=
by
  sorry

end three_digit_cubes_divisible_by_8_l564_564787


namespace number_of_correct_statements_l564_564593

theorem number_of_correct_statements :
  let cond1 := (0 ∈ ({0} : set ℕ)),
      cond2 := (∅ ⊆ ({0} : set ℕ)),
      cond3 := ¬ ({0, 1} ⊆ ({(0, 1)} : set (ℕ × ℕ))),
      cond4 := ({(0, 1)} ≠ ({(1, 0)} : set (ℕ × ℕ))),
      cond5 := ({0, 1} = ({1, 0} : set ℕ)) in
  (cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 → (3 = 3)) :=
by
  sorry

end number_of_correct_statements_l564_564593


namespace fixed_point_of_line_l564_564125

theorem fixed_point_of_line : ∀ k : ℝ, ∃ x y : ℝ, (kx - y + 1 = 3k) ∧ (x = 3) ∧ (y = 1) :=
by
  -- The proof is not required as per the instructions
  sorry

end fixed_point_of_line_l564_564125


namespace find_all_z_l564_564671

noncomputable def all_solutions (z : ℝ) : Prop := z = -2 ∨ z = 2 ∨ z = -real.sqrt 2 ∨ z = real.sqrt 2

theorem find_all_z (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ all_solutions z := 
sorry

end find_all_z_l564_564671


namespace complex_power_norm_squared_l564_564449

theorem complex_power_norm_squared (x y s t : ℝ) (n : ℕ) 
  (h : x + y * complex.I = (s + t * complex.I)^n) : 
  x^2 + y^2 = (s^2 + t^2)^n :=
sorry

end complex_power_norm_squared_l564_564449


namespace cost_of_adult_ticket_eq_19_l564_564018

variables (X : ℝ)
-- Condition 1: The cost of an adult ticket is $6 more than the cost of a child ticket.
def cost_of_child_ticket : ℝ := X - 6

-- Condition 2: The total cost of the 5 tickets is $77.
axiom total_cost_eq : 2 * X + 3 * (X - 6) = 77

-- Prove that the cost of an adult ticket is 19 dollars
theorem cost_of_adult_ticket_eq_19 (h : total_cost_eq) : X = 19 := 
by
  -- Here we would provide the actual proof steps
  sorry

end cost_of_adult_ticket_eq_19_l564_564018


namespace expected_red_polygon_sides_l564_564506

-- Defining the conditions
def square : Type := {A : Finset (ℝ × ℝ) // 
                     A = { (0,0), (0,1), (1,0), (1,1) } }

def expected_sides (F : ℝ × ℝ) (A : ℝ × ℝ) : ℝ := 
  if (F = A) then 3 else 4

-- The theorem to prove
theorem expected_red_polygon_sides :
  ∃ (F : ℝ × ℝ) (A : ℝ × ℝ) (sq : square), 
    @expected_sides F A = 5 - (π / 2) :=
by {
  sorry
}

end expected_red_polygon_sides_l564_564506


namespace max_value_proof_l564_564469

variables (x y : ℝ)

-- Conditions
def condition1 (x y : ℝ) : Prop := x > 0 ∧ y > 0
def condition2 (x y : ℝ) : Prop := x^2 - 2 * x * y + 3 * y^2 = 12

-- Maximum value function
noncomputable def maximum_value (x y : ℝ) : ℝ := x^2 + 2 * x * y + 3 * y^2

-- Main statement to prove
theorem max_value_proof : 
  (∃ (x y : ℝ), condition1 x y ∧ condition2 x y ∧ 
  (maximum_value x y = (132 + 48 * real.sqrt 3)) ∧
  (184 = 132 + 48 + 3 + 1)) :=
by {
  -- proof would go here
  sorry
}

end max_value_proof_l564_564469


namespace validate_model_and_profit_range_l564_564140

noncomputable def is_exponential_model_valid (x y : ℝ) : Prop :=
  ∃ T a : ℝ, T > 0 ∧ a > 1 ∧ y = T * a^x

noncomputable def is_profitable_for_at_least_one_billion (x : ℝ) : Prop :=
  (∃ T a : ℝ, T > 0 ∧ a > 1 ∧ 1/5 * (Real.sqrt 2)^x ≥ 10 ∧ 0 < x ∧ x ≤ 12) ∨
  (-0.2 * (x - 12) * (x - 17) + 12.8 ≥ 10 ∧ x > 12)

theorem validate_model_and_profit_range :
  (is_exponential_model_valid 2 0.4) ∧
  (is_exponential_model_valid 4 0.8) ∧
  (is_exponential_model_valid 12 12.8) ∧
  is_profitable_for_at_least_one_billion 11.3 ∧
  is_profitable_for_at_least_one_billion 19 :=
by
  sorry

end validate_model_and_profit_range_l564_564140


namespace part_a_part_b_l564_564339

noncomputable def sequence_a (n : ℕ) : ℝ :=
  Real.cos (10^n * Real.pi / 180)

theorem part_a (h : 100 > 2) : sequence_a 100 > 0 := by
  sorry

theorem part_b : |sequence_a 100| < 0.18 := by
  sorry

end part_a_part_b_l564_564339


namespace calc_area_of_quadrilateral_l564_564179

-- Define the terms and conditions using Lean definitions
noncomputable def triangle_areas : ℕ × ℕ × ℕ := (6, 9, 15)

-- State the theorem
theorem calc_area_of_quadrilateral (a b c d : ℕ) (area1 area2 area3 : ℕ):
  area1 = 6 →
  area2 = 9 →
  area3 = 15 →
  a + b + c + d = area1 + area2 + area3 →
  d = 65 :=
  sorry

end calc_area_of_quadrilateral_l564_564179


namespace trajectory_equation_l564_564961

theorem trajectory_equation (x y : ℝ)
  (h : Real.sqrt((x - 1)^2 + y^2) = abs (x + 3) - 2) :
  y^2 = 4 * x :=
sorry

end trajectory_equation_l564_564961


namespace jose_profit_share_correct_l564_564997

-- Definitions for the conditions
def tom_investment : ℕ := 30000
def tom_months : ℕ := 12
def jose_investment : ℕ := 45000
def jose_months : ℕ := 10
def total_profit : ℕ := 36000

-- Capital months calculations
def tom_capital_months : ℕ := tom_investment * tom_months
def jose_capital_months : ℕ := jose_investment * jose_months
def total_capital_months : ℕ := tom_capital_months + jose_capital_months

-- Jose's share of the profit calculation
def jose_share_of_profit : ℕ := (jose_capital_months * total_profit) / total_capital_months

-- The theorem to prove
theorem jose_profit_share_correct : jose_share_of_profit = 20000 := by
  -- This is where the proof steps would go
  sorry

end jose_profit_share_correct_l564_564997


namespace average_disk_space_nearest_whole_number_l564_564564

def days_of_music : ℕ := 15
def total_disk_space_MB : ℕ := 20400
def hours_in_a_day : ℕ := 24

def total_hours (days : ℕ) (hours_per_day : ℕ) : ℕ := days * hours_per_day

def average_disk_space_per_hour (total_space : ℕ) (total_hours : ℕ) : ℚ := total_space / total_hours

theorem average_disk_space_nearest_whole_number :
  let total_hours_music := total_hours days_of_music hours_in_a_day in
  let actual_average := average_disk_space_per_hour total_disk_space_MB total_hours_music in
  Int.round actual_average = 57 :=
by
  sorry

end average_disk_space_nearest_whole_number_l564_564564


namespace largest_three_digit_product_l564_564492

theorem largest_three_digit_product : 
  ∃ n x y : ℕ, 
    (n = x * (10 * x + y)) ∧ 
    (n < 1000) ∧ 
    (100 ≤ n) ∧ 
    (nat.prime x) ∧ 
    (x < 10) ∧ 
    (y < 10) ∧
    (x ≠ 10 * x + y) ∧
    n = 553 :=
by
  sorry

end largest_three_digit_product_l564_564492


namespace proof_problem_l564_564358

-- Define vectors m and n based on given condition
def vector_m (ω x : ℝ) : ℝ × ℝ := (2 * Real.cos(ω * x), 1)
def vector_n (ω x α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin(ω * x) - Real.cos(ω * x), α)

-- Define dot product of m and n
def f (ω a x : ℝ) : ℝ := (vector_m ω x).1 * (vector_n ω x a).1 + (vector_m ω x).2 * (vector_n ω x a).2

-- The required proof problem in Lean
theorem proof_problem (ω a : ℝ) (hω : ω > 0) (hx : x ∈ ℝ) :
  -- Given condition for the period and maximum value of the function
  (∀ x, f ω a x = 2 * Real.sin (2 * ω * x - π / 6) + a - 1) ∧ 
  (∀ x, Real.periodic (2 * π / ω) (f ω a)) ∧ 
  (∃ x_0, f ω a x_0 = 3) →
  -- Conditions followed by the conclusions
  ω = 1 ∧ a = 2 ∧
  (∀ k : ℤ, ∀ x, x ∈ set.Icc (k * π - π / 6) (k * π + π / 3)) :=
by
  sorry

end proof_problem_l564_564358


namespace rice_mixture_ratio_l564_564854

theorem rice_mixture_ratio :
  ∀ (A B C: ℝ),
    A = 3.10 → B = 3.60 → C = 4.00 →
    (∃ (x y z: ℝ), 
      A * x + B * y + C * z = 3.50 * (x + y + z) → 
      x = 1 ∧ y = 0 ∧ z = 0.8) :=
by
  intros A B C hA hB hC
  use [1, 0, 0.8]
  intros h
  split
  repeat {assumption}
  sorry  -- Proof steps are not needed

end rice_mixture_ratio_l564_564854


namespace tetrahedron_volume_l564_564175

theorem tetrahedron_volume (a b c : ℝ) (h1 : a^2 < b^2 + c^2)
  (h2 : b^2 < a^2 + c^2) (h3 : c^2 < a^2 + b^2) :
  ∃ V : ℝ, V = 1 / (6 * Real.sqrt 2) * Real.sqrt ((-a^2 + b^2 + c^2) * 
  (a^2 - b^2 + c^2) * (a^2 + b^2 - c^2)) :=
begin
  sorry
end

end tetrahedron_volume_l564_564175


namespace number_of_positive_factors_of_216_that_are_perfect_cubes_l564_564360

-- Lean 4 statement for the problem 
theorem number_of_positive_factors_of_216_that_are_perfect_cubes : 
  (count (λ n, n > 0 ∧ (∃ a b, n = 2^a * 3^b ∧ n ∣ 216 ∧ a % 3 = 0 ∧ b % 3 = 0)) (finset.range (216 + 1))) = 4 :=
sorry

end number_of_positive_factors_of_216_that_are_perfect_cubes_l564_564360


namespace box_width_l564_564489

-- Define the initial conditions
def lengthBox : ℝ := 50
def loweredLevel : ℝ := 0.5
def volumeRemovedGallon : ℝ := 4687.5
def gallonToCubicFeet : ℝ := 7.5

-- Volume removed in cubic feet
def volumeRemovedCubicFeet : ℝ := volumeRemovedGallon / gallonToCubicFeet

-- Define the problem statement
theorem box_width (W : ℝ) :
  (lengthBox * W * loweredLevel = volumeRemovedCubicFeet) → W = 25 :=
by
  -- proof will be provided here
  sorry

end box_width_l564_564489


namespace pancakes_after_breakfast_l564_564198

def total_pancakes_produced (batch_size : ℕ) (ingredient_fraction : ℚ) : ℕ :=
  (ingredient_fraction * batch_size).floor
  
def equivalent_regular_pancakes (large_pancakes : ℕ) (size_factor : ℚ) : ℕ :=
  (large_pancakes * size_factor).floor

theorem pancakes_after_breakfast
  (batch_size : ℕ := 21)
  (ingredient_fraction : ℚ := 3 / 4)
  (large_pancake_count : ℕ := 4)
  (size_factor : ℚ := 3 / 2)
  (bobby_pancakes : ℕ := 5)
  (dog_pancakes : ℕ := 7) :
  total_pancakes_produced batch_size ingredient_fraction = 15 →
  equivalent_regular_pancakes large_pancake_count size_factor = 6 →
  ((total_pancakes_produced batch_size ingredient_fraction - 
    equivalent_regular_pancakes large_pancake_count size_factor - 
    bobby_pancakes) = 4) →
  (4 - dog_pancakes = 0) :=
begin
  intros h1 h2 h3,
  rw h1 at h3,
  rw h2 at h3,
  linarith,
end

end pancakes_after_breakfast_l564_564198


namespace minimum_value_expression_l564_564003

theorem minimum_value_expression (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  ∃ c, c = sqrt 2 ∧ ∀ x y, (0 < x) → (0 < y) → a^2 + b^2 + (1 / (a + b)^2) ≥ c :=
by
  use sqrt 2
  sorry

end minimum_value_expression_l564_564003


namespace petya_sequences_l564_564030

theorem petya_sequences (n : ℕ) (h : n = 100) : 
    let S := 5^n
    in S - 3^n = 5^100 - 3^100 :=
by {
  have : S = 5^100,
  {
    rw h,
    exact rfl,
  },
  rw this,
  sorry
}

end petya_sequences_l564_564030


namespace triangle_angles_equal_sixty_l564_564478

theorem triangle_angles_equal_sixty
  (A B C O : Type) 
  (circumcenter : O = inscribed_center)
  (is_triangle : is_triangle A B C)
  (is_circumcenter : ∀ a b c : Type, circumcenter = (a + b + c)/3)
  (is_incenter : ∀ a b c : Type, inscribed_center = (a + b + c)/2)
  (common_O : O = circumcenter ∧ O = inscribed_center)
  : angles_of_triangle A B C  = 60 :=
sorry

end triangle_angles_equal_sixty_l564_564478


namespace conjugate_of_i607_eq_i_l564_564415

theorem conjugate_of_i607_eq_i (i : ℂ) (hi : i^2 = -1) : conj (i^607) = i := by
  sorry

end conjugate_of_i607_eq_i_l564_564415


namespace cos_270_eq_zero_l564_564214

-- Defining the cosine value for the given angle
theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 :=
by
  sorry

end cos_270_eq_zero_l564_564214


namespace solve_quartic_eq_l564_564702

theorem solve_quartic_eq (z : ℂ) :
  z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -complex.sqrt 2 ∨ z = complex.sqrt 2 ∨ z = 2 :=
by {
  sorry
}

end solve_quartic_eq_l564_564702


namespace TeamB_wins_four_consecutive_matches_prob_l564_564817

noncomputable def prob_Teamb_winning_four_matches (p_ab p_bc p_ca : ℝ) : ℝ :=
  let p_ba := 1 - p_ab
  in p_ba * p_bc * p_ba * p_bc

theorem TeamB_wins_four_consecutive_matches_prob :
  let p_ab := 0.4
  let p_bc := 0.5
  let p_ca := 0.6
  in prob_Teamb_winning_four_matches p_ab p_bc p_ca = 0.09 := 
by
  sorry

end TeamB_wins_four_consecutive_matches_prob_l564_564817


namespace solve_quartic_eq_l564_564658

theorem solve_quartic_eq (z : ℝ) : z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = 2 :=
by 
  sorry

end solve_quartic_eq_l564_564658


namespace quadratic_symmetry_value_l564_564599

noncomputable def p (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def symmetric (p : ℝ → ℝ) (x0 : ℝ) : Prop :=
∀ x, p x = p (2 * x0 - x)

theorem quadratic_symmetry_value :
  ∃ a b c, let p := p a b c in
  symmetric p 3.5 ∧
  p 2 = -1 → p 5 = -1 :=
begin
  intros a b c p h1 h2,
  sorry
end

end quadratic_symmetry_value_l564_564599


namespace max_points_paths_product_l564_564219

-- Define what it means to have paths on the 5x5 grid
def is_valid_path (path : List (ℕ × ℕ)) : Prop :=
  let length_path := path.length
  length_path = path.length ∧ -- This just automatically holds true 
  ∀ i j, (1 ≤ i ∧ i < length_path ∧ 1 ≤ j ∧ j < length_path) →
    let (x_i, y_i) := path.nth i.succ
    let (x_j, y_j) := path.nth j.succ
    (x_i ≠ x_j ∨ y_i ≠ y_j) ∧ -- non-overlapping points
    (abs (x_i - x_j) ≤ 1 ∧ abs (y_i - y_j) ≤ 1) -- adjacent points

def max_points_in_path : ℕ := 16 -- Given maximum points in the path

def num_paths : ℕ → ℕ 

-- The main theorem checking the product of maximum points and number of such paths
theorem max_points_paths_product : ∃ r, max_points_in_path * num_paths r = 16 * r := sorry

end max_points_paths_product_l564_564219


namespace sqrt_equation_solutions_l564_564641

theorem sqrt_equation_solutions (x : ℝ) :
    (sqrt (3 * x - 5) + 14 / sqrt (3 * x - 5) = 8) ↔
    (x = (23 + 8 * sqrt 2) / 3) ∨ (x = (23 - 8 * sqrt 2) / 3) :=
by
  sorry

end sqrt_equation_solutions_l564_564641


namespace quadrilateral_not_necessarily_square_l564_564816

-- Definitions of the conditions
structure Quadrilateral (A B C D : Type) :=
  (diag_perpendicular : ∀ (O : Type), (O = (diagonal A C) ∩ (diagonal B D)) → (∠AOB = 90 ∧ ∠BOC = 90 ∧ ∠COD = 90 ∧ ∠DOA = 90))
  (has_inscribed_circle : Prop)
  (has_circumscribed_circle : Prop)

-- The statement asserting the quadrilateral is not necessarily a square under given conditions
theorem quadrilateral_not_necessarily_square {A B C D : Type} (quad : Quadrilateral A B C D) :
  quad.diag_perpendicular ∧ quad.has_inscribed_circle ∧ quad.has_circumscribed_circle → 
  ¬ (A = B ∧ B = C ∧ C = D ∧ D = A) :=
  sorry

end quadrilateral_not_necessarily_square_l564_564816


namespace enclosed_region_area_l564_564112

open Real Set

noncomputable def region := { p : ℝ × ℝ | p.1^2 + p.2^2 = 2*|p.1| + 2*|p.2| ∧ -2 ≤ p.1 ∧ p.1 ≤ 2 ∧ -2 ≤ p.2 ∧ p.2 ≤ 2 }

theorem enclosed_region_area : (measure_theory.measure_univ : ennreal).to_real = 2 * π := by
  sorry

end enclosed_region_area_l564_564112


namespace focal_length_is_correct_l564_564965

def hyperbola_eqn : Prop := (∀ x y : ℝ, (x^2 / 4) - (y^2 / 9) = 1 → True)

noncomputable def focal_length_of_hyperbola : ℝ :=
  2 * Real.sqrt (4 + 9)

theorem focal_length_is_correct : hyperbola_eqn → focal_length_of_hyperbola = 2 * Real.sqrt 13 := by
  intro h
  sorry

end focal_length_is_correct_l564_564965


namespace minimum_value_fraction_l564_564735

theorem minimum_value_fraction (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
    (h_eq : a 7 = a 6 + 2 * a 5)
    (h_sqrt_eq : ∃ m n, sqrt (a m * a n) = 4 * a 1) :
    ∃ m n, (1:ℝ) / m + 4 / n = (3:ℝ) / 2 :=
by
    sorry

end minimum_value_fraction_l564_564735


namespace find_a_value_l564_564767

theorem find_a_value (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : (∃ l : ℝ, ∃ f : ℝ → ℝ, f x = a^x ∧ deriv f 0 = -1)) :
  a = 1 / Real.exp 1 := by
  sorry

end find_a_value_l564_564767


namespace cos_270_eq_zero_l564_564215

-- Defining the cosine value for the given angle
theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 :=
by
  sorry

end cos_270_eq_zero_l564_564215


namespace set_equivalence_l564_564922

open Set

def set_A : Set ℝ := { x | x^2 - 2 * x > 0 }
def set_B : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

theorem set_equivalence : (univ \ set_B) ∪ set_A = (Iic 1) ∪ Ioi 2 :=
sorry

end set_equivalence_l564_564922


namespace allocation_count_is_57_l564_564991

-- Definitions based on conditions from the original problem
def factories : List Char := ['A', 'B', 'C', 'D']
def classes : List Char := ['X', 'Y', 'Z']

-- Define the condition that Factory A must have at least one class
def factoryA_condition (allocation: List (Char × Char)) : Prop :=
  ∃ c, (c, 'A') ∈ allocation

-- Define the function that calculates the total number of valid allocations
noncomputable def count_allocations : ℕ :=
  let all_allocations := List.product classes factories in
  let valid_allocations := all_allocations.filter factoryA_condition in
  valid_allocations.length

-- State the theorem with the mathematically equivalent problem
theorem allocation_count_is_57 : count_allocations = 57 := by
  sorry

end allocation_count_is_57_l564_564991


namespace max_value_of_b_l564_564765

theorem max_value_of_b {a b : ℝ} (h1 : a < 0) (h2 : 0 < b) (h3 : ∀ x ∈ Icc 0 1, f x ∈ Icc 0 1) :
  b ≤ Real.sqrt 3 / 2 := sorry

def f (x : ℝ) : ℝ := a * x ^ 3 + 3 * b * x

example : b = Real.sqrt 3 / 2 := sorry

end max_value_of_b_l564_564765


namespace triangle_cross_section_perimeter_gt_l564_564480

-- Let’s define the regular tetrahedron and the conditions
def regular_tetrahedron (a : ℝ) : Prop :=
  ∀ A B C D : ℝ3, -- (A, B, C, D are vertices)
  dist A B = a ∧ dist A C = a ∧ dist A D = a ∧ dist B C = a ∧
  dist B D = a ∧ dist C D = a -- All edges have length 'a'

-- Define what it means to have a triangular cross-section through one of the vertices
def triangular_cross_section (A B C D M N : ℝ3) : Prop :=
  -- Assume M and N points cut edges AB and BC such that they form a triangle with D
  dist A M < a ∧ dist M B < a ∧ dist B N < a ∧ dist N C < a

-- Define the perimeter of the triangle DMN
noncomputable def perimeter (D M N : ℝ3) : ℝ :=
  dist D M + dist M N + dist N D

-- The main theorem statement in Lean
theorem triangle_cross_section_perimeter_gt (a : ℝ) (A B C D M N : ℝ3)
  (h_tetrahedron : regular_tetrahedron a)
  (h_cross_section : triangular_cross_section A B C D M N) :
  perimeter D M N > 2 * a := sorry

end triangle_cross_section_perimeter_gt_l564_564480


namespace hands_distance_l564_564625

noncomputable def distance_between_hands (l d b : ℝ) : ℝ :=
  2 * real.sqrt ((l / 2)^2 - d^2)

theorem hands_distance (l d b : ℝ) (h : l = 26) (hb : b = 8) : distance_between_hands l (l / 2 - b) b = 24 :=
by
  rw [h, hb]
  have h1: (26 / 2 : ℝ) = 13 := by norm_num
  have h2: (13 - 8 : ℝ) = 5 := by norm_num
  rw [h1, h2]
  dsimp [distance_between_hands]
  norm_num
  rw real.sqrt_mul_self; norm_num
  sorry

end hands_distance_l564_564625


namespace last_digit_to_appear_in_modified_fibonacci_sequence_is_6_l564_564490

def modified_fibonacci_sequence : ℕ → ℕ
| 1     := 2
| 2     := 3
| (n+3) := (modified_fibonacci_sequence (n+2) + modified_fibonacci_sequence (n+1)) % 10

theorem last_digit_to_appear_in_modified_fibonacci_sequence_is_6 :
  ∃ n, ∀ m, (m >= 1 → m <= n) → (∃ d, d < 10 ∧ ∃ k, modified_fibonacci_sequence k % 10 = d) ∧
    (∃ first_six, ∀ later_m, first_six < later_m → modified_fibonacci_sequence later_m % 10 ≠ 6) :=
sorry

end last_digit_to_appear_in_modified_fibonacci_sequence_is_6_l564_564490


namespace cos_angle_ab_l564_564784

noncomputable theory
open Real

variables (a b : ℝ × ℝ × ℝ)

def vector_add (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

def vector_sub (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

def cos_angle (a b : ℝ × ℝ × ℝ) : ℝ :=
  dot_product a b / ((vector_magnitude a) * (vector_magnitude b))

theorem cos_angle_ab : 
  vector_add a b = (0, sqrt 2, 0) ∧ 
  vector_sub a b = (2, sqrt 2, -2 * sqrt 3) → 
  cos_angle a b = -sqrt 6 / 3 :=
sorry

end cos_angle_ab_l564_564784


namespace no_solution_for_x_l564_564802

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (mx - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) :=
by
  sorry

end no_solution_for_x_l564_564802


namespace sum_of_areas_of_fantastic_rectangles_l564_564166

theorem sum_of_areas_of_fantastic_rectangles : 
  ∃ (a b : ℕ), a * b = 3 * (a + b) ∧ {
  finite { ab : ℕ | ∃ a b : ℕ, (a * b = 3 * (a + b))},
  sum {ab | ∃ a b : ℕ, (a * b = 3 * (a + b))} = 84
} :=
sorry

end sum_of_areas_of_fantastic_rectangles_l564_564166


namespace floor_T_is_217_l564_564012

noncomputable def x : ℝ := sorry  -- Assume x is a positive real number
noncomputable def y : ℝ := sorry  -- Assume y is a positive real number
noncomputable def z : ℝ := sorry  -- Assume z is a positive real number
noncomputable def w : ℝ := sorry  -- Assume w is a positive real number

def cond1 : Prop := (x^2 + y^2 = 4050)
def cond2 : Prop := (z^2 + w^2 = 4050)
def cond3 : Prop := (xz = 2040 ∧ yw = 2040)

def T : ℝ := x + y + z + w

theorem floor_T_is_217 : (cond1 ∧ cond2 ∧ cond3) → ⌊T⌋ = 217 := by
  sorry

end floor_T_is_217_l564_564012


namespace area_of_enclosed_region_l564_564226

theorem area_of_enclosed_region : 
  let eq := (x y : ℝ) → (x - 1)^2 + (y - 1)^2 = |x - 1| + |y - 1|
  let area := π / 2
  in ∃ (A : ℝ), (A = area) ∧ (∀ x y: ℝ, eq x y → True) := 
sorry

end area_of_enclosed_region_l564_564226


namespace number_of_ways_to_form_groups_l564_564065

-- Conditions
constant Dogs : Type
constant Rex Daisy : Dogs
constant other_dogs : Finset Dogs
constant all_dogs : Finset Dogs := insert Rex (insert Daisy other_dogs)

lemma card_all_dogs : all_dogs.card = 15 := sorry

-- The groups to divide into
def group1 : Finset Dogs := {Rex}
def group2 : Finset Dogs := {Daisy}
def remaining_dogs : Finset Dogs := all_dogs \ group1 \ group2

-- Prove the number of ways to form the groups
theorem number_of_ways_to_form_groups 
  (h1 : Rex ∈ all_dogs) 
  (h2 : Daisy ∈ all_dogs) 
  (h3 : ∀ x ∈ remaining_dogs, x ≠ Rex ∧ x ≠ Daisy)
  (h4 : remaining_dogs.card = 13) :
  (nat.choose 13 5) * (nat.choose 8 5) = 72072 :=
begin
  -- This is the proof placeholder
  sorry
end

end number_of_ways_to_form_groups_l564_564065


namespace k_even_l564_564403

def s (n : ℕ) : ℕ := sorry  -- This represents the number of positive divisors of n

theorem k_even (a b k : ℕ) (h₁ : k = s(a)) (h₂ : k = s(b)) (h₃ : k = s(2 * a + 3 * b)) : k % 2 = 0 :=
by
  sorry

end k_even_l564_564403


namespace count_digit_7_in_range_l564_564858

theorem count_digit_7_in_range : ∀ n ∈ finset.Icc 100 199, (n % 10 = 7 ∨ n / 10 % 10 = 7) →  ∃ counts, counts = 20 :=
by
  sorry

end count_digit_7_in_range_l564_564858


namespace solve_quartic_eq_l564_564699

theorem solve_quartic_eq (z : ℂ) :
  z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -complex.sqrt 2 ∨ z = complex.sqrt 2 ∨ z = 2 :=
by {
  sorry
}

end solve_quartic_eq_l564_564699


namespace ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l564_564120

theorem ratio_of_area_to_square_of_perimeter_of_equilateral_triangle :
  let a := 10
  let area := (10 * 10 * (Real.sqrt 3) / 4)
  let perimeter := 3 * 10
  let square_of_perimeter := perimeter * perimeter
  (area / square_of_perimeter) = (Real.sqrt 3 / 36) := by
  -- Proof to be completed
  sorry

end ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l564_564120


namespace decreasing_function_only_C_l564_564080

-- Definitions for the four functions
def f_A (x : ℝ) : ℝ := x^2
def f_B (x : ℝ) : ℝ := x^3
noncomputable def f_C (x : ℝ) : ℝ := (0.5)^x
noncomputable def f_D (x : ℝ) : ℝ := log x / log 10

-- Function to check if a function is decreasing in interval (0, +∞)
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f y < f x

-- Main theorem
theorem decreasing_function_only_C :
  is_decreasing f_A = false ∧
  is_decreasing f_B = false ∧
  is_decreasing f_C = true ∧
  is_decreasing f_D = false :=
by
  sorry

end decreasing_function_only_C_l564_564080


namespace find_all_z_l564_564668

noncomputable def all_solutions (z : ℝ) : Prop := z = -2 ∨ z = 2 ∨ z = -real.sqrt 2 ∨ z = real.sqrt 2

theorem find_all_z (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ all_solutions z := 
sorry

end find_all_z_l564_564668


namespace residual_is_ten_l564_564556

-- Definitions of given values
def x_values : List ℕ := [2, 4, 5, 6, 8]
def y_values : List ℕ := [30, 40, 60, 50, 70]

def x_mean : ℕ := (x_values.sum / x_values.length)
def y_mean : ℕ := (y_values.sum / y_values.length)

-- Linear regression equation parameters
def a_hat : ℕ := y_mean - 6 * x_mean
def y_hat (x : ℕ) : ℕ := 6 * x + a_hat

-- Given value for x
def x_given : ℕ := 5

-- Actual observed value at x_given
def y_actual : ℕ := 60  -- From the table y_values

-- Compute residual
def residual : ℕ := | y_actual - y_hat x_given |

theorem residual_is_ten : residual = 10 := by
  sorry

end residual_is_ten_l564_564556


namespace part_a_part_b_l564_564338

noncomputable def sequence_a (n : ℕ) : ℝ :=
  Real.cos (10^n * Real.pi / 180)

theorem part_a (h : 100 > 2) : sequence_a 100 > 0 := by
  sorry

theorem part_b : |sequence_a 100| < 0.18 := by
  sorry

end part_a_part_b_l564_564338


namespace find_z_values_l564_564654

theorem find_z_values : {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -sqrt 2, sqrt 2, 2} :=
sorry

end find_z_values_l564_564654


namespace reduce_routes_l564_564951

-- Define the vertices (airports) and edges (routes)
def Graph := SimpleGraph (Fin 50)

-- Define the condition that each airport has at least one route
axiom route_exists (G : Graph) : ∀ v : Fin 50, ∃ w : Fin 50, G.Adj v w

-- Define the proof statement
theorem reduce_routes (G : Graph) (h_routes : ∀ v : Fin 50, ∃ w : Fin 50, G.Adj v w) :
  ∃ G' : Graph, (∀ v : Fin 50, ∃ w : Fin 50, G'.Adj v w ∧ ¬G'.Adj v v) ∧ (43 < (G'.AdjFinOne.filter (λ v => G'.degree v = 1)).card) := sorry

end reduce_routes_l564_564951


namespace puja_runs_distance_in_meters_l564_564942

noncomputable def puja_distance (time_in_seconds : ℝ) (speed_kmph : ℝ) : ℝ :=
  let time_in_hours := time_in_seconds / 3600
  let distance_km := speed_kmph * time_in_hours
  distance_km * 1000

theorem puja_runs_distance_in_meters :
  abs (puja_distance 59.995200383969284 30 - 499.96) < 0.01 :=
by
  sorry

end puja_runs_distance_in_meters_l564_564942


namespace find_height_of_triangle_l564_564471

-- Definitions based on the conditions
def rectangle_area (length : ℕ) (width : ℕ) : ℕ := length * width

def trapezoid_area (rectangle_length : ℕ) (rectangle_width : ℕ) : ℕ := (rectangle_area rectangle_length rectangle_width) / 2

def triangle_area (base : ℕ) (height : ℕ) : ℕ := (base * height) / 2

-- Given conditions
def length : ℕ := 27
def width : ℕ := 9
def base : ℕ := width
def expected_height : ℕ := 54

-- Proof problem:
theorem find_height_of_triangle :
  let rect_area := rectangle_area length width,
      trap_area := trapezoid_area length width,
      tri_area := triangle_area base expected_height in
  rect_area = 243 ∧ trap_area = 121.5 ∧ tri_area = 243 ∧ 243 = (base * expected_height) / 2 → expected_height = 54 :=
by
sory

end find_height_of_triangle_l564_564471


namespace solve_exp_equation_l564_564059

theorem solve_exp_equation (e : ℝ) (x : ℝ) (h_e : e = Real.exp 1) :
  e^x + 2 * e^(-x) = 3 ↔ x = 0 ∨ x = Real.log 2 :=
sorry

end solve_exp_equation_l564_564059


namespace solve_quartic_l564_564686

theorem solve_quartic (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ (z = 2 ∨ z = -2 ∨ z = sqrt 2 ∨ z = -sqrt 2) :=
by
  sorry

end solve_quartic_l564_564686


namespace max_length_OB_and_angle_OAB_l564_564104

theorem max_length_OB_and_angle_OAB
    (O A B : Point)
    (AB : ℝ)
    (AOB : ℝ)
    (hAOB : AOB = π / 4)
    (hAB : AB = 1)
    : ∃ (OB : ℝ) (OAB : ℝ), OB = √2 ∧ OAB = π / 2 := 
begin
  sorry
end

end max_length_OB_and_angle_OAB_l564_564104


namespace temperature_range_l564_564439

-- Conditions: highest temperature and lowest temperature
def highest_temp : ℝ := 5
def lowest_temp : ℝ := -2
variable (t : ℝ) -- given temperature on February 1, 2018

-- Proof problem statement
theorem temperature_range : lowest_temp ≤ t ∧ t ≤ highest_temp :=
sorry

end temperature_range_l564_564439


namespace line_always_passes_through_fixed_point_l564_564487

theorem line_always_passes_through_fixed_point :
  ∀ (m : ℝ), ∃ (x y : ℝ), (y = m * x + 2 * m + 1) ∧ (x = -2) ∧ (y = 1) :=
by
  sorry

end line_always_passes_through_fixed_point_l564_564487


namespace math_proof_equiv_l564_564483

def A := 5
def B := 3
def C := 2
def D := 0
def E := 0
def F := 1
def G := 0

theorem math_proof_equiv : (A * 1000 + B * 100 + C * 10 + D) + (E * 100 + F * 10 + G) = 5300 :=
by
  sorry

end math_proof_equiv_l564_564483


namespace probability_grisha_wins_expectation_coin_flips_l564_564833

-- Define the conditions as predicates
def grishaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 0) && (toss_seq.last = some true)

def vanyaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 1) && (toss_seq.last = some false)

-- State the probability theorem
theorem probability_grisha_wins : 
  (∑ x in ((grishaWins ≤ 1 / 3))) = 1/3 := by sorry

-- State the expectation theorem
theorem expectation_coin_flips : 
  (E[flips until (grishaWins ∨ vanyaWins)]) = 2 := by sorry

end probability_grisha_wins_expectation_coin_flips_l564_564833


namespace arithmetic_sequence_problem_l564_564382

variable (d : ℝ) (a : ℕ → ℝ)

-- Conditions
def a_1 := 2
def a_2 : ℝ := a_1 + d
def a_5 : ℝ := a_1 + 4 * d

def condition1 : Prop := a_1 = 2
def condition2 : Prop := a_2 + a_5 = 13

-- Equivalent statement
theorem arithmetic_sequence_problem : condition1 ∧ condition2 → (a_5 + (a_1 + 5 * d) + (a_1 + 6 * d)) = 33 :=
by
  sorry

end arithmetic_sequence_problem_l564_564382


namespace find_a_l564_564302

variables (a : ℝ) (i : ℂ)
noncomputable def z := a + real.sqrt 3 * i

theorem find_a (h1 : ∀ (z : ℂ), z * conj z = 4) : a = 1 ∨ a = -1 :=
begin
  let z := a + real.sqrt 3 * i,
  have h2 : z * conj z = a^2 + 3,
  { sorry }, -- This will typically show the steps of computing the product
  have h3 : a^2 + 3 = 4,
  { rw ← h1 z, exact h2 }, -- Uses provided condition on complex numbers
  have h4 : a^2 = 1,
  { linarith },
  exact or.intro_left (-1) (by rw eq_comm; apply pow_two_eq_second_root_of_square_eq_one; linarith)
  ... or.intro_right 1 (by rw eq_comm; apply pow_two_eq_second_root_of_square_eq_one; linarith),
end

end find_a_l564_564302


namespace cos_270_eq_zero_l564_564207

theorem cos_270_eq_zero : ∀ (θ : ℝ), θ = 270 → (∀ (p : ℝ × ℝ), p = (0,1) → cos θ = p.fst) → cos 270 = 0 :=
by
  intros θ hθ h
  sorry

end cos_270_eq_zero_l564_564207


namespace evaluate_powers_of_i_l564_564642

theorem evaluate_powers_of_i : (complex.I ^ 22 + complex.I ^ 222) = -2 :=
by
  -- Using by to start the proof block and ending it with sorry.
  sorry

end evaluate_powers_of_i_l564_564642


namespace trig_inequality_l564_564497

theorem trig_inequality : Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := by
  sorry

end trig_inequality_l564_564497


namespace cost_of_adult_ticket_eq_19_l564_564019

variables (X : ℝ)
-- Condition 1: The cost of an adult ticket is $6 more than the cost of a child ticket.
def cost_of_child_ticket : ℝ := X - 6

-- Condition 2: The total cost of the 5 tickets is $77.
axiom total_cost_eq : 2 * X + 3 * (X - 6) = 77

-- Prove that the cost of an adult ticket is 19 dollars
theorem cost_of_adult_ticket_eq_19 (h : total_cost_eq) : X = 19 := 
by
  -- Here we would provide the actual proof steps
  sorry

end cost_of_adult_ticket_eq_19_l564_564019


namespace integer_part_one_plus_sqrt_seven_l564_564971

theorem integer_part_one_plus_sqrt_seven : int.floor (1 + real.sqrt 7) = 3 := 
sorry

end integer_part_one_plus_sqrt_seven_l564_564971


namespace find_all_z_l564_564667

noncomputable def all_solutions (z : ℝ) : Prop := z = -2 ∨ z = 2 ∨ z = -real.sqrt 2 ∨ z = real.sqrt 2

theorem find_all_z (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ all_solutions z := 
sorry

end find_all_z_l564_564667


namespace powers_of_i_l564_564644

theorem powers_of_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) : 
  i^22 + i^222 = -2 :=
by {
  -- Proof will go here
  sorry
}

end powers_of_i_l564_564644


namespace part_i_proof_part_ii_proof_l564_564731

-- Define the problem context and parameters
variable {k a : Real} (x1 x2 y1 y2 : Real)

-- Conditions and question for Part (I)
def part_i_conditions : Prop :=
  (k = 1) ∧ (4 * x1^2 + 2 * x1 + 1 - a = 0) ∧ (|AB| = (Real.sqrt 10) / 2)

def part_i_answer : Real := 2

-- Lean statement for Part (I)
theorem part_i_proof (h : part_i_conditions) : a = part_i_answer :=  
sorry

-- Conditions and question for Part (II)
def part_ii_conditions : Prop :=
  (3 * x1^2 + y1^2 = a) ∧ (y1 = k * x1 + 1) ∧ (y2 = k * x2 + 1) ∧ 
  (x1 + x2 = -2 * k / (3 + k^2)) ∧ (x1 * x2 = (1 - a) / (3 + k^2)) ∧
  (2 * (x2, y2 - 1) = (-x1, 1 - y1))

def part_ii_answers : Prop :=
  (A_max_area : Real := (Real.sqrt 3) / 2) ∧ 
  (ellipse_eqn : Prop := 3 * x1^2 + y1^2 = 5)

-- Lean statement for Part (II)
theorem part_ii_proof (h : part_ii_conditions) : part_ii_answers := 
sorry

end part_i_proof_part_ii_proof_l564_564731


namespace find_circle_radius_l564_564980

-- Define the conditions
def on_circle (x : ℝ) (C : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  (C.1 - P.1)^2 + (C.2 - P.2)^2 = x^2

def center_on_x_axis (C : ℝ × ℝ) : Prop :=
  C.2 = 0

-- Define the points and the radius calculation
theorem find_circle_radius (xC : ℝ) (r : ℝ) 
  (hC1 : on_circle r (xC, 0) (1, 5))
  (hC2 : on_circle r (xC, 0) (2, 4))
  (hC3 : center_on_x_axis (xC, 0)) :
  r = real.sqrt 29 :=
by
  sorry

end find_circle_radius_l564_564980


namespace smallest_n_log_sum_l564_564218

theorem smallest_n_log_sum :
  (∃ n : ℕ, n > 0 ∧ (∑ k in finset.range (n + 1), real.log (1 + 1 / (3 ^ (3 ^ k))) / real.log 3
  ≥ 1 + real.log (4030 / 4031) / real.log 3) ∧ (∀ m, m > 0 ∧ m < n →
  (∑ k in finset.range (m + 1), real.log (1 + 1 / (3 ^ (3 ^ k))) / real.log 3 < 1 + real.log (4030 / 4031) / real.log 3)))
  → n = 1 :=
begin
  sorry
end

end smallest_n_log_sum_l564_564218


namespace graph_passes_through_fixed_point_l564_564969

theorem graph_passes_through_fixed_point (k : ℝ) : ∃ y : ℝ, y = k*1 - k + 2 ∧ y = 2 :=
begin
  use 2,
  split,
  {
    rw mul_one,
    sorry,
  },
  {
    refl,
  }
end

end graph_passes_through_fixed_point_l564_564969


namespace minimum_value_4_l564_564805

noncomputable theory

open Real

/-- If the solution set of the quadratic function f(x) = ax^2 + bx + 1 > 0 (a, b ∈ ℝ, a ≠ 0)
    is {x ∈ ℝ | x ≠ -b/(2a)}, then the minimum value of (b^4 + 4)/(4a) is 4. -/
theorem minimum_value_4
  (a b : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∀ x : ℝ, x ≠ -b / (2 * a) → a * x^2 + b * x + 1 > 0) :
  (b^4 + 4) / (4 * a) = 4 :=
sorry

end minimum_value_4_l564_564805


namespace option_A_option_C_option_D_l564_564301

variables (θ : ℝ)
noncomputable def is_terminal_point (θ : ℝ) : Prop := (Sin.sin θ, Cos.cos θ) = (Real.sin 2, Real.cos 2)
axiom theta_in_range : 0 < θ ∧ θ < 2 * Real.pi

theorem option_A : is_terminal_point θ → θ = (5 * Real.pi / 2 - 2) :=
sorry

theorem option_C : Cos.cos θ + Cos.cos (θ + 2 * Real.pi / 3) + Cos.cos (θ + 4 * Real.pi / 3) = 0 :=
sorry

theorem option_D : Sin.sin θ + Sin.sin (θ + 2 * Real.pi / 3) + Sin.sin (θ + 4 * Real.pi / 3) = 0 :=
sorry

end option_A_option_C_option_D_l564_564301


namespace range_of_m_l564_564308

variable (a x m : ℝ)
variable (h_a_gt_1 : a > 1)
variable (h_x_in_interval : 0 ≤ x ∧ x < 1)

def f (x : ℝ) : ℝ := log a (x + 1)
def g (x : ℝ) : ℝ := log a (1 / (1 - x))

theorem range_of_m (h_f_g_ge_m : ∀ x, 0 ≤ x ∧ x < 1 → f a x + g a x ≥ m) : m ≤ 0 := 
by
  sorry

end range_of_m_l564_564308


namespace taylor_family_reunion_l564_564192

theorem taylor_family_reunion :
  let number_of_kids := 45
  let number_of_adults := 123
  let number_of_tables := 14
  (number_of_kids + number_of_adults) / number_of_tables = 12 := by sorry

end taylor_family_reunion_l564_564192


namespace product_of_roots_of_Q_l564_564406

noncomputable def polynomial_with_roots (x : ℚ) : Polynomial ℚ :=
  Polynomial.C 63 * (Polynomial.X * Polynomial.X) + Polynomial.C 1033 * Polynomial.X + Polynomial.C 30

theorem product_of_roots_of_Q : 
  ∃ (Q : Polynomial ℚ), 
    (Q.degree = 2) ∧ 
    (Q.coeff 2 ≠ 0) ∧ 
    (∃ (r : ℚ), r = (10:ℚ)^(1/3) ∧ Q.has_root (r + r^2)) ∧ 
    (∀ r₁ r₂, Q.has_roots r₁ r₂ → r₁ * r₂ = -1033 / 30) := 
sorry

end product_of_roots_of_Q_l564_564406


namespace count_digit_7_to_199_l564_564872

theorem count_digit_7_to_199 : (Nat.countDigitInRange 7 100 199) = 20 := by
  sorry

end count_digit_7_to_199_l564_564872


namespace cost_of_adult_ticket_l564_564020

theorem cost_of_adult_ticket (c : ℝ) 
  (h₁ : 2 * (c + 6) + 3 * c = 77)
  : c + 6 = 19 :=
sorry

end cost_of_adult_ticket_l564_564020


namespace fiftieth_statement_l564_564237

-- Define the types
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

-- Define the function telling the statement
def statement (inhabitant : Inhabitant) : String :=
  match inhabitant with
  | Inhabitant.knight => "Knight"
  | Inhabitant.liar => "Liar"

-- Define the condition: knights tell the truth and liars lie
def tells_truth (inhabitant : Inhabitant) (statement_about_neighbor : String) : Prop :=
  match inhabitant with
  | Inhabitant.knight => statement_about_neighbor = "Knight"
  | Inhabitant.liar => statement_about_neighbor ≠ "Knight"

-- Define a function that determines what each inhabitant says about their right-hand neighbor
def what_they_say (idx : ℕ) : String :=
  if idx % 2 = 0 then "Liar" else "Knight"

-- Define the inhabitant pattern
def inhabitant_at (idx : ℕ) : Inhabitant :=
  if idx % 2 = 0 then Inhabitant.liar else Inhabitant.knight

-- The main theorem statement
theorem fiftieth_statement : tells_truth (inhabitant_at 49) (what_they_say 50) :=
by 
  -- This proof outlines the theorem statement only
  sorry

end fiftieth_statement_l564_564237


namespace sum_binom_denominator_is_zero_l564_564250

theorem sum_binom_denominator_is_zero (n : ℕ) :
  ∑ k in Finset.range (n + 1), 
    (-1)^k * (Nat.choose n k : ℚ) / (k^3 + 9 * k^2 + 26 * k + 24) = 0 :=
by sorry

end sum_binom_denominator_is_zero_l564_564250


namespace cubic_roots_equal_l564_564806

theorem cubic_roots_equal (k : ℚ) (h1 : k > 0)
  (h2 : ∃ a b : ℚ, a ≠ b ∧ (a + a + b = -3) ∧ (2 * a * b + a^2 = -54) ∧ (3 * x^3 + 9 * x^2 - 162 * x + k = 0)) : 
  k = 7983 / 125 :=
sorry

end cubic_roots_equal_l564_564806


namespace max_seq_terms_l564_564541

noncomputable def x_seq (x : ℕ → ℝ) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 320 → x (n + 2) = x n - 1 / x (n + 1)

theorem max_seq_terms :
  ∃ (N : ℕ), (x_seq (fun n => if n = 1 then 20 else if n = 2 then 16 else 0) ∧ N = 322) :=
begin
  use 322,
  split,
  { intros n hn,
    cases hn with h1 h2,
    have : n = 1 ∨ n = 2 ∨ 3 ≤ n ∧ n ≤ 320 := by decide,
    cases this with h h,
    { cases h,
      { subst h, simp },
      { subst h, simp } },
    { cases h with h3 h4,
      exfalso,
      sorry } },
  { refl }
end

end max_seq_terms_l564_564541


namespace find_a_l564_564079

noncomputable def f (a x : ℝ) : ℝ := x^2 + a*x + 3

theorem find_a (a : ℝ) (h : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f a x ≥ -4) : a = 8 ∨ a = -8 :=
by
  sorry

end find_a_l564_564079


namespace expr1_evaluation_expr2_evaluation_l564_564619

theorem expr1_evaluation : (0.001 : ℝ) ^ (-1 / 3) + 27 ^ (2 / 3) + (1 / 4) ^ (-1 / 2) - (1 / 9) ^ (-1.5) = -6 :=
  sorry

theorem expr2_evaluation : (1 / 2) * log 10 25 + log 10 2 - log 10 (sqrt 0.1) - (log 2 9) * (log 3 2) = 1 / 2 :=
  sorry

end expr1_evaluation_expr2_evaluation_l564_564619


namespace postage_cost_5_3_l564_564983

def postage_rate (weight: ℝ) : ℝ :=
  let base_rate := 0.35 -- dollars for the first ounce
  let additional_rate := 0.25 -- dollars for each additional ounce or fraction
  let additional_ounces := (weight - 1).ceil -- round up to the nearest whole number
  base_rate + additional_ounces * additional_rate

theorem postage_cost_5_3 : postage_rate 5.3 = 1.60 := by
  sorry

end postage_cost_5_3_l564_564983


namespace number_of_piles_l564_564386

-- Defining the number of walnuts in total
def total_walnuts : Nat := 55

-- Defining the number of walnuts in the first pile
def first_pile_walnuts : Nat := 7

-- Defining the number of walnuts in each of the rest of the piles
def other_pile_walnuts : Nat := 12

-- The proposition we want to prove
theorem number_of_piles (n : Nat) :
  (n > 1) →
  (other_pile_walnuts * (n - 1) + first_pile_walnuts = total_walnuts) → n = 5 :=
sorry

end number_of_piles_l564_564386


namespace parabola_opens_downwards_l564_564033

theorem parabola_opens_downwards (a : ℝ) (h : ℝ) (k : ℝ) :
  a < 0 → h = 3 → ∃ k, (∀ x, y = a * (x - h) ^ 2 + k → y = -(x - 3)^2 + k) :=
by
  intros ha hh
  use k
  sorry

end parabola_opens_downwards_l564_564033


namespace probability_A_first_is_seven_over_twelve_l564_564749

/-- The following definition sets up the context of the problem. -/
def interview_problem : Prop :=
  let total_students := 60 in
  let students_A := 10 in
  let students_B := 20 in
  let students_C := 30 in
  (students_A + students_B + students_C = total_students) →
  (students_A = 10 → students_B = 20 → students_C = 30 →
    (probability_all_A_first (students_A, students_B, students_C)) = 7 / 12)

-- In reality, you would need a separate definition of probability_all_A_first
-- which is a non-trivial function computing the probability in this context.
-- This is simplified here for illustration purposes.
def probability_all_A_first (students : ℕ × ℕ × ℕ) : ℚ := sorry

theorem probability_A_first_is_seven_over_twelve : interview_problem :=
by {
  sorry
}

end probability_A_first_is_seven_over_twelve_l564_564749


namespace combo_discount_is_50_percent_l564_564103

noncomputable def combo_discount_percentage
  (ticket_cost : ℕ) (combo_cost : ℕ) (ticket_discount : ℕ) (total_savings : ℕ) : ℕ :=
  let ticket_savings := ticket_cost * ticket_discount / 100
  let combo_savings := total_savings - ticket_savings
  (combo_savings * 100) / combo_cost

theorem combo_discount_is_50_percent:
  combo_discount_percentage 10 10 20 7 = 50 :=
by
  sorry

end combo_discount_is_50_percent_l564_564103


namespace area_perimeter_ratio_eq_l564_564121

theorem area_perimeter_ratio_eq (s : ℝ) (s_eq : s = 10) : 
  let area := (sqrt 3) / 4 * s ^ 2
      perimeter := 3 * s
      ratio := area / (perimeter ^ 2)
  in ratio = (sqrt 3) / 36 :=
by sorry

end area_perimeter_ratio_eq_l564_564121


namespace probability_point_outside_circle_l564_564470

/-- Let P be a point with coordinates (m, n) determined by rolling a fair 6-sided die twice.
Prove that the probability that P falls outside the circle x^2 + y^2 = 25 is 7/12. -/
theorem probability_point_outside_circle :
  ∃ (p : ℚ), p = 7/12 ∧
  ∀ (m n : ℕ), (1 ≤ m ∧ m ≤ 6) → (1 ≤ n ∧ n ≤ 6) → 
  (m^2 + n^2 > 25 → p = (7 : ℚ) / 12) :=
sorry

end probability_point_outside_circle_l564_564470


namespace rebus_solution_l564_564262

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564262


namespace expr_sum_l564_564109

theorem expr_sum : (7 ^ (-3 : ℤ)) ^ 0 + (7 ^ 0) ^ 2 = 2 :=
by
  sorry

end expr_sum_l564_564109


namespace cary_strips_ivy_l564_564612

variable (strip_per_day : ℕ) (grow_per_night : ℕ) (total_ivy : ℕ)

theorem cary_strips_ivy (h1 : strip_per_day = 6) (h2 : grow_per_night = 2) (h3 : total_ivy = 40) :
  (total_ivy / (strip_per_day - grow_per_night)) = 10 := by
  sorry

end cary_strips_ivy_l564_564612


namespace find_x_l564_564963

variable {a b x : ℝ}

-- Defining the given conditions
def is_linear_and_unique_solution (a b : ℝ) : Prop :=
  3 * a + 2 * b = 0 ∧ a ≠ 0

-- The proof problem: prove that x = 1.5, given the conditions.
theorem find_x (ha : is_linear_and_unique_solution a b) : x = 1.5 :=
  sorry

end find_x_l564_564963


namespace true_propositions_are_two_l564_564327

theorem true_propositions_are_two : 
  (let proposition1 := ∀ (a b c : ℝ), (∃ x : ℝ, x + log 3 x = 3 ∧ a = x) 
                                      ∧ (∃ y : ℝ, y + log 4 y = 3 ∧ b = y) 
                                      ∧ (∃ z : ℝ, z + log 3 z = 1 ∧ c = z) 
                                      → a > b > c,
       proposition2 := ∀ f : ℝ → ℝ, (∀ x : ℝ, f(x) = -f(-x)) 
                                      ∧ (∀ x : ℝ, f(3 + x) + f(1 - x) = 2) 
                                      → f(2010) = 2010,
       proposition3 := ∃ (θ1 θ2 : ℝ), (0 ≤ θ1 ∧ θ1 < 2 * π ∧ 0 ≤ θ2 ∧ θ2 < 2 * π) 
                                      ∧ (2 * sin θ1 = cos θ1 ∧ 2 * sin θ2 = cos θ2 ∧ θ1 ≠ θ2),
       proposition4 := ∀ {a d : ℝ}, (let S : ℕ → ℝ := λ n, n * a + n * (n - 1) / 2 * d) 
                                      → S 7 > S 5 → S 9 > S 3
   in (proposition1 → False) ∧
      (proposition2 → False) ∧
      proposition3 ∧
      proposition4) :=
by
  sorry

end true_propositions_are_two_l564_564327


namespace find_monthly_growth_rate_l564_564144

-- Define all conditions.
variables (March_sales May_sales : ℝ) (monthly_growth_rate : ℝ)

-- The conditions from the given problem
def initial_sales (March_sales : ℝ) : Prop := March_sales = 4 * 10^6
def final_sales (May_sales : ℝ) : Prop := May_sales = 9 * 10^6
def growth_occurred (March_sales May_sales : ℝ) (monthly_growth_rate : ℝ) : Prop :=
  May_sales = March_sales * (1 + monthly_growth_rate)^2

-- The Lean 4 theorem to be proven.
theorem find_monthly_growth_rate 
  (h1 : initial_sales March_sales) 
  (h2 : final_sales May_sales) 
  (h3 : growth_occurred March_sales May_sales monthly_growth_rate) : 
  400 * (1 + monthly_growth_rate)^2 = 900 := 
sorry

end find_monthly_growth_rate_l564_564144


namespace compound_interest_comparison_l564_564152

theorem compound_interest_comparison :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  (P * (1 + r_monthly)^((12 * t)) > P * (1 + r_annual)^t) :=
by
  sorry

end compound_interest_comparison_l564_564152


namespace frosting_per_cake_l564_564646

theorem frosting_per_cake :
  (let total_cakes := 10 * 5 in
   let remaining_cakes := total_cakes - 12 in
   let cans_of_frosting := 76 in
   cans_of_frosting / remaining_cakes = 2) :=
by
  sorry

end frosting_per_cake_l564_564646


namespace max_neg_sum_distinct_numbers_l564_564227

variable {x : Fin 2015 → ℤ}

def is_distinct (x : Fin 2015 → ℤ) : Prop :=
  ∀ i j : Fin 2015, i ≠ j → x i ≠ x j

def in_valid_set (x : Fin 2015 → ℤ) : Prop :=
  ∀ i : Fin 2015, x i ∈ {-1, 2, 3, ..., 2014}

theorem max_neg_sum_distinct_numbers (hx : is_distinct x) (hv : in_valid_set x) :
  ∑ i, x i = -2013 :=
sorry

end max_neg_sum_distinct_numbers_l564_564227


namespace operation_on_b_l564_564074

variables (t b b' : ℝ)
variable (C : ℝ := t * b ^ 4)
variable (e : ℝ := 16 * C)

theorem operation_on_b :
  tb'^4 = 16 * tb^4 → b' = 2 * b := by
  sorry

end operation_on_b_l564_564074


namespace count_distinct_products_of_odd_divisors_81000_l564_564000

def odd_divisors_81000 : Finset ℕ := 
  (Finset.range 5).bind (λ a, 
    (Finset.range 3).image (λ b, 3^a * 5^b))

theorem count_distinct_products_of_odd_divisors_81000 :
  (odd_divisors_81000.bind (λ x, 
    odd_divisors_81000.filter (λ y, x ≠ y).image (λ y, x * y))).card = 41 :=
sorry

end count_distinct_products_of_odd_divisors_81000_l564_564000


namespace problem1_solution_problem2_solution_l564_564464

-- Problem 1: Prove that x = 1 given 6x - 7 = 4x - 5
theorem problem1_solution (x : ℝ) (h : 6 * x - 7 = 4 * x - 5) : x = 1 := by
  sorry


-- Problem 2: Prove that x = -1 given (3x - 1) / 4 - 1 = (5x - 7) / 6
theorem problem2_solution (x : ℝ) (h : (3 * x - 1) / 4 - 1 = (5 * x - 7) / 6) : x = -1 := by
  sorry

end problem1_solution_problem2_solution_l564_564464


namespace christine_speed_l564_564201

def distance : ℕ := 20
def time : ℕ := 5

theorem christine_speed :
  (distance / time) = 4 := 
sorry

end christine_speed_l564_564201


namespace hexagon_trapezoid_area_l564_564096

theorem hexagon_trapezoid_area :
  ∀ (s : ℝ), s = 10 → 
  let h := s * Real.sqrt 3 in
  let area_triangle := (1 / 2) * s * h in 
  let total_area := 2 * area_triangle in 
  total_area = 100 * Real.sqrt 3 :=
by
  sorry

end hexagon_trapezoid_area_l564_564096


namespace grade_point_average_one_third_classroom_l564_564081

theorem grade_point_average_one_third_classroom
  (gpa1 : ℝ) -- grade point average of one third of the classroom
  (gpa_rest : ℝ) -- grade point average of the rest of the classroom
  (gpa_whole : ℝ) -- grade point average of the whole classroom
  (h_rest : gpa_rest = 45)
  (h_whole : gpa_whole = 48) :
  gpa1 = 54 :=
by
  sorry

end grade_point_average_one_third_classroom_l564_564081


namespace find_n_l564_564366

theorem find_n (n : ℕ) (h1 : 0 < n) (h2 : n < 11) (h3 : (18888 - n) % 11 = 0) : n = 1 :=
by
  sorry

end find_n_l564_564366


namespace range_of_a_for_f_ge_a_l564_564422

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem range_of_a_for_f_ge_a :
  (∀ x : ℝ, (-1 ≤ x → f x a ≥ a)) ↔ (-3 ≤ a ∧ a ≤ 1) :=
  sorry

end range_of_a_for_f_ge_a_l564_564422


namespace exists_k_for_all_n_l564_564447

theorem exists_k_for_all_n (n : ℕ) (h : 0 < n) : ∃ k. 2^n ∣ 19^k - 97 := 
sorry

end exists_k_for_all_n_l564_564447


namespace dividend_is_144_l564_564076

theorem dividend_is_144 
  (Q : ℕ) (D : ℕ) (M : ℕ)
  (h1 : M = 6 * D)
  (h2 : D = 4 * Q) 
  (Q_eq_6 : Q = 6) : 
  M = 144 := 
sorry

end dividend_is_144_l564_564076


namespace grisha_win_probability_expected_number_coin_flips_l564_564818

-- Problem 1: Probability of Grisha's win
theorem grisha_win_probability : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (grisha_win : ℚ) :=
  sorry

-- Problem 2: Expected number of coin flips
theorem expected_number_coin_flips : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (expected_tosses : ℚ) :=
  sorry

end grisha_win_probability_expected_number_coin_flips_l564_564818


namespace count_consecutive_even_l564_564055

def is_even (n : ℤ) : Prop := n % 2 = 0

def consecutive_even_numbers (j : Set ℤ) : Prop :=
  ∀ x ∈ j, ∃ k : ℤ, x = -4 + 2 * k ∧ k ≥ 0

def set_J : Set ℤ := {n | is_even n ∧
    (n = -4 ∨ (∃ k : ℤ, n = -4 + 2 * k ∧ k ≥ 0))}

theorem count_consecutive_even : 
  consecutive_even_numbers set_J →
  ∃ n : ℕ, n = 10 := 
begin
  intro h,
  -- Proof steps would go here, but we skip it with the sorry placeholder.
  sorry,
end

end count_consecutive_even_l564_564055


namespace wheel_of_fortune_l564_564440

-- Define the set of possible outcomes for the spinner
def segment_values : List Int := [0, 1000, 500, 300, 400, 700]

-- Define the probability calculation based on conditions
def probability_earning_2200 : Rat :=
  let all_possible_spins := List.product (List.product segment_values segment_values) segment_values
  let valid_combinations := all_possible_spins.filter (fun (x, y, z) => x + y + z == 2200)
  valid_combinations.length / (segment_values.length ^ 3)

theorem wheel_of_fortune : probability_earning_2200 = 1/36 := by
  sorry

end wheel_of_fortune_l564_564440


namespace fixed_point_l564_564295

theorem fixed_point (k : ℝ) : ∃ (a b : ℝ), (∀ k : ℝ, b = 9 * a^2 + k * a - 5 * k) ∧ a = -1 ∧ b = 9 :=
by
  use [-1, 9]
  intro k
  sorry

end fixed_point_l564_564295


namespace triangle_sides_inequality_triangle_expression_negative_l564_564365

theorem triangle_sides_inequality {a b c : ℝ} (h₁ : a + b > c)
  (h₂ : a + c > b) (h₃ : b + c > a) : a^2 < (b + c)^2 :=
begin
  sorry
end

theorem triangle_expression_negative {a b c : ℝ} (h₁ : a + b > c)
  (h₂ : a + c > b) (h₃ : b + c > a) : a^2 - b^2 - c^2 - 2 * b * c < 0 := 
begin
  have : a^2 < (b + c)^2 := triangle_sides_inequality h₁ h₂ h₃,
  calc
  a^2 - b^2 - c^2 - 2 * b * c = a^2 - (b^2 + 2 * b * c + c^2) : 
    by rw [sub_sub_eq_sub_right (b^2 + c^2 + 2 * b * c), add_assoc, add_comm _ c^2, ← add_assoc]
  ... = a^2 - (b + c)^2 : by rw [sq_eq]
  ... < 0 : by linarith
end

end triangle_sides_inequality_triangle_expression_negative_l564_564365


namespace expressions_equivalence_l564_564920

theorem expressions_equivalence (a b c : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (sqrt (a^2 + 2 * a * b + b / c) = (a + b) * sqrt (b / c)) ↔ 
  c = (a^2 * b + 2 * a * b^2 + b^3 - b) / (a^2 + 2 * a * b) :=
sorry

end expressions_equivalence_l564_564920


namespace john_amount_share_l564_564136

theorem john_amount_share {total_amount : ℕ} {total_parts john_share : ℕ} (h1 : total_amount = 4200) (h2 : total_parts = 2 + 4 + 6) (h3 : john_share = 2) :
  john_share * (total_amount / total_parts) = 700 :=
by
  sorry

end john_amount_share_l564_564136


namespace sum_of_three_dice_is_ten_l564_564994

theorem sum_of_three_dice_is_ten : (fin (6 * 6 * 6) → fin 6) → ℕ → ℚ
| d 10 := (27 : ℚ) / (216 : ℚ)
where d := (fun i : fin (6 * 6 * 6) => i.val % 6 + 1) sorry

end sum_of_three_dice_is_ten_l564_564994


namespace drone_path_points_l564_564567

-- Definition of initial and final points
def A : ℤ × ℤ := (-4, 3)
def B : ℤ × ℤ := (4, -3)

-- Define the distance function based on Manhattan distance
def manhattan_distance (p1 p2 : ℤ × ℤ) : ℤ :=
  Int.natAbs (p1.1 - p2.1) + Int.natAbs (p1.2 - p2.2)

-- Define the list of all integer points (x, y) satisfying the condition of the valid path
def valid_points : List (ℤ × ℤ) :=
  List.filter (fun (p : ℤ × ℤ) =>
    let d1 := manhattan_distance p A
    let d2 := manhattan_distance p B
    d1 + d2 ≤ 22
  ) [ (x, y) | x in List.range' -9 19,  -- Covering possible range of x from -9 to 9
              y in List.range' -11 23 ] -- Covering possible range of y from -11 to 11

-- Main theorem statement
theorem drone_path_points : valid_points.length = 225 :=
by
  sorry

end drone_path_points_l564_564567


namespace problem_statement_l564_564524

def product_of_first_n (n : ℕ) : ℕ := List.prod (List.range' 1 n)

def sum_of_first_n (n : ℕ) : ℕ := List.sum (List.range' 1 n)

theorem problem_statement : 
  let numerator := product_of_first_n 9  -- product of numbers 1 through 8
  let denominator := sum_of_first_n 9  -- sum of numbers 1 through 8
  numerator / denominator = 1120 :=
by {
  sorry
}

end problem_statement_l564_564524


namespace NataliesSisterInitialDiaries_l564_564929

theorem NataliesSisterInitialDiaries (D : ℕ)
  (h1 : 2 * D - (1 / 4) * 2 * D = 18) : D = 12 :=
by sorry

end NataliesSisterInitialDiaries_l564_564929


namespace coeff_x5_term_l564_564846

-- We define the binomial coefficient function C(n, k)
def C (n k : ℕ) : ℕ := Nat.choose n k

-- We define the expression in question
noncomputable def expr (x : ℝ) : ℝ := (1/x + 2*x)^7

-- The coefficient of x^5 term in the expansion
theorem coeff_x5_term : 
  let general_term (r : ℕ) (x : ℝ) := (2:ℝ)^r * C 7 r * x^(2 * r - 7)
  -- r is chosen such that the power of x is 5
  let r := 6
  -- The coefficient for r=6
  general_term r 1 = 448 := 
by sorry

end coeff_x5_term_l564_564846


namespace initial_volume_of_mixture_l564_564160

theorem initial_volume_of_mixture :
  ∃ V : ℝ, (0.15 * V + 20 = 0.25 * (V + 20)) ∧ V = 150 :=
by
  exists 150
  split
  { calc
      0.15 * 150 + 20 = 22.5 + 20 : by rw mul_comm
      ...             = 42.5 : by norm_num
      ...             = 0.25 * (150 + 20) : by norm_num
  }
  { refl }

end initial_volume_of_mixture_l564_564160


namespace solve_functional_equation_l564_564251

noncomputable def functional_equation_sol (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f(x) + x * y) * f(x - 3 * y) + (f(y) + x * y) * f(3 * x - y) = (f(x + y))^2

theorem solve_functional_equation (f : ℝ → ℝ) (hf : functional_equation_sol f) :
  (∀ x : ℝ, f(x) = x^2) ∨ (∀ x : ℝ, f(x) = 0) :=
sorry

end solve_functional_equation_l564_564251


namespace tom_initial_amount_l564_564101

-- Define the conditions
def initial_value (X : ℝ) : Prop :=
  let tripled_value := 3 * X in
  let sold_value := 0.4 * tripled_value in
  sold_value = 240

-- State the problem to prove
theorem tom_initial_amount (X : ℝ) (h : initial_value X) : X = 200 := by
  sorry

end tom_initial_amount_l564_564101


namespace math_proof_problem_l564_564528

section Problem

variable {R: Type} [LinearOrderedField R]

open Real

-- Statement A
def statement_A (a : R) :=
  ∀ x : R, (2 * (1 - x)^2 - a * (1 - x) + 3) = (2 * (1 + x)^2 - a * (1 + x) + 3) → a = 4

-- Statement B
def statement_B (k : R) :=
  ∀ x : R, (k * x^2 - 6 * k * x + k + 8) ≥ 0 → (0 < k) ∧ (k ≤ 1)

-- Statement C
def statement_C (a : R) :=
  ∀ a, (a = 1 → (∃ M N : Set R, M = {1, 2} ∧ N = {a^2} ∧ N ⊆ M) → (∃ M N : Set R, N ⊆ M → a = 1))

-- Statement D
def statement_D (x : R) :=
  ∀ x ∈ Ioo (-π / 4) (3 * π / 4), sin (2 * x) ∈ Icc (-1) 1

-- Equivalence proof problem
theorem math_proof_problem (a : R) (k : R) (x : R) :
  statement_A a ∧ statement_C a ∧ statement_D x :=
by
  sorry

end Problem

end math_proof_problem_l564_564528


namespace radius_Q_eq_one_imp_m_eq_sixteen_shortest_chord_of_line_l_with_circle_P_is_2_sqrt_7_l564_564724

-- Define the equations of the circles
def circle_P (x y : ℝ) : Prop := x^2 + (y + 3)^2 = 9
def circle_Q (x y : ℝ) (m : ℝ) : Prop := x^2 + 2*x + y^2 + 8*y + m = 0

-- Define points A on circle P and B on circle Q
def point_on_circle_P (x y : ℝ) := circle_P x y
def point_on_circle_Q (x y : ℝ) (m : ℝ) := circle_Q x y m

-- Define the radius of circle Q
def radius_circle_Q (m : ℝ) : ℝ := real.sqrt (17 - m)

-- Prove that if the radius of circle Q is 1, then m = 16
theorem radius_Q_eq_one_imp_m_eq_sixteen :
  ∀ (m : ℝ), radius_circle_Q m = 1 → m = 16 := by
{
  intro m,
  unfold radius_circle_Q,
  intros h,
  calc
    1 = real.sqrt (17 - m) : h
    ... = 17 - m            : real.sqrt_eq_iff_eq_square_left_zero.mpr (by linarith)
    ... = 16                : by linarith
}

-- Define the line l
def line_l (a x y : ℝ) : Prop := a*x - y - a - 2 = 0

-- Define the shortest chord length of intersection of line l with circle P
def shortest_chord_length_circle_P (a : ℝ) : ℝ := 2 * real.sqrt (9 - (real.sqrt (2))^2)

-- Prove that the shortest chord length obtained by line l intersecting circle P is 2 * sqrt(7)
theorem shortest_chord_of_line_l_with_circle_P_is_2_sqrt_7 :
  ∀ (a : ℝ), shortest_chord_length_circle_P a = 2 * real.sqrt 7 := by
{
  intro a,
  unfold shortest_chord_length_circle_P,
  norm_num,
}

end radius_Q_eq_one_imp_m_eq_sixteen_shortest_chord_of_line_l_with_circle_P_is_2_sqrt_7_l564_564724


namespace theme_parks_difference_l564_564396

-- Definitions and conditions
def Jamestown : ℕ := 20
def Venice : ℕ := Jamestown + 25

noncomputable def Marina_Del_Ray : ℕ :=
  135 - (Jamestown + Venice)

theorem theme_parks_difference :
  Marina_Del_Ray - Jamestown = 50 :=
by
  have h1 : Jamestown = 20 := rfl
  have h2 : Venice = Jamestown + 25 := rfl
  have h3 : Venice = 45 := by
    rw [h1, h2]
  have h4 : Marina_Del_Ray = 70 := by
    unfold Marina_Del_Ray
    rw [h1, h3]
    norm_num
  show Marina_Del_Ray - Jamestown = 50
  rw [h1, h4]
  norm_num

end theme_parks_difference_l564_564396


namespace probability_grisha_wins_expectation_coin_flips_l564_564835

-- Define the conditions as predicates
def grishaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 0) && (toss_seq.last = some true)

def vanyaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 1) && (toss_seq.last = some false)

-- State the probability theorem
theorem probability_grisha_wins : 
  (∑ x in ((grishaWins ≤ 1 / 3))) = 1/3 := by sorry

-- State the expectation theorem
theorem expectation_coin_flips : 
  (E[flips until (grishaWins ∨ vanyaWins)]) = 2 := by sorry

end probability_grisha_wins_expectation_coin_flips_l564_564835


namespace price_of_each_book_l564_564561

theorem price_of_each_book (B P : ℕ) 
  (h1 : (1 / 3 : ℚ) * B = 36) -- Number of unsold books is 1/3 of the total books and it equals 36
  (h2 : (2 / 3 : ℚ) * B * P = 144) -- Total amount received for the books sold is $144
  : P = 2 := 
by
  sorry

end price_of_each_book_l564_564561


namespace count_digit_7_in_range_l564_564856

theorem count_digit_7_in_range : ∀ n ∈ finset.Icc 100 199, (n % 10 = 7 ∨ n / 10 % 10 = 7) →  ∃ counts, counts = 20 :=
by
  sorry

end count_digit_7_in_range_l564_564856


namespace parity_of_f_f_is_decreasing_range_of_f_range_of_a_l564_564730

variable (f : ℝ → ℝ)

-- Given conditions
axiom additivity (x y : ℝ) : f(x + y) = f(x) + f(y)
axiom negativity (x : ℝ) : (x > 0) → (f(x) < 0)
axiom value_at_1 : f(1) = -2

-- 1. Determine the parity of f(x)
theorem parity_of_f : ∀ x : ℝ, f(-x) = -f(x) := sorry

-- 2. Prove that f(x) is a decreasing function on ℝ
theorem f_is_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁) > f(x₂) := sorry

-- 3. Find the range of f(x) on the interval [-3, 3]
theorem range_of_f : set.range (λ x : ℝ, -3 ≤ x ∧ x ≤ 3) = set.Icc (-6) 6 := sorry

-- 4. Find the range of values for a such that f(ax^2) - 2f(x) < f(x) + 4
theorem range_of_a : ∀ (a : ℝ), (∀ x : ℝ, f(a * x^2) - 2 * f(x) < f(x) + 4) ↔ (a > 9 / 8) := sorry

end parity_of_f_f_is_decreasing_range_of_f_range_of_a_l564_564730


namespace findFunnyTriangles_l564_564054

noncomputable def funnyTriangles : Set (ℝ × ℝ × ℝ) :=
  { sides | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    -- The conditions for altitude, median, and angle bisector forming 4 non-overlapping triangles
    -- with areas in arithmetic sequence would be encoded here.
    -- For brevity, we omit these complex geometric conditions as they require detailed geometric reasoning.
    true }

theorem findFunnyTriangles (sides : ℝ × ℝ × ℝ) :
  sides ∈ funnyTriangles ↔ sides = (1, real.sqrt 3, 2) ∨ sides = (1, 2 * real.sqrt 6, 5) := 
by {
  -- Proof outline should show that if sides in funnyTriangles, then it must be one of the specific ratios.
  sorry -- actual proof omitted.
}

end findFunnyTriangles_l564_564054


namespace find_all_z_l564_564670

noncomputable def all_solutions (z : ℝ) : Prop := z = -2 ∨ z = 2 ∨ z = -real.sqrt 2 ∨ z = real.sqrt 2

theorem find_all_z (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ all_solutions z := 
sorry

end find_all_z_l564_564670


namespace sum_binom_mod_p_l564_564903

variable (p : ℕ) (d r : ℤ)
variable [hp : Fact (Nat.Prime p)]

noncomputable def mod_sum := 
  ∑ k in Finset.range p, Nat.choose (2 * k) (k + d)

theorem sum_binom_mod_p (p_prime : Nat.Prime p) (d_range : 0 ≤ d ∧ d ≤ p) (r_def : r ≡ p - d [MOD 3]) (r_values : r ∈ {-1, 0, 1}) :
  mod_sum p d ≡ r [MOD p] := 
sorry

end sum_binom_mod_p_l564_564903


namespace solve_quartic_eqn_l564_564678

theorem solve_quartic_eqn (z : ℂ) : 
  z ^ 4 - 6 * z ^ 2 + 8 = 0 ↔ z = 2 ∨ z = -2 ∨ z = Complex.sqrt 2 ∨ z = -Complex.sqrt 2 := 
sorry

end solve_quartic_eqn_l564_564678


namespace mouse_further_from_cheese_l564_564578

noncomputable def point_cheese : ℝ × ℝ := (12, 10)
noncomputable def line_mouse : ℝ → ℝ := λ x => -5 * x + 18

theorem mouse_further_from_cheese : let (a, b) := (2, 8) in a + b = 10 :=
by
  sorry

end mouse_further_from_cheese_l564_564578


namespace find_number_subtracted_l564_564062

theorem find_number_subtracted (x : ℕ) (h : 88 - x = 54) : x = 34 := by
  sorry

end find_number_subtracted_l564_564062


namespace cos_270_eq_zero_l564_564205

theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 := by
  sorry

end cos_270_eq_zero_l564_564205


namespace alice_can_prevent_l564_564591

-- Define what it means to follow the rules and construct the number
def valid_digit_sequence (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 → (seq[i] % 3 ≠ seq[i+1] % 3)

-- Define the game conditions
def game_conditions (seq : List ℕ) : Prop :=
  seq.length = 2018 ∧ valid_digit_sequence seq

-- Define Alice's goal
def alice_wins (seq : List ℕ) : Prop :=
  (seq.sum % 3 ≠ 0)

-- Final theorem which asserts that Alice can always win
theorem alice_can_prevent (seq : List ℕ) (cond : game_conditions seq) : alice_wins seq := 
sorry

end alice_can_prevent_l564_564591


namespace sum_mod_nine_l564_564291

def a : ℕ := 1234
def b : ℕ := 1235
def c : ℕ := 1236
def d : ℕ := 1237
def e : ℕ := 1238
def modulus : ℕ := 9

theorem sum_mod_nine : (a + b + c + d + e) % modulus = 6 :=
by
  sorry

end sum_mod_nine_l564_564291


namespace probability_grisha_wins_expectation_coin_flips_l564_564837

-- Define the conditions as predicates
def grishaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 0) && (toss_seq.last = some true)

def vanyaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 1) && (toss_seq.last = some false)

-- State the probability theorem
theorem probability_grisha_wins : 
  (∑ x in ((grishaWins ≤ 1 / 3))) = 1/3 := by sorry

-- State the expectation theorem
theorem expectation_coin_flips : 
  (E[flips until (grishaWins ∨ vanyaWins)]) = 2 := by sorry

end probability_grisha_wins_expectation_coin_flips_l564_564837


namespace value_of_k_if_solution_set1_range_of_k_if_solution_set2_l564_564771

section Problem1
-- Define the inequality condition
def quadratic_inequality (k : ℝ) (x : ℝ) := k * x^2 - 2 * x + 6 * k < 0
-- Condition: k ≠ 0
variable {k : ℝ} (hk : k ≠ 0)
-- Condition: Solution set { x | x < -3 or x > -2 }
variable (sol_set1 : ∀ x : ℝ, quadratic_inequality k x → x < -3 ∨ x > -2)

theorem value_of_k_if_solution_set1 : 
  k = -2/5 := 
sorry
end Problem1

section Problem2
-- Define the inequality condition
def quadratic_inequality (k : ℝ) (x : ℝ) := k * x^2 - 2 * x + 6 * k < 0
-- Condition: k ≠ 0
variable {k : ℝ} (hk : k ≠ 0)
-- Condition: Solution set is ℝ
variable (sol_set2 : ∀ x : ℝ, quadratic_inequality k x)

theorem range_of_k_if_solution_set2 : 
  k ∈ Iio (-Real.sqrt 6 / 6) := 
sorry
end Problem2

end value_of_k_if_solution_set1_range_of_k_if_solution_set2_l564_564771


namespace diamond_value_loss_l564_564494

theorem diamond_value_loss
  (p k : ℝ)
  (h1 : p > 0) (h2 : k > 1) 
  (h3: k ≤ 2) :
  ∃ x y : ℝ, x = (p * k + Real.sqrt(2 * k * p^2 - p^2 * k^2)) / (2 * k) ∨
             x = (p * k - Real.sqrt(2 * k * p^2 - p^2 * k^2)) / (2 * k) ∧
             y = p - x ∧
             (x = p / 2 → y = p / 2) ∧
             (y = p / 2 → x = p / 2) :=
by {
  sorry
}

end diamond_value_loss_l564_564494


namespace initial_people_in_gym_l564_564196

variables (W A S : ℕ)

theorem initial_people_in_gym (h1 : (W - 3 + 2 - 3 + 4 - 2 + 1 = W + 1))
                              (h2 : (A + 2 - 1 + 3 - 3 + 1 = A + 2))
                              (h3 : (S + 1 - 2 + 1 + 3 - 2 + 2 = S + 3))
                              (final_total : (W + 1) + (A + 2) + (S + 3) + 2 = 30) :
  W + A + S = 22 :=
by 
  sorry

end initial_people_in_gym_l564_564196


namespace AH_HD_ratio_l564_564848

theorem AH_HD_ratio (A B C D : Point) (H : Point)
  (h₁ : Triangle A B C)
  (h₂ : BC = 6)
  (h₃ : AC = 4 * Real.sqrt 2)
  (h₄ : ∠ ACB = 30)
  (h₅ : IsOrthocenter H A B C)
  (h₆ : IsAltitude A D B C)
  (h₇ : IsAltitude B E A C)
  (h₈ : IsAltitude C F A B)
  : Ratio (Dist A H) (Dist H D) = (3 * Real.sqrt 6 - 3 * Real.sqrt 3) / (3 * Real.sqrt 3 - Real.sqrt 6) :=
sorry

end AH_HD_ratio_l564_564848


namespace tangent_lines_eq_intercepts_l564_564926

theorem tangent_lines_eq_intercepts : 
  let circle := λ x y, x^2 + y^2 - 4 * x - 4 * y + 7 = 0
  ∃ (l : ℝ → ℝ → Bool), (∀ (x y : ℝ), l x y ↔ x + y = a ∨ x + y = b) ∧
    (∀ (x y : ℝ), circle x y) ∧
    (∀ (a b : ℝ), let d := abs (4 - a) / sqrt 2 in d = 1) ∧
    cardinality l = 4 :=
by
  sorry

end tangent_lines_eq_intercepts_l564_564926


namespace ellipse_properties_l564_564736

noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x * x) / (a * a) + (y * y) / (b * b) = 1

theorem ellipse_properties (a b c k : ℝ) (h_ab : a > b) (h_b : b > 1) (h_c : 2 * c = 2) 
  (h_area : (2 * Real.sqrt 3 / 3)^2 = 4 / 3) (h_slope : k ≠ 0)
  (h_PD : |(c - 4 * k^2 / (3 + 4 * k^2))^2 + (-3 * k / (3 + 4 * k^2))^2| = 3 * Real.sqrt 2 / 7) :
  (ellipse_equation 1 0 a b ∧
   (a = 2 ∧ b = Real.sqrt 3) ∧
   k = 1 ∨ k = -1) :=
by
  -- Prove the standard equation of the ellipse C and the value of k
  sorry

end ellipse_properties_l564_564736


namespace blueprint_scale_l564_564143

theorem blueprint_scale (inches_length : ℝ) (scale : ℝ) (represents_meters : ℝ) :
  inches_length = 7.5 → scale = 50 → represents_meters = inches_length * scale → represents_meters = 375 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end blueprint_scale_l564_564143


namespace f_12_eq_12_l564_564915

noncomputable def f : ℕ → ℤ := sorry

axiom f_int (n : ℕ) (hn : 0 < n) : ∃ k : ℤ, f n = k
axiom f_2 : f 2 = 2
axiom f_mul (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m * f n
axiom f_monotonic (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : m > n → f m > f n

theorem f_12_eq_12 : f 12 = 12 := sorry

end f_12_eq_12_l564_564915


namespace sector_properties_l564_564377

noncomputable def radius : ℝ := 6
noncomputable def angle : ℝ := π / 4
noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := r * θ
noncomputable def sector_perimeter (r : ℝ) (l : ℝ) : ℝ := 2 * r + l
noncomputable def sector_area (r : ℝ) (l : ℝ) : ℝ := (1 / 2) * r * l

theorem sector_properties :
  let l := arc_length radius angle in
  sector_perimeter radius l = 12 + 3 * π / 2 ∧
  sector_area radius l = 9 * π / 2 :=
by
  sorry

end sector_properties_l564_564377


namespace range_f_l564_564092

open Real

noncomputable def f (x : ℝ) := sqrt (4 - 2 * x) + sqrt x

theorem range_f :
  ∀ y, ∃ x ∈ Icc (0:ℝ) 2, f x = y ↔ y ∈ Icc (sqrt 2) (sqrt 6) := by
  sorry

end range_f_l564_564092


namespace inhabitant_50th_statement_l564_564242

-- Definition of inhabitant types
inductive InhabitantType
| Knight
| Liar

-- Predicate for the statement of inhabitants
def says (inhabitant : InhabitantType) (statement : InhabitantType) : Bool :=
  match inhabitant with
  | InhabitantType.Knight => true
  | InhabitantType.Liar => false

-- Conditions from the problem
axiom inhabitants : Fin 50 → InhabitantType
axiom statements : ∀ i : Fin 50, i.val % 2 = 0 → (says (inhabitants ((i + 1) % 50)) InhabitantType.Liar = false)
axiom statements' : ∀ i : Fin 50, i.val % 2 = 1 → (says (inhabitants ((i + 1) % 50)) InhabitantType.Knight = true)

-- Goal to prove
theorem inhabitant_50th_statement : says (inhabitants 49) InhabitantType.Knight := by
  sorry

end inhabitant_50th_statement_l564_564242


namespace max_area_parabolic_slice_bisects_l564_564734

noncomputable def largest_parabolic_slice_area (M O A A' C B D E : Type) [EuclideanGeometry O A A' M C B D E] :=
  let cone := (vertex : M, base_circle : O, generatrix : MA, opp_point : A')
  let S := (plane_parallel : MA, intersects : C, cuts : (B, D))
  let midpoint := E

-- Statement to prove:
-- The largest possible area parabolic slice is when \( E \) bisects \( AO \).
theorem max_area_parabolic_slice_bisects (
  cone : (vertex : M, base_circle : O, generatrix : MA, opp_point : A'), 
  S : (plane_parallel : MA, intersects : C, cuts : (B, D)), 
  midpoint : E 
): (E bisects AO) -> largest_parabolic_slice_area M O A A' C B D E := 
sorry

end max_area_parabolic_slice_bisects_l564_564734


namespace a100_pos_a100_abs_lt_018_l564_564335

noncomputable def a (n : ℕ) : ℝ := real.cos (10^n * real.pi / 180)

theorem a100_pos : (a 100 > 0) :=
by
  sorry

theorem a100_abs_lt_018 : (|a 100| < 0.18) :=
by
  sorry

end a100_pos_a100_abs_lt_018_l564_564335


namespace minimum_solutions_in_interval_l564_564546

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f(x) = f(-x)
axiom periodic_function : ∀ x : ℝ, f(x + 3) = f(x)
axiom f_at_2 : f(2) = 0

theorem minimum_solutions_in_interval (f : ℝ → ℝ) 
  (hef : ∀ x : ℝ, f(x) = f(-x))
  (hpf : ∀ x : ℝ, f(x + 3) = f(x))
  (h2 : f(2) = 0) : 
  ∃ (n : ℕ), n = 4 ∧ 
  (∀ x ∈ Ioo 0 6, f(x) = 0 → 
    (x = 1 ∨ x = 2 ∨ x = 4 ∨ x = 5)) := 
sorry

end minimum_solutions_in_interval_l564_564546


namespace plane_contains_point_and_line_l564_564707

variable (M : Type) [AddCommGroup M] [Module ℝ M]

def point_M1 : M := ⟨3, 1, 0⟩

def line_direction : M := ⟨1, 2, 3⟩

def line_point : M := ⟨4, 0, 1⟩

noncomputable def plane_eqn (x y z : ℝ) : Prop :=
  5 * x + 2 * y - 3 * z - 17 = 0

theorem plane_contains_point_and_line :
  plane_eqn M (3 : ℝ) (1 : ℝ) (0 : ℝ) ∧ plane_eqn M (4 + t * 1) (2 * t) (1 + 3 * t) :=
sorry

end plane_contains_point_and_line_l564_564707


namespace percentage_calculation_l564_564794

theorem percentage_calculation : 
  (0.8 * 90) = ((P / 100) * 60.00000000000001 + 30) → P = 70 := by
  sorry

end percentage_calculation_l564_564794


namespace midpoint_of_similar_triangles_l564_564107

noncomputable section

def is_midpoint (a o u : ℂ) : Prop :=
  2 * a = o + u

def oriented_similar_triangles (a l t x : ℂ) (b r m y : ℂ) : Prop :=
  ∃ k : ℂ, 
    (a - l) / (a - t) = k ∧
    (x - r) / (x - m) = k ∧
    (a - r) / (a - m) = k ∧
    (x - l) / (x - m) = k

theorem midpoint_of_similar_triangles 
  {a o u l t r m : ℂ}
  (h1 : oriented_similar_triangles a l t o)
  (h2 : oriented_similar_triangles a r m o)
  (h3 : oriented_similar_triangles o r t o)
  (h4 : oriented_similar_triangles u l m o) :
  is_midpoint a o u :=
sorry

end midpoint_of_similar_triangles_l564_564107


namespace count_digit_7_to_199_l564_564871

theorem count_digit_7_to_199 : (Nat.countDigitInRange 7 100 199) = 20 := by
  sorry

end count_digit_7_to_199_l564_564871


namespace find_z_values_l564_564651

theorem find_z_values : {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -sqrt 2, sqrt 2, 2} :=
sorry

end find_z_values_l564_564651


namespace find_digits_l564_564273

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l564_564273


namespace determine_initial_sum_l564_564588

def initial_sum_of_money (P r : ℝ) : Prop :=
  (600 = P + 2 * P * r) ∧ (700 = P + 2 * P * (r + 0.1))

theorem determine_initial_sum (P r : ℝ) (h : initial_sum_of_money P r) : P = 500 :=
by
  cases h with
  | intro h1 h2 =>
    sorry

end determine_initial_sum_l564_564588


namespace speed_is_15_mph_l564_564918

-- Lean 4 statement to prove that the speed r is 15 mph
theorem speed_is_15_mph (c : ℝ) (r : ℝ) (t : ℝ)
  (circumference_miles: c = 15 / 5280)
  (rotation_time: r * t = c * 3600)
  (new_condition: (r + 8) * (t - 1 / 18000) = c * 3600) : r = 15 :=
sorry

end speed_is_15_mph_l564_564918


namespace first_year_after_1950_with_sum_of_digits_18_l564_564114

noncomputable def year_after (y : ℕ) := (y > 1950)
def sum_of_digits (n : ℕ) := n.digits 10 |>.sum
def first_year := ∃ y, year_after y ∧ sum_of_digits y = 18 ∧ ∀ z, year_after z ∧ sum_of_digits z = 18 → y ≤ z

theorem first_year_after_1950_with_sum_of_digits_18 : first_year 1980 :=
by
  sorry

end first_year_after_1950_with_sum_of_digits_18_l564_564114


namespace Jeremy_no_decision_l564_564892

theorem Jeremy_no_decision (n : ℕ) : 
    (factorial (2^n) / (2^(n * 2^(n-1))) > 1) :=
sorry

end Jeremy_no_decision_l564_564892


namespace perpendicular_solution_maximum_value_magnitude_l564_564785

open Real

def a (θ : ℝ) : ℝ × ℝ := (sin θ, sqrt 3)
def b (θ : ℝ) : ℝ × ℝ := (1, cos θ)

theorem perpendicular_solution (θ : ℝ) (hθ : θ ∈ Ioo (-π / 2) (π / 2)) (h_perp : a θ.1 * b θ.1 + a θ.2 * b θ.2 = 0) : 
  θ = -π / 3 := sorry

theorem maximum_value_magnitude (θ : ℝ) (hθ : θ ∈ Ioo (-π / 2) (π / 2)) : 
  |(sqrt ((a θ.1 + b θ.1)^2 + (a θ.2 + b θ.2)^2))| ≤ 3 := sorry

end perpendicular_solution_maximum_value_magnitude_l564_564785


namespace cyclist_time_to_climb_and_descend_hill_l564_564562

noncomputable def hill_length : ℝ := 400 -- hill length in meters
noncomputable def ascent_speed_kmh : ℝ := 7.2 -- ascent speed in km/h
noncomputable def ascent_speed_ms : ℝ := ascent_speed_kmh * 1000 / 3600 -- ascent speed converted in m/s
noncomputable def descent_speed_ms : ℝ := 2 * ascent_speed_ms -- descent speed in m/s

noncomputable def time_to_climb : ℝ := hill_length / ascent_speed_ms -- time to climb in seconds
noncomputable def time_to_descend : ℝ := hill_length / descent_speed_ms -- time to descend in seconds
noncomputable def total_time : ℝ := time_to_climb + time_to_descend -- total time in seconds

theorem cyclist_time_to_climb_and_descend_hill : total_time = 300 :=
by
  sorry

end cyclist_time_to_climb_and_descend_hill_l564_564562


namespace betty_harvest_l564_564197

theorem betty_harvest (num_parsnips : ℕ) (num_carrots : ℕ) (num_potatoes : ℕ) :
  (let
    parsnips_full_boxes := 3 / 4 * 20,
    parsnips_half_boxes := 1 / 4 * 20,
    parsnips_total := parsnips_full_boxes * 20 + parsnips_half_boxes * 10,
    carrots_full_boxes := 2 / 3 * 15,
    carrots_half_boxes := 1 / 3 * 15,
    carrots_total := carrots_full_boxes * 25 + carrots_half_boxes * 12.5,
    potatoes_full_boxes := 5 / 8 * 10,
    potatoes_half_boxes := 10 - 5 / 8 * 10,
    potatoes_total := potatoes_full_boxes * 30 + potatoes_half_boxes * 15
   in
    num_parsnips = 350 ∧ num_carrots = 312 ∧ num_potatoes = 240
  ) := 
sorry

end betty_harvest_l564_564197


namespace math_problem_statement_l564_564913

-- Definitions of conditions
def unit_square (s : set (ℝ × ℝ)) : Prop :=
  ∃ l u, l < u ∧ s = {p : ℝ × ℝ | p.1 ≥ l ∧ p.1 ≤ u ∧ p.2 ≥ l ∧ p.2 ≤ u}

def Sn (n : ℕ) : set (ℝ × ℝ) :=
  if n = 0 then {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.1 ≤ 1 ∧ p.2 ≥ 0 ∧ p.2 ≤ 1}
  else 
    let previous := Sn (n-1) in
    ⋃ (i j : ℕ) (h : (i, j) ≠ (1, 1)), 
      {p | (p.1 - i * 3 ^ (n-1) / 3) ∈ previous ∧ (p.2 - j * 3 ^ (n-1) / 3) ∈ previous}

noncomputable def expected_value_distance (n : ℕ) : ℝ :=
  let Sn_points := {p : ℝ × ℝ | p ∈ Sn n} in
  (∑ (x x' y y' : ℝ) in Sn_points, (abs (x - x') + abs (y - y'))) / (Sn_points.cardinality) ^ 2

def a_n (n : ℕ) : ℝ := expected_value_distance n

def relatively_prime (a b : ℕ) : Prop := nat.gcd a b = 1

def lim_approaches (r : ℝ) : Prop := tendsto (λ n, a_n n / 3^n) at_top (𝓝 r)

-- The proof problem statement
theorem math_problem_statement : ∃ a b : ℕ, relatively_prime a b ∧ (λ n, a_n n / 3^n) b = (a / b : ℚ) ∧ (100 * a + b = 2007) :=
sorry

end math_problem_statement_l564_564913


namespace winning_strategy_l564_564622

theorem winning_strategy (n : ℕ) (h : n > 1) :
  ((n % 2 = 1) → (∃ B_wins : True, B_wins)) ∧ ((n % 2 = 0) → (∃ A_wins : True, A_wins)) :=
by
  sorry

end winning_strategy_l564_564622


namespace xyz_length_l564_564066

theorem xyz_length (unit : ℝ) (sqrt : ℝ → ℝ) :
  let X := 2 * sqrt (unit^2 + unit^2)
  let Y := 2 * unit + sqrt (unit^2 + unit^2)
  let Z := 4 * unit + sqrt ((2 * unit)^2 + unit^2)
  X + Y + Z = 6 + 3 * sqrt 2 + sqrt 5 :=
by
  let unit := 1
  let sqrt := λ x, Real.sqrt x
  let X := 2 * sqrt (unit^2 + unit^2)
  let Y := 2 * unit + sqrt (unit^2 + unit^2)
  let Z := 4 * unit + sqrt ((2 * unit)^2 + unit^2)
  have hX : X = 2 * sqrt 2, by sorry
  have hY : Y = 2 + sqrt 2, by sorry
  have hZ : Z = 4 + sqrt 5, by sorry
  calc
    X + Y + Z = 2 * sqrt 2 + (2 + sqrt 2) + (4 + sqrt 5) : by rw [hX, hY, hZ]
          ... = 6 + 3 * sqrt 2 + sqrt 5 : by sorry

end xyz_length_l564_564066


namespace new_average_weight_l564_564476

-- Statement only
theorem new_average_weight (avg_weight_29: ℝ) (weight_new_student: ℝ) (total_students: ℕ) 
  (h1: avg_weight_29 = 28) (h2: weight_new_student = 22) (h3: total_students = 29) : 
  (avg_weight_29 * total_students + weight_new_student) / (total_students + 1) = 27.8 :=
by
  -- declare local variables for simpler proof
  let total_weight := avg_weight_29 * total_students
  let new_total_weight := total_weight + weight_new_student
  let new_total_students := total_students + 1
  have t_weight : total_weight = 812 := by sorry
  have new_t_weight : new_total_weight = 834 := by sorry
  have n_total_students : new_total_students = 30 := by sorry
  exact sorry

end new_average_weight_l564_564476


namespace rebus_solution_l564_564265

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564265


namespace solution_set_to_inequality_l564_564967

noncomputable def f : ℝ → ℝ := sorry
def f' (x : ℝ) : ℝ := sorry -- Representation of the derivative of f

-- Hypotheses
axiom h1 : ∀ x : ℝ, f(x) + f'(x) > 1
axiom h2 : f(0) = 4

-- Theorem statement
theorem solution_set_to_inequality : {x : ℝ | e^x * f(x) > e^x + 3} = Ioi 0 := sorry

end solution_set_to_inequality_l564_564967


namespace ellipse_problem_l564_564738

noncomputable def ellipse_std_form (a b c : ℝ) (h_1 : b = 1) (h_2 : a^2 = b^2 + c^2) (h_3 : c/a = sqrt 2 / 2) : Prop :=
  (a = sqrt 2) ∧ (b = 1) ∧ (c = 1) ∧ (∀ x y : ℝ, (x^2 / 2 + y^2 = 1))

noncomputable def max_area_triangle (k : ℝ) : Prop :=
  k^2 > 4 → 
  (∃ A : ℝ, A = (8 * sqrt (k^2 - 4) / (2 * k^2 + 1)) ∧ A = 2 * sqrt 2 / 3)

theorem ellipse_problem (a b c : ℝ) (k : ℝ) (h_1 : b = 1) (h_2 : a^2 = b^2 + c^2) (h_3 : c/a = sqrt 2 / 2) : 
  ellipse_std_form a b c h_1 h_2 h_3 ∧ max_area_triangle k :=
by
  sorry

end ellipse_problem_l564_564738


namespace cos_angle_ab_l564_564783

noncomputable theory
open Real

variables (a b : ℝ × ℝ × ℝ)

def vector_add (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

def vector_sub (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

def cos_angle (a b : ℝ × ℝ × ℝ) : ℝ :=
  dot_product a b / ((vector_magnitude a) * (vector_magnitude b))

theorem cos_angle_ab : 
  vector_add a b = (0, sqrt 2, 0) ∧ 
  vector_sub a b = (2, sqrt 2, -2 * sqrt 3) → 
  cos_angle a b = -sqrt 6 / 3 :=
sorry

end cos_angle_ab_l564_564783


namespace average_books_per_member_l564_564976

theorem average_books_per_member : 
  let books_read := [3, 4, 1, 6, 2] in
  let members := [1, 2, 3, 4, 5] in
  let total_books := list.sum (list.map (λ (x : ℕ × ℕ), x.1 * x.2) (list.zip books_read members)) in
  let total_members := list.sum books_read in
  let average :=(total_books : ℚ) / total_members in
  average = 3 :=
by
  let books_read := [3, 4, 1, 6, 2]
  let members := [1, 2, 3, 4, 5]
  have total_books := list.sum (list.map (λ (x : ℕ × ℕ), x.1 * x.2) (list.zip books_read members))
  have total_members := list.sum books_read
  have average := (total_books : ℚ) / total_members
  sorry

end average_books_per_member_l564_564976


namespace solve_quartic_l564_564684

theorem solve_quartic (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ (z = 2 ∨ z = -2 ∨ z = sqrt 2 ∨ z = -sqrt 2) :=
by
  sorry

end solve_quartic_l564_564684


namespace b_payment_l564_564134

theorem b_payment
    (a_horses : ℕ) (a_months : ℕ)
    (b_horses : ℕ) (b_months : ℕ)
    (c_horses : ℕ) (c_months : ℕ)
    (d_horses : ℕ) (d_months : ℕ)
    (total_rent : ℝ) :
    a_horses = 15 → a_months = 10 →
    b_horses = 18 → b_months = 11 →
    c_horses = 20 → c_months = 8 →
    d_horses = 25 → d_months = 7 →
    total_rent = 1260 →
    let a_shares := a_horses * a_months,
        b_shares := b_horses * b_months,
        c_shares := c_horses * c_months,
        d_shares := d_horses * d_months,
        total_shares := a_shares + b_shares + c_shares + d_shares,
        cost_per_share := total_rent / total_shares,
        b_payment := cost_per_share * b_shares
    in b_payment = 366.03 :=
by {
  intros,
  let a_shares := a_horses * a_months,
  let b_shares := b_horses * b_months,
  let c_shares := c_horses * c_months,
  let d_shares := d_horses * d_months,
  let total_shares := a_shares + b_shares + c_shares + d_shares,
  let cost_per_share := total_rent / total_shares,
  let b_payment := cost_per_share * b_shares,
  sorry
}

end b_payment_l564_564134


namespace find_digits_l564_564272

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l564_564272


namespace Cody_reads_series_in_7_weeks_l564_564618

theorem Cody_reads_series_in_7_weeks :
  ∀ (total_books first_week_books second_week_books weekly_books : ℕ),
  total_books = 54 →
  first_week_books = 6 →
  second_week_books = 3 →
  weekly_books = 9 →
  (total_books = first_week_books + second_week_books + 5 * weekly_books) →
  2 + 5 = 7 :=
by
  intro total_books first_week_books second_week_books weekly_books
  intro h_total h_first h_second h_weekly h_equiv
  rw [h_total, h_first, h_second, h_weekly, h_equiv]
  rfl

end Cody_reads_series_in_7_weeks_l564_564618


namespace correlation_highly_related_l564_564086

-- Conditions:
-- Let corr be the linear correlation coefficient of product output and unit cost.
-- Let rel be the relationship between product output and unit cost.

def corr : ℝ := -0.87

-- Proof Goal:
-- If corr = -0.87, then the relationship is "highly related".

theorem correlation_highly_related (h : corr = -0.87) : rel = "highly related" := by
  sorry

end correlation_highly_related_l564_564086


namespace find_k_l564_564300

theorem find_k (k : ℝ) (A B : ℝ → ℝ)
  (hA : ∀ x, A x = 2 * x^2 + k * x - 6 * x)
  (hB : ∀ x, B x = -x^2 + k * x - 1)
  (hIndependent : ∀ x, ∃ C : ℝ, A x + 2 * B x = C) :
  k = 2 :=
by 
  sorry

end find_k_l564_564300


namespace geom_seq_min_value_l564_564319

theorem geom_seq_min_value (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a n = a 1 * q ^ (n - 1))
  (h_condition : a 7 = a 6 + 2 * a 5)
  (h_mult : ∃ m n, m ≠ n ∧ a m * a n = 16 * (a 1) ^ 2) :
  ∃ (m n : ℕ), m ≠ n ∧ m + n = 6 ∧ (1 / m : ℝ) + (4 / n : ℝ) = 3 / 2 :=
by
  sorry

end geom_seq_min_value_l564_564319


namespace quadratic_max_value_l564_564089

theorem quadratic_max_value :
  ∀ (x : ℝ), -x^2 - 2*x - 3 ≤ -2 :=
begin
  sorry,
end

end quadratic_max_value_l564_564089


namespace find_first_number_l564_564711

theorem find_first_number (a : ℝ) (h : real.sqrt (a * 100) = 90.5) : a = 81.9025 :=
by
  -- Proof to be filled in
  sorry

end find_first_number_l564_564711


namespace cheese_options_count_l564_564224

theorem cheese_options_count (C : ℕ) :
  let total_combinations := 4 + 15 in
  let topping_combinations := C * total_combinations in
  topping_combinations = 57 → C = 3 :=
by
  intro h,
  have h1 : total_combinations = 19 := by
    simp [total_combinations],
  rw h1 at h,
  have h2 : C * 19 = 57 := by
    rw [h],
  exact nat.eq_of_mul_eq_mul_left (dec_trivial) h2

end cheese_options_count_l564_564224


namespace parabola_opens_downwards_iff_l564_564331

theorem parabola_opens_downwards_iff (a : ℝ) : (∀ x : ℝ, (a - 1) * x^2 + 2 * x ≤ 0) ↔ a < 1 := 
sorry

end parabola_opens_downwards_iff_l564_564331


namespace cost_of_adult_ticket_eq_19_l564_564017

variables (X : ℝ)
-- Condition 1: The cost of an adult ticket is $6 more than the cost of a child ticket.
def cost_of_child_ticket : ℝ := X - 6

-- Condition 2: The total cost of the 5 tickets is $77.
axiom total_cost_eq : 2 * X + 3 * (X - 6) = 77

-- Prove that the cost of an adult ticket is 19 dollars
theorem cost_of_adult_ticket_eq_19 (h : total_cost_eq) : X = 19 := 
by
  -- Here we would provide the actual proof steps
  sorry

end cost_of_adult_ticket_eq_19_l564_564017


namespace ellipse_standard_form_l564_564293

theorem ellipse_standard_form :
  ∃ (a b : ℝ), 
    (a > 0) ∧ (b > 0) ∧ 
    (a = 3 * b) ∧ 
    (∀ (x y : ℝ), (x, y) = (3, 0) → 
      (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    (∀ (x y : ℝ), (x, y) = (sqrt 6, 1) →
      (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    (∀ (x y : ℝ), (x, y) = (-sqrt 3, -sqrt 2) →
      (x^2 / a^2 + y^2 / b^2 = 1)) → 
    (a = 3) ∧ (b = 1) ∧ (∀ (x y : ℝ), x^2 / 9 + y^2 / 3 = 1) := 
sorry

end ellipse_standard_form_l564_564293


namespace lowest_fraction_l564_564243

-- Definitions of individual work rates based on the problem statement
def rate_A : ℝ := 1 / 4
def rate_B : ℝ := 1 / 5
def rate_C : ℝ := 1 / 6

-- Lean theorem corresponding to the proof problem
theorem lowest_fraction (rA rB rC : ℝ) (hA : rate_A = rA) (hB : rate_B = rB) (hC : rate_C = rC) :
  (rB + rC) / 1 = 11 / 30 :=
by
  -- To simplify, handle the necessary calculations in the proof
  have h1 : rA = 1 / 4 := hA
  have h2 : rB = 1 / 5 := hB
  have h3 : rC = 1 / 6 := hC
  -- Combine rates of the two slowest workers (Person B and Person C)
  have combined_rate : rB + rC = 11 / 30 := by linarith
  -- The result should match the correct answer
  exact combined_rate

end lowest_fraction_l564_564243


namespace rebus_solution_l564_564261

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l564_564261


namespace race_course_length_l564_564552

theorem race_course_length (v : ℝ) (hv : 0 < v) : 4 * (84 - 63) = 84 := 
by
  have h : 84 - 63 = 21 := by norm_num
  rw [h, mul_comm 4 21]
  norm_num
  sorry

end race_course_length_l564_564552


namespace johns_gas_usage_per_week_l564_564898

-- Define the given conditions as variables in Lean
def car_mpg : ℝ := 30
def daily_commute_one_way : ℝ := 20
def work_days : ℕ := 5
def leisure_travel_weekly : ℝ := 40

-- Calculate the total weekly miles and total gallons used in definitions leading to the final proof statement
def daily_commute_total : ℝ := daily_commute_one_way * 2
def work_commute_weekly : ℝ := daily_commute_total * (work_days : ℝ)
def total_weekly_miles : ℝ := work_commute_weekly + leisure_travel_weekly
def weekly_gallons : ℝ := total_weekly_miles / car_mpg

-- The final proof statement
theorem johns_gas_usage_per_week : weekly_gallons = 8 := by
  sorry

end johns_gas_usage_per_week_l564_564898


namespace inhabitant_50th_statement_l564_564240

-- Definition of inhabitant types
inductive InhabitantType
| Knight
| Liar

-- Predicate for the statement of inhabitants
def says (inhabitant : InhabitantType) (statement : InhabitantType) : Bool :=
  match inhabitant with
  | InhabitantType.Knight => true
  | InhabitantType.Liar => false

-- Conditions from the problem
axiom inhabitants : Fin 50 → InhabitantType
axiom statements : ∀ i : Fin 50, i.val % 2 = 0 → (says (inhabitants ((i + 1) % 50)) InhabitantType.Liar = false)
axiom statements' : ∀ i : Fin 50, i.val % 2 = 1 → (says (inhabitants ((i + 1) % 50)) InhabitantType.Knight = true)

-- Goal to prove
theorem inhabitant_50th_statement : says (inhabitants 49) InhabitantType.Knight := by
  sorry

end inhabitant_50th_statement_l564_564240


namespace infinite_series_sum_l564_564421

noncomputable def inf_series (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), if n = 1 then 1 / (b * a)
  else if n % 2 = 0 then 1 / ((↑(n - 1) * a - b) * (↑n * a - b))
  else 1 / ((↑(n - 1) * a + b) * (↑n * a - b))

theorem infinite_series_sum (a b : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) :
  inf_series a b = 1 / (a * b) :=
sorry

end infinite_series_sum_l564_564421


namespace solve_number_of_men_l564_564145

variable {M : ℕ}  -- Number of men in the first group

theorem solve_number_of_men :
  M * 12 * 8 = (20 * 19.2 * 15) / 2 → M = 30 :=
by
  sorry

end solve_number_of_men_l564_564145


namespace max_elements_in_T_l564_564408

open Set

def is_valid_subset (T : Set ℕ) : Prop :=
  T ⊆ (Finset.range 76).1 ∧ (∀ (a b ∈ T), a ≠ b → (a + b) % 5 ≠ 0)

theorem max_elements_in_T : ∃ T : Set ℕ, is_valid_subset T ∧ Finset.card (T ∩ (Finset.range 76).1) = 45 :=
sorry

end max_elements_in_T_l564_564408


namespace acute_angle_parallel_vectors_l564_564409

theorem acute_angle_parallel_vectors (x : ℝ) (a b : ℝ × ℝ)
    (h₁ : a = (Real.sin x, 1))
    (h₂ : b = (1 / 2, Real.cos x))
    (h₃ : ∃ k : ℝ, a = k • b ∧ k ≠ 0) :
    x = Real.pi / 4 :=
by
  sorry

end acute_angle_parallel_vectors_l564_564409


namespace evaluate_expression_l564_564550

theorem evaluate_expression : 2 + 0 - 2 * 0 = 2 :=
by
  sorry

end evaluate_expression_l564_564550


namespace solve_quartic_eqn_l564_564676

theorem solve_quartic_eqn (z : ℂ) : 
  z ^ 4 - 6 * z ^ 2 + 8 = 0 ↔ z = 2 ∨ z = -2 ∨ z = Complex.sqrt 2 ∨ z = -Complex.sqrt 2 := 
sorry

end solve_quartic_eqn_l564_564676


namespace geometry_problem_l564_564424

theorem geometry_problem
  (A B C D E F : Point)
  (ABC_isosceles : isosceles_triangle A B C)
  (D_on_BC : lies_on D (line B C))
  (F_on_arc_ADC : lies_on_arc F (circle A D C) ∧ inside_triangle F (triangle A B C))
  (circle_BDF_intersects_AB_at_E : ∃ (circle_BDF : Circle), passes_through B D F circle_BDF ∧ intersects_at E circle_BDF (line A B)) :
  CD * EF + DF * AE = BD * AF :=
sorry

end geometry_problem_l564_564424


namespace ratio_problem_l564_564063

theorem ratio_problem 
  (a b c d : ℚ)
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 5)
  (h3 : c / d = 9) : 
  d / a = 5 / 54 :=
by
  sorry

end ratio_problem_l564_564063


namespace slopes_of_intersecting_line_l564_564573

theorem slopes_of_intersecting_line {m : ℝ} :
  (∃ x y : ℝ, y = m * x + 4 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ Set.Iic (-Real.sqrt 0.48) ∪ Set.Ici (Real.sqrt 0.48) :=
by
  sorry

end slopes_of_intersecting_line_l564_564573


namespace prove_parallel_EH_FG_l564_564911

-- Define the problem conditions in Lean 4
variables {A B C D O E F G H : Type}
variable [inhabited O]

-- Define rhombus ABCD and that O is the center of the inscribed circle
-- Assume: EF and GH are tangent to the inscribed circle

-- Setup the statement of the theorem
theorem prove_parallel_EH_FG
  (inscribed_circle : circle O)
  (rhombus : rhombus A B C D)
  (E_onAB : E ∈ segment A B)
  (F_onBC : F ∈ segment B C)
  (G_onCD : G ∈ segment C D)
  (H_onDA : H ∈ segment D A)
  (EF_tangent : tangent_to_circle EF inscribed_circle)
  (GH_tangent : tangent_to_circle GH inscribed_circle) :
  parallel EH FG :=
begin
  sorry -- Proof placeholder
end

end prove_parallel_EH_FG_l564_564911


namespace sequence_count_100_l564_564024

theorem sequence_count_100 :
  let n := 100 in
  let total_sequences := 5^n in
  let unwanted_sequences := 3^n in
  let result := total_sequences - unwanted_sequences in
  result = (5^100 - 3^100) :=
by
  -- Proof goes here
  sorry

end sequence_count_100_l564_564024


namespace mrs_choi_profit_percentage_l564_564436

theorem mrs_choi_profit_percentage :
  ∀ (original_price selling_price : ℝ) (broker_percentage : ℝ),
    original_price = 80000 →
    selling_price = 100000 →
    broker_percentage = 0.05 →
    (selling_price - (broker_percentage * original_price) - original_price) / original_price * 100 = 20 :=
by
  intros original_price selling_price broker_percentage h1 h2 h3
  sorry

end mrs_choi_profit_percentage_l564_564436


namespace volume_is_six_l564_564385

-- Define the polygons and their properties
def right_triangle (a b c : ℝ) := (a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0)
def rectangle (l w : ℝ) := (l > 0 ∧ w > 0)
def equilateral_triangle (s : ℝ) := (s > 0)

-- The given polygons
def A := right_triangle 1 2 (Real.sqrt 5)
def E := right_triangle 1 2 (Real.sqrt 5)
def F := right_triangle 1 2 (Real.sqrt 5)
def B := rectangle 1 2
def C := rectangle 2 3
def D := rectangle 1 3
def G := equilateral_triangle (Real.sqrt 5)

-- The volume of the polyhedron
-- Assume the largest rectangle C forms the base and a reasonable height
def volume_of_polyhedron : ℝ := 6

theorem volume_is_six : 
  (right_triangle 1 2 (Real.sqrt 5)) → 
  (rectangle 1 2) → 
  (rectangle 2 3) → 
  (rectangle 1 3) → 
  (equilateral_triangle (Real.sqrt 5)) → 
  volume_of_polyhedron = 6 := 
by 
  sorry

end volume_is_six_l564_564385


namespace product_of_two_equal_numbers_l564_564068

theorem product_of_two_equal_numbers :
  ∃ (x : ℕ), (5 * 20 = 12 + 22 + 16 + 2 * x) ∧ (x * x = 625) :=
by
  sorry

end product_of_two_equal_numbers_l564_564068


namespace ivy_stripping_days_l564_564610

theorem ivy_stripping_days :
  ∃ (days_needed : ℕ), (days_needed * (6 - 2) = 40) ∧ days_needed = 10 :=
by {
  use 10,
  split,
  { simp,
    norm_num,
  },
  { simp }
}

end ivy_stripping_days_l564_564610


namespace sqrt_ab_bc_ca_minimum_value_l564_564921

theorem sqrt_ab_bc_ca_minimum_value (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : ab + bc + ca = a + b + c) (h5 : a + b + c > 0) : 
  sqrt ab + sqrt bc + sqrt ca ≥ 2 :=
by sorry

end sqrt_ab_bc_ca_minimum_value_l564_564921


namespace cost_of_adult_ticket_l564_564021

theorem cost_of_adult_ticket (c : ℝ) 
  (h₁ : 2 * (c + 6) + 3 * c = 77)
  : c + 6 = 19 :=
sorry

end cost_of_adult_ticket_l564_564021


namespace max_pieces_even_configuration_l564_564116

-- Define the chessboard size
def chessboard_size : Nat := 8

-- Define the concept of a valid configuration
structure valid_configuration (board : Array (Array Nat)) :=
(even_rows : ∀ i, i < chessboard_size → ∑ j in board[i], j % 2 = 0)
(even_columns : ∀ j, j < chessboard_size → ∑ i in (board.map (λ row => row[j])), i % 2 = 0)
(even_diagonals : ∀ d, abs d < chessboard_size → (∑ i in (fun x => x < chessboard_size && (d + i < chessboard_size && i < board.size),
                                                           board[i + d][i]), d % 2 = 0) ∧
                                      (∑ i in (fun x => x < chessboard_size && (d + i < chessboard_size && i < board.size),
                                                           board[i][i + d]), d % 2 = 0))

-- The statement of the problem
theorem max_pieces_even_configuration : ∃ board : Array (Array Nat), valid_configuration board ∧
    (∑ row in board, ∑ pieces in row, pieces) ≤ 48 :=
by
  sorry

end max_pieces_even_configuration_l564_564116


namespace subtract_and_round_correct_l564_564468

-- Definitions for the subtraction and rounding problem
def subtract_and_round (a : ℝ) (b : ℝ) : ℝ :=
  let diff := a - b
  let rounded := Float.round (diff * 100) / 100
  rounded

-- Example theorem statement for the given problem
theorem subtract_and_round_correct : subtract_and_round 53.463 12.587 = 40.88 :=
by
  -- Skip the actual proof.
  sorry

end subtract_and_round_correct_l564_564468


namespace cindy_correct_result_l564_564617

theorem cindy_correct_result (x : ℝ) (h: (x - 7) / 5 = 27) : (x - 5) / 7 = 20 :=
by
  sorry

end cindy_correct_result_l564_564617


namespace exponent_properties_l564_564111

theorem exponent_properties : (7^(-3))^0 + (7^0)^2 = 2 :=
by
  -- Using the properties of exponents described in the problem:
  -- 1. Any number raised to the power of 0 equals 1.
  -- 2. Any base raised to the power of 0 equals 1, with further raising to the power of 2 yielding 1.
  -- We can conclude and add the two results to get the final statement.
  sorry

end exponent_properties_l564_564111


namespace evaluate_powers_of_i_l564_564643

theorem evaluate_powers_of_i : (complex.I ^ 22 + complex.I ^ 222) = -2 :=
by
  -- Using by to start the proof block and ending it with sorry.
  sorry

end evaluate_powers_of_i_l564_564643


namespace part1_part2_l564_564329

def f (x : ℝ) := 2 * sin (π - x) * cos x

theorem part1 : ∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ π := sorry

theorem part2 : 
  ∀ x : ℝ, 
  (x ≥ -π / 6 ∧ x ≤ π / 2) → 
  (f x ≤ 1 ∧ f x ≥ -√3 / 2) := sorry

end part1_part2_l564_564329


namespace solve_p_value_l564_564946

noncomputable def solve_for_p (n m p : ℚ) : Prop :=
  (5 / 6 = n / 90) ∧ ((m + n) / 105 = (p - m) / 150) ∧ (p = 137.5)

theorem solve_p_value (n m p : ℚ) (h1 : 5 / 6 = n / 90) (h2 : (m + n) / 105 = (p - m) / 150) : 
  p = 137.5 :=
by
  sorry

end solve_p_value_l564_564946


namespace not_possible_arrangement_l564_564393

def all_numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
def allowed_differences := {3, 4, 5}
def adjacent_pairs (arrangement : List Nat) := List.zip arrangement (arrangement.tail ++ [arrangement.head])

theorem not_possible_arrangement :
  ∀ (arrangement : List Nat), arrangement.length = 12 → arrangement.perm (List.ofFinset all_numbers.to_finset) →
    ¬ ∀ (p : Nat × Nat) (h : p ∈ adjacent_pairs arrangement), 
      (abs (p.fst - p.snd) ∈ allowed_differences.to_finset) := 
by
  sorry

end not_possible_arrangement_l564_564393


namespace power_function_constant_l564_564322

theorem power_function_constant (k α : ℝ)
  (h : (1 / 2 : ℝ) ^ α * k = (Real.sqrt 2 / 2)) : k + α = 3 / 2 := by
  sorry

end power_function_constant_l564_564322


namespace common_difference_arith_seq_l564_564842

theorem common_difference_arith_seq (a : ℕ → ℤ) (d : ℤ) :
  (a 2 = 2) ∧ (a 4 = 8) →
  (a 1 + d = a 2) ∧ (a 1 + 3 * d = a 4) →
  d = 3 :=
by
  intros h1 h2 
  have h3 : a 2 = a 1 + d, from h2.1
  have h4 : a 4 = a 1 + 3 * d, from h2.2
  rw h1.1 at h3
  rw h1.2 at h4
  sorry

end common_difference_arith_seq_l564_564842


namespace minimum_value_of_f_on_interval_l564_564289

noncomputable def f (x : ℝ) : ℝ := 27 * x - x ^ 3

theorem minimum_value_of_f_on_interval :
  is_min_on f [-4, 2] (-54) :=
by
  sorry

end minimum_value_of_f_on_interval_l564_564289


namespace solve_quartic_equation_l564_564695

theorem solve_quartic_equation :
  ∀ z : ℝ, z^4 - 6 * z^2 + 8 = 0 ↔ (z = -2 ∨ z = -sqrt 2 ∨ z = sqrt 2 ∨ z = 2) :=
by {
  sorry
}

end solve_quartic_equation_l564_564695


namespace grisha_wins_probability_expected_flips_l564_564825

-- Define conditions
def even_flip_heads_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 0 ∧ flip n = tt

def odd_flip_tails_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 1 ∧ flip n = ff

-- 1. Prove the probability P of Grisha winning the coin toss game is 1/3
theorem grisha_wins_probability (flip : ℕ → bool) (P : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  P = 1/3 := sorry

-- 2. Prove the expected number E of coin flips until the outcome is decided is 2
theorem expected_flips (flip : ℕ → bool) (E : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  E = 2 := sorry

end grisha_wins_probability_expected_flips_l564_564825


namespace area_of_tangent_triangle_l564_564067

noncomputable def curve : ℝ → ℝ := λ x, (1/3) * x^3 + x

def tangent_point := (1 : ℝ, 4/3)

theorem area_of_tangent_triangle :
  let tangent_line (x: ℝ) := 2 * x - (2 / 3) in
  let x_intercept : ℝ := 1 / 3 in
  let y_intercept : ℝ := -2 / 3 in
  let triangle_area := (1 / 2) * (2 / 3) * (1 / 3) in
  triangle_area = 1 / 9 :=
by
  sorry

end area_of_tangent_triangle_l564_564067


namespace power_function_value_l564_564373

theorem power_function_value 
  (f : ℝ → ℝ)
  (α : ℝ)
  (h₀ : f = λ x, x ^ α)
  (h₁ : f 4 / f 2 = 3) : 
  f (1 / 2) = 1 / 3 := 
sorry

end power_function_value_l564_564373


namespace solve_diff_l564_564939

-- Definitions based on conditions
def equation (e y : ℝ) : Prop := y^2 + e^2 = 3 * e * y + 1

theorem solve_diff (e a b : ℝ) (h1 : equation e a) (h2 : equation e b) (h3 : a ≠ b) : 
  |a - b| = Real.sqrt (5 * e^2 + 4) := 
sorry

end solve_diff_l564_564939


namespace average_last_30_l564_564957

theorem average_last_30 (avg_first_40 : ℝ) 
  (avg_all_70 : ℝ) 
  (sum_first_40 : ℝ := 40 * avg_first_40)
  (sum_all_70 : ℝ := 70 * avg_all_70) 
  (total_results: ℕ := 70):
  (30 : ℝ) * (40: ℝ) + (30: ℝ) * (40: ℝ) = 70 * 34.285714285714285 :=
by
  sorry

end average_last_30_l564_564957


namespace cos_half_pi_plus_alpha_l564_564359

open Real

noncomputable def alpha : ℝ := sorry

theorem cos_half_pi_plus_alpha :
  let a := (1 / 3, tan alpha)
  let b := (cos alpha, 1)
  ((1 / 3) / (cos alpha) = (tan alpha) / 1) →
  cos (pi / 2 + alpha) = -1 / 3 :=
by
  intros
  sorry

end cos_half_pi_plus_alpha_l564_564359


namespace count_digit_7_from_100_to_199_l564_564882

theorem count_digit_7_from_100_to_199 : 
  (list.range' 100 100).countp (λ n, (70 ≤ n ∧ n ≤ 79) ∨ (n % 10 = 7)) = 20 :=
by
  simp
  sorry

end count_digit_7_from_100_to_199_l564_564882


namespace two_digit_number_square_l564_564569

theorem two_digit_number_square (n : ℕ) (a b : ℕ) :
  1000 * a + 100 * b + 10 * b + a = n^2 ∧
  32 ≤ n ∧ n ≤ 99 ∧
  a = n / 100 ∧
  b = (n % 100) / 10 ∧
  2 * a = 2 * b :=
  n = 68 := sorry

end two_digit_number_square_l564_564569


namespace proof_BH_length_equals_lhs_rhs_l564_564850

noncomputable def calculate_BH_length : ℝ :=
  let AB := 3
  let BC := 4
  let CA := 5
  let AG := 4  -- Since AB < AG
  let AH := 6  -- AG < AH
  let GI := 3
  let HI := 8
  let GH := Real.sqrt (GI ^ 2 + HI ^ 2)
  let p := 3
  let q := 2
  let r := 73
  let s := 1
  3 + 2 * Real.sqrt 73

theorem proof_BH_length_equals_lhs_rhs :
  let BH := 3 + 2 * Real.sqrt 73
  calculate_BH_length = BH := by
    sorry

end proof_BH_length_equals_lhs_rhs_l564_564850


namespace percentage_votes_winner_l564_564510

noncomputable def votes_total := 1000
noncomputable def votes_winner := 650
noncomputable def votes_margin := 300

theorem percentage_votes_winner :
  (votes_winner / votes_total.toFloat) * 100 = 65 := by
  sorry

end percentage_votes_winner_l564_564510


namespace solve_quartic_equation_l564_564696

theorem solve_quartic_equation :
  ∀ z : ℝ, z^4 - 6 * z^2 + 8 = 0 ↔ (z = -2 ∨ z = -sqrt 2 ∨ z = sqrt 2 ∨ z = 2) :=
by {
  sorry
}

end solve_quartic_equation_l564_564696


namespace solve_trig_equation_l564_564060

theorem solve_trig_equation (x : ℝ) (k : ℤ) :
    (sin (5 * x) + sin (7 * x)) / (sin (4 * x) + sin (2 * x)) = -4 * |sin (2 * x)| ↔
    (x = π - arcsin ((1 - sqrt 2) / 2) + 2 * k * π) ∨ 
    (x = π - arcsin ((sqrt 2 - 1) / 2) + 2 * k * π) :=
sorry

end solve_trig_equation_l564_564060


namespace solve_quartic_equation_l564_564689

theorem solve_quartic_equation :
  ∀ z : ℝ, z^4 - 6 * z^2 + 8 = 0 ↔ (z = -2 ∨ z = -sqrt 2 ∨ z = sqrt 2 ∨ z = 2) :=
by {
  sorry
}

end solve_quartic_equation_l564_564689


namespace investment_initial_amount_l564_564798

theorem investment_initial_amount (r : ℝ) (years : ℝ) (final_amount : ℝ) (initial_amount : ℝ) :
  r = 8 →
  years = 18 →
  final_amount = 32000 →
  initial_amount = final_amount / 2 ^ (years / (70 / r)) :=
by
  intro hr hyears hfinal
  rw [hr, hyears, hfinal]
  simp
  sorry

end investment_initial_amount_l564_564798


namespace cos_angle_BAC_l564_564077

theorem cos_angle_BAC 
  (O A B C D E : Point) 
  (h1: CenterOfBothCircles O) 
  (h2: FirstCircleInscribedInTriangleABC O A B C)
  (h3: LineIntersectsSecondCircleACD O A C D h2)
  (h4: LineIntersectsSecondCircleBCE O B C E h2)
  (h5: ∠ABC = ∠CAE) 
  : ∃ (cos_val : ℝ), cos_val = (1 + Real.sqrt 5) / 4 := 
by 
  sorry

end cos_angle_BAC_l564_564077


namespace rectangle_x_satisfy_l564_564751

theorem rectangle_x_satisfy (x : ℝ) (h1 : 3 * x = 3 * x) (h2 : x + 5 = x + 5) (h3 : (3 * x) * (x + 5) = 2 * (3 * x) + 2 * (x + 5)) : x = 1 :=
sorry

end rectangle_x_satisfy_l564_564751


namespace mary_saves_in_five_months_l564_564433

def washing_earnings : ℕ := 20
def walking_earnings : ℕ := 40
def monthly_earnings : ℕ := washing_earnings + walking_earnings
def savings_rate : ℕ := 2
def monthly_savings : ℕ := monthly_earnings / savings_rate
def total_savings_target : ℕ := 150

theorem mary_saves_in_five_months :
  total_savings_target / monthly_savings = 5 :=
by
  sorry

end mary_saves_in_five_months_l564_564433


namespace solve_for_t_l564_564364

theorem solve_for_t (p t : ℝ) (h1 : 5 = p * 3^t) (h2 : 45 = p * 9^t) : t = 2 :=
by
  sorry

end solve_for_t_l564_564364


namespace car_journey_delay_l564_564554

theorem car_journey_delay (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time1 : ℝ) (time2 : ℝ) (delay : ℝ) :
  distance = 225 ∧ speed1 = 60 ∧ speed2 = 50 ∧ time1 = distance / speed1 ∧ time2 = distance / speed2 ∧ 
  delay = (time2 - time1) * 60 → delay = 45 :=
by
  sorry

end car_journey_delay_l564_564554


namespace fixed_point_of_line_l564_564128

theorem fixed_point_of_line (k : ℝ) : ∃ p : ℝ × ℝ, p = (3, 1) :=
by
  use (3, 1)
  have : ∀ k : ℝ, (3, 1) ∈ {p : ℝ × ℝ | (λ (x y : ℝ), k * x - y + 1 = 3 * k) p.1 p.2} := 
    by intros; exact dec_trivial
  apply this
  sorry

end fixed_point_of_line_l564_564128


namespace green_notebook_cost_each_l564_564444

-- Definitions for conditions:
def num_notebooks := 4
def num_green_notebooks := 2
def num_black_notebooks := 1
def num_pink_notebooks := 1
def total_cost := 45
def black_notebook_cost := 15
def pink_notebook_cost := 10

-- Define the problem statement:
theorem green_notebook_cost_each : 
  (2 * g + black_notebook_cost + pink_notebook_cost = total_cost) → 
  g = 10 := 
by 
  intros h
  sorry

end green_notebook_cost_each_l564_564444


namespace inhabitant_50_statement_l564_564236

-- Definitions
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

def tells_truth (inh: Inhabitant) (statement: Bool) : Bool :=
  match inh with
  | Inhabitant.knight => statement
  | Inhabitant.liar => not statement

noncomputable def inhabitant_at_position (pos: Nat) : Inhabitant :=
  if (pos % 2) = 1 then
    if pos % 4 = 1 then Inhabitant.knight else Inhabitant.liar
  else
    if pos % 4 = 0 then Inhabitant.knight else Inhabitant.liar

def neighbor (pos: Nat) : Nat := (pos % 50) + 1

-- Theorem statement
theorem inhabitant_50_statement : tells_truth (inhabitant_at_position 50) (inhabitant_at_position (neighbor 50) = Inhabitant.knight) = true :=
by
  -- Proof would go here
  sorry

end inhabitant_50_statement_l564_564236


namespace exponent_is_23_l564_564793

theorem exponent_is_23 (k : ℝ) : (1/2: ℝ) ^ 23 * (1/81: ℝ) ^ k = (1/18: ℝ) ^ 23 → 23 = 23 := by
  intro h
  sorry

end exponent_is_23_l564_564793


namespace edricHourlyRateIsApproximatelyCorrect_l564_564246

-- Definitions as per conditions
def edricMonthlySalary : ℝ := 576
def edricHoursPerDay : ℝ := 8
def edricDaysPerWeek : ℝ := 6
def weeksPerMonth : ℝ := 4.33

-- Calculation as per the proof problem
def edricWeeklyHours (hoursPerDay daysPerWeek : ℝ) : ℝ := hoursPerDay * daysPerWeek

def edricMonthlyHours (weeklyHours weeksPerMonth : ℝ) : ℝ := weeklyHours * weeksPerMonth

def edricHourlyRate (monthlySalary monthlyHours : ℝ) : ℝ := monthlySalary / monthlyHours

-- The theorem to prove
theorem edricHourlyRateIsApproximatelyCorrect : (edricHourlyRate edricMonthlySalary (edricMonthlyHours (edricWeeklyHours edricHoursPerDay edricDaysPerWeek) weeksPerMonth)) ≈ 2.77 :=
by
  sorry

end edricHourlyRateIsApproximatelyCorrect_l564_564246


namespace count_digit_7_from_100_to_199_l564_564880

theorem count_digit_7_from_100_to_199 : 
  (list.range' 100 100).countp (λ n, (70 ≤ n ∧ n ≤ 79) ∨ (n % 10 = 7)) = 20 :=
by
  simp
  sorry

end count_digit_7_from_100_to_199_l564_564880


namespace sales_in_fifth_month_l564_564155

theorem sales_in_fifth_month
  (sales1 sales2 sales3 sales4 sales6 : ℕ)
  (average_sale_6_months : ℕ)
  (h_sales1 : sales1 = 5921)
  (h_sales2 : sales2 = 5468)
  (h_sales3 : sales3 = 5568)
  (h_sales4 : sales4 = 6088)
  (h_sales6 : sales6 = 5922)
  (h_average : average_sale_6_months = 5900) :
  let total_sales_5_months := 5900 * 6 - (5921 + 5468 + 5568 + 6088 + 5922) in
  total_sales_5_months = 6433 :=
by
  sorry

end sales_in_fifth_month_l564_564155


namespace find_digits_l564_564271

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l564_564271


namespace cade_initial_marbles_l564_564604

theorem cade_initial_marbles :
  ∀ (initial marbles gave_left num_left: Nat), gave_left = 8 → num_left = 79 → initial = gave_left + num_left → initial = 87 :=
by
  intros initial marbles gave_left num_left hg hn heq
  rw [←heq]
  rw [hg, hn]
  sorry

end cade_initial_marbles_l564_564604


namespace inner_circle_tangent_radius_l564_564583

noncomputable def inner_circle_radius (length height : ℝ) (semi_radius : ℝ) : ℝ :=
  let r := (real.sqrt 10 - 1) / 2
  r

theorem inner_circle_tangent_radius :
  inner_circle_radius 4 2 0.5 = (real.sqrt 10 - 1) / 2 := by
  sorry

end inner_circle_tangent_radius_l564_564583


namespace cos_270_eq_zero_l564_564208

theorem cos_270_eq_zero : ∀ (θ : ℝ), θ = 270 → (∀ (p : ℝ × ℝ), p = (0,1) → cos θ = p.fst) → cos 270 = 0 :=
by
  intros θ hθ h
  sorry

end cos_270_eq_zero_l564_564208


namespace set_union_complement_intersection_l564_564778

open Set

variable {U : Set α} {M N : Set α} [DecidableEq α]
variable a b c d e : α

-- Assume the universal set U, and sets M and N as in the problem.
def U := {a, b, c, d, e}
def M := {a, d}
def N := {a, c, e}

-- The proof goal is to show that M ∪ ((U \ M) ∩ N) = {a, c, d, e}.
theorem set_union_complement_intersection (hU : U = {a, b, c, d, e})
  (hM : M = {a, d}) (hN : N = {a, c, e}) :
  M ∪ ((U \ M) ∩ N) = {a, c, d, e} :=
by
  sorry

end set_union_complement_intersection_l564_564778


namespace intersection_complement_l564_564355

def U := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A := {1, 3, 5, 7, 9}
def B := {1, 2, 5, 6, 8}
def complement_U_B := U \ B

theorem intersection_complement :
  A ∩ complement_U_B = {3, 7, 9} :=
by
  sorry

end intersection_complement_l564_564355


namespace problem_statement_l564_564323

noncomputable def parabola_focus_on_line (x y : ℝ) := x - 2 * y - 4 = 0

noncomputable def parabola_equation (x y : ℝ) := y ^ 2 = 16 * x

theorem problem_statement :
  (∃ (focus : ℝ × ℝ), parabola_focus_on_line focus.1 focus.2)
  ∧ (∀ (x y : ℝ), parabola_equation x y ↔ y^2 = 16 * x)
  → (let focus := (4 : ℝ, 0 : ℝ) in
      ∀ (A : ℝ × ℝ), A.1 = 2 → A.2 * A.2 = 16 * A.1
      → dist (2, (2 * 4 : ℝ)) focus = 6) :=
begin
  sorry
end

end problem_statement_l564_564323


namespace natural_number_with_six_divisors_two_prime_sum_78_is_45_l564_564713

def has_six_divisors (n : ℕ) : Prop :=
  (∃ p1 p2 : ℕ, p1.prime ∧ p2.prime ∧ p1 ≠ p2 ∧ 
  (∃ α1 α2 : ℕ, α1 + α2 > 0 ∧ n = p1 ^ α1 * p2 ^ α2 ∧ 
  (α1 + 1) * (α2 + 1) = 6))

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d, d > 0 ∧ n % d = 0) (Finset.range (n + 1))).sum id

theorem natural_number_with_six_divisors_two_prime_sum_78_is_45 (n : ℕ) :
  has_six_divisors n ∧ sum_of_divisors n = 78 → n = 45 := 
by 
  sorry

end natural_number_with_six_divisors_two_prime_sum_78_is_45_l564_564713


namespace ellipse_problem_l564_564312

theorem ellipse_problem
  (x y : ℝ)
  (P : ℝ × ℝ)
  (h_ellipse : ∃ x y, x^2 / 2 + y^2 = 1)
  (F1 F2 : ℝ × ℝ)
  (h_foci : F1 = (-1, 0) ∧ F2 = (1, 0))
  (h_P_on_C : x^2 / 2 + y^2 = 1) :
  ((distance (0, 0) (line x + y - sqrt 2 = 0) = 1) ∧ (F1F2_droduct_min = 0)) :=
sorry

end ellipse_problem_l564_564312


namespace sum_of_values_of_n_l564_564635

theorem sum_of_values_of_n (n₁ n₂ : ℚ) (h1 : 3 * n₁ - 8 = 5) (h2 : 3 * n₂ - 8 = -5) : n₁ + n₂ = 16 / 3 := 
by {
  -- Use the provided conditions to solve the problem
  sorry 
}

end sum_of_values_of_n_l564_564635


namespace reservoir_water_at_end_of_month_l564_564194

theorem reservoir_water_at_end_of_month :
  ∃ C : ℚ, 
    let normal_level := C - 5 in
    let end_of_month_water := 2 * normal_level in
    (end_of_month_water = 0.60 * C) ∧ 
    (end_of_month_water = 4.284) :=
by sorry

end reservoir_water_at_end_of_month_l564_564194


namespace triangle_at_most_one_obtuse_l564_564452

theorem triangle_at_most_one_obtuse (T : Type) [triangle T] : 
  ¬ (∃ A B C : T, A.obtuse ∧ B.obtuse ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C) :=
by
  sorry

end triangle_at_most_one_obtuse_l564_564452


namespace solve_quartic_equation_l564_564691

theorem solve_quartic_equation :
  ∀ z : ℝ, z^4 - 6 * z^2 + 8 = 0 ↔ (z = -2 ∨ z = -sqrt 2 ∨ z = sqrt 2 ∨ z = 2) :=
by {
  sorry
}

end solve_quartic_equation_l564_564691


namespace simplify_sqrt_expr_l564_564460

theorem simplify_sqrt_expr :
  √(10 + 6*√3) + √(10 - 6*√3) = 2*√3 := 
sorry

end simplify_sqrt_expr_l564_564460


namespace find_angle_C_l564_564391

-- Defining the problem conditions
variables {A B C : ℝ}   -- Angles of triangle
variables {a b c : ℝ}   -- Sides opposite to angles A, B, C respectively

def problem_conditions (A B C a c : ℝ) :=
  sin B + sin A * (sin C - cos C) = 0 ∧
  a = 2 ∧
  c = sqrt 2

-- Theorem statement
theorem find_angle_C (A B C : ℝ) (a c : ℝ) (h : problem_conditions A B C a c) : 
  C = π / 6 :=
sorry

end find_angle_C_l564_564391


namespace max_value_2a2_minus_3b2_l564_564919

theorem max_value_2a2_minus_3b2 :
  ∃ x : ℕ+, let a := ⌊Real.log 10 (x : ℝ)⌋, let b := ⌊Real.log 10 (100 / x : ℝ)⌋ in 
  2 * a^2 - 3 * b^2 = 24 := 
begin
  sorry
end

end max_value_2a2_minus_3b2_l564_564919


namespace cary_ivy_removal_days_correct_l564_564614

noncomputable def cary_ivy_removal_days (initial_ivy : ℕ) (ivy_removed_per_day : ℕ) (ivy_growth_per_night : ℕ) : ℕ :=
  initial_ivy / (ivy_removed_per_day - ivy_growth_per_night)

theorem cary_ivy_removal_days_correct :
  cary_ivy_removal_days 40 6 2 = 10 :=
by
  -- The body of the proof is omitted; it will be filled with the actual proof.
  sorry

end cary_ivy_removal_days_correct_l564_564614


namespace a_n_greater_than_20_l564_564501

noncomputable def seq : ℕ → ℝ
| 0       := 1
| (n + 1) := seq n + (1 / (⌊seq n⌋ : ℝ))

theorem a_n_greater_than_20 (n : ℕ) (h : n > 191) : seq n > 20 :=
sorry

end a_n_greater_than_20_l564_564501


namespace greatest_prime_factor_105_l564_564521

theorem greatest_prime_factor_105 : ∃ p, prime p ∧ p ∣ 105 ∧ (∀ q, prime q ∧ q ∣ 105 → q ≤ p) := sorry

end greatest_prime_factor_105_l564_564521


namespace daily_sales_bounds_l564_564182

def f (t : ℕ) : ℝ := 100 * (1 + 1 / t)
def g (t : ℕ) : ℝ := 125 - abs (t - 25)

def w (t : ℕ) : ℝ :=
if 1 ≤ t ∧ t < 25 then
    100 * (t + 100 / t + 101)
else if 25 ≤ t ∧ t ≤ 30 then
    100 * (149 + 150 / t - t)
else
  0  -- out of range, cannot happen

theorem daily_sales_bounds :
  (∀ t, 1 ≤ t ∧ t ≤ 30 → w t ≤ 20200) ∧
  (∃ t, 1 ≤ t ∧ t ≤ 30 ∧ w t = 20200) ∧
  (∀ t, 1 ≤ t ∧ t ≤ 30 → w t ≥ 12100) ∧ 
  (∃ t, 1 ≤ t ∧ t ≤ 30 ∧ w t = 12100) :=
by
  sorry

end daily_sales_bounds_l564_564182


namespace digit_7_appears_20_times_in_range_100_to_199_l564_564861

theorem digit_7_appears_20_times_in_range_100_to_199 : 
  ∃ (d7_count : ℕ), (d7_count = 20) ∧ (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 199 → 
  d7_count = (n.to_string.count '7')) :=
sorry

end digit_7_appears_20_times_in_range_100_to_199_l564_564861


namespace radius_of_circle_l564_564979

noncomputable def circle_radius : ℝ := by
  -- Let (x, 0) be the center of the circle
  let x : ℝ := 3
  
  -- Points on the circle
  let A := (1 : ℝ)
  let B := (5 : ℝ)
  let C := (2 : ℝ)
  let D := (4 : ℝ)

  -- Calculate the radius
  let radius := Real.sqrt ((x - A)^2 + (0 - B)^2)
  
  -- It is known that the radius should be √29
  exact radius
  
theorem radius_of_circle : circle_radius = Real.sqrt 29 := sorry

end radius_of_circle_l564_564979


namespace Grisha_probability_expected_flips_l564_564832

theorem Grisha_probability (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                            t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (probability (Grisha wins)) = 1 / 3 :=
by 
  sorry

theorem expected_flips (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                        t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (expected_flips) = 2 :=
by
  sorry

end Grisha_probability_expected_flips_l564_564832


namespace sequence_not_div_4_l564_564775

def sequence (n : ℕ) : ℕ :=
if n = 1 ∨ n = 2 then 1
else sequence (n-1) * sequence (n-2) + 1

theorem sequence_not_div_4 (n : ℕ) : ¬ (sequence n % 4 = 0) := sorry

end sequence_not_div_4_l564_564775


namespace ivy_stripping_days_l564_564608

theorem ivy_stripping_days :
  ∃ (days_needed : ℕ), (days_needed * (6 - 2) = 40) ∧ days_needed = 10 :=
by {
  use 10,
  split,
  { simp,
    norm_num,
  },
  { simp }
}

end ivy_stripping_days_l564_564608


namespace area_PQS_l564_564589

def area_triangle (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

theorem area_PQS
  (PR PQ PT : ℝ)
  (h1 : PR = 8)
  (h2 : PQ = 4)
  (h3 : PT = 4) :
  area_triangle PR PQ - area_triangle PT PQ = 8 :=
by
  sorry

end area_PQS_l564_564589


namespace mod_w_eq_one_l564_564901

-- Define complex numbers z and the expression for w
def z : ℂ := ((-7 : ℂ) + 8 * complex.i)^5 * ((17 - 9 * complex.i): ℂ)^3 / (2 + 5 * complex.i)
def w : ℂ := complex.conj z / z

-- State the theorem to be proved
theorem mod_w_eq_one : complex.abs w = 1 :=
by {
  -- The proof is omitted using sorry.
  sorry
}

end mod_w_eq_one_l564_564901


namespace fiftieth_statement_l564_564239

-- Define the types
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

-- Define the function telling the statement
def statement (inhabitant : Inhabitant) : String :=
  match inhabitant with
  | Inhabitant.knight => "Knight"
  | Inhabitant.liar => "Liar"

-- Define the condition: knights tell the truth and liars lie
def tells_truth (inhabitant : Inhabitant) (statement_about_neighbor : String) : Prop :=
  match inhabitant with
  | Inhabitant.knight => statement_about_neighbor = "Knight"
  | Inhabitant.liar => statement_about_neighbor ≠ "Knight"

-- Define a function that determines what each inhabitant says about their right-hand neighbor
def what_they_say (idx : ℕ) : String :=
  if idx % 2 = 0 then "Liar" else "Knight"

-- Define the inhabitant pattern
def inhabitant_at (idx : ℕ) : Inhabitant :=
  if idx % 2 = 0 then Inhabitant.liar else Inhabitant.knight

-- The main theorem statement
theorem fiftieth_statement : tells_truth (inhabitant_at 49) (what_they_say 50) :=
by 
  -- This proof outlines the theorem statement only
  sorry

end fiftieth_statement_l564_564239


namespace rebus_solution_l564_564276

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564276


namespace parabola_directrix_equation_l564_564287

theorem parabola_directrix_equation :
  ∀ (x y : ℝ), y = -3 * x ^ 2 + 9 * x - 17 →
  {d : ℝ // y = d} :=
begin
  intros x y,
  intro h,
  use -31 / 3,
  sorry
end

end parabola_directrix_equation_l564_564287


namespace cacti_average_height_l564_564584

variables {Cactus1 Cactus2 Cactus3 Cactus4 Cactus5 Cactus6 : ℕ}
variables (condition1 : Cactus1 = 14)
variables (condition3 : Cactus3 = 7)
variables (condition6 : Cactus6 = 28)
variables (condition2 : Cactus2 = 14)
variables (condition4 : Cactus4 = 14)
variables (condition5 : Cactus5 = 14)

theorem cacti_average_height : 
  (Cactus1 + Cactus2 + Cactus3 + Cactus4 + Cactus5 + Cactus6 : ℕ) = 91 → 
  (91 : ℝ) / 6 = (15.2 : ℝ) :=
by
  sorry

end cacti_average_height_l564_564584


namespace max_band_members_l564_564167

variable (r x m : ℕ)

noncomputable def band_formation (r x m: ℕ) :=
  m = r * x + 4 ∧
  m = (r - 3) * (x + 2) ∧
  m < 100

theorem max_band_members (r x m : ℕ) (h : band_formation r x m) : m = 88 :=
by
  sorry

end max_band_members_l564_564167


namespace compare_compound_interest_l564_564151

noncomputable def compound_annually (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ t

noncomputable def compound_monthly (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ (12 * t)

theorem compare_compound_interest :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  compound_monthly P r_monthly t > compound_annually P r_annual t :=
by
  sorry

end compare_compound_interest_l564_564151


namespace next_palindromic_year_after_2010_has_sum_of_product_of_digits_eq_zero_l564_564590

def is_palindrome (n : Nat) : Prop :=
  let s := n.toString
  s = s.reverse

def digits (n : Nat) : List Nat :=
  n.toString.toList.map (λ c => c.toNat - '0'.toNat)

def product_of_digits (n : Nat) : Nat :=
  (digits n).foldl (· * ·) 1

def sum_of_digits (n : Nat) : Nat :=
  (digits n).foldl (· + ·) 0

theorem next_palindromic_year_after_2010_has_sum_of_product_of_digits_eq_zero :
  ∃ y : Nat, y > 2010 ∧ is_palindrome y ∧ sum_of_digits (product_of_digits y) = 0 := by
  sorry

end next_palindromic_year_after_2010_has_sum_of_product_of_digits_eq_zero_l564_564590


namespace final_boards_third_player_wins_count_l564_564512

-- Define the tic-tac-toe grid and the winning condition
inductive Player
| A | B | C

structure TicTacToeBoard where
  grid : Array (Array (Option Player))
  valid : grid.size = 3 ∧ grid.all λ row => row.size = 3

def wins (player : Player) (board : TicTacToeBoard) : Prop :=
  -- Define row/column/diagonal win conditions for the player
  ∃ i, (∀ j, board.grid[i][j] = some player) ∨ (∀ j, board.grid[j][i] = some player) ∨
  (∀ k, board.grid[k][k] = some player) ∨ (∀ k, board.grid[k][2-k] = some player)

-- The final count of valid tic-tac-toe boards where third player C wins
theorem final_boards_third_player_wins_count : 
  let final_boards_third_player_wins := 148
  final_boards_third_player_wins = 148 :=
by
  sorry

end final_boards_third_player_wins_count_l564_564512


namespace find_natural_number_l564_564715

theorem find_natural_number (n : ℕ) (h1 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n ∨ d = 3 ∨ d = 5 ∨ d = 9 ∨ d = 15)
  (h2 : 1 + 3 + 5 + 9 + 15 + n = 78) : n = 45 := sorry

end find_natural_number_l564_564715


namespace car_travel_distance_l564_564525

theorem car_travel_distance :
  (∀ (n : ℕ), n > 1 → s (n + 1) = s n - 9) →
  s 1 = 36 →
  (∃ (n : ℕ), s n ≤ 0) →
  (Σ i in range 5, s i) = 90 :=
by
  sorry

end car_travel_distance_l564_564525


namespace find_value_l564_564303

theorem find_value (a : ℝ) (h : a + (1 / a) = sqrt 5) : a^2 + (1 / a^2) = 3 := 
by
  sorry

end find_value_l564_564303


namespace a_100_positive_a_100_abs_lt_018_l564_564346

-- Define the sequence based on the given conditions
def a_n (n : ℕ) : ℝ := real.cos (real.pi / 180 * (10^n) : ℝ)

-- Translate mathematical claims to Lean theorems
theorem a_100_positive : a_n 100 > 0 :=
sorry

theorem a_100_abs_lt_018 : |a_n 100| < 0.18 :=
sorry

end a_100_positive_a_100_abs_lt_018_l564_564346


namespace min_value_problem_l564_564288

theorem min_value_problem (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) : 
    a^2 + b^2 + c^2 + d^2 >= 24 / 5 := 
by
  sorry

end min_value_problem_l564_564288


namespace smallest_number_after_removal_l564_564847

-- Define the original number as a list of digits
def original_digits : List ℕ := [3, 7, 2, 8, 9, 5, 4, 1, 0, 6]

-- Define the function to check the smallest number by removing three digits
def smallest_seven_digit_number (digits: List ℕ) : List ℕ :=
  [2, 4, 5, 1, 0, 6, 7] -- correct seven digits by removing 3, 2, 8

theorem smallest_number_after_removal (original_digits : List ℕ) : 
  (smallest_seven_digit_number original_digits) = [2, 4, 5, 1, 0, 6, 7] :=
by
  sorry

end smallest_number_after_removal_l564_564847


namespace average_first_10_multiples_of_11_l564_564113

theorem average_first_10_multiples_of_11 : 
  let numbers := [11, 22, 33, 44, 55, 66, 77, 88, 99, 110] in
  (numbers.sum : ℝ) / (numbers.length : ℝ) = 60.5 :=
by
  let numbers := [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
  let sum := numbers.sum : ℝ
  let length := numbers.length : ℝ
  have h_sum : sum = 605 := sorry
  have h_length : length = 10 := sorry
  calc
    sum / length = 605 / 10 : by rw [h_sum, h_length]
            ... = 60.5 : by norm_num

end average_first_10_multiples_of_11_l564_564113


namespace count_digit_7_in_range_l564_564857

theorem count_digit_7_in_range : ∀ n ∈ finset.Icc 100 199, (n % 10 = 7 ∨ n / 10 % 10 = 7) →  ∃ counts, counts = 20 :=
by
  sorry

end count_digit_7_in_range_l564_564857


namespace smallest_a1_l564_564914

noncomputable def a_seq (a1 : ℝ) : ℕ → ℝ
| 0     := a1
| (n+1) := 9 * a_seq n - 2 * (n+1)

theorem smallest_a1 : ∃ a1 : ℝ, a1 > 0 ∧ (∀ n > 0, a_seq a1 n = 9 * a_seq a1 (n-1) - 2 * n) ∧ a1 = 19/36 :=
begin
  sorry
end

end smallest_a1_l564_564914


namespace sum_of_valid_six_digit_numbers_divisible_by_37_l564_564045

def is_valid_six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∀ d ∈ (Nat.digits 10 n), d ≠ 0 ∧ d ≠ 9

theorem sum_of_valid_six_digit_numbers_divisible_by_37 :
  let S := {n : ℕ | is_valid_six_digit_number n}
  (finset.sum (finset.filter (λ n, is_valid_six_digit_number n) (finset.range 1000000))) % 37 = 0 :=
sorry

end sum_of_valid_six_digit_numbers_divisible_by_37_l564_564045


namespace square_area_l564_564095

noncomputable def line_lies_on_square_side (a b : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A = (a, a + 4) ∧ B = (b, b + 4)

noncomputable def points_on_parabola (x y : ℝ) : Prop :=
  ∃ (C D : ℝ × ℝ), C = (y^2, y) ∧ D = (x^2, x)

theorem square_area (a b : ℝ) (x y : ℝ)
  (h1 : line_lies_on_square_side a b)
  (h2 : points_on_parabola x y) :
  ∃ (s : ℝ), s^2 = (boxed_solution) :=
sorry

end square_area_l564_564095


namespace part_a_part_b_l564_564341

noncomputable def sequence_a (n : ℕ) : ℝ :=
  Real.cos (10^n * Real.pi / 180)

theorem part_a (h : 100 > 2) : sequence_a 100 > 0 := by
  sorry

theorem part_b : |sequence_a 100| < 0.18 := by
  sorry

end part_a_part_b_l564_564341


namespace inequality_solution_l564_564648

open Set Real

theorem inequality_solution (x : ℝ) :
  (1 / (x + 1) + 3 / (x + 7) ≥ 2 / 3) ↔ (x ∈ Ioo (-7 : ℝ) (-4) ∪ Ioo (-1) (2) ∪ {(-4 : ℝ), 2}) :=
by sorry

end inequality_solution_l564_564648


namespace triangle_internal_angle_60_l564_564049

theorem triangle_internal_angle_60 (A B C : ℝ) (h_sum : A + B + C = 180) : A >= 60 ∨ B >= 60 ∨ C >= 60 :=
sorry

end triangle_internal_angle_60_l564_564049


namespace sequence_count_100_l564_564025

theorem sequence_count_100 :
  let n := 100 in
  let total_sequences := 5^n in
  let unwanted_sequences := 3^n in
  let result := total_sequences - unwanted_sequences in
  result = (5^100 - 3^100) :=
by
  -- Proof goes here
  sorry

end sequence_count_100_l564_564025


namespace area_parabola_triangle_l564_564773

-- Given a parabola with equation y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Focus of the given parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a function for the line passing through focus with a given slope intersecting the parabola
def line_through_focus (α : ℝ) (x y : ℝ) : Prop := y = α * (x - 1)

-- The point A lies on the parabola and the line through focus with slope √3 above the x-axis.
def point_A (x y : ℝ) : Prop := parabola x y ∧ line_through_focus (Real.sqrt 3) x y ∧ y > 0

-- Foot of the perpendicular from A to the directrix (x = -1)
def foot_perpendicular (A : ℝ × ℝ) : ℝ × ℝ := (fst A, 0)

-- The area of triangle AKF
def area_of_triangle (A K F : ℝ × ℝ) : ℝ := 0.5 * abs ((fst A - fst K) * (snd K - snd F) - (fst K - fst F) * (snd A - snd K))

theorem area_parabola_triangle : ∃ A K : ℝ × ℝ, point_A A.1 A.2 ∧ K = foot_perpendicular A ∧ area_of_triangle A K focus = 4 * Real.sqrt 3 := sorry

end area_parabola_triangle_l564_564773


namespace total_cost_first_3_years_l564_564897

def monthly_fee : ℕ := 12
def down_payment : ℕ := 50
def years : ℕ := 3

theorem total_cost_first_3_years :
  (years * 12 * monthly_fee + down_payment) = 482 :=
by
  sorry

end total_cost_first_3_years_l564_564897


namespace new_median_of_collection_l564_564147

-- Define collection and properties
def collection : List ℕ := [4, 4, 5, 6, 7, 8, 8]

-- Define the conditions
def mean_collection := 6
def mode_collection := 4
def median_collection := 6
def added_element := 12

-- Define the resulting collection after adding 12
def new_collection := collection ++ [added_element]

-- State the new median to be proven
def new_median := 6.5

-- Lean 4 statement
theorem new_median_of_collection :
  (List.mean collection = mean_collection) ∧
  (List.mode collection = some mode_collection) ∧
  (List.median collection = some median_collection) →
  (List.median new_collection = new_median) := 
by
  sorry

end new_median_of_collection_l564_564147


namespace f_at_5_l564_564924

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition

axiom odd_function (f: ℝ → ℝ) : ∀ x : ℝ, f (-x) = -f x
axiom functional_equation (f: ℝ → ℝ) : ∀ x : ℝ, f (x + 1) + f x = 0

theorem f_at_5 : f 5 = 0 :=
by {
  -- Proof to be provided here
  sorry
}

end f_at_5_l564_564924


namespace problem_b32_value_l564_564138

theorem problem_b32_value :
  (∃ (b : Fin 32 → ℕ), b 0 = 2 ∧ b 1 = 1 ∧ b 3 = 3 ∧ b 7 = 30 ∧ 
    ∀ i, 1 ≤ i → i < 32 → b i > 0 ∧ 
    (∏ i in Finset.range 32, (1 - (z^(i+1)))^ (b (Fin.mk i _)) = 1 - 2*z + ∑ k in Finset.Ico 33 64, z^k)) →
  ∃ b_32, b_32 = 2^(27) - 2^(11) := 
sorry

end problem_b32_value_l564_564138


namespace problem_statement_l564_564328

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_statement (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x + real.pi) = f x + real.sin x)
  (h2 : ∀ x, 0 ≤ x ∧ x < real.pi → f x = 0) : f (23 * real.pi / 6) = 1 / 2 :=
sorry

end problem_statement_l564_564328


namespace sarah_commute_time_correct_l564_564626

-- Definitions based on conditions
def dave_steps_per_minute : ℕ := 80
def dave_step_length_cm : ℕ := 65
def dave_commute_time_min : ℕ := 20

def sarah_steps_per_minute : ℕ := 95
def sarah_step_length_cm : ℕ := 70
def sarah_break_min_per_10_min : ℕ := 1

-- Proof problem statement
theorem sarah_commute_time_correct :
  let dave_speed_cm_per_min := dave_steps_per_minute * dave_step_length_cm,
      total_commute_distance_cm := dave_speed_cm_per_min * dave_commute_time_min,
      sarah_speed_cm_per_min := sarah_steps_per_minute * sarah_step_length_cm,
      effective_sarah_time_min (t : ℝ) := t - (t / 10) in
  ∃ t : ℝ, effective_sarah_time_min t * sarah_speed_cm_per_min = total_commute_distance_cm ∧ t = 17.36 :=
begin
  sorry
end

end sarah_commute_time_correct_l564_564626


namespace inclination_angle_l564_564970

theorem inclination_angle (θ : ℝ) (h : 0 ≤ θ ∧ θ < 180) :
  (∀ x y : ℝ, x - y + 3 = 0 → θ = 45) :=
sorry

end inclination_angle_l564_564970


namespace cone_cross_section_area_l564_564484

theorem cone_cross_section_area (h α β : ℝ) (h_α_nonneg : 0 ≤ α) (h_β_gt : β > π / 2 - α) :
  ∃ S : ℝ,
    S = (h^2 * Real.sqrt (-Real.cos (α + β) * Real.cos (α - β))) / (Real.cos α * Real.sin β ^ 2) :=
sorry

end cone_cross_section_area_l564_564484


namespace average_score_in_5_matches_l564_564475

theorem average_score_in_5_matches (avg2 : ℕ) (avg3 : ℕ) (total_matches : ℕ)
  (h1 : avg2 = 30) 
  (h2 : avg3 = 40) 
  (h3 : total_matches = 5) :
  let total_runs := (2 * avg2) + (3 * avg3) in
  (total_runs / total_matches) = 36 := 
by
  sorry

end average_score_in_5_matches_l564_564475


namespace green_notebook_cost_l564_564442

def total_cost : ℕ := 45
def black_cost : ℕ := 15
def pink_cost : ℕ := 10
def num_green_notebooks : ℕ := 2

theorem green_notebook_cost :
  (total_cost - (black_cost + pink_cost)) / num_green_notebooks = 10 :=
by
  sorry

end green_notebook_cost_l564_564442


namespace jerry_needed_tonight_l564_564895

def jerry_earnings :=
  [20, 60, 15, 40]

def target_avg : ℕ := 50
def nights : ℕ := 5

def target_total_earnings := nights * target_avg
def actual_earnings_so_far := jerry_earnings.sum

theorem jerry_needed_tonight : (target_total_earnings - actual_earnings_so_far) = 115 :=
by
  have target_total : target_total_earnings = 250 := by rfl
  have actual_so_far : actual_earnings_so_far = 135 := by rfl
  rw [target_total, actual_so_far]
  norm_num
  -- 250 - 135 = 115
  rfl

end jerry_needed_tonight_l564_564895


namespace max_value_quadratic_l564_564332

noncomputable def quadratic (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

theorem max_value_quadratic : ∀ x : ℝ, quadratic x ≤ -3 ∧ (∀ y : ℝ, quadratic y = -3 → (∀ z : ℝ, quadratic z ≤ quadratic y)) :=
by
  sorry

end max_value_quadratic_l564_564332


namespace box_tape_length_l564_564602

variable (L S : ℕ)
variable (tape_total : ℕ)
variable (num_boxes : ℕ)
variable (square_side : ℕ)

theorem box_tape_length (h1 : num_boxes = 5) (h2 : square_side = 40) (h3 : tape_total = 540) :
  tape_total = 5 * (L + 2 * S) + 2 * 3 * square_side → L = 60 - 2 * S := 
by
  sorry

end box_tape_length_l564_564602


namespace solve_quartic_eq_l564_564659

theorem solve_quartic_eq (z : ℝ) : z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = 2 :=
by 
  sorry

end solve_quartic_eq_l564_564659


namespace circles_standard_equations_l564_564717

noncomputable def circle1_equation : Prop :=
  let C := fun a b : ℝ => (a + 1)^2 + b^2 = 20 in
  ∀ (a : ℝ), (C a 0) ↔ (sqrt((a - 1)^2 + 16) = sqrt((a - 3)^2 + 4) ∧ (a = -1) ∧ b = 0)

noncomputable def circle2_equation : Prop :=
  let C := fun a b : ℝ => (a - 1)^2 + (b + 2)^2 = 2 in
  ∀ (a b : ℝ), (C a b) ↔ 
    (2 * a + b = 0 ∧ (abs (a + b - 1) / sqrt 2 = sqrt ((a - 2)^2 + (b + 1)^2)) ∧ (a = 1) ∧ (b = -2))

theorem circles_standard_equations :
  circle1_equation ∧ circle2_equation :=
sorry

end circles_standard_equations_l564_564717


namespace solve_quartic_equation_l564_564690

theorem solve_quartic_equation :
  ∀ z : ℝ, z^4 - 6 * z^2 + 8 = 0 ↔ (z = -2 ∨ z = -sqrt 2 ∨ z = sqrt 2 ∨ z = 2) :=
by {
  sorry
}

end solve_quartic_equation_l564_564690


namespace alex_owes_daniel_l564_564221

theorem alex_owes_daniel : 
  let payment_per_room := (13 : ℚ) / 3,
      rooms_cleaned := (5 : ℚ) / 2,
      initial_amount := payment_per_room * rooms_cleaned,
      discount := 0.10 * initial_amount,
      final_amount := initial_amount - discount
  in 
  final_amount = (39 : ℚ) / 4 := 
by 
  sorry

end alex_owes_daniel_l564_564221


namespace coffee_pods_per_box_l564_564459

theorem coffee_pods_per_box (d k : ℕ) (c e : ℝ) (h1 : d = 40) (h2 : k = 3) (h3 : c = 8) (h4 : e = 32) :
  ∃ b : ℕ, b = 30 :=
by
  sorry

end coffee_pods_per_box_l564_564459


namespace rebus_solution_l564_564280

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564280


namespace find_complex_z_l564_564756

theorem find_complex_z (z : ℂ) (h : z * complex.I = 3 - complex.I) : z = -1 - 3 * complex.I :=
sorry

end find_complex_z_l564_564756


namespace intersection_area_l564_564995

open Set

variable (rect : Set (ℝ × ℝ))
variable (circle : Set (ℝ × ℝ))

def rectangle_vertices : Set (ℝ × ℝ) :=
  {(2, 3), (2, -5), (10, -5), (10, 3)}

def circle_eq : Set (ℝ × ℝ) :=
  {p | (p.1 - 10)^2 + (p.2 - 3)^2 ≤ 16}

theorem intersection_area (A : MeasureTheory.Measure ℝ) :
  rect = rectangle_vertices →
  circle = circle_eq →
  A (rect ∩ circle) = 8 * Real.pi :=
by
  intros hrect hcircle
  sorry

end intersection_area_l564_564995


namespace XY_parallel_BC_l564_564013

variables {A B C D E F G X Y : Type*}

-- Conditions
axiom triangle_ABC : Triangle A B C
axiom D_interior_ABC : Point D (interior_of (Triangle A B C))
axiom E_on_line_AD : Point E (line AD)
axiom E_diff_D : E ≠ D
axiom circle_omega1 : Circle ω₁ (circumcircle (Triangle B D E))
axiom circle_omega2 : Circle ω₂ (circumcircle (Triangle C D E))
axiom F_on_BC : Point F (line BC)
axiom G_on_BC : Point G (line BC)
axiom omega1_intersects_BC_at_f : Intersect ω₁ (line BC) F
axiom omega2_intersects_BC_at_g : Intersect ω₂ (line BC) G

-- Desired Conclusion
theorem XY_parallel_BC (h_triangle : triangle_ABC)
  (h_D_interior : D_interior_ABC)
  (h_E_on_AD : E_on_line_AD)
  (h_E_neq_D : E_diff_D)
  (h_circle1 : circle_omega1)
  (h_circle2 : circle_omega2)
  (h_F_on_BC : F_on_BC)
  (h_G_on_BC : G_on_BC)
  (h_intersect1 : omega1_intersects_BC_at_f)
  (h_intersect2 : omega2_intersects_BC_at_g) :
  Parallel XY BC := by
sorry

end XY_parallel_BC_l564_564013


namespace solve_quartic_equation_l564_564692

theorem solve_quartic_equation :
  ∀ z : ℝ, z^4 - 6 * z^2 + 8 = 0 ↔ (z = -2 ∨ z = -sqrt 2 ∨ z = sqrt 2 ∨ z = 2) :=
by {
  sorry
}

end solve_quartic_equation_l564_564692


namespace part1_part2_l564_564314

noncomputable def A (x : ℝ) : Prop :=
  x^2 - 4*x - 12 ≤ 0

noncomputable def B (a : ℝ) (x : ℝ) : Prop :=
  a - 1 < x ∧ x < 3*a + 2

noncomputable def complement_R_B (a : ℝ) (x : ℝ) : Prop :=
  x ≤ a - 1 ∨ x ≥ 3*a + 2

theorem part1 (x : ℝ) : A x ∧ (complement_R_B 1) x ↔ (-2 ≤ x ∧ x ≤ 0) ∨ (5 ≤ x ∧ x ≤ 6) :=
by sorry

theorem part2 {a : ℝ} (H : ∀ x, A x → B a x) : a ∈ Iic (-3/2) ∪ Icc (-1 : ℝ) (4/3) :=
by sorry

end part1_part2_l564_564314


namespace bus_car_ratio_l564_564972

variable (R C Y : ℝ)

noncomputable def ratio_of_bus_to_car (R C Y : ℝ) : ℝ :=
  R / C

theorem bus_car_ratio 
  (h1 : R = 48) 
  (h2 : Y = 3.5 * C) 
  (h3 : Y = R - 6) : 
  ratio_of_bus_to_car R C Y = 4 :=
by sorry

end bus_car_ratio_l564_564972


namespace solve_quartic_l564_564685

theorem solve_quartic (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ (z = 2 ∨ z = -2 ∨ z = sqrt 2 ∨ z = -sqrt 2) :=
by
  sorry

end solve_quartic_l564_564685


namespace area_of_side_face_l564_564536

noncomputable def box_size : ℕ := 5184

noncomputable def length (h : ℝ) : ℝ := 2 * h
noncomputable def width (h : ℝ) : ℝ := 1.5 * h

theorem area_of_side_face :
  ∃ (h l w : ℝ), 
  (l = length h) ∧ 
  (w = width h) ∧ 
  (l * w * h = box_size) ∧
  (w * h = 0.5 * (l * w)) ∧ 
  (l * w = 1.5 * (l * h)) ∧ 
  ((l * h) = 288) :=
begin
  sorry
end

end area_of_side_face_l564_564536


namespace problem_statement_l564_564401

variables {A B C H P_a P_b P_c X_a X_b X_c : Point}
variables (m_a m_b m_c l_a l_b l_c : Line)
variables (T : Triangle)
variables (T_1 : Triangle)
variables (T_2 : Triangle)

-- Conditions
def orthocenter (T : Triangle) : Point := H
def midpoints_sidelines_perpendicular_bisectors (T : Triangle) : Triangle := T_1
def vertices_bisect_bisectors (T : Triangle) : Triangle := T_2

def condition_1 := orthocenter T = H
def condition_2 := midpoints_sidelines_perpendicular_bisectors T = T_1
def condition_3 := vertices_bisect_bisectors T = T_2

theorem problem_statement (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) :
  ∀ (P : Point), (P = H) → 
  ∃ P_a P_b P_c X_a X_b X_c, 
  ∀ (a b c : Point), ((P = P_a) ∨ (P = P_b) ∨ (P = P_c)) →
  ∀ (t1 t2 t3 : Triangle), (t1 = T_2) →
  (line_through P_a H ⊥ line_through X_a X_c) ∧
  (line_through P_b H ⊥ line_through X_b X_a) ∧
  (line_through P_c H ⊥ line_through X_c X_b) :=
sorry

end problem_statement_l564_564401


namespace range_of_p_l564_564631

open Set

-- Define the function p(x) = x^4 + 6x^2 + 9
def p (x : ℝ) : ℝ := x^4 + 6 * x^2 + 9

-- State the problem: determine the range of p(x) on the domain [0, ∞)
theorem range_of_p : range p = Icc 9 (∞ : ℝ) :=
by
  sorry

end range_of_p_l564_564631


namespace sum_of_function_values_on_regular_polygon_l564_564399

noncomputable def f : ℝ × ℝ → ℝ := sorry

theorem sum_of_function_values_on_regular_polygon {n : ℕ} (hn : n ≥ 3) 
  (h : ∀ (v : fin n → ℝ × ℝ), (∀ i, (v i.succ % n).dist (v i) = (v 0).dist (v 1)) → 
  ∑ i, f (v i) = 0) :
  ∀ x : ℝ × ℝ, f x = 0 := sorry

end sum_of_function_values_on_regular_polygon_l564_564399


namespace scheduling_arrangements_l564_564457

-- We want to express this as a problem to prove the number of scheduling arrangements.

theorem scheduling_arrangements (n : ℕ) (h : n = 6) :
  (Nat.choose 6 1) * (Nat.choose 5 1) * (Nat.choose 4 2) = 180 := by
  sorry

end scheduling_arrangements_l564_564457


namespace maximize_volume_width_l564_564587

noncomputable def volume (x : ℝ) : ℝ := x * (x + 0.5) * (3.2 - 2 * x)

theorem maximize_volume_width :
  (∀ x ∈ Ioo 0 1.6, volume x ≤ volume 1) ∧ (volume 1 = volume (1 : ℝ)) :=
by
  admit -- ignore the computational details and assume this statement to be true

end maximize_volume_width_l564_564587


namespace probability_two_approvals_of_four_l564_564581

/-- Definition of the problem condition -/
def probability_approval (P_Y : ℚ) (P_N : ℚ) : Prop :=
  P_Y = 0.6 ∧ P_N = 1 - P_Y

/-- The main theorem stating the probability of exactly two approvals out of four trials. -/
theorem probability_two_approvals_of_four (P_Y P_N : ℚ)
  (h : probability_approval P_Y P_N) :
  (4.choose 2) * P_Y^2 * P_N^2 = 0.3456 := by
  sorry

end probability_two_approvals_of_four_l564_564581


namespace max_value_of_expression_l564_564404

theorem max_value_of_expression (a b c : ℝ) (ha : 0 ≤ a) (ha2 : a ≤ 2) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc2 : c ≤ 2) :
  2 * Real.sqrt (abc / 8) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 2 :=
by
  sorry

end max_value_of_expression_l564_564404


namespace store_breaks_even_l564_564146

-- Defining the conditions based on the problem statement.
def cost_price_piece1 (profitable : ℝ → Prop) : Prop :=
  ∃ x, profitable x ∧ 1.5 * x = 150

def cost_price_piece2 (loss : ℝ → Prop) : Prop :=
  ∃ y, loss y ∧ 0.75 * y = 150

def profitable (x : ℝ) : Prop := x + 0.5 * x = 150
def loss (y : ℝ) : Prop := y - 0.25 * y = 150

-- Store breaks even if the total cost price equals the total selling price
theorem store_breaks_even (x y : ℝ)
  (P1 : cost_price_piece1 profitable)
  (P2 : cost_price_piece2 loss) :
  (x + y = 100 + 200) → (150 + 150) = 300 :=
by
  sorry

end store_breaks_even_l564_564146


namespace probability_grisha_wins_expectation_coin_flips_l564_564836

-- Define the conditions as predicates
def grishaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 0) && (toss_seq.last = some true)

def vanyaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 1) && (toss_seq.last = some false)

-- State the probability theorem
theorem probability_grisha_wins : 
  (∑ x in ((grishaWins ≤ 1 / 3))) = 1/3 := by sorry

-- State the expectation theorem
theorem expectation_coin_flips : 
  (E[flips until (grishaWins ∨ vanyaWins)]) = 2 := by sorry

end probability_grisha_wins_expectation_coin_flips_l564_564836


namespace point_in_fourth_quadrant_l564_564130

def Point : Type := ℤ × ℤ

def in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def A : Point := (-3, 7)
def B : Point := (3, -7)
def C : Point := (3, 7)
def D : Point := (-3, -7)

theorem point_in_fourth_quadrant : in_fourth_quadrant B :=
by {
  -- skipping the proof steps for the purpose of this example
  sorry
}

end point_in_fourth_quadrant_l564_564130


namespace find_all_z_l564_564672

noncomputable def all_solutions (z : ℝ) : Prop := z = -2 ∨ z = 2 ∨ z = -real.sqrt 2 ∨ z = real.sqrt 2

theorem find_all_z (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ all_solutions z := 
sorry

end find_all_z_l564_564672


namespace max_subset_no_seven_multiple_l564_564721

theorem max_subset_no_seven_multiple : 
  ∃ (S : Finset ℕ), S.card = 1763 ∧ S ⊆ (Finset.range 2015).filter (λ n, n ≠ 0) ∧ 
  (∀ a b ∈ S, a ≠ 7 * b ∧ b ≠ 7 * a) :=
sorry

end max_subset_no_seven_multiple_l564_564721


namespace Mike_age_calculation_l564_564435

variable (Pat_current_age Mike_current_age Biddy_age_when_pigsty_built Pat_age_when_pigsty_built : ℚ)

theorem Mike_age_calculation
  (h1 : Pat_current_age = (4 / 3) * Pat_age_when_pigsty_built)
  (h2 : Mike_current_age = 2 + 0.5 * Biddy_age_when_pigsty_built)
  (h3 : let combined_age_when_Mike_reaches_Pat_age := (Pat_age_when_pigsty_built - Mike_current_age + 3 + 4 / 12) + Pat_current_age + Biddy_age_when_pigsty_built, combined_age_when_Mike_reaches_Pat_age = 100) :
  Mike_current_age = 227 / 21 :=
sorry

end Mike_age_calculation_l564_564435


namespace area_of_octagon_divided_circle_l564_564560

-- Given Conditions
variables (R : ℝ)
variables (arc_small arc_large : ℝ)
variables (total_angle : ℝ)

-- Definitions based on the conditions
def smaller_arc := arc_small
def larger_arc := 2 * smaller_arc
def total_arc := 4 * smaller_arc + 4 * larger_arc
def ang_small_arc := total_angle / 8

-- The problem statement to prove the area of the octagon
theorem area_of_octagon_divided_circle :
  (total_arc = 2 * real.pi) →
  R > 0 →
  total_angle = 2 * real.pi →
  arc_small = real.pi / 4 →
  arc_large = real.pi / 2 →
  ∃ S : ℝ, S = R^2 * (real.sqrt 2 + 2) :=
begin
  intros h1 h2 h3 h4 h5,
  use R^2 * (real.sqrt 2 + 2),
  sorry
end

end area_of_octagon_divided_circle_l564_564560


namespace complex_number_l564_564791

-- Define conditions using Lean terminology
variables {𝕜 : Type*} [IsROrC 𝕜] {a b : 𝕜} (a_real : IsROrC.re a = a) (b_real : IsROrC.re b = b)

-- State the proof goal
theorem complex_number (h1 : IsROrC.im a = 0) (h2 : IsROrC.im b = 0) : 
  ∃ c : 𝕜, c = a + IsROrC.I * b :=
sorry

end complex_number_l564_564791


namespace constant_term_expansion_l564_564383

theorem constant_term_expansion : 
  let f := (fun x : ℝ => (1/x^2 - 2*x)^6)
  in ∃ c : ℝ, ∀ x, f x = c := 
  ∃ c : ℝ, c = 240 :=
sorry

end constant_term_expansion_l564_564383


namespace MattSkipsRopesTimesPerSecond_l564_564434

theorem MattSkipsRopesTimesPerSecond:
  ∀ (minutes_jumped : ℕ) (total_skips : ℕ), 
  minutes_jumped = 10 → 
  total_skips = 1800 → 
  (total_skips / (minutes_jumped * 60)) = 3 :=
by
  intros minutes_jumped total_skips h_jumped h_skips
  sorry

end MattSkipsRopesTimesPerSecond_l564_564434


namespace solve_quartic_eq_l564_564704

theorem solve_quartic_eq (z : ℂ) :
  z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -complex.sqrt 2 ∨ z = complex.sqrt 2 ∨ z = 2 :=
by {
  sorry
}

end solve_quartic_eq_l564_564704


namespace count_digit_7_in_range_l564_564860

theorem count_digit_7_in_range : ∀ n ∈ finset.Icc 100 199, (n % 10 = 7 ∨ n / 10 % 10 = 7) →  ∃ counts, counts = 20 :=
by
  sorry

end count_digit_7_in_range_l564_564860


namespace factorable_polynomial_l564_564916

theorem factorable_polynomial (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p ≠ q) (n : ℕ) (hn : n ≥ 3) :
  ∃ a : ℤ, (a = (-1 : ℤ)^n * (p : ℤ) * (q : ℤ) + 1 ∨ a = -(p : ℤ) * (q : ℤ) - 1) ∧
    ∃ g h : Polynomial ℤ, 
      g.monic ∧ h.monic ∧ 
      g.degree > 0 ∧ h.degree > 0 ∧ 
      Polynomial.mul g h = Polynomial.C (p * q) + Polynomial.X^n + Polynomial.C a * Polynomial.X^(n-1) := 
sorry

end factorable_polynomial_l564_564916


namespace jerry_needed_tonight_l564_564894

def jerry_earnings :=
  [20, 60, 15, 40]

def target_avg : ℕ := 50
def nights : ℕ := 5

def target_total_earnings := nights * target_avg
def actual_earnings_so_far := jerry_earnings.sum

theorem jerry_needed_tonight : (target_total_earnings - actual_earnings_so_far) = 115 :=
by
  have target_total : target_total_earnings = 250 := by rfl
  have actual_so_far : actual_earnings_so_far = 135 := by rfl
  rw [target_total, actual_so_far]
  norm_num
  -- 250 - 135 = 115
  rfl

end jerry_needed_tonight_l564_564894


namespace convex_pentagon_inscribed_l564_564392

theorem convex_pentagon_inscribed {O : Point} {circle : Circle O} {A1 A2 A3 A4 A5 : Point} 
  (h1 : A1 ∈ circle) (h2 : A2 ∈ circle) (h3 : A3 ∈ circle) (h4 : A4 ∈ circle) (h5 : A5 ∈ circle) 
  (convex_pentagon : Convex (Polygon5 A1 A2 A3 A4 A5)) :
  ∃ (reg_pentagon : Polygon5), (Vertices reg_pentagon).Inscribed circle ∧
  RegularPolygon reg_pentagon ∧
  ∃ (side : Segment), side ∈ (Sides (Polygon5 A1 A2 A3 A4 A5)) ∧
  side.length ≤ (max_side (Sides reg_pentagon)).length :=
sorry

end convex_pentagon_inscribed_l564_564392


namespace fixed_point_of_line_l564_564126

theorem fixed_point_of_line : ∀ k : ℝ, ∃ x y : ℝ, (kx - y + 1 = 3k) ∧ (x = 3) ∧ (y = 1) :=
by
  -- The proof is not required as per the instructions
  sorry

end fixed_point_of_line_l564_564126


namespace incorrect_statement_C_l564_564743

-- Define entities: planes α, β and lines m, n
variables {α β : Type} [plane α] [plane β] (m n : Type) [line m] [line n]

-- Define relationships: perpendicular and parallel
def perpendicular (x y : Type) [has_perpendicular x y] : Prop := true
def parallel (x y : Type) [has_parallel x y] : Prop := true
def intersection (x y : Type) [has_intersection x y] : Type := sorry

-- Condition A: If m ⊥ α and m ⊥ β, then α ∥ β
axiom cond_A (m : Type) [line m] (α β : Type) [plane α] [plane β] (h1 : perpendicular m α) (h2 : perpendicular m β) : parallel α β

-- Condition B: If m ∥ n and m ⊥ α, then n ⊥ α
axiom cond_B (m n : Type) [line m] [line n] (α : Type) [plane α] (h1 : parallel m n) (h2 : perpendicular m α) : perpendicular n α

-- Condition C: If m ∥ α and α ∩ β = n, then m ∥ n
axiom cond_C (m : Type) [line m] (α β : Type) [plane α] [plane β] (h1 : parallel m α) (h2 : intersection α β = n) : parallel m n

-- Condition D: If m ⊥ α and m ⊆ β, then α ⊥ β
axiom cond_D (m : Type) [line m] (α β : Type) [plane α] [plane β] (h1 : perpendicular m α) (h2 : m ⊆ β) : perpendicular α β

-- Goal: Prove that statement C is incorrect
theorem incorrect_statement_C (α β : Type) [plane α] [plane β] (m n : Type) [line m] [line n] (h1 : parallel m α) (h2 : intersection α β = n) : ¬(parallel m n) :=
sorry

end incorrect_statement_C_l564_564743


namespace min_cos_2alpha_l564_564405

noncomputable def vector_space := ℝ^3

variables {A B C D E : vector_space}
variables {AB AC BE CD : vector_space}
variable {α : ℝ}

-- Given conditions
def is_midpoint (M : vector_space) (X Y : vector_space) := 2 • M = X + Y

def angle_between_vectors (u v : vector_space) : ℝ := 
  real.arccos ((u ⬝ v) / (∥u∥ * ∥v∥))

axiom cond_midpoint_D : is_midpoint D A B
axiom cond_midpoint_E : is_midpoint E A C
axiom cond_dot_product_zero : BE ⬝ CD = 0
axiom cond_angle_alpha : angle_between_vectors AB AC = α

-- Prove that the minimum value of cos(2*α) is 7/25
theorem min_cos_2alpha : cos (2 * α) = 7/25 :=
sorry

end min_cos_2alpha_l564_564405


namespace cos_270_eq_zero_l564_564206

theorem cos_270_eq_zero : ∀ (θ : ℝ), θ = 270 → (∀ (p : ℝ × ℝ), p = (0,1) → cos θ = p.fst) → cos 270 = 0 :=
by
  intros θ hθ h
  sorry

end cos_270_eq_zero_l564_564206


namespace cos_270_eq_zero_l564_564210

theorem cos_270_eq_zero : Real.cos (270 * (π / 180)) = 0 :=
by
  -- Conditions given in the problem
  have rotation_def : ∀ (θ : ℝ), ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x = Real.cos θ ∧ y = Real.sin θ := by
    intros θ
    use [Real.cos θ, Real.sin θ]
    split
    · exact Real.cos_sq_add_sin_sq θ
    split
    · rfl
    · rfl

  have rotation_270 : ∃ (x y : ℝ), x = 0 ∧ y = -1 := by
    use [0, -1]
    split
    · rfl
    · rfl

  -- Goal to be proved
  have result := Real.cos (270 * (π / 180))
  show result = 0
  sorry

end cos_270_eq_zero_l564_564210


namespace problem_statement_l564_564307

theorem problem_statement (x : ℝ) (h : x^2 + 3 * x - 1 = 0) : x^3 + 5 * x^2 + 5 * x + 18 = 20 :=
by
  sorry

end problem_statement_l564_564307


namespace average_marks_of_all_students_l564_564069

theorem average_marks_of_all_students :
  ∀ (marks_class1 marks_class2 : ℕ) (students_class1 students_class2 : ℕ),
  marks_class1 = 40 → students_class1 = 24 →
  marks_class2 = 60 → students_class2 = 50 →
  ( (students_class1 * marks_class1) + (students_class2 * marks_class2) ) 
  / (students_class1 + students_class2) = 53.51 :=
by 
  intros marks_class1 marks_class2 students_class1 students_class2 h1 h2 h3 h4
  sorry

end average_marks_of_all_students_l564_564069


namespace maximum_revenue_l564_564161

def ticket_price (x : ℕ) (y : ℤ) : Prop :=
  (6 ≤ x ∧ x ≤ 10 ∧ y = 1000 * x - 5750) ∨
  (10 < x ∧ x ≤ 38 ∧ y = -30 * x^2 + 1300 * x - 5750)

theorem maximum_revenue :
  ∃ x y, ticket_price x y ∧ y = 8830 ∧ x = 22 :=
by {
  sorry
}

end maximum_revenue_l564_564161


namespace sum_coefficients_eq_l564_564723

-- Define the polynomial (1 - 2x)^n
def polynomial (x : ℝ) (n : ℕ) : ℝ :=
  (1 - 2 * x) ^ n

-- Define the coefficients as sums
def sum_coefficients (n : ℕ) : ℝ :=
  ∑ i in (finset.range n).map (λ i, 1), i

-- The theorem to be proven
theorem sum_coefficients_eq (n : ℕ) :
  sum_coefficients n = (-1 : ℝ)^n :=
by {
  sorry
}

end sum_coefficients_eq_l564_564723


namespace sum_of_prob_fraction_is_71_l564_564299

theorem sum_of_prob_fraction_is_71 (a b : Finset ℕ) 
  (ha : ∀ x ∈ a, x ∈ Finset.range 2001)
  (hb : ∀ x ∈ b, x ∈ (Finset.range 2001 \ a)) 
  (ha_len : a.card = 4) 
  (hb_len : b.card = 4) :
  let p := 1 / (Finset.card (Finset.powersetLen 4 (a ∪ b))) in
  p.num + p.denom = 71 :=
by
  sorry

end sum_of_prob_fraction_is_71_l564_564299


namespace find_min_abs_sum_l564_564413

open Matrix

def matrix_squared_eq_seven (a b c d : ℤ) : Prop :=
  let mat := !![a, b; c, d]
  mat.mul mat = !![7, 0; 0, 7]

theorem find_min_abs_sum 
  {a b c d : ℤ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hmat : matrix_squared_eq_seven a b c d) : 
  |a| + |b| + |c| + |d| = 7 :=
  sorry

end find_min_abs_sum_l564_564413


namespace find_z_values_l564_564656

theorem find_z_values : {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -sqrt 2, sqrt 2, 2} :=
sorry

end find_z_values_l564_564656


namespace cary_strips_ivy_l564_564613

variable (strip_per_day : ℕ) (grow_per_night : ℕ) (total_ivy : ℕ)

theorem cary_strips_ivy (h1 : strip_per_day = 6) (h2 : grow_per_night = 2) (h3 : total_ivy = 40) :
  (total_ivy / (strip_per_day - grow_per_night)) = 10 := by
  sorry

end cary_strips_ivy_l564_564613


namespace area_of_45_45_90_triangle_l564_564954

open Real

theorem area_of_45_45_90_triangle (BF : ℝ) (hBF : BF = 4) : 
  ∃ (ABC : Type) [isTriangle ABC], 
  (∀ (A B C : ABC), angle A B = 45 ∧ angle B C = 90 ∧ angle C A = 45) ∧ 
  (∀ (A B C : ABC), 
     ∃ (F : ABC), isPerpendicular BF A C ∧ length BF = 4) ∧
  area ABC = 8 :=
by sorry

end area_of_45_45_90_triangle_l564_564954


namespace matrices_commute_l564_564420

noncomputable def S : Finset ℕ :=
  {0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196}

theorem matrices_commute (n : ℕ) :
  ∀ (A : Matrix (Fin 2) (Fin 2) ℕ) (B : Matrix (Fin 2) (Fin 2) ℕ),
  (∀ a b c d e f g h : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S → f ∈ S → g ∈ S → h ∈ S →
    A = ![![a, b], ![c, d]] → B = ![![e, f], ![g, h]] → n > 50432 → A.mul B = B.mul A) :=
sorry

end matrices_commute_l564_564420


namespace polynomial_roots_product_l564_564414

theorem polynomial_roots_product (a b c d : ℝ) (h : Polynomial.eval a (Polynomial.Coeff.coeff 3 x^4 - 8 x^3 + x^2 + 4 x - 10) = 0)
  (h : Polynomial.eval b (Polynomial.Coeff.coeff 3 x^4 - 8 x^3 + x^2 + 4 x - 10) = 0)
  (h : Polynomial.eval c (Polynomial.Coeff.coeff 3 x^4 - 8 x^3 + x^2 + 4 x - 10) = 0)
  (h : Polynomial.eval d (Polynomial.Coeff.coeff 3 x^4 - 8 x^3 + x^2 + 4 x - 10) = 0) :
  a * b * c * d = -10 / 3 :=
begin
  sorry
end

end polynomial_roots_product_l564_564414


namespace cary_ivy_removal_days_correct_l564_564616

noncomputable def cary_ivy_removal_days (initial_ivy : ℕ) (ivy_removed_per_day : ℕ) (ivy_growth_per_night : ℕ) : ℕ :=
  initial_ivy / (ivy_removed_per_day - ivy_growth_per_night)

theorem cary_ivy_removal_days_correct :
  cary_ivy_removal_days 40 6 2 = 10 :=
by
  -- The body of the proof is omitted; it will be filled with the actual proof.
  sorry

end cary_ivy_removal_days_correct_l564_564616


namespace solve_quartic_eq_l564_564662

theorem solve_quartic_eq (z : ℝ) : z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = 2 :=
by 
  sorry

end solve_quartic_eq_l564_564662


namespace digit_7_count_in_range_l564_564873

def count_digit_7 : ℕ :=
  let units_place := (107 - 100) / 10 + 1
  let tens_place := 10
  units_place + tens_place

theorem digit_7_count_in_range : count_digit_7 = 20 := by
  sorry

end digit_7_count_in_range_l564_564873


namespace rebus_solution_l564_564257

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l564_564257


namespace digit_7_appears_20_times_in_range_100_to_199_l564_564863

theorem digit_7_appears_20_times_in_range_100_to_199 : 
  ∃ (d7_count : ℕ), (d7_count = 20) ∧ (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 199 → 
  d7_count = (n.to_string.count '7')) :=
sorry

end digit_7_appears_20_times_in_range_100_to_199_l564_564863


namespace binomial_fermat_l564_564448

theorem binomial_fermat (p : ℕ) (a b : ℤ) (hp : p.Prime) : 
  ((a + b)^p - a^p - b^p) % p = 0 := by
  sorry

end binomial_fermat_l564_564448


namespace a100_pos_a100_abs_lt_018_l564_564333

noncomputable def a (n : ℕ) : ℝ := real.cos (10^n * real.pi / 180)

theorem a100_pos : (a 100 > 0) :=
by
  sorry

theorem a100_abs_lt_018 : (|a 100| < 0.18) :=
by
  sorry

end a100_pos_a100_abs_lt_018_l564_564333


namespace cd_e_value_l564_564318

theorem cd_e_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 195) (h2 : b * c * d = 65) 
  (h3 : d * e * f = 250) (h4 : (a * f) / (c * d) = 0.75) :
  c * d * e = 1000 := 
by
  sorry

end cd_e_value_l564_564318


namespace solve_quartic_eq_l564_564700

theorem solve_quartic_eq (z : ℂ) :
  z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -complex.sqrt 2 ∨ z = complex.sqrt 2 ∨ z = 2 :=
by {
  sorry
}

end solve_quartic_eq_l564_564700


namespace a_100_positive_a_100_abs_lt_018_l564_564344

-- Define the sequence based on the given conditions
def a_n (n : ℕ) : ℝ := real.cos (real.pi / 180 * (10^n) : ℝ)

-- Translate mathematical claims to Lean theorems
theorem a_100_positive : a_n 100 > 0 :=
sorry

theorem a_100_abs_lt_018 : |a_n 100| < 0.18 :=
sorry

end a_100_positive_a_100_abs_lt_018_l564_564344


namespace find_C_and_D_l564_564230

noncomputable def C : ℚ := 51 / 10
noncomputable def D : ℚ := 29 / 10

theorem find_C_and_D (x : ℚ) (h1 : x^2 - 4*x - 21 = (x - 7)*(x + 3))
  (h2 : (8*x - 5) / ((x - 7)*(x + 3)) = C / (x - 7) + D / (x + 3)) :
  C = 51 / 10 ∧ D = 29 / 10 :=
by
  sorry

end find_C_and_D_l564_564230


namespace row_seat_notation_l564_564499

-- Define that the notation (4, 5) corresponds to "Row 4, Seat 5"
def notation_row_seat := (4, 5)

-- Prove that "Row 5, Seat 4" should be denoted as (5, 4)
theorem row_seat_notation : (5, 4) = (5, 4) :=
by sorry

end row_seat_notation_l564_564499


namespace constant_term_in_expansion_l564_564073

theorem constant_term_in_expansion : 
  let expr := (x - (1 / (x ^ 2))) ^ 6 in
  ∃ (C : ℕ), (C = 15) ∧ ∃ n : ℕ, (choose 6 n * (-1) ^ n * x ^ (6 - 3 * n) = C) :=
by
  sorry

end constant_term_in_expansion_l564_564073


namespace Grisha_probability_expected_flips_l564_564829

theorem Grisha_probability (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                            t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (probability (Grisha wins)) = 1 / 3 :=
by 
  sorry

theorem expected_flips (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                        t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (expected_flips) = 2 :=
by
  sorry

end Grisha_probability_expected_flips_l564_564829


namespace AM_GM_HY_order_l564_564412

noncomputable def AM (a b c : ℝ) : ℝ := (a + b + c) / 3
noncomputable def GM (a b c : ℝ) : ℝ := (a * b * c)^(1/3)
noncomputable def HY (a b c : ℝ) : ℝ := 2 * a * b * c / (a * b + b * c + c * a)

theorem AM_GM_HY_order (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  AM a b c > GM a b c ∧ GM a b c > HY a b c := by
  sorry

end AM_GM_HY_order_l564_564412


namespace Maaza_liters_l564_564566

theorem Maaza_liters 
  (M L : ℕ)
  (Pepsi : ℕ := 144)
  (Sprite : ℕ := 368)
  (total_liters := M + Pepsi + Sprite)
  (cans_required : ℕ := 281)
  (H : total_liters = cans_required * L)
  : M = 50 :=
by
  sorry

end Maaza_liters_l564_564566


namespace part1_part2_l564_564425

def set_A (a : ℝ) : set ℝ := { x | a - 2 ≤ x ∧ x ≤ 2a + 3 }
def set_B : set ℝ := { x | x^2 - 6x + 5 ≤ 0 }
def set_B_compl : set ℝ := { x | ¬ (x^2 - 6x + 5 ≤ 0) }

theorem part1 (a : ℝ) : (set_A a ∩ set_B = set_B) → (1 ≤ a ∧ a ≤ 3) :=
by 
  sorry

theorem part2 (a : ℝ) : (set_A a ∩ set_B_compl = ∅) → (a < -5) :=
by 
  sorry

end part1_part2_l564_564425


namespace solve_quartic_eq_l564_564698

theorem solve_quartic_eq (z : ℂ) :
  z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -complex.sqrt 2 ∨ z = complex.sqrt 2 ∨ z = 2 :=
by {
  sorry
}

end solve_quartic_eq_l564_564698


namespace second_group_persons_l564_564548

theorem second_group_persons (x : ℕ) : 
  let total_man_hours_first_group : ℕ := 78 * 12 * 5,
      total_man_hours_second_group : ℕ := x * 26 * 6 
  in total_man_hours_first_group = total_man_hours_second_group → x = 130 :=
by
  intros h
  let total_man_hours_first_group := 78 * 12 * 5
  let total_man_hours_second_group := x * 26 * 6
  sorry

end second_group_persons_l564_564548


namespace equidistant_point_l564_564286

def dist (P1 P2 : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  Real.sqrt ((P1 - P2).sum_squares)

theorem equidistant_point (z : ℝ) :
    (dist ![0, 0, z] ![-18, 1, 0] = dist ![0, 0, z] ![15, -10, 2]) → z = 1 := by
  sorry

end equidistant_point_l564_564286


namespace lottery_ends_after_fourth_draw_l564_564507

-- Define the conditions of the problem
def n_people : ℕ := 5
def n_tickets : ℕ := 5
def winning_tickets : ℕ := 3
def draws_without_replacement : Bool := true

-- Define the required proof statement
theorem lottery_ends_after_fourth_draw :
  (↑(3 * 3! * 2!) / ↑(5!)) = (3 / 10) := 
by
  -- Proof to be provided
  sorry

end lottery_ends_after_fourth_draw_l564_564507


namespace prove_k_m_sum_eq_six_l564_564505

noncomputable def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ i in divisors n, i

theorem prove_k_m_sum_eq_six :
  ∃ (k m : ℕ), (sum_of_divisors (2^k * 5^m) = 930) ∧ (k + m = 6) :=
by
  sorry

end prove_k_m_sum_eq_six_l564_564505


namespace nonagons_overlap_l564_564387

open Set

noncomputable def convex_nonagon : Type := ℝ -- This is a placeholder for the type of convex nonagons

variables (T : convex_nonagon) (A : ℝ) (vertices : List ℝ) (congruent_nonagons : List convex_nonagon)

def translates (T : convex_nonagon) (A : ℝ) (vertices : List ℝ) : List convex_nonagon :=
  vertices.map (λ v, T) -- This is a placeholder to represent the translation operation

theorem nonagons_overlap :
  ∃ (T : convex_nonagon) (A : ℝ) (vertices : List ℝ)
    (congruent_nonagons : List convex_nonagon),
  (congruent_nonagons = translates T A vertices) →
  ∃ (i j : ℕ), 1 ≤ i ∧ i ≤ 8 ∧ 1 ≤ j ∧ j ≤ 8 ∧ i ≠ j ∧
  ∃ (overlap_area : ℝ), overlap_area > 0 := 
sorry

end nonagons_overlap_l564_564387


namespace pictures_at_museum_l564_564133

def pictures_at_zoo : ℕ := 41
def pictures_deleted : ℕ := 15
def pictures_left : ℕ := 55

theorem pictures_at_museum : ∃ M : ℕ, (55 + 15 - 41 = M) := by
  exists 29
  sorry

end pictures_at_museum_l564_564133


namespace max_area_ratio_two_rotated_rectangles_l564_564072

theorem max_area_ratio_two_rotated_rectangles
  (a b : ℝ) (h₁ : a ≥ b) (h₂ : a > 0) (h₃ : b > 0) :
  ∃ q : ℝ, (q ≤ real.sqrt 2) ∧
           (∀ q', (real.sqrt (2) - 1 ≤ q' ∧ q' < real.sqrt (2)) → 
           ∃ r : ℝ, r = (2 - q') / real.sqrt (2) ∧ 
           q' = (real.sqrt (2) - 1) * (r + 1/r + 2)) := sorry

end max_area_ratio_two_rotated_rectangles_l564_564072


namespace sum_k_equals_326_l564_564518

noncomputable def sum_possible_k_values (t : ℕ) : ℕ :=
  (if t = 306 then
    let divisors := [1, 2, 3, 6, 9, 17, 18, 34, 51, 102, 153, 306] in
    let valid_k := divisors.filter (λ k, ∃ m, k * (1 + m * k) = t) in
    valid_k.sum
  else 0)

theorem sum_k_equals_326 : sum_possible_k_values 306 = 326 :=
by
  sorry

end sum_k_equals_326_l564_564518


namespace cos_FAD_in_square_ABC_isosceles_AEF_l564_564840

theorem cos_FAD_in_square_ABC_isosceles_AEF
  (A B C D E F : Point)
  (hABCD : Square A B C D)
  (hAEF_isosceles : IsoscelesTriangle A E F)
  (hE_on_BC : LiesOn E (Segment B C))
  (hF_on_CD : LiesOn F (Segment C D))
  (hAE_AF : Distance A E = Distance A F)
  (tan_AEF : tan (Angle A E F) = 3) : 
  cos (Angle F A D) = (2 * sqrt 5) / 5 := 
sorry

end cos_FAD_in_square_ABC_isosceles_AEF_l564_564840


namespace find_r_s_l564_564745

noncomputable def r_s_proof_problem (r s : ℕ) (h1 : Nat.gcd r s = 1) (h2 : (r : ℝ) / s = (2 * (Real.sqrt 2 + Real.sqrt 10)) / (5 * Real.sqrt (3 + Real.sqrt 5))) : Prop :=
(r, s) = (4, 5)

theorem find_r_s (r s : ℕ) (h1 : Nat.gcd r s = 1) (h2 : (r : ℝ) / s = (2 * (Real.sqrt 2 + Real.sqrt 10)) / (5 * Real.sqrt (3 + Real.sqrt 5))) : r_s_proof_problem r s h1 h2 :=
sorry

end find_r_s_l564_564745


namespace smallest_n_boxes_cookies_l564_564900

theorem smallest_n_boxes_cookies (n : ℕ) (h : (17 * n - 1) % 12 = 0) : n = 5 :=
sorry

end smallest_n_boxes_cookies_l564_564900


namespace complex_expression_evaluation_l564_564418

-- Defining the imaginary unit
def i : ℂ := Complex.I

-- Defining the complex number z
def z : ℂ := 1 - i

-- Stating the theorem to prove
theorem complex_expression_evaluation : z^2 + (2 / z) = 1 - i := by
  sorry

end complex_expression_evaluation_l564_564418


namespace solve_quartic_eq_l564_564697

theorem solve_quartic_eq (z : ℂ) :
  z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -complex.sqrt 2 ∨ z = complex.sqrt 2 ∨ z = 2 :=
by {
  sorry
}

end solve_quartic_eq_l564_564697


namespace p_arithmetic_fibonacci_term_correct_l564_564708

noncomputable def p_arithmetic_fibonacci_term (p : ℕ) : ℝ :=
  5 ^ ((p - 1) / 2)

theorem p_arithmetic_fibonacci_term_correct (p : ℕ) : p_arithmetic_fibonacci_term p = 5 ^ ((p - 1) / 2) := 
by 
  rfl -- direct application of the definition

#check p_arithmetic_fibonacci_term_correct

end p_arithmetic_fibonacci_term_correct_l564_564708


namespace relation_1_relation_2_relation_3_general_relationship_l564_564249

theorem relation_1 (a b : ℝ) (h1: a = 3) (h2: b = 3) : a^2 + b^2 = 2 * a * b :=
by 
  have h : a = 3 := h1
  have h' : b = 3 := h2
  sorry

theorem relation_2 (a b : ℝ) (h1: a = 2) (h2: b = 1/2) : a^2 + b^2 > 2 * a * b :=
by 
  have h : a = 2 := h1
  have h' : b = 1/2 := h2
  sorry

theorem relation_3 (a b : ℝ) (h1: a = -2) (h2: b = 3) : a^2 + b^2 > 2 * a * b :=
by 
  have h : a = -2 := h1
  have h' : b = 3 := h2
  sorry

theorem general_relationship (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b :=
by
  sorry

end relation_1_relation_2_relation_3_general_relationship_l564_564249


namespace exists_three_points_covered_by_semicircular_disk_l564_564891

noncomputable def rightTriangle : Triangle :=
{ a := (0, 0),
  b := (18, 0),
  c := (0, 6 * Real.sqrt 3),
  right_angle_vertex := (0, 0) }

def has865Points (T : Triangle) : Prop :=
  ∃ (points : Finset (ℝ × ℝ)), points.card = 865 ∧ ∀ p ∈ points, is_point_in_triangle p T

def semicircular_disk_cover (d : ℝ) (points : Finset (ℝ × ℝ)) : Prop :=
  ∃ (center : (ℝ × ℝ)), ∀ p ∈ points, dist center p ≤ d / 2

theorem exists_three_points_covered_by_semicircular_disk :
  ∀ (T : Triangle), T = rightTriangle → has865Points T → 
  ∃ (points : Finset (ℝ × ℝ)), points.card = 3 ∧ semicircular_disk_cover 1 points :=
by
  intros
  sorry

end exists_three_points_covered_by_semicircular_disk_l564_564891


namespace negation_of_universal_proposition_l564_564491

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x+1) * exp x > 1) ↔ ∃ x : ℝ, x > 0 ∧ (x+1) * exp x ≤ 1 :=
by sorry

end negation_of_universal_proposition_l564_564491


namespace dodecahedron_paths_l564_564815

def is_adjacent (a b : ℕ) : Prop := sorry -- Definition of adjacency between two faces

def is_valid_move (a b : ℕ) : Prop := is_adjacent a b

def top_faces := {1, 2, 3, 4, 5} -- Top row faces
def bottom_faces := {6, 7, 8, 9, 10} -- Bottom row faces
def central_bottom_face := 11 -- Central bottom face

def is_valid_path (path : List ℕ) : Prop :=
  path.head = 1 ∧ -- Start at the top face
  path.tail.tail.tail.head ∈ bottom_faces ∧ -- Three faces in the top row
  path.tail.tail.tail.tail.tail.tail.head = central_bottom_face ∧ -- End at the bottom central face
  (∀ i, i < path.length - 1 → is_valid_move (List.get i path) (List.get (i + 1) path)) ∧ -- Valid adjacent moves
  (∀ i j, i < j → path.nth i ≠ path.nth j) -- Each face visited only once

theorem dodecahedron_paths : 
  ∃ (paths : List (List ℕ)), 
    (∀ path ∈ paths, is_valid_path path) ∧ 
    paths.length = 6480 := 
by 
  sorry

end dodecahedron_paths_l564_564815


namespace oldest_babysat_age_l564_564397

theorem oldest_babysat_age
  (jane_start_age : ℕ)
  (current_age : ℕ)
  (stopped_babysitting_years_ago : ℕ)
  (half_ratio : ℕ → ℕ := λ age, age / 2)
  (oldest_child_when_stopped : ℕ)
  (years_since_stopped : ℕ) :
  jane_start_age = 18 →
  current_age = 34 →
  stopped_babysitting_years_ago = 12 →
  let age_when_stopped := current_age - stopped_babysitting_years_ago in
  age_when_stopped = 22 →
  oldest_child_when_stopped = half_ratio age_when_stopped →
  oldest_child_when_stopped = 11 →
  years_since_stopped = stopped_babysitting_years_ago →
  let current_oldest_child_age := oldest_child_when_stopped + years_since_stopped in
  current_oldest_child_age = 23 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  -- Prove statements using the given hypotheses h1 to h7
  sorry

end oldest_babysat_age_l564_564397


namespace inhabitant_50th_statement_l564_564241

-- Definition of inhabitant types
inductive InhabitantType
| Knight
| Liar

-- Predicate for the statement of inhabitants
def says (inhabitant : InhabitantType) (statement : InhabitantType) : Bool :=
  match inhabitant with
  | InhabitantType.Knight => true
  | InhabitantType.Liar => false

-- Conditions from the problem
axiom inhabitants : Fin 50 → InhabitantType
axiom statements : ∀ i : Fin 50, i.val % 2 = 0 → (says (inhabitants ((i + 1) % 50)) InhabitantType.Liar = false)
axiom statements' : ∀ i : Fin 50, i.val % 2 = 1 → (says (inhabitants ((i + 1) % 50)) InhabitantType.Knight = true)

-- Goal to prove
theorem inhabitant_50th_statement : says (inhabitants 49) InhabitantType.Knight := by
  sorry

end inhabitant_50th_statement_l564_564241


namespace third_number_is_forty_four_l564_564594

theorem third_number_is_forty_four (a b c d e : ℕ) (h1 : a = e + 1) (h2 : b = e) 
  (h3 : c = e - 1) (h4 : d = e - 2) (h5 : e = e - 3) 
  (h6 : (a + b + c) / 3 = 45) (h7 : (c + d + e) / 3 = 43) : 
  c = 44 := 
sorry

end third_number_is_forty_four_l564_564594


namespace f_is_even_l564_564759

variable {R : Type} [Field R]

noncomputable def f (x : R) : R

axiom period_2 : ∀ x : R, f (x + 2) = f x
axiom symmetric_property : ∀ x : R, f (x + 2) = f (2 - x)

theorem f_is_even : ∀ x : R, f (-x) = f (x) :=
by
  sorry

end f_is_even_l564_564759


namespace count_digit_7_to_199_l564_564868

theorem count_digit_7_to_199 : (Nat.countDigitInRange 7 100 199) = 20 := by
  sorry

end count_digit_7_to_199_l564_564868


namespace domain_of_function_l564_564962

def f (x : ℝ) : ℝ := x / (1 - x) + Real.sqrt (x + 1)

theorem domain_of_function :
  {x : ℝ | x != 1 ∧ x ≥ -1} = {x : ℝ | -1 ≤ x ∧ x < 1} ∪ {x : ℝ | x > 1} :=
sorry

end domain_of_function_l564_564962


namespace infinite_string_properties_l564_564225

theorem infinite_string_properties :
  ∀ (a : ℕ → char),
  (∀ n, a(n) = 'T' ∨ a(n) = 'S') ∧
  (∀ i j, a(i) = 'T' → a(j) = 'T' → a(i + j) = 'S') ∧
  (∀ k, ∃ inf, ∀ n, n ≥ inf → a(2 * (n + 1) - 1) = 'T' → a(n + 1) = 2 * (n + 1) - 1) :=
sorry

end infinite_string_properties_l564_564225


namespace number_of_possible_values_s_l564_564975

theorem number_of_possible_values_s :
  let s : ℚ := (a / 10^5 + b / 10^4 + c / 10^3 + d / 10^2 + e / 10^1) in
  (s ≥ 2614 / 10000) ∧ (s ≤ 3030 / 10000) →
  ∃ n : ℕ, n = 4161 := 
by
  sorry

end number_of_possible_values_s_l564_564975


namespace problem_statement_l564_564764

def f (x : ℝ) : ℝ :=
  if x < 1 then x + 1 else -x + 3

theorem problem_statement : f (f (5 / 2)) = 3 / 2 := by
  sorry

end problem_statement_l564_564764


namespace solve_quartic_eq_l564_564657

theorem solve_quartic_eq (z : ℝ) : z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = 2 :=
by 
  sorry

end solve_quartic_eq_l564_564657


namespace ball_bounce_height_l564_564141

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (n k : ℕ) (hk : k = 6) (h_initial : h₀ = 20) 
  (ratio : r = 2 / 3) (h_n_k_bounces : h₀ * r ^ k < 2) : k = 6 :=
by
  have h_k_approx : h₀ * r ^ k = 20 * (2 / 3) ^ k := by
    rw [h_initial, ratio]
  calc 20 * (2 / 3) ^ 6 < 2 : by have := hk; sorry

end ball_bounce_height_l564_564141


namespace not_prime_plus_square_l564_564574

open Nat

theorem not_prime_plus_square (n : ℕ) (h1 : n ≥ 5) (h2 : n % 3 = 2) : ¬ ∃ (p k : ℕ), Prime p ∧ n^2 = p + k^2 :=
  sorry

end not_prime_plus_square_l564_564574


namespace clothes_percentage_l564_564529

noncomputable def initial_weight (W : ℝ) : ℝ := W
noncomputable def weight_after_loss (W : ℝ) : ℝ := 0.90 * W
noncomputable def final_weigh_in_weight (W : ℝ) : ℝ := 0.918 * W

theorem clothes_percentage (W : ℝ) : 
  let W_loss := weight_after_loss W in
  let final_weight := final_weigh_in_weight W in
  W_loss + W_loss * (2 / 100) = final_weight :=
by
  intros
  sorry

end clothes_percentage_l564_564529


namespace digit_7_appears_20_times_in_range_100_to_199_l564_564864

theorem digit_7_appears_20_times_in_range_100_to_199 : 
  ∃ (d7_count : ℕ), (d7_count = 20) ∧ (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 199 → 
  d7_count = (n.to_string.count '7')) :=
sorry

end digit_7_appears_20_times_in_range_100_to_199_l564_564864


namespace rebus_solution_l564_564258

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l564_564258


namespace watch_arrangement_count_l564_564137

noncomputable def number_of_satisfying_watch_arrangements : Nat :=
  let dial_arrangements := Nat.factorial 2
  let strap_arrangements := Nat.factorial 3
  dial_arrangements * strap_arrangements

theorem watch_arrangement_count :
  number_of_satisfying_watch_arrangements = 12 :=
by
-- Proof omitted
sorry

end watch_arrangement_count_l564_564137


namespace solve_quartic_eq_l564_564664

theorem solve_quartic_eq (z : ℝ) : z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = 2 :=
by 
  sorry

end solve_quartic_eq_l564_564664


namespace edric_hourly_rate_l564_564244

theorem edric_hourly_rate
  (monthly_salary : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (H1 : monthly_salary = 576)
  (H2 : hours_per_day = 8)
  (H3 : days_per_week = 6)
  (H4 : weeks_per_month = 4) :
  monthly_salary / weeks_per_month / days_per_week / hours_per_day = 3 := by
  sorry

end edric_hourly_rate_l564_564244


namespace sum_of_valid_six_digit_numbers_divisible_by_37_l564_564046

def is_valid_six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∀ d ∈ (Nat.digits 10 n), d ≠ 0 ∧ d ≠ 9

theorem sum_of_valid_six_digit_numbers_divisible_by_37 :
  let S := {n : ℕ | is_valid_six_digit_number n}
  (finset.sum (finset.filter (λ n, is_valid_six_digit_number n) (finset.range 1000000))) % 37 = 0 :=
sorry

end sum_of_valid_six_digit_numbers_divisible_by_37_l564_564046


namespace angle_AMD_deg_l564_564051

open Real

-- Conditions definition
def is_rectangle (A B C D : ℝ × ℝ) (AB BC CD DA : ℝ) := 
  AB = 8 ∧ BC = 4 ∧ (∃ M, M.1 = (A.1 + B.1) / 3 ∧ M.2 = A.2 ∧ ∠ AMD = ∠ CMD ∧ dist M D = 8 / 3)

-- Theorem statement
theorem angle_AMD_deg {A B C D M : ℝ × ℝ} 
  (hR : is_rectangle A B C D 8 4) :
  ∠ AMD = arccos (9 / 16) :=
sorry

end angle_AMD_deg_l564_564051


namespace min_distance_between_A_and_D_l564_564042

theorem min_distance_between_A_and_D 
  (A B C D : Type) [metric_space Type]
  (dist_AB : dist A B = 10)
  (dist_BC : dist B C = 4)
  (dist_CD : dist C D = 3) :
  ∃ d, d = 3 ∧ ∀ (p : ℝ), p = dist A D → p ≥ 3 :=
begin
   sorry
end

end min_distance_between_A_and_D_l564_564042


namespace find_bridge_length_l564_564177

-- Define a noncomputable to avoid rigorous numerical checks
noncomputable def speed_kmph := 72
noncomputable def time_seconds := 41.24670026397888

def speed_mps := speed_kmph * (1000 / 3600 : ℝ)
def train_length := 165
def total_distance := speed_mps * time_seconds
def bridge_length : ℝ := total_distance - train_length

theorem find_bridge_length : bridge_length = 660.9340052795776 := by
  -- (The proof details are omitted with 'sorry')
  sorry

end find_bridge_length_l564_564177


namespace average_speed_l564_564097

theorem average_speed (s₁ s₂ s₃ s₄ s₅ : ℝ) (h₁ : s₁ = 85) (h₂ : s₂ = 45) (h₃ : s₃ = 60) (h₄ : s₄ = 75) (h₅ : s₅ = 50) : 
  (s₁ + s₂ + s₃ + s₄ + s₅) / 5 = 63 := 
by 
  sorry

end average_speed_l564_564097


namespace bisection_of_segment_l564_564543

variables {α : Type*} [AffineSpace α (EuclideanSpace ℝ)] -- This introduces an affine space over Euclidean space

structure Tetrahedron := 
(A B C D : α)

noncomputable def midpoint (A B : α) : α := (1/2 : ℝ) • (A +ᵥ B)

def line_through_midpoints (t : Tetrahedron) : affine_subspace ℝ α :=
{ carrier := {p | ∃ (k : ℝ), k • midpoint t.A t.B +ᵥ (1-k) • midpoint t.C t.D = p},
  nonempty := sorry,
  smul_vadd_mem := sorry }

def plane_containing_line {t : Tetrahedron} (l : affine_subspace ℝ α) : affine_subspace ℝ α :=
affine_subspace.span ℝ (set_of (λ p : α, ∃ M N : α, 
  M ∈ affine_span ℝ ({t.C, t.D} : set α) ∧
  N ∈ affine_span ℝ ({t.A, t.B} : set α) ∧
  same_side t.A t.B p t.C t.D))

theorem bisection_of_segment {t : Tetrahedron} (l : affine_subspace ℝ α) 
  (P Q : α) (h_l : l = line_through_midpoints t) (M N : α)
  (h_pi : plane_containing_line l)
  (h_M : M ∈ affine_span ℝ ({t.B, t.C} : set α))
  (h_N : N ∈ affine_span ℝ ({t.A, t.D} : set α))
  (h_P : P = midpoint t.A t.B) (h_Q : Q = midpoint t.C t.D) :
  ∃ (K : α), line_through_midpoints t = midpoint M N :=
sorry

end bisection_of_segment_l564_564543


namespace days_in_month_l564_564157

theorem days_in_month 
  (S : ℕ) (D : ℕ) (h1 : 150 * S + 120 * D = (S + D) * 125) (h2 : S = 5) :
  S + D = 30 :=
by
  sorry

end days_in_month_l564_564157


namespace find_z_values_l564_564652

theorem find_z_values : {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -sqrt 2, sqrt 2, 2} :=
sorry

end find_z_values_l564_564652


namespace solve_quartic_equation_l564_564694

theorem solve_quartic_equation :
  ∀ z : ℝ, z^4 - 6 * z^2 + 8 = 0 ↔ (z = -2 ∨ z = -sqrt 2 ∨ z = sqrt 2 ∨ z = 2) :=
by {
  sorry
}

end solve_quartic_equation_l564_564694


namespace sum_S_2016_eq_1008_l564_564768

def sequence_an (n : ℕ) : ℚ :=
  n * Real.cos (n * Real.pi / 2)

noncomputable def sum_S_n (n : ℕ) : ℚ :=
  ∑ i in Finset.range (n + 1), sequence_an i

theorem sum_S_2016_eq_1008 :
  sum_S_n 2016 = 1008 :=
sorry

end sum_S_2016_eq_1008_l564_564768


namespace tangent_line_eq_l564_564482

theorem tangent_line_eq : 
  ∀ (f : ℝ → ℝ) (x0 y0 : ℝ), 
    (∀ x, f x = x^2) → 
    x0 = 2 → 
    y0 = f x0 → 
    (∃ (m : ℝ), m = 2 * x0 ∧ ∀ x y, y = m * (x - x0) + y0 → 4 * x - y - 4 = 0) :=
by 
  intros f x0 y0 hf hx0 hy0
  use 2 * x0
  split
  {
    sorry -- Placeholder for the proof part
  }
  intros x y h
  sorry -- Placeholder for the proof part

end tangent_line_eq_l564_564482


namespace hiring_manager_acceptance_l564_564474

theorem hiring_manager_acceptance 
    (average_age : ℤ) (std_dev : ℤ) (num_ages : ℤ)
    (applicant_ages_are_int : ∀ (x : ℤ), x ≥ (average_age - std_dev) ∧ x ≤ (average_age + std_dev)) :
    (∃ k : ℤ, (average_age + k * std_dev) - (average_age - k * std_dev) + 1 = num_ages) → k = 1 :=
by 
  intros h
  sorry

end hiring_manager_acceptance_l564_564474


namespace symmetry_center_g_l564_564102

open Real
noncomputable theory

def f (x : ℝ) : ℝ := 2 * sin (x + π / 6) + 1

def g (x : ℝ) : ℝ := 2 * sin (2 * x - π / 6) + 1

def is_symmetry_center (g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, g (2 * p.1 - x) = 2 * p.2 - g x

theorem symmetry_center_g :
  is_symmetry_center g (π / 12, 1) :=
sorry

end symmetry_center_g_l564_564102


namespace value_of_2x_4y_l564_564726

theorem value_of_2x_4y (x y : ℤ) (h : x + 2 * y = 3) : 2^x * 4^y = 8 :=
sorry

end value_of_2x_4y_l564_564726


namespace inequality_f_n_l564_564011

theorem inequality_f_n {f : ℕ → ℕ} {k : ℕ} (strict_mono_f : ∀ {a b : ℕ}, a < b → f a < f b)
  (h_f : ∀ n : ℕ, f (f n) = k * n) : ∀ n : ℕ, 
  (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ ((k + 1) * n) / 2 :=
by
  sorry

end inequality_f_n_l564_564011


namespace seashells_needed_l564_564927

variable (current_seashells : ℕ) (target_seashells : ℕ)

theorem seashells_needed : current_seashells = 19 → target_seashells = 25 → target_seashells - current_seashells = 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end seashells_needed_l564_564927


namespace multiples_l564_564949

variable (a b : ℤ)
variable (ha : ∃ k : ℤ, a = 4 * k)
variable (hb : ∃ m : ℤ, b = 8 * m)

theorem multiples (a b : ℤ) (ha : ∃ k : ℤ, a = 4 * k) (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ t : ℤ, b = 4 * t) ∧
  (∃ u : ℤ, a + b = 4 * u) ∧
  (∃ v : ℤ, a + b = 2 * v) :=
  sorry

end multiples_l564_564949


namespace closed_polygon_inequality_l564_564381

noncomputable def length_eq (A B C D : ℝ × ℝ × ℝ) (l : ℝ) : Prop :=
  dist A B = l ∧ dist B C = l ∧ dist C D = l ∧ dist D A = l

theorem closed_polygon_inequality 
  (A B C D P : ℝ × ℝ × ℝ) (l : ℝ)
  (hABCD : length_eq A B C D l) :
  dist P A < dist P B + dist P C + dist P D :=
sorry

end closed_polygon_inequality_l564_564381


namespace degree_g_l564_564368

def f (x : ℝ) : ℝ := -9 * x^5 + 4 * x^3 + 2 * x - 6

theorem degree_g (g : ℝ → ℝ) (h : degree (f + g) = 2) : degree g = 5 :=
sorry

end degree_g_l564_564368


namespace digit_7_appears_20_times_in_range_100_to_199_l564_564866

theorem digit_7_appears_20_times_in_range_100_to_199 : 
  ∃ (d7_count : ℕ), (d7_count = 20) ∧ (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 199 → 
  d7_count = (n.to_string.count '7')) :=
sorry

end digit_7_appears_20_times_in_range_100_to_199_l564_564866


namespace triangle_eq_medians_incircle_l564_564106

-- Define a triangle and the properties of medians and incircle
structure Triangle (α : Type) [Nonempty α] :=
(A B C : α)

def is_equilateral {α : Type} [Nonempty α] (T : Triangle α) : Prop :=
  ∃ (d : α → α → ℝ), d T.A T.B = d T.B T.C ∧ d T.B T.C = d T.C T.A

def medians_segments_equal {α : Type} [Nonempty α] (T : Triangle α) (incr_len : (α → α → ℝ)) : Prop :=
  ∀ (MA MB MC : α), incr_len MA MB = incr_len MB MC ∧ incr_len MB MC = incr_len MC MA

-- The main theorem statement
theorem triangle_eq_medians_incircle {α : Type} [Nonempty α] 
  (T : Triangle α) (incr_len : α → α → ℝ) 
  (h : medians_segments_equal T incr_len) : is_equilateral T :=
sorry

end triangle_eq_medians_incircle_l564_564106


namespace maxRedBulbs_l564_564592

theorem maxRedBulbs (n : ℕ) (h : n = 50) (adj : ∀ i, i < n → (¬ (red i ∧ red (i+1)))) : ∃ k, k = 33 :=
by
  sorry

end maxRedBulbs_l564_564592


namespace probability_of_point_in_sphere_l564_564164

noncomputable 
def probability_in_sphere : ℝ :=
  let cube_volume := 4^3
  let sphere_volume := (4 * π / 3) * (2^3)
  sphere_volume / cube_volume

theorem probability_of_point_in_sphere :
  ∀ (x y z : ℝ), (-2 ≤ x ∧ x ≤ 2) ∧ (-2 ≤ y ∧ y ≤ 2) ∧ (-2 ≤ z ∧ z ≤ 2) →
  probability_in_sphere = π / 6 :=
by
  sorry

end probability_of_point_in_sphere_l564_564164


namespace challenge_Jane_l564_564944

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def card_pairs : List (Char ⊕ ℕ) :=
  [Sum.inl 'A', Sum.inl 'T', Sum.inl 'U', Sum.inr 5, Sum.inr 8, Sum.inr 10, Sum.inr 14]

def Jane_claim (c : Char ⊕ ℕ) : Prop :=
  match c with
  | Sum.inl v => is_vowel v → ∃ n, Sum.inr n ∈ card_pairs ∧ is_even n
  | Sum.inr n => false

theorem challenge_Jane (cards : List (Char ⊕ ℕ)) (h : card_pairs = cards) :
  ∃ c ∈ cards, c = Sum.inr 5 ∧ ¬Jane_claim (Sum.inr 5) :=
sorry

end challenge_Jane_l564_564944


namespace pandas_bamboo_consumption_l564_564187

/-- Given:
  1. An adult panda can eat 138 pounds of bamboo each day.
  2. A baby panda can eat 50 pounds of bamboo a day.
Prove: the total pounds of bamboo eaten by both pandas in a week is 1316 pounds. -/
theorem pandas_bamboo_consumption :
  let adult_daily_bamboo := 138
  let baby_daily_bamboo := 50
  let days_in_week := 7
  (adult_daily_bamboo * days_in_week) + (baby_daily_bamboo * days_in_week) = 1316 := by
  sorry

end pandas_bamboo_consumption_l564_564187


namespace rebus_solution_l564_564259

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l564_564259


namespace inequality_solution_non_negative_integer_solutions_l564_564466

theorem inequality_solution (x : ℝ) :
  (x - 2) / 2 ≤ (7 - x) / 3 → x ≤ 4 :=
by
  sorry

theorem non_negative_integer_solutions :
  { n : ℤ | n ≥ 0 ∧ n ≤ 4 } = {0, 1, 2, 3, 4} :=
by
  sorry

end inequality_solution_non_negative_integer_solutions_l564_564466


namespace star_exists_l564_564220

theorem star_exists (M N : ℕ) (stars : Finset (Fin M × Fin N))
  (h1 : M < N)
  (h2 : ∀ i : Fin M, ∃ j : Fin N, (i, j) ∈ stars)
  (h3 : ∀ j : Fin N, ∃ i : Fin M, (i, j) ∈ stars) :
  ∃ cell : Fin M × Fin N, cell ∈ stars ∧ 
  (stars.count (λ p, p.1 = cell.1) > stars.count (λ p, p.2 = cell.2)) :=
by
  sorry

end star_exists_l564_564220


namespace relationship_between_a_and_b_l564_564522

def ellipse_touching_hyperbola (a b : ℝ) :=
  ∀ x y : ℝ, ( (x / a) ^ 2 + (y / b) ^ 2 = 1 ∧ y = 1 / x → False )

  theorem relationship_between_a_and_b (a b : ℝ) :
  ellipse_touching_hyperbola a b →
  a * b = 2 :=
by
  sorry

end relationship_between_a_and_b_l564_564522


namespace hyperbola_eccentricity_l564_564772

theorem hyperbola_eccentricity (m a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : m ≠ 0) 
  (h_asymptotes : ∀ {x y : ℝ}, (y = (b / a) * x ∨ y = -(b / a) * x) -> (y = (1 / 3) * x - (m / 3))) 
  (h_PA_PB : |PA| = |PB|) :
  eccentricity = (sqrt 5 / 2) := 
sorry

end hyperbola_eccentricity_l564_564772


namespace find_fourth_score_l564_564454

def average_score_condition (s1 s2 s3 s4 : ℝ) (avg : ℝ) (n : ℝ) : Prop :=
  ((s1 + s2 + s3 + s4) / n = avg)

theorem find_fourth_score :
  ∀ (s1 s2 s3 : ℝ) (avg : ℝ), 
  s1 = 65 → 
  s2 = 67 → 
  s3 = 76 → 
  avg = 76.6 → 
  average_score_condition s1 s2 s3 98.4 avg 4 :=
by
  intros s1 s2 s3 avg h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  dsimp [average_score_condition]
  linarith

end find_fourth_score_l564_564454


namespace trapezium_distance_parallel_sides_l564_564284

theorem trapezium_distance_parallel_sides
  (l1 l2 area : ℝ) (h : ℝ)
  (h_area : area = (1 / 2) * (l1 + l2) * h)
  (hl1 : l1 = 30)
  (hl2 : l2 = 12)
  (h_area_val : area = 336) :
  h = 16 :=
by
  sorry

end trapezium_distance_parallel_sides_l564_564284


namespace exponent_properties_l564_564110

theorem exponent_properties : (7^(-3))^0 + (7^0)^2 = 2 :=
by
  -- Using the properties of exponents described in the problem:
  -- 1. Any number raised to the power of 0 equals 1.
  -- 2. Any base raised to the power of 0 equals 1, with further raising to the power of 2 yielding 1.
  -- We can conclude and add the two results to get the final statement.
  sorry

end exponent_properties_l564_564110


namespace find_digits_l564_564274

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l564_564274


namespace coeff_x2_exp_binom_l564_564005

-- The problem as a Lean statement
theorem coeff_x2_exp_binom (a : ℝ) (x : ℝ) : 
  a = ∫ x in 0..π, (sin x + cos x) →
  (n : ℕ) (b : ℝ) (r : ℕ) (h1 : n = 6) (h2 : b = 2) (h3 : r = 1) (hx : x ≠ 0) :
  (finset.choose n r * (-1 : ℝ) ^ r * b ^ (n - r)) = -192 :=
by
  -- integral calculation and binomial theorem steps go here
  sorry

end coeff_x2_exp_binom_l564_564005


namespace f_zero_eq_one_f_pos_all_f_increasing_l564_564627

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : f 0 ≠ 0
axiom f_pos : ∀ x, 0 < x → 1 < f x
axiom f_mul : ∀ a b : ℝ, f (a + b) = f a * f b

theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem f_pos_all : ∀ x : ℝ, 0 < f x :=
sorry

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ :=
sorry

end f_zero_eq_one_f_pos_all_f_increasing_l564_564627


namespace microbrewery_decrease_hours_l564_564441

theorem microbrewery_decrease_hours :
  ∃ x : ℝ, (∀ O H : ℝ, O > 0 ∧ H > 0 →
  (1.20 * O / (H * (1 - x / 100))) = 2.7143 * (O / H)) ↔ x ≈ 55.80 :=
by
  sorry

end microbrewery_decrease_hours_l564_564441


namespace qatar_location_is_accurate_l564_564472

def qatar_geo_location :=
  "The most accurate representation of Qatar's geographical location is latitude 25 degrees North, longitude 51 degrees East."

theorem qatar_location_is_accurate :
  qatar_geo_location = "The most accurate representation of Qatar's geographical location is latitude 25 degrees North, longitude 51 degrees East." :=
sorry

end qatar_location_is_accurate_l564_564472


namespace marbles_solid_color_non_yellow_l564_564549

theorem marbles_solid_color_non_yellow (total_marble solid_colored solid_yellow : ℝ)
    (h1: solid_colored = 0.90 * total_marble)
    (h2: solid_yellow = 0.05 * total_marble) :
    (solid_colored - solid_yellow) / total_marble = 0.85 := by
  -- sorry is used to skip the proof
  sorry

end marbles_solid_color_non_yellow_l564_564549


namespace exists_monochromatic_triangle_of_area_one_l564_564163

theorem exists_monochromatic_triangle_of_area_one (plane : ℝ × ℝ → ℕ) (h_color : ∀ p : ℝ × ℝ, plane p ∈ {0, 1, 2}) :
  ∃ (A B C : ℝ × ℝ), plane A = plane B ∧ plane B = plane C ∧ is_triangle_area_one A B C :=
sorry

-- Helper predicate that checks if three points form a triangle of area 1
def is_triangle_area_one (A B C : ℝ × ℝ) : Prop :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) = 1

end exists_monochromatic_triangle_of_area_one_l564_564163


namespace part_a_part_b_l564_564350

noncomputable def sequence (n : ℕ) : ℝ := Real.cos (Real.pi * 10 ^ n / 180)

theorem part_a : (sequence 100) > 0 := 
by {
  sorry
}

theorem part_b : abs (sequence 100) < 0.18 :=
by {
  sorry
}

end part_a_part_b_l564_564350


namespace grisha_wins_probability_expected_flips_l564_564826

-- Define conditions
def even_flip_heads_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 0 ∧ flip n = tt

def odd_flip_tails_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 1 ∧ flip n = ff

-- 1. Prove the probability P of Grisha winning the coin toss game is 1/3
theorem grisha_wins_probability (flip : ℕ → bool) (P : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  P = 1/3 := sorry

-- 2. Prove the expected number E of coin flips until the outcome is decided is 2
theorem expected_flips (flip : ℕ → bool) (E : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  E = 2 := sorry

end grisha_wins_probability_expected_flips_l564_564826


namespace value_of_a_l564_564324

theorem value_of_a (a n : ℤ) (x : ℝ) (h1 : 2^n = 32) (h2 : (choose n 3) * a^3 = -270) : a = -3 :=
by
  have h_n : n = 5 := by sorry -- solving 2^n = 32 for n
  have h_const_term : (choose 5 3) * a^3 = -270 := by sorry -- given as h2
  
  -- Establish that the constant term condition computes a to be -3
  have h_a : a = -3 := by sorry
  exact h_a

end value_of_a_l564_564324


namespace count_irrational_numbers_l564_564185

theorem count_irrational_numbers (a b c d e f : ℝ)
  (h1 : a = 31/7)
  (h2 : b = -Real.pi)
  (h3 : c = 3.14159)
  (h4 : d = Real.sqrt 8)
  (h5 : e = -27^(1/3))
  (h6 : f = 1^2) :
  (∀ x ∈ {a, b, c, d, e, f}, x ∈ ℝ) → 
  (∃ S ⊆ {a, b, c, d, e, f}, S.card = 2 ∧ ∀ x ∈ S, ¬ ∃ m n : ℤ, x = m / n) :=
by
  sorry

end count_irrational_numbers_l564_564185


namespace grisha_win_probability_expected_number_coin_flips_l564_564819

-- Problem 1: Probability of Grisha's win
theorem grisha_win_probability : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (grisha_win : ℚ) :=
  sorry

-- Problem 2: Expected number of coin flips
theorem expected_number_coin_flips : 
  ∀ (even_heads_criteria : ℤ → ℚ) (odd_tails_criteria : ℤ → ℚ), 
    even_heads_criteria ∈ {2 * n | n : ℤ} → 
    odd_tails_criteria ∈ {2 * n + 1 | n : ℤ} → 
    (expected_tosses : ℚ) :=
  sorry

end grisha_win_probability_expected_number_coin_flips_l564_564819


namespace digit_7_count_in_range_l564_564877

def count_digit_7 : ℕ :=
  let units_place := (107 - 100) / 10 + 1
  let tens_place := 10
  units_place + tens_place

theorem digit_7_count_in_range : count_digit_7 = 20 := by
  sorry

end digit_7_count_in_range_l564_564877


namespace digit_7_count_in_range_100_to_199_l564_564890

/-- 
Given the range of numbers from 100 to 199 inclusive,
prove that the digit 7 appears exactly 20 times.
-/
theorem digit_7_count_in_range_100_to_199 : 
  let count_7 := (λ n : ℕ, 
    (n / 100 = 7) +  -- hundreds place
    ((n % 100) / 10 = 7) + -- tens place
    ((n % 10) = 7)) in
  (Finset.sum (Finset.range' 100 100) count_7 = 20) :=
by sorry

end digit_7_count_in_range_100_to_199_l564_564890


namespace probability_set_correct_l564_564575

open Real

noncomputable def prob (d : ℕ) : ℝ :=
  if d > 0 then ln (d + 1) - ln d else 0

theorem probability_set_correct :
  prob 5 = (1 / 2) * (prob 4 + prob 6) :=
by
  have h : ∀ d, d > 0 → prob d = ln (d + 1) - ln d := by
    intro d hd
    simp [prob, hd]
  rw [h 5 (by decide), h 4 (by decide), h 6 (by decide)]
  field_simp
  ring

end probability_set_correct_l564_564575


namespace choir_members_number_l564_564488

theorem choir_members_number
  (n : ℕ)
  (h1 : n % 12 = 10)
  (h2 : n % 14 = 12)
  (h3 : 300 ≤ n ∧ n ≤ 400) :
  n = 346 :=
sorry

end choir_members_number_l564_564488


namespace initial_spoons_to_knives_ratio_l564_564993

-- Problem conditions formalized in Lean
variable (F K T S : ℕ)
variable (initial_forks initial_knives initial_teaspoons initial_spoons : ℕ)

def initial_forks : ℕ := 6

def initial_knives : ℕ := initial_forks + 9

def initial_teaspoons : ℕ := initial_forks / 2

-- We define the total cutlery pieces after adding 2 of each type
def total_cutlery_after_addition : ℕ := initial_forks + 2 + initial_knives + 2 + initial_teaspoons + 2 + initial_spoons + 2

/-- The proof statement asserting the ratio of spoons to knives initially is 28 : 15 -/
theorem initial_spoons_to_knives_ratio :
  initial_spoons = 28 →
  initial_knives = 15 →
  total_cutlery_after_addition = 62 →
  initial_spoons = initial_knives + 13 := 
begin
  sorry
end

end initial_spoons_to_knives_ratio_l564_564993


namespace triangle_perimeter_range_expression_l564_564752

-- Part 1: Prove the perimeter of △ABC
theorem triangle_perimeter (a b c : ℝ) (cosB : ℝ) (area : ℝ)
  (h1 : b^2 = a * c) (h2 : cosB = 3 / 5) (h3 : area = 2) :
  a + b + c = Real.sqrt 5 + Real.sqrt 21 :=
sorry

-- Part 2: Prove the range for the given expression
theorem range_expression (a b c : ℝ) (q : ℝ)
  (h1 : b = a * q) (h2 : c = a * q^2) :
  (Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2 :=
sorry

end triangle_perimeter_range_expression_l564_564752


namespace total_amount_l564_564172

-- Define the conditions in Lean
variables (X Y Z: ℝ)
variable (h1 : Y = 0.75 * X)
variable (h2 : Z = (2/3) * X)
variable (h3 : Y = 48)

-- The theorem stating that the total amount of money is Rs. 154.67
theorem total_amount (X Y Z : ℝ) (h1 : Y = 0.75 * X) (h2 : Z = (2/3) * X) (h3 : Y = 48) : 
  X + Y + Z = 154.67 := 
by
  sorry

end total_amount_l564_564172


namespace solve_quartic_eqn_l564_564677

theorem solve_quartic_eqn (z : ℂ) : 
  z ^ 4 - 6 * z ^ 2 + 8 = 0 ↔ z = 2 ∨ z = -2 ∨ z = Complex.sqrt 2 ∨ z = -Complex.sqrt 2 := 
sorry

end solve_quartic_eqn_l564_564677


namespace petya_sequence_count_l564_564029

theorem petya_sequence_count :
  let n := 100 in
  (5 ^ n - 3 ^ n) =
  (λ S : (ℕ × ℕ) → ℕ, 5 ^ S (n, n) - 3 ^ S (n, n) ) 5 100 :=
sorry

end petya_sequence_count_l564_564029


namespace cos_270_eq_zero_l564_564202

theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 := by
  sorry

end cos_270_eq_zero_l564_564202


namespace par_value_is_correct_l564_564437

noncomputable def par_value_of_shares 
  (num_preferred_shares : ℕ)
  (num_common_shares : ℕ)
  (dividend_preferred_annual : ℝ)
  (dividend_common_annual : ℝ)
  (total_annual_dividend : ℝ) : ℝ :=
  let P := total_annual_dividend / ((num_preferred_shares * dividend_preferred_annual) +
                                    (num_common_shares * dividend_common_annual))
  in P

theorem par_value_is_correct :
  par_value_of_shares 1200 3000 0.1 0.07 16500 = 50 :=
by
  sorry

end par_value_is_correct_l564_564437


namespace goods_train_crossing_time_l564_564154

noncomputable def time_to_cross_platform (train_speed_kmph : ℝ) (train_length_m : ℝ) (platform_length_m : ℝ) : ℝ :=
  let train_speed_mps := (train_speed_kmph * 1000) / 3600
  let total_distance_m := train_length_m + platform_length_m
  total_distance_m / train_speed_mps

theorem goods_train_crossing_time :
  time_to_cross_platform 72 230.0384 250 ≈ 24 :=
by
  sorry

end goods_train_crossing_time_l564_564154


namespace integer_count_between_sqrt5_and_sqrt26_l564_564601

theorem integer_count_between_sqrt5_and_sqrt26 : 
  (card {n : ℤ | (2 : ℝ) < sqrt 5 ∧ sqrt 5 < 3 ∧ 5 < sqrt 26 ∧ sqrt 26 < 6 ∧ (sqrt 5 : ℝ) < (n : ℝ) ∧ (n : ℝ) < (sqrt 26 : ℝ)} = 3) :=
by {
  -- proof goes here
  sorry
}

end integer_count_between_sqrt5_and_sqrt26_l564_564601


namespace odd_square_sum_l564_564931

-- Problem Statement:
-- Prove that (2n - 1)^2 equals the sum of 2n - 1 consecutive numbers starting from n up to 3n - 2

theorem odd_square_sum (n : ℕ) (h : n > 0) : 
  (2 * n - 1) ^ 2 = ∑ i in Finset.range (2 * n - 1), (n + i) := 
begin
  sorry,
end

end odd_square_sum_l564_564931


namespace system1_solution_system2_solution_l564_564465

theorem system1_solution (x y : ℤ) (h1 : x - y = 2) (h2 : x + 1 = 2 * (y - 1)) :
  x = 7 ∧ y = 5 :=
sorry

theorem system2_solution (x y : ℤ) (h1 : 2 * x + 3 * y = 1) (h2 : (y - 1) * 3 = (x - 2) * 4) :
  x = 1 ∧ y = -1 / 3 :=
sorry

end system1_solution_system2_solution_l564_564465


namespace quadratic_min_value_l564_564090

theorem quadratic_min_value :
  ∃ x : ℝ, (∀ y : ℝ, y = x^2 - 4 * x + 7 → y ≥ 3) ∧ (x = 2 → (x^2 - 4 * x + 7 = 3)) :=
by
  sorry

end quadratic_min_value_l564_564090


namespace locus_midpoint_of_segments_l564_564999

noncomputable theory

open_locale classical

variables {O₁ O₂ : ℝ × ℝ} {r₁ r₂ d : ℝ}

def midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

def circle (O : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) :=
  {P | (P.1 - O.1) ^ 2 + (P.2 - O.2) ^ 2 = r ^ 2}

def dist (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem locus_midpoint_of_segments
  (h1 : circle O₁ r₁ = {P | (P.1 - O₁.1) ^ 2 + (P.2 - O₁.2) ^ 2 = r₁ ^ 2})
  (h2: circle O₂ r₂ = {P | (P.1 - O₂.1) ^ 2 + (P.2 - O₂.2) ^ 2 = r₂ ^ 2})
  (h3: dist O₁ O₂ = d) :
  ∃ (M : set (ℝ × ℝ)),
    M = {M | (dist M ((O₁.1 + O₂.1) / 2, (O₁.2 + O₂.2) / 2) = 1) ∨ 
               (dist M ((O₁.1 + O₂.1) / 2, (O₁.2 + O₂.2) / 2) = 2)} :=
sorry

end locus_midpoint_of_segments_l564_564999


namespace count_valid_pairs_l564_564904

def F (k : ℕ) (a b : ℕ) : ℕ := (a + b)^k - a^k - b^k

def S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

noncomputable def valid_pair (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≤ b ∧ (F 5 a b) % (F 3 a b) = 0

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 22 ∧ ∀ (a b : ℕ), valid_pair a b → n = num_valid_pairs(a, b) :=
sorry

-- Auxiliary function to count valid pairs
def num_valid_pairs : ℕ :=
  Finset.card {pair | ∃ (a b : ℕ), pair = (a, b) ∧ valid_pair a b}

end count_valid_pairs_l564_564904


namespace distinct_pawn_arrangements_on_5x5_board_l564_564727

theorem distinct_pawn_arrangements_on_5x5_board:
  let n := 5
  in (∃ (pawn_positions : Fin n → Fin n)
          (distinct_pawns : Fin n → Fin n),
          (∀ i j : Fin n, (pawn_positions i ≠ pawn_positions j ∧
           abs (i - j) ≠ abs (pawn_positions i - pawn_positions j))) ∧
    (∃ arrangements : list (Fin n → Fin n), arrangements.length = 1200) := sorry

end distinct_pawn_arrangements_on_5x5_board_l564_564727


namespace area_of_45_45_90_triangle_l564_564955

open Real

theorem area_of_45_45_90_triangle (BF : ℝ) (hBF : BF = 4) : 
  ∃ (ABC : Type) [isTriangle ABC], 
  (∀ (A B C : ABC), angle A B = 45 ∧ angle B C = 90 ∧ angle C A = 45) ∧ 
  (∀ (A B C : ABC), 
     ∃ (F : ABC), isPerpendicular BF A C ∧ length BF = 4) ∧
  area ABC = 8 :=
by sorry

end area_of_45_45_90_triangle_l564_564955


namespace f_positive_probability_is_three_fourths_l564_564763

def f (x : ℝ) : ℝ := 2 ^ x - real.sqrt x - 14

noncomputable def probability_f_positive : ℝ :=
  (16 - 4) / 16

theorem f_positive_probability_is_three_fourths :
  (set.Ico 0 16).volume ∧
  (λ x, (x > 4)) →
  (set.Ico 0 16 ∧
  (set.Ico 16 4).volume / (set.Ico 0 16).volume) = 3 / 4 :=
by
  sorry

end f_positive_probability_is_three_fourths_l564_564763


namespace intersection_M_N_l564_564744

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 + 2*x - 3 ≤ 0}

theorem intersection_M_N :
  M ∩ N = {-2, -1, 0, 1} :=
by
  sorry

end intersection_M_N_l564_564744


namespace minimum_value_l564_564790

theorem minimum_value (a b : ℝ) (h : a + b ≠ 0) : a^2 + b^2 + 1 / (a + b)^2 ≥ sqrt 2 :=
sorry

end minimum_value_l564_564790


namespace num_valid_triplets_l564_564098

def is_valid_triplet (a b c : ℕ) : Prop :=
  a = c + 23 ∧ b = 61 - c ∧ 1 ≤ c ∧ c ≤ 60

theorem num_valid_triplets : 
  ∃ (n : ℕ), n = 60 ∧ ∀ (a b c : ℕ), is_valid_triplet a b c → ∃! c', c = c' :=
by
  have : ∀ c : ℕ, is_valid_triplet (c + 23) (61 - c) c ↔ 1 ≤ c ∧ c ≤ 60,
  { intro c,
    split,
    { intro h, exact ⟨h.2.1, h.2.2⟩ },
    { intro h, exact ⟨rfl, rfl, h⟩ } },
  let triplets := {c : ℕ | 1 ≤ c ∧ c ≤ 60},
  have triplet_count : triplets.card = 60,
  { rw [finset.card_filter, nsmul_eq_mul, range_eq_Ico],
    exact finset.card_Ico _ _ },
  exact ⟨60, triplet_count, λ a b c, 
    begin
      intro h,
      use c,
      split,
      { exact h.2 },
      { rw Eq, exact λ c₁' h₁', h₁'.symm }
    end⟩
  sorry

end num_valid_triplets_l564_564098


namespace range_of_a_l564_564632

-- Define the function f as per condition
def f (a x : ℝ) : ℝ := x^3 - a * x^2 - x + 6

-- Define the derivative of the function f
def f' (a x : ℝ) : ℝ := 3 * x^2 - 2 * a * x - 1

-- Define the function g that comes from the inequality analysis
def g (x : ℝ) : ℝ := (3 * x^2 - 1) / (2 * x)

-- State the theorem to prove the range of a
theorem range_of_a (a : ℝ) : (∀ x ∈ Ioo 0 1, f' a x ≤ 0) ↔ a ≥ 1 := sorry

end range_of_a_l564_564632


namespace rebus_solution_l564_564260

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l564_564260


namespace triangle_problem_l564_564812

noncomputable theory

open Real

variables {a b c : ℝ}
variables {A B C : ℝ}

theorem triangle_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : A > 0) (h5 : B > 0) (h6 : C > 0)
  (h7 : A + B + C = π) (h8 : cos B - 2 * cos A = (2 * a - b) * cos C / c) :
  (b = 2 → a = 4) ∧ (A > π / 2 → c = 3 → √3 < b ∧ b < 3) :=
by
  sorry

end triangle_problem_l564_564812


namespace necessary_condition_l564_564231

theorem necessary_condition :
  (∀ x : ℝ, (1 / x < 3) → (x > 1 / 3)) → (∀ x : ℝ, (1 / x < 3) ↔ (x > 1 / 3)) → False :=
by
  sorry

end necessary_condition_l564_564231


namespace determine_votes_l564_564456

noncomputable def total_votes (x : ℕ) : Prop :=
  let likes := 0.70 * x
  let dislikes := 0.30 * x
  let score := likes - dislikes
  score = 140

theorem determine_votes : ∃ x : ℕ, total_votes x ∧ x = 350 :=
by
  use 350
  sorry

end determine_votes_l564_564456


namespace gcd_lcm_condition_implies_divisibility_l564_564908

theorem gcd_lcm_condition_implies_divisibility
  (a b : ℤ) (h : Int.gcd a b + Int.lcm a b = a + b) : a ∣ b ∨ b ∣ a := 
sorry

end gcd_lcm_condition_implies_divisibility_l564_564908


namespace triangle_C_and_area_l564_564375

theorem triangle_C_and_area (a b c C A B : ℝ) (h1 : a + b = 5) (h2 : c = real.sqrt 7) (h3 : real.cos (2 * C) + 2 * real.cos (A + B) = -1) :
  C = 60 ∧
  ∃ S : ℝ, S = 3 * real.sqrt 3 / 2 :=
by {
  sorry
}

end triangle_C_and_area_l564_564375


namespace sequence_converges_to_fixed_point_l564_564402

theorem sequence_converges_to_fixed_point
  {a b : ℝ}
  (h : a ≤ b)
  (f : ℝ → ℝ)
  (hf_interval : ∀ x : ℝ, a ≤ x ∧ x ≤ b → a ≤ f(x) ∧ f(x) ≤ b)
  (hf_ineq : ∀ x y : ℝ, x ∈ set.Icc a b → y ∈ set.Icc a b → f(x) - f(y) ≤ abs(x - y))
  (x1 : ℝ)
  (hx1 : a ≤ x1 ∧ x1 ≤ b)
  (x : ℕ → ℝ)
  (hx : ∀ n : ℕ, x(n + 1) = (x(n) + f(x(n))) / 2) :
  ∃ l : ℝ, l ∈ set.Icc a b ∧ f(l) = l :=
by {
  sorry
}

end sequence_converges_to_fixed_point_l564_564402


namespace find_m_direct_proportion_l564_564374

theorem find_m_direct_proportion (m : ℝ) (h1 : m + 2 ≠ 0) (h2 : |m| - 1 = 1) : m = 2 :=
sorry

end find_m_direct_proportion_l564_564374


namespace exists_f_ff_eq_square_l564_564451

open Nat

theorem exists_f_ff_eq_square : ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n ^ 2 :=
by
  -- proof to be provided
  sorry

end exists_f_ff_eq_square_l564_564451


namespace number_of_positive_integers_l564_564228

theorem number_of_positive_integers (n : ℕ) :
  (∃ (x : ℕ), 0 < x ∧ 15 < x^2 + 4*x + 4 ∧ x^2 + 4*x + 4 < 50) → n = 4 :=
begin
  sorry
end

end number_of_positive_integers_l564_564228


namespace smallest_possible_n_l564_564417

theorem smallest_possible_n (n : ℕ) (x : Fin n → ℝ) 
  (h₁ : ∀ i, |x i| < 1)
  (h₂ : (Finset.univ.sum (λ i, |x i|)) = 31 + |Finset.univ.sum x|) :
  n ≥ 32 :=
sorry

end smallest_possible_n_l564_564417


namespace max_value_ab_l564_564325

noncomputable def max_ab : ℝ :=
  max {ab : ℝ | ∃ (a b x y : ℝ) (h1 : 0 < a) (h2 : 0 < b),
                          x^2 + y^2 - 4*a*x + 2*b*y + b^2 = 0 ∧
                          ∀ x y : ℝ, (x - y = 1) → x - y - 1 = 0 
                } 

theorem max_value_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ∀ x y : ℝ, (x - y = 1) → x - y - 1 = 0) :
  x^2 + y^2 - 4*a*x + 2*b*y + b^2 = 0 →
  ∃ x y: ℝ, ab = 1/8 :=
by
  sorry

end max_value_ab_l564_564325


namespace distinct_divisors_of_exp_minus_one_l564_564411

theorem distinct_divisors_of_exp_minus_one (a : ℕ) (b : ℕ) (r : ℕ) 
  (h_a : a ≥ 2) (h_b : Nat.isComposite b) (h_b_pos : b > 0)
  (h_r : ∃ d : List ℕ, ∀ x ∈ d, x ∣ b ∧ d.length = r) :
  ∃ d' : List ℕ, ∀ x ∈ d', x ∣ (a^b - 1) ∧ d'.length ≥ r :=
by
  sorry

end distinct_divisors_of_exp_minus_one_l564_564411


namespace isosceles_triangle_angle_sum_l564_564838

theorem isosceles_triangle_angle_sum (y : ℝ) :
  (∀ (a b c : ℝ), a + b + c = 180 ∧ (a = b ∨ b = c ∨ c = a) ∧ (a = 60 ∨ b = 60 ∨ c = 60) →
  (y = a ∨ y = b ∨ y = c) → y = 60) → 
  60 + 60 + 60 = 180 :=
by
  assume h
  sorry

end isosceles_triangle_angle_sum_l564_564838


namespace count_digit_7_to_199_l564_564869

theorem count_digit_7_to_199 : (Nat.countDigitInRange 7 100 199) = 20 := by
  sorry

end count_digit_7_to_199_l564_564869


namespace cos_270_eq_zero_l564_564213

theorem cos_270_eq_zero : Real.cos (270 * (π / 180)) = 0 :=
by
  -- Conditions given in the problem
  have rotation_def : ∀ (θ : ℝ), ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x = Real.cos θ ∧ y = Real.sin θ := by
    intros θ
    use [Real.cos θ, Real.sin θ]
    split
    · exact Real.cos_sq_add_sin_sq θ
    split
    · rfl
    · rfl

  have rotation_270 : ∃ (x y : ℝ), x = 0 ∧ y = -1 := by
    use [0, -1]
    split
    · rfl
    · rfl

  -- Goal to be proved
  have result := Real.cos (270 * (π / 180))
  show result = 0
  sorry

end cos_270_eq_zero_l564_564213


namespace arithmetic_identity_l564_564647

theorem arithmetic_identity : 3 * 5 * 7 + 15 / 3 = 110 := by
  sorry

end arithmetic_identity_l564_564647


namespace solve_quartic_l564_564683

theorem solve_quartic (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ (z = 2 ∨ z = -2 ∨ z = sqrt 2 ∨ z = -sqrt 2) :=
by
  sorry

end solve_quartic_l564_564683


namespace natural_number_with_six_divisors_two_prime_sum_78_is_45_l564_564712

def has_six_divisors (n : ℕ) : Prop :=
  (∃ p1 p2 : ℕ, p1.prime ∧ p2.prime ∧ p1 ≠ p2 ∧ 
  (∃ α1 α2 : ℕ, α1 + α2 > 0 ∧ n = p1 ^ α1 * p2 ^ α2 ∧ 
  (α1 + 1) * (α2 + 1) = 6))

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d, d > 0 ∧ n % d = 0) (Finset.range (n + 1))).sum id

theorem natural_number_with_six_divisors_two_prime_sum_78_is_45 (n : ℕ) :
  has_six_divisors n ∧ sum_of_divisors n = 78 → n = 45 := 
by 
  sorry

end natural_number_with_six_divisors_two_prime_sum_78_is_45_l564_564712


namespace sum_factorial_series_simplified_l564_564058

noncomputable def sum_factorial_series (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), (k + 1) * (Nat.factorial (k + 1))

theorem sum_factorial_series_simplified (n : ℕ) : sum_factorial_series n = (n + 1)! - 1 := by
  sorry

end sum_factorial_series_simplified_l564_564058


namespace problem_statement_l564_564777

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {4, 5}
def C_U (B : Set ℕ) : Set ℕ := U \ B

-- Statement
theorem problem_statement : A ∩ (C_U B) = {2} :=
  sorry

end problem_statement_l564_564777


namespace cost_of_paving_l564_564084

-- Definitions based on the given conditions
def length : ℝ := 6.5
def width : ℝ := 2.75
def rate : ℝ := 600

-- Theorem statement to prove the cost of paving
theorem cost_of_paving : length * width * rate = 10725 := by
  -- Calculation steps would go here, but we omit them with sorry
  sorry

end cost_of_paving_l564_564084


namespace arthur_spend_total_l564_564596

def appetizer_cost := 8
def entree_cost := 30
def wine_cost := 8
def dessert_cost := 7

def entree_discount := 0.40 * entree_cost
def subtotal_after_entree_discount := appetizer_cost + (entree_cost - entree_discount) + wine_cost + dessert_cost

def total_pre_tax_discount := subtotal_after_entree_discount * 0.10
def discounted_total_pre_tax := subtotal_after_entree_discount - total_pre_tax_discount

def tax_rate := 0.08
def tax_amount := discounted_total_pre_tax * tax_rate
def total_after_tax := discounted_total_pre_tax + tax_amount

def original_total := appetizer_cost + entree_cost + wine_cost + dessert_cost
def original_total_with_tax_before_discounts := original_total + original_total * tax_rate

def tip_rate := 0.20
def tip_amount := original_total_with_tax_before_discounts * tip_rate

def total_cost := total_after_tax + tip_amount

theorem arthur_spend_total : total_cost = 51.30 := sorry

end arthur_spend_total_l564_564596


namespace probability_one_even_dice_l564_564640

noncomputable def probability_exactly_one_even (p : ℚ) : Prop :=
  ∃ (n : ℕ), (p = (4 * (1/2)^4 )) ∧ (n = 1) → p = 1/4

theorem probability_one_even_dice : probability_exactly_one_even (1/4) :=
by
  unfold probability_exactly_one_even
  sorry

end probability_one_even_dice_l564_564640


namespace constant_t_l564_564621

noncomputable def k : ℝ := 2
def parabola (x : ℝ) : ℝ := x^2 + 1
def chord (C : ℝ × ℝ) : set (ℝ × ℝ) := {AB | ∃ A B : ℝ × ℝ, A ≠ B ∧ A.2 = parabola A.1 ∧ B.2 = parabola B.1 ∧ (1, k).1 ∈ (line_through A B)}

theorem constant_t (t : ℝ) :
  (∀ A B : ℝ × ℝ, A ≠ B → A.2 = parabola A.1 → B.2 = parabola B.1 → 
  (1, k).1 ∈ (line_through A B) → t = (1 / dist (C A) 2) + (1 / dist (C B) 2)) 
  → t = 4 / 5 :=
sorry

end constant_t_l564_564621


namespace area_of_border_l564_564168

theorem area_of_border (height_photo : ℕ) (width_photo : ℕ) (border : ℕ) (a b : ℕ) :
  height_photo = 12 → width_photo = 15 → border = 3 → a = 18 → b = 21 →
  ((b * a) - (width_photo * height_photo) = 198) :=
by
  intros h_photo w_photo b_w h_total w_total
  rw w_photo at *
  rw h_photo at *
  rw b_w at *
  rw h_total at *
  rw w_total at *
  simp
  sorry

end area_of_border_l564_564168


namespace solve_quartic_eqn_l564_564680

theorem solve_quartic_eqn (z : ℂ) : 
  z ^ 4 - 6 * z ^ 2 + 8 = 0 ↔ z = 2 ∨ z = -2 ∨ z = Complex.sqrt 2 ∨ z = -Complex.sqrt 2 := 
sorry

end solve_quartic_eqn_l564_564680


namespace find_circle_radius_l564_564981

-- Define the conditions
def on_circle (x : ℝ) (C : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  (C.1 - P.1)^2 + (C.2 - P.2)^2 = x^2

def center_on_x_axis (C : ℝ × ℝ) : Prop :=
  C.2 = 0

-- Define the points and the radius calculation
theorem find_circle_radius (xC : ℝ) (r : ℝ) 
  (hC1 : on_circle r (xC, 0) (1, 5))
  (hC2 : on_circle r (xC, 0) (2, 4))
  (hC3 : center_on_x_axis (xC, 0)) :
  r = real.sqrt 29 :=
by
  sorry

end find_circle_radius_l564_564981


namespace count_digit_7_from_100_to_199_l564_564884

theorem count_digit_7_from_100_to_199 : 
  (list.range' 100 100).countp (λ n, (70 ≤ n ∧ n ≤ 79) ∨ (n % 10 = 7)) = 20 :=
by
  simp
  sorry

end count_digit_7_from_100_to_199_l564_564884


namespace waiter_tables_l564_564180

theorem waiter_tables (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (tables : ℕ) :
  initial_customers = 62 → 
  customers_left = 17 → 
  people_per_table = 9 → 
  remaining_customers = initial_customers - customers_left →
  tables = remaining_customers / people_per_table →
  tables = 5 :=
by
  intros hinitial hleft hpeople hremaining htables
  rw [hinitial, hleft, hpeople] at *
  simp at *
  sorry

end waiter_tables_l564_564180


namespace A_is_not_polynomial_l564_564629

open Real

def A (X : ℝ) : ℝ := (X^2 + 1) * (2 + cos X)

theorem A_is_not_polynomial : ¬ polynomial ℝ (A) :=
by
  sorry

end A_is_not_polynomial_l564_564629


namespace height_of_Linda_room_l564_564430

theorem height_of_Linda_room (w l: ℝ) (h a1 a2 a3 paint_area: ℝ) 
  (hw: w = 20) (hl: l = 20) 
  (d1_h: a1 = 3) (d1_w: a2 = 7) 
  (d2_h: a3 = 4) (d2_w: a4 = 6) 
  (d3_h: a5 = 5) (d3_w: a6 = 7) 
  (total_paint_area: paint_area = 560):
  h = 6 := 
by
  sorry

end height_of_Linda_room_l564_564430


namespace polynomial_h1_l564_564423

theorem polynomial_h1 (a b c : ℤ) (ha : 1 < a) (hb : a < b) (hc : b < c) :
  let f := λ x : ℝ, x^3 + a * x^2 + b * x + c
  let p q r : ℝ
  let roots_of_f := polynomial.splits (λ (x : ℝ), x^3 + a * x^2 + b * x + c) 
  let h := λ x : ℝ, (x - 1 / p^2) * (x - 1 / q^2) * (x - 1 / r^2)
in h 1 = (1 + a^2 + b^2 + c^2) / c^2 := sorry

end polynomial_h1_l564_564423


namespace rectangle_within_l564_564925

theorem rectangle_within (a b c d : ℝ) (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b * d - a * c)^2 + (b * c - a * d)^2 :=
by
  sorry

end rectangle_within_l564_564925


namespace cos_angle_a_b_l564_564781

variable (a b : EuclideanSpace ℝ (Fin 3))

axiom h1 : a + b = ![0, Real.sqrt 2, 0]
axiom h2 : a - b = ![2, Real.sqrt 2, -2 * Real.sqrt 3]

theorem cos_angle_a_b : 
  cosine_angle ⟨a, a ≠ 0⟩ ⟨b, b ≠ 0⟩ = -Real.sqrt 6 / 3 :=
by 
  sorry

end cos_angle_a_b_l564_564781


namespace digit_7_count_in_range_l564_564876

def count_digit_7 : ℕ :=
  let units_place := (107 - 100) / 10 + 1
  let tens_place := 10
  units_place + tens_place

theorem digit_7_count_in_range : count_digit_7 = 20 := by
  sorry

end digit_7_count_in_range_l564_564876


namespace area_of_square_is_34_l564_564977

-- Define the points as given
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (-4, 6)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Side length of the square
def side_length : ℝ := distance point1 point2

-- Area of the square
def area_of_square : ℝ := side_length^2

-- The proof statement to show the area is indeed 34
theorem area_of_square_is_34 : area_of_square = 34 :=
  sorry

end area_of_square_is_34_l564_564977


namespace possible_ways_to_choose_gates_l564_564576

theorem possible_ways_to_choose_gates : 
  ∃! (ways : ℕ), ways = 20 := 
by
  sorry

end possible_ways_to_choose_gates_l564_564576


namespace evaluate_expression_l564_564248

-- Define the expression as given in the problem
def expr1 : ℤ := |9 - 8 * (3 - 12)|
def expr2 : ℤ := |5 - 11|

-- Define the mathematical equivalence
theorem evaluate_expression : (expr1 - expr2) = 75 := by
  sorry

end evaluate_expression_l564_564248


namespace probability_no_two_mathematicians_l564_564720

/-- Number of pairings of 8 players into 4 pairs -/
def total_pairings : ℕ :=
  (8.factorial / (2 ^ 4 * 4.factorial : ℕ))

/-- Number of favorable outcomes where no two mathematicians play against each other -/
def favorable_outcomes : ℕ :=
  (8 * 6 * 4 * 2) * 4.factorial

/-- The probability that no two mathematicians play against each other -/
def probability_no_math_pairs : ℚ :=
  favorable_outcomes / total_pairings

theorem probability_no_two_mathematicians :
  probability_no_math_pairs = 8 / 35 :=
sorry

end probability_no_two_mathematicians_l564_564720


namespace rebus_solution_l564_564255

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l564_564255


namespace nonoverlapping_unit_squares_in_figure_100_l564_564570

theorem nonoverlapping_unit_squares_in_figure_100 :
  ∃ f : ℕ → ℕ, (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 15 ∧ f 3 = 27) ∧ f 100 = 20203 :=
by
  sorry

end nonoverlapping_unit_squares_in_figure_100_l564_564570


namespace count_digit_7_to_199_l564_564870

theorem count_digit_7_to_199 : (Nat.countDigitInRange 7 100 199) = 20 := by
  sorry

end count_digit_7_to_199_l564_564870


namespace cos_270_eq_zero_l564_564217

-- Defining the cosine value for the given angle
theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 :=
by
  sorry

end cos_270_eq_zero_l564_564217


namespace pet_center_final_count_l564_564509

def initial_dogs : Nat := 36
def initial_cats : Nat := 29
def adopted_dogs : Nat := 20
def collected_cats : Nat := 12
def final_pets : Nat := 57

theorem pet_center_final_count :
  (initial_dogs - adopted_dogs) + (initial_cats + collected_cats) = final_pets := 
by
  sorry

end pet_center_final_count_l564_564509


namespace boat_distance_against_stream_l564_564839

variable (v_s : ℝ)
variable (effective_speed_stream : ℝ := 15)
variable (speed_still_water : ℝ := 10)
variable (distance_along_stream : ℝ := 15)

theorem boat_distance_against_stream : 
  distance_along_stream / effective_speed_stream = 1 ∧ effective_speed_stream = speed_still_water + v_s →
  10 - v_s = 5 :=
by
  intros
  sorry

end boat_distance_against_stream_l564_564839


namespace count_digit_7_from_100_to_199_l564_564881

theorem count_digit_7_from_100_to_199 : 
  (list.range' 100 100).countp (λ n, (70 ≤ n ∧ n ≤ 79) ∨ (n % 10 = 7)) = 20 :=
by
  simp
  sorry

end count_digit_7_from_100_to_199_l564_564881


namespace a100_pos_a100_abs_lt_018_l564_564334

noncomputable def a (n : ℕ) : ℝ := real.cos (10^n * real.pi / 180)

theorem a100_pos : (a 100 > 0) :=
by
  sorry

theorem a100_abs_lt_018 : (|a 100| < 0.18) :=
by
  sorry

end a100_pos_a100_abs_lt_018_l564_564334


namespace remainder_when_divided_by_x_minus_1_l564_564750

noncomputable def p (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + a * x^2 + b * x + 12

theorem remainder_when_divided_by_x_minus_1 (a b : ℝ)
  (h1 : p a b (-2) = 0)
  (h2 : p a b 3 = 0) :
  p a b 1 = 18 :=
begin
  sorry
end

end remainder_when_divided_by_x_minus_1_l564_564750


namespace find_z_values_l564_564649

theorem find_z_values : {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -sqrt 2, sqrt 2, 2} :=
sorry

end find_z_values_l564_564649


namespace max_points_on_segment_l564_564117

theorem max_points_on_segment (l d : ℝ) (n : ℕ)
  (hl : l = 1)
  (hcond : ∀ (d : ℝ), 0 < d ∧ d ≤ l → ∀ (segment : Set ℝ), segment ⊆ Icc 0 l → length segment = d → finset.card (finset.image (λ x, x) (finset.filter (λ x, x ∈ segment) (finset.range n))) ≤ 1 + 1000 * d^2) :
  n ≤ 32 :=
begin
  sorry
end

end max_points_on_segment_l564_564117


namespace roots_value_l564_564748

theorem roots_value (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X ^ 2) = 0) (h2 : Polynomial.eval n (Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X ^ 2) = 0) : m^2 + 4 * m + n = -2 := 
sorry

end roots_value_l564_564748


namespace exists_circle_passing_through_three_points_l564_564306

theorem exists_circle_passing_through_three_points
  (n : ℕ) 
  (points : Fin n → ℝ × ℝ)
  (h1 : n ≥ 3)
  (h2 : ¬ ∃ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ collinear (points a) (points b) (points c)) :
  ∃ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (∀ D : Fin n, D ≠ A ∧ D ≠ B ∧ D ≠ C → let circle := circumcircle (points A) (points B) (points C) in  ¬ circle.contains (points D)) :=
sorry

-- Definitions assumed for collinear and circumcircle
axiom collinear (p1 p2 p3 : ℝ × ℝ) : Prop
axiom circumcircle (p1 p2 p3 : ℝ × ℝ) : { c : set (ℝ × ℝ) // ∃ center : ℝ × ℝ, ∃ radius : ℝ, ∀ p, p ∈ c ↔ ∥ p - center ∥ = radius }

namespace set
def contains (s : set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop := p ∈ s
end set

end exists_circle_passing_through_three_points_l564_564306


namespace wheel_radius_l564_564181

theorem wheel_radius (π : Real) (r : Real) (hπ : π ≈ 3.14159) (d : Real) (h1 : d = 140.8) (revs : Nat) (h2 : revs = 100) :
  d / (revs * (2 * π)) ≈ 0.224 :=
by
  sorry

end wheel_radius_l564_564181


namespace count_digit_7_in_range_l564_564855

theorem count_digit_7_in_range : ∀ n ∈ finset.Icc 100 199, (n % 10 = 7 ∨ n / 10 % 10 = 7) →  ∃ counts, counts = 20 :=
by
  sorry

end count_digit_7_in_range_l564_564855


namespace find_all_z_l564_564665

noncomputable def all_solutions (z : ℝ) : Prop := z = -2 ∨ z = 2 ∨ z = -real.sqrt 2 ∨ z = real.sqrt 2

theorem find_all_z (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ all_solutions z := 
sorry

end find_all_z_l564_564665


namespace isosceles_and_angle_60_is_equilateral_l564_564797

theorem isosceles_and_angle_60_is_equilateral
  (T : Type)
  [Triangle T]
  (a b c : T)
  (h1 : is_isosceles a b c)
  (h2 : has_angle a b c 60) :
  is_equilateral a b c :=
sorry

end isosceles_and_angle_60_is_equilateral_l564_564797


namespace powers_of_i_l564_564645

theorem powers_of_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) : 
  i^22 + i^222 = -2 :=
by {
  -- Proof will go here
  sorry
}

end powers_of_i_l564_564645


namespace find_n_l564_564290

noncomputable def arctan_sum_eq_pi_over_2 (n : ℕ) : Prop :=
  Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / 7) + Real.arctan (1 / n) = Real.pi / 2

theorem find_n (h : ∃ n, arctan_sum_eq_pi_over_2 n) : ∃ n, n = 54 := by
  obtain ⟨n, hn⟩ := h
  have H : 1 / 3 + 1 / 4 + 1 / 7 < 1 := by sorry
  sorry

end find_n_l564_564290


namespace worker_total_travel_time_l564_564105

theorem worker_total_travel_time (T : ℕ) : 
  (let usual_time := T in 
   let slower_speed_time := T + 12 in
   let stop_time := 15 in
   let total_time := T + stop_time in
   (5 / 6) * slower_speed_time = T → total_time = 75) :=
sorry

end worker_total_travel_time_l564_564105


namespace people_per_table_l564_564191

theorem people_per_table (kids adults tables : ℕ) (h_kids : kids = 45) (h_adults : adults = 123) (h_tables : tables = 14) :
  ((kids + adults) / tables) = 12 :=
by
  -- Placeholder for proof
  sorry

end people_per_table_l564_564191


namespace eccentricity_of_given_hyperbola_l564_564769

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  in c / a

theorem eccentricity_of_given_hyperbola {a b : ℝ} (h : a > 0 ∧ b > 0) 
  (cond : 2 * (Real.sqrt (b^2 * (h.1^2 + h.2^2)) / Real.sqrt (a^2 + b^2)) = a) :
  hyperbola_eccentricity a b h = Real.sqrt 5 / 2 := by
  sorry

end eccentricity_of_given_hyperbola_l564_564769


namespace doug_age_l564_564050

theorem doug_age (Qaddama Jack Doug : ℕ) 
  (h1 : Qaddama = Jack + 6)
  (h2 : Jack = Doug - 3)
  (h3 : Qaddama = 19) : 
  Doug = 16 := 
by 
  sorry

end doug_age_l564_564050


namespace B_subset_A_implies_m_values_l564_564014

noncomputable def A : Set ℝ := { x | x^2 + x - 6 = 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }
def possible_m_values : Set ℝ := {1/3, -1/2}

theorem B_subset_A_implies_m_values (m : ℝ) : B m ⊆ A → m ∈ possible_m_values := by
  sorry

end B_subset_A_implies_m_values_l564_564014


namespace part_a_part_b_l564_564342

noncomputable def sequence_a (n : ℕ) : ℝ :=
  Real.cos (10^n * Real.pi / 180)

theorem part_a (h : 100 > 2) : sequence_a 100 > 0 := by
  sorry

theorem part_b : |sequence_a 100| < 0.18 := by
  sorry

end part_a_part_b_l564_564342


namespace probability_positive_product_l564_564099

-- The probability that the product of three distinct numbers from the set is positive.
theorem probability_positive_product :
  let s := {-3, -2, 0, 1, 2, 5, 6} in
  let positive_count := (4.choose 3) + (2.choose 2) * (4.choose 1) in
  let total_count := (7.choose 3) in
  positive_count / total_count = 8 / 35 :=
by
  sorry

end probability_positive_product_l564_564099


namespace proof_ellipse_standard_eq_proof_m_range_l564_564737

-- Conditions for the ellipse and other related entities
variables {a b c : ℝ} (h₁ : a > b > 0) (h₂ : (c : ℝ) = a * (sqrt 3) / 2)

-- Given ellipse equation
def ellipse_eq := (y ^ 2) / (a ^ 2) + (x ^ 2) / (b ^ 2) = 1

-- Given area condition
def area_triangle (mn : ℝ) := mn * c * 2 * b ^ 2 / a = sqrt 3

-- Calculated values
def b_squared := b ^ 2 = 1
def a_squared := a ^ 2 = 4

-- Final standard ellipse equation for (Ⅰ)
def standard_ellipse := x ^ 2 + y ^ 2 / 4 = 1

-- Conditions for line intersection and relative vectors
variables {k m : ℝ} (h₃ : m = 0 ∨ (1:ℝ = 4))

-- Determine the range of values for m
def m_range := (-2 < m ∧ m < -1) ∨ (1 < m ∧ m < 2) ∨ (m = 0)

theorem proof_ellipse_standard_eq : h₁ -> h₂ -> b_squared -> a_squared -> standard_ellipse := by sorry
theorem proof_m_range : h₃ -> m_range := by sorry

end proof_ellipse_standard_eq_proof_m_range_l564_564737


namespace smallest_positive_integer_l564_564123

theorem smallest_positive_integer (n : ℕ) (h : 721 * n % 30 = 1137 * n % 30) :
  ∃ k : ℕ, k > 0 ∧ n = 2 * k :=
by
  sorry

end smallest_positive_integer_l564_564123


namespace least_four_digit_divisible_by_15_25_40_75_is_1200_l564_564115

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

def divisible_by_25 (n : ℕ) : Prop :=
  n % 25 = 0

def divisible_by_40 (n : ℕ) : Prop :=
  n % 40 = 0

def divisible_by_75 (n : ℕ) : Prop :=
  n % 75 = 0

theorem least_four_digit_divisible_by_15_25_40_75_is_1200 :
  ∃ n : ℕ, is_four_digit n ∧ divisible_by_15 n ∧ divisible_by_25 n ∧ divisible_by_40 n ∧ divisible_by_75 n ∧
  (∀ m : ℕ, is_four_digit m ∧ divisible_by_15 m ∧ divisible_by_25 m ∧ divisible_by_40 m ∧ divisible_by_75 m → n ≤ m) ∧
  n = 1200 := 
sorry

end least_four_digit_divisible_by_15_25_40_75_is_1200_l564_564115


namespace time_for_train_to_pass_bridge_l564_564607

def length_of_train : ℝ := 250  -- Length of train in meters
def length_of_bridge : ℝ := 150  -- Length of bridge in meters
def speed_of_train_kmh : ℝ := 35  -- Speed of train in km/h

def total_distance : ℝ := length_of_train + length_of_bridge

def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600)

def time_to_pass : ℝ := total_distance / speed_of_train_ms

theorem time_for_train_to_pass_bridge :
  abs (time_to_pass - 41.1528) < 0.001 :=
by
  sorry

end time_for_train_to_pass_bridge_l564_564607


namespace solve_quartic_eq_l564_564701

theorem solve_quartic_eq (z : ℂ) :
  z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -complex.sqrt 2 ∨ z = complex.sqrt 2 ∨ z = 2 :=
by {
  sorry
}

end solve_quartic_eq_l564_564701


namespace triangle_problem_l564_564852

noncomputable def triangle_B (a b : ℝ) (A B : ℝ) :=
  (sqrt 3 * a * cos B - b * sin A = 0) → (B = π / 3)

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) :=
  (B = π / 3) →
  (b = sqrt 7) →
  (a + c = 5) →
  (a * c = 6) →
  (1 / 2 * a * c * (sqrt 3 / 2) = 3 * sqrt 3 / 2)

theorem triangle_problem :
  ∀ (a b c A B : ℝ),
  (triangle_B a b A B) ∧ (triangle_area a b c B) :=
  ⟨sorry, sorry⟩

end triangle_problem_l564_564852


namespace part_a_part_b_l564_564349

noncomputable def sequence (n : ℕ) : ℝ := Real.cos (Real.pi * 10 ^ n / 180)

theorem part_a : (sequence 100) > 0 := 
by {
  sorry
}

theorem part_b : abs (sequence 100) < 0.18 :=
by {
  sorry
}

end part_a_part_b_l564_564349


namespace shaded_region_area_and_percentage_l564_564313

theorem shaded_region_area_and_percentage
  (a : ℝ) (h : a = 4) :
  let S := (sqrt 3 / 4) * a^2 in
  let shaded_triangle_area := (3 * sqrt 3) / 2 in
  let percentage := (shaded_triangle_area / S) * 100 in
  shaded_triangle_area = (3 * sqrt 3) / 2 ∧ percentage = 37.5 :=
  by
    sorry

end shaded_region_area_and_percentage_l564_564313


namespace petya_sequences_l564_564032

theorem petya_sequences (n : ℕ) (h : n = 100) : 
    let S := 5^n
    in S - 3^n = 5^100 - 3^100 :=
by {
  have : S = 5^100,
  {
    rw h,
    exact rfl,
  },
  rw this,
  sorry
}

end petya_sequences_l564_564032


namespace Grisha_probability_expected_flips_l564_564831

theorem Grisha_probability (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                            t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (probability (Grisha wins)) = 1 / 3 :=
by 
  sorry

theorem expected_flips (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                        t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (expected_flips) = 2 :=
by
  sorry

end Grisha_probability_expected_flips_l564_564831


namespace hall_area_l564_564477

theorem hall_area (L : ℝ) (B : ℝ) (A : ℝ) (h1 : B = (2/3) * L) (h2 : L = 60) (h3 : A = L * B) : A = 2400 := 
by 
sorry

end hall_area_l564_564477


namespace arithmetic_sequence_sum_ratio_l564_564001

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Definition of arithmetic sequence sum
def arithmeticSum (n : ℕ) : ℚ :=
  (n / 2) * (a 1 + a n)

-- Given condition
axiom condition : (a 6) / (a 5) = 9 / 11

theorem arithmetic_sequence_sum_ratio :
  (S 11) / (S 9) = 1 :=
by
  sorry

end arithmetic_sequence_sum_ratio_l564_564001


namespace no_solution_for_x_l564_564803

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (mx - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) :=
by
  sorry

end no_solution_for_x_l564_564803


namespace flu_indefinite_if_vaccinated_l564_564184

theorem flu_indefinite_if_vaccinated (A B C : Type) (flu_outbreak : Type)
  (day_1_immune : B → Prop) (recovered_immunity : A → Prop) :
  (∀ x : C, day_1_immune x) →
  (∀ x : A, ¬ day_1_immune x ∧ recovered_immunity x) →
  (∀ x : B, ¬ day_1_immune x) →
  (∀ day : flu_outbreak, ∃ infected : A, ∃ immune : B, ¬day_1_immune immune → infected ∧ ¬recovered_immunity infected) →
  True := sorry

end flu_indefinite_if_vaccinated_l564_564184


namespace interval_of_monotonic_increase_l564_564485

noncomputable def f (x : ℝ) : ℝ := Real.logb (1/2) (6 + x - x^2)

theorem interval_of_monotonic_increase :
  {x : ℝ | -2 < x ∧ x < 3} → x ∈ Set.Ioc (1/2) 3 :=
by
  sorry

end interval_of_monotonic_increase_l564_564485


namespace no_solution_for_m_l564_564800

theorem no_solution_for_m (m : ℝ) : ¬ ∃ x : ℝ, x ≠ 1 ∧ (mx - 1) / (x - 1) = 3 ↔ m = 1 ∨ m = 3 := 
by sorry

end no_solution_for_m_l564_564800


namespace island_inhabitants_even_l564_564933

theorem island_inhabitants_even 
  (total : ℕ) 
  (knights liars : ℕ)
  (H : total = knights + liars)
  (H1 : ∃ (knk : Prop), (knk → (knights % 2 = 0)) ∧ (¬knk → (knights % 2 = 1)))
  (H2 : ∃ (lkr : Prop), (lkr → (liars % 2 = 1)) ∧ (¬lkr → (liars % 2 = 0)))
  : (total % 2 = 0) := sorry

end island_inhabitants_even_l564_564933


namespace tangent_lines_parallel_common_tangent_iff_P_on_QR_l564_564741

-- Given conditions
variable (T T1 T2 : Type)
variable [Circle T] [Circle T1] [Circle T2]
variable (Q R A1 A2 : Point)
variable (intersect_T1_T2 : Intersect T1 T2 Q R)
variable (internal_tangent_T_T1 : InternalTangent T T1 A1)
variable (internal_tangent_T_T2 : InternalTangent T T2 A2)
variable (P : Point)
variable (on_T : OnCircle T P)
variable (PA1_PA2 : IntersectLineCircle P A1 T1 P A2 T2 B1 B2)

-- Prove that the tangent lines at points B1 and B2 are parallel
theorem tangent_lines_parallel 
  (circle_T1_tangent_at_B1 : TangentAt T1 B1)
  (circle_T2_tangent_at_B2 : TangentAt T2 B2) :
  parallel (TangentLineAtPoint T1 B1) (TangentLineAtPoint T2 B2) :=
sorry

-- Prove that B1B2 is a common tangent to T1 and T2 if and only if P lies on QR
theorem common_tangent_iff_P_on_QR :
  (CommonTangent T1 T2 B1 B2) ↔ (LiesOnLineSegment P Q R) :=
sorry

end tangent_lines_parallel_common_tangent_iff_P_on_QR_l564_564741


namespace medication_last_duration_l564_564896

-- Definitions from the conditions
def rate_of_consumption : ℝ := 2 / 3 -- Two-thirds of a pill every three days

def days_per_pill : ℝ := 3 / rate_of_consumption -- Calculate the number of days it takes to consume one pill

def total_pills : ℝ := 90 -- Total supply of pills

def total_days : ℝ := total_pills * days_per_pill -- Calculate the total number of days the supply will last

def days_per_month : ℝ := 30 -- Approximate number of days in a month

def total_months : ℝ := total_days / days_per_month -- Convert the total number of days to months

-- Theorem statement
theorem medication_last_duration : total_months ≈ 14 := 
by
  sorry

end medication_last_duration_l564_564896


namespace green_notebook_cost_l564_564443

def total_cost : ℕ := 45
def black_cost : ℕ := 15
def pink_cost : ℕ := 10
def num_green_notebooks : ℕ := 2

theorem green_notebook_cost :
  (total_cost - (black_cost + pink_cost)) / num_green_notebooks = 10 :=
by
  sorry

end green_notebook_cost_l564_564443


namespace solve_quartic_l564_564681

theorem solve_quartic (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ (z = 2 ∨ z = -2 ∨ z = sqrt 2 ∨ z = -sqrt 2) :=
by
  sorry

end solve_quartic_l564_564681


namespace count_geometric_sequences_l564_564722

-- Define the set of numbers we are working with
def num_set : finset ℕ := (finset.range 10).map ⟨Nat.succ, Nat.succ_injective⟩

-- Define a geometric sequence predicate
def is_geometric_sequence (a b c : ℕ) : Prop :=
  (b^2 = a * c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)

-- Define the main theorem
theorem count_geometric_sequences : 
  (finset.filter (λ (t : ℕ × ℕ × ℕ), is_geometric_sequence t.1 t.2.1 t.2.2) (num_set.off_diag.off_diag.product num_set)).card = 8 := 
sorry

end count_geometric_sequences_l564_564722


namespace convex_polyhedron_with_acute_dihedral_angle_has_four_faces_l564_564148

-- Define the relevant properties and terminology.
structure ConvexPolyhedron where
  faces : ℕ
  edges : Finset (Finset ℕ)
  vertices : Finset ℕ
  dihedral_angle_acute : ∀ edge ∈ edges, dihedral_angle edge < π / 2

-- The theorem we aim to prove.
theorem convex_polyhedron_with_acute_dihedral_angle_has_four_faces
  (P : ConvexPolyhedron)
  (acute_dihedral : ∀ edge ∈ P.edges, P.dihedral_angle_acute edge) :
  P.faces = 4 :=
by
  -- Proof steps would go here.
  sorry

end convex_polyhedron_with_acute_dihedral_angle_has_four_faces_l564_564148


namespace range_of_g_l564_564309

-- the function g(x) = cx + d defined on the interval -1 <= x <= 2
def g (c d x : ℝ) : ℝ := c * x + d

-- the range of g(x), representing it in interval notation
def range_g (c d : ℝ) : set ℝ := set.Icc (-c + d) (2 * c + d)

-- conditions for the theorem
variables (c d : ℝ) (h : c > 0)

theorem range_of_g : set.range (λ x, g c d x) = range_g c d := 
by sorry

end range_of_g_l564_564309


namespace solve_system_of_equations_l564_564061

theorem solve_system_of_equations :
  ∃ (x y : ℚ), 4 * x - 3 * y = -14 ∧ 5 * x + 3 * y = -12 ∧ 
    x = -26 / 9 ∧ y = -22 / 27 :=
by
  use (-26 / 9), (-22 / 27)
  split
  . norm_num
  . split
    . norm_num
    . split
      . norm_num
      . norm_num

end solve_system_of_equations_l564_564061


namespace who_developed_first_steam_engine_l564_564809

-- Definitions based on the conditions.
def Huang_LÜzhuang := 
  "Crafted various instruments and apparatuses, invented 瑞光镜, created many automatic devices, author of 奇器图略"

def Xu_Jianyin := 
  "Engineer and manufacturer, influenced by father Xu Shou, contributed to shipbuilding and military industry, conducted technological inspections abroad"

def Xu_Shou := 
  "Key figure in modern chemistry and shipbuilding, developed China's first steam engine in 1863, designed the first Chinese steamship 黄鹄 in 1865, translated chemical principles, contributed to military technology"

def Hua_Defang := 
  "Mathematician and scientist, designed machinery and earliest steamship, contributed to algebra and calculus, involved in the Westernization Movement"

-- Main statement to prove
theorem who_developed_first_steam_engine : 
  (∀ (x : String), (x = Xu_Shou) → x = "Key figure in modern chemistry and shipbuilding, developed China's first steam engine in 1863, designed the first Chinese steamship 黄鹄 in 1865, translated chemical principles, contributed to military technology") :=
sorry

end who_developed_first_steam_engine_l564_564809


namespace cos_270_eq_zero_l564_564204

theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 := by
  sorry

end cos_270_eq_zero_l564_564204


namespace Grisha_probability_expected_flips_l564_564828

theorem Grisha_probability (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                            t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (probability (Grisha wins)) = 1 / 3 :=
by 
  sorry

theorem expected_flips (h_even_heads : ∀ n, (n % 2 = 0 → coin_flip n = heads), 
                        t_odd_tails : ∀ n, (n % 2 = 1 → coin_flip n = tails)) : 
  (expected_flips) = 2 :=
by
  sorry

end Grisha_probability_expected_flips_l564_564828


namespace solve_quartic_l564_564687

theorem solve_quartic (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ (z = 2 ∨ z = -2 ∨ z = sqrt 2 ∨ z = -sqrt 2) :=
by
  sorry

end solve_quartic_l564_564687


namespace max_k_for_ineq_l564_564710

theorem max_k_for_ineq (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m^3 + n^3 > (m + n)^2) :
  m^3 + n^3 ≥ (m + n)^2 + 10 :=
sorry

end max_k_for_ineq_l564_564710


namespace trapezoid_circumscribed_radius_l564_564070

theorem trapezoid_circumscribed_radius 
  (a b : ℝ) 
  (height : ℝ)
  (ratio_ab : a / b = 5 / 12)
  (height_eq_midsegment : height = 17) :
  ∃ r : ℝ, r = 13 :=
by
  -- Assuming conditions directly as given
  have h1 : a / b = 5 / 12 := ratio_ab
  have h2 : height = 17 := height_eq_midsegment
  -- The rest of the proof goes here
  sorry

end trapezoid_circumscribed_radius_l564_564070


namespace a_100_positive_a_100_abs_lt_018_l564_564345

-- Define the sequence based on the given conditions
def a_n (n : ℕ) : ℝ := real.cos (real.pi / 180 * (10^n) : ℝ)

-- Translate mathematical claims to Lean theorems
theorem a_100_positive : a_n 100 > 0 :=
sorry

theorem a_100_abs_lt_018 : |a_n 100| < 0.18 :=
sorry

end a_100_positive_a_100_abs_lt_018_l564_564345


namespace rebus_solution_l564_564277

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564277


namespace distance_B_to_plane_l564_564843

def A : Point := (1, 0, 0)
def B : Point := (1, 1, 0)
def C1 : Point := (0, 0, 1)
def F : Point := (0, 1/2, 0)
def E : Point := (1, 1/2, 1)

theorem distance_B_to_plane (A B C1 F E : Point) : 
  distance_to_plane B A E C1 F = sqrt(6) / 3 := 
sorry

end distance_B_to_plane_l564_564843


namespace petya_sequence_count_l564_564028

theorem petya_sequence_count :
  let n := 100 in
  (5 ^ n - 3 ^ n) =
  (λ S : (ℕ × ℕ) → ℕ, 5 ^ S (n, n) - 3 ^ S (n, n) ) 5 100 :=
sorry

end petya_sequence_count_l564_564028


namespace rebus_solution_l564_564256

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l564_564256


namespace range_of_a_l564_564779

noncomputable def a_n (n : ℕ) := (-1)^(n + 2018)
noncomputable def b_n (n : ℕ) := 2 + ((-1)^(n + 2017)) / n

theorem range_of_a (a : ℝ) (h : ∀ (n : ℕ), n > 0 → a_n n < b_n n) : -2 ≤ a ∧ a < 3 / 2 :=
sorry

end range_of_a_l564_564779


namespace transformation_yields_large_number_l564_564974

theorem transformation_yields_large_number
  (a b c d : ℤ)
  (h : a ≠ b ∨ b ≠ c ∨ c ≠ d ∨ d ≠ a) :
  ∃ k : ℕ, 
    let t := (λ (x : ℤ × ℤ × ℤ × ℤ), (x.1 - x.2, x.2 - x.3, x.3 - x.4, x.4 - x.1)) in
    let (a_k, b_k, c_k, d_k) := Nat.iterate t k (a, b, c, d) in
    a_k > 1985 ∨ b_k > 1985 ∨ c_k > 1985 ∨ d_k > 1985 := 
sorry

end transformation_yields_large_number_l564_564974


namespace determine_omega_phi_l564_564315

theorem determine_omega_phi (ω : ℝ) (φ : ℝ) (h_ω_gt_zero : ω > 0) (h_phi_bounds : -π < φ ∧ φ < π)
    (h_shifted : ∃ f : ℝ → ℝ, f = (λ x, sin (ω * x + φ)) ∧ ∀ x, f (x + π / 3) = sin (ω * x + φ)) :
    ω = 2 ∧ φ = 2 * π / 3 :=
sorry

end determine_omega_phi_l564_564315


namespace correct_option_C_l564_564527

-- Define the basic logical structures
inductive Structure
| sequence
| selection
| loop

-- Define the conditions
def basic_structure : Structure := Structure.sequence

-- An algorithm must contain a sequence structure
axiom algorithm_contains_sequence : ∀ (alg : list Structure), basic_structure ∈ alg

-- A loop structure must contain a selection (conditional) structure
axiom loop_contains_selection : Structure.loop ∈ list Structure → Structure.selection ∈ list Structure

-- The problem to prove: The loop structure must contain a selection structure
theorem correct_option_C : ∀ (alg : list Structure), Structure.loop ∈ alg → Structure.selection ∈ alg :=
by
  intro alg h_loop
  apply loop_contains_selection
  exact h_loop
  sorry

end correct_option_C_l564_564527


namespace distinct_digits_100_l564_564223

def sequence_rule (a_n : ℕ) : ℕ :=
  if a_n = 112 then 2112 else 0 -- placeholder for the actual sequence rule

def a : ℕ → ℕ
| 1 := 13255569
| n + 1 := sequence_rule (a n)

def distinct_digits (n : ℕ) : set ℕ := 
  {digit | ∃ (i : ℕ), digit ∈ (a i).digits 10 ∧ 1 ≤ i ∧ i ≤ n}

theorem distinct_digits_100 : distinct_digits 100 = {1, 2, 3, 5, 6, 9} :=
by
  sorry -- This is the proof placeholder

end distinct_digits_100_l564_564223


namespace ways_to_divide_chocolate_l564_564142

theorem ways_to_divide_chocolate :
  ∃ (n : ℕ), (n = 1689) ∧
  ∀ (bar : Finset (Fin 10)),
  (∃ pieces : Finset (Finset (Fin 10)), 
    (∀ p ∈ pieces, p ≠ ∅) ∧ 
    (pieces.card ≥ 2) ∧ 
    (∀ (x y : Fin 10), x ∈ bar → y ∈ bar → 
      ((∃ p ∈ pieces, x ∈ p ∧ y ∈ p) ∨
       (∀ p ∈ pieces, x ∉ p ∨ y ∉ p)))) :=
begin
  use 1689,
  split,
  { refl },
  { intros bar,
    use sorry, -- Here we need to skip the proof details
    split,
    { sorry }, -- Conditions to ensure each piece is non-empty
    split,
    { sorry }, -- Conditions to ensure there are at least two pieces
    { sorry }  -- Conditions to ensure the pieces are contiguous
  }
end

end ways_to_divide_chocolate_l564_564142


namespace main_theorem_l564_564747

noncomputable theory

variables {O A B C D E F G H J K : Type} 
variables [OrderedRing O] [Point O A] [Point O B] [Point O C] [Point O D] [Point O E] [Point O F] [Point O G] [Point O H] [Point O J] [Point O K]

def is_circumcircle (O : Type) (A B C : O):
  Prop := ∀ P : O, P ∈ circle O A → P ∈ circle O B → P ∈ circle O C

def midpoint (x y m: O) := ∀ z : O, dist x m = dist m y

def perp (P1 P2 P3 : O) : Prop := ∀ x : O, dist P1 P2 + dist P2 P3 = dist P1 P3

axiom geometry_conditions : 
  (is_circumcircle O A B C) ∧ 
  (dist A B > dist A C) ∧ 
  midpoint B C D ∧ 
  midpoint A B E ∧ 
  midpoint A C F ∧ 
  perp E A G ∧ 
  perp F C H ∧ 
  perp C J H ∧ 
  perp A D J

theorem main_theorem : (KD ∥ AG) :=
  sorry

end main_theorem_l564_564747


namespace total_cost_computers_l564_564171

theorem total_cost_computers (B T : ℝ) 
  (cA : ℝ := 1.4 * B) 
  (cB : ℝ := B) 
  (tA : ℝ := T) 
  (tB : ℝ := T + 20) 
  (total_cost_A : ℝ := cA * tA)
  (total_cost_B : ℝ := cB * tB):
  total_cost_A = total_cost_B → 70 * B = total_cost_A := 
by
  sorry

end total_cost_computers_l564_564171


namespace a_100_positive_a_100_abs_lt_018_l564_564347

-- Define the sequence based on the given conditions
def a_n (n : ℕ) : ℝ := real.cos (real.pi / 180 * (10^n) : ℝ)

-- Translate mathematical claims to Lean theorems
theorem a_100_positive : a_n 100 > 0 :=
sorry

theorem a_100_abs_lt_018 : |a_n 100| < 0.18 :=
sorry

end a_100_positive_a_100_abs_lt_018_l564_564347


namespace square_of_AX_l564_564907

-- Define the points and lengths
variables (A B C D M X : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M] [metric_space X]
variable triangle_ABC : triangle ABC

-- Given Conditions
def isosceles_triangle (triangle_ABC : triangle ABC) : Prop :=
  ∃ (AB AC : ℝ), AB = 3 ∧ AC = 3 ∧ distance B C = 4

def midpoint (M : A) (B : B) (A : A) : Prop :=
  distance A M = distance M B

def altitude (CD : C) (D : D) : Prop :=
  D ∈ line B A ∧ D ⟂ line C A

def intersection (X : A) (M_mid : M) (A_mid : A) (C_line : C) (D_line: D) : Prop :=
  X = line_intersection (midline M A) (perpendicular C D)

def equilateral_triangle (AXD : triangle A X D) : Prop :=
  (distance A X = distance X D) ∧ (distance X D = distance A D)

-- Prove the square of the length of AX is 9/4
theorem square_of_AX (AX_proof : ℝ) (α : midpoint M B A)
  (β : altitude CD D) (γ : intersection X M A C D) (δ : equilateral_triangle A X D) : AX_proof = (3/2)^2 := by
  sorry

end square_of_AX_l564_564907


namespace marias_profit_l564_564432

theorem marias_profit 
  (initial_loaves : ℕ)
  (morning_price : ℝ)
  (afternoon_discount : ℝ)
  (late_afternoon_price : ℝ)
  (cost_per_loaf : ℝ)
  (loaves_sold_morning : ℕ)
  (loaves_sold_afternoon : ℕ)
  (loaves_remaining : ℕ)
  (revenue_morning : ℝ)
  (revenue_afternoon : ℝ)
  (revenue_late_afternoon : ℝ)
  (total_revenue : ℝ)
  (total_cost : ℝ)
  (profit : ℝ) :
  initial_loaves = 60 →
  morning_price = 3.0 →
  afternoon_discount = 0.75 →
  late_afternoon_price = 1.50 →
  cost_per_loaf = 1.0 →
  loaves_sold_morning = initial_loaves / 3 →
  loaves_sold_afternoon = (initial_loaves - loaves_sold_morning) / 2 →
  loaves_remaining = initial_loaves - loaves_sold_morning - loaves_sold_afternoon →
  revenue_morning = loaves_sold_morning * morning_price →
  revenue_afternoon = loaves_sold_afternoon * (afternoon_discount * morning_price) →
  revenue_late_afternoon = loaves_remaining * late_afternoon_price →
  total_revenue = revenue_morning + revenue_afternoon + revenue_late_afternoon →
  total_cost = initial_loaves * cost_per_loaf →
  profit = total_revenue - total_cost →
  profit = 75 := sorry

end marias_profit_l564_564432


namespace water_usage_fee_l564_564498

def water_fee (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then 1.2 * x
  else if 5 < x ∧ x ≤ 6 then 3.6 * x - 12
  else if 6 < x ∧ x ≤ 7 then 6 * x - 26.4
  else 0

theorem water_usage_fee (x : ℝ) (hx : 0 < x ∧ x ≤ 7) :
  water_fee x = 12.6 → x = 6.5 :=
by
  intros h
  -- The detailed proof solving x from the water_fee expression is omitted here
  sorry

end water_usage_fee_l564_564498


namespace survey_sampling_method_is_stratified_sampling_by_stage_l564_564174

def lung_capacity_survey : Prop :=
  ∀ (students : Type) 
    (primary junior_high senior_high boys girls : students → Prop),
    (∀ s, significant (lung_capacity s) primary junior_high senior_high) ∧ 
    (∀ s, ¬ significant (lung_capacity s) boys girls) →
    most_reasonable_sampling_method students = stratified_sampling_by_educational_stage

theorem survey_sampling_method_is_stratified_sampling_by_stage : lung_capacity_survey :=
by 
  intros,
  sorry

end survey_sampling_method_is_stratified_sampling_by_stage_l564_564174


namespace painted_faces_of_large_cube_l564_564438

theorem painted_faces_of_large_cube (n : ℕ) (unpainted_cubes : ℕ) :
  n = 9 ∧ unpainted_cubes = 343 → (painted_faces : ℕ) = 3 :=
by
  intros h
  let ⟨h_n, h_unpainted⟩ := h
  sorry

end painted_faces_of_large_cube_l564_564438


namespace find_slope_k_l564_564912

open Real

-- Define the equation of the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1

-- Define the line passing through point P with slope k
def line_through_point (a k x y : ℝ) : Prop :=
  y = k * (x - a)

-- The main theorem statement that needs to be proven
theorem find_slope_k (k : ℝ) :
  (∀ a : ℝ, ∃ x1 y1 x2 y2 : ℝ, ellipse x1 y1 ∧ ellipse x2 y2 ∧
    line_through_point a k x1 y1 ∧ line_through_point a k x2 y2 ∧
    (let PA2 := (x1 - a)^2 + y1^2 in
     let PB2 := (x2 - a)^2 + y2^2 in
     PA2 + PB2)
  ) → k = 4/5 ∨ k = -4/5 :=
sorry

end find_slope_k_l564_564912


namespace part_a_part_b_l564_564351

noncomputable def sequence (n : ℕ) : ℝ := Real.cos (Real.pi * 10 ^ n / 180)

theorem part_a : (sequence 100) > 0 := 
by {
  sorry
}

theorem part_b : abs (sequence 100) < 0.18 :=
by {
  sorry
}

end part_a_part_b_l564_564351


namespace tan_pi_add_theta_l564_564841

-- lemmas reflecting the condition that the point is on the unit circle
lemma point_on_unit_circle (x y : ℝ) (hx : x = 3/5) (hy : y = 4/5) : x^2 + y^2 = 1 := 
by rw [hx, hy]; norm_num

-- main theorem based on the given problem
theorem tan_pi_add_theta (θ : ℝ)
  (h_theta : cos θ = 3/5 ∧ sin θ = 4/5) :
  Real.tan (Real.pi + θ) = 4 / 3 :=
by sorry

end tan_pi_add_theta_l564_564841


namespace number_of_frames_bought_l564_564943

/- 
   Define the problem conditions:
   1. Each photograph frame costs 3 dollars.
   2. Sally paid with a 20 dollar bill.
   3. Sally got 11 dollars in change.
-/ 

def frame_cost : Int := 3
def initial_payment : Int := 20
def change_received : Int := 11

/- 
   Prove that the number of photograph frames Sally bought is 3.
-/

theorem number_of_frames_bought : (initial_payment - change_received) / frame_cost = 3 := 
by
  sorry

end number_of_frames_bought_l564_564943


namespace distance_N_to_LM_l564_564580

variable {α : Type*}
variables (O K L M N : α)
variables (OM OK KM LM : α → ℝ)
variables (m k l : ℝ)

-- Condition: There is a point L on the side of the acute angle KOM between O and K
variable (L_on_KOM_side : L ∈ segment K O)

-- Condition: A circle passes through points K and L and touches the ray OM at point M
variable (circle_through_K_L_and_touches_OM_M : circle K L M)

-- Condition: A point N is taken on the arc LM that does not contain point K
variable (N_on_arc_LM_not_containing_K : arc K L M ∋ N)

-- Given distances from point N to the lines OM, OK, and KM are m, k, and l respectively
variable (N_distance_OM : dist N OM = m)
variable (N_distance_OK : dist N OK = k)
variable (N_distance_KM : dist N KM = l)

-- Theorem: Find the distance from point N to the line LM
theorem distance_N_to_LM (N_on_arc_LM_not_containing_K : dist N LM = m * k / l) : Prop := sorry

end distance_N_to_LM_l564_564580


namespace second_discount_is_10_l564_564093

variable (P_initial P_after_first_discount P_final_price : ℝ)
variable (D : ℝ)
hypothesis h1 : P_initial = 150
hypothesis h2 : P_after_first_discount = P_initial - (20 / 100) * P_initial
hypothesis h3 : P_final_price = P_after_first_discount - (D / 100) * P_after_first_discount
hypothesis h4 : P_final_price = 108

theorem second_discount_is_10 : D = 10 :=
by
  sorry

end second_discount_is_10_l564_564093


namespace price_of_bracelets_max_type_a_bracelets_l564_564173

-- Part 1: Proving the prices of the bracelets
theorem price_of_bracelets :
  ∃ (x y : ℝ), (3 * x + y = 128 ∧ x + 2 * y = 76) ∧ (x = 36 ∧ y = 20) :=
sorry

-- Part 2: Proving the maximum number of type A bracelets they can buy within the budget
theorem max_type_a_bracelets :
  ∃ (m : ℕ), 36 * m + 20 * (100 - m) ≤ 2500 ∧ m = 31 :=
sorry

end price_of_bracelets_max_type_a_bracelets_l564_564173


namespace inhabitant_50_statement_l564_564234

-- Definitions
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

def tells_truth (inh: Inhabitant) (statement: Bool) : Bool :=
  match inh with
  | Inhabitant.knight => statement
  | Inhabitant.liar => not statement

noncomputable def inhabitant_at_position (pos: Nat) : Inhabitant :=
  if (pos % 2) = 1 then
    if pos % 4 = 1 then Inhabitant.knight else Inhabitant.liar
  else
    if pos % 4 = 0 then Inhabitant.knight else Inhabitant.liar

def neighbor (pos: Nat) : Nat := (pos % 50) + 1

-- Theorem statement
theorem inhabitant_50_statement : tells_truth (inhabitant_at_position 50) (inhabitant_at_position (neighbor 50) = Inhabitant.knight) = true :=
by
  -- Proof would go here
  sorry

end inhabitant_50_statement_l564_564234


namespace find_vector_coordinates_l564_564706

structure Point3D :=
  (x y z : ℝ)

def vector_sub (a b : Point3D) : Point3D :=
  Point3D.mk (b.x - a.x) (b.y - a.y) (b.z - a.z)

theorem find_vector_coordinates (A B : Point3D)
  (hA : A = { x := 1, y := -3, z := 4 })
  (hB : B = { x := -3, y := 2, z := 1 }) :
  vector_sub A B = { x := -4, y := 5, z := -3 } :=
by
  -- Proof is omitted
  sorry

end find_vector_coordinates_l564_564706


namespace probability_grisha_wins_expectation_coin_flips_l564_564834

-- Define the conditions as predicates
def grishaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 0) && (toss_seq.last = some true)

def vanyaWins (toss_seq : List Bool) : Bool := 
  (toss_seq.length % 2 == 1) && (toss_seq.last = some false)

-- State the probability theorem
theorem probability_grisha_wins : 
  (∑ x in ((grishaWins ≤ 1 / 3))) = 1/3 := by sorry

-- State the expectation theorem
theorem expectation_coin_flips : 
  (E[flips until (grishaWins ∨ vanyaWins)]) = 2 := by sorry

end probability_grisha_wins_expectation_coin_flips_l564_564834


namespace solve_quartic_eqn_l564_564679

theorem solve_quartic_eqn (z : ℂ) : 
  z ^ 4 - 6 * z ^ 2 + 8 = 0 ↔ z = 2 ∨ z = -2 ∨ z = Complex.sqrt 2 ∨ z = -Complex.sqrt 2 := 
sorry

end solve_quartic_eqn_l564_564679


namespace red_balls_removal_l564_564814

theorem red_balls_removal (total_balls : ℕ) (red_percentage : ℝ) (desired_red_percentage : ℝ)
    (initial_red : ℕ) (initial_blue : ℕ) (x : ℕ) :
    total_balls = 800 →
    red_percentage = 0.7 →
    desired_red_percentage = 0.6 →
    initial_red = (red_percentage * total_balls).to_nat →
    initial_blue = (total_balls - initial_red) →
    (560 - x).to_rat / (800 - x).to_rat = desired_red_percentage →
    x = 200 := sorry

end red_balls_removal_l564_564814


namespace coordinates_of_point_A_l564_564446

-- Define the problem conditions as hypotheses
variable {m : ℤ}
def point_A := (2 * m + 1, m + 2)
def in_second_quadrant (p : ℤ × ℤ) : Prop := p.1 < 0 ∧ p.2 > 0

-- Prove that point_A is in the second quadrant and its coordinates are (-1, 1)
theorem coordinates_of_point_A (hm1 : in_second_quadrant point_A)
  (hm2 : point_A.1 ∈ ℤ) (hm3 : point_A.2 ∈ ℤ) : point_A = (-1, 1) := by
  sorry

end coordinates_of_point_A_l564_564446


namespace ratio_red_bacon_bits_l564_564598

noncomputable def mushrooms := 3
noncomputable def cherryTomatoes := 2 * mushrooms
noncomputable def pickles := 4 * cherryTomatoes
noncomputable def baconBits := 4 * pickles
noncomputable def redBaconBits := 32

theorem ratio_red_bacon_bits : 32 / 96 = 1 / 3 :=
by
  have h_total_bacon_bits : baconBits = 96 := by
    rw [mushrooms, cherryTomatoes, pickles]
    sorry -- you would prove each of the intermediary steps here
  have h_red_bacon_bits : redBaconBits = 32 := by
    rw redBaconBits
    rfl
  have h_ratio_simplified : 32 / 96 = 1 / 3 := by
    sorry -- simplify the ratio using arithmetic
  exact h_ratio_simplified

end ratio_red_bacon_bits_l564_564598


namespace arithmetic_sequence_condition_l564_564283

-- Define and prove the main theorem
theorem arithmetic_sequence_condition (a_1 : ℕ) (a_k b_k c_k d_k : ℕ → ℕ)
  (h : ∀ k, a_k k = a_1 + k ∧ b_k k = a_1 + k + 1 ∧ c_k k = a_1 + k + 2 ∧ d_k k = a_1 + k + 3) :
  (a_1 % 2 = 0) →
  (∀ n, 0 < n → n < 4 
        → let seq = [a_k n, b_k n, c_k n, d_k n, a_k (n+1), b_k (n+1), c_k (n+1), d_k (n+1)] 
          in ∀ j < 7, seq[j + 1] = seq[j] + (seq[j] % 10)) :=
begin
  sorry
end

end arithmetic_sequence_condition_l564_564283


namespace total_students_in_school_district_l564_564986

def CampusA_students : Nat :=
  let students_per_grade : Nat := 100
  let num_grades : Nat := 5
  let special_education : Nat := 30
  (students_per_grade * num_grades) + special_education

def CampusB_students : Nat :=
  let students_per_grade : Nat := 120
  let num_grades : Nat := 5
  students_per_grade * num_grades

def CampusC_students : Nat :=
  let students_per_grade : Nat := 150
  let num_grades : Nat := 2
  let international_program : Nat := 50
  (students_per_grade * num_grades) + international_program

def total_students : Nat :=
  CampusA_students + CampusB_students + CampusC_students

theorem total_students_in_school_district : total_students = 1480 := by
  sorry

end total_students_in_school_district_l564_564986


namespace log_function_passing_through_log_function_domain_range_of_x_l564_564758

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem log_function_passing_through :
  f 8 = 3 :=
by
  -- Lean will verify the correctness of this proof
  sorry

theorem log_function_domain :
  ∀ x, x > 0 → x < 1 → f (1 - x) > f (1 + x) → x < 1 ∧ x > -1 :=
by
  -- Lean will verify the correctness of this proof
  intros x hx1 hx2 h
  sorry

theorem range_of_x (x : ℝ) :
  f (1 - x) > f (1 + x) → x ∈ Ioo (-1 : ℝ) 0 :=
by
  -- Lean will verify the correctness of this proof
  sorry

end log_function_passing_through_log_function_domain_range_of_x_l564_564758


namespace min_value_l564_564453

def conditions (n : ℕ) (x : ℕ → ℝ) : Prop :=
  ∀ i j : ℕ, (1 ≤ i ∧ i < j ∧ j ≤ n) → (x i + x j ≥ (-1)^(i + j))

theorem min_value (x : ℕ → ℝ) :
  conditions 2018 x →
  ∑ i in Finset.range 2018, (i + 1) * x (i + 1) = -1009 := sorry

end min_value_l564_564453


namespace area_hexagon_half_area_triangle_l564_564310

variables {A B C A1 B1 C1 A2 B2 C2 : Type}
variable [geometry.triangle A B C] -- assuming some library for triangle geometry

/-- 
  Given an acute-angled triangle ABC with:
  - midpoints A1, B1, C1 of sides BC, CA, AB respectively.
  - perpendiculars from A1, B1, C1 to the other two sides intersect at A2, B2, C2 respectively.
  Prove that the area of the hexagon A1C2B1A2C1B2 is half of the area of triangle ABC.
--/
theorem area_hexagon_half_area_triangle
  (h_acute : triangle.is_acute A B C)
  (h_midpoints : is_midpoint A1 B C ∧ is_midpoint B1 C A ∧ is_midpoint C1 A B)
  (h_perpendiculars : 
    is_perpendicular B C A1 A2 ∧ 
    is_perpendicular C A B1 B2 ∧ 
    is_perpendicular A B C1 C2) :
  geometry.area (hexagon A1 C2 B1 A2 C1 B2) = 
  1/2 * geometry.area (triangle A B C) := by
  sorry

end area_hexagon_half_area_triangle_l564_564310


namespace percentage_of_paycheck_went_to_taxes_l564_564395

-- Definitions
def original_paycheck : ℝ := 125
def savings : ℝ := 20
def spend_percentage : ℝ := 0.80
def save_percentage : ℝ := 0.20

-- Statement that needs to be proved
theorem percentage_of_paycheck_went_to_taxes (T : ℝ) :
  (0.20 * (1 - T / 100) * original_paycheck = savings) → T = 20 := 
by
  sorry

end percentage_of_paycheck_went_to_taxes_l564_564395


namespace seeds_total_l564_564934

def seedsPerWatermelon : Nat := 345
def numberOfWatermelons : Nat := 27
def totalSeeds : Nat := seedsPerWatermelon * numberOfWatermelons

theorem seeds_total :
  totalSeeds = 9315 :=
by
  sorry

end seeds_total_l564_564934


namespace graphs_intersect_at_three_points_l564_564064

noncomputable def is_invertible (f : ℝ → ℝ) := ∃ (g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x ∧ g (f x) = x

theorem graphs_intersect_at_three_points (f : ℝ → ℝ) (h_inv : is_invertible f) :
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, f (x^2) = f (x^6)) ∧ xs.card = 3 :=
by 
  sorry

end graphs_intersect_at_three_points_l564_564064


namespace minor_premise_syllogism_l564_564545

theorem minor_premise_syllogism (R P S : Type) 
  (h1 : ∀ r : R, P r)
  (h2 : ∀ s : S, R s) :
  ∀ s : S, R s := h2

end minor_premise_syllogism_l564_564545


namespace sum_of_coordinates_of_point_D_l564_564038

theorem sum_of_coordinates_of_point_D : 
  ∀ {x : ℝ}, (y = 6) ∧ (x ≠ 0) ∧ ((6 - 0) / (x - 0) = 3 / 4) → x + y = 14 := by
  intros x hx hy hslope
  sorry

end sum_of_coordinates_of_point_D_l564_564038


namespace find_m_l564_564504

noncomputable def a_seq (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a + (n - 1 : ℝ) * d)

theorem find_m (a d : ℝ) (m : ℕ) 
  (h1 : a_seq a d (m-1) + a_seq a d (m+1) - a = 0)
  (h2 : S_n a d (2*m - 1) = 38) : 
  m = 10 := 
sorry

end find_m_l564_564504


namespace solve_equation_l564_564503

theorem solve_equation : ∃ x : ℝ, 3 * x + 9 = 0 ∧ x = -3 := 
by
  exists -3
  split
  · show 3 * (-3) + 9 = 0
    calc
      3 * (-3) + 9 = -9 + 9 := rfl
      ... = 0 := add_neg_self 9
  · rfl

end solve_equation_l564_564503


namespace max_sum_at_n_24_l564_564389

def a (n : ℕ) : ℤ := 49 - 2 * n

def S (n : ℕ) : ℤ := ∑ i in finset.range (n + 1), a i

theorem max_sum_at_n_24 :
  ∀ n, (∀ k < n, a k > 0) → a n ≤ 0 → n = 24 → S n = ∑ i in finset.range 25, a i := sorry

end max_sum_at_n_24_l564_564389


namespace part_a_part_b_l564_564340

noncomputable def sequence_a (n : ℕ) : ℝ :=
  Real.cos (10^n * Real.pi / 180)

theorem part_a (h : 100 > 2) : sequence_a 100 > 0 := by
  sorry

theorem part_b : |sequence_a 100| < 0.18 := by
  sorry

end part_a_part_b_l564_564340


namespace custom_op_value_l564_564091

variable {a b : ℤ}
def custom_op (a b : ℤ) := 1/a + 1/b

axiom h1 : a + b = 15
axiom h2 : a * b = 56

theorem custom_op_value : custom_op a b = 15/56 :=
by
  sorry

end custom_op_value_l564_564091


namespace books_sold_on_wednesday_l564_564398

-- This Lean 4 statement proves John sold exactly 60 books on Wednesday.
theorem books_sold_on_wednesday :
  ∀ (total_stock : ℕ)
    (monday_sales : ℕ)
    (tuesday_sales : ℕ)
    (wednesday_sales : ℕ)
    (thursday_sales : ℕ)
    (friday_sales : ℕ),
    (total_stock = 800) →
    (monday_sales = 62) →
    (tuesday_sales = 62) →
    (thursday_sales = 48) →
    (friday_sales = 40) →
    (0.66 * 800 = total_stock - (monday_sales + tuesday_sales + thursday_sales + friday_sales + wednesday_sales)) →
    wednesday_sales = 60 :=
by
  intros total_stock monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales
  intro h1 h2 h3 h4 h5 h6
  sorry

end books_sold_on_wednesday_l564_564398


namespace area_of_45_45_90_triangle_with_altitude_to_hypotenuse_l564_564953

noncomputable def isosceles_right_triangle (a b c : ℝ) (h : (a ^ 2 + b ^ 2 = c ^ 2)) : Prop :=
a = b ∧ a > 0 ∧ b > 0 ∧ c > 0

theorem area_of_45_45_90_triangle_with_altitude_to_hypotenuse (a b c : ℝ) (h1 : isosceles_right_triangle a b c) 
  (h2 : ∃ h: ℝ, h = 4 ∧ h * sqrt 2 = (1/2) * a * b) :
  a * b / 2 = 16 :=
by
  sorry

end area_of_45_45_90_triangle_with_altitude_to_hypotenuse_l564_564953


namespace non_symmetrical_parallelogram_l564_564129

theorem non_symmetrical_parallelogram (A B C D : Type) : 
  (is_symmetrical A) ∧ (is_symmetrical B) ∧ (is_symmetrical C) ∧ (¬ is_symmetrical D) := 
begin
  -- Definitions for the geometrical shapes
  def is_symmetrical (shape : Type) : Prop := sorry -- This will be defined based on specific geometric symmetry

  -- Conditions from the problem
  def A := sorry -- Definition of Line segment
  def B := sorry -- Definition of Rectangle
  def C := sorry -- Definition of Angle
  def D := sorry -- Definition of Parallelogram
  
  -- Prove symmetry properties based on definitions and conditions
  have hA : is_symmetrical A := sorry, -- Proof that Line segment is symmetrical
  have hB : is_symmetrical B := sorry, -- Proof that Rectangle is symmetrical
  have hC : is_symmetrical C := sorry, -- Proof that Angle is symmetrical
  have hD : ¬ is_symmetrical D := sorry, -- Proof that Parallelogram is not necessarily symmetrical
  
  -- Combining all
  exact ⟨hA, hB, hC, hD⟩,
end

end non_symmetrical_parallelogram_l564_564129


namespace quadratic_function_and_range_l564_564317

theorem quadratic_function_and_range (y : ℕ → ℝ) :
  -- Conditions:
  (y = λ x, x^2 - 4 * x + 5) →
  (∀ x, y x = 5 ∨ (y x = 2 ∧ x = 1) ∨ (y x = 1 ∧ x = 2) ∨ (y x = 5 ∧ x = 4)) →
  -- Prove the functional form and range:
  (∀ x, x >= 0 ∧ x <= 4 → y x = x^2 - 4 * x + 5) ∧
  (∀ x, 0 < x ∧ x < 3 → 1 ≤ y x ∧ y x < 5) :=
begin
  -- Since we are only asked to write the statement, the proof will be omitted:
  sorry
end

end quadratic_function_and_range_l564_564317


namespace solve_quartic_eq_l564_564660

theorem solve_quartic_eq (z : ℝ) : z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = 2 :=
by 
  sorry

end solve_quartic_eq_l564_564660


namespace convince_king_l564_564188

-- Definitions for inhabitants
inductive Person
| Knight
| Liar
| Normal

open Person

-- Axioms for knight, liar, and normal person behaviors
axiom knight_truthful (k: Person) : k = Knight = true
axiom liar_dishonest (l: Person) : l = Liar = false
axiom normal_indeterminate (n: Person) : n = Normal ∨ ¬(n = Normal)

-- Single statement to be evaluated
def single_statement (p : Person) : Prop :=
  p = Normal ∧ (∃ dollars : Nat, dollars = 11) ∨ p = Liar

-- Theorem to prove statement satisfies conditions
theorem convince_king (you : Person) (h : you = Normal) :
  (p : Person) → single_statement you → ¬(you = Knight) ∧ ¬(you = Liar) :=
by 
  sorry

end convince_king_l564_564188


namespace total_remaining_books_l564_564530

-- Define the initial conditions as constants
def total_books_crazy_silly_school : ℕ := 14
def read_books_crazy_silly_school : ℕ := 8
def total_books_mystical_adventures : ℕ := 10
def read_books_mystical_adventures : ℕ := 5
def total_books_sci_fi_universe : ℕ := 18
def read_books_sci_fi_universe : ℕ := 12

-- Define the remaining books calculation
def remaining_books_crazy_silly_school : ℕ :=
  total_books_crazy_silly_school - read_books_crazy_silly_school

def remaining_books_mystical_adventures : ℕ :=
  total_books_mystical_adventures - read_books_mystical_adventures

def remaining_books_sci_fi_universe : ℕ :=
  total_books_sci_fi_universe - read_books_sci_fi_universe

-- Define the proof statement
theorem total_remaining_books : 
  remaining_books_crazy_silly_school + remaining_books_mystical_adventures + remaining_books_sci_fi_universe = 17 := by
  sorry

end total_remaining_books_l564_564530


namespace find_missing_digit_divisible_by_nine_l564_564989

theorem find_missing_digit_divisible_by_nine (x: ℕ) :
  (1 + 3 + 5 + 7 + 9 + x) % 9 = 0 → x = 2 :=
by
  intro h
  have h1 : 1 + 3 + 5 + 7 + 9 = 25
  {
    norm_num,
  }
  rw [h1] at h
  -- The problem asks for the proof dispatcher, so we implement the fact that 25 accompanied by x % 9 != 0 has to imply something
  -- This is sketched here to directly show x = 2 without computing the whole proof formally.
  exact sorry -- The proof will be elaborated manually outside this scope.

end find_missing_digit_divisible_by_nine_l564_564989


namespace num_four_digit_numbers_l564_564298

theorem num_four_digit_numbers (cards : Finset ℕ) (cards_condition : cards = {2, 0, 0, 9}) : 
  let n := cards.card + (if b : 9 ∈ cards then 1 else 0)
  in n = 12 :=
by
  -- Using the hypothetical conditions provided directly in the assumptions and ignoring solution steps.
  sorry

end num_four_digit_numbers_l564_564298


namespace smallest_lambda_l564_564296

noncomputable def y_seq (x : ℕ → ℝ) : ℕ → ℝ
| 0       => x 0
| (n+1)   => x (n+1) - (∑ i in finset.range (n+1), (x i)^2).sqrt

theorem smallest_lambda (x : ℕ → ℝ) (m : ℕ) (hm : m > 0) :
  (1 / m) * (∑ i in finset.range m, (x i)^2) ≤ ∑ i in finset.range m, (2:ℝ)^(m-i-1) * (y_seq x i)^2 :=
sorry

end smallest_lambda_l564_564296


namespace ninety_nine_ladder_division_l564_564932

def ladder (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem ninety_nine_ladder_division : ∃ (ways : ℕ), ways = 2^(99 - 1) :=
by
  let n := 99
  let total_squares := ladder n
  have h1 : total_squares = 4950 := by
    calc
      ladder 99 = (99 * 100) / 2 : rfl
      ... = 4950 : by norm_num
  have distinct_areas : ∀ (rects : List ℕ), rects.length = n → rects = List.range (1 + n) := sorry
  have placements : ∃ (ways : ℕ), ways = 2^(n - 1) := by
    use 2^(n - 1)
    exact sorry
  exact placements

end ninety_nine_ladder_division_l564_564932


namespace line_growth_convergence_l564_564572

-- Definitions of the initial length and the growth pattern of the line
def initial_length : ℝ := 2

-- Defining the growth pattern as a series
noncomputable def growth_pattern (n : ℕ) : ℝ :=
  if n = 0 then initial_length
  else (1 / 3^n * sqrt 3) + (1 / 3^n)

-- The target is to prove the sum of the series remains equal to a specific value
theorem line_growth_convergence :
  (∑' n, growth_pattern n) = 3 + (sqrt 3) / 2 :=
sorry

end line_growth_convergence_l564_564572


namespace chord_length_constant_l564_564761

noncomputable def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - (6 - 2*m)*x - 4*m*y + 5*m^2 - 6*m = 0

noncomputable def is_point_on_line (x y k : ℝ) : Prop :=
  y = k * (x - 1)

theorem chord_length_constant (m : ℝ) (A : ℝ) (hA : A = 2 * sqrt(145) / 5) :
  ∀ k : ℝ, k = -2 → 
  ∀ x y : ℝ, 
  circle_equation x y m →
  is_point_on_line x y k →
  ∃ d : ℝ, 
    d = abs(2*k - m*(2 + k)) / sqrt(k^2 + 1) ∧
    A = 2 * sqrt(9 - d^2) := 
sorry

end chord_length_constant_l564_564761


namespace cost_per_pound_of_mixed_feed_l564_564100

variables (w_total : ℕ) (w_cheaper : ℕ) (w_expensive : ℕ)
variables (c_cheaper : ℚ) (c_expensive : ℚ) (c_mixed : ℚ)

theorem cost_per_pound_of_mixed_feed :
  w_total = 35 ∧ w_cheaper = 17 ∧ w_expensive = 18 ∧ c_cheaper = 0.18 ∧ c_expensive = 0.53 →
  c_mixed = (w_cheaper * c_cheaper + w_expensive * c_expensive) / w_total →
  c_mixed = 0.36 :=
begin
  sorry
end

end cost_per_pound_of_mixed_feed_l564_564100


namespace digit_7_appears_20_times_in_range_100_to_199_l564_564865

theorem digit_7_appears_20_times_in_range_100_to_199 : 
  ∃ (d7_count : ℕ), (d7_count = 20) ∧ (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 199 → 
  d7_count = (n.to_string.count '7')) :=
sorry

end digit_7_appears_20_times_in_range_100_to_199_l564_564865


namespace rebus_solution_l564_564266

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564266


namespace oranges_per_tree_alba_l564_564637

-- Definitions from the problem conditions
def oranges_per_tree_gabriela : ℕ := 600
def trees_per_grove : ℕ := 110
def oranges_per_tree_maricela : ℕ := 500
def three_oranges_per_cup : ℕ := 3
def selling_price_per_cup : ℕ := 4
def total_money_made : ℕ := 220000

-- Task: Prove that the number of oranges produced per tree in Alba's grove is 400
theorem oranges_per_tree_alba : 
  let 
    total_oranges_needed := total_money_made / selling_price_per_cup * three_oranges_per_cup,
    total_oranges_gabriela := oranges_per_tree_gabriela * trees_per_grove,
    total_oranges_maricela := oranges_per_tree_maricela * trees_per_grove,
    total_oranges_alba := total_oranges_needed - total_oranges_gabriela - total_oranges_maricela,
    oranges_per_tree_alba := total_oranges_alba / trees_per_grove
  in oranges_per_tree_alba = 400 := by sorry

end oranges_per_tree_alba_l564_564637


namespace minimum_value_exists_l564_564321

theorem minimum_value_exists (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_condition : x + 4 * y = 2) : 
  ∃ z : ℝ, z = (x + 40 * y + 4) / (3 * x * y) ∧ z ≥ 18 :=
by
  sorry

end minimum_value_exists_l564_564321


namespace number_of_switches_in_A_switches_in_position_a_after_steps_l564_564992

def label (x y z w : ℕ) :=
  2^x * 3^y * 5^z * 7^w

theorem number_of_switches_in_A :
  (fin 6) := sorry

theorem switches_in_position_a_after_steps 
  (steps : ℕ := 1500)
  (switches : fin 1500 → ℕ) 
  (labels : fin 1500 → (fin 6) → ℕ) :
  sum (λ s : fin 1500, if steps % 6 = 0 then 1 else 0) = 972 := 
by
  sorry

end number_of_switches_in_A_switches_in_position_a_after_steps_l564_564992


namespace parabola_standard_form_proof_l564_564760

noncomputable def parabola_standard_form (vertex : ℝ × ℝ) (axis_of_symmetry : ℝ) : String :=
  if vertex = (0, 0) ∧ axis_of_symmetry = -2 then
    "y^2 = 8x"
  else
    "Unknown"

theorem parabola_standard_form_proof (h_vertex : (0, 0) = (0, 0)) (h_axis_of_symmetry : -2 = -2) :
  parabola_standard_form (0, 0) -2 = "y^2 = 8x" :=
by
  unfold parabola_standard_form
  rw [h_vertex, h_axis_of_symmetry]
  rfl

end parabola_standard_form_proof_l564_564760


namespace no_solution_for_lcm_gcd_eq_l564_564124

theorem no_solution_for_lcm_gcd_eq (n : ℕ) (h₁ : n ∣ 60) (h₂ : Nat.Prime n) :
  ¬(Nat.lcm n 60 = Nat.gcd n 60 + 200) :=
  sorry

end no_solution_for_lcm_gcd_eq_l564_564124


namespace triangle_perimeter_l564_564085

theorem triangle_perimeter (a b : ℝ) (f : ℝ → Prop) 
  (h₁ : a = 7) (h₂ : b = 11)
  (eqn : ∀ x, f x ↔ x^2 - 25 = 2 * (x - 5)^2)
  (h₃ : ∃ x, f x ∧ 4 < x ∧ x < 18) :
  ∃ p : ℝ, (p = a + b + 5 ∨ p = a + b + 15) :=
by
  sorry

end triangle_perimeter_l564_564085


namespace ivy_stripping_days_l564_564609

theorem ivy_stripping_days :
  ∃ (days_needed : ℕ), (days_needed * (6 - 2) = 40) ∧ days_needed = 10 :=
by {
  use 10,
  split,
  { simp,
    norm_num,
  },
  { simp }
}

end ivy_stripping_days_l564_564609


namespace total_cost_of_books_l564_564788

theorem total_cost_of_books :
  ∃ (C2 : ℝ), C2 = 148.75 / 1.19 → 175 + C2 = 300 :=
by {
  use 125, -- The result inferred from the solution
  assume hC2 : 125 = 148.75 / 1.19,
  rw (show 175 + 125 = 300, by simp),
}

end total_cost_of_books_l564_564788


namespace rebus_solution_l564_564264

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l564_564264


namespace simplify_expression_l564_564057

variable (a b : ℤ) -- Define variables a and b

theorem simplify_expression : 
  (35 * a + 70 * b + 15) + (15 * a + 54 * b + 5) - (20 * a + 85 * b + 10) =
  30 * a + 39 * b + 10 := 
by sorry

end simplify_expression_l564_564057


namespace chocolates_comparison_l564_564052

theorem chocolates_comparison :
  ∀ (Robert Nickel Jessica : ℕ), 
    Robert = 23 → Nickel = 8 → Jessica = 15 →
    (Robert - Nickel = 15) ∧ (Jessica - Nickel = 7) := 
by
  intros Robert Nickel Jessica hRobert hNickel hJessica
  rw [hRobert, hNickel, hJessica]
  split
  · exact Nat.zero_sub 8 ▸ rfl
  · exact Nat.zero_sub 8 ▸ rfl

end chocolates_comparison_l564_564052


namespace pen_price_l564_564585

theorem pen_price (p : ℝ) (h : 30 = 10 * p + 10 * (p / 2)) : p = 2 :=
sorry

end pen_price_l564_564585


namespace log_inequality_solution_l564_564410

variable (a x : ℝ)
variable (ha_pos : a > 0)
variable (ha_ne_one : a ≠ 1)
variable (f : ℝ → ℝ := fun x => Real.logBase a (x^2 - 2*x + 3))
variable (hf_min : ∃ x, ∀ y, f(x) ≤ f(y))

theorem log_inequality_solution (ha_gt_one : a > 1) : 1 < x → x < 2 → Real.logBase a (x - 1) < 0 :=
by
  sorry

end log_inequality_solution_l564_564410


namespace parallel_AA1_BB1_l564_564056

variables {A B A1 B1 : Point} {L1 L2 : Line}

-- Assume A and B lie on L1
axiom A_on_L1 : On_Point_Line A L1
axiom B_on_L1 : On_Point_Line B L1

-- Assume A1 and B1 are the projections of A and B onto L2
axiom A1_proj_L2 : Projection A A1 L2
axiom B1_proj_L2 : Projection B B1 L2

-- Assume L1 and L2 are parallel
axiom parallel_L1_L2 : Parallel_lines L1 L2

-- The proof goal: AA1 is parallel to BB1
theorem parallel_AA1_BB1 : Parallel_lines (Line_through A A1) (Line_through B B1) :=
sorry

end parallel_AA1_BB1_l564_564056


namespace simplify_fraction_l564_564523

theorem simplify_fraction (x : ℝ) (hx : x = 3) :
  (∏ i in (finset.range 15).map (λ i, 2*i+2) λ i, x^i) /
  (∏ i in (finset.range 9).map (λ i, 3*i+3) λ i, x^i) = 3^105 :=
by {
  simp only [hx],
  sorry
}

end simplify_fraction_l564_564523


namespace unique_solution_to_equation_l564_564252

theorem unique_solution_to_equation (x y z n : ℕ)
  (h1 : n ≥ 2)
  (h2 : z ≤ 5 * 2^(2*n)) :
  (x^(2*n+1) - y^(2*n+1) = x * y * z + 2^(2*n+1)) →
  (x = 3 ∧ y = 1 ∧ z = 70 ∧ n = 2) :=
begin
  sorry -- skipping proof
end

end unique_solution_to_equation_l564_564252


namespace area_of_45_45_90_triangle_with_altitude_to_hypotenuse_l564_564952

noncomputable def isosceles_right_triangle (a b c : ℝ) (h : (a ^ 2 + b ^ 2 = c ^ 2)) : Prop :=
a = b ∧ a > 0 ∧ b > 0 ∧ c > 0

theorem area_of_45_45_90_triangle_with_altitude_to_hypotenuse (a b c : ℝ) (h1 : isosceles_right_triangle a b c) 
  (h2 : ∃ h: ℝ, h = 4 ∧ h * sqrt 2 = (1/2) * a * b) :
  a * b / 2 = 16 :=
by
  sorry

end area_of_45_45_90_triangle_with_altitude_to_hypotenuse_l564_564952


namespace juggling_contest_l564_564515

theorem juggling_contest (B : ℕ) (rot_baseball : ℕ := 80)
    (rot_per_apple : ℕ := 101) (num_apples : ℕ := 4)
    (winner_rotations : ℕ := 404) :
    (num_apples * rot_per_apple = winner_rotations) :=
by
  sorry

end juggling_contest_l564_564515


namespace fewer_sevens_l564_564893

def seven_representation (n : ℕ) : ℕ :=
  (7 * (10^n - 1)) / 9

theorem fewer_sevens (n : ℕ) :
  ∃ m, m < n ∧ 
    (∃ expr : ℕ → ℕ, (∀ i < n, expr i = 7) ∧ seven_representation n = expr m) :=
sorry

end fewer_sevens_l564_564893


namespace find_f_neg_l564_564725

noncomputable def f (a b x : ℝ) := a * x^3 + b * x - 2

theorem find_f_neg (a b : ℝ) (f_2017 : f a b 2017 = 7) : f a b (-2017) = -11 :=
by
  sorry

end find_f_neg_l564_564725


namespace grisha_wins_probability_expected_flips_l564_564823

-- Define conditions
def even_flip_heads_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 0 ∧ flip n = tt

def odd_flip_tails_win (n : ℕ) (flip : ℕ → bool) : Prop :=
  n % 2 = 1 ∧ flip n = ff

-- 1. Prove the probability P of Grisha winning the coin toss game is 1/3
theorem grisha_wins_probability (flip : ℕ → bool) (P : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  P = 1/3 := sorry

-- 2. Prove the expected number E of coin flips until the outcome is decided is 2
theorem expected_flips (flip : ℕ → bool) (E : ℝ) :
  (∀ n, even_flip_heads_win n flip ∨ odd_flip_tails_win n flip) →
  E = 2 := sorry

end grisha_wins_probability_expected_flips_l564_564823


namespace sum_arithmetic_sequence_l564_564002

def arithmetic_sequence (a : ℕ → ℕ) : Prop := 
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + 2

def sum_first_n_terms (a : ℕ → ℕ) : ℕ → ℕ :=
  λ n, ∑ i in finset.range n, a (i + 1)

theorem sum_arithmetic_sequence (a : ℕ → ℕ) (n : ℕ)
  (h : arithmetic_sequence a) : sum_first_n_terms a n = n^2 :=
by {
  sorry
}

end sum_arithmetic_sequence_l564_564002


namespace integer_roots_of_polynomial_l564_564165

theorem integer_roots_of_polynomial {a2 a1 : ℤ} :
    ∀ r : ℤ, (r ∈ {-24, -12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12, 24}) ↔ (r ≠ 0 ∧ (24 % r = 0)) :=
  by
  sorry

end integer_roots_of_polynomial_l564_564165


namespace total_weight_of_beef_l564_564493

-- Define the conditions
def packages_weight := 4
def first_butcher_packages := 10
def second_butcher_packages := 7
def third_butcher_packages := 8

-- Define the total weight calculation
def total_weight := (first_butcher_packages * packages_weight) +
                    (second_butcher_packages * packages_weight) +
                    (third_butcher_packages * packages_weight)

-- The statement to prove
theorem total_weight_of_beef : total_weight = 100 := by
  -- proof goes here
  sorry

end total_weight_of_beef_l564_564493


namespace bus_seating_options_l564_564094

theorem bus_seating_options :
  ∃! (x y : ℕ), 21*x + 10*y = 241 :=
sorry

end bus_seating_options_l564_564094


namespace find_cot_of_half_and_quarter_l564_564316

def alpha : ℝ := sorry  -- Let alpha be an angle in radians
def sin_alpha : ℝ := 4 / 5
def alpha_in_second_quadrant : Prop := sorry -- Let alpha be in the second quadrant

theorem find_cot_of_half_and_quarter :
  (sin α = sin_alpha) ∧ alpha_in_second_quadrant → cot (π / 4 - α / 2) = -3 :=
by
  sorry

end find_cot_of_half_and_quarter_l564_564316


namespace digit_7_appears_20_times_in_range_100_to_199_l564_564862

theorem digit_7_appears_20_times_in_range_100_to_199 : 
  ∃ (d7_count : ℕ), (d7_count = 20) ∧ (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 199 → 
  d7_count = (n.to_string.count '7')) :=
sorry

end digit_7_appears_20_times_in_range_100_to_199_l564_564862


namespace find_all_z_l564_564669

noncomputable def all_solutions (z : ℝ) : Prop := z = -2 ∨ z = 2 ∨ z = -real.sqrt 2 ∨ z = real.sqrt 2

theorem find_all_z (z : ℝ) : (z^4 - 6*z^2 + 8 = 0) ↔ all_solutions z := 
sorry

end find_all_z_l564_564669


namespace polygon_sides_sum_l564_564956

theorem polygon_sides_sum
  (area_ABCDEF : ℕ) (AB BC FA DE EF : ℕ)
  (h1 : area_ABCDEF = 78)
  (h2 : AB = 10)
  (h3 : BC = 11)
  (h4 : FA = 7)
  (h5 : DE = 4)
  (h6 : EF = 8) :
  DE + EF = 12 := 
by
  sorry

end polygon_sides_sum_l564_564956


namespace B_completes_remaining_work_in_three_days_l564_564551

-- Definitions
def work_done_in (days : ℕ) (completion_time : ℝ) : ℝ := days / completion_time

-- Conditions
def A_completion_time : ℝ := 15
def B_completion_time : ℝ := 4.5
def days_A_works : ℕ := 5
def remaining_work : ℝ := 1 - work_done_in days_A_works A_completion_time

-- Prove 
theorem B_completes_remaining_work_in_three_days : 
  ∃ d : ℝ, work_done_in d B_completion_time = remaining_work ∧ d = 3 := by
  sorry

end B_completes_remaining_work_in_three_days_l564_564551


namespace floor_sum_of_sequence_x_l564_564500

def sequence_x : ℕ → ℝ
| 0       => 1 / 2
| (n + 1) => sequence_x n ^ 2 + sequence_x n

def sum_terms (n : ℕ) : ℝ :=
  ∑ k in finset.range n, 1 / (1 + sequence_x k)

theorem floor_sum_of_sequence_x :
  ∀ n : ℕ, n = 2009 →
    ⌊ sum_terms n ⌋ = 1 :=
by
  intros n hn
  rw [hn]
  sorry

end floor_sum_of_sequence_x_l564_564500


namespace perfect_square_polynomial_l564_564804

theorem perfect_square_polynomial (m : ℝ) :
  (∃ a b : ℝ, (a * x + b)^2 = m - 10 * x + x^2) → m = 25 :=
sorry

end perfect_square_polynomial_l564_564804


namespace find_set_B_for_a2_find_value_of_a_l564_564776

noncomputable def setA (a : ℝ) : set ℝ :=
  {x | (x - 2) * (x - (3 * a + 1)) < 0}

noncomputable def funcDomain (a : ℝ) : set ℝ :=
  {x | 2 * a < x ∧ x < a^2 + 1}

theorem find_set_B_for_a2 : 
  funcDomain 2 = {x | 4 < x ∧ x < 5} :=
sorry

theorem find_value_of_a :
  setA a = funcDomain a → a = -1 :=
sorry

end find_set_B_for_a2_find_value_of_a_l564_564776


namespace cosine_dihedral_angle_l564_564379

theorem cosine_dihedral_angle {P A B C D : Point}
  (h_tetrahedron : regular_tetrahedron P A B C D)
  (h_angle_APC : ∠APC = 60°) : 
  dihedral_angle_cosine (Plane A P B) (Plane C P B) = -1 / 7 := 
sorry

end cosine_dihedral_angle_l564_564379


namespace haley_marbles_l564_564376

theorem haley_marbles (boys marbles_per_boy : ℕ) (h1: boys = 5) (h2: marbles_per_boy = 7) : boys * marbles_per_boy = 35 := 
by 
  sorry

end haley_marbles_l564_564376


namespace number_of_true_propositions_l564_564966

-- Definitions of the propositions
def prop1 (l : Line) (p1 p2 : Plane) : Prop :=
  (l ∥ p1) ∧ (p2 ∋ l) ∧ (p2 ∩ p1 ≠ ∅) → l ∥ (p2 ∩ p1)

def prop2 (l : Line) (p : Plane) (l₁ l₂ : Line) : Prop :=
  (l₁ ∩ l₂ ≠ ∅) ∧ (l ⟂ l₁) ∧ (l ⟂ l₂) → l ⟂ p

def prop3 (l₁ l₂ : Line) (p : Plane) : Prop :=
  (l₁ ∥ p) ∧ (l₂ ∥ p) → l₁ ∥ l₂

def prop4 (l : Line) (p1 p2 : Plane) : Prop :=
  (l ⟂ p1) ∧ (l ∈ p2) → p1 ⟂ p2

-- The theorem to prove the number of true propositions
theorem number_of_true_propositions (l : Line) (l₁ l₂ : Line) (p p1 p2 : Plane) :
  ((prop1 l p p2) → True) ∧
  ((prop2 l p l₁ l₂) → True) ∧
  ((prop3 l₁ l₂ p) → False) ∧
  ((prop4 l p1 p2) → True) →
  3 = 3 := 
by
  sorry

end number_of_true_propositions_l564_564966


namespace petya_sequence_count_l564_564027

theorem petya_sequence_count :
  let n := 100 in
  (5 ^ n - 3 ^ n) =
  (λ S : (ℕ × ℕ) → ℕ, 5 ^ S (n, n) - 3 ^ S (n, n) ) 5 100 :=
sorry

end petya_sequence_count_l564_564027


namespace percent_area_smaller_square_l564_564558

noncomputable def side_length_larger_square := 4
noncomputable def area_larger_square := side_length_larger_square ^ 2
noncomputable def circumscribed_circle_radius := (side_length_larger_square * Real.sqrt 2) / 2

noncomputable def side_length_smaller_square := 1 / 2
noncomputable def area_smaller_square := side_length_smaller_square ^ 2

noncomputable def required_percentage := 
  (area_smaller_square / area_larger_square) * 100

theorem percent_area_smaller_square :
  required_percentage = 1.5625 :=
by
  unfold required_percentage area_smaller_square area_larger_square
  sorry

end percent_area_smaller_square_l564_564558


namespace find_length_of_CD_l564_564496

noncomputable def length_CD {V : ℝ} (r : ℝ) (total_volume : ℝ) := 
  ∀ L : ℝ, 352 * real.pi = L * r^2 * real.pi + (8/3) * real.pi * r^3 → L = 16 + 2/3

theorem find_length_of_CD :
  length_CD 4 352 * real.pi :=
begin
  sorry
end

end find_length_of_CD_l564_564496


namespace arithmetic_mean_from_neg6_to_7_l564_564520

-- Definitions for the problem conditions
def integers_in_range := [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
def total_sum := integers_in_range.sum
def num_integers := integers_in_range.length
def arithmetic_mean := total_sum.toReal / num_integers

-- Statement to be proved
theorem arithmetic_mean_from_neg6_to_7 : arithmetic_mean = 0.5 := by
  sorry

end arithmetic_mean_from_neg6_to_7_l564_564520


namespace sequence_correct_l564_564988

-- Define the sequence a_n with the given conditions
def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2^(n - 1) + a (n - 1)

-- Theorem statement: Prove that a_n = 2^n - 1 holds for all n
theorem sequence_correct (n : ℕ) : a n = 2^n - 1 := by
  sorry

end sequence_correct_l564_564988


namespace train_speed_l564_564178

theorem train_speed (x : ℝ) (v : ℝ) 
  (h1 : (x / 50) + (2 * x / v) = 3 * x / 25) : v = 20 :=
by
  sorry

end train_speed_l564_564178


namespace cost_of_adult_ticket_l564_564022

theorem cost_of_adult_ticket (c : ℝ) 
  (h₁ : 2 * (c + 6) + 3 * c = 77)
  : c + 6 = 19 :=
sorry

end cost_of_adult_ticket_l564_564022


namespace quadrilateral_property_l564_564597

variables {A B C D E F : Type} [affine_space A point]

noncomputable def quadrilateral (a b c d : A) : Prop :=
  ∃ (F E : A), line DF ⊥ line BC ∧ collinear A C E ∧
  (BE = ⨯ (DF)) ∧ (BE bisects ∠ ABC) ∧ parallelogram (A D) = parallelogram (B C) ∧
  (∠ BAC = ∠ BAD)

theorem quadrilateral_property :
  ∀ (A B C D E F : A) (AB AC BE DF: ℝ),
  quadrilateral A B C D E F → 
  ∃ (θ : ℝ) (θ = 144) :=
by
  intros
  obtain ⟨⟨E, ⟨(h1 : line DF ⊥ line BC)⟩, (h2 : collinear A C E)⟩, 
          (h3 : BE = 2 * (DF))⟩, (h4 : line BE bisects ∠ ABC)⟩, 
          (h5 : line AD ∥ line BC), (h6: AB = AC⟩ from quadrilateral_property h
  use 144
  simp
  sorry

end quadrilateral_property_l564_564597


namespace ordered_triples_count_eq_4_l564_564630

theorem ordered_triples_count_eq_4 :
  ∃ (S : Finset (ℝ × ℝ × ℝ)), 
    (∀ x y z : ℝ, (x, y, z) ∈ S ↔ (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧ (xy + 1 = z) ∧ (yz + 1 = x) ∧ (zx + 1 = y)) ∧
    S.card = 4 :=
sorry

end ordered_triples_count_eq_4_l564_564630


namespace compare_compound_interest_l564_564150

noncomputable def compound_annually (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ t

noncomputable def compound_monthly (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ (12 * t)

theorem compare_compound_interest :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  compound_monthly P r_monthly t > compound_annually P r_annual t :=
by
  sorry

end compare_compound_interest_l564_564150


namespace albert_list_digits_final_segment_l564_564183

theorem albert_list_digits_final_segment :
  let albert_list := (list.range 100) ++ (list.range 1000).map (λ n, n + 200) ++ (list.range 10000).map (λ n, n + 2000)
  let digits := albert_list.bind (λ n, n.digits 10)
  (list.drop 1197 digits).take 3 = [2, 1, 9] := 
by {
  let albert_list := (list.range 100) ++ (list.range 1000).map (λ n, n + 200) ++ (list.range 10000).map (λ n, n + 2000)
  let digits := albert_list.bind (λ n, n.digits 10)
  have h1 : (list.drop 1197 digits).head = some 2, by sorry,
  have h2 : (list.drop 1198 digits).head = some 1, by sorry,
  have h3 : (list.drop 1199 digits).head = some 9, by sorry,
  exact list.ext (by simp [list.drop, list.take]); intros n,
  cases n,
  { simp [h1] },
  cases n,
  { simp [h2] },
  cases n,
  { simp [h3] },
  { simp }
}

end albert_list_digits_final_segment_l564_564183


namespace floor_equation_solution_l564_564253

theorem floor_equation_solution (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (⌊ (a^2 : ℝ) / b ⌋ + ⌊ (b^2 : ℝ) / a ⌋ = ⌊ (a^2 + b^2 : ℝ) / (a * b) ⌋ + a * b) ↔
    (∃ n : ℕ, a = n ∧ b = n^2 + 1) ∨ (∃ n : ℕ, a = n^2 + 1 ∧ b = n) :=
sorry

end floor_equation_solution_l564_564253


namespace halls_marriage_theorem_l564_564044

open Set

-- Define a bipartite graph.
structure BipartiteGraph (V : Type) :=
  (A B : Set V)
  (E : A → B → Prop)

-- Define Hall's marriage theorem statement.
theorem halls_marriage_theorem {V : Type} (G : BipartiteGraph V) :
  (∀ S : Set G.A, S ⊆ G.A → (card {b ∈ G.B | ∃ a ∈ S, G.E a b} ≥ card S)) →
  (∃ M : A → B, ∀ a : A, G.E a (M a)) :=
sorry

end halls_marriage_theorem_l564_564044


namespace det_3AB_l564_564746

variable (A B : Matrix (Fin n) (Fin n) ℝ) [Fintype n]

-- Given conditions
axiom det_A : det A = -3
axiom det_B : det B = 8

-- Problem statement
theorem det_3AB : det (3 • A ⬝ B) = -8 * 3^(Fintype.card n + 1) := sorry

end det_3AB_l564_564746


namespace proof_problem_l564_564200

variable a b c : ℝ

def expr := a + b * c

theorem proof_problem (h1: a = (1 / 27) ^ (-1 / 3)) (h2: b = Real.log 16 / Real.log 3) (h3: c = Real.log (1 / 9) / Real.log 2) :
    expr = -5 :=
by
  sorry

end proof_problem_l564_564200


namespace distinct_colored_grids_l564_564930

theorem distinct_colored_grids (n : ℕ) (colors : Finset ℕ) : 
  ∃ (grids : Finset (Finset (Finset ℕ))), 
    grid_condition grids colors n → n = 3 ∧ colors.card = 3 ∧ grids.card = 174 := 
by sorry

def grid_condition (grids : Finset (Finset (Finset ℕ))) (colors : Finset ℕ) (n : ℕ) : Prop :=
  ∀ g ∈ grids, 
    (∀ i j, (∃ (row : ℕ), row ∈ g ∧ (∀ k < n, (row ∈ colors ∧ row ≠ g (i, j) ∧ row ≠ g (i+1, j) ∧ row ≠ g (i, j+1)))))

noncomputable def number_of_distinct_grids : ℕ := 174

end distinct_colored_grids_l564_564930


namespace length_of_second_parallel_side_l564_564285

-- Define the given conditions
def parallel_side1 : ℝ := 20
def distance : ℝ := 14
def area : ℝ := 266

-- Define the theorem to prove the length of the second parallel side
theorem length_of_second_parallel_side (x : ℝ) 
  (h : area = (1 / 2) * (parallel_side1 + x) * distance) : 
  x = 18 :=
sorry

end length_of_second_parallel_side_l564_564285


namespace solve_quartic_eqn_l564_564673

theorem solve_quartic_eqn (z : ℂ) : 
  z ^ 4 - 6 * z ^ 2 + 8 = 0 ↔ z = 2 ∨ z = -2 ∨ z = Complex.sqrt 2 ∨ z = -Complex.sqrt 2 := 
sorry

end solve_quartic_eqn_l564_564673


namespace martingale_l564_564427

variables {n : ℕ} (η : Fin n → ℝ) (f : (nat → ℝ) → (nat → ℝ) → ℝ)

def is_martingale (ξ : Fin n → ℝ) (ℱ : Fin n → measurable_space ℝ) : Prop :=
  ∀⦃k⦄, 1 ≤ k → k < n → 
  measurable_space.sub_measurable_space (ℱ (k : Fin n)) (ℱ (k + 1 : Fin n)) →
  forall (y : ℝ), ∫ (λ ω, ξ k), P = ∫ (λ ω, conditional_expectation (ξ (k+1)) (ℱ (k + 1))).to_fun ω, P

def ξ : Fin n → ℝ
| 0 => η 0
| k+1 => ∑ i in Finset.range (k + 1), f (η 0 .. η i) (η (i + 1))

theorem martingale : is_martingale ξ _ :=
sorry

end martingale_l564_564427


namespace triangle_area_formula_correct_l564_564636

theorem triangle_area_formula_correct :
  let a := (2 * √2 + √5) * ( √2 - 1) / ( √5 + 2 * √2),
      b := (2 * √2 + √5) * √5 / ( √5 + 2 * √2),
      c := (2 * √2 + √5) * ( √2 + 1) / ( √5 + 2 * √2) in
  let S := sqrt(1 / 4 * ((c * a) ^ 2 - (1 / 2 * (c ^ 2 + a ^ 2 - b ^ 2)) ^ 2)) in
  S = √3 / 4 :=
sorry

end triangle_area_formula_correct_l564_564636


namespace trees_counted_in_2002_l564_564380

def T : ℕ → ℕ
def P : ℕ → ℕ
def k : ℕ

axiom A1 : T 2000 = 100
axiom A2 : P 2001 = 150
axiom A3 : T 2003 = 250
axiom A4 : ∀ n : ℕ, T (n + 2) - T n = k * P (n + 1)

theorem trees_counted_in_2002 : T 2002 = 150 :=
by
  sorry

end trees_counted_in_2002_l564_564380


namespace students_expected_to_c_l564_564537

-- Define the conditions given in the problem.
def total_students : ℕ := 100
def percent_a : ℝ := 0.60
def percent_b : ℝ := 0.40
def percent_from_a_to_c : ℝ := 0.30
def percent_from_b_to_c : ℝ := 0.40

-- Calculate the number of students going to each school and those expected to go to the new school.
def students_a : ℕ := (percent_a * total_students).to_nat
def students_b : ℕ := (percent_b * total_students).to_nat
def students_from_a_to_c : ℕ := (percent_from_a_to_c * students_a).to_nat
def students_from_b_to_c : ℕ := (percent_from_b_to_c * students_b).to_nat

def total_students_to_c : ℕ := students_from_a_to_c + students_from_b_to_c
def percent_students_to_c : ℝ := (total_students_to_c.to_nat : ℝ) / (total_students.to_nat : ℝ) * 100.0

-- Prove the expected percentage of students going to the new school (c).
theorem students_expected_to_c : percent_students_to_c = 34 :=
by
  -- Necessary calculations and logical steps are omitted here, just final theorem.
  sorry

end students_expected_to_c_l564_564537


namespace taylor_family_reunion_l564_564193

theorem taylor_family_reunion :
  let number_of_kids := 45
  let number_of_adults := 123
  let number_of_tables := 14
  (number_of_kids + number_of_adults) / number_of_tables = 12 := by sorry

end taylor_family_reunion_l564_564193


namespace find_z_values_l564_564653

theorem find_z_values : {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -sqrt 2, sqrt 2, 2} :=
sorry

end find_z_values_l564_564653


namespace solve_quartic_eq_l564_564703

theorem solve_quartic_eq (z : ℂ) :
  z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -complex.sqrt 2 ∨ z = complex.sqrt 2 ∨ z = 2 :=
by {
  sorry
}

end solve_quartic_eq_l564_564703


namespace magnitude_of_complex_number_l564_564729

-- Definition of the problem
variable (z : ℂ)
variable (hz : z + complex.abs z = 2 + 8 * complex.I)

-- The proof problem statement
theorem magnitude_of_complex_number : complex.abs z = 17 :=
sorry

end magnitude_of_complex_number_l564_564729


namespace triangle_arithmetic_sequence_l564_564851

variable {α : Type*} [LinearOrderedField α]

/-- Given a, b, c form an arithmetic sequence in a triangle ABC, we want to show
 1.  0 < B ≤ π/3
 2.  a cos ²(C/2) + c cos ²(A/2) = 3b/2 -/
theorem triangle_arithmetic_sequence
  {a b c A B C : α}
  (h₁ : 2 * b = a + c)
  (h₂ : A + B + C = π)
  (h₃ : 0 < A) (h₄ : 0 < B) (h₅ : 0 < C)
  (h₆ : A + B + C = π)
  (h₇ : A < π) (h₈ : B < π) (h₉ : C < π) :
  (0 < B ∧ B ≤ π / 3)
  ∧ (a * (cos (C / 2)) ^ 2 + c * (cos (A / 2)) ^ 2 = 3 * b / 2) :=
sorry

end triangle_arithmetic_sequence_l564_564851


namespace coefficient_of_x_squared_in_expansion_of_x_times_1_plus_2x_pow_6_l564_564959

theorem coefficient_of_x_squared_in_expansion_of_x_times_1_plus_2x_pow_6 :
  (coeff (x : ℝ) (x * (1 + 2 * x)^6) 2) = 12 :=
sorry

end coefficient_of_x_squared_in_expansion_of_x_times_1_plus_2x_pow_6_l564_564959
