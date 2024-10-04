import Mathlib
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Circle
import Mathlib.GroupTheory.Perm.Cycle
import Mathlib.Logic.Basic
import Mathlib.Probability
import Mathlib.Probability.Conditional
import Mathlib.Probability.Distribution
import Mathlib.Probability.Independence
import Mathlib.Probability.Notation
import Mathlib.Tactic
import Mathlib.Topology.TrigonometricFunctions
import ProbabilityTheory
import Real

namespace polygon_area_l590_590039

-- Given
def side_perpendicular_adjacents (polygon : Type) : Prop := sorry -- Placeholder definition

structure Polygon :=
  (n_sides : ℕ)
  (perimeter : ℕ)
  (side_length : ℕ)
  (sides_congruent : Prop)
  (sides_perpendicular: side_perpendicular_adjacents Polygon)

noncomputable def example_polygon : Polygon :=
{ n_sides := 20,
  perimeter := 60,
  side_length := 3,
  sides_congruent := true,
  sides_perpendicular := sorry  -- Perpendicular condition placeholder
}

-- Prove area calculation based on given conditions
theorem polygon_area (P : Polygon) 
  (h_sides : P.sides_congruent)
  (h_perpendicular : side_perpendicular_adjacents P)
  (h_sides_20 : P.n_sides = 20)
  (h_perimeter : P.perimeter = 60)
  (h_side_length : P.side_length = P.perimeter / P.n_sides) :
  P.n_sides * P.side_length * P.side_length / 4 = 144 :=
sorry

end polygon_area_l590_590039


namespace sum_faces_of_pentahedron_l590_590982

def pentahedron := {f : ℕ // f = 5}

theorem sum_faces_of_pentahedron (p : pentahedron) : p.val = 5 := 
by
  sorry

end sum_faces_of_pentahedron_l590_590982


namespace find_y_l590_590883

theorem find_y (a b y : ℝ) (ha : 0 < a) (hb : 0 < b) (hy : 0 < y) :
  let s := (a^2)^(2 * b)
  in s = (a^b) * (y^b) → y = a^3 :=
by 
  intro h
  sorry

end find_y_l590_590883


namespace distributor_income_proof_l590_590018

noncomputable def income_2017 (a k x : ℝ) : ℝ :=
  (a + k / (x - 7)) * (x - 5)

theorem distributor_income_proof (a : ℝ) (x : ℝ) (h_range : 10 ≤ x ∧ x ≤ 14) (h_k : k = 3 * a):
  income_2017 a (3 * a) x = 12 * a ↔ x = 13 := by
  sorry

end distributor_income_proof_l590_590018


namespace sequence_formula_l590_590362

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n - 2) :
  ∀ n : ℕ, a n = 3^(n - 1) + 1 :=
by sorry

end sequence_formula_l590_590362


namespace parities_D_2021_2022_2023_l590_590275

def sequence (n : ℕ) : ℕ
| 0 => 0
| 1 => 0
| 2 => 1
| (n + 3) => sequence (n + 2) + sequence n

theorem parities_D_2021_2022_2023 :
    (sequence 2021 % 2 = 0) ∧ (sequence 2022 % 2 = 1) ∧ (sequence 2023 % 2 = 0) :=
by
    sorry

end parities_D_2021_2022_2023_l590_590275


namespace volume_of_pyramid_l590_590914

/-- Definitions for the conditions -/
def is_regular_hexagon (ABCDEF : Type) : Prop :=
  -- Define here the necessary conditions for ABCDEF to be a regular hexagon
  sorry

def is_right_pyramid (Q : Type) (ABCDEF : Type) : Prop :=
  -- Define here the necessary conditions for the pyramid to be right pyramid with base ABCDEF
  sorry

def is_equilateral_triangle (QAD : Type) (side_length : ℝ) : Prop :=
  -- Define here the properties of an equilateral triangle with given side length
  sorry

/-- The main theorem we need to prove -/
theorem volume_of_pyramid (ABCDEF QAD : Type)
  (H1 : is_regular_hexagon ABCDEF)
  (H2 : is_right_pyramid Q ABCDEF)
  (H3 : is_equilateral_triangle QAD 10) :
  volume_pyramid Q ABCDEF = 187.5 :=
sorry

/-- Placeholder for the actual volume calculation, to be used in theorem above -/
noncomputable def volume_pyramid (Q : Type) (ABCDEF : Type) : ℝ :=
  -- Define the volume calculation formula for the pyramid here
  sorry

end volume_of_pyramid_l590_590914


namespace length_of_AB_l590_590506

theorem length_of_AB 
  (P Q A B : ℝ)
  (h_P_on_AB : P > 0 ∧ P < B)
  (h_Q_on_AB : Q > P ∧ Q < B)
  (h_ratio_P : P = 3 / 7 * B)
  (h_ratio_Q : Q = 4 / 9 * B)
  (h_PQ : Q - P = 3) 
: B = 189 := 
sorry

end length_of_AB_l590_590506


namespace contact_lenses_sales_l590_590990

theorem contact_lenses_sales :
  ∃ H S : ℕ,
    let price_soft := 150,
        price_hard := 85 in
    S = H + 5 ∧
    (price_soft * S + price_hard * H = 1455) ∧
    (H + S = 11) :=
by
  sorry

end contact_lenses_sales_l590_590990


namespace probability_sin_cos_in_intervals_l590_590038

open Real Set

theorem probability_sin_cos_in_intervals :
  let s := set.Icc (-π / 6) (π / 2)
  measure_theory.measure_space.measure (λ x : ℝ, sin x + cos x ∈ Icc 1 (sqrt 2)) s
  / measure_theory.measure_space.measure s = 3 / 4 :=
sorry

end probability_sin_cos_in_intervals_l590_590038


namespace house_ordering_count_l590_590974

variable (positions : Fin 5 → String)
variable (green red blue orange yellow : String)

def house_ordering_valid (positions : Fin 5 → String) : Prop :=
  ∃ (g r b o : Fin 5), 
    positions g = green ∧ positions r = red ∧ positions b = blue ∧ positions o = orange ∧
    (g < r) ∧ (b < o) ∧ (g < b) ∧ (∀ (i : Fin 4), positions i ≠ positions (i + 1))

theorem house_ordering_count 
  (positions : Fin 5 → String)
  (green red blue orange yellow : String)
  (h : house_ordering_valid positions) :
  { orderings : (Fin 5 → String) // house_ordering_valid orderings }.card = 5 :=
sorry

end house_ordering_count_l590_590974


namespace solve_abs_inequality_l590_590174

theorem solve_abs_inequality (x : ℝ) :
  (|x-2| ≥ |x|) → x ≤ 1 :=
by
  sorry

end solve_abs_inequality_l590_590174


namespace count_convergent_integers_l590_590462

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1
  else if n % 3 = 0 then n / 3
  else n / 2

def converges_to_one (g : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate g k n) = 1

def count_convergents (f : ℕ → bool) (s : set ℕ) : ℕ :=
  (s.filter f).card

theorem count_convergent_integers :
  count_convergents (converges_to_one g) (set.Icc 1 100) = 6 :=
sorry

end count_convergent_integers_l590_590462


namespace smaller_square_side_length_l590_590920

theorem smaller_square_side_length :
  ∃ (BG : ℝ), ((Square ABCD has sides of length 2) ∧
                (Point E is on BC and BE = 1) ∧
                (Point F is on CD and DF = 1) ∧
                (BG is perpendicular to BE and G is on AE)) →
                BG = 2 * (√3) / 3 :=
begin
  sorry
end

end smaller_square_side_length_l590_590920


namespace probability_unique_tens_digits_l590_590546

theorem probability_unique_tens_digits :
  let num_ways := 10^6 in
  let total_combinations := Nat.choose 70 6 in
  (num_ways : ℚ) / total_combinations = 625 / 74440775 :=
by 
  sorry

end probability_unique_tens_digits_l590_590546


namespace purely_imaginary_m_no_m_in_fourth_quadrant_l590_590593

def z (m : ℝ) : ℂ := ⟨m^2 - 8 * m + 15, m^2 - 5 * m⟩

theorem purely_imaginary_m :
  (∀ m : ℝ, z m = ⟨0, m^2 - 5 * m⟩ ↔ m = 3) :=
by
  sorry

theorem no_m_in_fourth_quadrant :
  ¬ ∃ m : ℝ, (m^2 - 8 * m + 15 > 0) ∧ (m^2 - 5 * m < 0) :=
by
  sorry

end purely_imaginary_m_no_m_in_fourth_quadrant_l590_590593


namespace aaron_beth_sheep_ratio_l590_590280

theorem aaron_beth_sheep_ratio (A B : ℕ) (hB : B = 76) (hSum : A + B = 608) :
  A / Int.gcd A B = 133 ∧ B / Int.gcd A B = 19 :=
by
  have hA : A = 532 := by
    calc
      A = 608 - B : by linarith
      _ = 532     : by rw [hB]
  have gcdAB : Int.gcd 532 76 = 4 := by norm_num
  simp [hA, Int.gcd, gcdAB]
  split
  · norm_num
  · norm_num
  sorry

end aaron_beth_sheep_ratio_l590_590280


namespace compute_a1d1_a2d2_a3d3_eq_1_l590_590876

theorem compute_a1d1_a2d2_a3d3_eq_1 {a1 a2 a3 d1 d2 d3 : ℝ}
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 1 := by
  sorry

end compute_a1d1_a2d2_a3d3_eq_1_l590_590876


namespace infinite_rational_distance_points_l590_590295

-- Define the conditions
def theta : ℝ := Real.arccos (3 / 5)

def circle (O : Point) (r : ℝ) : Set Point := 
{ P : Point | dist O P = r }

-- Define the point sequence on the circle
noncomputable def P (n : ℕ) : Point :=
let theta := Real.arccos (3 / 5) in
let angle := 2 * n * theta in
-- Use a parametric representation of points on the unit circle centered at (0,0)
let P := (cos angle, sin angle) in
P

-- Problem statement
theorem infinite_rational_distance_points : 
  ∃ (P : ℕ → Point), (∀ m n : ℕ, m ≠ n → dist (P m) (P n) ∈ ℚ) ∧ (∀ i j k : ℕ, ¬ collinear (P i) (P j) (P k)) :=
  sorry

end infinite_rational_distance_points_l590_590295


namespace ratios_bounds_l590_590499

variable {A B C X Y Z O : Type*}
variable [HasRatios A B C X Y Z O]
variable [IntersectsAt AX BY CZ O]

theorem ratios_bounds 
  (hAX : IntersectsAt AX O)
  (hBY : IntersectsAt BY O)
  (hCZ : IntersectsAt CZ O) :
  ∃ r, (r = OA / OX ∧ r ≤ 2) ∨ (r = OB / OY ∧ r ≥ 2) ∨ (r = OC / OZ ∧ r ≤ 2) ∨ (r = OC / OZ ∧ r ≥ 2) :=
sorry

end ratios_bounds_l590_590499


namespace interest_calculation_is_compound_l590_590656

-- Defining the conditions
def final_amount : ℝ := 8820
def principal : ℝ := 8000
def interest_rate : ℝ := 5 / 100
def time_years : ℕ := 2

-- The main theorem stating the interest calculation used
theorem interest_calculation_is_compound :
  (principal * (1 + interest_rate)^time_years = final_amount) :=
by
  sorry

end interest_calculation_is_compound_l590_590656


namespace runner_speed_comparison_l590_590967

theorem runner_speed_comparison
  (t1 t2 : ℕ → ℝ) -- function to map lap-time.
  (s v1 v2 : ℝ)  -- speed of runners v1 and v2 respectively, and the street distance s.
  (h1 : t1 1 < t2 1) -- first runner overtakes the second runner twice implying their lap-time comparison.
  (h2 : ∀ n, t1 (n + 1) = t1 n + t1 1) -- lap time consistency for runner 1
  (h3 : ∀ n, t2 (n + 1) = t2 n + t2 1) -- lap time consistency for runner 2
  (h4 : t1 3 < t2 2) -- first runner completes 3 laps faster than second runner completes 2 laps
   : 2 * v2 ≤ v1 := sorry

end runner_speed_comparison_l590_590967


namespace pure_imaginary_solution_l590_590007

theorem pure_imaginary_solution (a : ℝ) (i : ℂ) (h : i*i = -1) : (∀ z : ℂ, z = 1 + a * i → (z ^ 2).re = 0) → (a = 1 ∨ a = -1) := by
  sorry

end pure_imaginary_solution_l590_590007


namespace least_n_distance_ge_150_l590_590076

open Real

-- Define the conditions step by step
def A0 := (0 : ℝ, 0 : ℝ)

def onXaxis (A : ℕ → ℝ × ℝ) : Prop :=
  ∀ n : ℕ, (A n).snd = 0

def onCubeRootGraph (B : ℕ → ℝ × ℝ) : Prop :=
  ∀ n : ℕ, (B n).snd = (B n).fst ^ (1/3 : ℝ)

def isEquilateral (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) : Prop :=
  ∀ n : ℕ, ∃ t : ℝ,
    sqrt (((B n).fst - (A (n-1)).fst) ^ 2 + ((B n).snd - (A (n-1)).snd) ^ 2) = t ∧
    sqrt (((B n).fst - (A n).fst) ^ 2 + ((B n).snd - (A n).snd) ^ 2) = t ∧
    sqrt (((A (n-1)).fst - (A n).fst) ^ 2 + ((A (n-1)).snd - (A n).snd) ^ 2) = t

def distance (A0 An : ℝ × ℝ) : ℝ :=
  sqrt ((An.fst - A0.fst) ^ 2 + (An.snd - A0.snd) ^ 2)

-- Lean statement to prove the required property
theorem least_n_distance_ge_150 {A B : ℕ → ℝ × ℝ}
  (hx : onXaxis A) (hB : onCubeRootGraph B) (hE : isEquilateral A B) :
  ∃ n : ℕ, n = 12 ∧ distance A0 (A n) ≥ 150 :=
sorry

end least_n_distance_ge_150_l590_590076


namespace historical_figure_birth_day_l590_590153

theorem historical_figure_birth_day :
  (anniversary_date : Nat -> Nat -> Nat -> Prop) -> -- anniversary_date y m d means (y, m, d) corresponds to the anniversary date.
  (is_monday : Nat -> Nat -> Nat -> Prop) -> -- is_monday y m d means (y, m, d) is a Monday.
  (birth_year : Nat) -> -- birth year of the historical figure.
  (anniversary_year : Nat) -> -- anniversary year of the birth.
  (total_years : Nat) -> -- total number of years between birth and anniversary.
  anniversary_date 2022 2 7 -> -- Given the anniversary date is February 7, 2022.
  is_monday 2022 2 7 -> -- Given February 7, 2022 is a Monday.
  anniversary_year - birth_year = 250 -> -- The time span between birth_year and anniversary_year is 250 years.
  birth_day_of_week : String := -- The correct day of the week when the birth occurred.
  "Saturday" := -- Given birth_day_of_week is "Saturday".
  sorry -- Proof steps are omitted.

end historical_figure_birth_day_l590_590153


namespace perimeter_quadrilateral_l590_590849

theorem perimeter_quadrilateral {P Q R S : ℕ × ℕ} (horiz_dist vert_dist : ℕ) 
    (horiz_dist_eq : horiz_dist = 1) (vert_dist_eq : vert_dist = 1)
    (PQ : P.fst = Q.fst ∧ (Q.snd - P.snd).abs = 4 * horiz_dist)
    (QR : Q.fst = R.fst ∧ (R.snd - Q.snd).abs = 1 * vert_dist)
    (SP : S.fst = P.fst ∧ (S.snd - P.snd).abs = 1 * vert_dist)
    (RST : (R.fst - S.fst).abs = 4 * horiz_dist ∧ (R.snd - S.snd).abs = 3 * vert_dist) 
    : 4 * horiz_dist + 4 * horiz_dist + 5 * vert_dist + 1 * horiz_dist = 14 := 
by 
    sorry

end perimeter_quadrilateral_l590_590849


namespace max_n_for_distinct_squares_l590_590976

theorem max_n_for_distinct_squares (k : ℕ → ℕ) :
  (∀ i j, i ≠ j → k i ≠ k j) ∧
  (∀ i, k i ≠ 10) ∧
  (∑ i in range 17, (k i)^2 = 2500) ∧
  (∀ m, m ≥ 17 → (∑ i in range m, (k i)^2 ≤ 2500)) := sorry

end max_n_for_distinct_squares_l590_590976


namespace probability_sum_sixteen_l590_590318

-- Define the probabilities involved
def probability_of_coin_fifteen := 1 / 2
def probability_of_die_one := 1 / 6

-- Define the combined probability
def combined_probability : ℚ := probability_of_coin_fifteen * probability_of_die_one

theorem probability_sum_sixteen : combined_probability = 1 / 12 := by
  sorry

end probability_sum_sixteen_l590_590318


namespace isosceles_triangles_with_105_deg_angle_are_similar_l590_590985

/-- Two isosceles triangles with a vertex angle of 105 degrees are similar -/
theorem isosceles_triangles_with_105_deg_angle_are_similar (α β : ℝ) (hα : α = 105) (hβ : β = 105) :
  ∃ (T1 T2 : Triangle), isosceles T1 ∧ isosceles T2 ∧ T1.angle1 = α ∧ T2.angle1 = β → 
  similar_triangles T1 T2 :=
by
  sorry

end isosceles_triangles_with_105_deg_angle_are_similar_l590_590985


namespace negation_of_all_citizens_bad_l590_590715

-- Define the propositions
def all_citizens_good : Prop := ∀ x : Citizen, good_driver x
def some_citizens_good : Prop := ∃ x : Citizen, good_driver x
def no_women_good : Prop := ∀ x : Citizen, is_woman x → ¬good_driver x
def all_women_bad : Prop := ∀ x : Citizen, is_woman x → bad_driver x
def at_least_one_woman_bad : Prop := ∃ x : Citizen, is_woman x ∧ bad_driver x
def all_citizens_bad : Prop := ∀ x : Citizen, bad_driver x

-- Theorem statement
theorem negation_of_all_citizens_bad : ¬all_citizens_bad ↔ some_citizens_good := 
sorry

end negation_of_all_citizens_bad_l590_590715


namespace number_of_friends_l590_590459

theorem number_of_friends (total_skittles : ℕ) (skittles_per_friend : ℕ) (h1 : total_skittles = 40) (h2 : skittles_per_friend = 8) : total_skittles / skittles_per_friend = 5 :=
by
  rw [h1, h2]
  norm_num

end number_of_friends_l590_590459


namespace find_a_l590_590398

theorem find_a (a : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a^x + real.log x / real.log a)
  (h2 : 0 < a) (h3 : a ≠ 1)
  (h4 : f 1 + f 2 = (real.log 2 / real.log a) + 6) :
  a = 2 :=
sorry

end find_a_l590_590398


namespace solve_m_l590_590404

variable {R : Type} [LinearOrderedField R]

def vector_parallel (u v : R × R) : Prop :=
  u.1 * v.2 = u.2 * v.1

theorem solve_m (m : R) :
  let a := (1, m)
      b := (2, 5)
      c := (m, 3)
      ac := (a.1 + c.1, a.2 + c.2)
      ab := (a.1 - b.1, a.2 - b.2)
  (vector_parallel ac ab) →
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 :=
by
  sorry

end solve_m_l590_590404


namespace actual_time_greater_than_planned_time_l590_590669

def planned_time (a V : ℝ) : ℝ := a / V

def actual_time (a V : ℝ) : ℝ := (a / (2.5 * V)) + (a / (1.6 * V))

theorem actual_time_greater_than_planned_time (a V : ℝ) (hV : V > 0) : 
  actual_time a V > planned_time a V :=
by 
  sorry

end actual_time_greater_than_planned_time_l590_590669


namespace equilateral_sum_of_sides_l590_590867

theorem equilateral_sum_of_sides {s : ℕ} (h₀ : s > 30) (h₁ : s ≤ 34) :
  ∑ i in {31, 32, 33, 34}.to_finset, i = 130 :=
by sorry

end equilateral_sum_of_sides_l590_590867


namespace probability_sum_16_l590_590320

open ProbabilityTheory

noncomputable def coin_flip_probs : Finset ℚ := {5 , 15}
noncomputable def die_probs : Finset ℚ := {1, 2, 3, 4, 5, 6}

def fair_coin (x : ℚ) : ℚ := if x = 5 ∨ x = 15 then (1 : ℚ) / 2 else 0
def fair_die (x : ℚ) : ℚ := if x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 then (1 : ℚ) / 6 else 0

theorem probability_sum_16 : ∑ x in coin_flip_probs, ∑ y in die_probs, (if x + y = 16 then fair_coin x * fair_die y else 0) = 1 / 12 := 
    sorry

end probability_sum_16_l590_590320


namespace numberOfFourDigitNumbers_l590_590412

-- A function to count unique four-digit numbers from the digits 2, 0, 2, 1
def fourDigitNumbersCount : ℕ :=
  have h_unique_digits : list ℕ := [2, 0, 2, 1]

  -- Conditions
  def isValidFourDigit (n : ℕ) : Prop :=
    1000 ≤ n ∧ n < 10000 ∧
    n.digits 10 = h_unique_digits ∧
    (n.digits 10).head ≠ 0

  -- Target number from permutations satisfying the above
  number_of_valid_rearrangements

-- Statement to show the count equals 6
theorem numberOfFourDigitNumbers : fourDigitNumbersCount = 6 := by
  sorry

end numberOfFourDigitNumbers_l590_590412


namespace regular_polygon_sides_l590_590746

theorem regular_polygon_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle i = 160)) : n = 18 :=
by
  -- Proof goes here
  sorry

end regular_polygon_sides_l590_590746


namespace factorization_problem_l590_590286

theorem factorization_problem :
  (∃ (h : D), 
    (¬ ∃ (a b : ℝ) (x y : ℝ), a * (x - y) = a * x - a * y) ∧
    (¬ ∃ (x : ℝ), x^2 - 2 * x + 3 = x * (x - 2) + 3) ∧
    (¬ ∃ (x : ℝ), (x - 1) * (x + 4) = x^2 + 3 * x - 4) ∧
    (∃ (x : ℝ), x^3 - 2 * x^2 + x = x * (x - 1)^2)) :=
  sorry

end factorization_problem_l590_590286


namespace arrangement_of_students_l590_590244

theorem arrangement_of_students:
  let students : Fin 6 := {0, 1, 2, 3, 4, 5}
  let communities : Fin 3 := {A, B, C}
  let possible_arrangements := 
    {arrangement : students → communities // 
      (∀ s, arrangement s ∈ communities) ∧
      ((arrangement 0 = A ∧ arrangement 1 ≠ C ∧ arrangement 2 ≠ C) ∧ 
       ∃ (s1 s2 : Fin 6),
         (s1 ≠ s2 ∧ arrangement s1 = arrangement s2 ∧ arrangement s1 = A) ∧
         ∃ (s3 s4 : Fin 6),
           (s3 ≠ s4 ∧ arrangement s3 = arrangement s4 ∧ arrangement s3 = B) ∧
         ∃ (s5 s6 : Fin 6),
           (s5 ≠ s6 ∧ arrangement s5 = arrangement s6 ∧ arrangement s5 = C)) ∧
      (∀ c ∈ communities, ∃! (x y : {v : students // arrangement v = c}), x ≠ y)
    }
  in  ∥ possible_arrangements ∥ = 9 := sorry

end arrangement_of_students_l590_590244


namespace categorize_numbers_l590_590758

def is_positive_integer (n : Real) : Prop := n > 0 ∧ n.floor = n
def is_negative_fraction (n : Real) : Prop := n < 0 ∧ ∃ a b : Int, b ≠ 0 ∧ n = (a : Real) / (b : Real)
def is_irrational (n : Real) : Prop := ¬ ∃ a b : Int, b ≠ 0 ∧ n = (a : Real) / (b : Real)

def given_numbers : List Real := [0, Real.sqrt 5, -3.5, Real.pi, -12 / 7, Real.cbrt (-27), Real.abs (-8), 1.202002 ...]

theorem categorize_numbers :
  (∃ n ∈ given_numbers, is_positive_integer n ∧ n = Real.abs (-8))
  ∧ (∃ n ∈ given_numbers, is_negative_fraction n ∧ (n = -3.5 ∨ n = -12 / 7))
  ∧ (∃ n ∈ given_numbers, is_irrational n ∧ (n = Real.sqrt 5 ∨ n = Real.pi ∨ n = 1.202002 ...)) := 
by
    sorry

end categorize_numbers_l590_590758


namespace closest_integer_to_cube_root_of_150_l590_590610

theorem closest_integer_to_cube_root_of_150 : ∃ (n : ℤ), abs ((n: ℝ)^3 - 150) ≤ abs (((n + 1 : ℤ) : ℝ)^3 - 150) ∧
  abs ((n: ℝ)^3 - 150) ≤ abs (((n - 1 : ℤ) : ℝ)^3 - 150) ∧ n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590610


namespace power_sum_is_integer_l590_590886

variable {x : ℝ}

theorem power_sum_is_integer (hx : x ≠ 0) (h : x + 1/x ∈ ℤ) :
  ∀ n : ℕ, x^n + 1/x^n ∈ ℤ :=
by
  sorry

end power_sum_is_integer_l590_590886


namespace height_difference_l590_590070

variable {J L R : ℕ}

theorem height_difference
  (h1 : J = L + 15)
  (h2 : J = 152)
  (h3 : L + R = 295) :
  R - J = 6 :=
sorry

end height_difference_l590_590070


namespace max_ab_value_l590_590401

theorem max_ab_value (a b : ℝ) (h : 2 ^ a * 2 ^ b = 2) (ha : a > 0) (hb : b > 0) : ab <= 1/4 :=
by
  sorry

end max_ab_value_l590_590401


namespace expansion_terms_count_l590_590850

theorem expansion_terms_count : 
  let S := { n : ℕ × ℕ × ℕ | n.1 + n.2.1 + n.2.2 = 8 }
  in fintype.card S = nat.choose 10 2 := 
by sorry

end expansion_terms_count_l590_590850


namespace ratio_equality_l590_590885

-- Definitions for the geometric objects involved
variables (A B C D P Q R S T U : Type) [Incircle ABCD] 

-- Conditions of the problem
def conditions (A B C D P Q R S T U : Point) : Prop :=
  circle_in A B C D ∧
  parallel D P B C ∧ intersect_line AC D P = P ∧
  intersect_line AB D P = Q ∧ 
  reintersect_circle R D P = R ∧
  parallel D S AB ∧ intersect_line AC D S = S ∧
  intersect_line BC D S = T ∧ 
  reintersect_circle U D S = U

-- The theorem statement
theorem ratio_equality 
  (A B C D P Q R S T U : Point)
  (h : conditions A B C D P Q R S T U) :
  (ST / TU) = (QR / PQ) :=
sorry

end ratio_equality_l590_590885


namespace calculate_triangle_ABC_area_l590_590433

def point := (ℝ × ℝ)

def triangle (A B C : point) : Type :=
{ base : ℝ // base = dist A B ∧ height = (C.2 - 0) }

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

theorem calculate_triangle_ABC_area :
  let A := (0, 0)
  let B := (3, 0)
  let C := (0, 4)
  area_of_triangle A B C = 6 :=
by
  -- here we would provide the proof steps
  sorry

end calculate_triangle_ABC_area_l590_590433


namespace length_of_AB_l590_590505

theorem length_of_AB (A B C D E F G : Point)
  (midpoint_AC : midpoint C A B)
  (midpoint_AD : midpoint D A C)
  (midpoint_AE : midpoint E A D)
  (midpoint_AF : midpoint F A E)
  (midpoint_AG : midpoint G A F)
  (length_AG : AG = 3) : 
  AB = 96 :=
sorry

end length_of_AB_l590_590505


namespace total_mass_grain_l590_590169

-- Given: the mass of the grain is 0.5 tons, and this constitutes 0.2 of the total mass
theorem total_mass_grain (m : ℝ) (h : 0.2 * m = 0.5) : m = 2.5 :=
by {
    -- Proof steps would go here
    sorry
}

end total_mass_grain_l590_590169


namespace minimum_ping_pong_balls_l590_590631

theorem minimum_ping_pong_balls :
  ∃ (nums : List ℕ), nums.length = 10 ∧ 
  (∀ n ∈ nums, 11 ≤ n ∧ n ≠ 13 ∧ ¬(∃ k, k * 5 = n)) ∧ 
  nums.nodup ∧ nums.sum = 173 := sorry

end minimum_ping_pong_balls_l590_590631


namespace find_quartic_poly_l590_590328

noncomputable def quartic_poly : Polynomial ℚ := 
  Polynomial.of_coeffs [1, -10, 17, 18, -12]

def isMonic (p : Polynomial ℚ) : Prop := p.leadingCoeff = 1

def hasRationalCoeffs (p : Polynomial ℚ) : Prop := 
  ∀ i, is_rat (coeff p i)

def isRoot (p : Polynomial ℚ) (x : ℚ) : Prop := 
  p.eval x = 0

theorem find_quartic_poly :
  ∃ (p : Polynomial ℚ), isMonic p ∧ hasRationalCoeffs p ∧ isRoot p (3 + √5) ∧ isRoot p (2 - √7) ∧ p = quartic_poly :=
  sorry

end find_quartic_poly_l590_590328


namespace exists_parallelogram_C1D1_C2D2_l590_590633

open Function

-- Definitions corresponding to the conditions in the problem
variables {ω : Circle} {A B : Point}
hypothesis (A_interior : ω.contains_interior A)
hypothesis (B_on_circle : ω.contains B)

-- Statement of the theorem
theorem exists_parallelogram_C1D1_C2D2 :
  ∃ (C1 D1 C2 D2 : Point),
  ω.contains C1 ∧ ω.contains D1 ∧ ω.contains C2 ∧ ω.contains D2 ∧
  is_parallelogram A B C1 D1 ∧ is_parallelogram A B C2 D2 :=
sorry

end exists_parallelogram_C1D1_C2D2_l590_590633


namespace collinear_points_average_x_coord_l590_590513

theorem collinear_points_average_x_coord (x1 x2 x3 x4 : ℝ)
  (hx1 : y 2 x1 + y 7 x1 + y 3 x1 - y 5 = 2 * x1^4 + 7 * x1^3 + 3 * x1 - 5)
  (hx2 : y 2 x2 + y 7 x2 + y 3 x2 - y 5 = 2 * x2^4 + 7 * x2^3 + 3 * x2 - 5)
  (hx3 : y 2 x3 + y 7 x3 + y 3 x3 - y 5 = 2 * x3^4 + 7 * x3^3 + 3 * x3 - 5)
  (hx4 : y 2 x4 + y 7 x4 + y 3 x4 - y 5 = 2 * x4^4 + 7 * x4^3 + 3 * x4 - 5)
  (h_collinear : ∃ (m c : ℝ), ∀ i, i = x1 ∨ i = x2 ∨ i = x3 ∨ i = x4 → 2 * i^4 + 7 * i^3 + 3 * i - 5 = m * i + c) :
  (x1 + x2 + x3 + x4) / 4 = -7 / 8 :=
by
  sorry

end collinear_points_average_x_coord_l590_590513


namespace petya_time_comparison_l590_590687

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l590_590687


namespace probability_of_different_tens_digits_l590_590535

open Finset

-- Define the basic setup
def integers (n : ℕ) : Finset ℕ := {i in (range n) | i ≥ 10 ∧ i ≤ 79}

def tens_digit (n : ℕ) : ℕ := n / 10

def six_integers_with_different_tens_digits (s : Finset ℕ) : Prop :=
  s.card = 6 ∧ (s.map ⟨tens_digit, by simp⟩).card = 6

def favorable_ways : ℕ :=
  7 * 10^6

def total_ways : ℕ :=
  nat.choose 70 6

noncomputable def probability : ℚ :=
  favorable_ways / total_ways

-- The main statement
theorem probability_of_different_tens_digits :
  ∀ (s : Finset ℕ), six_integers_with_different_tens_digits s → 
  probability = 175 / 2980131 :=
begin
  intros s h,
  sorry
end

end probability_of_different_tens_digits_l590_590535


namespace correct_formulas_l590_590621

theorem correct_formulas (n : ℕ) :
  ((2 * n - 1)^2 - 4 * (n * (n - 1)) / 2) = (2 * n^2 - 2 * n + 1) ∧ 
  (1 + ((n - 1) * n) / 2 * 4) = (2 * n^2 - 2 * n + 1) ∧ 
  ((n - 1)^2 + n^2) = (2 * n^2 - 2 * n + 1) := by
  sorry

end correct_formulas_l590_590621


namespace problem1_problem2_l590_590999

-- Problem 1
theorem problem1 : (Real.exp (Real.log 3) - Real.logBase (Real.sqrt 2) (2 * Real.sqrt 2) + Real.pow 0.125 (2/3) + Real.root 2023 ((-2)^2023) = -7 / 4) := sorry

-- Problem 2
theorem problem2 (a : Real) (x : Real) (h : Real.pow a (2 * x) = 2) : 
  (a^(3 * x) + a^(-3 * x)) / (a^(x) + a^(-x)) = 3 / 2 := sorry

end problem1_problem2_l590_590999


namespace petya_time_l590_590692

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l590_590692


namespace work_days_difference_l590_590230

theorem work_days_difference (d_a d_b : ℕ) (H1 : d_b = 15) (H2 : d_a = d_b / 3) : 15 - d_a = 10 := by
  sorry

end work_days_difference_l590_590230


namespace gcd_of_72_and_90_l590_590212

theorem gcd_of_72_and_90 :
  Int.gcd 72 90 = 18 := 
sorry

end gcd_of_72_and_90_l590_590212


namespace quadrilateral_square_center_l590_590140

theorem quadrilateral_square_center (A B C D O : ℝ)
    (S : ℝ) (inside_quadrilateral : Point O inside quadrilateral ABCD) 
    (area_eq : 2 * S = OA^2 + OB^2 + OC^2 + OD^2) : 
    is_square ABCD ∧ is_center O ABCD :=
sorry

end quadrilateral_square_center_l590_590140


namespace polyhedron_volume_at_least_one_third_l590_590060

theorem polyhedron_volume_at_least_one_third
(cube : Type)
[divisible : Divisibility cube 1]
(polyhedron : Type)
(convex : convex polyhedron)
(projection_coincident : 
  ∀ face : Type, 
    face ∈ faces cube → 
    projection polyhedron face = face) :
  volume polyhedron ≥ (1 / 3) * volume cube :=
sorry

end polyhedron_volume_at_least_one_third_l590_590060


namespace probability_different_tens_digit_l590_590530

open Nat

theorem probability_different_tens_digit :
  let total_ways := choose 70 6,
      favorable_ways := 7 * 10^6
  in 
    (favorable_ways : ℝ) / total_ways = (2000 / 3405864 : ℝ) :=
by
  have h1 : total_ways = 70.choose 6 := rfl
  have h2 : favorable_ways = 7 * 10^6 := rfl
  rw [h1, h2]
  sorry

end probability_different_tens_digit_l590_590530


namespace cost_per_box_l590_590220

-- Define the conditions
def box_length := 20 -- in inches
def box_width := 20 -- in inches
def box_height := 12 -- in inches
def total_volume := 2.4 * 10^6 -- in cubic inches
def total_cost := 200 -- in dollars

-- Define the required proof
theorem cost_per_box (V_box : box_length * box_width * box_height = 4800) 
                      (num_boxes : total_volume / 4800 = 500) : 
                      total_cost / num_boxes = 0.40 :=
by
  sorry

end cost_per_box_l590_590220


namespace number_of_factors_l590_590721

theorem number_of_factors (M : ℕ) (hM : M = 2^4 * 3^3 * 5^2 * 7) : 
  ∃ n : ℕ, n = 120 ∧ (∀ d : ℕ, d ∣ M ↔ d = 2^a * 3^b * 5^c * 7^d ∧ 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 0 ≤ d ∧ d ≤ 1) :=
begin
  sorry
end

end number_of_factors_l590_590721


namespace equidistant_points_proof_l590_590260

noncomputable theory
open_locale classical

variables {O P : Point} {r d : ℝ}
variable (C : Circle O r)
variable (T1 : TangentLine O r d)
variable (T2 : TangentLineIntersection T1 P d)

def equidistant_points_count (C : Circle O r) (T1 T2 : TangentLine) : ℕ := sorry

theorem equidistant_points_proof
  (hC : Circle O r)
  (hT1 : TangentLine O r d)
  (hT2 : TangentLineIntersection hT1 P d)
  : equidistant_points_count hC hT1 hT2 = 3 :=
sorry

end equidistant_points_proof_l590_590260


namespace f_at_2_l590_590382

def isOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^3 + x^2
  else -2 * (-x)^3 - (-x)^2

theorem f_at_2 : f 2 = 12 := by
  sorry

end f_at_2_l590_590382


namespace fraction_of_area_above_line_l590_590940

open Real

-- Define the points and the line between them
noncomputable def pointA : (ℝ × ℝ) := (2, 3)
noncomputable def pointB : (ℝ × ℝ) := (5, 1)

-- Define the vertices of the square
noncomputable def square_vertices : List (ℝ × ℝ) := [(2, 1), (5, 1), (5, 4), (2, 4)]

-- Define the equation of the line
noncomputable def line_eq (x : ℝ) : ℝ :=
  (-2/3) * x + 13/3

-- Define the vertical and horizontal boundaries
noncomputable def x_min : ℝ := 2
noncomputable def x_max : ℝ := 5
noncomputable def y_min : ℝ := 1
noncomputable def y_max : ℝ := 4

-- Calculate the area of the triangle formed below the line
noncomputable def triangle_area : ℝ := 0.5 * 2 * 3

-- Calculate the area of the square
noncomputable def square_area : ℝ := 3 * 3

-- The fraction of the area above the line
noncomputable def area_fraction_above : ℝ := (square_area - triangle_area) / square_area

-- Prove the fraction of the area of the square above the line is 2/3
theorem fraction_of_area_above_line : area_fraction_above = 2 / 3 :=
  sorry

end fraction_of_area_above_line_l590_590940


namespace loss_percentage_is_correct_l590_590822

-- Definitions of the initial conditions
def total_cost : ℝ := 460
def cost_of_first_book : ℝ := 268.33
def gain_percentage : ℝ := 0.19

-- Calculate the cost of the second book
def cost_of_second_book := total_cost - cost_of_first_book

-- Calculate the selling price of the second book with a 19% gain
def selling_price_of_second_book := cost_of_second_book + (gain_percentage * cost_of_second_book)

-- Define that the selling price of both books is the same
def selling_price_of_first_book := selling_price_of_second_book

-- Calculate the loss on the first book
def loss := cost_of_first_book - selling_price_of_first_book

-- Calculate the loss percentage on the first book
def loss_percentage := (loss / cost_of_first_book) * 100

-- Lean statement to prove the loss percentage
theorem loss_percentage_is_correct : loss_percentage = 14.99 :=
by
  sorry

end loss_percentage_is_correct_l590_590822


namespace triangle_problem_part1_triangle_problem_part2_l590_590840

noncomputable def triangle_solution := 
  let a := 4
  let c := 3
  let cosA := -1 / 4
  let b := 2
  let sin_2B_plus_pi_6 := (17 + 21 * Real.sqrt 5) / 64
  (a, c, cosA, b, sin_2B_plus_pi_6)

theorem triangle_problem_part1 : 
  let (a, c, cosA, b) := (4, 3, -1/4, 2)
  ∃ b', b' = 2 :=
by { exact ⟨2, rfl⟩, sorry }

theorem triangle_problem_part2 : 
  let (a, c, cosA, b, sin_2B_plus_pi_6) := triangle_solution
  sin_2B_plus_pi_6 = (17 + 21 * Real.sqrt 5) / 64 :=
by sorry

end triangle_problem_part1_triangle_problem_part2_l590_590840


namespace domain_of_function_l590_590559

theorem domain_of_function {x : ℝ} : (0 ≤ x ∧ x < 1) ↔ (∃ y, y = x^\(1/2) + Real.logBase 2 (1 - x)) :=
by
  sorry

end domain_of_function_l590_590559


namespace height_of_cone_from_semicircle_l590_590654

theorem height_of_cone_from_semicircle (R : ℝ) : (∃ h : ℝ, h = sqrt (R^2 - (R/2)^2) ∧ h = sqrt 3 * R / 2) :=
sorry

end height_of_cone_from_semicircle_l590_590654


namespace Trumpington_marching_band_max_l590_590580

theorem Trumpington_marching_band_max (n : ℕ) (k : ℕ) 
  (h1 : 20 * n % 26 = 4)
  (h2 : n = 8 + 13 * k)
  (h3 : 20 * n < 1000) 
  : 20 * (8 + 13 * 3) = 940 := 
by
  sorry

end Trumpington_marching_band_max_l590_590580


namespace number_of_valid_quadruples_l590_590340

theorem number_of_valid_quadruples : 
  (∃ n : ℕ, n = 2017 ∧ 
  (∃ f : ℕ → ℕ → ℕ → ℕ → Prop, 
    (∀ a b c d : ℕ, a ∈ {1, 2, 3, 4, 5, 6, 7} → 
                     b ∈ {1, 2, 3, 4, 5, 6, 7} → 
                     c ∈ {1, 2, 3, 4, 5, 6, 7} → 
                     d ∈ {1, 2, 3, 4, 5, 6, 7} → 
                     (3 * a * b * c + 4 * a * b * d + 5 * b * c * d) % 2 = 0 →
                     f a b c d)) ∧ 
    fintype.card {t // f t.fst t.snd t.3 t.4} = 2017)) :=
begin
  sorry
end

end number_of_valid_quadruples_l590_590340


namespace valid_function_A_l590_590984

def is_function {A B : Type} (f : A → B) (S : set A) (T : set B) :=
  ∀ (x : A), x ∈ S → f x ∈ T

theorem valid_function_A :
  let A := {x : ℝ | x > 0}
  let B := {y : ℝ | y ≥ 0}
  is_function (λ x, 1/x) A B :=
by 
  sorry

end valid_function_A_l590_590984


namespace closest_integer_to_cube_root_of_150_l590_590608

theorem closest_integer_to_cube_root_of_150 : ∃ (n : ℤ), abs ((n: ℝ)^3 - 150) ≤ abs (((n + 1 : ℤ) : ℝ)^3 - 150) ∧
  abs ((n: ℝ)^3 - 150) ≤ abs (((n - 1 : ℤ) : ℝ)^3 - 150) ∧ n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590608


namespace a_n_formula_T_n_sum_l590_590048

noncomputable def a_n (n : ℕ) : ℕ :=
if n = 1 then 0 else n - 1

def S_n (n : ℕ) : ℕ := (Finset.range n).sum a_n

-- Theorem for the first part: Finding the general formula for {a_n}
theorem a_n_formula (n : ℕ) (h : n ≠ 0) : a_n (n + 1) = n := by
  sorry

noncomputable def T_n (n : ℕ) : ℝ :=
(Finset.range n).sum (λ k, (a_n (k + 1) + 1) / 2^(k + 1 : ℝ))

-- Theorem for the second part: Finding the sum T_n of the first n terms of the sequence { (a_n + 1) / 2^n }
theorem T_n_sum (n : ℕ) : T_n n = 2 - (n + 2) / 2^n := by
  sorry

end a_n_formula_T_n_sum_l590_590048


namespace evaluate_expression_is_39_l590_590322

noncomputable def evaluateExpression : ℕ :=
  let a := (16 : ℚ) / 5
  let term1 := (Real.ceil (Real.sqrt a))
  let term2 := (Real.ceil (a ^ 3))
  let term3 := (Real.ceil a)
  (term1 + term2 + term3).to_nat

theorem evaluate_expression_is_39 : evaluateExpression = 39 := by
  sorry

end evaluate_expression_is_39_l590_590322


namespace number_of_well_filled_subsets_eq_l590_590643

noncomputable def binom : ℕ → ℕ → ℕ
| n, k := if k ≤ n then nat.choose n k else 0

def is_well_filled (S : finset ℕ) : Prop :=
  ∀ (m ∈ S), (S.filter (λ x, x < m)).card < m / 2

def well_filled_subsets_count (n : ℕ) : ℕ :=
  (finset.powerset (finset.range (n + 1))).filter (λ S, S.nonempty ∧ is_well_filled S).card

theorem number_of_well_filled_subsets_eq :
  well_filled_subsets_count 42 = binom 43 21 - 1 :=
sorry

end number_of_well_filled_subsets_eq_l590_590643


namespace wallpaper_three_layers_l590_590958

theorem wallpaper_three_layers
  (A B C : ℝ)
  (hA : A = 300)
  (hB : B = 30)
  (wall_area : ℝ)
  (h_wall_area : wall_area = 180)
  (hC : C = A - (wall_area - B) - B)
  : C = 120 := by
  sorry

end wallpaper_three_layers_l590_590958


namespace probability_six_integers_diff_tens_l590_590522

-- Defining the range and conditions for the problem
def set_of_integers : Finset ℤ := Finset.range 70 \ Finset.range 10

def has_different_tens_digit (s : Finset ℤ) : Prop :=
  (s.card = 6) ∧ (∀ x y ∈ s, x ≠ y → (x / 10) ≠ (y / 10))

noncomputable def num_ways_choose_six_diff_tens : ℚ :=
  ((7 : ℚ) * (10^6 : ℚ))

noncomputable def total_ways_choose_six : ℚ :=
  (Nat.choose 70 6 : ℚ)

noncomputable def probability_diff_tens : ℚ :=
  num_ways_choose_six_diff_tens / total_ways_choose_six

-- Statement claiming the required probability
theorem probability_six_integers_diff_tens :
  probability_diff_tens = 1750 / 2980131 :=
by
  sorry

end probability_six_integers_diff_tens_l590_590522


namespace number_of_solutions_l590_590933

theorem number_of_solutions (f : ℝ → ℝ) (domain : set Icc (-4 : ℝ) 4) :
  (f.verify_domain domain) → (
  (∀ x ∈ domain, f x = 2 ↔ (x = -3 ∨ x = 0 ∨ x = 3)) ∧
  (∀ x ∈ domain, f x = 3 ↔ (x = -2 ∨ x ∈ Icc 3 4)) ∧
  (∀ x ∈ domain, f x = 0 ↔ x = -4)) →
  (∃ s : set ℝ, s.card = 3 ∧ ∀ x ∈ s, f(f(x)) = 2) :=
sorry

end number_of_solutions_l590_590933


namespace geometric_sequence_tenth_term_l590_590716

theorem geometric_sequence_tenth_term :
  let a := 4
  let r := (4 / 3 : ℚ)
  a * r ^ 9 = (1048576 / 19683 : ℚ) :=
by
  sorry

end geometric_sequence_tenth_term_l590_590716


namespace stream_cross_section_area_l590_590157

-- Definitions of the given conditions
def top_width : ℕ := 10
def bottom_width : ℕ := 6
def height : ℕ := 80

-- Statement of the proof
theorem stream_cross_section_area : (1 / 2) * (top_width + bottom_width) * height = 640 := by
  sorry

end stream_cross_section_area_l590_590157


namespace expected_lifetime_flashlight_l590_590081

noncomputable section

variables (ξ η : ℝ) -- lifetimes of the blue and red lightbulbs
variables [probability_space ℙ] -- assuming a probability space ℙ

-- condition: expected lifetime of the red lightbulb is 4 years
axiom expected_eta : ℙ.𝔼(η) = 4

-- the main proof problem
theorem expected_lifetime_flashlight : ℙ.𝔼(max ξ η) ≥ 4 :=
sorry

end expected_lifetime_flashlight_l590_590081


namespace total_slides_used_l590_590315

theorem total_slides_used (duration : ℕ) (initial_slides : ℕ) (initial_time : ℕ) (constant_rate : ℕ) (total_time: ℕ)
  (H1 : duration = 50)
  (H2 : initial_slides = 4)
  (H3 : initial_time = 2)
  (H4 : constant_rate = initial_slides / initial_time)
  (H5 : total_time = duration) 
  : (constant_rate * total_time) = 100 := 
by
  sorry

end total_slides_used_l590_590315


namespace minimum_sqrt_a_plus_fraction_minimum_sqrt_a_plus_fraction_equality_l590_590566

theorem minimum_sqrt_a_plus_fraction (a : ℝ) (ha : 0 < a) : 
  sqrt a + 4 / (sqrt a + 1) ≥ 3 :=
sorry

theorem minimum_sqrt_a_plus_fraction_equality :
  sqrt 1 + 4 / (sqrt 1 + 1) = 3 :=
by
  norm_num

end minimum_sqrt_a_plus_fraction_minimum_sqrt_a_plus_fraction_equality_l590_590566


namespace simplify_fractions_l590_590516

-- Define the fractions and their product.
def fraction1 : ℚ := 14 / 3
def fraction2 : ℚ := 9 / -42

-- Define the product of the fractions with scalar multiplication by 5.
def product : ℚ := 5 * fraction1 * fraction2

-- The target theorem to prove the equivalence.
theorem simplify_fractions : product = -5 := 
sorry  -- Proof is omitted

end simplify_fractions_l590_590516


namespace factorial_expression_l590_590592

namespace FactorialProblem

-- Definition of factorial function.
def factorial : ℕ → ℕ 
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Theorem stating the problem equivalently.
theorem factorial_expression : (factorial 12 - factorial 10) / factorial 8 = 11790 := by
  sorry

end FactorialProblem

end factorial_expression_l590_590592


namespace james_writing_hours_per_week_l590_590065

variables (pages_per_hour : ℕ) (pages_per_day_per_person : ℕ) (people : ℕ) (days_per_week : ℕ)

theorem james_writing_hours_per_week
  (h1 : pages_per_hour = 10)
  (h2 : pages_per_day_per_person = 5)
  (h3 : people = 2)
  (h4 : days_per_week = 7) :
  (pages_per_day_per_person * people * days_per_week) / pages_per_hour = 7 :=
by
  sorry

end james_writing_hours_per_week_l590_590065


namespace Jenna_has_20_less_than_Bob_l590_590831

-- Define amounts for Bob, Phil, Jenna, and John based on given conditions.
def Bob_amount : ℝ := 60
def Phil_amount : ℝ := (1/3) * Bob_amount
def Jenna_amount : ℝ := 2 * Phil_amount
def John_amount : ℝ := (1 + 0.35) * Phil_amount

-- The theorem to prove the difference between Bob's and Jenna's amounts.
theorem Jenna_has_20_less_than_Bob :
  Bob_amount - Jenna_amount = 20 :=
by
  -- Proof steps here, but adding sorry to skip for now.
  sorry

end Jenna_has_20_less_than_Bob_l590_590831


namespace petya_time_comparison_l590_590688

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l590_590688


namespace cosine_cone_proof_l590_590948

def cosine_angle_cone : ℝ :=
  let l := 18
  let sector_radius := 18
  let theta := 4 / 3 * Real.pi
  let r := (theta * l) / (2 * Real.pi)
  let cos_phi := r / l
  cos_phi

theorem cosine_cone_proof : cosine_angle_cone = 2 / 3 := by
  sorry

end cosine_cone_proof_l590_590948


namespace cleaning_cost_l590_590860

theorem cleaning_cost (num_cleanings : ℕ) (chemical_cost : ℕ) (monthly_cost : ℕ) (tip_percentage : ℚ) 
  (cleaning_sessions_per_month : num_cleanings = 30 / 3)
  (monthly_chemical_cost : chemical_cost = 2 * 200)
  (total_monthly_cost : monthly_cost = 2050)
  (cleaning_cost_with_tip : monthly_cost - chemical_cost =  num_cleanings * (1 + tip_percentage) * x) : 
  x = 150 := 
by
  sorry

end cleaning_cost_l590_590860


namespace petya_time_comparison_l590_590679

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l590_590679


namespace part_I_part_II_l590_590775

noncomputable def f (x b c : ℝ) : ℝ := x^2 + b * x + c

theorem part_I (b c : ℝ) (h : c = 4) (hx : f 3 b c = 1) : b = -4 :=
by {
  rw [h, f] at hx,
  simp at hx,
  linarith,
}

theorem part_II (b c : ℝ) (h1 : ∀ x, f x b c ≤ 1) 
(h2 : ∀ x, |x| > 2 → f x b c > 0) : 
  b + 1 / c ∈ set.Icc (-(34 / 7) : ℝ) (-(15 / 4) : ℝ) :=
sorry   -- Proof steps are skipped

end part_I_part_II_l590_590775


namespace probability_after_5_rounds_l590_590662

def initial_coins : ℕ := 5
def rounds : ℕ := 5
def final_probability : ℚ := 1 / 2430000

structure Player :=
  (name : String)
  (initial_coins : ℕ)
  (final_coins : ℕ)

def Abby : Player := ⟨"Abby", 5, 5⟩
def Bernardo : Player := ⟨"Bernardo", 4, 3⟩
def Carl : Player := ⟨"Carl", 3, 3⟩
def Debra : Player := ⟨"Debra", 4, 5⟩

def check_final_state (players : List Player) : Prop :=
  ∀ (p : Player), p ∈ players →
  (p.name = "Abby" ∧ p.final_coins = 5 ∨
   p.name = "Bernardo" ∧ p.final_coins = 3 ∨
   p.name = "Carl" ∧ p.final_coins = 3 ∨
   p.name = "Debra" ∧ p.final_coins = 5)

theorem probability_after_5_rounds :
  ∃ prob : ℚ, prob = final_probability ∧ check_final_state [Abby, Bernardo, Carl, Debra] :=
sorry

end probability_after_5_rounds_l590_590662


namespace complement_of_angle_l590_590932

theorem complement_of_angle (x : ℝ) (h : 90 - x = 3 * x + 10) : x = 20 := by
  sorry

end complement_of_angle_l590_590932


namespace min_value_fraction_l590_590884

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2 * y = 3) : 
  (∃ m, (∀ x y > 0, (x + 2 * y = 3) → (m ≤ (1 / x + 1 / y))) ∧ 
  (m = 1 + 2 * real.sqrt 2 / 3)) :=
by
  sorry

end min_value_fraction_l590_590884


namespace partA_partB_l590_590482

noncomputable def z (n : ℕ) : ℂ := Complex.exp (2 * Real.pi * Complex.I / n)

def sum_geom (n a : ℕ) : ℂ :=
  ∑ k in Finset.range n, (z(n) ^ (k * a))

theorem partA (n a : ℕ) : sum_geom n a = if a % n = 0 then n else 0 :=
sorry

def sum_weighted_geom (n a : ℕ) : ℂ :=
  ∑ k in Finset.range n, (k + 1) * (z(n) ^ (k * a))

theorem partB (n a : ℕ) :
  sum_weighted_geom n a =
  if a % n = 0 then (n*(n-1))/2
  else (n / (2 * Complex.sin (a * Real.pi / n))) * (Complex.sin (a * Real.pi / n) - Complex.I * Complex.cos (a * Real.pi / n)) :=
sorry

end partA_partB_l590_590482


namespace correct_simplification_a_l590_590226

def simplifications (n : Int) : Int :=
  if n = 1 then -(+6)
  else if n = 2 then -(-17)
  else if n = 3 then +(-9)
  else if n = 4 then +(+5)
  else 0

theorem correct_simplification_a :
  (simplifications 1 = -6) ∧
  (simplifications 2 ≠ -17) ∧
  (simplifications 3 ≠ 9) ∧
  (simplifications 4 ≠ -5) :=
by
  sorry

end correct_simplification_a_l590_590226


namespace Petya_time_comparison_l590_590674

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l590_590674


namespace probability_sum_ten_l590_590966

-- Define the properties and conditions
def dice_faces : Finset ℕ := Finset.range 9 \ {0}

-- The question translated to Lean 4 statement
theorem probability_sum_ten :
  let outcomes := { (a, b) | a ∈ dice_faces ∧ b ∈ dice_faces } in
  let favorable := { (a, b) | a + b = 10 ∧ a ∈ dice_faces ∧ b ∈ dice_faces } in
  (favorable.card : ℚ) / outcomes.card = 5 / 64 :=
sorry

end probability_sum_ten_l590_590966


namespace range_of_m_for_p_range_of_m_for_p_or_q_l590_590791

-- Define the propositions p and q
def p (m : ℝ) := ∃ (a b : ℝ) (h1 : a < 0) (h2 : b < 0), (x^2 + m*x + 1 = 0)
def q (m : ℝ) := ∀ (a : ℝ), x = -((m - 2)*x^2)/(4x^2 + 4(m - 2)^2 + 1)

-- Prove the range of m for proposition p
theorem range_of_m_for_p (m : ℝ) (hp : p m) : m > 2 := sorry

-- Prove the range of m when one of p or q is true and the other is false
theorem range_of_m_for_p_or_q (m : ℝ) (h : (p m ∧ ¬q m) ∨ (¬p m ∧ q m)) : m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 := sorry

end range_of_m_for_p_range_of_m_for_p_or_q_l590_590791


namespace closest_integer_to_cube_root_of_150_l590_590605

theorem closest_integer_to_cube_root_of_150 : 
  let cbrt := (150: ℝ)^(1/3) in
  abs (cbrt - 6) < abs (cbrt - 5) :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590605


namespace right_triangle_properties_l590_590168

theorem right_triangle_properties (a b c : ℝ) (h1 : c = 13) (h2 : a = 5)
  (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 30 ∧ a + b + c = 30 := by
  sorry

end right_triangle_properties_l590_590168


namespace ball_probability_l590_590258

theorem ball_probability :
  (∃ (total red purple black number_favorable : ℕ),
    total = 500 ∧
    red = 57 ∧
    purple = 33 ∧
    black = 30 ∧
    number_favorable = total - (red + purple + black) ∧
    (number_favorable / total : ℚ) = 19 / 25) :=
begin
  sorry,
end

end ball_probability_l590_590258


namespace product_of_digits_of_smallest_non_divisible_by_3_l590_590898

-- Definition to check the divisibility by 3
def divisible_by_3 (n : Nat) : Prop :=
  (n.digits 10).sum % 3 = 0

-- Definition of the list of numbers
def numbers : List Nat := [3545, 3555, 3565, 3573, 3577]

-- Definition to extract the units digit of a number
def units_digit (n : Nat) : Nat :=
  n % 10

-- Definition to extract the tens digit of a number
def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

-- The statement for the proof
theorem product_of_digits_of_smallest_non_divisible_by_3 :
  let non_divisible = numbers.filter (λ n => ¬ divisible_by_3 n)
  let smallest_non_divisible := non_divisible.min
  smallest_non_divisible.units_digit * smallest_non_divisible.tens_digit = 20 :=
by
  sorry

end product_of_digits_of_smallest_non_divisible_by_3_l590_590898


namespace find_DG_l590_590913

theorem find_DG (a b k l : ℕ) (h1 : a * k = 37 * (a + b)) (h2 : b * l = 37 * (a + b)) : 
  k = 1406 :=
by
  sorry

end find_DG_l590_590913


namespace unique_combined_friends_count_l590_590063

theorem unique_combined_friends_count 
  (james_friends : ℕ)
  (susan_friends : ℕ)
  (john_multiplier : ℕ)
  (shared_friends : ℕ)
  (maria_shared_friends : ℕ)
  (maria_friends : ℕ)
  (h_james : james_friends = 90)
  (h_susan : susan_friends = 50)
  (h_john : ∃ (john_friends : ℕ), john_friends = john_multiplier * susan_friends ∧ john_multiplier = 4)
  (h_shared : shared_friends = 35)
  (h_maria_shared : maria_shared_friends = 10)
  (h_maria : maria_friends = 80) :
  ∃ (total_unique_friends : ℕ), total_unique_friends = 325 :=
by
  -- Proof is omitted
  sorry

end unique_combined_friends_count_l590_590063


namespace midpoint_on_incircle_l590_590903

open Complex

/-- Given that Z and W are isogonally conjugate with respect to an equilateral
triangle centered at the origin with circumradius 1, and they are inversed with
respect to the circumcircle to become Z* and W*, respectively. We need to show
that the midpoint of Z* and W* lies on the incircle of the equilateral triangle. -/
theorem midpoint_on_incircle {Z W Z* W* : ℂ}
  (h_iso_conj : ∃ z w : ℂ, z + w + z.conj * w.conj = 0 ∧
    Z = z ∧ W = w ∧ Z* = -1 / z.conj ∧ W* = -1 / w.conj)
  (circumradius : ∀ x : ℂ, x ∈ {Z, W, Z*, W*} → abs x = 1) :
  abs ((Z* + W*) / 2) = 1 / 2 :=
by
  sorry

end midpoint_on_incircle_l590_590903


namespace isosceles_triangle_of_angle_ratios_l590_590141

theorem isosceles_triangle_of_angle_ratios
    (α β γ : ℝ)
    (h1 : α + β + γ = 180)
    (h2 : α / β = 1 / 2)
    (h3 : β / γ = 1 / 2) :
    ∃ D E F,
      angle_bisector α β γ D E F ∧
      isosceles_triangle D E F :=
sorry

end isosceles_triangle_of_angle_ratios_l590_590141


namespace find_m_l590_590381

theorem find_m (f : ℝ → ℝ) (m : ℝ) 
  (h_even : ∀ x, f (-x) = f x) 
  (h_fx : ∀ x, 0 < x → f x = 4^(m - x)) 
  (h_f_neg2 : f (-2) = 1/8) : 
  m = 1/2 := 
by 
  sorry

end find_m_l590_590381


namespace exist_matrices_with_dets_l590_590789

noncomputable section

open Matrix BigOperators

variables {α : Type} [Field α] [DecidableEq α]

theorem exist_matrices_with_dets (m n : ℕ) (h₁ : 1 < m) (h₂ : 1 < n)
  (αs : Fin m → α) (β : α) :
  ∃ (A : Fin m → Matrix (Fin n) (Fin n) α), (∀ i, det (A i) = αs i) ∧ det (∑ i, A i) = β :=
sorry

end exist_matrices_with_dets_l590_590789


namespace max_value_fraction_l590_590473

theorem max_value_fraction
  (x y : ℝ) (k : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hk : 0 < k) :
  (sup (λ x y, (kx + y)^2 / (x^2 + ky^2))) = k + 1 := 
sorry

end max_value_fraction_l590_590473


namespace initial_men_count_l590_590026

theorem initial_men_count
  (M : ℕ)
  (h1 : ∀ T : ℕ, (M * 8 * 10 = T) → (5 * 16 * 12 = T)) :
  M = 12 :=
by
  sorry

end initial_men_count_l590_590026


namespace stripe_area_l590_590256

-- Definitions based on conditions
def diameter : ℝ := 40
def stripe_width : ℝ := 4
def revolutions : ℝ := 3

-- The statement we want to prove
theorem stripe_area (π : ℝ) : 
  (revolutions * π * diameter * stripe_width) = 480 * π :=
by
  sorry

end stripe_area_l590_590256


namespace exists_finite_set_with_distance_property_l590_590138

theorem exists_finite_set_with_distance_property (m : ℕ) (h : 0 < m) : 
  ∃ S : finset (ℝ × ℝ), S.nonempty ∧ ∀ A ∈ S, (finset.filter (λ B, real.dist A B = 1) S).card = m :=
by
  sorry

end exists_finite_set_with_distance_property_l590_590138


namespace standard_normal_interval_probability_l590_590928

noncomputable def standard_normal_cdf : ℝ → ℝ := sorry

theorem standard_normal_interval_probability (p : ℝ) :
  (∀ ξ : ℝ, standard_normal_cdf ξ = (1/2) * (1 + real.erf(ξ / real.sqrt 2))) →
  (standard_normal_cdf 1 = 1 - p) →
  (standard_normal_cdf (-1) = p) →
  (standard_normal_cdf 0 - standard_normal_cdf (-1)) = (1/2) - p :=
by
  intros h_cdf h_p1 h_p_minus1
  sorry

end standard_normal_interval_probability_l590_590928


namespace apples_per_classmate_l590_590830

theorem apples_per_classmate 
  (total_apples : ℕ) 
  (people : ℕ) 
  (h : total_apples = 15) 
  (p : people = 3) : 
  total_apples / people = 5 :=
by
  rw [h, p]
  norm_num

end apples_per_classmate_l590_590830


namespace amusement_park_ticket_length_l590_590651

theorem amusement_park_ticket_length (Area Width Length : ℝ) (h₀ : Area = 1.77) (h₁ : Width = 3) (h₂ : Area = Width * Length) : Length = 0.59 :=
by
  -- Proof will go here
  sorry

end amusement_park_ticket_length_l590_590651


namespace expected_lifetime_flashlight_l590_590095

noncomputable def xi : ℝ := sorry
noncomputable def eta : ℝ := sorry

def T : ℝ := max xi eta

axiom E_eta_eq_4 : E eta = 4

theorem expected_lifetime_flashlight : E T ≥ 4 :=
by
  -- The solution will go here
  sorry

end expected_lifetime_flashlight_l590_590095


namespace probability_different_tens_digit_l590_590527

open Nat

theorem probability_different_tens_digit :
  let total_ways := choose 70 6,
      favorable_ways := 7 * 10^6
  in 
    (favorable_ways : ℝ) / total_ways = (2000 / 3405864 : ℝ) :=
by
  have h1 : total_ways = 70.choose 6 := rfl
  have h2 : favorable_ways = 7 * 10^6 := rfl
  rw [h1, h2]
  sorry

end probability_different_tens_digit_l590_590527


namespace determine_card_l590_590892

def card :=
  {suit : String, rank : String}

def all_cards : List card :=
  [{suit := "Hearts", rank := "A"}, {suit := "Hearts", rank := "Q"}, {suit := "Hearts", rank := "4"},
   {suit := "Spades", rank := "J"}, {suit := "Spades", rank := "8"}, {suit := "Spades", rank := "4"},
   {suit := "Spades", rank := "2"}, {suit := "Spades", rank := "7"}, {suit := "Spades", rank := "3"},
   {suit := "Clubs", rank := "K"}, {suit := "Clubs", rank := "Q"}, {suit := "Clubs", rank := "5"},
   {suit := "Clubs", rank := "4"}, {suit := "Clubs", rank := "6"},
   {suit := "Diamonds", rank := "A"}, {suit := "Diamonds", rank := "5"}]

def deduction_conditions (rank_hint_known_by_qian : String → Prop)
                         (suit_hint_known_by_sun : String → Prop)
                         (conversation_heard_by_zhao : (String × String) → Prop) : Prop :=
  (rank_hint_known_by_qian "A" ∨ rank_hint_known_by_qian "Q" ∨ rank_hint_known_by_qian "5" ∨ 
   rank_hint_known_by_qian "4") ∧ 
  (suit_hint_known_by_sun "Hearts" ∨ suit_hint_known_by_sun "Diamonds") ∧ 
  ∀ (r : String), rank_hint_known_by_qian r → r ≠ "A" →
  ∀ (s : String), suit_hint_known_by_sun s → conversation_heard_by_zhao (s, r)

theorem determine_card : deduction_conditions (fun r => r = "Q" ∨ r = "4" ∨ r = "5")
                                                (fun s => s = "Hearts" ∨ s = "Diamonds")
                                                (fun card => card = ("Diamonds", "5")) →
                          ∃ (card : String × String), card = ("Diamonds", "5") :=
by
  sorry

end determine_card_l590_590892


namespace regular_polygon_sides_l590_590745

theorem regular_polygon_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle i = 160)) : n = 18 :=
by
  -- Proof goes here
  sorry

end regular_polygon_sides_l590_590745


namespace petya_time_comparison_l590_590686

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l590_590686


namespace expected_lifetime_flashlight_l590_590084

noncomputable section

variables (ξ η : ℝ) -- lifetimes of the blue and red lightbulbs
variables [probability_space ℙ] -- assuming a probability space ℙ

-- condition: expected lifetime of the red lightbulb is 4 years
axiom expected_eta : ℙ.𝔼(η) = 4

-- the main proof problem
theorem expected_lifetime_flashlight : ℙ.𝔼(max ξ η) ≥ 4 :=
sorry

end expected_lifetime_flashlight_l590_590084


namespace train_speed_in_km_hr_l590_590277

-- Definitions based on conditions
def train_length : ℝ := 150  -- meters
def crossing_time : ℝ := 6  -- seconds

-- Definition for conversion factor
def meters_per_second_to_km_per_hour (speed_mps : ℝ) : ℝ := speed_mps * 3.6

-- Main theorem
theorem train_speed_in_km_hr : meters_per_second_to_km_per_hour (train_length / crossing_time) = 90 :=
by
  sorry

end train_speed_in_km_hr_l590_590277


namespace circumcircle_tangent_l590_590073

-- Defining the geometric objects and conditions
variables {Γ₁ Γ₂ : Type} [circle Γ₁] [circle Γ₂]
variables (P Q A B C R : point)

-- Conditions according to the problem
axiom circles_intersect : intersects Γ₁ Γ₂ P Q
axiom tangent_closer_P : tangent_closer P Γ₁ Γ₂ A B
axiom tangent_at_P_meets_Γ₂ : touches_at_P Γ₁ P C

-- Extension of AP meeting BC at R
axiom extension_AP_BC : meets (extension A P) (line B C) R

-- Prove the circumcircle of triangle PQR is tangent to BP and BR
theorem circumcircle_tangent (hΓ₁ : ∀ (x : point), x ∈ Γ₁ ↔ circle_center Γ₁ x)
  (hΓ₂ : ∀ (x : point), x ∈ Γ₂ ↔ circle_center Γ₂ x)
  (htangent_P : ∀ (x : point), tangent Γ₁ P x = C) :
  is_tangent (circumcircle P Q R) (line B P) ∧ is_tangent (circumcircle P Q R) (line B R) := 
sorry

end circumcircle_tangent_l590_590073


namespace refills_needed_l590_590704

theorem refills_needed 
  (cups_per_day : ℕ)
  (bottle_capacity_oz : ℕ)
  (oz_per_cup : ℕ)
  (total_oz : ℕ)
  (refills : ℕ)
  (h1 : cups_per_day = 12)
  (h2 : bottle_capacity_oz = 16)
  (h3 : oz_per_cup = 8)
  (h4 : total_oz = cups_per_day * oz_per_cup)
  (h5 : refills = total_oz / bottle_capacity_oz) :
  refills = 6 :=
by
  sorry

end refills_needed_l590_590704


namespace brownie_pieces_l590_590863

theorem brownie_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) 
  (h1 : pan_length = 24) (h2 : pan_width = 15) (h3 : piece_length = 3) (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := 
by
  -- Condition definitions
  let area_pan := pan_length * pan_width
  let area_piece := piece_length * piece_width
  have h5 : area_pan = 24 * 15 := by rw [h1, h2]
  have h6 : area_piece = 3 * 2 := by rw [h3, h4]
  
  -- Proof goals
  calc
    (area_pan / area_piece) = (24 * 15) / (3 * 2) : by rw [h5, h6]
                       ... = 360 / 6            : by norm_num
                       ... = 60                : by norm_num
  sorry

end brownie_pieces_l590_590863


namespace abs_eq_neg_iff_nonpositive_l590_590420

theorem abs_eq_neg_iff_nonpositive (x : ℝ) : |x| = -x ↔ x ≤ 0 := by
  sorry

end abs_eq_neg_iff_nonpositive_l590_590420


namespace shirts_not_washed_l590_590918

def total_shortsleeve_shirts : Nat := 40
def total_longsleeve_shirts : Nat := 23
def washed_shirts : Nat := 29

theorem shirts_not_washed :
  (total_shortsleeve_shirts + total_longsleeve_shirts) - washed_shirts = 34 :=
by
  sorry

end shirts_not_washed_l590_590918


namespace absents_probability_is_correct_l590_590029

-- Conditions
def probability_absent := 1 / 10
def probability_present := 9 / 10

-- Calculation of combined probability
def combined_probability : ℚ :=
  3 * (probability_absent * probability_absent * probability_present)

-- Conversion to percentage
def percentage_probability : ℚ :=
  combined_probability * 100

-- Theorem statement
theorem absents_probability_is_correct :
  percentage_probability = 2.7 := 
sorry

end absents_probability_is_correct_l590_590029


namespace area_of_PAB_l590_590376

open Classical

noncomputable def triangle_area {A B C P : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space P] (area_ABC : ℝ) : Prop :=
  sorry -- Define the area of triangles in terms of points A, B, C

theorem area_of_PAB (A B C P : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space P]
  (area_ABC : triangle_area A B C 360)
  (H : ∀ x : P, x = (1/4) • (A : P) + (1/4) • (B : P))
  : triangle_area A P B 90 :=
  sorry


end area_of_PAB_l590_590376


namespace negative_comparison_l590_590708

theorem negative_comparison : -2023 > -2024 :=
sorry

end negative_comparison_l590_590708


namespace number_of_people_chose_soda_l590_590432

theorem number_of_people_chose_soda (total_people : ℕ) (soda_angle : ℝ) (fraction_soda : ℝ) 
  (total_people_eq : total_people = 600) (soda_angle_eq : soda_angle = 108) 
  (fraction_soda_eq : fraction_soda = soda_angle / 360) :
  total_people * fraction_soda = 180 :=
by
  rw [total_people_eq, soda_angle_eq, fraction_soda_eq]
  norm_num
  sorry

end number_of_people_chose_soda_l590_590432


namespace number_of_correct_propositions_l590_590772

-- Definitions based on the given conditions
variables {P A B C : Type}

-- Assume specific properties
variable (h1 : PA = PB)
variable (h2 : PB = PC)

-- Defining the propositions
def prop1 : Prop := ¬(triangle_is_equilateral A B C)

def prop2 : Prop := ¬(foot_of_perpendicular_is_incenter P A B C)

def prop3 : Prop := foot_of_perpendicular_is_circumcenter P A B C

def prop4 : Prop := ¬(foot_of_perpendicular_is_orthocenter P A B C)

-- Stating the main theorem
theorem number_of_correct_propositions
(h1 : PA = PB) (h2 : PB = PC)
(h3 : foot_of_perpendicular_is_circumcenter P A B C) :
  (prop1 ∧ prop2 ∧ prop3 ∧ prop4) → (number_of_correct (prop1, prop2, prop3, prop4) = 1) := by
sory

end number_of_correct_propositions_l590_590772


namespace total_prayers_in_a_week_l590_590899

def prayers_per_week (pastor_prayers : ℕ → ℕ) : ℕ :=
  (pastor_prayers 0) + (pastor_prayers 1) + (pastor_prayers 2) +
  (pastor_prayers 3) + (pastor_prayers 4) + (pastor_prayers 5) + (pastor_prayers 6)

def pastor_paul (day : ℕ) : ℕ :=
  if day = 6 then 40 else 20

def pastor_bruce (day : ℕ) : ℕ :=
  if day = 6 then 80 else 10

def pastor_caroline (day : ℕ) : ℕ :=
  if day = 6 then 30 else 10

theorem total_prayers_in_a_week :
  prayers_per_week pastor_paul + prayers_per_week pastor_bruce + prayers_per_week pastor_caroline = 390 :=
sorry

end total_prayers_in_a_week_l590_590899


namespace lattice_point_exists_inside_l590_590701

-- Define a lattice point in the 2D integer coordinate grid
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Define a lattice (convex) pentagon using five lattice points A, B, C, D, and E
structure LatticePentagon where
  A B C D E : LatticePoint
  convex : Convex (Set.Union (Set.Union (Set.Union (convex_hull_4 A B C D) (convex_hull_4 A B D E)) (convex_hull_4 A C D E)) (convex_hull_4 B C D E))

-- The main theorem to prove the existence of a lattice point inside a convex lattice pentagon
theorem lattice_point_exists_inside (P : LatticePentagon) :
  ∃ (X : LatticePoint), X ∈ interior (convex_hull_5 P.A P.B P.C P.D P.E) :=
by
  sorry -- This is where the proof would go

end lattice_point_exists_inside_l590_590701


namespace ratio_JL_JM_l590_590912

theorem ratio_JL_JM (s w h : ℝ) (shared_area_25 : 0.25 * s^2 = 0.4 * w * h) (jm_eq_s : h = s) :
  w / h = 5 / 8 :=
by
  -- Proof will go here
  sorry

end ratio_JL_JM_l590_590912


namespace coordinates_of_point_P_in_third_quadrant_l590_590944

noncomputable def distance_from_y_axis (P : ℝ × ℝ) : ℝ := abs P.1
noncomputable def distance_from_x_axis (P : ℝ × ℝ) : ℝ := abs P.2

theorem coordinates_of_point_P_in_third_quadrant : 
  ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 < 0 ∧ distance_from_x_axis P = 2 ∧ distance_from_y_axis P = 5 ∧ P = (-5, -2) :=
by
  sorry

end coordinates_of_point_P_in_third_quadrant_l590_590944


namespace circle_centers_triangle_area_l590_590196

noncomputable def area_of_triangle (R1 R2 R3 : ℝ) (d_OP d_OQ d_PQ : ℝ) : ℝ :=
  if h : R1 = 2 ∧ R2 = 3 ∧ R3 = 3 ∧ d_OP = 5 ∧ d_OQ = 5 ∧ d_PQ = 6 then
    1 / 2 * d_PQ * 4
  else
    0

theorem circle_centers_triangle_area :
  area_of_triangle 2 3 3 5 5 6 = 12 :=
by
  dsimp [area_of_triangle]
  split_ifs
  · rfl
  · contradiction

end circle_centers_triangle_area_l590_590196


namespace closest_integer_to_cube_root_of_150_l590_590606

theorem closest_integer_to_cube_root_of_150 : 
  let cbrt := (150: ℝ)^(1/3) in
  abs (cbrt - 6) < abs (cbrt - 5) :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590606


namespace upper_limit_of_raise_l590_590143

theorem upper_limit_of_raise (lower upper : ℝ) (h_lower : lower = 0.05)
  (h_upper : upper > 0.08) (h_inequality : ∀ r, lower < r → r < upper)
  : upper < 0.09 :=
sorry

end upper_limit_of_raise_l590_590143


namespace probability_six_integers_unique_tens_digit_l590_590543

theorem probability_six_integers_unique_tens_digit :
  (∃ (x1 x2 x3 x4 x5 x6 : ℕ),
    10 ≤ x1 ∧ x1 ≤ 79 ∧
    10 ≤ x2 ∧ x2 ≤ 79 ∧
    10 ≤ x3 ∧ x3 ≤ 79 ∧
    10 ≤ x4 ∧ x4 ≤ 79 ∧
    10 ≤ x5 ∧ x5 ≤ 79 ∧
    10 ≤ x6 ∧ x6 ≤ 79 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x1 ≠ x6 ∧
    x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x2 ≠ x6 ∧
    x3 ≠ x4 ∧ x3 ≠ x5 ∧ x3 ≠ x6 ∧
    x4 ≠ x5 ∧ x4 ≠ x6 ∧
    x5 ≠ x6 ∧
    tens_digit x1 ≠ tens_digit x2 ∧
    tens_digit x1 ≠ tens_digit x3 ∧
    tens_digit x1 ≠ tens_digit x4 ∧
    tens_digit x1 ≠ tens_digit x5 ∧
    tens_digit x1 ≠ tens_digit x6 ∧
    tens_digit x2 ≠ tens_digit x3 ∧
    tens_digit x2 ≠ tens_digit x4 ∧
    tens_digit x2 ≠ tens_digit x5 ∧
    tens_digit x2 ≠ tens_digit x6 ∧
    tens_digit x3 ≠ tens_digit x4 ∧
    tens_digit x3 ≠ tens_digit x5 ∧
    tens_digit x3 ≠ tens_digit x6 ∧
    tens_digit x4 ≠ tens_digit x5 ∧
    tens_digit x4 ≠ tens_digit x6 ∧
    tens_digit x5 ≠ tens_digit x6)
    →
  (probability := \(\frac{4375}{744407}\)).sorry

end probability_six_integers_unique_tens_digit_l590_590543


namespace trisect_chord_exists_l590_590111

noncomputable def distance (O P : Point) : ℝ := sorry
def trisect (P : Point) (A B : Point) : Prop := 2 * (distance A P) = distance P B

-- Main theorem based on the given conditions and conclusions
theorem trisect_chord_exists (O P : Point) (r : ℝ) (hP_in_circle : distance O P < r) :
  (∃ A B : Point, trisect P A B) ↔ 
  (distance O P > r / 3 ∨ distance O P = r / 3) :=
by
  sorry

end trisect_chord_exists_l590_590111


namespace gold_to_brown_ratio_l590_590972

theorem gold_to_brown_ratio :
  ∃ (num_gold num_brown : ℕ), 
  num_brown = 4 ∧ 
  (∃ (num_blue : ℕ), 
  num_blue = 60 ∧ 
  num_blue = 5 * num_gold) ∧ 
  (num_gold : ℚ) / num_brown = 3 :=
by
  sorry

end gold_to_brown_ratio_l590_590972


namespace arrangements_of_volunteers_l590_590953

theorem arrangements_of_volunteers (volunteers places : ℕ) (h_volunteers : volunteers = 5) (h_places : places = 3) :
  let total_arrangements := places^volunteers - places * (places - 1)^volunteers + places in
  total_arrangements = 150 :=
by
  apply Eq.symm
  cases h_volunteers with rfl
  cases h_places with rfl
  have step1: 3^5 = 243 := by norm_num
  have step2: 3 * 2^5 = 96 := by norm_num
  have step3: 3 * 1^5 = 3 := by norm_num
  show 243 - 96 + 3 = 150
  rw [step1, step2, step3, Nat.sub_add, Nat.sub_eq_of_eq_add]
  exact (rfl : 150 = 150)

end arrangements_of_volunteers_l590_590953


namespace analyze_a_b_m_n_l590_590074

theorem analyze_a_b_m_n (a b m n : ℕ) (ha : 1 < a) (hb : 1 < b) (hm : 1 < m) (hn : 1 < n)
  (h1 : Prime (a^n - 1))
  (h2 : Prime (b^m + 1)) :
  n = 2 ∧ ∃ k : ℕ, m = 2^k :=
by
  sorry

end analyze_a_b_m_n_l590_590074


namespace gcd_adjacent_arrangement_count_l590_590291

theorem gcd_adjacent_arrangement_count :
  let numbers := [2, 3, 4, 6, 8, 9, 12, 15],
      gcd_adjacent (l : List ℕ) : Prop :=
        ∀ i, i < l.length - 1 → Nat.gcd (l.nthLe i (by linarith)) (l.nthLe (i + 1) (by linarith)) > 1 
  in ∃ l : List ℕ, l.perm numbers ∧ gcd_adjacent l ∧ l.countp gcd_adjacent = 1296 :=
sorry

end gcd_adjacent_arrangement_count_l590_590291


namespace flower_arrangements_l590_590190

open Classical

variables (pots : Finset ℕ) (yellow white red : Finset ℕ)
variables (adjacent : ∀ Y1 Y2 ∈ yellow, ∃ seq : List ℕ, seq = Y1 :: Y2 :: List.nil ∨ seq = Y2 :: Y1 :: List.nil)
variables (not_adjacent : ∀ W1 W2 ∈ white, ¬ ∃ seq : List ℕ, seq = List.insert_nth W1 (W2 :: List.nil) [W2])

theorem flower_arrangements (h_pots : pots.card = 5)
  (h_yellow : yellow.card = 2)
  (h_white : white.card = 2)
  (h_red : red.card = 1)
  (h_adj_yellow : adjacent yellow yellow)
  (h_not_adj_white : not_adjacent white white) : 
    ∃! n : ℕ, n = 24 := by 
  sorry

end flower_arrangements_l590_590190


namespace find_C2_eq_and_AB_distance_l590_590241

-- Let C1 be the curve defined by the polar equation ρ = 4 * sin θ.
-- Let C2 be the curve defined by the condition OP = 2 * OM for points on C1.
-- Let A be the intersection of ray θ with C1, and B be the intersection of ray θ with C2.

theorem find_C2_eq_and_AB_distance (θ : ℝ) :
  (∀ P M O: EuclideanSpace ℝ, curve_C1 : M ∈ curve_C1 ∧ OP = 2 * OM → P ∈ curve_C2) ∧ 
  let ρ1 := 4 * sin θ,
      ρ2 := 8 * sin θ in
  |ρ2 - ρ1| = 4 * sin θ :=
sorry

end find_C2_eq_and_AB_distance_l590_590241


namespace number_of_permutations_EXCEED_l590_590310

-- Defining the conditions
def word := ['E', 'X', 'C', 'E', 'E', 'D']

def total_letters := word.length -- 6

def count_E := (word.count 'E') -- 3

-- Defining the problem to prove
theorem number_of_permutations_EXCEED : 
  (Nat.factorial total_letters) / 
  (Nat.factorial count_E) = 120 := 
by 
s

end number_of_permutations_EXCEED_l590_590310


namespace expected_lifetime_of_flashlight_at_least_4_l590_590088

-- Definitions for the lifetimes of the lightbulbs
variable (ξ η : ℝ)

-- Condition: The expected lifetime of the red lightbulb is 4 years.
axiom E_η_eq_4 : 𝔼[η] = 4

-- Definition stating the lifetime of the flashlight
def T := max ξ η

theorem expected_lifetime_of_flashlight_at_least_4 
  (h : 𝔼η = 4) :
  𝔼[max ξ η] ≥ 4 :=
by {
  sorry
}

end expected_lifetime_of_flashlight_at_least_4_l590_590088


namespace probability_unique_tens_digits_l590_590545

theorem probability_unique_tens_digits :
  let num_ways := 10^6 in
  let total_combinations := Nat.choose 70 6 in
  (num_ways : ℚ) / total_combinations = 625 / 74440775 :=
by 
  sorry

end probability_unique_tens_digits_l590_590545


namespace geometric_sum_comparison_l590_590149

variable {a : ℕ → ℝ} (n : ℕ) (q : ℝ) 

-- Assumptions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range(n + 1), a i

def S_n := sum_first_n_terms a n
def S_n_plus_1 := sum_first_n_terms a (n + 1)

-- Condition
def q_positive_and_not_zero (q : ℝ) : Prop := q > 0

theorem geometric_sum_comparison 
    (h1 : geometric_sequence a q) 
    (h2 : q_positive_and_not_zero q) : 
    S_n a (n + 1) * a n > S_n a n * a (n + 1) :=
sorry

end geometric_sum_comparison_l590_590149


namespace marbles_lost_l590_590861

theorem marbles_lost (m_initial m_current : ℕ) (h_initial : m_initial = 19) (h_current : m_current = 8) : m_initial - m_current = 11 :=
by {
  sorry
}

end marbles_lost_l590_590861


namespace work_completion_time_l590_590257

theorem work_completion_time (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 12) (hAC : A + C = 1 / 2) :
  1 / (B + C) = 3 :=
by
  -- The proof goes here
  sorry

end work_completion_time_l590_590257


namespace positive_integers_n_l590_590304

theorem positive_integers_n (n a b : ℕ) (h1 : 2 < n) (h2 : n = a ^ 3 + b ^ 3) 
  (h3 : ∀ d, d > 1 ∧ d ∣ n → a ≤ d) (h4 : b ∣ n) : n = 16 ∨ n = 72 ∨ n = 520 :=
sorry

end positive_integers_n_l590_590304


namespace sum_q_t_12_l590_590468

open Finset

-- Definition of the set of 12-tuples where each entry is 0 or 1
def T : Finset (Fin 12 → ℕ) := univ.filter (λ t, ∀ i, t i = 0 ∨ t i = 1)

-- Definition of q_t(x)
def q_t (t : Fin 12 → ℕ) : ℕ → ℕ :=
  λ x, if h : 0 ≤ x ∧ x < 12 then t ⟨x, h.right⟩ else 0

-- Definition of the sum of q_t(x) over the set T
def q (x : ℕ) : ℕ := T.sum (λ t, q_t t x)

-- Statement to prove
theorem sum_q_t_12 : q 12 = 1024 :=
sorry

end sum_q_t_12_l590_590468


namespace smallest_num_assignments_25_points_l590_590640

theorem smallest_num_assignments_25_points : 
  (∑ i in range 25, (i + 1) / 5).ceil = 75 := by
  sorry

end smallest_num_assignments_25_points_l590_590640


namespace closest_integer_to_cube_root_of_150_l590_590604

theorem closest_integer_to_cube_root_of_150 : 
  let cbrt := (150: ℝ)^(1/3) in
  abs (cbrt - 6) < abs (cbrt - 5) :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590604


namespace saree_sale_price_l590_590576

-- Definitions based on conditions from part a)
def original_price : ℝ := 510
def first_discount : ℝ := 0.12
def second_discount : ℝ := 0.15
def third_discount : ℝ := 0.20
def tax_rate : ℝ := 0.05

-- The statement we need to prove
theorem saree_sale_price :
  let price_after_first_discount := original_price - (first_discount * original_price),
      price_after_second_discount := price_after_first_discount - (second_discount * price_after_first_discount),
      price_after_third_discount := price_after_second_discount - (third_discount * price_after_second_discount),
      final_price := price_after_third_discount + (tax_rate * price_after_third_discount)
  in final_price = 320.44 :=
by
  sorry

end saree_sale_price_l590_590576


namespace miniature_tower_height_l590_590703

theorem miniature_tower_height
  (actual_height : ℝ)
  (actual_volume : ℝ)
  (miniature_volume : ℝ)
  (actual_height_eq : actual_height = 60)
  (actual_volume_eq : actual_volume = 200000)
  (miniature_volume_eq : miniature_volume = 0.2) :
  ∃ (miniature_height : ℝ), miniature_height = 0.6 :=
by
  sorry

end miniature_tower_height_l590_590703


namespace roberto_outfits_l590_590916

theorem roberto_outfits (n_trousers : ℕ) (n_shirts : ℕ) (n_jackets : ℕ) : 
  n_trousers = 4 → n_shirts = 7 → n_jackets = 3 → n_trousers * n_shirts * n_jackets = 84 :=
by
  intros ht hs hj
  rw [ht, hs, hj]
  norm_num
  sorry

end roberto_outfits_l590_590916


namespace profit_percent_is_50_l590_590995

variable {P C : ℝ} -- P is the selling price, C is the cost price

-- Define the condition given in the problem
def loss_condition (P C : ℝ) : Prop := 2 / 3 * P = 0.82 * C

-- Define the profit percent when selling at price P
def profit_percent (P C : ℝ) : ℝ := (P - C) / C * 100

theorem profit_percent_is_50 (P C : ℝ) (h : loss_condition P C) : profit_percent P C = 50 :=
by
  sorry

end profit_percent_is_50_l590_590995


namespace value_expression_possible_values_l590_590119

open Real

noncomputable def value_expression (a b : ℝ) : ℝ :=
  a^2 + 2 * a * b + b^2 + 2 * a^2 * b + 2 * a * b^2 + a^2 * b^2

theorem value_expression_possible_values (a b : ℝ)
  (h1 : (a / b) + (b / a) = 5 / 2)
  (h2 : a - b = 3 / 2) :
  value_expression a b = 0 ∨ value_expression a b = 81 :=
sorry

end value_expression_possible_values_l590_590119


namespace michael_large_painting_price_l590_590889

theorem michael_large_painting_price (L : ℕ) 
  (small_painting_price : ℕ)
  (num_large_paintings : ℕ)
  (num_small_paintings : ℕ)
  (total_earnings : ℕ) :
  small_painting_price = 80 →
  num_large_paintings = 5 →
  num_small_paintings = 8 →
  total_earnings = 1140 →
  (num_large_paintings * L + num_small_paintings * small_painting_price = total_earnings) →
  L = 100 :=
by
  intros h1 h2 h3 h4 h5
  unfold small_painting_price at h1
  unfold num_large_paintings at h2
  unfold num_small_paintings at h3
  unfold total_earnings at h4
  rw [h1, h2, h3, h4] at h5
  sorry

end michael_large_painting_price_l590_590889


namespace jellybean_count_l590_590253

def black_beans : Nat := 8
def green_beans : Nat := black_beans + 2
def orange_beans : Nat := green_beans - 1
def total_jelly_beans : Nat := black_beans + green_beans + orange_beans

theorem jellybean_count : total_jelly_beans = 27 :=
by
  -- proof steps would go here.
  sorry

end jellybean_count_l590_590253


namespace regular_polygon_sides_l590_590744

theorem regular_polygon_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle i = 160)) : n = 18 :=
by
  -- Proof goes here
  sorry

end regular_polygon_sides_l590_590744


namespace cube_root_of_neg_27_over_8_l590_590158

theorem cube_root_of_neg_27_over_8 :
  (- (3 : ℝ) / 2) ^ 3 = - (27 / 8 : ℝ) := 
by
  sorry

end cube_root_of_neg_27_over_8_l590_590158


namespace sum_of_squares_divisible_l590_590464

theorem sum_of_squares_divisible (n : ℕ) (a : Fin n → ℕ) (h1 : ∀ i j, i < j → a i < a j)
  (h2 : ∀ i j, i ≠ j → Nat.coprime (a i) (a j)) (h3 : Nat.Prime (a 0))
  (h4 : a 0 ≥ n + 2) :
  let I := Finset.Icc 0 (Finset.prod Finset.univ a)
  let segments := Finset.filter (λ x, ∃ i, a i ∣ x ) I
  let square_lengths := (segments.map (λ s, s * s)).sum
  a 0 ∣ square_lengths :=
sorry

end sum_of_squares_divisible_l590_590464


namespace max_distance_diff_l590_590790

-- Definitions of the circles and moving points
def C1 : set (ℝ × ℝ) := { p | (p.1 + 2)^2 + (p.2 - 3)^2 = 1 }
def C2 : set (ℝ × ℝ) := { p | (p.1 - 3)^2 + (p.2 - 4)^2 = 9 }

-- Points A and B are moving points on C1 and C2, respectively
def A (p : ℝ × ℝ) : Prop := p ∈ C1
def B (p : ℝ × ℝ) : Prop := p ∈ C2

-- Point P is a moving point on the y-axis
def P (y : ℝ) : ℝ × ℝ := (0, y)

-- Function to calculate distance between two points
noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Main statement
theorem max_distance_diff : ∃ (y : ℝ), ∀ (a b : ℝ × ℝ), A a → B b → |dist (P y) b - dist (P y) a| ≤ √2 + 4 :=
sorry

end max_distance_diff_l590_590790


namespace percent_pear_juice_in_blend_l590_590124

def pearJuice (pears: ℕ) : ℝ := (10: ℝ) / 4 * (pears : ℝ)
def orangeJuice (oranges: ℕ) : ℝ := (12: ℝ) / 3 * (oranges : ℝ)

def totalJuice (pears: ℕ) (oranges: ℕ) : ℝ := pearJuice pears + orangeJuice oranges

def percentPearJuice (pears: ℕ) (oranges: ℕ) : ℝ := (pearJuice pears / totalJuice pears oranges) * 100

theorem percent_pear_juice_in_blend:
  percentPearJuice 8 6 = 45 := 
by
  unfold percentPearJuice
  unfold pearJuice
  unfold orangeJuice
  unfold totalJuice
  sorry

end percent_pear_juice_in_blend_l590_590124


namespace museum_opened_on_monday_l590_590152

theorem museum_opened_on_monday {year_opened : ℕ} :
  (120 : ℕ) + year_opened = 2023 →
  (2023 % 7 = 4) →
  ((∀ n : ℕ, (year_opened + n * 4) % 100 ≠ 0 ∨ (year_opened + n * 4) % 400 = 0) →
   year_opened % 4 = 3) →
  (10 + 31 + 30 + 31 + 30 + 31 + 31 + 10 + (120 + 30 * 2)) % 7 = 1 :=
begin
  sorry
end

end museum_opened_on_monday_l590_590152


namespace exists_kitten_with_black_neighbors_l590_590512

theorem exists_kitten_with_black_neighbors (kittens : Fin 14 → bool) (count_black : (Fin 14 → bool) → Nat) 
  (h_kittens_black : count_black kittens = 7)
  (h_kittens_ginger : count_black (λ x => !kittens x) = 7) : 
  ∃ i : Fin 14, kittens i = true ∧ kittens (i + 1) % 14 = true ∧ kittens (i + 2 % 14) = false := 
sorry

end exists_kitten_with_black_neighbors_l590_590512


namespace sqrt_sum_leq_sqrt_F_l590_590077

variables (a b c d F3 : ℝ)

theorem sqrt_sum_leq_sqrt_F 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h5 : 0 ≤ F3) :
  (sqrt ((a / (a + c)) * (b / (b + d)) * F3) + sqrt ((c / (a + c)) * (d / (b + d)) * F3) ≤ sqrt F3) :=
sorry

end sqrt_sum_leq_sqrt_F_l590_590077


namespace sum_of_absolute_slopes_l590_590557

theorem sum_of_absolute_slopes (P Q R S : ℤ × ℤ)
  (h1 : P = (30, 200))
  (h2 : S = (31, 215))
  (h3 : ∀ (Q R : ℤ × ℤ), Q ≠ R → 
        (Q.1 - P.1) * (R.2 - P.2) ≠ (Q.2 - P.2) * (R.1 - P.1))
  (h4 : ∀ (Q R : ℤ × ℤ), Q.1 ≠ S.1 → 
        (Q.1 - P.1) * (S.2 - R.2) ≠ (Q.2 - P.2) * (S.1 - R.1)) :
  let slopes := {7, 1 / 2, -7, -1 / 2, 1, -1, 7 / 2, -7 / 2} in
  ∑ s in slopes, abs s = 255 / 2 ∧ 255 / 2 = (255 : ℚ) / (2 : ℚ) :=
sorry

end sum_of_absolute_slopes_l590_590557


namespace combined_cost_of_apples_and_strawberries_l590_590186

theorem combined_cost_of_apples_and_strawberries :
  let cost_of_apples := 15
  let cost_of_strawberries := 26
  cost_of_apples + cost_of_strawberries = 41 :=
by
  sorry

end combined_cost_of_apples_and_strawberries_l590_590186


namespace inequality_solution_l590_590306

noncomputable def solve_inequality : set ℝ :=
  {x : ℝ | 7.05 ≤ x ∧ x < 9} ∪ {x : ℝ | 9 < x ∧ x ≤ 12.30}

theorem inequality_solution (x : ℝ) : 
  (x ≠ 9 ∧ (x * (x + 1)) / (x - 9)^2 ≥ 15) ↔ x ∈ solve_inequality :=
begin
  sorry
end

end inequality_solution_l590_590306


namespace number_of_arrangements_l590_590246

theorem number_of_arrangements:
  let students := {A, B, C, D, E, F},
      communityA := {studentA},
      communityB := {studentB, studentC},
      communityC := students \ (communityA ∪ communityB) 
  in students.card = 6 
  ∧ ∀ s, s ∈ communityA → s = A
  ∧ ∀ s, s ∈ communityB → s ≠ C
  ∧ ∀ s, s ∈ communityC → s ≠ B
  → (finset.pairs (communityA ∪ communityB ∪ communityC)).card = 9
:= by {
  sorry
}

end number_of_arrangements_l590_590246


namespace number_of_partitions_S_l590_590888

namespace PartitionProblem

def S : set ℕ := {1, 2}

def is_partition (A B : set ℕ) : Prop :=
  A ∪ B = S ∧ (A = B ∨ A = S \ B)

def num_partitions : ℕ := 9

theorem number_of_partitions_S : (∃n : ℕ, n = num_partitions ∧ n = 9) :=
by
  existsi 9
  split
  . rfl
  . rfl

end PartitionProblem

end number_of_partitions_S_l590_590888


namespace total_students_in_halls_l590_590951

open Nat

theorem total_students_in_halls : 
    let students_general : ℕ := 30
    let students_biology : ℕ := 2 * students_general
    let combined_general_biology : ℕ := students_general + students_biology
    let students_math : ℕ := (3 * combined_general_biology) / 5
    students_general + students_biology + students_math = 144 :=
by
    let students_general := 30
    let students_biology := 2 * students_general
    let combined_general_biology := students_general + students_biology
    let students_math := (3 * combined_general_biology) / 5
    have h_general : students_general = 30 := rfl
    have h_biology : students_biology = 60 := rfl
    have h_combined : combined_general_biology = 90 := rfl
    have h_math : students_math = 54 := rfl
    show students_general + students_biology + students_math = 144
    calc
    students_general + students_biology + students_math
        = 30 + 60 + 54 : by rw [h_general, h_biology, h_math]
    ... = 144 : rfl

end total_students_in_halls_l590_590951


namespace extreme_point_l590_590935

noncomputable def f (x : ℝ) : ℝ := (x^4 / 4) - (x^3 / 3)
noncomputable def f_prime (x : ℝ) : ℝ := deriv f x

theorem extreme_point (x : ℝ) : f_prime 1 = 0 ∧
  (∀ y, y < 1 → f_prime y < 0) ∧
  (∀ z, z > 1 → f_prime z > 0) :=
by
  sorry

end extreme_point_l590_590935


namespace seating_arrangements_l590_590893

-- Definitions for conditions
def num_parents : ℕ := 2
def num_children : ℕ := 3
def num_front_seats : ℕ := 2
def num_back_seats : ℕ := 3
def num_family_members : ℕ := num_parents + num_children

-- The statement we need to prove
theorem seating_arrangements : 
  (num_parents * -- choices for driver
  (num_family_members - 1) * -- choices for the front passenger
  (num_back_seats.factorial)) = 48 := -- arrangements for the back seats
by
  sorry

end seating_arrangements_l590_590893


namespace cube_root_fraction_l590_590324

theorem cube_root_fraction : 
  (∃ x : ℝ, x = real.cbrt (6 / 18) ∧ x = 1 / real.cbrt 3) :=
sorry

end cube_root_fraction_l590_590324


namespace complex_magnitude_difference_l590_590355

theorem complex_magnitude_difference (z1 z2 : ℂ) (h1 : complex.abs z1 = 1) (h2 : complex.abs z2 = 1) (h3 : complex.abs (z1 + z2) = √3) : 
  complex.abs (z1 - z2) = 1 := 
by 
  sorry

end complex_magnitude_difference_l590_590355


namespace hyperbola_asymptotes_l590_590389

noncomputable def circle_center_y (m : ℝ) : ℝ :=
- m + 4

noncomputable def circle_radius : ℝ := 1

def dist_circle_origin (m : ℝ) : ℝ :=
Real.sqrt (9 + (4 - m)^2)

def min_circle_origin_dist : ℝ := 3

def shortest_dist_from_origin_to_circle_point : ℝ := min_circle_origin_dist - circle_radius

def hyperbola_eccentricity : ℝ := 2

def relationship_between_params (a : ℝ) (b : ℝ) : Prop :=
b = Real.sqrt (3) * a

def hyperbola_asymptotes_eq (a : ℝ) (b : ℝ) : Prop :=
∀ x, (∀ y, y = (b / a) * x ∨ y = - (b / a) * x)

theorem hyperbola_asymptotes (a b : ℝ) (h1 : b = Real.sqrt 3 * a) :
  hyperbola_asymptotes_eq a b :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l590_590389


namespace find_a_and_a100_l590_590451

def seq (a : ℝ) (n : ℕ) : ℝ := (-1)^n * n + a

theorem find_a_and_a100 :
  ∃ a : ℝ, (seq a 1 + seq a 4 = 3 * seq a 2) ∧ (seq a 100 = 97) :=
by
  sorry

end find_a_and_a100_l590_590451


namespace regular_polygon_sides_l590_590742

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l590_590742


namespace problem_solution_l590_590904

theorem problem_solution (s t : ℕ) (hpos_s : 0 < s) (hpos_t : 0 < t) (h_eq : s * (s - t) = 29) : s + t = 57 :=
by
  sorry

end problem_solution_l590_590904


namespace A_subset_B_l590_590488

def is_in_A (x : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ x = a^2 + 1

def is_in_B (y : ℕ) : Prop :=
  ∃ b : ℕ, b > 0 ∧ y = b^2 - 4 * b + 5

theorem A_subset_B : ∀ x, is_in_A x → is_in_B x :=
by
  intros x h
  cases h with a ha
  cases ha with ha_pos ha_eq
  use [a+2, lt_add_of_pos_right a ha_pos]
  rw [ha_eq]
  ring
  sorry

end A_subset_B_l590_590488


namespace inverse_function_value_l590_590001

def f (x : ℝ) : ℝ := 25 / (4 + 5 * x)

theorem inverse_function_value :
  (f⁻¹ 5)⁻¹ = 5 := 
sorry

end inverse_function_value_l590_590001


namespace max_correct_percentage_is_92_l590_590706

noncomputable def max_overall_correct_percent (t : ℕ) : ℝ :=
  let y := 0.3533 * ↑t in
  let chloe_correct_alone := 0.7 * (2 / 3) * ↑t in
  let chloe_correct_total := 0.82 * ↑t in
  let max_correct_alone := 0.85 * (2 / 3) * ↑t in
  let max_correct_total := max_correct_alone + y in
  (max_correct_total / ↑t) * 100

theorem max_correct_percentage_is_92 (t : ℕ) : max_overall_correct_percent t = 92 := by
  sorry

end max_correct_percentage_is_92_l590_590706


namespace inequality_greater_sqrt_two_l590_590136

theorem inequality_greater_sqrt_two (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) : 
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 := 
by 
  sorry

end inequality_greater_sqrt_two_l590_590136


namespace regular_polygon_sides_l590_590732

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ α, α = 160 → ∑ x in range n, (180 - (180 - 160)) = 360) : n = 18 :=
by
  sorry

end regular_polygon_sides_l590_590732


namespace intersection_distances_l590_590851

noncomputable def point := (ℝ × ℝ)
noncomputable def line (P : point) (θ : ℝ) : point → Prop :=
  λ Q, ∃ t : ℝ, (Q.1 = t) ∧ (Q.2 = t - 3)

noncomputable def curve (ρ θ : ℝ) : Prop :=
  ρ * (sin θ)^2 = 2 * cos θ

noncomputable def rectangular_curve (p : point) : Prop :=
  p.2^2 = 2 * p.1

noncomputable def distance (P Q : point) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem intersection_distances (A B P : point) (l : point → Prop) (C : point → Prop)
    (hA : l A) (hB : l B) (hC_A : C A) (hC_B : C B) (hP : P = (1, -2)) :
  distance P A * distance P B = 4 :=
  sorry

end intersection_distances_l590_590851


namespace find_ab_l590_590800

noncomputable def complex_z : ℂ := (1 - complex.I)^2 + 1 + 3 * complex.I

theorem find_ab (a b : ℝ) (hz : complex_z = 1 + complex.I) : 
  (complex_z^2 + a * complex_z + b = 1 - complex.I) ↔ (a = -3 ∧ b = 4) := 
by 
  sorry

end find_ab_l590_590800


namespace sequence_value_at_99_l590_590852

theorem sequence_value_at_99 :
  ∃ a : ℕ → ℚ, (a 1 = 2) ∧ (∀ n : ℕ, a (n + 1) = a n + n / 2) ∧ (a 99 = 2427.5) :=
by
  sorry

end sequence_value_at_99_l590_590852


namespace milk_after_three_operations_l590_590991

-- Define the initial amount of milk and the proportion replaced each step
def initial_milk : ℝ := 100
def proportion_replaced : ℝ := 0.2

-- Define the amount of milk after each replacement operation
noncomputable def milk_after_n_operations (n : ℕ) (milk : ℝ) : ℝ :=
  if n = 0 then milk
  else (1 - proportion_replaced) * milk_after_n_operations (n - 1) milk

-- Define the statement about the amount of milk after three operations
theorem milk_after_three_operations : milk_after_n_operations 3 initial_milk = 51.2 :=
by
  sorry

end milk_after_three_operations_l590_590991


namespace limit_sqrt_cubert_l590_590697

open Real

theorem limit_sqrt_cubert (f : ℕ → ℝ) (h : ∀ n, f n = (sqrt (n^2 + 3 * n - 1) + (2 * n^2 + 1)^(1 / 3)) / (n + 2 * sin n)) :
  Tendsto f atTop (𝓝 1) :=
begin
  sorry
end

end limit_sqrt_cubert_l590_590697


namespace cost_price_is_60_l590_590267

variable (C S: ℝ)

-- Condition 1: original selling price S is 1.25 times the cost price C
def original_selling_price (h1 : S = 1.25 * C) : Prop := True

-- Condition 2: new selling price when bought at 20% less and sold for Rs. 12.60 less
def new_selling_price (S_new : ℝ) (h2 : S_new = S - 12.60) : Prop := True

-- Condition 3: new selling price is 1.04 times the cost price C after 30% profit on the new cost
def new_selling_price_after_profit (S_new : ℝ) (h3 : S_new = 1.04 * C) : Prop := True

-- Goal: proving the cost price C is 60
theorem cost_price_is_60 (S_new : ℝ) :
  original_selling_price C S (by rfl) →
  new_selling_price C S S_new (by rfl) →
  new_selling_price_after_profit C S_new (by rfl) →
  C = 60 := 
  sorry

end cost_price_is_60_l590_590267


namespace sqrt_eq_sum_iff_l590_590358

open Real

theorem sqrt_eq_sum_iff (a b : ℝ) : sqrt (a^2 + b^2) = a + b ↔ (a * b = 0) ∧ (a + b ≥ 0) :=
by
  sorry

end sqrt_eq_sum_iff_l590_590358


namespace range_of_values_for_a_l590_590810

theorem range_of_values_for_a 
  (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x = x - 1/x - a * Real.log x)
  (h2 : ∀ x > 0, (x^2 - a * x + 1) ≥ 0) : 
  a ≤ 2 :=
sorry

end range_of_values_for_a_l590_590810


namespace quadratic_eq_equal_real_roots_l590_590836

theorem quadratic_eq_equal_real_roots (m : ℝ) :
  (∃ (x : ℝ), x = 4 ∧ (b^2 - 4*a*c) = 0) :=
begin
    sorry
end

end quadratic_eq_equal_real_roots_l590_590836


namespace rectangle_area_ratio_l590_590271

theorem rectangle_area_ratio (x d : ℝ) (h_ratio : 5 * x / (2 * x) = 5 / 2) (h_diag : d = 13) :
  ∃ k : ℝ, 10 * x^2 = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_ratio_l590_590271


namespace conversation_erased_probability_l590_590567

noncomputable def tape_length : ℝ := 30  -- 30 minutes
noncomputable def conversation_start : ℝ := 30 / 60  -- 30 seconds in minutes
noncomputable def conversation_end : ℝ := 40 / 60  -- 40 seconds in minutes
noncomputable def interval_length : ℝ := conversation_end 
noncomputable def probability_erased : ℝ := interval_length / tape_length

theorem conversation_erased_probability
  (tape_length = 30)  -- in minutes
  (conversation_start = 30 / 60)  -- in minutes
  (conversation_end = 40 / 60)  -- in minutes
  (interval_length = 40 / 60)  -- in minutes
  : probability_erased = 1 / 45
:=
by
  sorry

end conversation_erased_probability_l590_590567


namespace sum_first_11_terms_eq_11_l590_590445

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_11_terms_eq_11 (a : ℕ → ℝ) (h_arith_seq : arithmetic_seq a)
    (h_a5_a7 : ∀ {x : ℝ}, x^2 - 2*x - 6 = 0 → x = a 5 ∨ x = a 7) :
  (∑ i in finset.range 11, a i) = 11 := 
by
  sorry

end sum_first_11_terms_eq_11_l590_590445


namespace problem_statement_l590_590416

theorem problem_statement (x1 x2 x3 x4 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 < 1) (h4 : 1 < x3) (h5 : x3 < x4) :
    x3 * real.exp x4 > x4 * real.exp x3 := 
sorry

end problem_statement_l590_590416


namespace mode_of_data_set_l590_590028

theorem mode_of_data_set : ∀ (data : List ℕ),
  data = [85, 63, 101, 85, 85, 101, 72] →
  (List.mode data) = 85 :=
begin
  intros data h,
  rw h,
  sorry
end

end mode_of_data_set_l590_590028


namespace total_salary_of_both_l590_590965

def n_salary := 270    -- n's salary
def m_salary := 1.20 * n_salary -- m's salary

theorem total_salary_of_both (n_salary : ℝ) (m_salary : ℝ) : (n_salary = 270) ∧ (m_salary = 1.20 * n_salary) → n_salary + m_salary = 594 :=
by
  sorry

end total_salary_of_both_l590_590965


namespace max_lambda_l590_590336

theorem max_lambda {
  λ : ℝ, 
  ∀ a b : ℝ, λ * a^2 * b^2 * (a + b)^2 ≤ (a^2 + a * b + b^2)^3
} : λ ≤ 27 / 4 :=
begin
  sorry
end

end max_lambda_l590_590336


namespace area_of_figure_l590_590237

theorem area_of_figure :
  (let x := 2 * Real.sqrt 2 * Real.cos t ^ 3,
       y := Real.sqrt 2 * Real.sin t ^ 3 in
    ∃ a b: Real, a ≤ b ∧
      (∀ t ∈ [a, b], x ≥ 1) ∧
      (2 * ∫ t in a..b, y * x' t = (3 / 8) * Real.pi - 1 / 2)) :=
begin
  sorry
end

end area_of_figure_l590_590237


namespace height_of_water_in_cylinder_l590_590289

-- Definitions
def radius_cone : ℝ := 15
def height_cone : ℝ := 20
def radius_cylinder : ℝ := 30

-- Statement of the theorem
theorem height_of_water_in_cylinder :
  let volume_cone := (1 / 3) * real.pi * radius_cone ^ 2 * height_cone,
      volume_cylinder := volume_cone,
      height_cylinder := volume_cylinder / (real.pi * radius_cylinder ^ 2)
  in height_cylinder = 1.67 :=
by
  sorry

end height_of_water_in_cylinder_l590_590289


namespace perpendicular_m_alpha_l590_590880

variable (m n : Line) (α β : Plane)

-- Assuming m is perpendicular to β
axiom perpendicular_m_beta : Perpendicular m β

-- Assuming n is perpendicular to β
axiom perpendicular_n_beta : Perpendicular n β

-- Assuming n is perpendicular to α
axiom perpendicular_n_alpha : Perpendicular n α

-- We need to prove m is perpendicular to α
theorem perpendicular_m_alpha
  (perpendicular_m_beta : Perpendicular m β)
  (perpendicular_n_beta : Perpendicular n β)
  (perpendicular_n_alpha : Perpendicular n α) :
  Perpendicular m α :=
sorry

end perpendicular_m_alpha_l590_590880


namespace geometric_sequence_S3_range_l590_590782

theorem geometric_sequence_S3_range (a : ℕ → ℝ) (q : ℝ) (hq : ∀ n, a (n + 1) = a n * q) (h2 : a 2 = 2) :
  (∃ S3, S3 = a 0 + a 1 + a 2 ∧ (S3 ∈ set.Iic (-2) ∨ S3 ∈ set.Ici 6)) := 
sorry

end geometric_sequence_S3_range_l590_590782


namespace number_of_red_balls_l590_590440

theorem number_of_red_balls (W R T : ℕ) (hW : W = 12) (h_freq : (R : ℝ) / (T : ℝ) = 0.25) (hT : T = W + R) : R = 4 :=
by
  sorry

end number_of_red_balls_l590_590440


namespace combined_mpg_is_30_l590_590911

-- Define the constants
def ray_efficiency : ℕ := 50 -- miles per gallon
def tom_efficiency : ℕ := 25 -- miles per gallon
def ray_distance : ℕ := 100 -- miles
def tom_distance : ℕ := 200 -- miles

-- Define the combined miles per gallon calculation and the proof statement.
theorem combined_mpg_is_30 :
  (ray_distance + tom_distance) /
  ((ray_distance / ray_efficiency) + (tom_distance / tom_efficiency)) = 30 :=
by
  -- All proof steps are skipped using sorry
  sorry

end combined_mpg_is_30_l590_590911


namespace smallest_positive_period_and_min_value_value_at_special_point_l590_590803

-- Definitions for the first proof problem
def f (x : ℝ) : ℝ := 1 + Math.sin x * Math.cos x

-- Conditions
theorem smallest_positive_period_and_min_value :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π) ∧ (∀ x, f x ≥ 1 / 2) ∧ (∃ x, f x = 1 / 2) :=
begin
  sorry
end

-- Definitions for the second proof problem
def g (x : ℝ) : ℝ := f (π/4 - x/2)

-- Conditions
theorem value_at_special_point (x : ℝ) (h1 : Math.tan x = 3 / 4) (h2 : x ∈ set.Ioo 0 (π / 2)) :
  g x = 7 / 5 :=
begin
  sorry
end

end smallest_positive_period_and_min_value_value_at_special_point_l590_590803


namespace find_c_for_square_of_binomial_l590_590000

theorem find_c_for_square_of_binomial (c : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 + 50 * x + c = (x + b)^2) → c = 625 :=
by
  intro h
  obtain ⟨b, h⟩ := h
  sorry

end find_c_for_square_of_binomial_l590_590000


namespace shaded_area_correct_l590_590037

noncomputable def total_shaded_area (r₁ r₂ : ℝ) : ℝ :=
  let area_small_shaded := (3 * 6) - (1 / 2) * Real.pi * r₁^2
  let area_large_shaded := (6 * 12) - (1 / 2) * Real.pi * r₂^2
  area_small_shaded + area_large_shaded

theorem shaded_area_correct :
  total_shaded_area 3 6 ≈ 19.3 :=
by
  sorry

end shaded_area_correct_l590_590037


namespace ants_no_collision_probability_l590_590209

noncomputable def probability_no_collision : ℚ :=
  let vertices := {A, B, C, D, E, F, G, H}
  let adj := (λ v, ({v' | v' ≠ v  
                          ∧ (v = A ∧ v' ⊆ {B, D, E}) 
                          ∨ (v = B ∧ v' ⊆ {A, C, F})
                          ∨ (v = C ∧ v' ⊆ {B, D, G})
                          ∨ (v = D ∧ v' ⊆ {A, C, H})
                          ∨ (v = E ∧ v' ⊆ {A, F, H})
                          ∨ (v = F ∧ v' ⊆ {B, E, G})
                          ∨ (v = G ∧ v' ⊆ {C, F, H})
                          ∨ (v = H ∧ v' ⊆ {D, E, G})}))
  let movements := {seq | ∀ v ∈ vertices, seq v ∈ adj v}
  let valid_movement := {p ∈ movements | ∀ v, seq v ≠ seq u for every u, v, u ≠ v}
  (finset.card valid_movement) / (3^8 : ℤ)

theorem ants_no_collision_probability :
  probability_no_collision = 1 / 6561 :=
sorry

end ants_no_collision_probability_l590_590209


namespace train_travel_distance_correct_l590_590659

noncomputable def train_distance (departure_time : DateTime) (arrival_time : DateTime) : ℕ :=
  let travel_hours := (arrival_time - departure_time).toDuration.toHours
  let segment_distance := 48
  let speed := 60
  let regular_stop_duration := 10 / 60 -- in hours
  let fifth_stop_duration := 30 / 60 -- in hours
  let total_stops := (travel_hours / (4 / 5 + regular_stop_duration)).floor + 1
  let total_fifth_stop_index := total_stops / 5
  let total_regular_stops := total_stops - total_fifth_stop_index
  let total_stop_time := total_regular_stops * regular_stop_duration + total_fifth_stop_index * fifth_stop_duration
  let total_travel_time := travel_hours - total_stop_time
  let distance := total_travel_time * speed
  distance

theorem train_travel_distance_correct : 
  train_distance (DateTime.mk 2023 9 29 12 0 0) (DateTime.mk 2023 10 1 22 0 0) = 2870 := 
sorry

end train_travel_distance_correct_l590_590659


namespace install_time_for_windows_l590_590641

theorem install_time_for_windows
  (total_windows installed_windows hours_per_window : ℕ)
  (h1 : total_windows = 200)
  (h2 : installed_windows = 65)
  (h3 : hours_per_window = 12) :
  (total_windows - installed_windows) * hours_per_window = 1620 :=
by
  sorry

end install_time_for_windows_l590_590641


namespace remaining_volume_fraction_l590_590717

noncomputable def fraction_of_remaining_volume (T : Tetrahedron) : ℝ :=
  let λ := (3 - 4 * real.cbrt (1 / 3)) in
  λ^3

theorem remaining_volume_fraction (T : Tetrahedron) (h_volume_cut : ∀ P ∈ {planes_parallel_to_faces T},
  let T' := cut_off_tetrahedron T P in
  volume T' = (1 / 3) * volume T) :
  fraction_of_remaining_volume T = 0.0118 :=
sorry

end remaining_volume_fraction_l590_590717


namespace rectangle_area_l590_590759

-- Define the given dimensions
def length : ℝ := 1.5
def width : ℝ := 0.75
def expected_area : ℝ := 1.125

-- State the problem
theorem rectangle_area (l w : ℝ) (h_l : l = length) (h_w : w = width) : l * w = expected_area :=
by sorry

end rectangle_area_l590_590759


namespace remainder_of_product_mod_9_l590_590698

theorem remainder_of_product_mod_9 : 
  (∏ i in finset.range 20, (7 + 10 * i)) % 9 = 4 := 
by 
  sorry

end remainder_of_product_mod_9_l590_590698


namespace right_triangle_BD_length_l590_590847

theorem right_triangle_BD_length (BC AC AD BD : ℝ ) (h_bc: BC = 1) (h_ac: AC = b) (h_ad: AD = 2) :
  BD = Real.sqrt (b^2 - 3) :=
by
  sorry

end right_triangle_BD_length_l590_590847


namespace length_of_chord_l590_590197

theorem length_of_chord 
  (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_tangent : ∀ x y : ℝ, (x = a ∧ y = b) ∨ (x = a ∧ y = c) ∨ (x = b ∧ y = c)) :
  (∃ l : ℝ, l = (4 * c * real.sqrt (a * b)) / (a + b)) :=
sorry

end length_of_chord_l590_590197


namespace jim_out_of_pocket_l590_590069

/-
Jim buys a wedding ring for $10,000. He gets his wife a ring that is twice that much and 
sells the first one for half its value. Prove that he is out of pocket $25,000.
-/

def first_ring_cost : ℕ := 10000
def second_ring_cost : ℕ := 2 * first_ring_cost
def selling_price_first_ring : ℕ := first_ring_cost / 2
def out_of_pocket : ℕ := (first_ring_cost - selling_price_first_ring) + second_ring_cost

theorem jim_out_of_pocket : out_of_pocket = 25000 := 
by
  -- type hint for clarity
  unfolding first_ring_cost second_ring_cost selling_price_first_ring out_of_pocket
  -- manually calculate and verify
  change ((10000 - 5000) + 20000) = 25000
  simp -- simplify the expression
  exact rfl -- reflexivity of equality

end jim_out_of_pocket_l590_590069


namespace petya_time_comparison_l590_590684

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l590_590684


namespace binomial_theorem_substitution_l590_590834

theorem binomial_theorem_substitution (a : Fin 2015 → ℝ) :
  (∀ x : ℝ, (1 - 2 * x)^2014 = ∑ i in Finset.range 2015, a i * x^i) →
  (∑ i in Finset.range 2015, a i / 2^i) = 0 := by
  sorry

end binomial_theorem_substitution_l590_590834


namespace eq_of_divides_l590_590628

theorem eq_of_divides (a b : ℕ) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b :=
sorry

end eq_of_divides_l590_590628


namespace general_formula_a_sum_T_l590_590040

noncomputable def a (n : ℕ) : ℕ := if n = 1 then 0 else n - 1
noncomputable def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, (a (i + 1) + 1 : ℕ) / (2 ^ (i + 1) : ℝ)

theorem general_formula_a (n : ℕ) (h1 : n ≥ 1) (h2 : a 2 = 1) (h3 : 2 * S n = n * a n) :
  a n = n - 1 := by
  sorry

theorem sum_T (n : ℕ) (h1 : ∀ k, a k = k - 1) :
  T n = 2 - (n + 2) / (2 ^ n) := by
  sorry

end general_formula_a_sum_T_l590_590040


namespace probability_of_different_tens_digits_l590_590533

open Finset

-- Define the basic setup
def integers (n : ℕ) : Finset ℕ := {i in (range n) | i ≥ 10 ∧ i ≤ 79}

def tens_digit (n : ℕ) : ℕ := n / 10

def six_integers_with_different_tens_digits (s : Finset ℕ) : Prop :=
  s.card = 6 ∧ (s.map ⟨tens_digit, by simp⟩).card = 6

def favorable_ways : ℕ :=
  7 * 10^6

def total_ways : ℕ :=
  nat.choose 70 6

noncomputable def probability : ℚ :=
  favorable_ways / total_ways

-- The main statement
theorem probability_of_different_tens_digits :
  ∀ (s : Finset ℕ), six_integers_with_different_tens_digits s → 
  probability = 175 / 2980131 :=
begin
  intros s h,
  sorry
end

end probability_of_different_tens_digits_l590_590533


namespace jellybean_total_l590_590251

theorem jellybean_total 
    (blackBeans : ℕ)
    (greenBeans : ℕ)
    (orangeBeans : ℕ)
    (h1 : blackBeans = 8)
    (h2 : greenBeans = blackBeans + 2)
    (h3 : orangeBeans = greenBeans - 1) :
    blackBeans + greenBeans + orangeBeans = 27 :=
by
    -- The proof will be placed here
    sorry

end jellybean_total_l590_590251


namespace sqrt_sum_inequality_l590_590485

theorem sqrt_sum_inequality (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h_sum : a + b + c = 3) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + c * a) :=
by
  sorry

end sqrt_sum_inequality_l590_590485


namespace regular_polygon_sides_l590_590747

theorem regular_polygon_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle i = 160)) : n = 18 :=
by
  -- Proof goes here
  sorry

end regular_polygon_sides_l590_590747


namespace general_formula_sum_Tn_l590_590054

-- Conditions
def a_seq (n : ℕ) : ℕ := if n = 1 then 0 else n - 1
def S_n (n : ℕ) : ℕ := (finset.range n).sum (λ i, a_seq (i + 1))

-- Given conditions
axiom a2 : a_seq 2 = 1
axiom S_n_cond (n : ℕ) : 2 * S_n n = n * a_seq n

-- (1) General formula for {a_n}
theorem general_formula (n : ℕ) : a_seq n = n - 1 :=
by sorry

-- (2) Sum of the first n terms of the sequence {(\frac{a_n + 1}{2^n})}
def T_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, (a_seq (k + 1) + 1 : ℝ) / (2 : ℝ) ^ (k + 1))

theorem sum_Tn (n : ℕ) : T_n n = 2 - (n + 2) / (2 : ℝ) ^ n :=
by sorry

end general_formula_sum_Tn_l590_590054


namespace sine_of_dihedral_angle_l590_590771

open Real

theorem sine_of_dihedral_angle (P A B C : ℝ)
  (PA PB PC AC : ℝ)
  (h_ratio : PA / PB = 1 / 2 ∧ PB / PC = 2 / 3)
  (h_perpendiculars: PA ⬝ PB = 0 ∧ PB ⬝ PC = 0 ∧ PA ⬝ PC = 0)
  (h_AC : AC = sqrt (PA^2 + PC^2)) :
  sin (atan 1) = PA / AC :=
by
  sorry

end sine_of_dihedral_angle_l590_590771


namespace problem_statement_l590_590795

noncomputable def C := { P : ℝ × ℝ | let (x, y) := P in y^2 = 8 * x }

structure Point (α : Type) :=
(x : α)
(y : α)

def distance (P Q : Point ℝ) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

theorem problem_statement :
  (∀ P : Point ℝ, P ∈ C →
    distance P ⟨-4, P.y⟩ = 2 + distance P ⟨2, 0⟩) →
  (∀ A : Point ℝ, A ∈ C →
    distance A (⟨2, 0⟩) = 4 →
    (A.y = 4 ∨ A.y = -4)) →
  (∀ M N : Point ℝ, M ∈ C ∧ N ∈ C →
    (distance M ⟨2, 0⟩ + distance N ⟨2, 0⟩) = 10 →
    ((M.x + N.x) / 2 = 3)) →
  (∃ P : Point ℝ, P ∈ C →
    let A := Point.mk 3 2 in
    (distance P A + distance P ⟨2, 0⟩) = 5) :=
sorry


end problem_statement_l590_590795


namespace regular_polygon_sides_l590_590753

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l590_590753


namespace greatest_prime_factor_f_22_l590_590768

noncomputable def f (m : ℕ) : ℕ :=
  if h : m % 2 = 0 then (finset.range m // 2).filter (λ x, x % 2 = 0) 1.prod else 1

theorem greatest_prime_factor_f_22 (m : ℕ) (h_even : m % 2 = 0) (h_pos : 0 < m)
  (h_11 : ∀ p : ℕ, prime p → p ∣ f m → p ≤ 11) : m = 22 := sorry

end greatest_prime_factor_f_22_l590_590768


namespace problem_1_l590_590242

theorem problem_1
  (α : ℝ)
  (h : Real.tan α = -1/2) :
  1 / (Real.sin α ^ 2 - Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) = -1 := 
sorry

end problem_1_l590_590242


namespace proof_angles_proof_area_l590_590429

noncomputable def angles_in_ABC (A B C: ℝ) (a b c: ℝ) :=
  A = π / 4 ∧
  b * sin (π / 4 + C) - c * sin (π / 4 + B) = a ∧
  B = 5 * π / 8 ∧
  C = π / 8

noncomputable def area_of_ABC (A B C: ℝ) (a b c: ℝ) :=
  a = 2 * sqrt 2 ∧
  angles_in_ABC A B C a b c →
  let area := 1 / 2 * a * b * sin C
  in area = 2

theorem proof_angles (A B C: ℝ) (a b c: ℝ) :
  angles_in_ABC A B C a b c := 
sorry

theorem proof_area (A B C: ℝ) (a b c: ℝ) :
  area_of_ABC A B C a b c :=
sorry

end proof_angles_proof_area_l590_590429


namespace cube_root_1728_simplified_l590_590219

theorem cube_root_1728_simplified :
  let a := 12
  let b := 1
  a + b = 13 :=
by
  sorry

end cube_root_1728_simplified_l590_590219


namespace expected_lifetime_at_least_four_l590_590104

universe u

variables (α : Type u) [MeasurableSpace α] {𝒫 : ProbabilitySpace α}
variables {ξ η : α → ℝ} [IsFiniteExpectation ξ] [IsFiniteExpectation η]

noncomputable def max_lifetime : α → ℝ := λ ω, max (ξ ω) (η ω)

theorem expected_lifetime_at_least_four 
  (h : ∀ ω, max (ξ ω) (η ω) ≥ η ω)
  (h_eta : @Expectation α _ _ η  = 4) : 
  @Expectation α _ _ max_lifetime ≥ 4 :=
by
  sorry

end expected_lifetime_at_least_four_l590_590104


namespace sqrt_eq_two_or_neg_two_l590_590183

theorem sqrt_eq_two_or_neg_two (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_eq_two_or_neg_two_l590_590183


namespace trapezoid_angle_equality_l590_590453

theorem trapezoid_angle_equality
  (A B C D K M : Type)
  (h_trapezoid : Trapezoid A B C D)
  (hK_on_AB : PointOnSide K A B)
  (hM_on_CD : PointOnSide M C D)
  (h_angle_equality : ∠BAM = ∠CDK) :
  ∠BMA = ∠CKD :=
sorry

end trapezoid_angle_equality_l590_590453


namespace equal_focal_lengths_of_ellipses_l590_590558

variable {k : ℝ}

def curve1 (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

def curve2 (x y : ℝ) (k : ℝ) : Prop :=
  x^2 / (25 - k) + y^2 / (9 - k) = 1

theorem equal_focal_lengths_of_ellipses (h : k < 9) : 
  let f1 := 8 in -- focal length of the first ellipse
  let f2 := 8 in -- focal length of the second ellipse calculated in solution
  f1 = f2 := 
by
  sorry

end equal_focal_lengths_of_ellipses_l590_590558


namespace necessary_but_not_sufficient_l590_590367

variable {x : ℝ}

-- Define the conditions
def p : Prop := abs x > 1
def q : Prop := x < -2

-- Statement to prove that p is necessary but not sufficient for q
theorem necessary_but_not_sufficient : (q → p) ∧ ¬(p → q) :=
by
  sorry

end necessary_but_not_sufficient_l590_590367


namespace range_of_x_l590_590573

noncomputable def function_domain (x : ℝ) : Prop :=
x + 2 > 0 ∧ x ≠ 1

theorem range_of_x {x : ℝ} (h : function_domain x) : x > -2 ∧ x ≠ 1 :=
by
  sorry

end range_of_x_l590_590573


namespace series_convergence_condition_l590_590766

def a_n (p : ℝ) (n : ℕ) : ℝ :=
  (1 / n^p) * ∫ x in 0..n, |Real.sin (Real.pi * x)|^x

noncomputable def series_converges (p : ℝ) : Prop :=
  ∃ l : ℝ, ∃ seq : ℕ → ℝ, (∀ N, seq N = ∑ i in finset.range N, a_n p i) ∧ filter.eventually (λ N, seq N = l) at_top

theorem series_convergence_condition (p : ℝ) : series_converges p ↔ p > 3 / 2 :=
by
  sorry

end series_convergence_condition_l590_590766


namespace find_large_number_l590_590233

namespace example

variables (L S : ℕ)

-- Conditions
axiom difference_eq : L - S = 1365
axiom quotient_remainder_eq : L = 6 * S + 10

-- Proof to find the larger number L
theorem find_large_number : L = 1636 := 
  sorry

end example

end find_large_number_l590_590233


namespace find_a_l590_590796

theorem find_a
  (a b c : ℝ) 
  (h1 : ∀ x : ℝ, x = 1 ∨ x = 2 → a * x * (x + 1) + b * x * (x + 2) + c * (x + 1) * (x + 2) = 0)
  (h2 : a + b + c = 2) : 
  a = 12 := 
sorry

end find_a_l590_590796


namespace volume_of_parallelepiped_l590_590150

open Real

variables (a b : ℝ^3)

-- a and b are unit vectors
def is_unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1

-- The angle between a and b is π/4
def angle_is_pi_over_4 (a b : ℝ^3) : Prop := real.angle a b = π / 4

-- Calculate the volume of the parallelepiped formed by a, b, and (b + b × a)
theorem volume_of_parallelepiped (ha : is_unit_vector a) (hb : is_unit_vector b) (hab : angle_is_pi_over_4 a b) :
  abs (a • ((b + b × a) × b)) = 1 / 2 :=
sorry

end volume_of_parallelepiped_l590_590150


namespace circle_with_diameter_not_passing_origin_l590_590390

theorem circle_with_diameter_not_passing_origin
  (curve_eq : ∀ x y : ℝ, 2 * x^2 - y^2 = 5)
  (l_intersects_curve : ∀ k : ℝ, ∃ x y : ℝ, y = k * x + 2 ∧ 2 * x^2 - (k * x + 2)^2 = 5)
  (A B : ℝ × ℝ)
  (A_on_curve : curve_eq A.1 A.2)
  (B_on_curve : curve_eq B.1 B.2)
  (A_on_line l B_on_line : ∃ k : ℝ, A.2 = k * A.1 + 2 ∧ B.2 = k * B.1 + 2) :
  ¬ ((A.1 * B.1 + A.2 * B.2) = 0) →
  ¬ circle_with_diameter_AB_passing_origin A B := sorry

end circle_with_diameter_not_passing_origin_l590_590390


namespace find_g_neg3_l590_590925

variable (g : ℚ → ℚ)

-- Given condition
axiom condition : ∀ x : ℚ, x ≠ 0 → 4 * g (1/x) + (3 * g x) / x = 3 * x^2

-- Theorem statement
theorem find_g_neg3 : g (-3) = -27 / 2 := 
by 
  sorry

end find_g_neg3_l590_590925


namespace binomial_constant_term_l590_590009

theorem binomial_constant_term (a : ℝ) :
  (∃ r : ℕ, 6 - 2 * r = 0 ∧ (-(a:ℝ))^r * nat.choose 6 r = 20) → a = -1 :=
by sorry

end binomial_constant_term_l590_590009


namespace probability_sum_greater_than_five_l590_590963

theorem probability_sum_greater_than_five (dice_outcomes : List (ℕ × ℕ)) (h: dice_outcomes = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (3,1), (3,2), (4,1), (5,1), (2,4)] ++ 
                              [(1,5), (2,6), (3,3), (3,4), (3,5), (3,6), (4,2), (4,3), (4,4), (4,5), (4,6), 
                               (5,2), (5,3), (5,4), (5,5), (5,6), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)]) :
  p_greater_5 = 2 / 3 := 
by
  sorry

end probability_sum_greater_than_five_l590_590963


namespace monotonicity_properties_range_of_a_inequality_a_one_l590_590812

section 
variable {a : ℝ} (x : ℝ) (n : ℕ)

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (a * x + 1)
noncomputable def g (x : ℝ) : ℝ := (x - 2) / (x + 2)
noncomputable def y (x : ℝ) (a : ℝ) : ℝ := f x a - g x

-- Proving the monotonicity for y = f(x) - g(x)
theorem monotonicity_properties (h1 : 0 < a) : 
  (∀ x:ℝ, 0 ≤ x → (a ≥ 1 → ∀ x:ℝ, 0 ≤ x → 0 ≤ derivative_derivative x / (1 + a * x) - 4 / (x + 2) ^ 2)) ∧
  ((0 < a ∧ a < 1) → ∀ x:ℝ, x > 2 * Real.sqrt (1 / a - 1) → 0 < derivative_derivative x / (1 + a * x) - 4 / (x + 2) ^ 2) := sorry

-- Proving the range of a
theorem range_of_a (h1 : ∀ x:ℝ, 0 ≤ x → f x a ≥ g x + 1) : 1 ≤ a :=
  sorry

-- When a = 1, proving the inequality
theorem inequality_a_one (h1 : a = 1) (n : ℕ) (hn: 0 < n) : 
  (∑ k in Finset.range n, 1 / (2 * k + 1)) < (1 / 2) * f n 1 := 
  sorry
end

end monotonicity_properties_range_of_a_inequality_a_one_l590_590812


namespace triangle_circle_property_l590_590166

-- Let a, b, and c be the lengths of the sides of a right triangle, where c is the hypotenuse.
variables {a b c : ℝ}

-- Let varrho_b be the radius of the circle inscribed around the leg b of the triangle.
variable {varrho_b : ℝ}

-- Assume the relationship a^2 + b^2 = c^2 (Pythagorean theorem).
axiom right_triangle : a^2 + b^2 = c^2

-- Prove that b + c = a + 2 * varrho_b
theorem triangle_circle_property (h : a^2 + b^2 = c^2) (radius_condition : varrho_b = (a*b)/(a+c-b)) : 
  b + c = a + 2 * varrho_b :=
sorry

end triangle_circle_property_l590_590166


namespace probability_six_integers_unique_tens_digit_l590_590542

theorem probability_six_integers_unique_tens_digit :
  (∃ (x1 x2 x3 x4 x5 x6 : ℕ),
    10 ≤ x1 ∧ x1 ≤ 79 ∧
    10 ≤ x2 ∧ x2 ≤ 79 ∧
    10 ≤ x3 ∧ x3 ≤ 79 ∧
    10 ≤ x4 ∧ x4 ≤ 79 ∧
    10 ≤ x5 ∧ x5 ≤ 79 ∧
    10 ≤ x6 ∧ x6 ≤ 79 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x1 ≠ x6 ∧
    x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x2 ≠ x6 ∧
    x3 ≠ x4 ∧ x3 ≠ x5 ∧ x3 ≠ x6 ∧
    x4 ≠ x5 ∧ x4 ≠ x6 ∧
    x5 ≠ x6 ∧
    tens_digit x1 ≠ tens_digit x2 ∧
    tens_digit x1 ≠ tens_digit x3 ∧
    tens_digit x1 ≠ tens_digit x4 ∧
    tens_digit x1 ≠ tens_digit x5 ∧
    tens_digit x1 ≠ tens_digit x6 ∧
    tens_digit x2 ≠ tens_digit x3 ∧
    tens_digit x2 ≠ tens_digit x4 ∧
    tens_digit x2 ≠ tens_digit x5 ∧
    tens_digit x2 ≠ tens_digit x6 ∧
    tens_digit x3 ≠ tens_digit x4 ∧
    tens_digit x3 ≠ tens_digit x5 ∧
    tens_digit x3 ≠ tens_digit x6 ∧
    tens_digit x4 ≠ tens_digit x5 ∧
    tens_digit x4 ≠ tens_digit x6 ∧
    tens_digit x5 ≠ tens_digit x6)
    →
  (probability := \(\frac{4375}{744407}\)).sorry

end probability_six_integers_unique_tens_digit_l590_590542


namespace position_of_2017_in_arithmetic_sequence_l590_590977

theorem position_of_2017_in_arithmetic_sequence :
  ∀ (n : ℕ), 4 + 3 * (n - 1) = 2017 → n = 672 :=
by
  intros n h
  sorry

end position_of_2017_in_arithmetic_sequence_l590_590977


namespace third_vertex_coordinates_l590_590588

def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem third_vertex_coordinates :
  (∀ x : ℝ, triangle_area 2 3 0 0 x 0 = 12 → x < 0 → x = -8) := by
  intro x hx hneg
  sorry

end third_vertex_coordinates_l590_590588


namespace total_robodinos_in_shipment_l590_590658

-- Definitions based on the conditions:
def percentage_on_display : ℝ := 0.30
def percentage_in_storage : ℝ := 0.70
def stored_robodinos : ℕ := 168

-- The main statement to prove:
theorem total_robodinos_in_shipment (T : ℝ) : (percentage_in_storage * T = stored_robodinos) → T = 240 := by
  sorry

end total_robodinos_in_shipment_l590_590658


namespace quadratic_roots_equal_l590_590949

theorem quadratic_roots_equal (a b c : ℝ) (h_eq : 4 * a = b) (h_eq2 : b * c = a): 
  b^2 - 4 * a * c = 0 → (4 * a^2 - 4 * a + c = 0) → 
  (root1 root2 : ℝ) (h_eq_roots : root1 = root2) :=
  sorry

end quadratic_roots_equal_l590_590949


namespace shortest_travel_distance_l590_590262

noncomputable def cowboy_travel_distance (C B : ℝ × ℝ) (d_stream d_east d_south : ℝ) : ℝ :=
  let C_reflect := (C.1, -C.2)
  sqrt ((B.1 - C_reflect.1)^2 + (B.2 - C_reflect.2)^2) + d_stream

theorem shortest_travel_distance
  (C B : ℝ × ℝ)
  (hC : C = (0, -6))
  (hB : B = (10, -11))
  (d_stream : ℝ)
  (h_stream : d_stream = 6)
  : cowboy_travel_distance C B d_stream = sqrt 389 + 6 :=
by
  rw [cowboy_travel_distance, hC, hB, h_stream]
  simp
  sorry

end shortest_travel_distance_l590_590262


namespace solve_for_x_l590_590517

theorem solve_for_x (x : ℝ) (h : x ≠ 0) : 4 * log 3 x = log 3 (6 * x) → x = real.cbrt 6 :=
by
  sorry

end solve_for_x_l590_590517


namespace sqrt_four_eq_two_or_neg_two_l590_590178

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 :=
by 
  sorry

end sqrt_four_eq_two_or_neg_two_l590_590178


namespace probability_unique_tens_digits_l590_590548

theorem probability_unique_tens_digits :
  let num_ways := 10^6 in
  let total_combinations := Nat.choose 70 6 in
  (num_ways : ℚ) / total_combinations = 625 / 74440775 :=
by 
  sorry

end probability_unique_tens_digits_l590_590548


namespace probability_A_or_B_complement_l590_590439

-- Define the sample space for rolling a die
def sample_space : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define Event A: the outcome is an even number not greater than 4
def event_A : Finset ℕ := {2, 4}

-- Define Event B: the outcome is less than 6
def event_B : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the complement of Event B
def event_B_complement : Finset ℕ := {6}

-- Mutually exclusive property of events A and B_complement
axiom mutually_exclusive (A B_complement: Finset ℕ) : A ∩ B_complement = ∅

-- Define the probability function
def probability (events: Finset ℕ) : ℚ := (events.card : ℚ) / (sample_space.card : ℚ)

-- Theorem stating the probability of event (A + B_complement)
theorem probability_A_or_B_complement : probability (event_A ∪ event_B_complement) = 1 / 2 :=
by 
  sorry

end probability_A_or_B_complement_l590_590439


namespace smallest_n_polynomials_sum_of_squares_l590_590341

theorem smallest_n_polynomials_sum_of_squares :
  ∃ (n : ℕ), (∀ (f : ℕ → ℚ[X]), (x : ℚ[X]), (x^2 + 7 = ∑ i in Finset.range n, (f i)^2) → n = 5) :=
sorry

end smallest_n_polynomials_sum_of_squares_l590_590341


namespace closest_integer_to_cube_root_of_150_l590_590599

theorem closest_integer_to_cube_root_of_150 : 
  ∃ (n : ℤ), ∀ m : ℤ, abs (150 - 5 ^ 3) < abs (150 - m ^ 3) → n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590599


namespace initial_average_correct_l590_590553

theorem initial_average_correct (A : ℕ) 
  (num_students : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ)
  (wrong_avg : ℕ) (correct_avg : ℕ) 
  (h1 : num_students = 30)
  (h2 : wrong_mark = 70)
  (h3 : correct_mark = 10)
  (h4 : correct_avg = 98)
  (h5 : num_students * correct_avg = (num_students * A) - (wrong_mark - correct_mark)) :
  A = 100 := 
sorry

end initial_average_correct_l590_590553


namespace centroids_coincide_l590_590786

theorem centroids_coincide {A B C M K P : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace K] [MetricSpace P]
  (BM_eq_one_third_BA : dist B M = (1 / 3) * dist B A)
  (AK_eq_one_third_AC : dist A K = (1 / 3) * dist A C)
  (CP_eq_one_third_CB : dist C P = (1 / 3) * dist C B)
  : centroid (A, B, C) = centroid (M, K, P) :=
sorry

end centroids_coincide_l590_590786


namespace petya_time_comparison_l590_590683

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l590_590683


namespace oranges_ratio_l590_590495

theorem oranges_ratio (T : ℕ) (h1 : 100 + T + 70 = 470) : T / 100 = 3 := by
  -- The solution steps are omitted.
  sorry

end oranges_ratio_l590_590495


namespace problem_1_min_problem_1_max_problem_2_l590_590808

noncomputable def f (x : ℝ) : ℝ := (2 - log x) / (x + 1)
noncomputable def F (x : ℝ) : ℝ := -(2 - log (x - 1)) / x

theorem problem_1_min (x : ℝ) (hx : 2 ≤ x ∧ x ≤ real.exp 2 + 1) : F x ≥ -1 := sorry

theorem problem_1_max (x : ℝ) (hx : 2 ≤ x ∧ x ≤ real.exp 2 + 1) : F x ≤ 0 := sorry

theorem problem_2 (x : ℝ) (hx : 0 < x) : 
  (x + 1) * (2 - log x) / (x + 1) + 1 / real.exp x < 
  3 + 2 / (real.exp x * x) := sorry

end problem_1_min_problem_1_max_problem_2_l590_590808


namespace smallest_m_l590_590568

noncomputable def is_prime (n : ℕ) : Prop :=
∃ p : ℕ, p > 1 ∧ nat.prime p ∧ p = n

theorem smallest_m : ∃ m x y : ℕ, 
  100 ≤ m ∧ m < 1000 ∧ 
  x < 10 ∧ y < 10 ∧ 
  x ≠ y ∧ 
  m = x * y * (10 * x + y) ∧ 
  is_prime (10 * x + y) ∧ 
  is_prime (x + y) ∧ 
  m = 138 :=
begin
  sorry
end

end smallest_m_l590_590568


namespace problem_statement_l590_590415

theorem problem_statement :
  (∑ n in Finset.range 1000, (n + 1) * (1001 - (n + 1))) = 1000 * 500 * 334 :=
by
  sorry

end problem_statement_l590_590415


namespace average_speed_l590_590646

theorem average_speed (d1 d2 d3 : ℕ) (t1 t2 t3 : ℕ)
  (h1 : d1 = 1000) (h2 : d2 = 1500) (h3 : d3 = 2000)
  (h4 : t1 = 10) (h5 : t2 = 15) (h6 : t3 = 20) :
  let total_distance := (d1 + d2 + d3) / 1000 in
  let total_time := (t1 + t2 + t3) / 60 in
  total_distance / total_time = 6 :=
by
  sorry

end average_speed_l590_590646


namespace spacy_subsets_15_l590_590695

def spacy (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | 2     => 3
  | n + 3 => spacy n + spacy (n-2)

theorem spacy_subsets_15 : spacy 15 = 406 := 
  sorry

end spacy_subsets_15_l590_590695


namespace max_min_values_of_f_l590_590761

def f (x : ℝ) : ℝ := (Real.sin x)^2 + 2 * (Real.cos x)

theorem max_min_values_of_f :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (Real.pi / 3) (4 * Real.pi / 3), f x ≤ max) ∧
    (∀ y ∈ Set.Icc (Real.pi / 3) (4 * Real.pi / 3), f y ≥ min) ∧
    max = 7 / 4 ∧ min = -2 :=
sorry

end max_min_values_of_f_l590_590761


namespace probability_same_color_boxes_l590_590025

def num_neckties := 6
def num_shirts := 5
def num_hats := 4
def num_socks := 3

def num_common_colors := 3

def total_combinations : ℕ := num_neckties * num_shirts * num_hats * num_socks

def same_color_combinations : ℕ := num_common_colors

def same_color_probability : ℚ :=
  same_color_combinations / total_combinations

theorem probability_same_color_boxes :
  same_color_probability = 1 / 120 :=
  by
    -- Proof would go here
    sorry

end probability_same_color_boxes_l590_590025


namespace fraction_of_red_marbles_after_tripling_blue_l590_590022

theorem fraction_of_red_marbles_after_tripling_blue (x : ℕ) (h₁ : ∃ y, y = (4 * x) / 7) (h₂ : ∃ z, z = (3 * x) / 7) :
  (3 * x / 7) / (((12 * x) / 7) + ((3 * x) / 7)) = 1 / 5 :=
by
  sorry

end fraction_of_red_marbles_after_tripling_blue_l590_590022


namespace sqrt_of_4_l590_590176

theorem sqrt_of_4 (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_of_4_l590_590176


namespace car_meeting_distance_l590_590960

theorem car_meeting_distance
  (distance_AB : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (midpoint_C : ℝ)
  (meeting_distance_from_C : ℝ) 
  (h1 : distance_AB = 245)
  (h2 : speed_A = 70)
  (h3 : speed_B = 90)
  (h4 : midpoint_C = distance_AB / 2) :
  meeting_distance_from_C = 15.31 := 
sorry

end car_meeting_distance_l590_590960


namespace general_formula_a_n_sum_T_formula_l590_590047

noncomputable def sequence_a : ℕ → ℕ
| 1     := 0
| 2     := 1
| (n+1) := n  -- This definition assumes the derived formula a_n = n - 1.

def sum_S (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_a

theorem general_formula_a_n (n : ℕ) (h1 : sequence_a 2 = 1) (h2 : 2 * sum_S n = n * sequence_a n) :
  sequence_a n = n - 1 := sorry

noncomputable def sequence_b (n : ℕ) : ℝ :=
  (sequence_a n + 1) / (2 ^ n)

def sum_T (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => sequence_b (i + 1))

theorem sum_T_formula (n : ℕ) (h : ∀ n, sequence_a n = n - 1) :
  sum_T n = 2 - (n + 2) / (2 ^ n) := sorry

end general_formula_a_n_sum_T_formula_l590_590047


namespace minimize_fraction_sum_l590_590490

theorem minimize_fraction_sum :
  ∃ p q r s : ℕ, 
    (p, q, r, s).perm (1, 2, 3, 4) ∧
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    (p : ℚ) / q + (r : ℚ) / s = 5 / 6 :=
begin
  sorry
end

end minimize_fraction_sum_l590_590490


namespace positive_value_of_A_l590_590079

theorem positive_value_of_A 
  (A : ℝ) 
  (h_rel : ∀ A B : ℝ, A \# B = A^2 + B^2) 
  (h_cond : A \# 5 = 169) : 
  A = 12 := by
  sorry

end positive_value_of_A_l590_590079


namespace quadratic_solution_exists_l590_590774

-- Define the conditions
variables (a b : ℝ) (h₀ : a ≠ 0)
-- The condition that the first quadratic equation has at most one solution
def has_at_most_one_solution (a b : ℝ) : Prop :=
  b^2 + 4*a*(a - 3) <= 0

-- The second quadratic equation
def second_equation (a b x : ℝ) : ℝ :=
  (b - 3) * x^2 + (a - 2 * b) * x + 3 * a + 3
  
-- The proof problem invariant in Lean 4
theorem quadratic_solution_exists (h₁ : has_at_most_one_solution a b) :
  ∃ x : ℝ, second_equation a b x = 0 :=
by
  sorry

end quadratic_solution_exists_l590_590774


namespace earthquake_relief_team_selection_l590_590146

theorem earthquake_relief_team_selection : 
    ∃ (ways : ℕ), ways = 590 ∧ 
      ∃ (orthopedic neurosurgeon internist : ℕ), 
      orthopedic + neurosurgeon + internist = 5 ∧ 
      1 ≤ orthopedic ∧ 1 ≤ neurosurgeon ∧ 1 ≤ internist ∧
      orthopedic ≤ 3 ∧ neurosurgeon ≤ 4 ∧ internist ≤ 5 := 
  sorry

end earthquake_relief_team_selection_l590_590146


namespace remainder_degrees_l590_590594

theorem remainder_degrees (p q : Polynomial ℝ) (deg_q : q.degree = 7) : 
  ∃ r : Polynomial ℝ, r.degree ∈ ({0, 1, 2, 3, 4, 5, 6} : set ℕ) :=
by
  sorry

end remainder_degrees_l590_590594


namespace coordinate_axes_not_in_any_quadrant_l590_590612

-- Define the notion of a point in 2D coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the quadrants
inductive Quadrant
| first
| second
| third
| fourth

-- Define the condition for a point belonging to a quadrant
def belongs_to_quadrant (p : Point) : Option Quadrant :=
  if p.x > 0 ∧ p.y > 0 then
    some Quadrant.first
  else if p.x < 0 ∧ p.y > 0 then
    some Quadrant.second
  else if p.x < 0 ∧ p.y < 0 then
    some Quadrant.third
  else if p.x > 0 ∧ p.y < 0 then
    some Quadrant.fourth
  else
    none

-- Define the theorem to be proved
theorem coordinate_axes_not_in_any_quadrant (p : Point) (h : p.x = 0 ∨ p.y = 0) :
  belongs_to_quadrant p = none :=
by
  sorry

end coordinate_axes_not_in_any_quadrant_l590_590612


namespace regular_polygon_sides_l590_590726

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l590_590726


namespace a_n_formula_T_n_sum_l590_590051

noncomputable def a_n (n : ℕ) : ℕ :=
if n = 1 then 0 else n - 1

def S_n (n : ℕ) : ℕ := (Finset.range n).sum a_n

-- Theorem for the first part: Finding the general formula for {a_n}
theorem a_n_formula (n : ℕ) (h : n ≠ 0) : a_n (n + 1) = n := by
  sorry

noncomputable def T_n (n : ℕ) : ℝ :=
(Finset.range n).sum (λ k, (a_n (k + 1) + 1) / 2^(k + 1 : ℝ))

-- Theorem for the second part: Finding the sum T_n of the first n terms of the sequence { (a_n + 1) / 2^n }
theorem T_n_sum (n : ℕ) : T_n n = 2 - (n + 2) / 2^n := by
  sorry

end a_n_formula_T_n_sum_l590_590051


namespace second_term_binomial_expansion_l590_590764

-- Definitions for binomial coefficients and conditions
def C (n k : ℕ) : ℕ := nat.fact n / (nat.fact k * nat.fact (n - k))

def term_in_binomial_expansion (a b : ℝ) (n r : ℕ) := C n r * (a ^ (n - r)) * (b ^ r)

-- Conditions for the specific problem
def a : ℝ := x
def b : ℝ := -1 / (7 * x)
def n : ℕ := 7
def r : ℕ := 1

-- The theorem to be proved
theorem second_term_binomial_expansion (x : ℝ) (hx : x ≠ 0) : 
  term_in_binomial_expansion x (-1 / (7 * x)) 7 1 = -x ^ 5 :=
by sorry

end second_term_binomial_expansion_l590_590764


namespace mr_smith_total_spending_l590_590493

def family_dining_cost : ℝ :=
  let seafood_station := 30
  let bbq_station := 25
  let salad_station := 15
  let dessert_station := 10
  let lobster_roll := 12
  let clam_chowder := 8
  
  let senior_discount := 0.10
  let college_discount := 0.05
  
  let drinks_cost := 2*2 + 2*3 + 1*4 + 3*1.50 + 1*5 + 1*10 + 6
  
  let cost_mr_smith_and_wife := 2 * seafood_station
  let cost_parents_before_discount := 2 * bbq_station + 2 * dessert_station
  let cost_parents_after_discount := cost_parents_before_discount * (1 - senior_discount)
  let cost_children := 3 * salad_station + 3 * dessert_station
  let cost_teenage_niece := seafood_station + lobster_roll
  let cost_nephews_before_discount := 2 * seafood_station
  let cost_nephews_after_discount := cost_nephews_before_discount * (1 - college_discount)
  let cost_cousin := bbq_station + clam_chowder

  let total_buffet_before_discount := cost_mr_smith_and_wife + cost_parents_before_discount + cost_children + cost_teenage_niece + cost_nephews_before_discount + cost_cousin
  let total_discount := cost_parents_before_discount * senior_discount + cost_nephews_before_discount * college_discount
  let total_buffet_after_discount := total_buffet_before_discount * (1 - total_discount / total_buffet_before_discount)

  total_buffet_after_discount + drinks_cost

theorem mr_smith_total_spending :
  family_dining_cost = 369.50 :=
by
  sorry

end mr_smith_total_spending_l590_590493


namespace no_sum_of_three_squares_l590_590514

theorem no_sum_of_three_squares (a k : ℕ) : 
  ¬ ∃ x y z : ℤ, 4^a * (8*k + 7) = x^2 + y^2 + z^2 :=
by
  sorry

end no_sum_of_three_squares_l590_590514


namespace probability_of_different_tens_digits_l590_590534

open Finset

-- Define the basic setup
def integers (n : ℕ) : Finset ℕ := {i in (range n) | i ≥ 10 ∧ i ≤ 79}

def tens_digit (n : ℕ) : ℕ := n / 10

def six_integers_with_different_tens_digits (s : Finset ℕ) : Prop :=
  s.card = 6 ∧ (s.map ⟨tens_digit, by simp⟩).card = 6

def favorable_ways : ℕ :=
  7 * 10^6

def total_ways : ℕ :=
  nat.choose 70 6

noncomputable def probability : ℚ :=
  favorable_ways / total_ways

-- The main statement
theorem probability_of_different_tens_digits :
  ∀ (s : Finset ℕ), six_integers_with_different_tens_digits s → 
  probability = 175 / 2980131 :=
begin
  intros s h,
  sorry
end

end probability_of_different_tens_digits_l590_590534


namespace simplify_expression_l590_590755

theorem simplify_expression (x : ℝ) :
  (√(x^2 - 4 * x + 4) + √(x^2 + 6 * x + 9)) = |x-2| + |x+3| :=
sorry

end simplify_expression_l590_590755


namespace smallest_x_for_multiple_of_450_and_648_l590_590980

theorem smallest_x_for_multiple_of_450_and_648 (x : ℕ) (hx : x > 0) :
  ∃ (y : ℕ), (450 * 36) = y ∧ (450 * 36) % 648 = 0 :=
by
  use (450 / gcd 450 648 * 648 / gcd 450 648)
  sorry

end smallest_x_for_multiple_of_450_and_648_l590_590980


namespace bike_average_speed_l590_590255

theorem bike_average_speed (distance time : ℕ)
    (h1 : distance = 48)
    (h2 : time = 6) :
    distance / time = 8 := 
  by
    sorry

end bike_average_speed_l590_590255


namespace digit_157_of_5_by_13_l590_590590

theorem digit_157_of_5_by_13 : ∀ seq : List Nat, 
  seq = [3, 8, 4, 6, 1, 5] → (seq.get! ((157 % 6) - 1)) = 1 :=
by
  intro seq h_seq
  rw [h_seq, List.get!, Nat.mod_eq_of_lt, List.nth_le] 
  exact rfl
  simp
  linarith

end digit_157_of_5_by_13_l590_590590


namespace problem_statement_l590_590248

noncomputable def probability_different_colors : ℚ :=
  let p_red := 7 / 11
  let p_green := 4 / 11
  (p_red * p_green) + (p_green * p_red)

theorem problem_statement :
  let p_red := 7 / 11
  let p_green := 4 / 11
  (p_red * p_green) + (p_green * p_red) = 56 / 121 := by
  sorry

end problem_statement_l590_590248


namespace problem_statement_l590_590887

noncomputable def sqrt_five : ℝ := Real.sqrt 5

def m : ℕ := 2

def n : ℝ := sqrt_five - 2

theorem problem_statement :
  m * (m - (1 / n)) ^ 3 = -10 * sqrt_five := 
by 
  sorry

end problem_statement_l590_590887


namespace evaluate_expression_l590_590321

-- Define the base value
def base := 3000

-- Define the exponential expression
def exp_value := base ^ base

-- Prove that base * exp_value equals base ^ (1 + base)
theorem evaluate_expression : base * exp_value = base ^ (1 + base) := by
  sorry

end evaluate_expression_l590_590321


namespace probability_of_rerolling_exactly_two_dice_l590_590857

-- Define the event of rerolling exactly two dice
def rerolling_two_dice (die1 die2 die3 : ℕ) : Prop :=
  ∃ (d1 d2 : ℕ), d1 ≠ die1 ∧ d2 ≠ die2 ∧ die3 ∈ {1, 2, 3, 4, 5, 6}

-- Define a fair roll of a six-sided die which can result in the values 1 through 6
def fair_six_sided_die := {1, 2, 3, 4, 5, 6}

-- Define the probability calculation
noncomputable def probability_reroll_two_dice : ℚ :=
  1 / 6

-- Main theorem statement: Given three fair six-sided dice, prove the probability
-- that Jason chooses to reroll exactly two of the dice to maximize his chances 
-- of getting a sum of 9 or more is 1/6.
theorem probability_of_rerolling_exactly_two_dice :
  ∀ (die1 die2 die3 : ℕ), 
  die1 ∈ fair_six_sided_die → die2 ∈ fair_six_sided_die → die3 ∈ fair_six_sided_die →
  rerolling_two_dice die1 die2 die3 → probability_reroll_two_dice = 1 / 6 :=
by sorry

end probability_of_rerolling_exactly_two_dice_l590_590857


namespace sum_of_4th_and_12th_term_l590_590437

theorem sum_of_4th_and_12th_term 
  (a d : ℝ) 
  (S : ℝ) 
  (h : S = 120) :
  2 * a + 14 * d = 16 :=
by
  have h1 : S = (15 / 2) * (2 * a + 14 * d), from
    have h_eq : S = (15 / 2) * (2 * a + 14 * d), from sorry,
    h_eq,
  have h2 : 120 = (15 / 2) * (2 * a + 14 * d), from h.symm ▸ h1,
  have h3 : 240 = 15 * (2 * a + 14 * d), from (by sorry),
  have h4 : 2 * a + 14 * d = 16, from (by sorry),
  exact h4  

end sum_of_4th_and_12th_term_l590_590437


namespace license_plates_count_l590_590266

def number_of_license_plates : ℕ :=
  let digit_choices := 10^5
  let letter_block_choices := 3 * 26^2
  let block_positions := 6
  digit_choices * letter_block_choices * block_positions

theorem license_plates_count : number_of_license_plates = 1216800000 := by
  -- proof steps here
  sorry

end license_plates_count_l590_590266


namespace max_intersection_points_l590_590261

def line := ℝ → ℝ

def circle (c : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) := 
  {p | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

def parabola (a : ℝ) (b : ℝ) (c : ℝ) : set (ℝ × ℝ) := 
  {p | p.2 = a * p.1^2 + b * p.1 + c}

def max_intersections (circle : set (ℝ × ℝ)) (line : line) (parabola : set (ℝ × ℝ)) : ℕ :=
  let circle_line_intersections := 2 in
  let line_parabola_intersections := 2 in
  let circle_parabola_intersections := 4 in
  circle_line_intersections + line_parabola_intersections + circle_parabola_intersections

theorem max_intersection_points (c : ℝ × ℝ) (r a b : ℝ) (line : line) :
  max_intersections (circle c r) line (parabola a b 0) = 8 :=
sorry

end max_intersection_points_l590_590261


namespace out_of_pocket_l590_590066

def cost_first_ring : ℕ := 10000
def cost_second_ring : ℕ := 2 * cost_first_ring
def sale_first_ring : ℕ := cost_first_ring / 2

theorem out_of_pocket : cost_first_ring + cost_second_ring - sale_first_ring = 25000 := by
  unfold cost_first_ring cost_second_ring sale_first_ring
  simp
  sorry

end out_of_pocket_l590_590066


namespace trajectory_equation_of_center_of_circle_l590_590378

-- Define the given conditions and the proof goal
theorem trajectory_equation_of_center_of_circle :
  ∀ (a P : ℝ) (center_P : ℝ × ℝ),
    let C := (-a, a) in
    let circle1 : ℝ × ℝ → ℝ := λ p, p.1^2 + p.2^2 - a * p.1 + 2 * p.2 + 1 in
    let circle2 : ℝ × ℝ → ℝ := λ p, p.1^2 + p.2^2 - 1 in
    let line : ℝ × ℝ → Prop := λ p, p.2 = p.1 - 1 in
    (∀ p, circle1 p = circle2 (p.1, p.2 - 1)) →
    center_P = (P, (2 * P + 4) / 2) →
    (sqrt ((-a - P)^2 + (a - (2 * P + 4) / 2)^2) = abs P) →
    y^2 + 4 * P - 4 * y + 8 = 0 :=
by sorry

end trajectory_equation_of_center_of_circle_l590_590378


namespace gross_profit_value_l590_590946

theorem gross_profit_value
  (SP : ℝ) (C : ℝ) (GP : ℝ)
  (h1 : SP = 81)
  (h2 : GP = 1.7 * C)
  (h3 : SP = C + GP) :
  GP = 51 :=
by
  sorry

end gross_profit_value_l590_590946


namespace general_formula_sum_Tn_l590_590053

-- Conditions
def a_seq (n : ℕ) : ℕ := if n = 1 then 0 else n - 1
def S_n (n : ℕ) : ℕ := (finset.range n).sum (λ i, a_seq (i + 1))

-- Given conditions
axiom a2 : a_seq 2 = 1
axiom S_n_cond (n : ℕ) : 2 * S_n n = n * a_seq n

-- (1) General formula for {a_n}
theorem general_formula (n : ℕ) : a_seq n = n - 1 :=
by sorry

-- (2) Sum of the first n terms of the sequence {(\frac{a_n + 1}{2^n})}
def T_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, (a_seq (k + 1) + 1 : ℝ) / (2 : ℝ) ^ (k + 1))

theorem sum_Tn (n : ℕ) : T_n n = 2 - (n + 2) / (2 : ℝ) ^ n :=
by sorry

end general_formula_sum_Tn_l590_590053


namespace well_centered_decomposable_iff_prime_divisors_l590_590116

variable (n : ℕ) (h1 : n > 2) (h2 : n % 2 = 1)

structure Polygon :=
(vertices : List (ℝ × ℝ))
(centered_at_origin : (vertices.map (λ p, p.1)).sum = 0 ∧ (vertices.map (λ p, p.2)).sum = 0)

structure SubPolygon (G: Polygon) :=
(vertices_subset_of : G.vertices)
(size_at_least_three : vertices_subtype.length ≥ 3)
(well_centered : (vertices_subtype.map (λ p, p.1)).sum = 0 ∧ (vertices_subtype.map (λ p, p.2)).sum = 0)

def decomposable (S : SubPolygon G) : Prop :=
∃ (disjoint_union_polygons : List SubPolygon), S.vertices_subset_of = disjoint_union_polygons.bind (λ p, p.vertices_subset_of) ∧ ∀ p ∈ disjoint_union_polygons, p.size_at_least_three

theorem well_centered_decomposable_iff_prime_divisors (G : Polygon) (h3 : n = G.vertices.length) :
  (∀ (S : SubPolygon G), S.well_centered → decomposable S) ↔ (n.prime_divisors.length ≤ 2) :=
sorry

end well_centered_decomposable_iff_prime_divisors_l590_590116


namespace measure_of_angle_B_area_of_triangle_ABC_l590_590375

theorem measure_of_angle_B (a b c : ℝ) (B A C : ℝ) 
  (h1 : sin B = sin (π / 3)) (h2 : ∀ (a₀ a₁ : ℝ), a₀ * a₁ = a) :
  B = π / 3 := by
  sorry

theorem area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ)
  (h1a : b = 4) (h1b : c = 4 * sqrt 6 / 3) (h1c : B = π / 3)
  (h2 : sin A = (sqrt 6 + sqrt 2) / 4) :
  (1 / 2) * b * c * sin A = 4 + (4 * sqrt 3 / 3) := by
  sorry

end measure_of_angle_B_area_of_triangle_ABC_l590_590375


namespace debby_total_photos_l590_590127

theorem debby_total_photos (friends_photos family_photos : ℕ) (h1 : friends_photos = 63) (h2 : family_photos = 23) : friends_photos + family_photos = 86 :=
by sorry

end debby_total_photos_l590_590127


namespace second_sibling_age_difference_l590_590188

theorem second_sibling_age_difference :
  ∃ x : ℕ, let Y := 17 in
  (Y + Y + x + (Y + 4) + (Y + 7)) / 4 = 21 ∧ x = 5 :=
begin
  let Y := 17,
  use 5,
  rw add_assoc,
  rw add_assoc,
  sorry
end

end second_sibling_age_difference_l590_590188


namespace problem_solution_l590_590478

-- Definitions of conditions
def f (x : ℤ) (a b : ℤ) : ℤ := a * x + b

variable (a b : ℤ)
axiom cond1 : f (f 0 a b) a b = 0
axiom cond2 : f (f (f 4 a b) a b) a b = 9

-- Theorem to be proved
theorem problem_solution: 
  f (f (f (f <$> (list.range 2015).tail) a b)) a b = 2029105 := 
sorry

-- Helper function to map f over a list
def map_f (l : list ℤ) (a b : ℤ) : list ℤ := list.map (λ x, f x a b) l

-- Define what f(f(f(f(...)))) == x does to a list
def f_iter (l : list ℤ) (a b : ℤ) (n : ℕ) : list ℤ := 
if n = 0 then l else f_iter (map_f l a b) a b (n - 1)

-- Map over 2014 terms and prove the sum
def iterate_f_2014 (a b : ℤ) : ℤ :=
(list.sum (f_iter ((list.range 2015).tail) a b 4))

#eval f (f (f (f <$> (list.range 2015).tail) (-1) 0)) (-1) 0  -- This should evaluate to 2029105

end problem_solution_l590_590478


namespace probability_at_least_one_event_completed_expected_value_of_X_l590_590841

-- Define the conditions for probabilities and independence
def P_A : ℝ := 3/4
def P_B : ℝ := 3/4
def P_C : ℝ := 2/3

-- Define the complement probabilities
def P_not_A : ℝ := 1 - P_A
def P_not_B : ℝ := 1 - P_B
def P_not_C : ℝ := 1 - P_C

-- Define the composite event of completing at least one event
def P_at_least_one : ℝ := 1 - (P_not_A * P_not_B * P_not_C)

-- Rule-based, let (a : ℝ) be an output factor
def a : ℝ := 1.0 -- can adjust for specific numerical factor's emphasis
def prob_X_0 : ℝ := 7 / 16
def prob_X_a : ℝ := 63 / 256
def prob_X_3a : ℝ := 21 / 256
def prob_X_6a : ℝ := 60 / 256

-- Expected value calculation according to given probabilities
def E_X : ℝ := 0 * prob_X_0 + a * prob_X_a + 3 * a * prob_X_3a + 6 * a * prob_X_6a

theorem probability_at_least_one_event_completed :
  P_at_least_one = 47 / 48 := by
  -- Proof steps
  sorry

theorem expected_value_of_X :
  E_X = 243 * a / 128 := by
  -- Proof steps
  sorry

end probability_at_least_one_event_completed_expected_value_of_X_l590_590841


namespace regular_polygon_sides_l590_590748

theorem regular_polygon_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle i = 160)) : n = 18 :=
by
  -- Proof goes here
  sorry

end regular_polygon_sides_l590_590748


namespace abs_g_2_l590_590878

noncomputable def g : Polynomial ℝ := sorry

axiom poly_deg : g.degree = 3
axiom poly_real_coeffs : ∀ n, g.coeff n ∈ ℝ
axiom g_values : abs (g.eval 0) = 10 
               ∧ abs (g.eval 1) = 10 
               ∧ abs (g.eval 3) = 10 
               ∧ abs (g.eval 4) = 10 
               ∧ abs (g.eval 5) = 10 
               ∧ abs (g.eval 8) = 10

theorem abs_g_2 : abs (g.eval 2) = 20 := sorry

end abs_g_2_l590_590878


namespace expression_equals_2_plus_sqrt_3_l590_590700

noncomputable def calculate_expression : ℝ :=
  (-1/2)^0 + (1/3)^(-1) * (2 / real.sqrt 3) - |real.tan (real.pi / 4) - real.sqrt 3|

theorem expression_equals_2_plus_sqrt_3 : calculate_expression = 2 + real.sqrt 3 :=
by
  sorry

end expression_equals_2_plus_sqrt_3_l590_590700


namespace polyhedron_impossible_l590_590134

-- Assume definitions for points and projections
def PointSet : Type := Type
def Line (points: PointSet) := points
def Plane (points: PointSet) := points
def Polyhedron (points: PointSet) := points

variables (ABCD A1B1C1D1: Polyhedron) (proj_plane: Plane)

-- Conditions: Polyhedron ABCD A1B1C1D1 is ortho-projected onto ABCD
axiom ortho_projection : Polyhedron → Plane → Prop

-- Question: Prove impossibility
theorem polyhedron_impossible 
  (H: ortho_projection ABCD A1B1C1D1 proj_plane) : 
  ¬ ∃ (A2 B2 C2 D2 : PointSet), 
    let common_line := Line (A2, B2, C2, D2) in
    (A2 ∈ Line (A1, B1)) ∧ 
    (B2 ∈ Line (B1, A1)) ∧ 
    (C2 ∈ Line (C1, A1)) ∧ 
    (D2 ∈ Line (D1, A1)) ∧ 
    common_line ⊆ Line (A2, B2, C2, D2) :=
sorry

end polyhedron_impossible_l590_590134


namespace num_different_lineups_l590_590894

def total_players : ℕ := 18

def num_goalie_choices : ℕ := 1

def num_field_players : ℕ := 11

theorem num_different_lineups (n k : ℕ) (hn : n = total_players - num_goalie_choices) (hk : k = num_field_players) :
  (total_players - num_goalie_choices).choose(num_field_players) * total_players = 222768 :=
by
  sorry

end num_different_lineups_l590_590894


namespace vasya_decision_min_increment_l590_590302

noncomputable def minimum_increment (n t : ℕ) : ℕ :=
  let businessmen := (0.8 * 0.25 * n : ℕ)
  let tourists := (0.2 * 0.25 * n : ℕ)
  let revenue_uniform := n * t
  let revenue_discriminative := (0.75 * n * t) + (businessmen * (t + (t / 4)))
  t / 4

theorem vasya_decision_min_increment (n t : ℕ) (H : t > 0) : minimum_increment n t = t / 4 :=
begin
  sorry
end

end vasya_decision_min_increment_l590_590302


namespace independent_functions_l590_590417

noncomputable theory

-- Definitions of the random variables and their functions
variables {ι : Type*} {Ω : Type*} [MeasurableSpace Ω]
variables {ξ : ι → Ω → ℝ} (indep_ξ : ∀ i j, i ≠ j → ∀ ω, ξ i ω ⊥ ξ j ω)
variables {k n : ℕ} {φ₁ : (ℕ → ℝ) → ℝ} {φ₂ : (ℕ → ℝ) → ℝ}

-- The statement to prove that φ₁ and φ₂ are independent functions of the random variables
theorem independent_functions
  (h1 : φ₁ = λ ω, φ₁ (λ i, ξ i ω))
  (h2 : φ₂ = λ ω, φ₂ (λ i, ξ (k + i) ω)) :
  ∀ z₁ z₂, Prob { ω | φ₁ ω = z₁ } * Prob { ω | φ₂ ω = z₂ } = Prob { ω | φ₁ ω = z₁ ∧ φ₂ ω = z₂ } :=
sorry

end independent_functions_l590_590417


namespace function_characterization_l590_590303

noncomputable def f (x : ℝ) : ℝ := 
  if h : 0 ≤ x ∧ x < 2 then 2 / (2 - x) else if 2 ≤ x then 0 else 0

theorem function_characterization :
  ∀ f : ℝ → ℝ, 
  (∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → f (x * f y) * f y = f (x + y)) →
  f 2 = 0 →
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0) →
  ∀ x : ℝ, f x = 
    if 0 ≤ x ∧ x < 2 then 2 / (2 - x) else if 2 ≤ x then 0 else 0 :=
begin
  intros f h1 h2 h3,
  ext x,
  by_cases h : (0 ≤ x ∧ x < 2),
  { rw [if_pos h], sorry },
  { by_cases h' : (2 ≤ x),
    { rw [if_neg h, if_pos h'], sorry },
    { rw [if_neg h, if_neg h'], exact false.elim (h.left.not_le h'.le) } },
end

end function_characterization_l590_590303


namespace find_bin_prices_and_min_cost_l590_590202

-- Definitions based on given conditions
def cost_eq1 (x y : ℕ) : Prop := x + 2 * y = 340
def cost_eq2 (x y : ℕ) : Prop := 3 * x + y = 420
def total_cost (x : ℕ) : ℕ := -20 * x + 3600

-- Problem Statement
theorem find_bin_prices_and_min_cost :
  ∃ (x y : ℕ), cost_eq1 x y ∧ cost_eq2 x y ∧
  (x = 100 ∧ y = 120) ∧
  (∀ t : ℕ, t ≤ 16 → total_cost t ≥ total_cost 16) ∧
  (total_cost 16 = 3280) :=
sorry

end find_bin_prices_and_min_cost_l590_590202


namespace actual_time_greater_than_planned_time_l590_590673

def planned_time (a V : ℝ) : ℝ := a / V

def actual_time (a V : ℝ) : ℝ := (a / (2.5 * V)) + (a / (1.6 * V))

theorem actual_time_greater_than_planned_time (a V : ℝ) (hV : V > 0) : 
  actual_time a V > planned_time a V :=
by 
  sorry

end actual_time_greater_than_planned_time_l590_590673


namespace two_f_one_lt_f_four_l590_590797

theorem two_f_one_lt_f_four
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f (-x - 2))
  (h2 : ∀ x, x > 2 → x * (deriv f x) > 2 * (deriv f x) + f x) :
  2 * f 1 < f 4 :=
sorry

end two_f_one_lt_f_four_l590_590797


namespace unit_digit_7_pow_2023_l590_590126

theorem unit_digit_7_pow_2023 : (7^2023) % 10 = 3 :=
by
  -- Provide proof here
  sorry

end unit_digit_7_pow_2023_l590_590126


namespace probability_six_integers_diff_tens_l590_590524

-- Defining the range and conditions for the problem
def set_of_integers : Finset ℤ := Finset.range 70 \ Finset.range 10

def has_different_tens_digit (s : Finset ℤ) : Prop :=
  (s.card = 6) ∧ (∀ x y ∈ s, x ≠ y → (x / 10) ≠ (y / 10))

noncomputable def num_ways_choose_six_diff_tens : ℚ :=
  ((7 : ℚ) * (10^6 : ℚ))

noncomputable def total_ways_choose_six : ℚ :=
  (Nat.choose 70 6 : ℚ)

noncomputable def probability_diff_tens : ℚ :=
  num_ways_choose_six_diff_tens / total_ways_choose_six

-- Statement claiming the required probability
theorem probability_six_integers_diff_tens :
  probability_diff_tens = 1750 / 2980131 :=
by
  sorry

end probability_six_integers_diff_tens_l590_590524


namespace triangle_third_side_length_l590_590897

theorem triangle_third_side_length (a b c : ℝ) (B C : ℝ) (h1 : b = 12) (h2 : c = 18) 
  (h3 : B = 3 * C) (h4 : 0 < B) (h5 : 0 < C) (h6 : B + C < π)
  (h7 : a = 6 * Real.sqrt 13 + 6 * Real.sqrt 10 ∨ a = 6 * Real.sqrt 13 - 6 * Real.sqrt 10) 
  : a = 6 * Real.sqrt 13 + 6 * Real.sqrt 10 ∨ a = 6 * Real.sqrt 13 - 6 * Real.sqrt 10 :=
begin
  sorry,
end

end triangle_third_side_length_l590_590897


namespace cube_construction_symmetry_l590_590634

/-- 
 The number of distinct ways to construct a 3x3x3 cube using 9 red, 9 white, and 9 blue unit 
 cubes (considering two constructions identical if one can be rotated to match the other) 
 is equal to the result given by applying Burnside's Lemma to the symmetry group of the cube.
-/ 
theorem cube_construction_symmetry :
  let G := SymmetricGroup.group 3;
  let fixed_points (g : G) : ℕ := sorry;
  (1 / G.card) * ∑ (g : G), fixed_points g = sorry := 
sorry

end cube_construction_symmetry_l590_590634


namespace value_of_a_l590_590418

theorem value_of_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h1 : a^b = b^a) (h2 : b = 4 * a) : 
  a = real.cbrt 4 :=
by
  sorry

end value_of_a_l590_590418


namespace factorial_ratio_integer_count_l590_590343

theorem factorial_ratio_integer_count :
  {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ ((n + 1) ^ 2 - 1)! / (n!^(n + 1)) \in ℤ}.finite.card = 75 :=
sorry

end factorial_ratio_integer_count_l590_590343


namespace vector_expression_result_l590_590347

variable (e : Type) [AddCommMonoid e] [Module ℝ e]

def a : e := 5 • e
def b : e := -3 • e
def c : e := 4 • e

theorem vector_expression_result : 2 • a - 3 • b + c = 23 • e := by
  sorry

end vector_expression_result_l590_590347


namespace no_solutions_interval_length_l590_590345

theorem no_solutions_interval_length : 
  (∀ x a : ℝ, |x| ≠ ax - 2) → ([-1, 1].length = 2) :=
by {
  sorry
}

end no_solutions_interval_length_l590_590345


namespace regular_polygon_sides_l590_590736

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ α, α = 160 → ∑ x in range n, (180 - (180 - 160)) = 360) : n = 18 :=
by
  sorry

end regular_polygon_sides_l590_590736


namespace decagon_diagonals_intersection_probability_l590_590206

theorem decagon_diagonals_intersection_probability :
  ∃ (p : ℚ), p = 42 / 119 ∧ ∀ (D : finset (fin 10 × fin 10)),
  let diagonals := D.filter (λ d, d.1 ≠ d.2) in
  ∃ (D₁ D₂ : (fin 10 × fin 10)), D₁ ∈ diagonals ∧ D₂ ∈ diagonals ∧ 
  (∃ (pts : finset (fin 10)), pts.card = 4 ∧ 
  ∀ (x y : D₁.1 :: D₁.2 :: D₂.1 :: D₂.2 :: ∅),
  (∃ (a b c d : fin 10), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  finset.card (insert a (insert b (insert c {d}))) = 4))
  → p = (diagonals.card.choose 2)⁻¹ * (∃ pts : finset (fin 10), pts.card = 4) :=
by
  intros D diagonals D₁ D₂
  sorry

end decagon_diagonals_intersection_probability_l590_590206


namespace correct_answer_l590_590354

def f : ℝ → ℝ :=
λ x, if x > 0 then 2 * x else f (x + 1)

theorem correct_answer : f (-4 / 3) = 4 / 3 :=
sorry

end correct_answer_l590_590354


namespace find_degree_of_alpha_l590_590837

theorem find_degree_of_alpha
  (x : ℝ)
  (alpha : ℝ := x + 40)
  (beta : ℝ := 3 * x - 40)
  (h_parallel : alpha + beta = 180) :
  alpha = 85 :=
by
  sorry

end find_degree_of_alpha_l590_590837


namespace find_positive_integer_triples_l590_590332

theorem find_positive_integer_triples (a m n : ℕ) (h1 : a > 1) (h2 : m < n)
  (h3 : Int.primeFactors (a^m - 1) = Int.primeFactors (a^n - 1)) :
  ∃ l : ℕ, l ≥ 2 ∧ a = 2^l - 1 ∧ m = 1 ∧ n = 2 :=
sorry

end find_positive_integer_triples_l590_590332


namespace y_completion_time_l590_590620

noncomputable def work_done (days : ℕ) (rate : ℚ) : ℚ := days * rate

theorem y_completion_time (X_days Y_remaining_days : ℕ) (X_rate Y_days : ℚ) :
  X_days = 40 →
  work_done 8 (1 / X_days) = 1 / 5 →
  work_done Y_remaining_days (4 / 5 / Y_remaining_days) = 4 / 5 →
  Y_days = 35 :=
by
  intros hX hX_work_done hY_work_done
  -- With the stated conditions, we should be able to conclude that Y_days is 35.
  sorry

end y_completion_time_l590_590620


namespace find_constants_l590_590080

def N : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![0, 4]
]

def I : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![1, 0],
  ![0, 1]
]

theorem find_constants (c d : ℚ) : 
  (N⁻¹ = c • N + d • I) ↔ (c = -1/12 ∧ d = 7/12) :=
by
  sorry

end find_constants_l590_590080


namespace general_formula_a_n_sum_T_formula_l590_590045

noncomputable def sequence_a : ℕ → ℕ
| 1     := 0
| 2     := 1
| (n+1) := n  -- This definition assumes the derived formula a_n = n - 1.

def sum_S (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_a

theorem general_formula_a_n (n : ℕ) (h1 : sequence_a 2 = 1) (h2 : 2 * sum_S n = n * sequence_a n) :
  sequence_a n = n - 1 := sorry

noncomputable def sequence_b (n : ℕ) : ℝ :=
  (sequence_a n + 1) / (2 ^ n)

def sum_T (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => sequence_b (i + 1))

theorem sum_T_formula (n : ℕ) (h : ∀ n, sequence_a n = n - 1) :
  sum_T n = 2 - (n + 2) / (2 ^ n) := sorry

end general_formula_a_n_sum_T_formula_l590_590045


namespace clothing_prices_and_purchase_plans_l590_590846

theorem clothing_prices_and_purchase_plans :
  ∃ (x y : ℕ) (a : ℤ), 
  x + y = 220 ∧
  6 * x = 5 * y ∧
  120 * a + 100 * (150 - a) ≤ 17000 ∧
  (90 ≤ a ∧ a ≤ 100) ∧
  x = 100 ∧
  y = 120 ∧
  (∀ b : ℤ, (90 ≤ b ∧ b ≤ 100) → 120 * b + 100 * (150 - b) ≥ 16800)
  :=
sorry

end clothing_prices_and_purchase_plans_l590_590846


namespace floor_sum_eq_n_solution_count_le_l590_590907

theorem floor_sum_eq_n_solution_count_le
  (k : ℕ) 
  (a : Fin k → ℕ)
  (h_sum_gt_one : (Finset.univ : Finset (Fin k)).sum (λ i, 1 / (a i : ℝ)) > 1) 
  : ∀ (n : ℕ), (∀ i, ⌊n / a i⌋ + ⌊n / a i⌋ + ... + ⌊n / a i⌋ = n) → 
          (∃ S : Finset ℕ, S.card ≤ (Finset.univ : Finset (Fin k)).prod a) := 
sorry

end floor_sum_eq_n_solution_count_le_l590_590907


namespace rotate_ray_acb_l590_590565

theorem rotate_ray_acb (h : ∠ACB = 50) : ∠ACB' = 90 :=
sorry

end rotate_ray_acb_l590_590565


namespace sally_found_more_balloons_l590_590917

def sally_original_balloons : ℝ := 9.0
def sally_new_balloons : ℝ := 11.0

theorem sally_found_more_balloons :
  sally_new_balloons - sally_original_balloons = 2.0 :=
by
  -- math proof goes here
  sorry

end sally_found_more_balloons_l590_590917


namespace max_value_e_l590_590172

def b (n : ℕ) : ℕ := 120 + n^2
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem max_value_e : ∃ n, ∀ m, e m ≤ 5 :=
by
  sorry

end max_value_e_l590_590172


namespace odd_function_property_l590_590384

def f (x : ℝ) : ℝ := if x >= 0 then x^2 - 2*x else -x*(x + 2)

theorem odd_function_property (f_odd : ∀ x : ℝ, f (-x) = -f x) : 
  ∀ x : ℝ, x < 0 → f x = -x*(x + 2) :=
by
  intros x hx
  have h1 : f (-x) = x^2 + 2*x := by
    rw f
    rw if_neg
    exact lt_neg_iff_eq_neg_of_pos.mpr hx
  have h2 : f x = -f (-x) := f_odd x
  rw h1 at h2
  sorry

end odd_function_property_l590_590384


namespace monic_quartic_polynomial_with_roots_l590_590330

theorem monic_quartic_polynomial_with_roots :
  ∃ (p : Polynomial ℚ), p.monic ∧ p.degree = 4 ∧
  (Polynomial.aeval (3 + Real.sqrt 5) p = 0) ∧ 
  (Polynomial.aeval (2 - Real.sqrt 7) p = 0) ∧
  p = Polynomial.C (1:ℚ) * (X^4 - 10*X^3 + 25*X^2 + 2*X - 12) := 
sorry

end monic_quartic_polynomial_with_roots_l590_590330


namespace tom_seashells_after_giving_l590_590585

variable (x y : ℝ)

theorem tom_seashells_after_giving : 
  let total_seashells := x + y
  let jessica_share := 0.30 * total_seashells
  let toms_seashells_after := total_seashells - jessica_share
  in toms_seashells_after = 0.70 * total_seashells := by 
  sorry

end tom_seashells_after_giving_l590_590585


namespace product_of_distances_l590_590128

noncomputable theory

open Set 

variable {Point : Type} [MetricSpace Point]

variable (O : Point) (r : ℝ) (A : Fin 5 → Point) (P : Point) (Q : Fin 5 → Point) (d : Fin 5 → ℝ)

def is_unit_circle (O : Point) (r : ℝ) (A : Fin 5 → Point) :=
  r = 1 ∧ ∀ i, dist O (A i) = r

def is_inside_circle (O : Point) (r : ℝ) (P : Point) :=
  dist O P < r

axiom intersect_segments (A : Point) (B : Point) (C : Point) (D : Point) : Point

def Q_def (A : Fin 5 → Point) (P : Point) (i : Fin 5) :=
  intersect_segments (A i) (A (i + 2)) P (A (i + 1))

def d_def (O : Point) (Q : Fin 5 → Point) (i : Fin 5) :=
  dist O (Q i)

theorem product_of_distances
  (O : Point) (A : Fin 5 → Point) (P : Point) (Q : Fin 5 → Point) (d : Fin 5 → ℝ)
  (h1 : is_unit_circle O 1 A)
  (h2 : is_inside_circle O 1 P)
  (h3 : ∀ i, Q i = intersect_segments (A i) (A (i + 2)) P (A (i + 1)))
  (h4 : ∀ i, d i = dist O (Q i))
  : (∏ i, dist (A i) (Q i)) = (∏ i, (1 - (d i)^2))^(1/2) :=
by
  sorry

end product_of_distances_l590_590128


namespace bad_carrots_count_l590_590296

-- Define the number of carrots each person picked and the number of good carrots
def carol_picked := 29
def mom_picked := 16
def good_carrots := 38

-- Define the total number of carrots picked and the total number of bad carrots
def total_carrots := carol_picked + mom_picked
def bad_carrots := total_carrots - good_carrots

-- State the theorem that the number of bad carrots is 7
theorem bad_carrots_count :
  bad_carrots = 7 :=
by
  sorry

end bad_carrots_count_l590_590296


namespace find_quartic_poly_l590_590329

noncomputable def quartic_poly : Polynomial ℚ := 
  Polynomial.of_coeffs [1, -10, 17, 18, -12]

def isMonic (p : Polynomial ℚ) : Prop := p.leadingCoeff = 1

def hasRationalCoeffs (p : Polynomial ℚ) : Prop := 
  ∀ i, is_rat (coeff p i)

def isRoot (p : Polynomial ℚ) (x : ℚ) : Prop := 
  p.eval x = 0

theorem find_quartic_poly :
  ∃ (p : Polynomial ℚ), isMonic p ∧ hasRationalCoeffs p ∧ isRoot p (3 + √5) ∧ isRoot p (2 - √7) ∧ p = quartic_poly :=
  sorry

end find_quartic_poly_l590_590329


namespace sum_f_values_l590_590807

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_f_values : 
  (2 * (Finset.range 2017).sum (λ i, f (i + 2))) +
  (Finset.range 2017).sum (λ i, f ((i + 2)⁻¹)) +
  (Finset.range 2017).sum (λ i, (i + 2)⁻² * f (i + 2)) = 4032 := 
by
  sorry

end sum_f_values_l590_590807


namespace relationship_A_B_l590_590465

def A := { p : ℝ × ℝ | |p.1 + 1| + (p.2 - 2) ^ 2 = 0 }
def B := {-1, 0, 1, 2}

theorem relationship_A_B : ¬(A ⊆ B ∨ B ⊆ A ∨ A ∈ B) :=
by {
  sorry
}

end relationship_A_B_l590_590465


namespace parallel_and_intersect_at_one_point_l590_590563

open Real EuclideanGeometry

variables {A B C A1 B1 C1 A2 B2 C2 : Point}

/-- Define points and conditions -/
def InscribedCircleTouchesSides (A B C A1 B1 C1 : Point) : Prop :=
  InscribedCircleTouches BC A1 ∧ InscribedCircleTouches CA B1 ∧ InscribedCircleTouches AB C1

def SymmetricPoints (A B C A1 B1 C1 A2 B2 C2 : Point) : Prop :=
  SymmetricToAngleBisector A A1 A2 ∧ SymmetricToAngleBisector B B1 B2 ∧ SymmetricToAngleBisector C C1 C2

/-- Main theorem statement -/
theorem parallel_and_intersect_at_one_point
  (h1 : InscribedCircleTouchesSides A B C A1 B1 C1)
  (h2 : SymmetricPoints A B C A1 B1 C1 A2 B2 C2) :
  ParallelLines A2 B2 AB ∧ IntersectAtOnePoint AA2 BB2 CC2 :=
sorry

end parallel_and_intersect_at_one_point_l590_590563


namespace magnitude_of_c_eq_l590_590817

def a : Real × Real := (2, 4)
def b : Real × Real := (-1, 2)
def dot_product (v w : Real × Real) : Real := v.1 * w.1 + v.2 * w.2

def c (a b : Real × Real) : Real × Real :=
  let d_prod := dot_product a b
  (a.1 - d_prod * b.1, a.2 - d_prod * b.2)

def magnitude (v : Real × Real) : Real :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_c_eq : magnitude (c a b) = 8 * Real.sqrt 2 := by
  sorry

end magnitude_of_c_eq_l590_590817


namespace find_a_and_solution_set_solution_set_given_a_l590_590386

variable {x : ℝ}

theorem find_a_and_solution_set (a : ℝ)
(h1 : ∀ x, 1/2 < x ∧ x < 2 → ax^2 + 5x - 2 > 0) :
  a = -2 ∧ (∀ x, -3 < x ∧ x < 1/2 → -2 * x^2 - 5 * x + 3 > 0) :=
by
  split
  sorry
  
theorem solution_set_given_a (a : ℝ)
(h1 : a = -2)
(h2 : ∀ x, -3 < x ∧ x < 1/2 → -2 * x^2 - 5 * x + 3 > 0) :
  ∀ x, -3 < x ∧ x < 1/2 → ax^2 - 5x + a^2 - 1 > 0 :=
by
  rw h1
  exact h2
  sorry

end find_a_and_solution_set_solution_set_given_a_l590_590386


namespace a_n_formula_T_n_sum_l590_590050

noncomputable def a_n (n : ℕ) : ℕ :=
if n = 1 then 0 else n - 1

def S_n (n : ℕ) : ℕ := (Finset.range n).sum a_n

-- Theorem for the first part: Finding the general formula for {a_n}
theorem a_n_formula (n : ℕ) (h : n ≠ 0) : a_n (n + 1) = n := by
  sorry

noncomputable def T_n (n : ℕ) : ℝ :=
(Finset.range n).sum (λ k, (a_n (k + 1) + 1) / 2^(k + 1 : ℝ))

-- Theorem for the second part: Finding the sum T_n of the first n terms of the sequence { (a_n + 1) / 2^n }
theorem T_n_sum (n : ℕ) : T_n n = 2 - (n + 2) / 2^n := by
  sorry

end a_n_formula_T_n_sum_l590_590050


namespace regular_polygon_sides_l590_590750

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l590_590750


namespace sum_of_cubes_sixth_row_correct_l590_590503

noncomputable def sum_of_cubes_sixth_row : Nat :=
  15^3 + 20^3 + 15^3

theorem sum_of_cubes_sixth_row_correct :
  (∀ n, 
    (∑ k in finset.range n, n.choose k) - 2 = 2^(n-1) - 2) → 
  sum_of_cubes_sixth_row = 14750 :=
by sorry

end sum_of_cubes_sixth_row_correct_l590_590503


namespace domain_of_sqrt_function_l590_590308

def f (x : ℝ) : ℝ := Real.sqrt (-15 * x^2 + 14 * x + 8)

theorem domain_of_sqrt_function :
  {x : ℝ | -15 * x^2 + 14 * x + 8 ≥ 0} = set.Icc (-2 / 5 : ℝ) (4 / 3 : ℝ) :=
by
  sorry

end domain_of_sqrt_function_l590_590308


namespace inequality_proof_l590_590481

theorem inequality_proof (x1 x2 x3 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
  (x1^2 + x2^2 + x3^2)^3 / (x1^3 + x2^3 + x3^3)^2 ≤ 3 :=
sorry

end inequality_proof_l590_590481


namespace simplify_T_l590_590078

variable (x : ℝ)

theorem simplify_T :
  9 * (x + 2)^2 - 12 * (x + 2) + 4 = 4 * (1.5 * x + 2)^2 :=
by
  sorry

end simplify_T_l590_590078


namespace petya_time_l590_590689

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l590_590689


namespace closest_integer_to_cuberoot_150_l590_590595

theorem closest_integer_to_cuberoot_150 : 
  let cube5 := 5^3 in 
  let cube6 := 6^3 in 
  let midpoint := (cube5 + cube6) / 2 in 
  125 < 150 ∧ 150 < 216 ∧ 150 < midpoint → 
  5 = round (150^(1/3)) := 
by 
  intro h
  sorry

end closest_integer_to_cuberoot_150_l590_590595


namespace independence_equivalence_l590_590613

theorem independence_equivalence (A B : Set Ω) (P : Measure Ω) :
  (P(A|B) = P(A|Bᶜ) ∨ P(A) = P(A|B)) →
  P(A ∩ B) = P(A) * P(B) :=
  sorry

end independence_equivalence_l590_590613


namespace probability_mean_of_three_numbers_l590_590199

theorem probability_mean_of_three_numbers :
  let S := {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}
  let total_combinations := Nat.choose 10 3
  let valid_combinations := 20
  let p := valid_combinations / total_combinations
  (120 / p) = 720 := by
    let S := {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}
    let total_combinations := Nat.choose 10 3
    let valid_combinations := 20
    let p := valid_combinations / total_combinations
    show 120 / p = 720
    sorry

end probability_mean_of_three_numbers_l590_590199


namespace maximum_n_and_integers_l590_590644

theorem maximum_n_and_integers :
  ∃ (a : ℕ → ℤ) (n : ℕ), n = 7 ∧ ∀ k < n, a k > a (k + 1) ∧
  (∑ i in finset.range n, a i) = 840 ∧
  (∀ k < n - 1, a k = (1: ℚ) / ((k + 1) + 1) * (∑ i in finset.range n \ {k}, a i)) ∧
  list.of_fn (λ i, a i) = [280, 210, 168, 140, 120, 105, -183] := sorry

end maximum_n_and_integers_l590_590644


namespace total_area_shaded_triangles_l590_590214

theorem total_area_shaded_triangles (n : ℕ) (h : n = 6) : 
  let center := (3, 3) in
  let vertices := λ i, if i < n then (0, i) else if i < 2 * n then (i - n, n) else if i < 3 * n then (n, 3 * n - i) else (4 * n - i, 0) in
  -- The number of shaded triangles is 4 sides * n vertices per side - 4 vertices shared at corners, hence 4 * 6
  let num_triangles := 4 * n in
  -- Each triangle has an area of 1.5 (calculated from one example triangle)
  let triangle_area := 1.5 in
  -- The total area is the number of triangles times the area of one triangle
  total_area := num_triangles * triangle_area in
  -- The theorem states the total area should be 36
  total_area = 36 :=
by 
  sorry

end total_area_shaded_triangles_l590_590214


namespace problem_PQ_length_l590_590430

noncomputable def PQ_length (PR : ℕ) (sin_Q : ℝ) : ℝ :=
  PR * sin_Q

theorem problem_PQ_length :
  (∠PQR.P = 90) → (sin Q = 3 / 5) → (PR = 200) → PQ = 120 :=
by
  -- Lean Environment Setup
  intro P90 sinQ PR200

  -- Definitions from conditions
  let PR := 200
  let sin_Q := 3 / 5
  let PQ := PQ_length PR sin_Q

  -- Conclusion / Proof
  sorry

end problem_PQ_length_l590_590430


namespace probability_unique_tens_digits_l590_590549

theorem probability_unique_tens_digits :
  let num_ways := 10^6 in
  let total_combinations := Nat.choose 70 6 in
  (num_ways : ℚ) / total_combinations = 625 / 74440775 :=
by 
  sorry

end probability_unique_tens_digits_l590_590549


namespace unfair_dice_sum_impossible_l590_590313

theorem unfair_dice_sum_impossible :
  (∀ a_n b_n : (fin 6) → ℝ, 
  (∀ (n : fin 11), 2 ≤ n.val + 2 → 
    (2 / 33 : ℝ) < (s n.val a_n b_n) ∧ (s n.val a_n b_n) < (4 / 33)) → 
  false) :=
sorry

def s (n : ℕ) (a_n b_n : (fin 6) → ℝ) : ℝ :=
  match n with
  | 2  => a_n 0 * b_n 0
  | 3  => a_n 0 * b_n 1 + a_n 1 * b_n 0
  | 4  => a_n 0 * b_n 2 + a_n 1 * b_n 1 + a_n 2 * b_n 0
  | 5  => a_n 0 * b_n 3 + a_n 1 * b_n 2 + a_n 2 * b_n 1 + a_n 3 * b_n 0
  | 6  => a_n 0 * b_n 4 + a_n 1 * b_n 3 + a_n 2 * b_n 2 + a_n 3 * b_n 1 + a_n 4 * b_n 0
  | 7  => a_n 0 * b_n 5 + a_n 1 * b_n 4 + a_n 2 * b_n 3 + a_n 3 * b_n 2 + a_n 4 * b_n 1 + a_n 5 * b_n 0
  | 8  => a_n 1 * b_n 5 + a_n 2 * b_n 4 + a_n 3 * b_n 3 + a_n 4 * b_n 2 + a_n 5 * b_n 1
  | 9  => a_n 2 * b_n 5 + a_n 3 * b_n 4 + a_n 4 * b_n 3 + a_n 5 * b_n 2
  | 10 => a_n 3 * b_n 5 + a_n 4 * b_n 4 + a_n 5 * b_n 3
  | 11 => a_n 4 * b_n 5 + a_n 5 * b_n 4
  | 12 => a_n 5 * b_n 5
  | _ => 0  -- n should not be out of {2, ..., 12}

end unfair_dice_sum_impossible_l590_590313


namespace vector_a_magnitude_angle_cosine_l590_590818

noncomputable def vector_a : ℝ × ℝ × ℝ := (-4, 2, 4)
noncomputable def vector_b : ℝ × ℝ × ℝ := (-6, 3, -2)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem vector_a_magnitude : magnitude vector_a = 6 :=
  sorry

theorem angle_cosine : (dot_product vector_a vector_b) / (magnitude vector_a * magnitude vector_b) = 11 / 21 :=
  sorry

end vector_a_magnitude_angle_cosine_l590_590818


namespace find_coordinates_of_c_l590_590816

theorem find_coordinates_of_c 
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) (c : ℝ × ℝ × ℝ)
  (ha : a = (3, -4, 5)) (hb : b = (-1, 0, -2))
  (hc : c = (2 * (3, -4, 5).1 + 5 * (-1, 0, -2).1,
             2 * (3, -4, 5).2 + 5 * (-1, 0, -2).2,
             2 * (3, -4, 5).2 + 5 * (-1, 0, -2).2)) :
    c = (1, -8, 0) := 
  by
  sorry

end find_coordinates_of_c_l590_590816


namespace expected_lifetime_at_least_four_l590_590101

universe u

variables (α : Type u) [MeasurableSpace α] {𝒫 : ProbabilitySpace α}
variables {ξ η : α → ℝ} [IsFiniteExpectation ξ] [IsFiniteExpectation η]

noncomputable def max_lifetime : α → ℝ := λ ω, max (ξ ω) (η ω)

theorem expected_lifetime_at_least_four 
  (h : ∀ ω, max (ξ ω) (η ω) ≥ η ω)
  (h_eta : @Expectation α _ _ η  = 4) : 
  @Expectation α _ _ max_lifetime ≥ 4 :=
by
  sorry

end expected_lifetime_at_least_four_l590_590101


namespace james_total_earnings_l590_590856

-- Define the earnings for January
def januaryEarnings : ℕ := 4000

-- Define the earnings for February based on January
def februaryEarnings : ℕ := 2 * januaryEarnings

-- Define the earnings for March based on February
def marchEarnings : ℕ := februaryEarnings - 2000

-- Define the total earnings including January, February, and March
def totalEarnings : ℕ := januaryEarnings + februaryEarnings + marchEarnings

-- State the theorem: total earnings should be 18000
theorem james_total_earnings : totalEarnings = 18000 := by
  sorry

end james_total_earnings_l590_590856


namespace expected_lifetime_flashlight_l590_590092

noncomputable def xi : ℝ := sorry
noncomputable def eta : ℝ := sorry

def T : ℝ := max xi eta

axiom E_eta_eq_4 : E eta = 4

theorem expected_lifetime_flashlight : E T ≥ 4 :=
by
  -- The solution will go here
  sorry

end expected_lifetime_flashlight_l590_590092


namespace tan_of_angle_in_third_quadrant_l590_590352

theorem tan_of_angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α = -12 / 13) (h2 : π < α ∧ α < 3 * π / 2) : Real.tan α = 12 / 5 := 
sorry

end tan_of_angle_in_third_quadrant_l590_590352


namespace greatest_possible_x_l590_590975

theorem greatest_possible_x (x : ℕ) (h : x^2 < 10) : x ≤ 3 :=
  sorry

example : greatest_possible_x 3 (by norm_num) :=
  by norm_num

end greatest_possible_x_l590_590975


namespace calculate_expression_l590_590696

theorem calculate_expression :
  50 * 24.96 * 2.496 * 500 = (1248)^2 :=
by
  sorry

end calculate_expression_l590_590696


namespace evaluate_expression_l590_590756

theorem evaluate_expression : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3 / 3) := 
by
  sorry

end evaluate_expression_l590_590756


namespace complex_product_conjugate_l590_590779

theorem complex_product_conjugate (z : ℂ) (h : z = 1 + 3 * complex.I) : z * conj z = 10 := 
by sorry

end complex_product_conjugate_l590_590779


namespace second_field_full_rows_l590_590637

theorem second_field_full_rows 
    (rows_field1 : ℕ) (cobs_per_row : ℕ) (total_cobs : ℕ)
    (H1 : rows_field1 = 13)
    (H2 : cobs_per_row = 4)
    (H3 : total_cobs = 116) : 
    (total_cobs - rows_field1 * cobs_per_row) / cobs_per_row = 16 :=
by sorry

end second_field_full_rows_l590_590637


namespace petya_time_comparison_l590_590682

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l590_590682


namespace find_p_l590_590502

noncomputable def satisfies_condition (P F : ℝ × ℝ) (ε : ℝ → ℝ → Prop) :=
  ∀ (A B : ℝ × ℝ), (A.1^2 / 9 + A.2^2 = 1) ∧ (B.1^2 / 9 + B.2^2 = 1) ∧ (A ≠ B) ∧
                   ε A F ∧ ε B F → ∠ A P F = ∠ B P F

theorem find_p :
  ∃ p : ℝ, p > 0 ∧ satisfies_condition (p, 0) (2*ℝ.sqrt 2, 0) :=
  let P := (4*ℝ.sqrt 2, 0) 
  in (∃ p > 0, satisfies_condition (p, 0) (2*ℝ.sqrt 2, 0) (λ A F, A.2 = mx - 2*ℝ.sqrt 2*mx)) := 
  sorry

end find_p_l590_590502


namespace sqrt_eq_two_or_neg_two_l590_590182

theorem sqrt_eq_two_or_neg_two (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_eq_two_or_neg_two_l590_590182


namespace general_formula_an_l590_590363

theorem general_formula_an (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = 2 * a n + 1) : 
  a = (λ n, -2^(n-1)) :=
by
  sorry

end general_formula_an_l590_590363


namespace collinear_midpoints_l590_590865

variable {Point : Type}
variables (A B C D P O M N : Point)

-- Assume definitions and conditions
def convex_quadrilateral (A B C D : Point) : Prop := sorry
def interior_point (P : Point) (A B C D : Point) : Prop := sorry
def projections_on_circle (P : Point) (A B C D : Point) (O : Point) : Prop := sorry
def midpoint (X Y : Point) : Point := sorry
def collinear (X Y Z : Point) : Prop := sorry

-- Given conditions
variables (h1 : convex_quadrilateral A B C D)
variables (h2 : interior_point P A B C D)
variables (h3 : projections_on_circle P A B C D O)

-- Statement to prove
theorem collinear_midpoints (M : Point) (N : Point) (hM : M = midpoint A C) (hN : N = midpoint B D) :
  collinear M N O :=
sorry

end collinear_midpoints_l590_590865


namespace combinedHeightCorrect_l590_590316

def empireStateBuildingHeightToTopFloor : ℕ := 1250
def empireStateBuildingAntennaHeight : ℕ := 204

def willisTowerHeightToTopFloor : ℕ := 1450
def willisTowerAntennaHeight : ℕ := 280

def oneWorldTradeCenterHeightToTopFloor : ℕ := 1368
def oneWorldTradeCenterAntennaHeight : ℕ := 408

def totalHeightEmpireStateBuilding := empireStateBuildingHeightToTopFloor + empireStateBuildingAntennaHeight
def totalHeightWillisTower := willisTowerHeightToTopFloor + willisTowerAntennaHeight
def totalHeightOneWorldTradeCenter := oneWorldTradeCenterHeightToTopFloor + oneWorldTradeCenterAntennaHeight

def combinedHeight := totalHeightEmpireStateBuilding + totalHeightWillisTower + totalHeightOneWorldTradeCenter

theorem combinedHeightCorrect : combinedHeight = 4960 := by
  sorry

end combinedHeightCorrect_l590_590316


namespace ratio_of_radii_l590_590586

-- Define the mathematical problem and the necessary conditions
def two_circles_in_angle (α : ℝ) (r R : ℝ) (h : r < R) : Prop :=
  ∃ x : ℝ, let ratio := r / x in
           (ratio = (2 * (1 + real.cos α)) / (1 + real.sin α))

-- The theorem stating the problem
theorem ratio_of_radii {α r R : ℝ} (h : r < R) :
  two_circles_in_angle α r R h :=
begin
  sorry
end

end ratio_of_radii_l590_590586


namespace line_through_orthocenter_l590_590483

-- Defining the rectangle, point P on segment AB, and projection Q on DC
variables {A B C D P Q X Y : Type}

-- Given Conditions
def is_rectangle (A B C D : Type) : Prop := sorry -- a placeholder for the rectangle definition
def on_segment (P : Type) (AB : Type) : Prop := sorry -- P is on segment AB
def orthogonal_projection (P : Type) (DC Q : Type) : Prop := sorry -- Q is the orthogonal projection of P on line DC
def circumcircle (A B Q X Y : Type) : Prop := sorry -- Gamma is the circumcircle of triangle ABQ and intersects AD at X and BC at Y

-- Theorem
theorem line_through_orthocenter {A B C D P Q X Y: Type} 
  (h1 : is_rectangle A B C D)
  (h2 : on_segment P (segment A B))
  (h3 : orthogonal_projection P (line D C) Q)
  (h4 : circumcircle A B Q X Y):
  passes_through_orthocenter (line X Y) (triangle D C P) := 
sorry

end line_through_orthocenter_l590_590483


namespace sum_a_b_eq_34_over_3_l590_590823

theorem sum_a_b_eq_34_over_3 (a b: ℚ)
  (h1 : 2 * a + 5 * b = 43)
  (h2 : 8 * a + 2 * b = 50) :
  a + b = 34 / 3 :=
sorry

end sum_a_b_eq_34_over_3_l590_590823


namespace hyperbola_min_PF1_plus_PQ_l590_590624

theorem hyperbola_min_PF1_plus_PQ :
  ∀ (P Q F1 : Real × Real),
    (∃ x y, P = (x, y) ∧ (x^2 / 2) - y^2 = 1 ∧ x > 0) ∧
    (∃ m b, m ≠ 0 ∧ l = { (x, y) | y = m * x + b } ∧
             ∀ (x y : Real), (x, y) ∈ l → ( y = x / (sqrt 2) ∨ y = -x / (sqrt 2) )) ∧
    (∃ x y, Q = (x, y) ∧ ∃ m b, l = { (x, y) | y = m * x + b } ∧
             y = m * x + b ∧ m * (x - xP) + yP = y) ∧
    (∃ x, F1 = (-sqrt 3, 0)) →
    (dist P F1 + dist P Q) = 2 * sqrt 2 + 1 :=
begin
  intros P Q F1 h,
  cases h with h1 h2,
  cases h1 with x_yP hP,
  cases h2 with l_h2 l_h3,
  sorry
end

end hyperbola_min_PF1_plus_PQ_l590_590624


namespace Morse_code_distinct_symbols_l590_590020

theorem Morse_code_distinct_symbols : 
  ∑ n in ({1, 2, 3, 4, 5} : Finset ℕ), 2^n = 62 :=
by
  sorry

end Morse_code_distinct_symbols_l590_590020


namespace society_selection_l590_590655

theorem society_selection (n k : ℕ) (h_n : n = 20) (h_k : k = 3) :
  nat.choose n k = 1140 :=
by
  rw [h_n, h_k]
  exact nat.choose_succ_succ_eq 19 18 17
  -- This confirm the necessaries calculations. Here assuming nat.choose calculate
  sorry

end society_selection_l590_590655


namespace pancake_division_l590_590820

-- Define the problem statement in Lean 4.
theorem pancake_division (cuts : ℕ) (intersections : ℕ) : 
  cuts = 3 → intersections ∈ {0, 1, 2, 3} → 

  (intersections = 0 → ∃ parts : ℕ, parts = 4) ∧
  (intersections = 1 → ∃ parts : ℕ, parts = 5) ∧
  (intersections = 2 → ∃ parts : ℕ, parts = 6) ∧
  (intersections = 3 → ∃ parts : ℕ, parts = 7) :=
begin
  sorry
end

end pancake_division_l590_590820


namespace correct_control_setup_l590_590449

variable (medium selective_medium : Type)
variable (same_soil_sample_liquid sterile_water : medium)
variable (control_group_setup : medium → Prop)
variable (inoculate : medium → selective_medium → Prop)
variable (separation_effect_observed : selective_medium → Prop)

-- Conditions
axiom purpose_of_experiment : ∀ s : selective_medium, separation_effect_observed s
axiom option_A : control_group_setup same_soil_sample_liquid
axiom option_B : ¬control_group_setup sterile_water
axiom option_C : ¬∃ s : selective_medium, inoculate medium s
axiom option_D : ∀ s : selective_medium, ¬inoculate sterile_water s

-- Proof that the correct control setup is option A
theorem correct_control_setup : control_group_setup same_soil_sample_liquid :=
by
  sorry

end correct_control_setup_l590_590449


namespace range_of_a_for_monotonicity_l590_590011

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

def f (a : ℝ) (x : ℝ) : ℝ :=
  x - (1/3) * Real.sin (2 * x) + a * Real.sin x

theorem range_of_a_for_monotonicity :
  (∀ x : ℝ, is_monotonically_increasing (f a)) → -1/3 ≤ a ∧ a ≤ 1/3 :=
by
  sorry

end range_of_a_for_monotonicity_l590_590011


namespace actual_time_greater_than_planned_time_l590_590672

def planned_time (a V : ℝ) : ℝ := a / V

def actual_time (a V : ℝ) : ℝ := (a / (2.5 * V)) + (a / (1.6 * V))

theorem actual_time_greater_than_planned_time (a V : ℝ) (hV : V > 0) : 
  actual_time a V > planned_time a V :=
by 
  sorry

end actual_time_greater_than_planned_time_l590_590672


namespace smallest_number_to_add_l590_590591

theorem smallest_number_to_add:
  ∃ x : ℕ, x = 119 ∧ (2714 + x) % 169 = 0 :=
by
  sorry

end smallest_number_to_add_l590_590591


namespace productMultipleOfFiveProbability_l590_590460

open ProbabilityTheory

-- Define the event that the product of two rolls is a multiple of 5
def isMultipleOfFive (product : ℕ) : Prop := product % 5 = 0

-- Define the distribution of Juan's roll (a uniform random variable over {1, 2, 3, 4, 5, 6, 7, 8})
noncomputable def JuanRoll : PMF ℕ := PMF.uniform (Finset.range 8).map (λ x => x + 1)

-- Define the distribution of Amal's roll (a uniform random variable over {1, 2, 3, 4, 5, 6, 7})
noncomputable def AmalRoll : PMF ℕ := PMF.uniform (Finset.range 7).map (λ x => x + 1)

-- Define the combined distribution as the product of Juan's and Amal's rolls
noncomputable def combinedRolls : PMF (ℕ × ℕ) :=
  PMF.bind JuanRoll (λ j => PMF.bind AmalRoll (λ a => PMF.pure (j, a)))

-- Define the event that the product of the combined rolls is a multiple of 5
def eventMultipleOfFive : Set (ℕ × ℕ) := { p | isMultipleOfFive (p.fst * p.snd) }

-- The lean statement to prove: The probability of the event is 1/4
theorem productMultipleOfFiveProbability :
  (PMF.prob (combinedRolls) eventMultipleOfFive) = 1 / 4 :=
sorry

end productMultipleOfFiveProbability_l590_590460


namespace sqrt_eq_two_or_neg_two_l590_590181

theorem sqrt_eq_two_or_neg_two (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_eq_two_or_neg_two_l590_590181


namespace sum_reciprocals_of_squares_eq_fifty_over_forty_nine_l590_590993

theorem sum_reciprocals_of_squares_eq_fifty_over_forty_nine (a b : ℕ) (h : a * b = 7) :
  (1 / (a:ℚ)^2 + 1 / (b:ℚ)^2) = 50 / 49 :=
by {
  sorry
}

end sum_reciprocals_of_squares_eq_fifty_over_forty_nine_l590_590993


namespace standard_eqn_of_ellipse_range_op_ab_l590_590788

section ellipse_problem

variables {a b c : ℝ}
variables (P : ℝ × ℝ) 

-- Conditions and variables
def ellipse (a b : ℝ) : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1
def foci (a c : ℝ) : ℕ → ℝ × ℝ 
| 0 := (-c, 0)
| 1 := (c, 0)

-- Part (1): Standard equation of the ellipse
theorem standard_eqn_of_ellipse : 
    (∀ A B : ℝ × ℝ, ellipse a b A ∧ ellipse a b B → 4 * a = 8 * real.sqrt 2 ∧ (c / a = real.sqrt 2 / 2) ∧ a = 2 * real.sqrt 2 ∧ c = 2 ∧ b = 2 ∧ a > b ∧ b > 0) → (a = 2 * real.sqrt 2 ∧ b = 2) :=
begin
  sorry
end

-- Part (2): Range of |OP|^2 / |AB|
theorem range_op_ab :
  (∀ A B : ℝ × ℝ, ellipse a b A ∧ ellipse a b B → (a = 2 * real.sqrt 2 ∧ b = 2) ∧ (∀ P, P ∈ ellipse a b → (P.1 * P.1 + P.2 * P.2 = (8 * (1 + (1 + P.2*P.2) / (P.1*P.1 + 2))) / ((1 + P.2*P.2) / (P.1*P.1 + 2))))) ∧ ∃ O x y, x = λ k:ℝ, (P.1 * P.1 + P.2 * P.2) = -k * (P.1 / P.2) → (real.sqrt 2 ) / 2 < (P.1 * P.1 + P.2 * P.2) / 2  ∧ (P.1 * P.1 + P.2 * P.2) / 2  ≤ 2 * real.sqrt 2 :=
begin
  sorry
end

end ellipse_problem

end standard_eqn_of_ellipse_range_op_ab_l590_590788


namespace arc_length_of_curve_l590_590996

-- Define the conditions
def rho (φ : ℝ) : ℝ := 5 * φ
axiom varphi_range : 0 ≤ (φ : ℝ) ∧ φ ≤ (12 / 5)

-- Define the theorem to be proven
theorem arc_length_of_curve :
  (∫ φ in 0..(12/5), Real.sqrt (25 * φ^2 + 25)) = 78 + (5 / 2) * Real.log 5 :=
by sorry

end arc_length_of_curve_l590_590996


namespace find_BE_l590_590853

-- Definitions from the conditions
variable {A B C D E : Point}
variable (AB BC CA BD BE CE : ℝ)
variable (angleBAE angleCAD : Real.Angle)

-- Given conditions
axiom h1 : AB = 12
axiom h2 : BC = 17
axiom h3 : CA = 15
axiom h4 : BD = 7
axiom h5 : angleBAE = angleCAD

-- Required proof statement
theorem find_BE :
  BE = 1632 / 201 := by
  sorry

end find_BE_l590_590853


namespace four_tuple_lcm_l590_590327

theorem four_tuple_lcm (m n : ℕ) : f (m, n) = (6 * m^2 + 3 * m + 1) * (6 * n^2 + 3 * n + 1) :=
  sorry

end four_tuple_lcm_l590_590327


namespace integral_f_l590_590394

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2
  else if 1 < x ∧ x ≤ 2 then 2 - x
  else 0

theorem integral_f : ∫ x in 0..2, f x = 5 / 6 := by
  sorry

end integral_f_l590_590394


namespace logarithmic_expression_evaluation_l590_590312

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem logarithmic_expression_evaluation : 
  log_base_10 (5 / 2) + 2 * log_base_10 2 - (1/2)⁻¹ = -1 := 
by 
  sorry

end logarithmic_expression_evaluation_l590_590312


namespace solve_eq1_solve_eq2_l590_590518

-- Define the first equation
def eq1 (x : ℝ) : Prop := x^2 - 2 * x - 1 = 0

-- Define the second equation
def eq2 (x : ℝ) : Prop := (x - 2)^2 = 2 * x - 4

-- State the first theorem
theorem solve_eq1 (x : ℝ) : eq1 x ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by sorry

-- State the second theorem
theorem solve_eq2 (x : ℝ) : eq2 x ↔ (x = 2 ∨ x = 4) :=
by sorry

end solve_eq1_solve_eq2_l590_590518


namespace sqrt_four_eq_two_or_neg_two_l590_590179

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 :=
by 
  sorry

end sqrt_four_eq_two_or_neg_two_l590_590179


namespace largest_whole_number_l590_590760

theorem largest_whole_number :
  ∃ x : ℕ, 9 * x - 8 < 130 ∧ (∀ y : ℕ, 9 * y - 8 < 130 → y ≤ x) ∧ x = 15 :=
sorry

end largest_whole_number_l590_590760


namespace pencil_cost_l590_590494

noncomputable def cost_per_pencil : ℝ :=
let
  notebooks := 3 * 0.95,
  rulers := 0.80,
  erasers := 7 * 0.25,
  highlighters := 6 * 0.45,
  stickynotes := 2 * 1.10,
  discount_notebooks := 0.20 * notebooks,
  discount_erasers := 0.10 * erasers,
  discounted_notebooks := notebooks - discount_notebooks,
  discounted_erasers := erasers - discount_erasers,
  taxed_notebooks := 1.05 * discounted_notebooks,
  taxed_rulers := 1.08 * rulers,
  taxed_erasers := 1.05 * discounted_erasers,
  taxed_highlighters := 1.08 * highlighters,
  taxed_stickynotes := 1.08 * stickynotes,
  total_without_pencils := taxed_notebooks + taxed_rulers + taxed_erasers + taxed_highlighters + taxed_stickynotes,
  remaining := 15.50 - total_without_pencils
in
remaining / 5

theorem pencil_cost : cost_per_pencil == 1.06 :=
sorry

end pencil_cost_l590_590494


namespace range_of_omega_l590_590395

theorem range_of_omega (ω : ℝ) (hω : 0 < ω) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 * Real.pi → sin (ω * x) = 0 ↔ 0 ≤ x ∧ x ≤ 3 * Real.pi ∧ Real.floor ((ω * x) / (Real.pi)) % 2 = 0) →
  (4 / 3 ≤ ω ∧ ω < 5 / 3) :=
begin
  sorry
end

end range_of_omega_l590_590395


namespace even_two_digit_numbers_l590_590346

theorem even_two_digit_numbers (s : Finset ℕ) (h : s = {0, 1, 2, 3, 4}) :
  ∃ n, n = 10 ∧
    ∃ l (hl : l ⊆ s) (hl_size : l.card = 2), 
      ∀ x ∈ l, ∃ d1 d2, d1 ≠ d2 ∧ (10 * d1 + d2) % 2 = 0 ∧ (d1 ∈ l ∧ d2 ∈ l) :=
by
  sorry

end even_two_digit_numbers_l590_590346


namespace absolute_value_inequality_perodic_sequence_l590_590667

theorem absolute_value_inequality_perodic_sequence :
  (∀ n : ℕ, a n = a (n + 100)) →    -- periodicity
  a 1 ≥ 0 →                         -- condition 1
  a 1 + a 2 ≤ 0 →                   -- condition 2
  (∀ n : ℕ, if odd n then ∑ i in finset.range (n + 1), a i ≥ 0 else ∑ i in finset.range (n + 1), a i ≤ 0) →    -- condition 3
  abs (a 99) ≥ abs (a 100) :=       -- conclusion
  sorry

end absolute_value_inequality_perodic_sequence_l590_590667


namespace perp_ED_BC_l590_590466

-- Definitions to be used in the problem statement
variable {A B C P E D : Type}
variable (triangle : Triangle A B C)
variable (Omega : Circumcircle triangle)
variable (tangent_point_P : OnTangent A Omega P)
variable (projection_E : OrthogonalProjection P A C E)
variable (projection_D : OrthogonalProjection P A B D)

theorem perp_ED_BC :
  Perpendicular (Line E D) (Line B C) :=
sorry

end perp_ED_BC_l590_590466


namespace dice_probability_correct_l590_590207

noncomputable def probability_at_least_one_two_or_three : ℚ :=
  let total_outcomes := 64
  let favorable_outcomes := 64 - 36
  favorable_outcomes / total_outcomes

theorem dice_probability_correct :
  probability_at_least_one_two_or_three = 7 / 16 :=
by
  -- Proof will be provided here
  sorry

end dice_probability_correct_l590_590207


namespace probability_six_integers_unique_tens_digit_l590_590540

theorem probability_six_integers_unique_tens_digit :
  (∃ (x1 x2 x3 x4 x5 x6 : ℕ),
    10 ≤ x1 ∧ x1 ≤ 79 ∧
    10 ≤ x2 ∧ x2 ≤ 79 ∧
    10 ≤ x3 ∧ x3 ≤ 79 ∧
    10 ≤ x4 ∧ x4 ≤ 79 ∧
    10 ≤ x5 ∧ x5 ≤ 79 ∧
    10 ≤ x6 ∧ x6 ≤ 79 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x1 ≠ x6 ∧
    x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x2 ≠ x6 ∧
    x3 ≠ x4 ∧ x3 ≠ x5 ∧ x3 ≠ x6 ∧
    x4 ≠ x5 ∧ x4 ≠ x6 ∧
    x5 ≠ x6 ∧
    tens_digit x1 ≠ tens_digit x2 ∧
    tens_digit x1 ≠ tens_digit x3 ∧
    tens_digit x1 ≠ tens_digit x4 ∧
    tens_digit x1 ≠ tens_digit x5 ∧
    tens_digit x1 ≠ tens_digit x6 ∧
    tens_digit x2 ≠ tens_digit x3 ∧
    tens_digit x2 ≠ tens_digit x4 ∧
    tens_digit x2 ≠ tens_digit x5 ∧
    tens_digit x2 ≠ tens_digit x6 ∧
    tens_digit x3 ≠ tens_digit x4 ∧
    tens_digit x3 ≠ tens_digit x5 ∧
    tens_digit x3 ≠ tens_digit x6 ∧
    tens_digit x4 ≠ tens_digit x5 ∧
    tens_digit x4 ≠ tens_digit x6 ∧
    tens_digit x5 ≠ tens_digit x6)
    →
  (probability := \(\frac{4375}{744407}\)).sorry

end probability_six_integers_unique_tens_digit_l590_590540


namespace sin_symmetry_value_l590_590013

theorem sin_symmetry_value (ϕ : ℝ) (hϕ₀ : 0 < ϕ) (hϕ₁ : ϕ < π / 2) :
  ϕ = 5 * π / 12 :=
sorry

end sin_symmetry_value_l590_590013


namespace irreducible_polynomials_count_l590_590864

noncomputable def number_of_irreducible_polynomials (p : ℕ) : ℕ :=
  2 * (Nat.choose ((p - 1) / 2) 2)

theorem irreducible_polynomials_count (p : ℕ) (h_prime : Nat.prime p) (h_ge_five : p ≥ 5) :
  (number_of_irreducible_polynomials p) = 2 * Nat.choose ((p - 1) / 2) 2 := by
  sorry

end irreducible_polynomials_count_l590_590864


namespace determine_domain_l590_590805

noncomputable def f (x : ℝ) : ℝ := log 2 (x-1) + log 2 (2*x + 1)

theorem determine_domain :
  { x : ℝ | x - 1 > 0 ∧ 2 * x + 1 > 0 ∧ f x ≤ 1 } = (1, 3/2] :=
by sorry

end determine_domain_l590_590805


namespace a_n_formula_T_n_sum_l590_590049

noncomputable def a_n (n : ℕ) : ℕ :=
if n = 1 then 0 else n - 1

def S_n (n : ℕ) : ℕ := (Finset.range n).sum a_n

-- Theorem for the first part: Finding the general formula for {a_n}
theorem a_n_formula (n : ℕ) (h : n ≠ 0) : a_n (n + 1) = n := by
  sorry

noncomputable def T_n (n : ℕ) : ℝ :=
(Finset.range n).sum (λ k, (a_n (k + 1) + 1) / 2^(k + 1 : ℝ))

-- Theorem for the second part: Finding the sum T_n of the first n terms of the sequence { (a_n + 1) / 2^n }
theorem T_n_sum (n : ℕ) : T_n n = 2 - (n + 2) / 2^n := by
  sorry

end a_n_formula_T_n_sum_l590_590049


namespace geometric_sequence_sum_l590_590359

noncomputable def S_n (a_1 : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a_1 * (1 - r ^ n) / (1 - r)

noncomputable def a (a_1 : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a_1 * r ^ (n - 1)

theorem geometric_sequence_sum (a_1 : ℝ) (r : ℝ) (a_2 a_5 a_3 : ℝ) (h1 : r = -1/2) 
  (h2 : S_n a_1 r 4 = a a_1 r 2 * a a_1 r 2 + 2 * a a_1 r 5) :
  a a_1 r 3 = 1/2 :=
by
  have h_seq := a a_1 r
  have h_3 := h_seq 3 sorry
  sorry

end geometric_sequence_sum_l590_590359


namespace coin_placement_count_l590_590132

theorem coin_placement_count :
  let board := (2, 100)
      no_adjacent (pos1 pos2 : (ℕ × ℕ)) : Prop := (abs (pos1.1 - pos2.1) + abs (pos1.2 - pos2.2)) ≠ 1
      position_valid (pos : (ℕ × ℕ)) := pos.1 < board.1 ∧ pos.2 < board.2
      coin_placements := {positions : set (ℕ × ℕ) // positions.card = 99 ∧ ∀ pos1 pos2 ∈ positions, no_adjacent pos1 pos2}
  in coin_placements.card = 396 :=
by sorry

end coin_placement_count_l590_590132


namespace catches_difference_is_sixteen_l590_590457

noncomputable def joe_catches : ℕ := 23
noncomputable def derek_catches : ℕ := 2 * joe_catches - 4
noncomputable def tammy_catches : ℕ := 30
noncomputable def one_third_derek : ℕ := derek_catches / 3
noncomputable def difference : ℕ := tammy_catches - one_third_derek

theorem catches_difference_is_sixteen :
  difference = 16 := 
by
  sorry

end catches_difference_is_sixteen_l590_590457


namespace regular_polygon_sides_l590_590737

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l590_590737


namespace similar_triangle_and_ratio_l590_590031

structure Triangle (α : Type) :=
(A B C : α)

def midpoint (α : Type) [AddGroup α] [Module ℚ α] (P Q : α) : α := (P + Q) / 2

def arc_midpoint (α : Type) [AddGroup α] [Module ℚ α] (P Q : α) : α := -- definition for arc midpoint

def circle_diameter (α : Type) (P Q : α) : α := -- definition for the circle diameter

structure Circle (α : Type) :=
(center : α)
(radius : ℚ)

def circle_of_diameter (α : Type) (P Q : α) : Circle α := {
  center := midpoint α P Q,
  radius := -- definition for radius of the circle with diameter PQ
}

def external_tangent (α : Type) [Field α] (w1 w2: Circle α) : α := -- definition for the common external tangent

theorem similar_triangle_and_ratio {α : Type} [Field α] [AddGroup α] [Module ℚ α] 
  (A B C : α)
  (Ma := midpoint _ B C) (Mb := midpoint _ C A) (Mc := midpoint _ A B)
  (Ta := arc_midpoint _ B C) (Tb := arc_midpoint _ C A) (Tc := arc_midpoint _ A B)
  (wa := circle_of_diameter _ Ma Ta) (wb := circle_of_diameter _ Mb Tb) (wc := circle_of_diameter _ Mc Tc)
  (pa := external_tangent _ wb wc) (pb := external_tangent _ wc wa) (pc := external_tangent _ wa wb) :
  similar_triangle pa pb pc A B C ∧ ratio pa pb pc A B C = 1 / 2 :=
sorry

end similar_triangle_and_ratio_l590_590031


namespace find_extremes_cos_sin_cos_prod_l590_590778

noncomputable def cos_sin_cos_prod (x y z : ℝ) : ℝ :=
  cos x * sin y * cos z

theorem find_extremes_cos_sin_cos_prod (x y z : ℝ) :
    x ≥ y → y ≥ z → z ≥ π / 12 → 
    x + y + z = π / 2 →
    (∃ x y z, cos_sin_cos_prod x y z = 1 / 8 ∧ 
              cos_sin_cos_prod x y z = (2 + sqrt 3) / 8) :=
  by
  sorry

end find_extremes_cos_sin_cos_prod_l590_590778


namespace expected_lifetime_flashlight_l590_590098

noncomputable theory

variables (ξ η : ℝ) -- ξ and η are continuous random variables representing the lifetimes
variables (T : ℝ) -- T is the lifetime of the flashlight

-- Define the maximum lifetime of the flashlight
def max_lifetime (ξ η : ℝ) : ℝ := max ξ η

-- Given condition: the expectation of η is 4
axiom expectation_eta : E η = 4

-- Theorem statement: expected lifetime of the flashlight is at least 4
theorem expected_lifetime_flashlight (ξ η : ℝ) (h : T = max_lifetime ξ η) : 
  E (max_lifetime ξ η) ≥ 4 :=
by 
  sorry

end expected_lifetime_flashlight_l590_590098


namespace sqrt_of_4_l590_590177

theorem sqrt_of_4 (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_of_4_l590_590177


namespace digital_black_hole_l590_590819
open Nat

def count_even_digits (n : ℕ) : ℕ := (digits 10 n).count (λ d, d % 2 = 0)
def count_odd_digits (n : ℕ) : ℕ := (digits 10 n).count (λ d, d % 2 = 1)
def total_digits (n : ℕ) : ℕ := (digits 10 n).length
def black_hole_step (n : ℕ) : ℕ := 100 * count_even_digits n + 10 * count_odd_digits n + total_digits n

theorem digital_black_hole : ∀ n : ℕ, ∃ k : ℕ, (∀ m : ℕ, black_hole_step^[k + 1] n = black_hole_step^[k] n) ∧ black_hole_step^[k] n = 123 :=
by
  sorry

end digital_black_hole_l590_590819


namespace regular_polygon_sides_l590_590730

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l590_590730


namespace number_of_true_statements_l590_590794

open Set

variable {m n : ℝ^3 → Prop} -- m and n are two non-coincident lines
variable {α β : Set (ℝ^3 → Prop)} -- α and β are two non-coincident planes

def is_perpendicular (l : ℝ^3 → Prop) (p : Set (ℝ^3 → Prop)) : Prop :=
  ∃ (v : ℝ^3), v ∈ l ∧ ∀ (u : ℝ^3), u ∈ p → v ⬝ u = 0 -- Defining perpendicularity

def is_parallel (l : ℝ^3 → Prop) (p : Set (ℝ^3 → Prop)) : Prop :=
  ∃ (v : ℝ^3), v ∈ l ∧ ∀ (u : ℝ^3), u ∈ p → v ⬝ u = v ⬝ v -- Defining parallelism

def subset_of (l : ℝ^3 → Prop) (p : Set (ℝ^3 → Prop)) : Prop :=
  ∀ (x : ℝ^3), x ∈ l → x ∈ p -- Defining subset relation

theorem number_of_true_statements :
  let s1 := if (is_perpendicular α β ∧ is_perpendicular m α) then is_parallel m β else false,
      s2 := if (is_perpendicular m α ∧ is_perpendicular m β) then is_parallel α β else false,
      s3 := if (is_parallel m α ∧ is_perpendicular n α) then is_perpendicular m n else false,
      s4 := if (is_parallel m α ∧ subset_of m β) then is_parallel α β else false
  in 2 = (cond s1 1 0 + cond s2 1 0 + cond s3 1 0 + cond s4 1 0)
:= by sorry

end number_of_true_statements_l590_590794


namespace Louisa_travel_distance_l590_590498

variables (D : ℕ)

theorem Louisa_travel_distance : 
  (200 / 50 + 3 = D / 50) → D = 350 :=
by
  intros h
  sorry

end Louisa_travel_distance_l590_590498


namespace ip_eq_iq_l590_590113

variable {A B C I K L P Q : Type} [linear_ordered_field A]
variable {triangle ABC : A}
variable {incenter_I : A}
variable (K L : segment BC := (BC : Type))
variable (P Q : point := fun (A B K L : segment BC := (BC : Type)))
variable {ABK : Type}
variable {ABL : Type}
variable {ACK : Type}
variable {ACL : Type}

theorem ip_eq_iq
  (h : is_incenter triangle ABC incenter_I)
  (hK_L_Chosen : chosen_on_segment K L BC)
  (hTangent_P : tangent_incircles ABK ABL P)
  (hTangent_Q : tangent_incircles ACK ACL Q)
  : dist incenter_I P = dist incenter_I Q :=
sorry

end ip_eq_iq_l590_590113


namespace f_2017_equals_2_pow_1007_l590_590397

noncomputable def f : ℝ → ℝ
| x := if 0 ≤ x ∧ x ≤ 2 then sin (π * x / 6) else 2 * f (x - 2)

theorem f_2017_equals_2_pow_1007 : f 2017 = 2 ^ 1007 := 
sorry

end f_2017_equals_2_pow_1007_l590_590397


namespace solve_a_l590_590372

theorem solve_a (a : ℝ) (h1 : a > 0) (h2 : (λ x : ℝ, 3 * x^2 + 12) ((λ x : ℝ, x^2 - 6) a) = 12) : a = Real.sqrt 6 := 
sorry

end solve_a_l590_590372


namespace total_pennies_l590_590500

theorem total_pennies (rachelle_pennies : ℕ) (gretchen_pennies : ℕ) (rocky_pennies : ℕ)
  (h1 : rachelle_pennies = 180)
  (h2 : gretchen_pennies = rachelle_pennies / 2)
  (h3 : rocky_pennies = gretchen_pennies / 3) :
  rachelle_pennies + gretchen_pennies + rocky_pennies = 300 :=
by
  sorry

end total_pennies_l590_590500


namespace two_times_x_equals_two_l590_590421

theorem two_times_x_equals_two (x : ℝ) (h : x = 1) : 2 * x = 2 := by
  sorry

end two_times_x_equals_two_l590_590421


namespace equation_of_locus_C_perpendicular_vectors_and_distance_l590_590848

open Real

theorem equation_of_locus_C (P : ℝ × ℝ)
    (h1 : dist P (0, -√3) + dist P (0, √3) = 4) :
    P.1 ^ 2 + (P.2 ^ 2) / 4 = 1 :=
    sorry

theorem perpendicular_vectors_and_distance (k : ℝ)
    (h1 : y = k * x + 1)
    (h2 : ∀ A B : ℝ × ℝ, (A ∈ C) → (B ∈ C) →
        let \overrightarrow OA := (A.1 - 0, A.2 - 0)
        let \overrightarrow OB := (B.1 - 0, B.2 - 0)
        (dot_product (\overrightarrow OA) (\overrightarrow OB) = 0) ↔ k = ±1/2)
    (h3 : \|\overrightarrow {AB}\| = sqrt (\overrightarrow {AB}.1^2 + \overrightarrow {AB}.2^2)) :
    \| \overrightarrow {AB}\| = (4* sqrt(65)) / 17 :=
    sorry


end equation_of_locus_C_perpendicular_vectors_and_distance_l590_590848


namespace grasshopper_unoccupied_cells_l590_590724

-- Define the board size and initial conditions
def board_size := 6
def initial_grasshoppers := board_size * board_size

-- Define the condition of the grasshopper jumps
def diagonal_jump (x y : ℕ) : ℕ × ℕ := 
  if (x % 2 = y % 2) then ((x + 2) % board_size, (y + 2) % board_size) 
  else ((x + 4) % board_size, (y + 4) % board_size)

-- The main theorem to prove
theorem grasshopper_unoccupied_cells :
  ∀ (board : array (array ℕ board_size) board_size), 
    (∀ (i j : ℕ), i < board_size → j < board_size → board[i][j] = 1) →
    ∃ (u ≥ 12),  ∀  (u, v : ℕ) →  board_size →  board_size  board[u][v] = 0 := sorry

end grasshopper_unoccupied_cells_l590_590724


namespace distance_between_mountains_on_map_l590_590497

noncomputable def map_distance (actual_distance_km : ℚ) (map_dist_to_base_inches : ℚ) (actual_dist_to_base_km : ℚ) : ℚ :=
  (actual_distance_km * map_dist_to_base_inches) / actual_dist_to_base_km

theorem distance_between_mountains_on_map :
  map_distance 136 25 10.897435897435898 ≈ 311.965812 :=
by
  -- Use exact or appropriate Lean method to confirm the proof
  sorry

end distance_between_mountains_on_map_l590_590497


namespace find_intersection_point_l590_590639

/-- Definition of the parabola -/
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 - 4 * y + 7

/-- Condition for intersection at exactly one point -/
def discriminant (m : ℝ) : ℝ := 4 ^ 2 - 4 * 3 * (m - 7)

/-- Main theorem stating the proof problem -/
theorem find_intersection_point (m : ℝ) :
  (discriminant m = 0) → m = 25 / 3 :=
by
  sorry

end find_intersection_point_l590_590639


namespace greater_segment_difference_l590_590160

theorem greater_segment_difference :
  ∀ (L1 L2 : ℝ), L1 = 7 ∧ L1^2 - L2^2 = 32 → L1 - L2 = 7 - Real.sqrt 17 :=
by
  intros L1 L2 h
  sorry

end greater_segment_difference_l590_590160


namespace last_number_in_sequence_l590_590843

def sequence (a : ℕ → ℤ) := 
a 1 = 1 ∧ 
(∀ n : ℕ, 2 ≤ n ∧ n ≤ 1998 → a n = a (n - 1) + a (n + 1))

theorem last_number_in_sequence {a : ℕ → ℤ} 
  (h : sequence a) : 
  a 1999 = 1 :=
sorry

end last_number_in_sequence_l590_590843


namespace jellybean_total_l590_590250

theorem jellybean_total 
    (blackBeans : ℕ)
    (greenBeans : ℕ)
    (orangeBeans : ℕ)
    (h1 : blackBeans = 8)
    (h2 : greenBeans = blackBeans + 2)
    (h3 : orangeBeans = greenBeans - 1) :
    blackBeans + greenBeans + orangeBeans = 27 :=
by
    -- The proof will be placed here
    sorry

end jellybean_total_l590_590250


namespace range_of_a_l590_590929

noncomputable def f (a x : ℝ) : ℝ := log a (x - 2 * a / x)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x ∧ x < 2 → f a x < f a (x + 0.01)) → 0 < a ∧ a ≤ 1 / 2 := 
by 
  sorry

end range_of_a_l590_590929


namespace tetrahedron_volume_l590_590056

variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Given conditions as definitions in Lean 4
def AB : ℝ := 1
def CD : ℝ := Real.sqrt 3
def distance_AB_CD : ℝ := 2
def angle_AB_CD : ℝ := Real.pi / 3

theorem tetrahedron_volume 
  (AB_length : AB = 1)
  (CD_length : CD = Real.sqrt 3)
  (distance_lines : distance_AB_CD = 2)
  (angle_lines : angle_AB_CD = Real.pi / 3) :
  ∃ V : ℝ, V = 1 / 2 :=
by
  sorry

end tetrahedron_volume_l590_590056


namespace prove_g_l590_590623

noncomputable def g (t : ℝ) : ℝ :=
  let α := (-t + real.sqrt (t^2 + 1)) / 2
  let β := (-t - real.sqrt (t^2 + 1)) / 2
  max (λ x, (2 * x - t) / (x^2 + 1)) β - min (λ x, (2 * x - t) / (x^2 + 1)) α

theorem prove_g (t : ℝ) : g t = (8 * real.sqrt (t^2 + 1) * (2 * t^2 + 5)) / (16 * t^2 + 25) :=
  sorry

end prove_g_l590_590623


namespace part_a_part_b_l590_590434

-- Define the conditions of the problem
def participants := 12
def max_points_per_participant := 11
def master_title_threshold := 7.7
def total_games := (participants * (participants - 1)) / 2

-- Define the questions as propositions to be proven
def can_seven_earn_title : Prop :=
  ∃(pts : ℕ), pts = 8 ∧ pts > master_title_threshold

def cannot_eight_earn_title : Prop :=
  ∀(pts : ℕ), pts ≥ 8 → ¬(8 * pts ≤ total_games * 1 + ((8 * (8 - 1)) / 2) * 0.5)

-- State theorems without proofs
theorem part_a : can_seven_earn_title := by sorry
theorem part_b : cannot_eight_earn_title := by sorry

end part_a_part_b_l590_590434


namespace inequality_f_neg1_gt_f_neg3_l590_590938

variable {f : ℝ → ℝ}

-- The conditions from part a)
-- Condition: f is an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Icc (-10 : ℝ) (10 : ℝ), f x = f (-x)

-- Condition: f(3) < f(1)
def condition_f3_lt_f1 (f : ℝ → ℝ) : Prop := f 3 < f 1

-- The question is to prove that f(-1) > f(-3)
theorem inequality_f_neg1_gt_f_neg3 
  (h_even : even_function f) 
  (h_inequality : condition_f3_lt_f1 f) : 
  f (-1) > f (-3) :=
sorry

end inequality_f_neg1_gt_f_neg3_l590_590938


namespace sin_product_identity_l590_590709

theorem sin_product_identity :
  sin (6 * π / 180) * sin (42 * π / 180) * sin (66 * π / 180) * sin (78 * π / 180) = 1 / 16 := 
by
  sorry

end sin_product_identity_l590_590709


namespace expected_lifetime_flashlight_l590_590097

noncomputable theory

variables (ξ η : ℝ) -- ξ and η are continuous random variables representing the lifetimes
variables (T : ℝ) -- T is the lifetime of the flashlight

-- Define the maximum lifetime of the flashlight
def max_lifetime (ξ η : ℝ) : ℝ := max ξ η

-- Given condition: the expectation of η is 4
axiom expectation_eta : E η = 4

-- Theorem statement: expected lifetime of the flashlight is at least 4
theorem expected_lifetime_flashlight (ξ η : ℝ) (h : T = max_lifetime ξ η) : 
  E (max_lifetime ξ η) ≥ 4 :=
by 
  sorry

end expected_lifetime_flashlight_l590_590097


namespace compute_expression_l590_590711

open Nat

-- Definitions of binomial coefficient and power functions
def binom (a : ℚ) (b : ℕ) : ℚ := a * (a - 1) * ... * (a - ↑b + 1) / (b!)

def pow (a : ℚ) (b : ℕ) : ℚ := a^b

theorem compute_expression : 
  (binom (1/2) 2017 * pow 4 2017) / binom 4034 2017 = -1 / 4033 :=
by
  sorry

end compute_expression_l590_590711


namespace number_of_real_solutions_l590_590942

theorem number_of_real_solutions :
  ∃! x : ℝ, (2 ^ (4 * x + 2)) * (4 ^ (2 * x + 4)) = 8 ^ (3 * x + 4) := 
by
  sorry

end number_of_real_solutions_l590_590942


namespace ratio_of_x_intercepts_l590_590208

variable (b : ℝ) (hb : b ≠ 0)

def x_intercept_s : ℝ := -b / 12
def x_intercept_t : ℝ := -b / 8

theorem ratio_of_x_intercepts (hs : x_intercept_s = -b / 12) (ht : x_intercept_t = -b / 8) : 
  x_intercept_s / x_intercept_t = (2 : ℝ) / (3 : ℝ) :=
by
  sorry

end ratio_of_x_intercepts_l590_590208


namespace angle_bfc_l590_590442

def isosceles_triangle (A B C : Type) [IsTriangle A B C] :=
  isoceles A C B

variables {ABC : Triangle}
variables (A B C D E F : Point)
variables (circ : Circle)
variables (ac_eq_bc : AC = BC)
variables (angle_acb_40 : ∠ ACB = 40)
variables (circle_diam_bc : IsDiameter circ BC)
variables (circle_intersect_ac : Intersect circ AC D)
variables (circle_intersect_ab : Intersect circ AB E)
variables (quad_diags_intersect : Intersect Diagonals ABC circ F)

theorem angle_bfc (h1 : Isosceles ∆ABC AC BC)
                   (h2 : Circle Diameter BC)
                   (h3 : Circle.Inter AC D)
                   (h4 : Circle.Inter AB E)
                   (h5 : Diagonals.Intersect BCDE F)
                   (h6 : ∠ACB = 40) : ∠BFC = 110 :=
  sorry

end angle_bfc_l590_590442


namespace number_of_elements_in_sequence_l590_590413

theorem number_of_elements_in_sequence :
  ∀ (a₀ d : ℕ) (n : ℕ), 
  a₀ = 4 →
  d = 2 →
  n = 64 →
  (a₀ + (n - 1) * d = 130) →
  n = 64 := 
by
  -- We will skip the proof steps as indicated
  sorry

end number_of_elements_in_sequence_l590_590413


namespace arrangement_of_students_l590_590245

theorem arrangement_of_students:
  let students : Fin 6 := {0, 1, 2, 3, 4, 5}
  let communities : Fin 3 := {A, B, C}
  let possible_arrangements := 
    {arrangement : students → communities // 
      (∀ s, arrangement s ∈ communities) ∧
      ((arrangement 0 = A ∧ arrangement 1 ≠ C ∧ arrangement 2 ≠ C) ∧ 
       ∃ (s1 s2 : Fin 6),
         (s1 ≠ s2 ∧ arrangement s1 = arrangement s2 ∧ arrangement s1 = A) ∧
         ∃ (s3 s4 : Fin 6),
           (s3 ≠ s4 ∧ arrangement s3 = arrangement s4 ∧ arrangement s3 = B) ∧
         ∃ (s5 s6 : Fin 6),
           (s5 ≠ s6 ∧ arrangement s5 = arrangement s6 ∧ arrangement s5 = C)) ∧
      (∀ c ∈ communities, ∃! (x y : {v : students // arrangement v = c}), x ≠ y)
    }
  in  ∥ possible_arrangements ∥ = 9 := sorry

end arrangement_of_students_l590_590245


namespace max_value_q_l590_590871

noncomputable def q (A M C : ℕ) : ℕ :=
  A * M * C + A * M + M * C + C * A + A + M + C

theorem max_value_q : ∀ A M C : ℕ, A + M + C = 15 → q A M C ≤ 215 :=
by 
  sorry

end max_value_q_l590_590871


namespace evaluate_expression_l590_590323

theorem evaluate_expression :
  sqrt ((25 / 36) + (16 / 9)) = sqrt 89 / 6 := 
sorry

end evaluate_expression_l590_590323


namespace real_root_exists_l590_590305

theorem real_root_exists (a : ℝ) : 
    (∃ x : ℝ, x^4 - a * x^3 - x^2 - a * x + 1 = 0) ↔ (-1 / 2 ≤ a) := by
  sorry

end real_root_exists_l590_590305


namespace ellipse_eq_max_area_line_eq_1_l590_590365

-- Define the conditions
variable (a b : ℝ)
variable (h1 : a > b)
variable (h2 : b > 0)
variable (maj_axis : a = sqrt 3 * b)
variable (pt : ℝ × ℝ)
variable (h3 : pt = (sqrt 2, (2 * sqrt 3) / 3))

-- Define the equation of the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Part 1: Prove the equation of the ellipse
theorem ellipse_eq : ellipse 6 2 :=
by sorry

-- Part 2: Prove the line that maximizes the area of triangle OAB
variable {k : ℝ}

def line_through_focus (k : ℝ) : ℝ × ℝ → Prop :=
  λ F, F.2 = k * (F.1 - 2) ∨ F.2 = -k * (F.1 - 2)

theorem max_area_line_eq_1 : line_through_focus 1 = { F | F.2 = F.1 - 2 ∨ F.2 = -F.1 + 2 } :=
by sorry

end ellipse_eq_max_area_line_eq_1_l590_590365


namespace maximum_lambda_l590_590334

theorem maximum_lambda (a b : ℝ) : (27 / 4) * a^2 * b^2 * (a + b)^2 ≤ (a^2 + a * b + b^2)^3 := 
sorry

end maximum_lambda_l590_590334


namespace spans_of_multiples_l590_590469

variables {α β γ : Type*} [Add α] [Mul α] [Add β] [Mul β] [Add γ] [Mul γ]
variables (α1 β1 γ1 α2 β2 γ2 : ℝ)
variables (ℓ1 : α1 + β1 + γ1 ≠ 0)
variables (ℓ2 : α2 + β2 + γ2 ≠ 0)

def spans_equal (u v : ℝ × ℝ × ℝ) : Prop :=
  (∃ a, a ≠ 0 ∧ u = a • v) ∧ (∃ b, b ≠ 0 ∧ v = b • u)

theorem spans_of_multiples :
  (span ℝ ({α1, β1, γ1} : set ℝ)) = (span ℝ ({α2, β2, γ2} : set ℝ)) ↔ (∃ k : ℝ, k ≠ 0 ∧ (α1, β1, γ1) = k • (α2, β2, γ2)) :=
sorry

end spans_of_multiples_l590_590469


namespace class_monitor_election_outcomes_l590_590952

-- Define the condition that student A cannot be the entertainment committee member
def students := {1, 2, 3, 4}
def is_student_a (s : ℕ) : Prop := s = 1

-- Define the number of choices for the entertainment committee member
def num_entertainment_choices :=
  (students.erase 1).card

-- Define the number of choices for the class monitor when 1 entertainment committee member is chosen
def num_class_monitor_choices :=
  students.erase 1.card

-- Theorem statement that proves the total number of outcomes
theorem class_monitor_election_outcomes : 
  num_entertainment_choices * num_class_monitor_choices = 9 :=
by
  -- Mathematical operations and deductions go here
  sorry

end class_monitor_election_outcomes_l590_590952


namespace petya_time_comparison_l590_590680

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l590_590680


namespace mark_jane_difference_l590_590122

-- Define M and J as specified in the problem
def M := 12 - (3 + 6)
def J := 12 - 3 + 6

-- State the proof problem
theorem mark_jane_difference : M - J = -12 :=
by
  -- skipping proof
  sorry

end mark_jane_difference_l590_590122


namespace exists_nat_satisfying_expression_l590_590062

theorem exists_nat_satisfying_expression : ∃ n : ℕ, n > 1 ∧ ∃ k: ℕ, (sqrt (n * sqrt (n * sqrt n)) = k) :=
by
  sorry

end exists_nat_satisfying_expression_l590_590062


namespace expected_lifetime_flashlight_l590_590099

noncomputable theory

variables (ξ η : ℝ) -- ξ and η are continuous random variables representing the lifetimes
variables (T : ℝ) -- T is the lifetime of the flashlight

-- Define the maximum lifetime of the flashlight
def max_lifetime (ξ η : ℝ) : ℝ := max ξ η

-- Given condition: the expectation of η is 4
axiom expectation_eta : E η = 4

-- Theorem statement: expected lifetime of the flashlight is at least 4
theorem expected_lifetime_flashlight (ξ η : ℝ) (h : T = max_lifetime ξ η) : 
  E (max_lifetime ξ η) ≥ 4 :=
by 
  sorry

end expected_lifetime_flashlight_l590_590099


namespace cube_root_arithmetic_square_root_l590_590379

-- Define the condition
def condition (a : ℝ) := (∛27 = a + 3)

-- State the theorem to be proved
theorem cube_root_arithmetic_square_root (a : ℝ) 
  (h : condition a) : 
  sqrt (a + 4) = 2 := 
by
  sorry

end cube_root_arithmetic_square_root_l590_590379


namespace last_four_digits_n_l590_590881

theorem last_four_digits_n : 
  ∃ n : ℕ, 
    (n > 0) ∧
    (n % 4 = 0) ∧ 
    (n % 9 = 0) ∧ 
    (∀ d ∈ (nat.digits 10 n), d = 4 ∨ d = 9) ∧
    (∃ d₁ d₂ ∈ (nat.digits 10 n), d₁ = 4 ∧ d₂ = 9) ∧
    (n % 10000 = 4944) :=
by sorry

end last_four_digits_n_l590_590881


namespace russom_greatest_number_of_envelopes_l590_590584

theorem russom_greatest_number_of_envelopes :
  ∃ n, n > 0 ∧ 18 % n = 0 ∧ 12 % n = 0 ∧ ∀ m, m > 0 ∧ 18 % m = 0 ∧ 12 % m = 0 → m ≤ n :=
sorry

end russom_greatest_number_of_envelopes_l590_590584


namespace balloons_given_by_mom_l590_590204

-- Definitions of the initial and total number of balloons
def initial_balloons := 26
def total_balloons := 60

-- Theorem: Proving the number of balloons Tommy's mom gave him
theorem balloons_given_by_mom : total_balloons - initial_balloons = 34 :=
by
  -- This proof is obvious from the setup, so we write sorry to skip the proof.
  sorry

end balloons_given_by_mom_l590_590204


namespace jellybean_count_l590_590252

def black_beans : Nat := 8
def green_beans : Nat := black_beans + 2
def orange_beans : Nat := green_beans - 1
def total_jelly_beans : Nat := black_beans + green_beans + orange_beans

theorem jellybean_count : total_jelly_beans = 27 :=
by
  -- proof steps would go here.
  sorry

end jellybean_count_l590_590252


namespace perfect_square_trinomial_l590_590827

theorem perfect_square_trinomial (m : ℤ) (h : ∃ b : ℤ, (x : ℤ) → x^2 - 10 * x + m = (x + b)^2) : m = 25 :=
sorry

end perfect_square_trinomial_l590_590827


namespace tan_alpha_plus_pi_over_4_trigonometric_expression_value_l590_590353

variable (α : ℝ)

noncomputable def tan_alpha_value : Prop := 
  tan α = 2

theorem tan_alpha_plus_pi_over_4 (h : tan_alpha_value α) : 
  tan (α + π / 4) = -3 := 
sorry

theorem trigonometric_expression_value (h : tan_alpha_value α) : 
  (sin (2 * α) / (sin (α) ^ 2 + sin (α) * cos (α) - cos (2 * α) - 1)) = 1 := 
sorry

end tan_alpha_plus_pi_over_4_trigonometric_expression_value_l590_590353


namespace sin_eq_is_iff_eq_l590_590428

-- Statement of the problem
theorem sin_eq_is_iff_eq (A B : ℝ) (h1 : ∠A + ∠B < π) (h2 : A ≠ π - B) :
  ∠A = ∠B ↔ sin A = sin B :=
begin
  sorry
end

end sin_eq_is_iff_eq_l590_590428


namespace profit_percent_correct_l590_590231

-- Definitions based on conditions
def marked_price_per_pen : ℝ := 1.0
def pens_bought : ℕ := 60
def cost_price : ℝ := 46.0
def discount_percent : ℝ := 1.0

-- Pause to define the Selling Price per pen after discount
def selling_price_per_pen (marked_price_per_pen : ℝ) (discount_percent : ℝ) : ℝ :=
  marked_price_per_pen - (marked_price_per_pen * discount_percent / 100.0)

-- Define the total selling price for all pens
def total_selling_price (selling_price_per_pen : ℝ) (pens_bought : ℕ) : ℝ :=
  selling_price_per_pen * pens_bought

-- Define the profit obtained
def profit (total_selling_price : ℝ) (cost_price : ℝ) : ℝ :=
  total_selling_price - cost_price

-- Define the profit percent
def profit_percent (profit : ℝ) (cost_price : ℝ) : ℝ :=
  (profit / cost_price) * 100.0

-- The main theorem to be proved
theorem profit_percent_correct :
  profit_percent (profit (total_selling_price (selling_price_per_pen marked_price_per_pen discount_percent) pens_bought) cost_price) cost_price ≈ 29.13 :=
by
  sorry

end profit_percent_correct_l590_590231


namespace number_of_integer_pairs_l590_590112

theorem number_of_integer_pairs (ω : ℂ) (hω : ω^4 = 1 ∧ ω ≠ 1 ∧ ω ≠ -1) :
  {p : ℤ × ℤ | |(p.1:ℂ) * ω + p.2| = 1}.to_finset.card = 4 :=
by
  sorry

end number_of_integer_pairs_l590_590112


namespace general_formula_a_sum_T_l590_590043

noncomputable def a (n : ℕ) : ℕ := if n = 1 then 0 else n - 1
noncomputable def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, (a (i + 1) + 1 : ℕ) / (2 ^ (i + 1) : ℝ)

theorem general_formula_a (n : ℕ) (h1 : n ≥ 1) (h2 : a 2 = 1) (h3 : 2 * S n = n * a n) :
  a n = n - 1 := by
  sorry

theorem sum_T (n : ℕ) (h1 : ∀ k, a k = k - 1) :
  T n = 2 - (n + 2) / (2 ^ n) := by
  sorry

end general_formula_a_sum_T_l590_590043


namespace jim_out_of_pocket_l590_590068

/-
Jim buys a wedding ring for $10,000. He gets his wife a ring that is twice that much and 
sells the first one for half its value. Prove that he is out of pocket $25,000.
-/

def first_ring_cost : ℕ := 10000
def second_ring_cost : ℕ := 2 * first_ring_cost
def selling_price_first_ring : ℕ := first_ring_cost / 2
def out_of_pocket : ℕ := (first_ring_cost - selling_price_first_ring) + second_ring_cost

theorem jim_out_of_pocket : out_of_pocket = 25000 := 
by
  -- type hint for clarity
  unfolding first_ring_cost second_ring_cost selling_price_first_ring out_of_pocket
  -- manually calculate and verify
  change ((10000 - 5000) + 20000) = 25000
  simp -- simplify the expression
  exact rfl -- reflexivity of equality

end jim_out_of_pocket_l590_590068


namespace problem_trig_l590_590854

noncomputable def composition (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, ∃ g : ℝ → ℝ, g = (nat.iterate f n) ∧ g 2 = 2010

theorem problem_trig (h : ∃ n : ℕ, ∃ f : ℝ → ℝ, 
  (f = λ x, sin x ∨ f = λ x, cos x ∨ f = λ x, tan x ∨ f = λ x, (1 / tan x) ∨ 
   f = λ x, arcsin x ∨ f = λ x, arccos x ∨ f = λ x, arctan x ∨ f = λ x, arccot x) ∧ 
  composition f) : 
  True :=
sorry

end problem_trig_l590_590854


namespace find_a_from_roots_l590_590387

theorem find_a_from_roots (m a : ℤ) (h1 : (2 - m)^2 = a) (h2 : (2m + 1)^2 = a) : a = 25 := 
by
  sorry

end find_a_from_roots_l590_590387


namespace correct_statements_count_l590_590471

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_property (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

theorem correct_statements_count (f : ℝ → ℝ)
  (h1 : is_odd_function f)
  (h2 : has_property f) :
  {s : finset ℕ // s.card = 4} :=
by
  -- Proof omitted
  sorry

end correct_statements_count_l590_590471


namespace fermats_little_theorem_for_q_plus_1_l590_590366

theorem fermats_little_theorem_for_q_plus_1 (q : ℕ) (h1 : Nat.Prime q) (h2 : q % 2 = 1) :
  (q + 1)^(q - 1) % q = 1 := by
  sorry

end fermats_little_theorem_for_q_plus_1_l590_590366


namespace area_T_l590_590713

-- Given conditions
def matrix := !![ [3, 4], [-8, 6] ] -- Transformation matrix
def area_T : ℝ := 9 -- Area of T

-- Define the problem
theorem area_T'_is_450 : |LinearAlgebra.Det matrix| * area_T = 450 := by sorry

end area_T_l590_590713


namespace subset_implies_bound_l590_590349

def setA := {x : ℝ | x < 2}
def setB (m : ℝ) := {x : ℝ | x < m}

theorem subset_implies_bound (m : ℝ) (h : setB m ⊆ setA) : m ≤ 2 :=
by 
  sorry

end subset_implies_bound_l590_590349


namespace EG_length_l590_590205

noncomputable def triangleDEF := {D E F : Type} [MetricSpace D] [MetricSpace E] [MetricSpace F]
                                  [IsTriangle D E F 8 10 12]

theorem EG_length (tDEF : triangleDEF) (bug1_speed bug2_speed : ℕ) (meet_point : tDEF.E) :
  bug1_speed = 1 → bug2_speed = 2 → 
  let perimeter := 8 + 10 + 12 in
  let total_distance := perimeter in
  let time_to_meet := total_distance / (bug1_speed + bug2_speed) in
  time_to_meet = 10 → -- calculated time from the solution
  let bug1_distance := bug1_speed * time_to_meet in
  let bug2_distance := bug2_speed * time_to_meet in
  bug1_distance = 10 → -- calculated distance from the solution
  bug2_distance = 20 → -- calculated distance from the solution
  ∃ (EG : ℝ), EG = 2 :=
begin
  intros h1 h2 perimeter total_distance time_to_meet ht bug1_distance bug2_distance hb1 hb2,
  -- Proof is not required, hence we use sorry.
  sorry
end

end EG_length_l590_590205


namespace product_slope_intercept_lt_neg1_l590_590164

theorem product_slope_intercept_lt_neg1 :
  let m := -3 / 4
  let b := 3 / 2
  m * b < -1 := 
by
  let m := -3 / 4
  let b := 3 / 2
  sorry

end product_slope_intercept_lt_neg1_l590_590164


namespace sum_of_first_7_terms_l590_590438

variable {a_n : ℕ → ℝ}

-- Conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∀ n, a_n > 0 ∧ (∃ d : ℝ, ∀ n, a_n = a_1 + n * d)

def condition (a_n : ℕ → ℝ) : Prop := a_1^2 + a_n 6^2 + 2 * a_1 * a_n 6 = 4

-- Statement of the problem:
theorem sum_of_first_7_terms (a_n : ℕ → ℝ) (h1 : is_arithmetic_sequence a_n) (h2 : condition a_n) :
  ∑ i in finset.range 7, a_n i = 7 :=
sorry

end sum_of_first_7_terms_l590_590438


namespace jellybean_total_l590_590249

theorem jellybean_total 
    (blackBeans : ℕ)
    (greenBeans : ℕ)
    (orangeBeans : ℕ)
    (h1 : blackBeans = 8)
    (h2 : greenBeans = blackBeans + 2)
    (h3 : orangeBeans = greenBeans - 1) :
    blackBeans + greenBeans + orangeBeans = 27 :=
by
    -- The proof will be placed here
    sorry

end jellybean_total_l590_590249


namespace find_the_liar_l590_590192

def is_multiple_of (n k : ℕ) : Prop := n % k = 0

def swap (n : ℕ) (i j : ℕ) : ℕ :=
  let digits := n.digits 10
  let swapped := digits.take (min i j) ++ [digits.get! (max i j)] ++ (digits.drop ((min i j) + 1)).take ((max i j) - (min i j) - 1) ++ [digits.get! (min i j)] ++ (digits.drop ((max i j) + 1))
  swapped.foldr (λ d acc, 10 * acc + d) 0

def statement_A (n i j : ℕ) : Prop := is_multiple_of (swap n i j) 8
def statement_B (n i j : ℕ) : Prop := ¬ is_multiple_of (swap n i j) 9
def statement_C (n i j : ℕ) : Prop := is_multiple_of (swap n i j) 10
def statement_D (n i j : ℕ) : Prop := is_multiple_of (swap n i j) 11

def exactly_one_is_false (conds : List Prop) : Prop :=
  conds.count (= false) = 1

theorem find_the_liar :
  let seven_digit_number := 2014315
  let cards := [2, 0, 1, 4, 3, 1, 5]
  (∃ i j : ℕ, i < j ∧ statement_A seven_digit_number i j) ∧
  (∃ i j : ℕ, i < j ∧ statement_B seven_digit_number i j) ∧
  (∃ i j : ℕ, i < j ∧ ¬ statement_C seven_digit_number i j) ∧
  (∃ i j : ℕ, i < j ∧ statement_D seven_digit_number i j) →
  ∃ p, p ∈ [statement_A seven_digit_number, statement_B seven_digit_number, ¬ statement_C seven_digit_number, statement_D seven_digit_number] ∧ exactly_one_is_false [statement_A seven_digit_number, statement_B seven_digit_number, statement_C seven_digit_number, statement_D seven_digit_number] :=
  sorry

end find_the_liar_l590_590192


namespace relationship_f_l590_590380

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := log a (abs (x + b))

-- Assume that f(x) = log_a |x + b| is an even function.
axiom even_f {a b : ℝ} : ∀ x, f a b x = f a b (-x)

-- Assume that f(x) is monotonic on (0, +∞).
axiom monotonic_f {a b : ℝ} : monotone_on (f a b) (set.Ioi 0) ∨ antitone_on (f a b) (set.Ioi 0)

-- The theorem we want to prove.
theorem relationship_f (a b : ℝ) (h_even : even_f) (h_mono : monotonic_f) : f a b (b - 2) < f a b (a + 1) :=
by
  sorry

end relationship_f_l590_590380


namespace original_cube_volume_l590_590501

theorem original_cube_volume (s : ℕ) (h : s + 2 = s - 2 + 4) :
  (s + 2) * (s - 2) * s = s^3 - 12 → s^3 = 27 :=
by
  intros h1 h2
  sorry

end original_cube_volume_l590_590501


namespace smallest_x_for_multiple_of_450_and_648_l590_590981

theorem smallest_x_for_multiple_of_450_and_648 (x : ℕ) (hx : x > 0) :
  ∃ (y : ℕ), (450 * 36) = y ∧ (450 * 36) % 648 = 0 :=
by
  use (450 / gcd 450 648 * 648 / gcd 450 648)
  sorry

end smallest_x_for_multiple_of_450_and_648_l590_590981


namespace f_increasing_g_is_odd_l590_590630

section
variable {x : ℝ}

def f (x : ℝ) : ℝ := x^3 - 1

theorem f_increasing : ∀ x : ℝ, (f'(x) ≥ 0) := 
by {
  -- Proof omitted
  sorry
}

def g (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) :=
by {
  -- Proof omitted
  sorry
}

end

end f_increasing_g_is_odd_l590_590630


namespace general_formula_a_n_sum_T_formula_l590_590046

noncomputable def sequence_a : ℕ → ℕ
| 1     := 0
| 2     := 1
| (n+1) := n  -- This definition assumes the derived formula a_n = n - 1.

def sum_S (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_a

theorem general_formula_a_n (n : ℕ) (h1 : sequence_a 2 = 1) (h2 : 2 * sum_S n = n * sequence_a n) :
  sequence_a n = n - 1 := sorry

noncomputable def sequence_b (n : ℕ) : ℝ :=
  (sequence_a n + 1) / (2 ^ n)

def sum_T (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => sequence_b (i + 1))

theorem sum_T_formula (n : ℕ) (h : ∀ n, sequence_a n = n - 1) :
  sum_T n = 2 - (n + 2) / (2 ^ n) := sorry

end general_formula_a_n_sum_T_formula_l590_590046


namespace Holder_Inequality_Application_l590_590357

theorem Holder_Inequality_Application (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
    (a / real.sqrt (a^2 + 8 * b * c) + b / real.sqrt (b^2 + 8 * c * a) + c / real.sqrt (c^2 + 8 * a * b)) ≥ 1 := 
sorry

end Holder_Inequality_Application_l590_590357


namespace triangle_construction_l590_590370

-- Defining the problem conditions
variable (a m : ℝ) (α β : ℝ)

-- Conditions: 
-- a is the known side length
-- m is the known height
-- α is angle B and β is angle C
-- Angle condition: α = 2 * β

def triangle_condition (A B C : Type) [EuclideanGeometry A B C] :=
  (∃ (B C : A), 
    side_length B C = a ∧
    height A B C = m ∧
    ∠B = 2 * ∠C
  )

-- Statement: Construction of such a triangle
theorem triangle_construction (A B C : Type) [EuclideanGeometry A B C] :
  ∃ (triangle : Triangle A B C), triangle_condition a m α β :=
sorry

end triangle_construction_l590_590370


namespace seq_expression_l590_590577

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := n^2 * a n

theorem seq_expression (a : ℕ → ℝ) (h₁ : a 1 = 2) (h₂ : ∀ n ≥ 1, S n a = n^2 * a n) :
  ∀ n ≥ 1, a n = 4 / (n * (n + 1)) :=
by
  sorry

end seq_expression_l590_590577


namespace range_of_a_l590_590012

theorem range_of_a (a : ℝ) : 
  4 * a^2 - 12 * (a + 6) > 0 ↔ a < -3 ∨ a > 6 := 
by sorry

end range_of_a_l590_590012


namespace binomial_coefficient_formula_l590_590769

-- Definitions based on conditions
variables (n k : ℕ)
variable (h : 0 ≤ k ∧ k ≤ n)

-- Proof statement
theorem binomial_coefficient_formula (n k : ℕ) (h : 0 ≤ k ∧ k ≤ n) :
  nat.choose n k = nat.factorial n / (nat.factorial (n - k) * nat.factorial k) :=
sorry

end binomial_coefficient_formula_l590_590769


namespace limit_sum_geometric_sequence_l590_590936

noncomputable def geometric_sequence (a₁ r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

def sum_geometric_sequence (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - r ^ n) / (1 - r)

theorem limit_sum_geometric_sequence (a₁ r : ℝ) (h1 : a₁ = 1/2) (h2 : r = 1/2) :
  (real.has_sum (λ n, a₁ * (1 - r ^ n) / (1 - r)) 1) :=
begin
  sorry  -- Proof omitted
end

end limit_sum_geometric_sequence_l590_590936


namespace probability_six_integers_diff_tens_l590_590525

-- Defining the range and conditions for the problem
def set_of_integers : Finset ℤ := Finset.range 70 \ Finset.range 10

def has_different_tens_digit (s : Finset ℤ) : Prop :=
  (s.card = 6) ∧ (∀ x y ∈ s, x ≠ y → (x / 10) ≠ (y / 10))

noncomputable def num_ways_choose_six_diff_tens : ℚ :=
  ((7 : ℚ) * (10^6 : ℚ))

noncomputable def total_ways_choose_six : ℚ :=
  (Nat.choose 70 6 : ℚ)

noncomputable def probability_diff_tens : ℚ :=
  num_ways_choose_six_diff_tens / total_ways_choose_six

-- Statement claiming the required probability
theorem probability_six_integers_diff_tens :
  probability_diff_tens = 1750 / 2980131 :=
by
  sorry

end probability_six_integers_diff_tens_l590_590525


namespace geometry_problem_l590_590900

variables {A B C P Q S R D : Type*}
[LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
[LinearOrderedField P] [LinearOrderedField Q] [LinearOrderedField S]
[LinearOrderedField R] [LinearOrderedField D]

-- Define the triangle and points
def triangle (A B C : Type*) := ∃ (ABC : Type*), True

def is_point_on_side (P : Type*) (AB : Type*) := ∃ (P : Type*), True
def angle (α : Type*) (degree : Int) := ∃ (α : Type*), True

def perpendicular (P : Type*) (line : Type*) := ∃ (P : Type*), True
def foot_of_altitude (A : Type*) (BC : Type*) := ∃ (D : Type*), True

def parallel (SR BC : Type*) := ∃ (SR BC : Type*), True
def concurrent (PS AD QR : Type*) := ∃ (PS AD QR : Type*), True

-- Main theorem statement
theorem geometry_problem 
  (P_on_AB : ∀ (P : Type*) (AB : Type*), is_point_on_side P AB)
  (Q_on_AC : ∀ (Q : Type*) (AC : Type*), is_point_on_side Q AC)
  (angle_APC_45 : ∀ (P C : Type*), angle (angle P C) 45)
  (angle_AQB_45 : ∀ (A Q B : Type*), angle (angle A Q B) 45)
  (P_perpendicular_AB : ∀ (P : Type*) (AB : Type*), perpendicular P AB)
  (Q_perpendicular_AC : ∀ (Q : Type*) (AC : Type*), perpendicular Q AC)
  (foot_D : ∀ (A : Type*) (BC : Type*), foot_of_altitude A BC)
  (S_definition : ∀ (P : Type*) (AB : Type*) (BQ : Type*), True)
  (R_definition : ∀ (Q : Type*) (AC : Type*) (CP : Type*), True):
    (parallel SR BC) ∧ (concurrent PS AD QR) :=
sorry

end geometry_problem_l590_590900


namespace FinalAnswer_l590_590866

noncomputable def N : ℕ :=
  let bipartition_6_4 := choose 10 4 * choose 24 2
  let bipartition_5_5 := choose 10 5 * choose 25 3 / 2
  bipartition_6_4 + bipartition_5_5

theorem FinalAnswer : (N : ℚ) / (factorial 9) = 23 / 24 := by
  sorry

end FinalAnswer_l590_590866


namespace no_triangle_with_prime_sides_and_nonzero_integer_area_l590_590723

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Heron's formula for the area of a triangle given sides a, b, c
noncomputable def triangle_area (a b c : ℕ) : ℝ := 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Predicate stating the triangle has prime side lengths and non-zero integer area
def triangle_with_prime_sides_and_integer_area (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ (∃ S : ℤ, S ≠ 0 ∧ S = Real.floor (triangle_area a b c))

theorem no_triangle_with_prime_sides_and_nonzero_integer_area :
  ¬ (∃ a b c : ℕ, triangle_with_prime_sides_and_integer_area a b c) :=
sorry

end no_triangle_with_prime_sides_and_nonzero_integer_area_l590_590723


namespace smallest_C_is_one_eighth_l590_590117

noncomputable
def smallest_constant_C (n : ℕ) (x : Fin n → ℝ) : Prop :=
  2 ≤ n ∧
  (∀ x : Fin n → ℝ, all x (λ xi, xi ≥ 0) →
  (∑ i in Finset.univ.pairs, (x i.1) * (x i.2) * ((x i.1)^2 + (x i.2)^2)) ≤
  (1 / 8) * (∑ i in Finset.univ, x i)^4)

theorem smallest_C_is_one_eighth (n : ℕ) (x : Fin n → ℝ) :
  smallest_constant_C n x := sorry

end smallest_C_is_one_eighth_l590_590117


namespace number_symmetry_equation_l590_590125

theorem number_symmetry_equation (a b : ℕ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10 * a + b) * (100 * b + 10 * (a + b) + a) = (100 * a + 10 * (a + b) + b) * (10 * b + a) :=
by
  sorry

end number_symmetry_equation_l590_590125


namespace right_triangle_candidate_l590_590285

theorem right_triangle_candidate :
  (∃ a b c : ℕ, (a, b, c) = (1, 2, 3) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (2, 3, 4) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (3, 4, 5) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (4, 5, 6) ∧ a^2 + b^2 = c^2) ↔
  (∃ a b c : ℕ, (a, b, c) = (3, 4, 5) ∧ a^2 + b^2 = c^2) :=
by
  sorry

end right_triangle_candidate_l590_590285


namespace regular_polygon_sides_l590_590754

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l590_590754


namespace triangle_AC_length_l590_590839

theorem triangle_AC_length (A B C : Point)
  (angle_A : ∠A = 30 * Real.pi / 180)
  (angle_C : ∠C = 120 * Real.pi / 180)
  (AB_length : dist A B = 6 * Real.sqrt 3) :
  dist A C = 6 :=
by sorry

end triangle_AC_length_l590_590839


namespace rhombus_diagonal_l590_590159

theorem rhombus_diagonal (d2 : ℝ) (area : ℝ) (d1 : ℝ) : d2 = 15 → area = 127.5 → d1 = 17 :=
by
  intros h1 h2
  sorry

end rhombus_diagonal_l590_590159


namespace inner_cubes_proof_l590_590956

def num_inner_cubes (W' L H : ℕ) : ℕ :=
  let W := W' - 2
  in if W = 2 * L ∧ W = 2 * H ∧ L = H then L * W * H else 0

theorem inner_cubes_proof (L H : ℕ) (hL : L = H) :
  num_inner_cubes 10 L H = 128 :=
by
  rw [num_inner_cubes]
  simp [hL, show 10 - 2 = 8, by norm_num]
  simp [hL, mul_comm]
  sorry

end inner_cubes_proof_l590_590956


namespace closest_integer_to_cuberoot_150_l590_590596

theorem closest_integer_to_cuberoot_150 : 
  let cube5 := 5^3 in 
  let cube6 := 6^3 in 
  let midpoint := (cube5 + cube6) / 2 in 
  125 < 150 ∧ 150 < 216 ∧ 150 < midpoint → 
  5 = round (150^(1/3)) := 
by 
  intro h
  sorry

end closest_integer_to_cuberoot_150_l590_590596


namespace no_distinct_differences_l590_590294

theorem no_distinct_differences :
  ¬ (∃ (a_1 a_2 b_1 b_2 b_3 c_1 c_2 d_1 d_2 d_3 : ℕ),
    ( a_1 ∈  { 0, 1, 2, ..., 14 }
    ∧ a_2 ∈ { 0, 1, 2, ..., 14 }
    ∧ b_1 ∈ { 0, 1, 2, ..., 14 }
    ∧ b_2 ∈ { 0, 1, 2, ..., 14 }
    ∧ b_3 ∈ { 0, 1, 2, ..., 14 }
    ∧ c_1 ∈ { 0, 1, 2, ..., 14 }
    ∧ c_2 ∈ { 0, 1, 2, ..., 14 }
    ∧ d_1 ∈ { 0, 1, 2, ..., 14 }
    ∧ d_2 ∈ { 0, 1, 2, ..., 14 }
    ∧ d_3 ∈ { 0, 1, 2, ..., 14 }
    ∧ a_1 ≠ a_2 ∧ a_1 ≠ b_1 ∧ a_1 ≠ b_2 ∧ a_1 ≠ b_3
    ∧ a_1 ≠ c_1 ∧ a_1 ≠ c_2 ∧ a_1 ≠ d_1 ∧ a_1 ≠ d_2 ∧ a_1 ≠ d_3
    ∧ a_2 ≠ b_1 ∧ a_2 ≠ b_2 ∧ a_2 ≠ b_3 ∧ a_2 ≠ c_1 ∧ a_2 ≠ c_2 ∧ a_2 ≠ d_1 ∧ a_2 ≠ d_2 ∧ a_2 ≠ d_3
    ∧ b_1 ≠ b_2 ∧ b_1 ≠ b_3 ∧ b_1 ≠ c_1 ∧ b_1 ≠ c_2 ∧ b_1 ≠ d_1 ∧ b_1 ≠ d_2 ∧ b_1 ≠ d_3
    ∧ b_2 ≠ b_3 ∧ b_2 ≠ c_1 ∧ b_2 ≠ c_2 ∧ b_2 ≠ d_1 ∧ b_2 ≠ d_2 ∧ b_2 ≠ d_3
    ∧ b_3 ≠ c_1 ∧ b_3 ≠ c_2 ∧ b_3 ≠ d_1 ∧ b_3 ≠ d_2 ∧ b_3 ≠ d_3
    ∧ c_1 ≠ c_2 ∧ c_1 ≠ d_1 ∧ c_1 ≠ d_2 ∧ c_1 ≠ d_3
    ∧ c_2 ≠ d_1 ∧ c_2 ≠ d_2 ∧ c_2 ≠ d_3
    ∧ d_1 ≠ d_2 ∧ d_1 ≠ d_3 ∧ d_2 ≠ d_3
    ∧ list.nodup [
    |a_1 - b_1|, |a_1 - b_2|, |a_1 - b_3|,
    |a_2 - b_1|, |a_2 - b_2|, |a_2 - b_3|,
    |c_1 - d_1|, |c_1 - d_2|, |c_1 - d_3|,
    |c_2 - d_1|, |c_2 - d_2|, |c_2 - d_3|,
    |a_1 - c_1|, |a_2 - c_2|
    ])).
by sorry

end no_distinct_differences_l590_590294


namespace obtuse_angle_bisectors_l590_590033

theorem obtuse_angle_bisectors (A B C P : Type) [IsRightTriangle ABC]
  (hA : angle A = 45) (hB : angle B = 45)
  (h_intersect : intersect (bisector A) (bisector B) = P) : 
  angle APB = 135 :=
sorry

end obtuse_angle_bisectors_l590_590033


namespace angle_ab_sum_l590_590448

noncomputable def angle_sum (PRM QRP NRM QPR PRN a b : ℝ) : Prop :=
  PRM = 125 ∧
  QRP = 55 ∧
  NRM = 55 ∧
  QPR = 180 - a - QRP ∧
  PRN = 180 - QRP ∧
  (55 + a + 55 + b) = 180

theorem angle_ab_sum (PRM QRP NRM QPR PRN a b : ℝ) (h : angle_sum PRM QRP NRM QPR PRN a b) :
  a + b = 70 :=
  by
    cases h with h1 h2
    sorry

end angle_ab_sum_l590_590448


namespace equation_has_infinite_solutions_l590_590569

theorem equation_has_infinite_solutions : 
  ∃ S : Set ℝ, (∀ x ∈ S, |x - 5| + x - 5 = 0) ∧ (S ⊆ { x | x ≤ 5 }) ∧ (S.infinite) :=
by
  sorry

end equation_has_infinite_solutions_l590_590569


namespace overall_average_runs_l590_590842

theorem overall_average_runs 
  (test_matches: ℕ) (test_avg: ℕ) 
  (odi_matches: ℕ) (odi_avg: ℕ) 
  (t20_matches: ℕ) (t20_avg: ℕ)
  (h_test_matches: test_matches = 25)
  (h_test_avg: test_avg = 48)
  (h_odi_matches: odi_matches = 20)
  (h_odi_avg: odi_avg = 38)
  (h_t20_matches: t20_matches = 15)
  (h_t20_avg: t20_avg = 28) :
  (25 * 48 + 20 * 38 + 15 * 28) / (25 + 20 + 15) = 39.67 :=
sorry

end overall_average_runs_l590_590842


namespace chord_lengths_arithmetic_sequence_l590_590986

-- Definitions based on the problem's conditions
noncomputable def center_of_circle := (5 / 2, 0 : ℝ)
noncomputable def point_on_circle := (5 / 2, 3 / 2 : ℝ)
def radius := 5 / 2

-- Geometric properties of the circle and arithmetic sequence conditions
def shortest_chord_length := 4
def longest_chord_length := 5
def common_difference (n : ℕ) := (longest_chord_length - shortest_chord_length) / (n - 1)

-- Main theorem statement
theorem chord_lengths_arithmetic_sequence (n : ℕ) :
  (1 / 6 < common_difference n ∧ common_difference n ≤ 1 / 3) ↔ n ∈ {4, 5, 6} :=
by
  sorry

end chord_lengths_arithmetic_sequence_l590_590986


namespace range_of_a_l590_590163

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ∃ M m : ℝ, (∀ y : ℝ, y ∈ set.Icc m M → f a y = y)) ↔ (a > 2 ∨ a < -1) :=
sorry

end range_of_a_l590_590163


namespace number_of_different_lines_l590_590424

def valid_numbers : Set ℕ := {0, 1, 2, 3, 5, 7}

def is_line (A B : ℕ) : Prop := A ≠ B

theorem number_of_different_lines : 
  ∑ (A : ℕ) in valid_numbers, ∑ (B : ℕ) in valid_numbers, if is_line A B then 1 else 0 = 22 :=
sorry

end number_of_different_lines_l590_590424


namespace c_zero_roots_arithmetic_seq_range_f1_l590_590806

section problem

variable (b : ℝ)
def f (x : ℝ) := x^3 + 3 * b * x^2 + 0 * x + (-2 * b^3)
def f' (x : ℝ) := 3 * x^2 + 6 * b * x + 0

-- Proving c = 0 if f(x) is increasing on (-∞, 0) and decreasing on (0, 2)
theorem c_zero (h_inc : ∀ x < 0, f' b x > 0) (h_dec : ∀ x > 0, f' b x < 0) : 0 = 0 := sorry

-- Proving f(x) = 0 has two other distinct real roots x1 and x2 different from -b, forming an arithmetic sequence
theorem roots_arithmetic_seq (hb : ∀ x : ℝ, f b x = 0 → (x = -b ∨ -b ≠ x)) : 
    ∃ (x1 x2 : ℝ), x1 ≠ -b ∧ x2 ≠ -b ∧ x1 + x2 = -2 * b := sorry

-- Proving the range of values for f(1) when the maximum value of f(x) is less than 16
theorem range_f1 (h_max : ∀ x : ℝ, f b x < 16 ) : 0 ≤ f b 1 ∧ f b 1 < 11 := sorry

end problem

end c_zero_roots_arithmetic_seq_range_f1_l590_590806


namespace exists_subset_sum_l590_590371

variable (x : Fin 100 → ℝ)

-- Condition 1: Sum of the 100 real numbers equals 1
def condition_sum : Prop := 
  (∑ k in Finset.range 100, x k) = 1

-- Condition 2: The absolute difference between consecutive terms is bounded by 1/50
def condition_abs_diff : Prop := 
  ∀ k : Fin 99, abs (x k.succ - x k) ≤ 1 / 50

-- The statement to prove
theorem exists_subset_sum :
  condition_sum x →
  condition_abs_diff x →
  ∃ (i : Fin 50 → Fin 100), (StrictMono i) ∧ (∑ j, x (i j) ∈ Set.Icc (49 / 100 : ℝ) (51 / 100 : ℝ)) := by
  sorry

end exists_subset_sum_l590_590371


namespace cookie_difference_l590_590665

def AlyssaCookies : ℕ := 129
def AiyannaCookies : ℕ := 140
def Difference : ℕ := 11

theorem cookie_difference : AiyannaCookies - AlyssaCookies = Difference := by
  sorry

end cookie_difference_l590_590665


namespace general_formula_a_sum_T_l590_590042

noncomputable def a (n : ℕ) : ℕ := if n = 1 then 0 else n - 1
noncomputable def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, (a (i + 1) + 1 : ℕ) / (2 ^ (i + 1) : ℝ)

theorem general_formula_a (n : ℕ) (h1 : n ≥ 1) (h2 : a 2 = 1) (h3 : 2 * S n = n * a n) :
  a n = n - 1 := by
  sorry

theorem sum_T (n : ℕ) (h1 : ∀ k, a k = k - 1) :
  T n = 2 - (n + 2) / (2 ^ n) := by
  sorry

end general_formula_a_sum_T_l590_590042


namespace scale_read_total_weight_l590_590144

theorem scale_read_total_weight (weight_blue weight_brown : ℝ) 
  (h1 : weight_blue = 6) 
  (h2 : weight_brown = 3.12) : 
  weight_blue + weight_brown = 9.12 := 
by
  rw [h1, h2]
  norm_num
  sorry

end scale_read_total_weight_l590_590144


namespace two_distinct_solutions_exist_l590_590133

theorem two_distinct_solutions_exist :
  ∃ (a1 b1 c1 d1 e1 a2 b2 c2 d2 e2 : ℕ), 
    1 ≤ a1 ∧ a1 ≤ 9 ∧ 1 ≤ b1 ∧ b1 ≤ 9 ∧ 1 ≤ c1 ∧ c1 ≤ 9 ∧ 1 ≤ d1 ∧ d1 ≤ 9 ∧ 1 ≤ e1 ∧ e1 ≤ 9 ∧
    1 ≤ a2 ∧ a2 ≤ 9 ∧ 1 ≤ b2 ∧ b2 ≤ 9 ∧ 1 ≤ c2 ∧ c2 ≤ 9 ∧ 1 ≤ d2 ∧ d2 ≤ 9 ∧ 1 ≤ e2 ∧ e2 ≤ 9 ∧
    (b1 - d1 = 2) ∧ (d1 - a1 = 3) ∧ (a1 - c1 = 1) ∧
    (b2 - d2 = 2) ∧ (d2 - a2 = 3) ∧ (a2 - c2 = 1) ∧
    ¬ (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) :=
by
  sorry

end two_distinct_solutions_exist_l590_590133


namespace subset_difference_eq_a_or_b_l590_590463

theorem subset_difference_eq_a_or_b
  (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
  (A : set ℕ) (hA_subset : A ⊆ finset.range (a + b + 1)) 
  (hA_card : finset.card A > (a + b) / 2) :
  ∃ x y ∈ A, x ≠ y ∧ (x - y = a ∨ x - y = b) :=
sorry

end subset_difference_eq_a_or_b_l590_590463


namespace find_m_l590_590115

-- Define the function with given conditions
def f (m : ℕ) (n : ℕ) : ℕ := 
if n > m^2 then n - m + 14 else sorry

-- Define the main problem
theorem find_m (m : ℕ) (hyp : m ≥ 14) : f m 1995 = 1995 ↔ m = 14 ∨ m = 45 :=
by
  sorry

end find_m_l590_590115


namespace general_formula_a_n_sum_T_formula_l590_590044

noncomputable def sequence_a : ℕ → ℕ
| 1     := 0
| 2     := 1
| (n+1) := n  -- This definition assumes the derived formula a_n = n - 1.

def sum_S (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_a

theorem general_formula_a_n (n : ℕ) (h1 : sequence_a 2 = 1) (h2 : 2 * sum_S n = n * sequence_a n) :
  sequence_a n = n - 1 := sorry

noncomputable def sequence_b (n : ℕ) : ℝ :=
  (sequence_a n + 1) / (2 ^ n)

def sum_T (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => sequence_b (i + 1))

theorem sum_T_formula (n : ℕ) (h : ∀ n, sequence_a n = n - 1) :
  sum_T n = 2 - (n + 2) / (2 ^ n) := sorry

end general_formula_a_n_sum_T_formula_l590_590044


namespace raja_journey_distance_l590_590619

theorem raja_journey_distance
  (T : ℝ) (D : ℝ)
  (H1 : T = 10)
  (H2 : ∀ t1 t2, t1 = D / 42 ∧ t2 = D / 48 → T = t1 + t2) :
  D = 224 :=
by
  sorry

end raja_journey_distance_l590_590619


namespace remainder_of_T_l590_590484

open Nat

theorem remainder_of_T (T : ℕ) 
  (hT : T = ∑ n in range 670, (-1 : ℤ) ^ n * (nat.choose 2005 (3 * n + 1))) : 
  T % 1000 = 18 := 
  sorry

end remainder_of_T_l590_590484


namespace regular_polygon_sides_l590_590740

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l590_590740


namespace expected_lifetime_flashlight_l590_590091

noncomputable def xi : ℝ := sorry
noncomputable def eta : ℝ := sorry

def T : ℝ := max xi eta

axiom E_eta_eq_4 : E eta = 4

theorem expected_lifetime_flashlight : E T ≥ 4 :=
by
  -- The solution will go here
  sorry

end expected_lifetime_flashlight_l590_590091


namespace det_matrix_4x4_l590_590622

def matrix_4x4 : Matrix (Fin 4) (Fin 4) ℤ :=
  ![
    ![3, 0, 2, 0],
    ![2, 3, -1, 4],
    ![0, 4, -2, 3],
    ![5, 2, 0, 1]
  ]

theorem det_matrix_4x4 : Matrix.det matrix_4x4 = -84 :=
by
  sorry

end det_matrix_4x4_l590_590622


namespace presentation_order_count_l590_590844

theorem presentation_order_count (contestants : List String)
    (girls boys : List String)
    (c1 : |girls| = 3)
    (c2 : |boys| = 2)
    (c3 : ∀ i : Fin 4, ¬ (contestants[nth i] = boys.head ∧ contestants[nth (i + 1)] = boys.slice 1 1))
    (c4 : ¬ (contestants.head = "Girl_A")) :
    (∃ count : Nat, count = 60) := 
sorry

end presentation_order_count_l590_590844


namespace find_a_l590_590798

theorem find_a (a x : ℝ) (h1 : 3 * a - x = x / 2 + 3) (h2 : x = 2) : a = 2 := 
by
  sorry

end find_a_l590_590798


namespace correct_simplification_l590_590225

theorem correct_simplification : -(+6) = -6 := by
  sorry

end correct_simplification_l590_590225


namespace annual_compound_interest_rate_l590_590292

-- Let r be the annual compound interest rate we want to determine
-- Let X be the annual inflation rate
-- Let Y be the annual tax rate on earned interest
-- Let t be the number of years

theorem annual_compound_interest_rate
  (P : ℝ) -- principal amount
  (X : ℝ) -- annual inflation rate
  (Y : ℝ) -- annual tax rate on earned interest
  (t : ℕ) -- number of years
  (h_t : t = 22) : -- condition given
  ∃ r : ℝ, r = \frac{(2 * (1 + X))^{1/(22:ℝ)} - 1}{1 - Y} :=
by
  sorry

end annual_compound_interest_rate_l590_590292


namespace closest_integer_to_cube_root_of_150_l590_590603

theorem closest_integer_to_cube_root_of_150 : 
  let cbrt := (150: ℝ)^(1/3) in
  abs (cbrt - 6) < abs (cbrt - 5) :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590603


namespace harmonic_series_segment_sum_l590_590034

noncomputable def euler_harmonic_approximation (n : ℕ) : ℝ := 
  Real.log n + 0.557

theorem harmonic_series_segment_sum : 
  (∑ k in Finset.range 10000, (1 : ℝ) / (10001 + k)) = Real.log 2 := 
by
  sorry

end harmonic_series_segment_sum_l590_590034


namespace apples_per_classmate_l590_590829

theorem apples_per_classmate 
  (total_apples : ℕ) 
  (people : ℕ) 
  (h : total_apples = 15) 
  (p : people = 3) : 
  total_apples / people = 5 :=
by
  rw [h, p]
  norm_num

end apples_per_classmate_l590_590829


namespace binary_div_mul_l590_590325

-- Define the binary numbers
def a : ℕ := 0b101110
def b : ℕ := 0b110100
def c : ℕ := 0b110

-- Statement to prove the given problem
theorem binary_div_mul : (a * b) / c = 0b101011100 := by
  -- Skipping the proof
  sorry

end binary_div_mul_l590_590325


namespace closest_integer_to_cuberoot_150_l590_590597

theorem closest_integer_to_cuberoot_150 : 
  let cube5 := 5^3 in 
  let cube6 := 6^3 in 
  let midpoint := (cube5 + cube6) / 2 in 
  125 < 150 ∧ 150 < 216 ∧ 150 < midpoint → 
  5 = round (150^(1/3)) := 
by 
  intro h
  sorry

end closest_integer_to_cuberoot_150_l590_590597


namespace min_jiong_circle_area_theorem_l590_590210

-- Define the Jiong function
def jiong_function (a b x : ℝ) : ℝ := b / (|x| - a)

-- Define the intersection point of the Jiong function with the y-axis when a = 1, b = 1
def jiong_intersection_point : ℝ × ℝ := (0, -1)

-- Define the Jiong point
def jiong_point (a b : ℝ) : ℝ × ℝ := (0, 2 * (b / a))

-- Define the Jiong circle centered at the jiong_point with radius r
def jiong_circle_equation (a b r : ℝ) (x y : ℝ) : Bool := x^2 + (y - jiong_point a b).snd^2 = r^2

-- Define the minimum area among all Jiong circles
def min_jiong_circle_area : ℝ := 3 * Real.pi

-- Statement of the problem
theorem min_jiong_circle_area_theorem : 
  ∀ (a b : ℝ),
  a = 1 → b = 1 → 
  (∃ r, jiong_circle_equation a b r (jiong_intersection_point.fst) (jiong_intersection_point.snd) 
    ∧ Real.pi * r^2 = min_jiong_circle_area) :=
by 
  -- Proof omitted
  intros a b ha hb
  use √3
  simp [jiong_circle_equation, jiong_point, jiong_intersection_point]
  sorry

end min_jiong_circle_area_theorem_l590_590210


namespace time_spent_reading_l590_590151

/-- Let the ratio of time spent on swimming, reading, and hanging out with friends be 1:4:10.
If Susan spends 20 hours hanging out with friends, then she spends 8 hours reading. -/
theorem time_spent_reading (h_ratio : 1 / 4 / 10) (h_friends : 20) : 8 :=
  sorry

end time_spent_reading_l590_590151


namespace subsets_intersection_l590_590114

theorem subsets_intersection (m n : ℕ) (h : m > n) :
  let A := {i : ℕ | 1 ≤ i ∧ i ≤ m}
  let B := {i : ℕ | 1 ≤ i ∧ i ≤ n}
  (number_of_subsets : ∀ (C ⊆ A), B ∩ C ≠ ∅) = 2^(m-n) * (2^n - 1) :=
begin
  sorry
end

end subsets_intersection_l590_590114


namespace union_M_N_l590_590793

def M : Set ℝ := { x | -3 < x ∧ x ≤ 5 }
def N : Set ℝ := { x | x > 3 }

theorem union_M_N : M ∪ N = { x | x > -3 } :=
by
  sorry

end union_M_N_l590_590793


namespace smallest_x_l590_590978

theorem smallest_x (x : ℕ) (h : 450 * x % 648 = 0) : x = 36 := 
sorry

end smallest_x_l590_590978


namespace probability_six_integers_unique_tens_digit_l590_590539

theorem probability_six_integers_unique_tens_digit :
  (∃ (x1 x2 x3 x4 x5 x6 : ℕ),
    10 ≤ x1 ∧ x1 ≤ 79 ∧
    10 ≤ x2 ∧ x2 ≤ 79 ∧
    10 ≤ x3 ∧ x3 ≤ 79 ∧
    10 ≤ x4 ∧ x4 ≤ 79 ∧
    10 ≤ x5 ∧ x5 ≤ 79 ∧
    10 ≤ x6 ∧ x6 ≤ 79 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x1 ≠ x6 ∧
    x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x2 ≠ x6 ∧
    x3 ≠ x4 ∧ x3 ≠ x5 ∧ x3 ≠ x6 ∧
    x4 ≠ x5 ∧ x4 ≠ x6 ∧
    x5 ≠ x6 ∧
    tens_digit x1 ≠ tens_digit x2 ∧
    tens_digit x1 ≠ tens_digit x3 ∧
    tens_digit x1 ≠ tens_digit x4 ∧
    tens_digit x1 ≠ tens_digit x5 ∧
    tens_digit x1 ≠ tens_digit x6 ∧
    tens_digit x2 ≠ tens_digit x3 ∧
    tens_digit x2 ≠ tens_digit x4 ∧
    tens_digit x2 ≠ tens_digit x5 ∧
    tens_digit x2 ≠ tens_digit x6 ∧
    tens_digit x3 ≠ tens_digit x4 ∧
    tens_digit x3 ≠ tens_digit x5 ∧
    tens_digit x3 ≠ tens_digit x6 ∧
    tens_digit x4 ≠ tens_digit x5 ∧
    tens_digit x4 ≠ tens_digit x6 ∧
    tens_digit x5 ≠ tens_digit x6)
    →
  (probability := \(\frac{4375}{744407}\)).sorry

end probability_six_integers_unique_tens_digit_l590_590539


namespace ratio_of_angles_l590_590059

theorem ratio_of_angles 
  (A B C D E F : Type) 
  [triangle ABC]
  (AD_eq_DC : AD = DC) 
  (AE_eq_EB : AE = EB) 
  (angle_BFC : ℝ)
  (given_angle_BFC : angle BFC = angle_BFC)
  (intersection : F = point_of_intersection BD CE) : 
  ratio (angle BFC) (angle BDF) = 1 / 2 :=
sorry

end ratio_of_angles_l590_590059


namespace min_sum_of_product_is_40_l590_590571

theorem min_sum_of_product_is_40 :
  ∃ (a b c : ℕ), a * b * c = 396 ∧ a % 2 = 1 ∧ (a + b + c = 40) ∧
  (∀ a' b' c' : ℕ, a' * b' * c' = 396 ∧ a' % 2 = 1 → a' + b' + c' ≥ 40) :=
begin
  sorry
end

end min_sum_of_product_is_40_l590_590571


namespace maximize_profit_l590_590268

variables (a x : ℝ) (t : ℝ := 5 - 12 / (x + 3)) (cost : ℝ := 10 + 2 * t) 
  (price : ℝ := 5 + 20 / t) (profit : ℝ := 2 * (price * t - cost - x))

-- Assume non-negativity and upper bound on promotional cost
variable (h_a_nonneg : 0 ≤ a)
variable (h_a_pos : 0 < a)

noncomputable def profit_function (x : ℝ) : ℝ := 20 - 4 / x - x

-- Prove the maximum promotional cost that maximizes the profit
theorem maximize_profit : 
  (if a ≥ 2 then ∃ y, y = 2 ∧ profit_function y = profit_function 2 
   else ∃ y, y = a ∧ profit_function y = profit_function a) := 
sorry

end maximize_profit_l590_590268


namespace period_of_f_find_constants_l590_590811

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  sin (x + π/6) + sin (x - π/6) + a * cos x + b

-- Proving the smallest positive period of the function
theorem period_of_f (a b : ℝ) : 
  (∀ x : ℝ, f a b (x + 2 * π) = f a b x) ∧ (∀ p : ℝ, (∀ x : ℝ, f a b (x + p) = f a b x) → 2 * π ≤ p) :=
  sorry

-- Given that f is monotonically increasing in the given interval and attains a minimum value of 2
theorem find_constants (a b : ℝ) : 
  (∀ x y : ℝ, -π/3 ≤ x → x ≤ y → y ≤ 0 → f a b x ≤ f a b y) ∧ 
  (∀ x : ℝ, -π/3 ≤ x ∧ x ≤ 0 → (∃ m : ℝ, f a b m = 2)) → 
  a = -1 ∧ b = 4 :=
  sorry

end period_of_f_find_constants_l590_590811


namespace regular_polygon_sides_l590_590738

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l590_590738


namespace quadrilateral_area_correct_l590_590443

-- Define the points and conditions
def square_side_length := 1
def A := (0 : ℝ, 0 : ℝ)
def B := (square_side_length : ℝ, 0 : ℝ)
def C := (square_side_length : ℝ, square_side_length : ℝ)
def D := (0 : ℝ, square_side_length : ℝ)
def E := ((0 + square_side_length) / 2, 0)
def F := (square_side_length, (0 + square_side_length) / 2)
def G := (2 / 3, 1 / 3)
def H := (4 / 5, 3 / 5)

-- The statement we want to prove
def area_AGHD := abs (0 * 1/3 + 2/3 * 3/5 + 4/5 * 1 + 0 * 0 - (0 * 2/3 + 1/3 * 4/5 + 3/5 * 0 + 1 * 0)) / 2

theorem quadrilateral_area_correct :
  area_AGHD = 7 / 15 :=
sorry

end quadrilateral_area_correct_l590_590443


namespace max_sequence_length_l590_590767

open Nat

theorem max_sequence_length (m : ℕ) (sequence : Seq ℕ) :
  (∀ (k : ℕ), k ≥ 1 ∧ k < sequence.length - 1 → (sequence.get k.pred ≠ sequence.get k.succ)) ∧
  (∀ (i₁ i₂ i₃ i₄ : ℕ), 1 ≤ i₁ ∧ i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ ≤ sequence.length →
    sequence.get i₁ = sequence.get i₃ ∧ sequence.get i₂ = sequence.get i₄ ∧ sequence.get i₁ ≠ sequence.get i₂ → false) →
  sequence.length ≤ 4 * m - 2 :=
sorry

end max_sequence_length_l590_590767


namespace equation_of_ellipse_max_value_of_S_l590_590368

-- Define the conditions
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  ( a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2) + (y^2 / b^2) = 1 )

def foci (a b c : ℝ) : Prop :=
  ( c > 0 ∧ c^2 = a^2 - b^2 ∧ (c / a) = 1 / 2 )

def point_on_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  ( ellipse x y a b ∧ x = sqrt 3 ∧ y = - (sqrt 3) / 2 )

-- Define the proof problems

-- (I) Prove the equation of the ellipse
theorem equation_of_ellipse (a b c : ℝ) : ∃ (a b : ℝ), 
  ellipse (sqrt 3) (- sqrt 3 / 2) a b ∧ foci a b c → 
  (a = 2 ∧ b = sqrt 3) ∧ (∀ x y, ellipse x y a b → (x^2 / 4) + (y^2 / 3) = 1) := 
sorry

-- (II) Prove the maximum value of S
theorem max_value_of_S (x y k : ℝ) (a b c : ℝ) :
  foci a b c ∧ ellipse x y a b ∧ k ≠ 0 → 
  ( ∃ S : ℝ, S = 6 * sqrt (k^2 * (k^2 + 1) / ((3 + 4 * k^2)^2)) →
  (∀ S_max, S_max = S ∧ (0 < S_max ∧ S_max <= 3 / 2)) ∧ (S_max = 3 / 2)) :=
sorry

end equation_of_ellipse_max_value_of_S_l590_590368


namespace fraction_of_groups_with_a_and_b_l590_590632

/- Definitions based on the conditions -/
def total_persons : ℕ := 6
def group_size : ℕ := 3
def person_a : ℕ := 1  -- arbitrary assignment for simplicity
def person_b : ℕ := 2  -- arbitrary assignment for simplicity

/- Hypotheses based on conditions -/
axiom six_persons (n : ℕ) : n = total_persons
axiom divided_into_two_groups (grp_size : ℕ) : grp_size = group_size
axiom a_and_b_included (a b : ℕ) : a = person_a ∧ b = person_b

/- The theorem to prove -/
theorem fraction_of_groups_with_a_and_b
    (total_groups : ℕ := Nat.choose total_persons group_size)
    (groups_with_a_b : ℕ := Nat.choose 4 1) :
    groups_with_a_b / total_groups = 1 / 5 :=
by
    sorry

end fraction_of_groups_with_a_and_b_l590_590632


namespace negative_comparison_l590_590707

theorem negative_comparison : -2023 > -2024 :=
sorry

end negative_comparison_l590_590707


namespace find_x_l590_590243

variable (x : ℕ)  -- we'll use natural numbers to avoid negative values

-- initial number of children
def initial_children : ℕ := 21

-- number of children who got off
def got_off : ℕ := 10

-- total children after some got on
def total_children : ℕ := 16

-- statement to prove x is the number of children who got on the bus
theorem find_x : initial_children - got_off + x = total_children → x = 5 :=
by
  sorry

end find_x_l590_590243


namespace probability_of_vowels_l590_590431

theorem probability_of_vowels (students : Fin 30 → Char) (unique_initials : ∀ i j, students i = students j → i = j)
  (same_initials : ∀ i, students i ∈ ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
  (Y_is_consonant : "Y" ∉ ['A', 'E', 'I', 'O', 'U']) :
  ∃ (prob : ℚ), prob = (5 / 26 : ℚ) := 
by
  sorry

end probability_of_vowels_l590_590431


namespace monic_quartic_polynomial_with_roots_l590_590331

theorem monic_quartic_polynomial_with_roots :
  ∃ (p : Polynomial ℚ), p.monic ∧ p.degree = 4 ∧
  (Polynomial.aeval (3 + Real.sqrt 5) p = 0) ∧ 
  (Polynomial.aeval (2 - Real.sqrt 7) p = 0) ∧
  p = Polynomial.C (1:ℚ) * (X^4 - 10*X^3 + 25*X^2 + 2*X - 12) := 
sorry

end monic_quartic_polynomial_with_roots_l590_590331


namespace expected_lifetime_at_least_four_l590_590102

universe u

variables (α : Type u) [MeasurableSpace α] {𝒫 : ProbabilitySpace α}
variables {ξ η : α → ℝ} [IsFiniteExpectation ξ] [IsFiniteExpectation η]

noncomputable def max_lifetime : α → ℝ := λ ω, max (ξ ω) (η ω)

theorem expected_lifetime_at_least_four 
  (h : ∀ ω, max (ξ ω) (η ω) ≥ η ω)
  (h_eta : @Expectation α _ _ η  = 4) : 
  @Expectation α _ _ max_lifetime ≥ 4 :=
by
  sorry

end expected_lifetime_at_least_four_l590_590102


namespace inscribed_angles_sum_270_l590_590714

-- Define the regular pentagon inscribed in a circle
def regular_pentagon := { A B C D E : Type }

-- Definitions for regularity and inscribed properties
def is_regular (pentagon : regular_pentagon) : Prop := sorry
def is_inscribed_in_circle (pentagon : regular_pentagon) : Prop := sorry

-- The sum of the angles inscribed in the segments outside the regular pentagon
def sum_of_inscribed_angles (pentagon : regular_pentagon) (is_reg : is_regular pentagon)
    (is_inscribed : is_inscribed_in_circle pentagon) : ℝ :=
  5 * (1 / 2 * (360 - 180 * (5-2) / 5))

-- The statement that needs proof: the sum is 270 degrees
theorem inscribed_angles_sum_270 (pentagon : regular_pentagon) 
    (is_reg : is_regular pentagon) (is_inscribed : is_inscribed_in_circle pentagon) :
  sum_of_inscribed_angles pentagon is_reg is_inscribed = 270 :=
by
  sorry

end inscribed_angles_sum_270_l590_590714


namespace probability_six_integers_diff_tens_l590_590521

-- Defining the range and conditions for the problem
def set_of_integers : Finset ℤ := Finset.range 70 \ Finset.range 10

def has_different_tens_digit (s : Finset ℤ) : Prop :=
  (s.card = 6) ∧ (∀ x y ∈ s, x ≠ y → (x / 10) ≠ (y / 10))

noncomputable def num_ways_choose_six_diff_tens : ℚ :=
  ((7 : ℚ) * (10^6 : ℚ))

noncomputable def total_ways_choose_six : ℚ :=
  (Nat.choose 70 6 : ℚ)

noncomputable def probability_diff_tens : ℚ :=
  num_ways_choose_six_diff_tens / total_ways_choose_six

-- Statement claiming the required probability
theorem probability_six_integers_diff_tens :
  probability_diff_tens = 1750 / 2980131 :=
by
  sorry

end probability_six_integers_diff_tens_l590_590521


namespace sum_of_medial_triangle_areas_l590_590785

theorem sum_of_medial_triangle_areas (T : Type) (area_T : ℝ) (h : area_T = 1):
  let area_T1 := area_T / 4
  let area_T2 := area_T1 / 4
  let area_T3 := area_T2 / 4
  -- The areas form an infinite geometric series with ratio (1/4)
  ∑' n : ℕ, (1/4)^(n+1) = 1/3 :=
begin
  let a := (1 : ℝ) / 4,
  let r := (1 : ℝ) / 4,
  have sum_geom_series : ∑' n : ℕ, a * r^n = a / (1 - r),
  { sorry },
  calc ∑' n : ℕ, (1 / 4)^(n + 1) = (1 / 4) * ∑' n : ℕ, (1 / 4)^n : sorry
  ... = (1 / 4) * ((1 / 4) / (1 - (1 / 4))) : by rw sum_geom_series
  ... = (1 / 4) * (1 / 3) : by norm_num
  ... = 1 / 3 : by norm_num
end

end sum_of_medial_triangle_areas_l590_590785


namespace max_students_satisfying_conditions_l590_590777

-- Definitions
variable (n : ℕ)
variable (students : Finset (Fin n))
variable ( knows : Fin n → Fin n → Prop)
variable [∀ x y : Fin n, Decidable (knows x y)]

-- Condition: Out of any 3 students, at least 2 know each other
def condition1 : Prop :=
  ∀ (s : Finset (Fin n)), s.card = 3 → ∃ (x y ∈ s), x ≠ y ∧ knows x y

-- Condition: Out of any 4 students, at least 2 do not know each other
def condition2 : Prop :=
  ∀ (s : Finset (Fin n)), s.card = 4 → ∃ (x y ∈ s), x ≠ y ∧ ¬ knows x y

-- Main theorem statement
theorem max_students_satisfying_conditions : 
  (∀ (students : Finset (Fin n)), condition1 n students knows) →
  (∀ (students : Finset (Fin n)), condition2 n students knows) → n ≤ 8 := 
sorry

end max_students_satisfying_conditions_l590_590777


namespace cube_root_1728_simplified_l590_590218

theorem cube_root_1728_simplified :
  let a := 12
  let b := 1
  a + b = 13 :=
by
  sorry

end cube_root_1728_simplified_l590_590218


namespace label_elements_even_n_l590_590480

theorem label_elements_even_n (n : ℕ)
  (B : Set ℕ)
  (A : Fin (2 * n + 1) → Set ℕ)
  (h1 : ∀ i, A i ∈ B → (A i).card = 2 * n)
  (h2 : ∀ i j, i < j → (A i ∩ A j).card = 1)
  (h3 : ∀ b ∈ B, ∃ i j, i < j ∧ b ∈ A i ∧ b ∈ A j) :
  (∃ f : B → Fin 2, (∀ i, (A i).filter (λ x, f x = 0)).card = n) ↔ (∃ k : ℕ, n = 2 * k) :=
sorry

end label_elements_even_n_l590_590480


namespace ellipse_equation_range_of_m_l590_590364

open Real

-- Definitions of conditions
def isEllipse (a b : ℝ) (a_gt_b : a > b) : Prop :=
  ∃ (x y : ℝ), (y^2 / a^2 + x^2 / b^2 = 1)

def isFoci (C : ℝ → ℝ → Prop) (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), (c > 0) ∧ (∀ (F1 F2 : ℝ × ℝ), (F1.2 = c) ∨ (F2.2 = -c) ∧ (F1.1 = 0) ∧ (F2.1 = 0))

def perpendicularLineThroughFocus (F1 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 = F1.1}

def intersectsEllipseAtPoints (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) (M N : ℝ × ℝ) : Prop :=
  l M ∧ l N ∧ C M ∧ C N

def triangleArea (M N F2 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((M.1 * (N.2 - F2.2) + N.1 * (F2.2 - M.2) + F2.1 * (M.2 - N.2)))

def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 - (b^2 / a^2))

def lineIntersectsYAxis (k m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + m}

-- Question Ⅰ: Prove the standard equation of the ellipse
theorem ellipse_equation (a b : ℝ) (ha : a > b > 0) (ecc : sqrt(3) / 2 = sqrt(1 - b^2/a^2)) 
    (area_Triangle_Condition : ∃ (c : ℝ), c > 0 ∧ (4 * c * b^2 / a = sqrt 3)) : 
    ∃ (C : Set (ℝ × ℝ)), ∀ (x y : ℝ), C (x, y) ↔ (x^2 / a^2 + y^2 / b^2 = 1) :=
  sorry

-- Question Ⅱ: Prove the range of m
theorem range_of_m (k : ℝ) (intersection_line_ellipse : ∀ (m : ℝ), ∃ (l : Set (ℝ × ℝ)) (P A B : ℝ × ℝ), lineIntersectsYAxis k m P ∧ intersectsEllipseAtPoints l isEllipse A B)
    (lambda_cond : ∀ (lambda : ℝ), ∃ (m : ℝ), (∃ (OA OB OP : ℝ × ℝ), OA + lambda * OB = 4 * OP)) :
    ∃ (m : ℝ), m ∈ ((-2 : ℝ), -1) ∪ (1, 2) ∪ {0} :=
  sorry

end ellipse_equation_range_of_m_l590_590364


namespace rhombus_area_l590_590193

theorem rhombus_area (side : ℝ) (half_side : ℝ) : 
  side = 4 → half_side = side / 2 →
  (side * side) / 2 = 8 :=
by
  intros hside hhalf_side
  have hside_eq : side = 4 := hside
  have hhalf_side_eq : half_side = 2 := by
    rw [hside, hhalf_side, Real.mul_div_cancel_left]
  rw [←hside_eq, ←hhalf_side_eq]
  have : 4 * 4 / 2 = 8 := by norm_num
  assumption

end rhombus_area_l590_590193


namespace arithmetic_sequence_general_formula_l590_590561

variable (a : ℤ) (n : ℕ)

def first_term := a - 1
def second_term := a + 1
def third_term := a + 3
def common_difference := second_term a - first_term a

theorem arithmetic_sequence_general_formula :
  ∃ a_n : ℕ → ℤ, a_n = (λ n, a + 2 * n - 3) :=
by
  use (λ n, a + 2 * n - 3)
  intros n
  sorry

end arithmetic_sequence_general_formula_l590_590561


namespace probability_different_tens_digit_l590_590531

open Nat

theorem probability_different_tens_digit :
  let total_ways := choose 70 6,
      favorable_ways := 7 * 10^6
  in 
    (favorable_ways : ℝ) / total_ways = (2000 / 3405864 : ℝ) :=
by
  have h1 : total_ways = 70.choose 6 := rfl
  have h2 : favorable_ways = 7 * 10^6 := rfl
  rw [h1, h2]
  sorry

end probability_different_tens_digit_l590_590531


namespace peaches_per_basket_l590_590191

-- Given conditions as definitions in Lean 4
def red_peaches : Nat := 7
def green_peaches : Nat := 3

-- The proof statement showing each basket contains 10 peaches in total.
theorem peaches_per_basket : red_peaches + green_peaches = 10 := by
  sorry

end peaches_per_basket_l590_590191


namespace find_x_value_l590_590006

theorem find_x_value:
  ∃ x: ℕ, x * 7000 = 28000 * 100 ∧ x = 400 :=
by
  let x := 400
  have h1 : 7000 * x = 28000 * 100 := by
    calc
      7000 * 400 = 2800000 : by norm_num
      28000 * 100 = 2800000 : by norm_num
  exact ⟨x, h1, rfl⟩

end find_x_value_l590_590006


namespace intersection_of_great_circles_l590_590455

theorem intersection_of_great_circles (n : ℕ) (h : n ≥ 2) :
  (∃ (points : set (set ℝ)), 
    ∀ (C1 C2 : set ℝ), C1 ∈ points ∧ C2 ∈ points ∧ C1 ≠ C2 → (C1 ∩ C2).nonempty ∧ 
    cardinality (⋃ (C ∈ points), C) ≥ 2 * n) :=
sorry

end intersection_of_great_circles_l590_590455


namespace misha_second_attempt_points_l590_590891

/--
Misha made a homemade dartboard at his summer cottage. The round board is 
divided into several sectors by circles, and you can throw darts at it. 
Points are awarded based on the sector hit.

Misha threw 8 darts three times. In his second attempt, he scored twice 
as many points as in his first attempt, and in his third attempt, he scored 
1.5 times more points than in his second attempt. How many points did he 
score in his second attempt?
-/
theorem misha_second_attempt_points:
  ∀ (x : ℕ), 
  (x ≥ 24) →
  (2 * x ≥ 48) →
  (3 * x = 72) →
  (2 * x = 48) :=
by
  intros x h1 h2 h3
  sorry

end misha_second_attempt_points_l590_590891


namespace construct_triangle_l590_590300

-- Define the given conditions for the problem
variable {α : Type} [EuclideanSpace α] -- Using Euclidean space for geometric definitions

-- Definitions as conditions for the median, angle bisector, and right angle at C
variable (A B C D P Q O : α)
variable (m_c l_c : ℝ)
variable (angle_C : PlaneAngle α := PlaneAngle.rightAngle) -- 90 degrees condition

-- Proof statement that asserts the construction of triangle ABC is possible
theorem construct_triangle (h_median : median_length C B m_c)
                           (h_bisector : angle_bisector_length C D l_c)
                           (h_right_angle : C = A ∧ C = B ∧ ∠ A C B = angle_C) :
   ConstructibleTriangle ABC := sorry

end construct_triangle_l590_590300


namespace parabola_chord_distance_l590_590377

-- Definitions based on the problem conditions
def parabola (x y : ℝ) := x^2 = 4 * y

def focus : ℝ × ℝ := (0, 1)

def line (k : ℝ) (x y : ℝ) := y = k * x + 1

def chord_length (k : ℝ) := Real.sqrt (1 + k^2) * Real.sqrt ((4 * k)^2 + 4 * 4) = 8

def origin : ℝ × ℝ := (0, 0)

def distance_from_origin (k : ℝ) := 
  if k = 1 then
    (abs (0 - 0 + 1) / (Real.sqrt (1^2 + (-1)^2))) = sqrt 2 / 2
  else if k = -1 then
    (abs (0 + 0 - 1) / (Real.sqrt (1^2 + 1^2))) = sqrt 2 / 2
  else
    false

-- Lean 4 statement for the above problem
theorem parabola_chord_distance : 
  ∀ k : ℝ, chord_length k → distance_from_origin k :=
by
  sorry

end parabola_chord_distance_l590_590377


namespace expected_lifetime_flashlight_l590_590082

noncomputable section

variables (ξ η : ℝ) -- lifetimes of the blue and red lightbulbs
variables [probability_space ℙ] -- assuming a probability space ℙ

-- condition: expected lifetime of the red lightbulb is 4 years
axiom expected_eta : ℙ.𝔼(η) = 4

-- the main proof problem
theorem expected_lifetime_flashlight : ℙ.𝔼(max ξ η) ≥ 4 :=
sorry

end expected_lifetime_flashlight_l590_590082


namespace maria_total_score_l590_590027

theorem maria_total_score : 
  let correct_answers := 15
      incorrect_answers := 10
      unanswered_questions := 5
      correct_score := 1
      incorrect_penalty := -0.25 in
  correct_answers * correct_score + incorrect_answers * incorrect_penalty = 12.5 :=
by
  sorry

end maria_total_score_l590_590027


namespace a_n_formula_T_n_sum_l590_590388

noncomputable def S (n : ℕ) : ℕ := n^2 + 2 * n
noncomputable def a (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b (n : ℕ) : ℕ := 2^n
noncomputable def T (n : ℕ) : ℕ := (6 * n + 1) * 4^(n + 1) - 4 / 9

theorem a_n_formula (n : ℕ) (h : 1 ≤ n) : a n = S n - S (n - 1) :=
sorry

theorem T_n_sum (n : ℕ) (h : 1 ≤ n) : ∑ i in Finset.range n, a (i + 1) * b (i + 1)^2 = T n :=
sorry

end a_n_formula_T_n_sum_l590_590388


namespace no_real_roots_of_f_l590_590170

def f (x : ℝ) : ℝ := (x + 1) * |x + 1| - x * |x| + 1

theorem no_real_roots_of_f :
  ∀ x : ℝ, f x ≠ 0 := by
  sorry

end no_real_roots_of_f_l590_590170


namespace factorization_example_l590_590162

theorem factorization_example (C D : ℤ) (h : 20 * y^2 - 122 * y + 72 = (C * y - 8) * (D * y - 9)) : C * D + C = 25 := by
  sorry

end factorization_example_l590_590162


namespace roots_negative_and_bounds_find_possible_values_of_b_and_c_l590_590802

theorem roots_negative_and_bounds
  (b c x₁ x₂ x₁' x₂' : ℤ) 
  (h1 : x₁ * x₂ > 0) 
  (h2 : x₁' * x₂' > 0)
  (h3 : x₁^2 + b * x₁ + c = 0) 
  (h4 : x₂^2 + b * x₂ + c = 0) 
  (h5 : x₁'^2 + c * x₁' + b = 0) 
  (h6 : x₂'^2 + c * x₂' + b = 0) :
  x₁ < 0 ∧ x₂ < 0 ∧ x₁' < 0 ∧ x₂' < 0 ∧ (b - 1 ≤ c ∧ c ≤ b + 1) :=
by
  sorry


theorem find_possible_values_of_b_and_c 
  (b c : ℤ) 
  (h's : ∃ x₁ x₂ x₁' x₂', 
    x₁ * x₂ > 0 ∧ 
    x₁' * x₂' > 0 ∧ 
    (x₁^2 + b * x₁ + c = 0) ∧ 
    (x₂^2 + b * x₂ + c = 0) ∧ 
    (x₁'^2 + c * x₁' + b = 0) ∧ 
    (x₂'^2 + c * x₂' + b = 0)) :
  (b = 4 ∧ c = 4) ∨ 
  (b = 5 ∧ c = 6) ∨ 
  (b = 6 ∧ c = 5) :=
by
  sorry

end roots_negative_and_bounds_find_possible_values_of_b_and_c_l590_590802


namespace petya_time_l590_590690

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l590_590690


namespace car_b_speed_l590_590702

theorem car_b_speed :
  ∀ (v : ℕ),
    (232 - 4 * v = 32) →
    v = 50 :=
  by
  sorry

end car_b_speed_l590_590702


namespace james_total_earnings_l590_590855

-- Define the earnings for January
def januaryEarnings : ℕ := 4000

-- Define the earnings for February based on January
def februaryEarnings : ℕ := 2 * januaryEarnings

-- Define the earnings for March based on February
def marchEarnings : ℕ := februaryEarnings - 2000

-- Define the total earnings including January, February, and March
def totalEarnings : ℕ := januaryEarnings + februaryEarnings + marchEarnings

-- State the theorem: total earnings should be 18000
theorem james_total_earnings : totalEarnings = 18000 := by
  sorry

end james_total_earnings_l590_590855


namespace find_a1_l590_590446

-- Define the arithmetic sequence and the given conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_mean (x y z : ℝ) : Prop :=
  y^2 = x * z

def problem_statement (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (arithmetic_sequence a d) ∧ (geometric_mean (a 1) (a 2) (a 4))

theorem find_a1 (a : ℕ → ℝ) (d : ℝ) (h : problem_statement a d) : a 1 = 1 := by
  have h_seq : arithmetic_sequence a d := h.1
  have h_geom : geometric_mean (a 1) (a 2) (a 4) := h.2
  sorry

end find_a1_l590_590446


namespace find_v_value_l590_590015

theorem find_v_value (x : ℝ) (v : ℝ) (h1 : x = 3.0) (h2 : 5 * x + v = 19) : v = 4 := by
  sorry

end find_v_value_l590_590015


namespace centers_of_rotation_exist_l590_590719

open Classical

-- Define centers of rotation for a given triangle ABC
noncomputable def triangle_center_of_rotation (A B C : Point) : (Point × Point) :=
  let circle1 := circle (A, B) ∩ tangent_to (BC),
      circle2 := circle (B, C) ∩ tangent_to (CA),
      circle3 := circle (C, A) ∩ tangent_to (AB) in
  (circle1 ∩ circle2, circle2 ∩ circle3)

-- The main theorem to prove the existence of centers of rotation
theorem centers_of_rotation_exist (A B C : Point) : ∃ O1 O2 : Point, 
  (O1, O2) = triangle_center_of_rotation A B C :=
sorry

end centers_of_rotation_exist_l590_590719


namespace find_lambda_l590_590407

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 3)
def c : ℝ × ℝ := (-4, -7)

-- Define the condition for collinearity
def collinear (u v : ℝ × ℝ) : Prop := 
  - (snd v) * (fst u) + (fst v) * (snd u) = 0

-- Define the vector expression λa + b
def lambda_vec (λ : ℝ) : ℝ × ℝ := (λ * (fst a) + (fst b), λ * (snd a) + (snd b))

-- State the theorem
theorem find_lambda (λ : ℝ) (h : collinear (lambda_vec λ) c) : λ = 2 :=
  sorry

end find_lambda_l590_590407


namespace fraction_ef_over_gh_l590_590507

variable (GH E F : Type) [MetricSpace GH] [MetricSpace E] [MetricSpace F]
variable (G H : GH) (E : E) (F : F)
variable (GE EH GF FH : ℝ)

-- Conditions
def ge_eq_3_times_eh (h1 : E ∈ line_segment GH G H) (h2 : GE = 3 * EH) : Prop := True
def gf_eq_7_times_fh (h3 : F ∈ line_segment GH G H) (h4 : GF = 7 * FH) : Prop := True

-- Question
theorem fraction_ef_over_gh (h1: E ∈ line_segment GH G H) (h2: F ∈ line_segment GH G H)
(h3: GE = 3 * EH) (h4: GF = 7 * FH) : 
  (length (line_segment GH E F)) / (length (line_segment GH G H)) = 1 / 8 := sorry

end fraction_ef_over_gh_l590_590507


namespace number_of_unreachable_integers_l590_590869

theorem number_of_unreachable_integers (k : ℕ) (h : k ≥ 0) :
  let a : ℕ → ℕ :=
    λ n, match n with
    | 0 => k
    | n + 1 => Nat.find (λ m, (m > a n) ∧ (∃ p, p * p = m + a n))
  in 
  let unreachables := {n | ∀ i j, i ≠ j → n ≠ a i - a j} in
  unreachables.card = Nat.floor (Real.sqrt (2 * k)) :=
sorry

end number_of_unreachable_integers_l590_590869


namespace impossible_event_l590_590666

/-- Given conditions:
 1. At standard atmospheric pressure, water boils at 100°C.
 2. A coin toss can result in either heads or tails.
 3. The absolute value of a real number is non-negative.
 Prove that the impossible event(s) among the options
 provided is only the first event: "Water boils at 80°C at
 standard atmospheric pressure." -/
theorem impossible_event :
  ∀ (boiling_temp : ℝ) (coin_toss_result : bool) (real_num : ℝ),
  (boiling_temp = 100) ∧ (coin_toss_result = tt ∨ coin_toss_result = ff) ∧ (abs real_num ≥ 0) →
  ¬ (boiling_temp ≠ 80) ∧ (false ∨ true) ∧ (abs real_num ≥ 0) :=
by
  intros boiling_temp coin_toss_result real_num h,
  cases h with h1 htemp,
  cases htemp with h2 h3,
  have h1' : boiling_temp = 100 := h1,
  have h1_untrue : boiling_temp ≠ 80 := by 
    rw h1
    sorry,
  have h2_true: false ∨ true := by exact or.inr trivial,
  have h3_true : abs real_num ≥ 0 := by exact h3,
  exact ⟨h1_untrue, h2_true, h3_true⟩

end impossible_event_l590_590666


namespace mySet_power_set_l590_590121

-- Define the set {0, 1}
def mySet : set ℕ := {0, 1}

-- Prove that the power set of mySet is {∅, {0}, {1}, {0, 1}}
theorem mySet_power_set :
  set.powerset mySet = {∅, {0}, {1}, {0, 1}} :=
by sorry

end mySet_power_set_l590_590121


namespace cube_root_simplification_l590_590217

theorem cube_root_simplification (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : b = 1) : 3
  := sorry

end cube_root_simplification_l590_590217


namespace regular_polygon_sides_l590_590725

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l590_590725


namespace smallest_k_integer_product_l590_590947

def a : ℕ → ℝ
| 0       := 2
| 1       := real.sqrt 3
| n + 2   := 2 * a (n + 1) * (a n)^3

-- Define the conditions by translating the given sequence into a Lean function.
theorem smallest_k_integer_product :
  ∃ k : ℕ, k > 0 ∧ ∃ s_k : ℕ, ((1:ℕ) to k).sum (λ n, 17 * real.logb 3 (a n)) = 17 * s_k := sorry

end smallest_k_integer_product_l590_590947


namespace probability_proof_l590_590450

-- Define the basic setup for the cuboid and relevant distances
def AB := 5
def AD := 5
def AA1 := 1
def P_in_square_base (P : ℝ × ℝ) : Prop := (0 ≤ P.1 ∧ P.1 ≤ AB) ∧ (0 ≤ P.2 ∧ P.2 ≤ AD)

-- Define the point M on edge AB such that 3 * AM + 2 * BM = 0
def M_coord : ℝ := (2 / 5) * AB

-- Define distances d1 and d2
def d1 (P : ℝ × ℝ) : ℝ := abs (P.2 - 1)
def d2 (P : ℝ × ℝ) : ℝ := (P.1 - M_coord)^2 + P.2^2

-- Define the probability calculation
def probability_condition (P : ℝ × ℝ) : Prop := d1(P)^2 - d2(P) ≥ 1

-- Prove that the probability of such P within the square base is 32/75
theorem probability_proof : ∃ p : ℝ, p = 32 / 75 ∧
  (let area_base := AB * AD in
   let feasible_area := area_base * p in
   p = feasible_area / area_base) :=
by {
  -- Calculation and proof skipped
  sorry
}

end probability_proof_l590_590450


namespace problem_statement_l590_590215

theorem problem_statement :
  ((3^1 - 2 + 4^2 + 1)⁻¹ * 7 = (7 / 18)) :=
by
  sorry

end problem_statement_l590_590215


namespace largest_value_l590_590663

theorem largest_value (n : ℕ) (h₁ : 10 ≤ n) (h₂ : n ≤ 99) : 3 * (300 - n) ≤ 870 :=
by
  have hmax : 3 * (300 - 10) = 870 := by norm_num
  have h3 : ∀ k, 10 ≤ k → k ≤ 99 → 3 * (300 - k) ≤ 870 :=
    by
      intros k hk₁ hk₂
      have h_le_10 : 300 - k ≤ 290 := by linarith
      calc
        3 * (300 - k) ≤ 3 * 290 : by nlinarith
        ...           = 870 : by norm_num
  exact h3 n h₁ h₂

end largest_value_l590_590663


namespace numValidPerms_l590_590845

-- Define the sequence
def seq := [2, 4, 5, 7, 8]

-- Predicate to check if no three consecutive terms are in increasing order
def noThreeConsecInc (l : List ℕ) : Prop :=
  ∀ (a b c : ℕ), [a, b, c] ∈ l.tails → ¬ (a < b ∧ b < c)

-- Predicate to check if no three consecutive terms are in decreasing order
def noThreeConsecDec (l : List ℕ) : Prop :=
  ∀ (a b c : ℕ), [a, b, c] ∈ l.tails → ¬ (a > b ∧ b > c)

-- Predicate to check if no same parity numbers are consecutive
def noSameParityConsec (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), [a, b] ∈ l.tails → ¬ (Nat.even a = Nat.even b)

-- Define a predicate to check all conditions hold for a given sequence
def validSeq (l : List ℕ) : Prop :=
  noThreeConsecInc l ∧ noThreeConsecDec l ∧ noSameParityConsec l

-- Proposition stating the number of valid permutations of the sequence is 4
theorem numValidPerms : List.permutations seq |>.filter validSeq |>.length = 4 :=
  sorry

end numValidPerms_l590_590845


namespace sqrt_four_eq_two_or_neg_two_l590_590180

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 :=
by 
  sorry

end sqrt_four_eq_two_or_neg_two_l590_590180


namespace max_lambda_l590_590337

theorem max_lambda {
  λ : ℝ, 
  ∀ a b : ℝ, λ * a^2 * b^2 * (a + b)^2 ≤ (a^2 + a * b + b^2)^3
} : λ ≤ 27 / 4 :=
begin
  sorry
end

end max_lambda_l590_590337


namespace find_cost_prices_calculate_total_profit_l590_590259

-- Define the cost price of unit A and unit B as variables
variables (x y : ℕ) (m : ℕ)

-- Conditions given in the problem
axiom cost_price_relation : x = y + 20
axiom cost_purchasing_5A_6B : 5 * x = 6 * y
axiom total_cost : 120 * m + 100 * (80 - m) = 9000

-- Define the cost prices and the profit calculation
def cost_of_A := 120
def cost_of_B := 100
def profit_A := (120 * 1.5 * 0.8 - 120) * 50
def profit_B := 30 * (80 - 50)
def total_profit := profit_A + profit_B

-- The target theorems to prove
theorem find_cost_prices : x = 120 ∧ y = 100 := by
  have h1 : x = y + 20 := cost_price_relation
  have h2 : 5 * x = 6 * y := cost_purchasing_5A_6B
  sorry -- proof of solving the system of equations

theorem calculate_total_profit : total_profit = 2100 := by
  have h1 : 120 * m + 100 * (80 - m) = 9000 := total_cost
  sorry -- proof of calculating the total profit

end find_cost_prices_calculate_total_profit_l590_590259


namespace line_through_M_intersects_lines_l590_590201

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def line1 (t : ℝ) : Point3D :=
  {x := 2 - t, y := 3, z := -2 + t}

def plane1 (p : Point3D) : Prop :=
  2 * p.x - 2 * p.y - p.z - 4 = 0

def plane2 (p : Point3D) : Prop :=
  p.x + 3 * p.y + 2 * p.z + 1 = 0

def param_eq (t : ℝ) : Point3D :=
  {x := -2 + 13 * t, y := -3 * t, z := 3 - 12 * t}

theorem line_through_M_intersects_lines : 
  ∀ (t : ℝ), plane1 (param_eq t) ∧ plane2 (param_eq t) -> 
  ∃ t, param_eq t = {x := -2 + 13 * t, y := -3 * t, z := 3 - 12 * t} :=
by
  intros t h
  sorry

end line_through_M_intersects_lines_l590_590201


namespace problem_l590_590489

def vec_a : ℝ × ℝ := (5, 3)
def vec_b : ℝ × ℝ := (1, -2)
def two_vec_b : ℝ × ℝ := (2 * 1, 2 * -2)
def expected_result : ℝ × ℝ := (3, 7)

theorem problem : (vec_a.1 - two_vec_b.1, vec_a.2 - two_vec_b.2) = expected_result :=
by
  sorry

end problem_l590_590489


namespace compute_sum_evaluations_at_three_l590_590461

-- Definitions and conditions
def y := 3
def p(y : ℤ) := y^6 - y^3 - 2*y - 2

axiom factorization :
  ∃ (q1 q2 q3 q4 : ℤ[X]), 
    p(y) = q1 * q2 * q3 * q4 ∧ 
    q1.monic ∧ q2.monic ∧ q3.monic ∧ q4.monic ∧ 
    q1.degree > 0 ∧ q2.degree > 0 ∧ q3.degree > 0 ∧ q4.degree > 0 ∧ 
    ¬(∃ (r1 r2 : ℤ[X]), q1 = r1 * r2 ∧ r1.degree > 0 ∧ r2.degree > 0) ∧
    ¬(∃ (r1 r2 : ℤ[X]), q2 = r1 * r2 ∧ r1.degree > 0 ∧ r2.degree > 0) ∧
    ¬(∃ (r1 r2 : ℤ[X]), q3 = r1 * r2 ∧ r1.degree > 0 ∧ r2.degree > 0) ∧
    ¬(∃ (r1 r2 : ℤ[X]), q4 = r1 * r2 ∧ r1.degree > 0 ∧ r2.degree > 0)

-- Proof problem statement
theorem compute_sum_evaluations_at_three : 
  ∃ (q1 q2 q3 q4 : ℤ[X]), 
    p(3) = q1 * q2 * q3 * q4 ∧ 
    q1.monic ∧ q2.monic ∧ q3.monic ∧ q4.monic ∧ 
    q1.degree > 0 ∧ q2.degree > 0 ∧ q3.degree > 0 ∧ q4.degree > 0 ∧ 
    ¬(∃ (r1 r2 : ℤ[X]), q1 = r1 * r2 ∧ r1.degree > 0 ∧ r2.degree > 0) ∧
    ¬(∃ (r1 r2 : ℤ[X]), q2 = r1 * r2 ∧ r1.degree > 0 ∧ r2.degree > 0) ∧
    ¬(∃ (r1 r2 : ℤ[X]), q3 = r1 * r2 ∧ r1.degree > 0 ∧ r2.degree > 0) ∧
    ¬(∃ (r1 r2 : ℤ[X]), q4 = r1 * r2 ∧ r1.degree > 0 ∧ r2.degree > 0) ∧
    q1.eval 3 + q2.eval 3 + q3.eval 3 + q4.eval 3 = 30 :=
by
  sorry

end compute_sum_evaluations_at_three_l590_590461


namespace river_width_l590_590661

theorem river_width
  (AC CD AE EC : ℝ)
  (h_AC : AC = 40)
  (h_CD : CD = 12)
  (h_AE : AE = 24)
  (h_EC : EC = 16)
  (similar_ABD_AEC : ∀ (AB : ℝ), triangle_similar (AB, CD) (AE, EC)) :
  ∃ AB, AB = 18 :=
by
  have h_sim_ratio : AE / EC = 3 / 2 := by sorry
  have h_CD_ratio : CD = AB * 2 / 3 := by sorry
  use 18
  have h_AB : AB * 2 / 3 = 12 := by sorry
  linarith

end river_width_l590_590661


namespace cost_per_square_meter_is_3_l590_590652
-- Import necessary library

noncomputable def lawn_dimensions : ℝ × ℝ := (80, 60)
noncomputable def road_width : ℝ := 10
noncomputable def total_cost : ℝ := 3900

def road_area (length breadth width : ℝ) : ℝ :=
  length * width + breadth * width - width^2

def cost_per_square_meter (total_cost area : ℝ) : ℝ :=
  total_cost / area

-- The statement to be proven
theorem cost_per_square_meter_is_3 :
  cost_per_square_meter total_cost (road_area 80 60 10) = 3 := by
  sorry

end cost_per_square_meter_is_3_l590_590652


namespace descent_time_on_moving_escalator_standing_l590_590263

theorem descent_time_on_moving_escalator_standing (l v_mont v_ek t : ℝ)
  (H1 : l / v_mont = 42)
  (H2 : l / (v_mont + v_ek) = 24)
  : t = 56 := by
  sorry

end descent_time_on_moving_escalator_standing_l590_590263


namespace find_g_neg3_l590_590926

variable (g : ℚ → ℚ)

-- Given condition
axiom condition : ∀ x : ℚ, x ≠ 0 → 4 * g (1/x) + (3 * g x) / x = 3 * x^2

-- Theorem statement
theorem find_g_neg3 : g (-3) = -27 / 2 := 
by 
  sorry

end find_g_neg3_l590_590926


namespace incorrect_statement_2_l590_590781

variable {a_n : ℕ → ℝ}
variable {q : ℝ}
variable {a_1 : ℝ}

-- Given conditions
-- The sequence is geometric with a common ratio q
def is_geometric (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a_n (n+1) = a_n n * q

-- All terms are positive
def all_positive (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < a_n n

-- Sum of the first n terms
def sum_n (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a_n i

-- Statement (2), which we need to prove is incorrect
def statement_2 (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ a : ℝ, (∀ n, a_n n < a) → q ∈ (0,1)

theorem incorrect_statement_2 (a_n : ℕ → ℝ) (q : ℝ) (a_1 : ℝ) 
  (h1 : is_geometric a_n q) (h2 : all_positive a_n) :
  ¬ statement_2 a_n q :=
by
  sorry

end incorrect_statement_2_l590_590781


namespace square_division_QT_length_l590_590921

theorem square_division_QT_length :
  ∀ {PQ RS : ℝ} {T U : ℝ},
    PQ = 4 ∧ RS = 4 ∧ 0 ≤ T ∧ T ≤ PQ ∧  0 ≤ U ∧ U ≤ RS ∧ 4 * 4 = 4 * 4 / 2 + 4 * 4 / 2 ->
    T = 2 -> 
    U = 2 -> 
    QT = 2 * sqrt 3 :=
by {
  sorry
}

end square_division_QT_length_l590_590921


namespace triangle_coloring_l590_590288

theorem triangle_coloring (T : Type*) [fintype T] 
  (vertices: fin 9000001 → T) 
  (color: T → fin 3) :
  ∃ (a b c : T), (color a = color b) ∧ (color b = color c) ∧ (color c = color a) ∧ 
  (∃ (triangle_abc : fin 9000000), -- we use fin 9000000 to represent the subdivision 
    sorry) := 
sorry

end triangle_coloring_l590_590288


namespace chord_length_problem_l590_590213

noncomputable def length_of_chord : ℝ :=
  let radius : ℝ := 5
  let distance_from_center_to_line : ℝ := (3 * Real.sqrt 2) / 2
  let chord_length := Real.sqrt (4 * (radius^2 - (distance_from_center_to_line^2) / 2))
  in chord_length

theorem chord_length_problem 
  (circle_eq : ∀ (x y : ℝ), (x - 3)^2 + (y + 1)^2 = 25)
  (parametric_line : ∀ (t : ℝ), x = -2 + t ∧ y = 1 - t)
  : length_of_chord = 8 :=
  by
    -- The proof is omitted.
    sorry

end chord_length_problem_l590_590213


namespace convex_cyclic_quadrilaterals_perimeter_40_l590_590411

theorem convex_cyclic_quadrilaterals_perimeter_40 :
  ∃ (n : ℕ), n = 750 ∧ ∀ (a b c d : ℕ), a + b + c + d = 40 → a ≥ b → b ≥ c → c ≥ d →
  (a < b + c + d) ∧ (b < a + c + d) ∧ (c < a + b + d) ∧ (d < a + b + c) :=
sorry

end convex_cyclic_quadrilaterals_perimeter_40_l590_590411


namespace regular_polygon_sides_l590_590729

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l590_590729


namespace sum_a1_a3_a5_l590_590939

-- Definitions
variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)

-- Conditions
axiom initial_condition : a 1 = 16
axiom relationship_ak_bk : ∀ k, b k = a k / 2
axiom ak_next : ∀ k, a (k + 1) = a k + 2 * (b k)

-- Theorem Statement
theorem sum_a1_a3_a5 : a 1 + a 3 + a 5 = 336 :=
by
  sorry

end sum_a1_a3_a5_l590_590939


namespace initial_weight_of_mixture_eq_20_l590_590269

theorem initial_weight_of_mixture_eq_20
  (W : ℝ) (h1 : 0.1 * W + 4 = 0.25 * (W + 4)) :
  W = 20 :=
by
  sorry

end initial_weight_of_mixture_eq_20_l590_590269


namespace difference_greater_than_one_l590_590998

theorem difference_greater_than_one (n : ℕ) (a : Fin n → ℝ) (k : ℝ)
  (h_pos : ∀ i, 0 < a i)
  (h1 : ∑ i, a i = 3 * k)
  (h2 : ∑ i, (a i)^2 = 3 * k^2)
  (h3 : ∑ i, (a i)^3 > 3 * k^3 + k) :
  ∃ i j, i ≠ j ∧ |a i - a j| > 1 :=
sorry

end difference_greater_than_one_l590_590998


namespace find_t_range_l590_590004

theorem find_t_range :
  (∀ (x : ℝ), abs x ≤ 1 → t + 1 > (t^2 - 4) * x) →
  ( ∃ (t : ℝ), (sqrt 13 - 1) / 2 < t ∧ t < (sqrt 21 + 1) / 2 ) :=
by
  sorry

end find_t_range_l590_590004


namespace construct_triangle_from_conditions_l590_590718

-- Given parameters
variables {m s d : ℝ} 

-- Hypothesis stating that m, s, and d are positive real numbers
hypothesis (hm : m > 0) (hs : s > 0) (hd : d > 0)

-- Hypothesis stating the conditions for constructing the triangle
hypothesis (h : exists (A B C : Type), 
  altitude A m ∧ median A B s ∧ distance A orthocenter d)

-- Theorem statement asserting that such a triangle can be constructed
theorem construct_triangle_from_conditions 
  (m s d : ℝ) (hm : m > 0) (hs : s > 0) (hd : d > 0) 
  (h : exists (A B C : Type), altitude A m ∧ median A B s ∧ distance A orthocenter d) :
  ∃ (A B C : Type), altitude A m ∧ median A B s ∧ distance A orthocenter d :=
sorry

end construct_triangle_from_conditions_l590_590718


namespace probability_different_tens_digit_l590_590532

open Nat

theorem probability_different_tens_digit :
  let total_ways := choose 70 6,
      favorable_ways := 7 * 10^6
  in 
    (favorable_ways : ℝ) / total_ways = (2000 / 3405864 : ℝ) :=
by
  have h1 : total_ways = 70.choose 6 := rfl
  have h2 : favorable_ways = 7 * 10^6 := rfl
  rw [h1, h2]
  sorry

end probability_different_tens_digit_l590_590532


namespace problem1_problem2_problem3_l590_590348

def A : Set ℝ := Set.Icc (-1) 1
def B : Set ℝ := Set.Icc (-2) 2
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1
def g (a m x : ℝ) : ℝ := 2 * abs (x - a) - x^2 - m * x

theorem problem1 (m : ℝ) : (∀ x, f m x ≤ 0 → x ∈ A) → m ∈ Set.Icc (-1) 1 :=
sorry

theorem problem2 (f_eq : ∀ x, f (-4) (1-x) = f (-4) (1+x)) : 
  Set.range (f (-4) ∘ id) ⊆ Set.Icc (-3) 15 :=
sorry

theorem problem3 (a : ℝ) (m : ℝ) :
  (a ≤ -1 → ∃ x, f m x + g a m x = -2*a - 2) ∧
  (-1 < a ∧ a < 1 → ∃ x, f m x + g a m x = a^2 - 1) ∧
  (a ≥ 1 → ∃ x, f m x + g a m x = 2*a - 2) :=
sorry

end problem1_problem2_problem3_l590_590348


namespace circle_tangent_intersection_angle_l590_590931

theorem circle_tangent_intersection_angle
  (k1 k2 : Circle)
  (A B : Point)
  (M N : Point)
  (t : Line)
  (h1 : intersect k1 k2 = {A, B})
  (h2 : tangent t k1 M)
  (h3 : tangent t k2 N)
  (h4 : perpendicular t (line_through A M))
  (h5 : distance M N = 2 * distance A M) :
  angle N M B = 45 := 
sorry

end circle_tangent_intersection_angle_l590_590931


namespace closest_integer_to_cube_root_of_150_l590_590607

theorem closest_integer_to_cube_root_of_150 : ∃ (n : ℤ), abs ((n: ℝ)^3 - 150) ≤ abs (((n + 1 : ℤ) : ℝ)^3 - 150) ∧
  abs ((n: ℝ)^3 - 150) ≤ abs (((n - 1 : ℤ) : ℝ)^3 - 150) ∧ n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590607


namespace general_formula_sum_Tn_l590_590055

-- Conditions
def a_seq (n : ℕ) : ℕ := if n = 1 then 0 else n - 1
def S_n (n : ℕ) : ℕ := (finset.range n).sum (λ i, a_seq (i + 1))

-- Given conditions
axiom a2 : a_seq 2 = 1
axiom S_n_cond (n : ℕ) : 2 * S_n n = n * a_seq n

-- (1) General formula for {a_n}
theorem general_formula (n : ℕ) : a_seq n = n - 1 :=
by sorry

-- (2) Sum of the first n terms of the sequence {(\frac{a_n + 1}{2^n})}
def T_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, (a_seq (k + 1) + 1 : ℝ) / (2 : ℝ) ^ (k + 1))

theorem sum_Tn (n : ℕ) : T_n n = 2 - (n + 2) / (2 : ℝ) ^ n :=
by sorry

end general_formula_sum_Tn_l590_590055


namespace regular_polygon_sides_l590_590728

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l590_590728


namespace circumcenter_lies_on_BF_l590_590036

/-
In the acute-angled triangle ABC, BH is the altitude, and D and E are the midpoints 
of sides AB and AC, respectively. Let F be the reflection of point H across line ED. 
Prove that the circumcenter of triangle ABC lies on line BF.
-/

variables {A B C D E F H O : Type}
variables [IsTriangle A B C] [IsAcuteTriangle A B C]
variables [IsMidpoint D A B] [IsMidpoint E A C]
variables [IsAltitude BH B C] [IsReflection F H (Line.mk E D)]
variables [IsCircumcenter O A B C]

theorem circumcenter_lies_on_BF :
  LiesOnLine O (Line.mk B F) :=
sorry

end circumcenter_lies_on_BF_l590_590036


namespace irr_sqrt6_l590_590287

open Real

theorem irr_sqrt6 : ¬ ∃ (q : ℚ), (↑q : ℝ) = sqrt 6 := by
  sorry

end irr_sqrt6_l590_590287


namespace equal_distance_after_3_hours_l590_590198

-- Define the speeds of the cyclists
variables {v₁ v₂ v₃ : ℝ}

-- Define the condition P₁: After 1.5 hours, the first cyclist is equidistant from the second and third cyclists
def P₁ : Prop :=
  1.5 * (v₁ - v₂) = 1 - 1.5 * (v₁ + v₃)

-- Define the condition P₂: After 2 hours, the third cyclist is equidistant from the first and second cyclists
def P₂ : Prop :=
  1 - 2 * (v₂ + v₃) = 2 * (v₁ + v₃) - 1

-- Define the target: to prove the second cyclist was equidistant from the first and third cyclists after 3 hours
theorem equal_distance_after_3_hours (hv₃ : v₃ = (1 - v₁ - v₂) / 2) (h₂₁: 3 * v₁ - 1.5 * v₂ + 1.5 * v₃ = 1) : 
  ∃ t : ℝ, t = 3 ∧ (v₂ + v₃) * t - 1 = (v₁ - v₂) * t :=
by
  existsi (3 : ℝ)
  -- Provide proof of the actual result
  -- Here's where you would provide the proof steps to establish the time t = 3 
  -- as the time at which the second cyclist is equidistant from the first and third
  have hv₃ : v₃ = (1 - v₁ - v₂) / 2,
  have h₂₁ : 3 * v₁ - 1.5 * v₂ + 1.5 * v₃ = 1,
  sorry

end equal_distance_after_3_hours_l590_590198


namespace length_segment_AB_l590_590444

-- Define the line l in parametric form
def line_l (t : ℝ) : ℝ × ℝ := 
  (1 - (Real.sqrt 2 / 2) * t, 2 + (Real.sqrt 2 / 2) * t)

-- Define the parabola
def parabola (x : ℝ) : ℝ := 
  Real.sqrt (4 * x)

theorem length_segment_AB : 
  let A := (1 : ℝ, 2 : ℝ),
      B := (9 : ℝ, -6 : ℝ) in
  (A.1, A.2) ∈ SetOf (λ p : ℝ × ℝ, parabola p.1 = p.2) ∧
  (B.1, B.2) ∈ SetOf (λ p : ℝ × ℝ, parabola p.1 = p.2) ∧
  ∀ t : ℝ, (line_l t).1 = A.1 ∧ (line_l t).2 = A.2 
  → ∀ t : ℝ, (line_l t).1 = B.1 ∧ (line_l t).2 = B.2
  → dist A B = 8 * Real.sqrt 2 :=
by
  sorry

end length_segment_AB_l590_590444


namespace transformation_impossible_l590_590642

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def last_digit (n : ℕ) : ℕ :=
  n % 10

def transform (n : ℕ) (i : ℕ) : ℕ :=
  let digits := n.digits
  let new_digit := last_digit (digit_sum n)
  let new_digits := digits.update_nth i new_digit
  nat.of_digits new_digits

theorem transformation_impossible : 
  ∀ n, n = 133355555 → (∀ m, (∃ i, m = transform n i) → m ≠ 123456789) := 
by {
  intros n hnm m hi,
  sorry
}

end transformation_impossible_l590_590642


namespace convert_point_8_8_to_polar_l590_590720

def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2) in
  let theta := if x = y then real.pi / 4 else sorry in -- Only handle this specific case
  (r, theta)

theorem convert_point_8_8_to_polar :
  rectangular_to_polar 8 8 = (8 * real.sqrt 2, real.pi / 4) :=
by
  unfold rectangular_to_polar
  rw [pow_two, pow_two, mul_self_add_mul_self_eq_mul_self_add_mul_self_eq_add, real.sqrt_eq_rfl, pow_two]
  simp
  sorry

end convert_point_8_8_to_polar_l590_590720


namespace length_of_AB_l590_590901

theorem length_of_AB (A B P Q : ℝ) 
  (hp : 0 < P) (hp' : P < 1) 
  (hq : 0 < Q) (hq' : Q < 1) 
  (H1 : P = 3 / 7) (H2 : Q = 5 / 12)
  (H3 : P * (1 - Q) + Q * (1 - P) = 4) : 
  (B - A) = 336 / 11 :=
by
  sorry

end length_of_AB_l590_590901


namespace range_of_x_l590_590572

variable (x : ℝ)

theorem range_of_x (h1 : 2 - x > 0) (h2 : x - 1 ≥ 0) : 1 ≤ x ∧ x < 2 := by
  sorry

end range_of_x_l590_590572


namespace necessary_condition_not_sufficient_condition_main_l590_590776

example (x : ℝ) : (x^2 - 3 * x > 0) → (x > 4) ∨ (x < 0 ∧ x > 0) := by
  sorry

theorem necessary_condition (x : ℝ) :
  (x^2 - 3 * x > 0) → (x > 4) :=
by
  sorry

theorem not_sufficient_condition (x : ℝ) :
  ¬ (x > 4) → (x^2 - 3 * x > 0) :=
by
  sorry

theorem main (x : ℝ) :
  (x^2 - 3 * x > 0) ↔ ¬ (x > 4) :=
by
  sorry

end necessary_condition_not_sufficient_condition_main_l590_590776


namespace remainder_of_3024_l590_590338

theorem remainder_of_3024 (M : ℤ) (hM1 : M = 3024) (h_condition : ∃ k : ℤ, M = 24 * k + 13) :
  M % 1821 = 1203 :=
by
  sorry

end remainder_of_3024_l590_590338


namespace consecutive_primes_sum_square_is_prime_l590_590342

-- Defining what it means for three numbers to be consecutive primes
def consecutive_primes (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  ((p < q ∧ q < r) ∨ (p < q ∧ q < r ∧ r < p) ∨ 
   (r < p ∧ p < q) ∨ (q < p ∧ p < r) ∨ 
   (q < r ∧ r < p) ∨ (r < q ∧ q < p))

-- Defining our main problem statement
theorem consecutive_primes_sum_square_is_prime :
  ∀ p q r : ℕ, consecutive_primes p q r → Nat.Prime (p^2 + q^2 + r^2) ↔ (p = 3 ∧ q = 5 ∧ r = 7) :=
by
  -- Sorry is used to skip the proof.
  sorry

end consecutive_primes_sum_square_is_prime_l590_590342


namespace circle_condition_l590_590307

theorem circle_condition (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 4 * m * x - 2 * y + 5 * m = 0) ↔ (m < 1 / 4 ∨ m > 1) :=
sorry

end circle_condition_l590_590307


namespace color_white_cells_l590_590279

-- Define the problem conditions
variables {Cell : Type} (is_black : Cell → Prop) (is_white : Cell → Prop) (adjacent : Cell → Cell → Prop)
variable [DecidableEq Cell]

-- Ensure the plane is partitioned into cells and the adjacency relation is symmetric
axiom partitioned_plane (c : Cell) : is_black c ∨ is_white c
axiom finite_black_cells : Finite { c : Cell | is_black c }
axiom symmetric_adj : ∀ a b, adjacent a b → adjacent b a

-- Condition: Each black cell has an even number of white neighbors
axiom even_white_neighbors (b : Cell) (h : is_black b) :
  Even (card { w : Cell | is_white w ∧ adjacent b w })

-- Prove the existence of a valid coloring for white cells
theorem color_white_cells :
  ∃ (color : Cell → Prop), 
    (∀ c, is_white c → (color c ∨ ¬(color c))) ∧
    (∀ b, is_black b → Even (card { w : Cell | is_white w ∧ adjacent b w ∧ color w })
     ∧ Even (card { w : Cell | is_white w ∧ adjacent b w ∧ ¬(color w) })) :=
sorry

end color_white_cells_l590_590279


namespace necessary_but_not_sufficient_l590_590135

theorem necessary_but_not_sufficient
  (x y : ℝ) :
  (x^2 + y^2 ≤ 2*x → x^2 + y^2 ≤ 4) ∧ ¬ (x^2 + y^2 ≤ 4 → x^2 + y^2 ≤ 2*x) :=
by {
  sorry
}

end necessary_but_not_sufficient_l590_590135


namespace correct_statements_l590_590350

theorem correct_statements (α β : Plane) (h_distinct : α ≠ β) (l : Line) (h_l_outside : ¬l ∈ α) :
  (∀ (a b : Line), a ∈ α ∧ b ∈ α ∧ a ≠ b → (∀ (c d : Line), c ∈ β ∧ d ∈ β → (a ∥ c ∧ b ∥ d)) → α ∥ β) ∧
  (∀ (m : Line), m ∈ α → l ∥ m → l ∥ α) ∧
  (¬(∀ (p : Line), p ∈ α → p ⊥ l → α ⊥ β)) ∧
  ¬(∀ (p q : Line), p ∈ α ∧ q ∈ α ∧ p ≠ q → (l ⊥ p ∧ l ⊥ q) → l ⊥ α) :=
by
  sorry

end correct_statements_l590_590350


namespace closest_integer_to_cube_root_of_150_l590_590602

theorem closest_integer_to_cube_root_of_150 : 
  ∃ (n : ℤ), ∀ m : ℤ, abs (150 - 5 ^ 3) < abs (150 - m ^ 3) → n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590602


namespace find_b_of_tangent_line_l590_590167

theorem find_b_of_tangent_line
  (k a b : ℝ)
  (h1 : ∀ x y : ℝ, y = k * x + b → y = x^3 + a * x + 1 → x = 2 ∧ y = 3)
  (h2 : ∀ x : ℝ, hasDerivAt (λ x, x^3 + a * x + 1) (3 * x^2 + a) x) :
  b = -15 :=
by
  sorry

end find_b_of_tangent_line_l590_590167


namespace jellybean_count_l590_590254

def black_beans : Nat := 8
def green_beans : Nat := black_beans + 2
def orange_beans : Nat := green_beans - 1
def total_jelly_beans : Nat := black_beans + green_beans + orange_beans

theorem jellybean_count : total_jelly_beans = 27 :=
by
  -- proof steps would go here.
  sorry

end jellybean_count_l590_590254


namespace train_passes_bus_time_l590_590232

theorem train_passes_bus_time :
  let length_of_train := 220 -- in meters
  let speed_of_train := 90 * 1000 / 3600 -- in m/s
  let speed_of_bus := 60 * 1000 / 3600 -- in m/s
  let relative_speed := speed_of_train + speed_of_bus -- in m/s
  length_of_train / relative_speed ≈ 5.28 := 
by
  -- Define the given constants
  let length_of_train : ℝ := 220
  let speed_of_train : ℝ := 90 * 1000 / 3600
  let speed_of_bus : ℝ := 60 * 1000 / 3600 
  let relative_speed : ℝ := speed_of_train + speed_of_bus
  -- Define the calculated time
  let time : ℝ := length_of_train / relative_speed
  -- Assert the approximation
  have h : time ≈ 5.28, by sorry
  exact h

end train_passes_bus_time_l590_590232


namespace equation_of_line_through_point_and_slope_angle_l590_590560

theorem equation_of_line_through_point_and_slope_angle (α : ℝ) (h : tan α = 4/3) :
  ∃ a b c : ℝ, a * 2 + b * 1 + c = 0 ∧ 4x - 3y - 5 = 0 :=
by
  sorry

end equation_of_line_through_point_and_slope_angle_l590_590560


namespace area_triangle_ABC_l590_590838

-- Define the triangle and its properties
structure Triangle where
  AB : ℝ
  AC : ℝ
  ∠A : ℝ

-- The given triangle ABC with the specified conditions
def triangle_ABC : Triangle :=
  { AB := 1, AC := Real.sqrt 3, ∠A := Real.pi / 3 }

-- The area function of a triangle given the properties
def area (t : Triangle) : ℝ :=
  1 / 2 * t.AB * t.AC * Real.sin t.∠A

-- The theorem to prove: given the conditions, the area of triangle ABC is 3/4
theorem area_triangle_ABC : area triangle_ABC = 3 / 4 :=
by
  sorry

end area_triangle_ABC_l590_590838


namespace length_BC_l590_590896

def circle_radius := 10
def cos_alpha := 2 / 5
def alpha (A O M B : Type) := sorry
def A O M B C : Type := sorry -- Assuming these represent points in the geometry

theorem length_BC (A O M B C : Type) (r : ℝ) (cos_alpha : ℝ) (h1 : r = circle_radius) (h2 : cos_alpha = 2/5) (h3 : α : angle) (h4 : angle_AMB : α) : 
  length_BC = 8 := 
by
  -- Necessary geometric constructions and conditions assumed as sorry
  sorry

end length_BC_l590_590896


namespace triangle_divides_into_four_congruent_l590_590906

theorem triangle_divides_into_four_congruent (A B C D E F : Type*)
  [Triangle A B C]
  (D_midpoint : Midpoint D A B)
  (E_midpoint : Midpoint E B C)
  (F_midpoint : Midpoint F C A)
  (DE : MedialLine D E)
  (EF : MedialLine E F)
  (FD : MedialLine F D) :
  CongruentTriangles (Triangle D E F) (Triangle A D E) (Triangle D F C) (Triangle E F B) (Triangle F D C) :=
sorry

end triangle_divides_into_four_congruent_l590_590906


namespace find_number_of_observations_l590_590195

-- Define the initial conditions
variables (n : ℕ)
variables (mean_orig mean_new : ℚ)
variables (wrong correct : ℤ)

-- Assume the conditions
axiom mean_orig_value : mean_orig = 41
axiom mean_new_value : mean_new = 41.5
axiom wrong_value : wrong = 23
axiom correct_value : correct = 48

-- Prove the number of observations
theorem find_number_of_observations (n : ℕ)
  (h_mean_orig : mean_orig_value)
  (h_mean_new : mean_new_value)
  (h_wrong : wrong_value)
  (h_correct : correct_value) :
  (41 * n - 23 + 48) / n = 41.5 → n = 50 :=
by
  sorry

end find_number_of_observations_l590_590195


namespace chad_odd_jobs_money_l590_590705

-- Define all the conditions as given in the problem.
def total_earnings (total: ℝ): Prop := 
  0.4 * total = 460

def money_from_mowing_yards : ℝ := 600
def money_from_birthday_holidays : ℝ := 250
def money_from_selling_video_games : ℝ := 150

-- Define the total money Chad saved.
def saved_money : ℝ := 460

-- The goal is to prove that the money Chad made from odd jobs is $150.
theorem chad_odd_jobs_money (total_earnings : ℝ) 
    (money_odd_jobs : ℝ):
  total_earnings total_earnings →
  (money_from_mowing_yards + money_from_birthday_holidays + 
  money_from_selling_video_games + money_odd_jobs = total_earnings) →
  money_odd_jobs = 150 :=
by 
  intros h1 h2
  sorry

end chad_odd_jobs_money_l590_590705


namespace probability_of_different_tens_digits_l590_590537

open Finset

-- Define the basic setup
def integers (n : ℕ) : Finset ℕ := {i in (range n) | i ≥ 10 ∧ i ≤ 79}

def tens_digit (n : ℕ) : ℕ := n / 10

def six_integers_with_different_tens_digits (s : Finset ℕ) : Prop :=
  s.card = 6 ∧ (s.map ⟨tens_digit, by simp⟩).card = 6

def favorable_ways : ℕ :=
  7 * 10^6

def total_ways : ℕ :=
  nat.choose 70 6

noncomputable def probability : ℚ :=
  favorable_ways / total_ways

-- The main statement
theorem probability_of_different_tens_digits :
  ∀ (s : Finset ℕ), six_integers_with_different_tens_digits s → 
  probability = 175 / 2980131 :=
begin
  intros s h,
  sorry
end

end probability_of_different_tens_digits_l590_590537


namespace yellow_yellow_pairs_count_l590_590435

theorem yellow_yellow_pairs_count :
  ∀ (blue_students yellow_students total_students total_pairs blue_blue_pairs yellow_yellow_pairs : ℕ),
    blue_students = 63 →
    yellow_students = 81 →
    total_students = blue_students + yellow_students →
    total_pairs = 72 →
    blue_blue_pairs = 27 →
    yellow_yellow_pairs = (yellow_students - (blue_students - 2 * blue_blue_pairs)) / 2 →
    yellow_yellow_pairs = 36 :=
by
  intros blue_students yellow_students total_students total_pairs blue_blue_pairs yellow_yellow_pairs
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, add_comm] at h3
  rw h3 at h6
  norm_num at h6
  exact h6

end yellow_yellow_pairs_count_l590_590435


namespace exists_zero_total_acceleration_in_region_IV_l590_590278

def vec (dim : Nat) := Fin dim → ℝ

def tangential_acceleration (p : vec 2) : vec 2 := sorry
def horizontal_acceleration (F : vec 2) : vec 2 := F
def centripetal_acceleration (p : vec 2) : vec 2 := sorry

def total_acceleration (p : vec 2) (F : vec 2) : vec 2 :=
  tangential_acceleration p + horizontal_acceleration F + centripetal_acceleration p

def region_IV (p : vec 2) : Prop := sorry -- Mathematical definition of Region IV

theorem exists_zero_total_acceleration_in_region_IV (F : vec 2) (h : F ≠ 0) :
  ∃ p : vec 2, region_IV p ∧ total_acceleration p F = 0 :=
sorry

end exists_zero_total_acceleration_in_region_IV_l590_590278


namespace polynomial_sum_is_integer_l590_590137

-- Define the integer polynomial and the integers a and b
variables (f : ℤ[X]) (a b : ℤ)

-- The theorem statement
theorem polynomial_sum_is_integer :
  ∃ c : ℤ, f.eval (a - real.sqrt b) + f.eval (a + real.sqrt b) = c :=
sorry

end polynomial_sum_is_integer_l590_590137


namespace range_of_a_l590_590426

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 1) * x < 1 ↔ x > 1 / (a - 1)) → a < 1 :=
by 
  sorry

end range_of_a_l590_590426


namespace evaluate_expression_l590_590297

theorem evaluate_expression : 8 * ((1 : ℚ) / 3)^3 - 1 = -19 / 27 := by
  sorry

end evaluate_expression_l590_590297


namespace expected_lifetime_of_flashlight_at_least_4_l590_590089

-- Definitions for the lifetimes of the lightbulbs
variable (ξ η : ℝ)

-- Condition: The expected lifetime of the red lightbulb is 4 years.
axiom E_η_eq_4 : 𝔼[η] = 4

-- Definition stating the lifetime of the flashlight
def T := max ξ η

theorem expected_lifetime_of_flashlight_at_least_4 
  (h : 𝔼η = 4) :
  𝔼[max ξ η] ≥ 4 :=
by {
  sorry
}

end expected_lifetime_of_flashlight_at_least_4_l590_590089


namespace average_age_of_students_l590_590023

theorem average_age_of_students (A : ℝ) : 
  (40 : ℝ) * A + 56 = 41 * 16 → A = 15 :=
by
  intro h
  have h1 : 40 * A + 56 = 656 := by rwa [(41 * 16).symm] at h
  have h2 : 40 * A = 600 := by linarith
  have h3 : A = 15 := by linarith
  exact h3

end average_age_of_students_l590_590023


namespace ratio_of_numbers_l590_590579

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_numbers_l590_590579


namespace competition_results_l590_590203

structure MatchScore :=
(scores : List (Nat × Nat))

structure TeamRecord :=
(wins : Nat)
(losses : Nat)

inductive Team
| First
| Second
| Third
| Fourth
| Fifth
| Sixth

def matchResult :=
  [⟨Team.First, Team.Third, (21, 19)⟩,
   ⟨Team.First, Team.Fourth, (21, 9)⟩,
   ⟨Team.First, Team.Second, (14, 21)⟩,
   ⟨Team.First, Team.Fifth, (22, 24)⟩,
   ⟨Team.First, Team.Sixth, (15, 21)⟩,
   ⟨Team.Second, Team.Third, (24, 22)⟩,
   ⟨Team.Second, Team.Fourth, (21, 9)⟩,
   ⟨Team.Second, Team.Fifth, (21, 23)⟩,
   ⟨Team.Second, Team.Sixth, (18, 21)⟩,
   ⟨Team.Third, Team.First, (12, 15)⟩]

def roundRobin (m : List (Team × Team × (Nat × Nat))) : List (TeamRecord) :=
  [-- Fill in the details to parse match results and calculate wins/losses for each team.
  ]

theorem competition_results :
  let records := roundRobin matchResult in
  (getRecord Team.First records = ⟨2, 3⟩) ∧
  (getRecord Team.Second records = ⟨4, 1⟩) ∧
  (getRecord Team.Third records = ⟨1, 4⟩) ∧
  (getRecord Team.Fourth records = ⟨0, 5⟩) ∧
  (getRecord Team.Fifth records = ⟨4, 1⟩) ∧
  (getRecord Team.Sixth records = ⟨4, 1⟩) ∧
  (strongerTeam Team.Fourth Team.Fifth = Team.Fifth) ∧
  (possibleScores B' = [(21, 19), (20, 18)]) ∧
  (strongestTeams records = [Team.Fifth, Team.Sixth]).
Proof
  sorry

end competition_results_l590_590203


namespace expected_lifetime_flashlight_l590_590093

noncomputable def xi : ℝ := sorry
noncomputable def eta : ℝ := sorry

def T : ℝ := max xi eta

axiom E_eta_eq_4 : E eta = 4

theorem expected_lifetime_flashlight : E T ≥ 4 :=
by
  -- The solution will go here
  sorry

end expected_lifetime_flashlight_l590_590093


namespace pete_total_miles_l590_590905

-- Definitions based on conditions
def flip_step_count : ℕ := 89999
def steps_full_cycle : ℕ := 90000
def total_flips : ℕ := 52
def end_year_reading : ℕ := 55555
def steps_per_mile : ℕ := 1900

-- Total steps Pete walked
def total_steps_pete_walked (flips : ℕ) (end_reading : ℕ) : ℕ :=
  flips * steps_full_cycle + end_reading

-- Total miles Pete walked
def total_miles_pete_walked (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

-- Given the parameters, closest number of miles Pete walked should be 2500
theorem pete_total_miles : total_miles_pete_walked (total_steps_pete_walked total_flips end_year_reading) steps_per_mile = 2500 :=
by
  sorry

end pete_total_miles_l590_590905


namespace odd_integers_sum_l590_590983

theorem odd_integers_sum (n : ℕ) (h1 : n = 25) (h2 : ∃ k : ℕ, n = k ∧ (∑ i in range(k+1), 2*i + 1) = 625) : n = 25 :=
by
  sorry

end odd_integers_sum_l590_590983


namespace two_digit_numbers_non_repeating_l590_590821

-- The set of available digits is given as 0, 1, 2, 3, 4
def digits : List ℕ := [0, 1, 2, 3, 4]

-- Ensure the tens place digits are subset of 1, 2, 3, 4 (exclude 0)
def valid_tens : List ℕ := [1, 2, 3, 4]

theorem two_digit_numbers_non_repeating :
  let num_tens := valid_tens.length
  let num_units := (digits.length - 1)
  num_tens * num_units = 16 :=
by
  -- Observe num_tens = 4, since valid_tens = [1, 2, 3, 4]
  -- Observe num_units = 4, since digits.length = 5 and we exclude the tens place digit
  sorry

end two_digit_numbers_non_repeating_l590_590821


namespace intersection_is_correct_l590_590792

-- Define sets A and B
def setA : Set ℝ := { x | Math.log10 (x - 1) ≤ 0 }
def setB : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- The statement of the problem
theorem intersection_is_correct : (setA ∩ setB) = { x | 1 < x ∧ x ≤ 2 } :=
by
  -- This is where the proof would go
  sorry

end intersection_is_correct_l590_590792


namespace ratio_diff_squares_eq_16_l590_590290

theorem ratio_diff_squares_eq_16 (x y : ℕ) (h1 : x + y = 16) (h2 : x ≠ y) :
  (x^2 - y^2) / (x - y) = 16 :=
by
  sorry

end ratio_diff_squares_eq_16_l590_590290


namespace divide_square_into_rectangles_l590_590722

theorem divide_square_into_rectangles :
  ∃ (rectangles : list (ℕ × ℕ)), 
    rectangles.length = 5 ∧ 
    ∀ (l : ℕ), l ∈ (rectangles.bind (λ r, [r.1, r.2])) →
    (1 ≤ l ∧ l ≤ 13) ∧
    (rectangles.bind (λ r, [r.1, r.2])).nodup :=
sorry

end divide_square_into_rectangles_l590_590722


namespace tens_digit_N_pow_20_is_7_hundreds_digit_N_pow_200_is_3_l590_590475

def tens_digit_N_pow_20 (N : ℕ) : Nat :=
if (N % 2 = 0 ∧ N % 10 ≠ 0) then
  if (N % 5 = 1 ∨ N % 5 = 2 ∨ N % 5 = 3 ∨ N % 5 = 4) then
    (N^20 % 100) / 10  -- tens digit of last two digits
  else
    sorry  -- N should be in form of 5k±1 or 5k±2
else
  sorry  -- N not satisfying conditions

def hundreds_digit_N_pow_200 (N : ℕ) : Nat :=
if (N % 2 = 0 ∧ N % 10 ≠ 0) then
  (N^200 % 1000) / 100  -- hundreds digit of the last three digits
else
  sorry  -- N not satisfying conditions

theorem tens_digit_N_pow_20_is_7 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) : 
  tens_digit_N_pow_20 N = 7 := sorry

theorem hundreds_digit_N_pow_200_is_3 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) : 
  hundreds_digit_N_pow_200 N = 3 := sorry

end tens_digit_N_pow_20_is_7_hundreds_digit_N_pow_200_is_3_l590_590475


namespace probability_of_different_tens_digits_l590_590536

open Finset

-- Define the basic setup
def integers (n : ℕ) : Finset ℕ := {i in (range n) | i ≥ 10 ∧ i ≤ 79}

def tens_digit (n : ℕ) : ℕ := n / 10

def six_integers_with_different_tens_digits (s : Finset ℕ) : Prop :=
  s.card = 6 ∧ (s.map ⟨tens_digit, by simp⟩).card = 6

def favorable_ways : ℕ :=
  7 * 10^6

def total_ways : ℕ :=
  nat.choose 70 6

noncomputable def probability : ℚ :=
  favorable_ways / total_ways

-- The main statement
theorem probability_of_different_tens_digits :
  ∀ (s : Finset ℕ), six_integers_with_different_tens_digits s → 
  probability = 175 / 2980131 :=
begin
  intros s h,
  sorry
end

end probability_of_different_tens_digits_l590_590536


namespace plane_equation_l590_590270

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + s + 2 * t, 4 - 2 * s, 1 - s + t)

def normal_vector : ℝ × ℝ × ℝ :=
  (-2, -3, 4)

def point_on_plane : ℝ × ℝ × ℝ :=
  (3, 4, 1)

theorem plane_equation : ∀ (x y z : ℝ),
  (∃ (s t : ℝ), (x, y, z) = parametric_plane s t) ↔
  2 * x + 3 * y - 4 * z - 14 = 0 :=
sorry

end plane_equation_l590_590270


namespace closest_integer_to_cuberoot_150_l590_590598

theorem closest_integer_to_cuberoot_150 : 
  let cube5 := 5^3 in 
  let cube6 := 6^3 in 
  let midpoint := (cube5 + cube6) / 2 in 
  125 < 150 ∧ 150 < 216 ∧ 150 < midpoint → 
  5 = round (150^(1/3)) := 
by 
  intro h
  sorry

end closest_integer_to_cuberoot_150_l590_590598


namespace area_triangle_eq_l590_590941

variables (P Q R : Type) [triangle P Q R]

-- Condition 1: Triangle PQR is a right triangle with ∠ R = 90°
axiom right_triangle (P Q R : Type) [triangle P Q R] : angle R = 90

-- Condition 2: The median to the hypotenuse is 5/4
axiom median_hypotenuse : med (P, Q, R) (hypotenuse (P, Q, R)) = 5/4

-- Condition 3: The perimeter of the triangle is 6
axiom perimeter : perimeter_triangle (P, Q, R) = 6

-- Proof statement to find the area of △PQR
theorem area_triangle_eq : area_triangle (P, Q, R) = 1.5 :=
by
  right_triangle P Q R,
  median_hypotenuse,
  perimeter,
  sorry

end area_triangle_eq_l590_590941


namespace volume_of_cylindrical_container_l590_590520

theorem volume_of_cylindrical_container (h : ℝ) (d : ℝ) (r : ℝ) (π: ℝ) (H_height : h = 15) (D_diameter : d = 8) (R_radius : r = 4):
  volume_of_cylinder: ∃ V, V = π * r^2 * h :=
begin
  sorry
end

end volume_of_cylindrical_container_l590_590520


namespace tangent_circle_area_eq_pi_r2_div_9_l590_590314

variables (r : ℝ) (AO BO : ℝ) (AOB : ℝ)
variables (tangent_circle_area : ℝ)

-- Given conditions.
hypothesis AO_eq_BO : AO = BO
hypothesis AO_and_BO_eq_r : AO = r ∧ BO = r
hypothesis angle_AOB_eq_60 : AOB = 60

-- Statement to prove.
theorem tangent_circle_area_eq_pi_r2_div_9 (h : AO_eq_BO ∧ AO_and_BO_eq_r ∧ angle_AOB_eq_60):
  tangent_circle_area = (r^2 * π) / 9 :=
sorry

end tangent_circle_area_eq_pi_r2_div_9_l590_590314


namespace total_oil_leak_l590_590957

theorem total_oil_leak : 
  let pipe1_leak_before := 6522
      pipe1_leak_during := 2443
      pipe2_leak_before := 8712
      pipe2_leak_during := 3894
      pipe3_leak_before := 9654
      pipe3_leak_rate_per_hour := 250
      pipe3_repair_hours := 7
      total_pipe1 := pipe1_leak_before + pipe1_leak_during
      total_pipe2 := pipe2_leak_before + pipe2_leak_during
      total_pipe3 := pipe3_leak_before + (pipe3_leak_rate_per_hour * pipe3_repair_hours)
  in total_pipe1 + total_pipe2 + total_pipe3 = 32975 :=
by
  sorry

end total_oil_leak_l590_590957


namespace expected_lifetime_of_flashlight_at_least_4_l590_590087

-- Definitions for the lifetimes of the lightbulbs
variable (ξ η : ℝ)

-- Condition: The expected lifetime of the red lightbulb is 4 years.
axiom E_η_eq_4 : 𝔼[η] = 4

-- Definition stating the lifetime of the flashlight
def T := max ξ η

theorem expected_lifetime_of_flashlight_at_least_4 
  (h : 𝔼η = 4) :
  𝔼[max ξ η] ≥ 4 :=
by {
  sorry
}

end expected_lifetime_of_flashlight_at_least_4_l590_590087


namespace explicit_expression_for_f_l590_590399

variable (f : ℕ → ℕ)

-- Define the condition
axiom h : ∀ x : ℕ, f (x + 1) = 3 * x + 2

-- State the theorem
theorem explicit_expression_for_f (x : ℕ) : f x = 3 * x - 1 :=
by {
  sorry
}

end explicit_expression_for_f_l590_590399


namespace regular_polygon_sides_l590_590727

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l590_590727


namespace exercise_hobby_gender_independence_l590_590626

-- Define given contingency table details
def num_males_hobby := 30
def num_females_no_hobby := 10
def total_num_employees := 100

-- Define the full contingency table
def contingency_table := 
  ⟨50, 50, 
    { hobby := 70, no_hobby := 30, total := total_num_employees },
    { hobby :=  { males := num_males_hobby, females := 40 },
      no_hobby := { males := 20, females := num_females_no_hobby } }⟩

-- Define probability values
def P_X_0 := 3 / 29
def P_X_1 := 40 / 87
def P_X_2 := 38 / 87

-- Define distribution table
def distribution_table := [(0, P_X_0), (1, P_X_1), (2, P_X_2)]

-- Define expectation
def E_X : ℝ := 4 / 3

-- Theorem statement
theorem exercise_hobby_gender_independence :
  let χ2 := 4.76 in
  let critical_value := 6.635 in
  χ2 < critical_value →
  -- Completing the contingency table assertion
  contingency_table =
    ⟨50, 50, 
      { hobby := 70, no_hobby := 30, total := total_num_employees },
      { hobby :=  { males := num_males_hobby, females := 40 },
        no_hobby := { males := 20, females := num_females_no_hobby } }⟩ ∧
  -- Conclusion of independence test assertion
  ∀ α (h : α = 0.01), χ2 < critical_value ∧ 
  -- Distribution and expectation assertion
  distribution_table = [(0, P_X_0), (1, P_X_1), (2, P_X_2)] ∧ 
  E_X = 4 / 3 :=
sorry

end exercise_hobby_gender_independence_l590_590626


namespace student_papers_count_l590_590276

theorem student_papers_count {F n k: ℝ}
  (h1 : 35 * k = 0.6 * n * F)
  (h2 : 5 * k > 0.5 * F)
  (h3 : 6 * k > 0.5 * F)
  (h4 : 7 * k > 0.5 * F)
  (h5 : 8 * k > 0.5 * F)
  (h6 : 9 * k > 0.5 * F) :
  n = 5 :=
by
  sorry

end student_papers_count_l590_590276


namespace Mike_can_play_300_minutes_l590_590890

-- Define the weekly earnings, spending, and costs as conditions
def weekly_earnings : ℕ := 100
def half_spent_at_arcade : ℕ := weekly_earnings / 2
def food_cost : ℕ := 10
def token_cost_per_hour : ℕ := 8
def hour_in_minutes : ℕ := 60

-- Define the remaining money after buying food
def money_for_tokens : ℕ := half_spent_at_arcade - food_cost

-- Define the hours he can play
def hours_playable : ℕ := money_for_tokens / token_cost_per_hour

-- Define the total minutes he can play
def total_minutes_playable : ℕ := hours_playable * hour_in_minutes

-- Prove that with his expenditure, Mike can play for 300 minutes
theorem Mike_can_play_300_minutes : total_minutes_playable = 300 := 
by
  sorry -- Proof will be filled here

end Mike_can_play_300_minutes_l590_590890


namespace sum_of_coefficients_without_z_l590_590184

theorem sum_of_coefficients_without_z :
  let polynomial := (λ x y z : ℕ, (x - 2 * y) ^ 3 * (y - 2 * z) ^ 5 * (z - 2 * x) ^ 7)
  let sum_of_coeffs := (λ x y : ℕ, ∑ k in finset.range 4, (binomial 3 k) * ((-2 : ℤ) ^ (7 + k)))
  (sum_of_coeffs 1 1 = 3456) :=
by
  sorry

end sum_of_coefficients_without_z_l590_590184


namespace prism_cubes_paint_condition_l590_590194

theorem prism_cubes_paint_condition
  (m n r : ℕ)
  (h1 : m ≤ n)
  (h2 : n ≤ r)
  (h3 : (m - 2) * (n - 2) * (r - 2)
        - 2 * ((m - 2) * (n - 2) + (m - 2) * (r - 2) + (n - 2) * (r - 2)) 
        + 4 * (m - 2 + n - 2 + r - 2)
        = 1985) :
  (m = 5 ∧ n = 7 ∧ r = 663) ∨
  (m = 5 ∧ n = 5 ∧ r = 1981) ∨
  (m = 3 ∧ n = 3 ∧ r = 1981) ∨
  (m = 1 ∧ n = 7 ∧ r = 399) ∨
  (m = 1 ∧ n = 3 ∧ r = 1987) := 
sorry

end prism_cubes_paint_condition_l590_590194


namespace isosceles_triangle_APQ_l590_590436

theorem isosceles_triangle_APQ
  {A B C D P Q : Type*}
  [A, B, C, D, P, Q are_points : Prop]
  (acute_triangle : is_acute_triangle A B C)
  (angle_bisector : bisects_angle A B C A D)
  (D_on_BC : lies_on_line D B C)
  (BP_perpendicular_AD : perpendicular B P A D)
  (CQ_perpendicular_AD : perpendicular C Q A D) : 
   isosceles_triangle A P Q := sorry

end isosceles_triangle_APQ_l590_590436


namespace equilateral_triangle_max_sum_regular_polygon_odd_distances_eq_l590_590989

-- Part 1: For the equilateral triangle problem
theorem equilateral_triangle_max_sum (A B C M : Point)
    (h_equilateral : isEquilateralTriangle A B C)
    (h_circumcircle : LiesOnCircumcircle M A B C) :
    (max (distance M A) (max (distance M B) (distance M C)) = distance M B) →
    (distance M B = distance M A + distance M C) :=
sorry

-- Part 2: For the regular polygon with an odd number of sides
theorem regular_polygon_odd_distances_eq (n : ℕ) (h_odd : n % 2 = 1)
    (A : ℕ → Point) (h_regular : is_regular_polygon A n)
    (M : Point) (h_circumcircle : LiesOnCircumcircle M (A 1) (A n)) :
    (∑ i in range n | odd i, distance M (A i)) = (∑ i in range n | even i, distance M (A i)) :=
sorry

end equilateral_triangle_max_sum_regular_polygon_odd_distances_eq_l590_590989


namespace closest_integer_to_cube_root_of_150_l590_590609

theorem closest_integer_to_cube_root_of_150 : ∃ (n : ℤ), abs ((n: ℝ)^3 - 150) ≤ abs (((n + 1 : ℤ) : ℝ)^3 - 150) ∧
  abs ((n: ℝ)^3 - 150) ≤ abs (((n - 1 : ℤ) : ℝ)^3 - 150) ∧ n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590609


namespace power_sum_identity_l590_590486

theorem power_sum_identity (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) : 
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49 / 60 := 
by
  sorry

end power_sum_identity_l590_590486


namespace general_formula_a_sum_T_l590_590041

noncomputable def a (n : ℕ) : ℕ := if n = 1 then 0 else n - 1
noncomputable def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, (a (i + 1) + 1 : ℕ) / (2 ^ (i + 1) : ℝ)

theorem general_formula_a (n : ℕ) (h1 : n ≥ 1) (h2 : a 2 = 1) (h3 : 2 * S n = n * a n) :
  a n = n - 1 := by
  sorry

theorem sum_T (n : ℕ) (h1 : ∀ k, a k = k - 1) :
  T n = 2 - (n + 2) / (2 ^ n) := by
  sorry

end general_formula_a_sum_T_l590_590041


namespace range_of_a_l590_590400

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 1 then -2 / x + 3 else |x - a^2 + 3|

theorem range_of_a {x1 x2 a : ℝ} (h1 : x1 ≠ x2)
  (h2 : 0 < x1 ∨ 0 < x2)
  (h3 : h1 → (x1 - x2) * (f a x1 - f a x2) > 0) :
  -sqrt 3 ≤ a ∧ a ≤ sqrt 3 :=
sorry

end range_of_a_l590_590400


namespace points_on_x_eq_3_is_vertical_line_points_with_x_lt_3_points_with_x_gt_3_points_on_y_eq_2_is_horizontal_line_points_with_y_gt_2_l590_590971

open Set

-- Define the point in the coordinate plane as a product of real numbers
def Point := ℝ × ℝ

-- Prove points with x = 3 form a vertical line
theorem points_on_x_eq_3_is_vertical_line : {p : Point | p.1 = 3} = {p : Point | ∀ y : ℝ, (3, y) = p} := sorry

-- Prove points with x < 3 lie to the left of x = 3
theorem points_with_x_lt_3 : {p : Point | p.1 < 3} = {p : Point | ∀ x y : ℝ, x < 3 → p = (x, y)} := sorry

-- Prove points with x > 3 lie to the right of x = 3
theorem points_with_x_gt_3 : {p : Point | p.1 > 3} = {p : Point | ∀ x y : ℝ, x > 3 → p = (x, y)} := sorry

-- Prove points with y = 2 form a horizontal line
theorem points_on_y_eq_2_is_horizontal_line : {p : Point | p.2 = 2} = {p : Point | ∀ x : ℝ, (x, 2) = p} := sorry

-- Prove points with y > 2 lie above y = 2
theorem points_with_y_gt_2 : {p : Point | p.2 > 2} = {p : Point | ∀ x y : ℝ, y > 2 → p = (x, y)} := sorry

end points_on_x_eq_3_is_vertical_line_points_with_x_lt_3_points_with_x_gt_3_points_on_y_eq_2_is_horizontal_line_points_with_y_gt_2_l590_590971


namespace triangles_with_equal_sides_are_congruent_l590_590165

theorem triangles_with_equal_sides_are_congruent 
  (T1 T2 : Triangle)
  (h : T1.side1 = T2.side1 ∧ T1.side2 = T2.side2 ∧ T1.side3 = T2.side3) : 
  T1 ≅ T2 :=
sorry

end triangles_with_equal_sides_are_congruent_l590_590165


namespace sequence_general_formula_l590_590016

noncomputable def sum_of_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ := 
  ∑ i in Finset.range (n + 1), a i

theorem sequence_general_formula:
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
    (∀ n, log (S n + 1) / log 2 = n + 1) →
    (∀ n, S n = sum_of_sequence a n) →
    (∀ n, a n = if n = 1 then 3 else 2^n) :=
by
  sorry

end sequence_general_formula_l590_590016


namespace PQRS_is_parallelogram_l590_590872

variable {P Q R S A B C D E : Type} [Points P Q R S A B C D E]

-- Defining a cyclic quadrilateral
def isCyclicQuadrilateral (A B C D : Type) : Prop := sorry

-- Defining the intersection point of diagonals
def intersectDiagonals (A B C D E : Type) : Prop := sorry

-- Defining circumcenters of triangles
def circumcenter (X Y Z P : Type) : Prop := sorry

-- Main theorem statement.
theorem PQRS_is_parallelogram (hCyclic : isCyclicQuadrilateral A B C D)
    (hIntersect : intersectDiagonals A B C D E)
    (hCircumP : circumcenter A B E P) 
    (hCircumQ : circumcenter B C E Q)
    (hCircumR : circumcenter C D E R) 
    (hCircumS : circumcenter A D E S) 
    : isParallelogram P Q R S := sorry

end PQRS_is_parallelogram_l590_590872


namespace jaclyns_polynomial_constant_term_l590_590491

theorem jaclyns_polynomial_constant_term 
  (P : Polynomial ℤ) (Q : Polynomial ℤ)
  (hP : P.Monic) (hQ : Q.Monic)
  (hdegP : P.degree = 3) (hdegQ : Q.degree = 3)
  (hconstP : P.eval 0 > 0) (hconstQ : Q.eval 0 > 0)
  (hprod : P * Q = Polynomial.ofCoeff [9, 2, 3, 2, 3, 2, 1, 0]) :
  Q.eval 0 = 3 := 
sorry

end jaclyns_polynomial_constant_term_l590_590491


namespace complex_modulus_l590_590801

theorem complex_modulus (b : ℝ) (h : (let z := (3 - b * complex.I) / complex.I in z.re = z.im)) :
  complex.abs ((3 - b * complex.I) / complex.I) = 3 * real.sqrt 2 :=
by sorry

end complex_modulus_l590_590801


namespace sum_of_reciprocals_of_roots_l590_590299

theorem sum_of_reciprocals_of_roots :
  let a := 1
  let b := -17
  let c := 8
  (∃ r1 r2: ℝ, r1 ≠ 0 ∧ r2 ≠ 0 ∧ a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 + c = 0) →
  (∑ r1 r2: ℝ, (1 / r1) + (1 / r2) = 17 / 8) :=
by
  intros
  sorry

end sum_of_reciprocals_of_roots_l590_590299


namespace mateo_deducted_salary_l590_590123

-- Define the conditions
def weekly_salary : ℝ := 791
def absences : ℝ := 4
def workdays_per_week : ℝ := 5

-- Define the daily wage
def daily_wage (weekly_salary : ℝ) (workdays_per_week : ℝ) : ℝ :=
  weekly_salary / workdays_per_week

-- Define the total deduction
def total_deduction (daily_wage : ℝ) (absences : ℝ) : ℝ :=
  daily_wage * absences

-- Define the deducted salary
def deducted_salary (weekly_salary : ℝ) (total_deduction : ℝ) : ℝ :=
  weekly_salary - total_deduction

-- Prove that the deducted salary is 158.20 dollars
theorem mateo_deducted_salary :
  deducted_salary weekly_salary (total_deduction (daily_wage weekly_salary workdays_per_week) absences) = 158.20 := 
by
  sorry

end mateo_deducted_salary_l590_590123


namespace number_of_arrangements_l590_590247

theorem number_of_arrangements:
  let students := {A, B, C, D, E, F},
      communityA := {studentA},
      communityB := {studentB, studentC},
      communityC := students \ (communityA ∪ communityB) 
  in students.card = 6 
  ∧ ∀ s, s ∈ communityA → s = A
  ∧ ∀ s, s ∈ communityB → s ≠ C
  ∧ ∀ s, s ∈ communityC → s ≠ B
  → (finset.pairs (communityA ∪ communityB ∪ communityC)).card = 9
:= by {
  sorry
}

end number_of_arrangements_l590_590247


namespace conjugate_in_fourth_quadrant_l590_590003

def z : ℂ := (-1 + 3 * complex.I) / (2 + complex.I)

theorem conjugate_in_fourth_quadrant (hz : z = (-1 + 3 * complex.I) / (2 + complex.I)) :
  (complex.conj z).re > 0 ∧ (complex.conj z).im < 0 := sorry

end conjugate_in_fourth_quadrant_l590_590003


namespace coefficient_of_x6_in_expansion_l590_590835

theorem coefficient_of_x6_in_expansion (a : ℝ) (h : (∑ r in Finset.range 11, (Nat.choose 10 r) * (30 - a * (Nat.choose 10 2)) * Term_equality == 30) : a = 2 := 
sorry

end coefficient_of_x6_in_expansion_l590_590835


namespace quad_roots_sum_l590_590712

noncomputable def roots_sum_eq : Real :=
  let φ := Real.pi / 12
  let z1 := Complex.cos φ + Complex.sin (2 * φ) * Complex.I
  let z2 := Complex.sin (2 * φ) + Complex.cos φ * Complex.I
  let z3 := Complex.sin (2 * φ) - Complex.cos φ * Complex.I
  let z4 := Complex.cos φ - Complex.sin (2 * φ) * Complex.I
  z1 + z2 + z3 + z4

theorem quad_roots_sum :
  let sum_roots := (2 * (Real.sqrt 6 + Real.sqrt 2) / 4) + 1
  roots_sum_eq = sum_roots :=
begin
  sorry
end

end quad_roots_sum_l590_590712


namespace prism_surface_area_l590_590554

-- Define the base of the prism as an isosceles trapezoid ABCD
structure Trapezoid :=
(AB CD : ℝ)
(BC : ℝ)
(AD : ℝ)

-- Define the properties of the prism
structure Prism :=
(base : Trapezoid)
(diagonal_cross_section_area : ℝ)

-- Define the specific isosceles trapezoid from the problem
def myTrapezoid : Trapezoid :=
{ AB := 13, CD := 13, BC := 11, AD := 21 }

-- Define the specific prism from the problem with the given conditions
noncomputable def myPrism : Prism :=
{ base := myTrapezoid, diagonal_cross_section_area := 180 }

-- Define the total surface area as a function
noncomputable def total_surface_area (p : Prism) : ℝ :=
2 * (1 / 2 * (p.base.AD + p.base.BC) * (Real.sqrt ((p.base.CD) ^ 2 - ((p.base.AD - p.base.BC) / 2) ^ 2))) +
(p.base.AB + p.base.BC + p.base.CD + p.base.AD) * (p.diagonal_cross_section_area / (Real.sqrt ((1 / 2 * (p.base.AD + p.base.BC)) ^ 2 + (Real.sqrt ((p.base.CD) ^ 2 - ((p.base.AD - p.base.BC) / 2) ^ 2)) ^ 2)))

-- The proof problem in Lean
theorem prism_surface_area :
  total_surface_area myPrism = 906 :=
sorry

end prism_surface_area_l590_590554


namespace pizza_distribution_l590_590647

theorem pizza_distribution :
  let pieces : List ℚ := [2 * 1/24, 4 * 1/12, 2 * 1/8, 2 * 1/6].map (λ x, x.norm.num)
  ∑ n in finset.Icc 2 10, n ∉ {2, 3, 4, 6} = 39 := by
  sorry

end pizza_distribution_l590_590647


namespace piper_gym_sessions_l590_590504

-- Define the conditions and the final statement as a theorem
theorem piper_gym_sessions (session_count : ℕ) (week_days : ℕ) (start_day : ℕ) 
  (alternate_day : ℕ) (skip_day : ℕ): (session_count = 35) ∧ (week_days = 7) ∧ 
  (start_day = 1) ∧ (alternate_day = 2) ∧ (skip_day = 7) → 
  (start_day + ((session_count - 1) / 3) * week_days + ((session_count - 1) % 3) * alternate_day) % week_days = 3 := 
by 
  sorry

end piper_gym_sessions_l590_590504


namespace train_length_l590_590660

-- Defining the conditions
def speed_kmh : ℕ := 64
def speed_m_per_s : ℚ := (64 * 1000) / 3600 -- 64 km/h converted to m/s
def time_to_cross_seconds : ℕ := 9 

-- The theorem to prove the length of the train
theorem train_length : speed_m_per_s * time_to_cross_seconds = 160 := 
by 
  unfold speed_m_per_s 
  norm_num
  sorry -- Placeholder for actual proof

end train_length_l590_590660


namespace expected_lifetime_flashlight_l590_590094

noncomputable def xi : ℝ := sorry
noncomputable def eta : ℝ := sorry

def T : ℝ := max xi eta

axiom E_eta_eq_4 : E eta = 4

theorem expected_lifetime_flashlight : E T ≥ 4 :=
by
  -- The solution will go here
  sorry

end expected_lifetime_flashlight_l590_590094


namespace count_valid_three_digit_numbers_l590_590414

def three_digit_number (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a * 100 + b * 10 + c < 1000) ∧
  (a * 100 + b * 10 + c >= 100) ∧
  (c = 2 * (b - a) + a)

theorem count_valid_three_digit_numbers : ∃ n : ℕ, n = 90 ∧
  ∃ (a b c : ℕ), three_digit_number a b c :=
by
  sorry

end count_valid_three_digit_numbers_l590_590414


namespace fg_neg_two_l590_590402

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x + 3

theorem fg_neg_two : f (g (-2)) = 2 := by
  sorry

end fg_neg_two_l590_590402


namespace probability_two_identical_l590_590969

-- Define the number of ways to choose 3 out of 4 attractions
def choose_3_out_of_4 := Nat.choose 4 3

-- Define the total number of ways for both tourists to choose 3 attractions out of 4
def total_basic_events := choose_3_out_of_4 * choose_3_out_of_4

-- Define the number of ways to choose exactly 2 identical attractions
def ways_to_choose_2_identical := Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 1 1

-- The probability that they choose exactly 2 identical attractions
def probability : ℚ := ways_to_choose_2_identical / total_basic_events

-- Prove that this probability is 3/4
theorem probability_two_identical : probability = 3 / 4 := by
  have h1 : choose_3_out_of_4 = 4 := by sorry
  have h2 : total_basic_events = 16 := by sorry
  have h3 : ways_to_choose_2_identical = 12 := by sorry
  rw [probability, h2, h3]
  norm_num

end probability_two_identical_l590_590969


namespace sequence_formula_l590_590405

-- Define the sequence a_n with the provided conditions
def a : ℕ → ℤ
| 0       := 1
| 1       := 5
| (n + 2) := (2 * a (n + 1) ^ 2 - 3 * a (n + 1) - 9) / (2 * a n)

-- State the theorem to be proven, that is, 'a_n = 2^(n+2) - 3' for all n
theorem sequence_formula (n : ℕ) : a n = 2^(n + 2) - 3 := by
  sorry

end sequence_formula_l590_590405


namespace abs_eq_n_when_three_integer_solutions_l590_590010

theorem abs_eq_n_when_three_integer_solutions (a : ℝ) : (∃ x : ℤ, (abs (abs (x - 3) - 1) = a)) ∧ (set.finite {x : ℤ | abs (abs (x - 3) - 1) = a} ∧ (set.to_finset {x : ℤ | abs (abs (x - 3) - 1) = a}).card = 3) → a = 1 :=
by
  sorry

end abs_eq_n_when_three_integer_solutions_l590_590010


namespace card_third_number_l590_590130

theorem card_third_number (a b : ℕ) : (numbers : set ℕ) :=
  18 ∈ numbers ∧ 75 ∈ numbers ∧ 
  (let ab := 10 * a + b in ab ∈ numbers ∧ 
  (let sum := (180000 + 7500 + ab) + (10000 * a + 1000 * b + 750) + 
                  (700000 + 10000 * a + 1000 * b + 518) + 
                  (750000 + 1000 * a + 100 * b + 518) + 
                  (10000 * ab + 7518) + (10000 * ab + 1875)
  in sum = 2606058)) -> 10 * a + b = 36 :=
sorry

end card_third_number_l590_590130


namespace limit_expression_equals_half_second_derivative_l590_590383

variable {f : ℝ → ℝ}
variable {hf : DifferentiableAt ℝ f 1}

theorem limit_expression_equals_half_second_derivative :
  (𝓝[≠] 0).lim (λ Δx : ℝ, (f(1 + Δx) - f(1)) / (-2 * Δx)) = - (1 / 2) * deriv (deriv f) 1 :=
by
  sorry

end limit_expression_equals_half_second_derivative_l590_590383


namespace count_satisfying_k_l590_590373

theorem count_satisfying_k :
  ∃ k_set : Finset ℕ, k_set.card = 11 ∧ ∀ k ∈ k_set, 1 ≤ k ∧ k ≤ 2017 ∧ 
  ( (Finset.range k).sum (λ n, Real.sin (n+1 : ℝ) * (Real.pi / 180)) =
    (Finset.range k).prod (λ n, Real.sin ((n+1 : ℝ) * (Real.pi / 180))) ) :=
sorry

end count_satisfying_k_l590_590373


namespace clock_distances_greater_l590_590954

-- Define the conditions used in the problem
variables (O : Point) (clocks : Finset Clock) [Fintype clocks] (minute_hand_end : Clock → Point) (clock_center : Clock → Point)

-- Define the sums of distances
def s1 : ℝ := ∑ clock in clocks, dist O (minute_hand_end clock)
def s0 : ℝ := ∑ clock in clocks, dist O (clock_center clock)

-- State the theorem to be proved
theorem clock_distances_greater (at_some_moment : Prop) :
  (∃ t : Time, s1 t > s0) := sorry

end clock_distances_greater_l590_590954


namespace molly_age_l590_590915

theorem molly_age
  (S M : ℕ)
  (h_ratio : S / M = 4 / 3)
  (h_sandy_future : S + 6 = 42)
  : M = 27 :=
sorry

end molly_age_l590_590915


namespace find_k_l590_590351

theorem find_k (x y k : ℤ) (h₁ : x = -3) (h₂ : y = 2) (h₃ : 2 * x + k * y = 6) : k = 6 :=
by
  rw [h₁, h₂] at h₃
  -- Substitute x and y in the equation
  -- 2 * (-3) + k * 2 = 6
  sorry

end find_k_l590_590351


namespace vector_parallel_l590_590406

theorem vector_parallel (x : ℝ) :
  let a : ℝ × ℝ := (2 * x + 1, 4)
  let b : ℝ × ℝ := (2 - x, 3)
  (3 * (2 * x + 1) - 4 * (2 - x) = 0) → (x = 1 / 2) :=
by
  intros a b h
  sorry

end vector_parallel_l590_590406


namespace probability_sum_greater_than_five_l590_590964

theorem probability_sum_greater_than_five (dice_outcomes : List (ℕ × ℕ)) (h: dice_outcomes = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (3,1), (3,2), (4,1), (5,1), (2,4)] ++ 
                              [(1,5), (2,6), (3,3), (3,4), (3,5), (3,6), (4,2), (4,3), (4,4), (4,5), (4,6), 
                               (5,2), (5,3), (5,4), (5,5), (5,6), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)]) :
  p_greater_5 = 2 / 3 := 
by
  sorry

end probability_sum_greater_than_five_l590_590964


namespace probability_six_integers_unique_tens_digit_l590_590544

theorem probability_six_integers_unique_tens_digit :
  (∃ (x1 x2 x3 x4 x5 x6 : ℕ),
    10 ≤ x1 ∧ x1 ≤ 79 ∧
    10 ≤ x2 ∧ x2 ≤ 79 ∧
    10 ≤ x3 ∧ x3 ≤ 79 ∧
    10 ≤ x4 ∧ x4 ≤ 79 ∧
    10 ≤ x5 ∧ x5 ≤ 79 ∧
    10 ≤ x6 ∧ x6 ≤ 79 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x1 ≠ x6 ∧
    x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x2 ≠ x6 ∧
    x3 ≠ x4 ∧ x3 ≠ x5 ∧ x3 ≠ x6 ∧
    x4 ≠ x5 ∧ x4 ≠ x6 ∧
    x5 ≠ x6 ∧
    tens_digit x1 ≠ tens_digit x2 ∧
    tens_digit x1 ≠ tens_digit x3 ∧
    tens_digit x1 ≠ tens_digit x4 ∧
    tens_digit x1 ≠ tens_digit x5 ∧
    tens_digit x1 ≠ tens_digit x6 ∧
    tens_digit x2 ≠ tens_digit x3 ∧
    tens_digit x2 ≠ tens_digit x4 ∧
    tens_digit x2 ≠ tens_digit x5 ∧
    tens_digit x2 ≠ tens_digit x6 ∧
    tens_digit x3 ≠ tens_digit x4 ∧
    tens_digit x3 ≠ tens_digit x5 ∧
    tens_digit x3 ≠ tens_digit x6 ∧
    tens_digit x4 ≠ tens_digit x5 ∧
    tens_digit x4 ≠ tens_digit x6 ∧
    tens_digit x5 ≠ tens_digit x6)
    →
  (probability := \(\frac{4375}{744407}\)).sorry

end probability_six_integers_unique_tens_digit_l590_590544


namespace leftmost_row_tile_count_l590_590234

noncomputable def leftmost_row_tiles (total_tiles : ℕ) (num_rows : ℕ) (diff : ℤ) : ℤ :=
  let a_1 : ℤ := (2 * total_tiles - num_rows * (num_rows - 1) * diff) / (2 * num_rows)
  a_1

theorem leftmost_row_tile_count :
  leftmost_row_tiles 405 9 (-2) = 53 :=
by
  simp [leftmost_row_tiles, Int.ofNat]
  sorry

end leftmost_row_tile_count_l590_590234


namespace intersection_of_A_and_B_l590_590423

def setA (x : Real) : Prop := -1 < x ∧ x < 3
def setB (x : Real) : Prop := -2 < x ∧ x < 2

theorem intersection_of_A_and_B : {x : Real | setA x} ∩ {x : Real | setB x} = {x : Real | -1 < x ∧ x < 2} := 
by
  sorry

end intersection_of_A_and_B_l590_590423


namespace sum_mod_7_is_zero_l590_590293

-- Define the sum of the first n natural numbers
def sum_n : ℕ → ℕ
| 0       := 0
| (n + 1) := sum_n n + (n + 1)

-- Define the statement to be proven
theorem sum_mod_7_is_zero : sum_n 127 % 7 = 0 :=
by
  sorry

end sum_mod_7_is_zero_l590_590293


namespace graph_of_y_squared_eq_sin_x_squared_l590_590222

theorem graph_of_y_squared_eq_sin_x_squared (x y : ℝ) :
    (y^2 = Real.sin (x^2)) → 
    (y = 0 ∨ y^2 = Real.sin (x^2) ∧ 0 ≤ Real.sin (x^2) ∧ Real.sin (x^2) ≤ 1) := 
by
  intros h
  split
  { intro yz
    rw [yz] at h
    rw [Real.sin_zero] at h
    rw [pow_zero] at h
    exact rfl }
  { intro ys
    split
    { exact h }
    { split
      { apply Real.sin_nonneg (abs x) }
      { apply Real.sin_le_one (abs x) } }
  }

-- Test if y = 0 ∨ y = sqrt (Real.sin (x^2)) ∨ y = -sqrt (Real.sin (x^2)) hold
example (x : ℝ) : (0 ≤ Real.sin (x^2) → Real.sin (x^2) ≤ 1) := sorry

end graph_of_y_squared_eq_sin_x_squared_l590_590222


namespace complex_number_solution_l590_590556

noncomputable def z : ℂ :=
  4 + 3 * complex.I

theorem complex_number_solution (z : ℂ) (h : (-3 + 4 * complex.I) * z = 25 * complex.I) : 
  z = 4 + 3 * complex.I :=
by
  sorry

end complex_number_solution_l590_590556


namespace Petya_time_comparison_l590_590678

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l590_590678


namespace find_a_range_l590_590804

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then -(x - 1) ^ 2 else (3 - a) * x + 4 * a

theorem find_a_range (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f x₁ a - f x₂ a) / (x₁ - x₂) > 0) ↔ (-1 ≤ a ∧ a < 3) :=
sorry

end find_a_range_l590_590804


namespace multiplicative_magic_square_l590_590021

theorem multiplicative_magic_square (a b x : ℕ) (ha : a = 2) (hb : b = 5 * a) 
    (col1 : 5 * x * a = x * b) (col2 : b * 4 = 5 * 4) 
    (mul_square : 5 * b * 4 = 10 * x) : x = 100 :=
by {
    rw hb at *,
    rw ha at *,
    sorry -- proof to be added
}

end multiplicative_magic_square_l590_590021


namespace trig_identity_l590_590627

theorem trig_identity : 4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by sorry

end trig_identity_l590_590627


namespace simplify_decimal_l590_590615

theorem simplify_decimal : (3416 / 1000 : ℚ) = 427 / 125 := by
  sorry

end simplify_decimal_l590_590615


namespace find_n_l590_590587

open Nat

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b

theorem find_n (n : ℕ) (h_lcm : lcm n 14 = 56) (h_gcf : gcd n 14 = 10) : n = 40 :=
by
  sorry

end find_n_l590_590587


namespace bad_arrangements_count_l590_590943

/-- Circle arrangement of numbers 1 through 6 -/
structure CircleArrangement :=
  (arrangement : Fin 6 -> Fin 6)
  (permutes : ∀ x, x ∈ Finset.univ -> arrangement x ∈ Finset.univ)

/-- A bad arrangement has no subsets of consecutive elements that sum to n for some 6 ≤ n ≤ 15. -/
def is_bad_arrangement (c : CircleArrangement) : Prop :=
  ∃ n, n ∈ Finset.range (10) ∧ (6 + n) ∉ { s | ∃ I, I ⊆ Finset.univ ∧ (Finset.card I = n + 6) ∧ (Finset.sum I c.arrangement = n + 6) }

/-- There are exactly 3 distinct bad arrangements. -/
theorem bad_arrangements_count : Finset.card {c : CircleArrangement | is_bad_arrangement c} = 3 :=
begin
  sorry
end

end bad_arrangements_count_l590_590943


namespace time_spent_on_homework_l590_590492

def time_spent_on_laundry := 30
def time_spent_on_bathroom := 15
def time_spent_on_room := 35
def total_time := 120

theorem time_spent_on_homework 
  (time_spent_on_laundry time_spent_on_bathroom time_spent_on_room total_time : ℕ) 
  (h_laundry : time_spent_on_laundry = 30) 
  (h_bathroom : time_spent_on_bathroom = 15) 
  (h_room : time_spent_on_room = 35) 
  (h_total : total_time = 120) : 
  total_time - (time_spent_on_laundry + time_spent_on_bathroom + time_spent_on_room) = 40 :=
by 
  rw [h_laundry, h_bathroom, h_room, h_total]
  simp
  sorry

end time_spent_on_homework_l590_590492


namespace medical_team_selection_l590_590583

theorem medical_team_selection : 
  let male_doctors := 6
  let female_doctors := 5
  let choose_male := Nat.choose male_doctors 2
  let choose_female := Nat.choose female_doctors 1
  choose_male * choose_female = 75 := 
by 
  sorry

end medical_team_selection_l590_590583


namespace carpet_dimensions_problem_l590_590282

def carpet_dimensions (width1 width2 : ℕ) (l : ℕ) :=
  ∃ x y : ℕ, width1 = 38 ∧ width2 = 50 ∧ l = l ∧ x = 25 ∧ y = 50

theorem carpet_dimensions_problem (l : ℕ) :
  carpet_dimensions 38 50 l :=
by
  sorry

end carpet_dimensions_problem_l590_590282


namespace max_single_player_salary_correct_l590_590653

noncomputable def max_single_player_salary (total_salary team_size min_salary max_team_salary : ℕ) : ℕ :=
  sorry

theorem max_single_player_salary_correct :
  ∀ (total_salary team_size min_salary : ℕ),
    total_salary = 800000 →
    team_size = 18 →
    min_salary = 20000 →
    max_single_player_salary total_salary team_size min_salary total_salary team_size = 460000 :=
  sorry

end max_single_player_salary_correct_l590_590653


namespace seashells_problem_l590_590283

theorem seashells_problem
  (F : ℕ)
  (h : (150 - F) / 2 = 55) :
  F = 40 :=
  sorry

end seashells_problem_l590_590283


namespace divisors_of_100n5_l590_590770

-- Define conditions and proof problem.
theorem divisors_of_100n5 (n : ℕ) (h1 : 0 < n) (h2 : (2 + 1) * (5 + 1) * (7 + 1) = 140) :
  number_of_divisors (100 * n^5) = 24 :=
sorry

end divisors_of_100n5_l590_590770


namespace square_to_4_isosceles_triangles_non_congruent_l590_590625

theorem square_to_4_isosceles_triangles_non_congruent (s : Square) :
  ∃ (t1 t2 t3 t4 : Triangle), is_isosceles t1 ∧ is_isosceles t2 ∧ is_isosceles t3 ∧ is_isosceles t4 ∧ 
  ¬ congruent t1 t2 ∧ ¬ congruent t1 t3 ∧ ¬ congruent t1 t4 ∧ ¬ congruent t2 t3 ∧ ¬ congruent t2 t4 ∧ ¬ congruent t3 t4 := 
sorry

end square_to_4_isosceles_triangles_non_congruent_l590_590625


namespace cos_diff_symmetric_l590_590799

variables (α β : ℝ) (k : ℤ)

-- Conditions from the original problem
def symmetric_angles (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + (2 * k - 1) * Real.pi

-- Theorem statement proving the question == answer given the conditions
theorem cos_diff_symmetric (h : symmetric_angles α β) : Mathlib.cos (α - β) = -1 :=
sorry

end cos_diff_symmetric_l590_590799


namespace sum_of_distances_less_than_perimeter_l590_590909

variable {A B C M : Type} [MetricSpace M] 

theorem sum_of_distances_less_than_perimeter (A B C M : M) (hM: InsideTriangle A B C M) :
  dist M A + dist M B + dist M C < dist A B + dist B C + dist C A :=
sorry

end sum_of_distances_less_than_perimeter_l590_590909


namespace existence_of_rational_distance_points_on_unit_circle_l590_590326

noncomputable def rational_distance_points_on_unit_circle : Prop :=
  ∃ (points : Fin 1975 → ℝ × ℝ),
    (∀ i j : Fin 1975, i ≠ j → dist (points i) (points j) ∈ ℚ) ∧
    (∀ i : Fin 1975, ∥points i∥ = 1)

theorem existence_of_rational_distance_points_on_unit_circle :
  rational_distance_points_on_unit_circle := 
sorry

end existence_of_rational_distance_points_on_unit_circle_l590_590326


namespace gcd_of_sum_and_product_l590_590185

theorem gcd_of_sum_and_product (x y : ℕ) (h1 : x + y = 1130) (h2 : x * y = 100000) : Int.gcd x y = 2 := 
sorry

end gcd_of_sum_and_product_l590_590185


namespace find_constant_term_l590_590391

theorem find_constant_term (q' : ℝ → ℝ) (c : ℝ) (h1 : ∀ q : ℝ, q' q = 3 * q - c)
  (h2 : q' (q' 7) = 306) : c = 252 :=
by
  sorry

end find_constant_term_l590_590391


namespace probability_six_integers_diff_tens_l590_590523

-- Defining the range and conditions for the problem
def set_of_integers : Finset ℤ := Finset.range 70 \ Finset.range 10

def has_different_tens_digit (s : Finset ℤ) : Prop :=
  (s.card = 6) ∧ (∀ x y ∈ s, x ≠ y → (x / 10) ≠ (y / 10))

noncomputable def num_ways_choose_six_diff_tens : ℚ :=
  ((7 : ℚ) * (10^6 : ℚ))

noncomputable def total_ways_choose_six : ℚ :=
  (Nat.choose 70 6 : ℚ)

noncomputable def probability_diff_tens : ℚ :=
  num_ways_choose_six_diff_tens / total_ways_choose_six

-- Statement claiming the required probability
theorem probability_six_integers_diff_tens :
  probability_diff_tens = 1750 / 2980131 :=
by
  sorry

end probability_six_integers_diff_tens_l590_590523


namespace number_of_trees_planted_l590_590582

-- Definition of initial conditions
def initial_trees : ℕ := 22
def final_trees : ℕ := 55

-- Theorem stating the number of trees planted
theorem number_of_trees_planted : final_trees - initial_trees = 33 := by
  sorry

end number_of_trees_planted_l590_590582


namespace smallest_x_l590_590979

theorem smallest_x (x : ℕ) (h : 450 * x % 648 = 0) : x = 36 := 
sorry

end smallest_x_l590_590979


namespace petya_time_l590_590691

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l590_590691


namespace correct_simplification_l590_590224

theorem correct_simplification : -(+6) = -6 := by
  sorry

end correct_simplification_l590_590224


namespace expected_lifetime_flashlight_l590_590085

noncomputable section

variables (ξ η : ℝ) -- lifetimes of the blue and red lightbulbs
variables [probability_space ℙ] -- assuming a probability space ℙ

-- condition: expected lifetime of the red lightbulb is 4 years
axiom expected_eta : ℙ.𝔼(η) = 4

-- the main proof problem
theorem expected_lifetime_flashlight : ℙ.𝔼(max ξ η) ≥ 4 :=
sorry

end expected_lifetime_flashlight_l590_590085


namespace traverse_paths_bound_l590_590129

noncomputable def t_n (n : ℕ) : ℕ := sorry

theorem traverse_paths_bound (n : ℕ) (hn : n > 0) (ht : ∀ path, path starts at (0, 0) and traverses every cell exactly once) :
  1.25 < real.sqrt (real.log (t_n n) / real.log ((n : ℕ) * (n : ℕ))) ∧
  real.sqrt (real.log (t_n n) / real.log ((n : ℕ) * (n : ℕ))) < 2 := 
sorry

end traverse_paths_bound_l590_590129


namespace product_divisible_by_sum_iff_not_odd_prime_l590_590908

theorem product_divisible_by_sum_iff_not_odd_prime (n : ℕ) :
  (∃ k : ℕ, 2 * nat.factorial n = k * (n * (n + 1))) ↔ ¬ nat.prime (n + 1) ∨ (n + 1) % 2 = 0 :=
by sorry

end product_divisible_by_sum_iff_not_odd_prime_l590_590908


namespace base_of_first_exponent_l590_590828

theorem base_of_first_exponent (a b : ℕ) (h₀ : a = 6) (h₁ : a > 0) (h₂ : b > 0) 
    (h₃ : (λ x : ℕ, x^a * 9^(3 * a - 1) = 2^6 * 3^b)) : 
  ∃ x, x = 2 :=
by
  existsi 2
  sorry

end base_of_first_exponent_l590_590828


namespace triangle_probability_l590_590648

theorem triangle_probability (AB BC CA : ℕ) (r : ℚ) (m n : ℕ) (h1 : AB = 5) (h2 : BC = 12) (h3 : CA = 13) (h4 : r = 1/3)
  (prob : ℚ) (h5 : prob = 25/36) (coprime : Nat.coprime m n) (h6 : m = 25) (h7 : n = 36) : m + n = 61 :=
by
  sorry

end triangle_probability_l590_590648


namespace replaced_person_weight_l590_590155

theorem replaced_person_weight 
  (avg_weight_increase : 3 = 3) 
  (new_person_weight : 89 = 89) 
  (total_weight_increase : 8 * avg_weight_increase = 24) : 
  let old_person_weight := new_person_weight - total_weight_increase in 
  old_person_weight = 65 :=
by
  sorry

end replaced_person_weight_l590_590155


namespace minimum_travel_time_l590_590200

structure TravelSetup where
  distance_ab : ℝ
  number_of_people : ℕ
  number_of_bicycles : ℕ
  speed_cyclist : ℝ
  speed_pedestrian : ℝ
  unattended_rule : Prop

theorem minimum_travel_time (setup : TravelSetup) : setup.distance_ab = 45 → 
                                                    setup.number_of_people = 3 → 
                                                    setup.number_of_bicycles = 2 → 
                                                    setup.speed_cyclist = 15 → 
                                                    setup.speed_pedestrian = 5 → 
                                                    setup.unattended_rule → 
                                                    ∃ t : ℝ, t = 3 := 
by
  intros
  sorry

end minimum_travel_time_l590_590200


namespace find_2a_plus_b_l590_590477

open Real

variables {a b : ℝ}

-- Conditions
def angles_in_first_quadrant (a b : ℝ) : Prop := 
  0 < a ∧ a < π / 2 ∧ 0 < b ∧ b < π / 2

def cos_condition (a b : ℝ) : Prop :=
  5 * cos a ^ 2 + 3 * cos b ^ 2 = 2

def sin_condition (a b : ℝ) : Prop :=
  5 * sin (2 * a) + 3 * sin (2 * b) = 0

-- Problem statement
theorem find_2a_plus_b (a b : ℝ) 
  (h1 : angles_in_first_quadrant a b)
  (h2 : cos_condition a b)
  (h3 : sin_condition a b) :
  2 * a + b = π / 2 := 
sorry

end find_2a_plus_b_l590_590477


namespace find_CD_l590_590955

variables (A B C D E : Type) [metric_space A] [metric_space B]
variables (radius_A radius_B : ℝ) 

-- Declare the conditions as Lean definitions
def circle_at_A : set A := sorry
def circle_at_B : set B := sorry

-- Assume the following distances
axiom radius_A_def : radius_A = 5
axiom radius_B_def : radius_B = 15
axiom BE_def : dist B E = 39
axiom tangent_points : C ∈ circle_at_A ∧ D ∈ circle_at_B
axiom intersection_point : ∃ E: Type, E ∈ line(B,A) ∧ E ∈ line(C,D)

-- Declare the theorem
theorem find_CD : dist C D = 48 := by
  sorry

end find_CD_l590_590955


namespace problem_statement_l590_590922

theorem problem_statement (a b c : ℤ) (h1 : a * (3 + complex.I)^5 + b * (3 + complex.I)^4 + c * (3 + complex.I)^3 + b * (3 + complex.I) + a = 0)
  (h2 : Int.gcd a (Int.gcd b c) = 1) : |b| = 60 :=
sorry

end problem_statement_l590_590922


namespace proof_problem_l590_590229

noncomputable def problem_statement (x : ℝ) : Prop :=
  9.280 * real.log (x) / real.log (7) - real.log (7) / real.log (3) * real.log (x) / real.log (3) > real.log (0.25) / real.log (2)

theorem proof_problem (x : ℝ) : problem_statement x → 0 < x ∧ x < real.exp (real.log 3 * (2 / (real.log 3 / real.log 7 - real.log 7 / real.log 3))) :=
by
  sorry

end proof_problem_l590_590229


namespace best_fitting_model_is_model1_l590_590441

noncomputable def model1_R2 : ℝ := 0.98
noncomputable def model2_R2 : ℝ := 0.80
noncomputable def model3_R2 : ℝ := 0.54
noncomputable def model4_R2 : ℝ := 0.35

theorem best_fitting_model_is_model1 :
  model1_R2 > model2_R2 ∧ model1_R2 > model3_R2 ∧ model1_R2 > model4_R2 :=
by
  sorry

end best_fitting_model_is_model1_l590_590441


namespace problem_1_problem_2_l590_590408

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def vector_c : ℝ × ℝ := (-1, 0)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem problem_1 (x : ℝ) (h_1 : x = Real.pi / 3) :
  let θ := Real.arccos (dot_product (vector_a x) vector_c / (norm (vector_a x) * norm vector_c)) in
  θ = 5 * Real.pi / 6 :=
sorry

noncomputable def f (x λ : ℝ) : ℝ := λ / 2 * (1 + Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4))

theorem problem_2 (x λ : ℝ) (h_2 : x ∈ Set.Icc (-3 * Real.pi / 8) (Real.pi / 4)) (h_3 : ∀ x, f x λ ≤ 1 / 2) :
  λ = 1 / 2 ∨ λ = -1 - Real.sqrt 2 :=
sorry

end problem_1_problem_2_l590_590408


namespace expected_lifetime_flashlight_l590_590083

noncomputable section

variables (ξ η : ℝ) -- lifetimes of the blue and red lightbulbs
variables [probability_space ℙ] -- assuming a probability space ℙ

-- condition: expected lifetime of the red lightbulb is 4 years
axiom expected_eta : ℙ.𝔼(η) = 4

-- the main proof problem
theorem expected_lifetime_flashlight : ℙ.𝔼(max ξ η) ≥ 4 :=
sorry

end expected_lifetime_flashlight_l590_590083


namespace choir_females_correct_l590_590581

noncomputable def number_of_females_in_choir : ℕ :=
  let orchestra_males := 11
  let orchestra_females := 12
  let orchestra_musicians := orchestra_males + orchestra_females
  let band_males := 2 * orchestra_males
  let band_females := 2 * orchestra_females
  let band_musicians := 2 * orchestra_musicians
  let total_musicians := 98
  let choir_males := 12
  let choir_musicians := total_musicians - (orchestra_musicians + band_musicians)
  let choir_females := choir_musicians - choir_males
  choir_females

theorem choir_females_correct : number_of_females_in_choir = 17 := by
  sorry

end choir_females_correct_l590_590581


namespace A_always_wins_l590_590131

def Pos := (ℕ × ℕ)

structure State :=
  (grid : Fin 100 × Fin 100 → Option Bool) -- None means empty, Some true means star, Some false means circle

def valid_move (s : State) (p : Pos) : Prop :=
  p.1 < 99 ∧ p.2 < 99 ∧ s.grid ⟨p.1, _⟩ ⟨p.2, _⟩ = none

def center_pos : Pos := (50, 50)

def initial_state : State :=
  { grid := λ (p : Fin 100) (q : Fin 100), if (p, q) = center_pos then some true else none }

def B_moves : Pos → Prop :=
  λ p, p ∈ {(49,49), (50,49), (51,49), (49,50), (51,50), (49,51), (50,51), (51,51)}

theorem A_always_wins (s : State) :
  s.grid ⟨1, _⟩ ⟨1, _⟩ = some true ∨
  s.grid ⟨1, _⟩ ⟨99, _⟩ = some true ∨
  s.grid ⟨99, _⟩ ⟨1, _⟩ = some true ∨
  s.grid ⟨99, _⟩ ⟨99, _⟩ = some true := sorry

end A_always_wins_l590_590131


namespace average_price_per_book_l590_590142

theorem average_price_per_book :
  let shop1_books := 42
  let shop1_price := 520
  let shop1_discount := 0.10
  let shop2_books := 22
  let shop2_price := 248
  let shop2_tax := 0.05
  let shop3_books := 35
  let shop3_price := 740
  let shop3_discount := 0.15
  let shop4_books := 18
  let shop4_price := 360
  let shop4_tax := 0.08
  let conversion_a_to_b := 1.2
  let conversion_b_to_c := 0.75
  let conversion_c_to_d := 1.35
  let shop1_cost := (shop1_price * (1 - shop1_discount)) / shop1_books * conversion_a_to_b * conversion_b_to_c * conversion_c_to_d
  let shop2_cost := (shop2_price * (1 + shop2_tax)) / shop2_books * conversion_b_to_c * conversion_c_to_d
  let shop3_cost := (shop3_price * (1 - shop3_discount)) / shop3_books * conversion_c_to_d
  let shop4_cost := (shop4_price * (1 + shop4_tax)) / shop4_books
  let total_cost := shop1_cost * shop1_books + shop2_cost * shop2_books + shop3_cost * shop3_books + shop4_cost * shop4_books
  let total_books := shop1_books + shop2_books + shop3_books + shop4_books
  let average_price_per_book := total_cost / total_books
  in average_price_per_book = 17.953162 := 
by sorry

end average_price_per_book_l590_590142


namespace water_depth_upright_l590_590273

noncomputable def radius : ℝ := 3
noncomputable def height : ℝ := 12
noncomputable def water_depth_side : ℝ := 4
noncomputable def tank_diameter : ℝ := 6

def depth_upright := 4.1

theorem water_depth_upright :
  -- Given conditions
  (r = radius) ∧ 
  (h = height) ∧ 
  (d_water = water_depth_side) ∧ 
  (d_tank = tank_diameter)
  -- Prove that upright water depth is 4.1 feet
  → (h_upright = depth_upright) :=
begin
  sorry
end

end water_depth_upright_l590_590273


namespace angle_parallel_sides_l590_590427

theorem angle_parallel_sides {a b : ℝ} (h_parallel: ∀ i ∈ [1, 2], parallel (side a i) (side b i))
  (h_a: a = 60) : b = 60 ∨ b = 120 :=
sorry

end angle_parallel_sides_l590_590427


namespace number_of_dozen_eggs_to_mall_l590_590032

-- Define the conditions as assumptions
def number_of_dozen_eggs_collected (x : Nat) : Prop :=
  x = 2 * 8

def number_of_dozen_eggs_to_market (x : Nat) : Prop :=
  x = 3

def number_of_dozen_eggs_for_pie (x : Nat) : Prop :=
  x = 4

def number_of_dozen_eggs_to_charity (x : Nat) : Prop :=
  x = 4

-- The theorem stating the answer to the problem
theorem number_of_dozen_eggs_to_mall 
  (h1 : ∃ x, number_of_dozen_eggs_collected x)
  (h2 : ∃ x, number_of_dozen_eggs_to_market x)
  (h3 : ∃ x, number_of_dozen_eggs_for_pie x)
  (h4 : ∃ x, number_of_dozen_eggs_to_charity x)
  : ∃ z, z = 5 := 
sorry

end number_of_dozen_eggs_to_mall_l590_590032


namespace complex_point_quadrant_l590_590120

theorem complex_point_quadrant 
  (z : ℂ) (h : (1 - complex.I) * z = 1 + complex.I) : 
  (complex.abs z - complex.I).re > 0 ∧ (complex.abs z - complex.I).im < 0 :=
by sorry

end complex_point_quadrant_l590_590120


namespace tangent_line_equation_range_of_m_l590_590393

noncomputable def f (x : ℝ) : ℝ := 2*x^3 - 3*x^2 + 3

theorem tangent_line_equation : 
  let y := 12*x - 17 in 
  (∀ x, y = f(x) - 7 ↔ x = 2 & f(2) = 7) := sorry

theorem range_of_m (m : ℝ) :
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f(a) + m = 0 ∧ f(b) + m = 0 ∧ f(c) + m = 0) ↔ (-3 < m ∧ m < -2) := sorry

end tangent_line_equation_range_of_m_l590_590393


namespace cos_A_minus_cos_C_l590_590454

-- Definitions representing the conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h₁ : 4 * b * Real.sin A = Real.sqrt 7 * a)
variables (h₂ : 2 * b = a + c) (h₃ : A < B) (h₄ : B < C)

-- Statement of the proof problem
theorem cos_A_minus_cos_C (A B C a b c : ℝ)
  (h₁ : 4 * b * Real.sin A = Real.sqrt 7 * a)
  (h₂ : 2 * b = a + c)
  (h₃ : A < B)
  (h₄ : B < C) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 :=
by
  sorry

end cos_A_minus_cos_C_l590_590454


namespace multiple_choice_problem_l590_590611

theorem multiple_choice_problem :
  (∀ x : ℝ, (2 * real.sqrt 3) ^ 2 ≠ 6) ∧
  (∀ x : ℝ, real.sqrt 32 / real.sqrt 8 = 2) ∧
  (∀ x : ℝ, real.sqrt 2 + real.sqrt 5 ≠ real.sqrt 7) ∧
  (∀ x : ℝ, real.sqrt 2 * (2 * real.sqrt 3) ≠ 2 * real.sqrt 5) := 
by
  have A := (2 * real.sqrt 3) ^ 2
  have B := real.sqrt 32 / real.sqrt 8
  have C := real.sqrt 2 + real.sqrt 5
  have D := real.sqrt 2 * (2 * real.sqrt 3)

  split,
  { intros x, exact ne_of_gt (A x) 12 (by sorry) }, -- change 6 -> 12
  split,
  { intros x, exact eq_of_neg_eq_pos_le (B x) (by sorry) }, -- sqrt{32}/sqrt{8} = 2
  split,
  { intros x, exact ne_of_gt (C x) 7 (by sorry)}, -- sqrt{2} + sqrt{5} = 7
  { intros x, exact ne_of_gt (D x) (by sorry) } -- sqrt{2} * 2sqrt{3} = 5


end multiple_choice_problem_l590_590611


namespace not_all_numbers_different_l590_590189

theorem not_all_numbers_different (n : ℕ) (hn : n > 3) 
  (h_exists : ∃ (x : Fin n → ℕ), (∀ i, x i < (n - 1)! ∧ ∀ i j, i ≠ j → x i ≠ x j)) :
  ∃ (i j : Fin n), i ≠ j ∧ (x i / x j) = (x j / x i) := 
sorry

end not_all_numbers_different_l590_590189


namespace expected_lifetime_of_flashlight_at_least_4_l590_590090

-- Definitions for the lifetimes of the lightbulbs
variable (ξ η : ℝ)

-- Condition: The expected lifetime of the red lightbulb is 4 years.
axiom E_η_eq_4 : 𝔼[η] = 4

-- Definition stating the lifetime of the flashlight
def T := max ξ η

theorem expected_lifetime_of_flashlight_at_least_4 
  (h : 𝔼η = 4) :
  𝔼[max ξ η] ≥ 4 :=
by {
  sorry
}

end expected_lifetime_of_flashlight_at_least_4_l590_590090


namespace necessary_but_not_sufficient_condition_l590_590002

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x = 1 → x^2 - 3 * x + 2 = 0) ∧ (∃ x : ℝ, x^2 - 3 * x + 2 = 0 ∧ x ≠ 1) :=
by
  sorry

end necessary_but_not_sufficient_condition_l590_590002


namespace range_H_l590_590757

def H (x : ℝ) : ℝ := 2 * |2 * x + 3| - 3 * |x - 2|

theorem range_H : set.range H = set.univ :=
by sorry

end range_H_l590_590757


namespace paint_intensity_new_mixture_paint_intensity_result_l590_590519

theorem paint_intensity_new_mixture (I₁ I₂ : ℝ) (f : ℝ) (I_new : ℝ) 
  (h₁ : I₁ = 0.60) (h₂ : I₂ = 0.30) (hf : f = 2/3) : 
  I_new = (1 - f) * I₁ + f * I₂ :=
begin
  have : I_new = 0.40,
  sorry
end

theorem paint_intensity_result (I_new : ℝ) (h : I_new = 0.40) : I_new = 0.40 := 
h

end paint_intensity_new_mixture_paint_intensity_result_l590_590519


namespace spot_reachable_area_l590_590148

theorem spot_reachable_area (h : RegularHexagon (SideLength := 1)) (r : TetheredAtVertex (RopeLength := 3)) 
  : area_reachable_outside_doghouse h r = (49 / 6) * Real.pi := 
by sorry

end spot_reachable_area_l590_590148


namespace math_solution_l590_590511

noncomputable def math_problem : Prop :=
  ∀ (x y : ℝ), (x + y) / 2 = 20 ∧ real.sqrt (x * y) = 10 → x^2 + y^2 = 1400

theorem math_solution : math_problem :=
by
  -- open up the assumptions and state the known values and what needs to be shown.
  intros x y h,
  cases h with h1 h2,
  -- convert the conditions to more workable forms
  rw [div_eq_iff (two_ne_zero : (2 : ℝ) ≠ 0), mul_comm] at h1,
  rw [real.sqrt_eq_iff_sq_eq, pow_two] at h2,
  -- the value of x + y is now known
  have h12 : x + y = 40 := h1,
  -- the value of x * y is now known
  have hxy : x * y = 100 := h2,
  -- use the identity (x + y)^2 = x^2 + 2xy + y^2
  have h_sq : (x + y) ^ 2 = x^2 + 2 * (x * y) + y^2 := by ring,
  -- substitute known values into the identity
  rw [h12, hxy] at h_sq,
  -- solve for x^2 + y^2
  norm_num at h_sq,
  exact h_sq,
  sorry

end math_solution_l590_590511


namespace ball_and_ring_problem_l590_590773

theorem ball_and_ring_problem (x y : ℕ) (m_x m_y : ℕ) : 
  m_x + 2 = y ∧ 
  m_y = x + 2 ∧
  x * m_x + y * m_y - 800 = 2 * (y - x) ∧
  x^2 + y^2 = 881 →
  (x = 25 ∧ y = 16) ∨ (x = 16 ∧ y = 25) := 
by 
  sorry

end ball_and_ring_problem_l590_590773


namespace joan_total_ticket_spent_l590_590858

-- Define the ticket prices for this year and last year.
def this_year_prices : List ℝ := [35, 45, 50, 62]
def last_year_prices : List ℝ := [25, 30, 40, 45, 55, 60, 65, 70, 75]

-- Define the total expenditure.
def total_spent := (this_year_prices.sum + last_year_prices.sum)

-- The theorem we want to prove.
theorem joan_total_ticket_spent : total_spent = 657 := by
  -- The proof would go here.
  sorry

end joan_total_ticket_spent_l590_590858


namespace line_passes_through_point_has_correct_equation_l590_590360

theorem line_passes_through_point_has_correct_equation :
  (∃ (L : ℝ × ℝ → Prop), (L (-2, 5)) ∧ (∃ m : ℝ, m = -3 / 4 ∧ ∀ (x y : ℝ), L (x, y) ↔ y - 5 = -3 / 4 * (x + 2))) →
  ∀ x y : ℝ, (3 * x + 4 * y - 14 = 0) ↔ (y - 5 = -3 / 4 * (x + 2)) :=
by
  intro h_L
  sorry

end line_passes_through_point_has_correct_equation_l590_590360


namespace expected_lifetime_flashlight_l590_590096

noncomputable theory

variables (ξ η : ℝ) -- ξ and η are continuous random variables representing the lifetimes
variables (T : ℝ) -- T is the lifetime of the flashlight

-- Define the maximum lifetime of the flashlight
def max_lifetime (ξ η : ℝ) : ℝ := max ξ η

-- Given condition: the expectation of η is 4
axiom expectation_eta : E η = 4

-- Theorem statement: expected lifetime of the flashlight is at least 4
theorem expected_lifetime_flashlight (ξ η : ℝ) (h : T = max_lifetime ξ η) : 
  E (max_lifetime ξ η) ≥ 4 :=
by 
  sorry

end expected_lifetime_flashlight_l590_590096


namespace coefficient_of_x4_in_expansion_l590_590156

theorem coefficient_of_x4_in_expansion :
  let p := (x : ℕ) + 1
  let q := (x : ℕ) - 2
  let expanded := p * q^6
  coefficient_of_x4 expanded = -100 := 
by 
  sorry

end coefficient_of_x4_in_expansion_l590_590156


namespace expected_lifetime_at_least_four_l590_590103

universe u

variables (α : Type u) [MeasurableSpace α] {𝒫 : ProbabilitySpace α}
variables {ξ η : α → ℝ} [IsFiniteExpectation ξ] [IsFiniteExpectation η]

noncomputable def max_lifetime : α → ℝ := λ ω, max (ξ ω) (η ω)

theorem expected_lifetime_at_least_four 
  (h : ∀ ω, max (ξ ω) (η ω) ≥ η ω)
  (h_eta : @Expectation α _ _ η  = 4) : 
  @Expectation α _ _ max_lifetime ≥ 4 :=
by
  sorry

end expected_lifetime_at_least_four_l590_590103


namespace intersecting_chords_product_equal_l590_590509

-- Definition for the intersecting chords theorem
theorem intersecting_chords_product_equal 
  (A B C D M : Type) 
  (segment : Type) 
  [has_mul segment] 
  (chord_intersect : ∀ (A B C D M : Type), Prop)
  (angle_subtends_same_arc : ∀ (A B M : Type), Prop) 
  (h_inter : chord_intersect A B C D M)
  (h_angle1 : angle_subtends_same_arc A B M)
  (h_angle2 : angle_subtends_same_arc A C M) :
  (∃ (AM BM CM DM : segment), AM * BM = CM * DM) :=
by
  sorry

end intersecting_chords_product_equal_l590_590509


namespace sqrt_expr_eq_l590_590240

theorem sqrt_expr_eq : (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 :=
by sorry

end sqrt_expr_eq_l590_590240


namespace sin_cos_alpha_sin2alpha_over_sin_cos_alpha_l590_590409

theorem sin_cos_alpha (α : ℝ) (h : (cos α - (sqrt 2)/3, -1) = k • (sin α, 1)) (hbounds : -π ≤ α ∧ α ≤ 0) :
  sin α + cos α = (sqrt 2) / 3 := sorry

theorem sin2alpha_over_sin_cos_alpha (α : ℝ) (h : (cos α - (sqrt 2)/3, -1) = k • (sin α, 1)) (hbounds : -π ≤ α ∧ α ≤ 0) :
  (sin (2 * α)) / (sin α - cos α) = 7 / 12 := sorry

end sin_cos_alpha_sin2alpha_over_sin_cos_alpha_l590_590409


namespace regular_polygon_sides_l590_590752

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l590_590752


namespace pentagon_diagonals_sum_96_l590_590873

theorem pentagon_diagonals_sum_96 :
  ∀ (FG HI GH IJ FJ x y z : ℝ),
  FG = 4 → HI = 4 → GH = 9 → IJ = 9 → FJ = 13 → 
  (z^2 = 4 * x + 81) → (z^2 = 9 * y + 16) →
  (x * y = 36 + 13 * z) → (x * z = 4 * z + 169) → (y * z = 9 * z + 52) →
  (z = 13) → 
  x = (FG * FJ + GH * HI) / FG →
  y = (GH * FJ + IJ * HI) / GH →
  3 * x + y + z = 96 :=
by {
  intros,
  sorry
}

end pentagon_diagonals_sum_96_l590_590873


namespace complex_division_l590_590555

def i : ℂ := complex.I

theorem complex_division :
  (sqrt 2 * i^2015) / (1 - sqrt 2 * i) = (2 / 3 : ℂ) - ((sqrt 2) / 3) * i :=
by
  sorry

end complex_division_l590_590555


namespace prob_pass_kth_intersection_l590_590265

variable {n k : ℕ}

-- Definitions based on problem conditions
def prob_approach_highway (n : ℕ) : ℚ := 1 / n
def prob_exit_highway (n : ℕ) : ℚ := 1 / n

-- Theorem stating the required probability
theorem prob_pass_kth_intersection (h_n : n > 0) (h_k : k > 0) (h_k_le_n : k ≤ n) :
  (prob_approach_highway n) * (prob_exit_highway n * n) * (2 * k - 1) / n ^ 2 = 
  (2 * k * n - 2 * k ^ 2 + 2 * k - 1) / n ^ 2 := sorry

end prob_pass_kth_intersection_l590_590265


namespace hall_breadth_l590_590264

theorem hall_breadth (l : ℝ) (w_s l_s b : ℝ) (n : ℕ)
  (hall_length : l = 36)
  (stone_width : w_s = 0.4)
  (stone_length : l_s = 0.5)
  (num_stones : n = 2700)
  (area_paving : l * b = n * (w_s * l_s)) :
  b = 15 := by
  sorry

end hall_breadth_l590_590264


namespace Petya_time_comparison_l590_590675

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l590_590675


namespace servant_leaving_months_l590_590410

-- The given conditions
def total_salary_year : ℕ := 90 + 110
def monthly_salary (months: ℕ) : ℕ := (months * total_salary_year) / 12
def total_received : ℕ := 40 + 110

-- The theorem to prove
theorem servant_leaving_months (months : ℕ) (h : monthly_salary months = total_received) : months = 9 :=
by {
    sorry
}

end servant_leaving_months_l590_590410


namespace find_ω_l590_590809

noncomputable def ω : ℝ := 2/3

def f (x ω : ℝ) : ℝ := sin (ω * x - π/6) + 1/2

variables (α β : ℝ) (ω > 0) (h1 : f α ω = -1/2) (h2 : f β ω = 1/2) (h3 : abs (α - β) = 3*π/4)

theorem find_ω : ω = 2/3 :=
by
  sorry

end find_ω_l590_590809


namespace fruit_seller_sold_150_apples_l590_590638

variable (C S : ℝ) (N : ℝ)

def selling_price := 1.25 * C

def total_gain := 30 * S

def gains_equation (gain_percent : ℝ) :=
  gain_percent = 0.25

def number_of_apples_sold (gain : ℝ) (selling_price_30 : ℝ) : ℝ :=
  (selling_price_30 * gain) / (0.25 * 1.25)

theorem fruit_seller_sold_150_apples (H1: total_gain = 37.5) (H2: gains_equation 0.25) : N = 150 := by
  sorry

end fruit_seller_sold_150_apples_l590_590638


namespace socks_selection_l590_590635

theorem socks_selection :
  let red_socks := 120
  let green_socks := 90
  let blue_socks := 70
  let black_socks := 50
  let yellow_socks := 30
  let total_socks :=  red_socks + green_socks + blue_socks + black_socks + yellow_socks 
  (∀ k : ℕ, k ≥ 1 → k ≤ total_socks → (∃ p : ℕ, p = 12 → (p ≥ k / 2)) → k = 28) :=
by
  sorry

end socks_selection_l590_590635


namespace at_most_two_integer_solutions_l590_590783

theorem at_most_two_integer_solutions (m : ℕ) : 
  ∃ at_most_two : ℕ, at_most_two ≤ 2 ∧ 
  ∀ a : ℤ, (⌊(a : ℝ) - real.sqrt a⌋ : ℤ) = m → ∃! n ∈ finset.range (at_most_two + 1), (a = n) :=
sorry

end at_most_two_integer_solutions_l590_590783


namespace g_minus3_is_correct_l590_590924

theorem g_minus3_is_correct (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 3 * x^2) : 
  g (-3) = 247 / 39 :=
by
  sorry

end g_minus3_is_correct_l590_590924


namespace length_segment_l590_590575

/--
Given a cylinder with a radius of 5 units capped with hemispheres at each end and having a total volume of 900π,
prove that the length of the line segment AB is 88/3 units.
-/
theorem length_segment (r : ℝ) (V : ℝ) (h : ℝ) : r = 5 ∧ V = 900 * Real.pi → h = 88 / 3 := by
  sorry

end length_segment_l590_590575


namespace fifth_generation_tail_length_l590_590064

theorem fifth_generation_tail_length :
  let initial_length := 16.0
  let growth_rates := [0.25, 0.18, 0.12, 0.06]
  let calc_length (len: ℝ) (rate: ℝ) := len * (1 + rate)
  let length2 := calc_length initial_length growth_rates[0]
  let length3 := calc_length length2 growth_rates[1]
  let length4 := calc_length length3 growth_rates[2]
  let length5 := calc_length length4 growth_rates[3]
  length5 ≈ 28.01792 :=
by {
  sorry
}

end fifth_generation_tail_length_l590_590064


namespace range_of_b_range_of_m_l590_590813

open Real

-- Part 1: Proving the range of b
theorem range_of_b : 
  ∀ b : ℝ, (∃ x : ℝ, x^2 < b * (x - 1)) ↔ (b < 0 ∨ b > 4) := 
  sorry

-- Part 2: Proving the range of m
theorem range_of_m : 
  ∀ m : ℝ, (∀ x ∈ Icc (0 : ℝ) 1, abs (x^2 - m*x + 1 - m - m^2) isMonotoneIncOn [0, 1]) ↔ (m ∈ Icc (-1) 0 ∨ m ∈ Ioc 2 ∞) := 
  sorry

end range_of_b_range_of_m_l590_590813


namespace find_m_l590_590616

theorem find_m (m : ℕ) (h1 : List ℕ := [27, 32, 39, m, 46, 47])
            (h2 : List ℕ := [30, 31, 34, 41, 42, 45])
            (h3 : (39 + m) / 2 = 42) :
            m = 45 :=
by {
  sorry
}

end find_m_l590_590616


namespace area_of_square_field_l590_590154

theorem area_of_square_field (x : ℝ) 
  (h₁ : 1.10 * (4 * x - 2) = 732.6) : 
  x = 167 → x ^ 2 = 27889 := by
  sorry

end area_of_square_field_l590_590154


namespace probability_different_tens_digit_l590_590528

open Nat

theorem probability_different_tens_digit :
  let total_ways := choose 70 6,
      favorable_ways := 7 * 10^6
  in 
    (favorable_ways : ℝ) / total_ways = (2000 / 3405864 : ℝ) :=
by
  have h1 : total_ways = 70.choose 6 := rfl
  have h2 : favorable_ways = 7 * 10^6 := rfl
  rw [h1, h2]
  sorry

end probability_different_tens_digit_l590_590528


namespace selection_count_Group3_selection_count_Group4_selection_count_Group5_probability_A_or_B_l590_590496

/-
  Conditions:
-/
def Group3 : ℕ := 18
def Group4 : ℕ := 12
def Group5 : ℕ := 6
def TotalParticipantsToSelect : ℕ := 12
def TotalFromGroups345 : ℕ := Group3 + Group4 + Group5

/-
  Questions:
  1. Prove that the number of people to be selected from each group using stratified sampling:
\ 2. Prove that the probability of selecting at least one of A or B from Group 5 is 3/5.
-/

theorem selection_count_Group3 : 
  (Group3 * TotalParticipantsToSelect / TotalFromGroups345) = 6 := 
  by sorry

theorem selection_count_Group4 : 
  (Group4 * TotalParticipantsToSelect / TotalFromGroups345) = 4 := 
  by sorry

theorem selection_count_Group5 : 
  (Group5 * TotalParticipantsToSelect / TotalFromGroups345) = 2 := 
  by sorry

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_A_or_B : 
  (combination 6 2 - combination 4 2) / combination 6 2 = 3 / 5 := 
  by sorry

end selection_count_Group3_selection_count_Group4_selection_count_Group5_probability_A_or_B_l590_590496


namespace monotonic_decrease_condition_l590_590223

open Set

/--
Prove that \( \forall a \in \mathbb{R}, (a > 12) \implies \forall x \in (1, 2), f(x) = x^3 - ax \text{ is monotonically decreasing} \)

Also prove that \(\exists b, b = 12 \land \forall x \in (1, 2), f(x) = x^3 - bx \text{ is monotonically decreasing} \)
--/
theorem monotonic_decrease_condition (a : ℝ) :
  (a > 12) → (∀ x : ℝ, (1 < x ∧ x < 2) → (3 * x^2 - a) ≤ 0) ∧
  (∃ b : ℝ, (b = 12) ∧ (∀ x : ℝ, (1 < x ∧ x < 2) → (3 * x^2 - b) ≤ 0)) :=
by
  intro ha
  split
  { intros x hx
    sorry
  }
  {
    use 12
    split
    { refl }
    { intros x hx
      sorry
    }
  }

end monotonic_decrease_condition_l590_590223


namespace parabola_vertex_l590_590934

theorem parabola_vertex (y x : ℝ) : y^2 - 4*y + 3*x + 7 = 0 → (x = -1 ∧ y = 2) := 
sorry

end parabola_vertex_l590_590934


namespace closest_integer_to_cube_root_of_150_l590_590601

theorem closest_integer_to_cube_root_of_150 : 
  ∃ (n : ℤ), ∀ m : ℤ, abs (150 - 5 ^ 3) < abs (150 - m ^ 3) → n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590601


namespace binary_sum_correct_l590_590281

theorem binary_sum_correct :
  nat.binary_add (nat.binary_add (nat.binary_add 0b1101 0b101) 0b1110) 0b10001 = 0b1000001 := by
  sorry

end binary_sum_correct_l590_590281


namespace trapezoid_inscribed_circles_radii_l590_590361

open Real

variables (a b m n : ℝ)
noncomputable def r := (a * sqrt b) / (sqrt a + sqrt b)
noncomputable def R := (b * sqrt a) / (sqrt a + sqrt b)

theorem trapezoid_inscribed_circles_radii
  (h : a < b)
  (hM : m = sqrt (a * b))
  (hN : m = sqrt (a * b)) :
  (r a b = (a * sqrt b) / (sqrt a + sqrt b)) ∧
  (R a b = (b * sqrt a) / (sqrt a + sqrt b)) :=
by
  sorry

end trapezoid_inscribed_circles_radii_l590_590361


namespace regular_polygon_sides_l590_590751

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l590_590751


namespace angle_A_is_120_degrees_l590_590058

theorem angle_A_is_120_degrees
  (a b c : ℝ)
  (h : a^2 = b^2 + b*c + c^2) :
  ∠A = 120 :=
sorry

end angle_A_is_120_degrees_l590_590058


namespace at_most_four_greater_than_one_l590_590239

open Real

theorem at_most_four_greater_than_one
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : (sqrt (a * b) - 1) * (sqrt (b * c) - 1) * (sqrt (a * c) - 1) = 1) :
  (∃ n ≤ 4, {x : ℝ | ∃ f : ℤ, (f : ℝ × ℝ × ℝ → ℝ) (a, b, c) = x ∧ x > 1} ⟨a - b / c, a - c / b, b - a / c, b - c / a, c - a / b, c - b / a⟩.to_finset.card = n) := by
sorry

end at_most_four_greater_than_one_l590_590239


namespace regular_polygon_sides_l590_590731

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ α, α = 160 → ∑ x in range n, (180 - (180 - 160)) = 360) : n = 18 :=
by
  sorry

end regular_polygon_sides_l590_590731


namespace find_phi_l590_590874

noncomputable def solve_for_phi (a : Real) : Real :=
  if (∃ (x : Real), 0 ≤ x ∧ x ≤ π ∧ sin (2 * x + (π / 3)) = a ∧ (7 * π / 6) = x 
    + (if ∃ (x : Real), 0 ≤ x ∧ x ≤ π ∧ sin (2 * x + (4 * π / 3)) = a then x else 0)) then π / 3 else 
  if (∃ (x : Real), 0 ≤ x ∧ x ≤ π ∧ sin (2 * x + (4 * π / 3)) = a ∧ (7 * π / 6) = x 
    + (if ∃ (x : Real), 0 ≤ x ∧ x ≤ π ∧ sin (2 * x + (π / 3)) = a then x else 0)) then 4 * π / 3 
  else 0

theorem find_phi (a : Real) (phi : Real) (cond1 : phi ∈ Set.Ico 0 (2 * π)) 
    (cond2 : (∃ (x1 x2 x3 : Real), 0 ≤ x1 ∧ x1 ≤ π ∧ 0 ≤ x2 ∧ x2 ≤ π ∧ 0 ≤ x3 ∧ x3 ≤ π ∧ sin (2*x1 + phi) = a 
             ∧ sin (2*x2 + phi) = a ∧ sin (2*x3 + phi) = a ∧ x1 + x2 + x3 = 7 * π / 6))
    : phi = π / 3 ∨ phi = 4 * π / 3 := sorry

end find_phi_l590_590874


namespace value_of_expression_l590_590826

theorem value_of_expression (m n : ℝ) (h : m + n = 4) : 2 * m^2 + 4 * m * n + 2 * n^2 - 5 = 27 :=
  sorry

end value_of_expression_l590_590826


namespace not_possible_cut_l590_590456

theorem not_possible_cut (n : ℕ) : 
  let chessboard_area := 8 * 8
  let rectangle_area := 3
  let rectangles_needed := chessboard_area / rectangle_area
  rectangles_needed ≠ n :=
by
  sorry

end not_possible_cut_l590_590456


namespace beneficial_for_kati_l590_590862

variables (n : ℕ) (x y : ℝ)

theorem beneficial_for_kati (hn : n > 0) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / (n + 2) > (x + y / 2) / (n + 1) :=
sorry

end beneficial_for_kati_l590_590862


namespace first_term_geometric_series_l590_590578

theorem first_term_geometric_series (a r : ℝ) (h1: |r| < 1) 
    (h2 : ∑' n : ℕ, a * r^n = 15) 
    (h3 : ∑' n : ℕ, (a * r^n)^2 = 45) : 
    a = 5 := 
begin 
    sorry 
end

end first_term_geometric_series_l590_590578


namespace pizza_cut_possible_l590_590895

theorem pizza_cut_possible (N : ℕ) (h1 : N = 201 ∨ N = 400) :
  ∃ (cuts : ℕ), cuts ≤ 100 ∧ 
  ∀ (friends : fin N), (total_area : real), 
  evenly_distributed (cuts = 100) (total_area = 1) :=
sorry

end pizza_cut_possible_l590_590895


namespace sum_of_squares_of_real_roots_l590_590868

theorem sum_of_squares_of_real_roots (k : ℝ)
  (h : ∃ x1 x2 x3 x4 : ℝ, 
     (x1 * x2 * x3 * x4 = -2013) ∧
     (x1 + x2 + x3 + x4 = -2 - 2 * k) ∧
     (x1^2 + x2^2 + x3^2 + x4^2 = 4 + 4 * k^2 + (2 + 2 * k)^2 + (1 + 2 * k) + (2 * k)^2)) :
  let P := polynomial in real,
  P.eval x = X^4 + 2 * X^3 + (2 + 2 * k) * X^2 + (1 + 2 * k) * X + 2 * k :=
  x1^2 + x2^2 + x3^2 + x4^2 = 4027 := 
begin 
  sorry
end

end sum_of_squares_of_real_roots_l590_590868


namespace largest_divisible_number_l590_590236

theorem largest_divisible_number : ∃ n, n = 9950 ∧ n ≤ 9999 ∧ (∀ m, m ≤ 9999 ∧ m % 50 = 0 → m ≤ n) :=
by {
  sorry
}

end largest_divisible_number_l590_590236


namespace total_triangles_in_rectangle_l590_590298

/-- Consider a rectangle with vertices A, B, C, D such that AB and CD are longer than AD and BC.
Let the diagonals of the rectangle be AC and BD.
Let M, N, P, and Q be the midpoints of sides AB, BC, CD, and DA, respectively.
The segments joining these midpoints form another rectangle inside the original one.
Prove that the total number of triangles formed in this figure is 20. -/
theorem total_triangles_in_rectangle : 
  ∃ (A B C D M N P Q : Point), 
      is_rectangle A B C D ∧ 
      is_diagonal A C B D ∧ 
      is_midpoint M A B ∧ 
      is_midpoint N B C ∧ 
      is_midpoint P C D ∧ 
      is_midpoint Q D A ∧ 
      number_of_triangles A B C D M N P Q = 20 :=
begin
  sorry,
end

end total_triangles_in_rectangle_l590_590298


namespace regular_polygon_sides_l590_590739

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l590_590739


namespace find_b6_l590_590106

def a₀ : ℚ := 3
def b₀ : ℚ := 4

def a (n : ℕ) : ℚ := 
  if n = 0 then a₀
  else (a (n - 1) ^ 2 + b (n - 1)) / (b (n - 1))

def b (n : ℕ) : ℚ := 
  if n = 0 then b₀
  else (b (n - 1) ^ 2 + a (n - 1)) / (a (n - 1))

theorem find_b6 (p q x y : ℤ) : b 6 = p ^ x / q ^ y := sorry

end find_b6_l590_590106


namespace regular_polygon_sides_l590_590743

theorem regular_polygon_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle i = 160)) : n = 18 :=
by
  -- Proof goes here
  sorry

end regular_polygon_sides_l590_590743


namespace petya_time_l590_590693

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l590_590693


namespace angle_CGD_l590_590075

theorem angle_CGD (O A B F E C D G : Type*)
  [Circular A B F E]
  (h1 : Diameter A B)
  (h2 : OnCircle F)
  (h3 : AF_not_diameter : ¬Diameter A F)
  (h4 : OnCircle E)
  (h5 : E_not_A_B : E ≠ A ∧ E ≠ B)
  (h6 : tangents_intersect_at C B F)
  (h7 : tangents_intersect_at D E F)
  (h8 : tangents_intersect_at G B E)
  (h9 : ∠BAF = 15)
  (h10 : ∠BAE = 43):
  ∠CGD = 94 :=
sorry

end angle_CGD_l590_590075


namespace ratio_of_black_to_white_after_border_l590_590272

def original_tiles (black white : ℕ) : Prop := black = 14 ∧ white = 21
def original_dimensions (length width : ℕ) : Prop := length = 5 ∧ width = 7

def border_added (length width l w : ℕ) : Prop := l = length + 2 ∧ w = width + 2

def total_white_tiles (initial_white new_white total_white : ℕ) : Prop :=
  total_white = initial_white + new_white

def black_white_ratio (black_tiles white_tiles : ℕ) (ratio : ℚ) : Prop :=
  ratio = black_tiles / white_tiles

theorem ratio_of_black_to_white_after_border 
  (black_white_tiles : ℕ → ℕ → Prop)
  (dimensions : ℕ → ℕ → Prop)
  (border : ℕ → ℕ → ℕ → ℕ → Prop)
  (total_white : ℕ → ℕ → ℕ → Prop)
  (ratio : ℕ → ℕ → ℚ → Prop)
  (black_tiles white_tiles initial_white total_white_new length width l w : ℕ)
  (rat : ℚ) :
  black_white_tiles black_tiles initial_white →
  dimensions length width →
  border length width l w →
  total_white initial_white (l * w - length * width) white_tiles →
  ratio black_tiles white_tiles rat →
  rat = 2 / 7 :=
by
  intros
  sorry

end ratio_of_black_to_white_after_border_l590_590272


namespace largest_integer_div_l590_590629

noncomputable def largest_integer_n (k : ℤ) : ℤ :=
  if h : ∃ n, ∀ d, (d > k → (d ∣ k) → d = n + 7) then (classical.some h) - 7 else -1

theorem largest_integer_div (k : ℤ) (hk : k = 1963) : ∃ n, largest_integer_n k = n ∧
  (n + 7 ∣ n^2 - 2012) :=
  by sorry

end largest_integer_div_l590_590629


namespace triangle_BEC_area_l590_590452

theorem triangle_BEC_area (A B C D E : Type)
  (AD_perpendicular_DC : is_perpendicular AD DC)
  (AD_length : AD = 4)
  (AB_length : AB = 4)
  (DC_length : DC = 10)
  (DE_length : DE = 7)
  (BE_parallel_AD : is_parallel BE AD) :
  area_triangle B E C = 6 := 
sorry

end triangle_BEC_area_l590_590452


namespace parabola_equation_l590_590333

-- Definitions and conditions
def vertex (p : ℝ × ℝ) : Prop := p = (3, 2)
def vertical_axis_of_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f (6 - x) = f x
def passes_through (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop := f (p.1) = p.2

-- Statement of the proof problem
theorem parabola_equation : 
  ∃ (a b c : ℝ), 
    (∀ x, (-4) * x^2 + 24 * x + (-34) = a * x^2 + b * x + c) ∧ 
    vertex (3, 2) ∧
    vertical_axis_of_symmetry (λ x, a * (x - 3)^2 + 2) ∧ 
    passes_through (λ x, a * (x - 3)^2 + 2) (2, -2) := 
begin
  sorry
end

end parabola_equation_l590_590333


namespace blocks_left_l590_590510

theorem blocks_left (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 59) (h_used : used_blocks = 36) : initial_blocks - used_blocks = 23 :=
by
  -- proof here
  sorry

end blocks_left_l590_590510


namespace actual_time_greater_than_planned_time_l590_590670

def planned_time (a V : ℝ) : ℝ := a / V

def actual_time (a V : ℝ) : ℝ := (a / (2.5 * V)) + (a / (1.6 * V))

theorem actual_time_greater_than_planned_time (a V : ℝ) (hV : V > 0) : 
  actual_time a V > planned_time a V :=
by 
  sorry

end actual_time_greater_than_planned_time_l590_590670


namespace sum_possible_values_l590_590479

theorem sum_possible_values (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
  (h3 : m * n^2 + 876 = 4 * m * n + 217 * n) : m ∈ {18, 75} →
  ∃ S : ℕ, S = 93 := by
  sorry

end sum_possible_values_l590_590479


namespace rahim_paid_first_shop_l590_590910

theorem rahim_paid_first_shop : 
  let num_books_first := 65 in
  let num_books_second := 50 in
  let amount_second := 920 in
  let avg_price := 18 in
  let total_books := num_books_first + num_books_second in
  let total_amount := total_books * avg_price in
  let amount_first := total_amount - amount_second in
  amount_first = 1150 :=
by
  sorry

end rahim_paid_first_shop_l590_590910


namespace max_profit_l590_590636

-- Define the total cost function G(x)
def G (x : ℝ) : ℝ := 15 + 5 * x

-- Define the sales revenue function R(x)
def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then
    -2 * x ^ 2 + 21 * x + 1
  else if x > 5 then 56
  else 0

-- Define the profit function f(x)
def f(x : ℝ) : ℝ :=
  R(x) - G(x)

-- Statement to prove the maximum profit
theorem max_profit :
  ∃ x, x = 4 ∧ f x = 18 ∧ (∀ y, y ≥ 0 → f y ≤ 18) := by
  sorry

end max_profit_l590_590636


namespace midline_of_triangle_false_l590_590061

theorem midline_of_triangle_false (A B C D E : Type) [LinearOrder A]
  {line_segment : (A → B) → (B → C) → (C → A) → Prop}
  (mid_segment_length : ∀ (L : A → B) (M : B → C) (N : C → A),
    L A = (1 / 2 : Real) * distance between B and C) :
  ¬ (line_segment A B C D E → (line_segment D E = (1 / 2 : Real) * distance between A and B)) :=
by
  sorry

end midline_of_triangle_false_l590_590061


namespace find_river_current_speed_l590_590657

variable (t s : ℝ)

-- Define swimmer's speed in still water
-- (Although we do not need to define it directly, it's implied by conditions)
variable (v₁ : ℝ)

-- Define the speed of the river current (v₂)
def river_current_speed : ℝ := s / (2 * t)

theorem find_river_current_speed (t s : ℝ) :
  ∃ v₂ : ℝ, v₂ = s / (2 * t) :=
by
  use s / (2 * t)
  sorry

end find_river_current_speed_l590_590657


namespace initial_sheep_count_l590_590019

theorem initial_sheep_count 
    (S : ℕ)
    (initial_horses : ℕ := 100)
    (initial_chickens : ℕ := 9)
    (gifted_goats : ℕ := 37)
    (male_animals : ℕ := 53)
    (total_animals_half : ℕ := 106) :
    ((initial_horses + S + initial_chickens) / 2 + gifted_goats = total_animals_half) → 
    S = 29 :=
by
  intro h
  sorry

end initial_sheep_count_l590_590019


namespace smallest_prime_sum_l590_590765

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_sum_of_distinct_primes (n k : ℕ) (s : List ℕ) : Prop :=
  s.length = k ∧ (∀ x ∈ s, is_prime x) ∧ (∀ (x y : ℕ), x ≠ y → x ∈ s → y ∈ s → x ≠ y) ∧ s.sum = n

theorem smallest_prime_sum :
  (is_prime 61) ∧ 
  (∃ s2, is_sum_of_distinct_primes 61 2 s2) ∧ 
  (∃ s3, is_sum_of_distinct_primes 61 3 s3) ∧ 
  (∃ s4, is_sum_of_distinct_primes 61 4 s4) ∧ 
  (∃ s5, is_sum_of_distinct_primes 61 5 s5) ∧ 
  (∃ s6, is_sum_of_distinct_primes 61 6 s6) :=
by
  sorry

end smallest_prime_sum_l590_590765


namespace find_f1_l590_590107

def f : ℕ → ℕ
| x := if x ≥ 2 then 2 * x - 1 else f (f (x + 1)) + 1

theorem find_f1 : f 1 = 6 := by
  sorry

end find_f1_l590_590107


namespace closest_integer_to_cube_root_of_150_l590_590600

theorem closest_integer_to_cube_root_of_150 : 
  ∃ (n : ℤ), ∀ m : ℤ, abs (150 - 5 ^ 3) < abs (150 - m ^ 3) → n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l590_590600


namespace complex_quadrant_l590_590173

-- Declare the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Declare the complex number z as per the condition
noncomputable def z : ℂ := (2 * i) / (i - 1)

-- State and prove that the complex number z lies in the fourth quadrant
theorem complex_quadrant : (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end complex_quadrant_l590_590173


namespace max_f_value_l590_590339

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt (x + 15) + sqrt (17 - x) + sqrt x

theorem max_f_value : ∀ x ∈ Icc (0 : ℝ) 17, f x ≤ sqrt 15 + sqrt 17 :=
begin
  intros x hx,
  sorry
end

end max_f_value_l590_590339


namespace log_factorial_solution_l590_590710

theorem log_factorial_solution (n : ℕ) (h : 2 < n) :
  log 2 ((n - 2)!) + log 2 ((n - 1)!) + 4 = 2 * log 2 (n!) → n = 8 :=
by
  sorry

end log_factorial_solution_l590_590710


namespace correct_calculation_l590_590221

theorem correct_calculation :
  (∀ (x y : ℝ), x = 3 ∧ y = 5 → sqrt x * sqrt y = sqrt (x * y)) :=
begin
  intros x y h,
  cases h with h1 h2,
  simp [h1, h2],
end

example : sqrt 3 * sqrt 5 = sqrt 15 :=
by { apply correct_calculation, split; refl }

end correct_calculation_l590_590221


namespace find_a_l590_590487

def A (a : ℝ) : Set ℝ := {a^2, a + 1, -1}
def B (a : ℝ) : Set ℝ := {2a - 1, |a - 2|, 3a^2 + 4}

theorem find_a (a : ℝ) (h : A a ∩ B a = {-1}) : a = 0 := by
  sorry

end find_a_l590_590487


namespace water_tank_height_l590_590187

theorem water_tank_height 
  (radius height : ℝ)
  (fill_percentage : ℝ)
  (h_radius : radius = 24)
  (h_height : height = 72)
  (h_fill_percentage : fill_percentage = 0.4) : 
  ∃ (a b : ℕ), 
    let height_of_water := a * (Real.cbrt b : ℝ)
    in height_of_water = height * Real.cbrt (4 / 10) ∧ a + b = 52 :=
by
  use 36
  use 16
  sorry

end water_tank_height_l590_590187


namespace probability_six_integers_diff_tens_l590_590526

-- Defining the range and conditions for the problem
def set_of_integers : Finset ℤ := Finset.range 70 \ Finset.range 10

def has_different_tens_digit (s : Finset ℤ) : Prop :=
  (s.card = 6) ∧ (∀ x y ∈ s, x ≠ y → (x / 10) ≠ (y / 10))

noncomputable def num_ways_choose_six_diff_tens : ℚ :=
  ((7 : ℚ) * (10^6 : ℚ))

noncomputable def total_ways_choose_six : ℚ :=
  (Nat.choose 70 6 : ℚ)

noncomputable def probability_diff_tens : ℚ :=
  num_ways_choose_six_diff_tens / total_ways_choose_six

-- Statement claiming the required probability
theorem probability_six_integers_diff_tens :
  probability_diff_tens = 1750 / 2980131 :=
by
  sorry

end probability_six_integers_diff_tens_l590_590526


namespace complex_solution_l590_590008

theorem complex_solution (z : ℂ) (h : (1 + z) * complex.I = 1 - z) : z = -complex.I :=
sorry

end complex_solution_l590_590008


namespace solve_for_w_squared_l590_590833

-- Define the original equation
def eqn (w : ℝ) := 2 * (w + 15)^2 = (4 * w + 9) * (3 * w + 6)

-- Define the goal to prove w^2 = 6.7585 based on the given equation
theorem solve_for_w_squared : ∃ w : ℝ, eqn w ∧ w^2 = 6.7585 :=
by
  sorry

end solve_for_w_squared_l590_590833


namespace parabola_point_distance_to_focus_is_3_l590_590649

-- Define the parabola equation.
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola.
def focus : ℝ × ℝ := (2, 0)

-- Distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Problem statement to prove x-coordinate
theorem parabola_point_distance_to_focus_is_3 (x y : ℝ) 
  (H : parabola x y) (H_dist : distance (x, y)(focus) = 5) : x = 3 := by
  sorry

end parabola_point_distance_to_focus_is_3_l590_590649


namespace greatest_integer_not_exceeding_100y_l590_590110

noncomputable def y : ℝ := (∑ n in Finset.range 30, Real.cos (3 * (n + 1 : ℝ))) / (∑ n in Finset.range 30, Real.sin (3 * (n + 1 : ℝ)))

theorem greatest_integer_not_exceeding_100y : ⌊100 * y⌋ = 41 :=
by
  sorry

end greatest_integer_not_exceeding_100y_l590_590110


namespace probability_unique_tens_digits_l590_590550

theorem probability_unique_tens_digits :
  let num_ways := 10^6 in
  let total_combinations := Nat.choose 70 6 in
  (num_ways : ℚ) / total_combinations = 625 / 74440775 :=
by 
  sorry

end probability_unique_tens_digits_l590_590550


namespace trig_identity_l590_590005

theorem trig_identity (a : ℝ) (h : sin a = -2 * cos a) : cos (2 * a + (Float.pi / 2)) = 4 / 5 :=
sorry

end trig_identity_l590_590005


namespace find_z_range_of_m_l590_590374

open Complex

-- Part 1: Prove z = 2 - 4i given the conditions.
theorem find_z (z : ℂ) (H1 : z + 3 + 4 * I ∈ ℝ) (H2 : (z / (1 - 2 * I)) ∈ ℝ) :
  z = 2 - 4 * I := by
  sorry

-- Part 2: Prove the range of m given the additional condition.
theorem range_of_m (z : ℂ) (m : ℝ) (H3 : (z - m * I) ^ 2 ∈ {z : ℂ | z.im > 0 ∧ z.re < 0}) :
  m < -6 := by
  sorry

end find_z_range_of_m_l590_590374


namespace probability_within_circle_correct_l590_590927

def is_within_circle (m n : ℕ) (r : ℕ) : Prop :=
  m * m + n * n ≤ r * r

noncomputable def probability_within_circle : ℚ :=
  let total_outcomes := 36 in
  let favorable_outcomes := (finset.product (finset.range 1 7) (finset.range 1 7)).filter (λ p, is_within_circle p.fst p.snd 4) in
  ↑(favorable_outcomes.card) / total_outcomes

theorem probability_within_circle_correct : probability_within_circle = 2 / 9 := by
  sorry

end probability_within_circle_correct_l590_590927


namespace g_minus3_is_correct_l590_590923

theorem g_minus3_is_correct (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 3 * x^2) : 
  g (-3) = 247 / 39 :=
by
  sorry

end g_minus3_is_correct_l590_590923


namespace cube_root_simplification_l590_590216

theorem cube_root_simplification (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : b = 1) : 3
  := sorry

end cube_root_simplification_l590_590216


namespace probability_of_different_tens_digits_l590_590538

open Finset

-- Define the basic setup
def integers (n : ℕ) : Finset ℕ := {i in (range n) | i ≥ 10 ∧ i ≤ 79}

def tens_digit (n : ℕ) : ℕ := n / 10

def six_integers_with_different_tens_digits (s : Finset ℕ) : Prop :=
  s.card = 6 ∧ (s.map ⟨tens_digit, by simp⟩).card = 6

def favorable_ways : ℕ :=
  7 * 10^6

def total_ways : ℕ :=
  nat.choose 70 6

noncomputable def probability : ℚ :=
  favorable_ways / total_ways

-- The main statement
theorem probability_of_different_tens_digits :
  ∀ (s : Finset ℕ), six_integers_with_different_tens_digits s → 
  probability = 175 / 2980131 :=
begin
  intros s h,
  sorry
end

end probability_of_different_tens_digits_l590_590538


namespace Kannon_apples_l590_590071

theorem Kannon_apples :
  let A := 7 in
  let total_fruits_last_night := 3 + 1 + 4 in
  let total_fruits_today := A + 10 + 2 * A in
  total_fruits_last_night + total_fruits_today = 39 →
  A - 3 = 4 :=
by
  intros A total_fruits_last_night total_fruits_today h,
  sorry

end Kannon_apples_l590_590071


namespace probability_of_triangle_area_one_l590_590919

-- Define the set of vectors
def vectors := { (2, 1), (2, 3), (4, 1), (4, 3) }

-- Area function using cross product for vectors
def area_of_triangle (a b : ℕ × ℕ) : ℝ :=
  abs ((a.1 * b.2 - a.2 * b.1) / 2)

-- Check if the area of a triangle is 1
def area_is_one (a b : ℕ × ℕ) : Prop :=
  area_of_triangle a b = 1

-- The set of all pairs of vectors
def all_pairs := [(2, 1), (2, 3), (4, 1), (4, 3)].combination 2

-- The probability that a randomly chosen pair of vectors forms a triangle with an area of 1
def probability_area_one : ℝ :=
  let total_pairs := all_pairs.length
  let valid_pairs := all_pairs.filter (λ ab => area_is_one ab[0] ab[1]).length
  valid_pairs / total_pairs

theorem probability_of_triangle_area_one :
  probability_area_one = 1 / 3 :=
sorry

end probability_of_triangle_area_one_l590_590919


namespace regular_polygon_sides_l590_590734

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ α, α = 160 → ∑ x in range n, (180 - (180 - 160)) = 360) : n = 18 :=
by
  sorry

end regular_polygon_sides_l590_590734


namespace least_constant_M_l590_590309

theorem least_constant_M :
  ∀ (x : Fin 2009 → ℝ), (∀ i, x i > 0) →
  (∑ i in (Finset.range 2009), x i / (x i + x (i + 1) % 2009)) < 2008 := 
by
  sorry

end least_constant_M_l590_590309


namespace Joey_SAT_Weeks_l590_590859

theorem Joey_SAT_Weeks
    (hours_per_night : ℕ) (nights_per_week : ℕ)
    (hours_per_weekend_day : ℕ) (days_per_weekend : ℕ)
    (total_hours : ℕ) (weekly_hours : ℕ) (weeks : ℕ)
    (h1 : hours_per_night = 2) (h2 : nights_per_week = 5)
    (h3 : hours_per_weekend_day = 3) (h4 : days_per_weekend = 2)
    (h5 : total_hours = 96) (h6 : weekly_hours = 16)
    (h7 : weekly_hours = (hours_per_night * nights_per_week) + (hours_per_weekend_day * days_per_weekend)) :
  weeks = total_hours / weekly_hours :=
sorry

end Joey_SAT_Weeks_l590_590859


namespace digit_expression_equals_2021_l590_590987

theorem digit_expression_equals_2021 :
  ∃ (f : ℕ → ℕ), 
  (f 0 = 0 ∧
   f 1 = 1 ∧
   f 2 = 2 ∧
   f 3 = 3 ∧
   f 4 = 4 ∧
   f 5 = 5 ∧
   f 6 = 6 ∧
   f 7 = 7 ∧
   f 8 = 8 ∧
   f 9 = 9 ∧
   43 * (8 * 5 + 7) + 0 * 1 * 2 * 6 * 9 = 2021) :=
sorry

end digit_expression_equals_2021_l590_590987


namespace focal_length_ellipse_l590_590562

theorem focal_length_ellipse :
  let a := 2
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 :=
by
  sorry

end focal_length_ellipse_l590_590562


namespace gcd_459_357_l590_590970

theorem gcd_459_357 : gcd 459 357 = 51 := 
sorry

end gcd_459_357_l590_590970


namespace product_of_solutions_l590_590574

theorem product_of_solutions :
  ∀ x : ℝ, (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4) →
  (∀ x1 x2 : ℝ, (x1 ≠ x2) → (x = x1 ∨ x = x2) → x1 * x2 = 0) :=
by
  sorry

end product_of_solutions_l590_590574


namespace minimum_value_ineq_l590_590763

noncomputable def expression (x y z : ℝ) : ℝ :=
  (x * y) / z + (y * z) / x + (z * x) / y

theorem minimum_value_ineq (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ inf_val, inf_val = √3 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → 0 < x → 0 < y → 0 < z → expression x y z ≥ inf_val :=
sorry

end minimum_value_ineq_l590_590763


namespace largest_number_in_set_l590_590356

theorem largest_number_in_set {a : ℤ} (h : a = -4) :
  let s := { -3 * a, 4 * a, 24 / a, a^2, 2 * a + 1, 1 }
  in ∃ x ∈ s, x = a^2 ∧ ∀ y ∈ s, y <= x :=
begin
  sorry
end

end largest_number_in_set_l590_590356


namespace sum_lent_borrowed_l590_590645

-- Define the given conditions and the sum lent
def sum_lent (P r t : ℝ) (I : ℝ) : Prop :=
  I = P * r * t / 100 ∧ I = P - 1540

-- Define the main theorem to be proven
theorem sum_lent_borrowed : 
  ∃ P : ℝ, sum_lent P 8 10 ((4 * P) / 5) ∧ P = 7700 :=
by
  sorry

end sum_lent_borrowed_l590_590645


namespace Petya_time_comparison_l590_590676

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l590_590676


namespace probability_sum_greater_than_five_l590_590962

-- Definitions for the conditions
def die_faces := {1, 2, 3, 4, 5, 6}
def possible_outcomes := (die_faces × die_faces).to_finset
def favorable_outcomes := possible_outcomes.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd > 5)
def probability_of_sum_greater_than_five := (favorable_outcomes.card : ℚ) / possible_outcomes.card

-- Problem statement
theorem probability_sum_greater_than_five :
  probability_of_sum_greater_than_five = 13 / 18 :=
sorry

end probability_sum_greater_than_five_l590_590962


namespace expected_lifetime_at_least_four_l590_590105

universe u

variables (α : Type u) [MeasurableSpace α] {𝒫 : ProbabilitySpace α}
variables {ξ η : α → ℝ} [IsFiniteExpectation ξ] [IsFiniteExpectation η]

noncomputable def max_lifetime : α → ℝ := λ ω, max (ξ ω) (η ω)

theorem expected_lifetime_at_least_four 
  (h : ∀ ω, max (ξ ω) (η ω) ≥ η ω)
  (h_eta : @Expectation α _ _ η  = 4) : 
  @Expectation α _ _ max_lifetime ≥ 4 :=
by
  sorry

end expected_lifetime_at_least_four_l590_590105


namespace dhoni_savings_l590_590617

-- Define the initial conditions and parameters
def dhoniSpentOnRent : ℝ := 20 / 100
def dhoniSpentOnDishwasher : ℝ := dhoniSpentOnRent - 5 / 100
def totalSpent : ℝ := dhoniSpentOnRent + dhoniSpentOnDishwasher
def percentLeftover : ℝ := 1 - totalSpent

-- State the theorem to be proven
theorem dhoni_savings : percentLeftover = 65 / 100 := by
  sorry

end dhoni_savings_l590_590617


namespace solve_inequality_l590_590147

theorem solve_inequality (x : ℝ) :
  abs (x + 3) + abs (2 * x - 1) < 7 ↔ -3 ≤ x ∧ x < 5 / 3 :=
by
  sorry

end solve_inequality_l590_590147


namespace correct_simplification_a_l590_590227

def simplifications (n : Int) : Int :=
  if n = 1 then -(+6)
  else if n = 2 then -(-17)
  else if n = 3 then +(-9)
  else if n = 4 then +(+5)
  else 0

theorem correct_simplification_a :
  (simplifications 1 = -6) ∧
  (simplifications 2 ≠ -17) ∧
  (simplifications 3 ≠ 9) ∧
  (simplifications 4 ≠ -5) :=
by
  sorry

end correct_simplification_a_l590_590227


namespace card_arrangement_count_l590_590284

theorem card_arrangement_count :
  let n := 11
  let red := 5
  let blue := 3
  let green := 2
  let yellow := 1
  -- The formula for permutation of a multiset
  ∃ f : ℕ, f = Nat.factorial n / (Nat.factorial red * Nat.factorial blue * Nat.factorial green * Nat.factorial yellow) ∧ f = 27720 :=
begin
  sorry
end

end card_arrangement_count_l590_590284


namespace complement_of_A_in_U_l590_590815

noncomputable def U := {x : ℝ | Real.exp x > 1}

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - 1)

def A := { x : ℝ | x > 1 }

def compl (U A : Set ℝ) := { x : ℝ | x ∈ U ∧ x ∉ A }

theorem complement_of_A_in_U : compl U A = { x : ℝ | 0 < x ∧ x ≤ 1 } := sorry

end complement_of_A_in_U_l590_590815


namespace cubic_sqrt_inequality_l590_590832

theorem cubic_sqrt_inequality (x : ℝ) (h : real.cbrt (x + 16) - real.cbrt (x - 16) = 4) :
  235 < x^2 ∧ x^2 < 245 :=
sorry

end cubic_sqrt_inequality_l590_590832


namespace probability_unique_tens_digits_l590_590547

theorem probability_unique_tens_digits :
  let num_ways := 10^6 in
  let total_combinations := Nat.choose 70 6 in
  (num_ways : ℚ) / total_combinations = 625 / 74440775 :=
by 
  sorry

end probability_unique_tens_digits_l590_590547


namespace largest_possible_value_of_m_l590_590472

theorem largest_possible_value_of_m :
  ∃ (X Y Z : ℕ), 0 ≤ X ∧ X ≤ 7 ∧ 0 ≤ Y ∧ Y ≤ 7 ∧ 0 ≤ Z ∧ Z ≤ 7 ∧
                 (64 * X + 8 * Y + Z = 475) ∧ 
                 (144 * Z + 12 * Y + X = 475) := 
sorry

end largest_possible_value_of_m_l590_590472


namespace projectile_height_at_45_l590_590650

-- Define the height function
def height (t : ℝ) : ℝ := 60 + 8 * t - 5 * t^2

-- The time when the height reaches 45 meters is 2.708 seconds
theorem projectile_height_at_45 :
  ∃ t : ℝ, height t = 45 ∧ t = 2.708 :=
by {
  sorry
}

end projectile_height_at_45_l590_590650


namespace zero_not_in_range_of_g_l590_590109

def g (x : ℝ) : ℤ :=
  if h : x > 3 then
    ⌈1 / (x - 3)⌉
  else if h' : x < 3 then
    ⌊1 / (x - 3)⌋
  else
    sorry -- undefined for x = 3

theorem zero_not_in_range_of_g : ∀ y : ℝ, g y ≠ 0 := by
  sorry

end zero_not_in_range_of_g_l590_590109


namespace VehicleB_travel_time_l590_590589

theorem VehicleB_travel_time 
    (v_A v_B : ℝ)
    (d : ℝ)
    (h1 : d = 3 * (v_A + v_B))
    (h2 : 3 * v_A = d / 2)
    (h3 : ∀ t ≤ 3.5 , d - t * v_B - 0.5 * v_A = 0)
    : d / v_B = 7.2 :=
by
  sorry

end VehicleB_travel_time_l590_590589


namespace replace_last_e_in_message_l590_590311

def shiftPattern : ℕ → ℕ
| 0 => 1
| 1 => 2 * shiftPattern 0
| 2 => shiftPattern 0 + shiftPattern 1
| 3 => 2 * (shiftPattern 0 + shiftPattern 1)
| n => if n % 2 = 0 then shiftPattern (n - 2) + shiftPattern (n - 1) else 2 * shiftPattern (n - 2)

def letterShift (c : Char) (shift : ℕ) : Char :=
  let base := ('a'.toNat : ℕ)
  let c_nat := c.toNat - base
  let shifted := (c_nat + shift) % 26
  Char.ofNat (base + shifted)

theorem replace_last_e_in_message :
  letterShift 'e' (shiftPattern 4) = 'k' :=
by
  sorry

end replace_last_e_in_message_l590_590311


namespace set_points_not_in_first_third_quadrants_l590_590564

theorem set_points_not_in_first_third_quadrants : 
  M = {p : ℝ × ℝ | (p.1 < 0 ∧ p.2 > 0) ∨ (p.1 > 0 ∧ p.2 < 0) ∨ p.1 = 0 ∨ p.2 = 0} :=
begin
  sorry
end

end set_points_not_in_first_third_quadrants_l590_590564


namespace petya_time_comparison_l590_590685

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l590_590685


namespace probability_sum_16_l590_590319

open ProbabilityTheory

noncomputable def coin_flip_probs : Finset ℚ := {5 , 15}
noncomputable def die_probs : Finset ℚ := {1, 2, 3, 4, 5, 6}

def fair_coin (x : ℚ) : ℚ := if x = 5 ∨ x = 15 then (1 : ℚ) / 2 else 0
def fair_die (x : ℚ) : ℚ := if x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 then (1 : ℚ) / 6 else 0

theorem probability_sum_16 : ∑ x in coin_flip_probs, ∑ y in die_probs, (if x + y = 16 then fair_coin x * fair_die y else 0) = 1 / 12 := 
    sorry

end probability_sum_16_l590_590319


namespace statement_004_l590_590875

variable {a b : Vector} -- Declare variables as vectors

theorem statement_004 (a b : Vector) : 
  ‖a + b‖^2 = (a + b) • (a + b) := 
sorry -- Proof is omitted

end statement_004_l590_590875


namespace athletes_meet_opposite_direction_l590_590959

variables (a b : ℝ)

def time_to_meet_opposite_direction (z x y : ℝ) : ℝ :=
  (a * b) / real.sqrt (a ^ 2 + 4 * a * b)

theorem athletes_meet_opposite_direction
  (h1 : (z / y) - (z / x) = a)
  (h2 : z / (x - y) = b) : time_to_meet_opposite_direction z x y = (a * b) / real.sqrt (a ^ 2 + 4 * a * b) := 
  sorry

end athletes_meet_opposite_direction_l590_590959


namespace students_preferred_cake_count_l590_590024

-- Define the conditions
def forty_percent_students_chose_ice_cream (T : ℕ) : Prop :=
  0.4 * T = 80

def thirty_percent_students_preferred_cake (T : ℕ) (C : ℕ) : Prop :=
  0.3 * T = C

-- Define the proof problem
theorem students_preferred_cake_count (T : ℕ) (C : ℕ) (H1 : forty_percent_students_chose_ice_cream T) (H2 : thirty_percent_students_preferred_cake T C) :  C = 60 :=
  by
  sorry

end students_preferred_cake_count_l590_590024


namespace division_proof_l590_590992

-- Define the given condition
def given_condition : Prop :=
  2084.576 / 135.248 = 15.41

-- Define the problem statement we want to prove
def problem_statement : Prop :=
  23.8472 / 13.5786 = 1.756

-- Main theorem stating that under the given condition, the problem statement holds
theorem division_proof (h : given_condition) : problem_statement :=
by sorry

end division_proof_l590_590992


namespace probability_sum_sixteen_l590_590317

-- Define the probabilities involved
def probability_of_coin_fifteen := 1 / 2
def probability_of_die_one := 1 / 6

-- Define the combined probability
def combined_probability : ℚ := probability_of_coin_fifteen * probability_of_die_one

theorem probability_sum_sixteen : combined_probability = 1 / 12 := by
  sorry

end probability_sum_sixteen_l590_590317


namespace euclidean_remainder_polynomial_l590_590476

theorem euclidean_remainder_polynomial (P : ℝ[X]) (h1 : P.eval 1 = 2) (h2 : P.eval 2 = 1) :
  ∃ (a b : ℝ), (a = -1) ∧ (b = 3) ∧ ∀ x, (P(x) % ((x - 1) * (x - 2)) = -x + 3) := 
sorry

end euclidean_remainder_polynomial_l590_590476


namespace find_s_l590_590882

theorem find_s (p q : Polynomial ℝ) (s : ℝ)
  (hp : ∀ x : ℝ, p = Polynomial.monicCubic Roots [s + 2, s + 8, c])
  (hq : ∀ x : ℝ, q = Polynomial.monicCubic Roots [s + 4, s + 10, d])
  (h : ∀ x : ℝ, p.eval x - q.eval x = 2 * s) : 
  s = 24 := 
sorry

end find_s_l590_590882


namespace proj_u0_to_u2_l590_590470

noncomputable def proj_matrix (v : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ :=
  let ⟨a, b⟩ := v in 
  (λ u, (let ⟨x, y⟩ := u in (a*x + b*y) / (a*a + b*b) • v))

def matrix_mult (m1 m2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (⟨a1, b1⟩, ⟨c1, d1⟩) := m1 in
  let (⟨a2, b2⟩, ⟨c2, d2⟩) := m2 in
  ((a1*a2 + b1*c2, a1*b2 + b1*d2), (c1*a2 + d1*c2, c1*b2 + d1*d2))

theorem proj_u0_to_u2 : 
  matrix_mult (proj_matrix (4, 2)) (proj_matrix (2, 3)) = ((28 / 65, 14 / 65), (42 / 65, 21 / 65)) :=
sorry

end proj_u0_to_u2_l590_590470


namespace necessary_and_sufficient_condition_l590_590369

variables (a b : ℝ^3) 
variables (α : set (ℝ^3))

def is_nonzero (v : ℝ^3) : Prop := v ≠ 0

def is_normal_vector (a : ℝ^3) (α : set (ℝ^3)) : Prop :=
  ∀ (p q : ℝ^3), p ∈ α → q ∈ α → (p - q) ⊥ a

def parallel_or_within_plane (b : ℝ^3) (α : set (ℝ^3)) : Prop :=
  ∀ (p : ℝ^3), (∃ (d : ℝ) (q ∈ α), p = q + d • b) 

theorem necessary_and_sufficient_condition (a b : ℝ^3) (α : set (ℝ^3)) 
  (ha : is_nonzero a) (hb : is_nonzero b) (hα : is_normal_vector a α) : 
  (a ⬝ b = 0 ↔ parallel_or_within_plane b α) := 
sorry

end necessary_and_sufficient_condition_l590_590369


namespace checkerboard_ratio_l590_590161

theorem checkerboard_ratio:
  ∃ m n : ℕ, -- m and n are natural numbers
  let r := Nat.choose 11 2 * Nat.choose 11 2, -- total rectangles
  let s := (List.range 10).sum (λ n, (n+1)*(n+1)), -- total squares
  let gcd := Nat.gcd s r, -- greatest common divisor
  let m := s / gcd,
  let n := r / gcd,
  Nat.coprime m n ∧ m + n = 62 := by
    sorry

end checkerboard_ratio_l590_590161


namespace sara_marbles_l590_590145

theorem sara_marbles : 10 - 7 = 3 :=
by
  sorry

end sara_marbles_l590_590145


namespace Petya_time_comparison_l590_590677

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l590_590677


namespace range_of_a_l590_590396

-- Defining the function f
noncomputable def f (x a : ℝ) : ℝ :=
  (Real.exp x) * (2 * x - 1) - a * x + a

-- Main statement
theorem range_of_a (a : ℝ)
  (h1 : a < 1)
  (h2 : ∃ x0 x1 : ℤ, x0 ≠ x1 ∧ f x0 a ≤ 0 ∧ f x1 a ≤ 0) :
  (5 / (3 * Real.exp 2)) < a ∧ a ≤ (3 / (2 * Real.exp 1)) :=
sorry

end range_of_a_l590_590396


namespace range_of_combined_set_is_91_l590_590235

-- Definitions based on conditions in a)
def isTwoDigitPrime (n : ℕ) : Prop := n ≥ 10 ∧ n < 100 ∧ Nat.Prime n
def isPositiveMultipleOf6LessThan100 (n : ℕ) : Prop := (n % 6 = 0) ∧ n > 0 ∧ n < 100

-- Define sets based on conditions in a)
def X : Set ℕ := {n | isTwoDigitPrime n}
def Y : Set ℕ := {n | isPositiveMultipleOf6LessThan100 n}

-- The union of sets X and Y
def Z : Set ℕ := X ∪ Y

-- Assertion to be proven based on the question and correct answer
theorem range_of_combined_set_is_91 : (Z.sup id) - (Z.inf id) = 91 := by
  sorry

end range_of_combined_set_is_91_l590_590235


namespace sheets_paper_150_l590_590668

def num_sheets_of_paper (S : ℕ) (E : ℕ) : Prop :=
  (S - E = 50) ∧ (3 * E - S = 150)

theorem sheets_paper_150 (S E : ℕ) : num_sheets_of_paper S E → S = 150 :=
by
  sorry

end sheets_paper_150_l590_590668


namespace quad_roots_expression_l590_590474

theorem quad_roots_expression (x1 x2 : ℝ) (h1 : x1 * x1 + 2019 * x1 + 1 = 0) (h2 : x2 * x2 + 2019 * x2 + 1 = 0) :
  x1 * x2 - x1 - x2 = 2020 :=
sorry

end quad_roots_expression_l590_590474


namespace price_per_exercise_book_is_correct_l590_590988

-- Define variables and conditions from the problem statement
variables (xM xH booksM booksH pricePerBook : ℝ)
variables (xH_gives_xM : ℝ)

-- Conditions set up from the problem statement
axiom pooled_money : xM = xH
axiom books_ming : booksM = 8
axiom books_hong : booksH = 12
axiom amount_given : xH_gives_xM = 1.1

-- Problem statement to prove
theorem price_per_exercise_book_is_correct :
  (8 + 12) * pricePerBook / 2 = 1.1 → pricePerBook = 0.55 := by
  sorry

end price_per_exercise_book_is_correct_l590_590988


namespace general_formula_sum_Tn_l590_590052

-- Conditions
def a_seq (n : ℕ) : ℕ := if n = 1 then 0 else n - 1
def S_n (n : ℕ) : ℕ := (finset.range n).sum (λ i, a_seq (i + 1))

-- Given conditions
axiom a2 : a_seq 2 = 1
axiom S_n_cond (n : ℕ) : 2 * S_n n = n * a_seq n

-- (1) General formula for {a_n}
theorem general_formula (n : ℕ) : a_seq n = n - 1 :=
by sorry

-- (2) Sum of the first n terms of the sequence {(\frac{a_n + 1}{2^n})}
def T_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, (a_seq (k + 1) + 1 : ℝ) / (2 : ℝ) ^ (k + 1))

theorem sum_Tn (n : ℕ) : T_n n = 2 - (n + 2) / (2 : ℝ) ^ n :=
by sorry

end general_formula_sum_Tn_l590_590052


namespace alyssa_kittens_l590_590664

theorem alyssa_kittens (original_kittens given_away: ℕ) (h1: original_kittens = 8) (h2: given_away = 4) :
  original_kittens - given_away = 4 :=
by
  sorry

end alyssa_kittens_l590_590664


namespace min_distance_curve_l590_590762

noncomputable def min_distance_to_focus (θ : ℝ) : ℝ :=
  let x := 2 * Real.cos θ in
  let y := 3 * Real.sin θ in
  let a := 3 in
  let b := 2 in
  let c := Real.sqrt (a^2 - b^2) in
  a - c

theorem min_distance_curve : ∀ (θ : ℝ), min_distance_to_focus θ = 3 - Real.sqrt 5 :=
by
  sorry

end min_distance_curve_l590_590762


namespace actual_time_greater_than_planned_time_l590_590671

def planned_time (a V : ℝ) : ℝ := a / V

def actual_time (a V : ℝ) : ℝ := (a / (2.5 * V)) + (a / (1.6 * V))

theorem actual_time_greater_than_planned_time (a V : ℝ) (hV : V > 0) : 
  actual_time a V > planned_time a V :=
by 
  sorry

end actual_time_greater_than_planned_time_l590_590671


namespace sequence_geometric_l590_590467

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  3 * a n - 2

theorem sequence_geometric (a : ℕ → ℝ) (h : ∀ n, S n a = 3 * a n - 2) :
  ∀ n, a n = (3/2)^(n-1) :=
by
  intro n
  sorry

end sequence_geometric_l590_590467


namespace proof_t_minus_s_l590_590274

def students : ℤ := 250
def teachers : ℤ := 6
def enrollments : List ℤ := [100, 50, 50, 25, 15, 10]

def average_students_per_teacher : ℤ := 
  (List.sum enrollments) / teachers

def student_distribution : List Rat := 
  List.map (λ n => n * (n / students)) enrollments

def average_students_per_student : ℤ := 
  List.sum student_distribution

def t_minus_s := average_students_per_teacher - average_students_per_student

theorem proof_t_minus_s : t_minus_s = -22.13 := by
  sorry

end proof_t_minus_s_l590_590274


namespace maximum_lambda_l590_590335

theorem maximum_lambda (a b : ℝ) : (27 / 4) * a^2 * b^2 * (a + b)^2 ≤ (a^2 + a * b + b^2)^3 := 
sorry

end maximum_lambda_l590_590335


namespace regular_polygon_sides_l590_590749

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l590_590749


namespace willy_number_of_crayons_l590_590614

theorem willy_number_of_crayons (lucy_crayons : ℤ) (additional_crayons : ℤ) (willy_crayons : ℤ) 
  (h_lucy : lucy_crayons = 290) 
  (h_additional : additional_crayons = 1110) : 
  willy_crayons = lucy_crayons + additional_crayons :=
by
  rw [h_lucy, h_additional]
  exact rfl

end willy_number_of_crayons_l590_590614


namespace frustum_midsection_area_relation_l590_590930

theorem frustum_midsection_area_relation 
  (S₁ S₂ S₀ : ℝ) 
  (h₁: 0 ≤ S₁ ∧ 0 ≤ S₂ ∧ 0 ≤ S₀)
  (h₂: ∃ a h, (a / (a + 2 * h))^2 = S₂ / S₁ ∧ (a / (a + h))^2 = S₂ / S₀) :
  2 * Real.sqrt S₀ = Real.sqrt S₁ + Real.sqrt S₂ := 
sorry

end frustum_midsection_area_relation_l590_590930


namespace question_l590_590694

noncomputable def f : ℝ → ℝ := sorry  -- Assume the function f is given and invertible

variables (a c : ℝ)

-- Conditions
axiom invertible_f : Function.Bijective f
axiom condition1 : f(a) = c
axiom condition2 : f(c) = 3

-- Theorem to prove
theorem question : a - c = -2 :=
by 
  -- Proof placeholder; no actual proof required
  sorry

end question_l590_590694


namespace regular_polygon_sides_l590_590741

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l590_590741


namespace john_total_feet_climbed_l590_590458

def first_stair_steps : ℕ := 20
def second_stair_steps : ℕ := 2 * first_stair_steps
def third_stair_steps : ℕ := second_stair_steps - 10
def step_height : ℝ := 0.5

theorem john_total_feet_climbed : 
  (first_stair_steps + second_stair_steps + third_stair_steps) * step_height = 45 :=
by
  sorry

end john_total_feet_climbed_l590_590458


namespace incenter_angle_bisector_l590_590057

theorem incenter_angle_bisector
  (P Q R I L M N : Type)
  (PL_bisector : is_angle_bisector P L R)
  (QM_bisector : is_angle_bisector Q M P)
  (RN_bisector : is_angle_bisector R N Q)
  (inter_incenter : I = incenter P Q R)
  (angle_PRQ : angle P R Q = 50) :
  angle P I Q = 65 :=
by
  sorry

end incenter_angle_bisector_l590_590057


namespace variance_of_sample_l590_590385

theorem variance_of_sample :
  ∀ x : ℝ,
    (2 + 3 + x + 6 + 8) / 5 = 5 →
    let sample := [2, 3, 6, 6, 8]
    in (1 / 5) * (∑ i in sample, (i - 5) ^ 2) = 24 / 5 :=
begin
  sorry
end

end variance_of_sample_l590_590385


namespace twelve_point_sphere_l590_590508

/-!
# Twelve-point Sphere Theorem for a Regular Tetrahedron
Declare and show that the bases of the altitudes, the midpoints of the altitudes,
and the points of intersection of the altitudes of the faces of a regular tetrahedron 
lie on a single sphere.
-/

/-- A regular tetrahedron is a tetrahedron where all faces are equilateral triangles. -/
structure RegularTetrahedron (V : Type) [InnerProductSpace ℝ V] :=
(a b c d : V)
(face_eq : (∀ x y z ∈ {a, b, c, d}, 
  ∃ r, (x - y).norm = r ∧ (y - z).norm = r ∧ (z - x).norm = r))

/-- The twelve-point sphere of a regular tetrahedron lies on a single sphere.
The points considered are the bases of the altitudes, the midpoints of these altitudes,
and the intersection points of the altitudes of the faces.
-/
theorem twelve_point_sphere 
  {V : Type} [InnerProductSpace ℝ V] (T : RegularTetrahedron V) : 
  ∃ (O : V) (r : ℝ), 
    (∀ p ∈ {
      -- The bases of the altitudes
      base_altitude T.a T.b T.c T.d,
      base_altitude T.b T.c T.d T.a,
      base_altitude T.c T.d T.a T.b,
      base_altitude T.d T.a T.b T.c,
      
      -- The midpoints of the altitudes
      midpoint_altitude T.a T.b T.c T.d,
      midpoint_altitude T.b T.c T.d T.a,
      midpoint_altitude T.c T.d T.a T.b,
      midpoint_altitude T.d T.a T.b T.c,

      -- The intersection points of the altitudes of the faces
      face_intersection T.a T.b T.c,
      face_intersection T.b T.c T.d,
      face_intersection T.c T.d T.a,
      face_intersection T.d T.a T.b
    },
      (p - O).norm = r
  ) :=
sorry

end twelve_point_sphere_l590_590508


namespace petya_time_comparison_l590_590681

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l590_590681


namespace minimum_moves_required_l590_590997

-- Define the board positions as natural numbers
def Position := ℕ

-- Define initial and target positions
def initial_white_positions := {1, 3}
def initial_black_positions := {5, 7}
def target_white_positions := {5, 7}
def target_black_positions := {1, 3}

-- Define a move function that follows knight's rule (Valid move logic can be expanded as required)
def is_valid_move (from to : Position) : Prop := sorry -- Definition of a valid knight move

-- Define the configuration of the board as a mapping from positions to knights
structure BoardConfiguration :=
(white_positions : Set Position)
(black_positions : Set Position)

-- Initial and target configurations
def initial_configuration : BoardConfiguration := {
  white_positions := initial_white_positions,
  black_positions := initial_black_positions
}

def target_configuration : BoardConfiguration := {
  white_positions := target_white_positions,
  black_positions := target_black_positions
}

-- Define a function to count the number of moves required to reach the target configuration
def min_moves (initial target : BoardConfiguration) : ℕ := sorry -- Function that computes minimum moves

-- The main theorem to be proven
theorem minimum_moves_required : min_moves initial_configuration target_configuration ≥ 16 :=
sorry

end minimum_moves_required_l590_590997


namespace polar_equation_and_length_segment_l590_590035

-- Definition of the Cartesian equation of circle C
def circle_cartesian (x y : ℝ) : Prop := (x - real.sqrt 3) ^ 2 + (y + 1) ^ 2 = 9

-- Definition of the polar equation of circle C
def circle_polar (ρ θ : ℝ) : Prop := ρ ^ 2 - 2 * real.sqrt 3 * ρ * real.cos θ + 2 * ρ * real.sin θ - 5 = 0

-- The main theorem to prove both parts (1) and (2)
theorem polar_equation_and_length_segment :
  (∀ (ρ θ : ℝ), circle_cartesian (ρ * real.cos θ) (ρ * real.sin θ) ↔ circle_polar ρ θ) ∧
  (∀ (ρ1 ρ2 : ℝ), ρ1^2 - 2*ρ1 - 5 = 0 → ρ2^2 - 2*ρ2 - 5 = 0 → ρ1 ≠ ρ2 → (ρ1 + ρ2 = 2) → (ρ1 * ρ2 = -5) → |ρ1 - ρ2| = 2 * real.sqrt 6) :=
by sorry

end polar_equation_and_length_segment_l590_590035


namespace arithmetic_sequence_a12_l590_590787

/-- A function that represents the nth term of an arithmetic sequence -/
variable {a : ℕ → ℝ}

theorem arithmetic_sequence_a12 (h1 : a 7 + a 9 = 16) (h2 : a 4 = 1) : a 12 = 15 := 
by 
  -- We state that for an arithmetic sequence, 
  -- the sum of two terms whose indices sum to the same value is equal.
  have h3 : a 7 + a 9 = a 4 + a 12 := sorry,
  -- Now substitute the given values
  rw [h1, h2] at h3,
  -- Simplify to get a 12
  exact h3.symm.trans (eq.subst h1 h2)

end arithmetic_sequence_a12_l590_590787


namespace find_f2_l590_590877

noncomputable def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 0) : f 2 a b = -16 :=
by {
  sorry
}

end find_f2_l590_590877


namespace digit_of_4736d_is_4_l590_590551

noncomputable def digit_of_4736d : ℕ :=
  let d := 4 in d

theorem digit_of_4736d_is_4 (d : ℕ) (hd : 0 ≤ d ∧ d ≤ 9) (h1 : d % 2 = 0) (h2 : (4 + 7 + 3 + 6 + d) % 3 = 0) :
  d = digit_of_4736d :=
by
  sorry

end digit_of_4736d_is_4_l590_590551


namespace abs_neg_one_ninth_l590_590552

theorem abs_neg_one_ninth : abs (- (1 / 9)) = 1 / 9 := by
  sorry

end abs_neg_one_ninth_l590_590552


namespace distinct_roots_of_cubic_l590_590171

noncomputable def roots_of_cubic : (ℚ × ℚ × ℚ) :=
  let a := 1
  let b := -2
  let c := 0
  (a, b, c)

theorem distinct_roots_of_cubic :
  ∃ a b c : ℚ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ roots_of_cubic = (a, b, c) := 
by
  use 1, -2, 0
  split; linarith
  split; linarith
  split; linarith
  dsimp only [roots_of_cubic]
  refl

end distinct_roots_of_cubic_l590_590171


namespace find_root_of_equation_l590_590238

theorem find_root_of_equation :
  ∃ x : ℝ, abs (x + 0.068) < 0.001 ∧ (sqrt 5 - sqrt 2) * (1 + x) = (sqrt 6 - sqrt 3) * (1 - x) :=
sorry

end find_root_of_equation_l590_590238


namespace imaginary_part_of_z_l590_590780

def is_imaginary_unit (i : ℂ) : Prop := i^2 = -1

noncomputable def z : ℂ := 1 / (complex.I)

theorem imaginary_part_of_z (i : ℂ) (h : is_imaginary_unit i) : complex.im z = -1 :=
by sorry

end imaginary_part_of_z_l590_590780


namespace point_in_all_figures_exists_l590_590030

theorem point_in_all_figures_exists : 
  ∀ (A : Fin 100 → Set (ℝ × ℝ)), (∀ i, MeasurableSet (A i)) → 
  (∑ i, volume (A i)) > 99 →
  ∃ x ∈ Icc (0 : ℝ) 1, ∀ i, x ∈ A i := 
by 
  sorry

end point_in_all_figures_exists_l590_590030


namespace out_of_pocket_l590_590067

def cost_first_ring : ℕ := 10000
def cost_second_ring : ℕ := 2 * cost_first_ring
def sale_first_ring : ℕ := cost_first_ring / 2

theorem out_of_pocket : cost_first_ring + cost_second_ring - sale_first_ring = 25000 := by
  unfold cost_first_ring cost_second_ring sale_first_ring
  simp
  sorry

end out_of_pocket_l590_590067


namespace y_intercept_of_line_l590_590950

theorem y_intercept_of_line (m : ℝ) (x₀ y₀ : ℝ) (h₁ : m = -3) (h₂ : x₀ = 7) (h₃ : y₀ = 0) :
  ∃ (b : ℝ), (0, b) = (0, 21) :=
by
  -- Our goal is to prove the y-intercept is (0, 21)
  sorry

end y_intercept_of_line_l590_590950


namespace count_sequences_equals_F_l590_590072

-- Define the conditions
def is_valid_sequence (n : ℕ) (seq : fin (2 * n) → ℤ) : Prop :=
  (∀ i, i.val < 2 * n → seq i ∈ {-1, 0, 1}) ∧
  (∀ (i j : fin (2 * n)), i.val % 2 = 0 → j.val % 2 = 1 →
    i.val < j.val →
    let s := (fin_range (j.val + 1)).map (λ k => seq ⟨k, fin.is_lt k⟩) in
    -2 ≤ s.sum ∧ s.sum ≤ 2)

-- Define the main problem of counting such sequences
noncomputable def count_valid_sequences (n : ℕ) : ℕ :=
  fintype.card {seq // is_valid_sequence n seq}

theorem count_sequences_equals_F (n : ℕ) : count_valid_sequences n = F(n) := 
  sorry

end count_sequences_equals_F_l590_590072


namespace probability_six_integers_unique_tens_digit_l590_590541

theorem probability_six_integers_unique_tens_digit :
  (∃ (x1 x2 x3 x4 x5 x6 : ℕ),
    10 ≤ x1 ∧ x1 ≤ 79 ∧
    10 ≤ x2 ∧ x2 ≤ 79 ∧
    10 ≤ x3 ∧ x3 ≤ 79 ∧
    10 ≤ x4 ∧ x4 ≤ 79 ∧
    10 ≤ x5 ∧ x5 ≤ 79 ∧
    10 ≤ x6 ∧ x6 ≤ 79 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x1 ≠ x6 ∧
    x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x2 ≠ x6 ∧
    x3 ≠ x4 ∧ x3 ≠ x5 ∧ x3 ≠ x6 ∧
    x4 ≠ x5 ∧ x4 ≠ x6 ∧
    x5 ≠ x6 ∧
    tens_digit x1 ≠ tens_digit x2 ∧
    tens_digit x1 ≠ tens_digit x3 ∧
    tens_digit x1 ≠ tens_digit x4 ∧
    tens_digit x1 ≠ tens_digit x5 ∧
    tens_digit x1 ≠ tens_digit x6 ∧
    tens_digit x2 ≠ tens_digit x3 ∧
    tens_digit x2 ≠ tens_digit x4 ∧
    tens_digit x2 ≠ tens_digit x5 ∧
    tens_digit x2 ≠ tens_digit x6 ∧
    tens_digit x3 ≠ tens_digit x4 ∧
    tens_digit x3 ≠ tens_digit x5 ∧
    tens_digit x3 ≠ tens_digit x6 ∧
    tens_digit x4 ≠ tens_digit x5 ∧
    tens_digit x4 ≠ tens_digit x6 ∧
    tens_digit x5 ≠ tens_digit x6)
    →
  (probability := \(\frac{4375}{744407}\)).sorry

end probability_six_integers_unique_tens_digit_l590_590541


namespace hyperbola_equation_l590_590814

theorem hyperbola_equation (a b c x y : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : c = 5) (h4 : c ^ 2 = a ^ 2 + b ^ 2) (h5 : b = 2 * a) 
  (eq_asymptote : ∀ x, y = 2 * x): 
  (a = √5 ∧ b = 2 * √5) ∧ (∃ (a b : ℝ), 
  (a > 0) ∧ (b > 0) ∧ (c ^ 2 = a ^ 2 + b ^ 2) ∧ (b = 2 * a) 
  ∧ (c = 5)) -> (∀ x y, (x^2 / (a^2) - y^2 / (b^2) = 1) ↔ (x^2 / 5 - y ^2 / 20 = 1)) :=
by
  sorry

end hyperbola_equation_l590_590814


namespace regular_polygon_sides_l590_590735

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ α, α = 160 → ∑ x in range n, (180 - (180 - 160)) = 360) : n = 18 :=
by
  sorry

end regular_polygon_sides_l590_590735


namespace regular_polygon_sides_l590_590733

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ α, α = 160 → ∑ x in range n, (180 - (180 - 160)) = 360) : n = 18 :=
by
  sorry

end regular_polygon_sides_l590_590733


namespace correct_statements_l590_590392

variable {V : Type*} [inner_product_space ℝ V]

-- Define the vectors a and b
variable (a b : V)

-- Assume the given conditions
axiom dot_product_zero_left (a : V) : a ⋅ 0 = 0
axiom dot_product_zero_right (a : V) : 0 ⋅ a = 0
axiom collinear_same_direction (a b : V) (h : ∃ (k : ℝ), k ≠ 0 ∧ b = k • a) : a ⋅ b = ∥a∥ * ∥b∥
axiom non_zero_implies_non_zero_dot_product (a b : V) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : a ⋅ b ≠ 0
axiom dot_product_zero_then_one_is_zero (a b : V) (h : a ⋅ b = 0) : a = 0 ∨ b = 0
axiom unit_vectors_have_equal_square (a b : V) (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) : a ⋅ a = b ⋅ b

-- The goal is to prove that statements 3 and 6 are correct
theorem correct_statements (h₃ : ∃ (k : ℝ), k ≠ 0 ∧ b = k • a) (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) :
  a ⋅ b = ∥a∥ * ∥b∥ ∧ a ⋅ a = b ⋅ b :=
by
  exact ⟨collinear_same_direction a b h₃, unit_vectors_have_equal_square a b ha hb⟩

end correct_statements_l590_590392


namespace convert_67_to_binary_l590_590301

theorem convert_67_to_binary :
  nat.binary_repr 67 = "1000011" :=
sorry

end convert_67_to_binary_l590_590301


namespace operation_lemma_l590_590014

-- Defining the operation ø
def operation_ø (x w : ℕ) : ℚ := (2^x) / (2^w)

theorem operation_lemma : operation_ø (operation_ø 4 2) 3 = 2 :=
by
  sorry

end operation_lemma_l590_590014


namespace gcd_of_72_and_90_l590_590211

theorem gcd_of_72_and_90 :
  Int.gcd 72 90 = 18 := 
sorry

end gcd_of_72_and_90_l590_590211


namespace sum_digit_products_1_to_2018_l590_590228

noncomputable def digit_product_sum : ℕ :=
  ∑ n in (Finset.range 2019) \ {0}, 
    (n.digits 10).prod id

theorem sum_digit_products_1_to_2018 : digit_product_sum = 184320 := by
  sorry

end sum_digit_products_1_to_2018_l590_590228


namespace triangle_angle_A_l590_590017

theorem triangle_angle_A (A B a b : ℝ) (h1 : b = 2 * a) (h2 : B = A + 60) : A = 30 :=
by sorry

end triangle_angle_A_l590_590017


namespace quadratic_has_solution_zero_l590_590784

theorem quadratic_has_solution_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 + 3 * x + k^2 - 4 = 0) →
  ((k - 2) ≠ 0) → k = -2 := 
by 
  sorry

end quadratic_has_solution_zero_l590_590784


namespace units_digit_35_87_plus_93_49_l590_590994

theorem units_digit_35_87_plus_93_49 : (35^87 + 93^49) % 10 = 8 := by
  sorry

end units_digit_35_87_plus_93_49_l590_590994


namespace find_phi_l590_590422

-- Define the problem and the conditions
def problem_statement (φ : ℝ) : Prop :=
  (0 < φ) ∧ (φ < π) ∧
  (∃ k k' : ℤ, -φ + π / 2 + 2 * k * π = 2 * (-φ - π / 2 + 2 * k' * π))

-- The theorem statement
theorem find_phi : ∃ (φ : ℝ), problem_statement φ ∧ φ = π / 2 :=
sorry

end find_phi_l590_590422


namespace min_value_at_2_l590_590425

theorem min_value_at_2 (m : ℝ) :
  (-(m + 7) ≤ 0) → (-(m + 5) ≤ 0) → (m - 5 ≥ 0) → (m + 7 ≥ 0) → (m + 2 ≥ 7) → (12 + m ≥ 7) → m ∈ set.Ici 5 :=
by
  intros h1 h2 h3 h4 h5 h6 
  sorry

end min_value_at_2_l590_590425


namespace probability_different_tens_digit_l590_590529

open Nat

theorem probability_different_tens_digit :
  let total_ways := choose 70 6,
      favorable_ways := 7 * 10^6
  in 
    (favorable_ways : ℝ) / total_ways = (2000 / 3405864 : ℝ) :=
by
  have h1 : total_ways = 70.choose 6 := rfl
  have h2 : favorable_ways = 7 * 10^6 := rfl
  rw [h1, h2]
  sorry

end probability_different_tens_digit_l590_590529


namespace expected_lifetime_of_flashlight_at_least_4_l590_590086

-- Definitions for the lifetimes of the lightbulbs
variable (ξ η : ℝ)

-- Condition: The expected lifetime of the red lightbulb is 4 years.
axiom E_η_eq_4 : 𝔼[η] = 4

-- Definition stating the lifetime of the flashlight
def T := max ξ η

theorem expected_lifetime_of_flashlight_at_least_4 
  (h : 𝔼η = 4) :
  𝔼[max ξ η] ≥ 4 :=
by {
  sorry
}

end expected_lifetime_of_flashlight_at_least_4_l590_590086


namespace exists_fraction_x_only_and_f_of_1_is_0_l590_590968

theorem exists_fraction_x_only_and_f_of_1_is_0 : ∃ f : ℚ → ℚ, (∀ x : ℚ, f x = (x - 1) / x) ∧ f 1 = 0 := 
by
  sorry

end exists_fraction_x_only_and_f_of_1_is_0_l590_590968


namespace interesting_p_p_minus_1_minimal_interesting_p_p_minus_1_l590_590870

def is_interesting (p n : ℕ) : Prop := 
  ∃ (f g : Polynomial ℤ), (Polynomial.X ^ n - 1) = (Polynomial.X ^ p - Polynomial.X + 1) * f + p * g

theorem interesting_p_p_minus_1 (p : ℕ) [Fact p.Prime] : is_interesting p (p^p - 1) := sorry

theorem minimal_interesting_p_p_minus_1 (p : ℕ) [Fact p.Prime] : 
  (p = 2 → ∀ n, is_interesting p n → n = 3) ∧
  (p ≠ 2 → ∀ n, is_interesting p n → n ≥ p^p - 1) := 
sorry

end interesting_p_p_minus_1_minimal_interesting_p_p_minus_1_l590_590870


namespace tetrahedron_opposite_edges_equal_l590_590139

variables {A B C D : Type} 
variables [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C] [linear_ordered_field D]

/-- 
  Given a tetrahedron with vertices A, B, C, D and with scalene triangular faces where the sides 
  of these triangles are denoted as a, b, and c, prove that if all faces of the tetrahedron are equal, 
  then opposite edges of the tetrahedron are pairwise equal.
-/
theorem tetrahedron_opposite_edges_equal 
  (tetra : A = B = C = D) 
  (face1_eq_face2 : A = B) 
  (face1_eq_face3 : B = C) 
  (face1_eq_face4 : C = D) 
  (face_sides_scalene : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  (A = D ∧ B = C) := 
by 
  sorry

end tetrahedron_opposite_edges_equal_l590_590139


namespace sequence_infinite_terms_form_l590_590515

noncomputable def sequence (n : ℕ) : ℕ :=
  Nat.floor ((n.to_real - 2).cbrt + (n.to_real + 3).cbrt)^3

theorem sequence_infinite_terms_form (n : ℕ) : ∃ N, ∃ m ≥ N, sequence m = 8 * m + 3 :=
sorry

end sequence_infinite_terms_form_l590_590515


namespace dihedral_angle_is_60_degrees_l590_590447

universe u

-- Basic setup definitions (can be adjusted according to the exact structure and context of the problem)
variables 
  (A B C D E A1 B1 C1 D1: Type u)
  [linear_ordered_field ℝ]

-- Definition of the cube, vertex coordinates, planes etc.
def ABD1_dihedral_angle (A B D1 A1 B1 C1 : ℝ^3) : ℝ :=
-- conditions associated with the cube structure
let AB := 1,
    AC := real.sqrt 2,
    AD1 := real.sqrt 2,
    BD1 := real.sqrt 3,
    AE := real.sqrt (2 / 3) in
-- using the cosine rule to calculate the dihedral angle
let cos_angle := (-1 / 2 : ℝ) in
real.arccos cos_angle

-- Main theorem statement
theorem dihedral_angle_is_60_degrees :
  ABD1_dihedral_angle A B D1 A1 B1 C1 = 60 :=
sorry

end dihedral_angle_is_60_degrees_l590_590447


namespace intersection_of_PS_QR_CI_l590_590118

open EuclidianGeometry

def Points := {A B C I P Q R S : Point | 
  is_incenter I triangle ABC ∧
  on_circle k A ∧ on_circle k B ∧
  intersec_point_on_line k (line AI) = (point A, point P) ∧
  intersec_point_on_line k (line BI) = (point B, point Q) ∧
  intersec_point_on_line k (line AC) = (point A, point R) ∧
  intersec_point_on_line k (line BC) = (point B, point S) ∧
  distinct_points [A, B, P, Q, R, S] ∧
  between R A C ∧
  between S B C }

theorem intersection_of_PS_QR_CI :
  ∀ {A B C I P Q R S : Point} {k : Circle}, 
    Points A B C I P Q R S →
    are_concurrent (line PS) (line QR) (line CI) :=
by {
  sorry, -- Proof omitted
}

end intersection_of_PS_QR_CI_l590_590118


namespace annual_decrease_rate_l590_590945

theorem annual_decrease_rate (P₀ P₂ : ℝ) (r : ℝ) (h₀ : P₀ = 8000) (h₂ : P₂ = 5120) :
  P₂ = P₀ * (1 - r / 100) ^ 2 → r = 20 :=
by
  intros h
  have h₀' : P₀ = 8000 := h₀
  have h₂' : P₂ = 5120 := h₂
  sorry

end annual_decrease_rate_l590_590945


namespace find_number_l590_590618

theorem find_number (n : ℝ) (h : 1 / 2 * n + 7 = 17) : n = 20 :=
by
  sorry

end find_number_l590_590618


namespace area_of_PQRS_l590_590902

theorem area_of_PQRS : 
  ∀ (W X Y Z P Q R S : ℝ)
  (WPY_equilateral : equilateral_triangle W P Y)
  (XQZ_equilateral : equilateral_triangle X Q Z)
  (YRW_equilateral : equilateral_triangle Y R W)
  (ZSX_equilateral : equilateral_triangle Z S X)
  (area_square_WXYZ : area_square W X Y Z = 36),
  area_square P Q R S = 72 + 36 * Real.sqrt 3 :=
by
  intro W X Y Z P Q R S
  intro WPY_equilateral XQZ_equilateral YRW_equilateral ZSX_equilateral area_square_WXYZ
  sorry

end area_of_PQRS_l590_590902


namespace log_equation_solution_l590_590824

theorem log_equation_solution :
  ∀ (log2 log5 : ℝ), log2 = 0.3010 → log5 = 0.6990 → 
  ∃ x : ℝ, 2^(x+2) = 200 ∧ x ≈ 5.64 :=
by {
  intros,
  sorry,
}

end log_equation_solution_l590_590824


namespace expected_lifetime_flashlight_l590_590100

noncomputable theory

variables (ξ η : ℝ) -- ξ and η are continuous random variables representing the lifetimes
variables (T : ℝ) -- T is the lifetime of the flashlight

-- Define the maximum lifetime of the flashlight
def max_lifetime (ξ η : ℝ) : ℝ := max ξ η

-- Given condition: the expectation of η is 4
axiom expectation_eta : E η = 4

-- Theorem statement: expected lifetime of the flashlight is at least 4
theorem expected_lifetime_flashlight (ξ η : ℝ) (h : T = max_lifetime ξ η) : 
  E (max_lifetime ξ η) ≥ 4 :=
by 
  sorry

end expected_lifetime_flashlight_l590_590100


namespace valid_password_l590_590973

-- Definition of the problem conditions
def valid_digit (d : ℕ) : Prop :=
  d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def unique_digits (l : List ℕ) : Prop :=
  l.nodup

def no_zeroes (l : List ℕ) : Prop :=
  0 ∉ l

def valid_connection (a b : ℕ) (c : ℕ): Prop :=
  (c > min a b) ∧ (c < max a b)

-- Main theorem statement
theorem valid_password : valid_digit 1 ∧ valid_digit 2 ∧ valid_digit 7 ∧ valid_digit 6 ∧ valid_digit 9 ∧
    unique_digits [1, 2, 7, 6, 9] ∧ no_zeroes [1, 2, 7, 6, 9] ∧
    ¬ (valid_connection 1 2 3 ∨ valid_connection 2 7 4 ∨ valid_connection 2 7 5 ∨
       valid_connection 7 6 8 ∨ valid_connection 6 9 8 ∨ valid_connection 7 9 8 ∨ valid_connection 6 7 5) :=
  sorry

end valid_password_l590_590973


namespace problem_D_correct_l590_590344

noncomputable def f (x : ℝ) : ℝ := 2^x + x

lemma strictly_increasing_f (x y : ℝ) (h : x < y) : f(x) < f(y) :=
begin
  sorry,
end

theorem problem_D_correct : f (real.sqrt 1.1) > f (real.logb 3 2) :=
begin
  apply strictly_increasing_f,
  exact real.sqrt_lt (by norm_num) (by linarith),
end

end problem_D_correct_l590_590344


namespace opposite_of_2023_is_minus_2023_l590_590570

def opposite (x y : ℤ) : Prop := x + y = 0

theorem opposite_of_2023_is_minus_2023 : opposite 2023 (-2023) :=
by
  sorry

end opposite_of_2023_is_minus_2023_l590_590570


namespace calculate_g_g_2_l590_590825

def g (x : ℤ) : ℤ := 2 * x^2 + 2 * x - 1

theorem calculate_g_g_2 : g (g 2) = 263 :=
by
  sorry

end calculate_g_g_2_l590_590825


namespace line_passing_through_fixed_point_l590_590403

variable {p a b : ℝ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : b^2 ≠ 2 * p * a)

theorem line_passing_through_fixed_point (h_parabola : ∀ (x y : ℝ), y^2 = 2 * p * x) 
    (A M M1 M2 : ℝ × ℝ) (hA : A = (a, b)) (hB : B = (-a, 0)) 
    (hM : ∃ (y0 : ℝ), M = (y0^2 / (2 * p), y0))
    (hM1 : ∃ (y1 : ℝ), M1 = (y1^2 / (2 * p), y1) ∧ collinear A M M1)
    (hM2 : ∃ (y2 : ℝ), M2 = (y2^2 / (2 * p), y2) ∧ collinear B M M2) :
    ∃ (x y : ℝ), (x, y) = (a, 2 * p * a / b) ∧ passes_through_line M1 M2 (x, y) := 
  sorry

end line_passing_through_fixed_point_l590_590403


namespace order_of_f_l590_590108

-- Define the function f
variables {f : ℝ → ℝ}

-- Definition of even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Definition of monotonic increasing function on [0, +∞)
def monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, (0 ≤ x ∧ 0 ≤ y ∧ x ≤ y) → f x ≤ f y

-- The main problem statement
theorem order_of_f (h_even : even_function f) (h_mono : monotonically_increasing_on_nonneg f) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
  sorry

end order_of_f_l590_590108


namespace value_of_f_neg_a_l590_590937

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end value_of_f_neg_a_l590_590937


namespace problem_statement_l590_590419

theorem problem_statement : 
  ∃ (a b : ℚ), (2 - real.sqrt 3)^2 = a + b * real.sqrt 3 ∧ a + b = 3 :=
by 
  have h : (2 - real.sqrt 3)^2 = 7 - 4 * real.sqrt 3 := 
    calc  (2 - real.sqrt 3)^2
          = 2^2 - 2 * 2 * real.sqrt 3 + (real.sqrt 3)^2 : by ring
      ... = 4 - 4 * real.sqrt 3 + 3 : by { rw real.sqrt_sq, ring, exact real.zero_le_sqrt_of_zero_le (le_of_lt (by norm_num))}
      ... = 7 - 4 * real.sqrt 3 : by ring,
  use [7, -4],
  exact ⟨h, by norm_num⟩

end problem_statement_l590_590419


namespace cos_pi_over_8_cos_5pi_over_8_l590_590699

theorem cos_pi_over_8_cos_5pi_over_8 :
  (Real.cos (Real.pi / 8)) * (Real.cos (5 * Real.pi / 8)) = - (Real.sqrt 2 / 4) :=
by
  sorry

end cos_pi_over_8_cos_5pi_over_8_l590_590699


namespace probability_sum_greater_than_five_l590_590961

-- Definitions for the conditions
def die_faces := {1, 2, 3, 4, 5, 6}
def possible_outcomes := (die_faces × die_faces).to_finset
def favorable_outcomes := possible_outcomes.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd > 5)
def probability_of_sum_greater_than_five := (favorable_outcomes.card : ℚ) / possible_outcomes.card

-- Problem statement
theorem probability_sum_greater_than_five :
  probability_of_sum_greater_than_five = 13 / 18 :=
sorry

end probability_sum_greater_than_five_l590_590961


namespace sqrt_of_4_l590_590175

theorem sqrt_of_4 (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_of_4_l590_590175


namespace find_m_l590_590879

def g (x : ℤ) (A : ℤ) (B : ℤ) (C : ℤ) : ℤ := A * x^2 + B * x + C

theorem find_m (A B C m : ℤ) 
  (h1 : g 2 A B C = 0)
  (h2 : 100 < g 9 A B C ∧ g 9 A B C < 110)
  (h3 : 150 < g 10 A B C ∧ g 10 A B C < 160)
  (h4 : 10000 * m < g 200 A B C ∧ g 200 A B C < 10000 * (m + 1)) : 
  m = 16 :=
sorry

end find_m_l590_590879
