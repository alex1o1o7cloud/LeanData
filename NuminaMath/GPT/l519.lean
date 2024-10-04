import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Matrix.Trace
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Polynomials
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Abs
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Permutations
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.Int.Basic
import Mathlib.Mathlib
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.Basic
import data.nat.prime

namespace Cauchy_theorem_l519_519448

theorem Cauchy_theorem (P Q : Polyhedron) 
  (hConvexP : Convex P) 
  (hConvexQ : Convex Q) 
  (hEqualFaces : (correspondingly_equal_faces P Q)) : 
  Congruent P Q ∨ MirrorImages P Q :=
sorry

end Cauchy_theorem_l519_519448


namespace cyclist_C_speed_l519_519450

variable (c d : ℕ)

def distance_to_meeting (c d : ℕ) : Prop :=
  d = c + 6 ∧
  90 + 30 = 120 ∧
  ((90 - 30) / c) = (120 / d) ∧
  (60 / c) = (120 / (c + 6))

theorem cyclist_C_speed : distance_to_meeting c d → c = 6 :=
by
  intro h
  -- To be filled in with the proof using the conditions
  sorry

end cyclist_C_speed_l519_519450


namespace newtons_cooling_problem_l519_519011

-- Define the constants and variables
variables (θ0 θ1 θ t : ℝ) (k : ℝ)

-- State the conditions
def condition1 := θ0 = 10
def condition2 := θ1 = 90
def condition3 := θ = 50
def condition4 (t : ℝ) := θ = θ0 + (θ1 - θ0) * real.exp (-k * t)
def condition5 := k = (1 / 10) * real.log 2
def condition6 := θ = 20

-- State the proof goal
theorem newtons_cooling_problem (H1 : condition1) (H2 : condition2)
  (H3 : condition3) (H4 : condition4 10) (H5 : condition5) (H6 : condition6) : t = 30 :=
sorry

end newtons_cooling_problem_l519_519011


namespace perfect_number_l519_519954

-- Definition of a Mersenne Prime
def isMersennePrime (p : ℕ) : Prop := 
  ∃ k : ℕ, p = 2^k - 1 ∧ Nat.Prime p

-- Definition of a perfect number
def isPerfect (n : ℕ) : Prop :=
  Nat.sigma n = 2 * n

theorem perfect_number (k : ℕ) (hk : Nat.Prime (2^k - 1)) :
  isPerfect (2^(k-1) * (2^k - 1)) :=
sorry

end perfect_number_l519_519954


namespace inverse_variation_l519_519406

theorem inverse_variation (w : ℝ) (h1 : ∃ (c : ℝ), ∀ (x : ℝ), x^4 * w^(1/4) = c)
  (h2 : (3 : ℝ)^4 * (16 : ℝ)^(1/4) = (6 : ℝ)^4 * w^(1/4)) : 
  w = 1 / 4096 :=
by
  sorry

end inverse_variation_l519_519406


namespace probability_of_drawing_yellow_ball_l519_519313

theorem probability_of_drawing_yellow_ball (red_balls yellow_balls : ℕ) 
    (h_red_balls : red_balls = 4) 
    (h_yellow_balls : yellow_balls = 7) : 
    (yellow_balls : ℚ) / (red_balls + yellow_balls) = 7 / 11 :=
by
  rw [h_red_balls, h_yellow_balls]
  norm_num
  sorry

end probability_of_drawing_yellow_ball_l519_519313


namespace gary_eggs_collected_per_week_l519_519216

def chicken_population_after_growth (initial: ℕ) (growth_factor: ℕ) : ℕ :=
  initial * growth_factor

def surviving_chickens (total: ℕ) (mortality_rate: ℝ) : ℕ :=
  (total : ℝ) * (1.0 - mortality_rate) |> floor

def average_eggs_per_day (rates: list ℕ) : ℕ :=
  rates.sum / rates.length

def eggs_per_day (population: ℕ) (avg_rate: ℝ) : ℕ :=
  (population : ℝ) * avg_rate |> floor

def eggs_per_week (daily: ℕ) : ℕ :=
  daily * 7

theorem gary_eggs_collected_per_week :
  let initial_chickens := 4
  let growth_rate := 8
  let mortality_rate := 0.2
  let egg_rates := [6, 5, 7, 4]
  let final_population := surviving_chickens (chicken_population_after_growth initial_chickens growth_rate) mortality_rate
  let avg_egg_rate := average_eggs_per_day egg_rates
  let total_daily_eggs := eggs_per_day final_population avg_egg_rate
  eggs_per_week total_daily_eggs = 959 :=
by
  sorry

end gary_eggs_collected_per_week_l519_519216


namespace tiles_needed_l519_519043

def room_area : ℝ := 2 * 4 * 2 * 6
def tile_area : ℝ := 1.5 * 2

theorem tiles_needed : room_area / tile_area = 32 := 
by
  sorry

end tiles_needed_l519_519043


namespace root_of_quadratic_poly_l519_519426

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

def poly_has_root (a b c : ℝ) (r : ℝ) : Prop := a * r^2 + b * r + c = 0

theorem root_of_quadratic_poly 
  (a b c : ℝ)
  (h1 : discriminant a b c = 0)
  (h2 : discriminant (-a) (b - 30 * a) (17 * a - 7 * b + c) = 0):
  poly_has_root a b c (-11) :=
sorry

end root_of_quadratic_poly_l519_519426


namespace count_library_visits_l519_519485

theorem count_library_visits (n : ℕ) (h : n = 10) : 
  let choices_per_student := 2 in
  (choices_per_student ^ n) = 2^10 :=
by
  sorry

end count_library_visits_l519_519485


namespace at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519719

theorem at_least_one_inhabitant_knows_no_fewer_than_810_people :
  ∀ (n : ℕ),
  (inhabitants : ℕ) (knows_someone : inhabitants > 1) (believes_santa : inhabitants * 9 / 10)
  (ten_percent_acquaintances_believe : ∀ x, (acquaintances : list x) → (|list.contacts_with_santa <- 0.10 * acquaintances|)),
  (h : inhabitants = 1000000),
  (one_inhabitant_knows : ∃ x, ∀ y, y \in acquaintances x → friends y ≥ 810) :=
sorry

end at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519719


namespace finite_solutions_implies_bounded_l519_519817

theorem finite_solutions_implies_bounded
  (a b : Fin 2020 → ℝ)
  (h_finite_two_solutions : ∃! n : ℕ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x : ℝ, (∑ i, |a i| * x - b i) = n) ∧ (∃ x : ℝ, ∑ i, |a i| * x - b i = n))) :
  (∃ n : ℕ, ∃ x : ℝ, (∑ i, |a i| * x - b i) = n) → (finite {n : ℕ | ∃ x : ℝ, (∑ i, |a i| * x - b i) = n}) :=
sorry

end finite_solutions_implies_bounded_l519_519817


namespace matrix_inverse_self_l519_519194

theorem matrix_inverse_self : 
  ∃ c d : ℚ, 
  let M := ![![4, 2], ![c, d]] in
    M ⬝ M = (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ 
    c = -15/2 ∧ 
    d = -4 :=
by
  sorry

end matrix_inverse_self_l519_519194


namespace area_of_region_l519_519915

theorem area_of_region : 
  (∃ x y : ℝ, (x + 5)^2 + (y - 3)^2 = 32) → (π * 32 = 32 * π) :=
by 
  sorry

end area_of_region_l519_519915


namespace E1_value_E2_value_l519_519538

noncomputable def E1 : ℝ :=
  log 3 (sqrt[4](27) / 3) + log (27) / log (9) + 2^(1 + log 3 / log 2)

theorem E1_value : E1 = 7.25 :=
  by
    sorry

noncomputable def E2 : ℝ :=
  0.027^(- 1 / 3) - (- 1 / 6)^(- 2) + 256^(0.75) + (1 / (sqrt 3 - 1))^0

theorem E2_value : E2 = 32.3333 :=
  by
    sorry

end E1_value_E2_value_l519_519538


namespace solution_set_inequality_l519_519226

noncomputable def exists_solution_set (f : ℝ → ℝ) :=
  (∀ x, f x = (λ x, if (lg x)^2 > 1 then 1 else 0 ) x ) ∧ 
  (∀ x, f' x < (1/2)) 

theorem solution_set_inequality (f : ℝ → ℝ) (H1 : f 1 = 1) 
  (H2 : ∀ x, f' x < (1/2)) :
  { x : ℝ | f (real.log x ^ 2) < (1 / 2) * (real.log x ^ 2 + 1 / 2) } = 
  { x : ℝ | 0 < x ∧ x < 1/10 } ∪ { x : ℝ | 10 < x} := 
by 
  sorry

end solution_set_inequality_l519_519226


namespace intersection_eq_single_point_l519_519264

def M : Set (ℝ × ℝ) := { p | p.1 + p.2 = 2 }
def P : Set (ℝ × ℝ) := { p | p.1 - p.2 = 4 }
def single_point := {(3 : ℝ, -1 : ℝ)}

theorem intersection_eq_single_point : M ∩ P = single_point :=
by 
  sorry

end intersection_eq_single_point_l519_519264


namespace exists_inhabitant_with_810_acquaintances_l519_519766

-- Definition of the main problem conditions
def inhabitants := 1000000
def belief_in_santa := 0.9 * inhabitants
def disbelief_in_santa := inhabitants - belief_in_santa

-- Conditions
def knows_at_least_one : ∀ x : ℕ, x < inhabitants → ∃ y : ℕ, y < inhabitants ∧ x ≠ y := sorry
def ten_percent_acquaintances (x : ℕ) (h : x < inhabitants) : Prop := ∀ y : ℕ, y < inhabitants ∧ x ≠ y → (y < belief_in_santa → y < 0.1 * x) := sorry

-- Main theorem
theorem exists_inhabitant_with_810_acquaintances :
  ∃ x : ℕ, x < inhabitants ∧ (knows_at_least_one x) ∧ (ten_percent_acquaintances x) ∧ (810 ≤ x) :=
sorry

end exists_inhabitant_with_810_acquaintances_l519_519766


namespace initial_money_l519_519334

-- Define the cost of the candy bar in dollars
def candy_bar_cost : ℝ := 0.45

-- Define the change received in dollars
def change_received : ℝ := 1.35

-- Define the initial amount of money Josh had
theorem initial_money (c : ℝ) (cr : ℝ) (initial : ℝ)
  (hc : c = candy_bar_cost)
  (hcr : cr = change_received) :
  initial = c + cr :=
begin
  sorry
end

end initial_money_l519_519334


namespace quadratic_completion_l519_519200

theorem quadratic_completion (x : ℝ) : 
  (2 * x^2 + 3 * x - 1) = 2 * (x + 3 / 4)^2 - 17 / 8 := 
by 
  -- Proof isn't required, we just state the theorem.
  sorry

end quadratic_completion_l519_519200


namespace units_digit_sum_sequence_l519_519918

noncomputable def units_digit : Nat → Nat
| n := n % 10

theorem units_digit_sum_sequence :
  let seq := [1!, 2!, 3!, 4!, 5!, 6!, 7!, 8!, 9!, 10!, 11!]
  let seq_plus_indices := List.map2 (λ a b => a + b) seq (List.range' 1 11)
  let seq_sum := seq_plus_indices.sum
  units_digit seq_sum = 9 :=
by
  sorry

end units_digit_sum_sequence_l519_519918


namespace pyramid_dihedral_angle_l519_519207

theorem pyramid_dihedral_angle (a : ℝ) : 
  let R := a / real.sqrt 3,
      OM := R * real.cos (real.pi / 6),
      DM := a * real.sqrt 3 / 2
  in 
  real.arccos (OM / DM) = real.arccos (real.sqrt 3 / 3) :=
begin
  -- skipping the detailed proof steps
  sorry
end

end pyramid_dihedral_angle_l519_519207


namespace average_salary_without_manager_l519_519410

theorem average_salary_without_manager (A : ℝ) (H : 15 * A + 4200 = 16 * (A + 150)) : A = 1800 :=
by {
  sorry
}

end average_salary_without_manager_l519_519410


namespace angle_between_vectors_perp_condition_min_value_l519_519269

open Real BigInt

-- Definitions for vectors a and b
def a (m : ℝ) : ℝ × ℝ := (m, -1)
def b : ℝ × ℝ := (1/2, sqrt (3) / 2)

-- Dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Norm of a 2D vector
def norm (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2)

-- Angular condition: If m = -sqrt(3), theta must be 5π/6
theorem angle_between_vectors (m : ℝ) (h : m = -sqrt 3) :
  let θ : ℝ := acos ((a m).dot_product b / (norm (a m) * norm b))
  θ = 5 * pi / 6 :=
by
  sorry

-- Perpendicular condition: If a is perpendicular to b, m must be sqrt(3)
theorem perp_condition (m : ℝ) (h : dot_product (a m) b = 0) : m = sqrt 3 :=
by
  sorry

-- Minimum value condition for k and t
theorem min_value (k t : ℝ) (hk : k ≠ 0) (ht : t ≠ 0)
  (perp_cond : dot_product ((a (sqrt 3)) + (t^2 - 3) • b) (-k • (a (sqrt 3)) + t • b) = 0) :
  (k + t^2) / t = -7 / 4 :=
by
  sorry

end angle_between_vectors_perp_condition_min_value_l519_519269


namespace area_S_l519_519824

noncomputable def greatest_integer_le (t : ℝ) : ℝ :=
  floor t

noncomputable def fractional_part (t : ℝ) : ℝ :=
  t - greatest_integer_le t

def S (T : ℝ) : set (ℝ × ℝ) :=
  {p | (p.fst - T)^2 + p.snd^2 ≤ T^2}

theorem area_S (t : ℝ) (ht : 0 ≤ t) : 
  let T := fractional_part t in 
  0 ≤ real.pi * T^2 ∧ real.pi * T^2 ≤ real.pi :=
by
  let T := fractional_part t
  have hT : 0 ≤ T ∧ T < 1 := sorry
  split
  · apply mul_nonneg
    · exact real.pi_pos.le
    · exact sq_nonneg T
  · apply mul_le_mul_of_nonneg_left
    · apply sq_le_sq
      exact hT.2
    · exact real.pi_pos.le
  sorry

end area_S_l519_519824


namespace find_a_plus_b_l519_519250

noncomputable def A : ℝ := 3
noncomputable def B : ℝ := -1

noncomputable def l : ℝ := -1 -- Slope of line l (since angle is 3π/4)

noncomputable def l1_slope : ℝ := 1 -- Slope of line l1 which is perpendicular to l

noncomputable def a : ℝ := 0 -- Calculated from k_{AB} = 1

noncomputable def b : ℝ := -2 -- Calculated from line parallel condition

theorem find_a_plus_b : a + b = -2 :=
by
  sorry

end find_a_plus_b_l519_519250


namespace count_g_applications_to_1_l519_519819

noncomputable def g : ℕ → ℕ
| n := if n % 2 = 1 then n^2 - 1 else if n % 3 = 0 then n / 3 else 0

theorem count_g_applications_to_1 :
  (Finset.filter (λ n => ∃ k, (Nat.repeat g k n = 1)) (Finset.range 101)).card = 4 :=
sorry

end count_g_applications_to_1_l519_519819


namespace at_least_one_integer_of_floor_sum_l519_519860

theorem at_least_one_integer_of_floor_sum (a b c : ℝ) (h : ∀ n : ℕ, ⌊ (n:ℝ) * a ⌋ + ⌊ (n:ℝ) * b ⌋ = ⌊ (n:ℝ) * c ⌋ ) : ∃ m : ℝ, m = a ∨ m = b ∧ m ∈ ℤ :=
sorry

end at_least_one_integer_of_floor_sum_l519_519860


namespace trigonometric_identity_l519_519218

theorem trigonometric_identity
  (θ : ℝ)
  (h : Real.tan θ = 1 / 3) :
  Real.sin (3 / 2 * Real.pi + 2 * θ) = -4 / 5 :=
by sorry

end trigonometric_identity_l519_519218


namespace sum_bound_l519_519596

theorem sum_bound (C : ℝ) (a : ℕ → ℝ) (hC : C ≥ 1) (ha_nonneg : ∀ n, 0 ≤ a n)
  (h_bound : ∀ x : ℝ, 1 ≤ x → abs (x * real.log x - ∑ k in finset.range (floor x).succ, floor (x / k) * a k) ≤ C * x) :
  ∀ y : ℝ, 1 ≤ y → ∑ k in finset.range (floor y).succ, a k < 3 * C * y :=
begin
  sorry
end

end sum_bound_l519_519596


namespace average_value_function_range_l519_519225

def f (x m : ℝ) : ℝ := -x^2 + m * x + 1

theorem average_value_function_range (m : ℝ) :
  (∃ x0 : ℝ, x0 ∈ Ioo (-1) 1 ∧ f x0 m = (f 1 m - f (-1) m) / 2) → 0 < m ∧ m < 2 :=
by
  sorry

end average_value_function_range_l519_519225


namespace coles_average_speed_l519_519178

theorem coles_average_speed (t_work : ℝ) (t_round : ℝ) (s_return : ℝ) (t_return : ℝ) (d : ℝ) (t_work_min : ℕ) :
  t_work_min = 72 ∧ t_round = 2 ∧ s_return = 90 ∧ 
  t_work = t_work_min / 60 ∧ t_return = t_round - t_work ∧ d = s_return * t_return →
  d / t_work = 60 := 
by
  intro h
  sorry

end coles_average_speed_l519_519178


namespace shortest_chord_length_l519_519053

theorem shortest_chord_length 
  (k : ℝ)
  (x y : ℝ)
  (circle_eq : (x - 2) ^ 2 + (y - 2) ^ 2 = 4)
  (line_eq : y - 1 = k * (x - 3))
  (center : (2, 2))
  (radius : 2)
  (fixed_point : (3, 1))
  : ∃ l : ℝ, l = 2 * sqrt 2 :=
begin
  sorry
end

end shortest_chord_length_l519_519053


namespace part1_part2_l519_519642

def f (x a : ℝ) : ℝ := (Exp x / x) - Log x + x - a

theorem part1 (x a : ℝ) (hx : f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 (x1 x2 a : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) 
  (hx1_range : 0 < x1) (hx2_range : x2 > 1) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519642


namespace intersecting_chords_l519_519067

noncomputable def length_of_other_chord (x : ℝ) : ℝ :=
  3 * x + 8 * x

theorem intersecting_chords
  (a b : ℝ) (h1 : a = 12) (h2 : b = 18) (r1 r2 : ℝ) (h3 : r1/r2 = 3/8) :
  length_of_other_chord 3 = 33 := by
  sorry

end intersecting_chords_l519_519067


namespace solutions_of_quadratic_eq_l519_519687

axiom a b c : ℝ

theorem solutions_of_quadratic_eq (h1 : a + b + c = 0) (h2 : a - b + c = 0) (h3 : a ≠ 0) :
  ∀ x : ℝ, x = 1 ∨ x = -1 ↔ a * x^2 + b * x + c = 0 :=
by
  -- Skipping proof
  sorry

end solutions_of_quadratic_eq_l519_519687


namespace b_range_l519_519223

/-- Given a circle with equation x^2 + y^2 = 4 and a line y = x + b,
    there are four points on the circle such that the distance from each point to 
    the line is 1. The range of real number b is (-√2, √2). -/
theorem b_range (b : ℝ) (circle_eq : ∀ x y : ℝ, x^2 + y^2 = 4) (line_eq : ∀ x y : ℝ, y = x + b) :
  abs(b) < real.sqrt 2 :=
by
  sorry

end b_range_l519_519223


namespace min_value_of_M_l519_519931

theorem min_value_of_M (P : ℕ → ℝ) (n : ℕ) (M : ℝ):
  (P 1 = 9 / 11) →
  (∀ n ≥ 2, P n = (3 / 4) * (P (n - 1)) + (2 / 3) * (1 - P (n - 1))) →
  (∀ n ≥ 2, P n ≤ M) →
  (M = 97 / 132) := 
sorry

end min_value_of_M_l519_519931


namespace sarah_bottle_caps_l519_519018

theorem sarah_bottle_caps (initial_caps : ℕ) (additional_caps : ℕ) (total_caps : ℕ) : initial_caps = 26 → additional_caps = 3 → total_caps = initial_caps + additional_caps → total_caps = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sarah_bottle_caps_l519_519018


namespace total_video_hours_in_june_l519_519967

-- Definitions for conditions
def upload_rate_first_half : ℕ := 10 -- one-hour videos per day
def upload_rate_second_half : ℕ := 20 -- doubled one-hour videos per day
def days_in_half_month : ℕ := 15
def total_days_in_june : ℕ := 30

-- Number of video hours uploaded in the first half of the month
def video_hours_first_half : ℕ := upload_rate_first_half * days_in_half_month

-- Number of video hours uploaded in the second half of the month
def video_hours_second_half : ℕ := upload_rate_second_half * days_in_half_month

-- Total number of video hours in June
theorem total_video_hours_in_june : video_hours_first_half + video_hours_second_half = 450 :=
by {
  sorry
}

end total_video_hours_in_june_l519_519967


namespace promotional_event_probabilities_l519_519306

def P_A := 1 / 1000
def P_B := 1 / 100
def P_C := 1 / 20
def P_A_B_C := P_A + P_B + P_C
def P_A_B := P_A + P_B
def P_complement_A_B := 1 - P_A_B

theorem promotional_event_probabilities :
  P_A = 1 / 1000 ∧
  P_B = 1 / 100 ∧
  P_C = 1 / 20 ∧
  P_A_B_C = 61 / 1000 ∧
  P_complement_A_B = 989 / 1000 :=
by
  sorry

end promotional_event_probabilities_l519_519306


namespace complement_P_subset_Q_l519_519545

def P : set ℝ := {y | ∃ x : ℝ, y = -x^2 + 1}
def Q : set ℝ := {y | ∃ x : ℝ, y = 2^x}

theorem complement_P_subset_Q : (set.univ \ P) ⊆ Q := by
  sorry

end complement_P_subset_Q_l519_519545


namespace lauren_tuesday_earnings_l519_519347

noncomputable def money_from_commercials (commercials_viewed : ℕ) : ℕ :=
  (1 / 2) * commercials_viewed

noncomputable def money_from_subscriptions (subscribers : ℕ) : ℕ :=
  1 * subscribers

theorem lauren_tuesday_earnings :
  let commercials_viewed := 100 in
  let subscribers := 27 in
  let total_money := money_from_commercials commercials_viewed + money_from_subscriptions subscribers in
  total_money = 77 :=
by 
  sorry

end lauren_tuesday_earnings_l519_519347


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519736

theorem exists_inhabitant_with_at_least_810_acquaintances
  (N : ℕ) (hN : N = 1000000)
  (B : Finset ℕ) (hB : B.card = 900000)
  (A : ℕ → Finset ℕ)
  (hA_nonempty : ∀ i, (A i).nonempty)
  (hA_believers : ∀ i, (A i).card / 10 = (A i ∩ B).card) :
  ∃ j, (A j).card ≥ 810 :=
by
  sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519736


namespace part1_part2_l519_519650

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519650


namespace proof_ineq_l519_519831

noncomputable def a : ℝ := Real.log 2 / Real.log 3  -- Real log base 3 of 2
noncomputable def b : ℝ := Real.ln 2  -- Natural log of 2
noncomputable def c : ℝ := 5^(-1/2 : ℝ)  -- 5 to the power of -1/2

theorem proof_ineq : c < a ∧ a < b := 
  by
  sorry

end proof_ineq_l519_519831


namespace sum_due_is_correct_l519_519411

-- Definitions of the given conditions
def BD : ℝ := 78
def TD : ℝ := 66

-- Definition of the sum due (S)
noncomputable def S : ℝ := (TD^2) / (BD - TD) + TD

-- The theorem to be proved
theorem sum_due_is_correct : S = 429 := by
  sorry

end sum_due_is_correct_l519_519411


namespace find_letter_l519_519369

def consecutive_dates (A B C D E F G : ℕ) : Prop :=
  B = A + 1 ∧ C = A + 2 ∧ D = A + 3 ∧ E = A + 4 ∧ F = A + 5 ∧ G = A + 6

theorem find_letter (A B C D E F G : ℕ) 
  (h_consecutive : consecutive_dates A B C D E F G) 
  (h_condition : ∃ y, (B + y = 2 * A + 6)) :
  y = F :=
by
  sorry

end find_letter_l519_519369


namespace infinite_expressible_and_nonexpressible_l519_519999

def a (n : ℕ) : ℕ := 2^n + 2^(n / 2)

theorem infinite_expressible_and_nonexpressible :
  (∃ᶠ k in at_top, ∃ b : list ℕ, b.length ≥ 2 ∧ b.nodup ∧ b.all (λ i, i < k) ∧ k = b.sum a) ∧
  (∃ᶠ k in at_top, ∀ b : list ℕ, b.length ≥ 2 → b.nodup → b.all (λ i, i < k) → k ≠ b.sum a) :=
sorry

end infinite_expressible_and_nonexpressible_l519_519999


namespace fireworks_display_l519_519153

-- Define numbers and conditions
def display_fireworks_for_number (n : ℕ) : ℕ := 6
def display_fireworks_for_letter (c : Char) : ℕ := 5
def fireworks_per_box : ℕ := 8
def number_boxes : ℕ := 50

-- Calculate fireworks for the year 2023
def fireworks_for_year : ℕ :=
  display_fireworks_for_number 2 * 2 +
  display_fireworks_for_number 0 * 1 +
  display_fireworks_for_number 3 * 1

-- Calculate fireworks for "HAPPY NEW YEAR"
def fireworks_for_phrase : ℕ :=
  12 * display_fireworks_for_letter 'H'

-- Calculate fireworks for 50 boxes
def fireworks_for_boxes : ℕ := number_boxes * fireworks_per_box

-- Total fireworks calculation
def total_fireworks : ℕ := fireworks_for_year + fireworks_for_phrase + fireworks_for_boxes

-- Proof statement
theorem fireworks_display : total_fireworks = 476 := 
  by
  -- This is where the proof would go.
  sorry

end fireworks_display_l519_519153


namespace trigonometric_identity_l519_519984

theorem trigonometric_identity :
  3 * Real.arcsin (Real.sqrt 3 / 2) - Real.arctan (-1) - Real.arccos 0 = (3 * Real.pi) / 4 := 
by
  sorry

end trigonometric_identity_l519_519984


namespace plane_midpoints_divides_equal_volume_l519_519014

variables {A B C D M N : Type} [metric_space (A × B × C × D)]
variables (midpoint : ∀ (x y : Type), Type) (volume : ∀ (x y z : Type), ℝ)

-- Define the midpoint condition given two points on a line segment
def midpoint_of_edge := midpoint (A × B) (C × D)

-- Formalize the statement that the plane passing through the midpoints
-- divides the tetrahedron into two equal volumes
theorem plane_midpoints_divides_equal_volume
  (hM : M = midpoint (A × B))
  (hN : N = midpoint (C × D)) :
  volume (A × B × C × D) = 2 * volume (A × B × M × N) :=
sorry

end plane_midpoints_divides_equal_volume_l519_519014


namespace sinA_minus_cosA_l519_519700

theorem sinA_minus_cosA {A : ℝ} (hA1 : 0 < A) (hA2 : A < pi) (h : sin (2 * A) = -2 / 3) :
  sin A - cos A = sqrt 15 / 3 := by
  sorry

end sinA_minus_cosA_l519_519700


namespace three_digit_number_cubed_sum_l519_519571

theorem three_digit_number_cubed_sum {a b c : ℕ} (h₁ : 1 ≤ a ∧ a ≤ 9)
                                      (h₂ : 0 ≤ b ∧ b ≤ 9)
                                      (h₃ : 0 ≤ c ∧ c ≤ 9) :
  (100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999) →
  (100 * a + 10 * b + c = (a + b + c) ^ 3) →
  (100 * a + 10 * b + c = 512) :=
by
  sorry

end three_digit_number_cubed_sum_l519_519571


namespace part1_part2_l519_519635

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l519_519635


namespace functional_equation_solution_l519_519205

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y z : ℚ, f(x + f(y + f(z))) = y + f(x + z)) →
  (f = (λ x, x) ∨ ∃ a : ℚ, f = (λ x, a - x)) :=
by
  sorry

end functional_equation_solution_l519_519205


namespace part1_part2_l519_519001

variables {x m : ℝ}

def set_A : set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) : set ℝ := { x | (x - m + 1) * (x - 2 * m - 1) < 0 }

theorem part1 (h1 : 2 < m ∧ m < 6) : (set_A ∩ set_B m ≠ ∅) ↔ true :=
by sorry

theorem part2 (h2 : set_B m ⊆ set_A) : m = -2 ∨ (-1 ≤ m ∧ m ≤ 2) :=
by sorry

end part1_part2_l519_519001


namespace parallel_vectors_perpendicular_vectors_min_magnitude_l519_519270

noncomputable def vec_a (m : ℝ) : ℝ × ℝ × ℝ := (m, 2 * m, 2)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ × ℝ := (2 * m - 5, -m, -1)

theorem parallel_vectors (m : ℝ) (h : (m = 2)) : vec_a m = vec_b m := sorry

theorem perpendicular_vectors (m : ℝ) (h : (m = -2 / 5)) : 
  let a := vec_a m;
  let b := vec_b m;
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0 := sorry

theorem min_magnitude (m : ℝ) : ∃ m, (∀ x, sqrt (m * m + (2 * m) * (2 * m) + 2 ^ 2) ≥ sqrt (4)) := sorry

end parallel_vectors_perpendicular_vectors_min_magnitude_l519_519270


namespace sum_QS_l519_519993

noncomputable def g (z : ℂ) : ℂ := 1 - 2 * complex.I * conj z

def R (z : ℂ) : ℂ := z^4 + 6 * z^3 + 11 * z^2 + 6 * z + 1

def roots_R : list ℂ := [z1, z2, z3, z4] -- NOTE: You need the actual roots here.

def S (z : ℂ) : ℂ := 
  let transformed_roots := roots_R.map g in
  polynomial.of_roots transformed_roots

theorem sum_QS : 
  ∃ (Q S : ℂ), Q + S = 109 - 100 * complex.I := 
by sorry  -- Proof unfolds based on transformations and Vieta-like operations.

end sum_QS_l519_519993


namespace evaluate_expression_l519_519565

theorem evaluate_expression : 
  (let x := 3; let y := 4) in 5 * x^y + 2 * y^x = 533 :=
by
  sorry

end evaluate_expression_l519_519565


namespace monotonic_increase_interval_l519_519421

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4 * x - 5) / Real.log 2

lemma domain_condition {x : ℝ} : x^2 - 4 * x - 5 > 0 ↔ x < -1 ∨ x > 5 := 
by 
  sorry

theorem monotonic_increase_interval : 
  (∀ x y : ℝ, f x < f y) = (x ∈ Ioi (5 : ℝ)) := 
by 
  sorry

end monotonic_increase_interval_l519_519421


namespace geometric_progression_seventh_term_l519_519897

theorem geometric_progression_seventh_term :
  ∃ b1 q : ℚ,
    (b1 * (1 + q + q^2) = 91 ∧
    2 * (b1 * q + 27) = b1 + 25 + b1 * q^2 + 1) ∧
    (let b7_1 := b1 * q^6 in
     let b7_2 := b1 * q^6 in
     b7_1 = (35 * 46656) / 117649 ∨
     b7_2 = (63 * 4096) / 117649) :=
begin
  sorry
end

end geometric_progression_seventh_term_l519_519897


namespace midpoint_pentagon_perimeter_l519_519888

theorem midpoint_pentagon_perimeter (a : ℝ) :
  let original_sum_diagonals : ℝ := a in
  let new_perimeter : ℝ := original_sum_diagonals / 2 in
  new_perimeter = a / 2 :=
sorry

end midpoint_pentagon_perimeter_l519_519888


namespace inclination_angle_tangent_zero_l519_519531

theorem inclination_angle_tangent_zero (x : ℝ) (h : x = π / 4) :
  let y := λ x : ℝ, sin x + cos x
  let y' := (λ x : ℝ, cos x - sin x)
  let m := y' x
  m = 0 ↔ ((sin (π / 4) + cos (π / 4)) = 0 ∧ x = π / 4) :=
by
  sorry

end inclination_angle_tangent_zero_l519_519531


namespace tan_periodic_example_l519_519529

theorem tan_periodic_example (x : ℝ) (h : x = 17 * Real.pi / 4) : Real.tan x = 1 :=
by
  have periodicity : Real.tan (x - 2 * Real.pi) = Real.tan x := Real.tan_period
  have angle_conversion : (17 * Real.pi / 4 - 2 * Real.pi) = Real.pi / 4 :=
    by sorry -- angle conversion calculation
  rw [h, angle_conversion]
  exact Real.tan_pi_div_four -- tan π/4 = 1

end tan_periodic_example_l519_519529


namespace sector_area_correct_l519_519148

-- Definitions based on the conditions given
def central_angle_deg : ℝ := 150
def radius : ℝ := real.sqrt 3

-- Convert central angle to radians
def central_angle_rad : ℝ := central_angle_deg * real.pi / 180

-- Calculate expected area
def expected_area : ℝ := (1/2) * central_angle_rad * radius ^ 2

-- The theorem to prove the area of the sector is as expected
theorem sector_area_correct : expected_area = (5 / 4) * real.pi := 
by sorry

end sector_area_correct_l519_519148


namespace rhombus_longer_diagonal_length_l519_519135

theorem rhombus_longer_diagonal_length (side_length : ℕ) (shorter_diagonal : ℕ) 
  (h_side : side_length = 65) (h_short_diag : shorter_diagonal = 56) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 118 :=
by
  sorry

end rhombus_longer_diagonal_length_l519_519135


namespace number_of_good_permutations_l519_519592

def is_good_permutation {α : Type*} [linear_order α] [fintype α] (a : list α) : Prop :=
∃ i j, i < j ∧ a[i] > a[i+1] ∧ a[j] > a[j+1]

noncomputable def count_good_permutations (n : ℕ) : ℕ :=
fintype.card {a : vector (fin n) n // is_good_permutation a.to_list}

theorem number_of_good_permutations (n : ℕ) (h : 1 ≤ n) : 
  count_good_permutations n = 3^n - (n+1) * 2^n + n * (n+1) / 2 :=
sorry

end number_of_good_permutations_l519_519592


namespace overtime_pay_rate_increase_l519_519495

theorem overtime_pay_rate_increase
  (regular_rate : ℝ)
  (total_compensation : ℝ)
  (total_hours : ℝ)
  (overtime_hours : ℝ)
  (expected_percentage_increase : ℝ)
  (h1 : regular_rate = 16)
  (h2 : total_hours = 48)
  (h3 : total_compensation = 864)
  (h4 : overtime_hours = total_hours - 40)
  (h5 : 40 * regular_rate + overtime_hours * (regular_rate + regular_rate * expected_percentage_increase / 100) = total_compensation) :
  expected_percentage_increase = 75 := 
by
  sorry

end overtime_pay_rate_increase_l519_519495


namespace exists_inhabitant_with_many_acquaintances_l519_519761

open Function

theorem exists_inhabitant_with_many_acquaintances :
  (∃ (n : ℕ), n = 1_000_000) →
  (∀ (p : ℕ), p < 1_000_000 → ∃ (q : ℕ), q ≠ p) →
  (∀ (p : ℕ), p < 900_000 → p <= 900_000) →
  (∀ (p : ℕ), p < 1_000_000 → 0.1 * real.to_real p < 0.9 * real.to_real p) →
  ∃ (p : ℕ), p >= 810 :=
by
  intros h1 h2 h3 h4
  sorry

end exists_inhabitant_with_many_acquaintances_l519_519761


namespace commutative_binary_op_no_identity_element_associative_binary_op_l519_519553

def binary_op (x y : ℤ) : ℤ :=
  2 * (x + 2) * (y + 2) - 3

theorem commutative_binary_op (x y : ℤ) : binary_op x y = binary_op y x := by
  sorry

theorem no_identity_element (x e : ℤ) : ¬ (∀ x, binary_op x e = x) := by
  sorry

theorem associative_binary_op (x y z : ℤ) : (binary_op (binary_op x y) z = binary_op x (binary_op y z)) ∨ ¬ (binary_op (binary_op x y) z = binary_op x (binary_op y z)) := by
  sorry

end commutative_binary_op_no_identity_element_associative_binary_op_l519_519553


namespace b_50_is_2502_l519_519186

-- Definitions of the sequence and the conditions.
def b : ℕ → ℕ
| 0       := 0  -- To handle indexing starting at 1
| 1       := 3
| (n + 2) := b (n + 1) + 2 * (n + 1) + 1

-- Statement for the proof.
theorem b_50_is_2502 : b 50 = 2502 := 
sorry

end b_50_is_2502_l519_519186


namespace james_dvds_sold_per_day_l519_519331

theorem james_dvds_sold_per_day :
  let cost_per_dvd : ℤ := 6
  let selling_price_per_dvd : ℤ := 15
  let profit_per_dvd : ℤ := selling_price_per_dvd - cost_per_dvd
  let total_profit : ℤ := 448000
  let total_days : ℤ := 20 * 5
  let profit_per_day : ℤ := total_profit / total_days
  let dvds_sold_per_day : ℤ := profit_per_day / profit_per_dvd
  dvds_sold_per_day = 497 :=
by
  let cost_per_dvd := 6
  let selling_price_per_dvd := 15
  let profit_per_dvd := selling_price_per_dvd - cost_per_dvd
  let total_profit := 448000
  let total_days := 20 * 5
  let profit_per_day := total_profit / total_days
  let dvds_sold_per_day := profit_per_day / profit_per_dvd
  show dvds_sold_per_day = 497
  from sorry

end james_dvds_sold_per_day_l519_519331


namespace max_sum_is_103_sum_is_not_50_sum_of_59_values_l519_519094

noncomputable def unique_digit_mapping (M A R D T E I K U : ℕ) : Prop :=
  M ∈ {1,2,3,4,5,6,7,8,9} ∧
  A ∈ {1,2,3,4,5,6,7,8,9} ∧
  R ∈ {1,2,3,4,5,6,7,8,9} ∧
  D ∈ {1,2,3,4,5,6,7,8,9} ∧
  T ∈ {1,2,3,4,5,6,7,8,9} ∧
  E ∈ {1,2,3,4,5,6,7,8,9} ∧
  I ∈ {1,2,3,4,5,6,7,8,9} ∧
  K ∈ {1,2,3,4,5,6,7,8,9} ∧
  U ∈ {1,2,3,4,5,6,7,8,9} ∧
  M ≠ A ∧ M ≠ R ∧ M ≠ D ∧ M ≠ T ∧ M ≠ E ∧ M ≠ I ∧ M ≠ K ∧ M ≠ U ∧
  A ≠ R ∧ A ≠ D ∧ A ≠ T ∧ A ≠ E ∧ A ≠ I ∧ A ≠ K ∧ A ≠ U ∧
  R ≠ D ∧ R ≠ T ∧ R ≠ E ∧ R ≠ I ∧ R ≠ K ∧ R ≠ U ∧
  D ≠ T ∧ D ≠ E ∧ D ≠ I ∧ D ≠ K ∧ D ≠ U ∧
  T ≠ E ∧ T ≠ I ∧ T ≠ K ∧ T ≠ U ∧
  E ≠ I ∧ E ≠ K ∧ E ≠ U ∧
  I ≠ K ∧ I ≠ U ∧
  K ≠ U

theorem max_sum_is_103 (M A R D T E I K U : ℕ) (h : unique_digit_mapping M A R D T E I K U) :
  4 * M + 4 * A + 2 * T + R + D + E + I + K + U ≤ 103 :=
sorry

theorem sum_is_not_50 (M A R D T E I K U : ℕ) (h : unique_digit_mapping M A R D T E I K U) :
  4 * M + 4 * A + 2 * T + R + D + E + I + K + U ≠ 50 :=
sorry

theorem sum_of_59_values (M A R D T E I K U : ℕ) (h : unique_digit_mapping M A R D T E I K U) :
  4 * M + 4 * A + 2 * T + R + D + E + I + K + U = 59 → (M + A + M = 4 ∨ M + A + M = 5 ∨ M + A + M = 7) :=
sorry

end max_sum_is_103_sum_is_not_50_sum_of_59_values_l519_519094


namespace factor_values_l519_519561

theorem factor_values (a b : ℤ) :
  (∀ s : ℂ, s^2 - s - 1 = 0 → a * s^15 + b * s^14 + 1 = 0) ∧
  (∀ t : ℂ, t^2 - t - 1 = 0 → a * t^15 + b * t^14 + 1 = 0) →
  a = 377 ∧ b = -610 :=
by
  sorry

end factor_values_l519_519561


namespace total_slices_at_picnic_l519_519549

def danny_watermelons : ℕ := 3
def danny_slices_per_watermelon : ℕ := 10
def sister_watermelons : ℕ := 1
def sister_slices_per_watermelon : ℕ := 15

def total_danny_slices : ℕ := danny_watermelons * danny_slices_per_watermelon
def total_sister_slices : ℕ := sister_watermelons * sister_slices_per_watermelon
def total_slices : ℕ := total_danny_slices + total_sister_slices

theorem total_slices_at_picnic : total_slices = 45 :=
by
  sorry

end total_slices_at_picnic_l519_519549


namespace simplify_complex_expression_l519_519391

theorem simplify_complex_expression :
  let a := (-1 + complex.I * real.sqrt 7) / 2
  let b := (-1 - complex.I * real.sqrt 7) / 2
  a^4 + b^4 = 1 :=
by 
  sorry

end simplify_complex_expression_l519_519391


namespace bridge_length_l519_519964

noncomputable def train_length : ℝ := 250 -- in meters
noncomputable def train_speed_kmh : ℝ := 60 -- in km/hr
noncomputable def crossing_time : ℝ := 20 -- in seconds

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600 -- converting to m/s

noncomputable def total_distance_covered : ℝ := train_speed_ms * crossing_time -- distance covered in 20 seconds

theorem bridge_length : total_distance_covered - train_length = 83.4 :=
by
  -- The proof would go here
  sorry

end bridge_length_l519_519964


namespace product_abc_is_one_l519_519235

def A : ℝ := (Real.sqrt 2019 + Real.sqrt 2020 + 1)
def B : ℝ := (-Real.sqrt 2019 - Real.sqrt 2020 - 1)
def C : ℝ := (Real.sqrt 2019 - Real.sqrt 2020 + 1)
def D : ℝ := (Real.sqrt 2020 - Real.sqrt 2019 - 1)

theorem product_abc_is_one : A * B * C * D = 1 := by
  sorry

end product_abc_is_one_l519_519235


namespace compute_star_l519_519423

def star (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

theorem compute_star : star 1 (star 2 3) = 1 := by
  sorry

end compute_star_l519_519423


namespace percentage_of_alcohol_in_new_mixture_l519_519946

theorem percentage_of_alcohol_in_new_mixture :
  let original_mixture := 18 -- original mixture in liters
  let percentage_alcohol := 0.20 -- percentage of alcohol in the original mixture
  let added_water := 3 -- added water in liters
  let alcohol := percentage_alcohol * original_mixture -- 3.6 liters of alcohol
  let water := original_mixture - alcohol -- 18 - 3.6 = 14.4 liters of water
  let new_water := water + added_water -- 14.4 + 3 = 17.4 liters
  let new_mixture := original_mixture + added_water -- 18 + 3 = 21 liters
  let new_percentage_alcohol := (alcohol / new_mixture) * 100 -- (3.6 / 21) * 100
  new_percentage_alcohol ≈ 17.14 := sorry

end percentage_of_alcohol_in_new_mixture_l519_519946


namespace no_prime_divisible_by_77_l519_519282

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → is_divisible p 77 → false :=
by
  sorry

end no_prime_divisible_by_77_l519_519282


namespace rhombus_longer_diagonal_l519_519132

theorem rhombus_longer_diagonal (a b d : ℝ) (h : a = 65) (h_d : d = 56) :
  ∃ l, l = 118 :=
by
  sorry

end rhombus_longer_diagonal_l519_519132


namespace poodle_terrier_bark_ratio_l519_519505

theorem poodle_terrier_bark_ratio :
  ∀ (P T : ℕ),
  (T = 12) →
  (P = 24) →
  (P / T = 2) :=
by intros P T hT hP
   sorry

end poodle_terrier_bark_ratio_l519_519505


namespace range_of_a_product_of_zeros_l519_519660

def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - (Real.log x) + x - a

theorem range_of_a (a : ℝ) : (∀ x, (0 < x) → (f x a ≥ 0)) ↔ (a ≤ Real.exp 1 + 1) :=
by
  sorry

theorem product_of_zeros (x1 x2 a : ℝ) : 
  (0 < x1) → (x1 < 1) → (1 < x2) → (f x1 a = 0 ∧ f x2 a = 0) → x1 * x2 < 1 :=
by
  sorry

end range_of_a_product_of_zeros_l519_519660


namespace eccentricity_range_l519_519617

-- Define the conditions
variables {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0)
def hyperbola := (x : ℝ) (y : ℝ) := x^2/a^2 - y^2/b^2 = 1
def focus := (x : ℝ) := (x = sqrt(a^2 + b^2))

-- Translate the slope condition and eccentricity formula
def slope_condition (line_slope : ℝ) := line_slope = sqrt(3)
def asymptote_slope := b / a
def eccentricity := sqrt(1 + (b / a)^2)

-- The proof goal
theorem eccentricity_range (h₃ : asymptote_slope ≥ sqrt(3)) : ∃ e, eccentricity = e ∧ 2 ≤ e := 
by 
  have h4 : eccentricity ≥ 2 := by 
    calc
      eccentricity = sqrt(1 + (b / a)^2) : rfl
      ... ≥ sqrt(1 + 3) : by apply sqrt_le_sqrt; norm_num
      ... = 2 : rfl
  use eccentricity;
  exact ⟨rfl, h4⟩ 

#check eccentricity_range

end eccentricity_range_l519_519617


namespace tangent_line_eq_l519_519574

-- Define the function
def curve (x : ℝ) : ℝ := x^2 + Real.log x + 1

-- Define the point at which the tangent is evaluated
def point : ℝ × ℝ := (1, 2)

-- The statement to prove that the equation of the tangent line at point (1, 2) is y = 3x - 1
theorem tangent_line_eq :
  let x_1 := (1 : ℝ)
  let y_1 := (2 : ℝ)
  let k := (deriv curve x_1)
  (k = 3) → (∀ x : ℝ, curve x_1 = y_1 → k = 3 → ∀ y : ℝ, (y - y_1) = k * (x - x_1) → y = 3*x - 1) :=
by 
  sorry

end tangent_line_eq_l519_519574


namespace final_water_volume_is_correct_l519_519006

-- Define the dimensions of the aquarium
def length := 4
def width := 6
def height := 3

-- Define the total volume of the aquarium based on its dimensions
def total_volume := length * width * height

-- Define the initial water volume when the aquarium is filled halfway
def initial_volume := total_volume / 2

-- Define the remaining water volume after the cat spills half
def remaining_volume_after_spill := initial_volume / 2

-- Define the final water volume after Nancy triples the remaining water
def final_volume := remaining_volume_after_spill * 3

-- Proof statement: Prove that the final volume of water in the aquarium is 54 cubic feet
theorem final_water_volume_is_correct : final_volume = 54 := by
  sorry

end final_water_volume_is_correct_l519_519006


namespace parquet_tiles_needed_l519_519042

def room_width : ℝ := 8
def room_length : ℝ := 12
def tile_width : ℝ := 1.5
def tile_length : ℝ := 2

def room_area : ℝ := room_width * room_length
def tile_area : ℝ := tile_width * tile_length

def tiles_needed : ℝ := room_area / tile_area

theorem parquet_tiles_needed : tiles_needed = 32 :=
by
  -- sorry to skip the detailed proof
  sorry

end parquet_tiles_needed_l519_519042


namespace michael_saved_cookies_l519_519366

def initial_cupcakes := 9
def saved_cupcakes := initial_cupcakes / 3
def final_desserts := 11

theorem michael_saved_cookies :
  let given_cupcakes := initial_cupcakes - saved_cupcakes in
  let received_cookies := final_desserts - given_cupcakes in
  received_cookies = 5 := sorry

end michael_saved_cookies_l519_519366


namespace tan_alpha_third_quadrant_l519_519615

theorem tan_alpha_third_quadrant (α : ℝ) 
  (h_eq: Real.sin α = Real.cos α) 
  (h_third: π < α ∧ α < 3 * π / 2) : Real.tan α = 1 := 
by 
  sorry

end tan_alpha_third_quadrant_l519_519615


namespace part1_part2_l519_519651

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519651


namespace acquaintance_paradox_proof_l519_519710

theorem acquaintance_paradox_proof :
  ∃ (x : ℕ), (acquaintances x ≥ 810) :=
by
  let n_population := 1000000
  let n_believers := 900000
  let n_nonBelievers := n_population - n_believers
  have h1 : n_population = 1000000 := rfl
  have h2 : n_believers = 0.90 * n_population := by sorry
  have h3 : n_nonBelievers = n_population - n_believers := by sorry
  have h4 : ∀ x, acquaintances x ≥ 1 := by sorry
  have h5 : ∀ x, 0.10 * acquaintances x = nonBelievers_acquaintances x := by sorry
  have h6 : nonBelievers_acquaintances_total >= 90 := by sorry
  
  show ∃ (x : ℕ), acquaintances x ≥ 810 from sorry
  
  sorry

end acquaintance_paradox_proof_l519_519710


namespace geometric_cosine_sequence_l519_519204

theorem geometric_cosine_sequence (a : ℝ) (h₀ : 0 < a ∧ a < 360) :
  (cos a, cos (2 * a), cos (3 * a)) form_geom_seq → 
  a = 45 ∨ a = 135 ∨ a = 225 ∨ a = 315 :=
by 
  sorry

end geometric_cosine_sequence_l519_519204


namespace function_fixed_point_l519_519884

theorem function_fixed_point (a : ℝ) : (1 : ℝ, 3 : ℝ) ∈ set_of (λ x, f x = a^(x-1) + 2) :=
sorry

end function_fixed_point_l519_519884


namespace exists_inhabitant_with_810_acquaintances_l519_519765

-- Definition of the main problem conditions
def inhabitants := 1000000
def belief_in_santa := 0.9 * inhabitants
def disbelief_in_santa := inhabitants - belief_in_santa

-- Conditions
def knows_at_least_one : ∀ x : ℕ, x < inhabitants → ∃ y : ℕ, y < inhabitants ∧ x ≠ y := sorry
def ten_percent_acquaintances (x : ℕ) (h : x < inhabitants) : Prop := ∀ y : ℕ, y < inhabitants ∧ x ≠ y → (y < belief_in_santa → y < 0.1 * x) := sorry

-- Main theorem
theorem exists_inhabitant_with_810_acquaintances :
  ∃ x : ℕ, x < inhabitants ∧ (knows_at_least_one x) ∧ (ten_percent_acquaintances x) ∧ (810 ≤ x) :=
sorry

end exists_inhabitant_with_810_acquaintances_l519_519765


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519735

theorem exists_inhabitant_with_at_least_810_acquaintances
  (N : ℕ) (hN : N = 1000000)
  (B : Finset ℕ) (hB : B.card = 900000)
  (A : ℕ → Finset ℕ)
  (hA_nonempty : ∀ i, (A i).nonempty)
  (hA_believers : ∀ i, (A i).card / 10 = (A i ∩ B).card) :
  ∃ j, (A j).card ≥ 810 :=
by
  sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519735


namespace interval_of_increase_of_f_l519_519191

noncomputable def f (x : ℝ) := Real.logb (0.5) (x - x^2)

theorem interval_of_increase_of_f :
  ∀ x : ℝ, x ∈ Set.Ioo (1/2) 1 → ∃ ε > 0, ∀ y : ℝ, y ∈ Set.Ioo (x - ε) (x + ε) → f y > f x :=
  by
    sorry

end interval_of_increase_of_f_l519_519191


namespace number_of_female_muscovy_ducks_l519_519901

def ducks_total := 40
def muscovy_percent := 0.50
def female_muscovy_percent := 0.30

theorem number_of_female_muscovy_ducks : ducks_total * muscovy_percent * female_muscovy_percent = 6 :=
by {
  let muscovy_ducks := ducks_total * muscovy_percent
  let female_muscovy := muscovy_ducks * female_muscovy_percent
  have : female_muscovy = 6, by {
    sorry -- Proof details are omitted
  },
  exact this,
}

end number_of_female_muscovy_ducks_l519_519901


namespace triangle_area_B_equals_C_tan_ratio_sum_l519_519326

-- First part: Proving the area of the triangle
theorem triangle_area_B_equals_C {A B C a b c : ℝ} (h1 : B = C) (h2 : a = 2) (h3 : b^2 + c^2 = 3 * b * c * cos A) 
    (h4 : B + C = 180) :
    0.5 * b * c * sin A = sqrt(5) := 
by
  sorry

-- Second part: Proving the value of tan A / tan B + tan A / tan C
theorem tan_ratio_sum {A B C a b c : ℝ} (h1 : b^2 + c^2 = 3 * b * c * cos A) (h2 : A + B + C = 180) :
    (tan A / tan B) + (tan A / tan C) = 1 :=
by
  sorry

end triangle_area_B_equals_C_tan_ratio_sum_l519_519326


namespace option_c_forms_triangle_l519_519926

theorem option_c_forms_triangle (a b c : ℕ) (h : {a, b, c} = {6, 8, 13}) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  fin_cases {a, b, c} using h
  · -- Case {a, b, c} = {6, 8, 13}
    simp
    sorry

end option_c_forms_triangle_l519_519926


namespace exists_inhabitant_with_many_acquaintances_l519_519756

open Function

theorem exists_inhabitant_with_many_acquaintances :
  (∃ (n : ℕ), n = 1_000_000) →
  (∀ (p : ℕ), p < 1_000_000 → ∃ (q : ℕ), q ≠ p) →
  (∀ (p : ℕ), p < 900_000 → p <= 900_000) →
  (∀ (p : ℕ), p < 1_000_000 → 0.1 * real.to_real p < 0.9 * real.to_real p) →
  ∃ (p : ℕ), p >= 810 :=
by
  intros h1 h2 h3 h4
  sorry

end exists_inhabitant_with_many_acquaintances_l519_519756


namespace total_area_correct_l519_519157

def base1 (trapezoid) : Nat := 3
def base2 (trapezoid) : Nat := 5
def height_trapezoid : Nat := 2
def base_triangle : Nat := 5
def height_triangle : Nat := 4

def area_trapezoid : Nat := (1 / 2) * (base1 trapezoid + base2 trapezoid) * height_trapezoid
def area_triangle : Nat := (1 / 2) * base_triangle * height_triangle

def total_area_tv : Nat := area_trapezoid + area_triangle

theorem total_area_correct : total_area_tv = 18 := by
  sorry

end total_area_correct_l519_519157


namespace digits_2_pow_2015_l519_519606

theorem digits_2_pow_2015 (log2 : Float) (h : log2 = 0.3010) : 
  let t := 2 ^ 2015 in
  Nat.log10 t + 1 = 607 :=
by
  sorry

end digits_2_pow_2015_l519_519606


namespace finite_fraction_n_iff_l519_519096

theorem finite_fraction_n_iff (n : ℕ) (h_pos : 0 < n) :
  (∃ (a b : ℕ), n * (n + 1) = 2^a * 5^b) ↔ (n = 1 ∨ n = 4) :=
by
  sorry

end finite_fraction_n_iff_l519_519096


namespace apples_per_friend_l519_519977

def Benny_apples : Nat := 5
def Dan_apples : Nat := 2 * Benny_apples
def Total_apples : Nat := Benny_apples + Dan_apples
def Number_of_friends : Nat := 3

theorem apples_per_friend : Total_apples / Number_of_friends = 5 := by
  sorry

end apples_per_friend_l519_519977


namespace correct_proposition_l519_519318

theorem correct_proposition (a b c : ℝ) (h₁ : a > b) (h₂ : b > c) (h₃ : a + b + c = 0) : ab > ac :=
by
  sorry

end correct_proposition_l519_519318


namespace fillSquares_l519_519203

theorem fillSquares {n : ℕ} (squares : Fin 4 → ℕ) :
  (∀ i : Fin 4, squares i ∈ {1, 2, 3, 4}) ∧
  (Function.Injective squares) ∧
  (∀ i : Fin 4, squares i ≠ i.succ) →
  ∃! (solutions : Finset (Fin 4 → ℕ)), solutions.card = 3 :=
by
  sorry

end fillSquares_l519_519203


namespace find_alpha_l519_519775

variable {α p₀ p_new : ℝ}
def Q_d (p : ℝ) : ℝ := 150 - p
def Q_s (p : ℝ) : ℝ := 3 * p - 10
def Q_d_new (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

theorem find_alpha 
  (h_eq_initial : Q_d p₀ = Q_s p₀)
  (h_eq_increase : p_new = 1.25 * p₀)
  (h_eq_new : Q_s p_new = Q_d_new α p_new) :
  α = 1.4 :=
by
  sorry

end find_alpha_l519_519775


namespace probability_of_perfect_square_sum_l519_519912

noncomputable def spinner_probability_perfect_square : ℚ :=
let outcomes1 := [2, 3, 4],
    outcomes2 := [1, 5, 7],
    perfect_squares := [4, 9] in
let sums := (outcomes1.product outcomes2).map (λ x => x.1 + x.2) in
let perfect_square_sums := sums.filter (λ sum => sum ∈ perfect_squares) in
(rat.of_int perfect_square_sums.length) / (rat.of_int sums.length)

theorem probability_of_perfect_square_sum :
  spinner_probability_perfect_square = 1 / 3 :=
by
  sorry

end probability_of_perfect_square_sum_l519_519912


namespace arithmetic_mean_neg5_to_4_l519_519455

theorem arithmetic_mean_neg5_to_4 :
  let s := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4] in
  (list.sum s : ℚ) / list.length s = 0.0 :=
by
  sorry

end arithmetic_mean_neg5_to_4_l519_519455


namespace volume_conversion_l519_519129

-- Define the given conditions
def V_feet : ℕ := 216
def C_factor : ℕ := 27

-- State the theorem to prove
theorem volume_conversion : V_feet / C_factor = 8 :=
  sorry

end volume_conversion_l519_519129


namespace proof_theorem_l519_519290

noncomputable def proof_problem : Prop :=
  let a := 6
  let b := 15
  let c := 7
  let lhs := a * b * c
  let rhs := (Real.sqrt ((a^2) + (2 * a) + (b^3) - (b^2) + (3 * b))) / (c^2 + c + 1) + 629.001
  lhs = rhs

theorem proof_theorem : proof_problem :=
  by
  sorry

end proof_theorem_l519_519290


namespace area_transformed_S_l519_519823

open Matrix

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![-2, 1]]

def area_S : ℝ := 5

theorem area_transformed_S (area_S': ℝ) :
  let det_A := det matrix_A in
  det_A = 11 →
  area_S' = det_A * area_S →
  area_S' = 55 :=
by
  intro det_A_eq
  intro area_S'_eq
  rw [det_A_eq, area_S'_eq]
  exact eq.refl 55

end area_transformed_S_l519_519823


namespace count_valid_B_sets_l519_519843

-- Define set A
def A : Set ℕ := {1, 2}

-- Define the condition on sets B
def valid_B (B : Set ℕ) : Prop :=
  (A ∪ B = {1, 2, 3})

-- The theorem stating the number of valid B sets is 4
theorem count_valid_B_sets : {B : Set ℕ // valid_B B}.to_finset.card = 4 :=
sorry

end count_valid_B_sets_l519_519843


namespace function_not_odd_domain_of_f_range_of_f_l519_519474

noncomputable def f (x : ℝ) : ℝ := (sqrt (x^2 + 1) + x - 1) / (sqrt (x^2 + 1) + x + 1)

theorem function_not_odd : ¬(∀ x : ℝ, f (-x) = -f x) :=
by
  unfold f
  sorry

theorem domain_of_f : ∀ x : ℝ, x ∈ set.univ :=
by
  intros x
  exact set.mem_univ x

theorem range_of_f : range f = set.Ioo (-1) 1 :=
by
  unfold f
  sorry

end function_not_odd_domain_of_f_range_of_f_l519_519474


namespace line_equation_through_point_circle_intersection_l519_519120

theorem line_equation_through_point_circle_intersection 
  (P : ℝ × ℝ) (hP : P = (1, 1)) 
  (C : set (ℝ × ℝ)) (hC : C = {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 9}) 
  (A B : ℝ × ℝ) (hA : A ∈ C) (hB : B ∈ C) 
  (hAB : dist A B = 4) :
  ∃ k : ℝ, ∃ b : ℝ, (b = -3) ∧ (k = 2) ∧ ( ∀ x y : ℝ, y = k * x + b ↔ x + 2 * y - 3 = 0 ) :=
sorry

end line_equation_through_point_circle_intersection_l519_519120


namespace problem_I_problem_II_problem_III_l519_519231

section SequenceProblem

variable (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_init : a 1 = 2)
          (h_seq : ∀ n, a (n+1)^2 + 2 * a (n+1) = a n + 2)

-- Problem (I)
theorem problem_I (n : ℕ) (hn : n > 0) :
  ((a (n+2) - a (n+1)) * (a (n+2) + a (n+1) + 2) = a (n+1) - a n) ∧ (a (n+1) < a n) := 
sorry 

-- Problem (II)
theorem problem_II (n : ℕ) (hn : n > 0) :
  |a (n+1) - 1| < (1/4) * |a n - 1| := 
sorry

-- Problem (III)
theorem problem_III (n : ℕ) (hn : n > 0) :
  (∑ i in finset.range n, |a i.succ - 1|) < (4/3 : ℝ) := 
sorry

end SequenceProblem

end problem_I_problem_II_problem_III_l519_519231


namespace lamp_probability_l519_519442

theorem lamp_probability : 
  let total_outlets := 7
  let wall_socket := 1
  let power_strips := 2
  let outlets_per_strip := 3
  let total_ways := 210
  let favorable_ways := 78 in 
  (favorable_ways / total_ways) = (13 / 35) :=
by
  let total_outlets := 7
  let wall_socket := 1
  let power_strips := 2
  let outlets_per_strip := 3
  let total_ways := 7 * 6 * 5
  let favorable_ways := 30 + 30 + 18
  have total_ways_calc : total_ways = 210 := by norm_num
  have favorable_ways_calc : favorable_ways = 78 := by norm_num
  rw [total_ways_calc, favorable_ways_calc]
  apply div_eq_div_of_eq
  norm_num
  sorry

end lamp_probability_l519_519442


namespace distance_between_vertices_of_hyperbola_l519_519208

theorem distance_between_vertices_of_hyperbola :
  let a := real.sqrt 48 in 2 * a = 8 * real.sqrt 3 → ∀ x y : ℝ, (y^2 / 48 - x^2 / 16 = 1) → (y = 0) := by
  sorry

end distance_between_vertices_of_hyperbola_l519_519208


namespace square_area_side4_l519_519032

theorem square_area_side4
  (s : ℕ)
  (A : ℕ)
  (P : ℕ)
  (h_s : s = 4)
  (h_A : A = s * s)
  (h_P : P = 4 * s)
  (h_eqn : (A + s) - P = 4) : A = 16 := sorry

end square_area_side4_l519_519032


namespace triangle_is_acute_l519_519300

-- Definition of the problem statements as conditions
def arithmetic_seq_common_diff (a3 a7 : ℤ) : ℤ := (a7 - a3) / 4
def geometric_seq_common_ratio (g3 g6 : ℝ) : ℝ := real.root (g6 / g3) 3

-- Conditions from the problem statement
def tan_A := arithmetic_seq_common_diff (-4 : ℤ) (4 : ℤ)
def tan_B := geometric_seq_common_ratio (1 / 3 : ℝ) (9 : ℝ)
def tan_sum (tanA tanB : ℝ) : ℝ := (tanA + tanB) / (1 - tanA * tanB)

-- Lean statement for the problem
theorem triangle_is_acute :
  tan_sum (tan_A : ℝ) (tan_B : ℝ) = -1 → ∀ A B C : ℝ, 
  (real.tan A = tan_A) → (real.tan B = tan_B) → (real.tan C = 1) → 
  A < real.pi / 2 ∧ B < real.pi / 2 ∧ C < real.pi / 2 := 
sorry

end triangle_is_acute_l519_519300


namespace quadratic_roots_l519_519868

theorem quadratic_roots : ∀ x : ℝ, (x^2 - 6 * x + 5 = 0) ↔ (x = 5 ∨ x = 1) :=
by sorry

end quadratic_roots_l519_519868


namespace problem_statement_l519_519683

theorem problem_statement (x : ℝ) (h : x + sqrt(x^2 - 1) + 1 / (x - sqrt(x^2 - 1)) = 15) :
  x^3 + sqrt(x^6 - 1) + 1 / (x^3 + sqrt(x^6 - 1)) = 3970049 / 36000 := 
by
  sorry

end problem_statement_l519_519683


namespace max_profit_l519_519107

noncomputable def revenue (x : ℝ) : ℝ := 
  if (0 < x ∧ x ≤ 10) then 13.5 - (1 / 30) * x^2 
  else if (x > 10) then (168 / x) - (2000 / (3 * x^2)) 
  else 0

noncomputable def cost (x : ℝ) : ℝ := 
  20 + 5.4 * x

noncomputable def profit (x : ℝ) : ℝ := revenue x * x - cost x

theorem max_profit : 
  ∃ (x : ℝ), 0 < x ∧ x ≤ 10 ∧ (profit x = 8.1 * x - (1 / 30) * x^3 - 20) ∧ 
    (∀ (y : ℝ), 0 < y ∧ y ≤ 10 → profit y ≤ profit 9) ∧ 
    ∀ (z : ℝ), z > 10 → profit z ≤ profit 9 :=
by
  sorry

end max_profit_l519_519107


namespace soccer_team_wins_l519_519512

theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℕ) (games_won : ℕ) :
  total_games = 280 → win_percentage = 65 → games_won = 65 * 280 / 100 → games_won = 182 :=
by
  intros h_total_games h_win_percentage h_games_won
  rw [h_total_games, h_win_percentage] at h_games_won
  exact h_games_won

end soccer_team_wins_l519_519512


namespace gcd_polynomial_l519_519608

-- Given definitions based on the conditions
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

-- Given the conditions: a is a multiple of 1610
variables (a : ℕ) (h : is_multiple_of a 1610)

-- Main theorem: Prove that gcd(a^2 + 9a + 35, a + 5) = 15
theorem gcd_polynomial (h : is_multiple_of a 1610) : gcd (a^2 + 9*a + 35) (a + 5) = 15 :=
sorry

end gcd_polynomial_l519_519608


namespace exists_inhabitant_with_many_acquaintances_l519_519763

open Function

theorem exists_inhabitant_with_many_acquaintances :
  (∃ (n : ℕ), n = 1_000_000) →
  (∀ (p : ℕ), p < 1_000_000 → ∃ (q : ℕ), q ≠ p) →
  (∀ (p : ℕ), p < 900_000 → p <= 900_000) →
  (∀ (p : ℕ), p < 1_000_000 → 0.1 * real.to_real p < 0.9 * real.to_real p) →
  ∃ (p : ℕ), p >= 810 :=
by
  intros h1 h2 h3 h4
  sorry

end exists_inhabitant_with_many_acquaintances_l519_519763


namespace choose_amber_bronze_cells_l519_519838

theorem choose_amber_bronze_cells (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (grid : Fin (a+b+1) × Fin (a+b+1) → Prop) 
  (amber_cells : ℕ) (h_amber_cells : amber_cells ≥ a^2 + a * b - b)
  (bronze_cells : ℕ) (h_bronze_cells : bronze_cells ≥ b^2 + b * a - a):
  ∃ (amber_choice : Fin (a+b+1) → Fin (a+b+1)), 
    ∃ (bronze_choice : Fin (a+b+1) → Fin (a+b+1)), 
    amber_choice ≠ bronze_choice ∧ 
    (∀ i j, i ≠ j → grid (amber_choice i) ≠ grid (bronze_choice j)) :=
sorry

end choose_amber_bronze_cells_l519_519838


namespace geometric_series_ratio_l519_519866

theorem geometric_series_ratio (a_1 a_2 S q : ℝ) (hq : |q| < 1)
  (hS : S = a_1 / (1 - q))
  (ha2 : a_2 = a_1 * q) :
  S / (S - a_1) = a_1 / a_2 := 
sorry

end geometric_series_ratio_l519_519866


namespace suna_total_distance_l519_519873

theorem suna_total_distance (D : ℝ) (h_train : D * (7 / 15)) 
(h_bus : D * (8 / 15) * (5 / 8)) 
(h_taxi : D * (1 / 5)  * (2 / 3)) 
(h_remaining : D * (1 / 15) = 2.6) :
D = 39 := 
sorry

end suna_total_distance_l519_519873


namespace diameter_intersects_7_chords_l519_519302

theorem diameter_intersects_7_chords
  (chords : set (set (ℝ × ℝ)))
  (h_circle : ∃ c : ℝ × ℝ, ∀ p ∈ chords, (dist p c <= 1))
  (h_chord_lengths : ∑ l in chords, (chord_length l) > 19) :
  ∃ d : set (ℝ × ℝ), (is_diameter d) ∧ (∑ l in (chords ∩ d), (chord_length l) ≥ 7) := 
sorry

noncomputable def chord_length (l : set (ℝ × ℝ)) : ℝ :=
sorry

noncomputable def is_diameter (d : set (ℝ × ℝ)) : Prop :=
sorry

-- Assuming necessary geometric definitions and functions will be implemented above.

end diameter_intersects_7_chords_l519_519302


namespace probability_of_xi_greater_than_one_l519_519248

noncomputable def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

variable (ξ : ℝ → ℝ)
variable (p : ℝ)

axiom normal_0_1_distribution : ξ = normal_distribution 0 1
axiom probability_condition : ∀ x, P(-1 < ξ x ∧ ξ x < 0) = p

theorem probability_of_xi_greater_than_one (ξ : ℝ → ℝ) (p : ℝ)
  (hξ : ξ = normal_distribution 0 1)
  (hp : ∀ x, P(-1 < ξ x ∧ ξ x < 0) = p) :
  P(ξ > 1) = 1 / 2 - p := by
  sorry

end probability_of_xi_greater_than_one_l519_519248


namespace slicing_results_in_ellipse_l519_519528

-- Definition and condition for the ellipse
structure Ellipse where
  a b : ℝ
  h : a > b

-- Definition of the oblate spheroid from the ellipse rotating around its minor axis
def OblateSpheroid (e : Ellipse) : Type := 
  -- Details of construction are abstracted for now
  sorry

-- Definition of a slicing plane parallel to the rotation axis
def SlicingPlaneParallelToRotationAxis : Type := 
  -- Details of the plane construction are abstracted
  sorry

-- The main theorem to be proved
theorem slicing_results_in_ellipse (e : Ellipse) (spheroid : OblateSpheroid e) (plane : SlicingPlaneParallelToRotationAxis) : 
  is_elliptical_intersection plane spheroid := 
  sorry

-- In a real proof, 'is_elliptical_intersection' would be defined in detail
def is_elliptical_intersection (plane : SlicingPlaneParallelToRotationAxis) (spheroid : OblateSpheroid e) : Prop :=
  -- This would spell out the intersection nature
  sorry

end slicing_results_in_ellipse_l519_519528


namespace at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519722

theorem at_least_one_inhabitant_knows_no_fewer_than_810_people :
  ∀ (n : ℕ),
  (inhabitants : ℕ) (knows_someone : inhabitants > 1) (believes_santa : inhabitants * 9 / 10)
  (ten_percent_acquaintances_believe : ∀ x, (acquaintances : list x) → (|list.contacts_with_santa <- 0.10 * acquaintances|)),
  (h : inhabitants = 1000000),
  (one_inhabitant_knows : ∃ x, ∀ y, y \in acquaintances x → friends y ≥ 810) :=
sorry

end at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519722


namespace skittles_distribution_l519_519168

-- Given problem conditions
variable (Brandon_initial : ℕ := 96) (Bonnie_initial : ℕ := 4) 
variable (Brandon_loss : ℕ := 9)
variable (combined_skittles : ℕ := (Brandon_initial - Brandon_loss) + Bonnie_initial)
variable (individual_share : ℕ := combined_skittles / 4)
variable (remainder : ℕ := combined_skittles % 4)
variable (Chloe_share : ℕ := individual_share)
variable (Dylan_share_initial : ℕ := individual_share)
variable (Chloe_to_Dylan : ℕ := Chloe_share / 2)
variable (Dylan_new_share : ℕ := Dylan_share_initial + Chloe_to_Dylan)
variable (Dylan_to_Bonnie : ℕ := Dylan_new_share / 3)
variable (final_Bonnie : ℕ := individual_share + Dylan_to_Bonnie)
variable (final_Chloe : ℕ := Chloe_share - Chloe_to_Dylan)
variable (final_Dylan : ℕ := Dylan_new_share - Dylan_to_Bonnie)

-- The theorem to be proved
theorem skittles_distribution : 
  individual_share = 22 ∧ final_Bonnie = 33 ∧ final_Chloe = 11 ∧ final_Dylan = 22 :=
by
  -- The proof would go here, but it’s not required for this task.
  sorry

end skittles_distribution_l519_519168


namespace average_of_all_results_l519_519033

theorem average_of_all_results (sum1 sum2 : ℝ) (n1 n2 : ℕ) 
  (h1 : n1 = 55) (h2 : n2 = 28) 
  (avg1 : sum1 / n1 = 28) (avg2 : sum2 / n2 = 55) :
  (sum1 + sum2) / (n1 + n2) ≈ 37.11 :=
by
  sorry

end average_of_all_results_l519_519033


namespace range_of_a_l519_519293

noncomputable def has_one_solution_in_interval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∃! x ∈ I, f x = 0

theorem range_of_a
  (a : ℝ)
  (h : has_one_solution_in_interval (λ x : ℝ, 2 * a * x ^ 2 - x - 1) {x | 0 < x ∧ x < 1}) :
  a > 1 := 
sorry

end range_of_a_l519_519293


namespace ellipse_equation_l519_519209

theorem ellipse_equation :
  ( ∃ (a b : ℝ), a = 3 ∧ b = sqrt(5) ∧ (forall (x y : ℝ), (x = 2 ∧ y = 5 / 3) →
  ((x^2 / a^2) + (y^2 / b^2) = 1)) ) :=
begin
  let F1 := (-2:ℝ, 0:ℝ),
  let F2 := (2:ℝ, 0:ℝ),
  let P := (2:ℝ, 5 / 3:ℝ),
  -- distances from P to foci
  let d1 := real.sqrt ((2 - (-2))^2 + (5 / 3 - 0)^2),
  let d2 := real.sqrt ((2 - 2)^2 + (5 / 3 - 0)^2),
  -- sum of distances equals 2a
  let a := 3, -- derived in solution, 2a = 6
  let c := 2, -- distance to each focus
  have a_sq : a^2 = 9 := by norm_num,
  have c_sq : c^2 = 4 := by norm_num,
  have b_sq := a_sq - c_sq,
  have b := real.sqrt b_sq,
  have b_expected: b = real.sqrt 5 := by simp [b_sq],
  use a, use b,
  split, exact rfl, split, exact b_expected,
  intros x y h,
  rw [h.1, h.2],
  simp only [(*), has_div.div, pow, has_pow.pow, has_add.add, div_eq_mul_inv, real.sqrt_sq_eq_abs, pow_two],
  exact rfl,
end

end ellipse_equation_l519_519209


namespace find_x_l519_519688

-- Conditions
variables {x y : ℝ}
hypothesis h1 : x ≠ 0
hypothesis h2 : x / 3 = y^2
hypothesis h3 : x / 5 = 5 * y

-- Goal
theorem find_x (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 5 = 5 * y) : x = 625 / 3 :=
by
  sorry

end find_x_l519_519688


namespace part1_part2_l519_519636

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l519_519636


namespace person_speed_is_5_31_kmph_l519_519956

noncomputable def speed_in_kmph (distance_m : ℕ) (time_min : ℕ) : ℝ :=
  let distance_km := (distance_m : ℝ) / 1000
  let time_hr := (time_min : ℝ) / 60
  distance_km / time_hr

theorem person_speed_is_5_31_kmph :
  speed_in_kmph 708 8 ≈ 5.31 :=
sorry

end person_speed_is_5_31_kmph_l519_519956


namespace r_cube_plus_inv_r_cube_eq_zero_l519_519289

theorem r_cube_plus_inv_r_cube_eq_zero {r : ℝ} (h : (r + 1/r)^2 = 3) : r^3 + 1/r^3 = 0 := 
sorry

end r_cube_plus_inv_r_cube_eq_zero_l519_519289


namespace max_gcd_value_l519_519189

open Nat

-- Definitions of the sequence and gcd condition
def a_n (n : ℕ) : ℕ := n^3 + 4

def d_n (n : ℕ) : ℕ := gcd (a_n n) (a_n (n + 1))

-- Statement of the proof problem
theorem max_gcd_value : ∃ n : ℕ, d_n n = 433 ∧ ∀ m : ℕ, d_n m ≤ 433 := 
by
  sorry

end max_gcd_value_l519_519189


namespace induction_sum_formula_l519_519913

open Nat

theorem induction_sum_formula (a : ℝ) (h : a ≠ 1) (n : ℕ) :
  (1 + ∑ i in range (n + 2), a ^ i) = (1 - a ^ (n + 2)) / (1 - a) := sorry

example (a : ℝ) (h : a ≠ 1) : (1 + a + a^2) = 1 + a + a^2 := by
    exact eq.refl _                                        -- This example verifies for n = 1

end induction_sum_formula_l519_519913


namespace number_of_rare_cards_l519_519446

-- Definitions based on conditions
def total_cost_of_deck (R : ℕ) : Prop :=
  R * 1 + 11 * 0.5 + 30 * 0.25 = 32

-- The proof statement
theorem number_of_rare_cards (R : ℕ) (h : total_cost_of_deck R) : R = 19 :=
sorry

end number_of_rare_cards_l519_519446


namespace curve_intersection_one_point_l519_519084

theorem curve_intersection_one_point (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = a^2 ↔ y = x^2 + a) → (x, y) = (0, a)) ↔ (a ≥ -1/2) := 
sorry

end curve_intersection_one_point_l519_519084


namespace basketball_opponents_score_l519_519491

theorem basketball_opponents_score :
  ∃ (opponents_scores : list ℕ),
    -- Initial conditions:
    length opponents_scores = 12 ∧
    let team_scores := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
    -- Condition 1: They lost by one point in exactly six games.
    let lost_games := [1, 3, 5, 7, 9, 11] in
    let won_games := [2, 4, 6, 8, 10, 12] in
    (∀ i ∈ lost_games, opponents_scores[i] = team_scores[i] + 1) ∧
    -- Condition 2: In each of their other games, they scored three times as many points as their opponent.
    (∀ i ∈ won_games, team_scores[i] = 3 * opponents_scores[i]) ∧
    -- Total score calculation:
    list.sum opponents_scores = 60 :=
begin
  sorry
end

end basketball_opponents_score_l519_519491


namespace eq_factorial_sum_l519_519556

theorem eq_factorial_sum (k l m n : ℕ) (hk : k > 0) (hl : l > 0) (hm : m > 0) (hn : n > 0) :
  (1 / (Nat.factorial k : ℝ) + 1 / (Nat.factorial l : ℝ) + 1 / (Nat.factorial m : ℝ) = 1 / (Nat.factorial n : ℝ))
  ↔ (k = 3 ∧ l = 3 ∧ m = 3 ∧ n = 2) :=
by
  sorry

end eq_factorial_sum_l519_519556


namespace part1_part2_l519_519654

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519654


namespace rhombus_longer_diagonal_l519_519146

theorem rhombus_longer_diagonal
  (a b : ℝ)
  (ha : a = 65) -- Side length
  (hb : b = 28) -- Half the length of the shorter diagonal
  : 2 * Real.sqrt (a^2 - b^2) = 118 :=  -- Statement of the longer diagonal length
by
  sorry -- Proof goes here

end rhombus_longer_diagonal_l519_519146


namespace triangle_ABC_angles_l519_519316

open EuclideanGeometry

theorem triangle_ABC_angles 
  (A B C D : Point) 
  (h1 : distance A D = distance B D) 
  (h2 : distance B D = distance C D) 
  (h3 : ∠ A D B = 90) 
  (h4 : ∠ A D C = 50) 
  (h5 : ∠ B D C = 140) 
  : ∠ A B C = 25 ∧ ∠ A C B = 45 ∧ ∠ B A C = 110 := 
sorry

end triangle_ABC_angles_l519_519316


namespace num_ordered_pairs_l519_519581

open Real

theorem num_ordered_pairs (a b : ℝ) : 
  (∃ x y : ℤ, a*x + b*y = 1 ∧ (x-3)^2 + (y+4)^2 = 85) ↔
  (sum_range_satisfy_eq_72 a b) := 
by 
  sorry

-- We assume sum_range_satisfy_eq_72 is a proven lemma or 
-- additional condition that cumulatively counts valid (a, b)
-- Based on assumptions taken from the solution, sum_range_satisfy_eq_72 
-- is expected to tally up to 72.

end num_ordered_pairs_l519_519581


namespace smallest_base_l519_519078

theorem smallest_base (b : ℕ) (h1 : b^2 ≤ 125) (h2 : 125 < b^3) : b = 6 := by
  sorry

end smallest_base_l519_519078


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519731

theorem exists_inhabitant_with_at_least_810_acquaintances :
  (∀ x, ∃ y, x ≠ y) → 
  (∀ x, (∀ k, k = 10% * |neighbors x| → |{y : neighbors x | believes_in_santa y}| = k)) →
  (1_000_000 ≥ ∑ x, believes_in_santa x → 900_000) → 
  (1_000_000 - 900_000 = 100_000) →
  (∃ x, |neighbors x| ≥ 810) :=
sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519731


namespace third_number_sixth_row_l519_519159

/-- Define the arithmetic sequence and related properties. -/
def sequence (n : ℕ) : ℕ := 2 * n - 1

/-- Define sum of first k terms in a series where each row length doubles the previous row length. -/
def sum_of_rows (k : ℕ) : ℕ :=
  2^k - 1

/-- Statement of the problem: Prove that the third number in the sixth row is 67. -/
theorem third_number_sixth_row : sequence (sum_of_rows 5 + 3) = 67 := by
  sorry

end third_number_sixth_row_l519_519159


namespace hyperbola_asymptotes_l519_519880

theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, x^2 / 16 - y^2 / 9 = -1 → (y = 3/4 * x ∨ y = -3/4 * x)) :=
by
  sorry

end hyperbola_asymptotes_l519_519880


namespace sqrt_recursive_equals_one_l519_519199

theorem sqrt_recursive_equals_one (x y : ℝ) (hx : x = 1) (hy : y = sqrt (2 - x)) : y = 1 :=
by sorry

end sqrt_recursive_equals_one_l519_519199


namespace overall_average_output_l519_519161

theorem overall_average_output :
  ∀ (cogs_per_hour1 cogs1 : ℕ) (cogs_per_hour2 cogs2 : ℕ) (cogs_per_hour3 cogs3 : ℕ),
  cogs_per_hour1 = 36 → cogs1 = 60 →
  cogs_per_hour2 = 60 → cogs2 = 90 →
  cogs_per_hour3 = 45 → cogs3 = 120 →
  let total_cogs := cogs1 + cogs2 + cogs3 in
  let total_time := (cogs1 / cogs_per_hour1 : ℚ) + (cogs2 / cogs_per_hour2 : ℚ) + (cogs3 / cogs_per_hour3 : ℚ) in
  (total_cogs : ℚ) / total_time = 46.2857 :=
by
  intros
  unfold total_cogs total_time
  sorry

end overall_average_output_l519_519161


namespace lunch_cost_before_tax_and_tip_l519_519930

theorem lunch_cost_before_tax_and_tip (C : ℝ) (h1 : 1.10 * C = 110) : C = 100 := by
  sorry

end lunch_cost_before_tax_and_tip_l519_519930


namespace motorboat_time_to_C_l519_519504

variables (r s p t_B : ℝ)

-- Condition declarations
def kayak_speed := r + s
def motorboat_speed := p
def meeting_time := 12

-- Problem statement: to prove the time it took for the motorboat to reach dock C before turning back
theorem motorboat_time_to_C :
  (2 * r + s) * t_B = r * 12 + s * 6 → t_B = (r * 12 + s * 6) / (2 * r + s) := 
by
  intros h
  sorry

end motorboat_time_to_C_l519_519504


namespace particle_line_sphere_distance_l519_519308

noncomputable def particle_line_intersection_distance (P Q: ℝ × ℝ × ℝ) (sphere_center: ℝ × ℝ × ℝ) (sphere_radius: ℝ)
    (a b: ℕ) : ℕ :=
if h : coprime a b ∧ line_intersects_sphere P Q sphere_center sphere_radius ∧ intersection_distance_eq P Q sphere_center a b then a + b else 0

theorem particle_line_sphere_distance :
    ∀ (P Q : ℝ × ℝ × ℝ) (sphere_center : ℝ × ℝ × ℝ) (sphere_radius : ℝ) (a b : ℕ),
    coprime a b →
    line_intersects_sphere P Q sphere_center sphere_radius →
    intersection_distance_eq P Q sphere_center a b →
    particle_line_intersection_distance (1,2,2) (0,-2,-2) (0,0,0) 1 4 33 = 37 :=
by
    intros
    unfold particle_line_intersection_distance
    simp [coprime, line_intersects_sphere, intersection_distance_eq]
    sorry

end particle_line_sphere_distance_l519_519308


namespace part1_part2_l519_519633

def f (x : ℝ) (a : ℝ) := (Real.exp x) / x - Real.log x + x - a

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → (a ≤ Real.exp 1 + 1) :=
sorry

theorem part2 (x1 x2 a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 ≠ x2 → 0 < x1 → 0 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519633


namespace derangements_8_letters_l519_519783

/-- The number of derangements of 8 letters A, B, C, D, E, F, G, H where A, C, E, G do not appear in their original positions is 24024. -/
theorem derangements_8_letters : ∃ n : ℕ, n = 24024 ∧
  derangement_count 8 {0, 2, 4, 6} n :=
sorry

end derangements_8_letters_l519_519783


namespace selected_numbers_divisible_l519_519586

theorem selected_numbers_divisible (S : Finset ℕ) (hS : S ⊆ Finset.range 201) (h_card : S.card = 100) (h_lt_16 : ∃ x ∈ S, x < 16) :
  ∃ a b ∈ S, a ≠ b ∧ (a ∣ b ∨ b ∣ a) := by
  sorry

end selected_numbers_divisible_l519_519586


namespace length_of_XY_l519_519301

theorem length_of_XY
  (MN_parallel_XZ : ∀ (M N X Z Y : ℝ), Line.parallel (Line.mk M N) (Line.mk X Z))
  (XM : ℝ) (MY : ℝ) (NZ : ℝ)
  (hXM : XM = 6) (hMY : MY = 15) (hNZ : NZ = 9) :
  ∃ (XY : ℝ), XY = 21 :=
by
  -- Using the given conditions to set up the proof.
  sorry

end length_of_XY_l519_519301


namespace part1_part2_l519_519630

def f (x : ℝ) (a : ℝ) := (Real.exp x) / x - Real.log x + x - a

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → (a ≤ Real.exp 1 + 1) :=
sorry

theorem part2 (x1 x2 a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 ≠ x2 → 0 < x1 → 0 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519630


namespace part1_part2_l519_519657

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519657


namespace cost_per_tree_l519_519019

theorem cost_per_tree
    (initial_temperature : ℝ := 80)
    (final_temperature : ℝ := 78.2)
    (total_cost : ℝ := 108)
    (temperature_drop_per_tree : ℝ := 0.1) :
    total_cost / ((initial_temperature - final_temperature) / temperature_drop_per_tree) = 6 :=
by sorry

end cost_per_tree_l519_519019


namespace tan_alpha_eq_neg_12_div_5_l519_519685

theorem tan_alpha_eq_neg_12_div_5 
  (α : Real)
  (h1 : sin α + cos α = 7 / 13)
  (h2 : 0 < α ∧ α < π) :
  tan α = -12 / 5 := 
sorry

end tan_alpha_eq_neg_12_div_5_l519_519685


namespace total_cost_of_tickets_l519_519806

-- Conditions
def normal_price : ℝ := 50
def website_tickets_cost : ℝ := 2 * normal_price
def scalper_tickets_cost : ℝ := 2 * (2.4 * normal_price) - 10
def discounted_ticket_cost : ℝ := 0.6 * normal_price

-- Proof Statement
theorem total_cost_of_tickets :
  website_tickets_cost + scalper_tickets_cost + discounted_ticket_cost = 360 :=
by
  sorry

end total_cost_of_tickets_l519_519806


namespace product_469157_9999_l519_519980

theorem product_469157_9999 : 469157 * 9999 = 4690872843 := by
  -- computation and its proof would go here
  sorry

end product_469157_9999_l519_519980


namespace fraction_subtraction_l519_519173

theorem fraction_subtraction : (18 : ℚ) / 45 - (3 : ℚ) / 8 = (1 : ℚ) / 40 := by
  sorry

end fraction_subtraction_l519_519173


namespace new_car_covers_207_miles_l519_519121

-- Define the distance covered by the older car
def older_car_distance := 180

-- Define the speed increase factor for the new car
def speed_increase_factor := 0.15

-- Calculate the new car's distance
def new_car_distance := older_car_distance + (speed_increase_factor * older_car_distance)

-- State the theorem: the new car covers 207 miles
theorem new_car_covers_207_miles : new_car_distance = 207 :=
by
  sorry

end new_car_covers_207_miles_l519_519121


namespace probability_two_green_balls_l519_519104

theorem probability_two_green_balls:
  let total_balls := 3 + 5 + 4 in
  let total_ways := Nat.choose total_balls 3 in
  let ways_to_choose_green := Nat.choose 4 2 in
  let ways_to_choose_non_green := Nat.choose 8 1 in
  let favorable_ways := ways_to_choose_green * ways_to_choose_non_green in
  let probability := favorable_ways / total_ways in
  probability = 12 / 55 :=
by
  sorry

end probability_two_green_balls_l519_519104


namespace leap_years_count_2000_to_4100_l519_519122

def is_leap_year (n : ℕ) : Prop :=
  (n % 800 = 400) ∨ (n % 800 = 600)

def count_leap_years (a b : ℕ) : ℕ :=
  (List.range' a (b - a)).count is_leap_year

theorem leap_years_count_2000_to_4100 : 
  count_leap_years 2000 4100 = 5 :=
by
  sorry

end leap_years_count_2000_to_4100_l519_519122


namespace input_value_of_x_l519_519518

theorem input_value_of_x (x y : ℤ) (h₁ : (x < 0 → y = (x + 1) * (x + 1)) ∧ (¬(x < 0) → y = (x - 1) * (x - 1)))
  (h₂ : y = 16) : x = 5 ∨ x = -5 :=
sorry

end input_value_of_x_l519_519518


namespace simplify_fraction_l519_519007

-- Define what it means for a fraction to be in simplest form
def coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

-- Define what it means for a fraction to be reducible
def reducible_fraction (num den : ℕ) : Prop := ∃ d > 1, d ∣ num ∧ d ∣ den

-- Main theorem statement
theorem simplify_fraction 
  (m n : ℕ) (h_coprime : coprime m n) 
  (h_reducible : reducible_fraction (4 * m + 3 * n) (5 * m + 2 * n)) : ∃ d, d = 7 :=
by {
  sorry
}

end simplify_fraction_l519_519007


namespace angle_A_is_pi_div_3_area_ABC_is_3_sqrt_3_div_2_l519_519782

def triangle_ABC_conditions (a b : ℝ) (sin_A sin_B : ℝ) : Prop :=
  a = Real.sqrt 7 ∧ b = 3 ∧ (Real.sqrt 7) * sin_B + sin_A = 2 * Real.sqrt 3

theorem angle_A_is_pi_div_3 (sin_A : ℝ) : 
  ∀ (a b : ℝ) (sin_B : ℝ) (h : triangle_ABC_conditions a b sin_A sin_B), 
  ∠ A = π/3 :=
by
  intros a b sin_B h,
  cases h with ha hb,
  cases hb with hb hc,
  sorry

theorem area_ABC_is_3_sqrt_3_div_2 (sin_A : ℝ) :
  ∀ (a b c : ℝ) (sin_B : ℝ) (h : triangle_ABC_conditions a b sin_A sin_B) 
    (h_ac : a^2 = b^2 + c^2 - 2 * b * c * Real.cos (π / 3) ∧ 0 < c), 
  Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) / 4 = 3 * Real.sqrt 3 / 2 :=
by
  intros a b c sin_B h h_ac,
  cases h with ha hb,
  cases hb with hb hc,
  sorry

end angle_A_is_pi_div_3_area_ABC_is_3_sqrt_3_div_2_l519_519782


namespace percentage_increase_area_rectangle_l519_519092

theorem percentage_increase_area_rectangle (L W : ℝ) :
  let new_length := 1.20 * L
  let new_width := 1.20 * W
  let original_area := L * W
  let new_area := new_length * new_width
  let percentage_increase := ((new_area - original_area) / original_area) * 100
  percentage_increase = 44 := by
  sorry

end percentage_increase_area_rectangle_l519_519092


namespace find_difference_pq_l519_519037

theorem find_difference_pq (p q : ℕ → ℕ) 
  (h_const : ∃ k : ℕ, ∀ n, n > 1 → p(n) = k * q(n)) 
  (h_equiv : ∀ n, n > 1 → 16^n + 4^n + 1 = (2^(p n) - 1) / (2^(q n) - 1)) :
  p(2006) - q(2006) = 8024 := 
  sorry

end find_difference_pq_l519_519037


namespace squares_equality_l519_519202

theorem squares_equality (n : ℕ) (h : n ∈ ({0, 4} : Finset ℕ)) : n + n + n + n = 4 * n :=
by
  cases h
  · rw [h]
    simp
  · rw [h]
    simp
  sorry

end squares_equality_l519_519202


namespace parallel_lines_slope_condition_l519_519921

-- Define the first line equation and the slope
def line1 (x : ℝ) : ℝ := 6 * x + 5
def slope1 : ℝ := 6

-- Define the second line equation and the slope
def line2 (x c : ℝ) : ℝ := (3 * c) * x - 7
def slope2 (c : ℝ) : ℝ := 3 * c

-- Theorem stating that if the lines are parallel, the value of c is 2
theorem parallel_lines_slope_condition (c : ℝ) : 
  (slope1 = slope2 c) → c = 2 := 
  by
    sorry -- Proof

end parallel_lines_slope_condition_l519_519921


namespace inhabitant_knows_at_least_810_l519_519744

open Function

def population : ℕ := 1000000

def believers_claus (p : ℕ) : Prop := p = (90 * population / 100)

def knows_someone (p : ℕ) (knows : p → p → Prop) : Prop := 
  ∀ x : p, ∃ y : p, knows x y

def acquaintances_claus (p : ℕ) (believes : p → Prop) (knows : p → p → Prop) : Prop :=
  ∀ x : p, (believes x →(∃ k : ℕ, ∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y ∧ (10 * k / 100) y ∧ ¬believes y))

theorem inhabitant_knows_at_least_810 :
  ∃ (p : ℕ) (believes : p → Prop) (knows : p → p → Prop),
    population = 1000000 ∧
    believers_claus p ∧
    knows_someone p knows ∧
    acquaintances_claus p believes knows →
    ∃ x : p, ∃ k : ℕ, k ≥ 810 ∧ (∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y) :=
sorry

end inhabitant_knows_at_least_810_l519_519744


namespace sum_odd_even_50_l519_519559

def sum_first_n_odd (n : ℕ) : ℕ := n * n

def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

theorem sum_odd_even_50 : 
  sum_first_n_odd 50 + sum_first_n_even 50 = 5050 := by
  sorry

end sum_odd_even_50_l519_519559


namespace inverse_variation_proof_l519_519404

variable (x w : ℝ)

-- Given conditions
def varies_inversely (k : ℝ) : Prop :=
  x^4 * w^(1/4) = k

-- Specific instances
def specific_instance1 : Prop :=
  varies_inversely x w 162 ∧ x = 3 ∧ w = 16

def specific_instance2 : Prop :=
  varies_inversely x w 162 ∧ x = 6 → w = 1/4096

theorem inverse_variation_proof : 
  specific_instance1 → specific_instance2 :=
sorry

end inverse_variation_proof_l519_519404


namespace solve_math_problem_l519_519833

open Real

noncomputable def math_problem (g : ℝ → ℝ) (h : ∀ x y : ℝ, g ((x - y)^2) = g x * g y - x * y) : Prop :=
  let m := {x : ℝ | ∃ y : ℝ, g y = x}.to_finset.card
  let t := {x : ℝ | ∃ y : ℝ, g y = x}.to_finset.sum id
  m * t = 0

theorem solve_math_problem :
  ∀ (g: ℝ → ℝ),
  (∀ x y : ℝ, g ((x - y) ^ 2) = g x * g y - x * y) →
  math_problem g (λ x y, g ((x - y) ^ 2) = g x * g y - x * y) :=
by
  intro g h
  sorry

end solve_math_problem_l519_519833


namespace sum_of_segments_eq_twice_side_length_l519_519311

universe u
variables {α : Type u}

-- Define the equilateral triangle and internal point properties
structure Triangle (α : Type*) :=
  (A B C : α) 
  (side_length : ℝ)
  (equilateral : ∀ (P : α), dist A B = side_length ∧
                            dist B C = side_length ∧
                            dist C A = side_length)
                            
-- Define lines parallel through an internal point
structure ParallelSegs (α : Type*) :=
  (D E F G H I P : α)
  (PD_AB : ℝ)
  (PE_AB : ℝ)
  (PF_BC : ℝ)
  (PG_BC : ℝ)
  (PH_CA : ℝ)
  (PI_CA : ℝ)

-- Declare the theorem to prove sum of segments is twice the side length
theorem sum_of_segments_eq_twice_side_length 
  (T : Triangle α) (P : α) 
  (S : ParallelSegs α)
  (h1 : ∀ (k : α), T.equilateral k)
  (h2 : ¬(P = T.A ∨ P = T.B ∨ P = T.C)) 
  : S.PD_AB + S.PE_AB + S.PF_BC + S.PG_BC + S.PH_CA + S.PI_CA = 2 * T.side_length := 
begin
  sorry
end

end sum_of_segments_eq_twice_side_length_l519_519311


namespace exists_inhabitant_with_810_acquaintances_l519_519753

-- Conditions
def total_population : ℕ := 1000000
def believers_percentage : ℕ := 90
def acquaintances_believers_percentage : ℝ := 0.10

-- Theorem statement: Prove that there exists an inhabitant knowing at least 810 people.
theorem exists_inhabitant_with_810_acquaintances 
    (H1 : ∀ (p : ℕ), p < total_population → ∃ q, q ≠ p)  -- Every person knows at least one other person
    (H2 : 0.9 * total_population = 900000)  -- 90% of the population are believers in Santa Claus
    (H3 : ∀ p, p < total_population → ∀ q, q != p → (belief p → (exists percentage, percentage = acquaintances_believers_percentage))) -- Every person claims exactly 10% of their acquaintances believe in Santa Claus
    : ∃ (habitants : ℕ), habitants ≥ 810
    :=
  sorry

end exists_inhabitant_with_810_acquaintances_l519_519753


namespace corn_increase_factor_l519_519501

noncomputable def field_area : ℝ := 1

-- Let x be the remaining part of the field
variable (x : ℝ)

-- First condition: if the remaining part is fully planted with millet
-- Millet will occupy half of the field
axiom condition1 : (field_area - x) + x = field_area / 2

-- Second condition: if the remaining part x is equally divided between oats and corn
-- Oats will occupy half of the field
axiom condition2 : (field_area - x) + 0.5 * x = field_area / 2

-- Prove the factor by which the amount of corn increases
theorem corn_increase_factor : (0.5 * x + x) / (0.5 * x / 2) = 3 :=
by
  sorry

end corn_increase_factor_l519_519501


namespace point_traces_sphere_l519_519433

-- Definitions of the geometric objects and conditions
variable (A B C D : Point)
variable (AB_eq_CD : distance A B = distance C D)
variable (AB_fixed : fixed A B)
variable (ABCD_rhombus : is_rhombus A B C D)

-- Definition of points P and O
def P := Point_on_line_segment (CD) (1/3)
def O := Point_on_line_segment (AB) (1/3)

-- The theorem to be proven
theorem point_traces_sphere : 
  ∀ P C D, 
  P ∈ segment C D ∧ (distance C P = distance D P) ∧ (distance C D = distance A B) ∧ (is_rhombus A B C D) →
  ∃ O r, 
  is_center_radius_sphere O O P r ∧ 
  r = distance A B ∧ 
  ¬(P ∈ line AB) :=
by sorry

end point_traces_sphere_l519_519433


namespace adjacent_number_in_grid_l519_519095

def adjacent_triangle_number (k n: ℕ) : ℕ :=
  if k % 2 = 1 then n - k else n + k

theorem adjacent_number_in_grid (n : ℕ) (bound: n = 350) :
  let k := Nat.ceil (Real.sqrt n)
  let m := (k * k) - n
  k = 19 ∧ m = 19 →
  adjacent_triangle_number k n = 314 :=
by
  sorry

end adjacent_number_in_grid_l519_519095


namespace rachel_plant_placement_l519_519385

def num_ways_to_place_plants : ℕ :=
  let plants := ["basil", "basil", "aloe", "cactus"]
  let lamps := ["white", "white", "red", "red"]
  -- we need to compute the number of ways to place 4 plants under 4 lamps
  22

theorem rachel_plant_placement :
  num_ways_to_place_plants = 22 :=
by
  -- Proof omitted for brevity
  sorry

end rachel_plant_placement_l519_519385


namespace a_13_eq_30_l519_519252

variable (a : ℕ → ℕ)
variable (d : ℕ)

-- Define arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom a_5_eq_6 : a 5 = 6
axiom a_8_eq_15 : a 8 = 15

-- Required proof
theorem a_13_eq_30 (h : arithmetic_sequence a d) : a 13 = 30 :=
  sorry

end a_13_eq_30_l519_519252


namespace coefficients_l519_519572

-- All the conditions provided in the problem
def expr := (1 + X^5 + X^7) ^ 20

-- Main theorem stating the questions and answers
theorem coefficients:
  (coeff expr 17 = 190) ∧ (coeff expr 18 = 0) :=
sorry

end coefficients_l519_519572


namespace number_of_solutions_at_most_n_l519_519836

open Classical

noncomputable def P (n : ℕ) (k : ℕ) : 𝕍 → 𝕍 := sorry -- n > 1, Integer coefficients constraint

def Q (P : 𝕍 → 𝕍) (k : ℕ) : 𝕍 → 𝕍 :=
  λ x, (Nat.iterate P k x)

theorem number_of_solutions_at_most_n (P : ℕ → ℤ → ℤ) {n k : ℕ} (hn : n > 1) (hk : k ≠ 0) :
  ∃ (Q : ℤ → ℤ), Q = λ x => Nat.iterate P k x ∧ (∀ x : ℤ, Q x = x → x_nat_of_solutions Q ≤ n) :=
begin
  sorry
end

end number_of_solutions_at_most_n_l519_519836


namespace minimum_chord_length_l519_519885

noncomputable def point := (ℝ, ℝ)

noncomputable def line (k : ℝ) := {p : point | k * p.1 - p.2 - 4 * k + 3 = 0}

noncomputable def circle := {p : point | (p.1 - 3)^2 + (p.2 - 4)^2 = 4}

theorem minimum_chord_length (k : ℝ) : 
  ∃ A B : point, A ∈ line k ∧ B ∈ line k ∧ A ≠ B ∧ A ∈ circle ∧ B ∈ circle ∧ 
  (∀ C D : point, C ∈ line k ∧ D ∈ line k ∧ C ≠ D ∧ C ∈ circle ∧ D ∈ circle → 
  dist A B ≤ dist C D) → dist A B = 2 * ℝ.sqrt 2 :=
sorry

end minimum_chord_length_l519_519885


namespace problem1_problem2_l519_519246

-- Define the conditions and what needs to be proven
theorem problem1 (m : ℕ) (h1 : ∃ k : ℤ, 3 * m - 9 = 2 * k) (h2 : 3 * m - 9 < 0) : m = 2 :=
sorry

theorem problem2 (a : ℝ) (h : (2 / 3 < a) ∧ (a < 4)) : (a + 1) ^ (- (2 / 3)) < (3 - 2 * a) ^ (- (2 / 3)) :=
sorry

end problem1_problem2_l519_519246


namespace circle_area_proof_l519_519975

noncomputable def circle_area_in_square (sq_area : ℝ) (circle_touches_square_sides : Bool) : ℝ :=
if circle_touches_square_sides then 
  let side := Real.sqrt sq_area in
  let radius := side / 2 in
  Real.pi * radius^2
else
  0

theorem circle_area_proof : circle_area_in_square 400 true = 100 * Real.pi :=
sorry

end circle_area_proof_l519_519975


namespace exists_inhabitant_with_810_acquaintances_l519_519764

-- Definition of the main problem conditions
def inhabitants := 1000000
def belief_in_santa := 0.9 * inhabitants
def disbelief_in_santa := inhabitants - belief_in_santa

-- Conditions
def knows_at_least_one : ∀ x : ℕ, x < inhabitants → ∃ y : ℕ, y < inhabitants ∧ x ≠ y := sorry
def ten_percent_acquaintances (x : ℕ) (h : x < inhabitants) : Prop := ∀ y : ℕ, y < inhabitants ∧ x ≠ y → (y < belief_in_santa → y < 0.1 * x) := sorry

-- Main theorem
theorem exists_inhabitant_with_810_acquaintances :
  ∃ x : ℕ, x < inhabitants ∧ (knows_at_least_one x) ∧ (ten_percent_acquaintances x) ∧ (810 ≤ x) :=
sorry

end exists_inhabitant_with_810_acquaintances_l519_519764


namespace min_value_of_quadratic_l519_519193

theorem min_value_of_quadratic (x : ℝ) : ∃ z : ℝ, z = 2 * x^2 + 16 * x + 40 ∧ z = 8 :=
by {
  sorry
}

end min_value_of_quadratic_l519_519193


namespace alice_tom_same_heads_p_plus_q_l519_519158

def fair_coin := 1/2
def biased_coin_heads := 2/5

def combined_generating_function :=
  (1 + fair_coin*x)^2 * (2 + 3*x)

def probability_same_heads : ℚ :=
  let numerator := 2^2 + 11^2 + 12^2 + 3^2 in
  let denominator := (2 + 11 + 12 + 3)^2 in
  numerator / denominator

theorem alice_tom_same_heads : probability_same_heads = 139 / 392 :=
  by {
    sorry
  }

def p := 139
def q := 392

theorem p_plus_q : p + q = 531 :=
  by {
    rw [p, q],
    exact rfl
  }

end alice_tom_same_heads_p_plus_q_l519_519158


namespace max_distinct_substrings_length_66_l519_519862

theorem max_distinct_substrings_length_66 (s : String) (h : s.length = 66) (h_A : ∀ c ∈ s, c = 'A' ∨ c = 'T' ∨ c = 'C' ∨ c = 'G') :
  ∑ i in Finset.range 66, min (4^i) (67 - i) = 2100 :=
  sorry

end max_distinct_substrings_length_66_l519_519862


namespace inequality_one_inequality_two_l519_519388

variable {a b r s : ℝ}

theorem inequality_one (h_a : 0 < a) (h_b : 0 < b) :
  a^2 * b ≤ 4 * ((a + b) / 3)^3 :=
sorry

theorem inequality_two (h_a : 0 < a) (h_b : 0 < b) (h_r : 0 < r) (h_s : 0 < s) 
  (h_eq : 1 / r + 1 / s = 1) : 
  (a^r / r) + (b^s / s) ≥ a * b :=
sorry

end inequality_one_inequality_two_l519_519388


namespace total_cats_in_center_l519_519972

def cats_training_center : ℕ := 45
def cats_can_fetch : ℕ := 25
def cats_can_meow : ℕ := 40
def cats_jump_and_fetch : ℕ := 15
def cats_fetch_and_meow : ℕ := 20
def cats_jump_and_meow : ℕ := 23
def cats_all_three : ℕ := 10
def cats_none : ℕ := 5

theorem total_cats_in_center :
  (cats_training_center - (cats_jump_and_fetch + cats_jump_and_meow - cats_all_three)) +
  (cats_all_three) +
  (cats_fetch_and_meow - cats_all_three) +
  (cats_jump_and_fetch - cats_all_three) +
  (cats_jump_and_meow - cats_all_three) +
  cats_none = 67 := by
  sorry

end total_cats_in_center_l519_519972


namespace minimum_value_of_AB_l519_519886

noncomputable def minimum_chord_length (P : Point ℝ) (C : Circle ℝ) : ℝ :=
  -- Function definition placeholder. We would normally compute based
  -- on given conditions.

def circle_center : Point ℝ := ⟨2, 3⟩
def circle_radius : ℝ := 3
def point_P : Point ℝ := ⟨1, 1⟩

theorem minimum_value_of_AB : minimum_chord_length point_P (mk_circle circle_center circle_radius) = 4 :=
sorry

end minimum_value_of_AB_l519_519886


namespace unique_lambda_for_real_roots_l519_519359

theorem unique_lambda_for_real_roots (n : ℕ) (h : n ≥ 4) 
  (α β : Fin n → ℝ)
  (hα : ∑ j in Finset.univ, (α j)^2 < 1)
  (hβ : ∑ j in Finset.univ, (β j)^2 < 1) : 
  (∀ λ : ℝ, (λ = 0) ↔ ∀ x : ℝ, ( (x^n + λ * (x^(n-1) + ... + x^3 + ((1/2) * (1 - ∑ j in Finset.univ, (α j) * (β j) )^2) * x^2 + (real.sqrt (1 - ∑ j in Finset.univ, (α j)^2) * real.sqrt (1 - ∑ j in Finset.univ, (β j)^2)) * x + 1)) = 0 → ∀ y : ℝ, is_root y) ) :=
by sorry

end unique_lambda_for_real_roots_l519_519359


namespace cone_geometry_l519_519147

noncomputable def cone_radius_from_volume (V : ℝ) (h : ℝ) : ℝ :=
  (3 * V / (π * h)).sqrt

noncomputable def cone_circumference (r : ℝ) : ℝ :=
  2 * π * r

noncomputable def cone_slant_height (r : ℝ) (h : ℝ) : ℝ :=
  (r^2 + h^2).sqrt

noncomputable def cone_lateral_surface_area (r : ℝ) (l : ℝ) : ℝ :=
  π * r * l

theorem cone_geometry (V h : ℝ) (V_eq : V = 16 * π) (h_eq : h = 6):
  let r := cone_radius_from_volume V h,
      c := cone_circumference r,
      l := cone_slant_height r h,
      A := cone_lateral_surface_area r l in
  c = 4 * √2 * π ∧ A = 4 * √22 * π := by
{
  let r := cone_radius_from_volume V h,
  let c := cone_circumference r,
  let l := cone_slant_height r h,
  let A := cone_lateral_surface_area r l,
  sorry
}

end cone_geometry_l519_519147


namespace gk_dk_eq_ae_ce_af_bf_l519_519794

variable (A B C D E F G J K : Point)
variable (triangleABC : Triangle A B C)
variable (incircle : Incircle triangleABC)
variable (BC : Segment B C)
variable (CA : Segment C A)
variable (AB : Segment A B)
variable (touchesD : incircle.Touches BC D)
variable (touchesE : incircle.Touches CA E)
variable (touchesF : incircle.Touches AB F)
variable (angleBisector : AngleBisector (∠ A B C) A)
variable (intersectsG : angleBisector.Intersects BC G)
variable (BE : Line B E)
variable (CF : Line C F)
variable (intersectsJ : BE.Intersects CF J)
variable (perpendicularJK : PerpendicularToLine (LineThrough J K) (LineThrough E F))
variable (intersectsK : PerpendicularIntersectAt J E F K BC)

theorem gk_dk_eq_ae_ce_af_bf : (GK / DK) = (AE / CE) + (AF / BF) := by
  sorry

end gk_dk_eq_ae_ce_af_bf_l519_519794


namespace probability_three_even_l519_519976

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the probability of exactly three dice showing an even number
noncomputable def prob_exactly_three_even (n : ℕ) (k : ℕ) (p : ℚ) : ℚ := 
  (binomial n k : ℚ) * (p^k) * ((1 - p)^(n - k))

-- The main theorem stating the desired probability
theorem probability_three_even (n : ℕ) (p : ℚ) (k : ℕ) (h₁ : n = 6) (h₂ : p = 1/2) (h₃ : k = 3) :
  prob_exactly_three_even n k p = 5 / 16 := by
  sorry

-- Include required definitions and expected values for the theorem
#check binomial
#check prob_exactly_three_even
#check probability_three_even

end probability_three_even_l519_519976


namespace shaded_region_area_l519_519893

noncomputable def area_of_shaded_region : ℝ :=
  let s := 12
  let diagonal := s * Real.sqrt 2
  let height := s
  1 / 2 * diagonal * height

theorem shaded_region_area (s : ℝ) (hs : s = 12) : area_of_shaded_region = 72 * Real.sqrt 2 := by
  rw [area_of_shaded_region, hs]
  norm_num
  exact Real.sqrt_mul_self_eq_abs (2:ℝ)

end shaded_region_area_l519_519893


namespace exists_inhabitant_with_810_acquaintances_l519_519754

-- Conditions
def total_population : ℕ := 1000000
def believers_percentage : ℕ := 90
def acquaintances_believers_percentage : ℝ := 0.10

-- Theorem statement: Prove that there exists an inhabitant knowing at least 810 people.
theorem exists_inhabitant_with_810_acquaintances 
    (H1 : ∀ (p : ℕ), p < total_population → ∃ q, q ≠ p)  -- Every person knows at least one other person
    (H2 : 0.9 * total_population = 900000)  -- 90% of the population are believers in Santa Claus
    (H3 : ∀ p, p < total_population → ∀ q, q != p → (belief p → (exists percentage, percentage = acquaintances_believers_percentage))) -- Every person claims exactly 10% of their acquaintances believe in Santa Claus
    : ∃ (habitants : ℕ), habitants ≥ 810
    :=
  sorry

end exists_inhabitant_with_810_acquaintances_l519_519754


namespace total_mail_l519_519338

def monday_mail := 65
def tuesday_mail := monday_mail + 10
def wednesday_mail := tuesday_mail - 5
def thursday_mail := wednesday_mail + 15

theorem total_mail : 
  monday_mail + tuesday_mail + wednesday_mail + thursday_mail = 295 := by
  sorry

end total_mail_l519_519338


namespace cubic_plus_six_square_twice_is_composite_l519_519378

theorem cubic_plus_six_square_twice_is_composite (n : ℕ) : ¬ prime (n^3 + 6*n^2 + 12*n + 16) := by
  sorry

end cubic_plus_six_square_twice_is_composite_l519_519378


namespace points_on_opposite_sides_of_line_l519_519262

theorem points_on_opposite_sides_of_line (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by 
  sorry

end points_on_opposite_sides_of_line_l519_519262


namespace remainder_of_t_50_l519_519188

def T : ℕ → ℕ
| 0       := 5
| (n + 1) := 3 ^ (T n)

theorem remainder_of_t_50 : (T 49) % 7 = 5 :=
by sorry

end remainder_of_t_50_l519_519188


namespace rhombus_longer_diagonal_l519_519134

theorem rhombus_longer_diagonal (a b d : ℝ) (h : a = 65) (h_d : d = 56) :
  ∃ l, l = 118 :=
by
  sorry

end rhombus_longer_diagonal_l519_519134


namespace quadratic_root_unique_l519_519428

theorem quadratic_root_unique 
  (a b c : ℝ)
  (hf1 : b^2 - 4 * a * c = 0)
  (hf2 : (b - 30 * a)^2 - 4 * a * (17 * a - 7 * b + c) = 0)
  (ha_pos : a ≠ 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = -11 := 
by
  sorry

end quadratic_root_unique_l519_519428


namespace problem1_problem2_l519_519986

-- Define the variables
variables (x y : ℝ)

-- Problem 1
theorem problem1 : (x + y) * (x - 2y) + (x - y)^2 + 3 * x * 2 * y = 2 * x^2 + 3 * x * y - y^2 :=
by
  sorry

-- Define variable z to represent the complex divisor expression in Problem 2
def z (x : ℝ) := x + 1 - 3 / (x - 1)

-- Problem 2
theorem problem2 : (x^2 - 4 * x + 4) / (x^2 - x) / z(x) = (x - 2) / (x^2 + 2 * x) :=
by
  sorry

end problem1_problem2_l519_519986


namespace chessboard_domino_tiling_possible_l519_519943

-- Define the chessboard and the removal condition
noncomputable def canCoverWithDominoes (board : List (List Bool)) (p1 p2 : (ℕ × ℕ)) : Prop :=
  let squaresCount := (board.length * board.head.length) - 2
  let colorsMatch := (board[p1.1][p1.2] != board[p2.1][p2.2])
  ∀ tiling : List ((ℕ × ℕ) × (ℕ × ℕ)), 
  colorsMatch ∧ 
  (∑ t in tiling, (t.1, t.2) contains p1 = false ∧ (t.1, t.2) contains p2 = false) -> 
  (∑ t in tiling, (t.1.fst + t.2.fst) % 2 = 0 ∧ (t.1.snd + t.2.snd) % 2 = 0)

theorem chessboard_domino_tiling_possible 
(board : List (List Bool)) (p1 p2 : (ℕ × ℕ)) 
(h1 : List.length board = 8) (h2 : List.length board.head = 8) (h3 : board[p1.1][p1.2] != board[p2.1][p2.2]) : 
  canCoverWithDominoes board p1 p2 :=
by
  sorry

end chessboard_domino_tiling_possible_l519_519943


namespace maria_profit_disks_l519_519849

-- Define the conditions as constants
def purchase_rate : ℝ := 10 / 6
def selling_rate : ℝ := 10 / 5

-- Define the profit per disk calculation based on the given conditions
def profit_per_disk : ℝ := selling_rate - purchase_rate

-- Define the total profit goal
def profit_goal : ℝ := 200

-- Calculate the number of disks Maria needs to sell to achieve the profit goal
def disks_to_sell : ℝ := profit_goal / profit_per_disk

theorem maria_profit_disks :
  round (disks_to_sell) = 607 := 
by
  -- Proof omitted
  sorry

end maria_profit_disks_l519_519849


namespace incorrect_operation_l519_519924

variable (a : ℕ)

-- Conditions
def condition1 := 4 * a ^ 2 - a ^ 2 = 3 * a ^ 2
def condition2 := a ^ 3 * a ^ 6 = a ^ 9
def condition3 := (a ^ 2) ^ 3 = a ^ 5
def condition4 := (2 * a ^ 2) ^ 2 = 4 * a ^ 4

-- Theorem to prove
theorem incorrect_operation : (a ^ 2) ^ 3 ≠ a ^ 5 := 
by
  sorry

end incorrect_operation_l519_519924


namespace triangles_from_points_l519_519064

theorem triangles_from_points (n : ℕ) (points : Finset (ℝ × ℝ)) (h : points.card = 3 * n)
  (h_no_collinear : ∀ (p1 p2 p3 : (ℝ × ℝ)), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
         ¬ collinear ({p1, p2, p3} : Finset (ℝ × ℝ))) : ∃ (tris : Finset (Finset (ℝ × ℝ))), 
      (tris.card = n) ∧ 
      (∀ t ∈ tris, t.card = 3) ∧ 
      (∀ t1 t2 ∈ tris, t1 ≠ t2 → (t1 ∩ t2 = ∅)) := 
sorry

end triangles_from_points_l519_519064


namespace at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519717

theorem at_least_one_inhabitant_knows_no_fewer_than_810_people :
  ∀ (n : ℕ),
  (inhabitants : ℕ) (knows_someone : inhabitants > 1) (believes_santa : inhabitants * 9 / 10)
  (ten_percent_acquaintances_believe : ∀ x, (acquaintances : list x) → (|list.contacts_with_santa <- 0.10 * acquaintances|)),
  (h : inhabitants = 1000000),
  (one_inhabitant_knows : ∃ x, ∀ y, y \in acquaintances x → friends y ≥ 810) :=
sorry

end at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519717


namespace symmetric_point_l519_519580

theorem symmetric_point :
  let line := λ x : ℝ, 2 * x
  let P := (-4 : ℝ, 2 : ℝ)
  ∃ Q : ℝ × ℝ, Q = (4, -2) ∧ 
  (P.2 - Q.2) / (P.1 - Q.1) = -1 / 2 ∧
  (Q.2 + P.2) / 2 = line ((Q.1 + P.1) / 2) :=
by
  sorry

end symmetric_point_l519_519580


namespace rhombus_longer_diagonal_l519_519145

theorem rhombus_longer_diagonal
  (a b : ℝ)
  (ha : a = 65) -- Side length
  (hb : b = 28) -- Half the length of the shorter diagonal
  : 2 * Real.sqrt (a^2 - b^2) = 118 :=  -- Statement of the longer diagonal length
by
  sorry -- Proof goes here

end rhombus_longer_diagonal_l519_519145


namespace part1_part2_l519_519639

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l519_519639


namespace regression_equation_correct_l519_519899

-- Defining the given data as constants
def x_data : List ℕ := [1, 2, 3, 4, 5, 6, 7]
def y_data : List ℕ := [891, 888, 351, 220, 200, 138, 112]

def sum_t_y : ℚ := 1586
def avg_t : ℚ := 0.37
def sum_t2_min7_avg_t2 : ℚ := 0.55

-- Defining the target regression equation
def target_regression (x : ℚ) : ℚ := 1000 / x + 30

-- Function to calculate the regression equation from data
noncomputable def calculate_regression (x_data y_data : List ℕ) : (ℚ → ℚ) :=
  let n : ℚ := x_data.length
  let avg_y : ℚ := y_data.sum / n
  let b : ℚ := (sum_t_y - n * avg_t * avg_y) / (sum_t2_min7_avg_t2)
  let a : ℚ := avg_y - b * avg_t
  fun x : ℚ => a + b / x

-- Theorem stating the regression equation matches the target regression equation
theorem regression_equation_correct :
  calculate_regression x_data y_data = target_regression :=
by
  sorry

end regression_equation_correct_l519_519899


namespace mean_median_mode_relation_l519_519071

-- Defining the data set of the number of fish caught in twelve outings.
def fish_catches : List ℕ := [3, 0, 2, 2, 1, 5, 3, 0, 1, 4, 3, 3]

-- Proof statement to show the relationship among mean, median and mode.
theorem mean_median_mode_relation (hs : fish_catches = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5]) :
  let mean := (fish_catches.sum : ℚ) / fish_catches.length
  let median := (fish_catches.nthLe 5 sorry + fish_catches.nthLe 6 sorry : ℚ) / 2
  let mode := 3
  mean < median ∧ median < mode := by
  -- Placeholder for the proof. Details are skipped here.
  sorry

end mean_median_mode_relation_l519_519071


namespace product_of_valid_c_l519_519211

theorem product_of_valid_c : 
  (∏ (c : ℕ) in finset.range 9, c) = 40320 :=
by
  rw finset.range
  sorry

end product_of_valid_c_l519_519211


namespace relationship_between_y_l519_519694

theorem relationship_between_y (y1 y2 y3 : ℝ)
  (hA : y1 = 3 / -5)
  (hB : y2 = 3 / -3)
  (hC : y3 = 3 / 2) : y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_l519_519694


namespace fishermen_catch_l519_519460

variables (x y : ℕ)

theorem fishermen_catch :
  (x = y / 2 + 10) → (y = x + 20) → (x + y = 100) :=
begin
  sorry
end

end fishermen_catch_l519_519460


namespace triple_sum_of_45_point_2_and_one_fourth_l519_519914

theorem triple_sum_of_45_point_2_and_one_fourth : 
  (3 * (45.2 + 0.25)) = 136.35 :=
by
  sorry

end triple_sum_of_45_point_2_and_one_fourth_l519_519914


namespace exists_inhabitant_with_many_acquaintances_l519_519758

open Function

theorem exists_inhabitant_with_many_acquaintances :
  (∃ (n : ℕ), n = 1_000_000) →
  (∀ (p : ℕ), p < 1_000_000 → ∃ (q : ℕ), q ≠ p) →
  (∀ (p : ℕ), p < 900_000 → p <= 900_000) →
  (∀ (p : ℕ), p < 1_000_000 → 0.1 * real.to_real p < 0.9 * real.to_real p) →
  ∃ (p : ℕ), p >= 810 :=
by
  intros h1 h2 h3 h4
  sorry

end exists_inhabitant_with_many_acquaintances_l519_519758


namespace no_prime_divisible_by_77_l519_519277

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 :
  ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  intros p h_prime h_div
  sorry

end no_prime_divisible_by_77_l519_519277


namespace incorrect_proposition_C_l519_519925

theorem incorrect_proposition_C (p q : Prop) :
  (¬(p ∨ q) → ¬p ∧ ¬q) ↔ False :=
by sorry

end incorrect_proposition_C_l519_519925


namespace total_watermelon_slices_l519_519547

theorem total_watermelon_slices 
(h1 : ∀ (n : Nat), n > 0 → n * 10 = 30)
(h2 : ∀ (m : Nat), m > 0 → m * 15 = 15) :
  3 * 10 + 1 * 15 = 45 := 
  by 
    have h_danny : 3 * 10 = 30 := h1 3 (by norm_num)
    have h_sister: 1 * 15 = 15 := h2 1 (by norm_num)
    calc
      3 * 10 + 1 * 15 = 30 + 15 := by rw [h_danny, h_sister]
                  ... = 45 := by norm_num

end total_watermelon_slices_l519_519547


namespace part1_part2_l519_519638

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l519_519638


namespace find_m_and_c_l519_519251

-- Definitions & conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 3 }
def B (m : ℝ) : Point := { x := -6, y := m }

def line (c : ℝ) (p : Point) : Prop := p.x + p.y + c = 0

-- Theorem statement
theorem find_m_and_c (m : ℝ) (c : ℝ) (hc : line c A) (hcB : line c (B m)) :
  m = 3 ∧ c = -2 :=
  by
  sorry

end find_m_and_c_l519_519251


namespace a_n_arithmetic_sequence_bn_general_formula_l519_519219

variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}

-- Given conditions
axiom a_pos (n : ℕ) (h : 0 < n) : 0 < a n
axiom Sn_def (n : ℕ) (h : 0 < n) : S n = (a n)^2 + a n / 2

-- Prove a_n is arithmetic: a_n = n
theorem a_n_arithmetic_sequence (n : ℕ) (h : 0 < n) : a n = n := sorry

-- Given sequence b_n with initial condition and recurrence relation
axiom b1 : b 1 = 2
axiom b_recurrence (n : ℕ) (h : 0 < n) : b (n + 1) = 2^(a n) + b n
axiom a_def (n : ℕ) (h : 0 < n) : a n = n

-- Prove general formula for b_n: b_n = 2^n
theorem bn_general_formula (n : ℕ) : b n = 2^n := sorry

end a_n_arithmetic_sequence_bn_general_formula_l519_519219


namespace no_real_roots_decreasing_on_interval_l519_519258

noncomputable def f (t x : ℝ) : ℝ := x * Real.exp(t * x) - Real.exp(x) + 1

theorem no_real_roots (t : ℝ) (h : t < 1 - 1 / Real.exp 1) : ∀ x : ℝ, f t x ≠ 1 := 
by
  sorry

theorem decreasing_on_interval (t : ℝ) (h : t ≤ 1 / 2) : ∀ {a b : ℝ}, 0 < a → a < b → f t a ≥ f t b := 
by
  sorry

end no_real_roots_decreasing_on_interval_l519_519258


namespace perfect_cube_divisor_count_l519_519681

noncomputable def num_perfect_cube_divisors : Nat :=
  let a_choices := Nat.succ (38 / 3)
  let b_choices := Nat.succ (17 / 3)
  let c_choices := Nat.succ (7 / 3)
  let d_choices := Nat.succ (4 / 3)
  a_choices * b_choices * c_choices * d_choices

theorem perfect_cube_divisor_count :
  num_perfect_cube_divisors = 468 :=
by
  sorry

end perfect_cube_divisor_count_l519_519681


namespace correct_calculation_l519_519923

theorem correct_calculation :
  (sqrt 2 * sqrt 3 = sqrt 6) ∧
  (sqrt 8 - sqrt 2 ≠ 2) ∧ 
  (sqrt 13 + sqrt 3 ≠ 4) ∧ 
  (sqrt 8 / sqrt 2 ≠ 4) :=
by {
  split,
  {
    -- sqrt(2) * sqrt(3) = sqrt(6)
    exact sqrt_mul (show 2 ≥ 0, by norm_num) (show 3 ≥ 0, by norm_num),
  },
  split,
  {
    -- sqrt(8) - sqrt(2) ≠ 2
    intro h,
    apply_fun (λ x, x * sqrt 2) at h,
    simp at h,
    norm_num at h,
    linarith,
  },
  split,
  {
    -- sqrt(13) + sqrt(3) ≠ 4
    intro h,
    apply_fun (λ x, x * h) at h,
    simp at h,
    norm_num at h,
    linarith,
  },
  {
    -- sqrt(8) / sqrt(2) ≠ 4
    intro h,
    apply_fun (λ x, x * sqrt 2) at h,
    simp at h,
    norm_num at h,
    linarith,
  }
}

end correct_calculation_l519_519923


namespace largest_C_pm1_sequence_l519_519103

-- Define the sequence constraint.
def is_pm1_sequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ n → a i = 1 ∨ a i = -1

-- Define the condition on indices.
def valid_indices (t : ℕ → ℕ) (k n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i < k → t (i + 1) - t i ≤ 2 ∧ 1 ≤ t i ∧ t i ≤ n

-- The main theorem stating the result of the proof problem.
theorem largest_C_pm1_sequence : ∃ (C : ℕ), 
  (∀ (a : ℕ → ℤ) (n : ℕ) , n = 2022 → is_pm1_sequence a n →
    ∃ (k : ℕ) (t : ℕ → ℕ), k > 0 ∧ valid_indices t k n ∧ 
    (|∑ i in finset.range k, a (t i)| ≥ C)) ∧ C = 506 :=
by
  sorry

end largest_C_pm1_sequence_l519_519103


namespace concurrency_of_ME_NF_BC_l519_519228

variable {α : Type*} [EuclideanSpace α]

-- Assume the given conditions
variables {A B C P D E F M N : α}

-- Making necessary assumptions on conditions
-- P is an interior point of the triangle ABC
variable (h_interior : P ∈ interior (triangle ABC))

-- D, E, F are the orthogonal projections of P on BC, CA, AB, respectively
variable (hD : orthogonal_projection (line B C) P = D)
variable (hE : orthogonal_projection (line C A) P = E)
variable (hF : orthogonal_projection (line A B) P = F)

-- M and N are the orthogonal projections of A on BP and CP, respectively
variable (hM : orthogonal_projection (line B P) A = M)
variable (hN : orthogonal_projection (line C P) A = N)

-- The proposition to prove
theorem concurrency_of_ME_NF_BC :
  Concurrent (line M E) (line N F) (line B C) :=
sorry

end concurrency_of_ME_NF_BC_l519_519228


namespace eqn_intersecting_straight_lines_l519_519879

theorem eqn_intersecting_straight_lines (x y : ℝ) : 
  x^2 - y^2 = 0 → (y = x ∨ y = -x) :=
by
  intros h
  sorry

end eqn_intersecting_straight_lines_l519_519879


namespace costs_equal_when_x_20_l519_519513

noncomputable def costA (x : ℕ) : ℤ := 150 * x + 3300
noncomputable def costB (x : ℕ) : ℤ := 210 * x + 2100

theorem costs_equal_when_x_20 : costA 20 = costB 20 :=
by
  -- Statements representing the costs equal condition
  have ha : costA 20 = 150 * 20 + 3300 := rfl
  have hb : costB 20 = 210 * 20 + 2100 := rfl
  rw [ha, hb]
  -- Simplification steps (represented here in Lean)
  sorry

end costs_equal_when_x_20_l519_519513


namespace distance_between_lines_l519_519673

open Real

theorem distance_between_lines : 
  ∀ (a b c1 c2 : ℝ), 
  a = 1 → b = 1 → c1 = -1 → c2 = 1 → 
  ∃ (d : ℝ), d = sqrt 2 := 
by 
  intros a b c1 c2 ha hb hc1 hc2 
  use sqrt 2 
  sorry

end distance_between_lines_l519_519673


namespace percentage_decrease_wages_l519_519810

variable (W : ℝ) -- Last week's wages
variable (P : ℝ) -- Percentage decrease in wages
variable (W' : ℝ) -- This week's wages after percentage decrease

-- Conditions translated to Lean.
def recreation_last_week := 0.40 * W
def recreation_this_week := 0.50 * W'
def wages_decrease := W - (P / 100) * W

-- Theorem statement
theorem percentage_decrease_wages : 
  recreation_this_week = 1.1875 * recreation_last_week →
  W' = wages_decrease →
  P = 5 := sorry

end percentage_decrease_wages_l519_519810


namespace inequality_proof_l519_519612

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  abc ≥ (a + b + c) / ((1 / a^2) + (1 / b^2) + (1 / c^2)) ∧
  (a + b + c) / ((1 / a^2) + (1 / b^2) + (1 / c^2)) ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by
  sorry

end inequality_proof_l519_519612


namespace larger_screen_diagonal_length_l519_519895

theorem larger_screen_diagonal_length :
  (∃ d : ℝ, (∀ a : ℝ, a = 16 → d^2 = 2 * (a^2 + 34)) ∧ d = Real.sqrt 580) :=
by
  sorry

end larger_screen_diagonal_length_l519_519895


namespace rhombus_longer_diagonal_l519_519142

theorem rhombus_longer_diagonal (a b : ℕ) (d1 : ℕ) (d2 : ℕ) (h0 : a = 65) (h1 : d1 = 56) (h2 : a * a = (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2)) :
  d2 = 118 :=
by
  have h3 : (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2) = 65 * 65, from h2,
  rw [←h0, ←h1] at h3,
  have h4 : (28 : ℕ) = d1 / 2, sorry,
  have h5 : (28 : ℕ) * (28 : ℕ) + ((d2 / 2) * (d2 / 2)) = 65 * 65, from h3,
  have h6 : (28 * 28) + (d2 / 2) * (d2 / 2) = (65*65), by rw h5,
  have h7 : (65 * 65) - 784 = (d2 / 2) * (d2 / 2), sorry,
  have h8 : (d2 / 2) * (d2 / 2) = 3441, by simp[h7],
  have h9 : (d2 / 2) = 59, from nat.sqrt_eq 3441,
  have h10 : d2 = 2 * 59, by rw [h9, two_mul],
  exact h10
 

end rhombus_longer_diagonal_l519_519142


namespace correct_answer_l519_519447

noncomputable def original_number (y : ℝ) :=
  (y - 14) / 2 = 50

theorem correct_answer (y : ℝ) (h : original_number y) :
  (y - 5) / 7 = 15 :=
by
  sorry

end correct_answer_l519_519447


namespace inverse_variation_l519_519405

theorem inverse_variation (w : ℝ) (h1 : ∃ (c : ℝ), ∀ (x : ℝ), x^4 * w^(1/4) = c)
  (h2 : (3 : ℝ)^4 * (16 : ℝ)^(1/4) = (6 : ℝ)^4 * w^(1/4)) : 
  w = 1 / 4096 :=
by
  sorry

end inverse_variation_l519_519405


namespace inhabitant_knows_at_least_810_l519_519742

open Function

def population : ℕ := 1000000

def believers_claus (p : ℕ) : Prop := p = (90 * population / 100)

def knows_someone (p : ℕ) (knows : p → p → Prop) : Prop := 
  ∀ x : p, ∃ y : p, knows x y

def acquaintances_claus (p : ℕ) (believes : p → Prop) (knows : p → p → Prop) : Prop :=
  ∀ x : p, (believes x →(∃ k : ℕ, ∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y ∧ (10 * k / 100) y ∧ ¬believes y))

theorem inhabitant_knows_at_least_810 :
  ∃ (p : ℕ) (believes : p → Prop) (knows : p → p → Prop),
    population = 1000000 ∧
    believers_claus p ∧
    knows_someone p knows ∧
    acquaintances_claus p believes knows →
    ∃ x : p, ∃ k : ℕ, k ≥ 810 ∧ (∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y) :=
sorry

end inhabitant_knows_at_least_810_l519_519742


namespace probability_passing_through_C_l519_519112

noncomputable def number_of_paths (east_moves south_moves : ℕ) : ℕ :=
nat.choose (east_moves + south_moves) east_moves

noncomputable def probability_passing_C : ℚ :=
let paths_A_to_C := number_of_paths 3 1 in
let paths_C_to_B := number_of_paths 1 3 in
let total_paths_A_to_B := number_of_paths 4 4 in
(paths_A_to_C * paths_C_to_B : ℚ) / total_paths_A_to_B

theorem probability_passing_through_C :
  probability_passing_C = 8 / 35 :=
by sorry

end probability_passing_through_C_l519_519112


namespace range_of_a_plus_b_l519_519832

theorem range_of_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b)
    (h3 : |2 - a^2| = |2 - b^2|) : 2 < a + b ∧ a + b < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_a_plus_b_l519_519832


namespace evaluate_sum_l519_519566

theorem evaluate_sum : 
  (∑ x in (finset.range 43).filter (λ x : ℕ, x ≥ 3), 2 * real.cos x * real.cos 1 * (1 + (real.csc (x - 1)) * (real.csc (x + 1)))) = 46 := by
sorry

end evaluate_sum_l519_519566


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519738

theorem exists_inhabitant_with_at_least_810_acquaintances
  (N : ℕ) (hN : N = 1000000)
  (B : Finset ℕ) (hB : B.card = 900000)
  (A : ℕ → Finset ℕ)
  (hA_nonempty : ∀ i, (A i).nonempty)
  (hA_believers : ∀ i, (A i).card / 10 = (A i ∩ B).card) :
  ∃ j, (A j).card ≥ 810 :=
by
  sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519738


namespace no_prime_divisible_by_77_l519_519276

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 :
  ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  intros p h_prime h_div
  sorry

end no_prime_divisible_by_77_l519_519276


namespace find_FC_l519_519588

theorem find_FC 
  (DC CB AD AB ED FC : ℝ)
  (h1 : DC = 10)
  (h2 : CB = 12)
  (h3 : AB = (1/5) * AD)
  (h4 : ED = (2/3) * AD)
  (h5 : AD = (5/4) * 22)  -- Derived step from solution for full transparency
  (h6 : FC = (ED * (CB + AB)) / AD) : 
  FC = 35 / 3 := 
sorry

end find_FC_l519_519588


namespace largest_house_number_l519_519387

theorem largest_house_number (house_num : ℕ) : 
  house_num ≤ 981 :=
  sorry

end largest_house_number_l519_519387


namespace find_final_painting_width_l519_519468

theorem find_final_painting_width
  (total_area : ℕ)
  (painting_areas : List ℕ)
  (total_paintings : ℕ)
  (last_painting_height : ℕ)
  (last_painting_width : ℕ) :
  total_area = 200
  → painting_areas = [25, 25, 25, 80]
  → total_paintings = 5
  → last_painting_height = 5
  → last_painting_width = 9 :=
by
  intros h_total_area h_painting_areas h_total_paintings h_last_height
  have h1 : 25 * 3 + 80 = 155 := by norm_num
  have h2 : total_area - 155 = last_painting_width * last_painting_height := by
    rw [h_total_area, show 155 = 25 * 3 + 80 by norm_num]
    norm_num
  exact eq_of_mul_eq_mul_right (by norm_num) h2

#print axioms find_final_painting_width -- this should ensure we don't leave any implicit assumptions. 

end find_final_painting_width_l519_519468


namespace smaller_angle_at_7_15_is_127_5_degrees_l519_519679

-- Defining the conditions
def angle_per_hour : ℝ := 30
def angle_per_minute : ℝ := 6
def hour_hand_angle_at_7_15 : ℝ := 7 * angle_per_hour + angle_per_hour / 4
def minute_hand_angle_at_7_15 : ℝ := 15 * angle_per_minute

-- Define the angle between the hands
def angle_between_hands (hour_hand_angle minute_hand_angle : ℝ) : ℝ :=
  let diff := abs (hour_hand_angle - minute_hand_angle)
  min diff (360 - diff)

-- Prove the main statement
theorem smaller_angle_at_7_15_is_127_5_degrees :
  angle_between_hands hour_hand_angle_at_7_15 minute_hand_angle_at_7_15 = 127.5 :=
by sorry

end smaller_angle_at_7_15_is_127_5_degrees_l519_519679


namespace rhombus_longer_diagonal_l519_519133

theorem rhombus_longer_diagonal (a b d : ℝ) (h : a = 65) (h_d : d = 56) :
  ∃ l, l = 118 :=
by
  sorry

end rhombus_longer_diagonal_l519_519133


namespace tanner_money_left_l519_519408

def tanner_september_savings := 17
def tanner_october_savings := 48
def tanner_november_savings := 25
def tanner_december_savings := 55
def total_savings := tanner_september_savings + tanner_october_savings + tanner_november_savings + tanner_december_savings

def video_game_original_cost := 49
def video_game_discount := 0.10 * video_game_original_cost
def video_game_final_cost := video_game_original_cost - video_game_discount

def new_shoes_cost := 65
def total_cost_before_tax := video_game_final_cost + new_shoes_cost

def sales_tax := 0.05 * total_cost_before_tax
def rounded_sales_tax := Float.ceil (sales_tax * 100) / 100  -- Assuming the use of ceil to simulate rounding to the nearest cent

def total_cost_with_tax := total_cost_before_tax + rounded_sales_tax

def money_left := total_savings - total_cost_with_tax

theorem tanner_money_left : money_left = 30.44 := by
  have h1 : total_savings = 145 := by
    simp [total_savings, tanner_september_savings, tanner_october_savings, tanner_november_savings, tanner_december_savings]
  have h2 : video_game_final_cost = 44.10 := by
    simp [video_game_final_cost, video_game_original_cost, video_game_discount]
  have h3 : total_cost_before_tax = 109.10 := by
    simp [total_cost_before_tax, video_game_final_cost, new_shoes_cost]
  have h4 : rounded_sales_tax = 5.46 := by
    simp [rounded_sales_tax, sales_tax, total_cost_before_tax]
  have h5 : total_cost_with_tax = 114.56 := by
    simp [total_cost_with_tax, total_cost_before_tax, rounded_sales_tax]
  simp [money_left, h1, h5]
  norm_num
  sorry  -- Replace sorry with detailed steps if doing a formal proof

end tanner_money_left_l519_519408


namespace Johnson_farm_budget_l519_519876

variable (total_land : ℕ) (corn_cost_per_acre : ℕ) (wheat_cost_per_acre : ℕ)
variable (acres_wheat : ℕ) (acres_corn : ℕ)

def total_money (total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn : ℕ) : ℕ :=
  acres_corn * corn_cost_per_acre + acres_wheat * wheat_cost_per_acre

theorem Johnson_farm_budget :
  total_land = 500 ∧
  corn_cost_per_acre = 42 ∧
  wheat_cost_per_acre = 30 ∧
  acres_wheat = 200 ∧
  acres_corn = total_land - acres_wheat →
  total_money total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn = 18600 := by
  sorry

end Johnson_farm_budget_l519_519876


namespace percentage_reduction_dodecagon_l519_519130

theorem percentage_reduction_dodecagon (s : ℝ) :
  let A := 3 * s^2 * Real.cot (Real.pi / 12)
  let A' := 3 * (s / 2)^2 * Real.cot (Real.pi / 12)
  (A - A') / A * 100 = 75 := by
  sorry

end percentage_reduction_dodecagon_l519_519130


namespace part1_part2_l519_519627

-- Define the function f and its conditions
def f (x a : ℝ) := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- Define the condition for part 1
theorem part1 (x : ℝ) : 
  f x 1 = 2 * x^3 - 6 * x^2 + 6 * x → 
  (∀ x : ℝ, (6 * x^2 - 12 * x + 6) ≥ 0) → 
  Monotone (λ x, f x 1) :=
sorry

-- Define the condition for part 2 when a=2
theorem part2 (a : ℝ) :
  (∀ x ∈ Icc (1:ℝ) 3, f x a ≥ 4) →
  (∃ a', a' = 2) :=
sorry

end part1_part2_l519_519627


namespace graph_passes_through_point_l519_519458

theorem graph_passes_through_point (a : ℝ) (h : 0 < a ∧ a < 1) : ∃ (x y : ℝ), x = 1 ∧ y = 2 ∧ y = 2 * a^(x - 1) :=
by {
  use [1, 2],
  split,
  { refl },
  split,
  { refl },
  { sorry }
}

end graph_passes_through_point_l519_519458


namespace c_10_eq_3_pow_89_l519_519554

section sequence
  open Nat

  -- Define the sequence c
  def c : ℕ → ℕ
  | 0     => 3  -- Note: Typically Lean sequences start from 0, not 1
  | 1     => 9
  | (n+2) => c n.succ * c n

  -- Define the auxiliary sequence d
  def d : ℕ → ℕ
  | 0     => 1  -- Note: Typically Lean sequences start from 0, not 1
  | 1     => 2
  | (n+2) => d n.succ + d n

  -- The theorem we need to prove
  theorem c_10_eq_3_pow_89 : c 9 = 3 ^ d 9 :=    -- Note: c_{10} in the original problem is c(9) in Lean
  sorry   -- Proof omitted
end sequence

end c_10_eq_3_pow_89_l519_519554


namespace radius_of_circle_l519_519356

theorem radius_of_circle (r : ℝ) (x y : ℝ) : 
  (r^2 = x^2 + 100) ∧
  (r^2 = (x + y)^2 + 64) ∧
  (r^2 = (x + 2y)^2 + 16) →
  r = 5 * real.sqrt 22 / 2 :=
begin
  sorry -- Proof is not provided intentionally
end

end radius_of_circle_l519_519356


namespace acquaintance_paradox_proof_l519_519713

theorem acquaintance_paradox_proof :
  ∃ (x : ℕ), (acquaintances x ≥ 810) :=
by
  let n_population := 1000000
  let n_believers := 900000
  let n_nonBelievers := n_population - n_believers
  have h1 : n_population = 1000000 := rfl
  have h2 : n_believers = 0.90 * n_population := by sorry
  have h3 : n_nonBelievers = n_population - n_believers := by sorry
  have h4 : ∀ x, acquaintances x ≥ 1 := by sorry
  have h5 : ∀ x, 0.10 * acquaintances x = nonBelievers_acquaintances x := by sorry
  have h6 : nonBelievers_acquaintances_total >= 90 := by sorry
  
  show ∃ (x : ℕ), acquaintances x ≥ 810 from sorry
  
  sorry

end acquaintance_paradox_proof_l519_519713


namespace car_mileage_l519_519546

theorem car_mileage :
  ∀ (cost_per_gallon total_cost miles: ℕ),
  cost_per_gallon = 4 →
  total_cost = 58 →
  miles = 464 →
  (miles / (total_cost / cost_per_gallon) = 32) :=
by
  assume cost_per_gallon total_cost miles H1 H2 H3
  sorry

end car_mileage_l519_519546


namespace red_tulips_for_smile_l519_519524

/-
Problem Statement:
Anna wants to plant red and yellow tulips in the shape of a smiley face. Given the following conditions:
1. Anna needs 8 red tulips for each eye.
2. She needs 9 times the number of red tulips in the smile to make the yellow background of the face.
3. The total number of tulips needed is 196.

Prove:
The number of red tulips needed for the smile is 18.
-/

-- Defining the conditions
def red_tulips_per_eye : Nat := 8
def total_tulips : Nat := 196
def yellow_multiplier : Nat := 9

-- Proving the number of red tulips for the smile
theorem red_tulips_for_smile (R : Nat) :
  2 * red_tulips_per_eye + R + yellow_multiplier * R = total_tulips → R = 18 :=
by
  sorry

end red_tulips_for_smile_l519_519524


namespace digit_of_one_seventh_l519_519074

theorem digit_of_one_seventh :
  ∀ n : ℕ, (n = 127) → 
  let seq := [1, 4, 2, 8, 5, 7] in
  let period := 6 in
  seq[(n % period)] = 1 :=
by 
  sorry

end digit_of_one_seventh_l519_519074


namespace inhabitant_knows_at_least_810_l519_519743

open Function

def population : ℕ := 1000000

def believers_claus (p : ℕ) : Prop := p = (90 * population / 100)

def knows_someone (p : ℕ) (knows : p → p → Prop) : Prop := 
  ∀ x : p, ∃ y : p, knows x y

def acquaintances_claus (p : ℕ) (believes : p → Prop) (knows : p → p → Prop) : Prop :=
  ∀ x : p, (believes x →(∃ k : ℕ, ∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y ∧ (10 * k / 100) y ∧ ¬believes y))

theorem inhabitant_knows_at_least_810 :
  ∃ (p : ℕ) (believes : p → Prop) (knows : p → p → Prop),
    population = 1000000 ∧
    believers_claus p ∧
    knows_someone p knows ∧
    acquaintances_claus p believes knows →
    ∃ x : p, ∃ k : ℕ, k ≥ 810 ∧ (∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y) :=
sorry

end inhabitant_knows_at_least_810_l519_519743


namespace next_palindrome_date_is_20211202_l519_519568

/-- Define what it means for a date to be a palindrome --/
def is_palindrome_date (d : String) : Prop :=
  d = d.reverse

/-- Define the next palindrome date after a given date --/
noncomputable def next_palindrome_date_after (d : String) : String :=
  sorry  -- In a real proof, this would involve search and validation logic

/-- The next palindrome date after 2020-02-02 (20200202) is 2021-12-02 (20211202) --/
theorem next_palindrome_date_is_20211202 : next_palindrome_date_after "20200202" = "20211202" :=
  sorry

end next_palindrome_date_is_20211202_l519_519568


namespace general_formula_sum_of_b_l519_519249

variable {a : ℕ → ℕ} (b : ℕ → ℕ)
variable (n : ℕ)

def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n+2) = q * a (n+1)

def initial_conditions (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 = 9 ∧ a 2 + a 3 = 18

theorem general_formula (q : ℕ) (h1 : is_geometric_sequence a q) (h2 : initial_conditions a) :
  a n = 3 * 2^(n - 1) :=
sorry

def b_n (a : ℕ → ℕ) (n : ℕ) : ℕ := a n + 2 * n

def sum_b (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem sum_of_b (h1 : ∀ m : ℕ, b m = a m + 2 * m) (h2 : initial_conditions a) :
  sum_b b n = 3 * 2^n + n * (n + 1) - 3 :=
sorry

end general_formula_sum_of_b_l519_519249


namespace evan_runs_200_more_feet_l519_519780

def street_width : ℕ := 25
def block_side : ℕ := 500

def emily_path : ℕ := 4 * block_side
def evan_path : ℕ := 4 * (block_side + 2 * street_width)

theorem evan_runs_200_more_feet : evan_path - emily_path = 200 := by
  sorry

end evan_runs_200_more_feet_l519_519780


namespace solve_logarithmic_equation_l519_519023

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem solve_logarithmic_equation (x : ℝ) (h_pos : x > 0) :
  log_base 8 x + log_base 4 (x^2) + log_base 2 (x^3) = 15 ↔ x = 2 ^ (45 / 13) :=
by
  have h1 : log_base 8 x = (1 / 3) * log_base 2 x :=
    by { sorry }
  have h2 : log_base 4 (x^2) = log_base 2 x :=
    by { sorry }
  have h3 : log_base 2 (x^3) = 3 * log_base 2 x :=
    by { sorry }
  have h4 : (1 / 3) * log_base 2 x + log_base 2 x + 3 * log_base 2 x = 15 ↔ log_base 2 x = 45 / 13 :=
    by { sorry }
  exact sorry

end solve_logarithmic_equation_l519_519023


namespace at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519716

theorem at_least_one_inhabitant_knows_no_fewer_than_810_people :
  ∀ (n : ℕ),
  (inhabitants : ℕ) (knows_someone : inhabitants > 1) (believes_santa : inhabitants * 9 / 10)
  (ten_percent_acquaintances_believe : ∀ x, (acquaintances : list x) → (|list.contacts_with_santa <- 0.10 * acquaintances|)),
  (h : inhabitants = 1000000),
  (one_inhabitant_knows : ∃ x, ∀ y, y \in acquaintances x → friends y ≥ 810) :=
sorry

end at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519716


namespace distance_between_homes_l519_519365

-- Define the conditions as Lean functions and values
def walking_speed_maxwell : ℝ := 3
def running_speed_brad : ℝ := 5
def distance_traveled_maxwell : ℝ := 15

-- State the theorem
theorem distance_between_homes : 
  ∃ D : ℝ, 
    (15 = walking_speed_maxwell * (distance_traveled_maxwell / walking_speed_maxwell)) ∧ 
    (D - 15 = running_speed_brad * (distance_traveled_maxwell / walking_speed_maxwell)) ∧ 
    D = 40 :=
by 
  sorry

end distance_between_homes_l519_519365


namespace num_triangles_with_fixed_vertex_l519_519065

-- Define the number of points on the circle
def num_points : ℕ := 12

-- Define the number of remaining points after fixing one point as a vertex
def remaining_points : ℕ := num_points - 1

-- Prove the number of triangles that can be formed with one fixed vertex
theorem num_triangles_with_fixed_vertex : combinatorics.choose remaining_points 2 = 55 := 
by sorry

end num_triangles_with_fixed_vertex_l519_519065


namespace nuts_distribution_l519_519953

def walnuts_initially : ℕ := 1021
def walnuts_remaining : ℕ := 321

theorem nuts_distribution:
  let x := walnuts_initially in
  let x_r := walnuts_remaining in
  ∀ (T E B J : ℕ),
    T = 1 + (x - 1) / 4 ∧
    E = 1 + (3 * x - 7) / 16 ∧
    B = 1 + (9 * x - 25) / 64 ∧
    J = 1 + (27 * x - 127) / 256 ∧
    T + B = E + J + 100 →
    x = 1021 ∧ x_r = (x - (T + E + B + J)) :=
sorry

end nuts_distribution_l519_519953


namespace knight_reach_h1_in_4_knight_not_reach_e6_in_5_l519_519452

-- Define the structure of the board and the knight's movements.
structure Position :=
  (row : ℕ)  -- row is 1 through 8
  (col : Nat)  -- column is a through h (represented by numbers 1 through 8)

structure Board :=
  (size : ℕ)
  (positions : List Position)

-- This instance represents the 8x8 chessboard
def chessboard : Board :=
  { size := 8,
    positions := (List.range 8).bind (λ i, (List.range 8).map (λ j, Position.mk (i + 1) (j + 1))) }

-- Knight movement function (return all valid positions a knight can move to from a given position)
def knight_moves (pos : Position) (b : Board) : List Position :=
  let moves := [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
  moves.filter_map 
    (λ (dr, dc), 
      let new_row := pos.row + dr
      let new_col := pos.col + dc
      if new_row ≥ 1 ∧ new_row ≤ b.size ∧ new_col ≥ 1 ∧ new_col ≤ b.size 
      then some { row := new_row, col := new_col } else none)

-- Function to check if a position is reachable within a given number of moves
noncomputable def reachable_in_n_moves (start_pos end_pos : Position) (n : ℕ) : Prop :=
  (λ b => sorry) -- actual steps would go here

-- Specific positions in terms of row and column indices.
def b1 : Position := {row := 1, col := 2}
def h1 : Position := {row := 1, col := 8}
def e6 : Position := {row := 6, col := 5}

-- Part 1: Prove knight can reach h1 in four moves from b1
theorem knight_reach_h1_in_4 : reachable_in_n_moves b1 h1 4 :=
  sorry

-- Part 2: Prove knight cannot reach e6 in five moves from b1
theorem knight_not_reach_e6_in_5 : ¬ reachable_in_n_moves b1 e6 5 :=
  sorry

end knight_reach_h1_in_4_knight_not_reach_e6_in_5_l519_519452


namespace factor_expression_l519_519201

theorem factor_expression (x y : ℝ) : x * y^2 - 4 * x = x * (y + 2) * (y - 2) := 
by
  sorry

end factor_expression_l519_519201


namespace components_per_month_l519_519500

theorem components_per_month : 
  ∀ (production_cost shipping_cost fixed_cost selling_price : ℕ) (production_function revenue_function : ℕ → ℕ),
  production_cost = 80 →
  shipping_cost = 2 →
  fixed_cost = 16200 →
  selling_price = 190 →
  production_function = λ x, fixed_cost + (production_cost + shipping_cost) * x →
  revenue_function = λ x, selling_price * x →
  ∃ x, production_function x = revenue_function x ∧ x = 150 :=
by
  intros pc sc fc sp prod_fun rev_fun hpc hsc hfc hsp hpf hrf
  use 150
  rw [hpc, hsc, hfc, hsp, hpf, hrf]
  have : 16200 + (80 + 2) * 150 = 190 * 150 := by
    norm_num
  exact ⟨this, rfl⟩

end components_per_month_l519_519500


namespace trigonometric_identity_l519_519184

theorem trigonometric_identity (α : ℝ) : 
  - (Real.sin α) + (Real.sqrt 3) * (Real.cos α) = 2 * (Real.sin (α + 2 * Real.pi / 3)) :=
by
  sorry

end trigonometric_identity_l519_519184


namespace probability_three_good_students_l519_519774

theorem probability_three_good_students (total_people three_good_students selected_people : ℕ)
  (h_total_people : total_people = 12)
  (h_three_good_students : three_good_students = 5)
  (h_selected_people : selected_people = 6)
  (ξ : ℕ → ℕ) :
  (∑ ξ = 3).card = (nat.choose three_good_students 3 * nat.choose (total_people - three_good_students) (selected_people - 3))
  /
  (nat.choose total_people selected_people) :=
by sorry

end probability_three_good_students_l519_519774


namespace product_of_values_of_k_l519_519187

-- Define the function g
def g (k x : ℝ) : ℝ := k / (2 * x - 5)

-- Define the inverse function of g when evaluated at k + 2
def g_inv (k x : ℝ) : ℝ := 2 * x - 5

theorem product_of_values_of_k 
  (k : ℝ) 
  (h1 : g k 3 = g_inv k (k + 2))
  (h2 : ∀ k, g k k = k + 2) :
  ∏ k ∈ {k : ℝ | 2 * k^2 + k - 10 = 0}, k = -5 :=
by
  sorry

end product_of_values_of_k_l519_519187


namespace fourth_vertex_of_tetrahedron_exists_l519_519110

theorem fourth_vertex_of_tetrahedron_exists (x y z : ℤ) :
  (∃ (x y z : ℤ), 
     ((x - 1) ^ 2 + y ^ 2 + (z - 3) ^ 2 = 26) ∧ 
     ((x - 5) ^ 2 + (y - 3) ^ 2 + (z - 2) ^ 2 = 26) ∧ 
     ((x - 4) ^ 2 + y ^ 2 + (z - 6) ^ 2 = 26)) :=
sorry

end fourth_vertex_of_tetrahedron_exists_l519_519110


namespace slope_angle_of_line_l519_519435

theorem slope_angle_of_line (x y : ℝ) (h : x + sqrt 3 * y - 5 = 0) : 
  ∃ α : ℝ, α = 150 ∧ tan α = (-(sqrt 3 / 3)) :=
sorry

end slope_angle_of_line_l519_519435


namespace degree_of_polynomial_l519_519414

theorem degree_of_polynomial : 
  degree (C (1 : ℝ) * (X ^ 2 * Y) + C (π : ℝ) * (X ^ 2 * Y ^ 2) - (X * Y ^ 3)) = 4 := 
sorry

end degree_of_polynomial_l519_519414


namespace sum_of_solutions_of_quadratic_eq_l519_519578

theorem sum_of_solutions_of_quadratic_eq :
  (∀ x : ℝ, 5 * x^2 - 3 * x - 2 = 0) → (∀ a b : ℝ, a = 5 ∧ b = -3 → -b / a = 3 / 5) :=
by
  sorry

end sum_of_solutions_of_quadratic_eq_l519_519578


namespace choose_amber_bronze_cells_l519_519837

theorem choose_amber_bronze_cells (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (grid : Fin (a+b+1) × Fin (a+b+1) → Prop) 
  (amber_cells : ℕ) (h_amber_cells : amber_cells ≥ a^2 + a * b - b)
  (bronze_cells : ℕ) (h_bronze_cells : bronze_cells ≥ b^2 + b * a - a):
  ∃ (amber_choice : Fin (a+b+1) → Fin (a+b+1)), 
    ∃ (bronze_choice : Fin (a+b+1) → Fin (a+b+1)), 
    amber_choice ≠ bronze_choice ∧ 
    (∀ i j, i ≠ j → grid (amber_choice i) ≠ grid (bronze_choice j)) :=
sorry

end choose_amber_bronze_cells_l519_519837


namespace at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519721

theorem at_least_one_inhabitant_knows_no_fewer_than_810_people :
  ∀ (n : ℕ),
  (inhabitants : ℕ) (knows_someone : inhabitants > 1) (believes_santa : inhabitants * 9 / 10)
  (ten_percent_acquaintances_believe : ∀ x, (acquaintances : list x) → (|list.contacts_with_santa <- 0.10 * acquaintances|)),
  (h : inhabitants = 1000000),
  (one_inhabitant_knows : ∃ x, ∀ y, y \in acquaintances x → friends y ≥ 810) :=
sorry

end at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519721


namespace incorrect_statement_B_l519_519927

variable {α β : Type*} [OrderedCommRing α] (f : α → β)

def monotonic_interval (f : α → β) := ∀ x y, x ≤ y → f x ≤ f y

def definition_domain (f : α → β) := true -- Assume f is defined on the whole type α for simplicity

def symmetric_about_origin (f : α → β) := ∀ x, f (-x) = f x

def odd_function (f : α → β) := ∀ x, f (-x) = -f x

theorem incorrect_statement_B :
  ¬ ∀ (f : α → β) (I J : Set α),
    monotonic_interval f →
    I ∪ J = monotonic_interval f → (I ∪ J) = monotonic_interval f :=
by sorry

end incorrect_statement_B_l519_519927


namespace tangent_and_normal_are_correct_at_point_l519_519088

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - 2*x*y + 3*y^2 - 2*y - 16 = 0

def tangent_line (x y : ℝ) : Prop :=
  2*x - 7*y + 19 = 0

def normal_line (x y : ℝ) : Prop :=
  7*x + 2*y - 13 = 0

theorem tangent_and_normal_are_correct_at_point
  (hx : point_on_curve 1 3) :
  tangent_line 1 3 ∧ normal_line 1 3 :=
by
  sorry

end tangent_and_normal_are_correct_at_point_l519_519088


namespace ratio_of_areas_C_D_compare_area_E_to_CD_l519_519894

noncomputable def side_length_C : ℝ := 24
noncomputable def side_length_D : ℝ := 30
noncomputable def rect_E_length : ℝ := 40
noncomputable def rect_E_width : ℝ := 45

def area_square (s : ℝ) : ℝ := s * s
def area_rectangle (l w : ℝ) : ℝ := l * w

def ratio_C_to_D_area : ℝ := (area_square side_length_C) / (area_square side_length_D)
def combined_area_squares_CD : ℝ := (area_square side_length_C) + (area_square side_length_D)
def ratio_E_to_combined_CD_area : ℝ := (area_rectangle rect_E_length rect_E_width) / combined_area_squares_CD

theorem ratio_of_areas_C_D : ratio_C_to_D_area = 16/25 := by sorry

theorem compare_area_E_to_CD : ratio_E_to_combined_CD_area = 1800/1476 := by sorry

end ratio_of_areas_C_D_compare_area_E_to_CD_l519_519894


namespace hiker_total_distance_correct_l519_519119

def distance_walked_day1 : ℝ := 18
def speed_day1 : ℝ := 3
def hours_day1 : ℝ := distance_walked_day1 / speed_day1

def speed_day2 : ℝ := speed_day1 + 1
def hours_day2 : ℝ := hours_day1 - 1
def distance_walked_day2 : ℝ := speed_day2 * hours_day2

def speed_day3 : ℝ := speed_day2
def hours_day3 : ℝ := hours_day1
def distance_walked_day3 : ℝ := speed_day3 * hours_day3

def total_distance_walked : ℝ :=
  distance_walked_day1 + distance_walked_day2 + distance_walked_day3

theorem hiker_total_distance_correct : total_distance_walked = 62 := by
  sorry

end hiker_total_distance_correct_l519_519119


namespace Mr_Zhang_reaches_school_on_time_R1_recommended_route_to_minimize_time_l519_519851

-- Definitions for probabilities and delays
def prob_green_R1_A : ℝ := 1 / 2
def prob_green_R1_B : ℝ := 2 / 3
def delay_red_R1_A : ℕ := 2
def delay_red_R1_B : ℕ := 3
def total_time_green_R1 : ℕ := 20

def prob_green_R2_a : ℝ := 3 / 4
def prob_green_R2_b : ℝ := 2 / 5
def delay_red_R2_a : ℕ := 8
def delay_red_R2_b : ℕ := 5
def total_time_green_R2 : ℕ := 15

-- Probability Mr. Zhang reaches the school in 20 minutes via Route 1
def prob_R1_reaches_school_in_20_min : ℝ :=
  prob_green_R1_A * prob_green_R1_B

theorem Mr_Zhang_reaches_school_on_time_R1 :
  prob_R1_reaches_school_in_20_min = 1 / 3 := by
  sorry

-- Expected commuting time for Route 1
def expected_delay_R1 : ℝ :=
  0 * (prob_green_R1_A * prob_green_R1_B) +
  delay_red_R1_A * (prob_green_R1_A * (1 - prob_green_R1_B)) +
  delay_red_R1_B * ((1 - prob_green_R1_A) * prob_green_R1_B) +
  (delay_red_R1_A + delay_red_R1_B) * ((1 - prob_green_R1_A) * (1 - prob_green_R1_B))

def avg_commute_time_R1 : ℝ :=
  total_time_green_R1 + expected_delay_R1

-- Expected commuting time for Route 2
def expected_delay_R2 : ℝ :=
  0 * (prob_green_R2_a * prob_green_R2_b) +
  delay_red_R2_a * (prob_green_R2_a * (1 - prob_green_R2_b)) +
  delay_red_R2_b * ((1 - prob_green_R2_a) * prob_green_R2_b) +
  (delay_red_R2_a + delay_red_R2_b) * ((1 - prob_green_R2_a) * (1 - prob_green_R2_b))

def avg_commute_time_R2 : ℝ :=
  total_time_green_R2 + expected_delay_R2

-- The recommended route to minimize commuting time
theorem recommended_route_to_minimize_time :
  avg_commute_time_R1 > avg_commute_time_R2 := by
  sorry

end Mr_Zhang_reaches_school_on_time_R1_recommended_route_to_minimize_time_l519_519851


namespace find_m_is_perpendicular_l519_519669

variable (m : ℝ)

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 3)
def b : ℝ × ℝ := (4, -2)

-- Definition of the perpendicular condition
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Statement to be proven in Lean 4
theorem find_m_is_perpendicular : is_perpendicular (m * a + b) a → m = 1 :=
by
  -- this by block would be filled with the required steps to prove the theorem, which is skipped here
  sorry

end find_m_is_perpendicular_l519_519669


namespace maximum_unique_sums_l519_519951

theorem maximum_unique_sums :
  let coins := [1, 1, 5, 5, 10, 10, 25, 50]
  let pairs := (coins.product coins).filter (λ pair, pair.fst ≤ pair.snd)
  let sums := pairs.map (λ pair, pair.fst + pair.snd)
  sums.to_finset.card = 15 :=
by
  sorry

end maximum_unique_sums_l519_519951


namespace number_of_spiders_l519_519973

theorem number_of_spiders (total_legs birds dogs snakes : ℕ) (legs_per_bird legs_per_dog legs_per_snake legs_per_spider : ℕ) (h1 : total_legs = 34)
  (h2 : birds = 3) (h3 : dogs = 5) (h4 : snakes = 4) (h5 : legs_per_bird = 2) (h6 : legs_per_dog = 4)
  (h7 : legs_per_snake = 0) (h8 : legs_per_spider = 8) : 
  (total_legs - (birds * legs_per_bird + dogs * legs_per_dog + snakes * legs_per_snake)) / legs_per_spider = 1 :=
by sorry

end number_of_spiders_l519_519973


namespace Y_subset_X_l519_519863

def X : Set ℕ := {n | ∃ m : ℕ, n = 4 * m + 2}

def Y : Set ℕ := {t | ∃ k : ℕ, t = (2 * k - 1)^2 + 1}

theorem Y_subset_X : Y ⊆ X := by
  sorry

end Y_subset_X_l519_519863


namespace algebraic_expression_value_l519_519212

-- Definitions for the problem conditions
def x := -1
def y := 1 / 2
def expr := 2 * (x^2 - 5 * x * y) - 3 * (x^2 - 6 * x * y)

-- The problem statement to be proved
theorem algebraic_expression_value : expr = 3 :=
by
  sorry

end algebraic_expression_value_l519_519212


namespace total_beverages_and_average_per_person_l519_519196

-- Variables to store lemonade and iced tea pitchers served in each intermission
def lemonade_first_inter : ℝ := 0.25
def iced_tea_first_inter : ℝ := 0.18
def people_first_inter : ℕ := 15

def lemonade_second_inter : ℝ := 0.42
def iced_tea_second_inter : ℝ := 0.30
def people_second_inter : ℕ := 22

def lemonade_third_inter : ℝ := 0.25
def iced_tea_third_inter : ℝ := 0.15
def people_third_inter : ℕ := 12

-- Prove the total amount of beverages served and the average amount of beverage per person
theorem total_beverages_and_average_per_person :
  (lemonade_first_inter + lemonade_second_inter + lemonade_third_inter +
   iced_tea_first_inter + iced_tea_second_inter + iced_tea_third_inter = 1.55) ∧
  (1.55 / (people_first_inter + people_second_inter + people_third_inter) ≈ 0.0316) :=
by
  sorry

end total_beverages_and_average_per_person_l519_519196


namespace perpendicular_vectors_angle_between_vectors_l519_519675

-- Definitions for the vectors
def a : Vector ℝ := ⟨1, -1⟩
def b (k : ℝ) : Vector ℝ := ⟨1, k⟩

-- Problem 1: Show that if a is perpendicular to b, then k = 1
theorem perpendicular_vectors (k : ℝ) (h : dot_product a (b k) = 0) : k = 1 := sorry

-- Problem 2: Show that if the angle between a and b is π/3, then k = 2 - sqrt 3
theorem angle_between_vectors (k : ℝ) (h : arccos (dot_product a (b k) / (norm a * norm (b k))) = π / 3) : k = 2 - sqrt 3 := sorry

end perpendicular_vectors_angle_between_vectors_l519_519675


namespace problem_statement_l519_519425

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

lemma omega_cubed_eq_one : omega^3 = 1 := 
  by sorry

lemma omega_is_root : omega^2 + omega + 1 = 0 := 
  by sorry

theorem problem_statement (A B : ℝ) 
  (h: ∀ (x : ℂ), x^2 + x + 1 = 0 → x^103 + A * x + B = 0) : A + B = -1 :=
by 
  have h_omega := h omega omega_is_root
  have omega_exp : omega^103 = omega :=
    by {
      calc omega^103 = (omega^3)^(34) * omega : by sorry
              ... = 1^34 * omega : by rw omega_cubed_eq_one
              ... = omega : by sorry
    }
  rw [omega_exp] at h_omega
  sorry

end problem_statement_l519_519425


namespace units_digit_sum_sequence_l519_519917

noncomputable def units_digit : Nat → Nat
| n := n % 10

theorem units_digit_sum_sequence :
  let seq := [1!, 2!, 3!, 4!, 5!, 6!, 7!, 8!, 9!, 10!, 11!]
  let seq_plus_indices := List.map2 (λ a b => a + b) seq (List.range' 1 11)
  let seq_sum := seq_plus_indices.sum
  units_digit seq_sum = 9 :=
by
  sorry

end units_digit_sum_sequence_l519_519917


namespace profit_per_meter_is_20_l519_519962

-- Define given conditions
def selling_price_total (n : ℕ) (price : ℕ) : ℕ := n * price
def cost_price_per_meter : ℕ := 85
def selling_price_total_85_meters : ℕ := 8925

-- Define the expected profit per meter
def expected_profit_per_meter : ℕ := 20

-- Rewrite the problem statement: Prove that with given conditions the profit per meter is Rs. 20
theorem profit_per_meter_is_20 
  (n : ℕ := 85)
  (sp : ℕ := selling_price_total_85_meters)
  (cp_pm : ℕ := cost_price_per_meter) 
  (expected_profit : ℕ := expected_profit_per_meter) :
  (sp - n * cp_pm) / n = expected_profit :=
by
  sorry

end profit_per_meter_is_20_l519_519962


namespace total_watermelon_slices_l519_519548

theorem total_watermelon_slices 
(h1 : ∀ (n : Nat), n > 0 → n * 10 = 30)
(h2 : ∀ (m : Nat), m > 0 → m * 15 = 15) :
  3 * 10 + 1 * 15 = 45 := 
  by 
    have h_danny : 3 * 10 = 30 := h1 3 (by norm_num)
    have h_sister: 1 * 15 = 15 := h2 1 (by norm_num)
    calc
      3 * 10 + 1 * 15 = 30 + 15 := by rw [h_danny, h_sister]
                  ... = 45 := by norm_num

end total_watermelon_slices_l519_519548


namespace total_spent_on_toys_and_clothes_l519_519807

def cost_toy_cars : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def cost_toy_trucks : ℝ := 5.86
def cost_pants : ℝ := 14.55
def cost_shirt : ℝ := 7.43
def cost_hat : ℝ := 12.50

theorem total_spent_on_toys_and_clothes :
  (cost_toy_cars + cost_skateboard + cost_toy_trucks) + (cost_pants + cost_shirt + cost_hat) = 60.10 :=
by
  sorry

end total_spent_on_toys_and_clothes_l519_519807


namespace limit_sequence_zero_l519_519979

open_locale topological_space

noncomputable def sequence (n : ℕ) : ℝ :=
  (n^3 - (n - 1)^3) / ((n + 1)^4 - n^4)

theorem limit_sequence_zero : 
  tendsto (λ n, sequence n) at_top (𝓝 0) :=
sorry

end limit_sequence_zero_l519_519979


namespace eval_f_at_2_l519_519072

def f(x : ℝ) : ℝ := 6 * x^6 + 4 * x^5 - 2 * x^4 + 5 * x^3 - 7 * x^2 - 2 * x + 5

theorem eval_f_at_2 : 
  let x := 2 in
  f x = 21 ∧ -- Evaluated polynomial
  6 = 6 ∧ -- Number of multiplications in Horner's method
  3 = 3 -- Number of additions in Horner's method
:= 
by
  -- The proof will be here
  sorry

end eval_f_at_2_l519_519072


namespace quadrilateral_fourth_side_length_l519_519507

theorem quadrilateral_fourth_side_length 
  (r : ℝ) (a b c : ℝ) (d : ℝ) 
  (condition_radius : r = 200 * real.sqrt 2)
  (condition_sides : a = 200 ∧ b = 200 ∧ c = 200) :
  d = 500 :=
sorry

end quadrilateral_fourth_side_length_l519_519507


namespace stockholm_to_malmo_road_distance_l519_519416

-- Define constants based on the conditions
def map_distance_cm : ℕ := 120
def scale_factor : ℕ := 10
def road_distance_multiplier : ℚ := 1.15

-- Define the real distances based on the conditions
def straight_line_distance_km : ℕ :=
  map_distance_cm * scale_factor

def road_distance_km : ℚ :=
  straight_line_distance_km * road_distance_multiplier

-- Assert the final statement
theorem stockholm_to_malmo_road_distance :
  road_distance_km = 1380 := 
sorry

end stockholm_to_malmo_road_distance_l519_519416


namespace ratio_of_radii_l519_519812

/-- Let ABC be a triangle and Γ be its incircle with radius r. 
    Let Γ' be a circle lying outside the triangle, 
    touching the incircle externally and the sides AB and AC with radius ρ.
    Show that the ratio of the radii of the circles Γ' and Γ is equal to tan²((π - A) / 4). -/
theorem ratio_of_radii (ABC : Triangle) (r ρ : ℝ) (A : ℝ) (Γ Γ' : Circle) 
  (h1 : Γ.radius = r) 
  (h2 : Γ'.radius = ρ) 
  (h3 : Γ' ∈ exterior of ABC) 
  (h4 : Γ' touches Γ externally) 
  (h5 : Γ' touches sides (AB : Line) and (AC : Line) of ABC) : 
  ρ / r = (Real.tan ((Real.pi - A) / 4))^2 := 
sorry

end ratio_of_radii_l519_519812


namespace product_of_six_consecutive_nat_not_equal_776965920_l519_519861

theorem product_of_six_consecutive_nat_not_equal_776965920 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ≠ 776965920) :=
by
  sorry

end product_of_six_consecutive_nat_not_equal_776965920_l519_519861


namespace true_converses_count_l519_519182

-- Definitions according to the conditions
def parallel_lines (L1 L2 : Prop) : Prop := L1 ↔ L2
def congruent_triangles (T1 T2 : Prop) : Prop := T1 ↔ T2
def vertical_angles (A1 A2 : Prop) : Prop := A1 = A2
def squares_equal (m n : ℝ) : Prop := m = n → (m^2 = n^2)

-- Propositions with their converses
def converse_parallel (L1 L2 : Prop) : Prop := parallel_lines L1 L2 → parallel_lines L2 L1
def converse_congruent (T1 T2 : Prop) : Prop := congruent_triangles T1 T2 → congruent_triangles T2 T1
def converse_vertical (A1 A2 : Prop) : Prop := vertical_angles A1 A2 → vertical_angles A2 A1
def converse_squares (m n : ℝ) : Prop := (m^2 = n^2) → (m = n)

-- Proving the number of true converses
theorem true_converses_count : 
  (∃ L1 L2, converse_parallel L1 L2) →
  (∃ T1 T2, ¬converse_congruent T1 T2) →
  (∃ A1 A2, converse_vertical A1 A2) →
  (∃ m n : ℝ, ¬converse_squares m n) →
  (2 = 2) := by
  intros _ _ _ _
  sorry

end true_converses_count_l519_519182


namespace inscribed_circle_radius_l519_519309

theorem inscribed_circle_radius
  (p : ℕ)
  (A : ℕ)
  (s : ℕ)
  (h1 : p = 24)
  (h2 : A = p ^ 2 / 4)
  (h3 : s = p / 2) :
  let r := A / s in r = 12 :=
by
  sorry

end inscribed_circle_radius_l519_519309


namespace part1_part2_l519_519656

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519656


namespace angles_of_triangle_CDE_l519_519008

variable {A B C M N E D : Type}
variable [EquilateralTriangle ABC]
variable (M : lies_on M A B)
variable (N : lies_on N B C)
variable (MN_parallel_AC : Parallel MN AC)
variable (E_mid_AN : Midpoint E A N)
variable (D_centroid_BMN : Centroid D B M N)

theorem angles_of_triangle_CDE (h : EquilateralTriangle ABC) 
  (MN_parallel_AC : Parallel MN AC) 
  (E_mid_AN : Midpoint E A N)
  (D_centroid_BMN : Centroid D B M N) :
  angles C D E = [30, 60, 90] :=
sorry

end angles_of_triangle_CDE_l519_519008


namespace ferris_wheel_small_seat_capacity_l519_519028

def num_small_seats : Nat := 2
def capacity_per_small_seat : Nat := 14

theorem ferris_wheel_small_seat_capacity : num_small_seats * capacity_per_small_seat = 28 := by
  sorry

end ferris_wheel_small_seat_capacity_l519_519028


namespace part_a_max_sum_part_b_infinite_triples_l519_519945

-- Define the conditions of the problem for part (a)
def problem_conditions (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ 16 * x * y * z = (x + y)^2 * (x + z)^2

-- Define the conditions of the problem for part (b)
def rational_conditions (x y z : ℚ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ 16 * x * y * z = (x + y)^2 * (x + z)^2

-- Part (a): Prove the maximum value of x + y + z is 4 under the given conditions
theorem part_a_max_sum (x y z : ℝ) (h : problem_conditions x y z) : x + y + z ≤ 4 :=
  sorry

-- Part (b): Prove there are infinitely many triples (x, y, z) of positive rational numbers such that 16xyz = (x+y)^2 (x+z)^2 and x + y + z = 4
theorem part_b_infinite_triples : ∃ᶠ x y z : ℚ in (set.univ), rational_conditions x y z ∧ x + y + z = 4 :=
  sorry

end part_a_max_sum_part_b_infinite_triples_l519_519945


namespace find_a_find_prob_density_func_find_prob_interval_l519_519477

def F (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 1 then 0 else
  if x ≤ Real.exp 1 then a * Real.log x else
  1

def f (x : ℝ) : ℝ :=
  if x ≤ 1 ∨ x > Real.exp 1 then 0 else
  1 / x

theorem find_a (a : ℝ) : (∀ x : ℝ, F x a = F x 1) → a = 1 :=
  sorry

theorem find_prob_density_func : (∀ x : ℝ, F x 1 = F x 1) → 
  (∀ x : ℝ, f x = (if x ≤ 1 ∨ x > Real.exp 1 then 0 else 1 / x)) :=
  sorry

theorem find_prob_interval : (∀ x : ℝ, F x 1 = F x 1) →
  F (Real.sqrt (Real.exp 1)) 1 - F ((Real.exp 1)^(-1/3)) 1 = 1 / 2 :=
  sorry

end find_a_find_prob_density_func_find_prob_interval_l519_519477


namespace vasya_guarantee_win_l519_519855

/-
  Define the types of players, the range of numbers on the cards, 
  and the game strategy leading to Vasya's guaranteed win.
-/

inductive Player
| Petya
| Vasya

/--
  Given the described card game and conditions, Vasya can always guarantee a win.
-/
theorem vasya_guarantee_win :
  ∃ strategy : (ℕ → Player → ℕ),
  (∀ seq, seq.length > 0 → 
   let total_number := seq.sum in
   (∃ k : ℤ, total_number = k^2 - (k-1)^2) ∨ (∃ k : ℤ, total_number = (k + 1)^2 - k^2)) :=
sorry

end vasya_guarantee_win_l519_519855


namespace intersection_M_N_l519_519994

noncomputable def M : set ℝ := {x | x^2 - x ≤ 0}
noncomputable def N : set ℝ := {x | x ≠ 0}
noncomputable def intersection : set ℝ := M ∩ N

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by sorry

end intersection_M_N_l519_519994


namespace last_three_digits_of_2_pow_9000_l519_519933

-- The proof statement
theorem last_three_digits_of_2_pow_9000 (h : 2 ^ 300 ≡ 1 [MOD 1000]) : 2 ^ 9000 ≡ 1 [MOD 1000] :=
by
  sorry

end last_three_digits_of_2_pow_9000_l519_519933


namespace turn_on_all_bulbs_l519_519319

-- Define positions of the light bulbs
inductive Position where
  | p00 : Position
  | p01 : Position
  | p10 : Position
  | p11 : Position

-- Define the state of a bulb
inductive BulbState where
  | Off : BulbState
  | On : BulbState

-- Define the initial state of the grid
def initialGridState : Position → BulbState
  | Position.p00 => BulbState.Off
  | Position.p01 => BulbState.Off
  | Position.p10 => BulbState.Off
  | Position.p11 => BulbState.Off

-- Define a move as turning on all bulbs on one side of a line
def makeMove (line : Position → Prop) (state : Position → BulbState) : Position → BulbState :=
  fun p => if line p then BulbState.On else state p

-- Define the condition of being able to turn on all bulbs in exactly 4 moves
def turnOnAllBulbsInFourMoves : Prop :=
  ∃ (line1 line2 line3 line4 : Position → Prop),
    let state1 := makeMove line1 initialGridState
    let state2 := makeMove line2 state1
    let state3 := makeMove line3 state2
    let state4 := makeMove line4 state3
    state4 Position.p00 = BulbState.On ∧
    state4 Position.p01 = BulbState.On ∧
    state4 Position.p10 = BulbState.On ∧
    state4 Position.p11 = BulbState.On

theorem turn_on_all_bulbs : turnOnAllBulbsInFourMoves :=
  sorry

end turn_on_all_bulbs_l519_519319


namespace percent_psychology_majors_enrolled_in_lib_arts_l519_519166

-- Define the total number of students
def total_students : ℕ := 100

-- Define the percentage of students that are freshmen
def percent_freshmen : ℝ := 0.5

-- Define the percentage of freshmen that are international
def percent_international_freshmen : ℝ := 0.3

-- Define the percentage of freshmen that are domestic
def percent_domestic_freshmen : ℝ := 0.7

-- Define the percentage of international freshmen in the school of liberal arts
def percent_int_freshmen_lib_arts : ℝ := 0.4

-- Define the percentage of domestic freshmen in the school of liberal arts
def percent_dom_freshmen_lib_arts : ℝ := 0.35

-- Define the percentage of international freshmen in liberal arts who are psychology majors
def percent_int_lib_arts_psych_majors : ℝ := 0.2

-- Define the percentage of domestic freshmen in liberal arts who are psychology majors
def percent_dom_lib_arts_psych_majors : ℝ := 0.25

-- The theorem to be proved
theorem percent_psychology_majors_enrolled_in_lib_arts :
  let freshman_students := total_students * percent_freshmen in
  let international_freshmen := freshman_students * percent_international_freshmen in
  let domestic_freshmen := freshman_students * percent_domestic_freshmen in
  let int_freshmen_lib_arts := international_freshmen * percent_int_freshmen_lib_arts in
  let dom_freshmen_lib_arts := domestic_freshmen * percent_dom_freshmen_lib_arts in
  let int_freshmen_psych_majors := int_freshmen_lib_arts * percent_int_lib_arts_psych_majors in
  let dom_freshmen_psych_majors := dom_freshmen_lib_arts * percent_dom_lib_arts_psych_majors in
  let total_freshmen_psych_majors := int_freshmen_psych_majors + dom_freshmen_psych_majors in
  total_freshmen_psych_majors / real.of_nat total_students = 0.04 :=
by
  sorry

end percent_psychology_majors_enrolled_in_lib_arts_l519_519166


namespace square_area_ratio_l519_519992

/-- Consider a square ABCD with side length 4. Points E and F are midpoints of sides AB and CD,
respectively, and points G and H are the midpoints of sides BC and DA. Prove that the ratio
of the area of the square EFGH to the area of the square ABCD is 1/4. -/
theorem square_area_ratio (A B C D E F G H : ℝ) (ABCD_square : is_square ABCD 4)
  (E_midpoint : midpoint AB E)
  (F_midpoint : midpoint CD F)
  (G_midpoint : midpoint BC G)
  (H_midpoint : midpoint DA H) :
  area (square E F G H) / area (square A B C D) = 1 / 4 := 
by
  sorry

end square_area_ratio_l519_519992


namespace equation_of_ellipse_product_of_slopes_is_constant_max_area_of_triangle_and_eq_of_line_l519_519620

variables (a b : ℝ) (h1 : a = 2 * b) (h_ab_gt_zero : a > 0 ∧ b > 0)
variables (C D M N l : ℝ × ℝ) (C_coords : C = (2, 1)) (D_coords: D = (-2, -1))
variables (ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
variables (h_passing_through_C : ellipse (2:ℝ) (1:ℝ))
variables (slopes_exist : ∀ P : ℝ × ℝ, P ∈ ellipse → P ≠ (2, 1) → P ≠ (-2, -1))

theorem equation_of_ellipse
  (ha2b : a = 2 * b)
  (passes_through_C : ellipse 2 1) :
  ∃ a b : ℝ, ellipse = λ x y, x^2 / (2 * √2)^2 + y^2 / (√2)^2 = 1 :=
sorry

theorem product_of_slopes_is_constant
  (P : ℝ × ℝ)
  (hP : ellipse P.1 P.2)
  (hP_not_C : P ≠ (2, 1))
  (hP_not_D : P ≠ (-2, -1)) :
  let k_CP := (P.2 - 1) / (P.1 - 2)
      k_DP := (P.2 + 1) / (P.1 + 2) in
  k_CP * k_DP = -1/4 :=
sorry

theorem max_area_of_triangle_and_eq_of_line
  (M N : ℝ × ℝ)
  (line_parallel_to_CD : ∀ x : ℝ, (l.1 : ℝ) = x / 2 + l.2)
  (intersects_ellipse_at_M_N : ellipse M.1 M.2 ∧ ellipse N.1 N.2)
  (line_eq : ∃ t : ℝ, line_parallel_to_CD = λ x, x / 2 + t) :
  ∃ max_area : ℝ, max_area = 2 :=
sorry

end equation_of_ellipse_product_of_slopes_is_constant_max_area_of_triangle_and_eq_of_line_l519_519620


namespace find_base_of_triangle_l519_519312

-- Given conditions of the problem
variables (AB AC AD BC : ℝ) (isosceles_triangle : AB = AC) (side_length : AB = 4) (median_length : AD = 3)

-- Median properties
variables (BD : ℝ) (median_definition : BD = BC / 2)

-- Expected answer
theorem find_base_of_triangle : BC = Real.sqrt 10 :=
by
  -- Utilize the conditions and definitions in the statement
  have h1 : AC = 4 := side_length
  have h2 : AD = 3 := median_length
  sorry

end find_base_of_triangle_l519_519312


namespace conic_section_eccentricity_l519_519286

theorem conic_section_eccentricity (m : ℝ) (h : ∃ x : ℝ, 4^(x + 1/2) - 9 * 2^x + 4 = 0 ∧ (m = 2 ∨ m = -1)) :
  (m = 2 → ∃ e : ℝ, e = real.sqrt 2 / 2) ∧ (m = -1 → ∃ e : ℝ, e = real.sqrt 2) :=
by {
  intros,
  cases h with x hx,
  cases hx.2,
  {
    use real.sqrt 2 / 2,
    exact ⟨rfl⟩,
  },
  {
    use real.sqrt 2,
    exact ⟨rfl⟩,
  }
}

end conic_section_eccentricity_l519_519286


namespace total_seashells_l519_519367

-- Define the conditions from part a)
def unbroken_seashells : ℕ := 2
def broken_seashells : ℕ := 4

-- Define the proof problem
theorem total_seashells :
  unbroken_seashells + broken_seashells = 6 :=
by
  sorry

end total_seashells_l519_519367


namespace acquaintance_paradox_proof_l519_519709

theorem acquaintance_paradox_proof :
  ∃ (x : ℕ), (acquaintances x ≥ 810) :=
by
  let n_population := 1000000
  let n_believers := 900000
  let n_nonBelievers := n_population - n_believers
  have h1 : n_population = 1000000 := rfl
  have h2 : n_believers = 0.90 * n_population := by sorry
  have h3 : n_nonBelievers = n_population - n_believers := by sorry
  have h4 : ∀ x, acquaintances x ≥ 1 := by sorry
  have h5 : ∀ x, 0.10 * acquaintances x = nonBelievers_acquaintances x := by sorry
  have h6 : nonBelievers_acquaintances_total >= 90 := by sorry
  
  show ∃ (x : ℕ), acquaintances x ≥ 810 from sorry
  
  sorry

end acquaintance_paradox_proof_l519_519709


namespace correct_statements_l519_519613

theorem correct_statements :
  let O : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 4 }
      C : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 - 2 * p.1 + 4 * p.2 + 4 = 0 }
      l : set (ℝ × ℝ) := { p | p.1 - 2 * p.2 + 5 = 0 }
      P : ℝ × ℝ := some (p ∈ l)
      is_tangent (p1 p2 : ℝ × ℝ) (C : set (ℝ × ℝ)) : Prop := 
        ∃ t : ℝ, ∃ u : ℝ, 4 * t * u + t + p1.1 + p2.1 = 0 ∧ 4 * u + u - p2.2 = 0
in
  -- Length of segment AB
  let A := (0, -2)
      B := (8/5, -6/5)
      AB_len := real.sqrt ((8/5)^2 + (4/5)^2)
  in AB_len = 4 * real.sqrt 5 / 5 ∧
  -- Line MN fixed point
  let fixed_point := (-4/5, 8/5)
  in ∀ P ∈ l, ∀ (M N : ℝ × ℝ),
    is_tangent P M O → is_tangent P N O → 
    ∃ c : ℝ, M.1 + c * (N.1 - M.1) = -4/5 ∧ M.2 + c * (N.2 - M.2) = 8/5  ∧
  -- Minimum value of |PM|
  let PM_min := 1
  in 
  ∀ P ∈ l, ∃ M : ℝ × ℝ, is_tangent P M O → real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) ≥ 1 :=
sorry

end correct_statements_l519_519613


namespace find_a_and_fg_neg1_l519_519623

def is_odd_fun (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f (a : ℝ) (g : ℝ → ℝ) : ℝ → ℝ :=
  λ x, if 0 < x then x^2 + 2*x - 5 else
       if x = 0 then a else g x

def g (x : ℝ) : ℝ := if x > 0 then x^2 + 2*x - 5 else 0  -- Dummy definition, will be substituted in the proof

theorem find_a_and_fg_neg1 (a : ℝ) (g : ℝ → ℝ) (h_odd : is_odd_fun (f a g))
  (hg_pos : ∀ x, x > 0 → g x = x^2 + 2*x - 5) :
  a = 0 ∧ f a g (g (-1)) = 3 :=
by {
  -- Insert proof here
  sorry
}

end find_a_and_fg_neg1_l519_519623


namespace equation_of_tangent_line_min_max_of_h_l519_519666

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 2)
noncomputable def g (x : ℝ) : ℝ := x / Real.exp 1
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem equation_of_tangent_line :
  ∃ m b, (∀ x, f x = m * x + b) ∧ (f 0 = 2 ∧ m = 2 ∧ b = 2) :=
sorry

theorem min_max_of_h :
  (∀ x ∈ Icc (-2 : ℝ) 0, h (-1) = Real.exp 1⁻¹ 4) ∧ (∀ x ∈ Icc (-2 : ℝ) 0, h 0 = 2) :=
sorry

end equation_of_tangent_line_min_max_of_h_l519_519666


namespace pats_college_years_l519_519693

-- Definitions based on the conditions provided
def interest_rate : ℝ := 0.08 -- 8 percent is 0.08 in decimal
def initial_investment : ℝ := 8000
def final_investment : ℝ := 32000
def doubling_time (r : ℝ) : ℝ := 70 / r
def n_doublings (initial_final_ratio : ℝ) : ℝ := real.logb 2 initial_final_ratio

-- The proof statement for the math problem
theorem pats_college_years :
  let r := interest_rate in
  let t_double := doubling_time r in
  let n := n_doublings (final_investment / initial_investment) in
  n * t_double = 17.5 := 
by
  sorry

end pats_college_years_l519_519693


namespace percentage_increase_l519_519299

variable (x r : ℝ)

theorem percentage_increase (h_x : x = 78.4) (h_r : x = 70 * (1 + r)) : r = 0.12 :=
by
  -- Proof goes here
  sorry

end percentage_increase_l519_519299


namespace sum_of_angles_equal_360_l519_519317

variables (A B C D F G : ℝ)

-- Given conditions.
def is_quadrilateral_interior_sum (A B C D : ℝ) : Prop := A + B + C + D = 360
def split_internal_angles (F G : ℝ) (C D : ℝ) : Prop := F + G = C + D

-- Proof problem statement.
theorem sum_of_angles_equal_360
  (h1 : is_quadrilateral_interior_sum A B C D)
  (h2 : split_internal_angles F G C D) :
  A + B + C + D + F + G = 360 :=
sorry

end sum_of_angles_equal_360_l519_519317


namespace cone_height_l519_519113

-- Definitions based on the conditions in the problem
def volume_of_cone (V : ℝ) (r h : ℝ) := (1 / 3) * Real.pi * r^2 * h
def vertex_angle_isosceles (angle : ℝ) := angle = 90

-- Problem statement
theorem cone_height:
  (volume_of_cone 19683 * Real.pi r h) ∧ (vertex_angle_isosceles 90) ∧ (h = r) → h = 39.0 :=
by
  sorry

end cone_height_l519_519113


namespace difference_fraction_reciprocal_l519_519415

theorem difference_fraction_reciprocal :
  let f := (4 : ℚ) / 5
  let r := (5 : ℚ) / 4
  f - r = 9 / 20 :=
by
  sorry

end difference_fraction_reciprocal_l519_519415


namespace no_prime_divisible_by_77_l519_519279

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  sorry

end no_prime_divisible_by_77_l519_519279


namespace triangle_area_eq_l519_519705

variable (a b c: ℝ) (A B C : ℝ)
variable (h_cosC : Real.cos C = 1/4)
variable (h_c : c = 3)
variable (h_ratio : a / Real.cos A = b / Real.cos B)

theorem triangle_area_eq : (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 15 / 4 :=
by
  sorry

end triangle_area_eq_l519_519705


namespace range_difference_l519_519625

open interval

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem range_difference (a b : ℝ) (h₁ : ∀ x, a ≤ x ∧ x ≤ b → f(x) ∈ Icc (-1) 3) :
  Icc 2 4 (b - a) :=
by
  sorry

end range_difference_l519_519625


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519727

theorem exists_inhabitant_with_at_least_810_acquaintances :
  (∀ x, ∃ y, x ≠ y) → 
  (∀ x, (∀ k, k = 10% * |neighbors x| → |{y : neighbors x | believes_in_santa y}| = k)) →
  (1_000_000 ≥ ∑ x, believes_in_santa x → 900_000) → 
  (1_000_000 - 900_000 = 100_000) →
  (∃ x, |neighbors x| ≥ 810) :=
sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519727


namespace domain_log_base2_l519_519878

theorem domain_log_base2 (x : ℝ) : x - 1 > 0 ↔ x ∈ set.Ioi 1 :=
by sorry

end domain_log_base2_l519_519878


namespace exists_inhabitant_with_810_acquaintances_l519_519768

-- Definition of the main problem conditions
def inhabitants := 1000000
def belief_in_santa := 0.9 * inhabitants
def disbelief_in_santa := inhabitants - belief_in_santa

-- Conditions
def knows_at_least_one : ∀ x : ℕ, x < inhabitants → ∃ y : ℕ, y < inhabitants ∧ x ≠ y := sorry
def ten_percent_acquaintances (x : ℕ) (h : x < inhabitants) : Prop := ∀ y : ℕ, y < inhabitants ∧ x ≠ y → (y < belief_in_santa → y < 0.1 * x) := sorry

-- Main theorem
theorem exists_inhabitant_with_810_acquaintances :
  ∃ x : ℕ, x < inhabitants ∧ (knows_at_least_one x) ∧ (ten_percent_acquaintances x) ∧ (810 ≤ x) :=
sorry

end exists_inhabitant_with_810_acquaintances_l519_519768


namespace line_equation_correct_l519_519882

noncomputable def x_intercept : ℝ := 2
noncomputable def inclination_angle : ℝ := 135
noncomputable def line_equation (x : ℝ) : ℝ := -x + 2

theorem line_equation_correct :
  (∀ x : ℝ, (x, line_equation x) ∈ {p : ℝ × ℝ | p.1 = x_intercept ∨ tan (inclination_angle * real.pi / 180) * (p.1 - x_intercept) = p.2}) :=
by
  sorry

end line_equation_correct_l519_519882


namespace number_of_tiles_l519_519047

theorem number_of_tiles (room_width room_height tile_width tile_height : ℝ) :
  room_width = 8 ∧ room_height = 12 ∧ tile_width = 1.5 ∧ tile_height = 2 →
  (room_width * room_height) / (tile_width * tile_height) = 32 :=
by
  intro h
  cases' h with rw h
  cases' h with rh h
  cases' h with tw th
  rw [rw, rh, tw, th]
  norm_num
  sorry

end number_of_tiles_l519_519047


namespace transportable_load_l519_519503

theorem transportable_load 
  (mass_of_load : ℝ) 
  (num_boxes : ℕ) 
  (box_capacity : ℝ) 
  (num_trucks : ℕ) 
  (truck_capacity : ℝ) 
  (h1 : mass_of_load = 13.5) 
  (h2 : box_capacity = 0.35) 
  (h3 : truck_capacity = 1.5) 
  (h4 : num_trucks = 11)
  (boxes_condition : ∀ (n : ℕ), n * box_capacity ≥ mass_of_load) :
  mass_of_load ≤ num_trucks * truck_capacity :=
by
  sorry

end transportable_load_l519_519503


namespace solve_fractional_equation_l519_519396

theorem solve_fractional_equation :
  ∀ x : ℝ, (4 / (x^2 + x) - 3 / (x^2 - x) = 0) → x = 7 :=
by
  intro x h,
  sorry

end solve_fractional_equation_l519_519396


namespace sum_of_a_and_b_is_24_l519_519527

theorem sum_of_a_and_b_is_24 
  (a b : ℕ) 
  (h_a_pos : a > 0) 
  (h_b_gt_one : b > 1) 
  (h_maximal : ∀ (a' b' : ℕ), (a' > 0) → (b' > 1) → (a'^b' < 500) → (a'^b' ≤ a^b)) :
  a + b = 24 := 
sorry

end sum_of_a_and_b_is_24_l519_519527


namespace original_number_increased_l519_519955

theorem original_number_increased (x : ℝ) (h : (1.10 * x) * 1.15 = 632.5) : x = 500 :=
sorry

end original_number_increased_l519_519955


namespace august_five_thursdays_if_july_five_mondays_l519_519398

noncomputable def july_has_five_mondays (N : ℕ) : Prop := ∃ m : ℕ, m ∈ {1, 8, 15, 22, 29, 2, 9, 16, 23, 30, 3, 10, 17, 24, 31} ∧ ∀ i ∈ {1, 8, 15, 22, 29, 2, 9, 16, 23, 30, 3, 10, 17, 24, 31}.erase m, i % 7 = 1

noncomputable def august_has_five_thursdays (N : ℕ) : Prop := ∃ n : ℕ, n % 7 = 3 ∧ ∀ k : ℕ, k < 31 → (k + n) % 7 ≠ 3

theorem august_five_thursdays_if_july_five_mondays (N : ℕ) (h : july_has_five_mondays N):
  august_has_five_thursdays N := 
sorry

end august_five_thursdays_if_july_five_mondays_l519_519398


namespace train_crossing_time_l519_519796

theorem train_crossing_time (len : ℝ) (speed_kmph : ℝ) (conversion_factor : ℝ) (time_approx : ℝ) : 
  len = 200 ∧ 
  speed_kmph = 124 ∧ 
  conversion_factor = 1000 / 3600 ∧ 
  time_approx ≈ 5.81 →
  (time_approx ≈ (len / (speed_kmph * conversion_factor))) := by
    sorry

end train_crossing_time_l519_519796


namespace find_loan_term_l519_519959

noncomputable def total_sum : ℝ := 2678
noncomputable def second_sum : ℝ := 1648
noncomputable def first_sum : ℝ := total_sum - second_sum
noncomputable def interest_rate_first : ℝ := 3 / 100
noncomputable def interest_rate_second : ℝ := 5 / 100
noncomputable def term_second : ℝ := 3
noncomputable def interest_first (n : ℝ) : ℝ := (first_sum * interest_rate_first * n)
noncomputable def interest_second : ℝ := (second_sum * interest_rate_second * term_second)

theorem find_loan_term : ∃ n : ℝ, interest_first n = interest_second ∧ n = 8 :=
by
  use 8
  have h1 : first_sum = 2678 - 1648 := rfl
  have h2 : first_sum = 1030 := by rwa [←h1]
  have h3 : interest_first 8 = interest_second := by
    rw [interest_first, interest_second, h2]
    norm_num
  exact ⟨h3, rfl⟩

end find_loan_term_l519_519959


namespace exists_inhabitant_with_810_acquaintances_l519_519751

-- Conditions
def total_population : ℕ := 1000000
def believers_percentage : ℕ := 90
def acquaintances_believers_percentage : ℝ := 0.10

-- Theorem statement: Prove that there exists an inhabitant knowing at least 810 people.
theorem exists_inhabitant_with_810_acquaintances 
    (H1 : ∀ (p : ℕ), p < total_population → ∃ q, q ≠ p)  -- Every person knows at least one other person
    (H2 : 0.9 * total_population = 900000)  -- 90% of the population are believers in Santa Claus
    (H3 : ∀ p, p < total_population → ∀ q, q != p → (belief p → (exists percentage, percentage = acquaintances_believers_percentage))) -- Every person claims exactly 10% of their acquaintances believe in Santa Claus
    : ∃ (habitants : ℕ), habitants ≥ 810
    :=
  sorry

end exists_inhabitant_with_810_acquaintances_l519_519751


namespace part1_part2_l519_519629

def f (x : ℝ) (a : ℝ) := (Real.exp x) / x - Real.log x + x - a

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → (a ≤ Real.exp 1 + 1) :=
sorry

theorem part2 (x1 x2 a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 ≠ x2 → 0 < x1 → 0 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519629


namespace length_of_PN_l519_519321

theorem length_of_PN {K L M N Q P : Point} 
  (hT : IsTrapezoid K L M N)
  (hBases : length K N = 12 ∧ length L M = 3)
  (hQonMN : OnSegment Q M N)
  (hPerp : Perpendicular Q P K L)
  (hMidpoint : Midpoint P K L)
  (hPM : length P M = 4)
  (hArea : Area (Quadrilateral P L M Q) = (1 / 4) * Area (Quadrilateral P K N Q)) :
  length P N = 16 := 
sorry

end length_of_PN_l519_519321


namespace isosceles_triangle_similarity_l519_519160

theorem isosceles_triangle_similarity (T1 T2 : Triangle) 
  (iso1 : is_isosceles T1)
  (iso2 : is_isosceles T2)
  (angle_60_T1 : ∃ A ∈ interior_angles T1, A = 60)
  (angle_60_T2 : ∃ A ∈ interior_angles T2, A = 60) :
  similar T1 T2 :=
sorry

end isosceles_triangle_similarity_l519_519160


namespace bogan_feeding_total_maggots_l519_519563

theorem bogan_feeding_total_maggots :
  let first_feeding := 10
  let second_feeding := 15
  let third_feeding := second_feeding * 2
  let fourth_feeding := third_feeding - 5
  (first_feeding + second_feeding + third_feeding + fourth_feeding) = 80 :=
by {
  let first_feeding := 10,
  let second_feeding := 15,
  let third_feeding := second_feeding * 2,
  let fourth_feeding := third_feeding - 5,
  calc
  first_feeding + second_feeding + third_feeding + fourth_feeding
      = 10 + 15 + third_feeding + fourth_feeding : by rfl
  ... = 10 + 15 + (second_feeding * 2) + (third_feeding - 5) : by rfl
  ... = 10 + 15 + 30 + 25 : by rfl
  ... = 80 : by rfl
}

end bogan_feeding_total_maggots_l519_519563


namespace volume_of_tetrahedron_is_half_l519_519877

-- Definitions based on given problem conditions
variables (A B C I D : Point) 
variable h : ℝ -- height AD
variable angle_AID : ℝ -- dihedral angle AID

-- Conditions from the problem
axiom AD_eq_2 : h = 2
axiom angle_AID_eq_pi_div_3 : angle_AID = π / 3

-- Goal: volume of tetrahedron A B C I is 1/2
theorem volume_of_tetrahedron_is_half : 
  volume_of_tetrahedron A B C I = 1 / 2 :=
begin
  -- Proof steps to be filled
  sorry
end

end volume_of_tetrahedron_is_half_l519_519877


namespace closest_integer_to_fourth_root_l519_519454

theorem closest_integer_to_fourth_root : 
  let a := 15
  let b := 10
  let sum := a^4 + b^4
  ∃ k : ℤ, k = 15 ∧ abs (k - real.sqrt fi $sum^4) < 1 :=
begin
    sorry
end

end closest_integer_to_fourth_root_l519_519454


namespace find_k_l519_519846

theorem find_k 
  (k : ℝ)
  (p_eq : ∀ x : ℝ, (4 * x + 3 = k * x - 9) → (x = -3 → (k = 0)))
: k = 0 :=
by sorry

end find_k_l519_519846


namespace max_ratio_l519_519217

-- Conditions
def is_right_triangle (A B C : Point) : Prop :=
  ∠B = 60 ∧ ∠C = 30 ∧ ∠A = 90

def is_isosceles_trapezium (P Q R B : Point) : Prop :=
  P.seg = Q.seg ∧ PQ.parallel BR

-- Given points A, B, C, P, Q, R with the conditions above
variables (A B C P Q R : Point)

-- Definition of areas
def area_triangle (A B C : Point) : ℝ := sorry
def area_trapezium (P Q R B : Point) : ℝ := sorry

-- The theorem
theorem max_ratio (h1 : is_right_triangle A B C) 
                (h2 : is_isosceles_trapezium P Q R B) :
  (2 * area_triangle A B C) / (area_trapezium P Q R B) ≤ 4 := sorry

end max_ratio_l519_519217


namespace exists_inhabitant_with_810_acquaintances_l519_519750

-- Conditions
def total_population : ℕ := 1000000
def believers_percentage : ℕ := 90
def acquaintances_believers_percentage : ℝ := 0.10

-- Theorem statement: Prove that there exists an inhabitant knowing at least 810 people.
theorem exists_inhabitant_with_810_acquaintances 
    (H1 : ∀ (p : ℕ), p < total_population → ∃ q, q ≠ p)  -- Every person knows at least one other person
    (H2 : 0.9 * total_population = 900000)  -- 90% of the population are believers in Santa Claus
    (H3 : ∀ p, p < total_population → ∀ q, q != p → (belief p → (exists percentage, percentage = acquaintances_believers_percentage))) -- Every person claims exactly 10% of their acquaintances believe in Santa Claus
    : ∃ (habitants : ℕ), habitants ≥ 810
    :=
  sorry

end exists_inhabitant_with_810_acquaintances_l519_519750


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519724

theorem exists_inhabitant_with_at_least_810_acquaintances :
  (∀ x, ∃ y, x ≠ y) → 
  (∀ x, (∀ k, k = 10% * |neighbors x| → |{y : neighbors x | believes_in_santa y}| = k)) →
  (1_000_000 ≥ ∑ x, believes_in_santa x → 900_000) → 
  (1_000_000 - 900_000 = 100_000) →
  (∃ x, |neighbors x| ≥ 810) :=
sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519724


namespace quadrilateral_fourth_side_length_l519_519510

-- Defining the given conditions for the problem
variable (radius : ℝ) (a b c d ab bc cd : ℝ)
variable (inscribed_in_circle : true)
variable (side_lengths : ab = 200 ∧ bc = 200 ∧ cd = 200 ∧ radius = 200 * Real.sqrt 2)

-- Formalize the goal: proving the length of the fourth side \(AD\)
theorem quadrilateral_fourth_side_length : ∃ ad : ℝ, ad = 500 :=
by
  -- Declare the variables and conditions
  intros ab bc cd radius inscribed_in_circle side_lengths
  -- Use the side lengths and circle radius to show AD = 500
  sorry

end quadrilateral_fourth_side_length_l519_519510


namespace no_linear_relationship_recalculate_stats_l519_519496

section SalarySeniority

-- Definitions based on conditions
def x̄ : ℝ := 9.97
def s : ℝ := 0.212
def seniority_variance_sqrt_sum : ℝ := 18.439
def covariance_salary_seniority : ℝ := -2.78

-- Problem 1: Correlation Coefficient
theorem no_linear_relationship :
  let r := covariance_salary_seniority / (s * sqrt 16 * seniority_variance_sqrt_sum) in
  abs r < 0.25 :=
by
  let r := covariance_salary_seniority / (s * sqrt 16 * seniority_variance_sqrt_sum)
  have h : r ≈ -0.18 := sorry  -- Approximation handled mathematically
  have abs_r := abs r
  have h_abs_r : abs_r < 0.25 := sorry  -- Based on the calculation above
  exact h_abs_r

-- Problem 2: Recalculate Mean and Standard Deviation
theorem recalculate_stats (annual_salaries : Fin 16 → ℝ)
  (H : annual_salaries 12 = 9.22) : -- Note: indices start from 0, so 12 corresponds to the 13th employee
  let new_salaries := fin_remap annual_salaries (Fin 1 16) 12
      new_mean := (∑ i, new_salaries i.toNat) / 15
      new_stddev := sqrt ((∑ i, (new_salaries i.toNat - new_mean)^2) / 15) in
  new_mean = 10.02 ∧ new_stddev ≈ 0.09 :=
by
  let new_salaries := fin_remap annual_salaries (Fin 15) 12
  have new_mean := ∑ i, new_salaries i.toNat / 15
  have new_stddev := sqrt ((∑ i, (new_salaries i.toNat - new_mean)^2) / 15)
  have h_mean : new_mean = 10.02 := sorry  -- Based on the calculation above
  have h_stddev : new_stddev ≈ 0.09 := sorry  -- Based on the calculation above
  exact ⟨h_mean, h_stddev⟩

end SalarySeniority

end no_linear_relationship_recalculate_stats_l519_519496


namespace digit_swap_division_l519_519514

theorem digit_swap_division (ab ba : ℕ) (k1 k2 : ℤ) (a b : ℕ) :
  (ab = 10 * a + b) ∧ (ba = 10 * b + a) →
  (ab % 7 = 1) ∧ (ba % 7 = 1) →
  ∃ n, n = 4 :=
by
  sorry

end digit_swap_division_l519_519514


namespace initial_rectangles_are_squares_l519_519314

theorem initial_rectangles_are_squares (n : ℕ) (h₁ : n > 1) 
  (h₂ : ∀ (rect : ℕ), rect = 1) 
  (total_squares_prime : ∀ (total : ℕ), prime total) : 
  ∀ (a b : ℕ), a = b :=
by
  sorry

end initial_rectangles_are_squares_l519_519314


namespace distinct_weights_count_l519_519439

theorem distinct_weights_count (n : ℕ) (h : n = 4) : 
  -- Given four weights and a two-pan balance scale without a pointer,
  ∃ m : ℕ, 
  -- prove that the number of distinct weights of cargo
  (m = 40) ∧  
  -- that can be exactly measured if the weights can be placed on both pans of the scale is 40.
  m = 3^n - 1 ∧ (m / 2 = 40) := by
  sorry

end distinct_weights_count_l519_519439


namespace area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l519_519324

-- Definitions and conditions
variable {A B C a b c : ℝ}
variable (cosA : ℝ) (sinA : ℝ)
variable (area : ℝ)
variable (tanA tanB tanC : ℝ)

-- Given conditions
axiom angle_identity : b^2 + c^2 = 3 * b * c * cosA
axiom sin_cos_identity : sinA^2 + cosA^2 = 1
axiom law_of_cosines : a^2 = b^2 + c^2 - 2 * b * c * cosA

-- Part (1) statement
theorem area_of_triangle_is_sqrt_5 (B_eq_C : B = C) (a_eq_2 : a = 2) 
    (cosA_eq_2_3 : cosA = 2/3) 
    (b_eq_sqrt6 : b = Real.sqrt 6) 
    (sinA_eq_sqrt5_3 : sinA = Real.sqrt 5 / 3) 
    : area = Real.sqrt 5 := sorry

-- Part (2) statement
theorem sum_of_tangents_eq_1 (tanA_eq : tanA = sinA / cosA)
    (tanB_eq : tanB = sinA * sinA / (cosA * cosA))
    (tanC_eq : tanC = sinA * sinA / (cosA * cosA))
    : (tanA / tanB) + (tanA / tanC) = 1 := sorry

end area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l519_519324


namespace exists_integer_combination_of_vectors_l519_519599

theorem exists_integer_combination_of_vectors
  (O : ℝ × ℝ)
  (A : ℕ → ℝ × ℝ)
  (h : ∀ i j : ℕ, ∃ n : ℕ, dist (A i) (A j) = real.sqrt n) :
  ∃ x y : ℝ × ℝ, ∀ i : ℕ, ∃ k l : ℤ, (A i).fst - O.fst = k * x.fst + l * y.fst ∧ (A i).snd - O.snd = k * x.snd + l * y.snd :=
sorry

end exists_integer_combination_of_vectors_l519_519599


namespace sum_of_digits_n_l519_519056

theorem sum_of_digits_n
  (n : ℕ)
  (h : (n + 1)! + (n + 3)! = n! * 1320) :
  n = 9 :=
by
  sorry

end sum_of_digits_n_l519_519056


namespace length_after_y_months_isabella_hair_length_l519_519798

-- Define the initial length of the hair
def initial_length : ℝ := 18

-- Define the growth rate of the hair per month
def growth_rate (x : ℝ) : ℝ := x

-- Define the number of months passed
def months_passed (y : ℕ) : ℕ := y

-- Prove the length of the hair after 'y' months
theorem length_after_y_months (x : ℝ) (y : ℕ) : ℝ :=
  initial_length + growth_rate x * y

-- Theorem statement to prove that the length of Isabella's hair after y months is 18 + xy
theorem isabella_hair_length (x : ℝ) (y : ℕ) : length_after_y_months x y = 18 + x * y :=
by sorry

end length_after_y_months_isabella_hair_length_l519_519798


namespace total_pages_in_novel_l519_519864

-- Given conditions as definitions
def fraction_read_yesterday := (3 : ℚ) / 10
def fraction_read_today := (4 : ℚ) / 10
def total_pages_read := 140

-- Proposition to prove
theorem total_pages_in_novel : 
  let total_fraction := fraction_read_yesterday + fraction_read_today in
  total_fraction = 7 / 10 → total_pages_read = 140 → 
  (140 / 7) * 10 = 200 :=
by
  intro h1 h2
  rw [h1, h2]
  sorry -- skip the proof

end total_pages_in_novel_l519_519864


namespace quadratic_to_square_l519_519701

theorem quadratic_to_square (x h k : ℝ) : 
  (x * x - 4 * x + 3 = 0) →
  ((x + h) * (x + h) = k) →
  k = 1 :=
by
  sorry

end quadratic_to_square_l519_519701


namespace physics_marks_l519_519936

theorem physics_marks (P C M : ℕ) 
  (h1 : P + C + M = 195)
  (h2 : P + M = 180)
  (h3 : P + C = 140) :
  P = 125 :=
by {
  sorry
}

end physics_marks_l519_519936


namespace det_A_eq_neg15_l519_519351

open Matrix

variable {x y : ℝ}
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![x, 3], ![-4, y]]
def B := 3 • A⁻¹
def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem det_A_eq_neg15 (h : A + B = I) : det A = -15 :=
by
  sorry

end det_A_eq_neg15_l519_519351


namespace solve_congruence_l519_519395

theorem solve_congruence (n : ℕ) (h₀ : 0 ≤ n ∧ n < 47) (h₁ : 13 * n ≡ 5 [MOD 47]) :
  n = 4 :=
sorry

end solve_congruence_l519_519395


namespace exists_inhabitant_with_810_acquaintances_l519_519769

-- Definition of the main problem conditions
def inhabitants := 1000000
def belief_in_santa := 0.9 * inhabitants
def disbelief_in_santa := inhabitants - belief_in_santa

-- Conditions
def knows_at_least_one : ∀ x : ℕ, x < inhabitants → ∃ y : ℕ, y < inhabitants ∧ x ≠ y := sorry
def ten_percent_acquaintances (x : ℕ) (h : x < inhabitants) : Prop := ∀ y : ℕ, y < inhabitants ∧ x ≠ y → (y < belief_in_santa → y < 0.1 * x) := sorry

-- Main theorem
theorem exists_inhabitant_with_810_acquaintances :
  ∃ x : ℕ, x < inhabitants ∧ (knows_at_least_one x) ∧ (ten_percent_acquaintances x) ∧ (810 ≤ x) :=
sorry

end exists_inhabitant_with_810_acquaintances_l519_519769


namespace smallest_integer_consecutive_set_l519_519305

theorem smallest_integer_consecutive_set :
  ∃ m : ℤ, (m+3 < 3*m - 5) ∧ (∀ n : ℤ, (n+3 < 3*n - 5) → n ≥ m) ∧ m = 5 :=
by
  sorry

end smallest_integer_consecutive_set_l519_519305


namespace coin_flip_probability_l519_519692

theorem coin_flip_probability :
  let p := 1 / 2 in
  let num_flips := 4 in
  let prob_first_two_tails := p * p in
  let prob_last_two_not_tails := (1 - p) * (1 - p) in
  prob_first_two_tails * prob_last_two_not_tails = 1 / 16 :=
by
  have p := 1 / 2
  have num_flips := 4
  have prob_first_two_tails := p * p
  have prob_last_two_not_tails := (1 - p) * (1 - p)
  calc
    prob_first_two_tails * prob_last_two_not_tails
      = (p * p) * ((1 - p) * (1 - p)) : by rfl
    ... = (1 / 2 * 1 / 2) * ((1 - 1 / 2) * (1 - 1 / 2)) : by rfl
    ... = (1 / 4) * (1 / 4) : by simp
    ... = 1 / 16 : by norm_num

sorry

end coin_flip_probability_l519_519692


namespace trig_identity_l519_519983

theorem trig_identity :
  (tan 150 * cos (-210) * sin (-420)) / (sin 1050 * cos (-600)) = -sqrt 3 :=
by
  sorry

end trig_identity_l519_519983


namespace max_value_k_min_value_7a_4b_l519_519591

-- Question (1)
theorem max_value_k (x k : ℝ) (h : ∀ x, |x + 2| + |6 - x| ≥ k) : k ≤ 8 :=
begin
  have g_min : ∀ x, |x + 2| + |6 - x| ≥ 8, 
  { intro x,
    calc |x + 2| + |6 - x| ≥ |(x + 2) + (6 - x)| : by apply abs_add
                        ... = 8 : by linarith },
  have : 8 ≤ k, from h,
  exact le_trans (le_refl 8) h,
end

-- Question (2)
theorem min_value_7a_4b (a b : ℝ) (n : ℝ) (h₁ : n = 8) (h₂ : 8 / (5 * a + b) + 2 / (2 * a + 3 * b) = n) : 7 * a + 4 * b ≥ 9 / 4 :=
begin
  have : 8 / (5 * a + b) + 2 / (2 * a + 3 * b) = 8, 
  { linarith },
  calc 7 * a + 4 * b 
        = (1 / 4) * (7 * a + 4 * b) * (8 / (5 * a + b) + 2 / (2 * a + 3 * b)) : by sorry
      ... ≥ 9 / 4 : by sorry
end

end max_value_k_min_value_7a_4b_l519_519591


namespace possible_third_altitude_l519_519619

variable {α : Type*} [LinearOrderedField α]

theorem possible_third_altitude (h1 : α = 4) (h2 : α = 12) (h3 : ∃ x : α, x ∈ ℤ) :
  { x : ℤ | 3 < x ∧ x < 6 } = {4, 5} :=
by
  sorry

end possible_third_altitude_l519_519619


namespace ratio_of_female_to_male_members_l519_519525

theorem ratio_of_female_to_male_members 
  (f m : ℕ)
  (avg_age_female avg_age_male avg_age_membership : ℕ)
  (hf : avg_age_female = 35)
  (hm : avg_age_male = 30)
  (ha : avg_age_membership = 32)
  (h_avg : (35 * f + 30 * m) / (f + m) = 32) : 
  f / m = 2 / 3 :=
sorry

end ratio_of_female_to_male_members_l519_519525


namespace projection_a_sub_b_onto_b_l519_519243

variables {R : Type*} [IsROrC R] (a b : R^3)
variables [normed_group R] [cmetric_space R]

-- Definitions of unit vectors
def is_unit_vector (v : R^3) : Prop := ∥v∥ = 1

-- Given conditions
variables (unit_a : is_unit_vector a) (unit_b : is_unit_vector b)
variable (angle_ab : real.angle a b = real.pi / 3)

-- Projection of the vector (a - b) onto b
def projection (u v : R^3) : R^3 := ((inner u v) / (inner v v)) • v

theorem projection_a_sub_b_onto_b :
  projection (a - b) b = (-3 / 2) • b :=
sorry

end projection_a_sub_b_onto_b_l519_519243


namespace find_angle_A_find_area_l519_519704

variables (a b c : ℝ) (A B C : ℝ)
variables (sinB sinC : ℝ)

-- Condition: 2b - c = 2a * cos C
axiom h1 : 2 * b - c = 2 * a * Real.cos C

-- Condition: a = sqrt 3
axiom h2 : a = Real.sqrt 3

-- Condition: sin B + sin C = 6 * sqrt 2 * sin B * sin C
axiom h3 : sinB + sinC = 6 * Real.sqrt 2 * sinB * sinC

theorem find_angle_A (h1 : 2 * b - c = 2 * a * Real.cos C) :
  A = Real.pi / 3 :=
sorry

theorem find_area (h2 : a = Real.sqrt 3) (h3 : sinB + sinC = 6 * Real.sqrt 2 * sinB * sinC) :
  let bc : ℝ := 1 / 2
  in (Real.sqrt 3 / 8) = 0 :=
sorry

end find_angle_A_find_area_l519_519704


namespace infinite_circles_through_A_B_inside_C_l519_519965

-- Define the main problem setup
structure Circle :=
(center : Point)
(radius : ℝ)

structure Point :=
(x : ℝ)
(y : ℝ)

variables (C : Circle) (A B : Point)

-- Define the conditions
def circle_contains (C : Circle) (P : Point) : Prop :=
  (P.x - C.center.x)^2 + (P.y - C.center.y)^2 ≤ C.radius^2

def bisector_contains (A B P : Point) : Prop :=
  (P.x - A.x) * (B.y - A.y) = (P.y - A.y) * (B.x - A.x)

-- Main statement to be proven
theorem infinite_circles_through_A_B_inside_C :
  (circle_contains C A) → (circle_contains C B) → 
  ∃ (f : ℕ → Point), ∀ n, bisector_contains A B (f n) ∧ circle_contains C (f n) :=
sorry

end infinite_circles_through_A_B_inside_C_l519_519965


namespace roberto_outfit_combinations_l519_519016

-- Define the components of the problem
def trousers_count : ℕ := 5
def shirts_count : ℕ := 7
def jackets_count : ℕ := 4
def disallowed_combinations : ℕ := 7

-- Define the requirements
theorem roberto_outfit_combinations :
  (trousers_count * shirts_count * jackets_count) - disallowed_combinations = 133 := by
  sorry

end roberto_outfit_combinations_l519_519016


namespace curve_standard_eq_line_cartesian_eq_max_distance_C_to_l_l519_519789
-- Importing the necessary Mathlib core library to incorporate various required modules.

-- Given conditions about curve and line.
def curve_parametric (θ : ℝ) : ℝ × ℝ := (√3 * cos θ, sin θ)
def line_polar (θ ρ: ℝ) : Prop := ρ * cos (θ - π / 4) = 2 * √2

-- Required proofs.
theorem curve_standard_eq (x y : ℝ) (θ : ℝ) (h : (x, y) = curve_parametric θ) :
  x^2 / 3 + y^2 = 1 := 
sorry

theorem line_cartesian_eq (x y : ℝ) (ρ : ℝ) (θ : ℝ) (h : (ρ, θ) = (sqrt(x^2 + y^2), atan2 y x)) (h_polar: line_polar θ ρ) :
  x + y - 4 = 0 := 
sorry

theorem max_distance_C_to_l (θ : ℝ) :
  ∃ p : ℝ × ℝ, p = curve_parametric θ ∧ (p.fst + p.snd - 4) / sqrt 2 = 3 * sqrt 2 :=
sorry

end curve_standard_eq_line_cartesian_eq_max_distance_C_to_l_l519_519789


namespace range_of_a_product_of_zeros_l519_519663

def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - (Real.log x) + x - a

theorem range_of_a (a : ℝ) : (∀ x, (0 < x) → (f x a ≥ 0)) ↔ (a ≤ Real.exp 1 + 1) :=
by
  sorry

theorem product_of_zeros (x1 x2 a : ℝ) : 
  (0 < x1) → (x1 < 1) → (1 < x2) → (f x1 a = 0 ∧ f x2 a = 0) → x1 * x2 < 1 :=
by
  sorry

end range_of_a_product_of_zeros_l519_519663


namespace smallest_t_no_H_Route_l519_519481

def isHorseMove (m n : ℕ) (x y : ℕ × ℕ) (endX endY : ℕ × ℕ) : Prop :=
  (endX = (x + m, y + n)) ∨ (endX = (x + n, y + m)) ∨ (endX = (x - m, y - n)) ∨ (endX = (x - n, y - m))
  
def validHorseRoute (t : ℕ) : Prop :=
  -- Define a sequence of 64 distinct positions in an 8x8 grid, visited exactly once
  ∃ (route : Fin 64 → ℕ × ℕ) (positions : set (ℕ × ℕ)),
    (∀ i, route i ∈ positions) ∧
    (∃ startPos, route 0 = startPos) ∧ -- Starting position
    (∀ i < 63, isHorseMove t (t + 1) (route i) (route (i+1)))

theorem smallest_t_no_H_Route : ∃ t : ℕ, t = 2 ∧ ¬ validHorseRoute t :=
by {
  sorry -- Proof goes here
}

end smallest_t_no_H_Route_l519_519481


namespace no_natural_number_with_sum_of_squares_of_smallest_divisors_is_perfect_square_l519_519570

theorem no_natural_number_with_sum_of_squares_of_smallest_divisors_is_perfect_square :
  ∀ n : ℕ, ¬ ∃ s : ℕ, 
  let divisors := (multiset.take 5 (multiset.sort (≤) (multiset.to_finset (finset.image nat.factorization.divisor_list n)))) in
  divisors.card = 5 ∧ 
  (divisors.map (λ d, d^2)).sum = s^2 :=
by sorry

end no_natural_number_with_sum_of_squares_of_smallest_divisors_is_perfect_square_l519_519570


namespace problem_statement_l519_519784

variables (J K L M N O : Type) [Inhabited J] [Inhabited K] [Inhabited L] [Inhabited M] [Inhabited N] [Inhabited O]

noncomputable def bisection_area_ratio (JL JM ML NO MO NMO_area JML_area : ℝ) (h1 : JL = NO) (h2 : JM = 3*ML) (h3 : JML_area = JM * ML)
  (h4 : ∠JML = real.pi / 2) (h5 : ∠JMO = real.pi / 4) (h6 : ∠NMO = real.pi / 2) : Real :=
  NMO_area / JML_area = 1/(12*Real.sqrt 2)

theorem problem_statement (JL JM ML NO MO : ℝ) (h1 : JL = NO) (h2 : JM = 3*ML) (h3 : ∠JML = real.pi / 2) (h4 : ∠JMO = real.pi / 4) 
  (h5 : ∠NMO = real.pi / 2) : ∃ (NMO_area JML_area : ℝ), bisection_area_ratio J K L M N O JL JM ML NO MO NMO_area JML_area h1 h2 h3 h4 h5 h6 :=
sorry

end problem_statement_l519_519784


namespace y_coordinate_of_third_vertex_eq_l519_519162

theorem y_coordinate_of_third_vertex_eq (x1 x2 y1 y2 : ℝ)
    (h1 : x1 = 0) 
    (h2 : y1 = 3) 
    (h3 : x2 = 10) 
    (h4 : y2 = 3) 
    (h5 : x1 ≠ x2) 
    (h6 : y1 = y2) 
    : ∃ y3 : ℝ, y3 = 3 + 5 * Real.sqrt 3 := 
by
  sorry

end y_coordinate_of_third_vertex_eq_l519_519162


namespace rhombus_longer_diagonal_l519_519144

theorem rhombus_longer_diagonal
  (a b : ℝ)
  (ha : a = 65) -- Side length
  (hb : b = 28) -- Half the length of the shorter diagonal
  : 2 * Real.sqrt (a^2 - b^2) = 118 :=  -- Statement of the longer diagonal length
by
  sorry -- Proof goes here

end rhombus_longer_diagonal_l519_519144


namespace minimum_positive_period_of_f_l519_519049

noncomputable def f (x : ℝ) : ℝ := abs (Real.sin (2 * x) + Real.sin (3 * x) + Real.sin (4 * x))

theorem minimum_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
begin
  use 2 * Real.pi,
  split,
  { exact Real.pi_pos.mul_two },
  sorry
end

end minimum_positive_period_of_f_l519_519049


namespace part1_part2_l519_519647

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519647


namespace problem_a_problem_b_problem_c_problem_d_l519_519932

def rotate (n : Nat) : Nat := 
  sorry -- Function definition for rotating the last digit to the start
def add_1001 (n : Nat) : Nat := 
  sorry -- Function definition for adding 1001
def subtract_1001 (n : Nat) : Nat := 
  sorry -- Function definition for subtracting 1001

theorem problem_a :
  ∃ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) ∧ (List.foldl (λacc step => step acc) 202122 steps = 313233) :=
sorry

theorem problem_b :
  ∃ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) ∧ (steps.length = 8) ∧ (List.foldl (λacc step => step acc) 999999 steps = 000000) :=
sorry

theorem problem_c (n : Nat) (hn : n % 11 = 0) : 
  ∀ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) → (List.foldl (λacc step => step acc) n steps) % 11 = 0 :=
sorry

theorem problem_d : 
  ∀ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) → ¬(List.foldl (λacc step => step acc) 112233 steps = 000000) :=
sorry

end problem_a_problem_b_problem_c_problem_d_l519_519932


namespace rectangle_with_perpendicular_diagonals_is_square_l519_519085

-- Definitions needed for the conditions
structure Quadrilateral where
  A B C D : Type
  -- Add any required properties or fields if necessary

structure Rectangle extends Quadrilateral where
  right_angle : Prop
  -- More properties of a rectangle can be defined here

structure Square extends Rectangle where
  sides_equal : Prop
  -- More properties defining a square can be added here

theorem rectangle_with_perpendicular_diagonals_is_square (R : Rectangle) (perpendicular_diagonals : Prop) : Square :=
by
  sorry

end rectangle_with_perpendicular_diagonals_is_square_l519_519085


namespace find_k_l519_519668

open Real

-- Define the operation "※"
def star (a b : ℝ) : ℝ := a * b + a + b^2

-- Define the main theorem stating the problem
theorem find_k (k : ℝ) (hk : k > 0) (h : star 1 k = 3) : k = 1 := by
  sorry

end find_k_l519_519668


namespace second_athlete_triple_jump_l519_519449

theorem second_athlete_triple_jump
  (long_jump1 triple_jump1 high_jump1 : ℕ) 
  (long_jump2 high_jump2 : ℕ)
  (average_winner : ℕ) 
  (H1 : long_jump1 = 26) (H2 : triple_jump1 = 30) (H3 : high_jump1 = 7)
  (H4 : long_jump2 = 24) (H5 : high_jump2 = 8) (H6 : average_winner = 22)
  : ∃ x : ℕ, (24 + x + 8) / 3 = 22 ∧ x = 34 := 
by
  sorry

end second_athlete_triple_jump_l519_519449


namespace incenter_on_MN_l519_519911

/-- Two circles are tangent to the circumcircle of triangle ABC at point K. 
One of these circles is tangent to side AB at point M, 
and the other is tangent to side AC at point N. 
Prove that the center of the inscribed circle of triangle ABC 
lies on the line connecting points M and N. -/
theorem incenter_on_MN
  (A B C K M N : Point)
  (h_circum_tangent_1 : TangentCircleAt K (circumcircle A B C) (circle_1))
  (h_circum_tangent_2 : TangentCircleAt K (circumcircle A B C) (circle_2))
  (h_tangent_AB_M : TangentCircleAt M (segment A B) (circle_1))
  (h_tangent_AC_N : TangentCircleAt N (segment A C) (circle_2)) :
  Collinear [incenter A B C, M, N] := 
sorry

end incenter_on_MN_l519_519911


namespace monotonicity_l519_519241

noncomputable def f (a x : ℝ) : ℝ := abs (x^2 - a*x) - log x

-- Define the domain
def domain (x : ℝ) : Prop := 0 < x

-- Correctness of monotonicity results:

theorem monotonicity (a : ℝ) : 
  if a ≤ 0 then 
    ∀ x, domain x → x < (a + sqrt (a^2 + 8)) / 4 → 
    f a x ≤ f a ((a + sqrt (a^2 + 8)) / 4) 
    ∧ ∀ x, domain x → x > (a + sqrt (a^2 + 8)) / 4 → 
    f a x ≥ f a ((a + sqrt (a^2 + 8)) / 4)
  else if 0 < a ∧ a < 1 then 
    ∀ x, domain x → a < x ∧ x < (a + sqrt (a^2 + 8)) / 4 → 
    f a x ≥ f a a 
    ∧ ∀ x, domain x → x > (a + sqrt (a^2 + 8)) / 4 → 
    f a x ≥ f a ((a + sqrt (a^2 + 8)) / 4)
  else if 1 ≤ a ∧ a ≤ 2*sqrt 2 then 
    ∀ x, domain x → x < a → 
    f a x ≤ f a a 
    ∧ ∀ x, domain x → x > a → 
    f a x ≥ f a a 
  else if a > 2*sqrt 2 then 
    ∀ x, domain x → x < (a - sqrt (a^2  - 8)) / 4 ∨ (a + sqrt (a^2 - 8)) / 4 < x ∧ x < a → 
    f a x ≤ f a ((a - sqrt (a^2 - 8)) / 4) ∨ 
    ∀ x, domain x → (a - sqrt (a^2 - 8)) / 4 < x ∧ x < (a + sqrt (a^2 - 8)) / 4 → 
    f a x ≥ f a ((a + sqrt (a^2 - 8)) / 4) 
    ∧ ∀ x, domain x → x > a → 
    f a x ≥ f a a 
else 
    false :=
by sorry

end monotonicity_l519_519241


namespace product_odd_integers_sign_units_digit_l519_519453

/-- Theorem: The product of all odd positive integers from 1 to 2021 is a positive number ending in 5. -/
theorem product_odd_integers_sign_units_digit :
  let product := ∏ i in (Finset.filter (λ x : ℕ, x % 2 = 1) (Finset.range 2022)), i
  in product > 0 ∧ product % 10 = 5 := 
by
  sorry

end product_odd_integers_sign_units_digit_l519_519453


namespace tiles_needed_l519_519045

def room_area : ℝ := 2 * 4 * 2 * 6
def tile_area : ℝ := 1.5 * 2

theorem tiles_needed : room_area / tile_area = 32 := 
by
  sorry

end tiles_needed_l519_519045


namespace angle_bisector_120_degrees_l519_519379

noncomputable theory

variables {a b l : ℝ}
variables {α : ℝ}

theorem angle_bisector_120_degrees 
  (h1 : l = a * b / (a + b))
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : 0 < α ∧ α < π) :
  α = 2 * π / 3 :=
sorry

end angle_bisector_120_degrees_l519_519379


namespace plane_angles_right_l519_519380

-- Definition of trihedral angle and its properties
structure TrihedralAngle :=
  (dihedral_angles : ℝ → ℝ → ℝ)
  (plane_angles : ℝ → ℝ → ℝ)

-- The condition
def dihedral_angles_right (t : TrihedralAngle) : Prop :=
  t.dihedral_angles = (λ _ _, π / 2)

-- The proposition to prove
theorem plane_angles_right (t : TrihedralAngle) 
  (h : dihedral_angles_right t) : 
  t.plane_angles = (λ _ _, π / 2) :=
sorry

end plane_angles_right_l519_519380


namespace arithmetic_proof_l519_519515

theorem arithmetic_proof :
  let sum := 75.892 + 34.5167
  let product := sum * 2
  let rounded := Float.round (product * 1000) / 1000
  rounded = 220.817 :=
by
  unfold sum product rounded
  exact sorry

end arithmetic_proof_l519_519515


namespace exists_inhabitant_with_810_acquaintances_l519_519767

-- Definition of the main problem conditions
def inhabitants := 1000000
def belief_in_santa := 0.9 * inhabitants
def disbelief_in_santa := inhabitants - belief_in_santa

-- Conditions
def knows_at_least_one : ∀ x : ℕ, x < inhabitants → ∃ y : ℕ, y < inhabitants ∧ x ≠ y := sorry
def ten_percent_acquaintances (x : ℕ) (h : x < inhabitants) : Prop := ∀ y : ℕ, y < inhabitants ∧ x ≠ y → (y < belief_in_santa → y < 0.1 * x) := sorry

-- Main theorem
theorem exists_inhabitant_with_810_acquaintances :
  ∃ x : ℕ, x < inhabitants ∧ (knows_at_least_one x) ∧ (ten_percent_acquaintances x) ∧ (810 ≤ x) :=
sorry

end exists_inhabitant_with_810_acquaintances_l519_519767


namespace binary_add_sub_l519_519536

theorem binary_add_sub : 
  (1101 + 111 - 101 + 1001 - 11 : ℕ) = (10101 : ℕ) := by
  sorry

end binary_add_sub_l519_519536


namespace biased_coin_die_even_probability_l519_519682

theorem biased_coin_die_even_probability (p_head : ℚ) (p_even : ℚ) (independent : Prop) :
  p_head = 2/3 → p_even = 1/2 → independent → (p_head * p_even) = 1/3 :=
by
  intro h_head h_even h_independent
  rw [h_head, h_even]
  norm_num
  sorry

end biased_coin_die_even_probability_l519_519682


namespace winnie_servings_l519_519461

theorem winnie_servings:
  ∀ (x : ℝ), 
  (2 / 5) * x + (21 / 25) * x = 82 →
  x = 30 :=
by
  sorry

end winnie_servings_l519_519461


namespace sequence_abs_lt_one_iff_a_zero_l519_519402

theorem sequence_abs_lt_one_iff_a_zero (a : ℝ) :
  (∀ n : ℕ, |a_n a n| < 1) ↔ a = 0 :=
by
  sorry

def a_n (a : ℝ) (n : ℕ) : ℝ :=
  nat.rec_on n a (λ n an, if an = 0 then 0 else an - (1 / an))

end sequence_abs_lt_one_iff_a_zero_l519_519402


namespace pond_50_percent_algae_free_l519_519030

theorem pond_50_percent_algae_free (a : ℕ → ℝ) (h_doubling : ∀ n, a(n + 1) = 2 * a(n)) (h_full_covered : a(20) = 1) :
  a(19) = 0.5 :=
by
  sorry

end pond_50_percent_algae_free_l519_519030


namespace part1_part2_l519_519628

def f (x : ℝ) (a : ℝ) := (Real.exp x) / x - Real.log x + x - a

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → (a ≤ Real.exp 1 + 1) :=
sorry

theorem part2 (x1 x2 a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 ≠ x2 → 0 < x1 → 0 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519628


namespace impossible_tromino_cover_l519_519488

def Dimension := (5, 7)
def TrominoCoveringConditions := 
  -- L-trominoes are defined as shapes formed by removing one square from a (2, 2) grid
  ∀ tromino : ℕ × ℕ × ℕ × ℕ × ℕ, 
  -- Each L-tromino covers 3 squares
  (tromino.1, tromino.2, tromino.3, tromino.4, tromino.5) ∈ { ((0,0), (0,1), (1,0)), ((0,0), (1,0), (1,1)), ((1,0), (1,1), (0,1)), ((0,0), (0,1), (1,1)) }

-- Define the structurally impossible condition
theorem impossible_tromino_cover : 
  ¬(∃ cover : Dimension → TrominoCoveringConditions, 
    ∀ x y : ℕ, x < 5 → y < 7 → 
      let cover_count := 
        if ∃ (tromino : ℕ × ℕ × ℕ × ℕ × ℕ), tromino ∈ cover (x, y) then 1 else 0
      in cover_count = k) 
  where k : ℕ 
:= sorry

end impossible_tromino_cover_l519_519488


namespace tea_milk_mixture_eq_l519_519857

theorem tea_milk_mixture_eq :
  ∀ (tea0 milk0 : ℕ) (transfer : ℕ),
  let cup1_before := tea0,
      cup2_before := milk0
      cup1_after_first := cup1_before + transfer,
      cup2_after_first := cup2_before - transfer,
      mixture_fraction_tea := tea0 / cup1_after_first,
      mixture_fraction_milk := transfer / cup1_after_first,
      second_transfer_tea := mixture_fraction_tea * transfer,
      second_transfer_milk := mixture_fraction_milk * transfer,
      cup1_final_tea := cup1_after_first - second_transfer_tea,
      cup1_final_milk := transfer - second_transfer_milk,
      cup2_final_tea := second_transfer_tea,
      cup2_final_milk := cup2_after_first + second_transfer_milk in
    cup2_final_tea = cup1_final_milk ∧ cup1_final_tea = (tea0 - second_transfer_tea) :=
by
  sorry

end tea_milk_mixture_eq_l519_519857


namespace length_of_XY_in_cube_l519_519584

structure Point (d : ℕ) :=
  (coords : fin d → ℝ)

def distance {d : ℕ} (p1 p2 : Point d) : ℝ :=
  real.sqrt (finset.univ.sum (λ i, (p1.coords i - p2.coords i) ^ 2))

def cube (edge_length : ℝ) :=
  {p : Point 3 // ∀ i, 0 ≤ p.coords i ∧ p.coords i ≤ edge_length}

def line_segment (p1 p2 : Point 3) :=
  {p : Point 3 // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.coords = λ i, (1 - t) * p1.coords i + t * p2.coords i}

noncomputable def length_within_cube (p1 p2 : Point 3) (c : set (Point 3)) : ℝ :=
  real.sqrt (finset.univ.sum (λ i, (p2.coords i - p1.coords i) ^ 2))

theorem length_of_XY_in_cube :
  let X := Point.mk ![(0 : ℝ), 0, 0],
      Y := Point.mk ![(7 : ℝ), 7, 16],
      cube_5 := {p : Point 3 // ∀ i, 0 ≤ p.coords i ∧ p.coords i ≤ 5}
  in length_within_cube (Point.mk ![(0 : ℝ), 0, 4]) (Point.mk ![(5 : ℝ), 5, 9]) (subtype.val '' cube_5)
  = 5 * real.sqrt 3 :=
by
  sorry

end length_of_XY_in_cube_l519_519584


namespace rectangular_to_cylindrical_l519_519996

theorem rectangular_to_cylindrical :
  ∃ r θ z, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ, z) = (6, 5 * Real.pi / 3, 4) ∧
    let x := 3
    let y := -3 * Real.sqrt(3)
    let z_rect := 4
    r = Real.sqrt (x^2 + y^2) ∧
    cos θ = x / r ∧
    sin θ = y / r ∧
    z = z_rect := by
  sorry

end rectangular_to_cylindrical_l519_519996


namespace magazines_cover_area_l519_519900

theorem magazines_cover_area (S : ℝ) (n : ℕ) (h1 : n = 15) (h2 : ∀ k, 0 < k → k ≤ n → ∃ A, ∀ t, t < k → A t ≤ S / k) :
  ∃ A', ∀ t, t < 8 → A' t ≤ 8 / 15 * S := 
by 
  have h0 : 0 < S := sorry
  have h3 : n > 1 := sorry
  sorry

end magazines_cover_area_l519_519900


namespace rectangle_area_l519_519126

-- Define the vertices of the rectangle
def A := (0, 0)
def B := (4, 0)
def C := (4, 3)
def D := (0, 3)

-- Define a function to compute the length of a side given the coordinates of two points
def length (p1 p2 : ℝ × ℝ) : ℝ :=
  if p1.1 = p2.1 then |p1.2 - p2.2| else |p1.1 - p2.1|

-- Define the side lengths of the rectangle using the vertices
def AB := length A B
def BC := length B C

-- Define the area of the rectangle
def area := AB * BC

-- The theorem we need to prove: The area of the given rectangle is 12
theorem rectangle_area : area = 12 := by
  sorry

end rectangle_area_l519_519126


namespace problem_part1_problem_part2_l519_519593

theorem problem_part1
    (f : ℝ → ℝ)
    (m n t : ℝ)
    (h1 : f = λ x => m * x^2 - n * x)
    (h2 : ∀ x, -3 ≤ x ∧ x ≤ 2 → f x ≥ t)
    (h3 : m ∈ Set.Icc 1 3)
    (h4 : f 1 > 0) :
    {x : ℝ | -3 ≤ x ∧ x ≤ 2} ⊆ {x : ℝ | n * x^2 + m * x + t ≤ 0} :=
sorry

theorem problem_part2
    (m n : ℝ)
    (h1 : 1 ≤ m ∧ m ≤ 3)
    (h2 : let f := λ x => m * x^2 - n * x; f 1 > 0) :
    ∃ x, (1/m-n + 9/m - n = 2) ∧ m - n > 0 :=
sorry

end problem_part1_problem_part2_l519_519593


namespace solve_eq1_solve_inq1_l519_519870

-- Define the equation condition
def Eq1 (n : ℕ) : Prop := 
  (A (2 * n + 1)) ^ 4 = 140 * (A n) ^ 3

-- Define the inequality condition
def Inq1 (N n : ℕ) : Prop := 
  (A N) ^ 4 ≥ 24 * (C n) ^ 6

-- Now state the proof problems
theorem solve_eq1 : ∃ n : ℕ, Eq1 n → n = 3 := 
  sorry

theorem solve_inq1 : ∃ n : ℕ, Inq1 N n → n ∈ {6, 7, 8, 9, 10} := 
  sorry

end solve_eq1_solve_inq1_l519_519870


namespace value_of_a_plus_b_l519_519026

open ProbabilityTheory

variables (Ω : Type) 
  (X : Ω → ℝ)
  (p : ProbabilityTheory.Measure Ω)
  (a b : ℝ)

noncomputable theory

def is_discrete_random_variable : Prop := ∃ (f : Ω → ℕ), ∀ ω, X ω = ↑(f ω)

axiom P_X_a : p {ω | X ω = a} = 1/3
axiom P_X_b : p {ω | X ω = b} = 2/3
axiom a_lt_b : a < b
axiom E_X : ∫ x, X x ∂p = 2/3
axiom D_X : ∫ x, (X x - (∫ x, X x ∂p))^2 ∂p = 2/9

theorem value_of_a_plus_b : a + b = 1 :=
sorry

end value_of_a_plus_b_l519_519026


namespace simplify_expression_l519_519985

theorem simplify_expression : 
  2 ^ (-1: ℤ) + Real.sqrt 16 - (3 - Real.sqrt 3) ^ 0 + |Real.sqrt 2 - 1 / 2| = 3 + Real.sqrt 2 := by
  sorry

end simplify_expression_l519_519985


namespace steve_break_even_l519_519872

noncomputable def break_even_performances
  (fixed_overhead : ℕ)
  (min_production_cost max_production_cost : ℕ)
  (venue_capacity percentage_occupied : ℕ)
  (ticket_price : ℕ) : ℕ :=
(fixed_overhead + (percentage_occupied / 100 * venue_capacity * ticket_price)) / (percentage_occupied / 100 * venue_capacity * ticket_price)

theorem steve_break_even
  (fixed_overhead : ℕ := 81000)
  (min_production_cost : ℕ := 5000)
  (max_production_cost : ℕ := 9000)
  (venue_capacity : ℕ := 500)
  (percentage_occupied : ℕ := 80)
  (ticket_price : ℕ := 40)
  (avg_production_cost : ℕ := (min_production_cost + max_production_cost) / 2) :
  break_even_performances fixed_overhead min_production_cost max_production_cost venue_capacity percentage_occupied ticket_price = 9 :=
by
  sorry

end steve_break_even_l519_519872


namespace parameterized_line_l519_519422

def f (t : ℝ) : ℝ := sorry 

theorem parameterized_line (t : ℝ) :
    let x := f(t)
    let y := 20 * t - 10
    (y = 2 * x - 30) → (f(t) = 10 * t + 10) :=
by
  intros x y h
  simp at h
  sorry

end parameterized_line_l519_519422


namespace pear_juice_percentage_l519_519850

/--
Miki has a dozen oranges and pears. She extracts juice as follows:
5 pears -> 10 ounces of pear juice
3 oranges -> 12 ounces of orange juice
She uses 10 pears and 10 oranges to make a blend.
Prove that the percent of the blend that is pear juice is 33.33%.
-/
theorem pear_juice_percentage :
  let pear_juice_per_pear := 10 / 5
  let orange_juice_per_orange := 12 / 3
  let total_pear_juice := 10 * pear_juice_per_pear
  let total_orange_juice := 10 * orange_juice_per_orange
  let total_juice := total_pear_juice + total_orange_juice
  total_pear_juice / total_juice = 1 / 3 :=
by
  sorry

end pear_juice_percentage_l519_519850


namespace part1_part2_l519_519648

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519648


namespace total_eggs_found_l519_519696

-- Defining the constants based on the conditions
def eggs_at_club_house : ℕ := 40
def eggs_at_park : ℕ := 25
def eggs_at_town_hall : ℕ := 15

-- Theorem statement to prove the total number of Easter eggs
theorem total_eggs_found : eggs_at_club_house + eggs_at_park + eggs_at_town_hall = 80 :=
by
  calc
  eggs_at_club_house + eggs_at_park + eggs_at_town_hall = 40 + 25 + 15 := by rfl
  ... = 65 + 15 := by rfl
  ... = 80 := by rfl

end total_eggs_found_l519_519696


namespace proof_set_intersection_l519_519605

open Set

variable (ℝ : Type) [LinearOrderedField ℝ] [TopologicalSpace ℝ] [OrderTopology ℝ]

def R := set.univ  -- The universal set.

def A : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x - 1 }

def B : Set ℝ := { x : ℝ | log 2 x ≤ 1 }

def CRB : Set ℝ := compl B  -- Complement of B in the universal set.

theorem proof_set_intersection : A ∩ CRB = { y : ℝ | -1 < y ∧ y ≤ 0 ∨ 2 < y } := 
by
  sorry

end proof_set_intersection_l519_519605


namespace rhombus_longer_diagonal_l519_519140

theorem rhombus_longer_diagonal (a b : ℕ) (d1 : ℕ) (d2 : ℕ) (h0 : a = 65) (h1 : d1 = 56) (h2 : a * a = (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2)) :
  d2 = 118 :=
by
  have h3 : (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2) = 65 * 65, from h2,
  rw [←h0, ←h1] at h3,
  have h4 : (28 : ℕ) = d1 / 2, sorry,
  have h5 : (28 : ℕ) * (28 : ℕ) + ((d2 / 2) * (d2 / 2)) = 65 * 65, from h3,
  have h6 : (28 * 28) + (d2 / 2) * (d2 / 2) = (65*65), by rw h5,
  have h7 : (65 * 65) - 784 = (d2 / 2) * (d2 / 2), sorry,
  have h8 : (d2 / 2) * (d2 / 2) = 3441, by simp[h7],
  have h9 : (d2 / 2) = 59, from nat.sqrt_eq 3441,
  have h10 : d2 = 2 * 59, by rw [h9, two_mul],
  exact h10
 

end rhombus_longer_diagonal_l519_519140


namespace positive_integers_congruent_to_2_mod_7_lt_500_count_l519_519274

theorem positive_integers_congruent_to_2_mod_7_lt_500_count : 
  ∃ n : ℕ, n = 72 ∧ ∀ k : ℕ, (k < n → (∃ m : ℕ, (m < 500 ∧ m % 7 = 2) ∧ m = 2 + 7 * k)) := 
by
  sorry

end positive_integers_congruent_to_2_mod_7_lt_500_count_l519_519274


namespace correct_calculation_l519_519811

theorem correct_calculation (y : ℤ) (h : (y + 4) * 5 = 140) : 5 * y + 4 = 124 :=
by {
  sorry
}

end correct_calculation_l519_519811


namespace axis_of_symmetry_closest_l519_519260

noncomputable def axis_of_symmetry (ω φ : ℝ) : ℝ :=
  if ω > 0 ∧ |φ| < π ∧ (∃ x, y = sin x ∧ y = sin (ω(x + φ))) then
    let x_val := π / 12 in x_val
  else
    0 -- This is a placeholder and isn't used in noncomputable context

theorem axis_of_symmetry_closest (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < π)
  (htrans : ∀ x, (sin (2(x + π / 6)) = sin x)) :
  axis_of_symmetry ω φ = π / 12 :=
by
  sorry

end axis_of_symmetry_closest_l519_519260


namespace determine_cost_price_l519_519937

def selling_price := 16
def loss_fraction := 1 / 6

noncomputable def cost_price (CP : ℝ) : Prop :=
  selling_price = CP - (loss_fraction * CP)

theorem determine_cost_price (CP : ℝ) (h: cost_price CP) : CP = 19.2 := by
  sorry

end determine_cost_price_l519_519937


namespace sowdharya_currency_notes_l519_519397

variable (x y : ℕ)

theorem sowdharya_currency_notes :
  95 * x + 45 * y = 5000 → y = 71 → x + y = 90 :=
by {
  intros h₁ h₂,
  have h₃ : 95 * 19 + 45 * 71 = 5000,
  { calc
    95 * 19 + 45 * 71 = 1805 + 3195 : by ring
    ... = 5000 : by norm_num,
  },
  have h₄ : y = 71 := h₂,
  have h₅ : 95 * x + 45 * 71 = 5000 := h₁,
  have h₆ : x = 19 := by linarith,
  have h₇ : y = 71 := h₂,
  change 19 + 71 = 90,
  rw h₆,
  rw h₇,
  exact calc
    19 + 71 = 90 : by norm_num,
  sorry
}

end sowdharya_currency_notes_l519_519397


namespace adam_and_simon_travel_time_l519_519156

theorem adam_and_simon_travel_time (x : ℝ) : 
    (x > 0) → 
    (12 * x) ^ 2 + (10 * x) ^ 2 = 130 ^ 2 ↔ x ≈ 8.322 := sorry

end adam_and_simon_travel_time_l519_519156


namespace probability_both_in_picture_l519_519384

theorem probability_both_in_picture:
  let T := 480 -- time in seconds (8 minutes)
  let rachelLap := 100 -- time for Rachel to complete a lap
  let robertLap := 70 -- time for Robert to complete a lap
  let windowStart := 0 -- start of the picture window we are concerned with
  let windowEnd := 20 -- end of the picture window we are concerned with 
  let overlapEnd := min 33.13 windowEnd -- end of overlap time
  let totalDuration := 60 -- total duration between 8 to 9 minutes in seconds
in ((overlapEnd - windowStart) / totalDuration) = 11 / 20 :=
sorry

end probability_both_in_picture_l519_519384


namespace quadrilateral_fourth_side_length_l519_519509

-- Defining the given conditions for the problem
variable (radius : ℝ) (a b c d ab bc cd : ℝ)
variable (inscribed_in_circle : true)
variable (side_lengths : ab = 200 ∧ bc = 200 ∧ cd = 200 ∧ radius = 200 * Real.sqrt 2)

-- Formalize the goal: proving the length of the fourth side \(AD\)
theorem quadrilateral_fourth_side_length : ∃ ad : ℝ, ad = 500 :=
by
  -- Declare the variables and conditions
  intros ab bc cd radius inscribed_in_circle side_lengths
  -- Use the side lengths and circle radius to show AD = 500
  sorry

end quadrilateral_fourth_side_length_l519_519509


namespace relative_prime_in_consecutive_integers_l519_519376

theorem relative_prime_in_consecutive_integers (n : ℤ) : 
  ∃ k, n ≤ k ∧ k ≤ n + 5 ∧ ∀ m, n ≤ m ∧ m ≤ n + 5 ∧ m ≠ k → Int.gcd k m = 1 :=
sorry

end relative_prime_in_consecutive_integers_l519_519376


namespace cosine_angle_BD1_AC_l519_519320

open Real 

-- Definitions of the vectors based on the problem's conditions
structure Parallelepiped :=
(A B C D A1 B1 C1 D1 : Point)
(edge_length : ℝ)
(angle : ℝ)

-- The given conditions
def parallelepiped_conditions : Prop := 
  ∀ (P : Parallelepiped),
    P.edge_length = 2 ∧
    P.angle = π / 3 -- 60 degrees converted to radians

-- The required proof
theorem cosine_angle_BD1_AC (P : Parallelepiped) (h : parallelepiped_conditions) :
  cos (angle_between (BD1_vector P) (AC_vector P)) = (sqrt 6) / 6 :=
sorry  -- proof to be completed

end cosine_angle_BD1_AC_l519_519320


namespace Daniel_correct_answers_l519_519304

theorem Daniel_correct_answers
  (c w : ℕ)
  (h1 : c + w = 12)
  (h2 : 4 * c - 3 * w = 21) :
  c = 9 :=
sorry

end Daniel_correct_answers_l519_519304


namespace prove_a_ge_neg_one_fourth_l519_519670

-- Lean 4 statement to reflect the problem
theorem prove_a_ge_neg_one_fourth
  (x y z a : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h1 : x * y - z = a)
  (h2 : y * z - x = a)
  (h3 : z * x - y = a) :
  a ≥ - (1 / 4) :=
sorry

end prove_a_ge_neg_one_fourth_l519_519670


namespace domain_of_k_function_l519_519190

-- Define the function k(x)
noncomputable def k (x : ℝ) : ℝ := 1 / (x + 5) + 1 / (x^2 - 9) + 1 / (x^3 - x + 1)

-- Define the statement of the proof in Lean
theorem domain_of_k_function : 
  {x : ℝ | x ≠ -5 ∧ x ≠ 3 ∧ x ≠ -3 ∧ x ≠ a ∧ x ≠ b ∧ x ≠ c} = 
    {x | x ∉ { (-∞ : set ℝ), -5 } ∪ { -5, -3 } ∪ { -3, 3 } ∪ { 3, (∞ : set ℝ) }} :=
sorry

end domain_of_k_function_l519_519190


namespace simplify_power_multiplication_l519_519393

theorem simplify_power_multiplication (x : ℝ) : (-x) ^ 3 * (-x) ^ 2 = -x ^ 5 :=
by sorry

end simplify_power_multiplication_l519_519393


namespace least_number_of_square_tiles_l519_519091

theorem least_number_of_square_tiles (length_cm breadth_cm : ℕ) (h_length : length_cm = 544) (h_breadth : breadth_cm = 374) :
  let tile_size_cm := Nat.gcd length_cm breadth_cm in
  let num_tiles_length := length_cm / tile_size_cm in
  let num_tiles_breadth := breadth_cm / tile_size_cm in
  num_tiles_length * num_tiles_breadth = 176 :=
by
  sorry

end least_number_of_square_tiles_l519_519091


namespace smallest_value_am_hm_inequality_l519_519828

theorem smallest_value_am_hm_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end smallest_value_am_hm_inequality_l519_519828


namespace rhombus_longer_diagonal_l519_519139

theorem rhombus_longer_diagonal (a b : ℕ) (d1 : ℕ) (d2 : ℕ) (h0 : a = 65) (h1 : d1 = 56) (h2 : a * a = (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2)) :
  d2 = 118 :=
by
  have h3 : (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2) = 65 * 65, from h2,
  rw [←h0, ←h1] at h3,
  have h4 : (28 : ℕ) = d1 / 2, sorry,
  have h5 : (28 : ℕ) * (28 : ℕ) + ((d2 / 2) * (d2 / 2)) = 65 * 65, from h3,
  have h6 : (28 * 28) + (d2 / 2) * (d2 / 2) = (65*65), by rw h5,
  have h7 : (65 * 65) - 784 = (d2 / 2) * (d2 / 2), sorry,
  have h8 : (d2 / 2) * (d2 / 2) = 3441, by simp[h7],
  have h9 : (d2 / 2) = 59, from nat.sqrt_eq 3441,
  have h10 : d2 = 2 * 59, by rw [h9, two_mul],
  exact h10
 

end rhombus_longer_diagonal_l519_519139


namespace albums_in_collections_l519_519969

noncomputable def album_count (A J S : ℕ) (shared : ℕ) (unique_J : ℕ) (unique_S : ℕ) : ℕ :=
  (A - shared) + unique_J + unique_S

theorem albums_in_collections:
  ∀ (A J S shared unique_J unique_S : ℕ),
  A = 20 →
  shared = 10 →
  unique_J = 5 →
  unique_S = 3 →
  album_count A J S shared unique_J unique_S = 18 :=
by
  intros
  simp [album_count, *]
  sorry

end albums_in_collections_l519_519969


namespace three_point_seven_five_minus_one_point_four_six_l519_519172

theorem three_point_seven_five_minus_one_point_four_six : 3.75 - 1.46 = 2.29 :=
by sorry

end three_point_seven_five_minus_one_point_four_six_l519_519172


namespace smallest_value_am_hm_inequality_l519_519829

theorem smallest_value_am_hm_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end smallest_value_am_hm_inequality_l519_519829


namespace angle_ELK_eq_half_angle_BCD_l519_519910

open EuclideanGeometry

-- Definitions derived from conditions
variables {A B C D E K L R S : Point}
variable {ω : Circle}
variables {AB AC BC : Segment}

-- Conditions specified in the problem
axiom cond1 : inscribed_in_triangle A B C ω
axiom cond2 : circle_with_chord BC intersects_segments_again AB AC S R
axiom cond3 : segments_meet_at BR CS L
axiom cond4 : rays_intersect_at LR LS ω D E
axiom cond5 : internal_angle_bisector_meets_line BDE ER K 
axiom cond6 : BE = BR

-- The statement to be proved
theorem angle_ELK_eq_half_angle_BCD :
  ∠ ELK = (1 / 2) * ∠ BCD :=
sorry

end angle_ELK_eq_half_angle_BCD_l519_519910


namespace find_m_l519_519259

noncomputable def is_power_function (y : ℝ → ℝ) := 
  ∃ (c : ℝ), ∃ (n : ℝ), ∀ x : ℝ, y x = c * x ^ n

theorem find_m (m : ℝ) :
  (∀ x : ℝ, (∃ c : ℝ, (m^2 - 2 * m + 1) * x^(m - 1) = c * x^n) ∧ (∀ x : ℝ, true)) → m = 2 :=
sorry

end find_m_l519_519259


namespace parquet_tiles_needed_l519_519041

def room_width : ℝ := 8
def room_length : ℝ := 12
def tile_width : ℝ := 1.5
def tile_length : ℝ := 2

def room_area : ℝ := room_width * room_length
def tile_area : ℝ := tile_width * tile_length

def tiles_needed : ℝ := room_area / tile_area

theorem parquet_tiles_needed : tiles_needed = 32 :=
by
  -- sorry to skip the detailed proof
  sorry

end parquet_tiles_needed_l519_519041


namespace suzanne_needs_weeks_l519_519852

theorem suzanne_needs_weeks
  (slices_per_weekend : ℕ)
  (slices_per_loaf : ℕ)
  (loaves_needed : ℕ)
  (weeks : ℕ) :
  (slices_per_weekend = 3) →
  (slices_per_loaf = 12) →
  (loaves_needed = 26) →
  (4 * loaves_needed = weeks) →
  weeks = 104 :=
by
  intros h1 h2 h3 h4
  rw [h3, ←h1, ←h2, mul_comm, mul_assoc, mul_eq_mul_right_iff] at h4
  exact h4.1
  sorry

end suzanne_needs_weeks_l519_519852


namespace total_cost_of_tickets_l519_519805

-- Conditions
def normal_price : ℝ := 50
def website_tickets_cost : ℝ := 2 * normal_price
def scalper_tickets_cost : ℝ := 2 * (2.4 * normal_price) - 10
def discounted_ticket_cost : ℝ := 0.6 * normal_price

-- Proof Statement
theorem total_cost_of_tickets :
  website_tickets_cost + scalper_tickets_cost + discounted_ticket_cost = 360 :=
by
  sorry

end total_cost_of_tickets_l519_519805


namespace part1_part2_l519_519641

def f (x a : ℝ) : ℝ := (Exp x / x) - Log x + x - a

theorem part1 (x a : ℝ) (hx : f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 (x1 x2 a : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) 
  (hx1_range : 0 < x1) (hx2_range : x2 > 1) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519641


namespace xyz_inequality_l519_519684

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := by
  sorry

end xyz_inequality_l519_519684


namespace poverty_alleviation_stationing_l519_519375

-- Define the conditions in Lean
def num_people_in_group := 7
def num_areas := 3
def num_men := 4
def num_women := 3

-- Define the requirement that every area must have at least 1 man and 1 woman
def man_woman_condition (area: ℕ) : Prop :=
  ∃ (man: ℕ) (woman: ℕ), man ∈ {1, 2, 3, 4} ∧ woman ∈ {5, 6, 7} 

-- Define the requirement that male A (assigned index 1 here) must be in area A (index 1)
def male_A_in_area_A (area: ℕ) : Prop :=
  area = 1

-- Define the final property to be proved
def ways_to_station (total_ways: ℕ) : Prop :=
  total_ways = 72

-- State the theorem with the specified conditions
theorem poverty_alleviation_stationing :
  ∀ (people: ℕ) (areas: ℕ) (men: ℕ) (women: ℕ),
    people = num_people_in_group →
    areas = num_areas →
    men = num_men →
    women = num_women →
    (∀ area, man_woman_condition area) ∧ (male_A_in_area_A 1) →
    ∃ ways, ways_to_station ways := 
by
  intros people areas men women h1 h2 h3 h4 h5,
  unfold num_people_in_group num_areas num_men num_women at *,
  -- we just assume the posited theorem conclusion using sorry as a placeholder for missing proof steps
  exact ⟨72, rfl⟩

end poverty_alleviation_stationing_l519_519375


namespace measure_of_angle_EFD_l519_519534

noncomputable def measure_angle_EFD (M N P : Type*) [angle_space M] (D E F : point_circle_space M)
  (angle_M : angle M E D = 50) (angle_N : angle N D F = 70) (angle_P : angle P F E = 60) 
  (circumcircle : circle (triangle E F D)) (incircle : circle (triangle M N P)) : Prop :=
  angle E F D = 125

theorem measure_of_angle_EFD : ∀ (M N P : Type*) [angle_space M] (D E F : point_circle_space M)
  (angle_M : angle M E D = 50) (angle_N : angle N D F = 70) (angle_P : angle P F E = 60) 
  (circumcircle : circle (triangle E F D)) (incircle : circle (triangle M N P)),
  measure_angle_EFD M N P D E F angle_M angle_N angle_P circumcircle incircle :=
sorry

end measure_of_angle_EFD_l519_519534


namespace slope_of_line_l519_519457

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the slope function
def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

-- Define the given points
def P1 : Point := {x := 4, y := -3}
def P2 : Point := {x := -1, y := 6}

-- Prove the slope of the line passing through P1 and P2
theorem slope_of_line : slope P1 P2 = - (9 / 5) :=
by
  sorry

end slope_of_line_l519_519457


namespace average_multiple_l519_519108

theorem average_multiple (L : List ℝ) (h_len : L.length = 21) (n : ℝ) (h_mem : n ∈ L)
    (h_n : n = (1/6) * L.sum) :
    ∃ (k : ℝ), k = 4 ∧ n = k * (L.erase n).average := by
  sorry

end average_multiple_l519_519108


namespace number_of_valid_pairs_is_three_l519_519558

def discriminant (b c : ℤ) : ℤ := b^2 * c^2 - 4 * b^3 - 4 * c^3 + 18 * b * c - 27

def valid_pair (b c : ℤ) : Prop := discriminant b c ≤ 0 ∧ discriminant c b ≤ 0

def is_valid_pair (b c : ℕ) : Prop :=
  discriminant b c ≤ 0 ∧ discriminant c b ≤ 0

noncomputable def valid_pairs_count : ℕ :=
  ([(1, 1), (2, 2), (3, 3)].countp (λ pair, valid_pair pair.fst pair.snd))

theorem number_of_valid_pairs_is_three :
  valid_pairs_count = 3 :=
by sorry

end number_of_valid_pairs_is_three_l519_519558


namespace savings_by_having_insurance_l519_519061

theorem savings_by_having_insurance :
  let insurance_cost := 24 * 20 in
  let procedure_cost := 5000 in
  let insurance_coverage := 0.2 in
  let amount_paid_with_insurance := procedure_cost * insurance_coverage in
  let total_paid_with_insurance := amount_paid_with_insurance + insurance_cost in
  let savings := procedure_cost - total_paid_with_insurance in
  savings = 3520 := 
by {
  let insurance_cost := 24 * 20 in
  let procedure_cost := 5000 in
  let insurance_coverage := 0.2 in
  let amount_paid_with_insurance := procedure_cost * insurance_coverage in
  let total_paid_with_insurance := amount_paid_with_insurance + insurance_cost in
  let savings := procedure_cost - total_paid_with_insurance in
  guard_hyp savings,
  sorry
}

end savings_by_having_insurance_l519_519061


namespace prob_event_A_l519_519938

theorem prob_event_A:
  ∀ (Ω : Type) [Fintype Ω] (p: ProbMassFunction Ω) (A B : Event Ω),
  (Independent A B) ∧ (0 < p (A.val)) ∧ (p (A.val) = 2 * p (B.val)) ∧ (p (A.val ∪ B.val) = 3 * p (A.val ∩ B.val))
  → p (A.val) = 3 / 4 :=
by
  intros Ω _ p A B hi hab heq hunion
  sorry

end prob_event_A_l519_519938


namespace find_a_l519_519883

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem find_a
  (a : ℝ)
  (h₁ : ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f a x ≤ 4)
  (h₂ : ∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f a x = 4) :
  a = -3 ∨ a = 3 / 8 :=
by
  sorry

end find_a_l519_519883


namespace prob_queen_then_diamond_is_correct_l519_519066

/-- Define the probability of drawing a Queen first and a diamond second -/
def prob_queen_then_diamond : ℚ := (3 / 52) * (13 / 51) + (1 / 52) * (12 / 51)

/-- The probability that the first card is a Queen and the second card is a diamond is 18/221 -/
theorem prob_queen_then_diamond_is_correct : prob_queen_then_diamond = 18 / 221 :=
by
  sorry

end prob_queen_then_diamond_is_correct_l519_519066


namespace rhombus_longer_diagonal_length_l519_519137

theorem rhombus_longer_diagonal_length (side_length : ℕ) (shorter_diagonal : ℕ) 
  (h_side : side_length = 65) (h_short_diag : shorter_diagonal = 56) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 118 :=
by
  sorry

end rhombus_longer_diagonal_length_l519_519137


namespace inner_polygon_perimeter_lt_outer_polygon_perimeter_l519_519856

-- defintion of convex polygons having lesser perimeter
theorem inner_polygon_perimeter_lt_outer_polygon_perimeter
  (P_inner P_outer : Polygon)
  (h_inner_convex: Convex P_inner)
  (h_outer_convex: Convex P_outer)
  (h_inner_in_outer : ⊆ P_inner P_outer) :
  Perimeter P_inner < Perimeter P_outer := 
  sorry

end inner_polygon_perimeter_lt_outer_polygon_perimeter_l519_519856


namespace david_overall_average_l519_519185

open Real

noncomputable def english_weighted_average := (74 * 0.20) + (80 * 0.25) + (77 * 0.55)
noncomputable def english_modified := english_weighted_average * 1.5

noncomputable def math_weighted_average := (65 * 0.15) + (75 * 0.25) + (90 * 0.60)
noncomputable def math_modified := math_weighted_average * 2.0

noncomputable def physics_weighted_average := (82 * 0.40) + (85 * 0.60)
noncomputable def physics_modified := physics_weighted_average * 1.2

noncomputable def chemistry_weighted_average := (67 * 0.35) + (89 * 0.65)
noncomputable def chemistry_modified := chemistry_weighted_average * 1.0

noncomputable def biology_weighted_average := (90 * 0.30) + (95 * 0.70)
noncomputable def biology_modified := biology_weighted_average * 1.5

noncomputable def overall_average := (english_modified + math_modified + physics_modified + chemistry_modified + biology_modified) / 5

theorem david_overall_average :
  overall_average = 120.567 :=
by
  -- Proof to be filled in
  sorry

end david_overall_average_l519_519185


namespace number_of_valid_labelings_l519_519015

-- Given conditions:
-- 1. Regular decagon ABCDEFGHIJ with center K.
-- 2. Each of the vertices and the center are associated with one of the digits 1 through 10, with each digit used once.
-- 3. The sums of the numbers on the lines AKC, BLD, CME, DNF, and EOG are all equal.

def decagon := Fin 10 → Fin 10

def is_valid_labeling (f: decagon) : Prop :=
  let sum_eq := f 0 + f 1 + f 9 = f 1 + f 2 + f 0 in
  sum_eq ∧ f.2.2 = f 3 + f 4 + f 1 ∧ f 4 + f 5 + f 2 = f 3 + f 6 + f 1 ∧
  f 5 + f 6 + f 3 = f 4 + f 7 + f 2 ∧ f 8 + f 7 + f 3 = f 1 + f 6 + f 0

theorem number_of_valid_labelings : 
  {f: Fin 10 → Fin 10 // is_valid_labeling f}.card = 3840 :=
sorry

end number_of_valid_labelings_l519_519015


namespace area_of_side_face_l519_519551

-- Define the dimensions of the box
variables {l w h : ℝ}

-- Given conditions
def front_face_area_condition := l * w = (1/2) * (l * h)
def top_face_side_face_condition := l * h = 1.5 * (w * h)
def volume_condition := l * w * h = 5184

-- Define the area of the side face
def side_face_area := w * h

-- Main theorem
theorem area_of_side_face : front_face_area_condition → top_face_side_face_condition → volume_condition → side_face_area = 288 := by
  intros,
  sorry

end area_of_side_face_l519_519551


namespace problem1_problem2_l519_519677

noncomputable theory

section Part1
variables {x : ℝ}

def m : ℝ × ℝ := (Real.sin (x - π / 3), 1)
def n : ℝ × ℝ := (Real.cos x, 1)
def tan_of_parallel (m n : ℝ × ℝ) : Prop := m.1 = n.1

theorem problem1 (h : tan_of_parallel m n) : Real.tan x = Real.sqrt 3 + 2 := 
sorry
end Part1

section Part2
variables {x : ℝ}

def m : ℝ × ℝ := (Real.sin (x - π / 3), 1)
def n : ℝ × ℝ := (Real.cos x, 1)
def f (x : ℝ) : ℝ := m.1 * n.1 + 1

theorem problem2 (h : 0 ≤ x ∧ x ≤ π / 2) :
  (1 - Real.sqrt 3 / 2) ≤ f x ∧ f x ≤ (1 + Real.sqrt 3 / 2) := 
sorry
end Part2

end problem1_problem2_l519_519677


namespace abs_expression_equals_l519_519989

theorem abs_expression_equals (h : Real.pi < 12) : 
  abs (Real.pi - abs (Real.pi - 12)) = 12 - 2 * Real.pi := 
by
  sorry

end abs_expression_equals_l519_519989


namespace min_value_of_squares_l519_519830

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) : a^2 + b^2 + c^2 ≥ t^2 / 3 :=
sorry

end min_value_of_squares_l519_519830


namespace function_passes_through_A_l519_519256

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + Real.log x / Real.log a

theorem function_passes_through_A 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a ≠ 1)
  : f a 2 = 4 := sorry

end function_passes_through_A_l519_519256


namespace sum_integers_end_in_3_div_4_l519_519982

theorem sum_integers_end_in_3_div_4 :
  (∑ n in Finset.filter (fun n => n % 10 = 3 ∧ n % 4 = 0) (Finset.Icc 100 500), n) = 5757 :=
by
  sorry

end sum_integers_end_in_3_div_4_l519_519982


namespace division_multiplication_expression_l519_519102

theorem division_multiplication_expression : 377 / 13 / 29 * 1 / 4 / 2 = 0.125 :=
by
  sorry

end division_multiplication_expression_l519_519102


namespace total_slices_at_picnic_l519_519550

def danny_watermelons : ℕ := 3
def danny_slices_per_watermelon : ℕ := 10
def sister_watermelons : ℕ := 1
def sister_slices_per_watermelon : ℕ := 15

def total_danny_slices : ℕ := danny_watermelons * danny_slices_per_watermelon
def total_sister_slices : ℕ := sister_watermelons * sister_slices_per_watermelon
def total_slices : ℕ := total_danny_slices + total_sister_slices

theorem total_slices_at_picnic : total_slices = 45 :=
by
  sorry

end total_slices_at_picnic_l519_519550


namespace train_speed_kmph_l519_519963

noncomputable def train_speed_mps : ℝ := 60.0048

def conversion_factor : ℝ := 3.6

theorem train_speed_kmph : train_speed_mps * conversion_factor = 216.01728 := by
  sorry

end train_speed_kmph_l519_519963


namespace inhabitant_knows_at_least_810_l519_519740

open Function

def population : ℕ := 1000000

def believers_claus (p : ℕ) : Prop := p = (90 * population / 100)

def knows_someone (p : ℕ) (knows : p → p → Prop) : Prop := 
  ∀ x : p, ∃ y : p, knows x y

def acquaintances_claus (p : ℕ) (believes : p → Prop) (knows : p → p → Prop) : Prop :=
  ∀ x : p, (believes x →(∃ k : ℕ, ∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y ∧ (10 * k / 100) y ∧ ¬believes y))

theorem inhabitant_knows_at_least_810 :
  ∃ (p : ℕ) (believes : p → Prop) (knows : p → p → Prop),
    population = 1000000 ∧
    believers_claus p ∧
    knows_someone p knows ∧
    acquaintances_claus p believes knows →
    ∃ x : p, ∃ k : ℕ, k ≥ 810 ∧ (∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y) :=
sorry

end inhabitant_knows_at_least_810_l519_519740


namespace upload_time_l519_519333

theorem upload_time (file_size upload_speed : ℕ) (h_file_size : file_size = 160) (h_upload_speed : upload_speed = 8) : file_size / upload_speed = 20 :=
by
  sorry

end upload_time_l519_519333


namespace shaded_fraction_l519_519164

-- Define the square ABCD and the points P and Q
structure Square (s : ℝ) :=
  (A B C D P Q : (ℝ × ℝ))
  (AB : A = (0, 0) ∧ B = (s, 0))
  (BC : B = (s, 0) ∧ C = (s, s))
  (CD : C = (s, s) ∧ D = (0, s))
  (DA : D = (0, s) ∧ A = (0, 0))
  (P_location : P = (s / 3, 0))
  (Q_location : Q = (s, 2 * s / 3))

-- Define the proof goal
theorem shaded_fraction {s : ℝ} (sq : Square s) : 
  let square_area := s ^ 2 in
  let triangle_area := (1 / 2) * s * (2 * s / 3) * (1 / 3) in
  square_area - triangle_area = (8 / 9) * square_area := 
by
  sorry

end shaded_fraction_l519_519164


namespace acquaintance_paradox_proof_l519_519712

theorem acquaintance_paradox_proof :
  ∃ (x : ℕ), (acquaintances x ≥ 810) :=
by
  let n_population := 1000000
  let n_believers := 900000
  let n_nonBelievers := n_population - n_believers
  have h1 : n_population = 1000000 := rfl
  have h2 : n_believers = 0.90 * n_population := by sorry
  have h3 : n_nonBelievers = n_population - n_believers := by sorry
  have h4 : ∀ x, acquaintances x ≥ 1 := by sorry
  have h5 : ∀ x, 0.10 * acquaintances x = nonBelievers_acquaintances x := by sorry
  have h6 : nonBelievers_acquaintances_total >= 90 := by sorry
  
  show ∃ (x : ℕ), acquaintances x ≥ 810 from sorry
  
  sorry

end acquaintance_paradox_proof_l519_519712


namespace sum_of_reciprocal_roots_l519_519539

open Polynomial

noncomputable def cubic_poly := (20 : ℝ) * X^3 - 40 * X^2 + 18 * X - 1

theorem sum_of_reciprocal_roots 
  (a b c : ℝ) 
  (h_roots : cubic_poly.roots = {a, b, c})
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_interval : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) :
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1) :=
  sorry

end sum_of_reciprocal_roots_l519_519539


namespace ratio_of_heights_l519_519430

theorem ratio_of_heights (r h : ℝ) (π : ℝ) (V1 V2 : ℝ) : 
  V1 = π * r^2 * h →
  let r' := 3 * r in
  V2 = 18 * V1 →
  V2 = π * (r')^2 * (h') →
  h' / h = 2 :=
begin
  intros h1,
  intros major_condition,
  skip,
  sorry
end

end ratio_of_heights_l519_519430


namespace correct_calculation_l519_519472

theorem correct_calculation (x : ℤ) (h : x + 54 = 78) : x + 45 = 69 :=
by
  sorry

end correct_calculation_l519_519472


namespace line_PH_passes_through_fixed_point_l519_519222

theorem line_PH_passes_through_fixed_point
  (circle : Type) (chord A B : circle) (not_diam : ¬(chord = diameter circle)) 
  (C_moving_onside_AB : ∀ C : circle, C ∈ major_arc A B)
  (circle_through_A_C_H : ∀ (C : circle) (H : orthocenter (triangle A B C)) (circle_AC_H : circle), ∃ P : circle, P ∈ line (B C) ∧ P ∈ circle_AC_H)
  : ∃ X : circle, ∀ (C : circle) (H : orthocenter (triangle A B C)) (P : circle),
      P ∈ line (B C) → P ∈ circle_through_A_C_H C H → X ∈ line (P H) :=
begin
  sorry
end

end line_PH_passes_through_fixed_point_l519_519222


namespace find_a5_l519_519607

-- Definition of the arithmetic sequence property
def arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℝ} {d : ℝ}
hypothesis h1 : arith_seq a d
hypothesis h2 : d ≠ 0
hypothesis h3 : a 3 + a 9 = a 10 - a 8

-- The goal to prove
theorem find_a5 : a 5 = 0 :=
sorry

end find_a5_l519_519607


namespace sports_shopping_costs_l519_519498

variable (x : ℤ)

theorem sports_shopping_costs 
  (x_gt_20 : x > 20) :
  let option1_cost := 40 * x + 3200,
      option2_cost := 36 * x + 3600
  in
  (option1_cost = 40 * x + 3200) ∧ 
  (option2_cost = 36 * x + 3600) ∧ 
  (option1_cost < option2_cost → x = 30) :=
by
  sorry

end sports_shopping_costs_l519_519498


namespace percentage_female_officers_on_duty_l519_519009

theorem percentage_female_officers_on_duty:
  ∀ (total_on_duty female_on_duty total_female_officers : ℕ),
    total_on_duty = 160 →
    female_on_duty = total_on_duty / 2 →
    total_female_officers = 500 →
    female_on_duty / total_female_officers * 100 = 16 :=
by
  intros total_on_duty female_on_duty total_female_officers h1 h2 h3
  -- Ensure types are correct
  change total_on_duty = 160 at h1
  change female_on_duty = total_on_duty / 2 at h2
  change total_female_officers = 500 at h3
  sorry

end percentage_female_officers_on_duty_l519_519009


namespace num_valid_arrangements_l519_519585

def men := {1, 2, 3, 4}
def women := {1, 2, 3, 4}
def alternating (arrangement : List (ℕ × String)) : Prop :=
  ∀ i < arrangement.length, 
  (arrangement.nth i).get.snd = "man" → (arrangement.nth (i+1 % arrangement.length)).get.snd = "woman"

def not_same_number (arrangement : List (ℕ × String)) : Prop :=
  ∀ i < arrangement.length, 
  (arrangement.nth i).get.fst ≠ (arrangement.nth (i+1 % arrangement.length)).get.fst

def valid_arrangement (arrangement : List (ℕ × String)) : Prop :=
  alternating arrangement ∧ not_same_number arrangement

theorem num_valid_arrangements : 
  ∃ (arrangements : List (List (ℕ × String))), 
  (∀ a ∈ arrangements, valid_arrangement a) ∧ arrangements.length = 12 :=
sorry

end num_valid_arrangements_l519_519585


namespace path_between_male_and_female_stationmasters_l519_519055

  -- Define the structure of the station and stationmasters
  structure Station := 
    (id : ℕ)
    (is_male_stationmaster : Bool)

  -- Assume the conditions
  variables (n : ℕ) [hn : fact (n % 2 = 1)] -- n is odd
  variables (stations : fin n → Station) -- a list of stations indexed by fin n
  variables (A B : fin n) (hAB : A ≠ B) -- Metro Line 1 connects A and B
  variables (has_male_stationmaster : ∃ i, (stations i).is_male_stationmaster = true)
  variables (has_female_stationmaster : ∃ i, (stations i).is_male_stationmaster = false)

  -- The statement we need to prove
  theorem path_between_male_and_female_stationmasters (k : ℕ) (hk : 0 < k ∧ k < n) :
    ∃ start end : fin n, 
      (stations start).is_male_stationmaster = true ∧
      (stations end).is_male_stationmaster = false ∧
      -- Path from start to end through k metro intervals (implicitly defined in the problem setup)
      sorry :=
  sorry
  
end path_between_male_and_female_stationmasters_l519_519055


namespace children_vehicle_wheels_l519_519526

theorem children_vehicle_wheels:
  ∀ (x : ℕ),
    (6 * 2) + (15 * x) = 57 →
    x = 3 :=
by
  intros x h
  sorry

end children_vehicle_wheels_l519_519526


namespace five_circles_intersect_l519_519010

-- Assume we have five circles
variables (circle1 circle2 circle3 circle4 circle5 : Set Point)

-- Assume every four of them intersect at a single point
axiom four_intersect (c1 c2 c3 c4 : Set Point) : ∃ p : Point, p ∈ c1 ∧ p ∈ c2 ∧ p ∈ c3 ∧ p ∈ c4

-- The goal is to prove that there exists a point through which all five circles pass.
theorem five_circles_intersect :
  (∃ p : Point, p ∈ circle1 ∧ p ∈ circle2 ∧ p ∈ circle3 ∧ p ∈ circle4 ∧ p ∈ circle5) :=
sorry

end five_circles_intersect_l519_519010


namespace selling_price_is_180_l519_519124

-- Definitions for the conditions
def gain : ℝ := 30
def gain_percentage : ℝ := 20
def cost_price : ℝ := gain / (gain_percentage / 100)
def selling_price : ℝ := cost_price + gain

-- Theorem statement
theorem selling_price_is_180 : selling_price = 180 :=
by
  unfold selling_price cost_price gain gain_percentage
  sorry

end selling_price_is_180_l519_519124


namespace width_of_final_painting_l519_519471

theorem width_of_final_painting
  (total_area : ℕ)
  (area_paintings_5x5 : ℕ)
  (num_paintings_5x5 : ℕ)
  (painting_10x8_area : ℕ)
  (final_painting_height : ℕ)
  (total_num_paintings : ℕ := 5)
  (total_area_paintings : ℕ := 200)
  (calculated_area_remaining : ℕ := total_area - (num_paintings_5x5 * area_paintings_5x5 + painting_10x8_area))
  (final_painting_width : ℕ := calculated_area_remaining / final_painting_height) :
  total_num_paintings = 5 →
  total_area = 200 →
  area_paintings_5x5 = 25 →
  num_paintings_5x5 = 3 →
  painting_10x8_area = 80 →
  final_painting_height = 5 →
  final_painting_width = 9 :=
by
  intros h1 h2 h3 h4 h5 h6
  dsimp [final_painting_width, calculated_area_remaining]
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end width_of_final_painting_l519_519471


namespace john_subtracts_number_from_50_squared_l519_519059

theorem john_subtracts_number_from_50_squared :
  ∀ (x : ℤ), x^2 - (50^2 - 49^2) = 49^2  → x = 99 :=
by
  intro x
  calc
    49^2 = (50-1)^2 : by rw [sub_self, pow_two, pow_two]
        ... = 50^2 - 2*50*1 + 1^2 : by norm_num
        ... = 50^2 - 100 + 1 : by norm_num
        ... = 50^2 - 99 : by norm_num
  sorry

end john_subtracts_number_from_50_squared_l519_519059


namespace zoe_boxes_found_l519_519090

/-- Zoe was unboxing some of her old winter clothes.
She found some boxes of clothing and inside each box there were 4 scarves and 6 mittens.
Zoe had a total of 80 pieces of winter clothing.
How many boxes did Zoe find? -/
theorem zoe_boxes_found :
  ∃ (B : ℕ), (∀ (scarves mittens total_pieces : ℕ),
    scarves = 4 -> mittens = 6 ->
    total_pieces = 80 ->
    total_pieces = B * (scarves + mittens))
  ∧ B = 8 :=
begin
  use 8,
  intros,
  intros,
  sorry
end

end zoe_boxes_found_l519_519090


namespace rhombus_longer_diagonal_l519_519131

theorem rhombus_longer_diagonal (a b d : ℝ) (h : a = 65) (h_d : d = 56) :
  ∃ l, l = 118 :=
by
  sorry

end rhombus_longer_diagonal_l519_519131


namespace tent_capacity_l519_519057

theorem tent_capacity : 
  let section1 := 246
  let section2 := 246
  let section3 := 246
  let section4 := 246
  let section5 := 314
  section1 + section2 + section3 + section4 + section5 = 1298 :=
by
  let section1 := 246
  let section2 := 246
  let section3 := 246
  let section4 := 246
  let section5 := 314
  have h1 : section1 + section2 + section3 + section4 = 984 := rfl
  have h2 : 984 + section5 = 1298 := rfl
  exact Eq.trans h1 h2

end tent_capacity_l519_519057


namespace no_prime_divisible_by_77_l519_519283

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → is_divisible p 77 → false :=
by
  sorry

end no_prime_divisible_by_77_l519_519283


namespace tom_savings_by_having_insurance_l519_519062

noncomputable def insurance_cost_per_month : ℝ := 20
noncomputable def total_months : ℕ := 24
noncomputable def surgery_cost : ℝ := 5000
noncomputable def insurance_coverage_rate : ℝ := 0.80

theorem tom_savings_by_having_insurance :
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  savings = 3520 :=
by
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  sorry

end tom_savings_by_having_insurance_l519_519062


namespace volume_ratio_octahedron_cube_l519_519604

theorem volume_ratio_octahedron_cube 
  (s : ℝ) -- edge length of the octahedron
  (h := s * Real.sqrt 2 / 2) -- height of one of the pyramids forming the octahedron
  (volume_O := s^3 * Real.sqrt 2 / 3) -- volume of the octahedron
  (a := (2 * s) / Real.sqrt 3) -- edge length of the cube
  (volume_C := (a ^ 3)) -- volume of the cube
  (diag_C : ℝ := 2 * s) -- diagonal of the cube
  (h_diag : diag_C = (a * Real.sqrt 3)) -- relation of diagonal to edge length of the cube
  (ratio := volume_O / volume_C) -- ratio of the volumes
  (desired_ratio := 3 / 8) -- given ratio in simplified form
  (m := 3) -- first part of the ratio
  (n := 8) -- second part of the ratio
  (rel_prime : Nat.gcd m n = 1) -- m and n are relatively prime
  (correct_ratio : ratio = desired_ratio) -- the ratio is correct
  : m + n = 11 :=
by
  sorry 

end volume_ratio_octahedron_cube_l519_519604


namespace inhabitant_knows_at_least_810_l519_519745

open Function

def population : ℕ := 1000000

def believers_claus (p : ℕ) : Prop := p = (90 * population / 100)

def knows_someone (p : ℕ) (knows : p → p → Prop) : Prop := 
  ∀ x : p, ∃ y : p, knows x y

def acquaintances_claus (p : ℕ) (believes : p → Prop) (knows : p → p → Prop) : Prop :=
  ∀ x : p, (believes x →(∃ k : ℕ, ∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y ∧ (10 * k / 100) y ∧ ¬believes y))

theorem inhabitant_knows_at_least_810 :
  ∃ (p : ℕ) (believes : p → Prop) (knows : p → p → Prop),
    population = 1000000 ∧
    believers_claus p ∧
    knows_someone p knows ∧
    acquaintances_claus p believes knows →
    ∃ x : p, ∃ k : ℕ, k ≥ 810 ∧ (∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y) :=
sorry

end inhabitant_knows_at_least_810_l519_519745


namespace chocolate_probability_sum_l519_519814

-- Condition: Define the setup of the rectangular chocolate bar, the point selections, and the division into trapezoidal pieces
def chocolate_bar (dark_left_points white_right_points : Finset ℝ) : Prop :=
  dark_left_points.card = 4 ∧ white_right_points.card = 4 ∧ 
  ∀ p ∈ dark_left_points ∪ white_right_points, 0 ≤ p ∧ p ≤ 1

-- Given the defined chocolate bar and point selections, prove the desired sum
theorem chocolate_probability_sum (m n : ℕ) (h : Nat.gcd m n = 1) 
  (dark_left_points white_right_points : Finset ℝ) 
  (h_chocolate : chocolate_bar dark_left_points white_right_points) : 
  (∃ (prob : ℚ), prob = 7 / 32) → m + n = 39 := 
by
  -- Define the probability such that the sum of m and n is 39
  have h_prob : (7 : ℚ) / (32 : ℚ) = (7 / 32) := by norm_num
  intro hprob
  cases hprob with prob hprob_eq
  rw hprob_eq at h_prob
  sorry

end chocolate_probability_sum_l519_519814


namespace bisection_next_interval_l519_519073

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 4

theorem bisection_next_interval :
  (0 < 1) →
  f 0 > 0 →
  f 1 < 0 →
  f (1/2) > 0 →
  ∃ x ∈ (0..1), f x = 0 →
  ∃ x ∈ (1/2..1), f x = 0 :=
by
  -- The proof would go here.
  sorry

end bisection_next_interval_l519_519073


namespace no_intersecting_segments_l519_519821

-- Definitions for the points and non-collinearity property
def nonCollinear (A B C : Point) : Prop := ¬ collinear A B C

-- Formal statement of the theorem
theorem no_intersecting_segments (A B : Fin n → Point) 
  (h : ∀ i j k, (i ≠ j ∧ j ≠ k ∧ i ≠ k) → ¬ collinear (A i) (B j) (A k)) :
  ∃ σ : Fin n → Fin n, 
    ∀ i j, i ≠ j → ¬ intersect (Segment.mk (A i) (B (σ i))) (Segment.mk (A j) (B (σ j))) :=
sorry

end no_intersecting_segments_l519_519821


namespace f_monotone_decreasing_inequality_solution_l519_519664

-- Part one: monotonicity of f(x)
def f (x : ℝ) : ℝ := real.log ((x + 1) / (x - 1))

theorem f_monotone_decreasing :
  ∀ x1 x2 : ℝ, (1 < x1) → (1 < x2) → (x1 < x2) → (f x2 < f x1) :=
sorry

-- Part two: solving the inequality
theorem inequality_solution (x : ℝ) :
  (f (x^2 + x + 3) + f (-2*x^2 + 4*x - 7) > 0) ↔ (x < 1 ∨ x > 4) :=
sorry

end f_monotone_decreasing_inequality_solution_l519_519664


namespace sparrows_gather_on_one_tree_l519_519115

/-
There are 64 sparrows distributed among some trees. 

1. If there are at least half of all the birds (32 sparrows) on any tree, 
   then exactly as many sparrows fly from that tree to each of the other trees 
   as there are already sitting on those trees (this counts as one flight).
2. If there are exactly 32 sparrows on two trees, sparrows from one of these 
   trees fly off following the same rule.
3. Initially, the sparrows are distributed such that the flock makes exactly 
   6 flights before calming down.
-/

-- Definitions
def num_sparrows : Nat := 64
def num_flights_to_calm_down : Nat := 6

-- Theorem: After 6 flights, all the sparrows are on one tree.
theorem sparrows_gather_on_one_tree 
  (sparrows : Nat) 
  (flights : Nat) 
  (initial_distribution : List Nat) 
  (flight_rule : (l : List Nat) → List Nat)
  (calm_condition : ∀ l, flights == 6 → l.sum == sparrows → l.filter (λ n => n > 0).length == 1) : 
  List.filter (λ n => n > 0) (n : List Nat) = [num_sparrows] := 
sorry

end sparrows_gather_on_one_tree_l519_519115


namespace inequality_proof_l519_519686

-- Given conditions
variables {a b : ℝ} (ha_lt_b : a < b) (hb_lt_0 : b < 0)

-- Question statement we want to prove
theorem inequality_proof : ab < 0 → a < b → b < 0 → ab > b^2 :=
by
  sorry

end inequality_proof_l519_519686


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519728

theorem exists_inhabitant_with_at_least_810_acquaintances :
  (∀ x, ∃ y, x ≠ y) → 
  (∀ x, (∀ k, k = 10% * |neighbors x| → |{y : neighbors x | believes_in_santa y}| = k)) →
  (1_000_000 ≥ ∑ x, believes_in_santa x → 900_000) → 
  (1_000_000 - 900_000 = 100_000) →
  (∃ x, |neighbors x| ≥ 810) :=
sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519728


namespace S_is_even_l519_519818

-- Given n is a natural number
def S (n : ℕ) : ℕ := (Finset.range n.succ).sum (λ k => n / (k + 1)) + Int.natAbs (Int.floor (Real.sqrt (n)))

theorem S_is_even (n : ℕ) : Even (S n) := 
by
  sorry

end S_is_even_l519_519818


namespace remainder_when_three_times_number_minus_seven_divided_by_seven_l519_519922

theorem remainder_when_three_times_number_minus_seven_divided_by_seven (n : ℤ) (h : n % 7 = 2) : (3 * n - 7) % 7 = 6 := by
  sorry

end remainder_when_three_times_number_minus_seven_divided_by_seven_l519_519922


namespace unique_solution_l519_519206

noncomputable def functional_eq (f : ℝ → ℝ) : Prop :=
  (f 1 = 1) ∧ 
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ 
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = f x / (x * x))

theorem unique_solution (f : ℝ → ℝ) : functional_eq f → (∀ x : ℝ, f x = x) :=
begin
  intro h,
  sorry
end

end unique_solution_l519_519206


namespace simplify_expression_l519_519390

def omega : ℂ := (-1 + complex.i * real.sqrt 7) / 2
def omega_star : ℂ := (-1 - complex.i * real.sqrt 7) / 2

theorem simplify_expression : omega^4 + omega_star^4 = 8 :=
by
  sorry

end simplify_expression_l519_519390


namespace total_spending_in_4_years_l519_519908

def trevor_spending_per_year : ℕ := 80
def reed_to_trevor_diff : ℕ := 20
def reed_to_quinn_factor : ℕ := 2

theorem total_spending_in_4_years :
  ∃ (reed_spending quinn_spending : ℕ),
  (reed_spending = trevor_spending_per_year - reed_to_trevor_diff) ∧
  (reed_spending = reed_to_quinn_factor * quinn_spending) ∧
  ((trevor_spending_per_year + reed_spending + quinn_spending) * 4 = 680) :=
sorry

end total_spending_in_4_years_l519_519908


namespace find_some_number_l519_519255

theorem find_some_number :
  ∃ some_number : ℝ, (3.242 * 10 / some_number) = 0.032420000000000004 ∧ some_number = 1000 :=
by
  sorry

end find_some_number_l519_519255


namespace limit_computation_l519_519990

noncomputable def limit_expr (x : ℝ) : ℝ :=
  ((1 + x * 2^x) / (1 + x * 3^x))^(1/(x^2))

theorem limit_computation : 
  tendsto (λ x : ℝ, limit_expr x) (𝓝 0) (𝓝 (2/3)) :=
sorry

end limit_computation_l519_519990


namespace average_salary_l519_519035

theorem average_salary (total_workers technicians : ℕ) (avg_salary_technicians avg_salary_rest total_salary : ℕ) :
  total_workers = 42 → 
  technicians = 7 → 
  avg_salary_technicians = 18000 → 
  avg_salary_rest = 6000 → 
  total_salary = technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest →
  total_salary / total_workers = 8000 := 
by
  intros h_total_workers h_technicians h_avg_salary_technicians h_avg_salary_rest h_total_salary
  rw [h_total_workers, h_technicians, h_avg_salary_technicians, h_avg_salary_rest] at h_total_salary
  have h_total_salary_calc : total_salary = 7 * 18000 + (42 - 7) * 6000 := by rw [h_technicians, h_avg_salary_technicians, h_avg_salary_rest]; simp
  rw h_total_salary_calc at h_total_salary
  have h_quot : total_salary / 42 = 8000 := by norm_num
  exact h_quot

end average_salary_l519_519035


namespace number_of_people_in_team_l519_519151

variables (n : ℕ)

-- Conditions
def best_marksman_scored_85 : Prop := ∃ s : ℕ, s = 85
def scored_92_points_average_84 : Prop := ∃ t : ℕ, t = 84 * n
def team_total_score_497 : Prop := ∃ u : ℕ, u = 497

-- Prove number of people in the team
theorem number_of_people_in_team (h1 : best_marksman_scored_85 n) 
  (h2 : scored_92_points_average_84 n) 
  (h3 : team_total_score_497 n) : n = 6 :=
sorry

end number_of_people_in_team_l519_519151


namespace max_radius_squared_l519_519069

-- Define the main properties of the cones and the distances given
def base_radius := 4
def height := 10
def intersection_distance := 4

-- Prove the maximum value of r² given these conditions
theorem max_radius_squared :
  ∃ (r : ℝ), r^2 = 144 / 29 ∧
  (∀(cone_base_radius cone_height intersection_dist : ℝ),
    cone_base_radius = base_radius →
    cone_height = height →
    intersection_dist = intersection_distance →
    r ≤ 4 * 6 / Real.sqrt ((10^2) + (4^2))) :=
begin
  -- statement only, proof omitted
  sorry
end

end max_radius_squared_l519_519069


namespace sum_of_x_coordinates_above_line_l519_519542

def Points : List (ℕ × ℕ) := [(2, 8), (7, 19), (11, 36), (17, 39), (21, 48)]

def above_line (p : ℕ × ℕ) : Prop :=
  let (x, y) := p
  y > 3 * x + 4

theorem sum_of_x_coordinates_above_line :
  (Points.filter above_line).map Prod.fst).sum = 0 := by
  sorry

end sum_of_x_coordinates_above_line_l519_519542


namespace sticks_difference_l519_519997

-- Definitions of the conditions
def d := 14  -- number of sticks Dave picked up
def a := 9   -- number of sticks Amy picked up
def total := 50  -- initial total number of sticks in the yard

-- The proof problem statement
theorem sticks_difference : (d + a) - (total - (d + a)) = 4 :=
by
  sorry

end sticks_difference_l519_519997


namespace part1_part2_l519_519667

section
variables {x m a b : ℝ} (m_int : m ∈ Int) (m_ab : a * b = m) (a_gt_b : a > b ∧ b > 0) 

-- Given |2x - m| < 1, x has exactly one integer solution x = 2 
theorem part1 (h1 : |2 * (2 : ℝ) - m| < 1) : m = 4 :=
sorry

-- Given ab = m, a > b > 0, prove (a^2 + b^2) / (a - b) ≥ 4√2
theorem part2 (h2 : a * b = 4) (h3 : a > b ∧ b > 0) : (a^2 + b^2) / (a - b) ≥ 4 * Real.sqrt 2 :=
sorry
end

end part1_part2_l519_519667


namespace at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519718

theorem at_least_one_inhabitant_knows_no_fewer_than_810_people :
  ∀ (n : ℕ),
  (inhabitants : ℕ) (knows_someone : inhabitants > 1) (believes_santa : inhabitants * 9 / 10)
  (ten_percent_acquaintances_believe : ∀ x, (acquaintances : list x) → (|list.contacts_with_santa <- 0.10 * acquaintances|)),
  (h : inhabitants = 1000000),
  (one_inhabitant_knows : ∃ x, ∀ y, y \in acquaintances x → friends y ≥ 810) :=
sorry

end at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519718


namespace inconsistent_equation_system_l519_519698

variables {a x c : ℝ}

theorem inconsistent_equation_system (h1 : (a + x) / 2 = 110) (h2 : (x + c) / 2 = 170) (h3 : a - c = 120) : false :=
by
  sorry

end inconsistent_equation_system_l519_519698


namespace axis_of_symmetry_monotonic_intervals_max_min_values_l519_519221

noncomputable def f (x : ℝ) : ℝ := real.sqrt 2 * real.sin (2 * x + real.pi / 4)

-- 1. Axis of symmetry
theorem axis_of_symmetry : ∃ k : ℤ, ∀ x : ℝ, (f x) = (f ((k : ℝ) * real.pi / 2 + real.pi / 8)) := sorry

-- 2. Monotonically increasing intervals
theorem monotonic_intervals : ∃ k : ℤ, ∀ x : ℝ, k * real.pi - 3 * real.pi / 8 ≤ x ∧ x ≤ k * real.pi + real.pi / 8 → 
  monotone_on (λ x, f x) (set.Icc (k * real.pi - 3 * real.pi / 8) (k * real.pi + real.pi / 8)) := sorry

-- 3. Maximum and minimum values in specific interval
theorem max_min_values : 
  ∃ (max_val min_val : ℝ), max_val = 1 ∧ min_val = -real.sqrt 2 ∧ 
  ∀ x : ℝ, x ∈ set.Icc (real.pi / 4) (3 * real.pi / 4) → 
    f x ≤ max_val ∧ f x ≥ min_val := sorry

end axis_of_symmetry_monotonic_intervals_max_min_values_l519_519221


namespace correct_statement_of_A_l519_519086

theorem correct_statement_of_A (
    (A : "In the test of independence, the larger the observed value of the random variable K^{2}, the smaller the probability of making the judgment 'the two categorical variables are related.'") :
    false
) (
    (B : "Given X∼ N(μ, σ^{2}), when μ remains constant, the larger σ is, the higher and thinner the normal density curve of X.") :
    false
) (
    (C : "If there exist three non-collinear points in plane α whose distances to plane β are equal, then plane α is parallel to plane β.") :
    false
) (
    (D : "If plane α is perpendicular to plane β, line m is perpendicular to α, and line n is parallel to m, then line n may be parallel to plane β.") :
    false
) :
    (A = true) :=
sorry

end correct_statement_of_A_l519_519086


namespace field_trip_students_l519_519431

theorem field_trip_students (seats_per_bus : ℕ) (number_of_buses : ℕ) (total_students : ℕ) 
    (h1 : seats_per_bus = 10) (h2 : number_of_buses = 6) : 
    total_students = 60 :=
by
  rw [h1, h2]
  exact rfl

end field_trip_students_l519_519431


namespace incorrect_statement_C_l519_519665

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem incorrect_statement_C : ¬(∀ x : ℝ, f(Real.pi / 4) = 0 → ∃ y, f(y) = f(x) ∧ x ≠ y) :=
by sorry

end incorrect_statement_C_l519_519665


namespace number_of_common_tangents_is_four_l519_519412

-- define the first circle C1: x^2 + y^2 + 2x + 2y - 2 = 0
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0

-- define the second circle C2: x^2 + y^2 - 4x - 2y + 4 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 4 = 0

-- define the proof problem
theorem number_of_common_tangents_is_four : 
  (∃ x y : ℝ, C1 x y ∧ C2 x y) → 
  (¬(∃ x y : ℝ, C1 x y ∧ C2 x y) → 4) := 
sorry

end number_of_common_tangents_is_four_l519_519412


namespace exists_inhabitant_with_810_acquaintances_l519_519755

-- Conditions
def total_population : ℕ := 1000000
def believers_percentage : ℕ := 90
def acquaintances_believers_percentage : ℝ := 0.10

-- Theorem statement: Prove that there exists an inhabitant knowing at least 810 people.
theorem exists_inhabitant_with_810_acquaintances 
    (H1 : ∀ (p : ℕ), p < total_population → ∃ q, q ≠ p)  -- Every person knows at least one other person
    (H2 : 0.9 * total_population = 900000)  -- 90% of the population are believers in Santa Claus
    (H3 : ∀ p, p < total_population → ∀ q, q != p → (belief p → (exists percentage, percentage = acquaintances_believers_percentage))) -- Every person claims exactly 10% of their acquaintances believe in Santa Claus
    : ∃ (habitants : ℕ), habitants ≥ 810
    :=
  sorry

end exists_inhabitant_with_810_acquaintances_l519_519755


namespace subtract_two_decimals_l519_519169

theorem subtract_two_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_two_decimals_l519_519169


namespace number_of_pairs_l519_519273

theorem number_of_pairs (x y : ℕ) (h_pos : 0 < x) (h_pos_y : 0 < y) (h_eq : x^2 - y^2 = 171) :
    ∃ (x₁ y₁ x₂ y₂ : ℕ), 0 < x₁ ∧ 0 < y₁ ∧ 0 < x₂ ∧ 0 < y₂ ∧
    (x₁^2 - y₁^2 = 171) ∧ (x₂^2 - y₂^2 = 171) ∧
    ((x, y) = (x₁, y₁) ∨ (x, y) = (x₂, y₂)) ∧
    x₁ ≠ x₂ ∧ y₁ ≠ y₂

end number_of_pairs_l519_519273


namespace common_difference_is_2_l519_519437

variable (S : ℕ → ℝ) -- S_n being the sum of the first n terms of an arithmetic sequence
variable (a₁ d : ℝ) -- a₁ is the first term, d is the common difference

-- Defining the sum S_n (nth partial sum) of an arithmetic sequence
def S_n (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

-- Given condition
axiom S6_eq_3S2_plus_24 : S_n 6 = 3 * S_n 2 + 24

theorem common_difference_is_2 : d = 2 :=
by
  sorry

end common_difference_is_2_l519_519437


namespace cyclic_shift_diagonal_nonnegative_l519_519310

-- Define a cyclic shift of columns for an n x n matrix
def cyclic_shift {α : Type*} [AddZeroClass α] (M : Matrix (Fin n) (Fin n) α) : Matrix (Fin n) (Fin n) α :=
  λ i j, M i ((j + 1) % n)

-- Define the sum of the main diagonal from bottom left to top right of an n x n matrix
def diagonal_sum {α : Type*} [AddZeroClass α] (M : Matrix (Fin n) (Fin n) α) : α :=
  ∑ i, M (Fin.ofNat i) (n - 1 - i)

-- Problem statement: Given a matrix M with non-negative sum, prove there exists a cyclic 
-- shift permutation of the columns where the diagonal sum is non-negative.
theorem cyclic_shift_diagonal_nonnegative {α : Type*} [OrderedAddCommMonoid α]
  (M : Matrix (Fin n) (Fin n) α) (hM : 0 ≤ ∑ i j, M i j) :
  ∃ k < n, 0 ≤ diagonal_sum (iterated (cyclic_shift M) k) :=
begin
  sorry
end

end cyclic_shift_diagonal_nonnegative_l519_519310


namespace tangent_line_eq_l519_519417

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

def M : ℝ×ℝ := (2, -3)

theorem tangent_line_eq (x y : ℝ) (h : y = f x) (h' : (x, y) = M) :
  2 * x - y - 7 = 0 :=
sorry

end tangent_line_eq_l519_519417


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519725

theorem exists_inhabitant_with_at_least_810_acquaintances :
  (∀ x, ∃ y, x ≠ y) → 
  (∀ x, (∀ k, k = 10% * |neighbors x| → |{y : neighbors x | believes_in_santa y}| = k)) →
  (1_000_000 ≥ ∑ x, believes_in_santa x → 900_000) → 
  (1_000_000 - 900_000 = 100_000) →
  (∃ x, |neighbors x| ≥ 810) :=
sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519725


namespace range_of_a_product_of_zeros_l519_519659

def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - (Real.log x) + x - a

theorem range_of_a (a : ℝ) : (∀ x, (0 < x) → (f x a ≥ 0)) ↔ (a ≤ Real.exp 1 + 1) :=
by
  sorry

theorem product_of_zeros (x1 x2 a : ℝ) : 
  (0 < x1) → (x1 < 1) → (1 < x2) → (f x1 a = 0 ∧ f x2 a = 0) → x1 * x2 < 1 :=
by
  sorry

end range_of_a_product_of_zeros_l519_519659


namespace exists_inhabitant_with_many_acquaintances_l519_519757

open Function

theorem exists_inhabitant_with_many_acquaintances :
  (∃ (n : ℕ), n = 1_000_000) →
  (∀ (p : ℕ), p < 1_000_000 → ∃ (q : ℕ), q ≠ p) →
  (∀ (p : ℕ), p < 900_000 → p <= 900_000) →
  (∀ (p : ℕ), p < 1_000_000 → 0.1 * real.to_real p < 0.9 * real.to_real p) →
  ∃ (p : ℕ), p >= 810 :=
by
  intros h1 h2 h3 h4
  sorry

end exists_inhabitant_with_many_acquaintances_l519_519757


namespace greatest_prime_factor_of_S_l519_519595

def non_zero_digits_product (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (λ d, d ≠ 0) |>.prod

def S : ℕ := (List.range' 1 999).sum (λ n, non_zero_digits_product n)

theorem greatest_prime_factor_of_S :
  nat.greatest_prime_factor S = 103 :=
by sorry

end greatest_prime_factor_of_S_l519_519595


namespace common_divisors_count_l519_519680

-- Definitions of factorizations
def factorization_9240 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1), (7, 1), (11, 1)]
def factorization_10080 : List (ℕ × ℕ) := [(2, 5), (3, 2), (5, 1), (7, 1)]

-- Definition of GCD from the factorizations
def gcd_from_factors (f1 f2 : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  f1.foldr (λ (p : ℕ × ℕ) acc =>
    if h : (List.find? (fun (q : ℕ × ℕ) => q.1 = p.1) f2).isSome then
      let (q, hq) := Option.getEqSome h
      (p.1, min p.2 q.2) :: acc
    else acc) []

-- Computing GCD of factorizations of 9240 and 10080
def gcd_9240_10080 := gcd_from_factors factorization_9240 factorization_10080

-- Definition of common divisor count based on GCD factorization
def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ p acc => acc * (p.2 + 1)) 1

-- Actual number of common divisors
def actual_common_divisors := num_divisors gcd_9240_10080

-- Proof goal: common positive divisors count is 48
theorem common_divisors_count : actual_common_divisors = 48 := by
  sorry

end common_divisors_count_l519_519680


namespace geom_seq_mult_l519_519239

variable {α : Type*} [LinearOrderedField α]

def is_geom_seq (a : ℕ → α) :=
  ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem geom_seq_mult (a : ℕ → α) (h : is_geom_seq a) (hpos : ∀ n, 0 < a n) (h4_8 : a 4 * a 8 = 4) :
  a 5 * a 6 * a 7 = 8 := 
sorry

end geom_seq_mult_l519_519239


namespace perp_intersect_at_single_point_l519_519111

open EuclideanGeometry

variables (A B C C₁ C₂ A₁ A₂ B₁ B₂ P Q : Point)
variables {AB BC CA : Line}
variables (hABC : Triangle A B C)
variables (hC1 : C₁ ∈ AB) (hC2 : C₂ ∈ AB)
variables (hA1 : A₁ ∈ BC) (hA2 : A₂ ∈ BC)
variables (hB1 : B₁ ∈ CA) (hB2 : B₂ ∈ CA)
variables (hP : IsIntersectionPoint (Perpendicular AB C₁) (Perpendicular CA B₁) (Perpendicular BC A₁) P)

theorem perp_intersect_at_single_point :
  IsIntersectionPoint (Perpendicular AB C₂) (Perpendicular CA B₂) (Perpendicular BC A₂) Q := 
sorry

end perp_intersect_at_single_point_l519_519111


namespace range_of_a_product_of_zeros_l519_519662

def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - (Real.log x) + x - a

theorem range_of_a (a : ℝ) : (∀ x, (0 < x) → (f x a ≥ 0)) ↔ (a ≤ Real.exp 1 + 1) :=
by
  sorry

theorem product_of_zeros (x1 x2 a : ℝ) : 
  (0 < x1) → (x1 < 1) → (1 < x2) → (f x1 a = 0 ∧ f x2 a = 0) → x1 * x2 < 1 :=
by
  sorry

end range_of_a_product_of_zeros_l519_519662


namespace tan_alpha_sub_beta_l519_519238

theorem tan_alpha_sub_beta (α β : ℝ) (h₁ : Real.tan α = 9) (h₂ : Real.tan β = 6) : Real.tan (α - β) = 3 / 55 := 
sorry

end tan_alpha_sub_beta_l519_519238


namespace range_of_a_l519_519587

section
variable {a : ℝ}

-- Define set A
def A : set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define function f and set B
def f (x : ℝ) : ℝ := x^2 + 2*x + a
def B : set ℝ := {x | f x ≥ 0}

-- Define non-empty intersection condition
def non_empty_intersection (A B : set ℝ) : Prop := ∃ x, x ∈ A ∧ x ∈ B

-- Main theorem statement
theorem range_of_a (h : non_empty_intersection A B) : a > 1 := sorry

end

end range_of_a_l519_519587


namespace find_m_from_ellipse_l519_519621

theorem find_m_from_ellipse (m : ℝ) (h_ellipse_eq : ∀ x y : ℝ, (x^2) / (10 - m) + (y^2) / (m - 2) = 1)
  (h_major_axis : /* proof that major axis is on x-axis */ sorry) 
  (h_focal_distance : 2 * real.sqrt 4 = 4) : 
  m = 4 :=
sorry

end find_m_from_ellipse_l519_519621


namespace number_of_sides_l519_519896

theorem number_of_sides (n : ℕ) : 
  (2 / 9) * (n - 2) * 180 = 360 → n = 11 := 
by
  intro h
  sorry

end number_of_sides_l519_519896


namespace find_a_plus_b_l519_519616

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 2^(a * x + b)

theorem find_a_plus_b
  (a b : ℝ)
  (h1 : f a b 2 = 1 / 2)
  (h2 : f a b (1 / 2) = 2) :
  a + b = 1 / 3 :=
sorry

end find_a_plus_b_l519_519616


namespace interval_of_monotonic_increase_l519_519192

def f (x : ℝ) : ℝ := log x - x

theorem interval_of_monotonic_increase :
  ∀ x : ℝ, 0 < x ∧ x < 1 → deriv f x > 0 :=
begin
  sorry
end

end interval_of_monotonic_increase_l519_519192


namespace area_calculation_error_l519_519093

variable (x : ℝ) (h : 0 < x)

def edge_with_error : ℝ := 1.02 * x
def actual_area : ℝ := x^2
def calculated_area : ℝ := (edge_with_error x)^2
def error : ℝ := calculated_area x - actual_area x
def percentage_error : ℝ := (error x / actual_area x) * 100

theorem area_calculation_error : percentage_error x = 4.04 := by
  sorry

end area_calculation_error_l519_519093


namespace acquaintance_paradox_proof_l519_519715

theorem acquaintance_paradox_proof :
  ∃ (x : ℕ), (acquaintances x ≥ 810) :=
by
  let n_population := 1000000
  let n_believers := 900000
  let n_nonBelievers := n_population - n_believers
  have h1 : n_population = 1000000 := rfl
  have h2 : n_believers = 0.90 * n_population := by sorry
  have h3 : n_nonBelievers = n_population - n_believers := by sorry
  have h4 : ∀ x, acquaintances x ≥ 1 := by sorry
  have h5 : ∀ x, 0.10 * acquaintances x = nonBelievers_acquaintances x := by sorry
  have h6 : nonBelievers_acquaintances_total >= 90 := by sorry
  
  show ∃ (x : ℕ), acquaintances x ≥ 810 from sorry
  
  sorry

end acquaintance_paradox_proof_l519_519715


namespace unique_three_topping_pizzas_l519_519125

theorem unique_three_topping_pizzas (n k : ℕ) (h_n : n = 7) (h_k : k = 3) :
  (nat.choose n k) = 35 :=
by sorry

end unique_three_topping_pizzas_l519_519125


namespace lines_intersect_at_single_point_l519_519995

theorem lines_intersect_at_single_point :
  (m : ℚ) (h : ∃ (p : ℚ × ℚ), p ∈ {x | x.snd = 3 * x.fst + 2} ∧ 
              p ∈ {x | x.snd = -4 * x.fst + 10} ∧ 
              p ∈ {x | x.snd = 2 * x.fst + m}) -> 
  m = 22 / 7 :=
by
  intro m h
  obtain ⟨p, hp1, hp2, hp3⟩ := h
  sorry

end lines_intersect_at_single_point_l519_519995


namespace perimeter_of_shaded_region_is_48_l519_519793

noncomputable def find_perimeter_of_shaded_region
  (n : ℕ)
  (circumference : ℝ)
  (angle : ℝ) :
  ℝ :=
  let r := circumference / (2 * real.pi) in
  let arc_length := angle / 360 * 2 * real.pi * r in
  n * arc_length

theorem perimeter_of_shaded_region_is_48 :
  find_perimeter_of_shaded_region 3 48 120 = 48 :=
by
  -- all necessary intermediate steps will go here
  sorry

end perimeter_of_shaded_region_is_48_l519_519793


namespace no_polyhedron_without_triangles_and_three_valent_vertices_l519_519382

-- Definitions and assumptions based on the problem's conditions
def f_3 := 0 -- no triangular faces
def p_3 := 0 -- no vertices with degree three

-- Euler's formula for convex polyhedra
def euler_formula (f p a : ℕ) : Prop := f + p - a = 2

-- Define general properties for faces and vertices in polyhedra
def polyhedron_no_triangular_no_three_valent (f p a f_4 f_5 p_4 p_5: ℕ) : Prop :=
  f_3 = 0 ∧ p_3 = 0 ∧ 2 * a ≥ 4 * (f_4 + f_5) ∧ 2 * a ≥ 4 * (p_4 + p_5) ∧ euler_formula f p a

-- Theorem to prove there does not exist such a polyhedron
theorem no_polyhedron_without_triangles_and_three_valent_vertices :
  ¬ ∃ (f p a f_4 f_5 p_4 p_5 : ℕ), polyhedron_no_triangular_no_three_valent f p a f_4 f_5 p_4 p_5 :=
by
  sorry

end no_polyhedron_without_triangles_and_three_valent_vertices_l519_519382


namespace expression_value_at_neg3_l519_519082

theorem expression_value_at_neg3 (p q : ℤ) (h : 27 * p + 3 * q = 14) :
  (p * (-3)^3 + q * (-3) - 1) = -15 :=
sorry

end expression_value_at_neg3_l519_519082


namespace part_a_part_b_part_c_l519_519479

variables {a b c : ℝ} {A : ℝ} {r R h_a d_a : ℝ}

-- Conditions
def condition := a = (b + c) / 2

-- Part (a): 0° ≤ A ≤ 60°
theorem part_a (h : condition) : 0 ≤ A ∧ A ≤ 60 := sorry

-- Part (b): The altitude relative to side a is three times the inradius r
theorem part_b (h : condition) : h_a = 3 * r := sorry

-- Part (c): The distance from the circumcenter to side a is R - r
theorem part_c (h : condition) : d_a = R - r := sorry

end part_a_part_b_part_c_l519_519479


namespace game_must_terminate_when_n_less_than_1994_game_cannot_terminate_when_n_equals_1994_game_cannot_terminate_variant_when_n_leq_1991_l519_519486

def game_must_terminate (n : ℕ) : Prop :=
  ∀ g : list ℕ, g.length = 1994 → (∃ i < 1994, g[i] = n) →
  (∃ t : ℕ, ∃ g' : list ℕ, g'.length = 1994 ∧ (∀ i < 1994, g'[i] = 1))

def game_cannot_terminate (n : ℕ) : Prop :=
  ∃ g : list ℕ, g.length = 1994 → (∃ i < 1994, g[i] = n) →
  ∀ t : ℕ, ∃ g' : list ℕ, g'.length = 1994 ∧ (∃ i < 1994, g'[i] ≠ 1)

def game_cannot_terminate_variant (n : ℕ) : Prop :=
  ∃ g : list ℕ, g.length = 1991 → (∃ i < 1991, g[i] = n) →
  ∀ t : ℕ, ∃ g' : list ℕ, g'.length = 1991 ∧ (∃ i < 1991, g'[i] ≠ 1)

theorem game_must_terminate_when_n_less_than_1994 : ∀ (n : ℕ), n < 1994 → game_must_terminate n :=
by
  intros n hn
  sorry

theorem game_cannot_terminate_when_n_equals_1994 : ∀ (n : ℕ), n = 1994 → game_cannot_terminate n :=
by
  intros n hn
  sorry

theorem game_cannot_terminate_variant_when_n_leq_1991 : ∀ (n : ℕ), n ≤ 1991 → game_cannot_terminate_variant n :=
by
  intros n hn
  sorry

end game_must_terminate_when_n_less_than_1994_game_cannot_terminate_when_n_equals_1994_game_cannot_terminate_variant_when_n_leq_1991_l519_519486


namespace zero_in_interval_l519_519438

noncomputable def f (x : ℝ) : ℝ := 3 * x - 7 + Real.log x

theorem zero_in_interval : ∃ x ∈ set.Ioo (2 : ℝ) 3, f x = 0 :=
by {
  sorry
}

end zero_in_interval_l519_519438


namespace range_of_f_t1_in_0_to_4_range_of_a_f_t1_leq_5_range_of_t_abs_f_diff_leq_8_l519_519000

-- Define the function f(x)
def f (x : ℝ) (t : ℝ) : ℝ := x^2 - 2 * t * x + 2

-- The first problem
theorem range_of_f_t1_in_0_to_4 :
  (∀ (x : ℝ), (0 ≤ x ∧ x ≤ 4) → (f x 1 ≥ 1 ∧ f x 1 ≤ 10)) :=
sorry

-- The second problem
theorem range_of_a_f_t1_leq_5 :
  (∀ (a : ℝ), (∀ (x : ℝ), (a ≤ x ∧ x ≤ a+2) → f x 1 ≤ 5) → (-1 ≤ a ∧ a ≤ 1)) :=
sorry

-- The third problem
theorem range_of_t_abs_f_diff_leq_8 :
  (∀ (t : ℝ), (∀ (x1 x2 : ℝ), (0 ≤ x1 ∧ x1 ≤ 4 ∧ 0 ≤ x2 ∧ x2 ≤ 4) → |f x1 t - f x2 t| ≤ 8) → (4 - 2 * √2 ≤ t ∧ t ≤ 2 * √2)) :=
sorry

end range_of_f_t1_in_0_to_4_range_of_a_f_t1_leq_5_range_of_t_abs_f_diff_leq_8_l519_519000


namespace rhombus_longer_diagonal_l519_519141

theorem rhombus_longer_diagonal (a b : ℕ) (d1 : ℕ) (d2 : ℕ) (h0 : a = 65) (h1 : d1 = 56) (h2 : a * a = (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2)) :
  d2 = 118 :=
by
  have h3 : (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2) = 65 * 65, from h2,
  rw [←h0, ←h1] at h3,
  have h4 : (28 : ℕ) = d1 / 2, sorry,
  have h5 : (28 : ℕ) * (28 : ℕ) + ((d2 / 2) * (d2 / 2)) = 65 * 65, from h3,
  have h6 : (28 * 28) + (d2 / 2) * (d2 / 2) = (65*65), by rw h5,
  have h7 : (65 * 65) - 784 = (d2 / 2) * (d2 / 2), sorry,
  have h8 : (d2 / 2) * (d2 / 2) = 3441, by simp[h7],
  have h9 : (d2 / 2) = 59, from nat.sqrt_eq 3441,
  have h10 : d2 = 2 * 59, by rw [h9, two_mul],
  exact h10
 

end rhombus_longer_diagonal_l519_519141


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519739

theorem exists_inhabitant_with_at_least_810_acquaintances
  (N : ℕ) (hN : N = 1000000)
  (B : Finset ℕ) (hB : B.card = 900000)
  (A : ℕ → Finset ℕ)
  (hA_nonempty : ∀ i, (A i).nonempty)
  (hA_believers : ∀ i, (A i).card / 10 = (A i ∩ B).card) :
  ∃ j, (A j).card ≥ 810 :=
by
  sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519739


namespace chipmunk_increase_l519_519974

theorem chipmunk_increase :
  ∀ (initial_beavers initial_chipmunks final_animals : ℕ),
  initial_beavers = 20 →
  initial_chipmunks = 40 →
  final_animals = 130 →
  let final_beavers := 2 * initial_beavers in
  let final_chipmunks := final_animals - final_beavers in
  final_chipmunks - initial_chipmunks = 50 :=
by
  intros initial_beavers initial_chipmunks final_animals hb hc ht
  let final_beavers := 2 * initial_beavers
  let final_chipmunks := final_animals - final_beavers
  sorry

end chipmunk_increase_l519_519974


namespace line_construction_condition_l519_519602

    -- Assume a and b are parallel lines in a Euclidean plane
    variables [EuclideanPlane] (a b : Line[EuclideanPlane]) (P : Point[EuclideanPlane]) (d : ℝ)
    
    -- Define the distances from point P to the lines a and b, respectively
    def distance_to_line (P : Point[EuclideanPlane]) (l : Line[EuclideanPlane]) : ℝ := 
        classical.some (exists_unique_distance P l)
    
    noncomputable def a_0 := distance_to_line P a
    noncomputable def b_0 := distance_to_line P b

    -- Define the condition for the problem to be solvable
    theorem line_construction_condition :
        (∃ (ℓ : Line[EuclideanPlane]), (ℓ.contains P) ∧ 
            (∀ (A B : Point[EuclideanPlane]), A ∈ (a ∩ ℓ) → B ∈ (b ∩ ℓ) → |(P.dist A) - (P.dist B)| = d)) ↔
            (d ≥ |a_0 - b_0| ∨ (d = 0 ∧ True)) :=
    begin
        sorry,
    end
    
end line_construction_condition_l519_519602


namespace car_gas_tank_capacity_l519_519533

theorem car_gas_tank_capacity (distance_to_home : ℕ) (additional_distance : ℕ) (mileage_per_gallon : ℕ)
  (h1 : distance_to_home = 220) (h2 : additional_distance = 100) (h3 : mileage_per_gallon = 20) : 
  ∃ tank_capacity : ℕ, tank_capacity = 16 :=
by
  let total_distance := distance_to_home + additional_distance
  have h_total_distance : total_distance = 320 := by
    rw [h1, h2]
    exact Nat.add_zero 320
  let tank_capacity := total_distance / mileage_per_gallon
  have h_tank_capacity : tank_capacity = 16 := by
    rw [h3, Nat.div_eq_of_eq_mul_right]
    sorry
  use tank_capacity
  exact h_tank_capacity

end car_gas_tank_capacity_l519_519533


namespace range_y_sin_cos_sinxcos_l519_519288

theorem range_y_sin_cos_sinxcos (x : ℝ) (hx : 0 < x ∧ x ≤ Real.pi / 3) :
  let y := Real.sin x + Real.cos x + Real.sin x * Real.cos x in
  1 < y ∧ y ≤ 1 / 2 + Real.sqrt 2 :=
by
  let y := Real.sin x + Real.cos x + Real.sin x * Real.cos x
  sorry

end range_y_sin_cos_sinxcos_l519_519288


namespace area_of_figure_l519_519942

noncomputable def area_integral : ℝ :=
  ∫ (x : ℝ) in 0..3, x * Real.sqrt (9 - x^2)

theorem area_of_figure : area_integral = 9 := by
  sorry

end area_of_figure_l519_519942


namespace poly_perimeter_eq_33_l519_519792

noncomputable theory

def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (4, 8)
def C : ℝ × ℝ := (4, 4)
def D : ℝ × ℝ := (0, -2)
def E : ℝ × ℝ := (9, -2)

def AB : ℝ := distance A B
def BC : ℝ := distance B C
def CD : ℝ := distance C D
def DE : ℝ := distance D E
def EA : ℝ := distance E A

def perimeter : ℝ := AB + BC + CD + DE + EA

theorem poly_perimeter_eq_33 : perimeter = 33 := by
  sorry

end poly_perimeter_eq_33_l519_519792


namespace area_of_triangle_ABC_l519_519791

theorem area_of_triangle_ABC
  (A B C D : Type)
  [trapezoid : Trapezoid A B C D]
  (area_ABCD : area trapezoid = 30)
  (CD_three_times_AB : length (side C D) = 3 * length (side A B)) :
  area (△ A B C) = 7.5 := by
sorry

end area_of_triangle_ABC_l519_519791


namespace stacy_homework_assignment_l519_519871

theorem stacy_homework_assignment : 
  ∃ (T F M : ℕ), T = 6 ∧ F = T + 7 ∧ M = 2 * F ∧ (T + F + M) = 45 :=
by {
  existsi 6,
  existsi (6 + 7),
  existsi (2 * (6 + 7)),
  split,
  refl,
  split,
  refl,
  split,
  refl,
  sorry
}

end stacy_homework_assignment_l519_519871


namespace smallest_n_sides_of_polygon_l519_519152

/--
Given a regular polygon that fits perfectly when rotated by 40 degrees or 60 degrees,
prove that the smallest number of sides of the polygon is 18.
-/
theorem smallest_n_sides_of_polygon :
  ∃ n : ℕ, n > 2 ∧ 
    (360 % (n / 9) = 0) ∧
    (360 % (n / 6) = 0) ∧
    ∀ m : ℕ, m > 2 ∧ 
    (360 % (m / 9) = 0) ∧
    (360 % (m / 6) = 0) → n ≤ m := 
begin
  sorry
end

end smallest_n_sides_of_polygon_l519_519152


namespace triangle_incenter_condition_l519_519842

-- Define the basic geometric entities
variable (A B C I P : Type) [triangle : triangle A B C]

-- Define the angle relations and the distances in question
variable (angle_PBA angle_PCA angle_PBC angle_PCB : ℝ)
variable (angle_ABC angle_ACB angle_BAC : ℝ)
variable (AP AI : ℝ)

-- State the given condition and the theorem to prove
theorem triangle_incenter_condition (h1: ∀ P, angle_PBA + angle_PCA ≥ angle_PBC + angle_PCB)
    (h2: angle_ABC + angle_ACB + angle_BAC = 180)
    (h3: is_incenter I A B C) :
    (AP ≥ AI) ↔ (P = I) := by
  sorry

end triangle_incenter_condition_l519_519842


namespace housewife_saving_l519_519952

theorem housewife_saving :
  let total_money := 450
  let groceries_fraction := 3 / 5
  let household_items_fraction := 1 / 6
  let personal_care_items_fraction := 1 / 10
  let groceries_expense := groceries_fraction * total_money
  let household_items_expense := household_items_fraction * total_money
  let personal_care_items_expense := personal_care_items_fraction * total_money
  let total_expense := groceries_expense + household_items_expense + personal_care_items_expense
  total_money - total_expense = 60 :=
by
  sorry

end housewife_saving_l519_519952


namespace august_five_mondays_l519_519400

-- Definitions for the conditions
def july_has_five_mondays (N : ℕ) : Prop :=
  ∃ (m : ℕ), m < 7 ∧ m ≠ 1 ∧ -- starting possibilities for Mondays other than July 1st
    (∀ d, d < 31 → (d ≡ m [MOD 7] → monday d))

def month_has_31_days (month : ℕ) : Prop := 
  month = 31

-- The known condition that both July and August have 31 days
def july_and_august_have_31_days (N : ℕ) : Prop :=
  month_has_31_days 31 ∧ month_has_31_days 31

-- The main theorem we need to prove
theorem august_five_mondays (N : ℕ) 
  (H1 : july_has_five_mondays N)
  (H2 : july_and_august_have_31_days N) :
  ∃ d, d_day_count_in_august == 5 → d = thursday := 
begin
  sorry
end

end august_five_mondays_l519_519400


namespace sally_and_mary_picked_16_lemons_l519_519017

theorem sally_and_mary_picked_16_lemons (sally_lemons mary_lemons : ℕ) (sally_picked : sally_lemons = 7) (mary_picked : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 :=
by {
  sorry
}

end sally_and_mary_picked_16_lemons_l519_519017


namespace polynomial_evaluation_at_specific_value_l519_519579

noncomputable def polynomial_evaluation (x : ℝ) : ℝ :=
  (x^4 - 6 * x^3 - 2 * x^2 + 18 * x + 23) / (x^2 - 8 * x + 15)

theorem polynomial_evaluation_at_specific_value :
  (x : ℝ) (hx1 : x = Real.sqrt (19 - 8 * Real.sqrt 3)) (hx2 : x^2 - 8 * x + 15 = 0) :
  polynomial_evaluation x = 5 := 
by
  sorry

end polynomial_evaluation_at_specific_value_l519_519579


namespace problem_solution_l519_519988

theorem problem_solution :
  20 * ((180 / 3) + (40 / 5) + (16 / 32) + 2) = 1410 := by
  sorry

end problem_solution_l519_519988


namespace coin_flip_probability_l519_519874

theorem coin_flip_probability :
  let total_outcomes := 2^5
  let successful_outcomes := 2 * 2^2
  total_outcomes > 0 → (successful_outcomes / total_outcomes) = (1 / 4) :=
by
  intros
  sorry

end coin_flip_probability_l519_519874


namespace count_homogeneous_functions_y_eq_x_squared_l519_519291

def homogeneous_functions (f : ℝ → ℝ) (s : Set ℝ) : Set (Set ℝ) :=
  { d | ∃ g : ℝ → ℝ, g = f ∧ (∀ x ∈ d, g(x) ∈ s) ∧ (∀ y ∈ s, ∃ x ∈ d, g(x) = y) }

theorem count_homogeneous_functions_y_eq_x_squared :
  let f : ℝ → ℝ := λ x, x^2,
      s : Set ℝ := {1, 2} in
  homogeneous_functions f s = 9 :=
by {
  let f : ℝ → ℝ := λ x, x^2,
  let s : Set ℝ := {1, 2},
  sorry
}

end count_homogeneous_functions_y_eq_x_squared_l519_519291


namespace arithmetic_sequence_length_l519_519272

theorem arithmetic_sequence_length :
  ∀ (a d l : ℤ), a = -18 → d = 6 → l = 48 → ∃ n : ℕ, n = (l - a) / d + 1 ∧ n = 12 :=
by
  intros a d l ha hd hl
  use (l - a) / d + 1
  simp [ha, hd, hl]
  norm_num
  sorry

end arithmetic_sequence_length_l519_519272


namespace acquaintance_paradox_proof_l519_519711

theorem acquaintance_paradox_proof :
  ∃ (x : ℕ), (acquaintances x ≥ 810) :=
by
  let n_population := 1000000
  let n_believers := 900000
  let n_nonBelievers := n_population - n_believers
  have h1 : n_population = 1000000 := rfl
  have h2 : n_believers = 0.90 * n_population := by sorry
  have h3 : n_nonBelievers = n_population - n_believers := by sorry
  have h4 : ∀ x, acquaintances x ≥ 1 := by sorry
  have h5 : ∀ x, 0.10 * acquaintances x = nonBelievers_acquaintances x := by sorry
  have h6 : nonBelievers_acquaintances_total >= 90 := by sorry
  
  show ∃ (x : ℕ), acquaintances x ≥ 810 from sorry
  
  sorry

end acquaintance_paradox_proof_l519_519711


namespace find_final_painting_width_l519_519467

theorem find_final_painting_width
  (total_area : ℕ)
  (painting_areas : List ℕ)
  (total_paintings : ℕ)
  (last_painting_height : ℕ)
  (last_painting_width : ℕ) :
  total_area = 200
  → painting_areas = [25, 25, 25, 80]
  → total_paintings = 5
  → last_painting_height = 5
  → last_painting_width = 9 :=
by
  intros h_total_area h_painting_areas h_total_paintings h_last_height
  have h1 : 25 * 3 + 80 = 155 := by norm_num
  have h2 : total_area - 155 = last_painting_width * last_painting_height := by
    rw [h_total_area, show 155 = 25 * 3 + 80 by norm_num]
    norm_num
  exact eq_of_mul_eq_mul_right (by norm_num) h2

#print axioms find_final_painting_width -- this should ensure we don't leave any implicit assumptions. 

end find_final_painting_width_l519_519467


namespace multiple_of_other_number_l519_519858

theorem multiple_of_other_number 
(m S L : ℕ) 
(hl : L = 33) 
(hrel : L = m * S - 3) 
(hsum : L + S = 51) : 
m = 2 :=
by
  sorry

end multiple_of_other_number_l519_519858


namespace magnitude_sum_l519_519244

theorem magnitude_sum (a b : EuclideanSpace ℝ (Fin 3)) (θ : ℝ) (hθ : θ = real.pi / 3)
  (ha : ‖a‖ = 2) (hb : ‖b‖ = 1) : ‖a + 2 • b‖ = 2 * real.sqrt 3 :=
by
  /- Proof goes here -/
  sorry

end magnitude_sum_l519_519244


namespace students_bought_pencils_l519_519432

theorem students_bought_pencils (h1 : 2 * 2 + 6 * 3 + 2 * 1 = 24) : 
  2 + 6 + 2 = 10 := by
  sorry

end students_bought_pencils_l519_519432


namespace f_even_and_periodic_l519_519626

noncomputable def f (x : ℝ) : ℝ := (1 + cos (2 * x)) * (sin x) ^ 2

-- Define the evenness of f.
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the periodicity of f.
def has_period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem f_even_and_periodic :
  is_even f ∧ has_period f (π / 2) :=
sorry

end f_even_and_periodic_l519_519626


namespace acquaintance_paradox_proof_l519_519714

theorem acquaintance_paradox_proof :
  ∃ (x : ℕ), (acquaintances x ≥ 810) :=
by
  let n_population := 1000000
  let n_believers := 900000
  let n_nonBelievers := n_population - n_believers
  have h1 : n_population = 1000000 := rfl
  have h2 : n_believers = 0.90 * n_population := by sorry
  have h3 : n_nonBelievers = n_population - n_believers := by sorry
  have h4 : ∀ x, acquaintances x ≥ 1 := by sorry
  have h5 : ∀ x, 0.10 * acquaintances x = nonBelievers_acquaintances x := by sorry
  have h6 : nonBelievers_acquaintances_total >= 90 := by sorry
  
  show ∃ (x : ℕ), acquaintances x ≥ 810 from sorry
  
  sorry

end acquaintance_paradox_proof_l519_519714


namespace geometric_sequence_a5_l519_519594

open_locale big_operators

noncomputable theory

variable (a : ℕ → ℝ)
variable (q : ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∃ (a1 q : ℝ),
  a 1 + a 1 * q = 4 ∧
  a 1 * q + a 1 * q ^ 2 = 12

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) (h : geometric_sequence a q) : a 5 = 81 :=
sorry

end geometric_sequence_a5_l519_519594


namespace inverse_proportion_relation_l519_519420

variable (k : ℝ) (y1 y2 : ℝ) (h1 : y1 = - (2 / (-1))) (h2 : y2 = - (2 / (-2)))

theorem inverse_proportion_relation : y1 > y2 := by
  sorry

end inverse_proportion_relation_l519_519420


namespace range_of_a_product_of_zeros_l519_519661

def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - (Real.log x) + x - a

theorem range_of_a (a : ℝ) : (∀ x, (0 < x) → (f x a ≥ 0)) ↔ (a ≤ Real.exp 1 + 1) :=
by
  sorry

theorem product_of_zeros (x1 x2 a : ℝ) : 
  (0 < x1) → (x1 < 1) → (1 < x2) → (f x1 a = 0 ∧ f x2 a = 0) → x1 * x2 < 1 :=
by
  sorry

end range_of_a_product_of_zeros_l519_519661


namespace max_taxi_ride_distance_l519_519413

open Real

def fixed_fare : ℝ := 2.5
def cost_per_100_meters : ℝ := 0.1
def total_money : ℝ := 10.0
def max_distance (fixed_fare cost_per_100_meters total_money : ℝ) : ℝ :=
  (total_money - fixed_fare) / cost_per_100_meters * 0.1

theorem max_taxi_ride_distance :
  max_distance fixed_fare cost_per_100_meters total_money = 7.5 :=
by
  sorry

end max_taxi_ride_distance_l519_519413


namespace subtract_two_decimals_l519_519170

theorem subtract_two_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_two_decimals_l519_519170


namespace triangle_line_equations_l519_519267

structure Point where
  x : ℝ
  y : ℝ

def line_eq (A B : Point) (a b c : ℝ) : Prop :=
  ∀ (P : Point), P = A ∨ P = B → a * P.x + b * P.y + c = 0

def A : Point := ⟨-5, 0⟩
def B : Point := ⟨3, -3⟩
def C : Point := ⟨0, 2⟩

theorem triangle_line_equations :
  line_eq A B 3 8 15 ∧
  line_eq A C 2 -5 10 ∧
  line_eq B C 5 3 -6 :=
by
  sorry

end triangle_line_equations_l519_519267


namespace max_disjoint_groups_l519_519077

/-- 
The maximum number of disjoint groups into which all the integers from 1 to 25 
can be divided such that the sum of the numbers in each group is a perfect square 
is 14.
-/
theorem max_disjoint_groups (f : Fin 25 → Fin 15) (h_disjoint : ∀ i j, i ≠ j → f i ≠ f j) 
    (h_sum_perfect_square : ∀ i, ∃ n, n^2 = ∑ k in Finset.filter (λ x, f x = i) (Finset.range 25), x + 1) :
    ∃ g : Fin 25 → Fin 15, ∀ (i : Fin 15) (j : Fin 25), g j = i → disjoint (Finset.filter (λ x, g x = i) (Finset.range 25)) (Finset.filter (λ x, g x = j) (Finset.range 25)) ∧ 
    ∃ n, n^2 = ∑ k in Finset.filter (λ x, g x = i) (Finset.range 25), x + 1 :=
by
  sorry

end max_disjoint_groups_l519_519077


namespace domino_end_points_l519_519521

def domino_chain_endpoints (chain : List (ℕ × ℕ)) : Prop :=
  ∃ a b, chain.head = some (5, a) ∧ chain.last = some (b, 5)

theorem domino_end_points (chain : List (ℕ × ℕ)) (standard_set : Finset (ℕ × ℕ)) :
  (∀ x ∈ standard_set, (x.1 = 5 ∨ x.2 = 5) → x ∈ chain) → 
  (∃ p in chain, p.1 = 5 ∨ p.2 = 5) →
  domino_chain_endpoints chain :=
sorry

end domino_end_points_l519_519521


namespace find_possible_values_of_y_l519_519835

noncomputable def solve_y (x : ℝ) : ℝ :=
  ((x - 3) ^ 2 * (x + 4)) / (2 * x - 4)

theorem find_possible_values_of_y (x : ℝ) 
  (h : x ^ 2 + 9 * (x / (x - 3)) ^ 2 = 90) : 
  solve_y x = 0 ∨ solve_y x = 105.23 := 
sorry

end find_possible_values_of_y_l519_519835


namespace jake_notes_total_l519_519330

-- Definitions of the conditions
def red_notes_rows : Nat := 5
def red_notes_per_row : Nat := 6
def red_scattered_notes : Nat := 3

def blue_notes_rows : Nat := 4
def blue_notes_per_row : Nat := 7
def blue_scattered_notes : Nat := 12

def green_triangle_bases : List Nat := [4, 5, 6]

def yellow_diagonal_1_notes : Nat := 5
def yellow_diagonal_2_notes : Nat := 3
def yellow_hexagon_notes : Nat := 6

-- Derived values from the conditions
def red_notes_total : Nat := red_notes_rows * red_notes_per_row + red_scattered_notes
def blue_notes_total : Nat := blue_notes_rows * blue_notes_per_row + blue_scattered_notes
def green_notes_total : Nat := (green_triangle_bases.map fun n => (n * (n + 1)) / 2).sum
def yellow_notes_total : Nat := yellow_diagonal_1_notes + yellow_diagonal_2_notes + yellow_hexagon_notes

-- Sum of all notes
def total_notes : Nat := red_notes_total + blue_notes_total + green_notes_total + yellow_notes_total

-- The theorem to prove
theorem jake_notes_total (red_notes_total = 33) (blue_notes_total = 40) (green_notes_total = 46) (yellow_notes_total = 14) : total_notes = 133 :=
by
  unfold total_notes
  unfold red_notes_total
  unfold blue_notes_total
  unfold green_notes_total
  unfold yellow_notes_total
  sorry

end jake_notes_total_l519_519330


namespace quotient_remainder_base5_l519_519567

theorem quotient_remainder_base5 :
    let n₅ := 1 * 5^4 + 3 * 5^3 + 2 * 5^2 + 0 * 5^1 + 4 * 5^0,
        d₅ := 2 * 5^1 + 3 * 5^0,
        q₁₀ := n₅ / d₅,
        r₁₀ := n₅ % d₅,
        q₅ := 1 * 5^2 + 1 * 5^1 + 1 * 5^0,
        r₅ := 1 * 5^0
    in n₅ = 1054 ∧ d₅ = 13 ∧ q₁₀ = 81 ∧ r₁₀ = 1 ∧ q₅ = 311 ∧ r₅ = 1 :=
by
  let n₅ := 1 * 5^4 + 3 * 5^3 + 2 * 5^2 + 0 * 5^1 + 4 * 5^0
  let d₅ := 2 * 5^1 + 3 * 5^0
  let q₁₀ := n₅ / d₅
  let r₁₀ := n₅ % d₅
  let q₅ := 1 * 5^2 + 1 * 5^1 + 1 * 5^0
  let r₅ := 1 * 5^0
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

end quotient_remainder_base5_l519_519567


namespace original_board_is_120_l519_519492

-- Define the two given conditions
def S : ℕ := 35
def L : ℕ := 2 * S + 15

-- Define the length of the original board
def original_board_length : ℕ := S + L

-- The theorem we want to prove
theorem original_board_is_120 : original_board_length = 120 :=
by
  -- Skipping the actual proof
  sorry

end original_board_is_120_l519_519492


namespace solve_for_x_l519_519487

theorem solve_for_x :
  ∃ x : ℕ, 40 * x + (12 + 8) * 3 / 5 = 1212 ∧ x = 30 :=
by
  use 30
  split
  easy
  rfl

end solve_for_x_l519_519487


namespace tens_digit_seven_last_digit_six_l519_519287

theorem tens_digit_seven_last_digit_six (n : ℕ) (h : ((n * n) / 10) % 10 = 7) :
  (n * n) % 10 = 6 :=
sorry

end tens_digit_seven_last_digit_six_l519_519287


namespace exists_inhabitant_with_810_acquaintances_l519_519749

-- Conditions
def total_population : ℕ := 1000000
def believers_percentage : ℕ := 90
def acquaintances_believers_percentage : ℝ := 0.10

-- Theorem statement: Prove that there exists an inhabitant knowing at least 810 people.
theorem exists_inhabitant_with_810_acquaintances 
    (H1 : ∀ (p : ℕ), p < total_population → ∃ q, q ≠ p)  -- Every person knows at least one other person
    (H2 : 0.9 * total_population = 900000)  -- 90% of the population are believers in Santa Claus
    (H3 : ∀ p, p < total_population → ∀ q, q != p → (belief p → (exists percentage, percentage = acquaintances_believers_percentage))) -- Every person claims exactly 10% of their acquaintances believe in Santa Claus
    : ∃ (habitants : ℕ), habitants ≥ 810
    :=
  sorry

end exists_inhabitant_with_810_acquaintances_l519_519749


namespace num_prime_factors_462_l519_519174

theorem num_prime_factors_462 : ∀ (p : ℕ → Prop) [prime p], ∏ pf in {2, 3, 7, 11}, p pf = 462 → (∃ s, s.card = 4 ∧ (∀ x ∈ s, prime x) ∧ ∏ x in s, x = 462) :=
by sorry

end num_prime_factors_462_l519_519174


namespace exists_inhabitant_with_many_acquaintances_l519_519759

open Function

theorem exists_inhabitant_with_many_acquaintances :
  (∃ (n : ℕ), n = 1_000_000) →
  (∀ (p : ℕ), p < 1_000_000 → ∃ (q : ℕ), q ≠ p) →
  (∀ (p : ℕ), p < 900_000 → p <= 900_000) →
  (∀ (p : ℕ), p < 1_000_000 → 0.1 * real.to_real p < 0.9 * real.to_real p) →
  ∃ (p : ℕ), p >= 810 :=
by
  intros h1 h2 h3 h4
  sorry

end exists_inhabitant_with_many_acquaintances_l519_519759


namespace part_a_l519_519100

variable (a0 a1 a2 : ℝ)
def P (x : ℝ) : ℝ := a0 + a1 * x + a2 * x^2

theorem part_a (h_neg1 : P a0 a1 a2 (-1) ∈ ℤ) (h_0 : P a0 a1 a2 0 ∈ ℤ) (h_1 : P a0 a1 a2 1 ∈ ℤ) :
  ∀ n : ℤ, P a0 a1 a2 n ∈ ℤ := 
sorry

end part_a_l519_519100


namespace range_of_a_l519_519418

-- Define the even function f
variable {f : ℝ → ℝ}

-- Conditions as given in the problem
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

-- The main theorem translates the conditions and question to a Lean proof problem
theorem range_of_a (h_even : is_even_function f) (h_incr : is_increasing_on_pos f) (a : ℝ) :
  f a ≥ f 2 → a ∈ set.Iic (-2) ∪ set.Ici 2 :=
  sorry

end range_of_a_l519_519418


namespace IJ_eq_AH_l519_519813

/-- Given an acute triangle ABC with orthocenter H, point G such that ABGH is a parallelogram,
and point I on line GH such that AC bisects HI.
Given that line AC intersects the circumcircle of triangle GCI at points C and J.
Prove that IJ = AH. -/
theorem IJ_eq_AH (A B C H G I J: Point) 
(acute_triangle : acute_triangle A B C)
(is_orthocenter : is_orthocenter A B C H)
(parallelogram_ABGH : parallelogram A B G H)
(I_on_GH : collinear (line_through G H) I)
(AC_bisects_HI : midpoint (line_through A C) I (line_segment H I))
(intersects_circumcircle_GCI : ∃k, k ∈ circumcircle G C I ∧ lies_on (line_through A C) k ∧ lies_on (line_through A C) J) :
distance I J = distance A H := 
sorry

end IJ_eq_AH_l519_519813


namespace pair_B_like_terms_l519_519522

-- Definition of like terms
def like_terms (e1 e2 : String) : Prop :=
  -- Simple check to ensure terms have same variables with same exponents
  ∃ coeff1 coeff2 exp1 exp2, 
    e1 = coeff1 * ("m" ^ exp1) * ("n" ^ exp2) ∧
    e2 = coeff2 * ("m" ^ exp1) * ("n" ^ exp2)

theorem pair_B_like_terms : like_terms (toString ((1 / 2) * m^3 * n)) (toString (-8 * n * m^3)) := 
  sorry -- Proof goes here

end pair_B_like_terms_l519_519522


namespace area_of_square_formed_by_roots_l519_519540

noncomputable def polynomial := (λ z : ℂ, z^4 + 4*z^3 + (6 - 6*complex.I)*z^2 + (4 - 8*complex.I)*z + (1 - 4*complex.I))

theorem area_of_square_formed_by_roots:
  let roots := {a, b, c, d | is_root polynomial z} in
  (∀ a b c d ∈ roots, ∃ center : ℂ, (a - center).abs = (b - center).abs ∧ (b - center).abs = (c - center).abs ∧ (c - center).abs = (d - center).abs ∧ a + b + c + d = -4) →
  ∃ p : ℝ, (∃ side_length : ℝ, side_length = real.sqrt 2 ∧ p = side_length * side_length) ∧ p = 2 :=
sorry

end area_of_square_formed_by_roots_l519_519540


namespace at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519723

theorem at_least_one_inhabitant_knows_no_fewer_than_810_people :
  ∀ (n : ℕ),
  (inhabitants : ℕ) (knows_someone : inhabitants > 1) (believes_santa : inhabitants * 9 / 10)
  (ten_percent_acquaintances_believe : ∀ x, (acquaintances : list x) → (|list.contacts_with_santa <- 0.10 * acquaintances|)),
  (h : inhabitants = 1000000),
  (one_inhabitant_knows : ∃ x, ∀ y, y \in acquaintances x → friends y ≥ 810) :=
sorry

end at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519723


namespace savings_by_having_insurance_l519_519060

theorem savings_by_having_insurance :
  let insurance_cost := 24 * 20 in
  let procedure_cost := 5000 in
  let insurance_coverage := 0.2 in
  let amount_paid_with_insurance := procedure_cost * insurance_coverage in
  let total_paid_with_insurance := amount_paid_with_insurance + insurance_cost in
  let savings := procedure_cost - total_paid_with_insurance in
  savings = 3520 := 
by {
  let insurance_cost := 24 * 20 in
  let procedure_cost := 5000 in
  let insurance_coverage := 0.2 in
  let amount_paid_with_insurance := procedure_cost * insurance_coverage in
  let total_paid_with_insurance := amount_paid_with_insurance + insurance_cost in
  let savings := procedure_cost - total_paid_with_insurance in
  guard_hyp savings,
  sorry
}

end savings_by_having_insurance_l519_519060


namespace vector_x_value_l519_519674

theorem vector_x_value (x : ℝ) :
  let a := (x, 2)
  let b := (-1, 1)
  (‖(a.1 - b.1, a.2 - b.2)‖ = ‖(a.1 + b.1, a.2 + b.2)‖) → x = 2 :=
begin
  intro h,
  sorry
end

end vector_x_value_l519_519674


namespace puzzle_pieces_l519_519800

theorem puzzle_pieces
  (total_puzzles : ℕ)
  (pieces_per_10_min : ℕ)
  (total_minutes : ℕ)
  (h1 : total_puzzles = 2)
  (h2 : pieces_per_10_min = 100)
  (h3 : total_minutes = 400) :
  ((total_minutes / 10) * pieces_per_10_min) / total_puzzles = 2000 :=
by
  sorry

end puzzle_pieces_l519_519800


namespace tip_percentage_proof_l519_519444

def tip_percentage_of_lunch_cost (L T : ℝ) : ℝ := ((T - L) / L) * 100

theorem tip_percentage_proof :
  tip_percentage_of_lunch_cost 50.50 60.60 = 20 :=
by 
  sorry

end tip_percentage_proof_l519_519444


namespace three_point_seven_five_minus_one_point_four_six_l519_519171

theorem three_point_seven_five_minus_one_point_four_six : 3.75 - 1.46 = 2.29 :=
by sorry

end three_point_seven_five_minus_one_point_four_six_l519_519171


namespace mean_temperature_eq_l519_519409

theorem mean_temperature_eq :
  let temps := [-8, -5, -3, 0, 4, 2, 7] in
  (temps.sum : ℚ) / temps.length = -3 / 7 :=
by
  sorry

end mean_temperature_eq_l519_519409


namespace carla_bought_marbles_l519_519532

def starting_marbles : ℕ := 2289
def total_marbles : ℝ := 2778.0

theorem carla_bought_marbles : (total_marbles - starting_marbles) = 489 := 
by
  sorry

end carla_bought_marbles_l519_519532


namespace quadratic_root_unique_l519_519429

theorem quadratic_root_unique 
  (a b c : ℝ)
  (hf1 : b^2 - 4 * a * c = 0)
  (hf2 : (b - 30 * a)^2 - 4 * a * (17 * a - 7 * b + c) = 0)
  (ha_pos : a ≠ 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = -11 := 
by
  sorry

end quadratic_root_unique_l519_519429


namespace number_decreased_by_13_l519_519489

theorem number_decreased_by_13 (x : ℕ) (hx : 100 ≤ x.digits 10 ∧ x.digits 10 < 10) :
  ∃ (b : ℕ), b ∈ {1, 2, 3} ∧ (x = 1625 * 10^96 ∨ x = 195 * 10^97 ∨ x = 2925 * 10^96 ∨ x = 13 * b * 10^98) :=
sorry

end number_decreased_by_13_l519_519489


namespace points_collinear_l519_519603

def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
-- Define the points Pn with given condition and sequences
def P (n : ℕ) (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) : ℝ × ℝ := (a_n n, b_n n)

-- The arithmetic and geometric sequences
noncomputable def a_n : ℕ → ℝ := sorry -- Define the arithmetic sequence here
noncomputable def b_n : ℕ → ℝ := sorry -- Define the geometric sequence here

-- The initial condition on A and B
def cond_AP1_P1B : Prop := ∀ P1 : ℝ × ℝ, 
  let AP1 : ℝ × ℝ := (P1.1 - A.1, P1.2 - A.2)
      P1B : ℝ × ℝ := (P1.1 - B.1, P1.2 - B.2)
  in AP1 = (2 * P1B.1, 2 * P1B.2)

-- Coordinates of P1
def P1_coordinates : ℝ × ℝ :=
  let P1 : ℝ × ℝ := (1 / 3, 2 / 3)
  in P1

-- Proof of collinearity of points P1, P2, ..., Pn
def collinear_points (a_n b_n (n : ℕ) : ℝ) : Prop :=
  let P : ℕ → ℝ × ℝ := λ n, (a_n n, b_n n)
  ∀ m1 m2 m3 : ℕ, m1 ≠ m2 ∧ m2 ≠ m3 ∧ m1 ≠ m3 →
  (P m1.2 * (P m2.1 / P m1.1) = P m2.2) ∧ (P m2.2 * (P m3.1 / P m2.1) = P m3.2)

theorem points_collinear :
  cond_AP1_P1B → P1_coordinates = (1 / 3, 2 / 3) → collinear_points a_n b_n :=
by 
  sorry

end points_collinear_l519_519603


namespace partial_fraction_sum_zero_l519_519544

theorem partial_fraction_sum_zero
    (A B C D E : ℝ)
    (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 = A * (x + 1) * (x + 2) * (x + 3) * (x + 5) +
        B * x * (x + 2) * (x + 3) * (x + 5) +
        C * x * (x + 1) * (x + 3) * (x + 5) +
        D * x * (x + 1) * (x + 2) * (x + 5) +
        E * x * (x + 1) * (x + 2) * (x + 3)) :
    A + B + C + D + E = 0 := by
    sorry

end partial_fraction_sum_zero_l519_519544


namespace length_of_first_train_l519_519154

-- Define the conditions as constants
def speed_first_train_kmph : ℝ := 120
def speed_second_train_kmph : ℝ := 80
def crossing_time_seconds : ℝ := 9
def length_second_train_meters : ℝ := 230

-- Conversion factor from kmph to m/s
def kmph_to_mps : ℝ := 5 / 18

-- Define the relative speed in m/s
def relative_speed_mps : ℝ := 
  (speed_first_train_kmph + speed_second_train_kmph) * kmph_to_mps

-- Lean statement to prove the length of the first train
theorem length_of_first_train : 
  relative_speed_mps * crossing_time_seconds - length_second_train_meters = 270.04 := 
by 
  -- skip the proof
  sorry

end length_of_first_train_l519_519154


namespace gcd_1080_920_is_40_l519_519939

theorem gcd_1080_920_is_40 : Nat.gcd 1080 920 = 40 :=
by
  sorry

end gcd_1080_920_is_40_l519_519939


namespace arithmetic_geometric_mean_inequality_l519_519589

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (A : ℝ) (G : ℝ)
  (hA : A = (a + b) / 2) (hG : G = Real.sqrt (a * b)) : A ≥ G :=
by
  sorry

end arithmetic_geometric_mean_inequality_l519_519589


namespace quadrilateral_area_l519_519891

-- Provide the definitions of the points as vectors
def P : ℝ × ℝ × ℝ := (2, -1, 3)
def Q : ℝ × ℝ × ℝ := (4, -5, 6)
def R : ℝ × ℝ × ℝ := (3, 0, 1)
def S : ℝ × ℝ × ℝ := (5, -4, 4)

-- Define the vectors corresponding to the parallelogram
def qp := (Q.1 - P.1, Q.2 - P.2, Q.3 - P.3) -- Q - P
def rp := (R.1 - P.1, R.2 - P.2, R.3 - P.3) -- R - P

-- Define the cross product of qp and rp
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

def area_vector := cross_product qp rp

-- Calculate the magnitude of the area vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Statement of the proof problem
theorem quadrilateral_area :
  magnitude area_vector = real.sqrt 110 := by
  sorry

end quadrilateral_area_l519_519891


namespace geometric_meaning_of_derivative_l519_519039

open Real

namespace DerivativeExample

def is_tangent_slope (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  ∀ (y : ℝ → ℝ), deriv f x0 = deriv y x0 → (∃ m : ℝ, y = λ x, f x0 + m * (x - x0))

theorem geometric_meaning_of_derivative (f : ℝ → ℝ) (x0 : ℝ) :
  differentiable_at ℝ f x0 →
  f' x0 = deriv f x0 →
  is_tangent_slope f x0 :=
by
  sorry

end DerivativeExample

end geometric_meaning_of_derivative_l519_519039


namespace roots_triangle_ineq_l519_519254

variable {m : ℝ}

def roots_form_triangle (x1 x2 x3 : ℝ) : Prop :=
  x1 + x2 > x3 ∧ x1 + x3 > x2 ∧ x2 + x3 > x1

theorem roots_triangle_ineq (h : ∀ x, (x - 2) * (x^2 - 4*x + m) = 0) :
  3 < m ∧ m < 4 :=
by
  sorry

end roots_triangle_ineq_l519_519254


namespace rhombus_longer_diagonal_length_l519_519136

theorem rhombus_longer_diagonal_length (side_length : ℕ) (shorter_diagonal : ℕ) 
  (h_side : side_length = 65) (h_short_diag : shorter_diagonal = 56) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 118 :=
by
  sorry

end rhombus_longer_diagonal_length_l519_519136


namespace mail_total_correct_l519_519342

def Monday_mail : ℕ := 65
def Tuesday_mail : ℕ := Monday_mail + 10
def Wednesday_mail : ℕ := Tuesday_mail - 5
def Thursday_mail : ℕ := Wednesday_mail + 15
def total_mail : ℕ := Monday_mail + Tuesday_mail + Wednesday_mail + Thursday_mail

theorem mail_total_correct : total_mail = 295 := by
  sorry

end mail_total_correct_l519_519342


namespace complex_number_property_l519_519253

-- Step d): Lean 4 Statement
theorem complex_number_property (z : ℂ) (h : z = -1/2 + (Real.sqrt 3)/2 * Complex.I) : z^2 = Complex.conj z := by
  sorry

end complex_number_property_l519_519253


namespace fraction_product_l519_519981

theorem fraction_product :
  (∏ n in (finset.range 49).map (λ n, (n+1)/(n+5))) = (1:ℚ) / 23426 := 
by sorry

end fraction_product_l519_519981


namespace sum_of_digits_is_21_l519_519816

theorem sum_of_digits_is_21 :
  ∃ (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧ 
  ((10 * a + b) * (10 * c + b) = 111 * d) ∧ 
  (d = 9) ∧ 
  (a + b + c + d = 21) := by
  sorry

end sum_of_digits_is_21_l519_519816


namespace johns_age_l519_519808

-- Define the variables and conditions
def age_problem (j d : ℕ) : Prop :=
j = d - 34 ∧ j + d = 84

-- State the theorem to prove that John's age is 25
theorem johns_age : ∃ (j d : ℕ), age_problem j d ∧ j = 25 :=
by {
  sorry
}

end johns_age_l519_519808


namespace find_number_l519_519292

theorem find_number (x : ℤ) (h : x - (28 - (37 - (15 - 16))) = 55) : x = 65 :=
sorry

end find_number_l519_519292


namespace seq_a_2012_value_l519_519263

theorem seq_a_2012_value :
  ∀ (a : ℕ → ℕ),
  (a 1 = 0) →
  (∀ n : ℕ, a (n + 1) = a n + 2 * n) →
  a 2012 = 2011 * 2012 :=
by
  intros a h₁ h₂
  sorry

end seq_a_2012_value_l519_519263


namespace solve_equation_l519_519054

theorem solve_equation : ∀ x : ℝ, (2 * x - 8 = 0) ↔ (x = 4) :=
by sorry

end solve_equation_l519_519054


namespace ethanol_concentration_l519_519440

theorem ethanol_concentration
  (w1 : ℕ) (c1 : ℝ) (w2 : ℕ) (c2 : ℝ)
  (hw1 : w1 = 400) (hc1 : c1 = 0.30)
  (hw2 : w2 = 600) (hc2 : c2 = 0.80) :
  (c1 * w1 + c2 * w2) / (w1 + w2) = 0.60 := 
by
  sorry

end ethanol_concentration_l519_519440


namespace chocolate_chips_per_member_l519_519335

/-
Define the problem conditions:
-/
def family_members := 4
def batches_choc_chip := 3
def cookies_per_batch_choc_chip := 12
def chips_per_cookie_choc_chip := 2
def batches_double_choc_chip := 2
def cookies_per_batch_double_choc_chip := 10
def chips_per_cookie_double_choc_chip := 4

/-
State the theorem to be proved:
-/
theorem chocolate_chips_per_member : 
  let total_choc_chip_cookies := batches_choc_chip * cookies_per_batch_choc_chip
  let total_choc_chips_choc_chip := total_choc_chip_cookies * chips_per_cookie_choc_chip
  let total_double_choc_chip_cookies := batches_double_choc_chip * cookies_per_batch_double_choc_chip
  let total_choc_chips_double_choc_chip := total_double_choc_chip_cookies * chips_per_cookie_double_choc_chip
  let total_choc_chips := total_choc_chips_choc_chip + total_choc_chips_double_choc_chip
  let chips_per_member := total_choc_chips / family_members
  chips_per_member = 38 :=
by
  sorry

end chocolate_chips_per_member_l519_519335


namespace smallest_positive_period_and_increasing_on_interval_l519_519543

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6))

theorem smallest_positive_period_and_increasing_on_interval :
  (∀ T > 0, ∃ n ∈ ℕ, f (x + T * n) = f x ∧ n ≥ 1 → T = Real.pi) ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 12 → f' x > 0) :=
by sorry

end smallest_positive_period_and_increasing_on_interval_l519_519543


namespace length_BF1_l519_519236

theorem length_BF1 :
  let F1 := (-1, 0)
  let F2 := (1, 0)
  let A := (0, -1)
  (point B, on_line, on_ellipse) \/
  B.2 = (B.1 - 1) /\ (B.1 ^ 2 / 2 + B.2 ^ 2 = 1)
  -> 
  let BF1 := (BF1, dist formula)
  |BF1| = (5 * sqrt 2 / 3) := 
  sorry

end length_BF1_l519_519236


namespace mail_total_correct_l519_519343

def Monday_mail : ℕ := 65
def Tuesday_mail : ℕ := Monday_mail + 10
def Wednesday_mail : ℕ := Tuesday_mail - 5
def Thursday_mail : ℕ := Wednesday_mail + 15
def total_mail : ℕ := Monday_mail + Tuesday_mail + Wednesday_mail + Thursday_mail

theorem mail_total_correct : total_mail = 295 := by
  sorry

end mail_total_correct_l519_519343


namespace problem_inequality_l519_519357

theorem problem_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
sorry

end problem_inequality_l519_519357


namespace candle_lighting_time_l519_519902

theorem candle_lighting_time :
  ∃ (t : ℝ), t = 4 - (12 / 5) ∧ (4 - t) = 1.6 :=
by
  exist t, sorry

end candle_lighting_time_l519_519902


namespace total_mail_l519_519336

def monday_mail := 65
def tuesday_mail := monday_mail + 10
def wednesday_mail := tuesday_mail - 5
def thursday_mail := wednesday_mail + 15

theorem total_mail : 
  monday_mail + tuesday_mail + wednesday_mail + thursday_mail = 295 := by
  sorry

end total_mail_l519_519336


namespace zinc_weight_l519_519935

/-- Given that zinc and copper are melted together in the ratio 9:11 and the total weight of the melted mixture is 
78 kg, prove that the weight of zinc consumed in the mixture is 35.1 kg. -/
theorem zinc_weight (total_weight : ℝ) (zinc_ratio : ℝ) (copper_ratio : ℝ) (total_ratio : ℝ) (weight_one_part : ℝ) (weight_zinc : ℝ) : 
  total_weight = 78 ∧ zinc_ratio = 9 ∧ copper_ratio = 11 ∧  total_ratio = zinc_ratio + copper_ratio ∧ 
  weight_one_part = total_weight / total_ratio ∧  weight_zinc = weight_one_part * zinc_ratio  → weight_zinc = 35.1 :=
by
  intro h
  cases h with ht hrest
  cases hrest with hz hrest
  cases hrest with hc hrest
  cases hrest with ht_total_ratio hrest
  cases hrest with hw_one hw_zinc
  sorry

end zinc_weight_l519_519935


namespace lamp_cost_l519_519552

def saved : ℕ := 500
def couch : ℕ := 750
def table : ℕ := 100
def remaining_owed : ℕ := 400

def total_cost_without_lamp : ℕ := couch + table

theorem lamp_cost :
  total_cost_without_lamp - saved + lamp = remaining_owed → lamp = 50 := by
  sorry

end lamp_cost_l519_519552


namespace consecutive_numbers_sum_digits_l519_519149

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem consecutive_numbers_sum_digits :
  ∃ n : ℕ, sum_of_digits n = 52 ∧ sum_of_digits (n + 4) = 20 := 
sorry

end consecutive_numbers_sum_digits_l519_519149


namespace number_of_tiles_l519_519048

theorem number_of_tiles (room_width room_height tile_width tile_height : ℝ) :
  room_width = 8 ∧ room_height = 12 ∧ tile_width = 1.5 ∧ tile_height = 2 →
  (room_width * room_height) / (tile_width * tile_height) = 32 :=
by
  intro h
  cases' h with rw h
  cases' h with rh h
  cases' h with tw th
  rw [rw, rh, tw, th]
  norm_num
  sorry

end number_of_tiles_l519_519048


namespace visitors_equal_cats_l519_519971

theorem visitors_equal_cats (U V : Type) (E : U → V → Prop)
  [finite U] [finite V]
  (hU : ∀ u : U, (finset.univ.filter (λ v, E u v)).card = 3)
  (hV : ∀ v : V, (finset.univ.filter (λ u, E u v)).card = 3) :
  fintype.card U = fintype.card V :=
by
  sorry

end visitors_equal_cats_l519_519971


namespace find_m_find_T_n_l519_519773

def geometric_sequence := {a_n : ℕ → ℝ} -- here we will consider a_n(n) = \(2^{2-n}\)
def common_ratio := 1 / 2
def specific_term := 1 / 16
def sum_m := 63 / 16
def b_n (a_n : ℕ → ℝ) (n : ℕ) : ℝ := a_n n * Real.log (2*a_n n)
def T_n (b_n : ℕ → ℝ) (n : ℕ) : ℝ := Σ i in finRange n, b_n i

theorem find_m (a_n : ℕ → ℝ) (m : ℕ)
  (h₁ : common_ratio = (1:ℝ)/2) 
  (h₂ : a_n m = specific_term)
  (h₃ : Σ i in finRange m + 1, a_n i = sum_m) : 
  m = 6 :=
  sorry

theorem find_T_n (a_n : ℕ → ℝ) (n : ℕ)
  (h₁ : common_ratio = (1:ℝ)/2)
  (h₂ : a_n 1 = 2)
  (h₃ : ∀ n : ℕ, a_n n = 2^(2-n)) :
  T_n (b_n a_n) n = n / 2^(n-2) :=
  sorry

end find_m_find_T_n_l519_519773


namespace no_prime_divisible_by_77_l519_519278

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  sorry

end no_prime_divisible_by_77_l519_519278


namespace find_a_b_find_m_l519_519099

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + b * x ^ 2

theorem find_a_b (a b : ℝ) (h1 : f a b 1 = 4) 
  (h2 : 3 * a + 2 * b = 9) : a = 1 ∧ b = 3 :=
by {
  sorry
}

theorem find_m (m : ℝ) (h3 : ∀ x ∈ set.Icc m (m + 1), 3 * x^2 + 6 * x ≥ 0) : m ≥ 0 ∨ m ≤ -3 :=
by {
  sorry
}

end find_a_b_find_m_l519_519099


namespace how_many_times_faster_l519_519502

theorem how_many_times_faster (A B : ℝ) (h1 : A = 1 / 32) (h2 : A + B = 1 / 24) : A / B = 3 := by
  sorry

end how_many_times_faster_l519_519502


namespace circular_limit_l519_519815

noncomputable def X : set (ℝ × ℝ) := sorry
def f (A : set (ℝ × ℝ)) : set (ℝ × ℝ) := { p | ∃ q ∈ A, dist p q ≤ 1 }
def f_n (n : ℕ) (A : set (ℝ × ℝ)) : set (ℝ × ℝ) :=
  match n with
  | 0       => A
  | k + 1   => f (f_n k A)

def r_n (n : ℕ) : ℝ := sorry  -- definition as per problem
def R_n (n : ℕ) : ℝ := sorry  -- definition as per problem

theorem circular_limit (H1 : bounded X) (H2 : X.nonempty) : 
  filter.tendsto (λ n, R_n n / r_n n) filter.at_top (𝓝 1) :=
sorry

end circular_limit_l519_519815


namespace part1_part2_l519_519640

def f (x a : ℝ) : ℝ := (Exp x / x) - Log x + x - a

theorem part1 (x a : ℝ) (hx : f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 (x1 x2 a : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) 
  (hx1_range : 0 < x1) (hx2_range : x2 > 1) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519640


namespace modulo_17_residue_l519_519175

theorem modulo_17_residue :
  (342 + 6 * 47 + 8 * 157 + 3^3 * 21) % 17 = 10 :=
by
  -- Definitions and conditions
  have h1 : 342 % 17 = 1 := by norm_num,
  have h2 : 47 % 17 = 13 := by norm_num,
  have h3 : 157 % 17 = 11 := by norm_num,
  have h4 : 21 % 17 = 4 := by norm_num,
  -- Intermediate calculations
  have h5 : (6 * 47) % 17 = 11 := by norm_num,
  have h6 : (8 * 157) % 17 = 4 := by norm_num,
  have h7 : (3^3 * 21) % 17 = 11 := by norm_num,
  -- Summing up residues and concluding the proof
  have h_sum : (1 + 11 + 4 + 11) % 17 = 10 := by norm_num,
  exact h_sum

end modulo_17_residue_l519_519175


namespace number_of_valid_n_l519_519576

theorem number_of_valid_n : 
  (set.count (λ n, 
           1 + n + n^2 / 2! + n^3 / 3! + n^4 / 4! + n^5 / 5! + n^6 / 6! ∈ ℤ 
           ∧ 1 ≤ n ∧ n < 2017)
   (set.Ico 1 2017)
  ) = 134 := 
sorry

end number_of_valid_n_l519_519576


namespace area_fraction_of_internal_points_with_triangle_distances_l519_519541

theorem area_fraction_of_internal_points_with_triangle_distances (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (d_a d_b d_c : ℝ), 
    (d_a < d_b + d_c) ∧ (d_b < d_a + d_c) ∧ (d_c < d_a + d_b) → 
    ((∃ K : ℝ, 0 < K ∧ K = (a * b * c) / ((a + b) * (b + c) * (c + a)))) :=
begin
  sorry
end

end area_fraction_of_internal_points_with_triangle_distances_l519_519541


namespace trader_profit_percentage_is_50_l519_519961

variable (buy_weight_claimed : ℕ) -- The weight the trader claims to buy, e.g., 1000 grams.
variable (buy_weight_actual : ℕ) -- The actual weight the trader buys, taking 10% more.
variable (sell_weight_claimed : ℕ) -- The weight the trader claims to sell.
variable (sell_weight_actual : ℕ) -- The actual weight the trader sells, such that 50% added equals the claimed weight.
variable (cost_price : ℕ) -- Cost price per unit weight.

-- Condition 1: When he buys from the supplier, he takes 10% more than the indicated weight.
def condition1 : Prop := buy_weight_actual = buy_weight_claimed + (buy_weight_claimed / 10)

-- Condition 2: When he sells to the customer, he gives a weight such that 50% added equals the claimed weight.
def condition2 : Prop := 3 * sell_weight_actual = 2 * sell_weight_claimed

-- Condition 3: The trader charges the cost price of the weight that he claims.
def condition3 : Prop := True -- Simplified as this is inherent in the calculation.

-- The goal is to prove the profit percentage is 50%.
theorem trader_profit_percentage_is_50 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  : (( (sell_weight_claimed * cost_price) - (sell_weight_actual * cost_price) ) / (sell_weight_actual * cost_price)) * 100 = 50 := 
sorry

end trader_profit_percentage_is_50_l519_519961


namespace intersect_altitudes_right_angle_triangle_l519_519297

-- Definitions needed for the statement
variable {T : Type} [MetricSpace T] [InnerProductSpace ℝ T]

def is_altitude (A B C P : T) : Prop :=
  ∃ D, D ∈ line B C ∧ P = orthogonalProjection lineSpanEquation(A, line B C) D

def is_orthocenter (A B C H : T) : Prop :=
  is_altitude A B C H ∧ is_altitude B A C H ∧ is_altitude C A B H

-- Lean statement equivalent to the mathematical proof problem
theorem intersect_altitudes_right_angle_triangle
  {A B C H : T}
  (h : is_orthocenter A B C H)
  (h_H_vertex : H = A ∨ H = B ∨ H = C) :
  ∃ (X Y Z : T), is_right_angled_triangle X Y Z ∧ (H = X ∨ H = Y ∨ H = Z) := sorry

end intersect_altitudes_right_angle_triangle_l519_519297


namespace compare_fx_l519_519257

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem compare_fx:
  f (-Real.pi / 3) > f (-1) ∧ f (-1) > f (Real.pi / 11) :=
by
  have f_even: ∀ x, f (-x) = f x := by
    intro x
    unfold f
    rw [Real.sin_neg, mul_neg, neg_neg]

  have f'_neg: ∀ x ∈ Ioo (-Real.pi / 2) (0 : ℝ), (Real.sin x + x * Real.cos x) < 0 :=
    sorry
  
  have dec_in_interval: ∀ x y ∈ Ioo (-Real.pi / 2) (0 : ℝ), x < y → f x > f y :=
  by
    intros x y hx hy hxy
    have d := calc
      f x - f y = ∫ t in x..y, (Real.sin t + t * Real.cos t) :=
        by simp [f]
          sorry
    rw [<- integral_neg, Subtype.coe_inj] at d
    apply integral_lt_of_forall_le ⟨-, d⟩
    simpa

  apply And.intro
  show f (-Real.pi / 3) > f (-1)
  apply dec_in_area
  repeat
    exact sorry
    show f (-1) > f (Real.pi / 11)
  apply dec_in_interval
  exact sorry

end compare_fx_l519_519257


namespace solve_inequality_l519_519025

theorem solve_inequality :
  ∀ x y : ℝ, x + y ^ 2 + real.sqrt (x - y ^ 2 - 1) ≤ 1 → x = 1 ∧ y = 0 :=
by
  intros x y h
  sorry

end solve_inequality_l519_519025


namespace increased_cost_per_person_l519_519407

-- Declaration of constants
def initial_cost : ℕ := 30000000000 -- 30 billion dollars in dollars
def people_sharing : ℕ := 300000000 -- 300 million people
def inflation_rate : ℝ := 0.10 -- 10% inflation rate

-- Calculation of increased cost per person
theorem increased_cost_per_person : (initial_cost * (1 + inflation_rate) / people_sharing) = 110 :=
by sorry

end increased_cost_per_person_l519_519407


namespace sum_of_radii_l519_519947

noncomputable def tangency_equation (r : ℝ) : Prop :=
  (r - 5)^2 + r^2 = (r + 1.5)^2

theorem sum_of_radii : ∀ (r1 r2 : ℝ), tangency_equation r1 ∧ tangency_equation r2 →
  r1 + r2 = 13 :=
by
  intros r1 r2 h
  sorry

end sum_of_radii_l519_519947


namespace sum_of_digits_b_n_l519_519358

def a_n (n : ℕ) : ℕ := 10^(2^n) - 1

def b_n (n : ℕ) : ℕ :=
  List.prod (List.map a_n (List.range (n + 1)))

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_b_n (n : ℕ) : sum_of_digits (b_n n) = 9 * 2^n :=
  sorry

end sum_of_digits_b_n_l519_519358


namespace find_window_cost_l519_519799

-- Definitions (conditions)
def total_damages : ℕ := 1450
def cost_of_tire : ℕ := 250
def number_of_tires : ℕ := 3
def cost_of_tires := number_of_tires * cost_of_tire

-- The cost of the window that needs to be proven
def window_cost := total_damages - cost_of_tires

-- We state the theorem that the window costs $700 and provide a sorry as placeholder for its proof
theorem find_window_cost : window_cost = 700 :=
by sorry

end find_window_cost_l519_519799


namespace find_number_l519_519081

theorem find_number (x : ℤ) (h : 3 * x + 3 * 12 + 3 * 13 + 11 = 134) : x = 16 :=
by
  sorry

end find_number_l519_519081


namespace intersecting_planes_and_parallel_intersection_line_l519_519611

variables {m n l : Line} {α β : Plane}

-- Define the skew lines relationship
axiom skew_lines (m n : Line) : ¬ (∃ (p : Plane), m ⊆ p ∧ n ⊆ p)

-- Define perpendicularity relationships
axiom perp_line_plane (m : Line) (α : Plane) : m ⊥ α
axiom perp_line_line (m l : Line) : l ⊥ m
axiom not_in_plane (l : Line) (α : Plane) : ¬ (l ⊆ α)

-- Define the proof problem
theorem intersecting_planes_and_parallel_intersection_line
  (skew_mn : skew_lines m n)
  (m_perp_α : perp_line_plane m α)
  (n_perp_β : perp_line_plane n β)
  (l_perp_m : perp_line_line l m)
  (l_perp_n : perp_line_line l n)
  (l_not_in_α : not_in_plane l α)
  (l_not_in_β : not_in_plane l β) :
  ∃ p : Plane, α ≠ β ∧ (∀ (x : Point), x ∈ α ∩ β → x ∈ p) ∧ (l ⊆ p ∧ p ⊥ α) :=
sorry

end intersecting_planes_and_parallel_intersection_line_l519_519611


namespace planes_parallel_l519_519352

-- Given definitions and conditions
variables {Line Plane : Type}
variables (a b : Line) (α β γ : Plane)

-- Conditions from the problem
axiom perp_line_plane (line : Line) (plane : Plane) : Prop
axiom parallel_line_plane (line : Line) (plane : Plane) : Prop
axiom parallel_plane_plane (plane1 plane2 : Plane) : Prop

-- Conditions
variable (h1 : parallel_plane_plane γ α)
variable (h2 : parallel_plane_plane γ β)

-- Proof statement
theorem planes_parallel (h1 : parallel_plane_plane γ α) (h2 : parallel_plane_plane γ β) : parallel_plane_plane α β := sorry

end planes_parallel_l519_519352


namespace work_together_l519_519105

theorem work_together (dA dB : ℕ) (hA : dA = 2 * 70) (hB : dB = 3 * 35) :
  (1 / dA + 1 / dB)⁻¹ = 60 :=
by
  -- Definitions directly from conditions
  have ha : dA = 140 := by rw [hA]; norm_num
  have hb : dB = 105 := by rw [hB]; norm_num
  -- Use the conditions to express the result
  rw [ha, hb]
  have work_rate : 1 / 140 + 1 / 105 = 1 / 60 := by norm_num [one_div, add_div]
  rw [← work_rate, inv_eq_one_div]
  exact rfl

#reduce work_together 140 105 rfl rfl -- This should return true

end work_together_l519_519105


namespace carolyn_initial_marbles_l519_519987

theorem carolyn_initial_marbles (x : ℕ) (h1 : x - 42 = 5) : x = 47 :=
by
  sorry

end carolyn_initial_marbles_l519_519987


namespace probability_of_selecting_double_is_one_sixth_l519_519114

-- Define the set of dominoes including integers 0 to 12 on each square with each integer paired with every other exactly once, including doubles
def domino_set : list (ℕ × ℕ) :=
  list.bind (list.range 13) (λ n, list.map (prod.mk n) (list.range n))

-- Define the set of doubles
def doubles : list (ℕ × ℕ) :=
  list.map (λ n, (n, n)) (list.range 13)

-- Calculate the total number of dominoes
def total_dominoes : ℕ :=
  domino_set.length

-- Calculate the number of doubles
def number_of_doubles : ℕ :=
  doubles.length

-- Define the probability of picking a double domino
def probability_of_double : ℚ :=
  number_of_doubles / total_dominoes

-- Prove that the probability of selecting a double domino is 1/6.
theorem probability_of_selecting_double_is_one_sixth : probability_of_double = 1 / 6 :=
by 
  -- we use sorry here to indicate that the proof is omitted
  sorry

end probability_of_selecting_double_is_one_sixth_l519_519114


namespace find_b_l519_519968

theorem find_b (b : ℝ) (x y : ℝ) (h1 : 2 * x^2 + b * x = 12) (h2 : y = x + 5.5) (h3 : y^2 * x + y * x^2 + y * (b * x) = 12) :
  b = -5 :=
sorry

end find_b_l519_519968


namespace exist_amusing_numbers_l519_519363

/-- Definitions for an amusing number -/
def is_amusing (x : ℕ) : Prop :=
  (x >= 1000) ∧ (x <= 9999) ∧
  ∃ y : ℕ, y ≠ x ∧
  ((∀ d ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10],
    (d ≠ 0 → (y % 1000) ∈ [(d-1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10, (d+1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10]) ∧
    (d ≠ 9 → (y % 1000) ∈ [(d-1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10, (d+1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10]))) ∧
  (y % x = 0)

/-- Prove the existence of four amusing four-digit numbers -/
theorem exist_amusing_numbers :
  ∃ x1 x2 x3 x4 : ℕ, is_amusing x1 ∧ is_amusing x2 ∧ is_amusing x3 ∧ is_amusing x4 ∧ 
                   x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 :=
by sorry

end exist_amusing_numbers_l519_519363


namespace circles_tangent_externally_l519_519672

def circle_center (h k r : ℝ) : ℝ × ℝ := (h, k)
def radius (r : ℝ) : ℝ := r
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem circles_tangent_externally :
  let C1_center := circle_center 2 (-1) 3 in
  let C2_center := circle_center (-1) 3 2 in
  let r1 := radius 3 in
  let r2 := radius 2 in
  distance C1_center C2_center = r1 + r2 :=
by {
  intros,
  unfold circle_center radius distance,
  simp,
  have : (2 - (-1))^2 = 9,
  { ring },
  have : (3 - (-1))^2 = 16,
  { ring },
  simp [*],
  norm_num,
}

end circles_tangent_externally_l519_519672


namespace distance_MN_intersection_l519_519787

noncomputable def curve_c1_parametric (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos θ, 2 * Real.sin θ)

def curve_c2_polar (θ : ℝ) : ℝ :=
  2 * Real.sin θ

def line_theta (θ : ℝ) : Prop :=
  θ = Real.pi / 4

theorem distance_MN_intersection :
  ∀ θ ∈ Set.Icc 0 Real.pi,
  line_theta θ →
  let M := curve_c1_parametric θ
  let N := (curve_c2_polar θ * Real.cos θ, curve_c2_polar θ * Real.sin θ)
  Real.dist M N = Real.sqrt 2 :=
by
  intros θ hθ hθ_line M N
  sorry

end distance_MN_intersection_l519_519787


namespace gelato_combination_l519_519117

theorem gelato_combination (kinds_of_gelato : ℕ) (three_scoop_choice : ℕ) (result : ℕ) :
  kinds_of_gelato = 8 →
  three_scoop_choice = 3 →
  (result = 56) →
  (nat.choose kinds_of_gelato three_scoop_choice = result) :=
by 
  intros h1 h2 h3
  rw [h1, h2]
  rw [nat.choose] -- This resolves to the definition of combinations.
  sorry

end gelato_combination_l519_519117


namespace max_additional_payment_expected_difference_l519_519197

theorem max_additional_payment :
  let current_readings := {1402, 1347, 1337, 1298, 1270, 1214}
  let previous_readings := {1214, 1270, 1298, 1337, 1347, 1402}
  let peak_rate := 4.03
  let semi_peak_rate := 3.39
  let night_rate := 1.01
  let actual_payment := 660.72
  let peak_consumption := max(current_readings) - min(previous_readings)
  let semi_peak_consumption := ? /* Next highest difference calculation */
  let night_consumption := ? /* Smallest difference calculation */
  let peak_cost := peak_consumption * peak_rate
  let semi_peak_cost := semi_peak_consumption * semi_peak_rate
  let night_cost := night_consumption * night_rate
  let total_cost := peak_cost + semi_peak_cost + night_cost
  let additional_payment := total_cost - actual_payment
  additional_payment = 397.34 :=
sorry

theorem expected_difference :
  let current_readings := {1402, 1347, 1337, 1298, 1270, 1214}
  let previous_readings := {1214, 1270, 1298, 1337, 1347, 1402}
  let peak_rate := 4.03
  let semi_peak_rate := 3.39
  let night_rate := 1.01
  let actual_payment := 660.72
  let expected_peak_consumption := ? /* Expected value calculation */
  let expected_semi_peak_consumption := ? /* Expected value calculation */
  let expected_night_consumption := ? /* Expected value calculation */
  let expected_peak_cost := expected_peak_consumption * peak_rate
  let expected_semi_peak_cost := expected_semi_peak_consumption * semi_peak_rate
  let expected_night_cost := expected_night_consumption * night_rate
  let expected_total_cost := expected_peak_cost + expected_semi_peak_cost + expected_night_cost
  let expected_difference := expected_total_cost - actual_payment
  expected_difference = 19.3 :=
sorry

end max_additional_payment_expected_difference_l519_519197


namespace workshop_worker_count_l519_519034

theorem workshop_worker_count (W T N : ℕ) (h1 : T = 7) (h2 : 8000 * W = 7 * 14000 + 6000 * N) (h3 : W = T + N) : W = 28 :=
by
  sorry

end workshop_worker_count_l519_519034


namespace minimum_c_value_l519_519600

noncomputable theory

-- Define sequences a_n and b_n
def aₙ (n : ℕ) : ℕ := 2 * n - 1
def bₙ (n : ℕ) : ℕ := 2^n

-- Define the sequence T_n
def Tₙ (n : ℕ) : ℝ := ∑ i in finset.range n, (aₙ (i+1) / bₙ (i+1))

-- Define the condition for c and find its minimum value
theorem minimum_c_value : ∃ c, (∀ n : ℕ, n > 0 → Tₙ n + (2 * n + 3) / (2^n) - 1 / n < c) ∧ c = 3 := 
sorry

end minimum_c_value_l519_519600


namespace exists_inhabitant_with_810_acquaintances_l519_519748

-- Conditions
def total_population : ℕ := 1000000
def believers_percentage : ℕ := 90
def acquaintances_believers_percentage : ℝ := 0.10

-- Theorem statement: Prove that there exists an inhabitant knowing at least 810 people.
theorem exists_inhabitant_with_810_acquaintances 
    (H1 : ∀ (p : ℕ), p < total_population → ∃ q, q ≠ p)  -- Every person knows at least one other person
    (H2 : 0.9 * total_population = 900000)  -- 90% of the population are believers in Santa Claus
    (H3 : ∀ p, p < total_population → ∀ q, q != p → (belief p → (exists percentage, percentage = acquaintances_believers_percentage))) -- Every person claims exactly 10% of their acquaintances believe in Santa Claus
    : ∃ (habitants : ℕ), habitants ≥ 810
    :=
  sorry

end exists_inhabitant_with_810_acquaintances_l519_519748


namespace point_in_fourth_quadrant_l519_519609

variables (z : ℂ) (x y : ℝ)

def fourth_quadrant (x y : ℝ) : Prop :=
  (0 < x) ∧ (y < 0)

theorem point_in_fourth_quadrant
  (h1 : z = 1 - Complex.i) :
  fourth_quadrant z.re z.im :=
by
  sorry

end point_in_fourth_quadrant_l519_519609


namespace candy_problem_l519_519394

theorem candy_problem (N a S : ℕ) 
  (h1 : ∀ i : ℕ, i < N → a = S - 7 - a)
  (h2 : ∀ i : ℕ, i < N → a > 1)
  (h3 : S = N * a) : 
  S = 21 :=
by
  sorry

end candy_problem_l519_519394


namespace width_of_final_painting_l519_519470

theorem width_of_final_painting
  (total_area : ℕ)
  (area_paintings_5x5 : ℕ)
  (num_paintings_5x5 : ℕ)
  (painting_10x8_area : ℕ)
  (final_painting_height : ℕ)
  (total_num_paintings : ℕ := 5)
  (total_area_paintings : ℕ := 200)
  (calculated_area_remaining : ℕ := total_area - (num_paintings_5x5 * area_paintings_5x5 + painting_10x8_area))
  (final_painting_width : ℕ := calculated_area_remaining / final_painting_height) :
  total_num_paintings = 5 →
  total_area = 200 →
  area_paintings_5x5 = 25 →
  num_paintings_5x5 = 3 →
  painting_10x8_area = 80 →
  final_painting_height = 5 →
  final_painting_width = 9 :=
by
  intros h1 h2 h3 h4 h5 h6
  dsimp [final_painting_width, calculated_area_remaining]
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end width_of_final_painting_l519_519470


namespace no_prime_divisible_by_77_l519_519280

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  sorry

end no_prime_divisible_by_77_l519_519280


namespace august_five_mondays_l519_519401

-- Definitions for the conditions
def july_has_five_mondays (N : ℕ) : Prop :=
  ∃ (m : ℕ), m < 7 ∧ m ≠ 1 ∧ -- starting possibilities for Mondays other than July 1st
    (∀ d, d < 31 → (d ≡ m [MOD 7] → monday d))

def month_has_31_days (month : ℕ) : Prop := 
  month = 31

-- The known condition that both July and August have 31 days
def july_and_august_have_31_days (N : ℕ) : Prop :=
  month_has_31_days 31 ∧ month_has_31_days 31

-- The main theorem we need to prove
theorem august_five_mondays (N : ℕ) 
  (H1 : july_has_five_mondays N)
  (H2 : july_and_august_have_31_days N) :
  ∃ d, d_day_count_in_august == 5 → d = thursday := 
begin
  sorry
end

end august_five_mondays_l519_519401


namespace flowers_count_l519_519678

theorem flowers_count (lilies : ℕ) (sunflowers : ℕ) (daisies : ℕ) (total_flowers : ℕ) (roses : ℕ)
  (h1 : lilies = 40) (h2 : sunflowers = 40) (h3 : daisies = 40) (h4 : total_flowers = 160) :
  lilies + sunflowers + daisies + roses = 160 → roses = 40 := 
by
  sorry

end flowers_count_l519_519678


namespace average_weight_remainder_l519_519303

variable (total_boys : ℕ) (boys1 : ℕ) (boys2 : ℕ)
variable (avg_weight1 : ℝ) (avg_weight_total : ℝ)
variable (total_boys == 34) (boys1 == 26) (boys2 == 8)
variable (avg_weight1 == 50.25) (avg_weight_total == 49.05)

theorem average_weight_remainder :
  let total_weight1 := boys1 * avg_weight1 in
  let total_weight_total := total_boys * avg_weight_total in
  let total_weight2 := total_weight_total - total_weight1 in
  let avg_weight2 := total_weight2 / boys2 in
  avg_weight2 = 45.15 :=
sorry

end average_weight_remainder_l519_519303


namespace cistern_filled_in_12_hours_l519_519948

def fill_rate := 1 / 6
def empty_rate := 1 / 12
def net_rate := fill_rate - empty_rate

theorem cistern_filled_in_12_hours :
  (1 / net_rate) = 12 :=
by
  -- Proof omitted for clarity
  sorry

end cistern_filled_in_12_hours_l519_519948


namespace determine_alkane_formula_alkane_formula_l519_519511

theorem determine_alkane_formula (n : ℕ) (ωC : ℝ) (h : ωC = 0.84) (h_alkane : ωC = 12 * n / (14 * n + 2)) : n = 7 :=
by
  -- Given conditions
  have ωC_eq : 12 * n / (14 * n + 2) = 0.84 := h_alkane
  
  -- Prove that n = 7
  sorry

theorem alkane_formula : ∃ (n : ℕ), (ωC : ℝ) (h : ωC = 0.84) (h_alkane : ωC = 12 * n / (14 * n + 2)), n = 7 :=
by
  exists 7
  exists 0.84
  split
  { refl }
  { rw [←eq_div_iff_mul_eq (14 * n + 2)] at h
    sorry }

end determine_alkane_formula_alkane_formula_l519_519511


namespace exists_at_least_60_positions_l519_519068

-- Define the problem's parameters
def num_sectors : ℕ := 1965
def num_red_sectors : ℕ := 200

-- Define a proposition to capture the alignment properties
def prop (positions : Finset ℕ) : Prop :=
  ∀ n ∈ positions, ∑ k in (Finset.range num_sectors).filter (λ k, (k + n) % num_sectors ∈ (Finset.range num_red_sectors)),
    if (k + n) % num_sectors ∈ Finset.range num_red_sectors then 1 else 0 ≤ 20

-- Prove that there exists at least 60 positions where the number of overlapping red sectors is no more than 20
theorem exists_at_least_60_positions :
  ∃ positions : Finset ℕ, positions.card = 60 ∧ prop positions :=
sorry

end exists_at_least_60_positions_l519_519068


namespace symmetric_sum_of_symmetric_symmetric_unimodal_sum_l519_519361

noncomputable theory

-- Define what it means to be a symmetric random variable
def symmetric_rv {Ω : Type*} [measurable_space Ω] (X : Ω → ℝ) : Prop :=
  ∀ t, (measure_theory.measure_preimage_eq (measure_theory.likelihood X) t) = 
       (measure_theory.measure_preimage_eq (measure_theory.likelihood (-X)) t)

-- Define a unimodal random variable with mode at zero
def unimodal_rv {Ω : Type*} [measurable_space Ω] (X : Ω → ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x y, x < 0 → y > 0 → h x ≥ h y) ∧ (∀ x, h x = X x)

variables {Ω : Type*} [measurable_space Ω]

-- Part (a) statement
theorem symmetric_sum_of_symmetric (X Y : Ω → ℝ) 
  (hX_symm : symmetric_rv X) 
  (hY_symm : symmetric_rv Y) 
  (indep : measure_theory.independent (set.range X) (set.range Y)) : 
  symmetric_rv (X + Y) := sorry

-- Part (b) statement
theorem symmetric_unimodal_sum (X Y : Ω → ℝ) 
  (hX_symm : symmetric_rv X) 
  (hY_symm : symmetric_rv Y) 
  (hX_unimodal : unimodal_rv X) 
  (hY_unimodal : unimodal_rv Y) 
  (indep : measure_theory.independent (set.range X) (set.range Y)) : 
  symmetric_rv (X + Y) ∧ unimodal_rv (X + Y) := sorry

end symmetric_sum_of_symmetric_symmetric_unimodal_sum_l519_519361


namespace part1_part2_l519_519645

def f (x a : ℝ) : ℝ := (Exp x / x) - Log x + x - a

theorem part1 (x a : ℝ) (hx : f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 (x1 x2 a : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) 
  (hx1_range : 0 < x1) (hx2_range : x2 > 1) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519645


namespace sin_abs_is_even_l519_519097

theorem sin_abs_is_even : 
  ∀ x : ℝ, sin (|x|) = sin (| -x |) :=
by
  assume x
  rw [abs_neg]
  -- concludes that sin (| x |) is an even function
  trivial

end sin_abs_is_even_l519_519097


namespace correct_sampling_methods_l519_519179

-- Definitions for the conditions in the problem
def community := (high_income: ℕ, middle_income: ℕ, low_income: ℕ)
def middle_school := (specialty_students: ℕ)

-- Conditions given in the problem
def community_population : community := (125, 280, 95)
def middle_school_students : middle_school := (15)

-- Theorem stating the correct sampling methods
theorem correct_sampling_methods (c_pop : community) (m_students : middle_school) :
  c_pop = community_population →
  m_students = middle_school_students →
  (stratified_sampling c_pop → simple_random_sampling m_students) := 
sorry

end correct_sampling_methods_l519_519179


namespace books_ratio_l519_519520

variable (books_Beatrix books_Alannah books_Queen : ℕ)
variable (total_books : ℕ := 140)

-- Given Conditions from the problem
def Beatrix_books : ℕ := 30
def Alannah_books : ℕ := Beatrix_books + 20
def Queen_books : ℕ := total_books - Alannah_books - Beatrix_books

theorem books_ratio : (Queen_books - Alannah_books) / 10 = 1 ∧ Alannah_books / 10 = 5 :=
by
  have Beatrix_books_eq : Beatrix_books = 30 := rfl
  have Alannah_books_eq : Alannah_books = 50 := by simp [Alannah_books, Beatrix_books_eq]
  have Queen_books_eq : Queen_books = 60 := by simp [Queen_books, total_books, Alannah_books_eq, Beatrix_books_eq]
  have diff_books_eq : Queen_books - Alannah_books = 10 := by simp [Queen_books_eq, Alannah_books_eq]
  have ratio_simplified : (10:50) = (1:5) := by norm_num
  sorry -- complete the proof steps by connecting these results to validate the statement

end books_ratio_l519_519520


namespace max_regions_7_dots_l519_519101

-- Definitions based on conditions provided.
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def R (n : ℕ) : ℕ := 1 + binom n 2 + binom n 4

-- The goal is to state the proposition that the maximum number of regions created by joining 7 dots on a circle is 57.
theorem max_regions_7_dots : R 7 = 57 :=
by
  -- The proof is to be filled in here
  sorry

end max_regions_7_dots_l519_519101


namespace julio_salary_l519_519809

-- Define the conditions
def customers_first_week : ℕ := 35
def customers_second_week : ℕ := 2 * customers_first_week
def customers_third_week : ℕ := 3 * customers_first_week
def commission_per_customer : ℕ := 1
def bonus : ℕ := 50
def total_earnings : ℕ := 760

-- Calculate total commission and total earnings
def commission_first_week : ℕ := customers_first_week * commission_per_customer
def commission_second_week : ℕ := customers_second_week * commission_per_customer
def commission_third_week : ℕ := customers_third_week * commission_per_customer
def total_commission : ℕ := commission_first_week + commission_second_week + commission_third_week
def total_earnings_commission_bonus : ℕ := total_commission + bonus

-- Define the proof problem
theorem julio_salary : total_earnings - total_earnings_commission_bonus = 500 :=
by
  sorry

end julio_salary_l519_519809


namespace xiao_ming_pass_probability_l519_519089

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability_success (p : ℚ) (n k : ℕ) : ℚ :=
  (binomial_coefficient n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem xiao_ming_pass_probability :
  let p := (3 : ℚ) / 4 in
  let n := 3 in
  let k := 1 in
  probability_success p n k = 9 / 64 :=
by
  sorry

end xiao_ming_pass_probability_l519_519089


namespace data_transmission_time_l519_519564

theorem data_transmission_time
    (blocks : ℕ)
    (chunks_per_block : ℕ)
    (transmission_rate : ℕ) :
    blocks = 100 → chunks_per_block = 256 → transmission_rate = 150 →
    (256 * 100) / 150 / 60 ≈ 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end data_transmission_time_l519_519564


namespace variance_transformation_l519_519266

theorem variance_transformation (x : ℕ → ℝ) (n : ℕ) (a b : ℝ)
  (h1 : variance (λ i, x i) n = 3)
  (h2 : variance (λ i, a * x i + b) n = 12) : a = 2 ∨ a = -2 := 
  sorry

end variance_transformation_l519_519266


namespace related_variables_l519_519087

-- Definitions based on the provided conditions
def edge_length (c : ℝ) : Prop := c > 0
def cube_volume (v : ℝ) : Prop := v > 0
def angle_radian (θ : ℝ) : Prop := 0 ≤ θ ∧ θ ≤ 2 * Real.pi
def sine_value (s : ℝ) : Prop := -1 ≤ s ∧ s ≤ 1
def daylight_duration (d : ℝ) : Prop := d > 0
def rice_yield (y : ℝ) : Prop := y ≥ 0
def height (h : ℝ) : Prop := h > 0
def eyesight (e : ℝ) : Prop := e > 0

-- Problem statement written in Lean
theorem related_variables :
  ¬ ∃ c v, edge_length c ∧ cube_volume v ∧ deterministic_relationship c v ∧
  ¬ ∃ θ s, angle_radian θ ∧ sine_value s ∧ deterministic_relationship θ s ∧
  (∃ d y, daylight_duration d ∧ rice_yield y ∧ related d y) ∧
  ¬ ∃ h e, height h ∧ eyesight e ∧ related h e :=
sorry

end related_variables_l519_519087


namespace find_C_and_cos_C_l519_519322

variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def triangle_conditions : Prop :=
  let a_contains := a = sin A
  let b_contains := b = sin B
  let c_contains := c = sin C
  -- Perimeter condition
  let perimeter := a + b + c = 9 
  -- Given sine condition
  let sine_condition := (sin A + sin B = (5 / 4) * sin C)
  -- Given area condition
  let area_condition := (1 / 2) * a * b * sin C = 3 * sin C
  a_contains ∧ b_contains ∧ c_contains ∧ perimeter ∧ sine_condition ∧ area_condition

-- Main theorem to prove C and cos(C)
theorem find_C_and_cos_C (h : triangle_conditions) :
  C = 4 ∧ cos C = -(1 / 4) :=
  by
    sorry

end find_C_and_cos_C_l519_519322


namespace coordinates_of_F_l519_519671

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem coordinates_of_F'' :
  let F := (-2, 1)
  let F' := reflect_y_axis F
  let F'' := reflect_x_axis F'
  F'' = (2, -1) :=
by
  let F := (-2, 1)
  let F' := reflect_y_axis F
  let F'' := reflect_x_axis F'
  show F'' = (2, -1)
  by sorry

end coordinates_of_F_l519_519671


namespace points_not_necessarily_midpoints_l519_519372

/-!
Given a triangle ABC with angles A, B, and C, and points C₁, A₁, and B₁ on sides 
AB, BC, and AC respectively such that:
- C₁ is the midpoint of AB 
- ∠B₁C₁A₁ = ∠C
- ∠C₁A₁B₁ = ∠A
- ∠A₁B₁C₁ = ∠B

Prove that A₁ and B₁ are not necessarily the midpoints of BC and AC respectively.
-/
theorem points_not_necessarily_midpoints 
  {A B C A₁ B₁ C₁: Type} [Point A B C A₁ B₁ C₁]
  (hC₁: is_midpoint C₁ A B)
  (h1: ∠B₁C₁A₁ = ∠C)
  (h2: ∠C₁A₁B₁ = ∠A)
  (h3: ∠A₁B₁C₁ = ∠B) 
  : ¬(is_midpoint A₁ B C ∧ is_midpoint B₁ A C) := sorry

end points_not_necessarily_midpoints_l519_519372


namespace find_distance_MN_l519_519785

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos θ, 2 * Real.sin θ)

noncomputable def cartesian_eq_C1 (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4 ∧ 0 ≤ y ∧ y ≤ 2

noncomputable def polar_eq_C2 (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sin θ

noncomputable def line_l (θ line_theta : ℝ) : Prop :=
  line_theta = Real.pi / 4

noncomputable def distance_MN (M N : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)

theorem find_distance_MN :
  (M N : ℝ × ℝ)
  (θ := Real.pi / 4)
  (ρ := 2 * Real.sqrt 2)
  (ρ1 := 2 * Real.cos θ)
  (ρ2 := 2 * Real.sin θ)
  (M = (ρ1, θ))
  (N = (ρ2, θ))
  (abs_dist := abs ((2 * Real.sqrt 2) - Real.sqrt 2)) :
  abs_dist = Real.sqrt 2 :=
sorry

end find_distance_MN_l519_519785


namespace edward_skee_ball_tickets_l519_519198

theorem edward_skee_ball_tickets (w_tickets : Nat) (candy_cost : Nat) (num_candies : Nat) (total_tickets : Nat) (skee_ball_tickets : Nat) :
  w_tickets = 3 ∧ candy_cost = 4 ∧ num_candies = 2 ∧ total_tickets = num_candies * candy_cost ∧ total_tickets - w_tickets = skee_ball_tickets → 
  skee_ball_tickets = 5 :=
by
  sorry

end edward_skee_ball_tickets_l519_519198


namespace total_spending_in_4_years_is_680_l519_519906

-- Define the spending of each person
def trevor_spending_yearly : ℕ := 80
def reed_spending_yearly : ℕ := trevor_spending_yearly - 20
def quinn_spending_yearly : ℕ := reed_spending_yearly / 2

-- Define the total spending in one year
def total_spending_yearly : ℕ := trevor_spending_yearly + reed_spending_yearly + quinn_spending_yearly

-- Let's state what we need to prove
theorem total_spending_in_4_years_is_680 : 
  4 * total_spending_yearly = 680 := 
by 
  -- this is where the proof steps would go
  sorry

end total_spending_in_4_years_is_680_l519_519906


namespace find_alpha_l519_519777

def demand_function (p : ℝ) : ℝ := 150 - p
def supply_function (p : ℝ) : ℝ := 3 * p - 10

def new_demand_function (p : ℝ) (α : ℝ) : ℝ := α * (150 - p)

theorem find_alpha (α : ℝ) :
  (∃ p₀ p_new, demand_function p₀ = supply_function p₀ ∧ 
    p_new = p₀ * 1.25 ∧ 
    3 * p_new - 10 = new_demand_function p_new α) →
  α = 1.4 :=
by
  sorry

end find_alpha_l519_519777


namespace amber_bronze_cells_selection_l519_519839

theorem amber_bronze_cells_selection (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b)
  (cells : Fin (a + b + 1) → Fin (a + b + 1) → Bool)
  (amber : ∀ i j, cells i j = tt → Bool) -- True means amber, False means bronze
  (bronze : ∀ i j, cells i j = ff → Bool)
  (amber_count : (Finset.univ.product Finset.univ).filter (λ p, cells p.1 p.2 = tt).card ≥ a^2 + a * b - b)
  (bronze_count : (Finset.univ.product Finset.univ).filter (λ p, cells p.1 p.2 = ff).card ≥ b^2 + b * a - a) :
  ∃ (α : Finset (Fin (a + b + 1) × Fin (a + b + 1))), 
    (α.filter (λ p, cells p.1 p.2 = tt)).card = a ∧ 
    (α.filter (λ p, cells p.1 p.2 = ff)).card = b ∧ 
    (∀ (p1 p2 : Fin (a + b + 1) × Fin (a + b + 1)), p1 ≠ p2 → (p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) :=
by sorry

end amber_bronze_cells_selection_l519_519839


namespace math_problem_l519_519284

theorem math_problem (x : ℝ) 
  (h : x + sqrt (x^2 - 4) + 1 / (x - sqrt (x^2 - 4)) = 30) :
  x^2 + sqrt (x^4 - 16) + 1 / (x^2 + sqrt (x^4 - 16)) = 52441 / 900 :=
by
  sorry

end math_problem_l519_519284


namespace dice_composite_probability_l519_519691

theorem dice_composite_probability :
  (∃ p : ℚ, p = 6298376/6298606 ∧
  (let outcomes := (fintype.card (fin 6))^9 in
   let non_composite_outcomes := 1 + 27 + 252 in
   (1 - non_composite_outcomes / outcomes) = p)) :=
sorry

end dice_composite_probability_l519_519691


namespace jayden_total_money_earned_l519_519802

/--
Jayden has a debt that he repays by gardening. He gardens for 47 hours where the payment per hour resets every 6 hours as follows:
1st hour: $2
2nd hour: $3
3rd hour: $4
4th hour: $5
5th hour: $6
6th hour: $7

If Jayden works 47 hours, show that the total amount of money he earns (and thus the amount he borrowed) is $209.
-/
theorem jayden_total_money_earned : 
  let cycle_payments := [2, 3, 4, 5, 6, 7]
  let total_hours := 47
  let complete_cycle_hours := 6
  let total_cycles := total_hours / complete_cycle_hours
  let remaining_hours := total_hours % complete_cycle_hours
  let complete_cycle_earnings := total_cycles * (cycle_payments.sum)
  let remaining_hours_earnings := (cycle_payments.take remaining_hours).sum
  complete_cycle_earnings + remaining_hours_earnings = 209 :=
by
  let cycle_payments := [2, 3, 4, 5, 6, 7]
  let total_hours := 47
  let complete_cycle_hours := 6
  let total_cycles := total_hours / complete_cycle_hours
  let remaining_hours := total_hours % complete_cycle_hours
  let complete_cycle_earnings := total_cycles * (cycle_payments.sum)
  let remaining_hours_earnings := (cycle_payments.take remaining_hours).sum
  sorry

end jayden_total_money_earned_l519_519802


namespace apples_eaten_by_children_l519_519950

theorem apples_eaten_by_children (initial_apples total_children apples_per_child apples_sold apples_left : ℕ) 
  (h1 : total_children = 5) 
  (h2 : apples_per_child = 15)
  (h3 : initial_apples = total_children * apples_per_child)
  (h4 : apples_sold = 7)
  (h5 : apples_left = 60) :
  initial_apples - apples_left - apples_sold = 8 :=
by {
  -- We will use the provided assumptions to prove the theorem
  subst h1,
  subst h2,
  subst h3,
  subst h4,
  subst h5,
  simp, -- Simplify the expression
  sorry -- Proof will be here
}

end apples_eaten_by_children_l519_519950


namespace range_of_a_product_of_zeros_l519_519658

def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - (Real.log x) + x - a

theorem range_of_a (a : ℝ) : (∀ x, (0 < x) → (f x a ≥ 0)) ↔ (a ≤ Real.exp 1 + 1) :=
by
  sorry

theorem product_of_zeros (x1 x2 a : ℝ) : 
  (0 < x1) → (x1 < 1) → (1 < x2) → (f x1 a = 0 ∧ f x2 a = 0) → x1 * x2 < 1 :=
by
  sorry

end range_of_a_product_of_zeros_l519_519658


namespace shaded_fraction_of_regular_octagon_l519_519374

noncomputable def fraction_shaded (O : Point) (octagon : RegularOctagon) (X : Point) : ℚ :=
  let A := octagon.vertex 1
  let B := octagon.vertex 2
  if O = octagon.center ∧ X = midpoint A B then
    7 / 16
  else
    0

theorem shaded_fraction_of_regular_octagon (O : Point) (octagon : RegularOctagon) (X : Point)
  (hO : O = octagon.center) (hX : X = midpoint (octagon.vertex 1) (octagon.vertex 2)) :
  fraction_shaded O octagon X = 7 / 16 := by
  sorry

end shaded_fraction_of_regular_octagon_l519_519374


namespace rectangle_diagonals_cos_angle_zero_l519_519036

theorem rectangle_diagonals_cos_angle_zero 
  (A B C D O : Type)
  [ordered_ring A] [ordered_ring B] [ordered_ring C] [ordered_ring D] [ordered_ring O]
  (is_rectangle : ∀ (A B C D : Type), (∃ (O : Type ), (AC = 15 ∧ BD = 18)))
  (diagonals_intersect : ∀ (A B C D O : Type), (∃ (AC : ordered_ring A), (∃ (BD : ordered_ring B), (AC = 15 ∧ BD = 18))))
  : cos_angle A O B = 0 := 
sorry

end rectangle_diagonals_cos_angle_zero_l519_519036


namespace problem_1_problem_2_l519_519706

noncomputable def A := Real.pi / 3
noncomputable def b := 5
noncomputable def c := 4 -- derived from the solution
noncomputable def S : ℝ := 5 * Real.sqrt 3

theorem problem_1 (A : ℝ) 
  (h : Real.cos (2 * A) - 3 * Real.cos (Real.pi - A) = 1) 
  : A = Real.pi / 3 :=
sorry

theorem problem_2 (a : ℝ) 
  (b : ℝ) 
  (S : ℝ) 
  (h_b : b = 5) 
  (h_S : S = 5 * Real.sqrt 3) 
  : a = Real.sqrt 21 :=
sorry

end problem_1_problem_2_l519_519706


namespace find_integer_a_l519_519215

theorem find_integer_a (a : ℤ) : 
  (∃ p : ℤ[X], x^13 + x + 90 = (x^2 - x + a) * p) → 
  a ∣ 90 → 
  a ∣ 92 → 
  (a + 2) ∣ 88 → 
  a = 2 :=
begin
  sorry
end

end find_integer_a_l519_519215


namespace necessary_but_not_sufficient_l519_519233

theorem necessary_but_not_sufficient (p q : Prop) : 
  (p ∨ q) → (p ∧ q) → False :=
by
  sorry

end necessary_but_not_sufficient_l519_519233


namespace range_of_a_l519_519353

theorem range_of_a (a : ℝ) (h : ∀ x ∈ Icc 3 4, x^2 - 3 > a * x - a) : a < 3 :=
sorry

end range_of_a_l519_519353


namespace inscribed_circle_radius_l519_519371

open Real

-- Define the convex quadrilateral and points
def is_convex_quadrilateral (A B C D : ℝ) : Prop :=
  -- Add convexity condition here (simplified for demonstration)
  sorry

-- Define the heights and points M criteria
def point_M (A B C D M : ℝ) : Prop :=
  -- Add conditions of point M here (simplified for demonstration)
  ∠AMD = ∠ADB ∧ ∠ACM = ∠ABC ∧ sorry

-- Define the ratio condition
def ratio_condition (h_A h_C : ℝ) : Prop :=
  3 * (h_A / h_C) ^ 2 = 2

-- Define the distance CD
def distance_CD : ℝ := 20

-- Define the radius of the inscribed circle of triangle ACD
noncomputable def inradius (A D C : ℝ) : ℝ :=
  -- Compute the inradius (simplified for demonstration)
  sorry

-- Theorem statement
theorem inscribed_circle_radius (A B C D M : ℝ) 
  (h_A h_C : ℝ) 
  (quad_convex : is_convex_quadrilateral A B C D)
  (pointM : point_M A B C D M)
  (ratio_cond : ratio_condition h_A h_C)
  (CD_dist : CD = 20) :
  inradius A D C = 4 * sqrt 10 - 2 * sqrt 15 :=
sorry


end inscribed_circle_radius_l519_519371


namespace price_of_item_a_l519_519459

theorem price_of_item_a : 
  let coins_1000 := 7
  let coins_100 := 4
  let coins_10 := 5
  let price_1000 := coins_1000 * 1000
  let price_100 := coins_100 * 100
  let price_10 := coins_10 * 10
  let total_price := price_1000 + price_100 + price_10
  total_price = 7450 := by
    sorry

end price_of_item_a_l519_519459


namespace part1_part2_l519_519637

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l519_519637


namespace calculate_complex_value_l519_519176

theorem calculate_complex_value (a b : ℂ) (h1 : a = 3 + 2 * Complex.i) (h2 : b = 2 - Complex.i) :
  3 * a + 4 * b = 17 + 2 * Complex.i :=
by
  sorry

end calculate_complex_value_l519_519176


namespace lauren_total_money_made_is_correct_l519_519345

-- Define the rate per commercial view
def rate_per_commercial_view : ℝ := 0.50
-- Define the rate per subscriber
def rate_per_subscriber : ℝ := 1.00
-- Define the number of commercial views on Tuesday
def commercial_views : ℕ := 100
-- Define the number of new subscribers on Tuesday
def subscribers : ℕ := 27
-- Calculate the total money Lauren made on Tuesday
def total_money_made (rate_com_view : ℝ) (rate_sub : ℝ) (com_views : ℕ) (subs : ℕ) : ℝ :=
  (rate_com_view * com_views) + (rate_sub * subs)

-- Theorem stating that the total money Lauren made on Tuesday is $77.00
theorem lauren_total_money_made_is_correct : total_money_made rate_per_commercial_view rate_per_subscriber commercial_views subscribers = 77.00 :=
by
  sorry

end lauren_total_money_made_is_correct_l519_519345


namespace ratio_of_angles_l519_519155

-- Define the geometric setup and given conditions
def triangle (A B C O : Point) (circle_centered_at_O : Circle O) : Prop :=
  △ABC is acute-angled ∧ inscribed_in_circle centered_at O

def given_conditions (A B C O E : Point) : Prop :=
  triangle A B C O ∧
  arc AB of circle_centered_at_O with measure 150 degrees ∧
  arc BC of circle_centered_at_O with measure 60 degrees ∧
  perpendicular OE to line AC ∧
  E is on the minor arc AC

-- Statement of the problem to prove
theorem ratio_of_angles {A B C O E : Point} (h : given_conditions A B C O E) : 
  (angle OBE / angle BAC) = 2 :=
sorry

end ratio_of_angles_l519_519155


namespace part1_acute_triangle_prove_B_eq_pi_over_4_part2_max_area_of_triangle_ABC_l519_519781

variables {A B C a b c : ℝ}

-- Part 1: Prove that B = π/4 given the conditions
theorem part1_acute_triangle_prove_B_eq_pi_over_4
  (h1 : 0 < A ∧ A < π / 2)
  (h2 : 0 < B ∧ B < π / 2)
  (h3 : 0 < C ∧ C < π / 2)
  (h4 : a^2 + b^2 = c^2 + ab * (cos (A + B) / (sin B * cos B))) :
  B = π / 4 :=
sorry

-- Part 2: Prove the maximum area of triangle ABC
theorem part2_max_area_of_triangle_ABC
  (hB : B = π / 4)
  (hb : b = 2) :
  let s : ℝ := 1 + sqrt 2
  in true :=
sorry

end part1_acute_triangle_prove_B_eq_pi_over_4_part2_max_area_of_triangle_ABC_l519_519781


namespace locus_of_point_G_l519_519598

noncomputable def locus_point_G (A B C D E F G : Real) : Prop :=
  let triangle_ABC : Prop := ∃ (A B C : Point), right_triangle A B C ∧ angle A B C = 90
  let D_on_hypotenuse : Prop := D ∈ segment B C
  let perpendiculars_constructed : Prop := is_perpendicular (line_through D E) (line_through B C) ∧
                                           is_perpendicular (line_through D F) (line_through B C)
  let intersection_point : Prop := G = intersection (line_through C E) (line_through B F)
  let on_circumcircle : Prop := G ∈ circumcircle (triangle A B C)
  triangle_ABC ∧ D_on_hypotenuse ∧ perpendiculars_constructed ∧ intersection_point → on_circumcircle

theorem locus_of_point_G
  (A B C D E F G : Point)
  (h1 : ∃ (A B C : Point), right_triangle A B C ∧ angle A B C = 90)
  (h2 : D ∈ segment B C)
  (h3 : is_perpendicular (line_through D E) (line_through B C) ∧ 
        is_perpendicular (line_through D F) (line_through B C))
  (h4 : G = intersection (line_through C E) (line_through B F)) :
  G ∈ circumcircle (triangle A B C) :=
begin
  sorry
end

end locus_of_point_G_l519_519598


namespace runs_by_running_percentage_l519_519473

def total_runs := 125
def boundaries := 5
def boundary_runs := boundaries * 4
def sixes := 5
def sixes_runs := sixes * 6
def runs_by_running := total_runs - (boundary_runs + sixes_runs)
def percentage_runs_by_running := (runs_by_running : ℚ) / total_runs * 100

theorem runs_by_running_percentage :
  percentage_runs_by_running = 60 := by sorry

end runs_by_running_percentage_l519_519473


namespace angles_measure_l519_519779

theorem angles_measure (A B C : ℝ) (h1 : A + B = 180) (h2 : C = 1 / 2 * B) (h3 : A = 6 * B) :
  A = 1080 / 7 ∧ B = 180 / 7 ∧ C = 90 / 7 :=
by
  sorry

end angles_measure_l519_519779


namespace root_of_quadratic_poly_l519_519427

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

def poly_has_root (a b c : ℝ) (r : ℝ) : Prop := a * r^2 + b * r + c = 0

theorem root_of_quadratic_poly 
  (a b c : ℝ)
  (h1 : discriminant a b c = 0)
  (h2 : discriminant (-a) (b - 30 * a) (17 * a - 7 * b + c) = 0):
  poly_has_root a b c (-11) :=
sorry

end root_of_quadratic_poly_l519_519427


namespace part1_part2_l519_519646

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519646


namespace ellipse_eccentricity_range_l519_519038

theorem ellipse_eccentricity_range (a b : ℝ) (e : ℝ) (F1 F2 P : ℝ × ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h_ellipse : ∀ x y, (x, y) = P → (x^2 / a^2 + y^2 / b^2 = 1))
  (h_PF1_PF2_dot : (λ (x1 y1 x2 y2 : ℝ),
    let PF1 := (x1 - F1.1, y1 - F1.2) in
    let PF2 := (x2 - F2.1, y2 - F2.2) in
    PF1.1 * PF2.1 + PF1.2 * PF2.2 = 1/2 * b^2) P.1 P.2 F1.1 F1.2 F2.1 F2.2) :
  e ∈ Ico (Real.sqrt 3 / 3) 1 :=
sorry

end ellipse_eccentricity_range_l519_519038


namespace sum_a1_through_a8_sum_a8_a6_a4_a2_a0_l519_519820

-- Define the conditions given in the problem
def poly_repr (x : ℝ) : ℝ := (3 * x ^ (-1) ^ 8)
noncomputable def poly_a (x : ℝ) : ℝ := (a_8 : ℝ) * x ^ 8 + a_7 * x ^ 7 + a_6 * x ^ 6 + a_5 * x ^ 5 + a_4 * x ^ 4 + a_3 * x ^ 3 + a_2 * x ^ 2 + a_1 * x + (1 : ℝ)

-- First proof
theorem sum_a1_through_a8 (a_8 a_7 a_6 a_5 a_4 a_3 a_2 a_1 : ℝ) (a_0 : ℝ) :
  (poly_repr 1 = 2^8) → 
  (poly_a 1 = poly_repr 1) →
  (a_0 = 1) →
  (a_8 + a_7 + a_6 + a_5 + a_4 + a_3 + a_2 + a_1 = 255) :=
by
  sorry

-- Second proof
theorem sum_a8_a6_a4_a2_a0 (a_8 a_7 a_6 a_5 a_4 a_3 a_2 a_1 : ℝ) :
  (poly_repr 1 = 2^8) →
  (poly_a (1:ℝ) = poly_repr 1) →
  (poly_repr (-1:ℝ) = 4^8) → 
  (poly_a (-1:ℝ) = poly_repr (-1:ℝ)) →
  (((a_8 - a_7 + a_6 - a_5 + a_4 - a_3 + a_2 - a_1 + (1:ℝ)) = 4 ^ 8) →
  (a_8 + a_6 + a_4 + a_2 + (1:ℝ) = 255 / 2 + 4 ^ 8 / 2) →
  (a_8 + a_6 + a_4 + a_2 + 1 = 32896)) :=
by
  sorry
  

end sum_a1_through_a8_sum_a8_a6_a4_a2_a0_l519_519820


namespace max_value_of_y_l519_519590

-- Defining the conditions and function
def y_function (x : ℝ) : ℝ := (1 / (4 * x - 2)) + 4 * x - 5

-- Stating the problem: Given x < 1/2, prove y_function x <= -5
theorem max_value_of_y (x : ℝ) (h : x < 1 / 2) : y_function x ≤ -5 :=
sorry

end max_value_of_y_l519_519590


namespace inverse_proportion_graph_l519_519295

theorem inverse_proportion_graph (k : ℝ) (x : ℝ) (y : ℝ) (h1 : y = k / x) (h2 : (3, -4) ∈ {p : ℝ × ℝ | p.snd = k / p.fst}) :
  k < 0 → ∀ x1 x2 : ℝ, x1 < x2 → y1 = k / x1 → y2 = k / x2 → y1 < y2 := by
  sorry

end inverse_proportion_graph_l519_519295


namespace range_of_a_product_of_extreme_points_l519_519624

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * real.log x - a * x^2
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 1 + real.log x - 2 * a * x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' x1 a = 0 ∧ f' x2 a = 0) ↔ (0 < a ∧ a < 1/2) :=
by
  sorry

theorem product_of_extreme_points (a : ℝ) (x1 x2 : ℝ) (h_distinct : x1 < x2)
    (hx1 : f' x1 a = 0) (hx2 : f' x2 a = 0) : x1 * x2 > 1 :=
by
  sorry

end range_of_a_product_of_extreme_points_l519_519624


namespace finite_transformation_l519_519227

-- Define the function representing the number transformation
def transform (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 5

-- Define the predicate stating that the process terminates
def process_terminates (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ transform^[k] n = 1

-- Lean 4 statement for the theorem
theorem finite_transformation (n : ℕ) (h : n > 1) : process_terminates n ↔ ¬ (∃ m : ℕ, m > 0 ∧ n = 5 * m) :=
by
  sorry

end finite_transformation_l519_519227


namespace problem_l519_519998

def op (x y : ℝ) : ℝ := x^2 - y

theorem problem (h : ℝ) : op h (op h h) = h :=
by
  sorry

end problem_l519_519998


namespace simplify_expression_1_simplify_expression_2_evaluate_simplified_expression_l519_519482

variables (a b x y : ℝ)

-- Problem 1
theorem simplify_expression_1 :
  2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b :=
by sorry

-- Problem 2 Part 1: Simplification
theorem simplify_expression_2 :
  3 * x^2 * y - (2 * x * y^2 - 2 * (x * y - 1.5 * x^2 * y) + x * y) + 3 * x * y^2 = x * (y^2 + y) :=
by sorry

-- Problem 2 Part 2: Specific values
theorem evaluate_simplified_expression (hx : x = -3) (hy : y = -2) :
  x * (y^2 + y) = -6 :=
by { rw [hx, hy], simp }

end simplify_expression_1_simplify_expression_2_evaluate_simplified_expression_l519_519482


namespace max_thoughtful_positions_5x5_grid_l519_519887

noncomputable def max_thoughtful_positions : Nat := 26

def manhattan_distance (p1 p2 : (ℤ × ℤ)) : ℤ :=
  abs (p1.1 - p2.1) + abs (p1.2 - p2.2)

theorem max_thoughtful_positions_5x5_grid :
(∀ (x_1 y_1 x_2 y_2 x_3 y_3 : ℤ),
  0 ≤ x_1 ∧ x_1 ≤ 4 ∧ 0 ≤ y_1 ∧ y_1 ≤ 4 ∧
  0 ≤ x_2 ∧ x_2 ≤ 4 ∧ 0 ≤ y_2 ∧ y_2 ≤ 4 ∧
  0 ≤ x_3 ∧ x_3 ≤ 4 ∧ 0 ≤ y_3 ∧ y_3 ≤ 4 ∧
  (∀ (x y : ℤ), 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 4 →
                manhattan_distance (x, y) (x_2, y_2) =
                manhattan_distance (x, y) (x_3, y_3)) →
  (count (λ (p : ℤ × ℤ), 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 4 ∧
                                manhattan_distance p (x_2, y_2) =
                                manhattan_distance p (x_3, y_3))
                        ((λ (x y : Nat), (x, y)) <$> finset.range 5 <*> finset.range 5) ≤ max_thoughtful_positions)) sorry

end max_thoughtful_positions_5x5_grid_l519_519887


namespace average_p_q_l519_519050

theorem average_p_q (p q : ℝ) 
  (h1 : (4 + 6 + 8 + 2 * p + 2 * q) / 7 = 20) : 
  (p + q) / 2 = 30.5 :=
by
  sorry

end average_p_q_l519_519050


namespace at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519720

theorem at_least_one_inhabitant_knows_no_fewer_than_810_people :
  ∀ (n : ℕ),
  (inhabitants : ℕ) (knows_someone : inhabitants > 1) (believes_santa : inhabitants * 9 / 10)
  (ten_percent_acquaintances_believe : ∀ x, (acquaintances : list x) → (|list.contacts_with_santa <- 0.10 * acquaintances|)),
  (h : inhabitants = 1000000),
  (one_inhabitant_knows : ∃ x, ∀ y, y \in acquaintances x → friends y ≥ 810) :=
sorry

end at_least_one_inhabitant_knows_no_fewer_than_810_people_l519_519720


namespace kishore_savings_l519_519519

-- Define the monthly expenses and condition
def expenses : Real :=
  5000 + 1500 + 4500 + 2500 + 2000 + 6100

-- Define the monthly salary and savings conditions
def salary (S : Real) : Prop :=
  expenses + 0.1 * S = S

-- Define the savings amount
def savings (S : Real) : Real :=
  0.1 * S

-- The theorem to prove
theorem kishore_savings : ∃ S : Real, salary S ∧ savings S = 2733.33 :=
by
  sorry

end kishore_savings_l519_519519


namespace area_of_enclosed_region_eq_12_l519_519181

noncomputable def area_enclosed_by_curves : ℝ :=
  let f1 := (λ x : ℝ, abs (x - 4))
  let f2 := (λ x : ℝ, 5 - abs (x - 2))
  let intersection1 := 1 -- x-coordinate of one intersection point
  let intersection2 := 5 -- x-coordinate of the other intersection point
  let height := 3 -- vertical distance at intersection points
  let base1 := (abs (intersection1 - intersection2))
  let base2 := (abs (intersection2 - intersection1))
  (1 / 2) * (base1 + base2) * height

theorem area_of_enclosed_region_eq_12 :
  area_enclosed_by_curves = 12 :=
sorry

end area_of_enclosed_region_eq_12_l519_519181


namespace range_of_a_l519_519702

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  (a < -1 ∨ a > 0) :=
by
  sorry

end range_of_a_l519_519702


namespace inscribed_polygon_division_l519_519013

-- Define a polygon inscribed in a circle
structure InscribedPolygon (α : Type*) :=
  (vertices : list α)
  (center : α)
  (radius : ℝ)

-- Define a function to check if a line divides the area and perimeter equally
def equal_area_perimeter_division {α : Type*} [metric_space α] (p : InscribedPolygon α) : Prop :=
  ∀ (L : α × α), -- any line passing through the center
    center L = p.center →
    divides_area L p ∧ divides_perimeter L p -- custom predicates to express division

-- Now state the theorem
theorem inscribed_polygon_division {α : Type*} [metric_space α] (p : InscribedPolygon α) :
  equal_area_perimeter_division p :=
by
  sorry 

end inscribed_polygon_division_l519_519013


namespace parquet_tiles_needed_l519_519040

def room_width : ℝ := 8
def room_length : ℝ := 12
def tile_width : ℝ := 1.5
def tile_length : ℝ := 2

def room_area : ℝ := room_width * room_length
def tile_area : ℝ := tile_width * tile_length

def tiles_needed : ℝ := room_area / tile_area

theorem parquet_tiles_needed : tiles_needed = 32 :=
by
  -- sorry to skip the detailed proof
  sorry

end parquet_tiles_needed_l519_519040


namespace assignment_ways_l519_519163

theorem assignment_ways {α : Type} [Fintype α] (A B C D : α) :
    ∀ (class1 class2 class3 : set α), 
    disjoint class1 class2 → 
    disjoint class2 class3 → 
    disjoint class1 class3 → 
    class1 ∪ class2 ∪ class3 = {A, B, C, D} → 
    A ∉ class2 → 
    B ∉ class2 → 
    (class1.nonempty ∧ class2.nonempty ∧ class3.nonempty) →
    (∃ n : ℕ, n = 30) :=
by
  sorry

end assignment_ways_l519_519163


namespace lauren_tuesday_earnings_l519_519348

noncomputable def money_from_commercials (commercials_viewed : ℕ) : ℕ :=
  (1 / 2) * commercials_viewed

noncomputable def money_from_subscriptions (subscribers : ℕ) : ℕ :=
  1 * subscribers

theorem lauren_tuesday_earnings :
  let commercials_viewed := 100 in
  let subscribers := 27 in
  let total_money := money_from_commercials commercials_viewed + money_from_subscriptions subscribers in
  total_money = 77 :=
by 
  sorry

end lauren_tuesday_earnings_l519_519348


namespace nth_equation_sum_l519_519213

theorem nth_equation_sum (n : ℕ) : 
  let terms := List.range (2 * n + 1).map (λ k => n^2 + k) in
  (terms.map (λ k => (Int.ofNat (Real.sqrt k).floor)).sum = 2 * n^2 + n) :=
by
  let terms := List.range (2 * n + 1).map (λ k => n^2 + k)
  show (terms.map (λ k => (Int.ofNat (Real.sqrt k).floor)).sum = 2 * n^2 + n)
  sorry

end nth_equation_sum_l519_519213


namespace find_MorkTaxRate_l519_519004

noncomputable def MorkIncome : ℝ := sorry
noncomputable def MorkTaxRate : ℝ := sorry 
noncomputable def MindyTaxRate : ℝ := 0.30 
noncomputable def MindyIncome : ℝ := 4 * MorkIncome 
noncomputable def combinedTaxRate : ℝ := 0.32 

theorem find_MorkTaxRate :
  (MorkTaxRate * MorkIncome + MindyTaxRate * MindyIncome) / (MorkIncome + MindyIncome) = combinedTaxRate →
  MorkTaxRate = 0.40 := sorry

end find_MorkTaxRate_l519_519004


namespace tensor_correct_l519_519601

-- Given definitions in the problem
def vec (α : Type) := α × α

-- Conditions as Lean definitions
variables {a b : vec ℝ} (theta : ℝ)

noncomputable def modulus (v : vec ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def dot_product (v1 v2 : vec ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def cos_theta (v1 v2 : vec ℝ) : ℝ :=
  (dot_product v1 v2) / ((modulus v1) * (modulus v2))

noncomputable def tensor_product (v1 v2 : vec ℝ) : ℝ :=
  (modulus v1) / (modulus v2) * cos (cos_theta v1 v2)

-- Problem statement translated to Lean
theorem tensor_correct:
  ∀ (a b : vec ℝ), modulus a ≠ 0 ∧ modulus b ≠ 0 ∧
  modulus a ≥ modulus b ∧
  θ ∈ Ioo 0 (real.pi / 4) ∧
  (tensor_product a b ∈ {x | ∃ n : ℕ, x = n / 2}) ∧
  (tensor_product b a ∈ {x | ∃ n : ℕ, x = n / 2}) →
  tensor_product a b = 3 / 2 :=
begin
  sorry
end

end tensor_correct_l519_519601


namespace emma_chocolates_l519_519271

theorem emma_chocolates 
  (x : ℕ) 
  (h1 : ∃ l : ℕ, x = l + 10) 
  (h2 : ∃ l : ℕ, l = x / 3) : 
  x = 15 := 
  sorry

end emma_chocolates_l519_519271


namespace angle_ACE_is_40_l519_519123

-- Define a point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle structure
structure Circle where
  center : Point
  radius : ℝ

-- Define a pentagon structure
structure Pentagon where
  A B C D E : Point

-- Define a property that the pentagon is circumscribed about the circle
def isCircumscribed (p : Pentagon) (c : Circle) : Prop :=
  ∀ (s : Pentagon) (pt ∈ {s.A, s.B, s.C, s.D, s.E}), dist pt c.center = c.radius

-- Define a linear angle
def angle {A B C : Point} : ℝ := sorry

-- Conditions
variable (pentagon : Pentagon)
variable (circle : Circle)
variable (isCircumscribed pentagon circle : Prop)
variable (angleA : angle = 100)
variable (angleC : angle = 100)
variable (angleE : angle = 100)

-- Theorem to prove the angle ACE
theorem angle_ACE_is_40 (pentagon : Pentagon) (circle : Circle) (is_circumscribed : isCircumscribed pentagon circle) 
  (angle_A : angle = 100) (angle_C : angle = 100) (angle_E : angle = 100) : angle = 40 :=
sorry

end angle_ACE_is_40_l519_519123


namespace find_final_painting_width_l519_519466

theorem find_final_painting_width
  (total_area : ℕ)
  (painting_areas : List ℕ)
  (total_paintings : ℕ)
  (last_painting_height : ℕ)
  (last_painting_width : ℕ) :
  total_area = 200
  → painting_areas = [25, 25, 25, 80]
  → total_paintings = 5
  → last_painting_height = 5
  → last_painting_width = 9 :=
by
  intros h_total_area h_painting_areas h_total_paintings h_last_height
  have h1 : 25 * 3 + 80 = 155 := by norm_num
  have h2 : total_area - 155 = last_painting_width * last_painting_height := by
    rw [h_total_area, show 155 = 25 * 3 + 80 by norm_num]
    norm_num
  exact eq_of_mul_eq_mul_right (by norm_num) h2

#print axioms find_final_painting_width -- this should ensure we don't leave any implicit assumptions. 

end find_final_painting_width_l519_519466


namespace area_G1G2G3_l519_519822

noncomputable def equilateral_triangle_area (R : ℝ) : ℝ := (3 * real.sqrt 3 / 4) * (R ^ 2)

noncomputable def centroid_triangle_area (A : ℝ) : ℝ := (1 / 9) * A

theorem area_G1G2G3 (P : Point) (A B C : Triangle) (R : ℝ)
  (h1 : is_point_in_triangle P A B C)
  (h2 : is_centroid G1 (Triangle P B C) and
        is_centroid G2 (Triangle P C A) and
        is_centroid G3 (Triangle P A B))
  (h3 : is_equilateral A B C)
  (h4 : circumscribed_radius A B C = R)
  (hR : R = 6) :
  centroid_triangle_area (equilateral_triangle_area R) = 3 * real.sqrt 3 :=
by
  sorry

end area_G1G2G3_l519_519822


namespace regression_analysis_incorrect_statement_l519_519928

theorem regression_analysis_incorrect_statement
  (y : ℕ → ℝ) (x : ℕ → ℝ) (b a : ℝ)
  (r : ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ)
  (H1 : ∀ i, y i = b * x i + a)
  (H2 : abs r = 1 → ∀ x1 x2, l x1 = l x2 → x1 = x2)
  (H3 : ∃ m k, ∀ x, l x = m * x + k)
  (H4 : P.1 = b → l P.1 = P.2)
  (cond_A : ∀ i, y i ≠ b * x i + a) : false := 
sorry

end regression_analysis_incorrect_statement_l519_519928


namespace price_increase_l519_519475

theorem price_increase (P : ℝ) : 
  let P1 := P * (1 + 0.20),
      P2 := P1 * (1 + 0.15)
  in (P2 - P) / P * 100 = 38 :=
sorry

end price_increase_l519_519475


namespace find_angle_C_find_length_AC_l519_519703

-- Conditions
def cos_A : ℝ := (Math.sqrt 2) / 10
def tan_B : ℝ := 4 / 3
def cos_B : ℝ := 3 / 5
def dot_product_BA_BC : ℝ := 21

-- Proving the angle C
theorem find_angle_C (h1 : cos_A = (Math.sqrt 2) / 10)
                      (h2 : tan_B = 4 / 3) :
  C = Real.pi / 4 :=
by
  sorry

-- Proving the length of AC
theorem find_length_AC (h3 : cos_B = 3 / 5)
                        (h4 : dot_product_BA_BC = 21) :
  AC = 5 * (Math.sqrt 2) :=
by
  sorry

end find_angle_C_find_length_AC_l519_519703


namespace cartesian_conversion_distance_C1_to_C2_l519_519261

-- Defining the parametric equations
def C1_parametric (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sin θ)

def C2_parametric (t : ℝ) : ℝ × ℝ :=
  (-3 + t, (3 + 3 * t) / 8)

-- Cartesian equations derived from the parametric equations
def C1_cartesian (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def C2_cartesian (x y : ℝ) : Prop :=
  3 * x - 8 * y + 12 = 0

-- Distance from a point on C1 to the curve C2
def distance_from_C1_to_C2 (θ : ℝ) : ℝ :=
  let x := 2 * Real.cos θ
  let y := Real.sin θ
  |6 * Real.cos θ - 8 * Real.sin θ + 12| / Real.sqrt 73

-- The proof statements
theorem cartesian_conversion :
    (∀ θ, ∃ x y, x = 2 * Real.cos θ ∧ y = Real.sin θ ∧ C1_cartesian x y) ∧
    (∀ t, ∃ x y, x = -3 + t ∧ y = (3 + 3 * t) / 8 ∧ C2_cartesian x y) :=
  sorry

theorem distance_C1_to_C2 :
    (∀ θ, let d := distance_from_C1_to_C2 θ in (d = (|10 * Real.sin (Real.atan (3/4) - θ) + 12| / Real.sqrt 73)) ∧ 
    d ∈ [2 * Real.sqrt 73 / 73, 22 * Real.sqrt 73 / 73]) :=
  sorry

end cartesian_conversion_distance_C1_to_C2_l519_519261


namespace diff_is_multiple_of_9_l519_519516

-- Definitions
def orig_num (a b : ℕ) : ℕ := 10 * a + b
def new_num (a b : ℕ) : ℕ := 10 * b + a

-- Statement of the mathematical proof problem
theorem diff_is_multiple_of_9 (a b : ℕ) : 
  9 ∣ (new_num a b - orig_num a b) :=
by
  sorry

end diff_is_multiple_of_9_l519_519516


namespace tom_savings_by_having_insurance_l519_519063

noncomputable def insurance_cost_per_month : ℝ := 20
noncomputable def total_months : ℕ := 24
noncomputable def surgery_cost : ℝ := 5000
noncomputable def insurance_coverage_rate : ℝ := 0.80

theorem tom_savings_by_having_insurance :
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  savings = 3520 :=
by
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  sorry

end tom_savings_by_having_insurance_l519_519063


namespace find_x_for_fx_neg_half_l519_519614

open Function 

theorem find_x_for_fx_neg_half (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 2) = -f x)
  (h_interval : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 1/2 * x) :
  {x : ℝ | f x = -1/2} = {x : ℝ | ∃ n : ℤ, x = 4 * n - 1} :=
by
  sorry

end find_x_for_fx_neg_half_l519_519614


namespace simplify_cbrt_expr_l519_519022

-- Define the cube root function.
def cbrt (x : ℝ) : ℝ := x^(1/3)

-- Define the original expression under the cube root.
def original_expr : ℝ := 40^3 + 70^3 + 100^3

-- Define the simplified expression.
def simplified_expr : ℝ := 10 * cbrt 1407

theorem simplify_cbrt_expr : cbrt original_expr = simplified_expr := by
  -- Declaration that proof is not provided to ensure Lean statement is complete.
  sorry

end simplify_cbrt_expr_l519_519022


namespace find_a_and_b_l519_519618

theorem find_a_and_b :
  (∃ (a b : ℝ), ∀ (x : ℝ), (|8 * x + 9| < 7) ↔ (a * x^2 + b * x > 2)) ↔ (a = -4 ∧ b = -9) :=
begin
  sorry
end

end find_a_and_b_l519_519618


namespace width_of_final_painting_l519_519469

theorem width_of_final_painting
  (total_area : ℕ)
  (area_paintings_5x5 : ℕ)
  (num_paintings_5x5 : ℕ)
  (painting_10x8_area : ℕ)
  (final_painting_height : ℕ)
  (total_num_paintings : ℕ := 5)
  (total_area_paintings : ℕ := 200)
  (calculated_area_remaining : ℕ := total_area - (num_paintings_5x5 * area_paintings_5x5 + painting_10x8_area))
  (final_painting_width : ℕ := calculated_area_remaining / final_painting_height) :
  total_num_paintings = 5 →
  total_area = 200 →
  area_paintings_5x5 = 25 →
  num_paintings_5x5 = 3 →
  painting_10x8_area = 80 →
  final_painting_height = 5 →
  final_painting_width = 9 :=
by
  intros h1 h2 h3 h4 h5 h6
  dsimp [final_painting_width, calculated_area_remaining]
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end width_of_final_painting_l519_519469


namespace annual_interest_rate_l519_519005

-- Define the conditions as given in the problem
def principal : ℝ := 5000
def maturity_amount : ℝ := 5080
def interest_tax_rate : ℝ := 0.2

-- Define the annual interest rate x
variable (x : ℝ)

-- Statement to be proved: the annual interest rate x is 0.02
theorem annual_interest_rate :
  principal + principal * x - interest_tax_rate * (principal * x) = maturity_amount → x = 0.02 :=
by
  sorry

end annual_interest_rate_l519_519005


namespace complex_division_l519_519610

noncomputable def imaginary_unit : ℂ := complex.I

theorem complex_division :
  (1 + 2 * imaginary_unit) / (1 + imaginary_unit) = (3 + imaginary_unit) / 2 :=
by
  sorry

end complex_division_l519_519610


namespace num_stickers_l519_519020

-- Variables and assumptions
variables {m n : ℕ} {t : ℚ}

-- Conditions stated as hypotheses
hypothesis h1 : m < n
hypothesis h2 : m > 0
hypothesis h3 : t > 1
hypothesis h4 : m * t + n = 100
hypothesis h5 : m + n * t = 101

-- The statement to prove
theorem num_stickers (m n : ℕ) (t : ℚ) (h1 : m < n) (h2 : m > 0) (h3 : t > 1) (h4 : m * t + n = 100) (h5 : m + n * t = 101) :
  n = 34 ∨ n = 66 :=
sorry

end num_stickers_l519_519020


namespace triangle_area_B_equals_C_tan_ratio_sum_l519_519325

-- First part: Proving the area of the triangle
theorem triangle_area_B_equals_C {A B C a b c : ℝ} (h1 : B = C) (h2 : a = 2) (h3 : b^2 + c^2 = 3 * b * c * cos A) 
    (h4 : B + C = 180) :
    0.5 * b * c * sin A = sqrt(5) := 
by
  sorry

-- Second part: Proving the value of tan A / tan B + tan A / tan C
theorem tan_ratio_sum {A B C a b c : ℝ} (h1 : b^2 + c^2 = 3 * b * c * cos A) (h2 : A + B + C = 180) :
    (tan A / tan B) + (tan A / tan C) = 1 :=
by
  sorry

end triangle_area_B_equals_C_tan_ratio_sum_l519_519325


namespace parallel_vectors_l519_519573

-- Definition of the vectors
def a : ℝ × ℝ × ℝ := (1, -3, 2)
def b : ℝ × ℝ × ℝ := (-1 / 2, 3 / 2, -1)

-- Property of scalar multiplication
def scalar_multiple (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v2 = (k * v1.1, k * v1.2, k * v1.3)

-- Statement of the proof problem
theorem parallel_vectors : scalar_multiple a b := 
  sorry

end parallel_vectors_l519_519573


namespace clock_hands_angle_bisect_l519_519296

-- Conditions from problem
def hour_hand_speed : ℝ := 1 / 720 -- hour hand moves 1/720 of a full circle per second
def minute_hand_speed : ℝ := 1 / 60 -- minute hand moves 1/60 of a full circle per second
def second_hand_speed : ℝ := 1 -- second hand moves 1 full circle per minute

noncomputable def times_angle_bisected_within_one_minute (start_angle_hour : ℝ) (start_angle_minute : ℝ) (start_angle_second : ℝ) : ℕ :=
  4 -- derived from the problem statement solution

-- Proof statement
theorem clock_hands_angle_bisect :
  let start_time := (3, 0, 0) in -- (hours, minutes, seconds) at 3:00:00
  times_angle_bisected_within_one_minute (start_time.1 * 30) (start_time.2 * 6) (start_time.3 * 6) = 4 :=
by
  sorry

end clock_hands_angle_bisect_l519_519296


namespace simplify_expression_l519_519389

def omega : ℂ := (-1 + complex.i * real.sqrt 7) / 2
def omega_star : ℂ := (-1 - complex.i * real.sqrt 7) / 2

theorem simplify_expression : omega^4 + omega_star^4 = 8 :=
by
  sorry

end simplify_expression_l519_519389


namespace largest_n_gon_l519_519076

noncomputable def largest_n : ℕ :=
  26

theorem largest_n_gon (n : ℕ) :
  (∃ (polygon : Type) (interior_angles : polygon → ℕ),
    ∀ i, (interior_angles i) > 0 ∧ 
    interior_angles i < 180 ∧ 
    (∀ j, i ≠ j → interior_angles i ≠ interior_angles j) ∧
    (∑ i, interior_angles i = 180 * (n - 2)) ∧
    convex polygon ∧
    (non-degenerate polygon)) ↔ n ≤ 26 :=
sorry

end largest_n_gon_l519_519076


namespace area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l519_519323

-- Definitions and conditions
variable {A B C a b c : ℝ}
variable (cosA : ℝ) (sinA : ℝ)
variable (area : ℝ)
variable (tanA tanB tanC : ℝ)

-- Given conditions
axiom angle_identity : b^2 + c^2 = 3 * b * c * cosA
axiom sin_cos_identity : sinA^2 + cosA^2 = 1
axiom law_of_cosines : a^2 = b^2 + c^2 - 2 * b * c * cosA

-- Part (1) statement
theorem area_of_triangle_is_sqrt_5 (B_eq_C : B = C) (a_eq_2 : a = 2) 
    (cosA_eq_2_3 : cosA = 2/3) 
    (b_eq_sqrt6 : b = Real.sqrt 6) 
    (sinA_eq_sqrt5_3 : sinA = Real.sqrt 5 / 3) 
    : area = Real.sqrt 5 := sorry

-- Part (2) statement
theorem sum_of_tangents_eq_1 (tanA_eq : tanA = sinA / cosA)
    (tanB_eq : tanB = sinA * sinA / (cosA * cosA))
    (tanC_eq : tanC = sinA * sinA / (cosA * cosA))
    : (tanA / tanB) + (tanA / tanC) = 1 := sorry

end area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l519_519323


namespace quadratic_real_equal_roots_l519_519560

theorem quadratic_real_equal_roots (k : ℝ) :
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 15 = 0 ∧ ∀ y : ℝ, (3 * y^2 - k * y + 2 * y + 15 = 0 → y = x)) ↔
  (k = 6 * Real.sqrt 5 + 2 ∨ k = -6 * Real.sqrt 5 + 2) :=
by
  sorry

end quadratic_real_equal_roots_l519_519560


namespace num_ordered_pairs_l519_519575

theorem num_ordered_pairs (a b : ℂ) (h1 : a^4 * b^6 = 1) (h2 : a^5 * b^3 = 1) :
  (a, b) = (λ n : ℤ, exp(2 * π * I * n / 12), λ n : ℤ, exp(2 * π * I * (7 * n) / 36)) :=
sorry

end num_ordered_pairs_l519_519575


namespace no_values_satisfy_equation_l519_519582

-- Define the sum of the digits function S
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the sum of digits of the sum of the digits function S(S(n))
noncomputable def sum_of_sum_of_digits (n : ℕ) : ℕ :=
  sum_of_digits (sum_of_digits n)

-- Theorem statement about the number of n satisfying n + S(n) + S(S(n)) = 2099
theorem no_values_satisfy_equation :
  (∃ n : ℕ, n > 0 ∧ n + sum_of_digits n + sum_of_sum_of_digits n = 2099) ↔ False := sorry

end no_values_satisfy_equation_l519_519582


namespace g_value_at_50_l519_519116

noncomputable def g : ℝ → ℝ := sorry

theorem g_value_at_50 (g : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, 0 < x → 0 < y → x * g y + y * g x = g (x * y)) :
  g 50 = 0 :=
sorry

end g_value_at_50_l519_519116


namespace shelves_used_l519_519960

-- Definitions from conditions
def initial_bears : ℕ := 6
def shipment_bears : ℕ := 18
def bears_per_shelf : ℕ := 6

-- Theorem statement
theorem shelves_used : (initial_bears + shipment_bears) / bears_per_shelf = 4 := by
  sorry

end shelves_used_l519_519960


namespace part1_part2_l519_519632

def f (x : ℝ) (a : ℝ) := (Real.exp x) / x - Real.log x + x - a

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → (a ≤ Real.exp 1 + 1) :=
sorry

theorem part2 (x1 x2 a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 ≠ x2 → 0 < x1 → 0 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519632


namespace two_people_property_l519_519377

noncomputable def exists_two_people {n : ℕ} (know : ℕ → ℕ → Prop) : Prop :=
  ∃ (A B : ℕ), A ≠ B ∧
  ∃ (subset : Finset ℕ), subset.card = 2 * n ∧
  ∀ (C ∈ subset), (know C A ∧ know C B) ∨ (¬(know C A) ∧ ¬(know C B))

theorem two_people_property (n : ℕ) (know : ℕ → ℕ → Prop) (h : ∀ A B, C, A ≠ B → C ∉ {A, B} →
  (know C A ∧ know C B) ∨ (¬(know C A) ∧ ¬(know C B))) : exists_two_people know :=
by
  sorry

end two_people_property_l519_519377


namespace exists_inhabitant_with_many_acquaintances_l519_519760

open Function

theorem exists_inhabitant_with_many_acquaintances :
  (∃ (n : ℕ), n = 1_000_000) →
  (∀ (p : ℕ), p < 1_000_000 → ∃ (q : ℕ), q ≠ p) →
  (∀ (p : ℕ), p < 900_000 → p <= 900_000) →
  (∀ (p : ℕ), p < 1_000_000 → 0.1 * real.to_real p < 0.9 * real.to_real p) →
  ∃ (p : ℕ), p >= 810 :=
by
  intros h1 h2 h3 h4
  sorry

end exists_inhabitant_with_many_acquaintances_l519_519760


namespace jennas_total_ticket_cost_l519_519803

theorem jennas_total_ticket_cost :
  let normal_price := 50
  let tickets_from_website := 2 * normal_price
  let scalper_price := 2 * normal_price * 2.4 - 10
  let friend_discounted_ticket := normal_price * 0.6
  tickets_from_website + scalper_price + friend_discounted_ticket = 360 :=
by
  sorry

end jennas_total_ticket_cost_l519_519803


namespace quadratic_trinomial_m_eq_2_l519_519051

theorem quadratic_trinomial_m_eq_2 (m : ℤ) (P : |m| = 2 ∧ m + 2 ≠ 0) : m = 2 :=
  sorry

end quadratic_trinomial_m_eq_2_l519_519051


namespace M_gt_N_l519_519240

variables (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
variables (m n : ℝ)
variables (h3 : m^2 * n^2 > a^2 * m^2 + b^2 * n^2)

noncomputable def M := Real.sqrt (m^2 + n^2)
noncomputable def N := a + b

theorem M_gt_N : M a b m n > N a b :=
begin
  sorry
end

end M_gt_N_l519_519240


namespace present_population_l519_519697

theorem present_population (P : ℝ) 
  (annual_increase : P * (1 + 0.10) ^ 2 = 16940) : 
  P = 14000 :=
begin
  sorry
end

end present_population_l519_519697


namespace part1_part2_l519_519634

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l519_519634


namespace Zach_scored_more_l519_519934

theorem Zach_scored_more :
  let Zach := 42
  let Ben := 21
  Zach - Ben = 21 :=
by
  let Zach := 42
  let Ben := 21
  exact rfl

end Zach_scored_more_l519_519934


namespace final_painting_width_l519_519464

theorem final_painting_width (total_area : ℕ) (n_paintings : ℕ) (a1 a2 : ℕ)
  (area1 : ℕ) (area2 : ℕ) (height_final : ℕ) (width_final : ℕ) :
  total_area = 200 →
  n_paintings = 5 →
  a1 = 3 →
  a2 = 1 →
  area1 = 5 * 5 →
  area2 = 10 * 8 →
  height_final = 5 →
  width_final = 9 →
  (a1 * area1 + a2 * area2 + height_final * width_final = total_area) :=
begin
  intros,
  sorry
end

end final_painting_width_l519_519464


namespace enchanted_creatures_gala_handshakes_l519_519058

theorem enchanted_creatures_gala_handshakes :
  let goblins := 30
  let trolls := 20
  let goblin_handshakes := goblins * (goblins - 1) / 2
  let troll_to_goblin_handshakes := trolls * goblins
  goblin_handshakes + troll_to_goblin_handshakes = 1035 := 
by
  sorry

end enchanted_creatures_gala_handshakes_l519_519058


namespace height_of_tower_l519_519029

-- Definitions for points and distances
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := 0, y := 0, z := 0 }
def C : Point := { x := 0, y := 0, z := 129 }
def D : Point := { x := 0, y := 0, z := 258 }
def B : Point  := { x := 0, y := 305, z := 305 }

-- Given conditions
def angle_elevation_A_to_B : ℝ := 45 -- degrees
def angle_elevation_D_to_B : ℝ := 60 -- degrees
def distance_A_to_D : ℝ := 258 -- meters

-- The problem is to prove the height of the tower is 305 meters given the conditions
theorem height_of_tower : B.y = 305 :=
by
  -- This spot would contain the actual proof
  sorry

end height_of_tower_l519_519029


namespace unpainted_area_of_five_inch_board_l519_519070

def width1 : ℝ := 5
def width2 : ℝ := 6
def angle : ℝ := 45

theorem unpainted_area_of_five_inch_board : 
  ∃ (area : ℝ), area = 30 :=
by
  sorry

end unpainted_area_of_five_inch_board_l519_519070


namespace final_painting_width_l519_519463

theorem final_painting_width (total_area : ℕ) (n_paintings : ℕ) (a1 a2 : ℕ)
  (area1 : ℕ) (area2 : ℕ) (height_final : ℕ) (width_final : ℕ) :
  total_area = 200 →
  n_paintings = 5 →
  a1 = 3 →
  a2 = 1 →
  area1 = 5 * 5 →
  area2 = 10 * 8 →
  height_final = 5 →
  width_final = 9 →
  (a1 * area1 + a2 * area2 + height_final * width_final = total_area) :=
begin
  intros,
  sorry
end

end final_painting_width_l519_519463


namespace solution_interval_l519_519557

theorem solution_interval {x : ℝ} (h : x ≥ 1) :
  (sqrt (x + 2 - 5 * sqrt (x - 1)) + sqrt (x + 5 - 7 * sqrt (x - 1)) = 2) ↔ (5 ≤ x ∧ x ≤ 17) :=
sorry

end solution_interval_l519_519557


namespace vector_b_length_l519_519268

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_b_length
  (a b : V)
  (hab : a ≠ 0 ∧ b ≠ 0 ∧ (∀ k : ℝ, a ≠ k • b)) -- non-collinear condition
  (h1 : ∥a - b∥ = 3) -- |a - b| = 3
  (h2 : inner_product_space.inner a (a - 2 • b) = 0) -- a ⊥ (a - 2b)
  : ∥b∥ = 3 := 
by sorry

end vector_b_length_l519_519268


namespace smallest_delicious_integer_l519_519021

def is_delicious (B : ℤ) : Prop :=
  ∃ n : ℕ, ∃ k : ℤ, B = k ∧ (list.sum (list.range (n + 1))) + k * (n + 1) = 2023

theorem smallest_delicious_integer : ∃ B : ℤ, is_delicious B ∧ ∀ B' : ℤ, is_delicious B' → B' ≥ B ∧ B = -2022 :=
by
  sorry

end smallest_delicious_integer_l519_519021


namespace prove_real_sum_inequality_l519_519386

noncomputable def real_sum_inequality (n : ℕ) (a : Fin n → ℝ) : Prop :=
  (∀ i, 0 < a i) ∧ (∑ i, (a i)^2 = 1) → (∑ i, a i) ≤ Real.sqrt n

theorem prove_real_sum_inequality (n : ℕ) (a : Fin n → ℝ):
  real_sum_inequality n a := by
  sorry

end prove_real_sum_inequality_l519_519386


namespace find_b_l519_519083

theorem find_b (h1 : 2.236 = 1 + (b - 1) * 0.618) 
               (h2 : 2.236 = b - (b - 1) * 0.618) : 
               b = 3 ∨ b = 4.236 := 
by
  sorry

end find_b_l519_519083


namespace find_quotient_l519_519370

theorem find_quotient
    (dividend divisor remainder : ℕ)
    (h1 : dividend = 136)
    (h2 : divisor = 15)
    (h3 : remainder = 1)
    (h4 : dividend = divisor * quotient + remainder) :
    quotient = 9 :=
by
  sorry

end find_quotient_l519_519370


namespace coordinates_of_point_p_l519_519944

noncomputable def point_p : ℝ × ℝ :=
  let x := -Real.log 2 in
  let y := Real.exp (-x) in
  (x, y)

theorem coordinates_of_point_p :
  ∀ P : ℝ × ℝ, 
  (P.2 = Real.exp (-P.1)) ∧ 
  (∃ m : ℝ, ∀ x : ℝ, m = -2 ∧ m = -(Real.exp (-P.1))) →
  P = point_p := 
by
  intros P hp ht
  sorry

end coordinates_of_point_p_l519_519944


namespace petya_waits_for_masha_l519_519373

noncomputable def rate_petya := 65 / 60
noncomputable def rate_masha := 52 / 60
noncomputable def target_time := 18.5 -- 6:30 PM in hours

def real_time_elapsed (rate: ℝ) (clock_time: ℝ) : ℝ := clock_time * (60 / rate)

noncomputable def real_time_petya := real_time_elapsed rate_petya target_time
noncomputable def real_time_masha := real_time_elapsed rate_masha target_time

def waiting_time (arrival1: ℝ) (arrival2: ℝ) : ℝ := arrival2 - arrival1

noncomputable def wait_time_for_petya := waiting_time real_time_petya real_time_masha

theorem petya_waits_for_masha : wait_time_for_petya = 4.266666666666667 :=
  by
    sorry

end petya_waits_for_masha_l519_519373


namespace august_five_thursdays_if_july_five_mondays_l519_519399

noncomputable def july_has_five_mondays (N : ℕ) : Prop := ∃ m : ℕ, m ∈ {1, 8, 15, 22, 29, 2, 9, 16, 23, 30, 3, 10, 17, 24, 31} ∧ ∀ i ∈ {1, 8, 15, 22, 29, 2, 9, 16, 23, 30, 3, 10, 17, 24, 31}.erase m, i % 7 = 1

noncomputable def august_has_five_thursdays (N : ℕ) : Prop := ∃ n : ℕ, n % 7 = 3 ∧ ∀ k : ℕ, k < 31 → (k + n) % 7 ≠ 3

theorem august_five_thursdays_if_july_five_mondays (N : ℕ) (h : july_has_five_mondays N):
  august_has_five_thursdays N := 
sorry

end august_five_thursdays_if_july_five_mondays_l519_519399


namespace sum_distances_equal_l519_519854

open Function

variable {AB : Segment}

def is_midpoint (M : Point) (A B : Point) : Prop :=
  dist A M = dist M B

def symmetric (x x' M : Point) : Prop :=
  dist x M = dist x' M

def blue_red_partition (points : List Point) (blue : List Point) (red : List Point) : Prop :=
  points.filter (λ x => x ∈ blue) = blue ∧
  points.filter (λ x => x ∈ red) = red

theorem sum_distances_equal
  (A B M : Point)
  (points : List (Point × Point))
  (h_midpoint : is_midpoint M A B)
  (h_symmetric : ∀ p : Point × Point, p ∈ points → symmetric p.1 p.2 M)
  (blue red : List (Point))
  (h_partition : blue_red_partition ((points.map Prod.fst) ++ (points.map Prod.snd)) blue red)
  (h_count : (List.length blue) = (List.length red)) :
  (List.sum (blue.map (λ p => dist A p))) = (List.sum (red.map (λ p => dist B p))) :=
by
  sorry

end sum_distances_equal_l519_519854


namespace monotonic_decreasing_interval_l519_519889

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1 / x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x > 0 → (∀ a b : ℝ, a < b → (f b ≤ f a → b ≤ (1 : ℝ) / 2)) :=
by sorry

end monotonic_decreasing_interval_l519_519889


namespace constants_exist_l519_519441

theorem constants_exist :
  ∃ (c1 c2 : ℚ), 2 * c1 - 2 * c2 = -1 ∧ 3 * c1 + 5 * c2 = 4 ∧ c1 = 3 / 16 ∧ c2 = 11 / 16 :=
by
  use (3 / 16)
  use (11 / 16)
  split
  {
    calc
      2 * (3 / 16) - 2 * (11 / 16)
      -- Prove the calculations result
      sorry
  }
  split
  {
    calc
      3 * (3 / 16) + 5 * (11 / 16)
      -- Prove the calculations result
      sorry
  }
  split
  {
    -- Directly show c1 = 3 / 16
    sorry
  }
  {
    -- Directly show c2 = 11 / 16
    sorry
  }

end constants_exist_l519_519441


namespace fraction_of_satisfactory_grades_l519_519506

theorem fraction_of_satisfactory_grades :
  let num_A := 6
  let num_B := 5
  let num_C := 4
  let num_D := 3
  let num_E := 2
  let num_F := 6
  -- Total number of satisfactory grades
  let satisfactory := num_A + num_B + num_C + num_D + num_E
  -- Total number of students
  let total := satisfactory + num_F
  -- Fraction of satisfactory grades
  satisfactory / total = (10 : ℚ) / 13 :=
by
  sorry

end fraction_of_satisfactory_grades_l519_519506


namespace find_x0_l519_519220

def f (x : ℝ) : ℝ := x * Real.log x

theorem find_x0 :
  ∃ x0 : ℝ, deriv f x0 = 1 ↔ x0 = 1 :=
begin
  sorry
end

end find_x0_l519_519220


namespace sum_of_selected_numbers_is_260_l519_519307

noncomputable def sum_of_selected_numbers : ℕ :=
  let grid := (List.range 64).map (λ n, n + 1)
  let selected := List.map (λ i, grid[i + 8 * i]) (List.range 8)
  selected.sum

theorem sum_of_selected_numbers_is_260 :
  sum_of_selected_numbers = 260 := by
  sorry

end sum_of_selected_numbers_is_260_l519_519307


namespace find_distance_MN_l519_519786

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos θ, 2 * Real.sin θ)

noncomputable def cartesian_eq_C1 (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4 ∧ 0 ≤ y ∧ y ≤ 2

noncomputable def polar_eq_C2 (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sin θ

noncomputable def line_l (θ line_theta : ℝ) : Prop :=
  line_theta = Real.pi / 4

noncomputable def distance_MN (M N : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)

theorem find_distance_MN :
  (M N : ℝ × ℝ)
  (θ := Real.pi / 4)
  (ρ := 2 * Real.sqrt 2)
  (ρ1 := 2 * Real.cos θ)
  (ρ2 := 2 * Real.sin θ)
  (M = (ρ1, θ))
  (N = (ρ2, θ))
  (abs_dist := abs ((2 * Real.sqrt 2) - Real.sqrt 2)) :
  abs_dist = Real.sqrt 2 :=
sorry

end find_distance_MN_l519_519786


namespace cube_volume_in_pyramid_l519_519957

theorem cube_volume_in_pyramid : 
  let a := 2        -- side length of pyramid base
  let h := sqrt 6   -- height of the pyramid
  let s := h / 3    -- edge length of the cube
  let V := s^3      -- volume of the cube
  V = 2 * sqrt 6 / 9 :=          -- the required volume
by sorry

end cube_volume_in_pyramid_l519_519957


namespace midpoints_on_circle_E_iff_l519_519355

-- Given two acute-angled triangles ABC and XYZ such that
-- the points A, B, C, X, Y, Z are concyclic.
variables {A B C X Y Z : Type}
variables [acute_triangle ABC]
variables [acute_triangle XYZ]
-- Assuming they are concyclic
axiom concyclic : ∀ {u v : Type}, is_concyclic [u, v, A, B, C, X, Y, Z]

-- Midpoints of segments [BC], [CA], [AB], [YZ], [ZX], [XY]
noncomputable def M_BC := midpoint B C
noncomputable def M_CA := midpoint C A
noncomputable def M_AB := midpoint A B
noncomputable def M_YZ := midpoint Y Z
noncomputable def M_ZX := midpoint Z X
noncomputable def M_XY := midpoint X Y

-- Given that the midpoints M_{BC}, M_{CA}, M_{AB}, M_{YZ} lie on the same circle \mathcal{E}.
axiom circle_E : is_on_circle [M_BC, M_CA, M_AB, M_YZ, E]

-- To be proved: M_{ZX} lies on \mathcal{E} if and only if M_{XY} lies on \mathcal{E}.
theorem midpoints_on_circle_E_iff (M_ZX_on_E : is_on_circle [M_ZX, E]) :
  is_on_circle [M_XY, E] ↔ is_on_circle [M_ZX, E] :=
sorry

end midpoints_on_circle_E_iff_l519_519355


namespace limit_sequence_zero_l519_519978

open_locale topological_space

noncomputable def sequence (n : ℕ) : ℝ :=
  (n^3 - (n - 1)^3) / ((n + 1)^4 - n^4)

theorem limit_sequence_zero : 
  tendsto (λ n, sequence n) at_top (𝓝 0) :=
sorry

end limit_sequence_zero_l519_519978


namespace factorial_division_l519_519180

theorem factorial_division (h8 : ℕ) (h9 : ℕ) (h10 : ℕ) (h6 : ℕ) : 
  (h8 = 8! ∧ h9 = 9! ∧ h10 = 10! ∧ h6 = 6!) → ( (h8 + h9 - h10) / h6 = -4480 ) := 
by
  -- Import necessary factorial definitions

  -- Assume values of factorials
  assume h : h8 = 8! ∧ h9 = 9! ∧ h10 = 10! ∧ h6 = 6!

  -- Result to be proven
  sorry

end factorial_division_l519_519180


namespace smallest_p_is_2_l519_519904

-- Define primes and conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d, d ∣ n → d = 1 ∨ d = n)

noncomputable def smallest_p : ℕ :=
  if h : ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ p + q = r ∧ 1 < p ∧ p < q ∧ r > 10
  then Classical.choose h
  else 0

theorem smallest_p_is_2 : smallest_p = 2 := by
  sorry -- Proof not required

end smallest_p_is_2_l519_519904


namespace radius_of_given_spherical_coords_l519_519052

def radius_of_circle_formed_by_spherical_coords (rho theta phi : ℝ) : ℝ :=
  if h : rho = 2 ∧ phi = π / 4 then
    √2
  else
    0

theorem radius_of_given_spherical_coords {theta : ℝ} :
  radius_of_circle_formed_by_spherical_coords 2 theta (π / 4) = √2 :=
by
  intro
  simp [radius_of_circle_formed_by_spherical_coords]
  split_ifs
  case inl h₀ => exact h₀.right.symm.trans √_root .. sorry -- proof skeleton
  case inr h₀ => contradiction

end radius_of_given_spherical_coords_l519_519052


namespace amber_bronze_cells_selection_l519_519840

theorem amber_bronze_cells_selection (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b)
  (cells : Fin (a + b + 1) → Fin (a + b + 1) → Bool)
  (amber : ∀ i j, cells i j = tt → Bool) -- True means amber, False means bronze
  (bronze : ∀ i j, cells i j = ff → Bool)
  (amber_count : (Finset.univ.product Finset.univ).filter (λ p, cells p.1 p.2 = tt).card ≥ a^2 + a * b - b)
  (bronze_count : (Finset.univ.product Finset.univ).filter (λ p, cells p.1 p.2 = ff).card ≥ b^2 + b * a - a) :
  ∃ (α : Finset (Fin (a + b + 1) × Fin (a + b + 1))), 
    (α.filter (λ p, cells p.1 p.2 = tt)).card = a ∧ 
    (α.filter (λ p, cells p.1 p.2 = ff)).card = b ∧ 
    (∀ (p1 p2 : Fin (a + b + 1) × Fin (a + b + 1)), p1 ≠ p2 → (p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) :=
by sorry

end amber_bronze_cells_selection_l519_519840


namespace larger_square_side_length_l519_519845

theorem larger_square_side_length (x y H : ℝ) 
  (smaller_square_perimeter : 4 * x = H - 20)
  (larger_square_perimeter : 4 * y = H) :
  y = x + 5 :=
by
  sorry

end larger_square_side_length_l519_519845


namespace solve_expression_l519_519098

noncomputable def given_expression : ℝ :=
  (10 * 1.8 - 2 * 1.5) / 0.3 + 3^(2 / 3) - Real.log 4 + Real.sin (Real.pi / 6) - Real.cos (Real.pi / 4) + Nat.factorial 4 / Nat.factorial 2

theorem solve_expression : given_expression = 59.6862 :=
by
  sorry

end solve_expression_l519_519098


namespace probability_is_193_over_1450_l519_519834

noncomputable def prob_calculation : ℝ :=
  let interval_1 := set.Icc 196 225 in
  let interval_2 := set.Icc (105^2 / 50) (106^2 / 50) in
  let intersect_interval := interval_1 ∩ interval_2 in
  let successful_length := (224.36 - 220.5 : ℝ) in
  let initial_length := (225 - 196 : ℝ) in
  successful_length / initial_length

theorem probability_is_193_over_1450 :
  prob_calculation = 193 / 1450 :=
by
  sorry

end probability_is_193_over_1450_l519_519834


namespace lisa_spent_on_tshirts_l519_519847

theorem lisa_spent_on_tshirts :
  ∃ T : ℝ, 
    let lisa_tshirts := T in
    let lisa_jeans := (1/2) * T in
    let lisa_coats := 2 * T in
    let carly_tshirts := (1/4) * T in
    let carly_jeans := (3/2) * (1/2) * T in
    let carly_coats := (1/4) * 2 * T in
    lisa_tshirts + lisa_jeans + lisa_coats + carly_tshirts + carly_jeans + carly_coats = 230 ∧
    T ≈ 48.42 :=
by
  sorry

end lisa_spent_on_tshirts_l519_519847


namespace percentage_profit_is_25_l519_519332

/-- The selling price of the statue. -/
def sale_price : ℝ := 620

/-- The original cost of the statue. -/
def original_cost : ℝ := 496

/-- The profit made from the sale. -/
def profit : ℝ := sale_price - original_cost

/-- The percentage of profit made. -/
def percentage_profit : ℝ := (profit / original_cost) * 100

/-- The main theorem stating that the percentage profit is 25%. -/
theorem percentage_profit_is_25 : percentage_profit = 25 := by
  sorry

end percentage_profit_is_25_l519_519332


namespace exists_inhabitant_with_many_acquaintances_l519_519762

open Function

theorem exists_inhabitant_with_many_acquaintances :
  (∃ (n : ℕ), n = 1_000_000) →
  (∀ (p : ℕ), p < 1_000_000 → ∃ (q : ℕ), q ≠ p) →
  (∀ (p : ℕ), p < 900_000 → p <= 900_000) →
  (∀ (p : ℕ), p < 1_000_000 → 0.1 * real.to_real p < 0.9 * real.to_real p) →
  ∃ (p : ℕ), p >= 810 :=
by
  intros h1 h2 h3 h4
  sorry

end exists_inhabitant_with_many_acquaintances_l519_519762


namespace trigonometric_identity_l519_519242

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.cos (A / 2)) ^ 2 = (Real.cos (B / 2)) ^ 2 + (Real.cos (C / 2)) ^ 2 - 2 * (Real.cos (B / 2)) * (Real.cos (C / 2)) * (Real.sin (A / 2)) :=
sorry

end trigonometric_identity_l519_519242


namespace selection_methods_l519_519118

theorem selection_methods (students : ℕ) (boys : ℕ) (girls : ℕ) (selected : ℕ) (h1 : students = 8) (h2 : boys = 6) (h3 : girls = 2) (h4 : selected = 4) : 
  ∃ methods, methods = 40 :=
by
  have h5 : students = boys + girls := by linarith
  sorry

end selection_methods_l519_519118


namespace measure_angle_A_l519_519327

open Real

-- Define a structure for a triangle
structure Triangle :=
( A B C : Point )

-- Define bisectors and angle measurement condition
def angle_bisector (P Q R S : Point) : Prop := 
 ∀ (θ : ℝ) (hPQR : ∠ P Q R = θ) (hQRS : ∠ Q R S = θ), True

noncomputable def measure_angle {P Q R : Point} (A : Point) : ℝ := sorry

theorem measure_angle_A {A B C D E : Point} 
  (ABC : Triangle)
  (hAD : angle_bisector A D B C)
  (hBE : angle_bisector B E A C)
  (hDE : angle_bisector D E A C) : 
  measure_angle A B C = 120 :=
sorry

end measure_angle_A_l519_519327


namespace variance_of_X_l519_519443

-- Defining the probability of the outcome for a random variable X
def probability_X_eq_1 : ℚ := 1 / 6
def probability_X_eq_0 : ℚ := 1 - probability_X_eq_1

-- Defining the expected value of X
def expected_value_X : ℚ := 1 * probability_X_eq_1 + 0 * probability_X_eq_0

-- Proving the variance of X
theorem variance_of_X : 
  let D_X := probability_X_eq_1 * (1 - expected_value_X)^2 + probability_X_eq_0 * (0 - expected_value_X)^2
  in D_X = 5 / 36 := by
  simp [probability_X_eq_1, probability_X_eq_0, expected_value_X]
  sorry

end variance_of_X_l519_519443


namespace marta_candies_received_l519_519002

theorem marta_candies_received:
  ∃ x y : ℕ, x + y = 200 ∧ x < 100 ∧ x > (4 * y) / 5 ∧ (x % 8 = 0) ∧ (y % 8 = 0) ∧ x = 96 ∧ y = 104 := 
sorry

end marta_candies_received_l519_519002


namespace no_positive_integer_makes_sum_prime_l519_519195

theorem no_positive_integer_makes_sum_prime : ¬ ∃ n : ℕ, 0 < n ∧ Prime (4^n + n^4) :=
by
  sorry

end no_positive_integer_makes_sum_prime_l519_519195


namespace inhabitant_knows_at_least_810_l519_519746

open Function

def population : ℕ := 1000000

def believers_claus (p : ℕ) : Prop := p = (90 * population / 100)

def knows_someone (p : ℕ) (knows : p → p → Prop) : Prop := 
  ∀ x : p, ∃ y : p, knows x y

def acquaintances_claus (p : ℕ) (believes : p → Prop) (knows : p → p → Prop) : Prop :=
  ∀ x : p, (believes x →(∃ k : ℕ, ∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y ∧ (10 * k / 100) y ∧ ¬believes y))

theorem inhabitant_knows_at_least_810 :
  ∃ (p : ℕ) (believes : p → Prop) (knows : p → p → Prop),
    population = 1000000 ∧
    believers_claus p ∧
    knows_someone p knows ∧
    acquaintances_claus p believes knows →
    ∃ x : p, ∃ k : ℕ, k ≥ 810 ∧ (∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y) :=
sorry

end inhabitant_knows_at_least_810_l519_519746


namespace triangle_inequality_l519_519354

variables {A B C M : Type} [MetricSpace M]

def is_triangle (A B C : M) : Prop := 
∃ (ABC : Set M), {A, B, C} ⊆ ABC ∧ Convex ABC

def is_interior_point (M : M) (A B C : M) : Prop := 
∃ (ABC : Set M), is_triangle A B C ∧ ConvexHull ABC M

theorem triangle_inequality (A B C M : M)
  (h_triangle : is_triangle A B C)
  (h_interior : is_interior_point M A B C) :
  min (dist M A) (min (dist M B) (dist M C)) + dist M A + dist M B + dist M C < dist A B + dist A C + dist B C :=
sorry

end triangle_inequality_l519_519354


namespace lauren_mail_total_l519_519339

theorem lauren_mail_total : 
  let monday := 65
  let tuesday := monday + 10
  let wednesday := tuesday - 5
  let thursday := wednesday + 15
  monday + tuesday + wednesday + thursday = 295 :=
by
  have monday := 65
  have tuesday := monday + 10
  have wednesday := tuesday - 5
  have thursday := wednesday + 15
  calc
    monday + tuesday + wednesday + thursday 
    = 65 + (65 + 10) + (65 + 10 - 5) + (65 + 10 - 5 + 15) : by rfl
    ... = 65 + 75 + 70 + 85 : by rfl
    ... = 295 : by rfl

end lauren_mail_total_l519_519339


namespace acquaintance_paradox_proof_l519_519708

theorem acquaintance_paradox_proof :
  ∃ (x : ℕ), (acquaintances x ≥ 810) :=
by
  let n_population := 1000000
  let n_believers := 900000
  let n_nonBelievers := n_population - n_believers
  have h1 : n_population = 1000000 := rfl
  have h2 : n_believers = 0.90 * n_population := by sorry
  have h3 : n_nonBelievers = n_population - n_believers := by sorry
  have h4 : ∀ x, acquaintances x ≥ 1 := by sorry
  have h5 : ∀ x, 0.10 * acquaintances x = nonBelievers_acquaintances x := by sorry
  have h6 : nonBelievers_acquaintances_total >= 90 := by sorry
  
  show ∃ (x : ℕ), acquaintances x ≥ 810 from sorry
  
  sorry

end acquaintance_paradox_proof_l519_519708


namespace second_part_of_ratio_l519_519109

theorem second_part_of_ratio (first_part : ℝ) (whole second_part : ℝ) (h1 : first_part = 5) (h2 : first_part / whole = 25 / 100) : second_part = 15 :=
by
  sorry

end second_part_of_ratio_l519_519109


namespace irwin_basketball_l519_519801

theorem irwin_basketball (A B C D : ℕ) (h1 : C = 2) (h2 : 2^A * 5^B * 11^C * 13^D = 2420) : A = 2 :=
by
  sorry

end irwin_basketball_l519_519801


namespace inverse_variation_proof_l519_519403

variable (x w : ℝ)

-- Given conditions
def varies_inversely (k : ℝ) : Prop :=
  x^4 * w^(1/4) = k

-- Specific instances
def specific_instance1 : Prop :=
  varies_inversely x w 162 ∧ x = 3 ∧ w = 16

def specific_instance2 : Prop :=
  varies_inversely x w 162 ∧ x = 6 → w = 1/4096

theorem inverse_variation_proof : 
  specific_instance1 → specific_instance2 :=
sorry

end inverse_variation_proof_l519_519403


namespace problem_proof_l519_519350

noncomputable def given_problem : ℝ := 2018 -- Given radius of the circle ω
noncomputable def max_value_OX : ℝ := 2018 * Real.sqrt 3 / 2 -- The maximum value of OX 

theorem problem_proof :
  (∃ R r : ℝ, R = 2018 ∧ 0 < r ∧ r < 1009 ∧ 
  (∃ I : Point, dist I O = Real.sqrt (R * (R - 2 * r)) ∧
  (∃ (A B C : Point) (E F : Point), 
  tangent A C γ ∧ tangent A B γ ∧
  (∃ D : Point, tangent_circle_to_AB_AC ω 5 * r D ∧
  (∃ (P1 Q1 : Point) (P2 P3 Q2 Q3 : Point), 
  meet_line_circle EF ω P1 Q1 ∧ 
  on_circle P2 P3 Q2 Q3 ω ∧
  tangent P1 P2 γ ∧ tangent P1 P3 γ ∧ tangent Q1 Q2 γ ∧ tangent Q1 Q3 γ ∧
  (∃ K : Point, P2P3_meet_Q2Q3 K ∧
  (∃ X : Point, KI_meet_AD_at_X K I AD X ∧ 
  OX_in_form a b c ∧  a = 2018 ∧ b = 3 ∧ c = 2)))))
  → 10 * a + b + c = 20185 := 
sorry

end problem_proof_l519_519350


namespace negation_of_universal_statement_l519_519265

theorem negation_of_universal_statement:
  (∀ x : ℝ, x ≥ 2) ↔ ¬ (∃ x : ℝ, x < 2) :=
by {
  sorry
}

end negation_of_universal_statement_l519_519265


namespace solve_quadratic_inequality_l519_519024

open Set Real

noncomputable def quadratic_inequality (x : ℝ) : Prop := -9 * x^2 + 6 * x + 8 > 0

theorem solve_quadratic_inequality :
  {x : ℝ | -9 * x^2 + 6 * x + 8 > 0} = {x : ℝ | -2/3 < x ∧ x < 4/3} :=
by
  sorry

end solve_quadratic_inequality_l519_519024


namespace sequence_num_terms_l519_519434

theorem sequence_num_terms :
  let seq := list.iterate (fun x => x / 3) 12150 5
  ∀ n, n < 5 → ∃ k, seq.n = 12150 / (3^k) := by
sorry

end sequence_num_terms_l519_519434


namespace propositions_correct_l519_519622

theorem propositions_correct :
  let P1 := ∀ x, (x > 0 → ∀ k < 0, (f x = k / x) → f x increases) ∧ (x < 0 → ∀ k < 0, (f x = k / x) → f x increases) →
             f is increasing
  let P2 := ∀ (f : ℝ → ℝ), (∃ k, ∀ x, f x = k * x) → (f 0 = 0)
  let P3 := ∀ (f : ℝ → ℝ), (domain f = set.Icc 0 2) → domain (λ x, f (2 * x)) = set.Icc 0 1
  let P4 := ∀ x, (x >= 1 → (deriv (λ x, x^2 - 2 * abs x - 3) x > 0)) →
                 (x < 1 ∧ x > -1 → (deriv (λ x, x^2 - 2 * abs x - 3) x < 0)) →
             increasing_interval (λ x, x^2 - 2 * abs x - 3) = set.Ici 1
  (P1 → False) ∧ P2 ∧ (P3 → False) ∧ (P4 → False) :=
by
  let P1 := sorry
  let P2 := sorry
  let P3 := sorry
  let P4 := sorry
  exact and.intro (λ h, sorry) (and.intro (sorry) (and.intro (λ h, sorry) (λ h, sorry)))

end propositions_correct_l519_519622


namespace part1_part2_l519_519649

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519649


namespace no_prime_divisible_by_77_l519_519281

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → is_divisible p 77 → false :=
by
  sorry

end no_prime_divisible_by_77_l519_519281


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519733

theorem exists_inhabitant_with_at_least_810_acquaintances
  (N : ℕ) (hN : N = 1000000)
  (B : Finset ℕ) (hB : B.card = 900000)
  (A : ℕ → Finset ℕ)
  (hA_nonempty : ∀ i, (A i).nonempty)
  (hA_believers : ∀ i, (A i).card / 10 = (A i ∩ B).card) :
  ∃ j, (A j).card ≥ 810 :=
by
  sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519733


namespace number_of_20_paise_coins_l519_519940

theorem number_of_20_paise_coins (x y : ℕ) (h1 : x + y = 336) (h2 : (20 / 100 : ℚ) * x + (25 / 100 : ℚ) * y = 71) :
    x = 260 :=
by
  sorry

end number_of_20_paise_coins_l519_519940


namespace dodecagon_pyramid_faces_l519_519903

/-- A pyramid with a dodecagon base has 13 faces. -/
theorem dodecagon_pyramid_faces : 
  ∀ (n : ℕ), (n = 12) → (n + 1 = 13) := 
by 
  intros n h 
  rw h 
  norm_num 
  done

end dodecagon_pyramid_faces_l519_519903


namespace part1_part2_l519_519652

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519652


namespace max_kings_on_12x12_board_each_attacks_one_other_l519_519456

theorem max_kings_on_12x12_board_each_attacks_one_other :
  let n := 12 in
  let vertices := (n + 1) * (n + 1) in
  -- conditions
  let pairs_vertices_marked := 6 in
  -- proof
  ∃ max_kings, max_kings = 2 * (vertices / pairs_vertices_marked).floor ∧ max_kings = 56 :=
by
  sorry

end max_kings_on_12x12_board_each_attacks_one_other_l519_519456


namespace part1_part2_l519_519653

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519653


namespace hyperbola_equation_ellipse_equation_l519_519483

-- Problem 1
theorem hyperbola_equation (x y : ℝ) : 
  (let shared_foci := (foci of the ellipse) and
  let ell_eq := (x^2 / 27) + (y^2 / 36) = 1 in 
  let point := (sqrt 15, 4)) :=
  ((shared_foci and ell_eq) = (shared_foci and (y^2 / 4 - x^2 / 5 = 1))
   := by sorry

-- Problem 2
theorem ellipse_equation (x y : ℝ) :
  (let major_axis := 2 * minor_axis in
  let point_A := (2, 0)) :=
  (major_axis and point_A) := 
  ((x^2 / 4 + y^2 = 1) ∨ (x^2 / 4 + y^2 / 16 = 1))
   := by sorry

end hyperbola_equation_ellipse_equation_l519_519483


namespace tetrahedron_volume_from_pentagon_l519_519229

noncomputable def volume_of_tetrahedron (side_length : ℝ) (diagonal_length : ℝ) (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem tetrahedron_volume_from_pentagon :
  ∀ (s : ℝ), s = 1 →
  volume_of_tetrahedron s ((1 + Real.sqrt 5) / 2) ((Real.sqrt 3) / 4) (Real.sqrt ((5 + 2 * Real.sqrt 5) / 4)) =
  (1 + Real.sqrt 5) / 24 :=
by
  intros s hs
  rw [hs]
  sorry

end tetrahedron_volume_from_pentagon_l519_519229


namespace childrens_ticket_cost_l519_519424

theorem childrens_ticket_cost (total_tickets : ℕ) (adult_ticket_cost : ℕ) 
                             (total_receipts : ℕ) (adult_tickets_sold : ℕ) :
  total_tickets = 522 → 
  adult_ticket_cost = 15 →
  total_receipts = 5086 → 
  adult_tickets_sold = 130 →
  (let childrens_ticket_cost := (total_receipts - (adult_tickets_sold * adult_ticket_cost)) / (total_tickets - adult_tickets_sold)
   in childrens_ticket_cost = 8) :=
by
  intros h1 h2 h3 h4
  sorry

end childrens_ticket_cost_l519_519424


namespace proj_distances_l519_519224

theorem proj_distances 
  (r : ℤ) (u v : ℤ) (p q : ℤ) (m n : ℕ)
  (h_circle_eq : u^2 + v^2 = r^2)
  (h_odd_r : r % 2 = 1)
  (h_u_gt_v : u > v)
  (h_u_prime : ∃ k, u = p^k ∧ Nat.prime p)
  (h_v_prime : ∃ l, v = q^l ∧ Nat.prime q)
  (h_m_nat : 0 < m) (h_n_nat : 0 < n)
  (P : (ℤ × ℤ)) (hP : P = (u, v))
  (A B C D M N : (ℤ × ℤ))
  (hA : A = (r, 0)) (hB : B = (-r, 0))
  (hC : C = (0, -r)) (hD : D = (0, r))
  (hM : M = (u, 0)) (hN : N = (0, v)) :
  |A.1 - M.1| = 1 ∧ |B.1 - M.1| = 9 ∧ |C.2 - N.2| = 8 ∧ |D.2 - N.2| = 2 :=
by sorry

end proj_distances_l519_519224


namespace sum_of_odd_power_coefficients_l519_519245

theorem sum_of_odd_power_coefficients (m n : ℕ) (h1 : m + 2 * n = 11) 
  (h2 : m = 5) (h3 : n = 3) : 
  ∑ i in (finset.range (m + n + 1)).filter (λ i, i % 2 = 1), 
  (binomial m i + binomial n i) = 30 := by 
sorry

end sum_of_odd_power_coefficients_l519_519245


namespace smallest_possible_value_l519_519827

theorem smallest_possible_value (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (b + d)) + (1 / (c + d))) ≥ 8 := 
sorry

end smallest_possible_value_l519_519827


namespace lauren_mail_total_l519_519341

theorem lauren_mail_total : 
  let monday := 65
  let tuesday := monday + 10
  let wednesday := tuesday - 5
  let thursday := wednesday + 15
  monday + tuesday + wednesday + thursday = 295 :=
by
  have monday := 65
  have tuesday := monday + 10
  have wednesday := tuesday - 5
  have thursday := wednesday + 15
  calc
    monday + tuesday + wednesday + thursday 
    = 65 + (65 + 10) + (65 + 10 - 5) + (65 + 10 - 5 + 15) : by rfl
    ... = 65 + 75 + 70 + 85 : by rfl
    ... = 295 : by rfl

end lauren_mail_total_l519_519341


namespace distance_MN_intersection_l519_519788

noncomputable def curve_c1_parametric (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos θ, 2 * Real.sin θ)

def curve_c2_polar (θ : ℝ) : ℝ :=
  2 * Real.sin θ

def line_theta (θ : ℝ) : Prop :=
  θ = Real.pi / 4

theorem distance_MN_intersection :
  ∀ θ ∈ Set.Icc 0 Real.pi,
  line_theta θ →
  let M := curve_c1_parametric θ
  let N := (curve_c2_polar θ * Real.cos θ, curve_c2_polar θ * Real.sin θ)
  Real.dist M N = Real.sqrt 2 :=
by
  intros θ hθ hθ_line M N
  sorry

end distance_MN_intersection_l519_519788


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519734

theorem exists_inhabitant_with_at_least_810_acquaintances
  (N : ℕ) (hN : N = 1000000)
  (B : Finset ℕ) (hB : B.card = 900000)
  (A : ℕ → Finset ℕ)
  (hA_nonempty : ∀ i, (A i).nonempty)
  (hA_believers : ∀ i, (A i).card / 10 = (A i ∩ B).card) :
  ∃ j, (A j).card ≥ 810 :=
by
  sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519734


namespace part1_part2_l519_519234

open Set Real

def A : Set ℝ := { x | x^2 - 4 * x - 5 ≤ 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - 2 * x - m < 0 }

theorem part1 (m : ℝ) (h_m : m = 3) : 
  (A \cap (compl (B m))) = {x | x = -1 ∨ 3 ≤ x ∧ x ≤ 5} :=
by
  rw h_m
  sorry

theorem part2 (h_eq : A \cap B m = {x | -1 ≤ x ∧ x < 4}) : m = 8 :=
by
  sorry

end part1_part2_l519_519234


namespace inhabitant_knows_at_least_810_l519_519747

open Function

def population : ℕ := 1000000

def believers_claus (p : ℕ) : Prop := p = (90 * population / 100)

def knows_someone (p : ℕ) (knows : p → p → Prop) : Prop := 
  ∀ x : p, ∃ y : p, knows x y

def acquaintances_claus (p : ℕ) (believes : p → Prop) (knows : p → p → Prop) : Prop :=
  ∀ x : p, (believes x →(∃ k : ℕ, ∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y ∧ (10 * k / 100) y ∧ ¬believes y))

theorem inhabitant_knows_at_least_810 :
  ∃ (p : ℕ) (believes : p → Prop) (knows : p → p → Prop),
    population = 1000000 ∧
    believers_claus p ∧
    knows_someone p knows ∧
    acquaintances_claus p believes knows →
    ∃ x : p, ∃ k : ℕ, k ≥ 810 ∧ (∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y) :=
sorry

end inhabitant_knows_at_least_810_l519_519747


namespace P_positive_l519_519597

variable (P : ℕ → ℝ)

axiom P_cond_0 : P 0 > 0
axiom P_cond_1 : P 1 > P 0
axiom P_cond_2 : P 2 > 2 * P 1 - P 0
axiom P_cond_3 : P 3 > 3 * P 2 - 3 * P 1 + P 0
axiom P_cond_n : ∀ n, P (n + 4) > 4 * P (n + 3) - 6 * P (n + 2) + 4 * P (n + 1) - P n

theorem P_positive (n : ℕ) (h : n > 0) : P n > 0 := by
  sorry

end P_positive_l519_519597


namespace problem_sequence_sum_l519_519892

theorem problem_sequence_sum (S : ℕ+ → ℤ) (a : ℕ+ → ℤ) (h : ∀ n : ℕ+, S n = 2 * n - 1) : 
  a 2018 = 2 :=
by
  have h1 : S 2018 = 2 * 2018 - 1 := h 2018,
  have h2 : S 2017 = 2 * 2017 - 1 := h 2017,
  have h3 : a 2018 = S 2018 - S 2017 := sorry,  -- this corresponds to: a_n = S_n - S_(n-1)
  rw [h1, h2] at h3,
  exact h3

end problem_sequence_sum_l519_519892


namespace fraction_of_water_is_half_l519_519494

-- Definitions based on given conditions
def total_weight : ℝ := 48
def weight_sand : ℝ := (1/3) * total_weight
def weight_gravel : ℝ := 8

-- Define the weight of water
def weight_water : ℝ := total_weight - (weight_sand + weight_gravel)

-- Define the fraction of the mixture that is water
def fraction_water : ℝ := weight_water / total_weight

-- Statement of the theorem
theorem fraction_of_water_is_half : fraction_water = 1/2 := by
  sorry

end fraction_of_water_is_half_l519_519494


namespace moe_mowing_time_l519_519368

noncomputable def effective_swath_width_inches : ℝ := 30 - 6
noncomputable def effective_swath_width_feet : ℝ := (effective_swath_width_inches / 12)
noncomputable def lawn_width : ℝ := 180
noncomputable def lawn_length : ℝ := 120
noncomputable def walking_rate : ℝ := 4500
noncomputable def total_strips : ℝ := lawn_width / effective_swath_width_feet
noncomputable def total_distance : ℝ := total_strips * lawn_length
noncomputable def time_required : ℝ := total_distance / walking_rate

theorem moe_mowing_time :
  time_required = 2.4 := by
  sorry

end moe_mowing_time_l519_519368


namespace hyperbola_eccentricity_proof_l519_519237

noncomputable def hyperbola_eccentricity (a b c : ℝ) (O F1 F2 P : ℝ × ℝ)
  (hyp : a > 0 ∧ b > 0 ∧ c^2 = a^2 + b^2 ∧
    (F1 = (-c, 0)) ∧ (F2 = (c, 0)) ∧
    (∃ P : ℝ × ℝ, (P.1 / a)^2 - (P.2 / b)^2 = 1 ∧ 
       ∥O - P∥ = 3 * b ∧
       angle F1 P F2 = π / 3)) : Prop :=
  let e := c / a in 
  e = √42 / 6

theorem hyperbola_eccentricity_proof :
  ∀ (a b c : ℝ) (O F1 F2 P : ℝ × ℝ),
  hyperbola_eccentricity a b c O F1 F2 P (by
    repeat {split};
    sorry) :=
by sorry

end hyperbola_eccentricity_proof_l519_519237


namespace inhabitant_knows_at_least_810_l519_519741

open Function

def population : ℕ := 1000000

def believers_claus (p : ℕ) : Prop := p = (90 * population / 100)

def knows_someone (p : ℕ) (knows : p → p → Prop) : Prop := 
  ∀ x : p, ∃ y : p, knows x y

def acquaintances_claus (p : ℕ) (believes : p → Prop) (knows : p → p → Prop) : Prop :=
  ∀ x : p, (believes x →(∃ k : ℕ, ∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y ∧ (10 * k / 100) y ∧ ¬believes y))

theorem inhabitant_knows_at_least_810 :
  ∃ (p : ℕ) (believes : p → Prop) (knows : p → p → Prop),
    population = 1000000 ∧
    believers_claus p ∧
    knows_someone p knows ∧
    acquaintances_claus p believes knows →
    ∃ x : p, ∃ k : ℕ, k ≥ 810 ∧ (∃ S : finset p, S.card = k ∧ ∀ y ∈ S, knows x y) :=
sorry

end inhabitant_knows_at_least_810_l519_519741


namespace shifted_sine_symmetry_l519_519865

theorem shifted_sine_symmetry (x : ℝ) : 
  let f := λ x, Real.sin (x + Real.pi / 2)
  ∃ p : ℝ × ℝ, p = (- Real.pi / 2, 0) ∧ SymmetricAbout (f, p) := sorry

end shifted_sine_symmetry_l519_519865


namespace minute_hand_position_l519_519165

noncomputable def degrees_per_minute_hour_hand : ℝ := 0.5
noncomputable def degrees_per_minute_minute_hand : ℝ := 6
noncomputable def initial_angle_at_3 : ℝ := 90
noncomputable def total_angle : ℝ := 130

theorem minute_hand_position (x : ℝ) :
  x = 40 / 6.5 →
  initial_angle_at_3 + degrees_per_minute_hour_hand * x + degrees_per_minute_minute_hand * x = total_angle :=
by
  intro h
  rw [←h]
  have h1 : degrees_per_minute_hour_hand * (40 / 6.5) = 20 / 6.5,
  { sorry },
  have h2 : degrees_per_minute_minute_hand * (40 / 6.5) = 240 / 6.5,
  { sorry },
  rw [h1, h2],
  have h3 : initial_angle_at_3 + 20 / 6.5 + 240 / 6.5 = 130,
  { sorry },
  exact h3

end minute_hand_position_l519_519165


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519726

theorem exists_inhabitant_with_at_least_810_acquaintances :
  (∀ x, ∃ y, x ≠ y) → 
  (∀ x, (∀ k, k = 10% * |neighbors x| → |{y : neighbors x | believes_in_santa y}| = k)) →
  (1_000_000 ≥ ∑ x, believes_in_santa x → 900_000) → 
  (1_000_000 - 900_000 = 100_000) →
  (∃ x, |neighbors x| ≥ 810) :=
sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519726


namespace ratio_depth_to_height_l519_519517

theorem ratio_depth_to_height
  (Dean_height : ℝ := 9)
  (additional_depth : ℝ := 81)
  (water_depth : ℝ := Dean_height + additional_depth) :
  water_depth / Dean_height = 10 :=
by
  -- Dean_height = 9
  -- additional_depth = 81
  -- water_depth = 9 + 81 = 90
  -- water_depth / Dean_height = 90 / 9 = 10
  sorry

end ratio_depth_to_height_l519_519517


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519737

theorem exists_inhabitant_with_at_least_810_acquaintances
  (N : ℕ) (hN : N = 1000000)
  (B : Finset ℕ) (hB : B.card = 900000)
  (A : ℕ → Finset ℕ)
  (hA_nonempty : ∀ i, (A i).nonempty)
  (hA_believers : ∀ i, (A i).card / 10 = (A i ∩ B).card) :
  ∃ j, (A j).card ≥ 810 :=
by
  sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519737


namespace parallelepiped_surface_area_l519_519150

-- Defining the properties of the right parallelepiped and sphere
variables (a b : ℝ)
variable (R : ℝ) -- Radius of the circumscribed sphere
variable (m : ℝ) -- Side length of the rhombus formed in the plane parallel to the base

-- Assuming the property of the rhombus and its diagonals
axiom sphere_circumscribed_parallelepiped_base_diagonal_proof 
  (radius_property : a * b = 2 * m * R) : 6 * a * b = S

-- The theorem to assert the surface area
theorem parallelepiped_surface_area 
  (a b : ℝ) 
  (circumscribed_constr : sphere_circumscribed_parallelepiped_base_diagonal_proof a b R m)
  : 6 * a * b = 6 * a * b :=
begin
  sorry
end

end parallelepiped_surface_area_l519_519150


namespace exists_inhabitant_with_810_acquaintances_l519_519770

-- Definition of the main problem conditions
def inhabitants := 1000000
def belief_in_santa := 0.9 * inhabitants
def disbelief_in_santa := inhabitants - belief_in_santa

-- Conditions
def knows_at_least_one : ∀ x : ℕ, x < inhabitants → ∃ y : ℕ, y < inhabitants ∧ x ≠ y := sorry
def ten_percent_acquaintances (x : ℕ) (h : x < inhabitants) : Prop := ∀ y : ℕ, y < inhabitants ∧ x ≠ y → (y < belief_in_santa → y < 0.1 * x) := sorry

-- Main theorem
theorem exists_inhabitant_with_810_acquaintances :
  ∃ x : ℕ, x < inhabitants ∧ (knows_at_least_one x) ∧ (ten_percent_acquaintances x) ∧ (810 ≤ x) :=
sorry

end exists_inhabitant_with_810_acquaintances_l519_519770


namespace jennas_total_ticket_cost_l519_519804

theorem jennas_total_ticket_cost :
  let normal_price := 50
  let tickets_from_website := 2 * normal_price
  let scalper_price := 2 * normal_price * 2.4 - 10
  let friend_discounted_ticket := normal_price * 0.6
  tickets_from_website + scalper_price + friend_discounted_ticket = 360 :=
by
  sorry

end jennas_total_ticket_cost_l519_519804


namespace relationship_between_y_l519_519695

theorem relationship_between_y (y1 y2 y3 : ℝ)
  (hA : y1 = 3 / -5)
  (hB : y2 = 3 / -3)
  (hC : y3 = 3 / 2) : y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_l519_519695


namespace central_angle_of_sector_l519_519031

open Real

theorem central_angle_of_sector (l S : ℝ) (α R : ℝ) (hl : l = 4) (hS : S = 4) (h1 : l = α * R) (h2 : S = 1/2 * α * R^2) : 
  α = 2 :=
by
  -- Proof will be supplied here
  sorry

end central_angle_of_sector_l519_519031


namespace find_equation_of_plane_l519_519210

-- Definitions corresponding to conditions.
def point := (2, -3, 1) -- The given point
def plane1 (x y z : ℝ) := 3 * x - 2 * y + z = 5 -- The given parallel plane

-- Statement of the equivalency to be proved.
theorem find_equation_of_plane :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcdA(A, B, C) = 1 ∧ 
  (∀ (x y z : ℝ), (x, y, z) = point → A * x + B * y + C * z + D = 0) ∧
  A = 3 ∧ B = -2 ∧ C = 1 ∧ D = -13 :=
by
  sorry

end find_equation_of_plane_l519_519210


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519729

theorem exists_inhabitant_with_at_least_810_acquaintances :
  (∀ x, ∃ y, x ≠ y) → 
  (∀ x, (∀ k, k = 10% * |neighbors x| → |{y : neighbors x | believes_in_santa y}| = k)) →
  (1_000_000 ≥ ∑ x, believes_in_santa x → 900_000) → 
  (1_000_000 - 900_000 = 100_000) →
  (∃ x, |neighbors x| ≥ 810) :=
sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519729


namespace solution_set_abs_inequality_l519_519436

theorem solution_set_abs_inequality (x : ℝ) : 
  | x + 1 | > 3 ↔ x < -4 ∨ x > 2 :=
by
  sorry

end solution_set_abs_inequality_l519_519436


namespace correct_product_is_l519_519462

theorem correct_product_is (n : ℕ) (mistaken_product : ℕ) (correct_product : ℕ) :
  mistaken_product = n * 21 →
  correct_product = n * 27 →
  correct_product = mistaken_product + 48 :=
begin
  sorry,
end

end correct_product_is_l519_519462


namespace cost_per_gallon_calc_l519_519905

theorem cost_per_gallon_calc : 
    let time_to_fill := 50
    let hose_flow_rate := 100
    let total_cost := 5
    let pool_capacity := hose_flow_rate * time_to_fill
    cost_per_gallon total_cost pool_capacity = 0.001 :=
by {
    sorry
}

end cost_per_gallon_calc_l519_519905


namespace mail_total_correct_l519_519344

def Monday_mail : ℕ := 65
def Tuesday_mail : ℕ := Monday_mail + 10
def Wednesday_mail : ℕ := Tuesday_mail - 5
def Thursday_mail : ℕ := Wednesday_mail + 15
def total_mail : ℕ := Monday_mail + Tuesday_mail + Wednesday_mail + Thursday_mail

theorem mail_total_correct : total_mail = 295 := by
  sorry

end mail_total_correct_l519_519344


namespace units_digit_of_sum_of_sequence_is_9_l519_519920

def units_digit_sum_sequence (s : ℕ → ℕ) : ℕ :=
  let seq_sum := (List.range 11).map (λ n, s (n + 1) + (n + 1))
  (seq_sum.sum % 10)

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem units_digit_of_sum_of_sequence_is_9 :
  units_digit_sum_sequence (λ n, factorial n) = 9 :=
by
  sorry

end units_digit_of_sum_of_sequence_is_9_l519_519920


namespace triangle_area_is_9_point_5_l519_519128

def Point : Type := (ℝ × ℝ)

def A : Point := (0, 1)
def B : Point := (4, 0)
def C : Point := (3, 5)

noncomputable def areaOfTriangle (A B C : Point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_is_9_point_5 :
  areaOfTriangle A B C = 9.5 :=
by
  sorry

end triangle_area_is_9_point_5_l519_519128


namespace factorial_divisibility_l519_519214

theorem factorial_divisibility (count : ℕ) : count = 100 :=
  by
    have h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → (n^3 - 1)! % (n!^(n+1)) = 0
    have h2 : ∃ (n : ℕ) (h : 1 ≤ n ∧ n ≤ 100), (n^3 - 1)! % (n!^(n+1)) ≠ 0
    sorry

end factorial_divisibility_l519_519214


namespace units_digit_of_sum_of_sequence_is_9_l519_519919

def units_digit_sum_sequence (s : ℕ → ℕ) : ℕ :=
  let seq_sum := (List.range 11).map (λ n, s (n + 1) + (n + 1))
  (seq_sum.sum % 10)

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem units_digit_of_sum_of_sequence_is_9 :
  units_digit_sum_sequence (λ n, factorial n) = 9 :=
by
  sorry

end units_digit_of_sum_of_sequence_is_9_l519_519919


namespace integer_part_sum_l519_519555

noncomputable def x : ℕ → ℚ
| 0       := 1 / 3
| (n + 1) := x n ^ 2 + x n

theorem integer_part_sum : 
  (⌊∑ k in Finset.range 2007, (1 / (x k + 1))⌋) = 2 :=
begin
  sorry
end

end integer_part_sum_l519_519555


namespace monotonic_power_function_l519_519298

theorem monotonic_power_function {m : ℝ} (hm1_pos : m^2 - 9 * m + 19 > 0)
    (hm2_pos : m - 4 > 0) : 
    (∀ x : ℝ, x > 0 → has_deriv_at (λ x, (m^2 - 9 * m + 19) * x^(m-4)) (((m^2 - 9 * m + 19) * (m-4) * x^(m-5))) x)
    → m = 6 := 
by
  sorry

end monotonic_power_function_l519_519298


namespace rhombus_longer_diagonal_length_l519_519138

theorem rhombus_longer_diagonal_length (side_length : ℕ) (shorter_diagonal : ℕ) 
  (h_side : side_length = 65) (h_short_diag : shorter_diagonal = 56) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 118 :=
by
  sorry

end rhombus_longer_diagonal_length_l519_519138


namespace problem_statement_l519_519476

-- Define the sum of integers from 60 to 80
def sum_integers_60_to_80 : ℕ := ∑ i in finset.Icc 60 80, i

-- Define the number of even integers from 60 to 80
def count_even_integers_60_to_80 : ℕ := (80 - 60) / 2 + 1

-- Prove that the sum and count equals to the given solution
theorem problem_statement : sum_integers_60_to_80 + count_even_integers_60_to_80 = 1481 :=
by {
  -- The given proof steps as conditions are provided above, not needed for the Lean statement.
  sorry
}

end problem_statement_l519_519476


namespace smallest_a_exists_l519_519825

noncomputable def polynomial_satisfies_conditions (P : ℤ[X]) (a : ℤ) : Prop :=
  P.eval 1 = a ∧ P.eval 3 = a ∧ P.eval 5 = a ∧ P.eval 7 = a ∧ P.eval 9 = a ∧
  P.eval 2 = -a ∧ P.eval 4 = -a ∧ P.eval 6 = -a ∧ P.eval 8 = -a ∧ P.eval 10 = -a

theorem smallest_a_exists :
  ∃ (P : ℤ[X]), (a : ℤ), a > 0 ∧ polynomial_satisfies_conditions P a ∧ a = 945 :=
by
  sorry

end smallest_a_exists_l519_519825


namespace g_neg_l519_519841

noncomputable def g : ℚ+ → ℚ := sorry -- Definition of the function g

axiom g_mul (a b : ℚ+) : g (a * b) = g a + g b
axiom g_prime (p : ℕ) (hp : Prime p) : g ⟨p, Nat.cast_pos.2 (by norm_num1)⟩ = p

theorem g_neg {x : ℚ+} : x = 10/33 → g x < 0 :=
by
  intro h
  rw h
  sorry -- Proof steps to conclude g(10/33) < 0

end g_neg_l519_519841


namespace simplify_complex_expression_l519_519392

theorem simplify_complex_expression :
  let a := (-1 + complex.I * real.sqrt 7) / 2
  let b := (-1 - complex.I * real.sqrt 7) / 2
  a^4 + b^4 = 1 :=
by 
  sorry

end simplify_complex_expression_l519_519392


namespace min_length_M_intersect_N_l519_519844

-- Define the sets M and N with the given conditions
def M (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2/3}
def N (n : ℝ) : Set ℝ := {x | n - 3/4 ≤ x ∧ x ≤ n}
def M_intersect_N (m n : ℝ) : Set ℝ := M m ∩ N n

-- Define the condition that M and N are subsets of [0, 1]
def in_interval (m n : ℝ) := (M m ⊆ {x | 0 ≤ x ∧ x ≤ 1}) ∧ (N n ⊆ {x | 0 ≤ x ∧ x ≤ 1})

-- Define the length of a set given by an interval [a, b]
def length_interval (a b : ℝ) := b - a

-- Define the length of the intersection of M and N
noncomputable def length_M_intersect_N (m n : ℝ) : ℝ :=
  let a := max m (n - 3/4)
  let b := min (m + 2/3) n
  length_interval a b

-- Prove that the minimum length of M ∩ N is 5/12
theorem min_length_M_intersect_N (m n : ℝ) (h : in_interval m n) : length_M_intersect_N m n = 5 / 12 :=
by
  sorry

end min_length_M_intersect_N_l519_519844


namespace inscribable_quadrilateral_and_angle_l519_519499

variables {A B C O P Q : Point}
variable {r : ℝ}

-- Assumptions
axiom inscribed_circle (ABC : Triangle) (O : Point) (rABC : ℝ) :
  InCircle ABC O rABC

axiom points_of_tangency (O ABC : Triangle) (P Q : Point) :
  TangentAt ABC A P ∧ TangentAt ABC B Q

axiom radius_condition (rABC rBPOQ : ℝ) (h : rBPOQ = rABC / 2) :
  radius O_P Q = radius O_P P / 2

theorem inscribable_quadrilateral_and_angle :
  (InscribableQuadrilateral B P O Q) ∧ (angle B A C = 90) :=
sorry

end inscribable_quadrilateral_and_angle_l519_519499


namespace not_both_zero_l519_519689

theorem not_both_zero (x y : ℝ) (h : x^2 + y^2 ≠ 0) : ¬ (x = 0 ∧ y = 0) :=
by {
  sorry
}

end not_both_zero_l519_519689


namespace committee_probability_l519_519027

theorem committee_probability :
  let total_committee := (27.choose 5)
  let all_boys_committee := (15.choose 5)
  let all_girls_committee := (12.choose 5)
  1 - (all_boys_committee + all_girls_committee) / total_committee = 76935 / 80730 :=
by
  have total_committee := (Nat.choose 27 5)
  have all_boys_committee := (Nat.choose 15 5)
  have all_girls_committee := (Nat.choose 12 5)
  sorry

end committee_probability_l519_519027


namespace cos_theta_correct_l519_519772

-- Define the points and vectors as required by condition.
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A : Point := ⟨0, 0, 0⟩
def B : Point := ⟨2, 0, 0⟩
def D : Point := ⟨0, 2, 0⟩
def C : Point := ⟨2, 2, 0⟩
def A1 : Point := ⟨0, 0, 2⟩
def B1 : Point := ⟨2, 0, 2⟩
def D1 : Point := ⟨0, 2, 2⟩
def C1 : Point := ⟨2, 2, 2⟩

def O : Point := ⟨1, 1, 0⟩
def E : Point := ⟨2, 2, 1⟩
def F : Point := ⟨0, 1, 1⟩

def vector (p1 p2 : Point) : Point :=
⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

def dot_product (v1 v2 : Point) : ℝ :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def norm (v : Point) : ℝ :=
Real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

noncomputable def cos_theta : ℝ :=
let OE := vector O E in
let FD1 := vector F D1 in
(dot_product OE FD1) / ((norm OE) * (norm FD1))

theorem cos_theta_correct : cos_theta = Real.sqrt 6 / 3 :=
sorry

end cos_theta_correct_l519_519772


namespace exists_inhabitant_with_810_acquaintances_l519_519771

-- Definition of the main problem conditions
def inhabitants := 1000000
def belief_in_santa := 0.9 * inhabitants
def disbelief_in_santa := inhabitants - belief_in_santa

-- Conditions
def knows_at_least_one : ∀ x : ℕ, x < inhabitants → ∃ y : ℕ, y < inhabitants ∧ x ≠ y := sorry
def ten_percent_acquaintances (x : ℕ) (h : x < inhabitants) : Prop := ∀ y : ℕ, y < inhabitants ∧ x ≠ y → (y < belief_in_santa → y < 0.1 * x) := sorry

-- Main theorem
theorem exists_inhabitant_with_810_acquaintances :
  ∃ x : ℕ, x < inhabitants ∧ (knows_at_least_one x) ∧ (ten_percent_acquaintances x) ∧ (810 ≤ x) :=
sorry

end exists_inhabitant_with_810_acquaintances_l519_519771


namespace sector_area_l519_519230

theorem sector_area (theta : ℝ) (r : ℝ) (h_theta : theta = 150)
  (h_r : r = 2) :
  (theta / 360 * π * r^2) = (5 / 3 * π) := 
by
  rw [h_theta, h_r]
  have : 150 / 360 = 5 / 12 := by norm_num
  rw this
  norm_num
  ring
  sorry

end sector_area_l519_519230


namespace no_positive_alpha_exists_l519_519797

theorem no_positive_alpha_exists :
  ¬ ∃ α > 0, ∀ x : ℝ, |Real.cos x| + |Real.cos (α * x)| > Real.sin x + Real.sin (α * x) :=
by
  sorry

end no_positive_alpha_exists_l519_519797


namespace fermat_theorem_composite_l519_519690

theorem fermat_theorem_composite (x y : ℤ) (p : ℕ) [Prime p] (hx : x = 2 * y)
  (h_prime: (x - 1)^p + x^p = (x + 1)^p) :
  ∀ (q : ℕ), Composite q → (x - 1)^q + x^q = (x + 1)^q :=
by
  sorry

end fermat_theorem_composite_l519_519690


namespace sum_of_digits_of_square_of_222222222_l519_519079

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

noncomputable def square (n : ℕ) : ℕ := n * n

theorem sum_of_digits_of_square_of_222222222 :
  sum_of_digits (square 222222222) = 162 :=
sorry

end sum_of_digits_of_square_of_222222222_l519_519079


namespace measure_angle_CAD_l519_519795

open Real

variables {A B C D : Type}
variables [Triangle A B C] [OnSegment D A B]

-- Definitions of constants
def AD := length A D
def DC := length D C
def CB := length C B
def ACB := angle A C B

-- Given conditions
axiom AD_eq_DC : AD = DC
axiom DC_eq_CB : DC = CB
axiom ACB_is_right : ACB = 90

-- Goal
theorem measure_angle_CAD : ∀ (s : ℝ), angle C A D = s → s = 90 :=
by sorry

end measure_angle_CAD_l519_519795


namespace formal_series_eq_sum_S_Dn_l519_519583

variable {R : Type*} [CommRing R] {x : R}
noncomputable def formal_series (a : ℕ → R) : R[X] :=
 ∑ n, (a n) * X^n

noncomputable def S (f : R[X]) : R :=
 constant_coeff f

noncomputable def D (f : R[X]) : R[X] :=
 deriv f

theorem formal_series_eq_sum_S_Dn (f : R[X]) (a : ℕ → R)
  (h0 : f = formal_series a)
  (h1 : ∀ n, S (D^[n] f) = n! * a n) :
  f = ∑ n, S (D^[n] f) * X^n / nat.factorial n :=
sorry

end formal_series_eq_sum_S_Dn_l519_519583


namespace six_digit_palindromes_count_l519_519315

theorem six_digit_palindromes_count : 
  let is_palindrome (n : ℕ) := n.digits = n.digits.reverse in
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ is_palindrome n ↔ 900 :=
sorry

end six_digit_palindromes_count_l519_519315


namespace length_of_introduction_l519_519177

theorem length_of_introduction (total_words : ℕ) (body_section_length : ℕ) (num_body_sections : ℕ) (intro_concl_ratio : ℕ) 
  (total_eq : total_words = 5000) (body_len_eq : body_section_length = 800) (num_body_eq : num_body_sections = 4) 
  (ratio_eq : intro_concl_ratio = 3) : 
  ∃ I : ℕ, I + ratio_eq * I = total_words - num_body_sections * body_section_length ∧ I = 450 := 
sorry

end length_of_introduction_l519_519177


namespace sqrt_cos_squared_660_eq_one_half_l519_519898

theorem sqrt_cos_squared_660_eq_one_half :
  sqrt (cos 660 * cos 660) = 1 / 2 :=
sorry

end sqrt_cos_squared_660_eq_one_half_l519_519898


namespace part1_part2_l519_519643

def f (x a : ℝ) : ℝ := (Exp x / x) - Log x + x - a

theorem part1 (x a : ℝ) (hx : f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 (x1 x2 a : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) 
  (hx1_range : 0 < x1) (hx2_range : x2 > 1) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519643


namespace log_ordering_l519_519970

theorem log_ordering :
  ∀ (log_3_2 log_5_3 log_625_75 two_thirds : ℝ),
  log_3_2 = log 2 / log 3 →
  log_5_3 = log 3 / log 5 →
  log_625_75 = (log 75 / (4 * log 5)) →
  two_thirds = 2 / 3 →
  log_3_2 < two_thirds ∧
  two_thirds < log_625_75 ∧
  log_625_75 < log_5_3 :=
by
  intros log_3_2 log_5_3 log_625_75 two_thirds
  intro h_log_3_2 h_log_5_3 h_log_625_75 h_two_thirds
  sorry

end log_ordering_l519_519970


namespace tiles_needed_l519_519044

def room_area : ℝ := 2 * 4 * 2 * 6
def tile_area : ℝ := 1.5 * 2

theorem tiles_needed : room_area / tile_area = 32 := 
by
  sorry

end tiles_needed_l519_519044


namespace least_number_to_subtract_l519_519941

theorem least_number_to_subtract (x : ℕ) (h : 5026 % 5 = x) : x = 1 :=
by sorry

end least_number_to_subtract_l519_519941


namespace ordered_quintuples_count_l519_519537

noncomputable def count_ordered_quintuples : Nat :=
  6528

theorem ordered_quintuples_count :
  ∃ (S : Fin 8 → Fin 8 → Fin 8 → Fin 8 → Fin 8 → Nat),
  (∀ (a_1 a_2 a_3 a_4 a_5 : Fin 8),
    5 ∣ (2^a_1 + 2^a_2 + 2^a_3 + 2^a_4 + 2^a_5)) →
    count_ordered_quintuples = (∑ (a_1 a_2 a_3 a_4 a_5 : Fin 8), S a_1 a_2 a_3 a_4 a_5) :=
begin
  sorry
end

end ordered_quintuples_count_l519_519537


namespace gcd_32_48_l519_519075

/--
The greatest common factor of 32 and 48 is 16.
-/
theorem gcd_32_48 : Int.gcd 32 48 = 16 :=
by
  sorry

end gcd_32_48_l519_519075


namespace angle_ACE_in_regular_pentagon_equals_30_degrees_l519_519790

theorem angle_ACE_in_regular_pentagon_equals_30_degrees 
  (ABCDE : Type)
  [ConvexPentagon ABCDE]
  [EquilateralPentagon ABCDE]
  (angle_ACE_eq_half_angle_BCD : ∀ (A B C D E : ABCDE),
    ∠ACE = (1 / 2) * ∠BCD) 
: ∀ (A B C D E : ABCDE), ∠ACE = 30 :=
by
  sorry

end angle_ACE_in_regular_pentagon_equals_30_degrees_l519_519790


namespace soda_price_ratio_l519_519167

theorem soda_price_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) :
  (let volume_x := 1.35 * v in
   let price_x := 0.85 * p in
   (price_x / volume_x) / (p / v) = 17 / 27) :=
by
  let volume_x := 1.35 * v
  let price_x := 0.85 * p
  calc (price_x / volume_x) / (p / v)
      = ((0.85 * p) / (1.35 * v)) / (p / v) : by rw [volume_x, price_x]
  ... = ((85 * p) / (135 * v)) / (p / v)   : by norm_num
  ... = (85 * p / (135 * v)) * (v / p)      : by field_simp [hv, hp]
  ... = 85 / 135                           : by ring
  ... = 17 / 27                            : by norm_num
  ... = 17 / 27                            : by sorry

end soda_price_ratio_l519_519167


namespace total_spending_in_4_years_is_680_l519_519907

-- Define the spending of each person
def trevor_spending_yearly : ℕ := 80
def reed_spending_yearly : ℕ := trevor_spending_yearly - 20
def quinn_spending_yearly : ℕ := reed_spending_yearly / 2

-- Define the total spending in one year
def total_spending_yearly : ℕ := trevor_spending_yearly + reed_spending_yearly + quinn_spending_yearly

-- Let's state what we need to prove
theorem total_spending_in_4_years_is_680 : 
  4 * total_spending_yearly = 680 := 
by 
  -- this is where the proof steps would go
  sorry

end total_spending_in_4_years_is_680_l519_519907


namespace final_painting_width_l519_519465

theorem final_painting_width (total_area : ℕ) (n_paintings : ℕ) (a1 a2 : ℕ)
  (area1 : ℕ) (area2 : ℕ) (height_final : ℕ) (width_final : ℕ) :
  total_area = 200 →
  n_paintings = 5 →
  a1 = 3 →
  a2 = 1 →
  area1 = 5 * 5 →
  area2 = 10 * 8 →
  height_final = 5 →
  width_final = 9 →
  (a1 * area1 + a2 * area2 + height_final * width_final = total_area) :=
begin
  intros,
  sorry
end

end final_painting_width_l519_519465


namespace pen_distribution_l519_519875

theorem pen_distribution (x : ℕ) :
  8 * x + 3 = 12 * (x - 2) - 1 :=
sorry

end pen_distribution_l519_519875


namespace number_of_homes_cleaned_l519_519003

-- Define constants for the amount Mary earns per home and the total amount she made.
def amount_per_home := 46
def total_amount_made := 276

-- Prove that the number of homes Mary cleaned is 6 given the conditions.
theorem number_of_homes_cleaned : total_amount_made / amount_per_home = 6 :=
by
  sorry

end number_of_homes_cleaned_l519_519003


namespace percentage_conversion_l519_519497

-- Define the condition
def decimal_fraction : ℝ := 0.05

-- Define the target percentage
def percentage : ℝ := 5

-- State the theorem
theorem percentage_conversion (df : ℝ) (p : ℝ) (h1 : df = 0.05) (h2 : p = 5) : df * 100 = p :=
by
  rw [h1, h2]
  sorry

end percentage_conversion_l519_519497


namespace problem_solution_l519_519183

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - 3)

theorem problem_solution : S = (13 / 4) + (3 / 4) * Real.sqrt 13 :=
by sorry

end problem_solution_l519_519183


namespace storm_deposit_l519_519966

theorem storm_deposit (C : ℝ) (original_amount after_storm_rate before_storm_rate : ℝ) (after_storm full_capacity : ℝ) :
  before_storm_rate = 0.40 →
  after_storm_rate = 0.60 →
  original_amount = 220 * 10^9 →
  before_storm_rate * C = original_amount →
  C = full_capacity →
  after_storm = after_storm_rate * full_capacity →
  after_storm - original_amount = 110 * 10^9 :=
by
  sorry

end storm_deposit_l519_519966


namespace lauren_total_money_made_is_correct_l519_519346

-- Define the rate per commercial view
def rate_per_commercial_view : ℝ := 0.50
-- Define the rate per subscriber
def rate_per_subscriber : ℝ := 1.00
-- Define the number of commercial views on Tuesday
def commercial_views : ℕ := 100
-- Define the number of new subscribers on Tuesday
def subscribers : ℕ := 27
-- Calculate the total money Lauren made on Tuesday
def total_money_made (rate_com_view : ℝ) (rate_sub : ℝ) (com_views : ℕ) (subs : ℕ) : ℝ :=
  (rate_com_view * com_views) + (rate_sub * subs)

-- Theorem stating that the total money Lauren made on Tuesday is $77.00
theorem lauren_total_money_made_is_correct : total_money_made rate_per_commercial_view rate_per_subscriber commercial_views subscribers = 77.00 :=
by
  sorry

end lauren_total_money_made_is_correct_l519_519346


namespace solve_inequality_l519_519562

theorem solve_inequality (m : ℝ) :
  (m < -1/2 → ∃ x, x ∈ Ioo (-2 : ℝ) (1 / m)) ∧
  (m = -1/2 → ∀ x, ¬(mx^2 + (2 * m - 1) * x - 2 > 0)) ∧
  ( -1/2 < m ∧ m < 0 → ∃ x, x ∈ Ioo (1 / m : ℝ) (-2)) ∧
  (m = 0 → ∃ x, x ∈ Iio (-2 : ℝ)) ∧
  (m > 0 → ∃ x, x ∈ (Iio (-2) ∪ Ioi (1 / m))) :=
by
  sorry

end solve_inequality_l519_519562


namespace prove_f_2_eq_10_prove_f_4_eq_10_prove_f_prove_f_l519_519232

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_cond1 : ∀ x : ℝ, f x + (deriv g x) = 10
axiom f_cond2 : ∀ x : ℝ, f x - (deriv g (4 - x)) = 10
axiom g_even : ∀ x : ℝ, g x = g (-x)

theorem prove_f_2_eq_10 : f 2 = 10 := sorry
theorem prove_f_4_eq_10 : f 4 = 10 := sorry
theorem prove_f'_neg1_eq_f'_neg3 : deriv f (-1) = deriv f (-3) := sorry
theorem prove_f'_2023_ne_0 : deriv f 2023 ≠ 0 := sorry

end prove_f_2_eq_10_prove_f_4_eq_10_prove_f_prove_f_l519_519232


namespace exists_inhabitant_with_810_acquaintances_l519_519752

-- Conditions
def total_population : ℕ := 1000000
def believers_percentage : ℕ := 90
def acquaintances_believers_percentage : ℝ := 0.10

-- Theorem statement: Prove that there exists an inhabitant knowing at least 810 people.
theorem exists_inhabitant_with_810_acquaintances 
    (H1 : ∀ (p : ℕ), p < total_population → ∃ q, q ≠ p)  -- Every person knows at least one other person
    (H2 : 0.9 * total_population = 900000)  -- 90% of the population are believers in Santa Claus
    (H3 : ∀ p, p < total_population → ∀ q, q != p → (belief p → (exists percentage, percentage = acquaintances_believers_percentage))) -- Every person claims exactly 10% of their acquaintances believe in Santa Claus
    : ∃ (habitants : ℕ), habitants ≥ 810
    :=
  sorry

end exists_inhabitant_with_810_acquaintances_l519_519752


namespace price_decrease_percentage_l519_519106

theorem price_decrease_percentage (original_price : ℝ) :
  let first_sale_price := (4/5) * original_price
  let second_sale_price := (1/2) * original_price
  let decrease := first_sale_price - second_sale_price
  let percentage_decrease := (decrease / first_sale_price) * 100
  percentage_decrease = 37.5 := by
  sorry

end price_decrease_percentage_l519_519106


namespace cyrus_mosquito_bites_l519_519929

theorem cyrus_mosquito_bites :
  ∃ x : ℕ, (let total_bites := 14 + x in
  let family_bites := total_bites / 2 in
  let equal_bites_per_family_member := family_bites / 6 in
  equal_bites_per_family_member = x / 6) ∧ x = 14 :=
begin
  sorry
end

end cyrus_mosquito_bites_l519_519929


namespace range_of_m_l519_519699

noncomputable def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, m * x^2 - 4 * x + 1 = 0

theorem range_of_m (m : ℝ) (h : intersects_x_axis m) : m ≤ 4 := by
  sorry

end range_of_m_l519_519699


namespace days_B_to_finish_work_l519_519493

-- Definition of work rates based on the conditions
def work_rate_A (A_days: ℕ) : ℚ := 1 / A_days
def work_rate_B (B_days: ℕ) : ℚ := 1 / B_days

-- Theorem that encapsulates the problem statement
theorem days_B_to_finish_work (A_days B_days together_days : ℕ) (work_rate_A_eq : work_rate_A 4 = 1/4) (work_rate_B_eq : work_rate_B 12 = 1/12) : 
  ∀ (remaining_work: ℚ), remaining_work = 1 - together_days * (work_rate_A 4 + work_rate_B 12) → 
  (remaining_work / (work_rate_B 12)) = 4 :=
by
  sorry

end days_B_to_finish_work_l519_519493


namespace max_area_rectangle_in_right_triangle_l519_519127

theorem max_area_rectangle_in_right_triangle :
  ∀ (x y : ℝ), 
  (∃ (θ : ℝ) (h : ℝ), h = 24 ∧ θ = 60 ∧ y = 24 - (x * (4 * sqrt 3 / 3)) ∧ x = 3 * sqrt 3 ∧  y = 12) → 
  (x * y) = 36 * sqrt 3 :=
by sorry

end max_area_rectangle_in_right_triangle_l519_519127


namespace part1_part2_l519_519644

def f (x a : ℝ) : ℝ := (Exp x / x) - Log x + x - a

theorem part1 (x a : ℝ) (hx : f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 (x1 x2 a : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) 
  (hx1_range : 0 < x1) (hx2_range : x2 > 1) : x1 * x2 < 1 := 
sorry

end part1_part2_l519_519644


namespace lauren_mail_total_l519_519340

theorem lauren_mail_total : 
  let monday := 65
  let tuesday := monday + 10
  let wednesday := tuesday - 5
  let thursday := wednesday + 15
  monday + tuesday + wednesday + thursday = 295 :=
by
  have monday := 65
  have tuesday := monday + 10
  have wednesday := tuesday - 5
  have thursday := wednesday + 15
  calc
    monday + tuesday + wednesday + thursday 
    = 65 + (65 + 10) + (65 + 10 - 5) + (65 + 10 - 5 + 15) : by rfl
    ... = 65 + 75 + 70 + 85 : by rfl
    ... = 295 : by rfl

end lauren_mail_total_l519_519340


namespace total_mail_l519_519337

def monday_mail := 65
def tuesday_mail := monday_mail + 10
def wednesday_mail := tuesday_mail - 5
def thursday_mail := wednesday_mail + 15

theorem total_mail : 
  monday_mail + tuesday_mail + wednesday_mail + thursday_mail = 295 := by
  sorry

end total_mail_l519_519337


namespace period_of_f_area_of_triangle_ABC_l519_519676

-- Define the vectors and function f
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), Real.cos (2 * x))
def vector_b : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)
def f (x : ℝ) : ℝ := vector_a x • vector_b

-- Define the period finding statement
theorem period_of_f : ∃ T, (∀ x, f (x + T) = f x) ∧ T = Real.pi :=
  sorry

-- Define the triangle properties and area statement
structure Triangle :=
  (A : ℝ)
  (AB : ℝ)
  (BC : ℝ)
  (fA : ℝ)

noncomputable def triangle_ABC : Triangle :=
  { A := Real.pi / 6,
    AB := 2 * Real.sqrt 3,
    BC := 2,
    fA := 1 / 2 }

theorem area_of_triangle_ABC : ∃ S, S = 2 * Real.sqrt 3 :=
  sorry

end period_of_f_area_of_triangle_ABC_l519_519676


namespace cosine_of_angle_between_vectors_l519_519451

variables {V : Type*} [InnerProductSpace ℝ V]

theorem cosine_of_angle_between_vectors
  (a b : V)
  (h1 : ¬(a = 0) ∧ ¬(b = 0) ∧ ¬(b ∈ {λ x : V, ∃ k : ℝ, x = k • a}))
  (h2 : inner (a + 2 • b) (2 • a - b) = 0)
  (h3 : inner (a - b) a = 0) :
  real.cos (real.angle (a) (b)) = (real.sqrt 10) / 5 := sorry

end cosine_of_angle_between_vectors_l519_519451


namespace compute_frac_value_l519_519991

theorem compute_frac_value (x : ℕ) (h : x = 9) :
  (x^9 - 27*x^6 + 19683) / (x^6 - 27) = 492804 :=
by {
  rw h,  -- Substitute x = 9
   sorry  -- Proof steps are omitted; add when completing the proof.
}

end compute_frac_value_l519_519991


namespace product_of_first_n_primes_not_square_l519_519381

-- Define the first n prime numbers
def first_n_primes (n : ℕ) : List ℕ :=
  [2] ++ (List.range (n - 1)).map (λ k => nat.prime_from (k + 1))

-- Define a function to compute the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldl (*) 1

-- The main theorem asserting that the product of the first n prime numbers is not a perfect square
theorem product_of_first_n_primes_not_square (n : ℕ) (h₁ : n > 0) : ¬ ∃ k : ℕ, (product (first_n_primes n)) = k * k :=
by
  sorry

end product_of_first_n_primes_not_square_l519_519381


namespace compute_z_pow_six_l519_519349

open Complex

noncomputable def z : ℂ := (-1 + I * Real.sqrt 3) / 2

theorem compute_z_pow_six : z ^ 6 = 1 / 4 := sorry

end compute_z_pow_six_l519_519349


namespace affine_transformation_decomposition_l519_519012

def affine_transformation (L : ℝ → ℝ → ℝ) : Prop :=
∃ T H : ℝ → ℝ → ℝ, ∀ O : ℝ, (T ∘ H) = L ∧ 
  (∀ triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ), 
    ∃ similar_triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ), 
    similar (T ∘ H) triangle similar_triangle)

theorem affine_transformation_decomposition :
  ∀ L : ℝ → ℝ → ℝ, affine_transformation L :=
by admit

end affine_transformation_decomposition_l519_519012


namespace total_spending_in_4_years_l519_519909

def trevor_spending_per_year : ℕ := 80
def reed_to_trevor_diff : ℕ := 20
def reed_to_quinn_factor : ℕ := 2

theorem total_spending_in_4_years :
  ∃ (reed_spending quinn_spending : ℕ),
  (reed_spending = trevor_spending_per_year - reed_to_trevor_diff) ∧
  (reed_spending = reed_to_quinn_factor * quinn_spending) ∧
  ((trevor_spending_per_year + reed_spending + quinn_spending) * 4 = 680) :=
sorry

end total_spending_in_4_years_l519_519909


namespace find_alpha_l519_519776

variable {α p₀ p_new : ℝ}
def Q_d (p : ℝ) : ℝ := 150 - p
def Q_s (p : ℝ) : ℝ := 3 * p - 10
def Q_d_new (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

theorem find_alpha 
  (h_eq_initial : Q_d p₀ = Q_s p₀)
  (h_eq_increase : p_new = 1.25 * p₀)
  (h_eq_new : Q_s p_new = Q_d_new α p_new) :
  α = 1.4 :=
by
  sorry

end find_alpha_l519_519776


namespace simplify_expression_and_evaluate_evaluate_expression_at_one_l519_519867

theorem simplify_expression_and_evaluate (x : ℝ)
  (h1 : x ≠ -2) (h2 : x ≠ 2) (h3 : x ≠ 3) :
  ( ((x^2 - 2*x) / (x^2 - 4*x + 4) - 3 / (x - 2)) / ((x - 3) / (x^2 - 4)) ) = x + 2 :=
by {
  sorry
}

theorem evaluate_expression_at_one :
  ( ((1^2 - 2*1) / (1^2 - 4*1 + 4) - 3 / (1 - 2)) / ((1 - 3) / (1^2 - 4)) ) = 3 :=
by {
  sorry
}

end simplify_expression_and_evaluate_evaluate_expression_at_one_l519_519867


namespace latus_rectum_equation_l519_519881

theorem latus_rectum_equation (y x : ℝ) :
  y^2 = 4 * x → x = -1 :=
sorry

end latus_rectum_equation_l519_519881


namespace time_to_wash_car_l519_519328

theorem time_to_wash_car (W : ℕ) 
    (t_oil : ℕ := 15) 
    (t_tires : ℕ := 30) 
    (n_wash : ℕ := 9) 
    (n_oil : ℕ := 6) 
    (n_tires : ℕ := 2) 
    (total_time : ℕ := 240) 
    (h : n_wash * W + n_oil * t_oil + n_tires * t_tires = total_time) 
    : W = 10 := by
  sorry

end time_to_wash_car_l519_519328


namespace no_prime_divisible_by_77_l519_519275

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 :
  ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  intros p h_prime h_div
  sorry

end no_prime_divisible_by_77_l519_519275


namespace original_number_possible_values_l519_519484

theorem original_number_possible_values (x : ℝ) (n : ℤ) (h1 : x > 0) (h2 : n = ⌈1.28 * x⌉) (h3 : n - 1 < x ∧ x ≤ n) :
  x = 25 / 32 ∨ x = 25 / 16 ∨ x = 75 / 32 ∨ x = 25 / 8 :=
by
  sorry

end original_number_possible_values_l519_519484


namespace angle_between_a_b_l519_519362

variables (a b c : EuclideanSpace ℝ (Fin 3))
variables (ha : ‖a‖ = 1)
variables (hb : ‖b‖ = 1)
variables (hc : ‖c‖ = 1)
variables (h0 : a + 2 • b + 2 • c = 0)

theorem angle_between_a_b :
  real.arccos ((inner a b) / (‖a‖ * ‖b‖)) = 104.477 :=
by sorry

end angle_between_a_b_l519_519362


namespace total_wicks_l519_519523

def length_in_feet := 25
def length_in_inches_per_foot := 12
def total_length_in_inches := length_in_feet * length_in_inches_per_foot
def wick1_length := 6.5
def wick2_length := 9.25
def wick3_length := 12.75
def wick_set_length := wick1_length + wick2_length + wick3_length

theorem total_wicks : total_length_in_inches / wick_set_length * 3 = 30 :=
by
  sorry

end total_wicks_l519_519523


namespace smallest_number_divisible_by_hundred_threes_l519_519577

theorem smallest_number_divisible_by_hundred_threes :
  ∃ n, (∀ m, m < n → ¬decidable (a m % b  = 0)) ∧ (m = 300 ∧ decidable (a m % b = 0)) :=
begin
  -- a_n represents the number composed of n digits of ones.
  -- b represents the number composed of 100 digits of threes.
  let a : ℕ → ℕ := λ n, (10^n - 1) // 9,
  let b : ℕ := (10^100 - 1) // 3,

  -- We need to show that n = 300 is the smallest number for which a(n) is divisible by b.
  sorry
end

end smallest_number_divisible_by_hundred_threes_l519_519577


namespace no_natural_numbers_satisfy_equation_l519_519478

theorem no_natural_numbers_satisfy_equation :
  ¬ ∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y + x + y = 2019 :=
by
  sorry

end no_natural_numbers_satisfy_equation_l519_519478


namespace largest_regular_hexagon_proof_l519_519949

noncomputable def largest_regular_hexagon_side_length (x : ℝ) (H : ConvexHexagon) 
  (hx : -5 < x ∧ x < 6) : ℝ := 11 / 2

-- Convex Hexagon Definition
structure ConvexHexagon :=
  (sides : Vector ℝ 6)
  (is_convex : true)  -- Placeholder for convex property

theorem largest_regular_hexagon_proof (x : ℝ) (H : ConvexHexagon) 
  (hx : -5 < x ∧ x < 6)
  (H_sides_length : H.sides = ⟨[5, 6, 7, 5+x, 6-x, 7+x], by simp⟩) :
  largest_regular_hexagon_side_length x H hx = 11 / 2 :=
sorry

end largest_regular_hexagon_proof_l519_519949


namespace jill_draws_spade_probability_l519_519329

/-- Given the conditions of the card game, we want to prove that the probability that Jill 
    draws the spade is 12/37. -/
theorem jill_draws_spade_probability :
  let jack_spade_prob : ℚ := 1 / 4
      jill_spade_prob : ℚ := (3 / 4) * (1 / 4)
      john_spade_prob : ℚ := ((3 / 4) * (3 / 4)) * (1 / 4)
      total_spade_prob : ℚ := jack_spade_prob + jill_spade_prob + john_spade_prob
      jill_conditional_prob : ℚ := jill_spade_prob / total_spade_prob
  in
  jill_conditional_prob = 12 / 37 :=
by
  sorry

end jill_draws_spade_probability_l519_519329


namespace tan_alpha_plus_pi_over_4_l519_519285

noncomputable def sin_cos_identity (α : ℝ) : Prop :=
  (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 3

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : sin_cos_identity α) :
  Real.tan (α + Real.pi / 4) = -3 :=
  by
  sorry

end tan_alpha_plus_pi_over_4_l519_519285


namespace sum_of_squares_of_sines_l519_519535

theorem sum_of_squares_of_sines :
  (∑ k in Icc (-45 : ℤ) 45, (Real.sin (k * π / 180))^2) = 46 :=
by sorry

end sum_of_squares_of_sines_l519_519535


namespace three_tangent_circles_shared_point_l519_519859

-- Define Triangle type
structure Triangle :=
(A B C : Point)

-- Define Circle type
structure Circle :=
(center : Point)
(radius : Real)

-- Define Tangency property
def tangent (circ : Circle) (side : Line) : Prop := 
-- Implementation of tangency
sorry

-- Define Common Point property
def common_point (circ1 circ2 circ3 : Circle) (p : Point) : Prop := 
-- Implementation of common point existence
sorry

theorem three_tangent_circles_shared_point (T : Triangle) : 
  ∃ (circ1 circ2 circ3 : Circle), 
    (circ1.radius = circ2.radius ∧ circ2.radius = circ3.radius) ∧
    (tangent circ1 (line T.A T.B) ∧ tangent circ1 (line T.B T.C)) ∧
    (tangent circ2 (line T.B T.C) ∧ tangent circ2 (line T.C T.A)) ∧
    (tangent circ3 (line T.C T.A) ∧ tangent circ3 (line T.A T.B)) ∧
    ∃ (p : Point), common_point circ1 circ2 circ3 p := 
sorry

end three_tangent_circles_shared_point_l519_519859


namespace functional_eq_solution_l519_519569

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_eq_solution :
  (∀ x y : ℝ, (x - y) * (f(f(x)^2) - f(f(y)^2)) = (f(x) + f(y)) * (f(x) - f(y))^2) ∧
  (f(0) = 0) → 
  ∃ c : ℝ, ∀ x : ℝ, f(x) = c * x :=
by
  intros h
  sorry

end functional_eq_solution_l519_519569


namespace quadrilateral_fourth_side_length_l519_519508

theorem quadrilateral_fourth_side_length 
  (r : ℝ) (a b c : ℝ) (d : ℝ) 
  (condition_radius : r = 200 * real.sqrt 2)
  (condition_sides : a = 200 ∧ b = 200 ∧ c = 200) :
  d = 500 :=
sorry

end quadrilateral_fourth_side_length_l519_519508


namespace part1_part2_l519_519631

def f (x : ℝ) (a : ℝ) := (Real.exp x) / x - Real.log x + x - a

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → (a ≤ Real.exp 1 + 1) :=
sorry

theorem part2 (x1 x2 a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 ≠ x2 → 0 < x1 → 0 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519631


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519730

theorem exists_inhabitant_with_at_least_810_acquaintances :
  (∀ x, ∃ y, x ≠ y) → 
  (∀ x, (∀ k, k = 10% * |neighbors x| → |{y : neighbors x | believes_in_santa y}| = k)) →
  (1_000_000 ≥ ∑ x, believes_in_santa x → 900_000) → 
  (1_000_000 - 900_000 = 100_000) →
  (∃ x, |neighbors x| ≥ 810) :=
sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519730


namespace exists_positive_int_and_nonzero_reals_l519_519383

theorem exists_positive_int_and_nonzero_reals (
  n : ℕ,
  a : Fin n → ℝ) :
  (∀ (x : ℝ), x ∈ Icc (-1) 1 → abs (x - ∑ i in Finset.range n, a ⟨i, sorry⟩ * x^(2 * i + 1)) < 1 / 1000) := 
sorry

end exists_positive_int_and_nonzero_reals_l519_519383


namespace sum_two_smallest_prime_factors_of_286_l519_519080

theorem sum_two_smallest_prime_factors_of_286 : ∃ p₁ p₂ : ℕ, nat.prime p₁ ∧ nat.prime p₂ ∧ p₁ * p₂ * 13 = 286 ∧ p₁ + p₂ = 13 :=
by
  sorry

end sum_two_smallest_prime_factors_of_286_l519_519080


namespace equation1_solution_equation2_solution_l519_519869

theorem equation1_solution : ∀ x : ℚ, x - 0.4 * x = 120 → x = 200 := by
  sorry

theorem equation2_solution : ∀ x : ℚ, 5 * x - 5/6 = 5/4 → x = 5/12 := by
  sorry

end equation1_solution_equation2_solution_l519_519869


namespace rhombus_longer_diagonal_l519_519143

theorem rhombus_longer_diagonal
  (a b : ℝ)
  (ha : a = 65) -- Side length
  (hb : b = 28) -- Half the length of the shorter diagonal
  : 2 * Real.sqrt (a^2 - b^2) = 118 :=  -- Statement of the longer diagonal length
by
  sorry -- Proof goes here

end rhombus_longer_diagonal_l519_519143


namespace range_of_b_l519_519294

noncomputable def f (x b : ℝ) : ℝ := -1/2 * (x - 2)^2 + b * Real.log (x + 2)
noncomputable def derivative (x b : ℝ) := -(x - 2) + b / (x + 2)

-- Lean theorem statement
theorem range_of_b (b : ℝ) :
  (∀ x > 1, derivative x b ≤ 0) → b ≤ -3 :=
by
  sorry

end range_of_b_l519_519294


namespace number_of_tiles_l519_519046

theorem number_of_tiles (room_width room_height tile_width tile_height : ℝ) :
  room_width = 8 ∧ room_height = 12 ∧ tile_width = 1.5 ∧ tile_height = 2 →
  (room_width * room_height) / (tile_width * tile_height) = 32 :=
by
  intro h
  cases' h with rw h
  cases' h with rh h
  cases' h with tw th
  rw [rw, rh, tw, th]
  norm_num
  sorry

end number_of_tiles_l519_519046


namespace find_alpha_l519_519778

def demand_function (p : ℝ) : ℝ := 150 - p
def supply_function (p : ℝ) : ℝ := 3 * p - 10

def new_demand_function (p : ℝ) (α : ℝ) : ℝ := α * (150 - p)

theorem find_alpha (α : ℝ) :
  (∃ p₀ p_new, demand_function p₀ = supply_function p₀ ∧ 
    p_new = p₀ * 1.25 ∧ 
    3 * p_new - 10 = new_demand_function p_new α) →
  α = 1.4 :=
by
  sorry

end find_alpha_l519_519778


namespace max_value_of_sample_l519_519445

theorem max_value_of_sample 
  (x : Fin 5 → ℤ)
  (h_different : ∀ i j, i ≠ j → x i ≠ x j)
  (h_mean : (x 0 + x 1 + x 2 + x 3 + x 4) / 5 = 7)
  (h_variance : ((x 0 - 7)^2 + (x 1 - 7)^2 + (x 2 - 7)^2 + (x 3 - 7)^2 + (x 4 - 7)^2) / 5 = 4)
  : ∃ i, x i = 10 := 
sorry

end max_value_of_sample_l519_519445


namespace number_decreased_by_13_l519_519490

theorem number_decreased_by_13 (x : ℕ) (hx : 100 ≤ x.digits 10 ∧ x.digits 10 < 10) :
  ∃ (b : ℕ), b ∈ {1, 2, 3} ∧ (x = 1625 * 10^96 ∨ x = 195 * 10^97 ∨ x = 2925 * 10^96 ∨ x = 13 * b * 10^98) :=
sorry

end number_decreased_by_13_l519_519490


namespace area_of_ABEFG_minimum_area_l519_519958

-- Problem 1: Proving the area of the pentagon.
theorem area_of_ABEFG 
(a b : ℝ) (h : a < b):
  let O := (a / 2, b / 2) in
  let E := (a / 2, 0) in
  let F := (0, b / 2) in
  let pentagon_area := a * (3 * b^2 - a^2) / (4 * b) in
  area_of_pentagon a b h O E F = pentagon_area

-- Problem 2: Finding the minimum area of the pentagon when a = 1.
theorem minimum_area
(b : ℝ) (h : b > 0):
  let a := 1 in
  let E := (1 / 2, 0) in
  let F := (0, b / 2) in
  let pentagon_area := 1 * (3 * b^2 - 1) / (4 * b) in
  let min_area := min (pentagon_area) b := (11 / 8) in
  minimum_area_pentagon a b h E F pentagon_area = min_area

end area_of_ABEFG_minimum_area_l519_519958


namespace polygon_area_l519_519916

-- Define the vertices of the polygon
def x1 : ℝ := 0
def y1 : ℝ := 0

def x2 : ℝ := 4
def y2 : ℝ := 0

def x3 : ℝ := 2
def y3 : ℝ := 3

def x4 : ℝ := 4
def y4 : ℝ := 6

-- Define the expression for the Shoelace Theorem
def shoelace_area (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : ℝ :=
  0.5 * abs (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

-- The theorem statement proving the area of the polygon
theorem polygon_area :
  shoelace_area x1 y1 x2 y2 x3 y3 x4 y4 = 6 := 
  by
  sorry

end polygon_area_l519_519916


namespace max_cars_quotient_l519_519853

-- Define the distance formula
def distance_occupied (s : ℝ) : ℝ :=
  4 + (4 * s / 15)

-- Define the number of cars function
def num_cars (s : ℝ) : ℝ :=
  1000 * s / distance_occupied s

-- Find the maximum number of cars
noncomputable def max_num_cars : ℝ :=
  3750

-- Define the quotient function
def quotient_when_divided_by_10 (n : ℝ) : ℝ :=
  n / 10

-- Prove that the quotient when max number of cars is divided by 10 is 375
theorem max_cars_quotient : quotient_when_divided_by_10 max_num_cars = 375 := by
  sorry

end max_cars_quotient_l519_519853


namespace exists_inhabitant_with_at_least_810_acquaintances_l519_519732

theorem exists_inhabitant_with_at_least_810_acquaintances
  (N : ℕ) (hN : N = 1000000)
  (B : Finset ℕ) (hB : B.card = 900000)
  (A : ℕ → Finset ℕ)
  (hA_nonempty : ∀ i, (A i).nonempty)
  (hA_believers : ∀ i, (A i).card / 10 = (A i ∩ B).card) :
  ∃ j, (A j).card ≥ 810 :=
by
  sorry

end exists_inhabitant_with_at_least_810_acquaintances_l519_519732


namespace power_sum_l519_519530

theorem power_sum :
  (-1:ℤ)^53 + 2^(3^4 + 4^3 - 6 * 7) = 2^103 - 1 :=
by
  sorry

end power_sum_l519_519530


namespace part1_part2_l519_519655

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l519_519655


namespace smallest_T_is_214_l519_519890

noncomputable def smallest_possible_T : ℕ :=
  let xs := [1, 2, 3, 4, 5, 6, 7, 8, 9] in
    (xs.combinations 3).map (λ t1, 
      let remaining := xs.diff t1 in 
        (remaining.combinations 3).map (λ t2, 
          let t3 := remaining.diff t2 in 
            t1.prod + t2.prod + t3.prod
        )
    ).join.min

theorem smallest_T_is_214 : smallest_possible_T = 214 :=
  sorry

end smallest_T_is_214_l519_519890


namespace find_C_and_D_l519_519419

theorem find_C_and_D : ∃ (C D : ℚ),
  (∀ x : ℚ, (7 * x - 15) / (3 * x ^ 2 - x - 10) = C / (x + 2) + D / (3 * x - 5)) ∧
  (3 * C + D = 7) ∧
  (-5 * C + 2 * D = -15) ∧
  C = 29 / 11 ∧ D = -9 / 11 :=
by 
    use [29/11, -9/11]
    sorry

end find_C_and_D_l519_519419


namespace replace_signs_to_achieve_2013_l519_519480

theorem replace_signs_to_achieve_2013 : 
  ∃ (f : ℕ → ℤ), (∀ n, 1 ≤ n ∧ n ≤ 2013 → (f n) = n^2 ∨ (f n) = -n^2) 
  ∧ (∑ n in finset.range (2014), f n) = 2013 :=
sorry

end replace_signs_to_achieve_2013_l519_519480


namespace rhombus_area_l519_519247

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 5) (h2 : d2 = 6) : 
  1 / 2 * d1 * d2 = 15 :=
by
  rw [h1, h2]
  norm_num

end rhombus_area_l519_519247


namespace determine_a_l519_519364

def eliminate_quadratic_terms (a x : ℝ) : Prop :=
  (3 - 2 * x^2 - 5 * x) - (a * x^2 + 3 * x - 4) -- The given expression
  = (-8 * x) + 7                                  -- The simplified form without quadratic terms

theorem determine_a (a : ℝ) : eliminate_quadratic_terms a (-2) ↔ a = -2 :=
by {
  -- sorry
}

end determine_a_l519_519364


namespace nice_sets_property_l519_519360

open Set

variables (B C : ℝ × ℝ) (S S' : Set (ℝ × ℝ))

-- Define the points B and C
def B : (ℝ × ℝ) := (-1, 0)
def C : (ℝ × ℝ) := (1, 0)

-- Define what it means for a set to be nice
def nice_set (S : Set (ℝ × ℝ)) : Prop :=
∃ T ∈ S, ∀ Q ∈ S, SegmentLine (T, Q) ⊆ S ∧ 
(∀ (P1 P2 P3 : (ℝ × ℝ)), 
 ∃ (A ∈ S) (σ : Equiv.Perm (Fin 3)), 
 Triangle A B C ≃ Triangle (P1 σ 0) (P2 σ 1) (P3 σ 2))

-- Define the two circles kB and kC
def kB : (ℝ × ℝ) → ℝ := λ P, dist P B = 2
def kC : (ℝ × ℝ) → ℝ := λ P, dist P C = 2

-- Define subsets S and S' of the first quadrant
def S : Set (ℝ × ℝ) := {P | kB P ∧ 0 ≤ P.1 ∧ 0 ≤ P.2}
def S' : Set (ℝ × ℝ) := {P | kC P ∧ 0 ≤ P.1 ∧ 0 ≤ P.2}

-- Define the property to be proven
theorem nice_sets_property :
  nice_set S ∧ nice_set S' ∧ 
  (∀ (P1 P2 P3 : (ℝ × ℝ)), 
   let A := classical.some (nice_set_condition S P1 P2 P3),
       A' := classical.some (nice_set_condition S' P1 P2 P3)
   in (BA (B, A) * BA (B, A')) = 4) :=
sorry

end nice_sets_property_l519_519360


namespace smallest_possible_value_l519_519826

theorem smallest_possible_value (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (b + d)) + (1 / (c + d))) ≥ 8 := 
sorry

end smallest_possible_value_l519_519826


namespace total_rainfall_2019_to_2021_l519_519707

theorem total_rainfall_2019_to_2021 :
  let R2019 := 50
  let R2020 := R2019 + 5
  let R2021 := R2020 - 3
  12 * R2019 + 12 * R2020 + 12 * R2021 = 1884 :=
by
  sorry

end total_rainfall_2019_to_2021_l519_519707


namespace initial_amount_l519_519848

theorem initial_amount (spent_sweets friends_each left initial : ℝ) 
  (h1 : spent_sweets = 3.25) (h2 : friends_each = 2.20) (h3 : left = 2.45) :
  initial = spent_sweets + (friends_each * 2) + left :=
by
  sorry

end initial_amount_l519_519848
