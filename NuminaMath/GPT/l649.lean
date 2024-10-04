import Mathlib
import Mathlib.Algebra.Align
import Mathlib.Algebra.Associated
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.ChineseRemainder
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.NormedSpace.FiniteDimension
import Mathlib.Analysis.NormedSpace.InnerProduct
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Det
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Float
import Mathlib.Probability.Basic
import Mathlib.Probability.Variance
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic

namespace minimum_quadrilateral_area_l649_649327

noncomputable def circle_center := (1, 1)
noncomputable def circle_radius := 2

def line1 (x y : ℝ) := x + y - 2
def line2 (x y : ℝ) := 3 * x + 4 * y + 8
def circle (x y : ℝ) := (x - 1)^2 + (y - 1)^2 - 4

theorem minimum_quadrilateral_area :
  ∀ (P : ℝ × ℝ), (line2 P.1 P.2 = 0) →
  let dist_PM := (P.1 - circle_center.1)^2 + (P.2 - circle_center.2)^2 in
  2 * real.sqrt (dist_PM - circle_radius^2) ≥ 2 * real.sqrt 5 :=
by
  sorry

end minimum_quadrilateral_area_l649_649327


namespace age_of_youngest_child_l649_649667

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 := 
by {
  sorry
}

end age_of_youngest_child_l649_649667


namespace max_difference_is_correct_l649_649252

noncomputable def max_y_difference : ℝ := 
  let x1 := Real.sqrt (2 / 3)
  let y1 := 2 + (x1 ^ 2) + (x1 ^ 3)
  let x2 := -x1
  let y2 := 2 + (x2 ^ 2) + (x2 ^ 3)
  abs (y1 - y2)

theorem max_difference_is_correct : max_y_difference = 4 * Real.sqrt 2 / 9 := 
  sorry -- Proof is omitted

end max_difference_is_correct_l649_649252


namespace infinite_series_sum_l649_649933

theorem infinite_series_sum (x : ℝ) (h : x > 1) :
  ∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (-(3 ^ n))) = 1 / (x - 1) :=
sorry

end infinite_series_sum_l649_649933


namespace jamie_total_balls_after_buying_l649_649536

theorem jamie_total_balls_after_buying (red_balls : ℕ) (blue_balls : ℕ) (yellow_balls : ℕ) (lost_red_balls : ℕ) (final_red_balls : ℕ) (total_balls : ℕ)
  (h1 : red_balls = 16)
  (h2 : blue_balls = 2 * red_balls)
  (h3 : lost_red_balls = 6)
  (h4 : final_red_balls = red_balls - lost_red_balls)
  (h5 : yellow_balls = 32)
  (h6 : total_balls = final_red_balls + blue_balls + yellow_balls) :
  total_balls = 74 := by
    sorry

end jamie_total_balls_after_buying_l649_649536


namespace part1_part2_l649_649867

def vector (α : Type*) := (α × α)
def a : vector ℝ := (3, 2)
def b : vector ℝ := (-1, 2)
def c : vector ℝ := (4, 1)

-- Part (Ⅰ)
theorem part1 (m n : ℝ) : a = (m • b.1 + n • c.1, m • b.2 + n • c.2) ↔ (m = 5 / 9 ∧ n = 8 / 9) :=
by
  sorry

-- Part (Ⅱ)
def parallel (v w : vector ℝ) : Prop := v.1 * w.2 = v.2 * w.1
def magnitude (v : vector ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem part2 (d : vector ℝ) :
  parallel (d.1 - c.1, d.2 - c.2) (a.1 + b.1, a.2 + b.2) ∧ magnitude (d.1 - c.1, d.2 - c.2) = real.sqrt 5 →
  d = (3, -1) ∨ d = (5, 3) :=
by
  sorry

end part1_part2_l649_649867


namespace fraction_of_sum_l649_649681

theorem fraction_of_sum (n : ℝ) (l : List ℝ) (h1 : l.length = 21) 
(h2 : ∃ k, List.perm l (n::k)) (h3 : l.nodup) 
(h4 : ∃ m, List.sum m = (list.sum l - n) ∧ (list.length m = 20) 
∧ n = 5 * (list.sum m / 20)) : n = (1 / 5) * list.sum l := 
by sorry

end fraction_of_sum_l649_649681


namespace probability_of_even_sum_selected_primes_l649_649421

open Nat

-- Define the first 12 prime numbers
def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

-- Define the probability calculation
noncomputable def probability_even_sum := (55 : ℚ) / 132

-- The theorem proving the probability
theorem probability_of_even_sum_selected_primes :
  let selected_primes := choose_sublist first_twelve_primes 5 in
  let even_sum := (selected_primes.sum % 2 = 0) in
  (∃ s, s ⊆ first_twelve_primes ∧ s.length = 5 ∧ even_sum) →
  (probability_even_sum = (55 : ℚ) / 132) :=
by sorry

end probability_of_even_sum_selected_primes_l649_649421


namespace eccentricity_of_ellipse_l649_649250

theorem eccentricity_of_ellipse (b : ℝ) (h : 6 + b^2 = 10) : 
  sqrt (6 - b^2) / sqrt 6 = sqrt 3 / 3 :=
by
  -- Given that a^2 = 6 and r^2 = 10, we can identify that h translates to b^2 = 4
  have h1 : b^2 = 4 := by linarith,
  -- Simplify the eccentricity formula
  have h2 : sqrt (6 - b^2) = sqrt 2 := by rw [h1, sub_eq_add_neg, add_neg_eq_iff_eq_sub]; norm_num,
  have h3 : sqrt 6 = sqrt 6 := rfl,
  rw [h2, h3],
  field_simp,
  norm_num,
  sorry -- Proof steps omitted

end eccentricity_of_ellipse_l649_649250


namespace transformation_C_factorization_l649_649365

open Function

theorem transformation_C_factorization (a b : ℤ) :
  (a - 1) * (b - 1) = ab - a - b + 1 :=
by sorry

end transformation_C_factorization_l649_649365


namespace vector_magnitude_parallel_l649_649089

/-- Given two plane vectors a = (1, 2) and b = (-2, y),
if a is parallel to b, then |2a - b| = 4 * sqrt 5. -/
theorem vector_magnitude_parallel (y : ℝ) 
  (h_parallel : (1 : ℝ) / (-2 : ℝ) = (2 : ℝ) / y) : 
  ‖2 • (1, 2) - (-2, y)‖ = 4 * Real.sqrt 5 := 
by
  sorry

end vector_magnitude_parallel_l649_649089


namespace shaded_area_circles_l649_649725

theorem shaded_area_circles (r1 r2 : ℝ) (h1: r1 = 2) (h2: r2 = 3) :
  let R := (2 * r1 + 2 * r2) / 2 in
  π * R ^ 2 - π * r1 ^ 2 - π * r2 ^ 2 = 12 * π :=
by
  -- Definitions of r1, r2, and R are given in the conditions.
  sorry

end shaded_area_circles_l649_649725


namespace solution_l649_649722

noncomputable def problem_statement : Prop :=
  (Real.logBase 4 3 + Real.logBase 8 3) * (Real.logBase 3 2 + Real.logBase 9 8) = 25 / 12

theorem solution : problem_statement :=
by
  sorry

end solution_l649_649722


namespace smallest_delicious_integer_l649_649981

def delicious (N : ℤ) : Prop :=
  ∃ (m k : ℤ), m ≤ k ∧ (∑ i in m..k, i) = 2023 ∧ N = k

theorem smallest_delicious_integer :
  ∀ (N : ℤ), delicious N → N ≥ -2022 :=
by
  sorry

end smallest_delicious_integer_l649_649981


namespace cone_volume_l649_649890

theorem cone_volume
  (θ : ℝ) (A : ℝ) (V : ℝ) 
  (hθ : θ = 4 / 3 * Real.pi) 
  (hA : A = 6 * Real.pi) :
  V = (4 * Real.sqrt 5 / 3) * Real.pi :=
begin
  sorry
end

end cone_volume_l649_649890


namespace part_one_part_two_l649_649857

noncomputable def f (a x : ℝ) : ℝ := x * (a + Real.log x)
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem part_one (a : ℝ) (h : ∀ x > 0, f a x ≥ -1/Real.exp 1) : a = 0 := sorry

theorem part_two {a x : ℝ} (ha : a > 0) (hx : x > 0) :
  g x - f a x < 2 / Real.exp 1 := sorry

end part_one_part_two_l649_649857


namespace prize_amount_l649_649700

theorem prize_amount (P : ℝ) (n : ℝ) (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : n = 40)
  (h2 : a = 40)
  (h3 : b = (2 / 5) * P)
  (h4 : c = (3 / 5) * 40)
  (h5 : b / c = 120) :
  P = 7200 := 
sorry

end prize_amount_l649_649700


namespace cos_450_eq_0_l649_649740

theorem cos_450_eq_0 : Real.cos (450 * Real.pi / 180) = 0 := by
  -- Angle equivalence: 450 degrees is equivalent to 90 degrees on the unit circle
  have angle_eq : (450 : Real) * Real.pi / 180 = (90 : Real) * Real.pi / 180 := by
    calc
      (450 : Real) * Real.pi / 180
        = (450 / 180) * Real.pi : by rw [mul_div_assoc]
        = (5 * 90 / 180) * Real.pi : by norm_num
        = (5 * 90 / (2 * 90)) * Real.pi : by norm_num
        = (5 / 2) * Real.pi : by norm_num

  -- Now use this equivalence: cos(450 degrees) = cos(90 degrees)
  have cos_eq : Real.cos (450 * Real.pi / 180) = Real.cos (90 * Real.pi / 180) := by
    rw [angle_eq]

  -- Using the fact that cos(90 degrees) = 0
  have cos_90 : Real.cos (90 * Real.pi / 180) = 0 := by
    -- This step can use a known trigonometric fact from mathlib
    exact Real.cos_pi_div_two

  -- Therefore
  rw [cos_eq, cos_90]
  exact rfl

end cos_450_eq_0_l649_649740


namespace value_of_nabla_expression_l649_649025

namespace MathProblem

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem value_of_nabla_expression : nabla (nabla 2 3) 2 = 4099 :=
by
  sorry

end MathProblem

end value_of_nabla_expression_l649_649025


namespace angle_A_area_of_triangle_l649_649510

theorem angle_A (A B C a b c : ℝ) (h1 : A + B + C = Real.pi) (h2 : b^2 + c^2 - a^2 = 2 * b * c * Real.sin (B + C)) : 
  A = Real.pi / 4 :=
sorry

theorem area_of_triangle (a B : ℝ) (A B C b : ℝ) (h1 : A + B + C = Real.pi) (h2 : b^2 + c^2 - a^2 = 2 * b * c * Real.sin (B + C))
  (ha : a = 2) (hB : B = Real.pi / 3) : 
  let C := Real.arcsin ((b * Real.sin (B + A)) / (a * Real.sin A)) in
  let area := (1/2) * a * b * Real.sin C in
  area = (3 + Real.sqrt 3) / 2 :=
sorry

end angle_A_area_of_triangle_l649_649510


namespace intersection_complement_l649_649865

-- Define the sets A and B
def A : Set ℝ := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x : ℝ | x > 0 }

-- Define the complement of B
def complement_B : Set ℝ := { x : ℝ | x ≤ 0 }

-- The theorem we need to prove
theorem intersection_complement :
  A ∩ complement_B = { x : ℝ | -1 ≤ x ∧ x ≤ 0 } := 
by
  sorry

end intersection_complement_l649_649865


namespace Jonas_needs_to_buy_35_pairs_of_socks_l649_649172

theorem Jonas_needs_to_buy_35_pairs_of_socks
  (socks : ℕ)
  (shoes : ℕ)
  (pants : ℕ)
  (tshirts : ℕ)
  (double_items : ℕ)
  (needed_items : ℕ)
  (pairs_of_socks_needed : ℕ) :
  socks = 20 →
  shoes = 5 →
  pants = 10 →
  tshirts = 10 →
  double_items = 2 * (2 * socks + 2 * shoes + pants + tshirts) →
  needed_items = double_items - (2 * socks + 2 * shoes + pants + tshirts) →
  pairs_of_socks_needed = needed_items / 2 →
  pairs_of_socks_needed = 35 :=
by sorry

end Jonas_needs_to_buy_35_pairs_of_socks_l649_649172


namespace range_of_function_l649_649456

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  a - 1 / (2^x - 1)

theorem range_of_function (a : ℝ) :
  (∀ x ∈ (-∞, -1] ∪ [1, +∞), f a (-x) = - f a x) →
  a = -1/2 →
  set.range (λ x : ℝ, if x ∈ (-∞, -1] ∪ [1, +∞) then f a x else 0) = 
    (set.Icc (-3/2 : ℝ) (-1/2) ∪ set.Icc (1/2 : ℝ) (3/2)) :=
begin
  intro h_odd,
  intro ha,
  sorry
end

end range_of_function_l649_649456


namespace find_constants_sum_l649_649808

theorem find_constants_sum :
  ∃ (a b c d : ℕ),
  (∀ x y : ℝ, x + y = 6 ∧ 3 * x * y = 6 → (x = (a : ℝ) + b * real.sqrt c ∨ x = (a : ℝ) - b * real.sqrt c) ∧ d = 1) ∧
  a + b + c + d = 12 :=
by sorry

end find_constants_sum_l649_649808


namespace translation_of_A_l649_649572

-- Definitions

def A : ℝ × ℝ := (-1, 2)

def M (t : ℝ) : ℝ × ℝ := (t - 1, 2 * t + 2)

def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Statement
theorem translation_of_A (A_1 : ℝ × ℝ) (hA_1 : distance A A_1 = real.sqrt 5) :
    A_1 = (-2, 0) ∨ A_1 = (0, 4) :=
sorry

end translation_of_A_l649_649572


namespace nick_average_speed_is_24_l649_649662

-- Definitions for the given conditions
def distance_home_to_market := D : ℝ
def bus_speed := 80 : ℝ
def walk_speed := 8 : ℝ
def cycle_speed := 60 : ℝ

-- Time calculations for each segment of the journey
def time_by_bus (D : ℝ) := distance_home_to_market / bus_speed
def time_by_walk (D : ℝ) := (distance_home_to_market / 2) / walk_speed
def time_by_cycle (D : ℝ) := (distance_home_to_market / 2) / cycle_speed

-- Total distance calculation
def total_distance (D : ℝ) := 2 * distance_home_to_market

-- Total time calculation
def total_time (D : ℝ) :=
  time_by_bus D + time_by_walk D + time_by_cycle D

-- Average speed calculation
def average_speed (D : ℝ) :=
  total_distance D / total_time D

-- Theorem stating the average speed is 24 kmph
theorem nick_average_speed_is_24 : ∀ (D : ℝ), D > 0 → average_speed D = 24 := by
  intros D hD_pos
  simp [total_distance, total_time, average_speed, time_by_bus, time_by_walk, time_by_cycle]
  -- Detailed proof would go here
  sorry

end nick_average_speed_is_24_l649_649662


namespace fourth_grade_students_end_year_l649_649904

theorem fourth_grade_students_end_year :
  ∀ (initial_students students_left new_students : ℕ),
    initial_students = 4 →
    students_left = 3 →
    new_students = 42 →
    (initial_students - students_left + new_students) = 43 :=
by
  intros initial_students students_left new_students h_initial h_left h_new
  rw [h_initial, h_left, h_new]
  simp
  sorry

end fourth_grade_students_end_year_l649_649904


namespace distance_descending_correct_l649_649010

/-- Define Chrystal's initial speed -/
def initial_speed : ℝ := 30

/-- Define the speed reduction factor while ascending -/
def ascending_factor : ℝ := 0.5

/-- Define the speed increase factor while descending -/
def descending_factor : ℝ := 1.2

/-- Define the distance to the top of the mountain -/
def distance_up : ℝ := 60

/-- Define the total time to pass the whole mountain -/
def total_time : ℝ := 6

/-- Calculate the speed while ascending -/
def ascending_speed : ℝ := initial_speed * ascending_factor

/-- Calculate the time taken to ascend the mountain -/
def time_up : ℝ := distance_up / ascending_speed

/-- Calculate the time taken to descend the mountain -/
def time_down : ℝ := total_time - time_up

/-- Calculate the speed while descending -/
def descending_speed : ℝ := initial_speed * descending_factor

/-- Calculate the distance going down to the foot of the mountain -/
def distance_down : ℝ := descending_speed * time_down

/-- Prove that the distance going down to the foot of the mountain is 72 miles -/
theorem distance_descending_correct : distance_down = 72 := by
  sorry

end distance_descending_correct_l649_649010


namespace total_balls_l649_649332

def num_white : ℕ := 50
def num_green : ℕ := 30
def num_yellow : ℕ := 10
def num_red : ℕ := 7
def num_purple : ℕ := 3

def prob_neither_red_nor_purple : ℝ := 0.9

theorem total_balls (T : ℕ) 
  (h : prob_red_purple = 1 - prob_neither_red_nor_purple) 
  (h_prob : prob_red_purple = (num_red + num_purple : ℝ) / (T : ℝ)) :
  T = 100 :=
by sorry

end total_balls_l649_649332


namespace count_of_diverse_dates_in_2013_l649_649641

def is_diverse_date (d : Nat) (m : Nat) (y : Nat) : Bool :=
  let digits := ((d / 10) % 10) :: (d % 10) :: ((m / 10) % 10) :: (m % 10) :: ((y / 10) % 10) :: (y % 10) :: []
  List.sort (Nat.compare) digits == [0, 1, 2, 3, 4, 5]

def count_diverse_dates_in_year (year : Nat) : Nat :=
  if year != 2013 then 0 else
  List.length [⟨d, m⟩ | d ← List.range' 1 31, m ← List.range' 1 12, is_diverse_date d m year]

theorem count_of_diverse_dates_in_2013 :
  count_diverse_dates_in_year 2013 = 2 :=
  sorry

end count_of_diverse_dates_in_2013_l649_649641


namespace log_inequality_solution_set_l649_649096

theorem log_inequality_solution_set (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x, x ∈ {y | f a y < ∞} → ∃ M, f a x ≤ M):
  (∀ x : ℝ, log a (x - 1) > 0 ↔ 1 < x ∧ x < 2) :=
by
  sorry

end log_inequality_solution_set_l649_649096


namespace zero_in_interval_1_2_exists_l649_649412

def f (x : ℝ) : ℝ := -1 / x + Real.log x / Real.log 2

theorem zero_in_interval_1_2_exists :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := sorry

end zero_in_interval_1_2_exists_l649_649412


namespace sets_are_equal_l649_649455

def int : Type := ℤ  -- Redefine integer as ℤ for clarity

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem sets_are_equal : SetA = SetB := by
  -- implement the proof here
  sorry

end sets_are_equal_l649_649455


namespace trigonometric_identity_l649_649465

-- Define the conditions
def sin_alpha (α : Real) : Real := -3 / 5
def cos_alpha (α : Real) : Real := 4 / 5

-- Lean 4 statement to prove the question
theorem trigonometric_identity : 2 * (sin_alpha α) + (cos_alpha α) = -2 / 5 :=
by
  sorry

end trigonometric_identity_l649_649465


namespace cos450_eq_zero_l649_649730

theorem cos450_eq_zero (cos_periodic : ∀ x, cos (x + 360) = cos x) (cos_90_eq_zero : cos 90 = 0) :
  cos 450 = 0 := by
  sorry

end cos450_eq_zero_l649_649730


namespace parallel_lines_from_conditions_l649_649868

variable (a b : Line)
variable (α β : Plane)

theorem parallel_lines_from_conditions
  (h1 : a ⊥ α)
  (h2 : b ⊥ β)
  (h3 : α ∥ β) : a ∥ b :=
sorry

end parallel_lines_from_conditions_l649_649868


namespace sum_of_n_binom_coefficient_l649_649649

theorem sum_of_n_binom_coefficient :
  (∑ n in { n : ℤ | nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15}, n) = 28 := 
by
  sorry

end sum_of_n_binom_coefficient_l649_649649


namespace area_enclosed_by_region_l649_649393

open Real

def condition (x y : ℝ) := abs (2 * x + 2 * y) + abs (2 * x - 2 * y) ≤ 8

theorem area_enclosed_by_region : 
  (∃ u v : ℝ, condition u v) → ∃ A : ℝ, A = 16 := 
sorry

end area_enclosed_by_region_l649_649393


namespace pencils_ratio_l649_649920

theorem pencils_ratio (C J : ℕ) (hJ : J = 18) 
    (hJ_to_A : J_to_A = J / 3) (hJ_left : J_left = J - J_to_A)
    (hJ_left_eq : J_left = C + 3) :
    (C : ℚ) / (J : ℚ) = 1 / 2 :=
by
  sorry

end pencils_ratio_l649_649920


namespace heath_time_spent_l649_649878

variables (rows_per_carrot : ℕ) (plants_per_row : ℕ) (carrots_per_hour : ℕ) (total_hours : ℕ)

def total_carrots (rows_per_carrot plants_per_row : ℕ) : ℕ :=
  rows_per_carrot * plants_per_row

def time_spent (total_carrots carrots_per_hour : ℕ) : ℕ :=
  total_carrots / carrots_per_hour

theorem heath_time_spent
  (h1 : rows_per_carrot = 400)
  (h2 : plants_per_row = 300)
  (h3 : carrots_per_hour = 6000)
  (h4 : total_hours = 20) :
  time_spent (total_carrots rows_per_carrot plants_per_row) carrots_per_hour = total_hours :=
by
  sorry

end heath_time_spent_l649_649878


namespace primes_solution_l649_649392

theorem primes_solution :
  ∃ (p : Fin 13 → ℕ),
    (∀ i, Nat.Prime (p i)) ∧
    (p 0 ≤ p 1 ∧ p 1 ≤ p 2 ∧ p 2 ≤ p 3 ∧ p 3 ≤ p 4 ∧ p 4 ≤ p 5 ∧ p 5 ≤ p 6 ∧ p 6 ≤ p 7 ∧ p 7 ≤ p 8 ∧ p 8 ≤ p 9 ∧ p 9 ≤ p 10 ∧ p 10 ≤ p 11 ∧ p 11 ≤ p 12) ∧
    (∑ i in Finset.filter (λ j, j < 12) Finset.univ, (p i)^2 = (p 12)^2) ∧
    (∃ i, i < 13 ∧ p i = 2 * p 0 + p 8) ∧
    ((p = Fin.mk [2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 7, 13, 17] (by norm_num)) ∨
     (p = Fin.mk [2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 7, 29, 31] (by norm_num))) := 
sorry

end primes_solution_l649_649392


namespace evaluate_expression_l649_649401

theorem evaluate_expression : -25 - 7 * (4 + 2) = -67 := by
  sorry

end evaluate_expression_l649_649401


namespace sets_satisfying_union_l649_649625

open Set

theorem sets_satisfying_union :
  {B : Set ℕ | {1, 2} ∪ B = {1, 2, 3}} = { {3}, {1, 3}, {2, 3}, {1, 2, 3} } :=
by
  sorry

end sets_satisfying_union_l649_649625


namespace x_coordinate_l649_649839

variables (α x : ℝ)
constant pointP : ℝ → ℝ → Prop
axiom cos_alpha : cos α = - (sqrt 3) / 2
axiom terminal_side_condition : pointP x 2

theorem x_coordinate :
  x = -2 * sqrt 3 :=
sorry

end x_coordinate_l649_649839


namespace line_through_fixed_point_bisects_segment_l649_649431

noncomputable theory
open_locale classical

section ellipse_tangents

-- Define the line l: x - 2y - 20 = 0
def line_l (P : ℝ × ℝ) : Prop := P.1 - 2 * P.2 - 20 = 0

-- Define the ellipse \frac{x^2}{16} + \frac{y^2}{9} = 1
def ellipse (Q : ℝ × ℝ) : Prop := Q.1^2 / 16 + Q.2^2 / 9 = 1

-- Define the fixed point Q
def fixed_point_Q : ℝ × ℝ := (4/5, -9/10)

-- Main statements
theorem line_through_fixed_point
 (P : ℝ × ℝ) (M N : ℝ × ℝ) (hP : line_l P) (hM : ellipse M) (hN : ellipse N) 
 (tangent_PM : ∃ k : ℝ, M = (k * P.1, k * P.2)) 
 (tangent_PN : ∃ k : ℝ, N = (k * P.1, k * P.2)) 
 : ∃ k : ℝ, M.1 + N.1 = k * fixed_point_Q.1 ∧ M.2 + N.2 = k * fixed_point_Q.2 :=
 sorry

theorem bisects_segment
 (P : ℝ × ℝ) (M N : ℝ × ℝ) (hP : line_l P) (hM : ellipse M) (hN : ellipse N)
 (tangent_PM : ∃ k : ℝ, M = (k * P.1, k * P.2)) 
 (tangent_PN : ∃ k : ℝ, N = (k * P.1, k * P.2))
 (parallel_MN_l : ∃ m : ℝ, P.2 = (m * P.1 + 10))
 : (M.1 + N.1) / 2 = fixed_point_Q.1 ∧ (M.2 + N.2) / 2 = fixed_point_Q.2 :=
 sorry

end ellipse_tangents

end line_through_fixed_point_bisects_segment_l649_649431


namespace find_base_of_exponential_l649_649847

theorem find_base_of_exponential (a : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a ≠ 1) 
  (h₃ : a ^ 2 = 1 / 16) : 
  a = 1 / 4 := 
sorry

end find_base_of_exponential_l649_649847


namespace earth_surface_areas_needed_l649_649128

noncomputable def factorial (n : Nat) : Nat := match n with
  | 0 => 1
  | k+1 => (k+1) * factorial k

-- Radius of the Earth in kilometers
def r : ℝ := 6370

-- Surface area of the Earth in km²
def earth_surface_area : ℝ := 4 * Real.pi * (r ^ 2)

-- Area required for each permutation in km²
def area_per_permutation_km2 : ℝ := 24 * 10^(-10)

-- Total area required for all permutations in km²
def total_area_required : ℝ := factorial 24 * area_per_permutation_km2

-- Ratio of total area required to Earth's surface area
def ratio : ℝ := total_area_required / earth_surface_area

-- Statement to prove the ratio is approximately 2921000
theorem earth_surface_areas_needed:
  ratio ≈ 2921000 := sorry

end earth_surface_areas_needed_l649_649128


namespace decreasing_interval_for_function_l649_649110

theorem decreasing_interval_for_function :
  ∀ (f : ℝ → ℝ) (ϕ : ℝ),
  (∀ x, f x = -2 * Real.tan (2 * x + ϕ)) →
  |ϕ| < Real.pi →
  f (Real.pi / 16) = -2 →
  ∃ a b : ℝ, 
  a = 3 * Real.pi / 16 ∧ 
  b = 11 * Real.pi / 16 ∧ 
  ∀ x, a < x ∧ x < b → ∀ y, x < y ∧ y < b → f y < f x :=
by sorry

end decreasing_interval_for_function_l649_649110


namespace range_of_f_l649_649053

def max (a b : ℝ) : ℝ :=
if a ≥ b then a else b

def f (x : ℝ) : ℝ :=
max (2^(-x)) (-|x-1| + 2)

theorem range_of_f :
  {a : ℝ | ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a} = {a : ℝ | 1 < a ∧ a < 2} :=
sorry

end range_of_f_l649_649053


namespace squares_after_erasing_segments_l649_649389

/-- 
Given a 4x4 large square divided into 16 smaller 1x1 squares, and two line segments erased,
the total number of squares remaining is 22.
-/
theorem squares_after_erasing_segments : 
  ∀ (erase_segments : ℕ), erase_segments = 2 → 
  total_squares_after_erasing (4, 4) 16 erase_segments = 22 := 
by 
  intros 
  sorry

end squares_after_erasing_segments_l649_649389


namespace repeating_decimal_mul_l649_649379

theorem repeating_decimal_mul (x : ℝ) (hx : x = 0.3333333333333333) :
  x * 12 = 4 :=
sorry

end repeating_decimal_mul_l649_649379


namespace parabola_hyperbola_focus_l649_649103

theorem parabola_hyperbola_focus (b : ℝ) (h : b > 0) :
  let focus_parabola := (2, 0),
      focus_hyperbola := (√(1 + b^2), 0) in
  focus_parabola = focus_hyperbola →
  b = √3 :=
by
  intros
  sorry

end parabola_hyperbola_focus_l649_649103


namespace positive_terms_count_l649_649767

def sequence (n : ℕ) : ℝ := Float.cos (10.0^(n-1) * Float.pi / 180.0)

theorem positive_terms_count :
  (List.filter (λ n, sequence n > 0) (List.range' 1 100)).length = 99 :=
sorry

end positive_terms_count_l649_649767


namespace recycled_products_sold_l649_649135

-- Define the conditions
def student_recycled_products := 195
def teacher_recycled_products := 70
def pass_quality_check_percent := 0.80

-- Total recycled products made
def total_recycled_products := student_recycled_products + teacher_recycled_products

-- Prove the number of recycled products sold
theorem recycled_products_sold : total_recycled_products * pass_quality_check_percent = 212 :=
by
  sorry

end recycled_products_sold_l649_649135


namespace no_consecutive_natural_solution_l649_649534

theorem no_consecutive_natural_solution :
  ∀ (a b c d : ℕ), (a = b+1 ∧ b = c+1 ∧ c = d+1) ∨ (a = c+1 ∧ c = d+1 ∧ d = b+1) ∨
  (a = d+1 ∧ d = b+1 ∧ b = c+1) ∨ (b = d+1 ∧ d = c+1 ∧ c = a+1) ∨
  (b = a+1 ∧ a = d+1 ∧ d = c+1) ∨ (c = a+1 ∧ a = b+1 ∧ b = d+1) ∨
  (c = b+1 ∧ b = d+1 ∧ d = a+1) ∨ (d = a+1 ∧ a = c+1 ∧ c = b+1) ∨
  (d = b+1 ∧ b = a+1 ∧ a = c+1) → 
  (a + b) * (b + c) * (c + d) ≠ (c + a) * (a + d) * (d + b) :=
begin
  intros a b c d,
  intro h,
  have h1 : a = b+1 ∨ b=a+1, from sorry,
  have h2 : b = c+1 ∨ c=b+1, from sorry,
  have h3 : c = d+1 ∨ d=c+1, from sorry,
  contradiction
end

end no_consecutive_natural_solution_l649_649534


namespace positive_integers_l649_649821

def recurrence_sequence (a : ℕ → ℕ) :=
  a 1 = 1 ∧
  a 2 = 1 ∧
  ∀ n ≥ 1, a (n + 2) = (a (n + 1) ^ 2 + (-1) ^ (n - 1)) / a n

theorem positive_integers (a : ℕ → ℕ) (h : recurrence_sequence a) :
  ∀ n : ℕ, a n > 0 := 
sorry

end positive_integers_l649_649821


namespace congruent_triangles_parallel_lines_l649_649271

open EuclideanGeometry

variables {P A B C A1 B1 C1 : Point}
variables (ω : Circle)
variables (tri1 : Triangle) (tri2 : Triangle)

-- Problem conditions in Lean
axiom inscribed_triangle : Circumscribed ω tri1
axiom point_on_circle : OnCircle P ω
axiom parallel_through_P : ParallelLineThrough P tri1 A C A1 ∧
                           ParallelLineThrough P tri1 B A B1 ∧
                           ParallelLineThrough P tri1 C B C1

-- Define the corresponding triangles
def triangle_abc := Triangle.mk A B C
def triangle_a1b1c1 := Triangle.mk A1 B1 C1

-- State the problems to prove
theorem congruent_triangles (H1 : inscribed_triangle) 
                            (H2 : point_on_circle)
                            (H3 : parallel_through_P) :
  Congruent triangle_abc triangle_a1b1c1 :=
sorry

theorem parallel_lines (H1 : inscribed_triangle) 
                       (H2 : point_on_circle)
                       (H3 : parallel_through_P) :
  Parallel (Line.mk A A1) (Line.mk B B1) ∧ 
  Parallel (Line.mk B B1) (Line.mk C C1) ∧ 
  Parallel (Line.mk C C1) (Line.mk A A1) :=
sorry

end congruent_triangles_parallel_lines_l649_649271


namespace num_positive_cos_terms_l649_649754

def sequence (n : ℕ) : ℝ := Real.cos (10^(n-1) * Real.pi / 180)

theorem num_positive_cos_terms : (Finset.card (Finset.filter (λ n, 0 < sequence n) (Finset.range 100))) = 99 := 
sorry

end num_positive_cos_terms_l649_649754


namespace solve_a_minus_b_l649_649883

theorem solve_a_minus_b (a b : ℝ) (h1 : 2010 * a + 2014 * b = 2018) (h2 : 2012 * a + 2016 * b = 2020) : a - b = -3 :=
sorry

end solve_a_minus_b_l649_649883


namespace snow_shoveling_l649_649175

def Volume (length width depth : ℝ) : ℝ := length * width * depth

theorem snow_shoveling : 
  let V1 := Volume 25 2 (2 / 3) in
  let V2 := Volume 15 2 (1 / 2) in
  V1 + V2 = 145 / 3 := 
by
  sorry

end snow_shoveling_l649_649175


namespace determine_P_l649_649183

-- Definitions for terms in the conditions
def U : Finset ℕ := {1, 2, 3, 4}
def M (P : ℝ) : Finset ℕ := U.filter (λ x, x * x - 5 * x + P = 0)
def complement_U (P : ℝ) : Finset ℕ := U \ M P

-- Statement of the theorem
theorem determine_P (P : ℝ) (h : complement_U P = {2, 3}) : P = 4 :=
by sorry

end determine_P_l649_649183


namespace students_in_class_l649_649281

/-- Conditions:
1. 20 hands in Peter’s class, not including his.
2. Every student in the class has 2 hands.

Prove that the number of students in Peter’s class including him is 11.
-/
theorem students_in_class (hands_without_peter : ℕ) (hands_per_student : ℕ) (students_including_peter : ℕ) :
  hands_without_peter = 20 →
  hands_per_student = 2 →
  students_including_peter = (hands_without_peter + hands_per_student) / hands_per_student →
  students_including_peter = 11 :=
by
  intros h₁ h₂ h₃
  sorry

end students_in_class_l649_649281


namespace precisely_hundred_million_l649_649289

-- Defining the options as an enumeration type
inductive Precision
| HundredBillion
| Billion
| HundredMillion
| Percent

-- The given figure in billions
def givenFigure : Float := 21.658

-- The correct precision is HundredMillion
def correctPrecision : Precision := Precision.HundredMillion

-- The theorem to prove the correctness of the figure's precision
theorem precisely_hundred_million : correctPrecision = Precision.HundredMillion :=
by
  sorry

end precisely_hundred_million_l649_649289


namespace condition_C_for_D_condition_A_for_B_l649_649482

theorem condition_C_for_D (C D : Prop) (h : C → D) : C → D :=
by
  exact h

theorem condition_A_for_B (A B D : Prop) (hA_to_D : A → D) (hD_to_B : D → B) : A → B :=
by
  intro hA
  apply hD_to_B
  apply hA_to_D
  exact hA

end condition_C_for_D_condition_A_for_B_l649_649482


namespace min_steps_ensure_no_sum_2014_l649_649532

-- Define the initial set of numbers on the blackboard
def initial_numbers : List ℕ := List.range' 1 2013

-- Define the operation on the blackboard (erase two numbers and replace with their sum)
def perform_step (l : List ℕ) (a b : ℕ) : List ℕ :=
  if h : a ∈ l ∧ b ∈ l then
    let l' := l.erase a in
    let l'' := l'.erase b in
    a + b :: l''
  else l

-- Define the goal to prove the minimum number of steps
def min_steps_required : ℕ := 503

-- Proving that the minimum number of steps to ensure no sum equals 2014 is 503
theorem min_steps_ensure_no_sum_2014 :
  ∀ l, l = initial_numbers → 
  (∃ (steps : ℕ), steps = min_steps_required ∧
  no_remaining_sum_2014 (perform_steps steps l)) := sorry

end min_steps_ensure_no_sum_2014_l649_649532


namespace Penny_total_species_identified_l649_649922

/-- Penny identified 35 species of sharks, 15 species of eels, and 5 species of whales.
    Prove that the total number of species identified is 55. -/
theorem Penny_total_species_identified :
  let sharks_species := 35
  let eels_species := 15
  let whales_species := 5
  sharks_species + eels_species + whales_species = 55 :=
by
  sorry

end Penny_total_species_identified_l649_649922


namespace width_of_rectangle_is_4_l649_649254

-- Defining conditions 
variables {w : ℝ}  -- width of the rectangle

-- Defining the problem statement with the given conditions
def rectangle_problem (w : ℝ) : Prop :=
  let length := 3 * w in
  let area := length * w in
  area = 48

-- The goal is to prove that width w of the rectangle is 4 inches
theorem width_of_rectangle_is_4 (w : ℝ) (h : rectangle_problem w) : w = 4 :=
by sorry

end width_of_rectangle_is_4_l649_649254


namespace equilateral_triangle_bisectors_area_l649_649177

variables {P Q R T: Type}
variables {A B C D E F G : P}
variables [EuclideanGeometry P Q R T]

theorem equilateral_triangle_bisectors_area :
  (equilateral_triangle A B C) →
  (D ∈ (segment A B)) →
  (E ∈ (segment A C)) →
  (F ∈ (segment A E)) →
  (G ∈ (segment A D)) →
  interior_angle_bisectors D E F G →
  area (triangle D E F) + area (triangle D E G) ≤ area (triangle A B C) ∧
  ((area (triangle D E F) + area (triangle D E G) = area (triangle A B C)) ↔ (D = B ∧ E = C)) :=
by
  intro h₀ h₁ h₂ h₃ h₄ h₅
  sorry

end equilateral_triangle_bisectors_area_l649_649177


namespace arrangement_count_correct_l649_649622

def isValidArrangement (arrangement : List (List String)) : Bool :=
  (arrangement.length = 3) ∧
  (∀ cls ∈ arrangement, cls.length ≥ 1) ∧
  (∀ cls ∈ arrangement, ¬ ("Mathematics" ∈ cls ∧ "Science" ∈ cls))

def subjectPool : List String := ["Chinese", "Mathematics", "English", "Science"]

noncomputable def numberOfValidArrangements : Nat :=
  (List.permutations subjectPool).filter isValidArrangement |>.length

theorem arrangement_count_correct : numberOfValidArrangements = 30 := 
  sorry

end arrangement_count_correct_l649_649622


namespace jerry_pick_up_trays_l649_649541

theorem jerry_pick_up_trays : 
  ∀ (trays_per_trip trips trays_from_second total),
  trays_per_trip = 8 →
  trips = 2 →
  trays_from_second = 7 →
  total = (trays_per_trip * trips) →
  (total - trays_from_second) = 9 :=
by
  intros trays_per_trip trips trays_from_second total
  intro h1 h2 h3 h4
  sorry

end jerry_pick_up_trays_l649_649541


namespace binomial_sum_sum_of_n_values_l649_649650

theorem binomial_sum (n : ℕ) (h : nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15) : n = 13 ∨ n = 15 := sorry

theorem sum_of_n_values : ∑ n in {n | nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15}.to_finset, n = 28 :=
by
  apply finset.sum_eq_from_set,
  intros x hx,
  cases binomial_sum x hx,
  { simp [h], },
  { simp [h], }

end binomial_sum_sum_of_n_values_l649_649650


namespace Penelope_winning_strategy_l649_649303

theorem Penelope_winning_strategy :
  ∃ strategy : (ℕ → ℕ), 
    (∀ m, (strategy m > 0 ∧ strategy m ≤ 4)) ∧ 
    (∀ n, n ≡ 1 [MOD 5] → 
      ∀ h_choice (h_choice > 0 ∧ h_choice ≤ 4), 
         let p_choice := strategy (n - h_choice) in 
         (p_choice + h_choice ≡ 1 [MOD 5])) :=
sorry

end Penelope_winning_strategy_l649_649303


namespace cone_vertex_angle_l649_649640

/--
Two spheres with radius 4 touch each other externally. A cone touches both 
spheres externally and the table with its lateral surface. The distance
from the apex of the cone to the points where the spheres touch the table is 5. 
Prove that the vertex angle of the cone is either 90 degrees or 2 * arccot 4.
-/
theorem cone_vertex_angle {r : ℝ} (dist apex_points : ℝ) :
  r = 4 → dist = 5 →
  (∃ θ : ℝ, θ = 90 ∨ θ = 2 * real.arccot 4) :=
by
 -- Insert proof here
  sorry

end cone_vertex_angle_l649_649640


namespace part1_q1_part1_q2_part2_l649_649814

-- Proof for Part 1
theorem part1_q1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a + b) / 2 = 1 → (sqrt (a * b) < 1) := 
sorry

theorem part1_q2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∀ (a_n b_n : ℕ → ℝ), 
    a_n 1 = (a + b) / 2 ∧ b_n 1 = sqrt (a * b) ∧ 
    (∀ n, a_n (n + 1) = (a_n n + b_n n) / 2 ∧ b_n (n + 1) = sqrt (a_n n * b_n n)) →
    ∀ n, a_n n > b_n n → 
    ∃ (p : ℝ), ∀ n : ℕ, a_n (n + 1) - b_n (n + 1) < p * (a_n n - b_n n) ∧ p = 1 / 2 := 
sorry

-- Proof for Part 2
theorem part2 (c_1 d : ℝ) (hc1 : c_1 > 0) (S : ℕ → ℝ) (A : ℕ → ℝ)
  (S_4043_gt_0 : S 4043 > 0) (S_4044_lt_0 : S 4044 < 0) :
  ∃ (p : ℝ), p ∈ (1 + d / c_1)..1 := 
sorry

end part1_q1_part1_q2_part2_l649_649814


namespace courtyard_width_is_14_l649_649686

-- Given conditions
def length_courtyard := 24   -- 24 meters
def num_bricks := 8960       -- Total number of bricks

@[simp]
def brick_length_m : ℝ := 0.25  -- 25 cm in meters
@[simp]
def brick_width_m : ℝ := 0.15   -- 15 cm in meters

-- Correct answer
def width_courtyard : ℝ := 14

-- Prove that the width of the courtyard is 14 meters
theorem courtyard_width_is_14 : 
  (length_courtyard * width_courtyard) = (num_bricks * (brick_length_m * brick_width_m)) :=
by
  -- Lean proof will go here
  sorry

end courtyard_width_is_14_l649_649686


namespace train_passing_time_approx_7_2_l649_649360

/- Conditions -/
def train_length : ℝ := 120 -- in meters
def train_speed : ℝ := 68 -- in kmph
def man_speed : ℝ := 8 -- in kmph

noncomputable def relative_speed : ℝ := (train_speed - man_speed) * (1000 / 3600) -- convert kmph to m/s

-- The proof problem: Prove that the time taken for the train to pass the man is approximately 7.2 seconds
theorem train_passing_time_approx_7_2 :
  (train_length / relative_speed) ≈ 7.2 :=
sorry

end train_passing_time_approx_7_2_l649_649360


namespace prob_B_not_in_school_B_given_A_not_in_school_A_l649_649059

theorem prob_B_not_in_school_B_given_A_not_in_school_A :
  let people := {A, B, C, D}
  let schools := {A, B, C, D}
  let assignments := {ass | ∃ (f : people → schools), bijective f}
  let prob_A_not_in_A := (#(assignments \ {ass | ass A = A}) : ℚ) / (#assignments : ℚ)
  let prob_AB_not_in_AB := (#(assignments \ {ass | ass A = A ∨ ass B = B}) : ℚ) / (#assignments : ℚ)
  let prob_B_not_in_B_given_A_not_in_A := prob_AB_not_in_AB / prob_A_not_in_A
  prob_B_not_in_B_given_A_not_in_A = 7 / 9 := sorry

end prob_B_not_in_school_B_given_A_not_in_school_A_l649_649059


namespace a_formula_Sn_formula_l649_649095

noncomputable def a : ℕ → ℕ
| 0       := 0
| (n + 1) := 2 * (n + 1)

def a_is_arithmetic_sequence : Prop := ∀ n, a (n + 1) = a n + 2

theorem a_formula :
  a 1 = 2 ∧ a 1 + a 2 + a 3 = 12 → (∀ n, a n = 2 * n) :=
sorry

noncomputable def b (n : ℕ) : ℕ := a n + 2^n

noncomputable def S (n : ℕ) : ℕ :=
∑ k in finset.range n, b (k + 1)

theorem Sn_formula :
  (∀ n, a n = 2 * n) →
  (∀ n, S n = n^2 + n + 2^(n+1) - 2) :=
sorry

end a_formula_Sn_formula_l649_649095


namespace num_positive_terms_l649_649772

noncomputable def seq (n : ℕ) : ℝ := float.cos (10^((n - 1).to_real))

theorem num_positive_terms : fin 100 → seq 100 99 :=
sorry

end num_positive_terms_l649_649772


namespace find_matrix_N_l649_649803

open Matrix

variable (u : Fin 3 → ℝ)

def cross_product (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0]

-- Define vector v as the fixed vector in the problem
def v : Fin 3 → ℝ := ![7, 3, -9]

-- Define matrix N as the matrix to be found
def N : Matrix (Fin 3) (Fin 3) ℝ := ![![0, 9, 3], ![-9, 0, -7], ![-3, 7, 0]]

-- Define the requirement condition
theorem find_matrix_N :
  ∀ (u : Fin 3 → ℝ), (N.mulVec u) = cross_product v u :=
by
  sorry

end find_matrix_N_l649_649803


namespace imo1989_q3_l649_649088

theorem imo1989_q3 (a b : ℤ) (h1 : ¬ (∃ x : ℕ, a = x ^ 2))
                   (h2 : ¬ (∃ y : ℕ, b = y ^ 2))
                   (h3 : ∃ (x y z w : ℤ), x ^ 2 - a * y ^ 2 - b * z ^ 2 + a * b * w ^ 2 = 0 
                                           ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) :
                   ∃ (x y z : ℤ), x ^ 2 - a * y ^ 2 - b * z ^ 2 = 0 ∧ (x, y, z) ≠ (0, 0, 0) := 
sorry

end imo1989_q3_l649_649088


namespace volume_of_pyramid_l649_649611

-- Definitions based on given conditions
def pyramid_height : ℝ := 8
def cross_section_distance_from_apex : ℝ := 3
def cross_section_area : ℝ := 4

-- The volume of the pyramid to be proved
def expected_volume : ℝ := 2048 / 27

theorem volume_of_pyramid (h_pyramid_height : pyramid_height = 8)
                           (h_cross_section_distance_from_apex : cross_section_distance_from_apex = 3)
                           (h_cross_section_area : cross_section_area = 4) :
  (1 / 3) * (cross_section_area / ((cross_section_distance_from_apex / pyramid_height) ^ 2)) * pyramid_height = expected_volume := 
sorry

end volume_of_pyramid_l649_649611


namespace exists_positive_integer_m_l649_649672

noncomputable def d (g1 : ℝ) (r : ℝ) : ℝ := g1 * (r - 1)
noncomputable def a_n (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d
noncomputable def g_n (n : ℕ) (g1 : ℝ) (r : ℝ) : ℝ := g1 * (r ^ (n - 1))

theorem exists_positive_integer_m (a1 g1 : ℝ) (r : ℝ) (h0 : g1 ≠ 0) (h1 : a1 = g1) (h2 : a2 = g2)
(h3 : a_n 10 a1 (d g1 r) = g_n 3 g1 r) :
  ∀ (p : ℕ), ∃ (m : ℕ), g_n p g1 r = a_n m a1 (d g1 r) := by
  sorry

end exists_positive_integer_m_l649_649672


namespace coconut_grove_l649_649661

theorem coconut_grove (x : ℕ) :
  (40 * (x + 2) + 120 * x + 180 * (x - 2) = 100 * 3 * x) → 
  x = 7 := by
  sorry

end coconut_grove_l649_649661


namespace series_sum_l649_649950

noncomputable def compute_series (x : ℝ) (hx : x > 1) : ℝ :=
  ∑' n, 1 / (x ^ (3 ^ n) - x ^ (- 3 ^ n))

theorem series_sum (x : ℝ) (hx : x > 1) : compute_series x hx = 1 / (x - 1) :=
sorry

end series_sum_l649_649950


namespace abc_sum_is_12_l649_649976

theorem abc_sum_is_12
  (a b c : ℕ)
  (h : 28 * a + 30 * b + 31 * c = 365) :
  a + b + c = 12 :=
by
  sorry

end abc_sum_is_12_l649_649976


namespace find_equation_of_ellipse_area_of_triangle_constant_l649_649830

section ellipse_problem

-- Define the ellipse with given conditions
def ellipse (a b : ℝ) := ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

-- Condition: Ellipse passes through the point (0, √2)
def passes_through (a b : ℝ) := (0, Real.sqrt 2) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}

-- Condition: Ellipse has given eccentricity
def has_eccentricity (a b : ℝ) := Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2

-- Combined conditions
def conditions (a b : ℝ) := ellipse a b ∧ passes_through a b ∧ has_eccentricity a b

-- Goal (Ⅰ): Equation of ellipse C
theorem find_equation_of_ellipse : ∃ a b : ℝ, conditions a b → a = 2 ∧ b = Real.sqrt 2 ∧ ∀ x y : ℝ, x^2 / 4 + y^2 / 2 = 1 := 
by
  sorry

-- Additional definitions and setup for part (Ⅱ)
def is_vertex (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  (p = (-a, 0)) ∨ (p = (a, 0))

def is_point_on_ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  p ∈ {q : ℝ × ℝ | q.1^2 / a^2 + q.2^2 / b^2 = 1}

def area_of_triangle (O M N : ℝ × ℝ) : ℝ :=
  (1 / 2) * Real.abs (O.1 * (M.2 - N.2) + M.1 * (N.2 - O.2) + N.1 * (O.2 - M.2))

-- Goal (Ⅱ): Prove the area of triangle OMN is constant
theorem area_of_triangle_constant : ∀ a b : ℝ, conditions a b → ∀ P : ℝ × ℝ, is_point_on_ellipse a b P → P ≠ (-a, 0) ∧ P ≠ (a, 0) →
  ∀ O M N : ℝ × ℝ, O = (0, 0) → M.1 > 0 → N.1 < 0 → is_point_on_ellipse a b M → is_point_on_ellipse a b N →
  (area_of_triangle O M N = Real.sqrt 2) := 
by
  sorry

end ellipse_problem

end find_equation_of_ellipse_area_of_triangle_constant_l649_649830


namespace max_intersection_square_triangle_l649_649328

theorem max_intersection_square_triangle :
  ∀ (square_sides triangle_sides : ℕ), 
    square_sides = 4 → triangle_sides = 3 → 
    (∀ s ∈ set.univ, ∀ t ∈ set.univ, s ≠ t) →
    (square_sides * triangle_sides = 12) :=
begin
  intros square_sides triangle_sides hs ht h_parallel,
  rw hs,
  rw ht,
  show 4 * 3 = 12,
  norm_num,
end

end max_intersection_square_triangle_l649_649328


namespace smallest_a_inequality_l649_649427

noncomputable def smallest_a := 0.79

theorem smallest_a_inequality (x : ℝ) (hx : x ∈ Ioo (3 * π / 2) 2 * π) :
  (∛(sin x ^ 2) - ∛(cos x ^ 2)) / (∛(tan x ^ 2) - ∛(cot x ^ 2)) < smallest_a / 2 :=
begin
  sorry
end

end smallest_a_inequality_l649_649427


namespace max_min_f_cos_2x0_l649_649853

-- Define the function
def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2 - 1

-- Define the range
def x_range := Icc (0:ℝ) (π/2)

-- Problem 1: Maximum and minimum values of the function on [0, π/2]
theorem max_min_f : 
  ∃ a b, (∀ x ∈ x_range, f x ≤ a ∧ f x ≥ b) ∧ a = 2 ∧ b = -1 :=
sorry

-- Problem 2: Find the value of cos(2 * x0) when f(x0) = 6/5 and x0 ∈ [π/4, π/2]
def x0_range := Icc (π/4) (π/2)

theorem cos_2x0 (x0 : ℝ) (h0 : x0 ∈ x0_range) (hf : f x0 = 6 / 5) : 
  cos (2 * x0) = (3 - 4 * sqrt 3) / 10 :=
sorry

end max_min_f_cos_2x0_l649_649853


namespace integral_value_l649_649244

noncomputable def binomial_expansion_second_term_coeff (a : ℝ) : ℝ :=
  let coeff := 3 * |a| * (|a|^2) * (-((sqrt 3) / 6)) in coeff

theorem integral_value (a : ℝ) (h : binomial_expansion_second_term_coeff a = - (sqrt 3) / 2) :
  ∃ (s : ℝ), s = ∫ x in -2 .. a, x^2 ∧ (s = 3 ∨ s = 7 / 3) :=
by
  sorry

end integral_value_l649_649244


namespace num_positive_terms_l649_649748

def sequence_cos (n : ℕ) : ℝ :=
  cos (10^(n-1) * π / 180)

theorem num_positive_terms : (finset.filter (λ n : ℕ, 0 < sequence_cos n) (finset.range 100)).card = 99 :=
by
  sorry

end num_positive_terms_l649_649748


namespace largest_acute_angles_convex_octagon_l649_649305

theorem largest_acute_angles_convex_octagon :
  ∀ (angles : Finset ℝ), 
  angles.sum = 1080 ∧ (∀ θ ∈ angles, θ < 180) 
  → ∃ (acute_angles : Finset ℝ), 
    acute_angles.card = 4 ∧ ∀ θ ∈ acute_angles, θ < 90 :=
by
  sorry

end largest_acute_angles_convex_octagon_l649_649305


namespace ratio_of_areas_l649_649002

-- Define the coordinates of points on the squares
structure Point :=
  (x : ℚ)
  (y : ℚ)

-- Define the two squares
def A : Point := ⟨0, 0⟩
def B1 : Point := ⟨4, 0⟩
def C : Point := ⟨4, 4⟩
def D : Point := ⟨0, 4⟩
def G : Point := ⟨4, 3⟩
def E : Point := ⟨7, 0⟩
def F : Point := ⟨7, 3⟩

-- Define the intersection point P
def P : Point := ⟨4, 12/7⟩

noncomputable def area_triangle (A B C : Point) : ℚ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_ape : ℚ := area_triangle A P E
def area_gfp : ℚ := area_triangle G F P

theorem ratio_of_areas : area_ape / area_gfp = 28 / 9 :=
  sorry

end ratio_of_areas_l649_649002


namespace number_of_solutions_to_gg_x_eq_6_l649_649136

def g (x : ℝ) : ℝ :=
if x ≥ -3 then x^2 - 6 else x + 4

theorem number_of_solutions_to_gg_x_eq_6 :
  {x : ℝ | g (g x) = 6}.to_finset.card = 4 :=
by
  sorry

end number_of_solutions_to_gg_x_eq_6_l649_649136


namespace number_of_primes_in_sequence_is_zero_l649_649320

-- Definitions based on conditions
def Q : ℕ := list.prod (list.filter prime (list.range 68)) -- Product of all primes ≤ 67
def seq (n : ℕ) : ℕ := Q + n -- Sequence term

-- Proof that the number of primes in the sequence is 0
theorem number_of_primes_in_sequence_is_zero :
  (∃ n_vals : list ℕ, n_vals = list.range' 2 60 ∧
   ∀ n ∈ n_vals, ¬ (prime (seq n))) :=
by
  let n_vals := list.range' 2 60 -- List of n values from 2 to 61
  use n_vals
  split
  . exact rfl
  . intros n hn
    -- Proof omitted
    sorry

end number_of_primes_in_sequence_is_zero_l649_649320


namespace chinese_remainder_theorem_sequences_and_sum_l649_649910

theorem chinese_remainder_theorem_sequences_and_sum :
  (∀ n : ℕ, (∃ (k : ℕ), ((n - 1) % 7 = 0) ∧ (3 * (7 * k + 1) - 1 - 20 * (7 * k + 1) + 20 = n + 1)) ∧
            (21 * n - 19 = 3 * (7 * n + 1) - 1)) →
  let c : ℕ → ℕ := λ n, 21 * n - 19,
      d : ℕ → ℕ := λ n, (c n) - 20 * n + 20,
      S : ℕ → ℚ := λ n, ∑ i in finset.range n, 1 / ((d i) * (d (i + 1))),
      S_2023 := S 2023 in
  S_2023 = 2023 / 4050 :=
begin
  sorry
end

end chinese_remainder_theorem_sequences_and_sum_l649_649910


namespace polynomial_degree_is_969_l649_649696

/-- 
Prove that the degree of a polynomial with rational coefficients, 
which has the set {1 + sqrt 3, 2 + sqrt 5, ..., 500 + sqrt 1003} as roots, is 969.
-/
theorem polynomial_degree_is_969 :
  ∃ (p : Polynomial ℚ), (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 500 → 
    Polynomial.eval (n + Real.sqrt (n + 2)) p = 0 ∧ 
    Polynomial.degree p = 969) := 
sorry

end polynomial_degree_is_969_l649_649696


namespace melissa_total_time_l649_649209

-- Definitions based on the conditions in the problem
def time_replace_buckle : Nat := 5
def time_even_heel : Nat := 10
def time_fix_straps : Nat := 7
def time_reattach_soles : Nat := 12
def pairs_of_shoes : Nat := 8

-- Translation of the mathematically equivalent proof problem
theorem melissa_total_time : 
  (time_replace_buckle + time_even_heel + time_fix_straps + time_reattach_soles) * 16 = 544 :=
by
  sorry

end melissa_total_time_l649_649209


namespace range_of_a_l649_649070

noncomputable theory

def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then x^3 - 3*x^2 + 2*x + 1 else a*x - 2*a + 1

def line_eq (k : ℝ) (x : ℝ) (y : ℝ) : Prop :=
k*x - y - k + 1 = 0

theorem range_of_a:
  (∀ (k : ℝ), 0 < k ∧ k < 3 → 
    ∃ x1 x2 x3 : ℝ, 
      line_eq k x1 (f 6 x1) ∧ line_eq k x2 (f 6 x2) ∧ line_eq k x3 (f 6 x3) ∧ 
      x1 + x2 + x3 < 3) →
  (∀ (a : ℝ), a ≥ 6)→
  true :=
by sorry

end range_of_a_l649_649070


namespace sum_of_squares_of_roots_l649_649745

theorem sum_of_squares_of_roots :
  (∀ y : ℝ, y^3 - 7 * y^2 + 9 * y - 2 = 0 → y ≥ 0) →
  let roots : Finset ℝ := {r | r^3 - 7 * r^2 + 9 * r - 2 = 0 ∧ r ≥ 0}.to_finset in
  roots.sum (λ r, r^2) = 31 :=
by
  sorry

end sum_of_squares_of_roots_l649_649745


namespace determine_x_l649_649032

theorem determine_x (x y : ℝ) (h : x / (x - 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y + 1)) : 
  x = 2 * y^2 + 6 * y + 4 := 
by
  sorry

end determine_x_l649_649032


namespace four_x_plus_four_negx_l649_649457

theorem four_x_plus_four_negx (x : ℝ) (h : 2^x + 2^(-x) = 5) : 4^x + 4^(-x) = 23 := 
by 
  sorry  -- Proof is omitted as per the instructions.

end four_x_plus_four_negx_l649_649457


namespace angle_between_vectors_equals_pi_div_3_l649_649480

variable (a b : ℝ^3) -- Assume vectors a and b in ℝ^3
variable (theta : ℝ) -- Assume theta is the angle between a and b

-- Conditions given in the problem
axiom magn_a : ‖a‖ = 3
axiom magn_a_b : ‖a - b‖ = sqrt 13
axiom dot_a_b : a ⬝ b = 6 -- Dot product of a and b

-- Lean 4 statement for the proof problem
theorem angle_between_vectors_equals_pi_div_3 :
  theta = real.arccos (dot_a_b / (magn_a * ‖b‖)) :=
sorry

end angle_between_vectors_equals_pi_div_3_l649_649480


namespace relationship_among_abc_l649_649864

noncomputable def a : ℝ := Real.log (1 / 2) / Real.log 3
noncomputable def b : ℝ := (2 : ℝ) ^ (0.1)
noncomputable def c : ℝ := (0.9 : ℝ) ^ (3 / 2)

theorem relationship_among_abc : a < c ∧ c < b := by
  -- Proof is to be done here
  sorry

end relationship_among_abc_l649_649864


namespace problem_l649_649045

-- Definitions of the conditions:
def eight_digit_palindrome (n : ℕ) : Prop := 
  n / 10^7 % 10 = n % 10 ∧ 
  n / 10^6 % 10 = n / 10 % 10 ∧ 
  n / 10^5 % 10 = n / 10^2 % 10 ∧ 
  n / 10^4 % 10 = n / 10^3 % 10

def composed_of_digits_0_and_1 (n : ℕ) : Prop :=
  n / 10^7 % 10 ∈ {0, 1} ∧
  n / 10^6 % 10 ∈ {0, 1} ∧
  n / 10^5 % 10 ∈ {0, 1} ∧
  n / 10^4 % 10 ∈ {0, 1} ∧
  n / 10^3 % 10 ∈ {0, 1} ∧
  n / 10^2 % 10 ∈ {0, 1} ∧
  n / 10^1 % 10 ∈ {0, 1} ∧
  n % 10 ∈ {0, 1}

def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def prime_divisors_use_only_1_3_percent (n : ℕ) : Prop :=
  ∀ p, prime p → p ∣ n → (p = 1 ∨ p = 3)

-- Main theorem statement
theorem problem :
  eight_digit_palindrome 10111101 ∧
  composed_of_digits_0_and_1 10111101 ∧
  divisible_by_3 10111101 ∧
  prime_divisors_use_only_1_3_percent 10111101 :=
sorry

end problem_l649_649045


namespace correct_operation_l649_649316

theorem correct_operation (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by
  sorry

end correct_operation_l649_649316


namespace magnitude_OP_l649_649842

-- Definitions for the vectors and conditions
variables (e1 e2 : EuclideanSpace ℝ (Fin 2))
variable (OP : EuclideanSpace ℝ (Fin 2))
variable (angle_e1_e2 : Real.angle := Real.angle.pi / 3)
variable (e1_unit : ∥e1∥ = 1)
variable (e2_unit : ∥e2∥ = 1)
variable (dot_product : ⟪e1, e2⟫ = Real.cos (Real.angle.pi / 3))

-- Condition that OP is a linear combination of e1 and e2
variable (OP_eq : OP = 3 • e1 + 4 • e2)

-- The statement to be proved
theorem magnitude_OP : ∥OP∥ = Real.sqrt 37 :=
by
  -- Here we would write the proof, but we leave it as sorry for now
  sorry

end magnitude_OP_l649_649842


namespace solve_x_eq_neg_a_div_3_l649_649065

variable (a x : ℝ)

-- Assuming the condition a ≠ 0
theorem solve_x_eq_neg_a_div_3 (ha : a ≠ 0) :
  det ![
    ![x + a, x, x],
    ![x, x + a, x],
    ![x, x, x + a]
  ] = 0 ↔ x = -a / 3 :=
  sorry

end solve_x_eq_neg_a_div_3_l649_649065


namespace trigonometric_identity_l649_649837

theorem trigonometric_identity (α : ℝ) (h1 : -π / 2 < α ∧ α < 0) (h2 : sin α + cos α = 1 / 5) : 
  1 / (cos α ^ 2 - sin α ^ 2) = 25 / 7 :=
sorry

end trigonometric_identity_l649_649837


namespace cost_of_mango_lassi_l649_649005

theorem cost_of_mango_lassi 
  (cost_samosas : ℝ) 
  (cost_pakoras : ℝ) 
  (total_cost : ℝ) 
  (tip : ℝ) 
  (cost_mango_lassi : ℝ) : 
  cost_samosas = 6 → 
  cost_pakoras = 12 → 
  total_cost = 25 → 
  tip = 4.5 → 
  cost_mango_lassi = total_cost - (cost_samosas + cost_pakoras + tip) := 
by 
  intros hsamosas hpakoras htotal htip 
  rw [hsamosas, hpakoras, htotal, htip] 
  norm_num 
  done 


end cost_of_mango_lassi_l649_649005


namespace sin_cos_alpha_l649_649436

open Real

theorem sin_cos_alpha (α : ℝ) (h1 : sin (2 * α) = -sqrt 2 / 2) (h2 : α ∈ Set.Ioc (3 * π / 2) (2 * π)) :
  sin α + cos α = sqrt 2 / 2 :=
sorry

end sin_cos_alpha_l649_649436


namespace avg_speed_40_l649_649670

def average_speed_60_30 (D : ℝ) (T1 T2 : ℝ) : ℝ :=
  let speedXY := 60
  let speedYX := 30
  let total_distance := 2 * D
  let total_time := D / speedXY + D / speedYX
  total_distance / total_time

theorem avg_speed_40 (D : ℝ) (T1 T2 : ℝ) (hT1 : T1 = D / 60) (hT2 : T2 = D / 30) :
  average_speed_60_30 D T1 T2 = 40 :=
by
  sorry

end avg_speed_40_l649_649670


namespace max_employees_l649_649342

/-- 
The maximum number of employees in the company such that for every pair of employees, there 
are at least three days in a week when one is working and the other is not, is 16. 
-/
theorem max_employees (n : ℕ) 
  (h : ∀ (v : Fin n → Fin 2^7), ∀ i j : Fin n, i ≠ j → (Finset.card (Finset.filter (λ k, v i k ≠ v j k) Finset.univ) ≥ 3)) 
  : n ≤ 16 := 
sorry

end max_employees_l649_649342


namespace cube_difference_l649_649639

theorem cube_difference (x y : ℕ) (h₁ : x + y = 64) (h₂ : x - y = 16) : x^3 - y^3 = 50176 := by
  sorry

end cube_difference_l649_649639


namespace rajesh_monthly_savings_l649_649326

theorem rajesh_monthly_savings
  (salary : ℝ)
  (percentage_food : ℝ)
  (percentage_medicines : ℝ)
  (percentage_savings : ℝ)
  (amount_food : ℝ := percentage_food * salary)
  (amount_medicines : ℝ := percentage_medicines * salary)
  (remaining_amount : ℝ := salary - (amount_food + amount_medicines))
  (save_amount : ℝ := percentage_savings * remaining_amount)
  (H_salary : salary = 15000)
  (H_percentage_food : percentage_food = 0.40)
  (H_percentage_medicines : percentage_medicines = 0.20)
  (H_percentage_savings : percentage_savings = 0.60) :
  save_amount = 3600 :=
by
  sorry

end rajesh_monthly_savings_l649_649326


namespace find_y_coordinate_C_l649_649990

-- Definitions of the points and conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 0, y := 6 }
def D : Point := { x := 6, y := 6 }
def E : Point := { x := 6, y := 0 }

-- Conditions derived from the problem statement
def area_square_ABDE : ℝ := 36
def area_pentagon_ABCDE : ℝ := 72
def area_triangle_BCD (h : ℝ) : ℝ := 1/2 * 6 * (h - 6)

-- Theorem stating the y-coordinate of C
theorem find_y_coordinate_C (h : ℝ) (area_triangle_BCD h = 36) : h = 18 :=
by
  sorry

end find_y_coordinate_C_l649_649990


namespace Sk_recursive_l649_649182

theorem Sk_recursive (k : ℕ) (h₀ : k ≥ 3) : 
  let S_k := (finset.range (2*k - (k + 2) + 1)).sum (λ n, 1 / (n + (k + 2)) : ℚ) in
  let S_{k+1} := (finset.range (2*(k+1) - (k + 3) + 1)).sum (λ n, 1 / (n + (k + 3)) : ℚ) in
  S_{k+1} = S_k + 1/(2*k) + 1/(2*k + 1) - 1/(k+2) :=
sorry

end Sk_recursive_l649_649182


namespace total_students_in_class_l649_649280

def total_students (H : Nat) (hands_per_student : Nat) (consider_teacher : Nat) : Nat :=
  (H / hands_per_student) + consider_teacher

theorem total_students_in_class (H : Nat) (hands_per_student : Nat) (consider_teacher : Nat) 
  (H_eq : H = 20) (hands_per_student_eq : hands_per_student = 2) (consider_teacher_eq : consider_teacher = 1) : 
  total_students H hands_per_student consider_teacher = 11 := by
  sorry

end total_students_in_class_l649_649280


namespace probability_both_chosen_are_girls_l649_649341

theorem probability_both_chosen_are_girls (club_size boys girls : ℕ) (choose : ℕ → ℕ → ℕ) 
  (h_club_size : club_size = 12) 
  (h_boys : boys = 6) 
  (h_girls : girls = 6) 
  (h_choose : ∀ n k, choose n k = n.choose k) :
  choose girls 2 / choose club_size 2 = 5 / 22 := 
by
  rw [h_club_size, h_girls, h_choose, nat.choose, show club_size = 12 from h_club_size, show girls = 6 from h_girls]
  rw [nat.choose, show club_size = 12 from h_club_size, show girls = 6 from h_girls]
  sorry

end probability_both_chosen_are_girls_l649_649341


namespace find_number_l649_649324

theorem find_number (N Q : ℕ) (h1 : N = 11 * Q) (h2 : Q + N + 11 = 71) : N = 55 :=
by {
  sorry
}

end find_number_l649_649324


namespace determine_set_l649_649788

theorem determine_set (S : Finset ℕ) (h_nonempty : S.Nonempty) (h_positive : ∀ x ∈ S, 0 < x)
  (h_condition : ∀ i j ∈ S, (i + j) / Nat.gcd i j ∈ S) :
  S = {2} :=
sorry

end determine_set_l649_649788


namespace log_div_log_inv_l649_649654

theorem log_div_log_inv (x : ℝ) (hx : x > 0) : real.log 27 / real.log (1/27) = -1 := by
  sorry

end log_div_log_inv_l649_649654


namespace infinite_product_a_seq_l649_649022

noncomputable def a_seq : ℕ → ℝ
| 0       := 1 / 3
| (n + 1) := 1 + (a_seq n - 1) ^ 3

theorem infinite_product_a_seq : (∏ n : ℕ, a_seq n) = 3 / 5 :=
  sorry

end infinite_product_a_seq_l649_649022


namespace club_membership_l649_649256

theorem club_membership (n : ℕ) 
  (h1 : n % 10 = 6)
  (h2 : n % 11 = 6)
  (h3 : 150 ≤ n)
  (h4 : n ≤ 300) : 
  n = 226 := 
sorry

end club_membership_l649_649256


namespace geom_seq_q_value_l649_649980

theorem geom_seq_q_value (a_n : ℕ → ℤ) (q : ℤ) :
  (∀ n, a_n n = a_n 0 * q ^ n) ∧ (|q| > 1) ∧ 
  ({a_n 0, a_n 1, a_n 2, a_n 3} = {-72, -32, 48, 108}) 
  → (2 * q = -3) :=
by
  sorry

end geom_seq_q_value_l649_649980


namespace sum_of_n_binom_coefficient_l649_649648

theorem sum_of_n_binom_coefficient :
  (∑ n in { n : ℤ | nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15}, n) = 28 := 
by
  sorry

end sum_of_n_binom_coefficient_l649_649648


namespace prob_select_A_l649_649477

open Finset

theorem prob_select_A :
  let individuals := {A, B, C} : Finset _
  let choices := indivduals.powerset.filter (λ s, s.card = 2)
  let with_A := choices.filter (λ s, A ∈ s)
  (with_A.card : ℚ) / choices.card = 2 / 3 := 
by
  sorry

end prob_select_A_l649_649477


namespace area_of_triangle_AED_l649_649995

noncomputable def point : Type := (ℝ × ℝ)
noncomputable def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def triangle_area (A B C : point) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_AED :
  let A := (0, 0)
  let B := (4, 0)
  let C := (4, 5)
  let D := (0, 5)
  let E := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  distance A B = 4 →
  distance B C = 5 →
  E = (2, 2.5) →
  triangle_area A E D = 5 :=
by
  intros
  sorry

end area_of_triangle_AED_l649_649995


namespace molar_ratio_H2_CH4_l649_649876

noncomputable def gas_mixture_conditions
    (total_volume : ℝ)
    (volume_per_mole : ℝ)
    (deltaH_H2O : ℝ)
    (deltaH_CH4 : ℝ)
    (total_heat_release : ℝ)
    (moles_H2 : ℝ)
    (moles_CH4 : ℝ) : Prop :=
   let total_moles := total_volume / volume_per_mole in
   moles_H2 + moles_CH4 = total_moles ∧
   (285.8 * moles_H2) + (890 * moles_CH4) = total_heat_release

theorem molar_ratio_H2_CH4
    (total_volume : ℝ := 112)
    (volume_per_mole : ℝ := 22.4)
    (deltaH_H2O : ℝ := -571.6)
    (deltaH_CH4 : ℝ := -890)
    (total_heat_release : ℝ := 3695)
    (moles_H2 : ℝ := 1.25)
    (moles_CH4 : ℝ := 3.75) :
    gas_mixture_conditions total_volume volume_per_mole deltaH_H2O deltaH_CH4 total_heat_release moles_H2 moles_CH4 →
    (moles_H2 / moles_CH4 = 1 / 3) :=
by
  sorry

end molar_ratio_H2_CH4_l649_649876


namespace print_gift_wrap_price_l649_649346

theorem print_gift_wrap_price (solid_price : ℝ) (total_rolls : ℕ) (total_money : ℝ)
    (print_rolls : ℕ) (solid_rolls_money : ℝ) (print_money : ℝ) (P : ℝ) :
  solid_price = 4 ∧ total_rolls = 480 ∧ total_money = 2340 ∧ print_rolls = 210 ∧
  solid_rolls_money = 270 * 4 ∧ print_money = 1260 ∧
  total_money = solid_rolls_money + print_money ∧ P = print_money / 210 
  → P = 6 :=
by
  sorry

end print_gift_wrap_price_l649_649346


namespace julia_more_kids_on_monday_l649_649543

-- Definition of the problem statement
def playedWithOnMonday : ℕ := 6
def playedWithOnTuesday : ℕ := 5
def difference := playedWithOnMonday - playedWithOnTuesday

theorem julia_more_kids_on_monday : difference = 1 :=
by
  -- Proof can be filled out here.
  sorry

end julia_more_kids_on_monday_l649_649543


namespace f_9_eq_1_l649_649609

def f : ℝ → ℝ
| x := if -1 ≤ x ∧ x < 3 then 2 * x - 1 else f (x - 4)

theorem f_9_eq_1 : f 9 = 1 :=
by
  -- Placeholder for proof
  sorry

end f_9_eq_1_l649_649609


namespace sin_inequalities_triangle_l649_649226

theorem sin_inequalities_triangle (A B C : ℝ) (h1 : A + B + C = 180) : 
  (-2 : ℝ) ≤ sin (3 * A) + sin (3 * B) + sin (3 * C) ∧ sin (3 * A) + sin (3 * B) + sin (3 * C) ≤ (3 * Real.sqrt 3 / 2) := 
by 
  sorry

end sin_inequalities_triangle_l649_649226


namespace diverse_dates_2013_l649_649643

theorem diverse_dates_2013 : ∃(n : ℕ), n = 2 ∧ forall (d1 d2 m1 m2 : ℕ), 
  (d1, d2, m1, m2) ∈ {(2, 0, 0, 5), (2, 5, 0, 4)} ∨
  (d1, d2, m1, m2) ∈ {(2, 5, 1, 0)} → n = 2 
:= sorry

end diverse_dates_2013_l649_649643


namespace Peter_can_always_ensure_three_distinct_real_roots_l649_649671

noncomputable def cubic_has_three_distinct_real_roots (b d : ℝ) : Prop :=
∃ (a : ℝ), ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
  (r1 * r2 * r3 = -a) ∧ (r1 + r2 + r3 = -b) ∧ (r1 * r2 + r2 * r3 + r3 * r1 = -d)

theorem Peter_can_always_ensure_three_distinct_real_roots (b d : ℝ) :
  cubic_has_three_distinct_real_roots b d :=
sorry

end Peter_can_always_ensure_three_distinct_real_roots_l649_649671


namespace number_of_positive_terms_in_sequence_l649_649764

noncomputable def a (n : ℕ) : ℝ := Real.cos (10^n).toReal

theorem number_of_positive_terms_in_sequence : 
  (Finset.univ.filter (λ n : Fin 100, 0 < a (n + 1))).card = 99 := by
  sorry

end number_of_positive_terms_in_sequence_l649_649764


namespace triangle_problem_l649_649145

noncomputable def triangle_B_angle (a b c : ℝ) (cosC : ℝ) (h : a - (sqrt 2 / 2) * c = b * cosC) : Prop :=
  ∃ B : ℝ, B = π / 4

noncomputable def triangle_area (a b c : ℝ) (cosC : ℝ) (h : a = 4) (h2 : cosC = 7 * sqrt 2 / 10) : Prop :=
  let sinC := sqrt (1 - (7 * sqrt 2 / 10)^2)
  let b := (a / (sqrt ((a - (sqrt 2 / 2) * c) ^ 2))).sqrt
  1 / 2 * a * b * sinC = 1

theorem triangle_problem (a b c : ℝ) (cosC : ℝ)
  (h : a - (sqrt 2 / 2) * c = b * cosC)
  (h2 : a = 4) 
  (h3 : cosC = 7 * sqrt 2 / 10) :
  triangle_B_angle a b c cosC h ∧ triangle_area a b c cosC h2 h3 := 
begin
  split,
  { use π / 4, },
  { sorry, }
end

end triangle_problem_l649_649145


namespace square_perimeter_l649_649307

def side_length : ℕ := 9

theorem square_perimeter (s : ℕ) (h : s = side_length) : 4 * s = 36 :=
by {
  rw h,
  exact rfl,
}

end square_perimeter_l649_649307


namespace least_perimeter_triangle_DEF_l649_649899

noncomputable def triangle_DEF_least_perimeter (cos_D cos_E cos_F : ℝ) : ℝ := sorry

theorem least_perimeter_triangle_DEF : 
    ∀ (D E F : ℕ),
    (cos_D D = 1 / 2) →
    (cos_E E = 3 / 5) →
    (cos_F F = -1 / 8) →
    triangle_DEF_least_perimeter (cos_D D) (cos_E E) (cos_F F) = 33 :=
sorry

end least_perimeter_triangle_DEF_l649_649899


namespace smallest_a_l649_649185

noncomputable def a_b_conditions (a b : ℝ) :=
  a ≥ 0 ∧ b ≥ 0 ∧ ∀ (x : ℤ), sin (a * x + b) = sin (29 * x)

theorem smallest_a (a b : ℝ) (h : a_b_conditions a b) : 
  a = 10 * Real.pi - 29 :=
sorry

end smallest_a_l649_649185


namespace find_area_of_closed_figure_l649_649106

-- Define the binomial expression
def binomial_expr (x : ℝ) (a : ℝ) : ℝ := (x^2 - a / x)^6

-- Given condition on the coefficient
axiom coefficient_condition :
  ∀ (a : ℝ), (∃ (coeff : ℝ), coeff = -160 ∧ true)

-- Integrate to find the area
noncomputable def area_under_curve (a : ℝ) : ℝ :=
  ∫ (x : ℝ) in 2..4, (x - 1 - a / x)

-- Main theorem to be proved
theorem find_area_of_closed_figure :
  ∀ a : ℝ, (∃ (coeff : ℝ), coeff = -160 ∧ true) → 
  (area_under_curve 2) = 4 - 2 * ln 2 := 
sorry

end find_area_of_closed_figure_l649_649106


namespace function_inverse_l649_649463

variable (a : ℝ) (x y : ℝ)

-- Define the original function f(x) = log_a(x - 1)
def f (x : ℝ) := log a (x - 1)

-- Conditions from the problem
-- 1. Point (9, 3) lies on the graph of f
-- 2. a > 0
-- 3. a ≠ 1
theorem function_inverse :
  9 > 1 ∧
  a > 0 ∧
  a ≠ 1 ∧
  f 9 = 3
  → ∀ x : ℝ, f⁻¹ x = 2^x + 1 :=
begin
  sorry
end

end function_inverse_l649_649463


namespace tubes_per_tub_l649_649982

theorem tubes_per_tub 
  (tubes_per_person : ℕ) 
  (total_people : ℕ) 
  (total_tubs : ℕ) 
  (h1 : tubes_per_person = 3) 
  (h2 : total_people = 36) 
  (h3 : total_tubs = 6) : 
  (total_people / tubes_per_person) / total_tubs = 2 := 
by 
  intros 
  rw [h1, h2, h3] 
  norm_num

end tubes_per_tub_l649_649982


namespace sin_cos_eq_frac_l649_649051

theorem sin_cos_eq_frac (k : ℕ) (hk: 0 < k) :
  (sin (π / (3 * k)) + cos (π / (3 * k)) = 2 * real.sqrt k / 3) ↔ k = 4 := sorry

end sin_cos_eq_frac_l649_649051


namespace overlapping_shaded_area_l649_649522

noncomputable def semicircle_area (d : ℝ) : ℝ := (1 / 8) * Real.pi * d^2

theorem overlapping_shaded_area :
  let d := 7
  let diameters := [7, 7, 7, 7, 7, 7, 42]
  let semicircle_areas := diameters.map semicircle_area
  let total_area := semicircle_areas.sum
  (total_area = (851 / 8) * Real.pi) :=
by
  iterate diameters.length { sorry }
  exact (851 / 8) * Real.pi

end overlapping_shaded_area_l649_649522


namespace clients_using_radio_l649_649366

theorem clients_using_radio (total_clients T R M TR TM RM TRM : ℕ)
  (h1 : total_clients = 180)
  (h2 : T = 115)
  (h3 : M = 130)
  (h4 : TR = 75)
  (h5 : TM = 85)
  (h6 : RM = 95)
  (h7 : TRM = 80) : R = 30 :=
by
  -- Using Inclusion-Exclusion Principle
  have h : total_clients = T + R + M - TR - TM - RM + TRM :=
    sorry  -- Proof of Inclusion-Exclusion principle for these sets
  rw [h1, h2, h3, h4, h5, h6, h7] at h
  -- Solve for R
  sorry

end clients_using_radio_l649_649366


namespace does_not_uniquely_determine_isosceles_triangle_l649_649317

theorem does_not_uniquely_determine_isosceles_triangle (r : ℝ) :
  ∃ (Δ₁ Δ₂ : Triangle), (isosceles Δ₁ ∧ isosceles Δ₂ ∧ (circumradius Δ₁ = r) ∧ (circumradius Δ₂ = r) ∧ Δ₁ ≠ Δ₂) :=
sorry

end does_not_uniquely_determine_isosceles_triangle_l649_649317


namespace number_of_handshakes_l649_649153

theorem number_of_handshakes (n : ℕ) (h : n = 15) : (n * (n - 1)) / 2 = 105 := by
  rw h
  norm_num

end number_of_handshakes_l649_649153


namespace problem_statement_l649_649202

noncomputable def z1 : ℂ := (√3 / 2) + (1 / 2) * Complex.I
noncomputable def z2 : ℂ := 3 + 4 * Complex.I

theorem problem_statement : (Complex.abs (z1 ^ 2016)) / (Complex.abs (Complex.conj z2)) = 1 / 5 :=
by
  sorry

end problem_statement_l649_649202


namespace find_f_2016_l649_649140

-- Definition of f being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Conditions given in the problem
variables (f : ℝ → ℝ)
hypothesis is_odd : odd_function f
hypothesis functional_equation : ∀ x, f (x + 2) = -f (x)

-- Theorem statement to prove
theorem find_f_2016 : f 2016 = 0 :=
  sorry

end find_f_2016_l649_649140


namespace packed_tents_and_food_truck_arrangements_minimum_transportation_cost_l649_649675

-- Define the conditions
def total_items : ℕ := 320
def tents_more_than_food : ℕ := 80
def total_trucks : ℕ := 8
def type_A_tent_capacity : ℕ := 40
def type_A_food_capacity : ℕ := 10
def type_B_tent_capacity : ℕ := 20
def type_B_food_capacity : ℕ := 20
def type_A_cost : ℕ := 4000
def type_B_cost : ℕ := 3600

-- Questions to prove:
theorem packed_tents_and_food:
  ∃ t f : ℕ, t + f = total_items ∧ t = f + tents_more_than_food ∧ t = 200 ∧ f = 120 :=
sorry

theorem truck_arrangements:
  ∃ A B : ℕ, A + B = total_trucks ∧
    (A * type_A_tent_capacity + B * type_B_tent_capacity = 200) ∧
    (A * type_A_food_capacity + B * type_B_food_capacity = 120) ∧
    ((A = 2 ∧ B = 6) ∨ (A = 3 ∧ B = 5) ∨ (A = 4 ∧ B = 4)) :=
sorry

theorem minimum_transportation_cost:
  ∃ A B : ℕ, A = 2 ∧ B = 6 ∧ A * type_A_cost + B * type_B_cost = 29600 :=
sorry

end packed_tents_and_food_truck_arrangements_minimum_transportation_cost_l649_649675


namespace problem_solution_l649_649783

def point (α : Type) := prod α α

noncomputable def d_C (A B C: point ℕ) : ℕ :=
  (abs (fst A - fst C) + abs (snd A - snd C)) + (abs (fst B - fst C) + abs (snd B - snd C))

noncomputable def find_N : ℕ := 4036 -- given in the problem as the correct answer

theorem problem_solution : 
  ∃ (C : set (point ℕ)), (∀ (x y : ℕ), 0 < x ∧ 0 < y → (C (x, y) → d_C (4, 1) (2, 5) (x, y) = find_N)) ∧ C.finite ∧ C.card = 2023 :=
sorry

end problem_solution_l649_649783


namespace distance_to_school_correct_l649_649169

-- Let d (total distance to school) be 31.25 miles.
constant d : Real

-- Define the durations
def jogging_duration_morning := 15 / 60 -- In hours.
def biking_duration_morning := 30 / 60 -- In hours.
def bus_duration_morning := 2 - (jogging_duration_morning + biking_duration_morning)

-- Define the speeds
def jogging_speed_morning := 5 -- In miles per hour.
def biking_speed_morning := 10 -- In miles per hour.
def bus_speed_morning := 20 -- In miles per hour.

-- Define the distances
def jogging_distance_morning := jogging_speed_morning * jogging_duration_morning
def biking_distance_morning := biking_speed_morning * biking_duration_morning
def bus_distance_morning := bus_speed_morning * bus_duration_morning

-- Total distance calculation
def total_distance_morning := jogging_distance_morning + biking_distance_morning + bus_distance_morning

-- Define return trip durations and speeds
def bus_skateboard_duration_return := 1.25 -- In hours (1 hour and 15 minutes).
def walking_duration_return := jogging_duration_morning -- Equivalent to jogging duration in morning in hours.
def walking_speed_return := 3 -- In miles per hour.

-- Define walking distance in return
def walking_distance_return := walking_speed_return * walking_duration_return

-- Remaining distance for bus and skateboard
def bus_skateboard_distance_return := d - walking_distance_return

-- Define the proof statement
theorem distance_to_school_correct : total_distance_morning = 31.25 :=
by { sorry }

end distance_to_school_correct_l649_649169


namespace triangle_inequality_l649_649923

theorem triangle_inequality
  (a b c R : ℝ) (A B C P : ℝ) 
  (inside_triangle : P ∈ triangle A B C)
  (circumradius : circumradius A B C = R)
  (sides : sides_of_triangle A B C = (a, b, c)) :
  (PA / a^2) + (PB / b^2) + (PC / c^2) ≥ (1 / R) :=
sorry

end triangle_inequality_l649_649923


namespace no_such_function_exists_l649_649928

theorem no_such_function_exists :
  ¬(∃ f : ℝ → ℝ, (∃ M > 0, ∀ x, -M ≤ f x ∧ f x ≤ M) ∧ 
    (f 1 = 1) ∧ 
    (∀ x, x ≠ 0 → f (x + (1 / (x^2))) = f x + (f (1 / x))^2)) :=
by
  unfold not
  intro hf
  cases hf with f hf_props
  -- proof steps would follow here
  sorry

end no_such_function_exists_l649_649928


namespace ellipse_standard_form_l649_649806

theorem ellipse_standard_form 
  (m n : ℝ) 
  (h_m_pos : 0 < m) 
  (h_n_pos : 0 < n) 
  (h_neq : m ≠ n)
  (h1 : 2 * m + 2 * n * (√2)^2 = 1) 
  (h2 : (√2)^2 * m + (-√3)^2 * n = 1) :
  (∃ (a b : ℝ), (a = 8) ∧ (b = 4) ∧ (∀ x y : ℝ, (a ≠ 0) ∧ (b ≠ 0) ∧ (x^2 / a + y^2 / b = 1))) :=
begin
  sorry
end

end ellipse_standard_form_l649_649806


namespace geometric_series_sum_l649_649406

theorem geometric_series_sum : 
  (∑ k in finset.range 7, (1/2)^(k+1)) = 127 / 128 :=
by sorry

end geometric_series_sum_l649_649406


namespace cat_litter_cost_l649_649537

theorem cat_litter_cost 
    (container_weight : ℕ) (container_cost : ℕ)
    (litter_box_capacity : ℕ) (change_interval : ℕ) 
    (days_needed : ℕ) (cost : ℕ) :
  container_weight = 45 → 
  container_cost = 21 → 
  litter_box_capacity = 15 → 
  change_interval = 7 →
  days_needed = 210 → 
  cost = 210 :=
by
  intros h1 h2 h3 h4 h5
  /- Here we would add the proof steps, but this is not required. -/
  sorry

end cat_litter_cost_l649_649537


namespace find_resistance_y_l649_649518

-- Define conditions
variables (r : ℝ) (x : ℝ) (y : ℝ)

-- Given conditions
def given_conditions : Prop :=
  r = 2.727272727272727 ∧ x = 5 ∧ (1 / r) = (1 / x) + (1 / y)

-- Theorem statement
theorem find_resistance_y (h : given_conditions r x y) : | y - 6 | < 1 :=
by
  sorry

end find_resistance_y_l649_649518


namespace sum_infinite_series_result_l649_649958

noncomputable def sum_infinite_series (x : ℝ) (h : 1 < x) : ℝ :=
  ∑' n, 1 / (x^(3^n) - x^(-3^n))

theorem sum_infinite_series_result (x : ℝ) (h : 1 < x) :
  sum_infinite_series x h = 1 / (x - 1) :=
sorry

end sum_infinite_series_result_l649_649958


namespace max_ratio_convergence_l649_649560
open ProbabilityTheory

noncomputable def xi_seq (ξ : ℕ → ℝ) (i : ℕ) : Prop :=
  ∀ n : ℕ, ∀ ε > 0, n * ennreal.of_real (measure_theory.measure.prob {ω | abs (ξ i ω) > ε * f n}) = o(1).

noncomputable def xi_max_ratio_conv (ξ : ℕ → ℝ) (f : ℕ → ℝ) : Prop :=
  ∀ ε > 0, limsup n (n * pr {ω | abs (ξ i ω) > ε * f n}) = 0.

theorem max_ratio_convergence 
  (ξ : ℕ → ℝ) (f : ℕ → ℝ)
  (indep_ident : ∀ i, xi_seq ξ i)
  (lim_seq : ∀ n (ε > 0), n * pr {ω | abs (ξ 1 ω) > ε * f n} = o(1)) :
    xi_max_ratio_conv ξ f :=
sorry

end max_ratio_convergence_l649_649560


namespace numeral_in_150th_decimal_place_l649_649653

theorem numeral_in_150th_decimal_place (n m : ℕ) (h₀ : n = 3) 
  (h₁ : m = 11) (h₂ : ∃ l : ℕ, l > 0 ∧ (∃ r : ℚ, r = n / m ∧ r.denom = 1))
  (h₃ : (decimal_expansion (n / m) = some (0, repeat_cycle seq 150)))
  (h₄ : seq = "272727") : 
  decimal_digit (n / m) 150 = '7' :=
sorry

end numeral_in_150th_decimal_place_l649_649653


namespace positive_terms_count_l649_649777

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then real.cos (1 * real.pi / 180) else real.cos (nat.pow 10 (n - 1) * real.pi / 180)

theorem positive_terms_count : finset.card (finset.filter (λ n, 0 < sequence n) (finset.range 100)) = 99 :=
by sorry

end positive_terms_count_l649_649777


namespace angle_C_magnitude_range_2a_plus_b_l649_649160

variables {A B C : ℝ} {a b c : ℝ}
variables (sin_A sin_B sin_C cos_C : ℝ)

/-- Conditions of the problem -/
def conditions : Prop := 
(A + B + C = π) ∧
(c = sqrt 3) ∧ 
(∃ m n : vector ℝ 2,
   m = vector.mk (sin_B + sin_C) (sin_A + sin_B) ∧ 
   n = vector.mk (sin_B - sin_C) sin_A ∧ 
   (vector.dot_product m n = 0))

/-- Part (1): Find the magnitude of angle C, given the conditions -/
theorem angle_C_magnitude (h : conditions) : C = 2 * π / 3 := 
sorry

/-- Part (2): Determine the range of possible values for 2a + b, given the conditions -/
theorem range_2a_plus_b (h : conditions) : 
  sqrt 3 < 2 * a + b ∧ 2 * a + b < 2 * sqrt 3 := 
sorry

end angle_C_magnitude_range_2a_plus_b_l649_649160


namespace number_of_positive_terms_in_sequence_l649_649762

noncomputable def a (n : ℕ) : ℝ := Real.cos (10^n).toReal

theorem number_of_positive_terms_in_sequence : 
  (Finset.univ.filter (λ n : Fin 100, 0 < a (n + 1))).card = 99 := by
  sorry

end number_of_positive_terms_in_sequence_l649_649762


namespace range_of_m_l649_649887

theorem range_of_m (m : ℝ) : (∀ (x : ℝ), |3 - x| + |5 + x| > m) → m < 8 :=
sorry

end range_of_m_l649_649887


namespace prove_max_days_hair_length_l649_649218

noncomputable def max_days_hair_length : ℕ :=
  let init_strands : ℕ := 200000
  let init_length : ℝ := 5
  let daily_growth : ℝ := 0.05
  let daily_loss : ℕ := 50
  let total_length (x : ℕ) : ℝ := (init_strands - daily_loss * x) * (init_length + daily_growth * x)
  let quadratic_expression := -2.5 * ((x : ℝ) - 1950) ^ 2 + 10506250
  1950

-- We state the theorem/proof problem.
theorem prove_max_days_hair_length : max_days_hair_length = 1950 :=
by
  sorry

end prove_max_days_hair_length_l649_649218


namespace total_carrots_l649_649219

-- Define the number of carrots grown by Sally and Fred
def sally_carrots := 6
def fred_carrots := 4

-- Theorem: The total number of carrots grown by Sally and Fred
theorem total_carrots : sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l649_649219


namespace triangle_problem_l649_649180

noncomputable def PA_squared (P A : ℝ^3) (R : ℝ) : ℝ := 
  2 * R^2 - 2 * (A • P)

noncomputable def PH_squared (P H : ℝ^3) (R : ℝ) : ℝ :=
  2 * R^2 - 2 * (H • P) -- assuming similar form for orthocenter H

noncomputable def PN_squared (P N : ℝ^3) (R : ℝ) (N_dot : ℝ) : ℝ :=
  R^2 + (1/2)^2 * (R^2 + R^2 + R^2) - R * N_dot

noncomputable def question_value (A B C P H N : ℝ^3) (a b c R : ℝ) (N_dot : ℝ ) : ℝ :=
  PA_squared P A R + PA_squared P B R + PA_squared P C R - PH_squared P H R - PN_squared P N R N_dot

theorem triangle_problem (A B C P H N : ℝ^3) (a b c R r N_dot: ℝ) : 
  question_value A B C P H N a b c R N_dot = a^2 + b^2 + c^2 - 9 * R^2 / 2 - R * N_dot :=
  sorry

end triangle_problem_l649_649180


namespace range_of_a_l649_649855

noncomputable def f (x : ℝ) : ℝ := 
  log 3 (3^(x - 1) + 3) - (1 / 2) * x

theorem range_of_a (a : ℝ) (h : f (a - 1) ≥ f (2 * a + 1)) : -2 ≤ a ∧ a ≤ 4 / 3 :=
by
  sorry

end range_of_a_l649_649855


namespace loss_percentage_second_article_l649_649357

-- Definitions based on conditions
def cost_price_each_article : ℝ := 1000
def selling_price_first_article : ℝ := cost_price_each_article * 1.10
def net_profit_loss_percent : ℝ := 99.99999999999946 / 100 
def total_cost_price : ℝ := 2 * cost_price_each_article 
def total_selling_price : ℝ := total_cost_price * (1 + net_profit_loss_percent)

-- Correct answer (to be proved): The loss percentage on the second article is 10%
theorem loss_percentage_second_article :
  let selling_price_second_article := total_selling_price - selling_price_first_article,
      loss := cost_price_each_article - selling_price_second_article,
      loss_percent := (loss / cost_price_each_article) * 100  in
  loss_percent = 10 :=
by
  sorry

end loss_percentage_second_article_l649_649357


namespace students_in_class_l649_649282

/-- Conditions:
1. 20 hands in Peter’s class, not including his.
2. Every student in the class has 2 hands.

Prove that the number of students in Peter’s class including him is 11.
-/
theorem students_in_class (hands_without_peter : ℕ) (hands_per_student : ℕ) (students_including_peter : ℕ) :
  hands_without_peter = 20 →
  hands_per_student = 2 →
  students_including_peter = (hands_without_peter + hands_per_student) / hands_per_student →
  students_including_peter = 11 :=
by
  intros h₁ h₂ h₃
  sorry

end students_in_class_l649_649282


namespace movie_ticket_cost_l649_649983

/--
Movie tickets cost a certain amount on a Monday, twice as much on a Wednesday, and five times as much as on Monday on a Saturday. If Glenn goes to the movie theater on Wednesday and Saturday, he spends $35. Prove that the cost of a movie ticket on a Monday is $5.
-/
theorem movie_ticket_cost (M : ℕ) 
  (wednesday_cost : 2 * M = 2 * M)
  (saturday_cost : 5 * M = 5 * M) 
  (total_cost : 2 * M + 5 * M = 35) : 
  M = 5 := 
sorry

end movie_ticket_cost_l649_649983


namespace projection_vector_correct_l649_649790

variables {R : Type*} [LinearOrderedField R]

noncomputable def projection_vector : ℝ × ℝ :=
  let p : ℝ × ℝ := (16 / 5, 8 / 5) in
  p

theorem projection_vector_correct :
  ∃ t : ℝ, (∃ v : ℝ × ℝ, v = (5, -2) + t • (-3, 6)) ∧ (v.1 = 16 / 5) ∧ (v.2 = 8 / 5)
  ∧ inner v (-3, 6) = 0 :=
begin
  sorry
end

end projection_vector_correct_l649_649790


namespace pyramid_height_cannot_lie_inside_l649_649398

-- Definitions representing the conditions
structure Triangle :=
(base: ℝ) (height: ℝ) (hypotenuse: ℝ)
  (right_angle_at_base: Prop)

-- Definition of a pyramid with right-angle triangles as lateral faces
structure Pyramid :=
(base: Fin n → Triangle) -- Each lateral face is a right triangle
(right_angle_at_base : ∀ i, (i < n) → Pyramid.base(i).right_angle_at_base)

-- The statement that the height's position within the pyramid
theorem pyramid_height_cannot_lie_inside (n : ℕ) (p : Pyramid n) :
  ∃ i, Pyramid.base(p) i ∧ (p.height i).lies_on_side_face :=
sorry

end pyramid_height_cannot_lie_inside_l649_649398


namespace sum_f_values_final_result_l649_649021

noncomputable def f : ℝ → ℝ
| x := if -3 ≤ x ∧ x < -1 then -(x + 2)^2 else if -1 ≤ x ∧ x < 3 then x else sorry

theorem sum_f_values :
    f 1 + f 2 + f 3 + f 4 + f 5 + f 6 = 1 :=
sorry

theorem final_result :
    (∑ i in finset.range 2012, f (i.succ)) = 338 :=
begin
  have period: ∀ x, f (x + 6) = f x, by sorry,
  have base_sum: f 1 + f 2 + f 3 + f 4 + f 5 + f 6 = 1, by apply sum_f_values,
  -- More steps to use period and base_sum to reach the final result
  sorry
end

end sum_f_values_final_result_l649_649021


namespace incorrect_inequality_given_conditions_l649_649137

variable {a b x y : ℝ}

theorem incorrect_inequality_given_conditions 
  (h1 : a > b) (h2 : x > y) : ¬ (|a| * x > |a| * y) :=
sorry

end incorrect_inequality_given_conditions_l649_649137


namespace positive_terms_count_l649_649765

def sequence (n : ℕ) : ℝ := Float.cos (10.0^(n-1) * Float.pi / 180.0)

theorem positive_terms_count :
  (List.filter (λ n, sequence n > 0) (List.range' 1 100)).length = 99 :=
sorry

end positive_terms_count_l649_649765


namespace attendance_rate_correct_l649_649726

def total_students : ℕ := 50
def students_on_leave : ℕ := 2
def given_attendance_rate : ℝ := 96

theorem attendance_rate_correct :
  ((total_students - students_on_leave) / total_students * 100 : ℝ) = given_attendance_rate := sorry

end attendance_rate_correct_l649_649726


namespace pyramid_vertices_l649_649498

-- Define the properties of a pyramid with a polygonal base having n sides and triangular sides
def pyramid_properties (n : ℕ) : Prop :=
  let E := 2 * n in -- Total edges of the pyramid
  (E = 10) → (n + 1 = 6)

theorem pyramid_vertices : (∃ n : ℕ, pyramid_properties n) :=
by
  -- Here we are stating the existence of such an n which satisfies the condition
  sorry

end pyramid_vertices_l649_649498


namespace divides_iterate_l649_649993

variable {α : Type*}

def divides (x f: α) [Semiring α] : Prop := ∃ q : α, f = x * q

theorem divides_iterate (f : α → α) (x : α) (k : ℕ) :
  divides x (f^[k] x) ↔ divides x (f x) :=
sorry

end divides_iterate_l649_649993


namespace length_of_EF_l649_649907

-- Define the given conditions
def Rectangle (A B C D : Type) (AB BC : ℝ) (h_AB: AB = 8) (h_BC: BC = 10) : Type :=
  {ae: ℝ // ae = 6} -- point E on AB such that AE = 6 cm
  {ef: ℝ // ef = 0} -- point F which will be determined by the conditions

-- Define a right triangle with perpendicular EF to AE
def RightTriangle (AE E F : ℝ) (AE_len: AE = 6) (EF_perpendicular: EF = ef) : Prop :=
  ∃ (a b : ℝ), a = 8 * 10 / 2 ∧ b = 0

-- Translate the solution as a proof problem in Lean
-- Prove that the length of EF is 40/3 cm
theorem length_of_EF {A B C D : Type} {AB BC : ℝ} (h_AB: AB = 8) (h_BC: BC = 10)
  (ae: Rectangle AB BC h_AB h_BC) (ef: RightTriangle ae 6 ef) :
  (ef = 40 / 3) :=
by
  sorry

end length_of_EF_l649_649907


namespace surface_area_of_square_prism_on_sphere_l649_649077

noncomputable def surface_area_of_prism (R : ℝ) (h : ℝ) : ℝ := 
  2 + 4 * 1 * h

theorem surface_area_of_square_prism_on_sphere (R : ℝ) 
  (H : (R / 2)^2 + 3 = R^2)
  (h : ℝ)
  (H_h : 2 * 1^2 + h^2 = 4^2) :
  surface_area_of_prism R h = 4 * ℝ.sqrt 14 + 2 :=
by
  sorry

end surface_area_of_square_prism_on_sphere_l649_649077


namespace not_always_similar_remaining_parts_l649_649793

noncomputable def similar (a b c d e f : ℝ) : Prop :=
  -- Define the criteria for similarity of two triangles (ratios of sides)
  a / b = d / e ∧ a / c = d / f ∧ b / c = e / f

theorem not_always_similar_remaining_parts (a b c d e f : ℝ) (h₁ : similar a b c d e f)
  (abe₁ ab'c₁ : ℝ) (abe₂ ab'c₂ : ℝ) (def₁ d'e'f₁ : ℝ) (def₂ d'e'f₂ : ℝ)
  (h₂ : similar abe₁ abe₂ def₁ def₂)
  : ¬ (similar ab'c₁ ab'c₂ d'e'f₁ d'e'f₂) := 
sorry

end not_always_similar_remaining_parts_l649_649793


namespace worker_b_time_l649_649319

theorem worker_b_time (time_A : ℝ) (time_A_B_together : ℝ) (T_B : ℝ) 
  (h1 : time_A = 8) 
  (h2 : time_A_B_together = 4.8) 
  (h3 : (1 / time_A) + (1 / T_B) = (1 / time_A_B_together)) :
  T_B = 12 :=
sorry

end worker_b_time_l649_649319


namespace money_allocation_l649_649523

theorem money_allocation (x y : ℝ) (h1 : x + 1/2 * y = 50) (h2 : y + 2/3 * x = 50) : 
  x + 1/2 * y = 50 ∧ y + 2/3 * x = 50 :=
by
  exact ⟨h1, h2⟩

end money_allocation_l649_649523


namespace prove_tangent_if_midpoint_l649_649441

-- Define the circles, points and lines
variable {Γ₁ Γ₂: Type*} [MetricSpace Γ₁] [MetricSpace Γ₂] -- Circles Γ₁ and Γ₂
variable {A B D E F: Prop} -- Points A, B, D, E, F
variable {tangent : Prop → Prop → Prop} -- Tangency definition
variable {midpoint : Prop → Prop → Prop → Prop} -- Midpoint definition

-- Conditions
variable (A_on_Gamma1 : A = OnCircle Γ₁)
variable (B_on_Gamma1 : B = OnCircle Γ₁)
variable (Center_Gamma2_on_Gamma1 : center(Γ₂) = OnCircle Γ₁)
variable (Gamma2_tangent_AB_at_B : tangent Γ₂ (LineSegment AB) ∧ PointOfTangency(Γ₂, LineSegment AB, B))
variable (Line_A_DE : LineThrough A (Intersections Γ₂ D E))
variable (Line_BD_intersects_Gamma1_at_F : Intersects (Line BD) Γ₁ F ∧ F ≠ B)

-- Theorem to prove that BE is tangent to Γ₁ at B if and only if D is the midpoint of BF
theorem prove_tangent_if_midpoint (
  h1 : Conditions A_on_Gamma1 B_on_Gamma1 Center_Gamma2_on_Gamma1 Gamma2_tangent_AB_at_B Line_A_DE Line_BD_intersects_Gamma1_at_F)
  : (tangent(Lines BE) Γ₁ B) ↔ midpoint B D F := 
sorry

end prove_tangent_if_midpoint_l649_649441


namespace hypotenuse_length_l649_649514

theorem hypotenuse_length (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a^2 + b^2 + c^2 = 1800) : 
  c = 30 :=
sorry

end hypotenuse_length_l649_649514


namespace total_population_eq_61t_l649_649515

variable (t g b : ℕ)

-- Conditions as definitions
def condition1 : b = 4 * g := sorry
def condition2 : g = 12 * t := sorry

-- The theorem to prove
theorem total_population_eq_61t (t : ℕ) (G : g = 12 * t) (B : b = 4 * g) : b + g + t = 61 * t :=
by
  rwa [B, G]

end total_population_eq_61t_l649_649515


namespace different_color_different_weight_l649_649037

theorem different_color_different_weight
  (exists_red_ball : ∃ b : Ball, b.color = Color.red)
  (exists_blue_ball : ∃ b : Ball, b.color = Color.blue)
  (exists_weight1_ball : ∃ b : Ball, b.weight = 1)
  (exists_weight2_ball : ∃ b : Ball, b.weight = 2) :
  ∃ b1 b2 : Ball, b1.color ≠ b2.color ∧ b1.weight ≠ b2.weight :=
sorry

structure Ball where
  color : Color
  weight : ℕ

inductive Color where
  | red
  | blue
  deriving DecidableEq, Repr

end different_color_different_weight_l649_649037


namespace value_of_f_at_2011_l649_649192

theorem value_of_f_at_2011 (a b c : ℝ) :
    let f := λ x : ℝ, a * x^5 + b * x^3 + c * x + 7 in
    f (-2011) = -17 → f (2011) = 31 :=
by 
    let f := λ x : ℝ, a * x^5 + b * x^3 + c * x + 7
    intro h
    sorry

end value_of_f_at_2011_l649_649192


namespace phase_shift_of_cosine_l649_649050

theorem phase_shift_of_cosine :
  ∀ (A B C : ℝ), A = 5 → B = 2 → C = π / 3 → 
  phase_shift (λ x, A * cos (B * x - C)) = π / 6 :=
begin
  intros A B C hA hB hC,
  -- The proof portion is omitted
  sorry
end

end phase_shift_of_cosine_l649_649050


namespace log_property_l649_649272

theorem log_property : 2 * log 2 + log 25 = 2 := by
  sorry

end log_property_l649_649272


namespace range_of_a_l649_649111

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log (x + 1) - x^2

theorem range_of_a (a : ℝ) (p q : ℝ) (h₀ : p ≠ q) (h₁ : -1 < p ∧ p < 0) (h₂ : -1 < q ∧ q < 0) :
  (∀ p q : ℝ, -1 < p ∧ p < 0 → -1 < q ∧ q < 0 → p ≠ q → (f a (p + 1) - f a (q + 1)) / (p - q) > 1) ↔ (6 ≤ a) :=
by
  -- proof is omitted
  sorry

end range_of_a_l649_649111


namespace line_equation_to_slope_intercept_slope_intercept_pair_l649_649347

theorem line_equation_to_slope_intercept :
  ∀ (x y : ℝ), (2 * (x - 3) - 1 * (y + 4) = 0) ↔ (y = 2 * x - 10) :=
by
  intro x y
  apply Iff.intro
  . intro h
    -- forward direction proof, which is skipped
    sorry 
  . intro h
    -- backward direction proof, which is skipped
    sorry

theorem slope_intercept_pair :
  (2, -10) = (2 : ℝ, -10 : ℝ) :=
by
  rfl

end line_equation_to_slope_intercept_slope_intercept_pair_l649_649347


namespace salary_increase_gt_90_percent_l649_649565

theorem salary_increase_gt_90_percent (S : ℝ) : 
  (S * (1.12^6) - S) / S > 0.90 :=
by
  -- Here we skip the proof with sorry
  sorry

end salary_increase_gt_90_percent_l649_649565


namespace convert_157_to_base_12_l649_649388

theorem convert_157_to_base_12 : (to_base 12 157 = [11, 2, 1]) :=
by
  sorry

end convert_157_to_base_12_l649_649388


namespace sum_of_cubes_mod_6_l649_649380

theorem sum_of_cubes_mod_6 :
  (∑ k in Finset.range 151, k^3) % 6 = 3 :=
by
  -- We'll insert the rest of the proof here, but we just require the statement for now.
  sorry

end sum_of_cubes_mod_6_l649_649380


namespace remaining_tanning_time_l649_649539

noncomputable def tanning_limit : ℕ := 200
noncomputable def daily_tanning_time : ℕ := 30
noncomputable def weekly_tanning_days : ℕ := 2
noncomputable def weeks_tanned : ℕ := 2

theorem remaining_tanning_time :
  let total_tanning_first_two_weeks := daily_tanning_time * weekly_tanning_days * weeks_tanned
  tanning_limit - total_tanning_first_two_weeks = 80 :=
by
  let total_tanning_first_two_weeks := daily_tanning_time * weekly_tanning_days * weeks_tanned
  have h : total_tanning_first_two_weeks = 120 := by sorry
  show tanning_limit - total_tanning_first_two_weeks = 80 from sorry

end remaining_tanning_time_l649_649539


namespace expression_not_defined_l649_649815

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : ℝ := x^2 - 25*x + 125

-- Theorem statement that the expression is not defined for specific values of x
theorem expression_not_defined (x : ℝ) : quadratic_eq x = 0 ↔ (x = 5 ∨ x = 20) :=
by
  sorry

end expression_not_defined_l649_649815


namespace jungkook_persimmons_l649_649485

-- Define the conditions
variables (J : ℕ) (H : ℕ)

-- Condition: Hoseok picked 35 persimmons
def hoseok_persimmons : ℕ := 35

-- Condition: Hoseok's persimmons are 3 more than 4 times Jungkook's persimmons
def hoseok_condition := hoseok_persimmons = 4 * J + 3

-- Statement to prove
theorem jungkook_persimmons :
  hoseok_condition →
  J = 8 :=
by
  sorry

end jungkook_persimmons_l649_649485


namespace find_value_of_k_find_value_of_t_l649_649468

theorem find_value_of_k (k : ℤ) (t : ℤ) (x y : ℤ) :
  ∃ k, (∀ x, k * x ^ 2 + (3 - 3 * k) * x + 2 * k - 6 = 0 → x ∈ ℤ) ∧
      (∀ y1 y2 : ℤ, (k + 3) * y1^2 - 15 * y1 + t = 0 ∧ (k + 3) * y2^2 - 15 * y2 + t = 0 
        ∧ y1 > 0 ∧ y2 > 0 ∧ y1 + y2 = 5 ∧ y1 * y2 = t / 3) :=
sorry

theorem find_value_of_t (k : ℤ) (t : ℤ) (y1 y2 : ℤ) :
  k = 0 → ∃ t, ( (3) * y1^2 - 15 * y1 + t = 0 ∧ (3) * y2^2 - 15 * y2 + t = 0 
    ∧ y1 > 0 ∧ y2 > 0 ∧ y1 + y2 = 5 ∧ y1 * y2 = t / 3 ∧ y1^2 + y2^2 = 17) :=
sorry

end find_value_of_k_find_value_of_t_l649_649468


namespace triangle_construction_possible_l649_649020

-- Define triangle existence with specific properties
noncomputable def triangle_exists (h_a b r : ℝ) : Prop :=
  ∃ A B C : ℝ × ℝ, 
  ∃ I : ℝ × ℝ, -- incenter
  let ha := (A.2 - ((B.2 + C.2) / 2)), 
      BC := (B.1 - C.1) * (B.1 - C.1) + (B.2 - C.2) * (B.2 - C.2),
      r_given := r 
  in
    ha = h_a ∧
    BC.sqrt = b ∧
    (let r_calc := ((A.1 - I.1)^2 + (A.2 - I.2)^2).sqrt in r_calc = r_given)

-- Our objective is to prove this definition:
theorem triangle_construction_possible (h_a b r : ℝ) : triangle_exists h_a b r := 
by
  sorry

end triangle_construction_possible_l649_649020


namespace range_of_a_l649_649848

noncomputable def f (x a : ℝ) := 2^(2*x) - a * 2^x + 4

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a ≤ 4 :=
by
  sorry

end range_of_a_l649_649848


namespace triangle_similarity_iff_quadrilateral_cyclic_l649_649176

noncomputable def centroid (A B C : Point) : Point := sorry
noncomputable def midpoint (P Q : Point) : Point := sorry
noncomputable def parallel (L1 L2 : Line) : Prop := sorry
noncomputable def cyclic (Q : Quadrilateral) : Prop := sorry
noncomputable def similar (T1 T2 : Triangle) : Prop := sorry

axiom Point : Type
axiom Line : Type
axiom Triangle : Type
axiom Quadrilateral : Type

variables {A B C G A_1 B_1 C_1 F : Point}

-- Given Conditions:
axiom h1 : G = centroid A B C
axiom h2 : A_1 = midpoint B C
axiom h3 : B_1 = midpoint C A
axiom h4 : C_1 = midpoint A B
axiom h5 : parallel (Line_through A_1 F) (Line_through B B_1)
axiom h6 : F = intersection (Line_through A_1 direction_parallel_to (Line_through B B_1)) (Line_through B_1 C_1)

-- Question to prove the equivalence:
theorem triangle_similarity_iff_quadrilateral_cyclic :
  (similar (Triangle A B C) (Triangle F A_1 A)) ↔ (cyclic (Quadrilateral A B_1 G C_1)) :=
sorry

end triangle_similarity_iff_quadrilateral_cyclic_l649_649176


namespace find_non_negative_dot_product_l649_649178

variables {ℝ : Type*} [inner_product_space ℝ (euclidean_space ℝ (fin 3))]

-- Define the five vectors in ℝ³
variables (v1 v2 v3 v4 v5 : euclidean_space ℝ (fin 3))

-- Define a proposition that represents the problem statement
def dot_product_non_negative (v1 v2 v3 v4 v5 : euclidean_space ℝ (fin 3)) : Prop :=
  ∃ (i j : fin 5), i ≠ j ∧ inner_product_space.dot_product (nth_fun i [v1, v2, v3, v4, v5]) (nth_fun j [v1, v2, v3, v4, v5]) ≥ 0

-- State the theorem based on the proposition
theorem find_non_negative_dot_product (v1 v2 v3 v4 v5 : euclidean_space ℝ (fin 3)) :
  dot_product_non_negative v1 v2 v3 v4 v5 :=
sorry

end find_non_negative_dot_product_l649_649178


namespace tan_of_sin_in_interval_l649_649817

theorem tan_of_sin_in_interval (α : ℝ) (h1 : Real.sin α = 4 / 5) (h2 : 0 < α ∧ α < Real.pi) :
  Real.tan α = 4 / 3 ∨ Real.tan α = -4 / 3 :=
  sorry

end tan_of_sin_in_interval_l649_649817


namespace max_elements_of_M_l649_649968

open Set

-- Define the main set M to be a subset of the given range
def M : Set ℕ := { x | 1 ≤ x ∧ x ≤ 2007 }

-- Define the condition of the divisibility property among every three elements
def has_divisibility_property (M : Set ℕ) : Prop :=
  ∀ a b c ∈ M, ∃ x y ∈ ({a, b, c} : Set ℕ), x ≠ y ∧ (x ∣ y ∨ y ∣ x)

-- Prove the given problem statement
theorem max_elements_of_M (M : Set ℕ) (hM : ∀ x ∈ M, 1 ≤ x ∧ x ≤ 2007) :
  has_divisibility_property M → ∃ n, n = 21 ∧ ∀ M', M' ⊆ M ∧ has_divisibility_property M' → card M' ≤ 21 := 
sorry

end max_elements_of_M_l649_649968


namespace pyramid_volume_l649_649243

variable (α β R : ℝ)
variable (hα : 0 < α ∧ α < π)
variable (hβ : 0 < β ∧ β < π / 2)

theorem pyramid_volume (α β R : ℝ) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π / 2)
  (hR : R > 0) : 
  let V := (2 / 3) * R^3 * (Real.sin (2 * β)) * (Real.cos β) * (Real.sin α) in 
  V = (2 / 3) * R^3 * (Real.sin (2 * β)) * (Real.cos β) * (Real.sin α) :=
by
  sorry

end pyramid_volume_l649_649243


namespace obtuse_angle_condition_l649_649873

def dot_product (a b : (ℝ × ℝ)) : ℝ := a.1 * b.1 + a.2 * b.2

def is_obtuse_angle (a b : (ℝ × ℝ)) : Prop := dot_product a b < 0

def is_parallel (a b : (ℝ × ℝ)) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem obtuse_angle_condition :
  (∀ (x : ℝ), x > 0 → is_obtuse_angle (-1, 0) (x, 1 - x) ∧ ¬is_parallel (-1, 0) (x, 1 - x)) ∧ 
  (∀ (x : ℝ), is_obtuse_angle (-1, 0) (x, 1 - x) → x > 0) :=
sorry

end obtuse_angle_condition_l649_649873


namespace g_f_neg2_eq_12_l649_649191

def f (x : ℝ) : ℝ := 2 * x^2 - 4

def g : ℝ → ℝ

axiom gf2_eq_12 : g (f 2) = 12

theorem g_f_neg2_eq_12 : g (f (-2)) = 12 :=
by
  sorry

end g_f_neg2_eq_12_l649_649191


namespace smallest_n_div_75_eq_432_l649_649199

theorem smallest_n_div_75_eq_432 :
  ∃ n k : ℕ, (n ∣ 75 ∧ (∃ (d : ℕ), d ∣ n → d ≠ 1 → d ≠ n → n = 75 * k ∧ ∀ x: ℕ, (x ∣ n) → (x ≠ 1 ∧ x ≠ n) → False)) → ( k =  432 ) :=
by
  sorry

end smallest_n_div_75_eq_432_l649_649199


namespace count_of_diverse_dates_in_2013_l649_649642

def is_diverse_date (d : Nat) (m : Nat) (y : Nat) : Bool :=
  let digits := ((d / 10) % 10) :: (d % 10) :: ((m / 10) % 10) :: (m % 10) :: ((y / 10) % 10) :: (y % 10) :: []
  List.sort (Nat.compare) digits == [0, 1, 2, 3, 4, 5]

def count_diverse_dates_in_year (year : Nat) : Nat :=
  if year != 2013 then 0 else
  List.length [⟨d, m⟩ | d ← List.range' 1 31, m ← List.range' 1 12, is_diverse_date d m year]

theorem count_of_diverse_dates_in_2013 :
  count_diverse_dates_in_year 2013 = 2 :=
  sorry

end count_of_diverse_dates_in_2013_l649_649642


namespace find_k_l649_649023

noncomputable def length_of_union (a b : ℝ) : ℝ := b - a

def floor (x : ℝ) : ℤ := int.floor x
def fractional_part (x : ℝ) : ℝ := x - floor x

noncomputable def f (x : ℝ) : ℝ := (floor x).to_real * (fractional_part x)
noncomputable def g (x : ℝ) : ℝ := x - 1

theorem find_k (k : ℝ) (h1 : 0 ≤ k)
  (h2 : (∑ i in (finset.filter (λ x, f x < g x) (finset.Icc 0 k)), (1 : ℝ)) = 5) : k = 7 :=
sorry

end find_k_l649_649023


namespace part1_part2_l649_649473

noncomputable def f (x : ℝ) : ℝ := (1 - 3^x) / (1 + 3^x)

theorem part1 (m : ℝ) : 
  (∀ x : ℝ, f (-x) = -f x) → m = 1 := 
by 
  sorry

theorem part2 (k : ℝ) : 
  (∀ t ∈ set.Icc (0 : ℝ) 5, f (t^2 + 2 * t + k) + f (-2 * t^2 + 2 * t - 5) > 0) → 
  k < 1 := 
by 
  sorry

end part1_part2_l649_649473


namespace question_I_solution_question_II_solution_l649_649474

def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

-- Statement for Question I
theorem question_I_solution (x : ℝ) :
  let a := 1, b := 2 in
  (f x a b < 4) ↔ (-3 / 2 < x ∧ x < 5 / 2) :=
sorry

-- Statement for Question II
theorem question_II_solution (a b : ℝ) (h : (1 / (2 * a) + 2 / b) = 1) :
  f 0 a b ≥ 9 / 2 ∧ (f 0 a b = 9 / 2 → a = 3 / 2 ∧ b = 3) :=
sorry

end question_I_solution_question_II_solution_l649_649474


namespace six_digit_nondecreasing_count_l649_649124

theorem six_digit_nondecreasing_count : 
  ∃ (count : ℕ), count = 3003 ∧ 
  ∀ (digits : Fin 6 → Fin 9), 
  (∀ i j, i ≤ j → digits i ≤ digits j) →
  (∀ i, digits i ≠ 0) :=
begin
  sorry
end

end six_digit_nondecreasing_count_l649_649124


namespace pair_perpendicular_lines_l649_649533

def slope_of_line (L : ℝ × ℝ → Prop) (s : ℝ) := ∀ x₁ x₂ y₁ y₂ : ℝ, L (x₁, y₁) ∧ L (x₂, y₂) → (y₂ - y₁) / (x₂ - x₁) = s

theorem pair_perpendicular_lines :
  (∀ L₁ L₂ : ℝ × ℝ → Prop, ∃ s₁ s₂ : ℝ, slope_of_line L₁ s₁ ∧ slope_of_line L₂ s₂ ∧ s₁ * s₂ = -1 ∨ (s₁ = 0 ∧ slope_of_line L₂ ∞) ∨ (slope_of_line L₁ ∞ ∧ s₂ = 0)) →
  ∀ (Plane : set (ℝ × ℝ)), ∃ (pairing : (ℝ × ℝ → Prop) → (ℝ × ℝ → Prop)), 
  (∀ L, Plane L → ∃ M, Plane M ∧ (slope_of_line L M)) :=
by
  sorry

end pair_perpendicular_lines_l649_649533


namespace line_bisects_segment_l649_649898

variables (A B C O E F G K : Type*) [ordered_geometry A B C O E F G K]

-- Given conditions
variables (h1 : AB = AC)
variables (h2 : O = mid_point B C)
variables (h3 : tangent_at_circle O AB E)
variables (h4 : tangent_at_circle O AC F)
variables (h5 : G ∈ circle O ∧ AG ⊥ EG)
variables (h6 : tangent_at_circle O G intersects AC K)

-- The goal is to prove
theorem line_bisects_segment (h: line_intersects_point BK : segment_bisects EF) : Prop := 
  sorry

end line_bisects_segment_l649_649898


namespace gcd_of_B_is_2_l649_649964

-- Condition: B is the set of all numbers which can be represented as the sum of four consecutive positive integers
def B := { n : ℕ | ∃ y : ℕ, n = (y - 1) + y + (y + 1) + (y + 2) }

-- Question: What is the greatest common divisor of all numbers in \( B \)
-- Mathematical equivalent proof problem: Prove gcd of all elements in set \( B \) is 2

theorem gcd_of_B_is_2 : ∀ n ∈ B, ∃ y : ℕ, n = 2 * (2 * y + 1) → ∀ m ∈ B, n.gcd m = 2 :=
by
  sorry

end gcd_of_B_is_2_l649_649964


namespace swimmers_meet_times_l649_649293

noncomputable def swimmers_passes (pool_length : ℕ) (time_minutes : ℕ) (speed_swimmer1 : ℕ) (speed_swimmer2 : ℕ) : ℕ :=
  let total_time_seconds := time_minutes * 60
  let speed_sum := speed_swimmer1 + speed_swimmer2
  let distance_in_time := total_time_seconds * speed_sum
  distance_in_time / pool_length

theorem swimmers_meet_times :
  swimmers_passes 120 15 4 3 = 53 :=
by
  -- Proof is omitted
  sorry

end swimmers_meet_times_l649_649293


namespace player_A_wins_l649_649624

-- Define the game conditions and rules
def sequence : List ℕ := List.range 100 |>.map (+1)

-- Definition of the result check
def result (expr : List (ℕ × Option Char)) : ℕ :=
expr.map (λ (n, c), 
  match c with
  | none => n
  | some '*' => n
  end
).sum

-- The main theorem to show player A has a winning strategy
theorem player_A_wins :
  (∃ strategyA : List (ℕ × Option Char) → List (ℕ × Option Char),
    ∀ (state : List (ℕ × Option Char)), result (strategyA state) % 2 = 1) :=
sorry

end player_A_wins_l649_649624


namespace parallel_segments_l649_649999

theorem parallel_segments {A B C D O : Type*} [AddGroup O] [MetricSpace O] [HasMidpoint O] 
  (h1 : midpoint A B = O) (h2 : midpoint C D = O) :
  (is_parallel (AC : Line) (BD : Line)) ∧ (is_parallel (AD : Line) (BC : Line)) :=
by
  sorry

end parallel_segments_l649_649999


namespace cosine_450_eq_0_l649_649735

theorem cosine_450_eq_0 : Real.cos (450 * Real.pi / 180) = 0 :=
by
  -- Convert 450 degrees to radians
  have h1 : (450 : ℝ) = 360 + 90 := by norm_num
  -- Use the periodic property of cosine
  have h2 : ∀ x : ℝ, Real.cos (x + 2 * Real.pi) = Real.cos x := Real.cos_periodic (2 * Real.pi)
  -- Since 450 degrees = 360 degrees + 90 degrees
  rw [h1, Real.cos_add, h2 (90 * Real.pi / 180)]
  -- Convert 90 degrees to radians and solve
  rw [Real.pi_mul_base_div, Real.pi_div_two, Real.cos_pi_div_two]
  norm_num

end cosine_450_eq_0_l649_649735


namespace zero_of_f_in_interval_l649_649420

noncomputable def f (x : ℝ) : ℝ := log x / log 3 - 1 / x

theorem zero_of_f_in_interval :
  ∃ n : ℕ, 0 < n ∧ (∃ c : ℝ, n < c ∧ c < n + 1 ∧ f c = 0) := 
begin
  use 1,
  split,
  { exact nat.succ_pos' 0, },
  { use (2 : ℝ),
    split,
    { exact zero_lt_one, },
    split,
    { exact one_lt_two, },
    { sorry, } 
  }
end

end zero_of_f_in_interval_l649_649420


namespace max_value_x_plus_2y_max_of_x_plus_2y_l649_649828

def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 4 = 1

theorem max_value_x_plus_2y (x y : ℝ) (h : on_ellipse x y) :
  x + 2 * y ≤ Real.sqrt 22 :=
sorry

theorem max_of_x_plus_2y (x y : ℝ) (h : on_ellipse x y) :
  ∃θ ∈ Set.Icc 0 (2 * Real.pi), (x = Real.sqrt 6 * Real.cos θ) ∧ (y = 2 * Real.sin θ) :=
sorry

end max_value_x_plus_2y_max_of_x_plus_2y_l649_649828


namespace n_fraction_of_sum_l649_649336

theorem n_fraction_of_sum (n S : ℝ) (h1 : n = S / 5) (h2 : S ≠ 0) :
  n = 1 / 6 * ((S + (S / 5))) :=
by
  sorry

end n_fraction_of_sum_l649_649336


namespace prism_is_parallelepiped_l649_649227

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]

-- Linearly independent vectors
variables {a b c : V}
variable (h_lin_ind : LinearIndependent ℝ ![a, b, c])

-- Prism vertices
variables (x y : ℝ)

-- Vertices of the cuboid
variables {O A B C D E F G M : V}
variables
  (h_OA : A = O + a)
  (h_OB : B = O + b)
  (h_OC : C = O + c)
  (h_OD : D = O + x • a + y • b)
  (h_OE : E = O + a + c)
  (h_OG : G = O + b + c)
  (h_OF : F = O + x • a + y • b + c)

-- Intersection conditions
variables (β γ δ : ℝ)
variables
  (h_M_AG : M = β • a + (1 - β) • (b + c))
  (h_M_BE : M = (1 - γ) • a + γ • (b + c))
  (h_M_OF : M = δ • (x • a + y • b + c))

theorem prism_is_parallelepiped
  (h_coeff_a : β = 1 - γ ∧ δ * x = 1 - γ)
  (h_coeff_b : 1 - β = γ ∧ δ * y = γ)
  (h_coeff_c : δ = 1 - γ) :
  x = 1 ∧ y = 1 :=
by
  sorry

end prism_is_parallelepiped_l649_649227


namespace variance_Z_l649_649576

variable (X Y : ℝ)
variable [OutParam (ProbabilitySpace X)]
variable [OutParam (ProbabilitySpace Y)]

theorem variance_Z (h_ind : Independent X Y) (h_varX : Variance X = 5) (h_varY : Variance Y = 6) : 
  Variance (3 * X + 2 * Y) = 69 := 
sorry

end variance_Z_l649_649576


namespace negation_proof_l649_649454

-- Given proposition p
def proposition_p (x y : ℝ) : Prop := x = y → sqrt x = sqrt y

-- The negation of proposition p
def negation_of_proposition_p (x y : ℝ) : Prop := x ≠ y → sqrt x ≠ sqrt y

-- The theorem to prove
theorem negation_proof (x y : ℝ) : ¬ proposition_p x y ↔ negation_of_proposition_p x y := by
  sorry

end negation_proof_l649_649454


namespace angle_between_circles_l649_649528

open Real EuclideanGeometry

theorem angle_between_circles {A B C O C₁ B₁ : Point} 
  (hC₁ : midpoint A B C₁) 
  (hB₁ : midpoint A C B₁) 
  (hO : circumcenter_triangle A B C O)
  (hCirclesIntersect : circles_intersect A B₁ C₁ B C O) :
  angle_between_circles_at_intersection A B₁ C₁ B C O = angle β - angle γ :=
by
  sorry

end angle_between_circles_l649_649528


namespace krishan_money_l649_649666

-- Define the constants
def Ram : ℕ := 490
def ratio1 : ℕ := 7
def ratio2 : ℕ := 17

-- Defining the relationship
def ratio_RG (Ram Gopal : ℕ) : Prop := Ram / Gopal = ratio1 / ratio2
def ratio_GK (Gopal Krishan : ℕ) : Prop := Gopal / Krishan = ratio1 / ratio2

-- Define the problem
theorem krishan_money (R G K : ℕ) (h1 : R = Ram) (h2 : ratio_RG R G) (h3 : ratio_GK G K) : K = 2890 :=
by
  sorry

end krishan_money_l649_649666


namespace sum_infinite_series_result_l649_649957

noncomputable def sum_infinite_series (x : ℝ) (h : 1 < x) : ℝ :=
  ∑' n, 1 / (x^(3^n) - x^(-3^n))

theorem sum_infinite_series_result (x : ℝ) (h : 1 < x) :
  sum_infinite_series x h = 1 / (x - 1) :=
sorry

end sum_infinite_series_result_l649_649957


namespace officer_permutations_count_l649_649087

theorem officer_permutations_count : 
  let members := ['Alice', 'Bob', 'Carol', 'Dave']
  let positions := ['president', 'vice_president', 'secretary', 'treasurer']
  (∀ (no_duplicate : members.nodup) (positions_no_duplicate : positions.nodup), perms members).length = 24 :=
begin
  sorry
end

end officer_permutations_count_l649_649087


namespace perfect_squares_are_good_exists_infinite_good_set_disjoint_from_S_l649_649706

-- Definition of a good set
def is_good_set (A : set ℕ) : Prop :=
  ∀ (n : ℕ), n > 0 → ∃! (p : ℕ), nat.prime p ∧ n - p ∈ A

-- The set of perfect squares
def S : set ℕ := { n | ∃ m : ℕ, n = m * m }

-- Problem (a): Prove that S is good
theorem perfect_squares_are_good : is_good_set S :=
sorry

-- Problem (b): Find an infinite good set disjoint from S
theorem exists_infinite_good_set_disjoint_from_S :
  ∃ (T : set ℕ), is_good_set T ∧ (T ∩ S) = ∅ ∧ set.infinite T :=
sorry

end perfect_squares_are_good_exists_infinite_good_set_disjoint_from_S_l649_649706


namespace find_n_from_degree_l649_649891

theorem find_n_from_degree (n : ℕ) (h : 2 + n = 5) : n = 3 :=
by {
  sorry
}

end find_n_from_degree_l649_649891


namespace kitty_cleaning_time_l649_649570

theorem kitty_cleaning_time
    (picking_up_toys : ℕ := 5)
    (vacuuming : ℕ := 20)
    (dusting_furniture : ℕ := 10)
    (total_time_4_weeks : ℕ := 200)
    (weeks : ℕ := 4)
    : (total_time_4_weeks - weeks * (picking_up_toys + vacuuming + dusting_furniture)) / weeks = 15 := by
    sorry

end kitty_cleaning_time_l649_649570


namespace michael_needs_more_money_l649_649210

noncomputable def michael_costs : ℝ :=
  let cake : ℝ := 20 * 0.9 in                            -- Applying 10% discount to the cake
  let bouquet : ℝ := 36 * 1.05 in                        -- Applying 5% sales tax to the bouquet
  let balloons : ℝ := 5 in                               -- Cost of the balloons (no discount or tax)
  let perfume : ℝ := (30 * 0.85) * 1.4 in                -- Applying 15% discount to perfume and converting to USD (1 GBP = 1.4 USD)
  let photo_album : ℝ := (25 * 1.08) * 1.2 in            -- Applying 8% sales tax to photo album and converting to USD (1 EUR = 1.2 USD)
  cake + bouquet + balloons + perfume + photo_album

noncomputable def total_cost : ℝ := michael_costs

noncomputable def michaels_money : ℝ := 50

noncomputable def additional_amount_needed : ℝ := total_cost - michaels_money

theorem michael_needs_more_money : additional_amount_needed = 78.90 :=
  by
    have h1 : total_cost = 128.90 := sorry
    have h2 : additional_amount_needed = total_cost - michaels_money := rfl
    rw [h1] at h2
    norm_num at h2
    exact h2

end michael_needs_more_money_l649_649210


namespace max_pairs_distance_one_l649_649970

theorem max_pairs_distance_one {S : Finset (ℝ × ℝ)} (hS : S.card = n) 
  (h_dist : ∀ p1 p2 ∈ S, dist p1 p2 ≤ 1) : 
  ∃ k : ℕ, k ≤ n ∧ ∀ p1 p2 ∈ S, dist p1 p2 = 1 → k ≤ n := 
sorry

end max_pairs_distance_one_l649_649970


namespace simplify_fraction_l649_649228

theorem simplify_fraction (x : ℚ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by sorry

end simplify_fraction_l649_649228


namespace max_determinant_value_l649_649414

def matrix_determinant (θ : ℝ) : ℝ :=
  (Matrix.det ![
    ![1, 1, 1],
    ![1, 1 + Real.sin θ, 1],
    ![1, 1, 1 + Real.cos θ]
  ])

theorem max_determinant_value : ∃ θ : ℝ, matrix_determinant θ = 1 :=
sorry

end max_determinant_value_l649_649414


namespace starting_player_win_l649_649354

-- Definitions for the game and conditions
def initial_chocolate_bar : ℕ × ℕ := (5, 10)

-- The assertion that the starting player has a winning strategy
theorem starting_player_win : 
  ∀ (game_variant : char), 
  (game_variant = 'a' ∨ game_variant = 'b') →
  ∃ strategy_for_starting_player : (ℕ × ℕ) → (ℕ × ℕ) × (ℕ × ℕ), 
    strategy_for_starting_player initial_chocolate_bar = 
    ((5, 5), (5, 5)) ∧
    ( ∀ second_player_move : (ℕ × ℕ) → (ℕ × ℕ) × (ℕ × ℕ), 
      ∃ first_player_response : (ℕ × ℕ) → (ℕ × ℕ) × (ℕ × ℕ), 
      first_player_response = second_player_move ) :=
sorry

end starting_player_win_l649_649354


namespace reciprocal_sum_l649_649621

theorem reciprocal_sum :
  let a := (1 / 4 : ℚ)
  let b := (1 / 5 : ℚ)
  1 / (a + b) = 20 / 9 :=
by
  let a := (1 / 4 : ℚ)
  let b := (1 / 5 : ℚ)
  have h : a + b = 9 / 20 := by sorry
  have h_rec : 1 / (a + b) = 20 / 9 := by sorry
  exact h_rec

end reciprocal_sum_l649_649621


namespace division_to_percentage_l649_649304

theorem division_to_percentage (a b : ℝ) (h1 : a = 0.15) (h2 : b = 0.005) :
  ((a / b) * 100) = 3000 :=
by
  have h : (a / b) = 30 := sorry
  calc
    ((a / b) * 100) = (30 * 100) : by rw h
                 ... = 3000 : by norm_num

end division_to_percentage_l649_649304


namespace sequence_monotonically_increasing_and_bounded_l649_649623

noncomputable def sequence (n : ℕ) : ℝ := 
  if n = 0 then 0.5
  else 0.5 + (sequence (n - 1))^2 / 2

theorem sequence_monotonically_increasing_and_bounded (n : ℕ) :
  (∀ n : ℕ, sequence (n + 1) > sequence n) ∧ (∀ n : ℕ, sequence n < 2) :=
by
  sorry

end sequence_monotonically_increasing_and_bounded_l649_649623


namespace christmas_sale_pricing_l649_649396

theorem christmas_sale_pricing (a b : ℝ) : 
  (forall (c : ℝ), c = a * (3 / 5)) ∧ (forall (d : ℝ), d = b * (5 / 3)) :=
by
  sorry  -- proof goes here

end christmas_sale_pricing_l649_649396


namespace find_b_perpendicular_lines_l649_649419

def are_perpendicular (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

theorem find_b_perpendicular_lines :
  let v1 := (b, -3, 2)
  let v2 := (2, 1, 3)
  are_perpendicular v1 v2 →
  b = -3 / 2 :=
by
  intros
  sorry

end find_b_perpendicular_lines_l649_649419


namespace square_root_4_square_root_9_cube_root_neg27_l649_649601

theorem square_root_4 :
  ∃ x : ℝ, x^2 = 4 ∧ x = 2 := 
begin
  use 2,
  split,
  { norm_num, },
  { refl, },
end

theorem square_root_9 :
  ∃ y : ℝ, y^2 = 9 ∧ (y = 3 ∨ y = -3) := 
begin
  use 3,
  split,
  { norm_num, },
  { left, refl, },
end

theorem cube_root_neg27 :
  ∃ z : ℝ, z^3 = -27 ∧ z = -3 := 
begin
  use -3,
  split,
  { norm_num, },
  { refl, },
end

end square_root_4_square_root_9_cube_root_neg27_l649_649601


namespace inequality_proof_l649_649975

theorem inequality_proof 
  (a b c x y z : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (h_sum : a + b + c = 1) : 
  (x^2 + y^2 + z^2) * 
  (a^3 / (x^2 + 2 * y^2) + b^3 / (y^2 + 2 * z^2) + c^3 / (z^2 + 2 * x^2)) 
  ≥ 1 / 9 := 
by 
  sorry

end inequality_proof_l649_649975


namespace min_val_YP_PQ_QZ_eq_sqrt_304_l649_649148

noncomputable def minimumPossibleValue
  (X Y Z P Q : Type)
  [triangle : Triangle X Y Z]
  (angleXYZ : angle X Y Z = 60)
  (XY_len : distance X Y = 8)
  (XZ_len : distance X Z = 12)
  (P_on_XY : P ∈ line_segment X Y)
  (Q_on_XZ : Q ∈ line_segment X Z) : ℝ :=
  sqrt 304

-- The theorem we want to prove
theorem min_val_YP_PQ_QZ_eq_sqrt_304 :
  ∃ (X Y Z P Q : Type),
    Triangle X Y Z ∧
    angle X Y Z = 60 ∧
    distance X Y = 8 ∧
    distance X Z = 12 ∧
    (P ∈ line_segment X Y) ∧
    (Q ∈ line_segment X Z) ∧
    minimumPossibleValue X Y Z P Q = sqrt 304 :=
sorry

end min_val_YP_PQ_QZ_eq_sqrt_304_l649_649148


namespace least_positive_integer_with_divisors_2023_l649_649253

-- Define the factorizations as conditions
def is_least_positive_integer_with_divisors_2023 (n : ℕ) : Prop :=
  ∃ m k : ℕ, n = m * 12^k ∧ 2023 = nat.totient n ∧ ∀ d, d ∣ m → ¬ (d = 12)

-- Question: prove that n can be written this way and the sum of m and k
theorem least_positive_integer_with_divisors_2023 : 
  ∃ n m k : ℕ, is_least_positive_integer_with_divisors_2023 n ∧ m + k = 1595347 :=
sorry

end least_positive_integer_with_divisors_2023_l649_649253


namespace certain_event_is_B_l649_649315

-- Define the conditions as propositions
def conditions (event: Type) : Type :=
  event = "Scooping the moon in the water" ∨
  event = "The water level rises, and the boat rises" ∨
  event = "Waiting for the rabbit by the tree" ∨
  event = "Hitting the target from a hundred steps away"

-- Define the proposition for a certain event (here event B)
def certain_event (event: Type) : Prop :=
  event = "The water level rises, and the boat rises"

-- The main theorem to prove
theorem certain_event_is_B : ∀ (event: Type), conditions event → certain_event event :=
by
  intro event cond
  sorry

end certain_event_is_B_l649_649315


namespace storks_more_than_birds_l649_649329

theorem storks_more_than_birds 
  (initial_birds : ℕ) 
  (joined_storks : ℕ) 
  (joined_birds : ℕ) 
  (h_init_birds : initial_birds = 3) 
  (h_joined_storks : joined_storks = 6) 
  (h_joined_birds : joined_birds = 2) : 
  (joined_storks - (initial_birds + joined_birds)) = 1 := 
by 
  -- Proof goes here
  sorry

end storks_more_than_birds_l649_649329


namespace bc_lt_3ad_l649_649217

theorem bc_lt_3ad {a b c d x1 x2 x3 : ℝ}
    (h1 : a ≠ 0)
    (h2 : x1 > 0 ∧ x2 > 0 ∧ x3 > 0)
    (h3 : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)
    (h4 : x1 + x2 + x3 = -b / a)
    (h5 : x1 * x2 + x2 * x3 + x1 * x3 = c / a)
    (h6 : x1 * x2 * x3 = -d / a) : 
    b * c < 3 * a * d := 
sorry

end bc_lt_3ad_l649_649217


namespace probability_perpendicular_is_one_sixth_l649_649577

noncomputable def probability_perpendicular : ℚ :=
  let A := {2, 3, 4, 5}
  let B := {1, 3, 5}
  let possible_pairs : Set (ℕ × ℕ) :=
    { (a, b) | a ∈ A ∧ b ∈ B }
  let perpendicular_pairs : Set (ℕ × ℕ) :=
    { (3, 3), (5, 5) }
  (perpendicular_pairs.to_finset.card : ℚ) / (possible_pairs.to_finset.card : ℚ)

theorem probability_perpendicular_is_one_sixth : probability_perpendicular = 1 / 6 :=
by sorry

end probability_perpendicular_is_one_sixth_l649_649577


namespace perimeter_of_shaded_region_l649_649159

-- Define the circles and their properties
def radius_of_circle (diameter : ℝ) : ℝ := diameter / 2
def circumference_of_circle (radius : ℝ) : ℝ := 2 * Real.pi * radius
def arc_length (circumference : ℝ) (angle : ℝ) : ℝ := angle / 360 * circumference

-- The main theorem
theorem perimeter_of_shaded_region :
  ∀ (d : ℝ) (n : ℕ),
    d = 24 → 
    n = 3 → 
    ∃ (P : ℝ), P = 12 * Real.pi := 
begin
  intros d n h1 h2,
  let r := radius_of_circle d,
  let C := circumference_of_circle r,
  let L := arc_length C 60,
  use 3 * L,
  rw [h1, h2],
  unfold radius_of_circle circumference_of_circle arc_length,
  norm_num,
  rw Real.pi_mul_assoc,
  ring,
  sorry,
end

end perimeter_of_shaded_region_l649_649159


namespace domain_proof_l649_649605

noncomputable def domain_of_f : Set ℝ := { x : ℝ | -2 ≤ x ∧ x < 0 ∨ 3 < x ∧ x ≤ 5 }

def is_in_domain (x : ℝ) : Prop :=
  (1 - Real.log10 (x^2 - 3 * x) ≥ 0) ∧ (x^2 - 3 * x > 0)

theorem domain_proof :
  ∀ x : ℝ, is_in_domain x ↔ x ∈ domain_of_f :=
sorry

end domain_proof_l649_649605


namespace imaginary_part_i_pow_2017_l649_649246

theorem imaginary_part_i_pow_2017 : 
  let z := Complex.i ^ 2017 
  in z.im = 1 :=
  by
  sorry

end imaginary_part_i_pow_2017_l649_649246


namespace num_positive_terms_l649_649752

def sequence_cos (n : ℕ) : ℝ :=
  cos (10^(n-1) * π / 180)

theorem num_positive_terms : (finset.filter (λ n : ℕ, 0 < sequence_cos n) (finset.range 100)).card = 99 :=
by
  sorry

end num_positive_terms_l649_649752


namespace quadratic_two_distinct_real_roots_find_m_and_other_root_l649_649449

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c
def roots_sum (a b : ℝ) : ℝ := -b / a

theorem quadratic_two_distinct_real_roots (m : ℝ)
  (hm : m < 0) :
  ∀ (a b c : ℝ), a = 1 → b = -2 → c = m → (discriminant a b c > 0) :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc, discriminant]
  sorry

theorem find_m_and_other_root (a b c r1 : ℝ)
  (ha : a = 1)
  (hb : b = -2)
  (hc : c = r1^2 - 2*r1 + c = 0)
  (hr1 : r1 = -1)
  :
  c = -3 ∧ 
  ∃ r2 : ℝ, (roots_sum a b = 2) ∧ (r1 + r2 = 2) ∧ (r1 = -1 → r2 = 3) :=
by
  intros
  rw [ha, hb, hr1]
  sorry

end quadratic_two_distinct_real_roots_find_m_and_other_root_l649_649449


namespace angle_sum_proof_l649_649129

noncomputable def triangle_angles (A B D F G: ℝ) : ℝ :=
  if (A = 20) ∧ (AFG = AGF) then
    180 - 100
  else
    0

theorem angle_sum_proof (A B D F G: ℝ) (hA : A = 20) (hAFG  : AFG = AGF) : B + D = 80 :=
  by
    have : triangle_angles A B D F G = 80
    sorry

end angle_sum_proof_l649_649129


namespace sum_of_digits_least_6_digit_number_l649_649562

theorem sum_of_digits_least_6_digit_number :
  let n := (100000.to_nat_ceil / Nat.lcm (Nat.lcm 4 610) 15 * Nat.lcm (Nat.lcm 4 610) 15) + 2 in
  n.sum_digits = 17 :=
by
  sorry

end sum_of_digits_least_6_digit_number_l649_649562


namespace magnitude_b_l649_649118

variables (a b : ℝ × ℝ)

def vector_dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

def vector_norm (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 * v.1 + v.2 * v.2)

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ :=
(v.1 + w.1, v.2 + w.2)

theorem magnitude_b (h1 : vector_dot_product a b = 10)
                   (h2 : vector_norm (vector_add a b) = 5 * Real.sqrt 2)
                   (ha : a = (2, 1)) : vector_norm b = 5 :=
sorry

end magnitude_b_l649_649118


namespace angle_C_max_area_l649_649144

-- Problem (1): Given \( \frac{b}{c} = \sqrt{3} \sin A + \cos A \), prove that angle \( C \) is \( \frac{\pi}{6} \).
theorem angle_C (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 1) -- assumption added to make the trigonometric functions well-defined
  (h: b / c = sqrt 3 * (Real.sin A) + Real.cos A) : C = π / 6 :=
by
  sorry

-- Problem (2): Given \( c = 2 \) and \( C = \frac{\pi}{6} \), prove that the maximum area of \( \triangle ABC \) is \( 2 + \sqrt{3} \).
theorem max_area (A B C : ℝ) (a b c : ℝ)
  (h1: c = 2) (h2: C = π / 6) : 
  (1 / 2) * a * b * Real.sin C ≤ 2 + sqrt 3 :=
by
  sorry

end angle_C_max_area_l649_649144


namespace sum_of_digits_of_M_l649_649257

theorem sum_of_digits_of_M :
  ∃ M : ℕ, M^2 = 36^50 * 50^36 ∧ nat.digits 10 M.sum = 344 :=
sorry

end sum_of_digits_of_M_l649_649257


namespace compute_integral_I_l649_649677

variable (a : ℝ) (σ : Set (ℝ × ℝ)) 

def integrand (x y : ℝ) := (x^2 + y^2)^(1/4)

theorem compute_integral_I (hσ : ∀ x y, (x, y) ∈ σ ↔ x^2 + y^2 ≤ a^2 ∧ y ≤ 0) :
  ∫∫ (x, y) in σ, integrand x y = (4 * Real.pi / 5) * a^(5/2) :=
sorry

end compute_integral_I_l649_649677


namespace initially_available_boxes_l649_649602

theorem initially_available_boxes (total_muffins: ℕ) (muffins_per_box: ℕ) (additional_boxes_needed: ℕ) (total_boxes_needed: ℕ):
  total_muffins = 95 → muffins_per_box = 5 → additional_boxes_needed = 9 → total_boxes_needed = total_muffins / muffins_per_box → 
  total_boxes_needed - additional_boxes_needed = 10 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end initially_available_boxes_l649_649602


namespace greatest_prime_factor_187_l649_649645

theorem greatest_prime_factor_187 : ∃ p : ℕ, Prime p ∧ p ∣ 187 ∧ ∀ q : ℕ, Prime q ∧ q ∣ 187 → p ≥ q := by
  sorry

end greatest_prime_factor_187_l649_649645


namespace smallest_positive_value_a_l649_649428

theorem smallest_positive_value_a :
  ∃ (a : ℝ), (a = 0.79) ∧ ∀ (x : ℝ), (x > (3 * π / 2)) ∧ (x < 2 * π) →
    (sin x ^ 2 ≥ 0) ∧ (cos x ^ 2 ≥ 0) →
    (sqrt (sin x ^ 2)^(1 / 3) - sqrt (cos x ^ 2)^(1 / 3)) / (sqrt (tan x ^ 2)^(1 / 3) - sqrt (cot x ^ 2)^(1 / 3)) < a / 2 :=
begin
  sorry
end

end smallest_positive_value_a_l649_649428


namespace remaining_bikes_at_4_is_945_remaining_bikes_exceeds_capacity_at_42_l649_649862

def a : ℕ → ℕ
| n := if 1 ≤ n ∧ n ≤ 3 then 5 * n^4 + 15 else -10 * n + 470

def b : ℕ → ℕ
| n := n + 5

def S : ℕ → ℕ
| n := -4 * (n - 46)^2 + 8800

def remaining_bikes (n : ℕ) : ℕ :=
  (∑ i in (range n.succ).filter (λ x, x ≥ 1), a i) - (∑ i in (range n.succ).filter (λ x, x ≥ 1), b i)

theorem remaining_bikes_at_4_is_945 :
  remaining_bikes 4 = 945 :=
sorry

theorem remaining_bikes_exceeds_capacity_at_42 :
  remaining_bikes 42 > S 42 :=
sorry

end remaining_bikes_at_4_is_945_remaining_bikes_exceeds_capacity_at_42_l649_649862


namespace Delta_k_u_n_eq_zero_iff_k_ge_5_l649_649809

def u (n : ℕ) : ℕ := n^4 + 2 * n^2

def Delta1 (u : ℕ → ℕ) (n : ℕ) : ℕ := u (n + 1) - u n

def Delta : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
| 0, u => u
| k + 1, u => λ n, Delta1 (Delta k u) n

theorem Delta_k_u_n_eq_zero_iff_k_ge_5 (k n : ℕ) :
  (Delta k u n = 0) ↔ (k ≥ 5) :=
sorry

end Delta_k_u_n_eq_zero_iff_k_ge_5_l649_649809


namespace Samantha_routes_l649_649581

theorem Samantha_routes :
  let num_routes_house_to_park := Nat.choose (3 + 1) 1,
      num_routes_park_to_school := Nat.choose (3 + 3) 3 in
  num_routes_house_to_park * 1 * num_routes_park_to_school = 80 :=
by
  sorry

end Samantha_routes_l649_649581


namespace find_m_find_z_l649_649467

-- Define the conditions
def is_purely_imaginary (z : ℂ) : Prop :=
  (z.re = 0) ∧ (z.im ≠ 0)

def z1 (m : ℂ) : ℂ := m * (m - 1) + (m - 1) * complex.I

-- Problem 1: Find the value of the real number m
theorem find_m (m : ℂ) (hm : is_purely_imaginary (z1 m)) : m = 0 := by
  sorry

-- Define z1 when m = 0
def z1_when_m_zero : ℂ := -complex.I

-- Problem 2: Solve for z
theorem find_z (z : ℂ) (hz : (3 - complex.I) * z = 4 + 2 * complex.I) : z = 1 + complex.I := by
  sorry

end find_m_find_z_l649_649467


namespace find_possible_values_1992_l649_649554

noncomputable def a : ℕ → ℤ
| 0       := 0
| (n + 1) := 2 * a n - a (n - 1) + 2

noncomputable def b : ℕ → ℤ
| 0       := 8
| (n + 1) := 2 * b n - b (n - 1)

def a_squared_1992 : Prop := a 1992 = 1992 ^ 2

def b_possible_values : Prop :=
  b 1992 = 1992 * 4 + 8 ∨ b 1992 = 1992 * -4 + 8

theorem find_possible_values_1992 :
  a_squared_1992 ∧ b_possible_values :=
by {
  sorry
}

end find_possible_values_1992_l649_649554


namespace find_x_value_l649_649874

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Dot product of two 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition a ⊥ (a - b)
def perpendicular_condition (x : ℝ) : Prop :=
  dot_product a (a.1 - b x.1, a.2 - b x.2) = 0

-- Theorem statement to prove the specific value of x that satisfies the condition
theorem find_x_value : perpendicular_condition 9 :=
  sorry

end find_x_value_l649_649874


namespace sequence_properties_l649_649072

noncomputable theory

open_locale big_operators

/-- The sequence (a_n) defined by the conditions:
  (1) 4 * S_n - 1 = a_n^2 + 2 * a_n where S_n = sum of first n terms 
  (2) a_n > 0 
  has the form a_n = 2n - 1,
  and if b_n = 1 / (a_n * (a_n + 2)), then the sum of the first n terms T_n satisfies 1/3 <= T_n < 1/2 -/
theorem sequence_properties :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ),
  (∀ n : ℕ, S n = ∑ i in finset.range (n + 1), a i) →
  (∀ n : ℕ, 4 * S (n+1) - 1 = (a (n+1))^2 + 2 * a (n+1)) →
  (∀ n : ℕ, a (n + 1) > 0) →
  (∀ n : ℕ, a n = (2 : ℕ) * n - 1) →
  (∀ n : ℕ, b n = 1 / (a n * (a n + 2))) →
  (∀ n : ℕ, T n = ∑ i in finset.range (n + 1), b i) →
  ∀ n : ℕ, (1 / 3 : ℝ) ≤ T n ∧ T n < (1 / 2 : ℝ) :=
begin
  sorry
end

end sequence_properties_l649_649072


namespace percentage_increase_in_earnings_l649_649325

theorem percentage_increase_in_earnings :
  let original_earnings : ℝ := 60
  let new_earnings : ℝ := 78
  (new_earnings - original_earnings) / original_earnings * 100 = 30 :=
by
  let original_earnings : ℝ := 60
  let new_earnings : ℝ := 78
  have h : (new_earnings - original_earnings) / original_earnings * 100 = 30
  sorry

end percentage_increase_in_earnings_l649_649325


namespace median_is_50_l649_649286

def data_set := [10, 30, 50, 50, 70]

noncomputable def median (lst : List ℤ) : ℤ :=
  let sorted := List.sort (≤) lst
  sorted.get! (sorted.length / 2)

theorem median_is_50 : median data_set = 50 :=
by
  sorry

end median_is_50_l649_649286


namespace sum_infinite_series_result_l649_649962

noncomputable def sum_infinite_series (x : ℝ) (h : 1 < x) : ℝ :=
  ∑' n, 1 / (x^(3^n) - x^(-3^n))

theorem sum_infinite_series_result (x : ℝ) (h : 1 < x) :
  sum_infinite_series x h = 1 / (x - 1) :=
sorry

end sum_infinite_series_result_l649_649962


namespace problem_l649_649908

theorem problem (
  -- Polar equation of the curve C
  (polar_eq : ∀ (ρ θ : ℝ), ρ * cos(θ - π / 3) = 1 ↔ (1 / 2) * ρ * cos θ + (√3 / 2) * ρ * sin θ = 1) :
  -- Cartesian equation of the curve C
  (cartesian_eq : ∀ (x y : ℝ), (1 / 2) * x + (√3 / 2) * y = 1 ↔ x + √3 * y = 2) :
  -- Coordinates of M and N
  (M_coords : ∃ (ρ θ : ℝ), θ = 0 ∧ ρ = 2 ∧ ρ * cos(θ - π / 3) = 1) :
  (N_coords : ∃ (ρ θ : ℝ), θ = π / 2 ∧ ρ = (2 * √3) / 3 ∧ ρ * cos(θ - π / 3) = 1) :
  -- Midpoint P and its polar equation
  (midpoint_polar : ∀ (P : ℝ × ℝ), 
      P = ((2 / 2 + 0 / 2), (0 / 2 + (2 * √3 / 3) / 2)) ∧ 
      (P.1 = 1 ∧ P.2 = √3 / 3) ∧
      (∃ (ρ θ : ℝ), (ρ, θ) = (2 * √3 / 3, π / 6)) ∧
      (∀ (line_eq : ℝ → ℝ → Prop), (∃ (ρ : ℝ), line_eq ρ (π / 6))) → 
      line_eq ρ (π / 6) → (ρ ∈ (-∞, ∞)) →
      ∀ θ, θ = π / 6) :
  True :=
sorry

end problem_l649_649908


namespace exists_kittens_on_segments_l649_649571

theorem exists_kittens_on_segments :
  ∃ (kittens : Finset (Fin 10)) (segments : Finset (Finset (Fin 10))),
  kittens.card = 10 ∧
  segments.card = 5 ∧
  ∀ s ∈ segments, s.card = 4 ∧
  (∃ p1 p2 p3 p4 ∈ kittens, s = {p1, p2, p3, p4}) := sorry

end exists_kittens_on_segments_l649_649571


namespace problem_statement_l649_649493

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
  else -- define elsewhere based on periodicity and oddness properties
    sorry 

theorem problem_statement : 
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x) → f 2015.5 = -0.5 :=
by
  intros
  sorry

end problem_statement_l649_649493


namespace smallest_number_of_candies_l649_649690

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem smallest_number_of_candies : 
  ∃ n : ℕ, (∀ k ∈ {2, 3, 4, 5, 7}, k ∣ n) ∧ (∀ m : ℕ, (∀ k ∈ {2, 3, 4, 5, 7}, k ∣ m) → n ≤ m) ∧ n = 420 :=
sorry

end smallest_number_of_candies_l649_649690


namespace number_of_perpendicular_points_on_ellipse_l649_649093

theorem number_of_perpendicular_points_on_ellipse :
  ∃ (P : ℝ × ℝ), (P ∈ {P : ℝ × ℝ | (P.1^2 / 8) + (P.2^2 / 4) = 1})
  ∧ (∀ (F1 F2 : ℝ × ℝ), F1 ≠ F2 → ∀ (P : ℝ × ℝ), ((P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2)) = 0) :=
sorry

end number_of_perpendicular_points_on_ellipse_l649_649093


namespace problem_angle_PQS_l649_649912

theorem problem_angle_PQS
  (P R S Q : Type)  
  [h1 : IsOnLine P R S]
  (h2 : AngleBisector Q P S R)
  (h3 : PQ = PR)
  (h4 : ∠ RSQ = 100)
  (h5 : ∠ RPQ = 36)
  : ∠ PQS = 32 :=
by
  sorry

end problem_angle_PQS_l649_649912


namespace tires_usage_l649_649265

theorem tires_usage :
  let total_miles := 50000
  let first_part_miles := 40000
  let second_part_miles := 10000
  let num_tires_first_part := 5
  let num_tires_total := 7
  let total_tire_miles_first := first_part_miles * num_tires_first_part
  let total_tire_miles_second := second_part_miles * num_tires_total
  let combined_tire_miles := total_tire_miles_first + total_tire_miles_second
  let miles_per_tire := combined_tire_miles / num_tires_total
  miles_per_tire = 38571 := 
by
  sorry

end tires_usage_l649_649265


namespace school_total_students_l649_649154

theorem school_total_students :
  ∀ (C1 C2 C3 C4 C5 : ℕ),
  C1 = 28 → 
  C2 = C1 - 2 →
  C3 = C2 - 2 →
  C4 = C3 - 2 →
  C5 = C4 - 2 →
  C1 + C2 + C3 + C4 + C5 = 120 := 
by {
  intros C1 C2 C3 C4 C5 hC1 hC2 hC3 hC4 hC5,
  sorry,
}

end school_total_students_l649_649154


namespace complex_magnitude_product_l649_649403

theorem complex_magnitude_product :
  (Complex.abs ((3 * Real.sqrt 3 - 3 * Complex.i) * (2 * Real.sqrt 2 + 2 * Complex.i)))
  = 12 * Real.sqrt 3 :=
by
  sorry

end complex_magnitude_product_l649_649403


namespace correct_statements_l649_649424

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem correct_statements :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (f (Real.log 3 / Real.log 2) ≠ 2) ∧
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (|x|) ≥ 0 ∧ f 0 = 0) :=
by
  sorry

end correct_statements_l649_649424


namespace ribeye_steak_cost_l649_649369

/-- Define the conditions in Lean -/
def appetizer_cost : ℕ := 8
def wine_cost : ℕ := 3
def wine_glasses : ℕ := 2
def dessert_cost : ℕ := 6
def total_spent : ℕ := 38
def tip_percentage : ℚ := 0.20

/-- Proving the cost of the ribeye steak before the discount -/
theorem ribeye_steak_cost (S : ℚ) (h : 20 + (S / 2) + (tip_percentage * (20 + S)) = total_spent) : S = 20 :=
by
  sorry

end ribeye_steak_cost_l649_649369


namespace rosie_pies_l649_649997

theorem rosie_pies (total_apples : ℕ) (apples_per_two_pies : ℕ) (extra_apples : ℕ) :
  total_apples = 36 → apples_per_two_pies = 12 → extra_apples = 2 → 
  ∃ pies : ℕ, pies = 9 :=
by
  intros h1 h2 h3
  use 9
  sorry

end rosie_pies_l649_649997


namespace value_range_of_quadratic_l649_649274

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem value_range_of_quadratic :
  ∀ x, -1 ≤ x ∧ x ≤ 2 → (2 : ℝ) ≤ quadratic_function x ∧ quadratic_function x ≤ 6 :=
by
  sorry

end value_range_of_quadratic_l649_649274


namespace circumradius_of_regular_tetrahedron_l649_649805

theorem circumradius_of_regular_tetrahedron (a : ℝ) (h : a > 0) :
    ∃ R : ℝ, R = a * (Real.sqrt 6) / 4 :=
by
  sorry

end circumradius_of_regular_tetrahedron_l649_649805


namespace arc_length_of_sector_l649_649450

-- Define the given conditions
def radius : ℝ := 4
def angle : ℝ := Real.pi / 3

-- Define the length of the arc based on given conditions
def arc_length (r : ℝ) (α : ℝ) : ℝ := α * r

-- State the theorem we need to prove
theorem arc_length_of_sector : arc_length radius angle = 4 * Real.pi / 3 :=
by sorry

end arc_length_of_sector_l649_649450


namespace half_cos_A_eq_three_fourths_area_of_triangle_l649_649078

variables {A B C a b c : ℝ}

-- Definition of the conditions
def triangle_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

def angle_condition (C A : ℝ) : Prop :=
  C = 2 * A

def area_triangle (a b c sinA : ℝ) : ℝ :=
  (1 / 2) * b * c * sinA

noncomputable def cos_A (A : ℝ) : ℝ := (3 / 4 : ℝ)

noncomputable def sin_A (A : ℝ) : ℝ := Real.sqrt (1 - cos_A A ^ 2)

noncomputable def sin_C (A : ℝ) : ℝ := 2 * sin_A A * cos_A A

-- Theorems to prove
theorem half_cos_A_eq_three_fourths (a b c A C : ℝ) 
  (ap : triangle_arithmetic_progression a b c)
  (ac : angle_condition C A) :
  Real.cos A = (3 / 4 : ℝ) :=
sorry

theorem area_of_triangle (A a : ℝ) (ha : a = 2)
  (hcA : Real.cos A = (3 / 4 : ℝ)) :
  let sinA := sin_A A,
      c := 3,
      b := (2 + c) / 2 in
  area_triangle a b c sinA = (15 * Real.sqrt 7 / 16 : ℝ) :=
sorry

end half_cos_A_eq_three_fourths_area_of_triangle_l649_649078


namespace solution_exists_l649_649024

def operation (a b : ℚ) : ℚ :=
if a ≥ b then a^2 * b else a * b^2

theorem solution_exists (m : ℚ) (h : operation 3 m = 48) : m = 4 := by
  sorry

end solution_exists_l649_649024


namespace ratio_of_times_l649_649496

-- Definitions based on conditions
variables {T D Ti A : ℝ}
def dev_time := T + 20
def tina_time := T + 5
def alex_time := T + 10
def dev_rate := 1 / dev_time
def tina_rate := 1 / tina_time
def together_rate := 1 / T

-- Combined rate of Dev and Tina
def combined_rate := dev_rate + tina_rate = together_rate

-- Main theorem statement
theorem ratio_of_times (h_main : combined_rate) : (dev_time, tina_time, alex_time) = (30, 15, 20) → 
  (6, 3, 4) :=
by
  sorry

end ratio_of_times_l649_649496


namespace not_quasi_prime_1000_l649_649826

def is_quasi_prime (q : ℕ) (seq : ℕ → ℕ) : Prop :=
  q > seq (q - 1) ∧
  ∀ i j : ℕ, 1 ≤ i → i ≤ j → j ≤ q - 1 → q ≠ seq i * seq j

def q_sequence : ℕ → ℕ
| 0 := 2
| (n + 1) := if h : ∃ q > q_sequence n, is_quasi_prime q q_sequence
             then Nat.find h
             else 0 -- this should never be used

theorem not_quasi_prime_1000 :
  ¬ is_quasi_prime 1000 q_sequence :=
sorry

end not_quasi_prime_1000_l649_649826


namespace general_formula_l649_649458

open Nat

/-- Define the sequence {a_n} -/
def a : ℕ → ℝ
| 1       := 2
| (n + 1) := 1 + a (n - 1)

lemma sequence_formula (n : ℕ) (h : 2 ≤ n) : 
  ( ∑ k in finset.range(n - 1), a k / (k + 2) ) = a n - 2 :=
sorry

theorem general_formula (n : ℕ) (h : 1 ≤ n) : a n = n + 1 :=
by
  cases n
  case zero => contradiction
  case succ =>
    cases n
    case zero => exact rfl
    case succ =>
      have : a 2 = 3 := sorry -- from a 1 = 2 and sequence_formula
      let IH := by sorry -- inductive hypothesis for a n = n + 1
      exact IH

#print axioms general_formula

end general_formula_l649_649458


namespace wall_height_to_breadth_ratio_l649_649629

theorem wall_height_to_breadth_ratio :
  ∀ (b : ℝ) (h : ℝ) (l : ℝ),
  b = 0.4 → h = n * b → l = 8 * h → l * b * h = 12.8 →
  n = 5 :=
by
  intros b h l hb hh hl hv
  sorry

end wall_height_to_breadth_ratio_l649_649629


namespace sum_infinite_series_result_l649_649961

noncomputable def sum_infinite_series (x : ℝ) (h : 1 < x) : ℝ :=
  ∑' n, 1 / (x^(3^n) - x^(-3^n))

theorem sum_infinite_series_result (x : ℝ) (h : 1 < x) :
  sum_infinite_series x h = 1 / (x - 1) :=
sorry

end sum_infinite_series_result_l649_649961


namespace first_sales_amount_l649_649694

theorem first_sales_amount (x : ℝ) : 
  (80000 / 2000000 : ℝ) = (1 / 3) * (125000 / x) → 
  x ≈ 1041667.67 := 
sorry

end first_sales_amount_l649_649694


namespace num_positive_terms_l649_649774

noncomputable def seq (n : ℕ) : ℝ := float.cos (10^((n - 1).to_real))

theorem num_positive_terms : fin 100 → seq 100 99 :=
sorry

end num_positive_terms_l649_649774


namespace base8_356_plus_base14_4CD_eq_1203_l649_649038

def base8_to_nat (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | _ =>
    let (r, m) := n.divMod 10
    8 * base8_to_nat r + m

def base14_to_nat (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | _ =>
    let (r, m) := n.divMod 100
    let (qm, rm) := m.divMod 10
    14 * (14 * base14_to_nat r + qm) + rm

theorem base8_356_plus_base14_4CD_eq_1203 :
  base8_to_nat 356 + base14_to_nat (4 * 196 + 12 * 14 + 13) = 1203 := by
sorry

end base8_356_plus_base14_4CD_eq_1203_l649_649038


namespace arithmetic_sequence_problem_l649_649156

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (d : ℝ)
  (a1 : ℝ)
  (h_arithmetic : ∀ n, a n = a1 + (n - 1) * d)
  (h_a4 : a 4 = 5) :
  2 * a 1 - a 5 + a 11 = 10 := 
by
  sorry

end arithmetic_sequence_problem_l649_649156


namespace integral_inequality_domain_l649_649179

theorem integral_inequality_domain (x y : ℝ) (f : ℝ → ℝ) (h₁ : ∀ t, f t = x * Real.sin t + y * Real.cos t):
  (|∫ t in -Real.pi..Real.pi, f t * Real.cos t| ≤ ∫ t in -Real.pi..Real.pi, (f t)^2) →
  (x^2 + (y - 0.5)^2 ≥ 0.25 ∧ x^2 + (y + 0.5)^2 ≥ 0.25) :=
begin
  sorry
end

end integral_inequality_domain_l649_649179


namespace rectangle_area_l649_649502

theorem rectangle_area (a : ℝ) : 
  let d := 5 * a 
  let l := 3 * a
  let w := sqrt (d^2 - l^2) 
  d = 5 * a ∧ l = 3 * a → l * w = 12 * a^2 := 
by 
  sorry

end rectangle_area_l649_649502


namespace chase_cardinals_count_l649_649658

variable (gabrielle_robins : Nat)
variable (gabrielle_cardinals : Nat)
variable (gabrielle_blue_jays : Nat)
variable (chase_robins : Nat)
variable (chase_blue_jays : Nat)
variable (chase_cardinals : Nat)

variable (gabrielle_total : Nat)
variable (chase_total : Nat)

variable (percent_more : Nat)

axiom gabrielle_robins_def : gabrielle_robins = 5
axiom gabrielle_cardinals_def : gabrielle_cardinals = 4
axiom gabrielle_blue_jays_def : gabrielle_blue_jays = 3

axiom chase_robins_def : chase_robins = 2
axiom chase_blue_jays_def : chase_blue_jays = 3

axiom gabrielle_total_def : gabrielle_total = gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
axiom chase_total_def : chase_total = chase_robins + chase_blue_jays + chase_cardinals
axiom percent_more_def : percent_more = 20

axiom gabrielle_more_birds : gabrielle_total = Nat.ceil ((chase_total * (100 + percent_more)) / 100)

theorem chase_cardinals_count : chase_cardinals = 5 := by sorry

end chase_cardinals_count_l649_649658


namespace Tolya_not_smaller_than_Vasya_l649_649339

def circle1000 (α : Type) := 
  {xs : list α // xs.length = 1000 ∧ ∀ (i : Nat) (h1 : i < 1000) (h2 : (i + 1) < 1000), xs.nthLe i h1 ≠ xs.nthLe (i + 1) h2}

namespace circle_proof

variables {α : Type} [linear_ordered_field α] (c : circle1000 α)

def Peter_numbers : list α := 
  (list.fin_range 1000).map (λ i, abs (c.val.nthLe i (by simp) - c.val.nthLe (i + 1) (by simp)))

def Vasya_numbers : list α := 
  (list.fin_range 1000).map (λ i, abs (c.val.nthLe i (by simp) - c.val.nthLe (i + 2) (by simp)))

def Tolya_numbers : list α := 
  (list.fin_range 1000).map (λ i, abs (c.val.nthLe i (by simp) - c.val.nthLe (i + 3) (by simp)))

theorem Tolya_not_smaller_than_Vasya (h : ∀ p ∈ Peter_numbers c, p ≥ 2 * (list.maximum (Vasya_numbers c)).get_or_else 0) : 
  ∀ t ∈ Tolya_numbers c, t ≥ (list.maximum (Vasya_numbers c)).get_or_else 0 :=
sorry

end circle_proof

end Tolya_not_smaller_than_Vasya_l649_649339


namespace angle_C_side_sum_l649_649509

noncomputable def problem1 (a b c A B C : ℝ) := 
  (2 * b - a) * real.cos (A + B) = -c * real.cos A

noncomputable def problem2 (c S a b : ℝ) := 
  c = 3 ∧ S = (4 * real.sqrt 3) / 3

theorem angle_C (a b c A B C : ℝ) : problem1 a b c A B C → C = real.pi / 3 := 
  by
  sorry

theorem side_sum (a b : ℝ) (c S: ℝ) : problem2 c S a b → a + b = 5 :=
  by
  sorry

end angle_C_side_sum_l649_649509


namespace find_N_l649_649048

theorem find_N : ∃ (N : ℕ), (1000 ≤ N ∧ N < 10000) ∧ (N^2 % 10000 = N) ∧ (N % 16 = 7) ∧ N = 3751 := 
by sorry

end find_N_l649_649048


namespace final_result_l649_649200

def problem_statement : ℝ :=
  let x := 1 + real.sqrt 3 / (1 + real.sqrt 3 / (1 + real.sqrt 3 / (1 + real.sqrt 3 / (1 + real.sqrt 3)))) in
  let A := 6 in
  let B := 3 in
  let C := -33 in
  |A| + |B| + |C|

theorem final_result : problem_statement = 42 :=
by sorry

end final_result_l649_649200


namespace slope_parallel_l649_649647

theorem slope_parallel {x y : ℝ} (h : 3 * x - 6 * y = 15) : 
  ∃ m : ℝ, m = -1/2 ∧ ( ∀ (x1 x2 : ℝ), 3 * x1 - 6 * y = 15 → ∃ y1 : ℝ, y1 = m * x1) :=
by
  sorry

end slope_parallel_l649_649647


namespace abs_inequality_k_ge_neg3_l649_649811

theorem abs_inequality_k_ge_neg3 (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k ≥ -3 :=
sorry

end abs_inequality_k_ge_neg3_l649_649811


namespace vasya_max_triangles_l649_649298

theorem vasya_max_triangles (n : ℕ) (h1 : n = 100)
  (h2 : ∀ (a b c : ℕ), a + b ≤ c ∨ b + c ≤ a ∨ c + a ≤ b) :
  ∃ (t : ℕ), t = n := 
sorry

end vasya_max_triangles_l649_649298


namespace cos450_eq_zero_l649_649729

theorem cos450_eq_zero (cos_periodic : ∀ x, cos (x + 360) = cos x) (cos_90_eq_zero : cos 90 = 0) :
  cos 450 = 0 := by
  sorry

end cos450_eq_zero_l649_649729


namespace part1_tangent_line_at_point_part2_find_range_of_a_part3_inequality_log_expr_l649_649109

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * real.log x - (a / 2) * x^2
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - x

theorem part1_tangent_line_at_point {a : ℝ} (h : a = 1) :
  let f_x := f 1 a
  let f_prime_x := derivative (λ x, f x a) 1
  f_prime_x = 0 → (y = f_x) = (y = -1/2) := by
  sorry

theorem part2_find_range_of_a {a : ℝ} :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g' x1 a = 0 ∧ g' x2 a = 0) → 
  0 < a ∧ a < 1 / real.exp 1 := by
  sorry

theorem part3_inequality_log_expr {x1 x2 : ℝ} {a : ℝ} 
  (hx1 : 0 < x1) (hx2 : 0 < x2) (hx1x2 : x1 ≠ x2) 
  (hg1 : g' x1 a = 0) (hg2 : g' x2 a = 0)
  (ha : 0 < a ∧ a < 1 / real.exp 1) :
  1 / real.log x1 + 1 / real.log x2 > 2 := by
  sorry

end part1_tangent_line_at_point_part2_find_range_of_a_part3_inequality_log_expr_l649_649109


namespace max_size_subset_l649_649548

-- Define the set T
def T : Set ℕ := {n | ∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 200 ∧ 0 ≤ b ∧ b ≤ 100 ∧ 0 ≤ c ∧ c ≤ 100 ∧ n = 2^a * 3^b * 167^c}

-- Define property P for no element being a multiple of another
def P (S : Set ℕ) : Prop := ∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → ¬ (x ∣ y)

-- Maximum size of S
theorem max_size_subset (S : Set ℕ) (hS : S ⊆ T) (hP : P S) : ∃ (N : ℕ), N = 101^2 ∧ ∀ (S' : Set ℕ), S' ⊆ T → P S' → S'.card ≤ N := 
sorry

end max_size_subset_l649_649548


namespace inequality_proof_l649_649860

variables (a b c d : ℝ)

theorem inequality_proof 
  (h1 : a + b > abs (c - d)) 
  (h2 : c + d > abs (a - b)) : 
  a + c > abs (b - d) := 
sorry

end inequality_proof_l649_649860


namespace trig_problem_l649_649433

theorem trig_problem (x : ℝ) (h1 : cos (x - π / 4) = sqrt 2 / 10) (h2 : x ∈ Ioo (π / 2) (3 * π / 4)) :
  sin x = 4 / 5 ∧ sin (2 * x + π / 6) = -(7 + 24 * sqrt 3) / 50 :=
by sorry

end trig_problem_l649_649433


namespace cosine_450_eq_0_l649_649733

theorem cosine_450_eq_0 : Real.cos (450 * Real.pi / 180) = 0 :=
by
  -- Convert 450 degrees to radians
  have h1 : (450 : ℝ) = 360 + 90 := by norm_num
  -- Use the periodic property of cosine
  have h2 : ∀ x : ℝ, Real.cos (x + 2 * Real.pi) = Real.cos x := Real.cos_periodic (2 * Real.pi)
  -- Since 450 degrees = 360 degrees + 90 degrees
  rw [h1, Real.cos_add, h2 (90 * Real.pi / 180)]
  -- Convert 90 degrees to radians and solve
  rw [Real.pi_mul_base_div, Real.pi_div_two, Real.cos_pi_div_two]
  norm_num

end cosine_450_eq_0_l649_649733


namespace complex_number_sum_l649_649273

theorem complex_number_sum : (λ i : ℂ, i + i^2 + i^3 + i^4) I = 0 := 
by
  sorry

end complex_number_sum_l649_649273


namespace magnitude_product_l649_649404

def z1 : ℂ := 3 * Real.sqrt 3 - 3 * Complex.i
def z2 : ℂ := 2 * Real.sqrt 2 + 2 * Complex.i

theorem magnitude_product :
  Complex.abs (z1 * z2) = 12 * Real.sqrt 3 :=
by 
  sorry

end magnitude_product_l649_649404


namespace sum_of_possible_values_l649_649263

theorem sum_of_possible_values (N : ℂ) (h : N * (N - 8) = 7) : 
  let roots := (N^2 - 8*N - 7).roots
  ∑ root in roots, root = 8 := 
sorry

end sum_of_possible_values_l649_649263


namespace solve_for_x_l649_649591

theorem solve_for_x (x : ℝ) :
  (1 / 8)^(3*x + 12) = 64^(3*x + 7) -> x = -26 / 9 :=
by
  sorry

end solve_for_x_l649_649591


namespace part1_cos_part1_sin_part2_l649_649158

variable (α β γ : ℝ)

theorem part1_cos : |cos (α + β)| ≤ |cos α| + |sin β| := sorry

theorem part1_sin : |sin (α + β)| ≤ |cos α| + |cos β| := sorry

theorem part2 (h : α + β + γ = 0) :
  |cos α| + |cos β| + |cos γ| ≥ 1 :=
sorry

end part1_cos_part1_sin_part2_l649_649158


namespace opposite_of_7_eq_neg7_l649_649260

theorem opposite_of_7_eq_neg7 : ∃ x : ℝ, 7 + x = 0 ∧ x = -7 :=
by
  use -7
  constructor
  · exact (add_left_neg 7).symm
  · rfl

end opposite_of_7_eq_neg7_l649_649260


namespace cos_450_eq_0_l649_649743

theorem cos_450_eq_0 : Real.cos (450 * Real.pi / 180) = 0 := by
  -- Angle equivalence: 450 degrees is equivalent to 90 degrees on the unit circle
  have angle_eq : (450 : Real) * Real.pi / 180 = (90 : Real) * Real.pi / 180 := by
    calc
      (450 : Real) * Real.pi / 180
        = (450 / 180) * Real.pi : by rw [mul_div_assoc]
        = (5 * 90 / 180) * Real.pi : by norm_num
        = (5 * 90 / (2 * 90)) * Real.pi : by norm_num
        = (5 / 2) * Real.pi : by norm_num

  -- Now use this equivalence: cos(450 degrees) = cos(90 degrees)
  have cos_eq : Real.cos (450 * Real.pi / 180) = Real.cos (90 * Real.pi / 180) := by
    rw [angle_eq]

  -- Using the fact that cos(90 degrees) = 0
  have cos_90 : Real.cos (90 * Real.pi / 180) = 0 := by
    -- This step can use a known trigonometric fact from mathlib
    exact Real.cos_pi_div_two

  -- Therefore
  rw [cos_eq, cos_90]
  exact rfl

end cos_450_eq_0_l649_649743


namespace water_required_to_prepare_saline_solution_l649_649900

theorem water_required_to_prepare_saline_solution (water_ratio : ℝ) (required_volume : ℝ) : 
  water_ratio = 3 / 8 ∧ required_volume = 0.64 → required_volume * water_ratio = 0.24 :=
by
  sorry

end water_required_to_prepare_saline_solution_l649_649900


namespace cosine_450_eq_0_l649_649732

theorem cosine_450_eq_0 : Real.cos (450 * Real.pi / 180) = 0 :=
by
  -- Convert 450 degrees to radians
  have h1 : (450 : ℝ) = 360 + 90 := by norm_num
  -- Use the periodic property of cosine
  have h2 : ∀ x : ℝ, Real.cos (x + 2 * Real.pi) = Real.cos x := Real.cos_periodic (2 * Real.pi)
  -- Since 450 degrees = 360 degrees + 90 degrees
  rw [h1, Real.cos_add, h2 (90 * Real.pi / 180)]
  -- Convert 90 degrees to radians and solve
  rw [Real.pi_mul_base_div, Real.pi_div_two, Real.cos_pi_div_two]
  norm_num

end cosine_450_eq_0_l649_649732


namespace find_expression_for_positive_x_l649_649844

-- Definition of an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Conditions from the problem
def f : ℝ → ℝ := λ x, if x < 0 then x - x^4 else 0

-- Statement to be proved
theorem find_expression_for_positive_x (x : ℝ) (h_even : even_function f) (h_cond : ∀ x : ℝ, x < 0 → f x = x - x^4) : 
  x > 0 → f x = -x^4 - x :=
by
  sorry

end find_expression_for_positive_x_l649_649844


namespace scientific_notation_l649_649676

theorem scientific_notation (n : ℝ) (h : n = 1300000) : n = 1.3 * 10^6 :=
by {
  sorry
}

end scientific_notation_l649_649676


namespace apex_angle_of_cone_l649_649505

-- Definitions based on the problem conditions
def circumference_of_base (R : ℝ) : ℝ := π * R
def diameter_of_base (R : ℝ) : ℝ := R

-- Proof problem statement
theorem apex_angle_of_cone (R : ℝ) (condition : circumference_of_base R = π * R) : 
  diameter_of_base R = R → 
  apex_angle = 60 :=
begin
  sorry
end

end apex_angle_of_cone_l649_649505


namespace relationship_between_abc_l649_649031

noncomputable def a : ℝ := 3 ^ 0.2
noncomputable def b : ℝ := 0.2 ^ 3
noncomputable def c : ℝ := Real.log 3 / Real.log 0.2

theorem relationship_between_abc : c < b ∧ b < a := by
  sorry

end relationship_between_abc_l649_649031


namespace fixed_point_on_line_find_m_values_l649_649069

-- Define the conditions and set up the statements to prove

/-- 
Condition 1: Line equation 
-/
def line_eq (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

/-- 
Condition 2: Circle equation 
-/
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 25

/-- 
Question (1): Fixed point (3,1) is always on the line
-/
theorem fixed_point_on_line (m : ℝ) : line_eq m 3 1 := by
  sorry

/-- 
Question (2): Finding the values of m for the given chord length
-/
theorem find_m_values (m : ℝ) (h_chord : ∀x y : ℝ, circle_eq x y → line_eq m x y → (x - y)^2 = 6) : 
  m = -1/2 ∨ m = 1/2 := by
  sorry

end fixed_point_on_line_find_m_values_l649_649069


namespace eight_hash_six_l649_649054

def op (r s : ℝ) : ℝ := sorry

axiom op_r_zero (r : ℝ): op r 0 = r + 1
axiom op_comm (r s : ℝ) : op r s = op s r
axiom op_r_add_one_s (r s : ℝ): op (r + 1) s = (op r s) + s + 2

theorem eight_hash_six : op 8 6 = 69 := 
by sorry

end eight_hash_six_l649_649054


namespace James_beverages_per_day_l649_649919

def total_sodas := 4 * 10 + 12
def total_juice_boxes := 3 * 8 + 5
def total_water_bottles := 2 * 15
def total_energy_drinks := 7
def total_beverages := total_sodas + total_juice_boxes + total_water_bottles + total_energy_drinks
def number_of_days := 14
def beverages_per_day := total_beverages / number_of_days

theorem James_beverages_per_day : beverages_per_day = 8 := by
  have total_beverages_eq : total_beverages = 118 := by 
    calc 
      total_sodas + total_juice_boxes + total_water_bottles + total_energy_drinks
        = 52 + 29 + 30 + 7 := by simp [total_sodas, total_juice_boxes, total_water_bottles, total_energy_drinks]
        = 118 := by norm_num
  calc
    total_beverages / number_of_days
      = 118 / 14 := by rw [total_beverages_eq, number_of_days]
      = 8 := by norm_num

end James_beverages_per_day_l649_649919


namespace susan_apple_ratio_l649_649122

theorem susan_apple_ratio {R : ℝ} : 
  let greg_apples := 9 in
  let sarah_apples := 9 in
  let susan_apples := R * 9 in
  let mark_apples := R * 9 - 5 in
  let mom_apples := 49 in
  (greg_apples + sarah_apples + susan_apples + mark_apples = mom_apples) →
  R = 2 :=
by
  sorry

end susan_apple_ratio_l649_649122


namespace part1_part2_l649_649858

theorem part1 (x p : ℝ) (h : abs p ≤ 2) : (x^2 + p * x + 1 > 2 * x + p) ↔ (x < -1 ∨ 3 < x) := 
by 
  sorry

theorem part2 (x p : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : (x^2 + p * x + 1 > 2 * x + p) ↔ (-1 < p) := 
by 
  sorry

end part1_part2_l649_649858


namespace cost_per_bag_proof_minimize_total_cost_l649_649568

-- Definitions of given conditions
variable (x y : ℕ) -- cost per bag for brands A and B respectively
variable (m : ℕ) -- number of bags of brand B

def first_purchase_eq := 100 * x + 150 * y = 7000
def second_purchase_eq := 180 * x + 120 * y = 8100
def cost_per_bag_A : ℕ := 25
def cost_per_bag_B : ℕ := 30
def total_bags := 300
def constraint := (300 - m) ≤ 2 * m

-- Prove the costs per bag
theorem cost_per_bag_proof (h1 : first_purchase_eq x y)
                           (h2 : second_purchase_eq x y) :
  x = cost_per_bag_A ∧ y = cost_per_bag_B :=
sorry

-- Define the cost function and prove the purchase strategy
def total_cost (m : ℕ) : ℕ := 25 * (300 - m) + 30 * m

theorem minimize_total_cost (h : constraint m) :
  m = 100 ∧ total_cost 100 = 8000 :=
sorry

end cost_per_bag_proof_minimize_total_cost_l649_649568


namespace find_x_l649_649521

theorem find_x 
    (ACD_right : ∠ ACD = 90) 
    (DCB_60 : ∠ DCB = 60) 
    (BCA_30 : ∠ BCA = 30) 
    (ACB_eq_x : ∠ ACB = x) 
    (line_AB : A - B is a line) : 
    x = 90 := 
by 
    sorry

end find_x_l649_649521


namespace stratified_sampling_correct_l649_649709

-- Definitions of conditions
def total_staff := 160
def business_personnel := 112
def management_personnel := 16
def logistics_personnel := 32

def sample_size : ℕ := 20
def ratio := (7, 1, 2)
def x := sample_size / (ratio.1 + ratio.2 + ratio.3)

-- Number of individuals to be drawn from each group
def business_draw := ratio.1 * x
def management_draw := ratio.2 * x
def logistics_draw := ratio.3 * x

-- Proof statement
theorem stratified_sampling_correct :
  business_draw = 14 ∧ management_draw = 2 ∧ logistics_draw = 4 :=
by
  have h : x = 2 := by
    -- Calculation for x
    simp [ratio, sample_size]
    sorry
  simp [business_draw, management_draw, logistics_draw, h]
  sorry

end stratified_sampling_correct_l649_649709


namespace orchestra_french_horn_players_l649_649224

open Nat

theorem orchestra_french_horn_players :
  ∃ (french_horn_players : ℕ), 
  french_horn_players = 1 ∧
  1 + 6 + 5 + 7 + 1 + french_horn_players = 21 :=
by
  sorry

end orchestra_french_horn_players_l649_649224


namespace count_integer_solutions_l649_649194

theorem count_integer_solutions (n : ℕ) (h_pos : 0 < n) :
  let A := { x : Fin (2 * n) → ℤ // (∀ i, 0 ≤ x i ∧ x i ≤ 1) ∧ 
                                       (x 0 + ... + x i < (1 / 2) * (i + 1) ∀ i < 2 * n - 1) ∧ 
                                       (x 0 + ... + x (2 * n - 1) = n) } in
  |A| = (1 / n) * Nat.choose (2 * n - 2) (n - 1) := by
  sorry

end count_integer_solutions_l649_649194


namespace periodic_even_function_value_l649_649462

theorem periodic_even_function_value (f : ℝ → ℝ)
  (h_even : ∀ x, f(x) = f(-x))
  (h_periodic : ∀ x, f(x) = f(x + 2))
  (h_interval : ∀ x, 0 < x ∧ x < 1 → f(x) = 2^x - 1) :
  f(real.log 12 / real.log 2) = -(2 / 3) := 
begin
  sorry,
end

end periodic_even_function_value_l649_649462


namespace zero_point_interval_l649_649276

-- Definition of the function
def f(x : ℝ) : ℝ := Real.log x - 1 / x

-- Defining the assumptions and the theorem
theorem zero_point_interval (x₀ k : ℝ) (k_int : k ∈ Set.Ioo 0 (k + 1)) :
  f x₀ = 0 → (k : ℝ) ∈ ℤ :=
  sorry

end zero_point_interval_l649_649276


namespace unique_x_star_23_eq_4_l649_649187

def star (a b : ℝ) : ℝ :=
  (Real.sqrt (a + b) / Real.sqrt ((a - b)^2))

theorem unique_x_star_23_eq_4 : ∃! x : ℝ, star x 23 = 4 ∧ x > 23 :=
sorry

end unique_x_star_23_eq_4_l649_649187


namespace find_b_for_continuity_l649_649978

def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 3 then 3 * x ^ 2 + 2 * x - 1 else b * x - 2

theorem find_b_for_continuity (b : ℝ) : (∀ x, f 3 b = f 3 b) → b = 34 / 3 :=
by
  sorry

end find_b_for_continuity_l649_649978


namespace angle_equality_l649_649195

-- Define the geometric setup
theorem angle_equality
  (C₁ C₂ : Circle)
  (P Q : Point)
  (A B C D : Point)
  (d : Line)
  (h1 : P ≠ Q)
  (h2 : d ∩ C₁ = {A, C})
  (h3 : d ∩ C₂ = {B, D})
  (h4 : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ D ∧ A ≠ C ∧ B ≠ D) -- Ensure AB corresponds strictly
  (h5 : ∀ X Y Z ∈ ({A, B, C, D}) , X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) -- Points are distinct
  (h6 : P ∈ C₁ ∩ C₂)
  (h7 : Q ∈ C₁ ∩ C₂) :
  ∠APB = ∠CQD :=
by
sorry

end angle_equality_l649_649195


namespace chord_circle_region_count_l649_649294

-- Define the maximum number of regions M and minimum number of regions m
-- based on the provided conditions.
theorem chord_circle_region_count (M m : ℕ) (hM : M = 14) (hm : m = 10) :
  M^2 + m^2 = 296 :=
by
  rw [hM, hm]
  simp
  exact sorry

end chord_circle_region_count_l649_649294


namespace bhanu_spends_on_petrol_l649_649008

-- Define the conditions as hypotheses
variable (income : ℝ)
variable (spend_on_rent : income * 0.7 * 0.14 = 98)

-- Define the theorem to prove
theorem bhanu_spends_on_petrol : (income * 0.3 = 300) :=
by
  sorry

end bhanu_spends_on_petrol_l649_649008


namespace statement_1_statement_2_statement_3_statement_4_l649_649471

theorem statement_1 (a b : ℝ) (ha : a > 0) (hb : b > 0) : (a + b) * ((1 / a) + (1 / b)) ≥ 4 :=
sorry

theorem statement_2 (a b : ℝ) : a^2 + b^2 + 3 > 2 * a + 2 * b :=
sorry

theorem statement_3 (a b m : ℝ) (hm : m > 0) (ha : a > b) (hb : b > 0) : (b / a) < ((b + m) / (a + m)) :=
sorry

theorem statement_4 : let a := (2 - real.sqrt 5)
                     let b := (real.sqrt 5 - 2)
                     let c := (5 - 2 * real.sqrt 5)
                     in c > b ∧ b > a :=
by
  let a := (2 - real.sqrt 5)
  let b := (real.sqrt 5 - 2)
  let c := (5 - 2 * real.sqrt 5)
  sorry

end statement_1_statement_2_statement_3_statement_4_l649_649471


namespace sum_series_eq_l649_649956

open Real

theorem sum_series_eq (x : ℝ) (h : 1 < x) : 
  (∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (- (3 ^ n)))) = 1 / (x - 1) :=
sorry

end sum_series_eq_l649_649956


namespace point_on_line_l649_649506

theorem point_on_line (x : ℝ) : 
  (∃ x, (x, 3) ∈ line_through (2, 14) (8, 2)) ↔ x = 7.5 :=
by
  sorry

end point_on_line_l649_649506


namespace triangle_is_right_angled_l649_649161

-- Define the internal angles of a triangle
variables (A B C : ℝ)
-- Condition: A, B, C are internal angles of a triangle
-- This directly implies 0 < A, B, C < pi and A + B + C = pi

-- Internal angles of a triangle sum to π
axiom angles_sum_pi : A + B + C = Real.pi

-- Condition given in the problem
axiom sin_condition : Real.sin A = Real.sin C * Real.cos B

-- We need to prove that triangle ABC is right-angled
theorem triangle_is_right_angled : C = Real.pi / 2 :=
by
  sorry

end triangle_is_right_angled_l649_649161


namespace binomial_sum_sum_of_n_values_l649_649651

theorem binomial_sum (n : ℕ) (h : nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15) : n = 13 ∨ n = 15 := sorry

theorem sum_of_n_values : ∑ n in {n | nat.choose 28 14 + nat.choose 28 n = nat.choose 29 15}.to_finset, n = 28 :=
by
  apply finset.sum_eq_from_set,
  intros x hx,
  cases binomial_sum x hx,
  { simp [h], },
  { simp [h], }

end binomial_sum_sum_of_n_values_l649_649651


namespace total_students_in_class_l649_649279

def total_students (H : Nat) (hands_per_student : Nat) (consider_teacher : Nat) : Nat :=
  (H / hands_per_student) + consider_teacher

theorem total_students_in_class (H : Nat) (hands_per_student : Nat) (consider_teacher : Nat) 
  (H_eq : H = 20) (hands_per_student_eq : hands_per_student = 2) (consider_teacher_eq : consider_teacher = 1) : 
  total_students H hands_per_student consider_teacher = 11 := by
  sorry

end total_students_in_class_l649_649279


namespace g_23_eq_84_l649_649488

noncomputable def g : ℕ → ℕ
| 3     := 1
| 4     := 1
| 5     := 1
| 6     := 2
| 7     := 2
| (n+8) := g(n+4) + 2 * g(n+3) + 2 * g(n+2)

theorem g_23_eq_84 : g 23 = 84 := by
  show g (3 + 20) = 84
  sorry

end g_23_eq_84_l649_649488


namespace find_A_find_B_substitution_find_b_for_diff_l649_649090

variables {A B a b : ℝ}

-- Given conditions
def cond1 := 2 * A + B = 7 * a * b + 6 * a - 2 * b - 11
def cond2 := 2 * B - A = 4 * a * b - 3 * a - 4 * b + 18

-- To prove statements
theorem find_A 
  (h1 : cond1)
  (h2 : cond2) : A = 2 * a * b + 3 * a - 8 := sorry

theorem find_B_substitution 
  (h1 : cond1)
  (h2 : cond2)
  (hab : a * b = 1) 
  (hA : A = 0): B = 7 := sorry

theorem find_b_for_diff
  (h1 : cond1)
  (h2 : cond2) 
  (h7 : ∀ a, B - A = 7): b = 3 := sorry

end find_A_find_B_substitution_find_b_for_diff_l649_649090


namespace find_integer_l649_649800

theorem find_integer (n : ℤ) (h1 : n ≥ 50) (h2 : n ≤ 100) (h3 : n % 7 = 0) (h4 : n % 9 = 3) (h5 : n % 6 = 3) : n = 84 := 
by 
  sorry

end find_integer_l649_649800


namespace probability_properties_l649_649364

/--
Theorem: The correct statement among the following is: "An event with a probability of 0 will definitely not occur."
Conditions:
  1. Probability is the same as frequency.
  2. An event with a probability of 1 might not occur.
  3. An event with a probability of 0 will definitely not occur.
  4. Probability cannot be an irrational number.
-/
theorem probability_properties :
  (∀ (P event : ℝ), P = event.frequency ∨ 
    (P = 1 → event.does_not_occur) ∨ 
    (P = 0 ↔ event.does_not_occur) ∨ 
    (∀ (P : ℝ), irrational P → false)) →
  ∃ statement : ℝ → Prop, (statement 0) :=
by
  sorry

end probability_properties_l649_649364


namespace cost_to_repaint_l649_649212

-- Definitions based on given conditions
def length : ℝ := 4
def width : ℝ := 3
def height : ℝ := 3
def area_doors_windows : ℝ := 4.7
def paint_per_sqm : ℝ := 0.6
def liters_per_bucket : ℝ := 4.5
def cost_per_bucket : ℝ := 286

-- Total cost function
def total_cost : ℝ := cost_per_bucket * (((2 * (length * height + width * height) - area_doors_windows) * paint_per_sqm) / liters_per_bucket).ceil

-- Theorem to prove the total cost equals the expected value
theorem cost_to_repaint (total_cost: ℝ) : total_cost = cost_per_bucket * (((2 * (length * height + width * height) - area_doors_windows) * paint_per_sqm) / liters_per_bucket).ceil := sorry

end cost_to_repaint_l649_649212


namespace largest_neg_int_solution_l649_649413

theorem largest_neg_int_solution :
  ∃ x : ℤ, 26 * x + 8 ≡ 4 [ZMOD 18] ∧ ∀ y : ℤ, 26 * y + 8 ≡ 4 [ZMOD 18] → y < -14 → false :=
by
  sorry

end largest_neg_int_solution_l649_649413


namespace quadratic_function_properties_l649_649104

noncomputable def f : ℝ → ℝ := λ x, (1/2) * (x - 2)^2 - 1/2

theorem quadratic_function_properties :
  -- Given conditions
  (f(-1) = 4 ∧ f(1) = 0 ∧ f(3) = 0) ∧ 
  -- Part (Ⅰ): Expression and minimum value
  (f = λ x, (1/2) * (x - 2)^2 - 1/2 ∧ ∀ x, f(x) ≥ -1/2 ∧ f(2) = -1/2) ∧
  -- Part (Ⅱ): Symmetry and existence of m
  (∃ m : ℝ, m = 4 ∧ ∀ x1 x2 : ℝ, x1 + x2 = m → f(x1) = f(x2)) :=
by
  -- The proof is omitted (sorry)
  sorry

end quadratic_function_properties_l649_649104


namespace infinite_series_sum_l649_649938

theorem infinite_series_sum (x : ℝ) (h : x > 1) :
  ∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (-(3 ^ n))) = 1 / (x - 1) :=
sorry

end infinite_series_sum_l649_649938


namespace find_m_if_parallel_l649_649481

theorem find_m_if_parallel 
  (m : ℚ) 
  (a : ℚ × ℚ := (-2, 3)) 
  (b : ℚ × ℚ := (1, m - 3/2)) 
  (h : ∃ k : ℚ, (a.1 = k * b.1) ∧ (a.2 = k * b.2)) : 
  m = 0 := 
  sorry

end find_m_if_parallel_l649_649481


namespace OC_expression_l649_649683

theorem OC_expression
  (O A B C : Point)
  (r : ℝ) 
  (h1 : radius O = 2)
  (h2 : circle_contains O A )
  (h3 : tangent_segment AB O A)
  (h4 : angle A O B = 2 * θ)
  (h5 : point_on_line C O A)
  (h6 : angle_bisector B C O A):

  OC = (2: ℝ) / (1 + Real.sin (2 * θ)) :=
sorry

end OC_expression_l649_649683


namespace exists_row_or_column_with_sqrt_n_distinct_numbers_l649_649323

theorem exists_row_or_column_with_sqrt_n_distinct_numbers
  (n : ℕ) (table : fin n → fin n → ℕ)
  (h_table_numbers : ∀ i j, table i j ∈ fin n)
  (h_table_counts : ∀ k : fin n, finset.card (finset.univ.filter (λ p : fin n × fin n, table p.1 p.2 = k)) = n) :
  ∃ i, finset.card (finset.image (λ j, table i j) finset.univ) ≥ nat.sqrt n ∨
      ∃ j, finset.card (finset.image (λ i, table i j) finset.univ) ≥ nat.sqrt n := 
sorry

end exists_row_or_column_with_sqrt_n_distinct_numbers_l649_649323


namespace projection_of_vec_c_onto_vec_b_l649_649119

def vec (x y : ℝ) : Prod ℝ ℝ := (x, y)

noncomputable def projection_of_c_onto_b := 
  let a := vec 2 3
  let b := vec (-4) 7
  let c := vec (-2) (-3)
  let dot_product_c_b := (-2) * (-4) + (-3) * 7
  let magnitude_b := Real.sqrt ((-4)^2 + 7^2)
  dot_product_c_b / magnitude_b
  
theorem projection_of_vec_c_onto_vec_b : 
  let a := vec 2 3
  let b := vec (-4) 7
  let c := vec (-2) (-3)
  let projection := projection_of_c_onto_b
  a + c = vec 0 0 ->
  projection = - Real.sqrt 65 / 5 := by
    sorry

end projection_of_vec_c_onto_vec_b_l649_649119


namespace speed_of_A_l649_649679

theorem speed_of_A :
  ∀ (v_A : ℝ), 
    (v_A * 2 + 7 * 2 = 24) → 
    v_A = 5 :=
by
  intro v_A
  intro h
  have h1 : v_A * 2 = 10 := by linarith
  have h2 : v_A = 5 := by linarith
  exact h2

end speed_of_A_l649_649679


namespace zeros_of_g_l649_649896

theorem zeros_of_g (a b : ℝ) (h : 2 * a + b = 0) :
  (∃ x : ℝ, (b * x^2 - a * x = 0) ∧ (x = 0 ∨ x = -1 / 2)) :=
by
  sorry

end zeros_of_g_l649_649896


namespace area_percentage_change_is_correct_l649_649893

-- Defining the original length l and width w as positive real numbers
variables {l w : ℝ} (hl : l > 0) (hw : w > 0)

-- Define the new length after a 30% increase
def new_length := 1.3 * l

-- Define the new width after a 15% increase followed by a 5% decrease
def new_width := 1.0925 * w

-- Calculate the original area
def original_area := l * w

-- Calculate the new area after length and width changes
def new_area := new_length l * new_width w

-- Define the percentage change in area
def percentage_change_in_area := (new_area l w / original_area l w) * 100 - 100

-- Prove that the percentage change in the area is 41.925%
theorem area_percentage_change_is_correct : percentage_change_in_area hl hw = 41.925 :=
by sorry

end area_percentage_change_is_correct_l649_649893


namespace monotonically_decreasing_functions_in_interval_l649_649112

noncomputable def f1 (x : ℝ) : ℝ := real.sqrt x
noncomputable def f2 (x : ℝ) : ℝ := real.log (x + 1) / real.log 0.5
noncomputable def f3 (x : ℝ) : ℝ := abs (x - 1)
noncomputable def f4 (x : ℝ) : ℝ := real.exp ((x + 1) * real.log 2)

theorem monotonically_decreasing_functions_in_interval :
  ∀ (f : ℝ → ℝ) (I : set ℝ),
  I = set.Ioo 0 1 →
  (∀ x y ∈ I, x < y → f y < f x) →
  (f = f2 ∨ f = f3) :=
by sorry

end monotonically_decreasing_functions_in_interval_l649_649112


namespace polynomial_evaluation_l649_649558

theorem polynomial_evaluation (p : ℝ[X]) (h_monic : p.monic) (h_deg : p.degree = 4)
  (h1 : p.eval 1 = 18) (h2 : p.eval 2 = 36) (h3 : p.eval 3 = 54) :
  p.eval 0 + p.eval 5 = 210 := by
  sorry

end polynomial_evaluation_l649_649558


namespace isosceles_triangle_height_eq_base_iff_constant_perimeter_l649_649786

-- Definitions of isosceles triangle and inscribed rectangle.
structure IsoscelesTriangle (α : Type*) :=
  (A B C : α)
  (CA_eq_CB : (distance C A = distance C B))
  (height_eq_base : Prop)

structure Rectangle (α : Type*) :=
  (E F G H : α)
  (inscribed_in : IsoscelesTriangle α)
  (perimeter_constant : Prop)

-- Lean 4 statement for the proof problem.
theorem isosceles_triangle_height_eq_base_iff_constant_perimeter
  {α : Type*} [metric_space α] {T : IsoscelesTriangle α}
  (R : Rectangle α) :
  (T.height_eq_base ↔ R.perimeter_constant) :=
sorry

end isosceles_triangle_height_eq_base_iff_constant_perimeter_l649_649786


namespace simplify_sqrt7_pow6_l649_649588

theorem simplify_sqrt7_pow6 :
  (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt7_pow6_l649_649588


namespace trigonometric_quadrant_l649_649492

theorem trigonometric_quadrant (θ : ℝ) (h : sin θ * tan θ > 0) : 
  (θ > 0 ∧ θ < π/2) ∨ (θ > π/2 ∧ θ < π) :=
by sorry

end trigonometric_quadrant_l649_649492


namespace area_of_sector_l649_649843

theorem area_of_sector (l : ℝ) (α : ℝ) (h_l : l = 5) (h_α : α = 5) : 
  ∃ S : ℝ, S = (1/2) * (l * (l / α)) ∧ S = 5 / 2 :=
by
  use (1/2) * (l * (l / α))
  split
  { rfl }
  { rw [h_l, h_α]
    norm_num }
  sorry

end area_of_sector_l649_649843


namespace shortest_chord_eq_l649_649841

-- Define the point M
def M : Point := ⟨1, 0⟩

-- Define the circle C with center and radius
def CircleC (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 5

-- Define the statement to prove
theorem shortest_chord_eq : 
    (∃ (x y : ℝ), CircleC x y ∧ M = ⟨x, y⟩ → x + y = 1) := sorry

end shortest_chord_eq_l649_649841


namespace inequality_solution_l649_649232

theorem inequality_solution (x : ℝ) :
  (| (3 * x - 2) / (x - 2) | > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by
  sorry

end inequality_solution_l649_649232


namespace yellow_tint_percentage_correct_l649_649348

-- Given conditions
def original_volume : ℕ := 40
def yellow_tint_percent : ℝ := 35 / 100
def added_yellow_tint : ℝ := 7
def removed_red_tint : ℝ := 3

-- Math proof problem
theorem yellow_tint_percentage_correct :
  let original_yellow_tint := yellow_tint_percent * original_volume
  let total_yellow_tint := original_yellow_tint + added_yellow_tint
  let new_volume := original_volume + added_yellow_tint - removed_red_tint
  (total_yellow_tint / new_volume) * 100 ≈ 47.73 :=
by
  sorry

end yellow_tint_percentage_correct_l649_649348


namespace hyperbola_eccentricity_l649_649092

noncomputable def hyperbola_description 
(a b : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
∀ (F1 F2 P : ℝ × ℝ), let e := Real.sq 2 in
-- Conditions from the problem statement
(F1 = ((-a, 0) : ℝ × ℝ)) ∧ (F2 = ((a, 0) : ℝ × ℝ)) ∧ (P.1 > a) ∧ 
-- Conditions inferred from the hyperbola definition
(|F1.1 - F2.1| = 2 * abs (P.1)) ∧ (Real.sqrt (F1.1^2 + F1.2^2 - P.1^2 + P.2^2)
                                            * Real.sqrt (F2.1^2 + F2.2^2 - P.1^2 + P.2^2) = 2 * a^2) ∧ 
-- Area condition
(0.5 * Real.abs (F1.1 * F2.2 - F2.1 * F1.2) = a^2) →

-- The eccentricity condition we want to prove
(e = Real.sqrt 2)

theorem hyperbola_eccentricity : 
∀ (a b : ℝ), a > 0 → b > 0 → hyperbola_description a b := sorry

end hyperbola_eccentricity_l649_649092


namespace num_positive_cos_terms_l649_649755

def sequence (n : ℕ) : ℝ := Real.cos (10^(n-1) * Real.pi / 180)

theorem num_positive_cos_terms : (Finset.card (Finset.filter (λ n, 0 < sequence n) (Finset.range 100))) = 99 := 
sorry

end num_positive_cos_terms_l649_649755


namespace smallest_s_triangle_l649_649626

theorem smallest_s_triangle (s : ℕ) :
  (7 + s > 11) ∧ (7 + 11 > s) ∧ (11 + s > 7) → s = 5 :=
sorry

end smallest_s_triangle_l649_649626


namespace positive_integers_18_1_solution_l649_649040

theorem positive_integers_18_1_solution :
    let a := 18 
    let b := 1 
    (¬ ∃ n : ℕ, n ≠ 0 ∧ n ≠ 1 ∧ n ≠ 6 ∧ n ≠ 18 ∧ n ≠ 1 ∧ n ≠ 19 ∧ n ≠ 18 ∧ n ≠ 399 ∧ n ≠ 399),
    ¬ (7 ∣ a * b * (a + b)) ∧ (7^7 ∣ (a + b)^7 - a^7 - b^7) := 
by
  sorry

end positive_integers_18_1_solution_l649_649040


namespace quadratic_function_symmetry_l649_649719

theorem quadratic_function_symmetry
  (p : ℝ → ℝ)
  (h_sym : ∀ x, p (5.5 - x) = p (5.5 + x))
  (h_0 : p 0 = -4) :
  p 11 = -4 :=
by sorry

end quadratic_function_symmetry_l649_649719


namespace lives_after_game_l649_649674

theorem lives_after_game (l0 : ℕ) (ll : ℕ) (lg : ℕ) (lf : ℕ) : 
  l0 = 10 → ll = 4 → lg = 26 → lf = l0 - ll + lg → lf = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end lives_after_game_l649_649674


namespace find_theta_l649_649871

variable θ : ℝ
variable h : θ ∈ set.Ioo (0 : ℝ) (Real.pi / 2)

theorem find_theta (h_parallel : (3 / 2) * (1 / 3) = (Real.sin θ) * (Real.cos θ)) : θ = Real.pi / 4 :=
sorry

end find_theta_l649_649871


namespace min_value_fraction_l649_649559

theorem min_value_fraction (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  (∃ T : ℝ, T = (5 * r / (3 * p + 2 * q) + 5 * p / (2 * q + 3 * r) + 2 * q / (p + r)) ∧ T = 19 / 4) :=
sorry

end min_value_fraction_l649_649559


namespace volume_of_region_l649_649550

noncomputable def region (x y : ℝ) : Prop :=
  (x^2 + (y - 5)^2 ≤ 25) ∧ (y ≥ x - 3)

noncomputable def volume (s : (ℝ × ℝ) → Prop) (line : ℝ × ℝ → ℝ) : ℝ :=
  π * ∫ (x : ℝ) in 3.5 .. 4.5, (y (x) - x - 3)^2

theorem volume_of_region :
  volume region (λ p, p.2 - p.1 - 3) = π / 3 :=
sorry

end volume_of_region_l649_649550


namespace value_of_p_minus_q_plus_r_l649_649134

theorem value_of_p_minus_q_plus_r
  (p q r : ℚ)
  (h1 : 3 / p = 6)
  (h2 : 3 / q = 18)
  (h3 : 5 / r = 15) :
  p - q + r = 2 / 3 :=
by
  sorry

end value_of_p_minus_q_plus_r_l649_649134


namespace max_points_with_two_distinct_manhattan_distances_l649_649318

/-- 
Given a set of points in the plane such that the Manhattan distance between any two distinct points is either 'long' or 'short', 
the maximal number of such points is 3.
-/
theorem max_points_with_two_distinct_manhattan_distances (S : Finset (ℝ × ℝ)) :
  (∀ x y ∈ S, x ≠ y → (|x.1 - y.1| + |x.2 - y.2| = L ∨ |x.1 - y.1| + |x.2 - y.2| = S)) → S.card ≤ 3 :=
by
  intros H
  sorry

end max_points_with_two_distinct_manhattan_distances_l649_649318


namespace find_starting_number_l649_649349

-- Helper function to compute the sum of digits of a natural number
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem
theorem find_starting_number (n : ℕ) :
  (∃ k : ℕ, k = 11 ∧ (iterate (λ x, x - digit_sum x) k n) = 0) →
  (n = 100 ∨ n = 101 ∨ n = 102 ∨ n = 103 ∨ n = 104 ∨ n = 105 ∨
   n = 106 ∨ n = 107 ∨ n = 108 ∨ n = 109) :=
by
  sorry -- Proof to be completed

end find_starting_number_l649_649349


namespace series_sum_l649_649947

noncomputable def compute_series (x : ℝ) (hx : x > 1) : ℝ :=
  ∑' n, 1 / (x ^ (3 ^ n) - x ^ (- 3 ^ n))

theorem series_sum (x : ℝ) (hx : x > 1) : compute_series x hx = 1 / (x - 1) :=
sorry

end series_sum_l649_649947


namespace sum_of_x_and_y_l649_649617

theorem sum_of_x_and_y :
  ∃ (x y : ℕ), (x > 0) ∧ (y > 0) ∧ Nat.is_square (450 * x) ∧ Nat.is_cube (450 * y) ∧ (x + y = 62) := by
  sorry

end sum_of_x_and_y_l649_649617


namespace seller_loss_l649_649680

theorem seller_loss :
  (goods : ℝ) (change : ℝ) (counterfeit_exchange : ℝ) (neighbor_refund : ℝ) 
  (buyer_goods : goods = 10) 
  (buyer_change : change = 15) 
  (neighbor_exchange : counterfeit_exchange = 25) 
  (seller_refund : neighbor_refund = 25)
  : goods + change = 25 :=
by
  sorry

end seller_loss_l649_649680


namespace area_triangle_ADG_eq_l649_649355

section
variable (A B C D E F G H : Type) 
-- Assume A, B, C, D, E, F, G, H represent points in a plane
variable [metric_space A B C D E F G H]
variable (a b c d e f g h : A)

-- Given conditions
variable (AB BC CD DE EF FG GH HA : ℝ)
variable (isRegularOctagon : isRegularOctagon = (AB = 2) ∧ (BC = 2) ∧ (CD = 2) ∧ (DE = 2) ∧ (EF = 2) ∧ (FG = 2) ∧ (GH = 2) ∧ (HA = 2))

-- Lengths AO and OG in the solution steps
noncomputable def lenAO : ℝ := real.sqrt 2
noncomputable def lenAD : ℝ := 2 + 2 * real.sqrt 2
noncomputable def lenOG : ℝ := 2 + real.sqrt 2

-- Defining the area to be computed
def area_ΔADG : ℝ := (1 / 2) * lenAD * lenOG

-- The statement to be proved
theorem area_triangle_ADG_eq : isRegularOctagon → area_ΔADG = 4 + 3 * real.sqrt 2 := 
by
  sorry
end

end area_triangle_ADG_eq_l649_649355


namespace total_carrots_l649_649220

-- Define the number of carrots grown by Sally and Fred
def sally_carrots := 6
def fred_carrots := 4

-- Theorem: The total number of carrots grown by Sally and Fred
theorem total_carrots : sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l649_649220


namespace nap_time_left_l649_649009

def train_ride_duration : ℕ := 9
def reading_time : ℕ := 2
def eating_time : ℕ := 1
def watching_movie_time : ℕ := 3

theorem nap_time_left :
  train_ride_duration - (reading_time + eating_time + watching_movie_time) = 3 :=
by
  -- Insert proof here
  sorry

end nap_time_left_l649_649009


namespace combine_like_terms_1_simplify_expression_2_l649_649727

-- Problem 1
theorem combine_like_terms_1 (m n : ℝ) :
  2 * m^2 * n - 3 * m * n + 8 - 3 * m^2 * n + 5 * m * n - 3 = -m^2 * n + 2 * m * n + 5 :=
by 
  -- Proof goes here 
  sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by 
  -- Proof goes here 
  sorry

end combine_like_terms_1_simplify_expression_2_l649_649727


namespace num_values_satisfy_l649_649610

noncomputable def f : ℝ → ℝ
| -3 := 3
| 1 := 3
| 5 := 3
| -1 := 1
| 3 := 5
-- Assume the rest of f is defined elsewhere

theorem num_values_satisfy : (∃ x₁ x₂, x₁ ≠ x₂ ∧ f(f(x₁)) = 3 ∧ f(f(x₂)) = 3) := 
begin
  use [-1, 3],
  split,
  { intro h,
    linarith, },
  split,
  { -- Show f(f(-1)) = 3
    dsimp [f],
    -- Assuming f(f(x)) step through how this equals 3
    sorry, },
  { -- Show f(f(3)) = 3
    dsimp [f],
    -- Assuming f(f(x)) step through how this equals 3
    sorry, }
end

end num_values_satisfy_l649_649610


namespace positive_terms_count_l649_649779

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then real.cos (1 * real.pi / 180) else real.cos (nat.pow 10 (n - 1) * real.pi / 180)

theorem positive_terms_count : finset.card (finset.filter (λ n, 0 < sequence n) (finset.range 100)) = 99 :=
by sorry

end positive_terms_count_l649_649779


namespace square_side_length_in_right_triangle_l649_649578

-- Proof problem statement
theorem square_side_length_in_right_triangle 
  (a b : ℝ) (ha : a = 10) (hb : b = 24) 
  (c : ℝ) (hc : c = Real.sqrt (a^2 + b^2)) 
  (s x : ℝ) 
  (h1 : s / c = x / a) 
  (h2 : s / c = (b - x) / b) : 
  s = 312 / 17 := 
by 
  sorry

end square_side_length_in_right_triangle_l649_649578


namespace diverse_dates_2013_l649_649644

theorem diverse_dates_2013 : ∃(n : ℕ), n = 2 ∧ forall (d1 d2 m1 m2 : ℕ), 
  (d1, d2, m1, m2) ∈ {(2, 0, 0, 5), (2, 5, 0, 4)} ∨
  (d1, d2, m1, m2) ∈ {(2, 5, 1, 0)} → n = 2 
:= sorry

end diverse_dates_2013_l649_649644


namespace banana_rearrangements_l649_649486

theorem banana_rearrangements : 
  let vowels := ['A', 'A', 'A']
  let consonants := ['B', 'N', 'N']
  (∃ v : List Char, v = vowels) ∧ (∃ c : List Char, c = consonants) 
  ∧ (∀ l, l = (list.permutations ('B' :: 'N' :: 'N' :: vowels)).length) 
  → l = 3 := sorry

end banana_rearrangements_l649_649486


namespace identity_implies_a_minus_b_l649_649884

theorem identity_implies_a_minus_b (a b : ℚ) :
  (∀ x : ℚ, 0 < x → (a / (2^x - 1) + b / (2^x + 2) = (4 * 2^x + 5) / ((2^x - 1) * (2^x + 2)))) →
  (a - b = 2) :=
by
  intros h
  sorry

end identity_implies_a_minus_b_l649_649884


namespace broken_line_lengths_equal_l649_649163

theorem broken_line_lengths_equal
  (X O Y M N : Point)
  (h1 : InsideAngle X O Y)
  (h2 : AngleEq (Angle X O N) (Angle Y O M))
  (Q : Point)
  (h3 : IsOnRay Q O X)
  (h4 : AngleEq (Angle N Q O) (Angle M Q X))
  (P : Point)
  (h5 : IsOnRay P O Y)
  (h6 : AngleEq (Angle N P O) (Angle M P Y)) :
  length M P + length P N = length M Q + length Q N :=
sorry

end broken_line_lengths_equal_l649_649163


namespace volume_of_resulting_shape_l649_649233

-- Define the edge lengths
def edge_length (original : ℕ) (small : ℕ) := original = 5 ∧ small = 1

-- Define the volume of a cube
def volume (a : ℕ) : ℕ := a * a * a

-- State the proof problem
theorem volume_of_resulting_shape : ∀ (original small : ℕ), edge_length original small → 
  volume original - (5 * volume small) = 120 := by
  sorry

end volume_of_resulting_shape_l649_649233


namespace carlson_max_candies_l649_649283

theorem carlson_max_candies : 
  (∀ (erase_two_and_sum : ℕ → ℕ → ℕ) 
    (eat_candies : ℕ → ℕ → ℕ), 
  ∃ (maximum_candies : ℕ), 
  (erase_two_and_sum 1 1 = 2) ∧
  (eat_candies 1 1 = 1) ∧ 
  (maximum_candies = 496)) :=
by
  sorry

end carlson_max_candies_l649_649283


namespace sum_of_max_min_a_l649_649057

theorem sum_of_max_min_a (a : ℝ) (x : ℝ) :
  (x^2 - a * x - 20 * a^2 < 0) →
  (∀ x1 x2 : ℝ, x1^2 - a * x1 - 20 * a^2 = 0 ∧ x2^2 - a * x2 - 20 * a^2 = 0 → |x1 - x2| ≤ 9) →
  (∀ max_min_sum : ℝ, max_min_sum = 1 + (-1) → max_min_sum = 0) := 
sorry

end sum_of_max_min_a_l649_649057


namespace radius_of_fourth_circle_l649_649913

theorem radius_of_fourth_circle 
  (n : ℕ) (r0 r6 : ℝ) (r : ℝ)
  (h1 : n = 7)
  (h2 : r6 = 36)
  (h3 : r0 = 9)
  (h4 : r = real.root (36 / 9) 6) : 
  r * r0 * (r ^ 3) = 72 :=
by
  sorry

end radius_of_fourth_circle_l649_649913


namespace drawing_red_ball_random_drawing_yellow_ball_impossible_probability_black_ball_number_of_additional_black_balls_l649_649903

-- Definitions for the initial conditions
def initial_white_balls := 2
def initial_black_balls := 3
def initial_red_balls := 5
def total_balls := initial_white_balls + initial_black_balls + initial_red_balls

-- Statement for part 1: Drawing a red ball is a random event
theorem drawing_red_ball_random : (initial_red_balls > 0) := by
  sorry

-- Statement for part 1: Drawing a yellow ball is impossible
theorem drawing_yellow_ball_impossible : (0 = 0) := by
  sorry

-- Statement for part 2: Probability of drawing a black ball
theorem probability_black_ball : (initial_black_balls : ℚ) / total_balls = 3 / 10 := by
  sorry

-- Definitions for the conditions in part 3
def additional_black_balls (x : ℕ) := initial_black_balls + x
def new_total_balls (x : ℕ) := total_balls + x

-- Statement for part 3: Finding the number of additional black balls
theorem number_of_additional_black_balls (x : ℕ)
  (h : (additional_black_balls x : ℚ) / new_total_balls x = 3 / 4) : x = 18 := by
  sorry

end drawing_red_ball_random_drawing_yellow_ball_impossible_probability_black_ball_number_of_additional_black_balls_l649_649903


namespace harmonic_inequality_l649_649994

theorem harmonic_inequality (n : ℕ) (h : n ≥ 2) : (∑ i in finset.Icc (n+1) (2*n), (1 : ℚ) / i) ≥ 13 / 24 :=
  sorry

end harmonic_inequality_l649_649994


namespace students_more_than_turtles_l649_649397

theorem students_more_than_turtles
  (students_per_classroom : ℕ)
  (turtles_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (h1 : students_per_classroom = 20)
  (h2 : turtles_per_classroom = 3)
  (h3 : number_of_classrooms = 5) :
  (students_per_classroom * number_of_classrooms)
  - (turtles_per_classroom * number_of_classrooms) = 85 :=
by
  sorry

end students_more_than_turtles_l649_649397


namespace fraction_of_area_above_line_l649_649613

theorem fraction_of_area_above_line : 
  let square_vertices : Set (ℝ × ℝ) := {(4, 1), (8, 1), (8, 5), (4, 5)},
      line_points : Set (ℝ × ℝ) := {(4, 3), (8, 1)} in
  (∃ line : ℝ → ℝ, ∀ (x y : ℝ), (x, y) ∈ line_points → y = line x) →
  (∀ A B ∈ square_vertices, (A.1, A.2) = (min A.1 B.1, min A.2 B.2) → (B.1, B.2) = (max A.1 B.1, max A.2 B.2)) →
  let square_area := (8 - 4) * (5 - 1),
      area_above_line := square_area -- because the entire square is above the line
  in area_above_line / square_area = 1 :=
by
  sorry

end fraction_of_area_above_line_l649_649613


namespace determine_BD_correct_l649_649147

noncomputable def determine_BD (A B C D : Point) : Real :=
  if AC = 7 ∧ BC = 7 ∧ AB = 4 ∧ B ∈ segment A D ∧ CD = 9 then 4 else sorry

theorem determine_BD_correct : 
    ∀ (A B C D : Point), AC = 7 ∧ BC = 7 ∧ AB = 4 ∧ B ∈ segment A D ∧ CD = 9 → determine_BD A B C D = 4 :=
by
  intros A B C D h
  rw determine_BD
  simp [h]
  sorry

end determine_BD_correct_l649_649147


namespace right_angled_triangles_in_18_sided_polygon_l649_649127

theorem right_angled_triangles_in_18_sided_polygon : 
  let n := 18 in 
  ∑ i in (finset.range (n / 2)), 16 = 144 := 
by
  have n_pos : 0 < 18 := by norm_num
  have h : finset.card (finset.range (18 / 2)) = 9 := by
    rw [nat.div_eq_of_eq_mul_right (by norm_num : (2 : ℕ) ≠ 0) rfl]
    norm_num
  rw [finset.sum_const, h]
  norm_num

end right_angled_triangles_in_18_sided_polygon_l649_649127


namespace smallest_positive_value_a_l649_649429

theorem smallest_positive_value_a :
  ∃ (a : ℝ), (a = 0.79) ∧ ∀ (x : ℝ), (x > (3 * π / 2)) ∧ (x < 2 * π) →
    (sin x ^ 2 ≥ 0) ∧ (cos x ^ 2 ≥ 0) →
    (sqrt (sin x ^ 2)^(1 / 3) - sqrt (cos x ^ 2)^(1 / 3)) / (sqrt (tan x ^ 2)^(1 / 3) - sqrt (cot x ^ 2)^(1 / 3)) < a / 2 :=
begin
  sorry
end

end smallest_positive_value_a_l649_649429


namespace num_positive_terms_l649_649773

noncomputable def seq (n : ℕ) : ℝ := float.cos (10^((n - 1).to_real))

theorem num_positive_terms : fin 100 → seq 100 99 :=
sorry

end num_positive_terms_l649_649773


namespace minji_combinations_l649_649213

theorem minji_combinations : (3 * 5) = 15 :=
by sorry

end minji_combinations_l649_649213


namespace seated_knights_probability_sum_l649_649288

theorem seated_knights_probability_sum:
  let total_knights := 30
  let chosen_knights := 4
  let adjacent_ways := choose 26 3
  let total_ways := choose 30 4
  let P := 1 - (adjacent_ways / total_ways).toRat
  let simplified_P := (24805 : ℚ) / 27405 -- simplified to lowest terms as 4961/5481
  let sum_of_num_den := (4961 + 5481)
  in sum_of_num_den = 10442 := sorry

end seated_knights_probability_sum_l649_649288


namespace frac_b_by_a_is_real_l649_649196

open Complex

theorem frac_b_by_a_is_real {a b x y : ℂ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : |x| = |y|) (h4 : x + y = a) (h5 : x * y = b) : isReal (b / a) :=
  sorry

end frac_b_by_a_is_real_l649_649196


namespace infinite_positive_integer_solutions_l649_649583

theorem infinite_positive_integer_solutions : ∃ (a b c : ℕ), (∃ k : ℕ, k > 0 ∧ a = k * (k^3 + 1990) ∧ b = (k^3 + 1990) ∧ c = (k^3 + 1990)) ∧ (a^3 + 1990 * b^3) = c^4 :=
sorry

end infinite_positive_integer_solutions_l649_649583


namespace bacon_calories_l649_649376

theorem bacon_calories (T : ℕ) (P : ℝ) (n : ℕ) (hT : T = 1250) (hP : P = 0.20) (hn : n = 2) : (T * P) / n = 125 := by
  -- Given conditions
  rw [hT, hP, hn]
  -- Required proof
  norm_num
  sorry

end bacon_calories_l649_649376


namespace a_33_is_3_l649_649527

noncomputable def sequence (n : ℕ) : ℤ :=
  if h : n = 1 then 3
  else if h : n = 2 then 6
  else (sequence (n-1)) - (sequence (n-2))

theorem a_33_is_3 : sequence 33 = 3 :=
by
  sorry

end a_33_is_3_l649_649527


namespace distance_from_midpoint_chord_to_x_axis_l649_649461

noncomputable def parabola_focus (y : ℝ) : ℝ × ℝ := (0, 1 / 4)

theorem distance_from_midpoint_chord_to_x_axis :
  ∃ A B : ℝ × ℝ,
    let F := parabola_focus 1 in
    (A ≠ B) ∧
    A.1^2 = A.2 ∧
    B.1^2 = B.2 ∧
    (A - B).norm = 4 ∧
    let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
    abs M.2 = 7 / 4 :=
by
  sorry

end distance_from_midpoint_chord_to_x_axis_l649_649461


namespace triangle_exists_with_sides_divisible_l649_649557

theorem triangle_exists_with_sides_divisible
  (p : ℕ) (n : ℕ)
  (hp_prime : p.prime)
  (hp_odd : odd p)
  (hn_pos : 0 < n)
  (points : fin 8 → ℤ × ℤ)
  (circle_diameter : ∃ (x y : ℤ) (r : ℚ), r = (p ^ n) / 2 ∧ (∀ i : fin 8, (points i).fst ^ 2 + (points i).snd ^ 2 = r ^ 2)): 
  ∃ (i j k : fin 8), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
  p^(n+1) ∣ ((points i).fst - (points j).fst)^2 + ((points i).snd - (points j).snd)^2 ∧ 
  p^(n+1) ∣ ((points j).fst - (points k).fst)^2 + ((points j).snd - (points k).snd)^2 ∧ 
  p^(n+1) ∣ ((points k).fst - (points i).fst)^2 + ((points k).snd - (points i).snd)^2 :=
sorry

end triangle_exists_with_sides_divisible_l649_649557


namespace range_of_k_l649_649892

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

def is_defined (k : ℝ) : Prop := ∀ x : ℝ, k-1 < x ∧ x < k+1 → x > 0

def is_not_monotonic (k : ℝ) : Prop := 
  ∃ x1 x2 : ℝ, k-1 < x1 ∧ x1 < x2 ∧ x2 < k+1 ∧ (x1, x2 ∈ (0, ∞)) ∧
  (f x1 < f x2 ∨ f x1 > f x2)

theorem range_of_k (k : ℝ) (h1 : is_defined k) (h2 : is_not_monotonic k) : 
  1 ≤ k ∧ k < 3 / 2 := 
sorry

end range_of_k_l649_649892


namespace max_handshakes_l649_649902

theorem max_handshakes (N : ℕ) (h : N > 4) (not_all_handshakes : ∃ p1 p2, (p1 ≠ p2) ∧ (¬(∀ i ≠ p1, i = p2) ∨ ¬(∀ i ≠ p2, i = p1))) :
  ∃ k, k = N - 2 :=
sorry

end max_handshakes_l649_649902


namespace stewart_farm_l649_649371

variable (sheep horses : Nat) (sheep_to_horses : ℚ) (total_horse_food_per_day : ℚ) (food_per_horse : ℚ)

-- The conditions given in the problem
def conditions : Prop :=
  sheep_to_horses = 5 / 7 ∧
  sheep = 40 ∧
  total_horse_food_per_day = 12880

-- The main proposition we need to prove
def proof_problem : Prop :=
  food_per_horse = 230

-- The statement combining the conditions and the problem
theorem stewart_farm :
  conditions sheep horses sheep_to_horses total_horse_food_per_day food_per_horse →
  proof_problem total_horse_food_per_day food_per_horse :=
by
  sorry

end stewart_farm_l649_649371


namespace q_investment_time_l649_649619

-- Define the investment ratio
def investment_ratio (x : ℝ) : ℝ × ℝ := (7 * x, 5.00001 * x)

-- Define the profit ratio
def profit_ratio : ℝ × ℝ := (7.00001, 10)

-- Define the investment time for p (in months)
def investment_time_p : ℝ := 5

-- Define the equation based on profit ratio
def profit_equation (t : ℝ) (x : ℝ) : Prop :=
  (7 * x * 5) / (5.00001 * x * t) = 7.00001 / 10

-- Define the proof goal
theorem q_investment_time (x : ℝ) : ∃ t : ℝ, profit_equation t x ∧ t ≈ 10 :=
by
  sorry

end q_investment_time_l649_649619


namespace equations_solutions_l649_649746

-- Define the statements for each equation
def eq1 (a b : ℂ) : Prop := (a = 0 ∧ b = 0) → (sqrt (a^2 + b^2) = 0)
def eq2 (a b : ℂ) : Prop := (a ≠ 0 ∨ b ≠ 0) → sqrt (a^2 + b^2) = a * b
def eq3 (a b : ℂ) : Prop := (a ≠ 0 ∨ b ≠ 0) → sqrt (a^2 + b^2) = a + b
def eq4 (a b : ℂ) : Prop := (a ≠ 0 ∨ b ≠ 0) → sqrt (a^2 + b^2) = a - b

-- Prove the conclusion based on the given conditions and equations
theorem equations_solutions :
  (∀ a b : ℂ, eq1 a b) ∧
  (∃ a b : ℂ, eq2 a b) ∧
  (∃ a b : ℂ, eq3 a b) ∧
  (∃ a b : ℂ, eq4 a b) :=
by
  sorry

end equations_solutions_l649_649746


namespace sum_series_eq_l649_649954

open Real

theorem sum_series_eq (x : ℝ) (h : 1 < x) : 
  (∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (- (3 ^ n)))) = 1 / (x - 1) :=
sorry

end sum_series_eq_l649_649954


namespace addition_correct_l649_649652

theorem addition_correct :
  1357 + 2468 + 3579 + 4680 + 5791 = 17875 := 
by
  sorry

end addition_correct_l649_649652


namespace number_of_such_z_elements_l649_649239

-- Defining a field F with 5^14 elements
def F : Type := { x // x ∈ Finset.univ_size (5^14) }

-- Definition of a "happy" function
def happy (f : F → F) : Prop :=
  ∀ x y : F, (f(x + y) + f(x)) * (f(x - y) + f(x)) = f(y^2) - f(x^2)

-- The number of elements z in F such that there exist distinct happy functions h1 and h2 with h1(z) = h2(z)
theorem number_of_such_z_elements : 
  ∃ z ∈ F, ∃ h1 h2 : F → F, happy h1 ∧ happy h2 ∧ h1 ≠ h2 ∧ h1 z = h2 z ∧ F.card = 5^14 := 
by
    -- placeholder for proof, the statement suffices
    sorry

end number_of_such_z_elements_l649_649239


namespace inequality_proof_l649_649875

theorem inequality_proof (a b x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  a * x^2 + b * y^2 ≥ (a * x + b * y)^2 := 
by {
  sorry,
}

end inequality_proof_l649_649875


namespace eigenvalues_and_eigenvectors_of_M_l649_649047

noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := ![![0, 0], ![0, 1]]

theorem eigenvalues_and_eigenvectors_of_M :
  ((λ x, Matrix.det (M - x • (1 : Matrix (Fin 2) (Fin 2) ℝ))) = λ x, x * (x - 1))
  ∧ (∃ (v : Fin 2 → ℝ), (M ⬝ v = (0 : ℝ) • v) ∧ v ≠ 0) ∧ ∀ (v : Fin 2 → ℝ), (M ⬝ v = (0 : ℝ) • v) → v = ![1, 0]
  ∧ (∃ (v : Fin 2 → ℝ), (M ⬝ v = (1 : ℝ) • v) ∧ v ≠ 0) ∧ ∀ (v : Fin 2 → ℝ), (M ⬝ v = (1 : ℝ) • v) → v = ![0, 1] := sorry

end eigenvalues_and_eigenvectors_of_M_l649_649047


namespace MN_perp_IO_l649_649437

theorem MN_perp_IO
  (A B C O I D E F Q P M N : Point)
  (hCircum : is_circumcenter A B C O)
  (hIncenter : is_incenter A B C I)
  (hDE : incircle_touches_side_at A I B C = D)
  (hEF : incircle_touches_side_at B I C A = E)
  (hFD : incircle_touches_side_at C I A B = F)
  (hQ : line_of_points E D ∩ line_of_points A B = Q)
  (hP : line_of_points F D ∩ line_of_points A C = P)
  (hM : is_midpoint P E M)
  (hN : is_midpoint Q F N) 
  : perpendicular_line MN IO :=
sorry

end MN_perp_IO_l649_649437


namespace zero_people_with_fewer_than_six_cards_l649_649494

theorem zero_people_with_fewer_than_six_cards (cards people : ℕ) (h_cards : cards = 60) (h_people : people = 9) :
  let avg := cards / people
  let remainder := cards % people
  remainder < people → ∃ n, n = 0 := by
  sorry

end zero_people_with_fewer_than_six_cards_l649_649494


namespace girls_from_valley_l649_649546

-- Define the given conditions 
variables (total_students total_boys total_girls students_highland students_valley boys_highland : ℕ)
variables (girls_valley : ℕ)

-- Assign given values to variables
def conditions := total_students = 120 ∧ 
                  total_boys = 70 ∧ 
                  total_girls = 50 ∧ 
                  students_highland = 45 ∧ 
                  students_valley = 75 ∧ 
                  boys_highland = 30

-- Define the proof statement
theorem girls_from_valley (h : conditions) : girls_valley = 35 :=
sorry

end girls_from_valley_l649_649546


namespace cos_450_eq_zero_l649_649737

theorem cos_450_eq_zero :
  ∀ (θ : ℝ), θ = 450 ∧ (θ mod 360 = 90) ∧ (cos 90 = 0) → cos θ = 0 :=
by
  intros θ hθ
  cases hθ with h1 hand
  cases hand with h2 h3
  rw h1
  rw h2
  exact h3

end cos_450_eq_zero_l649_649737


namespace roots_of_polynomial_l649_649264

theorem roots_of_polynomial : ∃ x1 x2 : ℝ, (x1 * (x1 - 1) = 0) ∧ (x2 * (x2 - 1) = 0) ∧ x1 ≠ x2 ∧ (x1 = 0 ∧ x2 = 1) := 
by {
  -- Solution steps as conditions
  have h1 : 0 * (0 - 1) = 0 := by linarith,
  have h2 : 1 * (1 - 1) = 0 := by linarith,
  have neq : 0 ≠ 1 := by linarith,
  existsi 0,
  existsi 1,
  exact ⟨h1, h2, neq, ⟨rfl, rfl⟩⟩
}

end roots_of_polynomial_l649_649264


namespace max_sum_of_digits_0_to_7_is_13951_l649_649295

theorem max_sum_of_digits_0_to_7_is_13951 :
  ∃ (a b : ℕ), (∀ (d : ℕ), d ∈ {0, 1, 2, 3, 4, 5, 6, 7} → d ∈ digits a ∨ d ∈ digits b) ∧
                (∀ (d : ℕ), d ∈ digits a ∨ d ∈ digits b → d ∈ {0, 1, 2, 3, 4, 5, 6, 7}) ∧
                (a < 10000) ∧ (b < 10000) ∧ (∀ (x y : ℕ), (x < 10000) → (y < 10000) → 
                                                (∀ (d' : ℕ), d' ∈ {0, 1, 2, 3, 4, 5, 6, 7} → d' ∈ digits x ∨ d' ∈ digits y) → 
                                                (a + b ≥ x + y)) ∧ 
                a + b = 13951 := 
by
  sorry

end max_sum_of_digits_0_to_7_is_13951_l649_649295


namespace negative_expression_b_negative_expression_c_negative_expression_e_l649_649657

theorem negative_expression_b:
  3 * Real.sqrt 11 - 10 < 0 := 
sorry

theorem negative_expression_c:
  18 - 5 * Real.sqrt 13 < 0 := 
sorry

theorem negative_expression_e:
  10 * Real.sqrt 26 - 51 < 0 := 
sorry

end negative_expression_b_negative_expression_c_negative_expression_e_l649_649657


namespace det_E_l649_649184

namespace MatrixProblem

variables {α : Type*} [Field α]

-- Define matrix R for 90-degree rotation
def R : Matrix (Fin 2) (Fin 2) α := ![
  ![0, -1],
  ![1, 0]
]

-- Define matrix S for dilation with scale factor 5
def S : Matrix (Fin 2) (Fin 2) α := ![
  ![5, 0],
  ![0, 5]
]

-- Matrix E is the product of S and R
def E : Matrix (Fin 2) (Fin 2) α := S ⬝ R

-- Statement to prove
theorem det_E : det E = (25 : α) := by
  sorry

end MatrixProblem

end det_E_l649_649184


namespace number_of_ways_to_assign_positions_l649_649084

-- Definitions based on conditions
def members : Finset String := {"Alice", "Bob", "Carol", "Dave"}

def positions : Finset String := {"President", "Vice President", "Secretary", "Treasurer"}

-- Theorem statement
theorem number_of_ways_to_assign_positions : Finset.card (members.permutations) = 24 :=
by
  have h : members = {"Alice", "Bob", "Carol", "Dave"} := rfl
  have p : Finset.card members = 4 := by simp [h]
  -- Given 4 distinct positions and 4 members
  have pos : positions = {"President", "Vice President", "Secretary", "Treasurer"} := rfl
  have pos_card : Finset.card positions = 4 := by simp [pos]
  -- Calculating the number of ways to assign positions using permutations
  exact Finset.card_permutations_eq_card_factorial members

end number_of_ways_to_assign_positions_l649_649084


namespace remainder_when_eight_n_plus_five_divided_by_eleven_l649_649138

theorem remainder_when_eight_n_plus_five_divided_by_eleven
  (n : ℤ) (h : n % 11 = 4) : (8 * n + 5) % 11 = 4 := 
  sorry

end remainder_when_eight_n_plus_five_divided_by_eleven_l649_649138


namespace number_of_valid_seating_l649_649794

-- Define the function T such that T n d represents the number of valid seatings for n women,
-- with exactly d of them moving to an adjacent seat.
def T : ℕ → ℕ → ℕ
| 0, _ => 1
| 1, 0 => 1
| 1, _ => 0
| n, d =>
  if d = 0 then 2^(n-1) else
  if d = 2 then 2^(n-2) else
  T (n-1) d + T (n-2) (d-2)

-- Problem statement: Prove that T 8 2 = 34
theorem number_of_valid_seating : T 8 2 = 34 := by
  sorry

end number_of_valid_seating_l649_649794


namespace trapezoid_area_l649_649370

/-- E is the midpoint of the leg AB of trapezoid ABCD. DF ⊥ EC, DF = 10, and EC = 24. Prove the area of trapezoid ABCD is 240. -/
theorem trapezoid_area (ABCD : Trapezoid) (E : Point)
  (h_midpoint : isMidpoint E ABCD.A B) 
  (h_perpendicular : Perpendicular DF EC) 
  (hDF : segmentLength DF = 10) 
  (hEC : segmentLength EC = 24) : 
  area ABCD = 240 := 
sorry

end trapezoid_area_l649_649370


namespace transform_kiosk_yields_isk_transformation_proof_l649_649162

-- Define the word transformations as per the conditions.
def transform_burjan_to_burya (word : String) : String :=
  -- Remove 4th and 6th characters from "БУРЬЯН" to get "БУРЯ".
  String.mk ⟨word.iterateAux (λ i c, if i == 3 ∨ i == 5 then none else some c) (⟨[], 0⟩ : List Char × Nat)⟩.fst

def transform_valenok_to_venok (word : String) : String :=
  -- Remove 2nd and 3rd characters from "ВАЛЕНОК" to get "ВЕНОК".
  String.mk ⟨word.iterateAux (λ i c, if i == 1 ∨ i == 2 then none else some c) (⟨[], 0⟩ : List Char × Nat)⟩.fst
  
def transform_kiosk_to_isk (word : String) : String :=
  -- Remove 1st and 3rd characters from "КИОСК" to get "ИСК".
  String.mk ⟨word.iterateAux (λ i c, if i == 0 ∨ i == 2 then none else some c) (⟨[], 0⟩ : List Char × Nat)⟩.fst

-- Prove that the transformation of "КИОСК" yields "ИСК" given the discovered pattern.
theorem transform_kiosk_yields_isk : transform_kiosk_to_isk "КИОСК" = "ИСК" :=
by sorry

-- Main proof of the problem combining the transformations of each pair according to the conditions.
theorem transformation_proof :
  transform_burjan_to_burya "БУРЬЯН" = "БУРЯ" ∧
  transform_valenok_to_venok "ВАЛЕНОК" = "ВЕНОК" ∧
  transform_kiosk_to_isk "КИОСК" = "ИСК" :=
by
  split
  · -- Proof for first transformation
    sorry
  split
  · -- Proof for second transformation
    sorry
  · -- Proof for third transformation
    exact transform_kiosk_yields_isk

end transform_kiosk_yields_isk_transformation_proof_l649_649162


namespace triangle_is_isosceles_range_of_expression_l649_649146

variable {a b c A B C : ℝ}
variable (triangle_ABC : 0 < A ∧ A < π ∧ 0 < B ∧ B < π)
variable (opposite_sides : a = 1 ∧ b = 1 ∧ c = 1)
variable (cos_condition : a * Real.cos B = b * Real.cos A)

theorem triangle_is_isosceles (h : a * Real.cos B = b * Real.cos A) : A = B := sorry

theorem range_of_expression 
  (h1 : 0 < A ∧ A < π/2) 
  (h2 : a * Real.cos B = b * Real.cos A) : 
  -3/2 < Real.sin (2 * A + π / 6) - 2 * Real.cos B ^ 2 ∧ Real.sin (2 * A + π / 6) - 2 * Real.cos B ^ 2 < 0 := 
sorry

end triangle_is_isosceles_range_of_expression_l649_649146


namespace max_sum_12th_powers_1997_l649_649620

theorem max_sum_12th_powers_1997 (x : Fin 1997 → ℝ)
  (h_sum : (∑ i, x i) = -318 * Real.sqrt 3)
  (h_range : ∀ i, -1 / Real.sqrt 3 ≤ x i ∧ x i ≤ Real.sqrt 3) :
  ∑ i, (x i) ^ 12 ≤ 189548 := sorry

end max_sum_12th_powers_1997_l649_649620


namespace sum_y_seq_l649_649553

noncomputable def y_seq (n m : ℕ) : ℕ → ℕ
| 0       := 0
| 1       := 1
| (k + 2) := (n + m - 1) * y_seq (k + 1) - (n - k) * y_seq k + m / (k + 1)

theorem sum_y_seq (n m : ℕ) (hn : n > m) :
  (∑ k in range (n + m), y_seq n m k) = 2^(n + m - 1) * (n + m - 1/2) :=
sorry

end sum_y_seq_l649_649553


namespace gcd_of_B_is_2_l649_649965

-- Definition of the set B based on the given condition.
def B : Set ℕ := {n | ∃ x, n = (x - 1) + x + (x + 1) + (x + 2)}

-- The core statement to prove, wrapped in a theorem.
theorem gcd_of_B_is_2 : gcd_set B = 2 :=
by
  sorry

end gcd_of_B_is_2_l649_649965


namespace domain_of_sqrt_2sinx_plus_1_l649_649606

theorem domain_of_sqrt_2sinx_plus_1 :
  ∀ x : ℝ, (∃ k : ℤ, - (π / 6) + 2 * k * π ≤ x ∧ x ≤ (7 * π / 6) + 2 * k * π) ↔ 2 * sin x + 1 ≥ 0 :=
by sorry

end domain_of_sqrt_2sinx_plus_1_l649_649606


namespace square_of_G_l649_649019

-- Conditions and Definitions
def G : ℂ := complex.exp (complex.I * (π / 6))

-- The proof problem statement
theorem square_of_G :
  (G^2 = complex.exp (complex.I * (π / 3))) :=
by
  -- Proof to be added
  sorry

end square_of_G_l649_649019


namespace quadrilateral_problem_l649_649520

noncomputable def BC := 10
noncomputable def CD := 15
noncomputable def AD := 13
noncomputable def angle_A := 70
noncomputable def angle_B := 70

theorem quadrilateral_problem 
  (BC: ℝ) (CD: ℝ) (AD: ℝ) (angle_A: ℝ) (angle_B: ℝ)
  (h_bc: BC = 10) (h_cd: CD = 15) (h_ad: AD = 13) (h_angle_a: angle_A = 70) (h_angle_b: angle_B = 70) :
  ∃ (p q : ℕ), AB = p + real.sqrt q ∧ p + q = 14 :=
by
  sorry

end quadrilateral_problem_l649_649520


namespace area_of_square_l649_649359

-- Define the conditions
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3 -- y = x^2 + 4x + 3
def line_y_const (y : ℝ) : Prop := y = 8 -- line y = 8

-- Define the square's side length based on the intersection points
noncomputable def intersection_points : set ℝ :=
  {x : ℝ | parabola x = 8}

noncomputable def side_length : ℝ :=
  let xs := intersection_points in
  if h : ∃ a b, a ∈ xs ∧ b ∈ xs ∧ a ≠ b
  then abs (classical.some h - classical.some (exists.elim h (λ _ h', h')))
  else 0

-- Prove that the area of the square is 36
theorem area_of_square : side_length ^ 2 = 36 :=
by 
  sorry

end area_of_square_l649_649359


namespace frog_jumps_initial_distance_l649_649290

open_locale classical

theorem frog_jumps (initial_positions : ℕ × ℕ × ℕ)
  (first_jump second_jump : ℕ)
  (H1 : (first_jump = 60 ∨ second_jump = 60) ∧ 
        (first_jump = 60 ∨ second_jump = 60) ∧ 
        (first_jump ≠ second_jump)) :
  (first_jump = 30 ∨ first_jump = 120) ∧
  (second_jump = 30 ∨ second_jump = 120) := 
sorry

theorem initial_distance (initial_positions : ℕ × ℕ × ℕ)
  (first_jump second_jump : ℕ)
  (H1 : (first_jump = 60 ∨ first_jump = 60)):
  ((first_jump ≠ second_jump) → (first_jump = 100 ∨ first_jump = 160)) ∧ 
  ((second_jump ≠ first_jump) → (second_jump = 100 ∨ second_jump = 160)) :=
sorry

end frog_jumps_initial_distance_l649_649290


namespace fraction_used_for_crepes_l649_649292

theorem fraction_used_for_crepes (total_eggs crepes_eggs remaining_eggs cupcakes_eggs : ℕ)
  (H1 : total_eggs = 3 * 12)
  (H2 : remaining_eggs = 9 * 3)
  (H3 : total_eggs - remaining_eggs = crepes_eggs)
  (H4 : cupcakes_eggs = (2 * remaining_eggs) / 3)
  (H5 : total_eggs - crepes_eggs - cupcakes_eggs = 9) :
  (crepes_eggs : ℚ) / total_eggs = 1 / 4 :=
by
  sorry

end fraction_used_for_crepes_l649_649292


namespace n_is_one_sixth_sum_of_list_l649_649334

-- Define the condition that n is 4 times the average of the other 20 numbers
def satisfies_condition (n : ℝ) (l : List ℝ) : Prop :=
  l.length = 21 ∧
  n ∈ l ∧
  n = 4 * (l.erase n).sum / 20

-- State the main theorem
theorem n_is_one_sixth_sum_of_list {n : ℝ} {l : List ℝ} (h : satisfies_condition n l) :
  n = (1 / 6) * l.sum :=
by
  sorry

end n_is_one_sixth_sum_of_list_l649_649334


namespace combinatorial_calculation_l649_649013

-- Define the proof problem.
theorem combinatorial_calculation : (Nat.choose 20 6) = 2583 := sorry

end combinatorial_calculation_l649_649013


namespace max_intersections_l649_649784

-- Given definitions and conditions
variables (Q1 Q2 : Type) [convex Q1] [convex Q2] (m : ℕ)
  (hQ1_sides : sides Q1 = m)
  (hQ2_sides : sides Q2 = 2 * m)
  (hQ1_in_Q2 : nested Q1 Q2)
  (h_m_geq_3 : m ≥ 3)

-- Theorem statement
theorem max_intersections (Q1 Q2 : Type)
  [convex Q1] [convex Q2]
  (m : ℕ)
  (hQ1_sides : sides Q1 = m)
  (hQ2_sides : sides Q2 = 2 * m)
  (hQ1_in_Q2 : nested Q1 Q2)
  (h_m_geq_3 : m ≥ 3)
  : max_intersections Q1 Q2 = 2 * m ^ 2 :=
sorry

end max_intersections_l649_649784


namespace positive_terms_count_l649_649780

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then real.cos (1 * real.pi / 180) else real.cos (nat.pow 10 (n - 1) * real.pi / 180)

theorem positive_terms_count : finset.card (finset.filter (λ n, 0 < sequence n) (finset.range 100)) = 99 :=
by sorry

end positive_terms_count_l649_649780


namespace gcd_of_B_is_2_l649_649963

-- Condition: B is the set of all numbers which can be represented as the sum of four consecutive positive integers
def B := { n : ℕ | ∃ y : ℕ, n = (y - 1) + y + (y + 1) + (y + 2) }

-- Question: What is the greatest common divisor of all numbers in \( B \)
-- Mathematical equivalent proof problem: Prove gcd of all elements in set \( B \) is 2

theorem gcd_of_B_is_2 : ∀ n ∈ B, ∃ y : ℕ, n = 2 * (2 * y + 1) → ∀ m ∈ B, n.gcd m = 2 :=
by
  sorry

end gcd_of_B_is_2_l649_649963


namespace pentagon_product_l649_649703

noncomputable def complex_coordinates (n : ℕ) : ℂ :=
if n = 1 then 2 + 0 * complex.I else
if n = 2 then 4 + 0 * complex.I else
if n = 3 then 4 + (4 - 4 * complex.I) * (cos ((2 * n - 2) * real.pi / 5) + sin ((2 * n - 2) * real.pi / 5)) else
if n = 4 then 4 + (4 - 4 * complex.I) * (cos ((2 * n -2) * real.pi / 5) + sin ((2 * n - 2) * real.pi / 5)) else
if n = 5 then 4 + (4 - 4 * complex.I) * (cos ((2 * n - 2) * real.pi / 5) + sin ((2 * n - 2) * real.pi / 5)) else sorry

theorem pentagon_product : 
  ∏ i in finset.range 1 6, (complex_coordinates i) = 1023 :=
by sorry

end pentagon_product_l649_649703


namespace part_a_part_b_l649_649979

variables {A B C D : ℝ}
variables {A_star B_star C_star D_star : ℝ}
variables {R : ℝ}

-- Given points A, B, C, D and their inverted points under a circle of radius R
-- centered at the origin, prove:

theorem part_a (hA : A_star = R / A) (hB : B_star = R / B) (hC : C_star = R / C) (hD : D_star = R / D):
  (A - C) / (A - D) / ((B - C) / (B - D)) = (A_star - C_star) / (A_star - D_star) / ((B_star - C_star) / (B_star - D_star)) :=
by sorry

theorem part_b (hA : A_star = R / A) (hB : B_star = R / B) (hC : C_star = R / C) (hD : D_star = R / D):
  ∠(D, A) - ∠(D, B) = ∠(D_star, B_star) - ∠(D_star, A_star) :=
by sorry

end part_a_part_b_l649_649979


namespace gcd_exponential_identity_l649_649225

theorem gcd_exponential_identity (a b : ℕ) :
  Nat.gcd (2^a - 1) (2^b - 1) = 2^(Nat.gcd a b) - 1 := sorry

end gcd_exponential_identity_l649_649225


namespace triangle_area_ratio_l649_649547

-- Definitions based on the conditions in the problem
def is_equilateral_triangle (A B C : Point) := (dist A B = dist B C) ∧ (dist B C = dist C A)
def extend_point (P Q : Point) (k : ℝ) : Point := 
  let PQ := Q - P in
  Q + k * PQ

-- The problem statement
theorem triangle_area_ratio (A B C : Point) (s : ℝ) (B' C' A' : Point)
  (h_eq_triangle : is_equilateral_triangle A B C)
  (h_BB' : dist B B' = 2 * dist A B)
  (h_CC' : dist C C' = 3 * dist B C)
  (h_AA' : dist A A' = 4 * dist C A) :
  ratio_of_triangle_areas (Triangle A' B' C') (Triangle A B C) = 8 * real.sqrt 3 :=
by 
  sorry -- We skip the proof steps as instructed

end triangle_area_ratio_l649_649547


namespace units_digit_diff_l649_649418

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_diff :
  let a := 8
  let b := 19
  let c := 1978
  let d := 8 ^ 3
  units_digit (a * b * c - d) = 4 :=
by
  let a := 8
  let b := 19
  let c := 1978
  let d := 8 ^ 3
  have h1 : units_digit a = 8 := rfl
  have h2 : units_digit b = 9 := rfl
  have h3 : units_digit (a * b) = 2 := rfl
  have h4 : units_digit c = 8 := rfl
  have h5 : units_digit (2 * 8) = 6 := rfl
  have h6 : units_digit d = 2 := rfl
  have h7 : units_digit (6 - 2) = 4 := rfl
  show units_digit (a * b * c - d) = 4 from h7
  sorry

end units_digit_diff_l649_649418


namespace infinite_sum_identity_l649_649939

theorem infinite_sum_identity (x : ℝ) (h : x > 1) :
  (∑' n : ℕ, 1 / (x^(3^n) - x^(-3^n))) = 1 / (x - 1) :=
sorry

end infinite_sum_identity_l649_649939


namespace num_positive_terms_l649_649747

def sequence_cos (n : ℕ) : ℝ :=
  cos (10^(n-1) * π / 180)

theorem num_positive_terms : (finset.filter (λ n : ℕ, 0 < sequence_cos n) (finset.range 100)).card = 99 :=
by
  sorry

end num_positive_terms_l649_649747


namespace number_of_common_tangents_l649_649888

/-- Define the circle C1 with center (2, -1) and radius 2. -/
def C1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 4}

/-- Define the symmetry line x + y - 3 = 0. -/
def symmetry_line := {p : ℝ × ℝ | p.1 + p.2 = 3}

/-- Circle C2 is symmetric to C1 about the line x + y = 3. -/
def C2 := {p : ℝ × ℝ | (p.1 - 4)^2 + (p.2 - 1)^2 = 4}

/-- Circle C3 with the given condition MA^2 + MO^2 = 10 for any point M on the circle. 
    A(0, 2) and O is the origin. -/
def C3 := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 4}

/-- The number of common tangents between circle C2 and circle C3 is 3. -/
theorem number_of_common_tangents
  (C1_sym_C2 : ∀ p : ℝ × ℝ, p ∈ C1 ↔ p ∈ C2)
  (M_on_C3 : ∀ M : ℝ × ℝ, M ∈ C3 → ((M.1)^2 + (M.2 - 2)^2) + ((M.1)^2 + (M.2)^2) = 10) :
  ∃ tangents : ℕ, tangents = 3 :=
sorry

end number_of_common_tangents_l649_649888


namespace find_palindrome_satisfying_conditions_l649_649043

-- Define a palindrome number condition
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 2 in s = s.reverse

-- Define 8-digit number condition
def is_eight_digit (n : ℕ) : Prop :=
  n >= 10^7 ∧ n < 10^8

-- Define digits composed of 0 and 1 condition
def is_composed_of_0_and_1 (n : ℕ) : Prop :=
  ∀ d ∈ n.to_digits 2, d = 0 ∨ d = 1

-- Define divisibility by 3 condition
def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Define prime divisor composition condition
def has_prime_divisors_with_digits_1_3_and_percent (n : ℕ) : Prop :=
  ∀ p ∈ n.factors, p.to_digits 10 ⊆ [1, 3, %].map (λ x, x.to_digit_var_constant x.nat)

-- Final statement combining all conditions
theorem find_palindrome_satisfying_conditions :
  ∃ n, is_palindrome n ∧ 
       is_eight_digit n ∧ 
       is_composed_of_0_and_1 n ∧ 
       is_multiple_of_3 n ∧ 
       has_prime_divisors_with_digits_1_3_and_percent n ∧ 
       n = 10111101 :=
sorry

end find_palindrome_satisfying_conditions_l649_649043


namespace part1_part2_l649_649849

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.exp x * (Real.log x + k)
noncomputable def f_prime (x : ℝ) (k : ℝ) : ℝ := Real.exp x * (Real.log x + k) + Real.exp x / x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f_prime x k - 2 * (f x k + Real.exp x)
noncomputable def phi (x : ℝ) : ℝ := Real.exp x / x

theorem part1 (h : f_prime 1 k = 0) : k = -1 := sorry

theorem part2 (t : ℝ) (h_g_le_phi : ∀ x > 0, g x (-1) ≤ t * phi x) : t ≥ 1 + 1 / Real.exp 2 := sorry

end part1_part2_l649_649849


namespace number_of_true_propositions_l649_649526

def parabola_opens_down (a b c : ℝ) : Prop := a < 0
def nonempty_solution_set (a b c : ℝ) : Prop := ∃ x : ℝ, a * x^2 + b * x + c < 0
def original_proposition (a b c : ℝ) : Prop := parabola_opens_down a b c → nonempty_solution_set a b c
def converse_proposition (a b c : ℝ) : Prop := nonempty_solution_set a b c → parabola_opens_down a b c
def inverse_proposition (a b c : ℝ) : Prop := ¬nonempty_solution_set a b c → ¬parabola_opens_down a b c
def contrapositive_proposition (a b c : ℝ) : Prop := ¬nonempty_solution_set a b c → ¬parabola_opens_down a b c

theorem number_of_true_propositions (a b c : ℝ) :
  (original_proposition a b c) ∧ ¬(converse_proposition a b c) ∧ ¬(inverse_proposition a b c) ∧ (contrapositive_proposition a b c) 
→ 2 := sorry

end number_of_true_propositions_l649_649526


namespace percentage_reduction_l649_649702

theorem percentage_reduction (P : ℝ) (h1 : 700 / P + 3 = 700 / 70) : 
  ((P - 70) / P) * 100 = 30 :=
by
  sorry

end percentage_reduction_l649_649702


namespace at_least_two_polyhedra_share_interior_point_l649_649827

open ConvexSet

variables (P1 : ConvexSet ℝ)
variables (A : Fin 9 → ℝ × ℝ × ℝ) -- representing the vertices A_1 to A_9

def translation (v : ℝ × ℝ × ℝ) (P : ConvexSet ℝ) : ConvexSet ℝ := {
  carrier := {p | ∃ q ∈ P, q + v = p},
  convex' := sorry
}

noncomputable def P (i : Fin 9) : ConvexSet ℝ :=
  if i = 0 then P1 else translation ((A i) - (A 0)) P1

theorem at_least_two_polyhedra_share_interior_point :
  ∃ i j : Fin 9, i ≠ j ∧ ∃ x, x ∈ interior (P i) ∧ x ∈ interior (P j) :=
sorry

end at_least_two_polyhedra_share_interior_point_l649_649827


namespace solve_system_of_equations_l649_649592

theorem solve_system_of_equations (x y : ℝ) :
  16 * x^3 + 4 * x = 16 * y + 5 ∧ 16 * y^3 + 4 * y = 16 * x + 5 → x = y ∧ 16 * x^3 - 12 * x - 5 = 0 :=
by
  sorry

end solve_system_of_equations_l649_649592


namespace ratio_of_managers_to_non_managers_l649_649901

theorem ratio_of_managers_to_non_managers (M N : ℕ) (hM : M = 8) (hN : N = 38) : M / N = 8 / 38 := by
  rw [hM, hN]
  sorry

end ratio_of_managers_to_non_managers_l649_649901


namespace fixed_points_range_a_l649_649813

noncomputable def has_two_fixed_points (f : ℝ → ℝ) (a : ℝ) (I : Set ℝ) : Prop :=
  ∃ x1 x2 ∈ I, x1 ≠ x2 ∧ f x1 = x1 ∧ f x2 = x2

theorem fixed_points_range_a :
  ∀ a : ℝ, (has_two_fixed_points (λ x, x^2 + a * x + 4) a (Set.Icc 1 3)) ↔ a ∈ Set.Ico (-10 / 3) (-3) := 
by
  sorry

end fixed_points_range_a_l649_649813


namespace triangle_side_ineq_l649_649459

theorem triangle_side_ineq (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a^2 * c + b^2 * a + c^2 * b < 1 / 8 := 
by 
  sorry

end triangle_side_ineq_l649_649459


namespace real_part_of_z_l649_649139

-- Definitions and conditions
def i : ℂ := complex.I
def z : ℂ := sorry -- z is a complex number such that condition holds
def magnitude : ℝ := complex.abs (1 + i)

-- Condition: (1 - i) z = |1 + i|
axiom condition : (1 - i) * z = magnitude

-- Objective: Prove the real part of z is sqrt(2) / 2
theorem real_part_of_z : complex.re z = real.sqrt 2 / 2 :=
by sorry

end real_part_of_z_l649_649139


namespace number_divisible_by_3_pow_2000_l649_649287

def largest_nontrivial_divisor (n : ℕ) : ℕ :=
  if h : n > 1 then
    (Finset.filter (λ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n) (Finset.range n)).max' sorry
  else 
    1

theorem number_divisible_by_3_pow_2000 (n : ℕ) (h : n > 1) : 
  ∃ m, (n + (Finset.range m).sum largest_nontrivial_divisor) ∣ 3^2000 :=
by
  sorry

end number_divisible_by_3_pow_2000_l649_649287


namespace determine_cubic_coeffs_l649_649411

-- Define the cubic function f(x)
def cubic_function (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Define the expression f(f(x) + x)
def composition_expression (a b c x : ℝ) : ℝ :=
  cubic_function a b c (cubic_function a b c x + x)

-- Given that the fraction of the compositions equals the given polynomial
def given_fraction_equals_polynomial (a b c : ℝ) : Prop :=
  ∀ x : ℝ, (composition_expression a b c x) / (cubic_function a b c x) = x^3 + 2023 * x^2 + 1776 * x + 2010

-- Prove that this implies specific values of a, b, and c
theorem determine_cubic_coeffs (a b c : ℝ) :
  given_fraction_equals_polynomial a b c →
  (a = 2022 ∧ b = 1776 ∧ c = 2010) :=
by
  sorry

end determine_cubic_coeffs_l649_649411


namespace two_digit_number_is_42_l649_649807

theorem two_digit_number_is_42 (a b : ℕ) (ha : a < 10) (hb : b < 10) (h : 10 * a + b = 42) :
  ((10 * a + b) : ℚ) / (10 * b + a) = 7 / 4 := by
  sorry

end two_digit_number_is_42_l649_649807


namespace vector_magnitude_l649_649870

def vec (x y : ℝ) : ℝ × ℝ := (x, y)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem vector_magnitude
  (k : ℝ)
  (a := vec (-2) k)
  (b := vec 2 4)
  (h : dot_product a b = 0) : magnitude (a.1 - b.1, a.2 - b.2) = 5 :=
by
  sorry

end vector_magnitude_l649_649870


namespace pie_not_eaten_fraction_l649_649545

theorem pie_not_eaten_fraction
  (lara_ate : ℚ) (ryan_ate : ℚ) (cassie_ate : ℚ) (not_eaten : ℚ) :
  lara_ate = 1 / 4 →
  ryan_ate = 3 / 10 →
  cassie_ate = 2 / 3 →
  not_eaten = 3 / 20 :=
by
  assume h_lara : lara_ate = 1 / 4,
  assume h_ryan : ryan_ate = 3 / 10,
  assume h_cassie : cassie_ate = 2 / 3,
  sorry

end pie_not_eaten_fraction_l649_649545


namespace infinite_series_sum_l649_649937

theorem infinite_series_sum (x : ℝ) (h : x > 1) :
  ∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (-(3 ^ n))) = 1 / (x - 1) :=
sorry

end infinite_series_sum_l649_649937


namespace investment_of_Q_l649_649663

theorem investment_of_Q (p_investment : ℕ) (ratio_pq : ℕ × ℕ) (h1 : p_investment = 12000) (h2 : ratio_pq = (3, 5)) :
  ∃ (q_investment : ℕ), q_investment = 20000 := 
by {
  use 20000,
  have hpq := ratio_pq,
  cases hpq with r1 r2,
  have hratio : (p_investment : ℕ) * r2 = r1 * 20000, from eq.refl _,
  rwa [h1, h2, nat.mul_comm, nat.mul_comm 3, nat.monotone_mul_left],
  exact nat.eq_of_mul_eq_mul_of_pos $ nat.zero_le 3,
  sorry
}

end investment_of_Q_l649_649663


namespace perimeter_parallelogram_ABCD_l649_649460

variable (AB BC : ℝ) (parallelogram_ABCD : Prop)

def is_parallelogram (A B C D : Prop) : Prop :=
  ∃ (A B C D : ℝ), parallelogram_ABCD ∧ AB = 14 ∧ BC = 16

theorem perimeter_parallelogram_ABCD (AB BC : ℝ) (h : is_parallelogram AB BC) : 
  AB + BC + AB + BC = 60 := 
  by
  sorry

end perimeter_parallelogram_ABCD_l649_649460


namespace problem_I_problem_II_problem_III_l649_649852

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / Real.exp x
noncomputable def h (x : ℝ) : ℝ := 1 - x - x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x * (deriv (fun x => (Real.log x + 1) / Real.exp x)) x

theorem problem_I : 
  let f_val := 1 / Real.exp 1 in
  ∀ x, (x = 1) → ((fun x => f x) 1 = f_val) → (deriv (fun x => (Real.log x + 1) / Real.exp x)) 1 = 0 → True := 
sorry

theorem problem_II : ∀ x, (h x).sup (1 + Real.exp (-2)) := 
sorry

theorem problem_III : ∀ x, 0 < x → g x < 1 + Real.exp (-2) := 
sorry

end problem_I_problem_II_problem_III_l649_649852


namespace bisectors_angle_APC_l649_649833

-- Definitions of points and line
variables {A B C D P O₂ : Point}
variables (Δ : Line A B C D) [LinePow Δ]
variables {Γ₁ : Circle A C P} [CirclePow Γ₁]
variables {Γ₂ : Circle B D P} [CirclePow Γ₂]

-- Intersection point of circles
variables {P : Point} (hP: P ∈ Γ₁ ∧ P ∈ Γ₂)

-- Condition for tangent through the center
variables (hp₁: TangentAt Γ₁ P O₂) (hp₂: Center Γ₂ O₂ D)

-- Definition of the angle bisectors
variables (α δ : Angle)
variables (γ₁ : ∠ A P C = α) (γ₂ : ∠ P D B = δ)

-- Proof statement
theorem bisectors_angle_APC : 
  IsBisector (Segment P B) (Segment A P C) ∧ IsBisector (Segment P D) (Segment A P C) :=
begin
  sorry
end

end bisectors_angle_APC_l649_649833


namespace infinite_sum_identity_l649_649940

theorem infinite_sum_identity (x : ℝ) (h : x > 1) :
  (∑' n : ℕ, 1 / (x^(3^n) - x^(-3^n))) = 1 / (x - 1) :=
sorry

end infinite_sum_identity_l649_649940


namespace arithmetic_sequence_a15_value_l649_649845

variables {a : ℕ → ℤ}

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a15_value
  (h1 : is_arithmetic_sequence a)
  (h2 : a 3 + a 13 = 20)
  (h3 : a 2 = -2) : a 15 = 24 :=
by sorry

end arithmetic_sequence_a15_value_l649_649845


namespace series_sum_l649_649949

noncomputable def compute_series (x : ℝ) (hx : x > 1) : ℝ :=
  ∑' n, 1 / (x ^ (3 ^ n) - x ^ (- 3 ^ n))

theorem series_sum (x : ℝ) (hx : x > 1) : compute_series x hx = 1 / (x - 1) :=
sorry

end series_sum_l649_649949


namespace opposite_sides_line_l649_649507

theorem opposite_sides_line (m : ℝ) : 
  (2 * 1 + 3 + m) * (2 * -4 + -2 + m) < 0 ↔ -5 < m ∧ m < 10 :=
by sorry

end opposite_sides_line_l649_649507


namespace f_of_1_l649_649552

def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then x^2 - (1/2)*x else 0

axiom f_odd : is_odd f

theorem f_of_1 : f 1 = -3/2 :=
by
  sorry

end f_of_1_l649_649552


namespace shortest_altitude_l649_649017

noncomputable def hypotenuse_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

noncomputable def area_of_triangle (a b : ℝ) : ℝ :=
  (1 / 2) * a * b

noncomputable def altitude_to_hypotenuse (a b : ℝ) : ℝ :=
  let c := hypotenuse_length a b
  in (2 * area_of_triangle a b) / c

theorem shortest_altitude (a b : ℝ) (ha : a = 9) (hb : b = 12) :
  altitude_to_hypotenuse a b = 7.2 :=
by
  rw [ha, hb, altitude_to_hypotenuse, hypotenuse_length, area_of_triangle]
  norm_num
  -- area = 54
  -- hypotenuse c = 15
  -- altitude = 2 * area / c = 108 / 15 = 7.2
  sorry

end shortest_altitude_l649_649017


namespace total_participants_l649_649632

open Real

theorem total_participants (F M : ℕ) (half_dem_fem : 2 * 165 = F) (one_quarter_dem_male : 0.25 * M = (1/4) * M ) (one_third_dem_total : (1/3) * (F + M) = 165 + (1/4) * M) (female_dem_id : (1/2) * F = 165) : F + M = 990 :=
by
  -- definitions to be used
  sorry

end total_participants_l649_649632


namespace card_at_position_52_l649_649921

def cards_order : List String := ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

theorem card_at_position_52 : cards_order[(52 % 13)] = "A" :=
by
  -- proof will be added here
  sorry

end card_at_position_52_l649_649921


namespace triangle_inequality_l649_649573

structure Triangle (A B C : Type) :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)
  (midpoint_AB : ℝ × ℝ) -- Midpoint of AB
  (midpoint_BC : ℝ × ℝ) -- Midpoint of BC
  (M : ℝ × ℝ) -- Point M on AC

-- Define the points D, E using midpoints
def D (T : Triangle ℝ ℝ ℝ) : ℝ × ℝ := T.midpoint_AB
def E (T : Triangle ℝ ℝ ℝ) : ℝ × ℝ := T.midpoint_BC

-- Conditions provided in the problem
def conditions (T : Triangle ℝ ℝ ℝ) : Prop :=
  let A := (0, 0) in -- Vertex A
  let B := (T.AB, 0) in -- Vertex B
  let C := (T.CA * cos (60 * pi/180), T.CA * sin (60 * pi/180)) in -- Vertex C on AC
  let M := T.M in
  let D := T.midpoint_AB in
  let E := T.midpoint_BC in
  ME_dist := dist M E > dist E C -- Given ME > EC
  -- More explicit conditions based on triangle properties can be added here

-- The statement to be proved
theorem triangle_inequality (T : Triangle ℝ ℝ ℝ) (h : conditions T) : 
  dist (T.M) (D T) < dist (D T) ((0, 0)) :=
sorry

end triangle_inequality_l649_649573


namespace difference_value_l649_649604

theorem difference_value (N : ℝ) (h : 0.25 * N = 100) : N - (3/4) * N = 100 :=
by sorry

end difference_value_l649_649604


namespace man_salary_value_l649_649659

-- Define the variables and conditions
def man_salary (S : ℝ) : Prop :=
  let food_expense := (1 / 3) * S in
  let rent_expense := (1 / 4) * S in
  let clothes_expense := (1 / 5) * S in
  let remaining_amount := 1760 in
  S - food_expense - rent_expense - clothes_expense = remaining_amount

-- State the theorem we want to prove
theorem man_salary_value: ∃ S : ℝ, man_salary S ∧ S = 812 :=
by
  sorry

end man_salary_value_l649_649659


namespace total_flowers_in_all_gardens_l649_649285

theorem total_flowers_in_all_gardens (pots_in_each_garden : ℕ) (flowers_in_each_pot : ℕ) (number_of_gardens : ℕ) :
  pots_in_each_garden = 544 → flowers_in_each_pot = 32 → number_of_gardens = 10 →
  pots_in_each_garden * flowers_in_each_pot * number_of_gardens = 174080 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact eq.refl _

end total_flowers_in_all_gardens_l649_649285


namespace plane_coloring_l649_649301

-- Defining the mathematically equivalent proof problem based on the conditions

theorem plane_coloring (plane_coloring : ℝ² → ℕ) :
  (∀ x : ℝ², plane_coloring x ∈ {0, 1, 2}) → 
  ∃ (x y : ℝ²), (dist x y = 1) ∧ (plane_coloring x = plane_coloring y) :=
by
  -- Define points A, B, C, and D
  let A := (0 : ℝ, 0 : ℝ)
  let B := (√3, 0 : ℝ)
  let C := (0.5, √3 / 2)
  let D := (0.5 + √3, √3 / 2)
  
  sorry -- Proof steps are omitted for this task

end plane_coloring_l649_649301


namespace complete_square_l649_649991

theorem complete_square (a b c : ℕ) (h : 49 * x ^ 2 + 70 * x - 121 = 0) :
  a = 7 ∧ b = 5 ∧ c = 146 ∧ a + b + c = 158 :=
by sorry

end complete_square_l649_649991


namespace combined_alligator_walk_time_l649_649165

theorem combined_alligator_walk_time :
  (Paul_to_Delta : ℕ) (Delta_to_River_inc : ℕ) (Number_of_alligators_on_return : ℕ) (Paul_to_Delta = 4) (Delta_to_River_inc = 2) (Number_of_alligators_on_return = 6) :
  let Delta_to_River := Paul_to_Delta + Delta_to_River_inc
  let Total_Paul := Paul_to_Delta + Delta_to_River
  let Alligator_return_total_time := Number_of_alligators_on_return * Delta_to_River
  let Combined_total_time := Total_Paul + Alligator_return_total_time
  Combined_total_time = 46 :=
by
  sorry

end combined_alligator_walk_time_l649_649165


namespace probability_of_above_parabola_l649_649236

def is_above_parabola (a b : ℕ) : Prop :=
  ∀ x : ℝ, b > a * x^2 + b * x

def count_valid_points : ℕ :=
  (Finset.range 10).sum (λ a, if a = 0 ∨ a = 1 then 9 else if a = 2 then 7 else if a = 3 then 3 else 0)

def total_combinations : ℕ :=
  10 * 10

theorem probability_of_above_parabola : ((count_valid_points : ℝ) / total_combinations) = 7 / 25 := by
  sorry

end probability_of_above_parabola_l649_649236


namespace solve_for_m_l649_649508

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 2 → - (1/2) * x^2 + 2 * x > m * x) → m = 1 :=
by
  -- Skip the proof by using sorry
  sorry

end solve_for_m_l649_649508


namespace intercept_sum_l649_649693

theorem intercept_sum (x y : ℝ) (h : y - 3 = -3 * (x + 2)) :
  (∃ (x_int : ℝ), y = 0 ∧ x_int = -1) ∧ (∃ (y_int : ℝ), x = 0 ∧ y_int = -3) →
  (-1 + (-3) = -4) := by
  sorry

end intercept_sum_l649_649693


namespace total_number_of_friends_l649_649234

theorem total_number_of_friends (
    (total_amount := 600),
    (share := 120),
    (friends_except_one := 5),
    (extra_paid := 100),
    (one_paid := 220)
  ) : 
  (find_total_friends := friends_except_one + 1)
  (total_friends n : ℕ) :=
  n = total_amount / share → 
  n + friends_except_one = find_total_friends → 
  find_total_friends = 6 :=
begin
  sorry
end

end total_number_of_friends_l649_649234


namespace gcd_square_le_sum_l649_649673

theorem gcd_square_le_sum (a b : ℕ) (h1 : (a + 1) % b = 0) (h2 : (b + 1) % a = 0) : 
  let d := Nat.gcd a b in d * d ≤ a + b :=
by
  sorry

end gcd_square_le_sum_l649_649673


namespace carlson_total_land_l649_649381

open Real

theorem carlson_total_land 
  (initial_land : ℝ)
  (cost_additional_land1 : ℝ)
  (cost_additional_land2 : ℝ)
  (cost_per_square_meter : ℝ) :
  initial_land = 300 →
  cost_additional_land1 = 8000 →
  cost_additional_land2 = 4000 →
  cost_per_square_meter = 20 →
  (initial_land + (cost_additional_land1 + cost_additional_land2) / cost_per_square_meter) = 900 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  done

end carlson_total_land_l649_649381


namespace triangular_pyramid_surface_area_correct_l649_649079

noncomputable def triangular_pyramid_surface_area (O : Type) [Metric O] : ℝ :=
  let A B C D : O
  let BC := dist B C
  let AB := dist A B
  let AC := dist A C
  let BD := dist B D
  let CD := dist C D
  let AD := dist A D
  let θ := angle A D {B, C, D}
  if AB = 2 ∧ AC = 2 ∧ BD = 2 ∧ CD = 2 ∧ BC = 2 * AD ∧ θ = π / 3 then 8 * π else 0

-- Now declare a theorem to assert this.
theorem triangular_pyramid_surface_area_correct :
  ∀ (O : Type) [Metric O],
  let A B C D : O
  let BC := dist B C
  let AB := dist A B
  let AC := dist A C
  let BD := dist B D
  let CD := dist C D
  let AD := dist A D
  let θ := angle A D {B, C, D}
  (AB = 2 ∧ AC = 2 ∧ BD = 2 ∧ CD = 2 ∧ BC = 2 * AD ∧ θ = π / 3) →
  triangular_pyramid_surface_area O = 8 * π :=
by
  intros
  sorry

end triangular_pyramid_surface_area_correct_l649_649079


namespace lemons_minus_pears_l649_649123

theorem lemons_minus_pears
  (apples : ℕ)
  (pears : ℕ)
  (tangerines : ℕ)
  (lemons : ℕ)
  (watermelons : ℕ)
  (h1 : apples = 8)
  (h2 : pears = 5)
  (h3 : tangerines = 12)
  (h4 : lemons = 17)
  (h5 : watermelons = 10) :
  lemons - pears = 12 := 
sorry

end lemons_minus_pears_l649_649123


namespace number_of_ways_l649_649061

theorem number_of_ways :
  ∃ (a : Fin 15 → Fin 15 → Fin 15 → Prop), 
  (∀ a1 a2 a3, a a1 a2 a3 = 
      (0 ≤ a1 ∧ a1 < a2 ∧ a2 < a3 ∧ a3 < 14 ∧ (a2 - a1) ≥ 3 ∧ (a3 - a2) ≥ 3)
      →
      ∃ (b : ℕ) (h : b = 120), (finset.univ.filter (λ (a1, a2, a3), 
        a a1.val a2.val a3.val ∧ a1 < a2 
        ∧ a2 < a3 ∧ (a2 - a1) ≥ 3 ∧ (a3 - a2) ≥ 3)).card = b) :=
sorry

end number_of_ways_l649_649061


namespace num_factors_of_1320_l649_649720

theorem num_factors_of_1320 : ∃ n : ℕ, (n = 24) ∧ (∃ a b c d : ℕ, 1320 = 2^a * 3^b * 5^c * 11^d ∧ (a + 1) * (b + 1) * (c + 1) * (d + 1) = 24) :=
by
  sorry

end num_factors_of_1320_l649_649720


namespace positive_terms_count_l649_649766

def sequence (n : ℕ) : ℝ := Float.cos (10.0^(n-1) * Float.pi / 180.0)

theorem positive_terms_count :
  (List.filter (λ n, sequence n > 0) (List.range' 1 100)).length = 99 :=
sorry

end positive_terms_count_l649_649766


namespace fraction_squares_sum_l649_649483

variable {ℝ : Type} [Field ℝ]

theorem fraction_squares_sum
    (x y z m n p : ℝ)
    (h1 : x / m + y / n + z / p = 1)
    (h2 : m / x + n / y + p / z = 0)
    :
    x^2 / m^2 + y^2 / n^2 + z^2 / p^2 = 1 :=
by
  sorry

end fraction_squares_sum_l649_649483


namespace solve_for_x_l649_649230

theorem solve_for_x (x : ℝ) (h : (2 / 7) * (1 / 3) * x = 14) : x = 147 :=
sorry

end solve_for_x_l649_649230


namespace find_functions_l649_649330

theorem find_functions {f : ℝ → ℝ} :
  (∀ I : set ℝ, is_open I ∧ ∃ a b, b > a ∧ I = set.Ioo a b →
  ∃ J : set ℝ, is_open J ∧ ∃ c d, d > c ∧ J = set.Ioo c d ∧ set.image f I = J ∧ (d - c = b - a)) →
  ∃ C : ℝ, ∀ x, f x = x + C ∨ f x = -x + C :=
begin
  sorry,
end

end find_functions_l649_649330


namespace sequence_expression_l649_649074

noncomputable def a : ℕ → ℕ
| 1       := 1
| (n + 2) := 2 * a (n + 1) + 1

theorem sequence_expression (n : ℕ) (hn : n ≥ 1) : a n = 2^n - 1 := by
  sorry

end sequence_expression_l649_649074


namespace chess_piece_return_even_l649_649831

-- Define the movement of the chess piece on the infinite chessboard
def move (m n : ℕ) (pos : ℤ × ℤ) : ℤ × ℤ :=
  match pos with
  | (a, b) =>
    (a + m, b + n) ∨ (a - m, b + n) ∨ (a + m, b - n) ∨ (a - m, b - n) ∨ 
    (a + n, b + m) ∨ (a - n, b + m) ∨ (a + n, b - m) ∨ (a - n, b - m)

theorem chess_piece_return_even (m n : ℕ) (a b : ℤ) (x : ℕ) :
  (∀ i, i < x → ∃ pos, move m n pos = (a, b)) →
  x % 2 = 0 :=
sorry

end chess_piece_return_even_l649_649831


namespace sum_series_eq_l649_649953

open Real

theorem sum_series_eq (x : ℝ) (h : 1 < x) : 
  (∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (- (3 ^ n)))) = 1 / (x - 1) :=
sorry

end sum_series_eq_l649_649953


namespace imaginary_part_of_1_plus_2i_l649_649612

theorem imaginary_part_of_1_plus_2i : complex.im (1 + 2 * complex.I) = 2 := 
by
  sorry

end imaginary_part_of_1_plus_2i_l649_649612


namespace solve_for_x_l649_649590

theorem solve_for_x (x : ℝ) (h : 4^x = Real.sqrt 64) : x = 3 / 2 :=
sorry

end solve_for_x_l649_649590


namespace largest_angle_is_120_l649_649529

variable {d e f : ℝ}

def condition1 : Prop := d + 3 * e + 3 * f = d^2
def condition2 : Prop := d + 3 * e - 3 * f = -3

theorem largest_angle_is_120 (h1 : condition1) (h2 : condition2) : ∃ θ : ℝ, θ = 120 ∧ θ = largest_angle d e f :=
by sorry

end largest_angle_is_120_l649_649529


namespace arrange_in_ascending_order_l649_649438

open Real

noncomputable def a := log 3 / log (1/2)
noncomputable def b := log 5 / log (1/2)
noncomputable def c := log (1/2) / log (1/3)

theorem arrange_in_ascending_order : b < a ∧ a < c :=
by
  sorry

end arrange_in_ascending_order_l649_649438


namespace sum_of_distinct_divisors_l649_649198

theorem sum_of_distinct_divisors (n : ℕ) (k : ℕ) (h : 1 < n) 
  (m : ℕ := 2 ^ (n - 1) * (2 ^ n - 1)) (hk : 1 ≤ k) (hmk : k ≤ m) :
  ∃ (s : Finset ℕ), (s.val.sum id = k) ∧ (∀ x ∈ s, x ∣ m) :=
sorry

end sum_of_distinct_divisors_l649_649198


namespace SallyCarrots_l649_649579

-- Definitions of the conditions
def FredGrew (F : ℕ) := F = 4
def TotalGrew (T : ℕ) := T = 10
def SallyGrew (S : ℕ) (F T : ℕ) := S + F = T

-- The theorem to be proved
theorem SallyCarrots : ∃ S : ℕ, FredGrew 4 ∧ TotalGrew 10 ∧ SallyGrew S 4 10 ∧ S = 6 :=
  sorry

end SallyCarrots_l649_649579


namespace percent_increase_stock_K_l649_649007

/-- The percentage increase in the price per share of stock K from $10 to $15 is 50%. -/
theorem percent_increase_stock_K : 
  let initial_price := 10
  let final_price := 15
  let percent_increase := ((final_price - initial_price) / initial_price.toReal) * 100
  percent_increase = 50 :=
by {
  sorry,
}

end percent_increase_stock_K_l649_649007


namespace maximum_triangles_formed_l649_649299

theorem maximum_triangles_formed (n : ℕ) (h : n = 100) (no_triangles : ∀ (s : ℕ) (hs : s < n), 
                                ¬ ∃ a b c : ℕ, a < b ∧ b < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  let new_n := n + 1 in n = 100 := by
  sorry

end maximum_triangles_formed_l649_649299


namespace sum_of_x_f_of_f_of_f_eq_neg3_l649_649699

def f (x : ℝ) : ℝ := x^2 / 4 + x - 3

theorem sum_of_x_f_of_f_of_f_eq_neg3 :
  ∑ x in {x : ℝ | f (f(f x)) = -3}.to_finset, x = -8 := 
  sorry

end sum_of_x_f_of_f_of_f_eq_neg3_l649_649699


namespace distinct_meals_count_l649_649006

def entries : ℕ := 3
def drinks : ℕ := 3
def desserts : ℕ := 3

theorem distinct_meals_count : entries * drinks * desserts = 27 :=
by
  -- sorry for skipping the proof
  sorry

end distinct_meals_count_l649_649006


namespace volume_pyramid_l649_649630

theorem volume_pyramid (V : ℝ) : 
  ∃ V_P : ℝ, V_P = V / 6 :=
by
  sorry

end volume_pyramid_l649_649630


namespace jack_needs_5_rocks_to_equal_weights_l649_649535

-- Given Conditions
def WeightJack : ℕ := 60
def WeightAnna : ℕ := 40
def WeightRock : ℕ := 4

-- Theorem Statement
theorem jack_needs_5_rocks_to_equal_weights : (WeightJack - WeightAnna) / WeightRock = 5 :=
by
  sorry

end jack_needs_5_rocks_to_equal_weights_l649_649535


namespace sum_of_reciprocal_roots_eq_three_halves_l649_649101

variable (a b c : ℝ)

noncomputable def polynomial : ℝ → ℝ := 
  λ x, 40 * x^3 - 60 * x^2 + 24 * x - 1

theorem sum_of_reciprocal_roots_eq_three_halves
  (ha : polynomial a = 0) (hb : polynomial b = 0) (hc : polynomial c = 0)
  (h_range : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) :
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 :=
sorry

end sum_of_reciprocal_roots_eq_three_halves_l649_649101


namespace last_three_digits_of_5_pow_9000_l649_649838

theorem last_three_digits_of_5_pow_9000 :
  (5^9000) % 1000 = 1 :=
by {
  have key_condition : (5^300) % 125 = 1,
  { sorry }, -- given condition
  sorry -- combining steps through CRT
}

end last_three_digits_of_5_pow_9000_l649_649838


namespace magnitude_product_l649_649405

def z1 : ℂ := 3 * Real.sqrt 3 - 3 * Complex.i
def z2 : ℂ := 2 * Real.sqrt 2 + 2 * Complex.i

theorem magnitude_product :
  Complex.abs (z1 * z2) = 12 * Real.sqrt 3 :=
by 
  sorry

end magnitude_product_l649_649405


namespace ratio_of_eighth_terms_l649_649269

noncomputable def sum_arith_seq (a₁ aₙ : ℕ) (n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

variables {a₁ a₈ b₁ b₈ : ℕ}
variables {S_n T_n : ℕ → ℕ}

-- Given conditions
axiom sum_condition (n : ℕ) : (S_n n / T_n n = (7 * n + 3) / (n + 3))

-- Proof statement
theorem ratio_of_eighth_terms (n : ℕ) (h₁ : ∃ (a₁ a₈ : ℕ), S_n n = sum_arith_seq a₁ a₈ n)
                              (h₂ : ∃ (b₁ b₈ : ℕ), T_n n = sum_arith_seq b₁ b₈ n) : a₁ = 7 * a₈ :=
begin
  -- Sorry substitute for the proof details
  sorry
end

end ratio_of_eighth_terms_l649_649269


namespace find_sum_of_exponents_l649_649633

theorem find_sum_of_exponents :
  ∃ (s : ℕ) (m : Fin s → ℕ) (b : Fin s → ℤ),
    (∀ i j, i < j → m i > m j) ∧
    (∀ k, b k = 1 ∨ b k = -1) ∧
    (∑ k, b k * 3 ^ m k = 2022) →
    (∑ k, m k = 28) :=
by
  sorry

end find_sum_of_exponents_l649_649633


namespace trig_inequality_l649_649810

theorem trig_inequality (ϕ : ℝ) (hϕ : 0 < ϕ ∧ ϕ < π / 2) : 
  sin (cos ϕ) < cos ϕ ∧ cos ϕ < cos (sin ϕ) := 
sorry

end trig_inequality_l649_649810


namespace example_not_5_order_repeatable_minimal_length_for_2_order_repeatable_l649_649076

-- Define what it means for a sequence to be k-repeated
def is_k_order_repeatable_seq (a : ℕ → ℕ) (k m : ℕ) : Prop :=
  ∃ i j, 1 ≤ i ∧ i + k - 1 ≤ m ∧ 1 ≤ j ∧ j + k - 1 ≤ m ∧ i ≠ j ∧
    (∀ n, 0 ≤ n ∧ n < k → a (i + n) = a (j + n))

def example_sequence := [0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
def example_seq_fun : ℕ → ℕ
| n := if n < 10 then example_sequence.nth_le n sorry else 0

-- (1) Proving the example sequence is not a 5-order repeatable sequence
theorem example_not_5_order_repeatable :
  ¬ is_k_order_repeatable_seq example_seq_fun 5 10 := by
  sorry

-- (2) Proving the minimal length m for all 0-1 sequences to be 2-order repeatable is 5
theorem minimal_length_for_2_order_repeatable : 
  ∀ (a : ℕ → ℕ), (∀ n, 0 ≤ n ∧ n < 5 → a n ∈ {0, 1}) →
  is_k_order_repeatable_seq a 2 5 := by
  sorry

end example_not_5_order_repeatable_minimal_length_for_2_order_repeatable_l649_649076


namespace min_sum_of_factors_l649_649608

theorem min_sum_of_factors (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
(h_prod : a * b * c * d = 8!) : a + b + c + d = 130 :=
by
  sorry

end min_sum_of_factors_l649_649608


namespace cost_for_Greg_l649_649377

theorem cost_for_Greg (N P M : ℝ)
(Bill : 13 * N + 26 * P + 19 * M = 25)
(Paula : 27 * N + 18 * P + 31 * M = 31) :
  24 * N + 120 * P + 52 * M = 88 := 
sorry

end cost_for_Greg_l649_649377


namespace total_coffee_drank_together_l649_649034

variables (x : ℕ)  -- Ed's cup size in ounces

-- Define Jane's cup size as 75% more than Ed's cup
def janes_cup : ℕ := (7 * x) / 4

-- Define the initial consumption for both Ed and Jane
def initial_consumed (c : ℕ) : ℕ := (2 * c) / 3

-- Define remaining coffee for Ed and Jane
def remaining (c : ℕ) : ℕ := c - initial_consumed c

-- Define the transfer amount from Jane to Ed
def transfer_amount : ℕ := (remaining (janes_cup x) / 2) + 3

-- Calculate the total amount of coffee each drank
def ed_final_amount : ℕ := initial_consumed x + transfer_amount
def jane_final_amount : ℕ := initial_consumed (janes_cup x) - transfer_amount

-- Define the proof problem statement
theorem total_coffee_drank_together : ed_final_amount x + jane_final_amount x = 44 :=
by
  sorry

end total_coffee_drank_together_l649_649034


namespace problem_solution_l649_649114

def sequence_a (n : ℕ) (hn : n > 0) : ℚ :=
  4 / (11 - 2 * n : ℚ)

theorem problem_solution (n : ℕ) (hn : n > 0) :
  (sequence_a (n + 1) (nat.succ_pos n) < sequence_a n hn) ↔ (n = 5) :=
sorry

end problem_solution_l649_649114


namespace num_subsets_neither_l649_649549

open Finset

variable {α : Type} [Finite α] -- Ensures we're dealing with finite sets

/-- Let X be a finite set such that |X| = 10, 
and A, B be disjoint subsets of X where |A| = 4 and |B| = 3.
Prove that the number of subsets of X that contain neither A nor B is 840. -/
theorem num_subsets_neither (X A B : Finset α) (hX : X.card = 10) (hA : A.card = 4) (hB : B.card = 3) (disjointAB : Disjoint A B) :
  (X.powerset.filter (λ S, Disjoint S A ∧ Disjoint S B)).card = 840 := 
sorry

end num_subsets_neither_l649_649549


namespace math_problem_l649_649384

theorem math_problem : -5 * (-6) - 2 * (-3 * (-7) + (-8)) = 4 := 
  sorry

end math_problem_l649_649384


namespace constant_term_l649_649247

noncomputable def polynomial := (4 * (x : ℝ)^2 - 2) * ((1 + (1 / x^2))^5)

theorem constant_term (x : ℝ) : 
  (polynomial : ℝ) = 18 :=
by sorry

end constant_term_l649_649247


namespace pentagon_angles_computation_l649_649351

/-- A pentagon ABCDE is inscribed in a circle,
with sides AB = BC = CD = 6, DE = 3, and AE = 2.
Prove that (1 - cos ∠B)(1 - cos ∠ADE) = 25/96. -/
theorem pentagon_angles_computation 
  (ABCDE : Type)
  [HasSideLengths ABCDE 6 6 6 3 2] : 
  (1 - cos (angle B)) * (1 - cos (angle ADE)) = (25 : ℚ) / 96 :=
sorry

end pentagon_angles_computation_l649_649351


namespace n_fraction_of_sum_l649_649337

theorem n_fraction_of_sum (n S : ℝ) (h1 : n = S / 5) (h2 : S ≠ 0) :
  n = 1 / 6 * ((S + (S / 5))) :=
by
  sorry

end n_fraction_of_sum_l649_649337


namespace problem_statement_l649_649444

variables {α β : Plane} {m : Line}

def parallel (a b : Plane) : Prop := sorry
def perpendicular (m : Line) (π : Plane) : Prop := sorry

axiom parallel_symm {a b : Plane} : parallel a b → parallel b a
axiom perpendicular_trans {m : Line} {a b : Plane} : perpendicular m a → parallel a b → perpendicular m b

theorem problem_statement (h1 : parallel α β) (h2 : perpendicular m α) : perpendicular m β :=
  perpendicular_trans h2 (parallel_symm h1)

end problem_statement_l649_649444


namespace calc_g_f_neg_2_l649_649931

def f (x : ℝ) : ℝ := x^3 - 4 * x + 3
def g (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 1

theorem calc_g_f_neg_2 : g (f (-2)) = 25 := by
  sorry

end calc_g_f_neg_2_l649_649931


namespace problem_statement_l649_649016

theorem problem_statement :
  let M := ∑ k in finset.range 67, (200 - 3 * k) ^ 2 - (197 - 3 * k) ^ 2 + (199 - 3 * k) ^ 2 - (196 - 3 * k) ^ 2 + (198 - 3 * k) ^ 2 - (195 - 3 * k) ^ 2
  in M = 60300 :=
by sorry

end problem_statement_l649_649016


namespace factorize_expression_l649_649039

theorem factorize_expression (a b m : ℝ) :
  a^2 * (m - 1) + b^2 * (1 - m) = (m - 1) * (a + b) * (a - b) :=
by sorry

end factorize_expression_l649_649039


namespace param_a_value_conditions_l649_649042

theorem param_a_value_conditions (a : ℝ) :
  (∃ x : ℝ, x ≥ 2 ∧ log (sqrt (a / 10)) ((a + 8 * x - x ^ 2) / 20) ≥ 2) ∧ 
  (∀ x : ℝ, x ≥ 2 → log (sqrt (a / 10)) ((a + 8 * x - x ^ 2) / 20) ≥ 2 → 
     abs (a + 2 * x - 16) + abs (a - 2 * x + 9) = abs (2 * a - 7)) ↔ 
  (9 ≤ a ∧ a < 10 ∨ 12 ≤ a ∧ a ≤ 16) :=
sorry

end param_a_value_conditions_l649_649042


namespace solve_E_rr4_eq_1296_l649_649787

def E (a b : ℝ) (c : ℕ) : ℝ := a * b^c

theorem solve_E_rr4_eq_1296 (r : ℝ) (h : E r r 4 = 1296) : r = 6^(0.8) :=
sorry

end solve_E_rr4_eq_1296_l649_649787


namespace fraction_left_after_two_days_l649_649685

theorem fraction_left_after_two_days (V : ℝ) (hV : V = 2) : 
  let remaining_after_first_day := V - V / 4 in
  let remaining_after_second_day := remaining_after_first_day - remaining_after_first_day / 2 in
  remaining_after_second_day / V = 3 / 8 := 
by
  sorry

end fraction_left_after_two_days_l649_649685


namespace math_books_count_l649_649669

theorem math_books_count (M H : ℕ) (h1 : M + H = 80) (h2 : 4 * M + 5 * H = 373) : M = 27 :=
by
  sorry

end math_books_count_l649_649669


namespace lower_limit_for_a_l649_649499

theorem lower_limit_for_a 
  {k : ℤ} 
  (a b : ℤ) 
  (h1 : k ≤ a) 
  (h2 : a < 17) 
  (h3 : 3 < b) 
  (h4 : b < 29) 
  (h5 : 3.75 = 4 - 0.25) 
  : (7 ≤ a) :=
sorry

end lower_limit_for_a_l649_649499


namespace bc_sum_l649_649189

open Real

variables (b1 b2 b3 c1 c2 c3 : ℝ)
definition Q (x : ℝ) := x^7 - x^6 + x^5 - x^4 + x^3 - x^2 + x - 1

theorem bc_sum :
  (∀ x : ℝ, Q x = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3)) →
  c3 = 1 →
  b1 + b2 + b3 = -1 →
  b1 * c1 + b2 * c2 + b3 = -1 :=
by
  intros h1 h2 h3
  /- Proof steps here -/
  sorry

end bc_sum_l649_649189


namespace min_value_of_m_squared_plus_n_squared_l649_649894

theorem min_value_of_m_squared_plus_n_squared (m n : ℝ) 
  (h : 4 * m - 3 * n - 5 * Real.sqrt 2 = 0) : m^2 + n^2 = 2 :=
sorry

end min_value_of_m_squared_plus_n_squared_l649_649894


namespace range_of_a_l649_649863

theorem range_of_a :
  (∃ a : ℝ, (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0)
    ↔ a ≤ -2 ∨ a = 1) := 
sorry

end range_of_a_l649_649863


namespace cosine_450_eq_0_l649_649734

theorem cosine_450_eq_0 : Real.cos (450 * Real.pi / 180) = 0 :=
by
  -- Convert 450 degrees to radians
  have h1 : (450 : ℝ) = 360 + 90 := by norm_num
  -- Use the periodic property of cosine
  have h2 : ∀ x : ℝ, Real.cos (x + 2 * Real.pi) = Real.cos x := Real.cos_periodic (2 * Real.pi)
  -- Since 450 degrees = 360 degrees + 90 degrees
  rw [h1, Real.cos_add, h2 (90 * Real.pi / 180)]
  -- Convert 90 degrees to radians and solve
  rw [Real.pi_mul_base_div, Real.pi_div_two, Real.cos_pi_div_two]
  norm_num

end cosine_450_eq_0_l649_649734


namespace area_of_ABCD_is_234_l649_649001

noncomputable def area_of_quadrilateral (AB BC CD DA : ℝ) (angleB : ℝ) : ℝ :=
  if angleB = 90 then
    let AC := Real.sqrt (AB^2 + BC^2) in
    let S_ABC := 1/2 * AB * BC in
    let S_ACD := 1/2 * DA * CD in
    S_ABC + S_ACD
  else
    0 -- The else branch is just a placeholder and not part of the actual problem conditions

theorem area_of_ABCD_is_234 : 
  area_of_quadrilateral 7 24 20 15 90 = 234 :=
by
  -- Skipping the proof
  sorry

end area_of_ABCD_is_234_l649_649001


namespace possible_values_of_Q_l649_649973

theorem possible_values_of_Q (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ∃ Q : ℝ, Q = 8 ∨ Q = -1 := 
sorry

end possible_values_of_Q_l649_649973


namespace exists_point_on_line_at_distance_from_plane_l649_649789

variables {ℝ : Type*} [linear_ordered_field ℝ]

-- Definitions for a line and a plane.
structure Line :=
  (point : ℝ × ℝ × ℝ)
  (direction : ℝ × ℝ × ℝ)

structure Plane :=
  (normal : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)

-- The perpendicular distance from a point to a plane.
def dist_point_to_plane (pt : ℝ × ℝ × ℝ) (pl : Plane) : ℝ :=
  let ⟨nx, ny, nz⟩ := pl.normal in
  let ⟨px, py, pz⟩ := pl.point in
  let ⟨x, y, z⟩ := pt in
  abs (nx * (x - px) + ny * (y - py) + nz * (z - pz)) / (real.sqrt (nx * nx + ny * ny + nz * nz))

-- Prove that there exists a point on a line g at distance d from the plane S.
theorem exists_point_on_line_at_distance_from_plane (g : Line) (S : Plane) (d : ℝ)
  (h_nonzero : d ≠ 0) (h_not_intersect : ∀ pt ∈ g, dist_point_to_plane pt S ≠ 0) :
  ∃ (P : ℝ × ℝ × ℝ), P ∈ g ∧ dist_point_to_plane P S = d :=
  sorry

end exists_point_on_line_at_distance_from_plane_l649_649789


namespace number_of_positive_terms_in_sequence_l649_649760

noncomputable def a (n : ℕ) : ℝ := Real.cos (10^n).toReal

theorem number_of_positive_terms_in_sequence : 
  (Finset.univ.filter (λ n : Fin 100, 0 < a (n + 1))).card = 99 := by
  sorry

end number_of_positive_terms_in_sequence_l649_649760


namespace cos_450_eq_zero_l649_649736

theorem cos_450_eq_zero :
  ∀ (θ : ℝ), θ = 450 ∧ (θ mod 360 = 90) ∧ (cos 90 = 0) → cos θ = 0 :=
by
  intros θ hθ
  cases hθ with h1 hand
  cases hand with h2 h3
  rw h1
  rw h2
  exact h3

end cos_450_eq_zero_l649_649736


namespace simplify_sqrt7_pow6_l649_649587

theorem simplify_sqrt7_pow6 :
  (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt7_pow6_l649_649587


namespace cost_price_is_3000_l649_649616

variable (CP SP : ℝ)

-- Condition: selling price (SP) is 20% more than the cost price (CP)
def sellingPrice : ℝ := CP + 0.20 * CP

-- Condition: selling price (SP) is Rs. 3600
axiom selling_price_eq : SP = 3600

-- Given the above conditions, prove that the cost price (CP) is Rs. 3000
theorem cost_price_is_3000 (h : sellingPrice CP = SP) : CP = 3000 := by
  sorry

end cost_price_is_3000_l649_649616


namespace measure_angle_AQP_l649_649400

-- Problem Definitions
variables {ABC : Type*} [equilateral_triangle ABC]
variables {P Q : point ABC}
variables (inside_P : inside_triangle P ABC) (on_BC_Q : on_segment Q (side BC ABC))
variables (PB_PQ_QC : PB = PQ = QC) (angle_PBC_20 : ∠PBC = 20)

-- The Goal
theorem measure_angle_AQP : ∠AQP = 60 :=
by sorry

end measure_angle_AQP_l649_649400


namespace percentage_of_men_not_speaking_french_or_spanish_l649_649150

theorem percentage_of_men_not_speaking_french_or_spanish 
  (total_employees : ℕ) 
  (men_percent women_percent : ℝ)
  (men_french percent men_spanish_percent men_other_percent : ℝ)
  (women_french_percent women_spanish_percent women_other_percent : ℝ)
  (h1 : men_percent = 60)
  (h2 : women_percent = 40)
  (h3 : men_french_percent = 55)
  (h4 : men_spanish_percent = 35)
  (h5 : men_other_percent = 10)
  (h6 : women_french_percent = 45)
  (h7 : women_spanish_percent = 25)
  (h8 : women_other_percent = 30) :
  men_other_percent = 10 := 
by
  sorry

end percentage_of_men_not_speaking_french_or_spanish_l649_649150


namespace asymptotes_when_lambda_is_one_third_eccentricity_range_l649_649443

-- Definitions of hyperbola and conditions
variable (a b λ : ℝ)
variable (Ha : 0 < a) (Hb : 0 < b)
variable (Hλ : λ ∈ set.Icc (1/9) (1/2))

-- The equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Asymptotes when λ = 1/3
theorem asymptotes_when_lambda_is_one_third : 
  λ = 1/3 → ∀ x y : ℝ, y = x ∨ y = -x :=
by sorry

-- Eccentricity range based on λ
theorem eccentricity_range : 
  ∀ e : ℝ, (1/9 ≤ λ ∧ λ ≤ 1/2) → (sqrt 5 / 2 ≤ e ∧ e ≤ sqrt 3) :=
by sorry

end asymptotes_when_lambda_is_one_third_eccentricity_range_l649_649443


namespace division_approx_eq_l649_649308

def numerator := 0.625 * 0.0729 * 28.9
def denominator := 0.0017 * 0.025 * 8.1
def result := 3826

theorem division_approx_eq :
  numerator / denominator ≈ result :=
sorry

end division_approx_eq_l649_649308


namespace infinite_series_sum_l649_649934

theorem infinite_series_sum (x : ℝ) (h : x > 1) :
  ∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (-(3 ^ n))) = 1 / (x - 1) :=
sorry

end infinite_series_sum_l649_649934


namespace obtuse_triangle_circumcircle_incircle_ratio_l649_649530

variables {A B C : Type} [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C]

-- Assuming a, b, c are side lengths opposite to angles A, B, C
variables (a b c : Real)
-- Ratio of the sums of the sides are 5:6:7
variables (r1 r2 r3 : Real)
-- Assuming R is the radius of the circumcircle and r is the radius of the incircle
variables (R r : Real)

def triangle_sides (a : Real) (b : Real) (c : Real) : Prop :=
  (a + b) / (b + c) = 5 / 6 ∧
  (b + c) / (c + a) = 6 / 7 ∧
  (c + a) / (a + b) = 7 / 5

theorem obtuse_triangle
  (h1 : triangle_sides a b c) :
  (exists C : Real, ∠A + ∠B + ∠C = 180 ∧ ∠C > 90) :=
sorry

theorem circumcircle_incircle_ratio
  (h2 : triangle_sides a b c) (h3 : R = (c / 2 * real.sin C)) (h4 : r = ((a + b + c) / 2) / (a * b * real.sin C)) :
  R / r = 16 / 5 :=
sorry

end obtuse_triangle_circumcircle_incircle_ratio_l649_649530


namespace calculate_expression_l649_649723

variables (a b : ℝ) -- declaring variables a and b to be real numbers

theorem calculate_expression :
  (-a * b^2) ^ 3 + (a * b^2) * (a * b) ^ 2 * (-2 * b) ^ 2 = 3 * a^3 * b^6 :=
by
  sorry

end calculate_expression_l649_649723


namespace no_valid_m_l649_649812

theorem no_valid_m
  (m : ℕ)
  (hm : m > 0)
  (h1 : ∃ k1 : ℕ, k1 > 0 ∧ 1806 = k1 * (m^2 - 2))
  (h2 : ∃ k2 : ℕ, k2 > 0 ∧ 1806 = k2 * (m^2 + 2)) :
  false :=
sorry

end no_valid_m_l649_649812


namespace sum_first_11_terms_eq_44_l649_649081

-- Define the arithmetic sequence and its properties
noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def sum_of_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

-- Given conditions
axiom a8_eq : ∃ (a d : ℝ), 2 * (arithmetic_sequence a d 8) + (arithmetic_sequence a d 2) = 12

-- Prove that the sum of the first 11 terms is 44
theorem sum_first_11_terms_eq_44 : ∃ (a d : ℝ), sum_of_first_n_terms a d 11 = 44 :=
by
-- Introduce the known condition
cases a8_eq with a hd
exists a, hd
-- Provide a placeholder for the proof
sorry

end sum_first_11_terms_eq_44_l649_649081


namespace value_of_nabla_expression_l649_649026

namespace MathProblem

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem value_of_nabla_expression : nabla (nabla 2 3) 2 = 4099 :=
by
  sorry

end MathProblem

end value_of_nabla_expression_l649_649026


namespace midpoint_locus_is_square_point_Z_locus_is_rectangle_l649_649107

-- Define the cube and points
variables {E : Type*} [euclidean_space E]
variables {A B C D A' B' C' D' : E}
variables {X Y P : E} -- X on AC, Y on B'D'

-- Part a: Locus of midpoints
theorem midpoint_locus_is_square
  (cube : cube ABCDA'B'C'D')
  (X_on_AC : X ∈ segment AC)
  (Y_on_B'D' : Y ∈ segment B'D') :
  ∃ square : quad E, locus_of_midpoints_is_square X_on_AC Y_on_B'D' square :=
sorry

-- Part b: Locus of points Z
variables {Z : E}

theorem point_Z_locus_is_rectangle
  (cube : cube ABCDA'B'C'D')
  (X_on_AC : X ∈ segment AC)
  (Y_on_B'D' : Y ∈ segment B'D')
  (Z_condition : distance Z Y = 2 * distance X Z) :
  ∃ rectangle : rect E, locus_of_points_Z_is_rectangle X_on_AC Y_on_B'D' Z_condition rectangle :=
sorry

end midpoint_locus_is_square_point_Z_locus_is_rectangle_l649_649107


namespace eccentricity_of_hyperbola_l649_649425

noncomputable def hyperbola (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def foci_condition (a b : ℝ) (c : ℝ) : Prop :=
  c = Real.sqrt (a^2 + b^2)

noncomputable def trisection_condition (a b c : ℝ) : Prop :=
  2 * c = 6 * a^2 / c

theorem eccentricity_of_hyperbola (a b c e : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hc : c = Real.sqrt (a^2 + b^2)) (ht : 2 * c = 6 * a^2 / c) :
  e = Real.sqrt 3 :=
by
  apply sorry

end eccentricity_of_hyperbola_l649_649425


namespace problem_statement_l649_649880

def is_even (n : ℤ) : Prop := n % 2 = 0
def is_odd (n : ℤ) : Prop := n % 2 ≠ 0

def z_star (y : ℤ) : ℤ := Inf {n : ℤ | is_even n ∧ n ≤ y}
def w_star (x : ℤ) : ℤ := Inf {n : ℤ | is_odd n ∧ n > x}

theorem problem_statement :
  let x : ℤ := 13 in
  let y : ℤ := 12 in
  let val : ℤ := (632 - z_star y) * (w_star x - x)
  in val = -994 :=
by {
  -- proof goes here
  sorry
}

end problem_statement_l649_649880


namespace part_i_part_ii_l649_649108

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then log x / log 2 else log (-x) / log (1 / 2)

theorem part_i : f (f (-1 / 4)) = 1 := by
  sorry

theorem part_ii {a : ℝ} (ha : f(a) > f(-a)) : a ∈ Ioo (-1:ℝ) 0 ∪ Ioi 1 := by
  sorry

end part_i_part_ii_l649_649108


namespace digit7_count_from_20_to_200_l649_649531

theorem digit7_count_from_20_to_200 : 
  let count_units := λ (n : Nat) => if n % 10 == 7 then 1 else 0
  let count_tens := λ (n : Nat) => if (n / 10) % 10 == 7 then 1 else 0
  let count_sevens := λ (n : Nat) => count_units n + count_tens n
  (List.range' 20 (200 - 20 + 1)).sum_by count_sevens = 38 :=
by
  sorry

end digit7_count_from_20_to_200_l649_649531


namespace part1_part3_l649_649850

def f (x m : ℝ) : ℝ := Real.exp x + m * Real.exp (-x) + (m - 1) * x

theorem part1 (x : ℝ) : (f x 0) ≥ 1 := 
by
  sorry

-- Monotonicity of f is a discussion and not exact proof, hence skipping lemma for part 2.

theorem part3 (x m : ℝ) (h : -Real.exp 1 ≤ m) (h₁ : m ≤ -1) (hx : 0 < x) : f x m ≥ -2 :=
by
  sorry

end part1_part3_l649_649850


namespace polynomial_result_l649_649435

noncomputable def polynomial_coeff_sum := 
  let a0 := (2 - 1)^5;
  let a1 := 5 * (2)^4 * -1;
  let a2 := 10 * (2)^3 * (-1)^2;
  let a3 := 10 * (2)^2  * (-1)^3;
  let a4 := 5 * (2)^1  * (-1)^4;
  let a5 := (-1)^5;
  (a0, a1, a2, a3, a4, a5)

theorem polynomial_result :
  let (a0, a1, a2, a3, a4, a5) := polynomial_coeff_sum in
  (a0 + a1 + a2 + a3 + a4 + a5 = 1) ∧
  (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| = 243) ∧
  (a1 + a3 + a5 = 122) ∧
  ((a0 + a2 + a4)^2 - (a1 + a3 + a5)^2 = -243) :=
begin
  sorry
end

end polynomial_result_l649_649435


namespace prize_winner_l649_649513

def Student := Type
def A : Student := sorry -- define A
def B : Student := sorry -- define B
def C : Student := sorry -- define C

noncomputable def Statement := Student → Prop
def Said (student : Student) (statement : Statement) : Prop := sorry -- placeholder for the actual logic

def A_statement (s : Student) : Prop := s ≠ A
def B_statement (s : Student) : Prop := s = B
def C_statement (s : Student) : Prop := s ≠ B

def exactly_two_false (conditions : List Prop) : Prop :=
  (conditions.filter (λ cond, cond = false)).length = 2

theorem prize_winner :
  (Said A (A_statement A) ∧ Said B (B_statement B) ∧ Said C (C_statement B) ∧ exactly_two_false ([A_statement A, B_statement B, C_statement B])) →
  won_prize A :=
sorry

end prize_winner_l649_649513


namespace plane_centroid_l649_649925

variable (α β γ : ℝ)

noncomputable def p : ℝ := α / 3
noncomputable def q : ℝ := β / 3
noncomputable def r : ℝ := γ / 3

theorem plane_centroid :
  (1 / α ^ 2 + 1 / β ^ 2 + 1 / γ ^ 2 = 1 / 4) →
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2 = 2.25) :=
by
  sorry

end plane_centroid_l649_649925


namespace number_of_subway_tickets_is_6_l649_649998

-- Define the problem setting
def has_18_bus_tickets : ℕ := 18
def max_envelopes : ℕ := 6

-- Given assumptions
axiom bus_tickets_divisible : has_18_bus_tickets % max_envelopes = 0
axiom subway_tickets (S : ℕ) : S % max_envelopes = 0

-- The statement asserting the number of subway tickets is 6
theorem number_of_subway_tickets_is_6 : ∃ (S : ℕ), S % max_envelopes = 0 ∧ S = 6 :=
by
  existsi 6
  split
  · exact subway_tickets 6
  · rfl

end number_of_subway_tickets_is_6_l649_649998


namespace num_positive_terms_l649_649751

def sequence_cos (n : ℕ) : ℝ :=
  cos (10^(n-1) * π / 180)

theorem num_positive_terms : (finset.filter (λ n : ℕ, 0 < sequence_cos n) (finset.range 100)).card = 99 :=
by
  sorry

end num_positive_terms_l649_649751


namespace Cantor_primitive_repr_l649_649422

noncomputable def cantor_set : Set ℕ :=
  { n | ∀ d ∈ n.digits 3, d ≠ 1 }

theorem Cantor_primitive_repr (A : Set ℤ) (a b : ℤ) (h : a ≠ 0) :
  (∀ C : Set ℕ, C = cantor_set →
    (∃ As : List (Set ℕ), (∀ i, As[i] = a • C + b) ∧
      (∀ i, ∃ a_i b_i : ℤ, As[i] = { x | x = a_i * x + b_i } ∧ a_i > 1 ∧
        (∃ k : ℕ, a_i = 3^k) ∧ a_i > b_i) ∧
      Multiset.ofList As = { C, 3 * C, 3 * C + 2 }) ∧
    (∀ R : List (Set ℕ), R = {C, 9 * C + 6, 3 * C + 2}) ∧
    (∀ (a_i b_i : ℤ), a_i > 1 ∧ (∃ k : ℕ, a_i = 3^k) ∧ a_i > b_i)) :=
sorry

end Cantor_primitive_repr_l649_649422


namespace part1_part2_l649_649846

def z1 (x λ : ℝ) := (Real.sin x) + (Complex.i * λ)
def z2 (x : ℝ) := (Real.sin x + Real.sqrt 3 * Real.cos x) - Complex.i

theorem part1 (x λ: ℝ) (h1 : 2 * z1 x λ = Complex.i * z2 x)
  (h2 : 0 < x ∧ x < Real.pi / 2) : 
  x = Real.pi / 6 ∧ λ = 1 := 
by 
  sorry 

def f (x : ℝ) := (Real.sin x) ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem part2 : 
  (∀ x : ℝ, f(x + Real.pi) = f x) ∧ 
  (∀ k : ℤ, ∀ x : ℝ, k * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 6) → 
  (∀ k : ℤ, [k * Real.pi + Real.pi / 3, k * Real.pi + 5 * Real.pi / 6]) :=
by
  sorry

end part1_part2_l649_649846


namespace domain_h_l649_649710

def h (t : ℝ) : ℝ := 130 * t - 5 * t ^ 2

theorem domain_h : ∀ t, (0 ≤ t ∧ t ≤ 26) ↔ (t ∈ set.Icc 0 26) :=
by
  sorry

end domain_h_l649_649710


namespace book_cost_l649_649221

variable (cost_per_book : ℝ) (students : ℕ) (school_money pocket_money total_money : ℝ)

-- Given conditions
def Sally_given_money := school_money = 320
def students_in_class := students = 30
def Sally_pocket_money := pocket_money = 40
def total_money_available := total_money = school_money + pocket_money

-- Prove that: the cost of each reading book is $12
theorem book_cost (h1: Sally_given_money) (h2: students_in_class) (h3: Sally_pocket_money) (h4: total_money_available) (h5: 30 * cost_per_book = total_money): cost_per_book = 12 := by
  sorry

end book_cost_l649_649221


namespace sixth_power_of_sqrt_l649_649266

variable (x : ℝ)
axiom h1 : x = Real.sqrt (2 + Real.sqrt 2)

theorem sixth_power_of_sqrt : x^6 = 16 + 10 * Real.sqrt 2 :=
by {
    sorry
}

end sixth_power_of_sqrt_l649_649266


namespace num_positive_cos_terms_l649_649757

def sequence (n : ℕ) : ℝ := Real.cos (10^(n-1) * Real.pi / 180)

theorem num_positive_cos_terms : (Finset.card (Finset.filter (λ n, 0 < sequence n) (Finset.range 100))) = 99 := 
sorry

end num_positive_cos_terms_l649_649757


namespace positive_terms_count_l649_649778

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then real.cos (1 * real.pi / 180) else real.cos (nat.pow 10 (n - 1) * real.pi / 180)

theorem positive_terms_count : finset.card (finset.filter (λ n, 0 < sequence n) (finset.range 100)) = 99 :=
by sorry

end positive_terms_count_l649_649778


namespace total_votes_l649_649331

theorem total_votes (V : ℝ) (C R : ℝ) 
  (hC : C = 0.10 * V)
  (hR1 : R = 0.10 * V + 16000)
  (hR2 : R = 0.90 * V) :
  V = 20000 :=
by
  sorry

end total_votes_l649_649331


namespace distribution_schemes_l649_649033

def students : Type := {student // student = "male" ∨ student = "female"}
def male_students := fin 3
def female_students := fin 2

def num_classes := 2

def has_at_least_two_students (class : set students) : Prop := class.size ≥ 2

def children_satisfied (classA : set students) (classB : set students) :=
  classA ∪ classB = {x | ∃ h : students, true} ∧
  has_at_least_two_students classA ∧
  has_at_least_two_students classB ∧
  ∃ female : students, female ∈ classA

theorem distribution_schemes :
  ∃ (classA classB : set students), children_satisfied classA classB → 
  (∃! (n : ℕ), n = 16) := sorry

end distribution_schemes_l649_649033


namespace number_of_true_propositions_l649_649615

theorem number_of_true_propositions (a b c : ℝ) (h : ¬(a > b → ac^2 > bc^2)) : 
  (let converse := (∀ a b c : ℝ, ac^2 > bc^2 → a > b),
       inverse := (∀ a b c : ℝ, a ≤ b → ac^2 ≤ bc^2),
       contrapositive := (∀ a b c : ℝ, ac^2 ≤ bc^2 → a ≤ b) in
  nat_of_bool (converse a b c) + nat_of_bool (inverse a b c) + nat_of_bool (contrapositive a b c) = 2) :=
by sorry

end number_of_true_propositions_l649_649615


namespace carrots_thrown_out_l649_649208

def initial_carrots := 19
def additional_carrots := 46
def total_current_carrots := 61

def total_picked := initial_carrots + additional_carrots

theorem carrots_thrown_out : total_picked - total_current_carrots = 4 := by
  sorry

end carrots_thrown_out_l649_649208


namespace compute_expression_l649_649385

theorem compute_expression : 2 + 4 * 3^2 - 1 + 7 * 2 / 2 = 44 := by
  sorry

end compute_expression_l649_649385


namespace five_fold_application_l649_649977

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - 3*x + 2 else x + 10

theorem five_fold_application :
  f (f (f (f (f 2)))) = -8 := by
  sorry

end five_fold_application_l649_649977


namespace coordinate_axes_points_correct_l649_649909

-- Definitions of conditions
def x_axis_points : set (ℝ × ℝ) := { p | p.snd = 0 }
def y_axis_points : set (ℝ × ℝ) := { p | p.fst = 0 }

-- Definition of the actual problem to prove
def coordinate_axes_points : set (ℝ × ℝ) := { p | p.fst * p.snd = 0 }

-- Theorem statement to be proved
theorem coordinate_axes_points_correct :
  coordinate_axes_points = x_axis_points ∪ y_axis_points :=
sorry

end coordinate_axes_points_correct_l649_649909


namespace extra_games_needed_l649_649362

def initial_games : ℕ := 500
def initial_success_rate : ℚ := 0.49
def target_success_rate : ℚ := 0.5

theorem extra_games_needed :
  ∀ (x : ℕ),
  (245 + x) / (initial_games + x) = target_success_rate → x = 10 := 
by
  sorry

end extra_games_needed_l649_649362


namespace inequality_problem_l649_649064

theorem inequality_problem (a b : ℝ) (h₁ : 1/a < 1/b) (h₂ : 1/b < 0) :
  (∃ (p q : Prop), 
    (p ∧ q) ∧ 
    ((p ↔ (a + b < a * b)) ∧ 
    (¬q ↔ |a| ≤ |b|) ∧ 
    (¬q ↔ a > b) ∧ 
    (q ↔ (b / a + a / b > 2)))) :=
sorry

end inequality_problem_l649_649064


namespace problem_l649_649046

-- Definitions of the conditions:
def eight_digit_palindrome (n : ℕ) : Prop := 
  n / 10^7 % 10 = n % 10 ∧ 
  n / 10^6 % 10 = n / 10 % 10 ∧ 
  n / 10^5 % 10 = n / 10^2 % 10 ∧ 
  n / 10^4 % 10 = n / 10^3 % 10

def composed_of_digits_0_and_1 (n : ℕ) : Prop :=
  n / 10^7 % 10 ∈ {0, 1} ∧
  n / 10^6 % 10 ∈ {0, 1} ∧
  n / 10^5 % 10 ∈ {0, 1} ∧
  n / 10^4 % 10 ∈ {0, 1} ∧
  n / 10^3 % 10 ∈ {0, 1} ∧
  n / 10^2 % 10 ∈ {0, 1} ∧
  n / 10^1 % 10 ∈ {0, 1} ∧
  n % 10 ∈ {0, 1}

def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def prime_divisors_use_only_1_3_percent (n : ℕ) : Prop :=
  ∀ p, prime p → p ∣ n → (p = 1 ∨ p = 3)

-- Main theorem statement
theorem problem :
  eight_digit_palindrome 10111101 ∧
  composed_of_digits_0_and_1 10111101 ∧
  divisible_by_3 10111101 ∧
  prime_divisors_use_only_1_3_percent 10111101 :=
sorry

end problem_l649_649046


namespace tom_paid_amount_correct_l649_649668

def kg (n : Nat) : Nat := n -- Just a type alias clarification

theorem tom_paid_amount_correct :
  ∀ (quantity_apples : Nat) (rate_apples : Nat) (quantity_mangoes : Nat) (rate_mangoes : Nat),
  quantity_apples = kg 8 →
  rate_apples = 70 →
  quantity_mangoes = kg 9 →
  rate_mangoes = 55 →
  (quantity_apples * rate_apples) + (quantity_mangoes * rate_mangoes) = 1055 :=
by
  intros quantity_apples rate_apples quantity_mangoes rate_mangoes
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end tom_paid_amount_correct_l649_649668


namespace product_area_perimeter_eq_104sqrt26_l649_649988

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2).sqrt

noncomputable def side_length := distance (5, 5) (0, 4)

noncomputable def area_of_square := side_length ^ 2

noncomputable def perimeter_of_square := 4 * side_length

noncomputable def product_area_perimeter := area_of_square * perimeter_of_square

theorem product_area_perimeter_eq_104sqrt26 :
  product_area_perimeter = 104 * Real.sqrt 26 :=
by 
  -- placeholder for the proof
  sorry

end product_area_perimeter_eq_104sqrt26_l649_649988


namespace older_brother_age_is_25_l649_649655

noncomputable def age_of_older_brother (father_age current_n : ℕ) (younger_brother_age : ℕ) : ℕ := 
  (father_age - current_n) / 2

theorem older_brother_age_is_25 
  (father_age : ℕ) 
  (h1 : father_age = 50) 
  (younger_brother_age : ℕ)
  (current_n : ℕ) 
  (h2 : (2 * (younger_brother_age + current_n)) = father_age + current_n) : 
  age_of_older_brother father_age current_n younger_brother_age = 25 := 
by
  sorry

end older_brother_age_is_25_l649_649655


namespace officer_permutations_count_l649_649086

theorem officer_permutations_count : 
  let members := ['Alice', 'Bob', 'Carol', 'Dave']
  let positions := ['president', 'vice_president', 'secretary', 'treasurer']
  (∀ (no_duplicate : members.nodup) (positions_no_duplicate : positions.nodup), perms members).length = 24 :=
begin
  sorry
end

end officer_permutations_count_l649_649086


namespace largest_possible_n_l649_649825

theorem largest_possible_n (n : ℕ) (hn : n > 2)
  (hnum : ∀ a b : ℕ, a ≠ b → a + b ∈ set_of is_prime)
  (hneq : ∀ a b c : ℕ, a = b → b = c → a ≠ c) : n ≤ 3 :=
sorry

end largest_possible_n_l649_649825


namespace trigonometric_identity_l649_649063

theorem trigonometric_identity 
  (x : ℝ)
  (h1 : cos (x - π / 4) = -1 / 3) 
  (h2 : 5 * π / 4 < x ∧ x < 7 * π / 4) : 
  sin (2 * x) - cos (2 * x) = (4 * real.sqrt 2 - 7) / 9 :=
by 
  sorry

end trigonometric_identity_l649_649063


namespace real_solution_range_of_inequality_l649_649141

theorem real_solution_range_of_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4a) ↔ (a < 1 ∨ 3 < a) :=
by
sorry

end real_solution_range_of_inequality_l649_649141


namespace triangles_area_sum_l649_649003

noncomputable def trapezoid_area: ℝ := 52
noncomputable def triangle_AOB_area: ℝ := 17
def DE_eq_FC : Prop := ∃ (DE FC : ℝ), DE = FC
def intersects_at_O : Prop := ∃ (O : Point), ∃ (AF BE : Line), intersect AF BE O

theorem triangles_area_sum
    (h1 : trapezoid_area = 52)
    (h2 : DE_eq_FC)
    (h3 : intersects_at_O)
    (h4 : triangle_AOB_area = 17) :
    ∃ AOE_area BOF_area, AOE_area + BOF_area = 18 := 
sorry

end triangles_area_sum_l649_649003


namespace determine_num_chickens_l649_649291

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def house_cost : ℕ := 120000
def num_cows : ℕ := 20
def cow_cost_per_cow : ℕ := 1000
def install_hours : ℕ := 6
def install_cost_per_hour : ℕ := 100
def equipment_cost : ℕ := 6000
def total_expenses : ℕ := 147700
def chicken_cost_per_chicken : ℕ := 5

def total_cost_before_chickens : ℕ := 
  (land_acres * land_cost_per_acre) + 
  house_cost + 
  (num_cows * cow_cost_per_cow) + 
  (install_hours * install_cost_per_hour) + 
  equipment_cost

def chickens_cost : ℕ := total_expenses - total_cost_before_chickens

def num_chickens : ℕ := chickens_cost / chicken_cost_per_chicken

theorem determine_num_chickens : num_chickens = 100 := by
  sorry

end determine_num_chickens_l649_649291


namespace maximum_triangles_formed_l649_649300

theorem maximum_triangles_formed (n : ℕ) (h : n = 100) (no_triangles : ∀ (s : ℕ) (hs : s < n), 
                                ¬ ∃ a b c : ℕ, a < b ∧ b < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  let new_n := n + 1 in n = 100 := by
  sorry

end maximum_triangles_formed_l649_649300


namespace scaling_matrix_unique_l649_649410

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

noncomputable def matrix_N : Matrix (Fin 4) (Fin 4) ℝ := ![![3, 0, 0, 0], ![0, 3, 0, 0], ![0, 0, 3, 0], ![0, 0, 0, 3]]

theorem scaling_matrix_unique (N : Matrix (Fin 4) (Fin 4) ℝ) :
  (∀ (w : Fin 4 → ℝ), N.mulVec w = 3 • w) → N = matrix_N :=
by
  intros h
  sorry

end scaling_matrix_unique_l649_649410


namespace ratio_Sarah_to_Eli_is_2_l649_649174

variable (Kaylin_age : ℕ := 33)
variable (Freyja_age : ℕ := 10)
variable (Eli_age : ℕ := Freyja_age + 9)
variable (Sarah_age : ℕ := Kaylin_age + 5)

theorem ratio_Sarah_to_Eli_is_2 : (Sarah_age : ℚ) / Eli_age = 2 := 
by 
  -- Proof would go here
  sorry

end ratio_Sarah_to_Eli_is_2_l649_649174


namespace sqrt_seven_to_six_power_eq_343_l649_649586

theorem sqrt_seven_to_six_power_eq_343 : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_six_power_eq_343_l649_649586


namespace reflection_point_on_circumcircle_l649_649829

variables {A B C P Q R : Type}

-- Assume the existence of functions that describe the conditions before (definitions)
axiom triangle_has_point (ABC : Triangle) (P : Point) : OnSide P BC
axiom perp_bisector_intersects_AC_at_Q (P C Q A : Point) : PerpBisector (Segment P C) AC Q
axiom perp_bisector_intersects_AB_at_R (P B R A : Point) : PerpBisector (Segment P B) AB R
axiom reflection_over_line (P Q R P' : Point) : ReflectionOverLine P Q R P'

-- Assume \(\triangle ABC\) has a circumcircle
axiom circumscribed_circle (ABC : Triangle) : Circumcircle ABC

-- Statement of the proof problem in Lean 4
theorem reflection_point_on_circumcircle
  (ABC : Triangle) (P Q R : Point)
  (h1 : OnSide P BC)
  (h2 : PerpBisector (Segment P C) AC Q)
  (h3 : PerpBisector (Segment P B) AB R)
  (h4 : ReflectionOverLine P Q R P') :
  OnCircle P' (Circumcircle ABC) :=
sorry

end reflection_point_on_circumcircle_l649_649829


namespace vector_norm_inequality_l649_649834

-- Define the vectors and their properties:
variables {V : Type*} [inner_product_space ℝ V]

-- Vectors a, b, c, d with their sum equaling zero
variables (a b c d : V)
hypothesis (h : a + b + c + d = 0)

-- Prove the inequality involving the norms of these vectors
theorem vector_norm_inequality (h : a + b + c + d = 0) :
  ∥a∥ + ∥b∥ + ∥c∥ + ∥d∥ ≥ ∥a + d∥ + ∥b + d∥ + ∥c + d∥ :=
sorry

end vector_norm_inequality_l649_649834


namespace infinite_sum_identity_l649_649944

theorem infinite_sum_identity (x : ℝ) (h : x > 1) :
  (∑' n : ℕ, 1 / (x^(3^n) - x^(-3^n))) = 1 / (x - 1) :=
sorry

end infinite_sum_identity_l649_649944


namespace steve_lingonberries_picking_l649_649595

theorem steve_lingonberries_picking :
  ∀ (pay_per_pound : ℝ) (picks_monday picks_thursday picks_tuesday : ℝ)
    (goal_amount : ℝ),
  pay_per_pound = 2 →
  picks_monday = 8 →
  picks_thursday = 18 →
  goal_amount = 100 →
  (goal_amount / pay_per_pound) - (picks_monday + picks_thursday) = picks_tuesday →
  (picks_tuesday / picks_monday) = 3 := 
by
  intros pay_per_pound picks_monday picks_thursday picks_tuesday goal_amount
  assume h1 h2 h3 h4 h5
  sorry

end steve_lingonberries_picking_l649_649595


namespace ratio_areas_of_circumscribed_circles_l649_649353

theorem ratio_areas_of_circumscribed_circles (P : ℝ) (A B : ℝ)
  (h1 : ∃ (x : ℝ), P = 8 * x)
  (h2 : ∃ (s : ℝ), s = P / 3)
  (hA : A = (5 * (P^2) * Real.pi) / 128)
  (hB : B = (P^2 * Real.pi) / 27) :
  A / B = 135 / 128 := by
  sorry

end ratio_areas_of_circumscribed_circles_l649_649353


namespace complement_intersection_l649_649866

universe u

variable (U M N : Set ℕ)

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 5}

theorem complement_intersection :
  ((U \ M) ∩ N) = {3, 5} := by
  sorry

end complement_intersection_l649_649866


namespace O_moves_straight_l649_649519

-- Setting up the problem with conditions
variables (O : Type) [MetricSpace O] [NormedSpace ℝ O]
variables {A B G A' B' O' : O}
variables (circle : O → ℝ → Set O) (diameter_AB : Segment ℝ O)
variables (G_moving : G ∈ diameter_AB)
variables (A_perp : ∃ h : ℝ, perpendicular (A - (G : ℝ))) -- Line AA'
variables (BG_len : ∥A' - A∥ = ∥B - G∥) -- Length condition AA' = AG
variables (B_perp : ∃ k : ℝ, perpendicular (B - (G : ℝ))) -- Line BB'
variables (AB_len : ∥B' - B∥ = ∥A' - G∥) -- Length condition BB' = BG

-- Setting up the coordinates
variables (A B G : ℝ) (A' := (G, G + radius)) (B' := (G, radius - G))

-- Theorem statement for the movement of O'
theorem O_moves_straight (G_movement : ∀ G ∈ Interval_closed (-radius) radius,
                         ((O' = midpoint A' B') → 
                         ( O' ∈ set.xy {y | y = radius}
)
) : true := 
sorry

end O_moves_straight_l649_649519


namespace general_term_compare_terms_l649_649115

open Real

noncomputable def sequence_a (t : ℝ) (a : ℕ → ℝ) : Prop :=
  a 1 = 2 * t - 3 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = (2 * t^(n + 1) - 3) * a n + 2 * (t - 1) * t^n - 1) / (a n + 2 * t^n - 1))

-- Recursive relation
theorem general_term (t : ℝ) (h1 : t ≠ 1) (h2 : t ≠ -1) (h : t ≠ 0) : ∀ n, 0 < n → sequence_a t a → a n = (2 * (t^n - 1) / n) - 1 := by
  sorry

-- Comparison of a_{n+1} and a_n
theorem compare_terms (t : ℝ) (h1 : 0 < t) : ∀ n, 0 < n → sequence_a t a → a (n + 1) > a n := by
  sorry
  
end general_term_compare_terms_l649_649115


namespace trig_identity_l649_649992

theorem trig_identity (α : ℝ) (h1 : α ≠ π/2) (h2 : α ≠ 0) (h3 : α ≠ π) (h4 : α ≠ 3 * π / 2):
  (cos α * csc α - sin α * sec α) / (cos α + sin α) = csc α - sec α :=
by
  sorry

end trig_identity_l649_649992


namespace a_range_l649_649835

variable (x1 x2 m a : ℝ)

-- Proposition P: definition of roots and given condition
def P : Prop :=
  ∃ (x1 x2 : ℝ), (x1 ∈ set_of (λ x, x^2 - m * x - 2 = 0)) ∧
                 (x2 ∈ set_of (λ x, x^2 - m * x - 2 = 0)) ∧
                 (a^2 - 5 * a - 3 < abs (x1 - x2))

-- Proposition Q: quadratic inequality condition
def Q : Prop :=
  ∀ (x : ℝ), x^2 + 2 * (real.sqrt 2) * a * x + 11 * a ≤ 0 → 
             x ∈ {y | y = (0) ∨ y = (11/2)}

-- Conditions that P is false and Q is true
axiom P_false : ¬P
axiom Q_true : Q

-- Required proof problem statement
theorem a_range : a = 0 ∨ a = 11 / 2 := sorry

end a_range_l649_649835


namespace question1_question2_l649_649075

variable (a : ℕ → ℕ) (S : ℕ → ℕ)
variable (b : ℕ → ℕ) (T : ℕ → ℕ)

noncomputable def isGeoSeq (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n

-- Conditions
axiom cond1 : ∀ n : ℕ, S n = (finset.range n).sum a  -- Sum of first n terms
axiom cond2 : ∀ n : ℕ, 2 * a n = S n + 2  -- Arithmetic sequence condition

-- Questions
theorem question1 : isGeoSeq a :=
sorry

noncomputable def log2 : ℕ → ℕ
| 0 := 0
| (n + 1) := 1 + log2 (n / 2)  -- Define log base 2 function

noncomputable def b (n : ℕ) : ℕ := a n + log2 (1 / a n)
noncomputable def T (n : ℕ) : ℕ := (finset.range n).sum b

-- Correct answer
theorem question2 : ∀ n : ℕ, T n = 2 ^ (n + 1) - 2 - n * (n + 1) / 2 :=
sorry

end question1_question2_l649_649075


namespace find_day_for_students_secret_spreading_day_l649_649985

noncomputable def total_students (n : ℕ) : ℤ :=
  (3^(n+1) - 1) / 2

theorem find_day_for_students (n : ℕ) :
  (total_students n >= 2186) → (n = 7) :=
by
  intros h,
  have calculation : total_students 7 = 2186 := by norm_num,
  have check_n: n = 7,
  sorry

theorem secret_spreading_day : 
  sorry

# You can check here if logic holds, and complete proof based on it.
example : find_day_for_students 7 := by
  unfold total_students
  norm_num

end find_day_for_students_secret_spreading_day_l649_649985


namespace binomial_coefficients_sum_l649_649434

theorem binomial_coefficients_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ),
  (1 - 2 * 0)^5 = a_0 + a_1 * (1 + 0) + a_2 * (1 + 0)^2 + a_3 * (1 + 0)^3 + a_4 * (1 + 0)^4 + a_5 * (1 + 0)^5 →
  (1 - 2 * 1)^5 = (-1)^5 * a_5 →
  a_0 + a_1 + a_2 + a_3 + a_4 = 33 :=
by sorry

end binomial_coefficients_sum_l649_649434


namespace ratio_of_square_areas_l649_649018

theorem ratio_of_square_areas (d s : ℝ)
  (h1 : d^2 = 2 * s^2) :
  (d^2) / (s^2) = 2 :=
by
  sorry

end ratio_of_square_areas_l649_649018


namespace impossible_arrangement_l649_649164

/- Definitions based on the problem conditions -/
def grid_100x100 : Type := (Fin 100) × (Fin 100)
def digit : Type := Fin 3 -- Representing digits 0, 1, and 2

-- Definition of a function representing the grid arrangement
def arrangement (g : grid_100x100 → digit) : Prop :=
  ∀ (i j : Fin 98), let counts := (List.filter (λ x, x = 0) (List.map g (univ.filter (λ ⟨x, y⟩, x ∈ (finRange 3) ∧ y ∈ (finRange 4))))) in
  let counts0 := List.length (counts.filter (λ x, x = 0)) in
  let counts1 := List.length (counts.filter (λ x, x = 1)) in
  let counts2 := List.length (counts.filter (λ x, x = 2)) in
  counts0 = 3 ∧ counts1 = 4 ∧ counts2 = 5

-- The primary theorem statement
theorem impossible_arrangement : ¬(∃ g : grid_100x100 → digit, arrangement g) :=
by {
  sorry
}

end impossible_arrangement_l649_649164


namespace jade_cal_difference_l649_649986

def Mabel_transactions : ℕ := 90

def Anthony_transactions : ℕ := Mabel_transactions + (Mabel_transactions / 10)

def Cal_transactions : ℕ := (2 * Anthony_transactions) / 3

def Jade_transactions : ℕ := 85

theorem jade_cal_difference : Jade_transactions - Cal_transactions = 19 := by
  sorry

end jade_cal_difference_l649_649986


namespace equivalence_negation_l649_649614

-- Define irrational numbers
def is_irrational (x : ℝ) : Prop :=
  ¬ (∃ q : ℚ, x = q)

-- Define rational numbers
def is_rational (x : ℝ) : Prop :=
  ∃ q : ℚ, x = q

-- Original proposition: There exists an irrational number whose square is rational
def original_proposition : Prop :=
  ∃ x : ℝ, is_irrational x ∧ is_rational (x * x)

-- Negation of the original proposition
def negation_of_proposition : Prop :=
  ∀ x : ℝ, is_irrational x → ¬is_rational (x * x)

-- Proof statement that the negation of the original proposition is equivalent to "Every irrational number has a square that is not rational"
theorem equivalence_negation :
  (¬ original_proposition) ↔ negation_of_proposition :=
sorry

end equivalence_negation_l649_649614


namespace gallons_added_in_fourth_hour_l649_649484

-- Defining the conditions
def initial_volume : ℕ := 40
def loss_rate_per_hour : ℕ := 2
def add_in_third_hour : ℕ := 1
def remaining_after_fourth_hour : ℕ := 36

-- Prove the problem statement
theorem gallons_added_in_fourth_hour :
  ∃ (x : ℕ), initial_volume - 2 * 4 + 1 - loss_rate_per_hour + x = remaining_after_fourth_hour :=
sorry

end gallons_added_in_fourth_hour_l649_649484


namespace sum_inequality_l649_649188

theorem sum_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h_pos : ∀ k, 1 ≤ k ∧ k ≤ n → 0 < a k)  -- Condition ensuring positivity
  (h_cond : ∀ k, 1 ≤ k ∧ k ≤ n → (∏ i in finset.range k, a (i + 1)) ≥ 1) :  -- Given condition
  ∑ k in finset.range n, k.succ / (∏ j in finset.range k.succ, (1 + a (j + 1))) < 2 := 
by
  sorry

end sum_inequality_l649_649188


namespace f_max_f_range_l649_649345

def f (x : ℤ) : ℤ :=
if x > 100 then x - 10 else sorry

theorem f_max (x : ℤ) : f(x) = max 91 (x - 10) := 
sorry

theorem f_range (z : ℤ) : (∃ x : ℤ, f(x) = z) ↔ z >= 91 := 
sorry

end f_max_f_range_l649_649345


namespace initial_customers_l649_649361

theorem initial_customers (S : ℕ) (initial : ℕ) (H1 : initial = S + (S + 5)) (H2 : S = 3) : initial = 11 := 
by
  sorry

end initial_customers_l649_649361


namespace road_reformation_possible_l649_649004

variable (V : Type) [Fintype V] [DecidableEq V]

namespace GraphTheory

structure Graph :=
(vertices : Finset V)
(edges : Finset (Sym2 V))
(degree : V → ℕ)
(degree_correct : ∀ v, degree v = 100)
(no_multi_edges : ∀ {u v}, u ≠ v → (u, v) ∈ edges ↔ (v, u) ∈ edges)
(no_self_loop : ∀ v, ¬(v, v) ∈ edges)

def swap_edge (G : Graph V) (a b c d : V) : Graph V :=
{ vertices := G.vertices,
  edges := (G.edges \ {(a, b), (c, d)}) ∪ {(a, d), (b, c)},
  degree := G.degree,
  degree_correct := G.degree_correct,
  no_multi_edges := λ u v h, by sorry,
  no_self_loop := λ v, by sorry }

noncomputable def reachable_via_swaps (initial final : Graph V) : Prop :=
∃ (sequence : list (V × V × V × V)), final = sequence.foldl (λ G ⟨a, b, c, d⟩, swap_edge G a b c d) initial

theorem road_reformation_possible (G H : Graph V) (hG : ∀ v, G.degree v = 100) (hH : ∀ v, H.degree v = 100) :
  reachable_via_swaps G H :=
sorry

end GraphTheory

end road_reformation_possible_l649_649004


namespace complex_magnitude_product_l649_649402

theorem complex_magnitude_product :
  (Complex.abs ((3 * Real.sqrt 3 - 3 * Complex.i) * (2 * Real.sqrt 2 + 2 * Complex.i)))
  = 12 * Real.sqrt 3 :=
by
  sorry

end complex_magnitude_product_l649_649402


namespace tangents_from_point_to_circle_l649_649068

theorem tangents_from_point_to_circle (x y k : ℝ) (
    P : ℝ × ℝ)
    (h₁ : P = (1, -1))
    (circle_eq : x^2 + y^2 + 2*x + 2*y + k = 0)
    (h₂ : P = (1, -1))
    (has_two_tangents : 1^2 + (-1)^2 - k / 2 > 0):
  -2 < k ∧ k < 2 :=
by 
    sorry

end tangents_from_point_to_circle_l649_649068


namespace distinct_real_roots_of_quadratic_find_m_and_other_root_l649_649446

theorem distinct_real_roots_of_quadratic (m : ℝ) (h_neg_m : m < 0) : 
    ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (∀ x, x^2 - 2*x + m = 0 → (x = x₁ ∨ x = x₂))) := 
by 
  sorry

theorem find_m_and_other_root (m : ℝ) (h_neg_m : m < 0) (root_minus_one : ∀ x, x^2 - 2*x + m = 0 → x = -1):
    m = -3 ∧ (∃ x, x^2 - 2*x - 3 = 0 ∧ x = 3) := 
by 
  sorry

end distinct_real_roots_of_quadratic_find_m_and_other_root_l649_649446


namespace smallest_a_inequality_l649_649426

noncomputable def smallest_a := 0.79

theorem smallest_a_inequality (x : ℝ) (hx : x ∈ Ioo (3 * π / 2) 2 * π) :
  (∛(sin x ^ 2) - ∛(cos x ^ 2)) / (∛(tan x ^ 2) - ∛(cot x ^ 2)) < smallest_a / 2 :=
begin
  sorry
end

end smallest_a_inequality_l649_649426


namespace sequence_convergence_l649_649071

theorem sequence_convergence (k : ℝ) (h_k_pos : 0 < k) (x0 : ℝ) :
  (∀ n : ℕ, x (n+1) = x n * (2 - k * x n)) → 
  (∃ L : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ m > N, abs (x m - L) < ε) ∧ L = 1 / k) ↔ (0 < x0 ∧ x0 < 2 / k) :=
sorry

end sequence_convergence_l649_649071


namespace cos_450_eq_zero_l649_649738

theorem cos_450_eq_zero :
  ∀ (θ : ℝ), θ = 450 ∧ (θ mod 360 = 90) ∧ (cos 90 = 0) → cos θ = 0 :=
by
  intros θ hθ
  cases hθ with h1 hand
  cases hand with h2 h3
  rw h1
  rw h2
  exact h3

end cos_450_eq_zero_l649_649738


namespace linear_function_quadrants_l649_649504

theorem linear_function_quadrants (m : ℝ) (h1 : m - 2 < 0) (h2 : m + 1 > 0) : -1 < m ∧ m < 2 := 
by 
  sorry

end linear_function_quadrants_l649_649504


namespace find_k_and_a_l649_649856

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := log 4 (4 ^ x + 1) + k * x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := log 4 (a * 2 ^ x - (4 / 3) * a)

theorem find_k_and_a (k a : ℝ):
  (∀ x : ℝ, f x k = f (-x) k) →
  (∃ x : ℝ, f x (-1 / 2) = g x a) →
  k = -1 / 2 ∧ (a > 1 ∨ a = -3) :=
by {
  sorry
}

end find_k_and_a_l649_649856


namespace series_sum_l649_649945

noncomputable def compute_series (x : ℝ) (hx : x > 1) : ℝ :=
  ∑' n, 1 / (x ^ (3 ^ n) - x ^ (- 3 ^ n))

theorem series_sum (x : ℝ) (hx : x > 1) : compute_series x hx = 1 / (x - 1) :=
sorry

end series_sum_l649_649945


namespace book_cost_proof_l649_649881

variable (C1 C2 : ℝ)

theorem book_cost_proof (h1 : C1 + C2 = 460)
                        (h2 : C1 * 0.85 = C2 * 1.19) :
    C1 = 268.53 := by
  sorry

end book_cost_proof_l649_649881


namespace stewart_farm_l649_649372

variable (sheep horses : Nat) (sheep_to_horses : ℚ) (total_horse_food_per_day : ℚ) (food_per_horse : ℚ)

-- The conditions given in the problem
def conditions : Prop :=
  sheep_to_horses = 5 / 7 ∧
  sheep = 40 ∧
  total_horse_food_per_day = 12880

-- The main proposition we need to prove
def proof_problem : Prop :=
  food_per_horse = 230

-- The statement combining the conditions and the problem
theorem stewart_farm :
  conditions sheep horses sheep_to_horses total_horse_food_per_day food_per_horse →
  proof_problem total_horse_food_per_day food_per_horse :=
by
  sorry

end stewart_farm_l649_649372


namespace jerry_age_l649_649211

theorem jerry_age (M J : ℕ) (h1 : M = 2 * J - 6) (h2 : M = 16) : J = 11 :=
sorry

end jerry_age_l649_649211


namespace number_of_mailboxes_l649_649695

theorem number_of_mailboxes
  (blocks : ℕ)
  (edges : ℕ)
  (faces : ℕ)
  (h_blocks : blocks = 12)
  (h_edges : edges = 37)
  (h_faces : faces = blocks + 1) :
  let V := 2 + edges - faces in V = 26 :=
by
  have h1 : V = 2 + edges - faces := rfl
  have h2 : faces = 12 + 1 := by rw [h_blocks, h_faces]
  have h3 : faces = 13 := rfl
  have h4 : V = 2 + 37 - 13 := by rw [h2, h_edges]
  have h5 : V = 26 := rfl
  exact h5

end number_of_mailboxes_l649_649695


namespace max_val_f_min_val_f_decreasing_interval_f_l649_649872

noncomputable def a : ℝ × ℝ := (1/2, real.sqrt 3 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (real.sin x, real.cos x)
noncomputable def f (x : ℝ) : ℝ := (1/2 * real.sin x + real.sqrt 3 / 2 * real.cos x + 2)

theorem max_val_f : 
  ∃ k : ℤ, f (2 * k * real.pi + real.pi / 6) = 3 := sorry

theorem min_val_f : 
  ∃ k : ℤ, f (2 * k * real.pi - 5 * real.pi / 6) = 1 := sorry

theorem decreasing_interval_f :
  ∀ x ∈ (set.Icc (real.pi / 6) (7 * real.pi / 6) : set ℝ), 
  derivative f(x) < 0 := sorry

end max_val_f_min_val_f_decreasing_interval_f_l649_649872


namespace base_difference_is_correct_l649_649408

-- Definitions of given conditions
def base9_to_base10 (n : Nat) : Nat :=
  match n with
  | 324 => 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  | _ => 0

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 231 => 2 * 6^2 + 3 * 6^1 + 1 * 6^0
  | _ => 0

-- Lean statement to prove the equivalence
theorem base_difference_is_correct : base9_to_base10 324 - base6_to_base10 231 = 174 :=
by
  sorry

end base_difference_is_correct_l649_649408


namespace combined_time_alligators_walked_l649_649167

theorem combined_time_alligators_walked
  (time_to_nile_delta : ℕ)
  (extra_return_time : ℕ)
  (return_time_with_alligators : ℕ)
  (combined_time : ℕ) :
  time_to_nile_delta = 4 →
  extra_return_time = 2 →
  return_time_with_alligators = time_to_nile_delta + extra_return_time →
  combined_time = time_to_nile_delta + return_time_with_alligators →
  combined_time = 10 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end combined_time_alligators_walked_l649_649167


namespace comb_20_6_l649_649014

theorem comb_20_6 : nat.choose 20 6 = 19380 :=
by sorry

end comb_20_6_l649_649014


namespace length_of_bridge_l649_649665

-- Definitions of conditions in the problem
def train_length : Real := 120 -- meters
def train_speed_kmh : Real := 45 -- km/hr
def crossing_time : Real := 30 -- seconds

-- Conversion Ratio
def kmh_to_ms (speed_kmh : Real) : Real := speed_kmh * 1000 / 3600

-- The main theorem we need to prove.
theorem length_of_bridge :
  let train_speed_ms := kmh_to_ms train_speed_kmh in
  let total_distance := train_speed_ms * crossing_time in
  let bridge_length := total_distance - train_length in
  bridge_length = 255 :=
by
  -- Implementation details of proof are omitted with 'sorry'
  sorry

end length_of_bridge_l649_649665


namespace length_of_rectangle_l649_649600

theorem length_of_rectangle (l : ℝ) (s : ℝ) 
  (perimeter_square : 4 * s = 160) 
  (area_relation : s^2 = 5 * (l * 10)) : 
  l = 32 :=
by
  sorry

end length_of_rectangle_l649_649600


namespace max_consecutive_integers_sum_lt_1000_l649_649235

theorem max_consecutive_integers_sum_lt_1000:
  (∀ n : ℕ, 3 + 4 + ... + (3 + n - 1) ≤ 1000 → n ≤ 42) :=
by
  have sum_integers : ∀ n : ℕ, 3 + 4 + ... + (3 + n - 1) = (n + 2) * (n + 3) / 2 - 3 := sorry
  have proof_inequality : ∀ n : ℕ, (n + 2) * (n + 3) / 2 - 3 ≤ 1000 → n ≤ 42 := sorry
  exact proof_inequality

end max_consecutive_integers_sum_lt_1000_l649_649235


namespace total_vegetables_l649_649204

variable (x y z g k h : ℝ)

theorem total_vegetables :
  x = 5 →
  y = 2 →
  k = 2 →
  h = 20 →
  z = 2*k →
  g = 0.5*h - 3 →
  x + y + z + g = 18 := by
  intros hx hy hk hh hz hg
  rw [hx, hy, hk, hh, hz, hg]
  norm_num

end total_vegetables_l649_649204


namespace log_base_2_of_a_l649_649840

theorem log_base_2_of_a (a : ℝ) (h₁ : a > 0) (h₂ : a^(1/2) = 4) : log 2 a = 4 :=
sorry

end log_base_2_of_a_l649_649840


namespace no_ten_digit_divisor_with_distinct_digits_l649_649395

theorem no_ten_digit_divisor_with_distinct_digits 
  (n : ℕ) 
  (hn : n = Nat.pow 10 1000 - 1 / 9) :
  ¬ ∃ d : ℕ, (Nat.digits 10 d).nodup ∧ (Nat.digits 10 d).length = 10 ∧ d ∣ n ∧ 
  ∀ x, x ∈ Nat.digits 10 d → x ≤ 9 :=
by
  sorry

end no_ten_digit_divisor_with_distinct_digits_l649_649395


namespace polynomial_factor_l649_649886

theorem polynomial_factor :
  ∀ (c : ℝ), (∃ (a : ℝ), (λ x, (x - 1) * (x + a)) = (λ x, x^2 - 5 * x + c)) → c = 4 :=
by
  sorry

end polynomial_factor_l649_649886


namespace sum_of_un_eq_u0_l649_649302

open_locale big_operators
noncomputable theory

variables (u0 z0 : ℝ × ℝ)
variables (un zn : ℕ → ℝ × ℝ)

-- Define the initial vectors
def u0 : ℝ × ℝ := (2, 4)
def z0 : ℝ × ℝ := (3, 1)

-- Define the projection functions
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let c := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2) in
  (c * v.1, c * v.2)

-- Define the sequences
def un (n : ℕ) : ℝ × ℝ :=
  if n = 0 then u0 else proj (zn (n - 1)) u0

def zn (n : ℕ) : ℝ × ℝ :=
  if n = 0 then z0 else proj (un n) z0

-- Define the infinite sum of the sequence un
def infinite_sum_un : ℝ × ℝ :=
  let c := 1 / 2 in
  let sum := (c / (1 - c)) in
  (sum * u0.1, sum * u0.2)

theorem sum_of_un_eq_u0 : infinite_sum_un = u0 :=
by
  -- This is where the proof would be constructed
  sorry

end sum_of_un_eq_u0_l649_649302


namespace distinct_collections_count_l649_649987

noncomputable def COMPUTATIONS : list char := ['C', 'O', 'M', 'P', 'U', 'T', 'A', 'T', 'I', 'O', 'N', 'S']

-- Define the vowels in "COMPUTATIONS"
def vowels : list char := ['O', 'U', 'A', 'I', 'O']

-- Define the consonants in "COMPUTATIONS"
def consonants : list char := ['C', 'M', 'P', 'T', 'T', 'N', 'S']

-- Define the main theorem
theorem distinct_collections_count : 
  (number of ways to select 4 vowels from vowels) *
  ((number of ways to select 3 from non-T consonants) + (number of ways to select 4 from non-T consonants)) +
  (number of ways to select 4 vowels from vowels) * (number of ways to select 2 from non-T consonants)
  = 125 := 
  sorry

end distinct_collections_count_l649_649987


namespace solve_for_x_l649_649312

variable (a b c d x : ℝ)

theorem solve_for_x (h1 : a ≠ b) (h2 : b ≠ 0) 
  (h3 : d ≠ c) (h4 : c % x = 0) (h5 : d % x = 0) 
  (h6 : (2*a + x) / (3*b + x) = c / d) : 
  x = (3*b*c - 2*a*d) / (d - c) := 
sorry

end solve_for_x_l649_649312


namespace inequality_solution_l649_649895

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) → a ≥ 2 :=
by
  sorry

end inequality_solution_l649_649895


namespace cotangent_difference_l649_649967

theorem cotangent_difference (A B C F : Point) (h_midpoint : midpoint F B C):
  cot (angle A F C) - cot (angle A F B) = 2 * (cot (angle A F B)) :=
sorry

end cotangent_difference_l649_649967


namespace coefficient_x3_in_expansion_l649_649603

theorem coefficient_x3_in_expansion :
  let f := (2 * x - 1 / (2 * sqrt x))^6,
      coef_x3 := 60 in
    coefficient_of_term_x3 f = coef_x3 := 
by sorry

end coefficient_x3_in_expansion_l649_649603


namespace max_digit_e_l649_649041

theorem max_digit_e 
  (d e : ℕ) 
  (digits : ∀ (n : ℕ), n ≤ 9) 
  (even_e : e % 2 = 0) 
  (div_9 : (22 + d + e) % 9 = 0) 
  : e ≤ 8 :=
sorry

end max_digit_e_l649_649041


namespace comb_20_6_l649_649015

theorem comb_20_6 : nat.choose 20 6 = 19380 :=
by sorry

end comb_20_6_l649_649015


namespace minimum_non_negative_pairs_l649_649824

theorem minimum_non_negative_pairs (s : Fin 100 → ℝ) (h_sum : (Finset.univ.sum s) = 0) : 
  ∃ (f : (Fin 100) → (Fin 100) → ℕ), (∀ (i j : Fin 100), f i j = 1 ∨ f i j = 0) ∧ 
  (Finset.univ.sum (λ (i : Fin 100), Finset.univ.sum (λ (j : Fin 100), f i j)) = 99) ∧ 
  (∀ (i j : Fin 100), f i j = 1 → s i + s j ≥ 0) :=
sorry

end minimum_non_negative_pairs_l649_649824


namespace polygon_with_140_degree_interior_angle_is_nonagon_l649_649500

theorem polygon_with_140_degree_interior_angle_is_nonagon
  (interior_angle : ℝ)
  (h1 : interior_angle = 140) :
  ∃ n : ℕ, n = 9 :=
by
  have exterior_angle := 180 - interior_angle
  have h2 : exterior_angle = 40 := by linarith
  have num_sides := 360 / exterior_angle
  have h3 : num_sides = 9 := by norm_num
  use 9
  exact h3


end polygon_with_140_degree_interior_angle_is_nonagon_l649_649500


namespace Sk_plus_one_l649_649818

def S (k : Nat) : ℚ :=
  ∑ i in Finset.range (2 * k - k), (1 / (i + (k + 1) : ℚ))

theorem Sk_plus_one (k : Nat) : S (k + 1) = S k + (1 / (2 * k + 1 : ℚ)) - (1 / (2 * k + 2 : ℚ)) :=
by
  sorry

end Sk_plus_one_l649_649818


namespace set_roster_method_correct_l649_649796

-- Define the set M
def M := {m : ℤ | 10 % (m + 1) = 0}

-- Define the expected set in roster method
def M_expected := {-11, -6, -3, -2, 0, 1, 4, 9}

-- Prove that M is equal to the expected set
theorem set_roster_method_correct : M = M_expected :=
by
  sorry

end set_roster_method_correct_l649_649796


namespace original_cost_of_car_l649_649575

noncomputable def original_cost (C : ℝ) : ℝ :=
  if h : C + 13000 ≠ 0 then (60900 - (C + 13000)) / (C + 13000) * 100 else 0

theorem original_cost_of_car 
  (C : ℝ) 
  (h1 : original_cost C = 10.727272727272727)
  (h2 : 60900 - (C + 13000) > 0) :
  C = 433500 :=
by
  sorry

end original_cost_of_car_l649_649575


namespace ratio_kept_to_deleted_l649_649636

def initial_songs : ℕ := 1200
def downloaded_songs : ℕ := 200
def fraction_removed : ℚ := 1 / 6

def updated_songs : ℕ := initial_songs + downloaded_songs
def removed_songs : ℕ := (updated_songs * fraction_removed).natPart
def kept_songs : ℕ := updated_songs - removed_songs

theorem ratio_kept_to_deleted : 
  let kept := updated_songs - removed_songs
  let deleted := removed_songs
  (kept : ℚ) / (deleted : ℚ) = (1167 : ℚ) / (233 : ℚ) := by
  sorry

end ratio_kept_to_deleted_l649_649636


namespace least_number_of_attendees_l649_649296

-- Definitions based on problem conditions
inductive Person
| Anna
| Bill
| Carl
deriving DecidableEq

inductive Day
| Mon
| Tues
| Wed
| Thurs
| Fri
deriving DecidableEq

def attends : Person → Day → Prop
| Person.Anna, Day.Mon => true
| Person.Anna, Day.Tues => false
| Person.Anna, Day.Wed => true
| Person.Anna, Day.Thurs => false
| Person.Anna, Day.Fri => false
| Person.Bill, Day.Mon => false
| Person.Bill, Day.Tues => true
| Person.Bill, Day.Wed => false
| Person.Bill, Day.Thurs => true
| Person.Bill, Day.Fri => true
| Person.Carl, Day.Mon => true
| Person.Carl, Day.Tues => true
| Person.Carl, Day.Wed => false
| Person.Carl, Day.Thurs => true
| Person.Carl, Day.Fri => false

-- Proof statement
theorem least_number_of_attendees : 
  (∀ d : Day, (∀ p : Person, attends p d → p = Person.Anna ∨ p = Person.Bill ∨ p = Person.Carl) ∧
              (d = Day.Wed ∨ d = Day.Fri → (∃ n : ℕ, n = 2 ∧ (∀ p : Person, attends p d → n = 2))) ∧
              (d = Day.Mon ∨ d = Day.Tues ∨ d = Day.Thurs → (∃ n : ℕ, n = 1 ∧ (∀ p : Person, attends p d → n = 1))) ∧
              ¬ (d = Day.Wed ∨ d = Day.Fri)) :=
sorry

end least_number_of_attendees_l649_649296


namespace slope_of_line_between_midpoints_l649_649306

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.fst + b.fst) / 2, (a.snd + b.snd) / 2)

def midpoint1 := midpoint (1, 2) (3, 6)
def midpoint2 := midpoint (4, 1) (7, 6)

theorem slope_of_line_between_midpoints :
  slope midpoint1 midpoint2 = -1 / 7 := by
  sorry

end slope_of_line_between_midpoints_l649_649306


namespace tenth_equation_sum_of_cubes_l649_649215

theorem tenth_equation_sum_of_cubes :
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3 + 9^3 + 10^3) = 55^2 := 
by sorry

end tenth_equation_sum_of_cubes_l649_649215


namespace count_valid_permutations_eq_X_l649_649028

noncomputable def valid_permutations_count : ℕ :=
sorry

theorem count_valid_permutations_eq_X : valid_permutations_count = X :=
sorry

end count_valid_permutations_eq_X_l649_649028


namespace simplify_4sqrt2_minus_sqrt2_l649_649314

/-- Prove that 4 * sqrt 2 - sqrt 2 = 3 * sqrt 2 given standard mathematical rules -/
theorem simplify_4sqrt2_minus_sqrt2 : 4 * Real.sqrt 2 - Real.sqrt 2 = 3 * Real.sqrt 2 :=
sorry

end simplify_4sqrt2_minus_sqrt2_l649_649314


namespace quadratic_two_distinct_real_roots_find_m_and_other_root_l649_649448

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c
def roots_sum (a b : ℝ) : ℝ := -b / a

theorem quadratic_two_distinct_real_roots (m : ℝ)
  (hm : m < 0) :
  ∀ (a b c : ℝ), a = 1 → b = -2 → c = m → (discriminant a b c > 0) :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc, discriminant]
  sorry

theorem find_m_and_other_root (a b c r1 : ℝ)
  (ha : a = 1)
  (hb : b = -2)
  (hc : c = r1^2 - 2*r1 + c = 0)
  (hr1 : r1 = -1)
  :
  c = -3 ∧ 
  ∃ r2 : ℝ, (roots_sum a b = 2) ∧ (r1 + r2 = 2) ∧ (r1 = -1 → r2 = 3) :=
by
  intros
  rw [ha, hb, hr1]
  sorry

end quadratic_two_distinct_real_roots_find_m_and_other_root_l649_649448


namespace zero_point_interval_l649_649278

theorem zero_point_interval (f : ℝ → ℝ) (x₀ : ℝ) (k : ℤ)
  (h₀ : f x₀ = 0)
  (h_def : ∀ x > 0, f x = Real.log x - 1 / x)
  (h_interval : x₀ ∈ Set.Ico (↑k) (↑k + 1)) :
  k = 1 :=
by
  sorry

end zero_point_interval_l649_649278


namespace find_integer_satisfying_condition_l649_649049

theorem find_integer_satisfying_condition : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [MOD 10] ∧ n = 2 :=
by
  sorry

end find_integer_satisfying_condition_l649_649049


namespace smaller_angle_at_8_15_pm_l649_649646

noncomputable def smaller_angle_between_clock_hands (minute_hand_degrees_per_min: ℝ) (hour_hand_degrees_per_min: ℝ) (time_in_minutes: ℝ) : ℝ := sorry

theorem smaller_angle_at_8_15_pm :
  smaller_angle_between_clock_hands 6 0.5 495 = 157.5 :=
sorry

end smaller_angle_at_8_15_pm_l649_649646


namespace log_arith_seq_necessity_condition_l649_649094

theorem log_arith_seq_necessity_condition (x y z : ℝ) (hx : 0 < x) (hz : 0 < z) :
  (∃ d : ℝ, log 10 x + d = log 10 y ∧ log 10 y + d = log 10 z) → y^2 = x * z :=
by
  sorry

end log_arith_seq_necessity_condition_l649_649094


namespace part1_part2_l649_649245

theorem part1 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 2) : a^2 + b^2 = 21 :=
  sorry

theorem part2 (a b : ℝ) (h1 : a + b = 10) (h2 : a^2 + b^2 = 50^2) : a * b = -1200 :=
  sorry

end part1_part2_l649_649245


namespace total_work_completion_days_l649_649363

theorem total_work_completion_days :
  let Amit_work_rate := 1 / 15
  let Ananthu_work_rate := 1 / 90
  let Chandra_work_rate := 1 / 45

  let Amit_days_worked_alone := 3
  let Ananthu_days_worked_alone := 6
  
  let work_by_Amit := Amit_days_worked_alone * Amit_work_rate
  let work_by_Ananthu := Ananthu_days_worked_alone * Ananthu_work_rate
  
  let initial_work_done := work_by_Amit + work_by_Ananthu
  let remaining_work := 1 - initial_work_done

  let combined_work_rate := Amit_work_rate + Ananthu_work_rate + Chandra_work_rate
  let days_all_worked_together := remaining_work / combined_work_rate

  Amit_days_worked_alone + Ananthu_days_worked_alone + days_all_worked_together = 17 :=
by
  sorry

end total_work_completion_days_l649_649363


namespace oranges_harvest_per_day_l649_649206

theorem oranges_harvest_per_day (total_sacks : ℕ) (days : ℕ) (sacks_per_day : ℕ) 
  (h1 : total_sacks = 498) (h2 : days = 6) : total_sacks / days = sacks_per_day ∧ sacks_per_day = 83 :=
by
  sorry

end oranges_harvest_per_day_l649_649206


namespace find_a_b_l649_649475

variables {a b c : ℝ}

noncomputable def f (x : ℝ) : ℝ := a * x^2 + 1
noncomputable def g (x : ℝ) : ℝ := x^3 + b * x

theorem find_a_b (h1: a > 0) (h2: deriv f 1 = deriv g 1) (h3: f 1 = g 1) : a = 3 ∧ b = 3 :=
begin
  -- Given conditions 
  have h4 : f'(x) = 2 * a * x, from sorry,
  have h5 : g'(x) = 3 * x^2 + b, from sorry,
  have h6 : f'(1) = 2 * a, from sorry,
  have h7 : g'(1) = 3 + b, from sorry,
  
  -- Equation from tangent slopes
  have eq1 : 2 * a = 3 + b, from sorry,
  -- Equation from function values at x = 1
  have eq2 : a + 1 = 1 + b, from sorry,
  
  -- Solving the system of equations
  have h8 : a = b, from sorry,
  have h9 : a = 3, from sorry,
  exact ⟨h9, h9⟩,
end

end find_a_b_l649_649475


namespace girls_boys_ratio_l649_649704

theorem girls_boys_ratio (G B : ℕ) (h1 : G + B = 100) (h2 : 0.20 * (G : ℝ) + 0.10 * (B : ℝ) = 15) : G / B = 1 :=
by
  -- Proof steps are omitted
  sorry

end girls_boys_ratio_l649_649704


namespace distinct_x_intercepts_l649_649125

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem distinct_x_intercepts (x : ℝ) :
  let y := (x - 5) * (x^2 + 5 * x + 10) in
  (∀ x, y = 0) ↔ ∃! x, y = 0 := by
{
  sorry
}

end distinct_x_intercepts_l649_649125


namespace sum_real_coeff_log_l649_649924

theorem sum_real_coeff_log (x : ℝ) (S : ℝ) : 
  (S = (list.sum (list.map (λ (cn : complex), cn.re) ((1 + complex.I * x) ^ 2009).toList))) →
  log2(S) = 1004 :=
by
  sorry

end sum_real_coeff_log_l649_649924


namespace sin_shift_right_l649_649635

theorem sin_shift_right : 
  ∀ x : ℝ, sin (x - (π / 12)) = sin ((x - (π / 12)) + (π / 12)) :=
by simp [sin, sub_eq_add_neg, add_comm, add_neg_cancel_right]

-- The definition for the shift we are proving
def shift_right (x shift : ℝ) : ℝ := x - shift

example : sin (shift_right x (π / 12)) = sin x := 
by unfold shift_right; simp [sin_shift_right]; sorry

end sin_shift_right_l649_649635


namespace next_two_terms_in_sequence_l649_649116

noncomputable def sequence : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| 3 => 6
| 4 => 16
| 5 => 44
| 6 => 120
| n => (sequence (n - 1) + sequence (n - 2)) * 2

theorem next_two_terms_in_sequence :
  sequence 7 = 328 ∧ sequence 8 = 896 :=
by
  sorry

end next_two_terms_in_sequence_l649_649116


namespace incenter_distance_PJ_l649_649637

noncomputable def P := (0, 0)
noncomputable def Q := (30, 0)
noncomputable def R := (4, 29) -- Arbitrary point ensuring sides' lengths; exact coordinates are not necessary for this statement

-- Distance function
def distance (A B : Real × Real) : Real := 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Definitions of the sides according to given lengths
def PQ : Real := distance P Q
def PR : Real := distance P R
def QR : Real := distance Q R

lemma PQ_length : PQ = 30 := by
  sorry

lemma PR_length : PR = 29 := by 
  sorry

lemma QR_length : QR = 31 := by
  sorry

-- Incenter (bisection point calculations are non-trivial and skipped for brevity)
def incenter (A B C : Real × Real) : Real × Real := sorry

-- Definition of the result to prove
theorem incenter_distance_PJ :
  distance P (incenter P Q R) = 14 := sorry

end incenter_distance_PJ_l649_649637


namespace book_selection_l649_649882

theorem book_selection (total_books novels : ℕ) (choose_books : ℕ)
  (h_total : total_books = 15)
  (h_novels : novels = 5)
  (h_choose : choose_books = 3) :
  (Nat.choose 15 3 - Nat.choose 10 3) = 335 :=
by
  sorry

end book_selection_l649_649882


namespace infinite_series_sum_l649_649383

open BigOperators

noncomputable def infinite_sum := ∑' n : ℕ, (3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3))

theorem infinite_series_sum : infinite_sum = 23 / 18 :=
by
  sorry

end infinite_series_sum_l649_649383


namespace cost_of_four_dozen_apples_l649_649715

-- Define the given conditions and problem
def half_dozen_cost : ℚ := 4.80 -- cost of half a dozen apples
def full_dozen_cost : ℚ := half_dozen_cost / 0.5
def four_dozen_cost : ℚ := 4 * full_dozen_cost

-- Statement of the theorem to prove
theorem cost_of_four_dozen_apples : four_dozen_cost = 38.40 :=
by
  sorry

end cost_of_four_dozen_apples_l649_649715


namespace trapezoid_bases_ratio_l649_649816

universe u

variables {α : Type u} {S A B C D : α}

-- Geometric definitions and conditions
variables (circle : α) 
variables (outside_point S : α) (tangent_point_A tangent_point_C : α)
variables (secant_point_B secant_point_D : α)

axiom tangent_lines_from_point : ∀ {x y}, x ∈ circle → y ∈ circle → x ≠ y → segment S x ∥ segment S y
axiom secant_line_intersects_circle : ∀ {x y}, x ∈ circle → y ∈ circle → segment S x ∩ circle = {x, y}

axiom forms_trapezoid : segment A B ∥ segment C D
axiom angle_between_tangents : ∀ {α}, angle (segment S A) (segment S C) = 60°

theorem trapezoid_bases_ratio :
  let x := distance A B in
  let y := distance C D in
  (∃ k: ℝ, k = sqrt ((3 - sqrt 5) / 2) ∧ (x / y) = k^2) :=
sorry

end trapezoid_bases_ratio_l649_649816


namespace third_number_LCM_l649_649802

theorem third_number_LCM (n : ℕ) : 
  (n > 0) ∧ (n ∣ 120) ∧ (¬(n ∣ 24)) ∧ (¬(n ∣ 30)) ∧ (nat.lcm 24 (nat.lcm 30 n) = 120) → n = 10 := 
sorry

end third_number_LCM_l649_649802


namespace ordered_triples_count_l649_649926

namespace LeanVerify

def S : Finset ℕ := {n | 1 ≤ n ∧ n ≤ 15}

def succ (a b : ℕ) : Prop := (0 < a - b ∧ a - b ≤ 7) ∨ (b - a > 7)

theorem ordered_triples_count : 
  (Finset.filter (λ (t : ℕ × ℕ × ℕ), succ t.1 t.2 ∧ succ t.2 t.3 ∧ succ t.3 t.1) 
    (S.product (S.product S))).card = 420 :=
by {
  sorry
}

end LeanVerify

end ordered_triples_count_l649_649926


namespace five_tuesdays_in_june_implies_five_wednesdays_in_july_l649_649237

theorem five_tuesdays_in_june_implies_five_wednesdays_in_july
  (N : ℕ)
  (June_has_30_days : ∀ N : ℕ, days_in_month N 6 = 30)
  (July_has_31_days : ∀ N : ℕ, days_in_month N 7 = 31)
  (June_has_five_Tuesdays : ∃ weeks : list week, (length weeks = 5) ∧ all_days_are_tuesday weeks) :
  ∃ weeks : list week, (length weeks = 5) ∧ all_days_are_wednesday weeks :=
sorry

end five_tuesdays_in_june_implies_five_wednesdays_in_july_l649_649237


namespace value_of_a_minus_b_l649_649823

variables (a b : ℝ)

theorem value_of_a_minus_b (h1 : abs a = 3) (h2 : abs b = 5) (h3 : a > b) : a - b = 8 :=
sorry

end value_of_a_minus_b_l649_649823


namespace sum_series_eq_l649_649951

open Real

theorem sum_series_eq (x : ℝ) (h : 1 < x) : 
  (∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (- (3 ^ n)))) = 1 / (x - 1) :=
sorry

end sum_series_eq_l649_649951


namespace BH_divides_CD_proof_l649_649149

noncomputable def BH_divides_CD (A B C D H : Type*)
  (h1 : convex A B C D)
  (h2 : angle A B C = 90)
  (h3 : angle B A C = angle C A D)
  (h4 : dist A C = dist A D)
  (h5 : altitude D H C) : Type* :=
sorry

theorem BH_divides_CD_proof (A B C D H : Type*)
  (h1 : convex A B C D)
  (h2 : angle A B C = 90)
  (h3 : angle B A C = angle C A D)
  (h4 : dist A C = dist A D)
  (h5 : altitude D H C) :
  divides_evenly H B C D :=
sorry

end BH_divides_CD_proof_l649_649149


namespace inverse_function_value_l649_649628

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 4
  | 2 => 3
  | 3 => 2
  | 4 => 5
  | 5 => 1
  | _ => 0 -- Just to handle other cases (not mentioned in problem)
  end

noncomputable def f_inv (y : ℕ) : ℕ :=
  match y with
  | 4 => 1
  | 3 => 2
  | 2 => 3
  | 5 => 4
  | 1 => 5
  | _ => 0 -- Just to handle other cases (not mentioned in problem)
  end

theorem inverse_function_value :
  f_inv (f_inv (f_inv 3)) = 2 := by
  sorry

end inverse_function_value_l649_649628


namespace min_fuse_length_l649_649687

theorem min_fuse_length 
  (safe_distance : ℝ := 70) 
  (personnel_speed : ℝ := 7) 
  (fuse_burning_speed : ℝ := 10.3) : 
  ∃ (x : ℝ), x ≥ 103 := 
by
  sorry

end min_fuse_length_l649_649687


namespace remainder_a37_div_45_l649_649551

open Nat

def a_n (n : Nat) : Nat :=
  String.join (List.map toString (List.range (n + 1))).toNat

theorem remainder_a37_div_45 : (a_n 37) % 45 = 37 :=
by
  sorry

end remainder_a37_div_45_l649_649551


namespace cakes_served_during_lunch_today_l649_649356

theorem cakes_served_during_lunch_today (L : ℕ) 
  (h_total : L + 6 + 3 = 14) : 
  L = 5 :=
sorry

end cakes_served_during_lunch_today_l649_649356


namespace eq_sqrt4_power_l649_649716

theorem eq_sqrt4_power (x : ℝ) (hx : 0 < x) : (x^2 * sqrt x) ^ (1/4) = x^(5/8) :=
sorry

end eq_sqrt4_power_l649_649716


namespace trapezoid_bc_length_l649_649241

theorem trapezoid_bc_length (area height ab cd : ℝ) (h_area : area = 164) (h_height : height = 8) (h_ab : ab = 10) (h_cd : cd = 17) :
  ∃ (bc : ℝ), bc = 10 :=
by 
  -- Given data
  have height_pos : height > 0 := by linarith,
  have ab_pos : ab > 0 := by linarith,
  have cd_pos : cd > 0 := by linarith,
  
  -- Set up the area formula for the trapezoid
  let total_base := ab + cd,
  have h_area_calc : area = 0.5 * (ab + cd) * height := by 
    rw [h_ab, h_cd, h_height],
    norm_num,
    exact h_area,
  
  -- Perpendicular lengths
  let ae := real.sqrt (ab^2 - height^2),
  let df := real.sqrt (cd^2 - height^2),
  
  -- Total length of the base (AD)
  let ad := ae + df + bc,
  have h_ad : ad = total_base := sorry,

  -- Horizontal distance between perpendiculars
  have h_bc : bc = ad - ae - df := sorry,
  
  -- Confirm the final length of bc
  have bc_length := 10,
  use bc_length,
  exact eq.refl bc_length,
  
  sorry

end trapezoid_bc_length_l649_649241


namespace rhombus_area_l649_649151

def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area :
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (9, 0), (0, -3.5), (-9, 0)]
  let d1 := 7
  let d2 := 18
  area_of_rhombus d1 d2 = 63 :=
by
  sorry

end rhombus_area_l649_649151


namespace number_of_positive_terms_in_sequence_l649_649763

noncomputable def a (n : ℕ) : ℝ := Real.cos (10^n).toReal

theorem number_of_positive_terms_in_sequence : 
  (Finset.univ.filter (λ n : Fin 100, 0 < a (n + 1))).card = 99 := by
  sorry

end number_of_positive_terms_in_sequence_l649_649763


namespace sum_of_integers_remainder_l649_649656

-- Definitions of the integers and their properties
variables (a b c : ℕ)

-- Conditions
axiom h1 : a % 53 = 31
axiom h2 : b % 53 = 17
axiom h3 : c % 53 = 8
axiom h4 : a % 5 = 0

-- The proof goal
theorem sum_of_integers_remainder :
  (a + b + c) % 53 = 3 :=
by
  sorry -- Proof to be provided

end sum_of_integers_remainder_l649_649656


namespace a_minus_b_eq_neg_9_or_neg_1_l649_649440

theorem a_minus_b_eq_neg_9_or_neg_1 (a b : ℝ) (h₁ : |a| = 5) (h₂ : |b| = 4) (h₃ : a + b < 0) :
  a - b = -9 ∨ a - b = -1 :=
by
  sorry

end a_minus_b_eq_neg_9_or_neg_1_l649_649440


namespace product_of_roots_eq_neg35_l649_649130

theorem product_of_roots_eq_neg35 (x : ℝ) : 
  (x + 3) * (x - 5) = 20 → ∃ a b c : ℝ, a ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1 * x2 = -35 := 
by
  sorry

end product_of_roots_eq_neg35_l649_649130


namespace problem_l649_649117

noncomputable def a : ℕ → ℤ
| 0       := -1  -- Note: Lean indices are 0-based; a_1 is a(0)
| (n + 1) := -b n

noncomputable def b : ℕ → ℤ
| 0       := 2  -- Note: Lean indices are 0-based; b_1 is b(0)
| (n + 1) := 2 * a n - 3 * b n

theorem problem : b 2014 + b 2015 = -3 * 2^2015 :=
by sorry

end problem_l649_649117


namespace train_speed_kmph_l649_649707

def speed_of_train (length_of_train : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  (length_of_train / time_to_cross_pole) * (3600 / 1000)

theorem train_speed_kmph :
  speed_of_train 441 21 = 75.6 :=
by
  simp [speed_of_train]
  sorry

end train_speed_kmph_l649_649707


namespace domain_h_l649_649597

-- Define the function f with its domain
variable (f : ℝ → ℝ)
variable h : ℝ → ℝ

-- Define a new function h in terms of f
def h (x : ℝ) : ℝ := f (-3 * x + 1)

-- State the domain of f
variable domain_f : set.Icc (-8 : ℝ) 3

-- Prove the domain of h
theorem domain_h :
  set.Icc (-2/3 : ℝ) 3 = { x : ℝ | -3 * x + 1 ∈ domain_f } :=
begin
  sorry
end

end domain_h_l649_649597


namespace not_divisible_by_5_for_4_and_7_l649_649430

-- Define a predicate that checks if a given number is not divisible by another number
def notDivisibleBy (n k : ℕ) : Prop := ¬ (n % k = 0)

-- Define the expression we are interested in
def expression (b : ℕ) : ℕ := 3 * b^3 - b^2 + b - 1

-- The theorem we want to prove
theorem not_divisible_by_5_for_4_and_7 :
  notDivisibleBy (expression 4) 5 ∧ notDivisibleBy (expression 7) 5 :=
by
  sorry

end not_divisible_by_5_for_4_and_7_l649_649430


namespace f_at_2_l649_649851

def f (x : ℝ) : ℝ :=
if x ≥ 0 then (x^2 - x) / (x + 1) else (x^2 - x) / (x + 1)

theorem f_at_2 : f 2 = 2 / 3 :=
by
  sorry

end f_at_2_l649_649851


namespace solution_of_ffx_eq_zero_l649_649555

-- Define the function f with the given condition
def functional_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x + f(x) = f(f(x))

-- Theorem statement
theorem solution_of_ffx_eq_zero (f : ℝ → ℝ)
  (hf : functional_condition f) : ∀ x, f(f(x)) = 0 ↔ x = 0 :=
sorry

end solution_of_ffx_eq_zero_l649_649555


namespace complex_sum_abc_values_l649_649238

noncomputable def complex_problem (a b c : ℂ) : Prop :=
  a^2 = b - c ∧ b^2 = c - a ∧ c^2 = a - b

theorem complex_sum_abc_values (a b c : ℂ) (h : complex_problem a b c) :
  a + b + c = 0 ∨ a + b + c = complex.I * real.sqrt 6 ∨ a + b + c = -complex.I * real.sqrt 6 :=
sorry

end complex_sum_abc_values_l649_649238


namespace num_positive_cos_terms_l649_649753

def sequence (n : ℕ) : ℝ := Real.cos (10^(n-1) * Real.pi / 180)

theorem num_positive_cos_terms : (Finset.card (Finset.filter (λ n, 0 < sequence n) (Finset.range 100))) = 99 := 
sorry

end num_positive_cos_terms_l649_649753


namespace number_of_positive_terms_in_sequence_l649_649761

noncomputable def a (n : ℕ) : ℝ := Real.cos (10^n).toReal

theorem number_of_positive_terms_in_sequence : 
  (Finset.univ.filter (λ n : Fin 100, 0 < a (n + 1))).card = 99 := by
  sorry

end number_of_positive_terms_in_sequence_l649_649761


namespace number_of_positive_terms_in_sequence_l649_649759

noncomputable def a (n : ℕ) : ℝ := Real.cos (10^n).toReal

theorem number_of_positive_terms_in_sequence : 
  (Finset.univ.filter (λ n : Fin 100, 0 < a (n + 1))).card = 99 := by
  sorry

end number_of_positive_terms_in_sequence_l649_649759


namespace find_c_for_maximum_at_2_l649_649503

noncomputable def f (x c : ℝ) := x * (x - c)^2

theorem find_c_for_maximum_at_2 :
  (∀ x, HasDerivAt (f x) (x * (x - c)^2) x (x = 2) ∧ deriv (f 2) = 0 → c = 6) :=
sorry

end find_c_for_maximum_at_2_l649_649503


namespace volume_of_rectangular_prism_l649_649310

variables (a b c : ℝ)

theorem volume_of_rectangular_prism 
  (h1 : a * b = 12) 
  (h2 : b * c = 18) 
  (h3 : c * a = 9) 
  (h4 : (1 / a) * (1 / b) * (1 / c) = (1 / 216)) :
  a * b * c = 216 :=
sorry

end volume_of_rectangular_prism_l649_649310


namespace find_m_l649_649491

theorem find_m (m : ℝ) (h : 8^(9/2) = 16^m) : m = 27/8 :=
sorry

end find_m_l649_649491


namespace series_sum_l649_649946

noncomputable def compute_series (x : ℝ) (hx : x > 1) : ℝ :=
  ∑' n, 1 / (x ^ (3 ^ n) - x ^ (- 3 ^ n))

theorem series_sum (x : ℝ) (hx : x > 1) : compute_series x hx = 1 / (x - 1) :=
sorry

end series_sum_l649_649946


namespace card_arrangements_count_l649_649589

-- Define the cards and envelopes
inductive Card : Type
| A : Card
| B : Card
| C : Card
| D : Card
| E : Card
| F : Card

inductive Envelope : Type
| e1 : Envelope
| e2 : Envelope
| e3 : Envelope

-- Define the problem conditions and statement
def conditions_met (arrangement : List (Envelope × Card)) : Prop :=
  (arrangement.filter $ fun p => p.2 = Card.A).map Prod.fst = (arrangement.filter $ fun p => p.2 = Card.B).map Prod.fst ∧
  arrangement.map Prod.fst.nodup ∧
  arrangement.length = 6 ∧
  ∀ e, (arrangement.filter $ fun p => p.1 = e).length = 2

theorem card_arrangements_count : 
  ∃ arrangement : List (Envelope × Card), conditions_met arrangement ∧ arrangement.length = 18 :=
sorry

end card_arrangements_count_l649_649589


namespace cone_volume_div_pi_proof_l649_649343

noncomputable def cone_volume_div_pi (R: ℝ) (θ: ℝ) : ℝ := 
  let C := 2 * Real.pi * R in
  let sector_arc_length := θ / 360 * C in
  let r := sector_arc_length / (2 * Real.pi) in
  let l := R in
  let h := Real.sqrt (l^2 - r^2) in
  let V := (1 / 3) * Real.pi * r^2 * h in
  V / Real.pi

theorem cone_volume_div_pi_proof :
  cone_volume_div_pi 20 270 = 1125 * Real.sqrt 7 := by
  sorry

end cone_volume_div_pi_proof_l649_649343


namespace C_D_meeting_time_l649_649058

-- Defining the conditions.
variables (A B C D : Type) [LinearOrderedField A] (V_A V_B V_C V_D : A)
variables (startTime meet_AC meet_BD meet_AB meet_CD : A)

-- Cars' initial meeting conditions
axiom init_cond : startTime = 0
axiom meet_cond_AC : meet_AC = 7
axiom meet_cond_BD : meet_BD = 7
axiom meet_cond_AB : meet_AB = 53
axiom speed_relation : V_A + V_C = V_B + V_D ∧ V_A - V_B = V_D - V_C

-- The problem asks for the meeting time of C and D
theorem C_D_meeting_time : meet_CD = 53 :=
by sorry

end C_D_meeting_time_l649_649058


namespace find_y_if_x_l649_649932

theorem find_y_if_x (x : ℝ) (hx : x^2 + 8 * (x / (x - 3))^2 = 53) :
  (∃ y, y = (x - 3)^3 * (x + 4) / (2 * x - 5) ∧ y = 17000 / 21) :=
  sorry

end find_y_if_x_l649_649932


namespace kimothy_paths_l649_649544

noncomputable def number_of_paths : ℕ :=
  12

theorem kimothy_paths (starts_bottom_left: Bool) (steps: ℕ) (ends_start: Bool)
  (visits_each_once: Bool): starts_bottom_left ∧ steps = 16 ∧ ends_start ∧ visits_each_once -> number_of_paths = 12 :=
by
  intro h
  simp at h
  exact sorry

end kimothy_paths_l649_649544


namespace find_natural_numbers_l649_649409

open Real

theorem find_natural_numbers (n : ℕ) :
  (cos (2 * π / 9) + cos (4 * π / 9) + ⋯ + cos (2 * π * n / 9) = cos (π / 9)) →
  (log 3 ^ 2 n + 14 < log 3 (9 * n ^ 7)) →
  ∃ m : ℕ, m ∈ {3, 4, 5, 6, 7, 8} ∧ n = 2 + 9 * m :=
by
  sorry

end find_natural_numbers_l649_649409


namespace min_value_eq_2_sqrt_11_l649_649415

noncomputable def min_value (x : ℝ) : ℝ :=
  (x^2 + 19) / real.sqrt (x^2 + 8)

theorem min_value_eq_2_sqrt_11 : ∃ x : ℝ, min_value x = 2 * real.sqrt 11 :=
sorry

end min_value_eq_2_sqrt_11_l649_649415


namespace xy_value_l649_649132

variable (x y : ℕ)

def condition1 : Prop := 8^x / 4^(x + y) = 16
def condition2 : Prop := 16^(x + y) / 4^(7 * y) = 256

theorem xy_value (h1 : condition1 x y) (h2 : condition2 x y) : x * y = 48 := by
  sorry

end xy_value_l649_649132


namespace last_number_greater_than_one_l649_649259

theorem last_number_greater_than_one :
  ∀ (S : Set ℕ), S = (Set.range (λ n, 2^n)) ∧ (∀ a b, a ∈ S → b ∈ S → (replace S a b = (a * b) / (a + b))) →
  (last_number S > 1) :=
by
  sorry

end last_number_greater_than_one_l649_649259


namespace icosagon_diagonals_l649_649258

-- Definitions for the number of sides and the diagonal formula
def sides_icosagon : ℕ := 20

def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Statement:
theorem icosagon_diagonals : diagonals sides_icosagon = 170 := by
  apply sorry

end icosagon_diagonals_l649_649258


namespace decreasing_interval_l649_649854

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 15 * x^4 - 15 * x^2

-- State the theorem
theorem decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → f' x < 0 :=
by sorry

end decreasing_interval_l649_649854


namespace symmetric_point_coordinates_l649_649155

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_about_x_axis (P : Point3D) : Point3D :=
  { x := P.x, y := -P.y, z := -P.z }

theorem symmetric_point_coordinates :
  symmetric_about_x_axis {x := 1, y := 3, z := 6} = {x := 1, y := -3, z := -6} :=
by
  sorry

end symmetric_point_coordinates_l649_649155


namespace minimum_value_fractions_l649_649445

variable (a : ℕ → ℝ) (m n : ℕ)
variable (q : ℝ)
variable (h1 : ∀ n, 0 < a n)
variable (h2 : a 7 = a 6 + 2 * a 5)
variable (h3 : ∃ m n, sqrt (a m * a n) = 4 * a 1)

-- The main theorem statement
theorem minimum_value_fractions :
  (∃ m n, (1 / m : ℝ) + 9 / n = 8 / 3 ∧ 
           h1 m ∧ h1 n ∧ h1 1 ∧ 
           h2 ∧ 
           h3) := sorry

end minimum_value_fractions_l649_649445


namespace number_of_factors_l649_649879

-- Define N with its prime factorization
def N : ℕ := 2^3 * 3^2 * 5^1

-- Define the statement to be proved
theorem number_of_factors (N := 2^3 * 3^2 * 5^1) : 
  nat.factors_count N = 24 :=
sorry

end number_of_factors_l649_649879


namespace jonas_socks_solution_l649_649170

theorem jonas_socks_solution (p_s p_h n_p n_t n : ℕ) (h_ps : p_s = 20) (h_ph : p_h = 5) (h_np : n_p = 10) (h_nt : n_t = 10) :
  2 * (p_s * 2 + p_h * 2 + n_p + n_t) = 2 * (p_s * 2 + p_h * 2 + n_p + n_t + n * 2) :=
by
  -- skipping the proof part
  sorry

end jonas_socks_solution_l649_649170


namespace trajectory_of_A_eq_l649_649479

/-- The given conditions for points B and C and the perimeter of ΔABC -/
def B : ℝ × ℝ := (-3, 0)
def C : ℝ × ℝ := (3, 0)
def perimeter (A : ℝ × ℝ) : ℝ :=
  Real.dist B A + Real.dist C A + Real.dist B C

/-- The statement we need to prove -/
theorem trajectory_of_A_eq : 
  ∀ A : ℝ × ℝ, perimeter A = 16 →
  (∃ k : ℝ, k ≠ 0 ∧ (A.2 / k) ^ 2 = 16 * (1 - (A.1 / 5) ^ 2) * k ^ 2) :=
by
  sorry

end trajectory_of_A_eq_l649_649479


namespace horse_food_per_day_l649_649374

-- Define the conditions
variable (num_sheep : ℕ) (num_horses : ℕ) (total_food : ℕ)
variable (ratio_sheep_horses : ℕ × ℕ)
variable (required_food_per_horse : ℕ)

-- Given conditions
def StewartFarmConditions :=
  ratio_sheep_horses = (5, 7) ∧
  num_sheep = 40 ∧
  total_food = 12880

-- Theorem statement
theorem horse_food_per_day (h : StewartFarmConditions) :
  required_food_per_horse = 230 :=
sorry

end horse_food_per_day_l649_649374


namespace proof_problem_l649_649201

theorem proof_problem (y : ℝ) (m : ℝ) (g : ℝ) 
  (h1 : y = (3 + real.sqrt 5) ^ 500) 
  (h2 : m = real.floor y) 
  (h3 : g = y - m) :
  y * (1 - g) = 4 ^ 500 := 
sorry

end proof_problem_l649_649201


namespace SallyCarrots_l649_649580

-- Definitions of the conditions
def FredGrew (F : ℕ) := F = 4
def TotalGrew (T : ℕ) := T = 10
def SallyGrew (S : ℕ) (F T : ℕ) := S + F = T

-- The theorem to be proved
theorem SallyCarrots : ∃ S : ℕ, FredGrew 4 ∧ TotalGrew 10 ∧ SallyGrew S 4 10 ∧ S = 6 :=
  sorry

end SallyCarrots_l649_649580


namespace range_tan_expression_l649_649914

-- Define the angles and sides in the triangle
variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
axiom acute_triangle (h : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
                     (h_sum : A + B + C = π) : true

axiom angle_condition (h_A : A ≤ π / 3)
                      (h_cos : cos C + 4 * cos A ^ 3 - 3 * cos A = 0) : true

-- The goal is to prove this
theorem range_tan_expression (hA : A ≤ π / 3) (h_cos : cos C + 4 * cos A ^ 3 - 3 * cos A = 0)
    (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) (h_sum : A + B + C = π):
    ∃ (y : ℝ), y = 4 * tan A + 1 / tan (B - A) ∧ y ∈ Ioo (7 * sqrt 3 / 3) 5 :=
sorry

end range_tan_expression_l649_649914


namespace GP_plus_GQ_plus_GR_l649_649915

noncomputable def XYZ_triangle := {X Y Z : Type} 
  (XY XZ YZ : ℝ) (h1 : XY = 5) (h2 : XZ = 7) (h3 : YZ = 8)

noncomputable def G_projections (XYZ_triangle : {X Y Z : Type}) := 
  {P Q R : Type} (hP : G ⟂ YZ) (hQ : G ⟂ XZ) (hR : G ⟂ XY)

theorem GP_plus_GQ_plus_GR (XYZ_triangle : {X Y Z : Type}) 
  (XY XZ YZ : ℝ) (h1 : XY = 5) (h2 : XZ = 7) (h3 : YZ = 8) 
  (P Q R : Type) (hP : G ⟂ YZ) (hQ : G ⟂ XZ) (hR : G ⟂ XY) : 
  GP + GQ + GR = (131 * real.sqrt 3) / 42 :=
sorry

end GP_plus_GQ_plus_GR_l649_649915


namespace charity_event_revenue_l649_649338

theorem charity_event_revenue :
  ∃ (f t p : ℕ), f + t = 190 ∧ f * p + t * (p / 3) = 2871 ∧ f * p = 1900 :=
by
  sorry

end charity_event_revenue_l649_649338


namespace sin_leq_one_l649_649618

theorem sin_leq_one (x : ℝ) : sin x ≤ 1 := 
sorry

end sin_leq_one_l649_649618


namespace profit_percentage_is_ten_l649_649698

-- Definitions based on conditions
def cost_price := 500
def selling_price := 550

-- Defining the profit percentage
def profit := selling_price - cost_price
def profit_percentage := (profit / cost_price) * 100

-- The proof that the profit percentage is 10
theorem profit_percentage_is_ten : profit_percentage = 10 :=
by
  -- Using the definitions given
  sorry

end profit_percentage_is_ten_l649_649698


namespace AuntWangProfit_l649_649375

-- Definitions of the problem context
def shirtsA := 30
def shirtsB := 50
def costA (x : ℝ) := x
def costB (y : ℝ) := y
def sellPrice (x y : ℝ) := (x + y) / 2

-- Assumption that x > y
variables {x y : ℝ} (h : x > y)

-- Definition of the profit calculation
def profitA (x y : ℝ) := shirtsA * (sellPrice x y - costA x)
def profitB (x y : ℝ) := shirtsB * (sellPrice x y - costB y)
def totalProfit (x y : ℝ) := profitA x y + profitB x y

-- Theorem statement
theorem AuntWangProfit (x y : ℝ) (h : x > y) : totalProfit x y > 0 :=
by sorry

end AuntWangProfit_l649_649375


namespace find_x_l649_649387

def S (x : ℝ) : List ℝ := List.range 11 |>.map (fun n => x ^ n)

def B (S : List ℝ) : List ℝ := 
  (List.zipWith (fun a b => (2 * a + 3 * b) / 5) (S.init) (S.tail!))

def iterateB (S : List ℝ) (m : ℕ) : List ℝ :=
  Nat.iterate B m S

theorem find_x (x : ℝ) (h : x > 0) :
  iterateB (S x) 10 = [1 / 5^5] ↔ x = (Real.sqrt 5 - 2) / 3 :=
sorry

end find_x_l649_649387


namespace percent_divisible_by_4_and_6_l649_649311

theorem percent_divisible_by_4_and_6 : 
  {n : ℕ | n ≤ 200 ∧ n % 4 = 0 ∧ n % 6 = 0}.card / 200 * 100 = 8 := 
by
  sorry

end percent_divisible_by_4_and_6_l649_649311


namespace continuity_at_minus_two_discontinuity_at_zero_l649_649918

noncomputable def f (x : ℝ) : ℝ :=
  if x < -2 then x + 1.5
  else if x < 0 then 1 / x
  else 2 * x

theorem continuity_at_minus_two : continuous_at f (-2) := by
  sorry

theorem discontinuity_at_zero : ¬ continuous_at f 0 := by
  sorry

end continuity_at_minus_two_discontinuity_at_zero_l649_649918


namespace distinct_midpoints_at_least_2n_minus_3_l649_649067

structure Point where
  x : ℝ
  y : ℝ

def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

theorem distinct_midpoints_at_least_2n_minus_3 (n : ℕ) (n_ge_3 : 3 ≤ n) (points : Fin n → Point) :
  ∃ S : Finset Point, (∀ i j : Fin n, i ≠ j → midpoint (points i) (points j) ∈ S) ∧ S.card ≥ 2 * n - 3 :=
  sorry

end distinct_midpoints_at_least_2n_minus_3_l649_649067


namespace cross_section_area_is_correct_l649_649352

noncomputable def maximal_cross_section_area : ℝ := 
  let prism_base := {A := (6, 4), B := (-6, 4), C := (-6, -4), D := (6, -4)}
  let intersection_points := {
    E := (6, 4, 0),
    F := (-6, 4, -20/3),
    G := (-6, -4, -80/3),
    H := (6, -4, 64/3)
  }
  let EF := (-12, 0, -20/3)
  let EH := (0, -8, 64/3)
  let cross_product := (512, 64, 96)
  let magnitude := real.sqrt (512^2 + 64^2 + 96^2)
  2 * magnitude / 2

theorem cross_section_area_is_correct : 
  maximal_cross_section_area = 528 := 
by
  sorry

end cross_section_area_is_correct_l649_649352


namespace xy_value_l649_649133

theorem xy_value (x y : ℝ) (h : x * (x + 2 * y) = x^2 + 10) : x * y = 5 :=
by
  sorry

end xy_value_l649_649133


namespace sum_T_eq_zero_l649_649927

def T (n : ℕ) : ℤ :=
  ∑ k in Finset.range n, (-1) ^ (⌊(k + 3 : ℚ) / 3⌋₊) * (k + 1)

theorem sum_T_eq_zero :
  T 18 + T 36 + T 45 = 0 := 
sorry

end sum_T_eq_zero_l649_649927


namespace sum_of_squares_l649_649205

theorem sum_of_squares (b j s : ℕ) 
  (h1 : 3 * b + 2 * j + 4 * s = 70)
  (h2 : 4 * b + 3 * j + 2 * s = 88) : 
  b^2 + j^2 + s^2 = 405 := 
sorry

end sum_of_squares_l649_649205


namespace distinct_circles_in_parallelogram_l649_649181

-- Define the problem context
variables {V : Type} [EuclideanSpace V] (A B C D : V)
def is_parallelogram (A B C D : V) : Prop :=
  (∃ AD BC : V, 
    (AD = A - D ∧ BC = B - C) ∧ 
    (AD + BC = 0))

theorem distinct_circles_in_parallelogram 
  (P : Set V)
  (hP : is_parallelogram A B C D)
  (hV : P = {A, B, C, D}) :
  (∃ n : ℕ, n = 5) := sorry

end distinct_circles_in_parallelogram_l649_649181


namespace digits_sum_1985_1986_l649_649561

theorem digits_sum_1985_1986 :
  let p := ⌊log 10 (2 ^ 1985)⌋ + 1
  let q := ⌊log 10 (2 ^ 1986)⌋ + 1
  in p + q = 1986 :=
by
  sorry

end digits_sum_1985_1986_l649_649561


namespace solve_for_x_l649_649497

theorem solve_for_x (x : ℝ) (h : 16^(x + 2) = 496 + 16^x) : 
  x = Real.logb 16 (496 / 255) :=
sorry

end solve_for_x_l649_649497


namespace calculate_length_EF_l649_649905

noncomputable def length_EF : ℝ :=
  let AB := 8
  let BC := 10
  let Area_ABCD := AB * BC
  let Area_DEF := Area_ABCD / 3
  let DE_sq := (2 * Area_DEF) 
  let EF_sq := DE_sq + DE_sq
  real.sqrt EF_sq / 3

theorem calculate_length_EF :
  length_EF = 16 * real.sqrt 15 / 3 := by
  sorry

end calculate_length_EF_l649_649905


namespace sum_infinite_series_result_l649_649960

noncomputable def sum_infinite_series (x : ℝ) (h : 1 < x) : ℝ :=
  ∑' n, 1 / (x^(3^n) - x^(-3^n))

theorem sum_infinite_series_result (x : ℝ) (h : 1 < x) :
  sum_infinite_series x h = 1 / (x - 1) :=
sorry

end sum_infinite_series_result_l649_649960


namespace proof_inequality_a_half_proof_inequality_a_pos_l649_649822

variable (a x : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a + 1/a) * x + 1

theorem proof_inequality_a_half : 
  (∀ x : ℝ, f x (1/2) ≤ 0 ↔ (1/2 : ℝ) ≤ x ∧ x ≤ 2) :=
by 
  sorry

theorem proof_inequality_a_pos (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, f x a ≤ 0 ↔ 
    ((a = 1 → x = 1) ∧ 
    (a > 1 → (1/a : ℝ) ≤ x ∧ x ≤ a) ∧ 
    ((0 < a ∧ a < 1) → a ≤ x ∧ x ≤ 1/a))) :=
by 
  sorry

end proof_inequality_a_half_proof_inequality_a_pos_l649_649822


namespace positive_terms_count_l649_649770

def sequence (n : ℕ) : ℝ := Float.cos (10.0^(n-1) * Float.pi / 180.0)

theorem positive_terms_count :
  (List.filter (λ n, sequence n > 0) (List.range' 1 100)).length = 99 :=
sorry

end positive_terms_count_l649_649770


namespace abs_diff_count_odd_even_S_le_100_l649_649056

def τ (n : ℕ) : ℕ := 
  finset.card (finset.filter (λ d, n % d = 0) (finset.range (n+1)))

def S (n : ℕ) := finset.sum (finset.range (n+1)) τ

def is_odd (n : ℕ) := n % 2 = 1

def count_odd_S_le_100 := finset.card (finset.filter (λ n, is_odd (S n)) (finset.range (101)))
def count_even_S_le_100 := finset.card (finset.filter (λ n, ¬is_odd (S n)) (finset.range (101)))

theorem abs_diff_count_odd_even_S_le_100 : 
  |count_odd_S_le_100 - count_even_S_le_100| = 10 :=
by 
  sorry

end abs_diff_count_odd_even_S_le_100_l649_649056


namespace varphi_range_l649_649472

noncomputable def f (x ω φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) + 1

theorem varphi_range (ω > 0)  (|φ| ≤ Real.pi / 2) 
  (∀ x ∈ Set.Ioo (Real.pi / 24) (Real.pi / 3), f x ω φ > 2)
  (∀ x, f x ω φ = 3 → (∃ d, d = Real.pi)) : 
  Set.Icc (Real.pi / 12) (Real.pi / 6) = 
    { φ | (Real.pi / 12 ≤ φ) ∧ (φ ≤ Real.pi / 6) } := sorry

end varphi_range_l649_649472


namespace positive_terms_count_l649_649782

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then real.cos (1 * real.pi / 180) else real.cos (nat.pow 10 (n - 1) * real.pi / 180)

theorem positive_terms_count : finset.card (finset.filter (λ n, 0 < sequence n) (finset.range 100)) = 99 :=
by sorry

end positive_terms_count_l649_649782


namespace find_palindrome_satisfying_conditions_l649_649044

-- Define a palindrome number condition
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 2 in s = s.reverse

-- Define 8-digit number condition
def is_eight_digit (n : ℕ) : Prop :=
  n >= 10^7 ∧ n < 10^8

-- Define digits composed of 0 and 1 condition
def is_composed_of_0_and_1 (n : ℕ) : Prop :=
  ∀ d ∈ n.to_digits 2, d = 0 ∨ d = 1

-- Define divisibility by 3 condition
def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Define prime divisor composition condition
def has_prime_divisors_with_digits_1_3_and_percent (n : ℕ) : Prop :=
  ∀ p ∈ n.factors, p.to_digits 10 ⊆ [1, 3, %].map (λ x, x.to_digit_var_constant x.nat)

-- Final statement combining all conditions
theorem find_palindrome_satisfying_conditions :
  ∃ n, is_palindrome n ∧ 
       is_eight_digit n ∧ 
       is_composed_of_0_and_1 n ∧ 
       is_multiple_of_3 n ∧ 
       has_prime_divisors_with_digits_1_3_and_percent n ∧ 
       n = 10111101 :=
sorry

end find_palindrome_satisfying_conditions_l649_649044


namespace sum_exponents_eq_27_l649_649634

theorem sum_exponents_eq_27:
  ∃ (s : ℕ) (m : Fin s → ℕ) (b : Fin s → ℤ),
  (∀ i j, i < j → m i > m j) ∧
  (∀ k, b k = 1 ∨ b k = -1) ∧
  (∑ i, b i * 3^(m i) = 1729) ∧
  (∑ i, m i = 27) :=
sorry

end sum_exponents_eq_27_l649_649634


namespace positive_terms_count_l649_649781

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then real.cos (1 * real.pi / 180) else real.cos (nat.pow 10 (n - 1) * real.pi / 180)

theorem positive_terms_count : finset.card (finset.filter (λ n, 0 < sequence n) (finset.range 100)) = 99 :=
by sorry

end positive_terms_count_l649_649781


namespace simplify_abs_value_l649_649229

theorem simplify_abs_value : abs (- 5 ^ 2 + 6) = 19 := by
  sorry

end simplify_abs_value_l649_649229


namespace leah_saving_days_l649_649542

noncomputable def days_leah_saved : ℕ :=
  let josiah_savings := 0.25 * 24
  let total_savings_leah_megan := 28 - josiah_savings
  let megan_savings := 1.0 * 12
  let total_leah_savings := total_savings_leah_megan - megan_savings
  total_leah_savings / 0.50

theorem leah_saving_days (h : days_leah_saved = 20) : days_leah_saved = 20 :=
by
  sorry

end leah_saving_days_l649_649542


namespace prob_of_third_grade_selection_l649_649711

theorem prob_of_third_grade_selection (total_parts first_grade_parts second_grade_parts third_grade_parts sample_size : ℕ)
  (h_total: total_parts = 120)
  (h_first: first_grade_parts = 24)
  (h_second: second_grade_parts = 36)
  (h_third: third_grade_parts = 60)
  (h_sample: sample_size = 20) :
  (sample_size : ℚ) / total_parts = 1 / 6 :=
by {
  have h_total_eq : (total_parts : ℚ) = 120 := by exact congr_arg coe h_total,
  have h_sample_eq : (sample_size : ℚ) = 20 := by exact congr_arg coe h_sample,
  rw [h_total_eq, h_sample_eq],
  norm_num,
  sorry
}

end prob_of_third_grade_selection_l649_649711


namespace sarah_reads_40_words_per_minute_l649_649223

-- Define the conditions as constants
def words_per_page := 100
def pages_per_book := 80
def reading_hours := 20
def number_of_books := 6

-- Convert hours to minutes
def total_reading_time := reading_hours * 60

-- Calculate the total number of words in one book
def words_per_book := words_per_page * pages_per_book

-- Calculate the total number of words in all books
def total_words := words_per_book * number_of_books

-- Define the words read per minute
def words_per_minute := total_words / total_reading_time

-- Theorem statement: Sarah reads 40 words per minute
theorem sarah_reads_40_words_per_minute : words_per_minute = 40 :=
by
  sorry

end sarah_reads_40_words_per_minute_l649_649223


namespace inradius_right_triangle_l649_649030

-- Definitions of triangle sides
def a := 12
def b := 35
def c := 37

-- Theorem to be proved: the inradius of the right triangle with sides 12, 35, and 37 is 5.
theorem inradius_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  let A := 1/2 * a * b,
      s := (a + b + c) / 2 in
  s = 42 -> A = 210 -> ∃ r, r = 5 :=
by sorry

end inradius_right_triangle_l649_649030


namespace find_vector_b_l649_649929

def a : ℝ^3 := ⟨3, 2, 4⟩
def b : ℝ^3 := ⟨2, 3, 3/2⟩

theorem find_vector_b : (a • b = 14) ∧ (a × b = ⟨-15, 5, 1⟩) :=
by
  sorry

end find_vector_b_l649_649929


namespace blue_point_count_equality_l649_649556

variable {n : ℕ}

def is_red {x y : ℕ} (colored : (ℕ × ℕ) → Bool) : Prop :=
  ∀ (x' y' : ℕ), x' ≤ x → y' ≤ y → colored (x', y') = tt → colored (x, y) = tt

theorem blue_point_count_equality (n : ℕ) (colored : (ℕ × ℕ) → Bool)
  (h0 : ∀ (x y : ℕ), x + y ≤ n → (colored (x, y) = tt ∨ colored (x, y) = ff))
  (h1 : ∀ (x y : ℕ), colored (x, y) = tt → is_red colored) :
  let A := (finset.range (n + 1)).val.sum (λ x, if colored (x, _))  -- This should represent A in terms of Finset and conditions
  let B := (finset.range (n + 1)).val.sum (λ y, if colored (_, y))  -- This should represent B in terms of Finset and conditions
in A = B :=
by
  sorry

end blue_point_count_equality_l649_649556


namespace infinite_series_sum_l649_649936

theorem infinite_series_sum (x : ℝ) (h : x > 1) :
  ∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (-(3 ^ n))) = 1 / (x - 1) :=
sorry

end infinite_series_sum_l649_649936


namespace number_of_ways_to_assign_positions_l649_649085

-- Definitions based on conditions
def members : Finset String := {"Alice", "Bob", "Carol", "Dave"}

def positions : Finset String := {"President", "Vice President", "Secretary", "Treasurer"}

-- Theorem statement
theorem number_of_ways_to_assign_positions : Finset.card (members.permutations) = 24 :=
by
  have h : members = {"Alice", "Bob", "Carol", "Dave"} := rfl
  have p : Finset.card members = 4 := by simp [h]
  -- Given 4 distinct positions and 4 members
  have pos : positions = {"President", "Vice President", "Secretary", "Treasurer"} := rfl
  have pos_card : Finset.card positions = 4 := by simp [pos]
  -- Calculating the number of ways to assign positions using permutations
  exact Finset.card_permutations_eq_card_factorial members

end number_of_ways_to_assign_positions_l649_649085


namespace horse_food_per_day_l649_649373

-- Define the conditions
variable (num_sheep : ℕ) (num_horses : ℕ) (total_food : ℕ)
variable (ratio_sheep_horses : ℕ × ℕ)
variable (required_food_per_horse : ℕ)

-- Given conditions
def StewartFarmConditions :=
  ratio_sheep_horses = (5, 7) ∧
  num_sheep = 40 ∧
  total_food = 12880

-- Theorem statement
theorem horse_food_per_day (h : StewartFarmConditions) :
  required_food_per_horse = 230 :=
sorry

end horse_food_per_day_l649_649373


namespace solve_for_y_l649_649490

theorem solve_for_y : (2^Real.logb 2 7 = 6 * y + 3) -> y = 2/3 :=
by
  sorry

end solve_for_y_l649_649490


namespace cannot_return_to_initial_after_1999_jumps_l649_649989

-- Definitions for initial positions
inductive Insect
| grasshopper
| locust
| cricket

inductive Position
| left
| middle
| right

-- Initial positions of insects
def initial_positions : Insect → Position
| Insect.grasshopper := Position.left
| Insect.locust := Position.middle
| Insect.cricket := Position.right

-- Function to determine if the insects are in initial order
def in_initial_order (positions : Insect → Position) : Prop :=
positions Insect.grasshopper = Position.left ∧
positions Insect.locust = Position.middle ∧
positions Insect.cricket = Position.right

-- Definition and states after certain jumps
def jump (positions : Insect → Position) : Insect → Position := sorry -- Define jumping function here in practice

-- After 1999 jumps
def positions_after_jumps (positions : Insect → Position) : Nat → Insect → Position
| 0 := positions
| (n + 1) := jump (positions_after_jumps positions n)

-- Main theorem to prove
theorem cannot_return_to_initial_after_1999_jumps :
  ∀ initial_positions : Insect → Position,
    in_initial_order initial_positions →
    ¬ in_initial_order (positions_after_jumps initial_positions 1999) :=
sorry

end cannot_return_to_initial_after_1999_jumps_l649_649989


namespace text_messages_in_march_l649_649538

theorem text_messages_in_march
  (nov_texts : ℕ)
  (dec_texts : ℕ)
  (jan_texts : ℕ)
  (feb_texts : ℕ)
  (double_pattern : ∀ n m : ℕ, m = 2 * n)
  (h_nov : nov_texts = 1)
  (h_dec : dec_texts = 2 * nov_texts)
  (h_jan : jan_texts = 2 * dec_texts)
  (h_feb : feb_texts = 2 * jan_texts) : 
  ∃ mar_texts : ℕ, mar_texts = 2 * feb_texts ∧ mar_texts = 16 := 
by
  sorry

end text_messages_in_march_l649_649538


namespace find_solutions_l649_649799

-- Defining the system of equations as conditions
def cond1 (a b : ℕ) := a * b + 2 * a - b = 58
def cond2 (b c : ℕ) := b * c + 4 * b + 2 * c = 300
def cond3 (c d : ℕ) := c * d - 6 * c + 4 * d = 101

-- Theorem to prove the solutions satisfy the system of equations
theorem find_solutions (a b c d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0):
  cond1 a b ∧ cond2 b c ∧ cond3 c d ↔ (a, b, c, d) ∈ [(3, 26, 7, 13), (15, 2, 73, 7)] :=
by sorry

end find_solutions_l649_649799


namespace num_positive_cos_terms_l649_649758

def sequence (n : ℕ) : ℝ := Real.cos (10^(n-1) * Real.pi / 180)

theorem num_positive_cos_terms : (Finset.card (Finset.filter (λ n, 0 < sequence n) (Finset.range 100))) = 99 := 
sorry

end num_positive_cos_terms_l649_649758


namespace num_positive_terms_l649_649771

noncomputable def seq (n : ℕ) : ℝ := float.cos (10^((n - 1).to_real))

theorem num_positive_terms : fin 100 → seq 100 99 :=
sorry

end num_positive_terms_l649_649771


namespace complex_products_and_quotients_l649_649832

noncomputable def z1 : ℂ := 2 - 3 * Complex.i
noncomputable def z2 : ℂ := (15 - 5 * Complex.i) / ((2 + Complex.i) ^ 2)

theorem complex_products_and_quotients :
  (z1 * z2 = -7 - 9 * Complex.i) ∧ 
  (z1 / z2 = 11 / 10 + (3 / 10) * Complex.i) :=
by
  sorry -- proof not required

end complex_products_and_quotients_l649_649832


namespace geometry_proof_problem_l649_649596

variables (m n : Line) (α β : Plane)

def parallel_planes (α β : Plane) := α.parallel β
def parallel_lines (l1 l2 : Line) := l1.parallel l2
def perpendicular_line_plane (l : Line) (p : Plane) := l.perpendicular p
def line_in_plane (l : Line) (p : Plane) := l ⊆ p
def plane_intersection (α β : Plane) : Line := α.intersection β

theorem geometry_proof_problem :
  (parallel_planes α β → (line_in_plane m α → parallel_lines m β)) ∧
  (parallel_lines m β → (line_in_plane m α → (plane_intersection α β = n → parallel_lines m n))) :=
by sorry

end geometry_proof_problem_l649_649596


namespace num_different_lists_of_four_draws_l649_649678

-- Conditions
def balls := fin 15
def draws := fin 4

-- Problem statement
def num_possible_lists (n : ℕ) : ℕ :=
  n ^ (draws.val + 1)

theorem num_different_lists_of_four_draws :
  num_possible_lists 15 = 50625 :=
by
  -- Proof is omitted
  sorry

end num_different_lists_of_four_draws_l649_649678


namespace sqrt_seven_to_six_power_eq_343_l649_649585

theorem sqrt_seven_to_six_power_eq_343 : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_six_power_eq_343_l649_649585


namespace sqrt_defined_real_l649_649885

theorem sqrt_defined_real (a : ℝ) : (∃ x : ℝ, x = sqrt (a + 1)) ↔ (a ≥ -1) :=
by
  sorry

end sqrt_defined_real_l649_649885


namespace infinite_sum_identity_l649_649941

theorem infinite_sum_identity (x : ℝ) (h : x > 1) :
  (∑' n : ℕ, 1 / (x^(3^n) - x^(-3^n))) = 1 / (x - 1) :=
sorry

end infinite_sum_identity_l649_649941


namespace positive_terms_count_l649_649768

def sequence (n : ℕ) : ℝ := Float.cos (10.0^(n-1) * Float.pi / 180.0)

theorem positive_terms_count :
  (List.filter (λ n, sequence n > 0) (List.range' 1 100)).length = 99 :=
sorry

end positive_terms_count_l649_649768


namespace max_edges_triangle_free_max_edges_no_complete_K4_l649_649660

noncomputable def max_edges_no_triangles (n : ℕ) := 
  if n = 30 then 225 else 0  -- Assuming n = 30 for our specific problem case

noncomputable def max_edges_no_K4 (n : ℕ) :=
  if n = 30 then 300 else 0  -- Assuming n = 30 for our specific problem case

theorem max_edges_triangle_free :
  ∀ (G : Type) [graph G], G.vertices = 30 → (∀ (H : subgraph G), ¬triangle H) → 
  G.edges ≤ max_edges_no_triangles 30 := 
sorry

theorem max_edges_no_complete_K4 :
  ∀ (G : Type) [graph G], G.vertices = 30 → (∀ (H : subgraph G), ¬complete_graph_of_size 4 H) →
  G.edges ≤ max_edges_no_K4 30 :=
sorry

end max_edges_triangle_free_max_edges_no_complete_K4_l649_649660


namespace non_acute_angles_in_quadrilaterals_l649_649451

-- Define a triangle and intersection points
variables {A B C S : Type} [Triangle A B C] (AS : LineSegment A S) (BS : LineSegment B S) (CS : LineSegment C S)
variables {A1 B1 C1 : Type} [IntersectionPoint AS B C A1] [IntersectionPoint BS C A B1] [IntersectionPoint CS A B C1]

-- Declare the main theorem statement
theorem non_acute_angles_in_quadrilaterals :
  ∃ (quad : Quadrilateral), 
    ((quad = Quadrilateral A B1 S C1 ∨ quad = Quadrilateral C1 S A1 B ∨ quad = Quadrilateral A1 S B1 C) ∧
      (both_angles_non_acute (vertex1 quad) (vertex2 quad))) := 
sorry

end non_acute_angles_in_quadrilaterals_l649_649451


namespace find_theta_sum_exponential_l649_649035

theorem find_theta_sum_exponential :
  ∃ r : ℝ, (complex.exp (11 * real.pi * complex.I / 80) +
            complex.exp (31 * real.pi * complex.I / 80) +
            complex.exp (51 * real.pi * complex.I / 80) +
            complex.exp (71 * real.pi * complex.I / 80) +
            complex.exp (91 * real.pi * complex.I / 80)) =
            r * complex.exp (51 * real.pi * complex.I / 80) :=
sorry

end find_theta_sum_exponential_l649_649035


namespace jonas_socks_solution_l649_649171

theorem jonas_socks_solution (p_s p_h n_p n_t n : ℕ) (h_ps : p_s = 20) (h_ph : p_h = 5) (h_np : n_p = 10) (h_nt : n_t = 10) :
  2 * (p_s * 2 + p_h * 2 + n_p + n_t) = 2 * (p_s * 2 + p_h * 2 + n_p + n_t + n * 2) :=
by
  -- skipping the proof part
  sorry

end jonas_socks_solution_l649_649171


namespace base7_to_base10_245_l649_649344

theorem base7_to_base10_245 : (2 * 7^2 + 4 * 7^1 + 5 * 7^0) = 131 := by
  sorry

end base7_to_base10_245_l649_649344


namespace hackathon_end_time_l649_649691

def start_time := Time.mk 12 0 -- Time at noon
def total_duration := 1440  -- Duration in minutes

theorem hackathon_end_time : 
  (start_time.add_minutes total_duration).to_24_hour_format = Time.mk 12 0 :=
by
  -- Proof skipped
  sorry

end hackathon_end_time_l649_649691


namespace num_five_digit_positive_integers_with_30_permutations_equals_9720_l649_649744

noncomputable def multiset_permutations (n : ℕ) (freqs : List ℕ) : ℕ :=
  Nat.factorial n / (List.prod (List.map Nat.factorial freqs))

noncomputable def count_five_digit_integers_with_30_permutations : ℕ :=
  let no_zero_case : ℕ :=
    let choices := Nat.choose 9 3 in
    choices * 3 * 30
  let one_zero_case : ℕ :=
    let choices := Nat.choose 9 2 in
    choices * (30 - Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2))
  let two_zero_case : ℕ :=
    let choices := Nat.choose 9 2 in
    choices * 2 * 12
  no_zero_case + one_zero_case + two_zero_case

theorem num_five_digit_positive_integers_with_30_permutations_equals_9720 :
  count_five_digit_integers_with_30_permutations = 9720 := by
  sorry

end num_five_digit_positive_integers_with_30_permutations_equals_9720_l649_649744


namespace cos_450_eq_0_l649_649741

theorem cos_450_eq_0 : Real.cos (450 * Real.pi / 180) = 0 := by
  -- Angle equivalence: 450 degrees is equivalent to 90 degrees on the unit circle
  have angle_eq : (450 : Real) * Real.pi / 180 = (90 : Real) * Real.pi / 180 := by
    calc
      (450 : Real) * Real.pi / 180
        = (450 / 180) * Real.pi : by rw [mul_div_assoc]
        = (5 * 90 / 180) * Real.pi : by norm_num
        = (5 * 90 / (2 * 90)) * Real.pi : by norm_num
        = (5 / 2) * Real.pi : by norm_num

  -- Now use this equivalence: cos(450 degrees) = cos(90 degrees)
  have cos_eq : Real.cos (450 * Real.pi / 180) = Real.cos (90 * Real.pi / 180) := by
    rw [angle_eq]

  -- Using the fact that cos(90 degrees) = 0
  have cos_90 : Real.cos (90 * Real.pi / 180) = 0 := by
    -- This step can use a known trigonometric fact from mathlib
    exact Real.cos_pi_div_two

  -- Therefore
  rw [cos_eq, cos_90]
  exact rfl

end cos_450_eq_0_l649_649741


namespace determine_k_circle_l649_649394

theorem determine_k_circle (k : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 14*y - k = 0) ∧ ((∀ x y : ℝ, (x + 4)^2 + (y + 7)^2 = 25) ↔ k = -40) :=
by
  sorry

end determine_k_circle_l649_649394


namespace problem_statement_l649_649836

variables {m : ℝ}
def p := m > 2
def q := 1 < m ∧ m < 3

theorem problem_statement (h1 : p ∨ q) (h2 : ¬ (p ∧ q)) : m ≥ 3 ∨ (1 < m ∧ m ≤ 2) :=
sorry

end problem_statement_l649_649836


namespace abs_inequality_solution_l649_649231

theorem abs_inequality_solution (x : ℝ) :
  abs (x - 3) + abs (x + 4) < 8 ↔ 
  (x ∈ Set.Ioo (-9 / 2 : ℝ) -4 ∨ x ∈ Set.Ico (-4 : ℝ) 3 ∨ x ∈ Set.Ico (3 : ℝ) (7 / 2)) := sorry

end abs_inequality_solution_l649_649231


namespace minimum_weights_l649_649000

/- 
Ana has an iron material of mass 20.2 kg.
Bilyana agrees to make n weights such that each weight is at least 10g.
Determine the smallest possible value of n for which Ana would always be able to determine the mass of any material (the mass can be any real number between 0 and 20.2 kg) with an error of at most 10g.
-/
theorem minimum_weights (n : ℕ) : 
  (∀ (weights : Fin n → ℝ),
     (∀ i, weights i ≥ 0.01) →
     (∀ m : ℝ, 0 ≤ m ∧ m ≤ 20.2 →
       ∃ (errors : Fin n → ℕ), (m - 0.01 ≤ errors.sum * 0.01) ∧ (errors.sum * 0.01 ≤ m + 0.01))) ↔ n = 2020 := 
sorry

end minimum_weights_l649_649000


namespace fraction_work_left_l649_649321

theorem fraction_work_left (A_days B_days : ℕ) (together_days : ℕ) 
  (H_A : A_days = 20) (H_B : B_days = 30) (H_t : together_days = 4) : 
  (1 : ℚ) - (together_days * ((1 : ℚ) / A_days + (1 : ℚ) / B_days)) = 2 / 3 :=
by
  sorry

end fraction_work_left_l649_649321


namespace relationship_among_neg_a_square_neg_a_cube_l649_649091

theorem relationship_among_neg_a_square_neg_a_cube (a : ℝ) (h : -1 < a ∧ a < 0) : (-a > a^2 ∧ a^2 > -a^3) :=
by
  sorry

end relationship_among_neg_a_square_neg_a_cube_l649_649091


namespace simplify_fraction_expression_l649_649186

variable (a b : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variable (h_eq : a^3 - b^3 = a - b)

theorem simplify_fraction_expression : (a / b) + (b / a) + (1 / (a * b)) = 2 := by
  sorry

end simplify_fraction_expression_l649_649186


namespace baker_sold_cakes_l649_649718

theorem baker_sold_cakes (S : ℕ) (h1 : 154 = S + 63) : S = 91 :=
by
  sorry

end baker_sold_cakes_l649_649718


namespace limit_sum_of_roots_as_n_tends_to_infinity_l649_649390

-- Define the sequence of polynomials
def a1 : Polynomial ℝ := 3 * Polynomial.X^2 - Polynomial.C 1 * Polynomial.X
def a2 : Polynomial ℝ := 3 * Polynomial.X^2 - 7 * Polynomial.X + 3
def a (n : ℕ) : Polynomial ℝ :=
  if n = 0 then a1
  else if n = 1 then a2
  else (5 / 2 : ℝ) • a (n - 1) - a (n - 2)

-- Prove the limit of the sum of the roots as n tends to infinity
theorem limit_sum_of_roots_as_n_tends_to_infinity :
    tendsto (fun n => Polynomial.sum_roots (a n)) at_top (𝓝 (13 / 3 : ℝ)) :=
sorry

end limit_sum_of_roots_as_n_tends_to_infinity_l649_649390


namespace num_positive_terms_l649_649776

noncomputable def seq (n : ℕ) : ℝ := float.cos (10^((n - 1).to_real))

theorem num_positive_terms : fin 100 → seq 100 99 :=
sorry

end num_positive_terms_l649_649776


namespace minimum_lambda_sequence_l649_649100

theorem minimum_lambda_sequence {λ : ℝ} (λ_pos : 0 < λ)
    (a : ℕ+ → ℝ)
    (h₁ : a 1 = 1 / 3)
    (h₂ : ∀ n, a (n + 1) = (2 * (a n)^2) / (4 * (a n)^2 - 2 * a n + 1)) :
    (∀ m : ℕ+, ∑ k in Icc 1 m, a k < λ) ↔ λ = 1 := by
  sorry

end minimum_lambda_sequence_l649_649100


namespace ratio_of_profit_shares_l649_649664

theorem ratio_of_profit_shares (P_investment Q_investment : ℕ) (h1 : P_investment = 12000) (h2 : Q_investment = 30000) :
  (P_investment * 5) = (Q_investment * 2) :=
by
  rw [h1, h2]
  norm_num
  sorry

end ratio_of_profit_shares_l649_649664


namespace find_number_l649_649350

theorem find_number 
  (n : ℤ)
  (h1 : n % 7 = 2)
  (h2 : n % 8 = 4)
  (quot_7 : ℤ)
  (quot_8 : ℤ)
  (h3 : n = 7 * quot_7 + 2)
  (h4 : n = 8 * quot_8 + 4)
  (h5 : quot_7 = quot_8 + 7) :
  n = 380 := by
  sorry

end find_number_l649_649350


namespace min_value_sinx_cosx_l649_649804

theorem min_value_sinx_cosx (x : ℝ) : sin x ^ 6 + 2 * cos x ^ 4 ≥ 2 / 3 :=
sorry

end min_value_sinx_cosx_l649_649804


namespace find_f_l649_649197

-- Define the function space and conditions
def func (f : ℕ+ → ℝ) :=
  (∀ m n : ℕ+, f (m * n) = f m + f n) ∧
  (∀ n : ℕ+, f (n + 1) ≥ f n)

-- Define the theorem statement
theorem find_f (f : ℕ+ → ℝ) (hf : func f) : ∀ n : ℕ+, f n = 0 :=
sorry

end find_f_l649_649197


namespace incorrect_radius_circle_M_l649_649476

noncomputable def circle_M_equation := ∀ x y : ℝ, x^2 + y^2 - 8 * x + 6 * y = 0

noncomputable def center_of_circle_M := (4, -3) : ℝ × ℝ

noncomputable def radius_of_circle_M := 5

theorem incorrect_radius_circle_M :
  circle_M_equation = λ x y, (x - 4)^2 + (y + 3)^2 = 25
  → radius_of_circle_M ≠ 25 := by
  sorry

end incorrect_radius_circle_M_l649_649476


namespace maximum_volume_of_pyramid_l649_649242

noncomputable def volume_of_pyramid (S : ℝ) (DO : ℝ) : ℝ :=
  (1 / 3) * S * DO

theorem maximum_volume_of_pyramid :
  ∀ (a b c : ℝ) (h : ℝ)
  (α : ℝ)
  (ha : a = 5)
  (hb : b = 12)
  (hc : c = 13)
  (hα : α = 30)
  (triangle_area : ℝ)
  (triangle_area_eq : triangle_area = (1 / 2) * a * b)
  (DO : ℝ)
  (DO_eq : DO = max (2 * Real.cot α) (max (10 * Real.cot α) (max (3 * Real.cot α) (15 * Real.cot α)))),
  volume_of_pyramid triangle_area DO = 150 * Real.sqrt 3 :=
by
  sorry

end maximum_volume_of_pyramid_l649_649242


namespace nori_gave_crayons_to_mae_l649_649567

-- Definitions as per given conditions
def total_boxes : ℕ := 4
def crayons_per_box : ℕ := 8
def total_crayons : ℕ := total_boxes * crayons_per_box
def remaining_crayons : ℕ := 15
def total_given_away : ℕ := total_crayons - remaining_crayons

-- Question: How many crayons did Nori give to Mae?
-- Let "M" be the number of crayons given to Mae
def M (n : ℕ) := (M + (M + 7) = 17)

theorem nori_gave_crayons_to_mae : M = 5 :=
by
  sorry

end nori_gave_crayons_to_mae_l649_649567


namespace major_axis_length_l649_649333

noncomputable def length_of_major_axis (f1 f2 : ℝ × ℝ) (tangent_y_axis : Bool) (tangent_line_y : ℝ) : ℝ :=
  if f1 = (-Real.sqrt 5, 2) ∧ f2 = (Real.sqrt 5, 2) ∧ tangent_y_axis ∧ tangent_line_y = 1 then 2
  else 0

theorem major_axis_length :
  length_of_major_axis (-Real.sqrt 5, 2) (Real.sqrt 5, 2) true 1 = 2 :=
by
  sorry

end major_axis_length_l649_649333


namespace total_cash_realized_l649_649386

def stockA_sale_proceeds : ℝ := 107.25
def stockA_brokerage_fee_percent : ℝ := 1/4 / 100
def stockB_sale_proceeds : ℝ := 155.40
def stockB_brokerage_fee_percent : ℝ := 1/2 / 100
def stockC_sale_proceeds : ℝ := 203.50
def stockC_brokerage_fee_percent : ℝ := 3/4 / 100

def cash_realized (sale_proceeds : ℝ) (brokerage_fee_percent : ℝ) : ℝ :=
  sale_proceeds - (brokerage_fee_percent * sale_proceeds)

theorem total_cash_realized :
  cash_realized stockA_sale_proceeds stockA_brokerage_fee_percent +
  cash_realized stockB_sale_proceeds stockB_brokerage_fee_percent +
  cash_realized stockC_sale_proceeds stockC_brokerage_fee_percent = 463.578625 := by
  sorry

end total_cash_realized_l649_649386


namespace minimum_value_l649_649416

/-- The minimum value of the expression (x+2)^2 / (y-2) + (y+2)^2 / (x-2)
    for real numbers x > 2 and y > 2 is 50. -/
theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ z, z = (x + 2) ^ 2 / (y - 2) + (y + 2) ^ 2 / (x - 2) ∧ z = 50 :=
sorry

end minimum_value_l649_649416


namespace relationship_among_numbers_l649_649066

theorem relationship_among_numbers :
  let a := 0.7 ^ 2.1
  let b := 0.7 ^ 2.5
  let c := 2.1 ^ 0.7
  b < a ∧ a < c := by
  sorry

end relationship_among_numbers_l649_649066


namespace find_f_x_find_max_min_f_l649_649439

noncomputable def f (x : ℝ) : ℝ := 
  if x = 1 then 0 else 2 / (x - 1)

theorem find_f_x (x : ℝ) (h : f ((x - 1) / (x + 1)) = -x - 1) 
  (h1 : x ≠ 1) : f x = 2 / (x - 1) :=
sorry

theorem find_max_min_f (x : ℝ) (h : 2 ≤ x ∧ x ≤ 6) 
  (h1 : f x = 2 / (x - 1)) : 
  let max_val := f 2,
      min_val := f 6
  in (∀ x ∈ [2,6], f x ≤ max_val ∧ f x ≥ min_val) ∧ max_val = 2 ∧ min_val = 2 / 5 :=
sorry

end find_f_x_find_max_min_f_l649_649439


namespace max_value_of_quadratic_function_l649_649599

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  -2 * x^2 + 4 * x - 18 

theorem max_value_of_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, quadratic_function y ≤ quadratic_function x ∧ quadratic_function x = -16 :=
begin
  sorry
end

end max_value_of_quadratic_function_l649_649599


namespace divisible_by_10101010101_min_six_non_zero_l649_649574

open Nat

theorem divisible_by_10101010101_min_six_non_zero (k : ℕ) (h : 10101010101 ∣ k) : 
  (count_non_zero_digits k ≥ 6) :=
sorry

-- Helper function to count non-zero digits
noncomputable def count_non_zero_digits (n : ℕ) : ℕ :=
((n.to_digits 10).filter (≠ 0)).length

end divisible_by_10101010101_min_six_non_zero_l649_649574


namespace molecular_weight_of_7_moles_of_NH4_2SO4_l649_649721

theorem molecular_weight_of_7_moles_of_NH4_2SO4 :
  let N_weight := 14.01
  let H_weight := 1.01
  let S_weight := 32.07
  let O_weight := 16.00
  let N_atoms := 2
  let H_atoms := 8
  let S_atoms := 1
  let O_atoms := 4
  let moles := 7
  let molecular_weight := (N_weight * N_atoms) + (H_weight * H_atoms) + (S_weight * S_atoms) + (O_weight * O_atoms)
  let total_weight := molecular_weight * moles
  total_weight = 924.19 :=
by
  sorry

end molecular_weight_of_7_moles_of_NH4_2SO4_l649_649721


namespace red_ball_value_l649_649525

theorem red_ball_value (r b g : ℕ) (blue_points green_points : ℕ)
  (h1 : blue_points = 4)
  (h2 : green_points = 5)
  (h3 : b = g)
  (h4 : r^4 * blue_points^b * green_points^g = 16000)
  (h5 : b = 6) :
  r = 1 :=
by
  sorry

end red_ball_value_l649_649525


namespace geometric_progression_common_ratio_l649_649512

theorem geometric_progression_common_ratio (a r : ℝ) (ha : 0 < a) (hr : 0 < r)
  (h : a = a * r + a * r^2 + a * r^3) : r ≈ 0.6823 :=
by
  sorry

end geometric_progression_common_ratio_l649_649512


namespace number_of_factors_l649_649142

theorem number_of_factors :
  ∀ (p q r : ℕ), Nat.Prime p → Nat.Prime q → Nat.Prime r → p ≠ q → q ≠ r → r ≠ p →
  let a := p^3
  let b := q^3
  let c := r^3
  let n := a^3 * b^4 * c^5
  Nat.factors n = 2080 := 
by
  intros p q r hp hq hr hpq hqr hrp
  let a := p^3
  let b := q^3
  let c := r^3
  let n := a^3 * b^4 * c^5
  sorry

end number_of_factors_l649_649142


namespace hundred_d_eq_106_l649_649391

def b : ℕ → ℝ
| 0       := 12 / 37
| (n + 1) := (3 * (b n) ^ 2) - 2

noncomputable def d := 37 / 35

theorem hundred_d_eq_106 :
  100 * d = 106 :=
by
simp [d]
-- Proof steps will be filled here.
sorry

end hundred_d_eq_106_l649_649391


namespace find_sums_l649_649157

def arithmetic_sequence_term_general (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ n, a n = 2 * n - 6

def sum_first_six_terms (a : ℕ → ℤ) : Prop :=
  (a 3 = 0 ∧ (6 * a 1 + (6 * 5) / 2 * (2:ℤ) = 6)) → ∀ n, a n = 2 * n - 6

def b_n (n : ℕ) : ℝ :=
  (sqrt 2) ^ (2 * n - 6)

def T_n (n : ℕ) : ℝ :=
  ((2 ^ n - 1) / 4)

theorem find_sums
  (a : ℕ → ℤ)
  (h1 : a 3 = 0)
  (h2 : (6 * a 1 + (6 * 5) / 2 * (2:ℤ) = 6))
  (n : ℕ) :
  (sum_first_six_terms a) → ∀ (n : ℕ), T_n n = ((2 ^ n - 1) / 4) :=
by {
  sorry
}

end find_sums_l649_649157


namespace int_sol_eq_2_l649_649126

theorem int_sol_eq_2 : 
  (∃ x : ℤ, (x - 3)^(30 - x^2) = 1) → 
  ((30 - x^2 = 0 → false) ∧ 
  (x - 3 = 1 → (x = 4 ∧ 30 - x^2 = 14)) ∧ 
  (x - 3 = -1 → (x = 2 ∧ (30 - x^2) % 2 = 0))) → 
  2 := 
sorry

end int_sol_eq_2_l649_649126


namespace percent_increase_correct_l649_649717

variable (p_initial p_final : ℝ)

theorem percent_increase_correct : p_initial = 25 → p_final = 28 → (p_final - p_initial) / p_initial * 100 = 12 := by
  intros h_initial h_final
  sorry

end percent_increase_correct_l649_649717


namespace rectangle_properties_l649_649255

theorem rectangle_properties (w l : ℝ) (h₁ : l = 4 * w) (h₂ : 2 * l + 2 * w = 200) :
  ∃ A d, A = 1600 ∧ d = 82.46 := 
by {
  sorry
}

end rectangle_properties_l649_649255


namespace not_same_depth_l649_649708

-- Define the properties of the triangular prism
structure TriangularPrism where
  edge_length : ℝ
  material_density : ℝ
  water_density : ℝ

-- Conditions based on the problem
axiom prism_conditions : ∀ P : TriangularPrism, 
  P.edge_length = 1 ∧ P.material_density < P.water_density

-- Define the volumes when floating on the base and on the side
def base_volume (x : ℝ) : ℝ := (Real.sqrt 3 / 4) * x
def side_volume (x : ℝ) : ℝ := x * (1 - x / Real.sqrt 3)

-- Main theorem to prove
theorem not_same_depth (P : TriangularPrism) (x : ℝ) :
  prism_conditions P →
  ¬ (base_volume x = side_volume x ∧ 0 < x ∧ x < Real.sqrt 3 / 2) :=
by sorry

end not_same_depth_l649_649708


namespace max_attendance_is_wed_and_fri_l649_649399

structure TeamMember :=
  (name : String)
  (unavailable_days : List String)

def team : List TeamMember :=
  [ 
    ⟨"Alice", ["Mon", "Thurs", "Fri"]⟩,
    ⟨"Bob", ["Tues", "Wed", "Sat"]⟩,
    ⟨"Cara", ["Mon", "Tues", "Sat"]⟩,
    ⟨"Dave", ["Thurs"]⟩,
    ⟨"Eve", ["Mon", "Tues", "Wed"]⟩
  ]

def days : List String := ["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat"]

def unavailability_counts (day : String) : Nat :=
  team.count (fun member => day ∈ member.unavailable_days)

def attendance_counts (day : String) : Nat :=
  team.length - unavailability_counts day

def max_attendance_days : List String :=
  let max_count := days.map attendance_counts |>.maximum
  days.filter (fun day => attendance_counts day == max_count)

theorem max_attendance_is_wed_and_fri :
  max_attendance_days = ["Wed", "Fri"] :=
by
  sorry

end max_attendance_is_wed_and_fri_l649_649399


namespace calculate_length_EF_l649_649906

noncomputable def length_EF : ℝ :=
  let AB := 8
  let BC := 10
  let Area_ABCD := AB * BC
  let Area_DEF := Area_ABCD / 3
  let DE_sq := (2 * Area_DEF) 
  let EF_sq := DE_sq + DE_sq
  real.sqrt EF_sq / 3

theorem calculate_length_EF :
  length_EF = 16 * real.sqrt 15 / 3 := by
  sorry

end calculate_length_EF_l649_649906


namespace find_a_if_perpendicular_l649_649819

def m (a : ℝ) : ℝ × ℝ := (3, a - 1)
def n (a : ℝ) : ℝ × ℝ := (a, -2)

theorem find_a_if_perpendicular (a : ℝ) (h : (m a).fst * (n a).fst + (m a).snd * (n a).snd = 0) : a = -2 :=
by sorry

end find_a_if_perpendicular_l649_649819


namespace parking_methods_count_l649_649688

theorem parking_methods_count : 
  ∃ (n : ℕ), n = 72 ∧ (∃ (spaces cars slots remainingSlots : ℕ), 
  spaces = 7 ∧ cars = 3 ∧ slots = 1 ∧ remainingSlots = 4 ∧
  ∃ (perm_ways slot_ways : ℕ), perm_ways = 6 ∧ slot_ways = 12 ∧ n = perm_ways * slot_ways) :=
  by
    sorry

end parking_methods_count_l649_649688


namespace tank_full_after_50_minutes_l649_649216

-- Define the conditions as constants
def tank_capacity : ℕ := 850
def pipe_a_rate : ℕ := 40
def pipe_b_rate : ℕ := 30
def pipe_c_rate : ℕ := 20
def cycle_duration : ℕ := 3  -- duration of each cycle in minutes
def net_water_per_cycle : ℕ := pipe_a_rate + pipe_b_rate - pipe_c_rate  -- net liters added per cycle

-- Define the statement to be proved: the tank will be full at exactly 50 minutes
theorem tank_full_after_50_minutes :
  ∀ minutes_elapsed : ℕ, (minutes_elapsed = 50) →
  ((minutes_elapsed / cycle_duration) * net_water_per_cycle = tank_capacity - pipe_c_rate) :=
sorry

end tank_full_after_50_minutes_l649_649216


namespace num_positive_terms_l649_649750

def sequence_cos (n : ℕ) : ℝ :=
  cos (10^(n-1) * π / 180)

theorem num_positive_terms : (finset.filter (λ n : ℕ, 0 < sequence_cos n) (finset.range 100)).card = 99 :=
by
  sorry

end num_positive_terms_l649_649750


namespace no_real_solution_l649_649029

-- Define the given equation as a function
def equation (x y : ℝ) : ℝ := 3 * x^2 + 5 * y^2 - 9 * x - 20 * y + 30 + 4 * x * y

-- State that the equation equals zero has no real solution.
theorem no_real_solution : ∀ x y : ℝ, equation x y ≠ 0 :=
by sorry

end no_real_solution_l649_649029


namespace heaviest_difference_lightest_total_weight_total_profit_l649_649368

-- Conditions for the weights and their differences
def standard_weight : ℝ := 10
def weight_differences : List ℝ := [-0.2, -0.1, 0, 0.1, 0.2, 0.3]
def number_of_boxes : List ℕ := [12, 3, 3, 7, 15, 10]

-- Additional conditions for the profit calculation
def purchase_price_per_kg : ℝ := 4
def selling_price_per_kg_60 : ℝ := 10
def selling_price_per_kg_40 : ℝ := 1.5
def percentage_sold_full_price : ℝ := 0.6

-- Box differences weighed (in indexed pairs for number of occurrences)
def weighted_boxes := List.zip number_of_boxes weight_differences

-- Part (1) Statement
theorem heaviest_difference_lightest : 
  (standard_weight + 0.3) - (standard_weight - 0.2) = 0.5 := 
sorry

-- Part (2) Statement 
theorem total_weight :
  List.sum (List.map (λ (pair : ℕ × ℝ), (pair.fst * pair.snd)) weighted_boxes) + 50 * standard_weight = 504 :=
sorry

-- Part (3) Statement
theorem total_profit :
  ((50 * standard_weight + List.sum (List.map (λ (pair : ℕ × ℝ), pair.fst * pair.snd)) weighted_boxes) * (selling_price_per_kg_60 * percentage_sold_full_price + selling_price_per_kg_40 * (1 - percentage_sold_full_price))) - (50 * standard_weight * purchase_price_per_kg) = 2721.6 :=
sorry

end heaviest_difference_lightest_total_weight_total_profit_l649_649368


namespace ab_over_10_eq_100_3_l649_649378

noncomputable def double_factorial_odd (n : ℕ) : ℚ :=
(2 * n - 1)!! / (2 * n)!!

theorem ab_over_10_eq_100_3 :
  let S := ∑ i in finset.range (1010 + 1), double_factorial_odd i
  in ∃ a b : ℕ, S = (c : ℚ) / 2^2020 ∧ (b % 2 = 1) ∧ (S = ∑ i in finset.range (1010 + 1), nat.choose (2 * i) i / (2 ^ (2 * i))) ∧ (a = 1003) ∧ (b = 1) ∧ (a * b) / 10 = 100.3 :=
by
  have h1 : S = ∑ i in finset.range (1010 + 1), nat.choose (2 * i) i / (2 ^ (2 * i)),
  sorry
  let a := 1003,
  let b := 1,
  have ha : S = c / 2^2020,
  have hb : b % 2 = 1,
  have h1 : (a * b) / 10 = 100.3,
  finish
  sorry

end ab_over_10_eq_100_3_l649_649378


namespace shaded_area_eq_sixteen_l649_649996

-- Define the points and dimensions
structure Rectangle :=
  (P Q R S : ℝ × ℝ)
  (length width : ℝ)

def PQRS : Rectangle :=
  { P := (0, 0), Q := (4, 0), R := (0, 5), S := (4, 5),
    length := 4, width := 5 }

-- Define conditions
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Given conditions
def PR := distance PQRS.P PQRS.R = 1
def QO := distance PQRS.Q (4, 1) = 1
def OS := distance (4, 1) PQRS.S = 1

-- Theorem to prove
theorem shaded_area_eq_sixteen : PR ∧ QO ∧ OS → ∃ (shaded_area : ℝ), shaded_area = 16 :=
begin
  sorry  -- Proof steps would go here
end

end shaded_area_eq_sixteen_l649_649996


namespace range_of_m_l649_649453

open Real

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m
def q (m : ℝ) : Prop := (2 - m) > 0

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) → 1 ≤ m ∧ m < 2 :=
by
  sorry

end range_of_m_l649_649453


namespace value_of_x4_plus_inv_x4_l649_649143

theorem value_of_x4_plus_inv_x4 (x : ℝ) (h : x^2 + 1 / x^2 = 6) : x^4 + 1 / x^4 = 34 := 
by
  sorry

end value_of_x4_plus_inv_x4_l649_649143


namespace bike_price_l649_649564

-- Definitions of the conditions
def maria_savings : ℕ := 120
def mother_offer : ℕ := 250
def amount_needed : ℕ := 230

-- Theorem statement
theorem bike_price (maria_savings mother_offer amount_needed : ℕ) : 
  maria_savings + mother_offer + amount_needed = 600 := 
by
  -- Sorry is used here to skip the actual proof steps
  sorry

end bike_price_l649_649564


namespace largest_possible_a_l649_649930

theorem largest_possible_a (a b c d : ℕ) (ha : a < 2 * b) (hb : b < 3 * c) (hc : c < 4 * d) (hd : d < 100) : 
  a ≤ 2367 :=
sorry

end largest_possible_a_l649_649930


namespace count_ordered_pairs_l649_649423

theorem count_ordered_pairs :
  let pairs := { (x, y) : ℕ × ℕ | 0 < y ∧ y < x ∧ x ≤ 150 ∧ (x % y = 0) ∧ ((x + 2) % (y + 2) = 0) } in
  pairs.to_finset.card = 82 :=
begin
  sorry
end

end count_ordered_pairs_l649_649423


namespace height_percentage_increase_l649_649495

theorem height_percentage_increase (B A : ℝ) (h : A = B - 0.3 * B) : 
  ((B - A) / A) * 100 = 42.857 :=
by
  sorry

end height_percentage_increase_l649_649495


namespace prob_two_segments_same_length_l649_649971

namespace hexagon_prob

noncomputable def prob_same_length : ℚ :=
  let total_elements : ℕ := 15
  let sides : ℕ := 6
  let diagonals : ℕ := 9
  (sides / total_elements) * ((sides - 1) / (total_elements - 1)) + (diagonals / total_elements) * ((diagonals - 1) / (total_elements - 1))

theorem prob_two_segments_same_length : prob_same_length = 17 / 35 :=
by
  sorry

end hexagon_prob

end prob_two_segments_same_length_l649_649971


namespace percent_area_square_in_rectangle_l649_649701

theorem percent_area_square_in_rectangle (s : ℝ) : 
  let width_rect := 3 * s,
      length_rect := 4.5 * s
  in ((s ^ 2) / (width_rect * length_rect)) * 100 = 7.41 := 
by
  sorry

end percent_area_square_in_rectangle_l649_649701


namespace problem_l649_649073

noncomputable def a_seq : ℕ → ℝ
| 0 := 0
| (n+1) := a_seq n + (1 / (n * (n + 1))) + 1

def is_arithmetic_sequence : Prop :=
∀ (n : ℕ), (a_seq (n + 1) + (1 / (n + 1)) - (a_seq n + (1 / n))) = 1

def general_formula (n : ℕ) : Prop :=
a_seq n = n - (1 / n)

def S_seq (n : ℕ) : ℝ :=
∑ i in finset.range n, a_seq i / i 

def sum_inequality (n : ℕ) : Prop :=
S_seq n < (n^2) / (n + 1)

theorem problem (n : ℕ) : is_arithmetic_sequence ∧ general_formula n ∧ (∀ n : ℕ, n > 0 → sum_inequality n) :=
by
  split
  · sorry
  · intro n
    sorry
  · intro n hn
    sorry

end problem_l649_649073


namespace necessary_not_sufficient_condition_l649_649083

noncomputable def S (a₁ q : ℝ) : ℝ := a₁ / (1 - q)

theorem necessary_not_sufficient_condition (a₁ q : ℝ) (h₁ : |q| < 1) :
  (a₁ + q = 1) → (S a₁ q = 1) ∧ ¬((S a₁ q = 1) → (a₁ + q = 1)) :=
by
  sorry

end necessary_not_sufficient_condition_l649_649083


namespace sum_series_eq_l649_649955

open Real

theorem sum_series_eq (x : ℝ) (h : 1 < x) : 
  (∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (- (3 ^ n)))) = 1 / (x - 1) :=
sorry

end sum_series_eq_l649_649955


namespace trapezoid_circle_diameter_l649_649714

theorem trapezoid_circle_diameter {a b D : ℝ} 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : D = sqrt (a * b)) :
  diameter_of_circumscribed_circle a b = D := sorry

end trapezoid_circle_diameter_l649_649714


namespace evaluate_expression_l649_649099

theorem evaluate_expression (x y z : ℝ) (hxy : x > y ∧ y > 1) (hz : z > 0) :
  (x^y * y^(x+z)) / (y^(y+z) * x^x) = (x / y)^(y - x) :=
by
  sorry

end evaluate_expression_l649_649099


namespace coefficient_of_1_div_x_l649_649524

open Nat

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℝ :=
  (1 / Real.sqrt x - 3)^n

theorem coefficient_of_1_div_x (x : ℝ) (n : ℕ) (h1 : n ∈ {m | m > 0}) (h2 : binomial_expansion x n = 16) :
  ∃ c : ℝ, c = 54 :=
by
  sorry

end coefficient_of_1_div_x_l649_649524


namespace C_increases_with_n_l649_649889

noncomputable def C (e n R r : ℝ) : ℝ := (e * n) / (R + n * r)

theorem C_increases_with_n (e R r : ℝ) (h_e : 0 < e) (h_R : 0 < R) (h_r : 0 < r) :
  ∀ {n₁ n₂ : ℝ}, 0 < n₁ → n₁ < n₂ → C e n₁ R r < C e n₂ R r :=
by
  sorry

end C_increases_with_n_l649_649889


namespace second_group_half_work_days_l649_649594

theorem second_group_half_work_days 
  (L : ℕ) (work_by_one_lady_per_day : ℝ := 1 / (L * 12)) : ℝ :=
  let second_group_ladies := 2 * L
  let work_done_by_second_group_per_day := 2 / (L * 12)
  let days_for_half_work := 1 / 2 / work_done_by_second_group_per_day
  days_for_half_work = 3

end second_group_half_work_days_l649_649594


namespace find_alpha_l649_649470

noncomputable def line_intersects_curve (α : ℝ) : Prop :=
  let t := ℕ
  let cos_α := ℝ.cos α
  let equation := t ^ 2 - 2 * t * cos_α - 3 = 0
  ∃ (t1 t2 : ℕ), (t1 + t2 = 2 * cos_α) ∧ (t1 * t2 = -3) ∧ (abs (t1 - t2) = sqrt 14)

theorem find_alpha :
  ∀ α, (line_intersects_curve α ↔ (α = π / 4 ∨ α = 3 * π / 4)) :=
by 
  sorry

end find_alpha_l649_649470


namespace max_socks_from_yarn_l649_649877

variables {x y z : ℕ}
variable largeBallWool : ℕ
variable smallBallWool : ℕ

-- Conditions
def largeBallCanKnitSweaterAndSocks : Prop :=
    largeBallWool = 5 * y

def smallBallCanKnitHalfSweater : Prop :=
    smallBallWool = 2 * y

-- Relations from given conditions
def oneSweaterEqInHats : Prop :=
    largeBallWool = (3 * x + z = 5 * y)

def halfSweaterEqHats : Prop :=
    (0.5 * z = 2 * y)

def sweaterInTermsHats : Prop :=
    z = 4 * y

def hatInTermsSocks : Prop :=
    y = 3 * x

-- Combine all
def totalSocksUsingBothBalls : ℕ := 
    (5 * 3 * x + 2 * 3 * x) / x

theorem max_socks_from_yarn (largeBallCanKnitSweaterAndSocks : largeBallCanKnitSweaterAndSocks) 
(smallBallCanKnitHalfSweater : smallBallCanKnitHalfSweater) 
(oneSweaterEqInHats : oneSweaterEqInHats) 
(halfSweaterEqHats : halfSweaterEqHats) 
(sweaterInTermsHats : sweaterInTermsHats) 
(hatInTermsSocks : hatInTermsSocks) : totalSocksUsingBothBalls = 21 :=
by 
  -- skipping the proof as instructed
  sorry

end max_socks_from_yarn_l649_649877


namespace alpha_beta_identity_l649_649466

open Real

theorem alpha_beta_identity 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h : cos β = tan α * (1 + sin β)) : 
  2 * α + β = π / 2 :=
by
  sorry

end alpha_beta_identity_l649_649466


namespace infinite_sum_identity_l649_649943

theorem infinite_sum_identity (x : ℝ) (h : x > 1) :
  (∑' n : ℕ, 1 / (x^(3^n) - x^(-3^n))) = 1 / (x - 1) :=
sorry

end infinite_sum_identity_l649_649943


namespace find_k_for_ellipse_area_l649_649261

variable (a b : ℝ)

def ellipse_area (a b : ℝ) : ℝ :=
  Real.pi * a * b

theorem find_k_for_ellipse_area :
  a = 12 → b = 6 → ∃ k, ellipse_area a b = k * Real.pi ∧ k = 72 :=
by
  sorry

end find_k_for_ellipse_area_l649_649261


namespace smallest_n_prime_in_subset_l649_649969

def S := Finset.range 2006

-- Define pairwise coprime predicate
def pairwise_coprime (A : Finset ℕ) : Prop :=
  ∀ (a b ∈ A), a ≠ b → Nat.coprime a b

-- Define the main theorem
theorem smallest_n_prime_in_subset {n : ℕ} (H : ∀ (A : Finset ℕ), A ⊆ S → A.card = n → pairwise_coprime A → ∃ p ∈ A, Nat.Prime p) : n = 16 := 
  by
  sorry

end smallest_n_prime_in_subset_l649_649969


namespace quadratic_discriminant_one_solution_l649_649598

theorem quadratic_discriminant_one_solution (m : ℚ) : 
  (3 * (1 : ℚ))^2 - 12 * m = 0 → m = 49 / 12 := 
by {
  sorry
}

end quadratic_discriminant_one_solution_l649_649598


namespace evaluate_exponentiation_l649_649795

theorem evaluate_exponentiation : (3 ^ 3) ^ 4 = 531441 := by
  sorry

end evaluate_exponentiation_l649_649795


namespace cos450_eq_zero_l649_649728

theorem cos450_eq_zero (cos_periodic : ∀ x, cos (x + 360) = cos x) (cos_90_eq_zero : cos 90 = 0) :
  cos 450 = 0 := by
  sorry

end cos450_eq_zero_l649_649728


namespace gcd_values_count_l649_649240

theorem gcd_values_count (a b : ℤ) (h : Int.gcd a b * Nat.lcm a b = 360) : 
  {d : ℤ | d = Int.gcd a b }.card = 11 :=
by
  sorry

end gcd_values_count_l649_649240


namespace oxygen_atoms_in_compound_l649_649684

theorem oxygen_atoms_in_compound : 
  ∃ x : ℕ, 
    let calcium_astomic_weight := 40.08,
        hydrogen_astomic_weight := 1.008,
        oxygen_astomic_weight := 16.00,
        total_molecular_weight := 74.00,
        calcium_count := 1,
        hydrogen_count := 2 in
    calcium_count * calcium_astomic_weight + 
    hydrogen_count * hydrogen_astomic_weight + 
    x * oxygen_astomic_weight = total_molecular_weight ∧ x = 2 :=
by
  -- Proof goes here.
  sorry

end oxygen_atoms_in_compound_l649_649684


namespace line_passes_fixed_point_l649_649214

theorem line_passes_fixed_point :
  ∀ k : ℝ, ∃ x y : ℝ, (2k-1) * x - (k-2) * y - (k+4) = 0 ∧ x = 2 ∧ y = 3 :=
by
  intro k
  use [2, 3]
  constructor
  · calc (2k-1) * 2 - (k-2) * 3 - (k+4)
      = (4k - 2) - (3k - 6) - k - 4 : by ring
    ... = k + 10 - k - 4 : by ring
    ... = 6 : by ring
  · constructor <;> rfl

end line_passes_fixed_point_l649_649214


namespace stratified_sampling_third_grade_students_l649_649682

variable (total_students : ℕ) (second_year_female_probability : ℚ) (sample_size : ℕ)

theorem stratified_sampling_third_grade_students
  (h_total : total_students = 2000)
  (h_probability : second_year_female_probability = 0.19)
  (h_sample_size : sample_size = 64) :
  let sampling_fraction := 64 / 2000
  let third_grade_students := 2000 * sampling_fraction
  third_grade_students = 16 :=
by
  -- the proof would go here, but we're skipping it per instructions
  sorry

end stratified_sampling_third_grade_students_l649_649682


namespace vasya_max_triangles_l649_649297

theorem vasya_max_triangles (n : ℕ) (h1 : n = 100)
  (h2 : ∀ (a b c : ℕ), a + b ≤ c ∨ b + c ≤ a ∨ c + a ≤ b) :
  ∃ (t : ℕ), t = n := 
sorry

end vasya_max_triangles_l649_649297


namespace gena_encoded_numbers_unique_l649_649432

theorem gena_encoded_numbers_unique : 
  ∃ (B AN AX NO FF d : ℕ), (AN - B = d) ∧ (AX - AN = d) ∧ (NO - AX = d) ∧ (FF - NO = d) ∧ 
  [B, AN, AX, NO, FF] = [5, 12, 19, 26, 33] := sorry

end gena_encoded_numbers_unique_l649_649432


namespace factor_expression_l649_649011

theorem factor_expression (x : ℝ) : 
  (9 * x^5 + 25 * x^3 - 4) - (x^5 - 3 * x^3 - 4) = 4 * x^3 * (2 * x^2 + 7) :=
by
  sorry

end factor_expression_l649_649011


namespace find_coordinates_of_P_l649_649062

structure Point where
  x : ℝ
  y : ℝ

def M := Point.mk (-2) 7
def N := Point.mk 10 (-2)

def is_on_segment (P : Point) (A B: Point) : Prop :=
  ∃ t ∈ (set.Icc 0 1 : set ℝ), P.x = (1 - t) * A.x + t * B.x ∧ P.y = (1 - t) * A.y + t * B.y

def P_condition (P : Point) (M N: Point) : Prop :=
  (N.x - P.x = -2 * (M.x - P.x)) ∧
  (N.y - P.y = -2 * (M.y - P.y))

theorem find_coordinates_of_P : ∃ P : Point, is_on_segment P M N ∧ P_condition P M N ∧ P = Point.mk 2 4 :=
by
  use Point.mk 2 4
  sorry

end find_coordinates_of_P_l649_649062


namespace sum_of_inverse_b_n_l649_649082

noncomputable def a (n : ℕ) : ℝ := 1 / 2^n

def b (n : ℕ) : ℝ := -(n * (n + 1)) / 2

theorem sum_of_inverse_b_n (n : ℕ) :
    (a 1 + 2 * a 2 = 1) →
    (a 3^2 = 4 * a 2 * a 6) →
    b n = (∑ i in range (n+1), log 2 (a i)) →
    (∑ i in range (n+1), 1 / b i) = -2 * (n / (n + 1)) := by
  sorry

end sum_of_inverse_b_n_l649_649082


namespace num_positive_cos_terms_l649_649756

def sequence (n : ℕ) : ℝ := Real.cos (10^(n-1) * Real.pi / 180)

theorem num_positive_cos_terms : (Finset.card (Finset.filter (λ n, 0 < sequence n) (Finset.range 100))) = 99 := 
sorry

end num_positive_cos_terms_l649_649756


namespace probability_of_one_side_of_triangle_being_side_of_decagon_is_one_half_l649_649060

noncomputable def probability_one_side_of_triangle_is_side_of_decagon : ℚ :=
  let total_number_of_triangles := 120 in
  let favorable_outcomes := 60 in
  favorable_outcomes / total_number_of_triangles

theorem probability_of_one_side_of_triangle_being_side_of_decagon_is_one_half :
  probability_one_side_of_triangle_is_side_of_decagon = 1 / 2 := 
  sorry

end probability_of_one_side_of_triangle_being_side_of_decagon_is_one_half_l649_649060


namespace number_of_schools_in_pythagoras_city_l649_649911

theorem number_of_schools_in_pythagoras_city (n : ℕ) (h1 : true) 
    (h2 : true) (h3 : ∃ m, m = (3 * n + 1) / 2)
    (h4 : true) (h5 : true) : n = 24 :=
by 
  have h6 : 69 < 3 * n := sorry
  have h7 : 3 * n < 79 := sorry
  sorry

end number_of_schools_in_pythagoras_city_l649_649911


namespace meaningful_sqrt_log_l649_649489

theorem meaningful_sqrt_log (a : ℝ) : 
  (1/2 ≤ a ∧ a < 2 ∨ 2 < a ∧ a < 3) ↔ (4a - 2 ≥ 0 ∧ 3 - a > 0 ∧ log 4 (3 - a) ≠ 0) :=
by
  sorry

end meaningful_sqrt_log_l649_649489


namespace mutuallyExclusiveNotContradictoryEvents_l649_649284

structure Products :=
  (qualified : ℕ)
  (unqualified : ℕ)

def isEventA (p : Products) (selected: Finset ℕ) : Prop :=
  selected.card = 2 ∧ (selected.filter (λ x => x ≤ p.unqualified)).card ≥ 1 ∧ (selected.filter (λ x => x > p.unqualified)).card ≤ 1

def isEventB (p : Products) (selected: Finset ℕ) : Prop :=
  selected.card = 2 ∧ (selected.filter (λ x => x ≤ p.unqualified)).card = 1 ∧ (selected.filter (λ x => x > p.unqualified)).card = 2

def mutuallyExclusive (a b : Prop) : Prop :=
  (a ∧ b) → False

def notContradictory (a b : Prop) : Prop :=
  ¬(a → False) ∧ ¬(b → False)

theorem mutuallyExclusiveNotContradictoryEvents (products : Products) (selected: Finset ℕ) :
  isEventB products selected → mutuallyExclusive (isEventB products selected) (isEventA products selected) ∧ notContradictory (isEventB products selected) (isEventA products selected) :=
by
  sorry

end mutuallyExclusiveNotContradictoryEvents_l649_649284


namespace number_of_boxes_needed_l649_649207

def total_bananas : ℕ := 40
def bananas_per_box : ℕ := 5

theorem number_of_boxes_needed : (total_bananas / bananas_per_box) = 8 := by
  sorry

end number_of_boxes_needed_l649_649207


namespace distances_product_eq_l649_649972

-- Define the distances
variables (d_ab d_ac d_bc d_ba d_cb d_ca : ℝ)

-- State the theorem
theorem distances_product_eq : d_ab * d_bc * d_ca = d_ac * d_ba * d_cb :=
sorry

end distances_product_eq_l649_649972


namespace find_mod_inverse_of_4_mod_21_l649_649417

theorem find_mod_inverse_of_4_mod_21 :
  ∃ a : ℕ, a < 21 ∧ 4 * a % 21 = 1 :=
begin
  use 16,
  split,
  { exact nat.lt_succ_self 20, },  -- verifies 16 < 21
  { norm_num, }
end

end find_mod_inverse_of_4_mod_21_l649_649417


namespace chessboard_pieces_count_is_multiple_of_4_l649_649593

noncomputable def pieces_on_infinite_chessboard (n : ℕ) : Prop :=
  ∀ (conditions : (∀ b, ∃ k, on_same_diagonal b k) ∧ (∀ k, ∃ b, distance b k = real.sqrt 5) ∧ (∀ p, removal_disrupts p)),
  ∃ k : ℕ, n = 4 * k ∧ k > 0

axiom exist_conditions : ∃ conditions : Prop, 
  (∀ b, ∃ k, on_same_diagonal b k) ∧ 
  (∀ k, ∃ b, distance b k = real.sqrt 5) ∧ 
  (∀ p, removal_disrupts p)

theorem chessboard_pieces_count_is_multiple_of_4 (n : ℕ) : 
  (pieces_on_infinite_chessboard n) :=
begin
  obtain ⟨conditions, h1, h2, h3⟩ := exist_conditions,
  sorry
end

end chessboard_pieces_count_is_multiple_of_4_l649_649593


namespace even_digit_sum_count_l649_649487

-- Define the necessary functions and conditions
def is_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d
def digit_sum (n : ℕ) : ℕ := n.digits.sum
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the arithmetic sequence conditions
def sequence_term (n : ℕ) : ℕ := is_arithmetic_sequence 4 5 n
noncomputable def sequence_set := {n | ∃ k : ℕ, k ≥ 1 ∧ k ≤ 403 ∧ n = sequence_term k}

-- Statement of our problem in Lean
theorem even_digit_sum_count : 
  {n // n ∈ sequence_set ∧ is_even (digit_sum n)}.card = 201 := 
sorry

end even_digit_sum_count_l649_649487


namespace sum_infinite_series_result_l649_649959

noncomputable def sum_infinite_series (x : ℝ) (h : 1 < x) : ℝ :=
  ∑' n, 1 / (x^(3^n) - x^(-3^n))

theorem sum_infinite_series_result (x : ℝ) (h : 1 < x) :
  sum_infinite_series x h = 1 / (x - 1) :=
sorry

end sum_infinite_series_result_l649_649959


namespace sum_of_digits_of_x_l649_649697

def two_digit_palindrome (x : ℕ) : Prop :=
  (10 ≤ x ∧ x ≤ 99) ∧ (x = (x % 10) * 10 + (x % 10))

def three_digit_palindrome (y : ℕ) : Prop :=
  (100 ≤ y ∧ y ≤ 999) ∧ (y = (y % 10) * 101 + (y % 10))

theorem sum_of_digits_of_x (x : ℕ) (h1 : two_digit_palindrome x) (h2 : three_digit_palindrome (x + 10)) : 
  (x % 10 + x / 10) = 10 :=
by
  sorry

end sum_of_digits_of_x_l649_649697


namespace n_is_one_sixth_sum_of_list_l649_649335

-- Define the condition that n is 4 times the average of the other 20 numbers
def satisfies_condition (n : ℝ) (l : List ℝ) : Prop :=
  l.length = 21 ∧
  n ∈ l ∧
  n = 4 * (l.erase n).sum / 20

-- State the main theorem
theorem n_is_one_sixth_sum_of_list {n : ℝ} {l : List ℝ} (h : satisfies_condition n l) :
  n = (1 / 6) * l.sum :=
by
  sorry

end n_is_one_sixth_sum_of_list_l649_649335


namespace total_wet_surface_area_is_200_l649_649340

-- Definitions for the dimensions of each cistern
def cistern1 := (length := 7, width := 4, depth := 1.25)
def cistern2 := (length := 10, width := 5, depth := 1.5)
def cistern3 := (length := 6, width := 3, depth := 1.75)

-- Function to compute the bottom area of a cistern
def bottomArea (length : ℝ) (width : ℝ) : ℝ := length * width

-- Function to compute the side areas of a cistern given depth
def sideArea (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ := 
  2 * (length * depth + width * depth)

-- Function to compute the total wet surface area of a cistern
def wetSurfaceArea (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ :=
  bottomArea length width + sideArea length width depth

-- Definitions for the wet surface area of each cistern
def wetSurfaceArea1 := wetSurfaceArea cistern1.length cistern1.width cistern1.depth
def wetSurfaceArea2 := wetSurfaceArea cistern2.length cistern2.width cistern2.depth
def wetSurfaceArea3 := wetSurfaceArea cistern3.length cistern3.width cistern3.depth

-- Total wet surface area of all three cisterns
def totalWetSurfaceArea := wetSurfaceArea1 + wetSurfaceArea2 + wetSurfaceArea3

-- The theorem to prove total wet surface area is 200 m²
theorem total_wet_surface_area_is_200 : totalWetSurfaceArea = 200 := by
  sorry

end total_wet_surface_area_is_200_l649_649340


namespace number_of_elements_in_B_l649_649563

open Complex

def find_elements_in_set_B (A : Set ℂ) (B : Set (ℂ × ℂ)) : Prop :=
  B = { p : ℂ × ℂ | p.1 ∈ A ∧ p.2 ∈ A ∧ 
                     Ree p.1 + Ree p.2 + Imm p.1 + Imm p.2 > 0 ∧
                     ∃ k m, 1 ≤ k ∧ k ≤ 2024 ∧ 1 ≤ m ∧ m ≤ 2024 ∧
                            (p.1 = A k) ∧ (p.2 = A m) }

def complex_set_conditions (A : Set ℂ) : Prop :=
  (∀ (k m : ℕ), 1 ≤ k ∧ k ≤ 2024 ∧ 1 ≤ m ∧ m ≤ 2024 →
      (A m) / (A k) ∈ A ∧ (A m) * (A k) ∈ A) ∧ 
  ∃ z : ℂ, ∀ z' ∈ A, z' ≠ 0

theorem number_of_elements_in_B : ∃ (A : Set ℂ), 
  complex_set_conditions A →
  ∃ (B : Set (ℂ × ℂ)), find_elements_in_set_B A B →
  |B| = 1023025 :=
sorry

end number_of_elements_in_B_l649_649563


namespace total_cost_of_items_is_correct_l649_649248

theorem total_cost_of_items_is_correct :
  ∀ (M R F : ℝ),
  (10 * M = 24 * R) →
  (F = 2 * R) →
  (F = 24) →
  (4 * M + 3 * R + 5 * F = 271.2) :=
by
  intros M R F h1 h2 h3
  sorry

end total_cost_of_items_is_correct_l649_649248


namespace sum_of_digits_10_pow_97_minus_97_eq_858_l649_649407

theorem sum_of_digits_10_pow_97_minus_97_eq_858 :
  (let n := 97 in (10^n - n).digits.sum) = 858 :=
by
  sorry

end sum_of_digits_10_pow_97_minus_97_eq_858_l649_649407


namespace series_sum_l649_649948

noncomputable def compute_series (x : ℝ) (hx : x > 1) : ℝ :=
  ∑' n, 1 / (x ^ (3 ^ n) - x ^ (- 3 ^ n))

theorem series_sum (x : ℝ) (hx : x > 1) : compute_series x hx = 1 / (x - 1) :=
sorry

end series_sum_l649_649948


namespace find_length_segment_od1_l649_649917

noncomputable def od_length {O D1 : Type*}
  (radius_sphere : ℝ) (r1_face : ℝ) (r2_face : ℝ) (r3_face : ℝ)
  (h_radius_sphere : radius_sphere = 10)
  (h_r1_face : r1_face = 1)
  (h_r2_face : r2_face = 1)
  (h_r3_face : r3_face = 3)
: ℝ :=
  let od1_squared := (radius_sphere ^ 2 - r1_face ^ 2) + (radius_sphere ^ 2 - r2_face ^ 2) + (radius_sphere ^ 2 - r3_face ^ 2) in
  real.sqrt od1_squared

theorem find_length_segment_od1 (radius_sphere r1_face r2_face r3_face length_od1 : ℝ)
  (h_radius_sphere : radius_sphere = 10)
  (h_r1_face : r1_face = 1)
  (h_r2_face : r2_face = 1)
  (h_r3_face : r3_face = 3)
  (h_length_od1 : length_od1 = 17) :
  od_length radius_sphere r1_face r2_face r3_face h_radius_sphere h_r1_face h_r2_face h_r3_face = length_od1 :=  sorry

end find_length_segment_od1_l649_649917


namespace confidence_interval_l649_649897

theorem confidence_interval {k : ℝ} (h : Pr(k > 3.841) = 0.05) : Pr(k^2 > 3.841) = 0.95 :=
sorry

end confidence_interval_l649_649897


namespace diff_present_students_l649_649984

theorem diff_present_students (T A1 A2 A3 P1 P2 : ℕ) 
  (hT : T = 280)
  (h_total_absent : A1 + A2 + A3 = 240)
  (h_absent_ratio : A2 = 2 * A3)
  (h_absent_third_day : A3 = 280 / 7) 
  (hP1 : P1 = T - A1)
  (hP2 : P2 = T - A2) :
  P2 - P1 = 40 :=
sorry

end diff_present_students_l649_649984


namespace max_area_of_rectangular_fence_l649_649222

theorem max_area_of_rectangular_fence (x y : ℕ) (h : x + y = 75) : 
  (x * (75 - x) ≤ 1406) ∧ (∀ x' y', x' + y' = 75 → x' * y' ≤ 1406) :=
by
  sorry

end max_area_of_rectangular_fence_l649_649222


namespace ratio_S4_over_a2_plus_a5_l649_649689

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)
variable (q a1 : ℝ)

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) : Prop := ∀ n : ℕ, a n = a1 * q^n

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop := ∀ n : ℕ, S n = ∑ i in finset.range(n + 1), a i

-- Problem conditions
variable (positive_q : q > 0)
variable (geo_seq : geometric_sequence a q a1)
variable (sum_n_terms : sum_of_first_n_terms S a)
variable (condition : 4 * a 1 = a 3)

-- Prove the required ratio
theorem ratio_S4_over_a2_plus_a5 : S 3 / (a 1 + a 4) = 5 / 6 := by
  sorry

end ratio_S4_over_a2_plus_a5_l649_649689


namespace sum_series_eq_l649_649952

open Real

theorem sum_series_eq (x : ℝ) (h : 1 < x) : 
  (∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (- (3 ^ n)))) = 1 / (x - 1) :=
sorry

end sum_series_eq_l649_649952


namespace max_black_squares_overlap_l649_649712

-- Conditions
def is_black_square (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

-- Chessboard dimensions
def chessboard_size : ℕ := 8

-- Rectangle dimensions
def rectangle_length : ℕ := 1
def rectangle_width : ℕ := 2

-- Definition of overlap, true if paper overlaps the square
def overlaps (i j : ℕ) (x y : ℕ) : Prop :=
  ((i = x ∨ i = x + 1) ∧ (j = y ∨ j = y + 1))

-- Theorem to prove: maximum number of black squares overlapped by the rectangle
theorem max_black_squares_overlap : ∀ (x y : ℕ), x < chessboard_size ∧ y < chessboard_size → 
  ∑ i in range chessboard_size, ∑ j in range chessboard_size, 
  if overlaps i j x y then (if is_black_square i j then 1 else 0) else 0 ≤ 6 := sorry

end max_black_squares_overlap_l649_649712


namespace shooting_test_probability_eq_l649_649516

noncomputable def shooting_probability : ℚ :=
  let p := 2 / 3
  let prob_3_hits := p^3
  let prob_4_hits := (binomial 4 3) * (p^3) * ((1 - p)^1)
  let prob_5_hits := (binomial 5 3) * (p^3) * ((1 - p)^2)
  prob_3_hits + prob_4_hits + prob_5_hits

theorem shooting_test_probability_eq : 
  shooting_probability = 64 / 81 := 
sorry

end shooting_test_probability_eq_l649_649516


namespace magnitude_of_3a_minus_b_l649_649120

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b : ℝ × ℝ := sorry

axiom mag_b : real.sqrt (b.1 ^ 2 + b.2 ^ 2) = 4
axiom dot_cond : 1 * (1 - b.1) + 1 * (1 - b.2) = -2

noncomputable def vec3a_minus_b : ℝ × ℝ :=
  (3 * a.1 - b.1, 3 * a.2 - b.2)

theorem magnitude_of_3a_minus_b : real.sqrt ((vec3a_minus_b.1)^2 + (vec3a_minus_b.2)^2) = real.sqrt 10 := by
  sorry

end magnitude_of_3a_minus_b_l649_649120


namespace intersection_correct_l649_649131

-- Definition of set A
def A : set ℝ := { x | x * (x - 2) < 0 }

-- Definition of set B
def B : set ℝ := { x | x - 1 ≤ 0 }

-- Complement of set B
def B_complement : set ℝ := { x | x > 1 }

-- The intersection of A and B_complement
def intersection : set ℝ := A ∩ B_complement

-- Statement to be proved
theorem intersection_correct : intersection = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_correct_l649_649131


namespace percentage_of_students_in_grade_8_combined_l649_649270

theorem percentage_of_students_in_grade_8_combined (parkwood_students maplewood_students : ℕ)
  (parkwood_percentages maplewood_percentages : ℕ → ℕ) 
  (H_parkwood : parkwood_students = 150)
  (H_maplewood : maplewood_students = 120)
  (H_parkwood_percent : parkwood_percentages 8 = 18)
  (H_maplewood_percent : maplewood_percentages 8 = 25):
  (57 / 270) * 100 = 21.11 := 
by
  sorry  -- Proof omitted

end percentage_of_students_in_grade_8_combined_l649_649270


namespace number_of_paths_in_grid_l649_649792

-- Define the condition of the problem where we traverse a grid from bottom-left to top-right with specific dimensions and path length.
theorem number_of_paths_in_grid : 
  ∀ (w h: ℕ), w = 6 → h = 3 → (∃ (n : ℕ), n = 9 ∧ 
  ∀ (paths : ℕ), paths = Nat.choose n h → paths = 84) :=
by
  intros w h w_eq h_eq
  use 9
  constructor
  { refl }
  { intros paths paths_eq
    rw [paths_eq]
    rw [w_eq, h_eq]
    exact Nat.choose_eq_factorial_div_factorial (by decide) 
      (by decide) (by decide) 
      ◻
  }

end number_of_paths_in_grid_l649_649792


namespace day_after_60_days_is_monday_l649_649382

theorem day_after_60_days_is_monday
    (birthday_is_thursday : ∃ d : ℕ, d % 7 = 0) :
    ∃ d : ℕ, (d + 60) % 7 = 4 :=
by
  -- Proof steps are omitted here
  sorry

end day_after_60_days_is_monday_l649_649382


namespace range_of_m_l649_649478

def cond1 (x : ℝ) : Prop := x^2 - 4 * x + 3 < 0
def cond2 (x : ℝ) : Prop := x^2 - 6 * x + 8 < 0
def cond3 (x m : ℝ) : Prop := 2 * x^2 - 9 * x + m < 0

theorem range_of_m (m : ℝ) : (∀ x, cond1 x → cond2 x → cond3 x m) → m < 9 :=
by
  sorry

end range_of_m_l649_649478


namespace series_sum_eq_l649_649309

theorem series_sum_eq :
  ∑ n in Finset.range 2000, (n + 1) / (n + 2)! = 1 - 1 / 2001! :=
  sorry

end series_sum_eq_l649_649309


namespace remaining_tanning_time_l649_649540

noncomputable def tanning_limit : ℕ := 200
noncomputable def daily_tanning_time : ℕ := 30
noncomputable def weekly_tanning_days : ℕ := 2
noncomputable def weeks_tanned : ℕ := 2

theorem remaining_tanning_time :
  let total_tanning_first_two_weeks := daily_tanning_time * weekly_tanning_days * weeks_tanned
  tanning_limit - total_tanning_first_two_weeks = 80 :=
by
  let total_tanning_first_two_weeks := daily_tanning_time * weekly_tanning_days * weeks_tanned
  have h : total_tanning_first_two_weeks = 120 := by sorry
  show tanning_limit - total_tanning_first_two_weeks = 80 from sorry

end remaining_tanning_time_l649_649540


namespace hyperbola_equation_l649_649692

-- Conditions
def center_origin (P : ℝ × ℝ) : Prop := P = (0, 0)
def focus_at (F : ℝ × ℝ) : Prop := F = (0, Real.sqrt 3)
def vertex_distance (d : ℝ) : Prop := d = Real.sqrt 3 - 1

-- Statement
theorem hyperbola_equation
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (d : ℝ)
  (h_center : center_origin center)
  (h_focus : focus_at focus)
  (h_vert_dist : vertex_distance d) :
  y^2 - (x^2 / 2) = 1 := 
sorry

end hyperbola_equation_l649_649692


namespace exist_triangle_with_properties_l649_649785

-- Conditions given in the problem
variables (r : ℝ) (beta : ℝ) (b c : ℝ)

-- Assume the standard conditions for angles and sides to be positive real numbers
axiom pos_r : r > 0
axiom pos_beta : 0 < beta ∧ beta < π
axiom pos_sides : b > 0 ∧ c > 0

-- The proof goal, to prove the existence of the triangle with the given properties
theorem exist_triangle_with_properties :
  ∃ (A B C : Type) [T : triangle A B C],
    circumradius T = r ∧ angle_at T B = beta ∧ side_length B A + side_length A C = b + c :=
sorry

end exist_triangle_with_properties_l649_649785


namespace convex_quadrilateral_inequality_l649_649974

theorem convex_quadrilateral_inequality
  (A B C D E : Point)
  (h_convex : ConvexQuadrilateral A B C D)
  (h_angle : ∠BCD = 90)
  (h_midpoint_E : IsMidpoint E A B) :
  2 * dist E C ≤ dist A D + dist B D :=
sorry

end convex_quadrilateral_inequality_l649_649974


namespace crayons_left_is_4_l649_649631

-- Define initial number of crayons in the drawer
def initial_crayons : Nat := 7

-- Define number of crayons Mary took out
def taken_by_mary : Nat := 3

-- Define the number of crayons left in the drawer
def crayons_left (initial : Nat) (taken : Nat) : Nat :=
  initial - taken

-- Prove the number of crayons left in the drawer is 4
theorem crayons_left_is_4 : crayons_left initial_crayons taken_by_mary = 4 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end crayons_left_is_4_l649_649631


namespace QR_parallel_AB_l649_649916

variables {α : Type*} [EuclideanGeometry α]

-- Definitions from the conditions
variables {A B C O E F P Q R : α}
variables (h_triangle : Triangle A B C)
variables (h_eq_sides : A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ AB = AC)
variables (h_midpoint : Midpoint O B C)
variables (h_circle_center : Circle O)
variables (h_intersect_AB_E : CircleIntersects h_circle_center AB E ∧ on_line_segment AB E)
variables (h_intersect_AC_F : CircleIntersects h_circle_center AC F ∧ on_line_segment AC F)
variables (h_c_p : LineIntersectsCircle CP h_circle_center P ∧ on_circle_arc CP P)
variables (h_intersect_EF_R : LineIntersects EF P ∧ on_line_segment EF R)

-- The statement of the theorem to prove: QR is parallel to AB
theorem QR_parallel_AB (h_conditions : ∧ h_triangle ∧ h_eq_sides ∧ h_midpoint ∧ h_circle_center ∧ 
                                      h_intersect_AB_E ∧ h_intersect_AC_F ∧ 
                                      h_c_p ∧ h_intersect_EF_R) : 
QR ∥ AB := 
sorry

end QR_parallel_AB_l649_649916


namespace samuel_doughnuts_l649_649582

theorem samuel_doughnuts (samuel_dozens cathy_dozens total_people doughnuts_per_person total_doughnuts_per_person: ℕ) (H_cathy: cathy_dozens = 3) (H_people: total_people = 10) (H_each_doughnuts: doughnuts_per_person = 6):
  samuel_dozens = 2 :=
  let total_doughnuts := total_people * doughnuts_per_person in
  have H_total_doughnuts: total_doughnuts = 60, by sorry,
  let cathy_doughnuts := cathy_dozens * 12 in
  have H_cathy_total_doughnuts: cathy_doughnuts = 36, by sorry,
  let samuel_doughnuts := total_doughnuts - cathy_doughnuts in
  have H_samuel_doughnuts: samuel_doughnuts = 24, by sorry,
  have H_samuel_dozens: samuel_dozens = samuel_doughnuts / 12, by sorry,
  sorry

end samuel_doughnuts_l649_649582


namespace distance_from_origin_to_point_is_25_l649_649152

-- Define the points and the distance function
noncomputable def distance_from_origin (x y : ℝ) : ℝ :=
  real.sqrt (x^2 + y^2)

-- Point definitions
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (7, -24)

-- Lean statement to prove the distance
theorem distance_from_origin_to_point_is_25 :
  distance_from_origin (point.fst) (point.snd) = 25 :=
by
  sorry

end distance_from_origin_to_point_is_25_l649_649152


namespace combined_time_alligators_walked_l649_649168

theorem combined_time_alligators_walked
  (time_to_nile_delta : ℕ)
  (extra_return_time : ℕ)
  (return_time_with_alligators : ℕ)
  (combined_time : ℕ) :
  time_to_nile_delta = 4 →
  extra_return_time = 2 →
  return_time_with_alligators = time_to_nile_delta + extra_return_time →
  combined_time = time_to_nile_delta + return_time_with_alligators →
  combined_time = 10 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end combined_time_alligators_walked_l649_649168


namespace percent_birth_month_in_march_l649_649251

theorem percent_birth_month_in_march (total_people : ℕ) (march_births : ℕ) (h1 : total_people = 100) (h2 : march_births = 8) : (march_births * 100 / total_people) = 8 := by
  sorry

end percent_birth_month_in_march_l649_649251


namespace min_ratio_ab_cd_l649_649569

noncomputable def cyclic_quadrilateral (A B C D : Type) := 
  ∃ (X : A), 
  (X ∈ segment A B) ∧ 
  (diagonal_bisects B D C X) ∧ 
  (diagonal_bisects A C D X) 

theorem min_ratio_ab_cd (A B C D X: Type) 
  [h1 : cyclic_quadrilateral A B C D]
  [h2 : X ∈ segment A B]
  [h3 : diagonal_bisects B D C X]
  [h4 : diagonal_bisects A C D X]
  : ∃ r, r = 2 ∧ ∀ (AB CD : ℝ), AB / CD ≥ r :=
sorry

end min_ratio_ab_cd_l649_649569


namespace exists_point_K_l649_649262

variables {ι : Type} [Fintype ι] {A B : ι → ℝ} {s : ℝ}
           
def points_not_collinear (A : ι → ℝ) : Prop := ∃ i j k : ι, i ≠ j ∧ j ≠ k ∧ k ≠ i

theorem exists_point_K (A : ι → ℝ) (P Q : ι → ℝ) (s : ℝ)
  (h_not_collinear : points_not_collinear A)
  (h_equal_sums : ∑ i, dist (A i) (P i) = s ∧ ∑ i, dist (A i) (Q i) = s)
  (h_distinct : P ≠ Q) :
  ∃ K, ∑ i, dist (A i) (K i) < s :=
sorry

end exists_point_K_l649_649262


namespace terminal_side_same_line_37_and_neg143_l649_649313

theorem terminal_side_same_line_37_and_neg143 :
  ∃ k : ℤ, (37 : ℝ) + 180 * k = (-143 : ℝ) :=
by
  -- Proof steps go here
  sorry

end terminal_side_same_line_37_and_neg143_l649_649313


namespace simplify_expression_l649_649584

theorem simplify_expression :
  (((0.3 * 0.8) / 0.2) + (0.1 * 0.5)^2 - 1 / (0.5 * 0.8)^2) = -5.0475 :=
by
  sorry

end simplify_expression_l649_649584


namespace cos_450_eq_zero_l649_649739

theorem cos_450_eq_zero :
  ∀ (θ : ℝ), θ = 450 ∧ (θ mod 360 = 90) ∧ (cos 90 = 0) → cos θ = 0 :=
by
  intros θ hθ
  cases hθ with h1 hand
  cases hand with h2 h3
  rw h1
  rw h2
  exact h3

end cos_450_eq_zero_l649_649739


namespace function_identity_l649_649190

-- The problem definitions and conditions
variable (f : ℕ → ℤ)
variable (h1 : ∀ n, n > 0 → f n >= 0) -- f(n) is defined for all positive n
variable (h2 : ∀ n, f n ∈ ℤ) -- f(n) is integer-valued
variable (h3 : f 2 = 2) -- f(2) = 2
variable (h4 : ∀ m n, f (m * n) = f m * f n) -- f(m * n) = f(m) * f(n) for all m, n
variable (h5 : ∀ m n, m > n → f m > f n) -- f(m) > f(n) when m > n

theorem function_identity : ∀ n, n > 0 → f n = n :=
by
  intro n h_pos
  -- sorry indicates the proof will be here
  sorry

end function_identity_l649_649190


namespace angle_between_vectors_is_45_degrees_l649_649801

def vector_2d := (ℝ × ℝ)

def dot_product (u v : vector_2d) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : vector_2d) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

def cos_angle (u v : vector_2d) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

def angle_in_degrees (u v : vector_2d) : ℝ :=
  Real.acos (cos_angle u v) * (180 / Real.pi)

theorem angle_between_vectors_is_45_degrees :
  angle_in_degrees (2, 5) (-3, 7) = 45 :=
by
  sorry

end angle_between_vectors_is_45_degrees_l649_649801


namespace find_probability_of_independent_events_l649_649501

def is_independent (A B : Type) (P : Set A → ℝ) := 
∀ (a b : Set A), P (a ∩ b) = P a * P b

variables {A : Type} {P : Set A → ℝ}
variables {a b : Set A}
hypothesis h1 : P a = 2/5
hypothesis h2 : P b = 2/5
hypothesis h3 : is_independent A B P

theorem find_probability_of_independent_events :
  P (a ∩ b) = 4 / 25 :=
by
  rw [h1, h2]
  rw is_independent at h3
  simp [h3 a b] -- This assumes is_independent includes the definition for all components.
  sorry

end find_probability_of_independent_events_l649_649501


namespace smallest_k_l649_649055

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits10.sum

theorem smallest_k :
  ∃ k : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 2014 → sum_of_digits (n * k) = sum_of_digits k) ∧ k = 9999 :=
begin
  use 9999,
  split,
  { intros n hn,
    sorry }, -- Proof to show sum_of_digits (n * 9999) = sum_of_digits 9999
  { refl }
end

end smallest_k_l649_649055


namespace zero_point_interval_l649_649277

theorem zero_point_interval (f : ℝ → ℝ) (x₀ : ℝ) (k : ℤ)
  (h₀ : f x₀ = 0)
  (h_def : ∀ x > 0, f x = Real.log x - 1 / x)
  (h_interval : x₀ ∈ Set.Ico (↑k) (↑k + 1)) :
  k = 1 :=
by
  sorry

end zero_point_interval_l649_649277


namespace zero_point_interval_l649_649275

-- Definition of the function
def f(x : ℝ) : ℝ := Real.log x - 1 / x

-- Defining the assumptions and the theorem
theorem zero_point_interval (x₀ k : ℝ) (k_int : k ∈ Set.Ioo 0 (k + 1)) :
  f x₀ = 0 → (k : ℝ) ∈ ℤ :=
  sorry

end zero_point_interval_l649_649275


namespace cos450_eq_zero_l649_649731

theorem cos450_eq_zero (cos_periodic : ∀ x, cos (x + 360) = cos x) (cos_90_eq_zero : cos 90 = 0) :
  cos 450 = 0 := by
  sorry

end cos450_eq_zero_l649_649731


namespace combined_alligator_walk_time_l649_649166

theorem combined_alligator_walk_time :
  (Paul_to_Delta : ℕ) (Delta_to_River_inc : ℕ) (Number_of_alligators_on_return : ℕ) (Paul_to_Delta = 4) (Delta_to_River_inc = 2) (Number_of_alligators_on_return = 6) :
  let Delta_to_River := Paul_to_Delta + Delta_to_River_inc
  let Total_Paul := Paul_to_Delta + Delta_to_River
  let Alligator_return_total_time := Number_of_alligators_on_return * Delta_to_River
  let Combined_total_time := Total_Paul + Alligator_return_total_time
  Combined_total_time = 46 :=
by
  sorry

end combined_alligator_walk_time_l649_649166


namespace Jonas_needs_to_buy_35_pairs_of_socks_l649_649173

theorem Jonas_needs_to_buy_35_pairs_of_socks
  (socks : ℕ)
  (shoes : ℕ)
  (pants : ℕ)
  (tshirts : ℕ)
  (double_items : ℕ)
  (needed_items : ℕ)
  (pairs_of_socks_needed : ℕ) :
  socks = 20 →
  shoes = 5 →
  pants = 10 →
  tshirts = 10 →
  double_items = 2 * (2 * socks + 2 * shoes + pants + tshirts) →
  needed_items = double_items - (2 * socks + 2 * shoes + pants + tshirts) →
  pairs_of_socks_needed = needed_items / 2 →
  pairs_of_socks_needed = 35 :=
by sorry

end Jonas_needs_to_buy_35_pairs_of_socks_l649_649173


namespace determine_m_l649_649442

theorem determine_m (m : ℝ) (z : ℂ) (h : z = (m - 1) + (m + 1) * complex.I) : m = 1 := 
by
  sorry

end determine_m_l649_649442


namespace domain_ln_2_minus_x_is_interval_l649_649607

noncomputable def domain_ln_2_minus_x : Set Real := { x : Real | 2 - x > 0 }

theorem domain_ln_2_minus_x_is_interval : domain_ln_2_minus_x = Set.Iio 2 :=
by
  sorry

end domain_ln_2_minus_x_is_interval_l649_649607


namespace valid_codes_count_l649_649566

open Finset

def four_digit_code := {x : Fin (8 × 8 × 8 × 8) // x < 4096}

def is_restricted_transpose (code : four_digit_code) : Bool :=
  let d0 := code.val % 8
  let d1 := (code.val / 8) % 8
  let d2 := (code.val / 64) % 8
  let d3 := code.val / 512
  (d0 == 1 && d1 == 0 && d2 == 2 && d3 == 3) || -- exact match with 1023
  (d1 == 1 && d0 == 0 && d2 == 2 && d3 == 3) || -- transpose first two digits
  (d2 == 1 && d1 == 0 && d0 == 2 && d3 == 3) ||
  (d3 == 1 && d1 == 0 && d2 == 2 && d0 == 3) || -- transpose first and fourth digits
  (d3 == 1 && d2 == 0 && d1 == 2 && d0 == 3) ||
  ... -- Add remaining transpose checks (total 6 types)

def is_restricted_three_match (code : four_digit_code) : Bool :=
  let d0 := code.val % 8
  let d1 := (code.val / 8) % 8
  let d2 := (code.val / 64) % 8
  let d3 := code.val / 512
  (d0 == 1 && d1 == 0 && d2 == 2) || 
  (d0 == 1 && d1 == 0 && d3 == 2) ||
  (d0 == 1 && d2 == 0 && d3 == 2) ||
  (d1 == 1 && d2 == 0 && d3 == 2) ||
  ... -- Add remaining 3-digit matched cases (total 4 types)

def count_valid_codes : ℕ :=
  let all_codes := (range 4096).val
  let restricted_codes := all_codes.filter (fun x =>
    is_restricted_transpose ⟨x, sorry⟩ || is_restricted_three_match ⟨x, sorry⟩ || x == 1023
  )
  (card all_codes) - (card restricted_codes)

theorem valid_codes_count : count_valid_codes = 4043 := by
  sorry

end valid_codes_count_l649_649566


namespace gleb_can_achieve_sum_of_fractional_parts_gt_100_l649_649121

noncomputable def exists_fractional_sum_gt_100 : Prop :=
  ∃ (N : ℕ) (a : ℕ → ℕ), 
  (a 1 = N) ∧
  (∀ k : ℕ, a (k + 1) = a k % N) ∧
  (a (succ (nat.pred (nat.find (λ k, a k = 0)))) = 0) ∧
  (∑ k in finset.range (nat.find (λ k, a k = 0)), (a k : ℤ) / (N : ℤ) > 100)

theorem gleb_can_achieve_sum_of_fractional_parts_gt_100 : exists_fractional_sum_gt_100 :=
sorry

end gleb_can_achieve_sum_of_fractional_parts_gt_100_l649_649121


namespace tournament_log2_n_l649_649638

noncomputable def log2_n : ℕ :=
  172

theorem tournament_log2_n :
  ∀ (n : ℕ), 
    let t := 20,
        total_games := (t * (t - 1) / 2),
        total_outcomes := 2^total_games,
        n_factorial := nat.factorial t,
        n_correct := 2 ^ (total_games - (nat.floor 20 / 2 + nat.floor 20 / 4 + nat.floor 20 / 8 + nat.floor 20 / 16))
    in
    ∃ m : ℕ, nat.coprime m n ∧ log2 (total_outcomes / n_factorial) = log2_n :=
by assumption

-- At this point the 'by assumption' and 'sorry' are placeholders for proof steps not included as per instruction.

end tournament_log2_n_l649_649638


namespace constant_term_binomial_expansion_l649_649097

theorem constant_term_binomial_expansion :
  let a := ∫ x in (0:ℝ)..(2:ℝ), (2 * x) in
  (a = 4) → (∃ (T : ℝ), T = 240 ∧ is_constant_term_in_binomial_expansion (λ x, (real.sqrt x - a / x)^6) T) :=
by
  intro a h
  have ha : a = 4 := h
  use 240
  split
  · refl
  sorry

end constant_term_binomial_expansion_l649_649097


namespace gcd_of_B_is_2_l649_649966

-- Definition of the set B based on the given condition.
def B : Set ℕ := {n | ∃ x, n = (x - 1) + x + (x + 1) + (x + 2)}

-- The core statement to prove, wrapped in a theorem.
theorem gcd_of_B_is_2 : gcd_set B = 2 :=
by
  sorry

end gcd_of_B_is_2_l649_649966


namespace distance_between_ann_and_glenda_l649_649367

def ann_distance : ℝ := 
  let speed1 := 6
  let time1 := 1
  let speed2 := 8
  let time2 := 1
  let break1 := 0
  let speed3 := 4
  let time3 := 1
  speed1 * time1 + speed2 * time2 + break1 * 0 + speed3 * time3

def glenda_distance : ℝ := 
  let speed1 := 8
  let time1 := 1
  let speed2 := 5
  let time2 := 1
  let break1 := 0
  let speed3 := 9
  let back_time := 0.5
  let back_distance := speed3 * back_time
  let continue_time := 0.5
  let continue_distance := speed3 * continue_time
  speed1 * time1 + speed2 * time2 + break1 * 0 + (-back_distance) + continue_distance

theorem distance_between_ann_and_glenda : 
  ann_distance + glenda_distance = 35.5 := 
by 
  sorry

end distance_between_ann_and_glenda_l649_649367


namespace remainder_when_divided_by_17_l649_649322

theorem remainder_when_divided_by_17 (N : ℤ) (k : ℤ) 
  (h : N = 221 * k + 43) : N % 17 = 9 := 
by
  sorry

end remainder_when_divided_by_17_l649_649322


namespace number_of_correct_propositions_is_zero_l649_649249

noncomputable def proposition1 (a b : ℝ) : Prop :=
  a = b → (a - b) + (a + b) * complex.i ∉ set.range (λ x, x * complex.i)

noncomputable def proposition2 : Prop :=
  ∀ (z₁ z₂ : ℂ), (complex.abs z₁ = complex.abs z₂ → z₁ = z₂ ∨ z₁ = -z₂)

noncomputable def proposition3 (z₁ z₂ : ℂ) : Prop :=
  z₁^2 + z₂^2 ≠ 0 → z₁ ≠ 0 ∧ z₂ ≠ 0

theorem number_of_correct_propositions_is_zero :
  ¬ proposition1 ∧ ¬ proposition2 ∧ ¬ proposition3 :=
by
  sorry

end number_of_correct_propositions_is_zero_l649_649249


namespace customer_buys_two_pens_l649_649511

def num_pens (total_pens non_defective_pens : Nat) (prob : ℚ) : Nat :=
  sorry

theorem customer_buys_two_pens :
  num_pens 16 13 0.65 = 2 :=
sorry

end customer_buys_two_pens_l649_649511


namespace tangent_line_right_triangle_l649_649859

theorem tangent_line_right_triangle {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (tangent_condition : a^2 + b^2 = c^2) : 
  (abs c)^2 = (abs a)^2 + (abs b)^2 :=
by
  sorry

end tangent_line_right_triangle_l649_649859


namespace prob_neg1_lt_ξ_lt_3_l649_649464

-- Define the random variable ξ and its properties
variable (ξ : ℝ → ℝ) (N : ℝ → ℝ → Type)
variable [NormalDist ξ]

-- Given conditions
axiom normal_dist_ξ : N 1 1 -- 𝜉 follows normal distribution N(1, 1)
axiom prob_ξ_lt_3 : Probability (ξ < 3) = 0.977

-- Proof goal
theorem prob_neg1_lt_ξ_lt_3 : Probability (-1 < ξ ∧ ξ < 3) = 0.954 := 
by 
  sorry

end prob_neg1_lt_ξ_lt_3_l649_649464


namespace calculation_result_l649_649724

theorem calculation_result :
  (2 : ℝ)⁻¹ - (1 / 2 : ℝ)^0 + (2 : ℝ)^2023 * (-0.5 : ℝ)^2023 = -3 / 2 := sorry

end calculation_result_l649_649724


namespace distinct_real_roots_of_quadratic_find_m_and_other_root_l649_649447

theorem distinct_real_roots_of_quadratic (m : ℝ) (h_neg_m : m < 0) : 
    ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (∀ x, x^2 - 2*x + m = 0 → (x = x₁ ∨ x = x₂))) := 
by 
  sorry

theorem find_m_and_other_root (m : ℝ) (h_neg_m : m < 0) (root_minus_one : ∀ x, x^2 - 2*x + m = 0 → x = -1):
    m = -3 ∧ (∃ x, x^2 - 2*x - 3 = 0 ∧ x = 3) := 
by 
  sorry

end distinct_real_roots_of_quadratic_find_m_and_other_root_l649_649447


namespace range_of_a_l649_649203

theorem range_of_a 
    (x y a : ℝ) 
    (hx_pos : 0 < x) 
    (hy_pos : 0 < y) 
    (hxy : x + y = 1) 
    (hineq : ∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → (1 / x + a / y) ≥ 4) :
    a ≥ 1 := 
by sorry

end range_of_a_l649_649203


namespace positive_terms_count_l649_649769

def sequence (n : ℕ) : ℝ := Float.cos (10.0^(n-1) * Float.pi / 180.0)

theorem positive_terms_count :
  (List.filter (λ n, sequence n > 0) (List.range' 1 100)).length = 99 :=
sorry

end positive_terms_count_l649_649769


namespace max_stickers_l649_649358

theorem max_stickers (n_players : ℕ) (avg_stickers : ℕ) (min_stickers : ℕ) 
  (total_players : n_players = 22) 
  (average : avg_stickers = 4) 
  (minimum : ∀ i, i < n_players → min_stickers = 1) :
  ∃ max_sticker : ℕ, max_sticker = 67 :=
by
  sorry

end max_stickers_l649_649358


namespace part1_part2_l649_649469

noncomputable def A : (ℝ × ℝ) := (1, 1)
noncomputable def B : (ℝ × ℝ) := (3, 2)
noncomputable def C : (ℝ × ℝ) := (5, 4)

-- Equation of the line containing the altitude from AB
theorem part1 (A B C : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (3, 2)) (hC : C = (5, 4)) :
  ∃ (line : ℝ → ℝ → Prop), (∀ x y, line x y ↔ 2 * x + y - 14 = 0) :=
sorry

-- Perimeter of the triangle formed by the two coordinate axes and a given line l
theorem part2 (A B C : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (3, 2)) (hC : C = (5, 4))
  (hLine : ∀ x y, l x y → (x / (1 + a)) + (y / a) = 1)
  (ha : ∃ a : ℝ, -4 * a = 3 * (a + 1)) :
  ∃ p : ℝ, p = 12 / 7 :=
sorry

end part1_part2_l649_649469


namespace infinite_radicals_pos_solution_l649_649052

theorem infinite_radicals_pos_solution :
  (∃ x : ℝ, x > 0 ∧
    (∃ z : ℝ, z = real.cbrt (x + 1 + real.cbrt (x + 1 + real.cbrt (x + 1 + ...)))) ∧
    (∃ y : ℝ, y = real.cbrt (3 * x * real.cbrt (3 * x * real.cbrt (3 * x * ...)))) ∧
    z = y ∧
    x = (4 + 2 * real.sqrt 3) / 3) :=
begin
  sorry
end

end infinite_radicals_pos_solution_l649_649052


namespace num_positive_terms_l649_649749

def sequence_cos (n : ℕ) : ℝ :=
  cos (10^(n-1) * π / 180)

theorem num_positive_terms : (finset.filter (λ n : ℕ, 0 < sequence_cos n) (finset.range 100)).card = 99 :=
by
  sorry

end num_positive_terms_l649_649749


namespace num_positive_terms_l649_649775

noncomputable def seq (n : ℕ) : ℝ := float.cos (10^((n - 1).to_real))

theorem num_positive_terms : fin 100 → seq 100 99 :=
sorry

end num_positive_terms_l649_649775


namespace binomial_sum_odd_coeff_l649_649268

theorem binomial_sum_odd_coeff (n : ℕ) (a b : ℕ) (h : (∑ k in finset.range(n+1) \ k.odd, (nat.choose n k)) = 32) : n = 6 := by
  sorry

end binomial_sum_odd_coeff_l649_649268


namespace combinatorial_calculation_l649_649012

-- Define the proof problem.
theorem combinatorial_calculation : (Nat.choose 20 6) = 2583 := sorry

end combinatorial_calculation_l649_649012


namespace octagon_area_percentage_decrease_l649_649705

-- Define the conditions and variables as per the problem
variables (x : ℝ) -- side length of the smaller square
variable h : (4 + 2 * x = 5.6) -- perimeter condition

-- The statement that needs to be proven
theorem octagon_area_percentage_decrease (h : 4 + 2 * x = 5.6) : 1 - x^2 = 0.36 :=
by {
  have x_val : x = 0.8, from eq_of_add_eq_add_right (by linarith), -- solve for x
  rw x_val,
  norm_num,
}

end octagon_area_percentage_decrease_l649_649705


namespace simple_interest_difference_l649_649627

/-- The simple interest on a certain amount at a 4% rate for 5 years amounted to a certain amount less than the principal. The principal was Rs 2400. Prove that the difference between the principal and the simple interest is Rs 1920. 
-/
theorem simple_interest_difference :
  let P := 2400
  let R := 4
  let T := 5
  let SI := (P * R * T) / 100
  P - SI = 1920 :=
by
  /- We introduce the let definitions for the conditions and then state the theorem
    with the conclusion that needs to be proved. -/
  let P := 2400
  let R := 4
  let T := 5
  let SI := (P * R * T) / 100
  /- The final step where we would conclude our theorem. -/
  sorry

end simple_interest_difference_l649_649627


namespace infinite_sum_identity_l649_649942

theorem infinite_sum_identity (x : ℝ) (h : x > 1) :
  (∑' n : ℕ, 1 / (x^(3^n) - x^(-3^n))) = 1 / (x - 1) :=
sorry

end infinite_sum_identity_l649_649942


namespace find_f_ln6_l649_649193

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x = x - Real.exp (-x)

noncomputable def given_function_value : ℝ := Real.log 6

theorem find_f_ln6 (f : ℝ → ℝ)
  (h1 : odd_function f)
  (h2 : condition1 f) :
  f given_function_value = given_function_value + 6 :=
by
  sorry

end find_f_ln6_l649_649193


namespace infinite_series_sum_l649_649935

theorem infinite_series_sum (x : ℝ) (h : x > 1) :
  ∑' n : ℕ, 1 / (x ^ (3 ^ n) - x ^ (-(3 ^ n))) = 1 / (x - 1) :=
sorry

end infinite_series_sum_l649_649935


namespace sum_first_five_terms_geometric_sequence_l649_649820

theorem sum_first_five_terms_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ):
  (∀ n, a (n+1) = a 1 * (1/2) ^ n) →
  a 1 = 16 →
  1/2 * (a 4 + a 7) = 9 / 8 →
  S 5 = (a 1 * (1 - (1 / 2) ^ 5)) / (1 - 1 / 2) →
  S 5 = 31 := by
  sorry

end sum_first_five_terms_geometric_sequence_l649_649820


namespace sin_neg_five_pi_over_six_l649_649791

theorem sin_neg_five_pi_over_six : sin (-5 * Real.pi / 6) = -1 / 2 := 
by 
  sorry

end sin_neg_five_pi_over_six_l649_649791


namespace cos_450_eq_0_l649_649742

theorem cos_450_eq_0 : Real.cos (450 * Real.pi / 180) = 0 := by
  -- Angle equivalence: 450 degrees is equivalent to 90 degrees on the unit circle
  have angle_eq : (450 : Real) * Real.pi / 180 = (90 : Real) * Real.pi / 180 := by
    calc
      (450 : Real) * Real.pi / 180
        = (450 / 180) * Real.pi : by rw [mul_div_assoc]
        = (5 * 90 / 180) * Real.pi : by norm_num
        = (5 * 90 / (2 * 90)) * Real.pi : by norm_num
        = (5 / 2) * Real.pi : by norm_num

  -- Now use this equivalence: cos(450 degrees) = cos(90 degrees)
  have cos_eq : Real.cos (450 * Real.pi / 180) = Real.cos (90 * Real.pi / 180) := by
    rw [angle_eq]

  -- Using the fact that cos(90 degrees) = 0
  have cos_90 : Real.cos (90 * Real.pi / 180) = 0 := by
    -- This step can use a known trigonometric fact from mathlib
    exact Real.cos_pi_div_two

  -- Therefore
  rw [cos_eq, cos_90]
  exact rfl

end cos_450_eq_0_l649_649742


namespace part1_part2_l649_649113

def f (x : ℝ) : ℝ := abs (x - 1)
def g (x : ℝ) : ℝ := abs (2 * x - 1)

theorem part1 (x : ℝ) :
  abs (x + 4) ≤ x * abs (2 * x - 1) ↔ x ≥ 2 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, abs ((x + 2) - 1) + abs (x - 1) + a = 0 → False) ↔ a ≤ -2 :=
sorry

end part1_part2_l649_649113


namespace find_ellipse_eq_fixed_point_l649_649452

noncomputable def ellipse_condition (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2 + p.2^2 / b^2 = 1)} ∧
  (1 / 2) =  (real.sqrt (a^2 - b^2) / a)

theorem find_ellipse_eq_fixed_point:
  ∀ (a b : ℝ),
  ellipse_condition a b →
  let e := (2, 0)
  let c := real.sqrt (a^2 - b^2) in
  ∀ {x y : ℝ},
  let C_line := λ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 in
  C_line 2 0 →
  (e.1^2 / a^2 + e.2^2 / b^2 = 1) →
  (b^2 = 3) →
  ∀ (P: ℝ × ℝ), 
  P.1 = -1 →
  ∃ (x y : ℝ), 
  x = -1 / 4 ∧ y = 0 :=
sorry

end find_ellipse_eq_fixed_point_l649_649452


namespace trigonometric_expression_result_l649_649267

variable (α : ℝ)
variable (line_eq : ∀ x y : ℝ, 6 * x - 2 * y - 5 = 0)
variable (tan_alpha : Real.tan α = 3)

theorem trigonometric_expression_result :
  (Real.sin (Real.pi - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (Real.pi + α)) = -2 := 
by
  sorry

end trigonometric_expression_result_l649_649267


namespace functional_equation_solution_l649_649798

noncomputable def alpha : ℝ := (Real.sqrt 5 - 1) / 2

def f (n : ℕ) [n > 0] := (floor (n * alpha)).to_nat + 1

theorem functional_equation_solution :
  ∀ f : ℕ → ℕ, (∀ n, n > 0 → f(f(n)) + f(n + 1) = n + 2) → 
    (∀ n, n > 0 → f(n) = (floor (n * alpha) : ℕ) + 1) :=
by
  sorry

end functional_equation_solution_l649_649798


namespace evaluate_expression_l649_649036

theorem evaluate_expression (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 :=
by
  rw h
  -- The rest of the steps would go here, but are omitted
-- The proof would use the fact that 2^3 * 2^4 = 2^7 = 128
sorry

end evaluate_expression_l649_649036


namespace primes_quad_roots_l649_649869

theorem primes_quad_roots (p q m: ℕ) (hp: p.prime) (hq: q.prime)
  (h_root: ∀ x : ℕ, x^2 - 99 * x + m = 0 ↔ (x = p ∨ x = q)) :
  (p / q + q / p = 9413 / 194) :=
by {
  -- The following is a sketch implementation to ensure code can build successfully
  have h_sum : p + q = 99,
  { sorry }, -- Derived from Vieta's formulas
  have h_product : p * q = m,
  { sorry }, -- Derived from Vieta's formulas
  
  sorry -- Detailed proof fulfilling the equivalency requirements goes here
}

end primes_quad_roots_l649_649869


namespace locus_of_center_is_ellipse_l649_649102

noncomputable def is_tangent_externally_tangent (c₁ c₂: Point) (r₁ r₂ : ℝ) :=
  dist c₁ c₂ = r₁ + r₂

noncomputable def is_tangent_internally_tangent (c₁ c₂: Point) (r₁ r₂ : ℝ) :=
  dist c₁ c₂ = r₂ - r₁

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem locus_of_center_is_ellipse :
  let O₁ : Point := (1, 0),
      O₂ : Point := (-1, 0),
      r₁ : ℝ := 1,
      r₂ : ℝ := 4,
      dist_O₁O₂ : ℝ := distance O₁ O₂,
      c : Point,
      r : ℝ in
  is_tangent_externally_tangent c O₁ r r₁ ∧ is_tangent_internally_tangent c O₂ r r₂ →
  ∃ e p := ellipse e p 5, 
  all_points c (shape e p) (dist c O₁ + dist c O₂ = 5) :=
by sorry

end locus_of_center_is_ellipse_l649_649102


namespace factor_expression_l649_649797

theorem factor_expression (b : ℝ) : 275 * b^2 + 55 * b = 55 * b * (5 * b + 1) := by
  sorry

end factor_expression_l649_649797


namespace ellipse_focal_distance_l649_649713

theorem ellipse_focal_distance (m : ℝ) :
  (∀ x y : ℝ, (x^2 / 16 + y^2 / m = 1) ∧ (2 * Real.sqrt (16 - m) = 2 * Real.sqrt 7)) → m = 9 :=
by
  intro h
  sorry

end ellipse_focal_distance_l649_649713


namespace compare_a_b_l649_649098

theorem compare_a_b (m : ℝ) (h : m > 1) 
  (a : ℝ := (Real.sqrt (m+1)) - (Real.sqrt m))
  (b : ℝ := (Real.sqrt m) - (Real.sqrt (m-1))) : a < b :=
by
  sorry

end compare_a_b_l649_649098


namespace product_of_sums_not_1111111111_l649_649027

theorem product_of_sums_not_1111111111 (n : ℕ) : 
  (let a := 3 * n + 3 in let b := 3 * n + 12 in a * b ≠ 1111111111) :=
by 
  let a := 3 * n + 3
  let b := 3 * n + 12
  sorry

end product_of_sums_not_1111111111_l649_649027


namespace at_least_one_person_remains_dry_l649_649517

noncomputable def people_remain_dry (n : ℕ) (h1 : n = 2007) 
  (distinct_distances : ∀ i j : fin n, i ≠ j → ∃! d : ℝ, d = dist i j) : Prop :=
∃ i : fin n, ∀ j : fin n, i ≠ j → ∃ k : fin n, closest_shot j = i → closest_shot j ≠ k

def closest_shot (i : fin 2007) : fin 2007 := sorry

theorem at_least_one_person_remains_dry : people_remain_dry 2007 (rfl) sorry := sorry

end at_least_one_person_remains_dry_l649_649517


namespace all_lines_pass_through_centroid_l649_649080

-- Define points and triangles
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

-- Define line with equation ax + by + c = 0
structure Line :=
(a b c : ℝ)

-- Distance from a point to a line
def distance_to_line (P : Point) (l : Line) : ℝ :=
(abs (l.a * P.x + l.b * P.y + l.c)) / (real.sqrt (l.a^2 + l.b^2))

-- Centroid of a triangle
def centroid (t : Triangle) : Point :=
{
  x := (t.A.x + t.B.x + t.C.x) / 3,
  y := (t.A.y + t.B.y + t.C.y) / 3
}

-- The problem statement
theorem all_lines_pass_through_centroid (t : Triangle) (l : Line) :
  distance_to_line t.A l = (distance_to_line t.B l + distance_to_line t.C l) →
  ∃ M : Point, M = centroid t ∧ (l.a * M.x + l.b * M.y + l.c) = 0 :=
by
  sorry

end all_lines_pass_through_centroid_l649_649080


namespace ratio_AF_AB_eqn_circle_D_l649_649861

open Real

-- Condition: Parabola C with y² = 4mx, m > 0
constant m : ℝ
axiom m_pos : m > 0

-- Condition: Line x - y - m = 0 intersects the parabola C at points A and B
def parabola (x y : ℝ) := y^2 = 4 * m * x
def intersect_line (x y : ℝ) := x - y - m = 0

-- Coordinates of the points of intersection A and B where A is above the x-axis and B is below the x-axis
constant A B : ℝ × ℝ
axiom A_intersect_parabola : parabola A.1 A.2 
axiom B_intersect_parabola : parabola B.1 B.2
axiom A_intersect_line : intersect_line A.1 A.2
axiom B_intersect_line : intersect_line B.1 B.2
axiom A_above_x_axis : A.2 > 0
axiom B_below_x_axis : B.2 < 0

-- Focus of the parabola
def focus : ℝ × ℝ := (m, 0)

-- Question 1: Prove that AF / AB = (sqrt(2) + 1) / (sqrt(2) - 1)
theorem ratio_AF_AB : (dist A focus) / (dist A B) = (sqrt 2 + 1) / (sqrt 2 - 1) := sorry

-- Question 2: Find the equation of circle D that passes through points A and B and is tangent to the line x + y + 3 = 0
constant D_center : ℝ × ℝ
axiom chord_length_AB : dist A B = 8
axiom circle_D_contains_A : dist A D_center = sqrt ((dist A B)^2 / 2)
axiom circle_D_contains_B : dist B D_center = sqrt ((dist A B)^2 / 2)
axiom tangent_line : dist D_center (3, -3) = 4 * sqrt 2

theorem eqn_circle_D : 
(D_center = (3 + 2 * sqrt 2, 2 - 2 * sqrt 2) ∨ D_center = (3 - 2 * sqrt 2, 2 + 2 * sqrt 2)) ∧
(circle_eqn : ℝ → ℝ → Prop := λ x y, (x - D_center.1)^2 + (y - D_center.2)^2 = 32) := sorry

end ratio_AF_AB_eqn_circle_D_l649_649861


namespace find_a_l649_649105

-- Conditions
def velocity (t : ℝ) : ℝ := t.sqrt
def displacement (a : ℝ) : ℝ := ∫ t in 0..a, velocity t

-- Statement to prove
theorem find_a (a : ℝ) (h : displacement a = 18) : a = 9 := 
sorry

end find_a_l649_649105
