import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GcdLcm.Basic
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.FinVecttors
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace four_divides_sum_of_squares_iff_even_l361_361873

theorem four_divides_sum_of_squares_iff_even (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (4 ∣ (a^2 + b^2 + c^2)) ↔ (Even a ∧ Even b ∧ Even c) :=
by
  sorry

end four_divides_sum_of_squares_iff_even_l361_361873


namespace average_movers_l361_361009

noncomputable def average_people_per_hour (total_people : ℕ) (total_hours : ℕ) : ℝ :=
  total_people / total_hours

theorem average_movers :
  average_people_per_hour 5000 168 = 29.76 :=
by
  sorry

end average_movers_l361_361009


namespace correct_prediction_l361_361371

-- Declare the permutation type
def Permutation := (ℕ × ℕ × ℕ × ℕ × ℕ)

-- Declare the actual finish positions
def actual_positions : Permutation := (3, 2, 1, 4, 5)

-- Declare conditions
theorem correct_prediction 
  (P1 : Permutation) (P2 : Permutation) 
  (hP1 : P1 = (1, 2, 3, 4, 5))
  (hP2 : P2 = (1, 2, 3, 4, 5))
  (hP1_correct : ∃! i ∈ (1..5), (λ i, nth (P1.to_list) (i - 1) == nth (actual_positions.to_list))  = 3)
  (hP2_correct : ∃! i ∈ (1..5), (λ i, nth (P2.to_list) (i - 1) == nth (actual_positions.to_list))  = 2)
  : actual_positions = (3, 2, 1, 4, 5) 
:= sorry

end correct_prediction_l361_361371


namespace distance_from_origin_to_point_l361_361804

-- Define the specific points in the coordinate system
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (8, -15)

-- The distance formula in Euclidean space
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem statement
theorem distance_from_origin_to_point : distance origin point = 17 := 
by
  sorry

end distance_from_origin_to_point_l361_361804


namespace sqrt_mul_sqrt_l361_361987

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361987


namespace pedro_average_speed_l361_361889

-- Definitions based on conditions
def initial_odometer_reading : ℕ := 2332
def final_odometer_reading : ℕ := 2552
def total_active_riding_time : ℕ := 8 -- in hours

-- Problem statement to be proven
theorem pedro_average_speed :
  (final_odometer_reading - initial_odometer_reading) / total_active_riding_time = 27.5 :=
by
  -- Skipping the proof
  sorry

end pedro_average_speed_l361_361889


namespace distance_from_origin_to_point_l361_361813

theorem distance_from_origin_to_point : 
  ∀ (x y : ℝ), x = 8 → y = -15 → real.sqrt ((x-0)^2 + (y-0)^2) = 17 :=
by
  intros x y hx hy
  rw [hx, hy]
  norm_num
  rw [real.sqrt_eq_rpow]
  norm_num
  sorry

end distance_from_origin_to_point_l361_361813


namespace largest_number_after_87_minutes_l361_361068

def operation (x : ℕ) : ℕ := if x % 100 = 0 then x / 100 else x - 1

def minutes_operations (x : ℕ) (n : ℕ) : ℕ :=
nat.iterate operation n x

theorem largest_number_after_87_minutes (x : ℕ) (h : 1 ≤ x ∧ x ≤ 2150) :
  ∃ y, y ∈ {minutes_operations n 87 | n ∈ finset.range 2150} ∧ y = 2012 :=
by {
  sorry
}

end largest_number_after_87_minutes_l361_361068


namespace inequality_proof_l361_361454

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b * c = 1) : 
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l361_361454


namespace solve_for_s_l361_361770

theorem solve_for_s (k s : ℝ) 
  (h1 : 7 = k * 3^s) 
  (h2 : 126 = k * 9^s) : 
  s = 2 + Real.log 2 / Real.log 3 := by
  sorry

end solve_for_s_l361_361770


namespace inequality_abc_l361_361448

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1):
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by
  sorry

end inequality_abc_l361_361448


namespace distance_to_point_is_17_l361_361827

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem distance_to_point_is_17 :
  distance_from_origin 8 (-15) = 17 :=
by
  sorry

end distance_to_point_is_17_l361_361827


namespace flagpole_break_ratio_l361_361595

-- Define the height of the flagpole
def original_height : ℝ := 12

-- Define the height from the base at which the flagpole broke
def break_height_from_base : ℝ := 7

-- Define the correct answer as the point where the flagpole broke in relation to its height
def break_point_ratio : ℝ := break_height_from_base / original_height

-- Theorem: The flagpole broke at the point equal to the height ratio 7/12
theorem flagpole_break_ratio : break_point_ratio = 7 / 12 := 
by 
  -- Here we would write the proof, but we'll use sorry for now
  sorry

end flagpole_break_ratio_l361_361595


namespace athlete_positions_l361_361386

theorem athlete_positions {A B C D E : Type} 
  (h1 : A = 1 ∧ B = 2 ∧ C = 3 ∧ D = 4 ∧ E = 5)
  (h2 : C = 1 ∧ E = 2 ∧ A = 3 ∧ B = 4 ∧ D = 5)
  (h3 : (A = 1 ∧ B = 2 ∧ C = 3 ∧ ¬(D = 4 ∨ E = 5)) ∨ 
       (A = 1 ∧ ¬(B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)) ∨ 
       (B = 2 ∧ C = 3 ∧ ¬(A = 1 ∨ D = 4 ∨ E = 5)) ∨ 
       (¬(A = 1 ∨ B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)))
  (h4 : (C = 1 ∧ ¬(A = 3 ∨ B = 4 ∨ D = 5)) ∨ 
       (A = 3 ∧ ¬(C = 1 ∨ B = 4 ∨ D = 5)) ∨ 
       (B = 4 ∧ ¬(C = 1 ∨ A = 3 ∨ D = 5)))
  : (C = 1 ∧ B = 2 ∧ A = 3 ∧ D = 4 ∧ E = 5) :=
by sorry

end athlete_positions_l361_361386


namespace Q_div_P_l361_361526

theorem Q_div_P (P Q : ℚ) (h : ∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5 →
  P / (x + 3) + Q / (x * (x - 5)) = (x^2 - 3 * x + 8) / (x * (x + 3) * (x - 5))) :
  Q / P = 1 / 3 :=
by
  sorry

end Q_div_P_l361_361526


namespace fraction_of_acid_in_third_flask_l361_361140

def mass_of_acid_first_flask := 10
def mass_of_acid_second_flask := 20
def mass_of_acid_third_flask := 30
def mass_of_acid_first_flask_with_water (w : ℝ) := mass_of_acid_first_flask / (mass_of_acid_first_flask + w) = 1 / 20
def mass_of_acid_second_flask_with_water (W w : ℝ) := mass_of_acid_second_flask / (mass_of_acid_second_flask + (W - w)) = 7 / 30
def mass_of_acid_third_flask_with_water (W : ℝ) := mass_of_acid_third_flask / (mass_of_acid_third_flask + W)

theorem fraction_of_acid_in_third_flask (W w : ℝ) (h1 : mass_of_acid_first_flask_with_water w) (h2 : mass_of_acid_second_flask_with_water W w) :
  mass_of_acid_third_flask_with_water W = 21 / 200 :=
by
  sorry

end fraction_of_acid_in_third_flask_l361_361140


namespace john_total_jury_duty_days_l361_361026

-- Definitions based on the given conditions
def jury_selection_days : ℕ := 2
def trial_length_multiplier : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_in_a_day : ℕ := 24

-- Define the total days John spends on jury duty
def total_days_on_jury_duty : ℕ :=
  jury_selection_days +
  (trial_length_multiplier * jury_selection_days) +
  (deliberation_days * deliberation_hours_per_day) / hours_in_a_day

-- The theorem to prove
theorem john_total_jury_duty_days : total_days_on_jury_duty = 14 := by
  sorry

end john_total_jury_duty_days_l361_361026


namespace prime_divides_a_minus_3_l361_361856

theorem prime_divides_a_minus_3 (a p : ℤ) (hp : Prime p) (h1 : p ∣ 5 * a - 1) (h2 : p ∣ a - 10) : p ∣ a - 3 := by
  sorry

end prime_divides_a_minus_3_l361_361856


namespace perfect_square_concatenation_l361_361971

theorem perfect_square_concatenation :
  ∃ x y z t : ℕ,
    x ∈ {4, 5, 6, 7, 8, 9} ∧
    y ∈ {4, 5, 6, 7, 8, 9} ∧
    z ∈ {4, 5, 6, 7, 8, 9} ∧
    (t^2 = 10000 * x^2 + 100 * y^2 + z^2) ∧
    (t^2 = 166464 ∨ t^2 = 646416) :=
begin
  sorry
end

end perfect_square_concatenation_l361_361971


namespace goldfish_left_l361_361472

theorem goldfish_left (original_gf : ℕ) (disappeared_gf : ℕ) (remaining_gf : ℕ) (h1 : original_gf = 15) (h2 : disappeared_gf = 11) : remaining_gf = original_gf - disappeared_gf → remaining_gf = 4 :=
by {
  assume h,
  exact h.trans (by rw [h1, h2]),
}

end goldfish_left_l361_361472


namespace water_in_tank_l361_361121

theorem water_in_tank : ∀ (x : ℝ), x + 7 = 14.75 → x = 7.75 :=
begin
  intro x,
  intro h,
  linarith,
end

end water_in_tank_l361_361121


namespace perimeter_of_triangle_is_13_l361_361109

-- Conditions
noncomputable def perimeter_of_triangle_with_two_sides_and_third_root_of_eq : ℝ :=
  let a := 3
  let b := 6
  let c1 := 2 -- One root of the equation x^2 - 6x + 8 = 0
  let c2 := 4 -- Another root of the equation x^2 - 6x + 8 = 0
  if a + b > c2 ∧ a + c2 > b ∧ b + c2 > a then
    a + b + c2
  else
    0 -- not possible to form a triangle with these sides

-- Assertion
theorem perimeter_of_triangle_is_13 :
  perimeter_of_triangle_with_two_sides_and_third_root_of_eq = 13 := 
sorry

end perimeter_of_triangle_is_13_l361_361109


namespace smallest_n_correct_l361_361870

noncomputable def smallest_n (m : ℕ) : ℕ :=
  if h : odd m then
  let k := Nat.find (λ k, 2 ^ k ∣ m - 1) in
  2 ^ (1989 - k)
  else 0

theorem smallest_n_correct (m : ℕ) (h : odd m) : 
  ∃ n, (2 ^ 1989 ∣ m ^ n - 1) ∧ ∀ n', (2 ^ 1989 ∣ m ^ n' - 1) → n' ≥ smallest_n m := 
by
  sorry

end smallest_n_correct_l361_361870


namespace maximum_number_of_divisors_l361_361877

open Nat

noncomputable def square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p^2 ∣ n → false

theorem maximum_number_of_divisors (n : ℕ) (d : ℕ) (k : ℕ) :
  square_free n → n > 1 → d = 2^k → 
  ∃ S : Finset ℕ, S.card ≤ 2^(k-1) ∧ 
  ∀ a b ∈ S, ¬((a^2 + a * b - n) = c^2 for some c : ℕ) :=
sorry

end maximum_number_of_divisors_l361_361877


namespace ronald_units_bought_l361_361485

theorem ronald_units_bought (x : ℕ) (initial_investment : ℕ := 3000) (profit_margin : ℚ := 1/3) (selling_price_per_unit : ℕ := 20) :
  let profit := (profit_margin * initial_investment : ℚ)
  let total_amount := (initial_investment + profit)
  total_amount / selling_price_per_unit = x := 
  x = 200 :=
by
  -- skip proof
  sorry

end ronald_units_bought_l361_361485


namespace lisa_total_spoons_l361_361459

def total_baby_spoons (children : ℕ) (spoons_per_child : ℕ) : ℕ := 
  children * spoons_per_child

def total_special_spoons (baby_spoons : ℕ) (decorative_spoons : ℕ) : ℕ := 
  baby_spoons + decorative_spoons

def total_new_spoons (large_spoons : ℕ) (dessert_spoons : ℕ) (teaspoons : ℕ) : ℕ := 
  large_spoons + dessert_spoons + teaspoons

def total_spoons (special_spoons : ℕ) (new_spoons : ℕ) : ℕ := 
  special_spoons + new_spoons

theorem lisa_total_spoons : 
  let children := 6 in
  let spoons_per_child := 4 in
  let decorative_spoons := 4 in
  let large_spoons := 20 in
  let dessert_spoons := 10 in
  let teaspoons := 25 in
  total_spoons (total_special_spoons (total_baby_spoons children spoons_per_child) decorative_spoons) 
               (total_new_spoons large_spoons dessert_spoons teaspoons) = 83 :=
by
  sorry

end lisa_total_spoons_l361_361459


namespace rectangle_folding_problem_l361_361228

-- Define Rectangle ABCD, point E, and the folding operation
structure Rectangle (P Q R S : Type) :=
(a b : P)  -- Definitions of points as Types for simplicity

def midpoint {P : Type} (A B : P) : P := sorry -- Midpoint definition placeholder

def fold_triangle (P Q R : Type) : P × Q × R := sorry -- Folding operation placeholder

def trajectory_circle (M : Type) : Prop := sorry -- Definition for trajectory being a circle

-- The main proof problem
theorem rectangle_folding_problem (A B C D E A1 M : Type) 
  [Rectangle A B C D] (mid_E : E = midpoint A B) (mid_M : M = midpoint A1 C)
  (folding : Triangle A D E = fold_triangle A1 D E) (AB_eq_2AD : AB = 2 * AD) :
  (∀ (A1 : Type), (M = midpoint A1 C) → (∃ (BM_fixed : Prop), BM_fixed) ∧ (trajectory_circle M)) :=
begin
  sorry -- The actual proof will be here.
end

end rectangle_folding_problem_l361_361228


namespace dot_product_v1_v2_l361_361283

def v1 : ℝ × ℝ × ℝ := (4, -5, 2)
def v2 : ℝ × ℝ × ℝ := (-3, 3, -4)

theorem dot_product_v1_v2 : 
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = -35 := 
by
  sorry

end dot_product_v1_v2_l361_361283


namespace central_angle_unfolded_cone_l361_361590

-- Condition: A cone has three generatrices that are mutually perpendicular.
-- Let the length of the generatrix be l.
-- Derived answer: The central angle of the lateral surface when unfolded is \frac{2\sqrt{6}\pi}{3}.

theorem central_angle_unfolded_cone (l : ℝ) :
  let gen_perp := true in -- This encodes the condition that generatrices are mutually perpendicular
  let gen_length := l in
  gen_perp → (2 * Real.sqrt 6 * Real.pi / 3) = ? := 
sorry

end central_angle_unfolded_cone_l361_361590


namespace g_sum_eq_four_l361_361315

-- Definitions as per conditions in part a)
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def symmetric_about_line (f g : ℝ → ℝ) : Prop :=
  ∀ x, g (x) = f (x - (x - f (x)))

variable (f g : ℝ → ℝ)

-- Conditions
variables (h_odd : odd_function (λ x, f (2 * x + 2) - 1))
          (h_symm : symmetric_about_line f g)
          (x₁ x₂ : ℝ)
          (h_sum : x₁ + x₂ = 2)

-- Theorem statement
theorem g_sum_eq_four : g x₁ + g x₂ = 4 := sorry

end g_sum_eq_four_l361_361315


namespace unique_sequence_137_l361_361548

theorem unique_sequence_137 :
  ∃ (a : Fin 137 → ℕ), StrictMono a ∧ ∑ i, 2 ^ (a i) = (2^289 + 1) / (2^17 + 1) :=
by
  sorry

end unique_sequence_137_l361_361548


namespace sum_distances_equal_height_l361_361895

open Classical

theorem sum_distances_equal_height
  (ABCD : Type)
  [RegularTetrahedron ABCD] 
  (h : ℝ)
  (P : Point ABCD)
  (a b c d : ℝ)
  (h_distances : distances_to_faces P = (a, b, c, d))
  (h_volume : tetrahedron_volume ABCD = (1/3) * face_area ABCD * h)
  (h_subvolume : tetrahedron_volume ABCD = 
                (1/3) * face_area ABCD * a +
                (1/3) * face_area ABCD * b +
                (1/3) * face_area ABCD * c +
                (1/3) * face_area ABCD * d) :
  h = a + b + c + d :=
by 
  sorry

end sum_distances_equal_height_l361_361895


namespace find_actual_positions_l361_361390

namespace RunningCompetition

variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]

/-- The first prediction: $A$ finishes first, $B$ is second, $C$ is third, $D$ is fourth, $E$ is fifth. --/
def first_prediction := [A, B, C, D, E]

/-- The second prediction: $C$ finishes first, $E$ is second, $A$ is third, $B$ is fourth, $D$ is fifth. --/
def second_prediction := [C, E, A, B, D]

/-- The first prediction correctly predicted the positions of exactly three athletes. --/
def first_prediction_correct_count : ℕ := 3

/-- The second prediction correctly predicted the positions of exactly two athletes. --/
def second_prediction_correct_count : ℕ := 2

/-- Proves that the actual finishing order is $C, B, A, D, E$. --/
theorem find_actual_positions : 
  ∀ (order : list (Type)), 
  (order = [C, B, A, D, E]) →
  list.count order first_prediction_correct_count = 3 ∧ 
  list.count order second_prediction_correct_count = 2 :=
by
  sorry

end RunningCompetition

end find_actual_positions_l361_361390


namespace john_weight_end_l361_361425

def initial_weight : ℝ := 220
def loss_percentage : ℝ := 0.1
def weight_loss : ℝ := loss_percentage * initial_weight
def weight_gain_back : ℝ := 2
def net_weight_loss : ℝ := weight_loss - weight_gain_back
def final_weight : ℝ := initial_weight - net_weight_loss

theorem john_weight_end :
  final_weight = 200 := 
by 
  sorry

end john_weight_end_l361_361425


namespace distance_from_origin_to_point_l361_361830

open Real

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 →
  y = -15 →
  sqrt (x^2 + y^2) = 17 :=
by
  intros hx hy
  rw [hx, hy]
  norm_num
  rw ←sqrt_sq (show 0 ≤ 289 by norm_num)
  norm_num

end distance_from_origin_to_point_l361_361830


namespace probability_obtuse_angle_l361_361074

-- Define the vertices of the pentagon
def F : ℝ × ℝ := (0, 3)
def G : ℝ × ℝ := (3, 0)
def H : ℝ × ℝ := (2 * Real.pi + 2, 0)
def I : ℝ × ℝ := (2 * Real.pi + 2, 3)
def J : ℝ × ℝ := (0, 3)

-- Define the center and radius of the semicircle where angle FQG is right angle
def semicircle_center : ℝ × ℝ := (1.5, 1.5)
def semicircle_radius : ℝ := Real.sqrt 4.5

-- Define the area of the pentagon
noncomputable def area_pentagon : ℝ := 6 * Real.pi + 3

-- Define the area of the semicircle
noncomputable def area_semicircle : ℝ := 0.5 * Real.pi * (Real.sqrt 4.5) ^ 2

-- Lean statement for the probability calculation
theorem probability_obtuse_angle :
  (area_semicircle / area_pentagon) = 3 / (8 * (2 * Real.pi + 1)) :=
by
  sorry

end probability_obtuse_angle_l361_361074


namespace Gage_skating_minutes_l361_361701

theorem Gage_skating_minutes (d1 d2 d3 : ℕ) (m1 m2 : ℕ) (avg : ℕ) (h1 : d1 = 6) (h2 : d2 = 4) (h3 : d3 = 1) (h4 : m1 = 80) (h5 : m2 = 105) (h6 : avg = 95) : 
  (d1 * m1 + d2 * m2 + d3 * x) / (d1 + d2 + d3) = avg ↔ x = 145 := 
by 
  sorry

end Gage_skating_minutes_l361_361701


namespace solve_system_l361_361503

theorem solve_system :
  ∀ (x y : ℝ),
    (4 * x ^ 2 - 3 * y = x * y ^ 3) ∧ (x ^ 2 + x ^ 3 * y ^ 2 = 2 * y) ↔
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = -real.root 5 (5/4) ∧ y = -real.root 5 (-50)) :=
by sorry

end solve_system_l361_361503


namespace integral_proof_l361_361920

-- Defining the binomial coefficient as a helper function
noncomputable def binomial_coeff (n k : ℕ) : ℝ :=
  if h : k ≤ n then (nat.choose n k : ℝ) else 0

-- Condition 1: Coefficient of the second term in the binomial expansion
def binomial_condition (a : ℝ) : Prop :=
  binomial_coeff 3 1 * a^2 * (- real.sqrt 3 / 6) = - real.sqrt 3 / 2

-- Condition 2: a must be positive
def pos_condition (a : ℝ) : Prop := 
  a > 0

-- Integral from -2 to a of x^2 dx
def integral_condition (a : ℝ) : Prop :=
  ∫ x in -2..a, x^2 = 3

-- Final statement to be proved
theorem integral_proof : ∃ a : ℝ, binomial_condition a ∧ pos_condition a ∧ integral_condition a := 
begin
  sorry
end

end integral_proof_l361_361920


namespace actual_positions_correct_l361_361363

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end actual_positions_correct_l361_361363


namespace distance_from_origin_to_point_l361_361807

-- Define the specific points in the coordinate system
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (8, -15)

-- The distance formula in Euclidean space
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem statement
theorem distance_from_origin_to_point : distance origin point = 17 := 
by
  sorry

end distance_from_origin_to_point_l361_361807


namespace quadratic_solution_l361_361691

theorem quadratic_solution (m : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_roots : ∀ x, 2 * x^2 + 4 * m * x + m = 0 ↔ x = x₁ ∨ x = x₂) 
  (h_sum_squares : x₁^2 + x₂^2 = 3 / 16) :
  m = -1 / 8 :=
by
  sorry

end quadratic_solution_l361_361691


namespace lcm_of_18_50_120_l361_361286

theorem lcm_of_18_50_120 : Nat.lcm (Nat.lcm 18 50) 120 = 1800 := by
  sorry

end lcm_of_18_50_120_l361_361286


namespace polar_to_rectangular_l361_361650

theorem polar_to_rectangular (r θ : ℝ) (hr : r = 6) (hθ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (3, -3 * Real.sqrt 3) :=
by
  -- Definitions and assertions from the conditions
  have cos_theta : Real.cos (5 * Real.pi / 3) = 1 / 2 :=
    by sorry  -- detailed trigonometric proof is omitted
  have sin_theta : Real.sin (5 * Real.pi / 3) = - Real.sqrt 3 / 2 :=
    by sorry  -- detailed trigonometric proof is omitted

  -- Proof that the converted coordinates match the expected result
  rw [hr, hθ, cos_theta, sin_theta]
  simp
  -- Detailed proof steps to verify (6 * (1 / 2), 6 * (- Real.sqrt 3 / 2)) = (3, -3 * Real.sqrt 3) omitted
  sorry

end polar_to_rectangular_l361_361650


namespace problem_1_problem_2_problem_3_problem_4_l361_361237

-- Condition 1
def Q1 := (+12) - (-18) + (-7) - (+15)
-- Proof step for Q1
theorem problem_1 : Q1 = 8 := 
by
  simp
  exact 8
-- Condition 2
def Q2 := (-81) / (9/4) * (4/9) / (-16)
-- Proof step for Q2
theorem problem_2 : Q2 = 1 := 
by
  simp
  exact 1
-- Condition 3
def Q3 := (1/3 - 5/6 + 7/9) * (-18)
-- Proof step for Q3
theorem problem_3 : Q3 = -5 := 
by
  simp
  exact -5
-- Condition 4
def Q4 := -1^4 - 1/5 * (2 - (-3))^2
-- Proof step for Q4
theorem problem_4 : Q4 = -6 := 
by
  simp
  exact -6

end problem_1_problem_2_problem_3_problem_4_l361_361237


namespace num_of_chords_l361_361259

theorem num_of_chords (n : ℕ) (h : n = 8) : (n.choose 2) = 28 :=
by
  -- Proof of this theorem will be here
  sorry

end num_of_chords_l361_361259


namespace total_time_proof_l361_361463

def time_assemble : ℕ := 1
def time_bake_normal : ℝ := 1.5
def time_decorate : ℕ := 1
def time_special_order : ℕ := 1 -- Two orders, each 30 minutes gives total 1 hour

def time_first_oven : ℝ := time_bake_normal * 2
def time_second_oven : ℝ := time_bake_normal * 3
def time_third_oven : ℝ := time_bake_normal * 4

def total_time : ℝ := time_assemble + max time_first_oven (max time_second_oven time_third_oven) + time_special_order

theorem total_time_proof : total_time = 8 := by
  have h1 : time_assemble = 1 := by rfl
  have h2 : time_bake_normal = 1.5 := by rfl
  have h3 : time_decorate = 1 := by rfl
  have h4 : time_special_order = 1 := by rfl
  have h5 : time_first_oven = 3 := by calc
    time_bake_normal * 2 = 1.5 * 2 := by rfl
                 ... = 3 := by norm_num
  have h6 : time_second_oven = 4.5 := by calc
    time_bake_normal * 3 = 1.5 * 3 := by rfl
                  ... = 4.5 := by norm_num
  have h7 : time_third_oven = 6 := by calc
    time_bake_normal * 4 = 1.5 * 4 := by rfl
                  ... = 6 := by norm_num
  have h8 : max time_first_oven (max time_second_oven time_third_oven) = 6 := by
    apply max_eq_right
    apply le_max_right
    apply le_max_left
  calc
    total_time = time_assemble + max time_first_oven (max time_second_oven time_third_oven) + time_special_order := by rfl
            ... = 1 + 6 + 1 := by rw [h1, h8, h4]
            ... = 8 := by norm_num

end total_time_proof_l361_361463


namespace point_in_third_quadrant_l361_361947

theorem point_in_third_quadrant :
  let sin2018 := Real.sin (2018 * Real.pi / 180)
  let tan117 := Real.tan (117 * Real.pi / 180)
  sin2018 < 0 ∧ tan117 < 0 → 
  (sin2018 < 0 ∧ tan117 < 0) :=
by
  intros
  sorry

end point_in_third_quadrant_l361_361947


namespace sum_of_divisors_30_l361_361981

-- Given a number n (specifically 30 in this problem)
def n : ℕ := 30

-- Define what it means to be a divisor
def is_divisor (d n : ℕ) : Prop := n % d = 0

-- Define the set of all positive divisors of a number
def divisors (n : ℕ) : set ℕ := {d | is_divisor d n ∧ d > 0}

-- Define the sum of a finite set of natural numbers
def sum_of_divisors (n : ℕ) : ℕ :=
  (divisors n).sum id

-- Given problem: sum of divisors of n
theorem sum_of_divisors_30 : sum_of_divisors 30 = 72 :=
sorry

end sum_of_divisors_30_l361_361981


namespace correct_propositions_l361_361309

open Real

variable {a b : ℝ}

-- We restate conditions a, b ∈ ℝ⁺ as positive real numbers
def pos_real (x : ℝ) := 0 < x

-- Proposition 1: If a^2 - b^2 = 1, then a - b < 1
def prop1 (a b : ℝ) (ha : pos_real a) (hb : pos_real b) (h_eq : a^2 - b^2 = 1) : Prop :=
  a - b < 1

-- Proposition 2: If 1/b - 1/a = 1, then a - b < 1
def prop2 (a b : ℝ) (ha : pos_real a) (hb : pos_real b) (h_eq : 1 / b - 1 / a = 1) : Prop :=
  a - b < 1

-- Proposition 3: If |sqrt a - sqrt b| = 1, then |a - b| < 1
def prop3 (a b : ℝ) (ha : pos_real a) (hb : pos_real b) (h_eq : abs (sqrt a - sqrt b) = 1) : Prop :=
  abs (a - b) < 1

-- Proposition 4: If |a^2 - b^2| = 1, then |a - b| < 1
def prop4 (a b : ℝ) (ha : pos_real a) (hb : pos_real b) (h_eq : abs (a^2 - b^2) = 1) : Prop :=
  abs (a - b) < 1

-- Final proof statement encapsulating correct propositions
theorem correct_propositions (a b : ℝ) (ha : pos_real a) (hb : pos_real b) :
  (prop1 a b ha hb) ∧ (prop4 a b ha hb) := 
by
  sorry

end correct_propositions_l361_361309


namespace sum_of_digits_next_multiple_l361_361912

theorem sum_of_digits_next_multiple :
  ∃ P S : ℕ, ∃ E : ℕ,
  (E = 4) ∧
  (P = S + 2) ∧
  (∃ n : ℕ, n < 6 ∧ (S + n) % (E + n) = 0) ∧
  (∃ k : ℕ, let next_multiple := P + k in (next_multiple % E = 0) ∧
  (next_multiple / 10 + next_multiple % 10 = 8)) :=
by
  -- Proof goes here.
  sorry

end sum_of_digits_next_multiple_l361_361912


namespace log_eq_seven_half_l361_361261

theorem log_eq_seven_half (h1 : 64 = 4^3) (h2 : sqrt 4 = 4^(1/2)) :
  log 4 (64 * sqrt 4) = 7/2 :=
sorry

end log_eq_seven_half_l361_361261


namespace sqrt_49_mul_sqrt_25_l361_361999

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l361_361999


namespace expected_value_sum_of_rook_positions_l361_361931

-- Define the chessboard size and the number of squares
def board_size : ℕ := 8
def total_squares : ℕ := board_size * board_size

-- Define the number of rooks
def num_rooks : ℕ := 6

-- Define the positions of the rooks
noncomputable def rook_positions : Fin num_rooks → ℕ := λ k => sorry

-- The sum of the positions of the rooks
noncomputable def sum_positions : ℕ := ∑ k in (Finset.range num_rooks), rook_positions ⟨k, sorry⟩

-- The expected value of a rook's position
noncomputable def expected_value_position : ℚ := (1 + total_squares) / 2

-- The expected value of the sum of the positions
noncomputable def expected_value_sum_positions : ℚ := num_rooks * expected_value_position

-- The theorem to prove
theorem expected_value_sum_of_rook_positions : expected_value_sum_positions = 195 := by
  sorry

end expected_value_sum_of_rook_positions_l361_361931


namespace find_a_plus_b_l361_361048

theorem find_a_plus_b (a b : ℝ) (f g : ℝ → ℝ) (h1 : ∀ x, f x = a * x + b) (h2 : ∀ x, g x = 3 * x - 4) 
(h3 : ∀ x, g (f x) = 4 * x + 5) : a + b = 13 / 3 :=
sorry

end find_a_plus_b_l361_361048


namespace number_of_positive_four_digit_integers_divisible_by_8_l361_361333

-- We define the condition that a number is divisible by 8 if its last three digits are also divisible by 8
def divisible_by_8 (n : ℕ) : Prop :=
  n % 8 = 0

-- We define a function to count the number of four-digit integers divisible by 8
def count_four_digit_integers_divisible_by_8 : ℕ :=
  let count_last_three_digits := (1000 / 8) - (999 / 8) in  -- Count multiples of 8 in range 000 to 999
  let valid_thousands_digit := 9 in
  valid_thousands_digit * count_last_three_digits

-- Finally, we state the theorem with the given conditions proving the result
theorem number_of_positive_four_digit_integers_divisible_by_8 :
  count_four_digit_integers_divisible_by_8 = 1125 := by
  sorry

end number_of_positive_four_digit_integers_divisible_by_8_l361_361333


namespace ab_multiple_of_7_2010_l361_361477

theorem ab_multiple_of_7_2010 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 7 ^ 2009 ∣ a^2 + b^2) : 7 ^ 2010 ∣ a * b :=
by
  sorry

end ab_multiple_of_7_2010_l361_361477


namespace sqrt_49_mul_sqrt_25_l361_361995

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l361_361995


namespace solve_trig_eq_has_values_l361_361020

noncomputable def solve_trig_eq (x y : ℝ) : ℝ :=
  if h1 : (cos x - sin x) / sin y = (1 / (3 * √2)) * cot ((x + y) / 2) ∧
         (sin x + cos x) / cos y = -6 * √2 * tan ((x + y) / 2) then
    tan (x + y)
  else
    0

theorem solve_trig_eq_has_values (x y : ℝ) (z : ℝ) :
  ( (cos x - sin x) / sin y = (1 / (3 * √2)) * cot ((x + y) / 2) ∧
    (sin x + cos x) / cos y = -6 * √2 * tan ((x + y) / 2) ) →
  (z = -1 ∨ z = 12 / 35 ∨ z = -12 / 35) →
  solve_trig_eq x y = z :=
by
  intros hcond hz
  sorry

end solve_trig_eq_has_values_l361_361020


namespace problem_part1_problem_part2_l361_361059

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else (n + 1) * (S_seq n) / n

noncomputable def S_seq (n : ℕ) : ℕ :=
  finset.sum (finset.range (n + 1)) a_seq

def T_seq (n : ℕ) : ℕ :=
  finset.sum (finset.range (n + 1)) S_seq

theorem problem_part1 (n : ℕ) (h : n ≥ 1) :
  S_seq (n + 1) / (n + 1) = 2 * (S_seq n / n) := sorry

theorem problem_part2 (n : ℕ) (h : n ≥ 1) :
  T_seq n = (n - 1) * 2^n + 1 := sorry

end problem_part1_problem_part2_l361_361059


namespace determinant_of_cross_product_matrix_l361_361041

/-- Given vectors p, q, r, let E be the determinant of the matrix whose columns are p, q, and r.
    The determinant of the matrix whose columns are p × q, q × r, and r × p is equal to E^2.
    Hence, the ordered pair (m, l) is (1, 2). -/
theorem determinant_of_cross_product_matrix {p q r : ℝ^3} 
  (E : ℝ) (hE : E = p.dot (q.cross r)) :
  ∃ m l, m * (E^l) = determinant_of_3x3_matrix (p.cross q) (q.cross r) (r.cross p) ∧ m = 1 ∧ l = 2 := 
sorry

end determinant_of_cross_product_matrix_l361_361041


namespace andrew_age_l361_361625

variables (a g : ℝ)

theorem andrew_age (h1 : g = 15 * a) (h2 : g - a = 60) : a = 30 / 7 :=
by sorry

end andrew_age_l361_361625


namespace unsafe_to_overtake_l361_361551

noncomputable def mph_to_fps (v : ℝ) : ℝ := v * 5280 / 3600

def can_overtake_safely (V_A V_B V_Cflat t_AB t_AC : ℝ) : Prop :=
  let V_C_inclined := 0.9 * V_Cflat
  let rel_velocity_AB := V_A - V_B
  let rel_velocity_AC := V_A + V_C_inclined
  let t_AB := t_AB / (rel_velocity_AB * mph_to_fps 1)
  let t_AC := t_AC / (rel_velocity_AC * mph_to_fps 1)
  t_AB < t_AC

theorem unsafe_to_overtake :
  can_overtake_safely 55 45 55 50 200 = False := 
by 
  let V_A := 55
  let V_B := 45
  let V_Cflat := 55
  let t_AB := 50
  let t_AC := 200
  let V_C_inclined := 0.9 * V_Cflat
  let rel_velocity_AB := V_A - V_B
  let rel_velocity_AC := V_A + V_C_inclined
  have h_AB : ℝ := t_AB / (rel_velocity_AB * mph_to_fps 1)
  have h_AC : ℝ := t_AC / (rel_velocity_AC * mph_to_fps 1)
  have comp : h_AB < h_AC := by
    have h1 : rel_velocity_AB * mph_to_fps 1 = 10 * mph_to_fps 1 := by sorry
    have h2 : rel_velocity_AC * mph_to_fps 1 = 104.5 * mph_to_fps 1 := by sorry
    have h_AB := 50 / h1
    have h_AC := 200 / h2
    sorry 

  exact False.intro sorry

end unsafe_to_overtake_l361_361551


namespace vacation_cost_difference_l361_361535

theorem vacation_cost_difference :
  ∀ (total_cost number_of_people_3 number_of_people_5 : ℕ),
  total_cost = 375 →
  number_of_people_3 = 3 →
  number_of_people_5 = 5 →
  (total_cost / number_of_people_3 - total_cost / number_of_people_5 = 50) :=
by
  intros total_cost number_of_people_3 number_of_people_5 h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end vacation_cost_difference_l361_361535


namespace concur_midpoints_of_quadrilateral_l361_361481

variables {Point : Type} [AffineSpace Point ℝ]

structure ConvexQuadrilateral (A B C D : Point) : Prop :=
(convex : Convex ℝ ({A, B, C, D} : Set Point))

theorem concur_midpoints_of_quadrilateral
  (A B C D K L M N P Q : Point)
  [ConvexQuadrilateral A B C D]
  (midpoint_AB : midpoint ℝ A B = K)
  (midpoint_BC : midpoint ℝ B C = L)
  (midpoint_CD : midpoint ℝ C D = M)
  (midpoint_DA : midpoint ℝ D A = N)
  (midpoint_AC : midpoint ℝ A C = P)
  (midpoint_BD : midpoint ℝ B D = Q) :
  ∃ O : Point, line_through ℝ K M O ∧ line_through ℝ L N O ∧ line_through ℝ P Q O :=
sorry

end concur_midpoints_of_quadrilateral_l361_361481


namespace length_of_semi_minor_axis_l361_361753

theorem length_of_semi_minor_axis (b : ℝ) (h1 : b > 0) 
  (h2 : eccentricity = 3 / 2) 
  (h3 : ∀ x y : ℝ, (x^2 / 4) - (y^2 / b^2) = 1) :
  b = real.sqrt 5 :=
by
  sorry

end length_of_semi_minor_axis_l361_361753


namespace actual_positions_correct_l361_361359

def athletes := {A, B, C, D, E}
def p1 : athletes → ℕ
| A := 1
| B := 2
| C := 3
| D := 4
| E := 5

def p2 : athletes → ℕ
| C := 1
| E := 2
| A := 3
| B := 4
| D := 5

def actual_positions : athletes → ℕ
| A := 3
| B := 2
| C := 1
| D := 4
| E := 5

theorem actual_positions_correct :
  (∑ a in athletes, ite (actual_positions a = p1 a) 1 0 = 3) ∧ 
  (∑ a in athletes, ite (actual_positions a = p2 a) 1 0 = 2) :=
by
  sorry

end actual_positions_correct_l361_361359


namespace cosine_sine_equation_solutions_l361_361335

theorem cosine_sine_equation_solutions :
  (∃! x ∈ set.Icc (0:ℝ) (2 * real.pi), cos ((real.pi / 2) * sin x) = sin ((real.pi / 2) * cos x)) :=
sorry

end cosine_sine_equation_solutions_l361_361335


namespace regression_line_passes_through_sample_mean_point_l361_361110

theorem regression_line_passes_through_sample_mean_point
  (a b : ℝ) (x y : ℝ)
  (hx : x = a + b*x) :
  y = a + b*x :=
by sorry

end regression_line_passes_through_sample_mean_point_l361_361110


namespace sum_of_complex_numbers_l361_361236

def z1 : ℂ := 2 + 5 * Complex.i
def z2 : ℂ := 3 - 7 * Complex.i

theorem sum_of_complex_numbers : (z1 + z2 = 5 - 2 * Complex.i) ∧ (z1 * z2 = -29 + Complex.i) :=
by
  sorry

end sum_of_complex_numbers_l361_361236


namespace lemons_needed_l361_361189

theorem lemons_needed (lemons_per_48_gallons : ℚ) (limeade_factor : ℚ) (total_gallons : ℚ) (split_gallons : ℚ) :
  lemons_per_48_gallons = 36 / 48 →
  limeade_factor = 2 →
  total_gallons = 18 →
  split_gallons = total_gallons / 2 →
  (split_gallons * (36 / 48) + split_gallons * (2 * (36 / 48))) = 20.25 :=
by
  intros h1 h2 h3 h4
  sorry

end lemons_needed_l361_361189


namespace correct_prediction_l361_361374

-- Declare the permutation type
def Permutation := (ℕ × ℕ × ℕ × ℕ × ℕ)

-- Declare the actual finish positions
def actual_positions : Permutation := (3, 2, 1, 4, 5)

-- Declare conditions
theorem correct_prediction 
  (P1 : Permutation) (P2 : Permutation) 
  (hP1 : P1 = (1, 2, 3, 4, 5))
  (hP2 : P2 = (1, 2, 3, 4, 5))
  (hP1_correct : ∃! i ∈ (1..5), (λ i, nth (P1.to_list) (i - 1) == nth (actual_positions.to_list))  = 3)
  (hP2_correct : ∃! i ∈ (1..5), (λ i, nth (P2.to_list) (i - 1) == nth (actual_positions.to_list))  = 2)
  : actual_positions = (3, 2, 1, 4, 5) 
:= sorry

end correct_prediction_l361_361374


namespace find_ages_l361_361198

theorem find_ages (M F S : ℕ) 
  (h1 : M = 2 * F / 5)
  (h2 : M + 10 = (F + 10) / 2)
  (h3 : S + 10 = 3 * (F + 10) / 4) :
  M = 20 ∧ F = 50 ∧ S = 35 := 
by
  sorry

end find_ages_l361_361198


namespace polynomial_identity_l361_361669

-- Define the conditions as Lean definitions
def P (x : ℝ) : ℝ := sorry

-- State the conditions given in the problem
axiom P_zero : P 0 = 0
axiom P_property : ∀ x : ℝ, P (x^2 + 1) = P (x)^2 + 1

-- State the theorem equivalent to proving the question == answer given 
-- the conditions.
theorem polynomial_identity : ∀ x : ℝ, P x = x := 
begin
  sorry
end

end polynomial_identity_l361_361669


namespace twice_a_minus_4_nonnegative_l361_361665

theorem twice_a_minus_4_nonnegative (a : ℝ) : 2 * a - 4 ≥ 0 ↔ 2 * a - 4 = 0 ∨ 2 * a - 4 > 0 := 
by
  sorry

end twice_a_minus_4_nonnegative_l361_361665


namespace right_rect_prism_volume_l361_361125

theorem right_rect_prism_volume (a b c : ℝ) 
  (h1 : a * b = 56) 
  (h2 : b * c = 63) 
  (h3 : a * c = 36) : 
  a * b * c = 504 := by
  sorry

end right_rect_prism_volume_l361_361125


namespace domain_of_function_l361_361102

noncomputable def functionDomain : Set ℝ := {x : ℝ | x > 1/2 ∧ x ≤ 1}

theorem domain_of_function :
  ∀ x : ℝ,
  ( ∀ y, y = sqrt (log (1/2) (2 * x - 1)) → (2 * x - 1 > 0) ∧ (log (1/2) (2 * x - 1) ≥ 0) ) ↔ 
  (x > 1/2 ∧ x ≤ 1) := 
sorry -- proof to be completed

end domain_of_function_l361_361102


namespace ratio_AM_MF_l361_361414

theorem ratio_AM_MF (A B C D E F M : Point) (AE BE CF BF : ℝ) 
  (h1 : parallelogram A B C D) 
  (h2 : E ∈ segment A B)
  (h3 : F ∈ segment B C)
  (h4 : M ∈ line_through A F)
  (h5 : M ∈ line_through D E)
  (h6 : AE = 2 * BE)
  (h7 : BF = 3 * CF) : 
  AM / MF = 4 / 5 := 
sorry

end ratio_AM_MF_l361_361414


namespace cylinder_radius_in_cone_l361_361603

theorem cylinder_radius_in_cone :
  ∀ (r : ℚ), (2 * r = r) → (0 < r) → (∀ (h : ℚ), h = 2 * r → 
  (∀ (c_r : ℚ), c_r = 4 (c_r is radius of cone)  ∧ (h_c : ℚ), h_c = 10 (h_c is height of cone) ∧ 
  (10 - h) / r = h_c / c_r) → r = 20 / 9) :=
begin
  sorry,
end

end cylinder_radius_in_cone_l361_361603


namespace chessboard_cell_e2_value_l361_361882

theorem chessboard_cell_e2_value (board : ℕ × ℕ → ℕ) 
  (mean_condition : ∀ i j : ℕ, i >= 1 → i <= 8 → j >= 1 → j <= 8 → 
                     board(i, j) = 
                     (if i > 1 then board(i - 1, j) else 0  + 
                      if i < 8 then board(i + 1, j) else 0 + 
                      if j > 1 then board(i, j - 1) else 0 + 
                      if j < 8 then board(i, j + 1) else 0) / 4)
  (corner_sum_condition : board(1, 1) + board(1, 8) + board(8, 1) + board(8, 8) = 16) 
  : board(5, 2) = 4 :=
sorry

end chessboard_cell_e2_value_l361_361882


namespace right_triangle_segment_ratio_l361_361953

noncomputable def ratio_of_segments (a b c r s : ℝ) : Prop :=
r = a^2 / c ∧ s = b^2 / c ∧ a / b = 1 / 3 → r / s = 1 / 9

theorem right_triangle_segment_ratio
  (a b c r s : ℝ)
  (h_triangle : a^2 + b^2 = c^2)
  (h_ratio : a / b = 1 / 3)
  (h_segments : r = a^2 / c ∧ s = b^2 / c) :
  ratio_of_segments a b c r s :=
begin
  sorry
end

end right_triangle_segment_ratio_l361_361953


namespace acid_concentration_in_third_flask_l361_361136

theorem acid_concentration_in_third_flask 
    (w : ℚ) (W : ℚ) (hw : 10 / (10 + w) = 1 / 20) (hW : 20 / (20 + (W - w)) = 7 / 30) 
    (W_total : W = 256.43) : 
    30 / (30 + W) = 21 / 200 := 
by 
  sorry

end acid_concentration_in_third_flask_l361_361136


namespace triangle_symmetry_ratios_l361_361306

theorem triangle_symmetry_ratios {A B C D E F D' E' F' : Type*} [PlaneGeo A B C D E F D' E' F'] :
  (acute_triangle A B C) →
  (isosceles_triangle D A C ∧ isosceles_triangle E A B ∧ isosceles_triangle F B C) →
  (line_intersection D B E F D' ∧ line_intersection E C D F E' ∧ line_intersection F A D E F') →
  ∠D A C = 2 * ∠B A C ∧ ∠B E A = 2 * ∠A B C ∧ ∠C F B = 2 * ∠A C B →
  (D A = D C ∧ E A = E B ∧ F B = F C) →
  (4 = (distance D B / distance D D' + distance E C / distance E E' + distance F A / distance F F')) :=
by
  intros h_acute_t h_isosceles h_intersections h_angles h_lengths
  sorry

end triangle_symmetry_ratios_l361_361306


namespace max_leap_years_in_200_years_l361_361000

theorem max_leap_years_in_200_years (leap_year_interval: ℕ) (span: ℕ) 
  (h1: leap_year_interval = 4) 
  (h2: span = 200) : 
  (span / leap_year_interval) = 50 := 
sorry

end max_leap_years_in_200_years_l361_361000


namespace complex_number_satisfies_equation_l361_361652

noncomputable def z : ℂ := -3 + (18 / 11) * complex.I

theorem complex_number_satisfies_equation :
  5 * z - 6 * (conj z) = 3 + 18 * complex.I :=
by {
  sorry
}

end complex_number_satisfies_equation_l361_361652


namespace g_zero_l361_361047

noncomputable def f (x : ℕ) : ℕ → ℝ := sorry -- assume f is given as a polynomial
noncomputable def g (x : ℕ) : ℕ → ℝ := sorry -- assume g is given as a polynomial
noncomputable def h (x : ℕ) : ℕ → ℝ := f(x) * g(x)

def constant_term (p : ℕ → ℝ) : ℝ := p 0

axiom h_eq_fg : ∀ x, h x = f x * g x
axiom f_const_term : constant_term f = 5
axiom h_const_term : constant_term h = -10

theorem g_zero : g 0 = -2 :=
by
  sorry

end g_zero_l361_361047


namespace probability_proof_l361_361040

-- Define the problem conditions
def square_side : ℕ := 2
def points_on_sides := true  -- representing the choice of points on the sides
def distance_threshold := Real.sqrt 2

-- Define the function to calculate the probability (noncomputable)
noncomputable def probability_calculation : ℝ := (76 - 12 * Real.pi) / 16

-- Define the main theorem
theorem probability_proof :
  ∃ a b c : ℕ,
    (Prob := (a - b * Real.pi) / c:Real) = probability_calculation ∧
    Nat.gcd (Nat.gcd a b) c = 1 ∧
    a + b + c = 76 :=
by
  -- No Proof Required
  sorry

end probability_proof_l361_361040


namespace can_determine_number_of_spies_l361_361223

def determine_spies (V : Fin 15 → ℕ) (S : Fin 15 → ℕ) : Prop :=
  V 0 = S 0 + S 1 ∧ 
  ∀ i : Fin 13, V (Fin.succ (Fin.succ i)) = S i + S (Fin.succ i) + S (Fin.succ (Fin.succ i)) ∧
  V 14 = S 13 + S 14

theorem can_determine_number_of_spies :
  ∃ S : Fin 15 → ℕ, ∀ V : Fin 15 → ℕ, determine_spies V S :=
sorry

end can_determine_number_of_spies_l361_361223


namespace cube_volume_given_surface_area_l361_361183

theorem cube_volume_given_surface_area (s : ℝ) (h₀ : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_given_surface_area_l361_361183


namespace solution_exists_l361_361909

-- Defining the variables x and y
variables (x y : ℝ)

-- Defining the conditions
def condition_1 : Prop :=
  3 * x ≥ 2 * y + 16

def condition_2 : Prop :=
  x^4 + 2 * (x^2) * (y^2) + y^4 + 25 - 26 * (x^2) - 26 * (y^2) = 72 * x * y

-- Stating the theorem that (6, 1) satisfies the conditions
theorem solution_exists : condition_1 6 1 ∧ condition_2 6 1 :=
by
  -- Convert conditions into expressions
  have h1 : condition_1 6 1 := by sorry
  have h2 : condition_2 6 1 := by sorry
  -- Conjunction of both conditions is satisfied
  exact ⟨h1, h2⟩

end solution_exists_l361_361909


namespace arithmetic_sequences_sum_l361_361874

theorem arithmetic_sequences_sum
  (a b : ℕ → ℕ)
  (d1 d2 : ℕ)
  (h1 : ∀ n, a (n + 1) = a n + d1)
  (h2 : ∀ n, b (n + 1) = b n + d2)
  (h3 : a 1 + b 1 = 7)
  (h4 : a 3 + b 3 = 21) :
  a 5 + b 5 = 35 :=
sorry

end arithmetic_sequences_sum_l361_361874


namespace yunas_candies_l361_361968

theorem yunas_candies (total_candies yuna_eat candies_left : ℕ) (h1 : total_candies = 23) (h2 : candies_left = 7) (h3 : yuna_eat = total_candies - candies_left) : yuna_eat = 16 := 
by {
  rw [h1, h2] at h3,
  rw [nat.sub_self] at h3,
  exact h3,
}

end yunas_candies_l361_361968


namespace correctFinishingOrder_l361_361399

-- Define the types of athletes
inductive Athlete
| A | B | C | D | E

-- Define the type for predictions
def Prediction := Athlete → Nat

-- Define the first prediction
def firstPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 1
    | Athlete.B => 2
    | Athlete.C => 3
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the second prediction
def secondPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 4
    | Athlete.C => 1
    | Athlete.D => 5
    | Athlete.E => 2

-- Define the actual finishing positions
noncomputable def actualFinishingPositions : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 2
    | Athlete.C => 1
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the correctness conditions for the predictions
def correctPositions (actual : Prediction) (prediction : Prediction) (k : Nat) : Prop :=
  (Set.card {x : Athlete | actual x = prediction x} = k)

-- The proof statement
theorem correctFinishingOrder :
  correctPositions actualFinishingPositions firstPrediction 3 ∧
  correctPositions actualFinishingPositions secondPrediction 2 :=
sorry

end correctFinishingOrder_l361_361399


namespace sin_theta_value_sin_2theta_add_pi_over_6_value_l361_361042

variables (θ : ℝ) (π : ℝ)

noncomputable def cos_add_pi_over_6_eq_one_third (θ : ℝ) : Prop :=
  cos (θ + π / 6) = 1 / 3

theorem sin_theta_value
  (h1 : 0 < θ) (h2 : θ < π / 2)
  (h3 : cos (θ + π / 6) = 1 / 3) : 
  sin θ = (2 * sqrt 6 - 1) / 6 :=
sorry

theorem sin_2theta_add_pi_over_6_value
  (h1 : 0 < θ) (h2 : θ < π / 2)
  (h3 : cos (θ + π / 6) = 1 / 3) :
  sin (2 * θ + π / 6) = (4 * sqrt 6 + 7) / 18 :=
sorry

end sin_theta_value_sin_2theta_add_pi_over_6_value_l361_361042


namespace alcohol_remaining_l361_361292

theorem alcohol_remaining (initial_alcohol : ℝ) (poured_fraction : ℝ) (refill_fraction : ℝ) :
  initial_alcohol = 1 → poured_fraction = 1/3 → refill_fraction = 1/3 →
  let after_first_pour := initial_alcohol - poured_fraction * initial_alcohol in
  let after_refill_first := after_first_pour + refill_fraction - poured_fraction * after_first_pour in
  let after_second_pour := after_refill_first - poured_fraction * after_refill_first in
  let after_refill_second := after_second_pour + refill_fraction - poured_fraction * after_second_pour in
  let after_third_pour := after_refill_second - poured_fraction * after_refill_second in
  let after_refill_third := after_third_pour + refill_fraction - poured_fraction * after_third_pour in
  after_refill_third = 8/27 :=
by
  intros h1 h2 h3
  have h4 : after_first_pour = 1 - 1/3 * 1 := by sorry
  have h5 : after_refill_first = 2/3 + 1/3 - 1/3 * 2/3 := by sorry
  have h6 : after_second_pour = 4/9 := by sorry
  have h7 : after_refill_second = 4/9 + 1/3 - 4/9 * 1/3 := by sorry
  have h8 : after_third_pour = 8/27 := by sorry
  have h9 : after_refill_third = 8/27 + 1/3 - 8/27 * 1/3 := by sorry
  show after_refill_third = 8/27

end alcohol_remaining_l361_361292


namespace ones_digit_power_sum_l361_361154

noncomputable def ones_digit_of_power_sum_is_5 : Prop :=
  (1^2010 + 2^2010 + 3^2010 + 4^2010 + 5^2010 + 6^2010 + 7^2010 + 8^2010 + 9^2010 + 10^2010) % 10 = 5

theorem ones_digit_power_sum : ones_digit_of_power_sum_is_5 :=
  sorry

end ones_digit_power_sum_l361_361154


namespace correct_statement_C_l361_361168

theorem correct_statement_C : ∀ y : ℝ, y^2 = 25 ↔ (y = 5 ∨ y = -5) := 
by sorry

end correct_statement_C_l361_361168


namespace length_of_AG_l361_361006

theorem length_of_AG (A B C D M G : Type) [has_dist A B C D M G]
  (h1 : right_angle ∠ BAC) (h2 : dist A B = 4) (h3 : dist A C = 4 * real.sqrt 2)
  (h4 : is_altitude D A B C) (h5 : is_intersection G (A D) (B M))
  (h6 : midpoint M B C) : dist A G = 4 * real.sqrt 6 / 3 :=
begin
  sorry
end

end length_of_AG_l361_361006


namespace hyperbola_asymptote_slope_positive_l361_361251

theorem hyperbola_asymptote_slope_positive :
  (∃ x y : Real, sqrt ((x - 1)^2 + (y + 2)^2) - sqrt ((x - 7)^2 + (y + 2)^2) = 4) →
  ∃ m : Real, m = sqrt 5 / 2 ∧ m > 0 :=
by
  sorry

end hyperbola_asymptote_slope_positive_l361_361251


namespace running_speed_multiple_l361_361191

noncomputable theory

variables (v_A v_B k : ℝ)
variables (h1 : v_A = k * v_B)     -- A's speed as a multiple of B's speed
variables (h2 : 84 / v_A = 21 / v_B) -- They finish at the same time

theorem running_speed_multiple :
  k = 4 :=
begin
  -- Provide the proof here
  sorry
end

end running_speed_multiple_l361_361191


namespace inequality_sqrt_sum_l361_361080

theorem inequality_sqrt_sum (a b c : ℝ) : 
  (Real.sqrt (a^2 + b^2 - a * b) + Real.sqrt (b^2 + c^2 - b * c)) ≥ Real.sqrt (a^2 + c^2 + a * c) :=
sorry

end inequality_sqrt_sum_l361_361080


namespace inequality_proof_l361_361453

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b * c = 1) : 
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l361_361453


namespace triangle_area_correct_l361_361674

noncomputable def triangleArea : ℝ :=
  let A : (ℝ × ℝ) := (2, 3)
  let B : (ℝ × ℝ) := (10, -2)
  let C : (ℝ × ℝ) := (15, 6)
  let v : (ℝ × ℝ) := (A.1 - C.1, A.2 - C.2)
  let w : (ℝ × ℝ) := (B.1 - C.1, B.2 - C.2)
  let areaParallelogram : ℝ := (v.1 * w.2 - v.2 * w.1).abs
  (areaParallelogram / 2)

/-- The area of the triangle with vertices (2, 3), (10, -2), and (15, 6) is 44.5. -/
theorem triangle_area_correct :
  triangleArea = 44.5 :=
by
  sorry

end triangle_area_correct_l361_361674


namespace angle_ECD_130_l361_361347

theorem angle_ECD_130
  (A B C D E : Type) [linear_ordered_field ℝ]
  (h1 : AC = BC)
  (h2 : ∠DCB = 50)
  (h3 : ¬(CD ∥ AB))
  (h4 : ∠BAC = 60)
  (h5 : on_extension D C E) :
  ∠ECD = 130 :=
by
  sorry

end angle_ECD_130_l361_361347


namespace area_of_triangle_l361_361003

theorem area_of_triangle (alpha theta : ℝ) (P Q A : ℝ × ℝ)
  (hC : ∀ α, (P.1 = 2 * Real.cos α) ∧ (P.2 = Real.sin α)) -- Parametric equations of C
  (hL : ∀ rho theta, ρ * Real.sin (theta + π / 4) = (sqrt 2) / 2) -- Polar equation of line l
  (hA : A = (2 * Real.cos (π / 6), 2 * Real.sin (π / 6))) -- Polar coordinates of A
  (hP : P = (0, 1)) -- Intersection Point P
  (hQ : Q = (8/5, -3/5)) -- Intersection Point Q
  : (1 / 2) * ((Real.abs (P.1 - A.1)) * (Real.abs (P.2 - A.2))) = (4 * Real.sqrt 3) / 5 :=
by
  sorry

end area_of_triangle_l361_361003


namespace probability_at_least_one_defective_l361_361122

variable (box_contains : ℕ := 100)
variable (prob_defective : ℝ := 0.01)
variable (prob_non_defective : ℝ := 0.99)
variable (num_boxes : ℕ := 3)

theorem probability_at_least_one_defective :
  (1 : ℝ) - prob_non_defective ^ num_boxes = 1 - 0.99^3 := 
by
  change 1 - prob_non_defective ^ num_boxes = 1 - 0.99^3
  rw [prob_non_defective, num_boxes]
  exact (by simp : 1 - 0.99^3 = 1 - 0.99^3)

end probability_at_least_one_defective_l361_361122


namespace sin_add_pi_over_4_eq_l361_361304

variable (α : Real)
variables (hα1 : 0 < α ∧ α < Real.pi) (hα2 : Real.tan (α - Real.pi / 4) = 1 / 3)

theorem sin_add_pi_over_4_eq : Real.sin (Real.pi / 4 + α) = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end sin_add_pi_over_4_eq_l361_361304


namespace unique_ordering_l361_361977

-- Define houses
inductive House
| green : House
| white : House
| blue : House
| purple : House
| tan : House -- An additional house used in the problem

open House

-- Define type for house orderings
def valid_order (h : List House) :=
  (h.indexOf green < h.indexOf white) ∧
  (h.indexOf blue < h.indexOf purple) ∧
  (h.indexOf white < h.indexOf blue) ∧
  (abs (h.indexOf green - h.indexOf purple) > 1)

-- Statement of the problem
theorem unique_ordering :
  ∃! h : List House, 
    h = [white, blue, green, tan, purple] ∧
    valid_order h :=
begin
  sorry
end

end unique_ordering_l361_361977


namespace sequence_length_137_l361_361549

theorem sequence_length_137 : 
  ∃ (a : ℕ → ℕ) (k : ℕ), (strict_mono a) ∧ (∀ i < k, a i ≥ 0) ∧ 
  (∑ i in range k, 2 ^ (a i) = (2 ^ 289 + 1) / (2 ^ 17 + 1)) ∧ 
  k = 137 :=
sorry

end sequence_length_137_l361_361549


namespace domain_of_function_l361_361923

theorem domain_of_function (x : ℝ) :
  x > 2 ∧ log 2 (x - 2) ≠ 0 ↔ (2 < x ∧ x < 3) ∨ x > 3 := 
sorry

end domain_of_function_l361_361923


namespace simplification_evaluation_l361_361497

theorem simplification_evaluation (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  ( (2 * x - 6) / (x - 2) ) / ( (5 / (x - 2)) - (x + 2) ) = Real.sqrt 2 - 2 :=
sorry

end simplification_evaluation_l361_361497


namespace distance_from_origin_l361_361820

theorem distance_from_origin :
  let origin := (0, 0)
      pt := (8, -15)
  in Real.sqrt ((pt.1 - origin.1)^2 + (pt.2 - origin.2)^2) = 17 :=
by
  let origin := (0, 0)
  let pt := (8, -15)
  have h1 : pt.1 - origin.1 = 8 := by rfl
  have h2 : pt.2 - origin.2 = -15 := by rfl
  simp [h1, h2]
  -- sorry will be used to skip the actual proof steps and to ensure the code builds successfully
  sorry

end distance_from_origin_l361_361820


namespace gcd_q_r_min_value_l361_361338

theorem gcd_q_r_min_value (p q r : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) : Nat.gcd q r = 10 :=
sorry

end gcd_q_r_min_value_l361_361338


namespace solve_r_l361_361272

-- Definitions related to the problem
def satisfies_equation (r : ℝ) : Prop := ⌊r⌋ + 2 * r = 16

-- Theorem statement
theorem solve_r : ∃ (r : ℝ), satisfies_equation r ∧ r = 5.5 :=
by
  sorry

end solve_r_l361_361272


namespace integral_limit_f_n_l361_361857

noncomputable def f (x : ℝ) := 2 * x * (1 - x)

def f_n (n : ℕ) : (ℝ → ℝ) :=
  if n = 0 then id else f^(n)

open Topology Filter

theorem integral_limit_f_n :
  tendsto (λ n : ℕ, intervalIntegral (f_n n) 0 1) atTop (𝓝 (1 / 2)) :=
sorry

end integral_limit_f_n_l361_361857


namespace simplify_fraction_l361_361268

theorem simplify_fraction (num denom : ℚ) (h_num: num = (3/7 + 5/8)) (h_denom: denom = (5/12 + 2/3)) :
  (num / denom) = (177/182) := 
  sorry

end simplify_fraction_l361_361268


namespace integer_part_product_range_l361_361525

theorem integer_part_product_range :
  ∀ (a b : ℝ), (0 ≤ a ∧ a < 1) → (0 ≤ b ∧ b < 1) →
  let A := (7 : ℝ) + a in
  let B := (10 : ℝ) + b in
  (∃ c : ℤ, 70 ≤ c ∧ c ≤ 87) ∧
  (∀ c : ℤ, 70 ≤ c ∧ c ≤ 87 → ∃ a b : ℝ, (0 ≤ a ∧ a < 1) ∧ (0 ≤ b ∧ b < 1) ∧
          (⌊(7 + a) * (10 + b)⌋ = c)) ∧
  (∃! n : ℤ, 70 ≤ n ∧ n ≤ 87) :=
by {
  sorry
}

end integer_part_product_range_l361_361525


namespace athlete_positions_l361_361387

theorem athlete_positions {A B C D E : Type} 
  (h1 : A = 1 ∧ B = 2 ∧ C = 3 ∧ D = 4 ∧ E = 5)
  (h2 : C = 1 ∧ E = 2 ∧ A = 3 ∧ B = 4 ∧ D = 5)
  (h3 : (A = 1 ∧ B = 2 ∧ C = 3 ∧ ¬(D = 4 ∨ E = 5)) ∨ 
       (A = 1 ∧ ¬(B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)) ∨ 
       (B = 2 ∧ C = 3 ∧ ¬(A = 1 ∨ D = 4 ∨ E = 5)) ∨ 
       (¬(A = 1 ∨ B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)))
  (h4 : (C = 1 ∧ ¬(A = 3 ∨ B = 4 ∨ D = 5)) ∨ 
       (A = 3 ∧ ¬(C = 1 ∨ B = 4 ∨ D = 5)) ∨ 
       (B = 4 ∧ ¬(C = 1 ∨ A = 3 ∨ D = 5)))
  : (C = 1 ∧ B = 2 ∧ A = 3 ∧ D = 4 ∧ E = 5) :=
by sorry

end athlete_positions_l361_361387


namespace neg_proposition_equiv_l361_361322

theorem neg_proposition_equiv (p : Prop) : (¬ (∃ n : ℕ, 2^n > 1000)) = (∀ n : ℕ, 2^n ≤ 1000) :=
by
  sorry

end neg_proposition_equiv_l361_361322


namespace part1_solution_part2_solution_l361_361751

-- Part (1)
theorem part1_solution (b x : ℝ) : 
  (x - (b - 1)) * (x - (b + 1)) < 0 → b - 1 < x ∧ x < b + 1 :=
sorry

-- Part (2)
theorem part2_solution (b : ℝ) (x : ℝ) :
  (x ∈ set.Icc (-1 : ℝ) (2 : ℝ)) ∧ (f x = 1 → min x = 1) → 
  (f x).max ∈ { 13, 4 + 2 * real.sqrt 2 } := 
sorry

-- Definition of the function f mentioned in both parts
def f (x b : ℝ) : ℝ := x^2 - 2 * b * x + 3

end part1_solution_part2_solution_l361_361751


namespace correct_statement_about_Digital_Earth_l361_361571

def DigitalEarth : Type := sorry

def StatementA : Prop := ∀ (DE : DigitalEarth), DE.is_reflection_of_real_earth_through_digital_means
def StatementB : Prop := ∀ (DE : DigitalEarth), DE.is_extension_of_GIS_technology
def StatementC : Prop := ∀ (DE : DigitalEarth), DE.can_only_achieve_global_information_sharing_through_internet
def StatementD : Prop := ∀ (DE : DigitalEarth), DE.uses_digital_means_to_uniformly_address_earth_issues

theorem correct_statement_about_Digital_Earth 
  (DE : DigitalEarth) 
  (H_A : ¬ StatementA DE) 
  (H_B : ¬ StatementB DE) 
  (H_C : StatementC DE) 
  (H_D : ¬ StatementD DE) : 
  StatementC DE := 
H_C

end correct_statement_about_Digital_Earth_l361_361571


namespace probability_reaches_or_exceeds_6_units_at_some_point_l361_361422

def fair_coin_toss_10 : Fin 1024 → Fin 11 → ℤ := sorry

/-- The probability that Jerry reaches or exceeds 6 units in the positive direction 
at some point during the 10 tosses is 193 / 512. -/
theorem probability_reaches_or_exceeds_6_units_at_some_point :
  (∑ (i : Fin 1024), if ∃ (j : Fin 11), fair_coin_toss_10 i j ≥ 6 then 1 else 0) / 1024 = 193 / 512 :=
  sorry

end probability_reaches_or_exceeds_6_units_at_some_point_l361_361422


namespace C_answers_yes_l361_361885

-- Define types for inhabitants
inductive InhabitantType
| knight
| liar

-- Define the inhabitants A, B, C
constants (A B C : InhabitantType)

-- A statement and logical constraint
axiom A_statement : (B = C ↔ A = InhabitantType.knight)

-- Theorem stating that C will answer "Yes" to the question "Are A and B the same type?"
theorem C_answers_yes : (B = A) ↔ (C = InhabitantType.knight) := by
  sorry

end C_answers_yes_l361_361885


namespace distance_between_vertices_l361_361279

/-
Problem statement:
Prove that the distance between the vertices of the hyperbola
\(\frac{x^2}{144} - \frac{y^2}{64} = 1\) is 24.
-/

/-- 
We define the given hyperbola equation:
\frac{x^2}{144} - \frac{y^2}{64} = 1
-/
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 64 = 1

/--
We establish that the distance between the vertices of the hyperbola is 24.
-/
theorem distance_between_vertices : 
  (∀ x y : ℝ, hyperbola x y → dist (12, 0) (-12, 0) = 24) :=
by
  sorry

end distance_between_vertices_l361_361279


namespace boy_total_time_gone_l361_361586

theorem boy_total_time_gone (distance : ℝ) (speed_to : ℝ) (speed_back : ℝ) (rest_time : ℝ) :
  distance = 7.5 → speed_to = 5 → speed_back = 3 → rest_time = 2 →
  (distance / speed_to + rest_time + distance / speed_back) = 6 :=
by
  intros h_distance h_speed_to h_speed_back h_rest_time
  rw [h_distance, h_speed_to, h_speed_back, h_rest_time]
  rw [(7.5 / 5), (7.5 / 3)]
  norm_num
  exact sorry


end boy_total_time_gone_l361_361586


namespace dot_product_of_v1_and_v2_l361_361280

-- We define the two vectors
def v1 : ℝ × ℝ × ℝ := (4, -5, 2)
def v2 : ℝ × ℝ × ℝ := (-3, 3, -4)

-- We state the theorem which says that the dot product of v1 and v2 is -35
theorem dot_product_of_v1_and_v2 : (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3) = -35 :=
by
  -- Proof will go here, for now we use sorry to skip it
  sorry

end dot_product_of_v1_and_v2_l361_361280


namespace remainder_pow_2023_l361_361564

theorem remainder_pow_2023 (a b : ℕ) (h : b = 2023) : (3 ^ b) % 11 = 5 :=
by
  sorry

end remainder_pow_2023_l361_361564


namespace susan_tenth_finger_l361_361106

noncomputable def g : ℕ → ℕ :=
  λ x, match x with
    | 0 => 0
    | 1 => 9
    | 2 => 1
    | 3 => 7
    | 4 => 2
    | 5 => 5
    | 6 => 3
    | 7 => 3
    | 8 => 4
    | 9 => 1
    | _ => 0 -- Assuming g(n) is not defined for values outside the given ones

def susan_finger (n : ℕ) : ℕ :=
  Nat.recOn n 4 (λ _ r, g r)

theorem susan_tenth_finger : susan_finger 10 = 9 := 
by
  sorry

end susan_tenth_finger_l361_361106


namespace no_sequence_bn_equals_one_l361_361438

theorem no_sequence_bn_equals_one (b : ℕ → ℕ) (b1 b2 b3 : ℕ) 
  (hb1 : 1 ≤ b1 ∧ b1 ≤ 15) (hb2 : 1 ≤ b2 ∧ b2 ≤ 15) (hb3 : 1 ≤ b3 ∧ b3 ≤ 15)
  (h_seq : ∀ n ≥ 4, b n = b (n-1) * |b (n-2) - b (n-3)| + 1) 
  (h_init : b 1 = b1 ∧ b 2 = b2 ∧ b 3 = b3) :
  ¬ (∃ n, b n = 1) := by
  sorry

end no_sequence_bn_equals_one_l361_361438


namespace sin_cos_double_angle_l361_361732

-- Definitions and conditions
variable (A : ℝ)
variable h_internal : A > 0 ∧ A < π
variable h_cosA : Real.cos A = 3 / 5

-- Prove the statement
theorem sin_cos_double_angle (h_internal : A > 0 ∧ A < π) (h_cosA : Real.cos A = 3 / 5) :
  Real.sin A = 4 / 5 ∧ Real.cos (2 * A) = -7 / 25 :=
by
  sorry

end sin_cos_double_angle_l361_361732


namespace square_problem_l361_361910

noncomputable def d : ℤ := 3
noncomputable def e : ℤ := 3
noncomputable def f : ℤ := 6
noncomputable def side_length_smaller_square := (d - Real.sqrt e) / f

theorem square_problem :
  let PQRS : Square := ⟨1, 1, 1, 1⟩ in
  let PMN : Triangle := ⟨P, M, N⟩ in
  is_square PQRS →
  is_equilateral_triangle PMN →
  is_parallel PQRS PMN →
  side_length_smaller_square * f + Real.sqrt e = d →
  d + e + f = 12 :=
by
  sorry

end square_problem_l361_361910


namespace TL_square_l361_361869

-- Define T_L
noncomputable def T (L : ℕ) : ℕ := (finset.range L).sum (λ n, ⌊(n + 1)^3 / 9⌋)

-- Define the problem statement
theorem TL_square (L : ℕ) (h1 : L ≥ 1) : T L = n^2 := sorry

end TL_square_l361_361869


namespace sqrt_49_mul_sqrt_25_l361_361997

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l361_361997


namespace distance_to_point_is_17_l361_361824

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem distance_to_point_is_17 :
  distance_from_origin 8 (-15) = 17 :=
by
  sorry

end distance_to_point_is_17_l361_361824


namespace dice_probability_l361_361643

-- Definitions for the problem
def is_fair_die (die : ℕ → ℚ) : Prop :=
  ∀ n ∈ {1, 2, 3, 4, 5, 6}, die n = 1/6

def is_biased_die (die : ℕ → ℚ) : Prop :=
  die 5 = 1/2 ∧ ∀ n ∈ {1, 2, 3, 4, 6}, die n = 1/10

def roll_probability (die : ℕ → ℚ) (rolls : list ℕ) : ℚ :=
  rolls.foldl (λ acc r, acc * die r) 1

-- Lean theorem
theorem dice_probability (fair_die biased_die : ℕ → ℚ)
  (h_fair : is_fair_die fair_die)
  (h_biased : is_biased_die biased_die)
  (rolls : list ℕ) (h_rolls : rolls = [5, 5, 5])
  : let p := ((1:ℚ) / 28 * (1 / 6)) + ((27 / 28) * (1 / 2)) in
    p = 41 / 84 :=
by sorry

end dice_probability_l361_361643


namespace probability_sum_lt_8_over_5_l361_361326

theorem probability_sum_lt_8_over_5 :
  (set.probability { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ p.1 + p.2 < (8:ℝ) / 5 }) = 23 / 25 :=
sorry

end probability_sum_lt_8_over_5_l361_361326


namespace ratio_is_two_l361_361556

-- Define the conditions
def weight_curl : ℝ := 90
def weight_squat : ℝ := 900
def squat_ratio : ℝ := 5

-- Define the ratio of military press weight to curl weight
def military_press_weight := weight_squat / squat_ratio
def ratio_military_press_to_curl := military_press_weight / weight_curl

-- Statement to prove
theorem ratio_is_two : ratio_military_press_to_curl = 2 := by
  -- Proof goes here
  sorry

end ratio_is_two_l361_361556


namespace smallest_a_plus_b_l361_361720

theorem smallest_a_plus_b (a b : ℕ) (p : ℕ) (c : ℤ)
  (ha : a > 0) (hb : b > 0) (h_gcd : Nat.gcd a b = 1) :
  ((∃ p, p.Prime ∧ ∃ c, ∀ x y : ℤ, (x ^ a + y ^ b) % p ≠ c % p) → a + b ≥ 7) :=
sorry

end smallest_a_plus_b_l361_361720


namespace circle_center_radius_l361_361560

open Real

theorem circle_center_radius :
  ∃ (a b r : ℝ), (-2 = a ∧ 1 = b ∧ sqrt 2 = r ∧ 
  ∀ (x y : ℝ), x^2 + y^2 + 4 * x - 2 * y + 3 = 0 → (x + a)^2 + (y - b)^2 = r^2) :=
begin
  sorry
end

end circle_center_radius_l361_361560


namespace constant_term_expansion_l361_361518

theorem constant_term_expansion : 
  (let expansion := (x^2 + 3) * (1 / x^2 - 1) ^ 5 in 
   ∃ (c : ℤ), expansion.eval x = c ∧ c = 2) :=
begin
  sorry
end

end constant_term_expansion_l361_361518


namespace hypotenuse_length_l361_361607

theorem hypotenuse_length (x y : ℝ) 
  (h1 : (1/3) * Real.pi * y^2 * x = 1080 * Real.pi) 
  (h2 : (1/3) * Real.pi * x^2 * y = 2430 * Real.pi) : 
  Real.sqrt (x^2 + y^2) = 6 * Real.sqrt 13 := 
  sorry

end hypotenuse_length_l361_361607


namespace num_solutions_l361_361516

noncomputable def z_count (z : ℂ) : Prop :=
  (z + (1 / z)).im = 0 ∧ abs(z - 2) = real.sqrt 2

theorem num_solutions : 
  (∃ z1 z2 z3 z4 : ℂ, z_count z1 ∧ z_count z2 ∧ z_count z3 ∧ z_count z4 ∧ 
   (z1 ≠ z2 ∧ z1 ≠ z3 ∧ z1 ≠ z4 ∧ z2 ≠ z3 ∧ z2 ≠ z4 ∧ z3 ≠ z4)) ∧
  (∀ z : ℂ, z_count z → z = z1 ∨ z = z2 ∨ z = z3 ∨ z = z4) :=
sorry

end num_solutions_l361_361516


namespace multiplication_problem_division_problem_l361_361640

theorem multiplication_problem :
  125 * 76 * 4 * 8 * 25 = 7600000 :=
sorry

theorem division_problem :
  (6742 + 6743 + 6738 + 6739 + 6741 + 6743) / 6 = 6741 :=
sorry

end multiplication_problem_division_problem_l361_361640


namespace train_crossing_time_l361_361767

namespace TrainCrossingProblem

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 300
def speed_of_train_kmph : ℕ := 36
def speed_of_train_mps : ℕ := 10 -- conversion from 36 kmph to m/s
def total_distance : ℕ := length_of_train + length_of_bridge -- 250 + 300
def expected_time : ℕ := 55

theorem train_crossing_time : 
  (total_distance / speed_of_train_mps) = expected_time :=
by
  sorry
end TrainCrossingProblem

end train_crossing_time_l361_361767


namespace evaluate_expression_l361_361498

theorem evaluate_expression (a b : ℤ) (h_a : a = 1) (h_b : b = -2) : 
  2 * (a^2 - 3 * a * b + 1) - (2 * a^2 - b^2) + 5 * a * b = 8 :=
by
  sorry

end evaluate_expression_l361_361498


namespace study_tour_buses_l361_361192

variable (x : ℕ) (num_people : ℕ)

def seats_A := 45
def seats_B := 60
def extra_people := 30
def fewer_B := 6

theorem study_tour_buses (h : seats_A * x + extra_people = seats_B * (x - fewer_B)) : 
  x = 26 ∧ (seats_A * 26 + extra_people = 1200) := 
  sorry

end study_tour_buses_l361_361192


namespace graph_of_composed_function_l361_361915

variables (g : ℕ → ℕ)

theorem graph_of_composed_function : 
  (g 2 = 4) → (g 3 = 2) → (g 4 = 6) → 
  (((2, g (g 2)) = (2, 6)) ∧ ((3, g (g 3)) = (3, 4))) ∧ 
  ((2 * 6 + 3 * 4) = 24) :=
by
  intros h1 h2 h3
  have h4 : g (g 2) = g 4 := by rw [h1]
  have h5 : g 4 = 6 := h3
  have h6 : g (g 2) = 6 := by rw [h4, h5]
  have h7 : g (g 3) = g 2 := by rw [h2]
  have h8 : g 2 = 4 := h1
  have h9 : g (g 3) = 4 := by rw [h7, h8]
  split
  split
  exact ⟨2, 6⟩
  exact ⟨3, 4⟩
  exact by norm_num  -- 24
  sorry

end graph_of_composed_function_l361_361915


namespace simplify_fraction_l361_361089

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 :=
by sorry

end simplify_fraction_l361_361089


namespace prove_positions_l361_361377

def athlete : Type := { A : ℕ, B : ℕ, C : ℕ, D : ℕ, E : ℕ }

def first_prediction (x : athlete) : athlete :=
  { A := 1, B := 2, C := 3, D := 4, E := 5 }

def second_prediction (x : athlete) : athlete :=
  { A := 3, B := 4, C := 1, D := 5, E := 2 }

def actual_positions : athlete :=
  { A := 3, B := 2, C := 1, D := 4, E := 5 }

theorem prove_positions :
  (λ x, first_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 3 ∧
  (λ x, second_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 2 ∧
  actual_positions.A = 3 ∧ actual_positions.B = 2 ∧ actual_positions.C = 1 ∧ actual_positions.D = 4 ∧ actual_positions.E = 5 := 
  sorry

end prove_positions_l361_361377


namespace actual_positions_correct_l361_361364

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end actual_positions_correct_l361_361364


namespace range_of_x_l361_361948

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = (2 / (Real.sqrt (x - 1)))) → (x > 1) :=
by
  sorry

end range_of_x_l361_361948


namespace find_sum_a_b_l361_361173

-- Define the problem's conditions as Lean definitions
def is_positive_integer (n : ℕ) : Prop := n > 0

def rounded_to_three_decimal_places (x y : ℝ) : Prop := 
  (Real.round (x * 1000)) = (Real.round (y * 1000))

-- Define the main theorem, translating the question and conditions into the Lean statement
theorem find_sum_a_b : 
  ∃ (a b : ℕ), is_positive_integer a ∧ is_positive_integer b ∧ 
  rounded_to_three_decimal_places ((a / 5) + (b / 7)) 1.51 ∧ a + b = 9 :=
sorry   -- No proof is required, just the statement

end find_sum_a_b_l361_361173


namespace final_weight_is_200_l361_361430

def initial_weight : ℕ := 220
def percentage_lost : ℕ := 10
def weight_gained : ℕ := 2

theorem final_weight_is_200 :
  initial_weight - (initial_weight * percentage_lost / 100) + weight_gained = 200 := by
  sorry

end final_weight_is_200_l361_361430


namespace distance_from_B_to_AC_l361_361455

theorem distance_from_B_to_AC (A B C : Type) [has_dist A B C]
  (h1 : dist A B = 3)
  (h2 : dist B C = 4)
  (h3 : dist C A = 5) :
  distance_to_line B (line_through A C) = 12 / 5 :=
sorry

end distance_from_B_to_AC_l361_361455


namespace pr_eq_qs_l361_361580

-- Definitions for quadrilateral and feet of perpendiculars
variable (A B C D P Q R S : Type) 
variable [Quad A B C D] [PerpendicularFoot D A B P] [PerpendicularFoot D B C Q]
variable [PerpendicularFoot B A D R] [PerpendicularFoot B D C S]

-- Statement to prove PR = QS given the angle equality condition 
theorem pr_eq_qs (h: angle PSR = angle SPQ) : length PR = length QS := 
  sorry

end pr_eq_qs_l361_361580


namespace shelly_total_money_l361_361494

theorem shelly_total_money :
  let ten_bills := 30 in
  let five_bills := ten_bills - 12 in
  let total_money := (ten_bills * 10) + (five_bills * 5) in
  total_money = 390 :=
by
  -- Whoops, the proof is not requested here, so we use sorry for the incomplete part.
  sorry

end shelly_total_money_l361_361494


namespace no_real_root_for_3_in_g_l361_361696

noncomputable def g (x c : ℝ) : ℝ := x^2 + 3 * x + c

theorem no_real_root_for_3_in_g (c : ℝ) :
  (21 - 4 * c) < 0 ↔ c > 21 / 4 := by
sorry

end no_real_root_for_3_in_g_l361_361696


namespace remaining_bollards_l361_361619

theorem remaining_bollards :
  let total_bollards := 4000 * 2 in
  let pi_fraction_installed := (total_bollards * Real.pi / 4).to_nat in
  (total_bollards - pi_fraction_installed) = 1717 :=
by
  let total_bollards := 4000 * 2
  let pi_fraction_installed := (total_bollards * Real.pi / 4).to_nat
  have h1 : total_bollards = 8000 := by norm_num
  have h2 : (8000 * Real.pi / 4).to_nat = 6283 := by sorry -- This step involves numerical approximation
  rw [h1, h2]
  norm_num -- Then compute the final step which should match 8000 - 6283 = 1717
  sorry

end remaining_bollards_l361_361619


namespace breadth_of_cuboid_l361_361675

theorem breadth_of_cuboid
  (surface_area : ℝ)
  (length : ℝ)
  (height : ℝ)
  (b : ℝ) :
  surface_area = 720 ∧ length = 12 ∧ height = 10 → b ≈ 10.91 :=
by
  sorry

end breadth_of_cuboid_l361_361675


namespace find_function_l361_361744

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x - 1) = 2 * x^2 - x) : ∀ x : ℝ, f(x) = 2 * x^2 + 3 * x + 1 :=
by
  sorry

end find_function_l361_361744


namespace elder_sister_money_l361_361879

-- Define variables
variables {x y : ℝ}

-- Define conditions based on the problem statement
def total_savings (x y : ℝ) : Prop := x + y = 108
def elder_sis_donation (x : ℝ) : ℝ := x - 0.75 * x
def younger_sis_donation (y : ℝ) : ℝ := y - 0.8 * y
def remaining_equal (x y : ℝ) : Prop := elder_sis_donation x = younger_sis_donation y

-- The theorem we want to prove
theorem elder_sister_money (x y : ℝ) 
  (h1 : total_savings x y)
  (h2 : remaining_equal x y) :
  x = 48 :=
begin
  sorry
end

end elder_sister_money_l361_361879


namespace sum_of_possible_values_f2_l361_361288

noncomputable def f (x : ℝ) : ℝ := sorry

theorem sum_of_possible_values_f2 (f : ℝ → ℝ) :
  (∀ x y : ℝ, 0 < x → 0 < y → f(x) * f(y) - f(x * y) = y / x + x / y) →
  (∃ f2_1 f2_2 : ℝ, (f2_1 = 5/2 ∨ f2_2 = -5/4) ∧ f2_1 + f2_2 = 5/4) :=
sorry

end sum_of_possible_values_f2_l361_361288


namespace loss_percentage_initial_selling_l361_361615

theorem loss_percentage_initial_selling (CP SP' : ℝ) 
  (hCP : CP = 1250) 
  (hSP' : SP' = CP * 1.15) 
  (h_diff : SP' - 500 = 937.5) : 
  (CP - 937.5) / CP * 100 = 25 := 
by 
  sorry

end loss_percentage_initial_selling_l361_361615


namespace students_play_neither_l361_361351

theorem students_play_neither (total_students football_players tennis_players both_players : ℕ)
    (h_total : total_students = 35)
    (h_football : football_players = 26)
    (h_tennis : tennis_players = 20)
    (h_both : both_players = 17) :
      total_students - (football_players + tennis_players - both_players) = 6 :=
by
  rw [h_total, h_football, h_tennis, h_both]
  calc
    35 - (26 + 20 - 17) = 35 - 29 : by norm_num
    ...                 = 6 : by norm_num

end students_play_neither_l361_361351


namespace ratio_buses_to_cars_l361_361949

theorem ratio_buses_to_cars (B C : ℕ) (h1 : B = C - 60) (h2 : C = 65) : B / C = 1 / 13 :=
by 
  sorry

end ratio_buses_to_cars_l361_361949


namespace final_weight_is_200_l361_361429

def initial_weight : ℕ := 220
def percentage_lost : ℕ := 10
def weight_gained : ℕ := 2

theorem final_weight_is_200 :
  initial_weight - (initial_weight * percentage_lost / 100) + weight_gained = 200 := by
  sorry

end final_weight_is_200_l361_361429


namespace two_students_follow_all_celebrities_l361_361965

theorem two_students_follow_all_celebrities :
  ∀ (students : Finset ℕ) (celebrities_followers : ℕ → Finset ℕ),
    (students.card = 120) →
    (∀ c : ℕ, c < 10 → (celebrities_followers c).card ≥ 85 ∧ (celebrities_followers c) ⊆ students) →
    ∃ (s1 s2 : ℕ), s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧
      (∀ c : ℕ, c < 10 → (s1 ∈ celebrities_followers c ∨ s2 ∈ celebrities_followers c)) :=
by
  intros students celebrities_followers h_students_card h_followers_cond
  sorry

end two_students_follow_all_celebrities_l361_361965


namespace trigonometric_identity_correct_l361_361622

theorem trigonometric_identity_correct (α : ℝ) :
  ∃ (α : ℝ), sin α = 0 ∧ cos α = -1 ∧ sin α ^ 2 + cos α ^ 2 = 1 := 
by {
  use α,
  split,
  assumption,
  split,
  assumption,
  assumption,
  sorry
}

end trigonometric_identity_correct_l361_361622


namespace mickys_sticks_more_l361_361904

theorem mickys_sticks_more 
  (simons_sticks : ℕ := 36)
  (gerrys_sticks : ℕ := (2 * simons_sticks) / 3)
  (total_sticks_needed : ℕ := 129)
  (total_simons_and_gerrys_sticks : ℕ := simons_sticks + gerrys_sticks)
  (mickys_sticks : ℕ := total_sticks_needed - total_simons_and_gerrys_sticks) :
  mickys_sticks - total_simons_and_gerrys_sticks = 9 :=
by
  sorry

end mickys_sticks_more_l361_361904


namespace find_youngest_age_l361_361199

noncomputable def youngest_child_age 
  (meal_cost_mother : ℝ) 
  (meal_cost_per_year : ℝ) 
  (total_bill : ℝ) 
  (triplets_count : ℕ) := 
  {y : ℝ // 
    (∃ t : ℝ, 
      meal_cost_mother + meal_cost_per_year * (triplets_count * t + y) = total_bill ∧ y = 2 ∨ y = 5)}

theorem find_youngest_age : 
  youngest_child_age 3.75 0.50 12.25 3 := 
sorry

end find_youngest_age_l361_361199


namespace angles_equal_l361_361070

-- Defining structures for points and lines
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  start : Point
  end : Point

-- Definitions for conditions in the problem
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def A1 : Point := sorry
def X : Point := sorry
def B1 : Point := sorry
def C1 : Point := sorry 
def P : Point := sorry
def Q : Point := sorry

-- Conditions for the triangle and its angle bisector
def AngleBisector (A B C A1 : Point) : Prop := sorry
def PointOnLine (X A1 : Point) : Prop := sorry
def LineIntersection (BX AC B1 : Point) : Prop := sorry
def LineIntersection (CX AB C1 : Point) : Prop := sorry
def SegmentIntersection (A1 B1 CC1 P : Point) : Prop := sorry
def SegmentIntersection (A1 C1 BB1 Q : Point) : Prop := sorry

-- Given conditions into Lean definitions
def conditions : Prop :=
  AngleBisector A B C A1 ∧
  PointOnLine X A1 ∧
  LineIntersection B X A C B1 ∧
  LineIntersection C X A B C1 ∧
  SegmentIntersection A1 B1 C C1 P ∧
  SegmentIntersection A1 C1 B B1 Q

-- The final goal we need to prove
theorem angles_equal (A B C A1 X B1 C1 P Q : Point) (h : conditions) : 
  sorry -- this would be the place for the proof
  ∠ P A C = ∠ Q A B

end angles_equal_l361_361070


namespace correct_operation_l361_361160

theorem correct_operation :
  ((∀ a : ℝ, (a^2)^3 ≠ a^5) ∧
   (\left(1/2\right) ^(-1:ℚ) ≠ -2) ∧
   ((2-Real.sqrt(5))^0 = 1) ∧
   (∀ a : ℝ, a^3 * a^3 ≠ 2*a^6)) →
   ((2 - Real.sqrt(5))^0 = 1) :=
by intros h; exact h.right.left.right

end correct_operation_l361_361160


namespace solution_set_eq_two_l361_361316

theorem solution_set_eq_two (m : ℝ) (h : ∀ x : ℝ, mx + 2 > 0 ↔ x < 2) :
  m = -1 :=
sorry

end solution_set_eq_two_l361_361316


namespace percentage_increase_l361_361602

theorem percentage_increase
  (W R : ℝ)
  (H1 : 0.70 * R = 1.04999999999999982 * W) :
  (R - W) / W * 100 = 50 :=
by
  sorry

end percentage_increase_l361_361602


namespace monotonic_interval_range_within_interval_l361_361704

noncomputable def m (x : ℝ) : ℝ × ℝ := (sqrt 3 * Real.cos x, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Question 1: Monotonically increasing interval
theorem monotonic_interval : 
  ∀ k : ℤ, ∀ x : ℝ , -Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi → 
  ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y < x + ε → f y > f x :=
sorry

-- Question 2: Range given domain
theorem range_within_interval :
  ∀ x : ℝ, -Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 2 → 0 ≤ f x ∧ f x ≤ 3 / 2 :=
sorry

end monotonic_interval_range_within_interval_l361_361704


namespace trains_clear_time_l361_361975

theorem trains_clear_time
  (length_train1 : ℕ) (length_train2 : ℕ)
  (speed_train1_kmph : ℕ) (speed_train2_kmph : ℕ)
  (conversion_factor : ℕ) -- 5/18 as a rational number (for clarity)
  (approx_rel_speed : ℚ) -- Approximate relative speed 
  (total_distance : ℕ) 
  (total_time : ℚ) :
  length_train1 = 160 →
  length_train2 = 280 →
  speed_train1_kmph = 42 →
  speed_train2_kmph = 30 →
  conversion_factor = 5 / 18 →
  approx_rel_speed = (42 * (5 / 18) + 30 * (5 / 18)) →
  total_distance = length_train1 + length_train2 →
  total_time = total_distance / approx_rel_speed →
  total_time = 22 := 
by
  sorry

end trains_clear_time_l361_361975


namespace sum_of_positive_k_l361_361521

theorem sum_of_positive_k : ∑ k in {23, 10, 5, 2}, k = 40 :=
by
  simp only [Finset.mem_insert, Finset.mem_singleton, finset_sum_add_distrib, Finset.sum_singleton]
  norm_num

end sum_of_positive_k_l361_361521


namespace art_club_artworks_l361_361961

-- Define the conditions
def students := 25
def artworks_per_student_per_quarter := 3
def quarters_per_year := 4
def years := 3

-- Calculate total artworks
theorem art_club_artworks : 
  students * artworks_per_student_per_quarter * quarters_per_year * years = 900 :=
by
  sorry

end art_club_artworks_l361_361961


namespace solve_log_equation_l361_361907

theorem solve_log_equation :
  ∀ x : ℝ, 2 * | Real.log x / Real.log (1/2) - 1 | - | Real.log (x^2) / Real.log 4 + 2 | = -1 / 2 * (Real.log x / Real.log (sqrt 2)) → x = 1 :=
by
  intro x h
  sorry

end solve_log_equation_l361_361907


namespace area_ratio_of_extended_equilateral_triangle_l361_361038

-- Define an equilateral triangle where the sides are extended as specified
structure EquilateralTriangle (A B C A' B' C' : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder A'] [LinearOrder B'] [LinearOrder C'] :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)
  (extend_BB' : BB' = 2 * AB)
  (extend_CC' : CC' = 2 * BC)
  (extend_AA' : AA' = 2 * CA)

-- The main theorem statement
theorem area_ratio_of_extended_equilateral_triangle
  (A B C A' B' C' : Type)
  [LinearOrder A] [LinearOrder B] [LinearOrder C]
  [LinearOrder A'] [LinearOrder B'] [LinearOrder C']
  (triangle : EquilateralTriangle A B C A' B' C') :
  let s := triangle.AB in
  let s' := 3 * s in
  (s' * s') / (s * s) = 9 :=
by
  simp
  sorry

end area_ratio_of_extended_equilateral_triangle_l361_361038


namespace correct_statement_C_l361_361167

theorem correct_statement_C : ∀ y : ℝ, y^2 = 25 ↔ (y = 5 ∨ y = -5) := 
by sorry

end correct_statement_C_l361_361167


namespace concentration_flask3_l361_361132

open Real

noncomputable def proof_problem : Prop :=
  ∃ (W : ℝ) (w : ℝ), 
    let flask1_acid := 10 in
    let flask2_acid := 20 in
    let flask3_acid := 30 in
    let flask1_total := flask1_acid + w in
    let flask2_total := flask2_acid + (W - w) in
    let flask3_total := flask3_acid + W in
    (flask1_acid / flask1_total = 1 / 20) ∧
    (flask2_acid / flask2_total = 7 / 30) ∧
    (flask3_acid / flask3_total = 21 / 200)

theorem concentration_flask3 : proof_problem :=
begin
  sorry
end

end concentration_flask3_l361_361132


namespace distance_from_origin_to_point_l361_361834

open Real

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 →
  y = -15 →
  sqrt (x^2 + y^2) = 17 :=
by
  intros hx hy
  rw [hx, hy]
  norm_num
  rw ←sqrt_sq (show 0 ≤ 289 by norm_num)
  norm_num

end distance_from_origin_to_point_l361_361834


namespace foot_of_altitude_on_Euler_line_l361_361894

theorem foot_of_altitude_on_Euler_line
(ABCD : Type*)
(ABC : Type*) 
(equilateral_tetrahedron : (∀ A B C D : ABCD, dist A B = dist C D))
:
∀ (D : ABCD) (foot_of_altitude : ABC), 
  (foot_of_altitude ∈ euler_line ABC) := 
sorry

end foot_of_altitude_on_Euler_line_l361_361894


namespace yield_percentage_of_stock_l361_361584

noncomputable def annual_dividend (par_value : ℝ) : ℝ := 0.21 * par_value
noncomputable def market_price : ℝ := 210
noncomputable def yield_percentage (annual_dividend : ℝ) (market_price : ℝ) : ℝ :=
  (annual_dividend / market_price) * 100

theorem yield_percentage_of_stock (par_value : ℝ)
  (h_par_value : par_value = 100) :
  yield_percentage (annual_dividend par_value) market_price = 10 :=
by
  sorry

end yield_percentage_of_stock_l361_361584


namespace find_points_D_E_l361_361273

theorem find_points_D_E (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
(h_triangle : triangle A B C) :
  ∃ D E : A,
  ∃ hD : D ∈ segment A B,
  ∃ hE : E ∈ segment A C,
  dist A D = dist D E ∧ dist D E = dist E C := 
sorry

end find_points_D_E_l361_361273


namespace range_of_a_l361_361749

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem range_of_a (a b : ℝ) (x : ℝ) (h_b : b ≤ 0) (h_x : x ∈ Set.Ioc Real.e (Real.exp 2)) (h_f : f a b x ≥ x) : 
  a ≥ Real.exp 2 / 2 :=
by
  sorry

end range_of_a_l361_361749


namespace fraction_of_odd_products_is_025_l361_361784

theorem fraction_of_odd_products_is_025 :
  let odd_numbers := {n | n % 2 = 1 ∧ 0 ≤ n ∧ n ≤ 15}
      num_odd := 8
      total_products := 16 * 16
      num_odd_products := num_odd * num_odd
      fraction_odd_products := (num_odd_products : ℝ) / total_products
  in
    Float.round (fraction_odd_products * 100) / 100 = 0.25 :=
by
  sorry

end fraction_of_odd_products_is_025_l361_361784


namespace simplify_fraction_l361_361088

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 :=
by sorry

end simplify_fraction_l361_361088


namespace trigonometric_identity_tangent_l361_361499

theorem trigonometric_identity_tangent :
  (tan 20 * Real.pi / 180 + tan 30 * Real.pi / 180 + tan 60 * Real.pi / 180 + tan 70 * Real.pi / 180) / (cos 40 * Real.pi / 180) 
  = (Real.sqrt 3 + 1) / (Real.sqrt 3 * cos 40 * Real.pi / 180) :=
by
  sorry

end trigonometric_identity_tangent_l361_361499


namespace PQ_is_tangent_to_circle_diameter_AB_l361_361052

noncomputable def problem_statement : Prop :=
  ∀ (O₁ O₂ : Type) [circle O₁] [circle O₂] (A B C D P Q : Point) (line_AC : Line) (line_AD : Line),
    (A = circle_intersection O₁ O₂) ∧
    (B = circle_intersection O₁ O₂) ∧
    (C = line_circle_intersection O₁ line_AC) ∧
    (D = line_circle_intersection O₂ line_AD) ∧
    (P = tangent_intersection_point C O₁) ∧
    (Q = tangent_intersection_point D O₂) ∧
    (perpendicular_from B P Q) →
    tangent_to_circle_with_diameter_AB P Q A B

theorem PQ_is_tangent_to_circle_diameter_AB : problem_statement :=
by
  sorry

end PQ_is_tangent_to_circle_diameter_AB_l361_361052


namespace quadratic_has_two_distinct_roots_l361_361710

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_has_two_distinct_roots 
  (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  discriminant a b c > 0 :=
sorry

end quadratic_has_two_distinct_roots_l361_361710


namespace coin_difference_l361_361897

theorem coin_difference (p n d : ℕ) (h1 : p + n + d = 3030) 
  (h2 : p ≥ 1) (h3 : n ≥ 1) (h4 : d ≥ 1) : 
  30286 - (30300 - 9 * 1514 - 5 * 1514) = 21182 :=
by {
  simp,
  exact h1,
  exact h2,
  exact h3,
  exact h4,
  sorry 
}

end coin_difference_l361_361897


namespace jane_paints_correct_area_l361_361023

def height_of_wall : ℕ := 10
def length_of_wall : ℕ := 15
def width_of_door : ℕ := 3
def height_of_door : ℕ := 5

def area_of_wall := height_of_wall * length_of_wall
def area_of_door := width_of_door * height_of_door
def area_to_be_painted := area_of_wall - area_of_door

theorem jane_paints_correct_area : area_to_be_painted = 135 := by
  sorry

end jane_paints_correct_area_l361_361023


namespace relationship_between_exponents_l361_361702

theorem relationship_between_exponents {x y z : ℝ} (h1 : 2^x = 3^y) (h2 : 3^y = 5^z) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
(3 * y < 2 * x ∧ 2 * x < 5 * z) :=
sorry

end relationship_between_exponents_l361_361702


namespace grocer_sales_l361_361597

theorem grocer_sales (sale1 sale2 sale3 sale4 sale5 sale6 : ℕ)
  (h1 : sale2 = 900)
  (h2 : sale3 = 1000)
  (h3 : sale4 = 700)
  (h4 : sale5 = 800)
  (h5 : sale6 = 900)
  (h6 : (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = 850) :
  sale1 = 800 :=
by
  sorry

end grocer_sales_l361_361597


namespace three_pow_2023_mod_eleven_l361_361565

theorem three_pow_2023_mod_eleven :
  (3 ^ 2023) % 11 = 5 :=
sorry

end three_pow_2023_mod_eleven_l361_361565


namespace a_plus_b_is_18_over_5_l361_361938

noncomputable def a_b_sum (a b : ℚ) : Prop :=
  (∃ (x y : ℚ), x = 2 ∧ y = 3 ∧ x = (1 / 3) * y + a ∧ y = (1 / 5) * x + b) → a + b = (18 / 5)

-- No proof provided, just the statement.
theorem a_plus_b_is_18_over_5 (a b : ℚ) : a_b_sum a b :=
sorry

end a_plus_b_is_18_over_5_l361_361938


namespace count_common_numbers_up_to_2017_l361_361124

-- Define sequence 1 as a set of numbers
def sequence1 (n : ℕ) : ℕ := 2 * n - 1

-- Define sequence 2 as a set of numbers
def sequence2 (n : ℕ) : ℕ := 3 * n - 2

-- Define a predicate to check if a number is in both sequence 1 and sequence 2
def in_both_sequences (x : ℕ) : Prop :=
  ∃ n1 n2 : ℕ, sequence1 n1 = x ∧ sequence2 n2 = x

-- Define a set of numbers common to both sequences up to 2017
def common_numbers_up_to_2017 : set ℕ :=
  {x | x ≤ 2017 ∧ in_both_sequences x}

-- State the main proof problem
theorem count_common_numbers_up_to_2017 : 
  (common_numbers_up_to_2017).to_finset.card = 337 := 
sorry

end count_common_numbers_up_to_2017_l361_361124


namespace athlete_positions_l361_361384

theorem athlete_positions {A B C D E : Type} 
  (h1 : A = 1 ∧ B = 2 ∧ C = 3 ∧ D = 4 ∧ E = 5)
  (h2 : C = 1 ∧ E = 2 ∧ A = 3 ∧ B = 4 ∧ D = 5)
  (h3 : (A = 1 ∧ B = 2 ∧ C = 3 ∧ ¬(D = 4 ∨ E = 5)) ∨ 
       (A = 1 ∧ ¬(B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)) ∨ 
       (B = 2 ∧ C = 3 ∧ ¬(A = 1 ∨ D = 4 ∨ E = 5)) ∨ 
       (¬(A = 1 ∨ B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)))
  (h4 : (C = 1 ∧ ¬(A = 3 ∨ B = 4 ∨ D = 5)) ∨ 
       (A = 3 ∧ ¬(C = 1 ∨ B = 4 ∨ D = 5)) ∨ 
       (B = 4 ∧ ¬(C = 1 ∨ A = 3 ∨ D = 5)))
  : (C = 1 ∧ B = 2 ∧ A = 3 ∧ D = 4 ∧ E = 5) :=
by sorry

end athlete_positions_l361_361384


namespace distance_from_origin_to_point_l361_361805

-- Define the specific points in the coordinate system
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (8, -15)

-- The distance formula in Euclidean space
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem statement
theorem distance_from_origin_to_point : distance origin point = 17 := 
by
  sorry

end distance_from_origin_to_point_l361_361805


namespace average_next_seven_l361_361902

variable (c : ℕ) (h : c > 0)

theorem average_next_seven (d : ℕ) (h1 : d = (2 * c + 3)) 
  : (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = 2 * c + 6 := by
  sorry

end average_next_seven_l361_361902


namespace greatest_carioca_points_l361_361608

structure Point (α : Type*) := (x : α) (y : α)

def Carioca (n : ℕ) (points : Fin n → Point ℝ) : Prop :=
  ∃ (a : Fin n → Fin n), (∀ i : Fin n, i < n - 1 → dist (points (a i)) (points (a (i + 1))) = dist (points (a 0)) (points (a 1)))
  ∧ dist (points (a 0)) (points (a (n - 1))) = dist (points (a 0)) (points (a 1))

theorem greatest_carioca_points : 
  ∃ k : ℕ, k = 1012 ∧ (∀ (points : Fin 1012 → Point ℝ), ∃ additional_points : Fin (2023 - 1012) → Point ℝ, Carioca 2023 (points ∘ Fin.cast_le (Nat.le_of_eq (rfl)) ∘ Sum.inl ⊕ additional_points)) :=
sorry

end greatest_carioca_points_l361_361608


namespace sum_first_10_terms_b_of_a_seq_l361_361327

open Nat

-- Sequence definitions and properties
def a_seq : ℕ → ℕ
| 1     := 1
| (n+1) := a_seq n + 2

def b_seq : ℕ → ℕ
| 1     := 1
| (n+1) := b_seq n * 2

def b_of_a_seq (n : ℕ) : ℕ :=
  b_seq (a_seq n)

def c_seq (n : ℕ) : ℕ :=
  b_of_a_seq n

-- Geometric sum calculation
noncomputable def geo_sum (r : ℕ) (n : ℕ) : ℕ :=
  (r ^ n - 1) / (r - 1)

-- Theorem statement
theorem sum_first_10_terms_b_of_a_seq :
  (Finset.range 10).sum (λ n, c_seq (n + 1)) = (4^10 - 1) / 3 :=
by
  sorry

end sum_first_10_terms_b_of_a_seq_l361_361327


namespace train_car_count_l361_361661

def cars_in_train (cars_in_15_seconds : ℕ) (conversion_factor : ℕ) (total_time_seconds : ℕ) : ℕ :=
  (cars_in_15_seconds * total_time_seconds) / conversion_factor

theorem train_car_count :
  let cars_in_15_seconds := 8
  let conversion_factor := 15
  let total_time_seconds := (3 * 60) + 30
  cars_in_train cars_in_15_seconds conversion_factor total_time_seconds = 112 := by
  let cars_in_15_seconds := 8
  let conversion_factor := 15
  let total_time_seconds := (3 * 60) + 30
  show cars_in_train cars_in_15_seconds conversion_factor total_time_seconds = 112 from
    sorry

end train_car_count_l361_361661


namespace smallest_x_l361_361656

theorem smallest_x (M : ℤ) (x : ℕ) (hx_pos : 0 < x)
  (h_factorization : 1800 = 2^3 * 3^2 * 5^2)
  (h_eq : 1800 * x = M^3) : x = 15 :=
sorry

end smallest_x_l361_361656


namespace simplify_and_evaluate_l361_361092

theorem simplify_and_evaluate (x : ℝ) (hx : x = Real.sqrt 2 + 1) :
  (x + 1) / x / (x - (1 + x^2) / (2 * x)) = Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_l361_361092


namespace expression_evaluate_l361_361561

theorem expression_evaluate :
  50 * (50 - 5) - (50 * 50 - 5) = -245 :=
by
  sorry

end expression_evaluate_l361_361561


namespace distance_from_origin_to_point_l361_361833

open Real

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 →
  y = -15 →
  sqrt (x^2 + y^2) = 17 :=
by
  intros hx hy
  rw [hx, hy]
  norm_num
  rw ←sqrt_sq (show 0 ≤ 289 by norm_num)
  norm_num

end distance_from_origin_to_point_l361_361833


namespace sum_div_seven_l361_361245

theorem sum_div_seven : (∑ i in Finset.range 15.succ, i.succ) / 7 = 17 + 1 / 7 :=
by sorry

end sum_div_seven_l361_361245


namespace area_KBLN_eq_area_DEN_l361_361353

theorem area_KBLN_eq_area_DEN
  (hex : regular_hexagon ABCDEF)
  (K : midpoint A B)
  (L : midpoint B C)
  (N : point_of_intersection (line.mk D K) (line.mk E L))
  : area (quadrilateral.mk K B L N) = area (triangle.mk D E N) := 
sorry

end area_KBLN_eq_area_DEN_l361_361353


namespace distance_O_to_plane_is_2sqrt14_l361_361611

-- Define the conditions: radius of the sphere and sides of the triangle
def radius_of_sphere : ℝ := 9
def side_a : ℝ := 12
def side_b : ℝ := 35
def hypotenuse_c : ℝ := 37

-- Define the center of the sphere O
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def center_of_sphere : Point3D := ⟨0, 0, 0⟩

-- Define the plane determined by the triangle
def plane_of_triangle : set Point3D := sorry -- The exact plane definition is abstracted

-- Define the distance function between a point and a plane
noncomputable def distance_point_to_plane (p : Point3D) (plane : set Point3D) : ℝ := sorry

-- The exact distance from the center of the sphere to the plane
def distance_from_O_to_plane : ℝ := 2 * Real.sqrt 14

-- Statement: Prove that the distance between O and the plane is 2 * sqrt(14)
theorem distance_O_to_plane_is_2sqrt14 :
  distance_point_to_plane center_of_sphere plane_of_triangle = distance_from_O_to_plane := sorry

end distance_O_to_plane_is_2sqrt14_l361_361611


namespace estimated_probability_exactly_two_days_of_rain_l361_361536

theorem estimated_probability_exactly_two_days_of_rain
    (prob_rain : ℝ)
    (rainy_days : set ℕ)
    (random_sets : list ℕ)
    (target_probability : ℝ) :
    prob_rain = 0.6 →
    rainy_days = {1, 2, 3, 4, 5, 6} →
    random_sets = [180, 792, 454, 417, 165, 809, 798, 386, 196, 206] →
    target_probability = 2/5 :=
by
  intros h_prob h_rainy h_sets
  sorry

end estimated_probability_exactly_two_days_of_rain_l361_361536


namespace sqrt_49_mul_sqrt_25_l361_361996

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l361_361996


namespace sector_area_ratio_l361_361470

theorem sector_area_ratio (angle_AOE angle_FOB : ℝ) (h1 : angle_AOE = 40) (h2 : angle_FOB = 60) : 
  (180 - angle_AOE - angle_FOB) / 360 = 2 / 9 :=
by
  sorry

end sector_area_ratio_l361_361470


namespace max_sum_seq_n_l361_361045

noncomputable def a_n (n : ℕ) : ℤ := -(n^2 : ℤ) + 10 * n + 11

noncomputable def sum_seq (n : ℕ) : ℤ :=
  ∑ k in Finset.range (n + 1), a_n k

theorem max_sum_seq_n (n : ℕ) :
  n = 10 ∨ n = 11 ↔ (∀ m : ℕ, sum_seq m ≤ sum_seq n) :=
sorry

end max_sum_seq_n_l361_361045


namespace average_and_variance_machine_performance_comparison_l361_361460

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (List.map (λ x => (x - m) ^ 2) l).sum / l.length

def machine_A_defective : List ℝ := [0, 2, 1, 0, 3, 0, 2, 1, 2, 4]

def machine_B_defective : List ℝ := [2, 1, 1, 2, 1, 0, 2, 1, 3, 2]

theorem average_and_variance : 
  mean machine_A_defective = 1.5 ∧ 
  mean machine_B_defective = 1.5 ∧ 
  variance machine_A_defective = 1.65 ∧ 
  variance machine_B_defective = 0.65 :=
by
  sorry

theorem machine_performance_comparison :
  mean machine_A_defective = mean machine_B_defective ∧
  variance machine_A_defective = 1.65 ∧ 
  variance machine_B_defective = 0.65 → 
  variance machine_B_defective < variance machine_A_defective :=
by
  intro h
  cases h with h_mean h_var
  cases h_var with h_varA h_varB
  simp [h_varA, h_varB]
  sorry

end average_and_variance_machine_performance_comparison_l361_361460


namespace sum_of_consecutive_not_divisible_l361_361893

theorem sum_of_consecutive_not_divisible (K : ℕ) (hK : K % 2 = 0) :
    ∃ (arr : List ℕ), (∀ i, i < arr.length - 1 → (arr[i] + arr[i+1]) % K ≠ 0) ∧
                      (arr.toFinset = Finset.range K \ {0}) :=
sorry

end sum_of_consecutive_not_divisible_l361_361893


namespace expected_number_of_digits_is_1_55_l361_361637

/-- Brent rolls a fair icosahedral die with numbers 1 through 20 on its faces -/
noncomputable def expectedNumberOfDigits : ℚ :=
  let P_one_digit := 9 / 20
  let P_two_digit := 11 / 20
  (P_one_digit * 1) + (P_two_digit * 2)

/-- The expected number of digits Brent will roll is 1.55 -/
theorem expected_number_of_digits_is_1_55 : expectedNumberOfDigits = 1.55 := by
  sorry

end expected_number_of_digits_is_1_55_l361_361637


namespace find_four_integers_l361_361698

noncomputable def four_integers_sum : ℕ :=
  let a := 16
  let b := 81
  let c := 625
  let d := 49
  in a + b + c + d

theorem find_four_integers :
  ∃ a b c d : ℕ, 
    a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧
    a * b * c * d = 63504000 ∧
    (∀ x y : ℕ, x ∈ [a, b, c, d] ∧ y ∈ [a, b, c, d] ∧ x ≠ y → nat.gcd x y = 1) ∧
    a + b + c + d = 771 :=
begin
  use [16, 81, 625, 49],
  repeat {split},
  all_goals { repeat {try {linarith}}, sorry },
end

end find_four_integers_l361_361698


namespace solve_equation_l361_361502

theorem solve_equation (x : ℝ) :
  x ^ (Real.log 16 x ^ 2 / Real.log 2) -
  4 * x ^ (Real.log (4 * x) / Real.log 2 + 1) -
  16 * x ^ (Real.log (4 * x) / Real.log 2 + 2) +
  64 * x ^ 3 = 0 →
  x = 4 ∨ x = 1/4 ∨ x = 2 := by
  sorry

end solve_equation_l361_361502


namespace quadratic_with_equal_real_roots_l361_361225

theorem quadratic_with_equal_real_roots : 
  ∃! (eq : ℝ → ℝ), (eq = λ x, x^2 + 1) ∨ (eq = λ x, x^2 - 1) ∨ 
                     (eq = λ x, x^2 - 2x + 1) ∨ (eq = λ x, x^2 - 2x - 1) ∧
                     (∃ x, eq x = 0 ∧ ∃ y, eq y = 0 ∧ x = y) :=
by {
  sorry
}

end quadratic_with_equal_real_roots_l361_361225


namespace liking_songs_proof_l361_361898

def num_ways_liking_songs : ℕ :=
  let total_songs := 6
  let pair1 := 1
  let pair2 := 2
  let ways_to_choose_pair1 := Nat.choose total_songs pair1
  let remaining_songs := total_songs - pair1
  let ways_to_choose_pair2 := Nat.choose remaining_songs pair2 * Nat.choose (remaining_songs - pair2) pair2
  let final_song_choices := 4
  ways_to_choose_pair1 * ways_to_choose_pair2 * final_song_choices * 3 -- multiplied by 3 for the three possible pairs

theorem liking_songs_proof :
  num_ways_liking_songs = 2160 :=
  by sorry

end liking_songs_proof_l361_361898


namespace number_of_equidistant_planes_l361_361543

-- Define the main condition: four non-coplanar points
variables (A B C D : Point)
-- Instead of Point, we'll abstractly define that these points exist in some space and are not coplanar
noncomputable def is_non_coplanar : Prop := 
  ¬ ∃ (plane : Plane), A ∈ plane ∧ B ∈ plane ∧ C ∈ plane ∧ D ∈ plane

-- Define the property of plane α being equidistant from all four points
noncomputable def is_equidistant (α : Plane) : Prop :=
  ∀ (point : Point), point = A ∨ point = B ∨ point = C ∨ point = D → dist_point_plane point α = k

-- State the main theorem: the number of such planes is 7
theorem number_of_equidistant_planes (h : is_non_coplanar A B C D): 
  ∃ (S : Set Plane), S.finite ∧ S.card = 7 ∧ ∀ α ∈ S, is_equidistant A B C D α :=
sorry

end number_of_equidistant_planes_l361_361543


namespace distance_origin_to_point_l361_361799

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l361_361799


namespace mark_more_hours_than_kate_l361_361473

-- Definitions for the problem
variable (K : ℕ)  -- K is the number of hours charged by Kate
variable (P : ℕ)  -- P is the number of hours charged by Pat
variable (M : ℕ)  -- M is the number of hours charged by Mark

-- Conditions
def total_hours := K + P + M = 216
def pat_kate_relation := P = 2 * K
def pat_mark_relation := P = (1 / 3) * M

-- The statement to be proved
theorem mark_more_hours_than_kate (K P M : ℕ) (h1 : total_hours K P M)
  (h2 : pat_kate_relation K P) (h3 : pat_mark_relation P M) :
  (M - K = 120) :=
by
  sorry

end mark_more_hours_than_kate_l361_361473


namespace Larry_wins_game_probability_l361_361035

noncomputable def winning_probability_Larry : ℚ :=
  ∑' n : ℕ, if n % 3 = 0 then (2 / 3) ^ (n / 3 * 3) * (1 / 3) else 0

theorem Larry_wins_game_probability : winning_probability_Larry = 9 / 19 :=
by
  sorry

end Larry_wins_game_probability_l361_361035


namespace find_angle_l361_361145

structure Square where
  center : Point
  vertices : List Point
  is_square : ∀ (A B C D : Point) (h : vertices = [A, B, C, D]),
                dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
                angle A B C = 90 ∧ angle B C D = 90 ∧ angle C D A = 90

structure RegularHexagon where
  center : Point
  vertices : List Point
  is_regular_hexagon : ∀ (A B C D E F : Point) (h : vertices = [A, B, C, D, E, F]),
                        dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E F ∧ dist E F = dist F A ∧
                        angle A B C = 120 ∧ angle B C D = 120 ∧ angle C D E = 120 ∧ angle D E F = 120 ∧ angle E F A = 120 ∧ angle F A B = 120

def angle_in_square_hexagon (sq : Square) (hex : RegularHexagon) : ℝ :=
  105 -- The calculated angle alpha in degrees

theorem find_angle {sq : Square} {hex : RegularHexagon} (h : ∃ A ∈ sq.vertices, ∃ O ∈ hex.vertices, O = sq.center) :
  angle_in_square_hexagon sq hex = 105 :=
sorry

end find_angle_l361_361145


namespace sampling_method_is_systematic_l361_361596

axiom total_population_large : Nat
axiom num_classes : Nat := 12
axiom students_per_class : Nat := 50
axiom chosen_student_number : Nat := 40
axiom sampling_method : Type -- Representing the sampling method as a type
axiom systematic_sampling : sampling_method -- Defining systematic_sampling as one of the types of sampling methods

-- Definition to represent that the given sampling method is systematic sampling under the given conditions
theorem sampling_method_is_systematic : 
  (total_population_large > students_per_class * num_classes) ∧ 
  (∀ class_id : Nat, class_id < num_classes → ∃ student_id : Nat, student_id = chosen_student_number) →
  sampling_method = systematic_sampling := 
  by
    sorry

end sampling_method_is_systematic_l361_361596


namespace students_with_equal_scores_l361_361093

theorem students_with_equal_scores 
  (n : ℕ)
  (scores : Fin n → Fin (n - 1)): 
  ∃ i j : Fin n, i ≠ j ∧ scores i = scores j := 
by 
  sorry

end students_with_equal_scores_l361_361093


namespace proj_ratio_l361_361111

theorem proj_ratio (a b : ℝ) (h : (λ (v : ℝ × ℝ), (9 / 50 * v.1 - 15 / 50 * v.2, -15 / 50 * v.1 + 41 / 50 * v.2)) (a, b) = (a, b)) :
  b / a = -41 / 15 :=
by
  sorry

end proj_ratio_l361_361111


namespace degree_divisor_l361_361205

noncomputable def f : polynomial ℝ := sorry  -- degree 17
noncomputable def q : polynomial ℝ := sorry  -- degree 10
noncomputable def r : polynomial ℝ := 5 * X^5 + 2 * X^4 - 3 * X^3 + X - 8

theorem degree_divisor :
  ∃ d : polynomial ℝ, degree f = 17 ∧ degree q = 10 ∧ r = 5 * X^5 + 2 * X^4 - 3 * X^3 + X - 8 ∧
  degree d + 10 = 17 :=
begin
  sorry  -- Proof is omitted as per instructions
end

end degree_divisor_l361_361205


namespace distance_to_point_is_17_l361_361823

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem distance_to_point_is_17 :
  distance_from_origin 8 (-15) = 17 :=
by
  sorry

end distance_to_point_is_17_l361_361823


namespace max_coins_identifiable_l361_361585

theorem max_coins_identifiable (n : ℕ) : exists (c : ℕ), c = 2 * n^2 + 1 :=
by
  sorry

end max_coins_identifiable_l361_361585


namespace number_of_solutions_l361_361113

theorem number_of_solutions : 
  { p : ℝ × ℝ // let a := p.1, b := p.2 in (a + b * complex.I)^6 = a - b * complex.I }.to_finset.card = 8 := 
sorry

end number_of_solutions_l361_361113


namespace number_of_ordered_triplets_l361_361308

theorem number_of_ordered_triplets :
  ∃ count : ℕ, (∀ (a b c : ℕ), lcm a b = 1000 ∧ lcm b c = 2000 ∧ lcm c a = 2000 →
  count = 70) :=
sorry

end number_of_ordered_triplets_l361_361308


namespace nancy_history_books_l361_361066

/-- Nancy started with 46 books in total on the cart.
    She shelved 8 romance books and 4 poetry books from the top section.
    She shelved 5 Western novels and 6 biographies from the bottom section.
    Half the books on the bottom section were mystery books.
    Prove that Nancy shelved 12 history books.
-/
theorem nancy_history_books 
  (total_books : ℕ)
  (romance_books : ℕ)
  (poetry_books : ℕ)
  (western_novels : ℕ)
  (biographies : ℕ)
  (bottom_books_half_mystery : ℕ)
  (history_books : ℕ) :
  (total_books = 46) →
  (romance_books = 8) →
  (poetry_books = 4) →
  (western_novels = 5) →
  (biographies = 6) →
  (bottom_books_half_mystery = 11) →
  (history_books = total_books - ((romance_books + poetry_books) + (2 * (western_novels + biographies)))) →
  history_books = 12 :=
by
  intros
  sorry

end nancy_history_books_l361_361066


namespace tess_heart_stickers_l361_361916

theorem tess_heart_stickers (H S T : ℕ) (h1 : S = 27) (h2 : T = H + S) (h3 : T % 9 = 0) : H = 9 := 
begin
  -- Proof goes here.
  sorry
end

end tess_heart_stickers_l361_361916


namespace find_angle_CDE_l361_361837

-- Define the angles as constants
constant angle_A : ℝ
constant angle_B : ℝ
constant angle_C : ℝ
constant angle_AEB : ℝ
constant angle_BED : ℝ

-- Given conditions as assumptions
axiom angle_A_is_right : angle_A = 90
axiom angle_B_is_right : angle_B = 90
axiom angle_C_is_right : angle_C = 90
axiom angle_AEB_is_30 : angle_AEB = 30
axiom angle_BED_is_60 : angle_BED = 60

-- Proof statement
theorem find_angle_CDE : ∃ angle_CDE : ℝ, angle_CDE = 90 :=
by
  sorry

end find_angle_CDE_l361_361837


namespace quadratic_polynomials_count_l361_361437

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 9)

def is_quadratic (Q : ℝ → ℝ) : Prop := ∃ a b c : ℝ, ∀ x : ℝ, Q(x) = a * x^2 + b * x + c

def condition_R (Q : ℝ → ℝ) : Prop :=
  ∃ R : ℝ → ℝ, polynomials.degree R = 4 ∧
  ∀ x : ℝ, P(Q(x)) = P(x) * R(x)

def valid_quad_polynomials : ℝ := 22

theorem quadratic_polynomials_count :
  ∃ Q : (ℝ → ℝ), is_quadratic Q ∧ condition_R Q ∧ valid_quad_polynomials = 22 := sorry

end quadratic_polynomials_count_l361_361437


namespace distance_from_origin_l361_361817

theorem distance_from_origin :
  let origin := (0, 0)
      pt := (8, -15)
  in Real.sqrt ((pt.1 - origin.1)^2 + (pt.2 - origin.2)^2) = 17 :=
by
  let origin := (0, 0)
  let pt := (8, -15)
  have h1 : pt.1 - origin.1 = 8 := by rfl
  have h2 : pt.2 - origin.2 = -15 := by rfl
  simp [h1, h2]
  -- sorry will be used to skip the actual proof steps and to ensure the code builds successfully
  sorry

end distance_from_origin_l361_361817


namespace range_of_f_l361_361523

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 / 2) * (Real.exp x) * (Real.sin x + Real.cos x)

-- Define the range of f(x) as a set
def range_f : Set ℝ := {y | ∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x = y }

-- Define the expected range
def expected_range : Set ℝ := {y | (1 / 2 ≤ y) ∧ (y ≤ (1 / 2) * (Real.exp (π / 2))) }

theorem range_of_f :
  range_f = expected_range :=
sorry

end range_of_f_l361_361523


namespace coverable_faces_l361_361944

theorem coverable_faces (a b c : ℕ) : (a % 3 = 0 ∧ b % 3 = 0) ∨ (b % 3 = 0 ∧ c % 3 = 0) ∨ (a % 3 = 0 ∧ c % 3 = 0) → 
(the three faces of a parallelepiped with dimensions a × b × c sharing a common vertex can be covered with three-cell strips without overlaps and gaps) :=
by sorry

end coverable_faces_l361_361944


namespace distance_to_point_is_17_l361_361826

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem distance_to_point_is_17 :
  distance_from_origin 8 (-15) = 17 :=
by
  sorry

end distance_to_point_is_17_l361_361826


namespace correct_statement_C_l361_361164

theorem correct_statement_C : (∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5)) :=
by
  sorry

end correct_statement_C_l361_361164


namespace correct_statement_C_l361_361165

theorem correct_statement_C : (∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5)) :=
by
  sorry

end correct_statement_C_l361_361165


namespace correct_prediction_l361_361373

-- Declare the permutation type
def Permutation := (ℕ × ℕ × ℕ × ℕ × ℕ)

-- Declare the actual finish positions
def actual_positions : Permutation := (3, 2, 1, 4, 5)

-- Declare conditions
theorem correct_prediction 
  (P1 : Permutation) (P2 : Permutation) 
  (hP1 : P1 = (1, 2, 3, 4, 5))
  (hP2 : P2 = (1, 2, 3, 4, 5))
  (hP1_correct : ∃! i ∈ (1..5), (λ i, nth (P1.to_list) (i - 1) == nth (actual_positions.to_list))  = 3)
  (hP2_correct : ∃! i ∈ (1..5), (λ i, nth (P2.to_list) (i - 1) == nth (actual_positions.to_list))  = 2)
  : actual_positions = (3, 2, 1, 4, 5) 
:= sorry

end correct_prediction_l361_361373


namespace peak_numbers_count_l361_361978

def is_peak_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), 
    100 ≤ n ∧ n < 1000 ∧ 
    n = 100 * x + 10 * y + z ∧ 
    x > y ∧ x > z

theorem peak_numbers_count : 
  ∃ (count : ℕ), 
    (count = (∑ x in Finset.range 9, (x + 1 - 1) * (x + 1 - 1)) + 
             (∑ x in Finset.range 9, x + 1 - 1)) ∧ 
    count = 240 :=
by
  use 240
  split
  . sorry
  . rfl

end peak_numbers_count_l361_361978


namespace cartesian_c1_cartesian_c2_range_mn_l361_361408

open Real

-- Given conditions: Parametric equations for C1 and Polar description for C2
def parametric_eq_c1 (φ : ℝ) : ℝ × ℝ := (2 * cos φ, sin φ)
def polar_center_c2 : ℝ × ℝ := (3, π / 2)
def radius_c2 : ℝ := 1

-- Cartesian equations for C1 and C2
def cartesian_eq_c1 (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1
def cartesian_eq_c2 (x y : ℝ) : Prop := x^2 + (y - 3) ^ 2 = 1

-- Range of |MN|
def mn_distance (M N : ℝ × ℝ) : ℝ := sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)
def range {|MN| } : Prop := ∀ (M N : ℝ × ℝ), 
  parametric_eq_c1 ∃ φ, 
  polar_center_c2 ∃ θ, 
  mn_distance M N = abs (mn_distance (parametric_eq_c1 φ) (polar_center_c2, radius_c2 θ)).

theorem cartesian_c1 (φ : ℝ) : cartesian_eq_c1 (parametric_eq_c1 φ).1 (parametric_eq_c1 φ).2 := 
  sorry

theorem cartesian_c2 (x y : ℝ) : (x, y) ∈ {(x, y) | cartesian_eq_c2 x y} := 
  sorry

theorem range_mn : range [1, 5] :=
  sorry

end cartesian_c1_cartesian_c2_range_mn_l361_361408


namespace range_of_m_eq_2_to_4_l361_361105

def f (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem range_of_m_eq_2_to_4 :
  (∀ m : ℝ, (∀ x ∈ set.Icc 0 m, f x ≤ 5) ∧ (∃ x ∈ set.Icc 0 m, f x = 1)) ↔ m ∈ set.Icc 2 4 :=
  sorry

end range_of_m_eq_2_to_4_l361_361105


namespace storm_time_l361_361924

def time_corrected (hours tens units : ℕ) (minutes tens units : ℕ) : Prop :=
(hours tens units = 1 ∨ hours tens units = 3) ∧
(hours units = 1 ∨ hours units = 9) ∧
(minutes tens units = 1 ∨ minutes tens units = 9) ∧
(minutes units = 8 ∨ minutes units = 0)

theorem storm_time :
  ∃ hours tens units: ℕ, ∃ hours units: ℕ, ∃ minutes tens units: ℕ, ∃ minutes units: ℕ,
  time_corrected hours tens units minutes tens units ∧
  (hours tens units = 1) ∧ (hours units = 1) ∧ (minutes tens units = 1) ∧ (minutes units = 8) :=
begin
  use 1,
  use 1,
  use 1,
  use 8,
  split,
  { split,
    { left, refl },
    split,
    { left, refl },
    split,
    { left, refl },
    { left, refl }},
  exact ⟨rfl, rfl, rfl, rfl⟩
end

end storm_time_l361_361924


namespace unique_handshakes_at_convention_l361_361591

def total_sets_of_twins := 10
def total_sets_of_triplets := 7
def total_twins := total_sets_of_twins * 2
def total_triplets := total_sets_of_triplets * 3

def handshakes_among_twins := (total_twins * (total_twins - 2)) / 2
def handshakes_among_triplets := (total_triplets * (total_triplets - 3)) / 2

def cross_handshakes_twins_triplets := (total_twins * (2 / 3 * total_triplets)).toInt
def cross_handshakes_triplets_twins := (total_triplets * (2 / 3 * (total_twins - 1)).toInt)

def total_cross_handshakes := cross_handshakes_twins_triplets + cross_handshakes_triplets_twins

def total_handshakes := handshakes_among_twins + handshakes_among_triplets + total_cross_handshakes

theorem unique_handshakes_at_convention : total_handshakes = 922 := by
  sorry

end unique_handshakes_at_convention_l361_361591


namespace eq_g_of_f_l361_361719

def f (x : ℝ) : ℝ := 3 * x - 5
def g (x : ℝ) : ℝ := 6 * x - 29

theorem eq_g_of_f (x : ℝ) : 2 * (f x) - 19 = g x :=
by 
  sorry

end eq_g_of_f_l361_361719


namespace sum_of_cubes_5_sum_of_cubes_formula_sum_of_cubes_100_sum_of_cubes_segment_l361_361883

-- Proof problem: 1
theorem sum_of_cubes_5 : (1^3 + 2^3 + 3^3 + 4^3 + 5^3) = 225 :=
sorry

-- Proof problem: 2
theorem sum_of_cubes_formula (n : ℕ) : ∑ i in Finset.range n, (i + 1)^3 = (n * (n + 1) / 2)^2 :=
sorry

-- Proof problem: 3
theorem sum_of_cubes_100 : ∑ i in Finset.range 100, (i + 1)^3 = 5050^2 :=
sorry

-- Proof problem: 4
theorem sum_of_cubes_segment : 
  11 ^ 3 + 12 ^ 3 + 13 ^ 3 + 14 ^ 3 + 15 ^ 3 + 16 ^ 3 + 17 ^ 3 + 18 ^ 3 + 19 ^ 3 + 20 ^ 3 = 
  41075 :=
sorry

end sum_of_cubes_5_sum_of_cubes_formula_sum_of_cubes_100_sum_of_cubes_segment_l361_361883


namespace circumcircle_radius_l361_361836

-- Define the variables and the conditions
variables (A B C O L N : Type*)
variable [acute_triangle : acute_angled_triangle A B C]
variables (a : ℝ) (alpha : ℝ)
variable (AL_is_altitude : altitude A B C L)
variable (CN_is_altitude : altitude C A B N)
variable (AC_eq_a : AC = a)
variable (angle_ABC_eq_alpha : ∡ABC = alpha)

-- Define the goal
theorem circumcircle_radius {a alpha : ℝ} (h1 : acute_angled_triangle A B C)
  (h2 : altitude A B C L) (h3 : altitude C A B N) (h4 : AC = a) (h5 : ∡ABC = alpha) :
  radius (circumcircle_through B L N) = (a / 2) * real.cot alpha :=
sorry

end circumcircle_radius_l361_361836


namespace value_of_polynomial_l361_361706

variable (a : ℝ)

-- Condition
axiom cond : a^2 + a - 3 = 0

-- Proof statement
theorem value_of_polynomial (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 :=
by
  sorry

end value_of_polynomial_l361_361706


namespace expected_value_is_correct_l361_361618

-- Given conditions
def prob_heads : ℚ := 2 / 5
def prob_tails : ℚ := 3 / 5
def win_amount_heads : ℚ := 5
def loss_amount_tails : ℚ := -3

-- Expected value calculation
def expected_value : ℚ := prob_heads * win_amount_heads + prob_tails * loss_amount_tails

-- Property to prove
theorem expected_value_is_correct : expected_value = 0.2 := sorry

end expected_value_is_correct_l361_361618


namespace train_speed_in_kmh_l361_361219

/-- Definition of length of the train in meters. -/
def train_length : ℕ := 200

/-- Definition of time taken to cross the electric pole in seconds. -/
def time_to_cross : ℕ := 20

/-- The speed of the train in km/h is 36 given the length of the train and time to cross. -/
theorem train_speed_in_kmh (length : ℕ) (time : ℕ) (h_len : length = train_length) (h_time: time = time_to_cross) : 
  (length / time : ℚ) * 3.6 = 36 := 
by
  sorry

end train_speed_in_kmh_l361_361219


namespace weightlifter_total_weight_l361_361221

theorem weightlifter_total_weight (weight_one_hand : ℕ) (num_hands : ℕ) (condition: weight_one_hand = 8 ∧ num_hands = 2) :
  2 * weight_one_hand = 16 :=
by
  sorry

end weightlifter_total_weight_l361_361221


namespace acid_fraction_in_third_flask_correct_l361_361127

noncomputable def acid_concentration_in_third_flask 
  (w : ℝ) (W : ℝ) 
  (h1 : 10 / (10 + w) = 1 / 20)
  (h2 : 20 / (20 + (W - w)) = 7 / 30) 
  : ℝ := 
30 / (30 + W)

theorem acid_fraction_in_third_flask_correct
  (w : ℝ) (W : ℝ)
  (h1 : 10 / (10 + w) = 1 / 20)
  (h2 : 20 / (20 + (W - w)) = 7 / 30) :
  acid_concentration_in_third_flask w W h1 h2 = 21 / 200 :=
begin
  sorry
end

end acid_fraction_in_third_flask_correct_l361_361127


namespace sqrt_49_mul_sqrt_25_l361_361998

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l361_361998


namespace age_of_15th_student_is_15_l361_361513

-- Define the total number of students
def total_students : Nat := 15

-- Define the average age of all 15 students together
def avg_age_all_students : Nat := 15

-- Define the average age of the first group of 7 students
def avg_age_first_group : Nat := 14

-- Define the average age of the second group of 7 students
def avg_age_second_group : Nat := 16

-- Define the total age based on the average age and number of students
def total_age_all_students : Nat := total_students * avg_age_all_students
def total_age_first_group : Nat := 7 * avg_age_first_group
def total_age_second_group : Nat := 7 * avg_age_second_group

-- Define the age of the 15th student
def age_of_15th_student : Nat := total_age_all_students - (total_age_first_group + total_age_second_group)

-- Theorem: prove that the age of the 15th student is 15 years
theorem age_of_15th_student_is_15 : age_of_15th_student = 15 := by
  -- The proof will go here
  sorry

end age_of_15th_student_is_15_l361_361513


namespace log_geometric_ratio_l361_361793

variable (a : ℕ → ℝ) (q : ℝ)

theorem log_geometric_ratio :
  (∀ n, a (n + 1) = a n * q) →
  (q = 2) →
  (∀ n, 0 < a n) →
  log 2 ((a 2 + a 3) / (a 0 + a 1)) = 2 :=
by
  intros h_geom_ratio h_q_pos h_a_pos
  -- Proof here
  sorry

end log_geometric_ratio_l361_361793


namespace quadratic_roots_l361_361693

theorem quadratic_roots (m x1 x2 : ℝ) 
  (h1 : 2*x1^2 + 4*m*x1 + m = 0)
  (h2 : 2*x2^2 + 4*m*x2 + m = 0)
  (h3 : x1 ≠ x2)
  (h4 : x1^2 + x2^2 = 3/16) : 
  m = -1/8 := 
sorry

end quadratic_roots_l361_361693


namespace sum_mod_9_l361_361235

theorem sum_mod_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by
  sorry

end sum_mod_9_l361_361235


namespace find_monic_quartic_polynomial_l361_361667

noncomputable def monic_quartic_polynomial_with_roots 
  (a b : ℚ) [char_zero ℚ] (p : polynomial ℚ) : Prop :=
monic p ∧ p.root_set ℚ = {a, a.conjugate, b, b.conjugate}

theorem find_monic_quartic_polynomial :
  ∃ p : polynomial ℚ, (monic_quartic_polynomial_with_roots 
      (3 + real.sqrt 5) 
      (2 - real.sqrt 7) 
      p) ∧ p = X^4 - 10*X^3 + 25*X^2 + 2*X - 12 :=
begin
  sorry

end find_monic_quartic_polynomial_l361_361667


namespace smallest_n_satisfying_sum_log2_l361_361644

theorem smallest_n_satisfying_sum_log2 :
  ∃ n : ℕ, (∑ k in Finset.range n.succ, Real.log2 (1 + 1 / 2 ^ (3^k))) ≥ 1 + Real.log2 (4030 / 4031) ∧ 
           ∀ m : ℕ, m < n → (∑ k in Finset.range m.succ, Real.log2 (1 + 1 / 2 ^ (3^k))) < 1 + Real.log2 (4030 / 4031) :=
begin
  sorry
end

end smallest_n_satisfying_sum_log2_l361_361644


namespace at_least_six_destinations_l361_361616

theorem at_least_six_destinations (destinations : ℕ) (tickets_sold : ℕ) (h_dest : destinations = 200) (h_tickets : tickets_sold = 3800) :
  ∃ k ≥ 6, ∃ t : ℕ, (∃ f : Fin destinations → ℕ, (∀ i : Fin destinations, f i ≤ t) ∧ (tickets_sold ≤ t * destinations) ∧ ((∃ i : Fin destinations, f i = k) → k ≥ 6)) :=
by
  sorry

end at_least_six_destinations_l361_361616


namespace distinct_values_in_expression_rearrangement_l361_361741

theorem distinct_values_in_expression_rearrangement : 
  ∀ (exp : ℕ), exp = 3 → 
  (∃ n : ℕ, n = 3 ∧ 
    let a := exp ^ (exp ^ exp)
    let b := exp ^ ((exp ^ exp) ^ exp)
    let c := ((exp ^ exp) ^ exp) ^ exp
    let d := (exp ^ (exp ^ exp)) ^ exp
    let e := (exp ^ exp) ^ (exp ^ exp)
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :=
by
  sorry

end distinct_values_in_expression_rearrangement_l361_361741


namespace exterior_angle_measure_130_degrees_l361_361214

noncomputable def measure_of_exterior_angle_BAC 
    (square : Type) (nonagon : Type) 
    (common_side_AD : square × nonagon) 
    : ℚ :=
  130

theorem exterior_angle_measure_130_degrees
    (Sqr : Type) (Ngon : Type)
    (shared_side : Sqr × Ngon)
    (common_plane : ∀ a b : Sqr, a = b → ∀ c d : Ngon, c = d → a ∈ common_plane → b ∈ common_plane → c ∈ common_plane → d ∈ common_plane) :
  measure_of_exterior_angle_BAC Sqr Ngon shared_side = 130 :=
sorry

end exterior_angle_measure_130_degrees_l361_361214


namespace scientific_notation_of_105000_l361_361884

theorem scientific_notation_of_105000 : (105000 : ℝ) = 1.05 * 10^5 := 
by {
  sorry
}

end scientific_notation_of_105000_l361_361884


namespace rectangle_area_l361_361207

-- Define the conditions as Lean definitions
def rectangle_area_from_diagonal_length (x : ℝ) : ℝ :=
  let w : ℝ := x / Real.sqrt 10
  let l : ℝ := 3 * w
  l * w

-- Statement to prove that the area of the rectangle equals 3 / 10 * x^2
theorem rectangle_area (x : ℝ) : rectangle_area_from_diagonal_length x = (3 / 10) * x^2 :=
  sorry

end rectangle_area_l361_361207


namespace BC_length_l361_361410

-- Define terms and conditions
variables (A B C D : Type) [Fig : Trapezoid ABCD]
variables (h1 : Parallel AB CD) (h2 : Perpendicular AC CD)
variables (h3 : CD = 15) (h4 : tan D = 2) (h5 : tan B = 3)

-- Define the type of the Lean statement
theorem BC_length : BC = 10 * Real.sqrt 10 :=
by
  sorry

end BC_length_l361_361410


namespace cards_per_set_is_13_l361_361847

-- Definitions based on the conditions
def total_cards : ℕ := 365
def sets_to_brother : ℕ := 8
def sets_to_sister : ℕ := 5
def sets_to_friend : ℕ := 2
def total_sets_given : ℕ := sets_to_brother + sets_to_sister + sets_to_friend
def total_cards_given : ℕ := 195

-- The problem to prove
theorem cards_per_set_is_13 : total_cards_given / total_sets_given = 13 :=
  by
  -- Here we would provide the proof, but for now, we use sorry
  sorry

end cards_per_set_is_13_l361_361847


namespace ny_mets_fans_l361_361180

-- Let Y be the number of NY Yankees fans
-- Let M be the number of NY Mets fans
-- Let R be the number of Boston Red Sox fans
variables (Y M R : ℕ)

-- Given conditions
def ratio_Y_M : Prop := 3 * M = 2 * Y
def ratio_M_R : Prop := 4 * R = 5 * M
def total_fans : Prop := Y + M + R = 330

-- The theorem to prove
theorem ny_mets_fans (h1 : ratio_Y_M Y M) (h2 : ratio_M_R M R) (h3 : total_fans Y M R) : M = 88 :=
sorry

end ny_mets_fans_l361_361180


namespace album_count_l361_361464

def albums_total (A B K M C : ℕ) : Prop :=
  A = 30 ∧ B = A - 15 ∧ K = 6 * B ∧ M = 5 * K ∧ C = 3 * M ∧ (A + B + K + M + C) = 1935

theorem album_count (A B K M C : ℕ) : albums_total A B K M C :=
by
  sorry

end album_count_l361_361464


namespace gcd_of_28430_and_39674_l361_361577

theorem gcd_of_28430_and_39674 : Nat.gcd 28430 39674 = 2 := 
by 
  sorry

end gcd_of_28430_and_39674_l361_361577


namespace problem_statement_l361_361845

variable {ABC : Type} -- Triangle type
variables {A B C : Angle} -- Angles in the triangle
variables {a b c R : Real} -- Sides and circumradius of the triangle
variables {p q : Prop}

-- Define the propositions and conditions
def law_of_sines (a b c A B C R : Real) : Prop :=
  a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C

def prop_p (a b c R : Real) (A B C : Angle) : Prop :=
  a > a * cos B + b * cos A → A > C

def prop_q (A B : Angle) : Prop :=
  A > B → sin A > sin B

-- The propositions to verify
def conclusion_1_true (A B : Angle) : Prop :=
  (A ≤ B → sin A ≤ sin B) ∧ (¬ (A > B) → ¬ (sin A > sin B)) ∧ (¬ (sin A > sin B) → ¬ (A > B))

def combined_proposition_false (a b c R : Real) (A B C : Angle) : Prop :=
  ¬ (prop_p a b c R A B C ∧ prop_q A B)

def disjunction_proposition_false (a b c R : Real) (A B C : Angle) : Prop :=
  ¬ (prop_p a b c R A B ∨ ¬ prop_q A B)

def negation_disjunction_false (a b c R : Real) (A B C : Angle) : Prop :=
  ¬ (¬ (prop_p a b c R A B) ∨ ¬ (prop_q A B))

-- The problem statement to prove
theorem problem_statement 
  (h_sines : law_of_sines a b c A B C R)
  (h_prop_p : prop_p a b c R A B C)
  (h_prop_q : prop_q A B) : 
  (conclusion_1_true A B ∧ negation_disjunction_false a b c R A B) :=
by
  sorry

end problem_statement_l361_361845


namespace two_a_minus_two_d_eq_zero_l361_361246

variable {a b c d : ℝ}
variable (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) (c_ne_zero : c ≠ 0) (d_ne_zero : d ≠ 0)
variable (b_eq_3a : b = 3 * a) (c_eq_3d : c = 3 * d)

-- Define the function g
def g (x : ℝ) : ℝ :=
  (2 * a * x - b) / (c * x - 2 * d)

-- Given that g(g(x)) = x for all x in the domain of g
axiom ggx_eq_x {x : ℝ} (h : x ≠ 2 * d / c) (h' : (2 * a * x - b) ≠ 0) : g (g x) = x

-- The statement to be proved
theorem two_a_minus_two_d_eq_zero : 2 * a - 2 * d = 0 := sorry

end two_a_minus_two_d_eq_zero_l361_361246


namespace initial_girls_count_l361_361588

variable (q : ℕ)

/-- The initial number of girls in the choir was 24 -/
theorem initial_girls_count (h1 : 0.6 * q - 4 = 0.5 * q) : 0.6 * q = 24 := by
  sorry

end initial_girls_count_l361_361588


namespace max_value_function_l361_361940

theorem max_value_function : ∃ x ∈ set.Icc 1 2, 4^x + 2^(x+1) + 5 = 29 :=
by
  sorry

end max_value_function_l361_361940


namespace paper_thickness_after_folds_l361_361118

theorem paper_thickness_after_folds :
  ∀ (initial_thickness : ℝ) (folds : ℕ)
  (h_initial : initial_thickness = 0.1) 
  (h_folds : folds = 10),
  initial_thickness * 2^folds ≈ 1 :=
by
  intros initial_thickness folds h_initial h_folds
  sorry

end paper_thickness_after_folds_l361_361118


namespace actual_time_of_storm_l361_361927

theorem actual_time_of_storm
  (malfunctioned_hours_tens_digit : ℕ)
  (malfunctioned_hours_units_digit : ℕ)
  (malfunctioned_minutes_tens_digit : ℕ)
  (malfunctioned_minutes_units_digit : ℕ)
  (original_time : ℕ × ℕ)
  (hours_tens_digit : ℕ := 2)
  (hours_units_digit : ℕ := 0)
  (minutes_tens_digit : ℕ := 0)
  (minutes_units_digit : ℕ := 9) :
  (malfunctioned_hours_tens_digit = hours_tens_digit + 1 ∨ malfunctioned_hours_tens_digit = hours_tens_digit - 1) →
  (malfunctioned_hours_units_digit = hours_units_digit + 1 ∨ malfunctioned_hours_units_digit = hours_units_digit - 1) →
  (malfunctioned_minutes_tens_digit = minutes_tens_digit + 1 ∨ malfunctioned_minutes_tens_digit = minutes_tens_digit - 1) →
  (malfunctioned_minutes_units_digit = minutes_units_digit + 1 ∨ malfunctioned_minutes_units_digit = minutes_units_digit - 1) →
  original_time = (11, 18) :=
by
  sorry

end actual_time_of_storm_l361_361927


namespace curve_C1_general_equation_max_AB_distance_l361_361011

-- Condition: Parametric equations of line l1
def l1 (t k : ℝ) : ℝ × ℝ :=
  (4 - t, k * t)

-- Condition: General equation of line l2
def l2 (x k : ℝ) : ℝ :=
  (1/k) * x 

-- Question 1: Find the general equation of curve C1
theorem curve_C1_general_equation (x y : ℝ) (h1 : ∃ t k, l1 t k = (x, y) ∧ k ≠ 0) 
  (h2 : y = l2 x (∃ t k, l1 t k = (x, y) ∧ ¬(y = 0))): 
  x^2 + y^2 - 4 * x = 0 :=
sorry

-- Condition: Equation of line l3 in polar coordinates
def l3_polar (ρ θ : ℝ) : Prop :=
  ρ * real.sin (theta - (real.pi / 4)) = real.sqrt 2

-- Condition: Equation of line l3 in Cartesian coordinates
def l3_cartesian (x y : ℝ) : Prop :=
  y = x + 2

-- Condition: Distance between two points in Cartesian coordinates
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Question 2: Find the maximum value of |AB|
theorem max_AB_distance (A B : ℝ × ℝ) (h1 : l3_cartesian A.1 A.2) 
  (h2 : curve_C1_general_equation B.1 B.2 sorry) 
  (h3 : ∃ k, abs (l2 A.1 k - A.2) = real.pi / 4):
  distance A B = 4 + 2 * real.sqrt 2 :=
sorry

end curve_C1_general_equation_max_AB_distance_l361_361011


namespace distance_between_vertices_l361_361278

/-
Problem statement:
Prove that the distance between the vertices of the hyperbola
\(\frac{x^2}{144} - \frac{y^2}{64} = 1\) is 24.
-/

/-- 
We define the given hyperbola equation:
\frac{x^2}{144} - \frac{y^2}{64} = 1
-/
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 64 = 1

/--
We establish that the distance between the vertices of the hyperbola is 24.
-/
theorem distance_between_vertices : 
  (∀ x y : ℝ, hyperbola x y → dist (12, 0) (-12, 0) = 24) :=
by
  sorry

end distance_between_vertices_l361_361278


namespace no_possible_k_of_prime_roots_l361_361635

theorem no_possible_k_of_prime_roots :
  ∀ k : ℤ, (∃ p q : ℤ, p.prime ∧ q.prime ∧ (x^2 - 65 * x + k = 0) = (x - p) * (x - q)) → false :=
by
  sorry

end no_possible_k_of_prime_roots_l361_361635


namespace actual_positions_correct_l361_361361

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end actual_positions_correct_l361_361361


namespace feeding_times_per_day_l361_361490

-- Definitions for the given conditions
def number_of_puppies : ℕ := 7
def total_portions : ℕ := 105
def number_of_days : ℕ := 5

-- Theorem to prove the answer to the question
theorem feeding_times_per_day : 
  let portions_per_day := total_portions / number_of_days in
  let times_per_puppy := portions_per_day / number_of_puppies in
  times_per_puppy = 3 :=
by
  -- We should provide the proof here, but we will use 'sorry' to skip it
  sorry

end feeding_times_per_day_l361_361490


namespace seating_arrangement_count_l361_361540

theorem seating_arrangement_count :
  ∃ (α β : Type) (teachers : Finset α) (students : Finset β), 
    ∀ (arrangement : List (α ⊕ β)), 
    (|teachers| = 3) → 
    (|students| = 5) → 
    (arrangement.length = 8) →
    (arrangement.head ∉ teachers) →
    (∀ (i : Nat), (i < arrangement.length - 1) → (arrangement[i] ∈ teachers → arrangement[i + 1] ∉ teachers)) →
    (∃ (perm : Finset (List (α ⊕ β))),
      perm.card = Nat.factorial 5 * Nat.factorial 5 ^ 3) :=
sorry

end seating_arrangement_count_l361_361540


namespace diamond_and_face_card_probability_l361_361973

noncomputable def probability_first_diamond_second_face_card : ℚ :=
  let total_cards := 52
  let total_faces := 12
  let diamond_faces := 3
  let diamond_non_faces := 10
  (9/52) * (12/51) + (3/52) * (11/51)

theorem diamond_and_face_card_probability :
  probability_first_diamond_second_face_card = 47 / 884 := 
by {
  sorry
}

end diamond_and_face_card_probability_l361_361973


namespace isosceles_trapezoid_condition_l361_361623

-- Defining the conditions
variables {a c h : ℝ} (P : ℝ × ℝ)

-- The problem statement
theorem isosceles_trapezoid_condition : 
  (∃ P : ℝ, P.fst = 0 ∧ h^2 ≤ a * c) :=
begin
  -- We don't need to provide the proof here, just set up the structure
  sorry
end

end isosceles_trapezoid_condition_l361_361623


namespace dot_product_value_l361_361303

open Real

variables (a b : ℝ) (v₁ v₂ : ℝ^3)

-- Given conditions
def vector_b_norm : Prop := ∥b∥ = 3
def projection_condition : Prop := proj v₁ v₂ = (1/2) • v₂

-- The theorem we need to prove
theorem dot_product_value (h₁ : vector_b_norm b) (h₂ : projection_condition v₁ v₂) : v₁ ⬝ v₂ = 9 / 2 :=
sorry

end dot_product_value_l361_361303


namespace find_mother_age_l361_361900

-- Definitions for the given conditions
def serena_age_now := 9
def years_in_future := 6
def serena_age_future := serena_age_now + years_in_future
def mother_age_future (M : ℕ) := 3 * serena_age_future

-- The main statement to prove
theorem find_mother_age (M : ℕ) (h1 : M = mother_age_future M - years_in_future) : M = 39 :=
by
  sorry

end find_mother_age_l361_361900


namespace handshake_count_l361_361352

-- Definitions based on the given conditions
def married_couples := 8
def total_people := married_couples * 2
def people_per_color := total_people / 4
def handshakes_per_person :=
  let shakes_with_others := total_people - 1
  shakes_with_others - 1 - (people_per_color - 1)

-- Theorem stating the number of handshakes given the conditions
theorem handshake_count : 
  ∑ person in Finset.range total_people, handshakes_per_person / 2 = 88 :=
by
  -- proof omitted
  sorry

end handshake_count_l361_361352


namespace sqrt_mul_sqrt_l361_361993

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361993


namespace train_travel_distance_l361_361617

theorem train_travel_distance (speed time: ℕ) (h1: speed = 85) (h2: time = 4) : speed * time = 340 :=
by
-- Given: speed = 85 km/hr and time = 4 hr
-- To prove: speed * time = 340
-- Since speed = 85 and time = 4, then 85 * 4 = 340
sorry

end train_travel_distance_l361_361617


namespace solution_set_log_inequality_l361_361297

theorem solution_set_log_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∃ m, ∀ x, log a (x^2 - 2x + 3) ≥ m) :
  {x : ℝ | log a (x - 1) > 0} = {x : ℝ | x > 2} := 
by
  sorry

end solution_set_log_inequality_l361_361297


namespace find_b2_b3_b4_b5_b6_b7_l361_361546

-- Define the problem conditions
def is_valid_b (b : ℕ → ℕ) : Prop :=
∀ i, 2 ≤ i ∧ i ≤ 7 → 0 ≤ b i ∧ b i < i

def fraction_sum (b : ℕ → ℕ) : ℚ :=
(b 2) / 2! + (b 3) / 3! + (b 4) / 4! + (b 5) / 5! + (b 6) / 6! + (b 7) / 7!

-- Define the problem statement
theorem find_b2_b3_b4_b5_b6_b7 :
  ∃ b : ℕ → ℕ, is_valid_b b ∧ fraction_sum b = 17 / 23 ∧
                (b 2) + (b 3) + (b 4) + (b 5) + (b 6) + (b 7) = 11 :=
by
  sorry

end find_b2_b3_b4_b5_b6_b7_l361_361546


namespace ellipse_problem_desired_ellipse_l361_361284

def ellipse_equation (x y m n : ℝ) := (x^2 / m) + (y^2 / n) = 1

def given_ellipse := ellipse_equation 0 0 9 4

def foci_of_given_ellipse := (5, 0) -- derived from the condition

theorem ellipse_problem (m n : ℝ) (hp : 3^2 / m + (-2)^2 / n = 1)
  (hf : m - n = 5) :
  ellipse_equation 3 (-2) m n :=
begin
  sorry
end

theorem desired_ellipse :
  ellipse_equation 3 (-2) 15 10 :=
begin
  have hp: 3^2 / 15 + (-2)^2 / 10 = 1,
  {
    norm_num,
  },
  have hf: 15 - 10 = 5,
  {
    norm_num,
  },
  exact ellipse_problem 15 10 hp hf,
end

end ellipse_problem_desired_ellipse_l361_361284


namespace no_possible_k_for_prime_roots_l361_361633

theorem no_possible_k_for_prime_roots :
  ¬ ∃ (k : ℕ), (∃ (p q : ℕ), prime p ∧ prime q ∧ (x^2 - 65 * x + k) = (x - p) * (x - q)) :=
sorry

end no_possible_k_for_prime_roots_l361_361633


namespace blueberries_in_blue_box_l361_361184

theorem blueberries_in_blue_box (S B : ℕ) (h1 : S - B = 15) (h2 : S + B = 87) : B = 36 :=
by sorry

end blueberries_in_blue_box_l361_361184


namespace athlete_positions_l361_361382

theorem athlete_positions {A B C D E : Type} 
  (h1 : A = 1 ∧ B = 2 ∧ C = 3 ∧ D = 4 ∧ E = 5)
  (h2 : C = 1 ∧ E = 2 ∧ A = 3 ∧ B = 4 ∧ D = 5)
  (h3 : (A = 1 ∧ B = 2 ∧ C = 3 ∧ ¬(D = 4 ∨ E = 5)) ∨ 
       (A = 1 ∧ ¬(B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)) ∨ 
       (B = 2 ∧ C = 3 ∧ ¬(A = 1 ∨ D = 4 ∨ E = 5)) ∨ 
       (¬(A = 1 ∨ B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)))
  (h4 : (C = 1 ∧ ¬(A = 3 ∨ B = 4 ∨ D = 5)) ∨ 
       (A = 3 ∧ ¬(C = 1 ∨ B = 4 ∨ D = 5)) ∨ 
       (B = 4 ∧ ¬(C = 1 ∨ A = 3 ∨ D = 5)))
  : (C = 1 ∧ B = 2 ∧ A = 3 ∧ D = 4 ∧ E = 5) :=
by sorry

end athlete_positions_l361_361382


namespace average_of_integers_l361_361967

-- We consider a set of 5 positive integers.
variable (a b c d e : ℕ)

-- The conditions given in the problem.
def condition1 : Prop := (a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)  -- Set has 5 positive integers
def condition2 : Prop := (e - a = 10)  -- The difference between the largest and the smallest of these numbers is 10
def condition3 : Prop := (e = 78)  -- The maximum value possible for the largest of these integers is 78

-- The question: What is the average of these integers?
def question : ℝ := (a + b + c + d + e) / 5

-- The correct answer we need to prove: The average of these integers is 83.6.
theorem average_of_integers : condition1 a b c d e ∧ condition2 a b c d e ∧ condition3 a b c d e → question a b c d e = 83.6 :=
by
  sorry

end average_of_integers_l361_361967


namespace inequality_abc_l361_361446

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1):
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by
  sorry

end inequality_abc_l361_361446


namespace solutions_to_x4_eq_1_l361_361775

theorem solutions_to_x4_eq_1 (x : ℂ) : (x^4 = 1) ↔ (x = 1 ∨ x = -1 ∨ x = Complex.i ∨ x = -Complex.i) := 
sorry

end solutions_to_x4_eq_1_l361_361775


namespace grid_cut_possible_l361_361255

-- Define the grid size
def grid_size : ℕ := 8

-- Define the total number of cells in the grid
def total_cells : ℕ := grid_size * grid_size

-- Define the max number of cells per part
def max_cells_per_part : ℕ := 16

-- Define that the grid can be cut into at least 27 parts
def min_parts : ℕ := 27

-- Statement: Prove that it's possible to cut the grid into required parts maintaining the conditions
theorem grid_cut_possible : 
  ∃ (parts : finset (finset (fin (grid_size * grid_size)))),
    parts.card ≥ min_parts ∧
    ∀ part ∈ parts, part.card ≤ max_cells_per_part ∧
    ∀ (i j : fin (grid_size * grid_size)), (part i ∈ parts → part j ∈ parts → part i = part j) :=
sorry

end grid_cut_possible_l361_361255


namespace sqrt_prime_irrational_sqrt_product_primes_irrational_sqrt_ratio_primes_irrational_l361_361082

-- Definitions used in the affine theorem
open Int Set

-- Part (a) condition and proof statement
theorem sqrt_prime_irrational (p : ℕ) (h_prime : Prime p) : irrational (Real.sqrt p) := 
sorry

-- Part (b) condition and proof statement
theorem sqrt_product_primes_irrational (k : ℕ) (p : Fin k → ℕ) (h_primes : ∀ i, Prime (p i)) : irrational (Real.sqrt (∏ i in Finset.univ, p i)) :=
sorry

-- Part (c) condition and proof statement
theorem sqrt_ratio_primes_irrational (k n : ℕ) (p : Fin n → ℕ) (h_primes : ∀ i, Prime (p i)) (hk : k < n) : irrational (Real.sqrt ((∏ i in Finset.range k, p i) / (∏ i in Finset.Ico k n, p i))) :=
sorry

end sqrt_prime_irrational_sqrt_product_primes_irrational_sqrt_ratio_primes_irrational_l361_361082


namespace arithmetic_sequence_k_value_l361_361004

variable {a : ℕ → ℕ}
variable {k : ℕ}

theorem arithmetic_sequence_k_value (h₁ : a 4 + a 7 + a 10 = 15)
                                  (h₂ : (∑ i in (finset.range 15).filter (λ n, 4 ≤ n ∧ n ≤ 14), a i) = 77)
                                  (h₃ : a k = 13) : 
                                  k = 15 :=
sorry

end arithmetic_sequence_k_value_l361_361004


namespace probability_a_plus_b_divisible_by_5_l361_361143

-- Condition: Definitions for numbers on dice.
def is_dice_value (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

-- Definition of rolling two dice.
def valid_die_rolls (a b : ℕ) : Prop := is_dice_value a ∧ is_dice_value b

-- The specific probability statement we want to prove.
theorem probability_a_plus_b_divisible_by_5 :
  (∑ a b, if valid_die_rolls a b ∧ (a + b) % 5 = 0 then 1 else 0).to_rat / 36 = 7 / 36 :=
by
  sorry

end probability_a_plus_b_divisible_by_5_l361_361143


namespace complex_number_quadrant_l361_361194

theorem complex_number_quadrant (z : ℂ) (h : z * (1 + complex.i) = 2) : 
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_quadrant_l361_361194


namespace no_nat_solution_l361_361256

theorem no_nat_solution (x y z : ℕ) : ¬ (x^3 + 2 * y^3 = 4 * z^3) :=
sorry

end no_nat_solution_l361_361256


namespace no_intersection_curves_l361_361708

theorem no_intersection_curves (k : ℕ) (hn : k > 0) 
  (h_intersection : ∀ x y : ℝ, ¬(x^2 + y^2 = k^2 ∧ x * y = k)) : 
  k = 1 := 
sorry

end no_intersection_curves_l361_361708


namespace draw_defective_products_l361_361662

-- Define the problem parameters
def total_products : ℕ := 100
def defective_products : ℕ := 3
def sample_size : ℕ := 4
def chosen_defective : ℕ := 2
def non_defective_products : ℕ := total_products - defective_products
def chosen_non_defective : ℕ := sample_size - chosen_defective

-- Calculate the combination C(n, k)
def combination (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Problem statement: Prove the total number of ways to draw 4 products with exactly 2 defective is 13968
theorem draw_defective_products : 
  combination defective_products chosen_defective * combination non_defective_products chosen_non_defective = 13968 :=
sorry

end draw_defective_products_l361_361662


namespace correctFinishingOrder_l361_361401

-- Define the types of athletes
inductive Athlete
| A | B | C | D | E

-- Define the type for predictions
def Prediction := Athlete → Nat

-- Define the first prediction
def firstPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 1
    | Athlete.B => 2
    | Athlete.C => 3
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the second prediction
def secondPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 4
    | Athlete.C => 1
    | Athlete.D => 5
    | Athlete.E => 2

-- Define the actual finishing positions
noncomputable def actualFinishingPositions : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 2
    | Athlete.C => 1
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the correctness conditions for the predictions
def correctPositions (actual : Prediction) (prediction : Prediction) (k : Nat) : Prop :=
  (Set.card {x : Athlete | actual x = prediction x} = k)

-- The proof statement
theorem correctFinishingOrder :
  correctPositions actualFinishingPositions firstPrediction 3 ∧
  correctPositions actualFinishingPositions secondPrediction 2 :=
sorry

end correctFinishingOrder_l361_361401


namespace expression_values_l361_361687

-- Define the conditions as a predicate
def conditions (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a^2 - b * c = b^2 - a * c ∧ b^2 - a * c = c^2 - a * b

-- The main theorem statement
theorem expression_values (a b c : ℝ) (h : conditions a b c) :
  (∃ x : ℝ, x = (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b)) ∧ (x = 7 / 2 ∨ x = -7)) :=
by
  sorry

end expression_values_l361_361687


namespace correct_proposition_four_l361_361764

universe u

-- Definitions
variable {Point : Type u} (A B : Point) (a α : Set Point)
variable (h5 : A ∉ α)
variable (h6 : a ⊂ α)

-- The statement to be proved
theorem correct_proposition_four : A ∉ a :=
sorry

end correct_proposition_four_l361_361764


namespace john_weight_end_l361_361426

def initial_weight : ℝ := 220
def loss_percentage : ℝ := 0.1
def weight_loss : ℝ := loss_percentage * initial_weight
def weight_gain_back : ℝ := 2
def net_weight_loss : ℝ := weight_loss - weight_gain_back
def final_weight : ℝ := initial_weight - net_weight_loss

theorem john_weight_end :
  final_weight = 200 := 
by 
  sorry

end john_weight_end_l361_361426


namespace eccentricity_of_hyperbola_is_2_l361_361715

noncomputable def hyperbola_eccentricity (a b : ℝ) (A B M O : ℝ × ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 then if h₁ : A.1^2 / a^2 - A.2^2 / b^2 = 1 ∧ B.1^2 / a^2 - B.2^2 / b^2 = 1
  ∧ (∃ m n : ℝ, B = (m, n) ∧ M = (-m, -n) ∧ A.1 ≠ m ∧ slope (A.1, A.2) (B.1, B.2) = 3 ∧ slope (A.1, A.2) (M.1, M.2) = 1)
  then let e := Real.sqrt (1 + b^2 / a^2) in if e = 2 then e else 0
  else 0
  else 0

theorem eccentricity_of_hyperbola_is_2 (a b : ℝ) (A B M O : ℝ × ℝ) (h : a > 0 ∧ b > 0)
  (h₁ : A.1^2 / a^2 - A.2^2 / b^2 = 1 ∧ B.1^2 / a^2 - B.2^2 / b^2 = 1)
  (hx₀ : A.1 ≠ B.1)
  (h₂ : ∃ m n : ℝ, B = (m, n) ∧ M = (-m, -n) ∧ slope (A.1, A.2) (B.1, B.2) = 3 ∧ slope (A.1, A.2) (M.1, M.2) = 1) :
  hyperbola_eccentricity a b A B M O = 2 :=
by
  sorry

end eccentricity_of_hyperbola_is_2_l361_361715


namespace correctFinishingOrder_l361_361398

-- Define the types of athletes
inductive Athlete
| A | B | C | D | E

-- Define the type for predictions
def Prediction := Athlete → Nat

-- Define the first prediction
def firstPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 1
    | Athlete.B => 2
    | Athlete.C => 3
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the second prediction
def secondPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 4
    | Athlete.C => 1
    | Athlete.D => 5
    | Athlete.E => 2

-- Define the actual finishing positions
noncomputable def actualFinishingPositions : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 2
    | Athlete.C => 1
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the correctness conditions for the predictions
def correctPositions (actual : Prediction) (prediction : Prediction) (k : Nat) : Prop :=
  (Set.card {x : Athlete | actual x = prediction x} = k)

-- The proof statement
theorem correctFinishingOrder :
  correctPositions actualFinishingPositions firstPrediction 3 ∧
  correctPositions actualFinishingPositions secondPrediction 2 :=
sorry

end correctFinishingOrder_l361_361398


namespace total_time_taken_l361_361954

theorem total_time_taken
  (speed_boat : ℝ)
  (speed_stream : ℝ)
  (distance : ℝ)
  (h_boat : speed_boat = 12)
  (h_stream : speed_stream = 5)
  (h_distance : distance = 325) :
  (distance / (speed_boat - speed_stream) + distance / (speed_boat + speed_stream)) = 65.55 :=
by
  sorry

end total_time_taken_l361_361954


namespace number_of_ways_to_choose_water_polo_team_l361_361888

theorem number_of_ways_to_choose_water_polo_team :
  let total_members := 15
  let team_size := 7
  let goalie_positions := 1
  let player_choices := team_size - goalie_positions
  ∃ (total_ways : ℕ), 
  total_ways = total_members * Nat.choose (total_members - 1) player_choices ∧ 
  total_ways = 45045 :=
by
  let total_members := 15
  let team_size := 7
  let goalie_positions := 1
  let player_choices := team_size - goalie_positions
  have total_ways : ℕ := total_members * Nat.choose (total_members - 1) player_choices
  use total_ways
  sorry

end number_of_ways_to_choose_water_polo_team_l361_361888


namespace circumcircle_radius_in_meters_l361_361258

theorem circumcircle_radius_in_meters (AB AC BC : ℝ) (hAB : AB = 13) (hAC : AC = 14) (hBC : BC = 15) (li_to_meters : ℝ) (hli_to_meters : li_to_meters = 500) : 
  let R_li := 8.125 in 
  R_li * li_to_meters = 4062.5 :=
by 
  sorry

end circumcircle_radius_in_meters_l361_361258


namespace sum_two_digit_primes_interchanged_l361_361156

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Prove the sum of all two-digit primes greater than 12 but less than 99 which remain prime
       when their digits are interchanged equals 418. -/
theorem sum_two_digit_primes_interchanged :
  let primes := {p : ℕ | is_prime p ∧ 12 < p ∧ p < 99 ∧ is_prime (((p % 10) * 10) + (p / 10))}
  ∑ p in primes, p = 418 := by sorry

end sum_two_digit_primes_interchanged_l361_361156


namespace distance_from_origin_to_point_l361_361811

theorem distance_from_origin_to_point : 
  ∀ (x y : ℝ), x = 8 → y = -15 → real.sqrt ((x-0)^2 + (y-0)^2) = 17 :=
by
  intros x y hx hy
  rw [hx, hy]
  norm_num
  rw [real.sqrt_eq_rpow]
  norm_num
  sorry

end distance_from_origin_to_point_l361_361811


namespace eq_condition_l361_361864

variables {a b c d k : ℝ}
variables {f : ℝ → ℝ}
variables {g : ℝ → ℝ}

-- Conditions
def condition_1 : Prop := k ≠ 0
def f_eq : f = λ x, a * x + b
def g_eq : g = λ x, k * (c * x + d)
def condition_2 : Prop := d * (1 - a) ≠ 0

-- Result
theorem eq_condition (h1 : condition_1) (h2 : f_eq) (h3 : g_eq) (h4 : condition_2) : 
  f ∘ g = g ∘ f ↔ (b * (1 - k * c) / (d * (1 - a)) = k) := 
by 
  sorry

end eq_condition_l361_361864


namespace investment_interest_min_l361_361227

theorem investment_interest_min (x y : ℝ) (hx : x + y = 25000) (hmax : x ≤ 11000) : 
  0.07 * x + 0.12 * y ≥ 2450 :=
by
  sorry

end investment_interest_min_l361_361227


namespace value_of_x_squared_minus_one_l361_361646

theorem value_of_x_squared_minus_one (x : ℤ) (h : 3^(x+1) + 3^(x+1) + 3^(x+1) = 243) : x^2 - 1 = 8 := 
by {
  sorry
}

end value_of_x_squared_minus_one_l361_361646


namespace find_t_l361_361763

-- Definitions of the vectors a, b, and c
def a : ℝ × ℝ := (real.sqrt 3, 1)
def b : ℝ × ℝ := (0, 1)
def c (t : ℝ) : ℝ × ℝ := (-real.sqrt 3, t)

-- The dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Definition of perpendicular
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Proof statement
theorem find_t (t : ℝ) (h : perpendicular (a.1 + 2 * b.1, a.2 + 2 * b.2) (c t)) : t = 1 :=
sorry

end find_t_l361_361763


namespace factors_of_12_valid_ratio_l361_361103

theorem factors_of_12 : 
  {x : ℕ | x ∣ 12} = {1, 2, 3, 4, 6, 12} :=
sorry

theorem valid_ratio :
  ∃ (a b c d : ℕ), a ∈ {1, 2, 3, 4, 6, 12} ∧ b ∈ {1, 2, 3, 4, 6, 12} ∧ c ∈ {1, 2, 3, 4, 6, 12} ∧ d ∈ {1, 2, 3, 4, 6, 12} ∧ a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 6 ∧ a * d = b * c :=
sorry


end factors_of_12_valid_ratio_l361_361103


namespace dihedral_angle_bisectors_intersect_at_single_point_l361_361403

/- 
 In an arbitrary tetrahedron, planes are drawn through each edge inside the tetrahedron, bisecting the corresponding dihedral angles.
 Prove that all six planes intersect at a single point.
-/
theorem dihedral_angle_bisectors_intersect_at_single_point
  (A B C D : Point)
  (tetrahedron : Tetrahedron A B C D)
  (pi1 pi2 pi3 pi4 pi5 pi6 : Plane)
  (h1 : bisects_dihedral_angle pi1 (edge A B) (face A B C) (face A B D))
  (h2 : bisects_dihedral_angle pi2 (edge A C) (face A C B) (face A C D))
  (h3 : bisects_dihedral_angle pi3 (edge A D) (face A D B) (face A D C))
  (h4 : bisects_dihedral_angle pi4 (edge B C) (face B C A) (face B C D))
  (h5 : bisects_dihedral_angle pi5 (edge B D) (face B D A) (face B D C))
  (h6 : bisects_dihedral_angle pi6 (edge C D) (face C D A) (face C D B)) :
  ∃ O : Point, π1 ∩ π2 ∩ π3 ∩ π4 ∩ π5 ∩ π6 = O :=
sorry

end dihedral_angle_bisectors_intersect_at_single_point_l361_361403


namespace log_eval_l361_361263

-- Lean statement
theorem log_eval : log 4 (64 * sqrt 4) = 7 / 2 := by
  sorry

end log_eval_l361_361263


namespace smallest_sum_of_relatively_prime_numbers_l361_361853

open Nat

/-- Leonhard has five cards. Each card has a nonnegative integer written on it,
and any two cards show relatively prime numbers. Compute the smallest possible value of the sum of
the numbers on Leonhard's cards. -/

theorem smallest_sum_of_relatively_prime_numbers :
  ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  (gcd a b = 1) ∧ (gcd a c = 1) ∧ (gcd a d = 1) ∧ (gcd a e = 1) ∧ 
  (gcd b c = 1) ∧ (gcd b d = 1) ∧ (gcd b e = 1) ∧ 
  (gcd c d = 1) ∧ (gcd c e = 1) ∧ (gcd d e = 1) ∧ 
  a + b + c + d + e = 4 :=
begin
  sorry
end

end smallest_sum_of_relatively_prime_numbers_l361_361853


namespace arithmetic_sequence_sum_19_l361_361862

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_19 (h1 : is_arithmetic_sequence a)
  (h2 : a 9 = 11) (h3 : a 11 = 9) (h4 : ∀ n, S n = n / 2 * (a 1 + a n)) :
  S 19 = 190 :=
sorry

end arithmetic_sequence_sum_19_l361_361862


namespace num_pairs_avg_six_l361_361087

open Finset

theorem num_pairs_avg_six : 
  (univ.filter (λ (ab : ℕ × ℕ), ab.fst + ab.snd = 12 ∧ ab.fst ≠ ab.snd)).card = 5 :=
by
  -- Assuming the universal set to be pairs of elements from {1, 2, ..., 11}
  -- Filtering pairs such that their sum is 12 and they are distinct
  let s := {1, 2, ..., 11}.product {1, 2, ..., 11}
  let valid_pairs := s.filter (λ (ab : ℕ × ℕ), ab.fst + ab.snd = 12 ∧ ab.fst ≠ ab.snd)
  exact valid_pairs.card = 5
  sorry

end num_pairs_avg_six_l361_361087


namespace area_of_rectangle_PQRS_l361_361406

-- Define the rectangle PQRS and its geometric properties
variables (P Q R S T U : Type)
variables (PQRS : Rectangle P Q R S)
variables (SU ST PQ PS : Segment)
variables (angle_S trisects : Angle)
variables (T_on_PQ U_on_PS : Point on Segment)
variables (QT PU : Real)
variables [fact (QT = 8)]
variables [fact (PU = 4)]

-- Define that angle S is trisected by SU and ST
axiom trisects_angle_S_by_SU_ST : trisects.angle S by (SU, ST)

-- Define area of rectangle PQRS
def area_PQRS : Real := PQ.length * PS.length

-- State the theorem we want to prove
theorem area_of_rectangle_PQRS : area_PQRS PQRS = 64 * Real.sqrt 3 :=
sorry

end area_of_rectangle_PQRS_l361_361406


namespace distance_from_origin_l361_361815

theorem distance_from_origin :
  let origin := (0, 0)
      pt := (8, -15)
  in Real.sqrt ((pt.1 - origin.1)^2 + (pt.2 - origin.2)^2) = 17 :=
by
  let origin := (0, 0)
  let pt := (8, -15)
  have h1 : pt.1 - origin.1 = 8 := by rfl
  have h2 : pt.2 - origin.2 = -15 := by rfl
  simp [h1, h2]
  -- sorry will be used to skip the actual proof steps and to ensure the code builds successfully
  sorry

end distance_from_origin_l361_361815


namespace sum_of_squares_of_roots_eq_zero_l361_361244

-- Given constants and problem definition
def polynomial : Polynomial ℝ :=
  Polynomial.X ^ 1009 + 100 * Polynomial.X ^ 1006 + 5 * Polynomial.X ^ 3 + 500

-- Main theorem to prove
theorem sum_of_squares_of_roots_eq_zero :
  (polynomial.roots.map (λ s, s ^ 2)).sum = 0 :=
sorry

end sum_of_squares_of_roots_eq_zero_l361_361244


namespace trains_cross_time_l361_361217

noncomputable def time_for_trains_to_cross (len1 len2 spd1 spd2 : ℝ) : ℝ :=
  let relative_speed := (spd1 + spd2) * (5 / 18)
  let total_length := len1 + len2
  total_length / relative_speed

theorem trains_cross_time :
  time_for_trains_to_cross 108 112 50 81.996 ≈ 5.997 :=
by
  sorry

end trains_cross_time_l361_361217


namespace distance_to_point_is_17_l361_361828

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem distance_to_point_is_17 :
  distance_from_origin 8 (-15) = 17 :=
by
  sorry

end distance_to_point_is_17_l361_361828


namespace radius_of_cylinder_is_correct_l361_361605

/-- 
  A right circular cylinder is inscribed in a right circular cone such that:
  - The diameter of the cylinder is equal to its height.
  - The cone has a diameter of 8.
  - The cone has an altitude of 10.
  - The axes of the cylinder and cone coincide.
  Prove that the radius of the cylinder is 20/9.
-/
theorem radius_of_cylinder_is_correct :
  ∀ (r : ℚ), 
    (2 * r = 8 - 2 * r ∧ 10 - 2 * r = (10 / 4) * r) → 
    r = 20 / 9 :=
by
  intro r
  intro h
  sorry

end radius_of_cylinder_is_correct_l361_361605


namespace john_jury_duty_days_l361_361033

-- Definitions based on the given conditions
variable (jury_selection_days : ℕ) (trial_multiple : ℕ) (deliberation_days : ℕ) (hours_per_day_deliberation : ℕ)
variable (total_days_jury_duty : ℕ)

-- Conditions
def condition1 : Prop := jury_selection_days = 2
def condition2 : Prop := trial_multiple = 4
def condition3 : Prop := deliberation_days = 6
def condition4 : Prop := hours_per_day_deliberation = 16
def correct_answer : Prop := total_days_jury_duty = 19

-- Total days calculation
def total_days_calc : ℕ :=
  let trial_days := jury_selection_days * trial_multiple
  let total_deliberation_hours := deliberation_days * 24
  let actual_deliberation_days := total_deliberation_hours / hours_per_day_deliberation
  jury_selection_days + trial_days + actual_deliberation_days

-- Statement we need to prove
theorem john_jury_duty_days : condition1 ∧ condition2 ∧ condition3 ∧ condition4 → total_days_calc = total_days_jury_duty :=
by {
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest1,
  cases h_rest1 with h3 h4,
  rw [condition1, condition2, condition3, condition4] at h1 h2 h3 h4,
  simp [h1, h2, h3, h4],
  sorry
} -- Proof omitted

end john_jury_duty_days_l361_361033


namespace sin_add_lt_cos_sin_sub_lt_cos_not_cos_add_lt_sin_not_cos_sub_lt_sin_l361_361290

-- Given: α and β are acute angles.
-- Prove that:
-- (1) sin(α + β) < cos α + cos β
-- (2) sin(α - β) < cos α + cos β
-- Disprove that:
-- (3) cos(α + β) < sin α + sin β
-- (4) cos(α - β) < sin α + sin β

variable {α β : ℝ}
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)

theorem sin_add_lt_cos (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  sin (α + β) < cos α + cos β :=
sorry

theorem sin_sub_lt_cos (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  sin (α - β) < cos α + cos β :=
sorry

theorem not_cos_add_lt_sin (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ¬ (cos (α + β) < sin α + sin β) :=
sorry

theorem not_cos_sub_lt_sin (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ¬ (cos (α - β) < sin α + sin β) :=
sorry

end sin_add_lt_cos_sin_sub_lt_cos_not_cos_add_lt_sin_not_cos_sub_lt_sin_l361_361290


namespace odd_products_fraction_l361_361789

theorem odd_products_fraction : 
  let factors := finset.range 16
  let odd_numbers := factors.filter (λ x, x % 2 = 1)
  let total_products := factors.card * factors.card
  let total_odd_products := (odd_numbers.card * odd_numbers.card)
  round_to_nearest_hundredth (total_odd_products / total_products) = 0.25 :=
by sorry

noncomputable def round_to_nearest_hundredth (x : ℚ) : ℚ :=
  ((x * 100).round / 100 : ℚ)

end odd_products_fraction_l361_361789


namespace rectangle_no_boundary_intersection_l361_361206

theorem rectangle_no_boundary_intersection (D : set ℝ) 
  (H : D = { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ b }) 
  (P : set (set ℝ)) 
  (HP : ∀ R ∈ P, ∃ a b c d, R = { p : ℝ × ℝ | a ≤ p.1 ∧ p.1 ≤ b ∧ c ≤ p.2 ∧ p.2 ≤ d })
  (partition : ∀ R ∈ P, ∀ l : ℝ, (∃ p : ℝ × ℝ, p ∈ D ∧ p.1 = l) → ∃ r ∈ P, ∃ q : ℝ × ℝ, q ∈ r ∧ q.1 = l):
  ∃ R ∈ P, ∀ p ∈ R, (p.1 ∈ {0, a} ∨ p.2 ∈ {0, b}) → false := 
sorry

end rectangle_no_boundary_intersection_l361_361206


namespace sequence_formula_l361_361952

theorem sequence_formula (a : ℕ → ℝ) : 
  (∀ n, a 1 + ∑ k in finset.range n, (a (k + 2)) / 2^k = 3^(n + 1)) →
  (∀ n, a n = if n = 1 then 9 else 6^n) :=
sorry

end sequence_formula_l361_361952


namespace distance_origin_to_point_l361_361797

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l361_361797


namespace cylinder_radius_in_cone_l361_361604

theorem cylinder_radius_in_cone :
  ∀ (r : ℚ), (2 * r = r) → (0 < r) → (∀ (h : ℚ), h = 2 * r → 
  (∀ (c_r : ℚ), c_r = 4 (c_r is radius of cone)  ∧ (h_c : ℚ), h_c = 10 (h_c is height of cone) ∧ 
  (10 - h) / r = h_c / c_r) → r = 20 / 9) :=
begin
  sorry,
end

end cylinder_radius_in_cone_l361_361604


namespace min_value_f_inequality_ln_l361_361707

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem min_value_f : ∃ x : ℝ, 0 < x ∧ f x = -1 / Real.exp 1 := 
sorry

theorem inequality_ln : ∀ x : ℝ, (0 < x ∧ x < ∞) → Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) := 
sorry

end min_value_f_inequality_ln_l361_361707


namespace find_n_divisible_by_35_l361_361695

-- Define the five-digit number for some digit n
def num (n : ℕ) : ℕ := 80000 + n * 1000 + 975

-- Define the conditions
def divisible_by_5 (d : ℕ) : Prop := d % 5 = 0
def divisible_by_7 (d : ℕ) : Prop := d % 7 = 0
def divisible_by_35 (d : ℕ) : Prop := divisible_by_5 d ∧ divisible_by_7 d

-- Statement of the problem for proving given conditions and the correct answer
theorem find_n_divisible_by_35 : ∃ (n : ℕ), (num n % 35 = 0) ∧ n = 6 := by
  sorry

end find_n_divisible_by_35_l361_361695


namespace range_of_m_l361_361742

noncomputable def y (m x : ℝ) := m * (1/4)^x - (1/2)^x + 1

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, y m x = 0) → (m ≤ 0 ∨ m = 1 / 4) := sorry

end range_of_m_l361_361742


namespace geom_sequence_analogous_l361_361323

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ m n, a m - a n = d * (m - n)

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r, ∀ m n, b m / b n = r ^ (m - n)

variables {a b : ℕ → ℝ} {m n : ℕ}
-- Given conditions
axiom am : m ≠ n ∧ m > 0 ∧ n > 0
axiom arith_seq : is_arithmetic_sequence a
axiom a_m_a : a m = a
axiom a_n_b : a n = b

axiom geo_seq : is_geometric_sequence b
axiom b_n_gt_0 : ∀ (n : ℕ), n > 0 → b n > 0
axiom b_m_a : b m = a
axiom b_n_b : b n = b

-- The conclusion to be proved using the given analogies
theorem geom_sequence_analogous :
  b (m + n) = (n - m) * (b n / a m) :=
sorry

end geom_sequence_analogous_l361_361323


namespace log_geometric_ratio_l361_361792

variable (a : ℕ → ℝ) (q : ℝ)

theorem log_geometric_ratio :
  (∀ n, a (n + 1) = a n * q) →
  (q = 2) →
  (∀ n, 0 < a n) →
  log 2 ((a 2 + a 3) / (a 0 + a 1)) = 2 :=
by
  intros h_geom_ratio h_q_pos h_a_pos
  -- Proof here
  sorry

end log_geometric_ratio_l361_361792


namespace min_isosceles_triangle_area_l361_361511

theorem min_isosceles_triangle_area 
  (x y n : ℕ)
  (h1 : 2 * x * y = 7 * n^2)
  (h2 : ∃ m k, m = n / 2 ∧ k = 2 * m) 
  (h3 : n % 3 = 0) : 
  x = 4 * n / 3 ∧ y = n / 3 ∧ 
  ∃ A, A = 21 / 4 := 
sorry

end min_isosceles_triangle_area_l361_361511


namespace total_shelves_used_l361_361612

theorem total_shelves_used (C B SC SB : ℕ) (CP BP : ℕ) (HC : C = 435) (HB : B = 523)
  (HSC : SC = 218) (HSB : SB = 304) (HCPS : CP = 17) (HBPS : BP = 22) :
  let remaining_coloring_books := C - SC,
      remaining_puzzle_books := B - SB,
      shelves_coloring_books := (remaining_coloring_books + CP - 1) / CP,
      shelves_puzzle_books := (remaining_puzzle_books + BP - 1) / BP
  in shelves_coloring_books + shelves_puzzle_books = 23 :=
by
  sorry

end total_shelves_used_l361_361612


namespace sqrt_expression_eq_three_l361_361959

theorem sqrt_expression_eq_three (h: (Real.sqrt 81) = 9) : Real.sqrt ((Real.sqrt 81 + Real.sqrt 81) / 2) = 3 :=
by 
  sorry

end sqrt_expression_eq_three_l361_361959


namespace quadruple_sequence_return_l361_361716

theorem quadruple_sequence_return (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (hreturn : ∃ (n : ℕ), iterate_quadruple (a, b, c, d) n = (a, b, c, d)) :
  a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 :=
sorry

def iterate_quadruple (quad : ℝ × ℝ × ℝ × ℝ) : ℕ → ℝ × ℝ × ℝ × ℝ
| 0     := quad
| (n+1) := let (a, b, c, d) := iterate_quadruple quad n in (a * b, b * c, c * d, d * a)

-- Helper definition for generating the sequences
noncomputable def generate_sequence (a b c d : ℝ) : ℕ → ℝ × ℝ × ℝ × ℝ
| 0     := (a, b, c, d)
| (n+1) := let (w, x, y, z) := generate_sequence n in (w * x, x * y, y * z, z * w)

end quadruple_sequence_return_l361_361716


namespace prime_divisor_congruence_l361_361443

theorem prime_divisor_congruence (p q : ℕ) (hp : Nat.Prime p) 
  (hq : Nat.Prime q) (hdiv : ∃ k : ℕ, k ≠ 0 ∧ q * k = p^p - 1) :
  q ≡ 1 [MOD p] :=
begin
  sorry,
end

end prime_divisor_congruence_l361_361443


namespace complementary_angles_l361_361728

theorem complementary_angles (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 :=
by 
  sorry

end complementary_angles_l361_361728


namespace closest_integer_to_a2013_l361_361531

noncomputable def sequence (n : ℕ) : ℝ :=
  nat.rec_on n 100 (λ k ak, ak + 1 / ak)

theorem closest_integer_to_a2013 : ∃ (z : ℤ), z = 118 ∧ abs (sequence 2013 - z) = min (abs (sequence 2013 - (z - 1))) (abs (sequence 2013 - (z + 1))) :=
begin
  sorry
end

end closest_integer_to_a2013_l361_361531


namespace value_of_f_neg_4_explicit_formula_l361_361932

def f (x : ℝ) : ℝ :=
  if h : x > 0 then log x / log (1/2)
  else if x = 0 then 0
  else log (-x) / log (1/2)

theorem value_of_f_neg_4 (x : ℝ) (hx : x = -4) : f x = -2 := by
  sorry

theorem explicit_formula (x : ℝ) :
  f x = 
  if x > 0 then log x / log (1/2)
  else if x = 0 then 0
  else log (-x) / log (1/2) := by
  sorry

end value_of_f_neg_4_explicit_formula_l361_361932


namespace find_angle_l361_361876

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

theorem find_angle (ahyp : ∥a∥ = 2) (bhyp : ∥b∥ = 1) (dot_hyp : ⟪a, a - b⟫ = 3) :
  real.angle a b = real.pi / 3 :=
by sorry

end find_angle_l361_361876


namespace probability_is_palindromic_div7_among_choices_l361_361200

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

noncomputable def is_5_digit_palindrome (n : ℕ) : Prop :=
  n >= 10000 ∧ n < 100000 ∧ is_palindrome n

noncomputable def probability_palindromic_div7 : ℚ :=
  let total_palindromes := 900 in
  let palindromic_div7 := (finset.filter (λ n, is_5_digit_palindrome n ∧ is_palindrome (n / 7) ∧ n % 7 = 0) (finset.Ico 10000 100000)).card in
  palindromic_div7 / total_palindromes

theorem probability_is_palindromic_div7_among_choices :
  probability_palindromic_div7 = 1/150 ∨
  probability_palindromic_div7 = 1/225 ∨
  probability_palindromic_div7 = 1/300 ∨
  probability_palindromic_div7 = 1/450 ∨
  probability_palindromic_div7 = 1/900 :=
sorry

end probability_is_palindromic_div7_among_choices_l361_361200


namespace engineer_sidorov_mistake_l361_361628

def expected_radius : ℝ := 0.5
def variance_radius : ℝ := 10 ^ (-4)
def predicted_weight : ℝ := 10000
def actual_area (radius : ℝ) : ℝ := π * radius^2
def disk_weight (radius : ℝ) : ℝ := 100 * actual_area(radius) / (π * 0.25)
def expected_disk_weight : ℝ := disk_weight expected_radius
def total_expected_weight : ℝ := 100 * expected_disk_weight
def sidorov_error (predicted_weight : ℝ) (total_expected_weight : ℝ) : ℝ := total_expected_weight - predicted_weight

theorem engineer_sidorov_mistake :
  sidorov_error predicted_weight total_expected_weight = 4 :=
by
  sorry

end engineer_sidorov_mistake_l361_361628


namespace line_intersection_reflected_ray_l361_361573

-- Definitions of the given equations for lines
def line_l1 (x y: ℝ) : Prop := x - 2 * y + 3 = 0
def line_l2 (x y: ℝ) : Prop := 2 * x + 3 * y - 8 = 0

-- Intersection point of l1 and l2
def intersection_point (x y: ℝ) : Prop :=
  line_l1 x y ∧ line_l2 x y

-- Distance from origin (0, 0)
def distance_from_origin (a b : ℝ) (d : ℝ) : Prop :=
  abs (a * 0 + b * 0 - d) / sqrt (a^2 + b^2) = 1

-- Symmetric point with respect to l1
def symmetric_point (M M' : ℝ × ℝ) : Prop :=
  ∃ a b : ℝ, M = (2, 5) ∧ M' = (a, b) ∧ (b - 5) = -2 * (a - 2) ∧ 
             ((a + 2)/2) - 2 * ((b + 5)/2) + 3 = 0

-- Reflected line equation passing through N
def line_through_points (M' N : ℝ × ℝ) : Prop :=
  ∃ a b c : ℝ, M' = (4, 1) ∧ N = (-2, 4) ∧
  a * (fst N) + b * (snd N) - c = 0 ∧ 
  a * (fst M') + b * (snd M') - c = 0

theorem line_intersection :
  ∃ (p : ℝ × ℝ),
  intersection_point p.1 p.2 ∧
  (distance_from_origin 1 0 1 ∨ distance_from_origin 3 (-4) 5) :=
sorry

theorem reflected_ray :
  symmetric_point (2, 5) (4, 1) ∧
  line_through_points (4,1) (-2,4) :=
sorry

end line_intersection_reflected_ray_l361_361573


namespace remainder_pow_2023_l361_361563

theorem remainder_pow_2023 (a b : ℕ) (h : b = 2023) : (3 ^ b) % 11 = 5 :=
by
  sorry

end remainder_pow_2023_l361_361563


namespace common_term_formula_and_sum_l361_361545

def b (n : ℕ) : ℕ := 4 * n - 2
def c (m : ℕ) : ℕ := 6 * m - 4

noncomputable def a (n : ℕ) : ℕ := 12 * n - 10

theorem common_term_formula_and_sum :
  (∀ n, ∃ m, b n = c m) →
  ∑ i in finset.range 16, a (i + 1) = 1472 := 
by
  intros,
  sorry

end common_term_formula_and_sum_l361_361545


namespace tangent_line_at_one_monotonicity_of_g_when_a_gt_zero_extreme_value_range_of_a_l361_361317

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * (Real.log x + 1)

theorem tangent_line_at_one (a : ℝ) : 
  let f' := λ x, Real.exp x - a / x in
  let tangent := λ x, (Real.exp 1 - a) * x in
  tangent = λ x, f' 1 * (x - 1) + f 1 a :=
by
  sorry

theorem monotonicity_of_g_when_a_gt_zero : 
  ∀ a > 0, ∀ x ∈ Ioo (1/2 : ℝ) 1, (Real.exp x - a / x) * x = x * Real.exp x - a > 0 :=
by
  sorry

theorem extreme_value_range_of_a (a : ℝ) :
  (f 1 a > 0) ∧ (exp (1 / 2) - 2 * a < 0) → (sqrt (Real.exp 1) / 2 < a ∧ a < Real.exp 1) :=
by
  sorry

end tangent_line_at_one_monotonicity_of_g_when_a_gt_zero_extreme_value_range_of_a_l361_361317


namespace inequality_abc_l361_361447

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1):
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by
  sorry

end inequality_abc_l361_361447


namespace correct_prediction_l361_361370

-- Declare the permutation type
def Permutation := (ℕ × ℕ × ℕ × ℕ × ℕ)

-- Declare the actual finish positions
def actual_positions : Permutation := (3, 2, 1, 4, 5)

-- Declare conditions
theorem correct_prediction 
  (P1 : Permutation) (P2 : Permutation) 
  (hP1 : P1 = (1, 2, 3, 4, 5))
  (hP2 : P2 = (1, 2, 3, 4, 5))
  (hP1_correct : ∃! i ∈ (1..5), (λ i, nth (P1.to_list) (i - 1) == nth (actual_positions.to_list))  = 3)
  (hP2_correct : ∃! i ∈ (1..5), (λ i, nth (P2.to_list) (i - 1) == nth (actual_positions.to_list))  = 2)
  : actual_positions = (3, 2, 1, 4, 5) 
:= sorry

end correct_prediction_l361_361370


namespace class_with_avg_40_students_l361_361123

theorem class_with_avg_40_students
  (x y : ℕ)
  (h : 40 * x + 60 * y = (380 * (x + y)) / 7) : x = 40 :=
sorry

end class_with_avg_40_students_l361_361123


namespace find_number_210_l361_361471

noncomputable def numbers := Fin 268 → ℤ

def sum_of_consecutive_20 (n : numbers) := ∀ i : Fin 268, (Finset.range 20).sum (λ j, n ⟨(i + j) % 268, sorry⟩) = 75

def specific_positions (n : numbers) :=
  n ⟨17, sorry⟩ = 3 ∧ n ⟨83, sorry⟩ = 4 ∧ n ⟨144, sorry⟩ = 9

theorem find_number_210 (n : numbers)
  (h_sum : sum_of_consecutive_20 n)
  (h_specific : specific_positions n) :
  n ⟨210, sorry⟩ = -1 :=
sorry

end find_number_210_l361_361471


namespace number_of_solutions_depends_on_a_l361_361654

theorem number_of_solutions_depends_on_a (a : ℝ) : 
  (∀ x : ℝ, 2^(3 * x) + 4 * a * 2^(2 * x) + a^2 * 2^x - 6 * a^3 = 0) → 
  (if a = 0 then 0 else if a > 0 then 1 else 2) = 
  (if a = 0 then 0 else if a > 0 then 1 else 2) :=
by 
  sorry

end number_of_solutions_depends_on_a_l361_361654


namespace length_of_PQ_l361_361842

open Real

def P := (2 : ℝ, π / 3)
def Q := (2 * sqrt 3 : ℝ, 5 * π / 6)

noncomputable def to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem length_of_PQ :
  let p_cart := to_cartesian 2 (π / 3)
  let q_cart := to_cartesian (2 * sqrt 3) (5 * π / 6)
  distance p_cart q_cart = 4 := by
  sorry

end length_of_PQ_l361_361842


namespace correct_statement_C_l361_361169

theorem correct_statement_C : ∀ y : ℝ, y^2 = 25 ↔ (y = 5 ∨ y = -5) := 
by sorry

end correct_statement_C_l361_361169


namespace find_actual_positions_l361_361391

namespace RunningCompetition

variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]

/-- The first prediction: $A$ finishes first, $B$ is second, $C$ is third, $D$ is fourth, $E$ is fifth. --/
def first_prediction := [A, B, C, D, E]

/-- The second prediction: $C$ finishes first, $E$ is second, $A$ is third, $B$ is fourth, $D$ is fifth. --/
def second_prediction := [C, E, A, B, D]

/-- The first prediction correctly predicted the positions of exactly three athletes. --/
def first_prediction_correct_count : ℕ := 3

/-- The second prediction correctly predicted the positions of exactly two athletes. --/
def second_prediction_correct_count : ℕ := 2

/-- Proves that the actual finishing order is $C, B, A, D, E$. --/
theorem find_actual_positions : 
  ∀ (order : list (Type)), 
  (order = [C, B, A, D, E]) →
  list.count order first_prediction_correct_count = 3 ∧ 
  list.count order second_prediction_correct_count = 2 :=
by
  sorry

end RunningCompetition

end find_actual_positions_l361_361391


namespace sqrt_mul_sqrt_l361_361990

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361990


namespace correct_prediction_l361_361368

-- Declare the permutation type
def Permutation := (ℕ × ℕ × ℕ × ℕ × ℕ)

-- Declare the actual finish positions
def actual_positions : Permutation := (3, 2, 1, 4, 5)

-- Declare conditions
theorem correct_prediction 
  (P1 : Permutation) (P2 : Permutation) 
  (hP1 : P1 = (1, 2, 3, 4, 5))
  (hP2 : P2 = (1, 2, 3, 4, 5))
  (hP1_correct : ∃! i ∈ (1..5), (λ i, nth (P1.to_list) (i - 1) == nth (actual_positions.to_list))  = 3)
  (hP2_correct : ∃! i ∈ (1..5), (λ i, nth (P2.to_list) (i - 1) == nth (actual_positions.to_list))  = 2)
  : actual_positions = (3, 2, 1, 4, 5) 
:= sorry

end correct_prediction_l361_361368


namespace max_elements_sequence_l361_361144

noncomputable def max_elements : ℕ := 
  let num_sequence : List ℕ := (41, 42, ..., 59)
  let squared_sequence : List ℕ := List.map (fun n => n^2) num_sequence
  if List.forall idx (List.init (List.length squared_sequence - 1)) (fun idx => 
      (squared_sequence[idx + 1] == squared_sequence[idx] + 200 * num_sequence[idx] + 100)) 
  then
    List.length num_sequence
  else 
    0

theorem max_elements_sequence : max_elements = 19 :=
by sorry

end max_elements_sequence_l361_361144


namespace max_length_OB_is_sqrt_2_l361_361974

noncomputable def max_length_OB (O A B : Point) (d : ℝ) :=
  dist A B = d ∧
  ∠ A O B = π / 4 ∧
  (∀ A B 
    (h1 : dist A B = d)
    (h2 : ∠ A O B = π / 4), 
     ∀ OB, OB ≤ dist O B → OB = dist O B → OB = sqrt 2)

theorem max_length_OB_is_sqrt_2 (O A B : Point) (d : ℝ):
  max_length_OB O A B 1 →
  (∃ OB, OB = sqrt 2) :=
by sorry

end max_length_OB_is_sqrt_2_l361_361974


namespace coverable_faces_l361_361943

theorem coverable_faces (a b c : ℕ) : (a % 3 = 0 ∧ b % 3 = 0) ∨ (b % 3 = 0 ∧ c % 3 = 0) ∨ (a % 3 = 0 ∧ c % 3 = 0) → 
(the three faces of a parallelepiped with dimensions a × b × c sharing a common vertex can be covered with three-cell strips without overlaps and gaps) :=
by sorry

end coverable_faces_l361_361943


namespace min_value_proof_l361_361044

noncomputable def min_value (t c : ℝ) :=
  (t^2 + c^2 - 2 * t * c + 2 * c^2) / 2

theorem min_value_proof (a b t c : ℝ) (h : a + b = t) :
  (a^2 + (b + c)^2) ≥ min_value t c :=
by
  sorry

end min_value_proof_l361_361044


namespace simplify_radicals_l361_361496

theorem simplify_radicals (x : ℝ) : 
  √(54 * x) * √(20 * x) * √(14 * x) = 12 * √(105 * x) := 
by 
  sorry

end simplify_radicals_l361_361496


namespace triangle_side_ratio_l361_361478

theorem triangle_side_ratio
  (α β γ : Real)
  (a b c p q r : Real)
  (h1 : (Real.tan α) / (Real.tan β) = p / q)
  (h2 : (Real.tan β) / (Real.tan γ) = q / r)
  (h3 : (Real.tan γ) / (Real.tan α) = r / p) :
  a^2 / b^2 / c^2 = (1/q + 1/r) / (1/r + 1/p) / (1/p + 1/q) := 
sorry

end triangle_side_ratio_l361_361478


namespace feeding_times_per_day_l361_361488

theorem feeding_times_per_day (p f d : ℕ) (h₁ : p = 7) (h₂ : f = 105) (h₃ : d = 5) : 
  (f / d) / p = 3 := by
  sorry

end feeding_times_per_day_l361_361488


namespace find_r_l361_361690

def polynomial (x : ℝ) : ℝ := 8*x^3 - 4*x^2 - 42*x + 45

theorem find_r (r : ℝ) (h1 : polynomial r = 0) (h2 : (derivative polynomial) r = 0) : r = 1.5 := 
sorry

end find_r_l361_361690


namespace correct_propositions_count_l361_361331

theorem correct_propositions_count :
  let original := ∀ x : ℝ, (x = 3) → (x^2 - 7 * x + 12 = 0)
  let converse := ∀ x : ℝ, (x^2 - 7 * x + 12 = 0) → (x = 3)
  let inverse := ∀ x : ℝ, (x ≠ 3) → (x^2 - 7 * x + 12 ≠ 0)
  let contrapositive := ∀ x : ℝ, (x^2 - 7 * x + 12 ≠ 0) → (x ≠ 3)
  (original = true) ∧
  (converse = false) ∧
  (inverse = false) ∧
  (contrapositive = true) →
  2 = 2 :=
by
  intro h
  exact h
sorry

end correct_propositions_count_l361_361331


namespace coeff_x5_in_binom_exp_l361_361007

theorem coeff_x5_in_binom_exp (n k : ℕ) (h_nk : n = 30 ∧ k = 5) : 
  (Nat.choose n k) = 65280 :=
by
  cases h_nk with
  | intro h_n h_k =>
    rw [h_n, h_k]
    decide

end coeff_x5_in_binom_exp_l361_361007


namespace dot_product_v1_v2_l361_361282

def v1 : ℝ × ℝ × ℝ := (4, -5, 2)
def v2 : ℝ × ℝ × ℝ := (-3, 3, -4)

theorem dot_product_v1_v2 : 
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = -35 := 
by
  sorry

end dot_product_v1_v2_l361_361282


namespace average_gas_mileage_round_trip_l361_361663

open BigOperators

def city_to_coast_distance : ℝ := 150
def coast_to_city_distance : ℝ := 150
def hybrid_car_mileage : ℝ := 40
def sedan_car_mileage : ℝ := 25

theorem average_gas_mileage_round_trip :
  let total_distance := city_to_coast_distance + coast_to_city_distance
  let hybrid_gas_used := city_to_coast_distance / hybrid_car_mileage
  let sedan_gas_used := coast_to_city_distance / sedan_car_mileage
  let total_gas_used := hybrid_gas_used + sedan_gas_used in
  total_distance / total_gas_used ≈ 30 :=
by
  let total_distance := city_to_coast_distance + coast_to_city_distance
  let hybrid_gas_used := city_to_coast_distance / hybrid_car_mileage
  let sedan_gas_used := coast_to_city_distance / sedan_car_mileage
  let total_gas_used := hybrid_gas_used + sedan_gas_used
  have : total_distance / total_gas_used ≈ 30 := by sorry
  exact this

end average_gas_mileage_round_trip_l361_361663


namespace max_sin_sum_of_triangle_l361_361871

theorem max_sin_sum_of_triangle (A B C : ℝ) (h : A + B + C = 180) :
  ⌊10 * (max (sin (3 * A) + sin (3 * B) + sin (3 * C)) 0)⌋ = 25 :=
sorry

end max_sin_sum_of_triangle_l361_361871


namespace john_trip_l361_361848

theorem john_trip (t : ℝ) (h : t ≥ 0) : 
  ∀ t : ℝ, 60 * t + 90 * ((7 / 2) - t) = 300 :=
by sorry

end john_trip_l361_361848


namespace digit_inequality_l361_361733

theorem digit_inequality (n : ℕ) (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ)
  (h_a_1 : count_digits n 1 = a_1)
  (h_a_2 : count_digits n 2 = a_2)
  (h_a_3 : count_digits n 3 = a_3)
  (h_a_4 : count_digits n 4 = a_4)
  (h_a_5 : count_digits n 5 = a_5)
  (h_a_6 : count_digits n 6 = a_6)
  (h_a_7 : count_digits n 7 = a_7)
  (h_a_8 : count_digits n 8 = a_8)
  (h_a_9 : count_digits n 9 = a_9) :
  2^a_1 * 3^a_2 * 4^a_3 * 5^a_4 * 6^a_5 * 7^a_6 * 8^a_7 * 9^a_8 * 10^a_9 ≤ n + 1 ∧ 
  ∀ r k : ℕ, 
    1 ≤ r ∧ r ≤ 9 ∧ n = r * 10^k + 10^k - 1 → 
    2^a_1 * 3^a_2 * 4^a_3 * 5^a_4 * 6^a_5 * 7^a_6 * 8^a_7 * 9^a_8 * 10^a_9 = n + 1 := 
by 
  sorry

end digit_inequality_l361_361733


namespace chord_length_square_l361_361241

noncomputable def square_of_chord_length : ℚ :=
let r1 := (4 : ℚ)
let r2 := (8 : ℚ)
let R := (12 : ℚ)
-- using the relationship and conditions from the problem
-- derived from geometry and tangency conditions
let chord_square := 4 * ((R ^ 2) - (4 * r2 / 3) ^ 2) in
chord_square

theorem chord_length_square :
  square_of_chord_length = 3584 / 9 := by
  sorry

end chord_length_square_l361_361241


namespace train_length_l361_361220

noncomputable def relative_speed (train_speed man_speed : ℝ) : ℝ :=
  train_speed - man_speed

noncomputable def speed_in_m_per_sec (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr * (5 / 18)

noncomputable def length_of_train (relative_speed_m_per_sec time_sec : ℝ) : ℝ :=
  relative_speed_m_per_sec * time_sec

theorem train_length :
  let train_speed := 63
      man_speed := 3
      time_to_cross := 53.99568034557235
      rel_speed := relative_speed train_speed man_speed
      rel_speed_m_per_sec := speed_in_m_per_sec rel_speed
  in
  length_of_train rel_speed_m_per_sec time_to_cross = 899.9280057595392 :=
by
  sorry

end train_length_l361_361220


namespace inversion_of_circle_l361_361479

theorem inversion_of_circle (O : Point) (S : Circle) :
  (S.passes_through O → S.inversion O is Line) ∧ 
  (¬ S.passes_through O → ∃ T : Circle, S.inversion O = T) := 
sorry

end inversion_of_circle_l361_361479


namespace correct_population_growth_pattern_statement_l361_361572

-- Definitions based on the conditions provided
def overall_population_growth_modern (world_population : ℕ) : Prop :=
  -- The overall pattern of population growth worldwide is already in the modern stage
  sorry

def transformation_synchronized (world_population : ℕ) : Prop :=
  -- The transformation of population growth patterns in countries or regions around the world is synchronized
  sorry

def developed_countries_transformed (world_population : ℕ) : Prop :=
  -- Developed countries have basically completed the transformation of population growth patterns
  sorry

def transformation_determined_by_population_size (world_population : ℕ) : Prop :=
  -- The process of transformation in population growth patterns is determined by the population size of each area
  sorry

-- The statement to be proven
theorem correct_population_growth_pattern_statement (world_population : ℕ) :
  developed_countries_transformed world_population := sorry

end correct_population_growth_pattern_statement_l361_361572


namespace distance_origin_to_point_l361_361800

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l361_361800


namespace repeating_decimal_as_fraction_l361_361267

-- Define the repeating decimal in question
def repeating_decimal : ℝ := 0.4 + (5 / 10) / (1 - (1 / 10))

-- Define the target fraction
def target_fraction : ℝ := 41 / 90

-- The statement to prove that the repeating decimal equals the fraction
theorem repeating_decimal_as_fraction : repeating_decimal = target_fraction := 
sorry

end repeating_decimal_as_fraction_l361_361267


namespace prove_positions_l361_361380

def athlete : Type := { A : ℕ, B : ℕ, C : ℕ, D : ℕ, E : ℕ }

def first_prediction (x : athlete) : athlete :=
  { A := 1, B := 2, C := 3, D := 4, E := 5 }

def second_prediction (x : athlete) : athlete :=
  { A := 3, B := 4, C := 1, D := 5, E := 2 }

def actual_positions : athlete :=
  { A := 3, B := 2, C := 1, D := 4, E := 5 }

theorem prove_positions :
  (λ x, first_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 3 ∧
  (λ x, second_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 2 ∧
  actual_positions.A = 3 ∧ actual_positions.B = 2 ∧ actual_positions.C = 1 ∧ actual_positions.D = 4 ∧ actual_positions.E = 5 := 
  sorry

end prove_positions_l361_361380


namespace acid_concentration_in_third_flask_l361_361137

theorem acid_concentration_in_third_flask 
    (w : ℚ) (W : ℚ) (hw : 10 / (10 + w) = 1 / 20) (hW : 20 / (20 + (W - w)) = 7 / 30) 
    (W_total : W = 256.43) : 
    30 / (30 + W) = 21 / 200 := 
by 
  sorry

end acid_concentration_in_third_flask_l361_361137


namespace num_valid_values_n_l361_361248

theorem num_valid_values_n :
  ∃ n : ℕ, (∃ a b c : ℕ,
    8 * a + 88 * b + 888 * c = 8880 ∧
    n = a + 2 * b + 3  * c) ∧
  (∃! k : ℕ, k = 119) :=
by sorry

end num_valid_values_n_l361_361248


namespace combined_is_distribution_function_l361_361483

variables {X₁ X₂ : Type*} {C₁ C₂ : ℝ} 
variables [dist₁ : distribution_function X₁] [dist₂ : distribution_function X₂]

noncomputable def combined_distribution_function (x : ℝ) : ℝ :=
  C₁ * dist₁.F x + C₂ * dist₂.F x

theorem combined_is_distribution_function 
  (hc₁ : 0 ≤ C₁) (hc₂ : 0 ≤ C₂) (hcsum : C₁ + C₂ = 1) :
  is_distribution_function (combined_distribution_function) :=
begin
  sorry -- proof goes here
end

end combined_is_distribution_function_l361_361483


namespace concentration_flask3_l361_361131

open Real

noncomputable def proof_problem : Prop :=
  ∃ (W : ℝ) (w : ℝ), 
    let flask1_acid := 10 in
    let flask2_acid := 20 in
    let flask3_acid := 30 in
    let flask1_total := flask1_acid + w in
    let flask2_total := flask2_acid + (W - w) in
    let flask3_total := flask3_acid + W in
    (flask1_acid / flask1_total = 1 / 20) ∧
    (flask2_acid / flask2_total = 7 / 30) ∧
    (flask3_acid / flask3_total = 21 / 200)

theorem concentration_flask3 : proof_problem :=
begin
  sorry
end

end concentration_flask3_l361_361131


namespace inequality_proof_l361_361450

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by
  sorry

end inequality_proof_l361_361450


namespace distance_from_origin_l361_361821

theorem distance_from_origin :
  let origin := (0, 0)
      pt := (8, -15)
  in Real.sqrt ((pt.1 - origin.1)^2 + (pt.2 - origin.2)^2) = 17 :=
by
  let origin := (0, 0)
  let pt := (8, -15)
  have h1 : pt.1 - origin.1 = 8 := by rfl
  have h2 : pt.2 - origin.2 = -15 := by rfl
  simp [h1, h2]
  -- sorry will be used to skip the actual proof steps and to ensure the code builds successfully
  sorry

end distance_from_origin_l361_361821


namespace equal_area_construction_possible_l361_361257

open EuclideanGeometry

noncomputable def midpoint (a b : Point ℝ) : Point ℝ :=
  ((a.x + b.x) / 2, (a.y + b.y) / 2)

noncomputable def arc_midpoint (a b : Point ℝ) (r : ℝ) : Point ℝ := sorry

theorem equal_area_construction_possible (A B C : Point ℝ) (r : ℝ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : AB ≥ AC):
  ∃ E : Point ℝ, ∃ F : Point ℝ, 
  midpoint E F = midpoint B C ∧
  area (FEB) = area (ABC) / 2 :=
begin
  sorry
end

end equal_area_construction_possible_l361_361257


namespace roots_of_polynomial_l361_361254

theorem roots_of_polynomial :
  (x^2 - 5 * x + 6) * (x - 1) * (x + 3) = 0 ↔ (x = -3 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by {
  sorry
}

end roots_of_polynomial_l361_361254


namespace volume_between_spheres_in_cone_l361_361852

theorem volume_between_spheres_in_cone (K : Cone) (s : Sphere) (r : ℝ) (S : Sphere) (R : ℝ) :
  (s.radius = r) → (S.radius = R) → (s.isInside K) → (S.isInside K) → (s.touches K) → (S.touches K) → (s.touches S) →
  volume_between_spheres s S K = (4 * π * (r^2) * (R^2)) / (3 * (R + r)) :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end volume_between_spheres_in_cone_l361_361852


namespace diff_of_squares_odd_divisible_by_8_l361_361903

theorem diff_of_squares_odd_divisible_by_8 (m n : ℤ) :
  ((2 * m + 1) ^ 2 - (2 * n + 1) ^ 2) % 8 = 0 :=
by 
  sorry

end diff_of_squares_odd_divisible_by_8_l361_361903


namespace tricias_age_is_5_l361_361146

open_locale classical

variables (T A Y E K R V S C B : ℕ)

theorem tricias_age_is_5
  (h1 : T = 1/3 * A)
  (h2 : A = 1/4 * Y)
  (h3 : Y = 2 * E)
  (h4 : K = 1/3 * E)
  (h5 : R = K + 10)
  (h6 : R = V - 2)
  (h7 : V = 22)
  (h8 : Y = S + 5)
  (h9 : S = A + 3)
  (h10 : C = 1/2 * (V + A))
  (h11 : B = T + V) :
  T = 5 :=
begin
  have hV : V = 22, from h7,
  have hR : R = V - 2, from h6,
  rw hV at hR,
  have hR' : R = 20, by linarith,
  have hK : K = 1/3 * E, from h4,
  have hR'' : R = K + 10, from h5,
  rw hR' at hR'',
  have hK' : K = 10, by linarith,
  rw hK' at hK,
  have hE : E = 30, by linarith,
  have hY : Y = 2 * E, from h3,
  rw hE at hY,
  have hY' : Y = 60, by linarith,
  have hA : A = 1/4 * Y, from h2,
  rw hY' at hA,
  have hA' : A = 15, by linarith,
  have hT : T = 1/3 * A, from h1,
  rw hA' at hT,
  have hT' : T = 5, by linarith,
  exact hT',
end

end tricias_age_is_5_l361_361146


namespace total_days_on_jury_duty_l361_361029

-- Definitions based on conditions
def jurySelectionDays := 2
def juryDeliberationDays := 6
def deliberationHoursPerDay := 16
def trialDurationMultiplier := 4

-- Calculate the total number of hours spent in deliberation
def totalDeliberationHours := juryDeliberationDays * 24

-- Calculate the number of days spent in deliberation based on hours per day
def deliberationDays := totalDeliberationHours / deliberationHoursPerDay

-- Calculate the trial days based on trial duration multiplier
def trialDays := jurySelectionDays * trialDurationMultiplier

-- Calculate the total days on jury duty
def totalJuryDutyDays := jurySelectionDays + trialDays + deliberationDays

theorem total_days_on_jury_duty : totalJuryDutyDays = 19 := by
  sorry

end total_days_on_jury_duty_l361_361029


namespace sufficient_not_necessary_l361_361157

theorem sufficient_not_necessary (a b : ℝ) :
  (a^2 + b^2 = 0 → ab = 0) ∧ (ab = 0 → ¬(a^2 + b^2 = 0)) := 
by
  have h1 : (a^2 + b^2 = 0 → ab = 0) := sorry
  have h2 : (ab = 0 → ¬(a^2 + b^2 = 0)) := sorry
  exact ⟨h1, h2⟩

end sufficient_not_necessary_l361_361157


namespace concentration_flask3_l361_361130

open Real

noncomputable def proof_problem : Prop :=
  ∃ (W : ℝ) (w : ℝ), 
    let flask1_acid := 10 in
    let flask2_acid := 20 in
    let flask3_acid := 30 in
    let flask1_total := flask1_acid + w in
    let flask2_total := flask2_acid + (W - w) in
    let flask3_total := flask3_acid + W in
    (flask1_acid / flask1_total = 1 / 20) ∧
    (flask2_acid / flask2_total = 7 / 30) ∧
    (flask3_acid / flask3_total = 21 / 200)

theorem concentration_flask3 : proof_problem :=
begin
  sorry
end

end concentration_flask3_l361_361130


namespace quadratic_solution_l361_361692

theorem quadratic_solution (m : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_roots : ∀ x, 2 * x^2 + 4 * m * x + m = 0 ↔ x = x₁ ∨ x = x₂) 
  (h_sum_squares : x₁^2 + x₂^2 = 3 / 16) :
  m = -1 / 8 :=
by
  sorry

end quadratic_solution_l361_361692


namespace tetrahderon_parallelogram_l361_361083

theorem tetrahderon_parallelogram (A B C D O1 O2 : Point) 
  (hA: Midpoint A D O1) (hB: Midpoint B C O2) 
  (ha: ∀ p : Point, p.line_through A ∥ o) 
  (hb: ∀ p : Point, p.line_through B ∥ o) 
  (hc: ∀ p : Point, p.line_through C ∥ o) 
  (hd: ∀ p : Point, p.line_through D ∥ o) :
  ∀ (P: Plane), 
  let A' := P.intersect_line $ A.line_through o
  let B' := P.intersect_line $ B.line_through o
  let C' := P.intersect_line $ C.line_through o
  let D' := P.intersect_line $ D.line_through o
  is_parallelogram A' B' C' D' :=
sorry

end tetrahderon_parallelogram_l361_361083


namespace complex_number_identity_l361_361936

theorem complex_number_identity (z : ℂ) (h : z * (2 + 1 * complex.i) = 1 + 3 * complex.i) : 
  z = (1 : ℂ) + (1 : ℂ) * complex.i :=
by
  sorry

end complex_number_identity_l361_361936


namespace reduce_markers_to_two_l361_361120

theorem reduce_markers_to_two (n : ℕ) : 
  (∃ seq : List ℕ, valid_sequence seq n ∧ seq.length = 2) ↔ (n - 1) % 3 ≠ 0 := by
  sorry

def valid_sequence (seq : List ℕ) (n : ℕ) : Prop :=
  -- Define the conditions that seq is a valid sequence of operations
  -- starting from n markers all with their white sides up
  sorry

end reduce_markers_to_two_l361_361120


namespace average_annual_percentage_decrease_l361_361231

theorem average_annual_percentage_decrease (P2018 P2020 : ℝ) (x : ℝ) 
  (h_initial : P2018 = 20000)
  (h_final : P2020 = 16200) :
  P2018 * (1 - x)^2 = P2020 :=
by
  sorry

end average_annual_percentage_decrease_l361_361231


namespace unique_sequence_137_l361_361547

theorem unique_sequence_137 :
  ∃ (a : Fin 137 → ℕ), StrictMono a ∧ ∑ i, 2 ^ (a i) = (2^289 + 1) / (2^17 + 1) :=
by
  sorry

end unique_sequence_137_l361_361547


namespace adults_had_meal_l361_361598

theorem adults_had_meal (A : ℕ) : 
  ∀ (adults : ℕ) (children : ℕ) (max_adults : ℕ) (max_children : ℕ) (remaining_children : ℕ),
  adults = 55 → children = 70 → max_adults = 70 → max_children = 90 → remaining_children = 81 →
  (max_adults - A) * max_children = max_adults * remaining_children →
  A = 7 :=
by
  intros adults children max_adults max_children remaining_children
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  have h : (70 - A) * 90 = 70 * 81 := h6
  sorry

end adults_had_meal_l361_361598


namespace points_scored_by_others_l361_361096

-- Define the conditions as hypothesis
variables (P_total P_Jessie : ℕ)
  (H1 : P_total = 311)
  (H2 : P_Jessie = 41)
  (H3 : ∀ P_Lisa P_Devin: ℕ, P_Lisa = P_Jessie ∧ P_Devin = P_Jessie)

-- Define what we need to prove
theorem points_scored_by_others (P_others : ℕ) :
  P_total = 311 → P_Jessie = 41 → 
  (∀ P_Lisa P_Devin: ℕ, P_Lisa = P_Jessie ∧ P_Devin = P_Jessie) → 
  P_others = 188 :=
by
  sorry

end points_scored_by_others_l361_361096


namespace distance_from_origin_to_point_l361_361810

theorem distance_from_origin_to_point : 
  ∀ (x y : ℝ), x = 8 → y = -15 → real.sqrt ((x-0)^2 + (y-0)^2) = 17 :=
by
  intros x y hx hy
  rw [hx, hy]
  norm_num
  rw [real.sqrt_eq_rpow]
  norm_num
  sorry

end distance_from_origin_to_point_l361_361810


namespace initial_paint_l361_361683

variable (total_needed : ℕ) (paint_bought : ℕ) (still_needed : ℕ)

theorem initial_paint (h_total_needed : total_needed = 70)
                      (h_paint_bought : paint_bought = 23)
                      (h_still_needed : still_needed = 11) : 
                      ∃ x : ℕ, x = 36 :=
by
  sorry

end initial_paint_l361_361683


namespace johns_revenue_l361_361849

def price_novel : ℕ := 10
def price_comic : ℕ := 5
def price_biography : ℕ := 15

def initial_stock : ℕ := 1200

-- Monday sales and returns
def monday_novels_sold : ℕ := 30
def monday_comics_sold : ℕ := 20
def monday_biographies_sold : ℕ := 25
def monday_comics_returned : ℕ := 5
def monday_novel_returned : ℕ := 1

-- Tuesday sales (20% discount)
def tuesday_novels_sold : ℕ := 20
def tuesday_comics_sold : ℕ := 10
def tuesday_biographies_sold : ℕ := 20
def tuesday_discount : ℚ := 0.8

-- Wednesday sales and returns
def wednesday_novels_sold : ℕ := 30
def wednesday_comics_sold : ℕ := 20
def wednesday_biographies_sold : ℕ := 14
def wednesday_novels_returned : ℕ := 5
def wednesday_biographies_returned : ℕ := 3

-- Thursday sales (10% discount)
def thursday_novels_sold : ℕ := 40
def thursday_comics_sold : ℕ := 25
def thursday_biographies_sold : ℕ := 13
def thursday_discount : ℚ := 0.9

-- Friday sales and returns
def friday_novels_sold : ℕ := 55
def friday_comics_sold : ℕ := 40
def friday_biographies_sold : ℕ := 40
def friday_comics_returned : ℕ := 5
def friday_novels_returned : ℕ := 2
def friday_biographies_returned : ℕ := 3

theorem johns_revenue :
  let monday_revenue :=  (monday_novels_sold * price_novel + monday_comics_sold * price_comic + monday_biographies_sold * price_biography
                        - monday_comics_returned * price_comic - monday_novel_returned * price_novel : ℤ) in
  let tuesday_revenue :=  (tuesday_novels_sold * price_novel * tuesday_discount + tuesday_comics_sold * price_comic * tuesday_discount + tuesday_biographies_sold * price_biography * tuesday_discount : ℚ) in
  let wednesday_revenue :=  (wednesday_novels_sold * price_novel + wednesday_comics_sold * price_comic + wednesday_biographies_sold * price_biography
                            - wednesday_novels_returned * price_novel - wednesday_biographies_returned * price_biography : ℤ) in
  let thursday_revenue :=  (thursday_novels_sold * price_novel * thursday_discount + thursday_comics_sold * price_comic * thursday_discount + thursday_biographies_sold * price_biography * thursday_discount : ℚ) in
  let friday_revenue :=  (friday_novels_sold * price_novel + friday_comics_sold * price_comic + friday_biographies_sold * price_biography
                         - friday_comics_returned * price_comic - friday_novels_returned * price_novel - friday_biographies_returned * price_biography : ℤ) in
  int.to_nat (monday_revenue + wednesday_revenue + friday_revenue) + (tuesday_revenue + thursday_revenue : ℚ).to_nat = 3603 := by sorry

end johns_revenue_l361_361849


namespace area_of_triangle_ABC_with_B_45_degrees_area_of_triangle_ABC_with_c_sqrt3_b_l361_361844

open Real

-- Definitions
def is_triangle (A B C: ℝ) := A + B + C = π
def side_opposite_to_angle (a b c A B C : ℝ) := is_triangle A B C ∧
  (a^2 + b^2 - 2 * a * b * cos A = c^2)

-- Given conditions
def given_conditions (A B c: ℝ) := sin A + sqrt 3 * cos A = 2 ∧ A = π / 6 ∧ B = π / 4 ∧ c = sqrt 3

-- 1. Prove that, given a = 2 and B = 45 degrees, the area is sqrt(3) + 1
theorem area_of_triangle_ABC_with_B_45_degrees :
  ∀ (a b c A B C : ℝ), a = 2 → given_conditions A B c →
    side_opposite_to_angle a b c A B C →
    (1 / 2) * a * b * sin C = sqrt 3 + 1 :=
begin
  intros a b c A B C h_a h_cond h_triangle,
  sorry,
end

-- 2. Prove that, given a = 2 and c = sqrt(3)b, the area is sqrt(3)
theorem area_of_triangle_ABC_with_c_sqrt3_b :
  ∀ (a b c A B C : ℝ), a = 2 → given_conditions A B c →
    side_opposite_to_angle a b c A B C →
    (1 / 2) * b * c * sin A = sqrt 3 :=
begin
  intros a b c A B C h_a h_cond h_triangle,
  sorry,
end

end area_of_triangle_ABC_with_B_45_degrees_area_of_triangle_ABC_with_c_sqrt3_b_l361_361844


namespace complex_conjugates_norm_l361_361439

theorem complex_conjugates_norm (α β : ℂ) (h1 : α.conj = β) (h2 : abs (α - β) = 2 * real.sqrt 3)
(h3 : is_real (α / (β ^ 2))) : abs α = 2 :=
sorry

end complex_conjugates_norm_l361_361439


namespace quadratic_has_two_distinct_roots_l361_361712

theorem quadratic_has_two_distinct_roots (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  (b^2 - 4 * a * c) > 0 :=
by {
  sorry
}

end quadratic_has_two_distinct_roots_l361_361712


namespace intersection_A_complement_B_eq_minus_three_to_zero_l361_361755

-- Define the set A
def A : Set ℝ := { x : ℝ | x^2 + x - 6 ≤ 0 }

-- Define the set B
def B : Set ℝ := { y : ℝ | ∃ x : ℝ, y = Real.sqrt x ∧ 0 ≤ x ∧ x ≤ 4 }

-- Define the complement of B
def C_RB : Set ℝ := { y : ℝ | ¬ (y ∈ B) }

-- The proof problem
theorem intersection_A_complement_B_eq_minus_three_to_zero :
  (A ∩ C_RB) = { x : ℝ | -3 ≤ x ∧ x < 0 } :=
by
  sorry

end intersection_A_complement_B_eq_minus_three_to_zero_l361_361755


namespace find_line_equation_l361_361341

-- Define points P and Q
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P (a b : ℝ) : Point := ⟨a, b⟩
def Q (a b : ℝ) : Point := ⟨b - 1, a + 1⟩

-- Define the condition for the points to be symmetric about line l
def symm_about_line (P Q : Point) (l : Point → Prop) : Prop :=
  l P ∧ l Q ∧ (Q.x - P.x) * (Q.y - P.y) = -1 -- simplifying symmetry about perpendicular bisector

-- Define the line equation form
def line_eq (x y k b : ℝ) := x - y + b = k

-- Main theorem statement
theorem find_line_equation (a b : ℝ) (h : a ≠ b - 1) :
  symm_about_line (P a b) (Q a b) (λ M : Point, line_eq M.x M.y 0 1) := 
sorry

end find_line_equation_l361_361341


namespace v_17_in_terms_of_b_l361_361649

noncomputable def sequence (n : ℕ) (b : ℝ) : ℝ :=
  if h : n = 1 then b
  else
    let v : ℕ → ℝ
    | 1 => b
    | (n + 1) => -1 / (2 * v n + 1)
    in v n

theorem v_17_in_terms_of_b (b : ℝ) (hb : 0 < b) : sequence 17 b = 1 / (2 * b - 1) :=
begin
  sorry
end

end v_17_in_terms_of_b_l361_361649


namespace sequence_sum_first_4_eq_64_l361_361249

-- Definition of the sequence based on initial conditions
def sequence (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 8 - sequence 1
  else if n = 3 then 24 - (sequence 1 + sequence 2)
  else 2 * n^2 + 2 * n - 4

-- Sum of the first 4 terms of the sequence
def sum_first_4_terms : ℕ :=
  (sequence 1) + (sequence 2) + (sequence 3) + (sequence 4)

-- Statement to prove the sum of the first 4 terms is 64
theorem sequence_sum_first_4_eq_64 : sum_first_4_terms = 64 :=
by sorry

end sequence_sum_first_4_eq_64_l361_361249


namespace tourist_spends_more_time_AB_than_BA_l361_361614

variable {v1 v2 s : ℝ}
variable (h_v1_ne_v2 : v1 ≠ v2)

noncomputable def T_AB : ℝ := s * (1 / (2 * v1) + 1 / (2 * v2))
noncomputable def T_BA : ℝ := s / (2 * (v1 + v2))

theorem tourist_spends_more_time_AB_than_BA :
  T_AB v1 v2 s > T_BA v1 v2 s :=
by
  -- Insert proof here
  sorry

end tourist_spends_more_time_AB_than_BA_l361_361614


namespace intersection_of_A_and_B_l361_361721

def A : Set (ℝ × ℝ) := {p | p.snd = 3 * p.fst - 2}
def B : Set (ℝ × ℝ) := {p | p.snd = p.fst ^ 2}

theorem intersection_of_A_and_B :
  {p : ℝ × ℝ | p ∈ A ∧ p ∈ B} = {(1, 1), (2, 4)} :=
by
  sorry

end intersection_of_A_and_B_l361_361721


namespace dancing_pairs_at_least_n_l361_361537

theorem dancing_pairs_at_least_n (n : ℕ) (a b : Fin 2n → ℤ) (h_a : ∀ i, a i = 1 ∨ a i = -1) 
  (h_b : ∀ i, b i = 1 ∨ b i = -1) (h_sum : (∑ i, a i) + (∑ i, b i) = 0) : 
  ∃ k ∈ Fin (2n), (∑ i, if a (i + k) = b i then 1 else 0) ≥ n :=
by {
  sorry
}

end dancing_pairs_at_least_n_l361_361537


namespace general_formula_for_a_sum_of_first_n_terms_of_b_l361_361533

variable (n : ℕ)

def a (n : ℕ) : ℤ := 13 - 3 * n
def b (n : ℕ) : ℚ := 1 / ((a n).toRat * (a (n + 1)).toRat)
def T (n : ℕ) : ℚ := ∑ i in Finset.range n, b i

theorem general_formula_for_a :
  a n = 13 - 3 * n :=
sorry

theorem sum_of_first_n_terms_of_b :
  T n = n / (10 * (10 - 3 * n)) :=
sorry

end general_formula_for_a_sum_of_first_n_terms_of_b_l361_361533


namespace find_actual_positions_l361_361393

namespace RunningCompetition

variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]

/-- The first prediction: $A$ finishes first, $B$ is second, $C$ is third, $D$ is fourth, $E$ is fifth. --/
def first_prediction := [A, B, C, D, E]

/-- The second prediction: $C$ finishes first, $E$ is second, $A$ is third, $B$ is fourth, $D$ is fifth. --/
def second_prediction := [C, E, A, B, D]

/-- The first prediction correctly predicted the positions of exactly three athletes. --/
def first_prediction_correct_count : ℕ := 3

/-- The second prediction correctly predicted the positions of exactly two athletes. --/
def second_prediction_correct_count : ℕ := 2

/-- Proves that the actual finishing order is $C, B, A, D, E$. --/
theorem find_actual_positions : 
  ∀ (order : list (Type)), 
  (order = [C, B, A, D, E]) →
  list.count order first_prediction_correct_count = 3 ∧ 
  list.count order second_prediction_correct_count = 2 :=
by
  sorry

end RunningCompetition

end find_actual_positions_l361_361393


namespace matrix_problem_l361_361289

variable {n : ℕ}

theorem matrix_problem
  (h1 : n > 2)
  (A B C D : Matrix (Fin n) (Fin n) ℝ)
  (h2 : A.mul C - B.mul D = 1)
  (h3 : A.mul D + B.mul C = 0) :
  (C.mul A - D.mul B = 1 ∧ D.mul A + C.mul B = 0) ∧
  (Matrix.det (A.mul C) ≥ 0 ∧ (-1)^n * Matrix.det (B.mul D) ≥ 0) :=
by
  sorry

end matrix_problem_l361_361289


namespace distance_from_origin_to_point_l361_361806

-- Define the specific points in the coordinate system
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (8, -15)

-- The distance formula in Euclidean space
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem statement
theorem distance_from_origin_to_point : distance origin point = 17 := 
by
  sorry

end distance_from_origin_to_point_l361_361806


namespace sqrt_sub_add_simplification_l361_361641

theorem sqrt_sub_add_simplification : sqrt 18 - sqrt 8 + sqrt 2 = 2 * sqrt 2 := 
by sorry

end sqrt_sub_add_simplification_l361_361641


namespace percentage_second_year_students_approx_l361_361838

def second_year_students_numeric_methods : ℕ := 230
def second_year_students_auto_control : ℕ := 423
def second_year_students_both : ℕ := 134
def total_faculty_students : ℕ := 653
def total_second_year_students : ℕ :=
  second_year_students_numeric_methods +
  second_year_students_auto_control -
  second_year_students_both

theorem percentage_second_year_students_approx :
  (total_second_year_students : ℝ) / (total_faculty_students : ℝ) * 100 ≈ 79.48 :=
  by sorry

end percentage_second_year_students_approx_l361_361838


namespace no_consistent_cube_edge_labeling_natural_consistent_cube_edge_labeling_integer_l361_361185

def cube_edge_labeling_natural (f : Fin 12 → ℕ) : Prop :=
  ∀ v : Fin 8, ∑ (e : Fin 12) in (cube_vertex_edges v), f e = (∑ i in (range 12), i + 1) / 8

theorem no_consistent_cube_edge_labeling_natural :
  ¬ ∃ f : Fin 12 → ℕ, cube_edge_labeling_natural f := 
sorry

def cube_edge_labeling_integer (f : Fin 12 → ℤ) : Prop :=
  ∀ v : Fin 8, ∑ (e : Fin 12) in (cube_vertex_edges v), f e = 0

theorem consistent_cube_edge_labeling_integer :
  ∃ f : Fin 12 → ℤ, cube_edge_labeling_integer f := 
sorry

end no_consistent_cube_edge_labeling_natural_consistent_cube_edge_labeling_integer_l361_361185


namespace inequality_proof_l361_361451

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by
  sorry

end inequality_proof_l361_361451


namespace intersection_PQ_l361_361860

def P := {x : ℝ | x < 1}
def Q := {x : ℝ | x^2 < 4}
def PQ_intersection := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_PQ : P ∩ Q = PQ_intersection := by
  sorry

end intersection_PQ_l361_361860


namespace translation_4_units_upwards_l361_361890

theorem translation_4_units_upwards (M N : ℝ × ℝ) (hx : M.1 = N.1) (hy_diff : N.2 - M.2 = 4) :
  N = (M.1, M.2 + 4) :=
by
  sorry

end translation_4_units_upwards_l361_361890


namespace exists_infinitely_many_m_l361_361078

theorem exists_infinitely_many_m (k : ℕ) (hk : 0 < k) : 
  ∃ᶠ m in at_top, 3 ^ k ∣ m ^ 3 + 10 :=
sorry

end exists_infinitely_many_m_l361_361078


namespace valid_permutations_count_l361_361440

theorem valid_permutations_count :
  (∃ (a : Fin 6 → Fin 6), (∀ i : Fin 6, i ∈ Finset.univ → a i ∈ Finset.univ) ∧ (∀ i : Fin 6, ↑(a i) ∈ {1, 2, 3, 4, 5, 6}) ∧ (∀ i : Fin 4, (a i).val + (a (i + 1)).val + (a (i + 2)).val % 3 ≠ 0)) ->
  ∃! p : list (Fin 6), (p.perm ⟨0, Finset.card_eq 6⟩) (Finset.univ : Finset (Fin 6)).val ∧
  (∀ (i : Fin 4), ↑(p.nth_le i i.is_lt + p.nth_le (⟨i.val + 1 % 6, ModK.mul_lt i i.val.pos⟩) +
  p.nth_le (⟨i.val + 2 % 6, ModK.mul_lt i i.val.pos⟩)) % 3 ≠ 0) ∧
  list.countp (λ x, ↑x.val % 3 = 0) p = 2 ∧
  list.countp (λ x, ↑x.val % 3 = 1) p = 2 ∧
  list.countp (λ x, ↑x.val % 3 = 2) p = 2 :=
begin
  sorry
end

end valid_permutations_count_l361_361440


namespace fibonacci_concatenation_base_l361_361079

-- Define Fibonacci sequence in Lean
def fibonacci : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

-- The main theorem statement
theorem fibonacci_concatenation_base (k : ℕ) (hk : k > 0) :
  ∃ (b : ℕ), ∃ (triples : list (ℕ × ℕ × ℕ)),
    (triples.length = k ∧
     (∀ (u v w : ℕ), (u, v, w) ∈ triples → 
                      u ∈ range(1, ⊤) ∧ v ∈ range(1, ⊤) ∧ w ∈ range(1, ⊤) ∧
                      is_fibonacci (fibonacci u) ∧ 
                      is_fibonacci (fibonacci v) ∧ 
                      is_fibonacci (fibonacci w)) ∧
     ∃ (F : ℕ), is_fibonacci (concat_triples_in_base b triples) ) :=
sorry

end fibonacci_concatenation_base_l361_361079


namespace total_resources_l361_361213

def base3_to_dec (n : String) : Nat :=
  n.foldl (λacc d, acc * 3 + (d.toNat - '0'.toNat)) 0

def crystal_base3 := "2120"
def rare_metals_base3 := "2102"
def alien_tech_base3 := "102"

def crystal := base3_to_dec crystal_base3
def rare_metals := base3_to_dec rare_metals_base3
def alien_tech := base3_to_dec alien_tech_base3

theorem total_resources : (crystal + rare_metals + alien_tech) = 145 := by
  sorry

end total_resources_l361_361213


namespace black_lambs_correct_l361_361664

-- Define the total number of lambs
def total_lambs : ℕ := 6048

-- Define the number of white lambs
def white_lambs : ℕ := 193

-- Define the number of black lambs
def black_lambs : ℕ := total_lambs - white_lambs

-- The goal is to prove that the number of black lambs is 5855
theorem black_lambs_correct : black_lambs = 5855 := by
  sorry

end black_lambs_correct_l361_361664


namespace intersection_at_one_point_l361_361301

theorem intersection_at_one_point (m : ℝ) :
  (∃ x : ℝ, (m - 4) * x^2 - 2 * m * x - m - 6 = 0 ∧
            ∀ x' : ℝ, (m - 4) * x'^2 - 2 * m * x' - m - 6 = 0 → x' = x) ↔
  m = -4 ∨ m = 3 ∨ m = 4 := 
by
  sorry

end intersection_at_one_point_l361_361301


namespace find_lambda_l361_361729

variable {R : Type*} [Field R]
variables (a b : R) (lambda : R)
variables (A B C : R → R)

def OA : R := a
def OB : R := lambda * b
def OC : R := 2 * a + b

def collinear (A B C : R → R) : Prop :=
  ∃ μ : R, ∀ t : R, B t = A t + μ * (C t - A t)

theorem find_lambda (h : collinear OA OB OC) : lambda = -1 := by
  sorry

end find_lambda_l361_361729


namespace lollipops_left_for_becky_l361_361232
-- Import the Mathlib library

-- Define the conditions as given in the problem
def lemon_lollipops : ℕ := 75
def peppermint_lollipops : ℕ := 210
def watermelon_lollipops : ℕ := 6
def marshmallow_lollipops : ℕ := 504
def friends : ℕ := 13

-- Total number of lollipops
def total_lollipops : ℕ := lemon_lollipops + peppermint_lollipops + watermelon_lollipops + marshmallow_lollipops

-- Statement to prove that the remainder after distributing the total lollipops among friends is 2
theorem lollipops_left_for_becky : total_lollipops % friends = 2 := by
  -- Proof goes here
  sorry

end lollipops_left_for_becky_l361_361232


namespace proposition_2_proposition_4_l361_361621

variables {α : Type*} [linear_order α] {f : α → ℝ}

theorem proposition_2 (hf : ∀ x y, x < y → f x > f y) :
  ∀ a b, f a = 0 ∧ f b = 0 → a = b :=
by sorry

theorem proposition_4 (hf_odd : ∀ x, f (-x) = -f x) (a : α) (h1 : f a = 1) :
  ∃ b, f b = -1 :=
by sorry

end proposition_2_proposition_4_l361_361621


namespace count_odd_integers_with_2_l361_361766

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (b : ℕ), n = b * 10^(nat.log10 n) + d

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_in_range (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 1001

theorem count_odd_integers_with_2 : 
  ∃ (N : ℕ), 
    N = 76 ∧ 
    N = (nat.count (λ n, is_in_range n ∧ is_odd n ∧ contains_digit n 2) (list.range 1002)) :=
sorry

end count_odd_integers_with_2_l361_361766


namespace find_s_for_g_neg1_zero_l361_361441

def g (x s : ℝ) : ℝ := 3 * x^4 + x^3 - 2 * x^2 - 4 * x + s

theorem find_s_for_g_neg1_zero (s : ℝ) : g (-1) s = 0 ↔ s = -4 := by
  sorry

end find_s_for_g_neg1_zero_l361_361441


namespace miracle_tree_fruit_count_l361_361187

theorem miracle_tree_fruit_count :
  ∃ (apples oranges pears : ℕ), 
  apples + oranges + pears = 30 ∧
  apples = 6 ∧ oranges = 9 ∧ pears = 15 := by
  sorry

end miracle_tree_fruit_count_l361_361187


namespace slope_of_equal_area_line_l361_361552

theorem slope_of_equal_area_line (
  (a : ℝ) (b : ℝ) : 
  let c1 := (10, 80): ℝ × ℝ,
  let c2 := (13, 64): ℝ × ℝ,
  let c3 := (15, 72): ℝ × ℝ,
  let r : ℝ := 4,
  let line_pass : (13, 64): ℝ × ℝ := True in
    ∃ m: ℝ,
      (∃ y_intercept: ℝ, line_pass = (3 * m - 3 * m = m)) ∧
      |m| = 24 / 5 :=
begin
  sorry
end

end slope_of_equal_area_line_l361_361552


namespace bc_length_l361_361891

/-- Given: Constructs points A, B, C on a circle ω (BC as diameter),
    extends AB to B' and AC to C' such that B'C' is parallel to BC
    and tangent to ω at D.
    Provided B'D = 4 and C'D = 6.
    Task: Prove that BC equals 24/5. -/
theorem bc_length
  (A B C : Point)
  (ω : Circle)
  (BC_diameter : is_diameter B C ω)
  (B' C' D : Point)
  (hB' : extension A B B')
  (hC' : extension A C C')
  (h_parallel : parallel B'C' BC)
  (h_tangent : tangent B'C' ω D)
  (BD_length : length_segment B' D = 4)
  (CD_length : length_segment C' D = 6) :
  length_segment B C = 24 / 5 :=
sorry

end bc_length_l361_361891


namespace distance_from_origin_to_point_l361_361812

theorem distance_from_origin_to_point : 
  ∀ (x y : ℝ), x = 8 → y = -15 → real.sqrt ((x-0)^2 + (y-0)^2) = 17 :=
by
  intros x y hx hy
  rw [hx, hy]
  norm_num
  rw [real.sqrt_eq_rpow]
  norm_num
  sorry

end distance_from_origin_to_point_l361_361812


namespace john_jury_duty_days_l361_361032

-- Definitions based on the given conditions
variable (jury_selection_days : ℕ) (trial_multiple : ℕ) (deliberation_days : ℕ) (hours_per_day_deliberation : ℕ)
variable (total_days_jury_duty : ℕ)

-- Conditions
def condition1 : Prop := jury_selection_days = 2
def condition2 : Prop := trial_multiple = 4
def condition3 : Prop := deliberation_days = 6
def condition4 : Prop := hours_per_day_deliberation = 16
def correct_answer : Prop := total_days_jury_duty = 19

-- Total days calculation
def total_days_calc : ℕ :=
  let trial_days := jury_selection_days * trial_multiple
  let total_deliberation_hours := deliberation_days * 24
  let actual_deliberation_days := total_deliberation_hours / hours_per_day_deliberation
  jury_selection_days + trial_days + actual_deliberation_days

-- Statement we need to prove
theorem john_jury_duty_days : condition1 ∧ condition2 ∧ condition3 ∧ condition4 → total_days_calc = total_days_jury_duty :=
by {
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest1,
  cases h_rest1 with h3 h4,
  rw [condition1, condition2, condition3, condition4] at h1 h2 h3 h4,
  simp [h1, h2, h3, h4],
  sorry
} -- Proof omitted

end john_jury_duty_days_l361_361032


namespace square_root_of_25_is_5_and_minus_5_l361_361161

theorem square_root_of_25_is_5_and_minus_5 : ∃ y : ℝ, y^2 = 25 ∧ (y = 5 ∨ y = -5) :=
by
  have h1 : 5^2 = 25 := by norm_num
  have h2 : (-5)^2 = 25 := by norm_num
  use 5
  use -5
  split
  · exact h1
  · exact h2

end square_root_of_25_is_5_and_minus_5_l361_361161


namespace part1_l361_361186

theorem part1 (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 ≠ b^2) :
  (a^2 + a * b + b^2) / (a + b) - (a^2 - a * b + b^2) / (a - b) + (2 * b^2 - b^2 + a^2) / (a^2 - b^2) = 1 := 
sorry

end part1_l361_361186


namespace part1_proof_part2_proof_l361_361312

open Real

-- Definitions for the conditions
variables (x y z : ℝ)
variable (h₁ : 0 < x)
variable (h₂ : 0 < y)
variable (h₃ : 0 < z)

-- Part 1
theorem part1_proof : (1 / x + 1 / y ≥ 4 / (x + y)) :=
by sorry

-- Part 2
theorem part2_proof : (1 / x + 1 / y + 1 / z ≥ 2 / (x + y) + 2 / (y + z) + 2 / (z + x)) :=
by sorry

end part1_proof_part2_proof_l361_361312


namespace regression_analysis_l361_361601

theorem regression_analysis :
  let n := 10
  let x_bar := 8
  let y_bar := 2
  let sum_x := ∑ i in range 10, x_i
  let sum_y := ∑ i in range 10, y_i
  let sum_xy := ∑ i in range 10, x_i * y_i
  let sum_x2 := ∑ i in range 10, x_i ^ 2
  (sum_x = 80) →
  (sum_y = 20) →
  (sum_xy = 184) →
  (sum_x2 = 720) →
  let b := (sum_xy - n * x_bar * y_bar) / (sum_x2 - n * x_bar ^ 2)
  let a := y_bar - b * x_bar
  b = 0.3 ∧ a = -0.4 ∧
  ∀ (income : ℝ), let savings := b * income + a in
                   income = 7 → savings = 1.7 := by sorry

end regression_analysis_l361_361601


namespace sakshi_days_l361_361899

theorem sakshi_days (Sakshi_efficiency Tanya_efficiency : ℝ) (Sakshi_days Tanya_days : ℝ) (h_efficiency : Tanya_efficiency = 1.25 * Sakshi_efficiency) (h_days : Tanya_days = 8) : Sakshi_days = 10 :=
by
  sorry

end sakshi_days_l361_361899


namespace mrs_sheridan_fish_distribution_l361_361466

theorem mrs_sheridan_fish_distribution :
  let initial_fish := 125
  let additional_fish := 250
  let total_fish := initial_fish + additional_fish
  let small_aquarium_capacity := 150
  let fish_in_small_aquarium := small_aquarium_capacity
  let fish_in_large_aquarium := total_fish - fish_in_small_aquarium
  fish_in_large_aquarium = 225 :=
by {
  let initial_fish := 125
  let additional_fish := 250
  let total_fish := initial_fish + additional_fish
  let small_aquarium_capacity := 150
  let fish_in_small_aquarium := small_aquarium_capacity
  let fish_in_large_aquarium := total_fish - fish_in_small_aquarium

  have : fish_in_large_aquarium = 225 := by sorry
  exact this
}

end mrs_sheridan_fish_distribution_l361_361466


namespace max_K_possible_l361_361681

noncomputable def f (a : List ℤ) (x : ℕ) : ℤ :=
  a.sum (λ m, m * x ^ m)

theorem max_K_possible (n : ℕ) (a : Fin n → ℤ) (h1 : 2 ≤ n) (h2 : a 0 = 1) (h3 : ∀ x ∈ List.range (n - 1), f (List.ofFn a) x = 0) : 
  ∃ K : ℕ, K = n - 1 :=
 sorry

end max_K_possible_l361_361681


namespace sum_of_six_numbers_l361_361216

theorem sum_of_six_numbers:
  ∃ (A B C D E F : ℕ), 
    A > B ∧ B > C ∧ C > D ∧ D > E ∧ E > F ∧
    E > F ∧ C > F ∧ D > F ∧ A + B + C + D + E + F = 141 := 
sorry

end sum_of_six_numbers_l361_361216


namespace exists_frac_part_sum_eq_one_frac_part_sum_not_rational_l361_361178

-- Define the fractional part function
def frac_part (x : ℝ) : ℝ :=
  x - Real.floor x

-- Problem (a): Provide an example of such a positive \( a \) such that \( \{a\} + \left\{ \frac{1}{a} \right\} = 1 \)
theorem exists_frac_part_sum_eq_one : ∃ a : ℝ, 0 < a ∧ frac_part a + frac_part (1/a) = 1 := sorry

-- Problem (b): Can such \( a \) be a rational number?
theorem frac_part_sum_not_rational (a : ℝ) (h : 0 < a) (ha : frac_part a + frac_part (1/a) = 1) : ¬ ∃ p q : ℤ, q ≠ 0 ∧ a = p / q := sorry

end exists_frac_part_sum_eq_one_frac_part_sum_not_rational_l361_361178


namespace number_of_zeros_of_f_l361_361743

def f (x : ℝ) : ℝ := 2 * x - 3 * x

theorem number_of_zeros_of_f :
  ∃ (n : ℕ), n = 2 ∧ (∀ x, f x = 0 → x ∈ {x | f x = 0}) :=
by {
  sorry
}

end number_of_zeros_of_f_l361_361743


namespace number_of_people_l361_361505

theorem number_of_people (total_bowls : ℕ) (bowls_per_person : ℚ) : total_bowls = 55 ∧ bowls_per_person = 1 + 1/2 + 1/3 → total_bowls / bowls_per_person = 30 :=
by
  sorry

end number_of_people_l361_361505


namespace feeding_times_per_day_l361_361486

theorem feeding_times_per_day (p f d : ℕ) (h₁ : p = 7) (h₂ : f = 105) (h₃ : d = 5) : 
  (f / d) / p = 3 := by
  sorry

end feeding_times_per_day_l361_361486


namespace tangent_line_at_1_l361_361929

-- Define the function f.
def f (x : ℝ) : ℝ := x^2 + 1/x

-- Define the point of tangency.
def point_of_tangency : ℝ × ℝ := (1, f 1)

-- Define the derivative of the function f.
def f_prime (x : ℝ) : ℝ := 2 * x - 1/x^2

-- Define the slope of the tangent line at x = 1.
def slope_at_1 : ℝ := f_prime 1

-- Define the tangent line equation in point-slope form.
def tangent_line (x y : ℝ) : Prop := y = slope_at_1 * (x - 1) + f 1

theorem tangent_line_at_1 :
  ∀ x y : ℝ, tangent_line x y ↔ x - y + 1 = 0 :=
sorry

end tangent_line_at_1_l361_361929


namespace solve_for_y_l361_361094

noncomputable def f (y : ℝ) := real.cbrt (30 * y + real.cbrt (30 * y + 19))

theorem solve_for_y : f 228 = 19 :=
by {
  -- Define f(y) as provided in the condition
  have h1 : f 228 = real.cbrt (30 * 228 + real.cbrt (30 * 228 + 19)) := rfl,
  -- Substitute y = 228
  rw [h1],
  -- Simplify the inner cubic root
  have h2 : real.cbrt (30 * 228 + 19) = 19,
  {
    -- Calculate the inside value
    calc
      30 * 228 + 19 = 6840 + 19 : by norm_num
      ... = 6859 : by norm_num
      ... = 19^3 : by norm_num,
    -- Apply the cubic root identity
    exact real.cbrt_eq_iff_mul_eq_cube.mp h2,
  },
  -- Substitute back to outer cubic root
  rw [h2],
  -- Finally simplify
  exact real.cbrt_eq_iff_mul_eq_cube.mp h2,
}

end solve_for_y_l361_361094


namespace quadratic_roots_square_of_other_l361_361697

theorem quadratic_roots_square_of_other :
  ∀ (p : ℝ), (∃ (a b : ℝ), a ≠ b ∧ (a = b^2 ∨ b = a^2) ∧ (∃ (q : polynomial ℝ), q = polynomial.X^2 - polynomial.C p * polynomial.X + polynomial.C p ∧ q.roots = {a, b})) →
  p = 2 + Real.sqrt 5 ∨ p = 2 - Real.sqrt 5 :=
by
  sorry

end quadratic_roots_square_of_other_l361_361697


namespace margo_age_in_three_years_l361_361630

theorem margo_age_in_three_years (benjie_age : ℕ) (benjie_older : ℕ) (margo_age : ℕ) :
  benjie_age = 6 →
  benjie_older = 5 →
  margo_age = benjie_age - benjie_older →
  margo_age + 3 = 4 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  have h4 : margo_age = 1 := by linarith
  rw [h4]
  linarith

end margo_age_in_three_years_l361_361630


namespace current_ratio_of_employees_l361_361350

-- Definitions for the number of current male employees and the ratio if 3 more men are hired
variables (M : ℕ) (F : ℕ)
variables (hM : M = 189)
variables (ratio_hired : (M + 3) / F = 8 / 9)

-- Conclusion we want to prove
theorem current_ratio_of_employees (M F : ℕ) (hM : M = 189) (ratio_hired : (M + 3) / F = 8 / 9) : 
  M / F = 7 / 8 :=
sorry

end current_ratio_of_employees_l361_361350


namespace fixed_point_l361_361682

variable (p : ℝ)

def f (x : ℝ) : ℝ := 9 * x^2 + p * x - 5 * p

theorem fixed_point : ∀ c d : ℝ, (∀ p : ℝ, f p c = d) → (c = 5 ∧ d = 225) :=
by
  intro c d h
  -- This is a placeholder for the proof
  sorry

end fixed_point_l361_361682


namespace remainder_a55_div_45_l361_361046

def a_n (n : ℕ) : ℕ :=
  String.toNat (String.join (List.map (fun x => x.repr) (List.range (n + 1))))

theorem remainder_a55_div_45 : a_n 55 % 45 = 10 :=
by
  sorry

end remainder_a55_div_45_l361_361046


namespace building_height_l361_361569

theorem building_height (shadow_length_building : ℝ) (pole_height : ℝ) (shadow_length_pole : ℝ) 
  (h_shadow_length_building : shadow_length_building = 20)
  (h_pole_height : pole_height = 2)
  (h_shadow_length_pole : shadow_length_pole = 3) : 
  ∃ building_height : ℝ, building_height = 40 / 3 :=
by
  exists (40 / 3)
  sorry

end building_height_l361_361569


namespace probability_prod_div4_l361_361506

open Set

def distinctIntegersBetween : Set ℤ := {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}

noncomputable def numPairs : ℤ := (Finset.card distinctIntegersBetween).choose 2

noncomputable def numPairsDivisibleByFour : ℤ := 
  let evens := {6, 8, 10, 12, 14, 16, 18} 
  have h : evens ⊆ distinctIntegersBetween := by {
    simp only [subset_univ, true_and, mem_univ]
  }
  Finset.card evens).choose 2

theorem probability_prod_div4 : 
  numPairsDivisibleByFour.toRat / numPairs.toRat = 33 / 78 :=
  sorry

end probability_prod_div4_l361_361506


namespace triangle_bisector_theorem_l361_361937

theorem triangle_bisector_theorem (A B C L M : Point) (R : ℝ)
  (hL : L = internal_bisector_intersection A B C)
  (hM : M = external_bisector_intersection A B C)
  (hCL_CM : dist C L = dist C M)
  (circumradius : radius_circumscribed_circle A B C = R) :
  dist A C ^ 2 + dist B C ^ 2 = 4 * R ^ 2 :=
sorry

end triangle_bisector_theorem_l361_361937


namespace dot_product_of_v1_and_v2_l361_361281

-- We define the two vectors
def v1 : ℝ × ℝ × ℝ := (4, -5, 2)
def v2 : ℝ × ℝ × ℝ := (-3, 3, -4)

-- We state the theorem which says that the dot product of v1 and v2 is -35
theorem dot_product_of_v1_and_v2 : (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3) = -35 :=
by
  -- Proof will go here, for now we use sorry to skip it
  sorry

end dot_product_of_v1_and_v2_l361_361281


namespace find_p_l361_361839

open Real 

def point := (ℝ × ℝ)

def Q : point := (0, 15)
def A : point := (3, 15)
def B : point := (15, 0)
def C (p : ℝ) : point := (0, p)

def area_A (p : ℝ) : ℝ := 135 - (3 / 2) * (15 - p) - (15 / 2) * p

theorem find_p (p : ℝ) (h : area_A p = 35) : p = 77.5 / 6 :=
by
  sorry

end find_p_l361_361839


namespace part1a_part1b_part2_l361_361458

noncomputable section
open Set Real

axiom R : Set ℝ
def A : Set ℝ := { x | abs (x - 2) ≤ 1 }
def B (a : ℝ) : Set ℝ := { x | x ^ 2 - a < 0 }
def C (a : ℝ) : Prop := B a ⊆ compl A

theorem part1a (a: ℝ) (h: a = 4): A ∩ B a = Icc 1 (real.sqrt a) := sorry
theorem part1b (a: ℝ) (h: a = 4): A ∪ B a = Ioo (-real.sqrt a) 3 := sorry
theorem part2 (a: ℝ) (h: C a): 0 ≤ a ∧ a ≤ 1 := sorry

end part1a_part1b_part2_l361_361458


namespace expected_value_of_ξ_l361_361324

noncomputable def ξ : ℕ → ℝ := λ n, if n = 3 then 1/2 else 0  -- Placeholder definition

theorem expected_value_of_ξ :
  ∀ (ξ : ℕ → ℝ), ξ = (λ n, if n = 3 then 1/2 else 0) →
  E(ξ) = 3 * (1/2) :=
by 
  sorry

end expected_value_of_ξ_l361_361324


namespace quadratic_has_two_distinct_roots_l361_361711

theorem quadratic_has_two_distinct_roots (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  (b^2 - 4 * a * c) > 0 :=
by {
  sorry
}

end quadratic_has_two_distinct_roots_l361_361711


namespace actual_positions_correct_l361_361358

def athletes := {A, B, C, D, E}
def p1 : athletes → ℕ
| A := 1
| B := 2
| C := 3
| D := 4
| E := 5

def p2 : athletes → ℕ
| C := 1
| E := 2
| A := 3
| B := 4
| D := 5

def actual_positions : athletes → ℕ
| A := 3
| B := 2
| C := 1
| D := 4
| E := 5

theorem actual_positions_correct :
  (∑ a in athletes, ite (actual_positions a = p1 a) 1 0 = 3) ∧ 
  (∑ a in athletes, ite (actual_positions a = p2 a) 1 0 = 2) :=
by
  sorry

end actual_positions_correct_l361_361358


namespace part1_part2_l361_361746

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a * x - a ^ 2 + 1) / (x ^ 2 + 1)

theorem part1 (x : ℝ) (hx : x = 2) (ha : 1 = 1) : 
  let y := f 1 x in 6 * x + 25 * y - 32 = 0 := 
sorry

theorem part2 (a x : ℝ) (ha : a ≠ 0) : 
  ∃ intervals : list (set ℝ), 
    if a < 0 then 
      intervals = [set.Iio a, set.Ioi ((-1:ℝ) / a)]
    else 
      intervals = [set.Ioo ((-1:ℝ) / a) a] :=
sorry

end part1_part2_l361_361746


namespace speed_conversion_l361_361666

theorem speed_conversion (speed_mps : ℝ) (conversion_factor : ℝ) (result_kmph : ℝ) :
  speed_mps = 17.5014 → conversion_factor = 3.6 → result_kmph = speed_mps * conversion_factor →
  result_kmph = 63.00504 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end speed_conversion_l361_361666


namespace area_of_triangle_AKB_l361_361242

theorem area_of_triangle_AKB
  (r R : ℝ)
  (h1 : r > 0)
  (h2 : R > 0) :
  let S := (2 * r * R * Real.sqrt (r * R)) / (r + R)
  in S = (2 * r * R * Real.sqrt (r * R)) / (r + R) :=
by
  sorry

end area_of_triangle_AKB_l361_361242


namespace Kylie_coins_left_l361_361034

-- Definitions based on given conditions
def piggyBank := 30
def brother := 26
def father := 2 * brother
def sofa := 15
def totalCoins := piggyBank + brother + father + sofa
def coinsGivenToLaura := totalCoins / 2
def coinsLeft := totalCoins - coinsGivenToLaura

-- Theorem statement
theorem Kylie_coins_left : coinsLeft = 62 := by sorry

end Kylie_coins_left_l361_361034


namespace log_exponent_sum_l361_361772

noncomputable def log4 : ℝ := Real.log 4
noncomputable def log25 : ℝ := Real.log 25

theorem log_exponent_sum (c d : ℝ) (hc : c = log4) (hd : d = log25) : 5^(c/d) + 2^(d/c) = 7 :=
by
  rw [hc, hd]
  -- The proof steps would go here
  sorry

end log_exponent_sum_l361_361772


namespace Vasya_can_guarantee_15_points_l361_361474

/-
  Problem Statement:
  Given a deck of 36 cards (4 suits with 9 cards each),
  Petya selects 18 cards for himself and gives the remaining 18 cards to Vasya.
  Players take turns laying down cards, with Petya starting.
  Vasya scores a point if his card matches the suit or rank of the previous card played by Petya.
  Prove that Vasya can guarantee a minimum of 15 points regardless of Petya's card distribution.
-/

def card : Type := (Σ (suit : Fin 4), Fin 9)

def Petya_cards (c : card → Prop) : Prop :=
  ∃ (p : Finset card), p.cardinality = 18 ∧ (∀ (x : card), x ∈ p ↔ c x)

def Vasya_guaranteed_points (p : Finset card) : ℕ :=
  sorry  -- Introduce the function definition that counts points considering the play strategy.

theorem Vasya_can_guarantee_15_points :
  ∀ (c : card → Prop), Petya_cards c → Vasya_guaranteed_points (get_vasya_cards c) ≥ 15 :=
sorry

end Vasya_can_guarantee_15_points_l361_361474


namespace betty_total_blue_and_green_beads_l361_361233

theorem betty_total_blue_and_green_beads (r b g : ℕ) (h1 : 5 * b = 3 * r) (h2 : 5 * g = 2 * r) (h3 : r = 50) : b + g = 50 :=
by
  sorry

end betty_total_blue_and_green_beads_l361_361233


namespace sum_of_positive_k_l361_361520

theorem sum_of_positive_k : ∑ k in {23, 10, 5, 2}, k = 40 :=
by
  simp only [Finset.mem_insert, Finset.mem_singleton, finset_sum_add_distrib, Finset.sum_singleton]
  norm_num

end sum_of_positive_k_l361_361520


namespace beth_sold_coins_l361_361631

theorem beth_sold_coins :
  let initial_coins := 125
  let gift_coins := 35
  let total_coins := initial_coins + gift_coins
  let sold_coins := total_coins / 2
  sold_coins = 80 :=
by
  sorry

end beth_sold_coins_l361_361631


namespace range_of_m_l361_361319

open Real

noncomputable def f (x m : ℝ) : ℝ := sqrt (2 * log (x + 1) + x - m)
def g (y : ℝ) : ℝ := 2 * log (y + 1) + y - y^2

theorem range_of_m:
  (∃ (y₀ : ℝ), cos y₀ ∧ (f ∘ f) (cos y₀) = cos y₀) ↔ (∀ (m : ℝ), 0 ≤ m ∧ m ≤ 2 * log 2) :=
sorry

end range_of_m_l361_361319


namespace first_negative_term_arithmetic_seq_l361_361409

theorem first_negative_term_arithmetic_seq : 
  let a_n := λ (n : ℕ), 55 - 4 * n in
  ∃ n : ℕ, a_n n < 0 ∧ (∀ m : ℕ, m < n → a_n m ≥ 0) :=
by
  sorry

end first_negative_term_arithmetic_seq_l361_361409


namespace lambda_values_l361_361768

variable (x λ : ℝ)

theorem lambda_values (h : ∀ x : ℝ, 2 * x^2 - λ * x + 1 ≥ 0) : λ = 1 ∨ λ = 2 * Real.sqrt 2 :=
by
  sorry

end lambda_values_l361_361768


namespace tetrahedron_volume_l361_361407

noncomputable def volumeOfTetrahedron (PQ PQR_area PQS_area angle_between : ℝ) : ℝ :=
  let sin_45 := Real.sin (Real.pi / 4)
  let volume := (1 / 3) * PQR_area * (PQS_area / PQR_area * PQ * sin_45)
  volume

theorem tetrahedron_volume
  (h1 : PQ = 5)
  (h2 : PQR_area = 20)
  (h3 : PQS_area = 18)
  (h4 : angle_between = Real.pi / 4) : volumeOfTetrahedron PQ PQR_area PQS_area angle_between = 15 * Real.sqrt 2 :=
by
  rw [volumeOfTetrahedron]
  sorry

-- Definitions for clarity
variable (PQ : ℝ)  -- length of edge PQ
variable (PQR_area : ℝ) -- area of face PQR
variable (PQS_area : ℝ) -- area of face PQS
variable (angle_between : ℝ) -- angle between planes PQR and PQS

/--
The volume of tetrahedron PQRS is 15√2 cm³ given the conditions:
- PQ = 5 cm
- Area of PQR = 20 cm²
- Area of PQS = 18 cm²
- Angle between planes PQR and PQS is 45 degrees
--/
example : volumeOfTetrahedron PQ PQR_area PQS_area angle_between = 15 * Real.sqrt 2 :=
by
  -- Assume given conditions as hypotheses
  have h1 : PQ = 5 := rfl
  have h2 : PQR_area = 20 := rfl
  have h3 : PQS_area = 18 := rfl
  have h4 : angle_between = Real.pi / 4 := rfl

  -- Use these hypotheses to prove the volume
  apply tetrahedron_volume
  exact h1
  exact h2
  exact h3
  exact h4

end tetrahedron_volume_l361_361407


namespace correct_statements_count_l361_361104

theorem correct_statements_count:
  (¬∀ (l r : Line), l.length > r.length) ∧
  (∀ (p q : Point), ∃ ! (l : Line), passes_through l p ∧ passes_through l q) ∧
  (¬∀ l : Line, ∃ a : Angle, l = a.straight) ∧
  (∀ (a b : Angle), supplementary a b → perpendicular (bisector a) (bisector b)) →
  (∃ n : Nat, n = 2) :=
by 
  sorry

end correct_statements_count_l361_361104


namespace maximum_modulus_inequality_l361_361049

theorem maximum_modulus_inequality (w : ℂ) (hw : complex.abs w = 2) : 
  complex.abs ((w - 2) ^ 2 * (w + 2)) ≤ 24 := sorry

end maximum_modulus_inequality_l361_361049


namespace solve_for_x_l361_361501

theorem solve_for_x (x : ℝ) : (2 / 7) * (4 / 11) * x^2 = 8 ↔ x = real.sqrt 77 ∨ x = -real.sqrt 77 :=
by
  sorry

end solve_for_x_l361_361501


namespace angle_DEF_l361_361058

noncomputable def D : ℝ × ℝ × ℝ := (1, 1, 1)
noncomputable def E : ℝ × ℝ × ℝ := (2, 3, 1)
noncomputable def F : ℝ × ℝ × ℝ := (4, 1, 1)

noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

noncomputable def cos_angle (p q r : ℝ × ℝ × ℝ) : ℝ :=
  let a := distance p q
  let b := distance q r
  let c := distance p r
  (a^2 + b^2 - c^2) / (2 * a * b)

noncomputable def angle (p q r : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (cos_angle p q r)

theorem angle_DEF :
  angle D E F = 71.565 := sorry

end angle_DEF_l361_361058


namespace counted_integer_twice_l361_361418

theorem counted_integer_twice (x n : ℕ) (hn : n = 100) 
  (h_sum : (n * (n + 1)) / 2 + x = 5053) : x = 3 := by
  sorry

end counted_integer_twice_l361_361418


namespace num_ways_excluding_specifics_correct_l361_361699

/--
To solve this problem, we need to determine the number of ways to pick 2 squares out of 7 (labeled 1 through 7) and exclude 11 specific configurations. 
-/
def num_ways_excluding_specifics : ℕ :=
  let total_combinations := Nat.choose 7 2
  let excluded_combinations := 11
  total_combinations - excluded_combinations

theorem num_ways_excluding_specifics_correct : num_ways_excluding_specifics = 10 :=
by {
  -- Let total_combinations = C(7, 2)
  let total_combinations := Nat.choose 7 2,
  -- total_combinations is equal to 21
  have h1 : total_combinations = 21 := by sorry,
  -- Define excluded_combinations as 11
  let excluded_combinations := 11,
  -- The result should be total_combinations - excluded_combinations
  let result := total_combinations - excluded_combinations,
  -- Therefore, replace result with 10
  show result = 10,
  calc
    total_combinations - excluded_combinations
      = 21 - 11 : by rw [h1]
      = 10 : by norm_num
}

#eval num_ways_excluding_specifics

end num_ways_excluding_specifics_correct_l361_361699


namespace actual_positions_correct_l361_361367

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end actual_positions_correct_l361_361367


namespace larger_integer_is_30_l361_361528

-- Define the problem statement using the given conditions
theorem larger_integer_is_30 (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h1 : a / b = 5 / 2) (h2 : a * b = 360) :
  max a b = 30 :=
sorry

end larger_integer_is_30_l361_361528


namespace option_C_implies_parallel_planes_l361_361659

variables (m n : Line)
variables (α β γ : Plane)

-- Given conditions for option C
variables [parallel_lines : Parallel m n]
variables [perpendicular_line_plane1 : Perpendicular n α]
variables [perpendicular_line_plane2 : Perpendicular m β]

theorem option_C_implies_parallel_planes : Parallel α β :=
sorry

end option_C_implies_parallel_planes_l361_361659


namespace possible_combinations_of_N_n_r_l361_361142

/-- Three numbers N, n, r such that the digits of N, n, r taken together are formed by 1, 2, 3, 4, 5, 6, 7, 8, 9 without repetition.
We are given that N = n^2 - r. We need to find all possible combinations of N, n, r.
The valid combinations are (N, n, r) = (45, 7, 4), (63, 8, 1), (72, 9, 9).
-/
theorem possible_combinations_of_N_n_r :
    ∃ (N n r : ℕ), ( (N, n, r) = (45, 7, 4) ∨ (N, n, r) = (63, 8, 1) ∨ (N, n, r) = (72, 9, 9) ) ∧
        (List.digits N ++ List.digits n ++ List.digits r).Permutation [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
        N = n^2 - r := 
by {
    sorry
}

end possible_combinations_of_N_n_r_l361_361142


namespace prove_positions_l361_361379

def athlete : Type := { A : ℕ, B : ℕ, C : ℕ, D : ℕ, E : ℕ }

def first_prediction (x : athlete) : athlete :=
  { A := 1, B := 2, C := 3, D := 4, E := 5 }

def second_prediction (x : athlete) : athlete :=
  { A := 3, B := 4, C := 1, D := 5, E := 2 }

def actual_positions : athlete :=
  { A := 3, B := 2, C := 1, D := 4, E := 5 }

theorem prove_positions :
  (λ x, first_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 3 ∧
  (λ x, second_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 2 ∧
  actual_positions.A = 3 ∧ actual_positions.B = 2 ∧ actual_positions.C = 1 ∧ actual_positions.D = 4 ∧ actual_positions.E = 5 := 
  sorry

end prove_positions_l361_361379


namespace vector_relation_l361_361724

-- Define the vectors in the vector space
noncomputable def vector_space (α : Type*) := AddCommGroup α

variables (V : Type*) [vector_space V] 
          (O A B C : V)
          (k : ℝ)

-- Assume AC = 2CB
axiom ac_eq_2cb : ∥A - C∥ = 2 * ∥C - B∥

-- The definition we want to prove
theorem vector_relation (O A B C : V) (h : ∥A - C∥ = 2 * ∥C - B∥) :
  C = (1 / 3) • A + (2 / 3) • B :=
sorry

end vector_relation_l361_361724


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l361_361908

-- Prove that (x-2)^2 = 9 has solutions x = 5 and x = -1
theorem solve_quadratic_1 (x : ℝ) : 
  ((x - 2) ^ 2 = 9) ↔ (x = 5 ∨ x = -1) :=
by sorry

-- Prove that x(x-3) + x = 3 has solutions x = 3 and x = -1
theorem solve_quadratic_2 (x : ℝ) : 
  (x * (x - 3) + x = 3) ↔ (x = 3 ∨ x = -1) :=
by sorry

-- Prove that 3x^2 - 1 = 4x has solutions x = (2 + sqrt 7) / 3 and x = (2 - sqrt 7) / 3
theorem solve_quadratic_3 (x : ℝ) :
  (3 * x ^ 2 - 1 = 4 * x) ↔ (x = (2 + real.sqrt 7) / 3 ∨ x = (2 - real.sqrt 7) / 3) :=
by sorry

-- Prove that (3x-1)^2 = (x-1)^2 has solutions x = 0 and x = 1/2
theorem solve_quadratic_4 (x : ℝ) : 
  ((3 * x - 1) ^ 2 = (x - 1) ^ 2) ↔ (x = 0 ∨ x = 1 / 2) :=
by sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l361_361908


namespace tangent_series_identity_l361_361495

noncomputable def series_tangent (x : ℝ) : ℝ := ∑' n, (1 / (2 ^ n)) * Real.tan (x / (2 ^ n))

theorem tangent_series_identity (x : ℝ) : 
  (1 / x) - (1 / Real.tan x) = series_tangent x := 
sorry

end tangent_series_identity_l361_361495


namespace calculate_base4_mult_div_l361_361234

def base4_to_base10 (n : ℕ) : ℕ :=
  if n = 231 then 2*4^2 + 3*4^1 + 1*4^0 else
  if n = 24 then 2*4^1 + 4*4^0 else
  if n = 3 then 3*4^0 else
  0

def base10_to_base4 (n : ℕ) : ℕ :=
  if n = 540 then 1*4^4 + 0*4^3 + 1*4^2 + 2*4^1 + 0*4^0 else
  if n = 180 then 1*4^2 + 1*4^1 + 3*4^0 else
  0

theorem calculate_base4_mult_div :
  let a := base4_to_base10 231
  let b := base4_to_base10 24
  let c := base4_to_base10 3
  let result := a * b / c
  base10_to_base4 result = 1130 :=
by
  intros
  rfl

end calculate_base4_mult_div_l361_361234


namespace negation_exists_x_squared_leq_abs_x_l361_361112

theorem negation_exists_x_squared_leq_abs_x :
  (¬ ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ) ∧ x^2 ≤ |x|) ↔ (∀ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ) → x^2 > |x|) :=
by
  sorry

end negation_exists_x_squared_leq_abs_x_l361_361112


namespace find_solution_l361_361336

variables (x a y b z c w d : ℝ)

noncomputable def condition1 : Prop := 
  (x / a + y / b + z / c + w / d = 4)

noncomputable def condition2 : Prop := 
  (a / x + b / y + c / z + d / w = 0)

theorem find_solution 
  (h1 : condition1 x a y b z c w d)
  (h2 : condition2 x a y b z c w d) :
  (x^2 / a^2 + y^2 / b^2 + z^2 / c^2 + w^2 / d^2) = 16 := 
begin
  sorry
end

end find_solution_l361_361336


namespace find_t_u_l361_361329

variables {V : Type*} [inner_product_space ℝ V]
variables (a b p : V)

def distance_condition : Prop :=
  ∥p - b∥ = 3 * ∥p - a∥

def fixed_distance (a b p : V) (t u : ℝ) : Prop :=
  ∥p - (t • a + u • b)∥ = ∥p - (frac 9 8 • a + (-frac 1 8) • b)∥

theorem find_t_u (h : distance_condition a b p) :
  fixed_distance a b p (9 / 8) (-1 / 8) := 
sorry

end find_t_u_l361_361329


namespace probability_same_color_l361_361970

theorem probability_same_color (h : true) : 
  let p := (13/20:Real)^3 + (7/20:Real)^3 in 
  p = 127 / 400 := by
  sorry

end probability_same_color_l361_361970


namespace solution_set_of_inequality_l361_361252

-- Let f be a real-valued function that is decreasing on the interval [1, 3]
variable {f : ℝ → ℝ}
variable h_decreasing : ∀ (x₁ x₂ : ℝ), 1 ≤ x₁ → x₁ ≤ 3 → 1 ≤ x₂ → x₂ ≤ 3 → x₁ ≤ x₂ → f x₂ ≤ f x₁

-- Define the theorem
theorem solution_set_of_inequality (a : ℝ) : (f (1 - a) - f (3 - a^2) > 0) ↔ (-1 < a ∧ a ≤ 0) := 
by
  sorry

end solution_set_of_inequality_l361_361252


namespace infinite_lines_intersecting_skew_lines_l361_361782

-- Definitions for skew lines in space
variables {a b c : Line ℝ}

-- Condition: The lines a, b, and c are pairwise skew
def pairwise_skew (a b c : Line ℝ) : Prop :=
  (∀ (p1 p2 : Point ℝ), ¬ (is_on_line p1 a ∧ is_on_line p2 b ∧ p1 = p2)) ∧
  (∀ (p1 p2 : Point ℝ), ¬ (is_on_line p1 b ∧ is_on_line p2 c ∧ p1 = p2)) ∧
  (∀ (p1 p2 : Point ℝ), ¬ (is_on_line p1 c ∧ is_on_line p2 a ∧ p1 = p2))

-- Main theorem statement
theorem infinite_lines_intersecting_skew_lines
  (h : pairwise_skew a b c) :
  ∃ (f : ℕ → Line ℝ), ∀ (n : ℕ), intersects_line (f n) a ∧ intersects_line (f n) b ∧ intersects_line (f n) c :=
sorry

end infinite_lines_intersecting_skew_lines_l361_361782


namespace flight_cost_DE_is_660_l361_361475

-- Define the necessary constants and conditions
def distance_DE := 4500 -- km
def cost_per_km_flight := 0.12 -- $/km
def booking_fee_flight := 120 -- $

-- Define the function to calculate flight cost
def flight_cost (distance : ℕ) (cost_per_km : ℕ) (booking_fee : ℕ) : ℕ :=
  distance * cost_per_km + booking_fee

-- Define the theorem to prove
theorem flight_cost_DE_is_660 : flight_cost distance_DE cost_per_km_flight booking_fee_flight = 660 := by
  sorry

end flight_cost_DE_is_660_l361_361475


namespace closest_diff_sqrt_l361_361570

def sqrt75 := Real.sqrt 75
def sqrt72 := Real.sqrt 72

theorem closest_diff_sqrt (h1 : sqrt75 ≈ 8.7) (h2 : sqrt72 ≈ 8.5) : abs (sqrt75 - sqrt72 - 0.17) < 0.01 :=
by
  sorry

end closest_diff_sqrt_l361_361570


namespace probability_distance_ge_side_length_of_square_l361_361700

theorem probability_distance_ge_side_length_of_square :
  let points := finset.univ : finset (fin 5),
      side_length := 1  -- without loss of generality
      num_pairs := (points.card.choose 2),
      num_valid_pairs := (finset.filter (λ (p : (fin 5) × (fin 5)),
                                          p.1 ≠ p.2 ∧
                                          (distance (coordinates p.1) (coordinates p.2) ≥ side_length)) (points.product points)).card
  in
  num_valid_pairs / num_pairs = 3 / 5 := 
sorry

-- Auxiliary definitions for distance and coordinates (can be further refined as needed)
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def coordinates (i : fin 5) : (ℝ × ℝ) :=
  match i with
  | 0 => (0, 0)
  | 1 => (1, 0)
  | 2 => (1, 1)
  | 3 => (0, 1)
  | 4 => (0.5, 0.5)
  end

end probability_distance_ge_side_length_of_square_l361_361700


namespace count_two_digit_numbers_l361_361846

theorem count_two_digit_numbers : ∃ n : ℕ, 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 → 
  (((10 * a + b) < 100 ∧ (100 * a + 10 * c + b = 9 * (10 * a + b))) → false)) ∧ n = 4 :=
begin
  sorry
end

end count_two_digit_numbers_l361_361846


namespace tan_alpha_minus_beta_eq_neg7_over_4_l361_361313

variable (α β : ℝ)

def tan_alpha_plus_pi_over_3_eq_neg3 : Prop := tan (α + π / 3) = -3
def tan_beta_minus_pi_over_6_eq_5 : Prop := tan (β - π / 6) = 5

theorem tan_alpha_minus_beta_eq_neg7_over_4
  (h1 : tan_alpha_plus_pi_over_3_eq_neg3 α β)
  (h2 : tan_beta_minus_pi_over_6_eq_5 α β) : tan (α - β) = -7 / 4 :=
  by
    sorry

end tan_alpha_minus_beta_eq_neg7_over_4_l361_361313


namespace radius_of_cylinder_is_correct_l361_361606

/-- 
  A right circular cylinder is inscribed in a right circular cone such that:
  - The diameter of the cylinder is equal to its height.
  - The cone has a diameter of 8.
  - The cone has an altitude of 10.
  - The axes of the cylinder and cone coincide.
  Prove that the radius of the cylinder is 20/9.
-/
theorem radius_of_cylinder_is_correct :
  ∀ (r : ℚ), 
    (2 * r = 8 - 2 * r ∧ 10 - 2 * r = (10 / 4) * r) → 
    r = 20 / 9 :=
by
  intro r
  intro h
  sorry

end radius_of_cylinder_is_correct_l361_361606


namespace find_g_of_3_l361_361774

theorem find_g_of_3 (f g : ℝ → ℝ) (h₁ : ∀ x, f x = 2 * x + 3) (h₂ : ∀ x, g (x + 2) = f x) :
  g 3 = 5 :=
sorry

end find_g_of_3_l361_361774


namespace magnitude_of_a_cos_angle_between_a_and_b_l361_361328

noncomputable def vector_a := (-4, 2, 4)
noncomputable def vector_b := (-6, 3, -2)

theorem magnitude_of_a:
  ∥vector_a∥ = 6 :=
sorry

theorem cos_angle_between_a_and_b:
  let dot_prod : ℝ := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 + vector_a.3 * vector_b.3
  ∥vector_a∥ = 6 →
  ∥vector_b∥ = 7 →
  dot_prod = 22 →
  dot_prod / (∥vector_a∥ * ∥vector_b∥) = 11 / 21 :=
sorry

end magnitude_of_a_cos_angle_between_a_and_b_l361_361328


namespace sqrt_mul_sqrt_l361_361984

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361984


namespace upstream_distance_l361_361955

variable (Vb Vs Vdown Vup Dup : ℕ)

def boatInStillWater := Vb = 36
def speedStream := Vs = 12
def downstreamSpeed := Vdown = Vb + Vs
def upstreamSpeed := Vup = Vb - Vs
def timeEquality := 80 / Vdown = Dup / Vup

theorem upstream_distance (Vb Vs Vdown Vup Dup : ℕ) 
  (h1 : boatInStillWater Vb)
  (h2 : speedStream Vs)
  (h3 : downstreamSpeed Vb Vs Vdown)
  (h4 : upstreamSpeed Vb Vs Vup)
  (h5 : timeEquality Vdown Vup Dup) : Dup = 40 := 
sorry

end upstream_distance_l361_361955


namespace d_min_not_unique_d_c_le_d_m_l361_361717

-- Define the function d(t)
def d (x : List ℝ) (t : ℝ) : ℝ :=
  (x.map (fun xi => abs (xi - t))).minimum + (x.map (fun xi => abs (xi - t))).maximum / 2

-- Define c and m
def c (x : List ℝ) : ℝ :=
  (x.minimum + x.maximum) / 2

def median (x : List ℝ) : ℝ :=
  let sorted_x := x.qsort (· < ·)
  if sorted_x.length % 2 = 0 then
    (sorted_x.nth (sorted_x.length / 2 - 1) + sorted_x.nth (sorted_x.length / 2)) / 2
  else
    sorted_x.nth (sorted_x.length / 2)

-- Theorem for Part (a): d(t) does not always attain its minimum value at a unique point.
theorem d_min_not_unique : 
  ∃ (x : List ℝ), 
  ∃ (t₁ t₂ : ℝ), 
  t₁ ≠ t₂ ∧ d x t₁ = d x t₂ ∧ d x t₁ = x.map (fun xi => d x xi).minimum := 
sorry

-- Theorem for Part (b): d(c) ≤ d(m).
theorem d_c_le_d_m (x : List ℝ) : 
  d x (c x) ≤ d x (median x) := 
sorry

end d_min_not_unique_d_c_le_d_m_l361_361717


namespace solve_m_n_l361_361484

theorem solve_m_n (m n : ℤ) (h : m^2 - 2 * m * n + 2 * n^2 - 8 * n + 16 = 0) : m = 4 ∧ n = 4 :=
sorry

end solve_m_n_l361_361484


namespace find_x_parallel_vectors_l361_361296

theorem find_x_parallel_vectors
   (x : ℝ)
   (ha : (x, 2) = (x, 2))
   (hb : (-2, 4) = (-2, 4))
   (hparallel : ∀ (k : ℝ), (x, 2) = (k * -2, k * 4)) :
   x = -1 :=
by
  sorry

end find_x_parallel_vectors_l361_361296


namespace rotation_A_120_degrees_l361_361557

def point_A := (5, 5 * Real.sqrt 3 / 3 : ℝ × ℝ)
def rotation_matrix (θ : ℝ) : ℝ × ℝ → ℝ × ℝ :=
  λ p, (p.1 * Real.cos θ - p.2 * Real.sin θ, p.1 * Real.sin θ + p.2 * Real.cos θ)

theorem rotation_A_120_degrees :
  rotation_matrix (2 * Real.pi / 3) point_A = (-5 / 2 * (1 + Real.sqrt 3), 5 / 2 * (Real.sqrt 3 - 1)) :=
by
  sorry

end rotation_A_120_degrees_l361_361557


namespace initial_number_correct_l361_361582

def initial_number_problem : Prop :=
  ∃ (x : ℝ), x + 3889 - 47.80600000000004 = 3854.002 ∧
            x = 12.808000000000158

theorem initial_number_correct : initial_number_problem :=
by
  -- proof goes here
  sorry

end initial_number_correct_l361_361582


namespace largest_area_of_rectangle_l361_361208

noncomputable def max_rectangle_area (P : ℕ) := max {A : ℕ | ∃ a b : ℕ, 2 * (a + b) = P ∧ a ≥ b / 2 ∧ A = a * b}

theorem largest_area_of_rectangle (P : ℕ) (hP : P = 60) :
  max_rectangle_area P = 200 := sorry

end largest_area_of_rectangle_l361_361208


namespace Theresa_video_games_l361_361969

variable (Theresa Julia Tory Alex : ℕ)

-- Given conditions as definitions
def condition1 : Prop := Theresa = 3 * Julia + 5
def condition2 : Prop := Julia = Tory / 3
def condition3 : Prop := Tory = 2 * Alex
def condition4 : Prop := Tory = 6
def condition5 : Prop := Alex = Theresa / 2

-- Final proof statement
theorem Theresa_video_games : condition1 Theresa Julia Tory Alex ∧
                              condition2 Theresa Julia Tory Alex ∧
                              condition3 Theresa Julia Tory Alex ∧
                              condition4 Tory ∧
                              condition5 Theresa Alex →
                              Theresa = 11 := by
  sorry

end Theresa_video_games_l361_361969


namespace range_of_quadratic_l361_361773

theorem range_of_quadratic (a b c : ℝ) (h_a : a > 0) :
  let f := λ x : ℝ, a*x^2 + b*x + c in
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 →
  set.range f = set.Icc (min (min (c + -b^2 / (4*a)) c) (4*a + 2*b + c))
                        (max c (4*a + 2*b + c)) :=
by
  sorry

end range_of_quadratic_l361_361773


namespace correct_prediction_l361_361369

-- Declare the permutation type
def Permutation := (ℕ × ℕ × ℕ × ℕ × ℕ)

-- Declare the actual finish positions
def actual_positions : Permutation := (3, 2, 1, 4, 5)

-- Declare conditions
theorem correct_prediction 
  (P1 : Permutation) (P2 : Permutation) 
  (hP1 : P1 = (1, 2, 3, 4, 5))
  (hP2 : P2 = (1, 2, 3, 4, 5))
  (hP1_correct : ∃! i ∈ (1..5), (λ i, nth (P1.to_list) (i - 1) == nth (actual_positions.to_list))  = 3)
  (hP2_correct : ∃! i ∈ (1..5), (λ i, nth (P2.to_list) (i - 1) == nth (actual_positions.to_list))  = 2)
  : actual_positions = (3, 2, 1, 4, 5) 
:= sorry

end correct_prediction_l361_361369


namespace find_digits_to_correct_sum_l361_361510

def change_digit (n : ℕ) (d e : ℕ) : ℕ :=
  let s := n.toString.map (λ c, if c.toNat = d + '0'.toNat then e + '0'.toNat else c.toNat)
  (String.mk s).toNat

theorem find_digits_to_correct_sum :
  ∃ (d e : ℕ), d + e = 14 ∧ 
    change_digit 953672 d e + change_digit 637528 d e = 1511200 :=
by
  sorry

end find_digits_to_correct_sum_l361_361510


namespace math_problem_l361_361673

theorem math_problem (n : ℕ) (hn1 : 8001 < n) (hn2 : n < 8200) 
  (hdiv : ∀ k : ℕ, k > n → (2^n - 1) ∣ (2^(k * (fact (n - 1)) + k^n) - 1)) : n = 8111 := sorry

end math_problem_l361_361673


namespace minimum_value_2m_plus_n_l361_361648

-- You can declare a noncomputable theory only if necessary
noncomputable theory

-- Define the focus of the parabola y = x^2
def focus : (ℝ × ℝ) := (0, 1/4)

-- Define the points P and Q on the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Assume P = (p_x, parabola p_x) and Q = (q_x, parabola q_x)
variables (p_x q_x : ℝ)

-- Define the lengths PF and QF
def length_PF : ℝ := dist (focus) (p_x, parabola p_x)
def length_QF : ℝ := dist (focus) (q_x, parabola q_x)

-- Assume that the line passing through F is horizontal, hence choice of k = 0
def line_through_focus : ℝ := focus.snd

-- Minimum value of 2m + n is the answer \(\frac {3+2 \sqrt {2}}{4}\)
theorem minimum_value_2m_plus_n : 
  ∃ m n: ℝ, m = length_PF ∧ n = length_QF ∧ 2 * m + n = (3+2*sqrt 2) / 4 :=
by sorry

end minimum_value_2m_plus_n_l361_361648


namespace altitude_eqn_median_eqn_l361_361718

def Point := (ℝ × ℝ)

def A : Point := (4, 0)
def B : Point := (6, 7)
def C : Point := (0, 3)

theorem altitude_eqn (B C: Point) : 
  ∃ (k b : ℝ), (b = 6) ∧ (k = - 3 / 2) ∧ (∀ x y : ℝ, y = k * x + b →
  3 * x + 2 * y - 12 = 0)
:=
sorry

theorem median_eqn (A B C : Point) :
  ∃ (k b : ℝ), (b = 20) ∧ (k = -3/5) ∧ (∀ x y : ℝ, y = k * x + b →
  5 * x + y - 20 = 0)
:=
sorry

end altitude_eqn_median_eqn_l361_361718


namespace exists_set_B_l361_361476

noncomputable def construct_set_B (A : Finset ℕ) : Finset ℕ :=
  let m := (A.max' (by exact' A.nonempty)).succ in
  let x := λ i : ℕ, Nat.fact m * List.prod (List.range i) - 1 in
  Finset.union A (Finset.range m ∪ Finset.range.succ (λ i : ℕ, x i))

theorem exists_set_B (A : Finset ℕ) (hA : ∀ x ∈ A, 0 < x) :
  ∃ B : Finset ℕ, A ⊆ B ∧ (Finset.prod B id) = (Finset.sum B (λ x, x^2)) :=
by
  have m : ℕ := (A.max' (by exact' A.nonempty)).succ
  let x := λ i : ℕ, Nat.fact m * List.prod (List.range i) - 1
  have B := construct_set_B A
  use B
  split
  · -- Proving A ⊆ B
    sorry
  · -- Proving Finset.prod B id = Finset.sum B (λ x, x^2)
    sorry

end exists_set_B_l361_361476


namespace principal_amount_is_approx_1200_l361_361922

noncomputable def find_principal_amount : Real :=
  let R := 0.10
  let n := 2
  let T := 1
  let SI (P : Real) := P * R * T / 100
  let CI (P : Real) := P * ((1 + R / n) ^ (n * T)) - P
  let diff (P : Real) := CI P - SI P
  let target_diff := 2.999999999999936
  let P := target_diff / (0.1025 - 0.10)
  P

theorem principal_amount_is_approx_1200 : abs (find_principal_amount - 1200) < 0.0001 := 
by
  sorry

end principal_amount_is_approx_1200_l361_361922


namespace am_gm_four_variables_l361_361077

theorem am_gm_four_variables (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) / 4 ≥ real.sqrt (real.sqrt (a * b * c * d)) :=
by { sorry }

end am_gm_four_variables_l361_361077


namespace sqrt_mul_sqrt_l361_361989

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361989


namespace distance_from_origin_to_point_l361_361809

theorem distance_from_origin_to_point : 
  ∀ (x y : ℝ), x = 8 → y = -15 → real.sqrt ((x-0)^2 + (y-0)^2) = 17 :=
by
  intros x y hx hy
  rw [hx, hy]
  norm_num
  rw [real.sqrt_eq_rpow]
  norm_num
  sorry

end distance_from_origin_to_point_l361_361809


namespace product_sign_and_digits_l361_361979

def sequence_term (n : ℕ) : ℤ := -2033 + (n - 1) * 4

def product_sequence_up_to (n : ℕ) : ℤ := ∏ i in finset.range (n + 1), sequence_term i

theorem product_sign_and_digits (n : ℕ) (hn : n = 509) : 
  product_sequence_up_to 509 < 0 ∧ (product_sequence_up_to 509 % 100 = -25) :=
  by
  sorry

end product_sign_and_digits_l361_361979


namespace max_items_with_discount_l361_361215

theorem max_items_with_discount (total_money items original_price discount : ℕ) 
  (h_orig: original_price = 30)
  (h_discount: discount = 24) 
  (h_limit: items > 5 → (total_money <= 270)) : items ≤ 10 :=
by
  sorry

end max_items_with_discount_l361_361215


namespace func_C_increasing_l361_361620

open Set

noncomputable def func_A (x : ℝ) : ℝ := 3 - x
noncomputable def func_B (x : ℝ) : ℝ := x^2 - x
noncomputable def func_C (x : ℝ) : ℝ := -1 / (x + 1)
noncomputable def func_D (x : ℝ) : ℝ := -abs x

theorem func_C_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → func_C x < func_C y := by
  sorry

end func_C_increasing_l361_361620


namespace geometric_progression_lemma_l361_361342

variable {x y z : ℝ} {a r : ℝ}

-- Conditions: x, y, z are distinct and non-zero, and (x * (y - z)), (y * (z - x)), (z * (x - y)) form a GP
axiom distinct_nonzero (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x) :
  ((∃ a r : ℝ, x * (y - z) = a ∧ y * (z - x) = a * r ∧ z * (x - y) = a * r^3) → (1 + r + r^3 = 0))

theorem geometric_progression_lemma : (x * (y - z), y * (z - x), z * (x - y)) = ([a, a * r, a * r^3]) → (1 + r + r^3 = 0) := by
  sorry

end geometric_progression_lemma_l361_361342


namespace distance_from_origin_to_point_l361_361802

-- Define the specific points in the coordinate system
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (8, -15)

-- The distance formula in Euclidean space
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem statement
theorem distance_from_origin_to_point : distance origin point = 17 := 
by
  sorry

end distance_from_origin_to_point_l361_361802


namespace lines_parallel_l361_361043

variable {A B : ℝ}
variable {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)

theorem lines_parallel (h1 : - (sin A) / a = - (sin B) / b) (h2 : c ≠ 0) :
  are_parallel (line (sin A) a c) (line (sin B) b) :=
sorry

end lines_parallel_l361_361043


namespace james_out_of_pocket_is_25657_50_l361_361419

def sale_price (value : ℕ) (percentage : ℕ) : ℕ :=
  value * percentage / 100

def purchase_price (sticker_price : ℕ) (discount_percentage : ℕ) : ℕ :=
  sticker_price * discount_percentage / 100

def sales_tax (price : ℕ) (tax_rate : ℕ) : ℕ :=
  price * tax_rate / 100

def processing_fee (amount : ℕ) (fee_rate : ℕ) : ℕ :=
  amount * fee_rate / 100

def out_of_pocket (purchase_prices : list (ℕ × ℕ)) (sale_prices : list (ℕ × ℕ)) (tax_rate : ℕ) (fee_rate : ℕ) : ℕ :=
  let total_purchase_cost := purchase_prices.sum (λ p, p.fst + sales_tax p.fst tax_rate) in
  let total_sale_revenue := sale_prices.sum (λ s, s.fst - processing_fee s.fst fee_rate) in
  total_purchase_cost - total_sale_revenue

theorem james_out_of_pocket_is_25657_50 :
  out_of_pocket [(purchase_price 30000 90, 1890), (purchase_price 25000 85, 1487)]
                [(sale_price 20000 80, 320), (sale_price 15000 70, 210)]
                7 2 
  = 25657 := by {
    sorry
  }

end james_out_of_pocket_is_25657_50_l361_361419


namespace abs_neg_one_eq_one_l361_361509

theorem abs_neg_one_eq_one : abs (-1 : ℚ) = 1 := 
by
  sorry

end abs_neg_one_eq_one_l361_361509


namespace b_is_arithmetic_seq_sum_sequence_a_formula_l361_361843

noncomputable def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * sequence_a (n - 1) + 2^(n-1)

def sequence_b (n : ℕ) : ℕ :=
  sequence_a n / 2^(n-1)

def is_arithmetic_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b(n + 1) - b(n) = 1

theorem b_is_arithmetic_seq : is_arithmetic_sequence sequence_b :=
sorry

noncomputable def sum_sequence_a (n : ℕ) : ℕ :=
  ∑ i in range n, sequence_a (i + 1)

theorem sum_sequence_a_formula (n : ℕ) : sum_sequence_a n = (n - 1) * 2^n + 1 :=
sorry

end b_is_arithmetic_seq_sum_sequence_a_formula_l361_361843


namespace chords_even_arcs_even_l361_361928

theorem chords_even_arcs_even (N : ℕ) 
  (h1 : ∀ k : ℕ, k ≤ N → ¬ ((k : ℤ) % 2 = 1)) : 
  N % 2 = 0 := 
sorry

end chords_even_arcs_even_l361_361928


namespace integer_solutions_count_l361_361055

-- Problem statement
theorem integer_solutions_count (n : ℕ) 
  (r : ℤ := (n % 2)) : 
  let is_solution (x y z : ℤ) : Prop := 
    (x + y + z = r) ∧ (|x| + |y| + |z| = n)
  in ∃ sol_count : ℕ, sol_count = 3 * n ∧ 
    ∀ (x y z : ℤ), is_solution x y z → sol_count = 3 * n := 
sorry

end integer_solutions_count_l361_361055


namespace portion_pump_X_initial_l361_361170

def portion_pumped_by_X (W : ℝ) (R_x R_y : ℝ) : ℝ :=
  3 * 17 * W / 120

theorem portion_pump_X_initial
  (W : ℝ)
  (R_x R_y : ℝ)
  (h1 : R_y = W / 20)
  (h2 : 3 * R_x + 3 * (R_x + R_y) = W) :
  portion_pumped_by_X W R_x R_y = 17 * W / 40 :=
by
  sorry

end portion_pump_X_initial_l361_361170


namespace kids_playing_soccer_l361_361431

theorem kids_playing_soccer (boxes bars_per_box bars_per_kid : ℕ) 
(h_boxes : boxes = 5) 
(h_bars_per_box : bars_per_box = 12) 
(h_bars_per_kid : bars_per_kid = 2) : 
(boxes * bars_per_box) / bars_per_kid = 30 :=
by
  rw [h_boxes, h_bars_per_box, h_bars_per_kid]
  norm_num

end kids_playing_soccer_l361_361431


namespace carly_flip_5_heads_l361_361776

theorem carly_flip_5_heads (n k : ℕ) (h_n : n = 9) (h_k : k = 5)
                            (hw : nat.choose n k = 126) :
  (nat.choose 9 5 * (1 / 2) ^ 5 * (1 / 2) ^ 4 = 126 / 512) :=
by {
  sorry
}

end carly_flip_5_heads_l361_361776


namespace sum_of_max_min_values_l361_361310

variable (a b m : ℝ)
variable (f : ℝ → ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def is_maximum_value (f : ℝ → ℝ) (m : ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ set.Icc a b, f x ≤ m

theorem sum_of_max_min_values (h_odd : is_odd_function f)
  (h_max : is_maximum_value f m a b) :
  let F := λ x, f(x) + 3 in
  m + 3 + (-m + 3) = 6 :=
by
  sorry

end sum_of_max_min_values_l361_361310


namespace semicircle_length_invariant_l361_361069

theorem semicircle_length_invariant (A B C : Point) (l : ℝ) (x : ℝ) 
  (hAB : segment_length A B = 2 * l)
  (hC : C ∈ segment A B) :
  let AC := segment_length A C,
      CB := segment_length C B,
      radius_AC := AC / 2,
      radius_CB := CB / 2 in
  (π * radius_AC + π * radius_CB = π * l) :=
by
  sorry

end semicircle_length_invariant_l361_361069


namespace sector_area_is_correct_l361_361734

noncomputable def area_of_sector (r : ℝ) (α : ℝ) : ℝ := 1/2 * α * r^2

theorem sector_area_is_correct (circumference : ℝ) (central_angle : ℝ) (r : ℝ) (area : ℝ) 
  (h1 : circumference = 8) 
  (h2 : central_angle = 2) 
  (h3 : circumference = central_angle * r + 2 * r)
  (h4 : r = 2) : area = 4 :=
by
  have h5: area = 1/2 * central_angle * r^2 := sorry
  exact sorry

end sector_area_is_correct_l361_361734


namespace distance_from_origin_to_point_l361_361803

-- Define the specific points in the coordinate system
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (8, -15)

-- The distance formula in Euclidean space
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem statement
theorem distance_from_origin_to_point : distance origin point = 17 := 
by
  sorry

end distance_from_origin_to_point_l361_361803


namespace storm_time_l361_361925

def time_corrected (hours tens units : ℕ) (minutes tens units : ℕ) : Prop :=
(hours tens units = 1 ∨ hours tens units = 3) ∧
(hours units = 1 ∨ hours units = 9) ∧
(minutes tens units = 1 ∨ minutes tens units = 9) ∧
(minutes units = 8 ∨ minutes units = 0)

theorem storm_time :
  ∃ hours tens units: ℕ, ∃ hours units: ℕ, ∃ minutes tens units: ℕ, ∃ minutes units: ℕ,
  time_corrected hours tens units minutes tens units ∧
  (hours tens units = 1) ∧ (hours units = 1) ∧ (minutes tens units = 1) ∧ (minutes units = 8) :=
begin
  use 1,
  use 1,
  use 1,
  use 8,
  split,
  { split,
    { left, refl },
    split,
    { left, refl },
    split,
    { left, refl },
    { left, refl }},
  exact ⟨rfl, rfl, rfl, rfl⟩
end

end storm_time_l361_361925


namespace prob_not_one_seventh_l361_361433

def is_leap_year (n : ℕ) : Prop :=
  (n % 4 = 0 ∧ n % 100 ≠ 0) ∨ (n % 400 = 0)

theorem prob_not_one_seventh :
  ∃ P : ℕ → ℚ, P (25 : ℕ, 12 : ℕ) ≠ 1 / 7 :=
by
  sorry

end prob_not_one_seventh_l361_361433


namespace seashells_total_l361_361911

theorem seashells_total :
  ∀ (S V A F : ℕ), 
    A = 20 → 
    V = A - 5 → 
    S = V + 16 → 
    F = 2 * A → 
    S + V + A + F = 106 :=
by
  intros S V A F hA hV hS hF
  rw [hA, hV, hS, hF]
  simp
  sorry

end seashells_total_l361_361911


namespace product_less_than_one_tenth_l361_361075

namespace MathProof

noncomputable def a : ℚ := (List.range' 1 50).map (λ n, n * (n + 1)⁻¹).prod
noncomputable def b : ℚ := (List.range' 2 50).map (λ n, n * (n + 1)⁻¹).prod

theorem product_less_than_one_tenth
    (ha : a = (List.range' 1 50).map (λ n, n * (n + 1)⁻¹).prod) 
    (hb : b = (List.range' 2 49).map (λ n, n * (n + 1)⁻¹).prod)
    (hab : a * b = (1 : ℚ) / 100) 
    (ho : a < b) : a < (1 : ℚ) / 10 :=
    sorry

end MathProof

end product_less_than_one_tenth_l361_361075


namespace distance_from_origin_to_point_l361_361808

theorem distance_from_origin_to_point : 
  ∀ (x y : ℝ), x = 8 → y = -15 → real.sqrt ((x-0)^2 + (y-0)^2) = 17 :=
by
  intros x y hx hy
  rw [hx, hy]
  norm_num
  rw [real.sqrt_eq_rpow]
  norm_num
  sorry

end distance_from_origin_to_point_l361_361808


namespace ellipse_properties_l361_361012

theorem ellipse_properties :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ↔ 
   x^2 / 4 + y^2 = 1) ∧ 
  (eccentricity a b = sqrt(3) / 2) ∧ 
  (point_on_ellipse x y a b ↔ x^2 / 4 + y^2 = 1) ∧ 
  (line_intersects_ellipse x y k m a b ↔ x^2 / 16 + y^2 / 4 = 1 ∧
  |OQ| / |OP| = 2) ∧ 
  (maximum_area_triangle A B Q = 6*sqrt(3)))) :=
sorry

end ellipse_properties_l361_361012


namespace adults_count_l361_361627

noncomputable def cost_adult_meal : ℕ := 6
noncomputable def cost_child_meal : ℕ := 4
noncomputable def cost_soda : ℕ := 2
noncomputable def total_bill : ℕ := 60
noncomputable def number_of_children : ℕ := 2

theorem adults_count :
  ∃ A : ℕ, (cost_adult_meal + cost_soda) * A + (cost_child_meal + cost_soda) * number_of_children = total_bill ∧ A = 6 :=
begin
  use 6,
  split,
  { unfold cost_adult_meal cost_child_meal cost_soda total_bill number_of_children,
    norm_num },
  { refl }
end

end adults_count_l361_361627


namespace solitaire_game_probability_l361_361610

theorem solitaire_game_probability :
  let P : ℕ → ℚ := λ k, if k = 1 then 1 else (3 * P (k - 1)) / (2 * k - 1)
  ∧ let P6 := P 6
  ∧ gcd P6.num P6.den = 1
  ∧ P6 = 1 / 43
in P6.num + P6.den = 44 :=
by
  sorry

end solitaire_game_probability_l361_361610


namespace find_speed_train2_l361_361558

def length_train1 : ℝ := 120
def length_train2 : ℝ := 280
def speed_train1 : ℝ := 42
def time_to_pass : ℝ := 19.99840012798976

theorem find_speed_train2 (length_train1 length_train2 speed_train1 time_to_pass : ℝ) : 
  (speed_train2 = 30.004800256) :=
by
  let distance := (length_train1 + length_train2) / 1000  -- convert to km
  let time := time_to_pass / 3600  -- convert to hours
  let relative_speed := distance / time
  let speed_train2 := relative_speed - speed_train1
  exact sorry

example : find_speed_train2 120 280 42 19.99840012798976 = 30.004800256 := sorry

end find_speed_train2_l361_361558


namespace find_k_l361_361723

theorem find_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (hk : k ≠ 1) (h3 : 2 * a + b = a * b) : 
  k = 18 :=
sorry

end find_k_l361_361723


namespace distance_from_origin_l361_361816

theorem distance_from_origin :
  let origin := (0, 0)
      pt := (8, -15)
  in Real.sqrt ((pt.1 - origin.1)^2 + (pt.2 - origin.2)^2) = 17 :=
by
  let origin := (0, 0)
  let pt := (8, -15)
  have h1 : pt.1 - origin.1 = 8 := by rfl
  have h2 : pt.2 - origin.2 = -15 := by rfl
  simp [h1, h2]
  -- sorry will be used to skip the actual proof steps and to ensure the code builds successfully
  sorry

end distance_from_origin_l361_361816


namespace marco_new_cards_l361_361061

theorem marco_new_cards (total_cards : ℕ) (fraction_duplicates : ℕ) (fraction_traded : ℕ) : 
  total_cards = 500 → 
  fraction_duplicates = 4 → 
  fraction_traded = 5 → 
  (total_cards / fraction_duplicates) / fraction_traded = 25 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.div_div_eq_div_mul]
  norm_num

-- Since Lean requires explicit values, we use the following lemma to lead the theorem to concrete values:
lemma marco_new_cards_concrete : 
  (500 / 4) / 5 = 25 :=
marco_new_cards 500 4 5 rfl rfl rfl

end marco_new_cards_l361_361061


namespace sum_of_numbers_ratio_1_2_4_l361_361182
noncomputable theory

variable {x : ℝ}

theorem sum_of_numbers_ratio_1_2_4 (h : x^2 + (2 * x)^2 + (4 * x)^2 = 1701) :
  x + 2 * x + 4 * x = 63 :=
sorry

end sum_of_numbers_ratio_1_2_4_l361_361182


namespace distance_origin_to_point_l361_361798

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l361_361798


namespace parallelogram_properties_l361_361010

variables (A B C D : Point)
variable (ABCD : parallelogram A B C D)

variables (l1 l2 : Line)
variable (l1_is_angle_bisector_A : angle_bisector l1 (angle A B D))
variable (l2_is_angle_bisector_C : angle_bisector l2 (angle C D B))

variables (m1 m2 : Line)
variable (m1_is_angle_bisector_B : angle_bisector m1 (angle B A C))
variable (m2_is_angle_bisector_D : angle_bisector m2 (angle D C A))

variable (distance_l1_l2 : ℝ)
variable (distance_m1_m2 : ℝ)
variable (distance_relation : distance_l1_l2 = sqrt 3 * distance_m1_m2)

variable (AC : ℝ := 3)
variable (BD : ℝ)
variable (BD_expression : BD = sqrt (5 * g / 3))

theorem parallelogram_properties :
  ∠(A B D) = 2 * π / 3 ∧
  incircle_radius (triangle A B D) = 1 / sqrt 3 :=
by
  sorry

end parallelogram_properties_l361_361010


namespace tiles_used_in_total_l361_361065

theorem tiles_used_in_total 
  (hallway_length : ℕ := 20) 
  (hallway_width : ℕ := 30) 
  (outer_border_tile_size : ℕ := 1) 
  (second_border_tile_size_length : ℕ := 2) 
  (second_border_tile_size_width : ℕ := 1) 
  (central_tile_size : ℕ := 3) : 
  (total_tiles : ℕ := 175 :=
    let outer_effective_length := hallway_length - 2 * outer_border_tile_size
    let outer_effective_width := hallway_width - 2 * outer_border_tile_size
    let outer_border_tiles := 2 * (outer_effective_length + outer_effective_width) + 4
    let second_effective_length := outer_effective_length - 2 * second_border_tile_size_width
    let second_effective_width := outer_effective_width - 2 * second_border_tile_size_width
    let second_border_tiles := 2 * (second_effective_length / second_border_tile_size_length + second_effective_width / second_border_tile_size_length)
    let central_area := (second_effective_length - 2 * second_border_tile_size_width) * (second_effective_width - 2 * second_border_tile_size_width)
    let central_tiles := central_area / (central_tile_size * central_tile_size)
    total_tiles = outer_border_tiles + second_border_tiles + central_tiles)
:= by
  sorry

end tiles_used_in_total_l361_361065


namespace polynomial_fourth_degree_abs_value_l361_361865

noncomputable def f (x : ℝ) : ℝ := sorry

theorem polynomial_fourth_degree_abs_value :
  (∀ x ∈ {1, 3, 4, 5, 7}, |f x| = 16) ∧ polynomial.degree (polynomial.C (f 0) * polynomial.X ^ 4) = 4 → |f 0| = 436 :=
by
  intro h
  have h1 : ∀ x ∈ ({1, 3, 4, 5, 7} : set ℝ), |f x| = 16 := by
    intros x hx
    exact h.1 x hx
  have h2 : polynomial.degree (polynomial.C (f 0) * polynomial.X ^ 4) = 4 := h.2
  sorry

end polynomial_fourth_degree_abs_value_l361_361865


namespace log_eval_l361_361262

-- Lean statement
theorem log_eval : log 4 (64 * sqrt 4) = 7 / 2 := by
  sorry

end log_eval_l361_361262


namespace john_percentage_increase_l361_361850

noncomputable def john_initial_salary : ℝ := 30
noncomputable def john_raises : list ℝ := [0.10, 0.15, 0.05]
noncomputable def john_freelance_income : ℝ := 10
noncomputable def tax_rate : ℝ := 0.05

def calculate_final_net_income (initial_salary : ℝ) (raises : list ℝ) (freelance_income : ℝ) (tax_rate : ℝ) : ℝ :=
  let final_salary := raises.foldl (λ salary raise, salary * (1 + raise)) initial_salary
  let total_income := final_salary + freelance_income
  let total_tax := total_income * tax_rate
  total_income - total_tax

def percentage_increase (initial_net_income final_net_income : ℝ) : ℝ :=
  (final_net_income - initial_net_income) / initial_net_income * 100

theorem john_percentage_increase :
  percentage_increase (john_initial_salary * (1 - tax_rate))
                      (calculate_final_net_income john_initial_salary john_raises john_freelance_income tax_rate)
  ≈ 66.17 :=
by
  sorry

end john_percentage_increase_l361_361850


namespace problem_l361_361769

variable (x : ℝ) (Q : ℝ)

theorem problem (h : 2 * (5 * x + 3 * Real.pi) = Q) : 4 * (10 * x + 6 * Real.pi + 2) = 4 * Q + 8 :=
by
  sorry

end problem_l361_361769


namespace isabella_eighth_test_score_l361_361417

/-- Given Isabella's test scores properties and ninth score, 
prove her eighth test score is 96. -/
theorem isabella_eighth_test_score :
  ∃ (scores : Fin 9 → ℤ),
    (∀ i, 88 ≤ scores i ∧ scores i ≤ 100) ∧
    (∃ (sum : ℤ), sum = ∑ i, scores i ∧ sum % 9 = 0) ∧
    scores 8 = 93 ∧ scores 7 = 96 := 
begin
  sorry
end

end isabella_eighth_test_score_l361_361417


namespace closest_point_on_line_l361_361677

open_locale real_inner_product_space

def point (x y z : ℝ) := (x, y, z)

def line_point (s : ℝ) : ℝ × ℝ × ℝ := 
  (3 - 3 * s, 4 + 9 * s, 2 - 4 * s)

def dist_to_point (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2)

theorem closest_point_on_line : 
    ∃ s : ℝ, 
    dist_to_point (3 - 3 * s, 4 + 9 * s, 2 - 4 * s) (1, 2, 3) = 
    dist_to_point (3 - 3 * (-8 / 53), 4 + 9 * (-8 / 53), 2 - 4 * (-8 / 53)) (1, 2, 3) :=
begin
    use (-8 / 53),
    sorry
end

end closest_point_on_line_l361_361677


namespace sum_of_row_and_column_products_not_zero_l361_361001

theorem sum_of_row_and_column_products_not_zero :
  ∀ (a : Fin 25 → Fin 25 → Int), 
    (∀ (i j : Fin 25), a i j = 1 ∨ a i j = -1) →
    let row_product (i : Fin 25) := ∏ j, a i j in
    let column_product (j : Fin 25) := ∏ i, a i j in
    (∑ i, row_product i + ∑ j, column_product j) ≠ 0 :=
begin
  intros a h,
  let row_product := λ i : Fin 25, ∏ j, a i j,
  let column_product := λ j : Fin 25, ∏ i, a i j,
  -- proof goes here
  sorry
end

end sum_of_row_and_column_products_not_zero_l361_361001


namespace determine_color_sum_or_product_l361_361960

theorem determine_color_sum_or_product {x : ℕ → ℝ} (h_distinct: ∀ i j : ℕ, i < j → x i < x j) (x_pos : ∀ i : ℕ, x i > 0) :
  ∃ c : ℕ → ℝ, (∀ i : ℕ, c i > 0) ∧
  (∀ i j : ℕ, i < j → (∃ r1 r2 : ℕ, (r1 ≠ r2) ∧ (c r1 + c r2 = x₆₄ + x₆₃) ∧ (c r1 * c r2 = x₆₄ * x₆₃))) :=
sorry

end determine_color_sum_or_product_l361_361960


namespace transport_cost_6725_l361_361482

variable (P : ℝ) (T : ℝ)

theorem transport_cost_6725
  (h1 : 0.80 * P = 17500)
  (h2 : 1.10 * P = 24475)
  (h3 : 17500 + T + 250 = 24475) :
  T = 6725 := 
sorry

end transport_cost_6725_l361_361482


namespace log_sqrt2_128sqrt2_eq_fifteen_l361_361264

theorem log_sqrt2_128sqrt2_eq_fifteen (a b : ℝ) (ha : a = sqrt 2) (hb : b = 128 * sqrt 2) : 
  Real.logBase a b = 15 := by
  sorry

end log_sqrt2_128sqrt2_eq_fifteen_l361_361264


namespace arithmetic_prog_n_value_l361_361250

theorem arithmetic_prog_n_value (x : ℝ) (a : ℕ → ℝ)
    (h1 : a 1 = x - 1)
    (h2 : a 2 = x^2 - 1)
    (h3 : a 3 = x^3 - 1)
    (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) :
    ∃ n : ℕ, n >= 1 ∧ a n = 2 * x^2 + 2 * x - 3 :=
by
  have d := x^2 - x -- from h2 and h1
  have a_n : ∀ n, a n = x - 1 + (n - 1) * d := sorry
  use 3 -- since the problem's conclusion is n = 3
  split
  · linarith -- n >= 1
  · rw [a_n 3]
    sorry

end arithmetic_prog_n_value_l361_361250


namespace at_least_two_sum_to_multiple_of_ten_l361_361293

theorem at_least_two_sum_to_multiple_of_ten :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 31) → (11 ≤ S.card) → ∃ (a b ∈ S), (a + b) % 10 = 0 :=
by
  sorry

end at_least_two_sum_to_multiple_of_ten_l361_361293


namespace sequence_proof_l361_361738

noncomputable def f (x : ℝ) (a : ℝ) := (1 / 2) * a^x

def a_n (n : ℕ) := 2 * (1 / 3)^n

def S_n (n : ℕ) := n^2

def T_n (n : ℕ) : ℝ := (1 / 2) * (1 - 1 / (2 * n + 1))

def sequence (n : ℕ) : ℝ := 1 / ((2 * n - 1) * (2 * n + 1))

theorem sequence_proof (a c : ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) :
  (f (1 : ℝ) a = 1 / 6) →
  (∃ c > 0, ∀ n, ∑ i in finset.range (n + 1), a_n i = c - f (n : ℝ) a) →
  (b 0 > 0 ∧ b 0 = 2 * c) →
  (∀ n ≥ 2, sqrt (S n) = sqrt (S (n - 1)) + 1) →
  (∀ n, T n = (1 / 2) * (1 - 1 / (2 * n + 1))) →
  (a_n (1 : ℕ) = 2 * (1 / 3)) ∧
  (∃ n > 0, T n > 1000 / 2009) :=
sorry

end sequence_proof_l361_361738


namespace hyperbola_equation_l361_361752

theorem hyperbola_equation (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (h_hyperbola : ∀ x y, (x^2 / a^2 - y^2 / b^2 = 1)
    (P : ℝ × ℝ) (hP_on_branch : P.1^2 / a^2 - P.2^2 / b^2 = 1 ∧ P.1 > a)
    (incircle_inters_x : P.1 = 1)
    (h_symmetric : (P.2 - 0) / (P.1 + c) = a / b): (∀ x y, x^2 - y^2 / 4 = 1) :=
by
  sorry

end hyperbola_equation_l361_361752


namespace proof_num_different_n_values_eq_10_l361_361867

def num_different_n_values_eq_10 : Prop :=
  let integer_pairs := [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6),
                        (-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)]
  ∧ ∀ (x y : ℤ), (x, y) ∈ integer_pairs → x * y = 36 ∧ n = x + y →
  ∃! (n : ℤ), ∃ (x y : ℤ), (x, y) ∈ integer_pairs ∧ 
  let distinct_values := (integer_pairs.map (λ p, p.1 + p.2)).to_finset in
  distinct_values.card = 10

-- Placeholder for the proof
theorem proof_num_different_n_values_eq_10 : num_different_n_values_eq_10 := 
by sorry

end proof_num_different_n_values_eq_10_l361_361867


namespace systematic_sampling_student_numbers_l361_361193

theorem systematic_sampling_student_numbers 
  (total_students : ℕ) (sample_size : ℕ) (sampled_students : list ℕ) 
  (interval : ℕ) (h_total : total_students = 55) 
  (h_size : sample_size = 5) 
  (h_sampled : sampled_students = [3, 25, 47]) 
  (h_interval : interval = total_students / sample_size) : 
  14 ∈ sampled_students ∧ 36 ∈ sampled_students := 
  sorry

end systematic_sampling_student_numbers_l361_361193


namespace john_total_jury_duty_days_l361_361025

-- Definitions based on the given conditions
def jury_selection_days : ℕ := 2
def trial_length_multiplier : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_in_a_day : ℕ := 24

-- Define the total days John spends on jury duty
def total_days_on_jury_duty : ℕ :=
  jury_selection_days +
  (trial_length_multiplier * jury_selection_days) +
  (deliberation_days * deliberation_hours_per_day) / hours_in_a_day

-- The theorem to prove
theorem john_total_jury_duty_days : total_days_on_jury_duty = 14 := by
  sorry

end john_total_jury_duty_days_l361_361025


namespace min_value_of_n_minus_m_plus_2_sqrt_2_l361_361780

-- Define the function f such that f(x) = 1/3 * x^3 - x for x >= 0
def f (x : ℝ) : ℝ := if x >= 0 then (1/3) * x^3 - x else 0

-- Define that a point (m, n) lies on the graph of f
def point_on_graph (m n : ℝ) : Prop := n = f m

-- Define the function g(m) = 1/3 * m^3 - 2m + 2 * sqrt 2
def g (m : ℝ) : ℝ := (1/3) * m^3 - 2 * m + 2 * Real.sqrt 2

-- Statement to be proven
theorem min_value_of_n_minus_m_plus_2_sqrt_2 :
  ∃ m : ℝ, ∃ n : ℝ, point_on_graph m n ∧ ∀ x ≥ 0, g x ≥ (g (Real.sqrt 2)) :=
  sorry

end min_value_of_n_minus_m_plus_2_sqrt_2_l361_361780


namespace max_band_members_l361_361542

theorem max_band_members 
  (m : ℤ)
  (h1 : 30 * m % 31 = 7)
  (h2 : 30 * m < 1500) : 
  30 * m = 720 :=
sorry

end max_band_members_l361_361542


namespace suff_but_not_nec_for_q_l361_361714

-- Define the conditions
def p (m : ℝ) : Prop := ∀ (x : ℝ), x > 0 → monotone_on (λ x, (m^2 - m - 1) * x^m) (set.Ioi 0)
def q (m : ℝ) : Prop := abs (m - 2) < 1

-- The proof problem
theorem suff_but_not_nec_for_q : (∃ (m : ℝ), p m) → (∃ (m : ℝ), q m) ∧ ¬((∃ (m : ℝ), q m) → (∃ (m : ℝ), p m)) := 
sorry

end suff_but_not_nec_for_q_l361_361714


namespace num_solutions_l361_361517

noncomputable def z_count (z : ℂ) : Prop :=
  (z + (1 / z)).im = 0 ∧ abs(z - 2) = real.sqrt 2

theorem num_solutions : 
  (∃ z1 z2 z3 z4 : ℂ, z_count z1 ∧ z_count z2 ∧ z_count z3 ∧ z_count z4 ∧ 
   (z1 ≠ z2 ∧ z1 ≠ z3 ∧ z1 ≠ z4 ∧ z2 ≠ z3 ∧ z2 ≠ z4 ∧ z3 ≠ z4)) ∧
  (∀ z : ℂ, z_count z → z = z1 ∨ z = z2 ∨ z = z3 ∨ z = z4) :=
sorry

end num_solutions_l361_361517


namespace log_ratio_condition_l361_361790

-- Definitions of the conditions
variables (a1 q : ℝ)
variables (a : ℕ → ℝ)
variables (n : ℕ)

-- Condition 1: Geometric sequence with common ratio q, and positive terms
def is_geometric_sequence := ∀ n : ℕ, a n = a1 * (q ^ n)

-- Condition 2: -6, q^2, 14 is an arithmetic sequence
def arithmetic_seq_condition := 2 * q^2 = -6 + 14

-- Target: Logarithmic ratio condition
theorem log_ratio_condition 
  (h1 : is_geometric_sequence a1 q a)
  (h2 : arithmetic_seq_condition q) : 
  log 2 ((a 2 + a 3) / (a 0 + a 1)) = 2 :=
sorry

end log_ratio_condition_l361_361790


namespace find_first_parallel_side_length_l361_361274

noncomputable def length_of_first_parallel_side
  (x : ℝ) (a : ℝ) (d : ℝ) (area : ℝ) : Prop :=
  area = (1 / 2) * (x + a) * d

theorem find_first_parallel_side_length :
  ∃ x : ℝ, length_of_first_parallel_side x 18 5 95 ∧ x = 20 :=
by
  use 20
  unfold length_of_first_parallel_side
  have h: 95 = (1 / 2) * (20 + 18) * 5 := by norm_num
  split
  { exact h}
  { simp }

end find_first_parallel_side_length_l361_361274


namespace cost_of_1000_gums_in_dollars_l361_361921

theorem cost_of_1000_gums_in_dollars :
  let cost_per_piece_in_cents := 1
  let pieces := 1000
  let cents_per_dollar := 100
  ∃ cost_in_dollars : ℝ, cost_in_dollars = (cost_per_piece_in_cents * pieces) / cents_per_dollar :=
sorry

end cost_of_1000_gums_in_dollars_l361_361921


namespace probability_on_duty_saturday_l361_361202

-- Defining the conditions
def on_duty_sunday : Prop := True -- The person is on duty on Sunday night

-- The proof problem statement as a Lean theorem
theorem probability_on_duty_saturday :
  on_duty_sunday → (1 / 6 : ℚ) = 1 / 6 :=
by
  intro h
  rw h
  sorry

end probability_on_duty_saturday_l361_361202


namespace clerical_percentage_is_correct_l361_361576

noncomputable def clerical_percentage_after_reduction (total_employees clerical_fraction clerical_reduction: ℕ) : ℚ :=
  let initial_clerical := clerical_fraction * total_employees
  let remaining_clerical := (1 - clerical_reduction) * initial_clerical
  let total_remaining := total_employees - (clerical_reduction * initial_clerical)
  (remaining_clerical / total_remaining) * 100

theorem clerical_percentage_is_correct :
  clerical_percentage_after_reduction 3600 (1/6) (1/3) ≈ 11.76 := 
by
  sorry

end clerical_percentage_is_correct_l361_361576


namespace dogs_meet_at_center_dogs_paths_no_intersect_before_meeting_dogs_paths_length_l361_361291

noncomputable def meet_time (side_length speed : ℝ) : ℝ := side_length / speed

theorem dogs_meet_at_center (side_length speed : ℝ) (h_side : side_length = 100) (h_speed : speed = 10) :
  meet_time side_length speed = 10 :=
by
  rw [h_side, h_speed, meet_time]
  ring

theorem dogs_paths_no_intersect_before_meeting (side_length speed : ℝ) (h_side : side_length = 100) (h_speed : speed = 10) :
  true :=
by
  trivial

theorem dogs_paths_length (side_length speed : ℝ) (h_side : side_length = 100) (h_speed : speed = 10) :
  side_length = 100 :=
by
  rw h_side
  trivial

end dogs_meet_at_center_dogs_paths_no_intersect_before_meeting_dogs_paths_length_l361_361291


namespace part_I_part_II_part_III_l361_361735

/-- Axiom stating that f is an even function -/
axiom f_even : ∀ x : ℝ, f (-x) = f x

/-- Definition of f(x) when x >= 0 -/
def f (x : ℝ) : ℝ := if x >= 0 then (1 / 2)^x else f (-x)

/-- Statement (I): Prove that f(-1) = 1 / 2 given the conditions -/
theorem part_I : f (-1) = 1 / 2 := by
  sorry

/-- Statement (II): Prove that the range of f(x) is (0, 1] given the conditions -/
theorem part_II : set.range f = set.Ioo 0 1 ∪ {1} := by
  sorry

/-- Definition of g(x) -/
def g (x : ℝ) (a : ℝ) : ℝ := real.sqrt (-x^2 + (a - 1) * x + a)

/-- Domain condition of g(x) -/
def domain_g (a : ℝ) : set ℝ := {x | -x^2 + (a - 1) * x + a ≥ 0}

/-- Constraint that A = (0, 1] -/
def A := set.Ioo 0 1 ∪ {1}

/-- Constraint that A ⊆ B -/
def B (a : ℝ) := domain_g a

/-- Statement (III): Prove that a ≥ 1 given the inclusion A ⊆ B(a) -/
theorem part_III (a : ℝ) : A ⊆ B a → a ≥ 1 := by
  sorry

end part_I_part_II_part_III_l361_361735


namespace log_eq_seven_half_l361_361260

theorem log_eq_seven_half (h1 : 64 = 4^3) (h2 : sqrt 4 = 4^(1/2)) :
  log 4 (64 * sqrt 4) = 7/2 :=
sorry

end log_eq_seven_half_l361_361260


namespace satisfaction_and_participation_l361_361629

variable vote_counts1 : List (ℕ × ℕ) := [(5, 130), (4, 105), (3, 61), (2, 54), (1, 33)]
variable vote_counts2 : List (ℕ × ℕ) := [(5, 78), (4, 174), (3, 115), (2, 81), (1, 27)]
variable vote_counts3 : List (ℕ × ℕ) := [(5, 95), (4, 134), (3, 102), (2, 51), (1, 31)]

def mean_satisfaction_rating (vote_counts : List (ℕ × ℕ)) : ℚ :=
  let numerator := vote_counts.map (fun (rating, count) => rating * count).sum
  let denominator := vote_counts.map (fun (_, count) => count).sum
  numerator / denominator

def overall_participation_count (vote_counts1 vote_counts2 vote_counts3 : List (ℕ × ℕ)) : ℕ :=
  (vote_counts1 ++ vote_counts2 ++ vote_counts3).map (fun (_, count) => count).sum

theorem satisfaction_and_participation :
  mean_satisfaction_rating vote_counts1 = 1394 / 383 ∧
  mean_satisfaction_rating vote_counts2 = 1620 / 475 ∧
  mean_satisfaction_rating vote_counts3 = 1450 / 413 ∧
  overall_participation_count vote_counts1 vote_counts2 vote_counts3 = 1271 :=
by
  sorry

end satisfaction_and_participation_l361_361629


namespace prob_greater_first_card_l361_361896

theorem prob_greater_first_card :
  (∃ (draw : fin 5 → fin 5 → bool), 
    let count := (Sum (λ (i : fin 5), Sum (λ (j : fin 5), if j.val < i.val then 1 else 0)))
    count = 10 
  ) → p = (2 : ℚ) / 5 :=
by
  sorry

end prob_greater_first_card_l361_361896


namespace athlete_positions_l361_361388

theorem athlete_positions {A B C D E : Type} 
  (h1 : A = 1 ∧ B = 2 ∧ C = 3 ∧ D = 4 ∧ E = 5)
  (h2 : C = 1 ∧ E = 2 ∧ A = 3 ∧ B = 4 ∧ D = 5)
  (h3 : (A = 1 ∧ B = 2 ∧ C = 3 ∧ ¬(D = 4 ∨ E = 5)) ∨ 
       (A = 1 ∧ ¬(B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)) ∨ 
       (B = 2 ∧ C = 3 ∧ ¬(A = 1 ∨ D = 4 ∨ E = 5)) ∨ 
       (¬(A = 1 ∨ B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)))
  (h4 : (C = 1 ∧ ¬(A = 3 ∨ B = 4 ∨ D = 5)) ∨ 
       (A = 3 ∧ ¬(C = 1 ∨ B = 4 ∨ D = 5)) ∨ 
       (B = 4 ∧ ¬(C = 1 ∨ A = 3 ∨ D = 5)))
  : (C = 1 ∧ B = 2 ∧ A = 3 ∧ D = 4 ∧ E = 5) :=
by sorry

end athlete_positions_l361_361388


namespace prove_positions_l361_361375

def athlete : Type := { A : ℕ, B : ℕ, C : ℕ, D : ℕ, E : ℕ }

def first_prediction (x : athlete) : athlete :=
  { A := 1, B := 2, C := 3, D := 4, E := 5 }

def second_prediction (x : athlete) : athlete :=
  { A := 3, B := 4, C := 1, D := 5, E := 2 }

def actual_positions : athlete :=
  { A := 3, B := 2, C := 1, D := 4, E := 5 }

theorem prove_positions :
  (λ x, first_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 3 ∧
  (λ x, second_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 2 ∧
  actual_positions.A = 3 ∧ actual_positions.B = 2 ∧ actual_positions.C = 1 ∧ actual_positions.D = 4 ∧ actual_positions.E = 5 := 
  sorry

end prove_positions_l361_361375


namespace geom_sequence_common_ratio_l361_361314

theorem geom_sequence_common_ratio (q a₁ : ℝ) (hq : 0 < q) 
  (h : (a₁ * q^2) * (a₁ * q^8) = 2 * (a₁ * q^4)^2) :
  q = real.sqrt 2 :=
sorry

end geom_sequence_common_ratio_l361_361314


namespace total_days_on_jury_duty_l361_361030

-- Definitions based on conditions
def jurySelectionDays := 2
def juryDeliberationDays := 6
def deliberationHoursPerDay := 16
def trialDurationMultiplier := 4

-- Calculate the total number of hours spent in deliberation
def totalDeliberationHours := juryDeliberationDays * 24

-- Calculate the number of days spent in deliberation based on hours per day
def deliberationDays := totalDeliberationHours / deliberationHoursPerDay

-- Calculate the trial days based on trial duration multiplier
def trialDays := jurySelectionDays * trialDurationMultiplier

-- Calculate the total days on jury duty
def totalJuryDutyDays := jurySelectionDays + trialDays + deliberationDays

theorem total_days_on_jury_duty : totalJuryDutyDays = 19 := by
  sorry

end total_days_on_jury_duty_l361_361030


namespace find_k_value_l361_361758

noncomputable def point_on_y1 (x : ℝ) : ℝ := 2 / x
noncomputable def point_on_y2 (x k : ℝ) : ℝ := k / x
noncomputable def dist_from_origin (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

theorem find_k_value :
  ∃ (k : ℝ),
  let A := (1, point_on_y1 1) in
  let B := (3, point_on_y2 3 k) in
  dist_from_origin B.1 B.2 = 3 * dist_from_origin A.1 A.2 →
  k = 18 :=
by
  sorry

end find_k_value_l361_361758


namespace smallest_positive_period_f_l361_361657

-- Define the function f
def f (x : ℝ) : ℝ := (sin (2 * x)) / (1 - 2 * sin (2 * (x / 2 - π / 4))) * (1 + 3 * tan x)

-- Theorem statement for the smallest positive period of f(x)
theorem smallest_positive_period_f : ∃ T > 0, (∀ x, f(x + T) = f(x)) ∧ (∀ ε > 0, ε < T → ∃ x, f(x + ε) ≠ f(x)) :=
sorry

end smallest_positive_period_f_l361_361657


namespace average_score_all_classes_l361_361553

open Real

variables (X Y Z : ℕ)
variable (average_score_all_three_classes : ℝ)

-- Conditions from the problem
def avg_score_class_x := 83
def avg_score_class_y := 76
def avg_score_class_z := 85
def avg_score_x_y := 79
def avg_score_y_z := 81

-- Conditions expressed as equations
axiom avg_x_y_eq : (83 * X + 76 * Y) / (X + Y) = 79
axiom avg_y_z_eq : (76 * Y + 85 * Z) / (Y + Z) = 81

-- The goal to prove
theorem average_score_all_classes :
  (83 * X + 76 * Y + 85 * Z) / (X + Y + Z) = 81.5 :=
sorry

end average_score_all_classes_l361_361553


namespace prove_positions_l361_361378

def athlete : Type := { A : ℕ, B : ℕ, C : ℕ, D : ℕ, E : ℕ }

def first_prediction (x : athlete) : athlete :=
  { A := 1, B := 2, C := 3, D := 4, E := 5 }

def second_prediction (x : athlete) : athlete :=
  { A := 3, B := 4, C := 1, D := 5, E := 2 }

def actual_positions : athlete :=
  { A := 3, B := 2, C := 1, D := 4, E := 5 }

theorem prove_positions :
  (λ x, first_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 3 ∧
  (λ x, second_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 2 ∧
  actual_positions.A = 3 ∧ actual_positions.B = 2 ∧ actual_positions.C = 1 ∧ actual_positions.D = 4 ∧ actual_positions.E = 5 := 
  sorry

end prove_positions_l361_361378


namespace trajectory_of_P_is_parabola_l361_361005

-- Provide necessary definitions for points, lines, distances, and plane.
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def dist_point_to_line (P : Point3D) (L : Line3D) : ℝ := sorry
def line_perpendicular_to_plane (L : Line3D) (Pl : Plane3D) : Prop := sorry

-- Define the line and plane entities involved.
def B := Point3D (0) (0) (0)
def C := Point3D (1) (0) (0)
def B1 := Point3D (0) (0) (1)
def C1 := Point3D (1) (0) (1)

def line_BC : Line3D := sorry -- Definition of BC line
def line_C1D1 : Line3D := sorry -- Definition of C1D1 line

def plane_BB1C1C : Plane3D := sorry -- The plane BB1C1C

-- The point P moving within the side BB1C1C
variable P : Point3D

-- Lean statement of the proof problem
theorem trajectory_of_P_is_parabola :
  P ∈ plane_BB1C1C →
  dist_point_to_line P line_BC = dist_point_to_line P line_C1D1 →
  is_parabola (trajectory_of_P) :=
begin
  sorry
end

end trajectory_of_P_is_parabola_l361_361005


namespace omega_values_l361_361745

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * sin (ω * x + (5 * Real.pi / 12)) + cos (ω * x - (Real.pi / 12))

theorem omega_values (ω : ℝ) :
  (ω = 24 + 1/12 ∨ ω = 26 + 1/12) ↔ 
  (∀ (x : ℝ), f ω x = f ω (Real.pi) / 3 ↔ (0 < x ∧ x < Real.pi / 12 ∧ 
  f(ω, x) = 3 * sqrt 3 / 2 ∧ 
  ω > 0) :=
sorry

end omega_values_l361_361745


namespace game_ends_and_last_numbers_depend_on_start_l361_361469
-- Given that there are three positive integers a, b, c initially.
variables (a b c : ℕ)
-- Assume that a, b, and c are greater than zero.
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Define the gcd of the three numbers.
def g := gcd (gcd a b) c

-- Define the game step condition.
def step_condition (a b c : ℕ): Prop := a > gcd b c

-- Define the termination condition.
def termination_condition (a b c : ℕ): Prop := ¬ step_condition a b c

-- The main theorem
theorem game_ends_and_last_numbers_depend_on_start (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ n, ∃ b' c', termination_condition n b' c' ∧
  n = g ∧ b' = g ∧ c' = g :=
sorry

end game_ends_and_last_numbers_depend_on_start_l361_361469


namespace man_swim_downstream_distance_l361_361197

-- Define the given conditions
def t_d : ℝ := 6
def t_u : ℝ := 6
def d_u : ℝ := 18
def V_m : ℝ := 4.5

-- The distance the man swam downstream
def distance_downstream : ℝ := 36

-- Prove that given the conditions, the man swam 36 km downstream
theorem man_swim_downstream_distance (V_c : ℝ) :
  (d_u / (V_m - V_c) = t_u) →
  (distance_downstream / (V_m + V_c) = t_d) →
  distance_downstream = 36 :=
by
  sorry

end man_swim_downstream_distance_l361_361197


namespace average_age_new_students_l361_361919

theorem average_age_new_students (O A_O N : ℕ) (A_O_eq : A_O = 40) (O_eq : O = 12) (N_eq : N = 12) (new_avg_eq : (O * A_O + N * 32) / (O + N) = 36) :
  N * 32 = 384 :=
by
  -- Original average age of the class
  have h1 : O = 12 := O_eq,
  have h2 : A_O = 40 := A_O_eq,
  -- The condition given in the problem (new average)
  have h3 : new_avg_eq = ((O * A_O + N * 32) / (O + N) = 36),
  have h4 : N = 12 := N_eq,
  sorry  

end average_age_new_students_l361_361919


namespace feeding_times_per_day_l361_361487

theorem feeding_times_per_day (p f d : ℕ) (h₁ : p = 7) (h₂ : f = 105) (h₃ : d = 5) : 
  (f / d) / p = 3 := by
  sorry

end feeding_times_per_day_l361_361487


namespace five_pm_is_seventeen_hours_ten_pm_is_twenty_two_hours_time_difference_is_forty_minutes_l361_361190

-- Define what "5 PM" and "10 PM" mean in hours
def five_pm: ℕ := 17
def ten_pm: ℕ := 22

-- Define function for converting from PM to 24-hour time
def pm_to_hours (n: ℕ): ℕ := n + 12

-- Define the times in minutes for comparison
def time_16_40: ℕ := 16 * 60 + 40
def time_17_20: ℕ := 17 * 60 + 20

-- Define the differences in minutes
def minutes_passed (start end_: ℕ): ℕ := end_ - start

-- Prove the equivalences
theorem five_pm_is_seventeen_hours: pm_to_hours 5 = five_pm := by 
  unfold pm_to_hours
  unfold five_pm
  rfl

theorem ten_pm_is_twenty_two_hours: pm_to_hours 10 = ten_pm := by 
  unfold pm_to_hours
  unfold ten_pm
  rfl

theorem time_difference_is_forty_minutes: minutes_passed time_16_40 time_17_20 = 40 := by 
  unfold time_16_40
  unfold time_17_20
  unfold minutes_passed
  rfl

#check five_pm_is_seventeen_hours
#check ten_pm_is_twenty_two_hours
#check time_difference_is_forty_minutes

end five_pm_is_seventeen_hours_ten_pm_is_twenty_two_hours_time_difference_is_forty_minutes_l361_361190


namespace proof_problem_l361_361740

-- Defining the equation for our context
def quadratic_eq (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c = 0

-- The conditions as given in the problem
def conditions (m n : ℝ) : Prop :=
  ∃ (roots : List ℝ), 
    (roots = [1/4, _, _, 7/4] ∧ 
     roots.sorted ∧ 
     ∀ x ∈ roots, quadratic_eq 1 (-2 : ℝ) (m : ℝ) x ∨ quadratic_eq 1 (-2 : ℝ) (n : ℝ) x)

-- The goal to prove |m - n| = 1/2
theorem proof_problem (m n : ℝ) 
  (h1 : conditions m n) : |m - n| = 1/2 :=
sorry

end proof_problem_l361_361740


namespace perp_line_to_plane_l361_361686

variables {α β : Plane}
variable {m : Line}

-- Conditions in the problem
axiom perp_alpha : m ⊥ α
axiom parallel_alpha_beta : α ∥ β

-- The theorem to prove
theorem perp_line_to_plane : m ⊥ β := 
by 
  sorry

end perp_line_to_plane_l361_361686


namespace BC_length_l361_361411

-- Define terms and conditions
variables (A B C D : Type) [Fig : Trapezoid ABCD]
variables (h1 : Parallel AB CD) (h2 : Perpendicular AC CD)
variables (h3 : CD = 15) (h4 : tan D = 2) (h5 : tan B = 3)

-- Define the type of the Lean statement
theorem BC_length : BC = 10 * Real.sqrt 10 :=
by
  sorry

end BC_length_l361_361411


namespace no_possible_k_for_prime_roots_l361_361634

theorem no_possible_k_for_prime_roots :
  ¬ ∃ (k : ℕ), (∃ (p q : ℕ), prime p ∧ prime q ∧ (x^2 - 65 * x + k) = (x - p) * (x - q)) :=
sorry

end no_possible_k_for_prime_roots_l361_361634


namespace number_of_subsets_of_M_l361_361942

open Set

noncomputable def M (a : ℝ) : Set ℝ :=
  {x | x^2 - 3 * x - a^2 + 2 = 0}

theorem number_of_subsets_of_M (a : ℝ) : ∃ (n : ℕ), n = 2^2 :=
  have h1 : ∃ x y : ℝ, x ≠ y ∧ x ∈ M a ∧ y ∈ M a :=
    -- implicitly use the fact that discriminant is always positive here
    sorry,
  have h2 : ∀ x y : ℝ, x ∈ M a → y ∈ M a → x = y → False := 
    -- implicitly use the fact that discriminant is always positive here
    sorry,
  exists.intro (2^2)
  (by sorry)

end number_of_subsets_of_M_l361_361942


namespace coverable_with_three_cell_strips_l361_361946

theorem coverable_with_three_cell_strips (a b c : ℕ) : 
  (∃ (A B C : ℕ), A = 3 * a ∧ B = 3 * b ∧ C = 3 * c ∧
  (A % 3 = 0 ∧ B % 3 = 0 ∧ C % 3 = 0) → 
  (∃ (facing_unit_strips_subset : set (ℕ × ℕ × ℕ)), 
  (∀ x ∈ facing_unit_strips_subset, (∃ (x' : ℕ), x' < 3 ∧ x' ∈ facing_unit_strips_subset) ∧ ∀ x y ∈ facing_unit_strips_subset, x ≠ y →
  x' ∉ facing_unit_strips_subset) → 
  (∀ i j k < 3, (i + j + k = 3)))) ↔ 
  ((a % 3 = 0 ∧ b % 3 = 0) ∨ (b % 3 = 0 ∧ c % 3 = 0) ∨ (c % 3 = 0 ∧ a % 3 = 0)) :=
sorry

end coverable_with_three_cell_strips_l361_361946


namespace smallest_positive_b_l361_361655

theorem smallest_positive_b (b : ℕ) : 
  (b % 3 = 2) ∧ 
  (b % 4 = 3) ∧ 
  (b % 5 = 4) ∧ 
  (b % 6 = 5) ↔ 
  b = 59 :=
by
  sorry

end smallest_positive_b_l361_361655


namespace complex_power_of_sum_l361_361311

theorem complex_power_of_sum (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end complex_power_of_sum_l361_361311


namespace fraction_of_acid_in_third_flask_l361_361138

def mass_of_acid_first_flask := 10
def mass_of_acid_second_flask := 20
def mass_of_acid_third_flask := 30
def mass_of_acid_first_flask_with_water (w : ℝ) := mass_of_acid_first_flask / (mass_of_acid_first_flask + w) = 1 / 20
def mass_of_acid_second_flask_with_water (W w : ℝ) := mass_of_acid_second_flask / (mass_of_acid_second_flask + (W - w)) = 7 / 30
def mass_of_acid_third_flask_with_water (W : ℝ) := mass_of_acid_third_flask / (mass_of_acid_third_flask + W)

theorem fraction_of_acid_in_third_flask (W w : ℝ) (h1 : mass_of_acid_first_flask_with_water w) (h2 : mass_of_acid_second_flask_with_water W w) :
  mass_of_acid_third_flask_with_water W = 21 / 200 :=
by
  sorry

end fraction_of_acid_in_third_flask_l361_361138


namespace gain_percent_calculation_l361_361175

variable (CP SP : ℝ)
variable (gain gain_percent : ℝ)

theorem gain_percent_calculation
  (h₁ : CP = 900) 
  (h₂ : SP = 1180)
  (h₃ : gain = SP - CP)
  (h₄ : gain_percent = (gain / CP) * 100) :
  gain_percent = 31.11 := by
sorry

end gain_percent_calculation_l361_361175


namespace age_of_15th_student_l361_361514

theorem age_of_15th_student (avg15 : ℕ) (avg7_first : ℕ) (avg7_second : ℕ) : 
  (avg15 = 15) → 
  (avg7_first = 14) → 
  (avg7_second = 16) →
  (let T := 15 * avg15 in
   let sum_first := 7 * avg7_first in
   let sum_second := 7 * avg7_second in
   T - (sum_first + sum_second) = 15) :=
by
  intros h1 h2 h3
  sorry

end age_of_15th_student_l361_361514


namespace driver_change_probability_zero_initial_driver_initial_coins_needed_l361_361930

/-- Part (a) Proof Statement -/
theorem driver_change_probability_zero_initial:
  (probability that the driver can give change to each of the 15 passengers paying with a 100 ruble bill when starting with zero coins is approximately 0.196) :=
sorry

/-- Part (b) Proof Statement -/
theorem driver_initial_coins_needed (min_initial_coins : ℕ) :
  (probability that the driver can give change to each of the 15 passengers paying with a 100 ruble bill is at least 0.95 when starting with min_initial_coins coins) → min_initial_coins = 275 :=
sorry

end driver_change_probability_zero_initial_driver_initial_coins_needed_l361_361930


namespace angle_between_a_and_a_add_b_l361_361468

variables {V : Type*} [inner_product_space ℝ V]
open_locale real_inner_product_space

variables (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h : ‖a‖ = ‖b‖ ∧ ‖a‖ = ‖a - b‖ )

theorem angle_between_a_and_a_add_b :
  real.angle a (a + b) = real.pi / 6 :=
sorry

end angle_between_a_and_a_add_b_l361_361468


namespace cookies_left_for_neil_l361_361467

variable (total_cookies : ℕ)
variable (first_fraction : ℚ) (second_fraction : ℚ) (third_fraction : ℚ)
variable (cookies_left : ℕ)

def cookies_after_first (total : ℕ) (frac : ℚ) : ℕ :=
  total - (frac * total).toInt

def cookies_after_second (total : ℕ) (frac : ℚ) : ℕ :=
  total - (frac * total).toInt

def cookies_after_third (total : ℕ) (frac : ℚ) : ℕ :=
  total - (frac * total).toInt

theorem cookies_left_for_neil :
  total_cookies = 60 →
  first_fraction = 1/3 →
  second_fraction = 1/4 →
  third_fraction = 2/5 →
  let after_first := cookies_after_first total_cookies first_fraction in
  let after_second := cookies_after_second after_first second_fraction in
  let after_third := cookies_after_third after_second third_fraction in
  after_third = 18 :=
by
  rintro rfl rfl rfl rfl
  have h1 : cookies_after_first 60 (1/3) = 40 := by simp [cookies_after_first, Int.ofNat_mul, Int.ofNat, Rat.ofInt_eq_mk]
  have h2 : cookies_after_second 40 (1/4) = 30 := by simp [cookies_after_second, Int.ofNat_mul, Int.ofNat, Rat.ofInt_eq_mk]
  have h3 : cookies_after_third 30 (2/5) = 18 := by simp [cookies_after_third, Int.ofNat_mul, Int.ofNat, Rat.ofInt_eq_mk]
  simp [h1, h2, h3]
  sorry

end cookies_left_for_neil_l361_361467


namespace proof_l361_361866

-- Define the basic entities: lines and planes
variables (l m n : Type) (α β : Type)

-- Define the predicates for perpendicularity and parallelism
variables (Perp : α → β → Prop) (Sub : l → α → Prop) (Par : l → β → Prop)

-- Define the propositions
def prop1 : Prop := ∀(α β : Type) (l : Type), Perp α β → Perp l α → Par l β
def prop2 : Prop := ∀(α β : Type) (l : Type), Perp α β → Sub l α → Perp l β
def prop3 : Prop := ∀(l m n : Type), Perp l m → Perp m n → Par l n
def prop4 : Prop := ∀(m n : Type) (α β : Type), Perp m α → Par n β → Par α β → Perp m n

-- Define the correct answer as 1
def correctAnswer : Nat := 1

-- Define the number of correct propositions
def numCorrectProps : Nat := 
  (if prop1 then 1 else 0) +
  (if prop2 then 1 else 0) +
  (if prop3 then 1 else 0) +
  (if prop4 then 1 else 0)

-- The statement we need to prove
theorem proof : numCorrectProps = correctAnswer := sorry

end proof_l361_361866


namespace pascal_triangle_10_to_30_l361_361765

-- Definitions
def pascal_row_numbers (n : ℕ) : ℕ := n + 1

def total_numbers_up_to (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

-- Proof Statement
theorem pascal_triangle_10_to_30 :
  total_numbers_up_to 29 - total_numbers_up_to 9 = 400 := by
  sorry

end pascal_triangle_10_to_30_l361_361765


namespace lines_intersect_at_single_point_l361_361054

theorem lines_intersect_at_single_point :
  ∀ (ABC : Triangle) (I : Point) (I_A : Point) 
    (l_A l_B l_C : Line) (orth_BIC : Line) 
    (orth_BI_AC : Line) (orth_A_coincs : l_A = Line.join orth_BIC orth_BI_AC)
    (orth_CIB : Line) (orth_CI_CB : Line) (orth_B_coincs : l_B = Line.join orth_CIB orth_CI_CB)
    (orth_AIC : Line) (orth_AI_CB : Line) (orth_c_coincs : l_C = Line.join orth_AIC orth_AI_CB), 
  LinesIntersect l_A l_B l_C := 
by sorry

end lines_intersect_at_single_point_l361_361054


namespace sqrt_mul_sqrt_l361_361986

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361986


namespace find_some_number_l361_361346

-- Conditions on operations
axiom plus_means_mult (a b : ℕ) : (a + b) = (a * b)
axiom minus_means_plus (a b : ℕ) : (a - b) = (a + b)
axiom mult_means_div (a b : ℕ) : (a * b) = (a / b)
axiom div_means_minus (a b : ℕ) : (a / b) = (a - b)

-- Problem statement
theorem find_some_number (some_number : ℕ) :
  (6 - 9 + some_number * 3 / 25 = 5 ↔
   6 + 9 * some_number / 3 - 25 = 5) ∧
  some_number = 8 := by
  sorry

end find_some_number_l361_361346


namespace jack_jill_next_in_step_l361_361021

theorem jack_jill_next_in_step (stride_jack : ℕ) (stride_jill : ℕ) : 
  stride_jack = 64 → stride_jill = 56 → Nat.lcm stride_jack stride_jill = 448 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end jack_jill_next_in_step_l361_361021


namespace transformation_1_transformation_2_transformation_3_transformation_4_transformation_5_l361_361321

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4

theorem transformation_1 :
  ∀ (g : ℝ → ℝ),
  (∀ x, g(x) = 3 * x + 5) →
  g = (λ x, 3 * x + 5) :=
by
  intro g h
  funext x
  exact h x

theorem transformation_2 :
  ∀ (g : ℝ → ℝ),
  (∀ x, g(x) = -3 * x + 4) →
  g = (λ x, -3 * x + 4) :=
by
  intro g h
  funext x
  exact h x

theorem transformation_3 :
  ∀ (g : ℝ → ℝ),
  (∀ x, g(x) = -3 * x - 2) →
  g = (λ x, -3 * x - 2) :=
by
  intro g h
  funext x
  exact h x

theorem transformation_4 :
  ∀ (g : ℝ → ℝ),
  (∀ x, g(x) = (x + 4) / 3) →
  g = (λ x, (x + 4) / 3) :=
by
  intro g h
  funext x
  exact h x

theorem transformation_5 (a b : ℝ) :
  ∀ (g : ℝ → ℝ),
  (∀ x, g(x) = 3 * x + 2 * b - 6 * a - 4) →
  g = (λ x, 3 * x + 2 * b - 6 * a - 4) :=
by
  intro g h
  funext x
  exact h x

end transformation_1_transformation_2_transformation_3_transformation_4_transformation_5_l361_361321


namespace max_crosses_4x10_impossible_crosses_5x10_l361_361177

-- Definition for Part (a)
def maximum_crosses_in_grid_4x10 : ℕ := 30

theorem max_crosses_4x10 (n : ℕ) (h : ∀ r c : ℕ, r < 4 → c < 10 → odd ((4x10_grid r).count (c = X))) : n ≤ maximum_crosses_in_grid_4x10 :=
sorry

-- Definition for Part (b)
theorem impossible_crosses_5x10 (h : ∀ r c : ℕ, r < 5 → c < 10 → odd ((5x10_grid r).count (c = X))) : false :=
sorry

end max_crosses_4x10_impossible_crosses_5x10_l361_361177


namespace actual_positions_correct_l361_361365

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end actual_positions_correct_l361_361365


namespace acid_concentration_in_third_flask_l361_361135

theorem acid_concentration_in_third_flask 
    (w : ℚ) (W : ℚ) (hw : 10 / (10 + w) = 1 / 20) (hW : 20 / (20 + (W - w)) = 7 / 30) 
    (W_total : W = 256.43) : 
    30 / (30 + W) = 21 / 200 := 
by 
  sorry

end acid_concentration_in_third_flask_l361_361135


namespace find_m_l361_361762

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

-- Define the function to calculate m * a + b
def m_a_plus_b (m : ℝ) : ℝ × ℝ := (2 * m - 1, 3 * m + 2)

-- Define the vector a - 2 * b
def a_minus_2b : ℝ × ℝ := (4, -1)

-- Define the condition for parallelism
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The theorem that states the equivalence
theorem find_m (m : ℝ) (H : parallel (m_a_plus_b m) a_minus_2b) : m = -1/2 :=
by
  sorry

end find_m_l361_361762


namespace unique_integer_solution_l361_361271

theorem unique_integer_solution (m n : ℤ) :
  (m + n)^4 = m^2 * n^2 + m^2 + n^2 + 6 * m * n ↔ m = 0 ∧ n = 0 :=
by
  sorry

end unique_integer_solution_l361_361271


namespace circle_minor_arc_probability_l361_361204

noncomputable def probability_minor_arc_length_lt_one 
  (circle_circumference : ℝ) 
  (A B : ℝ) 
  (h_circumference : circle_circumference = 3) 
  (h_A : 0 ≤ A ∧ A < circle_circumference) 
  (h_B : 0 ≤ B ∧ B < circle_circumference) 
  : ℝ := 
  if abs (B - A) < 1 then sorry else sorry

theorem circle_minor_arc_probability 
  (circle_circumference : ℝ) 
  (h_circumference : circle_circumference = 3) 
  : (probability (λ (B : ℝ), B ∈ set.Ico 0 circle_circumference ∧ 
                  abs (B - 0) < 1 ∨ abs (circle_circumference - abs (B - 0)) < 1)) = 2 / 3 :=
sorry

end circle_minor_arc_probability_l361_361204


namespace distance_to_point_is_17_l361_361822

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem distance_to_point_is_17 :
  distance_from_origin 8 (-15) = 17 :=
by
  sorry

end distance_to_point_is_17_l361_361822


namespace sqrt_mul_sqrt_l361_361985

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361985


namespace sum_of_x_y_l361_361295

theorem sum_of_x_y (x y : ℝ) (h1 : 3 * x + 2 * y = 10) (h2 : 2 * x + 3 * y = 5) : x + y = 3 := 
by
  sorry

end sum_of_x_y_l361_361295


namespace actual_positions_correct_l361_361356

def athletes := {A, B, C, D, E}
def p1 : athletes → ℕ
| A := 1
| B := 2
| C := 3
| D := 4
| E := 5

def p2 : athletes → ℕ
| C := 1
| E := 2
| A := 3
| B := 4
| D := 5

def actual_positions : athletes → ℕ
| A := 3
| B := 2
| C := 1
| D := 4
| E := 5

theorem actual_positions_correct :
  (∑ a in athletes, ite (actual_positions a = p1 a) 1 0 = 3) ∧ 
  (∑ a in athletes, ite (actual_positions a = p2 a) 1 0 = 2) :=
by
  sorry

end actual_positions_correct_l361_361356


namespace sum_T_l361_361670

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * sin (x - π / 3) + sqrt 3 * cos x

-- Define a_n
def a (n : ℕ) : ℝ := (2 * n - 1) * π / 2

-- Define b_n
def b (n : ℕ) : ℝ := π^2 / (a n * a (n + 1))

-- Define T_n as the sum of the first n terms of b_n
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b k)

-- Theorem stating the sum T_n = 2 - 2/(2n+1)
theorem sum_T (n : ℕ) : T n = 2 - 2 / (2 * n + 1) :=
by
  sorry

end sum_T_l361_361670


namespace marco_new_cards_l361_361060

theorem marco_new_cards (total_cards : ℕ) (fraction_duplicates : ℕ) (fraction_traded : ℕ) : 
  total_cards = 500 → 
  fraction_duplicates = 4 → 
  fraction_traded = 5 → 
  (total_cards / fraction_duplicates) / fraction_traded = 25 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.div_div_eq_div_mul]
  norm_num

-- Since Lean requires explicit values, we use the following lemma to lead the theorem to concrete values:
lemma marco_new_cards_concrete : 
  (500 / 4) / 5 = 25 :=
marco_new_cards 500 4 5 rfl rfl rfl

end marco_new_cards_l361_361060


namespace full_price_tickets_revenue_l361_361587

theorem full_price_tickets_revenue (f h d p : ℕ) 
  (h1 : f + h + d = 200) 
  (h2 : f * p + h * (p / 2) + d * (2 * p) = 5000) 
  (h3 : p = 50) : 
  f * p = 4500 :=
by
  sorry

end full_price_tickets_revenue_l361_361587


namespace find_x_l361_361771

theorem find_x (x : ℝ) (h1 : sin (π / 2 - x) = -sqrt 3 / 2) (h2 : π < x ∧ x < 2 * π) : 
  x = 7 * π / 6 :=
  sorry

end find_x_l361_361771


namespace quadratic_roots_l361_361694

theorem quadratic_roots (m x1 x2 : ℝ) 
  (h1 : 2*x1^2 + 4*m*x1 + m = 0)
  (h2 : 2*x2^2 + 4*m*x2 + m = 0)
  (h3 : x1 ≠ x2)
  (h4 : x1^2 + x2^2 = 3/16) : 
  m = -1/8 := 
sorry

end quadratic_roots_l361_361694


namespace χ_not_div_by_4_l361_361951

def χ : ℕ → ℕ
| 0     := 0  -- Since sequences typically start from index 1, we need to handle index 0.
| 1     := 1
| (n+1) := χ n + χ ((n/2).nat_ceil)

theorem χ_not_div_by_4 (n : ℕ) : ¬ (4 ∣ χ n) := by
  sorry

end χ_not_div_by_4_l361_361951


namespace correctFinishingOrder_l361_361402

-- Define the types of athletes
inductive Athlete
| A | B | C | D | E

-- Define the type for predictions
def Prediction := Athlete → Nat

-- Define the first prediction
def firstPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 1
    | Athlete.B => 2
    | Athlete.C => 3
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the second prediction
def secondPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 4
    | Athlete.C => 1
    | Athlete.D => 5
    | Athlete.E => 2

-- Define the actual finishing positions
noncomputable def actualFinishingPositions : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 2
    | Athlete.C => 1
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the correctness conditions for the predictions
def correctPositions (actual : Prediction) (prediction : Prediction) (k : Nat) : Prop :=
  (Set.card {x : Athlete | actual x = prediction x} = k)

-- The proof statement
theorem correctFinishingOrder :
  correctPositions actualFinishingPositions firstPrediction 3 ∧
  correctPositions actualFinishingPositions secondPrediction 2 :=
sorry

end correctFinishingOrder_l361_361402


namespace distance_between_vertices_of_hyperbola_l361_361277

theorem distance_between_vertices_of_hyperbola :
  (∃ a b : ℝ, a^2 = 144 ∧ b^2 = 64 ∧ 
    ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → true) → 
  (2 * real.sqrt 144 = 24) :=
by
  sorry

end distance_between_vertices_of_hyperbola_l361_361277


namespace div_mul_neg_one_third_l361_361639

theorem div_mul_neg_one_third : (2 : ℚ) / 3 * (-1/3) = -2/9 := by
  sorry

end div_mul_neg_one_third_l361_361639


namespace sqrt_mul_sqrt_l361_361982

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361982


namespace pizza_non_crust_percentage_l361_361203

theorem pizza_non_crust_percentage (total_weight crust_weight : ℕ) (h₁ : total_weight = 200) (h₂ : crust_weight = 50) :
  (total_weight - crust_weight) * 100 / total_weight = 75 :=
by
  sorry

end pizza_non_crust_percentage_l361_361203


namespace collinear_points_l361_361658

theorem collinear_points :
  ∃ k : ℚ, k = 19 / 3 ∧ collinear ℚ (λ x : ℚ, if x = 4 then 7 else if x = 0 then k else if x = -8 then 5 else (0 : ℚ)) :=
sorry

end collinear_points_l361_361658


namespace sufficient_but_not_necessary_condition_l361_361302

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x > 0 → |x| > 0) ∧ (¬ (|x| > 0 → x > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l361_361302


namespace correctFinishingOrder_l361_361400

-- Define the types of athletes
inductive Athlete
| A | B | C | D | E

-- Define the type for predictions
def Prediction := Athlete → Nat

-- Define the first prediction
def firstPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 1
    | Athlete.B => 2
    | Athlete.C => 3
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the second prediction
def secondPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 4
    | Athlete.C => 1
    | Athlete.D => 5
    | Athlete.E => 2

-- Define the actual finishing positions
noncomputable def actualFinishingPositions : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 2
    | Athlete.C => 1
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the correctness conditions for the predictions
def correctPositions (actual : Prediction) (prediction : Prediction) (k : Nat) : Prop :=
  (Set.card {x : Athlete | actual x = prediction x} = k)

-- The proof statement
theorem correctFinishingOrder :
  correctPositions actualFinishingPositions firstPrediction 3 ∧
  correctPositions actualFinishingPositions secondPrediction 2 :=
sorry

end correctFinishingOrder_l361_361400


namespace find_a_solve_inequality_l361_361320

noncomputable def f (a x : ℝ) : ℝ := log a (x - 1) + 2

theorem find_a (a : ℝ) (ha : a > 0 ∧ a ≠ 1) (h : f a 3 = 3) : a = 2 :=
by
  sorry

theorem solve_inequality (x : ℝ) (hx : 2 < x ∧ x < 3) :
  let a := 2 in f a (2^x - 3) < f a (21 - 2^(x + 1)) :=
by
  sorry

end find_a_solve_inequality_l361_361320


namespace birdhouse_volume_difference_l361_361492

noncomputable def volume_sara : ℝ := 1 * 2 * 2  -- Volume in cubic feet

noncomputable def volume_jake := 
  let width := 16 / 12  -- Convert inches to feet
  let height := 20 / 12
  let depth := 18 / 12
  width * height * depth  -- Volume in cubic feet

noncomputable def volume_tom := 
  let width := 0.4 * 3.28084  -- Convert meters to feet
  let height := 0.6 * 3.28084
  let depth := 0.5 * 3.28084
  width * height * depth  -- Volume in cubic feet

noncomputable def largest_volume := max volume_sara (max volume_jake volume_tom)
noncomputable def smallest_volume := min volume_sara (min volume_jake volume_tom)

noncomputable def volume_difference := largest_volume - smallest_volume

theorem birdhouse_volume_difference :
  abs (volume_difference - 0.913) < 0.001 := 
by
  sorry

end birdhouse_volume_difference_l361_361492


namespace remaining_people_l361_361508

def initial_football_players : ℕ := 13
def initial_cheerleaders : ℕ := 16
def quitting_football_players : ℕ := 10
def quitting_cheerleaders : ℕ := 4

theorem remaining_people :
  (initial_football_players - quitting_football_players) 
  + (initial_cheerleaders - quitting_cheerleaders) = 15 := by
    -- Proof steps would go here, if required
    sorry

end remaining_people_l361_361508


namespace distance_from_origin_to_point_l361_361835

open Real

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 →
  y = -15 →
  sqrt (x^2 + y^2) = 17 :=
by
  intros hx hy
  rw [hx, hy]
  norm_num
  rw ←sqrt_sq (show 0 ≤ 289 by norm_num)
  norm_num

end distance_from_origin_to_point_l361_361835


namespace area_of_triangle_XYZ_l361_361416

noncomputable theory
open Real

-- Defining the problem conditions: lengths of sides and the median
def XY : ℝ := 8
def XZ : ℝ := 18
def XM : ℝ := 12

-- Using Lean to establish the proof (statement only, proof not required)
theorem area_of_triangle_XYZ : ∀ (XYZ : Triangle), 
  XYZ.side_length XY XZ XM → 
  XYZ.area = 168 :=
sorry

end area_of_triangle_XYZ_l361_361416


namespace prove_positions_l361_361376

def athlete : Type := { A : ℕ, B : ℕ, C : ℕ, D : ℕ, E : ℕ }

def first_prediction (x : athlete) : athlete :=
  { A := 1, B := 2, C := 3, D := 4, E := 5 }

def second_prediction (x : athlete) : athlete :=
  { A := 3, B := 4, C := 1, D := 5, E := 2 }

def actual_positions : athlete :=
  { A := 3, B := 2, C := 1, D := 4, E := 5 }

theorem prove_positions :
  (λ x, first_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 3 ∧
  (λ x, second_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 2 ∧
  actual_positions.A = 3 ∧ actual_positions.B = 2 ∧ actual_positions.C = 1 ∧ actual_positions.D = 4 ∧ actual_positions.E = 5 := 
  sorry

end prove_positions_l361_361376


namespace rotated_triangle_surface_area_l361_361002

theorem rotated_triangle_surface_area :
  ∀ (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (ACLength : ℝ) (BCLength : ℝ) (right_angle : ℝ -> ℝ -> ℝ -> Prop)
    (pi_def : Real) (surface_area : ℝ -> ℝ -> ℝ),
    (right_angle 90 0 90) → (ACLength = 3) → (BCLength = 4) →
    surface_area ACLength BCLength = 24 * pi_def  :=
by
  sorry

end rotated_triangle_surface_area_l361_361002


namespace three_pow_2023_mod_eleven_l361_361566

theorem three_pow_2023_mod_eleven :
  (3 ^ 2023) % 11 = 5 :=
sorry

end three_pow_2023_mod_eleven_l361_361566


namespace candy_distribution_l361_361095

theorem candy_distribution (n : ℕ) (h1 : n > 0) (h2 : 100 % n = 0) (h3 : 99 % n = 0) : n = 11 :=
sorry

end candy_distribution_l361_361095


namespace actual_positions_correct_l361_361366

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end actual_positions_correct_l361_361366


namespace sqrt_mul_sqrt_l361_361983

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361983


namespace range_of_a₁_l361_361099

variable (a₁ d : ℝ) (n : ℕ)
variable (an : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (n : ℕ) : ℕ → ℝ 
| 0 := a₁
| (n + 1) := arithmetic_sequence n + d

def sin_squared (x : ℝ) : ℝ := (sin x) ^ 2

def common_diff : Prop :=
  0 < d ∧ d < 1

def trig_condition : Prop :=
  (sin_squared (an 2) - sin_squared (an 6)) / sin (an 2 + an 6) = -1

def sum_first_n_terms (n : ℕ) : ℝ :=
  (n : ℝ) * (a₁ + arithmetic_sequence a₁ d (n - 1)) / 2

def min_sum_condition : Prop :=
  sum_first_n_terms a₁ d 10 ≤ sum_first_n_terms a₁ d 9 ∧ 
  sum_first_n_terms a₁ d 11 ≥ sum_first_n_terms a₁ d 10

theorem range_of_a₁ 
  (h₁ : common_diff d)
  (h₂ : trig_condition an d)
  (h₃ : min_sum_condition a₁ d)
  : - (5 * π / 4) ≤ a₁ ∧ a₁ ≤ - (9 * π / 8) :=
begin
  sorry
end

end range_of_a₁_l361_361099


namespace units_digit_of_subtraction_l361_361524

theorem units_digit_of_subtraction (c : ℕ) :
  let a := c + 3,
      b := 2 * c,
      original := 100 * a + 10 * b + c,
      reversed := 100 * c + 10 * b + a,
      result := original - reversed
  in result % 10 = 7 :=
by
  sorry

end units_digit_of_subtraction_l361_361524


namespace billboards_color_schemes_is_55_l361_361072

def adjacent_color_schemes (n : ℕ) : ℕ :=
  if h : n = 8 then 55 else 0

theorem billboards_color_schemes_is_55 :
  adjacent_color_schemes 8 = 55 :=
sorry

end billboards_color_schemes_is_55_l361_361072


namespace sam_gave_fraction_l361_361461

/-- Given that Mary bought 1500 stickers and shared them between Susan, Andrew, 
and Sam in the ratio 1:1:3. After Sam gave some stickers to Andrew, Andrew now 
has 900 stickers. Prove that the fraction of Sam's stickers given to Andrew is 2/3. -/
theorem sam_gave_fraction (total_stickers : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
    (initial_A : ℕ) (initial_B : ℕ) (initial_C : ℕ) (final_B : ℕ) (given_stickers : ℕ) :
    total_stickers = 1500 → ratio_A = 1 → ratio_B = 1 → ratio_C = 3 →
    initial_A = total_stickers / (ratio_A + ratio_B + ratio_C) →
    initial_B = total_stickers / (ratio_A + ratio_B + ratio_C) →
    initial_C = 3 * (total_stickers / (ratio_A + ratio_B + ratio_C)) →
    final_B = 900 →
    initial_B + given_stickers = final_B →
    given_stickers / initial_C = 2 / 3 :=
by
  intros
  sorry

end sam_gave_fraction_l361_361461


namespace max_intersection_of_fifth_degree_polynomials_l361_361980

theorem max_intersection_of_fifth_degree_polynomials (r s : Polynomial ℝ) (hr : degree r = 5) (hs : degree s = 5) 
  (lr : leadingCoeff r = 1) (ls : leadingCoeff s = 1) : 
  (∀ x, r = s → x = 4) :=
  sorry

end max_intersection_of_fifth_degree_polynomials_l361_361980


namespace median_equals_1_76_l361_361660

def heights : List ℝ := [1.71, 1.78, 1.75, 1.80, 1.69, 1.77]

def sorted_heights : List ℝ := List.sort heights

def median : ℝ := (sorted_heights.nthLe 2 (by simp) + sorted_heights.nthLe 3 (by simp)) / 2

theorem median_equals_1_76 : median = 1.76 :=
sorry

end median_equals_1_76_l361_361660


namespace largest_fraction_l361_361158

theorem largest_fraction : 
  ∀ (x ∈ ({(7 / 15), (9 / 19), (35 / 69), (399 / 799), (150 / 299)} : set ℝ)), 
  x ≤ (35 / 69) := 
by 
  sorry

end largest_fraction_l361_361158


namespace algebraic_sum_parity_l361_361783

theorem algebraic_sum_parity :
  ∀ (f : Fin 2006 → ℤ),
    (∀ i, f i = i ∨ f i = -i) →
    (∑ i, f i) % 2 = 1 := by
  sorry

end algebraic_sum_parity_l361_361783


namespace sum_of_non_consecutive_fibs_l361_361917

def fibonacci : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 1) := fibonacci n + fibonacci (n - 1)

theorem sum_of_non_consecutive_fibs (n : ℕ) (hn : n ≥ 1) :
  ∃ (indices : list ℕ), (∀ i ∈ indices, i ≥ 0 ∧ i ≤ n ∧ (∀ j ∈ indices, i ≠ j + 1 ∧ i ≠ j - 1)) ∧ n = (indices.map fibonacci).sum :=
sorry

end sum_of_non_consecutive_fibs_l361_361917


namespace tetrahedron_volume_correct_l361_361415

noncomputable def volume_tetrahedron (α : ℝ) : ℝ :=
  \frac{9 * (Real.tan α) ^ 3}{4 * Real.sqrt (3 * (Real.tan α) ^ 2 - 1)}

theorem tetrahedron_volume_correct (α : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) (dist_center_to_edge : ℝ)
  (h_dist : dist_center_to_edge = 1) :
  volume_tetrahedron α = \frac{9 * (Real.tan α) ^ 3}{4 * Real.sqrt (3 * (Real.tan α) ^ 2 - 1)} :=
by
  sorry

end tetrahedron_volume_correct_l361_361415


namespace range_of_m_l361_361781

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x ∈ set.Icc 2 4 ∧ x^2 - 2 * x + 5 - m < 0) ↔ m ∈ set.Ioo 5 ⊤ := by
  sorry

end range_of_m_l361_361781


namespace coverable_with_three_cell_strips_l361_361945

theorem coverable_with_three_cell_strips (a b c : ℕ) : 
  (∃ (A B C : ℕ), A = 3 * a ∧ B = 3 * b ∧ C = 3 * c ∧
  (A % 3 = 0 ∧ B % 3 = 0 ∧ C % 3 = 0) → 
  (∃ (facing_unit_strips_subset : set (ℕ × ℕ × ℕ)), 
  (∀ x ∈ facing_unit_strips_subset, (∃ (x' : ℕ), x' < 3 ∧ x' ∈ facing_unit_strips_subset) ∧ ∀ x y ∈ facing_unit_strips_subset, x ≠ y →
  x' ∉ facing_unit_strips_subset) → 
  (∀ i j k < 3, (i + j + k = 3)))) ↔ 
  ((a % 3 = 0 ∧ b % 3 = 0) ∨ (b % 3 = 0 ∧ c % 3 = 0) ∨ (c % 3 = 0 ∧ a % 3 = 0)) :=
sorry

end coverable_with_three_cell_strips_l361_361945


namespace john_jury_duty_days_l361_361031

-- Definitions based on the given conditions
variable (jury_selection_days : ℕ) (trial_multiple : ℕ) (deliberation_days : ℕ) (hours_per_day_deliberation : ℕ)
variable (total_days_jury_duty : ℕ)

-- Conditions
def condition1 : Prop := jury_selection_days = 2
def condition2 : Prop := trial_multiple = 4
def condition3 : Prop := deliberation_days = 6
def condition4 : Prop := hours_per_day_deliberation = 16
def correct_answer : Prop := total_days_jury_duty = 19

-- Total days calculation
def total_days_calc : ℕ :=
  let trial_days := jury_selection_days * trial_multiple
  let total_deliberation_hours := deliberation_days * 24
  let actual_deliberation_days := total_deliberation_hours / hours_per_day_deliberation
  jury_selection_days + trial_days + actual_deliberation_days

-- Statement we need to prove
theorem john_jury_duty_days : condition1 ∧ condition2 ∧ condition3 ∧ condition4 → total_days_calc = total_days_jury_duty :=
by {
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest1,
  cases h_rest1 with h3 h4,
  rw [condition1, condition2, condition3, condition4] at h1 h2 h3 h4,
  simp [h1, h2, h3, h4],
  sorry
} -- Proof omitted

end john_jury_duty_days_l361_361031


namespace scrap_rate_increase_implication_l361_361950

def regression_line (x : ℝ) : ℝ := 256 + 2 * x

theorem scrap_rate_increase_implication:
  ∀ (x : ℝ), regression_line(x + 1) - regression_line(x) = 2 := 
sorry

end scrap_rate_increase_implication_l361_361950


namespace box_total_volume_l361_361247

theorem box_total_volume (m n p : ℕ) (h₁ : n.coprime p) 
    (box_dims : ℕ × ℕ × ℕ) 
    (h₂ : box_dims = (2, 3, 4))
    (total_vol : ℝ) :
  m = 228 → n = 31 → p = 3 → 
  total_vol = (24 + 52 + 9 * Real.pi + (4 / 3) * Real.pi) → 
  m + n + p = 262 → 
  True :=
by
  intros
  sorry

end box_total_volume_l361_361247


namespace div_40_of_prime_ge7_l361_361337

theorem div_40_of_prime_ge7 (p : ℕ) (hp_prime : Prime p) (hp_ge7 : p ≥ 7) : 40 ∣ (p^2 - 1) :=
sorry

end div_40_of_prime_ge7_l361_361337


namespace minimum_value_of_f_l361_361253

variable (a k : ℝ)
variable (k_gt_1 : k > 1)
variable (a_gt_0 : a > 0)

noncomputable def f (x : ℝ) : ℝ := k * Real.sqrt (a^2 + x^2) - x

theorem minimum_value_of_f : ∃ x_0, ∀ x, f a k x ≥ f a k x_0 ∧ f a k x_0 = a * Real.sqrt (k^2 - 1) :=
by
  sorry

end minimum_value_of_f_l361_361253


namespace find_initial_sum_of_money_l361_361613

theorem find_initial_sum_of_money
    (A : ℝ)
    (R : ℝ)
    (T : ℝ)
    (hA : A = 15500)
    (hR : R = 6)
    (hT : T = 4) :
    let P := 12500 in
    A = P + (P * R * T / 100) :=
by
  sorry

end find_initial_sum_of_money_l361_361613


namespace closest_integer_to_a2013_l361_361529

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 100 ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) + (1 / a (n + 1))

theorem closest_integer_to_a2013 (a : ℕ → ℝ) (h : seq a) : abs (a 2013 - 118) < 0.5 :=
sorry

end closest_integer_to_a2013_l361_361529


namespace correct_statement_A_l361_361226

-- Definitions for conditions
def general_dilution_range : Set ℕ := {10^3, 10^4, 10^5, 10^6, 10^7}

def actinomycetes_dilution_range : Set ℕ := {10^3, 10^4, 10^5}

def fungi_dilution_range : Set ℕ := {10^2, 10^3, 10^4}

def first_experiment_dilution_range : Set ℕ := {10^3, 10^4, 10^5, 10^6, 10^7}

-- Statement to prove
theorem correct_statement_A : 
  (general_dilution_range = {10^3, 10^4, 10^5, 10^6, 10^7}) :=
sorry

end correct_statement_A_l361_361226


namespace card_prob_sum_greater_than_eight_l361_361964

theorem card_prob_sum_greater_than_eight :
  let cards := {0, 1, 2, 3, 4, 5}.to_finset
  let total_pairs := (cards.product cards).card
  let favorable_pairs := (cards.product cards).filter (λ p, (p.1 + p.2 > 8)).card
  (favorable_pairs : ℚ) / total_pairs = 1 / 12 := by
  sorry

end card_prob_sum_greater_than_eight_l361_361964


namespace actual_positions_correct_l361_361354

def athletes := {A, B, C, D, E}
def p1 : athletes → ℕ
| A := 1
| B := 2
| C := 3
| D := 4
| E := 5

def p2 : athletes → ℕ
| C := 1
| E := 2
| A := 3
| B := 4
| D := 5

def actual_positions : athletes → ℕ
| A := 3
| B := 2
| C := 1
| D := 4
| E := 5

theorem actual_positions_correct :
  (∑ a in athletes, ite (actual_positions a = p1 a) 1 0 = 3) ∧ 
  (∑ a in athletes, ite (actual_positions a = p2 a) 1 0 = 2) :=
by
  sorry

end actual_positions_correct_l361_361354


namespace distance_between_centers_of_two_circles_l361_361861

noncomputable def triangle : Type := sorry

noncomputable def E := sorry  -- Type for vertices

def DE : ℝ := 16
def DF : ℝ := 17
def EF : ℝ := 15
def inradius : ℝ := 3.5 * real.sqrt 14
def distance_centers := 10 * real.sqrt 30

theorem distance_between_centers_of_two_circles (triangle DEF : Type)
  (DE DF EF : ℝ)
  (inner_circle_radius : ℝ)
  (expected_distance : ℝ) :
  inner_circle_radius = 3.5 * real.sqrt 14 ∧
  expected_distance = 10 * real.sqrt 30 → 
  (∃ I E : DEF, 
    DE = 16 ∧
    DF = 17 ∧
    EF = 15 ∧
    distance_centers = 10 * real.sqrt 30) :=
begin
  sorry
end

end distance_between_centers_of_two_circles_l361_361861


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l361_361962

-- Define the conditions for each problem explicitly
def cond1 : Prop := ∃ (A B C : Type), -- "A" can only be in the middle or on the sides (positions are constrainted)
  True -- (specific arrangements are abstracted here)

def cond2 : Prop := ∃ (A B C : Type), -- male students must be grouped together
  True

def cond3 : Prop := ∃ (A B C : Type), -- male students cannot be grouped together
  True

def cond4 : Prop := ∃ (A B C : Type), -- the order of "A", "B", "C" from left to right remains unchanged
  True

def cond5 : Prop := ∃ (A B C : Type), -- "A" is not on the far left and "B" is not on the far right
  True

def cond6 : Prop := ∃ (A B C D : Type), -- One more female student, males and females are not next to each other
  True

def cond7 : Prop := ∃ (A B C : Type), -- arranged in two rows, with 3 people in the front row and 2 in the back row
  True

def cond8 : Prop := ∃ (A B C : Type), -- there must be 1 person between "A" and "B"
  True

-- Prove each condition results in the specified number of arrangements

theorem problem1 : cond1 → True := by
  -- Problem (1) is to show 72 arrangements given conditions
  sorry

theorem problem2 : cond2 → True := by
  -- Problem (2) is to show 36 arrangements given conditions
  sorry

theorem problem3 : cond3 → True := by
  -- Problem (3) is to show 12 arrangements given conditions
  sorry

theorem problem4 : cond4 → True := by
  -- Problem (4) is to show 20 arrangements given conditions
  sorry

theorem problem5 : cond5 → True := by
  -- Problem (5) is to show 78 arrangements given conditions
  sorry

theorem problem6 : cond6 → True := by
  -- Problem (6) is to show 144 arrangements given conditions
  sorry

theorem problem7 : cond7 → True := by
  -- Problem (7) is to show 120 arrangements given conditions
  sorry

theorem problem8 : cond8 → True := by
  -- Problem (8) is to show 36 arrangements given conditions
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l361_361962


namespace fans_impressed_total_l361_361465

theorem fans_impressed_total (sets bleachers_per_set total_fans : ℕ) 
  (h1 : sets = 3) 
  (h2 : bleachers_per_set = 812) 
  (correct_total : total_fans = sets * bleachers_per_set) : 
  total_fans = 2436 := 
by 
  rw [h1, h2] at correct_total
  exact correct_total.symm
-- sorry

end fans_impressed_total_l361_361465


namespace equivalent_shifts_l361_361554

noncomputable def f (x : ℝ) := Real.sin (2 * x)

noncomputable def g (x : ℝ) := Real.sin (2 * (x - Real.pi / 6))

theorem equivalent_shifts :
    ∀ x : ℝ, g x = Real.sin (2 * x - Real.pi / 3) :=
by
  intro x
  have h1 : g x = Real.sin (2 * x - 2 * (Real.pi / 6)),
  {
    sorry
  }
  rw [h1]
  have h2 : 2 * (Real.pi / 6) = Real.pi / 3,
  {
    sorry
  }
  rw [h2]
  apply Eq.refl

end equivalent_shifts_l361_361554


namespace volume_of_rectangular_solid_l361_361119

theorem volume_of_rectangular_solid (a b c : ℝ) 
  (h1 : a * b = 20) 
  (h2 : b * c = 15) 
  (h3 : a * c = 12) 
  (h4 : a = 2 * b) : 
  a * b * c = 12 * real.sqrt 10 :=
by 
  sorry

end volume_of_rectangular_solid_l361_361119


namespace gloves_needed_l361_361935

theorem gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) (total_gloves : ℕ)
  (h1 : participants = 82)
  (h2 : gloves_per_participant = 2)
  (h3 : total_gloves = participants * gloves_per_participant) :
  total_gloves = 164 :=
by
  sorry

end gloves_needed_l361_361935


namespace sum_of_solutions_eq_zero_l361_361679

theorem sum_of_solutions_eq_zero :
  let f (x : ℝ) := 2^|x| + 4 * |x|
  (∀ x : ℝ, f x = 20) →
  (∃ x₁ x₂ : ℝ, f x₁ = 20 ∧ f x₂ = 20 ∧ x₁ + x₂ = 0) :=
sorry

end sum_of_solutions_eq_zero_l361_361679


namespace fraction_decomposition_l361_361579

-- We use a classical real number theory to deal with fractions and sums.
noncomputable def sum_of_fractions_decomposition : Prop :=
  ∃ (f : ℕ → ℚ), (∀ n, 0 < f n ∧ f n ≤ 1) ∧ (∑' n, f n = 3/7)

theorem fraction_decomposition :
  ∃ (f : ℕ → ℚ), (∀ n, numerator f n = 1) ∧ (∑' n, f n = 3/7) :=
by
  -- Here we will prove the existence of such a decomposition, but this proof is not provided here.
  sorry

end fraction_decomposition_l361_361579


namespace trigonometric_identity_in_triangle_l361_361015

-- Define the angles of the triangle
variables {A B C : ℝ}

-- State the condition that A + B + C = π
axiom angles_sum_to_pi : A + B + C = Real.pi

-- State the goal
theorem trigonometric_identity_in_triangle 
  (h : A + B + C = Real.pi) : 
  (tan A + tan B + tan C) / (2 * tan A * tan B * tan C) = 1 / 2 :=
by
  sorry

end trigonometric_identity_in_triangle_l361_361015


namespace weekly_writing_hours_l361_361022

def pages_per_hour_literature : ℕ := 12
def pages_per_hour_politics : ℕ := 8
def pages_per_day_politics_weekdays : ℕ := 5
def pages_per_day_literature_weekends : ℕ := 10
def people_weekdays : ℕ := 2
def people_weekends : ℕ := 3
def weekdays : ℕ := 5
def weekend_days : ℕ := 2

theorem weekly_writing_hours 
  (pages_per_hour_literature : ℕ) 
  (pages_per_hour_politics : ℕ) 
  (pages_per_day_politics_weekdays : ℕ) 
  (pages_per_day_literature_weekends : ℕ) 
  (people_weekdays : ℕ) 
  (people_weekends : ℕ) 
  (weekdays : ℕ) 
  (weekend_days : ℕ) :
  pages_per_hour_literature = 12 →
  pages_per_hour_politics = 8 →
  pages_per_day_politics_weekdays = 5 →
  pages_per_day_literature_weekends = 10 →
  people_weekdays = 2 →
  people_weekends = 3 →
  weekdays = 5 →
  weekend_days = 2 →
  let pages_politics_per_week := (pages_per_day_politics_weekdays * people_weekdays * weekdays) / pages_per_hour_politics in
  let pages_literature_per_week := (pages_per_day_literature_weekends * people_weekends * weekend_days) / pages_per_hour_literature in
  pages_politics_per_week + pages_literature_per_week = 11.25 := 
begin
  intros,
  sorry
end

end weekly_writing_hours_l361_361022


namespace vector_expression_value_l361_361761

variable {V : Type*} [inner_product_space ℝ V]

theorem vector_expression_value 
  (a b c : V)
  (h1 : a + b + 2 • c = 0)
  (h2 : ∥a∥ = 1)
  (h3 : ∥b∥ = 3)
  (h4 : ∥c∥ = 2) : 
  a ⬝ b + 2 * (a ⬝ c) + 2 * (b ⬝ c) = -13 := 
by
  sorry

end vector_expression_value_l361_361761


namespace log_ratio_condition_l361_361791

-- Definitions of the conditions
variables (a1 q : ℝ)
variables (a : ℕ → ℝ)
variables (n : ℕ)

-- Condition 1: Geometric sequence with common ratio q, and positive terms
def is_geometric_sequence := ∀ n : ℕ, a n = a1 * (q ^ n)

-- Condition 2: -6, q^2, 14 is an arithmetic sequence
def arithmetic_seq_condition := 2 * q^2 = -6 + 14

-- Target: Logarithmic ratio condition
theorem log_ratio_condition 
  (h1 : is_geometric_sequence a1 q a)
  (h2 : arithmetic_seq_condition q) : 
  log 2 ((a 2 + a 3) / (a 0 + a 1)) = 2 :=
sorry

end log_ratio_condition_l361_361791


namespace algebraic_expression_value_l361_361298

variable (a b : ℝ)

theorem algebraic_expression_value
  (h : a^2 + 2 * b^2 - 1 = 0) :
  (a - b)^2 + b * (2 * a + b) = 1 :=
by
  sorry

end algebraic_expression_value_l361_361298


namespace smallest_solution_eq_l361_361678

theorem smallest_solution_eq :
  (∀ x : ℝ, x ≠ 3 →
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 15) → 
  x = 1 - Real.sqrt 10 ∨ (∃ y : ℝ, y ≤ 1 - Real.sqrt 10 ∧ y ≠ 3 ∧ 3 * y / (y - 3) + (3 * y^2 - 27) / y = 15)) :=
sorry

end smallest_solution_eq_l361_361678


namespace yield_difference_l361_361887

variables (x y : ℝ)
variables (A1 A2 : ℝ)
variables (avg_yield : ℝ)

-- Conditions
def area_non_varietal := 14
def area_varietal := 4
def condition1 := avg_yield = x + 20
def condition2 := avg_yield * (area_non_varietal + area_varietal) = x * area_non_varietal + y * area_varietal

-- The proof problem
theorem yield_difference : condition1 → condition2 → (y - x = 90) :=
by sorry

end yield_difference_l361_361887


namespace actual_positions_correct_l361_361360

def athletes := {A, B, C, D, E}
def p1 : athletes → ℕ
| A := 1
| B := 2
| C := 3
| D := 4
| E := 5

def p2 : athletes → ℕ
| C := 1
| E := 2
| A := 3
| B := 4
| D := 5

def actual_positions : athletes → ℕ
| A := 3
| B := 2
| C := 1
| D := 4
| E := 5

theorem actual_positions_correct :
  (∑ a in athletes, ite (actual_positions a = p1 a) 1 0 = 3) ∧ 
  (∑ a in athletes, ite (actual_positions a = p2 a) 1 0 = 2) :=
by
  sorry

end actual_positions_correct_l361_361360


namespace sum_exists_in_sets_l361_361713

open Set

theorem sum_exists_in_sets (n : ℕ) (A B : Set ℕ) (hA : ∀ a ∈ A, a < n) (hB : ∀ b ∈ B, b < n)
  (h_distinctA : ∀ a1 a2 ∈ A, a1 ≠ a2)
  (h_distinctB : ∀ b1 b2 ∈ B, b1 ≠ b2)
  (h_size : A.card + B.card ≥ n) :
  ∃ a ∈ A, ∃ b ∈ B, a + b = n := by
  sorry

end sum_exists_in_sets_l361_361713


namespace total_shaded_area_l361_361583

-- Defining the conditions
def floor_length : ℝ := 12
def floor_width : ℝ := 16
def tile_size : ℝ := 2
def tile_area : ℝ := tile_size * tile_size
def quarter_circle_radius : ℝ := 1 / 2
def quarter_circle_area : ℝ := π * (quarter_circle_radius ^ 2)
def total_white_area_per_tile : ℝ := 4 * quarter_circle_area
def shaded_area_per_tile : ℝ := tile_area - total_white_area_per_tile
def total_tiles : ℕ := (floor_length * floor_width / tile_area).toNat

-- Stating the proof goal
theorem total_shaded_area : (total_tiles : ℝ) * shaded_area_per_tile = 192 - 48 * π := by
  sorry

end total_shaded_area_l361_361583


namespace farmer_broccoli_difference_l361_361594

theorem farmer_broccoli_difference :
  ∀ (this_year last_year : ℕ),
  (∀ (side_length : ℕ), side_length * side_length = this_year  → this_year = 2601 ∧ exists (last_side : ℕ), last_side < side_length ∧ last_side * last_side = last_year)
  → (this_year - last_year = 101) :=
by
  -- Let this_year be the area of the square this year
  -- Let last_year be the area of the square last year
  intros this_year last_year h
  cases h with side_length h_eq
  cases h_eq with h_this_year h_last_year
  cases h_last_year with last_side h_lt
  cases h_lt with h_side_cond h_last_eq

  -- Prove that side_length is 51
  have H1: side_length = 51,
  {
    sorry
  },
  -- Prove that last_side is 50
  have H2: last_side = 50,
  {
    sorry
  }
  -- Therefore
  have H3: this_year = 2601,
  {
    rw H1 at h_eq,
    exact h_eq
  },
  have H4: last_year = 2500,
  {
    rw H2 at h_last_eq,
    exact h_last_eq
  },
  -- The difference
  rw [H3, H4],
  exact rfl

end farmer_broccoli_difference_l361_361594


namespace trapezoid_tangential_properties_l361_361014

-- Define the given conditions
variables (A B C D M : Point)
variables (BC AD : Segment)
variable (h_trapezoid : is_trapezoid_with_bases A B C D BC AD)
variable (h_right_angle : angle D A B = 90)
variable (unique_M : ∃! M, M ∈ segment C D ∧ angle B M A = 90)

-- Statement to be proven
theorem trapezoid_tangential_properties :
  BC.length = (segment.point_distance C M) ∧ AD.length = (segment.point_distance D M) :=
sorry

end trapezoid_tangential_properties_l361_361014


namespace rectangle_perimeter_width_ratio_l361_361115

theorem rectangle_perimeter_width_ratio 
  (A : ℕ) (l : ℕ) (hA : A = 150) (hl : l = 15) : 
  let w := A / l in
  let P := 2 * (l + w) in
  P / w = 5 :=
by
  sorry

end rectangle_perimeter_width_ratio_l361_361115


namespace distance_origin_to_point_l361_361796

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l361_361796


namespace actual_positions_correct_l361_361357

def athletes := {A, B, C, D, E}
def p1 : athletes → ℕ
| A := 1
| B := 2
| C := 3
| D := 4
| E := 5

def p2 : athletes → ℕ
| C := 1
| E := 2
| A := 3
| B := 4
| D := 5

def actual_positions : athletes → ℕ
| A := 3
| B := 2
| C := 1
| D := 4
| E := 5

theorem actual_positions_correct :
  (∑ a in athletes, ite (actual_positions a = p1 a) 1 0 = 3) ∧ 
  (∑ a in athletes, ite (actual_positions a = p2 a) 1 0 = 2) :=
by
  sorry

end actual_positions_correct_l361_361357


namespace prob_S_lt_4_prob_abs_diff_le_2_l361_361739

-- Define the sets and probabilities for the first problem
def outcome_set_1 : set (ℕ × ℕ) := {(x, y) | x ∈ {1, 2, 3} ∧ y ∈ {1, 2, 3}}
def event_set_1 : set (ℕ × ℕ) := {(x, y) | x * y < 4}

theorem prob_S_lt_4 : 
  (set.card event_set_1 : ℚ) / set.card outcome_set_1 = 5 / 9 :=
by {
  sorry
}

-- Define the measurable sets and probabilities for the second problem
def region_omega : set (ℝ × ℝ) := {p | 0 < p.1 ∧ p.1 < 4 ∧ 0 < p.2 ∧ p.2 < 4}
def region_H : set (ℝ × ℝ) := {p | 0 < p.1 ∧ p.1 < 4 ∧ 0 < p.2 ∧ p.2 < 4 ∧ abs (p.1 - p.2) ≤ 2}

theorem prob_abs_diff_le_2 : 
  (measure_of region_H) / (measure_of region_omega) = 3 / 4 :=
by {
  sorry
}

end prob_S_lt_4_prob_abs_diff_le_2_l361_361739


namespace distance_from_origin_to_point_l361_361801

-- Define the specific points in the coordinate system
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (8, -15)

-- The distance formula in Euclidean space
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem statement
theorem distance_from_origin_to_point : distance origin point = 17 := 
by
  sorry

end distance_from_origin_to_point_l361_361801


namespace yard_flower_beds_fraction_l361_361210

/-- 
  Given:
  1. A yard with a trapezoidal shape, having parallel sides of lengths 18 meters and 30 meters, and a height of 6 meters.
  2. Two flower beds: one is a right triangle and the other is an isosceles right triangle.
  3. The base of the right triangle is 12 meters and its height is 6 meters.
  4. The legs of the isosceles right triangle are 6 meters in length.
-/
theorem yard_flower_beds_fraction 
  (height : ℝ)
  (base1 : ℝ) (base2 : ℝ)
  (triangle_base : ℝ) (triangle_height : ℝ)
  (isosceles_leg : ℝ) :
  (base1 = 18) →
  (base2 = 30) →
  (height = 6) →
  (triangle_base = base2 - base1) →
  (triangle_height = height) →
  (isosceles_leg = triangle_base / 2) →
  let yard_area := base2 * height,
      right_triangle_area := (triangle_base * triangle_height) / 2,
      isosceles_triangle_area := (isosceles_leg^2) / 2,
      flower_beds_area := right_triangle_area + isosceles_triangle_area
  in flower_beds_area / yard_area = 3 / 10 :=
begin
  intros h1 h2 h3 h4 h5 h6,
  sorry
end

end yard_flower_beds_fraction_l361_361210


namespace find_DA_l361_361434

theorem find_DA (ω : Circle) (A B C D E F : Point) 
  (h1 : inscribed_quadrilateral ω A B C D)
  (h2 : intersect_at E A C B D)
  (h3 : tangent_at B ω F meets_line A C)
  (h4 : lies_between C E F)
  (h5 : dist A E = 6)
  (h6 : dist E C = 4)
  (h7 : dist B E = 2)
  (h8 : dist B F = 12) : 
  dist D A = 2 * sqrt 42 :=
by 
  sorry

end find_DA_l361_361434


namespace train_crossing_time_l361_361016

-- Definitions of the given problem conditions
def train_length : ℕ := 120  -- in meters.
def speed_kmph : ℕ := 144   -- in km/h.

-- Conversion factor
def km_per_hr_to_m_per_s (speed : ℕ) : ℚ :=
  speed * (1000 / 3600 : ℚ)

-- Speed in m/s
def train_speed : ℚ := km_per_hr_to_m_per_s speed_kmph

-- Time calculation
def time_to_cross_pole (length : ℕ) (speed : ℚ) : ℚ :=
  length / speed

-- The theorem we want to prove.
theorem train_crossing_time :
  time_to_cross_pole train_length train_speed = 3 := by 
  sorry

end train_crossing_time_l361_361016


namespace tangent_line_at_3_9_when_m_1_monotonic_intervals_l361_361747

noncomputable def f (m x : ℝ) : ℝ := (m / 3) * x^3 + m * x^2 - 3 * m * x + m - 1

theorem tangent_line_at_3_9_when_m_1 :
  let m := 1,
      x := 3,
      y := 9,
      f_x := (1 / 3) * x ^ 3 + 1 * x ^ 2 - 3 * 1 * x + 1 - 1 in
  (12 * x - y - 27 = 0) := sorry

theorem monotonic_intervals (m x : ℝ) (h : m ≠ 0) :
  let f' := m * x^2 + 2 * m * x - 3 * m in
  (m > 0 → (∀ x < -3, f' > 0) ∧ (∀ x > 1, f' > 0) ∧ (∃ x, -3 < x ∧ x < 1 ∧ f' < 0)) ∧
  (m < 0 → (∀ x, -3 < x ∧ x < 1 → f' > 0) ∧ (∃ x, x < -3 ∧ f' < 0) ∧ (∃ x, x > 1 ∧ f' < 0)) := sorry

end tangent_line_at_3_9_when_m_1_monotonic_intervals_l361_361747


namespace maximize_x5_y3_ordered_pair_l361_361056

open Real

noncomputable def maximize_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 28) : Prop :=
  x = 17.5 ∧ y = 10.5 ∧ (∀ x' y', 0 < x' → 0 < y' → x' + y' = 28 → x'^5 * y'^3 ≤ 17.5^5 * 10.5^3)

theorem maximize_x5_y3_ordered_pair
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 28) :
  maximize_xy x y hx hy hxy :=
begin
  sorry
end

end maximize_x5_y3_ordered_pair_l361_361056


namespace age_of_15th_student_l361_361515

theorem age_of_15th_student (avg15 : ℕ) (avg7_first : ℕ) (avg7_second : ℕ) : 
  (avg15 = 15) → 
  (avg7_first = 14) → 
  (avg7_second = 16) →
  (let T := 15 * avg15 in
   let sum_first := 7 * avg7_first in
   let sum_second := 7 * avg7_second in
   T - (sum_first + sum_second) = 15) :=
by
  intros h1 h2 h3
  sorry

end age_of_15th_student_l361_361515


namespace expected_value_win_l361_361593

theorem expected_value_win : 
  ∑ i in Finset.range 8, (2 * (i + 1)^2) * (1 / 8 : ℝ) = 51 := by
  sorry

end expected_value_win_l361_361593


namespace y_is_one_y_is_neg_two_thirds_l361_361759

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (1, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove y = 1 given dot_product(vector_a, vector_b(y)) = 5
theorem y_is_one (h : dot_product vector_a (vector_b y) = 5) : y = 1 :=
by
  -- We assume the proof (otherwise it would go here)
  sorry

-- Prove y = -2/3 given |vector_a + vector_b(y)| = |vector_a - vector_b(y)|
theorem y_is_neg_two_thirds (h : (vector_a.1 + (vector_b y).1)^2 + (vector_a.2 + (vector_b y).2)^2 =
                                (vector_a.1 - (vector_b y).1)^2 + (vector_a.2 - (vector_b y).2)^2) : y = -2/3 :=
by
  -- We assume the proof (otherwise it would go here)
  sorry

end y_is_one_y_is_neg_two_thirds_l361_361759


namespace age_of_15th_student_is_15_l361_361512

-- Define the total number of students
def total_students : Nat := 15

-- Define the average age of all 15 students together
def avg_age_all_students : Nat := 15

-- Define the average age of the first group of 7 students
def avg_age_first_group : Nat := 14

-- Define the average age of the second group of 7 students
def avg_age_second_group : Nat := 16

-- Define the total age based on the average age and number of students
def total_age_all_students : Nat := total_students * avg_age_all_students
def total_age_first_group : Nat := 7 * avg_age_first_group
def total_age_second_group : Nat := 7 * avg_age_second_group

-- Define the age of the 15th student
def age_of_15th_student : Nat := total_age_all_students - (total_age_first_group + total_age_second_group)

-- Theorem: prove that the age of the 15th student is 15 years
theorem age_of_15th_student_is_15 : age_of_15th_student = 15 := by
  -- The proof will go here
  sorry

end age_of_15th_student_is_15_l361_361512


namespace relationship_among_a_b_c_l361_361705

noncomputable def a : ℝ := 2 ^ 0.2
noncomputable def b : ℝ := 0.4 ^ 0.2
noncomputable def c : ℝ := 0.4 ^ 0.6

theorem relationship_among_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_among_a_b_c_l361_361705


namespace range_of_m_for_G_square_of_linear_l361_361053

theorem range_of_m_for_G_square_of_linear (x : ℝ) (m : ℝ) :
  (G : ℝ) = ((8 * x^2 + 20 * x + 5 * m) / 5) →
  G = (8 / 5) * x^2 + 4 * x + m → 
  m = 125 / 32 → 
  3 < m ∧ m < 4 :=
by 
  intros h1 h2 h3
  rw h3
  split
  { norm_num }
  { norm_num }

end range_of_m_for_G_square_of_linear_l361_361053


namespace increase_is_150_l361_361064

-- Define the conditions given in the problem
def new_avg_commission : ℝ := 250
def num_sales : ℕ := 6
def big_sale_commission : ℝ := 1000

-- Define the total commission earnings after the big sale
def total_commission_earnings : ℝ := new_avg_commission * num_sales

-- Define the total commission earnings without the big sale
def total_commission_without_big_sale : ℝ := total_commission_earnings - big_sale_commission

-- Define the number of sales without the big sale
def num_sales_without_big_sale : ℕ := num_sales - 1

-- Define the average commission before the big sale
def avg_commission_before_big_sale : ℝ := total_commission_without_big_sale / num_sales_without_big_sale

-- Define the increase in average commission
def increase_in_avg_commission : ℝ := new_avg_commission - avg_commission_before_big_sale

-- The theorem to be proven: the increase in average commission is 150
theorem increase_is_150 : increase_in_avg_commission = 150 := by
  sorry

end increase_is_150_l361_361064


namespace distance_from_origin_to_point_l361_361829

open Real

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 →
  y = -15 →
  sqrt (x^2 + y^2) = 17 :=
by
  intros hx hy
  rw [hx, hy]
  norm_num
  rw ←sqrt_sq (show 0 ≤ 289 by norm_num)
  norm_num

end distance_from_origin_to_point_l361_361829


namespace calculate_wheel_radii_l361_361008

theorem calculate_wheel_radii (rpmA rpmB : ℕ) (length : ℝ) (r R : ℝ) :
  rpmA = 1200 →
  rpmB = 1500 →
  length = 9 →
  (4 : ℝ) / 5 * r = R →
  2 * (R + r) = 9 →
  r = 2 ∧ R = 2.5 :=
by
  intros
  sorry

end calculate_wheel_radii_l361_361008


namespace order_options_count_l361_361345

/-- Define the number of options for each category -/
def drinks : ℕ := 3
def salads : ℕ := 2
def pizzas : ℕ := 5

/-- The theorem statement that we aim to prove -/
theorem order_options_count : drinks * salads * pizzas = 30 :=
by
  sorry -- Proof is skipped as instructed

end order_options_count_l361_361345


namespace AB_eq_2AP_l361_361100

-- Definition of the geometric setup (Two regular heptagons sharing vertex P on side AB)
variables {A B C D E F G : Point}
variables {A' P Q R S T U : Point}
variables {O : Point}

-- Definitions corresponding to conditions of the problem
def AB := dist A B
def AP := dist A P
def heptagon (pts : list) : Prop := True  -- Definition of a regular heptagon

-- Given conditions: both heptagons are regular and the specific positioning of points
axiom regular_heptagons : heptagon [(A, B, C, D, E, F, G), (A, P, Q, R, S, T, U)]
axiom P_on_AB : lies P AB
axiom U_on_GA : lies U GA
axiom Q_on_OB : lies Q OB
axiom O_center : is_center O [(A, B, C, D, E, F, G)]

-- The goal to prove
theorem AB_eq_2AP : AB = 2 * AP := 
sorry

end AB_eq_2AP_l361_361100


namespace largest_increase_between_2006_and_2007_l361_361229

-- Define the number of students taking the AMC in each year
def students_2002 := 50
def students_2003 := 55
def students_2004 := 63
def students_2005 := 70
def students_2006 := 75
def students_2007_AMC10 := 90
def students_2007_AMC12 := 15

-- Define the total number of students participating in any AMC contest each year
def total_students_2002 := students_2002
def total_students_2003 := students_2003
def total_students_2004 := students_2004
def total_students_2005 := students_2005
def total_students_2006 := students_2006
def total_students_2007 := students_2007_AMC10 + students_2007_AMC12

-- Function to calculate percentage increase
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old : ℕ) : ℚ) / old * 100

-- Calculate percentage increases between the years
def inc_2002_2003 := percentage_increase total_students_2002 total_students_2003
def inc_2003_2004 := percentage_increase total_students_2003 total_students_2004
def inc_2004_2005 := percentage_increase total_students_2004 total_students_2005
def inc_2005_2006 := percentage_increase total_students_2005 total_students_2006
def inc_2006_2007 := percentage_increase total_students_2006 total_students_2007

-- Prove that the largest percentage increase is between 2006 and 2007
theorem largest_increase_between_2006_and_2007 :
  inc_2006_2007 > inc_2005_2006 ∧
  inc_2006_2007 > inc_2004_2005 ∧
  inc_2006_2007 > inc_2003_2004 ∧
  inc_2006_2007 > inc_2002_2003 := 
by {
  sorry
}

end largest_increase_between_2006_and_2007_l361_361229


namespace pyramid_volume_l361_361600

theorem pyramid_volume (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 15) :
  let base_area := (1 / 2) * a * b,
      hypotenuse := Real.sqrt (a^2 + b^2),
      radius := hypotenuse / 2,
      height := Real.sqrt (c^2 - radius^2),
      volume := (1 / 3) * base_area * height
  in volume = 10 * Real.sqrt 182.75 :=
by
  sorry

end pyramid_volume_l361_361600


namespace kids_played_on_tuesday_l361_361851

-- Definitions of the conditions
def kids_played_on_wednesday (julia : Type) : Nat := 4
def kids_played_on_monday (julia : Type) : Nat := 6
def difference_monday_wednesday (julia : Type) : Nat := 2

-- Define the statement to prove
theorem kids_played_on_tuesday (julia : Type) :
  (kids_played_on_monday julia - difference_monday_wednesday julia) = kids_played_on_wednesday julia :=
by
  sorry

end kids_played_on_tuesday_l361_361851


namespace area_of_triangle_PQR_is_zero_l361_361240

-- Defining the conditions
def P := (-5 : ℝ, 0 : ℝ)
def Q := (0 : ℝ, 0 : ℝ)
def R := (7 : ℝ, 0 : ℝ)

-- Proving that the area is 0
theorem area_of_triangle_PQR_is_zero :
  let area (A B C : ℝ × ℝ): ℝ := 
    0.5 * abs ((fst A) * ((snd B) - (snd C)) + (fst B) * ((snd C) - (snd A)) + (fst C) * ((snd A) - (snd B))) in
  area P Q R = 0 :=
by 
  let area := λ (A B C : ℝ × ℝ): ℝ, 
    0.5 * abs ((fst A) * ((snd B) - (snd C)) + (fst B) * ((snd C) - (snd A)) + (fst C) * ((snd A) - (snd B))) in
  show area P Q R = 0, from
  sorry

end area_of_triangle_PQR_is_zero_l361_361240


namespace miron_wins_if_both_play_optimally_l361_361150

theorem miron_wins_if_both_play_optimally :
  ∀ (piles : List ℕ), piles.length = 10 → (∀ pile ∈ piles, pile = 10) →
  (∀ optimal_play, optimal_play → (let final_state := play_game piles optimal_play in
    (∀ pile ∈ final_state, pile = 1) ∧ (last_move final_state = varja)))
  → (miron_wins optimal_play) :=
by
  intros piles h_len h_all_ten h_optimal_play
  sorry

end miron_wins_if_both_play_optimally_l361_361150


namespace sum_of_first_eight_terms_of_gp_l361_361957

-- Definitions based on the problem statement
variables {a1 d : ℝ}

def ap_sum_condition : Prop := (a1 + (a1 + d) + (a1 + 2 * d)) = 21

def gp_condition : Prop := 
  ∃ (r : ℝ),
  (a1 - 1) * (a1 + 2 * d + 2) = (a1 + d - 1) ^ 2 * r

noncomputable def sum_gp_first_eight (a1 d : ℝ) : ℝ :=
let b1 := a1 - 1 in
let q := 2 in
b1 * (q ^ 8 - 1) / (q - 1)

theorem sum_of_first_eight_terms_of_gp :
  ap_sum_condition →
  gp_condition →
  sum_gp_first_eight 4 3 = 765 :=
by
  sorry

end sum_of_first_eight_terms_of_gp_l361_361957


namespace actual_positions_correct_l361_361362

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end actual_positions_correct_l361_361362


namespace distributing_cousins_into_rooms_l361_361462

-- Variables definitions for the problem
def num_cousins : ℕ := 5
def num_rooms : ℕ := 4

-- Main theorem statement
theorem distributing_cousins_into_rooms : 
  (number_of_ways : ∀ (c : ℕ), c = num_cousins → ∀ (r : ℕ), r = num_rooms → ℕ) = 66 := 
by
  sorry

end distributing_cousins_into_rooms_l361_361462


namespace Nina_money_l361_361067

theorem Nina_money : ∃ (M : ℝ) (W : ℝ), M = 10 * W ∧ M = 14 * (W - 3) ∧ M = 105 :=
by
  sorry

end Nina_money_l361_361067


namespace sequence_difference_l361_361638

theorem sequence_difference :
  let S1 := (finset.range 100).sum (λ n, 3001 + n)
  let S2 := (finset.range 100).sum (λ n, 201 + n)
  S1 - S2 = 280000 :=
by
  sorry

end sequence_difference_l361_361638


namespace quadratic_has_two_distinct_roots_l361_361709

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_has_two_distinct_roots 
  (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  discriminant a b c > 0 :=
sorry

end quadratic_has_two_distinct_roots_l361_361709


namespace probability_event_A_occurrence_at_least_two_and_at_most_four_times_l361_361017

theorem probability_event_A_occurrence_at_least_two_and_at_most_four_times :
  let n := 2000
  let p := 0.001
  let lambda := n * p
  ∑ k in finset.range 5, if k >= 2 then (lambda^k / (k.factorial * Real.exp(lambda))) else 0 = 0.541 := 
sorry

end probability_event_A_occurrence_at_least_two_and_at_most_four_times_l361_361017


namespace number_of_mappings_l361_361875

-- Given sets A and B
def A : Set ℝ := {a | ∃ i : ℕ, 0 < i ∧ i ≤ 50 ∧ a = (a : ℝ)}
def B : Set ℝ := {b | ∃ j : ℕ, 0 < j ∧ j ≤ 25 ∧ b = (b : ℝ)}

-- Define the mapping f with the given properties
def f (a : ℝ) : ℝ := 
  if h : a ∈ A then
    classical.some (classical.some_spec h)
  else 
    0

-- Define the property that every element in B has a preimage in A and f is non-increasing
def mapping_property : Prop := 
  (∀ b ∈ B, ∃ a ∈ A, f a = b) ∧ 
  (∀ i j : ℕ, 0 < i ∧ i < j ∧ j ≤ 50 → f (classical.some (classical.some_spec (classical.some_spec (by sorry)))) ≥ f (classical.some (classical.some_spec (by sorry))))

-- The main theorem stating the number of such mappings
theorem number_of_mappings : mapping_property → C 49 24 := by
  sorry

end number_of_mappings_l361_361875


namespace integral_result_l361_361779

theorem integral_result (a : ℝ) (h : a > 0) 
  (coeff_condition : (∑ k in finset.range(7), nat.choose 6 k * (a ^ (6 - k)) * (-1)^k = 60)) : 
  ∫ x in -1..a, x^2 - 2*x = 0 :=
sorry

end integral_result_l361_361779


namespace feeding_times_per_day_l361_361491

-- Definitions for the given conditions
def number_of_puppies : ℕ := 7
def total_portions : ℕ := 105
def number_of_days : ℕ := 5

-- Theorem to prove the answer to the question
theorem feeding_times_per_day : 
  let portions_per_day := total_portions / number_of_days in
  let times_per_puppy := portions_per_day / number_of_puppies in
  times_per_puppy = 3 :=
by
  -- We should provide the proof here, but we will use 'sorry' to skip it
  sorry

end feeding_times_per_day_l361_361491


namespace inequality_proof_l361_361452

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b * c = 1) : 
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l361_361452


namespace determinant_of_triangle_angles_is_zero_l361_361858

variable (A B C : ℝ)

def is_triangle_angle (A B C : ℝ) : Prop :=
  A + B + C = π

theorem determinant_of_triangle_angles_is_zero
  (h : is_triangle_angle A B C) :
  det ![
    ![cos A ^ 2, tan A, 1],
    ![cos B ^ 2, tan B, 1],
    ![cos C ^ 2, tan C, 1]
  ] = 0 := by sorry

end determinant_of_triangle_angles_is_zero_l361_361858


namespace find_triples_l361_361671

theorem find_triples (a b p : ℕ) (hp : prime p) (ha : 0 < a) (hb : 0 < b) : 2 ^ a + p ^ b = 19 ^ a ↔ (a = 1 ∧ b = 1 ∧ p = 17) :=
by sorry

end find_triples_l361_361671


namespace sin_theta_solution_l361_361299

theorem sin_theta_solution (θ : ℝ) (hθ : θ ∈ Ioo 0 (Real.pi / 2))
  (h : Real.cos θ + Real.cos (θ + Real.pi / 3) = Real.sqrt 3 / 3) : 
  Real.sin θ = (-1 + 2 * Real.sqrt 6) / 6 :=
by
  --  Proof goes here
  sorry

end sin_theta_solution_l361_361299


namespace integral_x_squared_l361_361265

theorem integral_x_squared :
  ∫ x in (0 : ℝ)..(1 : ℝ), x^2 = (1 : ℝ) / 3 :=
by indeed sorry

end integral_x_squared_l361_361265


namespace total_balls_l361_361539

theorem total_balls (boxes : ℕ) (balls_per_box : ℕ) (h1 : boxes = 3) (h2 : balls_per_box = 2) : boxes * balls_per_box = 6 :=
by
  rw [h1, h2]
  exact rfl

end total_balls_l361_361539


namespace solution_l361_361270

open Nat

def isValidA (n : ℕ) (A : Finset ℕ) : Prop :=
  ∀ k ∈ A, k ∈ range n.succ → A.count (λ x, k ∣ x) = k

theorem solution :
  ∀ n : ℕ, n ∈ {1, 2} ↔ ∃ A : Finset ℕ, isValidA n A :=
by
  sorry

end solution_l361_361270


namespace problem_proof_l361_361581

-- Define the elements involved in the problem
def a : ℤ := 2004
def b : ℤ := 2003
def c : ℤ := 2002
def d : ℤ := b - a -- -1
def e : ℤ := (d:ℤ)^(2004 : ℤ) -- 1

-- Define the main expression
def expression : ℤ := a - (b - (a * ((b - c) * e)))

-- Define the correct answer
def correct_answer : ℤ := 2005

-- The theorem that needs to be proven
theorem problem_proof : expression = correct_answer := by
  sorry

end problem_proof_l361_361581


namespace athlete_positions_l361_361383

theorem athlete_positions {A B C D E : Type} 
  (h1 : A = 1 ∧ B = 2 ∧ C = 3 ∧ D = 4 ∧ E = 5)
  (h2 : C = 1 ∧ E = 2 ∧ A = 3 ∧ B = 4 ∧ D = 5)
  (h3 : (A = 1 ∧ B = 2 ∧ C = 3 ∧ ¬(D = 4 ∨ E = 5)) ∨ 
       (A = 1 ∧ ¬(B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)) ∨ 
       (B = 2 ∧ C = 3 ∧ ¬(A = 1 ∨ D = 4 ∨ E = 5)) ∨ 
       (¬(A = 1 ∨ B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)))
  (h4 : (C = 1 ∧ ¬(A = 3 ∨ B = 4 ∨ D = 5)) ∨ 
       (A = 3 ∧ ¬(C = 1 ∨ B = 4 ∨ D = 5)) ∨ 
       (B = 4 ∧ ¬(C = 1 ∨ A = 3 ∨ D = 5)))
  : (C = 1 ∧ B = 2 ∧ A = 3 ∧ D = 4 ∧ E = 5) :=
by sorry

end athlete_positions_l361_361383


namespace stacy_faster_than_heather_l361_361504

-- Definitions for the conditions
def distance : ℝ := 40
def heather_rate : ℝ := 5
def heather_distance : ℝ := 17.090909090909093
def heather_delay : ℝ := 0.4
def stacy_distance : ℝ := distance - heather_distance
def stacy_rate (S : ℝ) (T : ℝ) : Prop := S * T = stacy_distance
def heather_time (T : ℝ) : ℝ := T - heather_delay
def heather_walk_eq (T : ℝ) : Prop := heather_rate * heather_time T = heather_distance

-- The proof problem statement
theorem stacy_faster_than_heather :
  ∃ (S T : ℝ), stacy_rate S T ∧ heather_walk_eq T ∧ (S - heather_rate = 1) :=
by
  sorry

end stacy_faster_than_heather_l361_361504


namespace ellipse_properties_l361_361519

def ellipse_equation (a b x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1)

def focal_points (F1 F2 : ℝ × ℝ) (a b : ℝ) : Prop :=
  F1.1 = -sqrt (a^2 - b^2) ∧ F2.1 = sqrt (a^2 - b^2) ∧ F1.2 = 0 ∧ F2.2 = 0

def point_A_condition (A : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : ℝ :=
  abs (dist A F1 + dist A F2) 

def max_area (A : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : ℝ :=
  abs ((A.1 - F1.1) * (F2.2 - F1.2) - (A.2 - F1.2) * (F2.1 - F1.1)) / 2

def midpoint_P (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def slope (A B : ℝ × ℝ) : ℝ :=
  (A.2 - B.2) / (A.1 - B.1)

theorem ellipse_properties (a b : ℝ) (x y : ℝ) (A B P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
(A_ne_lr_vertex : A ≠ (a, 0) ∧ A ≠ (-a, 0)) 
(h1 : ellipse_equation a b x y)
(h2 : focal_points F1 F2 a b)
(h3 : point_A_condition A F1 F2 = 4 + 2 * sqrt 3)
(h4 : max_area A F1 F2 = sqrt 3)
(h5 : slope (0,0) A * slope (0,0) B = -1/4)
(h6 : P = midpoint_P A B) :
abs (dist (0, 0) P) ∈ set.Icc (sqrt 2 / 2) (sqrt 2) :=
sorry

end ellipse_properties_l361_361519


namespace number_of_x_for_Q_eq_zero_l361_361689

noncomputable def Q (x : ℝ) : ℂ :=
  2 + complex.cos x + complex.I * complex.sin x -
  2 * complex.cos (2*x) - 2 * complex.I * complex.sin (2*x) +
  complex.cos (3*x) + complex.I * complex.sin (3*x)

theorem number_of_x_for_Q_eq_zero : 
  (finset.card {x : ℝ | 0 ≤ x ∧ x < 4 * real.pi ∧ Q x = 0}.to_finset) = 2 :=
by
  sorry

end number_of_x_for_Q_eq_zero_l361_361689


namespace determine_a_l361_361933

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.log (2 / (1 - x) + a)

theorem determine_a (a : ℝ) : is_odd_function (λ x, f x a) ↔ a = -1 :=
by
  sorry

end determine_a_l361_361933


namespace distance_to_point_is_17_l361_361825

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem distance_to_point_is_17 :
  distance_from_origin 8 (-15) = 17 :=
by
  sorry

end distance_to_point_is_17_l361_361825


namespace find_X_value_l361_361680

theorem find_X_value (X : ℝ) : 
  (1.5 * ((3.6 * 0.48 * 2.5) / (0.12 * X * 0.5)) = 1200.0000000000002) → 
  X = 0.225 :=
by
  sorry

end find_X_value_l361_361680


namespace smallest_n_2000_divides_a_n_l361_361855

theorem smallest_n_2000_divides_a_n (a : ℕ → ℤ) 
  (h_rec : ∀ n, n ≥ 1 → (n - 1) * a (n + 1) = (n + 1) * a n - 2 * (n - 1)) 
  (h2000 : 2000 ∣ a 1999) : 
  ∃ n, n ≥ 2 ∧ 2000 ∣ a n ∧ n = 249 := 
by 
  sorry

end smallest_n_2000_divides_a_n_l361_361855


namespace perpendicular_lines_find_a_l361_361754

theorem perpendicular_lines_find_a (a : ℝ) (h : ∀ (L₁ L₂: ℝ → ℝ → ℝ),
  L₁ = (λ x y, x - 3 * y + 2) ∧ L₂ = (λ x y, 3 * x - a * y - 1) ∧
    ∀ x y, L₁ x y = L₂ x y → L₁ x y = 0 ∧ L₂ x y = 0 → ∀ (m₁ m₂: ℝ), 
      m₁ = -((1 : ℝ)/(3 : ℝ)) ∧ m₂ = -((3 : ℝ)/a) → m₁ * m₂ = -1) :
  a = -1 :=
sorry

end perpendicular_lines_find_a_l361_361754


namespace cube_volume_in_pyramid_l361_361645

def pyramid_base_side_length := 2
def triangle_leg_length := pyramid_base_side_length / 2
def triangle_height := Real.sqrt (triangle_leg_length^2 + triangle_leg_length^2) -- Using Pythagorean theorem.
def pyramid_apex_height := triangle_height -- Since h^2 + 1 = 2 implies h = 1 after simplifying.
def cube_side_length := 1 -- Based on the cube fitting inside the given pyramid.

theorem cube_volume_in_pyramid (base_side : ℝ) (apex_height : ℝ) (cube_side : ℝ) :
  base_side = 2 →
  apex_height = Real.sqrt(2) →
  cube_side = 1 →
  cube_side^3 = 1 :=
by
  intros base_side_eq apex_height_eq cube_side_eq
  rw [cube_side_eq]
  norm_num
  sorry

end cube_volume_in_pyramid_l361_361645


namespace find_a_find_lambda_l361_361300

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.exp x + a)

theorem find_a (a : ℝ) (h : ∀ x, f a x = - f a (-x)) : a = 0 := by
  sorry

noncomputable def g (λ : ℝ) (x : ℝ) : ℝ := λ * x

theorem find_lambda (λ : ℝ) 
  (h : ∀ x ∈ set.Icc (2 : ℝ) 3, g λ x ≤ x * Real.log x / Real.log 2) : λ ≤ 1 := by
  sorry

end find_a_find_lambda_l361_361300


namespace intersection_M_N_l361_361756

def M : Set ℝ := { x | -5 < x ∧ x < 3 }
def N : Set ℝ := { x | -2 < x ∧ x < 4 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 3 } := 
by sorry

end intersection_M_N_l361_361756


namespace distance_from_origin_l361_361818

theorem distance_from_origin :
  let origin := (0, 0)
      pt := (8, -15)
  in Real.sqrt ((pt.1 - origin.1)^2 + (pt.2 - origin.2)^2) = 17 :=
by
  let origin := (0, 0)
  let pt := (8, -15)
  have h1 : pt.1 - origin.1 = 8 := by rfl
  have h2 : pt.2 - origin.2 = -15 := by rfl
  simp [h1, h2]
  -- sorry will be used to skip the actual proof steps and to ensure the code builds successfully
  sorry

end distance_from_origin_l361_361818


namespace matrix_determinant_is_one_l361_361266

noncomputable def given_matrix (α β γ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![cos α * cos β, cos α * sin β * cos γ, -sin α * sin γ],
    ![-sin β * cos γ, cos β, -sin β * sin γ],
    ![sin α * cos β, sin α * sin β * cos γ, cos α * cos γ]
  ]

theorem matrix_determinant_is_one (α β γ : ℝ) : 
  Matrix.det (given_matrix α β γ) = 1 := 
sorry

end matrix_determinant_is_one_l361_361266


namespace fraction_of_acid_in_third_flask_l361_361139

def mass_of_acid_first_flask := 10
def mass_of_acid_second_flask := 20
def mass_of_acid_third_flask := 30
def mass_of_acid_first_flask_with_water (w : ℝ) := mass_of_acid_first_flask / (mass_of_acid_first_flask + w) = 1 / 20
def mass_of_acid_second_flask_with_water (W w : ℝ) := mass_of_acid_second_flask / (mass_of_acid_second_flask + (W - w)) = 7 / 30
def mass_of_acid_third_flask_with_water (W : ℝ) := mass_of_acid_third_flask / (mass_of_acid_third_flask + W)

theorem fraction_of_acid_in_third_flask (W w : ℝ) (h1 : mass_of_acid_first_flask_with_water w) (h2 : mass_of_acid_second_flask_with_water W w) :
  mass_of_acid_third_flask_with_water W = 21 / 200 :=
by
  sorry

end fraction_of_acid_in_third_flask_l361_361139


namespace grade_distribution_sum_l361_361785

theorem grade_distribution_sum (a b c d : ℝ) (ha : a = 0.6) (hb : b = 0.25) (hc : c = 0.1) (hd : d = 0.05) :
  a + b + c + d = 1.0 :=
by
  -- Introduce the hypothesis
  rw [ha, hb, hc, hd]
  -- Now the goal simplifies to: 0.6 + 0.25 + 0.1 + 0.05 = 1.0
  sorry

end grade_distribution_sum_l361_361785


namespace salad_calories_l361_361421

-- Define individual calorie calculations
def romaine_calories (grams : ℕ) : Float :=
  (grams * 17) / 100

def iceberg_calories (grams : ℕ) : Float :=
  (grams * 14) / 100

def cucumber_calories (grams : ℕ) : Float :=
  (grams * 15) / 100

def cherry_tomatoes_calories (count : ℕ) : Float :=
  count * 3

def baby_carrots_calories (count : ℕ) : Float :=
  count * 4

def dressing_calories (servings : ℕ) : Float :=
  servings * 120

def croutons_calories (count : ℕ) : Float :=
  (count / 4) * 30

def parmesan_cheese_calories (grams : ℕ) : Float :=
  (grams * 420) / 100

-- Combine all calculations to prove the final number of calories
theorem salad_calories :
  romaine_calories 100 +
  iceberg_calories 75 +
  cucumber_calories 120 +
  cherry_tomatoes_calories 18 +
  baby_carrots_calories 24 +
  dressing_calories 3 +
  croutons_calories 36 +
  parmesan_cheese_calories 45 = 1014.5 :=
by
  -- Here would be the proof steps, but we use sorry to skip it
  sorry

end salad_calories_l361_361421


namespace prove_positions_l361_361381

def athlete : Type := { A : ℕ, B : ℕ, C : ℕ, D : ℕ, E : ℕ }

def first_prediction (x : athlete) : athlete :=
  { A := 1, B := 2, C := 3, D := 4, E := 5 }

def second_prediction (x : athlete) : athlete :=
  { A := 3, B := 4, C := 1, D := 5, E := 2 }

def actual_positions : athlete :=
  { A := 3, B := 2, C := 1, D := 4, E := 5 }

theorem prove_positions :
  (λ x, first_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 3 ∧
  (λ x, second_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 2 ∧
  actual_positions.A = 3 ∧ actual_positions.B = 2 ∧ actual_positions.C = 1 ∧ actual_positions.D = 4 ∧ actual_positions.E = 5 := 
  sorry

end prove_positions_l361_361381


namespace john_weight_end_l361_361427

def initial_weight : ℝ := 220
def loss_percentage : ℝ := 0.1
def weight_loss : ℝ := loss_percentage * initial_weight
def weight_gain_back : ℝ := 2
def net_weight_loss : ℝ := weight_loss - weight_gain_back
def final_weight : ℝ := initial_weight - net_weight_loss

theorem john_weight_end :
  final_weight = 200 := 
by 
  sorry

end john_weight_end_l361_361427


namespace sheep_sale_ratio_l361_361886

variables (G S G_sold S_sold : ℕ)
variables (total_animals : ℕ) (total_revenue revenue_from_goats revenue_from_sheep : ℤ)
variables (goat_price sheep_price : ℤ)

# The conditions
def goat_to_sheep_ratio (G S : ℕ) := G = (5/7 : ℚ) * S
def total_number_of_animals (G S : ℕ) := G + S = 360
def goats_sold (G_sold G : ℕ) := G_sold = G / 2
def revenue_from_g (G_sold : ℕ) (goat_price : ℤ) := revenue_from_goats = G_sold * goat_price
def revenue_from_s (S_sold : ℕ) (sheep_price : ℤ) := revenue_from_sheep = S_sold * sheep_price
def total_revenue_eq (revenue_from_goats revenue_from_sheep : ℤ) := total_revenue = revenue_from_goats + revenue_from_sheep
def total_revenue_amount := total_revenue = 7200
def goat_price_amount := goat_price = 40
def sheep_price_amount := sheep_price = 30

# The theorem to prove
theorem sheep_sale_ratio (G S G_sold S_sold : ℕ) (total_animals : ℕ) (total_revenue revenue_from_goats revenue_from_sheep : ℤ) (goat_price sheep_price : ℤ)
  (h1 : goat_to_sheep_ratio G S) 
  (h2 : total_number_of_animals G S)
  (h3 : goats_sold G_sold G)
  (h4 : revenue_from_g G_sold goat_price)
  (h5 : revenue_from_s S_sold sheep_price)
  (h6 : total_revenue_eq revenue_from_goats revenue_from_sheep)
  (h7 : total_revenue_amount)
  (h8 : goat_price_amount)
  (h9 : sheep_price_amount) :
  S_sold / S = 2 / 3 :=
sorry

end sheep_sale_ratio_l361_361886


namespace distance_AF_minus_AP_l361_361039

noncomputable def distance_sq (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 - x2)^2 + (y1 - y2)^2

theorem distance_AF_minus_AP (a b x0 y0 : ℝ)
  (h_ellipse : x0^2 / a^2 + y0^2 / b^2 = 1)
  (h_circle : ∀ P : ℝ × ℝ, (x0 - P.1)^2 + (y0 - P.2)^2 = b^2)
  (e : ℝ := sqrt (1 - b^2 / a^2))
  (A : ℝ × ℝ := (x0, y0))
  (F : ℝ × ℝ := (-sqrt (a^2 - b^2), 0)) :
  abs (distance_sq A.1 A.2 F.1 F.2) - abs (distance_sq A.1 A.2 (x0 * e) 0) = a :=
by
  sorry

end distance_AF_minus_AP_l361_361039


namespace power_and_inverse_function_l361_361339

theorem power_and_inverse_function (α k : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f(x) = x^α) (h₂ : ∀ x, f(x) = k / x) : 
  f = λ x, x^(-1) :=
by sorry

end power_and_inverse_function_l361_361339


namespace fraction_of_area_above_line_l361_361172

open Set

noncomputable def square_vertices : Set (ℝ × ℝ) := {(2, 1), (5, 1), (2, 4), (5, 4)}

noncomputable def line_eq (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

noncomputable def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

theorem fraction_of_area_above_line :
  ∃ (a b c d : ℝ × ℝ), {(a, b), (c, d)} = square_vertices ∧
      triangle_area (2, 1) (5, 1) (5, 3) = 3 ∧ 
      (5 - 2) * (4 - 1) = 9 ∧
      1 - (3 / 9) = 2 / 3 :=
sorry

end fraction_of_area_above_line_l361_361172


namespace shorter_lateral_side_l361_361939

-- Definitions for sides and properties of the trapezoid
variables (AB CD AD BC : ℝ)
variables (h : ℝ)

-- Problem conditions
def trapezoid_conditions : Prop :=
  AD = 8 ∧
  CD - AB = 10 ∧
  90 = 90  -- Line intersection forming a right angle is inherent

-- Theorem statement to prove the shorter lateral side is 6
theorem shorter_lateral_side (h : ℝ) (AB CD AD BC : ℝ)
  (h_trapezoid_conditions : trapezoid_conditions AB CD AD BC) :
  BC = 6 :=
by {
  -- Proof omitted for brevity
  sorry
}

end shorter_lateral_side_l361_361939


namespace mary_can_work_max_hours_l361_361063

-- Define Mary's pay conditions and the problem statement
def mary_max_hours (regular_rate : ℝ) (overtime_multiplier : ℝ) (max_earnings : ℝ) (first_20_hours : ℝ) : ℝ :=
  let overtime_rate := overtime_multiplier * regular_rate
  let first_20_earnings := first_20_hours * regular_rate
  let remaining_earnings := max_earnings - first_20_earnings
  let overtime_hours := remaining_earnings / overtime_rate
  first_20_hours + overtime_hours

theorem mary_can_work_max_hours :
  ∀ (regular_rate overtime_multiplier max_earnings first_20_hours total_hours : ℝ),
    regular_rate = 8 →
    overtime_multiplier = 1.25 →
    max_earnings = 360 →
    first_20_hours = 20 →
    total_hours = 40 →
    mary_max_hours regular_rate overtime_multiplier max_earnings first_20_hours = total_hours :=
by
  intros
  subst_vars
  simp only [mary_max_hours]
  sorry

end mary_can_work_max_hours_l361_361063


namespace actual_positions_correct_l361_361355

def athletes := {A, B, C, D, E}
def p1 : athletes → ℕ
| A := 1
| B := 2
| C := 3
| D := 4
| E := 5

def p2 : athletes → ℕ
| C := 1
| E := 2
| A := 3
| B := 4
| D := 5

def actual_positions : athletes → ℕ
| A := 3
| B := 2
| C := 1
| D := 4
| E := 5

theorem actual_positions_correct :
  (∑ a in athletes, ite (actual_positions a = p1 a) 1 0 = 3) ∧ 
  (∑ a in athletes, ite (actual_positions a = p2 a) 1 0 = 2) :=
by
  sorry

end actual_positions_correct_l361_361355


namespace parallel_lines_condition_l361_361188

theorem parallel_lines_condition {a : ℝ} :
  (∀ x y : ℝ, a * x + 2 * y + 3 * a = 0) ∧ (∀ x y : ℝ, 3 * x + (a - 1) * y = a - 7) ↔ a = 3 :=
by
  sorry

end parallel_lines_condition_l361_361188


namespace last_digit_of_difference_is_5_l361_361152

def last_digit_of_difference : ℕ :=
  (list.prod (list.range 14).tail - list.prod [1, 3, 5, 7, 9, 11, 13]) % 10

theorem last_digit_of_difference_is_5 :
  last_digit_of_difference = 5 :=
by
  sorry

end last_digit_of_difference_is_5_l361_361152


namespace sqrt_mul_sqrt_l361_361991

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361991


namespace Scheherazade_condition_failure_l361_361073

theorem Scheherazade_condition_failure :
  ∀ (circle : Type) (points : finset circle)
  (initial_points_count : ℕ) (days : ℕ) (cut : finset circle → finset circle → finset circle),
  initial_points_count = 1001 →
  (∀ p : finset circle, p.card = 1001 → ∀ n < 1000, ∃ q, cut p q = p.sdiff q ∧ q.card < p.card) →
  (∀ p : finset circle, p.card < 1000 → ∀ n < (1000 - p.card), ∃ q, cut p q = p.sdiff q ∧ q.card < p.card) →
  days = 1998 → 
  ∃ final_shape : finset circle, ¬ ∀ q, cut final_shape q = final_shape.sdiff q → q.card < final_shape.card := 
by
  sorry

end Scheherazade_condition_failure_l361_361073


namespace correct_statement_C_l361_361166

theorem correct_statement_C : (∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5)) :=
by
  sorry

end correct_statement_C_l361_361166


namespace probability_floor_log_approx_l361_361050

open Real

noncomputable def probability_floor_log_condition : ℝ :=
  (probability (λ x : ℝ, 0 < x ∧ x < 1) (λ x, (⌊log (10 : ℝ) (9 * x)⌋ = ⌊log (10 : ℝ) x⌋)))

theorem probability_floor_log_approx :
  abs (probability_floor_log_condition - 0.0136) < 0.0001 := sorry

end probability_floor_log_approx_l361_361050


namespace simplify_fraction_l361_361090

theorem simplify_fraction:
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by
  sorry

end simplify_fraction_l361_361090


namespace compute_z_six_l361_361036

def z : ℂ := (-Real.sqrt 5 + Complex.i) / 2

theorem compute_z_six : z^6 = -1 :=
by
  sorry

end compute_z_six_l361_361036


namespace quadratic_factoring_even_a_l361_361522

theorem quadratic_factoring_even_a (a : ℤ) :
  (∃ (m p n q : ℤ), 21 * x^2 + a * x + 21 = (m * x + n) * (p * x + q) ∧ m * p = 21 ∧ n * q = 21 ∧ (∃ (k : ℤ), a = 2 * k)) :=
sorry

end quadratic_factoring_even_a_l361_361522


namespace final_weight_is_200_l361_361428

def initial_weight : ℕ := 220
def percentage_lost : ℕ := 10
def weight_gained : ℕ := 2

theorem final_weight_is_200 :
  initial_weight - (initial_weight * percentage_lost / 100) + weight_gained = 200 := by
  sorry

end final_weight_is_200_l361_361428


namespace triangle_tangency_condition_l361_361859

open EuclideanGeometry

variables {A B C D E F G : Point} 
variables {ℓBC : Line}

-- Conditions
def is_isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C

def points_on_minor_arcs (A B C D E : Point) : Prop := ∃ O : Point, O ∈ interiorConvexHull ({A, B, C} : Set Point) ∧
  D ∈ arc O A B ∧ E ∈ arc O A C

def lines_intersecting (A D B C F : Point) (ℓBC : Line) : Prop := 
  line_through A D ∩ ℓBC = {F} 

def line_intersects_circumcircle (A E F D E G : Point) : Prop := 
  ∃ γ : Circle, F ∈ γ ∧ D ∈ γ ∧ E ∈ γ ∧ G ∈ (γ ∩ line_through A E)

-- The proof statement
theorem triangle_tangency_condition (A B C D E F G : Point) (ℓBC : Line) 
  (h_isosceles : is_isosceles_triangle A B C)
  (h_minor_arcs : points_on_minor_arcs A B C D E)
  (h_intersecting_lines : lines_intersecting A D B C F ℓBC)
  (h_intersects_circumcircle : line_intersects_circumcircle A E F D E G) :
  tangent_line_to_circle A C (circumcircle E C G) :=
sorry

end triangle_tangency_condition_l361_361859


namespace prove_a2_minus_b2_l361_361778

theorem prove_a2_minus_b2 : 
  ∀ (a b : ℚ), 
  a + b = 9 / 17 ∧ a - b = 1 / 51 → a^2 - b^2 = 3 / 289 :=
by
  intros a b h
  cases' h
  sorry

end prove_a2_minus_b2_l361_361778


namespace all_positive_rationals_are_red_l361_361642

-- Define the property of being red for rational numbers
def is_red (x : ℚ) : Prop :=
  ∃ n : ℕ, ∃ (f : ℕ → ℚ), f 0 = 1 ∧ (∀ m : ℕ, f (m + 1) = f m + 1 ∨ f (m + 1) = f m / (f m + 1)) ∧ f n = x

-- Proposition stating that all positive rational numbers are red
theorem all_positive_rationals_are_red :
  ∀ x : ℚ, 0 < x → is_red x :=
  by sorry

end all_positive_rationals_are_red_l361_361642


namespace simplify_f_eval_f_specific_angle_cos_pi_plus_alpha_l361_361731

noncomputable theory

variables (α : ℝ) (f : ℝ → ℝ)
hypothesis h3 : (2 * Int.pi = Real.pi)

def quadrant_3 (α : ℝ) : Prop := 
  π < α ∧ α < 3 * π / 2

def f (α : ℝ) : ℝ :=
  (sin (3 * π / 2 - α) * cos (π / 2 - α) * tan (-α + π))
  / (sin (π / 2 + α) * tan (2 * π - α)) 

theorem simplify_f :
  ∀ α, quadrant_3 α → f α = -sin α :=
by
  intros α h
  sorry

theorem eval_f_specific_angle:
  ∀ α, quadrant_3 α → α = -32/3 * π → f α = (sqrt 3)/2 :=
by
  intros α h_hp h
  sorry

theorem cos_pi_plus_alpha:
  ∀ α, quadrant_3 α → (f α = 2 * sqrt 6 / 5) → cos (π + α) = 1/5 :=
by
  intros α h_hp h
  sorry

end simplify_f_eval_f_specific_angle_cos_pi_plus_alpha_l361_361731


namespace area_of_triangle_l361_361201

noncomputable def parabola := { x : ℝ // x = 0 ∧ ∀ y : ℝ, y^2 = 4 * x }

noncomputable def focus (p : parabola) : ℝ × ℝ := (1, 0)

noncomputable def origin : ℝ × ℝ := (0, 0)

noncomputable def point_a (p : parabola) (f : ℝ × ℝ) : ℝ × ℝ := sorry

noncomputable def point_b (p : parabola) (f : ℝ × ℝ) : ℝ × ℝ := sorry

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem area_of_triangle {p : parabola} {f : ℝ × ℝ} {a b o : ℝ × ℝ} (h₁ : a = point_a p f)
  (h₂ : b = point_b p f) (h₃ : o = origin) (h₄ : distance a f = 3) :
  let θ := real.angle (1 : ℝ × ℝ) (a - f)
  let sin_θ := real.sqrt (1 - real.cos θ^2)
  let ab := distance a b
  let area := 1 / 2 * 1 * ab * sin_θ
  area = 3 * real.sqrt 2 / 2 :=
sorry

end area_of_triangle_l361_361201


namespace sqrt_49_mul_sqrt_25_l361_361994

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l361_361994


namespace james_more_balloons_l361_361420

theorem james_more_balloons (james_balloons : ℕ) (amy_balloons : ℕ) (felix_balloons : ℕ) :
  james_balloons = 1222 → amy_balloons = 513 → felix_balloons = 687 →
  james_balloons - (amy_balloons + felix_balloons) = 22 :=
by
  intros h_james h_amy h_felix
  rw [h_james, h_amy, h_felix]
  sorry

end james_more_balloons_l361_361420


namespace sum_reciprocal_geom_seq_l361_361841

noncomputable def common_ratio (a1 a4 : ℝ) : ℝ :=
  real.cbrt (a4 / a1)

theorem sum_reciprocal_geom_seq (a1 a4 : ℝ) (ha1 : a1 = 3) (ha4 : a4 = 24) :
  let q := common_ratio a1 a4 in
  let a_n (n : ℕ) := a1 * q ^ (n - 1) in
  let inv_a_n (n : ℕ) := 1 / a_n n in
  ∑ i in finset.range 5, inv_a_n (i + 1) = 31 / 48 :=
by
  sorry

end sum_reciprocal_geom_seq_l361_361841


namespace actual_time_of_storm_l361_361926

theorem actual_time_of_storm
  (malfunctioned_hours_tens_digit : ℕ)
  (malfunctioned_hours_units_digit : ℕ)
  (malfunctioned_minutes_tens_digit : ℕ)
  (malfunctioned_minutes_units_digit : ℕ)
  (original_time : ℕ × ℕ)
  (hours_tens_digit : ℕ := 2)
  (hours_units_digit : ℕ := 0)
  (minutes_tens_digit : ℕ := 0)
  (minutes_units_digit : ℕ := 9) :
  (malfunctioned_hours_tens_digit = hours_tens_digit + 1 ∨ malfunctioned_hours_tens_digit = hours_tens_digit - 1) →
  (malfunctioned_hours_units_digit = hours_units_digit + 1 ∨ malfunctioned_hours_units_digit = hours_units_digit - 1) →
  (malfunctioned_minutes_tens_digit = minutes_tens_digit + 1 ∨ malfunctioned_minutes_tens_digit = minutes_tens_digit - 1) →
  (malfunctioned_minutes_units_digit = minutes_units_digit + 1 ∨ malfunctioned_minutes_units_digit = minutes_units_digit - 1) →
  original_time = (11, 18) :=
by
  sorry

end actual_time_of_storm_l361_361926


namespace complementary_angles_l361_361727

theorem complementary_angles (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 :=
by 
  sorry

end complementary_angles_l361_361727


namespace fraction_of_acid_in_third_flask_l361_361141

def mass_of_acid_first_flask := 10
def mass_of_acid_second_flask := 20
def mass_of_acid_third_flask := 30
def mass_of_acid_first_flask_with_water (w : ℝ) := mass_of_acid_first_flask / (mass_of_acid_first_flask + w) = 1 / 20
def mass_of_acid_second_flask_with_water (W w : ℝ) := mass_of_acid_second_flask / (mass_of_acid_second_flask + (W - w)) = 7 / 30
def mass_of_acid_third_flask_with_water (W : ℝ) := mass_of_acid_third_flask / (mass_of_acid_third_flask + W)

theorem fraction_of_acid_in_third_flask (W w : ℝ) (h1 : mass_of_acid_first_flask_with_water w) (h2 : mass_of_acid_second_flask_with_water W w) :
  mass_of_acid_third_flask_with_water W = 21 / 200 :=
by
  sorry

end fraction_of_acid_in_third_flask_l361_361141


namespace feeding_times_per_day_l361_361489

-- Definitions for the given conditions
def number_of_puppies : ℕ := 7
def total_portions : ℕ := 105
def number_of_days : ℕ := 5

-- Theorem to prove the answer to the question
theorem feeding_times_per_day : 
  let portions_per_day := total_portions / number_of_days in
  let times_per_puppy := portions_per_day / number_of_puppies in
  times_per_puppy = 3 :=
by
  -- We should provide the proof here, but we will use 'sorry' to skip it
  sorry

end feeding_times_per_day_l361_361489


namespace locus_of_M_is_circle_l361_361872

-- Define the points A, B, and C on the circle
variable {α : Type*} [MetricSpace α]
variables {o : α} {r : ℝ} {circle : Set α} (h : circle = Metric.Sphere o r)

variables (B C : α) (hB : B ∈ circle) (hC : C ∈ circle)

-- Define A as a variable point on the circle
variable (A : α) (hA : A ∈ circle)

-- Define midpoint K and foot of perpendicular M
def midpoint (A B : α) : α := (A + B) / 2
def perp_foot (K A C : α) : α := sorry  -- Need the definition for foot of perpendicular

noncomputable def K : α := midpoint A B
noncomputable def M : α := perp_foot K A C

-- Locus of points M on circle with diameter BC
theorem locus_of_M_is_circle (A : α) (hA : A ∈ circle) :
  ∃ M: α, M ∈ Metric.Sphere ((B + C) / 2) (dist B C / 2) :=
sorry

end locus_of_M_is_circle_l361_361872


namespace part1_part2_l361_361748

def f (x a : ℝ) : ℝ := x / Real.exp x + a * (x - 1) ^ 2

-- Part (1)
theorem part1 (x : ℝ) : f x 0 ≤ 1 / Real.exp 1 := sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∃ x, f x a = 1 / 2 ∧ (∀ y ≠ x, f y a < 1 / 2)) : a ∈ Set.union (Set.Iio (1 / (2 * Real.exp 1))) (Set.Ioc (1 / (2 * Real.exp 1)) (1 / 2)) := sorry

end part1_part2_l361_361748


namespace sum_of_numbers_l361_361941

noncomputable def mean (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem sum_of_numbers (a b c : ℕ) (h1 : mean a b c = a + 8)
  (h2 : mean a b c = c - 20) (h3 : b = 7) (h_le1 : a ≤ b) (h_le2 : b ≤ c) :
  a + b + c = 57 :=
by {
  sorry
}

end sum_of_numbers_l361_361941


namespace closest_integer_to_a2013_l361_361530

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 100 ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) + (1 / a (n + 1))

theorem closest_integer_to_a2013 (a : ℕ → ℝ) (h : seq a) : abs (a 2013 - 118) < 0.5 :=
sorry

end closest_integer_to_a2013_l361_361530


namespace equilateral_triangle_m_equals_l361_361737

theorem equilateral_triangle_m_equals (m : ℝ) :
  (∀ A B : ℝ → ℝ → Prop,
    (∃ x y : ℝ, A x y ∧ B x y ∧ x - y + m = 0 ∧ x^2 + y^2 = 1) ∧
    (∃ O : ℝ → ℝ → Prop, O 0 0 ∧ ∀ X Y : ℝ, A X 0 → B Y 0 →
     (O 0 0 ∧ (X = -Y))) →  -- representing A and B forming equilateral triangle with O
  m = sqrt(6) / 2 ∨ m = -sqrt(6) / 2) :=
sorry

end equilateral_triangle_m_equals_l361_361737


namespace sqrt_mul_sqrt_l361_361992

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361992


namespace find_angle_BDC_l361_361914

-- Definition of the conditions
variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Assume that the points form congruent triangles
axiom congruent_triangles : ∃ (ABC : triangle A B C) (ACD : triangle A C D), 
  congruent ABC ACD ∧ ABC.has_same_sides AC ∧ ACD.has_same_sides AD

-- Assume that AB = AC = AD
axiom side_lengths : ∃ (AB AC AD : ℝ), AB = AC ∧ AC = AD

-- Assume that ∠BAC = 30°
axiom angle_BAC : ∃ (angle : ℝ), angle = 30

-- Proving that ∠BDC = 15°
theorem find_angle_BDC : ∃ (angle_BDC : ℝ), angle_BDC = 15 := sorry

end find_angle_BDC_l361_361914


namespace slope_ratio_l361_361147

theorem slope_ratio (s t k b : ℝ) 
  (h1: b = -12 * s)
  (h2: b = k - 7) 
  (ht: t = (7 - k) / 7) 
  (hs: s = (7 - k) / 12): 
  s / t = 7 / 12 := 
  sorry

end slope_ratio_l361_361147


namespace closest_integer_to_a2013_l361_361532

noncomputable def sequence (n : ℕ) : ℝ :=
  nat.rec_on n 100 (λ k ak, ak + 1 / ak)

theorem closest_integer_to_a2013 : ∃ (z : ℤ), z = 118 ∧ abs (sequence 2013 - z) = min (abs (sequence 2013 - (z - 1))) (abs (sequence 2013 - (z + 1))) :=
begin
  sorry
end

end closest_integer_to_a2013_l361_361532


namespace cement_mixture_weight_l361_361574

theorem cement_mixture_weight :
  ∃ (W : ℝ), (1/4 * W + 2/5 * W + 14 = W) ∧ (W = 40) :=
by
  existsi 40
  split
  sorry -- This is where the rest of the proof steps would go
  refl

end cement_mixture_weight_l361_361574


namespace greatest_integer_jo_thinking_of_l361_361423

theorem greatest_integer_jo_thinking_of :
  ∃ n : ℕ, n < 150 ∧ (∃ k : ℕ, n = 9 * k - 1) ∧ (∃ m : ℕ, n = 5 * m - 2) ∧ n = 143 :=
by
  sorry

end greatest_integer_jo_thinking_of_l361_361423


namespace conjugate_of_z_l361_361442

-- Define the imaginary unit i.
def I : ℂ := Complex.i

-- Define z as given in the problem
def z : ℂ := I / (1 + I)

-- State the theorem that needs to be proved
theorem conjugate_of_z : Complex.conj z = (1/2 : ℂ) - (1/2 : ℂ) * I :=
sorry

end conjugate_of_z_l361_361442


namespace question1_question2_l361_361934

noncomputable def f (a : ℝ) (x : ℝ) := real.sqrt ((1 - a^2) * x^2 + 3 * (1 - a) * x + 6)

-- Question 1: Prove that a = 2 given the conditions
theorem question1 : ∀ a : ℝ, (∀ x ∈ set.Icc (-2:ℝ) 1, f a x = real.sqrt ((1 - a ^ 2) * x ^ 2 + 3 * (1 - a) * x + 6)) → a = 2 :=
begin
  sorry
end

-- Question 2: Prove that -5/11 <= a <= 1 given the conditions
theorem question2 : ∀ a : ℝ, (∀ x : ℝ, f a x = real.sqrt ((1 - a ^ 2) * x ^ 2 + 3 * (1 - a) * x + 6) ) → -5 / 11 ≤ a ∧ a ≤ 1 :=
begin
  sorry
end

end question1_question2_l361_361934


namespace perfect_even_multiples_of_3_under_3000_l361_361332

theorem perfect_even_multiples_of_3_under_3000 :
  ∃ n : ℕ, n = 9 ∧ ∀ (k : ℕ), (36 * k^2 < 3000) → (36 * k^2) % 2 = 0 ∧ (36 * k^2) % 3 = 0 ∧ ∃ m : ℕ, m^2 = 36 * k^2 :=
by
  sorry

end perfect_even_multiples_of_3_under_3000_l361_361332


namespace median_bisector_altitude_l361_361480

theorem median_bisector_altitude (A B C M : Point) (h_iso : is_isosceles_triangle A B C) (h_median : median_to_base B M A C) : angle_bisector B M A C ∧ altitude B M A C :=
sorry

end median_bisector_altitude_l361_361480


namespace number_of_inscribed_discs_l361_361854

/-- Let 𝓛 be a finite collection of lines in the plane in general position 
    (no two lines in 𝓛 are parallel and no three are concurrent). Then the number
    of inscribed discs that are intersected by no line in 𝓛 is given by 
    (|𝓛| - 1) * (|𝓛| - 2) / 2. -/
theorem number_of_inscribed_discs (𝓛 : Finset (Line ℝ)) 
  (h1 : ∀ l₁ l₂ ∈ 𝓛, l₁ ≠ l₂ → ¬ l₁ ∥ l₂)
  (h2 : ∀ l₁ l₂ l₃ ∈ 𝓛, l₁ ≠ l₂ → l₂ ≠ l₃ → l₁ ≠ l₃ → ¬ AreConcurrent ℝ l₁ l₂ l₃) :
  (disjoint_inscribed_discs_count 𝓛) = ((|𝓛| - 1) * (|𝓛| - 2)) / 2 :=
sorry

end number_of_inscribed_discs_l361_361854


namespace sqrt_mul_sqrt_l361_361988

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_mul_sqrt_l361_361988


namespace simplify_fraction_l361_361091

theorem simplify_fraction:
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by
  sorry

end simplify_fraction_l361_361091


namespace max_squares_at_a1_bksq_l361_361544

noncomputable def maximizePerfectSquares (a b : ℕ) : Prop := 
a ≠ b ∧ 
(∃ k : ℕ, k ≠ 1 ∧ b = k^2) ∧ 
a = 1

theorem max_squares_at_a1_bksq (a b : ℕ) : maximizePerfectSquares a b := 
by 
  sorry

end max_squares_at_a1_bksq_l361_361544


namespace marked_price_percentage_l361_361609

theorem marked_price_percentage
  (CP MP SP : ℝ)
  (h_profit : SP = 1.08 * CP)
  (h_discount : SP = 0.8307692307692308 * MP) :
  MP = CP * 1.3 :=
by sorry

end marked_price_percentage_l361_361609


namespace distance_from_origin_l361_361819

theorem distance_from_origin :
  let origin := (0, 0)
      pt := (8, -15)
  in Real.sqrt ((pt.1 - origin.1)^2 + (pt.2 - origin.2)^2) = 17 :=
by
  let origin := (0, 0)
  let pt := (8, -15)
  have h1 : pt.1 - origin.1 = 8 := by rfl
  have h2 : pt.2 - origin.2 = -15 := by rfl
  simp [h1, h2]
  -- sorry will be used to skip the actual proof steps and to ensure the code builds successfully
  sorry

end distance_from_origin_l361_361819


namespace area_of_triangle_BXC_l361_361972

-- Define the basic conditions of the problem
variables {AB CD : ℕ}
variables {h: ℝ}
variables {AreaABCD : ℝ}
variables {ratio: ℝ}
variables {AreaADC : ℝ}
variables {AreaABC : ℝ}
variables {HeightDXC HeightAXB : ℝ}
variables {AreaAXB : ℝ}
variables {AreaBXC : ℝ}

-- Given conditions
def trapezoid_conditions (AB CD : ℕ) (AreaABCD : ℝ) :=
  AB = 25 ∧ CD = 40 ∧ AreaABCD = 520

-- Prove the area of triangle BXC
theorem area_of_triangle_BXC (h: ℝ) (HeightDXC HeightAXB : ℝ)
  (AreaBXC AreaABC : ℝ):
  trapezoid_conditions 25 40 520 →
  AreaBXC = AreaABC - (1/2) * 25 * HeightAXB :=
  begin
    -- Assuming necessary steps for height and area calculations
    let h := 16,
    let AreaABC := 200,
    let HeightAXB := 80 / 13,
    let AreaAXB := 1000 / 13,
    sorry
  end

end area_of_triangle_BXC_l361_361972


namespace sequence_length_137_l361_361550

theorem sequence_length_137 : 
  ∃ (a : ℕ → ℕ) (k : ℕ), (strict_mono a) ∧ (∀ i < k, a i ≥ 0) ∧ 
  (∑ i in range k, 2 ^ (a i) = (2 ^ 289 + 1) / (2 ^ 17 + 1)) ∧ 
  k = 137 :=
sorry

end sequence_length_137_l361_361550


namespace kelsey_more_than_ekon_l361_361555

theorem kelsey_more_than_ekon :
  ∃ (K E U : ℕ), (K = 160) ∧ (E = U - 17) ∧ (K + E + U = 411) ∧ (K - E = 43) :=
by
  sorry

end kelsey_more_than_ekon_l361_361555


namespace line_divides_rectangle_l361_361500

theorem line_divides_rectangle (c : ℝ) :
  let area := 2 * 3 in
  let half_area := area / 2 in
  let base := 4 - c in
  let height := 3 in
  let triangle_area := (base * height) / 2 in
  triangle_area = half_area → c = 2 :=
by
  intros area half_area base height triangle_area h
  sorry

end line_divides_rectangle_l361_361500


namespace find_smallest_m_l361_361155

theorem find_smallest_m : ∃ m : ℕ, m > 0 ∧ (790 * m ≡ 1430 * m [MOD 30]) ∧ ∀ n : ℕ, n > 0 ∧ (790 * n ≡ 1430 * n [MOD 30]) → m ≤ n :=
by
  sorry

end find_smallest_m_l361_361155


namespace number_of_valid_sequences_l361_361334

theorem number_of_valid_sequences : 
  (∑ (a b c d e : ℕ) in {a | 0 < a} × {b | 0 < b} × {c | 0 < c} × {d | 0 < d} × {e | 0 < e}, 
  if a * b * c * d * e ≤ a + b + c + d + e ∧ a + b + c + d + e ≤ 10 then 1 else 0) = 116 :=
by { sorry }

end number_of_valid_sequences_l361_361334


namespace sum_powers_l361_361736

noncomputable def f (a b : ℂ) (x : ℂ) := log ((3^x + 1 : ℂ)) / log (1 / 3 : ℂ) + (1 / 2 : ℂ) * a * b * x
noncomputable def g (a b : ℂ) (x : ℂ) := 2^x + (a + b) / 2^x

axiom even_function_f (a b : ℂ) : ∀ x : ℂ, f a b x = f a b (-x)
axiom odd_function_g (a b : ℂ) : ∀ x : ℂ, g a b (-x) = -g a b x

theorem sum_powers (a b : ℂ) (h₁ : even_function_f a b) (h₂ : odd_function_g a b) :
  (a + b) + (a^2 + b^2) + (a^3 + b^3) + ∑ i in finset.range 98, (a^(i + 4) + b^(i + 4)) = -1 := sorry

end sum_powers_l361_361736


namespace portia_high_school_students_l361_361892

variables (P L M : ℕ)
axiom h1 : P = 4 * L
axiom h2 : P = 2 * M
axiom h3 : P + L + M = 4800

theorem portia_high_school_students : P = 2740 :=
by sorry

end portia_high_school_students_l361_361892


namespace sufficiently_large_n_has_three_prime_factors_l361_361084

-- Define what it means for an integer to have at least 3 distinct prime factors
def has_at_least_three_distinct_prime_factors (k : ℕ) : Prop :=
  (∃ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ prime p1 ∧ prime p2 ∧ prime p3 ∧ 
  (p1 ∣ k) ∧ (p2 ∣ k) ∧ (p3 ∣ k))

-- State the main theorem
theorem sufficiently_large_n_has_three_prime_factors (n : ℕ) (h : n > 0) (sufficiently_large : ∀ k, n > k) :
  ∃ m ∈ finset.range (10), has_at_least_three_distinct_prime_factors (n + m) :=
sorry

end sufficiently_large_n_has_three_prime_factors_l361_361084


namespace problem_statement_l361_361114

variable {a : ℕ+ → ℝ} 

theorem problem_statement (h : ∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) :
  (∀ n : ℕ+, a (n + 1) < a n) ∧ -- Sequence is decreasing (original proposition)
  (∀ (a : ℕ+ → ℝ), (∀ n : ℕ+, a (n + 1) < a n) → (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n)) ∧ -- Inverse
  ((∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) → (∀ n : ℕ+, a (n + 1) < a n)) ∧ -- Converse
  ((∀ (a : ℕ+ → ℝ), (∀ n : ℕ+, a (n + 1) < a n) → (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n))) -- Contrapositive
:= by
  sorry

end problem_statement_l361_361114


namespace max_roads_15_cities_l361_361538

def max_roads (n : ℕ) (cities : ℕ) (connection_rule : ℕ → ℕ → Prop) : ℕ :=
  if cities = 15 then 85 else sorry

theorem max_roads_15_cities : max_roads 15 15 (λ a b, a ≠ b) = 85 :=
begin
  sorry
end

end max_roads_15_cities_l361_361538


namespace distance_from_origin_to_point_l361_361832

open Real

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 →
  y = -15 →
  sqrt (x^2 + y^2) = 17 :=
by
  intros hx hy
  rw [hx, hy]
  norm_num
  rw ←sqrt_sq (show 0 ≤ 289 by norm_num)
  norm_num

end distance_from_origin_to_point_l361_361832


namespace sqrt_one_ninth_l361_361956

theorem sqrt_one_ninth : sqrt (1 / 9) = 1 / 3 ∨ sqrt (1 / 9) = - (1 / 3) :=
by
  sorry

end sqrt_one_ninth_l361_361956


namespace wrapping_paper_area_l361_361589

theorem wrapping_paper_area (a b h : ℝ) (h_neq : a ≠ b) : 
  ∃ (side_length : ℝ), side_length = a + 2 * h ∧ (side_length)^2 = (a + 2 * h)^2 :=
by
  let side_length := a + 2 * h
  use side_length
  split
  · refl
  · simp [side_length]

end wrapping_paper_area_l361_361589


namespace Xiao_Ming_vertical_height_increase_l361_361171

noncomputable def vertical_height_increase (slope_ratio_v slope_ratio_h : ℝ) (distance : ℝ) : ℝ :=
  let x := distance / (Real.sqrt (1 + (slope_ratio_h / slope_ratio_v)^2))
  x

theorem Xiao_Ming_vertical_height_increase
  (slope_ratio_v slope_ratio_h distance : ℝ)
  (h_ratio : slope_ratio_v = 1)
  (h_ratio2 : slope_ratio_h = 2.4)
  (h_distance : distance = 130) :
  vertical_height_increase slope_ratio_v slope_ratio_h distance = 50 :=
by
  unfold vertical_height_increase
  rw [h_ratio, h_ratio2, h_distance]
  sorry

end Xiao_Ming_vertical_height_increase_l361_361171


namespace distance_origin_to_point_l361_361795

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l361_361795


namespace total_coins_l361_361062

-- Define the number of stacks and the number of coins per stack
def stacks : ℕ := 5
def coins_per_stack : ℕ := 3

-- State the theorem to prove the total number of coins
theorem total_coins (s c : ℕ) (hs : s = stacks) (hc : c = coins_per_stack) : s * c = 15 :=
by
  -- Proof is omitted
  sorry

end total_coins_l361_361062


namespace exact_number_of_unit_size_cubes_l361_361149

theorem exact_number_of_unit_size_cubes (x : ℕ) (h : 0 < x) : 
  let l := 5 * x in
  let w := 5 * (x + 1) in
  let h := 5 * (x + 2) in
  l * w * h = 25 * x ^ 3 + 50 * x ^ 2 + 125 * x :=
by
  sorry

end exact_number_of_unit_size_cubes_l361_361149


namespace median_not_affected_l361_361343

-- Define the context for the problem
structure Scores where
  scores : Fin 5 → ℤ
  pairwise_different : ∀ i j, i ≠ j → scores i ≠ scores j

-- Function to calculate the median of 5 different scores
def median (ss : Fin 5 → ℤ) : ℤ := 
  let sorted_scores := List.sort (Finset.univ.val.map ss)
  sorted_scores.nthLe 2 sorry  -- 2 is the index of the median in a sorted list of 5 elements

-- Define the property that the highest score is reduced by 1
def highest_score_decreased (ss : Fin 5 → ℤ) : Fin 5 → ℤ :=
  let max_score := Finset.max' (Finset.univ.image ss) sorry
  fun i => if ss i = max_score then max_score - 1 else ss i

-- Theorem to prove
theorem median_not_affected (ss : Fin 5 → ℤ) (h : Scores ss) :
  median ss = median (highest_score_decreased ss) :=
sorry

end median_not_affected_l361_361343


namespace even_function_with_period_l361_361224

-- Define the functions
def f (x : ℝ) := cos (2 * x + π / 2)
def g (x : ℝ) := sin (2 * x) ^ 2 - cos (2 * x) ^ 2
def h (x : ℝ) := sin (2 * x) + cos (2 * x)
def k (x : ℝ) := sin (2 * x) * cos (2 * x)

-- Statement to prove
theorem even_function_with_period (x : ℝ) :
  (g(x) = g(-x)) ∧ (∀ x, g(x + π / 4) = g(x)) :=
sorry

end even_function_with_period_l361_361224


namespace find_b2023_l361_361457

def sequence (b : ℕ → ℚ) : Prop :=
  b 1 = 3 ∧
  b 2 = 4 ∧
  ∀ n, n ≥ 3 → b n = b (n - 1) / b (n - 2)

theorem find_b2023 :
  ∃ b : ℕ → ℚ, sequence b ∧ b 2023 = 1 / 4 :=
by
  have exists_sequence : ∃ b : ℕ → ℚ, sequence b := sorry
  exact exists.elim exists_sequence (λ b hb, ⟨b, hb, sorry⟩)

end find_b2023_l361_361457


namespace decreasing_interval_l361_361107

def f (x : ℝ) := x^3 - 3*x^2 + 1

theorem decreasing_interval : {x : ℝ | 0 < x ∧ x < 2} = {x : ℝ | f' x < 0} :=
  by
  sorry

end decreasing_interval_l361_361107


namespace jane_total_weekly_pages_l361_361024

def monday_pages : ℕ := 5 + 10
def tuesday_pages : ℕ := 7 + 8
def wednesday_pages : ℕ := 5 + (10 / 2)
def thursday_pages : ℕ := 7 + 8 + 15
def friday_pages : ℕ := 10 + 5
def saturday_pages : ℕ := 12 + 20
def sunday_pages : ℕ := 12

def total_weekly_pages : ℕ := monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages + saturday_pages + sunday_pages

theorem jane_total_weekly_pages : total_weekly_pages = 129 :=
by
  dsimp only [total_weekly_pages, monday_pages, tuesday_pages, wednesday_pages, thursday_pages, friday_pages, saturday_pages, sunday_pages]
  norm_num

end jane_total_weekly_pages_l361_361024


namespace distance_from_origin_to_point_l361_361814

theorem distance_from_origin_to_point : 
  ∀ (x y : ℝ), x = 8 → y = -15 → real.sqrt ((x-0)^2 + (y-0)^2) = 17 :=
by
  intros x y hx hy
  rw [hx, hy]
  norm_num
  rw [real.sqrt_eq_rpow]
  norm_num
  sorry

end distance_from_origin_to_point_l361_361814


namespace age_ratio_in_two_years_l361_361196

theorem age_ratio_in_two_years :
  ∀ (B M : ℕ), B = 10 → M = B + 12 → (M + 2) / (B + 2) = 2 := by
  intros B M hB hM
  sorry

end age_ratio_in_two_years_l361_361196


namespace prove_range_of_a_l361_361722

noncomputable def range_of_a (a : ℝ) : Prop :=
  let A := {0, 1}
  let B := {a^2, 2 * a}
  let C := {x | ∃ x1 ∈ A, ∃ x2 ∈ B, x = x1 + x2}
  ∀ x ∈ C, x ≤ 2 * a + 1

theorem prove_range_of_a (a : ℝ) : range_of_a a ↔ a ∈ Ioo (1 - Real.sqrt 2) (1 + Real.sqrt 2) :=
sorry

end prove_range_of_a_l361_361722


namespace acid_fraction_in_third_flask_correct_l361_361128

noncomputable def acid_concentration_in_third_flask 
  (w : ℝ) (W : ℝ) 
  (h1 : 10 / (10 + w) = 1 / 20)
  (h2 : 20 / (20 + (W - w)) = 7 / 30) 
  : ℝ := 
30 / (30 + W)

theorem acid_fraction_in_third_flask_correct
  (w : ℝ) (W : ℝ)
  (h1 : 10 / (10 + w) = 1 / 20)
  (h2 : 20 / (20 + (W - w)) = 7 / 30) :
  acid_concentration_in_third_flask w W h1 h2 = 21 / 200 :=
begin
  sorry
end

end acid_fraction_in_third_flask_correct_l361_361128


namespace find_complementary_angle_l361_361726

theorem find_complementary_angle (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 := 
by 
  sorry

end find_complementary_angle_l361_361726


namespace middle_of_consecutive_even_integers_l361_361116

/-- The sum of 20 consecutive even integers is 8000. Prove that the middle integer in this sequence is 399. -/
theorem middle_of_consecutive_even_integers 
  (x : ℤ)
  (h_sum : (finset.sum (finset.range 20) (λ n, x + 2 * n) = 8000)) :
  (x + 2 * (10 - 1)) = 399 :=
by sorry

end middle_of_consecutive_even_integers_l361_361116


namespace mark_18_points_l361_361018

-- Definitions for the conditions
def convex_pentagon (P : Type) : Prop := sorry -- Assuming P is a set representing the pentagon

def valid_triangles (P : Type) (points : set P) (triangles : set (set P)) : Prop :=
  sorry -- Assuming definition that establishes 10 triangles formed by vertices and diagonals.

def equal_distribution (triangles : set (set P)) (points : set P) (n : ℕ) : Prop :=
  ∀ t ∈ triangles, #({p : P | p ∈ t}) = n

-- Main statement
theorem mark_18_points (P : Type) (inside : set P) :
  convex_pentagon P → (∃ points : set P, #points = 18 ∧ valid_triangles P inside (triangles P) ∧ equal_distribution (triangles P) points 6) :=
sorry

end mark_18_points_l361_361018


namespace sum_of_cubes_l361_361019

theorem sum_of_cubes (n : ℕ) : (∑ i in finset.range (n + 1), (i + 1)^3) = (∑ i in finset.range (n + 1), (i + 1))^2 := by
  sorry

end sum_of_cubes_l361_361019


namespace distance_from_origin_to_point_l361_361831

open Real

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 →
  y = -15 →
  sqrt (x^2 + y^2) = 17 :=
by
  intros hx hy
  rw [hx, hy]
  norm_num
  rw ←sqrt_sq (show 0 ≤ 289 by norm_num)
  norm_num

end distance_from_origin_to_point_l361_361831


namespace checkerboard_sum_zero_l361_361098

theorem checkerboard_sum_zero (n : ℕ) (hn : n = 2001)
    (color : ℕ × ℕ → ℕ)
    (hcolor : ∀ i j, color (i, j) = (i + j) % 2)
    (vector_sum : ℕ × ℕ → ℕ × ℕ → ℤ × ℤ)
    (hvector_sum : ∀ i j, vector_sum (i, j) (i + 1, j) = if color(i, j) = 0 then (1, 0) else (-1, 0) ∧
                                 vector_sum (i, j) (i, j + 1) = if color(i, j) = 0 then (0, 1) else (0, -1)) :
  ∑ i j in finset.range n, (vector_sum (i, j) (i + 1, j) + vector_sum (i, j) (i, j + 1)) = (0, 0) :=
sorry

end checkerboard_sum_zero_l361_361098


namespace real_part_of_reciprocal_l361_361444

-- Definitions corresponding to the conditions in a)
variables {x y : ℝ}

-- Definition of z
noncomputable def z := x + y * complex.I

-- Condition that |z| = 1
axiom norm_z : complex.abs z = 1

-- The theorem to prove
theorem real_part_of_reciprocal (h : complex.abs z = 1) : 
  complex.re (1 / (2 - z)) = (2 - x) / (5 - 4 * x) :=
by sorry

end real_part_of_reciprocal_l361_361444


namespace digit_x_for_divisibility_by_29_l361_361097

-- Define the base 7 number 34x1_7 in decimal form
def base7_to_decimal (x : ℕ) : ℕ := 3 * 7^3 + 4 * 7^2 + x * 7 + 1

-- State the proof problem
theorem digit_x_for_divisibility_by_29 (x : ℕ) (h : base7_to_decimal x % 29 = 0) : x = 3 :=
by
  sorry

end digit_x_for_divisibility_by_29_l361_361097


namespace analogies_correct_l361_361966

theorem analogies_correct 
  (a_seq : ℕ → ℝ)
  (arithmetic_condition : (a_seq 6 + a_seq 7 + a_seq 8 + a_seq 9 + a_seq 10) / 5 = (a_seq 1 + a_seq 2 + a_seq 3 + a_seq 4 + a_seq 5 + a_seq 6 + a_seq 7 + a_seq 8 + a_seq 9 + a_seq 10 + a_seq 11 + a_seq 12 + a_seq 13 + a_seq 14 + a_seq 15) / 15)
  (b_seq : ℕ → ℝ)
  (geometric_condition : Real.geom_mean (b_seq <$> (list.range 5).map (+ 6)) = Real.geom_mean (b_seq <$> (list.range 15).map (+ 1))) :
  let tetrahedron_condition := 
    ∀ (A B C D : Point) (tri : Triangle),
    (sum_of_areas_of_three_faces tri A B C D) > area_of_fourth_face tri A B C D in
  tetrahedron_condition ∧ (arithmetic_condition = geometric_condition) :=
sorry

end analogies_correct_l361_361966


namespace k_shifted_eq1_k_shifted_eq2_k_shifted_eq3_l361_361344

-- Problem 1: Prove that 2x-3=0 is the 1-shifted equation of 2x-1=0
theorem k_shifted_eq1 : (∀ k : ℤ, 2 > 0 → k ≠ 1 → 
  let s₁ := (λ x : ℤ, (2 * x) - 3 = 0),
      s₂ := (λ x : ℤ, (2 * x) - 1 = 0)
  in (∃ x₁ x₂ : ℤ, s₁ x₁ ∧ s₂ x₂ ∧ x₁ - x₂ = k) → False) :=
sorry

-- Problem 2: Find n such that 2x+m+n=0 is the 2-shifted equation of 2x+m=0
theorem k_shifted_eq2 (m : ℤ) (n : ℤ) : 
  let s₁ := (λ x : ℤ, (2 * x) + m + n = 0),
      s₂ := (λ x : ℤ, (2 * x) + m = 0)
  in (2 > 0 ∧ ∃ x₁ x₂ : ℤ, s₁ x₁ ∧ s₂ x₂ ∧ x₁ - x₂ = 2) ↔ n = -4 :=
sorry

-- Problem 3: Prove 2b-2(c+3) for the shifted equations
theorem k_shifted_eq3 (b c: ℤ) : 
  let s₁ := (λ x : ℤ, (5 * x) + b = 1),
      s₂ := (λ x : ℤ, (5 * x) + c = 1),
      k := (2 * b) - (2 * (c + 3))
  in (5 > 0 ∧ ∃ x₁ x₂ : ℤ, s₁ x₁ ∧ s₂ x₂ ∧ x₁ - x₂ = 3) ↔ k = -36 :=
sorry

end k_shifted_eq1_k_shifted_eq2_k_shifted_eq3_l361_361344


namespace area_triangle_lines_constant_l361_361051

noncomputable theory

variables {A B C P A' B' C' A_1 B_1 C_1 : Type}
variable [euclidean_space.real]
variables (triangle : euclidean_geometry.triangle ℝ euclidean_space.real) 

-- Conditions
variables 
  (Γ : circle ℝ euclidean_space.real)
  (ABC : Π (point : euclidean_space.real), point ∈ Γ ↔ triangle.vertices point)
  (P ∈ Γ)
  (PA1 : line ℝ euclidean_space.real)
  (PB1 : line ℝ euclidean_space.real)
  (PC1 : line ℝ euclidean_space.real) 
  (Amap : midpoint ℝ euclidean_space.real (A, B)) 
  (Bmap : midpoint ℝ euclidean_space.real (B, C))
  (Cmap : midpoint ℝ euclidean_space.real (C, A))
  (A' : intersection_line_circle ℝ euclidean_space.real PA1 Γ)
  (B' : intersection_line_circle ℝ euclidean_space.real PB1 Γ)
  (C' : intersection_line_circle ℝ euclidean_space.real PC1 Γ)
  (distinct : ∀ {x : euclidean_space.real}, x ≠ x)

-- Question
theorem area_triangle_lines_constant :
  ∃ (P : euclidean_space.real) (lines : List (line ℝ euclidean_space.real)),
    IsIntersectionTriangle A B C A' B' C' lines →
    ∀ P', P' ∈ Γ →  
      area (triangle_intersections lines) = (1/2) * area ABC :=
begin
  sorry -- Proof is postponed
end

end area_triangle_lines_constant_l361_361051


namespace required_run_rate_l361_361840

/-
In the first 10 overs of a cricket game, the run rate was 3.5. 
What should be the run rate in the remaining 40 overs to reach the target of 320 runs?
-/

def run_rate_in_10_overs : ℝ := 3.5
def overs_played : ℕ := 10
def target_runs : ℕ := 320 
def remaining_overs : ℕ := 40

theorem required_run_rate : 
  (target_runs - (run_rate_in_10_overs * overs_played)) / remaining_overs = 7.125 := by 
sorry

end required_run_rate_l361_361840


namespace number_and_square_nines_l361_361081

theorem number_and_square_nines (a : ℝ) 
  (h₀ : 0 < a) 
  (h₁ : a < 1) 
  (h₂ : 0.9^(100 : ℕ) ≤ a^2) 
  (h₃ : a^2 < 1) : 
  0.9^(100 : ℕ / 2) ≤ a :=
sorry

end number_and_square_nines_l361_361081


namespace max_x_minus_2y_l361_361287

open Real

theorem max_x_minus_2y (x y : ℝ) (h : (x^2) / 16 + (y^2) / 9 = 1) : 
  ∃ t : ℝ, t = 2 * sqrt 13 ∧ x - 2 * y = t := 
sorry

end max_x_minus_2y_l361_361287


namespace no_rational_multiples_pi_tan_sum_two_l361_361269

theorem no_rational_multiples_pi_tan_sum_two (x y : ℚ) (hx : 0 < x * π ∧ x * π < y * π ∧ y * π < π / 2) (hxy : Real.tan (x * π) + Real.tan (y * π) = 2) : False :=
sorry

end no_rational_multiples_pi_tan_sum_two_l361_361269


namespace find_actual_positions_l361_361395

namespace RunningCompetition

variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]

/-- The first prediction: $A$ finishes first, $B$ is second, $C$ is third, $D$ is fourth, $E$ is fifth. --/
def first_prediction := [A, B, C, D, E]

/-- The second prediction: $C$ finishes first, $E$ is second, $A$ is third, $B$ is fourth, $D$ is fifth. --/
def second_prediction := [C, E, A, B, D]

/-- The first prediction correctly predicted the positions of exactly three athletes. --/
def first_prediction_correct_count : ℕ := 3

/-- The second prediction correctly predicted the positions of exactly two athletes. --/
def second_prediction_correct_count : ℕ := 2

/-- Proves that the actual finishing order is $C, B, A, D, E$. --/
theorem find_actual_positions : 
  ∀ (order : list (Type)), 
  (order = [C, B, A, D, E]) →
  list.count order first_prediction_correct_count = 3 ∧ 
  list.count order second_prediction_correct_count = 2 :=
by
  sorry

end RunningCompetition

end find_actual_positions_l361_361395


namespace correct_choices_l361_361086

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 4)

lemma proposition1 (x1 x2 : ℝ) (h1 : f x1 = 0) (h2 : f x2 = 0) : 
  ∃ k : ℤ, x1 - x2 = k * Real.pi := 
sorry

lemma proposition2 (x : ℝ) : 
  f x = 3 * Real.cos (2 * x - Real.pi / 4) := 
sorry

lemma proposition3 (x : ℝ) : 
  f (-Real.pi / 8) = f (Real.pi / 8) := 
sorry

lemma proposition4 (x : ℝ) : 
  ¬(f (-Real.pi / 8) = 0) := 
sorry

theorem correct_choices : 
  (proposition2 ∧ proposition3) ∧ (¬proposition1 ∧ ¬proposition4) := 
by 
  split 
  sorry

end correct_choices_l361_361086


namespace bob_password_probability_l361_361632

noncomputable def probability_bob_password : ℚ :=
  let odd_digits := {1, 3, 5, 7, 9}.size
  let total_digits := 10
  let odd_probability := odd_digits / total_digits
  
  let prime_digits := {2, 3, 5, 7}.size
  let prime_probability := prime_digits / total_digits
  
  let letter_probability : ℚ := 1

  odd_probability * letter_probability * prime_probability
   
theorem bob_password_probability :
  probability_bob_password = 1 / 5 :=
by
  sorry 

end bob_password_probability_l361_361632


namespace combined_tennis_preference_l361_361230

def students_at_north := 1800
def percentage_preferring_tennis_north := 0.30
def students_preferring_tennis_north := students_at_north * percentage_preferring_tennis_north

def students_at_south := 3000
def percentage_preferring_tennis_south := 0.35
def students_preferring_tennis_south := students_at_south * percentage_preferring_tennis_south

def total_students := students_at_north + students_at_south
def total_students_preferring_tennis := students_preferring_tennis_north + students_preferring_tennis_south

def combined_percentage_preferring_tennis := (total_students_preferring_tennis / total_students) * 100

theorem combined_tennis_preference : combined_percentage_preferring_tennis = 33 := by
    sorry

end combined_tennis_preference_l361_361230


namespace max_harmonic_mapping_M_l361_361868

noncomputable def harmonic_function (f : ℕ → ℕ) (A : Set ℕ) : Prop :=
  ∀ i : ℕ, f i ∈ A ∧ f (i + 2017) = f i

def f_iter {f : ℕ → ℕ} (k : ℕ) (x : ℕ) : ℕ
| 0     := x
| (n+1) := f (f_iter n x)

def harmonic_mapping_conditions (f : ℕ → ℕ) (M : ℕ) : Prop :=
  (∀ (m : ℕ), m < M → ∀ (i j : ℕ), i % 2017 = (j + 1) % 2017 → (f_iter m i - f_iter m j) % 2017 ≠ 1 ∧ (f_iter m i - f_iter m j) % 2017 ≠ -1)
  ∧ (∀ (i j : ℕ), i % 2017 = (j + 1) % 2017 → (f_iter M i - f_iter M j) % 2017 = 1 ∨ (f_iter M i - f_iter M j) % 2017 = -1)

theorem max_harmonic_mapping_M : 
  ∃ M : ℕ, harmonic_function f {0, 1, ..., 2016} → harmonic_mapping_conditions f M ∧ ∀ N : ℕ, harmonic_mapping_conditions f N → N ≤ 1008 :=
sorry

end max_harmonic_mapping_M_l361_361868


namespace greatest_integer_gcd_18_is_6_l361_361151

theorem greatest_integer_gcd_18_is_6 (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 18 = 6) : n = 138 := 
sorry

end greatest_integer_gcd_18_is_6_l361_361151


namespace find_actual_positions_l361_361389

namespace RunningCompetition

variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]

/-- The first prediction: $A$ finishes first, $B$ is second, $C$ is third, $D$ is fourth, $E$ is fifth. --/
def first_prediction := [A, B, C, D, E]

/-- The second prediction: $C$ finishes first, $E$ is second, $A$ is third, $B$ is fourth, $D$ is fifth. --/
def second_prediction := [C, E, A, B, D]

/-- The first prediction correctly predicted the positions of exactly three athletes. --/
def first_prediction_correct_count : ℕ := 3

/-- The second prediction correctly predicted the positions of exactly two athletes. --/
def second_prediction_correct_count : ℕ := 2

/-- Proves that the actual finishing order is $C, B, A, D, E$. --/
theorem find_actual_positions : 
  ∀ (order : list (Type)), 
  (order = [C, B, A, D, E]) →
  list.count order first_prediction_correct_count = 3 ∧ 
  list.count order second_prediction_correct_count = 2 :=
by
  sorry

end RunningCompetition

end find_actual_positions_l361_361389


namespace parallel_planes_condition_transitivity_parallel_planes_l361_361076

-- Definitions corresponding to the problem conditions.
variables {α β γ : Type*}
variables (α_plane β_plane γ_plane : set (set α))
variables (l : set α) (A B : α)

-- Necessary and sufficient condition for two planes to be parallel
theorem parallel_planes_condition :
  (∀ (l : set α), ∃ A ∈ α_plane, l ∩ α_plane = {A} → l ∩ β_plane ≠ ∅) ↔ (α_plane ∥ β_plane) :=
sorry

-- Transitivity of parallel planes
theorem transitivity_parallel_planes :
  (α_plane ∥ β_plane) → (β_plane ∥ γ_plane) → (α_plane ∥ γ_plane) :=
sorry

end parallel_planes_condition_transitivity_parallel_planes_l361_361076


namespace quadrilateral_OBEC_area_is_45_l361_361599

noncomputable def area_of_quadrilateral_OBEC : ℝ :=
  let A : point := (some (λ (p : point), 
    line_with_slope_intersects_xaxis (-3) p)) in
  let B : point := (some (λ (p : point), 
    line_with_slope_intersects_yaxis (-3) p)) in
  let C : point := (6, 0) in
  let D : point := (some (λ (p : point), 
    second_line_intersects_yaxis (6, 0) p)) in
  let E : point := (3, 3) in
  area_OBC O B C + area_BEC B E C

theorem quadrilateral_OBEC_area_is_45 (O B C E : point) : area_of_quadrilateral_OBEC O B C E = 45 := 
  by sorry

end quadrilateral_OBEC_area_is_45_l361_361599


namespace points_lie_on_line_l361_361527

/--
Let's define the given conditions and prove the points that lie on the line formed by given points (4,8) and (2,2)
are (3,5), (5,11) and (6,14).

Conditions:
- Points (4, 8) and (2, 2)
- Candidate points: (3, 5), (0, -2), (1, 1), (5, 11), (6, 14)
- Line formula: y = 3x - 4
--/

theorem points_lie_on_line :
  let P1 := (4, 8)
  let P2 := (2, 2)
  let line_eq := λ x : ℝ, 3 * x - 4
-- Candidate points
  let pointA := (3, 5)
  let pointB := (0, -2)
  let pointC := (1, 1)
  let pointD := (5, 11)
  let pointE := (6, 14)
in
-- Results
  pointA.2 = line_eq pointA.1 ∧
  pointD.2 = line_eq pointD.1 ∧
  pointE.2 = line_eq pointE.1 :=
by
  sorry

end points_lie_on_line_l361_361527


namespace parallel_line_slope_l361_361567

def line_equation (x y : ℝ) : Prop := 3 * x - 6 * y = 21

theorem parallel_line_slope : ∀ (x₁ y₁ x₂ y₂ : ℝ),
  line_equation x₁ y₁ → line_equation x₂ y₂ →
  (∃ m : ℝ, m = 1 / 2) :=
by
  intro x₁ y₁ x₂ y₂ h1 h2
  use 1 / 2
  sorry

end parallel_line_slope_l361_361567


namespace lisa_interest_correct_l361_361507

noncomputable def lisa_interest : ℝ :=
  let P := 2000
  let r := 0.035
  let n := 10
  let A := P * (1 + r) ^ n
  A - P

theorem lisa_interest_correct :
  lisa_interest = 821 := by
  sorry

end lisa_interest_correct_l361_361507


namespace inequality_proof_l361_361449

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by
  sorry

end inequality_proof_l361_361449


namespace sin_double_angle_l361_361703

open Real

theorem sin_double_angle (θ : ℝ) (h : cos (θ + π) = -1 / 3) : sin (2 * θ + π / 2) = -7 / 9 :=
by
  sorry

end sin_double_angle_l361_361703


namespace total_cost_price_is_584_l361_361085

-- Define the costs of individual items
def cost_watch : ℕ := 144
def cost_bracelet : ℕ := 250
def cost_necklace : ℕ := 190

-- The proof statement: the total cost price is 584
theorem total_cost_price_is_584 : cost_watch + cost_bracelet + cost_necklace = 584 :=
by
  -- We skip the proof steps here, assuming the above definitions are correct.
  sorry

end total_cost_price_is_584_l361_361085


namespace selling_price_of_cycle_l361_361592

theorem selling_price_of_cycle (cost_price : ℝ) (gain_percent : ℝ) (gain_decimal : ℝ) (gain : ℝ) (selling_price : ℝ) :
  cost_price = 900 ∧
  gain_percent = 27.77777777777778 ∧
  gain_decimal = gain_percent / 100 ∧
  gain = cost_price * gain_decimal ∧
  selling_price = cost_price + gain →
  selling_price = 1150 :=
by
  intro h
  obtain ⟨h_cost_price, h_gain_percent, h_gain_decimal, h_gain, h_selling_price⟩ := h
  rw [h_cost_price, h_gain_percent, h_gain_decimal, h_gain, h_selling_price]
  sorry

end selling_price_of_cycle_l361_361592


namespace total_sections_l361_361541

theorem total_sections (boys girls : ℕ) (h_boys : boys = 408) (h_girls : girls = 240) :
  let gcd_boys_girls := Nat.gcd boys girls
  let sections_boys := boys / gcd_boys_girls
  let sections_girls := girls / gcd_boys_girls
  sections_boys + sections_girls = 27 :=
by
  sorry

end total_sections_l361_361541


namespace binomial_expansion_largest_coefficient_seventh_term_l361_361413

theorem binomial_expansion_largest_coefficient_seventh_term (n : ℕ) :
  (∃ k, (x + y)^n = polynomial.monomial k x + polynomial.monomial (n - k) y + polynomial.monomial (n choose 6) x^(n-6) y^6)
  → (n = 11 ∨ n = 12 ∨ n = 13) :=
sorry

end binomial_expansion_largest_coefficient_seventh_term_l361_361413


namespace largest_integer_not_divisible_by_10_l361_361285

theorem largest_integer_not_divisible_by_10 :
  ∃ n : ℕ, 
    (∀ m : ℕ, m = n / 10 → n % 10 ≠ 0 → 
      ∀ a b c, m = a * 10^((n.length.-2) - 1) + b * 10^((n.length.-2) - 2) + c * 10^((n.length.-2) - 3) → 
      n % (a * 10 + c) = 0) ∧
    n = 9999 :=
by
  sorry

end largest_integer_not_divisible_by_10_l361_361285


namespace ann_password_prob_correct_l361_361626

def ann_password_prob : ℚ :=
  let even_digits := { 0, 2, 4, 6, 8 }
  let capital_letters := { 'A', 'B', 'C', 'D', 'E' }
  let odd_digits_gt_5 := { 7, 9 }
  (even_digits.size : ℚ) / 10 * (capital_letters.size : ℚ) / 26 * (odd_digits_gt_5.size : ℚ) / 10

theorem ann_password_prob_correct :
  ann_password_prob = 1 / 52 :=
by
  -- The proof will be filled in here.
  sorry

end ann_password_prob_correct_l361_361626


namespace find_constant_k_eq_3_l361_361436

variables {V : Type*} [inner_product_space ℝ V]

def centroid (a b c : V) : V := (a + b + c) / 3

theorem find_constant_k_eq_3 
  {A B C P : V}
  (G : V) (hG : G = centroid A B C) : 
  ∃ k : ℝ, (∥P - A∥^2 + ∥P - B∥^2 + ∥P - C∥^2) = k * ∥P - G∥^2 + (1/3) * (∥G - A∥^2 + ∥G - B∥^2 + ∥G - C∥^2) :=
begin
  use 3,
  sorry  -- Proof is omitted.
end

end find_constant_k_eq_3_l361_361436


namespace athlete_positions_l361_361385

theorem athlete_positions {A B C D E : Type} 
  (h1 : A = 1 ∧ B = 2 ∧ C = 3 ∧ D = 4 ∧ E = 5)
  (h2 : C = 1 ∧ E = 2 ∧ A = 3 ∧ B = 4 ∧ D = 5)
  (h3 : (A = 1 ∧ B = 2 ∧ C = 3 ∧ ¬(D = 4 ∨ E = 5)) ∨ 
       (A = 1 ∧ ¬(B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)) ∨ 
       (B = 2 ∧ C = 3 ∧ ¬(A = 1 ∨ D = 4 ∨ E = 5)) ∨ 
       (¬(A = 1 ∨ B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)))
  (h4 : (C = 1 ∧ ¬(A = 3 ∨ B = 4 ∨ D = 5)) ∨ 
       (A = 3 ∧ ¬(C = 1 ∨ B = 4 ∨ D = 5)) ∨ 
       (B = 4 ∧ ¬(C = 1 ∨ A = 3 ∨ D = 5)))
  : (C = 1 ∧ B = 2 ∧ A = 3 ∧ D = 4 ∧ E = 5) :=
by sorry

end athlete_positions_l361_361385


namespace exceed_2000_on_7th_day_sunday_exceeds_2000_l361_361432

def geometric_sum : ℕ → ℕ
| 0     := 0
| (n+1) := 3 + 3 * geometric_sum n

theorem exceed_2000_on_7th_day :
  geometric_sum 6 > 2000 := 
sorry  

theorem sunday_exceeds_2000 : 
  ∃ n : ℕ, n % 7 = 6 ∧ geometric_sum n > 2000 := 
by
  use 6
  exact ⟨rfl, exceed_2000_on_7th_day⟩

end exceed_2000_on_7th_day_sunday_exceeds_2000_l361_361432


namespace tan_alpha_fraction_value_l361_361730

theorem tan_alpha_fraction_value {α : Real} (h : Real.tan α = 2) : 
  (3 * Real.sin α + Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = 7 / 12 :=
by
  sorry

end tan_alpha_fraction_value_l361_361730


namespace compute_value_l361_361878

def diamond_op (x y : ℕ) : ℕ := 3 * x + 5 * y
def heart_op (z x : ℕ) : ℕ := 4 * z + 2 * x

theorem compute_value : heart_op (diamond_op 4 3) 8 = 124 := by
  sorry

end compute_value_l361_361878


namespace right_handed_players_count_l361_361181

theorem right_handed_players_count (total_players throwers left_handed_non_throwers right_handed_non_throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 46)
  (h3 : left_handed_non_throwers = (total_players - throwers) / 3)
  (h4 : right_handed_non_throwers = total_players - throwers - left_handed_non_throwers)
  (h5 : ∀ n, n = throwers + right_handed_non_throwers) : 
  (throwers + right_handed_non_throwers) = 62 := 
by 
  sorry

end right_handed_players_count_l361_361181


namespace acid_fraction_in_third_flask_correct_l361_361129

noncomputable def acid_concentration_in_third_flask 
  (w : ℝ) (W : ℝ) 
  (h1 : 10 / (10 + w) = 1 / 20)
  (h2 : 20 / (20 + (W - w)) = 7 / 30) 
  : ℝ := 
30 / (30 + W)

theorem acid_fraction_in_third_flask_correct
  (w : ℝ) (W : ℝ)
  (h1 : 10 / (10 + w) = 1 / 20)
  (h2 : 20 / (20 + (W - w)) = 7 / 30) :
  acid_concentration_in_third_flask w W h1 h2 = 21 / 200 :=
begin
  sorry
end

end acid_fraction_in_third_flask_correct_l361_361129


namespace bananas_per_friend_l361_361976

-- Define the conditions
def total_bananas : ℕ := 40
def number_of_friends : ℕ := 40

-- Define the theorem to be proved
theorem bananas_per_friend : 
  (total_bananas / number_of_friends) = 1 :=
by
  sorry

end bananas_per_friend_l361_361976


namespace christine_final_throw_difference_l361_361239

def christine_first_throw : ℕ := 20
def janice_first_throw : ℕ := christine_first_throw - 4
def christine_second_throw : ℕ := christine_first_throw + 10
def janice_second_throw : ℕ := janice_first_throw * 2
def janice_final_throw : ℕ := christine_first_throw + 17
def highest_throw : ℕ := 37

theorem christine_final_throw_difference :
  ∃ x : ℕ, christine_second_throw + x = highest_throw ∧ x = 7 := by 
sorry

end christine_final_throw_difference_l361_361239


namespace john_total_jury_duty_days_l361_361027

-- Definitions based on the given conditions
def jury_selection_days : ℕ := 2
def trial_length_multiplier : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_in_a_day : ℕ := 24

-- Define the total days John spends on jury duty
def total_days_on_jury_duty : ℕ :=
  jury_selection_days +
  (trial_length_multiplier * jury_selection_days) +
  (deliberation_days * deliberation_hours_per_day) / hours_in_a_day

-- The theorem to prove
theorem john_total_jury_duty_days : total_days_on_jury_duty = 14 := by
  sorry

end john_total_jury_duty_days_l361_361027


namespace dot_product_condition_l361_361760

variables (a b : ℝ^3)

theorem dot_product_condition (h1 : ∥a + b∥ = sqrt 9) (h2 : ∥a - b∥ = sqrt 5) :
  a • b = 1 :=
sorry

end dot_product_condition_l361_361760


namespace solve_ratios_l361_361906

theorem solve_ratios (q m n : ℕ) (h1 : 7 / 9 = n / 108) (h2 : 7 / 9 = (m + n) / 126) (h3 : 7 / 9 = (q - m) / 162) : q = 140 :=
by
  sorry

end solve_ratios_l361_361906


namespace hyperbola_focus_to_asymptote_distance_l361_361101

theorem hyperbola_focus_to_asymptote_distance (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  let c := sqrt (a^2 + b^2);
  distance (c, 0) (λ x y, b * x - a * y = 0) = b := 
by
  sorry

end hyperbola_focus_to_asymptote_distance_l361_361101


namespace starting_cell_corner_any_size_specific_field_starting_cell_general_case_any_size_differing_outcomes_same_start_l361_361148

-- a) Starting cell is in the corner, and the field is of any size
theorem starting_cell_corner_any_size (field : ℕ) (starting_cell : ℕ) 
  (h_corner : starting_cell ∈ {0, field - 1}) : 
  ∃ t:ℕ, ∀ t' ≤ t, move_by_second_player_wins := 
begin
  sorry
end

-- b) The field and starting cell are as depicted in the accompanying image
theorem specific_field_starting_cell (field : list ℕ) (starting_cell : ℕ) 
  (h_starting : starting_cell = field.head) : 
  first_player_wins := 
begin
  sorry
end

-- c) The general case: field of any size, starting cell positioned arbitrarily
theorem general_case_any_size (field : ℕ → bool) (starting_cell : ℕ) 
  (h_black : field starting_cell = tt) : 
  first_player_wins_optimal_strategy := 
begin
  sorry
end

-- d) Additional task: Examples of two games with the same initial cell and differing outcomes
theorem differing_outcomes_same_start (field : list ℕ) (starting_cell : ℕ) : 
  ∃ game1 game2, game1.start = game2.start ∧ first_player_diff_outcome game1 game2 := 
begin
  sorry
end

end starting_cell_corner_any_size_specific_field_starting_cell_general_case_any_size_differing_outcomes_same_start_l361_361148


namespace cube_surface_area_example_l361_361117

def cube_surface_area (V : ℝ) (S : ℝ) : Prop :=
  (∃ s : ℝ, s ^ 3 = V ∧ S = 6 * s ^ 2)

theorem cube_surface_area_example : cube_surface_area 8 24 :=
by
  sorry

end cube_surface_area_example_l361_361117


namespace median_of_list_is_1273_l361_361562

theorem median_of_list_is_1273 :
  let l := (List.range 1250).map (fun n => n + 1) ++ (List.range 1250).map (fun n => (n + 1) * (n + 1))
  let median := (l.sorted.nth 1249 + l.sorted.nth 1250) / 2
  median = 1273 :=
by
  let l := (List.range 1250).map (fun n => n + 1) ++ (List.range 1250).map (fun n => (n + 1) * (n + 1))
  let median := (l.sorted.nth 1249 + l.sorted.nth 1250) / 2
  sorry

end median_of_list_is_1273_l361_361562


namespace acid_concentration_in_third_flask_l361_361134

theorem acid_concentration_in_third_flask 
    (w : ℚ) (W : ℚ) (hw : 10 / (10 + w) = 1 / 20) (hW : 20 / (20 + (W - w)) = 7 / 30) 
    (W_total : W = 256.43) : 
    30 / (30 + W) = 21 / 200 := 
by 
  sorry

end acid_concentration_in_third_flask_l361_361134


namespace square_root_of_25_is_5_and_minus_5_l361_361162

theorem square_root_of_25_is_5_and_minus_5 : ∃ y : ℝ, y^2 = 25 ∧ (y = 5 ∨ y = -5) :=
by
  have h1 : 5^2 = 25 := by norm_num
  have h2 : (-5)^2 = 25 := by norm_num
  use 5
  use -5
  split
  · exact h1
  · exact h2

end square_root_of_25_is_5_and_minus_5_l361_361162


namespace velocity_for_second_ball_velocity_for_third_ball_l361_361195

-- Part (a)
theorem velocity_for_second_ball (g : ℝ) (v0 : ℝ) :
  g = 9.8 → (∀ t : ℝ, t = 0.5 → 0 = v0 - g * t) → v0 = 4.9 := by
  intros hg ht
  have h1 : g = 9.8 := hg
  have h2 : 0 = v0 - g * 0.5 := ht 0.5 rfl
  rw [h1] at h2
  linarith

-- Part (b)
theorem velocity_for_third_ball (g : ℝ) (v0 : ℝ) :
  g = 9.8 → (∀ t : ℝ, t = 1 → 0 = v0 - g * t) → v0 = 9.8 := by
  intros hg ht
  have h1 : g = 9.8 := hg
  have h2 : 0 = v0 - g * 1 := ht 1 rfl
  rw [h1] at h2
  linarith

end velocity_for_second_ball_velocity_for_third_ball_l361_361195


namespace distance_origin_to_point_l361_361794

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l361_361794


namespace acid_fraction_in_third_flask_correct_l361_361126

noncomputable def acid_concentration_in_third_flask 
  (w : ℝ) (W : ℝ) 
  (h1 : 10 / (10 + w) = 1 / 20)
  (h2 : 20 / (20 + (W - w)) = 7 / 30) 
  : ℝ := 
30 / (30 + W)

theorem acid_fraction_in_third_flask_correct
  (w : ℝ) (W : ℝ)
  (h1 : 10 / (10 + w) = 1 / 20)
  (h2 : 20 / (20 + (W - w)) = 7 / 30) :
  acid_concentration_in_third_flask w W h1 h2 = 21 / 200 :=
begin
  sorry
end

end acid_fraction_in_third_flask_correct_l361_361126


namespace orange_pear_difference_l361_361445

theorem orange_pear_difference :
  let O1 := 37
  let O2 := 10
  let O3 := 2 * O2
  let P1 := 30
  let P2 := 3 * P1
  let P3 := P2 + 4
  (O1 + O2 + O3 - (P1 + P2 + P3)) = -147 := 
by
  sorry

end orange_pear_difference_l361_361445


namespace find_constant_l361_361958

/-- Representing the conditions and the problem --/
theorem find_constant :
  ∃ (constant : ℝ),
  (∀ (x t : ℝ),
    x = 5 →
    t = 4 →
    t = constant + 0.50 * (x - 2)) ↔ constant = 2.50 :=
begin
  sorry
end

end find_constant_l361_361958


namespace q_inequality_l361_361456

theorem q_inequality
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d) :
  a^4 + b^4 + c^4 + d^4 - 4 * a * b * c * d ≥ 4 * (a - b)^2 * real.sqrt (a * b * c * d) :=
by
  sorry

end q_inequality_l361_361456


namespace expression_value_l361_361777

open Real

theorem expression_value (x : ℝ) (hx : x > 1) : x ^ (ln (ln x)) - (ln x) ^ (ln x) = 0 :=
by sorry

end expression_value_l361_361777


namespace simplify_expr_l361_361905

variable (a b : ℤ)  -- assuming a and b are elements of the ring ℤ

theorem simplify_expr : 105 * a - 38 * a + 27 * b - 12 * b = 67 * a + 15 * b := 
by
  sorry

end simplify_expr_l361_361905


namespace remainder_of_1993rd_term_divided_by_5_l361_361647

theorem remainder_of_1993rd_term_divided_by_5 :
  let sequence (n : ℕ) := if n = 1 then 1 else n + sequence (n - 1) - 1
  find_term : ℕ := 1993
  in (sequence find_term) % 5 = 3 :=
sorry

end remainder_of_1993rd_term_divided_by_5_l361_361647


namespace total_days_on_jury_duty_l361_361028

-- Definitions based on conditions
def jurySelectionDays := 2
def juryDeliberationDays := 6
def deliberationHoursPerDay := 16
def trialDurationMultiplier := 4

-- Calculate the total number of hours spent in deliberation
def totalDeliberationHours := juryDeliberationDays * 24

-- Calculate the number of days spent in deliberation based on hours per day
def deliberationDays := totalDeliberationHours / deliberationHoursPerDay

-- Calculate the trial days based on trial duration multiplier
def trialDays := jurySelectionDays * trialDurationMultiplier

-- Calculate the total days on jury duty
def totalJuryDutyDays := jurySelectionDays + trialDays + deliberationDays

theorem total_days_on_jury_duty : totalJuryDutyDays = 19 := by
  sorry

end total_days_on_jury_duty_l361_361028


namespace distance_inequality_equilateral_triangle_l361_361037

open Real

theorem distance_inequality_equilateral_triangle 
  (A B C P : Point)
  (h_equilateral : EquilateralTriangle A B C)
  (x y z : ℝ) (u v w : ℝ)
  (h_dist_vertices : dist P A = x ∧ dist P B = y ∧ dist P C = z)
  (h_dist_sides : dist P (line_through A B) = u ∧ dist P (line_through B C) = v ∧ dist P (line_through A C) = w) :
  x + y + z ≥ 2 * (u + v + w) ∧ 
  (x + y + z = 2 * (u + v + w) ↔ Centroid A B C = P) :=
sorry

end distance_inequality_equilateral_triangle_l361_361037


namespace largest_integer_satisfying_inequality_l361_361653

theorem largest_integer_satisfying_inequality :
  ∃ x : ℤ, (6 * x - 5 < 3 * x + 4) ∧ (∀ y : ℤ, (6 * y - 5 < 3 * y + 4) → y ≤ x) ∧ x = 2 :=
by
  sorry

end largest_integer_satisfying_inequality_l361_361653


namespace train_speed_l361_361218

theorem train_speed
  (length_of_train : ℕ)
  (time_to_cross_bridge : ℕ)
  (length_of_bridge : ℕ)
  (speed_conversion_factor : ℕ)
  (H1 : length_of_train = 120)
  (H2 : time_to_cross_bridge = 30)
  (H3 : length_of_bridge = 255)
  (H4 : speed_conversion_factor = 36) : 
  (length_of_train + length_of_bridge) / (time_to_cross_bridge / speed_conversion_factor) = 45 :=
by
  sorry

end train_speed_l361_361218


namespace find_biology_marks_l361_361651

variables (e m p c b : ℕ)
variable (a : ℝ)

def david_marks_in_biology : Prop :=
  e = 72 ∧
  m = 45 ∧
  p = 72 ∧
  c = 77 ∧
  a = 68.2 ∧
  (e + m + p + c + b) / 5 = a

theorem find_biology_marks (h : david_marks_in_biology e m p c b a) : b = 75 :=
sorry

end find_biology_marks_l361_361651


namespace statement4_statement6_l361_361757

variables (α β γ : Plane)
variables (a : Line α) (b : Line β) (c : Line γ)
variable (l : Line)

-- Conditions
axiom plane1 : β ⊥ γ
axiom plane2 : ∃ l, l ∈ α ∧ l ∈ γ ∧ ¬(α ⊥ γ)

-- Statements to be proven
theorem statement4 : ∃ a : Line α, parallel a γ := 
sorry

theorem statement6 : ∃ c : Line γ, perp c β := 
sorry

end statement4_statement6_l361_361757


namespace measure_angle_BCD_is_90_degrees_l361_361405

theorem measure_angle_BCD_is_90_degrees
  (AB BC CD DA : ℝ)
  (h1 : AB = BC)
  (h2 : BC = CD)
  (h3 : CD = DA)
  (h4 : AB = DA)
  (angle_ABC_is_90 : ∠ ABC = 90) :
  ∠ BCD = 90 :=
sorry

end measure_angle_BCD_is_90_degrees_l361_361405


namespace square_binomial_l361_361159

theorem square_binomial (x : ℝ) : (-x - 1) ^ 2 = x^2 + 2 * x + 1 :=
by
  sorry

end square_binomial_l361_361159


namespace non_empty_subsets_count_l361_361330

def is_odd_or_divisible_by_3 (n : ℕ) : Prop :=
  n % 2 = 1 ∨ n % 3 = 0

def special_subset : Finset ℕ :=
  ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ).filter is_odd_or_divisible_by_3

theorem non_empty_subsets_count : Finset.card special_subset = 6 → 2 ^ 6 - 1 = 63 :=
by
  intro h_card
  rw h_card
  norm_num
  sorry

end non_empty_subsets_count_l361_361330


namespace square_root_of_25_is_5_and_minus_5_l361_361163

theorem square_root_of_25_is_5_and_minus_5 : ∃ y : ℝ, y^2 = 25 ∧ (y = 5 ∨ y = -5) :=
by
  have h1 : 5^2 = 25 := by norm_num
  have h2 : (-5)^2 = 25 := by norm_num
  use 5
  use -5
  split
  · exact h1
  · exact h2

end square_root_of_25_is_5_and_minus_5_l361_361163


namespace area_of_given_triangle_l361_361275

open Real

noncomputable def area_of_triangle : ℝ :=
  let u : ℝ × ℝ × ℝ := (6, 5, 3)
  let v : ℝ × ℝ × ℝ := (3, 3, 1)
  let w : ℝ × ℝ × ℝ := (7, 8, 5)
  let vu : ℝ × ℝ × ℝ := (v.1 - u.1, v.2 - u.2, v.3 - u.3)
  let wu : ℝ × ℝ × ℝ := (w.1 - u.1, w.2 - u.2, w.3 - u.3)
  let cross_product : ℝ × ℝ × ℝ :=
    (vu.2 * wu.3 - vu.3 * wu.2, vu.3 * wu.1 - vu.1 * wu.3, vu.1 * wu.2 - vu.2 * wu.1)
  let magnitude : ℝ := sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2)
  (1 / 2) * magnitude

theorem area_of_given_triangle : area_of_triangle = sqrt 62 / 2 :=
by
  sorry

end area_of_given_triangle_l361_361275


namespace num_possible_configurations_l361_361404

-- Define the conditions and the problem in Lean 4

def knight_or_liar_problem : Prop :=
  let n := 8;
  let knave (grid : ℕ → ℕ → bool) :=
    ∀ i j, grid i j →
    (∑ i', if grid i' j then 1 else 0) > (∑ j', if grid i j' then 1 else 0);
  let liar (grid : ℕ → ℕ → bool) :=
    ∀ i j, ¬grid i j →
    ¬((∑ i', if grid i' j then 1 else 0) > (∑ j', if grid i j' then 1 else 0));
  ∃ grid : ℕ → ℕ → bool, knave grid ∧ liar grid ∧ (∑ i j, if grid i j then 1 else 0) = 255

theorem num_possible_configurations : knight_or_liar_problem :=
sorry

end num_possible_configurations_l361_361404


namespace gym_cost_l361_361424

theorem gym_cost (x : ℕ) (hx : x > 0) (h1 : 50 + 12 * x + 48 * x = 650) : x = 10 :=
by
  sorry

end gym_cost_l361_361424


namespace perimeter_less_than_1_km_perimeter_less_than_1_km_zero_thickness_perimeter_to_area_ratio_l361_361179

section Problem

-- Definitions based on the problem conditions

-- Condition: Side length of each square is 1 cm
def side_length : ℝ := 1

-- Condition: Thickness of the nail for parts a) and b)
def nail_thickness_a := 0.1
def nail_thickness_b := 0

-- Given a perimeter P and area S, the perimeter cannot exceed certain thresholds based on problem analysis

theorem perimeter_less_than_1_km (P : ℝ) (S : ℝ) (r : ℝ) (h1 : 2 * S ≥ r * P) (h2 : r = 0.1) : P < 1000 * 100 :=
  sorry

theorem perimeter_less_than_1_km_zero_thickness (P : ℝ) (S : ℝ) (r : ℝ) (h1 : 2 * S ≥ r * P) (h2 : r = 0) : P < 1000 * 100 :=
  sorry

theorem perimeter_to_area_ratio (P : ℝ) (S : ℝ) (h : P / S ≤ 700) : P / S < 100000 :=
  sorry

end Problem

end perimeter_less_than_1_km_perimeter_less_than_1_km_zero_thickness_perimeter_to_area_ratio_l361_361179


namespace max_f_value_l361_361685

def f : ℕ → ℕ 
| 0     := 0
| 1     := 1
| n     := f n.div2 + n - 2 * n.div2

theorem max_f_value : (∀ n, 0 ≤ n ∧ n ≤ 1997 → f n ≤ 10) ∧ (∃ n, 0 ≤ n ∧ n ≤ 1997 ∧ f n = 10) :=
by
  sorry

end max_f_value_l361_361685


namespace determinant_expression_l361_361863

-- Assuming a, b, c are the roots of the polynomial x^3 + px^2 + qx + r = 0
-- and expressing the given determinant in terms of p, q, and r.
theorem determinant_expression (a b c p q r : ℝ)
  (h : ∀ x : ℝ, x^3 + p * x^2 + q * x + r = 0 → x = a ∨ x = b ∨ x = c) :
  (∣ 1 + a^2  1       1     ∣
   ∣ 1        1 + b^2 1     ∣
   ∣ 1        1       1 + c^2 ∣) =
  r^2 + q^2 + p^2 - 2q :=
sorry

end determinant_expression_l361_361863


namespace sample_capacity_l361_361786

theorem sample_capacity (n : ℕ) (A B C : ℕ) (h_ratio : A / (A + B + C) = 3 / 14) (h_A : A = 15) : n = 70 :=
by
  sorry

end sample_capacity_l361_361786


namespace find_c_plus_one_div_b_l361_361913

-- Assume that a, b, and c are positive real numbers such that the given conditions hold.
variables (a b c : ℝ)
variables (habc : a * b * c = 1)
variables (hac : a + 1 / c = 7)
variables (hba : b + 1 / a = 11)

-- The goal is to show that c + 1 / b = 5 / 19.
theorem find_c_plus_one_div_b : c + 1 / b = 5 / 19 :=
by 
  sorry

end find_c_plus_one_div_b_l361_361913


namespace factorial_ratio_integer_l361_361057

theorem factorial_ratio_integer 
  (m n : ℕ) 
  (h : 0! = 1) : 
  ∃ k : ℤ, k = ((2 * m)! * (2 * n)!) / (m! * n! * (m + n)!) := 
sorry

end factorial_ratio_integer_l361_361057


namespace problem1_condition1_problem1_condition2_problem2_l361_361238

-- Problem 1: Proving that C = π / 3 under given conditions

theorem problem1_condition1 (a b c A C : ℝ) : 
  (b - c * cos A = a * (sqrt 3 * sin C - 1)) → 
  C = π / 3 :=
by 
  sorry

theorem problem1_condition2 (A B C : ℝ) :
  (sin (A + B) * cos (C - π / 6) = 3 / 4) → 
  C = π / 3 :=
by 
  sorry

-- Problem 2: Proving the maximum value of CD^2 / (a^2 + b^2)

theorem problem2 (a b : ℝ) :
  let CD := (b^2 + a^2 + a * b) / (4 * (a^2 + b^2))
  C = π / 3 → 
  CD ≤ 3 / 8 :=
by 
  sorry

end problem1_condition1_problem1_condition2_problem2_l361_361238


namespace range_of_a_l361_361174

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) → (-1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l361_361174


namespace hyperbola_eccentricity_l361_361676

theorem hyperbola_eccentricity :
  let e := Real.sqrt 2 in
  (∀ x y : ℝ, x^2 - y^2 = -2) → e = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l361_361676


namespace max_segments_no_triangle_l361_361305

theorem max_segments_no_triangle (length_wire : ℤ) (n : ℤ) (segment_lengths : List ℤ) 
  (h_length_wire : length_wire = 150)
  (h_n_pos : n > 2)
  (h_segment_lengths : ∀ l ∈ segment_lengths, l ≥ 1)
  (h_sum_segments : segment_lengths.sum = length_wire)
  (h_no_triangle : ∀ a b c, List.PerM (a::b::c::segment_lengths.filter (λ x, x ≠ a ∧ x ≠ b ∧ x ≠ c)) segment_lengths → a + b ≤ c) :
  n ≤ 10 := 
sorry

end max_segments_no_triangle_l361_361305


namespace rectangle_area_perimeter_l361_361209

-- Defining the problem conditions
def positive_int (n : Int) : Prop := n > 0

-- The main statement of the problem
theorem rectangle_area_perimeter (a b : Int) (h1 : positive_int a) (h2 : positive_int b) : 
  ¬ (a + 2) * (b + 2) - 4 = 146 :=
by
  sorry

end rectangle_area_perimeter_l361_361209


namespace minute_hand_rotation_l361_361222

theorem minute_hand_rotation :
  (10 / 60) * (2 * Real.pi) = (- Real.pi / 3) :=
by
  sorry

end minute_hand_rotation_l361_361222


namespace max_pentagon_area_is_1_l361_361435

-- Definition of the problem conditions
def isSquare (A B C D : Type) (side_length : ℝ) : Prop :=
  ∃ (a b c d : ℝ),
    side_length = abs (a - b) ∧ 
    side_length = abs (b - c) ∧
    side_length = abs (c - d) ∧ 
    side_length = abs (d - a)

def pointOnSide (E : Type) (A B : Type) (x : ℝ) : Prop := 
  ∃ (ae : ℝ), ae = x

def maxPentagonArea (side_length : ℝ) (x : ℝ) : ℝ :=
  let area_triangle_CDF := (2 - x) in
  let area_triangle_FEG := (1/2) * (2 - x) ^ 2 in
  area_triangle_CDF + area_triangle_FEG

-- Lean statement problem
theorem max_pentagon_area_is_1
  (A B C D E F G : Type)
  (side_length : ℝ) 
  (x : ℝ)
  (h_square : isSquare A B C D 2)
  (h_E_on_AB : pointOnSide E A B x)
  (h_F_on_AD : pointOnSide F A D x)
  (h_G_on_BC : pointOnSide G B C x)
  (hx_lt_1 : x < 1) :
  ∃ x, x = 1 ∧ maxPentagonArea side_length x = 1.5 := 
sorry

end max_pentagon_area_is_1_l361_361435


namespace employed_females_percentage_l361_361013

theorem employed_females_percentage (total_population : ℕ) :
  let,
  initial_employment_rate := 0.64,
  employed_males := 0.55,
  growth_rate := 0.02,
  years := 5,
  new_category_percentage := 0.10,
  employed_females := 1 - employed_males,
  employment_rate_fifth_year := initial_employment_rate + (growth_rate * years),
  unanswered := 1 - new_category_percentage,
  female_ratio := employed_females * (employment_rate_fifth_year / unanswered) in
  female_ratio = 0.45 ∧
  majority_category_females = "Education" :=
by
  sorry

end employed_females_percentage_l361_361013


namespace integral_ln_80_23_l361_361243

theorem integral_ln_80_23 :
  ∫ x in 1..2, (9 * x + 4) / (x ^ 5 + 3 * x ^ 2 + x) = Real.log (80 / 23) :=
by
  sorry

end integral_ln_80_23_l361_361243


namespace seagulls_left_l361_361963

theorem seagulls_left (initial_seagulls : ℕ) 
                        (scared_fraction : ℚ) 
                        (fly_fraction : ℚ)
                        (h1 : initial_seagulls = 36)
                        (h2 : scared_fraction = 1 / 4)
                        (h3 : fly_fraction = 1 / 3) : 
  (initial_seagulls - (scared_fraction * initial_seagulls).to_int 
  - (fly_fraction * (initial_seagulls - (scared_fraction * initial_seagulls).to_int)).to_int) = 18 := 
by
  sorry

end seagulls_left_l361_361963


namespace a_seq_formula_b_seq_formula_T_n_formula_l361_361307

-- Defining the arithmetic sequence with given conditions
def a_seq (n : ℕ) : ℕ := 3 * n - 3

-- Defining the geometric sequence with given conditions
def b_seq (n : ℕ) : ℕ := 3 * 2^(n-1)

-- Defining the sequence c_n
def c_seq (n : ℕ) : ℕ :=
  (a_seq n * b_seq n) / (a_seq (n+1) * a_seq (n+2))

-- Defining the sum of the first n terms of c_seq
def T (n : ℕ) : ℕ :=
  ∑ k in finset.range n, c_seq (k + 1)

theorem a_seq_formula : ∀ n, a_seq n = 3 * n - 3 := sorry
theorem b_seq_formula : ∀ n, b_seq n = 3 * 2^(n-1) := sorry
theorem T_n_formula : ∀ n, T n = 2^n / (n+1) - 1 := sorry

end a_seq_formula_b_seq_formula_T_n_formula_l361_361307


namespace race_head_start_l361_361176

-- This statement defines the problem in Lean 4
theorem race_head_start (Va Vb L H : ℝ) 
(h₀ : Va = 51 / 44 * Vb) 
(h₁ : L / Va = (L - H) / Vb) : 
H = 7 / 51 * L := 
sorry

end race_head_start_l361_361176


namespace inequality_real_numbers_l361_361688

theorem inequality_real_numbers
  (n : ℕ)
  (x y : Fin n → ℝ)
  (hx : ∀ i j, i < j → x i ≥ x j ∧ x i > 0)
  (hy : ∀ k, (∏ i in Finset.range (k + 1), y i) ≥ (∏ i in Finset.range (k + 1), x i)) :
  n * y 0 + (n - 1) * y 1 + ∑ i in Finset.range (n - 2 + 1), (n - 1 - i) * y (i + 2) ≥
  x 0 + 2 * x 1 + ∑ i in Finset.range (n - 2 + 1), (i + 3) * x (i + 2) :=
sorry

end inequality_real_numbers_l361_361688


namespace compute_3_pow_hypotenuse_l361_361108

def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3
def log6 (x : ℝ) : ℝ := Real.log x / Real.log 6

theorem compute_3_pow_hypotenuse :
  let a := log3 64
  let b := log6 32
  let h := Real.sqrt (a^2 + b^2)
  3^h = 243 :=
by
  let a := log3 64
  let b := log6 32
  let h := Real.sqrt (a^2 + b^2)
  sorry

end compute_3_pow_hypotenuse_l361_361108


namespace find_complementary_angle_l361_361725

theorem find_complementary_angle (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 := 
by 
  sorry

end find_complementary_angle_l361_361725


namespace unique_toothpicks_in_grid_15_by_8_l361_361211

theorem unique_toothpicks_in_grid_15_by_8 :
  let height := 15
  let width := 8
  let horizontal_toothpicks := (height + 1) * width
  let vertical_toothpicks := (width + 1) * height
  let intersections := (height + 1) * (width + 1)
  in horizontal_toothpicks + vertical_toothpicks - intersections = 119 :=
by
  sorry

end unique_toothpicks_in_grid_15_by_8_l361_361211


namespace symmetric_set_with_point_card_l361_361212

theorem symmetric_set_with_point_card {T : set (ℝ × ℝ)} 
  (h₀ : ∀ (x y : ℝ), (x, y) ∈ T → (-x, -y) ∈ T)
  (hx : ∀ (x y : ℝ), (x, y) ∈ T → (x, -y) ∈ T)
  (hy : ∀ (x y : ℝ), (x, y) ∈ T → (-x, y) ∈ T)
  (hxy : ∀ (x y : ℝ), (x, y) ∈ T → (y, x) ∈ T)
  (hnegxy : ∀ (x y : ℝ), (x, y) ∈ T → (-y, -x) ∈ T)
  (h : (3, 4) ∈ T) :
  ∃ (S : set (ℝ × ℝ)), S ⊆ T ∧ S.card = 8 := sorry

end symmetric_set_with_point_card_l361_361212


namespace find_actual_positions_l361_361394

namespace RunningCompetition

variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]

/-- The first prediction: $A$ finishes first, $B$ is second, $C$ is third, $D$ is fourth, $E$ is fifth. --/
def first_prediction := [A, B, C, D, E]

/-- The second prediction: $C$ finishes first, $E$ is second, $A$ is third, $B$ is fourth, $D$ is fifth. --/
def second_prediction := [C, E, A, B, D]

/-- The first prediction correctly predicted the positions of exactly three athletes. --/
def first_prediction_correct_count : ℕ := 3

/-- The second prediction correctly predicted the positions of exactly two athletes. --/
def second_prediction_correct_count : ℕ := 2

/-- Proves that the actual finishing order is $C, B, A, D, E$. --/
theorem find_actual_positions : 
  ∀ (order : list (Type)), 
  (order = [C, B, A, D, E]) →
  list.count order first_prediction_correct_count = 3 ∧ 
  list.count order second_prediction_correct_count = 2 :=
by
  sorry

end RunningCompetition

end find_actual_positions_l361_361394


namespace min_large_flasks_proba_lt_half_l361_361788

theorem min_large_flasks_proba_lt_half : ∃ (N : ℕ), (3 ≤ N ∧ N ≤ 97) → (let n := 100 - N in
    (0.8 * V + 0.5 * V + 0.2 * V / 2) / 3 * 100 = 50 ∧
    (0.8 * V / 2 + 0.5 * V / 2 + 0.2 * V / 2) / 3 * 100 = 50 ∧
    (0.8 * V + 0.5 * V / 2 + 0.2 * V) / 2.5 * 100 = 50 ∧
    (0.8 * V / 2 + 0.5 * V + 0.2 * V / 2) / 2 * 100 = 50) →
    ( (N * (N - 1) * (N - 2) + n * (n - 1) * (n - 2) + N * n * (N - 1) +  n * N * (n - 1)) / 
      (100 * 99 * 98) < 1 / 2 ) 

end min_large_flasks_proba_lt_half_l361_361788


namespace number_difference_l361_361534

theorem number_difference (x y : ℕ) (h₁ : x + y = 41402) (h₂ : ∃ k : ℕ, x = 100 * k) (h₃ : y = x / 100) : x - y = 40590 :=
sorry

end number_difference_l361_361534


namespace albums_not_in_both_l361_361624

-- Definitions representing the problem conditions
def andrew_albums : ℕ := 23
def common_albums : ℕ := 11
def john_unique_albums : ℕ := 8

-- Proof statement (not the actual proof)
theorem albums_not_in_both : 
  (andrew_albums - common_albums) + john_unique_albums = 20 :=
by
  sorry

end albums_not_in_both_l361_361624


namespace complement_union_eq_l361_361325

-- Define the universal set U, set A, and set B
def U := {1, 2, 3, 4, 5}
def A := {1, 3, 4}
def B := {2, 4}

-- Define the complement function
def complement (U A : Set Nat) : Set Nat :=
  { x | x ∈ U ∧ x ∉ A }

-- Define the union of the complement of A and B
def result := (complement U A) ∪ B

-- State the proof
theorem complement_union_eq :
  result = {2, 4, 5} :=
by
  sorry

end complement_union_eq_l361_361325


namespace percentage_of_students_with_puppies_and_parrots_l361_361349

theorem percentage_of_students_with_puppies_and_parrots
  (total_students : ℕ)
  (puppy_percentage : ℕ → ℚ)
  (students_with_puppies : ℕ)
  (students_with_puppies_and_parrots : ℕ) :
  total_students = 40 →
  puppy_percentage 80 = 0.80 →
  students_with_puppies = total_students * puppy_percentage 80 → 
  students_with_puppies_and_parrots = 8 →
  100 * (students_with_puppies_and_parrots / students_with_puppies) = 25 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  have puppy_students : ℕ := 32
  have both_pets_students : ℕ := 8
  have percentage : ℚ := 100 * (both_pets_students / puppy_students)
  have target : ℚ := 25
  exact sorry

end percentage_of_students_with_puppies_and_parrots_l361_361349


namespace chips_count_l361_361787

theorem chips_count (B G P R x : ℕ) 
  (hx1 : 5 < x) (hx2 : x < 11) 
  (h : 1^B * 5^G * x^P * 11^R = 28160) : 
  P = 2 :=
by 
  -- Hint: Prime factorize 28160 to apply constraints and identify corresponding exponents.
  have prime_factorization_28160 : 28160 = 2^6 * 5^1 * 7^2 := by sorry
  -- Given 5 < x < 11 and by prime factorization, x can only be 7 (since it factors into the count of 7)
  -- Complete the rest of the proof
  sorry

end chips_count_l361_361787


namespace numerology_eq_l361_361559

theorem numerology_eq : 2222 - 222 + 22 - 2 = 2020 :=
by
  sorry

end numerology_eq_l361_361559


namespace correctFinishingOrder_l361_361397

-- Define the types of athletes
inductive Athlete
| A | B | C | D | E

-- Define the type for predictions
def Prediction := Athlete → Nat

-- Define the first prediction
def firstPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 1
    | Athlete.B => 2
    | Athlete.C => 3
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the second prediction
def secondPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 4
    | Athlete.C => 1
    | Athlete.D => 5
    | Athlete.E => 2

-- Define the actual finishing positions
noncomputable def actualFinishingPositions : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 2
    | Athlete.C => 1
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the correctness conditions for the predictions
def correctPositions (actual : Prediction) (prediction : Prediction) (k : Nat) : Prop :=
  (Set.card {x : Athlete | actual x = prediction x} = k)

-- The proof statement
theorem correctFinishingOrder :
  correctPositions actualFinishingPositions firstPrediction 3 ∧
  correctPositions actualFinishingPositions secondPrediction 2 :=
sorry

end correctFinishingOrder_l361_361397


namespace end_of_workday_l361_361880

def start_time : Nat := 7 * 60 + 45 -- 7:45 A.M. in minutes
def lunch_start_time : Nat := 12 * 60 -- 12:00 P.M. in minutes
def lunch_duration : Nat := 1 * 60 + 15 -- 1 hour and 15 minutes in minutes
def work_duration : Nat := 9 * 60 -- 9 hours in minutes

theorem end_of_workday :
  let work_before_lunch := lunch_start_time - start_time,
      remaining_work := work_duration - work_before_lunch,
      resume_time := lunch_start_time + lunch_duration,
      end_time := resume_time + remaining_work
  in end_time = 18 * 60 := -- 6:00 P.M. in minutes
by
  sorry

end end_of_workday_l361_361880


namespace find_actual_positions_l361_361392

namespace RunningCompetition

variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]

/-- The first prediction: $A$ finishes first, $B$ is second, $C$ is third, $D$ is fourth, $E$ is fifth. --/
def first_prediction := [A, B, C, D, E]

/-- The second prediction: $C$ finishes first, $E$ is second, $A$ is third, $B$ is fourth, $D$ is fifth. --/
def second_prediction := [C, E, A, B, D]

/-- The first prediction correctly predicted the positions of exactly three athletes. --/
def first_prediction_correct_count : ℕ := 3

/-- The second prediction correctly predicted the positions of exactly two athletes. --/
def second_prediction_correct_count : ℕ := 2

/-- Proves that the actual finishing order is $C, B, A, D, E$. --/
theorem find_actual_positions : 
  ∀ (order : list (Type)), 
  (order = [C, B, A, D, E]) →
  list.count order first_prediction_correct_count = 3 ∧ 
  list.count order second_prediction_correct_count = 2 :=
by
  sorry

end RunningCompetition

end find_actual_positions_l361_361392


namespace words_to_learn_l361_361684

theorem words_to_learn (V G T : ℝ) (hV : V = 600) (hG : G = 0.05) (hT : T = 0.9) :
  ∃ (x : ℝ), x ≥ 537 ∧ (x + G * (V - x)) / V = T :=
by
  use 537
  split
  · simp
  · sorry

end words_to_learn_l361_361684


namespace common_chord_bisects_CH_l361_361071

theorem common_chord_bisects_CH
  (S : Circle) (A B : Point) (S_diameter : diameter S = A.distance_to B)
  (C : Point) (C_on_S : on_circle C S)
  (H : Point) (CH_perpendicular : perpendicular_line_from_point H C (line_through A B))
  (S1 : Circle) (S1_centered_at_C : center S1 = C) (CH_radius_of_S1 : radius S1 = C.distance_to H):
  ∃ M : Point, midpoint M C H ∧ lies_on_radical_axis M S S1 :=
sorry

end common_chord_bisects_CH_l361_361071


namespace concentration_flask3_l361_361133

open Real

noncomputable def proof_problem : Prop :=
  ∃ (W : ℝ) (w : ℝ), 
    let flask1_acid := 10 in
    let flask2_acid := 20 in
    let flask3_acid := 30 in
    let flask1_total := flask1_acid + w in
    let flask2_total := flask2_acid + (W - w) in
    let flask3_total := flask3_acid + W in
    (flask1_acid / flask1_total = 1 / 20) ∧
    (flask2_acid / flask2_total = 7 / 30) ∧
    (flask3_acid / flask3_total = 21 / 200)

theorem concentration_flask3 : proof_problem :=
begin
  sorry
end

end concentration_flask3_l361_361133


namespace least_possible_coins_l361_361568

theorem least_possible_coins : 
  ∃ b : ℕ, b % 7 = 3 ∧ b % 4 = 2 ∧ ∀ n : ℕ, (n % 7 = 3 ∧ n % 4 = 2) → b ≤ n :=
sorry

end least_possible_coins_l361_361568


namespace correct_prediction_l361_361372

-- Declare the permutation type
def Permutation := (ℕ × ℕ × ℕ × ℕ × ℕ)

-- Declare the actual finish positions
def actual_positions : Permutation := (3, 2, 1, 4, 5)

-- Declare conditions
theorem correct_prediction 
  (P1 : Permutation) (P2 : Permutation) 
  (hP1 : P1 = (1, 2, 3, 4, 5))
  (hP2 : P2 = (1, 2, 3, 4, 5))
  (hP1_correct : ∃! i ∈ (1..5), (λ i, nth (P1.to_list) (i - 1) == nth (actual_positions.to_list))  = 3)
  (hP2_correct : ∃! i ∈ (1..5), (λ i, nth (P2.to_list) (i - 1) == nth (actual_positions.to_list))  = 2)
  : actual_positions = (3, 2, 1, 4, 5) 
:= sorry

end correct_prediction_l361_361372


namespace output_correct_l361_361575

-- Definitions derived from the conditions
def initial_a : Nat := 3
def initial_b : Nat := 4

-- Proof that the final output of PRINT a, b is (4, 4)
theorem output_correct : 
  let a := initial_a;
  let b := initial_b;
  let a := b;
  let b := a;
  (a, b) = (4, 4) :=
by
  sorry

end output_correct_l361_361575


namespace Heath_current_age_l361_361348

variable (H J : ℕ) -- Declare variables for Heath's and Jude's ages
variable (h1 : J = 2) -- Jude's current age is 2
variable (h2 : H + 5 = 3 * (J + 5)) -- In 5 years, Heath will be 3 times as old as Jude

theorem Heath_current_age : H = 16 :=
by
  -- Proof to be filled in later
  sorry

end Heath_current_age_l361_361348


namespace number_of_10_and_6_divisors_of_20_factorial_l361_361672

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define a function that calculates the highest power of a prime p dividing n!
def highest_power_dividing_factorial (n p : ℕ) : ℕ :=
  if p < 2 then 0 else
    let rec count (n acc : ℕ) :=
      if n = 0 then acc
      else count (n / p) (acc + n / p)
    in count n 0

-- Define the problem
theorem number_of_10_and_6_divisors_of_20_factorial : 
  highest_power_dividing_factorial 20 10 + highest_power_dividing_factorial 20 6 = 12 :=
by sorry

end number_of_10_and_6_divisors_of_20_factorial_l361_361672


namespace trapezoid_angle_ADC_l361_361412

theorem trapezoid_angle_ADC (
  A B C D E F : Type -- Points in trapezoid
  (ABCD_trapezoid : ∀ (P : Type), P = BC ∨ P = AD)
  (BC_parallel_AD : Parallel BC AD)
  (AB_eq_CD : Distance AB = Distance CD)
  (E_on_AD : OnLine E AD)
  (BE_perp_AD : Perp BE AD)
  (F_intersection : PointOfIntersection AC BE F)
  (AF_eq_FB : Distance AF = Distance FB)
  (angle_AFE_50 : ∠AFE = 50)
) : ∠ADC = 65 :=
sorry

end trapezoid_angle_ADC_l361_361412


namespace distance_between_vertices_of_hyperbola_l361_361276

theorem distance_between_vertices_of_hyperbola :
  (∃ a b : ℝ, a^2 = 144 ∧ b^2 = 64 ∧ 
    ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → true) → 
  (2 * real.sqrt 144 = 24) :=
by
  sorry

end distance_between_vertices_of_hyperbola_l361_361276


namespace no_possible_k_of_prime_roots_l361_361636

theorem no_possible_k_of_prime_roots :
  ∀ k : ℤ, (∃ p q : ℤ, p.prime ∧ q.prime ∧ (x^2 - 65 * x + k = 0) = (x - p) * (x - q)) → false :=
by
  sorry

end no_possible_k_of_prime_roots_l361_361636


namespace three_friends_collected_at_least_50_l361_361901

theorem three_friends_collected_at_least_50 
  (a : Fin 7 → ℕ) 
  (distinct : Function.Injective a) 
  (sum_eq : ∑ i, a i = 100) : 
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i + a j + a k ≥ 50 := 
by 
  sorry

end three_friends_collected_at_least_50_l361_361901


namespace max_cars_in_parking_lot_l361_361578

theorem max_cars_in_parking_lot : 
  ∀ (lot : Fin 7 × Fin 7 → Prop) (gate : (Fin 7 × Fin 7)),
    (∀ c, lot c → c ≠ gate) →
    (∀ car, lot car → ∃ path : List (Fin 7 × Fin 7),
      (∀ p ∈ path, lot p ∨ p = gate) ∧ 
      path.head = car ∧ 
      path.getLast sorry = gate ∧ 
      ∀ (i < path.length - 1), 
        let (x1, y1) := path.nth i in 
        let (x2, y2) := path.nth (i + 1) in 
        (x1 - x2).abs + (y1 - y2).abs = 1) →
    (∃ cars, cars.card = 28 ∧ ∀ car ∈ cars, lot car) := sorry

end max_cars_in_parking_lot_l361_361578


namespace triangle_is_obtuse_l361_361918

theorem triangle_is_obtuse (A : ℝ) (hA1 : 0 < A ) (hA2 : A < real.pi)
    (h : real.sin A + real.cos A = 7 / 12) : 
    real.cos A < 0 :=
by
  sorry

end triangle_is_obtuse_l361_361918


namespace route_difference_l361_361881

noncomputable def time_route_A (distance_A : ℝ) (speed_A : ℝ) : ℝ :=
  (distance_A / speed_A) * 60

noncomputable def time_route_B (distance1_B distance2_B distance3_B : ℝ) (speed1_B speed2_B speed3_B : ℝ) : ℝ :=
  ((distance1_B / speed1_B) * 60) + 
  ((distance2_B / speed2_B) * 60) + 
  ((distance3_B / speed3_B) * 60)

theorem route_difference
  (distance_A : ℝ := 8)
  (speed_A : ℝ := 25)
  (distance1_B : ℝ := 2)
  (distance2_B : ℝ := 0.5)
  (speed1_B : ℝ := 50)
  (speed2_B : ℝ := 20)
  (distance_total_B : ℝ := 7)
  (speed3_B : ℝ := 35) :
  time_route_A distance_A speed_A - time_route_B distance1_B distance2_B (distance_total_B - distance1_B - distance2_B) speed1_B speed2_B speed3_B = 7.586 :=
by
  sorry

end route_difference_l361_361881


namespace quadratic_functions_count_even_quadratic_functions_count_l361_361493

-- Definitions
def isNonZero (a : Int) : Prop := a ≠ 0
def isDistinct (a b c : Int) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def isEvenFunctionCoefficient (b : Int) : Prop := b = 0

-- Problem Assertions
theorem quadratic_functions_count :
  ∃ (a b c : Int), isNonZero a ∧ isDistinct a b c ∧ (a ∈ [-1, 1, 2]) ∧ (b ∈ [-1, 0, 1]) ∧
  (c ∈ ([-1, 0, 1, 2] \ {a, b})) ∧ ∃ f_set, f_set.card = 18 :=
sorry

theorem even_quadratic_functions_count :
  ∃ (a b c : Int), isNonZero a ∧ isDistinct a b c ∧ isEvenFunctionCoefficient b ∧
  (a ∈ [-1, 1, 2]) ∧ (c ∈ ([-1, 0, 1, 2] \ {a, b})) ∧ ∃ f_set_even, f_set_even.card = 6 :=
sorry

end quadratic_functions_count_even_quadratic_functions_count_l361_361493


namespace complement_A_B_l361_361294

noncomputable def A : set ℝ := {y | ∃ x, y = |x + 1| ∧ x ∈ Icc (-2 : ℝ) 4}
noncomputable def B : set ℝ := Ico (2 : ℝ) 5

theorem complement_A_B : A \ B = Ico 0 (2 : ℝ) ∪ {5} :=
  sorry

end complement_A_B_l361_361294


namespace product_remainder_l361_361340

theorem product_remainder (m n : ℕ) :
  let a := 3 * m + 1
  let b := 3 * n + 2
  (a * b) % 3 = 2 :=
by
  let a := 3 * m + 1
  let b := 3 * n + 2
  have hab := (a * b) % 3
  show hab = 2
  sorry

end product_remainder_l361_361340


namespace find_nat_numbers_l361_361668

theorem find_nat_numbers (n : ℕ) : (∃ p : ℕ, prime p ∧ 2 * n^2 - 5 * n - 33 = p^2) ↔ n = 6 ∨ n = 14 :=
by
  sorry

end find_nat_numbers_l361_361668


namespace determine_a_from_decreasing_interval_l361_361750

theorem determine_a_from_decreasing_interval (a : ℝ) :
  (∀ x : ℝ, x < 1 → derivative (λ x : ℝ, x^2 + a * x - 2) x < 0) →
  a = -2 :=
by
  -- Proof omitted
  sorry

end determine_a_from_decreasing_interval_l361_361750


namespace median_quiz_score_l361_361153

theorem median_quiz_score :
  let scores := [8, 8, 8, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12] in
  list.nth scores 12 = some 11 :=
by
  -- Proof would go here
  sorry

end median_quiz_score_l361_361153


namespace correctFinishingOrder_l361_361396

-- Define the types of athletes
inductive Athlete
| A | B | C | D | E

-- Define the type for predictions
def Prediction := Athlete → Nat

-- Define the first prediction
def firstPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 1
    | Athlete.B => 2
    | Athlete.C => 3
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the second prediction
def secondPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 4
    | Athlete.C => 1
    | Athlete.D => 5
    | Athlete.E => 2

-- Define the actual finishing positions
noncomputable def actualFinishingPositions : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 2
    | Athlete.C => 1
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the correctness conditions for the predictions
def correctPositions (actual : Prediction) (prediction : Prediction) (k : Nat) : Prop :=
  (Set.card {x : Athlete | actual x = prediction x} = k)

-- The proof statement
theorem correctFinishingOrder :
  correctPositions actualFinishingPositions firstPrediction 3 ∧
  correctPositions actualFinishingPositions secondPrediction 2 :=
sorry

end correctFinishingOrder_l361_361396


namespace no_three_distinct_zeros_l361_361318

noncomputable def f (a x : ℝ) : ℝ := Real.exp (2 * x) + a * Real.exp x - (a + 2) * x

theorem no_three_distinct_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) → False :=
by
  sorry

end no_three_distinct_zeros_l361_361318
