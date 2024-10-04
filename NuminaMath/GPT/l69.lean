import Mathlib
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Multiple
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Graph.Basic
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Gcd
import Mathlib.Data.Int.Modeq
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.FieldTheory.Rat
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Triangle
import Mathlib.NumberTheory.Palindrome
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Sorry
import Probability.Probability

namespace total_cupcakes_l69_69728

theorem total_cupcakes (children : ℕ) (cupcakes_per_child : ℕ) (total_cupcakes : ℕ) 
  (h1 : children = 8) (h2 : cupcakes_per_child = 12) : total_cupcakes = 96 := 
by
  sorry

end total_cupcakes_l69_69728


namespace range_of_a_l69_69526

variable (x a : ℝ)

-- Definitions of conditions as hypotheses
def condition_p (x : ℝ) := |x + 1| ≤ 2
def condition_q (x a : ℝ) := x ≤ a
def sufficient_not_necessary (p q : Prop) := p → q ∧ ¬(q → p)

-- The theorem statement
theorem range_of_a : sufficient_not_necessary (condition_p x) (condition_q x a) → 1 ≤ a ∧ ∀ b, b < 1 → sufficient_not_necessary (condition_p x) (condition_q x b) → false :=
by
  intro h
  sorry

end range_of_a_l69_69526


namespace figure_with_2022_squares_is_404_l69_69755

theorem figure_with_2022_squares_is_404 :
  ∃ N : ℕ, (N > 0) ∧ (7 + (N - 1) * 5 = 2022) ∧ (N = 404) := 
by
  use 404
  split
  sorry
  split
  sorry
  sorry

end figure_with_2022_squares_is_404_l69_69755


namespace find_a7_l69_69138

theorem find_a7 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) (x : ℝ)
  (h : x^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 +
            a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 +
            a_7 * (x + 1)^7 + a_8 * (x + 1)^8) : 
  a_7 = -8 := 
sorry

end find_a7_l69_69138


namespace race_length_l69_69291

theorem race_length (cristina_speed nicky_speed : ℕ) (head_start total_time : ℕ) 
  (h1 : cristina_speed > nicky_speed) 
  (h2 : head_start = 12) (h3 : cristina_speed = 5) (h4 : nicky_speed = 3) 
  (h5 : total_time = 30) :
  let nicky_distance := nicky_speed * total_time,
      cristina_time := total_time - head_start,
      cristina_distance := cristina_speed * cristina_time in
  nicky_distance = cristina_distance 
  ∧ nicky_distance = 90 := 
by
  sorry

end race_length_l69_69291


namespace quadratic_inequality_real_solutions_l69_69823

-- Definitions and conditions
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main statement
theorem quadratic_inequality_real_solutions (c : ℝ) (h_pos : 0 < c) : 
  (∀ x : ℝ, x^2 - 8 * x + c < 0) ↔ (c < 16) :=
by 
  sorry

end quadratic_inequality_real_solutions_l69_69823


namespace clock_angle_at_3_20_l69_69793

-- Define the conditions
def time_hour : ℕ := 3
def time_minute : ℕ := 20
def angle_formula (h m : ℕ) : ℝ :=
  abs ((60 * h - 11 * m) / 2)

-- State the main theorem
theorem clock_angle_at_3_20 : angle_formula time_hour time_minute = 20 :=
by
  sorry

end clock_angle_at_3_20_l69_69793


namespace car_speed_l69_69372

/--
A car traveling at a certain constant speed takes 2 seconds longer to travel 1 kilometer
than it would take to travel 1 kilometer at 80 kilometers per hour. We aim to find the speed
at which the car is traveling in kilometers per hour.
--/
theorem car_speed (v : ℝ) (h : (1 / 80 + 2 / 3600) = 1 / v) : v ≈ 76.6 :=
sorry

end car_speed_l69_69372


namespace limit_sequence_eq_l69_69670

open Real

theorem limit_sequence_eq :
  ∀ (ε : ℝ) (hε : ε > 0),
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → abs ((7 * n + 4) / (2 * n + 1) - 7 / 2) < ε :=
by
  sorry

end limit_sequence_eq_l69_69670


namespace find_vector_at_t_4_l69_69405

open Matrix

-- Define the vectors in terms of ℝ
noncomputable def vec_t_neg2 : ℝ ^ 3 := ![2, 6, 16]
noncomputable def vec_t_1 : ℝ ^ 3 := ![-1, -4, -8]
noncomputable def vec_t_4 : ℝ ^ 3 := ![-4, -10, -32]

-- Define the line parameterization function
noncomputable def line_eq (a d : ℝ ^ 3) (t : ℝ) : ℝ ^ 3 := a + (t • d)

-- State the existence of vectors a and d such that they satisfy the conditions
theorem find_vector_at_t_4 :
  ∃ (a d : ℝ ^ 3), line_eq a d (-2) = vec_t_neg2 ∧
                   line_eq a d (1) = vec_t_1 ∧
                   line_eq a d (4) = vec_t_4 :=
sorry

end find_vector_at_t_4_l69_69405


namespace number_of_triangles_in_pentadecagon_l69_69988

open Finset

theorem number_of_triangles_in_pentadecagon :
  ∀ (n : ℕ), n = 15 → (n.choose 3 = 455) := 
by 
  intros n hn 
  rw hn
  rw Nat.choose_eq_factorial_div_factorial (show 3 ≤ 15)
  { norm_num }

-- Proof omitted with sorry

end number_of_triangles_in_pentadecagon_l69_69988


namespace curve_C_equation_line_l_equation_PA_max_value_PA_min_value_l69_69177

-- Conditions for curve C
def curve_C (theta : ℝ) : ℝ × ℝ := (3 * Real.cos theta, 2 * Real.sin theta)

-- General equation for curve C
def curve_C_general (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Conditions for line l
def line_l (t : ℝ) : ℝ × ℝ := (2 + t, 2 - 2 * t)

-- General equation for line l
def line_l_general (x y : ℝ) : Prop := 2 * x + y - 6 = 0

-- Distance from point P to line l
def distance_P_to_l (theta : ℝ) : ℝ :=
  let (x, y) := curve_C theta;
  |(6 * (Real.cos theta) + 2 * (Real.sin theta) - 6) / (Real.sqrt 5)|

-- Maximum and minimum values of |PA|
def PA_max_min (theta alpha : ℝ) : ℝ × ℝ :=
  let max_val : ℝ := (6 * Real.sqrt 10 + 20) / 5;
  let min_val : ℝ := (-6 * Real.sqrt 10 + 20) / 5;
  (max_val, min_val)

-- Proof problems
theorem curve_C_equation (x y : ℝ) (theta : ℝ)
  (hC : (x, y) = curve_C theta) : curve_C_general x y := sorry

theorem line_l_equation (x y : ℝ) (t : ℝ)
  (hl : (x, y) = line_l t) : line_l_general x y := sorry

theorem PA_max_value (theta alpha : ℝ) (hα : Real.tan alpha = 3)
  (hmax : Real.sin (theta + alpha) = -1) : 
  distance_P_to_l theta / (Real.sin (Real.pi / 4)) = (6 * Real.sqrt 10 + 20) / 5 := sorry

theorem PA_min_value (theta alpha : ℝ) (hα : Real.tan alpha = 3)
  (hmin : Real.sin (theta + alpha) = 1) : 
  distance_P_to_l theta / (Real.sin (Real.pi / 4)) = (-6 * Real.sqrt 10 + 20) / 5 := sorry

end curve_C_equation_line_l_equation_PA_max_value_PA_min_value_l69_69177


namespace triangles_in_pentadecagon_l69_69955

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69955


namespace angles_subset_l69_69172

def set_of_angles_skew_lines : Set ℝ := {θ | 0 < θ ∧ θ ≤ π / 2}
def set_of_angles_line_plane : Set ℝ := {θ | 0 ≤ θ ∧ θ ≤ π / 2}

theorem angles_subset : set_of_angles_skew_lines ⊆ set_of_angles_line_plane := 
by 
  sorry

end angles_subset_l69_69172


namespace min_omega_l69_69553

def f (ω x φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega (ω : ℝ) (φ : ℝ) (h1 : ω > 0) (h2 : |φ| < Real.pi / 2)
  (h3 : f ω 0 φ = 1 / 2) (h4 : ∀ x : ℝ, f ω x φ ≤ f ω (Real.pi / 12) φ) :
  ω = 4 :=
sorry

end min_omega_l69_69553


namespace sequence_general_term_l69_69864

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 1 then 9 else 10 * n - 2

theorem sequence_general_term (S : ℕ → ℕ) (n : ℕ) (h : S = λ n, 5 * n^2 + 3 * n + 1) :
  sequence n = if n = 1 then 9 else 10 * n - 2 := 
sorry

end sequence_general_term_l69_69864


namespace hyperbola_equation_and_triangle_area_l69_69602

-- Define the hyperbola and its properties
def hyperbola (a : ℝ) (h : a > 0) : set (ℝ × ℝ) :=
  { p | ∃ x y, p = (x, y) ∧ (x^2 / a^2 - y^2 = 1) }

-- Define the eccentricity condition
def eccentricity_condition (a : ℝ) : Prop :=
  sqrt (a^2 + 1) / a = (2 * sqrt 3) / 3

-- Define the intersection points A and B and their properties
def intersection_points (a : ℝ) (m : ℝ) (h : a > 0) (h' : m ≠ 0) 
  : set (ℝ × ℝ) :=
  { p | ∃ x y, p = (x, y) 
          ∧ (x - 2) / (m * y) = 1
          ∧ (x^2 / 3 - y^2 = 1) }

-- Define the area calculation method
def triangle_area_min (m : ℝ) (y1 y2 : ℝ) (h : y1 ≠ y2) : ℝ :=
  sqrt ((12 * m^2 + 12) / (m^2 - 3)^2) / 2

-- Define the final minimum area condition
def minimum_area_condition (A B F1 : ℝ×ℝ) (m : ℝ)
  : ℝ :=
  let y1 := A.2 in
  let y2 := B.2 in
  min (triangle_area_min m y1 y2 sorry) (triangle_area_min m y2 y1 sorry)

theorem hyperbola_equation_and_triangle_area
  (a : ℝ) (h : a > 0) 
  (m : ℝ) (nonzeroM : m ≠ 0)
  (A B F1 : ℝ×ℝ) (eq1 : eccentricity_condition a) : 
  (hyperbola a h = {p : ℝ×ℝ | p.1^2 / 3 - p.2^2 = 1}) ∧ 
  (minimum_area_condition A B F1 m ≥ 4 / 3) :=
begin
  sorry
end

end hyperbola_equation_and_triangle_area_l69_69602


namespace monic_quadratic_real_root_l69_69829

theorem monic_quadratic_real_root (a b : ℂ) (h : b = 2 - 3 * complex.I) :
  ∃ P : polynomial ℂ, P.monic ∧ P.coeff 2 = 1 ∧ P.coeff 1 = -4 ∧ P.coeff 0 = 13 ∧ P.is_root (2 - 3 * complex.I) :=
by
  sorry

end monic_quadratic_real_root_l69_69829


namespace sqrt_x_plus_inv_sqrt_x_l69_69638

variable (x : ℝ) (hx : 0 < x) (h : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (hx : 0 < x) (h : x + 1 / x = 50) : 
  sqrt x + 1 / sqrt x = 2 * sqrt 13 := 
sorry

end sqrt_x_plus_inv_sqrt_x_l69_69638


namespace sum_of_series_l69_69486

theorem sum_of_series :
  (\sum n in [2, 3, 4, 5, 6, 7],  1 / (n * (n+1))) = 3 / 8 :=
by
  sorry

end sum_of_series_l69_69486


namespace reaction_requires_two_moles_of_HNO3_l69_69930

def nitric_acid_reaction (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) 
  (reaction : HNO3 + NaHCO3 = NaNO3 + CO2 + H2O)
  (n_NaHCO3 : ℕ) : ℕ :=
  if n_NaHCO3 = 2 then 2 else sorry

theorem reaction_requires_two_moles_of_HNO3
  (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) 
  (reaction : HNO3 + NaHCO3 = NaNO3 + CO2 + H2O)
  (n_NaHCO3 : ℕ) :
  n_NaHCO3 = 2 → nitric_acid_reaction HNO3 NaHCO3 NaNO3 CO2 H2O reaction n_NaHCO3 = 2 :=
by sorry

end reaction_requires_two_moles_of_HNO3_l69_69930


namespace find_sinB_find_c_l69_69587

variables (a b c : ℝ) (cosA sinB : ℝ)

-- Conditions
def conditions := (a = 3) ∧ (b = 2) ∧ (cosA = 1/3)

-- Problem 1: Find the value of sin B
theorem find_sinB (hc : conditions) : sinB = 4 * real.sqrt 2 / 9 :=
by
  sorry

-- Problem 2: Find the value of c
theorem find_c (hc : conditions) : c = 3 :=
by
  sorry

end find_sinB_find_c_l69_69587


namespace minimum_value_of_expression_l69_69574

theorem minimum_value_of_expression {x : ℝ} (hx : x > 0) : (2 / x + x / 2) ≥ 2 :=
by sorry

end minimum_value_of_expression_l69_69574


namespace regular_hexagon_area_l69_69305

-- Given Conditions
def J : EuclideanGeometry.Point := (0, 0)
def L : EuclideanGeometry.Point := (10, 2)
def isRegularHexagon (H : EuclideanGeometry.Hexagon) : Prop := EuclideanGeometry.regular H
def JKLMPQ : EuclideanGeometry.Hexagon := (J, A, L, B, C, D) -- We assume A, B, C, D are points such that JKLMPQ forms a regular hexagon

-- Main Theorem Statement 
theorem regular_hexagon_area :
  isRegularHexagon JKLMPQ →
  EuclideanGeometry.area JKLMPQ = 156 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_l69_69305


namespace no_real_solutions_eq_l69_69324

theorem no_real_solutions_eq (x y : ℝ) :
  x^2 + y^2 - 2 * x + 4 * y + 6 ≠ 0 :=
sorry

end no_real_solutions_eq_l69_69324


namespace min_value_expression_l69_69506

noncomputable def satisfies_system (x y z : ℝ) : Prop :=
  2 * Real.cos x = Real.cot y ∧
  2 * Real.sin y = Real.tan z ∧
  Real.cos z = Real.cot x

noncomputable def target_expression (x z : ℝ) : ℝ :=
  Real.sin x + Real.cos z

theorem min_value_expression :
  ∃ (x y z : ℝ), satisfies_system x y z ∧
  ∀ (x' y' z' : ℝ), satisfies_system x' y' z' → 
  target_expression x' z' ≥ - (5 * Real.sqrt 3) / 6 ∧
  target_expression x z = - (5 * Real.sqrt 3) / 6 :=
sorry

end min_value_expression_l69_69506


namespace max_prizes_l69_69222

theorem max_prizes (n : ℕ) (h₁ : n = 50) (h₂ : ∀ i : ℕ, i < n → 1 ≤ prizes_received i) (h₃ : (∑ i in finset.range n, prizes_received i) / n = 7) : 
  ∃ (max_prizes_per_participant : ℕ), max_prizes_per_participant = 301 :=
by 
  sorry

end max_prizes_l69_69222


namespace working_light_bulbs_l69_69308

theorem working_light_bulbs (total_lamps : ℕ) (light_bulbs_per_lamp : ℕ) (fraction_burnt_out : ℚ) (burnt_out_per_lamp : ℕ) :
  total_lamps = 20 →
  light_bulbs_per_lamp = 7 →
  fraction_burnt_out = 1 / 4 →
  burnt_out_per_lamp = 2 →
  (total_lamps * light_bulbs_per_lamp) - (total_lamps * fraction_burnt_out.numerator * burnt_out_per_lamp / fraction_burnt_out.denominator) = 130 :=
by
  intros h_total h_per_lamp h_fraction h_burnt_out
  rw [h_total, h_per_lamp, h_fraction, h_burnt_out]
  norm_num
  sorry

end working_light_bulbs_l69_69308


namespace four_digit_perfect_square_palindrome_count_l69_69012

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69012


namespace monic_quadratic_with_root_2_minus_3i_l69_69852

theorem monic_quadratic_with_root_2_minus_3i :
  ∃ P : ℝ[X], P.monic ∧ (P.coeff 2 = 1)
    ∧ (P.coeff 1 = -4)
    ∧ (P.coeff 0 = 13)
    ∧ eval (2 - 3 * I) P = 0 := sorry

end monic_quadratic_with_root_2_minus_3i_l69_69852


namespace monic_quadratic_with_root_2_minus_3i_l69_69850

theorem monic_quadratic_with_root_2_minus_3i :
  ∃ P : ℝ[X], P.monic ∧ (P.coeff 2 = 1)
    ∧ (P.coeff 1 = -4)
    ∧ (P.coeff 0 = 13)
    ∧ eval (2 - 3 * I) P = 0 := sorry

end monic_quadratic_with_root_2_minus_3i_l69_69850


namespace tangent_line_at_point_curve_symmetric_range_of_a_extreme_l69_69913

def f (x a : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- Part 1
theorem tangent_line_at_point (a : ℝ) (h : a = -1) :
  let x := 1
  let t := Real.log(1 + 1)
  let f1 := (1 + a) * t
  let f1' := (-1) * t
  y = f1' * (x-1) + f1 := by sorry

-- Part 2
theorem curve_symmetric (b : ℝ) (a : ℝ) :
  let x := 1
  b = -(1/2)
  let lhs := (1 + a) * Real.log 2
  let rhs := (2 - a) * Real.log 2
  lhs = rhs → a = 1/2 := by sorry

-- Part 3
theorem range_of_a_extreme (a : ℝ) :
  ∃ x > 0, f x a = 0 ↔ 0 < a ∧ a < 1/2 := by sorry

end tangent_line_at_point_curve_symmetric_range_of_a_extreme_l69_69913


namespace count_valid_n_l69_69127

theorem count_valid_n : 
  ∃ n_values : Finset ℤ, 
    (∀ n ∈ n_values, (n + 2 ≤ 6 * n - 8) ∧ (6 * n - 8 < 3 * n + 7)) ∧
    (n_values.card = 3) :=
by sorry

end count_valid_n_l69_69127


namespace range_of_m_l69_69550

noncomputable def f (x : ℝ) := 4 * (Real.sin(π / 4 + x))^2 - 2 * Real.sqrt 3 * Real.cos (2 * x) - 1

def p (x : ℝ) := x < π / 4 ∨ x > π / 2

def q (x : ℝ) (m : ℝ) := -3 < f x - m ∧ f x - m < 3

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ¬p x → q x m) → 2 < m ∧ m < 6 :=
by
  sorry

end range_of_m_l69_69550


namespace puree_volume_correct_l69_69195

noncomputable def final_volume_puree (tomato_juice_volume carrot_juice_volume spinach_juice_volume : ℝ) 
(water_content_tomato water_content_carrot water_content_spinach final_water_content : ℝ)
(tomato_juice_volume = 20) (carrot_juice_volume = 12) (spinach_juice_volume = 8)
(water_content_tomato = 0.9) (water_content_carrot = 0.88) (water_content_spinach = 0.91)
(final_water_content = 0.25) : ℝ := 
  let water_tomato := tomato_juice_volume * water_content_tomato
  let water_carrot := carrot_juice_volume * water_content_carrot
  let water_spinach := spinach_juice_volume * water_content_spinach
  let total_water := water_tomato + water_carrot + water_spinach

  let solids_tomato := tomato_juice_volume * (1 - water_content_tomato)
  let solids_carrot := carrot_juice_volume * (1 - water_content_carrot)
  let solids_spinach := spinach_juice_volume * (1 - water_content_spinach)
  let total_solids := solids_tomato + solids_carrot + solids_spinach

  let total_volume_before_boiling := total_water + total_solids

  let final_volume := total_solids / (1 - final_water_content)
  
  final_volume

theorem puree_volume_correct (tomato_juice_volume = 20) (carrot_juice_volume = 12) (spinach_juice_volume = 8)
  (water_content_tomato = 0.9) (water_content_carrot = 0.88) (water_content_spinach = 0.91)
  (final_water_content = 0.25) :
  final_volume_puree 20 12 8 0.9 0.88 0.91 0.25 = 5.55 := sorry

end puree_volume_correct_l69_69195


namespace new_average_increased_by_40_percent_l69_69687

theorem new_average_increased_by_40_percent 
  (n : ℕ) (initial_avg : ℝ) (initial_marks : ℝ) (new_marks : ℝ) (new_avg : ℝ)
  (h1 : n = 37)
  (h2 : initial_avg = 73)
  (h3 : initial_marks = (initial_avg * n))
  (h4 : new_marks = (initial_marks * 1.40))
  (h5 : new_avg = (new_marks / n)) :
  new_avg = 102.2 :=
sorry

end new_average_increased_by_40_percent_l69_69687


namespace find_x_squared_plus_y_squared_l69_69572

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared
  (h1 : x - y = 10)
  (h2 : x * y = 9) :
  x^2 + y^2 = 118 :=
sorry

end find_x_squared_plus_y_squared_l69_69572


namespace distribution_of_books_l69_69474

theorem distribution_of_books :
  let books := 5
  let students := 4
  let total_ways := 4^5
  let no_books_from_one_student := 4 * 3^5
  let one_student_case := 4 * 3^5
  let exactly_two_students_case := choose 4 2 * 2^5
  total_ways - no_books_from_one_student + exactly_two_students_case = 292 :=
by
  let books := 5
  let students := 4
  let total_ways := 4^5
  let no_books_from_one_student := 4 * 3^5
  let one_student_case := 4 * 3^5
  let exactly_two_students_case := choose 4 2 * 2^5
  have total_ways_value : total_ways = 1024 := by norm_num,
  have no_books_value : no_books_from_one_student = 972 := by norm_num,
  have two_students_value : exactly_two_students_case = 240 := by norm_num,
  calc
    1024 - 972 + 240 = 292 := by norm_num

end distribution_of_books_l69_69474


namespace smallest_sector_angle_3_l69_69253

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_angles_is_360 (a : ℕ → ℕ) : Prop :=
  (Finset.range 15).sum a = 360

def smallest_possible_angle (a : ℕ → ℕ) (x : ℕ) : Prop :=
  ∀ i : ℕ, a i ≥ x

theorem smallest_sector_angle_3 :
  ∃ a : ℕ → ℕ,
    is_arithmetic_sequence a ∧
    sum_of_angles_is_360 a ∧
    smallest_possible_angle a 3 :=
sorry

end smallest_sector_angle_3_l69_69253


namespace sum_first_five_terms_l69_69149

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

theorem sum_first_five_terms (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2 : a 1 + a 3 = 6) : S_5 a = 15 :=
by
  -- skipping actual proof
  sorry

end sum_first_five_terms_l69_69149


namespace value_of_sqrt_x_plus_one_over_sqrt_x_l69_69648

noncomputable def find_value (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) : ℝ :=
  sqrt(x) + 1/sqrt(x)

theorem value_of_sqrt_x_plus_one_over_sqrt_x (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) :
  find_value x hx_pos hx = 2 * sqrt(13) :=
sorry

end value_of_sqrt_x_plus_one_over_sqrt_x_l69_69648


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l69_69647

variable (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_52 : (Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52) :=
by
  sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l69_69647


namespace minimum_value_func_l69_69333

-- Define the function
def func (x : ℝ) : ℝ := x + 3 / (4 * x)

-- State the problem: proving the minimum value for x > 0 is sqrt(3)
theorem minimum_value_func : ∃ x > 0, func x = Real.sqrt 3 ∧ ∀ y > 0, func y ≥ Real.sqrt 3 :=
sorry

end minimum_value_func_l69_69333


namespace sum_squares_divisible_by_4_iff_even_l69_69258

theorem sum_squares_divisible_by_4_iff_even (a b c : ℕ) (ha : a % 2 = 0) (hb : b % 2 = 0) (hc : c % 2 = 0) : 
(a^2 + b^2 + c^2) % 4 = 0 ↔ 
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) :=
sorry

end sum_squares_divisible_by_4_iff_even_l69_69258


namespace find_x_squared_plus_y_squared_l69_69571

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared
  (h1 : x - y = 10)
  (h2 : x * y = 9) :
  x^2 + y^2 = 118 :=
sorry

end find_x_squared_plus_y_squared_l69_69571


namespace divide_64x64_grid_l69_69219

theorem divide_64x64_grid :
  ∃ (f : Fin (64 * 64 - 1) → (Fin 64 × Fin 64) × (Fin 64 × Fin 64) × (Fin 64 × Fin 64)),
    ∀ (i : Fin (64 * 64 - 1)), 
      let (a, b, c) := f i
      in a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ -- ensures distinct cells
         (a.1 - b.1) * (a.1 - b.1) + (a.2 - b.2) * (a.2 - b.2) +
         (b.1 - c.1) * (b.1 - c.1) + (b.2 - c.2) * (b.2 - c.2) +
         (c.1 - a.1) * (c.1 - a.1) + (c.2 - a.2) * (c.2 - a.2) = 6 := 
sorry

end divide_64x64_grid_l69_69219


namespace number_of_triangles_in_pentadecagon_l69_69981

open Finset

theorem number_of_triangles_in_pentadecagon :
  ∀ (n : ℕ), n = 15 → (n.choose 3 = 455) := 
by 
  intros n hn 
  rw hn
  rw Nat.choose_eq_factorial_div_factorial (show 3 ≤ 15)
  { norm_num }

-- Proof omitted with sorry

end number_of_triangles_in_pentadecagon_l69_69981


namespace sqrt_sum_le_inv_sum_l69_69186

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 2|

def M : Set ℝ := {x | f x ≤ 2}

def m : ℝ := 1

theorem sqrt_sum_le_inv_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = m) : 
  sqrt(a) + sqrt(b) + sqrt(c) ≤ (1 / a) + (1 / b) + (1 / c) :=
by
  sorry

end sqrt_sum_le_inv_sum_l69_69186


namespace regular_18_gon_trig_eq_l69_69450

noncomputable def regular_18_gon_angle_relation : Prop :=
  let θ := 10 * Real.pi / 180
  in Real.cot θ = 4 * Real.cos θ + Real.sqrt 3

theorem regular_18_gon_trig_eq : regular_18_gon_angle_relation :=
by
  unfold regular_18_gon_angle_relation
  sorry

end regular_18_gon_trig_eq_l69_69450


namespace monic_quadratic_with_real_coeffs_l69_69835

open Complex Polynomial

theorem monic_quadratic_with_real_coeffs {x : ℂ} :
  (∀ a b c : ℝ, Polynomial.monic (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) ∧ 
  (x = 2 - 3 * Complex.I ∨ x = 2 + 3 * Complex.I) → Polynomial.eval (2 - 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0) ∧
  Polynomial.eval (2 + 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0 :=
begin
  sorry
end

end monic_quadratic_with_real_coeffs_l69_69835


namespace bong_paint_time_l69_69621

-- Definitions
def rate_jay := 1 / 2
def rate_together := 1 / 1.2

-- Theorem: Bong's time to paint the wall alone
theorem bong_paint_time : ∃ B : ℝ, (1 / B) + rate_jay = rate_together ∧ B = 3 :=
by
  sorry

end bong_paint_time_l69_69621


namespace monic_quadratic_with_root_2_minus_3i_l69_69849

theorem monic_quadratic_with_root_2_minus_3i :
  ∃ P : ℝ[X], P.monic ∧ (P.coeff 2 = 1)
    ∧ (P.coeff 1 = -4)
    ∧ (P.coeff 0 = 13)
    ∧ eval (2 - 3 * I) P = 0 := sorry

end monic_quadratic_with_root_2_minus_3i_l69_69849


namespace four_digit_perfect_square_palindrome_count_l69_69007

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69007


namespace find_a_plus_2b_l69_69135

variable {R : Type} [Field R] (a b x : R)
def f (x : R) := a * x + 2 * b
def f_inv (x : R) := b * x + 2 * a

-- Stating the condition that f and f_inv are true inverses
axiom Hf_inv : ∀ x, f (f_inv x) = x

-- Proving the required result
theorem find_a_plus_2b : a + 2 * b = -3 := by
  sorry

end find_a_plus_2b_l69_69135


namespace triangles_in_pentadecagon_l69_69972

def regular_pentadecagon := {vertices : Finset Point | vertices.card = 15 ∧ 
  ∀ a b c ∈ vertices, ¬Collinear a b c}

theorem triangles_in_pentadecagon (P : regular_pentadecagon) : 
  (P.vertices.card.choose 3) = 455 :=
by 
  sorry


end triangles_in_pentadecagon_l69_69972


namespace arithmetic_seq_S13_l69_69150

noncomputable def arithmetic_sequence_sum (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_seq_S13 (a_1 d : ℕ) (h : a_1 + 6 * d = 10) :
  arithmetic_sequence_sum a_1 d 13 = 130 :=
by
  sorry

end arithmetic_seq_S13_l69_69150


namespace recorder_new_price_l69_69663

theorem recorder_new_price (a b : ℕ) (y x: ℕ)
  (h₁ : x < 50)
  (h₂ : y = 10 * a + b)
  (h₃ : x = 10 * b + a)
  (h₄ : y = 1.2 * x) :
  y = 54 :=
by {
  have h₅ : 10 * a + b = 1.2 * (10 * b + a) := by rw [h₂, h₃, h₄],
  have h₆ : 10 * a + b = 12 * b + 1.2 * a := by linarith,
  have h₇ : 8.8 * a = 11 * b := by linarith,
  have h₈ : (8.8 / 11) * a = b := by linarith,
  have h₉ : (4 / 5) * a = b := by norm_num,
  have h₁₀ : (4 * a / 5) ∈ ℕ := sorry, -- a solution step to prove integer nature
  have h₁₁ : a = 5 := sorry, -- using digit constraints and org calculations
  have h₁₂ : b = 4 := by rw [h₉, h₁₁]; norm_num,
  have h₁₃ : x = 45 := by rw [h₃, h₁₁, h₁₂]; norm_num,
  have h₁₄ : y = 1.2 * 45 := by rw h₄; exact rfl,
  norm_num at h₁₄, exact h₁₄,
  sorry  -- provide complete proof steps in lean language
}

end recorder_new_price_l69_69663


namespace four_digit_palindrome_square_count_l69_69000

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l69_69000


namespace min_distance_l69_69818

noncomputable def min_distance_to_line (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) : ℝ :=
  (| sqrt 3 * (P.1 - l 0).1 - (P.2 - l 0).2 + 2 * sqrt 3 |) / sqrt 4

theorem min_distance (θ : ℝ) : 
  let l := λ t : ℝ, (-2 + (1/2) * t, (sqrt 3 / 2) * t)
  let C2 := λ θ : ℝ, (cos θ, sqrt 3 * sin θ)
  let P := C2 θ
  min_distance_to_line P l = (2 * sqrt 3 - sqrt 6) / 2 :=
sorry

end min_distance_l69_69818


namespace desired_minoxidil_percentage_l69_69409

-- Define the amounts of Minoxidil in the given solutions
def minoxidil_amount (volume : ℝ) (percentage : ℝ) : ℝ := volume * percentage / 100

-- Given conditions
def S1_volume : ℝ := 70
def S1_percentage : ℝ := 2
def S2_volume : ℝ := 35
def S2_percentage : ℝ := 5

-- Calculate the final Minoxidil percentage
def final_percentage (total_amount : ℝ) (total_volume : ℝ) : ℝ := (total_amount / total_volume) * 100

-- Theorem statement
theorem desired_minoxidil_percentage :
  final_percentage (minoxidil_amount S1_volume S1_percentage + minoxidil_amount S2_volume S2_percentage)
                   (S1_volume + S2_volume) = 3 := by
  sorry

end desired_minoxidil_percentage_l69_69409


namespace range_of_x_l69_69518

theorem range_of_x (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) : 
  12 ≤ x := 
sorry

end range_of_x_l69_69518


namespace value_of_f_neg1_range_of_x_for_fx_gt_1_l69_69906

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then 3^x else -x + 3

theorem value_of_f_neg1 : f (-1) = 1 / 3 :=
by sorry

theorem range_of_x_for_fx_gt_1 : {x : ℝ | f x > 1} = {x | 0 < x ∧ x < 2 } :=
by sorry

end value_of_f_neg1_range_of_x_for_fx_gt_1_l69_69906


namespace center_square_number_l69_69218

def in_center_square (grid : Matrix (Fin 3) (Fin 3) ℕ) : ℕ := grid 1 1

theorem center_square_number
  (grid : Matrix (Fin 3) (Fin 3) ℕ)
  (consecutive_share_edge : ∀ (i j : Fin 3) (n : ℕ), 
                              (i < 2 ∨ j < 2) →
                              (∃ d, d ∈ [(-1,0), (1,0), (0,-1), (0,1)] ∧ 
                              grid (i + d.1) (j + d.2) = n + 1))
  (corner_sum_20 : grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 20)
  (diagonal_sum_15 : 
    (grid 0 0 + grid 1 1 + grid 2 2 = 15) 
    ∨ 
    (grid 0 2 + grid 1 1 + grid 2 0 = 15))
  : in_center_square grid = 5 := sorry

end center_square_number_l69_69218


namespace second_number_value_l69_69140

-- Definition of the problem conditions
variables (x y z : ℝ)
axiom h1 : z = 4.5 * y
axiom h2 : y = 2.5 * x
axiom h3 : (x + y + z) / 3 = 165

-- The goal is to prove y = 82.5 given the conditions h1, h2, and h3
theorem second_number_value : y = 82.5 :=
by
  sorry

end second_number_value_l69_69140


namespace trigonometric_simplification_logarithmic_simplification_l69_69681

-- Definition for the first problem (trigonometric simplification)
theorem trigonometric_simplification (θ : ℝ) (h : Real.tan θ = 2) :
    (Real.sin (θ + (Real.pi / 2)) * Real.cos ((Real.pi / 2) - θ) - (Real.cos (Real.pi - θ)) ^ 2) / (1 + (Real.sin θ) ^ 2) = 1 / 3 :=
by
  sorry

-- Definition for the second problem (logarithmic simplification)
theorem logarithmic_simplification (x : ℝ) :
    Real.ln (Real.sqrt (x ^ 2 + 1) + x) + 
    Real.ln (Real.sqrt (x ^ 2 + 1) - x) + 
    (Real.log 2) ^ 2 + 
    (1 + Real.log 2) * Real.log 5 - 
    2 * Real.sin (Real.pi / 6) = 0 :=
by
  sorry

end trigonometric_simplification_logarithmic_simplification_l69_69681


namespace transformed_variance_is_nine_l69_69174

noncomputable def variance (x : list ℝ) : ℝ :=
let n := x.length in
let mean := (x.sum / n : ℝ) in
(1 / n) * (x.map (λ a, (a - mean)^2)).sum

theorem transformed_variance_is_nine (a1 a2 a3 : ℝ)
  (h : variance [a1, a2, a3] = 1) :
  variance [3 * a1 + 2, 3 * a2 + 2, 3 * a3 + 2] = 9 :=
sorry

end transformed_variance_is_nine_l69_69174


namespace triangles_in_pentadecagon_l69_69949

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69949


namespace evaluate_expression_l69_69480

theorem evaluate_expression :
  (∑ i in Finset.range 2023, (2022 - i + 1) / i.succ) /
  (∑ i in Finset.range (2022 + 2) \ Finset.singleton 0, (1 : ℚ) / i) = 2023 := 
begin
  sorry
end

end evaluate_expression_l69_69480


namespace arithmetic_geometric_seq_S6_l69_69151

variable S : ℕ → ℝ

theorem arithmetic_geometric_seq_S6 (h1 : S 2 = 1) (h2 : S 4 = 3) : S 6 = 7 := by
sorry

end arithmetic_geometric_seq_S6_l69_69151


namespace coefficient_x3_expansion_l69_69608

theorem coefficient_x3_expansion : 
  let expr1 := (2 - x) * (2 + x)^5 
  in polynomial.coeff (polynomial.expand ℝ (polynomial.C expr1)) 3 = 0 :=
by sorry

end coefficient_x3_expansion_l69_69608


namespace monic_quadratic_with_real_coeffs_l69_69836

open Complex Polynomial

theorem monic_quadratic_with_real_coeffs {x : ℂ} :
  (∀ a b c : ℝ, Polynomial.monic (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) ∧ 
  (x = 2 - 3 * Complex.I ∨ x = 2 + 3 * Complex.I) → Polynomial.eval (2 - 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0) ∧
  Polynomial.eval (2 + 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0 :=
begin
  sorry
end

end monic_quadratic_with_real_coeffs_l69_69836


namespace even_function_a_is_negative_one_over_four_l69_69206

noncomputable def f (a x : ℝ) : ℝ := 2 * a * x + real.log (exp x + 1)

theorem even_function_a_is_negative_one_over_four :
  ∀ a : ℝ, (∀ x : ℝ, f a x = f a (-x)) ↔ a = -1/4 := 
sorry

end even_function_a_is_negative_one_over_four_l69_69206


namespace sin_cos_equation_l69_69139

-- Conditions
variable (x : ℝ)
variable (hx : 0 < x ∧ x < π / 2)
variable (hcos : cos (x + π / 12) = sqrt 2 / 10)

-- Question and Answer as Lean theorem statement
theorem sin_cos_equation :
  sin x + sqrt 3 * cos x = 8 / 5 :=
by
  sorry

end sin_cos_equation_l69_69139


namespace reduced_rates_apply_on_weekend_l69_69393

def total_hours_in_week := 7 * 24

def fraction_of_week_with_reduced_rates := 0.6428571428571429

def reduced_rate_hours := total_hours_in_week * fraction_of_week_with_reduced_rates

def weekday_reduced_rate_hours_per_day := 12

def total_weekday_reduced_rate_hours := 5 * weekday_reduced_rate_hours_per_day

def remaining_reduced_rate_hours := reduced_rate_hours - total_weekday_reduced_rate_hours

def full_days_with_reduced_rate := remaining_reduced_rate_hours / 24

theorem reduced_rates_apply_on_weekend : full_days_with_reduced_rate = 2 := by
  sorry

end reduced_rates_apply_on_weekend_l69_69393


namespace max_value_norm_c_l69_69160

open_locale real_inner_product -- to allow usage of the inner product notation

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def max_norm_of_c
  (a b c : V)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (h : ∥c - (a + b)∥ = ∥a - b∥) : ℝ :=
  2 * real.sqrt 2

theorem max_value_norm_c
  (a b c : V)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (h : ∥c - (a + b)∥ = ∥a - b∥) :
  ∥c∥ ≤ 2 * real.sqrt 2 :=
sorry

end max_value_norm_c_l69_69160


namespace range_of_a_for_inequality_l69_69209

theorem range_of_a_for_inequality :
  {a : ℝ // ∀ (x : ℝ), a * x^2 + 2 * a * x + 1 > 0} = {a : ℝ // 0 ≤ a ∧ a < 1} :=
sorry

end range_of_a_for_inequality_l69_69209


namespace distance_between_parallel_lines_l69_69456

noncomputable def vector_a : ℝ × ℝ := (4, -2)
noncomputable def vector_b : ℝ × ℝ := (3, -1)
noncomputable def direction_d : ℝ × ℝ := (2, -3)

theorem distance_between_parallel_lines :
  let v := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let p := ((v.1 * direction_d.1 + v.2 * direction_d.2) / (direction_d.1 * direction_d.1 + direction_d.2 * direction_d.2) * direction_d.1,
            (v.1 * direction_d.1 + v.2 * direction_d.2) / (direction_d.1 * direction_d.1 + direction_d.2 * direction_d.2) * direction_d.2)
  let c := (vector_b.1 + p.1, vector_b.2 + p.2)
  let distance := real.sqrt ((vector_a.1 - c.1)^2 + (vector_a.2 - c.2)^2)
  distance = real.sqrt(13) / 13 :=
by
  sorry

end distance_between_parallel_lines_l69_69456


namespace wire_cut_ratio_l69_69422

-- Define lengths a and b
variable (a b : ℝ)

-- Define perimeter equal condition
axiom perimeter_eq : 4 * (a / 4) = 6 * (b / 6)

-- The statement to prove
theorem wire_cut_ratio (h : 4 * (a / 4) = 6 * (b / 6)) : a / b = 1 :=
by
  sorry

end wire_cut_ratio_l69_69422


namespace find_angle_D_l69_69234

-- Define the given angles and conditions
def angleA := 30
def angleB (D : ℝ) := 2 * D
def angleC (D : ℝ) := D + 40
def sum_of_angles (A B C D : ℝ) := A + B + C + D = 360

theorem find_angle_D (D : ℝ) (hA : angleA = 30) (hB : angleB D = 2 * D) (hC : angleC D = D + 40) (hSum : sum_of_angles angleA (angleB D) (angleC D) D):
  D = 72.5 :=
by
  -- Proof is omitted
  sorry

end find_angle_D_l69_69234


namespace rate_percent_per_annum_l69_69577

-- Definitions and conditions
def principal (P : ℝ) := P
def time := 5
def final_amount (P : ℝ) := 2 * P
def simple_interest (P R T : ℝ) := (P * R * T) / 100

-- The Lean statement proving the rate percent per annum R
theorem rate_percent_per_annum (P R : ℝ) : 
  final_amount P = 2 * P →
  simple_interest P R time = P →
  R = 20 :=
by
  intro h1 h2
  simp [simple_interest, time] at h2
  linarith

end rate_percent_per_annum_l69_69577


namespace range_of_a_l69_69181

def f (a : ℝ) := if x ≤ 1 then -x^2 + a * x else 2 * a * x - 5

def exists_intersections (a b : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 - b = 0 ∧ f a x2 - b = 0

theorem range_of_a (a : ℝ) : 
  (∃ b : ℝ, exists_intersections a b) → a < 4 := by
  sorry

end range_of_a_l69_69181


namespace enclosed_area_l69_69467

theorem enclosed_area {x y : ℝ} (h : x^2 + y^2 = 2 * |x| + 2 * |y|) : ∃ (A : ℝ), A = 8 :=
sorry

end enclosed_area_l69_69467


namespace four_digit_palindromic_perfect_square_count_l69_69040

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69040


namespace quadratic_inequality_real_solutions_l69_69824

-- Definitions and conditions
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main statement
theorem quadratic_inequality_real_solutions (c : ℝ) (h_pos : 0 < c) : 
  (∀ x : ℝ, x^2 - 8 * x + c < 0) ↔ (c < 16) :=
by 
  sorry

end quadratic_inequality_real_solutions_l69_69824


namespace fn_1992_k_l69_69861

-- Define the sum of the digits of a positive integer in decimal notation.
def sum_of_digits (k : ℕ) : ℕ :=
  k.digitsBase 10 |>.sum

-- Define f₁ which is the square of the sum of the digits of k.
def f1 (k : ℕ) : ℕ :=
  (sum_of_digits k) ^ 2

-- Define fₙ recursively for n > 1.
def f : ℕ → ℕ → ℕ
| 1, k := f1 k
| (n + 1), k := f1 (f n k)

-- Define the specific k.
def k : ℕ := 2 ^ 1991

-- The final statement we want to prove.
theorem fn_1992_k : f 1992 k = 256 := by
  sorry

end fn_1992_k_l69_69861


namespace find_b_from_quadratic_l69_69660

theorem find_b_from_quadratic (b n : ℤ)
  (h1 : b > 0)
  (h2 : (x : ℤ) → (x + n)^2 - 6 = x^2 + b * x + 19) :
  b = 10 :=
sorry

end find_b_from_quadratic_l69_69660


namespace anne_markers_l69_69792

theorem anne_markers (drawings_per_marker : ℝ) 
                     (drawings_made : ℕ) 
                     (drawings_remaining : ℕ) 
                     (h_drawings_per_marker : drawings_per_marker = 1.5)
                     (h_drawings_made : drawings_made = 8)
                     (h_drawings_remaining : drawings_remaining = 10) :
  let total_drawings := drawings_made + drawings_remaining in
  let markers : ℝ := total_drawings / drawings_per_marker in
  markers = 12 :=
by
  -- Introduce the conditions
  have h1 : drawings_per_marker = 1.5 := h_drawings_per_marker
  have h2 : drawings_made = 8 := h_drawings_made
  have h3 : drawings_remaining = 10 := h_drawings_remaining
  
  -- Compute total drawings:
  let total_drawings := drawings_made + drawings_remaining
  
  -- Compute number of markers
  let markers : ℝ := total_drawings / drawings_per_marker

  exact sorry

end anne_markers_l69_69792


namespace min_value_of_a_l69_69211

theorem min_value_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x^2 + 2*x*y ≤ a*(x^2 + y^2)) → (a ≥ (Real.sqrt 5 + 1) / 2) := 
sorry

end min_value_of_a_l69_69211


namespace log_seven_eighteen_l69_69568

theorem log_seven_eighteen (a b : ℝ) (h1 : log 10 2 = a) (h2 : log 10 3 = b) : 
  log 7 18 = (a + 2 * b) / log 10 7 := 
by 
  sorry

end log_seven_eighteen_l69_69568


namespace triangles_in_pentadecagon_l69_69976

theorem triangles_in_pentadecagon :
  let n := 15
  in (Nat.choose n 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69976


namespace triangles_in_pentadecagon_l69_69968

def regular_pentadecagon := {vertices : Finset Point | vertices.card = 15 ∧ 
  ∀ a b c ∈ vertices, ¬Collinear a b c}

theorem triangles_in_pentadecagon (P : regular_pentadecagon) : 
  (P.vertices.card.choose 3) = 455 :=
by 
  sorry


end triangles_in_pentadecagon_l69_69968


namespace inequality_harmonic_l69_69655

def a (n : ℕ) : ℝ := ∑ i in Finset.range (n+1) \ Finset.range 1, 1 / (i : ℝ)

theorem inequality_harmonic (n : ℕ) (h : 2 ≤ n) :
  (a n) ^ 2 > 2 * ∑ i in Finset.range (n+1) \ Finset.range 2, (a i) / i :=
sorry

end inequality_harmonic_l69_69655


namespace relationship_among_mnr_l69_69870

-- Definitions of the conditions
variables {a b c : ℝ}
variables (m n r : ℝ)

-- Assumption given by the conditions
def conditions (a b c : ℝ) := 0 < a ∧ a < b ∧ b < 1 ∧ 1 < c
def log_equations (a b c m n : ℝ) := m = Real.log c / Real.log a ∧ n = Real.log c / Real.log b
def r_definition (a c r : ℝ) := r = a^c

-- Statement: If the conditions are satisfied, then the relationship holds
theorem relationship_among_mnr (a b c m n r : ℝ)
  (h1 : conditions a b c)
  (h2 : log_equations a b c m n)
  (h3 : r_definition a c r) :
  n < m ∧ m < r := by
  sorry

end relationship_among_mnr_l69_69870


namespace train_length_l69_69747

theorem train_length 
  (V : ℝ → ℝ) (L : ℝ) 
  (length_of_train : ∀ (t : ℝ), t = 8 → V t = L / 8) 
  (pass_platform : ∀ (d t : ℝ), d = L + 273 → t = 20 → V t = d / t) 
  : L = 182 := 
by
  sorry

end train_length_l69_69747


namespace find_value_sum_l69_69126

noncomputable def f : ℝ → ℝ
  := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 3) = f x
axiom value_at_minus_one : f (-1) = 1

theorem find_value_sum :
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry

end find_value_sum_l69_69126


namespace integral_sin2_sin_cos2_cos_l69_69798

open Real

theorem integral_sin2_sin_cos2_cos :
    ∫ x in 0..(π / 2), (sin (sin x))^2 + (cos (cos x))^2 = π / 2 :=
by
  sorry

end integral_sin2_sin_cos2_cos_l69_69798


namespace range_of_a_l69_69155

open Set

variable (a : ℝ)

def P(a : ℝ) : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0

def Q(a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (hP : P a) (hQ : Q a) : a ≤ -2 ∨ a = 1 := sorry

end range_of_a_l69_69155


namespace circumscribed_circle_radius_area_l69_69398

-- Define the primary concepts and assumptions
variables (R : ℝ) (A D E M T : ℝ)
variables (distance_arc_DE : ℝ) (distance_arc_AD : ℝ)
variables (is_triangle_isosceles_acute : Prop) (is_circumcircle : Prop)

-- Define the conditions given in the problem
def conditions : Prop :=
  is_triangle_isosceles_acute ∧
  is_circumcircle ∧
  distance_arc_DE = 5 ∧
  distance_arc_AD = (1 / 3)

-- Define the radius and area of triangle ADE
def radius (R : ℝ) : Prop := R = 6
def area_triangle_ADE (R : ℝ) : ℝ := (35 * real.sqrt 35) / 9

-- The theorem to be proven
theorem circumscribed_circle_radius_area (A D E M T : ℝ) :
  conditions A D (distance_arc_DE) (distance_arc_AD) is_triangle_isosceles_acute is_circumcircle →
  (radius R ∧ area_triangle_ADE R) := 
sorry

end circumscribed_circle_radius_area_l69_69398


namespace sum_of_midpoint_coordinates_l69_69692

theorem sum_of_midpoint_coordinates 
  (x1 y1 z1 x2 y2 z2 : ℝ) 
  (h1 : (x1, y1, z1) = (2, 3, 4)) 
  (h2 : (x2, y2, z2) = (8, 15, 12)) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 + (z1 + z2) / 2 = 22 := 
by
  sorry

end sum_of_midpoint_coordinates_l69_69692


namespace processing_fee_is_correct_l69_69677

-- Definitions based on given conditions
def ticket_cost_per_person : ℝ := 50
def parking_fee : ℝ := 10
def entrance_fee_per_person : ℝ := 5
def total_cost : ℝ := 135

-- Calculate derived values
def total_ticket_cost : ℝ := 2 * ticket_cost_per_person
def total_entrance_fee : ℝ := 2 * entrance_fee_per_person
def cost_without_processing_fee : ℝ := total_ticket_cost + total_entrance_fee + parking_fee
def processing_fee : ℝ := total_cost - cost_without_processing_fee
def processing_fee_percentage : ℝ := (processing_fee / total_ticket_cost) * 100

-- The theorem to prove
theorem processing_fee_is_correct : processing_fee_percentage = 15 := by
  sorry

end processing_fee_is_correct_l69_69677


namespace count_four_digit_palindrome_squares_l69_69033

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69033


namespace no_heptahedron_with_all_faces_quadrilateral_l69_69671

theorem no_heptahedron_with_all_faces_quadrilateral :
  ¬ ∃ (H : Polyhedron), H.faces = 7 ∧ (∀ face ∈ H.faces, face.edges = 4) :=
by
  sorry

end no_heptahedron_with_all_faces_quadrilateral_l69_69671


namespace sum_of_sequence_l69_69923

noncomputable def S (n : ℕ) : ℚ := ∑ k in Finset.range n, (2 : ℚ) * (1 / k.succ - 1 / (k + 2))

theorem sum_of_sequence (n : ℕ) : S n = (2 * n : ℚ) / (n + 1) := by
  sorry

end sum_of_sequence_l69_69923


namespace perimeter_of_semicircular_cubicle_l69_69412

noncomputable def pi_approx : ℝ := 3.14159  -- Define pi as an approximation.

theorem perimeter_of_semicircular_cubicle (r : ℝ) (h_r : r = 14) : 
  (pi_approx * r + 2 * r) ≈ 72 := 
by
  have h1 : pi_approx * r = pi_approx * 14 := by rw [h_r]
  have h2 : 2 * r = 28 := by rw [h_r, mul_two]
  have h3 : pi_approx * 14 ≈ 44.0 := sorry  -- Use appropriate steps or approx for Lean's ~ symbol.
  rw [h1, h2]
  exact sorry -- complete the proof using approximation if necessary.


end perimeter_of_semicircular_cubicle_l69_69412


namespace least_m_mod_1000_l69_69498

def f (m : ℕ) : ℕ :=
  (Nat.digits 5 m).sum

def g (m : ℕ) : ℕ :=
  (Nat.digits 9 (f m)).sum

theorem least_m_mod_1000 :
  ∃ m : ℕ, g(m) ≥ 10
  ∧ ∀ k : ℕ, k < m → g(k) < 10
  ∧ m % 1000 = 123 := by
  sorry

end least_m_mod_1000_l69_69498


namespace fraction_of_ponies_with_horseshoes_l69_69050

noncomputable theory

variables (P H : ℕ) (F : ℚ)
variables (P_nonneg : 0 ≤ P) (H_nonneg : 0 ≤ H)
variables (cond1 : H = P + 4) (cond2 : H + P ≥ 40)
variables (cond3 : ∃ x : ℚ, x = F * P ∧ (2/3) * x = ∃ y : ℚ, y = (2/3) * x ∧  ∃ z, (y = z ∧ z ≠ 0))

theorem fraction_of_ponies_with_horseshoes : F = 1 / 12 := by
  sorry

end fraction_of_ponies_with_horseshoes_l69_69050


namespace alice_favorite_number_l69_69424

/-- Definition of what Alice loves -/
def is_fav_number (n : ℕ) : Prop :=
  (n ≥ 70 ∧ n ≤ 150) ∧
  (n % 13 = 0) ∧
  (n % 3 ≠ 0) ∧
  (Nat.digits 10 n).sum.Prime

/-- Verifies that the number 104 is Alice's favorite number -/
theorem alice_favorite_number : is_fav_number 104 :=
  by
    sorry

end alice_favorite_number_l69_69424


namespace sequence_term_l69_69223

theorem sequence_term (a : ℕ → ℕ) 
  (h1 : a 1 = 2009) 
  (h2 : a 2 = 2011) 
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n + 1) 
  : a 1000 = 2342 := 
by 
  sorry

end sequence_term_l69_69223


namespace acute_triangle_probability_l69_69054

noncomputable def probability_of_acute_triangle (l : ℝ) : ℝ := sorry

theorem acute_triangle_probability :
  probability_of_acute_triangle > 0.081 ∧ probability_of_acute_triangle < 0.083 :=
sorry

end acute_triangle_probability_l69_69054


namespace job_completion_in_time_l69_69814

theorem job_completion_in_time (t_total t_1 w_1 : ℕ) (work_done : ℚ) (h : (t_total = 30) ∧ (t_1 = 6) ∧ (w_1 = 8) ∧ (work_done = 1/3)) :
  ∃ w : ℕ, w = 4 ∧ (t_total - t_1) * w_1 / t_1 * (1 / work_done) / w = 3 :=
by
  sorry

end job_completion_in_time_l69_69814


namespace shaded_area_l69_69415

-- Define the points as per the problem
structure Point where
  x : ℝ
  y : ℝ

@[simp]
def A : Point := ⟨0, 0⟩
@[simp]
def B : Point := ⟨0, 7⟩
@[simp]
def C : Point := ⟨7, 7⟩
@[simp]
def D : Point := ⟨7, 0⟩
@[simp]
def E : Point := ⟨7, 0⟩
@[simp]
def F : Point := ⟨14, 0⟩
@[simp]
def G : Point := ⟨10.5, 7⟩

-- Define function for area of a triangle given three points
def triangle_area (P Q R : Point) : ℝ :=
  0.5 * abs ((P.x - R.x) * (Q.y - P.y) - (P.x - Q.x) * (R.y - P.y))

-- The theorem stating the area of the shaded region
theorem shaded_area : triangle_area D G H - triangle_area D E H = 24.5 := by
  sorry

end shaded_area_l69_69415


namespace triangles_in_pentadecagon_l69_69956

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69956


namespace alex_initial_silk_l69_69423

theorem alex_initial_silk (m_per_dress : ℕ) (m_per_friend : ℕ) (num_friends : ℕ) (num_dresses : ℕ) (initial_silk : ℕ) :
  m_per_dress = 5 ∧ m_per_friend = 20 ∧ num_friends = 5 ∧ num_dresses = 100 ∧ 
  (initial_silk - (num_friends * m_per_friend)) / m_per_dress * m_per_dress = num_dresses * m_per_dress → 
  initial_silk = 600 :=
by
  intros
  sorry

end alex_initial_silk_l69_69423


namespace similar_triangles_find_k_min_l69_69810

def PointOnSegment (A B P : Point) : Prop :=
  ∃ k : ℝ, 0 < k ∧ k < 1 ∧ P = (1 - k) • A + k • B

def AllSatisfied (A B C M N P : Point) : Prop :=
  ∃ k : ℝ, 0 < k ∧ k < 1 ∧
  PointOnSegment A B M ∧ 
  PointOnSegment B C N ∧ 
  PointOnSegment C A P ∧
  PointOnSegment M N R ∧
  PointOnSegment N P S ∧
  PointOnSegment P M T ∧
  (∃ k1 : ℝ, k1 = 1 - k ∧
    (k * (M - A) = MR : ℝ) ∧
    (1 - k * (N - B) = NR : ℝ) ∧
    (1 - k * (P - A) = PR : ℝ)
  )

theorem similar_triangles_find_k_min (A B C M N P R S T : Point) (k : ℝ) :
  AllSatisfied A B C M N P R S T →
  (A - B) = (S - T) → (B - C) = (S - R) → (C - A) = (T - R) →
  ∃ k = 1/2, sorry :=
begin
  sorry
end

end similar_triangles_find_k_min_l69_69810


namespace intersection_of_A_and_B_l69_69554

open Set

def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def B : Set ℝ := {x : ℝ | x^2 - 3*x ≤ 0}

theorem intersection_of_A_and_B : (A ∩ B.to_finset) = {1, 2, 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l69_69554


namespace egg_shape_area_l69_69077

-- Definitions based on conditions
def O_midpoint (A B : Point) (O : Point) : Prop := dist A O = dist O B ∧ dist A B = 4

def circle_with_center_O_radius_2 (O : Point) (E F : Point) : Prop :=
  dist O E = 2 ∧ dist O F = 2 ∧ is_perpendicular_bisector A B E F

def arcs_centered_A_B_radius_4 (A B C D E : Point) : Prop :=
  dist A C = 4 ∧ dist B D = 4 ∧ on_ray A C E ∧ on_ray B D E 

def arc_centered_E_with_radius_DE (D E C : Point) : Prop :=
  dist E D = dist E C 

-- Theorem statement to prove the area of the "egg-shaped" figure
theorem egg_shape_area (A B C D E F O : Point)
  (hO : O_midpoint A B O)
  (hCircle : circle_with_center_O_radius_2 O E F)
  (hArcs : arcs_centered_A_B_radius_4 A B C D E)
  (hArcE : arc_centered_E_with_radius_DE D E C) :
  area_of_egg_shape A F B C D A = (12 - 4 * sqrt 2) * π - 4 := sorry

end egg_shape_area_l69_69077


namespace evaluate_complex_fraction_l69_69819

theorem evaluate_complex_fraction : (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3)))))) = (8 / 21)) :=
by
  sorry

end evaluate_complex_fraction_l69_69819


namespace quadratic_has_two_distinct_real_roots_rectangle_perimeter_quadratic_eq_rectangle_l69_69921

-- Define the quadratic equation conditions
def quadratic_eq (k : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - (2*k + 1)*x + 4*k - 3 = 0

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem quadratic_has_two_distinct_real_roots (k : ℝ) : ∀ Δ : ℝ,
  Δ = (2*k - 3)^2 + 4 → Δ > 0 :=
by {
  intro Δ,
  intro hΔ,
  rw hΔ,
  exact lt_add_of_le_of_pos (by norm_num) (by norm_num)
}

-- Part 2: Perimeter of the rectangle
theorem rectangle_perimeter (k : ℝ) (ab bc : ℝ)
  (h_ab_bc_roots : ab + bc = 2*k + 1 ∧ ab * bc = 4*k - 3)
  (h_ac : (ab^2 + bc^2 = 31)) : ab + bc = 7 → 2*(ab + bc) = 14 :=
by {
  intro h_sum_ab_bc,
  rw h_sum_ab_bc,
  exact eq.refl 14,
}

-- Proof of the second part given additional condition
theorem quadratic_eq_rectangle (k : ℝ) (ab bc : ℝ)
  (h_ab_bc_roots : ab + bc = 2*k + 1 ∧ ab * bc = 4*k - 3)
  (h_ac : (ab^2 + bc^2 = 31)) : 
  ab + bc = 7 →
  2*(ab + bc) = 14 :=
by {
  exact rectangle_perimeter k ab bc h_ab_bc_roots h_ac,
}


end quadratic_has_two_distinct_real_roots_rectangle_perimeter_quadratic_eq_rectangle_l69_69921


namespace triangles_in_pentadecagon_l69_69953

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69953


namespace power_mod_remainder_l69_69361

theorem power_mod_remainder (a b : ℕ) (h1 : a = 3) (h2 : b = 167) :
  (3^167) % 11 = 9 := by
  sorry

end power_mod_remainder_l69_69361


namespace working_light_bulbs_l69_69306

def total_lamps : Nat := 20
def bulbs_per_lamp : Nat := 7
def fraction_burnt_out : ℚ := 1 / 4
def bulbs_burnt_per_lamp : Nat := 2

theorem working_light_bulbs : 
  let total_bulbs := total_lamps * bulbs_per_lamp
  let burnt_out_lamps := (fraction_burnt_out * total_lamps).toNat
  let total_bulbs_burnt_out := burnt_out_lamps * bulbs_burnt_per_lamp
  let working_bulbs := total_bulbs - total_bulbs_burnt_out
  working_bulbs = 130 :=
by
  sorry

end working_light_bulbs_l69_69306


namespace find_BC_l69_69265

theorem find_BC (A B C P Q : Type) [inner_product_space ℝ P]
  (hBAC : ∠BAC = 90) (hP : midpoint ℝ A B P) (hQ : midpoint ℝ A C Q)
  (hBP : dist B P = 25) (hQC : dist Q C = 15) : dist B C = 2 * sqrt 170 :=
by sorry

end find_BC_l69_69265


namespace seven_circles_equality_l69_69678

theorem seven_circles_equality
  (fixed_circle : Circle)
  (small_circles : Fin 6 → Circle)
  (A : Fin 6 → Point)
  (h1 : ∀ i, Tangent small_circles[i] fixed_circle)
  (h2 : ∀ i, Tangent small_circles[i] small_circles[(i + 1) % 6])
  (h3 : ∀ i, OnCircle A[i] fixed_circle)
  : A[0]dist A[1] * A[2]dist A[3] * A[4]dist A[5] = A[1]dist A[2] * A[3]dist A[4] * A[5]dist A[0] :=
sorry

end seven_circles_equality_l69_69678


namespace complement_of_A_relative_to_I_l69_69190

def I : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | x^2 < 3}

def complement_I_A : Set ℤ := {x ∈ I | x ∉ A}

theorem complement_of_A_relative_to_I :
  complement_I_A = {-2, 2} := by
  sorry

end complement_of_A_relative_to_I_l69_69190


namespace sequence_sum_100_terms_l69_69524

theorem sequence_sum_100_terms :
  (∀ n : ℕ, a n = 2 * n - 1) →
  (∀ n : ℕ, b n = 2 ^ n) →
  (∑ i in finset.range 93, a (i + 1) + ∑ i in finset.range 7, b (i + 1)) = 8903 :=
by
  intros ha hb
  sorry

end sequence_sum_100_terms_l69_69524


namespace probability_A3_given_white_l69_69726

noncomputable def P (A : Type) (event : set A) (p : A → ℝ) : ℝ :=
  ∑ x in event, p x

def urns := {A_1, A_2, A_3, A_4}
def balls := {white, black}

def composition : urns → balls → ℕ
| A_1 white := 3 | A_1 black := 4
| A_2 white := 2 | A_2 black := 8
| A_3 white := 6 | A_3 black := 1
| A_4 white := 4 | A_4 black := 3

def num_urns : urns → ℕ
| A_1 := 6 | A_2 := 3 | A_3 := 2 | A_4 := 1

def P_A (A : urns) : ℝ :=
  (num_urns A) / 12

def P_B_given_A (A : urns) : ℝ :=
  (composition A white) / (composition A white + composition A black)

def P_B : ℝ :=
  P_A A_1 * P_B_given_A A_1 +
  P_A A_2 * P_B_given_A A_2 +
  P_A A_3 * P_B_given_A A_3 +
  P_A A_4 * P_B_given_A A_4

def P_A3_given_B : ℝ :=
  (P_A A_3 * P_B_given_A A_3) / P_B

theorem probability_A3_given_white :
  P_A3_given_B = 30 / 73 :=
sorry

end probability_A3_given_white_l69_69726


namespace pentadecagon_triangle_count_l69_69948

theorem pentadecagon_triangle_count :
  ∑ k in finset.range 15, if k = 3 then nat.choose 15 3 else 0 = 455 :=
by {
  sorry
}

end pentadecagon_triangle_count_l69_69948


namespace prove_p_or_q_l69_69188

variable {a : ℝ}
def p : Prop := ∀ x : ℝ, x^2 + a * x + a^2 ≥ 0
def q : Prop := ∃ x0 : ℕ, x0 > 0 ∧ 2 * (x0 : ℝ)^2 - 1 ≤ 0

theorem prove_p_or_q : p ∨ q := by
  sorry

end prove_p_or_q_l69_69188


namespace match_first_digit_l69_69513

-- Definition of the problem conditions
def digit_set : set ℕ := {1, 2, 3}

def is_four_digit_number (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, a ∈ digit_set ∧ b ∈ digit_set ∧ c ∈ digit_set ∧ d ∈ digit_set ∧ n = 1000 * a + 100 * b + 10 * c + d

def unique_assignment_to_74_numbers (assign : ℕ → ℕ) : Prop :=
  ∀ a b c d : ℕ, 
  a ∈ digit_set ∧ b ∈ digit_set ∧ c ∈ digit_set ∧ d ∈ digit_set ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  assign (1000 * a + 100 * b + 10 * c + d) ≠ assign (1000 * b + 100 * c + 10 * d + a)

def specific_assignment (assign : ℕ → ℕ) : Prop :=
  assign 1111 = 1 ∧ assign 2222 = 2 ∧ assign 3333 = 3 ∧ assign 1222 = 1

-- The statement to be proved
theorem match_first_digit (assign : ℕ → ℕ) (h_74 : unique_assignment_to_74_numbers assign) (h_specific : specific_assignment assign) :
  ∀ n : ℕ, is_four_digit_number n → assign n = n / 1000 :=
sorry

end match_first_digit_l69_69513


namespace sale_in_third_month_l69_69769

theorem sale_in_third_month (sale1 sale2 sale4 sale5 sale6 avg_sale : ℝ) (n_months : ℝ) (sale3 : ℝ):
  sale1 = 5400 →
  sale2 = 9000 →
  sale4 = 7200 →
  sale5 = 4500 →
  sale6 = 1200 →
  avg_sale = 5600 →
  n_months = 6 →
  (n_months * avg_sale) - (sale1 + sale2 + sale4 + sale5 + sale6) = sale3 →
  sale3 = 6300 :=
by
  intros
  sorry

end sale_in_third_month_l69_69769


namespace exists_convex_body_projection_l69_69246

theorem exists_convex_body_projection :
  ∃ (C : set (ℝ^3)), 
    convex C ∧ orthogonal_projection C (λ (x y z : ℝ), (y, z)) = set_of (λ (y z : ℝ), y^2 + z^2 ≤ 1) ∧
    orthogonal_projection C (λ (x y z : ℝ), (x, z)) = set_of (λ (x z : ℝ), x^2 + z^2 ≤ 1) ∧
    orthogonal_projection C (λ (x y z : ℝ), (x, y)) = set_of (λ (x y : ℝ), x^2 + y^2 ≤ 1) ∧
    (C ≠ set_of (λ (x y z : ℝ), x^2 + y^2 + z^2 ≤ 1)) :=
begin
  sorry
end

end exists_convex_body_projection_l69_69246


namespace five_digit_numbers_equality_l69_69740

theorem five_digit_numbers_equality :
  let count_not_divisible_by_5 := 9 * 8 * 10^3
  let count_no_first_second_5 := 8 * 9 * 10^3
  count_not_divisible_by_5 = count_no_first_second_5 :=
by
  -- defining the counts
  let count_not_divisible_by_5 := 9 * 8 * 10^3
  let count_no_first_second_5 := 8 * 9 * 10^3

  -- assertion
  show count_not_divisible_by_5 = count_no_first_second_5 from sorry

end five_digit_numbers_equality_l69_69740


namespace hyperbola_equation_l69_69915

theorem hyperbola_equation (a b : Real) (ha : a > 0) (hb : b > 0) 
  (asymptote : b / a = sqrt 3) 
  (focus_hyperbola : ∃ c : Real, c = 4 ∧ c^2 = a^2 + b^2)
  : (a^2 = 4 ∧ b^2 = 12) → (∀ x y : Real, (x^2 / 4 - y^2 / 12 = 1) ↔ (x^2 / a^2 - y^2 / b^2 = 1)) :=
by
  intros h
  obtain ⟨ha2, hb2⟩ := h
  field_simp
  congr
  exact ha2
  exact hb2

#check hyperbola_equation

end hyperbola_equation_l69_69915


namespace length_PC_l69_69217

-- Definitions of lengths for the sides
def AB : ℝ := 10
def BC : ℝ := 9
def CA : ℝ := 7

-- Defining the similarity condition
structure SimilarTriangles (PAB PCA : Type) :=
(similarity : ∀ P A B C, ∀ (h1: PAB P A B = PCA P C A), True)

-- Proving that the length of PC is 31.5.
theorem length_PC {P A B C : Type} (h : SimilarTriangles PAB PCA) :
  PC = 31.5 :=
sorry

end length_PC_l69_69217


namespace quadratic_inequality_l69_69825

theorem quadratic_inequality (c : ℝ) (h₁ : 0 < c) (h₂ : c < 16): ∃ x : ℝ, x^2 - 8 * x + c < 0 :=
sorry

end quadratic_inequality_l69_69825


namespace cone_csa_l69_69748

theorem cone_csa (r l : ℝ) (h_r : r = 8) (h_l : l = 18) : 
  (Real.pi * r * l) = 144 * Real.pi :=
by 
  rw [h_r, h_l]
  norm_num
  sorry

end cone_csa_l69_69748


namespace apples_shared_apples_shared_seven_l69_69095

theorem apples_shared (Ci Cr Cs : ℕ) (h1 : Ci = 20) (h2 : Cr = 13) : Cs = Ci - Cr := by
  rw [h1, h2]
  exact rfl

-- To specifically conclude the amount shared equals 7
theorem apples_shared_seven (h1 : 20 - 13 = 7) : ∀ (Ci Cr : ℕ), Ci = 20 → Cr = 13 → Ci - Cr = 7 := by
  intros Ci Cr hCi hCr
  rw [hCi, hCr]
  exact h1

end apples_shared_apples_shared_seven_l69_69095


namespace difference_of_squares_l69_69735

def largestNumber : ℕ := 98765421
def smallestNumber : ℕ := 12456789

theorem difference_of_squares : (largestNumber^2 - smallestNumber^2) = 9599477756293120 :=
by
  let largestSquare := 98765421^2
  let smallestSquare := 12456789^2
  have : largestSquare = 9754610577890641 := sorry
  have : smallestSquare = 155132821597521 := sorry
  show (largestNumber^2 - smallestNumber^2) = 9599477756293120 from sorry
  sorry

end difference_of_squares_l69_69735


namespace find_smallest_d_l69_69318

noncomputable def smallest_possible_d (c d : ℕ) : ℕ :=
  if c - d = 8 ∧ Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16 then d else 0

-- Proving the smallest possible value of d given the conditions
theorem find_smallest_d :
  ∀ c d : ℕ, (0 < c) → (0 < d) → (c - d = 8) → 
  Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16 → d = 4 :=
by
  sorry

end find_smallest_d_l69_69318


namespace find_u_correct_l69_69346

open Matrix

noncomputable def u : ℚ :=
  let a : Matrix (Fin 3) (Fin 1) ℚ := ![![2], ![3], ![-1]]
  let b : Matrix (Fin 3) (Fin 1) ℚ := ![![-1], ![4], ![2]]
  let d : Matrix (Fin 3) (Fin 1) ℚ := ![![1], ![2], ![1]]
  let c := cross_product a b
  9 / 115

theorem find_u_correct :
  let a : Matrix (Fin 3) (Fin 1) ℚ := ![![2], ![3], ![-1]]
  let b : Matrix (Fin 3) (Fin 1) ℚ := ![![-1], ![4], ![2]]
  let d : Matrix (Fin 3) (Fin 1) ℚ := ![![1], ![2], ![1]]
  let c := cross_product a b
  ∃ (s t : ℚ), d = s • a + t • b + u • c ∧ u = 9 / 115 :=
by {
  let a : Matrix (Fin 3) (Fin 1) ℚ := ![![2], ![3], ![-1]]
  let b : Matrix (Fin 3) (Fin 1) ℚ := ![![-1], ![4], ![2]]
  let d : Matrix (Fin 3) (Fin 1) ℚ := ![![1], ![2], ![1]]
  let c := cross_product a b
  existsi (0 : ℚ),
  existsi (0 : ℚ),
  split,
  sorry,
}

end find_u_correct_l69_69346


namespace simplified_expression_l69_69454

theorem simplified_expression : 
  ((1/3)⁻¹ + sqrt 12 - abs (sqrt 3 - 2) - (Real.pi - 2023) ^ 0) = 3 * sqrt 3 := 
by
  sorry

end simplified_expression_l69_69454


namespace locus_of_P_and_slope_l69_69601

/-
Three points A, B, C are given:
- A: (0, 4/3)
- B: (-1, 0)
- C: (1, 0)

The distance from point P to line BC (y = 0) is the geometric mean of the distances from P to lines AB and AC.
We need to prove two things:
1. The locus of point P is given by the equations: 
   2x^2 + 2y^2 + 3y - 2 = 0 (circle) 
   and 8x^2 - 17y^2 + 12y - 8 = 0 (hyperbola).

2. For any line L passing through the incenter of triangle ABC, if it intersects the locus of P at exactly 3 points,
   the possible slopes k are: 
   {0, ±1/2, ±2√34/17, ±√2/2}.
-/

noncomputable def equations_of_locus (P : ℝ × ℝ) : Prop :=
  let x := P.1 in
  let y := P.2 in
  (2 * x^2 + 2 * y^2 + 3 * y - 2 = 0) ∨
  (8 * x^2 - 17 * y^2 + 12 * y - 8 = 0)

theorem locus_of_P_and_slope {D : ℝ × ℝ} (k : ℝ) :
  D = (0, 1/2) →
  (∃ P : ℝ × ℝ, equations_of_locus P) →
  ∃ PS : set (ℝ × ℝ), PS = {P | equations_of_locus P} ∧
  (∀ L : set (ℝ × ℝ), 
    (∃ P1 P2 P3 : ℝ × ℝ, 
      (L = { P : ℝ × ℝ | ∃ y, y = k * (P.1) + 1/2 } ∧ P1 ∈ PS ∧ P2 ∈ PS ∧ P3 ∈ PS)) →
      k ∈ ({0, 1/2, -1/2, 2 * real.sqrt 34 / 17, -2 * real.sqrt 34 / 17, real.sqrt 2 / 2, -real.sqrt 2 / 2} : set ℝ))
  :=
sorry

end locus_of_P_and_slope_l69_69601


namespace billy_distance_from_starting_point_l69_69080

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.fst - p2.fst) ^ 2 + (p1.snd - p2.snd) ^ 2

noncomputable def cos15 : ℝ := Mathlib.Real.cos (15 * Mathlib.Real.pi / 180)
noncomputable def sin15 : ℝ := Mathlib.Real.sin (15 * Mathlib.Real.pi / 180)

noncomputable def billy_displacement : ℝ :=
  let b := (6, 0)
  let c := (6 + 4 / Mathlib.Real.sqrt 2, 4 / Mathlib.Real.sqrt 2)
  let d := (c.fst + 8 * cos15, c.snd + 8 * sin15)
  Mathlib.Real.sqrt (distance (0, 0) d)

theorem billy_distance_from_starting_point :
  abs (billy_displacement - 17.26) < 0.01 :=
sorry

end billy_distance_from_starting_point_l69_69080


namespace sum_reciprocal_b_l69_69544

noncomputable def sequence_a (n : ℕ) : ℝ := 3^n

noncomputable def sequence_b (n : ℕ) : ℝ := (n + 1) * Real.logBase 3 (sequence_a n)

noncomputable def sequence_reciprocal_b (n : ℕ) : ℝ := 1 / (sequence_b n)

noncomputable def sum_first_n_terms (n : ℕ) : ℝ := ∑ k in Finset.range n, sequence_reciprocal_b (k + 1)

theorem sum_reciprocal_b (n : ℕ) : sum_first_n_terms n = n / (n + 1) := by
  sorry

end sum_reciprocal_b_l69_69544


namespace abs_eq_non_pos_2x_plus_4_l69_69365

-- Condition: |2x + 4| = 0
-- Conclusion: x = -2
theorem abs_eq_non_pos_2x_plus_4 (x : ℝ) : (|2 * x + 4| = 0) → x = -2 :=
by
  intro h
  -- Here lies the proof, but we use sorry to indicate the unchecked part.
  sorry

end abs_eq_non_pos_2x_plus_4_l69_69365


namespace trapezoid_problem_l69_69351

variables {A B C D O P : Type*} [MetricSpace A]
variables [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace O] [MetricSpace P]
variables (AB CD BD AC AD : ℝ)
variables (OP : ℝ := 13) (BC : ℝ := 53) (BD_length : ℝ := 26)

-- Defining the conditions
def is_trapezoid (A B C D : Type*) : Prop :=
  parallel A B C D ∧ length B C = length C D ∧
  length B C = BC ∧ length C D = BC ∧ OP = 13

def right_triangle (A D B : Type*) : Prop := 
  ⟦ ⟨A, D, B⟩ ⟧ ∈ is_right (angle A D B) 

noncomputable def midpoint_condition (BD : Type*) : Prop :=
  let P := midpoint (B, D) in
  norm (P, O) = 13 

noncomputable def pythagoras (A B D : Type*) (AD : ℝ) : Prop :=
  norm (A, B) ^ 2 = AD ^ 2 + BD_length ^ 2 

noncomputable def final_condition (A D B : Type*) (m n : ℕ) : Prop :=
  A = A ∧ B = B ∧ D = D ∧ m = 9 ∧ n = 53 ∧ m + n = 62  

theorem trapezoid_problem {A B C D O P : Type*}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace O] [MetricSpace P]
  (h₁ : is_trapezoid A B C D)
  (h₂ : right_triangle A D B)
  (h₃ : midpoint_condition BD)
  (h₄ : ∃ AD : ℝ, pythagoras A B D AD)
  : final_condition A D B 9 53 := 
sorry

end trapezoid_problem_l69_69351


namespace determine_num_a_l69_69099

theorem determine_num_a :
  let solutions_a := { a : ℕ | ∃ (x : ℕ), x > 0 ∧ 3x > 4x - 4 ∧ 4x - a > -8 ∧ 
                     ∀ y, y > 0 ∧ 3y > 4y - 4 ∧ 4y - a > -8 → y = 3 };
  solutions_a.card = 4 :=
begin
  sorry
end

end determine_num_a_l69_69099


namespace min_pos_sum_ai_aj_l69_69475

-- Define the sequence a_i such that each a_i is 1 or -1
def a : ℕ → ℤ
:= λ i, ite (i ≤ 50) (if i % 2 = 0 then 1 else -1) 0

-- Proposition for the minimum positive value of S
theorem min_pos_sum_ai_aj : 
  (∀ i, 1 ≤ i ∧ i ≤ 50 → (a i = 1 ∨ a i = -1)) →
  ∃ S, S = ∑ i in finset.range 50, ∑ j in finset.Ico (i + 1) 51, a i * a j 
    ∧ S = 7 :=
by
  sorry

end min_pos_sum_ai_aj_l69_69475


namespace hyperbola_center_focus_vertex_sum_l69_69231

theorem hyperbola_center_focus_vertex_sum : 
  ∃ h k a b : ℝ, 
  (h, k) = (-3, 1) ∧
  (h + sqrt 41, k) = (-3 + sqrt 41, 1) ∧ 
  (h - 4, k) = (-7, 1) ∧ 
  h + k + a + b = 7 :=
by {
  use [-3, 1, 4, 5],
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  exact rfl,
}

end hyperbola_center_focus_vertex_sum_l69_69231


namespace amanda_candy_bars_kept_l69_69429

noncomputable def amanda_initial_candy_bars : ℕ := 7
noncomputable def candy_bars_given_first_time : ℕ := 3
noncomputable def additional_candy_bars : ℕ := 30
noncomputable def multiplier : ℕ := 4

theorem amanda_candy_bars_kept :
  let remaining_after_first_give := amanda_initial_candy_bars - candy_bars_given_first_time in
  let total_after_buying_more := remaining_after_first_give + additional_candy_bars in
  let candy_bars_given_second_time := multiplier * candy_bars_given_first_time in
  total_after_buying_more - candy_bars_given_second_time = 22 :=
by
  sorry

end amanda_candy_bars_kept_l69_69429


namespace intersecting_lines_parallel_l69_69381

theorem intersecting_lines_parallel 
  (A B C A' B' C' : Type) [add_comm_group A] [module ℝ A] [affine_space A]
  (BC CA AB B'C' C'A' A'B' : A → A → Prop)
  (lines_intersect' : 
    parallel BC (λ _, A') ∧
    parallel CA (λ _, B') ∧
    parallel AB (λ _, C') ∧
    (∃ P : A, 
      BC C B P ∧ 
      CA A C P ∧ 
      AB B A P))
  : (∃ Q : A, 
       parallel B'C' (λ _, A) ∧
       parallel C'A' (λ _, B) ∧
       parallel A'B' (λ _, C) ∧
       B'C' C' B Q ∧ 
       C'A' A' C Q ∧ 
       A'B' B' A Q) :=
sorry

end intersecting_lines_parallel_l69_69381


namespace correct_propositions_l69_69179

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin(2 * x + Real.pi / 3)

def proposition_1 : Prop := ∀ x : ℝ, f(x) = 4 * Real.cos(2 * x - Real.pi / 6)

def proposition_2 : Prop := Real.periodic f (2 * Real.pi)

def symmetric_point (f : ℝ → ℝ) (a : ℝ) (b : ℝ) : Prop :=
∀ x, f(a + x) = f(a - x) → f x = b

def symmetric_line (f : ℝ → ℝ) (l : ℝ) : Prop :=
∀ x, f l = f x → f(l - x) = f(l + x)

def proposition_3 : Prop := symmetric_point f (-Real.pi / 6) 0

def proposition_4 : Prop := symmetric_line f (-Real.pi / 6)

theorem correct_propositions :
  proposition_1 ∧ ¬proposition_2 ∧ proposition_3 ∧ ¬proposition_4 :=
by
  sorry

end correct_propositions_l69_69179


namespace cos_value_l69_69159

-- Given condition
def cos_condition (θ : ℝ) : Prop := 
  cos (π / 6 - θ) = 2 * real.sqrt 2 / 3

-- Target conclusion
theorem cos_value (θ : ℝ) (h : cos_condition θ) : 
  cos (π / 3 + θ) = 1 / 3 ∨ cos (π / 3 + θ) = -1 / 3 := 
sorry

end cos_value_l69_69159


namespace iron_sphere_radius_l69_69780

theorem iron_sphere_radius (r : ℝ) : 
  (∃ (cone_radius cone_height cone_slant_height : ℝ), 
    cone_radius = 3 ∧ cone_slant_height = 5 ∧ 
    cone_height = real.sqrt (cone_slant_height^2 - cone_radius^2) ∧ 
    (1/3) * π * cone_radius^2 * cone_height = (4/3) * π * r^3) → 
  r = real.cbrt 9 :=
by
  sorry

end iron_sphere_radius_l69_69780


namespace monic_quadratic_with_root_2_minus_3i_l69_69851

theorem monic_quadratic_with_root_2_minus_3i :
  ∃ P : ℝ[X], P.monic ∧ (P.coeff 2 = 1)
    ∧ (P.coeff 1 = -4)
    ∧ (P.coeff 0 = 13)
    ∧ eval (2 - 3 * I) P = 0 := sorry

end monic_quadratic_with_root_2_minus_3i_l69_69851


namespace rectangle_area_inscribed_circle_l69_69396

theorem rectangle_area_inscribed_circle {r w l : ℕ} (h1 : r = 7) (h2 : w = 2 * r) (h3 : l = 3 * w) : l * w = 588 :=
by 
  -- The proof details are omitted as per instructions.
  sorry

end rectangle_area_inscribed_circle_l69_69396


namespace arithmetic_sequence_and_sum_max_value_l69_69522

variable {α : Type*} (a b : ℕ → ℕ) (s T : ℕ → ℕ)
variable [LinearOrderedSemiring α] [Nontrivial α]

theorem arithmetic_sequence_and_sum_max_value
  (h_pos : ∀ n, 0 < a n)
  (h_sum : ∀ n, s n = (a n + 1) ^ 2 / 4)
  (h_bn_def : ∀ n, b n = 10 - a n) :
  (∃ d, ∀ n, a (n + 1) - a n = d) ∧ (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n = (-n^2 + 10*n)) ∧ (∃ m, ∀ n, T n ≤ m ∧ m = 25) :=
by 
  sorry

end arithmetic_sequence_and_sum_max_value_l69_69522


namespace obtuse_triangle_side_range_l69_69533

theorem obtuse_triangle_side_range (a : ℝ) :
  (a > 0) ∧
  ((a < 3 ∧ a > -1) ∧ 
  (2 * a + 1 > a + 2) ∧ 
  (a > 1)) → 1 < a ∧ a < 3 := 
by
  sorry

end obtuse_triangle_side_range_l69_69533


namespace third_oldest_is_Bernice_l69_69078

noncomputable def Adyant_is_older_than_Bernice (Adyant Bernice : ℕ) (h1 : Adyant > Bernice) : Prop := h1
noncomputable def Dara_is_youngest (Dara : ℕ) (Adyant Bernice Cici Ellis : ℕ) (h2 : Dara < Adyant ∧ Dara < Bernice ∧ Dara < Cici ∧ Dara < Ellis) : Prop := h2
noncomputable def Bernice_is_older_than_Ellis (Bernice Ellis : ℕ) (h3 : Bernice > Ellis) : Prop := h3
noncomputable def Bernice_is_younger_than_Cici (Bernice Cici : ℕ) (h4 : Bernice < Cici) : Prop := h4
noncomputable def Cici_is_not_the_oldest (Adyant Cici : ℕ) (h5 : Cici < Adyant) : Prop := h5

theorem third_oldest_is_Bernice (Adyant Bernice Cici Dara Ellis : ℕ)
    (h1 : Adyant > Bernice)
    (h2 : Dara < Adyant ∧ Dara < Bernice ∧ Dara < Cici ∧ Dara < Ellis)
    (h3 : Bernice > Ellis)
    (h4 : Bernice < Cici)
    (h5 : Cici < Adyant) :
    ∃ (students : List ℕ), students = [Adyant, Cici, Bernice, Ellis, Dara] ∧ students.nth 2 = some Bernice :=
by
  sorry

end third_oldest_is_Bernice_l69_69078


namespace committee_count_l69_69449

theorem committee_count (departments : List (String × (ℕ × ℕ)))
  (H1 : ∀ d ∈ departments, d.2.1 = 3 ∧ d.2.2 = 3)
  (H2 : length departments = 3)
  (H3 : ∃ a ∈ departments, ∃ b ∈ departments, ∃ c ∈ departments,
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    (a.2.1 + b.2.1 + c.2.1 = 4 ∧ a.2.2 + b.2.2 + c.2.2 = 4) ∧
    (a.2.1 = 3 ∨ b.2.1 = 3 ∨ c.2.1 = 3) ∧
    (a.2.2 = 3 ∨ b.2.2 = 3 ∨ c.2.2 = 3)) :
  ∃ committees_count : ℕ,
    committees_count = 4374 :=
by
  sorry

end committee_count_l69_69449


namespace expected_value_ten_sided_die_l69_69061

noncomputable def expected_value (die_faces : Finset ℕ) : ℚ :=
  (∑ i in die_faces, (i : ℚ)) / die_faces.card

theorem expected_value_ten_sided_die :
  expected_value (Finset.range 10 \ {0}) = 5.5 := by
  sorry

end expected_value_ten_sided_die_l69_69061


namespace find_minimum_angle_l69_69494

-- Define condition for B
def B := π / 6

-- Define the cosine and sine values for B
def cos_B := Real.cos B = sqrt 3 / 2
def sin_B := Real.sin B = 1 / 2

-- The main statement to prove
theorem find_minimum_angle (A : ℝ) (hB : B = π / 6) (hcos : Real.cos B = sqrt 3 / 2) (hsin : Real.sin B = 1 / 2) :
  (∃ k : ℤ, A = 2 * (π + k * 2 * π) - π / 3) ↔ A = 5 * π / 3 :=
by 
  sorry

end find_minimum_angle_l69_69494


namespace cupcakes_sold_l69_69507

theorem cupcakes_sold (initial_made sold additional final : ℕ) (h1 : initial_made = 42) (h2 : additional = 39) (h3 : final = 59) :
  (initial_made - sold + additional = final) -> sold = 22 :=
by
  intro h
  rw [h1, h2, h3] at h
  sorry

end cupcakes_sold_l69_69507


namespace list_price_proof_l69_69071

theorem list_price_proof (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  sorry

end list_price_proof_l69_69071


namespace distance_center_to_plane_l69_69154

noncomputable def sphere_center_to_plane_distance 
  (volume : ℝ) (AB AC : ℝ) (angleACB : ℝ) : ℝ :=
  let R := (3 * volume / 4 / Real.pi)^(1 / 3);
  let circumradius := AB / (2 * Real.sin (angleACB / 2));
  Real.sqrt (R^2 - circumradius^2)

theorem distance_center_to_plane 
  (volume : ℝ) (AB : ℝ) (angleACB : ℝ)
  (h_volume : volume = 500 * Real.pi / 3)
  (h_AB : AB = 4 * Real.sqrt 3)
  (h_angleACB : angleACB = Real.pi / 3) :
  sphere_center_to_plane_distance volume AB angleACB = 3 :=
by
  sorry

end distance_center_to_plane_l69_69154


namespace ellipse_equation_l69_69166

-- Defining the conditions
def center_origin (E : Type) : Prop :=
  ∃ (h: E → ℝ × ℝ), (h • 0 = (0, 0))

def foci_on_x_axis (E : Type) : Prop :=
  ∀ (F G : ℝ × ℝ), F.2 = 0 ∧ G.2 = 0

def min_distance_to_focus (E : Type) (d : ℝ) : Prop :=
  ∃ (P: ℝ × ℝ), P ≠ (0, 0) ∧ d = 2 * real.sqrt 2 - 2

def eccentricity (E : Type) (e : ℝ) : Prop :=
  e = (real.sqrt 2) / 2

-- Required to find the equation of the given ellipse.
theorem ellipse_equation (E : Type) 
  (h1 : center_origin E)
  (h2 : foci_on_x_axis E)
  (h3 : min_distance_to_focus E (2 * real.sqrt 2 - 2))
  (h4 : eccentricity E ((real.sqrt 2) / 2)) :
  ∃ a b : ℝ, a = 2 * real.sqrt 2 ∧ b = 2 ∧ 
  (∀ x y : ℝ, (x^2 / 8) + (y^2 / 4) = 1) :=
sorry

end ellipse_equation_l69_69166


namespace range_y_l69_69088

noncomputable def y (x : ℝ) : ℝ := |x + 5| - |x - 3|

theorem range_y : set.range (y) = set.Iic (14) :=
sorry

end range_y_l69_69088


namespace probability_red_or_blue_l69_69750

theorem probability_red_or_blue (total_marbles : ℕ) (prob_white prob_green : ℚ)
(h_total : total_marbles = 90)
(h_prob_white : prob_white = 1/3)
(h_prob_green : prob_green = 1/5) :
  let prob_red_or_blue := 1 - (prob_white + prob_green) in
  prob_red_or_blue = 7/15 :=
by
  sorry

end probability_red_or_blue_l69_69750


namespace num_four_digit_palindromic_squares_l69_69021

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69021


namespace area_increase_by_50_percent_radius_l69_69583

theorem area_increase_by_50_percent_radius (r : ℝ) (π_pos : 0 < Real.pi) : 
  let A := (Real.pi * r^2) in
  let new_r := 1.5 * r in
  let A_new := Real.pi * new_r^2 in
  let percentage_increase := ((A_new - A) / A) * 100 in
  percentage_increase = 125 := by sorry

end area_increase_by_50_percent_radius_l69_69583


namespace smallest_x_for_g_eq_g1536_l69_69465

noncomputable def g : ℝ → ℝ :=
λ x, if 2 ≤ x ∧ x ≤ 4 then 2 - |x - 3| else
if x > 4 then (4:ℝ) * g (x / 4) else
if x < 2 then g (x * 4) / (4:ℝ) else 0

theorem smallest_x_for_g_eq_g1536 : 
  ∃ x : ℝ, (g x = g 1536) ∧ (∀ y : ℝ, (g y = g 1536) → x ≤ y) :=
by {
  use 384,
  split,
  sorry,  -- Proof that g(384) = g(1536)
  intro y,
  intros yg1536,
  sorry  -- Proof that x is the smallest
}

end smallest_x_for_g_eq_g1536_l69_69465


namespace cut_rectangles_l69_69511

theorem cut_rectangles (N : ℕ) (a b : ℕ) (ha : a < b) :
  ∃ (square_parts rectangle_parts : Finset (ℕ × ℕ)), 
    (square_parts.card = N ∧ 
     rectangle_parts.card = N ∧ 
     (∃ (s : ℕ), s = a ∧ 
      ∀ (p ∈ square_parts), p = ⟨s, s⟩) ∧ 
     (∃ (w h : ℕ), w = b - a ∧ h = b ∧ 
      ∀ (q ∈ rectangle_parts), q = ⟨w, h⟩)) := sorry

end cut_rectangles_l69_69511


namespace identify_counterfeit_coin_correct_l69_69998

noncomputable def identify_counterfeit_coin (coins : Fin 8 → ℝ) : ℕ :=
  sorry

theorem identify_counterfeit_coin_correct (coins : Fin 8 → ℝ) (h_fake : 
  ∃ i : Fin 8, ∀ j : Fin 8, j ≠ i → coins i > coins j) : 
  ∃ i : Fin 8, identify_counterfeit_coin coins = i ∧ ∀ j : Fin 8, j ≠ i → coins i > coins j :=
by
  sorry

end identify_counterfeit_coin_correct_l69_69998


namespace intersection_two_points_l69_69469

noncomputable def number_of_intersections 
  (m : ℝ) 
  (circle_eq : ℝ → ℝ → Prop := fun x y => x^2 + y^2 + 2x - 6y - 15 = 0) 
  (line_eq : ℝ → ℝ → ℝ → Prop := fun x y m => (1 + 3 * m) * x + (3 - 2 * m) * y + 4 * m - 17 = 0) 
  : ℕ :=
  if ∀ (m : ℝ), ∃ (x y : ℝ), circle_eq x y ∧ line_eq x y m then 2 else 0

theorem intersection_two_points (m : ℝ) :
  number_of_intersections m = 2 :=
sorry

end intersection_two_points_l69_69469


namespace symmetric_projections_l69_69148

variables (A B C S : Type) [PlaneGeometry A B C S]
variables (πb πc πb' πc' : Plane)
variable (BS_id : BSC Bisector)
variables (α β : dihedral_angle)

-- Condition: Trihedral angle SABC with given angles
def trihedral_angle_condition (S A B C : Type) [PlaneGeometry A B C S] : Prop :=
  dihedral_angle_value (angle A S B) = 90 ∧ dihedral_angle_value (angle A S C) = 90

-- Condition: Planes passing through edges SB and SC
def planes_through_edges (S B C : Type) [PlaneGeometry S B C] (πb πc : Plane) : Prop :=
  πb.contains_edge SB ∧ πc.contains_edge SC

-- Condition: Symmetric planes
def symmetric_planes (πb πc πb' πc' : Plane) (BS_id : Plane) : Prop :=
  πb' = reflect_about_bisector_plane πb BS_id ∧ πc' = reflect_about_bisector_plane πc BS_id

-- Main theorem statement
theorem symmetric_projections
  (h1 : trihedral_angle_condition S A B C)
  (h2 : planes_through_edges S B C πb πc)
  (h3 : symmetric_planes πb πc πb' πc' BS_id)
  : symmetric_projections_of_intersections BSC (intersection_lines πb πc) (intersection_lines πb' πc') :=
sorry

end symmetric_projections_l69_69148


namespace suji_age_problem_l69_69668

theorem suji_age_problem (x : ℕ) 
  (h1 : 5 * x + 6 = 13 * (4 * x + 6) / 11)
  (h2 : 11 * (4 * x + 6) = 9 * (3 * x + 6)) :
  4 * x = 16 :=
by
  sorry

end suji_age_problem_l69_69668


namespace inscribed_circle_radius_l69_69597

theorem inscribed_circle_radius (b h : ℝ) (hb : b = 10) (hh : h = 12) : 
  let r := (10 : ℝ) / 3 in
  r = (60 / 18 : ℝ) :=
by sorry

end inscribed_circle_radius_l69_69597


namespace solution_set_inequality_l69_69341

theorem solution_set_inequality (x : ℝ) : 
  abs(x - 1) + abs(x + 2) ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 :=
by
  sorry

end solution_set_inequality_l69_69341


namespace num_triangles_pentadecagon_l69_69989

/--
  The number of triangles that can be formed using the vertices of a regular pentadecagon
  (a 15-sided polygon where no three vertices are collinear) is 455.
-/
theorem num_triangles_pentadecagon : ∀ (n : ℕ), n = 15 → ∃ (num_triangles : ℕ), num_triangles = Nat.choose n 3 ∧ num_triangles = 455 :=
by
  intros n hn
  use Nat.choose n 3
  split
  · rfl
  · sorry

end num_triangles_pentadecagon_l69_69989


namespace prism_width_l69_69410

theorem prism_width (l h d : ℝ) (hl : l = 5) (hh : h = 15) (hd : d = 17) : 
  ∃ w, w = real.sqrt 39 :=
by
  -- Define the equation for the diagonal of the rectangular prism
  let diag_eq := real.sqrt (l^2 + w^2 + h^2)
  -- Use the given conditions
  have hl_eq : l = 5 := hl
  have hh_eq : h = 15 := hh
  have hd_eq : d = 17 := hd
  sorry

end prism_width_l69_69410


namespace pentadecagon_triangle_count_l69_69942

theorem pentadecagon_triangle_count :
  ∑ k in finset.range 15, if k = 3 then nat.choose 15 3 else 0 = 455 :=
by {
  sorry
}

end pentadecagon_triangle_count_l69_69942


namespace min_dot_product_l69_69193

variables {a b : ℝ^3}
noncomputable def dot_product (u v : ℝ^3) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
noncomputable def norm (v : ℝ^3) := real.sqrt (dot_product v v)

theorem min_dot_product (ha : norm a = 3) (h : norm (a - 2 • b) ≤ 1) : dot_product a b ≥ -1 / 8 :=
by
  sorry

end min_dot_product_l69_69193


namespace monic_quadratic_with_root_2_minus_3i_l69_69848

theorem monic_quadratic_with_root_2_minus_3i :
  ∃ P : ℝ[X], P.monic ∧ (P.coeff 2 = 1)
    ∧ (P.coeff 1 = -4)
    ∧ (P.coeff 0 = 13)
    ∧ eval (2 - 3 * I) P = 0 := sorry

end monic_quadratic_with_root_2_minus_3i_l69_69848


namespace count_four_digit_palindrome_squares_l69_69026

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69026


namespace abs_h_proof_l69_69717

noncomputable def abs_h (h : ℝ) : ℝ := 
  real.sqrt (3 / 2)

theorem abs_h_proof (h : ℝ) (roots_sum : ℝ) (roots_product : ℝ) 
(roots_squares_sum : ℝ)
(h_roots_sum : roots_sum = 4 * h)
(h_roots_product : roots_product = -5)
(h_roots_squares_sum : roots_squares_sum = 34) :
  |h| = abs_h h :=
by
  sorry

end abs_h_proof_l69_69717


namespace number_of_ways_to_place_balls_l69_69867

open finset

noncomputable def balls : finset ℕ := {1, 2, 3, 4}
noncomputable def boxes : finset ℕ := {1, 2, 3, 4}

theorem number_of_ways_to_place_balls : (∃ f : balls → option boxes, (∀ b ∈ balls, f b ≠ none) ∧ card (range f) = 3) → 144 := by
  sorry

end number_of_ways_to_place_balls_l69_69867


namespace sphere_ratio_and_total_volume_l69_69771

theorem sphere_ratio_and_total_volume (V1 V2 : ℝ) (r1 r2 : ℝ) 
(hV1 : V1 = 432 * Real.pi) 
(hV2 : V2 = 0.15 * 432 * Real.pi) 
(hr1 : (4/3) * Real.pi * r1^3 = V1) 
(hr2 : (4/3) * Real.pi * r2^3 = V2) : 
  (r2 / r1 = (Real.cbrt 1.8) / 2) ∧ (V1 + V2 = 496.8 * Real.pi) :=
by
  sorry

end sphere_ratio_and_total_volume_l69_69771


namespace percent_increase_in_sales_l69_69782

theorem percent_increase_in_sales (sales_this_year : ℕ) (sales_last_year : ℕ) (percent_increase : ℚ) :
  sales_this_year = 400 ∧ sales_last_year = 320 → percent_increase = 25 :=
by
  sorry

end percent_increase_in_sales_l69_69782


namespace point_A_final_position_supplement_of_beta_l69_69298

-- Define the initial and final position of point A on the number line
def initial_position := -5
def moved_position_right := initial_position + 4
def final_position := moved_position_right - 1

theorem point_A_final_position : final_position = -2 := 
by 
-- Proof can be added here
sorry

-- Define the angles and the relationship between them
def alpha := 40
def beta := 90 - alpha
def supplement_beta := 180 - beta

theorem supplement_of_beta : supplement_beta = 130 := 
by 
-- Proof can be added here
sorry

end point_A_final_position_supplement_of_beta_l69_69298


namespace acute_triangle_of_sin_cos_l69_69164

theorem acute_triangle_of_sin_cos (A B C : ℝ) 
  (h1 : 0 < A ∧ A < 180) 
  (h2 : sin A + cos A = 7 / 12) : 
  A < 90 := 
sorry

end acute_triangle_of_sin_cos_l69_69164


namespace polynomial_form_l69_69491

theorem polynomial_form (P : Polynomial ℝ) (hP : P ≠ 0)
    (h : ∀ x : ℝ, P.eval x * P.eval (2 * x^2) = P.eval (2 * x^3 + x)) :
    ∃ k : ℕ, k > 0 ∧ P = (X^2 + 1) ^ k :=
by sorry

end polynomial_form_l69_69491


namespace max_sum_inscribed_radii_l69_69756

open Real

theorem max_sum_inscribed_radii 
(side_len : ℝ) 
(h_side_len : side_len = 2)
(r_a r_b r_c : ℝ)
(C1 B1 A1 : ℝ×ℝ)
(h_triangle : let ABC ≡ (0, 0), (side_len, 0), (side_len / 2, (side_len * sqrt 3 / 2)))
(h_C1B1A1_on_sides : C1 ∈ segment (0, 0) (side_len, 0) ∧ B1 ∈ segment (side_len / 2, (side_len * sqrt 3 / 2)) (0, 0) ∧ A1 ∈ segment (side_len, 0) (side_len / 2, (side_len * sqrt 3 / 2)))
: r_a + r_b + r_c ≤ sqrt 3 / 2 :=
by
  sorry

end max_sum_inscribed_radii_l69_69756


namespace prove_heron_formula_prove_S_squared_rrarc_l69_69303

variables {r r_a r_b r_c p a b c S : ℝ}

-- Problem 1: Prove Heron's Formula
theorem prove_heron_formula (h1 : r * p = r_a * (p - a))
                            (h2 : r * r_a = (p - b) * (p - c))
                            (h3 : r_b * r_c = p * (p - a)) :
  S^2 = p * (p - a) * (p - b) * (p - c) :=
sorry

-- Problem 2: Prove S^2 = r * r_a * r_b * r_c
theorem prove_S_squared_rrarc (h1 : r * p = r_a * (p - a))
                              (h2 : r * r_a = (p - b) * (p - c))
                              (h3 : r_b * r_c = p * (p - a)) :
  S^2 = r * r_a * r_b * r_c :=
sorry

end prove_heron_formula_prove_S_squared_rrarc_l69_69303


namespace sum_b_1000_eq_23264_l69_69862

noncomputable def b (p : ℕ) : ℕ :=
  ⟨some int.sqrt (p - 1/2)^2⟩

noncomputable def sum_b (n : ℕ) : ℕ :=
  ∑ p in (finset.range (n + 1)).filter (λ p, p ≠ 0), b p

theorem sum_b_1000_eq_23264 : sum_b 1000 = 23264 :=
by
  sorry

end sum_b_1000_eq_23264_l69_69862


namespace pentadecagon_triangle_count_l69_69957

-- Define the problem of selecting 3 vertices out of 15 to form a triangle
theorem pentadecagon_triangle_count : 
  ∃ (n : ℕ), n = nat.choose 15 3 ∧ n = 455 := 
by {
  sorry
}

end pentadecagon_triangle_count_l69_69957


namespace arithmetic_sequence_term_20_l69_69812

theorem arithmetic_sequence_term_20
  (a : ℕ := 2)
  (d : ℕ := 4)
  (n : ℕ := 20) :
  a + (n - 1) * d = 78 :=
by
  sorry

end arithmetic_sequence_term_20_l69_69812


namespace matrix_y_solution_l69_69101

theorem matrix_y_solution (y : ℝ) : 
  let a := (3 : ℝ) * y
  let b := y
  let c := (2 : ℝ)
  let d := (3 : ℝ)
  (a * b - c * d = 4) ↔ (y = sqrt (10 / 3) ∨ y = -sqrt (10 / 3)) :=
by sorry

end matrix_y_solution_l69_69101


namespace min_value_expr_l69_69871

theorem min_value_expr (a b : ℝ) (h1 : 2 * a + b = a * b) (h2 : a > 0) (h3 : b > 0) : 
  ∃ a b, (a > 0 ∧ b > 0 ∧ 2 * a + b = a * b) ∧ (∀ x y, (x > 0 ∧ y > 0 ∧ 2 * x + y = x * y) → (1 / (x - 1) + 2 / (y - 2)) ≥ 2) ∧ ((1 / (a - 1) + 2 / (b - 2)) = 2) :=
by
  sorry

end min_value_expr_l69_69871


namespace monic_quadratic_real_root_l69_69832

theorem monic_quadratic_real_root (a b : ℂ) (h : b = 2 - 3 * complex.I) :
  ∃ P : polynomial ℂ, P.monic ∧ P.coeff 2 = 1 ∧ P.coeff 1 = -4 ∧ P.coeff 0 = 13 ∧ P.is_root (2 - 3 * complex.I) :=
by
  sorry

end monic_quadratic_real_root_l69_69832


namespace intersection_of_sets_l69_69928

open Set Real

theorem intersection_of_sets :
  let M := {y | ∃ x : ℝ, y = Real.sin x}
  let N := ({0, 1, 2} : Set ℝ)
  M ∩ N = {0, 1} := 
by {
  -- Definitions based on conditions
  let M := {y | ∃ x : ℝ, y = Real.sin x},
  let N := ({0, 1, 2} : Set ℝ),
  
  -- Goal: Prove M ∩ N = {0, 1}
  sorry
}

end intersection_of_sets_l69_69928


namespace sphere_surface_area_l69_69666

theorem sphere_surface_area
  (A B C : Point)
  (AB BC AC : ℝ)
  (h_AB : AB = 6)
  (h_BC : BC = 8)
  (h_AC : AC = 10)
  (h_right_triangle : AB^2 + BC^2 = AC^2)
  (r : ℝ)
  (h_r : r = 5)
  (R : ℝ)
  (h_dist : (R / 2)^2 = R^2 - r^2)
  : 4 * real.pi * R^2 = (400 * real.pi) / 3 := by
  sorry

end sphere_surface_area_l69_69666


namespace range_of_g_l69_69263

def floor_function (x : ℝ) : ℤ := int.floor x

def g (x : ℝ) : ℝ := 2 * (floor_function x) - x

theorem range_of_g : set.Ioo (-∞ : ℝ) ∞ = set.univ :=
by {
  sorry
}

end range_of_g_l69_69263


namespace ratio_y_to_x_l69_69462

variable (x y z : ℝ)

-- Conditions
def condition1 (x y z : ℝ) := 0.6 * (x - y) = 0.4 * (x + y) + 0.3 * (x - 3 * z)
def condition2 (y z : ℝ) := ∃ k : ℝ, z = k * y
def condition3 (y z : ℝ) := z = 7 * y
def condition4 (x y : ℝ) := y = 5 * x / 7

theorem ratio_y_to_x (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 y z) (h3 : condition3 y z) (h4 : condition4 x y) : y / x = 5 / 7 :=
by
  sorry

end ratio_y_to_x_l69_69462


namespace students_did_not_eat_2_l69_69662

-- Define the given conditions
def total_students : ℕ := 20
def total_crackers_eaten : ℕ := 180
def crackers_per_pack : ℕ := 10

-- Calculate the number of packs eaten
def packs_eaten : ℕ := total_crackers_eaten / crackers_per_pack

-- Calculate the number of students who did not eat their animal crackers
def students_who_did_not_eat : ℕ := total_students - packs_eaten

-- Prove that the number of students who did not eat their animal crackers is 2
theorem students_did_not_eat_2 :
  students_who_did_not_eat = 2 :=
  by
    sorry

end students_did_not_eat_2_l69_69662


namespace num_four_digit_palindromic_squares_l69_69014

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69014


namespace balance_difference_l69_69808

noncomputable def cedric_balance (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

noncomputable def daniel_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem balance_difference :
  let P := 10000
  let cedric_rate := 0.06
  let daniel_rate := 0.08
  let years := 10
  ∃ (d : ℝ), d = daniel_balance P daniel_rate years - cedric_balance P cedric_rate years ∧
  abs d ≈ 91.52 := sorry

end balance_difference_l69_69808


namespace chessboard_coloring_formula_l69_69141

noncomputable def h'_m (m n : ℕ) : ℕ :=
  ∑ k in finset.range (m + 1), if k >= 2 then (-1)^(m - k) * nat.choose m k * (k^2 - 3*k + 3)^(n - 1) * k * (k - 1) else 0

theorem chessboard_coloring_formula (m n : ℕ) (h : m >= 2) : 
  h'_m m n = ∑ k in (finset.range (m + 1)).filter (λ k, k >= 2), (-1)^(m - k) * nat.choose m k * (k^2 - 3*k + 3)^(n - 1) * k * (k - 1) :=
sorry

end chessboard_coloring_formula_l69_69141


namespace no_solutions_xn_plus_1_eq_yn_plus_1_l69_69461

theorem no_solutions_xn_plus_1_eq_yn_plus_1 {n x y : ℕ} (h1 : n ≥ 2) (h2 : Nat.coprime x (n + 1)) :
  ¬ (x^n + 1 = y^(n+1)) :=
sorry

end no_solutions_xn_plus_1_eq_yn_plus_1_l69_69461


namespace falcons_win_ratio_l69_69227

theorem falcons_win_ratio {N : ℕ} (H : 3 + N ≥ 42) : 
    (3 + N : ℝ) / (8 + N : ℝ) ≥ 0.9 :=
by {
    suffices : (3 + 42 : ℝ) / (8 + 42 : ℝ) ≥ 0.9, from le_trans (by linarith only) this,
    norm_num,
    sorry
}

end falcons_win_ratio_l69_69227


namespace construction_is_valid_l69_69594

noncomputable def construction_valid (ABC : Triangle) (a b c : Point)  
  (H_tri: is_triangle ABC) (H_AB_lt_AC: ABC.AB < ABC.AC) (a_1: Line)
  (H_a1_parallel: parallel a_1 ABC.BC) (C1 B1: Point)
  (H_C1_on_AB: C1 ∈ ABC.AB) (H_B1_on_AC: B1 ∈ ABC.AC) : Prop :=
bc1_distance (abc_side_lengths A B C) + c1b1_distance (abc_side_lengths A B C) = cb1_distance (abc_side_lengths A B C)

theorem construction_is_valid (ABC : Triangle) (a b c : Point)  
  (H_tri: is_triangle ABC) (H_AB_lt_AC: ABC.AB < ABC.AC) (a_1: Line)
  (H_a1_parallel: parallel a_1 ABC.BC) (C1 B1: Point)
  (H_C1_on_AB: C1 ∈ ABC.AB) (H_B1_on_AC: B1 ∈ ABC.AC) :
  construction_valid ABC a b c H_tri H_AB_lt_AC a_1 H_a1_parallel C1 B1 H_C1_on_AB H_B1_on_AC := 
sorry

end construction_is_valid_l69_69594


namespace pentadecagon_triangle_count_l69_69943

theorem pentadecagon_triangle_count :
  ∑ k in finset.range 15, if k = 3 then nat.choose 15 3 else 0 = 455 :=
by {
  sorry
}

end pentadecagon_triangle_count_l69_69943


namespace magnitude_of_z_l69_69549

theorem magnitude_of_z (z : Complex) (h : z = (1 - 3 * Complex.i) / (1 + 2 * Complex.i)) : Complex.abs z = Real.sqrt 2 := by
  sorry

end magnitude_of_z_l69_69549


namespace sequence_periodicity_l69_69924

noncomputable def a : ℕ → ℤ
| 1 := 3
| 2 := 6
| n+2 := a (n+1) - a n

theorem sequence_periodicity : a 2014 = -3 := by
  sorry

end sequence_periodicity_l69_69924


namespace train_length_l69_69417

variable (L_train : ℝ)
variable (speed_kmhr : ℝ := 45)
variable (time_seconds : ℝ := 30)
variable (bridge_length_m : ℝ := 275)
variable (train_speed_ms : ℝ := speed_kmhr * (1000 / 3600))
variable (total_distance : ℝ := train_speed_ms * time_seconds)

theorem train_length
  (h_total : total_distance = L_train + bridge_length_m) :
  L_train = 100 :=
by 
  sorry

end train_length_l69_69417


namespace roots_of_polynomial_l69_69843

noncomputable def quadratic_polynomial (x : ℝ) := x^2 - 4 * x + 13

theorem roots_of_polynomial : ∀ (z : ℂ), z = 2 - 3 * complex.I → quadratic_polynomial (z.re) = 0 :=
by
  intro z h
  sorry

end roots_of_polynomial_l69_69843


namespace count_four_digit_palindrome_squares_l69_69034

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69034


namespace correct_number_of_conclusions_l69_69169

variable {f : ℝ → ℝ}
variable {a b : ℝ}

-- Defining that f is increasing
def increasing (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f(x) ≤ f(y)

-- Given conditions
variable (hf : increasing f)
variable (hab : a + b ≥ 0)

-- The Proposition
def proposition : Prop := f(a) + f(b) ≥ f(-a) + f(-b)

-- The converse proposition
def converse_proposition : Prop := f(a) + f(b) < f(-a) + f(-b) -> a + b < 0

-- The negation of the proposition
def negation : Prop := a + b < 0 -> f(a) + f(b) < f(-a) + f(-b)

-- The contrapositive of the proposition
def contrapositive : Prop := f(a) + f(b) < f(-a) + f(-b) -> a + b < 0

-- The proof statement
theorem correct_number_of_conclusions (hf : increasing f) : 
    (proposition hf hab) ∧ 
    (converse_proposition hf) ∧ 
    (negation hf) ∧ 
    (contrapositive hf) :=
by {
    sorry
}

end correct_number_of_conclusions_l69_69169


namespace complex_square_eq_l69_69142

variables {a b : ℝ} {i : ℂ}

theorem complex_square_eq :
  a + i = 2 - b * i → (a + b * i) ^ 2 = 3 - 4 * i :=
by sorry

end complex_square_eq_l69_69142


namespace inverse_matrix_l69_69896

theorem inverse_matrix {α : ℝ} (M : Matrix (Fin 2) (Fin 2) ℝ)
  (hM : M = Matrix.of ![![Real.cos α, -Real.sin α], ![Real.sin α, Real.cos α]])
  (hA : Vector 2 ℝ)
  (hA_coords : hA = ![2, 2])
  (hB : Vector 2 ℝ)
  (hB_coords : hB = ![-2, 2]) 
  (h_transformation : M.mulVec hA = hB) :
  Matrix.inv M = Matrix.of ![![0, 1], ![-1, 0]] :=
by
  sorry

end inverse_matrix_l69_69896


namespace find_floors_l69_69070

theorem find_floors
  (a b : ℕ)
  (alexie_bathrooms_per_floor : ℕ := 3)
  (alexie_bedrooms_per_floor : ℕ := 2)
  (baptiste_bathrooms_per_floor : ℕ := 4)
  (baptiste_bedrooms_per_floor : ℕ := 3)
  (total_bathrooms : ℕ := 25)
  (total_bedrooms : ℕ := 18)
  (h1 : alexie_bathrooms_per_floor * a + baptiste_bathrooms_per_floor * b = total_bathrooms)
  (h2 : alexie_bedrooms_per_floor * a + baptiste_bedrooms_per_floor * b = total_bedrooms) :
  a = 3 ∧ b = 4 :=
by
  sorry

end find_floors_l69_69070


namespace vector_sum_zero_l69_69806

variables {V : Type*} [AddCommGroup V]

/-- Given three points A, B, and C, and their corresponding vectors AB, CA, and CB. --/
variables (A B C : V)

/-- Given conditions on the vectors CA and CB. --/
axiom CA_eq_neg_AC : ∀ (A B C : V), A = C → \overrightarrow{CA} = -\overrightarrow{AC}
axiom CB_eq_neg_BC : ∀ (A B C : V), B = C → \overrightarrow{CB} = -\overrightarrow{BC}

/-- We want to prove that the vector sum of AB, CA, and CB leads to the zero vector. --/
theorem vector_sum_zero : 
  \overrightarrow{AB} + \overrightarrow{CA} - \overrightarrow{CB} = \overrightarrow{0} :=
sorry

end vector_sum_zero_l69_69806


namespace equal_real_roots_of_quadratic_l69_69205

theorem equal_real_roots_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 3 = 0 ∧ 
               (∀ y : ℝ, 3 * y^2 - m * y + 3 = 0 → y = x)) → 
  m = 6 ∨ m = -6 :=
by
  sorry  -- proof to be filled in.

end equal_real_roots_of_quadratic_l69_69205


namespace greatest_x_value_l69_69355

noncomputable def max_x_value : ℚ :=
  let y := (6 * x - 15) / (4 * x - 5) in
  if h : (y^2 - 3 * y - 10 = 0) then
    let x₁ := (5 / 7) in
    let x₂ := (25 / 14) in
    if x₁ < x₂ then x₂ else x₁
  else 0

theorem greatest_x_value : max_x_value = (25 / 14) :=
by
  sorry -- Proof is not required as per the prompt

end greatest_x_value_l69_69355


namespace eval_poly_at_root_l69_69481

noncomputable def poly (y : ℝ) : ℝ := y^3 - 3*y^2 - 9*y + 5

theorem eval_poly_at_root :
  ∀ (y : ℝ), (y^2 - 3*y - 9 = 0) ∧ (y > 0) → poly y = 5 :=
by
  intros y h
  cases h with h1 h2
  -- proof goes here
  sorry

end eval_poly_at_root_l69_69481


namespace chord_length_l69_69119

-- Define the line
def line_eq (x y : ℝ) : Prop := x + y = 3

-- Define the curve (circle)
def curve_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- The main theorem stating the length of the chord
theorem chord_length
  (h1 : ∀ (x y : ℝ), line_eq x y → curve_eq x y)
  (h2 : ∀ (x : ℝ), x ∉ set.range (λ y: ℝ, line_eq x y ∧ curve_eq x y)) :
  ∃ (d : ℝ), d = 2 * real.sqrt(2) :=
sorry

end chord_length_l69_69119


namespace symmetric_qr_code_5x5_l69_69390

theorem symmetric_qr_code_5x5 :
  let valid_codes := 
    {code : (Fin 5 × Fin 5) → Bool // 
      (∃ i j, code (i, j) = true ∧ ∃ i j, code (i, j) = false) ∧
      ∀ θ : ℤ, (θ % 90 = 0 → code = code ∘ rotation (Fin 5) θ) ∧
      ∀ (L : Fin 5 → Fin 5), is_reflection L → code = code ∘ L} in
  |valid_codes| = 62 := 
by
  sorry

end symmetric_qr_code_5x5_l69_69390


namespace value_of_m_l69_69537

-- Problem Statement
theorem value_of_m (m : ℝ) : (∃ x : ℝ, (m-2)*x^(|m|-1) + 16 = 0 ∧ |m| - 1 = 1) → m = -2 :=
by
  sorry

end value_of_m_l69_69537


namespace exists_convex_body_diff_from_sphere_with_circular_projections_l69_69247

-- Definitions of the coordinate planes
def plane_α : set (ℝ × ℝ × ℝ) := {p | p.1 = 0}
def plane_β : set (ℝ × ℝ × ℝ) := {p | p.2 = 0}
def plane_γ : set (ℝ × ℝ × ℝ) := {p | p.3 = 0}

-- Definitions of the unit sphere and the cylinders
def sphere_B : set (ℝ × ℝ × ℝ) := {p | p.1^2 + p.2^2 + p.3^2 ≤ 1}
def cylinder_C1 : set (ℝ × ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}
def cylinder_C2 : set (ℝ × ℝ × ℝ) := {p | p.2^2 + p.3^2 ≤ 1}
def cylinder_C3 : set (ℝ × ℝ × ℝ) := {p | p.3^2 + p.1^2 ≤ 1}

-- Definition of the intersection of the cylinders
def body_C : set (ℝ × ℝ × ℝ) := cylinder_C1 ∩ cylinder_C2 ∩ cylinder_C3

-- The main proof statement
theorem exists_convex_body_diff_from_sphere_with_circular_projections :
  ∃ C : set (ℝ × ℝ × ℝ), 
  C ≠ sphere_B ∧ 
  (∀ p ∈ C, is_convex C) ∧ 
  (∀ p ∈ plane_α, orthogonal_projection C plane_α p = {q | q.1^2 + q.2^2 ≤ 1}) ∧ 
  (∀ p ∈ plane_β, orthogonal_projection C plane_β p = {q | q.2^2 + q.3^2 ≤ 1}) ∧ 
  (∀ p ∈ plane_γ, orthogonal_projection C plane_γ p = {q | q.3^2 + q.1^2 ≤ 1}) :=
sorry

end exists_convex_body_diff_from_sphere_with_circular_projections_l69_69247


namespace smallest_positive_integer_l69_69124

theorem smallest_positive_integer :
  ∃ (n a b m : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ n = 153846 ∧
  (n = 10^m * a + b) ∧
  (7 * n = 2 * (10 * b + a)) :=
by
  sorry

end smallest_positive_integer_l69_69124


namespace condition_is_necessary_but_not_sufficient_l69_69384

noncomputable def sequence_satisfies_condition (a : ℕ → ℤ) : Prop :=
  a 3 + a 7 = 2 * a 5

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d

theorem condition_is_necessary_but_not_sufficient (a : ℕ → ℤ) :
  (sequence_satisfies_condition a ∧ (¬ arithmetic_sequence a)) ∨
  (arithmetic_sequence a → sequence_satisfies_condition a) :=
sorry

end condition_is_necessary_but_not_sufficient_l69_69384


namespace rhombus_triangle_area_ratio_l69_69067

-- Define the given conditions
variables (m n : ℝ) (h_common_angle : Prop) (h_ratio : (m / n = m / n))

-- Define the ratio of the areas
def ratio_of_areas := (2 * m * n) / (m + n)^2

-- State the theorem
theorem rhombus_triangle_area_ratio (h_common_angle : Prop) (h_ratio : (m / n = m / n)) :
  ratio_of_areas m n = (2 * m * n) / (m + n)^2 :=
sorry

end rhombus_triangle_area_ratio_l69_69067


namespace sum_of_coefficients_l69_69693

theorem sum_of_coefficients : 
  let A := 1
  let B := -2
  let C := 3
  let D := 6
  let E := 12
  let F := 1
  let G := 2
  let H := 3
  let J := -6
  let K := 12
  (A + B + C + D + E + F + G + H + J + K) = 32 :=
begin
  let A := 1,
  let B := -2,
  let C := 3,
  let D := 6,
  let E := 12,
  let F := 1,
  let G := 2,
  let H := 3,
  let J := -6,
  let K := 12,
  calc (A + B + C + D + E + F + G + H + J + K)
      = 1 + -2 + 3 + 6 + 12 + 1 + 2 + 3 + -6 + 12 : by sorry
   ... = 32                                   : by sorry
end

end sum_of_coefficients_l69_69693


namespace smallest_number_of_students_l69_69051

theorem smallest_number_of_students
    (g11 g10 g9 : Nat)
    (h_ratio1 : 4 * g9 = 3 * g11)
    (h_ratio2 : 6 * g10 = 5 * g11) :
  g11 + g10 + g9 = 31 :=
sorry

end smallest_number_of_students_l69_69051


namespace factorize_expression_l69_69489

variable {a x y : ℝ}

theorem factorize_expression : (a * x^2 + 2 * a * x * y + a * y^2) = a * (x + y)^2 := by
  sorry

end factorize_expression_l69_69489


namespace original_number_is_8_l69_69774

open Real

theorem original_number_is_8 
  (x : ℝ)
  (h1 : |(x + 5) - (x - 5)| = 10)
  (h2 : (10 / (x + 5)) * 100 = 76.92) : 
  x = 8 := 
by
  sorry

end original_number_is_8_l69_69774


namespace number_of_triangles_in_pentadecagon_l69_69987

open Finset

theorem number_of_triangles_in_pentadecagon :
  ∀ (n : ℕ), n = 15 → (n.choose 3 = 455) := 
by 
  intros n hn 
  rw hn
  rw Nat.choose_eq_factorial_div_factorial (show 3 ≤ 15)
  { norm_num }

-- Proof omitted with sorry

end number_of_triangles_in_pentadecagon_l69_69987


namespace count_four_digit_palindrome_squares_l69_69028

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69028


namespace general_formula_l69_69525

noncomputable def a (n : ℕ) : ℕ := -2 * n + 5

def S (n : ℕ) := n * (2 * a 1 + (n - 1) * -2) / 2

def b (n : ℕ) : ℕ := 1 / (a n * a (n + 1))

def T (n : ℕ) := ∑ i in range n, b i

theorem general_formula (n : ℕ) :
  S 5 = -5 ∧ (∀ n, T n - 1 / (2 * a (n + 1)) = -1 / 6) := sorry

end general_formula_l69_69525


namespace proof_problem1_proof_problem2_l69_69800

open Real

noncomputable def problem1 : ℝ :=
  2^(-1/2) + (-3)^0 / sqrt 2 - (1 / (sqrt 2 + 1)) + sqrt ((1 - sqrt 6) ^ 0)

theorem proof_problem1 : problem1 = 2 := by sorry

noncomputable def lg (x : ℝ) := log x / log 10

noncomputable def problem2 : ℝ :=
  (lg 5 * lg 20) - (lg 2 * lg 50) - lg 25

theorem proof_problem2 : problem2 = -1 := by sorry

end proof_problem1_proof_problem2_l69_69800


namespace trailing_zeros_1500_factorial_l69_69799

theorem trailing_zeros_1500_factorial : 
  let count_factors(n: ℕ, p: ℕ) := n/ₚ┘ + n/(p^2) + n/(p^3) + n/(p^4) 
  in count_factors(1500, 5) = 374 :=
by
  unfold count_factors
  have h1: 1500 / 5 = 300 := by norm_num
  have h2: 1500 / 25 = 60 := by norm_num
  have h3: 1500 / 125 = 12 := by norm_num
  have h4: 1500 / 625 = 2 := by norm_num
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end trailing_zeros_1500_factorial_l69_69799


namespace proof_a_and_b_parity_of_f_monotonicity_and_range_of_f_l69_69182

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem proof_a_and_b (h1 : f 1 = 5 / 2) (h2 : f 2 = 17 / 4) : ∀ x : ℝ, f x = 2^x + 2^(-x) :=
by
  sorry

theorem parity_of_f : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem monotonicity_and_range_of_f : (∀ x1 x2 : ℝ, 0 ≤ x1 → x1 < x2 → x2 ∈ set.Ici 0 → f x1 < f x2) ∧ (set.range f = set.Ici 2) :=
by
  sorry

end proof_a_and_b_parity_of_f_monotonicity_and_range_of_f_l69_69182


namespace min_value_cos_sin_expr_l69_69509

theorem min_value_cos_sin_expr (x : ℝ) : 
  (∃ t : ℤ, x = (±π / 3 + π * t : ℝ)) → 
  ∀ θ : ℝ, cos^2 (π * cos θ) + sin^2 (2 * π * sqrt 3 * sin θ) = 0 := 
by 
suffices : cos^2(π * cos θ) = 0 ∧ sin^2(2 * π * sqrt 3 * sin θ) = 0,
  sorry

end min_value_cos_sin_expr_l69_69509


namespace union_sets_l69_69529

def A : Set ℝ := {x | log 3 (x + 2) < 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_sets :
  A ∪ B = {x : ℝ | -2 < x ∧ x < 2} := 
by
  sorry

end union_sets_l69_69529


namespace phase_shift_of_sine_l69_69122

theorem phase_shift_of_sine :
  let B := 5
  let C := (3 * Real.pi) / 2
  let phase_shift := C / B
  phase_shift = (3 * Real.pi) / 10 := by
    sorry

end phase_shift_of_sine_l69_69122


namespace fill_tank_time_l69_69751

theorem fill_tank_time (R1 R2 R3 : ℚ) :
  R1 = 1 / 18 ∧ R2 = 1 / 30 ∧ R3 = - (1 / 45) →
  (1 / (R1 + R2 + R3) = 15) :=
by
  intros h
  cases h with hR1 hRest
  cases hRest with hR2 hR3
  have h_total : R1 + R2 + R3 = 1 / 15 := sorry
  have h_time : 1 / (1 / 15) = 15 := sorry
  rw h_total at h_time
  assumption

end fill_tank_time_l69_69751


namespace nonagon_triangle_probability_l69_69512

theorem nonagon_triangle_probability :
  (∃ (t : Triangle), t.vertices ⊆ (vertices : Finset Point nonagon) ∧
  (∃ (s : Side), s ∈ t.sides ∧ s ∈ nonagon.sides)) →
  (realize (probability (triangle_with_side_nonagon nonagon) = 9 / 14)) :=
by
  sorry

end nonagon_triangle_probability_l69_69512


namespace areas_equal_l69_69592

variables {A1 A2 A3 A4 A5 A6 : Type} [ConvexHexagon A1 A2 A3 A4 A5 A6]

axiom opposite_sides_parallel :
  (A1A2 ∥ A4A5) ∧ (A2A3 ∥ A5A6) ∧ (A3A4 ∥ A6A1)

noncomputable def area_triangle_A1A3A5 :
  ℝ := sorry

noncomputable def area_triangle_A2A4A6 :
  ℝ := sorry

theorem areas_equal :
  area_triangle_A1A3A5 = area_triangle_A2A4A6 :=
by {
  sorry
}

end areas_equal_l69_69592


namespace calculate_moment_of_inertia_l69_69084

noncomputable def moment_of_inertia (a ρ₀ k : ℝ) : ℝ :=
  8 * (a ^ (9/2)) * ((ρ₀ / 7) + (k * a / 9))

theorem calculate_moment_of_inertia (a ρ₀ k : ℝ) 
  (h₀ : 0 ≤ a) :
  moment_of_inertia a ρ₀ k = 8 * a ^ (9/2) * ((ρ₀ / 7) + (k * a / 9)) :=
sorry

end calculate_moment_of_inertia_l69_69084


namespace boy_scouts_percentage_l69_69770

variable (S B G : ℝ)

-- Conditions
-- Given B + G = S
axiom condition1 : B + G = S

-- Given 0.75B + 0.625G = 0.7S
axiom condition2 : 0.75 * B + 0.625 * G = 0.7 * S

-- Goal
theorem boy_scouts_percentage : B / S = 0.6 :=
by sorry

end boy_scouts_percentage_l69_69770


namespace find_angle_D_l69_69519

-- Define the conditions
axiom is_convex_pentagon (ABCDE : Type) [is_convex_pentagon ABCDE] : Prop
axiom equal_sides (ABCDE : Type) [is_convex_pentagon ABCDE] : Prop
axiom angle_A_is_120 (ABCDE : Type) [is_convex_pentagon ABCDE] : ∀ (A : ABCDE), A.angle = 120
axiom angle_C_is_135 (ABCDE : Type) [is_convex_pentagon ABCDE] : ∀ (C : ABCDE), C.angle = 135
axiom sum_of_interior_angles_is_540 (ABCDE : Type) [is_convex_pentagon ABCDE] : 
  ∑ angles (pentagon_contains angles ABCDE) = 540

-- Prove the question
theorem find_angle_D (ABCDE : Type) [is_convex_pentagon ABCDE] :
  ∃ (D : ABCDE), angle_D D = 90 :=
begin
  sorry
end

end find_angle_D_l69_69519


namespace ball_cost_l69_69510

theorem ball_cost (C x y : ℝ)
  (H1 :  x = 1/3 * (C/2 + y + 5) )
  (H2 :  y = 1/4 * (C/2 + x + 5) )
  (H3 :  C/2 + x + y + 5 = C ) : C = 20 := 
by
  sorry

end ball_cost_l69_69510


namespace kolka_mistake_l69_69316

-- Definitions of the problem.
def notebook_sheets : ℕ := 96
def pages_per_sheet : ℕ := 2
def total_pages : ℕ := notebook_sheets * pages_per_sheet
def torn_sheets : ℕ := 25
def sum_per_sheet : ℕ := 193
def kolka_reported_sum : ℕ := 2002

-- Prove that Kolka made a mistake.
theorem kolka_mistake : ∃ (sum_of_pages_on_torn_sheets: ℕ), sum_of_pages_on_torn_sheets ≠ kolka_reported_sum := by {
  -- Derive the expected sum of the pages on the torn sheets.
  let expected_sum := torn_sheets * sum_per_sheet,
  use expected_sum,
  sorry
}

end kolka_mistake_l69_69316


namespace find_pq_sum_l69_69317

theorem find_pq_sum
  (y : ℝ)
  (h1 : Real.csc y + Real.cot y = 25 / 7)
  (h2 : (∃ p q : ℕ, (Nat.gcd p q = 1) ∧ Real.sec y + Real.tan y = p / q)) :
  ∃ p q, (Real.sec y + Real.tan y = p / q ∧ Nat.gcd p q = 1) ∧ p + q = 29517 := 
by
  sorry

end find_pq_sum_l69_69317


namespace system_of_equations_solution_l69_69629

theorem system_of_equations_solution (n : ℕ) (h : 0 < n) :
  ∃! (x : fin n → ℝ), (∀ i, 0 ≤ x i) ∧
  (∑ i in finset.range n, (i+1) * x ⟨i, fin.is_lt i⟩ = n * (n + 1) / 2) ∧
  (∑ i in finset.range n, x ⟨i, fin.is_lt i⟩ ^ (i + 1) = n) :=
begin
  sorry
end

end system_of_equations_solution_l69_69629


namespace new_edges_of_modified_prism_l69_69411

theorem new_edges_of_modified_prism :
  ∀ (prism : Type) (v e : ℕ),
  (v = 8) →
  (e = 12) →
  (∀ v_intersections : ℕ, v_intersections = 2) →
  (∀ additional_edges_per_intersection : ℕ, additional_edges_per_intersection = 1) →
  (∀ vertex_triangle_edges : ℕ, vertex_triangle_edges = 3) →
  let new_edges_from_triangles := v * vertex_triangle_edges in
  let additional_intersection_edges := v * v_intersections * additional_edges_per_intersection in
  let total_new_edges := new_edges_from_triangles + additional_intersection_edges in
  (e + total_new_edges = 52) :=
sorry

end new_edges_of_modified_prism_l69_69411


namespace square_root_sqrt_81_eq_pm3_l69_69364

theorem square_root_sqrt_81_eq_pm3 :
  sqrt 81 = 9 → sqrt 9 = 3 ∨ sqrt 9 = -3 := 
by
  sorry

end square_root_sqrt_81_eq_pm3_l69_69364


namespace find_salary_l69_69772

theorem find_salary (S : ℝ) (h_food : S / 3) (h_rent : S / 4) (h_clothes : S / 5) (h_left : S - S / 3 - S / 4 - S / 5 = 1760) : S = 8123.08 := 
by
  sorry

end find_salary_l69_69772


namespace distance_between_lines_is_five_thirteenths_l69_69458

noncomputable def distance_between_parallel_lines : ℝ :=
  let a := (4, -2)
  let b := (3, -1)
  let d := (2, -3)
  let v := (b.1 - a.1, b.2 - a.2) -- Vector from point a to point b
  let dot_product_vd := v.1 * d.1 + v.2 * d.2
  let dot_product_dd := d.1 * d.1 + d.2 * d.2
  let p := (dot_product_vd / dot_product_dd * d.1, dot_product_vd / dot_product_dd * d.2)
  let c := (v.1 - p.1, v.2 - p.2) -- Orthogonal projection of v on the line direction
  real.sqrt (c.1 * c.1 + c.2 * c.2)

theorem distance_between_lines_is_five_thirteenths :
  distance_between_parallel_lines = 5 / 13 :=
sorry

end distance_between_lines_is_five_thirteenths_l69_69458


namespace highest_point_of_shell_l69_69779

noncomputable def shell_height (a b c x : ℝ) := a * x^2 + b * x + c

theorem highest_point_of_shell 
  (a b c : ℝ) (h : a ≠ 0)
  (h_eq : shell_height a b c 7 = shell_height a b c 14) : 
  ∃ x, x ∈ {8, 10, 12, 15} ∧ 
    (∀ y, y ∈ {8, 10, 12, 15} → shell_height a b c 10 ≥ shell_height a b c y) :=
sorry

end highest_point_of_shell_l69_69779


namespace question1_question2_l69_69559

variables {θ : ℝ}

def a := (Real.cos θ, Real.sin θ)
def b := (2, -1)

-- Question 1
theorem question1 (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 / 3 := by
  sorry

-- Question 2
theorem question2 (h1 : (Real.cos θ - 2)^2 + (Real.sin θ + 1)^2 = 4) (h2 : θ ∈ Ioo 0 (Real.pi / 2)) :
  Real.sin (θ + Real.pi / 4) = 7 * Real.sqrt 2 / 10 := by
  sorry

end question1_question2_l69_69559


namespace frequency_of_score_range_l69_69589

theorem frequency_of_score_range (total_students : ℕ) (students_80_to_100 : ℕ) (htotal : total_students = 500) (hscore : students_80_to_100 = 180) : 
  (students_80_to_100 : ℝ) / (total_students : ℝ) = 0.36 :=
by {
  rw [htotal, hscore],
  norm_cast, -- Convert naturals to reals
  exact div_eq_iff_mul_eq.mpr (by norm_num),
}

end frequency_of_score_range_l69_69589


namespace polynomial_degree_12_l69_69097

noncomputable def polynomial_degree (x y z : ℕ) (a b c d e f : ℝ) := 
  (polynomial.monomial 5 x + polynomial.monomial 8 (a * x^8) + polynomial.monomial 2 (b * x^2) + polynomial.C c) *
  (polynomial.monomial 3 y + polynomial.monomial 2 (d * y^2) + polynomial.C e) *
  (polynomial.monomial 1 z + polynomial.C f)

theorem polynomial_degree_12 (x y z : ℕ) (a b c d e f : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) : 
  polynomial.degree (polynomial_degree x y z a b c d e f) = 12 := 
by
  -- proof steps would go here
  sorry

end polynomial_degree_12_l69_69097


namespace find_f_l69_69614

section TriangleDEF

-- Conditions
variables (d e : ℝ) (cos_diff_de : ℝ)
hypothesis h_d : d = 7
hypothesis h_e : e = 3
hypothesis h_cos_diff_de : cos_diff_de = 39 / 40

-- Goal
noncomputable def f : ℝ :=
  sqrt (58 + 42 * (39 / 40))

theorem find_f :
  f d e cos_diff_de = sqrt (9937) / 10 := 
by
  rw [f, h_d, h_e, h_cos_diff_de]
  sorry

end TriangleDEF

end find_f_l69_69614


namespace sqrt_eq_implies_range_l69_69202

theorem sqrt_eq_implies_range {x : ℝ} : sqrt ((1 - 2 * x) ^ 2) = 2 * x - 1 → x ≥ 1 / 2 := by
  sorry

end sqrt_eq_implies_range_l69_69202


namespace medians_sum_of_squares_l69_69802

-- Define the side lengths a, b, c
def a : ℝ := 13
def b : ℝ := 14
def c : ℝ := 15

-- Define the medians using Apollonius's theorem
def median_length (x y z : ℝ) : ℝ := (1 / 2) * Real.sqrt(2 * y^2 + 2 * z^2 - x^2)

def m_a := median_length a b c
def m_b := median_length b a c
def m_c := median_length c a b

-- Prove that the sum of the squares of the medians is 442.5
theorem medians_sum_of_squares : m_a^2 + m_b^2 + m_c^2 = 442.5 := by
  -- Proof is omitted
  sorry

end medians_sum_of_squares_l69_69802


namespace BD_length_l69_69278

theorem BD_length :
  ∀ (A B C D : Point) (r : ℝ),
    right_triangle A B C ∧
    angle B = 90 ∧
    tangent_circle A C D ∧
    tangent_circle B C D ∧
    triangle_area A B C = 216 ∧
    distance A C = 36 →
    distance B D = 12 :=
by
  intros A B C D r h
  sorry

end BD_length_l69_69278


namespace rank_leq_n_div_k_l69_69256

open Matrix

-- Define the problem conditions
variables {n k : ℕ} (A : Fin k → Matrix (Fin n) (Fin n) ℂ)
hypothesis (idempotent : ∀ i, A i @* @* A i = A i) -- Idempotent matrices
hypothesis (anti_commute : ∀ {i j : Fin k}, i < j → A i @* A j = -A i @* A j)

-- Define the goal
theorem rank_leq_n_div_k : ∃ i, rank (A i) ≤ n / k := 
sorry

end rank_leq_n_div_k_l69_69256


namespace hyperbola_asymptote_passing_through_point_l69_69916

theorem hyperbola_asymptote_passing_through_point (a : ℝ) (h_pos : a > 0) :
  (∃ m : ℝ, ∃ b : ℝ, ∀ x y : ℝ, y = m * x + b ∧ (x, y) = (2, 1) ∧ m = 2 / a) → a = 4 :=
by
  sorry

end hyperbola_asymptote_passing_through_point_l69_69916


namespace find_z_l69_69374

theorem find_z 
  (x : ℝ) (y : ℝ) (z : ℝ) 
  (h₁ : x = 100.48) 
  (h₂ : y = 100.70) 
  (h₃ : x * z = y^2) : 
  z ≈ 100.92 := 
by 
  -- Adding the proof is skipped here
  sorry

end find_z_l69_69374


namespace monic_quadratic_real_root_l69_69833

theorem monic_quadratic_real_root (a b : ℂ) (h : b = 2 - 3 * complex.I) :
  ∃ P : polynomial ℂ, P.monic ∧ P.coeff 2 = 1 ∧ P.coeff 1 = -4 ∧ P.coeff 0 = 13 ∧ P.is_root (2 - 3 * complex.I) :=
by
  sorry

end monic_quadratic_real_root_l69_69833


namespace amanda_candies_total_l69_69432

theorem amanda_candies_total :
  let initial_candies := 7
  let given_first_time := 3
  let additional_candies := 30
  let given_second_time := 4 * given_first_time
  let remaining_after_first := initial_candies - given_first_time
  let remaining_after_second := additional_candies - given_second_time
  let total_remaining := remaining_after_first + remaining_after_second
  total_remaining = 22 :=
by
  let initial_candies := 7
  let given_first_time := 3
  let additional_candies := 30
  let given_second_time := 4 * given_first_time
  let remaining_after_first := initial_candies - given_first_time
  let remaining_after_second := additional_candies - given_second_time
  let total_remaining := remaining_after_first + remaining_after_second
  show total_remaining = 22 from
  sorry

end amanda_candies_total_l69_69432


namespace value_of_sqrt_x_plus_one_over_sqrt_x_l69_69652

noncomputable def find_value (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) : ℝ :=
  sqrt(x) + 1/sqrt(x)

theorem value_of_sqrt_x_plus_one_over_sqrt_x (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) :
  find_value x hx_pos hx = 2 * sqrt(13) :=
sorry

end value_of_sqrt_x_plus_one_over_sqrt_x_l69_69652


namespace pentadecagon_triangle_count_l69_69958

-- Define the problem of selecting 3 vertices out of 15 to form a triangle
theorem pentadecagon_triangle_count : 
  ∃ (n : ℕ), n = nat.choose 15 3 ∧ n = 455 := 
by {
  sorry
}

end pentadecagon_triangle_count_l69_69958


namespace smallest_rational_number_is_neg_one_l69_69439

theorem smallest_rational_number_is_neg_one : 
  let a := -6 / 7
  let b := 2
  let c := 0
  let d := -1
  (min (min a b) (min c d)) = d := 
by 
  let a := -6 / 7
  let b := 2
  let c := 0
  let d := -1
  have h1 : a < 0 := by norm_num
  have h2 : d < 0 := by norm_num
  have h3 : c = 0 := by norm_num
  have h4 : b > 0 := by norm_num
  have h5 : abs a = 6 / 7 := by norm_cast
  have h6 : abs d = 1 := by norm_num
  have h7 : abs a < abs d := by norm_num
  have h8 : d < a := by linarith
  exact min_eq_right h8

end smallest_rational_number_is_neg_one_l69_69439


namespace real_roots_lg2_equation_l69_69633

noncomputable def greatest_int_le (y : ℝ) : ℤ :=
  if h : ∃ z : ℤ, ↑z ≥ y then 
    Classical.epsilon h 
  else 
    0

theorem real_roots_lg2_equation : 
  ∀ x : ℝ, (lg x)^2 - (greatest_int_le (lg x)) - 2 = 0 → 
  {x : ℝ // x > 0} :=
begin
  sorry -- Proof will be here
end

end real_roots_lg2_equation_l69_69633


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l69_69646

variable (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_52 : (Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52) :=
by
  sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l69_69646


namespace tangent_line_curve_l69_69212

variable {x a : ℝ}

theorem tangent_line_curve (h : ∀ x, (x ln x - x) = x - 2 * a) : a = (Math.exp 1) / 2 :=
by
  sorry

end tangent_line_curve_l69_69212


namespace value_of_sqrt_x_plus_one_over_sqrt_x_l69_69649

noncomputable def find_value (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) : ℝ :=
  sqrt(x) + 1/sqrt(x)

theorem value_of_sqrt_x_plus_one_over_sqrt_x (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) :
  find_value x hx_pos hx = 2 * sqrt(13) :=
sorry

end value_of_sqrt_x_plus_one_over_sqrt_x_l69_69649


namespace triangles_in_pentadecagon_l69_69979

theorem triangles_in_pentadecagon :
  let n := 15
  in (Nat.choose n 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69979


namespace exponent_of_4_l69_69737

theorem exponent_of_4 (x : ℕ) (h₁ : (1 / 4 : ℚ) ^ 2 = 1 / 16) (h₂ : 16384 * (1 / 16 : ℚ) = 1024) :
  4 ^ x = 1024 → x = 5 :=
by
  sorry

end exponent_of_4_l69_69737


namespace angle_LOR_measure_l69_69759

def RegularHeptagon := {angles : List ℝ // angles.length = 7 ∧ ∀ a ∈ angles, a = 128.57}

theorem angle_LOR_measure (H : RegularHeptagon) : 
  let L := H.1.head
  let M := H.1.nth 1
  let N := H.1.nth 2
  let O := H.1.nth 3
  let P := H.1.nth 4
  let Q := H.1.nth 5
  let R := H.1.nth 6
  ∃ angle_LOR : ℝ, angle_LOR = 25.71 :=
by
  sorry

end angle_LOR_measure_l69_69759


namespace number_of_true_propositions_l69_69270

variables {l : Line} {α β γ : Plane}

-- Define the perpendicularity between planes and lines
def perpendicular_planes (α β : Plane) : Prop := ∀ l : Line, (l ∈ α) → (l ∈ β) → l ⊥ α ∧ l ⊥ β
def parallel_line_plane (l : Line) (α : Plane) : Prop := l ∈ α ∧ ∃ m : Line, m ∈ α ∧ m ‖ β

-- Propositions
def prop1 (α β : Plane) : Prop := (perpendicular_planes α β) → ∃ l : Line, l ∈ α ∧ l ‖ β
def prop2 (α β : Plane) : Prop := (¬ perpendicular_planes α β) → ¬ ∃ l : Line, l ∈ α ∧ l ⊥ β
def prop3 (α β γ : Plane) (l : Line) : Prop := (α ⊥ γ) ∧ (β ⊥ γ) ∧ (l = α ∩ β) → l ⊥ γ

-- The theorem stating the number of true propositions
theorem number_of_true_propositions (l : Line) (α β γ : Plane) :
  prop1 α β ∧ prop2 α β ∧ prop3 α β γ l → 3 := 
by sorry -- Proof is skipped

end number_of_true_propositions_l69_69270


namespace range_of_a_l69_69207

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end range_of_a_l69_69207


namespace range_of_a_l69_69280

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2) : a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by
  sorry

end range_of_a_l69_69280


namespace general_term_formula_sum_of_first_n_terms_l69_69548

def arithmetic_sequence_condition (a : ℕ → ℝ) :=
  (log 10 (a 1) = 0) ∧ (log 10 (a 4) = 1)

def geometric_sequence_condition (a : ℕ → ℝ) (k : ℕ) :=
  a k ^ 2 = a 1 * a 6

theorem general_term_formula (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence_condition a →
  (∀ n : ℕ, a n = 3 * n - 2) :=
sorry

theorem sum_of_first_n_terms (a b S : ℕ → ℝ) (n k : ℕ) :
  arithmetic_sequence_condition a →
  geometric_sequence_condition a k →
  k = 2 →
  (∀ n : ℕ, b n = 4^(n-1)) →
  (∀ n : ℕ, S n = ∑ m in finset.range n, (a m + b m)) →
  (∀ n : ℕ, S n = (3/2) * n^2 - (1/2) * n + (1/3) * (4^n - 1)) :=
sorry

end general_term_formula_sum_of_first_n_terms_l69_69548


namespace find_primes_l69_69827

theorem find_primes (p : ℕ) (hp : Nat.Prime p) :
  (∃ a b c k : ℤ, a^2 + b^2 + c^2 = p ∧ a^4 + b^4 + c^4 = k * p) ↔ (p = 2 ∨ p = 3) :=
by
  sorry

end find_primes_l69_69827


namespace savings_value_l69_69686

def total_cost_individual (g : ℕ) (s : ℕ) : ℝ :=
  let cost_per_window := 120
  let cost (n : ℕ) : ℝ := 
    let paid_windows := n - (n / 6) -- one free window per five
    cost_per_window * paid_windows
  let discount (amount : ℝ) : ℝ :=
    if s > 10 then 0.95 * amount else amount
  discount (cost g) + discount (cost s)

def total_cost_joint (g : ℕ) (s : ℕ) : ℝ :=
  let cost_per_window := 120
  let n := g + s
  let paid_windows := n - (n / 6) -- one free window per five
  let joint_cost := cost_per_window * paid_windows
  if n > 10 then 0.95 * joint_cost else joint_cost

def savings (g : ℕ) (s : ℕ) : ℝ :=
  total_cost_individual g s - total_cost_joint g s

theorem savings_value (g s : ℕ) (hg : g = 9) (hs : s = 13) : savings g s = 162 := 
by 
  simp [savings, total_cost_individual, total_cost_joint, hg, hs]
  -- Detailed calculation is omitted, since it's not required according to the instructions.
  sorry

end savings_value_l69_69686


namespace distance_travelled_downstream_in_24_minutes_l69_69749

def speed_of_boat_still_water : ℝ := 20
def speed_of_current : ℝ := 3
def time_minutes : ℝ := 24
def effective_speed_downstream : ℝ := speed_of_boat_still_water + speed_of_current
def effective_speed_downstream_km_per_minute : ℝ := effective_speed_downstream / 60

theorem distance_travelled_downstream_in_24_minutes :
  (effective_speed_downstream_km_per_minute * time_minutes) = 9.2 := 
by
  sorry

end distance_travelled_downstream_in_24_minutes_l69_69749


namespace percentage_returned_l69_69068

theorem percentage_returned (R : ℕ) (S : ℕ) (total : ℕ) (least_on_lot : ℕ) (max_rented : ℕ)
  (h1 : total = 20) (h2 : least_on_lot = 10) (h3 : max_rented = 20) (h4 : R = 20) (h5 : S ≥ 10) :
  (S / R) * 100 ≥ 50 := sorry

end percentage_returned_l69_69068


namespace ellipse_major_axis_length_l69_69073

noncomputable def major_axis_length_ellipse (f1 f2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := f1
  let (x2, y2) := f2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem ellipse_major_axis_length :
  let f1 : ℝ × ℝ := (3, 15)
  let f2 : ℝ × ℝ := (28, 45)
  major_axis_length_ellipse (f1, f2) = Real.sqrt 1861 := 
by
  sorry

end ellipse_major_axis_length_l69_69073


namespace greg_first_day_rain_eq_3_l69_69741

/-- Greg's rain experience conditions -/
def first_day_rain (x : ℕ) : Prop :=
  let total_camping_rain := x + 6 + 5 in
  let total_house_rain := 26 in
  total_camping_rain = total_house_rain - 12

theorem greg_first_day_rain_eq_3 : first_day_rain 3 :=
by
  unfold first_day_rain
  simp
  sorry

end greg_first_day_rain_eq_3_l69_69741


namespace ferry_distance_l69_69321

theorem ferry_distance:
  let water_speed := 24
  let ferry_speed := 40
  let time_multiplier := 43 / 18
  let double_speed := ferry_speed * 2 in
  ∃ (x : ℝ), 
    (∀ y > x / 2, 
      43 / 18 * (x / (ferry_speed + water_speed)) = y / (ferry_speed + water_speed) + (x - y) / water_speed
    ) ∧
    (∀ y ≥ 0,
      x / (ferry_speed - water_speed) = (x - y) / water_speed + 1 + (y + 24) / double_speed
    ) → 
    x = 192 :=
begin
  sorry
end

end ferry_distance_l69_69321


namespace domain_of_f_l69_69468

noncomputable def f : ℝ → ℝ := λ x, 1 / (3 * x - 15)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = { x : ℝ | x ≠ 5 } :=
by
  sorry

end domain_of_f_l69_69468


namespace smallest_four_digit_integer_l69_69362

theorem smallest_four_digit_integer (n : ℤ) : 
  75 * n ≡ 225 [MOD 375] ∧ 1000 ≤ n ∧ n ≤ 9999 → n = 1003 :=
sorry

end smallest_four_digit_integer_l69_69362


namespace non_existence_complex_numbers_l69_69104

theorem non_existence_complex_numbers :
  ∀ (h : ℕ), ∀ (a b c : ℂ), (a ≠ 0) → (b ≠ 0) → (c ≠ 0) →
  ∃ (k l m : ℤ), |k| + |l| + |m| ≥ 1996 ∧ |1 + k * a + l * b + m * c| ≤ (1 : ℝ) / h :=
by {
  sorry -- The proof goes here
}

end non_existence_complex_numbers_l69_69104


namespace value_of_f_37_5_l69_69892

-- Mathematical definitions and conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f (x)
def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f (x)
def interval_condition (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f (x) = x

-- Main theorem to be proved
theorem value_of_f_37_5 (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_periodic : satisfies_condition f) 
  (h_interval : interval_condition f) : 
  f 37.5 = 0.5 := 
sorry

end value_of_f_37_5_l69_69892


namespace find_a_b_l69_69532

-- Define the conditions
def imaginary_unit : ℂ := Complex.i
def z : ℂ := ((1 + imaginary_unit) ^ 2 + 3 * (1 - imaginary_unit)) / (2 + imaginary_unit)
def equation (a b : ℝ) : Prop := (z ^ 2 + a * z + b = 1 + imaginary_unit)

-- The theorem statement to be proved
theorem find_a_b : ∃ a b : ℝ, (equation a b) ∧ a = -3 ∧ b = 4 := by
  sorry

end find_a_b_l69_69532


namespace min_value_inverse_ab_l69_69531

theorem min_value_inverse_ab (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 2 * a + b = 4) : 
  (∀ a b, a > 0 ∧ b > 0 ∧ 2 * a + b = 4 → (∃ v, v = \frac{1}{ab} ∧ ∀ w, w = \frac{1}{ab} → w ≥ min v)) :=
sorry

end min_value_inverse_ab_l69_69531


namespace pentadecagon_triangle_count_l69_69945

theorem pentadecagon_triangle_count :
  ∑ k in finset.range 15, if k = 3 then nat.choose 15 3 else 0 = 455 :=
by {
  sorry
}

end pentadecagon_triangle_count_l69_69945


namespace find_a_l69_69133

theorem find_a (a : ℝ)
    (h₁ : matrix.det ![ ![a, 1], ![3, 2] ] = matrix.det ![ ![a, 0], ![4, 1] ]) :
    a = 3 :=
by
    sorry

end find_a_l69_69133


namespace range_of_m_l69_69873

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 > 0
def B (x : ℝ) (m : ℝ) : Prop := 2 * m - 1 ≤ x ∧ x ≤ m + 3
def subset (B A : ℝ → Prop) : Prop := ∀ x, B x → A x

theorem range_of_m (m : ℝ) : (∀ x, B x m → A x) ↔ (m < -4 ∨ m > 2) :=
by 
  sorry

end range_of_m_l69_69873


namespace num_triangles_pentadecagon_l69_69991

/--
  The number of triangles that can be formed using the vertices of a regular pentadecagon
  (a 15-sided polygon where no three vertices are collinear) is 455.
-/
theorem num_triangles_pentadecagon : ∀ (n : ℕ), n = 15 → ∃ (num_triangles : ℕ), num_triangles = Nat.choose n 3 ∧ num_triangles = 455 :=
by
  intros n hn
  use Nat.choose n 3
  split
  · rfl
  · sorry

end num_triangles_pentadecagon_l69_69991


namespace danny_steve_ratio_l69_69464

theorem danny_steve_ratio :
  ∃ S : ℝ, (D = 25) ∧ (D / 2 + 12.5 = S / 2) ∧ (D / S = 1 / 2) :=
by
  let D := 25
  have h1 : D = 25 := rfl
  have h2 : D / 2 + 12.5 = 25 := by norm_num
  have h3 : 25 / S = 1 / 2 := sorry
  exact ⟨50, h1, h2, h3⟩

end danny_steve_ratio_l69_69464


namespace part1_part2_1_part2_2_l69_69886

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a - 2) * x + 4
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x + b - 3) / (a * x^2 + 2)

theorem part1 (a : ℝ) (b : ℝ) :
  (∀ x, f x a = f (-x) a) → b = 3 :=
by sorry

theorem part2_1 (a : ℝ) (b : ℝ) :
  a = 2 → b = 3 →
  ∀ x₁ x₂, -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧ x₁ < x₂ → g x₁ a b < g x₂ a b :=
by sorry

theorem part2_2 (a : ℝ) (b : ℝ) (t : ℝ) :
  a = 2 → b = 3 →
  g (t - 1) a b + g (2 * t) a b < 0 →
  0 < t ∧ t < 1 / 3 :=
by sorry

end part1_part2_1_part2_2_l69_69886


namespace common_root_discriminant_zero_l69_69731

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem common_root_discriminant_zero
  (p q r s : ℝ) 
  (f := λ x : ℝ, x^2 + p * x + q)
  (g := λ x : ℝ, x^2 + r * x + s)
  (common_root : ∃ a : ℝ, f a = 0 ∧ g a = 0)
  (discriminants_equal : discriminant 1 (p + r) (q + s) = discriminant 1 p q + discriminant 1 r s) :
  discriminant 1 p q = 0 ∨ discriminant 1 r s = 0 := 
sorry

end common_root_discriminant_zero_l69_69731


namespace remaining_ribbons_l69_69311

theorem remaining_ribbons (initial_A initial_B : ℕ) (used_A1 used_B1 gifts_1_4 used_A2 used_B2 gifts_5_8 : ℕ) :
  initial_A = 10 →
  initial_B = 12 →
  used_A1 = 2 →
  used_B1 = 1 →
  gifts_1_4 = 4 →
  used_A2 = 1 →
  used_B2 = 3 →
  gifts_5_8 = 4 →
  let total_used_A := (used_A1 * gifts_1_4) + (used_A2 * gifts_5_8),
      total_used_B := (used_B1 * gifts_1_4) + (used_B2 * gifts_5_8) in
  total_used_A ≤ initial_A → total_used_B ≤ initial_B →
  ∃ remaining_A remaining_B : ℕ,
    remaining_A = initial_A - total_used_A ∧
    remaining_B = initial_B - total_used_B :=
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8,
  let total_used_A := (used_A1 * gifts_1_4) + (used_A2 * gifts_5_8),
  let total_used_B := (used_B1 * gifts_1_4) + (used_B2 * gifts_5_8),
  have hA : total_used_A = 12, {
    simp [h3, h5, h6, h8],
    norm_num,
  },
  have hB : total_used_B = 16, {
    simp [h4, h5, h7, h8],
    norm_num,
  },
  intro hA_le,
  intro hB_le,
  have hA_inadequate : ¬total_used_A ≤ initial_A := by
  { rw [hA, h1], norm_num },
  have hB_inadequate : ¬total_used_B ≤ initial_B := by
  { rw [hB, h2], norm_num },
  contradiction
end

end remaining_ribbons_l69_69311


namespace framing_needed_is_12_feet_l69_69389

noncomputable def initial_width : ℝ := 5
noncomputable def initial_height : ℝ := 7
noncomputable def scale_factor : ℝ := 5
noncomputable def border_width : ℝ := 3
noncomputable def inches_to_feet : ℝ := 12

def minimum_framing_needed (initial_width initial_height scale_factor border_width inches_to_feet : ℝ) : ℝ :=
  let scaled_width := initial_width * scale_factor
  let scaled_height := initial_height * scale_factor
  let total_width := scaled_width + 2 * border_width
  let total_height := scaled_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  let perimeter_feet := perimeter_inches / inches_to_feet
  perimeter_feet

theorem framing_needed_is_12_feet :
  minimum_framing_needed initial_width initial_height scale_factor border_width inches_to_feet = 12 := by
  sorry

end framing_needed_is_12_feet_l69_69389


namespace glucose_poured_volume_l69_69402

noncomputable def glucose_solution_volume 
  (glucose_concentration : ℚ) 
  (total_solution_volume : ℚ) 
  (glucose_grams : ℚ) : ℚ :=
  (glucose_grams * total_solution_volume) / glucose_concentration

theorem glucose_poured_volume 
  (glucose_concentration : ℚ)
  (total_solution_volume : ℚ)
  (glucose_grams : ℚ)
  (V : ℚ) :
  glucose_concentration = 15 →
  total_solution_volume = 100 →
  glucose_grams = 9.75 →
  glucose_solution_volume glucose_concentration total_solution_volume glucose_grams = V →
  V = 65 :=
by
  intros h1 h2 h3 h4
  simp [glucose_solution_volume, h1, h2, h3] at h4
  assumption

end glucose_poured_volume_l69_69402


namespace sum_c_less_than_1_over_12_d_arithmetic_l69_69538

-- Definitions for sequences a_n and b_n
def a (n : ℕ) : ℕ := 3 * n + 1
def b (n : ℕ) : ℕ := 5 * n + 2
def c (n : ℕ) : ℚ := 1 / (a n * b n : ℚ)

-- Part 1: Summation less than 1/12
theorem sum_c_less_than_1_over_12 (n : ℕ) (h_n : 0 < n) : 
  (∑ i in Finset.range n, c (i + 1)) < 1 / 12 :=
sorry

-- Definitions for sequence d_n
def d_eq (n m : ℕ) : Prop := a n = b m

noncomputable def dn_seq (k : ℕ) : ℕ := 
  if h : ∃ n m, a n = b m ∧ a n = k then k else 0

-- Additional details for finding common terms, omitted for simplicity

-- Part 2: Proving d_n is arithmetic
theorem d_arithmetic : 
  ∃ (first term: ℕ) (common_diff: ℕ), ∀ k: ℕ, dn_seq (k + 1) = dn_seq k + common_diff :=
sorry

end sum_c_less_than_1_over_12_d_arithmetic_l69_69538


namespace shape_described_by_theta_eq_c_is_plane_l69_69502

-- Definitions based on conditions in the problem
def spherical_coordinates (ρ θ φ : ℝ) := true

def is_plane_condition (θ c : ℝ) := θ = c

-- Statement to prove
theorem shape_described_by_theta_eq_c_is_plane (c : ℝ) :
  ∀ ρ θ φ : ℝ, spherical_coordinates ρ θ φ → is_plane_condition θ c → "Plane" = "Plane" :=
by sorry

end shape_described_by_theta_eq_c_is_plane_l69_69502


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l69_69644

variable (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_52 : (Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52) :=
by
  sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l69_69644


namespace range_of_a_l69_69268

theorem range_of_a (f : ℝ → ℝ)
  (hf_even : ∀ x : ℝ, f x = f (-x))
  (hf_increasing : ∀ {x y : ℝ}, x < 0 → y < 0 → x < y → f x < f y)
  (h_condition : ∀ (a : ℝ), f (2 * a ^ 2 + a + 1) < f (2 * a ^ 2 - 2 * a + 3)) : 
  ∀ a : ℝ, 3 * a - 2 > 0 → a > 2 / 3 :=
begin
  sorry
end

end range_of_a_l69_69268


namespace polynomial_degree_one_l69_69881

noncomputable def hasFirstDegree (P : Polynomial ℝ) : Prop :=
  P.degree = 1

theorem polynomial_degree_one
  (P : Polynomial ℝ)
  (a : ℕ → ℕ)
  (h_distinct: ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (h_inf: ∀ n : ℕ, ∃ m > n, a m > a n)
  (h_seq: ∀ n : ℕ, P (a (n + 1)) = a n) :
  hasFirstDegree P := 
sorry

end polynomial_degree_one_l69_69881


namespace four_digit_palindromic_perfect_square_count_l69_69044

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69044


namespace nicky_running_time_l69_69375

theorem nicky_running_time :
  ∀ (head_start : ℕ) (cristina_speed nicky_speed : ℕ),
  (head_start = 12) →
  (cristina_speed = 5) →
  (nicky_speed = 3) →
  nicky_running_time = head_start + (36 / (cristina_speed - nicky_speed)) :=
begin
  sorry
end

end nicky_running_time_l69_69375


namespace num_four_digit_palindromic_squares_l69_69018

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69018


namespace projection_of_vector_a_on_vector_b_l69_69345

variable (a b : ℝ × ℝ)
variable (ha : a = (-1, 3))
variable (hb : b = (3, -4))

def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2
def norm (v : ℝ × ℝ) := Real.sqrt (v.1 * v.1 + v.2 * v.2)
def proj (u v : ℝ × ℝ) := dot_product u v / norm v

theorem projection_of_vector_a_on_vector_b : proj a b = -3 :=
by
  rw [proj, dot_product, ha, hb]
  simp
  norm_num
  sorry

end projection_of_vector_a_on_vector_b_l69_69345


namespace find_function_l69_69696

noncomputable def f (x : ℝ) : ℝ := (5/7) * (4^x - 3^x)

theorem find_function (x y : ℝ) :
  f(2) = 5 ∧ (∀ x y : ℝ, f(x + y) = 4^y * f(x) + 3^x * f(y)) :=
by {
  sorry
}

end find_function_l69_69696


namespace evaluate_expression_at_two_l69_69453

theorem evaluate_expression_at_two :
  let x := 2 in
  3 * x^2 - 4 * x + 2 = 6 :=
by
  sorry

end evaluate_expression_at_two_l69_69453


namespace Marta_max_piles_l69_69286

theorem Marta_max_piles (a b c : ℕ) (ha : a = 42) (hb : b = 60) (hc : c = 90) : 
  Nat.gcd (Nat.gcd a b) c = 6 := by
  rw [ha, hb, hc]
  have h : Nat.gcd (Nat.gcd 42 60) 90 = Nat.gcd 6 90 := by sorry
  exact h    

end Marta_max_piles_l69_69286


namespace min_distance_parabola_l69_69879

/-- Given a parabola \( y^2 = 4x \) and a point \( P \) on it, 
let \( d_1 \) be the distance from point \( P \) to the parabola's axis of symmetry, 
and \( d_2 \) the distance to the line \( 3x - 4y + 9 = 0 \). 
Prove that the minimum value of \( d_1 + d_2 \) is \( \frac{12}{5} \). -/
theorem min_distance_parabola :
  let P : ℝ × ℝ := (x, y)
  let d1 := 1 + x 
  let d2 := |3*x - 8*Real.sqrt x + 9| / Real.sqrt(9 + 16)
  y^2 = 4 * x → 
  ∃ d : ℝ, (d = d1 + d2) ∧ d = 12 / 5 :=
by
  sorry

end min_distance_parabola_l69_69879


namespace probability_of_selecting_GEARS_letter_l69_69476

def bag : List Char := ['A', 'L', 'G', 'E', 'B', 'R', 'A', 'S']
def target_word : List Char := ['G', 'E', 'A', 'R', 'S']

theorem probability_of_selecting_GEARS_letter :
  (6 : ℚ) / 8 = 3 / 4 :=
by
  sorry

end probability_of_selecting_GEARS_letter_l69_69476


namespace smallest_n_with_16_divisors_has_16_divisors_l69_69738

noncomputable def smallest_n_with_16_divisors : ℕ :=
  120

theorem smallest_n_with_16_divisors_has_16_divisors :
  ∃ (n : ℕ), (∀ m : ℕ, m < n → ∃ d, d ∣ m) ∧ n = 120 ∧ (∀ d : ℕ, d ∣ 120 → ∃ e : ℕ, d = 2 ^ (e.1) * 3 ^ (e.2) * 5 ^ (e.3)) ∧ (2 + 1) * (1 + 1) * (1 + 1) = 16 :=
sorry

end smallest_n_with_16_divisors_has_16_divisors_l69_69738


namespace tiger_population_2006_l69_69706

theorem tiger_population_2006 
  (p : ℕ → ℝ) 
  (k : ℝ) 
  (h1 : ∀ n, p(n + 2) - p(n) = k * p(n + 1))
  (h2004 : p 2004 = 120)
  (h2005 : p 2005 = 220)
  (h2007 : p 2007 = 500) 
  : p 2006 = 315 :=
by
  sorry

end tiger_population_2006_l69_69706


namespace four_digit_palindrome_square_count_l69_69001

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l69_69001


namespace survey_is_sample_of_population_l69_69787

-- Definitions based on the conditions in a)
def population_size := 50000
def sample_size := 2000
def is_comprehensive_survey := false
def is_sampling_survey := true
def is_population_student (n : ℕ) : Prop := n ≤ population_size
def is_individual_unit (n : ℕ) : Prop := n ≤ sample_size

-- Theorem that encapsulates the proof problem
theorem survey_is_sample_of_population : is_sampling_survey ∧ ∃ n, is_individual_unit n :=
by
  sorry

end survey_is_sample_of_population_l69_69787


namespace find_foci_l69_69816

noncomputable def hyperbola_foci : Prop :=
  let a_sq := 15
  let b_sq := 12
  let c := Real.sqrt (a_sq + b_sq)
  foci := (0, -c) ∨ foci := (0, c)
  
  
theorem find_foci :
  ∃ foci : ℝ × ℝ, 
  let a_sq := 15
  let b_sq := 12
  let c := Real.sqrt (a_sq + b_sq)
  foci = (0, c) ∨ foci = (0, -c) := 
begin
  sorry
end

end find_foci_l69_69816


namespace triangles_in_pentadecagon_l69_69954

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69954


namespace balls_into_boxes_l69_69565

theorem balls_into_boxes (num_balls : ℕ) (num_boxes : ℕ) (h_balls : num_balls = 4) (h_boxes : num_boxes = 2) :
  num_ways_put_balls_into_boxes num_balls num_boxes = 8 :=
by
  sorry

end balls_into_boxes_l69_69565


namespace num_triangles_pentadecagon_l69_69992

/--
  The number of triangles that can be formed using the vertices of a regular pentadecagon
  (a 15-sided polygon where no three vertices are collinear) is 455.
-/
theorem num_triangles_pentadecagon : ∀ (n : ℕ), n = 15 → ∃ (num_triangles : ℕ), num_triangles = Nat.choose n 3 ∧ num_triangles = 455 :=
by
  intros n hn
  use Nat.choose n 3
  split
  · rfl
  · sorry

end num_triangles_pentadecagon_l69_69992


namespace part_one_part_two_l69_69911

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ln (-x) + a*x - 1/x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := ln x + 1/x + 2*x

theorem part_one (a : ℝ) :
  (∃ x, x < 0 ∧ deriv (f a) x = 0) ↔ a = 0 := by sorry

theorem part_two :
  g 0 (1/2) = 3 - ln 2 := by sorry

end part_one_part_two_l69_69911


namespace find_x_y_sum_of_squares_l69_69110

theorem find_x_y_sum_of_squares :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (xy + x + y = 47) ∧ (x^2 * y + x * y^2 = 506) ∧ (x^2 + y^2 = 101) :=
by {
  sorry
}

end find_x_y_sum_of_squares_l69_69110


namespace triangle_segment_inequality_l69_69243

variables {A B C M N : Type}
variables [IsTriangle A B C] [IsSegment B C]
variables [AngleTrisection AN AM A B C]   -- Assuming the definition for angle trisection exists
variables (angle_ge_90 : Angle B A C ≥ 90)

theorem triangle_segment_inequality (h : Angle_trisected_by AN AM A B C) : BM < MN ∧ MN < NC :=
by
  -- Required proof here
  sorry

end triangle_segment_inequality_l69_69243


namespace factorize_expression_l69_69490

variable {a x y : ℝ}

theorem factorize_expression : (a * x^2 + 2 * a * x * y + a * y^2) = a * (x + y)^2 := by
  sorry

end factorize_expression_l69_69490


namespace next_sales_amount_l69_69048

theorem next_sales_amount (royalties_first : ℝ) (sales_first : ℝ) (royalties_next : ℝ) (percentage_decrease : ℝ) :
  royalties_first = 8 ∧
  sales_first = 20 ∧
  royalties_next = 9 ∧
  percentage_decrease = 79.16666666666667 →
  let original_rate := royalties_first / sales_first,
      new_rate := royalties_next / S,
      decreased_rate := original_rate * (1 - percentage_decrease / 100)
  in  new_rate = decreased_rate →
      S = 108 := sorry

end next_sales_amount_l69_69048


namespace length_of_each_train_l69_69380

-- Conditions
def faster_train_speed_kmh : ℝ := 45 -- km/hr
def slower_train_speed_kmh : ℝ := 36 -- km/hr
def passing_time_sec : ℝ := 36 -- seconds
def relative_speed_ms : ℝ := (faster_train_speed_kmh - slower_train_speed_kmh) * (1000 / 3600) -- m/s

-- Question and proof statement
theorem length_of_each_train
  (faster_train_speed_kmh : ℝ := 45)
  (slower_train_speed_kmh : ℝ := 36)
  (passing_time_sec : ℝ := 36)
  : ∃ L : ℝ, L = 45 :=
by
  let relative_speed_ms : ℝ := (faster_train_speed_kmh - slower_train_speed_kmh) * (1000 / 3600)
  let distance := relative_speed_ms * passing_time_sec
  have length_of_each_train := distance / 2
  exists length_of_each_train
  exact sorry

end length_of_each_train_l69_69380


namespace difference_between_angles_of_parallelogram_l69_69330

-- Define a parallelogram and its angles
structure Parallelogram where
  angleA : ℝ -- Smaller interior angle
  angleB : ℝ -- Larger interior angle
  (angle_sum : angleA + angleB = 180) -- Adjacent angles are supplementary

noncomputable def parallelogram_difference (P : Parallelogram) : ℝ :=
  P.angleB - P.angleA

-- Given condition
def given_parallelogram : Parallelogram :=
  { angleA := 45,
    angleB := 135,
    angle_sum := by norm_num }

theorem difference_between_angles_of_parallelogram (P : Parallelogram) (h : P.angleA = 45) : parallelogram_difference P = 90 :=
by
  rw [h, parallelogram_difference]
  -- Here P.angleB is supposed to be 135 because given P.angle_sum = 180,
  -- angleB = 180 - angleA = 180 - 45 = 135.
  have : P.angleB = 180 - P.angleA := by linarith
  rw [this, h]
  norm_num
  sorry  -- Proof ends with sorry for now

end difference_between_angles_of_parallelogram_l69_69330


namespace find_m_solve_inequality_l69_69185

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 2|

theorem find_m (m : ℝ) : (∀ x : ℝ, m - |x| ≥ 0 ↔ x ∈ [-1, 1]) → m = 1 :=
by
  sorry

theorem solve_inequality (x : ℝ) : |x + 1| + |x - 2| > 4 * 1 ↔ x < -3 / 2 ∨ x > 5 / 2 :=
by
  sorry

end find_m_solve_inequality_l69_69185


namespace largest_tan_PAS_l69_69352

def Triangle (A B C : Type) := ∃ (a b c : ℝ), a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0

structure Point (ℝ : Type) :=
(x : ℝ)
(y : ℝ)

def midpoint (A B : Point ℝ) : Point ℝ :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

def tan (θ : ℝ) : ℝ := sorry -- Define tangent function properly

theorem largest_tan_PAS (P Q R A S : Point ℝ)
  (h_triangle : Triangle P Q R)
  (h_angle_R : ∃ (θ : ℝ), θ = 45)
  (h_QR : ∃ (d : ℝ), d = 5)
  (h_midpoint : S = midpoint Q R) :
  ∃ t, t = (3 * real.sqrt 6) / (10 * (real.sqrt 3 - 1)) := 
sorry

end largest_tan_PAS_l69_69352


namespace smallest_rat_num_l69_69436

theorem smallest_rat_num (a b c d : ℚ) (ha : a = -6 / 7) (hb : b = 2) (hc : c = 0) (hd : d = -1) :
  min (min a (min b c)) d = -1 :=
sorry

end smallest_rat_num_l69_69436


namespace arithmetic_sequence_sum_l69_69229

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 3 + a 9 + a 15 + a 21 = 8) :
  a 1 + a 23 = 4 :=
sorry

end arithmetic_sequence_sum_l69_69229


namespace triangles_in_pentadecagon_l69_69978

theorem triangles_in_pentadecagon :
  let n := 15
  in (Nat.choose n 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69978


namespace cos_2theta_l69_69889

variable (θ : ℝ)

theorem cos_2theta (h : 2^(1 - 3 * (Math.cos θ)) + 1 = 2^((1 / 2) - (Math.cos θ))) :
  Math.cos (2 * θ) = -17 / 18 :=
sorry

end cos_2theta_l69_69889


namespace hyperbola_eccentricity_l69_69145

variable (a b : ℝ) (ha : a > 0) (hb : b > 0)

def hyperbola := 
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

def circle := 
  ∀ x y : ℝ, (x - a)^2 + y^2 = b^2 / 4

theorem hyperbola_eccentricity 
  (H1 : hyperbola a b ha hb) 
  (H2 : circle a b ha hb) 
  : eccentricity = 2 :=
sorry

end hyperbola_eccentricity_l69_69145


namespace part1_min_value_part2_find_b_part3_range_b_div_a_l69_69194

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 - abs (a*x - b)

-- Part (1)
theorem part1_min_value : f 1 1 1 = -5/4 :=
by 
  sorry

-- Part (2)
theorem part2_find_b (b : ℝ) (h : b ≥ 2) (h_domain : ∀ x, 1 ≤ x ∧ x ≤ b) (h_range : ∀ y, 1 ≤ y ∧ y ≤ b) : 
  b = 2 :=
by 
  sorry

-- Part (3)
theorem part3_range_b_div_a (a b : ℝ) (h_distinct : (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x a b = 1 ∧ ∀ y : ℝ, 0 < y ∧ y < 2 ∧ f y a b = 1 ∧ x ≠ y)) : 
  1 < b / a ∧ b / a < 2 :=
by 
  sorry

end part1_min_value_part2_find_b_part3_range_b_div_a_l69_69194


namespace m_power_of_prime_no_m_a_k_l69_69387

-- Part (i)
theorem m_power_of_prime (m : ℕ) (p : ℕ) (k : ℕ) (h1 : m ≥ 1) (h2 : Prime p) (h3 : m * (m + 1) = p^k) : m = 1 :=
by sorry

-- Part (ii)
theorem no_m_a_k (m a k : ℕ) (h1 : m ≥ 1) (h2 : a ≥ 1) (h3 : k ≥ 2) (h4 : m * (m + 1) = a^k) : False :=
by sorry

end m_power_of_prime_no_m_a_k_l69_69387


namespace cos_double_angle_l69_69874

theorem cos_double_angle (x : ℝ) (h : Real.sin (x + Real.pi / 2) = 1 / 3) : Real.cos (2 * x) = -7 / 9 :=
sorry

end cos_double_angle_l69_69874


namespace max_apartment_size_l69_69075

theorem max_apartment_size (rate cost per_sqft : ℝ) (budget : ℝ) (h1 : rate = 1.20) (h2 : budget = 864) : cost = 720 :=
by
  sorry

end max_apartment_size_l69_69075


namespace simplify_expression_l69_69314

variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 9) - (x + 6) * (3 * x - 2) = 7 * x - 24 :=
by
  sorry

end simplify_expression_l69_69314


namespace value_of_N3_l69_69238

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def meets_conditions (n : ℕ) : Prop :=
  (32 * 5 * n).digits 10 |>.length = 1992 
  ∧ ∀ d ∈ [1, 2, 3, 4, 5, 6, 7, 8], (∃ k, (32 * 5 * n).digits.count d = 9 * k)

theorem value_of_N3 (N : ℕ) (h : meets_conditions N) : sum_of_digits (sum_of_digits (sum_of_digits N)) = 9 :=
sorry

end value_of_N3_l69_69238


namespace max_min_A_l69_69176

noncomputable def A (x y : ℝ) : ℝ :=
  let z := x + y * complex.I
  x * (complex.abs (z - complex.I))^2 - 1

theorem max_min_A (x y : ℝ) (h : complex.abs (complex.mk x y - complex.I) ≤ 1) :
  max ((x * ((complex.abs (complex.mk x y - complex.I))^2 - 1))) = (2 * real.sqrt 3 / 9) ∧ 
  min ((x * ((complex.abs (complex.mk x y - complex.I))^2 - 1))) = -(2 * real.sqrt 3 / 9) :=
sorry

end max_min_A_l69_69176


namespace digits_in_8_pow_12_3_pow_25_l69_69801

theorem digits_in_8_pow_12_3_pow_25 :
  ∀ (x y : ℕ), x = 8 → y = 12 → ∃ d : ℕ, nat.log10 (8^12 * 3^25) + 1 = 23 :=
by
  intros x y hx hy
  sorry

end digits_in_8_pow_12_3_pow_25_l69_69801


namespace min_cookies_to_receive_l69_69665

def minimum_cookies_strategy (n : ℕ) : ℕ := 2^n

theorem min_cookies_to_receive (n : ℕ) (m : ℕ) (initial_distribution : fin (2 * n) → ℕ)
  (valid_moves : ∀ (p : fin (2 * n)), initial_distribution p = 0 ∨ ∃ q, 
    (q = (p + 1) % (2 * n) ∨ q = (p - 1 + 2 * n) % (2 * n)) ∧ initial_distribution p > initial_distribution q + 1 ∧
    initial_distribution (p - 1) < initial_distribution p + 1) : 
  ∃ (moves : fin (2 * n) → ℕ → fin (2 * n) × (fin (2 * n))), 
    (∀ i, valid_moves i) ∧ 
    initial_distribution 0 > 0 → 
    m >= minimum_cookies_strategy n :=
by 
    sorry

end min_cookies_to_receive_l69_69665


namespace smallest_rational_number_is_neg_one_l69_69438

theorem smallest_rational_number_is_neg_one : 
  let a := -6 / 7
  let b := 2
  let c := 0
  let d := -1
  (min (min a b) (min c d)) = d := 
by 
  let a := -6 / 7
  let b := 2
  let c := 0
  let d := -1
  have h1 : a < 0 := by norm_num
  have h2 : d < 0 := by norm_num
  have h3 : c = 0 := by norm_num
  have h4 : b > 0 := by norm_num
  have h5 : abs a = 6 / 7 := by norm_cast
  have h6 : abs d = 1 := by norm_num
  have h7 : abs a < abs d := by norm_num
  have h8 : d < a := by linarith
  exact min_eq_right h8

end smallest_rational_number_is_neg_one_l69_69438


namespace village_population_l69_69760

theorem village_population (P : ℕ) (h : 0.6 * P = 23040) : P = 38400 := by
  sorry

end village_population_l69_69760


namespace walking_speed_correct_l69_69776

noncomputable def walking_speed (W : ℕ) : Prop := 
  ∃ W : ℕ, 
    (∀ r : ℕ, r = 8) ∧ 
    (∀ d : ℕ, d = 16) ∧ 
    (∀ t : ℝ, t = 3) ∧ 
    (d / 2 / W + r / 8 = t)

theorem walking_speed_correct : walking_speed 4 :=
by 
  unfold walking_speed 
  use 4
  split
  · intro r 
    exact rfl
  split
  · intro d 
    exact rfl
  split
  · intro t 
    exact rfl
  ring_nf 
  norm_num
  sorry

end walking_speed_correct_l69_69776


namespace find_x_l69_69732

variables (x a b : ℝ)
variable (hneq : a ≠ b)
variable (heq : (1 / 2 * a * sqrt (x^2 - (a / 2)^2)) = (1 / 2 * b * sqrt (x^2 - (b / 2)^2)))

theorem find_x (hneq : a ≠ b) (heq : (1 / 2 * a * sqrt (x^2 - (a / 2)^2)) = (1 / 2 * b * sqrt (x^2 - (b / 2)^2))) : 
  x = sqrt (a^2 + b^2) / 2 :=
by
  sorry

end find_x_l69_69732


namespace four_digit_palindromic_perfect_square_count_l69_69039

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69039


namespace expected_value_ten_sided_die_l69_69062

noncomputable def expected_value (die_faces : Finset ℕ) : ℚ :=
  (∑ i in die_faces, (i : ℚ)) / die_faces.card

theorem expected_value_ten_sided_die :
  expected_value (Finset.range 10 \ {0}) = 5.5 := by
  sorry

end expected_value_ten_sided_die_l69_69062


namespace exists_convex_body_projection_l69_69245

theorem exists_convex_body_projection :
  ∃ (C : set (ℝ^3)), 
    convex C ∧ orthogonal_projection C (λ (x y z : ℝ), (y, z)) = set_of (λ (y z : ℝ), y^2 + z^2 ≤ 1) ∧
    orthogonal_projection C (λ (x y z : ℝ), (x, z)) = set_of (λ (x z : ℝ), x^2 + z^2 ≤ 1) ∧
    orthogonal_projection C (λ (x y z : ℝ), (x, y)) = set_of (λ (x y : ℝ), x^2 + y^2 ≤ 1) ∧
    (C ≠ set_of (λ (x y z : ℝ), x^2 + y^2 + z^2 ≤ 1)) :=
begin
  sorry
end

end exists_convex_body_projection_l69_69245


namespace capital_ratio_l69_69445

-- Defining the variables for the losses
def total_loss := 1000
def Pyarelal_loss := 900
def Ashok_loss := total_loss - Pyarelal_loss

-- Statement to prove that the ratio of Ashok's capital to Pyarelal's capital is 1 : 9
theorem capital_ratio (total_loss Pyarelal_loss : ℕ) (h1 : total_loss = 1000) (h2 : Pyarelal_loss = 900) :
  let Ashok_loss := total_loss - Pyarelal_loss in
  Ashok_loss / Pyarelal_loss = 1 / 9 :=
by
  sorry

end capital_ratio_l69_69445


namespace sum_slope_y_intercept_l69_69235

-- Define the points A, B, C, D, and F based on the problem conditions
def A : (ℝ × ℝ) := (0, 8)
def B : (ℝ × ℝ) := (0, 0)
def C : (ℝ × ℝ) := (10, 0)
def D : (ℝ × ℝ) := (0, 4)
def F : (ℝ × ℝ) := (20 / 3, 8 / 3)

-- Define the line passing through points C and D
def lineCD (x : ℝ) : ℝ := -0.4 * x + 4

-- The sum of the slope and y-intercept of line passing through C and D is 3.6
theorem sum_slope_y_intercept : -0.4 + 4 = 3.6 := by
  sorry

end sum_slope_y_intercept_l69_69235


namespace find_x_squared_minus_y_squared_l69_69903

variable (x y : ℝ)

theorem find_x_squared_minus_y_squared 
(h1 : y + 6 = (x - 3)^2)
(h2 : x + 6 = (y - 3)^2)
(h3 : x ≠ y) :
x^2 - y^2 = 27 := by
  sorry

end find_x_squared_minus_y_squared_l69_69903


namespace graph_not_in_first_quadrant_l69_69580

theorem graph_not_in_first_quadrant (a b : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) 
  (h_not_in_first_quadrant : ∀ x : ℝ, a^x + b - 1 ≤ 0) : 
  0 < a ∧ a < 1 ∧ b ≤ 0 :=
sorry

end graph_not_in_first_quadrant_l69_69580


namespace ellipse_other_intersection_point_l69_69072

theorem ellipse_other_intersection_point :
  let F1 := (0, 3)
  let F2 := (4, 0)
  let P := (5, 0)
  let total_distance := @euclidean_distance ℝ (0, 3) (5, 0) + @euclidean_distance ℝ (4, 0) (5, 0)
  ∃ x : ℝ, (x = 91 / 20) ∧ total_distance = 6 :=
begin
  let F1 := (0, 3),
  let F2 := (4, 0),
  let P := (5, 0),
  let distance := @euclidean_distance ℝ,
  have h : distance F1 P + distance F2 P = 6,
  { simp [distance, F1, F2, P] },
  use 91 / 20,
  rw [eq.symm (by simp [distance, F1, F2, mk_eq])],
end

end ellipse_other_intersection_point_l69_69072


namespace proof_l69_69279

noncomputable def question (a b c : ℂ) : ℂ := (a^3 + b^3 + c^3) / (a * b * c)

theorem proof (a b c : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 15)
  (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2 * a * b * c) :
  question a b c = 18 :=
by
  sorry

end proof_l69_69279


namespace no_solution_intervals_l69_69828

theorem no_solution_intervals (a : ℝ) :
  (a < -17 ∨ a > 0) → ¬∃ x : ℝ, 7 * |x - 4 * a| + |x - a^2| + 6 * x - 3 * a = 0 :=
by
  sorry

end no_solution_intervals_l69_69828


namespace num_four_digit_palindromic_squares_l69_69023

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69023


namespace finite_prime_factors_l69_69634

theorem finite_prime_factors 
  (a : ℕ → ℕ) 
  (h1 : ∀ ⦃S : Finset ℕ⦄, S.Nonempty → Nat.prime (-1 + ∏ k in S, a k)) 
  (m : ℕ) : 
  {k | Nat.card (Nat.factors (a k)) < m}.finite :=
sorry

end finite_prime_factors_l69_69634


namespace transformed_avg_is_4_l69_69534

variable {n : ℕ}
variable {x : Fin n → ℝ}

-- If the average of the original data is 2
def avg_is_2 (x : Fin n → ℝ) : Prop := (∑ i, x i) / n = 2

-- Prove that the new average is 4
theorem transformed_avg_is_4 (h : avg_is_2 x) :
  (∑ i, x i + 2) / n = 4 :=
sorry

end transformed_avg_is_4_l69_69534


namespace maximize_ab_l69_69875

theorem maximize_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ab + a + b = 1) : 
  ab ≤ 3 - 2 * Real.sqrt 2 :=
sorry

end maximize_ab_l69_69875


namespace evan_45_l69_69259

theorem evan_45 (k n : ℤ) (h1 : n + (k * (2 * k - 1)) = 60) : 60 - n = 45 :=
by sorry

end evan_45_l69_69259


namespace area_of_quadrilateral_l69_69496

-- Define the conditions
def diagonal : ℝ := 20
def offset1 : ℝ := 5
def offset2 : ℝ := 4

-- Define the problem
theorem area_of_quadrilateral (diag : ℝ) (off1 : ℝ) (off2 : ℝ) (h_diag : diag = diagonal) (h_off1 : off1 = offset1) (h_off2 : off2 = offset2) :
  let area1 := (1 / 2) * diag * off1
  let area2 := (1 / 2) * diag * off2 in
  area1 + area2 = 90 := 
sorry

end area_of_quadrilateral_l69_69496


namespace subset_sum_divisibility_l69_69260

theorem subset_sum_divisibility (p : ℕ) (A : ℕ → ℕ)
    (hp_prime : Nat.Prime p)
    (hp_odd : p % 2 = 1)
    (h2_pow_p_minus_1 : ¬p^2 ∣ 2^(p-1) - 1) :
    ∀ k : ℤ, ∃ m : ℕ, ∃ᶠ m in filter.atTop, (A m - k) % p = 0 := 
sorry

end subset_sum_divisibility_l69_69260


namespace revenue_effect_l69_69376

noncomputable def price_increase_factor : ℝ := 1.425
noncomputable def sales_decrease_factor : ℝ := 0.627

theorem revenue_effect (P Q R_new : ℝ) (h_price_increase : P ≠ 0) (h_sales_decrease : Q ≠ 0) :
  R_new = (P * price_increase_factor) * (Q * sales_decrease_factor) →
  ((R_new - P * Q) / (P * Q)) * 100 = -10.6825 :=
by
  sorry

end revenue_effect_l69_69376


namespace cos_alpha_minus_beta_l69_69558

variable {α β : ℝ}
def a : ℝ × ℝ := (Real.cos α, Real.sin α)
def b : ℝ × ℝ := (Real.cos β, Real.sin β)

noncomputable def vec_diff_norm : ℝ :=
  (a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2

theorem cos_alpha_minus_beta :
  (vec_diff_norm = (2 * Real.sqrt 2 / 5) ^ 2) → (Real.cos (α - β) = 21 / 25) :=
by
  sorry

end cos_alpha_minus_beta_l69_69558


namespace new_boarders_l69_69711

theorem new_boarders (init_boarders : ℕ) (init_day_students : ℕ) (ratio_b : ℕ) (ratio_d : ℕ) (ratio_new_b : ℕ) (ratio_new_d : ℕ) (x : ℕ) :
    init_boarders = 240 →
    ratio_b = 8 →
    ratio_d = 17 →
    ratio_new_b = 3 →
    ratio_new_d = 7 →
    init_day_students = (init_boarders * ratio_d) / ratio_b →
    (ratio_new_b * init_day_students) = ratio_new_d * (init_boarders + x) →
    x = 21 :=
by sorry

end new_boarders_l69_69711


namespace initial_cats_count_l69_69777

theorem initial_cats_count :
  ∀ (initial_birds initial_puppies initial_spiders final_total initial_cats: ℕ),
    initial_birds = 12 →
    initial_puppies = 9 →
    initial_spiders = 15 →
    final_total = 25 →
    (initial_birds / 2 + initial_puppies - 3 + initial_spiders - 7 + initial_cats = final_total) →
    initial_cats = 5 := by
  intros initial_birds initial_puppies initial_spiders final_total initial_cats h1 h2 h3 h4 h5
  sorry

end initial_cats_count_l69_69777


namespace opposite_face_of_silver_is_orange_l69_69399

-- Definitions for the colors
inductive Color
| Blue | Orange | Black | Yellow | Silver | Violet

open Color

-- Definitions for the faces
inductive Face
| Top | Bottom | Front | Back | Left | Right

open Face

-- Definitions for the views
structure View :=
(top : Face)
(front : Face)
(right : Face)
(colors : Face -> Color)

-- Conditions as Views
def view1 : View :=
{ top := Top, front := Front, right := Right, colors := function
  | Top    => Blue
  | Front  => Yellow
  | Right  => Violet
  | _      => sorry -- remaining faces }

def view2 : View :=
{ top := Top, front := Front, right := Right, colors := function
  | Top    => Blue
  | Front  => Silver
  | Right  => Violet
  | _      => sorry -- remaining faces }
  
def view3 : View :=
{ top := Top, front := Front, right := Right, colors := function
  | Top    => Blue
  | Front  => Black
  | Right  => Violet
  | _      => sorry -- remaining faces }

-- The theorem to prove
theorem opposite_face_of_silver_is_orange :
  (∀ v, (v = view1 ∨ v = view2 ∨ v = view3) →
  v.colors Top = Blue →
  v.colors Right = Violet →
  ∃ f, v.colors f = Silver → v.colors (opposite_face f) = Orange) :=
sorry

end opposite_face_of_silver_is_orange_l69_69399


namespace seatingArrangementsCorrect_l69_69762

-- Defining the conditions
def numDemocrats : ℕ := 6
def numRepublicans : ℕ := 4

-- A function to determine the number of valid seating arrangements around a circular table
noncomputable def countValidArrangements : ℕ :=
  let democratsArrangements := (numDemocrats - 1)!
  let gaps := numDemocrats
  let chooseGaps := Nat.choose gaps numRepublicans
  let republicansArrangements := numRepublicans!
  democratsArrangements * chooseGaps * republicansArrangements

-- Statement to prove
theorem seatingArrangementsCorrect :
  countValidArrangements = 43200 :=
by
  sorry

end seatingArrangementsCorrect_l69_69762


namespace exist_xyz_modular_l69_69882

theorem exist_xyz_modular {n a b c : ℕ} (hn : 0 < n) (ha : a ≤ 3 * n ^ 2 + 4 * n) (hb : b ≤ 3 * n ^ 2 + 4 * n) (hc : c ≤ 3 * n ^ 2 + 4 * n) :
  ∃ (x y z : ℤ), abs x ≤ 2 * n ∧ abs y ≤ 2 * n ∧ abs z ≤ 2 * n ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 :=
sorry

end exist_xyz_modular_l69_69882


namespace exists_set_satisfying_condition_l69_69673

theorem exists_set_satisfying_condition (n : ℕ) (h : n ≥ 2) : 
  ∃ S : Finset ℤ, S.card = n ∧ (∀ a b ∈ S, a ≠ b → (a - b)^2 ∣ (a * b)) := 
sorry

end exists_set_satisfying_condition_l69_69673


namespace distance_between_A_and_B_is_750_l69_69296

def original_speed := 150 -- derived from the solution

def distance (S D : ℝ) :=
  (D / S) - (D / ((5 / 4) * S)) = 1 ∧
  ((D - 150) / S) - ((5 * (D - 150)) / (6 * S)) = 2 / 3

theorem distance_between_A_and_B_is_750 :
  ∃ D : ℝ, distance original_speed D ∧ D = 750 :=
by
  sorry

end distance_between_A_and_B_is_750_l69_69296


namespace volume_of_tank_in_liters_l69_69419

-- Define the conditions given in the problem
def cube_face_area : ℝ := 100 -- area of one face of the cube in square meters

-- Define the volume of the tank to be a cube with face area given by the conditions
noncomputable def tank_volume_cubic_meters (side_length : ℝ) : ℝ :=
  side_length ^ 3

-- Define the conversion from cubic meters to liters
def cubic_meters_to_liters (volume_cubic_meters : ℝ) : ℝ :=
  volume_cubic_meters * 1000

-- Define the side length of the cube given the face area
def side_length_of_cube (face_area : ℝ) : ℝ :=
  real.sqrt face_area

-- State the problem in Lean
theorem volume_of_tank_in_liters :
  cubic_meters_to_liters (tank_volume_cubic_meters (side_length_of_cube cube_face_area)) = 1000000 := by
  sorry

end volume_of_tank_in_liters_l69_69419


namespace gcd_of_three_numbers_l69_69497

-- Definition of the numbers we are interested in
def a : ℕ := 9118
def b : ℕ := 12173
def c : ℕ := 33182

-- Statement of the problem to prove GCD
theorem gcd_of_three_numbers : Int.gcd (Int.gcd a b) c = 47 := 
sorry  -- Proof skipped

end gcd_of_three_numbers_l69_69497


namespace vector_DO_l69_69757

noncomputable def vector_e1 : Type := sorry
noncomputable def vector_e2 : Type := sorry
noncomputable def vector : Type := sorry

def AB : vector := 4 * vector_e1
def BC : vector := 6 * vector_e2
def O : vector := sorry -- Intersection of diagonals not explicitly needed but placeholder for intersection point.

-- Proof statement
theorem vector_DO (AB BC : vector) : (AB = 4 * vector_e1) → (BC = 6 * vector_e2) →
                                    (∃ DO : vector, DO = 2 * vector_e1 - 3 * vector_e2) :=
begin
  intros hAB hBC,
  existsi (2 * vector_e1 - 3 * vector_e2),
  exact sorry, -- detailed proof is skipped.
end

end vector_DO_l69_69757


namespace bacteria_initial_count_l69_69688

def quadruple_growth (initial_bacteria : ℕ) (intervals : ℕ) : ℕ :=
  initial_bacteria * 4^intervals

theorem bacteria_initial_count (final_bacteria : ℕ) (intervals : ℕ)
  (h_intervals : intervals = 8) (h_final : final_bacteria = 1048576) :
  ∃ initial_bacteria, quadruple_growth initial_bacteria intervals = final_bacteria ∧ initial_bacteria = 16 :=
begin
  use 16,
  split,
  { calc
      quadruple_growth 16 8
        = 16 * 4^8 : rfl
    ... = 16 * 65536 : by norm_num
    ... = 1048576 : by norm_num },
  { refl }
end

end bacteria_initial_count_l69_69688


namespace length_segment_PQ_half_perimeter_l69_69523

open EuclideanGeometry

variables (A B C D P Q M N : Point)
variables (h_trap : Trapezoid A B C D)
variables (h_base : Base A D)
variables (h_P : ExternalAngleBisectorIntersection A B P)
variables (h_Q : ExternalAngleBisectorIntersection C D Q)

theorem length_segment_PQ_half_perimeter :
  length (segment P Q) = (length (segment A B) + length (segment B C) + length (segment C D) + length (segment D A)) / 2 :=
sorry

end length_segment_PQ_half_perimeter_l69_69523


namespace triangles_in_pentadecagon_l69_69980

theorem triangles_in_pentadecagon :
  let n := 15
  in (Nat.choose n 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69980


namespace monotonic_quadratic_interval_l69_69328

namespace MathProof

noncomputable def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + m * x - 1

theorem monotonic_quadratic_interval (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 ∈ set.Icc (-1) 3 → x2 ∈ set.Icc (-1) 3 → x1 ≤ x2 → quadratic m x1 ≤ quadratic m x2) ∨ 
  (∀ x1 x2 : ℝ, x1 ∈ set.Icc (-1) 3 → x2 ∈ set.Icc (-1) 3 → x1 ≤ x2 → quadratic m x1 ≥ quadratic m x2) →
  m ∈ set.Iic (-6) ∨ m ∈ set.Ici 2 :=
by sorry

end MathProof

end monotonic_quadratic_interval_l69_69328


namespace find_x_given_conditions_l69_69131

theorem find_x_given_conditions (x y : ℚ) (h1 : x / y = 12 / 5) (h2 : y = 20) : x = 48 := by 
  sorry

end find_x_given_conditions_l69_69131


namespace total_cars_parked_l69_69251

theorem total_cars_parked
  (area_a : ℕ) (util_a : ℕ)
  (area_b : ℕ) (util_b : ℕ)
  (area_c : ℕ) (util_c : ℕ)
  (area_d : ℕ) (util_d : ℕ)
  (space_per_car : ℕ) 
  (ha: area_a = 400 * 500)
  (hu_a: util_a = 80)
  (hb: area_b = 600 * 700)
  (hu_b: util_b = 75)
  (hc: area_c = 500 * 800)
  (hu_c: util_c = 65)
  (hd: area_d = 300 * 900)
  (hu_d: util_d = 70)
  (h_sp: space_per_car = 10) :
  (util_a * area_a / 100 / space_per_car + 
   util_b * area_b / 100 / space_per_car + 
   util_c * area_c / 100 / space_per_car + 
   util_d * area_d / 100 / space_per_car) = 92400 :=
by sorry

end total_cars_parked_l69_69251


namespace num_triangles_pentadecagon_l69_69990

/--
  The number of triangles that can be formed using the vertices of a regular pentadecagon
  (a 15-sided polygon where no three vertices are collinear) is 455.
-/
theorem num_triangles_pentadecagon : ∀ (n : ℕ), n = 15 → ∃ (num_triangles : ℕ), num_triangles = Nat.choose n 3 ∧ num_triangles = 455 :=
by
  intros n hn
  use Nat.choose n 3
  split
  · rfl
  · sorry

end num_triangles_pentadecagon_l69_69990


namespace certain_number_divisibility_l69_69123

theorem certain_number_divisibility :
  ∃ k : ℕ, 3150 = 1050 * k :=
sorry

end certain_number_divisibility_l69_69123


namespace monic_quadratic_with_root_2_minus_3i_l69_69847

theorem monic_quadratic_with_root_2_minus_3i :
  ∃ P : ℝ[X], P.monic ∧ (P.coeff 2 = 1)
    ∧ (P.coeff 1 = -4)
    ∧ (P.coeff 0 = 13)
    ∧ eval (2 - 3 * I) P = 0 := sorry

end monic_quadratic_with_root_2_minus_3i_l69_69847


namespace nails_needed_l69_69128

-- Define the number of nails needed for each plank
def nails_per_plank : ℕ := 2

-- Define the number of planks used by John
def planks_used : ℕ := 16

-- The total number of nails needed.
theorem nails_needed : (nails_per_plank * planks_used) = 32 :=
by
  -- Our goal is to prove that nails_per_plank * planks_used = 32
  sorry

end nails_needed_l69_69128


namespace num_valid_6digit_numbers_l69_69561

/-- The number of 6-digit numbers with digits from the set {1, 2, 3, 4, 5} such that each digit appears at least twice is 1255. -/
theorem num_valid_6digit_numbers (digits : Finset ℕ) (count : ℕ) :
  digits = {1, 2, 3, 4, 5} → count = 6 → (∃ n : ℕ, 1255 = n)
:=
by
  assume digits_eq count_eq
  have h1 : ∀ n, 1255 = n → ∃ n : ℕ, 1255 = n from sorry
  exact h1 1255 rfl

end num_valid_6digit_numbers_l69_69561


namespace num_four_digit_palindromic_squares_l69_69015

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69015


namespace possible_values_first_term_l69_69056

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (Math.sqrt 3 / 4) * s^2

noncomputable def sequence_term (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then x
  else if n = 1 then equilateral_triangle_area (x / 3)
  else if n = 2 then do
    let y := equilateral_triangle_area (x / 3)
    equilateral_triangle_area (y / 3)
  else 0 -- placeholder for terms not needed in this problem

theorem possible_values_first_term (x : ℝ) (h_pos : 0 < x) (h_ap : 
  2 * (equilateral_triangle_area (x / 3)) =
  x + (equilateral_triangle_area ((equilateral_triangle_area (x / 3)) / 3))) :
  x = 12 * Math.sqrt 3 ∨ x = 6 * Math.sqrt 15 - 6 * Math.sqrt 3 :=
sorry

end possible_values_first_term_l69_69056


namespace arrangement_AB_not_adjacent_l69_69868

theorem arrangement_AB_not_adjacent (A B C D : Type) : 
  ∃! n, n = 12 ∧ 
  ∀ arrangements : List (A × B × C × D), 
  arrangements.filter (fun arr => not (adjacent arr A B)).length = n :=
begin
  sorry
end

end arrangement_AB_not_adjacent_l69_69868


namespace triangle_integer_solutions_l69_69714

theorem triangle_integer_solutions (x : ℕ) (h1 : 13 < x) (h2 : x < 43) : 
  ∃ (n : ℕ), n = 29 :=
by 
  sorry

end triangle_integer_solutions_l69_69714


namespace not_perfect_square_l69_69300

theorem not_perfect_square (n : ℕ) (hn : n > 1) : ¬ ∃ a : ℕ, a^2 = 4 * 10^n + 9 :=
begin
  sorry
end

end not_perfect_square_l69_69300


namespace fraction_evaluation_l69_69804

theorem fraction_evaluation :
  let p := 8579
  let q := 6960
  p.gcd q = 1 ∧ (32 / 30 - 30 / 32 + 32 / 29) = p / q :=
by
  sorry

end fraction_evaluation_l69_69804


namespace num_valid_sequences_l69_69226

def is_valid_sequence (seq : List Char) : Prop :=
  seq.length = 20 ∧ 
  (count_subsequences seq 'H' 'H' = 3) ∧ 
  (count_subsequences seq 'H' 'T' = 5) ∧ 
  (count_subsequences seq 'T' 'H' = 6) ∧ 
  (count_subsequences seq 'T' 'T' = 5)

def count_subsequences : (List Char) → Char → Char → Nat
| [] , _ , _    => 0
| (x::xs), c1, c2 => if x = c1 ∧ xs.head = c2
                     then 1 + count_subsequences xs.tail c1 c2
                     else count_subsequences xs c1 c2
                     
theorem num_valid_sequences : 
  ∃ seqs : List (List Char), 
  (∀ seq ∈ seqs, is_valid_sequence seq) ∧ 
  seqs.length = 283140 := 
sorry

end num_valid_sequences_l69_69226


namespace rearrange_CCAMB_at_least_one_C_before_A_l69_69997

theorem rearrange_CCAMB_at_least_one_C_before_A : 
  (∃ t : Finset (Finset (Fin 5)) (n : ℕ), 
  let total_permutations := nat.factorial 5 / nat.factorial 2,
  let invalid_permutations := (nat.choose 5 3) * nat.factorial 2,
  n = total_permutations - invalid_permutations,
  t.card = n,
  n = 40) :=
by
    sorry

end rearrange_CCAMB_at_least_one_C_before_A_l69_69997


namespace triangles_in_pentadecagon_l69_69967

def regular_pentadecagon := {vertices : Finset Point | vertices.card = 15 ∧ 
  ∀ a b c ∈ vertices, ¬Collinear a b c}

theorem triangles_in_pentadecagon (P : regular_pentadecagon) : 
  (P.vertices.card.choose 3) = 455 :=
by 
  sorry


end triangles_in_pentadecagon_l69_69967


namespace convert_to_base13_l69_69093

theorem convert_to_base13 : ∀ n : ℕ, n = 136 → (n = 10 * 13 + 6) → "A6" = "A6" :=
by
  intros n hn hcalc
  have h0 : 136 = 10 * 13 + 6, by rw hcalc
  exact eq.refl "A6"

end convert_to_base13_l69_69093


namespace sports_lottery_systematic_sampling_l69_69613

-- Definition of the sports lottery condition
def is_first_prize_ticket (n : ℕ) : Prop := n % 1000 = 345

-- Statement of the proof problem
theorem sports_lottery_systematic_sampling :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → is_first_prize_ticket n) →
  ∃ interval, (∀ segment_start : ℕ,  segment_start < 1000 → is_first_prize_ticket (segment_start + interval * 999))
  := by sorry

end sports_lottery_systematic_sampling_l69_69613


namespace min_circles_triangle_min_circles_convex_polygon_l69_69382

-- Define the problem conditions
def diameter_1_circle (r : ℝ) := r = 1

-- Define the proof problem for part (a) - covering a triangle
theorem min_circles_triangle : ∀ T : Triangle, is_triangle T → minimum_circles_cover T (diameter_1_circle 2) := 
sorry

-- Define the proof problem for part (b) - covering a convex polygon with diameter 1
theorem min_circles_convex_polygon : ∀ P : ConvexPolygon, convex_polygon_diameter P = 1 → minimum_circles_cover P (diameter_1_circle 3) := 
sorry

end min_circles_triangle_min_circles_convex_polygon_l69_69382


namespace carol_savings_l69_69807

theorem carol_savings (S : ℝ) (h1 : ∀ t : ℝ, t = S - (2/3) * S) (h2 : S + (S - (2/3) * S) = 1/4) : S = 3/16 :=
by {
  sorry
}

end carol_savings_l69_69807


namespace max_not_divisible_by_3_l69_69789

theorem max_not_divisible_by_3 (a b c d e f : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) (h7 : 3 ∣ (a * b * c * d * e * f)) : 
  ∃ x y z u v, ((x = a ∧ y = b ∧ z = c ∧ u = d ∧ v = e) ∨ (x = a ∧ y = b ∧ z = c ∧ u = d ∧ v = f) ∨ (x = a ∧ y = b ∧ z = c ∧ u = e ∧ v = f) ∨ (x = a ∧ y = b ∧ z = d ∧ u = e ∧ v = f) ∨ (x = a ∧ y = c ∧ z = d ∧ u = e ∧ v = f) ∨ (x = b ∧ y = c ∧ z = d ∧ u = e ∧ v = f)) ∧ (¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z) ∧ ¬ (3 ∣ u) ∧ ¬ (3 ∣ v)) :=
sorry

end max_not_divisible_by_3_l69_69789


namespace number_of_solutions_l69_69121

theorem number_of_solutions (f g : ℝ → ℝ) (h : ∀ x, f x ≠ g x ∧ (∀ x ∈ Icc (-50 : ℝ) 50, -1 ≤ g x ∧ g x ≤ 1)) :
  (∃ s : finset ℝ, (∀ x ∈ s, f x = g x ∧ x ∈ Icc (-50 : ℝ) 50) ∧ finset.card s = 31) :=
sorry

end number_of_solutions_l69_69121


namespace find_subtracted_value_l69_69773

theorem find_subtracted_value (n x : ℕ) (h1 : n = 120) (h2 : n / 6 - x = 5) : x = 15 := by
  sorry

end find_subtracted_value_l69_69773


namespace triangles_in_pentadecagon_l69_69975

theorem triangles_in_pentadecagon :
  let n := 15
  in (Nat.choose n 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69975


namespace investment_difference_rounded_l69_69103

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

noncomputable def diana_amount : ℝ :=
  compound_interest 100000 (0.05) 3

noncomputable def emily_amount : ℝ :=
  compound_interest 100000 (0.05 / 12) 36

noncomputable def investment_difference : ℝ :=
  emily_amount - diana_amount

theorem investment_difference_rounded :
  Real.floor (investment_difference + 0.5) = 405 := by
    sorry

end investment_difference_rounded_l69_69103


namespace amanda_kept_candies_l69_69433

noncomputable def amanda_candy_bars (initial_candies: ℕ) (first_give: ℕ) (bought_candies: ℕ) (multiplier: ℕ): ℕ :=
  let remaining_after_first = initial_candies - first_give
  let second_give = first_give * multiplier
  let remaining_after_second = bought_candies - second_give
  remaining_after_first + remaining_after_second

theorem amanda_kept_candies (initial_candies: ℕ) (first_give: ℕ) (bought_candies: ℕ) (multiplier: ℕ) :
  initial_candies = 7 →
  first_give = 3 →
  bought_candies = 30 →
  multiplier = 4 →
  amanda_candy_bars initial_candies first_give bought_candies multiplier = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calculate_amanda_kept_candies
  sorry

end amanda_kept_candies_l69_69433


namespace largest_prime_17p_625_l69_69356

theorem largest_prime_17p_625 (p : ℕ) (h_prime : Nat.Prime p) (h_sqrt : ∃ q, 17 * p + 625 = q^2) : p = 67 :=
by
  sorry

end largest_prime_17p_625_l69_69356


namespace series_convergence_l69_69753

theorem series_convergence (a : ℕ → ℝ) (h : Summable (λ n, a n + 2 * a (n + 1))) : Summable a :=
sorry

end series_convergence_l69_69753


namespace systematic_sampling_third_group_number_l69_69392

theorem systematic_sampling_third_group_number :
  ∀ (total_members groups sample_number group_5_number group_gap : ℕ),
  total_members = 200 →
  groups = 40 →
  sample_number = total_members / groups →
  group_5_number = 22 →
  group_gap = 5 →
  (group_this_number : ℕ) = group_5_number - (5 - 3) * group_gap →
  group_this_number = 12 :=
by
  intros total_members groups sample_number group_5_number group_gap Htotal Hgroups Hsample Hgroup5 Hgap Hthis_group
  sorry

end systematic_sampling_third_group_number_l69_69392


namespace quadratic_has_real_root_l69_69672

theorem quadratic_has_real_root (a b c : ℝ) : ∃ x : ℝ, (x - a) * (x - b) - c^2 = 0 :=
by
  let D := (a - b)^2 + 4 * c^2
  have hD : D ≥ 0 := by sorry
  have h : ∀ A B C : ℝ, B^2 - 4 * A * C ≥ 0 → ∃ x : ℝ, A * x^2 + B * x + C = 0 := by sorry
  exact h 1 (-(a + b)) (ab - c^2) hD

end quadratic_has_real_root_l69_69672


namespace quadruples_characterization_l69_69114

/-- Proving the characterization of quadruples (a, b, c, d) of non-negative integers 
such that ab = 2(1 + cd) and there exists a non-degenerate triangle with sides (a - c), 
(b - d), and (c + d). -/
theorem quadruples_characterization :
  ∀ (a b c d : ℕ), 
    ab = 2 * (1 + cd) ∧ 
    (a - c) + (b - d) > c + d ∧ 
    (a - c) + (c + d) > b - d ∧ 
    (b - d) + (c + d) > a - c ∧
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    (a = 1 ∧ b = 2 ∧ c = 0 ∧ d = 1) ∨ 
    (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 0) :=
by sorry

end quadruples_characterization_l69_69114


namespace four_digit_palindromic_perfect_square_count_l69_69046

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69046


namespace four_digit_palindromic_perfect_square_count_l69_69038

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69038


namespace tank_volume_in_liters_l69_69420

theorem tank_volume_in_liters :
  (let side_length := 10 in
   let volume_in_cubic_meters := side_length ^ 3 in
   let volume_in_liters := volume_in_cubic_meters * 1000 in
   volume_in_liters = 1000000) :=
by
  let side_length := 10
  let volume_in_cubic_meters := side_length ^ 3
  let volume_in_liters := volume_in_cubic_meters * 1000
  have equation : volume_in_liters = 1000000 := by
    calc 
      volume_in_cubic_meters * 1000 = (10 ^ 3) * 1000 : rfl
      ... = 1000 * 1000 : by norm_num
      ... = 1000000 : by norm_num
  exact equation

end tank_volume_in_liters_l69_69420


namespace solution_set_of_inequality_l69_69715

open Set Real

theorem solution_set_of_inequality :
  {x : ℝ | sqrt (x + 3) > 3 - x} = {x : ℝ | 1 < x} ∪ {x : ℝ | x ≥ 3} := by
  sorry

end solution_set_of_inequality_l69_69715


namespace find_angles_l69_69240

namespace TriangleAngles

def angle_measures (A B C : ℝ) : Prop :=
  ∠ B - ∠ A = 5 ∧ ∠ C - ∠ B = 20 ∧ ∠ A + ∠ B + ∠ C = 180

theorem find_angles (A B C : ℝ) :
  angle_measures A B C → A = 50 ∧ B = 55 ∧ C = 75 :=
  by
  sorry

end TriangleAngles

end find_angles_l69_69240


namespace integer_polynomial_roots_l69_69821

theorem integer_polynomial_roots (n : ℕ) (h : n ≥ 2)
  (H : ∀ (k : ℕ) (a : fin k → ℕ), (∀ i j, i ≠ j → a i % n ≠ a j % n) → 
    ∃ f : polynomial ℤ, ∀ i, (f.eval (a i)) % n = 0 ∧
    ∀ x, (f.eval x) % n = 0 → ∃ j, x ≡ a j [MOD n]) :
  n = 2 ∨ n = 4 ∨ ∃ p : ℕ, nat.prime p ∧ n = p := 
sorry

end integer_polynomial_roots_l69_69821


namespace xyz_inequality_l69_69276

theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end xyz_inequality_l69_69276


namespace triangles_in_pentadecagon_l69_69951

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69951


namespace proof_either_p_or_q_true_l69_69521

variable (a : ℝ) (p q : Prop)

def proposition_p (a : ℝ) : Prop :=
  ∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), 2 - a * x > 0

def proposition_q (a : ℝ) : Prop :=
  ∀ x, (x^2 < 1 → x < a) ∧ ¬(x < a → x^2 < 1)

theorem proof_either_p_or_q_true (h_cond : 1 < a ∧ a < 2) :
  (proposition_p a) ∨ (proposition_q a) :=
sorry

end proof_either_p_or_q_true_l69_69521


namespace problem1_l69_69143

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + 1

theorem problem1 :
  (f 0 = 1) ∧
  (deriv f 1 = 1) ∧
  (f 1 = 1) ∧
  (f (0 : ℝ) = 1) ∧
  (f (2 / 3 : ℝ) = 23 / 27)
  := by
  unfold f
  norm_num
  rw [derivative]
  norm_num
  sorry

end problem1_l69_69143


namespace monic_quadratic_real_root_l69_69831

theorem monic_quadratic_real_root (a b : ℂ) (h : b = 2 - 3 * complex.I) :
  ∃ P : polynomial ℂ, P.monic ∧ P.coeff 2 = 1 ∧ P.coeff 1 = -4 ∧ P.coeff 0 = 13 ∧ P.is_root (2 - 3 * complex.I) :=
by
  sorry

end monic_quadratic_real_root_l69_69831


namespace maximize_binomial_prob_l69_69922

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  nat.choose n k * p^k * (1 - p)^(n - k)

theorem maximize_binomial_prob :
  ∃ k : ℕ, (k = 6 ∨ k = 7) ∧
  (∀ j : ℕ, binomial_prob 20 k (1/3) ≥ binomial_prob 20 j (1/3)) :=
by 
  sorry

end maximize_binomial_prob_l69_69922


namespace division_approx_l69_69082

-- Define the expression inside the parentheses
def inner_expr : ℝ := 12 + 13 * 2

-- Define the entire division expression
def expr : ℝ := 180 / inner_expr

-- State the proposition that the expression is approximately 4.74
theorem division_approx : expr ≈ 4.74 := by
  sorry

end division_approx_l69_69082


namespace collinear_points_x_value_l69_69582

theorem collinear_points_x_value
  (x : ℝ)
  (h : ∃ m : ℝ, m = (1 - (-4)) / (-1 - 2) ∧ m = (-9 - (-4)) / (x - 2)) :
  x = 5 :=
by
  sorry

end collinear_points_x_value_l69_69582


namespace percentage_of_seats_filled_l69_69224

-- Define the conditions
def total_seats : ℕ := 600
def vacant_seats : ℕ := 330

-- Define the number of occupied seats
def occupied_seats : ℕ := total_seats - vacant_seats

-- Define the percentage calculation function
def percentage_filled (total : ℕ) (occupied : ℕ) : ℝ := 
  (occupied.toReal / total.toReal) * 100.0

-- State the main theorem
theorem percentage_of_seats_filled : 
  percentage_filled total_seats occupied_seats = 45 := 
by
  sorry

end percentage_of_seats_filled_l69_69224


namespace company_fund_initial_amount_l69_69702

theorem company_fund_initial_amount (n : ℕ) 
  (h : 45 * n + 95 = 50 * n - 5) : 50 * n - 5 = 995 := by
  sorry

end company_fund_initial_amount_l69_69702


namespace infinite_pairs_natural_numbers_l69_69675

theorem infinite_pairs_natural_numbers :
  ∃ (infinite_pairs : ℕ × ℕ → Prop), (∀ a b : ℕ, infinite_pairs (a, b) ↔ (b ∣ (a^2 + 1) ∧ a ∣ (b^2 + 1))) ∧
    ∀ n : ℕ, ∃ (a b : ℕ), infinite_pairs (a, b) :=
sorry

end infinite_pairs_natural_numbers_l69_69675


namespace length_of_CE_is_10_l69_69607

noncomputable def length_of_CE (AE : ℝ) (BE : ℝ) (CE : ℝ) : Prop :=
  ∀ (AE BE CE : ℝ), AE = 20 →
  BE = AE / Real.sqrt 2 →
  CE = BE / Real.sqrt 2 →
  CE = 10

theorem length_of_CE_is_10 : length_of_CE 20 (20 / Real.sqrt 2) ((20 / Real.sqrt 2) / Real.sqrt 2) :=
by
  intros AE BE CE hAE hBE hCE
  rw [hAE, hBE, hCE]
  sorry

end length_of_CE_is_10_l69_69607


namespace cost_of_baseball_cards_l69_69869

variables (cost_football cost_pokemon total_spent cost_baseball : ℝ)
variable (h1 : cost_football = 2 * 2.73)
variable (h2 : cost_pokemon = 4.01)
variable (h3 : total_spent = 18.42)
variable (total_cost_football_pokemon : ℝ)
variable (h4 : total_cost_football_pokemon = cost_football + cost_pokemon)

theorem cost_of_baseball_cards
  (h : cost_baseball = total_spent - total_cost_football_pokemon) : 
  cost_baseball = 8.95 :=
by
  sorry

end cost_of_baseball_cards_l69_69869


namespace g_diff_l69_69570

def g (n : ℤ) : ℤ := (1 / 4 : ℤ) * n * (n + 1) * (n + 2) * (n + 3)

theorem g_diff (r : ℤ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
  sorry

end g_diff_l69_69570


namespace vertex_of_parabola_l69_69342

theorem vertex_of_parabola (a b : ℝ) (roots_condition : ∀ x, -x^2 + a * x + b ≤ 0 ↔ (x ≤ -3 ∨ x ≥ 5)) :
  ∃ v : ℝ × ℝ, v = (1, 16) :=
by
  sorry

end vertex_of_parabola_l69_69342


namespace smallest_n_divisibility_l69_69358

theorem smallest_n_divisibility (n : ℕ) (h : n = 5) : 
  ∃ k, (1 ≤ k ∧ k ≤ n + 1) ∧ (n^2 - n) % k = 0 ∧ ∃ i, (1 ≤ i ∧ i ≤ n + 1) ∧ (n^2 - n) % i ≠ 0 :=
by
  sorry

end smallest_n_divisibility_l69_69358


namespace count_four_digit_palindrome_squares_l69_69035

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69035


namespace tank_fill_time_l69_69620

-- Define the conditions
def start_time : ℕ := 1 -- 1 pm
def first_hour_rainfall : ℕ := 2 -- 2 inches rainfall in the first hour from 1 pm to 2 pm
def next_four_hours_rate : ℕ := 1 -- 1 inch/hour rainfall rate from 2 pm to 6 pm
def following_rate : ℕ := 3 -- 3 inches/hour rainfall rate from 6 pm onwards
def tank_height : ℕ := 18 -- 18 inches tall fish tank

-- Define what needs to be proved
theorem tank_fill_time : 
  ∃ t : ℕ, t = 22 ∧ (tank_height ≤ (first_hour_rainfall + 4 * next_four_hours_rate + (t - 6)) + (t - 6 - 4) * following_rate) := 
by 
  sorry

end tank_fill_time_l69_69620


namespace amanda_kept_candies_l69_69434

noncomputable def amanda_candy_bars (initial_candies: ℕ) (first_give: ℕ) (bought_candies: ℕ) (multiplier: ℕ): ℕ :=
  let remaining_after_first = initial_candies - first_give
  let second_give = first_give * multiplier
  let remaining_after_second = bought_candies - second_give
  remaining_after_first + remaining_after_second

theorem amanda_kept_candies (initial_candies: ℕ) (first_give: ℕ) (bought_candies: ℕ) (multiplier: ℕ) :
  initial_candies = 7 →
  first_give = 3 →
  bought_candies = 30 →
  multiplier = 4 →
  amanda_candy_bars initial_candies first_give bought_candies multiplier = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calculate_amanda_kept_candies
  sorry

end amanda_kept_candies_l69_69434


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l69_69643

variable (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_52 : (Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52) :=
by
  sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l69_69643


namespace birds_left_after_week_l69_69413

def chickens_initial := 300
def turkeys_initial := 200
def guinea_fowls_initial := 80

def daily_loss_chickens := 20
def daily_loss_turkeys := 8
def daily_loss_guinea_fowls := 5
def days_in_week := 7

theorem birds_left_after_week : 
  let chickens_left := chickens_initial - daily_loss_chickens * days_in_week in
  let turkeys_left := turkeys_initial - daily_loss_turkeys * days_in_week in
  let guinea_fowls_left := guinea_fowls_initial - daily_loss_guinea_fowls * days_in_week in
  chickens_left + turkeys_left + guinea_fowls_left = 349 :=
by
  sorry

end birds_left_after_week_l69_69413


namespace range_of_a_l69_69178

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 < a ∧ a ≤ 3 := by
  sorry

end range_of_a_l69_69178


namespace find_y_value_l69_69788

noncomputable def AliceRotation (P Q R : Type) : Prop :=
  ∃ θ₁, θ₁ = 480 ∧ rotate_clockwise θ₁ P Q = R

noncomputable def BobRotation (P Q R : Type) (y : ℝ) : Prop :=
  ∃ θ₂, θ₂ = y ∧ rotate_counterclockwise θ₂ P Q = R

theorem find_y_value (P Q R : Type) (y : ℝ) (h1 : AliceRotation P Q R) (h2 : BobRotation P Q R y) (h3 : y < 360) : y = 240 :=
begin
  sorry
end

end find_y_value_l69_69788


namespace solution_set_xf_gt_zero_l69_69575

def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) := ∀ x y ∈ s, x < y → f x < f y

theorem solution_set_xf_gt_zero (f : ℝ → ℝ) 
  (odd_f : is_odd_function f) 
  (increasing_on_pos : is_increasing_on f (Set.Ioi 0)) 
  (f_3_eq_zero : f 3 = 0) :
  {x : ℝ | x * f x > 0} = Set.Iio (-3) ∪ Set.Ioi 3 := 
sorry

end solution_set_xf_gt_zero_l69_69575


namespace find_m_for_equal_roots_l69_69236

theorem find_m_for_equal_roots :
  ∃ (m : ℝ), 
  (∀ (x : ℝ),
  x * (x - 1) - (m^2 + 2 * m + 1)) / ((x - 1) * (m^2 - 1) + 1) = x / m ∧
  ((m^2 - m) * x^2 + (m - m^2) * x + (m^3 + 2 * m^2 + m) = 0) ∧
  discrim_cond (m^2, m^2, m^3 + 2 * m^2 + m) = 0 := m = -1 :=
sorry

noncomputable def discrim_cond (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

end find_m_for_equal_roots_l69_69236


namespace exponential_expression_evaluation_l69_69805

theorem exponential_expression_evaluation :
  (3 ^ 1010 + 7 ^ 1011) ^ 2 - (3 ^ 1010 - 7 ^ 1011) ^ 2 = 59 * 10 ^ 1010 := 
by
sorrry

end exponential_expression_evaluation_l69_69805


namespace parabola_angle_statements_l69_69901

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def directrix_point (t : ℝ) : Prop := t ∈ {-1}

def line_intersects_parabola (m k x1 y1 x2 y2 : ℝ) : Prop :=
  parabola x1 y1 ∧ x1 = k * y1 + m ∧
  parabola x2 y2 ∧ x2 = k * y2 + m ∧
  x1 ≠ x2

def angle_ADB_LT_90 (m k t x1 y1 x2 y2 : ℝ) : Prop :=
  m = 3 → line_intersects_parabola m k x1 y1 x2 y2 →
  let dot_product := (x1 + 1) * (x2 + 1) + (y1 - t) * (y2 - t)
  in dot_product > 0

def angle_AFB_GT_90 (m k f_x f_y x1 y1 x2 y2 : ℝ) : Prop :=
  m = 3 → line_intersects_parabola m k x1 y1 x2 y2 →
  let dot_product := (x1 - f_x) * (x2 - f_x) + (y1 - f_y) * (y2 - f_y)
  in dot_product < 0

def angle_AOB_EQ_90 (m x1 y1 x2 y2 : ℝ) : Prop :=
  m = 4 → line_intersects_parabola m 0 x1 y1 x2 y2 →
  let dot_product := x1 * x2 + y1 * y2
  in dot_product = 0

theorem parabola_angle_statements :
  ∀ (m k t f_x f_y x1 y1 x2 y2 : ℝ),
    parabola (f_x) (f_y) →
    directrix_point t →
    line_intersects_parabola m k x1 y1 x2 y2 →
    angle_ADB_LT_90 m k t x1 y1 x2 y2 ∧
    angle_AFB_GT_90 m k f_x f_y x1 y1 x2 y2 ∧
    angle_AOB_EQ_90 m x1 y1 x2 y2 :=
by
  intros
  sorry

end parabola_angle_statements_l69_69901


namespace expr_positive_for_all_x_sum_of_squares_of_roots_eq_24_l69_69904

def quadratic_expr (a x : ℝ) : ℝ := x^2 - (8 * a - 2) * x + 15 * a^2 - 2 * a - 7

theorem expr_positive_for_all_x (a : ℝ) : 2 ≤ a ∧ a ≤ 4 →
  ∀ x : ℝ, quadratic_expr a x > 0 :=
sorry

theorem sum_of_squares_of_roots_eq_24 (a : ℝ) : (a = 1 ∨ a = -3 / 17) →
  let x1 := 4 * a - 1, x2 := -4 * a + 1 in 
  x1^2 + x2^2 = 24 :=
sorry

end expr_positive_for_all_x_sum_of_squares_of_roots_eq_24_l69_69904


namespace num_functions_is_correct_l69_69555

def A := {i | 1 ≤ i ∧ i ≤ 2012}
def B := {i | 1 ≤ i ∧ i ≤ 19}
def S := set (set A)

noncomputable def valid_function (f : S → B) : Prop :=
∀ A1 A2 : S, f (A1 ∩ A2) = min (f A1) (f A2)

noncomputable def num_valid_functions := 1^2012 + 2^2012 + 3^2012 + 4^2012 +
                                          5^2012 + 6^2012 + 7^2012 + 8^2012 +
                                          9^2012 + 10^2012 + 11^2012 + 12^2012 +
                                          13^2012 + 14^2012 + 15^2012 + 16^2012 +
                                          17^2012 + 18^2012 + 19^2012

theorem num_functions_is_correct :
  ∃ f : (S → B), valid_function f → num_valid_functions = ∑ n in B, n^2012 :=
sorry

end num_functions_is_correct_l69_69555


namespace shape_described_by_theta_eq_c_is_plane_l69_69503

-- Definitions based on conditions in the problem
def spherical_coordinates (ρ θ φ : ℝ) := true

def is_plane_condition (θ c : ℝ) := θ = c

-- Statement to prove
theorem shape_described_by_theta_eq_c_is_plane (c : ℝ) :
  ∀ ρ θ φ : ℝ, spherical_coordinates ρ θ φ → is_plane_condition θ c → "Plane" = "Plane" :=
by sorry

end shape_described_by_theta_eq_c_is_plane_l69_69503


namespace expression_not_defined_l69_69865

theorem expression_not_defined (x : ℝ) : 
  (x^2 - 21 * x + 110 = 0) ↔ (x = 10 ∨ x = 11) := by
sorry

end expression_not_defined_l69_69865


namespace find_x_plus_y_l69_69893

theorem find_x_plus_y (x y : Real) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y = 2009) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 2009 + Real.pi / 2 :=
by
  sorry

end find_x_plus_y_l69_69893


namespace inequality_solution_l69_69682

def trig_ineq_sol_set (x y : Real) : Prop :=
  ∃ (n k : ℤ), x = ((-1:ℤ) ^ n * π / 6) + 2 * n * π ∧ y = π / 2 + k * π

def inequality_hold (x y : ℝ) : Prop :=
  4 * sin x - sqrt (cos y) - sqrt (cos y - 16 * cos x ^ 2 + 12) ≥ 2

theorem inequality_solution (x y : ℝ) :
  inequality_hold x y →
  trig_ineq_sol_set x y :=
sorry

end inequality_solution_l69_69682


namespace cleaning_time_is_correct_l69_69076

-- Define the given conditions
def vacuuming_minutes_per_day : ℕ := 30
def vacuuming_days_per_week : ℕ := 3
def dusting_minutes_per_day : ℕ := 20
def dusting_days_per_week : ℕ := 2

-- Define the total cleaning time per week
def total_cleaning_time_per_week : ℕ :=
  (vacuuming_minutes_per_day * vacuuming_days_per_week) + (dusting_minutes_per_day * dusting_days_per_week)

-- State the theorem we want to prove
theorem cleaning_time_is_correct : total_cleaning_time_per_week = 130 := by
  sorry

end cleaning_time_is_correct_l69_69076


namespace bob_remaining_ears_of_corn_l69_69081

noncomputable def bob_initial_bushels := 225
noncomputable def terry_kg := 45
noncomputable def jerry_lb := 18
noncomputable def linda_bushels := 60
noncomputable def stacy_ears := 100
noncomputable def susan_bushels := 16
noncomputable def tim_kg := 36
noncomputable def tim_ears := 50
noncomputable def ears_per_bushel := 75
noncomputable def pounds_per_bushel := 40
noncomputable def kg_per_pound := 0.453592

theorem bob_remaining_ears_of_corn :
  let terry_lb := terry_kg / kg_per_pound in
  let terry_bushels := terry_lb / pounds_per_bushel in
  let jerry_bushels := jerry_lb / pounds_per_bushel in
  let tim_lb := tim_kg / kg_per_pound in
  let tim_bushels := tim_lb / pounds_per_bushel in
  let total_bushels_given := terry_bushels + jerry_bushels + linda_bushels + susan_bushels + tim_bushels in
  let total_ears_in_bushels_given := total_bushels_given * ears_per_bushel in
  let total_ears_given := total_ears_in_bushels_given + stacy_ears + tim_ears in
  let bob_initial_ears := bob_initial_bushels * ears_per_bushel in
  bob_initial_ears - total_ears_given = 10654 :=
by
  sorry

end bob_remaining_ears_of_corn_l69_69081


namespace factorize_expression_l69_69487

theorem factorize_expression (a x y : ℝ) :
  a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 :=
by
  sorry

end factorize_expression_l69_69487


namespace rows_seating_exactly_10_people_exists_l69_69590

theorem rows_seating_exactly_10_people_exists :
  ∃ y x : ℕ, 73 = 10 * y + 9 * x ∧ (73 - 10 * y) % 9 = 0 := 
sorry

end rows_seating_exactly_10_people_exists_l69_69590


namespace value_of_a_l69_69636

theorem value_of_a (a : ℤ) (h0 : 0 ≤ a) (h1 : a < 13) (h2 : 13 ∣ 12^20 + a) : a = 12 :=
by sorry

end value_of_a_l69_69636


namespace P_surjective_l69_69628

noncomputable def P : ℤ → ℤ := sorry -- Polynomial of degree at most 10 with integer coefficients

axiom P_degree_le_10 : ∃ n, n ≤ 10 ∧ ∀ x, P x = polynomial.eval (polynomial.coeff n) x

axiom P_takes_values_1_to_10 : ∀ k : ℕ, (1 ≤ k ∧ k ≤ 10) → ∃ m : ℤ, P m = k

axiom P_bound : |P 10 - P 0| < 1000

theorem P_surjective : ∀ k : ℤ, ∃ m : ℤ, P m = k := 
  sorry

end P_surjective_l69_69628


namespace quadratic_inequality_l69_69826

theorem quadratic_inequality (c : ℝ) (h₁ : 0 < c) (h₂ : c < 16): ∃ x : ℝ, x^2 - 8 * x + c < 0 :=
sorry

end quadratic_inequality_l69_69826


namespace length_of_faster_train_l69_69752

/-- 
Let the faster train have a speed of 144 km per hour, the slower train a speed of 
72 km per hour, and the time taken for the faster train to cross a man in the 
slower train be 19 seconds. Then the length of the faster train is 380 meters.
-/
theorem length_of_faster_train 
  (speed_faster_train : ℝ) (speed_slower_train : ℝ) (time_to_cross : ℝ)
  (h_speed_faster_train : speed_faster_train = 144) 
  (h_speed_slower_train : speed_slower_train = 72) 
  (h_time_to_cross : time_to_cross = 19) :
  (speed_faster_train - speed_slower_train) * (5 / 18) * time_to_cross = 380 :=
by
  sorry

end length_of_faster_train_l69_69752


namespace find_interest_rate_of_first_investment_l69_69284

noncomputable def total_interest : ℚ := 73
noncomputable def interest_rate_7_percent : ℚ := 0.07
noncomputable def invested_400 : ℚ := 400
noncomputable def interest_7_percent := invested_400 * interest_rate_7_percent
noncomputable def interest_first_investment := total_interest - interest_7_percent
noncomputable def invested_first : ℚ := invested_400 - 100
noncomputable def interest_first : ℚ := 45  -- calculated as total_interest - interest_7_percent

theorem find_interest_rate_of_first_investment (r : ℚ) :
  interest_first = invested_first * r * 1 → 
  r = 0.15 :=
by
  sorry

end find_interest_rate_of_first_investment_l69_69284


namespace triangle_AGF_area_is_one_third_of_triangle_ABC_area_l69_69615

noncomputable def area (ABC AGF : ℝ) : ℝ := ABC * (1 / 3)

theorem triangle_AGF_area_is_one_third_of_triangle_ABC_area
    {A B C D E G F : Type}
    (medians_intersect_at_G : (median AD G) ∧ (median CE G))
    (F_is_midpoint_of_BC : is_midpoint F B C)
    (area_ABC : ℝ) :
    area (ABC) (AGF) = ⅓ * area_ABC := 
begin
    sorry
end

end triangle_AGF_area_is_one_third_of_triangle_ABC_area_l69_69615


namespace find_special_numbers_l69_69820

def is_digit_sum_equal (n m : Nat) : Prop := 
  (n.digits 10).sum = (m.digits 10).sum

def is_valid_number (n : Nat) : Prop := 
  100 ≤ n ∧ n ≤ 999 ∧ is_digit_sum_equal n (6 * n)

theorem find_special_numbers :
  {n : Nat | is_valid_number n} = {117, 135} :=
sorry

end find_special_numbers_l69_69820


namespace triangles_in_pentadecagon_l69_69974

theorem triangles_in_pentadecagon :
  let n := 15
  in (Nat.choose n 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69974


namespace amanda_candy_bars_kept_l69_69428

noncomputable def amanda_initial_candy_bars : ℕ := 7
noncomputable def candy_bars_given_first_time : ℕ := 3
noncomputable def additional_candy_bars : ℕ := 30
noncomputable def multiplier : ℕ := 4

theorem amanda_candy_bars_kept :
  let remaining_after_first_give := amanda_initial_candy_bars - candy_bars_given_first_time in
  let total_after_buying_more := remaining_after_first_give + additional_candy_bars in
  let candy_bars_given_second_time := multiplier * candy_bars_given_first_time in
  total_after_buying_more - candy_bars_given_second_time = 22 :=
by
  sorry

end amanda_candy_bars_kept_l69_69428


namespace amanda_kept_candies_l69_69435

noncomputable def amanda_candy_bars (initial_candies: ℕ) (first_give: ℕ) (bought_candies: ℕ) (multiplier: ℕ): ℕ :=
  let remaining_after_first = initial_candies - first_give
  let second_give = first_give * multiplier
  let remaining_after_second = bought_candies - second_give
  remaining_after_first + remaining_after_second

theorem amanda_kept_candies (initial_candies: ℕ) (first_give: ℕ) (bought_candies: ℕ) (multiplier: ℕ) :
  initial_candies = 7 →
  first_give = 3 →
  bought_candies = 30 →
  multiplier = 4 →
  amanda_candy_bars initial_candies first_give bought_candies multiplier = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calculate_amanda_kept_candies
  sorry

end amanda_kept_candies_l69_69435


namespace divisibility_of_product_l69_69743

def three_consecutive_integers (a1 a2 a3 : ℤ) : Prop :=
  a1 = a2 - 1 ∧ a3 = a2 + 1

theorem divisibility_of_product (a1 a2 a3 : ℤ) (h : three_consecutive_integers a1 a2 a3) : 
  a2^3 ∣ (a1 * a2 * a3 + a2) :=
by
  cases h with
  | intro ha1 ha3 =>
    sorry

end divisibility_of_product_l69_69743


namespace discount_percentage_during_sale_l69_69767

variable (d : ℝ) (x : ℝ)
variable (h1 : d > 0)
variable (h2 : (d * (1 - x / 100) * 0.9 = 0.765 * d))

theorem discount_percentage_during_sale : x = 15 :=
by
  -- Given conditions and the goal
  have h3 : d ≠ 0 := (ne_of_gt h1)
  rw [mul_assoc] at h2
  have eq1 : (1 - x / 100) * 0.9 = 0.765 :=
    (mul_right_cancel₀ h3 h2)
  sorry

end discount_percentage_during_sale_l69_69767


namespace determinant_of_given_matrix_l69_69087

noncomputable def given_matrix : Matrix (Fin 4) (Fin 4) ℤ :=
![![1, -3, 3, 2], ![0, 5, -1, 0], ![4, -2, 1, 0], ![0, 0, 0, 6]]

theorem determinant_of_given_matrix :
  Matrix.det given_matrix = -270 := by
  sorry

end determinant_of_given_matrix_l69_69087


namespace greatest_integer_neg_2_7_l69_69083

def greatest_integer (x : ℝ) : ℤ :=
  ⌊x⌋

theorem greatest_integer_neg_2_7 : greatest_integer (-2.7) = -3 :=
by
  sorry

end greatest_integer_neg_2_7_l69_69083


namespace gpa_at_least_4_is_correct_l69_69446

def Literature_grades := {A : ℝ, B : ℝ, C : ℝ} -- Probability of grades in Literature
def Sociology_grades := {A : ℝ, B : ℝ, C : ℝ} -- Probability of grades in Sociology

-- Define the probability values for the grades
def literature_prob : Literature_grades := {A := 1/5, B := 2/5, C := 2/5}
def sociology_prob : Sociology_grades := {A := 1/3, B := 1/2, C := 1/6}

-- Define the points for each grade
def points (grade : String) : ℝ :=
  if grade = "A" then 5 else if grade = "B" then 4 else if grade = "C" then 2 else if grade = "D" then 1 else 0

-- Calculus and Physics are A's
def calc_and_physics_points : ℝ := 5 + 5

-- Remaining points needed for GPA of at least 4
def remaining_points_needed (total_points : ℝ) : ℝ := 20 - total_points

-- Calculate the probability of Jackson achieving a GPA of at least 4
noncomputable def gpa_at_least_4 : ℝ := 
  let probability_two_As := (literature_prob.A * sociology_prob.A)
  let probability_one_A_lit_B_soc := (literature_prob.A * sociology_prob.B)
  let probability_one_A_soc_B_lit := (sociology_prob.A * literature_prob.B)
  probability_two_As + probability_one_A_lit_B_soc + probability_one_A_soc_B_lit

theorem gpa_at_least_4_is_correct : gpa_at_least_4 = 2/5 := 
  sorry

end gpa_at_least_4_is_correct_l69_69446


namespace find_a_l69_69552

-- Define f and g as given in the conditions
def f (x : ℝ) : ℝ := 5^(abs x)
def g (a x : ℝ) : ℝ := a * x^2 - x

-- The main proof statement
theorem find_a (a : ℝ) (h : f (g a 1) = 1) : a = 1 :=
by
  -- Placeholder for proof
  sorry

end find_a_l69_69552


namespace annual_parking_savings_l69_69775

theorem annual_parking_savings :
  let weekly_rate := 10
  let monthly_rate := 40
  let weeks_in_year := 52
  let months_in_year := 12
  let annual_weekly_cost := weekly_rate * weeks_in_year
  let annual_monthly_cost := monthly_rate * months_in_year
  let savings := annual_weekly_cost - annual_monthly_cost
  savings = 40 := by
{
  sorry
}

end annual_parking_savings_l69_69775


namespace hours_per_day_l69_69395

variable (M : ℕ)

noncomputable def H : ℕ := 9
noncomputable def D1 : ℕ := 24
noncomputable def Men2 : ℕ := 12
noncomputable def D2 : ℕ := 16

theorem hours_per_day (H_new : ℝ) : 
  (M * H * D1 : ℝ) = (Men2 * H_new * D2) → 
  H_new = (M * 9 : ℝ) / 8 := 
  sorry

end hours_per_day_l69_69395


namespace solution_is_permutations_of_2_neg2_4_l69_69466

-- Definitions of the conditions
def cond1 (x y z : ℤ) : Prop := x * y + y * z + z * x = -4
def cond2 (x y z : ℤ) : Prop := x^2 + y^2 + z^2 = 24
def cond3 (x y z : ℤ) : Prop := x^3 + y^3 + z^3 + 3 * x * y * z = 16

-- The set of all integer solutions as permutations of (2, -2, 4)
def is_solution (x y z : ℤ) : Prop :=
  (x = 2 ∧ y = -2 ∧ z = 4) ∨ (x = 2 ∧ y = 4 ∧ z = -2) ∨
  (x = -2 ∧ y = 2 ∧ z = 4) ∨ (x = -2 ∧ y = 4 ∧ z = 2) ∨
  (x = 4 ∧ y = 2 ∧ z = -2) ∨ (x = 4 ∧ y = -2 ∧ z = 2)

-- Lean statement for the proof problem
theorem solution_is_permutations_of_2_neg2_4 (x y z : ℤ) :
  cond1 x y z → cond2 x y z → cond3 x y z → is_solution x y z :=
by
  -- sorry, the proof goes here
  sorry

end solution_is_permutations_of_2_neg2_4_l69_69466


namespace sports_club_problem_l69_69596

theorem sports_club_problem (total_members : ℕ) (members_playing_badminton : ℕ) 
  (members_playing_tennis : ℕ) (members_not_playing_either : ℕ) 
  (h_total_members : total_members = 100) (h_badminton : members_playing_badminton = 60) 
  (h_tennis : members_playing_tennis = 70) (h_neither : members_not_playing_either = 10) : 
  (members_playing_badminton + members_playing_tennis - 
   (total_members - members_not_playing_either) = 40) :=
by {
  sorry
}

end sports_club_problem_l69_69596


namespace separation_lines_requirement_l69_69283

noncomputable def convex_polygon (n : ℕ) := {vertices // convex vertices}

theorem separation_lines_requirement (K : Type) (P : K)
  (h_convex: convex_polygon K)
  (h_interior: ∃ (P : K), is_interior P K) :
  (∃ lines : set (line K), lines.card = 4 ∧ ∀ side ∈ K, ∃ l ∈ lines, separates l P side) ∧
  ¬(∃ lines : set (line K), lines.card = 3 ∧ ∀ side ∈ K, ∃ l ∈ lines, separates l P side) :=
by
  sorry

end separation_lines_requirement_l69_69283


namespace speed_of_sound_in_kmph_is_correct_l69_69094

def speed_of_sound_mps : ℝ := 343
def conversion_factor : ℝ := 3.6
def speed_of_sound_kmph := speed_of_sound_mps * conversion_factor

theorem speed_of_sound_in_kmph_is_correct :
  (Real.toDecimalString speed_of_sound_kmph 3) = "1234.800" := by
  sorry

end speed_of_sound_in_kmph_is_correct_l69_69094


namespace four_digit_integers_count_l69_69562

theorem four_digit_integers_count : 
  let first_three_choices := {2, 5}
  let last_digit_choices := {2, 5, 8}
  ∃ (count : Nat), count = (first_three_choices.card ^ 3) * last_digit_choices.card :=
by
  have h_first : first_three_choices.card = 2 := by decide
  have h_last : last_digit_choices.card = 3 := by decide
  use 24
  simp [h_first, h_last]
  sorry

end four_digit_integers_count_l69_69562


namespace number_of_ordered_pairs_l69_69563

theorem number_of_ordered_pairs :
  {m n : ℕ // 0 < m ∧ 0 < n ∧ m ≥ n ∧ m^2 - n^2 = 120}.card = 4 :=
sorry

end number_of_ordered_pairs_l69_69563


namespace prob_C_or_D_eq_l69_69397

theorem prob_C_or_D_eq :
  ∀ (P : Type → ℝ) (A B C D : Type),
  P A = 1 / 4 →
  P B = 1 / 3 →
  P A + P B + P C + P D = 1 →
  P C + P D = 5 / 12 :=
by
  intros,
  sorry

end prob_C_or_D_eq_l69_69397


namespace ellipse_eccentricity_l69_69152

theorem ellipse_eccentricity
  (a b : ℝ) (h_ab : a > b) (h_b : b > 0)
  (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ)
  (P : ℝ × ℝ) (hP : P = (1, 1))
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hA : A = (x₁, y₁)) (hB : B = (x₂, y₂)) (hC : C = (x₃, y₃)) (hD : D = (x₄, y₄))
  (h_ellipse : ∀ p : ℝ × ℝ, (p = A ∨ p = B ∨ p = C ∨ p = D) → (p.1 / a) ^ 2 + (p.2 / b) ^ 2 = 1)
  (h_vectors : ∀ λ : ℝ, (1 - x₁, 1 - y₁) = λ • (x₃ - 1, y₃ - 1) ∧ (1 - x₂, 1 - y₂) = λ • (x₄ - 1, y₄ - 1))
  (h_slope_CD : ∀ λ : ℝ, (y₄ - y₃) / (x₄ - x₃) = -1/4) :
  (sqrt(1 - (b / a) ^ 2) = sqrt(3) / 2) :=
sorry

end ellipse_eccentricity_l69_69152


namespace shifted_parabola_l69_69336

theorem shifted_parabola (x : ℝ) :
  let y := 2 * x^2 in
  let y_shifted_right := 2 * (x - 4)^2 in
  let y_shifted_final := 2 * (x - 4)^2 - 3 in
  y_shifted_final = 2 * (x - 4)^2 - 3 :=
by
  sorry

end shifted_parabola_l69_69336


namespace vector_add_sub_l69_69192

open Matrix

def a : Fin 3 → ℝ := ![3, -4, 2]
def b : Fin 3 → ℝ := ![-2, 5, 3]
def c : Fin 3 → ℝ := ![1, 1, -4]

theorem vector_add_sub:
  a + 2 • b - c = ![-2, 5, 12] :=
by
  /- Proof will go here -/
  sorry

end vector_add_sub_l69_69192


namespace trig_expression_identity_l69_69344

noncomputable def trig_expression (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2 * π - α) * tan (-α + 3 / 2 * π)) /
  (cot (-α - π) * sin (-π + α))

theorem trig_expression_identity (α : ℝ) : trig_expression α = cos α * sin α := by
  sorry

end trig_expression_identity_l69_69344


namespace intersection_M_N_l69_69926

def M : set ℝ := { x | x < 1 }
def N : set ℝ := { x | 2 ^ x > 1 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
by sorry

end intersection_M_N_l69_69926


namespace least_n_condition_l69_69360

theorem least_n_condition (n : ℕ) (h1 : ∀ k : ℕ, 1 ≤ k → k ≤ n + 1 → (k ∣ n * (n - 1) → k ≠ n + 1)) : n = 4 :=
sorry

end least_n_condition_l69_69360


namespace polynomial_roots_sum_l69_69271

theorem polynomial_roots_sum (p q : ℂ) (hp : p + q = 5) (hq : p * q = 7) : 
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 559 := 
by 
  sorry

end polynomial_roots_sum_l69_69271


namespace equalSumSeqDefinition_l69_69442

def isEqualSumSeq (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n - 1) + s n = s (n + 1)

theorem equalSumSeqDefinition (s : ℕ → ℝ) :
  isEqualSumSeq s ↔ 
  ∀ n : ℕ, n > 0 → s n = s (n - 1) + s (n + 1) :=
by
  sorry

end equalSumSeqDefinition_l69_69442


namespace tetrahedron_faces_acute_l69_69784

theorem tetrahedron_faces_acute
  {A B C D : Type} 
  (ABC : triangle A B C) 
  (AB_DC : distance A B = distance D C)
  (AC_DB : distance A C = distance D B)
  (BC_DA : distance B C = distance D A) :
  (acute_triangle (face1 ABC D) ∧ acute_triangle (face2 ABC D) ∧ acute_triangle (face3 ABC D)) :=
sorry

end tetrahedron_faces_acute_l69_69784


namespace algebraic_expression_value_l69_69578

theorem algebraic_expression_value (m n : ℕ) (h1 : m + 2 = 6) (h2 : n + 1 = 4) : - (m ^ n) = -64 := by
  sorry

end algebraic_expression_value_l69_69578


namespace surfaces_ratio_l69_69781

def surface_area_inscribed_cone (r : ℝ) : ℝ :=
  let R := (r / 2) * Real.sqrt 3
  let l := (Real.sqrt 7 / 2) * r
  π * R * l

def surface_area_sphere (r : ℝ) : ℝ :=
  4 * π * r^2

def surface_area_circumscribed_cone (r : ℝ) : ℝ :=
  let R' := r * Real.sqrt 3
  let l := Real.sqrt 4 * r -- l for circumscribed cone also equals r in this configuration
  π * R' * l

theorem surfaces_ratio (r : ℝ) :
  let f1 := surface_area_inscribed_cone r
  let f2 := surface_area_sphere r
  let f3 := surface_area_circumscribed_cone r
  f1 / f2 = 9 / 16 ∧ f2 / f3 = 16 / 36 :=
by
  sorry

end surfaces_ratio_l69_69781


namespace race_distance_l69_69292

variable (speed_cristina speed_nicky head_start time_nicky : ℝ)

theorem race_distance
  (h1 : speed_cristina = 5)
  (h2 : speed_nicky = 3)
  (h3 : head_start = 12)
  (h4 : time_nicky = 30) :
  let time_cristina := time_nicky - head_start
  let distance_nicky := speed_nicky * time_nicky
  let distance_cristina := speed_cristina * time_cristina
  distance_nicky = 90 ∧ distance_cristina = 90 :=
by
  sorry

end race_distance_l69_69292


namespace students_per_school_in_lansing_l69_69254

theorem students_per_school_in_lansing (total_students : ℕ) (num_schools : ℕ) (students : ℕ) (h1 : total_students = 6175) (h2 : num_schools = 25) : students = total_students / num_schools :=
by
  rw [h1, h2]
  norm_num
  exact dec_trivial

end students_per_school_in_lansing_l69_69254


namespace geometric_sequence_logarithm_identity_l69_69899

variable {a : ℕ+ → ℝ}

-- Assumptions
def common_ratio (a : ℕ+ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_logarithm_identity
  (r : ℝ)
  (hr : r = -Real.sqrt 2)
  (h : common_ratio a r) :
  Real.log (a 2017)^2 - Real.log (a 2016)^2 = Real.log 2 :=
by
  sorry

end geometric_sequence_logarithm_identity_l69_69899


namespace sequence_b_geometric_general_term_of_a_sum_first_n_terms_of_c_l69_69712

-- Definitions
def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = 3 * a n + 4

def sequence_b (b : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, b n = a n + 2

-- Proving b is geometric
theorem sequence_b_geometric (a b : ℕ → ℤ) (h_a : sequence_a a) (h_b : sequence_b b a) :
  ∃ q : ℤ, (∀ n : ℕ, b (n + 1) = q * b n) :=
by
  sorry

-- Finding the general term of a
theorem general_term_of_a (a b : ℕ → ℤ) (h_a : sequence_a a) (h_b : sequence_b b a)
  (h_b_geom : ∃ q : ℤ, (∀ n : ℕ, b (n + 1) = q * b n)) :
  ∀ n : ℕ, a n = 3 ^ n - 2 :=
by
  sorry

-- Sum of the first n terms of sequence c
def sequence_c (c a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, c n = (2 * n - 1) * (a n + 2)

theorem sum_first_n_terms_of_c (a b c : ℕ → ℤ) (h_a : sequence_a a) (h_b : sequence_b b a)
  (h_b_geom : ∃ q : ℤ, (∀ n : ℕ, b (n + 1) = q * b n)) (h_term : ∀ n : ℕ, a n = 3 ^ n - 2)
  (h_c : sequence_c c a) :
  ∀ n : ℕ, (∑ k in Finset.range (n + 1), c k) = (n + 1) * 3 ^ (n + 1) + 3 :=
by
  sorry

end sequence_b_geometric_general_term_of_a_sum_first_n_terms_of_c_l69_69712


namespace inequality_equivalence_l69_69102

theorem inequality_equivalence (x : ℝ) : 
  (x + 2) / (x - 1) ≥ 0 ↔ (x + 2) * (x - 1) ≥ 0 :=
sorry

end inequality_equivalence_l69_69102


namespace product_of_digits_of_next_palindromic_year_after_2021_l69_69724

def is_palindromic_year (y : ℕ) : Prop :=
  let digits := y.to_digits 10
  digits = digits.reverse

noncomputable def next_palindromic_year (start : ℕ) : ℕ :=
  Nat.find (λ y, y > start ∧ is_palindromic_year y)

theorem product_of_digits_of_next_palindromic_year_after_2021 :
  (next_palindromic_year 2021) = 2121 ∧ (2121.to_digits 10).prod = 4 := by
  sorry

end product_of_digits_of_next_palindromic_year_after_2021_l69_69724


namespace monic_quadratic_real_root_l69_69834

theorem monic_quadratic_real_root (a b : ℂ) (h : b = 2 - 3 * complex.I) :
  ∃ P : polynomial ℂ, P.monic ∧ P.coeff 2 = 1 ∧ P.coeff 1 = -4 ∧ P.coeff 0 = 13 ∧ P.is_root (2 - 3 * complex.I) :=
by
  sorry

end monic_quadratic_real_root_l69_69834


namespace technician_round_trip_l69_69783

-- Definitions based on conditions
def trip_to_center_completion : ℝ := 0.5 -- Driving to the center is 50% of the trip
def trip_from_center_completion (percent_completed: ℝ) : ℝ := 0.5 * percent_completed -- Completion percentage of the return trip
def total_trip_completion : ℝ := trip_to_center_completion + trip_from_center_completion 0.3 -- Total percentage completed

-- Theorem statement
theorem technician_round_trip : total_trip_completion = 0.65 :=
by
  sorry

end technician_round_trip_l69_69783


namespace toaster_total_cost_l69_69626

theorem toaster_total_cost :
  let MSRP := 30
  let insurance_rate := 0.20
  let premium_upgrade := 7
  let recycling_fee := 5
  let tax_rate := 0.50

  -- Calculate costs
  let insurance_cost := insurance_rate * MSRP
  let total_insurance_cost := insurance_cost + premium_upgrade
  let cost_before_tax := MSRP + total_insurance_cost + recycling_fee
  let state_tax := tax_rate * cost_before_tax
  let total_cost := cost_before_tax + state_tax

  -- Total cost Jon must pay
  total_cost = 72 :=
by
  sorry

end toaster_total_cost_l69_69626


namespace count_four_digit_palindrome_squares_l69_69027

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69027


namespace angles_with_same_terminal_side_l69_69713

theorem angles_with_same_terminal_side (k : ℤ) : 
  (∃ (α : ℝ), α = -437 + k * 360) ↔ (∃ (β : ℝ), β = 283 + k * 360) := 
by
  sorry

end angles_with_same_terminal_side_l69_69713


namespace find_k_l69_69515

-- Defining the vectors and the condition for parallelism
def vector_a := (2, 1)
def vector_b (k : ℝ) := (k, 3)

def vector_parallel_condition (k : ℝ) : Prop :=
  let a2b := (2 + 2 * k, 7)
  let a2nb := (4 - k, -1)
  (2 + 2 * k) * (-1) = 7 * (4 - k)

theorem find_k (k : ℝ) (h : vector_parallel_condition k) : k = 6 :=
by
  sorry

end find_k_l69_69515


namespace triangles_from_pentadecagon_l69_69933

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end triangles_from_pentadecagon_l69_69933


namespace justin_and_tim_play_together_210_l69_69815

theorem justin_and_tim_play_together_210 :
  ∀ (P : Finset ℕ), P.card = 12 → 
  (∀ S : Finset ℕ, S.card = 6 → ∃ games : Finset (Finset ℕ), games.card = 924 ∧ ∀ S' ∈ games, S'.card = 6) →
  ∃ J T : ℕ, J ≠ T ∧ J ∈ P ∧ T ∈ P ∧ 
  (∃ match_count : ℕ, match_count = 210 ∧ 
   ∀ game : Finset (Finset ℕ), game ∈ games → 
   (J ∈ game ∧ T ∈ game → count_appearance (J, T) games = match_count)) :=
sorry

end justin_and_tim_play_together_210_l69_69815


namespace medians_intersect_at_centroid_l69_69367

-- Only the relevant definitions and conditions from a)
-- Restate the problem in Lean

def median (A B C : Point) (M : Point) : Prop :=
  ∃ P : Point, is_midpoint B C P ∧ line_through A M

def centroid (A B C : Point) (G : Point) : Prop :=
  median A B C P ∧ median B C A Q ∧ median C A B R ∧ G = three_point_intersection P Q R

theorem medians_intersect_at_centroid (A B C : Point) :
  ∃ G : Point, centroid A B C G :=
sorry

end medians_intersect_at_centroid_l69_69367


namespace simplify_trigonometric_expr_l69_69313

theorem simplify_trigonometric_expr (x : ℝ) :
  sin (x + π / 3) + 2 * sin (x - π / 3) - sqrt 3 * cos (2 * π / 3 - x) = 0 :=
by
  sorry

end simplify_trigonometric_expr_l69_69313


namespace four_digit_palindrome_square_count_l69_69002

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l69_69002


namespace same_type_square_root_l69_69215

theorem same_type_square_root (k : ℕ) : (real.sqrt (2 * k - 4) = 2 * real.sqrt 3) ↔ (k = 8) :=
by
  sorry

end same_type_square_root_l69_69215


namespace hoseoks_social_studies_score_l69_69196

theorem hoseoks_social_studies_score 
  (avg_three_subjects : ℕ) 
  (new_avg_with_social_studies : ℕ) 
  (total_score_three_subjects : ℕ) 
  (total_score_four_subjects : ℕ) 
  (S : ℕ)
  (h1 : avg_three_subjects = 89) 
  (h2 : new_avg_with_social_studies = 90) 
  (h3 : total_score_three_subjects = 3 * avg_three_subjects) 
  (h4 : total_score_four_subjects = 4 * new_avg_with_social_studies) :
  S = 93 :=
sorry

end hoseoks_social_studies_score_l69_69196


namespace four_digit_perfect_square_palindrome_count_l69_69009

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69009


namespace pentadecagon_triangle_count_l69_69962

-- Define the problem of selecting 3 vertices out of 15 to form a triangle
theorem pentadecagon_triangle_count : 
  ∃ (n : ℕ), n = nat.choose 15 3 ∧ n = 455 := 
by {
  sorry
}

end pentadecagon_triangle_count_l69_69962


namespace pentadecagon_triangle_count_l69_69960

-- Define the problem of selecting 3 vertices out of 15 to form a triangle
theorem pentadecagon_triangle_count : 
  ∃ (n : ℕ), n = nat.choose 15 3 ∧ n = 455 := 
by {
  sorry
}

end pentadecagon_triangle_count_l69_69960


namespace find_f_at_neg_one_third_l69_69900

noncomputable theory

variables (f : ℝ → ℝ)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

axiom f_defined_on_reals : ∀ x, f x ∈ ℝ
axiom f_even : is_even f
axiom f_definition : ∀ x, x > 0 → f x = 0.001^x

theorem find_f_at_neg_one_third : f (-1/3) = 1/10 := 
by 
  have h1 : f (1/3) = 0.001^(1/3), from f_definition 1/3 (by norm_num),
  have h2 : f (-1/3) = f (1/3), from f_even 1/3,
  rw h2,
  rw h1,
  -- finished proof; we're skipping it with 'sorry'
  sorry

end find_f_at_neg_one_third_l69_69900


namespace general_formula_a_minimum_value_Sn_l69_69171

-- Define the sequence {a_n} with given product T_n
def T : ℕ → ℝ := λ n, 2^(n^2 - 12 * n)

-- Define the sequence {a_n}
def a (n : ℕ) : ℝ := if n = 1 then 2^(-11) else T n / T (n - 1)

-- Define the sequence {b_n} as the logarithm base 2 of {a_n}
def b (n : ℕ) : ℝ := Real.logb 2 (a n)

-- Define S_n as the sum of the first n terms of {b_n}
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k + 1))

-- Theorem Part (Ⅰ)
theorem general_formula_a : ∀ n : ℕ, a n = 2^(2 * n - 13) :=
by 
-- proof omitted 
sorr

-- Theorem Part (Ⅱ)
theorem minimum_value_Sn : ∃ n : ℕ, S n = -36 :=
by 
-- proof omitted 
sorr

end general_formula_a_minimum_value_Sn_l69_69171


namespace math_problem_l69_69273
noncomputable section

variables {x y : ℝ}

def condition1 : Prop := (tan x / tan y) = 2
def condition2 : Prop := (sin x / sin y) = 4

theorem math_problem (h1 : condition1) (h2 : condition2) : 
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 4.6 :=
by
  sorry

end math_problem_l69_69273


namespace factorize_expression_l69_69488

theorem factorize_expression (a x y : ℝ) :
  a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 :=
by
  sorry

end factorize_expression_l69_69488


namespace incorrect_statement_option_B_l69_69368

theorem incorrect_statement_option_B 
  (A : ∃ (data : List ℝ), (data.sum / data.length) = median data)
  (B : ∀ (data : List ℝ), card (filter (λ x, x < median data) data) = card (filter (λ x, x > median data) data))
  (C : ∀ (data : List ℝ), (mean data = central_tendency data) ∧ (median data = central_tendency data) ∧ (mode data = central_tendency data))
  (D : ∀ (data : List ℝ), (range data = dispersion data) ∧ (variance data = dispersion data) ∧ (std_dev data = dispersion data)) : 
  B = false :=
sorry

end incorrect_statement_option_B_l69_69368


namespace propositions_are_correct_l69_69891

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then log (x + 1) / log 2
  else if 1 ≤ x ∧ x < 2 then -log (x - 1 + 1) / log 2
  else if x < 0 then f (-x)
  else f ((x % 2) + 2 * ((* ! 0 ≤ x ? 0) - (* ! x < 2 ? 0)))

-- Conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x → f (x + 1) = -f x

def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x < 1 → f x = log (x + 1) / log 2

def prop1 (f : ℝ → ℝ) : Prop :=
  f 2014 + f (-2015) = 0

def prop2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

def prop3 (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x = y ∧ x = y ∧ (x, y) ≠ (0, 0))

def prop4 (f : ℝ → ℝ) : Prop :=
  ∀ y, -1 < y ∧ y < 1 → ∃ x, f x = y

theorem propositions_are_correct :
  even_function f ∧ condition1 f ∧ condition2 f →
  prop1 f ∧ prop4 f ∧ ¬ prop2 f ∧ ¬ prop3 f :=
by
  sorry

end propositions_are_correct_l69_69891


namespace count_four_digit_palindrome_squares_l69_69031

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69031


namespace max_value_of_function_l69_69703

noncomputable def function_y (x : ℝ) : ℝ := x + Real.sin x

theorem max_value_of_function : 
  ∀ (a b : ℝ), a = 0 → b = Real.pi → 
  (∀ x : ℝ, x ∈ Set.Icc a b → x + Real.sin x ≤ Real.pi) :=
by
  intros a b ha hb x hx
  sorry

end max_value_of_function_l69_69703


namespace two_dice_same_result_probability_l69_69201

theorem two_dice_same_result_probability :
  let p_purple := (6 / 30) * (6 / 30)
  let p_green := (10 / 30) * (10 / 30)
  let p_orange := (12 / 30) * (12 / 30)
  let p_glittery := (2 / 30) * (2 / 30)
  p_purple + p_green + p_orange + p_glittery = 71 / 225 :=
by
  let p_purple := (6 / 30) * (6 / 30)
  let p_green := (10 / 30) * (10 / 30)
  let p_orange := (12 / 30) * (12 / 30)
  let p_glittery := (2 / 30) * (2 / 30)
  have h1 : p_purple = 1 / 25 := sorry
  have h2 : p_green = 1 / 9 := sorry
  have h3 : p_orange = 4 / 25 := sorry
  have h4 : p_glittery = 1 / 225 := sorry
  calc 
    p_purple + p_green + p_orange + p_glittery
      = 1 / 25 + 1 / 9 + 4 / 25 + 1 / 225 : by rw [h1, h2, h3, h4]
  ... = 71 / 225 : sorry

end two_dice_same_result_probability_l69_69201


namespace part1_part2_l69_69055

-- Proof for Part 1
theorem part1 (x : ℝ) (h1 : 360 / x + 10 = 360 / (0.9 * x)) : x = 4 :=
sorry

-- Proof for Part 2
theorem part2 (y : ℕ) 
  (h2 : 80 - y ≥ 0)
  (h3 : 400 ≤ (4 * 0.8 * y + 10 * 0.8 * (80 - y)))
  : y ≤ 50 :=
sorry

end part1_part2_l69_69055


namespace additional_people_can_ride_l69_69720

def num_cars := 2
def num_vans := 3
def people_per_car := 5
def people_per_van := 3
def max_people_per_car := 6
def max_people_per_van := 8

theorem additional_people_can_ride :
  let actual_people := num_cars * people_per_car + num_vans * people_per_van in
  let max_people := num_cars * max_people_per_car + num_vans * max_people_per_van in
  max_people - actual_people = 17 :=
by
  sorry

end additional_people_can_ride_l69_69720


namespace part1_part2_l69_69173

-- Define the universal set U as real numbers ℝ
def U : Set ℝ := Set.univ

-- Define Set A
def A (a : ℝ) : Set ℝ := {x | abs (x - a) ≤ 1 }

-- Define Set B
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0 }

-- Part 1: Prove A ∪ B when a = 4
theorem part1 : A 4 ∪ B = {x : ℝ | x ≥ 3 ∨ x ≤ 1} :=
sorry

-- Part 2: Prove the range of values for a given A ∩ B = A
theorem part2 (a : ℝ) (h : A a ∩ B = A a) : a ≥ 5 ∨ a ≤ 0 :=
sorry

end part1_part2_l69_69173


namespace rachel_homework_total_l69_69304

-- Definitions based on conditions
def math_homework : Nat := 8
def biology_homework : Nat := 3

-- Theorem based on the problem statement
theorem rachel_homework_total : math_homework + biology_homework = 11 := by
  -- typically, here you would provide a proof, but we use sorry to skip it
  sorry

end rachel_homework_total_l69_69304


namespace triangles_from_pentadecagon_l69_69934

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end triangles_from_pentadecagon_l69_69934


namespace Q_at_1_value_l69_69463

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 6
def mean_coeff_P : ℝ := ((3 : ℝ) + (-5 : ℝ) + 2 + (-6)) / 4
def Q (x : ℝ) : ℝ := mean_coeff_P * x^3 + mean_coeff_P * x^2 + mean_coeff_P * x + mean_coeff_P

theorem Q_at_1_value : Q 1 = -6 :=
by
  rw [Q, mean_coeff_P]
  norm_num
  sorry

end Q_at_1_value_l69_69463


namespace infinite_product_value_l69_69473

theorem infinite_product_value : (3 ^ (1 / 4) * (9 : ℝ) ^ (1 / 16) * (27 : ℝ) ^ (1 / 64) * (81 : ℝ) ^ (1 / 256) * ∏ n in (set.Icc 5 (∞ : ℕ)), (3 ^ n : ℝ) ^ (1 / (4 ^ n))) = real.sqrt 3 := 
by
  sorry

end infinite_product_value_l69_69473


namespace add_to_fraction_eq_l69_69366

theorem add_to_fraction_eq (n : ℤ) : (4 + n : ℤ) / (7 + n) = (2 : ℤ) / 3 → n = 2 := 
by {
  sorry
}

end add_to_fraction_eq_l69_69366


namespace roots_of_polynomial_l69_69845

noncomputable def quadratic_polynomial (x : ℝ) := x^2 - 4 * x + 13

theorem roots_of_polynomial : ∀ (z : ℂ), z = 2 - 3 * complex.I → quadratic_polynomial (z.re) = 0 :=
by
  intro z h
  sorry

end roots_of_polynomial_l69_69845


namespace spencer_jumps_per_minute_l69_69739

theorem spencer_jumps_per_minute :
  (10 * 2 * 5 * J = 400) → (J = 4) :=
by {
  intros h,
  rw [mul_assoc, mul_assoc (10 * 2) 5 J],
  rw [mul_comm 5 J, ← mul_assoc, ← mul_assoc 10 2 5, ← mul_assoc 20 5 J] at h,
  linarith,
  sorry -- This is to skip the proof
}

end spencer_jumps_per_minute_l69_69739


namespace expansion_terms_max_min_l69_69545

-- Definitions of binomial coefficient and the expansion terms
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

def expansion_term (n r : ℕ) : ℤ × ℕ :=
  let coeff := binomial_coefficient n r * (-1)^r
  let exponent := 16 - 3 * r
  (coeff, exponent)

-- Main theorem to prove
theorem expansion_terms_max_min
  (n : ℕ)
  (H : 2^n - 2^7 = 128) :
  let T_max := expansion_term n 4
  let T_min_1 := expansion_term n 3
  let T_min_2 := expansion_term n 5 in
  n = 8 ∧ 
  T_max = (70, 4) ∧
  (T_min_1 = (-56, 7) ∨ T_min_2 = (-56, 1)) := by
  sorry

end expansion_terms_max_min_l69_69545


namespace sqrt_expression_l69_69085

theorem sqrt_expression : 2 * Real.sqrt 3 - (3 * Real.sqrt 2 + Real.sqrt 3) = Real.sqrt 3 - 3 * Real.sqrt 2 :=
by
  sorry

end sqrt_expression_l69_69085


namespace monic_quadratic_polynomial_with_root_l69_69857

theorem monic_quadratic_polynomial_with_root (x : ℝ) : 
  ∃ p : polynomial ℝ, monic p ∧ p.coeff 1 =  -4 ∧ p.coeff 0 = 13 ∧ (∀ z : ℂ, z = (2 - 3 * I) → p.eval z.re = 0) :=
sorry

end monic_quadratic_polynomial_with_root_l69_69857


namespace price_is_75_percent_of_combined_assets_l69_69455

variable (P A B : ℝ)

def company_A_condition : Prop :=
  P = 1.20 * A

def company_B_condition : Prop :=
  P = 2.00 * B

theorem price_is_75_percent_of_combined_assets
  (hA : company_A_condition P A B)
  (hB : company_B_condition P A B) : 
  P / (A + B) * 100 = 75 :=
by sorry

end price_is_75_percent_of_combined_assets_l69_69455


namespace problem_b_l69_69912

noncomputable def f (x : ℝ) := x - x^2 + 3 * Real.log x

theorem problem_b (x : ℝ) (hx : 0 < x) : f x ≤ 2 * x - 2 :=
by {
  have h1 : f(1) = 0,
  { unfold f,
    norm_num,
    exact Real.log_one },
  have h2 : deriv f 1 = 2,
  { unfold f,
    rw deriv_sub,
    rw deriv_sub,
    rw deriv_add_const,
    rw deriv_neg,
    rw deriv_pow 2,
    rw deriv_mul,
    rw deriv_const_mul,
    rw deriv_log,
    norm_num,
    split,
    norm_num,
    exact Real.log_pos one_ne_zero },

  /- g(x) defined for x > 0, such that
     g(x) = f(x) -(2x - 2) -/
  let g : ℝ → ℝ := λ x, f x - (2 * x - 2),
  unfold f at h1,
  unfold f at h2,
  have hg : deriv g 1 = 0,
  { unfold g,
    rw deriv_sub,
    rw h2,
    norm_num,
    simp only [deriv_const_mul],
    simp only [deriv_add_const, deriv_sub, deriv_id', deriv_const] },  

  sorry
}

end problem_b_l69_69912


namespace minimize_f_at_75_l69_69768

-- Definitions for the conditions
def num_workers : ℕ := 100
def total_units : ℕ := 1000
def devices_A_per_unit : ℕ := 9
def devices_B_per_unit : ℕ := 3
def workers_A_per_hour : ℕ := 1
def workers_B_per_hour : ℕ := 3

-- Function definitions
def total_devices_A : ℕ := total_units * devices_A_per_unit
def total_devices_B : ℕ := total_units * devices_B_per_unit

-- Define t1 and t2 in terms of x
def t1 (x : ℕ) := total_devices_A / x
def t2 (x : ℕ) := total_devices_B / (workers_B_per_hour * (num_workers - x))

-- Define f(x)
def f (x : ℕ) := t1 x + t2 x

-- Define the domain
def domain (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 99 ∧ x ∈ {1, 2, ..., 99}

-- The theorem statement
theorem minimize_f_at_75 : ∃ x : ℕ, domain x ∧ f x = 75 := 
sorry

end minimize_f_at_75_l69_69768


namespace inequality_induction_harmonic_sum_exceeds_harmonic_sum_4_5_l69_69674

theorem inequality_induction (n : ℕ) : 
  (∑ i in finset.range (2 * n + 1), 1 / (n + i + 1)) > 1 := 
sorry

theorem harmonic_sum_exceeds (A : ℝ) : ∃ N : ℕ, (∑ i in finset.range N, 1 / (i + 1)) > A :=
sorry

theorem harmonic_sum_4_5 : ∃ n ≤ 67, (∑ i in finset.range (n + 1), 1 / (i + 1)) > 4.5 :=
sorry

end inequality_induction_harmonic_sum_exceeds_harmonic_sum_4_5_l69_69674


namespace value_of_sqrt_x_plus_one_over_sqrt_x_l69_69650

noncomputable def find_value (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) : ℝ :=
  sqrt(x) + 1/sqrt(x)

theorem value_of_sqrt_x_plus_one_over_sqrt_x (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) :
  find_value x hx_pos hx = 2 * sqrt(13) :=
sorry

end value_of_sqrt_x_plus_one_over_sqrt_x_l69_69650


namespace area_of_quadrilateral_APQD_l69_69414

-- Definitions for the geometric setup
def point := (ℝ × ℝ)
def square (A B C D : point) (radius : ℝ) (center : point) : Prop :=
  A = (0, 2) ∧ B = (2, 2) ∧ C = (2, 0) ∧ D = (0, 0) ∧ 
  center = (1, 1) ∧ radius = real.sqrt 2

def lies_on_arc (M : point) (center : point) (radius : ℝ) : Prop :=
  (M.1 - center.1) ^ 2 + (M.2 - center.2) ^ 2 = radius ^ 2 ∧ M.1 > 1

def intersection_point (A M B D P Q : point) : Prop :=
  ∃ k : ℝ, P = (A.1 + k * (M.1 - A.1), A.2 + k * (M.2 - A.2)) ∧ 
  ∃ l : ℝ, Q = (D.1 + l * (M.1 - D.1), D.2 + l * (M.2 - D.2))

-- Final theorem statement
theorem area_of_quadrilateral_APQD (A B C D M P Q : point) (center : point) (radius : ℝ) :
  square A B C D radius center →
  lies_on_arc M center radius →
  intersection_point A M B D P Q →
  let area_sq := 4 in
  let area_quad_APQD := 2 in
  (area_AQD : ℝ) :=
  sorry

end area_of_quadrilateral_APQD_l69_69414


namespace count_even_rows_in_first_30_pascal_l69_69096

theorem count_even_rows_in_first_30_pascal : 
  let rows := [2, 4, 8, 16] in
  rows.length = 4 :=
by 
  let rows := [2, 4, 8, 16]
  exact rfl

end count_even_rows_in_first_30_pascal_l69_69096


namespace players_count_l69_69425

def total_socks : ℕ := 22
def socks_per_player : ℕ := 2

theorem players_count : total_socks / socks_per_player = 11 :=
by
  sorry

end players_count_l69_69425


namespace binomial_expansion_coefficient_l69_69606

theorem binomial_expansion_coefficient :
  let f := λ (x : ℕ) (r : ℕ), (Nat.choose 5 r) * 2^(5-r) * (-1)^r
  in f 1 3 = -40 := by sorry

end binomial_expansion_coefficient_l69_69606


namespace proof_problem_l69_69887

noncomputable theory
open_locale classical

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨1, 0⟩
def B : Point := ⟨0, 1⟩
def C : Point := ⟨-3, -2⟩

def line_equation_bc (p1 p2 : Point) : ℝ × ℝ × ℝ :=
  (p2.y - p1.y, p1.x - p2.x, p1.y * p2.x - p1.x * p2.y)

def triangle_is_right_angled (a b c : Point) : Prop :=
  let ab_slope := (b.y - a.y) / (b.x - a.x) in
  let bc_slope := (c.y - b.y) / (c.x - b.x) in
  ab_slope * bc_slope = -1

def circumcircle_equation (a b c : Point) : (ℝ × ℝ) × ℝ :=
  let x_center := (a.x + b.x + c.x) / 3 in
  let y_center := (a.y + b.y + c.y) / 3 in
  let radius := dist ⟨a.x, a.y⟩ ⟨x_center, y_center⟩ in
  ((-1, -1), 5)

theorem proof_problem :
  let bc_line := line_equation_bc B C in
  bc_line = (1, -1, 1) ∧
  triangle_is_right_angled A B C ∧
  circumcircle_equation A B C = ((-1, -1), 5) :=
sorry

end proof_problem_l69_69887


namespace parabola_directrix_l69_69919

theorem parabola_directrix (a : ℝ) (h1 : ∀ x : ℝ, - (1 / (4 * a)) = 2):
  a = -(1 / 8) :=
sorry

end parabola_directrix_l69_69919


namespace triangles_from_pentadecagon_l69_69937

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end triangles_from_pentadecagon_l69_69937


namespace shape_of_theta_eq_c_l69_69505

-- Definitions based on given conditions
def azimuthal_angle (rho theta phi : ℝ) : ℝ := theta

-- The main theorem we want to prove
theorem shape_of_theta_eq_c (c : ℝ) : 
  (∀ ρ φ, azimuthal_angle ρ c φ = c) → 
  (∃ a b, is_plane (λ (ρ θ φ : ℝ), θ = c) a b) :=
sorry

end shape_of_theta_eq_c_l69_69505


namespace total_books_l69_69727

open Finset

-- Define a set of students as a finite type
inductive Student : Type
| student1 : Student
| student2 : Student
| student3 : Student
| student4 : Student
| student5 : Student
| student6 : Student

-- Define a set of books as an indeterminate set
inductive Book : Type
| book : Nat → Book

-- Define a relation representing each student owning a set of books
def owns (s : Student) (b : Book) : Prop :=
  match s with
  | Student.student1 => b = Book.book 1 ∨ b = Book.book 2 ∨ b = Book.book 3 ∨ b = Book.book 4 ∨ b = Book.book 5
  | Student.student2 => b = Book.book 1 ∨ b = Book.book 6 ∨ b = Book.book 7 ∨ b = Book.book 8 ∨ b = Book.book 9
  | Student.student3 => b = Book.book 2 ∨ b = Book.book 6 ∨ b = Book.book 10 ∨ b = Book.book 11 ∨ b = Book.book 12
  | Student.student4 => b = Book.book 3 ∨ b = Book.book 7 ∨ b = Book.book 10 ∨ b = Book.book 13 ∨ b = Book.book 14
  | Student.student5 => b = Book.book 4 ∨ b = Book.book 8 ∨ b = Book.book 11 ∨ b = Book.book 13 ∨ b = Book.book 15
  | Student.student6 => b = Book.book 5 ∨ b = Book.book 9 ∨ b = Book.book 12 ∨ b = Book.book 14 ∨ b = Book.book 15

-- Define the lean proof problem statement
theorem total_books {s1 s2 s3 s4 s5 s6 : Student}: 
  (∀ (s1 s2 : Student), s1 ≠ s2 → ∃! b : Book, owns s1 b ∧ owns s2 b) →
  (∀ b : Book, ∃! s1 s2 : Student, s1 ≠ s2 ∧ owns s1 b ∧ owns s2 b) →
  ∃ (n : Nat), n = 15 :=
by
  sorry

end total_books_l69_69727


namespace number_of_real_solutions_l69_69501

theorem number_of_real_solutions : 
  ∃! x : ℝ, (x = 2 ∨ x = 2 ∧ x = -2) ∧ 
  (frac (3 * x) (x^2 + 2 * x + 4)) + (frac (4 * x) (x^2 - 4 * x + 4)) = -1 :=
sorry

end number_of_real_solutions_l69_69501


namespace calorie_limit_l69_69622

variable (breakfastCalories lunchCalories dinnerCalories extraCalories : ℕ)
variable (plannedCalories : ℕ)

-- Given conditions
axiom breakfast_calories : breakfastCalories = 400
axiom lunch_calories : lunchCalories = 900
axiom dinner_calories : dinnerCalories = 1100
axiom extra_calories : extraCalories = 600

-- To Prove
theorem calorie_limit (h : plannedCalories = (breakfastCalories + lunchCalories + dinnerCalories - extraCalories)) :
  plannedCalories = 1800 := by sorry

end calorie_limit_l69_69622


namespace equilateral_triangle_six_congruent_in_opposite_pairs_l69_69230

theorem equilateral_triangle_six_congruent_in_opposite_pairs :
  ∀ (A B C P : Type) 
    [is_tri A B C] 
    [is_equilateral A B C] 
    [is_centroid P A B C]
    [is_median_from A B C P]
    [is_median_from B A C P],
    are_congruent_in_opposite_pairs (triangles_from_medians A B C P) :=
by
  sorry

end equilateral_triangle_six_congruent_in_opposite_pairs_l69_69230


namespace statement_A_statement_C_l69_69129

namespace TriangleProofs

noncomputable def sin_square (x : ℝ) : ℝ := (sin x) ^ 2

variables (A B C a b c : ℝ)

-- Statement A: If A > B, then sin A > sin B
theorem statement_A (h1 : A > B) : sin A > sin B :=
by sorry

-- Statement C: If sin^2 A + sin^2 B < sin^2 C, then Δ ABC is an obtuse triangle
theorem statement_C (h2 : sin_square A + sin_square B < sin_square C) : c ^ 2 > a ^ 2 + b ^ 2 :=
by sorry

end TriangleProofs

end statement_A_statement_C_l69_69129


namespace problem_statement_l69_69089

noncomputable def a := Real.sqrt 3 + Real.sqrt 2
noncomputable def b := Real.sqrt 3 - Real.sqrt 2
noncomputable def expression := a^(2 * Real.log (Real.sqrt 5) / Real.log b)

theorem problem_statement : expression = 1 / 5 := by
  sorry

end problem_statement_l69_69089


namespace range_of_a_l69_69918

-- Definitions and theorems
theorem range_of_a (a : ℝ) : 
  (∀ (x y z : ℝ), x + y + z = 1 → abs (a - 2) ≤ x^2 + 2*y^2 + 3*z^2) → (16 / 11 ≤ a ∧ a ≤ 28 / 11) := 
by
  sorry

end range_of_a_l69_69918


namespace sum_of_series_eq_three_eighths_l69_69483

theorem sum_of_series_eq_three_eighths :
  (∑ n in (range 6).map (λ i, 1 / (i + 2) / (i + 3))) = (3 / 8) :=
by 
  sorry

end sum_of_series_eq_three_eighths_l69_69483


namespace sqrt_x_plus_inv_sqrt_x_l69_69639

variable (x : ℝ) (hx : 0 < x) (h : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (hx : 0 < x) (h : x + 1 / x = 50) : 
  sqrt x + 1 / sqrt x = 2 * sqrt 13 := 
sorry

end sqrt_x_plus_inv_sqrt_x_l69_69639


namespace sara_total_cents_l69_69312

/-- The total value in cents of a collection of coins given 
the number of quarters, dimes, nickels, and pennies they contain. 
This example specifically considers a case where there are 
11 quarters, 8 dimes, 15 nickels, and 23 pennies. -/
theorem sara_total_cents (quarters dimes nickels pennies : ℕ) :
  quarters = 11 → dimes = 8 → nickels = 15 → pennies = 23 →
  (25 * quarters + 10 * dimes + 5 * nickels + pennies = 453) := 
by
  intros hq hd hn hp
  rw [hq, hd, hn, hp]
  sorry  -- Proof goes here

end sara_total_cents_l69_69312


namespace no_such_X_Y_exists_l69_69244

theorem no_such_X_Y_exists : 
  ¬ ∃ (X Y : ℕ), 
    (∃ f : Fin (nat.digits 10 X).length ≃ₗ (Fin (nat.digits 10 Y).length), 
      ∀ i, f i = (nat.digits 10 Y)[i]) 
    ∧ X + Y = 10^1111 - 1 :=
by 
  sorry

end no_such_X_Y_exists_l69_69244


namespace all_propositions_imply_l69_69479

variables (p q r : Prop)

theorem all_propositions_imply (hpqr : p ∧ q ∧ r)
                               (hnpqr : ¬p ∧ q ∧ ¬r)
                               (hpnqr : p ∧ ¬q ∧ r)
                               (hnpnqr : ¬p ∧ ¬q ∧ ¬r) :
  (p → q) ∨ r :=
by { sorry }

end all_propositions_imply_l69_69479


namespace intersection_M_N_l69_69927

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l69_69927


namespace tangency_sphere_placement_l69_69600

noncomputable def number_of_spheres (num_planes : Nat) (has_sphere : Bool) : Nat :=
  if num_planes = 3 ∧ has_sphere then 16 else 0

theorem tangency_sphere_placement :
  ∀ (num_planes : Nat) (has_sphere : Bool),
    (num_planes = 3 ∧ has_sphere) → (0 ≤ number_of_spheres num_planes has_sphere) ∧ 
    (number_of_spheres num_planes has_sphere ≤ 16) :=
by
  intros num_planes has_sphere h
  rw [number_of_spheres]
  split_ifs
  · exact ⟨by decide, by decide⟩
  · exact ⟨zero_le 0, by decide⟩

end tangency_sphere_placement_l69_69600


namespace factorial_expression_l69_69736

open Nat

theorem factorial_expression : (12.factorial - 11.factorial - 10.factorial) / 10.factorial = 120 := 
by 
sorry

end factorial_expression_l69_69736


namespace point_Q_circular_motion_l69_69204

variable (ω : ℝ) 

-- Define the point P on the unit circle with angular velocity ω
def P (t : ℝ) : ℝ × ℝ := (Real.cos (ω * t), Real.sin (ω * t))

-- Define the point Q based on the coordinates of point P
def Q (t : ℝ) : ℝ × ℝ := let (x, y) := P ω t in (-2 * x * y, y * y - x * x)

theorem point_Q_circular_motion (t : ℝ) :
  ∃ t' : ℝ, Q ω t = (Real.cos (2 * ω * t'), Real.sin (2 * ω * t')) :=
by 
  sorry

end point_Q_circular_motion_l69_69204


namespace minimum_period_f_l69_69704

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x / 2 + Real.pi / 4)

theorem minimum_period_f :
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T) :=
sorry

end minimum_period_f_l69_69704


namespace polynomial_relatively_prime_condition_l69_69112

open Polynomial

noncomputable def relatively_prime (a b : ℤ) : Prop := Int.gcd a b = 1

theorem polynomial_relatively_prime_condition (P : Polynomial ℤ) :
  (∀ a b : ℤ, relatively_prime a b → relatively_prime (P.eval a) (P.eval b)) ↔
  (∃ n : ℕ, P = Polynomial.C (↑(1 : ℤ)) * Polynomial.X ^ n ∨ P = Polynomial.C (↑(-1 : ℤ)) * Polynomial.X ^ n) :=
sorry

end polynomial_relatively_prime_condition_l69_69112


namespace cos_C_triangle_identity_l69_69216

theorem cos_C_triangle_identity 
  {A B C : ℝ} 
  (sin_A : ℝ) (sin_B : ℝ) (sin_C : ℝ) 
  (side_a : ℝ) (side_b : ℝ) (side_c : ℝ)  
  (h1 : 6 * sin_A = 4 * sin_B)
  (h2 : 4 * sin_B = 3 * sin_C)
  (h3 : 6 * sin_A = 3 * sin_C)
  (h4 : side_a = sin_A)
  (h5 : side_b = sin_B)
  (h6 : side_c = sin_C)
  : cos C = -1 / 4 :=
begin
  sorry
end

end cos_C_triangle_identity_l69_69216


namespace marcia_oranges_l69_69285

noncomputable def averageCost
  (appleCost bananaCost orangeCost : ℝ) 
  (numApples numBananas numOranges : ℝ) : ℝ :=
  (numApples * appleCost + numBananas * bananaCost + numOranges * orangeCost) /
  (numApples + numBananas + numOranges)

theorem marcia_oranges : 
  ∀ (appleCost bananaCost orangeCost avgCost : ℝ) 
  (numApples numBananas numOranges : ℝ),
  appleCost = 2 → 
  bananaCost = 1 → 
  orangeCost = 3 → 
  numApples = 12 → 
  numBananas = 4 → 
  avgCost = 2 → 
  averageCost appleCost bananaCost orangeCost numApples numBananas numOranges = avgCost → 
  numOranges = 4 :=
by 
  intros appleCost bananaCost orangeCost avgCost numApples numBananas numOranges
         h1 h2 h3 h4 h5 h6 h7
  sorry

end marcia_oranges_l69_69285


namespace magnitude_b_l69_69929

-- Definitions for vectors and collinearity
def vec2 := ℝ × ℝ

def collinear (u v : vec2) : Prop := 
  ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Condition for collinearity and magnitude calculation
theorem magnitude_b (x : ℝ) (b : vec2) :
  let a : vec2 := (2, 1)
  let b : vec2 := (1, x)
  collinear (2 * a.1 - b.1, 2 * a.2 - b.2) b →
  b = (1, 1 / 2) →
  ∥b∥ = √5 / 2 := 
by
  sorry

end magnitude_b_l69_69929


namespace train_lengths_l69_69331

theorem train_lengths (L_A L_P L_B : ℕ) (speed_A_km_hr speed_B_km_hr : ℕ) (time_A_seconds : ℕ)
                      (h1 : L_P = L_A)
                      (h2 : speed_A_km_hr = 72)
                      (h3 : speed_B_km_hr = 80)
                      (h4 : time_A_seconds = 60)
                      (h5 : L_B = L_P / 2)
                      (h6 : L_A + L_P = (speed_A_km_hr * 1000 / 3600) * time_A_seconds) :
  L_A = 600 ∧ L_B = 300 :=
by
  sorry

end train_lengths_l69_69331


namespace minimum_xy_l69_69137

theorem minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : 
  x * y ≥ 18 :=
sorry

end minimum_xy_l69_69137


namespace triangles_in_pentadecagon_l69_69965

def regular_pentadecagon := {vertices : Finset Point | vertices.card = 15 ∧ 
  ∀ a b c ∈ vertices, ¬Collinear a b c}

theorem triangles_in_pentadecagon (P : regular_pentadecagon) : 
  (P.vertices.card.choose 3) = 455 :=
by 
  sorry


end triangles_in_pentadecagon_l69_69965


namespace road_trip_days_l69_69623

theorem road_trip_days 
    (jade_hours_per_day : ℕ) 
    (krista_hours_per_day : ℕ) 
    (total_hours : ℕ) 
    (total_days : ℕ) :
    jade_hours_per_day = 8 
    → krista_hours_per_day = 6 
    → total_hours = 42 
    → total_hours = total_days * (jade_hours_per_day + krista_hours_per_day) 
    → total_days = 3 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  norm_num at h4
  exact h4

end road_trip_days_l69_69623


namespace probability_of_roots_l69_69049

noncomputable def probability_roots_satisfy_condition : ℚ :=
let k := (6 : ℝ) .. (11 : ℝ) in
let equation := λ k : ℝ, (k^2 - 2*k - 24) * (Polynomial.X^2) + (3*k - 8) * (Polynomial.X) + 2 in
-- Define the root condition
let roots_satisfy_condition := λ k x₁ x₂, x₁ + x₂ = (8 - 3 * k) / (k^2 - 2 * k - 24) ∧ x₁ * x₂ = 2 / (k^2 - 2 * k - 24) ∧ x₁ ≤ 2 * x₂ in
-- Find valid k range
let valid_k := (6 : ℚ) .. (28 / 3 : ℚ) in
-- Calculate ratio
(valid_k.count / k.count : ℚ)

theorem probability_of_roots :
  probability_roots_satisfy_condition = 2 / 3 := 
sorry

end probability_of_roots_l69_69049


namespace f_2013_is_cosine_l69_69267

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def f_seq : ℕ → (ℝ → ℝ)
| 0       := f
| (n + 1) := fun x => (f_seq n)' x

theorem f_2013_is_cosine : f_seq 2013 = Real.cos := sorry

end f_2013_is_cosine_l69_69267


namespace loan_repayment_amount_l69_69058

variables (a p : ℝ) (m : ℕ)

theorem loan_repayment_amount :
  let x := a * p * (1 + p) ^ m / ((1 + p) ^ m - 1)
  in ∀ (a p : ℝ) (m : ℕ), ∃ x : ℝ, x = a * p * (1 + p) ^ m / ((1 + p) ^ m - 1) :=
by
  intro a p m
  use a * p * (1 + p) ^ m / ((1 + p) ^ m - 1)
  rfl

end loan_repayment_amount_l69_69058


namespace solution_l69_69183

noncomputable def f (a b c : ℝ) (x : ℝ) := (a * x^2 + b * x + c) / Real.exp x

def f_deriv_zeros (a b c : ℝ) : Prop :=
  let g (x : ℝ) := -a * x^2 + (2 * a - b) * x + b - c
  g (-3) = 0 ∧ g 0 = 0

def f_monotonic_intervals (a b c : ℝ) : set (set ℝ) :=
  { (Ici (-∞)) ∩ (Iio (-3)), Ioo (-3) 0, Ici 0 ∩ (Ici (+∞)) }

def f_min_max_value (a b c : ℝ) (min_value max_value : ℝ) : Prop :=
  let f_val (x : ℝ) := (f a b c x)
  f_val (-3) = min_value ∧ max_value = max (f_val 0) (f_val (-5))

theorem solution (a b c : ℝ) (f_min f_max : ℝ) (h : a > 0 ∧ f_deriv_zeros a b c ∧ f_min = -Real.exp 3) :
  f_monotonic_intervals a b c = {(Set.Ioo (-∞) (-3)), (Set.Ioo (-3) 0), (Set.Ici 0)} ∧
  f_min_max_value a b c f_min f_max → f_max = 5 * Real.exp 5 :=
begin
  sorry
end

end solution_l69_69183


namespace num_four_digit_palindromic_squares_l69_69022

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69022


namespace parabola_eq_given_hyperbola_conditions_l69_69914

theorem parabola_eq_given_hyperbola_conditions
  (a b p : ℝ) (ha : 0 < a) (hb : 0 < b) (hp : 0 < p)
  (focal_distance_condition : 2 * a = real.sqrt (a^2 + b^2))
  (focus_distance_condition : (|p/2| / real.sqrt (1/a^2 + 1/b^2)) = 2) :
  x^2 = 16 * y := by
  sorry

end parabola_eq_given_hyperbola_conditions_l69_69914


namespace _l69_69111

noncomputable def triangleSidesAndArea (x : ℝ) : Prop :=
  let a := x - 1
  let b := x
  let c := x + 1
  let S := x + 2
  let s := (a + b + c) / 2
  S = real.sqrt (s * (s - a) * (s - b) * (s - c))

example : triangleSidesAndArea 4 := by
  -- This will assert the theorem with given value of x
  sorry

end _l69_69111


namespace number_of_triangles_in_pentadecagon_l69_69983

open Finset

theorem number_of_triangles_in_pentadecagon :
  ∀ (n : ℕ), n = 15 → (n.choose 3 = 455) := 
by 
  intros n hn 
  rw hn
  rw Nat.choose_eq_factorial_div_factorial (show 3 ≤ 15)
  { norm_num }

-- Proof omitted with sorry

end number_of_triangles_in_pentadecagon_l69_69983


namespace num_four_digit_palindromic_squares_l69_69016

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69016


namespace two_digit_number_difference_perfect_square_l69_69325

theorem two_digit_number_difference_perfect_square (N : ℕ) (a b : ℕ)
  (h1 : N = 10 * a + b)
  (h2 : N % 100 = N)
  (h3 : 1 ≤ a ∧ a ≤ 9)
  (h4 : 0 ≤ b ∧ b ≤ 9)
  (h5 : (N - (10 * b + a : ℕ)) = 64) : 
  N = 90 := 
sorry

end two_digit_number_difference_perfect_square_l69_69325


namespace vlad_taller_than_sister_l69_69734

def height_vlad_meters : ℝ := 1.905
def height_sister_cm : ℝ := 86.36

theorem vlad_taller_than_sister :
  (height_vlad_meters * 100 - height_sister_cm = 104.14) :=
by 
  sorry

end vlad_taller_than_sister_l69_69734


namespace estimate_red_balls_l69_69599

-- Definitions based on conditions
def total_balls : ℕ := 20
def total_draws : ℕ := 100
def red_draws : ℕ := 30

-- The theorem statement
theorem estimate_red_balls (h1 : total_balls = 20) (h2 : total_draws = 100) (h3 : red_draws = 30) :
  (total_balls * (red_draws / total_draws) : ℤ) = 6 := 
by
  sorry

end estimate_red_balls_l69_69599


namespace value_of_sqrt_x_plus_one_over_sqrt_x_l69_69651

noncomputable def find_value (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) : ℝ :=
  sqrt(x) + 1/sqrt(x)

theorem value_of_sqrt_x_plus_one_over_sqrt_x (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) :
  find_value x hx_pos hx = 2 * sqrt(13) :=
sorry

end value_of_sqrt_x_plus_one_over_sqrt_x_l69_69651


namespace ellipse_solution_l69_69440

-- Define the points
def foci1 := (1, 1)
def foci2 := (1, 3)
def point_on_ellipse := (6, 2)

-- Define the function to calculate the distance between two points.
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define the major axis as twice the distance from point_on_ellipse to any focus
noncomputable def major_axis := 2 * distance point_on_ellipse foci1

-- Define the minor axis based on the difference between the squares of the major axis length and the distance between the foci.
noncomputable def minor_axis := real.sqrt ((major_axis) ^ 2 - (distance foci1 foci2) ^ 2)

-- Define the center of the ellipse as the midpoint of the foci
def center := ((foci1.1 + foci2.1) / 2, (foci1.2 + foci2.2) / 2)

-- Identify the constants h, k, and a from the derived ellipse equation
def h := center.1
def k := center.2
def a := minor_axis / 2

-- Create the proof problem
theorem ellipse_solution : a + k = 7 :=
by
  sorry

end ellipse_solution_l69_69440


namespace locus_midpoints_chords_through_Q_l69_69656

theorem locus_midpoints_chords_through_Q 
  (O Q : Point) (C : Circle) (r1 r2 : ℝ) 
  (hC : C.radius = 10) (hQ_inside_C : distance O Q = 8) 
  (hQ_in_C : distance O Q < C.radius) : 
  ∃ (center : Point) (r : ℝ), 
    (∀ (M : Point), (∃ (chord : Line), midpoint (chord ab) = M ∧ passes_through Q chord) → 
    ∀ (p : Point), distance center p = r) ∧ 
  r = 4 ∧ 
  center = midpoint O Q :=
sorry

end locus_midpoints_chords_through_Q_l69_69656


namespace parallelogram_area_l69_69277

theorem parallelogram_area :
  let v := ![7, -4]
  let w := ![12, -1]
  ∃ (A : ℝ), abs(det (matrix.of (λ i j, if j = 0 then v i else w i))) = A :=
begin
  let v := ![7, -4]
  let w := ![12, -1],
  use 41,
  sorry
end

end parallelogram_area_l69_69277


namespace least_possible_multiple_l69_69586

theorem least_possible_multiple (x y z k : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 1 ≤ k)
  (h1 : 3 * x = k * z) (h2 : 4 * y = k * z) (h3 : x - y + z = 19) : 3 * x = 12 :=
by
  sorry

end least_possible_multiple_l69_69586


namespace count_even_thousands_digit_palindromes_l69_69932

-- Define the set of valid digits
def valid_A : Finset ℕ := {2, 4, 6, 8}
def valid_B : Finset ℕ := Finset.range 10

-- Define the condition of a four-digit palindrome ABBA where A is even and non-zero
def is_valid_palindrome (a b : ℕ) : Prop :=
  a ∈ valid_A ∧ b ∈ valid_B

-- The proof problem: Prove that the total number of valid palindromes ABBA is 40
theorem count_even_thousands_digit_palindromes :
  (valid_A.card) * (valid_B.card) = 40 :=
by
  -- Skipping the proof itself
  sorry

end count_even_thousands_digit_palindromes_l69_69932


namespace grid_division_l69_69105

theorem grid_division :
  ∃ (G : list (list char)), 
  (∀ (c : char), c ∈ ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']) ∧
  (∀ (shapes : list (list char)),
    List.length shapes = 4 ∧ -- Four shapes
    (∀ (s : list char), List.length s = 5 ∧ -- Each shape has 5 cells 
    (∀ (c : char), c ∈ s → c ∈ ['C'])) → -- Each shape in the form of a "C"
    (∀ (identicals : list (list char)),
      List.length identicals = 5 ∧ -- Five shapes
      (∀ (i : list char), List.length i = 4))) -- Each shape has 4 cells
  -- Proof part skipped
  sorry

end grid_division_l69_69105


namespace polynomial_product_conditions_and_distance_l69_69157

-- Main statement organizing the problem into a theorem.
theorem polynomial_product_conditions_and_distance :
  ∀ (x m n P : ℝ), 
  let A := 2 * x^2 - m * x + 1
  let B := n * x^2 - 3
  (∀ y : ℝ, (A * B) y ≠ x^3 ∧ (A * B) y ≠ x^2) →
  (n = 6 ∧ m = 0) ∧
  (m, n, P ∈ ℝ ∧ n ≠ 0 ∧ (abs (P - m) = 2 * abs (P - n)) → (P = 4 ∨ P = 12))
:= by
  sorry -- Proof steps will go here

end polynomial_product_conditions_and_distance_l69_69157


namespace clubsuit_problem_l69_69074

def clubsuit (x y : ℤ) : ℤ :=
  (x^2 + y^2) * (x - y)

theorem clubsuit_problem : clubsuit 2 (clubsuit 3 4) = 16983 := 
by 
  sorry

end clubsuit_problem_l69_69074


namespace quadrilateral_area_l69_69547

-- Define the unit vectors e1 and e2 along x-axis and y-axis
def e1 := (1 : ℝ, 0 : ℝ)
def e2 := (0 : ℝ, 1 : ℝ)

-- Define the vectors AC and BD
def AC := (3 * e1.1 - e2.1, 3 * e1.2 - e2.2)
def BD := (2 * e1.1 + 6 * e2.1, 2 * e1.2 + 6 * e2.2)

-- Define the dot product operation
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem quadrilateral_area :
  dot_product AC BD = 0 ∧
  magnitude AC = Real.sqrt 10 ∧
  magnitude BD = 2 * Real.sqrt 10 →
  1 / 2 * magnitude AC * magnitude BD = 10 :=
sorry

end quadrilateral_area_l69_69547


namespace triangles_in_pentadecagon_l69_69969

def regular_pentadecagon := {vertices : Finset Point | vertices.card = 15 ∧ 
  ∀ a b c ∈ vertices, ¬Collinear a b c}

theorem triangles_in_pentadecagon (P : regular_pentadecagon) : 
  (P.vertices.card.choose 3) = 455 :=
by 
  sorry


end triangles_in_pentadecagon_l69_69969


namespace calculate_expression_l69_69272

theorem calculate_expression (p q r s : ℝ)
  (h1 : p + q + r + s = 10)
  (h2 : p^2 + q^2 + r^2 + s^2 = 26) :
  6 * (p^4 + q^4 + r^4 + s^4) - (p^3 + q^3 + r^3 + s^3) =
    6 * ((p-1)^4 + (q-1)^4 + (r-1)^4 + (s-1)^4) - ((p-1)^3 + (q-1)^3 + (r-1)^3 + (s-1)^3) :=
by {
  sorry
}

end calculate_expression_l69_69272


namespace find_a_l69_69541

-- Define the circle equation in its standard form, and define the radius and center.
def circle_center : ℝ × ℝ := (2, -3)
def circle_radius : ℝ := 5

-- The line equation
def line (a : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * x - y + a = 0

-- Length of the chord given by the problem
def chord_length : ℝ := 4 * Real.sqrt 5

-- Condition on a
axiom a_gt_neg5 (a : ℝ) : a > -5

-- Completed proof statement
theorem find_a (a : ℝ) :
  (∃ M N : ℝ × ℝ,
    (line a M.1 M.2) ∧ (line a N.1 N.2) ∧
    (M ≠ N) ∧
    (M.1^2 + M.2^2 - 4 * M.1 + 6 * M.2 - 12 = 0) ∧
    (N.1^2 + N.2^2 - 4 * N.1 + 6 * N.2 - 12 = 0) ∧ 
    (Real.dist M N = chord_length)) → 
  a = -2 :=
by sorry

end find_a_l69_69541


namespace rational_roots_of_quadratic_l69_69508

theorem rational_roots_of_quadratic (k : ℤ) (h : k > 0) :
  (∃ x : ℚ, k * x^2 + 12 * x + k = 0) ↔ (k = 3 ∨ k = 6) :=
by
  sorry

end rational_roots_of_quadratic_l69_69508


namespace std_dev_unchanged_l69_69902

def dataM : List ℝ := [80, 82, 82, 84, 84, 84, 86, 86, 86, 86]
def dataN := dataM.map (λ x => x + 4)

theorem std_dev_unchanged (M N : List ℝ) (h : N = M.map (λ x => x + 4)) : 
  std_dev N = std_dev M :=
sorry

#eval std_dev_unchanged dataM dataN (by simp [dataN])

end std_dev_unchanged_l69_69902


namespace four_digit_palindromic_perfect_square_count_l69_69041

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69041


namespace find_shortest_height_l69_69281

variable (T S P Q : ℝ)

theorem find_shortest_height (h1 : T = 77.75) (h2 : T = S + 9.5) (h3 : P = S + 5) (h4 : Q = P - 3) : S = 68.25 :=
  sorry

end find_shortest_height_l69_69281


namespace set_cardinality_bound_l69_69257

theorem set_cardinality_bound (M : Type*) (A B : ℕ → set M) (n : ℕ)
  (hA : ∀ m, A m ⊆ M)
  (hB : ∀ m, B m ⊆ M)
  (h_disjointA : ∀ i j, i ≠ j → disjoint (A i) (A j))
  (h_disjointB : ∀ i j, i ≠ j → disjoint (B i) (B j))
  (h_coveredA : (⋃ i, A i) = M)
  (h_coveredB : (⋃ i, B i) = M)
  (h_union_bound : ∀ j k, j < n → k < n → n ≤ (A j ∪ B k).to_finset.card) :
  (M.to_finset.card) ≥ n^2 / 2 := sorry

end set_cardinality_bound_l69_69257


namespace monic_quadratic_with_real_coeffs_l69_69839

open Complex Polynomial

theorem monic_quadratic_with_real_coeffs {x : ℂ} :
  (∀ a b c : ℝ, Polynomial.monic (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) ∧ 
  (x = 2 - 3 * Complex.I ∨ x = 2 + 3 * Complex.I) → Polynomial.eval (2 - 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0) ∧
  Polynomial.eval (2 + 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0 :=
begin
  sorry
end

end monic_quadratic_with_real_coeffs_l69_69839


namespace sequences_count_l69_69198

-- Define the condition that no two adjacent digits have the same parity.
def no_adjacent_same_parity (xs : List ℕ) : Prop :=
  ∀ i, i < xs.length - 1 → (xs[i] % 2 ≠ xs[i+1] % 2)

-- Define the set of digits {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.
def digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the condition that the first digit is odd.
def starts_with_odd (xs : List ℕ) : Prop :=
  xs.head % 2 = 1

-- Define the total number of sequences that satisfy the conditions.
-- This will be our proof goal.
def count_valid_sequences : ℕ :=
  78125

-- Now state the theorem.
theorem sequences_count :
  ∃ xs : List ℕ, xs.length = 7 ∧
  no_adjacent_same_parity xs ∧
  starts_with_odd xs ∧
  (∀ x ∈ xs, x ∈ digits) ∧
  count_valid_sequences = 5 * 5^6 :=
sorry

end sequences_count_l69_69198


namespace smallest_n_for_log_sum_l69_69809

theorem smallest_n_for_log_sum :
  (∃ n : ℕ, 
  (∑ k in Finset.range (n + 1), Real.log10 (1 + 1 / 2 ^ (2 ^ k))) ≥ (1 + Real.log10 (1/2))) 
  ∧ ∀ m : ℕ, m < n → ¬ ((∑ k in Finset.range (m + 1), Real.log10 (1 + 1 / 2 ^ (2 ^ k))) ≥ (1 + Real.log10 (1/2))) :=
begin
  sorry,
end

end smallest_n_for_log_sum_l69_69809


namespace complete_set_contains_all_rationals_l69_69059

theorem complete_set_contains_all_rationals (T : Set ℚ) (hT : ∀ (p q : ℚ), p / q ∈ T → p / (p + q) ∈ T ∧ q / (p + q) ∈ T) (r : ℚ) : 
  (r = 1 ∨ r = 1 / 2) → (∀ x : ℚ, 0 < x ∧ x < 1 → x ∈ T) :=
by
  sorry

end complete_set_contains_all_rationals_l69_69059


namespace permutation_exists_l69_69630

theorem permutation_exists (n : ℕ) (h : Even n) (hn : n > 0) :
  ∃ (x : Fin n → Fin n), 
    (∀ i : Fin n, x (Fin.succ i) = 2 * x i ∨ x (Fin.succ i) = 2 * x i - 1 ∨ 
      x (Fin.succ i) = 2 * x i - n ∨ x (Fin.succ i) = 2 * x i - n - 1) ∧ 
    (x ⟨n, hn⟩ = x 0) :=
by
  sorry

end permutation_exists_l69_69630


namespace sum_of_squares_l69_69813

variable (x y z : ℝ)

def N : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 3 * y, 2 * z],
  ![2 * x, y, -z],
  ![2 * x, -y, z]
]

theorem sum_of_squares : (N x y z)ᵀ ⬝ (N x y z) = 1 →
  x^2 + y^2 + z^2 = 47 / 120 :=
by
  intro h
  have h1 : (N x y z)ᵀ ⬝ (N x y z) = Matrix.eye 3 := h
  sorry

end sum_of_squares_l69_69813


namespace James_total_area_l69_69249

theorem James_total_area :
  let initial_length := 13
  let initial_width := 18
  let increased_length := initial_length + 2
  let increased_width := initial_width + 2
  let single_room_area := increased_length * increased_width
  let four_rooms_area := 4 * single_room_area
  let larger_room_area := 2 * single_room_area
  let total_area := four_rooms_area + larger_room_area
  total_area = 1800 :=
by
  let initial_length := 13
  let initial_width := 18
  let increased_length := initial_length + 2
  let increased_width := initial_width + 2
  let single_room_area := increased_length * increased_width
  let four_rooms_area := 4 * single_room_area
  let larger_room_area := 2 * single_room_area
  let total_area := four_rooms_area + larger_room_area
  have h : total_area = 1800 := by sorry
  exact h

end James_total_area_l69_69249


namespace largest_common_term_l69_69791

def first_seq (n : ℕ) := 1 + 8 * n
def second_seq (m : ℕ) := 5 + 9 * m

def common_in_range (a b : ℕ) := a = b ∧ 1 ≤ a ∧ a < 150

theorem largest_common_term (x : ℕ) :
  (∃ n m, common_in_range (first_seq n) (second_seq m)) →
  (∀ y, (∃ n m, common_in_range (first_seq n) (second_seq m) ∧ y = first_seq n ∧ y x) → y ≤ x) →
  x = 81 :=
by
  sorry

end largest_common_term_l69_69791


namespace simplify_S_generalize_S_l69_69676

noncomputable def S := 4 - 3 * (5:ℝ)^(1/4) + 2 * (5:ℝ)^(1/2) - (5:ℝ)^(3/4)

theorem simplify_S :
  (2 / (Real.sqrt S)) = 1 + Real.root (5:ℝ) 4 :=
by
  sorry

theorem generalize_S (n : ℕ) (hn : n > 0):
  let t := (4 - 3 * Real.root ((2 * n + 1):ℝ) (2 * n) + 2 * Real.root ((2 * n + 1) ^ 2:ℝ) (2 * n) - Real.root ((2 * n + 1) ^ 3:ℝ) (2 * n))
  (2 / (Real.sqrt t)) = 1 + Real.root ((2 * n + 1):ℝ) (2 * n) :=
by
  sorry

end simplify_S_generalize_S_l69_69676


namespace transformed_triangle_area_l69_69319

-- Define the function g and its properties
variable {R : Type*} [LinearOrderedField R]
variable (g : R → R)
variable (a b c : R)
variable (area_original : R)

-- Given conditions
-- The function g is defined such that the area of the triangle formed by 
-- points (a, g(a)), (b, g(b)), and (c, g(c)) is 24
axiom h₀ : {x | x = a ∨ x = b ∨ x = c} ⊆ Set.univ
axiom h₁ : area_original = 24

-- Define a function that computes the area of a triangle given three points
noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : R) : R := 
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem transformed_triangle_area (h₀ : {x | x = a ∨ x = b ∨ x = c} ⊆ Set.univ)
  (h₁ : area_triangle a (g a) b (g b) c (g c) = 24) :
  area_triangle (a / 3) (3 * g a) (b / 3) (3 * g b) (c / 3) (3 * g c) = 24 :=
sorry

end transformed_triangle_area_l69_69319


namespace point_in_second_quadrant_l69_69337

-- Given definitions and conditions
def complex_subtract (z1 z2 : ℂ) : ℂ := z1 - z2

-- Problem statement
theorem point_in_second_quadrant : complex_subtract (1 + 2 * complex.I) (3 - 4 * complex.I) = -2 + 6 * complex.I ∧ 
    is_in_second_quadrant (-2 : ℝ) (6 : ℝ) :=
by
    -- Define helper function to check the second quadrant
    def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
    sorry

end point_in_second_quadrant_l69_69337


namespace house_prices_and_yields_l69_69683

theorem house_prices_and_yields :
  ∃ x y : ℝ, 
    (425 = (y / 100) * x) ∧ 
    (459 = ((y - 0.5) / 100) * (6/5) * x) ∧ 
    (x = 8500) ∧ 
    (y = 5) ∧ 
    ((6/5) * x = 10200) ∧ 
    (y - 0.5 = 4.5) :=
by
  sorry

end house_prices_and_yields_l69_69683


namespace absolute_value_difference_of_roots_l69_69116

theorem absolute_value_difference_of_roots :
  let r1 r2 : ℝ in
  (∀ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = 12 → |r1 - r2| = 1) :=
begin
  sorry
end

end absolute_value_difference_of_roots_l69_69116


namespace counting_error_l69_69315

theorem counting_error
  (b g : ℕ)
  (initial_balloons := 5 * b + 4 * g)
  (popped_balloons := g + 2 * b)
  (remaining_balloons := initial_balloons - popped_balloons)
  (Dima_count := 100) :
  remaining_balloons ≠ Dima_count := by
  sorry

end counting_error_l69_69315


namespace start_time_6am_l69_69763

def travel_same_time (t : ℝ) (x : ℝ) (y : ℝ) (constant_speed : Prop) : Prop :=
  (x = t + 4) ∧ (y = t + 9) ∧ constant_speed 

theorem start_time_6am
  (x y t: ℝ)
  (constant_speed : Prop) 
  (meet_noon : travel_same_time t x y constant_speed)
  (eqn : 1/t + 1/(t + 4) + 1/(t + 9) = 1) :
  t = 6 :=
by
  sorry

end start_time_6am_l69_69763


namespace exists_convex_body_diff_from_sphere_with_circular_projections_l69_69248

-- Definitions of the coordinate planes
def plane_α : set (ℝ × ℝ × ℝ) := {p | p.1 = 0}
def plane_β : set (ℝ × ℝ × ℝ) := {p | p.2 = 0}
def plane_γ : set (ℝ × ℝ × ℝ) := {p | p.3 = 0}

-- Definitions of the unit sphere and the cylinders
def sphere_B : set (ℝ × ℝ × ℝ) := {p | p.1^2 + p.2^2 + p.3^2 ≤ 1}
def cylinder_C1 : set (ℝ × ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}
def cylinder_C2 : set (ℝ × ℝ × ℝ) := {p | p.2^2 + p.3^2 ≤ 1}
def cylinder_C3 : set (ℝ × ℝ × ℝ) := {p | p.3^2 + p.1^2 ≤ 1}

-- Definition of the intersection of the cylinders
def body_C : set (ℝ × ℝ × ℝ) := cylinder_C1 ∩ cylinder_C2 ∩ cylinder_C3

-- The main proof statement
theorem exists_convex_body_diff_from_sphere_with_circular_projections :
  ∃ C : set (ℝ × ℝ × ℝ), 
  C ≠ sphere_B ∧ 
  (∀ p ∈ C, is_convex C) ∧ 
  (∀ p ∈ plane_α, orthogonal_projection C plane_α p = {q | q.1^2 + q.2^2 ≤ 1}) ∧ 
  (∀ p ∈ plane_β, orthogonal_projection C plane_β p = {q | q.2^2 + q.3^2 ≤ 1}) ∧ 
  (∀ p ∈ plane_γ, orthogonal_projection C plane_γ p = {q | q.3^2 + q.1^2 ≤ 1}) :=
sorry

end exists_convex_body_diff_from_sphere_with_circular_projections_l69_69248


namespace sqrt_x_plus_inv_sqrt_x_l69_69642

variable (x : ℝ) (hx : 0 < x) (h : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (hx : 0 < x) (h : x + 1 / x = 50) : 
  sqrt x + 1 / sqrt x = 2 * sqrt 13 := 
sorry

end sqrt_x_plus_inv_sqrt_x_l69_69642


namespace student_percentage_in_math_l69_69416

theorem student_percentage_in_math (M H T : ℝ) (H_his : H = 84) (H_third : T = 69) (H_avg : (M + H + T) / 3 = 75) : M = 72 :=
by
  sorry

end student_percentage_in_math_l69_69416


namespace number_of_triangles_in_pentadecagon_l69_69986

open Finset

theorem number_of_triangles_in_pentadecagon :
  ∀ (n : ℕ), n = 15 → (n.choose 3 = 455) := 
by 
  intros n hn 
  rw hn
  rw Nat.choose_eq_factorial_div_factorial (show 3 ≤ 15)
  { norm_num }

-- Proof omitted with sorry

end number_of_triangles_in_pentadecagon_l69_69986


namespace triangles_in_pentadecagon_l69_69966

def regular_pentadecagon := {vertices : Finset Point | vertices.card = 15 ∧ 
  ∀ a b c ∈ vertices, ¬Collinear a b c}

theorem triangles_in_pentadecagon (P : regular_pentadecagon) : 
  (P.vertices.card.choose 3) = 455 :=
by 
  sorry


end triangles_in_pentadecagon_l69_69966


namespace triangle_find_x_l69_69332

theorem triangle_find_x :
  ∀ {k : ℝ},
    3 * k + 4 * k + 9 * k = 180 →
    let a := 3 * k in
    let b := 4 * k in
    let c := 9 * k in
    2 * a = a →
    180 - (c - 33.75) = 2 * a + b :=
  by
    intros k hsum ha hb hc
    let a := 3 * k
    let b := 4 * k
    let c := 9 * k
    rfl
    sorry

end triangle_find_x_l69_69332


namespace smallest_n_divisibility_l69_69357

theorem smallest_n_divisibility (n : ℕ) (h : n = 5) : 
  ∃ k, (1 ≤ k ∧ k ≤ n + 1) ∧ (n^2 - n) % k = 0 ∧ ∃ i, (1 ≤ i ∧ i ≤ n + 1) ∧ (n^2 - n) % i ≠ 0 :=
by
  sorry

end smallest_n_divisibility_l69_69357


namespace maximum_distinct_characters_l69_69348

noncomputable def card_problem : Prop :=
  ∃ (X : Finset (Finset Char)),
    X.card = 15 ∧
    ∀ A ∈ X, A.card = 3 ∧
    (∀ A B ∈ X, A ≠ B → A ∩ B ≠ A) ∧
    (∀ S ⊆ X, S.card = 6 → ∃ A B ∈ S, A ≠ B ∧ (A ∩ B).nonempty) ∧
    (∃ C: Finset Char, C.card = 35 ∧ ∀ A ∈ X, A ⊆ C)

theorem maximum_distinct_characters : card_problem := sorry

end maximum_distinct_characters_l69_69348


namespace no_isomorphic_components_after_edge_removal_l69_69591

variables {V : Type*} [Fintype V] [DecidableEq V]

/-- A connected graph with specific vertex degrees. -/
structure special_graph (G : SimpleGraph V) : Prop :=
(connected : G.Connected)
(degree_three_vertices : Fintype.card {v : V // G.degree v = 3} = 4)
(degree_four_vertices  : ∀ v : V, G.degree v = 4 ∨ G.degree v = 3)

theorem no_isomorphic_components_after_edge_removal (G : SimpleGraph V) 
  (hG : special_graph G) :
  ∀ (e : G.EdgeSet),
    ¬ (G.deleteEdge e).Cuts e.toApex.connected
    ∧ (G.deleteEdge e).Cuts e.toApex.symmetric :=
sorry

end no_isomorphic_components_after_edge_removal_l69_69591


namespace sum_first_six_terms_l69_69611

variable (a1 q : ℤ)
variable (n : ℕ)

noncomputable def geometric_sum (a1 q : ℤ) (n : ℕ) : ℤ :=
  a1 * (1 - q^n) / (1 - q)

theorem sum_first_six_terms :
  geometric_sum (-1) 2 6 = 63 :=
sorry

end sum_first_six_terms_l69_69611


namespace platform_length_l69_69066

theorem platform_length (speed_kmph : ℕ) (time_platform : ℕ) (time_man : ℕ) :
  speed_kmph = 72 → time_platform = 35 → time_man = 18 → 
  let speed_mps := speed_kmph * 1000 / 3600 in
  let length_train := speed_mps * time_man in
  let total_distance := speed_mps * time_platform in
  total_distance - length_train = 340 := 
by intros h1 h2 h3;
  sorry

end platform_length_l69_69066


namespace quadrilateral_three_properties_l69_69576

theorem quadrilateral_three_properties
  (ABCD : Type)
  [quadrilateral ABCD]
  {A B C D : pt}
  (h1 : perpendicular_diagonals ABCD)
  (h2 : cyclic ABCD) :
  passes_through_intersection_point ABCD :=
by
  sorry

end quadrilateral_three_properties_l69_69576


namespace true_proposition_l69_69920

open Classical

-- Define real numbers and basic operations on real numbers
variable {x : ℝ}

-- Formalizing the statements:
def p := ∀ x : ℝ, 2^x < 3^x

def q := ∃ x_0 : ℝ, x_0^2 - 2*x_0 + 1 > 0

-- The statement to be proved
theorem true_proposition : ¬p ∧ q :=
by
  sorry

end true_proposition_l69_69920


namespace train_speed_is_144_kmph_l69_69785

noncomputable def length_of_train : ℝ := 130 -- in meters
noncomputable def time_to_cross_pole : ℝ := 3.249740020798336 -- in seconds
noncomputable def speed_m_per_s : ℝ := length_of_train / time_to_cross_pole -- in m/s
noncomputable def conversion_factor : ℝ := 3.6 -- 1 m/s = 3.6 km/hr

theorem train_speed_is_144_kmph : speed_m_per_s * conversion_factor = 144 :=
by
  sorry

end train_speed_is_144_kmph_l69_69785


namespace travel_speed_l69_69408

theorem travel_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 195) (h_time : time = 3) : 
  distance / time = 65 :=
by 
  rw [h_distance, h_time]
  norm_num

end travel_speed_l69_69408


namespace least_n_divisible_by_10_l69_69999

def series (n : ℕ) : ℤ :=
  (Finset.range n).sum (λ k, (k + 1) * (2 : ℤ) ^ (k + 1))

theorem least_n_divisible_by_10 (n : ℕ) (h_n : n ≥ 2012) :
  10 ∣ series n ↔ n = 2018 :=
by {
  sorry
}

end least_n_divisible_by_10_l69_69999


namespace coordinates_of_point_l69_69603

theorem coordinates_of_point (x y : ℝ) (hx : x < 0) (hy : y > 0) (dx : |x| = 3) (dy : |y| = 2) :
  (x, y) = (-3, 2) := 
sorry

end coordinates_of_point_l69_69603


namespace num_four_digit_palindromic_squares_l69_69019

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69019


namespace expectation_equality_probability_X_le_x_eq_l69_69274
noncomputable theory

section

variable {Ω : Type*} [MeasureSpace Ω]
variable (X : Ω → ℝ)
variable (φ : ℝ → ℝ)
variable (filter : MeasureTheory.ConditionalProbability Ω [MeasureTheory.MeasurableSpace.pullback MeasureTheory.MeasureSpace.toMeasurableSpace X])
variable [MeasureTheory.NeedsProof]

-- Symmetric distribution of X
def symmetric_distribution (X : Ω → ℝ) : Prop :=
∀ ω, X ω = -X ω

-- The given conditions
axiom symmetric_X : symmetric_distribution X
axiom φ_defined : ∀ ω, MeasureTheory.ExpectedValue φ (MeasureTheory.Norm.on X ω)

-- Question 1: Showing the expectation equality
theorem expectation_equality :
  ∀ ω, MeasureTheory.ExpectedValue φ (|X| ω) = 1 / 2 * (φ (|X ω|) + φ (-|X ω|)) := 
sorry

-- The Indicator function
def I (p : Prop) : ℝ := if p then 1 else 0

-- Question 2: Probability computation
def probability_X_le_x (x : ℝ) : Ω → ℝ :=
λ ω, MeasureTheory.ExpectedValue (λ y, I (X y ≤ x)) (|X| ω)

theorem probability_X_le_x_eq :
 ∀ x ≥ 0, probability_X_le_x X x = 1 / 2 * (I (|X| ≤ x) + 1) :=
sorry

end

end expectation_equality_probability_X_le_x_eq_l69_69274


namespace range_of_a_l69_69125

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x ≥ 1/2) → 2 * x + a ≥ sqrt (2 * x - 1)) ↔ a ∈ Icc (-3/4) (real.infinity) := by
  sorry

end range_of_a_l69_69125


namespace count_ks_not_exceeding_267000_l69_69859

noncomputable def count_divisible_k (N : ℕ) (d : ℕ) : ℕ :=
  (finset.filter (λ k, (k^2 - 1) % d = 0) (finset.range (N + 1))).card

theorem count_ks_not_exceeding_267000 :
  count_divisible_k 267000 267 = 4000 :=
begin
  sorry
end

end count_ks_not_exceeding_267000_l69_69859


namespace number_of_days_l69_69569

noncomputable def days_to_lay_bricks (b c f : ℕ) : ℕ :=
(b * b) / f

theorem number_of_days (b c f : ℕ) (h_nonzero_f : f ≠ 0) (h_bc_pos : b > 0 ∧ c > 0) :
  days_to_lay_bricks b c f = (b * b) / f :=
by 
  sorry

end number_of_days_l69_69569


namespace arithmetic_seq_problem_l69_69605

variable (a : ℕ → ℕ)
variable h_seq : ArithmeticSequence a
variable h_cond : a 5 + a 13 = 40

theorem arithmetic_seq_problem : a 8 + a 9 + a 10 = 60 :=
by
  sorry

end arithmetic_seq_problem_l69_69605


namespace tournament_problem_l69_69086

noncomputable def probability_of_winning_sequence (sequence : List Bool) : ℚ :=
  match sequence with
  | [] => 1
  | [true] => 1 / 2
  | [true, true] => 1 / 2 * 3 / 4
  | [true, true, true] => 1 / 2 * 3 / 4 * 3 / 4
  | [false] => 1 / 2
  | [false, true] => 1 / 2 * 1 / 3
  | [false, true, true] => 1 / 2 * 1 / 3 * 3 / 4
  | [false, true, true, true] => 1 / 2 * 1 / 3 * 3 / 4 * 3 / 4
  | [true, false] => 1 / 2 * 1 / 4
  | [true, false, true] => 1 / 2 * 1 / 4 * 1 / 3
  | [true, false, true, true] => 1 / 2 * 1 / 4 * 1 / 3 * 3 / 4
  | [true, true, false] => 1 / 2 * 3 / 4 * 1 / 4
  | [true, true, false, true] => 1 / 2 * 3 / 4 * 1 / 4 * 1 / 3
  | _ => 0

def probability_of_winning_3_games : ℚ :=
  probability_of_winning_sequence [true, true, true] +
  probability_of_winning_sequence [false, true, true, true] +
  probability_of_winning_sequence [true, false, true, true] +
  probability_of_winning_sequence [true, true, false, true]

def reduced_fraction : ℚ := probability_of_winning_3_games

lemma reduction_to_lowest_terms (a b: ℚ) (p: ℚ) (h: p = a / b) : ∃ x y: ℚ, p = x / y ∧ x + y = 23 :=
sorry

theorem tournament_problem: 
  ∃ x y: ℚ, reduced_fraction = x / y ∧ x + y = 23 :=
reduction_to_lowest_terms 7 16 _ rfl

end tournament_problem_l69_69086


namespace pentadecagon_triangle_count_l69_69963

-- Define the problem of selecting 3 vertices out of 15 to form a triangle
theorem pentadecagon_triangle_count : 
  ∃ (n : ℕ), n = nat.choose 15 3 ∧ n = 455 := 
by {
  sorry
}

end pentadecagon_triangle_count_l69_69963


namespace range_of_real_number_a_l69_69584

theorem range_of_real_number_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 1 = 0 → x = a) ↔ (a = 0 ∨ a ≥ 9/4) :=
sorry

end range_of_real_number_a_l69_69584


namespace additional_people_can_ride_l69_69721

def num_cars := 2
def num_vans := 3
def people_per_car := 5
def people_per_van := 3
def max_people_per_car := 6
def max_people_per_van := 8

theorem additional_people_can_ride :
  let actual_people := num_cars * people_per_car + num_vans * people_per_van in
  let max_people := num_cars * max_people_per_car + num_vans * max_people_per_van in
  max_people - actual_people = 17 :=
by
  sorry

end additional_people_can_ride_l69_69721


namespace monotonically_increasing_interval_range_of_m_l69_69910

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * sin (π / 4 + x) ^ 2 - sqrt 3 * cos (2 * x) - 1

-- Define the domain
def domain (x : ℝ) : Prop := π / 4 ≤ x ∧ x ≤ π / 2

-- Monotonically increasing interval proof statement
theorem monotonically_increasing_interval :
  ∀ ⦃x : ℝ⦄, domain x → 
  (π / 4 ≤ x ∧ x ≤ 5 * π / 12) :=
sorry

-- Range of m proof statement
theorem range_of_m :
  ∀ (m : ℝ), (∀ ⦃x : ℝ⦄, domain x → |f x - m| < 2) → 
  (0 < m ∧ m < 3) :=
sorry

end monotonically_increasing_interval_range_of_m_l69_69910


namespace sqrt_x_plus_inv_sqrt_x_l69_69641

variable (x : ℝ) (hx : 0 < x) (h : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (hx : 0 < x) (h : x + 1 / x = 50) : 
  sqrt x + 1 / sqrt x = 2 * sqrt 13 := 
sorry

end sqrt_x_plus_inv_sqrt_x_l69_69641


namespace highway_extension_l69_69403

def initial_length : ℕ := 200
def final_length : ℕ := 650
def first_day_construction : ℕ := 50
def second_day_construction : ℕ := 3 * first_day_construction
def total_construction : ℕ := first_day_construction + second_day_construction
def total_extension_needed : ℕ := final_length - initial_length
def miles_still_needed : ℕ := total_extension_needed - total_construction

theorem highway_extension : miles_still_needed = 250 := by
  sorry

end highway_extension_l69_69403


namespace anita_apples_l69_69443

theorem anita_apples (num_students : ℕ) (apples_per_student : ℕ) (total_apples : ℕ) 
  (h1 : num_students = 60) 
  (h2 : apples_per_student = 6) 
  (h3 : total_apples = num_students * apples_per_student) : 
  total_apples = 360 := 
by
  sorry

end anita_apples_l69_69443


namespace probability_of_point_in_spheres_l69_69063

noncomputable def radius_of_inscribed_sphere (R : ℝ) : ℝ := 2 * R / 3
noncomputable def radius_of_tangent_spheres (R : ℝ) : ℝ := 2 * R / 3

theorem probability_of_point_in_spheres
  (R : ℝ)  -- Radius of the circumscribed sphere
  (r : ℝ := radius_of_inscribed_sphere R)  -- Radius of the inscribed sphere
  (r_t : ℝ := radius_of_tangent_spheres R)  -- Radius of each tangent sphere
  (volume : ℝ := 4/3 * Real.pi * r^3)  -- Volume of each smaller sphere
  (total_small_volume : ℝ := 5 * volume)  -- Total volume of smaller spheres
  (circumsphere_volume : ℝ := 4/3 * Real.pi * (2 * R)^3)  -- Volume of the circumscribed sphere
  : 
  total_small_volume / circumsphere_volume = 5 / 27 :=
by
  sorry

end probability_of_point_in_spheres_l69_69063


namespace sum_of_digits_in_binary_l69_69520

theorem sum_of_digits_in_binary (A n : ℕ) (hA: A = 2^n - 1) :
  (nat.binary_digits (n * A)).sum = n := 
by
  sorry

end sum_of_digits_in_binary_l69_69520


namespace range_trig_func_l69_69339

noncomputable def trig_func (x : ℝ) : ℝ :=
  sin x * (cos x - real.sqrt 3 * sin x)

theorem range_trig_func :
  ∀ (y : ℝ), (∃ (x : ℝ), (0 ≤ x ∧ x ≤ real.pi / 2) ∧ y = trig_func x) ↔ (-real.sqrt 3 ≤ y ∧ y ≤ 1 - real.sqrt 3 / 2) := 
  sorry

end range_trig_func_l69_69339


namespace problem_solution_l69_69822

def is_desirable_n (n : ℕ) : Prop :=
  ∃ (r b : ℕ), n = r + b ∧ r^2 - r*b + b^2 = 2007 ∧ 3 ∣ r ∧ 3 ∣ b

theorem problem_solution :
  ∀ n : ℕ, (is_desirable_n n → n = 69 ∨ n = 84) :=
by
  sorry

end problem_solution_l69_69822


namespace triangles_from_pentadecagon_l69_69939

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end triangles_from_pentadecagon_l69_69939


namespace mitzi_amount_brought_l69_69661

-- Define the amounts spent on different items
def ticket_cost : ℕ := 30
def food_cost : ℕ := 13
def tshirt_cost : ℕ := 23

-- Define the amount of money left
def amount_left : ℕ := 9

-- Define the total amount spent
def total_spent : ℕ :=
  ticket_cost + food_cost + tshirt_cost

-- Define the total amount brought to the amusement park
def amount_brought : ℕ :=
  total_spent + amount_left

-- Prove that the amount of money Mitzi brought to the amusement park is 75
theorem mitzi_amount_brought : amount_brought = 75 := by
  sorry

end mitzi_amount_brought_l69_69661


namespace distance_p_ran_l69_69659

variable (d t v : ℝ)
-- d: head start distance in meters
-- t: time in minutes
-- v: speed of q in meters per minute

theorem distance_p_ran (h1 : d = 0.3 * v * t) : 1.3 * v * t = 1.3 * v * t :=
by
  sorry

end distance_p_ran_l69_69659


namespace distance_from_tee_to_hole_l69_69790

-- Define the constants based on the problem conditions
def s1 : ℕ := 180
def s2 : ℕ := (1 / 2 * s1 + 20 - 20)

-- Define the total distance calculation
def total_distance := s1 + s2

-- State the ultimate theorem that needs to be proved
theorem distance_from_tee_to_hole : total_distance = 270 := by
  sorry

end distance_from_tee_to_hole_l69_69790


namespace pentadecagon_triangle_count_l69_69944

theorem pentadecagon_triangle_count :
  ∑ k in finset.range 15, if k = 3 then nat.choose 15 3 else 0 = 455 :=
by {
  sorry
}

end pentadecagon_triangle_count_l69_69944


namespace cupcakes_needed_l69_69252

theorem cupcakes_needed :
  let fourth_grade_classes := 12
  let students_per_fourth_grade_class := 45
  let pe_classes := 2
  let students_per_pe_class := 90
  let afterschool_clubs := 4
  let students_per_afterschool_club := 60
  let total_students := 
    (fourth_grade_classes * students_per_fourth_grade_class) +
    (pe_classes * students_per_pe_class) +
    (afterschool_clubs * students_per_afterschool_club)
  in
  total_students = 960 :=
by 
  sorry

end cupcakes_needed_l69_69252


namespace working_light_bulbs_l69_69307

def total_lamps : Nat := 20
def bulbs_per_lamp : Nat := 7
def fraction_burnt_out : ℚ := 1 / 4
def bulbs_burnt_per_lamp : Nat := 2

theorem working_light_bulbs : 
  let total_bulbs := total_lamps * bulbs_per_lamp
  let burnt_out_lamps := (fraction_burnt_out * total_lamps).toNat
  let total_bulbs_burnt_out := burnt_out_lamps * bulbs_burnt_per_lamp
  let working_bulbs := total_bulbs - total_bulbs_burnt_out
  working_bulbs = 130 :=
by
  sorry

end working_light_bulbs_l69_69307


namespace pentadecagon_triangle_count_l69_69964

-- Define the problem of selecting 3 vertices out of 15 to form a triangle
theorem pentadecagon_triangle_count : 
  ∃ (n : ℕ), n = nat.choose 15 3 ∧ n = 455 := 
by {
  sorry
}

end pentadecagon_triangle_count_l69_69964


namespace max_value_sqrt_diff_l69_69326

noncomputable def max_distance_diff : ℝ :=
  Real.sqrt ((1 - 0)^2 + (2 - 1)^2)

theorem max_value_sqrt_diff (x : ℝ) : 
  Real.sqrt (x^2 - 2 * x + 5) - Real.sqrt (x^2 + 1) ≤ max_distance_diff :=
by {
  sorry,
}

end max_value_sqrt_diff_l69_69326


namespace max_sum_of_multiplication_table_l69_69335

theorem max_sum_of_multiplication_table :
  let numbers := [3, 5, 7, 11, 17, 19]
  let repeated_num := 19
  ∃ d e f, d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧ d ≠ e ∧ e ≠ f ∧ d ≠ f ∧
  3 * repeated_num * (d + e + f) = 1995 := 
by {
  sorry
}

end max_sum_of_multiplication_table_l69_69335


namespace hyperbola_eccentricity_is_5_l69_69297

-- Definitions for conditions in the problem
variables {a b : ℝ} (ha : a > 0) (hb : b > 0)

def is_on_hyperbola (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def is_focus (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (c, 0) ∧ F2 = (-c, 0) ∧ c = sqrt(a^2 + b^2)

def is_eccentricity (e : ℝ) : Prop :=
  e = c / a

axiom P_is_on_hyperbola (P : ℝ × ℝ) : is_on_hyperbola a b P.1 P.2

-- Conditions about the triangle F1PF2
axiom angle_F1PF2_is_right (F1 F2 P : ℝ × ℝ) [is_focus F1 F2] :
  ∃ angle : ℝ, angle = geometry.angle F1 P F2 ∧ angle = π / 2

axiom sides_form_arithmetic_sequence (F1 F2 P : ℝ × ℝ) [is_focus F1 F2] :
  ∃ (m d : ℝ), abs (abs (dist P F2) - abs (dist P F1)) = 2*a ∧
  abs (dist P F1) + abs (dist P F2) = 2*sqrt(a^2 + b^2) ∧
  abs (dist P F1 - dist P F2)^2 + abs(dist P F1)^2 = abs (dist F1 F2)^2

-- The proof statement we need to show
theorem hyperbola_eccentricity_is_5 (c a : ℝ) [is_focus (c, 0) (-c, 0)] :
  c = sqrt(a^2 + b^2) → is_eccentricity (5 : ℝ) :=
by
  sorry

end hyperbola_eccentricity_is_5_l69_69297


namespace cyclic_quadrilateral_MNEF_l69_69730

open EuclideanGeometry

-- Define the necessary points and circles
variables (O O' A B P Q M N E F : Point)
variables (circleO : Circle O)
variables (circleO' : Circle O')
variables (H1 : A ∈ circleO)
variables (H2 : A ∈ circleO')
variables (H3 : B ∈ circleO)
variables (H4 : B ∈ circleO')
variables (H5 : P ∈ circleO)
variables (H6 : Q ∈ circleO')
variables (H7 : AP = AQ)
variables (H8 : M ∈ Line PQ)
variables (H9 : N ∈ Line PQ)
variables (H10 : M ∈ circleO)
variables (H11 : N ∈ circleO')

-- Define E and F as centers of arcs BP and BQ not containing A
-- This defines E and F more algebraically, you might need a condition to specify they are on the perpendicular bisectors
variables (arcBP : Arc B P O)
variables (arcBQ : Arc B Q O')

def is_center_of_arc (X : Point) (arc : Arc) : Prop :=
  X ∈ arc ∧ ∃ midpoint (X = arc.equiv_cp_arc_center)

axiom center_def : 
  (is_center_of_arc E arcBP) ∧ ¬(A ∈ arcBP) ↔ 
  (is_center_of_arc F arcBQ) ∧ ¬(A ∈ arcBQ)

-- State the theorem as a theorem, skipping the actual proof
theorem cyclic_quadrilateral_MNEF : CyclicQuadrilateral M N E F :=
sorry

end cyclic_quadrilateral_MNEF_l69_69730


namespace mini_bottles_needed_to_fill_jumbo_l69_69407

def mini_bottle_capacity : ℕ := 45
def jumbo_bottle_capacity : ℕ := 600

-- The problem statement expressed as a Lean theorem.
theorem mini_bottles_needed_to_fill_jumbo :
  (jumbo_bottle_capacity + mini_bottle_capacity - 1) / mini_bottle_capacity = 14 :=
by
  sorry

end mini_bottles_needed_to_fill_jumbo_l69_69407


namespace range_of_f_on_interval_l69_69471

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem range_of_f_on_interval :
  set.Icc (-4) 0 \{0} = set.range (λ x, f x) ∩ set.Icc 0 3 :=
sorry

end range_of_f_on_interval_l69_69471


namespace find_scaling_matrix_3d_l69_69499

open Matrix

theorem find_scaling_matrix_3d {n : Type*} [DecidableEq n] [Fintype n] (v : Fin 3 → ℝ) :
  (∃ M : Matrix (Fin 3) (Fin 3) ℝ, ∀ (v : Fin 3 → ℝ), mulVec M v = (3 : ℝ) • v) ->
  (∃ M : Matrix (Fin 3) (Fin 3) ℝ, M = ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]]) :=
begin
  sorry
end

end find_scaling_matrix_3d_l69_69499


namespace find_sum_a_f_neg2_l69_69908

noncomputable def f (x a : ℝ) : ℝ :=
  if |x| ≤ 1 then log 2 (x + a) else -10 / (|x| + 3)

theorem find_sum_a_f_neg2 (a : ℝ) (f : ℝ → ℝ → ℝ) 
  (hf0 : f 0 a = 2) (ha : a = 4) : a + f (-2) a = 2 := by
  -- the proof is omitted, the statement should be provable with further steps
  sorry

end find_sum_a_f_neg2_l69_69908


namespace correct_propositions_l69_69905

theorem correct_propositions (a b : ℝ) (h1 : a > b > 0) (h2 : 2 * a + b = 1) :
  ¬((a > b > 0) → (1 / a > 1 / b)) ∧
  ((a > b > 0) → (a - 1 / a > b - 1 / b)) ∧
  ¬((a > b > 0) → (2 * a + b) / (a + 2 * b) > a / b) ∧
  ((a > 0 ∧ b > 0 ∧ 2 * a + b = 1) → (2 / a + 1 / b ≥ 9)) :=
by
  -- Assuming the minimum is attained when a = b = 1/3 in proposition 4
  sorry

end correct_propositions_l69_69905


namespace perimeter_of_right_triangle_l69_69052

def right_triangle_perimeter (a b : ℝ) :=
  a + b + real.sqrt (a^2 + b^2)

theorem perimeter_of_right_triangle (y : ℝ) (z : ℝ) (h1 : 1 / 2 * 30 * y = 150) (h2: z^2 = 30^2 + y^2) :
  right_triangle_perimeter 30 y = 40 + 10 * real.sqrt 10 :=
by
  sorry

end perimeter_of_right_triangle_l69_69052


namespace monic_quadratic_polynomial_with_root_l69_69855

theorem monic_quadratic_polynomial_with_root (x : ℝ) : 
  ∃ p : polynomial ℝ, monic p ∧ p.coeff 1 =  -4 ∧ p.coeff 0 = 13 ∧ (∀ z : ℂ, z = (2 - 3 * I) → p.eval z.re = 0) :=
sorry

end monic_quadratic_polynomial_with_root_l69_69855


namespace find_a_plus_b_l69_69698

noncomputable def f (a : ℝ) (x : ℝ) := x / (x + a)
noncomputable def g (b : ℝ) (x : ℝ) := Real.log10 (10^x + 1) + b * x

def is_symmetric_about (a : ℝ) (p : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (p.1 + x) = p.2 + f (p.1 - x)

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

theorem find_a_plus_b (a b : ℝ) :
  is_symmetric_about a (1, 1) (f a) →
  is_even (g b) →
  a + b = -3 / 2 :=
sorry

end find_a_plus_b_l69_69698


namespace train_speed_first_part_l69_69065

-- Define the necessary variables and constants
variables {x v : ℝ}

-- Given conditions in formal statements 
def train_first_part (d₁ t₁ : ℝ) (v : ℝ) := d₁ = x ∧ t₁ = d₁ / v
def train_second_part (d₂ t₂ : ℝ) := d₂ = 2 * x ∧ t₂ = d₂ / 20
def average_speed (total_d total_t : ℝ) := total_d = 4 * x ∧ total_t = total_d / 32

-- Statement problem
theorem train_speed_first_part:
  ∀ (d₁ d₂ total_d t₁ t₂ total_t : ℝ), train_first_part d₁ t₁ v → train_second_part d₂ t₂ →
  average_speed total_d total_t → t₁ + t₂ = total_t → v = 8.8 :=
by
  intros
  sorry

end train_speed_first_part_l69_69065


namespace probability_rodney_correct_guess_l69_69310

open Probability

def valid_numbers : Set ℕ := {n | n ≥ 10 ∧ n < 100 ∧ odd (n / 10) ∧ ∃ m, n % 10 = m ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ n > 75}

theorem probability_rodney_correct_guess : Probs (fun n => n ∈ valid_numbers) = 1 / 3 := by
  sorry

end probability_rodney_correct_guess_l69_69310


namespace count_sequences_with_perfect_square_l69_69632

def T : List (ℤ × ℤ × ℤ) :=
  List.filter (λ (t : ℤ × ℤ × ℤ), (2 ≤ t.1 ∧ t.1 ≤ 20) ∧ (2 ≤ t.2 ∧ t.2 ≤ 20) ∧ (2 ≤ t.3 ∧ t.3 ≤ 20))
              (List.product (List.product (List.range' 2 19) (List.range' 2 19)) (List.range' 2 19))

def sequence (b1 b2 b3 : ℤ) : ℕ → ℤ
| 0     := b1
| 1     := b2
| 2     := b3
| (n+3) := sequence (n+2) + sequence (n+1) - sequence n

def is_perfect_square (n : ℤ) : Bool :=
  let s := Int.sqrt n
  s * s = n

def has_perfect_square (b1 b2 b3 : ℤ) : Bool :=
  let seq := sequence b1 b2 b3
  List.any (List.map seq [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]) is_perfect_square

theorem count_sequences_with_perfect_square : ∃ α : ℕ, α = T.count (λ t, has_perfect_square t.1 t.2 t.3) :=
  sorry

end count_sequences_with_perfect_square_l69_69632


namespace find_oxygen_weight_l69_69120

-- Definitions of given conditions
def molecular_weight : ℝ := 68
def weight_hydrogen : ℝ := 1
def weight_chlorine : ℝ := 35.5

-- Definition of unknown atomic weight of oxygen
def weight_oxygen : ℝ := 15.75

-- Mathematical statement to prove
theorem find_oxygen_weight :
  weight_hydrogen + weight_chlorine + 2 * weight_oxygen = molecular_weight := by
sorry

end find_oxygen_weight_l69_69120


namespace triangles_in_pentadecagon_l69_69950

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69950


namespace product_zero_when_a_is_3_l69_69107

theorem product_zero_when_a_is_3 : 
  let a := 3 in
  (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by 
  let a := 3
  -- Proof here
  sorry

end product_zero_when_a_is_3_l69_69107


namespace infinite_occurence_of_coefficient_sum_l69_69514

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def polynomial_sum (P : ℤ[x]) : ℕ :=
  (P.support.map (λ n, P.coeff n)).sum.natAbs

theorem infinite_occurence_of_coefficient_sum {P : ℤ[x]} (h : ∀ n, P.coeff n ∈ ℕ) :
  ∃ S, ∀ N, ∃ n > N, digit_sum (P.eval n) = S :=
sorry

end infinite_occurence_of_coefficient_sum_l69_69514


namespace sports_popularity_order_l69_69447

theorem sports_popularity_order :
  let soccer := (13 : ℚ) / 40
  let baseball := (9 : ℚ) / 30
  let basketball := (7 : ℚ) / 20
  let volleyball := (3 : ℚ) / 10
  basketball > soccer ∧ soccer > baseball ∧ baseball = volleyball :=
by
  sorry

end sports_popularity_order_l69_69447


namespace poly_plus_three_has_distinct_real_roots_l69_69880

-- Definitions for the conditions
variables {α : Type*} [linear_ordered_field α]
variables (P : polynomial α) (n : ℕ)

-- Conditions
def degree_gt_five := P.nat_degree > 5
def integer_coeffs := ∀ coeff ∈ P.coeff.range, coeff ∈ ℤ
def distinct_integer_roots := ∃ (roots : fin n → ℤ), ∀ (i j : fin n), i ≠ j → roots i ≠ roots j ∧ P.eval (roots i) = 0

-- The theorem to prove
theorem poly_plus_three_has_distinct_real_roots
  (hn : n > 5)
  (hdeg : degree_gt_five P)
  (hint : integer_coeffs P)
  (hroots : distinct_integer_roots P n) :
  ∃ (roots_R : fin n → α), ∀ (i j : fin n), i ≠ j → roots_R i ≠ roots_R j ∧ (P + polynomial.C 3).eval (roots_R i) = 0 :=
sorry

end poly_plus_three_has_distinct_real_roots_l69_69880


namespace real_part_of_z_l69_69175

def z : ℂ := (-1 + 2 * complex.I) * complex.I

theorem real_part_of_z : z.re = -2 :=
by
  sorry

end real_part_of_z_l69_69175


namespace find_f_expression_find_g_expression_l69_69543

-- Given conditions
def minimum_value_f (f : ℝ → ℝ) (y : ℝ) : Prop :=
  ∀ x, f(x) ≥ y

def f_satisfies_conditions (f : ℝ → ℝ) : Prop :=
  minimum_value_f f 1 ∧ f 0 = 9 ∧ f 4 = 9

-- The function f(x) we need to prove
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 9

-- Proving the analytical expression of f(x)
theorem find_f_expression : f_satisfies_conditions f :=
sorry

-- The function g(m) we need to prove
def g (m : ℝ) : ℝ :=
  if m ≥ 2 then
    2 * m^2 - 8 * m + 9
  else if 0 < m ∧ m < 2 then
    1
  else
    2 * m^2 + 1

-- Proving the expression of g(m)
theorem find_g_expression : ∀ (m : ℝ), g(m) = 
  if m ≥ 2 then
    2 * m^2 - 8 * m + 9
  else if 0 < m ∧ m < 2 then
    1
  else
    2 * m^2 + 1 :=
sorry

end find_f_expression_find_g_expression_l69_69543


namespace notable_points_exists_l69_69350

noncomputable def construct_notable_points (a b c : Line) : Prop :=
  ∃ incenter centroid circumcenter orthocenter : Point,
    (not using vertices, transformations and geometric construction methods should apply) ∧
    incenter ∈ notable_points a b c ∧
    centroid ∈ notable_points a b c ∧
    circumcenter ∈ notable_points a b c ∧
    orthocenter ∈ notable_points a b c

theorem notable_points_exists (a b c : Line) : 
  construct_notable_points a b c := 
sorry

end notable_points_exists_l69_69350


namespace diagonal_of_square_l69_69897

-- Definitions based on conditions
def square_area := 8 -- Area of the square is 8 square centimeters

def diagonal_length (x : ℝ) : Prop :=
  (1/2) * x ^ 2 = square_area

-- Proof problem statement
theorem diagonal_of_square : ∃ x : ℝ, diagonal_length x ∧ x = 4 := 
sorry  -- statement only, proof skipped

end diagonal_of_square_l69_69897


namespace part_a_part_b_part_c_l69_69320

-- Define what it means for a ticket to be lucky
def is_lucky (n : ℕ) : Prop :=
  let digits := (List.range 6).map (λ i => (n / 10 ^ (5 - i)) % 10)
  (digits.take 3).sum = (digits.drop 3).sum

-- Number of lucky tickets in the range 0 to 999999
def N : ℕ := (List.range 1000000).countp is_lucky

-- Theorem statements to prove
theorem part_a : (∑ i in (Finset.range 10), x ^ i) ^ 3 * (∑ i in (Finset.range 10), x ^ -i) ^ 3 = ∑ i in (Finset.range 55), if i = 27 then N else coeff else a_i x else N else a_i else x ^ i
:= sorry

theorem part_b : (∑ i in (Finset.range 10), x ^ i) ^ 6 = ∑ i in (Finset.range 55), if i = 27 then N else coeff else x ^ i :=
  sorry

theorem part_c : N = 55252 :=
  sorry

end part_a_part_b_part_c_l69_69320


namespace triangle_area_l69_69609

theorem triangle_area (area_WXYZ : ℝ) (side_small_squares : ℝ) 
  (AB_eq_AC : (AB = AC)) (A_on_center : (A = O)) :
  area_WXYZ = 64 ∧ side_small_squares = 2 →
  ∃ (area_triangle_ABC : ℝ), area_triangle_ABC = 8 :=
by
  intros h
  sorry

end triangle_area_l69_69609


namespace roots_of_polynomial_l69_69841

noncomputable def quadratic_polynomial (x : ℝ) := x^2 - 4 * x + 13

theorem roots_of_polynomial : ∀ (z : ℂ), z = 2 - 3 * complex.I → quadratic_polynomial (z.re) = 0 :=
by
  intro z h
  sorry

end roots_of_polynomial_l69_69841


namespace four_digit_perfect_square_palindrome_count_l69_69008

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69008


namespace unique_root_conditions_l69_69115

theorem unique_root_conditions (m : ℝ) (x y : ℝ) :
  (x^2 = 2 * abs x ∧ abs x - y - m = 1 - y^2) ↔ m = 3 / 4 := sorry

end unique_root_conditions_l69_69115


namespace monic_quadratic_with_real_coeffs_l69_69837

open Complex Polynomial

theorem monic_quadratic_with_real_coeffs {x : ℂ} :
  (∀ a b c : ℝ, Polynomial.monic (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) ∧ 
  (x = 2 - 3 * Complex.I ∨ x = 2 + 3 * Complex.I) → Polynomial.eval (2 - 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0) ∧
  Polynomial.eval (2 + 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0 :=
begin
  sorry
end

end monic_quadratic_with_real_coeffs_l69_69837


namespace triangles_in_pentadecagon_l69_69977

theorem triangles_in_pentadecagon :
  let n := 15
  in (Nat.choose n 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69977


namespace g_a_eq_g_inv_a_l69_69266

noncomputable def f (a x : ℝ) := a * real.sqrt (1 - x^2) + real.sqrt (1 + x) + real.sqrt (1 - x)
noncomputable def t (x : ℝ) := real.sqrt (1 + x) + real.sqrt (1 - x)
noncomputable def m (a t : ℝ) := (1 / 2) * a * t^2 + t - a

noncomputable def g (a : ℝ) : ℝ :=
if a > -1/2 then a + 2
else if -real.sqrt(2) / 2 < a ∧ a ≤ -1/2 then -a - 1 / (2 * a)
else real.sqrt(2)

theorem g_a_eq_g_inv_a (a : ℝ) : g a = g (1 / a) ↔ 
  (a ∈ Icc (-real.sqrt(2)) (-real.sqrt(2) / 2)) ∨ (a = 1) :=
sorry

end g_a_eq_g_inv_a_l69_69266


namespace find_prime_after_six_nonprime_l69_69239

-- Definition of prime number
def is_prime (n : ℕ) : Prop := nat.prime n

-- Check if six consecutive numbers are nonprime
def six_consecutive_nonprime (start : ℕ) : Prop :=
  ∀ i : ℕ, i < 6 → ¬ is_prime (start + i)

theorem find_prime_after_six_nonprime :
  ∃ k : ℕ, (six_consecutive_nonprime (k - 7)) ∧ is_prime k ∧ k = 97 :=
by
  sorry

end find_prime_after_six_nonprime_l69_69239


namespace roots_of_polynomial_l69_69844

noncomputable def quadratic_polynomial (x : ℝ) := x^2 - 4 * x + 13

theorem roots_of_polynomial : ∀ (z : ℂ), z = 2 - 3 * complex.I → quadratic_polynomial (z.re) = 0 :=
by
  intro z h
  sorry

end roots_of_polynomial_l69_69844


namespace choose_1983_integers_no_arithmetic_progression_l69_69618

theorem choose_1983_integers_no_arithmetic_progression :
  ∃ (S : Finset ℕ), S.card = 1983 ∧ (∀ (a b c ∈ S), a ≠ b ∧ a ≠ c ∧ b ≠ c → ¬(2 * b = a + c)) ∧ ∀ x ∈ S, x ≤ 100000 :=
begin
  sorry
end

end choose_1983_integers_no_arithmetic_progression_l69_69618


namespace roots_of_polynomial_l69_69846

noncomputable def quadratic_polynomial (x : ℝ) := x^2 - 4 * x + 13

theorem roots_of_polynomial : ∀ (z : ℂ), z = 2 - 3 * complex.I → quadratic_polynomial (z.re) = 0 :=
by
  intro z h
  sorry

end roots_of_polynomial_l69_69846


namespace count_rectangles_4x5_grid_l69_69197

theorem count_rectangles_4x5_grid : ∀ (grid : ℕ × ℕ), (grid = (4, 5)) → (number_of_rectangles grid = 24) :=
by
  intros grid h_grid
  have number_of_rectangles : ℕ × ℕ → ℕ
  | (h, w) := (h * (h + 1) * w * (w + 1)) / 4
  sorry

end count_rectangles_4x5_grid_l69_69197


namespace profit_percentage_l69_69047

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 500) (h_selling : selling_price = 750) :
  ((selling_price - cost_price) / cost_price) * 100 = 50 :=
by
  sorry

end profit_percentage_l69_69047


namespace sqrt_fraction_combination_l69_69369

theorem sqrt_fraction_combination :
  (real.sqrt ((25 / 36) + (16 / 81) + (4 / 9))) = (real.sqrt 433 / 18) :=
by sorry

end sqrt_fraction_combination_l69_69369


namespace coefficient_equality_in_sum_l69_69237

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_equality_in_sum (n : ℕ) (h : n = 13) :
  binomial (n + 1) 3 = binomial (n + 1) 11 :=
by
  have h1 : n + 1 = 14 := by
    rw [h]
    rfl
  rw [h1]
  sorry

end coefficient_equality_in_sum_l69_69237


namespace min_value_expr_l69_69872

theorem min_value_expr (a b : ℝ) (h1 : 2 * a + b = a * b) (h2 : a > 0) (h3 : b > 0) : 
  ∃ a b, (a > 0 ∧ b > 0 ∧ 2 * a + b = a * b) ∧ (∀ x y, (x > 0 ∧ y > 0 ∧ 2 * x + y = x * y) → (1 / (x - 1) + 2 / (y - 2)) ≥ 2) ∧ ((1 / (a - 1) + 2 / (b - 2)) = 2) :=
by
  sorry

end min_value_expr_l69_69872


namespace smallest_abs_sum_l69_69637

theorem smallest_abs_sum (p q r s : ℤ) 
  (hnz_p : p ≠ 0) (hnz_q : q ≠ 0) (hnz_r : r ≠ 0) (hnz_s : s ≠ 0)
  (h_matrix : (Matrix.mul 
                  (Matrix.mul (Matrix.vec2 p q) (Matrix.vec2 r s)) 
                  (Matrix.vec2 p q)) 
               = (Matrix.vec2 15 0 0 15)):
  |p| + |q| + |r| + |s| = 11 := sorry

end smallest_abs_sum_l69_69637


namespace monic_quadratic_real_root_l69_69830

theorem monic_quadratic_real_root (a b : ℂ) (h : b = 2 - 3 * complex.I) :
  ∃ P : polynomial ℂ, P.monic ∧ P.coeff 2 = 1 ∧ P.coeff 1 = -4 ∧ P.coeff 0 = 13 ∧ P.is_root (2 - 3 * complex.I) :=
by
  sorry

end monic_quadratic_real_root_l69_69830


namespace chair_arrangements_48_l69_69220

theorem chair_arrangements_48 :
  ∃ (n : ℕ), n = 8 ∧ (∀ (r c : ℕ), r * c = 48 → 2 ≤ r ∧ 2 ≤ c) := 
sorry

end chair_arrangements_48_l69_69220


namespace inequality_a6_b6_l69_69517

theorem inequality_a6_b6 (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  a^6 + b^6 ≥ ab * (a^4 + b^4) :=
sorry

end inequality_a6_b6_l69_69517


namespace num_valid_colorings_equals_16800_l69_69477

def hexagon_vertices := {A, B, C, D, E, F}
def colors := {1, 2, 3, 4, 5, 6, 7}
def valid_coloring (c : hexagon_vertices → colors) :=
  (c A ≠ c B) ∧ (c B ≠ c C) ∧ (c C ≠ c D) ∧ (c D ≠ c E) ∧ (c E ≠ c F) ∧ (c F ≠ c A) ∧
  (c A ≠ c C) ∧ (c A ≠ c D) ∧ (c B ≠ c D) ∧ (c B ≠ c E) ∧ (c C ≠ c E) ∧ (c C ≠ c F)

theorem num_valid_colorings_equals_16800 :
  ∃ (f : hexagon_vertices → colors), valid_coloring f ∧
  (finset.card ((finset.univ : finset (hexagon_vertices → colors)).filter valid_coloring) = 16800) :=
sorry

end num_valid_colorings_equals_16800_l69_69477


namespace amanda_candy_bars_kept_l69_69427

noncomputable def amanda_initial_candy_bars : ℕ := 7
noncomputable def candy_bars_given_first_time : ℕ := 3
noncomputable def additional_candy_bars : ℕ := 30
noncomputable def multiplier : ℕ := 4

theorem amanda_candy_bars_kept :
  let remaining_after_first_give := amanda_initial_candy_bars - candy_bars_given_first_time in
  let total_after_buying_more := remaining_after_first_give + additional_candy_bars in
  let candy_bars_given_second_time := multiplier * candy_bars_given_first_time in
  total_after_buying_more - candy_bars_given_second_time = 22 :=
by
  sorry

end amanda_candy_bars_kept_l69_69427


namespace time_to_pass_tree_l69_69761

constant length_train : ℕ
constant length_platform : ℕ
constant time_to_pass_platform : ℝ
constant speed_of_train : ℝ

axiom length_train_def : length_train = 1200
axiom length_platform_def : length_platform = 1000
axiom time_to_pass_platform_def : time_to_pass_platform = 146.67
axiom speed_of_train_def : speed_of_train = (length_train + length_platform) / time_to_pass_platform

theorem time_to_pass_tree : (length_train : ℝ) / speed_of_train = 80 :=
by
  rw [←length_train_def, ←speed_of_train_def]
  sorry

end time_to_pass_tree_l69_69761


namespace trigonometric_solution_l69_69742

theorem trigonometric_solution (x : ℝ) (k : ℤ) :
  (cos (3 * x) ≠ 0) ∧ (cos (2 * x) ≠ 0) ∧ (3 * tg (3 * x) - 4 * tg (2 * x) = (tg (2 * x) ^ 2) * (tg (3 * x))) ↔ (x = ↑k * π) := 
begin
  sorry
end

end trigonometric_solution_l69_69742


namespace average_shifted_samples_l69_69535

variables (x1 x2 x3 x4 : ℝ)

theorem average_shifted_samples (h : (x1 + x2 + x3 + x4) / 4 = 2) :
  ((x1 + 3) + (x2 + 3) + (x3 + 3) + (x4 + 3)) / 4 = 5 :=
by
  sorry

end average_shifted_samples_l69_69535


namespace four_digit_perfect_square_palindrome_count_l69_69011

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69011


namespace tetrahedron_distance_height_sum_l69_69302

-- Definitions for distances from point P inside the tetrahedron to its faces
variables (x1 x2 x3 x4 h1 h2 h3 h4 : ℝ)

-- Condition stating that x1, x2, x3, x4 are distances from P to the faces, and h1, h2, h3, h4 are the corresponding heights
axiom distances_heights (x1 x2 x3 x4 h1 h2 h3 h4 : ℝ) -- This acts as the condition in Lean

-- The theorem we want to prove
theorem tetrahedron_distance_height_sum (h1_ne_zero : h1 ≠ 0) (h2_ne_zero : h2 ≠ 0) (h3_ne_zero : h3 ≠ 0) (h4_ne_zero : h4 ≠ 0) :
  (x1 / h1) + (x2 / h2) + (x3 / h3) + (x4 / h4) = 1 :=
by
  sorry

end tetrahedron_distance_height_sum_l69_69302


namespace count_four_digit_palindrome_squares_l69_69025

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69025


namespace monic_quadratic_polynomial_with_root_l69_69854

theorem monic_quadratic_polynomial_with_root (x : ℝ) : 
  ∃ p : polynomial ℝ, monic p ∧ p.coeff 1 =  -4 ∧ p.coeff 0 = 13 ∧ (∀ z : ℂ, z = (2 - 3 * I) → p.eval z.re = 0) :=
sorry

end monic_quadratic_polynomial_with_root_l69_69854


namespace race_distance_l69_69293

variable (speed_cristina speed_nicky head_start time_nicky : ℝ)

theorem race_distance
  (h1 : speed_cristina = 5)
  (h2 : speed_nicky = 3)
  (h3 : head_start = 12)
  (h4 : time_nicky = 30) :
  let time_cristina := time_nicky - head_start
  let distance_nicky := speed_nicky * time_nicky
  let distance_cristina := speed_cristina * time_cristina
  distance_nicky = 90 ∧ distance_cristina = 90 :=
by
  sorry

end race_distance_l69_69293


namespace repeating_decimal_0_7_36_as_fraction_l69_69108

noncomputable def repeating_decimal_to_fraction (x : ℝ) : ℝ :=
  if h : x = 0.736363636... 
  then (2452091 / 3330000) 
  else 0

theorem repeating_decimal_0_7_36_as_fraction :
  repeating_decimal_to_fraction 0.736363636... = 2452091 / 3330000 :=
by
  sorry

end repeating_decimal_0_7_36_as_fraction_l69_69108


namespace find_a_l69_69540

-- Define the equation of the circle and the line
def circle_eq (x y : ℝ) := x^2 + y^2 - 4*x + 6*y - 12 = 0
def line_eq (x y a : ℝ) := 2*x - y + a = 0

-- Define the center and radius of the circle
def circle_center : ℝ × ℝ := (2, -3)
def circle_radius : ℝ := 5

-- Define the chord length condition
def chord_length : ℝ := 4 * Real.sqrt 5

-- Define the distance from center to the line and the condition a > -5
def distance_from_center_to_line (a : ℝ) : ℝ := abs (2 * circle_center.1 - circle_center.2 + a) / Real.sqrt (2^2 + (-1)^2)
def condition_a (a : ℝ) := a > -5

-- Total function to verify value of a
def verify_a (a : ℝ) : Prop :=
  distance_from_center_to_line a = 2 ^ Real.sqrt 5 ∧
  chord_length = 4 * Real.sqrt 5 ∧
  condition_a a

-- Lean statement to prove the equivalent math proof problem
theorem find_a : ∃ a : ℝ, verify_a a ∧ a = -2 :=
sorry

end find_a_l69_69540


namespace symmetric_trapezoid_proof_l69_69092

theorem symmetric_trapezoid_proof 
  (A B C D E F G : Type) 
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] [metric_space F] 
  [metric_space G] :
  (dist A C = dist B D) → 
  (dist A D = dist B C) → 
  (dist G E = (1/2) * dist C D) → 
  (dist G F = (1/2) * dist B A) → 
  (dist E F = (1/2) * (dist B A - dist C D)) → 
  dist A C > dist B C ∧
  dist B C > dist E F := 
sorry

end symmetric_trapezoid_proof_l69_69092


namespace four_digit_perfect_square_palindrome_count_l69_69003

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69003


namespace michelle_gas_usage_l69_69287

theorem michelle_gas_usage :
  let initial_gas := 0.5
  let final_gas := 0.17
  initial_gas - final_gas = 0.33 :=
by
  let initial_gas := 0.5
  let final_gas := 0.17
  have h : initial_gas - final_gas = 0.33 := by norm_num
  exact h

end michelle_gas_usage_l69_69287


namespace minimum_value_frac_l69_69888

theorem minimum_value_frac (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (2 / a) + (3 / b) ≥ 5 + 2 * Real.sqrt 6 := 
sorry

end minimum_value_frac_l69_69888


namespace cyclist_wait_time_l69_69744

theorem cyclist_wait_time
  (hiker_speed : ℝ)
  (hiker_speed_pos : hiker_speed = 4)
  (cyclist_speed : ℝ)
  (cyclist_speed_pos : cyclist_speed = 24)
  (waiting_time_minutes : ℝ)
  (waiting_time_minutes_pos : waiting_time_minutes = 5) :
  (waiting_time_minutes / 60) * cyclist_speed = 2 →
  (2 / hiker_speed) * 60 = 30 :=
by
  intros
  sorry

end cyclist_wait_time_l69_69744


namespace arithmetic_sequence_common_diff_l69_69695

theorem arithmetic_sequence_common_diff (d : ℝ) (a : ℕ → ℝ) 
  (h_first_term : a 0 = 24) 
  (h_arithmetic_sequence : ∀ n, a (n + 1) = a n + d)
  (h_ninth_term_nonneg : 24 + 8 * d ≥ 0) 
  (h_tenth_term_neg : 24 + 9 * d < 0) : 
  -3 ≤ d ∧ d < -8/3 :=
by 
  sorry

end arithmetic_sequence_common_diff_l69_69695


namespace monic_quadratic_with_real_coeffs_l69_69840

open Complex Polynomial

theorem monic_quadratic_with_real_coeffs {x : ℂ} :
  (∀ a b c : ℝ, Polynomial.monic (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) ∧ 
  (x = 2 - 3 * Complex.I ∨ x = 2 + 3 * Complex.I) → Polynomial.eval (2 - 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0) ∧
  Polynomial.eval (2 + 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0 :=
begin
  sorry
end

end monic_quadratic_with_real_coeffs_l69_69840


namespace sqrt_x_plus_inv_sqrt_x_l69_69640

variable (x : ℝ) (hx : 0 < x) (h : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (hx : 0 < x) (h : x + 1 / x = 50) : 
  sqrt x + 1 / sqrt x = 2 * sqrt 13 := 
sorry

end sqrt_x_plus_inv_sqrt_x_l69_69640


namespace matrix_eigenvalues_inverse_matrix_transformed_ellipse_l69_69631

-- Define the matrix M
def M := Matrix.of (λ i j, if i = j then (if i = 0 then (2 : ℝ) else (if i = 1 then (3 : ℝ) else 0)) else 0)

-- Define the inverse matrix M^(-1)
def M_inv := Matrix.of (λ i j, if i = j then (if i = 0 then (1 / 2 : ℝ) else (if i = 1 then (1 / 3 : ℝ) else 0)) else 0)

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) := (x^2 / 4) + (y^2 / 9) = 1

-- Define the transformed ellipse equation
def transformed_ellipse_eq (x y : ℝ) := x^2 + y^2 = 1

theorem matrix_eigenvalues (λ : ℝ) (v : Matrix (Fin 2) (Fin 1) ℝ) :
    (v = ![(1 : ℝ), 0] ∧ λ = 2) ∨ (v = ![0, 1] ∧ λ = 3) ↔ 
    (∃ λ v, M.mul_vec v = λ • v) :=
sorry

theorem inverse_matrix :
    M.mul M_inv = 1 ∧ M_inv.mul M = 1 :=
sorry

theorem transformed_ellipse (x y : ℝ) :
    ellipse_eq x y → transformed_ellipse_eq (M_inv 0 0 • x + M_inv 0 1 • y) (M_inv 1 0 • x + M_inv 1 1 • y) :=
sorry

end matrix_eigenvalues_inverse_matrix_transformed_ellipse_l69_69631


namespace charge_difference_l69_69379

def  charge_single_room_percent_greater (P_s R_s G_s : ℝ) : ℝ :=
  P_s = 0.45 * R_s → P_s = 0.90 * G_s → 100

def charge_double_room_percent_greater (P_d R_d G_d : ℝ) : ℝ :=
  P_d = 0.70 * R_d → P_d = 0.80 * G_d → (1 / 7) * 100

theorem charge_difference (P_s R_s G_s P_d R_d G_d : ℝ) : 
  (P_s = 0.45 * R_s) →
  (P_s = 0.90 * G_s) →
  (P_d = 0.70 * R_d) →
  (P_d = 0.80 * G_d) →
  charge_single_room_percent_greater P_s R_s G_s - 
  charge_double_room_percent_greater P_d R_d G_d = 85.71 :=
by
  sorry

end charge_difference_l69_69379


namespace min_c_value_l69_69299

def y_eq_abs_sum (x a b c : ℝ) : ℝ := |x - a| + |x - b| + |x - c|
def y_eq_line (x : ℝ) : ℝ := -2 * x + 2023

theorem min_c_value (a b c : ℕ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (order : a ≤ b ∧ b < c)
  (unique_sol : ∃! x : ℝ, y_eq_abs_sum x a b c = y_eq_line x) :
  c = 2022 := sorry

end min_c_value_l69_69299


namespace solve_for_a_and_b_l69_69567

theorem solve_for_a_and_b (a b : ℤ) :
  (∀ x : ℤ, (x + a) * (x - 2) = x^2 + b * x - 6) →
  a = 3 ∧ b = 1 :=
by
  sorry

end solve_for_a_and_b_l69_69567


namespace perpendicular_segments_l69_69689

-- Definitions of circles ω1 and ω2
variable {ω1 ω2 : Circle}
variable {O1 O2 : Point} -- Centers of the circles ω1 and ω2
variable {C A B E F : Point} -- Points as defined in the problem

-- Statement of the theorem
theorem perpendicular_segments
  (H1 : ω1.contains O2) -- Circle ω1 passes through the center O2 of circle ω2
  (H2 : C ∈ ω1) -- Point C lies on circle ω1
  (H3 : Tangent C ω2 E) -- C is a point of tangency to ω2 at E
  (H4 : Tangent C ω2 F) -- C is a point of tangency to ω2 at F
  (H5 : Line C E ∩ ω1 = {A}) -- Tangent intersects ω1 at point A
  (H6 : Line C F ∩ ω1 = {B}) -- Tangent intersects ω1 at point B
  : ⊥(O1 O2) (A B) := -- Line segment O1 O2 is perpendicular to line segment AB
sorry

end perpendicular_segments_l69_69689


namespace four_digit_perfect_square_palindrome_count_l69_69006

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69006


namespace seq_a_property_seq_b_property_sum_cn_l69_69885

open Nat

-- Definitions for the sequences and initial conditions
def a (n : ℕ) : ℝ := 1 / 2 ^ (n - 1)
def b (n : ℕ) : ℕ := 2 * n - 1

-- c_n sequence definition
def c (n : ℕ) : ℝ := 1 / (a n) + b n

-- Sum of the sequence {c_n}
def S (n : ℕ) : ℝ := (finset.range n).sum (λ i => c (i + 1))

-- Theorem statements
theorem seq_a_property (n : ℕ) : a 1 = 1 ∧ ∀ n, a n = 1 / 2 ^ (n - 1) :=
by sorry

theorem seq_b_property (n : ℕ) : b 1 = 1 ∧ b 2 + b 4 = 10 ∧ ∀ n, b n = 2 * n - 1 :=
by sorry

theorem sum_cn (n : ℕ) : S n = 2^n + n^2 - 1 :=
by sorry

end seq_a_property_seq_b_property_sum_cn_l69_69885


namespace range_of_g_l69_69327

def floor (x : ℝ) : ℤ := ⌊x⌋

def g (x : ℝ) : ℝ := floor x + x

theorem range_of_g : set.range g = set.univ :=
by
sorry

end range_of_g_l69_69327


namespace integer_part_sqrt39_sub_3_l69_69699

theorem integer_part_sqrt39_sub_3 : ∀ (x : ℝ), x = real.sqrt 39 - 3 → ⌊x⌋ = 3 :=
by
  intro x
  assume h : x = real.sqrt 39 - 3
  sorry

end integer_part_sqrt39_sub_3_l69_69699


namespace number_of_triangles_in_pentadecagon_l69_69985

open Finset

theorem number_of_triangles_in_pentadecagon :
  ∀ (n : ℕ), n = 15 → (n.choose 3 = 455) := 
by 
  intros n hn 
  rw hn
  rw Nat.choose_eq_factorial_div_factorial (show 3 ≤ 15)
  { norm_num }

-- Proof omitted with sorry

end number_of_triangles_in_pentadecagon_l69_69985


namespace trajectory_eq_of_moving_point_Q_l69_69158

-- Define the conditions and the correct answer
theorem trajectory_eq_of_moving_point_Q 
(a b : ℝ) (h : a > b) (h_pos : b > 0)
(P Q : ℝ × ℝ)
(h_ellipse : (P.1^2) / (a^2) + (P.2^2) / (b^2) = 1)
(h_Q : Q = (P.1 * 2, P.2 * 2)) :
  (Q.1^2) / (4 * a^2) + (Q.2^2) / (4 * b^2) = 1 :=
by 
  sorry

end trajectory_eq_of_moving_point_Q_l69_69158


namespace sum_of_distances_focus_parabola_l69_69264

open Real

theorem sum_of_distances_focus_parabola :
  let P := (0 : ℝ, 1 / 4 : ℝ),
      p1 := (-2 : ℝ, 4 : ℝ),
      p2 := (0 : ℝ, 0 : ℝ),
      p3 := (10 : ℝ, 100 : ℝ),
      p4 := (-8 : ℝ, 64 : ℝ)
  in dist P p1 + dist P p2 + dist P p3 + dist P p4 = 
     sqrt 18.0625 + 1 / 4 + sqrt 9901.0625 + sqrt 4114.5625 :=
by
  sorry

end sum_of_distances_focus_parabola_l69_69264


namespace four_digit_perfect_square_palindrome_count_l69_69013

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69013


namespace num_four_digit_palindromic_squares_l69_69024

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69024


namespace largest_unorderable_l69_69794

theorem largest_unorderable :
  ∃ n, n = 43 ∧ (∀ a b c : ℕ, n ≠ 6 * a + 9 * b + 20 * c) :=
begin
  existsi 43,
  split,
  { refl },
  { intros a b c,
    sorry }
end

end largest_unorderable_l69_69794


namespace problem_part1_problem_part2_l69_69556

theorem problem_part1 (m n : ℝ) 
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 ↔ -1 ≤ 2 * x - 3 ∧ 2 * x - 3 ≤ 1)
  (h2 : ∀ x : ℝ, -1 ≤ 2 * x - 3 ∧ 2 * x - 3 ≤ 1 ↔ 1 ≤ x ∧ x ≤ 2) :
  m + n = 3 :=
by
  have h3 : 1 = m := sorry
  have h4 : 2 = n := sorry
  rw [h3, h4]
  norm_num
  
theorem problem_part2 (x a m : ℝ) 
  (h1 : |x - a| < m)
  (h2 : m = 1) :
  |x| < |a| + 1 :=
by
  have h3 : |x - a| < 1 := by rwa h2 at h1
  have h4 : |x| = |x - a + a| := by rw abs_add
  have h5 : |x - a + a| ≤ |x - a| + |a| := abs_add (x - a) a
  have h6 : |x - a| + |a| < 1 + |a| := add_lt_add_right h3 (|a|)
  rw h4
  exact lt_of_le_of_lt h5 h6

end problem_part1_problem_part2_l69_69556


namespace proof_problem_l69_69684

axiom is_line (m : Type) : Prop
axiom is_plane (α : Type) : Prop
axiom is_subset_of_plane (m : Type) (β : Type) : Prop
axiom is_perpendicular (a : Type) (b : Type) : Prop
axiom is_parallel (a : Type) (b : Type) : Prop

theorem proof_problem
  (m n : Type) 
  (α β : Type)
  (h1 : is_line m)
  (h2 : is_line n)
  (h3 : is_plane α)
  (h4 : is_plane β)
  (h_prop2 : is_parallel α β → is_subset_of_plane m α → is_parallel m β)
  (h_prop3 : is_perpendicular n α → is_perpendicular n β → is_perpendicular m α → is_perpendicular m β)
  : (is_subset_of_plane m β → is_perpendicular α β → ¬ (is_perpendicular m α)) ∧ 
    (is_parallel m α → is_parallel m β → ¬ (is_parallel α β)) :=
sorry

end proof_problem_l69_69684


namespace positive_integer_value_l69_69746

noncomputable def a : ℕ := (8 + 16 + 24 + 32 + 40 + 48 + 56) / 7
def b (n : ℕ) : ℕ := 2 * n

theorem positive_integer_value (n : ℕ) (h : a^2 - (b n)^2 = 0) : n = 16 :=
by
  have ha : a = 32 := by norm_num
  have hb : b n = 2 * n := by trivial
  rw [ha, hb] at h
  have : (32 + 2 * n) * (32 - 2 * n) = 0 := by assumption
  have h1 : 32 - 2 * n = 0 := by linarith
  linarith

end positive_integer_value_l69_69746


namespace square_possible_b_product_l69_69701

theorem square_possible_b_product : 
  (let b_values := {b : ℝ | (∃ h : ℝ, h = 5 ∧ ((b = 2 - h) ∨ (b = 2 + h)) ∧ (y = -1 ∨ y = 4 ∨ x = 2 ∨ x = b))}
  ∃ b1 b2 ∈ b_values, b1 * b2 = -21) :=
sorry

end square_possible_b_product_l69_69701


namespace range_of_f_in_interval_range_of_m_for_inequality_l69_69269

noncomputable def f (x : ℝ) : ℝ :=
  (Real.log x / Real.log 2 - 2) * (Real.log x / (2 * Real.log 2) - 1/2)

theorem range_of_f_in_interval :
  ∃ (a b : ℝ), (∀ x ∈ set.Icc (1 : ℝ) 4, f x ∈ set.Icc a b) ∧ set.Icc a b = set.Icc (-1/8) 1 :=
sorry

theorem range_of_m_for_inequality :
  ∃ (m : ℝ), (∀ x ∈ set.Icc (4 : ℝ) 16, f x > m * (Real.log x / (2 * Real.log 2))) ∧ m < 0 :=
sorry

end range_of_f_in_interval_range_of_m_for_inequality_l69_69269


namespace four_digit_perfect_square_palindrome_count_l69_69004

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69004


namespace quarterly_insurance_payment_l69_69289

theorem quarterly_insurance_payment (annual_payment : ℕ) (quarters_in_year : ℕ) (quarterly_payment : ℕ) : 
  annual_payment = 1512 → quarters_in_year = 4 → quarterly_payment * quarters_in_year = annual_payment → quarterly_payment = 378 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  sorry

end quarterly_insurance_payment_l69_69289


namespace volume_of_tank_in_liters_l69_69418

-- Define the conditions given in the problem
def cube_face_area : ℝ := 100 -- area of one face of the cube in square meters

-- Define the volume of the tank to be a cube with face area given by the conditions
noncomputable def tank_volume_cubic_meters (side_length : ℝ) : ℝ :=
  side_length ^ 3

-- Define the conversion from cubic meters to liters
def cubic_meters_to_liters (volume_cubic_meters : ℝ) : ℝ :=
  volume_cubic_meters * 1000

-- Define the side length of the cube given the face area
def side_length_of_cube (face_area : ℝ) : ℝ :=
  real.sqrt face_area

-- State the problem in Lean
theorem volume_of_tank_in_liters :
  cubic_meters_to_liters (tank_volume_cubic_meters (side_length_of_cube cube_face_area)) = 1000000 := by
  sorry

end volume_of_tank_in_liters_l69_69418


namespace smallest_n_mod5_l69_69472

theorem smallest_n_mod5 :
  ∃ n : ℕ, n > 0 ∧ 6^n % 5 = n^6 % 5 ∧ ∀ m : ℕ, m > 0 ∧ 6^m % 5 = m^6 % 5 → n ≤ m :=
by
  sorry

end smallest_n_mod5_l69_69472


namespace Hexagon_Concurrent_Parallelity_l69_69817

universe u

structure Point where
  x : ℝ
  y : ℝ

structure Hexagon where
  C1 C2 C3 C4 C5 C6 : Point

def perpendicular (p1 p2 p3 : Point) : Prop := 
  (p2.x - p1.x) * (p3.x - p2.x) + (p2.y - p1.y) * (p3.y - p2.y) = 0

def not_convex (hex : Hexagon) : Prop :=
  -- you may need to define how to confirm that the hexagon is not convex
  sorry

def concurrent (hex : Hexagon) : Prop :=
  let line1 (p1 p2 : Point) := \lambda x, (p2.y - p1.y) / (p2.x - p1.x) * (x - p1.x) + p1.y
  let intersect := \lambda p1 p2 p3, ∃ x y, line1 p1 p3 x = line1 p2 p4 x ∧ 
    line1 p3 p5 x = line1 p4 p6 x ∧ line1 p1 p4 x = line1 p3 p6 x
  intersect hex.C1 hex.C2 hex.C3 ∧ intersect hex.C1 hex.C3 hex.C4 ∧ 
    intersect hex.C1 hex.C5 hex.C6 ∧ intersect hex.C2 hex.C4 hex.C5 ∧ 
    intersect hex.C2 hex.C5 hex.C6 ∧ intersect hex.C3 hex.C4 hex.C5 ∧ 
    intersect hex.C3 hex.C5 hex.C6


theorem Hexagon_Concurrent_Parallelity
  (hex : Hexagon)
  (h1 : perpendicular hex.C1 hex.C2 hex.C3)
  (h2 : perpendicular hex.C2 hex.C3 hex.C4)
  (h3 : perpendicular hex.C3 hex.C4 hex.C5)
  (h4 : perpendicular hex.C4 hex.C5 hex.C6)
  (h5 : perpendicular hex.C5 hex.C6 hex.C1)
  (h6 : not_convex hex) :
  concurrent hex :=
sorry

end Hexagon_Concurrent_Parallelity_l69_69817


namespace number_of_triangles_in_pentadecagon_l69_69984

open Finset

theorem number_of_triangles_in_pentadecagon :
  ∀ (n : ℕ), n = 15 → (n.choose 3 = 455) := 
by 
  intros n hn 
  rw hn
  rw Nat.choose_eq_factorial_div_factorial (show 3 ≤ 15)
  { norm_num }

-- Proof omitted with sorry

end number_of_triangles_in_pentadecagon_l69_69984


namespace number_of_triangles_in_pentadecagon_l69_69982

open Finset

theorem number_of_triangles_in_pentadecagon :
  ∀ (n : ℕ), n = 15 → (n.choose 3 = 455) := 
by 
  intros n hn 
  rw hn
  rw Nat.choose_eq_factorial_div_factorial (show 3 ≤ 15)
  { norm_num }

-- Proof omitted with sorry

end number_of_triangles_in_pentadecagon_l69_69982


namespace largest_positive_real_root_l69_69091

theorem largest_positive_real_root (b2 b1 b0 : ℤ) (h2 : |b2| ≤ 3) (h1 : |b1| ≤ 3) (h0 : |b0| ≤ 3) :
  ∃ r : ℝ, (r > 0) ∧ (r^3 + (b2 : ℝ) * r^2 + (b1 : ℝ) * r + (b0 : ℝ) = 0) ∧ 3.5 < r ∧ r < 4.0 :=
sorry

end largest_positive_real_root_l69_69091


namespace temp_on_monday_l69_69343

theorem temp_on_monday (
  sunday : ℝ := 40,
  tuesday : ℝ := 65,
  wednesday : ℝ := 36,
  thursday : ℝ := 82,
  friday : ℝ := 72,
  saturday : ℝ := 26,
  avg_temp : ℝ := 53
) : ∃ monday : ℝ, monday = 50 :=
by
  sorry

end temp_on_monday_l69_69343


namespace second_intersection_point_l69_69255

def f (x : ℝ) : ℝ := (x^2 - 8*x + 12) / (2*x - 6)
def g (x : ℝ) : ℝ := (-2*x^2 + 6*x - 9) / (x - 3)

theorem second_intersection_point :
  ∃ x y : ℝ, x ≠ -3 ∧ f x = g x ∧ x = 7/3 ∧ y = -5/2 :=
by sorry

end second_intersection_point_l69_69255


namespace rectangle_ratio_l69_69232

theorem rectangle_ratio (A B C D G H R S : Point) (h_rect : is_rectangle ABCD) 
  (h_AB : AB = 8) (h_BC : BC = 4) (h_G_on_BC : G ∈ BC) 
  (h_H_on_BC : H ∈ BC) 
  (h_BG_GH_HC : BG = GH ∧ GH = HC) 
  (h_intersect_R : intersection_line AG BD R) 
  (h_intersect_S : intersection_line AH BD S) :
  ratio BR RS SD = (2:1:1) :=
sorry

end rectangle_ratio_l69_69232


namespace triangles_in_pentadecagon_l69_69970

def regular_pentadecagon := {vertices : Finset Point | vertices.card = 15 ∧ 
  ∀ a b c ∈ vertices, ¬Collinear a b c}

theorem triangles_in_pentadecagon (P : regular_pentadecagon) : 
  (P.vertices.card.choose 3) = 455 :=
by 
  sorry


end triangles_in_pentadecagon_l69_69970


namespace smallest_rat_num_l69_69437

theorem smallest_rat_num (a b c d : ℚ) (ha : a = -6 / 7) (hb : b = 2) (hc : c = 0) (hd : d = -1) :
  min (min a (min b c)) d = -1 :=
sorry

end smallest_rat_num_l69_69437


namespace tank_volume_in_liters_l69_69421

theorem tank_volume_in_liters :
  (let side_length := 10 in
   let volume_in_cubic_meters := side_length ^ 3 in
   let volume_in_liters := volume_in_cubic_meters * 1000 in
   volume_in_liters = 1000000) :=
by
  let side_length := 10
  let volume_in_cubic_meters := side_length ^ 3
  let volume_in_liters := volume_in_cubic_meters * 1000
  have equation : volume_in_liters = 1000000 := by
    calc 
      volume_in_cubic_meters * 1000 = (10 ^ 3) * 1000 : rfl
      ... = 1000 * 1000 : by norm_num
      ... = 1000000 : by norm_num
  exact equation

end tank_volume_in_liters_l69_69421


namespace monic_quadratic_polynomial_with_root_l69_69853

theorem monic_quadratic_polynomial_with_root (x : ℝ) : 
  ∃ p : polynomial ℝ, monic p ∧ p.coeff 1 =  -4 ∧ p.coeff 0 = 13 ∧ (∀ z : ℂ, z = (2 - 3 * I) → p.eval z.re = 0) :=
sorry

end monic_quadratic_polynomial_with_root_l69_69853


namespace working_light_bulbs_l69_69309

theorem working_light_bulbs (total_lamps : ℕ) (light_bulbs_per_lamp : ℕ) (fraction_burnt_out : ℚ) (burnt_out_per_lamp : ℕ) :
  total_lamps = 20 →
  light_bulbs_per_lamp = 7 →
  fraction_burnt_out = 1 / 4 →
  burnt_out_per_lamp = 2 →
  (total_lamps * light_bulbs_per_lamp) - (total_lamps * fraction_burnt_out.numerator * burnt_out_per_lamp / fraction_burnt_out.denominator) = 130 :=
by
  intros h_total h_per_lamp h_fraction h_burnt_out
  rw [h_total, h_per_lamp, h_fraction, h_burnt_out]
  norm_num
  sorry

end working_light_bulbs_l69_69309


namespace ice_cream_vendor_l69_69441

theorem ice_cream_vendor (M : ℕ) (h3 : 50 - (3 / 5) * 50 = 20) (h4 : (2 / 3) * M = 2 * M / 3) 
  (h5 : (50 - 30) + M - (2 * M / 3) = 38) :
  M = 12 :=
by
  sorry

end ice_cream_vendor_l69_69441


namespace sequence_formula_l69_69147

theorem sequence_formula (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) = a n / (1 + a n)) : 
  ∀ n : ℕ, 0 < n → a n = 1 / n := 
by 
  sorry

end sequence_formula_l69_69147


namespace four_digit_palindromic_perfect_square_count_l69_69042

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69042


namespace solve_conversion_problem_l69_69295

def conversion_problem (NAD_to_USD : ℚ) (USD_to_CNY : ℚ) (cost_NAD : ℚ) : Prop :=
  (NAD_to_USD = 1/8) →
  (USD_to_CNY = 5) →
  (cost_NAD = 160) →
  let cost_USD := cost_NAD * NAD_to_USD in
  let cost_CNY := cost_USD * USD_to_CNY in
  cost_CNY = 100

theorem solve_conversion_problem : conversion_problem (1/8) 5 160 :=
by
  intros h1 h2 h3
  let cost_USD := 160 * (1/8)
  let cost_CNY := cost_USD * 5
  sorry

end solve_conversion_problem_l69_69295


namespace calculate_f_5_5_l69_69161

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic_condition (x : ℝ) (h₂ : 2 ≤ x ∧ x ≤ 3) : f (x + 2) = -1 / f x
axiom defined_segment (x : ℝ) (h₂ : 2 ≤ x ∧ x ≤ 3) : f x = x

theorem calculate_f_5_5 : f 5.5 = 2.5 := sorry

end calculate_f_5_5_l69_69161


namespace rook_captures_bishop_l69_69723

-- Define the 3 x 1969 board as a type (or as a grid of positions)
structure Position :=
(row : ℕ)
(col : ℕ)
(valid : row < 3 ∧ col < 1969)

-- Function to determine if a move keeps the rook within a knight's move of the bishop
def is_knights_move_away (p1 p2 : Position) : Bool :=
  ((abs (p1.row - p2.row) = 2 ∧ abs (p1.col - p2.col) = 1) ∨ 
   (abs (p1.row - p2.row) = 1 ∧ abs (p1.col - p2.col) = 2))

-- Predicate capturing the conditions - White starts, rook and bishop move alternately
inductive turn : Type
| white : turn
| black : turn

-- Initial configuration
structure InitialConfig :=
(rook_pos : Position)
(bishop_pos : Position)
(first_move : turn = turn.white)

-- Definition of the main theorem to prove
theorem rook_captures_bishop (ic : InitialConfig) :
  ∃ moves : list (Position × Position × turn),
    moves.head = (ic.rook_pos, ic.bishop_pos, turn.white) ∧
    (∀ (p : Position × Position × turn) ∈ moves, 
      p.1.valid ∧ p.2.valid ∧ (p.3 = turn.white → is_knights_move_away p.1 p.2 = true)) ∧
    moves.last.1 = moves.last.2 :=
sorry

end rook_captures_bishop_l69_69723


namespace neg_power_of_square_l69_69803

theorem neg_power_of_square (a : ℝ) : (-a^2)^3 = -a^6 :=
by sorry

end neg_power_of_square_l69_69803


namespace perpendicular_lines_l69_69581

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, x - y + 1 = 0 ∧ (x + a * y - 1 = 0 → a = 1)) :=
begin
  assume x y,
  split,
  { intro h1,
    simp [h1] },
  { assume h2 hy,
    have hSlopes : (1 : ℝ) * ( - (1 / a)) = -1,
    { sorry },
    have eq : - (1 / a) = -1,
    { sorry },
    linarith }
end

end perpendicular_lines_l69_69581


namespace goods_train_passes_in_15_seconds_l69_69786

noncomputable def woman_train_speed_kmph : ℝ := 20
noncomputable def goods_train_speed_kmph : ℝ := 51.99424046076314
noncomputable def goods_train_length_m : ℝ := 300

def relative_speed_m_s := (woman_train_speed_kmph + goods_train_speed_kmph) * (1000 / 3600)

theorem goods_train_passes_in_15_seconds : 
  (goods_train_length_m / relative_speed_m_s) = 15 := 
by
  sorry

end goods_train_passes_in_15_seconds_l69_69786


namespace normal_price_of_soup_is_correct_l69_69625

variable (totalCans : Nat) (totalPaid : Real) (effectiveCans : Nat) (pricePerCan : Real)

axiom h1 : totalCans = 30
axiom h2 : totalPaid = 9
axiom h3 : effectiveCans = totalCans / 2
axiom h4 : totalPaid / effectiveCans = pricePerCan

theorem normal_price_of_soup_is_correct : pricePerCan = 0.60 := by
  have hc : effectiveCans = 15 := by
    sorry
  have hp : pricePerCan = totalPaid / effectiveCans := by
    sorry
  show pricePerCan = 0.60 from
    calc
      pricePerCan = totalPaid / effectiveCans := by sorry
      ... = 9 / 15 := by sorry
      ... = 0.60 := by sorry

end normal_price_of_soup_is_correct_l69_69625


namespace find_point_D_l69_69604

structure Point :=
  (x : ℤ)
  (y : ℤ)

def translation_rule (A C : Point) : Point :=
{
  x := C.x - A.x,
  y := C.y - A.y
}

def translate (P delta : Point) : Point :=
{
  x := P.x + delta.x,
  y := P.y + delta.y
}

def A := Point.mk (-1) 4
def C := Point.mk 1 2
def B := Point.mk 2 1
def D := Point.mk 4 (-1)
def translation_delta : Point := translation_rule A C

theorem find_point_D : translate B translation_delta = D :=
by
  sorry

end find_point_D_l69_69604


namespace roots_of_polynomial_l69_69842

noncomputable def quadratic_polynomial (x : ℝ) := x^2 - 4 * x + 13

theorem roots_of_polynomial : ∀ (z : ℂ), z = 2 - 3 * complex.I → quadratic_polynomial (z.re) = 0 :=
by
  intro z h
  sorry

end roots_of_polynomial_l69_69842


namespace income_expenditure_ratio_l69_69329

theorem income_expenditure_ratio (I E S : ℝ) (hI : I = 10000) (hS : S = 2000) (hEq : S = I - E) : I / E = 5 / 4 :=
by {
  sorry
}

end income_expenditure_ratio_l69_69329


namespace c_should_pay_rs_45_l69_69371

theorem c_should_pay_rs_45
  (A_months : ℕ) (B_months : ℕ) (C_months : ℕ)
  (A_oxen : ℕ) (B_oxen : ℕ) (C_oxen : ℕ)
  (total_rent : ℕ)
  (hA : A_oxen = 10) (hA_months : A_months = 7)
  (hB : B_oxen = 12) (hB_months : B_months = 5)
  (hC : C_oxen = 15) (hC_months : C_months = 3)
  (hrent : total_rent = 175) :
  let A_oxen_months := A_oxen * A_months,
      B_oxen_months := B_oxen * B_months,
      C_oxen_months := C_oxen * C_months,
      total_oxen_months := A_oxen_months + B_oxen_months + C_oxen_months
  in C_oxen_months / total_oxen_months * total_rent = 45 := 
by
  sorry

end c_should_pay_rs_45_l69_69371


namespace scalar_triple_product_zero_l69_69635

-- Define vectors a, b, and c
def a : ℝ^3 := ![2, -4, Real.sqrt 3]
def b : ℝ^3 := ![-1, Real.sqrt 5, 2]
def c : ℝ^3 := ![3, 0, Real.sqrt 7]

-- Define the vector operations for subtractions and cross product
def v1 : ℝ^3 := a - b
def v2 : ℝ^3 := b - c
def v3 : ℝ^3 := c - a

-- Define the cross product and dot product
def cross_prod (u v : ℝ^3) : ℝ^3 :=
  ![u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x]

def dot_prod (u v : ℝ^3) : ℝ :=
  u.x*v.x + u.y*v.y + u.z*v.z

-- The main statement to prove
theorem scalar_triple_product_zero : dot_prod v1 (cross_prod v2 v3) = 0 :=
by
  sorry

end scalar_triple_product_zero_l69_69635


namespace fraction_of_odd_products_in_table_l69_69588

theorem fraction_of_odd_products_in_table
  : (let 
       num_elements := 121,
       odd_elements := 25,
       fraction := odd_elements / num_elements
     in Real.round (fraction * 100) / 100 = 0.21) :=
by
  sorry

end fraction_of_odd_products_in_table_l69_69588


namespace smallest_number_l69_69370

theorem smallest_number (n : ℕ) (h : n = 10) 
  (numbers : Fin n → ℕ) 
  (h_init : numbers (0 : Fin n) = 1 ∧ ∀ i, i ≠ 0 → numbers i = 0) : 
  ∃ m, m = 1 / 512 ∧ 
       (let replaced (x y : ℕ) := (x + y) / 2
        in ∀ i j, i ≠ j → replaced (numbers i) (numbers j) ≥ 2 ^ (-9)) := sorry

end smallest_number_l69_69370


namespace monic_quadratic_polynomial_with_root_l69_69858

theorem monic_quadratic_polynomial_with_root (x : ℝ) : 
  ∃ p : polynomial ℝ, monic p ∧ p.coeff 1 =  -4 ∧ p.coeff 0 = 13 ∧ (∀ z : ℂ, z = (2 - 3 * I) → p.eval z.re = 0) :=
sorry

end monic_quadratic_polynomial_with_root_l69_69858


namespace rank_matrix_sum_l69_69627

theorem rank_matrix_sum (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (h : ∀ i j, A i j = ↑i + ↑j) : Matrix.rank A = 2 := by
  sorry

end rank_matrix_sum_l69_69627


namespace triangles_from_pentadecagon_l69_69935

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end triangles_from_pentadecagon_l69_69935


namespace min_distance_ants_l69_69383

open Real

theorem min_distance_ants (points : Fin 1390 → ℝ × ℝ) :
  (∀ i j : Fin 1390, i ≠ j → dist (points i) (points j) > 0.02) → 
  (∀ i : Fin 1390, |(points i).snd| < 0.01) → 
  ∃ i j : Fin 1390, i ≠ j ∧ dist (points i) (points j) > 10 :=
by
  sorry

end min_distance_ants_l69_69383


namespace generatrix_length_l69_69700

-- Define conditions
def lateral_area (r l : ℝ) : ℝ := π * r * l
def r := 2
def lateral_area_value := 6 * π

-- Define the theorem statement
theorem generatrix_length : ∃ l : ℝ, lateral_area r l = lateral_area_value ∧ l = 3 :=
by 
  sorry

end generatrix_length_l69_69700


namespace triangles_in_pentadecagon_l69_69952

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69952


namespace hyperbola_equation_l69_69917

theorem hyperbola_equation (a b c : ℝ)
  (ha : a > 0) (hb : b > 0)
  (eccentricity : c = 2 * a)
  (distance_foci_asymptote : b = 1)
  (hyperbola_eq : c^2 = a^2 + b^2) :
  (3 * x^2 - y^2 = 1) :=
by
  sorry

end hyperbola_equation_l69_69917


namespace g_value_at_8_l69_69275

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem g_value_at_8 (g : ℝ → ℝ) (h1 : ∀ x : ℝ, g x = (1/216) * (x - (a^3)) * (x - (b^3)) * (x - (c^3))) 
  (h2 : g 0 = 1) 
  (h3 : ∀ a b c : ℝ, f (a) = 0 ∧ f (b) = 0 ∧ f (c) = 0) : 
  g 8 = 0 :=
sorry

end g_value_at_8_l69_69275


namespace two_intersection_points_l69_69090

-- Define the three lines as functions from x to y
def line1 (x : ℝ) : ℝ := (2 / 3) * x + 1
def line2 (x : ℝ) : ℝ := (2 - x) / 2
def line3 (x : ℝ) : ℝ := (2 / 3) * x + 2 / 3

-- Prove that the number of distinct intersection points of the lines is 2
theorem two_intersection_points : 
  ∃ p₁ p₂ : ℝ × ℝ, p₁ ≠ p₂ ∧
  -- p₁ and p₂ are distinct points of intersections
  (line1 p₁.1 = p₁.2 ∧ line2 p₁.1 = p₁.2 ∨ line1 p₁.1 = p₁.2 ∧ line3 p₁.1 = p₁.2 ∨ line2 p₁.1 = p₁.2 ∧ line3 p₁.1 = p₁.2) ∧
  (line1 p₂.1 = p₂.2 ∧ line2 p₂.1 = p₂.2 ∨ line1 p₂.1 = p₂.2 ∧ line3 p₂.1 = p₂.2 ∨ line2 p₂.1 = p₂.2 ∧ line3 p₂.1 = p₂.2) := by
  sorry

end two_intersection_points_l69_69090


namespace Sherry_catches_train_within_5_minutes_l69_69679

-- Defining the probabilities given in the conditions
def P_A : ℝ := 0.75  -- Probability of train arriving
def P_N : ℝ := 0.75  -- Probability of Sherry not noticing the train

-- Event that no train arrives combined with event that train arrives but not noticed
def P_not_catch_in_a_minute : ℝ := 1 - P_A + P_A * P_N

-- Generalizing to 5 minutes
def P_not_catch_in_5_minutes : ℝ := P_not_catch_in_a_minute ^ 5

-- Probability Sherry catches the train within 5 minutes
def P_C : ℝ := 1 - P_not_catch_in_5_minutes

theorem Sherry_catches_train_within_5_minutes : P_C = 1 - (13 / 16) ^ 5 := by
  sorry

end Sherry_catches_train_within_5_minutes_l69_69679


namespace four_digit_perfect_square_palindrome_count_l69_69005

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69005


namespace pentadecagon_triangle_count_l69_69959

-- Define the problem of selecting 3 vertices out of 15 to form a triangle
theorem pentadecagon_triangle_count : 
  ∃ (n : ℕ), n = nat.choose 15 3 ∧ n = 455 := 
by {
  sorry
}

end pentadecagon_triangle_count_l69_69959


namespace negation_of_forall_ge_2_l69_69187

theorem negation_of_forall_ge_2 :
  (¬ ∀ x : ℝ, x ≥ 2) = (∃ x₀ : ℝ, x₀ < 2) :=
sorry

end negation_of_forall_ge_2_l69_69187


namespace four_digit_perfect_square_palindrome_count_l69_69010

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l69_69010


namespace fifteenth_number_with_digit_sum_14_l69_69426

theorem fifteenth_number_with_digit_sum_14 : 
  ∃ n : ℕ, (∀ m < n, ∀ d ∈ digits m, d.sum = 14) → nth (list_pos_integers_with_digit_sum_14) 14 = 266 := 
by
  -- The sequence of positive integers with digit sum 14, indexed by their original position
  let list_pos_integers_with_digit_sum_14 := 
    list.filter (λ n, (digits n).sum = 14) (list.iota 1000) -- Example range; actual implementation will depend on the specific sequence generation
  
  -- The condition that the list is sorted in increasing order is implicit in the definition
  have sorted : list.sorted (<=) list_pos_integers_with_digit_sum_14 := sorry,
  -- nth is the nth element in a list, 0-indexed, so 14 is the 15th element
  show nth list_pos_integers_with_digit_sum_14 14 = 266, from sorry

end fifteenth_number_with_digit_sum_14_l69_69426


namespace distance_between_parallel_lines_l69_69729

/-- Three equally spaced parallel lines intersect a circle, creating three chords of lengths 40, 40, and 36.
    Prove that the distance between two adjacent parallel lines is 7.38. -/
theorem distance_between_parallel_lines {r : ℝ} :
  ∃ (d : ℝ), 7.38 = d ∧
  (∀ A B C D M N : ℝ, 
  A = 20 ∧
  M = 20 ∧
  B = 36 ∧
  N = 18 ∧
  16000 + 10 * (d / 2)^2 = 40 * r^2 ∧
  11664 + 81 * (3 * d / 2)^2 = 36 * r^2 ) :=
begin
  sorry
end

end distance_between_parallel_lines_l69_69729


namespace day_of_week_45_days_after_jennas_birthday_l69_69624

-- Defining the days of the week as an inductive type.
inductive DayOfWeek : Type
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday
deriving DecidableEq

open DayOfWeek

-- Defining the function to determine the day of the week after a certain number of days.
def day_after (start_day : DayOfWeek) (n : ℕ) : DayOfWeek :=
  let days := [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
  days[(days.indexOf start_day + n % 7) % 7]

-- The main theorem statement.
theorem day_of_week_45_days_after_jennas_birthday : day_after Monday 45 = Thursday :=
by
  sorry

end day_of_week_45_days_after_jennas_birthday_l69_69624


namespace num_triangles_pentadecagon_l69_69993

/--
  The number of triangles that can be formed using the vertices of a regular pentadecagon
  (a 15-sided polygon where no three vertices are collinear) is 455.
-/
theorem num_triangles_pentadecagon : ∀ (n : ℕ), n = 15 → ∃ (num_triangles : ℕ), num_triangles = Nat.choose n 3 ∧ num_triangles = 455 :=
by
  intros n hn
  use Nat.choose n 3
  split
  · rfl
  · sorry

end num_triangles_pentadecagon_l69_69993


namespace cube_painted_probability_l69_69400

theorem cube_painted_probability :
  let total_cubes := 125
  let cubes_with_3_faces := 1
  let cubes_with_no_faces := 76
  let total_ways := Nat.choose total_cubes 2
  let favorable_ways := cubes_with_3_faces * cubes_with_no_faces
  let probability := (favorable_ways : ℚ) / total_ways
  probability = (2 : ℚ) / 205 :=
by
  sorry

end cube_painted_probability_l69_69400


namespace find_a_l69_69907

def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2^x - 1 else -x^2 - 2 * x

theorem find_a (a : ℝ) (h : f a = 1) : a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l69_69907


namespace alberto_biked_more_than_bjorn_l69_69593

-- Define the distances traveled by Bjorn and Alberto after 5 hours.
def b_distance : ℝ := 75
def a_distance : ℝ := 100

-- Statement to prove the distance difference after 5 hours.
theorem alberto_biked_more_than_bjorn : a_distance - b_distance = 25 := 
by
  -- Proof is skipped, focusing only on the statement.
  sorry

end alberto_biked_more_than_bjorn_l69_69593


namespace f_value_l69_69654

def B := {x : ℚ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2}

def f (x : ℚ) : ℝ := sorry

axiom f_property : ∀ x ∈ B, f x + f (2 - (1 / x)) = Real.log (abs (x ^ 2))

theorem f_value : f 2023 = Real.log 2023 :=
by
  sorry

end f_value_l69_69654


namespace intersection_of_A_and_B_l69_69191

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

-- State the theorem about the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {2, 4} :=
  sorry

end intersection_of_A_and_B_l69_69191


namespace count_four_digit_palindrome_squares_l69_69032

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69032


namespace total_blocks_l69_69451

def initial_blocks := 2
def multiplier := 3
def father_blocks := multiplier * initial_blocks

theorem total_blocks :
  initial_blocks + father_blocks = 8 :=
by 
  -- skipping the proof with sorry
  sorry

end total_blocks_l69_69451


namespace kruh_kub_values_correct_l69_69619

-- Define КРУГ and КУБ
def КРУГ : ℕ := 1728
def КУБ : ℕ := 125

-- Proof statement
theorem kruh_kub_values_correct : ∃ x y : ℕ, x ^ 3 = КРУГ ∧ y ^ 3 = КУБ ∧
  ((∃ d1 d2 d3 d4 : ℕ, d1 ≠ d2 ∧ d2 ≠ d3 ∧ d3 ≠ d4 ∧
      КРУГ = d1 * 1000 + d2 * 100 + d3 * 10 + d4) ∧
   (∃ d5 d6 d7 : ℕ, d5 ≠ d6 ∧ d6 ≠ d7 ∧
      КУБ = d5 * 100 + d6 * 10 + d7)) :=
begin
  sorry
end

end kruh_kub_values_correct_l69_69619


namespace nathan_strawberry_plants_l69_69664

noncomputable def strawberry_plants (S : ℕ) : Prop :=
  let strawberry_baskets := 2 * S in
  let tomato_baskets := 16 in
  let total_income := (strawberry_baskets * 9) + (tomato_baskets * 6) in
  total_income = 186

theorem nathan_strawberry_plants : ∃ S : ℕ, strawberry_plants S ∧ S = 5 :=
by
  use 5
  rw [strawberry_plants]
  sorry

end nathan_strawberry_plants_l69_69664


namespace value_of_2a_minus_1_l69_69530

theorem value_of_2a_minus_1 (a : ℝ) (h : ∀ x : ℝ, (x = 2 → (3 / 2) * x - 2 * a = 0)) : 2 * a - 1 = 2 :=
sorry

end value_of_2a_minus_1_l69_69530


namespace book_arrangement_identical_l69_69200

theorem book_arrangement_identical (n m : ℕ) (hn : n = 7) (hm : m = 3) : 
  (Nat.factorial n) / (Nat.factorial m) = 840 :=
by
  rw [hn, hm]
  show (Nat.factorial 7) / (Nat.factorial 3) = 840
  rw [Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_two, Nat.factorial_zero, mul_comm 6, ←mul_assoc, mul_comm 7]
  calc
  7 * 6 * 5 * 4 * 3 * 2 * 1 / (3 * 2 * 1) = 5040 / 6 := by norm_num
  ... = 840 := by norm_num
   
end book_arrangement_identical_l69_69200


namespace developer_land_purchase_l69_69766

theorem developer_land_purchase :
  ∃ A : ℕ, (1863 * A = 9 * 828) ∧ A = 4 :=
by
  use 4
  have h : 1863 * 4 = 9 * 828 := by norm_num
  exact ⟨h, rfl⟩

end developer_land_purchase_l69_69766


namespace find_a_for_symmetric_graph_l69_69579

noncomputable def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

theorem find_a_for_symmetric_graph :
  (∃ a : ℝ, ∀ x, (f : ℝ → ℝ) = λ x, x^2 / ((2 * x + 1) * (x + a))
    ∧ is_even f) → a = -1/2 :=
begin
  sorry
end

end find_a_for_symmetric_graph_l69_69579


namespace marathon_finishers_l69_69388

-- Define the conditions
def totalParticipants : ℕ := 1250
def peopleGaveUp (F : ℕ) : ℕ := F + 124

-- Define the final statement to be proved
theorem marathon_finishers (F : ℕ) (h1 : totalParticipants = F + peopleGaveUp F) : F = 563 :=
by sorry

end marathon_finishers_l69_69388


namespace num_triangles_pentadecagon_l69_69996

/--
  The number of triangles that can be formed using the vertices of a regular pentadecagon
  (a 15-sided polygon where no three vertices are collinear) is 455.
-/
theorem num_triangles_pentadecagon : ∀ (n : ℕ), n = 15 → ∃ (num_triangles : ℕ), num_triangles = Nat.choose n 3 ∧ num_triangles = 455 :=
by
  intros n hn
  use Nat.choose n 3
  split
  · rfl
  · sorry

end num_triangles_pentadecagon_l69_69996


namespace abs_expression_value_l69_69460

theorem abs_expression_value : ∀ (π : ℝ), π = 3.14 → |π + |π - 10|| = 10 :=
by
  intro π hπ
  -- To be proved
  sorry

end abs_expression_value_l69_69460


namespace exponential_ordering_l69_69516

noncomputable def a := (0.4:ℝ)^(0.3:ℝ)
noncomputable def b := (0.3:ℝ)^(0.4:ℝ)
noncomputable def c := (0.3:ℝ)^(-0.2:ℝ)

theorem exponential_ordering : b < a ∧ a < c := by
  sorry

end exponential_ordering_l69_69516


namespace triangles_in_pentadecagon_l69_69971

def regular_pentadecagon := {vertices : Finset Point | vertices.card = 15 ∧ 
  ∀ a b c ∈ vertices, ¬Collinear a b c}

theorem triangles_in_pentadecagon (P : regular_pentadecagon) : 
  (P.vertices.card.choose 3) = 455 :=
by 
  sorry


end triangles_in_pentadecagon_l69_69971


namespace calc_f_of_f_l69_69551

def f (x : ℝ) : ℝ :=
if x > 1 then Real.log x / Real.log 2 
else (1 / 2) ^ x

theorem calc_f_of_f : f(f (-1 / 2)) = 1 / 2 := by
  sorry

end calc_f_of_f_l69_69551


namespace math_problem_l69_69690

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

def a_n (n : ℕ) : ℕ := 3 * n - 5

theorem math_problem (C5_4 : ℕ) (C6_4 : ℕ) (C7_4 : ℕ) :
  C5_4 = binomial 5 4 →
  C6_4 = binomial 6 4 →
  C7_4 = binomial 7 4 →
  C5_4 + C6_4 + C7_4 = 55 →
  ∃ n : ℕ, a_n n = 55 ∧ n = 20 :=
by
  sorry

end math_problem_l69_69690


namespace problem_1_problem_2_problem_3_l69_69057

theorem problem_1 (avg_daily_production : ℕ) (deviation_wed : ℤ) :
  avg_daily_production = 3000 →
  deviation_wed = -15 →
  avg_daily_production + deviation_wed = 2985 :=
by intros; sorry

theorem problem_2 (avg_daily_production : ℕ) (deviation_sat : ℤ) (deviation_fri : ℤ) :
  avg_daily_production = 3000 →
  deviation_sat = 68 →
  deviation_fri = -20 →
  (avg_daily_production + deviation_sat) - (avg_daily_production + deviation_fri) = 88 :=
by intros; sorry

theorem problem_3 (planned_weekly_production : ℕ) (deviations : List ℤ) :
  planned_weekly_production = 21000 →
  deviations = [35, -12, -15, 30, -20, 68, -9] →
  planned_weekly_production + deviations.sum = 21077 :=
by intros; sorry

end problem_1_problem_2_problem_3_l69_69057


namespace min_value_frac_l69_69163

theorem min_value_frac (x y : ℝ) (hx : x > 0) (hy : y > 0) (hlog : log 2 ^ x + log 8 ^ y = log 4) :
  ∃ x y, x > 0 ∧ y > 0 ∧ ((log 2 ^ x + log 8 ^ y = log 4) ∧ (∃ lower_bound, lower_bound = 2 ∧ ∀x y, x > 0 ∧ y > 0 ∧ log 2 ^ x + log 8 ^ y = log 4 → (1 / x + 1 / (3 * y)  ≥ lower_bound))):=
  
-- Note: Lower_bound is defined as 2, we are asserting that the lower bound exists and is equal to 2. 
begin
  sorry
end

end min_value_frac_l69_69163


namespace prob1_arrangements_prob2_arrangements_prob3_arrangements_l69_69866

-- Define the number of people involved
def num_boys : ℕ := 4
def num_girls : ℕ := 3

-- Define the total number of people in the line
def total_people : ℕ := num_boys + num_girls

-- Boy A constraint definition
def isBoyAAtEnd (arrangement : Fin total_people → Fin total_people) : Prop :=
  arrangement 0 = 0 ∨ arrangement (total_people - 1) = 0

-- Girl B and Girl C constraint
def isGirlBNotLeftOfGirlC (arrangement : Fin total_people → Fin total_people) : Prop :=
  ∀ i j : Fin total_people, arrangement i = 4 → arrangement j = 5 → i < j

-- Girl B and Girl C position constraint for third part of question
def girlBNotAtEndsAndGirlCNotInMiddle (arrangement : Fin total_people → Fin total_people) : Prop :=
  arrangement i ≠ 4 ∧ ¬ (arrangement (total_people / 2) = 5)

-- Calculations to prove
theorem prob1_arrangements : 
  ∃ (arrangement : Fin total_people → Fin total_people), 
  isBoyAAtEnd arrangement → fintype.card ((Fin (total_people - 1)) → Fin (total_people - 1)) = 1440 := 
sorry

theorem prob2_arrangements : 
  ∃ (arrangement : Fin total_people → Fin total_people), 
  isGirlBNotLeftOfGirlC arrangement → fintype.card ((Fin total_people) → Fin total_people) = 2520 := 
sorry

theorem prob3_arrangements : 
  ∃ (arrangement : Fin total_people → Fin total_people), 
  girlBNotAtEndsAndGirlCNotInMiddle arrangement → fintype.card ((Fin total_people) → Fin total_people) = 3120 := 
sorry

end prob1_arrangements_prob2_arrangements_prob3_arrangements_l69_69866


namespace races_needed_to_declare_winner_l69_69225

noncomputable def total_sprinters : ℕ := 275
noncomputable def sprinters_per_race : ℕ := 7
noncomputable def sprinters_advance : ℕ := 2
noncomputable def sprinters_eliminated : ℕ := 5

theorem races_needed_to_declare_winner :
  (total_sprinters - 1 + sprinters_eliminated) / sprinters_eliminated = 59 :=
by
  sorry

end races_needed_to_declare_winner_l69_69225


namespace number_of_programs_correct_l69_69452

-- Conditions definition
def solo_segments := 5
def chorus_segments := 3

noncomputable def number_of_programs : ℕ :=
  let solo_permutations := Nat.factorial solo_segments
  let available_spaces := solo_segments + 1
  let chorus_placements := Nat.choose (available_spaces - 1) chorus_segments
  solo_permutations * chorus_placements

theorem number_of_programs_correct : number_of_programs = 7200 :=
  by
    -- The proof is omitted
    sorry

end number_of_programs_correct_l69_69452


namespace zero_of_my_function_l69_69725

-- Define the function y = e^(2x) - 1
noncomputable def my_function (x : ℝ) : ℝ :=
  Real.exp (2 * x) - 1

-- Statement that the zero of the function is at x = 0
theorem zero_of_my_function : my_function 0 = 0 :=
by sorry

end zero_of_my_function_l69_69725


namespace right_triangle_coprime_d_le_two_l69_69811

theorem right_triangle_coprime_d_le_two (a b c d : ℕ) 
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a < b)
  (h₃ : b < c)
  (h₄ : Nat.coprime a b)
  (h₅ : Nat.coprime a c)
  (h₆ : Nat.coprime b c)
  (h₇ : d = c - b)
  (h₈ : d ∣ a) :
  d ≤ 2 :=
sorry

end right_triangle_coprime_d_le_two_l69_69811


namespace dot_product_of_midlines_l69_69242

variable {A B C a b c : ℝ}

theorem dot_product_of_midlines
  (h1: b * Real.cos C + c * Real.cos B = 2 * a * Real.cos A)
  (h2 : c = 2)
  (h3 : b = 5)
  (am : ℝ)
  (bn : ℝ) :
    a = Real.cos((2:Real)⁻¹ * Real.pi) →
    (am * bn = (3:ℝ))
  := by sorry

end dot_product_of_midlines_l69_69242


namespace proof_x_square_ab_a_square_l69_69573

variable {x b a : ℝ}

/-- Given that x < b < a < 0 where x, b, and a are real numbers, we need to prove x^2 > ab > a^2. -/
theorem proof_x_square_ab_a_square (hx : x < b) (hb : b < a) (ha : a < 0) :
  x^2 > ab ∧ ab > a^2 := 
by
  sorry

end proof_x_square_ab_a_square_l69_69573


namespace arithmetic_sequence_solution_l69_69546

variable (a d : ℤ)
variable (n : ℕ)

/-- Given the following conditions:
1. The sum of the first three terms of an arithmetic sequence is -3.
2. The product of the first three terms is 8,
This theorem proves that:
1. The general term formula of the sequence is 3 * n - 7.
2. The sum of the first n terms is (3 / 2) * n ^ 2 - (11 / 2) * n.
-/
theorem arithmetic_sequence_solution
  (h1 : (a - d) + a + (a + d) = -3)
  (h2 : (a - d) * a * (a + d) = 8) :
  (∃ a d : ℤ, (∀ n : ℕ, (n ≥ 1) → (3 * n - 7 = a + (n - 1) * d) ∧ (∃ S : ℕ → ℤ, S n = (3 / 2) * n ^ 2 - (11 / 2) * n))) :=
by
  sorry

end arithmetic_sequence_solution_l69_69546


namespace cone_base_radius_l69_69322

/-- A hemisphere of radius 3 rests on the base of a circular cone and is tangent to the cone's lateral surface along a circle. 
Given that the height of the cone is 9, prove that the base radius of the cone is 10.5. -/
theorem cone_base_radius
  (r_h : ℝ) (h : ℝ) (r : ℝ) 
  (hemisphere_tangent_cone : r_h = 3)
  (cone_height : h = 9)
  (tangent_circle_height : r - r_h = 3) :
  r = 10.5 := by
  sorry

end cone_base_radius_l69_69322


namespace sum_invested_l69_69060

-- Define the conditions
variables (P R : ℝ) (T : ℝ := 15) (SI1 SI2 : ℝ)
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100
def condition_1 : SI1 = simple_interest P R T := by simp [simple_interest, SI1, P, R, T]
def condition_2 : SI2 = simple_interest P (R + 7) T := by simp [simple_interest, SI2, P, R, T]
def condition_3 : SI2 = SI1 + 1500 := by simp [SI2, SI1]

-- Define the proof
theorem sum_invested (P : ℝ) (SI1 SI2 : ℝ)
  (h1 : SI1 = simple_interest P R T)
  (h2 : SI2 = simple_interest P (R + 7) T)
  (h3 : SI2 = SI1 + 1500) :
  P = 1428.57 :=
by
  sorry

end sum_invested_l69_69060


namespace analytical_expression_function_range_l69_69184

-- Definition of the function f(x) and its properties
def f (x 𝜔 𝜙 : ℝ) : ℝ := sqrt 13 * cos (𝜔 * x) * cos (𝜔 * x - 𝜙) - sin (𝜔 * x) ^ 2

-- Conditions
variable (𝜔 : ℝ) (𝜙 : ℝ) (hf : 𝜔 > 0) (h𝜙1 : 0 < 𝜙) (h𝜙2 : 𝜙 < π / 2) (htan : tan 𝜙 = 2 * sqrt 3)
variable (hperiod : ∀ x : ℝ, f x 𝜔 𝜙 = f (x + π) 𝜔 𝜙)

-- The first proof problem: the analytical expression of f(x)
theorem analytical_expression : 
  ∀ x : ℝ, f x 1 𝜙 = 2 * sin (2 * x + π / 6) :=
sorry

-- The second proof problem: the range of f(x)
theorem function_range : 
  ∀ (x : ℝ), x ∈ set.Icc (π / 12) 𝜙 → (1 / 13) ≤ f x 1 𝜙 ∧ f x 1 𝜙 ≤ 2 :=
sorry

end analytical_expression_function_range_l69_69184


namespace valid_permutations_remainder_l69_69261

def countValidPermutations : Nat :=
  let total := (Finset.range 3).sum (fun j =>
    Nat.choose 3 (j + 2) * Nat.choose 5 j * Nat.choose 7 (j + 3))
  total % 1000

theorem valid_permutations_remainder :
  countValidPermutations = 60 := 
  sorry

end valid_permutations_remainder_l69_69261


namespace lisa_age_in_2005_l69_69795

theorem lisa_age_in_2005
  (y : ℕ)
  (h1 : ∃ (y : ℕ), Lisa's age at the end of 2000 (Lisa's age in 2000 y and her grandfather's age in 2000 is 2y))
  (h2 : (2000 - y) + (2000 - 2 * y) = 3904) :
  (y + 5) = 37 :=
sorry

end lisa_age_in_2005_l69_69795


namespace tetrahedron_projection_l69_69709

theorem tetrahedron_projection (T : Type) [Tetrahedron T]
  (P1 P2 : Plane) (h1 : trapezoid_projection T P1 = 1) :
  ∀ (P2 : Plane), ¬ (square_projection T P2 = 1) :=
by
  sorry

end tetrahedron_projection_l69_69709


namespace exist_2004x2004_golden_matrix_13_not_exist_2004x2004_golden_matrix_12_l69_69860

open Matrix

-- Let A be an 2004 x 2004 matrix
noncomputable def is_golden_matrix (A : Matrix (Fin 2004) (Fin 2004) (Fin 13)) : Prop :=
  let rows := {i : Fin 2004 // True}.image (λ i => (λ j => A i j).toSet)
  let cols := {j : Fin 2004 // True}.image (λ j => (λ i => A i j).toSet)
  Set.pairwise_disjoint (rows ∪ cols)

theorem exist_2004x2004_golden_matrix_13 :
  ∃ A : Matrix (Fin 2004) (Fin 2004) (Fin 13), is_golden_matrix A :=
sorry

theorem not_exist_2004x2004_golden_matrix_12 :
  ¬ ∃ A : Matrix (Fin 2004) (Fin 2004) (Fin 12), is_golden_matrix A :=
sorry

end exist_2004x2004_golden_matrix_13_not_exist_2004x2004_golden_matrix_12_l69_69860


namespace triangles_from_pentadecagon_l69_69936

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end triangles_from_pentadecagon_l69_69936


namespace range_of_a_l69_69208

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end range_of_a_l69_69208


namespace ratio_of_a_to_b_is_half_l69_69228

axiom exist_numbers (a b : ℕ) (x y z : ℕ) :
  (x = 1.4 * a) ∧ 
  (z = 0.4 * a) ∧ 
  (b = 2 * z) ∧ 
  (1.25 * y = 2 * x) → 
  (2 * z = 1.25 * y) → 
  (2 = 1.4) → 
  (a + b = x + y) → 
  (b = 2 * a) →  
  (a / b = 1/2)
  
theorem ratio_of_a_to_b_is_half (a b x y z : ℕ) :
  (x = 1.4 * a) ∧ 
  (z = 0.4 * a) ∧ 
  (b = 2 * z) ∧ 
  (1.25 * y = 2 * x) → 
  (2 * z = 1.25 * y) → 
  (2 = 1.4) → 
  (a + b = x + y) → 
  (b = 2 * a) →  
  a / b = 1 / 2 :=
by
  sorry

end ratio_of_a_to_b_is_half_l69_69228


namespace necessary_and_sufficient_condition_l69_69136

def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y a : ℝ) : Prop := x + a * y - 2 = 0

def p (a : ℝ) : Prop := ∀ x y : ℝ, line1 x y → line2 x y a
def q (a : ℝ) : Prop := a = -1

theorem necessary_and_sufficient_condition (a : ℝ) : (p a) ↔ (q a) :=
by
  sorry

end necessary_and_sufficient_condition_l69_69136


namespace stratified_sampling_l69_69221

theorem stratified_sampling
  (total_employees : ℕ)
  (senior_staff : ℕ)
  (middle_staff : ℕ)
  (general_staff : ℕ)
  (sample_size : ℕ)
  (h_total : total_employees = 150)
  (h_senior : senior_staff = 15)
  (h_middle : middle_staff = 45)
  (h_general : general_staff = 90)
  (h_sample : sample_size = 30) :
  let senior_sample := sample_size * senior_staff / total_employees,
      middle_sample := sample_size * middle_staff / total_employees,
      general_sample := sample_size * general_staff / total_employees
  in (senior_sample, middle_sample, general_sample) = (3, 9, 18) :=
by
  sorry

end stratified_sampling_l69_69221


namespace set_intersection_complement_l69_69557

def U := {x : ℝ | x > -3}
def A := {x : ℝ | x < -2 ∨ x > 3}
def B := {x : ℝ | -1 ≤ x ∧ x ≤ 4}

theorem set_intersection_complement :
  A ∩ (U \ B) = {x : ℝ | -3 < x ∧ x < -2 ∨ x > 4} :=
by sorry

end set_intersection_complement_l69_69557


namespace sqrt_log_equality_l69_69162

noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem sqrt_log_equality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
    Real.sqrt (log4 x + 2 * log2 y) = Real.sqrt (log2 (x * y^2)) / Real.sqrt 2 :=
sorry

end sqrt_log_equality_l69_69162


namespace cuboid_length_l69_69495

-- Define the variables for the problem
def breadth : ℝ := 8
def height : ℝ := 6
def total_surface_area : ℝ := 480

-- The statement to prove the length
theorem cuboid_length : ∃ l : ℝ, l ≈ 13.71 ∧ 2 * (l * breadth + breadth * height + height * l) = total_surface_area :=
sorry

end cuboid_length_l69_69495


namespace extremum_condition_l69_69134

theorem extremum_condition (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a * x^2 + b * x + a^2)
  (h2 : f 1 = 10)
  (h3 : deriv f 1 = 0) :
  a + b = -7 :=
sorry

end extremum_condition_l69_69134


namespace problem_solution_l69_69705

noncomputable def number_of_distinct_pairs : ℕ :=
  {n : ℕ // ∃ x y : ℝ, (x = x^2 + y^2 + x) ∧ (y = 3 * x * y) ∧ n = 1}

theorem problem_solution :
  ∃! n : ℕ, number_of_distinct_pairs = n := by
  sorry

end problem_solution_l69_69705


namespace range_of_a_second_quadrant_l69_69233

theorem range_of_a_second_quadrant (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 + 2 * a * x - 4 * a * y + 5 * a^2 - 4 = 0) → x < 0 ∧ y > 0) →
  a > 2 :=
sorry

end range_of_a_second_quadrant_l69_69233


namespace find_number_l69_69109

theorem find_number (x : ℝ) (h : 54 / 2 + 3 * x = 75) : x = 16 :=
by
  sorry

end find_number_l69_69109


namespace sin_average_inequality_l69_69528

theorem sin_average_inequality (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < π) (h3 : 0 < x2) (h4 : x2 < π) :
  (sin x1 + sin x2) / 2 < sin ((x1 + x2) / 2) :=
sorry

end sin_average_inequality_l69_69528


namespace charge_difference_l69_69378

def  charge_single_room_percent_greater (P_s R_s G_s : ℝ) : ℝ :=
  P_s = 0.45 * R_s → P_s = 0.90 * G_s → 100

def charge_double_room_percent_greater (P_d R_d G_d : ℝ) : ℝ :=
  P_d = 0.70 * R_d → P_d = 0.80 * G_d → (1 / 7) * 100

theorem charge_difference (P_s R_s G_s P_d R_d G_d : ℝ) : 
  (P_s = 0.45 * R_s) →
  (P_s = 0.90 * G_s) →
  (P_d = 0.70 * R_d) →
  (P_d = 0.80 * G_d) →
  charge_single_room_percent_greater P_s R_s G_s - 
  charge_double_room_percent_greater P_d R_d G_d = 85.71 :=
by
  sorry

end charge_difference_l69_69378


namespace divisible_by_pow3_l69_69301

-- Define the digit sequence function
def num_with_digits (a n : Nat) : Nat :=
  a * ((10 ^ (3 ^ n) - 1) / 9)

-- Main theorem statement
theorem divisible_by_pow3 (a n : Nat) (h_pos : 0 < n) : (num_with_digits a n) % (3 ^ n) = 0 := 
by
  sorry

end divisible_by_pow3_l69_69301


namespace compare_points_l69_69890

def parabola (x : ℝ) : ℝ := -x^2 - 4 * x + 1

theorem compare_points (y₁ y₂ : ℝ) :
  parabola (-3) = y₁ →
  parabola (-2) = y₂ →
  y₁ < y₂ :=
by
  intros hy₁ hy₂
  sorry

end compare_points_l69_69890


namespace distance_between_lines_is_five_thirteenths_l69_69459

noncomputable def distance_between_parallel_lines : ℝ :=
  let a := (4, -2)
  let b := (3, -1)
  let d := (2, -3)
  let v := (b.1 - a.1, b.2 - a.2) -- Vector from point a to point b
  let dot_product_vd := v.1 * d.1 + v.2 * d.2
  let dot_product_dd := d.1 * d.1 + d.2 * d.2
  let p := (dot_product_vd / dot_product_dd * d.1, dot_product_vd / dot_product_dd * d.2)
  let c := (v.1 - p.1, v.2 - p.2) -- Orthogonal projection of v on the line direction
  real.sqrt (c.1 * c.1 + c.2 * c.2)

theorem distance_between_lines_is_five_thirteenths :
  distance_between_parallel_lines = 5 / 13 :=
sorry

end distance_between_lines_is_five_thirteenths_l69_69459


namespace num_four_digit_palindromic_squares_l69_69020

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69020


namespace _l69_69241

noncomputable def X := 0
noncomputable def Y := 1
noncomputable def Z := 2

noncomputable def angle_XYZ (X Y Z : ℝ) : ℝ := 90 -- Triangle XYZ where ∠X = 90°

noncomputable def length_YZ := 10 -- YZ = 10 units
noncomputable def length_XY := 6 -- XY = 6 units
noncomputable def length_XZ : ℝ := Real.sqrt (length_YZ^2 - length_XY^2) -- Pythagorean theorem to find XZ
noncomputable def cos_Z : ℝ := length_XZ / length_YZ -- cos Z = adjacent/hypotenuse

example : cos_Z = 0.8 :=
by {
  sorry
}

end _l69_69241


namespace williams_land_percentage_correct_l69_69373

-- Define the given conditions and variables.
variables 
  (T W : ℝ)
  (total_tax_paid : ℝ := 3840)
  (william_tax_paid : ℝ := 480)
  (taxable_fraction : ℝ := 0.9)

-- Define the condition that tax is a fraction of the total taxable land.
def land_fraction_tax : Prop :=
  william_tax_paid / total_tax_paid = W / T

-- Define what we need to show: Mr. William's land as a percentage of the total taxable land.
def william_land_percentage : Prop :=
  W / T = 12.5 / 100

-- The main statement: Given the conditions, the percentage of Mr. William's land.
theorem williams_land_percentage_correct (h : land_fraction_tax) : william_land_percentage :=
  sorry -- placeholder for the actual proof

end williams_land_percentage_correct_l69_69373


namespace find_a_l69_69542

-- Define the circle equation in its standard form, and define the radius and center.
def circle_center : ℝ × ℝ := (2, -3)
def circle_radius : ℝ := 5

-- The line equation
def line (a : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * x - y + a = 0

-- Length of the chord given by the problem
def chord_length : ℝ := 4 * Real.sqrt 5

-- Condition on a
axiom a_gt_neg5 (a : ℝ) : a > -5

-- Completed proof statement
theorem find_a (a : ℝ) :
  (∃ M N : ℝ × ℝ,
    (line a M.1 M.2) ∧ (line a N.1 N.2) ∧
    (M ≠ N) ∧
    (M.1^2 + M.2^2 - 4 * M.1 + 6 * M.2 - 12 = 0) ∧
    (N.1^2 + N.2^2 - 4 * N.1 + 6 * N.2 - 12 = 0) ∧ 
    (Real.dist M N = chord_length)) → 
  a = -2 :=
by sorry

end find_a_l69_69542


namespace find_a_from_coefficient_l69_69898

theorem find_a_from_coefficient :
  (∀ x : ℝ, (x + 1)^6 * (a*x - 1)^2 = 20 → a = 0 ∨ a = 5) :=
by
  sorry

end find_a_from_coefficient_l69_69898


namespace percentage_increase_l69_69585

theorem percentage_increase (A B x y : ℝ) (h1 : A / B = (5 * y^2) / (6 * x)) (h2 : 2 * x + 3 * y = 42) :  
  (B - A) / A * 100 = ((126 - 9 * y - 5 * y^2) / (5 * y^2)) * 100 :=
by
  sorry

end percentage_increase_l69_69585


namespace num_four_digit_palindromic_squares_l69_69017

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l69_69017


namespace mono_increasing_intervals_area_of_triangle_l69_69560

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (sin x, -1)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, -1 / 2)
noncomputable def f (x : ℝ) : ℝ := (prod.fst (vector_a x) + prod.fst (vector_b x)) * prod.fst (vector_a x) + 
  (prod.snd (vector_a x) + prod.snd (vector_b x)) * prod.snd (vector_a x) - 2

theorem mono_increasing_intervals (x : ℝ) :
  ∃ k : ℤ, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 ↔ monomially_increasing f x := sorry

theorem area_of_triangle (A B C : ℝ) (a b c : ℝ) (A_acute : 0 < A ∧ A < π / 2) 
  (a_eq : a = sqrt 3) (c_eq : c = 1) (f_A_eq : f A = 1) :
  let area := 1 / 2 * b * c * sin A
  in area = sqrt 3 / 2 := sorry

end mono_increasing_intervals_area_of_triangle_l69_69560


namespace eccentricity_of_ellipse_equation_of_ellipse_l69_69170

-- Define the conditions in Lean
variables (a b c : ℝ) (A B M : ℝ × ℝ)
variables (line_eq : ℝ → ℝ → Prop) (ellipse_eq : ℝ → ℝ → Prop) (midpoint_cond : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ)
variables (foci_eq : ℝ → ℝ → Prop) (unit_circle_eq : ℝ → ℝ → Prop)
variables (eccentricity : ℝ → ℝ → ℝ) (line : ℝ → ℝ)

-- Given conditions
def line_eq (x y : ℝ) : Prop := x + y - 1 = 0
def ellipse_eq (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def midpoint_cond (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def line (x : ℝ) : ℝ := (1 / 2) * x
def unit_circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Prove the eccentricity of the ellipse
theorem eccentricity_of_ellipse (h1 : ∃ A B : ℝ × ℝ, line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧ ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2) 
    (h2 : M = midpoint_cond A B) (h3 : M.2 = line M.1) : eccentricity a b = sqrt 2 / 2 := sorry

-- Prove the equation of the ellipse
theorem equation_of_ellipse (h1 : ∃ F G : ℝ × ℝ, foci_eq F.1 F.2 ∧ foci_eq G.1 G.2 ∧ 
                  (∃ x0 y0 : ℝ, foci_eq x0 y0 ∧ unit_circle_eq x0 y0)) 
    (h2 : b = 1) : ∀ x y : ℝ, (x^2 / 2 + y^2 = 1) := sorry

end eccentricity_of_ellipse_equation_of_ellipse_l69_69170


namespace product_first_2019_terms_l69_69189

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 1 then 2 else 2 - 1 / (sequence (n - 1))

theorem product_first_2019_terms :
  ∏ i in Finset.range 2020, sequence (i + 1) = 2020 :=
sorry

end product_first_2019_terms_l69_69189


namespace distance_between_parallel_lines_l69_69457

noncomputable def vector_a : ℝ × ℝ := (4, -2)
noncomputable def vector_b : ℝ × ℝ := (3, -1)
noncomputable def direction_d : ℝ × ℝ := (2, -3)

theorem distance_between_parallel_lines :
  let v := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let p := ((v.1 * direction_d.1 + v.2 * direction_d.2) / (direction_d.1 * direction_d.1 + direction_d.2 * direction_d.2) * direction_d.1,
            (v.1 * direction_d.1 + v.2 * direction_d.2) / (direction_d.1 * direction_d.1 + direction_d.2 * direction_d.2) * direction_d.2)
  let c := (vector_b.1 + p.1, vector_b.2 + p.2)
  let distance := real.sqrt ((vector_a.1 - c.1)^2 + (vector_a.2 - c.2)^2)
  distance = real.sqrt(13) / 13 :=
by
  sorry

end distance_between_parallel_lines_l69_69457


namespace max_value_of_f_l69_69098

-- Define the function f(x) = 5x - x^2
def f (x : ℝ) : ℝ := 5 * x - x^2

-- The theorem we want to prove, stating the maximum value of f(x) is 6.25
theorem max_value_of_f : ∃ x, f x = 6.25 :=
by
  -- Placeholder proof, to be completed
  sorry

end max_value_of_f_l69_69098


namespace amount_put_in_by_a_l69_69745

theorem amount_put_in_by_a :
  ∃ (x : ℝ), 
    let total_profit := 9600 in
    let b_contrib := 1500 in
    let a_management_fee := 0.1 * total_profit in
    let remaining_profit := total_profit - a_management_fee in
    let total_contrib := x + b_contrib in
    let a_share_of_remaining_profit := (x / total_contrib) * remaining_profit in
    let a_total_received := a_management_fee + a_share_of_remaining_profit in
    a_total_received = 7008 ∧ x = 3500 :=
begin
  sorry
end

end amount_put_in_by_a_l69_69745


namespace tangent_circles_eq_length_l69_69658

-- Definitions for the objects in the problem
variables {C1 C2 : Type} [circle C1] [circle C2]
variables {A B P Q P' Q' : Type} [point A] [point B] [point P] [point Q] [point P'] [point Q']
variables (tangent_P : tangent C2 A P) (tangent_Q : tangent C1 A Q)
variables (intersect_P' : intersec_other_than B (line_through P B) C2 P')
variables (intersect_Q' : intersec_other_than B (line_through Q B) C1 Q')

-- The theorem statement
theorem tangent_circles_eq_length :
  PP' = QQ' :=
sorry

end tangent_circles_eq_length_l69_69658


namespace equation_of_C_angle_constant_l69_69294

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

noncomputable def onCircle (P : ℝ × ℝ) : Prop := circle P.1 P.2

def perpendicularFoot (P : ℝ × ℝ) : ℝ × ℝ := (0, P.2)

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def G (P : ℝ × ℝ) : ℝ × ℝ := midpoint P (perpendicularFoot P)

noncomputable def curveC (G : ℝ × ℝ) : Prop := (G.2^2 / 4) + G.1^2 = 1

theorem equation_of_C (P : ℝ × ℝ) (hP : onCircle P) : curveC (G P) := sorry

noncomputable def intersects_line (l : ℝ → Prop) (O : ℝ × ℝ) (curve : ℝ × ℝ → Prop) : Prop :=
∃ M N E F : ℝ × ℝ,
  l M.1 ∧ l N.1 ∧
  circle M.1 M.2 ∧ circle N.1 N.2 ∧
  curve E ∧ curve F ∧
  M ≠ N ∧ E ≠ F ∧ 
  (abs (N.1 - M.1) = 8 * sqrt 5 / 5 ∨ abs (N.2 - M.2) = 8 * sqrt 5 / 5)

theorem angle_constant (l : ℝ → Prop) (H : intersects_line l (0, 0) curveC) :
  let O : ℝ × ℝ := (0, 0)
  let E F : ℝ × ℝ × ℝ × ℝ := sorry 
  ∃ E F : ℝ × ℝ, 
    (∃ M N : ℝ × ℝ, 
    l M.1 ∧ l N.1 ∧
    circle M.1 M.2 ∧ circle N.1 N.2 ∧
    curveC E ∧ curveC F ∧
    M ≠ N ∧ E ≠ F ∧ 
    (abs (N.1 - M.1) = 8 * sqrt 5 / 5 ∨ abs (N.2 - M.2) = 8 * sqrt 5 / 5) ∧
    let OE := (E.1 - O.1, E.2 - O.2)
    let OF := (F.1 - O.1, F.2 - O.2)
    ∠(OE, OF) = π / 2) := sorry

end equation_of_C_angle_constant_l69_69294


namespace river_depth_l69_69778

theorem river_depth (width flow_rate volume_per_minute : ℝ) (hf : flow_rate = 5000 / 60) (hw : width = 45) (hv : volume_per_minute = 7500) : ∃ d, d = 2 := 
by
  -- Define the given constants
  let flow_rate_m_min := 5000 / 60
  let width := 45
  let volume_per_minute := 7500
  
  -- Calculate depth
  let depth := volume_per_minute / (width * flow_rate_m_min)
  
  -- Show that the depth equals 2 meters
  have h : depth = 2,
  { calc 
      depth = volume_per_minute / (width * flow_rate_m_min) : by rfl
      ... = 7500 / (45 * (5000 / 60))         : by rw [hw, hv, hf]
      ... = 7500 / 3750                       : by norm_num              
      ... = 2                                 : by norm_num },
  
  -- Conclude that there exists a depth of 2 meters
  exact ⟨depth, h⟩

end river_depth_l69_69778


namespace total_value_of_coins_is_correct_l69_69349

-- Definitions for the problem conditions
def number_of_dimes : ℕ := 22
def number_of_quarters : ℕ := 10
def value_of_dime : ℝ := 0.10
def value_of_quarter : ℝ := 0.25
def total_value_of_dimes : ℝ := number_of_dimes * value_of_dime
def total_value_of_quarters : ℝ := number_of_quarters * value_of_quarter
def total_value : ℝ := total_value_of_dimes + total_value_of_quarters

-- Theorem statement
theorem total_value_of_coins_is_correct : total_value = 4.70 := sorry

end total_value_of_coins_is_correct_l69_69349


namespace flower_stones_per_bracelet_l69_69079

theorem flower_stones_per_bracelet (total_stones : ℝ) (bracelets : ℝ)  (H_total: total_stones = 88.0) (H_bracelets: bracelets = 8.0) :
  (total_stones / bracelets = 11.0) :=
by
  rw [H_total, H_bracelets]
  norm_num

end flower_stones_per_bracelet_l69_69079


namespace remainder_division_l69_69262

open Polynomial

variables (Q : Polynomial ℝ)

def condition1 : Prop := (Q % (X - 19)).eval 19 = 20 
def condition2 : Prop := (Q % (X - 15)).eval 15 = 10

theorem remainder_division (h1 : condition1 Q) (h2 : condition2 Q) :
  (Q % ((X - 15) * (X - 19))) = 2.5 * X - 27.5 := 
sorry

end remainder_division_l69_69262


namespace four_digit_palindromic_perfect_square_count_l69_69043

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69043


namespace shape_is_flat_disc_l69_69863

noncomputable def shape_desc (ρ θ φ : ℝ) : Prop :=
  ρ = c ∧ φ = π / 4

theorem shape_is_flat_disc (c : ℝ) (h_pos : 0 < c) :
  ∀ (ρ θ φ : ℝ), shape_desc ρ θ φ → 
  (∀ x y, (x, y, c * cos (π / 4)) ∈ (λ x y, (x, y, √(c^2 - x^2 - y^2))) ↔ x^2 + y^2 ≤ c^2) :=
sorry

end shape_is_flat_disc_l69_69863


namespace problem1_problem2_l69_69180

-- Definitions for the problem
def f (x : ℝ) := sorry -- Definition of f(x) is not given
def g (x : ℝ) (m : ℝ) := log(-x^2 + 2 * x + m)

def A : Set ℝ := { x | -1 < x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | -1 < x ∧ x < 3 }

-- Problem (1)
theorem problem1 : B 3 = { x | -1 < x ∧ x < 3 } → 
  A ∩ B 3ᶜ = { x | 3 ≤ x ∧ x ≤ 5 } := sorry

-- Appropriately finding complement set
noncomputable def complement (S : Set ℝ) : Set ℝ := { x | x ∉ S}

-- Problem (2)
theorem problem2 (m : ℝ) : 
  A ∩ {x | -1 < x ∧ x < 4} = {x | -1 < x ∧ x < 4} → 
  -4^2 + 2 * 4 + m = 0 → 
  m = 8 := sorry

end problem1_problem2_l69_69180


namespace coefficient_of_x4_is_one_l69_69168

theorem coefficient_of_x4_is_one (a : ℝ) : 
  (let expansion := (x+a)^2*(x-1)^3 
  in (expansion.coeff 4 = 1) → a = 2) :=
sorry

end coefficient_of_x4_is_one_l69_69168


namespace perpendicular_lines_l69_69527

theorem perpendicular_lines (a : ℝ) :
  (∃ l₁ l₂ : ℝ, 2 * l₁ + l₂ + 1 = 0 ∧ l₁ + a * l₂ + 3 = 0 ∧ 2 * l₁ + 1 * l₂ + 1 * a = 0) → a = -2 :=
by
  sorry

end perpendicular_lines_l69_69527


namespace age_proof_l69_69444

/-- Define the total number of offspring -/
def n_offspring : ℕ := 11

/-- Define the average ages of children, grandchildren, and great-grandchildren -/
def avg_child_age : ℝ := 63
def avg_grandchild_age : ℝ := 35
def avg_great_grandchild_age : ℝ := 4

/-- Calculate the sum of ages of all offspring -/
def sum_ages : ℝ := (2 * avg_child_age) + (4 * avg_grandchild_age) + (5 * avg_great_grandchild_age)

/-- Calculate the average age of all offspring -/
def avg_age_all_offspring : ℝ := sum_ages / n_offspring

/-- Define Anna's age using given condition -/
def anna_age : ℝ := 3.5 * avg_age_all_offspring

/-- Define Josef's age using conditions -/
def josef_age (five_year_avg_age : ℝ) : ℝ := 2 * five_year_avg_age - (anna_age + 5) - 5

/-- The correct ages of Anna and Josef -/
def anna_correct_age : ℝ := 91
def josef_correct_age : ℝ := 99

/-- Prove that the calculated ages match the correct ages -/
theorem age_proof 
  (five_year_avg_age : ℝ)
  (h_avg : five_year_avg_age = 100) :
  anna_age = anna_correct_age ∧ 
  josef_age five_year_avg_age = josef_correct_age :=
by
  sorry

end age_proof_l69_69444


namespace shoe_size_percentage_difference_l69_69406

theorem shoe_size_percentage_difference :
  ∀ (size8_len size15_len size17_len : ℝ)
  (h1 : size8_len = size15_len - (7 * (1 / 5)))
  (h2 : size17_len = size15_len + (2 * (1 / 5)))
  (h3 : size15_len = 10.4),
  ((size17_len - size8_len) / size8_len) * 100 = 20 := by
  intros size8_len size15_len size17_len h1 h2 h3
  sorry

end shoe_size_percentage_difference_l69_69406


namespace amanda_candies_total_l69_69430

theorem amanda_candies_total :
  let initial_candies := 7
  let given_first_time := 3
  let additional_candies := 30
  let given_second_time := 4 * given_first_time
  let remaining_after_first := initial_candies - given_first_time
  let remaining_after_second := additional_candies - given_second_time
  let total_remaining := remaining_after_first + remaining_after_second
  total_remaining = 22 :=
by
  let initial_candies := 7
  let given_first_time := 3
  let additional_candies := 30
  let given_second_time := 4 * given_first_time
  let remaining_after_first := initial_candies - given_first_time
  let remaining_after_second := additional_candies - given_second_time
  let total_remaining := remaining_after_first + remaining_after_second
  show total_remaining = 22 from
  sorry

end amanda_candies_total_l69_69430


namespace fewest_coach_handshakes_l69_69069

theorem fewest_coach_handshakes (total_handshakes : ℕ) (player_handshakes : ℕ) (coach_handshakes : ℕ) 
  (players : ℕ) (h₁ : ∑ i in finset.range(players), i = player_handshakes)
  (h₂ : player_handshakes + coach_handshakes = total_handshakes) 
  (h₃ : total_handshakes = 435) :
  coach_handshakes = 0 := 
by
  sorry

end fewest_coach_handshakes_l69_69069


namespace race_length_l69_69290

theorem race_length (cristina_speed nicky_speed : ℕ) (head_start total_time : ℕ) 
  (h1 : cristina_speed > nicky_speed) 
  (h2 : head_start = 12) (h3 : cristina_speed = 5) (h4 : nicky_speed = 3) 
  (h5 : total_time = 30) :
  let nicky_distance := nicky_speed * total_time,
      cristina_time := total_time - head_start,
      cristina_distance := cristina_speed * cristina_time in
  nicky_distance = cristina_distance 
  ∧ nicky_distance = 90 := 
by
  sorry

end race_length_l69_69290


namespace y_increase_when_x_increases_by_9_units_l69_69288

-- Given condition as a definition: when x increases by 3 units, y increases by 7 units.
def x_increases_y_increases (x_increase y_increase : ℕ) : Prop := 
  (x_increase = 3) → (y_increase = 7)

-- Stating the problem: when x increases by 9 units, y increases by how many units?
theorem y_increase_when_x_increases_by_9_units : 
  ∀ (x_increase y_increase : ℕ), x_increases_y_increases x_increase y_increase → ((x_increase * 3 = 9) → (y_increase * 3 = 21)) :=
by
  intros x_increase y_increase cond h
  sorry

end y_increase_when_x_increases_by_9_units_l69_69288


namespace count_four_digit_palindrome_squares_l69_69030

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69030


namespace dirichlet_function_properties_l69_69694

noncomputable def D (x : ℝ) : ℝ :=
if x ∈ ℚ then 1 else 0

theorem dirichlet_function_properties :
  (∀ r : ℚ, ∀ x : ℝ, D (r - x) = D (r + x)) ∧
  (∀ x : ℝ, D (D x) = D (D (-x))) ∧
  (set.range D = {0, 1}) :=
by
  sorry

end dirichlet_function_properties_l69_69694


namespace geometryville_schools_l69_69482

variable (n : ℕ) -- Number of schools

-- Each participant received a different score, and each school's team has 4 students
def total_students : ℕ := 4 * n

-- Andrea's score is the third quartile (75th percentile) among all students
def andrea_rank : ℕ := (3 * total_students + 1) / 4

-- Andrea's score is the highest on her team
-- One of Andrea's teammates is in the top 50%, two are in the bottom 50%
def teammate_ranks_valid : Prop :=
  ∃ t1 t2 t3 : ℕ,
    t1 > t2 ∧ t1 < andrea_rank ∧ -- one of her teammates top 50% (but below Andrea)
    t2 < andrea_rank ∧ t3 < andrea_rank ∧
    t2 < t3 ∧                       -- two of her teammates in the bottom 50%
    t1 < total_students / 2 ∧      -- confirming top 50% teammate's rank is valid
    t2 < total_students / 2 ∧
    t3 < total_students / 2 

-- Lean 4 theorem to prove the number of schools
theorem geometryville_schools : total_students = 4 * 3 → andrea_rank = (12 * 3 + 1) / 4 → teammate_ranks_valid n → n = 3 := 
by
  intros ht hs hr
  -- Proof steps would go here
  sorry

end geometryville_schools_l69_69482


namespace checkers_puzzle_solvable_l69_69199

-- Define the movement rules and initial condition of the puzzle
def movement_allowed (current : (ℕ × ℕ)) (dest : (ℕ × ℕ)) (board : list (ℕ × ℕ)) : Prop :=
  (current.1 = dest.1 ∧ (current.2 = dest.2 + 1 ∨ current.2 + 1 = dest.2))
  ∨ (current.2 = dest.2 ∧ (current.1 = dest.1 + 1 ∨ current.1 + 1 = dest.1))
  ∧ (dest ∉ board)

-- Define the initial position of 10 checkers
def initial_positions : list (ℕ × ℕ) :=
  [ (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), 
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5) ]

-- Define the positions of additional checkers making puzzle solvable
def additional_checker_positions : list (ℕ × ℕ) :=
  [(2, 4), (4, 2)]

-- The main theorem which needs to be proved:
theorem checkers_puzzle_solvable :
  ∃ final_positions : list (ℕ × ℕ),
  (length final_positions = 12) ∧ 
  (∀ pos ∈ initial_positions ++ additional_checker_positions, 
    ∃ final_pos ∈ final_positions, movement_allowed pos final_pos (initial_positions ++ additional_checker_positions)) :=
sorry

end checkers_puzzle_solvable_l69_69199


namespace problem1_problem2_l69_69386

-- Definitions and Lean statement for Problem 1
noncomputable def curve1 (x : ℝ) : ℝ := x / (2 * x - 1)
def point1 : ℝ × ℝ := (1, 1)
noncomputable def tangent_line1 (x y : ℝ) : Prop := x + y - 2 = 0

theorem problem1 : tangent_line1 (point1.fst) (curve1 (point1.fst)) :=
sorry -- proof goes here

-- Definitions and Lean statement for Problem 2
def parabola (x : ℝ) : ℝ := x^2
def point2 : ℝ × ℝ := (2, 3)
noncomputable def tangent_line2a (x y : ℝ) : Prop := 2 * x - y - 1 = 0
noncomputable def tangent_line2b (x y : ℝ) : Prop := 6 * x - y - 9 = 0

theorem problem2 : (tangent_line2a point2.fst point2.snd ∨ tangent_line2b point2.fst point2.snd) :=
sorry -- proof goes here

end problem1_problem2_l69_69386


namespace odd_function_and_period_l69_69697

noncomputable def y (x : ℝ) : ℝ := Math.sin x * Math.cos x

theorem odd_function_and_period : 
  (∀ x : ℝ, y (-x) = -y x) ∧ (∃ T : ℝ, T = π ∧ ∀ x : ℝ, y (x + T) = y x) :=
by sorry

end odd_function_and_period_l69_69697


namespace pentadecagon_triangle_count_l69_69941

theorem pentadecagon_triangle_count :
  ∑ k in finset.range 15, if k = 3 then nat.choose 15 3 else 0 = 455 :=
by {
  sorry
}

end pentadecagon_triangle_count_l69_69941


namespace impossible_to_obtain_one_l69_69146

theorem impossible_to_obtain_one (N : ℕ) (h : N % 3 = 0) : ¬(∃ k : ℕ, (∀ m : ℕ, (∃ q : ℕ, (N + 3 * m = 5 * q) ∧ (q = 1 → m + 1 ≤ k)))) :=
sorry

end impossible_to_obtain_one_l69_69146


namespace pentadecagon_triangle_count_l69_69947

theorem pentadecagon_triangle_count :
  ∑ k in finset.range 15, if k = 3 then nat.choose 15 3 else 0 = 455 :=
by {
  sorry
}

end pentadecagon_triangle_count_l69_69947


namespace four_digit_palindromic_perfect_square_count_l69_69045

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69045


namespace dice_product_equals_5_probability_l69_69130

theorem dice_product_equals_5_probability :
  ∀ (a b c d : ℕ), (1 ≤ a ∧ a ≤ 6) → (1 ≤ b ∧ b ≤ 6) → (1 ≤ c ∧ c ≤ 6) → (1 ≤ d ∧ d ≤ 6) →
  (let num_ways := (if (a = 5 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
                        (a = 1 ∧ b = 5 ∧ c = 1 ∧ d = 1) ∨
                        (a = 1 ∧ b = 1 ∧ c = 5 ∧ d = 1) ∨
                        (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 5) then 1 else 0) in
  num_ways * (1 / 6)^4 = 1 / 324) :=
by intros a b c d Ha Hb Hc Hd
   let num_ways := if (a = 5 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
                      (a = 1 ∧ b = 5 ∧ c = 1 ∧ d = 1) ∨
                      (a = 1 ∧ b = 1 ∧ c = 5 ∧ d = 1) ∨
                      (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 5) then 1 else 0
   have calc_ways: (if (a = 5 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
                       (a = 1 ∧ b = 5 ∧ c = 1 ∧ d = 1) ∨
                       (a = 1 ∧ b = 1 ∧ c = 5 ∧ d = 1) ∨
                       (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 5) then 1 else 0) * (1 / 6)^4 = 1 / 324 := sorry
   exact calc_ways

end dice_product_equals_5_probability_l69_69130


namespace distances_sum_in_triangle_l69_69617

variable (A B C O : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
variable (a b c P AO BO CO : ℝ)

def triangle_sides (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def triangle_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
  P = a + b + c

def point_inside_triangle (O : Type) : Prop := 
  ∃ (A B C : Type), True -- Placeholder for the actual geometric condition

def distances_to_vertices (O : Type) (AO BO CO : ℝ) : Prop := 
  AO >= 0 ∧ BO >= 0 ∧ CO >= 0

theorem distances_sum_in_triangle
  (h1 : triangle_sides a b c)
  (h2 : triangle_perimeter a b c P)
  (h3 : point_inside_triangle O)
  (h4 : distances_to_vertices O AO BO CO) :
  P / 2 < AO + BO + CO ∧ AO + BO + CO < P :=
sorry

end distances_sum_in_triangle_l69_69617


namespace max_distance_from_origin_l69_69165

-- Define the parametric equations for the curve C
def curve (θ : ℝ) : ℝ × ℝ :=
  (3 + Real.cos θ, Real.sin θ)

-- Define the distance function from the origin
def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

-- Statement for the maximum distance problem
theorem max_distance_from_origin : 
  ∃ θ : ℝ, ∀ θ : ℝ, distance_from_origin (curve θ) ≤ 4 ∧ (∃ θ_max : ℝ, distance_from_origin (curve θ_max) = 4) :=
sorry

end max_distance_from_origin_l69_69165


namespace A_eq_B_A_in_C_l69_69156

def A : Set ℕ := {0, 1}
def B : Set ℕ := {x | x ∈ A ∧ x ∈ Nat}
def C : Set (Set ℕ) := {x | x ⊆ A}

theorem A_eq_B : A = B := by
  -- proof goes here
  sorry

theorem A_in_C : A ∈ C := by
  -- proof goes here
  sorry

end A_eq_B_A_in_C_l69_69156


namespace monic_quadratic_with_real_coeffs_l69_69838

open Complex Polynomial

theorem monic_quadratic_with_real_coeffs {x : ℂ} :
  (∀ a b c : ℝ, Polynomial.monic (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) ∧ 
  (x = 2 - 3 * Complex.I ∨ x = 2 + 3 * Complex.I) → Polynomial.eval (2 - 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0) ∧
  Polynomial.eval (2 + 3 * Complex.I) (Polynomial.C a * x ^ 2 + Polynomial.C b * x + Polynomial.C c) = 0 :=
begin
  sorry
end

end monic_quadratic_with_real_coeffs_l69_69838


namespace triangle_perimeter_is_nine_l69_69340

theorem triangle_perimeter_is_nine (x : ℚ) (h1 : x > 0) 
  (h2 : ∃ α : ℝ, α = 60 ∧ (x + 2)^2 = x^2 + (2*x + 1)^2 - 2 * x * (2 * x + 1) * Real.cos (α.toRadians)) : 
  (x + (2 * x + 1) + (x + 2)) = 9 := 
sorry

end triangle_perimeter_is_nine_l69_69340


namespace two_hundredth_digit_of_fraction_5_11_l69_69354

theorem two_hundredth_digit_of_fraction_5_11 :
  (Nat.digits 10 (200 % 2 + 1)) = 5 :=
by
  sorry

end two_hundredth_digit_of_fraction_5_11_l69_69354


namespace effective_purification_optimal_purification_range_l69_69394

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 4 then Real.log2 (x + 4)
  else if x > 4 then 6 / (x - 2)
  else 0

-- Problem (1): Effective purification for m = 4
theorem effective_purification (x : ℝ) (h₁ : 0 < x ∧ x ≤ 4 ∨ x > 4) (h₂ : 4 * f x ≥ 6) :
  x ≤ 6 := by
  sorry

-- Problem (2): Optimal purification within 7 days for mass m
theorem optimal_purification_range (m : ℝ) (h₁ : ∀ x, 0 < x ∧ x ≤ 4 → 6 ≤ m * Real.log2 (x + 4) ∧ m * Real.log2 (x + 4) ≤ 18)
  (h₂ : ∀ x, 4 < x ∧ x ≤ 7 → 6 ≤ 6 * m / (x - 2) ∧ 6 * m / (x - 2) ≤ 18)
  (h₃ : 0 < 7) : 5 ≤ m ∧ m ≤ 6 := by
  sorry

end effective_purification_optimal_purification_range_l69_69394


namespace range_of_a_for_inequality_l69_69210

theorem range_of_a_for_inequality :
  {a : ℝ // ∀ (x : ℝ), a * x^2 + 2 * a * x + 1 > 0} = {a : ℝ // 0 ≤ a ∧ a < 1} :=
sorry

end range_of_a_for_inequality_l69_69210


namespace sequence_zero_if_divisible_l69_69493

theorem sequence_zero_if_divisible (x : ℕ → ℤ) 
  (h : ∀ i j : ℕ, i ≠ j → i * j ∣ x i + x j) : 
  ∀ i : ℕ, x i = 0 := 
sorry

end sequence_zero_if_divisible_l69_69493


namespace find_second_number_l69_69118

theorem find_second_number (G N: ℕ) (h1: G = 101) (h2: 4351 % G = 8) (h3: N % G = 10) : N = 4359 :=
by 
  sorry

end find_second_number_l69_69118


namespace prime_number_conditions_l69_69657

theorem prime_number_conditions (p n : ℕ) (hp_prime : nat.prime p) (hp_gt_two : 2 < p) (hn_multiple_p : ∃ k : ℕ, n = p * k) (hn_one_odd_divisor : nat.factors n = [2 ^ (nat.log n 2)]) :
  ∃ k : ℕ, k = 1 ∧ n = p * k :=
by {
  sorry
}

end prime_number_conditions_l69_69657


namespace num_triangles_pentadecagon_l69_69995

/--
  The number of triangles that can be formed using the vertices of a regular pentadecagon
  (a 15-sided polygon where no three vertices are collinear) is 455.
-/
theorem num_triangles_pentadecagon : ∀ (n : ℕ), n = 15 → ∃ (num_triangles : ℕ), num_triangles = Nat.choose n 3 ∧ num_triangles = 455 :=
by
  intros n hn
  use Nat.choose n 3
  split
  · rfl
  · sorry

end num_triangles_pentadecagon_l69_69995


namespace circle_intersects_sides_in_equal_segments_l69_69616

variables {A B : Point} {angle : Angle}

-- Define the necessary geometric entities:
def perpendicular_bisector (A B : Point) : Line := sorry
def angle_bisector (angle : Angle) : Line := sorry
def intersection_point (l1 l2 : Line) : Point := sorry
def circle (center : Point) (radius : ℝ) : Set Point := sorry

-- Define the conditions:
def circle_through_points (O A B : Point) : Prop :=
  O ∈ perpendicular_bisector A B ∧ ∃ r, circle O r A ∧ circle O r B

def equal_segments (O : Point) (angle : Angle) : Prop :=
  O ∈ angle_bisector angle

-- Combine the conditions in the proof:
theorem circle_intersects_sides_in_equal_segments (A B : Point) (angle : Angle) : ∃ O, 
  circle_through_points O A B ∧ equal_segments O angle :=
begin
  sorry
end

end circle_intersects_sides_in_equal_segments_l69_69616


namespace competition_winner_l69_69478

theorem competition_winner :
  let A_statement := (B_won ∨ C_won)
  let B_statement := ¬A_won ∧ ¬C_won
  let C_statement := C_won
  let D_statement := B_won
  let statements := [A_statement, B_statement, C_statement, D_statement]
  
  ∀ (A_won B_won C_won D_won : Prop),
    (∃ (p) (q) (p ≠ q), p ∈ statements ∧ q ∈ statements) → C_won :=
by simp [statements]; sorry

end competition_winner_l69_69478


namespace exists_positive_C_l69_69669

theorem exists_positive_C (C : ℝ) (hC : 0 < C) :
  ∀ (n : ℕ), 0 < n →
  ∀ (a : ℕ → ℝ),
  max (Icc 0 2) (λ x, ∏ j in finset.range n, |x - a j|) ≤ (C^n) * max (Icc 0 1) (λ x, ∏ j in finset.range n, |x - a j|) :=
sorry

end exists_positive_C_l69_69669


namespace triangles_in_pentadecagon_l69_69973

theorem triangles_in_pentadecagon :
  let n := 15
  in (Nat.choose n 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l69_69973


namespace generating_function_chebyshev_T_generating_function_chebyshev_U_l69_69117

noncomputable def generating_function_T (x z : ℂ) :=
  (∑ n in (Finset.range n).toFinset, (T n x) * (z ^ n))

noncomputable def generating_function_U (x z : ℂ) :=
  (∑ n in (Finset.range n).toFinset, (U n x) * (z ^ n))

theorem generating_function_chebyshev_T (x z : ℂ) :
  generating_function_T x z = (1 - x*z) / (1 - 2*x*z + z^2) :=
sorry

theorem generating_function_chebyshev_U (x z : ℂ) :
  generating_function_U x z = 1 / (1 - 2*x*z + z^2) :=
sorry

end generating_function_chebyshev_T_generating_function_chebyshev_U_l69_69117


namespace waiting_period_l69_69250

-- Variable declarations
variables (P : ℕ) (H : ℕ) (W : ℕ) (A : ℕ) (T : ℕ)
-- Condition declarations
variables (hp : P = 3) (hh : H = 5 * P) (ha : A = 3 * 7) (ht : T = 39)
-- Total time equation
variables (h_total : P + H + W + A = T)

-- Statement to prove
theorem waiting_period (hp : P = 3) (hh : H = 5 * P) (ha : A = 3 * 7) (ht : T = 39) (h_total : P + H + W + A = T) : 
  W = 3 :=
sorry

end waiting_period_l69_69250


namespace work_alone_days_l69_69391

theorem work_alone_days (A B C : ℝ) 
    (hB : B = 7) 
    (hC : C = 28/3) 
    (combined_work_time : ℝ)
    (combined_work_rate : ℝ)
    (habc_rate : (1/A + 1/B + 3/28 = combined_work_rate))
    (h_combined_time : combined_work_time = 2):
  A = 4 := 
  sorry

end work_alone_days_l69_69391


namespace sum_of_valid_y_is_zero_l69_69100

theorem sum_of_valid_y_is_zero : (∑ y in (Finset.range 10), if (36 * 10000 + y * 1000 + 25) % 6 = 0 then y else 0) = 0 :=
by
  -- y must be a single-digit number (0 <= y < 10)
  -- The number 36,y25 is represented as 36*10000 + y*1000 + 25
  -- It must be divisible by both 2 and 3 for it to be divisible by 6
  -- Since the last digit (5) is not even, it cannot be divisible by 2
  sorry

end sum_of_valid_y_is_zero_l69_69100


namespace cubic_root_equation_solution_l69_69353

theorem cubic_root_equation_solution :
  let x := (Real.cbrt (4 + Real.sqrt 80)) - (Real.cbrt (4 - Real.sqrt 80))
  in x^3 + 12 * x - 8 = 0 := by
  sorry

end cubic_root_equation_solution_l69_69353


namespace train_braking_distance_l69_69691

def braking_speed (t : ℝ) : ℝ := 10 - t + 108 / (t + 2)

theorem train_braking_distance :
  let S := ∫ t in 0..16, braking_speed t
  S = 32 + 108 * real.log 18 :=
by
  sorry

end train_braking_distance_l69_69691


namespace count_four_digit_palindrome_squares_l69_69029

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l69_69029


namespace shape_of_theta_eq_c_l69_69504

-- Definitions based on given conditions
def azimuthal_angle (rho theta phi : ℝ) : ℝ := theta

-- The main theorem we want to prove
theorem shape_of_theta_eq_c (c : ℝ) : 
  (∀ ρ φ, azimuthal_angle ρ c φ = c) → 
  (∃ a b, is_plane (λ (ρ θ φ : ℝ), θ = c) a b) :=
sorry

end shape_of_theta_eq_c_l69_69504


namespace math_problem_l69_69167

theorem math_problem (x y : ℝ) 
  (h1 : 1/5 + x + y = 1) 
  (h2 : 1/5 * 1 + 2 * x + 3 * y = 11/5) : 
  (x = 2/5) ∧ 
  (y = 2/5) ∧ 
  (1/5 + x = 3/5) ∧ 
  ((1 - 11/5)^2 * (1/5) + (2 - 11/5)^2 * (2/5) + (3 - 11/5)^2 * (2/5) = 14/25) :=
by {
  sorry
}

end math_problem_l69_69167


namespace harmonic_mean_1985th_row_l69_69722

noncomputable def triangular_array (n k : ℕ) : ℚ :=
  if k = 1 then 1 / n else triangular_array (n - 1) (k - 1) - triangular_array n k

theorem harmonic_mean_1985th_row : 
  let row_1985 := (1..1985).map (λ k, triangular_array 1985 k) in
  (1985 * (row_1985.sum / (row_1985.filter (≠ 0)).card))⁻¹ = (1 / 2 ^ 1984) := 
by
  sorry

end harmonic_mean_1985th_row_l69_69722


namespace triangles_from_pentadecagon_l69_69938

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end triangles_from_pentadecagon_l69_69938


namespace charge_and_electric_field_l69_69064

noncomputable def electric_field (Q x r k : ℝ) : ℝ :=
  k * Q * x / (x^2 + r^2)^(3/2)

noncomputable def maximum_electric_field (Q r k : ℝ) : ℝ :=
  electric_field Q (r / Real.sqrt 2) r k

theorem charge_and_electric_field (r : ℝ) (E_max : ℝ) (k : ℝ) :
  r = 0.1 ∧ E_max = 8 * 10^4 * Real.sqrt 3 ∧ k = 9 * 10^9 →
  let Q := 4 * 10^{-7} in
  maximum_electric_field Q r k = E_max ∧
  electric_field Q 0.2 r k = 6.44 * 10^4 :=
by
  intro h
  sorry

end charge_and_electric_field_l69_69064


namespace sum_arithmetic_sequence_l69_69282

open Nat

noncomputable def arithmetic_sum (a1 d n : ℕ) : ℝ :=
  (2 * a1 + (n - 1) * d) * n / 2

theorem sum_arithmetic_sequence (m n : ℕ) (h1 : m ≠ n) (h2 : m > 0) (h3 : n > 0)
    (S_m S_n : ℝ) (h4 : S_m = m / n) (h5 : S_n = n / m) 
    (a1 d : ℕ) (h6 : S_m = arithmetic_sum a1 d m) (h7 : S_n = arithmetic_sum a1 d n) 
    : arithmetic_sum a1 d (m + n) > 4 :=
by
  sorry

end sum_arithmetic_sequence_l69_69282


namespace four_digit_palindromic_perfect_square_count_l69_69036

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69036


namespace tan_alpha_plus_beta_l69_69132

theorem tan_alpha_plus_beta (α β : ℝ) (hα : sin α = -4 / 5) (hα_range : 3 * π / 2 ≤ α ∧ α ≤ 2 * π) (h : sin (α + β) / cos β = 2) : tan (α + β) = 6 / 13 :=
by
  -- proof steps would go here
  sorry

end tan_alpha_plus_beta_l69_69132


namespace positive_slope_asymptote_hyperbola_l69_69470

theorem positive_slope_asymptote_hyperbola :
  (∃ x y : ℝ, (real.sqrt ((x - 2)^2 + (y + 3)^2) - real.sqrt ((x - 8)^2 + (y + 3)^2) = 4) →
  ∃ m : ℝ, m = real.sqrt 5 / 2 :=
sorry

end positive_slope_asymptote_hyperbola_l69_69470


namespace derivative_at_zero_l69_69797

noncomputable def f : ℝ → ℝ
| x => if x = 0 then 0 else Real.arcsin (x^2 * Real.cos (1 / (9 * x))) + (2 / 3) * x

theorem derivative_at_zero : HasDerivAt f (2 / 3) 0 := sorry

end derivative_at_zero_l69_69797


namespace max_triangle_area_l69_69883

noncomputable def max_area_of_triangle (a b c S : ℝ) : ℝ := 
if h : 4 * S = a^2 - (b - c)^2 ∧ b + c = 4 then 
  2 
else
  sorry

-- The statement we want to prove
theorem max_triangle_area : ∀ (a b c S : ℝ),
  (4 * S = a^2 - (b - c)^2) →
  (b + c = 4) →
  S ≤ max_area_of_triangle a b c S ∧ max_area_of_triangle a b c S = 2 :=
by sorry

end max_triangle_area_l69_69883


namespace geometric_sequence_min_value_l69_69894

noncomputable def minimum_value (m n : ℕ) : ℚ :=
  if h : m ≠ 0 ∧ n ≠ 0 then 1 / m + 4 / n else 0

theorem geometric_sequence_min_value :
  (∀ n : ℕ, 0 < n → ∃ a_n : ℝ, a_n > 0) ∧  -- condition: all terms are positive
  (∃ (a₁ a₅ a₆ a₇ q : ℝ),
    a₇ = a₆ + 2 * a₅ ∧
    a₇ = a₁ * q ^ 6 ∧ 
    a₁ * q ^ 5 = a₁ * q ^ 4 + 2 * a₁ * q ^ 4 ∧ 
    q = 2) ∧               -- condition: a7 = a6 + 2a5 and q = 2
  (∃ (a₁ aₘ aₙ q : ℝ) (m n : ℕ),
    aₘ = a₁ * q ^ (m-1) ∧ 
    aₙ = a₁ * q ^ (n-1) ∧ 
    (sqrt (aₘ * aₙ) = 4 * a₁) ∧ 
    q = 2 ∧ 
    m + n = 6) →     -- condition: sqrt(a_m a_n) = 4a1 and m + n = 6
  minimum_value 2 4 = 3 / 2 :=                 -- question: minimum value is 3/2
by
  sorry

end geometric_sequence_min_value_l69_69894


namespace sequence_term_formula_sequence_sum_l69_69925

noncomputable def a_n : Nat → ℝ
| 1        => 2 / (2 * 1 - 1)
| (n + 1)  => 2 / (2 * (n + 1) - 1)

noncomputable def S_n (n: Nat) : ℝ :=
  (Finset.range n).sum (λ i => a_n i / (2 * (i + 1) + 1))

theorem sequence_term_formula (n : ℕ) : 
  a_n n = 2 / (2 * n - 1) :=
sorry

theorem sequence_sum (n : ℕ) :
  S_n n = 2 * n / (2 * n + 1) :=
sorry

end sequence_term_formula_sequence_sum_l69_69925


namespace amanda_candies_total_l69_69431

theorem amanda_candies_total :
  let initial_candies := 7
  let given_first_time := 3
  let additional_candies := 30
  let given_second_time := 4 * given_first_time
  let remaining_after_first := initial_candies - given_first_time
  let remaining_after_second := additional_candies - given_second_time
  let total_remaining := remaining_after_first + remaining_after_second
  total_remaining = 22 :=
by
  let initial_candies := 7
  let given_first_time := 3
  let additional_candies := 30
  let given_second_time := 4 * given_first_time
  let remaining_after_first := initial_candies - given_first_time
  let remaining_after_second := additional_candies - given_second_time
  let total_remaining := remaining_after_first + remaining_after_second
  show total_remaining = 22 from
  sorry

end amanda_candies_total_l69_69431


namespace find_a_l69_69539

-- Define the equation of the circle and the line
def circle_eq (x y : ℝ) := x^2 + y^2 - 4*x + 6*y - 12 = 0
def line_eq (x y a : ℝ) := 2*x - y + a = 0

-- Define the center and radius of the circle
def circle_center : ℝ × ℝ := (2, -3)
def circle_radius : ℝ := 5

-- Define the chord length condition
def chord_length : ℝ := 4 * Real.sqrt 5

-- Define the distance from center to the line and the condition a > -5
def distance_from_center_to_line (a : ℝ) : ℝ := abs (2 * circle_center.1 - circle_center.2 + a) / Real.sqrt (2^2 + (-1)^2)
def condition_a (a : ℝ) := a > -5

-- Total function to verify value of a
def verify_a (a : ℝ) : Prop :=
  distance_from_center_to_line a = 2 ^ Real.sqrt 5 ∧
  chord_length = 4 * Real.sqrt 5 ∧
  condition_a a

-- Lean statement to prove the equivalent math proof problem
theorem find_a : ∃ a : ℝ, verify_a a ∧ a = -2 :=
sorry

end find_a_l69_69539


namespace max_value_expression_l69_69877

noncomputable def squared_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem max_value_expression (A B C P : ℝ × ℝ) :
  let PA_squared := squared_distance P A
      PB_squared := squared_distance P B
      PC_squared := squared_distance P C
      AB_squared := squared_distance A B
      BC_squared := squared_distance B C
      CA_squared := squared_distance C A
  in (AB_squared + BC_squared + CA_squared) / (PA_squared + PB_squared + PC_squared) ≤ 3 :=
by
  sorry

end max_value_expression_l69_69877


namespace diameter_formula_l69_69401

noncomputable def diameter_of_large_part (L H d : ℝ) (hL : L > 0) (hH : H > 0) (hd : d > 0) : ℝ :=
  (L^2 + H^2 - H * d) / H

theorem diameter_formula 
  {L H d : ℝ} 
  (hL : L > 0) 
  (hH : H > 0) 
  (hd : d > 0) 
  (hD : ∃ D : ℝ, D > 2) :
  (∃ D : ℝ, D = diameter_of_large_part L H d hL hH hd) :=
begin
  use diameter_of_large_part L H d hL hH hd,
  sorry
end

end diameter_formula_l69_69401


namespace range_of_a_l69_69385

-- Define the assumptions and target proof
theorem range_of_a {f : ℝ → ℝ}
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h_condition : ∀ a : ℝ, f (2 - a) + f (4 - a) < 0)
  : ∀ a : ℝ, f (2 - a) + f (4 - a) < 0 → a < 3 :=
by
  intro a h
  sorry

end range_of_a_l69_69385


namespace sequence_count_property_l69_69931

theorem sequence_count_property :
  let ℕ := Nat
  let digits := Fin 10
  let sequences := (digits → ℕ)
  (∃ (X : sequences),
    (∀ (i : digits), i ∈ X) ∧
    (∀ (sum_digits : ℕ),
      (digits.sum_seq X) % 10 = sum_digits →
      (sum_digits ∉ X))
  ) ->
  (∃ (count : ℕ),
    count = 9^36 + 4) := by 
  sorry

end sequence_count_property_l69_69931


namespace area_PXY_one_quarter_area_ABCD_l69_69710

variables {A B C D P X Y : Type*} [affine_space A B C D P X Y]
variables (ABCD : convex_quadrilateral A B C D)
variables (X : midpoint A C)
variables (Y : midpoint B D)
variables (P : intersection AD BC)

theorem area_PXY_one_quarter_area_ABCD (area : area_measure A B C D P X Y) :
  area.pxy = 1 / 4 * area.abcd :=
sorry

end area_PXY_one_quarter_area_ABCD_l69_69710


namespace sum_of_three_numbers_is_98_l69_69716

variable (A B C : ℕ) (h_ratio1 : A = 2 * (B / 3)) (h_ratio2 : B = 30) (h_ratio3 : B = 5 * (C / 8))

theorem sum_of_three_numbers_is_98 : A + B + C = 98 := by
  sorry

end sum_of_three_numbers_is_98_l69_69716


namespace prob1_1_prob1_2_prob2_1_prob2_2_prob2_3_prob2_4_l69_69758

def U1 := {2, 3, 4}
def A1 := {4, 3}
def B1 := (∅ : Set ℕ)

noncomputable def U2 := {x : ℤ | x ≤ 4}
noncomputable def A2 := {x : ℤ | -2 < x ∧ x < 3}
noncomputable def B2 := {x : ℤ | -3 < x ∧ x ≤ 3}

theorem prob1_1 : (U1 \ A1) = {2} := by sorry
theorem prob1_2 : (U1 \ B1) = {2, 3, 4} := by sorry
theorem prob2_1 : (U2 \ A2) = {x : ℤ | x ≤ -2 ∨ 3 ≤ x ∧ x ≤ 4} := by sorry
theorem prob2_2 : (A2 ∩ B2) = {x : ℤ | -2 < x ∧ x < 3} := by sorry
theorem prob2_3 : (U2 \ (A2 ∩ B2)) = {x : ℤ | x ≤ -2 ∨ 3 ≤ x ∧ x ≤ 4} := by sorry
theorem prob2_4 : ((U2 \ A2) ∩ B2) = {x : ℤ | (-3 < x ∧ x ≤ -2) ∨ (x = 3)} := by sorry

end prob1_1_prob1_2_prob2_1_prob2_2_prob2_3_prob2_4_l69_69758


namespace expression_simplification_l69_69680

noncomputable def given_expression : ℝ :=
  1 / ((1 / (Real.sqrt 2 + 2)) + (3 / (2 * Real.sqrt 3 - 1)))

noncomputable def expected_expression : ℝ :=
  1 / (25 - 11 * Real.sqrt 2 + 6 * Real.sqrt 3)

theorem expression_simplification :
  given_expression = expected_expression :=
by
  sorry

end expression_simplification_l69_69680


namespace quirky_polynomial_product_l69_69764

theorem quirky_polynomial_product :
  (∀ Q : Polynomial Complex, 
    (∀ k : Complex, 
      (∀ a b c d : Complex, 
        (Roots Q = [a, b, c, d] ∧ d = a + b + c ∧ Q = x^4 - k * x^3 - x^2 - x - 45) →
    ∃ k_values : list Complex, k_values.length = 4 ∧ (k_values.product = 720)))) :=
sorry

end quirky_polynomial_product_l69_69764


namespace probability_overlap_80_year_lifetimes_l69_69733

theorem probability_overlap_80_year_lifetimes (total_years : ℕ) (lifespan : ℕ) (area : ℕ) :
  total_years = 300 ∧ lifespan = 80 ∧ area = 90_000 ∧ 
  (∀ x y, (abs (x - y) < 80) → (0 ≤ x ∧ x ≤ 300 ∧ 0 ≤ y ∧ y ≤ 300)) 
  → (208 / 450 = 104 / 225) :=
by
  intros
  sorry

end probability_overlap_80_year_lifetimes_l69_69733


namespace solution_set_of_quadratic_inequality_l69_69214

theorem solution_set_of_quadratic_inequality (a b : ℝ)
  (h : ∀ x : ℝ, ax^2 + bx - 2 > 0 ↔ x ∈ Set.Ioo (-4 : ℝ) 1) :
  a + b = 2 := 
sorry

end solution_set_of_quadratic_inequality_l69_69214


namespace mixture_price_pecans_cashews_l69_69564

noncomputable def price_per_pound_mixture (price_pecans price_cashews num_pounds_pecans num_pounds_cashews : ℝ) : ℝ :=
  let total_weight := num_pounds_pecans + num_pounds_cashews
  let total_cost := (price_pecans * num_pounds_pecans) + (price_cashews * num_pounds_cashews)
  total_cost / total_weight

theorem mixture_price_pecans_cashews :
  price_per_pound_mixture 5.6 3.5 1.33333333333 2 ≈ 4.34 := sorry

end mixture_price_pecans_cashews_l69_69564


namespace four_digit_palindromic_perfect_square_count_l69_69037

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l69_69037


namespace s_eq_sin_c_eq_cos_l69_69754

open Real

variables (s c : ℝ → ℝ)

-- Conditions
def s_prime := ∀ x, deriv s x = c x
def c_prime := ∀ x, deriv c x = -s x
def initial_conditions := (s 0 = 0) ∧ (c 0 = 1)

-- Theorem to prove
theorem s_eq_sin_c_eq_cos
  (h1 : s_prime s c)
  (h2 : c_prime s c)
  (h3 : initial_conditions s c) :
  (∀ x, s x = sin x) ∧ (∀ x, c x = cos x) :=
sorry

end s_eq_sin_c_eq_cos_l69_69754


namespace constant_term_is_3_l69_69323

noncomputable def constant_term_expansion : ℤ :=
  -- Define the expression
  let expr := (x^2 + 2) * (1 / x^2 - 1)^5 in
  -- Find the constant term in the expansion of expr
  sorry

theorem constant_term_is_3 : constant_term_expansion = 3 :=
  sorry

end constant_term_is_3_l69_69323


namespace sum_of_series_l69_69485

theorem sum_of_series :
  (\sum n in [2, 3, 4, 5, 6, 7],  1 / (n * (n+1))) = 3 / 8 :=
by
  sorry

end sum_of_series_l69_69485


namespace area_of_HXYZ_is_8_l69_69595

-- Definitions and conditions
def Rectangle (w h : ℝ) := {p : ℝ × ℝ // 0 ≤ p.1 ∧ p.1 ≤ w ∧ 0 ≤ p.2 ∧ p.2 ≤ h}

variables (w h : ℝ) (AH HD AE EB : ℝ)
variables E G H F : ℝ × ℝ -- Points on the rectangle sides
variables (B H X Y Z : ℝ × ℝ)
variables (A D C : ℝ × ℝ)

-- Given conditions
def AH_val : AH = 4 := sorry
def HD_val : HD = 6 := sorry
def AE_val : AE = 4 := sorry
def EB_val : EB = 5 := sorry

-- Total dimensions of the rectangle
def width := AE + EB
def height := AH + HD

-- Specific coordinates of points based on dimensions assumed
def coord_A : A = (0, 0) := sorry
def coord_B : B = (width, 0) := sorry
def coord_D : D = (0, height) := sorry
def coord_C : C = (width, height) := sorry
def coord_E : E = (AE, 0) := sorry
def coord_H : H = (width, AH) := sorry

-- Area of quadrilateral HXYZ
def area_HXYZ : ℝ := 
  let base := AE
  let height := AH
  0.5 * base * height

-- The proof goal
theorem area_of_HXYZ_is_8 
(A D C E G H F B X Y Z : ℝ × ℝ) 
(AH HD AE EB : ℝ) 
(h1 : AH = 4) 
(h2 : HD = 6) 
(h3 : AE = 4) 
(h4 : EB = 5) :
area_HXYZ AE AH = 8 := 
sorry

end area_of_HXYZ_is_8_l69_69595


namespace max_x_plus_one_over_x_l69_69347

variable (x : ℝ)
variable (ys : Fin 10 → ℝ)

-- Conditions
def sum_condition : Prop := x + (∑ i, ys i) = 102
def reciprocal_sum_condition : Prop := (1/x) + (∑ i, (1 / (ys i))) = 102

-- Question and Answer
def max_value_x_plus_recip_x : Prop := x + (1/x) ≤ 10304 / 102

theorem max_x_plus_one_over_x :
  sum_condition x ys →
  reciprocal_sum_condition x ys →
  max_value_x_plus_recip_x x := 
sorry

end max_x_plus_one_over_x_l69_69347


namespace sum_of_series_eq_three_eighths_l69_69484

theorem sum_of_series_eq_three_eighths :
  (∑ n in (range 6).map (λ i, 1 / (i + 2) / (i + 3))) = (3 / 8) :=
by 
  sorry

end sum_of_series_eq_three_eighths_l69_69484


namespace max_median_number_of_cans_l69_69667

-- Definitions based on the conditions
def total_cans_sold : ℕ := 300
def total_customers : ℕ := 120
def min_cans_per_customer : ℕ := 1

-- The maximum possible median number of cans
def max_possible_median : ℝ := 3.5

theorem max_median_number_of_cans :
  (∃ counts : list ℕ,
      counts.length = total_customers ∧
      (∀ c ∈ counts, c ≥ min_cans_per_customer) ∧
      counts.sum = total_cans_sold ∧
      list.median counts = max_possible_median) :=
sorry

end max_median_number_of_cans_l69_69667


namespace geom_seq_inverse_sum_l69_69610

theorem geom_seq_inverse_sum 
  (a_2 a_3 a_4 a_5 : ℚ) 
  (h1 : a_2 * a_5 = -3 / 4) 
  (h2 : a_2 + a_3 + a_4 + a_5 = 5 / 4) :
  1 / a_2 + 1 / a_3 + 1 / a_4 + 1 / a_5 = -4 / 3 :=
sorry

end geom_seq_inverse_sum_l69_69610


namespace alpha_value_given_equal_magnitudes_given_dot_product_find_expression_value_l69_69536

-- Definitions based on the given conditions
def pointA := (3 : ℝ, 0 : ℝ)
def pointB := (0 : ℝ, 3 : ℝ)
def pointC (α : ℝ) := (Real.cos α, Real.sin α)

def vectorAC (α : ℝ) := (Real.cos α - 3, Real.sin α)
def vectorBC (α : ℝ) := (Real.cos α, Real.sin α - 3)

-- Proof 1: Given |AC| = |BC|, find α
theorem alpha_value_given_equal_magnitudes (α : ℝ) (k : ℤ) :
  Real.sqrt ((vectorAC α).fst ^ 2 + (vectorAC α).snd ^ 2) = 
  Real.sqrt ((vectorBC α).fst ^ 2 + (vectorBC α).snd ^ 2) → 
  α = k * Real.pi + Real.pi / 4 := sorry

-- Proof 2: Given (AC · BC) = -1, find the value of the given expression
theorem given_dot_product_find_expression_value (α : ℝ) :
  (vectorAC α).fst * (vectorBC α).fst + (vectorAC α).snd * (vectorBC α).snd = -1 →
  (2 * Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α) / 
  (1 + Real.tan α) = -5 / 9 := sorry

end alpha_value_given_equal_magnitudes_given_dot_product_find_expression_value_l69_69536


namespace smallest_positive_integer_div_conditions_l69_69363

theorem smallest_positive_integer_div_conditions :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 2 ∧ x % 7 = 3 ∧ ∀ y : ℕ, (y > 0 ∧ y % 5 = 2 ∧ y % 7 = 3) → x ≤ y :=
  sorry

end smallest_positive_integer_div_conditions_l69_69363


namespace least_n_condition_l69_69359

theorem least_n_condition (n : ℕ) (h1 : ∀ k : ℕ, 1 ≤ k → k ≤ n + 1 → (k ∣ n * (n - 1) → k ≠ n + 1)) : n = 4 :=
sorry

end least_n_condition_l69_69359


namespace moles_of_C2H6_l69_69500

-- Define the reactive coefficients
def ratio_C := 2
def ratio_H2 := 3
def ratio_C2H6 := 1

-- Given conditions
def moles_C := 6
def moles_H2 := 9

-- Function to calculate moles of C2H6 formed
def moles_C2H6_formed (m_C : ℕ) (m_H2 : ℕ) : ℕ :=
  min (m_C * ratio_C2H6 / ratio_C) (m_H2 * ratio_C2H6 / ratio_H2)

-- Theorem statement: the number of moles of C2H6 formed is 3
theorem moles_of_C2H6 : moles_C2H6_formed moles_C moles_H2 = 3 :=
by {
  -- Sorry is used since we are not providing the proof here
  sorry
}

end moles_of_C2H6_l69_69500


namespace range_a_of_extreme_points_f_l69_69909

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - 2 * a * x)

theorem range_a_of_extreme_points :
  (∃ x y : ℝ, x ≠ y ∧ (0 < x) ∧ (0 < y) ∧ f'.f x a = 0 ∧ f'.f y a = 0) →
  (a ∈ Ioo 0 (1/4)) :=
by
  sorry

theorem f'.f (x a : ℝ) : ℝ := Real.log x + 1 - 4 * a * x

end range_a_of_extreme_points_f_l69_69909


namespace equalize_table_l69_69598

-- Define an initial configuration and rook's move
def initial_configuration (n : ℕ) : matrix (fin n) (fin n) ℕ :=
  λ i j, if i = j then 1 else 0

def rook_transform (m : matrix (fin n) (fin n) ℕ) (path : list (fin n × fin n)) : matrix (fin n) (fin n) ℕ :=
  ∀ x ∈ path, m x.fst x.snd = m x.fst x.snd + 1

-- The main proof problem in Lean
theorem equalize_table (n : ℕ) :
  (∃ k : ℕ, ∀ i j : fin n, matrix (fin n) (fin n) ℕ i j) ↔ (∃ transforms : list (matrix (fin n) (fin n) ℕ), all_equalised (n : ℕ) transforms) :=
begin
  sorry
end

end equalize_table_l69_69598


namespace num_triangles_pentadecagon_l69_69994

/--
  The number of triangles that can be formed using the vertices of a regular pentadecagon
  (a 15-sided polygon where no three vertices are collinear) is 455.
-/
theorem num_triangles_pentadecagon : ∀ (n : ℕ), n = 15 → ∃ (num_triangles : ℕ), num_triangles = Nat.choose n 3 ∧ num_triangles = 455 :=
by
  intros n hn
  use Nat.choose n 3
  split
  · rfl
  · sorry

end num_triangles_pentadecagon_l69_69994


namespace pentadecagon_triangle_count_l69_69961

-- Define the problem of selecting 3 vertices out of 15 to form a triangle
theorem pentadecagon_triangle_count : 
  ∃ (n : ℕ), n = nat.choose 15 3 ∧ n = 455 := 
by {
  sorry
}

end pentadecagon_triangle_count_l69_69961


namespace point_not_in_first_quadrant_l69_69153

theorem point_not_in_first_quadrant (m n : ℝ) (h : m * n ≤ 0) : ¬ (m > 0 ∧ n > 0) :=
sorry

end point_not_in_first_quadrant_l69_69153


namespace new_light_wattage_l69_69404

-- Define the original wattage of the light
def original_wattage : ℕ := 80

-- Define the percentage increase
def percentage_increase : ℝ := 0.25

-- Statement to prove the wattage of the new light
theorem new_light_wattage :
  let new_wattage := original_wattage + (percentage_increase * original_wattage)
  in new_wattage = 100 :=
by
  sorry

end new_light_wattage_l69_69404


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l69_69645

variable (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_52 : (Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52) :=
by
  sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l69_69645


namespace bowling_tournament_prize_orders_l69_69796

/-- In a professional bowling tournament, the number of possible prize orders for bowlers #1 through #5, given the conditions of the playoff, is 16. --/
theorem bowling_tournament_prize_orders : 
  let possible_outcomes := 2 * 2 * 2 * 2
  in possible_outcomes = 16 := 
by
  sorry

end bowling_tournament_prize_orders_l69_69796


namespace eval_operation_l69_69708

def operation (a b : ℝ) : ℝ := a + (4 * a) / (3 * b)

theorem eval_operation : operation 10 4 = 13 + 1 / 3 :=
by
  sorry

end eval_operation_l69_69708


namespace diamond_fifteen_two_l69_69707

def diamond (a b : ℤ) : ℤ := a + (a / (b + 1))

theorem diamond_fifteen_two : diamond 15 2 = 20 := 
by 
    sorry

end diamond_fifteen_two_l69_69707


namespace trailing_zeros_2006_fact_l69_69334

def count_factors (n : ℕ) (p : ℕ) : ℕ :=
  (List.range (n+1)).map (λ k, if k > 0 then n / p^k else 0).sum

def num_trailing_zeros (n : ℕ) : ℕ :=
  count_factors n 5

theorem trailing_zeros_2006_fact : num_trailing_zeros 2006 = 500 :=
  by sorry

end trailing_zeros_2006_fact_l69_69334


namespace additional_people_l69_69719

def cars := 2
def vans := 3
def people_per_car := 5
def people_per_van := 3
def max_capacity_per_car := 6
def max_capacity_per_van := 8

theorem additional_people :
  let total_max_capacity := cars * max_capacity_per_car + vans * max_capacity_per_van in
  let total_actual_people := cars * people_per_car + vans * people_per_van in
  total_max_capacity - total_actual_people = 17 :=
by
sory 

end additional_people_l69_69719


namespace find_n_that_makes_expr_equal_digits_l69_69113

theorem find_n_that_makes_expr_equal_digits :
  ∃ (n : ℤ), 12 * n^2 + 12 * n + 11 ∈ {1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999} ↔ n = 21 ∨ n = -22 :=
by
  sorry

end find_n_that_makes_expr_equal_digits_l69_69113


namespace gyurka_decision_false_l69_69338

noncomputable def initial_balance : ℝ := 1700
noncomputable def monthly_interest_rate : ℝ := 0.02
noncomputable def tram_pass_cost : ℝ := 465

def remaining_balance (balance : ℝ) := (balance - tram_pass_cost) * (1 + monthly_interest_rate)

theorem gyurka_decision_false :
  let balance_after_1_month := remaining_balance initial_balance,
      balance_after_2_months := remaining_balance balance_after_1_month,
      balance_after_3_months := remaining_balance balance_after_2_months,
      balance_after_4_months := (balance_after_3_months - tram_pass_cost) * (1 + monthly_interest_rate)
  in balance_after_4_months < tram_pass_cost :=
by
  -- proof goes here
  sorry

end gyurka_decision_false_l69_69338


namespace geometric_sequence_general_formula_sum_first_n_terms_na_n_l69_69144

-- Conditions
variables {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom geom_seq (n : ℕ) : S n = 2 * a n - 2
axiom a1 : a 1 = 2

-- Proof of general formula for the sequence {a_n}
theorem geometric_sequence_general_formula (n : ℕ) :
  (∀ n, a n = 2 ^ n) :=
sorry

-- Sum of the first n terms of the sequence {n a_n}
theorem sum_first_n_terms_na_n (n : ℕ) :
  (∀ n, (∑ k in finset.range (n + 1), k * a k) = (n-1) * 2 ^ (n + 1) + 2) :=
sorry

end geometric_sequence_general_formula_sum_first_n_terms_na_n_l69_69144


namespace circle_sector_area_l69_69377

/-- 
Given a circle with radius 12 meters and a central angle of 39 degrees, 
prove that the area of the sector is approximately 48.9432 square meters.
-/
theorem circle_sector_area (r : ℝ) (θ : ℝ) (h_r : r = 12) (h_θ : θ = 39) :
  (θ / 360) * Real.pi * r^2 ≈ 48.9432 :=
by
  sorry

end circle_sector_area_l69_69377


namespace additional_people_l69_69718

def cars := 2
def vans := 3
def people_per_car := 5
def people_per_van := 3
def max_capacity_per_car := 6
def max_capacity_per_van := 8

theorem additional_people :
  let total_max_capacity := cars * max_capacity_per_car + vans * max_capacity_per_van in
  let total_actual_people := cars * people_per_car + vans * people_per_van in
  total_max_capacity - total_actual_people = 17 :=
by
sory 

end additional_people_l69_69718


namespace total_doors_for_apartments_l69_69765

theorem total_doors_for_apartments :
  let b1_floors := 15 in
  let b1_apartments := 5 in
  let b1_doors := 8 in
  let b2_floors := 25 in
  let b2_apartments := 6 in
  let b2_doors := 10 in
  let b3_floors := 20 in
  let b3_odd_apartments := 7 in
  let b3_even_apartments := 5 in
  let b3_doors := 9 in
  let b4_floors := 10 in
  let b4_odd_apartments := 8 in
  let b4_even_apartments := 4 in
  let b4_doors := 7 in
  b1_floors * b1_apartments * b1_doors +
  b2_floors * b2_apartments * b2_doors +
  (b3_floors / 2) * b3_odd_apartments * b3_doors +
  (b3_floors / 2) * b3_even_apartments * b3_doors +
  (b4_floors / 2) * b4_odd_apartments * b4_doors +
  (b4_floors / 2) * b4_even_apartments * b4_doors = 3600 :=
by
  -- Let b1, b2, b3_odd, b3_even, b4_odd, and b4_even denote the number of doors for each case.
  let b1_doors_needed := 15 * 5 * 8
  let b2_doors_needed := 25 * 6 * 10
  let b3_doors_odd_needed := 10 * 7 * 9
  let b3_doors_even_needed := 10 * 5 * 9
  let b4_doors_odd_needed := 5 * 8 * 7
  let b4_doors_even_needed := 5 * 4 * 7
  have b3_doors_needed := b3_doors_odd_needed + b3_doors_even_needed
  have b4_doors_needed := b4_doors_odd_needed + b4_doors_even_needed
  exact b1_doors_needed + b2_doors_needed + b3_doors_needed + b4_doors_needed = 3600

end total_doors_for_apartments_l69_69765


namespace find_circle_equation_l69_69895

noncomputable def circle_standard_eq (h : ℝ → ℂ) := sorry

structure Point :=
  (x : ℝ)
  (y : ℝ)

def circle_passes_through (center : Point) (radius : ℝ) (p : Point) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

def center_on_line (center : Point) : Prop :=
  center.x - center.y + 1 = 0

def circle_standard_form (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 2)^2 = 25

theorem find_circle_equation : ∃ center radius,
  circle_passes_through center radius ⟨0, -6⟩ ∧
  circle_passes_through center radius ⟨1, -5⟩ ∧
  center_on_line center ∧
  circle_standard_form center radius _ _ :=
sorry

end find_circle_equation_l69_69895


namespace OP_perp_EF_l69_69876

theorem OP_perp_EF
  (A O B E F E1 E2 F1 F2 P : Point)
  (h1 : angle BOF = angle AOE)
  (h2 : Foot E O A = E1)
  (h3 : Foot E O B = E2)
  (h4 : Foot F O A = F1)
  (h5 : Foot F O B = F2)
  (h6 : Intersection (Line E1 E2) (Line F1 F2) = P)
  : perpendicular (Line O P) (Line E F) :=
begin
  sorry
end

end OP_perp_EF_l69_69876


namespace maximum_value_among_ratios_of_S_over_a_l69_69884

variable {a n : ℕ} (aSeq : ℕ → ℝ) (S : ℕ → ℝ)
variable (hS15_pos : S 15 > 0) (hS16_neg : S 16 < 0)

def max_value_is_S8_over_a8 : Prop :=
  ∀ n, (1 ≤ n ∧ n ≤ 15) →
    (S 8 / aSeq 8) ≥ (S n / aSeq n)

theorem maximum_value_among_ratios_of_S_over_a :
  max_value_is_S8_over_a8 aSeq S hS15_pos hS16_neg :=
sorry

end maximum_value_among_ratios_of_S_over_a_l69_69884


namespace larger_triangle_perimeter_l69_69053

theorem larger_triangle_perimeter 
    (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (h1 : a = 6) (h2 : b = 8)
    (hypo_large : ∀ c : ℝ, c = 20) : 
    (2 * a + 2 * b + 20 = 48) :=
by {
  sorry
}

end larger_triangle_perimeter_l69_69053


namespace triangles_from_pentadecagon_l69_69940

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end triangles_from_pentadecagon_l69_69940


namespace S9_l69_69878

-- Definitions and conditions
def geometric_sum (a₁ r : ℝ) (n : ℕ) : ℝ := a₁ * (1 - r^n) / (1 - r)

variables (a₁ r : ℝ)
variables (S : ℕ → ℝ) -- S is the sum of first n terms

axiom S3 : S 3 = 10
axiom S6 : S 6 = 20
axiom Sn_geometric : ∀ n, S n = geometric_sum a₁ r n

-- The theorem to prove
theorem S9 : S 9 = 30 :=
by
  sorry

end S9_l69_69878


namespace evaluate_complex_expression_l69_69106

noncomputable def i : ℂ := Complex.I

theorem evaluate_complex_expression :
  abs((5 * Real.sqrt 2 - 5 * (i ^ 3)) * (Real.sqrt 5 + 5 * (i ^ 2))) = 15 * Real.sqrt 2 :=
by
  have h1 : i ^ 2 = -1 := by exact Complex.i_sq
  have h2 : i ^ 3 = -i := by ring_exp
  sorry

end evaluate_complex_expression_l69_69106


namespace smallest_y_of_arithmetic_sequence_l69_69653

theorem smallest_y_of_arithmetic_sequence
  (x y z d : ℝ)
  (h_arith_series_x : x = y - d)
  (h_arith_series_z : z = y + d)
  (h_positive_x : x > 0)
  (h_positive_y : y > 0)
  (h_positive_z : z > 0)
  (h_product : x * y * z = 216) : y = 6 :=
sorry

end smallest_y_of_arithmetic_sequence_l69_69653


namespace place_value_ratio_l69_69612

def number : ℝ := 90347.6208
def place_value_0 : ℝ := 10000 -- tens of thousands
def place_value_6 : ℝ := 0.1 -- tenths

theorem place_value_ratio : 
  place_value_0 / place_value_6 = 100000 := by 
    sorry

end place_value_ratio_l69_69612


namespace monic_quadratic_polynomial_with_root_l69_69856

theorem monic_quadratic_polynomial_with_root (x : ℝ) : 
  ∃ p : polynomial ℝ, monic p ∧ p.coeff 1 =  -4 ∧ p.coeff 0 = 13 ∧ (∀ z : ℂ, z = (2 - 3 * I) → p.eval z.re = 0) :=
sorry

end monic_quadratic_polynomial_with_root_l69_69856


namespace odd_function_behavior_on_interval_l69_69213

theorem odd_function_behavior_on_interval
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 4 → f x₁ < f x₂)
  (h_max : ∀ x, 1 ≤ x → x ≤ 4 → f x ≤ 5) :
  (∀ x, -4 ≤ x → x ≤ -1 → f (-4) ≤ f x ∧ f x ≤ f (-1)) ∧ f (-4) = -5 :=
sorry

end odd_function_behavior_on_interval_l69_69213


namespace ratio_brownies_to_cookies_l69_69448

-- Conditions and definitions
def total_items : ℕ := 104
def cookies_sold : ℕ := 48
def brownies_sold : ℕ := total_items - cookies_sold

-- Problem statement
theorem ratio_brownies_to_cookies : (brownies_sold : ℕ) / (Nat.gcd brownies_sold cookies_sold) = 7 ∧ (cookies_sold : ℕ) / (Nat.gcd brownies_sold cookies_sold) = 6 :=
by
  sorry

end ratio_brownies_to_cookies_l69_69448


namespace inverse_expression_l69_69203

-- Given x, y, and 3x + 4y are not zero, prove that
-- (3x + 4y)^(-1) * ((3x)^(-1) + (4y)^(-1)) = 1 / (12 * x * y).

theorem inverse_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : 3 * x + 4 * y ≠ 0) :
  (3 * x + 4 * y)^(-1) * ((3 * x)^(-1) + (4 * y)^(-1)) = 1 / (12 * x * y) :=
sorry

end inverse_expression_l69_69203


namespace pentadecagon_triangle_count_l69_69946

theorem pentadecagon_triangle_count :
  ∑ k in finset.range 15, if k = 3 then nat.choose 15 3 else 0 = 455 :=
by {
  sorry
}

end pentadecagon_triangle_count_l69_69946


namespace hulk_jump_kilometer_l69_69685

theorem hulk_jump_kilometer (n : ℕ) (h : ∀ n : ℕ, n ≥ 1 → (2^(n-1) : ℕ) ≤ 1000 → n-1 < 10) : n = 11 :=
by
  sorry

end hulk_jump_kilometer_l69_69685


namespace find_coefficients_sum_l69_69566

theorem find_coefficients_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 * x - 3)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  intro h
  sorry

end find_coefficients_sum_l69_69566


namespace find_m_n_l69_69492

theorem find_m_n : ∃ (m n : ℕ), m > n ∧ m^3 - n^3 = 999 ∧ ((m = 10 ∧ n = 1) ∨ (m = 12 ∧ n = 9)) :=
by
  sorry

end find_m_n_l69_69492
