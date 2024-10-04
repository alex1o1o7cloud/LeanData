import Data.Real.Basic
import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Parity
import Mathlib.Analysis.Distances
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.Real
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Composition
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Modeq
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.Determinant
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Real

namespace g_is_zero_l368_368812

noncomputable def g (x : Real) : Real := 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) - 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2)

theorem g_is_zero : ∀ x : Real, g x = 0 := by
  sorry

end g_is_zero_l368_368812


namespace buffy_sailing_hours_l368_368646

-- Definitions based on conditions
def speed_with_two_sails := 50 -- knots
def speed_with_one_sail := 25 -- knots
def nautical_mile_in_land_miles := 1.15
def total_distance_in_land_miles := 345

-- Proof statement
theorem buffy_sailing_hours :
  let h := (total_distance_in_land_miles / nautical_mile_in_land_miles) / (speed_with_two_sails + speed_with_one_sail) in
  h = 4 :=
by
  sorry

end buffy_sailing_hours_l368_368646


namespace measurement_accuracy_same_l368_368423

noncomputable def x_values : List ℝ := [9.6, 10.0, 9.8, 10.2, 10.6]
noncomputable def y_values : List ℝ := [10.4, 9.7, 10.0, 10.3]

def transformation (x : ℝ) : ℝ := 10 * x - 100

noncomputable def u_values : List ℝ := x_values.map transformation
noncomputable def v_values : List ℝ := y_values.map transformation

noncomputable def sample_variance (values : List ℝ) : ℝ :=
  let n := values.length
  let sum_values := values.sum
  let sum_squares := values.map (λ x => x ^ 2).sum
  (sum_squares - (sum_values ^ 2 / n)) / (n - 1)

noncomputable def su_squared : ℝ := sample_variance u_values
noncomputable def sv_squared : ℝ := sample_variance v_values

noncomputable def f_statistic : ℝ := su_squared / sv_squared

theorem measurement_accuracy_same :
  f_statistic < 9.12 :=
by
  -- sorry indicates where the proof should follow
  sorry

end measurement_accuracy_same_l368_368423


namespace root_nature_l368_368068

theorem root_nature (a : ℝ) (h : a < -1) :
  let A := a^3 + 1
      B := a^2 + 1
      C := -(a + 1)
  in (A * (A * B^2 - C * 4) > 0 → 
      (∃ x1 x2 : ℝ, A * x1^2 + B * x1 + C = 0 ∧ A * x2^2 + B * x2 + C = 0 ∧ 
      x1 * x2 < 0 ∧ x1 + x2 > 0 ∧ abs x1 < abs x2)) := 
begin
  sorry
end

end root_nature_l368_368068


namespace parabola_is_x_squared_equals_8y_l368_368347

noncomputable def parabola_equation (p : ℝ) : ℝ → ℝ → Prop := 
  λ x y, x^2 = 2 * p * y

theorem parabola_is_x_squared_equals_8y
  (p : ℝ) 
  (hp : p > 0)
  (h_focus_distance : ∀ P : ℝ × ℝ, P.1^2 = 2 * p * P.2 → (dist (P, (0, p / 2)) = 8))
  (h_x_axis_distance : ∀ P : ℝ × ℝ, P.1^2 = 2 * p * P.2 → P.2 = 6) :
  parabola_equation 4 :=
begin
  sorry
end

end parabola_is_x_squared_equals_8y_l368_368347


namespace exists_n_perfect_square_l368_368159

section
variables {R : Type*} [LinearOrderedField R]

def op (x y : R) : R := (x * y + 4) / (x + y)

def T (n : ℕ) (h : n ≥ 4) : R :=
  (list.range (n - 2)).map (λ i, (i + 3 : R)).foldl op 3

theorem exists_n_perfect_square : ∃ (n : ℕ), n ≥ 4 ∧ 
  (96 / (T n (by linarith) - 2)) ∈ ({x : R | ∃ m : ℕ, x = m ^ 2}) :=
sorry
end

end exists_n_perfect_square_l368_368159


namespace pascal_triangle_sum_first_30_rows_l368_368290

theorem pascal_triangle_sum_first_30_rows :
  (Finset.range 30).sum (λ n, n + 1) = 465 :=
begin
  sorry
end

end pascal_triangle_sum_first_30_rows_l368_368290


namespace sphere_surface_area_l368_368089

theorem sphere_surface_area (side_length : ℝ) (h : side_length = 2) :
  let radius := (side_length * Real.sqrt 2) / 2 in
  4 * Real.pi * radius^2 = 8 * Real.pi :=
by
  sorry

end sphere_surface_area_l368_368089


namespace area_of_part_of_circle_contained_in_triangle_l368_368880

theorem area_of_part_of_circle_contained_in_triangle
  (a : ℝ)
  (h45 : real.angle)
  (h15 : real.angle)
  (h_a : h45 = real.angle.of_deg 45)
  (h_b : h15 = real.angle.of_deg 15)
  : 
  let h := (a * (real.sqrt 3 - 1)) / (2 * real.sqrt 3)
  in  ((1 / 3 : ℝ) * real.pi * (h * h) = (real.pi * (a * a) * (2 - real.sqrt 3)) / 18) := 
sorry

end area_of_part_of_circle_contained_in_triangle_l368_368880


namespace median_and_mode_of_shooting_data_l368_368519

theorem median_and_mode_of_shooting_data :
  let data := [32, 31, 16, 16, 14, 12] in
  let mode := 16 in
  let median := 16 in
  List.mode data = some mode ∧ List.median data = some median := 
by
  let data := [32, 31, 16, 16, 14, 12]
  let mode := 16
  let median := 16
  sorry

end median_and_mode_of_shooting_data_l368_368519


namespace number_of_pupils_present_l368_368528

theorem number_of_pupils_present 
  (num_parents : ℕ)
  (num_teachers : ℕ)
  (total_people : ℕ)
  (h_parents : num_parents = 73)
  (h_teachers : num_teachers = 744)
  (h_total : total_people = 1541) :
  total_people - (num_parents + num_teachers) = 724 :=
by
  rw [h_parents, h_teachers, h_total]
  sorry

end number_of_pupils_present_l368_368528


namespace inequality_proof_infinitely_many_cases_of_equality_l368_368413

variable (a b c d : ℝ)

theorem inequality_proof (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1)
                        (h5 : 0 < c) (h6 : c < 1) (h7 : 0 < d) (h8 : d < 1)
                        (h_sum : a + b + c + d = 2) :
   sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ≤ (a * c + b * d) / 2 :=
sorry

theorem infinitely_many_cases_of_equality (h1 : 0 < a) (h2 : a < 1) 
                                          (h3 : 0 < b) (h4 : b < 1)
                                          (h5 : 0 < c) (h6 : c < 1) 
                                          (h7 : 0 < d) (h8 : d < 1) 
                                          (h_sum : a + b + c + d = 2) :
   (sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) = (a * c + b * d) / 2) ↔
   (a^2 + c^2 = b^2 + d^2) :=
sorry

end inequality_proof_infinitely_many_cases_of_equality_l368_368413


namespace pascal_triangle_row_sum_l368_368277

theorem pascal_triangle_row_sum : (∑ n in Finset.range 30, n + 1) = 465 := by
  sorry

end pascal_triangle_row_sum_l368_368277


namespace num_zeros_after_decimal_in_fraction_l368_368664

theorem num_zeros_after_decimal_in_fraction : 
  let f : ℚ := 1 / (30 : ℚ)^(30 : ℕ)
  in count_zeros_after_decimal f = 44 := by
  sorry

noncomputable def count_zeros_after_decimal (q : ℚ) : ℕ :=
  sorry

end num_zeros_after_decimal_in_fraction_l368_368664


namespace diff_of_squares_535_465_l368_368556

theorem diff_of_squares_535_465 : (535^2 - 465^2) = 70000 :=
sorry

end diff_of_squares_535_465_l368_368556


namespace polynomial_divisible_by_prime_l368_368388

/-!
  Given the integer polynomial P(x) = x³ + mx + n and the condition that 
  if P(x) - P(y) is divisible by 107, then x - y is divisible by 107 
  for integers x and y. Prove that 107 divides m.
-/

theorem polynomial_divisible_by_prime (m n : ℤ) (P : ℤ -> ℤ := λ x, x^3 + m * x + n) 
(h : ∀ x y : ℤ, (107 ∣ P x - P y) → (107 ∣ x - y)) :
  107 ∣ m :=
sorry

end polynomial_divisible_by_prime_l368_368388


namespace neither_sufficient_nor_necessary_condition_l368_368023

noncomputable def p (x : ℝ) : Prop := (x - 2) * (x - 1) > 0

noncomputable def q (x : ℝ) : Prop := x - 2 > 0 ∨ x - 1 > 0

theorem neither_sufficient_nor_necessary_condition (x : ℝ) : ¬(p x → q x) ∧ ¬(q x → p x) :=
by
  sorry

end neither_sufficient_nor_necessary_condition_l368_368023


namespace area_parallelogram_l368_368367

-- Define the points dividing AB and CD
variable {A B C D A4 B2 C4 D2 : Type}

-- Define conditions as predicates
def quintisection (p1 p2 p3 p4 p5 : Type) (line : (Type → Type)) : Prop :=
  line = split_into_equal_parts 5 [p1, p2, p3, p4, p5]

def trisection (p1 p2 p3 : Type) (line : (Type → Type)) : Prop :=
  line = split_into_equal_parts 3 [p1, p2, p3]

-- Area of quadrilateral A4 B2 C4 D2
def area (quad : Type) : ℝ := sorry

-- Define the statements using the conditions
theorem area_parallelogram (h1 : quintisection A1 A2 A3 A4 A5 AB)
                           (h2 : quintisection C1 C2 C3 C4 C5 CD)
                           (h3 : trisection B1 B2 B3 BC)
                           (h4 : trisection D1 D2 D3 DA)
                           (h5 : area (quad A4 B2 C4 D2) = 1) :
  area (parallelogram A B C D) = 15 :=
sorry

end area_parallelogram_l368_368367


namespace a_gt_abs_b_suff_not_necc_l368_368516

theorem a_gt_abs_b_suff_not_necc (a b : ℝ) (h : a > |b|) : 
  a^2 > b^2 ∧ ∀ a b : ℝ, (a^2 > b^2 → |a| > |b|) → ¬ (a < -|b|) := 
by
  sorry

end a_gt_abs_b_suff_not_necc_l368_368516


namespace sheets_in_stack_l368_368970

theorem sheets_in_stack 
  (num_sheets : ℕ) 
  (initial_thickness final_thickness : ℝ) 
  (t_per_sheet : ℝ) 
  (h_initial : num_sheets = 800) 
  (h_thickness : initial_thickness = 4) 
  (h_thickness_per_sheet : initial_thickness / num_sheets = t_per_sheet) 
  (h_final_thickness : final_thickness = 6) 
  : num_sheets * (final_thickness / t_per_sheet) = 1200 := 
by 
  sorry

end sheets_in_stack_l368_368970


namespace find_y1_l368_368229

theorem find_y1
  (y1 y2 y3 : ℝ)
  (h1 : 0 ≤ y3)
  (h2 : y3 ≤ y2)
  (h3 : y2 ≤ y1)
  (h4 : y1 ≤ 1)
  (h5 : (1 - y1)^2 + 2 * (y1 - y2)^2 + 2 * (y2 - y3)^2 + y3^2 = 1 / 2) :
  y1 = 3 / 4 :=
sorry

end find_y1_l368_368229


namespace product_of_positive_integer_c_l368_368992

theorem product_of_positive_integer_c (h : ∀ c : ℕ, 10 * x ^ 2 + 25 * x + c = 0 → (625 - 40 * c) ≥ 0) :
  ∏ i in (finset.range 16).filter (λ c, 10 * x ^ 2 + 25 * x + c = 0), c = 1307674368000 :=
by
  sorry

end product_of_positive_integer_c_l368_368992


namespace Tadashi_winning_strategy_l368_368855

-- Define the properties of the valid triples (a, b, c)
def valid_triple (a b c : ℕ) : Prop :=
  a + b + c = 2021 ∧ (∀ k : ℕ, k > 0 → 
  (a + k > 0 → b - k ≥ 0 → c - k ≥ 0) ∨ 
  (b + k > 0 → a - k ≥ 0 → c - k ≥ 0) ∨ 
  (c + k > 0 → a - k ≥ 0 → b - k ≥ 0))

-- State the number of winning triples for Tadashi
def num_winning_triples : ℕ :=
  3^8

theorem Tadashi_winning_strategy :
  ∃ (count : ℕ), count = num_winning_triples ∧ 
  count = (λ triple, ∃ a b c : ℕ, valid_triple a b c).card :=
sorry

end Tadashi_winning_strategy_l368_368855


namespace shaded_area_of_modified_design_l368_368779

noncomputable def radius_of_circles (side_length : ℝ) (grid_size : ℕ) : ℝ :=
  (side_length / grid_size) / 2

noncomputable def area_of_circle (radius : ℝ) : ℝ :=
  Real.pi * radius^2

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length^2

noncomputable def shaded_area (side_length : ℝ) (grid_size : ℕ) : ℝ :=
  let r := radius_of_circles side_length grid_size
  let total_circle_area := 9 * area_of_circle r
  area_of_square side_length - total_circle_area

theorem shaded_area_of_modified_design :
  shaded_area 24 3 = (576 - 144 * Real.pi) :=
by
  sorry

end shaded_area_of_modified_design_l368_368779


namespace num_good_pairs_l368_368658

-- Define the lines as given in the problem statement
def line1 : ℝ × ℝ := (3, 5)
def line2 : ℝ × ℝ := (2, 4)
def line3 : ℝ × ℝ := (3, -2 / 3)
def line4 : ℝ × ℝ := (1 / 2, -3 / 2)
def line5 : ℝ × ℝ := (1 / 4, -5 / 4)

-- Define a function to check if two slopes are parallel or perpendicular
def is_good_pair (m1 m2 : ℝ) : Prop :=
  m1 = m2 ∨ m1 * m2 = -1

-- Define the slopes of given lines
def slope1 : ℝ := 3
def slope2 : ℝ := 2
def slope3 : ℝ := 3
def slope4 : ℝ := 1 / 2
def slope5 : ℝ := 1 / 4

-- Define the good pairs count
def good_pairs_count : ℕ :=
  let slopes := [slope1, slope2, slope3, slope4, slope5];
  let pairs := [(slope1, slope3)] in -- Only pair (1, 3) is parallel
  pairs.length

-- Prove the total number of good pairs is 1
theorem num_good_pairs : good_pairs_count = 1 :=
by
  sorry

end num_good_pairs_l368_368658


namespace people_who_didnt_show_up_l368_368955

-- Definitions based on the conditions
def invited_people : ℕ := 68
def people_per_table : ℕ := 3
def tables_needed : ℕ := 6

-- Theorem statement
theorem people_who_didnt_show_up : 
  (invited_people - tables_needed * people_per_table = 50) :=
by 
  sorry

end people_who_didnt_show_up_l368_368955


namespace traditional_population_growth_pattern_l368_368931

/-- Define the sets of countries as given in the conditions -/
def developed_countries : set string := {"United Kingdom", "Japan", "United States", "Germany", "New Zealand", "China"}
def developing_countries : set string := {"Egypt", "India", "Libya"}

def traditional_growth_countries : set string := {"Egypt", "India", "Libya"}

/-- Main proof statement -/
theorem traditional_population_growth_pattern :
  ("Egypt" ∈ traditional_growth_countries) ∧
  ("India" ∈ traditional_growth_countries) ∧
  ("Libya" ∈ traditional_growth_countries) :=
by
  sorry

end traditional_population_growth_pattern_l368_368931


namespace distance_between_stripes_correct_l368_368971

noncomputable def distance_between_stripes : ℝ :=
  let base1 := 20
  let height1 := 50
  let base2 := 65
  let area := base1 * height1
  let d := area / base2
  d

theorem distance_between_stripes_correct : distance_between_stripes = 200 / 13 := by
  sorry

end distance_between_stripes_correct_l368_368971


namespace pascal_triangle_elements_count_l368_368269

theorem pascal_triangle_elements_count :
  ∑ n in finset.range 30, (n + 1) = 465 :=
by 
  sorry

end pascal_triangle_elements_count_l368_368269


namespace train_john_arrival_probability_l368_368027

-- Define the probability of independent uniform distributions on the interval [0, 120]
noncomputable def probability_train_present_when_john_arrives : ℝ :=
  let total_square_area := (120 : ℝ) * 120
  let triangle_area := (1 / 2) * 90 * 30
  let trapezoid_area := (1 / 2) * (30 + 0) * 30
  let total_shaded_area := triangle_area + trapezoid_area
  total_shaded_area / total_square_area

theorem train_john_arrival_probability :
  probability_train_present_when_john_arrives = 1 / 8 :=
by {
  sorry
}

end train_john_arrival_probability_l368_368027


namespace current_rate_is_1_point_2_l368_368964

def rate_of_current (c : ℝ) : Prop :=
  let man_speed := 3.6
  ∧ 2 * man_speed = man_speed + c 
  ∧ (2 * (man_speed - c) = man_speed + c)

theorem current_rate_is_1_point_2 : rate_of_current 1.2 :=
  by
  let man_speed := 3.6
  sorry

end current_rate_is_1_point_2_l368_368964


namespace not_axiomA_l368_368790

-- Definitions of propositions
def propA : Prop :=
  ∀ (P Q R : Plane), (P ∥ R ∧ Q ∥ R) → P ∥ Q

def propB : Prop :=
  ∀ (A B C : Point), ¬ Collinear A B C → ∃! (P : Plane), A ∈ P ∧ B ∈ P ∧ C ∈ P

def propC : Prop :=
  ∀ (l : Line) (P : Plane), (∃ A B : Point, A ∈ l ∧ B ∈ l ∧ A ∈ P ∧ B ∈ P) → ∀ (X : Point), X ∈ l → X ∈ P

def propD : Prop :=
  ∀ (P Q : Plane) (A : Point), A ∈ P ∧ A ∈ Q ∧ P ≠ Q → ∃! (l : Line), A ∈ l ∧ l ⊆ P ∧ l ⊆ Q

-- Axioms
axiom axiom1 : propC
axiom axiom2 : propB
axiom axiom3 : propD

-- Theorem to prove
theorem not_axiomA : ¬ propA := by
  sorry

end not_axiomA_l368_368790


namespace large_slices_per_bell_pepper_l368_368877

-- Define the conditions
variable (x : ℕ) -- Number of large slices per bell pepper
constant bellPeppers : ℕ := 5
constant totalSlicesAndPieces : ℕ := 200

-- Define derived quantities based on conditions
def largeSlicesTotal := bellPeppers * x
def smallPiecesTotal := bellPeppers * (3 * (x / 2))

-- State the problem as a theorem to prove x = 16
theorem large_slices_per_bell_pepper : 5 * x + (5 * (3 * (x / 2))) = 200 → x = 16 :=
by
  intros h
  sorry

end large_slices_per_bell_pepper_l368_368877


namespace solve_system_l368_368462

theorem solve_system :
  ∃ x y : ℝ, 3^x * 2^y = 972 ∧ log (sqrt 3) (x - y) = 2 ∧ x = 5 ∧ y = 2 :=
by { sorry }

end solve_system_l368_368462


namespace difference_of_squares_l368_368552

theorem difference_of_squares :
  535^2 - 465^2 = 70000 :=
by
  sorry

end difference_of_squares_l368_368552


namespace inner_right_triangle_area_l368_368781

theorem inner_right_triangle_area
  (a1 a2 a3 a4 : ℝ)
  (a1_eq : a1 = 64)
  (a2_eq : a2 = 81)
  (a3_eq : a3 = 49)
  (a4_eq : a4 = 36)
  (side1 : ℝ) (side2 : ℝ)
  (side1_eq : side1 = real.sqrt a1)
  (side2_eq : side2 = real.sqrt a2) :
  (1/2) * side1 * side2 = 36 := 
by
  sorry

end inner_right_triangle_area_l368_368781


namespace PQST_theorem_l368_368397

noncomputable def PQST_product : Prop :=
  ∀ (P Q S T : ℝ), 0 < P ∧ 0 < Q ∧ 0 < S ∧ 0 < T →
  log 10 (P * S) + log 10 (P * T) = 3 →
  log 10 (S * T) + log 10 (S * Q) = 4 →
  log 10 (Q * P) + log 10 (Q * T) = 5 →
  P * Q * S * T = 10000

theorem PQST_theorem : PQST_product :=
by {
  intros P Q S T hpos h1 h2 h3,
  -- The proof is to be filled in here
  sorry
}

end PQST_theorem_l368_368397


namespace find_m_l368_368086

theorem find_m (x m : ℤ) (h : (Polynomial.X + 5) ∣ (Polynomial.X ^ 2 - m * Polynomial.X - 40)) : m = 13 :=
sorry

end find_m_l368_368086


namespace number_of_divisors_gcd_120_180_240_l368_368533

theorem number_of_divisors_gcd_120_180_240 : 
  (Nat.divisors (Nat.gcd (Nat.gcd 120 180) 240)).length = 12 := 
by
  sorry

end number_of_divisors_gcd_120_180_240_l368_368533


namespace calculate_length_AP_l368_368352

noncomputable def length_AP (AB : ℝ) (angle_A : ℝ) (angle_B : ℝ) (BC H D M N P : ℝ) : ℝ :=
by sorry

theorem calculate_length_AP :
  let AB := 12
  let angle_A := real.to_radians 45
  let angle_B := real.to_radians 60

  -- Conditions
  let BC := (12 * (real.sin angle_A)) / (real.sin (real.to_radians 75))
  let AH := (AB * (real.sin angle_A)) / (real.cos angle_B)
  let HN := (BC / 2)
  let AP := real.sqrt ((AH ^ 2) + (HN ^ 2))

  -- Question and Answer
  in AP ≈ 17.52 :=
by sorry

end calculate_length_AP_l368_368352


namespace total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368308

-- Define the number of elements in the nth row of Pascal's Triangle
def num_elements_in_row (n : ℕ) : ℕ := n + 1

-- Define the sum of numbers from 0 to n, inclusive
def sum_of_first_n_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the total number of elements in the first 30 rows (0th to 29th)
def total_elements_in_first_30_rows : ℕ := sum_of_first_n_numbers 30

-- The main statement to prove
theorem total_numbers_in_first_30_rows_of_Pascals_Triangle :
  total_elements_in_first_30_rows = 465 :=
by
  simp [total_elements_in_first_30_rows, sum_of_first_n_numbers]
  sorry

end total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368308


namespace pascal_triangle_rows_sum_l368_368312

theorem pascal_triangle_rows_sum :
  ∑ k in finset.range 30, (k + 1) = 465 := by
  sorry

end pascal_triangle_rows_sum_l368_368312


namespace min_balls_to_ensure_fifteen_same_color_l368_368587

theorem min_balls_to_ensure_fifteen_same_color (r g y b w bl : ℕ) (total : ℕ) (h: total = 100) :
  r = 28 ∧ g = 20 ∧ y = 13 ∧ b = 19 ∧ w = 11 ∧ bl = 9 →
  (∃ n, n >= 76 ∧ (∀ (draws : finset (fin total)), draws.card = n → ∃ c ∈ draws.toList, (draws.toList.filter (λ x, c = x)).length ≥ 15)) :=
begin
  sorry,
end

end min_balls_to_ensure_fifteen_same_color_l368_368587


namespace arithmetic_seq_a4_l368_368239

theorem arithmetic_seq_a4 (a : ℕ → ℕ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 4) 
  (h3 : a 3 = 6) : 
  a 4 = 8 :=
by
  sorry

end arithmetic_seq_a4_l368_368239


namespace scientific_notation_of_600000_l368_368429

theorem scientific_notation_of_600000 : 600000 = 6 * 10^5 :=
by
  sorry

end scientific_notation_of_600000_l368_368429


namespace tangent_line_to_curve_at_point_l368_368001

-- Define the function
def f (x : ℝ) : ℝ := x * (3 * Real.log x + 1)

-- The definition of the point
def point : (ℝ × ℝ) := (1, 1)

-- The equation of the tangent line at the given point
def tangent_line_eq (x : ℝ) : ℝ := 4 * x - 3

theorem tangent_line_to_curve_at_point :
  ∀ (x y : ℝ), (x = 1 ∧ y = 1) → (f x = y) → ∀ (t : ℝ), tangent_line_eq t = 4 * t - 3 := by
  assume x y hxy hfx t
  sorry

end tangent_line_to_curve_at_point_l368_368001


namespace count_isosceles_numbers_correct_l368_368419

open Finset

def count_isosceles_numbers : ℕ :=
  let digits := range 1 10
  let equilateral_count := digits.card
  let isosceles_count :=
    (digits.product digits).filter (λ ab : ℕ × ℕ,
      let (a, b) := ab in a ≠ b ∧ 
      let aa := {a, a, b}
      ab.2 > 0 ∧ (2 * a > b)).card * 3
  equilateral_count + isosceles_count - 20

theorem count_isosceles_numbers_correct :
  count_isosceles_numbers = 165 :=
by
  sorry

end count_isosceles_numbers_correct_l368_368419


namespace Youseff_walk_time_per_block_l368_368072

theorem Youseff_walk_time_per_block :
  ∃ t_w : ℕ,
    let d := 12 in
    let t_b := 20 in
    let t_r := 8 * 60 in
    (d * t_w = d * t_b + t_r) ∧ (t_w = 60) :=
by
  sorry

end Youseff_walk_time_per_block_l368_368072


namespace vertices_square_condition_l368_368945

variables {a b c d e f : ℝ}

/-- 
  Suppose the graphs of the linear functions y = ax + c, y = ax + d, y = bx + e, 
  y = bx + f intersect at the vertices of a square P. Can the points K(a, c), L(a, d), 
  M(b, e), N(b, f) be located at the vertices of a square equal to the square P?
-/
theorem vertices_square_condition :
  ¬ (∃ (a b c d e f : ℝ),
    (∃ P : set (ℝ × ℝ), -- P is a set of points representing a square
      P = {((x : ℝ), (y : ℝ)) | (∃ t : ℝ, x = t * a + c ∨ x = t * a + d ∨ x = t * b + e ∨ x = t * b + f)} ∧
      (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ d ∧ e ≠ f ∧ a ≠ b)) ∧
    ∀ K L M N : ℝ × ℝ, 
      (K = (a, c) ∧ L = (a, d) ∧ M = (b, e) ∧ N = (b, f) →
       K ≠ L ∧ K ≠ M ∧ K ≠ N ∧ L ≠ M ∧ L ≠ N ∧ M ≠ N ∧
       dist K L = dist M N ∧ dist K M = dist L N ∧
       dist K N = dist L M)) := sorry

end vertices_square_condition_l368_368945


namespace banana_difference_l368_368103

theorem banana_difference (d : ℕ) :
  (8 + (8 + d) + (8 + 2 * d) + (8 + 3 * d) + (8 + 4 * d) = 100) →
  d = 6 :=
by
  sorry

end banana_difference_l368_368103


namespace impossible_transformation_l368_368797

def f (x : ℝ) := x^2 + 5 * x + 4
def g (x : ℝ) := x^2 + 10 * x + 8

theorem impossible_transformation :
  (∀ x, f (x) = x^2 + 5 * x + 4) →
  (∀ x, g (x) = x^2 + 10 * x + 8) →
  (¬ ∃ t : ℝ → ℝ → ℝ, (∀ x, t (f x) x = g x)) :=
by
  sorry

end impossible_transformation_l368_368797


namespace exists_k_for_arcs_l368_368741

noncomputable def length_of_arcs (m : ℕ) (M : Set ℝ) : ℝ := sorry -- Define length function properly

variables 
  (A B : Set ℝ)
  (l : ℝ) 
  (m : ℕ)
  (arc_length_eq : ∀ x ∈ B, length_of_arcs m {x} = π / m)
  (radius_eq_one : ∀ x ∈ A, ∀ y ∈ B, dist x y = 1)

-- Main theorem
theorem exists_k_for_arcs :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 2 * m ∧
    (length_of_arcs m ((A.rotate (k * π / m)) ∩ B) ≥ (1 / (2 * π)) * (length_of_arcs m A) * (length_of_arcs m B)) :=
sorry

end exists_k_for_arcs_l368_368741


namespace university_math_students_l368_368635

theorem university_math_students
  (total_students : ℕ)
  (math_only : ℕ)
  (stats_only : ℕ)
  (both_courses : ℕ)
  (H1 : total_students = 75)
  (H2 : math_only + stats_only + both_courses = total_students)
  (H3 : math_only = 2 * (stats_only + both_courses))
  (H4 : both_courses = 9) :
  math_only + both_courses = 53 :=
by
  sorry

end university_math_students_l368_368635


namespace min_lines_to_cover_centers_l368_368354

/-- A predicate to represent a 10x10 grid where the centers of all unit squares are marked.
    Our task is to find the minimum number of lines not parallel to the grid sides that can pass through all these points
    and to prove that this number is 18. -/
theorem min_lines_to_cover_centers (n : ℕ) (centers : ℕ → ℕ → Prop) 
  (H : centers 0 0 = true ∧ centers 9 9 = true) : 
  minimum_number_of_lines (n = 10 ∧ ∀ (i j : ℕ), i < n ∧ j < n → centers i j) = 18 :=
sorry

end min_lines_to_cover_centers_l368_368354


namespace Jose_share_land_l368_368808

theorem Jose_share_land (total_land : ℕ) (num_siblings : ℕ) (total_parts : ℕ) (share_per_person : ℕ) :
  total_land = 20000 → num_siblings = 4 → total_parts = (1 + num_siblings) → share_per_person = (total_land / total_parts) → 
  share_per_person = 4000 :=
by
  sorry

end Jose_share_land_l368_368808


namespace infinite_equal_pairs_l368_368225

theorem infinite_equal_pairs {α : Type*} (a : Int → α) (h : ∀ n : Int, a n = (1/4 : ℝ) * (a (n - 1) + a (n + 1))) :
  (∃ i j : Int, i < j ∧ a i = a j) → (∃ (f g : ℕ → Int), (∀ n, a (f n) = a (g n)) ∧ ∀ n, f n < g n) :=
by
  sorry

end infinite_equal_pairs_l368_368225


namespace product_fib_eq_l368_368395

noncomputable def G : ℕ → ℕ
| 1     := 2
| 2     := 1
| (n+1) := if n = 0 then 1 else G n + G (n - 1)

theorem product_fib_eq :
  (∏ k in finset.range 148, (G (k + 3) / G (k + 2) - G (k + 3) / G (k + 4))) = (G 150 / G 151) :=
sorry

end product_fib_eq_l368_368395


namespace a_n_bounds_b_n_formula_lim_a_n_over_b_n_l368_368256

-- Sequence definitions
def a_seq : ℕ → ℝ
| 0       := 1
| (n + 1) := 1 - (1 / (4 * (a_seq n)))

def b_seq (n : ℕ) : ℝ := 2 / (2 * a_seq n - 1)
def b_term (n : ℕ) : ℝ := 2 * n

-- Problem 1: Prove ∀ n ≥ 1, 1/2 < a_n ≤ 1
theorem a_n_bounds (n : ℕ) (hn : n ≥ 1) : (1/2 : ℝ) < a_seq n ∧ a_seq n ≤ 1 := sorry

-- Problem 2: Prove ∀ n ≥ 1, b_n = 2n
theorem b_n_formula (n : ℕ) (hn : n ≥ 1) : b_seq n = b_term n := sorry

-- Problem 3: Prove lim (n → +∞) a_n / b_n = 0
theorem lim_a_n_over_b_n : tendsto (λ n : ℕ, a_seq n / b_seq n) at_top (nhds 0) := sorry

end a_n_bounds_b_n_formula_lim_a_n_over_b_n_l368_368256


namespace radius_of_original_bubble_l368_368615

theorem radius_of_original_bubble (r R : ℝ) (h : r = 4 * real.cbrt 2) :
  (2 / 3) * real.pi * r^3 = (4 / 3) * real.pi * R^3 → R = 2 * real.cbrt 2 :=
by
  sorry

end radius_of_original_bubble_l368_368615


namespace melted_ice_cream_depth_l368_368116

theorem melted_ice_cream_depth :
  ∀ (r_sphere r_cylinder : ℝ) (h_cylinder : ℝ),
    r_sphere = 3 ∧ r_cylinder = 12 ∧
    (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h_cylinder →
    h_cylinder = 1 / 4 :=
by
  intros r_sphere r_cylinder h_cylinder h
  have r_sphere_eq : r_sphere = 3 := h.1
  have r_cylinder_eq : r_cylinder = 12 := h.2.1
  have volume_eq : (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h_cylinder := h.2.2
  sorry

end melted_ice_cream_depth_l368_368116


namespace area_of_triangle_l368_368925

theorem area_of_triangle : 
  let f := λ x : ℝ, 3 * x + 6 in
  let g := λ x : ℝ, -2 * x + 8 in
  let area := 21.6 in
  ∃ (x_intersect : ℝ) (y_intersect : ℝ), 
    (f x_intersect = y_intersect) ∧ (g x_intersect = y_intersect) ∧ 
    (∃ (x1 x2 x3 y1 y2 y3 : ℝ), 
      x1 = -2 ∧ y1 = 0 ∧ 
      x2 = 4 ∧ y2 = 0 ∧ 
      x3 = x_intersect ∧ y3 = y_intersect ∧ 
      (area = (1 / 2) * abs (x2 - x1) * y3)) :=
by sorry

end area_of_triangle_l368_368925


namespace pascal_triangle_elements_count_l368_368270

theorem pascal_triangle_elements_count :
  ∑ n in finset.range 30, (n + 1) = 465 :=
by 
  sorry

end pascal_triangle_elements_count_l368_368270


namespace solve_system_of_equations_l368_368466

noncomputable def system_solution (x y z : ℝ) : Prop :=
  (∃ k l m : ℤ, x = ↑k * Real.pi ∧ y = ↑l * Real.pi ∧ z = ↑m * Real.pi)

theorem solve_system_of_equations (x y z : ℝ) :
  (sin x + 2 * sin (x + y + z) = 0) ∧
  (sin y + 3 * sin (x + y + z) = 0) ∧
  (sin z + 4 * sin (x + y + z) = 0) ↔
  system_solution x y z :=
by sorry

end solve_system_of_equations_l368_368466


namespace find_x_l368_368341

noncomputable def h (x : ℝ) : ℝ := real.root 4 ((x + 5) / 5)

theorem find_x (x : ℝ) : 
  h (3 * x) = 3 * h x -> x = -200 / 39 :=
by
  unfold h
  sorry

end find_x_l368_368341


namespace meaningful_expression_l368_368534

theorem meaningful_expression (m : ℝ) :
  (2 - m ≥ 0) ∧ (m + 2 ≠ 0) ↔ (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end meaningful_expression_l368_368534


namespace household_A_bill_bill_formula_household_B_usage_household_C_usage_l368_368037

-- Definition of the tiered water price system
def water_bill (x : ℕ) : ℕ :=
if x <= 22 then 3 * x
else if x <= 30 then 3 * 22 + 5 * (x - 22)
else 3 * 22 + 5 * 8 + 7 * (x - 30)

-- Prove that if a household uses 25m^3 of water, the water bill is 81 yuan.
theorem household_A_bill : water_bill 25 = 81 := by 
  sorry

-- Prove that the formula for the water bill when x > 30 is y = 7x - 104.
theorem bill_formula (x : ℕ) (hx : x > 30) : water_bill x = 7 * x - 104 := by 
  sorry

-- Prove that if a household paid 120 yuan for water, their usage was 32m^3.
theorem household_B_usage : ∃ x : ℕ, water_bill x = 120 ∧ x = 32 := by 
  sorry

-- Prove that if household C uses a total of 50m^3 over May and June with a total bill of 174 yuan, their usage was 18m^3 in May and 32m^3 in June.
theorem household_C_usage (a b : ℕ) (ha : a + b = 50) (hb : a < b) (total_bill : water_bill a + water_bill b = 174) :
  a = 18 ∧ b = 32 := by
  sorry

end household_A_bill_bill_formula_household_B_usage_household_C_usage_l368_368037


namespace platinum_to_gold_ratio_l368_368871

variables (G P : ℝ)

-- Conditions:
def balance_on_gold_card (G: ℝ) := G / 3
def balance_on_platinum_card (P: ℝ) := P / 4
def new_balance_on_platinum_card (G P: ℝ) := balance_on_platinum_card P + balance_on_gold_card G
def portion_of_platinum_limit_spent (P: ℝ) := P * 5 / 12

-- Theorem to prove:
theorem platinum_to_gold_ratio (G P : ℝ)
  (h1 : new_balance_on_platinum_card G P = portion_of_platinum_limit_spent P) :
  P / G = 1 / 2 :=
by sorry

end platinum_to_gold_ratio_l368_368871


namespace factor_of_7_l368_368442

theorem factor_of_7 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 7 ∣ (a + 2 * b)) : 7 ∣ (100 * a + 11 * b) :=
by sorry

end factor_of_7_l368_368442


namespace point_A_is_B_l368_368403

-- Define the setup
structure Circle (P : Type) (r : ℝ) := 
(center : P)
(radius : r)

variables {P : Type} [metric_space P]
variables (C1 C2 : Circle P ℝ) (B : P)

-- Given conditions
def concentric_circles (C1 C2 : Circle P ℝ) (B : P) : Prop :=
C1.center = C2.center ∧ C2.radius > C1.radius ∧ dist C2.center B = C2.radius

-- Property to prove
theorem point_A_is_B (A B : P) (r1 r2 : ℝ) (h : r2 > r1) (hB : dist C2.center B = r2) :
  (∀ Q : P, dist A B ≤ dist A Q ∧ dist A C2.center = dist B C2.center) → A = B :=
sorry

end point_A_is_B_l368_368403


namespace bus_driver_total_hours_l368_368096

theorem bus_driver_total_hours 
  (regular_rate : ℝ := 14) 
  (regular_hours : ℕ := 40) 
  (overtime_rate : ℝ := 1.75 * regular_rate) 
  (total_earnings : ℝ := 998) : 
  let overtime_hours := (total_earnings - (regular_hours * regular_rate)) / overtime_rate in
  regular_hours + odd_round overtime_hours = 58 :=
by
  sorry

end bus_driver_total_hours_l368_368096


namespace johnson_potatoes_left_l368_368387

def number_of_potatoes_left (p_G p_total : ℕ) : ℕ :=
  let p_T := 2 * p_G
  let p_A := p_T / 3
  let p_given := p_G + p_T + p_A
  p_total - p_given

theorem johnson_potatoes_left (p_G p_total : ℕ) (h1 : p_G = 69) (h2 : p_total = 300) : number_of_potatoes_left p_G p_total = 47 :=
by
  rw [number_of_potatoes_left, h1, h2]
  -- Calculation steps would typically go here
  sorry

end johnson_potatoes_left_l368_368387


namespace diff_of_squares_example_l368_368557

theorem diff_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end diff_of_squares_example_l368_368557


namespace extraordinary_numbers_count_l368_368849

-- Definition of an extraordinary number
def is_extraordinary (n : ℕ) : Prop :=
  ∃ p : ℕ, p.prime ∧ 2 * p = n

-- Interval constraint
def in_interval (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 75

-- Combine definitions
def is_extraordinary_in_interval (n : ℕ) : Prop :=
  is_extraordinary n ∧ in_interval n

-- The final theorem to prove
theorem extraordinary_numbers_count : 
  {n : ℕ | is_extraordinary_in_interval n}.toFinset.card = 11 := sorry

end extraordinary_numbers_count_l368_368849


namespace base_addition_l368_368365

theorem base_addition (R1 R3 : ℕ) (F1 F2 : ℚ)
    (hF1_baseR1 : F1 = 45 / (R1^2 - 1))
    (hF2_baseR1 : F2 = 54 / (R1^2 - 1))
    (hF1_baseR3 : F1 = 36 / (R3^2 - 1))
    (hF2_baseR3 : F2 = 63 / (R3^2 - 1)) :
  R1 + R3 = 20 :=
sorry

end base_addition_l368_368365


namespace area_of_arithmetic_sequence_triangle_l368_368238

-- Definitions based on conditions
def sides_arithmetic_sequence (a b c : ℝ) (d : ℝ) : Prop :=
  b - a = d ∧ c - b = d

def largest_angle_sine (A B C : triangle) (sin_large_angel : ℝ) : Prop :=
  sin (max (angle A B C) (max (angle B C A) (angle C A B))) = sin_large_angel

-- Lean statement for the proof problem
theorem area_of_arithmetic_sequence_triangle (A B C : triangle) :
  sides_arithmetic_sequence A B C 4 →
  largest_angle_sine A B C (sqrt 3 / 2) →
  area_of_triangle A B C = 15 * sqrt 3 :=
by
  sorry

end area_of_arithmetic_sequence_triangle_l368_368238


namespace germinated_seeds_l368_368581

noncomputable def germination_deviation_bound (n : ℕ) (p : ℝ) (P : ℝ) : Prop :=
  let q := 1 - p in
  ∃ ε : ℝ, ε ≈ 0.034 ∧
  P (|((m : ℝ) / n : ℝ) - p| < ε) = P

theorem germinated_seeds :
  germination_deviation_bound 600 0.9 0.995 :=
sorry

end germinated_seeds_l368_368581


namespace minimum_value_sum_l368_368701

theorem minimum_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + 3 * c) + b / (8 * c + 4 * a) + 9 * c / (3 * a + 2 * b)) ≥ 47 / 48 :=
by sorry

end minimum_value_sum_l368_368701


namespace intersection_of_A_and_B_l368_368710

-- Defining the sets A and B based on given conditions
def setA : Set ℝ := {x | x^2 - 3 * x - 10 < 0}
def setB : Set ℝ := {x | 2^x < 2}

-- The proof problem to prove the intersection of sets A and B
theorem intersection_of_A_and_B : setA ∩ setB = {x | -2 < x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l368_368710


namespace range_of_m_l368_368700

theorem range_of_m (m : ℝ) : 
  ¬(∀ x : ℝ, x^2 - 2*x - 1 ≥ m^2 - 3*m) ↔ m ∈ set.Iio 1 ∪ set.Ioi 2 :=
sorry

end range_of_m_l368_368700


namespace Queen_High_School_teachers_needed_l368_368634

def students : ℕ := 1500
def classes_per_student : ℕ := 6
def students_per_class : ℕ := 25
def classes_per_teacher : ℕ := 5

theorem Queen_High_School_teachers_needed : 
  (students * classes_per_student) / students_per_class / classes_per_teacher = 72 :=
by 
  sorry

end Queen_High_School_teachers_needed_l368_368634


namespace shell_highest_point_time_l368_368611

theorem shell_highest_point_time (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : a * 7^2 + b * 7 + c = a * 14^2 + b * 14 + c) :
  (-b / (2 * a)) = 10.5 :=
by
  -- The proof is omitted as per the instructions
  sorry

end shell_highest_point_time_l368_368611


namespace jerry_sister_increase_temp_l368_368383

theorem jerry_sister_increase_temp :
  let T0 := 40
  let T1 := 2 * T0
  let T2 := T1 - 30
  let T3 := T2 - 0.3 * T2
  let T4 := 59
  T4 - T3 = 24 := by
  sorry

end jerry_sister_increase_temp_l368_368383


namespace problem1_problem2_m_eq_0_problem2_m_eq_3_l368_368083

noncomputable def problem1Expr : ℚ :=
  -(((-2 : ℚ)^2) + abs (-real.sqrt 3) - 2 * real.sin 60 + (1 / 2)⁻¹ )

theorem problem1 : problem1Expr = -2 :=
  sorry

noncomputable def problem2Expr (m : ℚ) : ℚ :=
  ((m / (m - 2)) - (2 * m / (m^2 - 4))) + (m / (m + 2))

theorem problem2_m_eq_0 : problem2Expr 0 = 0 :=
  sorry

theorem problem2_m_eq_3 : problem2Expr 3 = 12 / 5 :=
  sorry

end problem1_problem2_m_eq_0_problem2_m_eq_3_l368_368083


namespace pascal_triangle_count_30_rows_l368_368285

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l368_368285


namespace find_judy_rotation_l368_368143

-- Definition of the problem
def CarlaRotation := 480 % 360 -- This effectively becomes 120
def JudyRotation (y : ℕ) := (360 - 120) % 360 -- This should effectively be 240

-- Theorem stating the problem and solution
theorem find_judy_rotation (y : ℕ) (h : y < 360) : 360 - CarlaRotation = y :=
by 
  dsimp [CarlaRotation, JudyRotation] 
  sorry

end find_judy_rotation_l368_368143


namespace fill_table_with_numbers_l368_368866

-- Define the main theorem based on the conditions and question.
theorem fill_table_with_numbers (numbers : Finset ℤ) (table : ℕ → ℕ → ℤ)
  (h_numbers_card : numbers.card = 100)
  (h_sum_1x3_horizontal : ∀ i j, (table i j + table i (j + 1) + table i (j + 2) ∈ numbers))
  (h_sum_1x3_vertical : ∀ i j, (table i j + table (i + 1) j + table (i + 2) j ∈ numbers)):
  ∃ (t : ℕ → ℕ → ℤ), (∀ k, 1 ≤ k ∧ k ≤ 6 → ∃ i j, t i j = k) :=
sorry

end fill_table_with_numbers_l368_368866


namespace find_t_eq_l368_368404

variable (a V V_0 S t : ℝ)

theorem find_t_eq (h1 : V = a * t + V_0) (h2 : S = (1/3) * a * t^3 + V_0 * t) : t = (V - V_0) / a :=
sorry

end find_t_eq_l368_368404


namespace no_real_root_in_2_3_l368_368545

noncomputable def f : ℝ → ℝ := λ x, x^5 - 3*x - 1

theorem no_real_root_in_2_3 : ∀ x ∈ (Set.Ioo (2:ℝ) 3), f x ≠ 0 :=
by
  sorry

end no_real_root_in_2_3_l368_368545


namespace noel_baked_dozens_l368_368428

theorem noel_baked_dozens (total_students : ℕ) (percent_like_donuts : ℝ)
    (donuts_per_student : ℕ) (dozen : ℕ) (h_total_students : total_students = 30)
    (h_percent_like_donuts : percent_like_donuts = 0.80)
    (h_donuts_per_student : donuts_per_student = 2)
    (h_dozen : dozen = 12) :
    total_students * percent_like_donuts * donuts_per_student / dozen = 4 := 
by
  sorry

end noel_baked_dozens_l368_368428


namespace remainder_of_expression_l368_368758

theorem remainder_of_expression (n : ℤ) : (10 + n^2) % 7 = (3 + n^2) % 7 := 
by {
  sorry
}

end remainder_of_expression_l368_368758


namespace product_multiple_of_126_probability_correct_l368_368351

/-- 
The set of numbers we're considering.
-/
def numbers : Finset ℕ := {6, 14, 18, 28, 42, 49, 54}

/-- 
The prime factorization requirements for a product to be a multiple of 126:
- at least two factors of 2,
- at least one factor of 3,
- at least two factors of 7.
-/
def product_factors_requirement (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), x ∈ numbers ∧ y ∈ numbers ∧ x ≠ y ∧ (a * b) % 126 == 0

/-- 
Total number of ways to pick any two distinct numbers from the set.
-/
def total_pairs : ℕ := (numbers.card.choose 2)

/-- 
Number of successful outcomes where the product is a multiple of 126.
-/
def successful_pairs : ℕ := 
  let possible_pairs := numbers.to_finset.to_list.pair_combinations,
  possible_pairs.filter (uncurry product_factors_requirement).length

/-- 
The probability of the product of two distinct numbers being a multiple of 126.
-/
def probability : ℚ :=
  (successful_pairs : ℚ) / (total_pairs : ℚ)

/-- 
Proof that the probability of the product of two distinct members
of this set being a multiple of 126 is 1/7.
-/
theorem product_multiple_of_126_probability_correct : probability = 1 / 7 := 
by
  sorry

end product_multiple_of_126_probability_correct_l368_368351


namespace car_discount_l368_368119

variable (P D : ℝ)

theorem car_discount (h1 : 0 < P)
                     (h2 : (P - D) * 1.45 = 1.16 * P) :
                     D = 0.2 * P := by
  sorry

end car_discount_l368_368119


namespace find_a18_l368_368785

variable {a : ℕ → ℤ}  -- Define the arithmetic sequence as a function from natural numbers to integers

axiom arithmetic_sequence (n : ℕ) : a (n + 1) = a n + d  -- Arithmentic sequence property, with common difference d

variable a4_eq_10 : a 4 = 10  -- Given a_4 = 10
variable s12_eq_90 : (∑ i in finset.range 12, a (i + 1)) = 90  -- Given S_12 = 90

theorem find_a18 : a 18 = -4 := by
  sorry

end find_a18_l368_368785


namespace total_cups_sold_l368_368960

theorem total_cups_sold (plastic_cups : ℕ) (ceramic_cups : ℕ) (total_sold : ℕ) :
  plastic_cups = 284 ∧ ceramic_cups = 284 → total_sold = 568 :=
by
  intros h
  cases h
  sorry

end total_cups_sold_l368_368960


namespace foci_distance_l368_368156

theorem foci_distance :
  ∀ (x y : ℝ), 
  (sqrt ((x - 4)^2 + (y - 5)^2) + sqrt ((x + 6)^2 + (y - 9)^2) = 25) → 
  (real.dist (4, 5) (-6, 9) = 2 * sqrt 29) :=
by 
  intros x y h
  sorry

end foci_distance_l368_368156


namespace speed_of_man_in_still_water_l368_368073

variable (V_m V_s : ℝ)

/-- The speed of a man in still water -/
theorem speed_of_man_in_still_water (h_downstream : 18 = (V_m + V_s) * 3)
                                     (h_upstream : 12 = (V_m - V_s) * 3) :
    V_m = 5 := 
sorry

end speed_of_man_in_still_water_l368_368073


namespace problem_solution_l368_368408

theorem problem_solution (n : ℕ) (h : n^3 - n = 5814) : (n % 2 = 0) :=
by sorry

end problem_solution_l368_368408


namespace ice_cream_children_count_ice_cream_girls_count_l368_368864

-- Proof Problem for part (a)
theorem ice_cream_children_count (n : ℕ) (h : 3 * n = 24) : n = 8 := sorry

-- Proof Problem for part (b)
theorem ice_cream_girls_count (x y : ℕ) (h : x + y = 8) 
  (hx_even : x % 2 = 0) (hy_even : y % 2 = 0) (hx_pos : x > 0) (hxy : x < y) : y = 6 := sorry

end ice_cream_children_count_ice_cream_girls_count_l368_368864


namespace new_sales_tax_percentage_l368_368509

-- Definitions for given conditions
def T_original : ℝ := 3.5
def market_price : ℝ := 8400
def savings : ℝ := 14

-- Definitions based on the solution steps
def Tax_original : ℝ := (T_original / 100) * market_price
def Tax_new : ℝ := Tax_original - savings

-- The proposition to prove the new sales tax percentage
theorem new_sales_tax_percentage :
  Tax_new = (3.33 / 100) * market_price :=
sorry

end new_sales_tax_percentage_l368_368509


namespace volume_of_cone_is_correct_l368_368906

noncomputable def cone_volume (r l h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

noncomputable def volume_of_cone_with_given_areas (base_area lateral_area : ℝ) : ℝ :=
  let r := Real.sqrt (base_area / π) in
  let l := lateral_area / (π * r) in
  let h := Real.sqrt (l^2 - r^2) in
  cone_volume r l h

theorem volume_of_cone_is_correct :
  volume_of_cone_with_given_areas (2 * π) (6 * π) = (8 * π) / 3 :=
by
  sorry

end volume_of_cone_is_correct_l368_368906


namespace fraction_is_A_l368_368932

def expr_A (a : ℝ) : ℝ := 1 / (2 - a)
def expr_B (x : ℝ) : ℝ := x / (Real.pi - 3)
def expr_C (y : ℝ) : ℝ := - y / 5
def expr_D (x y : ℝ) : ℝ := x / 2 + y

theorem fraction_is_A (a x y : ℝ) : 
  expr_A a = 1 / (2 - a) ∧ 
  (expr_B x ≠ x / (Real.pi - 3)) ∧ 
  (expr_C y ≠ - y / 5) ∧ 
  (expr_D x y ≠ x / 2 + y) := 
sorry

end fraction_is_A_l368_368932


namespace probability_all_black_l368_368582

open ProbabilityTheory

-- Conditions
def probability (p : ℝ) : ℝ := if 0 ≤ p ∧ p ≤ 1 then p else 0

def initial_state (x : ℕ × ℕ) : ℝ :=
  let black := probability (1 / 3)
  let white := probability (1 / 3)
  let red := probability (1 / 3)
  if x.1 < 4 ∧ x.2 < 4 then black + white + red else 0

def rotated_state (x : ℕ × ℕ) : ℝ :=
  if x.1 < 4 ∧ x.2 < 4 then
    let pos := (x.2, 3 - x.1)
    let is_red := (initial_state pos = probability (1 / 3)) -- red
    let is_black := (initial_state x = probability (1 / 3)) -- black
    if is_red ∧ is_black then probability (1 / 3) -- black
    else initial_state x
  else 0

def final_grid_black (x : ℕ × ℕ) : ℝ :=
  if x.1 < 4 ∧ x.2 < 4 then
    let pos := (x.2, 3 - x.1)
    let res := initial_state pos + (probability (1 / 9)) -- considering the rotation affecting black
    res
  else 0

-- Goal
theorem probability_all_black : 
  (Π x : ℕ × ℕ, x.1 < 4 ∧ x.2 < 4 → final_grid_black x = probability (4 / 9)) 
  → (probability (4 / 9) ^ 16) = (4 / 9)^16 :=
by sorry

end probability_all_black_l368_368582


namespace solve_eq_l368_368066

theorem solve_eq : ∃ x : ℝ, (2 / x - (3 / x) * (5 / x) = -1 / 2) ↔ (x = -2 + Real.sqrt 34 ∨ x = -2 - Real.sqrt 34) :=
by
  intro x
  sorry

end solve_eq_l368_368066


namespace pascal_triangle_rows_sum_l368_368317

theorem pascal_triangle_rows_sum :
  ∑ k in finset.range 30, (k + 1) = 465 := by
  sorry

end pascal_triangle_rows_sum_l368_368317


namespace original_bubble_radius_l368_368620

theorem original_bubble_radius (r : ℝ) (R : ℝ) (π : ℝ) 
  (h₁ : r = 4 * real.cbrt 2)
  (h₂ : (4/3) * π * R^3 = (2/3) * π * r^3) : 
  R = 4 :=
by 
  sorry

end original_bubble_radius_l368_368620


namespace trajectory_eq_slope_range_l368_368434

-- Define the circle and conditions
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point M on the circle
def on_circle (M : ℝ × ℝ) : Prop := circle_equation M.1 M.2

-- Define perpendicular segment MD and point D on x-axis
def perpendicular_to_x_axis (M D : ℝ × ℝ) : Prop := M = (M.1, M.2) ∧ D = (M.1, 0)

-- Define point N derived from vector DN
def point_N (D M N : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, k = (sqrt 3) / 2 ∧ N = (M.1, k * M.2)

-- (1) Prove the equation of the trajectory T of point N
theorem trajectory_eq (M N: ℝ × ℝ) (hM : on_circle M) (hN : point_N ⟨M.1, 0⟩ M N) :
  N.1^2 / 4 + N.2^2 / 3 = 1 := 
sorry

-- (2) Prove the range of values for the slope of line AP
def line_slope (A P : ℝ × ℝ) : ℝ := (P.2 - A.2) / (P.1 - A.1)

theorem slope_range (A P : ℝ × ℝ) (hA : A = (2, 0)) (hF : 
  ∀ l : ℝ × ℝ → Prop, ∃ E F : ℝ × ℝ, l E ∧ l F ∧ E ≠ A ∧ F ≠ A ∧ (E.1 - A.1) * (F.1 - A.1) + (E.2 - A.2) * (F.2 - A.2) = 0 ∧ 
  2 * ⟨P.1, P.2⟩ = ⟨E.1 + F.1, E.2 + F.2⟩) : 
  - sqrt 14 / 56 ≤ line_slope A P ∧ line_slope A P ≤ sqrt 14 / 56 := 
sorry

end trajectory_eq_slope_range_l368_368434


namespace wall_area_160_l368_368608

noncomputable def wall_area (small_tile_area : ℝ) (fraction_small : ℝ) : ℝ :=
  small_tile_area / fraction_small

theorem wall_area_160 (small_tile_area : ℝ) (fraction_small : ℝ) (h1 : small_tile_area = 80) (h2 : fraction_small = 1 / 2) :
  wall_area small_tile_area fraction_small = 160 :=
by
  rw [wall_area, h1, h2]
  norm_num

end wall_area_160_l368_368608


namespace problem1_problem2_l368_368732

noncomputable theory

-- Parametric equation of line l
def parametric_line (m t : ℝ) : ℝ × ℝ :=
  let x := m + (sqrt 2) / 2 * t
  let y := (sqrt 2) / 2 * t
  (x, y)

-- Polar equation of the ellipse C
def ellipse_polar (ρ θ : ℝ) : Prop :=
  ρ^2 * cos θ^2 + 3 * ρ^2 * sin θ^2 = 12

-- Cartesian equation of the ellipse C
def ellipse_cartesian (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 4 = 1

-- Prove |FA| * |FB| = 2 given conditions
theorem problem1 (m t : ℝ) (A B : ℝ × ℝ) (F : ℝ × ℝ) (x y : ℝ) :
  ellipse_cartesian x y →
  parametric_line m t = F → 
  (A = ((sqrt 6 - 3 * sqrt 2) / 2, (sqrt 6 + sqrt 2) / 2)) →
  (B = (-(sqrt 6 + 3 * sqrt 2) / 2, (sqrt 2 - sqrt 6) / 2)) →
  abs (dist F A) * abs (dist F B) = 2 :=
sorry

-- Prove max perimeter of the inscribed rectangle in ellipse C is 16
theorem problem2 (θ : ℝ) (max_perimeter : ℝ) :
  (0 < θ ∧ θ < π/2) →
  let perimeter := 16 * sin (θ + π / 3)
  max_perimeter = 16 :=
sorry

end problem1_problem2_l368_368732


namespace bricks_not_sprinkled_with_lime_l368_368606

-- Define the given conditions
def length : ℕ := 30
def width : ℕ := 20
def height : ℕ := 10

-- Define the problem statement
theorem bricks_not_sprinkled_with_lime : (length - 2) * (width - 2) * (height - 2) = 4032 :=
by
  sorry

end bricks_not_sprinkled_with_lime_l368_368606


namespace problem_solution_l368_368682

theorem problem_solution (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ 3) :
    (8.17 * real.sqrt (3 * x - x ^ 2) < 4 - x) :=
sorry

end problem_solution_l368_368682


namespace problem_statement_l368_368231

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x
axiom f_def : ∀ x : ℝ, (0 < x ∧ x ≤ 1) → f x = x

theorem problem_statement :
  (f (-1 / 2) = f (5 / 2)) ∧
  (f 7 < f 8) ∧
  (∀ x : ℝ, f x ≥ f 3) :=
begin
  sorry
end

end problem_statement_l368_368231


namespace roll_in_second_round_roll_B_two_times_four_rounds_l368_368436

-- Part (1)
theorem roll_in_second_round (total_outcomes : ℕ) (favorable_outcomes : ℕ)
  (prob_sum_multiple_of_3 : ℚ) : 
  (favorable_outcomes = 12) → 
  (total_outcomes = 36) → 
  (prob_sum_multiple_of_3 = 1/3) → 
  (prob_sum_multiple_of_3 = favorable_outcomes / total_outcomes) → 
  prob_sum_multiple_of_3 = 1/3 :=
by
  intros favorable_outcomes_eq total_outcomes_eq prob_eq
  rw [favorable_outcomes_eq, total_outcomes_eq]
  exact prob_eq

-- Part (2)
theorem roll_B_two_times_four_rounds (non_multiple_prob : ℚ) 
  (multiple_prob : ℚ) 
  (comb1 : ℚ) (comb2 : ℚ) (comb3 : ℚ)
  (total_prob : ℚ) : 
  (multiple_prob = 1/3) → 
  (non_multiple_prob = 2/3) → 
  (comb1 = (2/3 * 1/3 * 2/3)) → 
  (comb2 = (2/3)^3) → 
  (comb3 = (1/3 * 2/3 * 1/3)) → 
  (total_prob = comb1 + comb2 + comb3) → 
  total_prob = 14/27 :=
by
  intros multiple_prob_eq non_multiple_prob_eq comb1_eq comb2_eq comb3_eq total_prob_eq
  rw [multiple_prob_eq, non_multiple_prob_eq, comb1_eq, comb2_eq, comb3_eq]
  exact total_prob_eq

end roll_in_second_round_roll_B_two_times_four_rounds_l368_368436


namespace range_of_b_l368_368733

theorem range_of_b (b c : ℝ) (h1 : ∀ x ∈ set.Icc (-1 : ℝ) 1, x^2 + 2 * b * x + c = 0) (h2 : 0 ≤ 4 * b + c ∧ 4 * b + c ≤ 3) :
  -1 ≤ b ∧ b ≤ 2 :=
sorry

end range_of_b_l368_368733


namespace pascal_triangle_sum_first_30_rows_l368_368295

theorem pascal_triangle_sum_first_30_rows :
  (Finset.range 30).sum (λ n, n + 1) = 465 :=
begin
  sorry
end

end pascal_triangle_sum_first_30_rows_l368_368295


namespace proof_total_legs_and_wheels_l368_368804

def leg_count : ℕ :=
  2                                                       -- Johnny's legs
  + 2 * 4                                                 -- Johnny's two dogs
  + 2                                                     -- Johnny's son's legs
  + 2                                                     -- Wheelchair-using friend's legs
  + 4                                                     -- Service dog's legs
  + 4                                                     -- Alice, Bob, and Carol each having 2 legs
  + 2                                                     -- Woman's legs
  + 4                                                     -- Cat's legs
  + 2                                                     -- Horse rider's legs
  + 4                                                     -- Horse's legs
  + 2                                                     -- Man's legs
  + 4                                                     -- Pet monkey's legs

def wheel_count : ℕ :=
  4 -- Wheelchair wheels

def total_legs_and_wheels : ℕ :=
  leg_count + wheel_count

theorem proof_total_legs_and_wheels : total_legs_and_wheels = 46 :=
by
  let leg_sum := 2 + (2 * 4) + 2 + 2 + 4 + 6 + 2 + 4 + 2 + 4 + 2 + 4
  let wheel_sum := 4
  show leg_sum + wheel_sum = 46
  have : leg_sum = 42 := by norm_num
  have : wheel_sum = 4 := by norm_num
  calc
    leg_sum + wheel_sum = 42 + 4 := by rw [this, this]
    ... = 46 := by norm_num

end proof_total_legs_and_wheels_l368_368804


namespace diff_of_squares_example_l368_368558

theorem diff_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end diff_of_squares_example_l368_368558


namespace appearance_equally_likely_all_selections_not_equally_likely_l368_368181

variables {n k : ℕ} (numbers : finset ℕ)

-- Conditions
def chosen_independently (x : ℕ) : Prop := sorry
def move_clockwise_if_chosen (x : ℕ) (chosen : finset ℕ) : Prop := sorry
def end_with_k_different_numbers (final_set : finset ℕ) : Prop := final_set.card = k

-- Part (a)
theorem appearance_equally_likely (x : ℕ) (h_independent : chosen_independently x)
  (h_clockwise : ∀ y ∈ numbers, move_clockwise_if_chosen y numbers) :
  (∃ y ∈ numbers, true) → true :=
by { sorry } -- Conclusion: Yes

-- Part (b)
theorem all_selections_not_equally_likely (samples : list (finset ℕ))
  (h_independent : ∀ x ∈ samples, chosen_independently x)
  (h_clockwise : ∀ y ∈ samples, move_clockwise_if_chosen y samples) :
  ¬ (∀ x y, x ≠ y → samples x = samples y) :=
by { sorry } -- Conclusion: No

end appearance_equally_likely_all_selections_not_equally_likely_l368_368181


namespace appropriate_sampling_method_is_stratified_l368_368591

-- Definition of the problem conditions
def total_students := 500 + 500
def male_students := 500
def female_students := 500
def survey_sample_size := 100

-- The goal is to show that given these conditions, the appropriate sampling method is Stratified sampling method.
theorem appropriate_sampling_method_is_stratified :
  total_students = 1000 ∧
  male_students = 500 ∧
  female_students = 500 ∧
  survey_sample_size = 100 →
  sampling_method = "Stratified" :=
by
  intros h
  sorry

end appropriate_sampling_method_is_stratified_l368_368591


namespace monotonicity_intervals_log_inequality_l368_368247

-- Definition for function f(x) based on given conditions
def f (x : ℝ) (k : ℝ) : ℝ := real.log (1 + x) - x + (k / 2) * x^2

-- Conditions for k, indicating it is non-negative
variable {k : ℝ} (h_nonneg : k ≥ 0)

-- First problem requires to prove intervals for k != 1
theorem monotonicity_intervals (h_k_ne_one : k ≠ 1) : 
  -- Statement about the intervals of monotonicity of f(x)
  sorry

-- Second problem requires the statement about k = 0 and x > -1
theorem log_inequality (h_k_zero : k = 0) (x : ℝ) (h_x_gt_neg_one : x > -1) : 
  real.log (x + 1) ≥ 1 - (1 / (x + 1)) :=
  sorry

end monotonicity_intervals_log_inequality_l368_368247


namespace who_is_in_middle_l368_368084

-- Define the carriages and their constraints
inductive Carriage : Type
| first
| second
| third
| fourth
| fifth

open Carriage

-- Define the people
inductive Person : Type
| A
| B
| C
| D
| E

open Person

-- Define the seating arrangement function
def arrangement : Carriage → Option Person

-- Define conditions as given in the problem
axiom D_in_fifth : arrangement fifth = some D
axiom A_immediately_behind_E : ∃ (c1 c2 : Carriage), arrangement c1 = some E ∧ arrangement c2 = some A ∧ (c2 = Carriage.pred c1)
axiom B_before_A : ∃ (c1 c2 : Carriage), arrangement c1 = some B ∧ arrangement c2 = some A ∧ (c1 < c2)
axiom B_and_C_apart : ∃ (c1 c2 : Carriage), arrangement c1 = some B ∧ arrangement c2 = some C ∧ (|c1.val - c2.val| > 1)

-- The target theorem to prove 
theorem who_is_in_middle : arrangement third = some A :=
sorry

end who_is_in_middle_l368_368084


namespace prove_fraction_of_25_smaller_by_22_when_compared_to_80_percent_of_40_l368_368542

def fraction_of_25_smaller_by_22_when_compared_to_80_percent_of_40 : Prop :=
  let x := 2 / 5 in
  let fraction_25 := x * 25 in
  let percent_80_of_40 := 0.80 * 40 in
  fraction_25 = percent_80_of_40 - 22

theorem prove_fraction_of_25_smaller_by_22_when_compared_to_80_percent_of_40 :
  fraction_of_25_smaller_by_22_when_compared_to_80_percent_of_40 :=
by
  sorry

end prove_fraction_of_25_smaller_by_22_when_compared_to_80_percent_of_40_l368_368542


namespace eccentricity_of_hyperbola_l368_368715

variable {a b c : ℝ}
variable {x y : Set (ℝ × ℝ)}
variable {O A B : ℝ × ℝ}

/-- F is the right focus of the hyperbola defined by a, b, and c -/
axiom right_focus (a b c : ℝ) (ha : a > b > 0) : F = (c, 0)

/-- The hyperbola equation -/
def hyperbola : Prop := ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1

/-- The coordinates of points A and B are given by their intersection with the asymptotes -/
axiom line_through_F_perpendicular (a b : ℝ) (ha : a > b > 0) (F : ℝ × ℝ) (A B : ℝ × ℝ) :
  ∀ (x1 y1 x2 y2 : ℝ), (y1 = b / a * x1 ∨ y1 = -b / a * x1) ∧ (y2 = b / a * x2 ∨ y2 = -b / a * x2)

/-- The origin is the coordinate origin -/
axiom origin (O : ℝ × ℝ) : O = (0, 0)

/-- The area of triangle OAB -/
axiom area_triangle_OAB (O A B : ℝ × ℝ) (A_ne_B : A ≠ B) :
  ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ (1 / 2 * a^2 * (b / a - (-b / a)) = (12 * a^2) / 7) 

/-- Prove the eccentricity of the hyperbola -/
theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > b > 0)
  (F : ℝ × ℝ) (A B : ℝ × ℝ) (hF : right_focus a b c ha)
  (hH : hyperbola)
  (hAB : line_through_F_perpendicular a b ha F A B)
  (hO : origin O)
  (hArea : area_triangle_OAB O A B A_ne_B) :
  (c / a = 5 / 4) :=
  sorry

end eccentricity_of_hyperbola_l368_368715


namespace sum_of_P_points_l368_368437

def A : ℤ := -2
def B : ℤ := -A
def P_points (p : ℤ) : Prop := (p = A - 3) ∨ (p = A + 3) ∨ (p = B - 3) ∨ (p = B + 3)

theorem sum_of_P_points : ∑ p in {p | P_points p}.to_finset, p = 0 :=
by
  sorry

end sum_of_P_points_l368_368437


namespace range_of_sum_of_squares_of_distances_l368_368794

noncomputable def parametric_curve_C1 (φ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos φ, 3 * Real.sin φ)

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def polar_coordinates_of_A := (2, Real.pi / 3 : ℝ)

def cartesian_coordinates_of_vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  (polar_to_cartesian 2 (Real.pi / 3), 
   polar_to_cartesian 2 (5 * Real.pi / 6), 
   polar_to_cartesian 2 (4 * Real.pi / 3), 
   polar_to_cartesian 2 (11 * Real.pi / 6))

def distance_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

noncomputable def sum_of_squares_of_distances (P : ℝ × ℝ) : ℝ :=
  let A := (1, Real.sqrt 3)
  let B := (-Real.sqrt 3, 1)
  let C := (-1, -Real.sqrt 3)
  let D := (Real.sqrt 3, -1)
  distance_squared P A + distance_squared P B + distance_squared P C + distance_squared P D

theorem range_of_sum_of_squares_of_distances (φ : ℝ) : 
  let P := parametric_curve_C1 φ in
  32 ≤ sum_of_squares_of_distances P ∧ sum_of_squares_of_distances P ≤ 52 :=
sorry

end range_of_sum_of_squares_of_distances_l368_368794


namespace students_neither_l368_368863

-- Define the given conditions
def total_students : Nat := 460
def football_players : Nat := 325
def cricket_players : Nat := 175
def both_players : Nat := 90

-- Define the Lean statement for the proof problem
theorem students_neither (total_students football_players cricket_players both_players : Nat) (h1 : total_students = 460)
  (h2 : football_players = 325) (h3 : cricket_players = 175) (h4 : both_players = 90) :
  total_students - (football_players + cricket_players - both_players) = 50 := by
  sorry

end students_neither_l368_368863


namespace youngest_child_age_l368_368571

theorem youngest_child_age (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 55) : x = 7 := 
by
  sorry

end youngest_child_age_l368_368571


namespace part_one_part_two_l368_368727

def f (x : ℝ) (a : ℝ) := (x - a) * Real.log x

theorem part_one (m : ℝ) : (∀ a ≥ 0, f x a) ∧ 
  (∀ m : ℝ, f x 0 = x * Real.log x → (∃ t m, Real.log (t) + 1 = 2 ∧ 2 * t + m = t) → m = -Real.exp 1) :=
sorry

theorem part_two (a : ℝ) : (∀ a ≥ 0, f x a) ∧ 
  (∀ a : ℝ, (∀ x ∈ (Set.Icc 1 2), (Real.log x + 1 - a / x) ≤ 0 ) → a = 2 * Real.log 2 + 2) :=
sorry

end part_one_part_two_l368_368727


namespace equilateral_triangle_altitude_length_correct_l368_368947

noncomputable def equilateral_triangle_altitude_length
  (K W U P : Point)
  (h1 : Triangle K W U)
  (h2 : EquilateralTriangle K W U)
  (h3 : SideLength K W U = 12)
  (h4 : OnMinorArcCircumcircle P (TriangleCircumcircle K W U) W U)
  (h5 : Distance K P = 13) :
  Real :=
  let altitude := Sorry -- placeholder for the altitude calculation
  altitude

theorem equilateral_triangle_altitude_length_correct
  (K W U P : Point)
  (h1 : Triangle K W U)
  (h2 : EquilateralTriangle K W U)
  (h3 : SideLength K W U = 12)
  (h4 : OnMinorArcCircumcircle P (TriangleCircumcircle K W U) W U)
  (h5 : Distance K P = 13) :
  equilateral_triangle_altitude_length K W U P h1 h2 h3 h4 h5 = (25 * sqrt 3) / 24 :=
by
  sorry

end equilateral_triangle_altitude_length_correct_l368_368947


namespace expected_rice_yield_l368_368769

theorem expected_rice_yield (x : ℝ) (y : ℝ) (h : y = 5 * x + 250) (hx : x = 80) : y = 650 :=
by
  sorry

end expected_rice_yield_l368_368769


namespace four_digit_sum_10_divisible_by_9_is_0_l368_368746

theorem four_digit_sum_10_divisible_by_9_is_0 : 
  ∀ (N : ℕ), (1000 * ((N / 1000) % 10) + 100 * ((N / 100) % 10) + 10 * ((N / 10) % 10) + (N % 10) = 10) ∧ (N % 9 = 0) → false :=
by
  sorry

end four_digit_sum_10_divisible_by_9_is_0_l368_368746


namespace prove_ab_leq_one_l368_368884

theorem prove_ab_leq_one (a b : ℝ) (h : (a + b + a) * (a + b + b) = 9) : ab ≤ 1 := 
by
  sorry

end prove_ab_leq_one_l368_368884


namespace sum_of_squares_l368_368698

theorem sum_of_squares (a b c : ℝ) :
  a + b + c = 4 → ab + ac + bc = 4 → a^2 + b^2 + c^2 = 8 :=
by
  sorry

end sum_of_squares_l368_368698


namespace find_t_u_P_ratio_l368_368393

noncomputable def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def A := point 1 2 3
def B := point 4 6 5

def vec (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2, q.3 - p.3)

def scalar_mult (a : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a * v.1, a * v.2, a * v.3)

def add_vec (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2, u.3 + v.3)

def P (t u : ℝ) (A B : ℝ × ℝ × ℝ) :=
  add_vec (scalar_mult t A) (scalar_mult u B)

theorem find_t_u_P_ratio (A B : ℝ × ℝ × ℝ) (hA : A = (1, 2, 3)) (hB : B = (4, 6, 5)) :
  ∃ t u, t = 1 / 5 ∧ u = 4 / 5 ∧ P t u A B = (17 / 5, 26 / 5, 23 / 5) := by
  use 1 / 5, 4 / 5
  -- additional proof steps to be provided here
  sorry

end find_t_u_P_ratio_l368_368393


namespace pascals_triangle_total_numbers_l368_368297

theorem pascals_triangle_total_numbers (N : ℕ) (hN : N = 29) :
  (∑ n in Finset.range (N + 1), (n + 1)) = 465 :=
by
  rw hN
  calc (∑ n in Finset.range 30, (n + 1))
      = ∑ k in Finset.range 30, (k + 1) : rfl
  -- Here we are calculating the sum of the first 30 terms of the sequence (n + 1)
  ... = 465 : sorry

end pascals_triangle_total_numbers_l368_368297


namespace cyclist_journey_time_l368_368957

theorem cyclist_journey_time
  (a v : ℝ)  -- total distance and planned constant speed
  (H1 : a / v = 5)  -- condition that the trip was planned to be 5 hours
  (H2 : ∀ t : ℝ, 0 ≤ t → t ≤ 1 → some_prop)  -- placeholder for the cyclist's constant speed until halfway
  (H3 : ∀ t : ℝ, 1 < t → t ≤ 2 → some_prop_inc_speed)  -- placeholder for the speed increased by 25% after halfway
  : (a / (2 * v)) + (a / (2.5 * v)) = 4.5 :=
by
  calc
    (a / (2 * v)) + (a / (2.5 * v))
      = a / 2v + a / 2.5v : by sorry
      ... = (5a/10v) + (4a/10v) : by sorry   -- common denominator step
      ... = (5a + 4a) / 10v : by sorry
      ... = 9a / 10v : by sorry
      ... = 9 * (a/v) / 10 : by sorry
      ... = 9 * 5 / 10 : by sorry
      ... = 4.5 : by sorry

end cyclist_journey_time_l368_368957


namespace pascal_triangle_sum_l368_368320

theorem pascal_triangle_sum (n : ℕ) (h₀ : n = 29) :
  (∑ k in Finset.range (n + 1), k + 1) = 465 := sorry

end pascal_triangle_sum_l368_368320


namespace point_B_coordinates_l368_368737

-- Defining the vector a
def vec_a : ℝ × ℝ := (1, 0)

-- Defining the point A
def A : ℝ × ℝ := (4, 4)

-- Definition of the line y = 2x
def on_line (P : ℝ × ℝ) : Prop := P.2 = 2 * P.1

-- Defining a vector as being parallel to another vector
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

-- Lean statement for the proof
theorem point_B_coordinates (B : ℝ × ℝ) (h1 : on_line B) (h2 : parallel (B.1 - 4, B.2 - 4) vec_a) :
  B = (2, 4) :=
sorry

end point_B_coordinates_l368_368737


namespace inequality_reciprocal_l368_368702

theorem inequality_reciprocal (a b : ℝ) (h₀ : a < b) (h₁ : b < 0) : (1 / a) > (1 / b) :=
sorry

end inequality_reciprocal_l368_368702


namespace exists_nat_b_l368_368415

theorem exists_nat_b (p : ℕ) (a : ℕ) (hp : p.prime) (ha : ¬ p ∣ a) :
  ∃ b : ℕ, a * b ≡ 1 [MOD p] :=
sorry -- The proof is omitted as requested.

end exists_nat_b_l368_368415


namespace general_term_formula_not_arithmetic_sequence_l368_368705

noncomputable def geometric_sequence (n : ℕ) : ℕ := 2^n

theorem general_term_formula :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n = geometric_sequence n) →
    (∃ (q : ℕ),
      ∀ n, a n = 2^n) :=
by
  sorry

theorem not_arithmetic_sequence :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n = geometric_sequence n) →
    ¬(∃ m n p : ℕ, m < n ∧ n < p ∧ (2 * a n = a m + a p)) :=
by
  sorry

end general_term_formula_not_arithmetic_sequence_l368_368705


namespace incorrect_number_read_as_l368_368477

theorem incorrect_number_read_as (n a_incorrect a_correct correct_number incorrect_number : ℕ) 
(hn : n = 10) (h_inc_avg : a_incorrect = 18) (h_cor_avg : a_correct = 22) (h_cor_num : correct_number = 66) :
incorrect_number = 26 := by
  sorry

end incorrect_number_read_as_l368_368477


namespace solve_equation_l368_368514

theorem solve_equation : ∀ x : ℝ, 4 * x - 2 * x + 1 - 3 = 0 → x = 1 :=
by
  intro x
  intro h
  sorry

end solve_equation_l368_368514


namespace quadratic_value_l368_368967

theorem quadratic_value (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : 4 * a + 2 * b + c = 3) :
  a + 2 * b + 3 * c = 7 :=
by
  sorry

end quadratic_value_l368_368967


namespace tv_price_change_l368_368943

theorem tv_price_change (P : ℝ) :
  let decrease := 0.20
  let increase := 0.45
  let new_price := P * (1 - decrease)
  let final_price := new_price * (1 + increase)
  final_price - P = 0.16 * P := 
by
  sorry

end tv_price_change_l368_368943


namespace find_p7_value_l368_368155

def quadratic (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem find_p7_value (d e f : ℝ)
  (h1 : quadratic d e f 1 = 4)
  (h2 : quadratic d e f 2 = 4) :
  quadratic d e f 7 = 5 := by
  sorry

end find_p7_value_l368_368155


namespace vector_calculation_l368_368998

def v1 : ℝ × ℝ := (3, -5)
def v2 : ℝ × ℝ := (-1, 6)
def v3 : ℝ × ℝ := (2, -1)

theorem vector_calculation :
  (5:ℝ) • v1 - (3:ℝ) • v2 + v3 = (20, -44) :=
by
  sorry

end vector_calculation_l368_368998


namespace total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368310

-- Define the number of elements in the nth row of Pascal's Triangle
def num_elements_in_row (n : ℕ) : ℕ := n + 1

-- Define the sum of numbers from 0 to n, inclusive
def sum_of_first_n_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the total number of elements in the first 30 rows (0th to 29th)
def total_elements_in_first_30_rows : ℕ := sum_of_first_n_numbers 30

-- The main statement to prove
theorem total_numbers_in_first_30_rows_of_Pascals_Triangle :
  total_elements_in_first_30_rows = 465 :=
by
  simp [total_elements_in_first_30_rows, sum_of_first_n_numbers]
  sorry

end total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368310


namespace eugene_pencils_after_giving_l368_368185

-- Define Eugene's initial number of pencils and the number of pencils given away.
def initial_pencils : ℝ := 51.0
def pencils_given : ℝ := 6.0

-- State the theorem that should be proved.
theorem eugene_pencils_after_giving : initial_pencils - pencils_given = 45.0 :=
by
  -- We would normally provide the proof steps here, but as per instructions, we'll use "sorry" to skip it.
  sorry

end eugene_pencils_after_giving_l368_368185


namespace complex_argument_l368_368928

theorem complex_argument:
  ∀ (re im : ℝ), re = 1 → im = √7 → ∃ θ : ℝ, ∃ r : ℝ, (r = √(re^2 + im^2)) ∧ (cos θ = re / r) ∧ (sin θ = im / r) ∧ (θ = π / 8) :=
by
  intros re im hre him
  use θ
  use r
  split; sorry
  split; sorry
  split; sorry
  sorry

end complex_argument_l368_368928


namespace perfect_squares_diff_two_consecutive_l368_368748

theorem perfect_squares_diff_two_consecutive (n : ℕ) (h : n = 20000) :
  {a : ℕ | ∃ b : ℕ, b * 2 + 1 < n ∧ a^2 = b * 2 + 1}.card = 71 :=
by
  sorry

end perfect_squares_diff_two_consecutive_l368_368748


namespace front_view_correct_l368_368191

-- Define the input conditions: heights of the stacks in each column
def firstColumnHeights : List Nat := [3, 2]
def secondColumnHeights : List Nat := [2, 4, 1]
def thirdColumnHeights : List Nat := [5, 1]

-- Define a function to determine the tallest stack in a list of heights
def tallestStack (heights : List Nat) : Nat :=
  heights.foldl max 0

-- Define the expected front view based on the heights given
def expectedFrontView := [3, 4, 5]

-- The actual front view determined by finding the tallest stack in each column
def actualFrontView :=
  [tallestStack firstColumnHeights, tallestStack secondColumnHeights, tallestStack thirdColumnHeights]

-- Theorem: The front view of the stacked cubes is as expected
theorem front_view_correct :
  actualFrontView = expectedFrontView :=
by
  simp [actualFrontView, expectedFrontView, tallestStack, firstColumnHeights, secondColumnHeights, thirdColumnHeights]
  sorry

end front_view_correct_l368_368191


namespace charge_difference_percentage_l368_368480

-- Given definitions
variables (G R P : ℝ)
def hotelR := 1.80 * G
def hotelP := 0.90 * G

-- Theorem statement
theorem charge_difference_percentage (G : ℝ) (hR : R = 1.80 * G) (hP : P = 0.90 * G) :
  (R - P) / R * 100 = 50 :=
by sorry

end charge_difference_percentage_l368_368480


namespace union_of_A_and_B_l368_368709

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 2} := sorry

end union_of_A_and_B_l368_368709


namespace planet_not_observed_l368_368793

theorem planet_not_observed (n : ℕ) (h : n = 2015) :
  ∃ p : Fin n, ∀ q : Fin n, p ≠ q → ¬ (closest q = p) :=
by
  sorry

end planet_not_observed_l368_368793


namespace john_earnings_ratio_l368_368802

noncomputable def john_earned_saturday : ℤ := 18
noncomputable def john_earned_sunday (x : ℤ) : ℤ := x
noncomputable def john_earned_previous_weekend : ℤ := 20
noncomputable def pogo_stick_cost : ℤ := 60
noncomputable def john_needs_more : ℤ := 13
noncomputable def total_earned : ℤ := john_earned_saturday + john_earned_sunday x + john_earned_previous_weekend

theorem john_earnings_ratio {x : ℤ} (h : total_earned = 47) : john_earned_saturday / john_earned_sunday x = 2 / 1 :=
by
  sorry

end john_earnings_ratio_l368_368802


namespace unit_vector_orthogonal_to_given_vectors_l368_368672

noncomputable def unit_vector : ℝ × ℝ × ℝ :=
  (1/Real.sqrt 11, -3/Real.sqrt 11, 1/Real.sqrt 11)

theorem unit_vector_orthogonal_to_given_vectors :
  let v := (2, 1, 1)
  let w := (0, 1, 3)
  let x := unit_vector in
  (v.1 * x.1 + v.2 * x.2 + v.3 * x.3 = 0) ∧
  (w.1 * x.1 + w.2 * x.2 + w.3 * x.3 = 0) ∧
  (x.1^2 + x.2^2 + x.3^2 = 1) :=
by
  sorry

end unit_vector_orthogonal_to_given_vectors_l368_368672


namespace max_good_pairs_1_to_30_l368_368605

def is_good_pair (a b : ℕ) : Prop := a % b = 0 ∨ b % a = 0

def max_good_pairs_in_range (n : ℕ) : ℕ :=
  if n = 30 then 13 else 0

theorem max_good_pairs_1_to_30 : max_good_pairs_in_range 30 = 13 :=
by
  sorry

end max_good_pairs_1_to_30_l368_368605


namespace solution_l368_368997

theorem solution (n : Nat) (h1 : ∀ n : Nat, ⌈n / 3⌉ - ⌈n / 4⌉ = 10) : n = 140 → (1 + 4 + 0 = 5) :=
by
  intros n h1 h2
  rw h2
  rfl

end solution_l368_368997


namespace series_sum_eval_l368_368668

noncomputable def series_sum (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), 1 / ((4 * k - 3) * (4 * k + 1) : ℚ)

theorem series_sum_eval (n : ℕ) : series_sum n = n / (4 * n + 1) :=
  sorry

end series_sum_eval_l368_368668


namespace polyhedron_faces_sum_zero_l368_368343

noncomputable def vec_sum_zero (faces : List (ℝ × ℝ × ℝ)) (areas : List ℝ) : Prop :=
  let vectors := faces.zip areas |>.map (λ ⟨(nx, ny, nz), a⟩, (a * nx, a * ny, a * nz))
  vectors.foldr (λ (v : ℝ × ℝ × ℝ) acc, (acc.1 + v.1, acc.2 + v.2, acc.3 + v.3)) (0,0,0) = (0,0,0)

-- Now, formulate our main theorem
theorem polyhedron_faces_sum_zero (faces : List (ℝ × ℝ × ℝ)) (areas : List ℝ) (h1 : ∀ (f : ℝ × ℝ × ℝ), f ∈ faces → f.1^2 + f.2^2 + f.3^2 = 1) 
(h2 : ∀ (a : ℝ), a ∈ areas → a > 0) 
(h3 : faces.length = areas.length) : 
vec_sum_zero faces areas :=
sorry

end polyhedron_faces_sum_zero_l368_368343


namespace John_needs_more_days_l368_368386

theorem John_needs_more_days (days_worked : ℕ) (amount_earned : ℕ) :
  days_worked = 10 ∧ amount_earned = 250 ∧ 
  (∀ d : ℕ, d < days_worked → amount_earned / days_worked = amount_earned / 10) →
  ∃ more_days : ℕ, more_days = 10 ∧ amount_earned * 2 = (days_worked + more_days) * (amount_earned / days_worked) :=
sorry

end John_needs_more_days_l368_368386


namespace paul_birthday_erasers_crayons_l368_368666

theorem paul_birthday_erasers_crayons
    (initial_regular_erasers : ℕ)
    (initial_jumbo_erasers : ℕ)
    (initial_standard_crayons : ℕ)
    (initial_jumbo_crayons : ℕ)
    (lost_regular_erasers : ℕ)
    (used_standard_crayons : ℕ)
    (used_jumbo_crayons : ℕ)
    (final_regular_erasers : ℕ)
    (final_jumbo_erasers : ℕ)
    (final_standard_crayons : ℕ)
    (final_jumbo_crayons : ℕ) :
    initial_regular_erasers = 307 → 
    initial_jumbo_erasers = 150 → 
    initial_standard_crayons = 317 →
    initial_jumbo_crayons = 300 →
    lost_regular_erasers = 52 →
    used_standard_crayons = 123 →
    used_jumbo_crayons = 198 →
    final_regular_erasers = initial_regular_erasers - lost_regular_erasers →
    final_jumbo_erasers = initial_jumbo_erasers →
    final_standard_crayons = initial_standard_crayons - used_standard_crayons →
    final_jumbo_crayons = initial_jumbo_crayons - used_jumbo_crayons →
    (final_standard_crayons + final_jumbo_crayons) - (final_regular_erasers + final_jumbo_erasers) = -109 :=
by
  sorry

end paul_birthday_erasers_crayons_l368_368666


namespace brendan_yards_per_week_l368_368134

def original_speed_flat : ℝ := 8  -- Brendan's speed on flat terrain in yards/day
def improvement_flat : ℝ := 0.5   -- Lawn mower improvement on flat terrain (50%)
def reduction_uneven : ℝ := 0.35  -- Speed reduction on uneven terrain (35%)
def days_flat : ℝ := 4            -- Days on flat terrain
def days_uneven : ℝ := 3          -- Days on uneven terrain

def improved_speed_flat : ℝ := original_speed_flat * (1 + improvement_flat)
def speed_uneven : ℝ := improved_speed_flat * (1 - reduction_uneven)

def total_yards_week : ℝ := (improved_speed_flat * days_flat) + (speed_uneven * days_uneven)

theorem brendan_yards_per_week : total_yards_week = 71.4 :=
sorry

end brendan_yards_per_week_l368_368134


namespace evaluate_at_neg_one_l368_368242

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x^2 else x^2 + x - 2

theorem evaluate_at_neg_one : f (-1) = 0 := by
  sorry

end evaluate_at_neg_one_l368_368242


namespace event_relation_l368_368908

-- Defining the events and number of balls
variable (red_balls : Finset ℕ) (yellow_balls : Finset ℕ) (white_balls : Finset ℕ)
variable (A_event B_event : Prop)

-- Defining conditions
def Event_A := ∃ (r ∈ red_balls) (y ∈ yellow_balls), true
def Event_B := ∃ (x₁ x₂ ∈ red_balls ∪ yellow_balls ∪ white_balls), x₁ ≠ x₂

-- Problem statement: Proving sufficient but not necessary relationship
theorem event_relation (hA : Event_A red_balls yellow_balls) 
    (hB : Event_B red_balls yellow_balls white_balls) :
    (Event_A red_balls yellow_balls → Event_B red_balls yellow_balls white_balls) ∧ 
    ¬ (Event_B red_balls yellow_balls white_balls → Event_A red_balls yellow_balls) :=
by
  sorry

end event_relation_l368_368908


namespace espresso_cost_l368_368652

theorem espresso_cost
  (price_cappuccino : ℕ)
  (price_iced_tea : ℕ)
  (price_cafe_latte : ℕ)
  (count_cappuccino count_iced_tea count_cafe_latte count_espresso : ℕ)
  (total_paid change : ℕ) :
  price_cappuccino = 2 →
  price_iced_tea = 3 →
  price_cafe_latte = 1.5 →
  count_cappuccino = 3 →
  count_iced_tea = 2 →
  count_cafe_latte = 2 →
  count_espresso = 2 →
  total_paid = 20 →
  change = 3 →
  (total_paid - change - (count_cappuccino * price_cappuccino + count_iced_tea * price_iced_tea + count_cafe_latte * price_cafe_latte))/ count_espresso = 1 :=
by {
  sorry,
}

end espresso_cost_l368_368652


namespace pascal_triangle_row_sum_l368_368275

theorem pascal_triangle_row_sum : (∑ n in Finset.range 30, n + 1) = 465 := by
  sorry

end pascal_triangle_row_sum_l368_368275


namespace math_problem_l368_368065

def a : ℕ := 2013
def b : ℕ := 2014

theorem math_problem :
  (a^3 - 2 * a^2 * b + 3 * a * b^2 - b^3 + 1) / (a * b) = a := by
  sorry

end math_problem_l368_368065


namespace max_number_of_squares_with_twelve_points_l368_368039

-- Define the condition: twelve marked points in a grid
def twelve_points_marked_on_grid : Prop := 
  -- Assuming twelve specific points represented in a grid-like structure
  -- (This will be defined concretely in the proof implementation context)
  sorry

-- Define the problem statement to be proved
theorem max_number_of_squares_with_twelve_points : 
  twelve_points_marked_on_grid → (∃ n, n = 11) :=
by 
  sorry

end max_number_of_squares_with_twelve_points_l368_368039


namespace identify_boxes_with_90g_bars_l368_368961

theorem identify_boxes_with_90g_bars :
  ∀ (m_V m_W m_X m_Y m_Z : ℕ),
    (m_V = 100 ∨ m_V = 90) →
    (m_W = 100 ∨ m_W = 90) →
    (m_X = 100 ∨ m_X = 90) →
    (m_Y = 100 ∨ m_Y = 90) →
    (m_Z = 100 ∨ m_Z = 90) →
    (20 * m_V + 20 * m_W + 20 * m_X + 20 * m_Y + 20 * m_Z = 5 * 20 * (100 + 100 + 100 + 90 + 90)) →
    (m_V + 2 * m_W + 4 * m_X + 8 * m_Y + 16 * m_Z = 2920) →
    (m_W = 90 ∧ m_Z = 90) :=
begin
  sorry
end

end identify_boxes_with_90g_bars_l368_368961


namespace find_train_probability_l368_368803

-- Define the time range and parameters
def start_time : ℕ := 120
def end_time : ℕ := 240
def wait_time : ℕ := 30

-- Define the conditions
def is_in_range (t : ℕ) : Prop := start_time ≤ t ∧ t ≤ end_time

-- Define the probability function
def probability_of_finding_train : ℚ :=
  let area_triangle : ℚ := (1 / 2) * 30 * 30
  let area_parallelogram : ℚ := 90 * 30
  let shaded_area : ℚ := area_triangle + area_parallelogram
  let total_area : ℚ := (end_time - start_time) * (end_time - start_time)
  shaded_area / total_area

-- The theorem to prove
theorem find_train_probability :
  probability_of_finding_train = 7 / 32 :=
by
  sorry

end find_train_probability_l368_368803


namespace g_neg6_eq_neg28_l368_368828

-- Define the given function g
def g (x : ℝ) : ℝ := 2 * x^7 - 3 * x^3 + 4 * x - 8

-- State the main theorem to prove g(-6) = -28 under the given conditions
theorem g_neg6_eq_neg28 (h1 : g 6 = 12) : g (-6) = -28 :=
by
  sorry

end g_neg6_eq_neg28_l368_368828


namespace cal_fraction_of_anthony_transactions_l368_368430

theorem cal_fraction_of_anthony_transactions 
    (Mabel_transactions : ℕ) 
    (Jade_transactions : ℕ) 
    (Anthony_transactions Cal_transactions : ℕ) 
    (h_mabel : Mabel_transactions = 90) 
    (h_anthony : Anthony_transactions = Mabel_transactions + Nat.floor (0.10 * Mabel_transactions)) 
    (h_jade : Jade_transactions = 84) 
    (h_cal_jade : Jade_transactions = Cal_transactions + 18) : 
    (Cal_transactions : ℚ) / (Anthony_transactions : ℚ) = 2 / 3 :=
by
  sorry

end cal_fraction_of_anthony_transactions_l368_368430


namespace find_common_ratio_l368_368236

-- Define the variables and constants involved.
variables (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)

-- Define the conditions of the problem.
def is_geometric_sequence := ∀ n, a (n + 1) = q * a n
def sum_of_first_n_terms := ∀ n, S n = a 0 * (1 - q^(n + 1)) / (1 - q)
def condition1 := a 5 = 4 * S 4 + 3
def condition2 := a 6 = 4 * S 5 + 3

-- The main statement that needs to be proved.
theorem find_common_ratio
  (h1: is_geometric_sequence a q)
  (h2: sum_of_first_n_terms a S q)
  (h3: condition1 a S)
  (h4: condition2 a S) : 
  q = 5 :=
sorry -- proof to be provided

end find_common_ratio_l368_368236


namespace appropriate_speech_length_l368_368809

def speech_length_min := 20
def speech_length_max := 40
def speech_rate := 120

theorem appropriate_speech_length 
  (min_words := speech_length_min * speech_rate) 
  (max_words := speech_length_max * speech_rate) : 
  ∀ n : ℕ, n >= min_words ∧ n <= max_words ↔ (n = 2500 ∨ n = 3800 ∨ n = 4600) := 
by 
  sorry

end appropriate_speech_length_l368_368809


namespace regular_octagon_side_length_eq_l368_368511

noncomputable def herons_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def octagon_area_side_length (x : ℝ) : ℝ :=
  2 * x^2 * (Real.sqrt 2 + 1)

theorem regular_octagon_side_length_eq (a b c : ℝ) (x : ℝ) 
  (ha : a = 20) (hb : b = 25) (hc : c = 30) (hx : x = 7.167) :
  herons_area a b c = octagon_area_side_length x :=
by
  sorry

end regular_octagon_side_length_eq_l368_368511


namespace graphic_artist_pages_l368_368861

theorem graphic_artist_pages (a₁ d n : ℕ) (a₁_eq : a₁ = 3) (d_eq : d = 2) (n_eq : n = 15) :
  (n * (a₁ + (a₁ + (n - 1) * d))) / 2 = 255 :=
by 
  rw [a₁_eq, d_eq, n_eq]
  sorry

end graphic_artist_pages_l368_368861


namespace chord_perpendicular_sum_l368_368532

theorem chord_perpendicular_sum (S : Type) (M : S) (AB : S) (A B : S) 
    (P Q : S) (h_circle : ∃ (R : ℝ), ∀ (X : S), X ∈ S → dist X M = R)
    (h_chord : ∃ (X Y : S), AB = X ∧ AB = Y ∧ X, Y ∈ S)
    (h_perp_MP : ∀ (X : S), dist M P = dist M X)
    (h_perp_MQ : ∀ (X : S), dist M Q = dist M X) :
    ∃ (c : ℝ), ∀ (chord : S), (1 / dist M P) + (1 / dist M Q) = c := sorry

end chord_perpendicular_sum_l368_368532


namespace side_length_of_equilateral_triangle_on_circle_l368_368098

-- Definitions based on conditions
def circle_area : ℝ := 25 * Real.pi
def point_O : ℝ × ℝ := (0, 0)  -- Place holder for point O coordinates
def r : ℝ := Real.sqrt 25  -- Since area = 25 * π implies r^2 = 25

def triangle_ABC (A B C : ℝ × ℝ) : Prop := 
  ∃ (s : ℝ), -- side length s
    -- Equilateral Triangle condition
    dist A B = s ∧ dist B C = s ∧ dist C A = s ∧
    -- Chord condition
    ∃ (M : ℝ × ℝ), midpoint B C = M ∧ dist (0, 0) M = r ∧
    -- O is outside triangle ABC
    ∃ (x_A : ℝ), dist point_O A = 6

-- Final theorem statement
theorem side_length_of_equilateral_triangle_on_circle (A B C : ℝ × ℝ) (h : triangle_ABC A B C) : 
  ∃ s, s = Real.sqrt 22 :=
by
  sorry

end side_length_of_equilateral_triangle_on_circle_l368_368098


namespace union_of_A_and_B_l368_368841

open Set -- to use set notation and operations

def A : Set ℝ := { x | -1/2 < x ∧ x < 2 }

def B : Set ℝ := { x | x^2 ≤ 1 }

theorem union_of_A_and_B :
  A ∪ B = Ico (-1:ℝ) 2 := 
by
  -- proof steps would go here, but we skip these with sorry.
  sorry

end union_of_A_and_B_l368_368841


namespace arithmetic_mean_of_35_integers_not_6_35_l368_368141

theorem arithmetic_mean_of_35_integers_not_6_35
    (a : Fin 35 → ℤ)
    (h : (∑ i, a i) = 6.35 * 35) :
    False := by
  sorry

end arithmetic_mean_of_35_integers_not_6_35_l368_368141


namespace extraordinary_numbers_count_l368_368850

-- Definition of an extraordinary number
def is_extraordinary (n : ℕ) : Prop :=
  ∃ p : ℕ, p.prime ∧ 2 * p = n

-- Interval constraint
def in_interval (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 75

-- Combine definitions
def is_extraordinary_in_interval (n : ℕ) : Prop :=
  is_extraordinary n ∧ in_interval n

-- The final theorem to prove
theorem extraordinary_numbers_count : 
  {n : ℕ | is_extraordinary_in_interval n}.toFinset.card = 11 := sorry

end extraordinary_numbers_count_l368_368850


namespace largest_odd_digit_multiple_of_11_l368_368548

theorem largest_odd_digit_multiple_of_11 (n : ℕ) (h1 : n < 10000) (h2 : ∀ d ∈ (n.digits 10), d % 2 = 1) (h3 : 11 ∣ n) : n ≤ 9559 :=
sorry

end largest_odd_digit_multiple_of_11_l368_368548


namespace grunters_win_all_5_games_grunters_win_at_least_one_game_l368_368476

/-- Given that the Grunters have an independent probability of 3/4 of winning any single game, 
and they play 5 games, the probability that the Grunters will win all 5 games is 243/1024. --/
theorem grunters_win_all_5_games :
  (3/4)^5 = 243 / 1024 :=
sorry

/-- Given that the Grunters have an independent probability of 3/4 of winning any single game, 
and they play 5 games, the probability that the Grunters will win at least one game is 1023/1024. --/
theorem grunters_win_at_least_one_game :
  1 - (1/4)^5 = 1023 / 1024 :=
sorry

end grunters_win_all_5_games_grunters_win_at_least_one_game_l368_368476


namespace order_of_y1_y2_y3_l368_368712

/-
Given three points A(-3, y1), B(3, y2), and C(4, y3) all lie on the parabola y = 2*(x - 2)^2 + 1,
prove that y2 < y3 < y1.
-/
theorem order_of_y1_y2_y3 :
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  sorry

end order_of_y1_y2_y3_l368_368712


namespace other_acute_angle_in_right_triangle_l368_368361

theorem other_acute_angle_in_right_triangle (a : ℝ) (h : a = 25) :
    ∃ b : ℝ, b = 65 :=
by
  sorry

end other_acute_angle_in_right_triangle_l368_368361


namespace find_f_six_l368_368006

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the function definition

axiom f_property : ∀ x y : ℝ, f (x - y) = f x * f y
axiom f_nonzero : ∀ x : ℝ, f x ≠ 0
axiom f_two : f 2 = 5

theorem find_f_six : f 6 = 1 / 5 :=
sorry

end find_f_six_l368_368006


namespace max_food_cost_l368_368565

theorem max_food_cost (total_cost : ℝ) (food_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_allowable : ℝ)
  (h1 : tax_rate = 0.07) (h2 : tip_rate = 0.15) (h3 : max_allowable = 75) (h4 : total_cost = food_cost * (1 + tax_rate + tip_rate)) :
  food_cost ≤ 61.48 :=
sorry

end max_food_cost_l368_368565


namespace proposition_1_correct_l368_368004

theorem proposition_1_correct:
  (∀ (α β : ℝ), (α = β) → (∀ (f : ℝ → ℝ), (f α = f β))) ∧
  ¬(∀ (α β : ℝ), (∀ (f : ℝ → ℝ), (α ≠ β) → (f α ≠ f β))) ∧
  ¬(∀ (α : ℝ), (sin α > 0) → (0 < α ∧ α < π ∨ π < α ∧ α < 2 * π)) ∧
  ¬(∀ (α β : ℝ), (sin α = sin β) → (∃ k : ℤ, α = 2 * k * π + β)) ∧
  ¬(∀ (α : ℝ), (π < α ∧ α < 2 * π) → (0 < α / 2 ∧ α / 2 < π)) :=
begin
  sorry
end

end proposition_1_correct_l368_368004


namespace solution_set_of_inequality_l368_368721

noncomputable def f : ℝ → ℝ
| x => if (x >= 0) then (x^2 - x) else f (-x)

theorem solution_set_of_inequality (x : ℝ) :
  (f (x + 2) < 6) ↔ (-5 < x ∧ x < 1) :=
by
  have h_even : ∀ x, f (-x) = f x := sorry -- f is even
  have h_defn : ∀ x, 0 ≤ x → f x = x^2 - x := sorry
  sorry

end solution_set_of_inequality_l368_368721


namespace range_of_a_l368_368708

variable {a : ℝ}

def A (a : ℝ) : Set ℝ := { x | (x - 2) * (x - (a + 1)) < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - 2 * a) / (x - (a^2 + 1)) < 0 }

theorem range_of_a (a : ℝ) : B a ⊆ A a ↔ (a = -1 / 2) ∨ (2 ≤ a ∧ a ≤ 3) := by
  sorry

end range_of_a_l368_368708


namespace Q_on_fixed_circle_l368_368151

open EuclideanGeometry

variable (Γ : Circle) (A B C : Point) (λ : Real) (h : 0 < λ ∧ λ < 1)
variable (P : Point) (hP : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ∈ Γ)

noncomputable def M (C P : Point) (λ : Real) : Point :=
  (1 - λ) • C + λ • P

theorem Q_on_fixed_circle (A B C P : Point) (Γ : Circle) (λ : Real) (h : 0 < λ ∧ λ < 1)
  (hP : P ∈ Γ ∧ P ≠ A ∧ P ≠ B ∧ P ≠ C) : 
  ∃ Q D : Point, (Q ∈ CircleCircum (A, M C P λ, P)) ∧
                 (Q ∈ CircleCircum (B, M C P λ, C)) ∧
                 (Q lies on a fixed circle as P varies) :=
sorry

end Q_on_fixed_circle_l368_368151


namespace import_tax_excess_amount_l368_368107

theorem import_tax_excess_amount (X : ℝ)
  (total_value : ℝ) (tax_paid : ℝ)
  (tax_rate : ℝ) :
  total_value = 2610 → tax_paid = 112.70 → tax_rate = 0.07 → 0.07 * (2610 - X) = 112.70 → X = 1000 :=
by
  intros h1 h2 h3 h4
  sorry

end import_tax_excess_amount_l368_368107


namespace alexa_pages_left_l368_368123

theorem alexa_pages_left 
  (total_pages : ℕ) 
  (first_day_read : ℕ) 
  (next_day_read : ℕ) 
  (total_pages_val : total_pages = 95) 
  (first_day_read_val : first_day_read = 18) 
  (next_day_read_val : next_day_read = 58) : 
  total_pages - (first_day_read + next_day_read) = 19 := by
  sorry

end alexa_pages_left_l368_368123


namespace isosceles_triangle_angle_sum_eq_90_l368_368796

theorem isosceles_triangle_angle_sum_eq_90 
  {A B C D M N : Point} 
  (h_triangle : triangle ABC)
  (h_bisector : angle_bisector A D B C)
  (h_circle_B : circle B BD ∩ AB = {M})
  (h_circle_C : circle C CD ∩ AC = {N})
  (h_isosceles : AB = AC) :
  angle BMN + angle CNM = 90 :=
sorry

end isosceles_triangle_angle_sum_eq_90_l368_368796


namespace percent_decrease_correct_l368_368667

def original_price_per_notebook := 10 / 6
def new_price_per_notebook := 9 / 8

def percent_decrease (price1 price2 : ℝ) := ((price1 - price2) / price1) * 100

theorem percent_decrease_correct :
  percent_decrease original_price_per_notebook new_price_per_notebook ≈ 33 :=
sorry

end percent_decrease_correct_l368_368667


namespace paint_left_l368_368261

-- Define the conditions
def total_paint_needed : ℕ := 333
def paint_needed_to_buy : ℕ := 176

-- State the theorem
theorem paint_left : total_paint_needed - paint_needed_to_buy = 157 := 
by 
  sorry

end paint_left_l368_368261


namespace trajectory_of_midpoint_l368_368109

theorem trajectory_of_midpoint 
  (x y : ℝ)
  (P : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : (M.fst - 4)^2 + M.snd^2 = 16)
  (hP : P = (x, y))
  (h_mid : M = (2 * P.1 + 4, 2 * P.2 - 8)) :
  x^2 + (y - 4)^2 = 4 :=
by
  sorry

end trajectory_of_midpoint_l368_368109


namespace tangent_eq_h_bounds_l368_368729

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) * (x^2 + 2)
noncomputable def g (x : ℝ) : ℝ := x / Real.exp 1
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem tangent_eq: ∀ (x f' : ℝ), f' = 2 → (∀ y, y = (2 * x + 2)) := sorry

theorem h_bounds : 
  ∀ (x : ℝ), x ∈ Icc (-2 : ℝ) 0 → (∀ y, y ∈ Icc (4 / Real.exp 1) 2) := sorry

end tangent_eq_h_bounds_l368_368729


namespace unique_solution_l368_368834

variable (N : Type) [Nat : N]
variable (a b : N) 

theorem unique_solution (x y : N) (h : x^(a + b) + y = x^a * y^b) : (x, y) = (2, 4) :=
sorry

end unique_solution_l368_368834


namespace find_m_l368_368009

-- Definitions
def line1 (m : ℝ) := ∀ x y : ℝ, (m + 3) * x + m * y - 2 = 0
def line2 (m : ℝ) := ∀ x y : ℝ, m * x - 6 * y + 5 = 0
def slope1 (m : ℝ) := -((m + 3) / m)
def slope2 (m : ℝ) := m / 6

-- Perpendicular condition
def perpendicular (m : ℝ) : Prop :=
  slope1 m * slope2 m = -1 ∧ m ≠ 0

theorem find_m (m : ℝ) : perpendicular m ↔ m = 3 := by
  sorry

end find_m_l368_368009


namespace min_operations_2014_to_1_l368_368628

theorem min_operations_2014_to_1 : 
  ∃ (n : ℕ), n = 5 ∧ 
  ∃ (ops : list (ℤ × ℤ → ℤ)), 
  (∀ (op : ℤ × ℤ → ℤ) (a b : ℤ), 
    op ∈ ops → 
    (∃ k, k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
           (op = λ x y, x + k ∨ 
            op = λ x y, x - k ∨ 
            op = λ x y, x * k ∨ 
            op = λ x y, x / k))) ∧
  (foldl (λ acc (op : ℤ × ℤ → ℤ), op acc 0) 2014 ops = 1) := 
begin 
  sorry 
end

end min_operations_2014_to_1_l368_368628


namespace value_of_f_at_3_l368_368005

theorem value_of_f_at_3 
  (f : ℝ → ℝ)
  (power_function : ∃ α : ℝ, ∀ x : ℝ, f(x) = x^α)
  (point (h : f 2 = 8)) : f 3 = 27 := 
by
  sorry

end value_of_f_at_3_l368_368005


namespace class_student_numbers_l368_368145

theorem class_student_numbers (a b c d : ℕ) 
    (h_avg : (a + b + c + d) / 4 = 46)
    (h_diff_ab : a - b = 4)
    (h_diff_bc : b - c = 3)
    (h_diff_cd : c - d = 2)
    (h_max_a : a > b ∧ a > c ∧ a > d) : 
    a = 51 ∧ b = 47 ∧ c = 44 ∧ d = 42 := 
by 
  sorry

end class_student_numbers_l368_368145


namespace problem1_problem2_l368_368432

-- Definitions of the conditions and properties
def loves (a b : Prop) := a → b → Prop
def aliens_love (x y : Prop): Prop := ∀ (a b : x ∧ y), loves a b

-- Define the problem conditions
variables (n k : ℕ)
variable loves_k : ∀ (x : Prop), ∀ (y : Prop), x → y → loves x y

-- Establish the two proof problems
theorem problem1 : n = 2 * k → (∃ x y z, ¬(aliens_love x y ∧ aliens_love y z ∧ aliens_love z x)) := sorry
theorem problem2 : 4 * k ≥ 3 * n → (∃ (families : list (Prop × Prop × Prop)), list.length families = n ∧ 
  ∀ (family : Prop × Prop × Prop), family ∈ families → aliens_love (family.fst) (family.snd) ∧ aliens_love (family.snd) (family.snd.fst)) := sorry

end problem1_problem2_l368_368432


namespace sin_plus_cos_eq_sqrt_five_over_two_l368_368697

theorem sin_plus_cos_eq_sqrt_five_over_two (α : ℝ) (h1 : sin α * cos α = 1 / 8) (h2 : 0 < α ∧ α < π / 2) : 
  sin α + cos α = sqrt 5 / 2 :=
by
  sorry

end sin_plus_cos_eq_sqrt_five_over_two_l368_368697


namespace radius_of_sphere_is_3_l368_368233

noncomputable theory

variables {Point : Type}
variables {dist : Point → Point → ℝ} -- distance function

-- Given points A, B, C, D and O (as the center of the sphere)
variables (A B C D O : Point)

-- Given distance axioms
axiom dist_AB : dist A B = 3
axiom dist_AC : dist A C = 4
axiom dist_AD : dist A D = Real.sqrt 11

-- Mutually perpendicular line segments
axiom perp_AB_AC : dist A B ≠ 0 ∧ dist A C ≠ 0
axiom perp_AB_AD : dist A B ≠ 0 ∧ dist A D ≠ 0
axiom perp_AC_AD : dist A C ≠ 0 ∧ dist A D ≠ 0

-- Points lying on a sphere centered at O
axiom on_sphere_B : dist O B = dist O A
axiom on_sphere_C : dist O C = dist O A
axiom on_sphere_D : dist O D = dist O A

-- Prove the radius of the sphere
theorem radius_of_sphere_is_3 : dist O A = 3 := sorry

end radius_of_sphere_is_3_l368_368233


namespace stripe_length_l368_368596

theorem stripe_length 
  (circumference height : ℝ)
  (h_circumference : circumference = 20)
  (h_height : height = 8)
  : sqrt (height^2 + (2 * (circumference / 2))^2) = sqrt 464 :=
by
  sorry

end stripe_length_l368_368596


namespace not_reduced_fraction_count_l368_368015

theorem not_reduced_fraction_count : 
  (fintype.card {N : ℕ | 1 ≤ N ∧ N ≤ 1990 ∧ ¬ (is_coprime 23 (N + 4))}) = 86 :=
begin
  sorry
end

end not_reduced_fraction_count_l368_368015


namespace units_digit_of_quotient_is_zero_l368_368648

theorem units_digit_of_quotient_is_zero (a b : ℕ) (h : (4^2065 + 6^2065) % 7 = 0) : 
  ((4^2065 + 6^2065) / 7) % 10 = 0 :=
begin
  sorry
end

end units_digit_of_quotient_is_zero_l368_368648


namespace length_of_chord_angle_between_tangents_l368_368592

-- Definitions based on conditions
def parabola (x : ℝ) : ℝ := x * sqrt(12 * x)
def chord_equation (x : ℝ) : ℝ := (x - 3) * sqrt(3)

-- The problem statements to prove:
theorem length_of_chord : 
  let A := (1 : ℝ, chord_equation 1)
  let B := (9 : ℝ, chord_equation 9)
  dist A B = 16 :=
sorry

theorem angle_between_tangents :
  let phi := 30 * (π / 180)
  2 * phi = π / 2 :=
sorry

end length_of_chord_angle_between_tangents_l368_368592


namespace largest_class_students_l368_368567

theorem largest_class_students (x : ℕ)
  (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 95) :
  x = 23 :=
by
  sorry

end largest_class_students_l368_368567


namespace zeros_after_decimal_in_fraction_l368_368137

theorem zeros_after_decimal_in_fraction :
  (let n := (6 * 10)^10 in
  let fraction := 1 / n in
  let num_zeros := String.length (fraction.toString.splitOn '.'[1].takeWhile (· = '0')) in
  num_zeros) = 17 :=
by
  -- necessary imports and setup would go here if needed
  sorry

end zeros_after_decimal_in_fraction_l368_368137


namespace lollipops_distribution_l368_368421

theorem lollipops_distribution :
  let initial_lollipops := 158
  let given_to_emily := (5:ℚ) / 6 * initial_lollipops
  let remaining_after_emily := initial_lollipops - given_to_emily
  let given_to_jack := (4:ℚ) / 7 * remaining_after_emily
  let remaining_after_jack := remaining_after_emily - given_to_jack
  let given_to_kyla := remaining_after_jack / 2
  let remaining_after_kyla := remaining_after_jack - given_to_kyla
  let given_to_lou := remaining_after_kyla
  in given_to_lou = 6 := by sorry

end lollipops_distribution_l368_368421


namespace complex_transform_result_l368_368050

noncomputable def complex_transform : ℂ :=
  let z : ℂ := 1 - 3 * complex.I
  let rotation_factor : ℂ := complex.of_real (real.sqrt 2) * (0.5 + 0.5 * complex.I * real.sqrt 3)
  z * rotation_factor

theorem complex_transform_result :
  complex_transform = (real.sqrt 2 + 3 * real.sqrt 6) / 2 + ((real.sqrt 6 - 3 * real.sqrt 2) / 2) * complex.I :=
by
  sorry

end complex_transform_result_l368_368050


namespace solve_eqn_in_integers_l368_368459

theorem solve_eqn_in_integers :
  ∃ (x y : ℤ), xy + 3*x - 5*y = -3 ∧ 
  ((x, y) = (6, 9) ∨ (x, y) = (7, 3) ∨ (x, y) = (8, 1) ∨ 
  (x, y) = (9, 0) ∨ (x, y) = (11, -1) ∨ (x, y) = (17, -2) ∨ 
  (x, y) = (4, -15) ∨ (x, y) = (3, -9) ∨ (x, y) = (2, -7) ∨ 
  (x, y) = (1, -6) ∨ (x, y) = (-1, -5) ∨  (x, y) = (-7, -4)) :=
sorry

end solve_eqn_in_integers_l368_368459


namespace innermost_rectangle_length_l368_368597

theorem innermost_rectangle_length :
  ∃ l : ℝ,
    l = 4 ∧
    ∃ (a₁ a₂ a₃ : ℝ),
      a₁ = 2 * l ∧
      (∃ w : ℝ, w = 2 ∧ a₂ = (l + 2 * w) * (w + 4) ∧ a₃ = (l + 4 * w) * (w + 8)) ∧
      (a₂ - a₁ = (a₃ - a₂) / 2) :=
begin
  sorry,
end

end innermost_rectangle_length_l368_368597


namespace no_number_satisfies_conditions_l368_368580

theorem no_number_satisfies_conditions :
  ∀ x : ℕ, 137 + x = 435 ∧ reverse_digits x = 672 → false :=
begin
  -- Definition of reverse_digits (assuming it is not predefined):
  def reverse_digits (n : ℕ) : ℕ :=
    n.digits 10.reverse.mk_nat 10
    
  sorry
end

end no_number_satisfies_conditions_l368_368580


namespace max_marked_numbers_not_exceeding_2016_l368_368547

theorem max_marked_numbers_not_exceeding_2016 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, n ≤ 2016) ∧ (∀ a b ∈ S, ∃ k, a * b = k^2) ∧ S.card = 44 := 
by
  sorry

end max_marked_numbers_not_exceeding_2016_l368_368547


namespace min_colors_needed_for_12_centers_l368_368939

def min_colors_for_coding (m : ℕ) : ℕ :=
  let pairs (n : ℕ) := n * (n - 1) / 2
  in
  Nat.find (λ n, n + pairs n ≥ m)

theorem min_colors_needed_for_12_centers : min_colors_for_coding 12 = 5 :=
by
  sorry

end min_colors_needed_for_12_centers_l368_368939


namespace find_wall_width_l368_368891

-- Definitions of given conditions
def height (W : ℕ) : ℕ := 6 * W
def length (W : ℕ) : ℕ := 7 * height W
def volume (W : ℕ) : ℕ := W * height W * length W

-- Problem Statement: There exists a width W such that volume W = 16128 and W = 4
theorem find_wall_width : ∃ (W : ℕ), volume W = 16128 ∧ W = 4 :=
  by
    sorry

end find_wall_width_l368_368891


namespace simplify_fraction_complex_l368_368220

theorem simplify_fraction_complex : 
  (2 - (1:ℂ) * complex.I) / (1 + complex.I) = 1/2 - (3/2) * complex.I := by
  sorry

end simplify_fraction_complex_l368_368220


namespace polygon_sides_from_angle_sum_l368_368517

-- Let's define the problem
theorem polygon_sides_from_angle_sum : 
  ∀ (n : ℕ), (n - 2) * 180 = 900 → n = 7 :=
by
  intros n h
  sorry

end polygon_sides_from_angle_sum_l368_368517


namespace max_guards_l368_368886

/-- Define a concave polygon with n sides -/
def concave_ngon (n : ℕ) : Prop := sorry  -- Placeholder for a formal definition of a concave n-gon

/-- Define the visibility condition -/
def visibility (n : ℕ) (k : ℕ) : Prop := 
  ∀ (polygon : Type) (guards : ℕ → Prop), concave_ngon n → (polygon → Prop)  -- The exact formalizations will depend on more detailed definitions, so we use a placeholder for now

/-- The maximum number of guards needed for an n-sided concave polygon -/
theorem max_guards (n : ℕ) : ∃ k, visibility n k ∧ k = (n / 3).floor :=
by {
  sorry  -- Proof here
}

end max_guards_l368_368886


namespace minimum_phi_l368_368007

def original_function (x : ℝ) : ℝ := real.sin (2 * x - real.pi / 3)

-- Translated function to the left by φ units
def translated_function (x φ : ℝ) : ℝ := real.sin (2 * (x + φ) - real.pi / 3)

-- Odd function property for translated function
def is_odd_function (φ : ℝ) : Prop :=
  ∀ x : ℝ, translated_function (-x) φ = -translated_function x φ

-- The minimum value of φ such that the translated function is odd
theorem minimum_phi : ∃ φ > 0, φ = real.pi / 6 ∧ is_odd_function φ :=
sorry

end minimum_phi_l368_368007


namespace least_possible_value_of_a_plus_b_l368_368826

theorem least_possible_value_of_a_plus_b : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  Nat.gcd (a + b) 330 = 1 ∧
  b ∣ a^a ∧ 
  ∀ k : ℕ, b^3 ∣ a^a → (k ∣ a → k = 1) ∧
  a + b = 392 :=
by
  sorry

end least_possible_value_of_a_plus_b_l368_368826


namespace negate_neg_two_l368_368454

theorem negate_neg_two : -(-2) = 2 := by
  -- The proof goes here
  sorry

end negate_neg_two_l368_368454


namespace integral_arctan_tangent_l368_368149

theorem integral_arctan_tangent (x : ℝ) : ∫ x in 0..Real.arctan 2, (12 + Real.tan x) / (3 * Real.sin x ^ 2 + 12 * Real.cos x ^ 2) = (Real.pi / 2) + (1 / 6) * Real.log 2 := by
  sorry

end integral_arctan_tangent_l368_368149


namespace fraction_value_l368_368067

theorem fraction_value (x : ℕ) (h : x = 3) :
  (\(∏ i in finset.range 15, x ^ (i + 1) * 2\) / \(∏ i in finset.range 9, x ^ (i + 1) * 3\)) = 3 ^ 105 :=
by
  sorry

end fraction_value_l368_368067


namespace non_upgraded_sensor_ratio_l368_368114

theorem non_upgraded_sensor_ratio 
  (N U S : ℕ) 
  (units : ℕ := 24) 
  (fraction_upgraded : ℚ := 1 / 7) 
  (fraction_non_upgraded : ℚ := 6 / 7)
  (h1 : U / S = fraction_upgraded)
  (h2 : units * N = (fraction_non_upgraded * S)) : 
  N / U = 1 / 4 := 
by 
  sorry

end non_upgraded_sensor_ratio_l368_368114


namespace triangle_area_l368_368048

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area : 
  ∀ (A B C : (ℝ × ℝ)),
  (A = (3, 3)) →
  (B = (4.5, 7.5)) →
  (C = (7.5, 4.5)) →
  area_of_triangle A B C = 8.625 :=
by
  intros A B C hA hB hC
  rw [hA, hB, hC]
  unfold area_of_triangle
  norm_num
  sorry

end triangle_area_l368_368048


namespace area_of_triangle_tangent_line_l368_368197

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def tangent_line_at_1 (y x : ℝ) : Prop := y = x - 1

theorem area_of_triangle_tangent_line :
  let tangent_intercept_x : ℝ := 1
  let tangent_intercept_y : ℝ := -1
  let area_of_triangle : ℝ := 1 / 2 * tangent_intercept_x * -tangent_intercept_y
  area_of_triangle = 1 / 2 :=
by
  sorry

end area_of_triangle_tangent_line_l368_368197


namespace pascal_triangle_elements_count_l368_368268

theorem pascal_triangle_elements_count :
  ∑ n in finset.range 30, (n + 1) = 465 :=
by 
  sorry

end pascal_triangle_elements_count_l368_368268


namespace sin_shift_right_pi_over_3_l368_368348

theorem sin_shift_right_pi_over_3 (x : ℝ) :
  (∃ f : ℝ → ℝ, f = (λ x, sin (2 * x))) → 
  (∃ g : ℝ → ℝ, g = (λ x, sin (2 * x - 2 * (π / 3)))) := 
by
  sorry

end sin_shift_right_pi_over_3_l368_368348


namespace pascal_triangle_elements_count_l368_368264

theorem pascal_triangle_elements_count :
  ∑ n in finset.range 30, (n + 1) = 465 :=
by 
  sorry

end pascal_triangle_elements_count_l368_368264


namespace speed_of_stream_l368_368105

theorem speed_of_stream 
  (swim_speed_still : ℝ)
  (time_upstream_time_downstream_ratio : ℝ)
  (v : ℝ) 
  (h_swim_speed : swim_speed_still = 12)
  (h_time_ratio : time_upstream_time_downstream_ratio = 2) :
  (∀ t : ℝ, (swim_speed_still + v) * t = (swim_speed_still - v) * (time_upstream_time_downstream_ratio * t)) → 
  v = 4 :=
by 
  assume h
  sorry

end speed_of_stream_l368_368105


namespace conjugate_of_z_l368_368240

noncomputable def z := (1 + 2 * Complex.I) / 2 * (1 + Complex.I) ^ 2 * Complex.I

theorem conjugate_of_z :
  Complex.conj z = -2 - Complex.I := by
  sorry

end conjugate_of_z_l368_368240


namespace intersecting_rectangles_area_l368_368049

-- Define the dimensions of the rectangles
def rect1_length : ℝ := 12
def rect1_width : ℝ := 4
def rect2_length : ℝ := 7
def rect2_width : ℝ := 5

-- Define the areas of the individual rectangles
def area_rect1 : ℝ := rect1_length * rect1_width
def area_rect2 : ℝ := rect2_length * rect2_width

-- Assume overlapping region area
def area_overlap : ℝ := rect1_width * rect2_width

-- Define the total shaded area
def shaded_area : ℝ := area_rect1 + area_rect2 - area_overlap

-- Prove the shaded area is 63 square units
theorem intersecting_rectangles_area : shaded_area = 63 :=
by 
  -- Insert proof steps here, we only provide the theorem statement and leave the proof unfinished
  sorry

end intersecting_rectangles_area_l368_368049


namespace total_time_taken_l368_368515

theorem total_time_taken (speed_boat : ℕ) (speed_stream : ℕ) (distance : ℕ) 
    (h1 : speed_boat = 12) (h2 : speed_stream = 4) (h3 : distance = 480) : 
    ((distance / (speed_boat + speed_stream)) + (distance / (speed_boat - speed_stream)) = 90) :=
by
  -- Sorry is used to skip the proof
  sorry

end total_time_taken_l368_368515


namespace tangent_line_at_point_l368_368000

noncomputable def tangent_line_equation (x : ℝ) : Prop :=
  ∀ y : ℝ, y = x * (3 * Real.log x + 1) → (x = 1 ∧ y = 1) → y = 4 * x - 3

theorem tangent_line_at_point : tangent_line_equation 1 :=
sorry

end tangent_line_at_point_l368_368000


namespace tangent_line_to_curve_at_point_l368_368003

-- Define the function
def f (x : ℝ) : ℝ := x * (3 * Real.log x + 1)

-- The definition of the point
def point : (ℝ × ℝ) := (1, 1)

-- The equation of the tangent line at the given point
def tangent_line_eq (x : ℝ) : ℝ := 4 * x - 3

theorem tangent_line_to_curve_at_point :
  ∀ (x y : ℝ), (x = 1 ∧ y = 1) → (f x = y) → ∀ (t : ℝ), tangent_line_eq t = 4 * t - 3 := by
  assume x y hxy hfx t
  sorry

end tangent_line_to_curve_at_point_l368_368003


namespace relationship_above_l368_368827

noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 15 / (2 * Real.log 2)
noncomputable def c : ℝ := Real.sqrt 2

theorem relationship_above (ha : a = Real.log 5 / Real.log 2) 
                           (hb : b = Real.log 15 / (2 * Real.log 2))
                           (hc : c = Real.sqrt 2) : a > b ∧ b > c :=
by
  sorry

end relationship_above_l368_368827


namespace ratio_of_areas_l368_368126

-- Define the areas of the equilateral triangles and the trapezoid
noncomputable def area_equilateral (side_length : ℝ) : ℝ :=
  (sqrt 3 / 4) * side_length^2

noncomputable def large_triangle_side : ℝ := 8
noncomputable def small_triangle_side : ℝ := 4

noncomputable def large_triangle_area : ℝ :=
  area_equilateral large_triangle_side
noncomputable def small_triangle_area : ℝ :=
  area_equilateral small_triangle_side

noncomputable def trapezoid_area : ℝ :=
  large_triangle_area - small_triangle_area

-- Statement of the proof problem
theorem ratio_of_areas :
  (small_triangle_area / trapezoid_area) = 1 / 3 :=
by
  sorry

end ratio_of_areas_l368_368126


namespace sector_area_l368_368344

theorem sector_area (radius area : ℝ) (θ : ℝ) (h1 : 2 * radius + θ * radius = 16) (h2 : θ = 2) : area = 16 :=
  sorry

end sector_area_l368_368344


namespace at_most_two_greater_than_one_l368_368814

theorem at_most_two_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ¬ (2 * a - 1 / b > 1 ∧ 2 * b - 1 / c > 1 ∧ 2 * c - 1 / a > 1) :=
by
  sorry

end at_most_two_greater_than_one_l368_368814


namespace isabella_hair_growth_l368_368798

theorem isabella_hair_growth :
  ∀ (initial final : ℤ), initial = 18 → final = 24 → final - initial = 6 :=
by
  intros initial final h_initial h_final
  rw [h_initial, h_final]
  exact rfl
-- sorry

end isabella_hair_growth_l368_368798


namespace sum_theta_3_6_9_12_l368_368029

noncomputable def complex_numbers := {z : ℂ // z^12 - z^6 - 1 = 0 ∧ complex.abs z = 1}

noncomputable def z_k (k : ℕ) (h : 1 ≤ k ∧ k ≤ 12) : ℂ := sorry -- function to get the z_k of cos(θ_k) + i*sin(θ_k) for kth complex number

noncomputable def theta_k (k : ℕ) (h : 1 ≤ k ∧ k ≤ 12) : ℝ :=
  real.angle.to_degrees (complex.arg (z_k k h))

theorem sum_theta_3_6_9_12 : 
  θ3 + θ6 + θ9 + θ12 = 840 :=
begin
  sorry
end

end sum_theta_3_6_9_12_l368_368029


namespace house_orderings_l368_368922

def house := ℕ  -- Represent houses by natural numbers, assuming 1 to 5 are the houses.

variables (green violet indigo teal black : house) 
          (orderings : list (house × house × house × house × house))
          (valid_orderings : list (house × house × house × house × house)) 
          (number_of_valid_orderings : ℕ)

-- Definitions of the constraints
def green_before_violet : Prop := green < violet
def indigo_before_teal : Prop := indigo < teal
def indigo_not_next_teal : Prop := (indigo + 1 ≠ teal) ∧ (indigo - 1 ≠ teal)
def green_not_next_violet : Prop := (green + 1 ≠ violet) ∧ (green - 1 ≠ violet)
def all_different_colors : Prop := list.nodup [green, violet, indigo, teal, black]

-- Prove the number of valid orderings is 4
theorem house_orderings : 
  (green_before_violet ∧ indigo_before_teal ∧ indigo_not_next_teal ∧ green_not_next_violet ∧ all_different_colors) → 
  number_of_valid_orderings = 4 :=
sorry

end house_orderings_l368_368922


namespace lambda_3_sufficient_not_necessary_l368_368215

-- Define the vectors
def a (λ : ℝ) : ℝ × ℝ := (3, λ)
def b (λ : ℝ) : ℝ × ℝ := (λ - 1, 2)

-- Define the parallel condition
def are_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

-- Define the specific lambda value
def λ₃ := 3

-- State the problem about sufficient but not necessary condition
theorem lambda_3_sufficient_not_necessary (λ : ℝ) :
  (are_parallel (a λ) (b λ)) -> λ₃ = 3 ∨ λ₃ = -2 :=
sorry

end lambda_3_sufficient_not_necessary_l368_368215


namespace maximize_g_l368_368200

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem maximize_g : ∃ x ∈ Set.Icc (0 : ℝ) (Real.root 4 4), g x = 4 :=
by
  intros
  let root4 := Real.root 4 4
  have h_g_bound : ∀ x ∈ Set.Icc (0 : ℝ) root4, g x ≤ 4 := sorry
  exact ⟨(0 : ℝ), And.intro (le_refl 0) (Real.root_nonneg 4 (by norm_num)), h_g_bound 0 (by norm_num)⟩

end maximize_g_l368_368200


namespace total_cookies_prepared_l368_368985

-- Definition of conditions
def cookies_per_guest : ℕ := 19
def number_of_guests : ℕ := 2

-- Theorem statement
theorem total_cookies_prepared : (cookies_per_guest * number_of_guests) = 38 :=
by
  sorry

end total_cookies_prepared_l368_368985


namespace pascal_triangle_rows_sum_l368_368316

theorem pascal_triangle_rows_sum :
  ∑ k in finset.range 30, (k + 1) = 465 := by
  sorry

end pascal_triangle_rows_sum_l368_368316


namespace find_a_value_l368_368728

theorem find_a_value (a m : ℝ) (f g : ℝ → ℝ)
  (h₁ : 0 < a) 
  (h₂ : a ≠ 1) 
  (h₃ : ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x = a ^ x) 
  (h₄ : ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≤ 4) 
  (h₅ : ∃ x ∈ Set.Icc (-1 : ℝ) 2, f x = 4) 
  (h₆ : ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≥ m) 
  (h₇ : ∃ x ∈ Set.Icc (-1 : ℝ) 2, f x = m) 
  (h₈ : ∀ x, g x = (1 - 4 * m) * Real.sqrt x) 
  (h_g_inc : ∀ x y : ℝ, 0 ≤ x → x ≤ y → g x ≤ g y) :
  a = 1 / 4 :=
begin
  sorry
end

end find_a_value_l368_368728


namespace find_value_of_expression_l368_368717

theorem find_value_of_expression (a : ℝ) (h : a^2 + 3 * a - 1 = 0) : 2 * a^2 + 6 * a + 2021 = 2023 := 
by
  sorry

end find_value_of_expression_l368_368717


namespace imaginary_part_conjugate_z_l368_368726

noncomputable def z : ℂ := complex.i ^ 2018 - (complex.abs (3 - 4 * complex.i) / (3 - 4 * complex.i))

theorem imaginary_part_conjugate_z : complex.im (conj z) = 4 / 5 :=
by
  sorry

end imaginary_part_conjugate_z_l368_368726


namespace parabola_y_values_order_l368_368713

theorem parabola_y_values_order :
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  -- The proof is omitted
  sorry

end parabola_y_values_order_l368_368713


namespace smallest_five_consecutive_even_sum_320_l368_368899

theorem smallest_five_consecutive_even_sum_320 : ∃ (a b c d e : ℤ), a + b + c + d + e = 320 ∧ (∀ i j : ℤ, (i = a ∨ i = b ∨ i = c ∨ i = d ∨ i = e) → (j = a ∨ j = b ∨ j = c ∨ j = d ∨ j = e) → (i = j + 2 ∨ i = j - 2 ∨ i = j)) ∧ (a ≤ b ∧ a ≤ c ∧ a ≤ d ∧ a ≤ e) ∧ a = 60 :=
by
  sorry

end smallest_five_consecutive_even_sum_320_l368_368899


namespace anna_dinner_cost_l368_368129

def calculate_pre_tax_tip_cost (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  total_cost / (1 + tax_rate + tip_rate)

theorem anna_dinner_cost :
  calculate_pre_tax_tip_cost 30 0.08 0.20 = 23.44 :=
by 
  unfold calculate_pre_tax_tip_cost
  norm_num

end anna_dinner_cost_l368_368129


namespace area_prob_thm_l368_368127

-- Define the isosceles right triangle ABC
structure Triangle :=
  (AB AC BC : ℝ)
  (right_angle_C : AB = AC ∧ BC = AB * Real.sqrt 2)

-- Define the area computation for a triangle
def triangle_area (AB AC : ℝ) : ℝ :=
  1 / 2 * AB * AC

-- Define the probability computation for point within the triangle
noncomputable def area_probability (T : Triangle) (P : ℝ) : ℝ :=
  if (1 / 2 * T.BC * P) < 1 / 3 * (triangle_area T.AB T.AC) then
    sorry -- Skip the computation step
  else
    sorry -- Skip the computation step

-- Given T : Triangle, compute and prove the probability
theorem area_prob_thm (T : Triangle) (h_T : T.AB = 8 ∧ T.AC = 8 ∧ T.BC = 8 * Real.sqrt 2) : 
  area_probability T (8 * Real.sqrt 2 / 3) = 2 / 9 :=
sorry -- proof part is omitted

end area_prob_thm_l368_368127


namespace largest_of_A_B_C_l368_368934

-- Define the quantities A, B, and C based on the problem statement
def A : ℝ := (2010 / 2009) + (2010 / 2011)
def B : ℝ := (2010 / 2011) * (2012 / 2011)
def C : ℝ := (2011 / 2010) + (2011 / 2012) + (1 / 10000)

-- State the theorem to prove
theorem largest_of_A_B_C : C > A ∧ C > B :=
by 
  sorry

end largest_of_A_B_C_l368_368934


namespace decreasing_on_neg_interval_l368_368950

noncomputable def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y : ℝ, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

variables (f : ℝ → ℝ)
variables (h_even : is_even f)
variables (h_inc : is_increasing_on f 1 3)
variables (h_min : ∀ x : ℝ, f x ≥ 0 ∧ (∀ y : ℝ, y ≠ 0 → f y ≠ 0))

theorem decreasing_on_neg_interval (f : ℝ → ℝ) 
(h_even : is_even f)
(h_inc : is_increasing_on f 1 3)
(h_min : ∀ x : ℝ, f x ≥ 0 ∧ (∀ y : ℝ, y ≠ 0 → f y ≠ 0)) :
is_decreasing_on f (-3) (-1) ∧ (∀ x : ℝ, -3 ≤ x → x ≤ -1 → f x ≥ 0) :=
begin
  sorry
end

end decreasing_on_neg_interval_l368_368950


namespace factor_polynomial_l368_368190

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end factor_polynomial_l368_368190


namespace min_balls_to_ensure_fifteen_same_color_l368_368586

theorem min_balls_to_ensure_fifteen_same_color (r g y b w bl : ℕ) (total : ℕ) (h: total = 100) :
  r = 28 ∧ g = 20 ∧ y = 13 ∧ b = 19 ∧ w = 11 ∧ bl = 9 →
  (∃ n, n >= 76 ∧ (∀ (draws : finset (fin total)), draws.card = n → ∃ c ∈ draws.toList, (draws.toList.filter (λ x, c = x)).length ≥ 15)) :=
begin
  sorry,
end

end min_balls_to_ensure_fifteen_same_color_l368_368586


namespace min_value_expression_l368_368837

theorem min_value_expression (k x y z : ℝ) (hk : 0 < k) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ x_min y_min z_min : ℝ, (0 < x_min) ∧ (0 < y_min) ∧ (0 < z_min) ∧
  (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
    k * (4 * z / (2 * x + y) + 4 * x / (y + 2 * z) + y / (x + z))
    ≥ 3 * k) ∧
  k * (4 * z_min / (2 * x_min + y_min) + 4 * x_min / (y_min + 2 * z_min) + y_min / (x_min + z_min)) = 3 * k :=
by sorry

end min_value_expression_l368_368837


namespace pascal_triangle_rows_sum_l368_368318

theorem pascal_triangle_rows_sum :
  ∑ k in finset.range 30, (k + 1) = 465 := by
  sorry

end pascal_triangle_rows_sum_l368_368318


namespace seating_arrangement_l368_368366

theorem seating_arrangement (people : Finset ℕ) (Wilma Paul Harry Sally : ℕ)
  (h_wilma : Wilma ∈ people) (h_paul : Paul ∈ people)
  (h_harry : Harry ∈ people) (h_sally : Sally ∈ people)
  (h_card : people.card = 8) :
  (number_of_valid_arrangements people Wilma Paul Harry Sally) = 23040 := 
sorry

noncomputable def number_of_valid_arrangements (people : Finset ℕ) (Wilma Paul Harry Sally : ℕ) : ℕ := 
-- Here we would define the function based on the conditions,
-- though the actual implementation is omitted
0 -- Placeholder implementation

end seating_arrangement_l368_368366


namespace fewer_onions_correct_l368_368643

-- Define the quantities
def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

-- Calculate the total number of tomatoes and corn
def tomatoes_and_corn : ℕ := tomatoes + corn

-- Calculate the number of fewer onions
def fewer_onions : ℕ := tomatoes_and_corn - onions

-- State the theorem and provide the proof
theorem fewer_onions_correct : fewer_onions = 5200 :=
by
  -- The statement is proved directly by the calculations above
  -- Providing the actual proof is not necessary as per the guidelines
  sorry

end fewer_onions_correct_l368_368643


namespace total_roses_after_picks_l368_368962

def initial_roses : ℝ := 37.0
def first_pick : ℝ := 16.0
def second_pick : ℝ := 19.0

theorem total_roses_after_picks : initial_roses + first_pick + second_pick = 72.0 :=
by
  calc initial_roses + first_pick + second_pick
    = 37.0 + 16.0 + 19.0 : by refl
    ... = 53.0 + 19.0 : by norm_num
    ... = 72.0 : by norm_num

end total_roses_after_picks_l368_368962


namespace cos_150_eq_neg_sqrt3_2_l368_368905

theorem cos_150_eq_neg_sqrt3_2 : cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  have cos_diff_identity : ∀ θ : ℝ, cos (Real.pi - θ) = - cos θ := sorry
  have cos_30 : cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  sorry

end cos_150_eq_neg_sqrt3_2_l368_368905


namespace max_number_of_squares_l368_368042

theorem max_number_of_squares (points : finset (fin 12)) : 
  ∃ (sq_count : ℕ), sq_count = 11 := 
sorry

end max_number_of_squares_l368_368042


namespace parabola_has_one_x_intercept_l368_368663

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- State the theorem that proves the number of x-intercepts
theorem parabola_has_one_x_intercept : ∃! x, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
by
  -- Proof goes here, but it's omitted
  sorry

end parabola_has_one_x_intercept_l368_368663


namespace Laurent_number_greater_than_Chloe_l368_368144

noncomputable def probability_sum_greater_than : ℝ :=
  let area_total := 1000 * 2500
  let integral_value := (2_000_000_000 : ℝ) / 3
  integral_value / area_total

theorem Laurent_number_greater_than_Chloe :
  probability_sum_greater_than = 0.8 + 2 / 3 :=
by
  sorry

end Laurent_number_greater_than_Chloe_l368_368144


namespace distance_from_Asheville_to_Darlington_l368_368978

theorem distance_from_Asheville_to_Darlington (BC AC BD AD : ℝ) 
(h0 : BC = 12) 
(h1 : BC = (1/3) * AC) 
(h2 : BC = (1/4) * BD) :
AD = 72 :=
sorry

end distance_from_Asheville_to_Darlington_l368_368978


namespace num_and_sum_of_divisors_of_36_l368_368329

noncomputable def num_divisors_and_sum (n : ℕ) : ℕ × ℕ :=
  let divisors := (List.range (n + 1)).filter (λ x => n % x = 0)
  (divisors.length, divisors.sum)

theorem num_and_sum_of_divisors_of_36 : num_divisors_and_sum 36 = (9, 91) := by
  sorry

end num_and_sum_of_divisors_of_36_l368_368329


namespace evaluate_limit_l368_368815

noncomputable def a_n (n : ℕ) : ℝ :=
  ∫ x in ((n-1) * Real.pi)..(n * Real.pi), Real.exp (-x) * (1 - |Real.cos x|)

theorem evaluate_limit :
  (Real.NormedSpace normed_field ℝ) →
  ∑' n, a_n n = 1 / 2 :=
by
  sorry

end evaluate_limit_l368_368815


namespace inner_square_area_l368_368467

theorem inner_square_area (side_ABCD : ℝ) (dist_BI : ℝ) (area_IJKL : ℝ) :
  side_ABCD = Real.sqrt 72 →
  dist_BI = 2 →
  area_IJKL = 39 :=
by
  sorry

end inner_square_area_l368_368467


namespace parabola_y_values_order_l368_368714

theorem parabola_y_values_order :
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  -- The proof is omitted
  sorry

end parabola_y_values_order_l368_368714


namespace remainder_101_mul_103_mod_11_l368_368549

theorem remainder_101_mul_103_mod_11 : (101 * 103) % 11 = 8 :=
by
  sorry

end remainder_101_mul_103_mod_11_l368_368549


namespace sqrt_22_gt_4_l368_368147

theorem sqrt_22_gt_4 : Real.sqrt 22 > 4 := 
sorry

end sqrt_22_gt_4_l368_368147


namespace max_min_points_for_E_l368_368688

section FootballTournament

-- Definitions for teams and matches
def Team := ℕ

def match_result := ℕ → ℕ → ℕ

-- Points conditions
axiom points_A : ℕ := 1
axiom points_B : ℕ := 4
axiom points_C : ℕ := 7
axiom points_D : ℕ := 8

-- Number of points in the tournament with T matches
def total_matches (n : ℕ) := (n * (n - 1)) / 2

def total_points (matches : ℕ) (draws : ℕ) := 3 * matches - draws

def points_for_team_E (total_points : ℕ) := total_points - (points_A + points_B + points_C + points_D)

def maximum_points_for_E (matches : ℕ) (draws : ℕ) := points_for_team_E $ total_points matches draws

def minimum_points_for_E (total_points : ℕ) := points_for_team_E total_points

-- Theorem for maximum and minimum points for team E
theorem max_min_points_for_E (matches draws_points : ℕ) : 
  (total_matches 5 = matches) →
  (draws_points = 30 - matches) →
  (3 ≤ draws_points ∧ draws_points <= 5) →
  (maximum_points_for_E matches 3 = 7) ∧ (minimum_points_for_E (30 - 5) = 5) := by
  sorry

end FootballTournament

end max_min_points_for_E_l368_368688


namespace problem1_problem2_problem3_problem4_l368_368651

theorem problem1 : (-3 + 8 - 7 - 15) = -17 := 
sorry

theorem problem2 : (23 - 6 * (-3) + 2 * (-4)) = 33 := 
sorry

theorem problem3 : (-8 / (4 / 5) * (-2 / 3)) = 20 / 3 := 
sorry

theorem problem4 : (-2^2 - 9 * (-1 / 3)^2 + abs (-4)) = -1 := 
sorry

end problem1_problem2_problem3_problem4_l368_368651


namespace fewer_onions_than_tomatoes_and_corn_l368_368641

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions_than_tomatoes_and_corn :
  (tomatoes + corn - onions) = 5200 :=
by
  sorry

end fewer_onions_than_tomatoes_and_corn_l368_368641


namespace grape_juice_percentage_l368_368339

theorem grape_juice_percentage
    (initial_volume : ℕ)
    (initial_percentage : ℕ)
    (added_volume : ℕ) :
    initial_volume = 40 →
    initial_percentage = 10 →
    added_volume = 20 →
    ((initial_percentage * initial_volume / 100 + added_volume) * 100 / 
    (initial_volume + added_volume) = 40) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end grape_juice_percentage_l368_368339


namespace negation_of_proposition_l368_368255

theorem negation_of_proposition :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ (1 / 2 - 2 ^ (-x₀) = 5 / 8)) ↔ (∀ x : ℝ, x > 0 → 1 / 2 - 2 ^ (-x) ≠ 5 / 8) :=
by
  sorry

end negation_of_proposition_l368_368255


namespace tangent_lines_circle_l368_368740

theorem tangent_lines_circle (a : ℝ) :
  let l1 := λ x y : ℝ, 2 * x - y + a
  let l2 := λ x y : ℝ, 2 * x - y + a^2 + 1
  let circle := λ x y : ℝ, x^2 + y^2 + 2 * x - 4 = 0
  let dist := λ x₀ y₀ A B C, abs (A * x₀ + B * y₀ + C) / sqrt (A ^ 2 + B ^ 2)
  let center_x := -1
  let center_y := 0
  let radius := sqrt 5
  let dist_l1 := dist center_x center_y 2 -1 a
  let dist_l2 := dist center_x center_y 2 -1 (a^2 + 1)
  dist_l1 = radius ∧ dist_l2 = radius → 
  (-3 ≤ a ∧ a ≤ -sqrt 6) ∨ (sqrt 6 ≤ a ∧ a ≤ 7) :=
by
  sorry

end tangent_lines_circle_l368_368740


namespace regular_tetrahedron_of_angle_l368_368500

-- Definition and condition from the problem
def angle_between_diagonals (shape : Type _) (adj_sides_diag_angle : ℝ) : Prop :=
  adj_sides_diag_angle = 60

-- Theorem stating the problem in Lean 4
theorem regular_tetrahedron_of_angle (shape : Type _) (adj_sides_diag_angle : ℝ) 
  (h : angle_between_diagonals shape adj_sides_diag_angle) : 
  shape = regular_tetrahedron :=
sorry

end regular_tetrahedron_of_angle_l368_368500


namespace find_radius_original_bubble_l368_368616

-- Define the given radius of the hemisphere.
def radius_hemisphere : ℝ := 4 * real.cbrt 2

-- Define the volume of a hemisphere.
def vol_hemisphere (r : ℝ) := (2 / 3) * π * r^3

-- Define the volume of a sphere.
def vol_sphere (R : ℝ) := (4 / 3) * π * R^3

-- Given condition: The volume of the hemisphere is equal to the volume of the original bubble.
def volume_equivalence (r R : ℝ) := vol_sphere R = vol_hemisphere r

-- State the theorem to prove: Given the radius of the hemisphere, find the radius of the original bubble.
theorem find_radius_original_bubble (R : ℝ) (h : volume_equivalence radius_hemisphere R) : R = 4 :=
sorry

end find_radius_original_bubble_l368_368616


namespace range_of_abc_l368_368707

noncomputable def real_numbers (a b c : ℝ) (f : ℝ → ℝ) :=
  ∃ b c : ℝ, b^2 + c^2 = 1 ∧
  (∃ g : ℝ → ℝ, ∃ h : ℝ → ℝ, 
      (∃ m n : ℝ, g(m) * h(n) = -1) ∧
      (f = λ x, ax + b * Real.sin x + c * Real.cos x))

theorem range_of_abc (a b c : ℝ) (f : ℝ → ℝ) :
  real_numbers a b c f →
  ∃ r, r = (a + b + c) ∧ -Real.sqrt 2 ≤ r ∧ r ≤ Real.sqrt 2 :=
by
  sorry

end range_of_abc_l368_368707


namespace max_number_of_squares_l368_368044

theorem max_number_of_squares (points : finset (fin 12)) : 
  ∃ (sq_count : ℕ), sq_count = 11 := 
sorry

end max_number_of_squares_l368_368044


namespace equal_likelihood_each_number_not_equal_likelihood_all_selections_l368_368169

def k : ℕ := sorry
def n : ℕ := sorry
def selection : list ℕ := sorry

-- Condition 1: Each subsequent number is chosen independently of the previous ones.
axiom independent_choice (m : ℕ) : m ∈ selection → m < n

-- Condition 2: If it matches one of the already chosen numbers, you should move clockwise to the first unchosen number.
axiom move_clockwise (m : ℕ) : m ∈ selection → ∃ l, l ≠ m ∧ l ∉ selection

-- Condition 3: In the end, k different numbers are obtained.
axiom k_distinct : list.distinct selection ∧ list.length selection = k

-- Define the events A and B
def A : Prop := ∀ m : ℕ, m ∈ selection → ∃ p, probability m = p
def B : Prop := ∀ s1 s2 : list ℕ, s1 ⊆ selection ∧ s1.length = k ∧ s2 ⊆ selection ∧ s2.length = k → s1 ≠ s2 → probability s1 ≠ probability s2

-- Prove A holds given the conditions
theorem equal_likelihood_each_number : A :=
sorry

-- Prove B does not hold given the conditions
theorem not_equal_likelihood_all_selections : ¬B :=
sorry

end equal_likelihood_each_number_not_equal_likelihood_all_selections_l368_368169


namespace frank_spent_on_mower_blades_l368_368209

def money_made := 19
def money_spent_on_games := 4 * 2
def money_left := money_made - money_spent_on_games

theorem frank_spent_on_mower_blades : money_left = 11 :=
by
  -- we are providing the proof steps here in comments, but in the actual code, it's just sorry
  -- calc money_left
  --    = money_made - money_spent_on_games : by refl
  --    = 19 - 8 : by norm_num
  --    = 11 : by norm_num
  sorry

end frank_spent_on_mower_blades_l368_368209


namespace lambda_three_sufficient_not_necessary_l368_368213

-- Define the vectors and the condition of parallelism
def vector_a (λ : ℝ) : ℝ × ℝ := (3, λ)
def vector_b (λ : ℝ) : ℝ × ℝ := (λ - 1, 2)
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)

-- Prove that λ = 3 is a sufficient but not necessary condition for vector_a λ to be parallel to vector_b λ
theorem lambda_three_sufficient_not_necessary (λ : ℝ) :
  parallel (vector_a 3) (vector_b 3) ∧ ∃ λ' : ℝ, λ' ≠ 3 ∧ parallel (vector_a λ') (vector_b λ') :=
by
  sorry

end lambda_three_sufficient_not_necessary_l368_368213


namespace on_perpendicular_ab_l368_368867

variables {O C E D F N A B : Type*}
variables [PointsOnCircle C E D F O]

-- Definitions of the properties given in the conditions
def circle_center (O : Type*) := has_center O
def intersect_chords (N : Type*) (CD EF : chords) := intersects N CD EF
def intersect_tangents (A : Type*) (C D : points) := tangent_intersect A C D
def intersect_tangents' (B : Type*) (E F : points) := tangent_intersect B E F

-- Proof statement
theorem on_perpendicular_ab :
  ∀ (O C E D F N A B : Type*) [PointsOnCircle C E D F O] 
    (h1 : circle_center O) (h2 : intersect_chords N CD EF)
    (h3 : intersect_tangents A C D) (h4 : intersect_tangents' B E F),
    perp ON AB :=
by {
  sorry
}


end on_perpendicular_ab_l368_368867


namespace max_XG_l368_368439

theorem max_XG :
  ∀ (G X Y Z : ℝ),
    Y - X = 5 ∧ Z - Y = 3 ∧ (1 / G + 1 / (G - 5) + 1 / (G - 8) = 0) →
    G = 20 / 3 :=
by
  sorry

end max_XG_l368_368439


namespace find_polynomial_Q_l368_368400

theorem find_polynomial_Q :
  ∃ Q : ℚ[X], (∀ x : ℚ, Q x = Q 0 + Q 1 * x + Q 3 * x^2) ∧ 
              (Q (-1) = 2) ∧ 
              (Q = (λ x, x^2 - x + 2)) := 
sorry

end find_polynomial_Q_l368_368400


namespace BP_eq_CQ_MN_parallel_AD_l368_368418

/-
Let the circumcircle of triangle \( \triangle ABC \) be denoted as \( \odot O \).
The angle bisector of \(\angle BAC\) intersects \(BC\) at point \(D\),
and \(M\) is the midpoint of \(BC\).
If the circumcircle of \(\triangle ADM\), denoted as \(\odot Z\), intersects \(AB\) and \(AC\) at points \(P\) and \(Q\) respectively,
and \(N\) is the midpoint of \(PQ\),
then:
1. Prove \(BP = CQ\).
2. Prove \(MN \parallel AD\).
-/

theorem BP_eq_CQ
  (A B C D M P Q N : Point)
  (O Z : Circle)
  (h1 : O.isCircumcircleOfTriangle A B C)
  (h2 : Z.isCircumcircleOfTriangle A D M)
  (h3 : ∠BAC.bisector_intersect D B C)
  (h4 : midpoint M B C)
  (h5 : Z.intersectAt P A B)
  (h6 : Z.intersectAt Q A C)
  (h7 : midpoint N P Q):
  BP = CQ :=
sorry

theorem MN_parallel_AD
  (A B C D M P Q N : Point)
  (O Z : Circle)
  (h1 : O.isCircumcircleOfTriangle A B C)
  (h2 : Z.isCircumcircleOfTriangle A D M)
  (h3 : ∠BAC.bisector_intersect D B C)
  (h4 : midpoint M B C)
  (h5 : Z.intersectAt P A B)
  (h6 : Z.intersectAt Q A C)
  (h7 : midpoint N P Q):
  MN ∥ AD :=
sorry

end BP_eq_CQ_MN_parallel_AD_l368_368418


namespace average_distance_is_6_l368_368101

noncomputable def avg_distance_from_sides (side_length diag_length : ℝ) 
  (initial_distance_diagonal moved_distance post_turn_distance : ℝ) : ℝ :=
let x_coord := (initial_distance_diagonal / diag_length) * side_length,
    y_coord := (initial_distance_diagonal / diag_length) * side_length,
    new_x_coord := x_coord + post_turn_distance,
    d1 := new_x_coord,
    d2 := y_coord,
    d3 := side_length - new_x_coord,
    d4 := side_length - y_coord in
(d1 + d2 + d3 + d4) / 4

theorem average_distance_is_6 (side_length : ℝ) (initial_distance_diagonal : ℝ) (post_turn_distance : ℝ) :
  avg_distance_from_sides side_length (real.sqrt (side_length ^ 2 + side_length ^ 2)) initial_distance_diagonal post_turn_distance = 6 :=
by
  let side_length := 12
  let initial_distance_diagonal := 7.8
  let post_turn_distance := 3
  let diag_length := real.sqrt (side_length ^ 2 + side_length ^ 2)
  sorry

end average_distance_is_6_l368_368101


namespace proof_intersection_complement_l368_368420

open Set

variable (U : Set ℝ) (A B : Set ℝ)

theorem proof_intersection_complement:
  U = univ ∧ A = {x | -1 < x ∧ x ≤ 5} ∧ B = {x | x < 2} →
  A ∩ (U \ B) = {x | 2 ≤ x ∧ x ≤ 5} :=
by
  intros h
  rcases h with ⟨hU, hA, hB⟩
  simp [hU, hA, hB]
  sorry

end proof_intersection_complement_l368_368420


namespace volume_cube_box_for_pyramid_l368_368852

theorem volume_cube_box_for_pyramid (h_pyramid : height_of_pyramid = 18) 
  (base_side_pyramid : side_of_square_base = 15) : 
  volume_of_box = 18^3 :=
by
  sorry

end volume_cube_box_for_pyramid_l368_368852


namespace smallest_K_exists_l368_368574

theorem smallest_K_exists :
  ∀ (x : ℕ → ℕ → ℝ) (h1 : ∀ i j, 0 ≤ x i j)
    (h2 : ∀ i, (finset.range 25).sum (x i) ≤ 1),
  ∃ K : ℕ, K = 97 ∧ ∀ i, i ≥ K → (finset.range 25).sum (λ j, x i j) ≤ 1 :=
sorry

end smallest_K_exists_l368_368574


namespace largest_real_part_l368_368470

noncomputable def max_re_z_w (z w : ℂ) (hz : abs z = 2) (hw : abs w = 2) (hzw : z * conj w + conj z * w = 1) : ℝ :=
  let a := z.re
  let b := z.im
  let c := w.re
  let d := w.im
  
  (if h1: 2 * a * c + 2 * b * d = 1 then
     (a + c).nat_abs.max (3 - (a + c).nat_abs)
   else 0)

theorem largest_real_part (z w : ℂ) (hz : abs z = 2) (hw : abs w = 2) (hzw : z * conj w + conj z * w = 1) :
  max_re_z_w z w hz hw hzw = 3 := sorry

end largest_real_part_l368_368470


namespace hyperbola_equation_l368_368381

noncomputable def hyperbola (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def parabola_focus_same_as_hyperbola_focus (c : ℝ) : Prop :=
  ∃ x y : ℝ, y^2 = 4 * (10:ℝ).sqrt * x ∧ (c, 0) = ((10:ℝ).sqrt, 0)

def hyperbola_eccentricity (c a : ℝ) := (c / a) = (10:ℝ).sqrt / 3

theorem hyperbola_equation :
  ∃ a b : ℝ, (hyperbola a b) ∧
  (parabola_focus_same_as_hyperbola_focus ((10:ℝ).sqrt)) ∧
  (hyperbola_eccentricity ((10:ℝ).sqrt) a) ∧
  ((a = 3) ∧ (b = 1)) :=
sorry

end hyperbola_equation_l368_368381


namespace pi_times_difference_volumes_l368_368654

-- Definitions for the conditions
variables {h_c h_d : ℝ} -- heights
variables {c_circ d_circ : ℝ} -- circumferences

def r_c := c_circ / (2 * π)
def r_d := d_circ / (2 * π)
def V_c (h_c : ℝ) := π * (r_c ^ 2) * h_c
def V_d (h_d : ℝ) := π * (r_d ^ 2) * h_d

-- Given conditions
axiom chris_conditions : h_c = 12 ∧ c_circ = 10
axiom dana_conditions : h_d = 10 ∧ d_circ = 12

-- The theorem to prove the positive difference multiplied by π is 60
theorem pi_times_difference_volumes : π * ((V_d h_d) - (V_c h_c)) = 60 := by
  sorry

end pi_times_difference_volumes_l368_368654


namespace find_positive_integer_x_l368_368681

theorem find_positive_integer_x :
  ∃ x : ℕ, x > 0 ∧ (5 * x + 1) / (x - 1) > 2 * x + 2 ∧
  ∀ y : ℕ, y > 0 ∧ (5 * y + 1) / (y - 1) > 2 * x + 2 → y = 2 :=
sorry

end find_positive_integer_x_l368_368681


namespace total_painting_cost_is_12012_l368_368561

noncomputable section

open Real

def areas : List ℝ := [196, 150, 250]
def prices_per_sqft : List ℝ := [15, 18, 20]
def labor_cost : ℝ := 800
def tax_rate : ℝ := 0.05

theorem total_painting_cost_is_12012 :
  (areas.zip prices_per_sqft).sum (λ x, x.1 * x.2) + labor_cost * (1 + tax_rate) = 12012 :=
by
  -- This is where we would provide the proof, but here we state the theorem
  sorry

end total_painting_cost_is_12012_l368_368561


namespace common_tangent_sum_l368_368399

-- Definitions of the two parabolas
def P1 (x : ℝ) : ℝ := 2 * x^2 + (125 / 100)
def P2 (y : ℝ) : ℝ := 2 * y^2 + (65 / 4)

-- Definition of the line equation ax + by = c
def tangent_line (a b c x y : ℝ) : Prop := a * x + b * y = c

-- Main theorem stating the conditions and proving the desired result
theorem common_tangent_sum (a b c : ℕ) (h_slope_rational : ∃ q : ℚ, q = b / a) 
  (h_tangent_P1 : ∀ x : ℝ, tangent_line a b c x (P1 x)) 
  (h_tangent_P2 : ∀ y : ℝ, tangent_line a b c (P2 y) y) 
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  a + b + c = 289 :=
sorry

end common_tangent_sum_l368_368399


namespace PQR_product_l368_368691

def PQR_condition (P Q R S : ℕ) : Prop :=
  P + Q + R + S = 100 ∧
  ∃ x : ℕ, P = x - 4 ∧ Q = x + 4 ∧ R = x / 4 ∧ S = 4 * x

theorem PQR_product (P Q R S : ℕ) (h : PQR_condition P Q R S) : P * Q * R * S = 61440 :=
by 
  sorry

end PQR_product_l368_368691


namespace triangle_OPQ_rotation_l368_368038

open Real

noncomputable def rotate_90_deg_ccw (x y : ℝ) : ℝ × ℝ :=
  (-y, x)

theorem triangle_OPQ_rotation
  (Q : ℝ × ℝ)
  (h_angle_PQO : ∠PQO = 90)
  (h_angle_POQ : ∠POQ = 45)
  (h_P : P = (7, 0))
  (h_O : O = (0, 0))
  (h_Q_in_first_quadrant : Q.1 > 0 ∧ Q.2 > 0) :
  let Q_new := rotate_90_deg_ccw Q.1 Q.2 in
  Q_new = (-7 * sqrt 2 / 2, 7 * sqrt 2 / 2) := 
sorry

end triangle_OPQ_rotation_l368_368038


namespace A_job_days_l368_368588

noncomputable def job_days_for_A (x : ℝ) : Prop :=
  -- Conditions
  let work_rate_B := 1 / 20 in
  let work_rate_A := 1 / x in
  let combined_work_rate := work_rate_A + work_rate_B in
  -- Equation formulating the fraction of work left after 4 days
  4 * combined_work_rate = 0.4666666666666667

theorem A_job_days : job_days_for_A 15 :=
  by {
    -- Proof goes here
    sorry
  }

end A_job_days_l368_368588


namespace sine_graph_transformation_l368_368917

def transform_sine_graph (x : ℝ) : Prop :=
  ∀ y : ℝ, y = sin (x + π/4) → (y = sin (3*x + π/4))

theorem sine_graph_transformation :
  transform_sine_graph (x / 3) = transform_sine_graph (x) :=
by 
  sorry

end sine_graph_transformation_l368_368917


namespace planting_methods_l368_368977

theorem planting_methods (n : ℕ) (types : ℕ) (adjacency_constraint : ℕ) (unique_plants : ℕ) : 
  (n = 11) ∧ (types = 4) ∧ (adjacency_constraint = 4) ∧ (unique_plants = 1) → 
  number_of_planting_methods(n, types, adjacency_constraint, unique_plants) = 4224 :=
begin
  sorry, -- proof is omitted
end

end planting_methods_l368_368977


namespace probability_one_card_each_l368_368629

-- Define the total number of cards
def total_cards := 12

-- Define the number of cards from Adrian
def adrian_cards := 7

-- Define the number of cards from Bella
def bella_cards := 5

-- Calculate the probability of one card from each cousin when selecting two cards without replacement
theorem probability_one_card_each :
  (adrian_cards / total_cards) * (bella_cards / (total_cards - 1)) +
  (bella_cards / total_cards) * (adrian_cards / (total_cards - 1)) =
  35 / 66 := sorry

end probability_one_card_each_l368_368629


namespace solve_system_l368_368463

noncomputable def log_base_sqrt_3 (z : ℝ) : ℝ := Real.log z / Real.log (Real.sqrt 3)

theorem solve_system :
  ∃ x y : ℝ, (3^x * 2^y = 972) ∧ (log_base_sqrt_3 (x - y) = 2) ∧ (x = 5 ∧ y = 2) :=
by
  sorry

end solve_system_l368_368463


namespace lineup_condition1_lineup_condition2_lineup_condition3_l368_368907

-- Define the problem for Condition 1
theorem lineup_condition1 (total_positions : ℕ) (middle_positions : ℕ) (positions_to_choose : ℕ) (ways_middle : ℕ) (ways_remaining : ℕ) 
  (total_ways : ℕ) :
  total_positions = 7 → middle_positions = 5 → positions_to_choose = 2 →
  ways_middle * ways_remaining = total_ways → ways_middle = (middle_positions !)/(positions_to_choose ! * (middle_positions - positions_to_choose) !) →
  ways_remaining = (middle_positions !)/((middle_positions - way_remaining)!) → 
  total_ways = 2400 :=
by {
  sorry
}

-- Define the problem for Condition 2
theorem lineup_condition2 (boys : ℕ) (girls : ℕ) (ways_boys : ℕ) (ways_girls : ℕ) (ways_units : ℕ) (total_ways : ℕ) :
  boys = 3 → girls = 4 → 
  ways_boys = boys ! → ways_girls = girls ! →
  ways_units = 2 ! →
  total_ways = ways_boys * ways_girls * ways_units →
  total_ways = 288 :=
by {
  sorry
}

-- Define the problem for Condition 3
theorem lineup_condition3 (girls : ℕ) (positions_for_boys : ℕ) (ways_girls : ℕ) (ways_boys : ℕ) (total_ways : ℕ) :
  girls = 4 → positions_for_boys = girls + 1 →
  ways_girls = girls ! → ways_boys = (positions_for_boys !)/((positions_for_boys - 3)!) →
  total_ways = ways_girls * ways_boys →
  total_ways = 1440 :=
by {
  sorry
}

end lineup_condition1_lineup_condition2_lineup_condition3_l368_368907


namespace math_problem_l368_368487

theorem math_problem (c d : ℝ) (hc : c^2 - 6 * c + 15 = 27) (hd : d^2 - 6 * d + 15 = 27) (h_cd : c ≥ d) : 
  3 * c + 2 * d = 15 + Real.sqrt 21 :=
by
  sorry

end math_problem_l368_368487


namespace sum_of_solutions_l368_368064

noncomputable def abs (x : ℝ) : ℝ := if x >= 0 then x else -x

theorem sum_of_solutions :
  (∑ x in { y : ℝ | y = abs (2 * y - abs (100 - 2 * y)) }.to_finset, id x) = 460 / 3 :=
by
  sorry

end sum_of_solutions_l368_368064


namespace eggs_given_by_Andrew_l368_368260

variable (total_eggs := 222)
variable (eggs_to_buy := 67)
variable (eggs_given : ℕ)

theorem eggs_given_by_Andrew :
  eggs_given = total_eggs - eggs_to_buy ↔ eggs_given = 155 := 
by 
  sorry

end eggs_given_by_Andrew_l368_368260


namespace proof_expression_value_l368_368154

noncomputable def a : ℝ := 0.15
noncomputable def b : ℝ := 0.06
noncomputable def x : ℝ := a^3
noncomputable def y : ℝ := b^3
noncomputable def z : ℝ := a^2
noncomputable def w : ℝ := b^2

theorem proof_expression_value :
  ( (x - y) / (z + w) ) + 0.009 + w^4 = 0.1300341679616 := sorry

end proof_expression_value_l368_368154


namespace fish_count_l368_368912

theorem fish_count (T : ℕ) :
  (T > 10 ∧ T ≤ 18) ∧ ((T > 18 ∧ T > 15 ∧ ¬(T > 10)) ∨ (¬(T > 18) ∧ T > 15 ∧ T > 10) ∨ (T > 18 ∧ ¬(T > 15) ∧ T > 10)) →
  T = 16 ∨ T = 17 ∨ T = 18 :=
sorry

end fish_count_l368_368912


namespace pascal_triangle_rows_sum_l368_368315

theorem pascal_triangle_rows_sum :
  ∑ k in finset.range 30, (k + 1) = 465 := by
  sorry

end pascal_triangle_rows_sum_l368_368315


namespace max_students_proof_l368_368471

noncomputable def max_students_in_auditorium : ℕ :=
∑ i in Finset.range 15, (11 + 2 * (i + 1)) / 3

theorem max_students_proof :
  max_students_in_auditorium = 116 :=
by
  sorry

end max_students_proof_l368_368471


namespace find_interest_rate_l368_368600

variable (P : ℝ) (R_B : ℝ) (T : ℝ) (gain_B : ℝ)
variable (R_C : ℝ) (I_A : ℝ) (I_C : ℝ)

def principal : ℝ := 25000
def rate_B : ℝ := 10
def time : ℝ := 3
def gain_B : ℝ := 1125

noncomputable def interest_paid_to_A (P : ℝ) (R_B : ℝ) (T : ℝ) : ℝ :=
  P * R_B * T / 100

noncomputable def interest_received_from_C (P : ℝ) (R_C : ℝ) (T : ℝ) : ℝ :=
  P * R_C * T / 100

theorem find_interest_rate (h1 : I_A = interest_paid_to_A principal rate_B time)
                            (h2 : I_C = I_A + gain_B)
                            (h3 : I_C = interest_received_from_C principal R_C time) :
  R_C = 11.5 := by
  sorry

end find_interest_rate_l368_368600


namespace ratio_of_diagonals_l368_368019

theorem ratio_of_diagonals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (4 * b) / (4 * a) = 11) : (b * Real.sqrt 2) / (a * Real.sqrt 2) = 11 := 
by 
  sorry

end ratio_of_diagonals_l368_368019


namespace part_a_part_b_part_c_part_d_l368_368379

def square (ABCD : set Point) : Prop :=
  ∃ A B C D : Point, is_square ABCD A B C  D

def equilateral_triangle (ABE : set Point) : Prop :=
  ∃ A B E : Point, is_equilateral_triangle A B E

def point_intersection (P : Point) (AC BE : set Point) : Prop :=
  ∃ A C B E : Point, P ∈ AC ∧  P ∈ BE

def symmetric_point (P F : Point) (DC : set Point) : Prop :=
  ∃ D C : Point, is_symmetric DC P F

variables (A B C D E P F : Point) 
variables (ABCD : set Point) (ABE AC BE DC CEF DEF BDF PDF : set Point)

theorem part_a (h1 : square ABCD)
            (h2 : equilateral_triangle ABE)
            (h3 : point_intersection P AC BE)
            (h4 : symmetric_point P F DC) :
            equilateral_triangle CEF :=
sorry

theorem part_b (h1 : square ABCD)
            (h2 : equilateral_triangle ABE)
            (h3 : point_intersection P AC BE)
            (h4 : symmetric_point P F DC) :
            right_angle_isosceles DEF :=
sorry

theorem part_c (h1 : square ABCD)
            (h2 : equilateral_triangle ABE)
            (h3 : point_intersection P AC BE)
            (h4 : symmetric_point P F DC) :
            isosceles_triangle BDF :=
sorry

theorem part_d (h1 : square ABCD)
            (h2 : equilateral_triangle ABE)
            (h3 : point_intersection P AC BE)
            (h4 : symmetric_point P F DC) :
            equilateral_triangle PDF :=
sorry

end part_a_part_b_part_c_part_d_l368_368379


namespace problem_l368_368502

def P (x : ℝ) : Prop := x^2 - 2*x + 1 > 0

theorem problem (h : ¬ ∀ x : ℝ, P x) : ∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0 :=
by {
  sorry
}

end problem_l368_368502


namespace pascal_triangle_sum_l368_368327

theorem pascal_triangle_sum (n : ℕ) (h₀ : n = 29) :
  (∑ k in Finset.range (n + 1), k + 1) = 465 := sorry

end pascal_triangle_sum_l368_368327


namespace valid_license_plates_count_l368_368973

def num_valid_license_plates := (26 ^ 3) * (10 ^ 4)

theorem valid_license_plates_count : num_valid_license_plates = 175760000 :=
by
  sorry

end valid_license_plates_count_l368_368973


namespace area_bounded_by_arccos_sin_l368_368676

theorem area_bounded_by_arccos_sin (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = arccos (sin (x + (π / 6)))) →
  a = 0 →
  b = 2 * π →
  ∫ x in a..b, f x = π^2 :=
by
  sorry

end area_bounded_by_arccos_sin_l368_368676


namespace distance_from_point_to_plane_l368_368081

noncomputable def M1 : ℝ × ℝ × ℝ := (-2, 0, -4)
noncomputable def M2 : ℝ × ℝ × ℝ := (-1, 7, 1)
noncomputable def M3 : ℝ × ℝ × ℝ := (4, -8, -4)
noncomputable def M0 : ℝ × ℝ × ℝ := (-6, 5, 5)

-- This theorem states that the distance from M0 to the plane passing through M1, M2, and M3 
-- is equal to 23 * sqrt(2) / 5.
theorem distance_from_point_to_plane :
  let d := (fun M1 M2 M3 M0 : ℝ × ℝ × ℝ =>
    let normal := λx y z : ℝ, (39, 55, -79)
    let A := 4
    let B := 3
    let C := -5
    let D := -12
    (Real.abs ((-6) * 4 + 5 * 3 + 5 * (-5) + D) / Real.sqrt (A * A + B * B + C * C)) in
  d M1 M2 M3 M0 = 23 * Real.sqrt 2 / 5 := by
  sorry

end distance_from_point_to_plane_l368_368081


namespace tetrahedron_in_spheres_l368_368226

noncomputable def sphere_with_diameter (X Y : Point) : Sphere :=
  sphere_with_radius_center (dist X Y / 2) ((X + Y) / 2)

theorem tetrahedron_in_spheres {A B C D P : Point} :
  (P ∈ tetrahedron A B C D) →
  (P ∈ sphere_with_diameter A B ∪ sphere_with_diameter A C ∪ sphere_with_diameter A D) :=
by
  sorry

end tetrahedron_in_spheres_l368_368226


namespace initial_people_count_l368_368133

-- Definitions from conditions
def initial_people (W : ℕ) : ℕ := W
def net_increase : ℕ := 5 - 2
def current_people : ℕ := 19

-- Theorem to prove: initial_people == 16 given conditions
theorem initial_people_count (W : ℕ) (h1 : W + net_increase = current_people) : initial_people W = 16 :=
by
  sorry

end initial_people_count_l368_368133


namespace correct_reasoning_l368_368948

-- Define that every multiple of 9 is a multiple of 3
def multiple_of_9_is_multiple_of_3 : Prop :=
  ∀ n : ℤ, n % 9 = 0 → n % 3 = 0

-- Define that a certain odd number is a multiple of 9
def odd_multiple_of_9 (n : ℤ) : Prop :=
  n % 2 = 1 ∧ n % 9 = 0

-- The goal: Prove that the reasoning process is completely correct
theorem correct_reasoning (H1 : multiple_of_9_is_multiple_of_3)
                          (n : ℤ)
                          (H2 : odd_multiple_of_9 n) : 
                          (n % 3 = 0) :=
by
  -- Explanation of the proof here
  sorry

end correct_reasoning_l368_368948


namespace midpoint_theorem_converse_midpoint_theorem_l368_368445

section midpoint_theorem
/-- Definition of a midpoint -/
def is_midpoint {α : Type*} [AddGroup α] (A B M : α) : Prop :=
  A + B = 2 • M

/-- Midpoint line theorem -/
theorem midpoint_theorem {α : Type*} [AddGroup α] [Module ℝ α] {A B C M N : α}
  (hM : is_midpoint A B M) (hN : is_midpoint A C N) :
  (∃ K : α, K = (1/2) • (B - C) ∧ M + K = N - K ∧ 2 • K = N - M) :=
sorry

/-- Converse of the midpoint line theorem -/
theorem converse_midpoint_theorem {α : Type*} [AddGroup α] [Module ℝ α] {A B C M N : α}
  (hM : is_midpoint A B M) (hN : is_midpoint A C N) :
  (∃ K : α, K = (1/2) • (B - C) ∧ M + K = N - K ∧ 2 • K = N - M) :=
sorry

end midpoint_theorem

end midpoint_theorem_converse_midpoint_theorem_l368_368445


namespace simplify_polynomial_l368_368451

theorem simplify_polynomial (s : ℝ) :
  (2*s^2 + 5*s - 3) - (2*s^2 + 9*s - 7) = -4*s + 4 :=
by
  sorry

end simplify_polynomial_l368_368451


namespace largest_prime_factor_of_sum_of_divisors_of_360_is_13_l368_368820

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (finset.range (n + 1)).filter (λ k, n % k = 0), k

theorem largest_prime_factor_of_sum_of_divisors_of_360_is_13 :
  let M := sum_divisors 360 in
  M = 1170 →
  ∃ p, is_prime p ∧ p ∣ 1170 ∧ ∀ q, is_prime q ∧ q ∣ 1170 → q ≤ p ∧ p = 13 :=
by
  sorry

end largest_prime_factor_of_sum_of_divisors_of_360_is_13_l368_368820


namespace train_average_speed_l368_368121

theorem train_average_speed (D : ℝ) (hD : D > 0) : 
  let T1 := D / 50;
      T2 := 2 * D / 20;
      total_time := T1 + T2;
      total_distance := 3 * D;
      average_speed := total_distance / total_time
  in average_speed = 25 :=
by
  sorry

end train_average_speed_l368_368121


namespace fewer_onions_grown_l368_368637

def num_tomatoes := 2073
def num_cobs_of_corn := 4112
def num_onions := 985

theorem fewer_onions_grown : num_tomatoes + num_cobs_of_corn - num_onions = 5200 := by
  sorry

end fewer_onions_grown_l368_368637


namespace stripe_area_l368_368958

/-
Given:
1. Cylinder diameter = 25 feet
2. Cylinder height = 60 feet
3. Green stripe width = 2 feet
4. Stripe wraps twice around the cylinder horizontally

Prove:
The area of the stripe is 100π square feet.
-/

noncomputable def cylinder_diameter : ℝ := 25
noncomputable def cylinder_height : ℝ := 60
noncomputable def stripe_width : ℝ := 2
noncomputable def wraps : ℕ := 2

theorem stripe_area (d h w : ℝ) (n : ℕ) (h_d : d = 25) (h_h : h = 60) (h_w : w = 2) (h_n : n = 2) : 
  let circumference := π * d in
  let total_length := n * circumference in
  let area := total_length * w in
  area = 100 * π :=
by
  sorry

end stripe_area_l368_368958


namespace top_card_is_queen_probability_l368_368117

theorem top_card_is_queen_probability :
  let num_queens := 4
  let total_cards := 52
  let prob := num_queens / total_cards
  prob = 1 / 13 :=
by 
  sorry

end top_card_is_queen_probability_l368_368117


namespace sum_of_256_and_125_in_base_5_l368_368892

-- Define the conversion from decimal to base 5 for 256 and 125
def to_base_5 (n : ℕ) : List ℕ :=
  if n == 0 then [0]
  else
    let rec aux n acc :=
      if n == 0 then acc
      else aux (n / 5) (n % 5 :: acc)
    aux n []

-- Define the base 5 representation of 256
def base_5_of_256 := to_base_5 256
-- Define the base 5 representation of 125
def base_5_of_125 := to_base_5 125

-- Define the expected sum in base 5
def expected_sum_in_base_5 := [3, 0, 1, 1]

theorem sum_of_256_and_125_in_base_5 :
  (to_base_5 256 = [2, 0, 1, 1]) ∧ (to_base_5 125 = [1, 0, 0, 0])
  → to_base_5 (256 + 125) = expected_sum_in_base_5 := by
  sorry

end sum_of_256_and_125_in_base_5_l368_368892


namespace sin_double_angle_l368_368696

theorem sin_double_angle (x : ℝ) (h : sin (x + π / 4) = 1 / 4) : sin (2 * x) = -7 / 8 := 
by
  sorry

end sin_double_angle_l368_368696


namespace c_value_l368_368336

theorem c_value (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 200 * x + c = (x + a)^2) → c = 10000 := 
by
  intro h
  sorry

end c_value_l368_368336


namespace conjugate_of_z_l368_368718

open Complex

noncomputable def z : ℂ := 1 + I / (1 - I)

theorem conjugate_of_z :
  (conj z = -I) :=
by
  have eq1 : z * (1 - I) = 1 + I := by sorry
  show conj z = -I from sorry

end conjugate_of_z_l368_368718


namespace measure_of_delta_sum_of_cosines_arccos_identity_delta_is_67_degrees_l368_368201

noncomputable def delta : ℝ :=
  Real.arccos ((Finset.sum (Finset.range (6504 - 2903)) (λ i, Real.sin (2903 + i))) ^ 
  (Finset.sum (Finset.range (6481 - 2880)) (λ i, Real.cos (2880 + i))))

theorem measure_of_delta :
  (Finset.sum (Finset.range (6504 - 2903)) (λ i, Real.sin (2903 + i))) = Real.sin 23 :=
begin
  sorry -- Proving sum of sines
end

theorem sum_of_cosines :
  (Finset.sum (Finset.range (6481 - 2880)) (λ i, Real.cos (2880 + i))) = 1 :=
begin
  sorry -- Proving sum of cosines
end

theorem arccos_identity (θ : ℝ) (h : 0 < θ ∧ θ < 90) :
  Real.arccos (Real.sin θ) = 90 - θ :=
begin
  sorry -- Proving arccos identity
end

theorem delta_is_67_degrees :
  delta = 67 :=
begin
  rw [←arccos_identity, measure_of_delta, sum_of_cosines],
  exact eq.symm (real.coe_degrees (67))
end

end measure_of_delta_sum_of_cosines_arccos_identity_delta_is_67_degrees_l368_368201


namespace triangle_area_bound_l368_368074

-- Definitions and conditions
variable {ABC : Type} -- representing our triangle ABC
variable (A B C : ABC)

variable (length_bisectors : (ABC × ABC) → ℝ) -- Function to measure length of bisectors
variable (length_segments : (ABC × ABC × ABC) → ℝ) -- Function to measure length of segments from vertices

-- Conditions
axiom bisector_length_condition : ∀ x y, length_bisectors (x, y) ≤ 1
axiom segment_length_condition : ∀ x y z, length_segments (x, y, z) ≤ 1

-- Area calculation
variable (area_triangle : ABC → ℝ) -- Function to compute the area of the triangle

-- Theorem statement
theorem triangle_area_bound : area_triangle ABC ≤ 1 / Real.sqrt 3 :=
sorry

end triangle_area_bound_l368_368074


namespace alternating_series_sum_l368_368148

theorem alternating_series_sum : (Finset.range 100).sum (λ n, if even n then -(n+1) else n+1) = -50 :=
by
  sorry

end alternating_series_sum_l368_368148


namespace factor_polynomial_l368_368189

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end factor_polynomial_l368_368189


namespace max_squares_from_twelve_points_l368_368047

/-- Twelve points are marked on a grid paper. Prove that the maximum number of squares 
that can be formed by connecting four of these points is 11. -/
theorem max_squares_from_twelve_points : ∀ (points : list (ℝ × ℝ)), points.length = 12 → ∃ (squares : set (set (ℝ × ℝ))), squares.card = 11 ∧ ∀ square ∈ squares, ∃ (p₁ p₂ p₃ p₄ : (ℝ × ℝ)), p₁ ∈ points ∧ p₂ ∈ points ∧ p₃ ∈ points ∧ p₄ ∈ points ∧ set.to_finset {p₁, p₂, p₃, p₄}.card = 4 ∧ is_square {p₁, p₂, p₃, p₄} :=
by
  sorry

end max_squares_from_twelve_points_l368_368047


namespace pascals_triangle_total_numbers_l368_368301

theorem pascals_triangle_total_numbers (N : ℕ) (hN : N = 29) :
  (∑ n in Finset.range (N + 1), (n + 1)) = 465 :=
by
  rw hN
  calc (∑ n in Finset.range 30, (n + 1))
      = ∑ k in Finset.range 30, (k + 1) : rfl
  -- Here we are calculating the sum of the first 30 terms of the sequence (n + 1)
  ... = 465 : sorry

end pascals_triangle_total_numbers_l368_368301


namespace relationship_between_p_and_q_l368_368402

variable {α β γ p q : ℝ}
variable {x1 x2 : ℝ}

def is_root (p q r : ℝ) (x : ℝ) : Prop := x^2 + p * x + q = 0

theorem relationship_between_p_and_q 
  (h_root1 : is_root p q x1) 
  (h_root2 : is_root p q x2) 
  (h_eq : α * x1^2 + β * x1 + γ = α * x2^2 + β * x2 + γ) :
  p^2 = 4 * q ∨ p = -β / α :=
sorry

end relationship_between_p_and_q_l368_368402


namespace perpendicular_lines_m_values_l368_368012

theorem perpendicular_lines_m_values (m : ℝ) :
    (∀ x y, (m + 3) * x + m * y - 2 = 0 ↔ mx - 6y + 5 = 0) → (m = 0 ∨ m = 3) := by
  sorry

end perpendicular_lines_m_values_l368_368012


namespace negate_proposition_l368_368897

theorem negate_proposition : (¬ ∀ x : ℝ, x^2 + 2*x + 1 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 1 ≤ 0 := by
  sorry

end negate_proposition_l368_368897


namespace prob_consecutive_pairs_prob_sum_five_pairs_l368_368092

def labels : List ℕ := [1, 2, 3, 4]

def drawn_pairs : List (ℕ × ℕ) := [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

def is_consecutive_pair (a b : ℕ) : Bool :=
  (a = b + 1) || (b = a + 1)

def is_sum_five (a b : ℕ) : Bool :=
  (a + b = 5)

theorem prob_consecutive_pairs :
  (drawn_pairs.count (λ p => is_consecutive_pair p.1 p.2) : ℚ) / (drawn_pairs.length : ℚ) = 1 / 2 :=
by sorry

theorem prob_sum_five_pairs :
  (drawn_pairs.count (λ p => is_sum_five p.1 p.2) : ℚ) / (drawn_pairs.length : ℚ) = 1 / 3 :=
by sorry

end prob_consecutive_pairs_prob_sum_five_pairs_l368_368092


namespace shadow_indeterminate_under_streetlight_l368_368920

-- Definitions based on the given conditions
variables (A B : Type) -- Person A and Person B
variable  (sunlight : Prop) -- Condition about sunlight
variable  (streetlight : Prop) -- Condition about streetlight
variable  (shadow_length_sunlight : A → B → Prop) -- Assumption about shadow lengths under sunlight
variable  (shadow_length_streetlight : A → B → Prop) -- Assumption about shadow lengths under streetlight

-- The condition: Under sunlight, A's shadow is longer than B's shadow.
axiom sunlight_condition : sunlight → shadow_length_sunlight A B

-- The question (translated to a proof): Under the same streetlight, 
-- it's impossible to determine whose shadow would be longer without 
-- additional context about their positions relative to the streetlight.
theorem shadow_indeterminate_under_streetlight (s : sunlight) (t : streetlight) : 
  ¬∃ (h : shadow_length_streetlight A B), true :=
sorry

end shadow_indeterminate_under_streetlight_l368_368920


namespace jane_days_to_complete_task_l368_368801

theorem jane_days_to_complete_task :
  ∃ (J : ℕ), 
    (6 * (1 / 20 + 1 / J) + 4 * (1 / 20) = 1) ∧ 
    J = 12 :=
begin
  use 12,
  split,
  { norm_num,
    sorry },
  { refl }
end

end jane_days_to_complete_task_l368_368801


namespace log_inequality_solution_set_l368_368513

theorem log_inequality_solution_set (x : ℝ) : 
  log 2 (1 - x) ≤ 3 → -7 ≤ x ∧ x < 1 := by
  sorry

end log_inequality_solution_set_l368_368513


namespace find_other_number_l368_368518

theorem find_other_number (x y : ℕ) (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) (h3 : x = 7) : y = 3 :=
by
  sorry

end find_other_number_l368_368518


namespace t_f_8_l368_368830

def t (x : ℝ) : ℝ := Real.sqrt (5 * x + 1)

def f (x : ℝ) : ℝ := 8 - t x

theorem t_f_8 : t (f 8) = Real.sqrt (41 - 5 * Real.sqrt 41) :=
  by
  sorry

end t_f_8_l368_368830


namespace pascal_triangle_row_sum_l368_368273

theorem pascal_triangle_row_sum : (∑ n in Finset.range 30, n + 1) = 465 := by
  sorry

end pascal_triangle_row_sum_l368_368273


namespace tagged_to_total_fish_ratio_l368_368355

theorem tagged_to_total_fish_ratio
  (total_fish_in_pond : ℕ)
  (initial_catch : ℕ)
  (tagged_in_initial_catch : ℕ)
  (second_catch : ℕ)
  (tagged_in_second_catch : ℕ)
  (approx_fish_in_pond : ℕ) :
  total_fish_in_pond = 1800 →
  initial_catch = 60 →
  tagged_in_initial_catch = 60 →
  second_catch = 60 →
  tagged_in_second_catch = 2 →
  approx_fish_in_pond = 1800 →
  tagged_in_second_catch / second_catch = 1 / 30 :=
by
  intro h1 h2 h3 h4 h5 h6
  have _: tagged_in_second_catch = 2 := h5
  have _: second_catch = 60 := h4
  rw [this, this]
  exact Iff.rfl

end tagged_to_total_fish_ratio_l368_368355


namespace order_of_y1_y2_y3_l368_368711

/-
Given three points A(-3, y1), B(3, y2), and C(4, y3) all lie on the parabola y = 2*(x - 2)^2 + 1,
prove that y2 < y3 < y1.
-/
theorem order_of_y1_y2_y3 :
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  sorry

end order_of_y1_y2_y3_l368_368711


namespace sum_of_angles_is_90_l368_368911

variables (α β γ : ℝ)
-- Given angles marked on squared paper, which imply certain geometric properties
axiom angle_properties : α + β + γ = 90

theorem sum_of_angles_is_90 : α + β + γ = 90 := 
by
  apply angle_properties

end sum_of_angles_is_90_l368_368911


namespace diagonal_of_larger_screen_l368_368021

theorem diagonal_of_larger_screen (d : ℝ) 
  (h1 : ∃ s : ℝ, s^2 = 20^2 + 42) 
  (h2 : ∀ s, d = s * Real.sqrt 2) : 
  d = Real.sqrt 884 :=
by
  sorry

end diagonal_of_larger_screen_l368_368021


namespace lions_min_games_for_90_percent_wins_l368_368122

theorem lions_min_games_for_90_percent_wins : 
  ∀ N : ℕ, (N ≥ 26) ↔ 1 + N ≥ (9 * (4 + N)) / 10 := 
by 
  sorry

end lions_min_games_for_90_percent_wins_l368_368122


namespace negation_of_existence_lt_zero_l368_368440

theorem negation_of_existence_lt_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by sorry

end negation_of_existence_lt_zero_l368_368440


namespace max_true_statements_l368_368491

variables (Joe : Type) 
  (Skillful Lucky : Joe → Prop) 
  -- Definitions of the statements
  (H1 : Skillful Joe) 
  (H2 : ¬ Lucky Joe) 
  (H3 : Lucky Joe ∧ ¬ Skillful Joe) 
  (H4 : Skillful Joe → ¬ Lucky Joe) 
  (H5 : Skillful Joe ↔ Lucky Joe) 
  (H6 : (Skillful Joe ∨ Lucky Joe) ∧ ¬ (Skillful Joe ∧ Lucky Joe))

-- Statement to prove the maximum number of true statements
theorem max_true_statements : 
  ∃ (n : ℕ), n = 3 := 
sorry

end max_true_statements_l368_368491


namespace angle_BAC_is_60_degrees_l368_368378

theorem angle_BAC_is_60_degrees
    (A B C D E F : Type → Prop)
    [IsTriangle A B C]
    (h₁ : AC = 2 * AB)
    (h₂ : OnSegment D AB)
    (h₃ : OnSegment E BC)
    (h₄ : Angle_BAE_eq_Angle_ACD)
    (h₅ : F = IntersectionSegment AE CD)
    (h₆ : IsEquilateralTriangle B F E) :
    Angle BAC = 60 :=
by
  sorry

end angle_BAC_is_60_degrees_l368_368378


namespace land_per_person_l368_368805

noncomputable def total_land_area : ℕ := 20000
noncomputable def num_people_sharing : ℕ := 5

theorem land_per_person (Jose_land : ℕ) (h : Jose_land = total_land_area / num_people_sharing) :
  Jose_land = 4000 :=
by
  sorry

end land_per_person_l368_368805


namespace set_subset_condition_l368_368735

theorem set_subset_condition (a : ℝ) :
  (∀ x, (1 < a * x ∧ a * x < 2) → (-1 < x ∧ x < 1)) → (|a| ≥ 2 ∨ a = 0) :=
by
  intro h
  sorry

end set_subset_condition_l368_368735


namespace total_students_playing_one_sport_l368_368362

noncomputable def students_playing_at_least_one_sport (total_students B S Ba C B_S B_Ba B_C S_Ba C_S C_Ba B_C_S: ℕ) : ℕ :=
  B + S + Ba + C - B_S - B_Ba - B_C - S_Ba - C_S - C_Ba + B_C_S

theorem total_students_playing_one_sport : 
  students_playing_at_least_one_sport 200 50 60 35 80 10 15 20 25 30 5 10 = 130 := by
  sorry

end total_students_playing_one_sport_l368_368362


namespace minimum_balls_needed_l368_368584

noncomputable def ballColors : Type := {red green yellow blue white black}

def ballsInBox : ballColors → ℕ := 
  λ color, 
    match color with
    | red => 28
    | green => 20
    | yellow => 13
    | blue => 19
    | white => 11
    | black => 9

theorem minimum_balls_needed 
  (balls : ballColors → ℕ := ballsInBox)
  (h : ∀ color : ballColors, balls color > 0) :
  ∃ n, (∀ draws : fin n → ballColors, ∃ color, (draws.val.filter (λ c, c = color)).length ≥ 15) ↔ (n ≥ 76) :=
sorry

end minimum_balls_needed_l368_368584


namespace perpendicular_slope_of_line_l368_368204

theorem perpendicular_slope_of_line (x y : ℤ) : 
    (5 * x - 4 * y = 20) → 
    ∃ m : ℚ, m = -4 / 5 := 
by 
    sorry

end perpendicular_slope_of_line_l368_368204


namespace find_x_l368_368090

theorem find_x : ∃ x : ℝ, 45 * x = 0.35 * 900 ∧ x = 7 :=
by
  use 7
  split
  · have h : 0.35 * 900 = 315 := by norm_num
    rw [h]
    norm_num
  · refl

end find_x_l368_368090


namespace constant_sequence_no_integer_cubes_l368_368731

variable (x y : ℕ → ℤ)
variable (n : ℕ)

-- Definition of the sequences based on the given recurrence relations
def x_seq : ℕ → ℤ 
| 0 => 3
| (n + 1) => 3 * x_seq n + 2 * y_seq n

def y_seq : ℕ → ℤ 
| 0 => 4 
| (n + 1) => 4 * x_seq n + 3 * y_seq n

-- The theorem to prove that 2x_n^2 - y_n^2 is always constant, and equals to 2
theorem constant_sequence (n : ℕ) : 2 * (x_seq n)^2 - (y_seq n)^2 = 2 :=
by
  sorry

-- The theorem to verify there are no integer cubes in the sequences
theorem no_integer_cubes (n : ℕ) : ¬(∃ m : ℤ, m^3 = x_seq n ∨ m^3 = y_seq n) :=
by
  sorry

end constant_sequence_no_integer_cubes_l368_368731


namespace problem1_problem2_l368_368140

-- First Problem Statement:
theorem problem1 :  12 - (-18) + (-7) - 20 = 3 := 
by 
  sorry

-- Second Problem Statement:
theorem problem2 : -4 / (1 / 2) * 8 = -64 := 
by 
  sorry

end problem1_problem2_l368_368140


namespace initial_alcohol_percentage_l368_368579

-- Define the initial variables and constants
variables (P : ℝ) (V_initial : ℝ := 11) (V_added : ℝ := 13) (final_percentage : ℝ := 7.333333333333333) (V_final : ℝ := V_initial + V_added)

-- Define the mathematical expressions and the conditions
def initial_alcohol_volume (P : ℝ) : ℝ := (P / 100) * V_initial
def final_alcohol_volume : ℝ := (final_percentage / 100) * V_final

-- Statement to be proven
theorem initial_alcohol_percentage (P : ℝ) : initial_alcohol_volume P = final_alcohol_volume → P = 56 / 11 :=
by sorry

end initial_alcohol_percentage_l368_368579


namespace extraordinary_numbers_in_interval_l368_368843

def is_extraordinary (n : ℕ) : Prop :=
  n % 2 = 0 ∧ ∃ p : ℕ, Nat.Prime p ∧ n = 2 * p

def count_extraordinary (a b : ℕ) : ℕ :=
  (Finset.filter (λ n, is_extraordinary n) (Finset.range' a (b + 1))).card

theorem extraordinary_numbers_in_interval :
  count_extraordinary 1 75 = 12 := 
sorry

end extraordinary_numbers_in_interval_l368_368843


namespace find_m_l368_368330

theorem find_m (m : ℕ) :
  (2022 ^ 2 - 4) * (2021 ^ 2 - 4) = 2024 * 2020 * 2019 * m → 
  m = 2023 :=
by
  sorry

end find_m_l368_368330


namespace scientific_notation_101_49_billion_l368_368168

-- Define the term "one hundred and one point four nine billion"
def billion (n : ℝ) := n * 10^9

-- Axiomatization of the specific number in question
def hundredOnePointFourNineBillion := billion 101.49

-- Theorem stating that the scientific notation for 101.49 billion is 1.0149 × 10^10
theorem scientific_notation_101_49_billion : hundredOnePointFourNineBillion = 1.0149 * 10^10 :=
by
  sorry

end scientific_notation_101_49_billion_l368_368168


namespace extraordinary_numbers_count_l368_368847

/-- An integer is called "extraordinary" if it has exactly one even divisor other than 2.
We need to count the number of extraordinary numbers in the interval [1, 75]. -/
def is_extraordinary (n : ℕ) : Prop :=
  ∃ p : ℕ, nat.prime p ∧ p % 2 = 1 ∧ n = 2 * p

theorem extraordinary_numbers_count :
  (finset.filter (λ n : ℕ, n ≥ 1 ∧ n ≤ 75 ∧ is_extraordinary n) (finset.range 76)).card = 11 :=
by
  sorry

end extraordinary_numbers_count_l368_368847


namespace power_mod_residue_l368_368058

theorem power_mod_residue (n : ℕ) (h : n = 1234) : (7^n) % 19 = 9 := by
  sorry

end power_mod_residue_l368_368058


namespace angle_MBC_45_l368_368391

-- Definitions based on problem conditions
variables {A B C D E M : Type} 
variables [triangle ABC : A → B → C] 
variables [is_right_triangle ABC B]
variables [square ACDE : A → C → D → E]
variables [center ACDE M : M]

-- Main theorem statement
theorem angle_MBC_45 (H1 : is_right_triangle ABC B)  
                    (H2 : square ACDE A C D E) 
                    (H3 : center ACDE M) : 
  angle M B C = 45 :=
sorry

end angle_MBC_45_l368_368391


namespace minimum_barrels_drawn_to_guarantee_divisibility_l368_368778

theorem minimum_barrels_drawn_to_guarantee_divisibility (n : ℕ) (h : n = 90) : 
  ∃ k, k = 49 ∧ (∀ barrels : fin n, (barrels % 3 = 0 ∨ barrels % 5 = 0) → k >= barrels) :=
by {
  sorry
}

end minimum_barrels_drawn_to_guarantee_divisibility_l368_368778


namespace max_distance_of_SUV_l368_368626

-- Definitions
def highwayMileage := 12.2
def cityMileage := 7.6
def gallons := 25

-- Theorem Statement
theorem max_distance_of_SUV : highwayMileage * gallons = 305 :=
by
  sorry

end max_distance_of_SUV_l368_368626


namespace pascal_triangle_sum_first_30_rows_l368_368288

theorem pascal_triangle_sum_first_30_rows :
  (Finset.range 30).sum (λ n, n + 1) = 465 :=
begin
  sorry
end

end pascal_triangle_sum_first_30_rows_l368_368288


namespace isosceles_triangle_l368_368410

-- Definitions based on conditions
variables (A B C D E F : Type)
variables [concyclic_points A B C D] -- A, B, C, D are concyclic points
variables [intersection_point E (line_through A B) (line_through C D)] -- E is the intersection of lines AB and CD
variables [tangent_to_circumcircle_at_D (triangle A D E) (line_through D F)] -- Tangent at D to circumcircle of triangle ADE intersects BC at F

-- Statement of the theorem to prove
theorem isosceles_triangle (A B C D E F : Point) 
  [concyclic_points A B C D] 
  [intersection_point E (line_through A B) (line_through C D)]
  [tangent_to_circumcircle_at_D (triangle A D E) (line_through D F) (line_through C F)] :
  isosceles_at_F (triangle D C F) := 
sorry

end isosceles_triangle_l368_368410


namespace range_of_a_l368_368244

def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * a^x

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x1 x2 : ℝ, (f a x1 - f a x2) / (x1 - x2) > 0) : a ∈ Set.Ioo 0 1 ∪ Set.Ioi 2 :=
sorry

end range_of_a_l368_368244


namespace smallest_positive_integer_l368_368062

theorem smallest_positive_integer {x : ℕ} (h1 : x % 6 = 3) (h2 : x % 8 = 5) : x = 21 :=
sorry

end smallest_positive_integer_l368_368062


namespace fly_distance_l368_368538

def fly_distance_from_ceiling (x y z : ℝ) : ℝ :=
  sqrt (x^2 + y^2 + z^2)

theorem fly_distance (x y d : ℝ) (h1 : x = 3) (h2 : y = 7) (h3 : d = 10) :
  ∃ z, fly_distance_from_ceiling x y z = d ∧ z = sqrt 42 :=
by {
  use sqrt 42,
  split,
  {
    rw [h1, h2, h3],
    simp [fly_distance_from_ceiling],
    norm_num,
  },
  {
    refl,
  }
}

end fly_distance_l368_368538


namespace appearance_equally_likely_all_selections_not_equally_likely_l368_368184

variables {n k : ℕ} (numbers : finset ℕ)

-- Conditions
def chosen_independently (x : ℕ) : Prop := sorry
def move_clockwise_if_chosen (x : ℕ) (chosen : finset ℕ) : Prop := sorry
def end_with_k_different_numbers (final_set : finset ℕ) : Prop := final_set.card = k

-- Part (a)
theorem appearance_equally_likely (x : ℕ) (h_independent : chosen_independently x)
  (h_clockwise : ∀ y ∈ numbers, move_clockwise_if_chosen y numbers) :
  (∃ y ∈ numbers, true) → true :=
by { sorry } -- Conclusion: Yes

-- Part (b)
theorem all_selections_not_equally_likely (samples : list (finset ℕ))
  (h_independent : ∀ x ∈ samples, chosen_independently x)
  (h_clockwise : ∀ y ∈ samples, move_clockwise_if_chosen y samples) :
  ¬ (∀ x y, x ≠ y → samples x = samples y) :=
by { sorry } -- Conclusion: No

end appearance_equally_likely_all_selections_not_equally_likely_l368_368184


namespace perpendicular_lines_m_values_l368_368011

theorem perpendicular_lines_m_values (m : ℝ) :
    (∀ x y, (m + 3) * x + m * y - 2 = 0 ↔ mx - 6y + 5 = 0) → (m = 0 ∨ m = 3) := by
  sorry

end perpendicular_lines_m_values_l368_368011


namespace determinant_zero_l368_368469

variables (a b c d : ℝ^3) (α β γ : ℝ) (D : ℝ)

-- Define the determinant condition
def matrix_det (M : Matrix (Fin 3) (Fin 3) ℝ) : ℝ :=
  Matrix.det M

theorem determinant_zero (a b c d : ℝ^3) (α β γ : ℝ) : 
  matrix_det ![![α * (a × b), β * (b × c), γ * (c × (a × d))]] = 0 := 
by {
  -- providing the proof is not necessary
  sorry,
}

end determinant_zero_l368_368469


namespace double_neg_cancel_l368_368456

theorem double_neg_cancel (a : ℤ) : - (-2) = 2 :=
sorry

end double_neg_cancel_l368_368456


namespace car_speed_624km_in_2_2_5_hours_l368_368589

theorem car_speed_624km_in_2_2_5_hours : 
  ∀ (distance time_in_hours : ℝ), distance = 624 → time_in_hours = 2 + (2/5) → distance / time_in_hours = 260 :=
by
  intros distance time_in_hours h_dist h_time
  sorry

end car_speed_624km_in_2_2_5_hours_l368_368589


namespace cone_base_circumference_l368_368593

theorem cone_base_circumference (r : ℝ) (theta : ℝ) (h_r : r = 6) (h_theta : theta = 240) :
  (2 / 3) * (2 * Real.pi * r) = 8 * Real.pi :=
by
  have circle_circumference : ℝ := 2 * Real.pi * r
  sorry

end cone_base_circumference_l368_368593


namespace cans_difference_l368_368996

theorem cans_difference 
  (n_cat_packages : ℕ) (n_dog_packages : ℕ) 
  (n_cat_cans_per_package : ℕ) (n_dog_cans_per_package : ℕ) :
  n_cat_packages = 6 →
  n_dog_packages = 2 →
  n_cat_cans_per_package = 9 →
  n_dog_cans_per_package = 3 →
  (n_cat_packages * n_cat_cans_per_package) - (n_dog_packages * n_dog_cans_per_package) = 48 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end cans_difference_l368_368996


namespace train_crosses_platform_in_60_seconds_l368_368499

theorem train_crosses_platform_in_60_seconds :
  ∀ (length_train length_platform speed_kmph : ℕ), 
  length_train = length_platform →
  length_train = 1500 →
  speed_kmph = 180 →
  (60 : ℕ) = (2 * length_train) / (speed_kmph * (5 / 18 : ℚ)).toRational :=
by
  sorry

end train_crosses_platform_in_60_seconds_l368_368499


namespace product_of_roots_4x4_minus_3x3_plus_2x2_minus_5x_plus_6_eq_3_div_2_l368_368999

noncomputable def polynomial_coeffs := (4, -3, 2, -5, 6)

noncomputable def product_of_roots (coeffs : ℚ × ℚ × ℚ × ℚ × ℚ) : ℚ :=
  let a := coeffs.1
  let z := coeffs.2.2.2.2
  (-1 : ℚ) ^ 4 * z / a

theorem product_of_roots_4x4_minus_3x3_plus_2x2_minus_5x_plus_6_eq_3_div_2 :
  product_of_roots polynomial_coeffs = 3 / 2 :=
by
  sorry

end product_of_roots_4x4_minus_3x3_plus_2x2_minus_5x_plus_6_eq_3_div_2_l368_368999


namespace cube_red_face_probability_l368_368859

theorem cube_red_face_probability :
  let faces_total := 6
  let red_faces := 3
  let probability_red := red_faces / faces_total
  probability_red = 1 / 2 :=
by
  sorry

end cube_red_face_probability_l368_368859


namespace negate_neg_two_l368_368453

theorem negate_neg_two : -(-2) = 2 := by
  -- The proof goes here
  sorry

end negate_neg_two_l368_368453


namespace cream_ratio_correct_l368_368384

def coffee_scenario (joe_start_coffee : ℝ) (joe_initial_drink : ℝ) (cream_added_joe : ℝ) 
                    (joe_second_drink : ℝ) (joann_start_coffee : ℝ) (cream_added_joann : ℝ)
                    (joann_drink : ℝ) (joe_remaining_cream : ℝ) (joann_remaining_cream : ℝ) 
                    (expected_ratio : ℝ) : Prop :=
  let joe_total_first : ℝ := joe_start_coffee - joe_initial_drink + cream_added_joe in
  let joe_total_second : ℝ := joe_total_first - joe_second_drink in
  let joann_total : ℝ := joann_start_coffee + cream_added_joann in
  let joe_cream_remaining : ℝ := cream_added_joe - (joe_second_drink * (cream_added_joe / joe_total_first)) in
  let joann_cream_remaining : ℝ := cream_added_joann - (joann_drink * (cream_added_joann / joann_total)) in 
  joe_cream_remaining = joe_remaining_cream ∧
  joann_cream_remaining = joann_remaining_cream ∧
  (joe_remaining_cream / joann_remaining_cream = expected_ratio)

theorem cream_ratio_correct : coffee_scenario 20 4 3 3 20 3 3 (48 / 19) (60 / 23) (92 / 95) :=
by sorry

end cream_ratio_correct_l368_368384


namespace pascal_triangle_sum_first_30_rows_l368_368294

theorem pascal_triangle_sum_first_30_rows :
  (Finset.range 30).sum (λ n, n + 1) = 465 :=
begin
  sorry
end

end pascal_triangle_sum_first_30_rows_l368_368294


namespace breath_holding_time_initial_l368_368135

variable (T : ℝ)

theorem breath_holding_time_initial :
  let T : ℝ := T in
  (2 * T) * 2 * 1.5 = 60 → T = 10 :=
by
  intro h
  sorry

end breath_holding_time_initial_l368_368135


namespace tel_aviv_rain_probability_4_of_6_l368_368570

def rain_probability (p : ℝ) (n k : ℕ) : ℝ :=
  (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem tel_aviv_rain_probability_4_of_6 :
  rain_probability 0.5 6 4 = 15 / 64 :=
by sorry

end tel_aviv_rain_probability_4_of_6_l368_368570


namespace find_x_value_l368_368598

-- We define the conditions
def five_digits_problem (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 9 ∧ 
  (let digits := [1, 3, 4, 6, x] in
  let five_digit_nums_sum := 120 * (1 + 3 + 4 + 6 + x) in
  five_digit_nums_sum = 2640)

-- We state the theorem which proves that x = 8 given the conditions
theorem find_x_value : ∃ x: ℕ, five_digits_problem x ∧ x = 8 :=
begin
  sorry
end

end find_x_value_l368_368598


namespace necessary_condition_for_ellipse_l368_368895

theorem necessary_condition_for_ellipse (m : ℝ) : 
  (5 - m > 0) → (m + 3 > 0) → (5 - m ≠ m + 3) → (-3 < m ∧ m < 5 ∧ m ≠ 1) :=
by sorry

end necessary_condition_for_ellipse_l368_368895


namespace proposition_1_l368_368576

theorem proposition_1 (n : ℕ) :
  (∀ n : ℕ, ∑ i in finset.range (n + 1), (i + 1) * (i + 2)) = n * (n + 1) * (n + 2) / 3 :=
sorry

end proposition_1_l368_368576


namespace shaded_triangle_area_l368_368924

-- Definitions for points and the dimensions
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def A : Point := ⟨4, 0⟩
def B : Point := ⟨15, 0⟩
def C : Point := ⟨15, 11⟩
def D : Point := ⟨4, 11⟩
def E : Point := ⟨4, 11 / 3⟩

-- Conditions
def OA_length : ℝ := 4
def AB_length : ℝ := 11
def EA_length : ℝ := (AB_length) * (OA_length / (OA_length + AB_length))

-- Distance between D and E
def DE_length : ℝ := AB_length - EA_length

-- Area of triangle CDE
def area_CDE : ℝ := (1 / 2) * DE_length * 11

-- Problem statement in Lean
theorem shaded_triangle_area :
  round (area_CDE) = 28 := by
  sorry

end shaded_triangle_area_l368_368924


namespace eccentricity_of_ellipse_l368_368720

def ellipse_eq (a b : ℝ) (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

theorem eccentricity_of_ellipse {a b : ℝ} (h1 : a > b) (h2 : b > 0)
  (F1 F2 P I : ℝ × ℝ) (λ : ℝ)
  (hf1 : F1 = (-c, 0)) (hf2 : F2 = (c, 0)) (hp : ellipse_eq a b P.1 P.2)
  (h3 : (1 + λ) * (P.1 - F1.1, P.2 - F1.2) + (1 - λ) * (P.1 - F2.1, P.2 - F2.2) =
        3 * (P.1 - I.1, P.2 - I.2))
  (h4 : c = Real.sqrt (a^2 - b^2)) :
  let e := c / a in
  e = 1 / 2 :=
by sorry

end eccentricity_of_ellipse_l368_368720


namespace odd_function_property_l368_368723

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x >= 0 then log 3 (x + 1) + a else -f (-x) a

theorem odd_function_property (a : ℝ) : f (-8) a = -2 :=
begin
  sorry
end

end odd_function_property_l368_368723


namespace smallest_positive_integer_remainder_l368_368061

theorem smallest_positive_integer_remainder :
  ∃ (a : ℕ), a % 6 = 3 ∧ a % 8 = 5 ∧ ∀ b : ℕ, (b % 6 = 3 ∧ b % 8 = 5) → a ≤ b :=
begin
  use 21,
  split,
  { norm_num, },
  split,
  { norm_num, },
  intro b,
  intro hb,
  rcases hb with ⟨hb1, hb2⟩,
  sorry
end

end smallest_positive_integer_remainder_l368_368061


namespace min_students_l368_368777

variable (L : ℕ) (H : ℕ) (M : ℕ) (e : ℕ)

def find_min_students : Prop :=
  H = 2 * L ∧ 
  M = L + H ∧ 
  e = L + M + H ∧ 
  e = 6 * L ∧ 
  L ≥ 1

theorem min_students (L : ℕ) (H : ℕ) (M : ℕ) (e : ℕ) : find_min_students L H M e → e = 6 := 
by 
  intro h 
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end min_students_l368_368777


namespace parallelogram_sides_and_diagonal_l368_368014

theorem parallelogram_sides_and_diagonal 
(angle_ABC : ∀ A B C D : ℝ, A ≠ B → B ≠ C → C ≠ D → D ≠ A → 60 * real.pi / 180 = angle A B C) 
(diag_AC : ∀ A C : ℝ, A ≠ C → 2 * real.sqrt 31 = interval A C)
(perpendicular : ∀ O M : ℝ, O ≠ M → real.sqrt 75 / 2 = distance O M) :
  ∃ (AB AD AC : ℝ), 
    AB = 10 ∧ AD = 12 ∧ AC = 2 * real.sqrt 91 := by
begin
  sorry
end

end parallelogram_sides_and_diagonal_l368_368014


namespace distance_to_place_l368_368104

open Real

theorem distance_to_place (row_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  row_speed = 8 ∧ current_speed = 2.5 ∧ total_time = 3 ∧
  let downstream_speed := row_speed + current_speed,
      upstream_speed := row_speed - current_speed,
      T_d := total_time * upstream_speed / (upstream_speed + downstream_speed) in
  distance = T_d * downstream_speed → distance = 10.83 :=
by
  intros h
  have h_row_speed : row_speed = 8 := h.1
  have h_current_speed : current_speed = 2.5 := h.2.1
  have h_total_time : total_time = 3 := h.2.2
  let downstream_speed := row_speed + current_speed
  let upstream_speed := row_speed - current_speed
  let T_d := total_time * upstream_speed / (upstream_speed + downstream_speed)
  have T_d_def : T_d = 3 * 5.5 / (5.5 + 10.5) := by rfl
  have distance_def : distance = T_d * downstream_speed := by rw [← h.2.2.2, T_d_def]; rfl
  sorry

end distance_to_place_l368_368104


namespace mark_notebooks_at_126_percent_l368_368603

variable (L : ℝ) (C : ℝ) (M : ℝ) (S : ℝ)

def merchant_condition1 := C = 0.85 * L
def merchant_condition2 := C = 0.75 * S
def merchant_condition3 := S = 0.9 * M

theorem mark_notebooks_at_126_percent :
    merchant_condition1 L C →
    merchant_condition2 C S →
    merchant_condition3 S M →
    M = 1.259 * L := by
  intros h1 h2 h3
  sorry

end mark_notebooks_at_126_percent_l368_368603


namespace part_a_part_b_l368_368174

namespace problem

variables (n k : ℕ) (numbers : Finset ℕ)

-- Condition: Each subsequent number is chosen independently of the previous ones.
-- Condition: If a chosen number matches one of the already chosen numbers, move clockwise to the first unchosen number.
-- Condition: In the end, k different numbers are obtained.

-- Part (a): Appearance of each specific number is equally likely.
theorem part_a (h1 : ∀ (chosen : Finset ℕ), chosen.card = k → (∃! x ∈ numbers, x ∉ chosen)) :
  ∃ (p : ℚ), ∀ (x ∈ numbers), p = k / n :=
sorry

-- Part (b): Appearance of all selections is not equally likely.
theorem part_b (h1 : ∀ (chosen : Finset ℕ), chosen.card = k → (∃! x ∈ numbers, x ∉ chosen)) :
  ¬ ∃ (p : ℚ), ∀ (s : Finset ℕ), s.card = k → p = (card s.subevents) / (n ^ k) :=
sorry

end problem

end part_a_part_b_l368_368174


namespace average_encounter_interval_if_stationary_l368_368944

-- Definitions for given conditions in the problem

def encounter_interval_towards := 7 -- ship encounters meteors coming towards it every 7 seconds
def encounter_interval_same_direction := 13 -- ship overtakes meteors traveling in the same direction every 13 seconds

-- Hypothesize and prove the average encounter interval for a stationary ship is 9.1 seconds

theorem average_encounter_interval_if_stationary :
  let harmonic_mean (a b : ℕ) := (2 * a * b) / (a + b : ℕ) in
  harmonic_mean encounter_interval_towards encounter_interval_same_direction = 91 / 10 :=
by
  sorry

end average_encounter_interval_if_stationary_l368_368944


namespace original_bubble_radius_l368_368619

theorem original_bubble_radius (r : ℝ) (R : ℝ) (π : ℝ) 
  (h₁ : r = 4 * real.cbrt 2)
  (h₂ : (4/3) * π * R^3 = (2/3) * π * r^3) : 
  R = 4 :=
by 
  sorry

end original_bubble_radius_l368_368619


namespace tony_spends_2_dollars_per_sqft_l368_368535

def master_bedroom_bath_sqft : ℕ := 500
def guest_bedroom_sqft : ℕ := 200
def number_of_guest_bedrooms : ℕ := 2
def other_areas_sqft : ℕ := 600
def monthly_rent_dollars : ℕ := 3000

theorem tony_spends_2_dollars_per_sqft :
  let total_sqft := master_bedroom_bath_sqft 
                  + (guest_bedroom_sqft * number_of_guest_bedrooms) 
                  + other_areas_sqft in
  let cost_per_sqft := monthly_rent_dollars / total_sqft in
  cost_per_sqft = 2 := sorry

end tony_spends_2_dollars_per_sqft_l368_368535


namespace arithmetic_sequence_5th_term_l368_368371

theorem arithmetic_sequence_5th_term :
  let a1 := 3
  let d := 4
  a1 + 4 * (5 - 1) = 19 :=
by
  sorry

end arithmetic_sequence_5th_term_l368_368371


namespace y_intercept_of_tangent_line_l368_368097

noncomputable def point (x y : ℝ) := (x, y)

def circle1 : ℝ × ℝ × ℝ := (3, 0, 3)  -- radius 3, center (3, 0)
def circle2 : ℝ × ℝ × ℝ := (8, 0, 2)  -- radius 2, center (8, 0)

def distance (p q : ℝ × ℝ) : ℝ := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem y_intercept_of_tangent_line :
  let D := point 3 0,
      E := point 3 (3 + 2),
      F := point 8 0,
      G := point 8 (2 + (distance (3, 0) (8, 0))),
      line_tangent := λ q : ℝ × ℝ, q.2 = 2 * real.sqrt (104) in
  (line_tangent (0, 0)).2 = 2 * real.sqrt (104) :=
by
  sorry

end y_intercept_of_tangent_line_l368_368097


namespace pascal_triangle_sum_l368_368321

theorem pascal_triangle_sum (n : ℕ) (h₀ : n = 29) :
  (∑ k in Finset.range (n + 1), k + 1) = 465 := sorry

end pascal_triangle_sum_l368_368321


namespace find_positive_integer_x_l368_368680

theorem find_positive_integer_x :
  ∃ x : ℕ, x > 0 ∧ (5 * x + 1) / (x - 1) > 2 * x + 2 ∧
  ∀ y : ℕ, y > 0 ∧ (5 * y + 1) / (y - 1) > 2 * x + 2 → y = 2 :=
sorry

end find_positive_integer_x_l368_368680


namespace pascal_triangle_sum_l368_368326

theorem pascal_triangle_sum (n : ℕ) (h₀ : n = 29) :
  (∑ k in Finset.range (n + 1), k + 1) = 465 := sorry

end pascal_triangle_sum_l368_368326


namespace pascals_triangle_total_numbers_l368_368302

theorem pascals_triangle_total_numbers (N : ℕ) (hN : N = 29) :
  (∑ n in Finset.range (N + 1), (n + 1)) = 465 :=
by
  rw hN
  calc (∑ n in Finset.range 30, (n + 1))
      = ∑ k in Finset.range 30, (k + 1) : rfl
  -- Here we are calculating the sum of the first 30 terms of the sequence (n + 1)
  ... = 465 : sorry

end pascals_triangle_total_numbers_l368_368302


namespace factorization_correct_l368_368669

def polynomial : ℝ[X] := X^4 + 256

theorem factorization_correct :
  polynomial = (X^2 - 8*X + 16) * (X^2 + 8*X + 16) :=
by
  sorry

end factorization_correct_l368_368669


namespace geometric_seq_sum_l368_368222

theorem geometric_seq_sum (a : ℕ → ℝ) (a1 a2 a3 : ℝ) (n : ℕ) (q : ℝ) (h1 : a 1 = a1)
  (h2 : a 2 = a2) (h3 : a 3 = a3) (h_geometric : ∀ n, a(n+1) = a(n) * q) 
  (h_a1_a2 : a1 + a2 = 9) (h_a1_a2_a3 : a1 * a2 * a3 = 27) :
  (∑ i in finset.range n, a i) = 12 * (1 - (1 / 2 : ℝ)^n) :=
by 
  sorry

end geometric_seq_sum_l368_368222


namespace calc_3_op_4_l368_368338

-- Defining the operation x ⊕ y
def op (x y : ℝ) : ℝ := (x^2 + y^2) / (1 + x * y^2)

-- Stating the theorem
theorem calc_3_op_4 : 3 > 0 ∧ 4 > 0 → op 3 4 = 25 / 49 := by
  sorry

end calc_3_op_4_l368_368338


namespace problem_solution_l368_368599

noncomputable def is_geometric_sequence_preserving (f : ℝ → ℝ) : Prop :=
  ∀ (a : ℕ → ℝ), (∀ n, a n * a (n + 2) = (a (n + 1))^2) →
  ∀ n, f (a n) * f (a (n + 2)) = (f (a (n + 1)))^2

noncomputable def f1 : ℝ → ℝ := λ x, x^2
noncomputable def f2 : ℝ → ℝ := λ x, 2^x
noncomputable def f3 : ℝ → ℝ := λ x, real.sqrt (real.abs x)
noncomputable def f4 : ℝ → ℝ := λ x, real.log (real.abs x)

theorem problem_solution :
  is_geometric_sequence_preserving f1 ∧ ¬is_geometric_sequence_preserving f2 ∧
  is_geometric_sequence_preserving f3 ∧ ¬is_geometric_sequence_preserving f4 :=
by sorry

end problem_solution_l368_368599


namespace pants_cost_is_250_l368_368989

-- Define the cost of a T-shirt
def tshirt_cost := 100

-- Define the total amount spent
def total_amount := 1500

-- Define the number of T-shirts bought
def num_tshirts := 5

-- Define the number of pants bought
def num_pants := 4

-- Define the total cost of T-shirts
def total_tshirt_cost := tshirt_cost * num_tshirts

-- Define the total cost of pants
def total_pants_cost := total_amount - total_tshirt_cost

-- Define the cost per pair of pants
def pants_cost_per_pair := total_pants_cost / num_pants

-- Proving that the cost per pair of pants is $250
theorem pants_cost_is_250 : pants_cost_per_pair = 250 := by
  sorry

end pants_cost_is_250_l368_368989


namespace pascal_triangle_count_30_rows_l368_368282

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l368_368282


namespace smallest_circle_radius_polygonal_chain_l368_368946

theorem smallest_circle_radius_polygonal_chain (l : ℝ) (hl : l = 1) : ∃ (r : ℝ), r = 0.5 := 
sorry

end smallest_circle_radius_polygonal_chain_l368_368946


namespace cyclic_quadrilateral_B_equals_F_l368_368373
open_locale big_operators

theorem cyclic_quadrilateral_B_equals_F
  {O A B C D M E F : Type}
  [is_cyclic_quadrilateral O A B C D]
  (h1 : are_perpendicular A C B D)
  (h2 : is_midpoint_of_arc M A D C)
  (h3 : passes_through_circle M O D E F)
  (h4 : intersects_at DA DC E F) :
  BE = BF :=
sorry

end cyclic_quadrilateral_B_equals_F_l368_368373


namespace pascal_triangle_elements_count_l368_368271

theorem pascal_triangle_elements_count :
  ∑ n in finset.range 30, (n + 1) = 465 :=
by 
  sorry

end pascal_triangle_elements_count_l368_368271


namespace pascal_triangle_row_sum_l368_368278

theorem pascal_triangle_row_sum : (∑ n in Finset.range 30, n + 1) = 465 := by
  sorry

end pascal_triangle_row_sum_l368_368278


namespace range_of_f_l368_368684

def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f :
  set.range f = set.Icc (-3 / 2) 3 :=
sorry

end range_of_f_l368_368684


namespace equal_length_cycles_exists_smaller_function_still_applicable_l368_368656

theorem equal_length_cycles_exists (n : ℕ) (h1 : n > 1) :
  ∃ (G : SimpleGraph), G.vertices = n ∧ G.edges = (7 * n / 4) ∧ 
  ∃ (c1 c2 : Cycle), c1.length = c2.length := 
sorry

theorem smaller_function_still_applicable (n : ℕ) (h1 : n > 1) :
  ∃ (G : SimpleGraph), G.vertices = n ∧ G.edges = (3 * n / 2) ∧ 
  ∃ (c1 c2 : Cycle), c1.length = c2.length := 
sorry

end equal_length_cycles_exists_smaller_function_still_applicable_l368_368656


namespace cat_food_more_than_dog_food_l368_368993

theorem cat_food_more_than_dog_food :
  let cat_food_packs := 6
  let cans_per_cat_pack := 9
  let dog_food_packs := 2
  let cans_per_dog_pack := 3
  let total_cat_food_cans := cat_food_packs * cans_per_cat_pack
  let total_dog_food_cans := dog_food_packs * cans_per_dog_pack
  total_cat_food_cans - total_dog_food_cans = 48 :=
by
  sorry

end cat_food_more_than_dog_food_l368_368993


namespace coeff_x6_in_expansion_l368_368163

noncomputable def binom_coeff : ℕ → ℕ → ℕ
| n, k => Nat.choose n k

def term_coeff (n k : ℕ) : ℤ :=
  (-2)^k * binom_coeff n k

theorem coeff_x6_in_expansion : term_coeff 8 2 = 112 := by
  sorry

end coeff_x6_in_expansion_l368_368163


namespace total_time_round_trip_l368_368022

-- Define the given conditions
def speed_boat : ℝ := 16 -- Speed of the boat in standing water in kmph
def speed_stream : ℝ := 2 -- Speed of the stream in kmph
def distance : ℝ := 7020 -- Distance to the place in km

-- Define the derived speeds
def speed_upstream : ℝ := speed_boat - speed_stream
def speed_downstream : ℝ := speed_boat + speed_stream

-- Define the times
def time_upstream : ℝ := distance / speed_upstream
def time_downstream : ℝ := distance / speed_downstream

-- Define the total time
def total_time : ℝ := time_upstream + time_downstream

-- The proof statement
theorem total_time_round_trip : total_time = 891.4286 := by
  sorry

end total_time_round_trip_l368_368022


namespace solve_system_l368_368464

noncomputable def log_base_sqrt_3 (z : ℝ) : ℝ := Real.log z / Real.log (Real.sqrt 3)

theorem solve_system :
  ∃ x y : ℝ, (3^x * 2^y = 972) ∧ (log_base_sqrt_3 (x - y) = 2) ∧ (x = 5 ∧ y = 2) :=
by
  sorry

end solve_system_l368_368464


namespace pascal_triangle_sum_l368_368324

theorem pascal_triangle_sum (n : ℕ) (h₀ : n = 29) :
  (∑ k in Finset.range (n + 1), k + 1) = 465 := sorry

end pascal_triangle_sum_l368_368324


namespace time_for_goods_train_to_pass_l368_368965

-- Define conditions
def speed_of_mans_train : ℝ := 100 -- in kmph
def speed_of_goods_train : ℝ := 12 -- in kmph
def length_of_goods_train : ℝ := 280 -- in meters

-- Define the problem statement
theorem time_for_goods_train_to_pass :
  let relative_speed_in_kmph := speed_of_mans_train + speed_of_goods_train,
      relative_speed_in_mps := (relative_speed_in_kmph * 1000) / 3600,
      time_to_pass := length_of_goods_train / relative_speed_in_mps
  in abs (time_to_pass - 9) < 0.01 := by
  sorry

end time_for_goods_train_to_pass_l368_368965


namespace count_integers_leq_zero_l368_368158

noncomputable def P : Polynomial ℤ :=
  ∏ k in Finset.range 50 + 1, Polynomial.X - Polynomial.C (k^2)

def num_ints_leq_zero_P : ℕ :=
  1300

theorem count_integers_leq_zero (P : Polynomial ℤ) (num_ints_leq_zero_P : ℕ) :
  (∏ k in Finset.range 50 + 1, Polynomial.X - Polynomial.C (k^2)) = P →
  num_ints_leq_zero_P = 1300 := by
  sorry

end count_integers_leq_zero_l368_368158


namespace product_of_twelfth_roots_of_unity_l368_368136

noncomputable def twelfthRootOfUnity (k : ℕ) : ℂ := complex.exp (2 * real.pi * complex.I * k / 12)

theorem product_of_twelfth_roots_of_unity :
  let w := complex.exp (2 * real.pi * complex.I / 12)
  in (∏ k in finset.range 11, (-2 : ℂ) - twelfthRootOfUnity (k + 1)) = -1373 := 
by
  sorry

end product_of_twelfth_roots_of_unity_l368_368136


namespace total_handshakes_l368_368522

variable (r : ℕ) (c : ℕ)
variable (handshakes : ℕ)

axiom reps_per_company : r = 5
axiom num_companies : c = 3
axiom total_participants : r * c = 15
axiom shake_pattern : handshakes = (total_participants * (total_participants - 1 - (r - 1)) / 2)

theorem total_handshakes : handshakes = 75 := 
by 
  sorry

end total_handshakes_l368_368522


namespace greatest_number_l368_368633

-- Define the base conversions
def octal_to_decimal (n : Nat) : Nat := 3 * 8^1 + 2
def quintal_to_decimal (n : Nat) : Nat := 1 * 5^2 + 1 * 5^1 + 1
def binary_to_decimal (n : Nat) : Nat := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0
def senary_to_decimal (n : Nat) : Nat := 5 * 6^1 + 4

theorem greatest_number :
  max (max (octal_to_decimal 32) (quintal_to_decimal 111)) (max (binary_to_decimal 101010) (senary_to_decimal 54))
  = binary_to_decimal 101010 := by sorry

end greatest_number_l368_368633


namespace sum_of_first_n_terms_l368_368824

section
variables (S : ℕ → ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℝ)

-- Definition of Sequence a_n
def is_arithmetic (a : ℕ → ℕ) : Prop :=
  ∃ (d : ℕ) (a₀ : ℕ), ∀ n ≥ 1, a n = a₀ + (n - 1) * d

def satisfies_conditions : Prop :=
  (∀ n, 2 * S n = n * a n) ∧ (a 2 = 1)

def sequence_arithmetic : Prop :=
  satisfies_conditions S a ∧ (is_arithmetic a) ∧ (∀ n ≥ 1, a n = n - 1)

-- Sum of b_n where b_n = (-1)^n a_n + 2^n
def bn_sum (b : ℕ → ℕ) (T : ℕ → ℝ) : Prop :=
  (b 1 = (-1)^1 * a 1 + 2^1) ∧ (b 2 = 1 * a 2 + 2^2) ∧
  ∀ n, T n = if n % 2 = 0 then 2^(n+1) - 2 + n / 2
             else 2^(n+1) - 3 / 2 - n / 2

-- The theorem statement
theorem sum_of_first_n_terms (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℝ) :
  satisfies_conditions S a →
  sequence_arithmetic S a →
  bn_sum b T :=
sorry
end

end sum_of_first_n_terms_l368_368824


namespace part_I_part_II_l368_368248

def f (x a : ℝ) : ℝ := abs (x - a) + abs (2 * x + 1)

-- Part (I)
theorem part_I (x : ℝ) : f x 1 ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

-- Part (II)
theorem part_II (a : ℝ) : (∃ x ∈ Set.Ici a, f x a ≤ 2 * a + x) ↔ a ≥ 1 :=
by sorry

end part_I_part_II_l368_368248


namespace diff_of_squares_535_465_l368_368554

theorem diff_of_squares_535_465 : (535^2 - 465^2) = 70000 :=
sorry

end diff_of_squares_535_465_l368_368554


namespace numbers_from_five_threes_l368_368540

theorem numbers_from_five_threes :
  (∃ (a b c d e : ℤ), (3*a + 3*b + 3*c + 3*d + 3*e = 11 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 12 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 13 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 14 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 15) ) :=
by
  -- Proof provided by the problem statement steps, using:
  -- 11 = (33/3)
  -- 12 = 3 * 3 + 3 + 3 - 3
  -- 13 = 3 * 3 + 3 + 3/3
  -- 14 = (33 + 3 * 3) / 3
  -- 15 = 3 + 3 + 3 + 3 + 3
  sorry

end numbers_from_five_threes_l368_368540


namespace product_in_M_infinitely_many_pairs_l368_368510

-- Set M defined as integers of the form a^2 + 13b^2 where a and b are nonzero integers.
def M : set ℤ := {n : ℤ | ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ n = a^2 + 13*b^2}

-- First part: The product of any two elements in M is also an element of M.
theorem product_in_M (a b c d : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hM1 : a^2 + 13*b^2 ∈ M)
  (hM2 : c^2 + 13*d^2 ∈ M)
  : (a^2 + 13*b^2) * (c^2 + 13*d^2) ∈ M := 
sorry

-- Second part: There exist infinitely many pairs (x, y) such that x + y is not in M but x^13 + y^13 is in M.
theorem infinitely_many_pairs 
  : ∃ (f : ℕ → ℤ × ℤ), injective f ∧ ∀ n, 
    let x := (f n).1 
    let y := (f n).2 
    in (x + y ∉ M) ∧ (x^13 + y^13 ∈ M) := 
sorry

end product_in_M_infinitely_many_pairs_l368_368510


namespace coeff_exists_l368_368380

theorem coeff_exists :
  ∃ (A B C : ℕ), 
    ¬(8 ∣ A) ∧ ¬(8 ∣ B) ∧ ¬(8 ∣ C) ∧ 
    (∀ (n : ℕ), 8 ∣ (A * 5^n + B * 3^(n-1) + C))
    :=
sorry

end coeff_exists_l368_368380


namespace recurrence_relation_proof_l368_368813

theorem recurrence_relation_proof (a : ℕ → ℚ) (h₀ : a 0 = 1994)
  (h_rec : ∀ n, a (n + 1) = (a n) ^ 2 / ((a n) + 1)) :
  ∀ n, 0 ≤ n → n ≤ 998 → floor (a n) = 1994 - n :=
by
  sorry

end recurrence_relation_proof_l368_368813


namespace desk_height_l368_368539

variables (h l w : ℝ)

theorem desk_height
  (h_eq_2l_50 : h + 2 * l = 50)
  (h_eq_2w_40 : h + 2 * w = 40)
  (l_minus_w_eq_5 : l - w = 5) :
  h = 30 :=
by {
  sorry
}

end desk_height_l368_368539


namespace axis_of_symmetry_l368_368494

theorem axis_of_symmetry (m : ℝ) : ∃ x, (x = -1) ∧ (x = ((-4 : ℝ) + 2) / 2) :=
by
  use (-1 : ℝ)
  split
  . refl
  . norm_num

end axis_of_symmetry_l368_368494


namespace determine_c_l368_368334

theorem determine_c (c : ℝ) 
  (h : ∃ a : ℝ, (∀ x : ℝ, x^2 + 200 * x + c = (x + a)^2)) : c = 10000 :=
sorry

end determine_c_l368_368334


namespace determine_alpha_l368_368254

theorem determine_alpha (α : ℝ) (y : ℝ → ℝ) (h : ∀ x, y x = x^α) (hp : y 2 = Real.sqrt 2) : α = 1 / 2 :=
sorry

end determine_alpha_l368_368254


namespace value_of_P_dot_Q_l368_368821

def P : Set ℝ := {x | Real.log x / Real.log 2 < 1}
def Q : Set ℝ := {x | abs (x - 2) < 1}
def P_dot_Q (P Q : Set ℝ) : Set ℝ := {x | x ∈ P ∧ x ∉ Q}

theorem value_of_P_dot_Q : P_dot_Q P Q = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end value_of_P_dot_Q_l368_368821


namespace math_exam_scores_comprehensive_survey_l368_368071

/-- Condition 1: A large survey scope requires a sampling survey -/
def large_survey_scope_requires_sampling (survey_scope : Prop) : Prop :=
  survey_scope → ¬comprehensive_survey

/-- Condition 2: A destructive process requires a sampling survey -/
def destructive_process_requires_sampling (destructive : Prop) : Prop :=
  destructive → ¬comprehensive_survey

/-- Condition 3: Understanding the vision condition of seventh-grade students nationwide -/
def vision_condition_survey (nationwide : Prop) : Prop :=
  large_survey_scope_requires_sampling nationwide

/-- Condition 4: Understanding the lifespan of a batch of light bulbs -/
def light_bulb_lifespan_survey (destructive_process : Prop) : Prop :=
  destructive_process_requires_sampling destructive_process

/-- Condition 5: Understanding the per capita income of families in Hanjiang District -/
def per_capita_income_survey (large_scope : Prop) : Prop :=
  large_survey_scope_requires_sampling large_scope

/-- Proving that understanding the math exam scores of a class is suitable for a comprehensive survey -/
theorem math_exam_scores_comprehensive_survey
  (small_scope : Prop)
  (suitable_comprehensive : small_scope) :
  comprehensive_survey :=
begin
  sorry
end

end math_exam_scores_comprehensive_survey_l368_368071


namespace distance_is_4sqrt5_midpoint_is_neg2_5_l368_368678

/-- Define the points -/
def point1 := (2 : ℝ, 3 : ℝ)
def point2 := (-6 : ℝ, 7 : ℝ)

/-- Define the distance formula -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

/-- Define the midpoint formula -/
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

/-- The theorem that states the distance between the points is 4√5 -/
theorem distance_is_4sqrt5 :
  distance point1 point2 = 4 * real.sqrt 5 := sorry

/-- The theorem that states the midpoint coordinates are (-2, 5) -/
theorem midpoint_is_neg2_5 :
  midpoint point1 point2 = (-2 : ℝ, 5 : ℝ) := sorry

end distance_is_4sqrt5_midpoint_is_neg2_5_l368_368678


namespace range_of_dot_product_l368_368377

theorem range_of_dot_product (A B C : ℝ)
  (hAB : dist A B = 4)
  (hB : angle A B C = π / 3)
  (hA_range : π / 6 < angle B A C ∧ angle B A C < π / 2) :
  0 < (\overrightarrow{AB} \cdot \overrightarrow{AC}) ∧ (\overrightarrow{AB} \cdot \overrightarrow{AC}) < 12 :=
sorry

end range_of_dot_product_l368_368377


namespace tangent_line_to_curve_at_point_l368_368002

-- Define the function
def f (x : ℝ) : ℝ := x * (3 * Real.log x + 1)

-- The definition of the point
def point : (ℝ × ℝ) := (1, 1)

-- The equation of the tangent line at the given point
def tangent_line_eq (x : ℝ) : ℝ := 4 * x - 3

theorem tangent_line_to_curve_at_point :
  ∀ (x y : ℝ), (x = 1 ∧ y = 1) → (f x = y) → ∀ (t : ℝ), tangent_line_eq t = 4 * t - 3 := by
  assume x y hxy hfx t
  sorry

end tangent_line_to_curve_at_point_l368_368002


namespace complement_union_when_m_eq_neg2_range_of_m_if_B_subset_A_l368_368257

open Set

variable A : Set ℝ
variable B : ℝ → Set ℝ

definition A_def : A = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

definition B_def (m : ℝ) : B m = {x : ℝ | m ≤ x ∧ x ≤ m + 1} := by sorry

theorem complement_union_when_m_eq_neg2 :
  complement (λ x, -2 ≤ x ∧ x ≤ 2) = (λ x, x < -2 ∨ 2 < x) :=
by sorry

theorem range_of_m_if_B_subset_A :
  {m : ℝ | ∀ x : ℝ, (B m x → A x)} = {m : ℝ | -1 ≤ m ∧ m ≤ 1} :=
by sorry

end complement_union_when_m_eq_neg2_range_of_m_if_B_subset_A_l368_368257


namespace laila_scores_possible_values_l368_368810

theorem laila_scores_possible_values :
  ∃ (num_y_values : ℕ), num_y_values = 4 ∧ 
  (∀ (x y : ℤ), 0 ≤ x ∧ x ≤ 100 ∧
                 0 ≤ y ∧ y ≤ 100 ∧
                 4 * x + y = 410 ∧
                 y > x → 
                 (y = 86 ∨ y = 90 ∨ y = 94 ∨ y = 98)
  ) :=
  ⟨4, by sorry⟩

end laila_scores_possible_values_l368_368810


namespace C_share_of_profit_l368_368975

def A_investment : ℕ := 12000
def B_investment : ℕ := 16000
def C_investment : ℕ := 20000
def total_profit : ℕ := 86400

theorem C_share_of_profit: 
  (C_investment / (A_investment + B_investment + C_investment) * total_profit) = 36000 :=
by
  sorry

end C_share_of_profit_l368_368975


namespace derivative_dy_dx_l368_368198

noncomputable def x (t : ℝ) : ℝ := Real.exp (Real.sec t ^ 2)

noncomputable def y (t : ℝ) : ℝ := (Real.tan t) * Real.log (Real.cos t) + Real.tan t - t

noncomputable def dy_dx (t : ℝ) : ℝ := 
  (1 / 2) * (Real.cot t) * (Real.log (Real.cos t)) * Real.exp (-(Real.sec t ^ 2))

theorem derivative_dy_dx (t : ℝ) : 
  let dx_dt := (Real.exp (Real.sec t ^ 2)) * 2 * (Real.tan t) * (Real.sec t ^ 2)
  let dy_dt := (Real.log (Real.cos t)) * (Real.sec t ^ 2)
  dy_dx t = dy_dt / dx_dt := by
  sorry

end derivative_dy_dx_l368_368198


namespace periodic_symmetry_mono_f_l368_368951

-- Let f be a function from ℝ to ℝ.
variable (f : ℝ → ℝ)

-- f has the domain of ℝ.
-- f(x) = f(x + 6) for all x ∈ ℝ.
axiom periodic_f : ∀ x : ℝ, f x = f (x + 6)

-- f is monotonically decreasing in (0, 3).
axiom mono_f : ∀ ⦃x y : ℝ⦄, 0 < x → x < y → y < 3 → f y < f x

-- The graph of f is symmetric about the line x = 3.
axiom symmetry_f : ∀ x : ℝ, f x = f (6 - x)

-- Prove that f(3.5) < f(1.5) < f(6.5).
theorem periodic_symmetry_mono_f : f 3.5 < f 1.5 ∧ f 1.5 < f 6.5 :=
sorry

end periodic_symmetry_mono_f_l368_368951


namespace a_even_is_perfect_square_l368_368835

def is_number_with_sum_of_digits (n : ℕ) : ℕ → Prop
| 0 => n = 0
| N + 1 => ((N % 10 = 1 ∨ N % 10 = 3 ∨ N % 10 = 4) ∧ is_number_with_sum_of_digits (n - (N % 10))) ∧ (N / 10).is_number_with_sum_of_digits m

def a (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 1
  else a (n-1) + a (n-3) + a (n-4)

theorem a_even_is_perfect_square (n : ℕ) : ∃ k : ℕ, k * k = a (2 * n) :=
sorry

end a_even_is_perfect_square_l368_368835


namespace candy_vs_chocolate_l368_368983

theorem candy_vs_chocolate
  (candy1 candy2 chocolate : ℕ)
  (h1 : candy1 = 38)
  (h2 : candy2 = 36)
  (h3 : chocolate = 16) :
  (candy1 + candy2) - chocolate = 58 :=
by
  sorry

end candy_vs_chocolate_l368_368983


namespace intersection_area_is_correct_l368_368536

noncomputable def area_of_intersection_of_circles (r : ℝ) (c1 c2 : ℝ × ℝ) : ℝ :=
  let d := real.dist c1 c2 in
  if d ≥ 2 * r then 0
  else
    let θ1 := 2 * real.arccos (d / (2 * r)) in
    let θ2 := θ1 in
    2 * (r^2 * (θ1 / 2 - (real.sin θ1 / 2) * (r * real.sin (θ1 / 2))))

theorem intersection_area_is_correct :
  area_of_intersection_of_circles 3 (3, 0) (0, 3) = (9 * real.pi / 2 - 27) :=
by sorry

end intersection_area_is_correct_l368_368536


namespace digit_not_two_in_decimal_of_c_l368_368450

open Real

def a : ℕ := 10^6
def b : ℕ := 3141592
def c : ℝ := Real.cbrt 3

theorem digit_not_two_in_decimal_of_c :
  ∃ k : ℕ, a ≤ k ∧ k ≤ b ∧ (decimal_digit c k ≠ 2) := 
sorry

end digit_not_two_in_decimal_of_c_l368_368450


namespace pascal_triangle_row_sum_l368_368276

theorem pascal_triangle_row_sum : (∑ n in Finset.range 30, n + 1) = 465 := by
  sorry

end pascal_triangle_row_sum_l368_368276


namespace failing_grades_on_saturday_l368_368872

-- Conditions: Seven students in a class receive one failing grade every two days
constant Group1_students : Nat := 7
constant Group1_grades_per_two_days : Nat := 1

-- Nine other students receive one failing grade every three days
constant Group2_students : Nat := 9
constant Group2_grades_per_three_days : Nat := 1

-- From Monday to Friday, 30 new failing grades appeared in the class register
constant grades_monday_to_friday : Nat := 30

-- Prove that the number of new failing grades appearing in the class register on Saturday is 9
theorem failing_grades_on_saturday : 
  let grades_group1 : Nat := Group1_students * (6 / 2) * Group1_grades_per_two_days in
  let grades_group2 : Nat := Group2_students * (6 / 3) * Group2_grades_per_three_days in
  let total_grades_in_six_days : Nat := grades_group1 + grades_group2 in
  total_grades_in_six_days - grades_monday_to_friday = 9 :=
by
  sorry

end failing_grades_on_saturday_l368_368872


namespace rhombus_proportion_l368_368113

variables {A B C D E F P Q : Type}
variables [EuclideanGeometry ABCD] -- Assumes this bundles necessary axioms
open EuclideanGeometry -- Open geometry definitions

theorem rhombus_proportion
    (h_rhombus : rhombus ABCD)
    (h_angle_BAD : ∠BAD = 60)
    (h_E : E ∈ segment AB)
    (h_F : F ∈ segment AD)
    (h_angle_ECF : ∠ECF = ∠ABD)
    (h_CEP : CE ∈ line BD)
    (h_CFP : CF ∈ line BD)
    (h_P_intersection : P = intersection CE BD)
    (h_Q_intersection : Q = intersection CF BD) :
  (length PQ / length EF = length AB / length BD) :=
begin
  sorry
end

end rhombus_proportion_l368_368113


namespace rotate_180_about_C_l368_368868

theorem rotate_180_about_C {A B C : Type} 
  (ax : ℤ) (ay : ℤ) (bx : ℤ) (by : ℤ) (cx : ℤ) (cy : ℤ) 
  (hA : ax = -4 ∧ ay = 1) 
  (hC : cx = -1 ∧ cy = 1)
  (hB : bx = -1 ∧ by = 4):
  (2 * cx - ax, 2 * cy - ay) = (2, 1) := 
by
  -- Use the conditions provided to prove the coordinates
  rw [← hA, ← hC]
  sorry

end rotate_180_about_C_l368_368868


namespace prove_mb_product_l368_368489

-- Definitions for the points and the line equation conditions
def Point : Type := ℝ × ℝ
def Line (m b : ℝ) : ℝ → ℝ := λ x, m * x + b

-- The given points
def P1 : Point := (1, 4)
def P2 : Point := (-2, -2)

-- The slope calculation condition
def slope (P1 P2 : Point) : ℝ :=
  (P2.2 - P1.2) / (P2.1 - P1.1)

-- The y-intercept calculation condition
def y_intercept (m : ℝ) (P : Point) : ℝ :=
  P.2 - m * P.1

-- The product mb to be computed
def mb (m b : ℝ) : ℝ :=
  m * b

-- The problem statement in Lean 4
theorem prove_mb_product :
  let m := slope P1 P2 in
  let b := y_intercept m P1 in
  mb m b = 4 := by
  sorry

end prove_mb_product_l368_368489


namespace monomial_same_type_C_l368_368976

def same_type (m1 m2 : MvPolynomial (Fin 2) ℝ) : Prop :=
  ∀ n, m1.coeff n > 0 → m2.coeff n > 0 ∧ ∀ m, (m1.coeff n = 0) = (m2.coeff n = 0)

theorem monomial_same_type_C :
  let a : MvPolynomial (Fin 2) ℝ := MvPolynomial.monomial (special_basis_monomial 0 3 1) 1
  let b : MvPolynomial (Fin 2) ℝ := MvPolynomial.monomial (special_basis_monomial 0 1 3) 2
  let c : MvPolynomial (Fin 2) ℝ := MvPolynomial.monomial (special_basis_monomial 0 3 1) (-4)
  let d : MvPolynomial (Fin 2) ℝ := MvPolynomial.monomial (special_basis_monomial 0 3 3) 3
  (same_type (MvPolynomial.monomial (special_basis_monomial 0 3 1) 1) c) ∧
  ¬(same_type (MvPolynomial.monomial (special_basis_monomial 0 3 1) 1) a) ∧
  ¬(same_type (MvPolynomial.monomial (special_basis_monomial 0 3 1) 1) b) ∧
  ¬(same_type (MvPolynomial.monomial (special_basis_monomial 0 3 1) 1) d) :=
by
  sorry

end monomial_same_type_C_l368_368976


namespace area_not_in_choices_l368_368660

-- Definitions of the points
def point1 := (0, 5)
def point2 := (5, 0)
def point3 := (1, 0)
def point4 := (5, 3)

-- Line equations defined by the points
def line1 (x : ℝ) := -x + 5
def line2 (x : ℝ) := (3 / 4) * x - 3 / 4

-- Integral bounds
def x_lower_bound := 1
def x_upper_bound := 5

-- Theorem stating the region's area is not in the given choices
theorem area_not_in_choices : 
    let area := abs (∫ x in (x_lower_bound)..(19 / 7), line2 x - 0)
             + abs (∫ x in (19 / 7)..(x_upper_bound), line1 x - line2 x)
    in area ≠ 2 ∧ area ≠ 3 ∧ area ≠ 4 ∧ area ≠ 5 :=
sorry

end area_not_in_choices_l368_368660


namespace pascals_triangle_total_numbers_l368_368298

theorem pascals_triangle_total_numbers (N : ℕ) (hN : N = 29) :
  (∑ n in Finset.range (N + 1), (n + 1)) = 465 :=
by
  rw hN
  calc (∑ n in Finset.range 30, (n + 1))
      = ∑ k in Finset.range 30, (k + 1) : rfl
  -- Here we are calculating the sum of the first 30 terms of the sequence (n + 1)
  ... = 465 : sorry

end pascals_triangle_total_numbers_l368_368298


namespace smallest_degree_polynomial_l368_368110

theorem smallest_degree_polynomial 
  (P : Polynomial ℚ) 
  (hP : ∀ n, 1 ≤ n ∧ n ≤ 500 → IsRoot P (n + Real.sqrt (2 * n + 1))) : 
  P.degree = 1000 := 
sorry

end smallest_degree_polynomial_l368_368110


namespace radius_first_field_l368_368902

theorem radius_first_field (r_2 : ℝ) (h_r2 : r_2 = 10) (h_area : ∃ A_2, ∃ A_1, A_1 = 0.09 * A_2 ∧ A_2 = π * r_2^2) : ∃ r_1 : ℝ, r_1 = 3 :=
by
  sorry

end radius_first_field_l368_368902


namespace cost_per_meter_of_fencing_l368_368903

/-- The sides of the rectangular field -/
def sides_ratio (length width : ℕ) : Prop := 3 * width = 4 * length

/-- The area of the rectangular field -/
def area (length width area : ℕ) : Prop := length * width = area

/-- The cost per meter of fencing -/
def cost_per_meter (total_cost perimeter : ℕ) : ℕ := total_cost * 100 / perimeter

/-- Prove that the cost per meter of fencing the field in paise is 25 given:
 1) The sides of a rectangular field are in the ratio 3:4.
 2) The area of the field is 8112 sq. m.
 3) The total cost of fencing the field is 91 rupees. -/
theorem cost_per_meter_of_fencing
  (length width perimeter : ℕ) 
  (h1 : sides_ratio length width)
  (h2 : area length width 8112)
  (h3 : perimeter = 2 * (length + width))
  (total_cost : ℕ)
  (h4 : total_cost = 91) :
  cost_per_meter total_cost perimeter = 25 :=
by
  sorry

end cost_per_meter_of_fencing_l368_368903


namespace angle_A_measure_l368_368152

theorem angle_A_measure (A B C D E : ℝ) 
(h1 : A = 3 * B)
(h2 : A = 4 * C)
(h3 : A = 5 * D)
(h4 : A = 6 * E)
(h5 : A + B + C + D + E = 540) : 
A = 277 :=
by
  sorry

end angle_A_measure_l368_368152


namespace power_function_quadrants_l368_368562

-- Define the conditions
constant α : ℝ
noncomputable def power_function := λ x : ℝ, x^α

-- Define the quadrants and the proof goal
theorem power_function_quadrants : 
  (∀ x ∈ {x : ℝ | x < 0 ∧ x^α < 0}, x < 0) ∧ 
  (∀ x ∈ {x : ℝ | x < 0 ∧ x^α > 0}, x > 0) :=
sorry

end power_function_quadrants_l368_368562


namespace problem1_problem2_l368_368874

-- Part (1)
theorem problem1 : Real.exp (Real.ln 3) + Real.log (Real.sqrt 5) 25 + (0.125) ^ (-2 / 3) = 11 := 
sorry

-- Part (2)
variable (a : ℝ)
theorem problem2 (h : Real.sqrt a + (Real.sqrt a)⁻¹ = 3) : a^2 + a⁻² = 47 :=
sorry

end problem1_problem2_l368_368874


namespace solve_system_l368_368461

theorem solve_system :
  ∃ x y : ℝ, 3^x * 2^y = 972 ∧ log (sqrt 3) (x - y) = 2 ∧ x = 5 ∧ y = 2 :=
by { sorry }

end solve_system_l368_368461


namespace ratio_pen_pencil_l368_368853

theorem ratio_pen_pencil (P : ℝ) (pencil_cost total_cost : ℝ) 
  (hc1 : pencil_cost = 8) 
  (hc2 : total_cost = 12)
  (hc3 : P + pencil_cost = total_cost) : 
  P / pencil_cost = 1 / 2 :=
by 
  sorry

end ratio_pen_pencil_l368_368853


namespace matrix_determinant_l368_368818

variables (a b c : ℝ^3)

noncomputable def E : ℝ := Matrix.det ![![2 * a, 3 * b, 4 * c]].transpose

theorem matrix_determinant (a b c : ℝ^3) :
  Matrix.det ![![2 * a + 3 * b, 3 * b + 4 * c, 4 * c + 2 * a]].transpose = 48 * E a b c :=
sorry

end matrix_determinant_l368_368818


namespace fewer_onions_grown_l368_368638

def num_tomatoes := 2073
def num_cobs_of_corn := 4112
def num_onions := 985

theorem fewer_onions_grown : num_tomatoes + num_cobs_of_corn - num_onions = 5200 := by
  sorry

end fewer_onions_grown_l368_368638


namespace approx_probability_defective_l368_368941

noncomputable def probability_defective (total_phones : ℕ) (defective_phones : ℕ) : ℝ :=
  let pA := (defective_phones : ℝ) / (total_phones : ℝ)
  let pB_given_A := (defective_phones - 1 : ℝ) / (total_phones - 1 : ℝ)
  pA * pB_given_A

theorem approx_probability_defective :
  probability_defective 240 84 ≈ 0.12145 :=
by
  unfold probability_defective
  norm_num
  sorry

end approx_probability_defective_l368_368941


namespace extraordinary_numbers_in_interval_l368_368844

def is_extraordinary (n : ℕ) : Prop :=
  n % 2 = 0 ∧ ∃ p : ℕ, Nat.Prime p ∧ n = 2 * p

def count_extraordinary (a b : ℕ) : ℕ :=
  (Finset.filter (λ n, is_extraordinary n) (Finset.range' a (b + 1))).card

theorem extraordinary_numbers_in_interval :
  count_extraordinary 1 75 = 12 := 
sorry

end extraordinary_numbers_in_interval_l368_368844


namespace eccentricity_of_ellipse_l368_368819

-- Definitions
def ellipse_eq : ℝ → ℝ → ℝ → ℝ → Prop :=
  λ a b x y, (x^2 / 4 + y^2 / b^2 = 1)

def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-a, 0), (a, 0))

def line_through_focus (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  p.1 * y - p.2 * x = 0

-- Problem statement
theorem eccentricity_of_ellipse (a b: ℝ) (F1 F2 : ℝ × ℝ)
    (A B : ℝ × ℝ) (h1 : F1 = (-a, 0)) (h2 : F2 = (a, 0))
    (h3 : ellipse_eq a b A.1 A.2) (h4 : ellipse_eq a b B.1 B.2)
    (h5 : line_through_focus F1 A.1 A.2) (h6 : line_through_focus F1 B.1 B.2)
    (h_max : |(A.1 - F2.1, A.2 - F2.2)| + |(B.1 - F2.1, B.2 - F2.2)| = 5):
  sqrt (1 - (b^2 / a^2)) = sqrt 3 / 2 :=
sorry

end eccentricity_of_ellipse_l368_368819


namespace base_k_rep_fraction_eq_l368_368689

theorem base_k_rep_fraction_eq (k : ℕ) (h_pos : k > 1) : 
  (k = 14) ↔ (0.\overline{14}_k : ℚ) = 5 / 31 :=
by
  sorry

end base_k_rep_fraction_eq_l368_368689


namespace poster_area_l368_368153

theorem poster_area (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : (3 * x + 4) * (y + 3) = 63) : x * y = 15 :=
begin
  sorry
end

end poster_area_l368_368153


namespace unique_n_divisible_by_210_l368_368263

theorem unique_n_divisible_by_210 
  (n : ℤ)
  (h1 : 1 ≤ n) 
  (h2 : n ≤ 210)
  (h3 : ∀ k, 1 ≤ k ∧ k ≤ 2013 → (n + k.factorial) % 210 = 0) :
  n = 1 := 
by 
  sorry

end unique_n_divisible_by_210_l368_368263


namespace profit_per_unit_and_minimum_units_l368_368856

noncomputable def conditions (x y m : ℝ) : Prop :=
  2 * x + 7 * y = 41 ∧
  x + 3 * y = 18 ∧
  0.5 * m + 0.3 * (30 - m) ≥ 13.1

theorem profit_per_unit_and_minimum_units (x y m : ℝ) :
  conditions x y m → x = 3 ∧ y = 5 ∧ m ≥ 21 :=
by
  sorry

end profit_per_unit_and_minimum_units_l368_368856


namespace domain_of_tan_shifted_l368_368486

open Real

noncomputable def is_domain_of_tan_shifted (x : ℝ) : Prop :=
  ∀ (k : ℤ), (x ≠ 2 * k * π + π / 2)

theorem domain_of_tan_shifted (x : ℝ) :
  (∀ (z : ℩), (1 / 2) * x + π / 4 ≠ z * π + π / 2) →
  is_domain_of_tan_shifted x :=
by
  intro h
  rw [is_domain_of_tan_shifted]
  intro k
  specialize h (k : ℩)
  linarith

end domain_of_tan_shifted_l368_368486


namespace perfect_squares_as_difference_l368_368750

theorem perfect_squares_as_difference (N : ℕ) (hN : N = 20000) : 
  (∃ (n : ℕ), n = 71 ∧ 
    ∀ m < N, 
      (∃ a b : ℤ, 
        a^2 = m ∧
        b^2 = m + ((b + 1)^2 - b^2) - 1 ∧ 
        (b + 1)^2 - b^2 = 2 * b + 1)) :=
by 
  sorry

end perfect_squares_as_difference_l368_368750


namespace mt_product_l368_368407

def g : ℝ → ℝ := sorry

axiom func_eqn (x y : ℝ) : g (x * g y + 2 * x) = 2 * x * y + g x

axiom g3_value : g 3 = 6

def m : ℕ := 1

def t : ℝ := 6

theorem mt_product : m * t = 6 :=
by 
  sorry

end mt_product_l368_368407


namespace tangent_line_at_x1_l368_368490

noncomputable def curve (x : ℝ) : ℝ := x^3 - 2 * x + 3

def tangent_line (x : ℝ) : ℝ := x - 2 + 1

theorem tangent_line_at_x1 :
  ∃ (m b : ℝ), (∀ (x : ℝ), curve x = x * m + b) ∧ (m = 1) ∧ (curve 1 = 1 - 2 + 3) :=
begin
  use 1,
  use 1,
  split,
  { intro x,
    simp [curve, tangent_line],
    sorry, -- Proof will follow here },
  { split,
    { refl },
    { norm_num } }
end

end tangent_line_at_x1_l368_368490


namespace quadrilateral_areas_l368_368417

theorem quadrilateral_areas (ABCD : Quadrilateral) (hABCD : area ABCD = 1) :
  ∃ (P Q R S : Point), 
    (P ∈ edges ABCD ∨ P ∈ interior ABCD) ∧ 
    (Q ∈ edges ABCD ∨ Q ∈ interior ABCD) ∧
    (R ∈ edges ABCD ∨ R ∈ interior ABCD) ∧ 
    (S ∈ edges ABCD ∨ S ∈ interior ABCD) ∧ 
    ∀ (X Y Z : Point), {X, Y, Z} ⊆ {P, Q, R, S} → area (triangle X Y Z) > 1/4 := 
sorry

end quadrilateral_areas_l368_368417


namespace fewer_onions_than_tomatoes_and_corn_l368_368639

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions_than_tomatoes_and_corn :
  (tomatoes + corn - onions) = 5200 :=
by
  sorry

end fewer_onions_than_tomatoes_and_corn_l368_368639


namespace required_percentage_increase_is_approximately_74_29_l368_368901

noncomputable def original_price : ℝ := 100
noncomputable def first_discounted_price : ℝ := original_price * (1 - 0.25)
noncomputable def second_discounted_price : ℝ := first_discounted_price * (1 - 0.15)
noncomputable def third_discounted_price : ℝ := second_discounted_price * (1 - 0.1)

theorem required_percentage_increase_is_approximately_74_29 :
  ( (100 / third_discounted_price) - 1) * 100 ≈ 74.29 := 
sorry

end required_percentage_increase_is_approximately_74_29_l368_368901


namespace planes_perpendicular_l368_368211

def u : ℝ × ℝ × ℝ := (-2, 2, 5)
def v : ℝ × ℝ × ℝ := (6, -4, 4)

theorem planes_perpendicular (α β : Plane) (hu : α.normal_vector = u) (hv : β.normal_vector = v) :
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0 :=
by simp [u, v]; norm_num; exact rfl

end planes_perpendicular_l368_368211


namespace probability_of_70th_percentile_is_25_over_56_l368_368795

-- Define the weights of the students
def weights : List ℕ := [90, 100, 110, 120, 140, 150, 150, 160]

-- Define the number of students to select
def n_selected_students : ℕ := 3

-- Define the percentile value
def percentile_value : ℕ := 70

-- Define the corresponding weight for the 70th percentile
def percentile_weight : ℕ := 150

-- Define the combination function
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability calculation
noncomputable def probability_70th_percentile : ℚ :=
  let total_ways := C 8 3
  let favorable_ways := (C 2 2) * (C 5 1) + (C 2 1) * (C 5 2)
  favorable_ways / total_ways

-- Define the theorem to prove the probability
theorem probability_of_70th_percentile_is_25_over_56 :
  probability_70th_percentile = 25 / 56 := by
  sorry

end probability_of_70th_percentile_is_25_over_56_l368_368795


namespace circles_through_at_least_three_points_count_l368_368088

open set

def is_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := p1 in
  let ⟨x2, y2⟩ := p2 in
  let ⟨x3, y3⟩ := p3 in
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

def distinct_circles_pass_three_points_count (A : set (ℝ × ℝ)) : ℕ :=
  let combinations := powerset_len 3 A in
  let non_collinear_combinations := filter (λ s, ∃ p1 p2 p3, s = {p1, p2, p3} ∧ ¬is_collinear p1 p2 p3) combinations in
  let total_circles := card non_collinear_combinations in
  -- Correct the count for overlapping circles.
  total_circles - (overlapping_circles_count A)

def overlapping_circles_count (A : set (ℝ × ℝ)) : ℕ := 6 -- This should be the real overlapping circles count logic.

theorem circles_through_at_least_three_points_count :
  let A := {(0, 0), (2, 0), (0, 2), (2, 2), (3, 1), (1, 3)} in
  distinct_circles_pass_three_points_count A = 13 := by
  sorry

end circles_through_at_least_three_points_count_l368_368088


namespace common_factor_of_polynomials_l368_368481

noncomputable def P (x : ℝ) := x^2 - 2 * x - 3
noncomputable def Q (x : ℝ) := x^2 - 6 * x + 9
def common_factor (x : ℝ) := x - 3

theorem common_factor_of_polynomials : ∀ x, (P x % common_factor x = 0) ∧ (Q x % common_factor x = 0) :=
by
  intro x
  have h1: P x = (x - 3) * (x + 1),
  -- Factor P(x)
  sorry,
  have h2: Q x = (x - 3) * (x - 3),
  -- Factor Q(x)
  sorry,
  split,
  -- Show P(x) is divisible by (x - 3)
  {
    suffices : (x - 3) ∣ P x, 
    -- Proving suffices the condition
    {
      convert this,
      sorry
    },
    exact dvd_of_mul_right_eq h1
  },
  -- Show Q(x) is divisible by (x - 3)
  {
    suffices : (x - 3) ∣ Q x,
    -- Proving suffices the condition
    {
      convert this,
      sorry
    },
    exact dvd_of_mul_right_eq h2
  }

end common_factor_of_polynomials_l368_368481


namespace probability_one_out_of_three_l368_368623

def probability_passing_exactly_one (p : ℚ) (n k : ℕ) :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_one_out_of_three :
  probability_passing_exactly_one (1/3) 3 1 = 4/9 :=
by sorry

end probability_one_out_of_three_l368_368623


namespace cost_of_pants_is_250_l368_368987

variable (costTotal : ℕ) (costTShirt : ℕ) (numTShirts : ℕ) (numPants : ℕ)

def costPants (costTotal costTShirt numTShirts numPants : ℕ) : ℕ :=
  let costTShirts := numTShirts * costTShirt
  let costPantsTotal := costTotal - costTShirts
  costPantsTotal / numPants

-- Given conditions
axiom h1 : costTotal = 1500
axiom h2 : costTShirt = 100
axiom h3 : numTShirts = 5
axiom h4 : numPants = 4

-- Prove each pair of pants costs $250
theorem cost_of_pants_is_250 : costPants costTotal costTShirt numTShirts numPants = 250 :=
by
  -- Place proof here
  sorry

end cost_of_pants_is_250_l368_368987


namespace pascal_triangle_row_sum_l368_368279

theorem pascal_triangle_row_sum : (∑ n in Finset.range 30, n + 1) = 465 := by
  sorry

end pascal_triangle_row_sum_l368_368279


namespace diff_of_squares_example_l368_368559

theorem diff_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end diff_of_squares_example_l368_368559


namespace trajectory_of_center_of_circle_parabola_l368_368949

theorem trajectory_of_center_of_circle_parabola :
  (∀ (C : Type) (center : C → ℝ × ℝ) (radius : C → ℝ),
    (∀ p1 : ℝ × ℝ, p1 = (0, 3) →
      (∀ r1, (x^2 + (y-3)^2 = 1) →
      (∀ line1 : ℝ → Prop, line1 = (y = 0) →
        (radius C + 1 = distance (center C) p1) ∧ (radius C = abs (center C).snd) →
        (distance (center C) p1 = abs ((center C).snd + 1)) →
        trajectory_of (center C) = parabola)))) sorry

end trajectory_of_center_of_circle_parabola_l368_368949


namespace q_investment_time_l368_368020

theorem q_investment_time (x t : ℝ)
  (h1 : (7 * 20 * x) / (5 * t * x) = 7 / 10) : t = 40 :=
by
  sorry

end q_investment_time_l368_368020


namespace angle_between_AC_AB_is_90_degrees_l368_368937

def isEquilateralTriangle (A B C : Point) (side_length : ℝ) : Prop := 
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

def isPerpendicular (P A B C : Point) : Prop := 
  ∀ (X : Point), X ∈ plane A B C → (PA • X) = 0

def reflection (P A B C A' : Point) : Prop := 
  ∀ (X : Point), X ∈ plane P B C → dist A' X = dist A X

def find_angle_between_AC_AB (A B C P A'_ O : Point) (side_length PA_length : ℝ) : Prop := 
  isEquilateralTriangle A B C side_length ∧
  isPerpendicular P A (plane A B C) ∧ 
  dist P A = PA_length ∧
  reflection P A B C A' →
  angle_between (A', C) (A, B) = 90°

theorem angle_between_AC_AB_is_90_degrees (A B C P A' : Point) :
  find_angle_between_AC_AB A B C P A' (1 : ℝ) (sqrt(6)/4) :=
by 
  sorry

end angle_between_AC_AB_is_90_degrees_l368_368937


namespace num_sequences_of_student_helpers_l368_368981

-- Define the conditions
def num_students : ℕ := 15
def num_meetings : ℕ := 3

-- Define the statement to prove
theorem num_sequences_of_student_helpers : 
  (num_students ^ num_meetings) = 3375 :=
by sorry

end num_sequences_of_student_helpers_l368_368981


namespace midpoint_coordinates_l368_368411

theorem midpoint_coordinates
  (x1 y1 x2 y2 x0 y0 : ℝ)
  (h : ∀ (x : ℝ) (y : ℝ), x = y ↔ (2 * y = x + y)) 
  (midpoint : x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2) : 
  x0 = \frac{1}{2} * (x1 + x2) ∧ y0 = \frac{1}{2} * (y1 + y2) := 
sorry

end midpoint_coordinates_l368_368411


namespace find_range_of_k_l368_368250

theorem find_range_of_k (a b k : ℝ)
  (h : a < b)
  (f : ℝ → ℝ := λ x, sqrt (x + 2) + k)
  (H : ∀ y, y ∈ set.Icc a b ↔ ∃ x ∈ set.Icc a b, f x = y) :
  k ∈ set.Ioc (-(9/4) : ℝ) (-2 : ℝ) := by
  sorry

end find_range_of_k_l368_368250


namespace find_a_l368_368368

open Real

-- Definitions for the curves C1 and C2
def C1 (a t : ℝ) : ℝ × ℝ := (a + sqrt 2 * t, 1 + sqrt 2 * t)

def C2 (x y : ℝ) : Prop := y^2 = 4 * x

-- Proof statement
theorem find_a (a : ℝ) 
  (h₁ : ∀ t : ℝ, ∃ x y : ℝ, (x, y) = C1 a t)
  (h₂ : ∀ x y : ℝ, C2 x y)
  (PA : ℝ × ℝ, PB : ℝ × ℝ)
  (h₃ : PA = (a, 1))
  (h₄ : ∃ A B : ℝ × ℝ, A ∈ range (λ t, C1 a t) ∧ B ∈ range (λ t, C1 a t) ∧ C2 A.1 A.2 ∧ C2 B.1 B.2 ∧ abs (PA.1 - A.1) + abs (PA.2 - A.2) = 2 * (abs (PB.1 - B.1) + abs (PB.2 - B.2))) :
  a = 1/36 ∨ a = 9/4 :=
sorry

end find_a_l368_368368


namespace championship_winner_is_902_l368_368508

namespace BasketballMatch

inductive Class : Type
| c901
| c902
| c903
| c904

open Class

def A_said (champ third : Class) : Prop :=
  champ = c902 ∧ third = c904

def B_said (fourth runner_up : Class) : Prop :=
  fourth = c901 ∧ runner_up = c903

def C_said (third champ : Class) : Prop :=
  third = c903 ∧ champ = c904

def half_correct (P Q : Prop) : Prop := 
  (P ∧ ¬Q) ∨ (¬P ∧ Q)

theorem championship_winner_is_902 (A_third B_fourth B_runner_up C_third : Class) 
  (H_A : half_correct (A_said c902 A_third) (A_said A_third c902))
  (H_B : half_correct (B_said B_fourth B_runner_up) (B_said B_runner_up B_fourth))
  (H_C : half_correct (C_said C_third c904) (C_said c904 C_third)) :
  ∃ winner, winner = c902 :=
sorry

end BasketballMatch

end championship_winner_is_902_l368_368508


namespace part_a_part_b_l368_368176

namespace problem

variables (n k : ℕ) (numbers : Finset ℕ)

-- Condition: Each subsequent number is chosen independently of the previous ones.
-- Condition: If a chosen number matches one of the already chosen numbers, move clockwise to the first unchosen number.
-- Condition: In the end, k different numbers are obtained.

-- Part (a): Appearance of each specific number is equally likely.
theorem part_a (h1 : ∀ (chosen : Finset ℕ), chosen.card = k → (∃! x ∈ numbers, x ∉ chosen)) :
  ∃ (p : ℚ), ∀ (x ∈ numbers), p = k / n :=
sorry

-- Part (b): Appearance of all selections is not equally likely.
theorem part_b (h1 : ∀ (chosen : Finset ℕ), chosen.card = k → (∃! x ∈ numbers, x ∉ chosen)) :
  ¬ ∃ (p : ℚ), ∀ (s : Finset ℕ), s.card = k → p = (card s.subevents) / (n ^ k) :=
sorry

end problem

end part_a_part_b_l368_368176


namespace smallest_difference_of_factors_l368_368752

theorem smallest_difference_of_factors (n : ℕ) (h : n = 2310) :
  ∃ (a b : ℕ), a * b = n ∧ |a - b| = 13 :=
sorry

end smallest_difference_of_factors_l368_368752


namespace max_container_volume_l368_368054

theorem max_container_volume :
  ∀ (x : ℝ), 0 < x ∧ 2x + 2 * (x + 0.5) + 4 * (3.45 - x) = 14.8 →
  (x * (x + 0.5) * (3.45 - x) ≤ 3.675) :=
by
  sorry

end max_container_volume_l368_368054


namespace find_DF_l368_368424

open Real

variables (D E F P Q : Type*) [metric_space D] [metric_space E] [metric_space F]
[metric_space P] [metric_space Q]

structure triangle_def :=
(D E F : D)
(is_right_triangle : is_right_triangle D E F)
(median_DP : P)
(median_EQ : Q)
(perpendicular_medians : P ⊥ Q)
(length_DP : DP = 15)
(length_EQ : EQ = 20)

theorem find_DF (h : triangle_def D E F P Q) : DF = (20 * sqrt 15) / 3 :=
sorry

end find_DF_l368_368424


namespace algebra_sum_l368_368893

noncomputable def letter_value (n : ℕ) : ℤ :=
  match n % 10 with
  | 0 | 6 | 7 => 2
  | 1 | 5 => 3
  | 2 | 8 => -2
  | 3 => 1
  | 4 => 0
  | 9 => -1
  | _ => 0  -- This should practically never happen

def algebra_value : ℤ :=
  letter_value 1 + letter_value 12 + letter_value 7 +
  letter_value 5 + letter_value 2 + letter_value 18 +
  letter_value 1

theorem algebra_sum : algebra_value = 5 := by
  have h1 : letter_value 1 = 2 := by rfl
  have h2 : letter_value 12 = 3 := by rfl
  have h3 : letter_value 7 = -2 := by rfl
  have h4 : letter_value 5 = 0 := by rfl
  have h5 : letter_value 2 = 3 := by rfl
  have h6 : letter_value 18 = -3 := by rfl
  have h7 : letter_value 1 = 2 := by rfl
  calc algebra_value
      = (2 + 3 - 2 + 0 + 3 - 3 + 2) : by rw [h1, h2, h3, h4, h5, h6, h7]
  ... = 5 : by ring

end algebra_sum_l368_368893


namespace probability_sum_9_l368_368910

theorem probability_sum_9 : 
  let bags := ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]),
      total_combinations := 6 * 6,
      successful_combinations := [ (3, 6), (6, 3), (4, 5), (5, 4) ] in
  (successful_combinations.length : ℚ) / total_combinations = 1 / 9 :=
by
  sorry

end probability_sum_9_l368_368910


namespace radius_of_sphere_l368_368108

-- Define the conditions.
def radius_wire : ℝ := 8
def length_wire : ℝ := 36

-- Given the volume of the metallic sphere is equal to the volume of the wire,
-- Prove that the radius of the sphere is 12 cm.
theorem radius_of_sphere (r_wire : ℝ) (h_wire : ℝ) (r_sphere : ℝ) : 
    r_wire = radius_wire → h_wire = length_wire →
    (π * r_wire^2 * h_wire = (4/3) * π * r_sphere^3) → 
    r_sphere = 12 :=
by
  intros h₁ h₂ h₃
  -- Add proof steps here.
  sorry

end radius_of_sphere_l368_368108


namespace pascal_triangle_sum_first_30_rows_l368_368293

theorem pascal_triangle_sum_first_30_rows :
  (Finset.range 30).sum (λ n, n + 1) = 465 :=
begin
  sorry
end

end pascal_triangle_sum_first_30_rows_l368_368293


namespace cone_base_circumference_l368_368099

theorem cone_base_circumference (r : ℝ) (sector_angle : ℝ) (total_angle : ℝ) (C : ℝ) (h1 : r = 6) (h2 : sector_angle = 180) (h3 : total_angle = 360) (h4 : C = 2 * r * Real.pi) :
  (sector_angle / total_angle) * C = 6 * Real.pi :=
by
  -- Skipping proof
  sorry

end cone_base_circumference_l368_368099


namespace arithmetic_series_sum_l368_368025

theorem arithmetic_series_sum (n P q S₃n : ℕ) (h₁ : 2 * S₃n = 3 * P - q) : S₃n = 3 * P - q :=
by
  sorry

end arithmetic_series_sum_l368_368025


namespace limit_half_derivative_neg_two_l368_368766

variable {ℝ : Type}
variable {f : ℝ → ℝ}
variable {x₀ : ℝ}

theorem limit_half_derivative_neg_two (h : tendsto (λ h : ℝ, f (x₀ - (1/2) * h)) (nhds_within 0 (set.Ioi 0)) (nhds (f x₀)))
  (h_deriv : deriv f x₀ = -2) :
  tendsto (λ h : ℝ, (f (x₀ - (1/2) * h) - f x₀) / h) (nhds_within 0 (set.Ioi 0)) (nhds (1 : ℝ)) :=
begin
  sorry -- proof goes here
end

end limit_half_derivative_neg_two_l368_368766


namespace cans_difference_l368_368995

theorem cans_difference 
  (n_cat_packages : ℕ) (n_dog_packages : ℕ) 
  (n_cat_cans_per_package : ℕ) (n_dog_cans_per_package : ℕ) :
  n_cat_packages = 6 →
  n_dog_packages = 2 →
  n_cat_cans_per_package = 9 →
  n_dog_cans_per_package = 3 →
  (n_cat_packages * n_cat_cans_per_package) - (n_dog_packages * n_dog_cans_per_package) = 48 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end cans_difference_l368_368995


namespace necessary_but_not_sufficient_l368_368933

theorem necessary_but_not_sufficient (a : ℝ) :
  (a < -1 → a < 1 / a)
  ∧ (∃ (b : ℝ), 0 < b ∧ b < 1 ∧ b < 1 / b) :=
begin
  -- Proof goes here
  sorry
end

end necessary_but_not_sufficient_l368_368933


namespace correct_number_of_statements_l368_368888

-- Definitions based on the problem's conditions
def condition_1 : Prop :=
  ∀ (n : ℕ) (a b c d e : ℚ), n = 5 ∧ ∃ x y z, (x = a ∧ y = b ∧ z = c) ∧ (x < 0 ∧ y < 0 ∧ z < 0 ∧ d ≥ 0 ∧ e ≥ 0) →
  (a * b * c * d * e < 0 ∨ a * b * c * d * e = 0)

def condition_2 : Prop := 
  ∀ m : ℝ, |m| + m = 0 → m ≤ 0

def condition_3 : Prop := 
  ∀ a b : ℝ, (1 / a < 1 / b) → ¬ (a < b ∨ b < a)

def condition_4 : Prop := 
  ∀ a : ℝ, ∃ max_val, max_val = 5 ∧ 5 - |a - 5| ≤ max_val

-- Main theorem to state the correct number of true statements
theorem correct_number_of_statements : 
  (condition_2 ∧ condition_4) ∧
  ¬condition_1 ∧ 
  ¬condition_3 :=
by
  sorry

end correct_number_of_statements_l368_368888


namespace derivative_at_1_correct_l368_368219

def f (x : ℝ) : ℝ := Real.sin x + Real.log x

theorem derivative_at_1_correct : (Real.deriv f 1) = Real.cos 1 + 1 :=
by
  -- Proof will be written here
  sorry

end derivative_at_1_correct_l368_368219


namespace unit_vector_orthogonal_to_v1_v2_l368_368671

-- Definitions for the given vectors
def v1 : ℝ × ℝ × ℝ := (2, 1, 1)
def v2 : ℝ × ℝ × ℝ := (0, 1, 3)

-- The unit vector that needs to be verified
def u : ℝ × ℝ × ℝ := (1 / Real.sqrt 11, -3 / Real.sqrt 11, 1 / Real.sqrt 11)

-- Proof statement
theorem unit_vector_orthogonal_to_v1_v2 : 
  (∥u∥ = 1) ∧ (u.1 * v1.1 + u.2 * v1.2 + u.3 * v1.3 = 0) ∧ (u.1 * v2.1 + u.2 * v2.2 + u.3 * v2.3 = 0) :=
by 
  sorry

end unit_vector_orthogonal_to_v1_v2_l368_368671


namespace train_cross_pole_time_l368_368077

def train_length : ℝ := 80 -- length of the train in meters
def train_speed_kmh : ℝ := 144 -- speed of the train in km/hr
def conversion_factor : ℝ := 1000 / 3600 -- conversion factor from km/hr to m/s
def train_speed_ms : ℝ := train_speed_kmh * conversion_factor -- speed of the train in m/s
def time_to_cross_pole : ℝ := train_length / train_speed_ms -- time to cross the pole in seconds

theorem train_cross_pole_time :
  time_to_cross_pole = 2 := by
  sorry

end train_cross_pole_time_l368_368077


namespace actual_time_is_1240pm_l368_368662

def kitchen_and_cellphone_start (t : ℕ) : Prop := t = 8 * 60  -- 8:00 AM in minutes
def kitchen_clock_after_breakfast (t : ℕ) : Prop := t = 8 * 60 + 30  -- 8:30 AM in minutes
def cellphone_after_breakfast (t : ℕ) : Prop := t = 8 * 60 + 20  -- 8:20 AM in minutes
def kitchen_clock_at_3pm (t : ℕ) : Prop := t = 15 * 60  -- 3:00 PM in minutes

theorem actual_time_is_1240pm : 
  (kitchen_and_cellphone_start 480) ∧ 
  (kitchen_clock_after_breakfast 510) ∧ 
  (cellphone_after_breakfast 500) ∧
  (kitchen_clock_at_3pm 900) → 
  real_time_at_kitchen_clock_time_3pm = 12 * 60 + 40 :=
by
  sorry

end actual_time_is_1240pm_l368_368662


namespace max_pairs_of_acquaintances_l368_368982

theorem max_pairs_of_acquaintances (n : ℕ) (h_n : n = 45) 
  (h_condition : ∀ (k : ℕ), 0 ≤ k ∧ k ≤ 44 → (∀ (a b : fin n), a ≠ b → (∃ (ga gb : set (fin n)), ga.card = k ∧ gb.card = k ∧ (a ∈ ga ∧ b ∈ gb) → ¬ (a ∈ gb) ∧ ¬ (b ∈ ga)))) :
  ∃ (max_pairs : ℕ), max_pairs = 870 := 
begin
  sorry
end

end max_pairs_of_acquaintances_l368_368982


namespace count_valid_tables_l368_368036

-- Defining the conditions
def no_porridge_liked_by_all_three (tables: list (list bool)): Prop :=
  ¬ ∃ (p: ℕ), (tables.all (λ(row: list bool), row.get p = tt))

def each_pair_likes_one (tables: list (list bool)): Prop :=
  ∀ (s1 s2: ℕ), s1 ≠ s2 → ∃ (p: ℕ), (tables.get s1).get p = tt ∧ (tables.get s2).get p = tt

-- Statement of the problem
noncomputable def number_of_valid_tables: ℕ :=
  132

theorem count_valid_tables (tables: list (list bool)) (h1: no_porridge_liked_by_all_three tables) (h2: each_pair_likes_one tables):
  tables.length = 132 :=
by
  sorry

end count_valid_tables_l368_368036


namespace triathlete_average_speed_l368_368625

theorem triathlete_average_speed :
  ∀ (L : ℝ) (t_swim t_bike t_run : ℝ),
  L / (2 : ℝ) = t_swim ∧
  L / (15 : ℝ) = t_bike ∧
  L / (8 : ℝ) = t_run →
  (3 * L) / (t_swim + t_bike + t_run) ≈ 4 :=
by
  intros L t_swim t_bike t_run h
  sorry

end triathlete_average_speed_l368_368625


namespace sqrt_inverse_is_inverse_l368_368497

noncomputable def sqrt_inverse (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 else 0

theorem sqrt_inverse_is_inverse (x : ℝ) : 
  (f : ℝ → ℝ) (f := fun x => sqrt x) → 
  (∀ x, x ≥ 0 → sqrt_inverse (f x) = x) :=
by
  intro f hf
  unfold sqrt_inverse
  split_ifs with h
  · sorry
  · contradiction -- since sqrt x is always non-negative

end sqrt_inverse_is_inverse_l368_368497


namespace problem1_problem2_l368_368716

noncomputable def f (a b x : ℝ) : ℝ := -a * x + b + a * x * Real.log x

theorem problem1 (a b : ℝ) (h : a ≠ 0) (he : f a b Real.exp = 2) : b = 2 := sorry

theorem problem2 (a : ℝ) (h : a ≠ 0) :
  let f' (x : ℝ) := a * Real.log x in
  (a > 0 →
    (∀ x, 1 < x → 0 < f' x) ∧
    (∀ x, 0 < x ∧ x < 1 → f' x < 0)) ∧
  (a < 0 →
    (∀ x, 0 < x ∧ x < 1 → 0 < f' x) ∧
    (∀ x, 1 < x → f' x < 0)) := sorry

end problem1_problem2_l368_368716


namespace delta_value_l368_368331

theorem delta_value (Δ : ℝ) (h : 4 * 3 = Δ - 6) : Δ = 18 :=
sorry

end delta_value_l368_368331


namespace exists_periodic_functions_l368_368693

noncomputable def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f x = f (x + p)

def f1 (a b : ℝ) (x : ℝ) : ℝ :=
  if ∃ (n m : ℤ), x = n * a + m * b then 2^id.snd (Exists.choose (Exists.choose_spec _ : _)) else 0

def f2 (a b : ℝ) (x : ℝ) : ℝ :=
  if ∃ (n m : ℤ), x = n * a + m * b then 2^(-id.fst (Exists.choose (Exists.choose_spec _ : _))) else 0

theorem exists_periodic_functions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (f1 f2 : ℝ → ℝ), periodic_function f1 a ∧ periodic_function f2 b ∧ periodic_function (λ x, f1 x * f2 x) (a + b) :=
begin
  let f1 := f1 a b,
  let f2 := f2 a b,
  use [f1, f2],
  split,
  { -- f1 periodic with period a
    sorry },
  split,
  { -- f2 periodic with period b
    sorry },
  { -- f1 * f2 periodic with period a + b
    sorry }
end

end exists_periodic_functions_l368_368693


namespace pascal_triangle_row_sum_l368_368274

theorem pascal_triangle_row_sum : (∑ n in Finset.range 30, n + 1) = 465 := by
  sorry

end pascal_triangle_row_sum_l368_368274


namespace max_squares_from_twelve_points_l368_368046

/-- Twelve points are marked on a grid paper. Prove that the maximum number of squares 
that can be formed by connecting four of these points is 11. -/
theorem max_squares_from_twelve_points : ∀ (points : list (ℝ × ℝ)), points.length = 12 → ∃ (squares : set (set (ℝ × ℝ))), squares.card = 11 ∧ ∀ square ∈ squares, ∃ (p₁ p₂ p₃ p₄ : (ℝ × ℝ)), p₁ ∈ points ∧ p₂ ∈ points ∧ p₃ ∈ points ∧ p₄ ∈ points ∧ set.to_finset {p₁, p₂, p₃, p₄}.card = 4 ∧ is_square {p₁, p₂, p₃, p₄} :=
by
  sorry

end max_squares_from_twelve_points_l368_368046


namespace lambda_three_sufficient_not_necessary_l368_368214

-- Define the vectors and the condition of parallelism
def vector_a (λ : ℝ) : ℝ × ℝ := (3, λ)
def vector_b (λ : ℝ) : ℝ × ℝ := (λ - 1, 2)
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)

-- Prove that λ = 3 is a sufficient but not necessary condition for vector_a λ to be parallel to vector_b λ
theorem lambda_three_sufficient_not_necessary (λ : ℝ) :
  parallel (vector_a 3) (vector_b 3) ∧ ∃ λ' : ℝ, λ' ≠ 3 ∧ parallel (vector_a λ') (vector_b λ') :=
by
  sorry

end lambda_three_sufficient_not_necessary_l368_368214


namespace factorization_correct_l368_368187

-- Define noncomputable to deal with the natural arithmetic operations
noncomputable def a : ℕ := 66
noncomputable def b : ℕ := 231

-- Define the given expressions
noncomputable def lhs (x : ℕ) : ℤ := ((a : ℤ) * x^6) - ((b : ℤ) * x^12)
noncomputable def rhs (x : ℕ) : ℤ := (33 : ℤ) * x^6 * (2 - 7 * x^6)

-- The theorem to prove the equality
theorem factorization_correct (x : ℕ) : lhs x = rhs x :=
by sorry

end factorization_correct_l368_368187


namespace inclination_angle_and_y_intercept_l368_368008

theorem inclination_angle_and_y_intercept :
  let line_eq : ℝ → ℝ → Prop := λ x y, -x + sqrt 3 * y - 6 = 0 in
  let tan_inv := real.arctan (sqrt 3 / 3) in
  let alpha : ℝ := 30 * real.pi / 180 in
  let y_intercept : ℝ := 2 * sqrt 3 in
  (∀ x y, line_eq x y → x = 0 → y = y_intercept) ∧ tan_inv = alpha :=
begin
  sorry
end

end inclination_angle_and_y_intercept_l368_368008


namespace seq100_is_981_l368_368496

-- In Lean, we need to exist an increasing sequence
-- formed by positive integers that are either powers of 3 or sums of different powers of 3
def isSumOfDistinctPowersOfThree (n : ℕ) : Prop :=
  ∃ (s : Multiset ℕ), (∀ x ∈ s, ∃ k, x = 3^k) ∧ s.sum = n

-- Define the sequence predicate
def isInSequence (a : ℕ) : Prop :=
  isSumOfDistinctPowersOfThree a

-- Define the 100th term in the sequence
def seq100 := ((list.range 1000).filter isInSequence).nth 99

theorem seq100_is_981 : seq100 = some 981 := by
  sorry

end seq100_is_981_l368_368496


namespace pascal_triangle_count_30_rows_l368_368283

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l368_368283


namespace pascal_triangle_elements_count_l368_368266

theorem pascal_triangle_elements_count :
  ∑ n in finset.range 30, (n + 1) = 465 :=
by 
  sorry

end pascal_triangle_elements_count_l368_368266


namespace difference_of_squares_l368_368553

theorem difference_of_squares :
  535^2 - 465^2 = 70000 :=
by
  sorry

end difference_of_squares_l368_368553


namespace largest_perpendicular_section_area_l368_368234

theorem largest_perpendicular_section_area (a : ℝ) :
  ∃ S, is_section_perpendicular_to_AC1 S ∧ area S = (√3 / 2) * a^2 := sorry

end largest_perpendicular_section_area_l368_368234


namespace function_shift_l368_368495

theorem function_shift {f : ℝ → ℝ} (h : ∀ x, f(x + π / 12) = cos (π / 2 - 2 * x)) :
  f = λ x, sin (2 * x - π / 6) :=
by
  sorry

end function_shift_l368_368495


namespace rectangle_ratio_l368_368780

theorem rectangle_ratio (side : ℝ) (h_side : side = 3)
                        (E F : Point) (h_E : E.isMidpoint(P1, P2))
                        (h_F : F.isMidpoint(P3, P4))
                        (AG CE XY YZ : ℝ)
                        (perpendicular_AG_CE : AG ⊥ CE)
                        (h_area : AG * CE = 9)
                        : XY / YZ = 4 / 5 := sorry

end rectangle_ratio_l368_368780


namespace lambda_3_sufficient_not_necessary_l368_368216

-- Define the vectors
def a (λ : ℝ) : ℝ × ℝ := (3, λ)
def b (λ : ℝ) : ℝ × ℝ := (λ - 1, 2)

-- Define the parallel condition
def are_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

-- Define the specific lambda value
def λ₃ := 3

-- State the problem about sufficient but not necessary condition
theorem lambda_3_sufficient_not_necessary (λ : ℝ) :
  (are_parallel (a λ) (b λ)) -> λ₃ = 3 ∨ λ₃ = -2 :=
sorry

end lambda_3_sufficient_not_necessary_l368_368216


namespace area_DBC_l368_368786

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 10 }
def B : Point := { x := 0, y := 0 }
def C : Point := { x := 12, y := 0 }
def D : Point := { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }
def E : Point := { x := (B.x + C.x) / 2, y := (B.y + C.y) / 2 }

def area_triangle (p1 p2 p3 : Point) : ℝ :=
  (1 / 2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem area_DBC : area_triangle D B C = 30 := sorry

end area_DBC_l368_368786


namespace volleyballs_remaining_l368_368034

def initial_volleyballs := 9
def lent_volleyballs := 5

theorem volleyballs_remaining : initial_volleyballs - lent_volleyballs = 4 := 
by
  sorry

end volleyballs_remaining_l368_368034


namespace perpendicular_divides_AH_in_ratio_l368_368438

variables (O H A B C K L : Point)
variables [Circumcenter O A B C] [Orthocenter H A B C]
variables (A H : Segment A H) (OK : Segment O K) (AC : Segment A C)

-- Statement of the problem in Lean 4
theorem perpendicular_divides_AH_in_ratio (h1 : Parallel (Line O H) (Line B C))
  (h2 : Parallelogram A B H K)
  (h3 : Intersect OK AC L)
  : divides_in_ratio (Perpendicular_from L A H) (Segment A H) 1 1 :=
sorry  -- proof is not required as per the instructions

end perpendicular_divides_AH_in_ratio_l368_368438


namespace complex_pure_imaginary_l368_368345

theorem complex_pure_imaginary (a : ℝ) : (1 + complex.I * a) ^ 2 = complex.I * (2 * a) → a = 1 ∨ a = -1 := 
by
  sorry

end complex_pure_imaginary_l368_368345


namespace orange_distribution_l368_368630

theorem orange_distribution :
  ∃ (a b c : ℕ), a + b + c = 30 ∧ 3 ≤ a ∧ 3 ≤ b ∧ 3 ≤ c ∧ (
    ∃ (a' b' c' : ℕ), a = a' + 3 ∧ b = b' + 3 ∧ c = c' + 3 ∧
    ((a' + b' + c' = 21) ∧ (nat.choose 23 2 = 253))
  ) :=
by
  exist a := 3 + a'
  exist b := 3 + b'
  exist c := 3 + c'
  have habc : a' + b' + c' = 21
  have hchoose : nat.choose 23 2 = 253
  sorry

end orange_distribution_l368_368630


namespace quadrilateral_OBEC_area_l368_368963

variables {xA yA xB yB xC yC xD yD xE yE : ℝ}

noncomputable def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

-- Given Points:
def A : (ℝ × ℝ) := (7, 0)
def C : (ℝ × ℝ) := (9, 0)
def E : (ℝ × ℝ) := (6, 3)
def D : (ℝ × ℝ) := (0, 3)

def lineA_B (x y : ℝ) : Prop := slope 7 0 x y = slope 7 0 6 3 -- Line AB passing through E
def B : (ℝ × ℝ) := (0, 21)
def lineC_D (x y : ℝ) : Prop := slope 9 0 x y = -1 -- Line CD
def O : (ℝ × ℝ) := (0, 0)
def HBE : (ℝ × ℝ) := (7, 21) -- base - hypotenuse triangle for OBE
def HDE : (ℝ × ℝ) := (3, 3)

noncomputable def area (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * | A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) |

theorem quadrilateral_OBEC_area :
  ( area O B E ) - ( area O E C + area O D E ) = 51 :=
by
  sorry

end quadrilateral_OBEC_area_l368_368963


namespace carrie_tomatoes_l368_368653

theorem carrie_tomatoes (T : ℕ) (total_revenue : ℕ) (carrots : ℕ)
  (tomato_price carrot_price : ℕ)
  (harvest_revenue : total_revenue = tomatoes T * tomato_price + carrots * carrot_price)
  (carrots_amount : carrots = 350)
  (tomato_price_val : tomato_price = 1)
  (carrot_price_val : carrot_price = 150)
  (total_revenue_val : total_revenue = 725) :
  T = 200 := 
sorry

end carrie_tomatoes_l368_368653


namespace tamara_coin_difference_l368_368473

-- Define the conditions from the problem
def tamaraHasCoins (n m : ℕ) : Prop :=
  n + m = 3030 ∧ 1 ≤ n ∧ 1 ≤ m

-- Define the value function for the coins
def totalValue (n m : ℕ) : ℕ := n + 10 * m

-- The main theorem to prove the difference in the total value
theorem tamara_coin_difference (n m : ℕ) (h : tamaraHasCoins n m) : 
  let maxVal := totalValue 1 (3030 - 1);
      minVal := totalValue 3029 1
  in maxVal - minVal = 27252 :=
by
  -- We skip the proof as we're required to provide the statement only
  sorry

end tamara_coin_difference_l368_368473


namespace total_cost_jackie_february_cost_l368_368095

theorem total_cost (text_messages : Nat) (talk_hours : Real) (data_used : Real) : Real :=
  let base_cost := 25
  let cost_per_text := 0.10
  let text_cost := text_messages * cost_per_text
  let included_hours := 25
  let cost_per_extra_minute := 0.15
  let extra_minutes := if talk_hours > included_hours then (talk_hours - included_hours) * 60 else 0
  let extra_minutes_cost := extra_minutes * cost_per_extra_minute
  let included_data := 3
  let cost_per_extra_gb := 2
  let extra_data := if data_used > included_data then data_used - included_data else 0
  let extra_data_cost := extra_data * cost_per_extra_gb
  base_cost + text_cost + extra_minutes_cost + extra_data_cost

theorem jackie_february_cost : total_cost 200 26.5 4.5 = 61.5 := by
  sorry

end total_cost_jackie_february_cost_l368_368095


namespace spinner_probability_D_l368_368953

theorem spinner_probability_D :
  let pa := 1 / 5
  let pb := 1 / 10
  let pD := 1 / 2 * (1 - (pa + pb))
  pD = 7 / 20 :=
by 
  -- Definitions and setup
  let pa := (1 : ℚ) / 5
  let pb := (1 : ℚ) / 10
  have pa_plus_pb : (1 / 5) + (1 / 10) = (2 / 10) + (1 / 10) := by norm_num
  have pa_plus_pb_equals : (2 / 10) + (1 / 10) = 3 / 10 := by norm_num
  have pDE : 1 - (pa + pb) = 1 - (3 / 10) := by rw [pa_plus_pb, pa_plus_pb_equals]
  have pD := 1 / 2 * (1 - (pa + pb))
  -- Simplify and solve for pD
  have two_x : 2 * pD = 7 / 10 := by
    simp [pD]
    sorry

  -- Divide both sides by 2
  have pD_solved : pD = 7 / 20 := by
    field_simp [two_x]
    norm_num
    sorry

  exact pD_solved

end spinner_probability_D_l368_368953


namespace vector_statement_c_l368_368918

-- Definitions
variable {A B C D : Type}
variable [normedAddCommGroup A]
variable [normedAddCommGroup B]
variable [normedAddCommGroup C]
variable [normedAddCommGroup D]
variable [normedSpace ℝ A]
variable [normedSpace ℝ B]
variable [normedSpace ℝ C]
variable [normedSpace ℝ D]

def is_equilateral (A B C : Type) [normedSpace ℝ A] [normedSpace ℝ B] [normedSpace ℝ C] :=
  ∃ (s : ℝ), s = 2 ∧ dist A B = s ∧ dist B C = s ∧ dist C A = s

def midpoint (B C D : Type) [hasVadd B C] [hasVadd C B] :=
  D = (B +ᵥ C) / 2

-- Proof statement
theorem vector_statement_c (A B C D : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] 
  (eq_triangle : is_equilateral A B C) (mid_d : midpoint B C D) :
  (∃ (x1 x2 : ℝ), x1 = 1 / 2 ∧ x2 = 1 / 2 ∧ (D = x1 • A + x2 • A)) :=
by
  sorry

end vector_statement_c_l368_368918


namespace intersection_point_exists_l368_368770

theorem intersection_point_exists
  (m n a b : ℝ)
  (h1 : m * a + 2 * m * b = 5)
  (h2 : n * a - 2 * n * b = 7)
  : (∃ x y : ℝ, 
    (y = (5 / (2 * m)) - (1 / 2) * x) ∧ 
    (y = (1 / 2) * x - (7 / (2 * n))) ∧
    (x = a) ∧ (y = b)) :=
sorry

end intersection_point_exists_l368_368770


namespace pascal_triangle_count_30_rows_l368_368280

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l368_368280


namespace area_difference_correct_l368_368751

def radius_large := 30
def circumference_small := 30

noncomputable def area_difference : ℝ :=
  let r_large := radius_large
  let C_small := circumference_small
  let A_large := π * r_large^2
  let r_small := C_small / (2 * π)
  let A_small := π * r_small^2
  A_large - A_small

theorem area_difference_correct :
  area_difference = 225 * (4 * π^2 - 1) / π := 
  by 
    sorry

end area_difference_correct_l368_368751


namespace sum_of_two_rationals_negative_l368_368771

theorem sum_of_two_rationals_negative (a b : ℚ) (h : a + b < 0) : a < 0 ∨ b < 0 := sorry

end sum_of_two_rationals_negative_l368_368771


namespace number_of_good_sets_l368_368959

   -- Definition of the conditions
   def card_deck :=
     { color : Type → Type | ∃ (red yellow blue : Type) 
       (cards : Π {c : Type}, c ≃ Fin 11), true } ∧
     { jokers : Fin 2 | true }

   def card_value (k : ℕ) : ℕ :=
     2^k

   def is_good_set (cards : Finset ℕ) : Prop :=
     cards.sum card_value = 2004

   -- Proof problem statement in Lean
   theorem number_of_good_sets :
     ∃ (count : ℕ), count = 1006009 :=
   sorry
   
end number_of_good_sets_l368_368959


namespace find_mb_l368_368885

-- Definitions based on conditions
def equation_of_line (m b : ℝ) : (ℝ → ℝ) := λ x, m * x + b
def m := 3
def b := -1

-- Statement of the problem
theorem find_mb (m b : ℝ) (h1 : m = 3) (h2 : b = -1) : m * b = -3 :=
by sorry

end find_mb_l368_368885


namespace smallest_positive_integer_remainder_l368_368060

theorem smallest_positive_integer_remainder :
  ∃ (a : ℕ), a % 6 = 3 ∧ a % 8 = 5 ∧ ∀ b : ℕ, (b % 6 = 3 ∧ b % 8 = 5) → a ≤ b :=
begin
  use 21,
  split,
  { norm_num, },
  split,
  { norm_num, },
  intro b,
  intro hb,
  rcases hb with ⟨hb1, hb2⟩,
  sorry
end

end smallest_positive_integer_remainder_l368_368060


namespace rounds_until_game_ends_l368_368889

-- Definitions of initial conditions
def initial_tokens_A := 20
def initial_tokens_B := 18
def initial_tokens_C := 16

-- Function to define the token dynamics per round
def tokens_after_round (tokens_A tokens_B tokens_C : ℕ) : ℕ × ℕ × ℕ :=
  let (max_player, max_tokens) :=
    if tokens_A > tokens_B ∧ tokens_A > tokens_C then (1, tokens_A)
    else if tokens_B > tokens_A ∧ tokens_B > tokens_C then (2, tokens_B)
    else if tokens_C > tokens_A ∧ tokens_C > tokens_B then (3, tokens_C)
    else (0, 0)
  -- After determining the player with the max tokens, we distribute and discard tokens accordingly
  match max_player with
  | 1 => (tokens_A - 6, tokens_B + 2, tokens_C + 2)
  | 2 => (tokens_A + 2, tokens_B - 6, tokens_C + 2)
  | 3 => (tokens_A + 2, tokens_B + 2, tokens_C - 6)
  | _ => (tokens_A, tokens_B, tokens_C)

-- Main proof problem statement
theorem rounds_until_game_ends : ∃ n : ℕ,
  let rounds : list (ℕ × ℕ × ℕ) :=
    list.map (λ i, let '(a, b, c) := tokens_after_round (a, b, c) in (a, b, c))
      (list.range n)
  in (rounds.length = 17) ∧
     (rounds.last.fst = 0 ∨ rounds.last.snd = 0 ∨ rounds.last.snd.snd = 0)
  :=
  sorry

end rounds_until_game_ends_l368_368889


namespace melanie_missed_games_l368_368529

-- Define the total number of games and the number of games attended by Melanie
def total_games : ℕ := 7
def games_attended : ℕ := 3

-- Define the number of games missed as total games minus games attended
def games_missed : ℕ := total_games - games_attended

-- Theorem stating the number of games missed by Melanie
theorem melanie_missed_games : games_missed = 4 := by
  -- The proof is omitted
  sorry

end melanie_missed_games_l368_368529


namespace problem_statement_l368_368251

noncomputable def f (x : ℝ) := 2 * x + 3
noncomputable def g (x : ℝ) := 3 * x - 2

theorem problem_statement : (f (g (f 3)) / g (f (g 3))) = 53 / 49 :=
by
  -- The proof is not provided as requested.
  sorry

end problem_statement_l368_368251


namespace solve_cubic_eq_with_geo_prog_coeff_l368_368458

variables {a q x : ℝ}

theorem solve_cubic_eq_with_geo_prog_coeff (h_a_nonzero : a ≠ 0) 
    (h_b : b = a * q) (h_c : c = a * q^2) (h_d : d = a * q^3) :
    (a * x^3 + b * x^2 + c * x + d = 0) → (x = -q) :=
by
  intros h_cubic_eq
  have h_b' : b = a * q := h_b
  have h_c' : c = a * q^2 := h_c
  have h_d' : d = a * q^3 := h_d
  sorry

end solve_cubic_eq_with_geo_prog_coeff_l368_368458


namespace problem_solution_l368_368683

theorem problem_solution (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ 3) :
    (8.17 * real.sqrt (3 * x - x ^ 2) < 4 - x) :=
sorry

end problem_solution_l368_368683


namespace proof_problem_l368_368205

noncomputable def f (x : ℝ) : ℝ := sorry -- Assume the existence of f(x)

theorem proof_problem (f_odd : ∀ x : ℝ, f (-x) = -f (x))
  (f_deriv : ∀ x : ℝ, has_deriv_at f (f' x) x)
  (condition : ∀ x : ℝ, x ≠ 0 → f' (x) + (f x / x) > 0)
  (a : ℝ := f 1)
  (b : ℝ := (log 3 (1 / 9)) * f (log 3 (1 / 9)))
  (c : ℝ := (log base := Real.exp) (1/2) ) * f (log base := Real.exp (1/2)))
  : c < a ∧ a < b :=
begin
  sorry
end

end proof_problem_l368_368205


namespace cookies_with_five_cups_l368_368811

-- Define the initial condition: Lee can make 24 cookies with 3 cups of flour
def cookies_per_cup := 24 / 3

-- Theorem stating Lee can make 40 cookies with 5 cups of flour
theorem cookies_with_five_cups : 5 * cookies_per_cup = 40 :=
by
  sorry

end cookies_with_five_cups_l368_368811


namespace more_customers_after_lunch_rush_l368_368627

-- Definitions for conditions
def initial_customers : ℝ := 29.0
def added_customers : ℝ := 20.0
def total_customers : ℝ := 83.0

-- The number of additional customers that came in after the lunch rush
def additional_customers (initial additional total : ℝ) : ℝ :=
  total - (initial + additional)

-- Statement to prove
theorem more_customers_after_lunch_rush :
  additional_customers initial_customers added_customers total_customers = 34.0 :=
by
  sorry

end more_customers_after_lunch_rush_l368_368627


namespace math_problem_l368_368488

theorem math_problem (c d : ℝ) (hc : c^2 - 6 * c + 15 = 27) (hd : d^2 - 6 * d + 15 = 27) (h_cd : c ≥ d) : 
  3 * c + 2 * d = 15 + Real.sqrt 21 :=
by
  sorry

end math_problem_l368_368488


namespace alternating_sum_of_binomial_coefficients_l368_368754

theorem alternating_sum_of_binomial_coefficients :
  (C(23, 3 * n + 1) = C(23, n + 6) ∧ n ∈ (ℕ \ {0}) ∧ (3 - x)^n = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 ∧ n = 4) →
  (a_0 - a_1 + a_2 - a_3 + a_4 = 256) := 
by
  sorry

end alternating_sum_of_binomial_coefficients_l368_368754


namespace unique_digit_count_in_B_or_C_l368_368563

theorem unique_digit_count_in_B_or_C (n : ℕ) (h : n > 1) :
  ∃! k : ℕ, let b_k := (⌊k * Real.log 2 10⌋ + 1) in
            let c_k := (⌊k * Real.log 5 10⌋ + 1) in
            b_k = n ∨ c_k = n :=
sorry

end unique_digit_count_in_B_or_C_l368_368563


namespace max_container_volume_l368_368053

theorem max_container_volume :
  ∀ (x : ℝ), 0 < x ∧ 2x + 2 * (x + 0.5) + 4 * (3.45 - x) = 14.8 →
  (x * (x + 0.5) * (3.45 - x) ≤ 3.675) :=
by
  sorry

end max_container_volume_l368_368053


namespace max_squares_from_twelve_points_l368_368045

/-- Twelve points are marked on a grid paper. Prove that the maximum number of squares 
that can be formed by connecting four of these points is 11. -/
theorem max_squares_from_twelve_points : ∀ (points : list (ℝ × ℝ)), points.length = 12 → ∃ (squares : set (set (ℝ × ℝ))), squares.card = 11 ∧ ∀ square ∈ squares, ∃ (p₁ p₂ p₃ p₄ : (ℝ × ℝ)), p₁ ∈ points ∧ p₂ ∈ points ∧ p₃ ∈ points ∧ p₄ ∈ points ∧ set.to_finset {p₁, p₂, p₃, p₄}.card = 4 ∧ is_square {p₁, p₂, p₃, p₄} :=
by
  sorry

end max_squares_from_twelve_points_l368_368045


namespace ratio_of_numbers_l368_368767

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : 2 * ((a + b) / 2) = Real.sqrt (10 * a * b)) : abs (a / b - 8) < 1 :=
by
  sorry

end ratio_of_numbers_l368_368767


namespace log_product_l368_368650

theorem log_product : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 2 := by
  sorry

end log_product_l368_368650


namespace slower_pipe_fills_tank_in_200_minutes_l368_368568

noncomputable def slower_pipe_filling_time (F S : ℝ) (h1 : F = 4 * S) (h2 : F + S = 1 / 40) : ℝ :=
  1 / S

theorem slower_pipe_fills_tank_in_200_minutes (F S : ℝ) (h1 : F = 4 * S) (h2 : F + S = 1 / 40) :
  slower_pipe_filling_time F S h1 h2 = 200 :=
sorry

end slower_pipe_fills_tank_in_200_minutes_l368_368568


namespace man_l368_368120

theorem man's_speed_kmph (length_train : ℝ) (time_seconds : ℝ) (speed_train_kmph : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * (5/18)
  let rel_speed_mps := length_train / time_seconds
  let man_speed_mps := rel_speed_mps - speed_train_mps
  man_speed_mps * (18/5)

example : man's_speed_kmph 120 6 65.99424046076315 = 6.00735873483709 := by
  sorry

end man_l368_368120


namespace new_parallelepiped_volume_l368_368258

def vec3 := ℝ × ℝ × ℝ

def cross_product (a b : vec3) : vec3 :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

def dot_product (a b : vec3) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def scale (a : ℝ) (v : vec3) : vec3 :=
  (a * v.1, a * v.2, a * v.3)

def add (a b : vec3) : vec3 :=
  (a.1 + b.1, a.2 + b.2, a.3 + b.3)

def parallelepiped_volume (a b c : vec3) : ℝ :=
  dot_product a (cross_product b c)

variable (a b c : vec3)

theorem new_parallelepiped_volume (h : |parallelepiped_volume a b c| = 8) :
  |parallelepiped_volume (add (scale 2 a) b) (add b (scale 5 c)) (add (scale 2 c) (scale 4 a))| = 32 := 
sorry

end new_parallelepiped_volume_l368_368258


namespace smallest_composite_sum_l368_368573

noncomputable def smallest_sum_of_consecutive_composite_numbers : ℕ :=
  102

theorem smallest_composite_sum :
  ∃ (n : ℕ), (Nat.all_composites [n, n+1, n+2, n+3]) ∧ (n + (n+1) + (n+2) + (n+3) = smallest_sum_of_consecutive_composite_numbers) :=
sorry

end smallest_composite_sum_l368_368573


namespace log_base_8_of_4_l368_368232

theorem log_base_8_of_4
  (h : real.logb 8 2 = 0.2525) : real.logb 8 4 = 0.5050 := 
by {
  sorry
}

end log_base_8_of_4_l368_368232


namespace range_of_m_l368_368246

def f (x : ℝ) : ℝ := 1 / (x^2 + 1) - 2^|x|

theorem range_of_m (m : ℝ) : f (2 * m - 1) > f m ↔ m > 1 ∨ m < 1 / 3 :=
by
  sorry

end range_of_m_l368_368246


namespace two_p_plus_q_l368_368566

variable {p q : ℚ}

theorem two_p_plus_q (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 :=
by
  sorry

end two_p_plus_q_l368_368566


namespace Psi_gcd_l368_368836

-- Define the prime number p
variable (p : ℕ) [Fact (Nat.Prime p)]

-- Define the finite field Fp and the polynomial ring Fp[x]
def Fp := ZMod p
def Fp_x := Polynomial Fp

-- Define the linear map Psi
def Psi (f : Fp_x) : Fp_x :=
  f.sum (λ n a, Polynomial.C a * Polynomial.X^p^n)

-- Define the gcd of two polynomials in Fp[x]
def gcd (F G : Fp_x) : Fp_x :=
  Polynomial.gcd F G

-- Statement of the theorem to be proved
theorem Psi_gcd (F G : Fp_x) (hF : F ≠ 0) (hG : G ≠ 0) :
  Psi (gcd F G) = gcd (Psi F) (Psi G) := sorry

end Psi_gcd_l368_368836


namespace shaded_region_area_l368_368372

/-- Four circles of radius 4 units intersect at the origin. -/
def radius : ℝ := 4

def area_quarter_circle : ℝ := (π * radius^2) / 4

def area_isosceles_right_triangle : ℝ := (radius * radius) / 2

def area_checkered_region : ℝ := area_quarter_circle - area_isosceles_right_triangle

def area_shaded_region : ℝ := 8 * area_checkered_region

theorem shaded_region_area : area_shaded_region = 32 * π - 64 := by
  sorry

end shaded_region_area_l368_368372


namespace find_an_l368_368374

noncomputable def S (n : ℕ) : ℕ :=
  if n = 0 then 0 else n^2 + n

theorem find_an (n : ℕ) (hn : n > 0) : ∃ a : ℕ, a = 2 * n :=
by
  use 2 * n
  sorry

end find_an_l368_368374


namespace probability_sum_less_than_12_l368_368031

-- Define the arithmetic sequence
def arithmetic_sequence : list ℕ := [2, 4, 6, 8, 10, 12]

-- Total number of pairs (combinations of 6 elements taken 2 at a time)
def total_pairs : ℕ := Nat.choose 6 2

-- Define the pairs whose sums are less than 12
def favorable_pairs : list (ℕ × ℕ) :=
[(2, 4), (2, 6), (2, 8), (4, 6)]

-- Calculate the probability
def desired_probability : ℚ := (favorable_pairs.length : ℚ) / (total_pairs : ℚ)

-- Statement of the proof problem
theorem probability_sum_less_than_12 :
  desired_probability = 4 / 15 :=
by
  sorry

end probability_sum_less_than_12_l368_368031


namespace distance_between_first_and_last_bushes_l368_368474

theorem distance_between_first_and_last_bushes 
  (bushes : Nat)
  (spaces_per_bush : ℕ) 
  (distance_first_to_fifth : ℕ) 
  (total_bushes : bushes = 10)
  (fifth_bush_distance : distance_first_to_fifth = 100)
  : ∃ (d : ℕ), d = 225 :=
by
  sorry

end distance_between_first_and_last_bushes_l368_368474


namespace region_area_l368_368195

noncomputable def area_of_region : ℝ :=
  let x_floor := Real.floor in
  let x_frac := λ x: ℝ, x - x_floor x in
  let y_floor := Real.floor in
  ∑ k in Finset.range 80, 0.0125 * (k * (k + 1) / 2)

theorem region_area : area_of_region = 1080.0125 :=
sorry

end region_area_l368_368195


namespace land_per_person_l368_368806

noncomputable def total_land_area : ℕ := 20000
noncomputable def num_people_sharing : ℕ := 5

theorem land_per_person (Jose_land : ℕ) (h : Jose_land = total_land_area / num_people_sharing) :
  Jose_land = 4000 :=
by
  sorry

end land_per_person_l368_368806


namespace distinct_collections_of_letters_l368_368858

open Finset

noncomputable def GEOGRAPHY : Multiset Char := ['G', 'E', 'O', 'G', 'R', 'A', 'P', 'H', 'Y']

def vowels : Finset Char := {'E', 'O', 'A'}
def consonants : Finset Char := {'G', 'R', 'P', 'H', 'Y'}

theorem distinct_collections_of_letters :
  (count_combinations GEOGRAPHY 2 vowels) * (count_combinations GEOGRAPHY 3 consonants) = 68 :=
by
  sorry

end distinct_collections_of_letters_l368_368858


namespace find_third_side_l368_368112

theorem find_third_side
  (cubes : ℕ) (cube_volume : ℚ) (side1 side2 : ℚ)
  (fits : cubes = 24) (vol_cube : cube_volume = 27)
  (dim1 : side1 = 8) (dim2 : side2 = 9) :
  (side1 * side2 * (cube_volume * cubes) / (side1 * side2)) = 9 := by
  sorry

end find_third_side_l368_368112


namespace pascals_triangle_total_numbers_l368_368300

theorem pascals_triangle_total_numbers (N : ℕ) (hN : N = 29) :
  (∑ n in Finset.range (N + 1), (n + 1)) = 465 :=
by
  rw hN
  calc (∑ n in Finset.range 30, (n + 1))
      = ∑ k in Finset.range 30, (k + 1) : rfl
  -- Here we are calculating the sum of the first 30 terms of the sequence (n + 1)
  ... = 465 : sorry

end pascals_triangle_total_numbers_l368_368300


namespace arc_length_of_sector_max_area_of_sector_area_of_segment_l368_368703

-- Problem 1: Arc length of the sector
theorem arc_length_of_sector (α : ℝ) (R : ℝ) (l : ℝ) :
  α = 60 * (π / 180) → R = 10 → l = 10 * (π / 3) :=
begin
  assume hα hR,
  sorry
end

-- Problem 2: Central angle for maximum area of the sector
theorem max_area_of_sector (R : ℝ) (α : ℝ) (S : ℝ) :
  (l + 2 * R = 20) → l = R * α → l = 10 → α = 2 :=
begin
  assume h1 h2 h3,
  sorry
end

-- Problem 3: Area of segment of the circle
theorem area_of_segment (α : ℝ) (R : ℝ) (S_segment : ℝ) :
  α = π / 3 → R = 2 → S_segment = (1/2) * (2 * π / 3) * 2 - (1/2) * 2^2 * (sin (π / 3)) :=
begin
  assume hα hR,
  sorry
end

end arc_length_of_sector_max_area_of_sector_area_of_segment_l368_368703


namespace range_of_a_l368_368349

theorem range_of_a (a : ℝ) (h : ¬ ∃ x0 : ℝ, a * x0^2 - a * x0 - 2 ≥ 0) 
: a ∈ Icc (-8 : ℝ) 0 := sorry

end range_of_a_l368_368349


namespace compare_logarithmic_values_l368_368406

def ln : ℝ → ℝ := Real.log

theorem compare_logarithmic_values :
  let a := ln 2 / 2
  let b := ln 3 / 3
  let c := ln 5 / 5
  b > a ∧ a > c :=
by 
  let a := ln 2 / 2
  let b := ln 3 / 3
  let c := ln 5 / 5
  sorry

end compare_logarithmic_values_l368_368406


namespace initial_tax_rate_eq_20_l368_368100

-- Define the initial tax rate as a variable
variables {T : ℝ}

-- Define the given conditions
def current_tax_rate : ℝ := 30
def initial_income : ℝ := 1_000_000
def current_income : ℝ := 1_500_000
def additional_taxes_paid : ℝ := 250_000

-- Define the relationship based on the problem statement
theorem initial_tax_rate_eq_20 :
  current_income * (current_tax_rate / 100) - initial_income * (T / 100) = additional_taxes_paid →
  T = 20 :=
begin
  sorry
end

end initial_tax_rate_eq_20_l368_368100


namespace world_book_day_l368_368130

theorem world_book_day
  (x y : ℕ)
  (h1 : x + y = 22)
  (h2 : x = 2 * y + 1) :
  x = 15 ∧ y = 7 :=
by {
  -- The proof is omitted as per the instructions
  sorry
}

end world_book_day_l368_368130


namespace conclusion_1_conclusion_2_conclusion_3_conclusion_4_conclusion_5_l368_368887

variables {R : Type*} [LinearOrder R] [Algebra.Ring R]

theorem conclusion_1 (x : R) (h : |x| = |(-3 : R)|) : x = 3 ∨ x = -3 :=
by {
  sorry
}

theorem conclusion_2 (x : R) (h : |-x| = |(-3 : R)|) : x = 3 :=
by {
  sorry
}

theorem conclusion_3 (x y : R) (h : |x| = |y|) : x = y :=
by {
  sorry
}

theorem conclusion_4 (x y : R) (h : x + y = 0) : |x| / |y| = 1 :=
by {
  sorry
}

theorem conclusion_5 (a b c : ℚ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a < 0) (h₅ : a + b < 0) (h₆ : a + b + c < 0) :
  (|a| / a + |b| / b + |c| / c - |a * b * c| / (a * b * c)) = 2 ∨ (|a| / a + |b| / b + |c| / c - |a * b * c| / (a * b * c)) = -2 :=
by {
  sorry
}

end conclusion_1_conclusion_2_conclusion_3_conclusion_4_conclusion_5_l368_368887


namespace exists_integer_K_l368_368665

theorem exists_integer_K (Z : ℕ) (K : ℕ) : 
  1000 < Z ∧ Z < 2000 ∧ Z = K^4 → 
  ∃ K, K = 6 := 
by
  sorry

end exists_integer_K_l368_368665


namespace extraordinary_numbers_count_l368_368851

-- Definition of an extraordinary number
def is_extraordinary (n : ℕ) : Prop :=
  ∃ p : ℕ, p.prime ∧ 2 * p = n

-- Interval constraint
def in_interval (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 75

-- Combine definitions
def is_extraordinary_in_interval (n : ℕ) : Prop :=
  is_extraordinary n ∧ in_interval n

-- The final theorem to prove
theorem extraordinary_numbers_count : 
  {n : ℕ | is_extraordinary_in_interval n}.toFinset.card = 11 := sorry

end extraordinary_numbers_count_l368_368851


namespace average_score_all_test_takers_l368_368478

def avg (scores : List ℕ) : ℕ := scores.sum / scores.length

theorem average_score_all_test_takers (s_avg u_avg n : ℕ) 
  (H1 : s_avg = 42) (H2 : u_avg = 38) (H3 : n = 20) : avg ([s_avg * n, u_avg * n]) / (2 * n) = 40 := 
by sorry

end average_score_all_test_takers_l368_368478


namespace pq_b1_b2_bn_exists_even_integers_l368_368392

theorem pq_b1_b2_bn_exists_even_integers
  (p q : ℤ) (h_coprime : Int.gcd p q = 1) (h_le_one : |(p:ℚ) / q| ≤ 1) :
  (p, q) ∈ {pq : ℤ × ℤ | 
               (Int.coprime pq.fst pq.snd) ∧ 
               (|pq.fst| < |pq.snd|) ∧ 
               (pq.fst ≠ 0 ∧ pq.snd ≠ 0) ∧ 
               (pq.fst.even ↔ ¬pq.snd.even)} := 
  sorry

end pq_b1_b2_bn_exists_even_integers_l368_368392


namespace wax_initial_amount_l368_368259

def needed : ℕ := 17
def total : ℕ := 574
def initial : ℕ := total - needed

theorem wax_initial_amount :
  initial = 557 :=
by
  sorry

end wax_initial_amount_l368_368259


namespace part_i_part_ii_l368_368730

-- Define the function f
def f (x : ℝ) (k : ℝ) : ℝ := x^2 + 3 * x - 3 - k * Real.exp x

-- Part (I)
theorem part_i (x k : ℝ) (h1 : x ≥ -5) (h2 : ∀ x ≥ -5, f x k ≤ 0) :
  k ≥ 7 * Real.exp 5 := sorry 

-- Define the function f for part II with k = -1
def f_k_minus_one (x : ℝ) : ℝ := x^2 + 3 * x - 3 + Real.exp x

-- Part (II)
theorem part_ii (x : ℝ) (h : ∀ x, f_k_minus_one x > -6) : f_k_minus_one x > -6 := sorry

end part_i_part_ii_l368_368730


namespace part_a_part_b_l368_368175

namespace problem

variables (n k : ℕ) (numbers : Finset ℕ)

-- Condition: Each subsequent number is chosen independently of the previous ones.
-- Condition: If a chosen number matches one of the already chosen numbers, move clockwise to the first unchosen number.
-- Condition: In the end, k different numbers are obtained.

-- Part (a): Appearance of each specific number is equally likely.
theorem part_a (h1 : ∀ (chosen : Finset ℕ), chosen.card = k → (∃! x ∈ numbers, x ∉ chosen)) :
  ∃ (p : ℚ), ∀ (x ∈ numbers), p = k / n :=
sorry

-- Part (b): Appearance of all selections is not equally likely.
theorem part_b (h1 : ∀ (chosen : Finset ℕ), chosen.card = k → (∃! x ∈ numbers, x ∉ chosen)) :
  ¬ ∃ (p : ℚ), ∀ (s : Finset ℕ), s.card = k → p = (card s.subevents) / (n ^ k) :=
sorry

end problem

end part_a_part_b_l368_368175


namespace smaller_cone_radius_correct_l368_368612

noncomputable def smaller_cone_radius (k : ℝ) (h : ℝ) (r : ℝ) : ℝ :=
  let sqrt_52 := Real.sqrt (6^2 + 4^2)
  let area_total := (Real.pi * r^2) + (Real.pi * r * sqrt_52)
  let volume_total := (1 / 3) * Real.pi * r^2 * h

  let x := real_root 3 (3 / 10) * 4 / 3

  let surface_area_c := Real.pi * x^2 + Real.pi * x * (sqrt_52 * x / 4)
  let volume_c := (1 / 3) * Real.pi * x^2 * (6 * x / 4)

  let surface_area_f := area_total - surface_area_c
  let volume_f := volume_total - volume_c

  let ratio_area := surface_area_c / surface_area_f
  let ratio_volume := volume_c / volume_f

  if (ratio_area = k) ∧ (ratio_volume = k) then x else 0 -- Returning 0 if the condition is not met

theorem smaller_cone_radius_correct (h : ℝ) (r : ℝ) (k : ℝ):
  k = (3 / 7) → h = 6 → r = 4 → smaller_cone_radius k h r = 4 * (real_root 3 (3 / 10)) / 3 :=
by
  sorry

end smaller_cone_radius_correct_l368_368612


namespace find_f_neg_one_l368_368724

variable (f : ℝ → ℝ)

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Condition 2: Definition of f(x) for x > 0
def fx_positive (x : ℝ) (h : 0 < x) : ℝ :=
  x^2 + 1 / x

-- The theorem we need to prove
theorem find_f_neg_one (h_odd : is_odd_function f) (h_pos : ∀ x : ℝ, 0 < x → f x = fx_positive x (by assumption)) :
  f (-1) = -2 :=
by
  sorry

end find_f_neg_one_l368_368724


namespace triangle_area_max_l368_368376

noncomputable def a : ℕ := 162
noncomputable def b : ℕ := 81
noncomputable def c : ℕ := 3

theorem triangle_area_max :
  let a := 162,
  let b := 81,
  let c := 3,
  a + b + c = 246
:= by
  sorry

end triangle_area_max_l368_368376


namespace number_of_persimmons_l368_368030

theorem number_of_persimmons (t p total : ℕ) (h1 : t = 19) (h2 : total = 37) (h3 : t + p = total) : p = 18 := 
by 
  rw [h1, h2] at h3
  linarith

end number_of_persimmons_l368_368030


namespace distinct_solutions_abs_eq_l368_368745

theorem distinct_solutions_abs_eq : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (|2 * x1 - 14| = |x1 + 4| ∧ |2 * x2 - 14| = |x2 + 4|) ∧ (∀ x, |2 * x - 14| = |x + 4| → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end distinct_solutions_abs_eq_l368_368745


namespace total_rate_is_6_percent_l368_368106

-- Definitions of the problem conditions
def P1 : ℝ := 8000
def R1 : ℝ := 0.05
def P2 : ℝ := 4000
def R2 : ℝ := 0.08

-- The total principal invested
def total_principal : ℝ := P1 + P2

-- The total interest earned
def total_interest : ℝ := (P1 * R1) + (P2 * R2)

-- The total rate he wants to earn
def total_rate : ℝ := (total_interest / total_principal) * 100

-- The proof statement
theorem total_rate_is_6_percent : total_rate = 6 :=
by
  sorry

end total_rate_is_6_percent_l368_368106


namespace tan_double_angle_l368_368028

theorem tan_double_angle (θ : ℝ) (h1 : θ ≠ 0) (h2 : tan(θ) = 2) : tan(2 * θ) = -4 / 3 := 
sorry

end tan_double_angle_l368_368028


namespace cleo_fraction_of_marbles_l368_368146

theorem cleo_fraction_of_marbles 
  (initial_marbles : ℕ := 30)
  (fraction_taken_day_2 : ℚ := 3/5)
  (marbles_taken_by_cleo_day_3 : ℕ := 15)
  (marbles_each_after_divide_day_2 : ℕ := 9) :
  let remaining_marbles := initial_marbles - (fraction_taken_day_2 * initial_marbles).to_nat
      cleo_took_day_3 := marbles_taken_by_cleo_day_3 - marbles_each_after_divide_day_2
  in (cleo_took_day_3 : ℚ) / (remaining_marbles : ℚ) = 1/2 :=
by
  sorry

end cleo_fraction_of_marbles_l368_368146


namespace smallest_positive_period_f_max_min_f_on_interval_l368_368245

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi / 6) + 1

theorem smallest_positive_period_f : 
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ Real.pi) :=
sorry

theorem max_min_f_on_interval :
  let a := Real.pi / 4
  let b := 2 * Real.pi / 3
  ∃ M m, (∀ x, a ≤ x ∧ x ≤ b → f x ≤ M ∧ f x ≥ m) ∧ (M = 2) ∧ (m = -1) :=
sorry

end smallest_positive_period_f_max_min_f_on_interval_l368_368245


namespace trigonometric_identity_solution_l368_368938

theorem trigonometric_identity_solution (t : ℝ) :
  (16 * sin (t / 2) - 25 * cos (t / 2) ≠ 0) →
  (t = 2 * arctan (4 / 5) + 2 * π * (t / (2 * π) - floor(t / (2 * π))) ∘ abs) :=
sorry

end trigonometric_identity_solution_l368_368938


namespace regular_pencil_cost_l368_368784

noncomputable def pencil_cost (x : ℝ) : Prop :=
  let cost_with_eraser := 0.8
  let cost_short := 0.4
  let number_with_eraser := 200
  let number_regular := 40
  let number_short := 35
  let total_revenue := 194
  ((number_with_eraser * cost_with_eraser) + (number_short * cost_short) + (number_regular * x) = total_revenue)

theorem regular_pencil_cost : ∃ x : ℝ, pencil_cost(x) ∧ x = 0.5 :=
by
  use 0.5
  unfold pencil_cost
  simp
  norm_num
  sorry

end regular_pencil_cost_l368_368784


namespace solve_system_l368_368465

noncomputable def log_base_sqrt_3 (z : ℝ) : ℝ := Real.log z / Real.log (Real.sqrt 3)

theorem solve_system :
  ∃ x y : ℝ, (3^x * 2^y = 972) ∧ (log_base_sqrt_3 (x - y) = 2) ∧ (x = 5 ∧ y = 2) :=
by
  sorry

end solve_system_l368_368465


namespace total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368306

-- Define the number of elements in the nth row of Pascal's Triangle
def num_elements_in_row (n : ℕ) : ℕ := n + 1

-- Define the sum of numbers from 0 to n, inclusive
def sum_of_first_n_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the total number of elements in the first 30 rows (0th to 29th)
def total_elements_in_first_30_rows : ℕ := sum_of_first_n_numbers 30

-- The main statement to prove
theorem total_numbers_in_first_30_rows_of_Pascals_Triangle :
  total_elements_in_first_30_rows = 465 :=
by
  simp [total_elements_in_first_30_rows, sum_of_first_n_numbers]
  sorry

end total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368306


namespace diff_of_squares_535_465_l368_368555

theorem diff_of_squares_535_465 : (535^2 - 465^2) = 70000 :=
sorry

end diff_of_squares_535_465_l368_368555


namespace sum_b_first_n_eq_l368_368840

-- Define the sequences {a_n} and {b_n}
def a_seq (n : ℕ) := if n = 0 then 0 else 2^(n - 1)

def b_seq : ℕ → ℕ
| 0 => 3
| (n + 1) => a_seq n + b_seq n

-- Define the sum of the sequence {a_n}
def S_a (n : ℕ) : ℕ := 2 * a_seq n - 1

-- Define the sum of the first n terms of the sequence {b_n}
def sum_b (n : ℕ) := ∑ i in Finset.range n, b_seq i

-- Theorem statement: Sum of the first n terms of the sequence {b_n} is 2^n + 2n - 1
theorem sum_b_first_n_eq (n : ℕ) : sum_b n = 2^n + 2 * n - 1 := by
  sorry

end sum_b_first_n_eq_l368_368840


namespace simplify_expression_l368_368412

theorem simplify_expression (a c b : ℝ) (h1 : a > c) (h2 : c ≥ 0) (h3 : b > 0) :
  (a * b^2 * (1 / (a + c)^2 + 1 / (a - c)^2) = a - b) → (2 * a * b = a^2 - c^2) :=
by
  sorry

end simplify_expression_l368_368412


namespace hexagon_area_sum_eq_297_l368_368223

theorem hexagon_area_sum_eq_297 (a : ℕ) (b : ℕ) (side_length : ℕ)
  (h1 : 12 = 6 * 2) -- The hexagon is composed of 12 line segments
  (h2 : side_length = 3) -- Each line segment has a length of 3
  (hex_area : ℝ := (3 * 3 * (√3)) * 3 / 2) -- Calculating the area: 6 * (9√3 / 4) = 27√3/2
  (p : ℕ := 54) -- part value used in expression of the area
  (q : ℕ := 243) : -- part value used in expression of the area
  a = 54 → b = 243 → a + b = 297 :=
by
  intros ha hb
  rw [ha, hb]
  rfl

end hexagon_area_sum_eq_297_l368_368223


namespace interest_rate_second_year_l368_368193

theorem interest_rate_second_year
  (initial_amount : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (amount_first_year : initial_amount * (1 + first_year_rate / 100) = 6240)
  (final_amount_given : final_amount = 6552) :
  ∃ (second_year_rate : ℝ), second_year_rate = 5 :=
by
  -- Given initial amount, rate for the first year and final amount, 
  -- we aim to prove that second year rate is 5%
  have amount_first := 6240,
  have interest_second := 312,
  have rate := interest_second / amount_first,
  use (rate * 100),
  sorry

end interest_rate_second_year_l368_368193


namespace time_to_cross_train_B_l368_368537

-- Definitions for the problem conditions
def length_train_A : ℝ := 175
def length_train_B : ℝ := 150
def speed_train_A_kmph : ℝ := 54
def speed_train_B_kmph : ℝ := 36
def kmph_to_mps_conversion_factor : ℝ := 5 / 18 -- Conversion factor from km/hr to m/s

-- Calculate the required values
def relative_speed_mps : ℝ := (speed_train_A_kmph + speed_train_B_kmph) * kmph_to_mps_conversion_factor
def total_distance : ℝ := length_train_A + length_train_B

-- Lean theorem statement to prove the time taken
theorem time_to_cross_train_B : total_distance / relative_speed_mps = 13 := by
  sorry

end time_to_cross_train_B_l368_368537


namespace A_race_time_l368_368782

-- Definitions of the problem conditions
def race_distance : ℝ := 130
def time_B : ℝ := 25
def distance_diff : ℝ := 26
def speed_B : ℝ := race_distance / time_B  -- Speed of B

-- We need to prove that the time A takes to finish the race (t) is 20 seconds
theorem A_race_time : 
  ∃ t : ℝ, (speed_B * t = race_distance - distance_diff) → t = 20 := 
sorry

end A_race_time_l368_368782


namespace taylor_vase_sale_l368_368426

theorem taylor_vase_sale :
  let selling_price := 1.50
  let profit_percentage := 0.25
  let loss_percentage := 0.25
  let cost_price_first_vase := selling_price / (1 + profit_percentage)
  let cost_price_second_vase := selling_price / (1 - loss_percentage)
  let total_cost := cost_price_first_vase + cost_price_second_vase
  let total_revenue := selling_price * 2
  in total_revenue - total_cost = -0.20 :=
by
  sorry

end taylor_vase_sale_l368_368426


namespace radius_of_original_bubble_l368_368614

theorem radius_of_original_bubble (r R : ℝ) (h : r = 4 * real.cbrt 2) :
  (2 / 3) * real.pi * r^3 = (4 / 3) * real.pi * R^3 → R = 2 * real.cbrt 2 :=
by
  sorry

end radius_of_original_bubble_l368_368614


namespace distance_between_parallel_lines_l368_368484

-- Definition of the first line l1
def line1 (x y : ℝ) (c1 : ℝ) : Prop := 3 * x + 4 * y + c1 = 0

-- Definition of the second line l2
def line2 (x y : ℝ) (c2 : ℝ) : Prop := 6 * x + 8 * y + c2 = 0

-- The problem statement in Lean:
theorem distance_between_parallel_lines (c1 c2 : ℝ) :
  ∃ d : ℝ, d = |2 * c1 - c2| / 10 :=
sorry

end distance_between_parallel_lines_l368_368484


namespace find_integer_n_l368_368923

theorem find_integer_n : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ -215 ≡ n [ZMOD 23] :=
by {
  use 15,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
}

end find_integer_n_l368_368923


namespace handshake_count_l368_368525

theorem handshake_count (m n : ℕ) (h1 : m = 5) (h2 : n = 3) : 
  let total_people := m * n in
  let handshakes_per_person := (total_people - 1 - (m - 1)) in
  (total_people * handshakes_per_person) / 2 = 75 :=
by 
  sorry

end handshake_count_l368_368525


namespace range_of_cos_pi_over_3_l368_368221

theorem range_of_cos_pi_over_3 (x : ℤ) : 
  set.range (λ x : ℤ, Real.cos (π / 3 * (x : ℝ))) = {-1, -1/2, 1/2, 1} :=
sorry

end range_of_cos_pi_over_3_l368_368221


namespace gcd_A_B_l368_368831

def A : ℤ := 1989^1990 - 1988^1990
def B : ℤ := 1989^1989 - 1988^1989

theorem gcd_A_B : Int.gcd A B = 1 := 
by
  -- Conditions
  have h1 : A = 1989^1990 - 1988^1990 := rfl
  have h2 : B = 1989^1989 - 1988^1989 := rfl
  -- Conclusion
  sorry

end gcd_A_B_l368_368831


namespace problem_proof_l368_368157

def delta (a b : ℕ) : ℕ := a^2 + b

theorem problem_proof :
  let x := 6
  let y := 8
  let z := 4
  let w := 2
  let u := 5^delta x y
  let v := 7^delta z w
  delta u v = 5^88 + 7^18 :=
by
  let x := 6
  let y := 8
  let z := 4
  let w := 2
  let u := 5^delta x y
  let v := 7^delta z w
  have h1: delta x y = 44 := by sorry
  have h2: delta z w = 18 := by sorry
  have hu: u = 5^44 := by sorry
  have hv: v = 7^18 := by sorry
  have hdelta: delta u v = 5^88 + 7^18 := by sorry
  exact hdelta

end problem_proof_l368_368157


namespace FC_value_l368_368695

variables (DC CB AB AD ED FC CA BD : ℝ)

-- Set the conditions as variables
variable (h_DC : DC = 10)
variable (h_CB : CB = 12)
variable (h_AB : AB = (1/3) * AD)
variable (h_ED : ED = (2/3) * AD)
variable (h_BD : BD = 22)
variable (BD_eq : BD = DC + CB)
variable (CA_eq : CA = CB + AB)

-- Define the relationship for the final result
def find_FC (DC CB AB AD ED FC CA BD : ℝ) := FC = (ED * CA) / AD

-- The main statement to be proven
theorem FC_value : 
  find_FC DC CB AB (33 : ℝ) (22 : ℝ) FC (23 : ℝ) (22 : ℝ) → 
  FC = (506/33) :=
by 
  intros h
  sorry

end FC_value_l368_368695


namespace sum_of_reciprocals_of_factors_of_30_l368_368550

def factors_30 := {1, 2, 3, 5, 6, 10, 15, 30}
def recip_sum (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ x, (1 : ℚ) / x)

theorem sum_of_reciprocals_of_factors_of_30 :
  recip_sum factors_30 = 12 / 5 :=
by {
  -- Proof goes here
  sorry
}

end sum_of_reciprocals_of_factors_of_30_l368_368550


namespace equal_likelihood_each_number_not_equal_likelihood_all_selections_l368_368172

def k : ℕ := sorry
def n : ℕ := sorry
def selection : list ℕ := sorry

-- Condition 1: Each subsequent number is chosen independently of the previous ones.
axiom independent_choice (m : ℕ) : m ∈ selection → m < n

-- Condition 2: If it matches one of the already chosen numbers, you should move clockwise to the first unchosen number.
axiom move_clockwise (m : ℕ) : m ∈ selection → ∃ l, l ≠ m ∧ l ∉ selection

-- Condition 3: In the end, k different numbers are obtained.
axiom k_distinct : list.distinct selection ∧ list.length selection = k

-- Define the events A and B
def A : Prop := ∀ m : ℕ, m ∈ selection → ∃ p, probability m = p
def B : Prop := ∀ s1 s2 : list ℕ, s1 ⊆ selection ∧ s1.length = k ∧ s2 ⊆ selection ∧ s2.length = k → s1 ≠ s2 → probability s1 ≠ probability s2

-- Prove A holds given the conditions
theorem equal_likelihood_each_number : A :=
sorry

-- Prove B does not hold given the conditions
theorem not_equal_likelihood_all_selections : ¬B :=
sorry

end equal_likelihood_each_number_not_equal_likelihood_all_selections_l368_368172


namespace is_monotonically_decreasing_on_interval_slope_of_tangent_line_at_pi_div4_l368_368249

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 3

theorem is_monotonically_decreasing_on_interval :
  ∀ x y ∈ set.Icc (π / 6) (π / 2), x < y → f y < f x := sorry

theorem slope_of_tangent_line_at_pi_div4 :
  ∀ x, x = π / 4 → deriv f x = -2 := sorry

end is_monotonically_decreasing_on_interval_slope_of_tangent_line_at_pi_div4_l368_368249


namespace sufficient_guards_l368_368125

def point := ℝ × ℝ

structure polygon (n : ℕ) :=
  (vertices : fin n → point)
  (non_intersecting_edges : Prop)  -- Interpret as constraints defining non-intersecting edges
  (interior_empty : Prop)   -- No pillars or walls inside

-- Given definition of monitoring for a guard and a point A:
def is_monitored (p : polygon n) (G A : point) :=
  ∀ edge ∈ edges p, ¬ segments_intersect (G, A) edge

theorem sufficient_guards (p : polygon) (n : ℕ) (h_n_geq_3 : n ≥ 3) (h_non_intersecting_edges : p.non_intersecting_edges) (h_interior_empty : p.interior_empty) :
  ∃ S : fin n → point, (∀ G ∈ S, is_guard G) ∧ (∀ A ∈ polygon_interior p, ∃ G ∈ S, is_monitored p G A) ∧ (|S| ≤ ⌊n / 3⌋) :=
by
  sorry

end sufficient_guards_l368_368125


namespace probability_of_three_cards_l368_368913

-- Conditions
def deck_size : ℕ := 52
def spades : ℕ := 13
def spades_face_cards : ℕ := 3
def face_cards : ℕ := 12
def diamonds : ℕ := 13

-- Probability of drawing specific cards
def prob_first_spade_non_face : ℚ := 10 / 52
def prob_second_face_given_first_spade_non_face : ℚ := 12 / 51
def prob_third_diamond_given_first_two : ℚ := 13 / 50

def prob_first_spade_face : ℚ := 3 / 52
def prob_second_face_given_first_spade_face : ℚ := 9 / 51

-- Final probability
def final_probability := 
  (prob_first_spade_non_face * prob_second_face_given_first_spade_non_face * prob_third_diamond_given_first_two) +
  (prob_first_spade_face * prob_second_face_given_first_spade_face * prob_third_diamond_given_first_two)

theorem probability_of_three_cards :
  final_probability = 1911 / 132600 := 
by
  sorry

end probability_of_three_cards_l368_368913


namespace log2_m_n_product_l368_368217

theorem log2_m_n_product (m n : ℝ) (hm : log 2 m = 3.5) (hn : log 2 n = 0.5) : m * n = 16 := by
  sorry

end log2_m_n_product_l368_368217


namespace appearance_equally_likely_selections_not_equally_likely_l368_368178

-- Define the conditions under which the numbers are chosen
def independent_choice (n k : ℕ) (selection : ℕ → Set ℕ) : Prop :=
∀ i j : ℕ, i ≠ j → selection i ∩ selection j = ∅

def move_clockwise (chosen : Set ℕ) (n : ℕ) : ℕ → ℕ := sorry  -- Placeholder for clockwise movement function

-- Condition: Each subsequent number is chosen independently of the previous ones
def conditions (n k : ℕ) : Prop :=
∃ selection : ℕ → Set ℕ, independent_choice n k selection

-- Problem a: Prove that the appearance of each specific number is equally likely
theorem appearance_equally_likely (n k : ℕ) (h : conditions n k) : 
  (∀ num : ℕ, 1 ≤ num ∧ num ≤ n → appearance_probability num k n) := sorry

-- Problem b: Prove that the appearance of all selections is not equally likely
theorem selections_not_equally_likely (n k : ℕ) (h : conditions n k) : 
  ¬(∀ selection : Set ℕ, appearance_probability selection k n) := sorry

end appearance_equally_likely_selections_not_equally_likely_l368_368178


namespace part_a_part_b_l368_368173

namespace problem

variables (n k : ℕ) (numbers : Finset ℕ)

-- Condition: Each subsequent number is chosen independently of the previous ones.
-- Condition: If a chosen number matches one of the already chosen numbers, move clockwise to the first unchosen number.
-- Condition: In the end, k different numbers are obtained.

-- Part (a): Appearance of each specific number is equally likely.
theorem part_a (h1 : ∀ (chosen : Finset ℕ), chosen.card = k → (∃! x ∈ numbers, x ∉ chosen)) :
  ∃ (p : ℚ), ∀ (x ∈ numbers), p = k / n :=
sorry

-- Part (b): Appearance of all selections is not equally likely.
theorem part_b (h1 : ∀ (chosen : Finset ℕ), chosen.card = k → (∃! x ∈ numbers, x ∉ chosen)) :
  ¬ ∃ (p : ℚ), ∀ (s : Finset ℕ), s.card = k → p = (card s.subevents) / (n ^ k) :=
sorry

end problem

end part_a_part_b_l368_368173


namespace number_of_points_on_P_shape_l368_368564

-- Definitions based on the conditions
def square_side_length : ℕ := 10
def points_per_cm : ℕ := 1

-- Theorem stating the problem
theorem number_of_points_on_P_shape : 
  let side_points := square_side_length + 1 in
  let total_points := 3 * side_points - 2 in
  total_points = 31 := by
  sorry

end number_of_points_on_P_shape_l368_368564


namespace students_chose_apples_l368_368792

theorem students_chose_apples (total students choosing_bananas : ℕ) (h1 : students_choosing_bananas = 168) 
  (h2 : 3 * total = 4 * students_choosing_bananas) : (total / 4) = 56 :=
  by
  sorry

end students_chose_apples_l368_368792


namespace total_rainfall_cm_l368_368984

theorem total_rainfall_cm :
  let monday := 0.12962962962962962
  let tuesday := 3.5185185185185186 * 0.1
  let wednesday := 0.09259259259259259
  let thursday := 0.10222222222222223 * 2.54
  let friday := 12.222222222222221 * 0.1
  let saturday := 0.2222222222222222
  let sunday := 0.17444444444444446 * 2.54
  monday + tuesday + wednesday + thursday + friday + saturday + sunday = 2.721212629851652 :=
by
  sorry

end total_rainfall_cm_l368_368984


namespace sum_abc_l368_368756

theorem sum_abc (a b c : ℝ) 
  (h : (a - 6)^2 + (b - 3)^2 + (c - 2)^2 = 0) : 
  a + b + c = 11 := 
by 
  sorry

end sum_abc_l368_368756


namespace goose_eggs_l368_368131

theorem goose_eggs (E : ℝ) :
  (E / 2 * 3 / 4 * 2 / 5 + (1 / 3 * (E / 2)) * 2 / 3 * 3 / 4 + (1 / 6 * (E / 2 + E / 6)) * 1 / 2 * 2 / 3 = 150) →
  E = 375 :=
by
  sorry

end goose_eggs_l368_368131


namespace f_at_2_l368_368218

-- Define the function f with the given condition
def f (x : ℝ) : ℝ := sorry

-- Add the condition in the form of an axiom
axiom f_condition (x : ℝ) : f (1 + real.sqrt x) = x + 1

-- Prove that f(2) = 2
theorem f_at_2 : f 2 = 2 :=
by
  -- insert steps here as required
  sorry

end f_at_2_l368_368218


namespace ratio_of_60_to_12_l368_368057

theorem ratio_of_60_to_12 : 60 / 12 = 5 := 
by 
  sorry

end ratio_of_60_to_12_l368_368057


namespace cost_per_meter_l368_368498

-- Definitions of the conditions
def length_of_plot : ℕ := 63
def breadth_of_plot : ℕ := length_of_plot - 26
def perimeter_of_plot := 2 * length_of_plot + 2 * breadth_of_plot
def total_cost : ℕ := 5300

-- Statement to prove
theorem cost_per_meter : (total_cost : ℚ) / perimeter_of_plot = 26.5 :=
by sorry

end cost_per_meter_l368_368498


namespace find_other_endpoint_l368_368894

def Point : Type := ℝ × ℝ

def midpoint (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem find_other_endpoint (mid : Point) (endpoint1 : Point) (endpoint2 : Point) :
  mid = (3, 1) →
  endpoint1 = (7, -3) →
  midpoint endpoint1 endpoint2 = mid →
  endpoint2 = (-1, 5) :=
  by
  intros hmid hendpt1 hmid_eq
  sorry

end find_other_endpoint_l368_368894


namespace min_area_triangle_contains_unit_square_l368_368055

theorem min_area_triangle_contains_unit_square : ∃ (T : set (ℝ × ℝ)), 
  (∀ (x y : ℝ), (0 ≤ x) ∧ (x ≤ 1) ∧ (0 ≤ y) ∧ (y ≤ 1) → (x, y) ∈ T) ∧ 
  (∃ (A B C : ℝ × ℝ), T = convex_hull ℝ ({A, B, C} : set (ℝ × ℝ))) ∧ 
  (∀ (A B C : ℝ × ℝ), T = convex_hull ℝ ({A, B, C} : set (ℝ × ℝ)) → 
     ∃ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 
     (area_of_triangle A B C = a + b + c) ∧ (a + b + c ≥ 2)) :=
sorry

end min_area_triangle_contains_unit_square_l368_368055


namespace calculation_result_l368_368926

theorem calculation_result : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := 
by
  sorry

end calculation_result_l368_368926


namespace evaluate_expression_l368_368414

def g (x : ℝ) : ℝ := Real.log (2^x + 1)

theorem evaluate_expression :
  g (-4) - g (-3) + g (-2) - g (-1) + g (1) - g (2) + g (3) - g (4) = -2 * Real.log 2 :=
sorry

end evaluate_expression_l368_368414


namespace segment_PQ_length_l368_368504

noncomputable def length_segment_PQ (a b : ℝ) : ℝ :=
  Real.sqrt (a ^ 2 + b ^ 2)

theorem segment_PQ_length (a b : ℝ) :
  let PQ := length_segment_PQ a b
  in PQ = Real.sqrt (a ^ 2 + b ^ 2) :=
by
  sorry

end segment_PQ_length_l368_368504


namespace pascals_triangle_total_numbers_l368_368299

theorem pascals_triangle_total_numbers (N : ℕ) (hN : N = 29) :
  (∑ n in Finset.range (N + 1), (n + 1)) = 465 :=
by
  rw hN
  calc (∑ n in Finset.range 30, (n + 1))
      = ∑ k in Finset.range 30, (k + 1) : rfl
  -- Here we are calculating the sum of the first 30 terms of the sequence (n + 1)
  ... = 465 : sorry

end pascals_triangle_total_numbers_l368_368299


namespace total_pencils_given_out_l368_368520

theorem total_pencils_given_out (n p : ℕ) (h1 : n = 10) (h2 : p = 5) : n * p = 50 :=
by
  sorry

end total_pencils_given_out_l368_368520


namespace probability_white_given_popped_l368_368583

/-
Given the following probabilities:
- A bag contains 1/2 white kernels, 1/3 yellow kernels, and 1/6 red kernels.
- Half of the white kernels will pop.
- Two-thirds of the yellow kernels will pop.
- One-third of the red kernels will pop.
Prove that the probability that the selected kernel was white if it pops is 9/19.
-/

theorem probability_white_given_popped :
  let P_white := 1/2,
      P_yellow := 1/3,
      P_red := 1/6,
      P_popped_given_white := 1/2,
      P_popped_given_yellow := 2/3,
      P_popped_given_red := 1/3 in
  ((P_white * P_popped_given_white) / 
   ((P_white * P_popped_given_white) + 
    (P_yellow * P_popped_given_yellow) + 
    (P_red * P_popped_given_red))) = 9/19 :=
by
  let P_white := 1/2
  let P_yellow := 1/3
  let P_red := 1/6
  let P_popped_given_white := 1/2
  let P_popped_given_yellow := 2/3
  let P_popped_given_red := 1/3
  have calc : ((P_white * P_popped_given_white) /
    ((P_white * P_popped_given_white) + (P_yellow * P_popped_given_yellow) + (P_red * P_popped_given_red))) = 9/19
  sorry

end probability_white_given_popped_l368_368583


namespace probability_f_inequality_l368_368243

noncomputable def f (a x : ℝ) : ℝ := log a (x / 8)

theorem probability_f_inequality :
  let S := {1/4, 1/3, 1/2, 3, 4, 5, 6, 7}
  let n := 8
  let valid_a := {a ∈ S | f a (3 * a + 1) > f a (2 * a) ∧ f a (2 * a) > 0}
  let m := valid_a.card
  (m / n) = 5 / 8 :=
by
  sorry

end probability_f_inequality_l368_368243


namespace equal_likelihood_each_number_not_equal_likelihood_all_selections_l368_368170

def k : ℕ := sorry
def n : ℕ := sorry
def selection : list ℕ := sorry

-- Condition 1: Each subsequent number is chosen independently of the previous ones.
axiom independent_choice (m : ℕ) : m ∈ selection → m < n

-- Condition 2: If it matches one of the already chosen numbers, you should move clockwise to the first unchosen number.
axiom move_clockwise (m : ℕ) : m ∈ selection → ∃ l, l ≠ m ∧ l ∉ selection

-- Condition 3: In the end, k different numbers are obtained.
axiom k_distinct : list.distinct selection ∧ list.length selection = k

-- Define the events A and B
def A : Prop := ∀ m : ℕ, m ∈ selection → ∃ p, probability m = p
def B : Prop := ∀ s1 s2 : list ℕ, s1 ⊆ selection ∧ s1.length = k ∧ s2 ⊆ selection ∧ s2.length = k → s1 ≠ s2 → probability s1 ≠ probability s2

-- Prove A holds given the conditions
theorem equal_likelihood_each_number : A :=
sorry

-- Prove B does not hold given the conditions
theorem not_equal_likelihood_all_selections : ¬B :=
sorry

end equal_likelihood_each_number_not_equal_likelihood_all_selections_l368_368170


namespace appearance_equally_likely_all_selections_not_equally_likely_l368_368182

variables {n k : ℕ} (numbers : finset ℕ)

-- Conditions
def chosen_independently (x : ℕ) : Prop := sorry
def move_clockwise_if_chosen (x : ℕ) (chosen : finset ℕ) : Prop := sorry
def end_with_k_different_numbers (final_set : finset ℕ) : Prop := final_set.card = k

-- Part (a)
theorem appearance_equally_likely (x : ℕ) (h_independent : chosen_independently x)
  (h_clockwise : ∀ y ∈ numbers, move_clockwise_if_chosen y numbers) :
  (∃ y ∈ numbers, true) → true :=
by { sorry } -- Conclusion: Yes

-- Part (b)
theorem all_selections_not_equally_likely (samples : list (finset ℕ))
  (h_independent : ∀ x ∈ samples, chosen_independently x)
  (h_clockwise : ∀ y ∈ samples, move_clockwise_if_chosen y samples) :
  ¬ (∀ x y, x ≠ y → samples x = samples y) :=
by { sorry } -- Conclusion: No

end appearance_equally_likely_all_selections_not_equally_likely_l368_368182


namespace enclosed_area_of_ellipse_l368_368162

def ellipse_equation (x y: ℝ) : Prop := 
  (x^2 / 4 + y^2 / 9 = |x| + |y|)

theorem enclosed_area_of_ellipse :
  (∃ (f : ℝ → ℝ → Prop), (∀ x y, f x y = ellipse_equation x y) ∧ area_enclosed_by_ellipse (f x y) = 6 * Real.pi) :=
sorry

end enclosed_area_of_ellipse_l368_368162


namespace radius_of_original_bubble_l368_368613

theorem radius_of_original_bubble (r R : ℝ) (h : r = 4 * real.cbrt 2) :
  (2 / 3) * real.pi * r^3 = (4 / 3) * real.pi * R^3 → R = 2 * real.cbrt 2 :=
by
  sorry

end radius_of_original_bubble_l368_368613


namespace angle_bisector_slope_l368_368685

theorem angle_bisector_slope :
  ∀ m1 m2 : ℝ, m1 = 2 → m2 = 4 → (∃ k : ℝ, k = (6 - Real.sqrt 21) / (-7) → k = (-6 + Real.sqrt 21) / 7) :=
by
  sorry

end angle_bisector_slope_l368_368685


namespace train_speed_l368_368624

theorem train_speed (time : ℤ) (length : ℤ) (time_sec_eq : time = 18) (length_m_eq : length = 260) :
  let time_hr := (time : ℚ) / 3600 in
  let length_km := (length : ℚ) / 1000 in
  (length_km / time_hr) = 52 :=
by
  let time_hr := (time : ℚ) / 3600
  let length_km := (length : ℚ) / 1000
  have : (52 : ℚ) = (260 / 1000) / (18 / 3600) := by norm_num
  exact this

end train_speed_l368_368624


namespace textbook_problem_l368_368622

theorem textbook_problem (x y : ℕ) 
  (h1 : 0.5 * x + 0.2 * y = 390)
  (h2 : 0.5 * x = 3 * (0.8 * y)) : 
  x = 720 ∧ y = 150 := 
by
  sorry

end textbook_problem_l368_368622


namespace total_bill_l368_368980

theorem total_bill 
  (adults : ℕ) 
  (children : ℕ) 
  (cost_per_meal : ℕ) 
  (total_people := adults + children)
  (total_cost := total_people * cost_per_meal) 
  (h_adults : adults = 2) 
  (h_children : children = 5) 
  (h_cost : cost_per_meal = 3) : 
  total_cost = 21 := 
by
  rw [h_adults, h_children, h_cost]
  simp
  sorry

end total_bill_l368_368980


namespace must_choose_exactly_1982_nums_l368_368842

theorem must_choose_exactly_1982_nums (A : Finset ℕ) (hA : A = Finset.range 6798)
    (C : Finset ℕ) (hC: C.card = 3399)
    (h_proper_subset: ∀ a b ∈ C, a ∣ b → a = b) :
    ∃ D : Finset ℕ, D.card = 1982 ∧ (∀ D' : Finset ℕ, D' ⊆ A → D'.card = 1982 → D = D') :=
sorry

end must_choose_exactly_1982_nums_l368_368842


namespace range_of_s_l368_368056

noncomputable def s (x : ℝ) := 1 / (2 + x)^3

theorem range_of_s :
  Set.range s = {y : ℝ | y < 0} ∪ {y : ℝ | y > 0} :=
by
  sorry

end range_of_s_l368_368056


namespace sharks_at_newport_l368_368661

theorem sharks_at_newport :
  ∃ (x : ℕ), (∃ (y : ℕ), y = 4 * x ∧ x + y = 110) ∧ x = 22 :=
by {
  sorry
}

end sharks_at_newport_l368_368661


namespace million_to_scientific_notation_l368_368645

theorem million_to_scientific_notation (population_henan : ℝ) (h : population_henan = 98.83 * 10^6) :
  population_henan = 9.883 * 10^7 :=
by sorry

end million_to_scientific_notation_l368_368645


namespace angle_B_30_degrees_length_of_b_l368_368364

-- Define the acute triangle property and angle B
variable (a b c A B C : ℝ)
axiom acute_triangle (a b c A B C : ℝ) : a = 2 * b * sin A ∧ a = 3 * sqrt 3 ∧ c = 5 ∧ B = 30

-- Prove that angle B is 30 degrees
theorem angle_B_30_degrees (a b c A B C : ℝ) (h : acute_triangle a b c A B C) : B = 30 := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  exact h4

-- Prove the length of side b is sqrt(7)
theorem length_of_b (a b c A B C : ℝ) (h : acute_triangle a b c A B C) : b = sqrt 7 := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  have by law_of_cosines : b^2 = (3 * sqrt 3)^2 + 5^2 - 2 * (3 * sqrt 3) * 5 * (cos (B * π / 180)) := by
    sorry -- proof details are skipped
  have b_sq : b^2 = 7 := by
    rw [h, cos_30_eq (B * π / 180)] at law_of_cosines -- apply specific cosine value
    sorry -- calculations skipped
  exact sqrt_eq b_sq

end angle_B_30_degrees_length_of_b_l368_368364


namespace find_p_q_l368_368817

def primes (n : ℕ) := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def seq (p q : ℕ) : ℕ → ℤ 
| 0     := 1
| 1     := p
| (n+2) := p * seq (n + 1) - q * seq n

theorem find_p_q (p q : ℕ) (k : ℕ) 
  (h1 : primes p) 
  (h2 : primes q) 
  (h3 : seq p q (3 * k) = -3) : 
  p = 2 ∧ q = 7 := 
sorry

end find_p_q_l368_368817


namespace estimate_pi_l368_368966

/-- A member of a mathematics interest group estimated π by throwing beans into a square region. -/
theorem estimate_pi :
  let side_length := 1 : ℝ
  let beans_in_square := 5120 : ℝ
  let beans_in_circle := 4009 : ℝ
  let circle_ratio := beans_in_circle / beans_in_square
  let estimated_pi := (circle_ratio * 4)
  (estimated_pi ≈ 3.13) :=
by
  let side_length := 1
  let beans_in_square := 5120
  let beans_in_circle := 4009
  let circle_ratio := beans_in_circle / beans_in_square
  let estimated_pi := (circle_ratio * 4)
  sorry

end estimate_pi_l368_368966


namespace find_n_l368_368768

noncomputable def parabola_focus : ℝ × ℝ :=
  (2, 0)

noncomputable def hyperbola_focus (n : ℝ) : ℝ × ℝ :=
  (Real.sqrt (3 + n), 0)

theorem find_n (n : ℝ) : hyperbola_focus n = parabola_focus → n = 1 :=
by
  sorry

end find_n_l368_368768


namespace temperature_max_time_l368_368860

theorem temperature_max_time (t : ℝ) (h : 0 ≤ t) : 
  (-t^2 + 10 * t + 60 = 85) → t = 15 := 
sorry

end temperature_max_time_l368_368860


namespace third_offense_percentage_increase_l368_368427

theorem third_offense_percentage_increase 
    (base_per_5000 : ℕ)
    (goods_stolen : ℕ)
    (additional_years : ℕ)
    (total_sentence : ℕ) :
    base_per_5000 = 1 →
    goods_stolen = 40000 →
    additional_years = 2 →
    total_sentence = 12 →
    100 * (total_sentence - additional_years - goods_stolen / 5000) / (goods_stolen / 5000) = 25 :=
by
  intros h_base h_goods h_additional h_total
  sorry

end third_offense_percentage_increase_l368_368427


namespace pete_and_son_age_l368_368865

theorem pete_and_son_age :
  ∃ x : ℕ, (35 + x = 3 * (9 + x)) ∧ x = 4 := 
by {
  use 4,
  split,
  {
    calc 35 + 4 = 39 : by rfl
         ... = 3 * 13 : by rfl
         ... = 3 * (9 + 4) : by rfl,
  },
  {
    rfl,
  }
}

end pete_and_son_age_l368_368865


namespace line_intersects_circle_l368_368900

-- Define the line equation
def line (k : ℝ) : ℝ × ℝ → Prop := λ p, k * p.1 - 2 * p.2 + 1 = 0

-- Define the circle equation
def circle : ℝ × ℝ → Prop := λ p, p.1^2 + (p.2 - 1)^2 = 1

-- The fixed point inside the circle
def fixed_point : ℝ × ℝ := (0, 1 / 2)

-- The statement of the proof
theorem line_intersects_circle (k : ℝ) : line k fixed_point → (fixed_point.1^2 + (fixed_point.2 - 1)^2 < 1) → 
  ∃ p : ℝ × ℝ, line k p ∧ circle p :=
by sorry

end line_intersects_circle_l368_368900


namespace pascal_triangle_count_30_rows_l368_368287

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l368_368287


namespace find_q_of_quadratic_with_roots_ratio_l368_368503

theorem find_q_of_quadratic_with_roots_ratio {q : ℝ} :
  (∃ r1 r2 : ℝ, r1 ≠ 0 ∧ r2 ≠ 0 ∧ r1 / r2 = 3 / 1 ∧ r1 + r2 = -10 ∧ r1 * r2 = q) →
  q = 18.75 :=
by
  sorry

end find_q_of_quadratic_with_roots_ratio_l368_368503


namespace solve_for_d_l368_368761

theorem solve_for_d :
  ∃ (d : ℕ), 
  d = 3306 ∧ 
  (d / 114 : ℚ) = ( ∏ n in finset.range 57, (1 - 1 / (n+2)^2)) := 
sorry

end solve_for_d_l368_368761


namespace find_ellipse_eq_C_find_lambda_range_l368_368882

variables (a b : ℝ) (k m λ : ℝ) (x y : ℝ)
variable (Q : ℝ × ℝ)
variables (A B : ℝ × ℝ)

def ellipse (a b : ℝ) : set (ℝ × ℝ) := {p | let (x, y) := p in (x^2 / a^2) + (y^2 / b^2) = 1}

def line (k m : ℝ) : set (ℝ × ℝ) := {p | let (x, y) := p in y = k * x + m}

def intersects (C l : set (ℝ × ℝ)) := ∃ A B, A ≠ B ∧ A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l

def vector_eq (A B Q O : ℝ × ℝ) (λ : ℝ) : Prop :=
  let (xA, yA) := A in
  let (xB, yB) := B in
  let (xQ, yQ) := Q in
  let (xO, yO) := O in
  (xA - xO + xB - xO, yA - yO + yB - yO) = λ * (xQ - xO, yQ - yO)

theorem find_ellipse_eq_C : 
  let C := ellipse a b in
  (a > b > 0) → 
  foci_on_x_axis center_origin (a foci_dist f _) → 
  share_eccentricity_with (a b e ec _) → 
  (intersects C (line k m)) →
  (∃ Q, Q ∈ C ∧ vector_eq A B Q (0,0) λ) →
  (∀ a > b > 0, a = sqrt 2 ∧ b = 1 → ellipse a b = ⋃ p, (x^2 / 2) + y^2 = 1 := sorry

theorem find_lambda_range :
  let C := ellipse (sqrt 2) 1 in
  (intersects C (line k m)) →
  (∃ Q, Q ∈ C ∧ vector_eq A B Q (0,0) λ) →
  (-2 < λ ∧ λ < 2 ∧ λ ≠ 0) := sorry

end find_ellipse_eq_C_find_lambda_range_l368_368882


namespace cos_five_pi_over_three_l368_368085

theorem cos_five_pi_over_three : Real.cos (5 * Real.pi / 3) = 1 / 2 := 
by 
  sorry

end cos_five_pi_over_three_l368_368085


namespace stock_price_rise_l368_368210

theorem stock_price_rise {P : ℝ} (h1 : P > 0)
    (h2007 : P * 1.20 = 1.20 * P)
    (h2008 : 1.20 * P * 0.75 = P * 0.90)
    (hCertainYear : P * 1.17 = P * 0.90 * (1 + 30 / 100)) :
  30 = 30 :=
by sorry

end stock_price_rise_l368_368210


namespace factorization_correct_l368_368188

-- Define noncomputable to deal with the natural arithmetic operations
noncomputable def a : ℕ := 66
noncomputable def b : ℕ := 231

-- Define the given expressions
noncomputable def lhs (x : ℕ) : ℤ := ((a : ℤ) * x^6) - ((b : ℤ) * x^12)
noncomputable def rhs (x : ℕ) : ℤ := (33 : ℤ) * x^6 * (2 - 7 * x^6)

-- The theorem to prove the equality
theorem factorization_correct (x : ℕ) : lhs x = rhs x :=
by sorry

end factorization_correct_l368_368188


namespace complex_expression_power_l368_368655

theorem complex_expression_power {i : ℂ} (hi : i = complex.I) :
  ( (1 + i) / (1 - i) ) ^ 2017 = i :=
by
  sorry

end complex_expression_power_l368_368655


namespace log_sum_20_l368_368357

-- Define the positive geometric sequence and conditions
variable {a : ℕ → ℝ}
variable (h_geo : ∀ n : ℕ, a n > 0 ∧ a (n + 1) = a n * r) -- Geometric sequence
variable (cond : a 5 * a 6 = 81)

-- State the theorem
theorem log_sum_20 : (log 3 (a 1) + log 3 (a 2) + log 3 (a 3) + log 3 (a 4) + log 3 (a 5) + log 3 (a 6) + log 3 (a 7) + log 3 (a 8) + log 3 (a 9) + log 3 (a 10)) = 20 :=
by
  sorry

end log_sum_20_l368_368357


namespace pto_shirts_total_cost_l368_368878

theorem pto_shirts_total_cost :
  let cost_Kindergartners : ℝ := 101 * 5.80
  let cost_FirstGraders : ℝ := 113 * 5.00
  let cost_SecondGraders : ℝ := 107 * 5.60
  let cost_ThirdGraders : ℝ := 108 * 5.25
  cost_Kindergartners + cost_FirstGraders + cost_SecondGraders + cost_ThirdGraders = 2317.00 := by
  sorry

end pto_shirts_total_cost_l368_368878


namespace roots_of_quadratic_l368_368512

theorem roots_of_quadratic (k : ℝ) : 
  let a := 3
  let b := k
  let c := -5
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  let a := 3
  let b := k
  let c := -5
  have Δ_nonneg : Δ = b^2 - 4 * a * c := rfl
  rw [Δ_nonneg, sq, mul_assoc]
  sorry

end roots_of_quadratic_l368_368512


namespace extraordinary_numbers_count_l368_368846

/-- An integer is called "extraordinary" if it has exactly one even divisor other than 2.
We need to count the number of extraordinary numbers in the interval [1, 75]. -/
def is_extraordinary (n : ℕ) : Prop :=
  ∃ p : ℕ, nat.prime p ∧ p % 2 = 1 ∧ n = 2 * p

theorem extraordinary_numbers_count :
  (finset.filter (λ n : ℕ, n ≥ 1 ∧ n ≤ 75 ∧ is_extraordinary n) (finset.range 76)).card = 11 :=
by
  sorry

end extraordinary_numbers_count_l368_368846


namespace problem1_problem2_l368_368139

-- First Problem Statement:
theorem problem1 :  12 - (-18) + (-7) - 20 = 3 := 
by 
  sorry

-- Second Problem Statement:
theorem problem2 : -4 / (1 / 2) * 8 = -64 := 
by 
  sorry

end problem1_problem2_l368_368139


namespace count_arithmetic_sequences_l368_368262

theorem count_arithmetic_sequences :
  let a1 := 1783
  let an := 1993
  ∃ pairs : Finset (ℕ × ℕ), 
    (∀ n d, (n, d) ∈ pairs ↔ a1 + (n - 1) * d = an ∧ 3 ≤ n ∧ d > 2) ∧
    pairs.card = 13 :=
by
  sorry

end count_arithmetic_sequences_l368_368262


namespace problem1_problem2_l368_368577

-- Problem 1: Proof for expression calculation
theorem problem1 : (sqrt 24 - sqrt (1/2)) - (sqrt (1/8) + sqrt 6) = sqrt 6 - 3 * sqrt 2 / 4 := 
sorry

-- Problem 2: Proof for solving the quadratic equation
theorem problem2 (x : ℝ) : (x - 2)^2 = 3 * (x - 2) → x = 2 ∨ x = 5 := 
sorry

end problem1_problem2_l368_368577


namespace simplify_expression_l368_368115

def a : ℕ := 1050
def p : ℕ := 2101
def q : ℕ := 1050 * 1051

theorem simplify_expression : 
  (1051 / 1050) - (1050 / 1051) = (p : ℚ) / (q : ℚ) ∧ Nat.gcd p a = 1 ∧ Nat.gcd p (a + 1) = 1 :=
by 
  sorry

end simplify_expression_l368_368115


namespace no_non_neg_integer_y_segment_len_7_l368_368102

theorem no_non_neg_integer_y_segment_len_7 :
  ¬∃ y : ℕ, real.sqrt ((y - 2)^2 + (6 - 2)^2) = 7 :=
by
  sorry

end no_non_neg_integer_y_segment_len_7_l368_368102


namespace expected_distinct_value_correct_l368_368541

noncomputable def expected_distinct_values : ℕ :=
  let n := 2013 in
  n * (1 - (Real.exp ((n - 1 : ℝ) * (Math.log (1 - 1 / n) : ℝ))))

theorem expected_distinct_value_correct :
  expected_distinct_values = 2013 * (1 - (2012 / 2013) ^ 2013) :=
sorry

end expected_distinct_value_correct_l368_368541


namespace fewer_onions_correct_l368_368644

-- Define the quantities
def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

-- Calculate the total number of tomatoes and corn
def tomatoes_and_corn : ℕ := tomatoes + corn

-- Calculate the number of fewer onions
def fewer_onions : ℕ := tomatoes_and_corn - onions

-- State the theorem and provide the proof
theorem fewer_onions_correct : fewer_onions = 5200 :=
by
  -- The statement is proved directly by the calculations above
  -- Providing the actual proof is not necessary as per the guidelines
  sorry

end fewer_onions_correct_l368_368644


namespace composite_function_value_l368_368699

def f (x : ℝ) : ℝ :=
if x < 1 then 1 / x else x^2 - 1

theorem composite_function_value : f (f (1 / 3)) = 8 := by
  sorry

end composite_function_value_l368_368699


namespace height_from_AB_l368_368774

-- Given conditions
variables {A B C : Type}
variable [triangle ABC] -- assume triangle ABC exists
variable h : 2 * sqrt(3) = AB
variable j : 2 = AC
variable k : 30 = B

theorem height_from_AB {h j k} : height AB = 1 ∨ height AB = 2 :=
sorry

end height_from_AB_l368_368774


namespace students_in_classroom_l368_368032

/-- There are some students in a classroom. Half of them have 5 notebooks each and the other half have 3 notebooks each. There are 112 notebooks in total in the classroom. Prove the number of students is 28. -/
theorem students_in_classroom (S : ℕ) (h1 : (S / 2) * 5 + (S / 2) * 3 = 112) : S = 28 := 
sorry

end students_in_classroom_l368_368032


namespace unique_p_value_l368_368690

theorem unique_p_value (p : Nat) (h₁ : Nat.Prime (p+10)) (h₂ : Nat.Prime (p+14)) : p = 3 := by
  sorry

end unique_p_value_l368_368690


namespace find_first_number_l368_368610

variable (a : ℕ → ℤ)

axiom recurrence_rel : ∀ (n : ℕ), n ≥ 4 → a n = a (n - 1) + a (n - 2) + a (n - 3)
axiom a8_val : a 8 = 29
axiom a9_val : a 9 = 56
axiom a10_val : a 10 = 108

theorem find_first_number : a 1 = 32 :=
sorry

end find_first_number_l368_368610


namespace vector_subtraction_magnitude_l368_368742

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition1 : Real := 3 -- |a|
def condition2 : Real := 2 -- |b|
def condition3 : Real := 4 -- |a + b|

-- Proving the statement
theorem vector_subtraction_magnitude (h1 : ‖a‖ = condition1) (h2 : ‖b‖ = condition2) (h3 : ‖a + b‖ = condition3) :
  ‖a - b‖ = Real.sqrt 10 :=
by
  sorry

end vector_subtraction_magnitude_l368_368742


namespace digit_101_in_decimal_of_3_over_11_l368_368755

theorem digit_101_in_decimal_of_3_over_11 :
  let decimal_rep := "27".cycle
  (decimal_rep.nth 100) = '2' :=
by {
  simp,
  sorry
}

end digit_101_in_decimal_of_3_over_11_l368_368755


namespace Jose_share_land_l368_368807

theorem Jose_share_land (total_land : ℕ) (num_siblings : ℕ) (total_parts : ℕ) (share_per_person : ℕ) :
  total_land = 20000 → num_siblings = 4 → total_parts = (1 + num_siblings) → share_per_person = (total_land / total_parts) → 
  share_per_person = 4000 :=
by
  sorry

end Jose_share_land_l368_368807


namespace side_length_larger_triangle_l368_368531

theorem side_length_larger_triangle
  (side_of_smaller_triangle : ℝ)
  (total_shaded_area : ℝ)
  (area_larger_triangle : ℝ)
  (h1 : side_of_smaller_triangle = 1)
  (h2 : total_shaded_area = 3 * (sqrt 3 / 4 * side_of_smaller_triangle^2))
  (h3 : total_shaded_area = 1/2 * area_larger_triangle) :
  (∃ s : ℝ, area_larger_triangle = (sqrt 3 / 4 * s^2) ∧ s = sqrt 6) :=
by
  sorry

end side_length_larger_triangle_l368_368531


namespace f_monotonically_decreasing_l368_368443

noncomputable def f (ν : ℝ) : ℝ :=
  ∫ x in (1 : ℝ)..(1 / ν), 1 / Real.sqrt((x ^ 2 - 1) * (1 - ν ^ 2 * x ^ 2))

theorem f_monotonically_decreasing (hν : 0 < ν ∧ ν < 1) : 
  ∀ (ν : ℝ), 0 < ν ∧ ν < 1 → (∂/∂ν, f ν) < 0 := sorry

end f_monotonically_decreasing_l368_368443


namespace root_interval_exists_l368_368679

def f (x : ℝ) : ℝ := 3 * x - 7 + Real.log x

theorem root_interval_exists : ∃ n : ℕ, 2 = n ∧ ∃ x : ℝ, x ∈ Set.Ioo (n : ℝ) (n+1 : ℝ) ∧ f x = 0 :=
by
  use 2
  split
  sorry

end root_interval_exists_l368_368679


namespace poster_height_proportion_l368_368111

-- Defining the given conditions
def original_width : ℕ := 3
def original_height : ℕ := 2
def new_width : ℕ := 12
def scale_factor := new_width / original_width

-- The statement to prove the new height
theorem poster_height_proportion :
  scale_factor = 4 → (original_height * scale_factor) = 8 :=
by
  sorry

end poster_height_proportion_l368_368111


namespace greatest_possible_value_l368_368753

theorem greatest_possible_value (x : ℝ) (hx : x^3 + (1 / x^3) = 9) : x + (1 / x) = 3 := by
  sorry

end greatest_possible_value_l368_368753


namespace minimum_balls_needed_l368_368585

noncomputable def ballColors : Type := {red green yellow blue white black}

def ballsInBox : ballColors → ℕ := 
  λ color, 
    match color with
    | red => 28
    | green => 20
    | yellow => 13
    | blue => 19
    | white => 11
    | black => 9

theorem minimum_balls_needed 
  (balls : ballColors → ℕ := ballsInBox)
  (h : ∀ color : ballColors, balls color > 0) :
  ∃ n, (∀ draws : fin n → ballColors, ∃ color, (draws.val.filter (λ c, c = color)).length ≥ 15) ↔ (n ≥ 76) :=
sorry

end minimum_balls_needed_l368_368585


namespace handshake_count_l368_368524

theorem handshake_count (m n : ℕ) (h1 : m = 5) (h2 : n = 3) : 
  let total_people := m * n in
  let handshakes_per_person := (total_people - 1 - (m - 1)) in
  (total_people * handshakes_per_person) / 2 = 75 :=
by 
  sorry

end handshake_count_l368_368524


namespace pascal_triangle_sum_first_30_rows_l368_368289

theorem pascal_triangle_sum_first_30_rows :
  (Finset.range 30).sum (λ n, n + 1) = 465 :=
begin
  sorry
end

end pascal_triangle_sum_first_30_rows_l368_368289


namespace solve_triangle_l368_368875

noncomputable def sin_deg (d : ℝ) : ℝ := Real.sin (d * Real.pi / 180)

theorem solve_triangle :
  ∃ a b c : ℝ,
    t = 4920 ∧
    α = 43 + 36 / 60 + 10 / 3600 ∧
    β = 72 + 23 / 60 + 11 / 3600 ∧
    γ = 180 - (43 + 36 / 60 + 10 / 3600) - (72 + 23 / 60 + 11 / 3600) ∧
    a ≈ 89 ∧
    b ≈ 123 ∧
    c ≈ 116 :=
by
  let t := 4920
  let α := 43 + 36 / 60 + 10 / 3600
  let β := 72 + 23 / 60 + 11 / 3600
  let γ := 180 - α - β
  let sin_α := sin_deg α
  let sin_β := sin_deg β
  let sin_γ := sin_deg γ
  let a := Real.sqrt (2 * t * sin_α / (sin_β * sin_γ))
  let b := Real.sqrt (2 * t * sin_β / (sin_α * sin_γ))
  let c := Real.sqrt (2 * t * sin_γ / (sin_α * sin_β))
  use [a, b, c]
  split
  . 
  -- Area of the triangle condition
  exact t = 4920
  split
  . 
  -- Angle α condition
  exact α = 43 + 36 / 60 + 10 / 3600
  split
  . 
  -- Angle β condition
  exact β = 72 + 23 / 60 + 11 / 3600
  split
  . 
  -- Angle γ condition
  exact γ = 180 - α - β
  split
  .
  -- Side a approximation (calculation skipped)
  exact a ≈ 89
  split
  .
  -- Side b approximation (calculation skipped)
  exact b ≈ 123
  .
  -- Side c approximation (calculation skipped)
  exact c ≈ 116
  -- Sorry used to skip proofs
  sorry

end solve_triangle_l368_368875


namespace range_of_a_l368_368765

noncomputable  theory

-- Define the curve
def curve (x : ℝ) : ℝ := abs (x - real.exp 1 * real.log x) + x

-- Define the statement to be proven
theorem range_of_a (a : ℝ) : (∃ m b : ℝ, ∀ x : ℝ, curve x = m * x + b ∧ curve 1 = m * 1 + b = a) → a < 2 :=
by
  sorry

end range_of_a_l368_368765


namespace probability_of_meeting_at_cafe_l368_368124

noncomputable def alice_charlie_meet_probability : ℝ :=
  let meet_event_area : ℝ :=
    let total_area : ℝ := 1
    let nonmeet_area_triangles : ℝ := 2 * (1 / 2 * (2 / 3) ^ 2)
    total_area - nonmeet_area_triangles
  meet_event_area

theorem probability_of_meeting_at_cafe :
  alice_charlie_meet_probability = 5 / 9 :=
by
  sorry

end probability_of_meeting_at_cafe_l368_368124


namespace probability_distinct_zeros_of_f_l368_368069

noncomputable def discriminant (a : ℤ) : ℤ :=
  (2 * a) ^ 2 - 4 * 1 * 2

theorem probability_distinct_zeros_of_f 
  (a : ℤ) 
  (h : 1 ≤ a ∧ a ≤ 6) :
  (1 : ℚ)  / 6 * ∑ a in {2, 3, 4, 5, 6}, if discriminant a > 0 then 1 else 0 = 5 / 6 :=
sorry

end probability_distinct_zeros_of_f_l368_368069


namespace evaluate_expression_l368_368186

theorem evaluate_expression (x : ℝ) (h1 : x^3 + 2 ≠ 0) (h2 : x^3 - 2 ≠ 0) :
  (( (x+2)^3 * (x^2-x+2)^3 / (x^3+2)^3 )^3 * ( (x-2)^3 * (x^2+x+2)^3 / (x^3-2)^3 )^3 ) = 1 :=
by
  sorry

end evaluate_expression_l368_368186


namespace ball_hits_ground_in_seconds_l368_368883

theorem ball_hits_ground_in_seconds :
  ∃ t : ℝ, y(t) = 0 ∧ t ≈ 2.34 :=
  let y (t : ℝ) : ℝ := -8 * t^2 - 12 * t + 72 in
  sorry

end ball_hits_ground_in_seconds_l368_368883


namespace equal_likelihood_each_number_not_equal_likelihood_all_selections_l368_368171

def k : ℕ := sorry
def n : ℕ := sorry
def selection : list ℕ := sorry

-- Condition 1: Each subsequent number is chosen independently of the previous ones.
axiom independent_choice (m : ℕ) : m ∈ selection → m < n

-- Condition 2: If it matches one of the already chosen numbers, you should move clockwise to the first unchosen number.
axiom move_clockwise (m : ℕ) : m ∈ selection → ∃ l, l ≠ m ∧ l ∉ selection

-- Condition 3: In the end, k different numbers are obtained.
axiom k_distinct : list.distinct selection ∧ list.length selection = k

-- Define the events A and B
def A : Prop := ∀ m : ℕ, m ∈ selection → ∃ p, probability m = p
def B : Prop := ∀ s1 s2 : list ℕ, s1 ⊆ selection ∧ s1.length = k ∧ s2 ⊆ selection ∧ s2.length = k → s1 ≠ s2 → probability s1 ≠ probability s2

-- Prove A holds given the conditions
theorem equal_likelihood_each_number : A :=
sorry

-- Prove B does not hold given the conditions
theorem not_equal_likelihood_all_selections : ¬B :=
sorry

end equal_likelihood_each_number_not_equal_likelihood_all_selections_l368_368171


namespace positive_rational_as_factorial_primes_quotient_l368_368449

theorem positive_rational_as_factorial_primes_quotient 
  (q : ℚ) (hq : 0 < q) : 
  ∃ (p q : list ℕ), (∀ x ∈ p, nat.prime x) ∧ (∀ y ∈ q, nat.prime y) ∧ 
    (q = (list.prod (p.map nat.factorial) : ℚ) / (list.prod (q.map nat.factorial) : ℚ)) :=
sorry

end positive_rational_as_factorial_primes_quotient_l368_368449


namespace appearance_equally_likely_selections_not_equally_likely_l368_368179

-- Define the conditions under which the numbers are chosen
def independent_choice (n k : ℕ) (selection : ℕ → Set ℕ) : Prop :=
∀ i j : ℕ, i ≠ j → selection i ∩ selection j = ∅

def move_clockwise (chosen : Set ℕ) (n : ℕ) : ℕ → ℕ := sorry  -- Placeholder for clockwise movement function

-- Condition: Each subsequent number is chosen independently of the previous ones
def conditions (n k : ℕ) : Prop :=
∃ selection : ℕ → Set ℕ, independent_choice n k selection

-- Problem a: Prove that the appearance of each specific number is equally likely
theorem appearance_equally_likely (n k : ℕ) (h : conditions n k) : 
  (∀ num : ℕ, 1 ≤ num ∧ num ≤ n → appearance_probability num k n) := sorry

-- Problem b: Prove that the appearance of all selections is not equally likely
theorem selections_not_equally_likely (n k : ℕ) (h : conditions n k) : 
  ¬(∀ selection : Set ℕ, appearance_probability selection k n) := sorry

end appearance_equally_likely_selections_not_equally_likely_l368_368179


namespace tory_video_games_l368_368530

theorem tory_video_games (T J: ℕ) :
    (3 * J + 5 = 11) → (J = T / 3) → T = 6 :=
by
  sorry

end tory_video_games_l368_368530


namespace shaded_area_l368_368196

theorem shaded_area (R : ℝ) (α := 20 * (π / 180)) : 
  let S₀ := (π * R^2) / 2 in
  let sector_area := (1 / 2) * (2 * R)^2 * (α) in
  sector_area = (2 * π * R^2) / 9 :=
by 
  let S₀ := (π * R^2) / 2
  let sector_area := (1 / 2) * (2 * R)^2 * (20 * (π / 180))
  sorry

end shaded_area_l368_368196


namespace value_of_expression_l368_368763

variable (x y : ℝ)

theorem value_of_expression 
  (h1 : x + Real.sqrt (x * y) + y = 9)
  (h2 : x^2 + x * y + y^2 = 27) :
  x - Real.sqrt (x * y) + y = 3 :=
sorry

end value_of_expression_l368_368763


namespace perfect_squares_as_difference_l368_368749

theorem perfect_squares_as_difference (N : ℕ) (hN : N = 20000) : 
  (∃ (n : ℕ), n = 71 ∧ 
    ∀ m < N, 
      (∃ a b : ℤ, 
        a^2 = m ∧
        b^2 = m + ((b + 1)^2 - b^2) - 1 ∧ 
        (b + 1)^2 - b^2 = 2 * b + 1)) :=
by 
  sorry

end perfect_squares_as_difference_l368_368749


namespace trapezoid_problem_l368_368862

theorem trapezoid_problem (b h x : ℝ) 
  (h1 : x = (12500 / (x - 75)) - 75)
  (h_cond : (b + 75) / (b + 25) = 3 / 2)
  (b_solution : b = 75) :
  (⌊(x^2 / 100)⌋ : ℤ) = 181 :=
by
  -- The statement only requires us to assert the proof goal
  sorry

end trapezoid_problem_l368_368862


namespace find_center_of_symmetry_l368_368479

def center_of_symmetry (x y : ℝ) : Prop :=
  ∃ k : ℤ, x = k * π / 2 - π / 6 ∧ y = - √3 / 2

theorem find_center_of_symmetry : 
  center_of_symmetry (⟨λ x, sin x * cos x + √3 * cos x ^ 2 - √3⟩) :=
sorry

end find_center_of_symmetry_l368_368479


namespace right_triangle_acute_angle_l368_368359

/-- In a right triangle, if one acute angle is 25 degrees, then the measure of the other acute angle is 65 degrees. -/
theorem right_triangle_acute_angle (α : ℝ) (hα : α = 25) : ∃ β : ℝ, β = 65 := 
by
  have h_sum : α + 65 = 90 := by
    rw hα
    norm_num
  use 65
  exact h_sum.symm
  sorry

end right_triangle_acute_angle_l368_368359


namespace pascal_triangle_sum_l368_368325

theorem pascal_triangle_sum (n : ℕ) (h₀ : n = 29) :
  (∑ k in Finset.range (n + 1), k + 1) = 465 := sorry

end pascal_triangle_sum_l368_368325


namespace finite_set_of_functions_identity_l368_368674

theorem finite_set_of_functions_identity {A : set (ℝ → ℝ)} (hA : finite A) (hneA : A.nonempty) :
  (∀ f1 f2 ∈ A, ∃ g ∈ A, ∀ x y : ℝ, f1 (f2 y - x) + 2 * x = g (x + y)) →
  A = {id} :=
by
  sorry

end finite_set_of_functions_identity_l368_368674


namespace max_number_of_squares_with_twelve_points_l368_368041

-- Define the condition: twelve marked points in a grid
def twelve_points_marked_on_grid : Prop := 
  -- Assuming twelve specific points represented in a grid-like structure
  -- (This will be defined concretely in the proof implementation context)
  sorry

-- Define the problem statement to be proved
theorem max_number_of_squares_with_twelve_points : 
  twelve_points_marked_on_grid → (∃ n, n = 11) :=
by 
  sorry

end max_number_of_squares_with_twelve_points_l368_368041


namespace unit_vector_orthogonal_to_v1_v2_l368_368670

-- Definitions for the given vectors
def v1 : ℝ × ℝ × ℝ := (2, 1, 1)
def v2 : ℝ × ℝ × ℝ := (0, 1, 3)

-- The unit vector that needs to be verified
def u : ℝ × ℝ × ℝ := (1 / Real.sqrt 11, -3 / Real.sqrt 11, 1 / Real.sqrt 11)

-- Proof statement
theorem unit_vector_orthogonal_to_v1_v2 : 
  (∥u∥ = 1) ∧ (u.1 * v1.1 + u.2 * v1.2 + u.3 * v1.3 = 0) ∧ (u.1 * v2.1 + u.2 * v2.2 + u.3 * v2.3 = 0) :=
by 
  sorry

end unit_vector_orthogonal_to_v1_v2_l368_368670


namespace main_theorem_l368_368916

open EuclideanGeometry

variables {A B C D E F M P : Point}

def midpoint (M B C : Point) := dist M B = dist M C

def is_circumcenter (P A D E : Point) : Prop :=
dist P A = dist P D ∧ dist P D = dist P E

def foot_of_perpendicular (A F B C : Point) : Prop :=
∃. (Line A F) ∧ orthogonal_projection (Line B C) A = F

def triangle_isosceles (A D E : Point) : Prop :=
dist A D = dist A E

noncomputable def triangular_configuration :=
∀ (A B C M D E F P : Point),
midpoint M B C ∧ 
foot_of_perpendicular A F B C ∧ 
is_circumcenter P A D E ∧ 
triangle_isosceles A D E → 
dist P F = dist P M

theorem main_theorem (h : triangular_configuration A B C M D E F P) : 
  dist P F = dist P M :=
by sorry

end main_theorem_l368_368916


namespace slope_AB_is_pm_2sqrt2_l368_368839

noncomputable def parabola_focus (p : ℝ) (hp : p > 0) : ℝ × ℝ := 
  (1 / (8 * p), 0)

noncomputable def parabola_directrix (p : ℝ) (hp : p > 0) : ℝ → ℝ := 
  λ y, -1 / (8 * p)

noncomputable def point_on_parabola (p x : ℝ) (hp : p > 0) : ℝ :=
  sqrt (x / (2 * p))

noncomputable def point_on_directrix (p y : ℝ) (hp : p > 0) : ℝ :=
  -1 / (8 * p)

theorem slope_AB_is_pm_2sqrt2
  (p : ℝ) (hp : p > 0)
  (F := parabola_focus p hp)
  (l := parabola_directrix p hp)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hA : A.1 = 2 * p * (A.2)^2)
  (hB : B.1 = l B.2)
  (hAF_FB : (F.1 - A.1, -A.2) = (1 / 2) • (B.1 - F.1, B.2)) :
  (A.2 = 1 / (2 * sqrt 2 * p) ∧ (B.2 = - 1 / (sqrt 2 * p) ∧ (B.1 - A.1 = - 1 / (4 * p))) ∧
    (-2 * sqrt 2 ≤ (B.2 - A.2) / (B.1 - A.1) ∧ (B.2 - A.2) / (B.1 - A.1) ≤ 2 * sqrt 2)) ∨
  (A.2 = -1 / (2 * sqrt 2 * p) ∧ (B.2 = 1 / (sqrt 2 * p) ∧ (B.1 - A.1 = - 1 / (4 * p))) ∧
    (-2 * sqrt 2 ≤ (B.2 - A.2) / (B.1 - A.1) ∧ (B.2 - A.2) / (B.1 - A.1) ≤ 2 * sqrt 2)) :=
sorry

end slope_AB_is_pm_2sqrt2_l368_368839


namespace tetrahedron_area_analogous_pythagorean_l368_368762

variable (S S1 S2 S3 : ℝ)

/-- In a tetrahedron O-ABC with right angles at O, the square of the area of the face opposite O 
    is equal to the sum of the squares of the areas of the other three faces. -/
theorem tetrahedron_area_analogous_pythagorean (h : S = sqrt (S1^2 + S2^2 + S3^2)) :
  S^2 = S1^2 + S2^2 + S3^2 :=
by sorry

end tetrahedron_area_analogous_pythagorean_l368_368762


namespace smallest_n_with_four_pairs_l368_368927

theorem smallest_n_with_four_pairs :
  ∃ n : ℕ, n = 72 ∧ (finset.univ.filter (λ (ab : ℕ × ℕ), ab.fst > 0 ∧ ab.snd > 0 ∧ ab.fst^2 + ab.snd^2 = n)).card = 4 :=
by 
  sorry

end smallest_n_with_four_pairs_l368_368927


namespace sin_of_angle_through_point_l368_368772

theorem sin_of_angle_through_point (x y : ℝ) (h : x = -1 ∧ y = 2) :
  let r := real.sqrt (x^2 + y^2)
  let sinα := y / r
  sinα = (2 * real.sqrt 5) / 5 :=
by
  obtain ⟨hx, hy⟩ := h
  -- sesorry

end sin_of_angle_through_point_l368_368772


namespace pascal_triangle_sum_l368_368323

theorem pascal_triangle_sum (n : ℕ) (h₀ : n = 29) :
  (∑ k in Finset.range (n + 1), k + 1) = 465 := sorry

end pascal_triangle_sum_l368_368323


namespace smallest_positive_integer_l368_368063

theorem smallest_positive_integer {x : ℕ} (h1 : x % 6 = 3) (h2 : x % 8 = 5) : x = 21 :=
sorry

end smallest_positive_integer_l368_368063


namespace min_liked_both_l368_368435

noncomputable def total_people := 130
noncomputable def liked_beethoven := 110
noncomputable def liked_chopin := 90

theorem min_liked_both : ∃ (x : ℕ), x ≥ 70 ∧
  liked_beethoven + liked_chopin - total_people = x :=
by
  let B := liked_beethoven
  let C := liked_chopin
  let T := total_people
  let min_liked_both := B + C - T
  have h : min_liked_both = 70 := by sorry
  show ∃ (x : ℕ), x ≥ min_liked_both ∧ min_liked_both = 70 from
    ⟨min_liked_both, by norm_num, h⟩
  sorry

end min_liked_both_l368_368435


namespace moscow_olympiad_1967_l368_368207

theorem moscow_olympiad_1967 (k : ℕ) (h_k : k > 4) :
  let primes := (nat.prime_filter_lt (p_k+1)).filter (λ p, nat.prime p),
      S := (∑ m in powerset primes, m.prod) 
  in (S + 1).factors.length ≥ 2 * k :=
by
  let primes := (nat.prime_filter_lt (p_k+1)).filter (λ p, nat.prime p)
  let S := (∑ m in powerset primes, m.prod)
  let R := primes.map (λ p, (p + 1))
  have len_factors : (∏ i in R, i).factors.length ≥ 2 * k
  { sorry }
  exact len_factors

end moscow_olympiad_1967_l368_368207


namespace estimate_employees_between_1000_and_2000_l368_368594

variable (total_employees sampled_employees within_1000 within_2000 : ℕ)

def employees_between_1000_and_2000 (total_employees sampled_employees within_1000 within_2000 : ℕ) : ℕ :=
  (within_2000 - within_1000) * (total_employees / sampled_employees)

theorem estimate_employees_between_1000_and_2000 
  (h_total : total_employees = 2000)
  (h_sample : sampled_employees = 200)
  (h_within_1000 : within_1000 = 10)
  (h_within_2000 : within_2000 = 30) :
  employees_between_1000_and_2000 total_employees sampled_employees within_1000 within_2000 = 200 := 
  by {
    rw [h_total, h_sample, h_within_1000, h_within_2000],
    unfold employees_between_1000_and_2000,
    norm_num,
    sorry
  }

end estimate_employees_between_1000_and_2000_l368_368594


namespace donuts_per_box_l368_368382

-- Define the conditions and the theorem
theorem donuts_per_box :
  (10 * 12 - 12 - 8) / 10 = 10 :=
by
  sorry

end donuts_per_box_l368_368382


namespace extraordinary_numbers_in_interval_l368_368845

def is_extraordinary (n : ℕ) : Prop :=
  n % 2 = 0 ∧ ∃ p : ℕ, Nat.Prime p ∧ n = 2 * p

def count_extraordinary (a b : ℕ) : ℕ :=
  (Finset.filter (λ n, is_extraordinary n) (Finset.range' a (b + 1))).card

theorem extraordinary_numbers_in_interval :
  count_extraordinary 1 75 = 12 := 
sorry

end extraordinary_numbers_in_interval_l368_368845


namespace pascal_triangle_elements_count_l368_368267

theorem pascal_triangle_elements_count :
  ∑ n in finset.range 30, (n + 1) = 465 :=
by 
  sorry

end pascal_triangle_elements_count_l368_368267


namespace brownies_count_l368_368385

theorem brownies_count (pan_length : ℕ) (pan_width : ℕ) (piece_side : ℕ) 
  (h1 : pan_length = 24) (h2 : pan_width = 15) (h3 : piece_side = 3) : 
  (pan_length * pan_width) / (piece_side * piece_side) = 40 :=
by {
  sorry
}

end brownies_count_l368_368385


namespace combined_forgotten_angles_l368_368167

-- Define primary conditions
def initial_angle_sum : ℝ := 2873
def correct_angle_sum : ℝ := 16 * 180

-- The theorem to prove
theorem combined_forgotten_angles : correct_angle_sum - initial_angle_sum = 7 :=
by sorry

end combined_forgotten_angles_l368_368167


namespace perfect_square_pairs_l368_368675

theorem perfect_square_pairs (p : ℕ) (hp : Nat.Prime p) (h2 : p > 2) :
  ∃ a b : ℕ, a - b = p ∧ ∃ k : ℕ, a * b = k^2 ∧ a = ((p + 1) / 2)^2 ∧ b = ((p - 1) / 2)^2 :=
begin
  let b := ((p - 1) / 2)^2,
  let a := ((p + 1) / 2)^2,
  use [a, b],
  have hab : a - b = p, by sorry,
  have hsq : ∃ k : ℕ, a * b = k^2, by sorry,
  exact ⟨hab, hsq, rfl, rfl⟩
end

end perfect_square_pairs_l368_368675


namespace minimize_blue_surface_l368_368956

noncomputable def fraction_blue_surface_area : ℚ := 1 / 8

theorem minimize_blue_surface
  (total_cubes : ℕ)
  (blue_cubes : ℕ)
  (green_cubes : ℕ)
  (edge_length : ℕ)
  (surface_area : ℕ)
  (blue_surface_area : ℕ)
  (fraction_blue : ℚ)
  (h1 : total_cubes = 64)
  (h2 : blue_cubes = 20)
  (h3 : green_cubes = 44)
  (h4 : edge_length = 4)
  (h5 : surface_area = 6 * edge_length^2)
  (h6 : blue_surface_area = 12)
  (h7 : fraction_blue = blue_surface_area / surface_area) :
  fraction_blue = fraction_blue_surface_area :=
by
  sorry

end minimize_blue_surface_l368_368956


namespace trajectory_is_parabola_l368_368604

noncomputable def distancePoint (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def distanceLine (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / real.sqrt (a^2 + b^2)

theorem trajectory_is_parabola :
  ∀ P : ℝ × ℝ, distancePoint P (3,0) = distanceLine P 1 0 2 + 1 →
  (∃ (a b c : ℝ), (a * P.1 ^ 2 + b * P.1 * P.2 + c * P.2 ^ 2 + d * P.1 + e * P.2 + f = 0) 
    ∧ ∃ D : ℝ, D ≠ 0 ∧ D = b^2 - 4 * a * c ∧ D = 0 := (by sorry)) :=
sorry

end trajectory_is_parabola_l368_368604


namespace domain_of_f_l368_368485

def f (x : ℝ) : ℝ := (Real.log (4 - x)) / (x - 3)

theorem domain_of_f :
  ∀ x : ℝ, (x < 4 ∧ x ≠ 3) ↔ ∃ y : ℝ, f y = f x := 
sorry

end domain_of_f_l368_368485


namespace smallest_scalene_triangle_perimeter_l368_368968

-- Define what it means for a number to be a prime number greater than 3
def prime_gt_3 (n : ℕ) : Prop := Prime n ∧ 3 < n

-- Define the main theorem
theorem smallest_scalene_triangle_perimeter : ∃ (a b c : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  prime_gt_3 a ∧ prime_gt_3 b ∧ prime_gt_3 c ∧
  Prime (a + b + c) ∧ 
  (∀ (x y z : ℕ), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    prime_gt_3 x ∧ prime_gt_3 y ∧ prime_gt_3 z ∧
    Prime (x + y + z) → (a + b + c) ≤ (x + y + z)) ∧
  a + b + c = 23 := by
    sorry

end smallest_scalene_triangle_perimeter_l368_368968


namespace maximum_volume_of_container_l368_368051

noncomputable def max_volume_of_container : ℝ :=
  let total_length : ℝ := 14.8
  let side_diff : ℝ := 0.5
  let volume (x : ℝ) : ℝ := x * (x + 0.5) * (3.45 - x)
  let critical_x : ℝ := 1
  volume critical_x

theorem maximum_volume_of_container (total_length : ℝ) (side_diff : ℝ) (h_total_length : total_length = 14.8) (h_side_diff : side_diff = 0.5) :
  max_volume_of_container = 3.675 :=
by 
  let x := 1
  let h := 3.45 - x
  let volume := x * (x + 0.5) * h
  have h1 : 2 * x + 2 * (x + 0.5) + 4 * h = total_length := by sorry
  have h2 : volume = 3.675 := by sorry
  exact h2

end maximum_volume_of_container_l368_368051


namespace pairs_satisfying_inequality_l368_368328

noncomputable def count_pairs : ℕ :=
  fintype.card {p : ℕ × ℕ // p.1 ≤ 1000 ∧ p.2 ≤ 1000 ∧ p.1 / (p.2 + 1) < real.sqrt 2 ∧ real.sqrt 2 < (p.1 + 1) / p.2}

theorem pairs_satisfying_inequality : count_pairs = 1706 :=
sorry

end pairs_satisfying_inequality_l368_368328


namespace specified_time_is_30_total_constuction_cost_is_180000_l368_368590

noncomputable def specified_time (x : ℕ) :=
  let teamA_rate := 1 / (x:ℝ)
  let teamB_rate := 2 / (3 * (x:ℝ))
  (teamA_rate + teamB_rate) * 15 + 5 * teamA_rate = 1

theorem specified_time_is_30 : specified_time 30 :=
  by 
    sorry

noncomputable def total_constuction_cost (x : ℕ) (costA : ℕ) (costB : ℕ) :=
  let teamA_rate := 1 / (x:ℝ)
  let teamB_rate := 2 / (3 * (x:ℝ))
  let total_time := 1 / (teamA_rate + teamB_rate)
  total_time * (costA + costB)

theorem total_constuction_cost_is_180000 : total_constuction_cost 30 6500 3500 = 180000 :=
  by 
    sorry

end specified_time_is_30_total_constuction_cost_is_180000_l368_368590


namespace volume_of_solid_rotated_about_y_axis_l368_368138

theorem volume_of_solid_rotated_about_y_axis :
  ∫ y in 0..(Real.pi / 2), π * (9 * (cos y)^2 - (cos y)^2) dy = 2 * π^2 := 
by
  sorry

end volume_of_solid_rotated_about_y_axis_l368_368138


namespace power_function_value_at_3_l368_368235

theorem power_function_value_at_3
  (f : ℝ → ℝ)
  (h1 : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α)
  (h2 : f 2 = 1 / 4) :
  f 3 = 1 / 9 := 
sorry

end power_function_value_at_3_l368_368235


namespace original_bubble_radius_l368_368621

theorem original_bubble_radius (r : ℝ) (R : ℝ) (π : ℝ) 
  (h₁ : r = 4 * real.cbrt 2)
  (h₂ : (4/3) * π * R^3 = (2/3) * π * r^3) : 
  R = 4 :=
by 
  sorry

end original_bubble_radius_l368_368621


namespace dogs_barking_ratio_l368_368909

theorem dogs_barking_ratio
  (total_dogs : ℕ)
  (running_dogs : ℕ)
  (playing_dogs : ℕ)
  (doing_nothing_dogs : ℕ)
  (barking_dogs : ℕ)
  (h_total : total_dogs = 88)
  (h_running : running_dogs = 12)
  (h_playing : playing_dogs = 44)
  (h_nothing : doing_nothing_dogs = 10)
  (h_accounted : running_dogs + playing_dogs + doing_nothing_dogs + barking_dogs = total_dogs) :
  barking_dogs.to_rat / total_dogs.to_rat = 1 / 4 :=
by
  sorry

end dogs_barking_ratio_l368_368909


namespace remainder_division_l368_368738

variable (P D K Q R R'_q R'_r : ℕ)

theorem remainder_division (h1 : P = Q * D + R) (h2 : R = R'_q * K + R'_r) (h3 : K < D) : 
  P % (D * K) = R'_r :=
sorry

end remainder_division_l368_368738


namespace vertical_asymptote_values_l368_368208

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := (x^2 - x + c) / (x^2 + x - 20)

theorem vertical_asymptote_values (c : ℝ) :
  (∃ x : ℝ, (x^2 + x - 20 = 0 ∧ x^2 - x + c = 0) ↔
   (c = -12 ∨ c = -30)) := sorry

end vertical_asymptote_values_l368_368208


namespace relationship_of_products_l368_368332

theorem relationship_of_products
  {a1 a2 b1 b2 : ℝ}
  (h1 : a1 < a2)
  (h2 : b1 < b2) :
  a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 :=
sorry

end relationship_of_products_l368_368332


namespace sum_hyperbola_constants_eq_l368_368832

noncomputable def hyperbola_parameters : ℝ :=
  let F1 : ℝ × ℝ := (-4, 2 - real.sqrt 3) in
  let F2 : ℝ × ℝ := (-4, 2 + real.sqrt 3) in
  let h := -4 in
  let k := 2 in
  let a := 1 in
  let b := real.sqrt 2 in
  h + k + a + b

theorem sum_hyperbola_constants_eq : 
  hyperbola_parameters = -1 + real.sqrt 2 :=
by
  sorry

end sum_hyperbola_constants_eq_l368_368832


namespace pascal_triangle_sum_l368_368322

theorem pascal_triangle_sum (n : ℕ) (h₀ : n = 29) :
  (∑ k in Finset.range (n + 1), k + 1) = 465 := sorry

end pascal_triangle_sum_l368_368322


namespace average_speed_of_bike_l368_368091

theorem average_speed_of_bike (distance : ℕ) (time : ℕ) (h1 : distance = 21) (h2 : time = 7) : distance / time = 3 := by
  sorry

end average_speed_of_bike_l368_368091


namespace pages_copyable_l368_368799

-- Define the conditions
def cents_per_dollar : ℕ := 100
def dollars_available : ℕ := 25
def cost_per_page : ℕ := 3

-- Define the total cents available
def total_cents : ℕ := dollars_available * cents_per_dollar

-- Define the expected number of full pages
def expected_pages : ℕ := 833

theorem pages_copyable :
  (total_cents : ℕ) / cost_per_page = expected_pages := sorry

end pages_copyable_l368_368799


namespace choir_members_l368_368881

theorem choir_members (k m n : ℕ) (h1 : n = k^2 + 11) (h2 : n = m * (m + 5)) : n ≤ 325 :=
by
  sorry -- A proof would go here, showing that n = 325 meets the criteria

end choir_members_l368_368881


namespace unique_sum_count_l368_368093

def coins : List ℕ := [1, 1, 5, 5, 10, 10, 25, 50]

def coin_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  have pairs := for x in l, y in l, if x ≠ y then (x, y) else none
  pairs.filter_map id

def unique_sums (pairs : List (ℕ × ℕ)) : Finset ℕ :=
  pairs.toFinset.map (⟨λ p => p.1 + p.2, sorry⟩ : Finset (ℕ × ℕ) ↪ Finset ℕ)

theorem unique_sum_count : (unique_sums (coin_pairs coins)).card = 15 :=
  sorry

end unique_sum_count_l368_368093


namespace pascal_triangle_elements_count_l368_368265

theorem pascal_triangle_elements_count :
  ∑ n in finset.range 30, (n + 1) = 465 :=
by 
  sorry

end pascal_triangle_elements_count_l368_368265


namespace find_general_term_and_sum_l368_368401

noncomputable def a_n (n : ℕ) : ℕ := 2 * n

noncomputable def b_n (n : ℕ) : ℚ := (a_n (n + 1) / a_n n) + (a_n n / a_n (n + 1)) - 2

noncomputable def S_n (n : ℕ) : ℕ := n * a_n 1 + (n * (n - 1) / 2) * 2

theorem find_general_term_and_sum (S_10_def : S_n 10 = 110) (S_15_def : S_n 15 = 240) :
  (∀ n, a_n n = 2 * n) ∧ (∀ n, (finset.range(n).sum (λ k, b_n k) = (2 * n^2 + 3 * n) / (n + 1))) :=
by
  sorry

end find_general_term_and_sum_l368_368401


namespace appearance_equally_likely_all_selections_not_equally_likely_l368_368183

variables {n k : ℕ} (numbers : finset ℕ)

-- Conditions
def chosen_independently (x : ℕ) : Prop := sorry
def move_clockwise_if_chosen (x : ℕ) (chosen : finset ℕ) : Prop := sorry
def end_with_k_different_numbers (final_set : finset ℕ) : Prop := final_set.card = k

-- Part (a)
theorem appearance_equally_likely (x : ℕ) (h_independent : chosen_independently x)
  (h_clockwise : ∀ y ∈ numbers, move_clockwise_if_chosen y numbers) :
  (∃ y ∈ numbers, true) → true :=
by { sorry } -- Conclusion: Yes

-- Part (b)
theorem all_selections_not_equally_likely (samples : list (finset ℕ))
  (h_independent : ∀ x ∈ samples, chosen_independently x)
  (h_clockwise : ∀ y ∈ samples, move_clockwise_if_chosen y samples) :
  ¬ (∀ x y, x ≠ y → samples x = samples y) :=
by { sorry } -- Conclusion: No

end appearance_equally_likely_all_selections_not_equally_likely_l368_368183


namespace perfect_squares_diff_two_consecutive_l368_368747

theorem perfect_squares_diff_two_consecutive (n : ℕ) (h : n = 20000) :
  {a : ℕ | ∃ b : ℕ, b * 2 + 1 < n ∧ a^2 = b * 2 + 1}.card = 71 :=
by
  sorry

end perfect_squares_diff_two_consecutive_l368_368747


namespace percentage_increase_each_year_is_50_l368_368033

-- Definitions based on conditions
def students_passed_three_years_ago : ℕ := 200
def students_passed_this_year : ℕ := 675

-- The prove statement
theorem percentage_increase_each_year_is_50
    (N3 N0 : ℕ)
    (P : ℚ)
    (h1 : N3 = students_passed_three_years_ago)
    (h2 : N0 = students_passed_this_year)
    (h3 : N0 = N3 * (1 + P)^3) :
  P = 0.5 :=
by
  sorry

end percentage_increase_each_year_is_50_l368_368033


namespace ratio_abc_xyz_l368_368405

theorem ratio_abc_xyz
  (a b c x y z : ℝ)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z) 
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a * x + b * y + c * z = 56) :
  (a + b + c) / (x + y + z) = 7 / 8 := 
sorry

end ratio_abc_xyz_l368_368405


namespace PQST_theorem_l368_368398

noncomputable def PQST_product : Prop :=
  ∀ (P Q S T : ℝ), 0 < P ∧ 0 < Q ∧ 0 < S ∧ 0 < T →
  log 10 (P * S) + log 10 (P * T) = 3 →
  log 10 (S * T) + log 10 (S * Q) = 4 →
  log 10 (Q * P) + log 10 (Q * T) = 5 →
  P * Q * S * T = 10000

theorem PQST_theorem : PQST_product :=
by {
  intros P Q S T hpos h1 h2 h3,
  -- The proof is to be filled in here
  sorry
}

end PQST_theorem_l368_368398


namespace pascals_triangle_total_numbers_l368_368303

theorem pascals_triangle_total_numbers (N : ℕ) (hN : N = 29) :
  (∑ n in Finset.range (N + 1), (n + 1)) = 465 :=
by
  rw hN
  calc (∑ n in Finset.range 30, (n + 1))
      = ∑ k in Finset.range 30, (k + 1) : rfl
  -- Here we are calculating the sum of the first 30 terms of the sequence (n + 1)
  ... = 465 : sorry

end pascals_triangle_total_numbers_l368_368303


namespace sum_a_c_eq_13_l368_368078

noncomputable def conditions (a b c d k : ℤ) :=
  d = a * b * c ∧
  1 < a ∧ a < b ∧ b < c ∧
  233 = d * k + 79

theorem sum_a_c_eq_13 (a b c d k : ℤ) (h : conditions a b c d k) : a + c = 13 := by
  sorry

end sum_a_c_eq_13_l368_368078


namespace pascal_triangle_count_30_rows_l368_368281

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l368_368281


namespace product_of_xy_l368_368340

theorem product_of_xy : 
  ∃ (x y : ℝ), 3 * x + 4 * y = 60 ∧ 6 * x - 4 * y = 12 ∧ x * y = 72 :=
by
  sorry

end product_of_xy_l368_368340


namespace unit_vector_orthogonal_to_given_vectors_l368_368673

noncomputable def unit_vector : ℝ × ℝ × ℝ :=
  (1/Real.sqrt 11, -3/Real.sqrt 11, 1/Real.sqrt 11)

theorem unit_vector_orthogonal_to_given_vectors :
  let v := (2, 1, 1)
  let w := (0, 1, 3)
  let x := unit_vector in
  (v.1 * x.1 + v.2 * x.2 + v.3 * x.3 = 0) ∧
  (w.1 * x.1 + w.2 * x.2 + w.3 * x.3 = 0) ∧
  (x.1^2 + x.2^2 + x.3^2 = 1) :=
by
  sorry

end unit_vector_orthogonal_to_given_vectors_l368_368673


namespace find_y_l368_368787

noncomputable def y_value (D C : ℝ × ℝ) (slope_AC slope_CB : ℝ) (is_right_triangle : Prop) (is_30_60_90 : Prop) : ℝ :=
  if D = (13, 0) ∧ C = (8, 4 * Real.sqrt 3) ∧ slope_AC = -Real.sqrt 3 ∧ slope_CB = 1 / Real.sqrt 3 ∧ is_right_triangle ∧ is_30_60_90 then
    5 / Real.sqrt 3
  else
    0

theorem find_y :
  ∀ (D C : ℝ × ℝ) (slope_AC slope_CB : ℝ) (is_right_triangle : Prop) (is_30_60_90 : Prop),
  D = (13, 0) →
  C = (8, 4 * Real.sqrt 3) →
  slope_AC = -Real.sqrt 3 →
  slope_CB = 1 / Real.sqrt 3 →
  is_right_triangle →
  is_30_60_90 →
  y_value D C slope_AC slope_CB is_right_triangle is_30_60_90 = 5 / Real.sqrt 3 :=
by
  intros D C slope_AC slope_CB is_right_triangle is_30_60_90 hD hC hAC hCB hRT h3090
  simp [y_value, hD, hC, hAC, hCB, hRT, h3090]
  sorry

end find_y_l368_368787


namespace range_of_a_for_local_extrema_l368_368493

theorem range_of_a_for_local_extrema (a : ℝ) :
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ g x = a * (x - 1)^2 - Real.log x ∧ g y = a * (y - 1)^2 - Real.log y ∧ g' x = 0 ∧ g' y = 0 ∧ x ≠ y) ↔ a < -2 :=
by sorry

noncomputable def g (x : ℝ) (a : ℝ) := a * (x - 1)^2 - Real.log x

noncomputable def g' (x : ℝ) (a : ℝ) := (2 * a * x ^ 2 - 2 * a * x - 1) / x

#eval range_of_a_for_local_extrema

end range_of_a_for_local_extrema_l368_368493


namespace finite_set_with_equal_distances_l368_368444

theorem finite_set_with_equal_distances :
  ∃ (S : Finset (ℝ × ℝ)), ∀ p ∈ S, (Finset.filter (λ q, dist p q = 1) S).card ≥ 100 :=
sorry

end finite_set_with_equal_distances_l368_368444


namespace value_of_k_l368_368687

theorem value_of_k (k : ℕ) (h : 24 / k = 4) : k = 6 := by
  sorry

end value_of_k_l368_368687


namespace stratified_sampling_l368_368609

-- Definitions
def total_staff : ℕ := 150
def senior_titles : ℕ := 45
def intermediate_titles : ℕ := 90
def clerks : ℕ := 15
def sample_size : ℕ := 10

-- Ratios for stratified sampling
def senior_sample : ℕ := (senior_titles * sample_size) / total_staff
def intermediate_sample : ℕ := (intermediate_titles * sample_size) / total_staff
def clerks_sample : ℕ := (clerks * sample_size) / total_staff

-- Theorem statement
theorem stratified_sampling :
  senior_sample = 3 ∧ intermediate_sample = 6 ∧ clerks_sample = 1 :=
by
  sorry

end stratified_sampling_l368_368609


namespace value_of_f_l368_368416

noncomputable
def f (k l m x : ℝ) : ℝ := k + m / (x - l)

theorem value_of_f (k l m : ℝ) (hk : k = -2) (hl : l = 2.5) (hm : m = 12) :
  f k l m (k + l + m) = -4 / 5 :=
by
  sorry

end value_of_f_l368_368416


namespace intersection_complement_l368_368838

open Set

-- Defining sets A, B and universal set U
def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {x | 1 < x ∧ x ≤ 6}
def U : Set ℕ := A ∪ B

-- Statement of the proof problem
theorem intersection_complement :
  A ∩ (U \ B) = {1, 7} :=
by
  sorry

end intersection_complement_l368_368838


namespace numberOfCorrectRelations_l368_368070

open ProbabilityTheory Set

def eventA : Set (Set String) := {{"HH"}}
def eventB : Set (Set String) := {{"HH", "TT"}}

theorem numberOfCorrectRelations : ∃ n : Nat, n = 1 ∧
  (eventA ⊆ eventB ∧
   ¬ (P(eventA) * P(eventB) = P(eventA ∩ eventB))) :=
by
  exists 1
  sorry

end numberOfCorrectRelations_l368_368070


namespace y_intercept_of_line_k_l368_368076

-- Define the line with slope 4a and passing through the point (a, 0)
def line_k (a : ℝ) : Type :=
  {b : ℝ // ∃ (m : ℝ), m = 4 * a ∧ (0 = m * a + b)}

-- The theorem stating the y-intercept of the line k
theorem y_intercept_of_line_k (a : ℝ) (H : ∃ (b : ℝ), (line_k a)) : ∃ b, b = -4 * a^2 :=
by 
  sorry

end y_intercept_of_line_k_l368_368076


namespace simpsons_paradox_example_l368_368919

theorem simpsons_paradox_example :
  ∃ n1 n2 a1 a2 b1 b2,
    n1 = 10 ∧ a1 = 3 ∧ b1 = 2 ∧
    n2 = 90 ∧ a2 = 45 ∧ b2 = 488 ∧
    ((a1 : ℝ) / n1 > (b1 : ℝ) / n1) ∧
    ((a2 : ℝ) / n2 > (b2 : ℝ) / n2) ∧
    ((a1 + a2 : ℝ) / (n1 + n2) < (b1 + b2 : ℝ) / (n1 + n2)) :=
by
  use 10, 90, 3, 45, 2, 488
  simp
  sorry

end simpsons_paradox_example_l368_368919


namespace total_lives_l368_368094

noncomputable def C : ℝ := 9.5
noncomputable def D : ℝ := C - 3.25
noncomputable def M : ℝ := D + 7.75
noncomputable def E : ℝ := 2 * C - 5.5
noncomputable def F : ℝ := 2/3 * E

theorem total_lives : C + D + M + E + F = 52.25 :=
by
  sorry

end total_lives_l368_368094


namespace value_of_v3_at_2_l368_368991

def f (x : ℝ) : ℝ := x^5 - 2 * x^4 + 3 * x^3 - 7 * x^2 + 6 * x - 3

def v3 (x : ℝ) := (x - 2) * x + 3 
def v3_eval_at_2 : ℝ := (2 - 2) * 2 + 3

theorem value_of_v3_at_2 : v3 2 - 7 = -1 := by
    sorry

end value_of_v3_at_2_l368_368991


namespace square_frame_side_length_l368_368743

/-- Given that there are 3 columns and 2 rows of glass panes, each pane has a height-to-width ratio
    of 3:1, and each pane is surrounded by a consistent 3-inch-wide frame, prove that the side length 
    of the square framing structure is 15 inches. -/
theorem square_frame_side_length :
  ∀ (w : ℝ),
    (3 * w + 12 = 6 * w + 9) →
    (3 * w + 12 = 15) :=
by
  assume w,
  intro h,
  /- skipping the actual proof -/
  sorry

end square_frame_side_length_l368_368743


namespace find_vertical_shift_l368_368132

variable (a b c d : ℝ) -- Declaring variables

-- Given conditions as definitions
def max_value := 5
def min_value := -3
def func (x : ℝ) := a * cos (b * x + c) + d

-- Theorem statement
theorem find_vertical_shift (h_max : ∀ x, func a b c d x ≤ max_value)
                            (h_min : ∀ x, min_value ≤ func a b c d x) :
    d = 1 :=
by
  sorry

end find_vertical_shift_l368_368132


namespace proposition4_l368_368706

-- Definitions for lines, planes and their relationships
variable (Line : Type) (Plane : Type)
variable (a b : Line) (α : Plane)

-- Conditions from the problem
variable (par_plane : a ∥ α)
variable (in_plane : b ⊆ α)
variable (par_line : a ∥ b)

-- Proposition 4 (which we want to show is true)
theorem proposition4 : a ∥ α ∨ a ⊆ α :=
sorry

end proposition4_l368_368706


namespace problem1_problem2_problem3_l368_368649

-- Problem 1
theorem problem1 (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) : 
  (-((3 * b) / (2 * a)) * (6 * a) / (b^3)) = -9 / (b^2) := sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h_a : a ≠ b): 
  (a^2 / (a - b)) - (a + b) = (b^2 / (a - b)) := sorry

-- Problem 3
theorem problem3 (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) : 
  ((a^(-3))^2 * (a * b^2)^(-3)) = 1 / (a^9 * b^6) := sorry

end problem1_problem2_problem3_l368_368649


namespace ratio_AD_DC_l368_368353

-- Definitions of given conditions
variables {A B C D : Type} [Point : Type]

def AB : ℝ := 8
def BC : ℝ := 10
def AC : ℝ := 6
def BD : ℝ := 8

-- The target statement to prove
theorem ratio_AD_DC : (AD / DC) = 0 := 
by
  sorry

end ratio_AD_DC_l368_368353


namespace pascal_triangle_rows_sum_l368_368314

theorem pascal_triangle_rows_sum :
  ∑ k in finset.range 30, (k + 1) = 465 := by
  sorry

end pascal_triangle_rows_sum_l368_368314


namespace sufficient_but_not_necessary_l368_368337

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 0): (x = 1 → x > 0) ∧ ¬(x > 0 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_l368_368337


namespace median_perpendicular_to_segment_l368_368035

noncomputable theory

open Point

structure Square :=
  (A B C D : Point)

def median (B B1 B2 : Point) : Point :=
  let M := midpoint B1 B2
  let BM := vector B M
  BM

def segment (D1 D2 : Point) : vector :=
  vector D1 D2

def is_perpendicular (v1 v2 : vector) : Prop :=
  inner_product v1 v2 = 0

def square1 : Square := ⟨A, B, C, D⟩
def square2 : Square := ⟨A, B1, C1, D1⟩
def square3 : Square := ⟨A2, B2, C, D2⟩

theorem median_perpendicular_to_segment
  (square1 : Square) (square2 : Square) (square3 : Square)
  (h1 : square1.A = square2.A) (h2 : square1.C = square2.C)
  (h3 : square1.A = square3.A) (h4 : square1.C = square3.C) :
  is_perpendicular (median square1.B square2.B1 square3.B2) (segment square2.D1 square3.D2) :=
  sorry

end median_perpendicular_to_segment_l368_368035


namespace probability_one_male_correct_probability_atleast_one_female_correct_l368_368448

def total_students := 5
def female_students := 2
def male_students := 3
def number_of_selections := 2

noncomputable def probability_only_one_male : ℚ :=
  (6 : ℚ) / 10

noncomputable def probability_atleast_one_female : ℚ :=
  (7 : ℚ) / 10

theorem probability_one_male_correct :
  (6 / 10 : ℚ) = 3 / 5 :=
by
  sorry

theorem probability_atleast_one_female_correct :
  (7 / 10 : ℚ) = 7 / 10 :=
by
  sorry

end probability_one_male_correct_probability_atleast_one_female_correct_l368_368448


namespace wage_difference_l368_368569

variable (P Q : ℝ)
variable (h : ℝ)
axiom wage_relation : P = 1.5 * Q
axiom time_relation : 360 = P * h
axiom time_relation_q : 360 = Q * (h + 10)

theorem wage_difference : P - Q = 6 :=
  by
  sorry

end wage_difference_l368_368569


namespace fewer_onions_correct_l368_368642

-- Define the quantities
def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

-- Calculate the total number of tomatoes and corn
def tomatoes_and_corn : ℕ := tomatoes + corn

-- Calculate the number of fewer onions
def fewer_onions : ℕ := tomatoes_and_corn - onions

-- State the theorem and provide the proof
theorem fewer_onions_correct : fewer_onions = 5200 :=
by
  -- The statement is proved directly by the calculations above
  -- Providing the actual proof is not necessary as per the guidelines
  sorry

end fewer_onions_correct_l368_368642


namespace extraordinary_numbers_count_l368_368848

/-- An integer is called "extraordinary" if it has exactly one even divisor other than 2.
We need to count the number of extraordinary numbers in the interval [1, 75]. -/
def is_extraordinary (n : ℕ) : Prop :=
  ∃ p : ℕ, nat.prime p ∧ p % 2 = 1 ∧ n = 2 * p

theorem extraordinary_numbers_count :
  (finset.filter (λ n : ℕ, n ≥ 1 ∧ n ≤ 75 ∧ is_extraordinary n) (finset.range 76)).card = 11 :=
by
  sorry

end extraordinary_numbers_count_l368_368848


namespace probability_A_selected_B_not_selected_l368_368647

def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_A_selected_B_not_selected :
  let totalWays := comb 5 2,
      favorableWays := comb 3 1
  in (favorableWays : ℚ) / totalWays = 3 / 10 :=
by
  let totalWays := comb 5 2
  let favorableWays := comb 3 1
  have totalWays_eq : totalWays = 10 := by sorry
  have favorableWays_eq : favorableWays = 3 := by sorry
  calc
    (favorableWays : ℚ) / totalWays
        = (3 : ℚ) / 10 : by rw [favorableWays_eq, totalWays_eq]
        ... = 3 / 10   : by norm_num

end probability_A_selected_B_not_selected_l368_368647


namespace finite_triples_l368_368873

theorem finite_triples (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  (\frac{1}{a} + \frac{1}{b} + \frac{1}{c} = \frac{1}{1000}) →
  {t : ℕ × ℕ × ℕ | ∃ a b c, (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (\frac{1}{a} + \frac{1}{b} + \frac{1}{c} = \frac{1}{1000})}.finite :=
by
  sorry

end finite_triples_l368_368873


namespace sum_converges_to_two_l368_368150

noncomputable def infinite_sum : ℝ :=
  ∑' k : ℕ, (10^k / ((5^k - 3^k) * (5^(k + 1) - 3^(k + 1))))

theorem sum_converges_to_two : infinite_sum = 2 :=
by
  sorry

end sum_converges_to_two_l368_368150


namespace find_m_l368_368010

-- Definitions
def line1 (m : ℝ) := ∀ x y : ℝ, (m + 3) * x + m * y - 2 = 0
def line2 (m : ℝ) := ∀ x y : ℝ, m * x - 6 * y + 5 = 0
def slope1 (m : ℝ) := -((m + 3) / m)
def slope2 (m : ℝ) := m / 6

-- Perpendicular condition
def perpendicular (m : ℝ) : Prop :=
  slope1 m * slope2 m = -1 ∧ m ≠ 0

theorem find_m (m : ℝ) : perpendicular m ↔ m = 3 := by
  sorry

end find_m_l368_368010


namespace max_x_plus_inv_x_2011_l368_368521

noncomputable def max_x_plus_inv_x (n : ℕ) (a : ℝ) (x : ℝ): ℝ :=
  if h_pos : 1 ≤ n ∧ 0 < a ∧ 0 < x then
    let y_sum := a - x
    let y_inv_sum := a - (1 / x)
    let max_val := (a^2 - (a * (x + (1 / x))) + 1) / a
    if (y_sum * y_inv_sum ≥ (n - 1)^2) then max_val else 0
  else 0

theorem max_x_plus_inv_x_2011 (x : ℝ) (hx : 0 < x) (sum : ℝ) (sum_inv : ℝ) :
  (sum = 2012) →
  (sum_inv = 2012) →
  ∀ y : ℝ, y ∈ (finset.range 2011).erase x →
  x + 1 / x ≤ 8045 / 2012 :=
begin
  intros hsum hsum_inv,
  sorry,
end

end max_x_plus_inv_x_2011_l368_368521


namespace exist_coloring_for_nm_crocodile_l368_368389

-- Define the predicate for a (n, m)-crocodile move
def nm_crocodile_move (n m : ℕ) (x y : ℤ × ℤ) (new_x new_y : ℤ × ℤ) : Prop :=
(new_x = (x.1 + n, x.2) ∧ new_y = (new_x.1, new_x.2 + m)) ∨
(new_x = (x.1 - n, x.2) ∧ new_y = (new_x.1, new_x.2 - m)) ∨
(new_x = (x.1, x.2 + n) ∧ new_y = (new_x.1 + m, new_x.2)) ∨
(new_x = (x.1, x.2 - n) ∧ new_y = (new_x.1 - m, new_x.2))

-- Define the coloring function for the chessboard
def chessboard_coloring (color : ℤ × ℤ → bool) : Prop :=
∀ n m x y new_x new_y, nm_crocodile_move n m x y new_x new_y → (color y ≠ color new_y)

theorem exist_coloring_for_nm_crocodile (n m : ℕ) (hn : 0 < n) (hm : 0 < m) :
  ∃ color : (ℤ × ℤ) → bool, chessboard_coloring color :=
sorry

end exist_coloring_for_nm_crocodile_l368_368389


namespace age_difference_l368_368940

variable (A B C : ℕ)

def age_relationship (B C : ℕ) : Prop :=
  B = 2 * C

def total_ages (A B C : ℕ) : Prop :=
  A + B + C = 72

theorem age_difference (B : ℕ) (hB : B = 28) (h1 : age_relationship B C) (h2 : total_ages A B C) :
  A - B = 2 :=
sorry

end age_difference_l368_368940


namespace monotonic_decreasing_intervals_l368_368501

def is_monotonically_decreasing_in (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ > f x₂

theorem monotonic_decreasing_intervals : 
  ∀ x, x ≠ 1 → y = (1 / (x - 1)) → 
  is_monotonically_decreasing_in y { x | x < 1 } ∧ is_monotonically_decreasing_in y { x | x > 1 } :=
sorry

end monotonic_decreasing_intervals_l368_368501


namespace closest_integer_to_cube_root_of_sums_l368_368543

theorem closest_integer_to_cube_root_of_sums :
  ∃ (n : ℤ), n = 10 ∧ abs (n - Int.ofReal (Real.cbrt (7^3 + 9^3))) < 1 :=
begin
  sorry
end

end closest_integer_to_cube_root_of_sums_l368_368543


namespace length_of_BD_l368_368433

theorem length_of_BD (a : ℝ) (ha : a ≥ sqrt 7) : 
  let AC := a,
      BC := 3,
      AD := 4 in
  let AB := sqrt (AC^2 + BC^2),
      BD := sqrt (AB^2 - AD^2) in
  BD = sqrt (a^2 - 7) := by
  sorry

end length_of_BD_l368_368433


namespace total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368305

-- Define the number of elements in the nth row of Pascal's Triangle
def num_elements_in_row (n : ℕ) : ℕ := n + 1

-- Define the sum of numbers from 0 to n, inclusive
def sum_of_first_n_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the total number of elements in the first 30 rows (0th to 29th)
def total_elements_in_first_30_rows : ℕ := sum_of_first_n_numbers 30

-- The main statement to prove
theorem total_numbers_in_first_30_rows_of_Pascals_Triangle :
  total_elements_in_first_30_rows = 465 :=
by
  simp [total_elements_in_first_30_rows, sum_of_first_n_numbers]
  sorry

end total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368305


namespace original_price_l368_368128

theorem original_price (x : ℝ) (h : 0.9504 * x = 108) : x = 10800 / 9504 :=
by
  sorry

end original_price_l368_368128


namespace area_of_triangle_l368_368833

-- Definitions and conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def focal_distance (a b c : ℝ) : Prop := c = real.sqrt (a^2 - b^2)
def in_ratio (d1 d2 : ℝ) : Prop := d1 = 2 * d2
def point_P_on_ellipse (x y : ℝ) := ellipse x y
def distance_sum (PF1 PF2 : ℝ) (a : ℝ) : Prop := PF1 + PF2 = 2 * a

-- Given parameters
def a := 3
def b := 2
def c := real.sqrt (a^2 - b^2)
def F1 := (-c, 0)
def F2 := (c, 0)

-- Main statement to prove
theorem area_of_triangle
    (x y : ℝ)
    (H1 : ellipse x y)
    (PF1 PF2 : ℝ)
    (H2 : in_ratio PF1 PF2)
    (H3 : distance_sum PF1 PF2 a) :
  let base := 4 -- PF1
  let height := 2 -- PF2
  in (1 / 2) * base * height = 4 := by
  -- You can include the definitions here if necessary for clarity but not the proof itself.
  sorry

end area_of_triangle_l368_368833


namespace find_n_l368_368757

-- Define the nth triangular number
def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Define the product of 16 consecutive triangular numbers
def product_triangular_numbers (n : ℕ) : ℕ := 
  List.prod (List.map triangular_number (List.range (n + 16)).drop n)

-- Define a predicate to check if a number is a perfect square
def is_perfect_square (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

-- The main theorem to be proved
theorem find_n (n : ℕ) :
  is_perfect_square (product_triangular_numbers n) ↔ n = 2 ∨ n = 9 := 
sorry

end find_n_l368_368757


namespace ordered_pairs_count_l368_368202

theorem ordered_pairs_count :
  (∑ a in finset.range 50, (∑ b in finset.filter (λ b, b % 2 = 0) (finset.range (a + 1)), 
  if (∃ r s : ℤ, r + s = -a ∧ r * s = b) then 1 else 0)) = 676 :=
by
  sorry

end ordered_pairs_count_l368_368202


namespace smallest_y_l368_368759

theorem smallest_y (y : ℕ) : 
    (y % 5 = 4) ∧ 
    (y % 7 = 6) ∧ 
    (y % 8 = 7) → 
    y = 279 :=
sorry

end smallest_y_l368_368759


namespace unique_solution_l368_368192

open Classical

noncomputable def pos_real := { x : ℝ // 0 < x }

def property (f : pos_real → pos_real) : Prop := 
  ∀ (x y : pos_real), f ⟨x.val^2 + x.val * f y, sorry⟩ = x.val * f ⟨x.val + y.val, sorry⟩

theorem unique_solution (f : pos_real → pos_real) (h : property f) : 
  ∀ (y : pos_real), f y = y := 
sorry

end unique_solution_l368_368192


namespace pascals_triangle_total_numbers_l368_368296

theorem pascals_triangle_total_numbers (N : ℕ) (hN : N = 29) :
  (∑ n in Finset.range (N + 1), (n + 1)) = 465 :=
by
  rw hN
  calc (∑ n in Finset.range 30, (n + 1))
      = ∑ k in Finset.range 30, (k + 1) : rfl
  -- Here we are calculating the sum of the first 30 terms of the sequence (n + 1)
  ... = 465 : sorry

end pascals_triangle_total_numbers_l368_368296


namespace find_x_l368_368816

theorem find_x (p : ℕ) (hprime : Nat.Prime p) (hgt5 : p > 5) (x : ℕ) (hx : x ≠ 0) :
    (∀ n : ℕ, 0 < n → (5 * p + x) ∣ (5 * p ^ n + x ^ n)) ↔ x = p := by
  sorry

end find_x_l368_368816


namespace number_of_divisors_of_27n3_l368_368409

theorem number_of_divisors_of_27n3 
  (n : ℤ) (h_odd : ¬ even n) (h_divisors : ∃ d, 0 < d ∧ d = 12) :
  nat.factors 27 ∣ n^3 → nat.totient (27 * n^3) = 256 :=
sorry

end number_of_divisors_of_27n3_l368_368409


namespace retirement_percentage_l368_368800

-- Define the conditions
def gross_pay : ℝ := 1120
def tax_deduction : ℝ := 100
def net_paycheck : ℝ := 740

-- Define the total deduction
def total_deduction : ℝ := gross_pay - net_paycheck
def retirement_deduction : ℝ := total_deduction - tax_deduction

-- Define the theorem to prove
theorem retirement_percentage :
  (retirement_deduction / gross_pay) * 100 = 25 :=
by
  sorry

end retirement_percentage_l368_368800


namespace find_values_of_a_and_b_l368_368087

theorem find_values_of_a_and_b (a b x y : ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < x) (h4: 0 < y) 
  (h5 : a + b = 10) (h6 : a / x + b / y = 1) (h7 : x + y = 18) : 
  (a = 2 ∧ b = 8) ∨ (a = 8 ∧ b = 2) := 
sorry

end find_values_of_a_and_b_l368_368087


namespace vector_x_solution_l368_368773

theorem vector_x_solution (x : ℝ) (a b c : ℝ × ℝ)
  (ha : a = (-2,0))
  (hb : b = (2,1))
  (hc : c = (x,1))
  (collinear : ∃ k : ℝ, 3 • a + b = k • c) :
  x = -4 :=
by
  sorry

end vector_x_solution_l368_368773


namespace sum_of_constants_l368_368914

-- Given the conditions
def radii : (ℕ × ℕ × ℕ) := (3, 4, 5)

-- Define the area of the equilateral triangle as per the problem
noncomputable def largest_possible_area : ℚ := 9 + 25 / 4 * Real.sqrt 3

-- Main statement to verify
theorem sum_of_constants :
  (let (a, b, c, d) := (9, 25, 4, 3) in a + b + c + d = 41) :=
sorry

end sum_of_constants_l368_368914


namespace fewer_onions_grown_l368_368636

def num_tomatoes := 2073
def num_cobs_of_corn := 4112
def num_onions := 985

theorem fewer_onions_grown : num_tomatoes + num_cobs_of_corn - num_onions = 5200 := by
  sorry

end fewer_onions_grown_l368_368636


namespace boys_in_parkway_l368_368788

theorem boys_in_parkway (total_students : ℕ) (students_playing_soccer : ℕ) (percentage_boys_playing_soccer : ℝ)
                        (girls_not_playing_soccer : ℕ) :
                        total_students = 420 ∧ students_playing_soccer = 250 ∧ percentage_boys_playing_soccer = 0.86 
                        ∧ girls_not_playing_soccer = 73 → 
                        ∃ total_boys : ℕ, total_boys = 312 :=
by
  -- Proof omitted
  sorry

end boys_in_parkway_l368_368788


namespace max_profit_at_100_l368_368475

noncomputable def C (x : ℝ) : ℝ :=
if 0 < x ∧ x < 40 then 10 * x^2 + 100 * x
else if x ≥ 40 then 501 * x + (10000 / x) - 4500
else 0

noncomputable def L (x : ℝ) : ℝ :=
if 0 < x ∧ x < 40 then -10 * x^2 + 400 * x - 2500
else if x ≥ 40 then 2000 - (x + 10000 / x)
else 0

theorem max_profit_at_100 : 
  (∀ x, (0 < x ∧ x < 40 → L(x) = -10 * x^2 + 400 * x - 2500) 
       ∧ (x ≥ 40 → L(x) = 2000 - (x + 10000 / x))) ∧ 
  (∀ x, L(x) ≤ 1800) ∧ 
  (L(100) = 1800) :=
by
  sorry

end max_profit_at_100_l368_368475


namespace solutionTriangle_l368_368776

noncomputable def solveTriangle (a b : ℝ) (B : ℝ) : (ℝ × ℝ × ℝ) :=
  let A := 30
  let C := 30
  let c := 2
  (A, C, c)

theorem solutionTriangle :
  solveTriangle 2 (2 * Real.sqrt 3) 120 = (30, 30, 2) :=
by
  sorry

end solutionTriangle_l368_368776


namespace term_sequence_l368_368578

theorem term_sequence (n : ℕ) (h : (-1:ℤ) ^ (n + 1) * n * (n + 1) = -20) : n = 4 :=
sorry

end term_sequence_l368_368578


namespace total_number_of_subsets_of_P_l368_368396

-- Given data
def M := {0, 1, 2, 3, 4}
def N := {0, 2, 4}
def P := M ∩ N

-- To Prove
theorem total_number_of_subsets_of_P : P.subsets.card = 8 :=
by
  sorry

end total_number_of_subsets_of_P_l368_368396


namespace remaining_wire_length_l368_368974

theorem remaining_wire_length (total_length : ℝ) (fraction_cut : ℝ) (remaining_length : ℝ) (h1 : total_length = 3) (h2 : fraction_cut = 1 / 3) (h3 : remaining_length = 2) :
  total_length * (1 - fraction_cut) = remaining_length :=
by
  -- Proof goes here
  sorry

end remaining_wire_length_l368_368974


namespace domain_of_function_l368_368199

theorem domain_of_function : 
  { x : ℝ | (x - 2 ≥ 0) ∧ (3 - x > 0) ∧ (ln (3 - x) ≠ 0) } = { x : ℝ | 2 < x ∧ x < 3 } := 
by
  sorry

end domain_of_function_l368_368199


namespace interest_years_l368_368972

theorem interest_years (P : ℝ) (R : ℝ) (N : ℝ) (H1 : P = 2400) (H2 : (P * (R + 1) * N) / 100 - (P * R * N) / 100 = 72) : N = 3 :=
by
  -- Proof can be filled in here
  sorry

end interest_years_l368_368972


namespace complex_pure_imaginary_l368_368346

theorem complex_pure_imaginary (m : ℝ) (i : ℝ) (H_imaginary : i^2 = -1) :
  (m^2 + complex.I) * (1 + m * complex.I) = complex.mk 0 0 → m = 0 ∨ m = 1 :=
by
  sorry

end complex_pure_imaginary_l368_368346


namespace number_of_elements_M_inter_N_l368_368212

def M : set ℝ := {y | ∃ x : ℝ, y = x + 1}
def N : set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 1}

theorem number_of_elements_M_inter_N : (M ∩ N).card = 0 := by
  sorry

end number_of_elements_M_inter_N_l368_368212


namespace max_number_of_squares_l368_368043

theorem max_number_of_squares (points : finset (fin 12)) : 
  ∃ (sq_count : ℕ), sq_count = 11 := 
sorry

end max_number_of_squares_l368_368043


namespace c_value_l368_368335

theorem c_value (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 200 * x + c = (x + a)^2) → c = 10000 := 
by
  intro h
  sorry

end c_value_l368_368335


namespace number_of_single_pieces_is_100_l368_368425

def cost_per_circle : ℝ := 0.01
def total_earned : ℝ := 10
def double_pieces : ℕ := 45
def triple_pieces : ℕ := 50
def quadruple_pieces : ℕ := 165

noncomputable def total_earned_from_double_pieces : ℝ :=
  double_pieces * 2 * cost_per_circle

noncomputable def total_earned_from_triple_pieces : ℝ :=
  triple_pieces * 3 * cost_per_circle

noncomputable def total_earned_from_quadruple_pieces : ℝ :=
  quadruple_pieces * 4 * cost_per_circle

noncomputable def total_earned_from_larger_pieces : ℝ :=
  total_earned_from_double_pieces + total_earned_from_triple_pieces + total_earned_from_quadruple_pieces

noncomputable def total_earned_from_single_pieces : ℝ :=
  total_earned - total_earned_from_larger_pieces

noncomputable def number_of_single_pieces_sold : ℕ :=
  (total_earned_from_single_pieces / cost_per_circle : ℝ).to_nat

theorem number_of_single_pieces_is_100 : number_of_single_pieces_sold = 100 := by
  sorry

end number_of_single_pieces_is_100_l368_368425


namespace special_9_digit_integer_divisible_by_101_l368_368825

def is_special_9_digit_integer (W : ℕ) : Prop :=
  ∃ x y z : ℕ, x > 0 ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
  W = 1001001 * (100 * x + 10 * y + z)

theorem special_9_digit_integer_divisible_by_101 (W : ℕ) :
  is_special_9_digit_integer W → 101 ∣ W :=
by
  intro h,
  obtain ⟨x, y, z, hx, hy, hz, HW⟩ := h,
  rw HW,
  use 9901 * (100 * x + 10 * y + z),
  exact mul_assoc 1001001 9901 (100 * x + 10 * y + z)

end special_9_digit_integer_divisible_by_101_l368_368825


namespace pascal_triangle_rows_sum_l368_368313

theorem pascal_triangle_rows_sum :
  ∑ k in finset.range 30, (k + 1) = 465 := by
  sorry

end pascal_triangle_rows_sum_l368_368313


namespace relationship_among_abc_l368_368230

def a : ℝ := Real.sqrt 2
def b : ℝ := Real.log 2
def c : ℝ := Real.logb (1/3) 2

theorem relationship_among_abc :
  a > b ∧ b > c :=
by
  sorry

end relationship_among_abc_l368_368230


namespace cubic_product_of_roots_l368_368203

theorem cubic_product_of_roots : 
    let u := real.cbrt 4 in
    let x := 3 * u in
    let poly := polynomial.X ^ 3 - 108 in
    polynomial.product_roots poly = -108 :=
begin
    let u := real.cbrt 4,
    let x := 3 * u,
    let poly := polynomial.X ^ 3 - 108,
    sorry
end

end cubic_product_of_roots_l368_368203


namespace arrangement_plans_l368_368526

theorem arrangement_plans (classes students: ℕ) (selectClasses: ℕ): 
  classes = 6 → students = 4 → selectClasses = 2 →
  (nat.choose classes selectClasses) * (nat.fact students / (nat.fact selectClasses * nat.fact (students - selectClasses))) = 90 :=
by
  intros h_classes h_students h_selectClasses
  rw [h_classes, h_students, h_selectClasses]
  norm_num
  sorry

end arrangement_plans_l368_368526


namespace largest_digit_M_divisible_by_6_l368_368546

theorem largest_digit_M_divisible_by_6 (M : ℕ) (h1 : 5172 * 10 + M % 2 = 0) (h2 : (5 + 1 + 7 + 2 + M) % 3 = 0) : M = 6 := by
  sorry

end largest_digit_M_divisible_by_6_l368_368546


namespace find_angle_l368_368915

noncomputable def radii := [4, 3, 2]

def total_area : ℝ := (radii.map (λ r, real.pi * r ^ 2)).sum

def ratio_shaded_unshaded (S U : ℝ) : Prop := S = (3 / 4) * U

def areas_of_circles := [real.pi * 4 ^ 2, real.pi * 3 ^ 2, real.pi * 2 ^ 2]

def shaded_area (theta : ℝ) : ℝ := 16 * theta + 9 * real.pi - 9 * theta + 4 * theta

theorem find_angle
  (S : ℝ)
  (U : ℝ)
  (ratio : ratio_shaded_unshaded S U)
  (total : total_area = S + U)
  (shaded : shaded_area theta = S) :
  theta = (12 * real.pi) / 77 :=
sorry

end find_angle_l368_368915


namespace total_handshakes_l368_368523

variable (r : ℕ) (c : ℕ)
variable (handshakes : ℕ)

axiom reps_per_company : r = 5
axiom num_companies : c = 3
axiom total_participants : r * c = 15
axiom shake_pattern : handshakes = (total_participants * (total_participants - 1 - (r - 1)) / 2)

theorem total_handshakes : handshakes = 75 := 
by 
  sorry

end total_handshakes_l368_368523


namespace right_triangle_segment_count_l368_368446

/-- 
Prove that the number of line segments with integer length that can be drawn 
from vertex E to a point on the hypotenuse DF in a right triangle DEF with 
DE = 24 and EF = 25 is 9.
-/
theorem right_triangle_segment_count (DE EF DF : ℝ) (hDE : DE = 24) (hEF : EF = 25) :
  ∃ n : ℕ, n = 9 :=
by
  have h1 : DF = Real.sqrt (DE^2 + EF^2), from sorry,
  have h2 : DF ≈ 34.66, from sorry,
  have EP ≈ 17.33, from sorry,
  use 9,
  sorry

end right_triangle_segment_count_l368_368446


namespace find_second_number_l368_368026

theorem find_second_number (x y z : ℝ) 
  (h1 : x + y + z = 120) 
  (h2 : x = (3/4) * y) 
  (h3 : z = (9/7) * y) 
  : y = 40 :=
sorry

end find_second_number_l368_368026


namespace find_positive_k_l368_368165

noncomputable def cubic_roots (a b k : ℝ) : Prop :=
  (3 * a * a * a + 9 * a * a - 135 * a + k = 0) ∧
  (a * a * b = -45 / 2)

theorem find_positive_k :
  ∃ (a b : ℝ), ∃ (k : ℝ) (pos : k > 0), (cubic_roots a b k) ∧ (k = 525) :=
by
  sorry

end find_positive_k_l368_368165


namespace equilateral_triangle_MNQ_l368_368370

-- Definitions and conditions
variables (A B C M N Q : Type) [acute_triangle ABC]
variables (AM_altitude : segment AM ⊥ line BC)
variables (CN_altitude : segment CN ⊥ line AB)
variables (Q_midpoint : midpoint Q A C)
variables (angle_B : ∠ B = 60)

-- Hypothesis and goal
theorem equilateral_triangle_MNQ :
  is_equilateral_triangle (triangle MNQ) :=
begin
  sorry
end

end equilateral_triangle_MNQ_l368_368370


namespace prob_of_drawing_one_red_ball_distribution_of_X_l368_368952

-- Definitions for conditions
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def total_balls : ℕ := red_balls + white_balls
def balls_drawn : ℕ := 3

-- Combinations 
noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Probabilities
noncomputable def prob_ex_one_red_ball : ℚ :=
  (combination red_balls 1 * combination white_balls 2) / combination total_balls balls_drawn

noncomputable def prob_X_0 : ℚ := (combination white_balls 3) / combination total_balls balls_drawn
noncomputable def prob_X_1 : ℚ := prob_ex_one_red_ball
noncomputable def prob_X_2 : ℚ := (combination red_balls 2 * combination white_balls 1) / combination total_balls balls_drawn

-- Theorem statements
theorem prob_of_drawing_one_red_ball : prob_ex_one_red_ball = 3/5 := by
  sorry

theorem distribution_of_X : prob_X_0 = 1/10 ∧ prob_X_1 = 3/5 ∧ prob_X_2 = 3/10 := by
  sorry

end prob_of_drawing_one_red_ball_distribution_of_X_l368_368952


namespace smallest_even_five_digit_tens_digit_l368_368483

theorem smallest_even_five_digit_tens_digit (digits : set ℕ) (h : digits = {1, 3, 5, 6, 8}) :
  ∃ n : ℕ, even n ∧ n < 100000 ∧ n > 9999 ∧ (∀ m : ℕ, m < 100000 ∧ m > 9999 ∧ even m → n ≤ m)
  ∧ (n / 10 % 10 = 8) :=
by
  sorry

end smallest_even_five_digit_tens_digit_l368_368483


namespace D_lies_on_AC_l368_368829

open EuclideanGeometry

variables {k : Circle} {O A B C D : Point}

-- Given Conditions
axiom h1 : k.center = O
axiom h2 : k.contains A
axiom h3 : k.contains B
axiom h4 : k.contains C
axiom h5 : ∠ A B C > π / 2
axiom h6 : ∠ A O B is_bisected_by (Line O D)
axiom h7 : D lies_on (circumcircle (Triangle B O C))

-- Goal
theorem D_lies_on_AC : lies_on D (Line A C) :=
by
  sorry

end D_lies_on_AC_l368_368829


namespace find_c_in_triangle_l368_368775

theorem find_c_in_triangle (a b c : ℝ) (A : ℝ) (ha : a = sqrt 5) (hb : b = sqrt 15) (hA : A = π / 6) :
  c = 2 * sqrt 5 :=
sorry

end find_c_in_triangle_l368_368775


namespace correct_number_of_propositions_l368_368482

def proposition1 : Prop := 
  ∀ (A B : Type) (f : A → B), 
  ∃ (b : B), ¬∃ (a : A), f a = b

def proposition2 (f : ℝ → ℝ) (t : ℝ) : Prop := 
  ∃! (x : ℝ), f x = t

def proposition3 (f : ℝ → ℝ) : Prop := 
  (∀ x y, f (x + y) = f x + f y) → (∀ x, f (-x) = -f x)

def proposition4 (f : ℝ → ℝ) : Prop := 
  (∀ x, 0 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 1) → (∀ x, -1 ≤ x ∧ x ≤ 1)

def number_of_correct_propositions : Nat := 2

theorem correct_number_of_propositions : 
  ((proposition1 ∧ ¬ proposition2) ∧ (proposition3 ∧ ¬ proposition4)) → number_of_correct_propositions = 2 := 
begin
  sorry
end

end correct_number_of_propositions_l368_368482


namespace pascal_triangle_sum_first_30_rows_l368_368291

theorem pascal_triangle_sum_first_30_rows :
  (Finset.range 30).sum (λ n, n + 1) = 465 :=
begin
  sorry
end

end pascal_triangle_sum_first_30_rows_l368_368291


namespace cat_food_more_than_dog_food_l368_368994

theorem cat_food_more_than_dog_food :
  let cat_food_packs := 6
  let cans_per_cat_pack := 9
  let dog_food_packs := 2
  let cans_per_dog_pack := 3
  let total_cat_food_cans := cat_food_packs * cans_per_cat_pack
  let total_dog_food_cans := dog_food_packs * cans_per_dog_pack
  total_cat_food_cans - total_dog_food_cans = 48 :=
by
  sorry

end cat_food_more_than_dog_food_l368_368994


namespace cost_price_computer_table_l368_368505

variable (C : ℝ) -- Cost price of the computer table
variable (S : ℝ) -- Selling price of the computer table

-- Conditions based on the problem
axiom h1 : S = 1.10 * C
axiom h2 : S = 8800

-- The theorem to be proven
theorem cost_price_computer_table : C = 8000 :=
by
  -- Proof will go here
  sorry

end cost_price_computer_table_l368_368505


namespace division_value_of_712_5_by_12_5_is_57_l368_368075

theorem division_value_of_712_5_by_12_5_is_57 : 712.5 / 12.5 = 57 :=
  by
    sorry

end division_value_of_712_5_by_12_5_is_57_l368_368075


namespace counterexample_to_statement_l368_368659

theorem counterexample_to_statement (n : ℕ) (h1 : n ∈ {5, 7, 8, 15, 26}) (h2 : n % 3 ≠ 0) : 
  ¬ Prime (n^2 - 1) := 
by
  -- The actual proof steps would follow here, but we will omit them with sorry
  sorry

end counterexample_to_statement_l368_368659


namespace area_ABCD_l368_368575

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def intersection (A C M N : Point) : Point := sorry
noncomputable def area (A B C : Point) : ℝ := sorry

variables {A B C D M N O : Point}

-- The conditions
axiom M_midpoint : midpoint B C = M
axiom N_midpoint : midpoint A D = N
axiom O_intersection : intersection A C M N = O
axiom MO_eq_ON : dist M O = dist O N
axiom area_ABC : area A B C = 2019

-- The theorem to prove
theorem area_ABCD : area A B C + area A D C = 4038 := 
sorry

end area_ABCD_l368_368575


namespace appearance_equally_likely_selections_not_equally_likely_l368_368177

-- Define the conditions under which the numbers are chosen
def independent_choice (n k : ℕ) (selection : ℕ → Set ℕ) : Prop :=
∀ i j : ℕ, i ≠ j → selection i ∩ selection j = ∅

def move_clockwise (chosen : Set ℕ) (n : ℕ) : ℕ → ℕ := sorry  -- Placeholder for clockwise movement function

-- Condition: Each subsequent number is chosen independently of the previous ones
def conditions (n k : ℕ) : Prop :=
∃ selection : ℕ → Set ℕ, independent_choice n k selection

-- Problem a: Prove that the appearance of each specific number is equally likely
theorem appearance_equally_likely (n k : ℕ) (h : conditions n k) : 
  (∀ num : ℕ, 1 ≤ num ∧ num ≤ n → appearance_probability num k n) := sorry

-- Problem b: Prove that the appearance of all selections is not equally likely
theorem selections_not_equally_likely (n k : ℕ) (h : conditions n k) : 
  ¬(∀ selection : Set ℕ, appearance_probability selection k n) := sorry

end appearance_equally_likely_selections_not_equally_likely_l368_368177


namespace rational_number_definition_l368_368935

theorem rational_number_definition :
  ∃ (rational : Type) (integer : Type) (fraction : Type),
    (∀ x : rational, x ∈ integer ∨ x ∈ fraction) ∧
    (∀ x : integer, ∃ a : ℤ, x = a) ∧
    (∀ x : fraction, ∃ a b : ℤ, b ≠ 0 ∧ x = (a / b : ℚ)) :=
by
  simp
  sorry

end rational_number_definition_l368_368935


namespace find_a_b_c_eq_32_l368_368686

variables {a b c : ℤ}

theorem find_a_b_c_eq_32
  (h1 : ∃ a b : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b))
  (h2 : ∃ b c : ℤ, x^2 - 21 * x + 108 = (x - b) * (x - c)) :
  a + b + c = 32 :=
sorry

end find_a_b_c_eq_32_l368_368686


namespace seq_formula_l368_368734

-- Define the sequence according to given conditions
def a : ℕ → ℚ
| 0     := 1
| (n+1) := (1 / 2) * a n + 1

-- Prove the conjectured formula
theorem seq_formula (n : ℕ) : a (n - 1) = (2^n - 1) / 2^(n - 1) :=
sorry

end seq_formula_l368_368734


namespace max_elements_in_set_A_l368_368390

open Nat

def is_not_prime (n : ℕ) : Prop :=
  ¬ prime n

def valid_subset (A : set ℕ) : Prop :=
  (∀ a ∈ A, a > 0 ∧ a <= 2020) ∧
  (∀ a b ∈ A, a ≠ b → is_not_prime (abs (a - b)))

theorem max_elements_in_set_A :
  ∃ (A : set ℕ), valid_subset A ∧ finset.card (set.to_finset A) = 505 :=
sorry

end max_elements_in_set_A_l368_368390


namespace number_of_boys_l368_368572

variables (x g : ℕ)

theorem number_of_boys (h1 : x + g = 1150) (h2 : g = (x * 1150) / 100) : x = 92 :=
by
  sorry

end number_of_boys_l368_368572


namespace a4_l368_368704

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Define the sum of the first n terms
def Sn : ℕ → ℝ :=
  λ n, S n

-- Condition 1: For n ≥ 2, Sn, S(n-1), and S(n+1) form an arithmetic sequence
axiom arith_seq (n : ℕ) (h : n ≥ 2) : 2 * (Sn S (n - 1)) = Sn S n + Sn S (n + 1)

-- Condition 2: a_2 = -2
axiom a2 : a 2 = -2

-- Our main goal: Prove that a_4 = -8
theorem a4 : a 4 = -8 :=
sorry

end a4_l368_368704


namespace total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368309

-- Define the number of elements in the nth row of Pascal's Triangle
def num_elements_in_row (n : ℕ) : ℕ := n + 1

-- Define the sum of numbers from 0 to n, inclusive
def sum_of_first_n_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the total number of elements in the first 30 rows (0th to 29th)
def total_elements_in_first_30_rows : ℕ := sum_of_first_n_numbers 30

-- The main statement to prove
theorem total_numbers_in_first_30_rows_of_Pascals_Triangle :
  total_elements_in_first_30_rows = 465 :=
by
  simp [total_elements_in_first_30_rows, sum_of_first_n_numbers]
  sorry

end total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368309


namespace appearance_equally_likely_selections_not_equally_likely_l368_368180

-- Define the conditions under which the numbers are chosen
def independent_choice (n k : ℕ) (selection : ℕ → Set ℕ) : Prop :=
∀ i j : ℕ, i ≠ j → selection i ∩ selection j = ∅

def move_clockwise (chosen : Set ℕ) (n : ℕ) : ℕ → ℕ := sorry  -- Placeholder for clockwise movement function

-- Condition: Each subsequent number is chosen independently of the previous ones
def conditions (n k : ℕ) : Prop :=
∃ selection : ℕ → Set ℕ, independent_choice n k selection

-- Problem a: Prove that the appearance of each specific number is equally likely
theorem appearance_equally_likely (n k : ℕ) (h : conditions n k) : 
  (∀ num : ℕ, 1 ≤ num ∧ num ≤ n → appearance_probability num k n) := sorry

-- Problem b: Prove that the appearance of all selections is not equally likely
theorem selections_not_equally_likely (n k : ℕ) (h : conditions n k) : 
  ¬(∀ selection : Set ℕ, appearance_probability selection k n) := sorry

end appearance_equally_likely_selections_not_equally_likely_l368_368180


namespace Nathan_ate_gumballs_l368_368744

theorem Nathan_ate_gumballs :
  ∀ (boxes : ℕ) (gumballs_per_box : ℕ), boxes = 4 → gumballs_per_box = 5 → boxes * gumballs_per_box = 20 :=
by {
  intros boxes gumballs_per_box h1 h2,
  rw [h1, h2],
  norm_num,
  sorry
}

end Nathan_ate_gumballs_l368_368744


namespace initial_cell_count_l368_368527

theorem initial_cell_count (f : ℕ → ℕ) (h₁ : ∀ n, f (n + 1) = 2 * (f n - 2)) (h₂ : f 5 = 164) : f 0 = 9 :=
sorry

end initial_cell_count_l368_368527


namespace angle_AHB_l368_368632

-- Define the conditions and the problem statement.
theorem angle_AHB 
  (A B C H D E : Point)
  (h₁ : Line[AD] ⊥ Line[BC]) 
  (h₂ : Line[BE] ⊥ Line[AC])
  (h₃ : ∠BAC = 60) 
  (h₄ : ∠ABC = 55) 
  (h₅ : Line[AD] ∩ Line[BE] = H) :
  ∠AHB = 117 :=
begin
  sorry
end

end angle_AHB_l368_368632


namespace square_plot_area_l368_368560

theorem square_plot_area (cost_per_foot total_cost : ℕ) (hcost_per_foot : cost_per_foot = 60) (htotal_cost : total_cost = 4080) :
  ∃ (A : ℕ), A = 289 :=
by
  have h : 4 * 60 * 17 = 4080 := by rfl
  have s : 17 = 4080 / (4 * 60) := by sorry
  use 17 ^ 2
  have hsquare : 17 ^ 2 = 289 := by rfl
  exact hsquare

end square_plot_area_l368_368560


namespace calculate_expression_l368_368990

theorem calculate_expression : 16^4 * 8^2 / 4^12 = (1 : ℚ) / 4 := by
  sorry

end calculate_expression_l368_368990


namespace mean_and_variance_of_transformed_data_l368_368725

open Real

theorem mean_and_variance_of_transformed_data 
  (x : Fin 5 → ℝ) 
  (mean_x : (∑ i, x i) / 5 = 2) 
  (var_x : (∑ i, (x i - (∑ j, x j) / 5) ^ 2) / 5 = 1 / 3) : 
  let y := λ i, (3 * x i - 2) in 
  (∑ i, y i) / 5 = 4 ∧ (∑ i, (y i - (∑ j, y j) / 5) ^ 2) / 5 = 4 :=
by
  sorry

end mean_and_variance_of_transformed_data_l368_368725


namespace card_profit_l368_368024

def purchasing_price : ℤ := 21 -- price in fen
def total_sale_amount : ℕ := 1457 -- total revenue in fen
def num_cards_sold : ℕ := total_sale_amount / 31 -- 1457 / 31 = 47 cards
def selling_price_per_card : ℤ := 31 -- price in fen
def twice_purchasing_price : ℤ := 2 * purchasing_price

theorem card_profit 
  (purchasing_price : ℤ) (total_sale_amount : ℕ) (num_cards_sold : ℕ) 
  (selling_price_per_card : ℤ) (twice_purchasing_price : ℤ) :
  selling_price_per_card <= twice_purchasing_price →
  total_sale_amount = selling_price_per_card * num_cards_sold →
  purchasing_price * num_cards_sold + 470 = 4.7 * 100 :=
sorry

end card_profit_l368_368024


namespace find_radius_original_bubble_l368_368618

-- Define the given radius of the hemisphere.
def radius_hemisphere : ℝ := 4 * real.cbrt 2

-- Define the volume of a hemisphere.
def vol_hemisphere (r : ℝ) := (2 / 3) * π * r^3

-- Define the volume of a sphere.
def vol_sphere (R : ℝ) := (4 / 3) * π * R^3

-- Given condition: The volume of the hemisphere is equal to the volume of the original bubble.
def volume_equivalence (r R : ℝ) := vol_sphere R = vol_hemisphere r

-- State the theorem to prove: Given the radius of the hemisphere, find the radius of the original bubble.
theorem find_radius_original_bubble (R : ℝ) (h : volume_equivalence radius_hemisphere R) : R = 4 :=
sorry

end find_radius_original_bubble_l368_368618


namespace initial_investment_calc_l368_368764

-- Define the function to calculate doubling time
def doubling_time (rate : ℝ) : ℝ := 50 / rate

-- Example conditions given
def interest_rate := 12
def time_period := 12
def final_amount := 20000

-- Lean 4 proof statement
theorem initial_investment_calc : 
  ∃ (initial_investment : ℝ), 
    let num_doubles := time_period / doubling_time interest_rate in
    let correct_initial_investment := final_amount / (2 ^ (num_doubles.to_int)) in
    initial_investment = correct_initial_investment :=
sorry

end initial_investment_calc_l368_368764


namespace sum_max_min_ratio_ellipse_l368_368657

theorem sum_max_min_ratio_ellipse :
  ∃ (a b : ℝ), (∀ (x y : ℝ), 3*x^2 + 2*x*y + 4*y^2 - 18*x - 28*y + 50 = 0 → (y/x = a ∨ y/x = b)) ∧ a + b = 13 :=
by
  sorry

end sum_max_min_ratio_ellipse_l368_368657


namespace pascal_triangle_rows_sum_l368_368319

theorem pascal_triangle_rows_sum :
  ∑ k in finset.range 30, (k + 1) = 465 := by
  sorry

end pascal_triangle_rows_sum_l368_368319


namespace quadratic_inequality_solution_set_l368_368237

/- Given a quadratic function with specific roots and coefficients, prove a quadratic inequality. -/
theorem quadratic_inequality_solution_set :
  ∀ (a b : ℝ),
    (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 + a*x + b < 0) →
    a = -3 →
    b = 2 →
    ∀ x : ℝ, (x < 1/2 ∨ x > 1) ↔ (2*x^2 - 3*x + 1 > 0) :=
by
  intros a b h cond_a cond_b x
  sorry

end quadratic_inequality_solution_set_l368_368237


namespace pascal_triangle_row_sum_l368_368272

theorem pascal_triangle_row_sum : (∑ n in Finset.range 30, n + 1) = 465 := by
  sorry

end pascal_triangle_row_sum_l368_368272


namespace cost_of_pants_is_250_l368_368986

variable (costTotal : ℕ) (costTShirt : ℕ) (numTShirts : ℕ) (numPants : ℕ)

def costPants (costTotal costTShirt numTShirts numPants : ℕ) : ℕ :=
  let costTShirts := numTShirts * costTShirt
  let costPantsTotal := costTotal - costTShirts
  costPantsTotal / numPants

-- Given conditions
axiom h1 : costTotal = 1500
axiom h2 : costTShirt = 100
axiom h3 : numTShirts = 5
axiom h4 : numPants = 4

-- Prove each pair of pants costs $250
theorem cost_of_pants_is_250 : costPants costTotal costTShirt numTShirts numPants = 250 :=
by
  -- Place proof here
  sorry

end cost_of_pants_is_250_l368_368986


namespace right_triangle_acute_angle_l368_368358

/-- In a right triangle, if one acute angle is 25 degrees, then the measure of the other acute angle is 65 degrees. -/
theorem right_triangle_acute_angle (α : ℝ) (hα : α = 25) : ∃ β : ℝ, β = 65 := 
by
  have h_sum : α + 65 = 90 := by
    rw hα
    norm_num
  use 65
  exact h_sum.symm
  sorry

end right_triangle_acute_angle_l368_368358


namespace max_number_of_squares_with_twelve_points_l368_368040

-- Define the condition: twelve marked points in a grid
def twelve_points_marked_on_grid : Prop := 
  -- Assuming twelve specific points represented in a grid-like structure
  -- (This will be defined concretely in the proof implementation context)
  sorry

-- Define the problem statement to be proved
theorem max_number_of_squares_with_twelve_points : 
  twelve_points_marked_on_grid → (∃ n, n = 11) :=
by 
  sorry

end max_number_of_squares_with_twelve_points_l368_368040


namespace boat_speed_in_still_water_l368_368602

-- Definitions for conditions
def Vs : ℝ := 12
def time_downstream : ℝ := t -- Assume t is some positive real number
def time_upstream : ℝ := 2 * time_downstream

-- Assume the distances are the same in both directions
def D_downstream (Vb : ℝ) : ℝ := (Vb + Vs) * time_downstream
def D_upstream (Vb : ℝ) : ℝ := (Vb - Vs) * time_upstream

-- Statement to prove
theorem boat_speed_in_still_water (Vb : ℝ) (t : ℝ) (h : D_downstream Vb = D_upstream Vb) : 
  Vb = 36 :=
by
  sorry

end boat_speed_in_still_water_l368_368602


namespace jill_runs_more_than_jack_l368_368363

noncomputable def streetWidth : ℝ := 15 -- Street width in feet
noncomputable def blockSide : ℝ := 300 -- Side length of the block in feet

noncomputable def jacksPerimeter : ℝ := 4 * blockSide -- Perimeter of Jack's running path
noncomputable def jillsPerimeter : ℝ := 4 * (blockSide + 2 * streetWidth) -- Perimeter of Jill's running path on the opposite side of the street

theorem jill_runs_more_than_jack :
  jillsPerimeter - jacksPerimeter = 120 :=
by
  sorry

end jill_runs_more_than_jack_l368_368363


namespace find_length_of_PB_l368_368822

theorem find_length_of_PB
  (PA : ℝ) -- Define PA
  (h_PA : PA = 4) -- Condition PA = 4
  (PB : ℝ) -- Define PB
  (PT : ℝ) -- Define PT
  (h_PT : PT = PB - 2 * PA) -- Condition PT = PB - 2 * PA
  (h_power_of_a_point : PA * PB = PT^2) -- Condition PA * PB = PT^2
  : PB = 16 :=
sorry

end find_length_of_PB_l368_368822


namespace min_Fx_value_l368_368017

noncomputable def Fx (a : ℝ) : ℝ := 2 * (a + 1)^2 / (a - 1)^2 + (2 * a)^2 / (a - 1)^2

theorem min_Fx_value : (∃ a : ℝ, -0.5 ≤ a ∧ a ≤ 2 ∧ Fx(a) = -2 + 4 / (a - 1)^2) →
  ∃ a : ℝ, -0.5 ≤ a ∧ a ≤ 2 ∧ Fx(a) = -2 + 4 / (a - 1)^2 := by
  sorry

end min_Fx_value_l368_368017


namespace josie_shopping_time_l368_368857

theorem josie_shopping_time :
  let waiting_time := 5 + 10 + 8 + 15 + 20 + 10 in
  let walking_time := float.to_minutes (1.5 * 7) in
  let browsing_time := 12 + 7 + 10 in
  let total_trip_time := 2 * 60 + 45 in
  total_trip_time - (waiting_time + walking_time).to_nat = 86 :=
by sorry

end josie_shopping_time_l368_368857


namespace f_2019_val_l368_368082

def a_n (n : ℕ) : ℕ := nat.find_greatest (λ k, k*k ≤ n) n
def b_n (n : ℕ) : ℕ := n - (a_n n) * (a_n n)

def f (n : ℕ) : ℕ := 
  if b_n n ≤ a_n n then
    (a_n n) * (a_n n) + 1
  else
    (a_n n) * (a_n n) + (a_n n) + 1

theorem f_2019_val : f 2019 = 1981 := by sorry

end f_2019_val_l368_368082


namespace relationship_am_gm_hm_l368_368760

theorem relationship_am_gm_hm (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_neq : a ≠ b) :
  let c := a^2
      d := b^2 in
  (c + d) / 2 > Real.sqrt (c * d) ∧ Real.sqrt (c * d) > (2 * c * d) / (c + d) :=
by nothsorry

end relationship_am_gm_hm_l368_368760


namespace solve_system_l368_368460

theorem solve_system :
  ∃ x y : ℝ, 3^x * 2^y = 972 ∧ log (sqrt 3) (x - y) = 2 ∧ x = 5 ∧ y = 2 :=
by { sorry }

end solve_system_l368_368460


namespace find_matrix_and_inverse_l368_368161

noncomputable theory
open Matrix

variables {c d : ℝ}
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 3; c, d]
def α₁ : Fin 2 → ℝ := !![1, 1]
def α₂ : Fin 2 → ℝ := !![3, -2]
def λ₁ : ℝ := 6
def λ₂ : ℝ := 1

theorem find_matrix_and_inverse (h1 : A.mulVec α₁ = λ₁ • α₁)
                                (h2 : A.mulVec α₂ = λ₂ • α₂) :
  A = !![3, 3; 2, 4] ∧ A.inverse = !![2/3, -1/2; -1/3, 1/2] :=
sorry

end find_matrix_and_inverse_l368_368161


namespace expression_in_ad2_bd_c_l368_368468

theorem expression_in_ad2_bd_c (d : ℤ) (h : d ≠ 0) : 
  let a := 17
  let b := 18
  let c := 18
  (15 * d + 16 + 17 * d^2) + (3 * d + 2) = a * d^2 + b * d + c ∧ a + b + c = 53 :=
by 
  let a := 17
  let b := 18
  let c := 18
  show (15 * d + 16 + 17 * d^2) + (3 * d + 2) = a * d^2 + b * d + c ∧ a + b + c = 53
  sorry

end expression_in_ad2_bd_c_l368_368468


namespace total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368311

-- Define the number of elements in the nth row of Pascal's Triangle
def num_elements_in_row (n : ℕ) : ℕ := n + 1

-- Define the sum of numbers from 0 to n, inclusive
def sum_of_first_n_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the total number of elements in the first 30 rows (0th to 29th)
def total_elements_in_first_30_rows : ℕ := sum_of_first_n_numbers 30

-- The main statement to prove
theorem total_numbers_in_first_30_rows_of_Pascals_Triangle :
  total_elements_in_first_30_rows = 465 :=
by
  simp [total_elements_in_first_30_rows, sum_of_first_n_numbers]
  sorry

end total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368311


namespace sin_double_angle_BAD_l368_368870

-- Defining the objects and their properties
noncomputable def square_side : ℝ := 2
noncomputable def diagonal (a : ℝ) := Real.sqrt (a^2 + a^2)
noncomputable def angle_measure := 45 -- 45 degrees in the context of a square

-- Main theorem statement
theorem sin_double_angle_BAD :
  let AC := diagonal square_side in
  let BAD_deg := angle_measure in
  let sin_BAD := Real.sin (BDD_deg * Real.pi / 180) in
  let cos_BAD := Real.cos (BDD_deg * Real.pi / 180) in
  2 * sin_BAD * cos_BAD = 1 :=
by
  -- Introducing the specifics of the problem
  let ACD_area := (1/2) * AC * square_side
  let ACD_perimeter := AC + square_side + (diagonal square_side)
  -- Using conditions to demonstrate the equality
  have : ACD_area = ACD_perimeter, sorry

  -- Showing that \(\sin(2\angle BAD) = 1\)
  have double_angle_identity : 2 * sin_BAD * cos_BAD = Real.sin (2 * BAD_deg * Real.pi / 180), from sorry
  rw double_angle_identity
  exact Real.sin_pi, sorry

end sin_double_angle_BAD_l368_368870


namespace minimum_points_to_guarantee_highest_score_l368_368783

theorem minimum_points_to_guarantee_highest_score :
  ∃ (score1 score2 score3 : ℕ), 
   (score1 = 7 ∨ score1 = 4 ∨ score1 = 2) ∧ (score2 = 7 ∨ score2 = 4 ∨ score2 = 2) ∧
   (score3 = 7 ∨ score3 = 4 ∨ score3 = 2) ∧ 
   (∀ (score4 : ℕ), 
     (score4 = 7 ∨ score4 = 4 ∨ score4 = 2) → 
     (score1 + score2 + score3 + score4 < 25)) → 
  score1 + score2 + score3 + 7 ≥ 25 :=
   sorry

end minimum_points_to_guarantee_highest_score_l368_368783


namespace equilateral_triangle_ab_value_l368_368506

theorem equilateral_triangle_ab_value
  (a b : ℝ)
  (h : ∃ (a b : ℝ), (a - 1 + 6 * complex.I) * (1 / 2 + sqrt 3 / 2 * complex.I) = b - 1 + 18 * complex.I)
  : a * b = 61 + 12 * sqrt 3 :=
sorry

end equilateral_triangle_ab_value_l368_368506


namespace graph_of_equation_is_two_lines_l368_368164

theorem graph_of_equation_is_two_lines :
  ∀ (x y : ℝ), (2 * x - y)^2 = 4 * x^2 - y^2 ↔ (y = 0 ∨ y = 2 * x) :=
by
  sorry

end graph_of_equation_is_two_lines_l368_368164


namespace area_trapezoid_integer_l368_368789

-- Define the points A, B, C, D, and center O
variable {O : Point}
variable {A B C D : Point}
variable {AB CD : ℝ}

-- Define the conditions
variable (h1 : perpendicular A B B C)
variable (h2 : perpendicular B C C D)
variable (h3 : tangent B C (circle O (diameter A D)))
variable (h4 : AB = 8)
variable (h5 : CD = 1)

-- Define the result
theorem area_trapezoid_integer (h1 : perpendicular A B B C) (h2 : perpendicular B C C D)
    (h3 : tangent B C (circle O (diameter A D))) (h4 : AB = 8) (h5 : CD = 1) : 
    is_integer (area_trapezoid A B C D) := 
by
  sorry

end area_trapezoid_integer_l368_368789


namespace Donna_episodes_per_weekday_l368_368166

noncomputable def episodes_per_weekday (E : ℕ) : Prop :=
  5 * E + 6 * E = 88 → E = 8

theorem Donna_episodes_per_weekday (E : ℕ) : episodes_per_weekday E := by
  intro h
  have h1 := (5 * E + 6 * E)
  have h2 := (11 * E)
  rw h1 at h
  rw h2 at h
  exact Nat.eq_of_mul_eq_mul_left (by norm_num) h


end Donna_episodes_per_weekday_l368_368166


namespace loaned_out_books_l368_368601

theorem loaned_out_books (A B : ℕ) 
  (initA : A = 75) 
  (initB : B = 60) 
  (returnedA : ∀ (la : ℕ), 0.8 * la) 
  (returnedB : ∀ (lb : ℕ), 0.7 * lb)
  (endA : ∀ (la : ℕ), 67 = A - 0.2 * la) 
  (endB : ∀ (lb : ℕ), 48 = B - 0.3 * lb) : 
  A + B = 80 := 
sorry

end loaned_out_books_l368_368601


namespace fewer_onions_than_tomatoes_and_corn_l368_368640

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions_than_tomatoes_and_corn :
  (tomatoes + corn - onions) = 5200 :=
by
  sorry

end fewer_onions_than_tomatoes_and_corn_l368_368640


namespace pascal_triangle_sum_first_30_rows_l368_368292

theorem pascal_triangle_sum_first_30_rows :
  (Finset.range 30).sum (λ n, n + 1) = 465 :=
begin
  sorry
end

end pascal_triangle_sum_first_30_rows_l368_368292


namespace maximum_volume_of_container_l368_368052

noncomputable def max_volume_of_container : ℝ :=
  let total_length : ℝ := 14.8
  let side_diff : ℝ := 0.5
  let volume (x : ℝ) : ℝ := x * (x + 0.5) * (3.45 - x)
  let critical_x : ℝ := 1
  volume critical_x

theorem maximum_volume_of_container (total_length : ℝ) (side_diff : ℝ) (h_total_length : total_length = 14.8) (h_side_diff : side_diff = 0.5) :
  max_volume_of_container = 3.675 :=
by 
  let x := 1
  let h := 3.45 - x
  let volume := x * (x + 0.5) * h
  have h1 : 2 * x + 2 * (x + 0.5) + 4 * h = total_length := by sorry
  have h2 : volume = 3.675 := by sorry
  exact h2

end maximum_volume_of_container_l368_368052


namespace trajectory_equation_shortest_distance_to_line_l368_368369

-- Definitions for the conditions
def point_F : ℝ × ℝ := (0, 1)
def line_l (x : ℝ) : ℝ := -1
def on_perpendicular_line_at_Q (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove the trajectory equation
theorem trajectory_equation (P : ℝ × ℝ) :
  let Q := on_perpendicular_line_at_Q P in
  dot_product ⟨0, P.2 + 1⟩ ⟨-P.1, 2⟩ = dot_product ⟨P.1, P.2 - 1⟩ ⟨P.1, -2⟩ →
  P.1 * P.1 = 4 * P.2 :=
by
  intro Q
  intro h
  sorry

-- Prove the shortest distance from a point on curve to a given line
theorem shortest_distance_to_line (C : ℝ × ℝ → Prop) (M : ℝ × ℝ) (line_eq : ℝ × ℝ → ℝ) :
  (∀ P : ℝ × ℝ, C P ↔ P.1 * P.1 = 4 * P.2) →
  C M → line_eq (M.1, M.2) = M.1 - 3 →
  ∃ d : ℝ, d = 2 ^ (-1/2) ∧
    ∀ x₀ : ℝ, let P := (x₀, x₀^2 / 4) in
    abs (x₀ - M.2 - 3) / real.sqrt 2 = d :=
by
  intros hC hM hline
  use 2 ^ (-1/2)
  split
  { sorry }
  { intros x₀ P
    sorry }

end trajectory_equation_shortest_distance_to_line_l368_368369


namespace sum_of_consecutive_integers_largest_set_of_consecutive_positive_integers_l368_368898

theorem sum_of_consecutive_integers (n : ℕ) (a : ℕ) (h : n ≥ 1) (h_sum : n * (2 * a + n - 1) = 56) : n ≤ 7 := 
by
  sorry

theorem largest_set_of_consecutive_positive_integers : ∃ n a, n ≥ 1 ∧ n * (2 * a + n - 1) = 56 ∧ n = 7 := 
by
  use 7, 1
  repeat {split}
  sorry

end sum_of_consecutive_integers_largest_set_of_consecutive_positive_integers_l368_368898


namespace correct_statement_S_growth_curve_l368_368791

/-- 
Given conditions about the "S" shaped growth curve:
- A: The "S" shaped growth curve represents the relationship between population size and food.
- B: The population growth rate varies at different stages.
- C: The "S" shaped growth curve indicates that the population size is unrelated to time.
- D: Population growth is not restricted by population density.

Prove that statement B is correct.
-/
theorem correct_statement_S_growth_curve 
  (A : Prop := "The 'S' shaped growth curve represents the relationship between population size and food.")
  (B : Prop := "The population growth rate varies at different stages.")
  (C : Prop := "The 'S' shaped growth curve indicates that the population size is unrelated to time.")
  (D : Prop := "Population growth is not restricted by population density.") 
  (H_A : ¬ A)
  (H_B : B)
  (H_C : ¬ C)
  (H_D : ¬ D) :
  B :=
by
  exact H_B

end correct_statement_S_growth_curve_l368_368791


namespace initial_discount_l368_368422

theorem initial_discount (total_amount price_after_initial_discount additional_disc_percent : ℝ)
  (H1 : total_amount = 1000)
  (H2 : price_after_initial_discount = total_amount - 280)
  (H3 : additional_disc_percent = 0.20) :
  let additional_discount := additional_disc_percent * price_after_initial_discount
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let total_discount := total_amount - price_after_additional_discount
  let initial_discount := total_discount - additional_discount
  initial_discount = 280 := by
  sorry

end initial_discount_l368_368422


namespace common_difference_is_2_l368_368228

variables {d a1 a2 a3 S1 S2 S3 : ℝ}
def arithmetic_sequence (a1 a2 a3 : ℝ) (d : ℝ) : Prop :=
  a2 = a1 + d ∧ a3 = a2 + d

def sum_of_first_n_terms (a1 a2 a3 : ℝ) (d : ℝ) (n : ℝ) : ℝ :=
  match n with
  | 1 => a1
  | 2 => a1 + a2
  | 3 => a1 + a2 + a3
  | _ => 0 -- Simplified for the first three terms used

theorem common_difference_is_2 :
  (arithmetic_sequence a1 a2 a3 d) →
  (sum_of_first_n_terms a1 a2 a3 d 3 = S3) →
  (sum_of_first_n_terms a1 a2 a3 d 2 = S2) →
  (S3 / 3 - S2 / 2 = 1) →
  d = 2 :=
by
  sorry

end common_difference_is_2_l368_368228


namespace determine_c_l368_368333

theorem determine_c (c : ℝ) 
  (h : ∃ a : ℝ, (∀ x : ℝ, x^2 + 200 * x + c = (x + a)^2)) : c = 10000 :=
sorry

end determine_c_l368_368333


namespace burger_cost_eq_78_l368_368921

theorem burger_cost_eq_78 :
  ∃ (b s : ℕ), 3 * b + 2 * s = 450 ∧ 2 * b + 3 * s = 480 ∧ b = 78 :=
by {
  use (78, 108),
  split; ring,
  split; ring,
  refl,
  sorry
}

end burger_cost_eq_78_l368_368921


namespace find_ratio_l368_368823
noncomputable theory

variables {a1 a2 d S1 S2 S4 : ℝ}

def arithmetic_sequence (a1 : ℝ) (d : ℝ) : (ℕ → ℝ) :=
λ n, a1 + n * d

def sum_of_first_n_terms (a1 d : ℝ) (n : ℕ) : ℝ :=
n * a1 + d * (n * (n - 1) / 2)

def geometric_sequence (S1 S2 S4 : ℝ) : Prop :=
S2^2 = S1 * S4

theorem find_ratio
  (h1 : ∃ d, S1 = a1 ∧ S2 = 2 * a1 + d ∧ S4 = 4 * a1 + 6 * d)
  (h2 : S2^2 = S1 * S4)
  (h3 : a1 ≠ 0)
  : a2 / a1 = 3 ∨ a2 / a1 = 1 := 
sorry

end find_ratio_l368_368823


namespace total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368307

-- Define the number of elements in the nth row of Pascal's Triangle
def num_elements_in_row (n : ℕ) : ℕ := n + 1

-- Define the sum of numbers from 0 to n, inclusive
def sum_of_first_n_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the total number of elements in the first 30 rows (0th to 29th)
def total_elements_in_first_30_rows : ℕ := sum_of_first_n_numbers 30

-- The main statement to prove
theorem total_numbers_in_first_30_rows_of_Pascals_Triangle :
  total_elements_in_first_30_rows = 465 :=
by
  simp [total_elements_in_first_30_rows, sum_of_first_n_numbers]
  sorry

end total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368307


namespace other_acute_angle_in_right_triangle_l368_368360

theorem other_acute_angle_in_right_triangle (a : ℝ) (h : a = 25) :
    ∃ b : ℝ, b = 65 :=
by
  sorry

end other_acute_angle_in_right_triangle_l368_368360


namespace unique_solution_for_star_l368_368160

def star (x y : ℝ) : ℝ := 4 * x - 5 * y + 2 * x * y

theorem unique_solution_for_star :
  ∃! y : ℝ, star 2 y = 5 :=
by
  -- We know the definition of star and we need to verify the condition.
  sorry

end unique_solution_for_star_l368_368160


namespace george_initial_amount_l368_368694

-- Definitions as per conditions
def cost_of_shirt : ℕ := 24
def cost_of_socks : ℕ := 11
def amount_left : ℕ := 65

-- Goal: Prove that the initial amount of money George had is 100
theorem george_initial_amount : (cost_of_shirt + cost_of_socks + amount_left) = 100 := 
by sorry

end george_initial_amount_l368_368694


namespace simplify_expression_l368_368452

theorem simplify_expression (x : ℝ) (h1 : -x^3 ≥ 0) (h2 : x ≠ 0) : 
  (sqrt (-x^3) / x = -sqrt (-x)) :=
by
  sorry

end simplify_expression_l368_368452


namespace sin_A_in_triangle_ABC_l368_368356

theorem sin_A_in_triangle_ABC 
  (A B C : Triangle)
  (h_non_obtuse : ¬ obtuse A B C)
  (h_AB_gt_AC : length A B > length A C)
  (h_angle_B : angle_at A B C = 45)
  (O : Circumcenter A B C)
  (I : Incenter A B C)
  (h_relation : sqrt 2 * distance O I = length A B - length A C) :
  (sin (angle_at B A C) = sqrt 2 / 2 ∨ sin (angle_at B A C) = sqrt (sqrt 2 - 1 / 2)) :=
by sorry

end sin_A_in_triangle_ABC_l368_368356


namespace probability_of_a_b_c_l368_368930

noncomputable def probability_condition : ℚ :=
  5 / 6 * 5 / 6 * 7 / 8

theorem probability_of_a_b_c : 
  let a_outcome := 6
  let b_outcome := 6
  let c_outcome := 8
  (1 / a_outcome) * (1 / b_outcome) * (1 / c_outcome) = probability_condition :=
sorry

end probability_of_a_b_c_l368_368930


namespace double_neg_cancel_l368_368455

theorem double_neg_cancel (a : ℤ) : - (-2) = 2 :=
sorry

end double_neg_cancel_l368_368455


namespace area_under_curve_l368_368194

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then x^2
  else if 3 < x ∧ x ≤ 10 then 3 * x - 6
  else 0

theorem area_under_curve :
  let A1 := ∫ x in 0..3, x^2
  let A2 := (1 / 2) * (3 + 24) * (10 - 3)
  let K := A1 + A2
  K = 103.5 :=
by
  sorry

end area_under_curve_l368_368194


namespace complement_of_A_in_U_l368_368350

open Set

noncomputable def U : Set ℤ := {-1, 0, 1, 2}
noncomputable def A : Set ℤ := {x ∈ (id : Set ℤ) | x^2 < 2}

theorem complement_of_A_in_U :
  U \ A = {2} :=
by {
  sorry
}

end complement_of_A_in_U_l368_368350


namespace total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368304

-- Define the number of elements in the nth row of Pascal's Triangle
def num_elements_in_row (n : ℕ) : ℕ := n + 1

-- Define the sum of numbers from 0 to n, inclusive
def sum_of_first_n_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the total number of elements in the first 30 rows (0th to 29th)
def total_elements_in_first_30_rows : ℕ := sum_of_first_n_numbers 30

-- The main statement to prove
theorem total_numbers_in_first_30_rows_of_Pascals_Triangle :
  total_elements_in_first_30_rows = 465 :=
by
  simp [total_elements_in_first_30_rows, sum_of_first_n_numbers]
  sorry

end total_numbers_in_first_30_rows_of_Pascals_Triangle_l368_368304


namespace seating_arrangements_l368_368692

-- Defining the conditions as Lean functions
structure Family :=
  (child1 : String)
  (child2 : String)

structure Siblings :=
  (fam1 : Family)
  (fam2 : Family)

structure Van :=
  (rows: Nat)
  (seats_per_row: Nat)
  (seating_arrangement_count : Nat)

-- Define the condition statements
def noAdjacency (van : Van) (siblings : Siblings) : Prop :=
  -- No two siblings can be next to each other or front to back
  sorry

-- Calculate number of seating arrangements based on conditions
def calc_seating_arrangements (van : Van) (siblings : Siblings) : Nat :=
  if noAdjacency van siblings then 48 else 0

-- The main theorem to state proof problem
theorem seating_arrangements (van : Van) (siblings: Siblings) (h : noAdjacency van siblings) : calc_seating_arrangements van siblings = 48 :=
  sorry

end seating_arrangements_l368_368692


namespace cost_of_72_tulips_is_115_20_l368_368979

/-
Conditions:
1. A package containing 18 tulips costs $36.
2. The price of a package is directly proportional to the number of tulips it contains.
3. There is a 20% discount applied for packages containing more than 50 tulips.
Question:
What is the cost of 72 tulips?

Correct answer:
$115.20
-/

def costOfTulips (numTulips : ℕ)  : ℚ :=
  if numTulips ≤ 50 then
    36 * numTulips / 18
  else
    (36 * numTulips / 18) * 0.8 -- apply 20% discount for more than 50 tulips

theorem cost_of_72_tulips_is_115_20 :
  costOfTulips 72 = 115.2 := 
sorry

end cost_of_72_tulips_is_115_20_l368_368979


namespace find_principal_l368_368942

variable (P : ℝ) -- Define P (Principal)

/-- Given conditions --/
def SI : ℝ := 4016.25
def R : ℕ := 9
def T : ℕ := 5
def SI_formula (P : ℝ) (R : ℕ) (T : ℕ) : ℝ := P * R * T / 100

theorem find_principal : SI_formula P R T = SI → P = 8925 := by
  sorry

end find_principal_l368_368942


namespace quadrilateral_is_square_l368_368607

noncomputable def is_square (ABCD : Quadrilateral) : Prop := 
  ∀ (fold1 : Quadrilateral → Bool) (fold2 : Quadrilateral → Bool),
  fold1 ABCD = true ∧ fold2 ABCD = true ↔ 
  (is_rhombus ABCD ∧ (length (diagonal1 ABCD) = length (diagonal2 ABCD)))

-- Axioms related to the properties of the quadrilateral
axiom is_rhombus : Quadrilateral → Prop
axiom length : Diagonal → ℝ
axiom diagonal1 : Quadrilateral → Diagonal
axiom diagonal2 : Quadrilateral → Diagonal

-- Definition of a quadrilateral (abstracted)
structure Quadrilateral

-- Definitions of folding rules for determining if a shape is a rhombus and ensuring equal diagonals
axiom fold1 : Quadrilateral → Bool
axiom fold2 : Quadrilateral → Bool

-- The theorem stating the proof problem
theorem quadrilateral_is_square (ABCD : Quadrilateral) 
  (H1 : fold1 ABCD = true) 
  (H2 : fold2 ABCD = true) : 
  is_square ABCD :=
by {
  sorry
}

end quadrilateral_is_square_l368_368607


namespace qbasic_input_statement_correct_l368_368890

theorem qbasic_input_statement_correct :
  (∃ (input_statement : String), input_statement = "INPUT \"prompt content\"; variable" ∧
  "INPUT" is a keyword for keyboard input statement in QBASIC ∧
  ∀ variable : String, user inputs during the program's execution) :=
sorry

end qbasic_input_statement_correct_l368_368890


namespace f_monotonically_decreasing_in_interval_l368_368241

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

theorem f_monotonically_decreasing_in_interval :
  ∀ x y : ℝ, -2 < x ∧ x < 1 → -2 < y ∧ y < 1 → (y > x → f y < f x) :=
by
  sorry

end f_monotonically_decreasing_in_interval_l368_368241


namespace range_of_m_l368_368869

theorem range_of_m :
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0 → -2 ≤ m ∧ m ≤ 2) ∧
  (∀ m : ℝ, 0 < m ∧ m < 2 → 
    ∃ k : ℝ, ∀ x y : ℝ, k = 1 / m ∧ (x^2 / m + y^2 / 2 = 1) ∧ ellipse_with_y_foci (x / √m, y / √2) ) ∧
  ((∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ∧ 
    (¬(∀ m : ℝ, 0 < m ∧ m < 2 → ellipse_with_y_foci (1 / √m, 1 / √2)) ∨ 
         ¬(∀ x : ℝ, x^2 + m * x + 1 ≥ 0))) → 
  m ∈ [(-2), 0] ∪ {2} :=
begin
  sorry
end

end range_of_m_l368_368869


namespace maximum_value_of_z_l368_368719

theorem maximum_value_of_z :
  ∃ x y : ℝ, (x - y ≥ 0) ∧ (x + y ≤ 2) ∧ (y ≥ 0) ∧ (∀ u v : ℝ, (u - v ≥ 0) ∧ (u + v ≤ 2) ∧ (v ≥ 0) → 3 * u - v ≤ 6) :=
by
  sorry

end maximum_value_of_z_l368_368719


namespace find_radius_original_bubble_l368_368617

-- Define the given radius of the hemisphere.
def radius_hemisphere : ℝ := 4 * real.cbrt 2

-- Define the volume of a hemisphere.
def vol_hemisphere (r : ℝ) := (2 / 3) * π * r^3

-- Define the volume of a sphere.
def vol_sphere (R : ℝ) := (4 / 3) * π * R^3

-- Given condition: The volume of the hemisphere is equal to the volume of the original bubble.
def volume_equivalence (r R : ℝ) := vol_sphere R = vol_hemisphere r

-- State the theorem to prove: Given the radius of the hemisphere, find the radius of the original bubble.
theorem find_radius_original_bubble (R : ℝ) (h : volume_equivalence radius_hemisphere R) : R = 4 :=
sorry

end find_radius_original_bubble_l368_368617


namespace pascal_triangle_count_30_rows_l368_368286

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l368_368286


namespace system_of_inequalities_solution_l368_368929

theorem system_of_inequalities_solution (a : ℝ) (x : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  (log a (abs (x - π/3)) > log a (2*π/3)) ∧ (cos x ≥ 0) ↔
  (0 ≤ x ∧ x < π/3 ∨ π/3 < x ∧ x ≤ π/2) := 
by
  sorry

end system_of_inequalities_solution_l368_368929


namespace most_likely_top_quality_products_l368_368018

theorem most_likely_top_quality_products
  (p : ℝ) (n : ℝ) (h1 : p = 0.31) (h2 : n = 75) : 
  ∃ m_0 : ℝ, 22.56 ≤ m_0 ∧ m_0 ≤ 23.56 ∧ m_0 = 23 :=
by
  let m_0 := n * p
  have : 22.56 ≤ m_0 := by {
    rw [← h1, ← h2],
    norm_num,
  }
  have : m_0 ≤ 23.56 := by {
    rw [← h1, ← h2],
    norm_num,
  }
  have : m_0 = 23 := by {
    rw [← h1, ← h2],
    norm_num,
  }
  exact ⟨m_0, this_left, this_right, this_equal⟩
sorry

end most_likely_top_quality_products_l368_368018


namespace num_three_digit_numbers_ending_in_product_of_digits_l368_368739

-- Definition statements setting up the conditions
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n < 10
def non_zero_digit (n : ℕ) : Prop := is_digit n ∧ n ≠ 0
def is_three_digit_number (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def digits_of (n : ℕ) (a b c : ℕ) : Prop := non_zero_digit a ∧ is_digit b ∧ is_digit c ∧ n = 100 * a + 10 * b + c
def ends_in_product_of_digits (n a b c : ℕ) : Prop := digits_of n a b c ∧ n % 10^(Int.length (a * b * c).digits) = a * b * c

-- Main theorem
theorem num_three_digit_numbers_ending_in_product_of_digits : (finset.univ.filter (λ n, is_three_digit_number n ∧ ∃ a b c, ends_in_product_of_digits n a b c)).card = 95 :=
sorry

end num_three_digit_numbers_ending_in_product_of_digits_l368_368739


namespace find_vector_b_coords_l368_368253

noncomputable def vector_b_coords
  (a : ℝ × ℝ)
  (mag_b : ℝ)
  (cos_theta : ℝ)
  (h_a : a = (-1, 2))
  (h_mag_b : mag_b = 3 * Real.sqrt 5)
  (h_cos_theta : cos_theta = -1) :
  (ℝ × ℝ) :=
let θ := π,
    λ := 3,
    b := (λ, -2 * λ) in
b

theorem find_vector_b_coords
  (a : ℝ × ℝ)
  (mag_b : ℝ)
  (cos_theta : ℝ)
  (h_a : a = (-1, 2))
  (h_mag_b : mag_b = 3 * Real.sqrt 5)
  (h_cos_theta : cos_theta = -1) :
  vector_b_coords a mag_b cos_theta h_a h_mag_b h_cos_theta = (3, -6) :=
by {
  -- start of the proof part
  sorry
}

end find_vector_b_coords_l368_368253


namespace initial_mean_corrected_l368_368013

theorem initial_mean_corrected 
  (M : ℝ) 
  (h1 : ∑ i in range 40, M = 40 * M)
  (h2 : ∑ i in range 40, if i = 0 then 75 else M = 40 * M + 25)
  (h3 : ∑ i in range 40, if i = 0 then 50 else M = 40 * 99.075) :
  M = 98.45 :=
by
  sorry

end initial_mean_corrected_l368_368013


namespace negation_of_forall_l368_368896

theorem negation_of_forall (h : ¬ ∀ x > 0, Real.exp x > x + 1) : ∃ x > 0, Real.exp x < x + 1 :=
sorry

end negation_of_forall_l368_368896


namespace correct_statements_l368_368016

/-- The coefficient of determination R^2 can characterize the fit of the regression model.
  The closer R^2 is to 1, the better the fit of the model. -/
def statement_1 : Prop :=
  ∀ (R2 : ℝ), 0 ≤ R2 ∧ R2 ≤ 1 → (R2 = 1 → (∀ ⦃x⦄, R2 characterizes the fit of the regression model))

/-- In a linear regression model, R^2 indicates the proportion of the variance for the dependent 
  variable that's explained by the independent variable(s). The closer R^2 is to 1, the stronger 
  the linear relationship between the independent and dependent variables. -/
def statement_2 : Prop :=
  ∀ (R2 : ℝ), 0 ≤ R2 ∧ R2 ≤ 1 → (R2 = 1 → (R2 indicates the proportion of the variance for the 
  dependent variable explained by the independent variable(s)))

/-- If there are individual points in the residual plot with relatively large residuals,
  it should be confirmed whether there are manual errors in the process of collecting 
  sample points or whether the model is appropriate. -/
def statement_3 : Prop :=
  ∀ ⦃e : ℝ⦄, (e is large in residual plot) → (check for manual errors ∨ model appropriateness)

theorem correct_statements : Nat :=
  if statement_1 ∧ statement_2 ∧ statement_3 then 3 else 0

#eval correct_statements  -- should evaluate to 3

end correct_statements_l368_368016


namespace dice_probability_on_line_l368_368118

theorem dice_probability_on_line (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 6) (hy : 1 ≤ y ∧ y ≤ 6) : 
  (∃ x y, ((2 * x - y = 1) ∧ (1 ≤ x ∧ x ≤ 6) ∧ (1 ≤ y ∧ y ≤ 6))) -> 
  (fin_prob : ℝ) := 
  begin 
    have h : 36 = (6 * 6), sorry,
    have s : set (fin 36),
    have F := { (x,y) // 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 2 * x - y = 1 }, sorry,
    let n := card s,
    let f := card F,
    let p := (f:ℝ)/ (n:ℝ), 
    exact p = 1/12
  end

end dice_probability_on_line_l368_368118


namespace omega_value_l368_368492

theorem omega_value {ω : ℝ} (h1: ∀ x : ℝ, (sin (ω * x + π / 6) ≤ sin (ω * π + π / 6)))
                    (h2: ∀ x1 x2 : ℝ, -π / 6 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 6 → sin (ω * x1 + π / 6) ≤ sin (ω * x2 + π / 6)) :
  ω = 1 / 3 :=
by
  sorry

end omega_value_l368_368492


namespace at_least_area_n_uncovered_l368_368969

theorem at_least_area_n_uncovered 
    (n : ℕ) 
    (tile_is_parallelogram : ∀ (t : ℕ), t = 2 * (1/2 * 1) * (√2/2 * 1)) 
    (tiles_in_nxn_room : ∀ (i j : ℕ), 0 ≤ i ∧ i < n ∧ 0 ≤ j ∧ j < n)
    (distance_int : ∀ (i j : ℕ), ∃ k : ℤ, distance (tile_vertex i j) (room_side i j) = k)
    (no_overlap : ∀ (i j : ℕ), disjoint (tile_area i j) (tile_area (i+1) (j+1))) : 
  ∃ uncovered_area ≥ n ∧ uncovered_area ≤ n * n, true :=
sorry

end at_least_area_n_uncovered_l368_368969


namespace simplify_175_sub_57_sub_43_simplify_128_sub_64_sub_36_simplify_156_sub_49_sub_51_l368_368142

theorem simplify_175_sub_57_sub_43 : 175 - 57 - 43 = 75 :=
by
  sorry

theorem simplify_128_sub_64_sub_36 : 128 - 64 - 36 = 28 :=
by
  sorry

theorem simplify_156_sub_49_sub_51 : 156 - 49 - 51 = 56 :=
by
  sorry

end simplify_175_sub_57_sub_43_simplify_128_sub_64_sub_36_simplify_156_sub_49_sub_51_l368_368142


namespace width_of_identical_rectangle_l368_368457

theorem width_of_identical_rectangle (
  (h1 : 6 * w * w = 5400)
  (h2 : 3 * w = 2 * (2 * w))
) : w = 30 := 
sorry

end width_of_identical_rectangle_l368_368457


namespace difference_of_areas_l368_368854

theorem difference_of_areas (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 56) (h_diff : abs (l - w) ≥ 5) : 
  ∃ A_max A_min : ℕ, A_max = l * w ∧ A_min = l * w ∧ A_max - A_min = 5 :=
by
  sorry

end difference_of_areas_l368_368854


namespace product_of_three_equal_numbers_l368_368879

theorem product_of_three_equal_numbers
    (a b : ℕ) (x : ℕ)
    (h1 : a = 12)
    (h2 : b = 22)
    (h_mean : (a + b + 3 * x) / 5 = 20) :
    x * x * x = 10648 := by
  sorry

end product_of_three_equal_numbers_l368_368879


namespace tax_diminished_percentage_l368_368904

theorem tax_diminished_percentage (T T' C R R': ℝ)
  (h1 : ∀ T' T : ℝ, T' = (T * 0.9999999999999858) / 1.1)
  (h2 : ∀ C' C : ℝ, C' = C * 1.1)
  (h3 : ∀ R T C : ℝ, R = T * C)
  (h4 : ∀ R' T' C' : ℝ, R' = T' * C')
  (h5 : ∀ R' R : ℝ, R' = R * 0.9999999999999858)
  : (1 - 0.9999999999999858 / 1.1) * 100 ≈ 9.09 := 
sorry

end tax_diminished_percentage_l368_368904


namespace possible_values_of_x_l368_368394

theorem possible_values_of_x :
  let A := {1, 4, x} : Set ℝ
  let B := {1, x^2} : Set ℝ
  (A ∩ B = B) ↔ (x ∈ ({0, -2, 2} : Set ℝ)) :=
by
  sorry

end possible_values_of_x_l368_368394


namespace triangle_properties_l368_368227

theorem triangle_properties (b c : ℝ) (area : ℝ) (h1 : b = 3) (h2 : c = 1) (h3 : area = sqrt 2) :
  (∃ (A : ℝ), cos A = 1 / 3 ∨ cos A = -1 / 3) ∧
  (∃ (a : ℝ), a = 2 * sqrt 3 ∨ a = 2 * sqrt 2) :=
by
  sorry

end triangle_properties_l368_368227


namespace yoongi_calculation_l368_368936

theorem yoongi_calculation (x : ℝ) (h : x - 5 = 30) : x / 7 = 5 :=
by
  sorry

end yoongi_calculation_l368_368936


namespace volumes_of_cone_and_cylinder_l368_368595

-- Define base and height of cylinder and cone
variables (B h : ℝ)

-- Define volumes of cone and cylinder
def V_cone := (1/3) * B * h
def V_cylinder := B * h

-- Define the difference in volumes
def volume_difference := V_cylinder B h - V_cone B h

-- Prove that the volumes of the cone and the cylinder are as stated
theorem volumes_of_cone_and_cylinder (B h : ℝ) (h_diff : volume_difference B h = 24) :
  V_cone B h = 12 ∧ V_cylinder B h = 36 :=
by
  sorry

end volumes_of_cone_and_cylinder_l368_368595


namespace difference_of_squares_l368_368551

theorem difference_of_squares :
  535^2 - 465^2 = 70000 :=
by
  sorry

end difference_of_squares_l368_368551


namespace cannot_restore_unique_numbers_l368_368631

theorem cannot_restore_unique_numbers :
  ¬ (∀ f : (ℕ → ℕ) → ℕ, (gcd 2 (f 2)) = (gcd 3 (f 3)) ∧ 
  (∀ x y, 2 ≤ x ∧ x ≤ 20000 ∧ 2 ≤ y ∧ y ≤ 20000 → gcd (f x) (f y) = gcd x y) →
   (∀ x y, 2 ≤ x ∧ x ≤ 20000 ∧ 2 ≤ y ∧ y ≤ 20000 → x = y)) :=
sorry

end cannot_restore_unique_numbers_l368_368631


namespace hyperbola_perimeter_l368_368954

theorem hyperbola_perimeter :
  ∀ (a b : ℝ) (F1 F2 A B : ℝ×ℝ), 
    (a = 4) ∧ (F1 = (-a, 0)) ∧ (F2 = (a, 0)) ∧ (∃ k: ℝ, (k > 0) ∧ (k = 6)) ∧ 
    (∀ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ (x1, y1) ≠ (x2, y2) ∧
    (x1 - x2)^2/16 - (y1 - y2)^2/9 = 1) ->
  let AF2 := dist (A.1, A.2) F2 in
  let BF2 := dist (B.1, B.2) F2 in
  let AB := dist (A.1, A.2) (B.1, B.2) in
  AF2 + BF2 + AB = 28 :=
begin
  sorry
end

end hyperbola_perimeter_l368_368954


namespace solution_set_l368_368722

variables (f g : ℝ → ℝ)
variable (g' : ℝ → ℝ)

-- Assume f is an odd function defined on ℝ
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- Assume g(x) = f(x + 1) + 5
def g_definition (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, g(x) = f(x + 1) + 5

-- Assume ∀ x ∈ ℝ, g'(x) > 2x
def g_prime_condition (g' : ℝ → ℝ) : Prop := ∀ x : ℝ, g'(x) > 2x

theorem solution_set (H_f_odd : is_odd_function f) (H_g_def : g_definition f g) (H_g_prime_cond : g_prime_condition g') :
  { x : ℝ | g x < x^2 + 4 } = set.Iio (-1) :=
sorry

end solution_set_l368_368722


namespace triangle_PQR_area_l368_368544

-- Define the points P, Q, and R
def P : (ℝ × ℝ) := (-2, 2)
def Q : (ℝ × ℝ) := (8, 2)
def R : (ℝ × ℝ) := (4, -4)

-- Define a function to calculate the area of triangle
def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Lean statement to prove the area of triangle PQR is 30 square units
theorem triangle_PQR_area : triangle_area P Q R = 30 := by
  sorry

end triangle_PQR_area_l368_368544


namespace average_of_remaining_two_numbers_l368_368079

theorem average_of_remaining_two_numbers :
  ∀ (a b c d e f : ℝ),
    (a + b + c + d + e + f) / 6 = 3.95 →
    (a + b) / 2 = 3.6 →
    (c + d) / 2 = 3.85 →
    ((e + f) / 2 = 4.4) :=
by
  intros a b c d e f h1 h2 h3
  have h4 : a + b + c + d + e + f = 23.7 := sorry
  have h5 : a + b = 7.2 := sorry
  have h6 : c + d = 7.7 := sorry
  have h7 : e + f = 8.8 := sorry
  exact sorry

end average_of_remaining_two_numbers_l368_368079


namespace probability_unit_square_not_touch_central_2x2_square_l368_368431

-- Given a 6x6 checkerboard with a marked 2x2 square at the center,
-- prove that the probability of choosing a unit square that does not touch
-- the marked 2x2 square is 2/3.

theorem probability_unit_square_not_touch_central_2x2_square : 
    let total_squares := 36
    let touching_squares := 12
    let squares_not_touching := total_squares - touching_squares
    (squares_not_touching : ℚ) / (total_squares : ℚ) = 2 / 3 := by
  sorry

end probability_unit_square_not_touch_central_2x2_square_l368_368431


namespace probability_two_slate_rocks_l368_368080

theorem probability_two_slate_rocks {
  total_rocks : ℕ, slate_rocks : ℕ, pumice_rocks : ℕ, granite_rocks : ℕ,
  choose_rocks : ℕ, first_probability: ℚ, second_probability: ℚ, combined_probability: ℚ
} [fact (total_rocks = 25)] [fact (slate_rocks = 10)] [fact (pumice_rocks = 11)] [fact (granite_rocks = 4)] 
  [fact (choose_rocks = 2)] [fact (first_probability = (10 : ℚ) / (25 : ℚ))]
  [fact (second_probability = (9 : ℚ) / (24 : ℚ))] 
  (P_two_slate : combined_probability = first_probability * second_probability):
  combined_probability = (3 : ℚ) / (20 : ℚ) :=
by { sorry }

end probability_two_slate_rocks_l368_368080


namespace cdf_correct_l368_368677

noncomputable def p (x : ℝ) : ℝ :=
  if x ≤ 0 ∨ x > 2 then 0
  else if 0 < x ∧ x ≤ 1 then x
  else if 1 < x ∧ x ≤ 2 then 2 - x
  else 0  -- This case should never occur due to the previous conditions

noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x <= 1 then x^2 / 2
  else if 1 < x ∧ x <= 2 then -x^2 / 2 + 2 * x - 1
  else 1

theorem cdf_correct : ∀ x : ℝ, F(x) = ∫ t in -∞..x, p t :=
by 
  sorry

end cdf_correct_l368_368677


namespace polynomial_real_roots_interval_l368_368472

theorem polynomial_real_roots_interval {n : ℕ} (a : Fin n → ℝ) (h_real_roots : ∀ i, (root_of_polynomial a i).is_real) :
  ∀ i, root_of_polynomial a i ∈ set.Icc
  (-((a n.pred) / n)) 
  (
    (a n.pred) / n
    + ((n - 1) / n) * (sqrt ((a n.pred) ^ 2 - ((2 * n) / (n - 1)) * (a (n.pred - 1))))
  ) :=
sorry

end polynomial_real_roots_interval_l368_368472


namespace min_degree_g_l368_368876

def polynomial (R : Type*) [comm_ring R] := mv_polynomial unit R

variables {R : Type*} [comm_ring R]
variables (f g h : polynomial R)

theorem min_degree_g (hf : f.degree = 10) (hh : h.degree = 12) (H : 5 * f + 6 * g = h) : 
  g.degree ≥ 12 :=
sorry

end min_degree_g_l368_368876


namespace triangle_right_iff_two_R_plus_r_eq_p_l368_368441

theorem triangle_right_iff_two_R_plus_r_eq_p
  {α β γ : Type*} [metric_space α] [metric_space β] [metric_space γ]
  {A B C : α} {ABC : β} {R r p : ℝ}
  (circumradius : R = circumradius ABC)
  (inradius : r = inradius ABC)
  (semiperimeter : p = (side_length A B + side_length B C + side_length C A) / 2) :
  (angle A B C = 90) ↔ (2 * R + r = p) := 
sorry

end triangle_right_iff_two_R_plus_r_eq_p_l368_368441


namespace image_Q_after_rotation_l368_368375

-- Definitions based on conditions
def P := (0, 0 : ℝ × ℝ)
def R := (8, 0 : ℝ × ℝ)
def Q : ℝ × ℝ := ⟨8, 8⟩  -- We determine Q based on the conditions and solution steps.
def angle_QRP : ℝ := 90
def angle_QPR : ℝ := 45

-- Rotation function
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

-- Statement to prove
theorem image_Q_after_rotation :
  rotate90 Q = (-8, 8) :=
sorry

end image_Q_after_rotation_l368_368375


namespace two_distinct_intersections_range_segment_length_f_less_than_g_l368_368224

theorem two_distinct_intersections
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a + b + c = 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + (b-a)x + (c-b) = 0) :=
sorry

theorem range_segment_length
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a + b + c = 0) :
  ∃ x1 x2 : ℝ, (−2 < c/a ∧ c/a < −1/2) ∧ (3/2 < abs (x2 - x1) ∧ abs (x2 - x1) < 2 * sqrt 3) :=
sorry

theorem f_less_than_g
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a + b + c = 0) :
  ∀ x : ℝ, x ≤ - sqrt 3 → ax + b < ax^2 + bx + c :=
sorry

end two_distinct_intersections_range_segment_length_f_less_than_g_l368_368224


namespace parabola_intersection_points_l368_368342

theorem parabola_intersection_points
  (a m : ℝ)
  (hx1 : a * (-1 + m) ^ 2 = 3)
  (hx2 : a * (3 + m) ^ 2 = 3) :
  let y := (λ x, a * (x + m - 2) ^ 2 - 3) in
  (y 5 = 0) ∧ (y 1 = 0) := by
  have h : a = 3 / (m - 1) ^ 2 := 
    by sorry
  let y := (λ x, 3 / (4 * (x - 3) ^ 2 - 3)) -- when m = -1
  sorry

end parabola_intersection_points_l368_368342


namespace problem_1_problem_2_l368_368252

-- Define the functions
def f (x : ℝ) : ℝ := sqrt (x^2 - 2 * x + 1)
def g (x : ℝ) (a : ℝ) : ℝ := - sqrt (x^2 + 6 * x + 9) + a

-- First problem statement: Solve g(x) > 6
theorem problem_1 (a : ℝ) (h : a > 6) : {x : ℝ | g x a > 6} = {x : ℝ | 3 - a < x ∧ x < a - 9} :=
sorry

-- Second problem statement: Find the range of a
def two_f (x : ℝ) : ℝ := 2 * f x

theorem problem_2 (a : ℝ) : (∀ x : ℝ, two_f x > g x a) ↔ a < 4 :=
sorry

end problem_1_problem_2_l368_368252


namespace highest_power_of_5_dividing_S_l368_368206

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def f (n : ℕ) : ℤ :=
  if sum_of_digits n % 2 = 0 then n ^ 100 else -n ^ 100

def S : ℤ :=
  (Finset.range (10 ^ 100)).sum (λ n => f n)

theorem highest_power_of_5_dividing_S :
  ∃ m : ℕ, 5 ^ m ∣ S ∧ ∀ k : ℕ, 5 ^ (k + 1) ∣ S → k < 24 :=
by
  sorry

end highest_power_of_5_dividing_S_l368_368206


namespace total_fish_is_22_l368_368447

def gold_fish : ℕ := 15
def blue_fish : ℕ := 7
def total_fish : ℕ := gold_fish + blue_fish

theorem total_fish_is_22 : total_fish = 22 :=
by
  -- the proof should be written here
  sorry

end total_fish_is_22_l368_368447


namespace condition_sufficient_not_necessary_l368_368736

variable (x : ℝ)
def a := (x, 3)

theorem condition_sufficient_not_necessary (h : ∥a∥ = 5) : (x = 4 → ∥a∥ = 5) ∧ (∥a∥ = 5 → (x = 4 ∨ x = -4)) :=
by
  sorry

end condition_sufficient_not_necessary_l368_368736


namespace quadratic_function_min_value_l368_368507

theorem quadratic_function_min_value (x : ℝ) (y : ℝ) :
  (y = x^2 - 2 * x + 6) →
  (∃ x_min, x_min = 1 ∧ y = (1 : ℝ)^2 - 2 * (1 : ℝ) + 6 ∧ (∀ x, y ≥ x^2 - 2 * x + 6)) :=
by
  sorry

end quadratic_function_min_value_l368_368507


namespace pants_cost_is_250_l368_368988

-- Define the cost of a T-shirt
def tshirt_cost := 100

-- Define the total amount spent
def total_amount := 1500

-- Define the number of T-shirts bought
def num_tshirts := 5

-- Define the number of pants bought
def num_pants := 4

-- Define the total cost of T-shirts
def total_tshirt_cost := tshirt_cost * num_tshirts

-- Define the total cost of pants
def total_pants_cost := total_amount - total_tshirt_cost

-- Define the cost per pair of pants
def pants_cost_per_pair := total_pants_cost / num_pants

-- Proving that the cost per pair of pants is $250
theorem pants_cost_is_250 : pants_cost_per_pair = 250 := by
  sorry

end pants_cost_is_250_l368_368988


namespace pascal_triangle_count_30_rows_l368_368284

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l368_368284


namespace smallest_integer_satisfies_inequality_l368_368059

theorem smallest_integer_satisfies_inequality :
  ∃ (x : ℤ), (x^2 < 2 * x + 3) ∧ ∀ (y : ℤ), (y^2 < 2 * y + 3) → x ≤ y ∧ x = 0 :=
sorry

end smallest_integer_satisfies_inequality_l368_368059
