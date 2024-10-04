import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.EuclideanDomain
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Star.Basic
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.Calculus.SpecificFunctions
import Mathlib.Analysis.Limits
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.SimpleGraph.Coloring
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Gcd
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Irrational
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Seq.Seq
import Mathlib.Data.Set.Basic
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith
import Mathlib.Topology.MetricSpace.Basic

namespace track_length_420_l544_544544

variables (x : ℝ) (brenda_distance_first_meet sally_distance_after_first_meet : ℝ)
  (brenda_const_speed sally_const_speed : Prop)

-- Brenda's first meet distance
def first_meet := brenda_const_speed ∧ brenda_distance_first_meet = 90

-- Sally's distance covered after first meet
def second_meet := sally_const_speed ∧ sally_distance_after_first_meet = 120

-- Prove track length
theorem track_length_420 (h1 : first_meet brenda_distance_first_meet sally_distance_after_first_meet)
    (h2 : second_meet brenda_distance_first_meet sally_distance_after_first_meet) : x = 420 := 
sorry

end track_length_420_l544_544544


namespace hyperbola_equation_1_hyperbola_equation_2_l544_544588

theorem hyperbola_equation_1 :
  ∃ a b: ℝ, a = 15 ∧ b = 5 ∧ (∀ x y: ℝ, 
    (x, y) ∈ { p : ℝ × ℝ | p.1^2 / 16 - p.2^2 / 4 = 1 }
    → (3 * real.sqrt 2, 2) ∈ { p : ℝ × ℝ | p.1^2 / a - p.2^2 / b = 1 })
  ∧ (  ∃ (eq: ∀ x y: ℝ, 
    (x, y) ∈ { p : ℝ × ℝ| (x^2 / a) - (y^2 / b) = 1 } →
            (x^2 / 15) - (y^2/5) = 1
    )
  ) := sorry

theorem hyperbola_equation_2 :
  ∃ λ: ℝ, λ = 8 ∧ ∀ x y: ℝ,
    (x^2 / 8 - y^2 / 2 = 1) ∧ 
    (∀ x y: ℝ, 
        (x^2 / 8 - y^2 / 2 = 1)
        ↔ x^2 - 4 * y^2 = λ ∧ 
           3 * x^2 + 13 * y^2 = 39 ) := sorry

end hyperbola_equation_1_hyperbola_equation_2_l544_544588


namespace find_c_squared_l544_544480

noncomputable def f (a c : ℝ) (z : ℂ) : ℂ := (a + complex.I * c) * z

theorem find_c_squared (a c : ℝ) 
  (h1 : ∀ z : ℂ, complex.abs ((f a c z) - z) = complex.abs (f a c z))
  (h2 : complex.abs (a + complex.I * c) = 5) : 
  c^2 = 24.75 :=
sorry

end find_c_squared_l544_544480


namespace seq1_realizable_seq2_non_realizable_l544_544288

-- Definitions based on the problem conditions
def polygon (n : ℕ) := True -- A polygon with n sides
def has_diagonals (p : polygon 1994) (d : ℕ) := True -- A polygon having d diagonals

-- Statement about the first sequence being realizable
theorem seq1_realizable :
  ∃ (d : ℕ) (p : polygon 1994), has_diagonals p 997 ∧
  ∃ (sequence1 : list ℕ), sequence1.length = 997 ∧
  (∀ x ∈ sequence1, x = 3 → sequence1.count 3 = 991) ∧
  (∀ x ∈ sequence1, x = 2 → sequence1.count 2 = 6) ∧
  sequence1 = list.repeat 3 991 ++ list.repeat 2 6 :=
sorry

-- Statement about the second sequence not being realizable
theorem seq2_non_realizable :
  ∀ (d : ℕ) (p : polygon 1994),
  has_diagonals p 997 →
  ∀ (sequence2 : list ℕ),
  sequence2.length = 997 →
  (∀ x ∈ sequence2, x = 8 → sequence2.count 8 = 4) →
  (∀ x ∈ sequence2, x = 6 → sequence2.count 6 = 985) →
  (∀ x ∈ sequence2, x = 3 → sequence2.count 3 = 8) →
  sequence2 ≠ list.repeat 8 4 ++ list.repeat 6 985 ++ list.repeat 3 8 :=
sorry

end seq1_realizable_seq2_non_realizable_l544_544288


namespace general_circle_vector_form_general_circle_coordinate_form_circle_on_ox_axis_circle_on_oy_axis_circle_center_at_origin_circle_passing_through_origin_l544_544474

-- Definition of a circle in vector form
def circle_vector_form (c : ℝ × ℝ) (R : ℝ) : Prop :=
  ∀ (r : ℝ × ℝ), (r - c)•(r - c) = R * R

-- Definition of a circle in coordinate form
def circle_coordinate_form (a b R : ℝ) : Prop :=
  ∀ (x y : ℝ), (x - a) * (x - a) + (y - b) * (y - b) = R * R

-- Specific case: Center on O_x axis (b = 0)
def circle_ox_axis (a R : ℝ) : Prop :=
  ∀ (x y : ℝ), (x - a) * (x - a) + y * y = R * R

-- Specific case: Center on O_y axis (a = 0)
def circle_oy_axis (b R : ℝ) : Prop :=
  ∀ (x y : ℝ), x * x + (y - b) * (y - b) = R * R

-- Specific case: Center at the origin (a = 0, b = 0)
def circle_origin_centered (R : ℝ) : Prop :=
  ∀ (x y : ℝ), x * x + y * y = R * R

-- Circle passing through the origin
def circle_passing_origin (a b : ℝ) : Prop :=
  a * a + b * b = R * R → ∀ (x y : ℝ), x * x - 2 * a * x + y * y - 2 * b * y = 0

-- Example proving the properties as theorem statements
theorem general_circle_vector_form (c : ℝ × ℝ) (R : ℝ) : circle_vector_form c R :=
  sorry

theorem general_circle_coordinate_form (a b R : ℝ) : circle_coordinate_form a b R :=
  sorry

theorem circle_on_ox_axis (a R : ℝ) : circle_ox_axis a R :=
  sorry

theorem circle_on_oy_axis (b R : ℝ) : circle_oy_axis b R :=
  sorry

theorem circle_center_at_origin (R : ℝ) : circle_origin_centered R :=
  sorry

theorem circle_passing_through_origin (a b R : ℝ) : circle_passing_origin a b :=
  sorry

end general_circle_vector_form_general_circle_coordinate_form_circle_on_ox_axis_circle_on_oy_axis_circle_center_at_origin_circle_passing_through_origin_l544_544474


namespace shortest_path_rectangle_parallelepiped_l544_544293

theorem shortest_path_rectangle_parallelepiped :
  let A := (0, 1, 2)
  let B := (22, 4, 2)
  let x_max := 22
  let y_max := 5
  let z_max := 4
  let coord_planes := [(fun (x y z : ℝ) => x = 0), (fun (x y z : ℝ) => y = 0), (fun (x y z : ℝ) => z = 0)]
  sqrt (24^2 + 9^2) = sqrt 657 := by
  let A := (0, 1, 2) -- point A
  let B := (22, 4, 2) -- point B
  let path_1 := sqrt ((1 + 22 + 1)^2 + (2 + 5 + 2)^2)
  have path_length := path_1
  rw [path_length, sqrt.inj_eq] at *,
  norm_num,
  sorry

end shortest_path_rectangle_parallelepiped_l544_544293


namespace number_of_people_only_went_to_aquarium_is_5_l544_544138

-- Define the conditions
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group_size : ℕ := 10
def total_earnings : ℕ := 240

-- Define the problem in Lean
theorem number_of_people_only_went_to_aquarium_is_5 :
  ∃ x : ℕ, (total_earnings - (group_size * (admission_fee + tour_fee)) = x * admission_fee) → x = 5 :=
by
  sorry

end number_of_people_only_went_to_aquarium_is_5_l544_544138


namespace transistors_in_2002_l544_544350

theorem transistors_in_2002 
  (initial_transistors : ℕ)
  (year_1988 : ℕ)
  (year_2002 : ℕ)
  (years_doubled : ℕ)
  (final_transistors : ℕ)
  (moore_law : ∀ (n : ℕ), initial_transistors * 2^(n / 2) = final_transistors)
  (h_initial : initial_transistors = 500000)
  (h_year_1988 : year_1988 = 1988)
  (h_year_2002 : year_2002 = 2002)
  (h_years_doubled : (year_2002 - year_1988) / 2 = years_doubled)
  (h_final : final_transistors = 64000000)
  (h_years_doubled_calculated : years_doubled = 7)
  : final_transistors = 64000000 :=
by
  have : year_2002 - year_1988 = 14 := by 
    rw [h_year_2002, h_year_1988]
  have : 14 / 2 = 7 := rfl
  have : years_doubled = 7 := by 
    rw [h_years_doubled]
  have : initial_transistors * 2^years_doubled = final_transistors := by
    rw [h_initial, h_years_doubled_calculated]
  have : 500000 * 2^7 = 64000000 := rfl
  exact h_final

end transistors_in_2002_l544_544350


namespace jeffrey_total_steps_l544_544442

def steps_taken (total_distance : ℕ) (first_part_distance : ℕ) (first_pattern : ℕ × ℕ) (second_pattern : ℕ × ℕ) : ℕ :=
  let (f1, b1) := first_pattern in
  let (f2, b2) := second_pattern in
  let steps_first_part := first_part_distance * (f1 + b1) in
  let remaining_distance := total_distance - first_part_distance in
  let steps_second_part := (remaining_distance / (f2 - b2)) * (f2 + b2) in
  steps_first_part + steps_second_part

theorem jeffrey_total_steps : 
  steps_taken 66 30 (3, 2) (4, 1) = 210 :=
  by sorry

end jeffrey_total_steps_l544_544442


namespace no_same_last_four_digits_of_powers_of_five_and_six_l544_544930

theorem no_same_last_four_digits_of_powers_of_five_and_six : 
  ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ (5 ^ n % 10000 = 6 ^ m % 10000) := 
by 
  sorry

end no_same_last_four_digits_of_powers_of_five_and_six_l544_544930


namespace least_sum_possible_l544_544829

theorem least_sum_possible (x y z w k : ℕ) (hpos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w) 
  (hx : 4 * x = k) (hy : 5 * y = k) (hz : 6 * z = k) (hw : 7 * w = k) :
  x + y + z + w = 319 := 
  sorry

end least_sum_possible_l544_544829


namespace common_tangents_of_circles_l544_544222

noncomputable def center (c : ℝ × ℝ) : ℝ × ℝ := c
noncomputable def radius (r : ℝ) : ℝ := r

noncomputable def distance (c1 c2 : ℝ × ℝ) : ℝ := Real.sqrt ((c1.1 - c2.1) ^ 2 + (c1.2 - c2.2) ^ 2)

noncomputable def number_of_common_tangents (c1 c2 : ℝ × ℝ) (r1 r2 d : ℝ) : ℕ :=
if d < r1 + r2 ∧ d > abs (r1 - r2) then 2 else 0

theorem common_tangents_of_circles :
  let c1 := center (1, 2),
      c2 := center (2, 5),
      r1 := radius 1,
      r2 := radius 3,
      d := distance c1 c2 in
  number_of_common_tangents c1 c2 r1 r2 d = 2 :=
by
  let c1 := center (1, 2)
  let c2 := center (2, 5)
  let r1 := radius 1
  let r2 := radius 3
  let d := distance c1 c2
  have h1 : d = Real.sqrt ((2-1)^2 + (5-2)^2) := by sorry
  have h2 : d = Real.sqrt (1 + 9) := by sorry
  have h3 : d = Real.sqrt 10 := by sorry
  have h4 : Real.sqrt 10 < 4 := by sorry
  have h5 : Real.sqrt 10 > 2 := by sorry
  show number_of_common_tangents c1 c2 r1 r2 d = 2 from 
    if_pos (And.intro h4 h5)

end common_tangents_of_circles_l544_544222


namespace range_of_x_range_of_a_l544_544709

-- Problem (1) representation
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (m x : ℝ) : Prop := 1 < m ∧ m < 2 ∧ x = (1 / 2)^(m - 1)

theorem range_of_x (x : ℝ) :
  (∀ m, 1 < m ∧ m < 2 → x = (1 / 2)^(m - 1)) ∧ p (1/4) x →
  1/2 < x ∧ x < 3/4 :=
sorry

-- Problem (2) representation
theorem range_of_a (a : ℝ) :
  (∀ m, 1 < m ∧ m < 2 → ∀ x, x = (1 / 2)^(m - 1) → p a x) →
  1/3 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_x_range_of_a_l544_544709


namespace median_is_fourteen_l544_544290

-- Define the conditions as Lean statements
variables (x y : ℕ)
def data_set := {2, 2 * x, y, 14}

def is_mode (a : ℕ) (s : set ℕ) : Prop :=
  ∀ ⦃b⦄, b ∈ s → a = b → 2 * x = 14 ∨ y = 14

def mean_is_twelve (s : set ℕ) : Prop :=
  (2 + 2 * x + y + 14) / 4 = 12

-- The main theorem
theorem median_is_fourteen (h1 : is_mode 14 data_set) (h2 : mean_is_twelve data_set) : 
  (let sorted := [2, 2 * x, y, 14].qsort (≤) in (sorted.nth 1 + sorted.nth 2) / 2) = 14 :=
sorry

end median_is_fourteen_l544_544290


namespace christina_age_half_in_five_years_l544_544903

theorem christina_age_half_in_five_years (C Y : ℕ) 
  (h1 : C + 5 = Y / 2)
  (h2 : 21 = 3 * C / 5) :
  Y = 80 :=
sorry

end christina_age_half_in_five_years_l544_544903


namespace interest_rate_calculation_l544_544115

theorem interest_rate_calculation :
  ∀ (P : ℝ) (R₁ R₂ : ℝ) (t : ℝ) (gain_B : ℝ),
    P = 1500 → R₁ = 11.5 → t = 3 → gain_B = 67.5 →
    (B_Interest := P * R₁ * t / 100) →
    (A_Interest := B_Interest - gain_B) →
    (R := A_Interest * 100 / (P * t)) →
    R = 10 :=
by
    intros P R₁ R₂ t gain_B hP hR₁ ht hgain_B hB_Interest hA_Interest hR
    sorry

end interest_rate_calculation_l544_544115


namespace length_of_fourth_side_l544_544514

-- Define the conditions
def quadrilateral_inscribed_with_right_angle (AB BC CD : ℝ) (R : ℝ) : Prop :=
  let O := R * real.sqrt 3 in  
  AB = 150 ∧
  BC = 300 ∧
  CD = 150 ∧
  R = 150 * real.sqrt 3 ∧
  ∃ A B C D : ℝ, 
    ∠ B O C = π / 2 ∧ -- Right angle at BOC
  
  AB + BC + CD + AD = 2 * π * O
      
-- The theorem to prove
theorem length_of_fourth_side :
  quadrilateral_inscribed_with_right_angle 150 300 150 (150 * (real.sqrt 3)) →
  ∃ AD, AD = 300 * (real.sqrt 3) :=
sorry

end length_of_fourth_side_l544_544514


namespace hexagon_line_segment_ratio_l544_544822

variables {α : Type*} [add_comm_group α] [vector_space ℝ α] [finite_dimensional ℝ α]

def line_segment_passes_through_center (PQ UV XY : set α) (O : α) : Prop :=
  ∃ P Q, P ∈ UV ∧ Q ∈ XY ∧ (P + Q) / 2 = O

def divides_in_same_ratio (PQ UV XY : set α) : Prop :=
  ∃ P Q, P ∈ UV ∧ Q ∈ XY ∧ ∃ r s : ℝ, 0 < r ∧ r < 1 ∧ P = (1-r) • (sorry : α) + r • (sorry : α) ∧ Q = (1-s) • (sorry : α) + s • (sorry : α) ∧ r = s

theorem hexagon_line_segment_ratio (PQ UV XY : set α) (O : α) :
  line_segment_passes_through_center PQ UV XY O ↔ divides_in_same_ratio PQ UV XY :=
sorry

end hexagon_line_segment_ratio_l544_544822


namespace seven_positive_integers_divide_12n_l544_544595

theorem seven_positive_integers_divide_12n (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, k = 7 ∧ n = some (List.nth [1, 2, 3, 5, 7, 11, 23] k)) → 
  (12 * n) % ((n * (n + 1)) / 2) = 0 := 
by
  sorry

end seven_positive_integers_divide_12n_l544_544595


namespace prime_pairs_satisfying_conditions_l544_544941

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def satisfies_conditions (p q : ℕ) : Prop :=
  (7 * p + 1) % q = 0 ∧ (7 * q + 1) % p = 0

theorem prime_pairs_satisfying_conditions :
  { (p, q) | is_prime p ∧ is_prime q ∧ satisfies_conditions p q } = {(2, 3), (2, 5), (3, 11)} := 
sorry

end prime_pairs_satisfying_conditions_l544_544941


namespace total_money_spent_by_all_12_l544_544735

def total_expenditure (P1 P2 P3 P4 P5 P6 P7 P8 X Y Z W : ℝ) : ℝ :=
  let total_8 := P1 + P2 + P3 + P4 + P5 + P6 + P7 + P8
  let A := (total_8 + (A + X) + Y + Z + W) / 12
  total_8 + A + X + Y + Z + W

theorem total_money_spent_by_all_12
  (P1 P2 P3 P4 P5 P6 P7 P8 X Y Z W : ℝ)
  (total_8 := P1 + P2 + P3 + P4 + P5 + P6 + P7 + P8)
  (A : ℝ := (total_8 + (A + X) + Y + Z + W) / 12)
  :
  total_expenditure P1 P2 P3 P4 P5 P6 P7 P8 X Y Z W = total_8 + A + X + Y + Z + W :=
by
  sorry

end total_money_spent_by_all_12_l544_544735


namespace cyclic_permutation_sum_l544_544299

theorem cyclic_permutation_sum (a : ℕ) (digits : Fin 6 → ℕ)
  (h1 : ∀ i : Fin 6, 0 < digits i)
  (h2 : ∑ i, digits i = 18)
  (h3 : a = ∑ i, digits i * 10 ^ (5 - i : ℕ)) :
  let cyclic_permutations_sum := (∑ i, digits i) * (1 + 10 + 10^2 + 10^3 + 10^4 + 10^5)
  in cyclic_permutations_sum = 1999998 := by sorry

end cyclic_permutation_sum_l544_544299


namespace find_length_PM_l544_544285

noncomputable def length_PM (P Q R M : Point)
  (hPQ : dist P Q = 34)
  (hQR : dist Q R = 34)
  (hPR : dist P R = 32)
  (hM_midpoint : midpoint Q R M) : ℝ :=
  dist P M

theorem find_length_PM (P Q R M : Point)
  (hPQ : dist P Q = 34)
  (hQR : dist Q R = 34)
  (hPR : dist P R = 32)
  (hM_midpoint : midpoint Q R M) :
  length_PM P Q R M hPQ hQR hPR hM_midpoint = 3 * (Real.sqrt 89) :=
  sorry

end find_length_PM_l544_544285


namespace find_certain_number_l544_544849

theorem find_certain_number (x : ℝ) (h : ((7 * (x + 5)) / 5) - 5 = 33) : x = 22 :=
by
  sorry

end find_certain_number_l544_544849


namespace similar_triangle_perimeter_l544_544141

theorem similar_triangle_perimeter (a b c d : ℝ)
  (h1 : a = 8) (h2 : b = 8) (h3 : c = 10) (h4 : d = 25)
  (h5 : a = b) (h6 : ∃ k, c = k * d ∧ a = k * a ∧ b = k * b) :
  let kato : ℝ := a + b + c,
    perim_ratio : ℝ := d / c,
    large_perim : ℝ := perim_ratio * kato in
  large_perim = 65 :=
sorry

end similar_triangle_perimeter_l544_544141


namespace ellipse_standard_eq_PA_plus_PB_const_l544_544977

/-
  Given an ellipse \( C \) with the equation \( \frac{x^2}{a^2} + \frac{y^2}{b^2} = 1 \), where \( a > b \) and \( 0 < b < 2 \),
  left and right foci \( F_1 \) and \( F_2 \),
  point \( M \) lies on the ellipse such that \( MF_2 \perp F_1F_2 \),
  the perimeter of \( \triangle MF_1F_2 \) is \( 6 \), and its area is \( \frac{3}{2} \),
  and a line \( l \) passing through point \( F_2 \) intersects the ellipse at points \( A \) and \( B \),
  and intersects the \( y \)-axis at point \( P \).

  Prove:
  (1) The standard equation of the ellipse \( C \) is \( \frac{x^2}{4} + \frac{y^2}{3} = 1 \).
  (2) \( \lambda_1 + \lambda_2 = -\frac{8}{3} \), where \( \overrightarrow{PA} = \lambda_1 \overrightarrow{AF_2} \) and \( \overrightarrow{PB} = \lambda_2 \overrightarrow{BF_2} \).
-/

theorem ellipse_standard_eq (a b : ℝ) (h1 : a > b) (h2 : 0 < b ∧ b < 2)
    (F₁ F₂ : ℝ × ℝ) (M : ℝ × ℝ)
    (h3 : M ∈ set_of (λ p: ℝ × ℝ, (p.1^2 / a^2) + (p.2^2 / b^2) = 1))
    (h4 : 6 = dist M F₁ + dist M F₂ + dist F₁ F₂)
    (h5 : 3 / 2 = abs (1 / 2 * (M.1 * (F₁.2 - F₂.2) + M.2 * (F₂.1 - F₁.1)))) :
    (x y : ℝ) (hx : (x, y) ∈ set_of (λ p: ℝ × ℝ, (p.1^2 / 4) + (p.2^2 / 3) = 1)) :
    (x, y) ∈ set_of (λ p: ℝ × ℝ, (p.1^2 / a^2) + (p.2^2 / b^2) = 1) :=
sorry

theorem PA_plus_PB_const (F₂ : ℝ × ℝ) (A B P : ℝ × ℝ)
    (h1 : ∃ λ₁ : ℝ, ∃ λ₂ : ℝ, P = (0, -k) ∧ λ₁ * (1 - A.1) = A.1 ∧ λ₂ * (1 - B.1) = B.1) :
    λ₁ + λ₂ = -8 / 3 :=
sorry

end ellipse_standard_eq_PA_plus_PB_const_l544_544977


namespace geometric_seq_A_not_geometric_seq_B_geometric_seq_C_not_geometric_seq_D_l544_544819

-- Define a function to check if a sequence is geometric
def is_geometric (seq : Stream ℝ) : Prop :=
  ∃ r, ∃ a, seq = Stream.iterate (λ x, x * r) a

-- Define sequences according to the problem
def seqA : Stream ℝ := Stream.constant 1
def seqB : Stream ℝ := Stream.constant 0
def seqC : Stream ℝ := Stream.iterate (λ x, x * (1/2)) (1/2)
def seqD : Stream ℝ := Stream.of_list [-1, -1, 1, -1]

-- The propositions to prove
theorem geometric_seq_A : is_geometric seqA := sorry
theorem not_geometric_seq_B : ¬ is_geometric seqB := sorry
theorem geometric_seq_C : is_geometric seqC := sorry
theorem not_geometric_seq_D : ¬ is_geometric seqD := sorry

end geometric_seq_A_not_geometric_seq_B_geometric_seq_C_not_geometric_seq_D_l544_544819


namespace two_digit_numbers_with_5_as_second_last_digit_l544_544178

theorem two_digit_numbers_with_5_as_second_last_digit:
  ∀ N : ℕ, (10 ≤ N ∧ N ≤ 99) → (∃ k : ℤ, (N * k) % 100 / 10 = 5) ↔ ¬(N % 20 = 0) :=
by
  sorry

end two_digit_numbers_with_5_as_second_last_digit_l544_544178


namespace total_pigs_in_barn_l544_544038

-- Define the number of pigs initially in the barn
def initial_pigs : ℝ := 2465.25

-- Define the number of pigs that join
def joining_pigs : ℝ := 5683.75

-- Define the total number of pigs after they join
def total_pigs : ℝ := 8149

-- The theorem that states the total number of pigs is the sum of initial and joining pigs
theorem total_pigs_in_barn : initial_pigs + joining_pigs = total_pigs := 
by
  sorry

end total_pigs_in_barn_l544_544038


namespace series_convergence_l544_544339

noncomputable def digits_base (b n : ℕ) : ℕ := (Int.toNat (Int.floor (Real.log n / Real.log b))) + 1

def f (b : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 2
  | k+3 => k+3 * f b (digits_base b (k+3))

theorem series_convergence (b : ℕ) (h : b > 1) : (∀ u, real.summable (λ n : ℕ, 1 / (f b n.toReal)) ↔ b = 2) :=
by
  sorry

end series_convergence_l544_544339


namespace polynomial_remainder_is_correct_l544_544030

theorem polynomial_remainder_is_correct :
  ∃ (R : Polynomial ℝ), (∃ Q : Polynomial ℝ, (Polynomial.X ^ 50) = (Polynomial.X^2 - 5 * Polynomial.X + 6) * Q + R) ∧ R.degree < 2 ∧ 
                        R = (Polynomial.C (3^50 - 2^50)) * Polynomial.X + Polynomial.C (2^50 - 2 * 3^50 + 2 * 2^50) :=
begin
  -- The proof goes here.
  sorry
end

end polynomial_remainder_is_correct_l544_544030


namespace anna_age_l544_544191

-- Define the conditions as given in the problem
variable (x : ℕ)
variable (m n : ℕ)

-- Translate the problem statement into Lean
axiom perfect_square_condition : x - 4 = m^2
axiom perfect_cube_condition : x + 3 = n^3

-- The proof problem statement in Lean 4
theorem anna_age : x = 5 :=
by
  sorry

end anna_age_l544_544191


namespace sequence_a3_equals_1_over_3_l544_544304

theorem sequence_a3_equals_1_over_3 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n = 1 - 1 / (a (n - 1) + 1)) : 
  a 3 = 1 / 3 :=
sorry

end sequence_a3_equals_1_over_3_l544_544304


namespace average_total_goals_l544_544551

theorem average_total_goals (carter_avg shelby_avg judah_avg total_avg : ℕ) 
    (h1: carter_avg = 4) 
    (h2: shelby_avg = carter_avg / 2)
    (h3: judah_avg = 2 * shelby_avg - 3) 
    (h4: total_avg = carter_avg + shelby_avg + judah_avg) :
  total_avg = 7 :=
by
  sorry

end average_total_goals_l544_544551


namespace range_of_t_l544_544765

noncomputable def point := ℝ × ℝ

def ellipse (a b : ℝ) (p : point) : Prop := (p.1 ^ 2) / a ^ 2 + (p.2 ^ 2) / b ^ 2 = 1

def intersects_ellipse (a b : ℝ) (x0 : ℝ) (k : ℝ) : Prop :=
  ∃ p1 p2 : point, ellipse a b p1 ∧ ellipse a b p2 ∧ p1.2 = k * (p1.1 - x0) ∧ p2.2 = k * (p2.1 - x0)

def linear_combination (A B P : point) (t : ℝ) : Prop :=
  A.1 + B.1 = t * P.1 ∧ A.2 + B.2 = t * P.2

def point_distance (A B : point) : ℝ :=
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2

theorem range_of_t :
  let a := √2
  let b := 1
  ∀ x k t : ℝ,
  (k ≠ 0) →
  intersects_ellipse a b x k →
  (∃ A B P : point, ellipse a b P ∧ linear_combination A B P t ∧ point_distance P A - point_distance P B < (2 * √5) / 3) →
  -2 < t ∧ t < - (2 * √6) / 3 ∨ (2 * √6) / 3 < t ∧ t < 2 :=
sorry

end range_of_t_l544_544765


namespace subset_exists_diff_96_l544_544192

theorem subset_exists_diff_96
  (S : Finset ℕ)
  (h_card_S : S.card = 1996)
  (h_S_subset : S ⊆ Finset.range 3840) :
  ∃ a b ∈ S, a ≠ b ∧ (a - b).natAbs = 96 :=
sorry

end subset_exists_diff_96_l544_544192


namespace allan_initial_balloons_l544_544529

theorem allan_initial_balloons (A : ℕ) (jak : ℕ) 
  (allan_extra : ℕ) (jake_more : ℕ) 
  (jak_val : jak = 6) (allan_extra_val : allan_extra = 3) 
  (jake_more_val : jake_more = 1) 
  (jake_eq_allan : jak = A + allan_extra + jake_more):
  A = 2 :=
by
  have h1 : 6 = A + 3 + 1 := by rw [jak_val, allan_extra_val, jake_more_val, jake_eq_allan]
  linarith


end allan_initial_balloons_l544_544529


namespace ajay_spends_7_5_percent_on_medicines_l544_544883

variable (monthly_income : ℕ)
variable (savings : ℕ)
variable (household_percentage : ℝ)
variable (clothes_percentage : ℝ)
variable (medicine_spending_percentage : ℝ)

-- Conditions
def conditions (monthly_income : ℕ) (savings : ℕ) (household_percentage : ℝ) (clothes_percentage : ℝ) : Prop :=
  monthly_income = 40000 ∧
  savings = 9000 ∧
  household_percentage = 0.45 ∧
  clothes_percentage = 0.25

-- Question
def ajay_spends_on_medicines (monthly_income : ℕ) (savings : ℕ) (household_percentage : ℝ) (clothes_percentage : ℝ) : Prop :=
  medicine_spending_percentage = 0.075

-- Problem Statement
theorem ajay_spends_7_5_percent_on_medicines (monthly_income : ℕ) (savings : ℕ) (household_percentage : ℝ) (clothes_percentage : ℝ) :
  conditions monthly_income savings household_percentage clothes_percentage → 
  ajay_spends_on_medicines monthly_income savings household_percentage clothes_percentage :=
sorry

end ajay_spends_7_5_percent_on_medicines_l544_544883


namespace track_width_proof_l544_544517

noncomputable def track_width (r1 r2 : ℝ) : ℝ := r1 - r2

theorem track_width_proof (r1 r2 : ℝ) (h1 : 2 * real.pi * r1 - 2 * real.pi * r2 = 40 * real.pi)
                          (h2 : 2 * real.pi * r1 = 160 * real.pi) : 
  track_width r1 r2 = 20 := 
by
  -- The proof would follow here
  sorry

end track_width_proof_l544_544517


namespace avg_goals_per_game_l544_544554

def carter_goals_per_game := 4
def shelby_goals_per_game := carter_goals_per_game / 2
def judah_goals_per_game := (2 * shelby_goals_per_game) - 3
def average_total_goals_team := carter_goals_per_game + shelby_goals_per_game + judah_goals_per_game

theorem avg_goals_per_game : average_total_goals_team = 7 :=
by
  -- Proof would go here
  sorry

end avg_goals_per_game_l544_544554


namespace smallest_n_interval_l544_544786

theorem smallest_n_interval :
  ∃ n : ℕ, (∃ x : ℤ, ⌊10 ^ n / x⌋ = 2006) ∧ 7 ≤ n ∧ n ≤ 12 :=
sorry

end smallest_n_interval_l544_544786


namespace amount_spent_on_milk_l544_544882

-- Define conditions
def monthly_salary (S : ℝ) := 0.10 * S = 1800
def rent := 5000
def groceries := 4500
def education := 2500
def petrol := 2000
def miscellaneous := 700
def total_expenses (S : ℝ) := S - 1800
def known_expenses := rent + groceries + education + petrol + miscellaneous

-- Define the proof problem
theorem amount_spent_on_milk (S : ℝ) (milk : ℝ) :
  monthly_salary S →
  total_expenses S = known_expenses + milk →
  milk = 1500 :=
by
  sorry

end amount_spent_on_milk_l544_544882


namespace problem1_problem2_l544_544835

-- Problem 1: (-3xy)² * 4x² = 36x⁴y²
theorem problem1 (x y : ℝ) : ((-3 * x * y) ^ 2) * (4 * x ^ 2) = 36 * x ^ 4 * y ^ 2 := by
  sorry

-- Problem 2: (x + 2)(2x - 3) = 2x² + x - 6
theorem problem2 (x : ℝ) : (x + 2) * (2 * x - 3) = 2 * x ^ 2 + x - 6 := by
  sorry

end problem1_problem2_l544_544835


namespace mixed_tea_sale_price_l544_544128

theorem mixed_tea_sale_price (cost1 cost2 : ℕ) (weight1 weight2 : ℕ) (profit_percent : ℕ) 
  (h1 : cost1 = 15) (h2 : weight1 = 80) (h3 : cost2 = 20) (h4 : weight2 = 20) (h5 : profit_percent = 25) :
  let total_cost := (weight1 * cost1) + (weight2 * cost2) in
  let total_weight := weight1 + weight2 in
  let cost_price_per_kg := total_cost / total_weight in
  let profit_per_kg := profit_percent * cost_price_per_kg / 100 in
  let sale_price_per_kg := cost_price_per_kg + profit_per_kg in
  sale_price_per_kg = 20 :=
by
  sorry

end mixed_tea_sale_price_l544_544128


namespace cone_lateral_surface_area_l544_544243

theorem cone_lateral_surface_area
  (slant_height : ℝ)
  (lateral_surface_shape : String)
  (h_slant_height : slant_height = 10)
  (h_lateral_surface_shape : lateral_surface_shape = "semicircle") :
  lateral_surface_area slant_height lateral_surface_shape = 50 * Real.pi :=
by
  sorry

-- We need to define lateral_surface_area according to the problem conditions
def lateral_surface_area (slant_height : ℝ) (shape : String) : ℝ :=
  if shape = "semicircle" then
    (1 / 2) * Real.pi * (slant_height ^ 2)
  else
    0

end cone_lateral_surface_area_l544_544243


namespace max_sqrt3x_add_sqrt2y_eq_2sqrt5_min_3divx_add_2divy_eq_5div2_l544_544619

theorem max_sqrt3x_add_sqrt2y_eq_2sqrt5
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 2 * y = 10) : 
  sqrt (3 * x) + sqrt (2 * y) ≤ 2 * sqrt 5 :=
sorry

theorem min_3divx_add_2divy_eq_5div2
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 2 * y = 10) :
  5 / 2 ≤ 3 / x + 2 / y :=
sorry

end max_sqrt3x_add_sqrt2y_eq_2sqrt5_min_3divx_add_2divy_eq_5div2_l544_544619


namespace dress_designs_count_l544_544479

theorem dress_designs_count :
  let colors := 5
  let patterns := 4
  let sizes := 3
  colors * patterns * sizes = 60 :=
by
  let colors := 5
  let patterns := 4
  let sizes := 3
  have h : colors * patterns * sizes = 60 := by norm_num
  exact h

end dress_designs_count_l544_544479


namespace composite_prime_fraction_l544_544729

theorem composite_prime_fraction :
  let P1 : ℕ := 4 * 6 * 8 * 9 * 10 * 12 * 14 * 15
  let P2 : ℕ := 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26
  let first_prime : ℕ := 2
  let second_prime : ℕ := 3
  (P1 + first_prime) / (P2 + second_prime) =
    (4 * 6 * 8 * 9 * 10 * 12 * 14 * 15 + 2) / (16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 + 3) := by
  sorry

end composite_prime_fraction_l544_544729


namespace kevin_initial_cards_l544_544697

theorem kevin_initial_cards (found_cards : ℕ) (total_cards : ℕ) (h1 : found_cards = 47) (h2 : total_cards = 54) : total_cards - found_cards = 7 := 
by
  rw [h1, h2]
  norm_num
  done

end kevin_initial_cards_l544_544697


namespace book_weight_l544_544471

theorem book_weight (total_weight : ℕ) (num_books : ℕ) (each_book_weight : ℕ) 
  (h1 : total_weight = 42) (h2 : num_books = 14) :
  each_book_weight = total_weight / num_books :=
by
  sorry

end book_weight_l544_544471


namespace expected_value_is_correct_l544_544447

noncomputable def expected_value_unfair_die : ℝ := 
  let P : ℕ → ℝ := λ n, if n = 8 then 1/3 else if 1 ≤ n ∧ n ≤ 4 then 1/15 else if 5 ≤ n ∧ n ≤ 7 then 2/15 else 0
  ∑ n in finset.range 9, n * P n

theorem expected_value_is_correct : expected_value_unfair_die = 7.7333 := by
  sorry

end expected_value_is_correct_l544_544447


namespace arithmetic_sequence_sum_l544_544323

open_locale classical

noncomputable def sum_arithmetic (n : ℕ) (a1 aN : ℝ) : ℝ :=
  (n / 2) * (a1 + aN)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = sum_arithmetic n (a 1) (a n))
  (h2 : sum_arithmetic 1009 (a 1) (a 2017) = 2018) :
  S 2017 = 4034 :=
by
  sorry

end arithmetic_sequence_sum_l544_544323


namespace roots_numerically_equal_opposite_signs_l544_544654

theorem roots_numerically_equal_opposite_signs
  (a b c : ℝ) (k : ℝ)
  (h : (∃ x : ℝ, x^2 - (b+1) * x ≠ 0) →
    ∃ x : ℝ, x ≠ 0 ∧ x ∈ {x : ℝ | (k+2)*(x^2 - (b+1)*x) = (k-2)*((a+1)*x - c)} ∧ -x ∈ {x : ℝ | (k+2)*(x^2 - (b+1)*x) = (k-2)*((a+1)*x - c)}) :
  k = (-2 * (b - a)) / (b + a + 2) :=
by
  sorry

end roots_numerically_equal_opposite_signs_l544_544654


namespace angle_A_triangle_area_l544_544231

-- Part (1)
theorem angle_A (a b c A : ℝ) (h_eq : (b - c)^2 = a^2 - bc) 
  (h_triangle : ∀ (a b c A : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ A < π) : A = π / 3 :=
sorry

-- Part (2)
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (h_a : a = 3) 
  (h_sin : sin C = 2 * sin B) (h_bc_eq : (b - c)^2 = a^2 - bc)
  (h_A : A = π / 3) 
  (h_triangle : ∀ (a b c A B C : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π) :
  1/2 * b * c * sin A = 3 * sqrt 3 / 2 :=
sorry

end angle_A_triangle_area_l544_544231


namespace flyDistanceCeiling_l544_544674

variable (P : ℝ × ℝ × ℝ)
variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- Defining the conditions
def isAtRightAngles (P : ℝ × ℝ × ℝ) : Prop :=
  P = (0, 0, 0)

def distanceFromWall1 (x : ℝ) : Prop :=
  x = 2

def distanceFromWall2 (y : ℝ) : Prop :=
  y = 5

def distanceFromPointP (x y z : ℝ) : Prop :=
  7 = Real.sqrt (x^2 + y^2 + z^2)

-- Proving the distance from the ceiling
theorem flyDistanceCeiling (P : ℝ × ℝ × ℝ) (x y z : ℝ) :
  isAtRightAngles P →
  distanceFromWall1 x →
  distanceFromWall2 y →
  distanceFromPointP x y z →
  z = 2 * Real.sqrt 5 := 
sorry

end flyDistanceCeiling_l544_544674


namespace friend_gain_percentage_l544_544863

variable (original_cost_price man_selling_price friend_selling_price gain friend_cost_price : ℝ)

-- Definitions from conditions
def original_cost_price := 52941.17647058824
def man_loss := 0.15 * original_cost_price
def man_selling_price := original_cost_price - man_loss
def friend_selling_price := 54000

-- Theorem statement to prove the gain percentage for the friend
theorem friend_gain_percentage :
  (friend_selling_price - man_selling_price) / man_selling_price * 100 = 3.55 :=
by
  -- Given definitions
  let gain := friend_selling_price - man_selling_price
  let gain_percentage := (gain / man_selling_price) * 100
  have h1: friend_selling_price = 54000 := by rfl
  have h2: original_cost_price = 52941.17647058824 := by rfl
  have h3: man_loss = 0.15 * original_cost_price := by rfl
  have h4: man_selling_price = original_cost_price - man_loss := by rfl
  have h5: gain = friend_selling_price - man_selling_price := by rfl
  have h6: gain_percentage = (gain / man_selling_price) * 100 := by rfl
  -- Prove the final equality
  sorry

end friend_gain_percentage_l544_544863


namespace cos_double_angle_l544_544228

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + 3 * Real.pi / 2) = Real.sqrt 3 / 3) : Real.cos (2 * α) = -1 / 3 := 
  sorry

end cos_double_angle_l544_544228


namespace magician_can_identify_matching_coin_l544_544501

-- Define the types and conditions
variable (coins : Fin 11 → Bool) -- Each coin is either heads (true) or tails (false).
variable (uncovered_index : Fin 11) -- The index of the initially uncovered coin.

-- Define the property to be proved: the magician can point to an adjacent coin with the same face as the uncovered one.
theorem magician_can_identify_matching_coin (h : ∃ i : Fin 11, coins i = coins (i + 1) % 11) :
  ∃ j : Fin 11, (j ≠ uncovered_index) ∧ coins j = coins uncovered_index :=
  sorry

end magician_can_identify_matching_coin_l544_544501


namespace simplify_fraction_mult_l544_544751

theorem simplify_fraction_mult (h : \dfrac{150}{6000} * 75 = \dfrac{15}{8}) : True :=
by sorry

end simplify_fraction_mult_l544_544751


namespace angle_AXC_angle_ACB_l544_544032

-- Definitions of the problem conditions
variables (A B C D X : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty X]
variables (AD DC: Type) [Nonempty AD] [Nonempty DC]
variables (angleB angleXDC angleAXC angleACB : ℝ)
variables (AB BX: ℝ)

-- Given conditions
axiom equal_sides: AD = DC
axiom pointX: BX = AB
axiom given_angleB: angleB = 34
axiom given_angleXDC: angleXDC = 52

-- Proof goals (no proof included, only the statements)
theorem angle_AXC: angleAXC = 107 :=
sorry

theorem angle_ACB: angleACB = 47 :=
sorry

end angle_AXC_angle_ACB_l544_544032


namespace number_of_positive_integers_satisfying_inequality_l544_544575

theorem number_of_positive_integers_satisfying_inequality :
  {n : ℤ | 0 < n ∧ (n + 10) * (n - 2) * (n - 15) < 0}.to_finset.card = 12 :=
sorry

end number_of_positive_integers_satisfying_inequality_l544_544575


namespace product_inequality_l544_544205

variable {n : ℕ}
variable (x : Fin n → ℝ) (m : ℝ) (a s : ℝ)

theorem product_inequality (hx_pos : ∀ i, 0 < x i)
  (hm_pos : 0 < m) (ha_ge_zero : 0 ≤ a) (hx_sum : (Finset.univ.sum x) = s)
  (hs_le_n : s ≤ n) :
  (Finset.univ.prod (λ i, (x i)^m + (1 / (x i)^m) + a)) ≥ 
  ((s / n)^m + (n / s)^m + a)^n := sorry

end product_inequality_l544_544205


namespace number_of_integers_with_D_eq_three_l544_544953

def D (n : ℕ) : ℕ :=
  (n.bits.zip (n.bits.drop 1)).count (λ p => p.1 ≠ p.2)

theorem number_of_integers_with_D_eq_three :
  (Nat.filter (λ n => D n = 3) (List.range' 1 200)).length = 9 := 
sorry

end number_of_integers_with_D_eq_three_l544_544953


namespace solve_trig_equation_l544_544249

theorem solve_trig_equation (a : ℝ) :
  (∃ x : ℝ, sqrt 3 * sin x + cos x = 2 * a - 1) ↔ -1 / 2 ≤ a ∧ a ≤ 3 / 2 :=
by sorry

end solve_trig_equation_l544_544249


namespace find_abs_c_l544_544908

theorem find_abs_c (a b c : ℤ) (h1 : gcd a b = 1) (h2 : gcd b c = 1) (h3 : gcd a c = 1)
  (h4 : a * (3 - complex.i)^4 + b * (3 - complex.i)^3 + c * (3 - complex.i)^2 + b * (3 - complex.i) + a = 0) :
  |c| = 109 := sorry

end find_abs_c_l544_544908


namespace cat_does_not_eat_20_cells_fish_l544_544011

theorem cat_does_not_eat_20_cells_fish
  (eat_rate: ℤ)
  (total_fish: ℤ)
  (total_cells: ℤ)
  (fish_cells: fin 6 → ℤ)
  (remains: (fin 6 → Prop)) :
  eat_rate = 3 →
  total_fish = 5 →
  total_cells = 68 →
  (∀ i, remains i ↔ fish_cells i % 3 = 2) →
  (∑ i in (fin 6).attach.filter (λ i, ¬remains i), fish_cells i) = total_cells - fish_cells ⟨5, sorry⟩ →
  (∑ i in (fin 6).attach.filter (λ i, ¬remains i), fish_cells i) % 3 = 0 →
  remains ⟨5, sorry⟩ :=
by
  sorry

end cat_does_not_eat_20_cells_fish_l544_544011


namespace two_percent_as_decimal_l544_544266

-- Definition for converting percentage to decimal
def percentage_to_decimal (p : ℝ) : ℝ := p / 100

-- The specific proof problem
theorem two_percent_as_decimal : percentage_to_decimal 2 = 0.02 := by 
  -- adding sorry to skip the actual proof steps
  sorry 

end two_percent_as_decimal_l544_544266


namespace charming_number_unique_l544_544104

theorem charming_number_unique :
  ∃! (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (10 * a + b = 2 * a + b^3) :=
begin
  use [3, 3],
  simp,
  split,
  {
    split,
    { exact dec_trivial, },
    { exact nat.succ_le_succ (nat.zero_le 3), }
  },
  {
    split,
    { exact dec_trivial, },
    {
      norm_num,
    }
  },
  split,
  { norm_num, },
  intros y hy,
  cases hy with hy1 hy2,
  rcases hy1 with ⟨a1, b1, h1, h2, h3⟩,
  rcases hy2 with ⟨ha1, hb1, he1⟩,
  norm_num at h3 h2,
  assumption,
  cases h2,
end

end charming_number_unique_l544_544104


namespace marble_problem_l544_544107

theorem marble_problem : 
  ∃ T : ℕ, 
    (∀ p : ℚ, (p = ((T - 12) / T) * ((T - 12) / T) → p = 36/49)) → 
    T = 84 := 
by
  -- Definitions
  let T := 84,
  assume h,
  have prob := h ((T - 12) / T * (T - 12) / T),
  sorry -- Skipping the full proof

end marble_problem_l544_544107


namespace rotate_segments_l544_544302

-- Define the points A, B, C, D as elements in the plane
variables {A B C D P : ℝ × ℝ}

-- Define the lengths of the segments AB and CD
def segment_length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Condition: The segments AB and CD are of equal length
axiom equal_length : segment_length A B = segment_length C D

-- Question: Show there exists a point P such that a plane rotation around P maps segment CD to AB
theorem rotate_segments (A B C D : ℝ × ℝ) :
  ∃ P, ∃ θ : ℝ, ∀ x : ℝ × ℝ, 
    (x = C ∨ x = D) → 
    (cos θ * (x.fst - P.fst) - sin θ * (x.snd - P.snd) + P.fst = x.fst) ∧ 
    (sin θ * (x.fst - P.fst) + cos θ * (x.snd - P.snd) + P.snd = x.snd) :=
  sorry

end rotate_segments_l544_544302


namespace toaster_sales_promotion_l544_544913

theorem toaster_sales_promotion :
  ∀ (p : ℕ) (c₁ c₂ : ℕ) (k : ℕ), 
    (c₁ = 600 ∧ p = 15 ∧ k = p * c₁) ∧ 
    (c₂ = 450 ∧ (p * c₂ = k) ) ∧ 
    (p' = p * 11 / 10) →
    p' = 22 :=
by 
  sorry

end toaster_sales_promotion_l544_544913


namespace right_triangle_condition_l544_544825

variables {R r : ℝ} (p : ℝ) (is_right_triangle : Bool)
def semi_perimeter (a b c : ℝ) := (a + b + c) / 2
def circumradius (a b c : ℝ) := c / 2 -- assuming c is the hypotenuse
def inradius (a b c : ℝ) := (a + b - c) / 2 -- simplified formula for right triangle

theorem right_triangle_condition (a b c : ℝ) (h : p = 2 * circumradius a b c + inradius a b c) :
  is_right_triangle = true :=
sorry

end right_triangle_condition_l544_544825


namespace angle_A_is_pi_over_4_l544_544975

theorem angle_A_is_pi_over_4
  (A B C : ℝ)
  (a b c : ℝ)
  (h : a^2 = b^2 + c^2 - 2 * b * c * Real.sin A) :
  A = Real.pi / 4 :=
  sorry

end angle_A_is_pi_over_4_l544_544975


namespace work_completion_days_l544_544454

theorem work_completion_days :
  let x_work_rate := (1 : ℝ) / 15,
      y_work_rate := (1 : ℝ) / 10,
      total_payment := 720,
      z_payment := 120.00000000000001,
      z_work_rate := z_payment / total_payment
  in x_work_rate + y_work_rate = (1 / 6) → (x_work_rate + y_work_rate + z_work_rate = 1 / 3) → 1 / (x_work_rate + y_work_rate + z_work_rate) = 3 :=
by
  intros x_work_rate y_work_rate z_payment z_work_rate
  intros hx hy
  sorry

end work_completion_days_l544_544454


namespace part_1_part_2_part_3_l544_544252

def f (x : ℝ) : ℝ := sin x ^ 2 - cos x ^ 2 - 2 * sqrt 3 * sin x * cos x

theorem part_1 : f (2 * π / 3) = 2 := 
sorry

theorem part_2 : ∀ x : ℝ, f (x + π) = f x := 
sorry

theorem part_3 : ∀ k : ℤ, ∀ x : ℝ, 
  k * π - 2 * π / 3 ≤ x ∧ x ≤ k * π - π / 3 → 
  2 * sin (2 * x + 7 * π / 6) < 2 * sin (2 * (x + 1) + 7 * π / 6) := 
sorry

end part_1_part_2_part_3_l544_544252


namespace periodicity_of_f_shift_plus_2_l544_544638

noncomputable def f (x : ℝ) : ℝ := 
  let k : ℤ := ⌊(x + 1) / 2⌋;
  (-1)^k * sqrt (1 - (x - 2 * k)^2)

theorem periodicity_of_f (x : ℝ) : f (x + 4) = f x :=
sorry

theorem shift_plus_2 (x : ℝ) : f (x + 2) + f x = 0 :=
sorry

end periodicity_of_f_shift_plus_2_l544_544638


namespace H_is_orthocenter_circumradius_is_R_l544_544041

noncomputable theory

variable (R : ℝ)
variable (H A B C : ℝ × ℝ) -- Points in the plane

-- Assumptions about the circles and their intersections
variable (circle1 circle2 circle3 : ℝ × ℝ → Prop) -- circle equations

-- h is a point on all three circles
axiom h_on_circle1 : circle1 H
axiom h_on_circle2 : circle2 H
axiom h_on_circle3 : circle3 H

-- Points of pairwise intersection, different from H
axiom a_intersection : circle1 A ∧ circle2 A ∧ A ≠ H
axiom b_intersection : circle2 B ∧ circle3 B ∧ B ≠ H
axiom c_intersection : circle3 C ∧ circle1 C ∧ C ≠ H

-- Definitions of orthocenter and circumradius
def is_orthocenter (H A B C : ℝ × ℝ) :=
  ∃ X Y Z : ℝ × ℝ, 
    X = perpendicular_from H A B ∧
    Y = perpendicular_from H B C ∧
    Z = perpendicular_from H C A ∧
    ∀ T : ℝ × ℝ, T = X ∨ T = Y ∨ T = Z ↔ T = H

def circumradius (A B C : ℝ × ℝ) : ℝ := R

-- Theorems to prove
theorem H_is_orthocenter : is_orthocenter H A B C := sorry

theorem circumradius_is_R : circumradius A B C = R := sorry

end H_is_orthocenter_circumradius_is_R_l544_544041


namespace nearest_integer_3_add_sqrt_5_pow_6_l544_544061

noncomputable def approx (x : ℝ) : ℕ := Real.floor (x + 0.5)

theorem nearest_integer_3_add_sqrt_5_pow_6 : 
  approx ((3 + Real.sqrt 5)^6) = 22608 :=
by
  -- Proof omitted, sorry
  sorry

end nearest_integer_3_add_sqrt_5_pow_6_l544_544061


namespace exponent_property_l544_544197

theorem exponent_property (a x y : ℝ) (h1 : 0 < a) (h2 : a ^ x = 2) (h3 : a ^ y = 3) : a ^ (x - y) = 2 / 3 := 
by
  sorry

end exponent_property_l544_544197


namespace ellipse_trace_l544_544013

theorem ellipse_trace
  (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 2 * Real.pi)
  (w : ℂ) (hw : w = 3 * Complex.exp(θ * Complex.I)) :
  ∃ (a b : ℝ), a = 29 / 9 ∧ b = 25 / 9 ∧ 
  ∀ (x y : ℝ), (x = (a * Real.cos θ) ∧ y = (b * Real.sin θ)) 
    → ((x / a)^2 + (y / b)^2 = 1) :=
sorry

end ellipse_trace_l544_544013


namespace probability_abc_eq_72_l544_544083

theorem probability_abc_eq_72 : 
  let die_results := ({1, 2, 3, 4, 5, 6} : Finset ℕ)
  in (∃ (a b c : ℕ), (a ∈ die_results) ∧ (b ∈ die_results) ∧ (c ∈ die_results) ∧ (a * b * c = 72)) → 
     (↑(die_results.card) ^ 3 = 216) →
     (∃ (count_abc_eq_72 : ℕ), count_abc_eq_72 = 9) →
     (count_abc_eq_72 / 216 = (1 : ℚ) / 24) :=
by
  sorry

end probability_abc_eq_72_l544_544083


namespace four_digit_even_divisible_by_5_l544_544886

theorem four_digit_even_divisible_by_5 : 
  ∃ n : ℕ, n = 100 ∧ ∀ x : ℕ, 1000 ≤ x ∧ x ≤ 9999 ∧ (∀ d : ℕ, d ∈ [digit_list x] → (d % 2 = 0)) ∧ (x % 5 = 0) → 
  (digit_list x).length = 4 :=
sorry

end four_digit_even_divisible_by_5_l544_544886


namespace find_r_l544_544333

theorem find_r (f g : ℝ → ℝ) (monic_f : ∀x, f x = (x - r - 2) * (x - r - 8) * (x - a))
  (monic_g : ∀x, g x = (x - r - 4) * (x - r - 10) * (x - b)) (h : ∀ x, f x - g x = r):
  r = 32 :=
by
  sorry

end find_r_l544_544333


namespace find_fixed_point_l544_544771

noncomputable def f (z : ℂ) : ℂ := ((-1 - Complex.i * Real.sqrt 3) * z + (2 * Real.sqrt 3 + 18 * Complex.i)) / 2

theorem find_fixed_point :
  ∃ c : ℂ, f c = c ∧ c = 13.5 + 0.5 * Real.sqrt 3 + 3 * Complex.i := 
by
  -- proof is omitted
  sorry

end find_fixed_point_l544_544771


namespace proof_problem_l544_544655

theorem proof_problem (f g g_inv : ℝ → ℝ) (hinv : ∀ x, f (x ^ 4 - 1) = g x)
  (hginv : ∀ y, g (g_inv y) = y) (h : ∀ y, f (g_inv y) = g (g_inv y)) :
  g_inv (f 15) = 2 :=
by
  sorry

end proof_problem_l544_544655


namespace triangles_not_congruent_l544_544281

theorem triangles_not_congruent (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (AB DE AC EF BC DF : ℝ)
  (hA : ∠A = ∠D ∧ ∠A = 90) 
  (h_cond : AC = EF ∧ BC = DF) 
  : ¬(congruent_tris A B C D E F) :=
sorry

end triangles_not_congruent_l544_544281


namespace center_in_fourth_quadrant_l544_544833

-- Defining the conditions
def ellipse_condition (α : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 * cos α - y^2 * sin α + 2 = 0

-- Defining the circle's center location condition
def center_of_circle (α : ℝ) (cx cy : ℝ) : Prop :=
  cx = -cos α ∧ cy = -sin α

-- The main theorem statement
theorem center_in_fourth_quadrant (α : ℝ) (cx cy : ℝ) (h_ellipse : ellipse_condition α) (h_center : center_of_circle α cx cy) :
  (cx > 0 ∧ cy < 0) :=
sorry

end center_in_fourth_quadrant_l544_544833


namespace dave_initial_tickets_l544_544144

theorem dave_initial_tickets (spent: ℕ) (left: ℕ) (h1: spent = 43) (h2: left = 55) : spent + left = 98 :=
by
  rw [h1, h2]
  simp
  exact eq.refl 98

end dave_initial_tickets_l544_544144


namespace exponentiation_and_multiplication_of_fractions_l544_544905

-- Let's define the required fractions
def a : ℚ := 3 / 4
def b : ℚ := 1 / 5

-- Define the expected result
def expected_result : ℚ := 81 / 1280

-- State the theorem to prove
theorem exponentiation_and_multiplication_of_fractions : (a^4) * b = expected_result := by 
  sorry

end exponentiation_and_multiplication_of_fractions_l544_544905


namespace inscribed_angle_sum_l544_544387

theorem inscribed_angle_sum : 
  let arcs := 24 
  let arc_to_angle (n : ℕ) := 360 / arcs * n / 2 
  (arc_to_angle 4 + arc_to_angle 6 = 75) :=
by
  sorry

end inscribed_angle_sum_l544_544387


namespace minimize_expression_l544_544439

theorem minimize_expression : ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  exists 3
  split
  -- Here, you would prove the inequality and the fact that x = 3 gives the minimum.
  sorry

end minimize_expression_l544_544439


namespace greatest_and_next_values_l544_544530

theorem greatest_and_next_values :
  let a₁ := real.exp ((1 / 4) * real.log 4)
  let a₂ := real.exp ((1 / 5) * real.log 5)
  let a₃ := real.exp ((1 / 16) * real.log 16)
  let a₄ := real.exp ((1 / 25) * real.log 25) in
  (a₁ > a₂) ∧ (a₁ > a₃) ∧ (a₁ > a₄) ∧ (a₂ > a₃) ∧ (a₂ > a₄) :=
by
  sorry

end greatest_and_next_values_l544_544530


namespace integral1_result_integral2_result_l544_544549

noncomputable def integral1 : ℝ :=
  ∫ x in 1..2, (x - x^2 + 1/x)

noncomputable def integral2 : ℝ :=
  ∫ x in -Real.pi..0, (Real.cos x + Real.exp x)

theorem integral1_result :
  integral1 = Real.log 2 - 5 / 6 :=
sorry

theorem integral2_result :
  integral2 = 1 - 1 / Real.exp Real.pi :=
sorry

end integral1_result_integral2_result_l544_544549


namespace arrange_moon_l544_544922

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ways_to_arrange_moon : ℕ :=
  let total_letters := 4
  let repeated_O_count := 2
  factorial total_letters / factorial repeated_O_count

theorem arrange_moon : ways_to_arrange_moon = 12 := 
by {
  sorry -- Proof is omitted as instructed
}

end arrange_moon_l544_544922


namespace no_solution_inequality_l544_544660

theorem no_solution_inequality (m : ℝ) : ¬(∃ x : ℝ, 2 * x - 1 > 1 ∧ x < m) → m ≤ 1 :=
by
  intro h
  sorry

end no_solution_inequality_l544_544660


namespace focus_of_parabola_l544_544585

theorem focus_of_parabola (a : ℝ) (h : ℝ) (k : ℝ) (x y : ℝ) :
  (∀ x, y = a * (x - h) ^ 2 + k) →
  a = -2 ∧ h = 0 ∧ k = 4 →
  (0, y - (1 / (4 * a))) = (0, 31 / 8) := by
  sorry

end focus_of_parabola_l544_544585


namespace modulus_of_complex_raised_to_eight_l544_544581

-- Define the complex number 2 + i in Lean
def z : Complex := Complex.mk 2 1

-- State the proof problem with conditions
theorem modulus_of_complex_raised_to_eight : Complex.abs (z ^ 8) = 625 := by
  sorry

end modulus_of_complex_raised_to_eight_l544_544581


namespace xiao_li_probability_l544_544671

-- Define the group structure
structure Group where
  female_members : Finset String -- Set of female students
  male_members : Finset String -- Set of male students
  total_members : Finset String -- Total group members including both females and males
  xiao_li : String -- Xiao Li's identifier
  is_female : xiao_li ∈ female_members -- Xiao Li must be a female member
  all_members : female_members ∪ male_members = total_members -- All members combined

-- Given the conditions
def group : Group := {
  female_members := Finset.fromList ["Xiao Li", "Alice", "Eve"],
  male_members := Finset.fromList ["Bob", "Charlie"],
  total_members := Finset.fromList ["Xiao Li", "Alice", "Eve", "Bob", "Charlie"],
  xiao_li := "Xiao Li",
  is_female := by simp,
  all_members := by simp
}

-- Define the probability calculation
def probability_of_xiao_li (group : Group) : ℚ :=
  if Finset.card group.total_members = 5 then
    1 / 5 
  else 
    0

-- Statement of the theorem
theorem xiao_li_probability : probability_of_xiao_li group = 1 / 5 := by
  sorry

end xiao_li_probability_l544_544671


namespace derivative_of_f_l544_544998

-- Define the function f
def f (x : ℝ) : ℝ := x * cos (2 * x)

-- State the theorem that we need to prove
theorem derivative_of_f (x : ℝ) : deriv f x = cos (2 * x) - 2 * x * sin (2 * x) :=
by
  -- Skip the proof for now
  sorry

end derivative_of_f_l544_544998


namespace left_handed_like_jazz_l544_544037

theorem left_handed_like_jazz (total_members left_handed like_jazz right_handed_dislike_jazz : ℕ)
  (h1 : total_members = 20)
  (h2 : left_handed = 8)
  (h3 : like_jazz = 15)
  (h4 : right_handed_dislike_jazz = 2)
  (h5 : left_handed + (total_members - left_handed) = total_members) :
  ∃ x, x = 5 ∧ 
    left_handed + (left_handed - x) + (like_jazz - x) + right_handed_dislike_jazz = total_members := 
by
  use 5
  split
  · rfl
  · sorry

end left_handed_like_jazz_l544_544037


namespace distance_between_B_and_C_l544_544668

theorem distance_between_B_and_C (r : ℝ) (d : ℝ) (ac : ℝ) (ab : ℝ) (bc : ℝ) :
  r = 10 ∧ d = 2 * r ∧ ac = 12 ∧ ab = d ∧ (ab ^ 2 = ac ^ 2 + bc ^ 2) → bc = 16 :=
by
  intro h
  cases h with hr hr1
  cases hr1 with hd hr1
  cases hr1 with hac hr1
  cases hr1 with hab hpyth
  rw [hr, hd, hac, hab] at hpyth
  -- hr : r = 10
  -- hd : d = 2 * r
  -- hac : ac = 12
  -- hab : ab = d
  -- hpyth : ab ^ 2 = ac ^ 2 + bc ^ 2
  sorry

end distance_between_B_and_C_l544_544668


namespace distance_from_origin_l544_544124

theorem distance_from_origin 
  (diagonal_length : Real)
  (h_diagonal : diagonal_length = 2 * Real.sqrt 2)
  (sides_aligned : true) -- The sides_aligned condition indicates the diagonals coincide with the coordinate axes.
  : ∃ d, d = 1 :=
by
  have side_length : Real := diagonal_length / Real.sqrt 2
  exists side_length / 2
  rw [h_diagonal, Real.mul_div_cancel' _ (Real.ne_of_gt (Real.sqrt_pos.mpr (by norm_num)))]
  norm_num
  sorry

end distance_from_origin_l544_544124


namespace nearest_integer_to_expr_l544_544050

theorem nearest_integer_to_expr : 
  let a := 3 + Real.sqrt 5
  let b := (a)^6
  abs (b - 2744) < 1
:= sorry

end nearest_integer_to_expr_l544_544050


namespace number_of_bricks_needed_l544_544647

theorem number_of_bricks_needed :
  ∀ (brick_length brick_width brick_height wall_length wall_height wall_width : ℝ),
  brick_length = 25 → 
  brick_width = 11.25 → 
  brick_height = 6 → 
  wall_length = 750 → 
  wall_height = 600 → 
  wall_width = 22.5 → 
  (wall_length * wall_height * wall_width) / (brick_length * brick_width * brick_height) = 6000 :=
by
  intros brick_length brick_width brick_height wall_length wall_height wall_width
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end number_of_bricks_needed_l544_544647


namespace average_integers_37_l544_544895

-- Define N and the conditions for N
def is_in_range (N : ℕ) : Prop := 20 < N ∧ N < 54

-- Define the average function
def average (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem average_integers_37 : average (List.filter is_in_range (List.range 54)) = 37 :=
  sorry

end average_integers_37_l544_544895


namespace x_intercept_of_parallel_line_is_correct_l544_544238

theorem x_intercept_of_parallel_line_is_correct (a : ℝ) (h_parallel : -(a + 2) / 3 = -(a - 1) / 2) : 
    let l1_eq := (a + 2) * x + 3 * y = 5,
        x_intercept := -5 / (3 * (a + 2))
    in 9 * x + 3 * y = 5 → (x_intercept = 5 / 9) := sorry

end x_intercept_of_parallel_line_is_correct_l544_544238


namespace five_fridays_in_september_l544_544001

theorem five_fridays_in_september (year : ℕ) :
  (∃ (july_wednesdays : ℕ × ℕ × ℕ × ℕ × ℕ), 
     (july_wednesdays = (1, 8, 15, 22, 29) ∨ 
      july_wednesdays = (2, 9, 16, 23, 30) ∨ 
      july_wednesdays = (3, 10, 17, 24, 31)) ∧ 
      september_days = 30) → 
  ∃ (september_fridays : ℕ × ℕ × ℕ × ℕ × ℕ), 
  (september_fridays = (1, 8, 15, 22, 29)) :=
by
  sorry

end five_fridays_in_september_l544_544001


namespace circle_standard_equation_l544_544182

noncomputable def circle_equation (a : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - a)^2 + y^2 = 1

theorem circle_standard_equation : circle_equation 2 := by
  sorry

end circle_standard_equation_l544_544182


namespace two_S_is_int_divisible_by_n_l544_544714

-- Define that n is an odd positive integer.
axiom odd_positive_integer (n : ℕ) : n % 2 = 1 ∧ n > 0 

-- Define P as a cyclic polygon with integral coordinates and area S.
structure Polygon :=
  (vertices : list (ℤ × ℤ))
  (cyclic : ℕ) -- cyclic is a dummy parameter to represent that this polygon is cyclic.
  (area : ℚ)

-- Assume the side lengths squared are divisible by n.
def side_lengths_squared_divisible (n : ℕ) (P : Polygon) : Prop :=
  ∀ (i j : ℕ), (i < P.vertices.length ∧ j < P.vertices.length) →
  (let (x1, y1) := P.vertices.nth_le i sorry;
       (x2, y2) := P.vertices.nth_le j sorry  in
   ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) % n = 0)

theorem two_S_is_int_divisible_by_n
  (n : ℕ) (P : Polygon) (S : ℚ)
  (h1 : odd_positive_integer n)
  (h2 : P.area = S)
  (h3 : side_lengths_squared_divisible n P) :
  ∃ k : ℤ, 2 * S = k ∧ n ∣ k :=
sorry

end two_S_is_int_divisible_by_n_l544_544714


namespace guarantee_genuine_coin_discovery_l544_544796

theorem guarantee_genuine_coin_discovery :
  ∀ (coins : Fin 100 → ℝ),
  (∃ (c : Fin 100), ∀ (d : Fin 100), coins c < coins d)
  ∧ (∃ (c : Fin 100), ∀ (d : Fin 100), coins c > coins d)
  ∧ (∀ (i j : Fin 100), (coins i = coins j ∨ coins i < coins j ∨ coins i > coins j)) ->
  ∃ g : Fin 100, ∀ i : Fin 100, coins g > coins i :=
begin
  sorry
end

end guarantee_genuine_coin_discovery_l544_544796


namespace area_of_square_KMLD_l544_544690

theorem area_of_square_KMLD 
    (A B C D K L M : Point)
    (h1 : rhombus A B C D)
    (h2 : distance A B = 4)
    (h3 : distance A D = 4)
    (h4 : midpoint K B C)
    (h5 : square K M L D) :
  area K M L D = 28 :=
sorry

end area_of_square_KMLD_l544_544690


namespace product_inequality_l544_544206

variable {n : ℕ}
variable (x : Fin n → ℝ) (m : ℝ) (a s : ℝ)

theorem product_inequality (hx_pos : ∀ i, 0 < x i)
  (hm_pos : 0 < m) (ha_ge_zero : 0 ≤ a) (hx_sum : (Finset.univ.sum x) = s)
  (hs_le_n : s ≤ n) :
  (Finset.univ.prod (λ i, (x i)^m + (1 / (x i)^m) + a)) ≥ 
  ((s / n)^m + (n / s)^m + a)^n := sorry

end product_inequality_l544_544206


namespace false_statement_l544_544983

variables {a b : Type} [Line a] [Line b]
variables {α β : Type} [Plane α] [Plane β]

axiom a_perpendicular_alpha : a ⊥ α
axiom b_perpendicular_beta : b ⊥ β
axiom a_diff_b : a ≠ b
axiom alpha_diff_beta : α ≠ β

theorem false_statement (h : α ∩ β ≠ ∅) : a ∩ b = ∅ :=
sorry

end false_statement_l544_544983


namespace magician_trick_l544_544507

def coin (a : Fin 11 → Bool) : Prop :=
  ∃ i : Fin 11, a i = a (i + 1) % 11

theorem magician_trick (a : Fin 11 → Bool) 
(H : ∃ i : Fin 11, a i = a (i + 1) % 11) : 
  ∃ j : Fin 11, j ≠ 0 ∧ a j = a 0 :=
sorry

end magician_trick_l544_544507


namespace range_of_a_if_p_or_q_l544_544980

variables {a : ℝ} {x : ℝ}
def proposition_p : Prop :=
  ∀ x ≥ 1, (2 : ℝ)^(x^2 - 2 * a * x) is_increasing_on (Ici (1 : ℝ))

def proposition_q : Prop :=
  ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

theorem range_of_a_if_p_or_q (h : proposition_p ∨ proposition_q) : -2 < a ∧ a <= 2 :=
sorry

end range_of_a_if_p_or_q_l544_544980


namespace arithmetic_sequence_L2_condition_geometric_sequence_Lhalf_condition_sum_sequence_Lk0_condition_l544_544596

-- Define arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

-- 1. Prove the arithmetic sequence with a common difference of 2 satisfies the "L(2) condition".
theorem arithmetic_sequence_L2_condition (a₁ : ℝ) : ∀ m n : ℕ, m ≠ n → |arithmetic_sequence a₁ 2 m - arithmetic_sequence a₁ 2 n| ≤ 2 * |m - n| :=
by sorry

-- Define geometric sequence
def geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * q ^ (n - 1)

-- 2. Prove the positive geometric sequence with first term 1 and common ratio q satisfies the "L(1/2) condition" for q in [1/2, 1]
theorem geometric_sequence_Lhalf_condition : ∀ q : ℝ, (1 / 2) ≤ q ∧ q < 1 → ∀ m n : ℕ, m ≠ n → |geometric_sequence 1 q m - geometric_sequence 1 q n| ≤ (1 / 2) * |m - n| :=
by sorry

-- Define sum of first n terms of a geometric sequence
def sum_geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * (1 - q ^ n) / (1 - q)

-- 3. Prove for the sum sequence Sn, there exists a positive number k0 such that Sn satisfies the "L(k0) condition"
theorem sum_sequence_Lk0_condition : ∀ q : ℝ, (1 / 2) ≤ q ∧ q < 1 → ∃ k₀ : ℝ, 0 < k₀ ∧ ∀ m n : ℕ, m ≠ n → |sum_geometric_sequence 1 q m - sum_geometric_sequence 1 q n| ≤ k₀ * |m - n| :=
by sorry

end arithmetic_sequence_L2_condition_geometric_sequence_Lhalf_condition_sum_sequence_Lk0_condition_l544_544596


namespace distance_between_points_l544_544808

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_between_points : distance 2 3 9 6 = Real.sqrt 58 := by
  sorry

end distance_between_points_l544_544808


namespace polar_equation_of_semicircle_length_segment_PQ_l544_544685

-- Definition of the semicircle C in Cartesian coordinates
def semicircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + y ^ 2 = 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Definition of points O, P, and Q in polar coordinates
def point_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * cos θ ∧ 0 ≤ θ ∧ θ ≤ π / 2

-- Definition of the polar equation of the line (l)
def line_l (ρ θ : ℝ) : Prop :=
  ρ * (sin θ + sqrt 3 * cos θ) = 5 * sqrt 3

-- Definitions of polar coordinates of points P and Q and the ray OM
def point_P (ρ θ : ℝ) : Prop :=
  ρ = 1 ∧ θ = π / 3

def point_Q (ρ θ : ℝ) : Prop :=
  ρ = 5 ∧ θ = π / 3

def interval (y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ 1

-- Theorem to determine the polar equation of the semicircle
theorem polar_equation_of_semicircle :
  ∀ θ, 0 ≤ θ ∧ θ ≤ π / 2 ↔ ∃ ρ, point_polar ρ θ → (ρ = 2 * cos θ) :=
by
  sorry

-- Theorem to find the length of the line segment PQ
theorem length_segment_PQ : 
  (∀ θ : ℝ, θ = π / 3 → ∃ ρ1 ρ2, point_P ρ1 θ ∧ point_Q ρ2 θ → |ρ1 - ρ2| = 4) :=
by
  sorry

end polar_equation_of_semicircle_length_segment_PQ_l544_544685


namespace find_box_width_l544_544859

theorem find_box_width :
  (∃ w : ℕ, 7 * w * 3 = 42 * 9) → ∃ w : ℕ, w = 18 :=
by
  intros h
  obtain ⟨w, hw⟩ := h
  use w
  have h_eq : 7 * w * 3 = 42 * 9 := hw,
  calc
    7 * w * 3 = 21 * w      : by ring
    ...     = 42 * 9        : by exact h_eq
    ...     = 21 * 18       : by norm_num
  exact (eq_of_mul_eq_mul_left (by norm_num) (by norm_num))

#fetch find_box_width

end find_box_width_l544_544859


namespace evaluate_expression_l544_544332

def f (x : ℕ) : ℕ := 4 * x + 2
def g (x : ℕ) : ℕ := 3 * x + 4

theorem evaluate_expression : f (g (f 3)) = 186 := 
by 
  sorry

end evaluate_expression_l544_544332


namespace smallest_possible_b_l544_544338

-- Definition of the polynomial Q(x)
def Q (x : ℤ) : ℤ := sorry -- Polynomial with integer coefficients

-- Initial conditions for b and Q
variable (b : ℤ) (hb : b > 0)
variable (hQ1 : Q 2 = b)
variable (hQ2 : Q 4 = b)
variable (hQ3 : Q 6 = b)
variable (hQ4 : Q 8 = b)
variable (hQ5 : Q 1 = -b)
variable (hQ6 : Q 3 = -b)
variable (hQ7 : Q 5 = -b)
variable (hQ8 : Q 7 = -b)

theorem smallest_possible_b : b = 315 :=
by
  sorry

end smallest_possible_b_l544_544338


namespace temperature_reading_l544_544764

theorem temperature_reading (scale_min scale_max : ℝ) (arrow : ℝ) (h1 : scale_min = -6.0) (h2 : scale_max = -5.5) (h3 : scale_min < arrow) (h4 : arrow < scale_max) : arrow = -5.7 :=
sorry

end temperature_reading_l544_544764


namespace nearest_integer_to_expr_l544_544053

theorem nearest_integer_to_expr : 
  let a := 3 + Real.sqrt 5
  let b := (a)^6
  abs (b - 2744) < 1
:= sorry

end nearest_integer_to_expr_l544_544053


namespace vector_dot_product_l544_544400

variables {V : Type*} [InnerProductSpace ℝ V]

-- Definition of points and vectors
variables (O A B C : V)
-- Radius of circumcircle
axiom radius_one : ∥(O - A)∥ = 1
-- Given vector equation
axiom given_vector_eq : 2 • (O - A) + (B - A) + (C - A) = 0
-- Magnitudes equality
axiom magnitude_eq : ∥(O - A)∥ = ∥(B - A)∥

-- Target statement to prove
theorem vector_dot_product : (C - A) ⬝ (C - B) = 3 :=
sorry

end vector_dot_product_l544_544400


namespace smallest_positive_n_l544_544076

theorem smallest_positive_n (n : ℕ) (h : 77 * n ≡ 308 [MOD 385]) : n = 4 :=
sorry

end smallest_positive_n_l544_544076


namespace find_f_1987_l544_544535

-- Define the function f and the sequence of digits as given
def digit_sequence : ℕ → ℕ
| n := nat.digits 10 (n + 1)

-- Define the function f
noncomputable def f (n : ℕ) : ℕ :=
let digits_up_to : ℕ → ℕ := λ m, 9 * ((10 ^ m) - 1) * m // 9 in
nat.find_greatest (λ m, digits_up_to m ≤ 10^n) (10^n)

-- Main theorem
theorem find_f_1987 : f 1987 = 1984 :=
by {
  -- proof steps go here
  sorry
}

end find_f_1987_l544_544535


namespace magician_can_identify_matching_coin_l544_544502

-- Define the types and conditions
variable (coins : Fin 11 → Bool) -- Each coin is either heads (true) or tails (false).
variable (uncovered_index : Fin 11) -- The index of the initially uncovered coin.

-- Define the property to be proved: the magician can point to an adjacent coin with the same face as the uncovered one.
theorem magician_can_identify_matching_coin (h : ∃ i : Fin 11, coins i = coins (i + 1) % 11) :
  ∃ j : Fin 11, (j ≠ uncovered_index) ∧ coins j = coins uncovered_index :=
  sorry

end magician_can_identify_matching_coin_l544_544502


namespace magician_trick_l544_544493

theorem magician_trick (coins : Fin 11 → Bool) 
  (exists_pair : ∃ i : Fin 11, coins i = coins ((i + 1) % 11)) 
  (assistant_left_uncovered : Fin 11) 
  (uncovered_coin_same_face : coins assistant_left_uncovered = coins (some (exists_pair))) :
  ∃ j : Fin 11, j ≠ assistant_left_uncovered ∧ coins j = coins assistant_left_uncovered := 
  sorry

end magician_trick_l544_544493


namespace sum_first_9_terms_l544_544607

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Assume an arithmetic sequence {a_n}
variable (a : ℕ → α)

-- Hypothesis: a2 + a8 = 10 (in some real number context)
def condition : Prop := a 2 + a 8 = (10 : α)

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (n : ℕ) : α :=
∑ i in Finset.range n, a (i + 1)

-- The theorem we need to prove
theorem sum_first_9_terms : condition a → sum_arithmetic_seq a 9 = 45 := 
by
  sorry

end sum_first_9_terms_l544_544607


namespace total_flour_l544_544373

def bought_rye_flour := 5
def bought_bread_flour := 10
def bought_chickpea_flour := 3
def had_pantry_flour := 2

theorem total_flour : bought_rye_flour + bought_bread_flour + bought_chickpea_flour + had_pantry_flour = 20 :=
by
  sorry

end total_flour_l544_544373


namespace find_a2023_l544_544972

noncomputable def a_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n, 2 * S n = a n * (a n + 1))

theorem find_a2023 (a S : ℕ → ℕ) 
  (h_seq : a_sequence a S)
  (h_pos : ∀ n, 0 < a n) 
  : a 2023 = 2023 :=
sorry

end find_a2023_l544_544972


namespace max_projection_area_of_tetrahedron_l544_544424

theorem max_projection_area_of_tetrahedron :
  ∃ (tetrahedron : Type) (edge : Type) (area : ℝ),
    (tetrahedron.has_property (is_equilateral_with_side_length 1) ∧
     tetrahedron.adjacent_faces_with_dihedral_angle edge (π / 4)) →
    (rotate_tetrahedron_around edge tetrahedron) →
    (project_onto_plane_containing edge) →
    max_area_of_projection = (√3 / 4) :=
by sorry

end max_projection_area_of_tetrahedron_l544_544424


namespace Toph_caught_12_fish_l544_544880

-- Define the number of fish each person caught
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def average_fish : ℕ := 8
def num_people : ℕ := 3

-- The total number of fish based on the average
def total_fish : ℕ := average_fish * num_people

-- Define the number of fish Toph caught
def Toph_fish : ℕ := total_fish - Aang_fish - Sokka_fish

-- Prove that Toph caught the correct number of fish
theorem Toph_caught_12_fish : Toph_fish = 12 := sorry

end Toph_caught_12_fish_l544_544880


namespace total_students_l544_544407

theorem total_students : 
  let grade_3 := 19
  let grade_4 := 2 * grade_3
  let boys_2 := 10
  let girls_2 := 19
  let grade_2 := boys_2 + girls_2
  let total := grade_2 + grade_3 + grade_4
  in total = 86 :=
by
  let grade_3 := 19
  let grade_4 := 2 * grade_3
  let boys_2 := 10
  let girls_2 := 19
  let grade_2 := boys_2 + girls_2
  let total := grade_2 + grade_3 + grade_4
  sorry

end total_students_l544_544407


namespace calculate_number_l544_544105

theorem calculate_number (tens ones tenths hundredths : ℝ) 
  (h_tens : tens = 21) 
  (h_ones : ones = 8) 
  (h_tenths : tenths = 5) 
  (h_hundredths : hundredths = 34) :
  tens * 10 + ones * 1 + tenths * 0.1 + hundredths * 0.01 = 218.84 :=
by
  sorry

end calculate_number_l544_544105


namespace Teresa_current_age_l544_544005

-- Definitions of the conditions
def Morio_current_age := 71
def Morio_age_when_Michiko_born := 38
def Teresa_age_when_Michiko_born := 26

-- Definition of Michiko's current age
def Michiko_current_age := Morio_current_age - Morio_age_when_Michiko_born

-- The Theorem statement
theorem Teresa_current_age : Teresa_age_when_Michiko_born + Michiko_current_age = 59 :=
by
  -- Skip the proof
  sorry

end Teresa_current_age_l544_544005


namespace prism_volume_l544_544435

theorem prism_volume (x y z : ℝ) (h1 : x * y = 24) (h2 : y * z = 8) (h3 : x * z = 3) : 
  x * y * z = 24 :=
sorry

end prism_volume_l544_544435


namespace sum_of_powers_zero_l544_544268

noncomputable def sum_of_powers (a : ℤ) : ℤ :=
  a^0 + a^1 + a^2 + a^3 + a^4 + a^5 + a^6 + a^7 + a^8

theorem sum_of_powers_zero (a : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) 
  (h : (λ x, (x - a) ^ 8) = λ x, a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 +
                                     a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8) 
  (h1 : a_5 = 56)
  (ha : a^3 = 56) : 
  sum_of_powers (-1) = 0 :=
begin
  sorry
end

end sum_of_powers_zero_l544_544268


namespace find_a2023_l544_544973

noncomputable def a_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n, 2 * S n = a n * (a n + 1))

theorem find_a2023 (a S : ℕ → ℕ) 
  (h_seq : a_sequence a S)
  (h_pos : ∀ n, 0 < a n) 
  : a 2023 = 2023 :=
sorry

end find_a2023_l544_544973


namespace average_salary_correct_l544_544401

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 15000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 9000 := by
  -- proof is skipped
  sorry

end average_salary_correct_l544_544401


namespace gas_consumption_100_l544_544666

noncomputable def gas_consumption (x : ℝ) : Prop :=
  60 * 1 + (x - 60) * 1.5 = 1.2 * x

theorem gas_consumption_100 (x : ℝ) (h : gas_consumption x) : x = 100 := 
by {
  sorry
}

end gas_consumption_100_l544_544666


namespace max_hotdogs_proof_l544_544892

noncomputable def max_hotdogs (budget : ℝ) : ℝ :=
let cost_8 := 1.55 in
let cost_20 := 3.05 in
let cost_50 := 7.45 in
let cost_100 := 14.10 in
let cost_250 := 22.95 in
let pack_8 := 8 in
let pack_20 := 20 in
let pack_50 := 50 in
let pack_100 := 100 in
let pack_250 := 250 in
let discount_threshold := 10 in
let discount_rate := 0.05 in
let max_packs := 30 in
let min_packs := 15 in
if budget > 0 then
  let base_cost_250 := cost_250 * min_packs in
  if base_cost_250 <= budget then
    let discounted_cost_250 := cost_250 * (1 - discount_rate) in
    let packs_within_budget := floor ((budget - (min_packs * cost_250)) / discounted_cost_250) + min_packs in
    if packs_within_budget <= max_packs then
      packs_within_budget * pack_250
    else
      max_packs * pack_250
  else
    0
else
  0

-- Now, stating the theorem.
theorem max_hotdogs_proof (budget : ℝ) (h : budget = 300) : max_hotdogs budget = 3250 :=
by {
  rw [h],
  sorry
}

end max_hotdogs_proof_l544_544892


namespace AMC_score_count_l544_544137

theorem AMC_score_count (q p_c p_i p_u : ℕ) (h_q : q = 30) (h_pc : p_c = 5) (h_pi : p_i = 0) (h_pu : p_u = 2) : 
  let scores := {5 * x + 2 * y | x y : ℕ // x + y ≤ q} 
  in scores.card = 145 := 
  by 
    sorry

end AMC_score_count_l544_544137


namespace JohnSpentAtSupermarketTotal_l544_544826

theorem JohnSpentAtSupermarketTotal (X : ℝ)
  (h_fruits : (1/2 : ℝ) * X)
  (h_meat : (1/3 : ℝ) * X)
  (h_bakery : (1/10 : ℝ) * X)
  (h_candy : X - (1/2 * X + 1/3 * X + 1/10 * X) = 5) :
  X = 75 := by
  sorry

end JohnSpentAtSupermarketTotal_l544_544826


namespace sum_of_ratios_of_parallel_intersected_segments_is_one_l544_544326

theorem sum_of_ratios_of_parallel_intersected_segments_is_one
    (A B C O D E F : Type)
    [triangle ABC : triangle A B C]
    [O_in_triangle_ABC : inside_triangle O A B C]
    [D_parallel_BC : ∃ D, line_parallel D BC ∧ line_passing_through O D ∧ line_intersects D CA]
    [E_parallel_CA : ∃ E, line_parallel E CA ∧ line_passing_through O E ∧ line_intersects E AB]
    [F_parallel_AB : ∃ F, line_parallel F AB ∧ line_passing_through O F ∧ line_intersects F BC] :
    ∀ {BC AB CA BF AE CD} (H : add (div BF BC) (add (div AE AB) (div CD CA)) = 1), is_true H := sorry

end sum_of_ratios_of_parallel_intersected_segments_is_one_l544_544326


namespace find_c_plus_d_l544_544187

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
if x < 3 then c * x + d else 10 - 2 * x

theorem find_c_plus_d (c d : ℝ) (h : ∀ x, g c d (g c d x) = x) : c + d = 4.5 :=
sorry

end find_c_plus_d_l544_544187


namespace vertices_lattice_points_l544_544790

/-
Define an integer lattice point as a 3-tuple of integers.
-/
def lattice_point := (ℤ × ℤ × ℤ)

/-
Define the vertices of the cube as instances of lattice_point.
-/
variables (A1 B C1 D A B1 C D1 : lattice_point)

/-
The given conditions are that A1, B, C1, and D are on the integer lattice (already defined by the types).
Now we will set up the main theorem that if A1, B, C1, and D are lattice points,
then the other vertices A, B1, C, and D1 must also be lattice points.
-/
theorem vertices_lattice_points 
  (hA1 : lattice_point) 
  (hB : lattice_point) 
  (hC1 : lattice_point) 
  (hD : lattice_point) : 
  (A : lattice_point) → 
  (B1 : lattice_point) → 
  (C : lattice_point) → 
  (D1 : lattice_point) := 
sorry

end vertices_lattice_points_l544_544790


namespace magician_assistant_strategy_l544_544499

-- Define the possible states of a coin
inductive CoinState | heads | tails

-- Define the circle of coins
def coinCircle := Fin 11 → CoinState

-- The strategy ensures there are adjacent coins with the same state
theorem magician_assistant_strategy (c : coinCircle) :
  ∃ i : Fin 11, c i = c ((i + 1) % 11) := by sorry

end magician_assistant_strategy_l544_544499


namespace arcsin_one_half_l544_544155

theorem arcsin_one_half : real.arcsin (1 / 2) = real.pi / 6 :=
by
  sorry

end arcsin_one_half_l544_544155


namespace apple_distribution_l544_544133

theorem apple_distribution : 
  ∃ (a b c : ℕ), a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 2 ∧ c ≥ 2 ∧ 
  ∃ n : ℕ, n = nat.choose 25 2 ∧ n = 300 :=
by
  use 3
  use 2
  use 2
  have : 3 + 2 + 2 ≤ 30 := by norm_num
  have : nat.choose 25 2 = 300 := by
    rw [nat.choose_eq_factorial_div_factorial 24 1 2]
    norm_num
  use 300
  sorry

end apple_distribution_l544_544133


namespace inequality_proof_l544_544275

theorem inequality_proof (a b : ℝ) (n : ℕ) (x : ℕ → ℝ)
  (h1 : ∀ i, 0 < x i ∧ a ≤ x i ∧ x i ≤ b) (h2 : 0 < a) (h3 : a < b) :
  (∑ i in finRange n, x i) * (∑ i in finRange n, (1 / (x i))) ≤ ((a + b) ^ 2 / (4 * a * b)) * (n ^ 2) :=
by
  sorry

end inequality_proof_l544_544275


namespace seq_bound_2023_l544_544807

-- Defining the sequence a_n as per the problem statement.
noncomputable def a : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := Nat.find (λ k, k > a (n+1) ∧ ∀ i j, i < j → j < (n+2) → (2 * a k ≠ a i + a j))

-- The statement to prove
theorem seq_bound_2023 : a 2023 ≤ 100000 :=
sorry

end seq_bound_2023_l544_544807


namespace hexagon_coloring_l544_544932

def colors : Type := Fin 7

def adjacent (v1 v2 : Fin 6) : Prop := 
  (v1 + 1) % 6 = v2 % 6 ∨ (v1 + 5) % 6 = v2 % 6

def diagonally_opposite (v1 v2 : Fin 6) : Prop := 
  (v1 + 3) % 6 = v2 % 6

theorem hexagon_coloring :
  ∃ (f : Fin 6 → colors), (∀ v1 v2 : Fin 6, adjacent v1 v2 → f v1 ≠ f v2)
                           ∧ (∀ v1 v2 : Fin 6, diagonally_opposite v1 v2 → f v1 ≠ f v2)
                           ∧ finset.card {f // (∀ v1 v2, adjacent v1 v2 → f v1 ≠ f v2)
                                              ∧ (∀ v1 v2, diagonally_opposite v1 v2 → f v1 ≠ f v2)} = 19404 :=
by
  sorry

end hexagon_coloring_l544_544932


namespace slope_of_tangent_line_min_AF_BF_l544_544989

theorem slope_of_tangent_line_min_AF_BF :
  let F : ℝ × ℝ := (1, 0)
  let parabola : set (ℝ × ℝ) := { p | p.2^2 = 4 * p.1 }
  let line_through_focus (k : ℝ) : ℝ → ℝ × ℝ := λ x, (x, k * (x - 1))
  ∃ k, (|F.1 + line_through_focus k 1.1| + 4 * |F.2 + line_through_focus k 1.2|)
  ∧ k = 2 * real.sqrt 2 ∨ k = -2 * real.sqrt 2
  sorry

end slope_of_tangent_line_min_AF_BF_l544_544989


namespace arithmetic_sequence_eleventh_term_l544_544168

theorem arithmetic_sequence_eleventh_term 
  (a d : ℚ)
  (h_sum_first_six : 6 * a + 15 * d = 30)
  (h_seventh_term : a + 6 * d = 10) : 
    a + 10 * d = 110 / 7 := 
by
  sorry

end arithmetic_sequence_eleventh_term_l544_544168


namespace trapezoid_area_l544_544876

theorem trapezoid_area (A B : ℝ) (hA : 0 ≤ A) (hB : 0 ≤ B) :
  let S := (real.sqrt A + real.sqrt B) in
  (S ^ 2) = A + B + 2 * real.sqrt (A * B) :=
by
  sorry

end trapezoid_area_l544_544876


namespace inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed_l544_544782

theorem inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed:
  (∀ a b : ℝ, a > b → a^3 > b^3) → (∀ a b : ℝ, a^3 > b^3 → a > b) :=
  by
  sorry

end inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed_l544_544782


namespace law_of_cosines_l544_544461

theorem law_of_cosines (a b c : ℝ) (A : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ A ≥ 0 ∧ A ≤ π) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A :=
sorry

end law_of_cosines_l544_544461


namespace remaining_vacation_days_l544_544536

-- Conditions
def total_days_worked : ℕ := 290
def public_holidays : ℕ := 10
def sick_leave_days : ℕ := 5
def days_off_march : ℕ := 5
def multiplier_september : ℕ := 2
def vacation_day_ratio : ℕ := 10

-- Question expressed as a theorem
theorem remaining_vacation_days :
  let days_that_count := total_days_worked - public_holidays - sick_leave_days in
  let vacation_days_earned := days_that_count / vacation_day_ratio in
  let vacation_days_taken := days_off_march + (days_off_march * multiplier_september) in
  vacation_days_earned - vacation_days_taken = 12 :=
by sorry

end remaining_vacation_days_l544_544536


namespace bug_movement_limit_l544_544108

theorem bug_movement_limit :
  let x_series := 1 - (1/4) + (1/16) - (1/64) + ...
  let y_series := (1/2) - (1/8) + (1/32) - (1/128) + ...
  x_series = 4/5 ∧ y_series = 2/5 :=
sorry

end bug_movement_limit_l544_544108


namespace triangle_bisector_ratio_l544_544306

theorem triangle_bisector_ratio (AB BC CA : ℝ) (h_AB_pos : 0 < AB) (h_BC_pos : 0 < BC) (h_CA_pos : 0 < CA)
  (AA1_bisector : True) (BB1_bisector : True) (O_intersection : True) : 
  AA1 / OA1 = 3 :=
by
  sorry

end triangle_bisector_ratio_l544_544306


namespace difference_of_two_numbers_l544_544802

theorem difference_of_two_numbers (x y : ℕ) (hxy : x ≠ y) (hx : 1 ≤ x ∧ x ≤ 38) (hy : 1 ≤ y ∧ y ≤ 38) 
  (h : ((38 * 39 / 2) - x - y = x * y + 1)) : abs (y - x) = 20 := 
sorry

end difference_of_two_numbers_l544_544802


namespace no_double_number_with_given_digits_l544_544900

theorem no_double_number_with_given_digits :
  ¬ ∃ (a b : ℕ), a > b ∧ a = 2 * b ∧ 
    (∀ d ∈ {2, 3, 4, 5, 6, 7, 8, 9}, d ∈ nat.digits 10 a ∨ d ∈ nat.digits 10 b) ∧
    (∀ d ∈ nat.digits 10 a ++ nat.digits 10 b, d ∈ {2, 3, 4, 5, 6, 7, 8, 9}) ∧
    nat.digits 10 a ++ nat.digits 10 b = {2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

end no_double_number_with_given_digits_l544_544900


namespace quadratic_expression_invariant_l544_544560

noncomputable section

variable {p q : ℝ}

def has_two_distinct_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem quadratic_expression_invariant
  (h1 : has_two_distinct_roots 1 p q)
  (h2 : has_two_distinct_roots 1 q p) :
  let x1 x2 := roots_of_quadratic_eq 1 p q
  let x3 x4 := roots_of_quadratic_eq 1 q p
  in (1 / (x1 * x3) + 1 / (x1 * x4) + 1 / (x2 * x3) + 1 / (x2 * x4) = 1) :=
sorry

end quadratic_expression_invariant_l544_544560


namespace magician_assistant_strategy_l544_544498

-- Define the possible states of a coin
inductive CoinState | heads | tails

-- Define the circle of coins
def coinCircle := Fin 11 → CoinState

-- The strategy ensures there are adjacent coins with the same state
theorem magician_assistant_strategy (c : coinCircle) :
  ∃ i : Fin 11, c i = c ((i + 1) % 11) := by sorry

end magician_assistant_strategy_l544_544498


namespace exponent_logarithm_simplifies_l544_544148

theorem exponent_logarithm_simplifies :
  (1/2 : ℝ) ^ (Real.log 3 / Real.log 2 - 1) = 2 / 3 :=
by sorry

end exponent_logarithm_simplifies_l544_544148


namespace amelia_score_points_l544_544170

theorem amelia_score_points (x y z : ℕ) (hx : x + y + z = 40) (hz : z = 10) :
  ((0.25 * 3) * x + (0.4 * 2) * y + (0.5 * 1) * z) = 28.5 :=
by
  simp [hx, hz]
  sorry

end amelia_score_points_l544_544170


namespace kevin_sold_13_crates_of_grapes_l544_544315

-- Define the conditions
def total_crates : ℕ := 50
def crates_of_mangoes : ℕ := 20
def crates_of_passion_fruits : ℕ := 17

-- Define the question and expected answer
def crates_of_grapes : ℕ := total_crates - (crates_of_mangoes + crates_of_passion_fruits)

-- Prove that the crates of grapes equals to 13
theorem kevin_sold_13_crates_of_grapes :
  crates_of_grapes = 13 :=
by
  -- The proof steps are omitted as per instructions
  sorry

end kevin_sold_13_crates_of_grapes_l544_544315


namespace minimum_area_of_quadrilateral_PADB_l544_544621

noncomputable def f (x : ℝ) : ℝ := 2 * real.sqrt (2 * x)

structure Point where
  x : ℝ
  y : ℝ

def on_graph_of_f (P : Point) : Prop := P.y = f P.x

def is_circle (D : Point → Prop) : Prop := 
  ∀ P, D P ↔ P.x^2 + P.y^2 - 4 * P.x + 3 = 0

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

def radius_of_circle (r : ℝ) : ℝ := 1

def minimum_value_quadrilateral_area (P A D B : Point) (f : ℝ → ℝ): (ℝ → ℝ) → ℝ := sorry

theorem minimum_area_of_quadrilateral_PADB (P A D B : Point)
  (D_circle : is_circle (λ Q, Q.x^2 + Q.y^2 - 4 * Q.x + 3 = 0))
  (P_on_f : on_graph_of_f P) :
  minimum_value_quadrilateral_area P A D B f = real.sqrt 3 := sorry

end minimum_area_of_quadrilateral_PADB_l544_544621


namespace simplify_trig_identity_l544_544377

theorem simplify_trig_identity (x y : ℝ) : 
  sin x ^ 2 + sin (x + y) ^ 2 - 2 * sin x * cos y * sin (x + y) = cos y ^ 2 :=
by
  sorry

end simplify_trig_identity_l544_544377


namespace OD1_length_l544_544309

noncomputable def cube_length_of_OD1 
  (O : point)
  (cube : cube)
  (r : ℝ)
  (r1 : ℝ) 
  (r2 : ℝ) 
  (r3 : ℝ) : ℝ 
  (h1 : r = 10) 
  (h2 : r1 = 1)
  (h3 : r2 = 1)
  (h4 : r3 = 3) := 
17

theorem OD1_length 
  (O : point)
  (cube : cube)
  (r : ℝ) 
  (r1 : ℝ) 
  (r2 : ℝ) 
  (r3 : ℝ) 
  (h1 : r = 10) 
  (h2 : r1 = 1)
  (h3 : r2 = 1)
  (h4 : r3 = 3) : 
  cube_length_of_OD1 O cube r r1 r2 r3 h1 h2 h3 h4 = 17 := 
sorry

end OD1_length_l544_544309


namespace beta_still_water_speed_l544_544800

-- Definitions that are used in the conditions
def alpha_speed_still_water : ℝ := 56 
def beta_speed_still_water : ℝ := 52  
def water_current_speed : ℝ := 4

-- The main theorem statement 
theorem beta_still_water_speed : β_speed_still_water = 61 := 
  sorry -- the proof goes here

end beta_still_water_speed_l544_544800


namespace k_ge_a_l544_544336

theorem k_ge_a (a k : ℕ) (h_pos_a : 0 < a) (h_pos_k : 0 < k) 
  (h_div : (a ^ 2 + k) ∣ (a - 1) * a * (a + 1)) : k ≥ a := 
sorry

end k_ge_a_l544_544336


namespace circles_divide_plane_l544_544672

theorem circles_divide_plane (n : ℕ) (h : n ≥ 2) :
  ∀ (f : ℕ → ℕ),
  (f 2 = 4) →
  (f 3 = 8) →
  (f 4 = 14) →
  (∀ m, 2 ≤ m → f(m) - f(m-1) = 2*(m-1)) →
  f(n) = n^2 - n + 2 :=
by
  intro f h2 h3 h4 h_ind
  sorry

end circles_divide_plane_l544_544672


namespace nearest_integer_is_11304_l544_544072

def nearest_integer_to_a_plus_b_pow_six (a b : ℝ) (h : b = Real.sqrt 5) : ℝ :=
  (a + b) ^ 6

theorem nearest_integer_is_11304 : nearest_integer_to_a_plus_b_pow_six 3 (Real.sqrt 5) rfl = 11304 := 
  sorry

end nearest_integer_is_11304_l544_544072


namespace calculate_expression_l544_544152

theorem calculate_expression : real.sqrt 4 - 2023^0 + 2 * real.cos (real.pi / 3) = 2 := by
  have h1: real.sqrt 4 = 2 := by
    exact real.sqrt_eq_iff_mul_self_eq.mpr ((show 2 * 2 = 4 from rfl)) (le_of_eq (rfl))
  
  have h2: 2023^0 = 1 := by
    exact pow_zero 2023
  
  have h3: real.cos (real.pi / 3) = 1 / 2 := by
    exact real.cos_pi_div_three
  
  rw [h1, h2, h3]
  norm_num
  done

end calculate_expression_l544_544152


namespace floor_sum_arithmetic_progression_l544_544556

theorem floor_sum_arithmetic_progression :
  (∑ i in Finset.range 142, floor (2 + i * 1.4)) = 14281.6 := 
sorry

end floor_sum_arithmetic_progression_l544_544556


namespace exists_common_interior_l544_544704

noncomputable def convex_polyhedron (n : ℕ) : Type :=
{ P : Type // ... } -- Assuming detailed properties related to convex polyhedra are defined elsewhere

variables (P1 : convex_polyhedron 9)
variables (vertices : fin 9 → ℝ^3)
variables (translations : fin 9 → ℝ^3)
variables (P : fin 9 → convex_polyhedron 9)

def translate (P1 : convex_polyhedron 9) (dx : ℝ^3) : convex_polyhedron 9 := ... -- Define translation
def interior_point (P : convex_polyhedron 9) (p : ℝ^3) : Prop := ... -- Define what makes a point interior

axiom vertices_P1 : ∀ i : fin 9, i.val < 9 → vertices i ∈ P1

axiom P_def : ∀ i : fin 9, i.val > 0 → P i = translate P1 (translations i)

theorem exists_common_interior :
  ∃ i j : fin 9, i ≠ j ∧ ∃ p : ℝ^3, interior_point (P i) p ∧ interior_point (P j) p :=
sorry

end exists_common_interior_l544_544704


namespace min_n_for_reals_l544_544583

theorem min_n_for_reals (n : ℕ) (h : n ≥ 2) : 
  (∃ (a : Fin n → ℝ), 
   { d : ℝ | ∃ i j, i < j ∧ d = |a i - a j| } = {1..n * (n - 1) / 2}) ↔ 
   n = 41 := 
sorry

end min_n_for_reals_l544_544583


namespace pentagon_diagonals_sum_l544_544701

theorem pentagon_diagonals_sum : 
  ∀ (FG HI GH IJ FJ : ℕ), 
    FG = 4 → HI = 4 → GH = 9 → IJ = 9 → FJ = 15 → 
    (let z := 18 in 
    let x := (z^2 - 81) / 4 in 
    let y := (z^2 - 16) / 9 in 
    3 * z + x + y = 5363 / 36) → (let p := 5363 in let q := 36 in p + q = 5399) :=
by
  intros FG HI GH IJ FJ FG_eq HI_eq GH_eq IJ_eq FJ_eq assumption_eq
  sorry

end pentagon_diagonals_sum_l544_544701


namespace nearest_integer_is_11304_l544_544070

def nearest_integer_to_a_plus_b_pow_six (a b : ℝ) (h : b = Real.sqrt 5) : ℝ :=
  (a + b) ^ 6

theorem nearest_integer_is_11304 : nearest_integer_to_a_plus_b_pow_six 3 (Real.sqrt 5) rfl = 11304 := 
  sorry

end nearest_integer_is_11304_l544_544070


namespace magician_trick_l544_544495

theorem magician_trick (coins : Fin 11 → Bool) 
  (exists_pair : ∃ i : Fin 11, coins i = coins ((i + 1) % 11)) 
  (assistant_left_uncovered : Fin 11) 
  (uncovered_coin_same_face : coins assistant_left_uncovered = coins (some (exists_pair))) :
  ∃ j : Fin 11, j ≠ assistant_left_uncovered ∧ coins j = coins assistant_left_uncovered := 
  sorry

end magician_trick_l544_544495


namespace triangle_side_relation_l544_544643

theorem triangle_side_relation (a b c : ℝ) (h1 : a^2 - 16 * b^2 - c^2 + 6 * a * b + 10 * b * c = 0) (h2 : a + b > c) :
  a + c = 2 * b := 
sorry

end triangle_side_relation_l544_544643


namespace nearest_integer_3_add_sqrt_5_pow_6_l544_544063

noncomputable def approx (x : ℝ) : ℕ := Real.floor (x + 0.5)

theorem nearest_integer_3_add_sqrt_5_pow_6 : 
  approx ((3 + Real.sqrt 5)^6) = 22608 :=
by
  -- Proof omitted, sorry
  sorry

end nearest_integer_3_add_sqrt_5_pow_6_l544_544063


namespace sum_of_distinct_y_satisfying_equation_l544_544589

theorem sum_of_distinct_y_satisfying_equation :
  (∑ y in { y : ℝ | 3 * y + 16 / y = 13 }.to_finset, y) = 16 / 3 :=
by sorry

end sum_of_distinct_y_satisfying_equation_l544_544589


namespace circumcircle_radius_l544_544307

-- Definitions of the given conditions
def b : ℝ := 4
def c : ℝ := 2
def A : ℝ := real.pi / 3  -- 60 degrees in radians

-- The Law of Cosines to find the length of side a
def a : ℝ := real.sqrt (b^2 + c^2 - 2 * b * c * real.cos A)

-- The proof problem statement
theorem circumcircle_radius (hA : A = real.pi / 3) (hb : b = 4) (hc : c = 2) : 
  let R := a / (2 * real.sin A) in 
  R = 2 := 
by sorry

end circumcircle_radius_l544_544307


namespace inequality_proof_l544_544207

variable {R : Type*} [LinearOrderedField R]

theorem inequality_proof (n : ℕ) (x : Fin n → R) (m : R) (a s : R) 
  (hxi_pos : ∀ i, 0 < x i) 
  (hm_pos : 0 < m) 
  (ha_nonneg : 0 ≤ a) 
  (hs_sum : (∑ i, x i) = s) 
  (hs_le_n : s ≤ n) : 
  (∏ i, (x i ^ m + 1 / (x i ^ m) + a)) ≥ (( (s / n : R) ^ m + (n / s : R) ^ m + a ) ^ n) := 
sorry

end inequality_proof_l544_544207


namespace larger_cube_volume_l544_544147

theorem larger_cube_volume (v : ℝ) (n : ℕ) (hn : n = 125) :
  let V := 125 * v in V = 125 * v :=
by
  sorry

end larger_cube_volume_l544_544147


namespace number_of_elements_in_M_l544_544708

def f (x : ℝ) : ℝ := Real.cos x

def nth_derivative (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ :=
match n with
| 0     => f
| m + 1 => (λ x, deriv (nth_derivative f m) x)

def M := {m : ℕ | nth_derivative f m 0 = Real.sin 0 ∧ m ≤ 2013}

theorem number_of_elements_in_M : fintype.card M = 503 :=
by {
  -- Proof goes here
  sorry
}

end number_of_elements_in_M_l544_544708


namespace hyperbola_condition_iff_l544_544618

variable (m n : ℝ)

theorem hyperbola_condition_iff (h : m ≠ 0 ∧ n ≠ 0) :
  (m * n < 0) ↔ (∀ x y : ℝ, (x^2 / m + y^2 / n = 1) → is_hyperbola x y) :=
sorry

end hyperbola_condition_iff_l544_544618


namespace polygon_side_length_l544_544603

variable {M : Type} [ConvexPolygon M]
variable (p : ℕ) [Prime p]
variable (decomposition_count : M → ℕ)

theorem polygon_side_length (h : decomposition_count M = p) : 
  ∃ side : ℕ, side = p - 1 := 
sorry

end polygon_side_length_l544_544603


namespace games_bought_at_garage_sale_l544_544723

-- Definitions based on conditions
def games_from_friend : ℕ := 2
def defective_games : ℕ := 2
def good_games : ℕ := 2

-- Prove the number of games bought at the garage sale equals 2
theorem games_bought_at_garage_sale (G : ℕ) 
  (h : games_from_friend + G - defective_games = good_games) : G = 2 :=
by 
  -- use the given information and work out the proof here
  sorry

end games_bought_at_garage_sale_l544_544723


namespace nearest_integer_to_expression_correct_l544_544058

open Real

noncomputable def nearest_integer_to_expression : ℝ :=
  let a := (3 + (sqrt 5))^6
  let b := (3 - (sqrt 5))^6
  let s := a + b
  floor s

theorem nearest_integer_to_expression_correct :
  nearest_integer_to_expression = 74608 :=
begin
  sorry
end

end nearest_integer_to_expression_correct_l544_544058


namespace square_has_correct_area_l544_544519

noncomputable theory

def side_lengths := (7.2 : ℝ, 9.5 : ℝ, 11.3 : ℝ)

def perimeter_of_triangle (a b c : ℝ) : ℝ :=
  a + b + c

def side_of_square (p : ℝ) : ℝ :=
  p / 4

def area_of_square (s : ℝ) : ℝ :=
  s * s

theorem square_has_correct_area :
  perimeter_of_triangle side_lengths.1 side_lengths.2 side_lengths.3 = 28 →
  area_of_square (side_of_square 28) = 49 :=
by
  intro h
  sorry

end square_has_correct_area_l544_544519


namespace annie_can_see_aaron_l544_544890

def relative_speed (v1 v2 : ℕ) : ℕ := v1 - v2
def time_to_travel (d v : ℕ) : Rational := d / v

theorem annie_can_see_aaron :
  let distance := 1 / 4;        -- 1/4 mile
  let annie_speed := 10;        -- 10 mph
  let aaron_speed := 6;         -- 6 mph
  let rel_speed := relative_speed annie_speed aaron_speed;  -- Relative speed
  let time_to_reach := time_to_travel distance rel_speed;   -- Time to reach Aaron
  let time_to_pass := time_to_travel distance rel_speed;    -- Time to be 1/4 mile ahead
  let total_time_hours := time_to_reach + time_to_pass;     -- Total time in hours
  let total_time_minutes := total_time_hours * 60;          -- Convert total time to minutes
  total_time_minutes = 7.5 := 
sorry

end annie_can_see_aaron_l544_544890


namespace range_of_a_undefined_sqrt_l544_544271

theorem range_of_a_undefined_sqrt (a : ℝ) : (∀ x : ℝ, x = sqrt (2 * a - 1) → False) ↔ a < 1 / 2 :=
by
  sorry

end range_of_a_undefined_sqrt_l544_544271


namespace inequality_proof_l544_544208

variable {R : Type*} [LinearOrderedField R]

theorem inequality_proof (n : ℕ) (x : Fin n → R) (m : R) (a s : R) 
  (hxi_pos : ∀ i, 0 < x i) 
  (hm_pos : 0 < m) 
  (ha_nonneg : 0 ≤ a) 
  (hs_sum : (∑ i, x i) = s) 
  (hs_le_n : s ≤ n) : 
  (∏ i, (x i ^ m + 1 / (x i ^ m) + a)) ≥ (( (s / n : R) ^ m + (n / s : R) ^ m + a ) ^ n) := 
sorry

end inequality_proof_l544_544208


namespace solve_for_k_l544_544956

noncomputable def base_k_representation (k : ℕ) (r : ℚ) := 
  ∑' n : ℕ, r.num * k^(-n*r.denom) / (1 - k^(-n*r.denom))

theorem solve_for_k (k : ℕ) (h_positive : k > 0) 
  (h_representation : base_k_representation k (2 * k^1 + 3 * k^2) = 7 / 51) : 
  k = 16 :=
by
  sorry

end solve_for_k_l544_544956


namespace relation_among_a_b_c_l544_544707

variable {f : ℝ → ℝ}

-- Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def monotonic_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Logarithmic expressions
def log_sqrt2_inv_sqrt3 : ℝ := -Real.logb (Float.sqrt 3) (Float.sqrt 2)
def log_sqrt3_inv_sqrt2 : ℝ := -Real.logb (Float.sqrt 2) (Float.sqrt 3)

-- Variables a, b, c based on function f
def a : ℝ := f (Real.logb (Float.sqrt 3) (Float.sqrt 2))
def b : ℝ := f (Real.logb (Float.sqrt 2) (Float.sqrt 3))
def c : ℝ := f 2

-- Theorem statement to prove b < a < c
theorem relation_among_a_b_c (hf_even : even_function f) (hf_mono : monotonic_increasing_on_nonneg f) : b < a ∧ a < c := by
  sorry

end relation_among_a_b_c_l544_544707


namespace odd_function_definition_l544_544836

-- Definitions and conditions from the problem statement
def f (x : ℝ) : ℝ := if x > 0 then Real.log x / Real.log 2 - 2 * x else 0

-- The statement to be proved
theorem odd_function_definition (x : ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_pos : ∀ x, 0 < x → f x = Real.log x / Real.log 2 - 2 * x):
  x < 0 → f x = -Real.log (-x) / Real.log 2 - 2 * x :=
by
  sorry

end odd_function_definition_l544_544836


namespace magnitude_of_OP_l544_544986

def point (α : Type) := α × α × α

def M : point ℝ := (5, 4, 3)

def proj_oyz (p : point ℝ) : point ℝ :=
  (0, p.2.1, p.2.2)

def O : point ℝ := (0, 0, 0)

def distance (p q : point ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2)

theorem magnitude_of_OP :
  distance O (proj_oyz M) = 5 :=
sorry

end magnitude_of_OP_l544_544986


namespace rounding_nearest_hundredth_does_not_round_to_3456_l544_544821

theorem rounding_nearest_hundredth_does_not_round_to_3456 :
  ∀ (n : ℝ), (n = 34.5597 ∨ n = 34.555 ∨ n = 34.5603 ∨ n = 34.5541 ∨ n = 34.563) →
  (n.round (rounding := rounding_mode.Nearest) / 0.01).round = 34.56 ↔ n ≠ 34.5541 :=
by sorry

end rounding_nearest_hundredth_does_not_round_to_3456_l544_544821


namespace exist_colored_points_r_gt_pi_div_sqrt3_exist_colored_points_r_gt_pi_div_2_l544_544321

theorem exist_colored_points_r_gt_pi_div_sqrt3 (r : ℝ) (hr : r > π / Real.sqrt 3) 
    (coloring : ℝ × ℝ → Prop) : 
    (∀ p, 0 ≤ p.1 ∧ p.1^2 + p.2^2 < r^2 → coloring p ∨ ¬coloring p) → 
    ∃ A B : ℝ × ℝ, (0 ≤ A.1 ∧ 0 ≤ B.1 ∧ A.1^2 + A.2^2 < r^2 ∧ B.1^2 + B.2^2 < r^2 ∧ dist A B = π ∧ coloring A = coloring B) :=
sorry

theorem exist_colored_points_r_gt_pi_div_2 (r : ℝ) (hr : r > π / 2)
    (coloring : ℝ × ℝ → Prop) : 
    (∀ p, 0 ≤ p.1 ∧ p.1^2 + p.2^2 < r^2 → coloring p ∨ ¬coloring p) → 
    ∃ A B : ℝ × ℝ, (0 ≤ A.1 ∧ 0 ≤ B.1 ∧ A.1^2 + A.2^2 < r^2 ∧ B.1^2 + B.2^2 < r^2 ∧ dist A B = π ∧ coloring A = coloring B) :=
sorry

end exist_colored_points_r_gt_pi_div_sqrt3_exist_colored_points_r_gt_pi_div_2_l544_544321


namespace sum_of_digits_is_23_l544_544026

def digits_sum_of_number_with_product_810 (a b c d : ℕ) : Prop :=
  (a * b * c * d = 810) ∧ (list.nodup [a, b, c, d]) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧ (1 ≤ d ∧ d ≤ 9)

theorem sum_of_digits_is_23 {a b c d : ℕ} (h : digits_sum_of_number_with_product_810 a b c d) :
  a + b + c + d = 23 := sorry

end sum_of_digits_is_23_l544_544026


namespace find_a_l544_544311

theorem find_a :
  (∀ x : ℝ, log 7 (log 3 (log 2 x)) = 0 → x = 8) →
  ∀ x a : ℝ, (x = 8) → (a = x^(1/3)) → (a = 2) :=
by
  intros h x a hx ha
  rw [hx] at ha
  rw [ha]
  norm_num
  sorry

end find_a_l544_544311


namespace conjugate_of_z_l544_544628

open Complex

theorem conjugate_of_z (z : ℂ) (h : z * (1 + Complex.i) = 1) : conj z = (1 / 2) + (1 / 2) * Complex.i :=
sorry

end conjugate_of_z_l544_544628


namespace find_r_l544_544245

variable (n : ℕ) (q r : ℝ)

-- n must be a positive natural number
axiom n_pos : n > 0

-- q is a positive real number and not equal to 1
axiom q_pos : q > 0
axiom q_ne_one : q ≠ 1

-- Define the sequence sum S_n according to the problem statement
def S_n (n : ℕ) (q r : ℝ) : ℝ := q^n + r

-- The goal is to prove that the correct value of r is -1
theorem find_r : r = -1 :=
sorry

end find_r_l544_544245


namespace solve_for_s_l544_544927

theorem solve_for_s (s : ℚ) (h : 8 = 2^(3*s - 2)) : s = 5/3 := sorry

end solve_for_s_l544_544927


namespace pet_store_cages_l544_544457

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h₁ : initial_puppies = 78)
(h₂ : sold_puppies = 30) (h₃ : puppies_per_cage = 8) : (initial_puppies - sold_puppies) / puppies_per_cage = 6 :=
by
  -- assumptions: initial_puppies = 78, sold_puppies = 30, puppies_per_cage = 8
  -- goal: (initial_puppies - sold_puppies) / puppies_per_cage = 6
  sorry

end pet_store_cages_l544_544457


namespace radius_of_sphere_in_truncated_cone_l544_544525

theorem radius_of_sphere_in_truncated_cone (r1 r2 : ℝ) (hr1 : r1 = 24) (hr2 : r2 = 8) :
  ∃ (r : ℝ), r = 8 * sqrt 3 ∧
  -- conditions that the sphere is tangent to both bases and the lateral surface
  -- are implicitly used in the mathematical formulation
  true :=
by
  sorry

end radius_of_sphere_in_truncated_cone_l544_544525


namespace inverse_function_correct_l544_544928

theorem inverse_function_correct (x : ℝ) (h_nonneg : x ≥ 0) : 
  let f := λ x, Real.sqrt x in  
  let f_inv := λ x, x * x in
  f_inv 2 = 4 :=
by
  sorry

end inverse_function_correct_l544_544928


namespace magician_assistant_trick_l544_544489

/-- A coin can be either heads or tails. -/
inductive Coin
| heads : Coin
| tails : Coin

/-- Given a cyclic arrangement of 11 coins, there exists at least one pair of adjacent coins with the same face. -/
theorem magician_assistant_trick (coins : Fin 11 → Coin) : 
  ∃ i : Fin 11, coins i = coins (i + 1) := 
by
  sorry

end magician_assistant_trick_l544_544489


namespace loss_percentage_l544_544139

variable (C : ℝ) -- Cost price of the article

def original_selling_price : ℝ := 1.35 * C -- Selling price with 35% profit

def new_selling_price : ℝ := (2 / 3) * original_selling_price C -- New selling price at 2/3 of the original selling price

theorem loss_percentage (hC : C > 0) : (C - new_selling_price C) / C * 100 = 10 :=
by
  have h1 : original_selling_price C = 1.35 * C := rfl
  have h2 : new_selling_price C = (2 / 3) * (1.35 * C) := by rw [original_selling_price]
  rw [new_selling_price, h2]
  have h3 : new_selling_price C = 0.9 * C := by ring
  rw [h3]
  sorry

end loss_percentage_l544_544139


namespace num_real_solutions_eq_63_l544_544778

theorem num_real_solutions_eq_63 : (set.countb {x : ℝ | x / 100 = Real.sin x}) = 63 := 
sorry

end num_real_solutions_eq_63_l544_544778


namespace tan_alpha_value_l544_544613

-- Definitions & Conditions
variables (α : ℝ)
def alpha_in_range : Prop := 0 < α ∧ α < π
def sin_plus_cos (α : ℝ) := sin α + cos α = -7 / 13

-- Statement of the theorem
theorem tan_alpha_value (h1 : alpha_in_range α) (h2 : sin_plus_cos α) : tan α = -5 / 12 :=
by sorry

end tan_alpha_value_l544_544613


namespace consecutive_circles_in_directions_l544_544964

-- Define the hexagonal grid structure and total circles
def hexagonal_grid (total_circles : ℕ) : Prop :=
  total_circles = 33

-- Define the condition to select 3 consecutive circles in any one direction.
def select_3_consecutive (total_ways : ℕ) : Prop :=
  total_ways = 57

-- Prove the number of ways to select three consecutive circles in any direction.
theorem consecutive_circles_in_directions {total_circles total_ways : ℕ}
  (h : hexagonal_grid total_circles)
  (c : select_3_consecutive total_ways) : 
  total_circles = 33 → total_ways = 57 :=
by
  intros h_total_circles
  rw [h] at h_total_circles
  intros c
  rw [c]
  sorry -- Proof to be completed

end consecutive_circles_in_directions_l544_544964


namespace petya_series_sum_l544_544359

theorem petya_series_sum (n k : ℕ) (h1 : (n + k) * (k + 1) = 20 * (n + 2 * k)) 
                                      (h2 : (n + k) * (k + 1) = 60 * n) :
  n = 29 ∧ k = 29 :=
by
  sorry

end petya_series_sum_l544_544359


namespace distance_from_foci_to_asymptotes_proof_l544_544016

def distance_from_foci_to_asymptotes : Prop :=
  let hyperbola_eq := ∀ x y : ℝ, x^2 - y^2 = 1
  let foci := {p : ℝ × ℝ | p = (sqrt 2, 0) ∨ p = (-sqrt 2, 0)}
  let asymptotes := {l : ℝ × ℝ → ℝ | l = (λ (p : ℝ × ℝ), p.1 + p.2) ∨ l = (λ (p : ℝ × ℝ), p.1 - p.2)}
  ∀ (f : ℝ × ℝ) (a : ℝ × ℝ → ℝ),
    f ∈ foci → a ∈ asymptotes → 
    (∣ a f ∣ / sqrt 2 = 1)

theorem distance_from_foci_to_asymptotes_proof : distance_from_foci_to_asymptotes :=
begin
  sorry
end

end distance_from_foci_to_asymptotes_proof_l544_544016


namespace total_oranges_and_apples_l544_544521

-- Given conditions as definitions
def bags_with_5_oranges_and_7_apples (m : ℕ) : ℕ × ℕ :=
  (5 * m + 1, 7 * m)

def bags_with_9_oranges_and_7_apples (n : ℕ) : ℕ × ℕ :=
  (9 * n, 7 * n + 21)

theorem total_oranges_and_apples (m n : ℕ) (k : ℕ) 
  (h1 : (5 * m + 1, 7 * m) = (9 * n, 7 * n + 21)) 
  (h2 : 4 * n ≡ 1 [MOD 5]) : 85 = 36 + 49 :=
by
  sorry

end total_oranges_and_apples_l544_544521


namespace simplify_expression_l544_544378

def a : ℕ := 1 + 27
def b : ℕ := 8 + nat.cbrt 27
theorem simplify_expression (c : ℕ) (a b : ℕ) (h1 : a = 1 + 27) (h2 : b = 8 + nat.cbrt 27) : nat.cbrt a * nat.cbrt b = nat.cbrt 308 :=
by
  sorry

end simplify_expression_l544_544378


namespace find_ED_length_l544_544834

-- Definitions and given conditions
variables (A B C D E: Type) [AddCommGroup A] 
variables (angle_B : A → A) (angle_D : A → A) (AC BD : A → Set A)
variables (x z : ℝ)
variable (cyclic_quadrilateral : A → A → A → A → Prop)
variable (intersect : A → A → Set A → Prop)
variable (measure_angle : A → ℝ)
variables (BCD : A → ℝ)
variable (BE_len AC_len ED_len : A → ℝ)
variable (isosceles_conditions : BCD = 150 ∧ measure_angle B = measure_angle D ∧ BE_len E = x ∧ AC_len E = z)

-- Given conditions
axiom cyclic_quad : cyclic_quadrilateral A B C D
axiom angle_eq_B_D : angle_B B = angle_D D
axiom intersection : intersect AC BD {E}
axiom angle_BCD : measure_angle BCD = 150
axiom BE_length : BE_len E = x
axiom AC_length : AC_len E = z

-- The statement we want to prove in Lean
theorem find_ED_length :
  ∃ (ED_length : ℝ), ED_length = (z - 2 * x) / 2 := sorry

end find_ED_length_l544_544834


namespace circuit_boards_fail_inspection_l544_544287

theorem circuit_boards_fail_inspection (P F : ℝ) (h1 : P + F = 3200)
    (h2 : (1 / 8) * P + F = 456) : F = 64 :=
by
  sorry

end circuit_boards_fail_inspection_l544_544287


namespace arrange_moon_l544_544920

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ways_to_arrange_moon : ℕ :=
  let total_letters := 4
  let repeated_O_count := 2
  factorial total_letters / factorial repeated_O_count

theorem arrange_moon : ways_to_arrange_moon = 12 := 
by {
  sorry -- Proof is omitted as instructed
}

end arrange_moon_l544_544920


namespace ellipse_equation_fixed_point_l544_544219

-- Define the ellipse and conditions
def ellipse_is_standard (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ eccentricity a b = 1 / 2 ∧ (0, sqrt 3) ∈ ellipse_pts a b

def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 - (b ^ 2 / a ^ 2))

def ellipse_pts (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1

def intersects_line (a b : ℝ) (k m : ℝ) : Prop :=
  ∃ x1 y1 x2 y2,
    (y1 = k * x1 + m) ∧ (ellipse_pts a b (x1, y1)) ∧ 
    (y2 = k * x2 + m) ∧ (ellipse_pts a b (x2, y2)) ∧ (x1 ≠ x2)

def line_passes_through_fixed_point (k : ℝ) : Prop :=
  ∀ m, intersects_line 2 (sqrt 3) k m → (m = -2 * k / 7)

-- The theorem to be proved
theorem ellipse_equation_fixed_point : 
  (∃ (a b : ℝ), a = 2 ∧ b = sqrt 3 ∧ ellipse_is_standard a b ∧ 
    line_passes_through_fixed_point k := sorry

end ellipse_equation_fixed_point_l544_544219


namespace range_of_t_minus_1_over_t_minus_3_l544_544241

variable {f : ℝ → ℝ}

-- Function conditions: monotonically decreasing and odd
axiom f_mono_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Condition on the real number t
variable {t : ℝ}
axiom f_condition : f (t^2 - 2 * t) + f (-3) > 0

-- Question: Prove the range of (t-1)/(t-3)
theorem range_of_t_minus_1_over_t_minus_3 (h : -1 < t ∧ t < 3) : 
  ((t - 1) / (t - 3)) < 1/2 :=
  sorry

end range_of_t_minus_1_over_t_minus_3_l544_544241


namespace landscape_length_l544_544391

theorem landscape_length (b : ℝ) 
  (h1 : ∀ (l : ℝ), l = 8 * b) 
  (A : ℝ)
  (h2 : A = 8 * b^2)
  (Playground_area : ℝ)
  (h3 : Playground_area = 1200)
  (h4 : Playground_area = (1 / 6) * A) :
  ∃ (l : ℝ), l = 240 :=
by 
  sorry

end landscape_length_l544_544391


namespace total_pixels_l544_544868

theorem total_pixels (width height dpi : ℕ) (h_width : width = 21) (h_height : height = 12) (h_dpi : dpi = 100) :
  width * dpi * height * dpi = 2520000 := 
by
  rw [h_width, h_height, h_dpi]
  simp
  sorry

end total_pixels_l544_544868


namespace range_of_b_over_a_l544_544200

noncomputable def f (a b x : ℝ) : ℝ := (a * x - b / x - 2 * a) * Real.exp x

noncomputable def f' (a b x : ℝ) : ℝ := (b / x^2 + a * x - b / x - a) * Real.exp x

theorem range_of_b_over_a (a b : ℝ) (h₀ : a > 0) (h₁ : ∃ x : ℝ, 1 < x ∧ f a b x + f' a b x = 0) : 
  -1 < b / a := sorry

end range_of_b_over_a_l544_544200


namespace gcd_m_n_l544_544430

def m : ℕ := 131^2 + 243^2 + 357^2
def n : ℕ := 130^2 + 242^2 + 358^2

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_l544_544430


namespace length_of_BD_l544_544686

theorem length_of_BD (
  {A B C D : Type*}
  [inhabited A] [inhabited B] [inhabited C] [inhabited D]
  (r : EuclideanSpace ℝ (fin 2))
  (h₁ : B = (0, 0))
  (h₂ : A = (45, 60))
  (h₃ : C = (125, 0))
  (h₄ : D = (45, 0))
  (h₅ : dist A B = 75)
  (h₆ : dist A C = 100)
  (h₇ : D ∈ line_through(B, C))
  (h₈ : dist A D ⊥ dist B C)) :
  dist B D = 45 :=
sorry

end length_of_BD_l544_544686


namespace avg_weight_of_first_group_l544_544670

variable (G : ℕ) -- number of girls in the first group
variable (W1 W2 : ℝ) -- total weights of the first and second groups

-- Conditions:
-- 1. The average weight of the first group of girls is 50.25 kg.
def avg_weight_first_group := W1 = G * 50.25

-- 2. The second group's average weight is 45.15 kg, consisting of 8 girls.
def weight_second_group := W2 = 8 * 45.15

-- 3. The overall average weight of the class (G + 8 girls) is 48.55 kg.
def overall_avg_weight := (G + 8) * 48.55 = W1 + W2

-- Theorem: The average weight of the first group of girls is 50.25 kg.
theorem avg_weight_of_first_group : avg_weight_first_group G W1 ∧ weight_second_group W2 ∧ overall_avg_weight G W1 W2 → W1 / G = 50.25 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [avg_weight_first_group, weight_second_group, overall_avg_weight] at h1 h3 h4
  sorry -- Proof here

end avg_weight_of_first_group_l544_544670


namespace median_of_data_set_l544_544216

theorem median_of_data_set :
  ∃ x : ℕ, (6 + 3 + 8 + x + 7) / 5 = 6 ∧ 
  let data_set := [6, 3, 8, x, 7] in
  let sorted_set := data_set.qsort (≤) in
  sorted_set.nth 2 = 6 :=
by
  sorry

end median_of_data_set_l544_544216


namespace greatest_3_digit_base_12_div_by_7_l544_544429

-- Define the conditions
def greatest_three_digit_base_12 := 1727
def div_by_seven (n : ℕ) : Prop := n % 7 = 0

-- Define the target number in base 12
def to_base_12 (n : ℕ) : string :=
  let d1 := n / (12 * 12)
  let r1 := n % (12 * 12)
  let d2 := r1 / 12
  let d3 := r1 % 12
  (char.ofInt (d1 + if d1 >= 10 then 55 else 48)).toString ++
  (char.ofInt (d2 + if d2 >= 10 then 55 else 48)).toString ++
  (char.ofInt (d3 + if d3 >= 10 then 55 else 48)).toString

-- Define the proof problem
theorem greatest_3_digit_base_12_div_by_7 :
  ∃ n, n ≤ greatest_three_digit_base_12 ∧ div_by_seven n ∧ to_base_12 n = "BB6" :=
sorry

end greatest_3_digit_base_12_div_by_7_l544_544429


namespace gary_money_left_l544_544597

theorem gary_money_left (initial_amount spent_amount remaining_amount : ℕ)
  (h1 : initial_amount = 73)
  (h2 : spent_amount = 55)
  (h3 : remaining_amount = 18) : initial_amount - spent_amount = remaining_amount := 
by 
  sorry

end gary_money_left_l544_544597


namespace sum_2014_series_l544_544150

theorem sum_2014_series :
  (∑ n in Finset.range 2014, (1 : ℚ) / ((n+1) * (n+2))) = 2014 / 2015 := 
by
  sorry

end sum_2014_series_l544_544150


namespace rectangular_to_polar_coordinates_l544_544567

theorem rectangular_to_polar_coordinates :
  ∃ r θ, (r > 0) ∧ (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ (r, θ) = (5, 7 * Real.pi / 4) :=
by
  sorry

end rectangular_to_polar_coordinates_l544_544567


namespace probability_interval_70_110_estimate_students_80_100_l544_544286

noncomputable def normal_distribution : Type :=
{ μ : ℝ,
  σ^2 : ℝ+,
  P (a b : ℝ) : ℝ }

axiom xi : normal_distribution
axiom mean_xi : xi.μ = 90
axiom variance_xi : xi.σ^2 = 100

-- Proof problem for question 1:
theorem probability_interval_70_110 :
  xi.P 70 110 = 0.9544 :=
sorry

-- Proof problem for question 2:
theorem estimate_students_80_100 (total_students : ℕ) :
  total_students = 2000 →
  xi.P 80 100 * total_students = 1366 :=
sorry

end probability_interval_70_110_estimate_students_80_100_l544_544286


namespace conjugate_of_complex_number_l544_544762

-- Define the given complex number and its conjugate
def z : ℂ := 1 / (1 - I)
def z_conj : ℂ := conj z

-- State the problem as a theorem in Lean 4
theorem conjugate_of_complex_number : z_conj = (1 / 2 - 1 / 2 * I) :=
sorry

end conjugate_of_complex_number_l544_544762


namespace sphere_coloring_area_balance_l544_544974

-- Definitions for the problem
def great_circle (S : Type) (n : ℕ) := 
  ∃ (circles : fin n → set S), ∀ i, is_great_circle (circles i) S

def good_coloring_exists (S : Type) (n : ℕ) := 
  ∀ (C : coloring S), 
  (∀ x y, adjacent x y → C x ≠ C y) → 
  (let (black_areas, white_areas) := 
    C.regions.partition (λ r, C.color r = black) in 
    black_areas.sum_area = white_areas.sum_area)

-- Statement of the theorem in Lean
theorem sphere_coloring_area_balance (S : Type) (n : ℕ) 
  (h1: ∃ circles : fin n → set S, ∀ i, is_great_circle (circles i) S)
  (h2: ∀ x y, adjacent x y → adjacent_regions_different_colors S n) :
  good_coloring_exists S n ↔ odd n := 
sorry

end sphere_coloring_area_balance_l544_544974


namespace contrapositive_equivalence_l544_544086

variable {R S : Prop}

theorem contrapositive_equivalence (h : R → S) : ¬S → ¬R := 
by intent sorry

end contrapositive_equivalence_l544_544086


namespace crossing_time_proof_l544_544831

noncomputable def total_time (distance : ℝ) (run_speed : ℝ) (bike_speed : ℝ) : ℝ :=
  let trip_time := distance / bike_speed
  let return_time_1 := (trip_time * run_speed) / bike_speed
  let return_time_2 := (2 * trip_time * run_speed) / bike_speed
  3 * trip_time + return_time_1 + return_time_2

theorem crossing_time_proof :
  total_time 300 10 50 = 21.6 :=
by
  unfold total_time
  norm_num
  sorry

end crossing_time_proof_l544_544831


namespace moon_arrangement_l544_544919

theorem moon_arrangement : 
  let M_count := 1
  let O_count := 2
  let N_count := 1
  let total_letters := 4
  ∑ perm : List Nat, perm.permutations.length = total_letters! // (M_count! * O_count! * N_count!) :=
  12
 :=
begin
  -- Definitions from the condition
  have M_count := 1,
  have O_count := 2,
  have N_count := 1,
  have total_letters := 4,
  
  -- Applying formulas for permutation counts
  let num_unique_arrangements := total_letters.factorial / (M_count.factorial * O_count.factorial * N_count.factorial),
  show num_unique_arrangements = 12 from sorry
end

end moon_arrangement_l544_544919


namespace ellipse_meets_sine_more_than_8_points_l544_544533

noncomputable def ellipse_intersects_sine_curve_more_than_8_times (a b : ℝ) (h k : ℝ) :=
  ∃ p : ℕ, p > 8 ∧ 
  ∃ (x y : ℝ), 
    (∃ (i : ℕ), y = Real.sin x ∧ 
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)

theorem ellipse_meets_sine_more_than_8_points : 
  ∀ (a b h k : ℝ), ellipse_intersects_sine_curve_more_than_8_times a b h k := 
by sorry

end ellipse_meets_sine_more_than_8_points_l544_544533


namespace max_unique_triangles_l544_544559

open Function

-- Define a set of triangles with integer sides (a, b, c)
def is_triangle (a b c : ℕ) := a + b > c ∧ a + c > b ∧ b + c > a

-- Define a set of unique triangles (no congruent or similar elements allowed)
def unique_triangles (S : Finset (ℕ × ℕ × ℕ)) : Prop :=
  ∀ t₁ t₂ ∈ S, t₁ ≠ t₂ → ¬(∃ (k : ℕ), k ≠ 0 ∧ t₁ = (k * t₂.1, k * t₂.2, k * t₂.3))

-- The set of all triangles with integer sides less than 8
def all_triangles : Finset (ℕ × ℕ × ℕ) :=
  {t | let ⟨a, b, c⟩ := t in a < 8 ∧ b < 8 ∧ c < 8 ∧ is_triangle a b c}.to_finset

-- Given the conditions, we want to show the largest number of unique elements in S is 11
theorem max_unique_triangles : ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S ⊆ all_triangles ∧ unique_triangles S ∧ S.card = 11 :=
sorry

end max_unique_triangles_l544_544559


namespace sum_of_prime_factors_240255_l544_544433

def is_prime (n : ℕ) : Prop := nat.prime n

def prime_factors (n : ℕ) : list ℕ := (nat.factors n).erase_dup

def sum_of_prime_factors (n : ℕ) : ℕ := (prime_factors n).sum

theorem sum_of_prime_factors_240255 : sum_of_prime_factors 240255 = 966 :=
by {
  -- Verification for each relevant prime factor and their correctness has to be performed here.
  -- The Lean prover would ideally validate the prime factorizations and calculations.
  sorry
}

end sum_of_prime_factors_240255_l544_544433


namespace range_of_x_sqrt_eq_l544_544272

theorem range_of_x_sqrt_eq (x : ℝ) (h : sqrt (x / (1 - x)) = sqrt x / sqrt (1 - x)) : 
  0 ≤ x ∧ x < 1 :=
by {
  sorry
}

end range_of_x_sqrt_eq_l544_544272


namespace sum_of_squares_area_l544_544537

-- Define the points and lengths
variables (A E H B : Type)
variable (BE : ℝ) -- Length of BE
variable (AH : ℝ) -- Length of AH
variable (right_angle_EAB : IsRightAngle A E B)
variable (right_triangle_EAH : IsRightTriangle E A H)

-- Assume the given conditions
axiom BE_length : BE = 12
axiom AH_length : AH = 5

-- The theorem to prove
theorem sum_of_squares_area : 
  BE = 12 → AH = 5 → 
  IsRightAngle A E B → IsRightTriangle E A H → 
  let AB_sq := AB^2 in
  let AE_sq := AE^2 in 
  let EH_sq := (math.sqrt (AE_sq + AH^2))^2 in
  AB_sq + AE_sq + EH_sq = 169 :=
by sorry

end sum_of_squares_area_l544_544537


namespace probability_of_sum_divisible_by_six_l544_544278

-- Define the set of numbers
def num_set : Finset ℕ := {1, 2, 3, 4}

-- Define the target divisor
def divisor : ℕ := 6

-- Define the function to compute probability
def prob_sum_divisible_by_six : ℚ :=
  let combinations := num_set.powerset.filter (λ s, s.card = 3)
  let favorable := combinations.filter (λ s, s.sum id % divisor = 0)
  favorable.card / combinations.card

-- The statement to prove
theorem probability_of_sum_divisible_by_six :
  prob_sum_divisible_by_six = 1 / 4 :=
sorry

end probability_of_sum_divisible_by_six_l544_544278


namespace time_point_P_moves_l544_544308

theorem time_point_P_moves (A B C : ℝ) (t : ℝ) (hABC : ∠ B = 90) (hAB : B - A = 6) (hBC : C - B = 8) 
  (h_speed_P : t = (1 : ℝ) * t) (h_speed_Q : t = (1 / 2) * (C - B)) 
  (h_area : 1 / 2 * (B - t) * 2 * t = 5) : t = 1 :=
sorry

end time_point_P_moves_l544_544308


namespace solve_for_x_and_compute_value_l544_544381

theorem solve_for_x_and_compute_value (x : ℝ) (h : 5 * x - 3 = 15 * x + 15) : 6 * (x + 5) = 19.2 := by
  sorry

end solve_for_x_and_compute_value_l544_544381


namespace point_on_circle_x_value_l544_544682

/-
In the xy-plane, the segment with endpoints (-3,0) and (21,0) is the diameter of a circle.
If the point (x,12) is on the circle, then x = 9.
-/
theorem point_on_circle_x_value :
  let c := (9, 0) -- center of the circle
  let r := 12 -- radius of the circle
  let circle := {p | (p.1 - 9)^2 + p.2^2 = 144} -- equation of the circle
  ∀ x : Real, (x, 12) ∈ circle → x = 9 :=
by
  intros
  sorry

end point_on_circle_x_value_l544_544682


namespace eventually_composite_appending_threes_l544_544799

theorem eventually_composite_appending_threes (n : ℕ) :
  ∃ n' : ℕ, n' = 10 * n + 3 ∧ ∃ k : ℕ, k > 0 ∧ (3 * k + 3) % 7 ≠ 1 ∧ (3 * k + 3) % 7 ≠ 2 ∧ (3 * k + 3) % 7 ≠ 3 ∧
  (3 * k + 3) % 7 ≠ 5 ∧ (3 * k + 3) % 7 ≠ 6 :=
sorry

end eventually_composite_appending_threes_l544_544799


namespace movie_box_office_growth_l544_544777

theorem movie_box_office_growth 
  (x : ℝ) 
  (r₁ r₃ : ℝ) 
  (h₁ : r₁ = 1) 
  (h₃ : r₃ = 2.4) 
  (growth : r₃ = (1 + x) ^ 2) : 
  (1 + x) ^ 2 = 2.4 :=
by sorry

end movie_box_office_growth_l544_544777


namespace sin_210_degrees_l544_544898

open Real

theorem sin_210_degrees : sin (210 * π / 180) = -1/2 :=
by
  -- Condition 1: General trigonometric identity for angle + 180 degrees
  have h_identity : ∀ θ, sin (θ + π) = -sin θ,
    from λ θ, by rw [sin_add, sin_pi, cos_pi]; simp,
  -- Apply the identity with θ = 30 degrees
  have h_210 : sin (210 * π / 180) = sin (π + π / 6),
    by norm_num,
  rw h_210,
  rw h_identity (π / 6),
  -- Condition 2: Known value for sine of 30 degrees
  have h_sin_30 : sin (π / 6) = 1 / 2,
    from sin_pi_div_six,
  rw h_sin_30,
  norm_num

end sin_210_degrees_l544_544898


namespace base_price_lowered_percentage_l544_544869

theorem base_price_lowered_percentage (P : ℝ) (new_price final_price : ℝ) (x : ℝ)
    (h1 : new_price = P - (x / 100) * P)
    (h2 : final_price = 0.9 * new_price)
    (h3 : final_price = P - (14.5 / 100) * P) :
    x = 5 :=
  sorry

end base_price_lowered_percentage_l544_544869


namespace inequality_proof_l544_544328

theorem inequality_proof (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) : 
  1 / (a * b^2) > 1 / (a^2 * b) :=
sorry

end inequality_proof_l544_544328


namespace remaining_string_length_l544_544414

theorem remaining_string_length (original_length : ℝ) (given_to_Minyoung : ℝ) (fraction_used : ℝ) :
  original_length = 70 →
  given_to_Minyoung = 27 →
  fraction_used = 7/9 →
  abs (original_length - given_to_Minyoung - fraction_used * (original_length - given_to_Minyoung) - 9.56) < 0.01 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end remaining_string_length_l544_544414


namespace ellipse_equation_l544_544988

theorem ellipse_equation (a b : ℝ) (h_pos : a > 0) (focus_parabola : (2, 0) = (2, 0)) : 
  (a^2 = b^2 + 4) → (a^2 = 8) → (b^2 = 4) → 
  (∀ x y : ℝ, (x^2 / 8 + y^2 / 4 = 1 ↔ (x, y) lies_on_ellipse)) :=
by
  sorry

end ellipse_equation_l544_544988


namespace max_product_sum_1976_l544_544244

theorem max_product_sum_1976:
  ∃ x : ℕ → ℕ, (∀ i, 0 < x i) ∧ (∑ i in finset.range 1976, x i = 1976)
    ∧ (∏ i in finset.range 1976, x i = 2 * 3 ^ 658) := sorry

end max_product_sum_1976_l544_544244


namespace limit_of_function_l544_544097

open Real

theorem limit_of_function :
  (tendsto (λ x : ℝ, (6^(2 * x) - 7^(-2 * x)) / (sin (3 * x) - 2 * x)) (𝓝 0) (𝓝 (2 * log 42))) :=
sorry

end limit_of_function_l544_544097


namespace solve_for_t_l544_544279

theorem solve_for_t (t : ℝ) (h1 : x = 1 - 4 * t) (h2 : y = 2 * t - 2) : x = y → t = 1/2 :=
by
  sorry

end solve_for_t_l544_544279


namespace find_k_range_g_l544_544250

-- Define the function f(x) and provide the given condition that f(x) is even.
def f (x : ℝ) (k : ℝ) : ℝ := x * (2 / (2^x - 1) + k)

-- Define the condition that f(x) is an even function.
axiom f_even (k : ℝ) : ∀ x : ℝ, f x k = f (-x) k

-- The first part of the problem: determining the value of k such that f(x) is even.
-- We show that k must be -1 for f(x) to be even.
theorem find_k : k = -1 := by sorry

-- Define the function g(x) and the interval for x.
def g (x : ℝ) : ℝ := (2 / (2^x - 1)) - 1

-- The second part of the problem: finding the range of g(x) when x ∈ (0, 1].
theorem range_g : Set.range g = Set.Iic 1 := by sorry

end find_k_range_g_l544_544250


namespace convert_300_degree_to_radian_l544_544566

theorem convert_300_degree_to_radian : (300 : ℝ) * π / 180 = 5 * π / 3 :=
by
  sorry

end convert_300_degree_to_radian_l544_544566


namespace problem1_problem2_l544_544103

-- Definitions based on conditions
def f (x : ℝ) : ℝ := 2^x - 1
def S (n : ℕ) : ℝ := 2^n - 1
def a (n : ℕ) : ℝ := 2^(n - 1)
def c (n : ℕ) : ℝ := 6 * n * a n - n
def T (n : ℕ) : ℝ := 3 * (n - 1) * 2^(n + 1) + 6

-- Proof statement for problems:
theorem problem1 (n : ℕ) (hn : 0 < n) : S n = 2^n - 1 ∧ a n = 2^(n - 1) :=
sorry

theorem problem2 (n : ℕ) (hn : 0 < n) : T n = 3 * (n - 1) * 2^(n + 1) + 6 :=
sorry

end problem1_problem2_l544_544103


namespace prove_product_of_b_l544_544571

noncomputable def g (x b : ℝ) := b / (5 * x - 7)

noncomputable def g_inv (y b : ℝ) := (b + 7 * y) / (5 * y)

theorem prove_product_of_b (b1 b2 : ℝ) (h1 : g 3 b1 = g_inv (b1 + 2) b1) (h2 : g 3 b2 = g_inv (b2 + 2) b2) :
  b1 * b2 = -22.39 := by
  sorry

end prove_product_of_b_l544_544571


namespace part_a_least_L_pieces_part_b_tiling_l544_544093

section part_a
-- Definition of the L-shaped figure
structure LPiece where
  cells : Finset (ℕ × ℕ)
  is_L_shape : cells = {(0, 0), (0, 1), (1, 0)}

-- Problem Part (a)
theorem part_a_least_L_pieces :
  ∃ (n : ℕ), (∀ (placement : Finset (ℕ × ℕ)), 
    ((8 * 8) % 3 = 0 → ∀ (lpieces : Finset LPiece), 
      (Finset.card lpieces = n) ∧ 
      (∀ lpiece ∈ lpieces, ∀ (cell ∈ lpiece.cells), cell ∈ placement) ∧ 
      (placement.card = 64) ∧ 
      (∀ lpieces' : Finset LPiece, 
         (∀ lpiece ∈ lpieces’, ∀ cell ∈ lpiece.cells, cell ∈ placement) → 
         lpieces’.card ≥ lpieces.card)
  )) ∧ n = 22 :=
sorry
end part_a

section part_b
-- Definition of the grid and LPiece again
structure Grid (m n : ℕ) where
  cells : Finset (Fin (m) × Fin (n))

def remove_cell (g : Grid m n) (coord : (Fin m) × (Fin n)) : Grid m n := {
  cells := g.cells.erase coord
}

-- Problem Part (b)
theorem part_b_tiling (g : Grid 1987 1987) (c : Fin 1987 × Fin 1987):
  ∃ (lpieces : Finset LPiece), ∀ lpiece ∈ lpieces, ∀ cell ∈ lpiece.cells, cell ∉ (g.remove_cell c).cells :=
sorry
end part_b

end part_a_least_L_pieces_part_b_tiling_l544_544093


namespace sum_of_series_eq_15_over_26_l544_544912

theorem sum_of_series_eq_15_over_26 :
  ∑' (n : ℕ), (if n % 3 = 0 then (1 / 27^((n / 3) : ℕ))
        else if n % 3 = 1 then - (1 / (3 * 27^((n / 3) : ℕ)))
        else - (1 / (9 * 27^((n / 3) : ℕ)))) = (15 / 26) :=
by {
  sorry
}

end sum_of_series_eq_15_over_26_l544_544912


namespace probability_of_diff_ge_5_correct_l544_544416

noncomputable def probability_of_difference_ge_5 : ℚ :=
  let total_ways := 84 in
  let favorable_sets := 50 in
  favorable_sets / total_ways

theorem probability_of_diff_ge_5_correct :
  probability_of_difference_ge_5 = 25 / 42 :=
by
  sorry

end probability_of_diff_ge_5_correct_l544_544416


namespace eccentricity_and_equation_of_ellipse_l544_544993

def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def point_P (b : ℝ) : Prop :=
  (0, b)

def focus (c : ℝ) : Prop :=
  (c, 0)

def tan_angle_PFO (b c : ℝ) : Prop :=
  c = sqrt(2) * b ∧ b > 0

def slopes (k1 k2 : ℝ) : Prop :=
  k1 + k2 = 2

theorem eccentricity_and_equation_of_ellipse :
  ∀ (a b c : ℝ) (e : ℝ),
  a > b → b > 0 →
  ellipse a b ⟨a > b, b > 0⟩ →
  point_P b →
  focus c →
  tan_angle_PFO b c →
  slopes 0 0 →
  e = sqrt(6) / 3 ∧
  (a = sqrt(3) * b ∧ b ^ 2 = 1 → a = sqrt(3)) →
  ∀ x y : ℝ, x^2 / 3 + y^2 = 1 :=
by
  sorry

end eccentricity_and_equation_of_ellipse_l544_544993


namespace train_crossing_bridge_time_l544_544828

theorem train_crossing_bridge_time :
  ∀ (lt lb : ℕ) (v_kmph : ℕ),
  lt = 110 → lb = 190 → v_kmph = 60 → 
  (lt + lb : ℕ) / (v_kmph * 1000 / 3600) ≈ 18 :=
by
  intros lt lb v_kmph hlt hlb hv_kmph
  rw [hlt, hlb, hv_kmph]
  norm_num
  sorry

end train_crossing_bridge_time_l544_544828


namespace sequence_integer_k_l544_544784

theorem sequence_integer_k
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : ∀ n ≥ 1, 5^(a (n + 1) - a n) - 1 = 1 / (n + 2 / 3)) :
  ∃ k > 1, k = 41 ∧ a k ∈ ℤ :=
by
  sorry

end sequence_integer_k_l544_544784


namespace sin_2alpha_value_l544_544196

theorem sin_2alpha_value (α : ℝ) (h : sin α + cos α = 1 / 2) : sin (2 * α) = -3 / 4 :=
  sorry

end sin_2alpha_value_l544_544196


namespace at_least_one_even_l544_544135

theorem at_least_one_even (S : Finset ℕ) (h₁ : S.card = 12)
  (h₂ : (S.filter (λ n, even n)).card = 10)
  (h₃ : (S.filter (λ n, ¬(even n))).card = 2) :
  ∀ t ⦃T : Finset ℕ⦄, T ⊆ S → T.card = 3 → (∃ x ∈ T, even x) :=
by
  sorry

end at_least_one_even_l544_544135


namespace divide_triangle_l544_544929

-- Define an arbitrary triangle with a side of length 'a'
structure Triangle :=
  (A B C : Point)
  (side_length : ℝ)
  (side_eq : (distance A B) = a)

-- Define a function that states a triangle can be divided into 3 parts to get another triangle with side '2a'
theorem divide_triangle (a : ℝ) (T : Triangle) (h : T.side_length = a) : 
    ∃ T' : Triangle, T'.side_length = 2 * a := 
by 
  sorry

end divide_triangle_l544_544929


namespace mrs_jane_total_coins_l544_544131

theorem mrs_jane_total_coins (Jayden_coins Jason_coins : ℕ) (h1 : Jayden_coins = 300) (h2 : Jason_coins = Jayden_coins + 60) :
  Jayden_coins + Jason_coins = 660 :=
sorry

end mrs_jane_total_coins_l544_544131


namespace number_of_members_l544_544483

theorem number_of_members (n h : ℕ) (h1 : n * n * h = 362525) : n = 5 :=
sorry

end number_of_members_l544_544483


namespace total_students_l544_544406

theorem total_students : 
  let grade_3 := 19
  let grade_4 := 2 * grade_3
  let boys_2 := 10
  let girls_2 := 19
  let grade_2 := boys_2 + girls_2
  let total := grade_2 + grade_3 + grade_4
  in total = 86 :=
by
  let grade_3 := 19
  let grade_4 := 2 * grade_3
  let boys_2 := 10
  let girls_2 := 19
  let grade_2 := boys_2 + girls_2
  let total := grade_2 + grade_3 + grade_4
  sorry

end total_students_l544_544406


namespace max_consecutive_sum_le_1000_l544_544809

theorem max_consecutive_sum_le_1000 : 
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → (m * (m + 1)) / 2 < 1000) ∧ ¬∃ n' : ℕ, n < n' ∧ (n' * (n' + 1)) / 2 < 1000 :=
sorry

end max_consecutive_sum_le_1000_l544_544809


namespace magician_trick_l544_544508

def coin (a : Fin 11 → Bool) : Prop :=
  ∃ i : Fin 11, a i = a (i + 1) % 11

theorem magician_trick (a : Fin 11 → Bool) 
(H : ∃ i : Fin 11, a i = a (i + 1) % 11) : 
  ∃ j : Fin 11, j ≠ 0 ∧ a j = a 0 :=
sorry

end magician_trick_l544_544508


namespace trig_ineq_l544_544362

open Real

theorem trig_ineq (θ : ℝ) (h : sin θ * sqrt(1 - cos θ ^ 2) + cos θ * sqrt(1 - sin θ ^ 2) = 1) :
  ¬ (sin θ < 0 ∨ cos θ < 0) :=
sorry

end trig_ineq_l544_544362


namespace line_does_not_pass_through_third_quadrant_l544_544399

def linear_function (x : ℝ) : ℝ := - (1 / 2) * x + 1

theorem line_does_not_pass_through_third_quadrant : ¬ ∃ (x y : ℝ), (x < 0 ∧ y < 0) ∧ y = linear_function x := by
  sorry

end line_does_not_pass_through_third_quadrant_l544_544399


namespace find_amplitude_l544_544542

-- Definition of a that we need to prove.
def amplitude(a b c d : ℝ) : Prop :=
  ∀ (f : ℝ → ℝ),
    (f = λ x, a * Real.sin (b * x + c) + d) →
    (∀ x : ℝ, -1 ≤ f x ∧ f x ≤ 3) →
    (a = 2)

-- Now we write the lemma.
theorem find_amplitude : amplitude 2 b c d := sorry

end find_amplitude_l544_544542


namespace probability_white_ball_l544_544844

theorem probability_white_ball (num_black_balls num_white_balls : ℕ) 
  (black_balls : num_black_balls = 6) 
  (white_balls : num_white_balls = 5) : 
  (num_white_balls / (num_black_balls + num_white_balls) : ℚ) = 5 / 11 :=
by
  sorry

end probability_white_ball_l544_544844


namespace cookies_left_after_three_days_l544_544312

theorem cookies_left_after_three_days
  (initial_cookies : ℕ)
  (first_day_fraction_eaten : ℚ)
  (second_day_fraction_eaten : ℚ)
  (initial_value : initial_cookies = 64)
  (first_day_fraction : first_day_fraction_eaten = 3/4)
  (second_day_fraction : second_day_fraction_eaten = 1/2) :
  initial_cookies - (first_day_fraction_eaten * 64) - (second_day_fraction_eaten * ((1 - first_day_fraction_eaten) * 64)) = 8 :=
by
  sorry

end cookies_left_after_three_days_l544_544312


namespace total_running_time_l544_544366

-- Defining the conditions
def rudy_runs_first_part : ℕ := 5 * 10
def rudy_runs_second_part : ℕ := 4 * 9.5.toInt

-- The theorem statement to prove the total time
theorem total_running_time : rudy_runs_first_part + rudy_runs_second_part = 88 :=
by
  sorry

end total_running_time_l544_544366


namespace magician_assistant_trick_l544_544490

/-- A coin can be either heads or tails. -/
inductive Coin
| heads : Coin
| tails : Coin

/-- Given a cyclic arrangement of 11 coins, there exists at least one pair of adjacent coins with the same face. -/
theorem magician_assistant_trick (coins : Fin 11 → Coin) : 
  ∃ i : Fin 11, coins i = coins (i + 1) := 
by
  sorry

end magician_assistant_trick_l544_544490


namespace convert_mps_to_kmph_l544_544827

theorem convert_mps_to_kmph (v_mps : ℝ) (c : ℝ) (h_c : c = 3.6) (h_v_mps : v_mps = 20) : (v_mps * c = 72) :=
by
  rw [h_v_mps, h_c]
  sorry

end convert_mps_to_kmph_l544_544827


namespace remainder_when_divided_by_23_l544_544815

theorem remainder_when_divided_by_23 (y : ℕ) (h : y % 276 = 42) : y % 23 = 19 := by
  sorry

end remainder_when_divided_by_23_l544_544815


namespace valid_5_digit_count_l544_544645

def digits : list char := ['O', '0', '1', '2', '3']

-- Define a condition for a 5-digit number formed from the above list of digits
def is_valid_5_digit (s : list char) : Prop :=
  s.length = 5 ∧ (∀ ch ∈ s, ch ∈ digits)

-- Define the number of valid 5-digit numbers using the given digits
def number_of_valid_5_digit_numbers : ℕ :=
  -- calculation (actual proof required to compute this properly is omitted here)
  sorry -- placeholder for the actual proof

theorem valid_5_digit_count : number_of_valid_5_digit_numbers = 36 := 
by
  -- equivalent proof problem's conclusion provided here
  sorry -- placeholder for the actual proof

end valid_5_digit_count_l544_544645


namespace ratatouille_cost_per_quart_l544_544747

def eggplants := 88 * 0.22
def zucchini := 60.8 * 0.15
def tomatoes := 73.6 * 0.25
def onions := 43.2 * 0.07
def basil := (16 / 4) * 2.70
def bell_peppers := 12 * 0.20

def total_cost := eggplants + zucchini + tomatoes + onions + basil + bell_peppers
def yield := 4.5

def cost_per_quart := total_cost / yield

theorem ratatouille_cost_per_quart : cost_per_quart = 14.02 := 
by
  unfold cost_per_quart total_cost eggplants zucchini tomatoes onions basil bell_peppers 
  sorry

end ratatouille_cost_per_quart_l544_544747


namespace trig_identity_example_l544_544897

open Real -- Using the Real namespace for trigonometric functions

theorem trig_identity_example :
  sin (135 * π / 180) * cos (-15 * π / 180) + cos (225 * π / 180) * sin (15 * π / 180) = 1 / 2 :=
by 
  -- sorry to skip the proof steps
  sorry

end trig_identity_example_l544_544897


namespace magician_trick_l544_544506

def coin (a : Fin 11 → Bool) : Prop :=
  ∃ i : Fin 11, a i = a (i + 1) % 11

theorem magician_trick (a : Fin 11 → Bool) 
(H : ∃ i : Fin 11, a i = a (i + 1) % 11) : 
  ∃ j : Fin 11, j ≠ 0 ∧ a j = a 0 :=
sorry

end magician_trick_l544_544506


namespace line_curve_properties_l544_544963

noncomputable theory

-- Definitions based on given conditions
def parametric_line (t : ℝ) (φ : ℝ) : ℝ × ℝ :=
  (t * cos φ, 2 + t * sin φ)

def polar_curve (θ : ℝ) : ℝ :=
  8 * sin θ / cos^2 θ

-- Theorem based on questions and correct answer
theorem line_curve_properties (φ θ : ℝ) (t : ℝ) (hφ : 0 ≤ φ ∧ φ < π) : 
  ∀ (x y : ℝ), 
  (parametric_line t φ = (x, y) → (x * sin φ - y * cos φ + 2 * cos φ = 0)) ∧
  ((exists ρ, polar_curve θ = ρ) → (x^2 = 8 * y)) ∧
  (let t1 := (8 * sin φ / cos^2 φ), t2 := (-16 / cos^2 φ) in 
  |t1 - t2| = 8) :=
begin
  sorry
end

end line_curve_properties_l544_544963


namespace solution_set_l544_544787

def equation_holds (x : ℝ) : Prop :=
  real.sqrt (2*x + 2 - 2*real.sqrt (2*x + 1)) + 
  real.sqrt (2*x + 10 - 6*real.sqrt (2*x + 1)) = 2

theorem solution_set :
  {x : ℝ | equation_holds x} = set.Icc 0 4 :=
sorry

end solution_set_l544_544787


namespace permutation_sum_eq_744_l544_544600

open Nat

theorem permutation_sum_eq_744 (n : ℕ) (h1 : n ≠ 0) (h2 : n + 3 ≤ 2 * n) (h3 : n + 1 ≤ 4) :
  choose (2 * n) (n + 3) + choose 4 (n + 1) = 744 := by
  sorry

end permutation_sum_eq_744_l544_544600


namespace volume_of_region_eq_one_l544_544950

noncomputable def region (x y z : ℝ) : Prop :=
  (2 * Real.sqrt (2*y) <= x) ∧ (x <= 17 * Real.sqrt (2*y)) ∧
  (0 <= y) ∧ (y <= 1/2) ∧
  (0 <= z) ∧ (z <= 1/2 - y)

theorem volume_of_region_eq_one :
  ∫⁻ x in (∫⁻ y in (∫⁻ z in (indicator (region x y z) 1)), dx) = 1 :=
sorry

end volume_of_region_eq_one_l544_544950


namespace sum_of_coefficients_l544_544590

theorem sum_of_coefficients : 
  (∑ i in (Finset.range 13), (nat.choose 12 i) * (2^(12 - i) * (-1)^i)) = 1 :=
sorry

end sum_of_coefficients_l544_544590


namespace value_of_shares_eq_l544_544851

variable (d_r r_i P : ℝ) (V N : ℝ)

-- Conditions
def dividend_rate := (d_r = 0.125)
def return_on_investment := (r_i = 0.25)
def share_purchase_price := (P = 25)
def value_of_shares := V
def number_of_shares := N

theorem value_of_shares_eq 
  (h1: dividend_rate)
  (h2: return_on_investment)
  (h3: share_purchase_price) :
  V = 25 * 0.5 * N := 
sorry

end value_of_shares_eq_l544_544851


namespace vasya_gift_ways_l544_544738

theorem vasya_gift_ways :
  let cars := 7
  let constructor_sets := 5
  (cars * constructor_sets) + (Nat.choose cars 2) + (Nat.choose constructor_sets 2) = 66 :=
by
  let cars := 7
  let constructor_sets := 5
  sorry

end vasya_gift_ways_l544_544738


namespace circle_diameter_point_x_l544_544684

-- Define the endpoints of the circle's diameter
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (21, 0)

-- Define the point (x, 12)
def P_x : ℝ → ℝ × ℝ := λ x, (x, 12)

-- Mathematical statement to prove: for a point on the circle with diameter endpoints (-3,0) and (21,0), 
--   and y-coordinate 12, the x-coordinate must be 9.
theorem circle_diameter_point_x (x : ℝ) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in  -- center of the circle
  let r := (Mathlib.Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) / 2 in  -- radius of the circle
  let D := (9, 0) in  -- center is (9, 0)
  let R := 12 in  -- radius is 12
  (P_x x).fst = 9 := by
  -- stating that the point (x,12) lies on the circle with the center (9,0) and radius 12
  sorry

end circle_diameter_point_x_l544_544684


namespace part1_part2_l544_544539

section
  variable {x a : ℝ}

  def f (x a : ℝ) := |x - a| + 3 * x

  theorem part1 (h : a = 1) : 
    (∀ x, f x a ≥ 3 * x + 2 ↔ (x ≥ 3 ∨ x ≤ -1)) :=
    sorry

  theorem part2 : 
    (∀ x, (f x a) ≤ 0 ↔ (x ≤ -1)) → a = 2 :=
    sorry
end

end part1_part2_l544_544539


namespace thirtieth_gradually_increasing_number_l544_544466

/-- A "gradually increasing number" is a positive integer where each digit is greater than the digit to its left. --/
def gradually_increasing_number (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → (n.digit i < n.digit j)

/-- The 30th "gradually increasing number" when all four-digit "gradually increasing numbers" are arranged in ascending order is 1359. --/
theorem thirtieth_gradually_increasing_number :
  ∃ n : ℕ, gradually_increasing_number n ∧ n.digits.length = 4 ∧ (order_val n 30 = 1359) :=
sorry

end thirtieth_gradually_increasing_number_l544_544466


namespace division_problem_l544_544789

theorem division_problem (D d q r : ℕ) 
  (h1 : D + d + q + r = 205)
  (h2 : q = d) :
  D = 174 ∧ d = 13 :=
by {
  sorry
}

end division_problem_l544_544789


namespace max_handshakes_without_cycles_l544_544464

open BigOperators

theorem max_handshakes_without_cycles :
  ∀ n : ℕ, n = 20 → ∑ i in Finset.range (n - 1), i = 190 :=
by intros;
   sorry

end max_handshakes_without_cycles_l544_544464


namespace cos_val_of_acute_angle_proof_l544_544642

noncomputable def cos_val_of_acute_angle (α : ℝ) : Prop :=
  (cos α = 2 * sqrt 2 / 3)

theorem cos_val_of_acute_angle_proof (α : ℝ) (h_parallel_vectors : (1/3) / cos α = tan α) (h_acute : 0 < α ∧ α < π / 2) :
  cos_val_of_acute_angle α :=
sorry

end cos_val_of_acute_angle_proof_l544_544642


namespace evens_between_8000_and_9000_l544_544267

theorem evens_between_8000_and_9000 :
  let even_integers := {n ∈ finset.range 8001 9000 | (n % 2 = 0) ∧ (finset.card (finset.to_set (digit_set n)) = 4)} in
  finset.card even_integers = 560 := 
sorry

def digit_set (n : ℕ) : finset ℕ :=
  finset.insert (n % 10) (finset.insert ((n / 10) % 10) (finset.insert ((n / 100) % 10) (finset.singleton (n / 1000 % 10))))

end evens_between_8000_and_9000_l544_544267


namespace man_speed_l544_544841

theorem man_speed (L T V_t V_m : ℝ) (hL : L = 400) (hT : T = 35.99712023038157) (hVt : V_t = 46 * 1000 / 3600) (hE : L = (V_t - V_m) * T) : V_m = 1.666666666666684 :=
by
  sorry

end man_speed_l544_544841


namespace area_of_rectangle_ABCD_l544_544515

-- Definitions based on conditions
def side_length_smaller_square := 2
def area_smaller_square := side_length_smaller_square ^ 2
def side_length_larger_square := 3 * side_length_smaller_square
def area_larger_square := side_length_larger_square ^ 2
def area_rect_ABCD := 2 * area_smaller_square + area_larger_square

-- Lean theorem statement for the proof problem
theorem area_of_rectangle_ABCD : area_rect_ABCD = 44 := by
  sorry

end area_of_rectangle_ABCD_l544_544515


namespace linear_combination_of_matrices_l544_544945

variable (A B : Matrix (Fin 3) (Fin 3) ℤ) 

def matrixA : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![2, -4, 0],
    ![-1, 5, 1],
    ![0, 3, -7]
  ]

def matrixB : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![4, -1, -2],
    ![0, -3, 5],
    ![2, 0, -4]
  ]

theorem linear_combination_of_matrices :
  3 • matrixA - 2 • matrixB = 
  ![
    ![-2, -10, 4],
    ![-3, 21, -7],
    ![-4, 9, -13]
  ] :=
sorry

end linear_combination_of_matrices_l544_544945


namespace coefficient_x4_sq_l544_544710

def p (x : ℝ) : ℝ := x^5 - 2 * x^2 + 1

theorem coefficient_x4_sq (x : ℝ) : coeff (x^4) ((p x)^2) = 4 := by
  sorry

end coefficient_x4_sq_l544_544710


namespace car_speed_l544_544472

variable (fuel_efficiency : ℝ) (fuel_decrease_gallons : ℝ) (time_hours : ℝ) 
          (gallons_to_liters : ℝ) (kilometers_to_miles : ℝ)
          (car_speed_mph : ℝ)

-- Conditions given in the problem
def fuelEfficiency : ℝ := 40 -- km per liter
def fuelDecreaseGallons : ℝ := 3.9 -- gallons
def timeHours : ℝ := 5.7 -- hours
def gallonsToLiters : ℝ := 3.8 -- liters per gallon
def kilometersToMiles : ℝ := 1.6 -- km per mile

theorem car_speed (fuel_efficiency fuelDecreaseGallons timeHours gallonsToLiters kilometersToMiles : ℝ) : 
  let fuelDecreaseLiters := fuelDecreaseGallons * gallonsToLiters
  let distanceKm := fuelDecreaseLiters * fuel_efficiency
  let distanceMiles := distanceKm / kilometersToMiles
  let averageSpeed := distanceMiles / timeHours
  averageSpeed = 65 := sorry

end car_speed_l544_544472


namespace nearest_integer_to_expression_correct_l544_544054

open Real

noncomputable def nearest_integer_to_expression : ℝ :=
  let a := (3 + (sqrt 5))^6
  let b := (3 - (sqrt 5))^6
  let s := a + b
  floor s

theorem nearest_integer_to_expression_correct :
  nearest_integer_to_expression = 74608 :=
begin
  sorry
end

end nearest_integer_to_expression_correct_l544_544054


namespace no_values_of_t_satisfy_equation_l544_544770

theorem no_values_of_t_satisfy_equation (t : ℂ) (h : abs (t.re) ≤ 5) : sqrt (25 - t^2) + 5 ≠ 0 := 
by {
  sorry
}

end no_values_of_t_satisfy_equation_l544_544770


namespace outermost_tiles_l544_544528

theorem outermost_tiles (perimeter : ℕ) (h : perimeter = 52) : ∃ n : ℕ, 4 * n - 4 = perimeter ∧ n = 14 :=
by
  use 14
  rw [h]
  simp
  sorry

end outermost_tiles_l544_544528


namespace arrangement_problem_l544_544793

def numWaysToArrangeParticipants : ℕ := 90

theorem arrangement_problem :
  ∃ (boys : ℕ) (girls : ℕ) (select_boys : ℕ → ℕ) (select_girls : ℕ → ℕ)
    (arrange : ℕ × ℕ × ℕ → ℕ),
  boys = 3 ∧ girls = 5 ∧
  select_boys boys = 3 ∧ select_girls girls = 5 ∧ 
  arrange (select_boys boys, select_girls girls, 2) = numWaysToArrangeParticipants :=
by
  sorry

end arrangement_problem_l544_544793


namespace tangent_line_through_origin_eq_ex_l544_544169

theorem tangent_line_through_origin_eq_ex :
  ∃ (k : ℝ), (∀ x : ℝ, y = e^x) ∧ (∃ x₀ : ℝ, y - e^x₀ = e^x₀ * (x - x₀)) ∧ 
  (y = k * x) :=
sorry

end tangent_line_through_origin_eq_ex_l544_544169


namespace average_total_goals_l544_544552

theorem average_total_goals (carter_avg shelby_avg judah_avg total_avg : ℕ) 
    (h1: carter_avg = 4) 
    (h2: shelby_avg = carter_avg / 2)
    (h3: judah_avg = 2 * shelby_avg - 3) 
    (h4: total_avg = carter_avg + shelby_avg + judah_avg) :
  total_avg = 7 :=
by
  sorry

end average_total_goals_l544_544552


namespace Q_div_P_l544_544390

theorem Q_div_P (P Q : ℤ) (h : ∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 →
  P / (x + 7) + Q / (x * (x - 6)) = (x^2 - x + 15) / (x^3 + x^2 - 42 * x)) :
  Q / P = 7 :=
sorry

end Q_div_P_l544_544390


namespace problem1_problem2_l544_544183

-- Define the set A_n
def A (n : ℕ) : Set ℕ := {a | a > 0 ∧ a ≤ n ∧ n ∣ (a^n + 1)}

-- Define a function to check if A_n is non-empty
def A_non_empty (n : ℕ) : Prop := ∃ a, a ∈ A n

-- Define a function to count the size of A_n
def A_size (n : ℕ) : ℕ := {a | a > 0 ∧ a ≤ n ∧ n ∣ (a^n + 1)}.to_finset.card

-- Problem 1: Find all positive integers n such that A_n is non-empty
theorem problem1 (n : ℕ) (h : A_non_empty n) : 
  n = 2 ∨ n % 2 = 1 ∨ (n = 2 * ∏ (i : ℕ) in (finset.range n).filter (λ i, nat.prime i ∧ i % 4 = 1), i) := 
sorry

-- Problem 2: Find all positive integers n such that A_n is non-empty and |A_n| is odd
theorem problem2 (n : ℕ) (h : A_non_empty n) (h_odd : A_size n % 2 = 1) : 
  n = 2 * ∏ (i : ℕ) in (finset.range n).filter (λ i, nat.prime i ∧ i % 4 = 1), i := 
sorry

end problem1_problem2_l544_544183


namespace total_pixels_correct_l544_544865

-- Define the monitor's dimensions and pixel density as given conditions
def width_inches : ℕ := 21
def height_inches : ℕ := 12
def pixels_per_inch : ℕ := 100

-- Define the width and height in pixels based on the given conditions
def width_pixels : ℕ := width_inches * pixels_per_inch
def height_pixels : ℕ := height_inches * pixels_per_inch

-- State the objective: proving the total number of pixels on the monitor
theorem total_pixels_correct : width_pixels * height_pixels = 2520000 := by
  sorry

end total_pixels_correct_l544_544865


namespace max_m_eq_half_l544_544997

noncomputable def f (x m : ℝ) : ℝ := (1/2) * x^2 + m * x + m * Real.log x

theorem max_m_eq_half :
  ∃ m : ℝ, (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 ≤ 2) → (1 ≤ x2 ∧ x2 ≤ 2) → 
  x1 < x2 → |f x1 m - f x2 m| < x2^2 - x1^2)) ∧ m = 1/2 :=
sorry

end max_m_eq_half_l544_544997


namespace natural_pairs_l544_544572

theorem natural_pairs (x y : ℕ) : 2^(2 * x + 1) + 2^x + 1 = y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by sorry

end natural_pairs_l544_544572


namespace second_set_parallel_lines_l544_544656

theorem second_set_parallel_lines (n : ℕ) (h1 : 5 * (n - 1) = 420) : n = 85 :=
by sorry

end second_set_parallel_lines_l544_544656


namespace charlie_and_elle_crayons_l544_544344

theorem charlie_and_elle_crayons :
  (∃ (Lizzie Bobbie Billie Charlie Dave Elle : ℕ),
  Billie = 18 ∧
  Bobbie = 3 * Billie ∧
  Lizzie = Bobbie / 2 ∧
  Charlie = 2 * Lizzie ∧
  Dave = 4 * Billie ∧
  Elle = (Bobbie + Dave) / 2 ∧
  Charlie + Elle = 117) :=
sorry

end charlie_and_elle_crayons_l544_544344


namespace avg_goals_per_game_l544_544553

def carter_goals_per_game := 4
def shelby_goals_per_game := carter_goals_per_game / 2
def judah_goals_per_game := (2 * shelby_goals_per_game) - 3
def average_total_goals_team := carter_goals_per_game + shelby_goals_per_game + judah_goals_per_game

theorem avg_goals_per_game : average_total_goals_team = 7 :=
by
  -- Proof would go here
  sorry

end avg_goals_per_game_l544_544553


namespace total_flour_l544_544372

def bought_rye_flour := 5
def bought_bread_flour := 10
def bought_chickpea_flour := 3
def had_pantry_flour := 2

theorem total_flour : bought_rye_flour + bought_bread_flour + bought_chickpea_flour + had_pantry_flour = 20 :=
by
  sorry

end total_flour_l544_544372


namespace probability_distance_ge_l544_544804

theorem probability_distance_ge (β α : ℝ) (h : α < β) : 
  let prob := (β - α)^2 / (β^2) in
  prob = ((β - α)^2 / β^2) :=
sorry

end probability_distance_ge_l544_544804


namespace find_AB_AC_eq_l544_544990

-- Definitions
def Point := (ℝ × ℝ)

def Vertex_B : Point := (5, 1)

def altitude_AB_eq : ℝ → ℝ → Prop := λ x y, x - 2 * y - 5 = 0

def line_eq (k : ℝ) (x₀ y₀ : ℝ) : ℝ → ℝ → Prop := λ x y, y - y₀ = k * (x - x₀)

-- Example input to check specific conditions
def angle_bisector_eq : ℝ → ℝ → Prop := λ x y, x + 2 * y - 13 = 0
def median_BC_eq : ℝ → ℝ → Prop := λ x y, 2 * x - y - 5 = 0

-- Theorem stating the equivalent proof problem in Lean 4
theorem find_AB_AC_eq :
  ∃ (k₁ k₂ : ℝ) (A : Point) (C : Point),
    (∀ (x y : ℝ), altitude_AB_eq x y → line_eq k₁ 5 1 x y) ∧
    ((∀ (x y : ℝ), angle_bisector_eq x y → line_eq k₂ A.1 A.2 x y) ∨ 
     (∀ (x y : ℝ), median_BC_eq x y → line_eq k₂ A.1 A.2 x y)) → False :=
begin
  sorry
end

end find_AB_AC_eq_l544_544990


namespace min_distinct_complex_roots_l544_544703

-- Define the conditions for the polynomials P, Q, and R
variables {P Q R : ℂ[X]}
axiom degree_P : degree P = 4
axiom degree_Q : degree Q = 5
axiom degree_R : degree R = 10
axiom const_term_P : coeff P 0 = 3
axiom const_term_Q : coeff Q 0 = 5
axiom const_term_R : coeff R 0 = 8

theorem min_distinct_complex_roots : 
  ∃ (z : ℂ), P * Q = R → ∃ (n : ℕ), n = 1 :=
by {
  sorry
}

end min_distinct_complex_roots_l544_544703


namespace problem_223_values_of_c_l544_544186

theorem problem_223_values_of_c:
  ∃ s : set ℕ, (∀ c ∈ s, c ≥ 0 ∧ c ≤ 1000 ∧ (∃ x : ℝ, 7 * floor x + 2 * (ceil x) = c)) ∧ (set.card s = 223) :=
by
  sorry

end problem_223_values_of_c_l544_544186


namespace possible_values_b2_l544_544873

theorem possible_values_b2 : 
  ∃ (b_2 : ℕ), 0 ≤ b_2 ∧ b_2 < 1001 ∧ 
  b_2 % 2 = 1 ∧ 
  Nat.gcd 1001 b_2 = 1 ∧
  (Nat.count_filter (λ x : ℕ, x < 1001 ∧ x % 2 = 1 ∧ Nat.gcd 1001 x = 1) 1001) = 220 := sorry

end possible_values_b2_l544_544873


namespace find_a_and_b_l544_544985

theorem find_a_and_b (a b : ℤ) (h : ∀ x : ℝ, x ≤ 0 → (a*x + 2)*(x^2 + 2*b) ≤ 0) : a = 1 ∧ b = -2 := 
by 
  -- Proof steps would go here, but they are omitted as per instructions.
  sorry

end find_a_and_b_l544_544985


namespace problem_statement_l544_544633

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * x

def a := f (Real.log (3 / 2))
def b := f (Real.logb 2 (1 / 3))
def c := f (Real.pow 2 0.3)

theorem problem_statement : b > a ∧ a > c :=
by 
  have h1 : f' (x : ℝ) = Real.cos x - 2, from by sorry -- provided monotonicity from derivative
  have monotonic_decreasing : ∀ x y : ℝ, x < y → f y < f x, from by sorry -- monotonicity proof
  have x1 : Real.logb 2 (1 / 3) < Real.log (3 / 2), from by sorry, -- provided
  have x2 : Real.log (3 / 2) < Real.pow 2 0.3, from by sorry,     -- provided
  exact ⟨monotonic_decreasing _ _ x1, monotonic_decreasing _ _ x2⟩

end problem_statement_l544_544633


namespace neg_prop_p_l544_544979

def prop_p (x : ℝ) : Prop := x ≥ 0 → Real.log (x^2 + 1) ≥ 0

theorem neg_prop_p : (¬ (∀ x ≥ 0, Real.log (x^2 + 1) ≥ 0)) ↔ (∃ x ≥ 0, Real.log (x^2 + 1) < 0) := by
  sorry

end neg_prop_p_l544_544979


namespace math_problem_l544_544255

theorem math_problem
  (A : ℝ) (ω : ℝ) (φ : ℝ) (k : ℤ)
  (hA : A = 4)
  (hω : ω = 1/7)
  (hφ : φ = 5 * Real.pi / 14) :
  (∀ x : ℝ, 0 < x ∧ x < 9 * Real.pi →
    (4 * Real.sin ((1/7) * x + 5 * Real.pi / 14) = 4 * Real.sin ((1/7) * x + φ))) ∧
  (∀ x : ℝ, (14 * k * Real.pi - 6 * Real.pi ≤ x ∧ x ≤ 14 * k * Real.pi + Real.pi) →
    (4 * Real.sin ((1/7) * x + φ) is strictly monotonic increasing)) ∧
  (∃ m : ℝ, 3/2 < m ∧ m ≤ 4 ∧
    4 * Real.sin ((1/7) * Real.sqrt (m + 1) + φ) > 4 * Real.sin ((1/7) * Real.sqrt (-m + 4) + φ)) :=
by sorry

end math_problem_l544_544255


namespace connected_graph_has_at_least_n_minus_1_edges_and_exactly_n_minus_1_if_tree_l544_544376

-- Definitions relevant to the problem
variable {V : Type} -- Type of vertices
variable {G : SimpleGraph V} -- SimpleGraph is a definition in Mathlib

-- Number of vertices
variable (n : ℕ)
-- Number of edges
variable (edges : Finset (Sym2 V))
-- Condition: Graph is connected
variable [Fintype V] [DecidableRel G.Adj]
variable (connected : G.IsConnected)

-- The statement to be proven
theorem connected_graph_has_at_least_n_minus_1_edges_and_exactly_n_minus_1_if_tree :
  G.IsConnected → Fintype.card V = n → (∃ (T : SimpleGraph V), T.IsTree ∧ T.edgeFinset = edges ∧ T.card_edges = n - 1) ↔ 
    (T.card_edges = n - 1 ∧ T.IsTree) := by
  sorry

end connected_graph_has_at_least_n_minus_1_edges_and_exactly_n_minus_1_if_tree_l544_544376


namespace rudy_total_running_time_l544_544368

-- Definitions for the conditions
def time_first_part := 5 * 10
def time_second_part := 4 * 9.5

-- Statement to be proved
theorem rudy_total_running_time : time_first_part + time_second_part = 88 := by
  sorry

end rudy_total_running_time_l544_544368


namespace correct_equation_l544_544818

theorem correct_equation :
  (2 * Real.sqrt 2) / (Real.sqrt 2) = 2 :=
by
  -- Proof goes here
  sorry

end correct_equation_l544_544818


namespace A_3_2_eq_29_l544_544914

-- Define the recursive function A(m, n).
def A : Nat → Nat → Nat
| 0, n => n + 1
| (m + 1), 0 => A m 1
| (m + 1), (n + 1) => A m (A (m + 1) n)

-- Prove that A(3, 2) = 29
theorem A_3_2_eq_29 : A 3 2 = 29 := by 
  sorry

end A_3_2_eq_29_l544_544914


namespace retailer_overhead_expenses_l544_544516

theorem retailer_overhead_expenses (purchase_price selling_price profit_percent : ℝ) (overhead_expenses : ℝ) 
  (h1 : purchase_price = 225) 
  (h2 : selling_price = 300) 
  (h3 : profit_percent = 25) 
  (h4 : selling_price = (purchase_price + overhead_expenses) * (1 + profit_percent / 100)) : 
  overhead_expenses = 15 := 
by
  sorry

end retailer_overhead_expenses_l544_544516


namespace modulus_of_power_of_complex_l544_544580

theorem modulus_of_power_of_complex (z : ℂ) (n : ℕ) : 
  |(2 + 1*I)^8| = 625 :=
by
  sorry

end modulus_of_power_of_complex_l544_544580


namespace draw_shorter_segments_l544_544641

theorem draw_shorter_segments (A B : Point) (ruler_length : ℝ) (AB_length : ℝ) (hAB : A ≠ B ∧ 0 < ruler_length ∧ ruler_length < AB_length)
  : ∃ (C : Point), is_between C A B ∧ distance A C < AB_length ∧ distance C B < AB_length :=
sorry

end draw_shorter_segments_l544_544641


namespace nearest_integer_to_expression_correct_l544_544057

open Real

noncomputable def nearest_integer_to_expression : ℝ :=
  let a := (3 + (sqrt 5))^6
  let b := (3 - (sqrt 5))^6
  let s := a + b
  floor s

theorem nearest_integer_to_expression_correct :
  nearest_integer_to_expression = 74608 :=
begin
  sorry
end

end nearest_integer_to_expression_correct_l544_544057


namespace nearest_integer_to_expression_correct_l544_544055

open Real

noncomputable def nearest_integer_to_expression : ℝ :=
  let a := (3 + (sqrt 5))^6
  let b := (3 - (sqrt 5))^6
  let s := a + b
  floor s

theorem nearest_integer_to_expression_correct :
  nearest_integer_to_expression = 74608 :=
begin
  sorry
end

end nearest_integer_to_expression_correct_l544_544055


namespace blood_cells_in_second_sample_l544_544526

theorem blood_cells_in_second_sample (total_cells : ℕ) (sample1_cells : ℕ) (sample2_cells : ℕ) 
  (h1 : total_cells = 7341) (h2 : sample1_cells = 4221) : sample2_cells = total_cells - sample1_cells := 
by 
  rw [h1, h2]
  exact (3120 : ℕ)

end blood_cells_in_second_sample_l544_544526


namespace part_a_part_b_part_c_part_d_l544_544451

-- Part (a)
theorem part_a : ∃ (a b c : ℕ), (a ^ 2 + 31 ^ 2 = b ^ 2 + (31 + 1) ^ 2) ∧ (b ^ 2 + (31 + 1) ^ 2 = c ^ 2 + (31 + 2) ^ 2) := 
by
  sorry

-- Part (b)
theorem part_b : ∀ n, ∃ k (a b c : ℕ), (a ^ 2 + k ^ 2 = b ^ 2 + (k + 1) ^ 2) ∧ (b ^ 2 + (k + 1) ^ 2 = c ^ 2 + (k + 2) ^ 2) :=
by
  sorry

-- Part (c)
theorem part_c : ∀ (k a b c : ℕ), (a ^ 2 + k ^ 2 = b ^ 2 + (k + 1) ^ 2) ∧ (b ^ 2 + (k + 1) ^ 2 = c ^ 2 + (k + 2) ^ 2) → 144 ∣ (a * b * c) :=
by
  sorry

-- Part (d)
theorem part_d : ∀ (k a b c d : ℕ), ¬ ((a ^ 2 + k ^ 2 = b ^ 2 + (k + 1) ^ 2) ∧ (b ^ 2 + (k + 1) ^ 2 = c ^ 2 + (k + 2) ^ 2) ∧ (c ^ 2 + (k + 2) ^ 2 = d ^ 2 + (k + 3) ^ 2)) :=
by
  sorry

end part_a_part_b_part_c_part_d_l544_544451


namespace moon_arrangement_l544_544917

theorem moon_arrangement : 
  let M_count := 1
  let O_count := 2
  let N_count := 1
  let total_letters := 4
  ∑ perm : List Nat, perm.permutations.length = total_letters! // (M_count! * O_count! * N_count!) :=
  12
 :=
begin
  -- Definitions from the condition
  have M_count := 1,
  have O_count := 2,
  have N_count := 1,
  have total_letters := 4,
  
  -- Applying formulas for permutation counts
  let num_unique_arrangements := total_letters.factorial / (M_count.factorial * O_count.factorial * N_count.factorial),
  show num_unique_arrangements = 12 from sorry
end

end moon_arrangement_l544_544917


namespace magician_assistant_trick_l544_544487

/-- A coin can be either heads or tails. -/
inductive Coin
| heads : Coin
| tails : Coin

/-- Given a cyclic arrangement of 11 coins, there exists at least one pair of adjacent coins with the same face. -/
theorem magician_assistant_trick (coins : Fin 11 → Coin) : 
  ∃ i : Fin 11, coins i = coins (i + 1) := 
by
  sorry

end magician_assistant_trick_l544_544487


namespace ounces_of_water_l544_544395

variable (x: ℝ)

theorem ounces_of_water (h_original_ounces: 12)
                        (h_original_concentration: 0.6)
                        (h_desired_concentration: 0.4) :
  let Alcohol = 12 * 0.6 in
  let Total_Volume = 12 + x in
  Alcohol = 0.4 * Total_Volume → x = 6 :=
by
  intros h
  have h1 : Alcohol = 7.2 := by norm_num
  have h2 : 0.4 * Total_Volume = 0.4 * (12 + x) := by norm_num
  rw [h1] at h
  rw [h2] at h
  sorry -- completion of the proof step is not required

end ounces_of_water_l544_544395


namespace no_equalities_l544_544000

noncomputable def x (t : ℝ) := t ^ (1 / (t - 1))
noncomputable def y (t : ℝ) := t ^ (t / (t-1))
noncomputable def z (t : ℝ) := t ^ (2 / (t-1))

theorem no_equalities (t : ℝ) (ht : 0 < t) (ht1 : t ≠ 1) :
  (z t ^ x t ≠ x t ^ z t) ∧ 
  (z t ^ x t ≠ y t ^ z t) ∧ 
  (x t ^ z t ≠ y t ^ z t) ∧ 
  (x t ^ z t ≠ z t ^ x t) ∧ 
  (x t ^ z t ≠ y t ^ z t) ∧ 
  (z t ^ x t ≠ x t ^ y t) ∧ 
  (x t ^ z t ≠ y t ^ x t): 
  sorry

end no_equalities_l544_544000


namespace int_solution_exists_l544_544750

theorem int_solution_exists (a b : ℤ) (h : even (a * b)) : 
  ∃ (x y : ℤ), a^2 + b^2 + x^2 = y^2 :=
by {
  sorry
}

end int_solution_exists_l544_544750


namespace bounds_x_i_l544_544204

-- Define the fundamental elements in the problem
variables {x : ℕ → ℝ} {n : ℕ} {a : ℝ}

-- Declare the conditions as hypotheses
axiom sum_eq_a (h1 : ∑ i in finset.range n, x i = a)
axiom sum_sq_eq (h2 : ∑ i in finset.range n, (x i)^2 = a^2 / (n - 1))
axiom pos_a (h3 : a > 0)
axiom n_nat (h4 : n ∈ ℕ)
axiom n_ge_two (h5 : n ≥ 2)

-- State the goal
theorem bounds_x_i : ∀ i ≤ n, 0 ≤ x i ∧ x i ≤ 2 * a / n :=
by
  -- proof will be constructed here
  sorry

end bounds_x_i_l544_544204


namespace solve_for_k_l544_544957

noncomputable def base_k_representation (k : ℕ) (r : ℚ) := 
  ∑' n : ℕ, r.num * k^(-n*r.denom) / (1 - k^(-n*r.denom))

theorem solve_for_k (k : ℕ) (h_positive : k > 0) 
  (h_representation : base_k_representation k (2 * k^1 + 3 * k^2) = 7 / 51) : 
  k = 16 :=
by
  sorry

end solve_for_k_l544_544957


namespace magician_can_identify_matching_coin_l544_544505

-- Define the types and conditions
variable (coins : Fin 11 → Bool) -- Each coin is either heads (true) or tails (false).
variable (uncovered_index : Fin 11) -- The index of the initially uncovered coin.

-- Define the property to be proved: the magician can point to an adjacent coin with the same face as the uncovered one.
theorem magician_can_identify_matching_coin (h : ∃ i : Fin 11, coins i = coins (i + 1) % 11) :
  ∃ j : Fin 11, (j ≠ uncovered_index) ∧ coins j = coins uncovered_index :=
  sorry

end magician_can_identify_matching_coin_l544_544505


namespace composite_number_l544_544744

theorem composite_number (n : ℕ) (h : n ≥ 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = (10 ^ n + 1) * (10 ^ (n + 1) - 1) / 9 :=
by sorry

end composite_number_l544_544744


namespace geometric_sequence_and_binomial_expansion_l544_544623

theorem geometric_sequence_and_binomial_expansion :
  (∃ (a : ℕ → ℚ), (a 5 = nat.choose 6 2 * (-(1/3) : ℚ) ^ 2) 
  → (a 3 * a 7 = (a 5) ^ 2)) := 
by
  sorry

end geometric_sequence_and_binomial_expansion_l544_544623


namespace average_price_of_remaining_packets_l544_544727

variables (initial_avg_price : ℕ) (initial_packets : ℕ) (returned_packets : ℕ) (returned_avg_price : ℕ)

def total_initial_cost := initial_avg_price * initial_packets
def total_returned_cost := returned_avg_price * returned_packets
def remaining_packets := initial_packets - returned_packets
def total_remaining_cost := total_initial_cost initial_avg_price initial_packets - total_returned_cost returned_avg_price returned_packets
def remaining_avg_price := total_remaining_cost initial_avg_price initial_packets returned_avg_price returned_packets / remaining_packets initial_packets returned_packets

theorem average_price_of_remaining_packets :
  initial_avg_price = 20 →
  initial_packets = 5 →
  returned_packets = 2 →
  returned_avg_price = 32 →
  remaining_avg_price initial_avg_price initial_packets returned_avg_price returned_packets = 12
:=
by
  intros h1 h2 h3 h4
  rw [remaining_avg_price, total_remaining_cost, total_initial_cost, total_returned_cost]
  norm_num [h1, h2, h3, h4]
  sorry

end average_price_of_remaining_packets_l544_544727


namespace length_of_AB_l544_544029

theorem length_of_AB (r : ℝ) (h_right : true) (angle : true) (h_radius : r = 4) : 
  let a := 4 * real.sqrt 2 in let b := 4 * real.sqrt 2 in
  sqrt((a ^ 2) + (b ^ 2)) = 8 :=
by
  let a := 4 * real.sqrt 2
  let b := 4 * real.sqrt 2
  calc
  sqrt((a ^ 2) + (b ^ 2)) = sqrt(2 * (4 ^ 2) * 2) : by sorry  -- Calculation of the hypotenuse
    ... = 8 : by sorry      -- Simplification to get the final answer

end length_of_AB_l544_544029


namespace frozen_yogurt_price_l544_544748

variable (F G S : ℝ) -- Define the variables F, G, S as real numbers

-- Define the conditions given in the problem
variable (h1 : 5 * F + 2 * G + 5 * S = 55)
variable (h2 : S = 5)
variable (h3 : G = 1 / 2 * F)

-- State the proof goal
theorem frozen_yogurt_price : F = 5 :=
by
  sorry

end frozen_yogurt_price_l544_544748


namespace Alex_age_is_12_l544_544884

-- Defining the scenario
def ages : List Nat := [2, 4, 6, 8, 10, 12]

-- Condition: Two friends whose ages sum to 18 went to concert
def went_to_concert (a b : Nat) := a + b = 18

-- Condition: Three friends younger than 8 went to park
def went_to_park (a b c : Nat) := a < 8 ∧ b < 8 ∧ c < 8

-- Condition: Alex and the youngest friend stayed home
def stayed_home (alex youngest : Nat) (remaining: List Nat) := 
  youngest = 2 ∧ alex ∈ (12::remaining) 

-- The theorem to prove
theorem Alex_age_is_12 : 
  (∃ a b, went_to_concert a b) →
  (∃ a b c, went_to_park a b c) →
  (∃ alex youngest, stayed_home alex youngest ages.filter(λ x, x ≠ 2)) →
  true := 
by
  intro _ _ _
  trivial

end Alex_age_is_12_l544_544884


namespace parabola_diff_zeros_l544_544779

-- Parabola definition and conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def vertex (a b c : ℝ) := (1, -3)
def point_on_curve (a b c : ℝ) := (3, 9)

-- The zeros of the quadratic function
def zeros (a b c : ℝ) : (ℝ × ℝ) :=
  let disc := b^2 - 4 * a * c
  ((-b + Real.sqrt disc) / (2 * a), (-b - Real.sqrt disc) / (2 * a))

theorem parabola_diff_zeros (a b c : ℝ) (h_vertex : vertex a b c = (1, -3))
  (h_point : point_on_curve a b c = (3, 9))
  (zeros (a b c) = (m, n)) (h_mn : m > n) : m - n = 2 :=
sorry

end parabola_diff_zeros_l544_544779


namespace selling_price_for_loss_l544_544398

noncomputable def cp : ℝ := 640
def sp1 : ℝ := 768
def sp2 : ℝ := 448
def sp_profitable_sale : ℝ := 832

theorem selling_price_for_loss :
  sp_profitable_sale - cp = cp - sp2 :=
by
  sorry

end selling_price_for_loss_l544_544398


namespace part1_part2_l544_544933

-- Assuming x is a real number
variable (x : ℝ) (a : ℝ)

theorem part1 : ∀ a : ℝ, (∀ x : ℝ, ¬ (| x - 4 | + | 3 - x | < a)) → a ≤ 1 :=
by sorry

theorem part2 : ∀ a : ℝ, (∃ x : ℝ, | x - 4 | + | 3 - x | < a) → a > 1 :=
by sorry

end part1_part2_l544_544933


namespace find_angle_ACB_l544_544644

-- Define an acute-angled triangle and its orthocenter properties
variables {A B C H : Point}
variable acute_angle_triangle : triangle A B C
variable orthocenter : is_orthocenter H A B C
variable HC_eq_AB : distance H C = distance A B

-- Define angle ACB
noncomputable def angle_ACB : ℝ := angle A C B

-- The statement we want to prove
theorem find_angle_ACB : angle_ACB = 45 :=
  sorry -- proof not required, hence 'sorry'

end find_angle_ACB_l544_544644


namespace sum_of_sequence_b_l544_544637

open Nat

-- Define the sequence a_n
def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * (a n) + 1

-- Define the sequence b_n
def b (n : ℕ) : ℕ :=
  (n * (a n + 1)) / 2

-- Define the sum of the first n terms T_n of the sequence b_n
def T (n : ℕ) : ℕ :=
  ∑ i in Ico 1 (n + 1), b i

-- The statement to prove in Lean 4
theorem sum_of_sequence_b (n : ℕ) : T n = 1 + (n - 1) * 2 ^ n := by
  sorry

end sum_of_sequence_b_l544_544637


namespace max_distance_to_D_l544_544606

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem max_distance_to_D (x y : ℝ) (h : (distance (x, y) (0, 0))^2 + 2 * (distance (x, y) (2, 0))^2 = 3 * (distance (x, y) (2, 2))^2) :
  ∃ P : ℝ × ℝ, P = (x, y) ∧ distance P (0, 2) ≤ real.sqrt 29 := 
sorry

end max_distance_to_D_l544_544606


namespace fifth_equation_l544_544731

theorem fifth_equation
: 1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 :=
by
  sorry

end fifth_equation_l544_544731


namespace count_valid_positive_n_values_l544_544593

def positive_divisors_of_24 : List ℕ := [1, 2, 3, 4, 6, 8, 12, 24]

def valid_positive_n_values : List ℕ := List.filter (λ d, d - 1 > 0) (List.map (λ d, d - 1) positive_divisors_of_24)

theorem count_valid_positive_n_values : List.length valid_positive_n_values = 7 := by
  -- The following proof step is skipped.
  sorry

end count_valid_positive_n_values_l544_544593


namespace enclosed_area_eq_half_pi_l544_544718

noncomputable def g (x : ℝ) : ℝ := 2 - real.sqrt (4 - x^2)

theorem enclosed_area_eq_half_pi :
  (1 : ℝ) * π / 2 = 1.57 :=
by
  sorry

end enclosed_area_eq_half_pi_l544_544718


namespace probability_no_consecutive_heads_l544_544111

-- Definitions based on conditions
def fibonacci_no_consec_heads : ℕ → ℕ
| 0     := 1  -- No coin tossed
| 1     := 2  -- H or T
| n     := fibonacci_no_consec_heads (n-1) + fibonacci_no_consec_heads (n-2)

theorem probability_no_consecutive_heads :
  (fibonacci_no_consec_heads 10 : ℚ) / (2^10) = 9 / 64 := by
sorry

end probability_no_consecutive_heads_l544_544111


namespace projectiles_meet_in_84_minutes_l544_544805

theorem projectiles_meet_in_84_minutes :
  ∀ (d v₁ v₂ : ℝ), d = 1386 → v₁ = 445 → v₂ = 545 → (20 : ℝ) = 20 → 
  ((1386 / (445 + 545) / 60) * 60 * 60 = 84) :=
by
  intros d v₁ v₂ h_d h_v₁ h_v₂ h_wind
  sorry

end projectiles_meet_in_84_minutes_l544_544805


namespace nearest_integer_to_expr_l544_544049

theorem nearest_integer_to_expr : 
  let a := 3 + Real.sqrt 5
  let b := (a)^6
  abs (b - 2744) < 1
:= sorry

end nearest_integer_to_expr_l544_544049


namespace moles_AgOH_and_limiting_reagent_l544_544123

-- Define the given amounts of reactants
def moles_AgNO₃ : ℝ := 0.50
def moles_NaOH : ℝ := 0.35

-- Define the balanced reaction's mole ratio
def mole_ratio := 1

-- Define the limiting reagent function
def limiting_reagent (moles_AgNO₃ moles_NaOH : ℝ) : String :=
  if moles_NaOH < moles_AgNO₃ then "NaOH" else "AgNO₃"

-- Define the moles of AgOH formed based on the limiting reagent
def moles_AgOH_formed (moles_AgNO₃ moles_NaOH : ℝ) : ℝ :=
  if moles_NaOH < moles_AgNO₃ then moles_NaOH else moles_AgNO₃

-- Problem statement to be proven
theorem moles_AgOH_and_limiting_reagent :
  (moles_AgOH_formed moles_AgNO₃ moles_NaOH = 0.35) ∧
  (limiting_reagent moles_AgNO₃ moles_NaOH = "NaOH") :=
by
  sorry

end moles_AgOH_and_limiting_reagent_l544_544123


namespace find_q_l544_544101

noncomputable def value_of_q : ℝ := 0.942808153803174
def percent_of(x: ℝ, y: ℝ) : ℝ := 0.45 * 1200
def denominator : ℝ := 12 + 4 * 3
def rhs_value : ℝ := (percent_of 45 1200) * 80 / denominator

theorem find_q :
  (45 * value_of_q)^2 = rhs_value :=
begin
  sorry,
end

end find_q_l544_544101


namespace transform_properties_correct_l544_544257

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ set.Icc (-3 : ℝ) 0 then
  ((-2 - 1) / (0 - (-3)) * (x - (-3)) + 1)
else if x ∈ set.Icc 0 2 then
  -√(4 - (x - 2) ^ 2) - 2
else if x ∈ set.Icc 2 3 then
  ((2 - 0) / (3 - 2) * (x - 2))
else
  0

noncomputable def g (x : ℝ) : ℝ := 2 * f (2 * x - 1) - 3

def transform_triple : ℝ × ℝ × ℝ := (2, 2, -3)

theorem transform_properties_correct :
  ∃ a b c : ℝ, g x = a * f (b * x - 1) + c ∧ transform_triple = (a, b, c) :=
by
  use 2, 2, -3
  split
  { intros x,
    rw g,
    rw f,
    sorry }
  { refl }

end transform_properties_correct_l544_544257


namespace jane_savings_l544_544854

-- Given conditions
def cost_pair_1 : ℕ := 50
def cost_pair_2 : ℕ := 40

def promotion_A (cost1 cost2 : ℕ) : ℕ :=
  cost1 + cost2 / 2

def promotion_B (cost1 cost2 : ℕ) : ℕ :=
  cost1 + (cost2 - 15)

-- Define the savings calculation
def savings (promoA promoB : ℕ) : ℕ :=
  promoB - promoA

-- Specify the theorem to prove
theorem jane_savings :
  savings (promotion_A cost_pair_1 cost_pair_2) (promotion_B cost_pair_1 cost_pair_2) = 5 := 
by
  sorry

end jane_savings_l544_544854


namespace findBK_length_l544_544693

-- Definitions of given conditions
variables (A B C K L : Type*)
variables [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] [EuclideanSpace ℝ K] [EuclideanSpace ℝ L]

variables (AB BC AC BK BL BK BN KN : ℝ)
variable (angleBAC : Real.angle)
variable (angleB : Real.angle)

-- Notes:
-- - triangle B is the vertex and angles are right angles
-- - AB = BC = 2
-- - The three angles between BA and BK, BK and BL, BL and BC are all equal

-- Hypotheses
axiom h1 : angleB = 90
axiom h2 : AB = 2
axiom h3 : BC = 2
axiom h4 : BK = 2 * (Real.sqrt 3 - 1) -- Given equal angles imply the result

-- Goal
theorem findBK_length :
  BK = 2 * (Real.sqrt 3 - 1) :=
by
  -- The proof will be filled in here
  sorry

end findBK_length_l544_544693


namespace integer_satisfies_inequality_l544_544816

theorem integer_satisfies_inequality (n : ℤ) : 
  (3 : ℚ) / 10 < n / 20 ∧ n / 20 < 2 / 5 → n = 7 :=
sorry

end integer_satisfies_inequality_l544_544816


namespace magician_assistant_strategy_l544_544497

-- Define the possible states of a coin
inductive CoinState | heads | tails

-- Define the circle of coins
def coinCircle := Fin 11 → CoinState

-- The strategy ensures there are adjacent coins with the same state
theorem magician_assistant_strategy (c : coinCircle) :
  ∃ i : Fin 11, c i = c ((i + 1) % 11) := by sorry

end magician_assistant_strategy_l544_544497


namespace folding_positions_l544_544015

theorem folding_positions (positions : Finset ℕ) (h_conditions: positions = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}) : 
  ∃ valid_positions : Finset ℕ, valid_positions = {1, 2, 3, 4, 9, 10, 11, 12} ∧ valid_positions.card = 8 :=
by
  sorry

end folding_positions_l544_544015


namespace average_of_a_and_b_l544_544760

theorem average_of_a_and_b (a b c M : ℝ)
  (h1 : (a + b) / 2 = M)
  (h2 : (b + c) / 2 = 180)
  (h3 : a - c = 200) : 
  M = 280 :=
sorry

end average_of_a_and_b_l544_544760


namespace ratio_ashley_mary_l544_544008

-- Definitions based on conditions
def sum_ages (A M : ℕ) := A + M = 22
def ashley_age (A : ℕ) := A = 8

-- Theorem stating the ratio of Ashley's age to Mary's age
theorem ratio_ashley_mary (A M : ℕ) 
  (h1 : sum_ages A M)
  (h2 : ashley_age A) : 
  (A : ℚ) / (M : ℚ) = 4 / 7 :=
by
  -- Skipping the proof as specified
  sorry

end ratio_ashley_mary_l544_544008


namespace avg_height_trees_l544_544468

-- Assuming heights are defined as h1, h2, ..., h7 with known h2
noncomputable def avgHeight (h1 h2 h3 h4 h5 h6 h7 : ℝ) : ℝ := 
  (h1 + h2 + h3 + h4 + h5 + h6 + h7) / 7

theorem avg_height_trees :
  ∃ (h1 h3 h4 h5 h6 h7 : ℝ), 
    h2 = 15 ∧ 
    (h1 = 2 * h2 ∨ h1 = 3 * h2) ∧
    (h3 = h2 / 3 ∨ h3 = h2 / 2) ∧
    (h4 = 2 * h3 ∨ h4 = 3 * h3 ∨ h4 = h3 / 2 ∨ h4 = h3 / 3) ∧
    (h5 = 2 * h4 ∨ h5 = 3 * h4 ∨ h5 = h4 / 2 ∨ h5 = h4 / 3) ∧
    (h6 = 2 * h5 ∨ h6 = 3 * h5 ∨ h6 = h5 / 2 ∨ h6 = h5 / 3) ∧
    (h7 = 2 * h6 ∨ h7 = 3 * h6 ∨ h7 = h6 / 2 ∨ h7 = h6 / 3) ∧
    avgHeight h1 h2 h3 h4 h5 h6 h7 = 26.4 :=
by
  sorry

end avg_height_trees_l544_544468


namespace equal_focal_distances_of_hyperbolas_l544_544193

open Real

theorem equal_focal_distances_of_hyperbolas (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 4) :
  let C₁ := (λ x y, x^2 / (sin θ)^2 - y^2 / (cos θ)^2 = 1)
      C₂ := (λ x y, y^2 / (cos θ)^2 - x^2 / (sin θ)^2 = 1)
  in
  (let a₁ := sin θ in let b₁ := cos θ in 2 * sqrt (a₁^2 + b₁^2) = 2) ∧
  (let a₂ := cos θ in let b₂ := sin θ in 2 * sqrt (a₂^2 + b₂^2) = 2) :=
by
  sorry

end equal_focal_distances_of_hyperbolas_l544_544193


namespace example_inequality_l544_544337

variable (a b c : ℝ)

theorem example_inequality 
  (h : a^6 + b^6 + c^6 = 3) : a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := 
by
  sorry

end example_inequality_l544_544337


namespace total_students_l544_544408

theorem total_students (third_grade_students fourth_grade_students second_grade_boys second_grade_girls : ℕ)
  (h1 : third_grade_students = 19)
  (h2 : fourth_grade_students = 2 * third_grade_students)
  (h3 : second_grade_boys = 10)
  (h4 : second_grade_girls = 19) :
  third_grade_students + fourth_grade_students + (second_grade_boys + second_grade_girls) = 86 :=
by
  rw [h1, h3, h4, h2]
  norm_num
  sorry

end total_students_l544_544408


namespace volume_of_new_pyramid_l544_544122

theorem volume_of_new_pyramid (l w h : ℝ) (h_vol : (1 / 3) * l * w * h = 80) :
  (1 / 3) * (3 * l) * w * (1.8 * h) = 432 :=
by
  sorry

end volume_of_new_pyramid_l544_544122


namespace roots_cubic_roots_sum_of_squares_l544_544235

variables {R : Type*} [CommRing R] {p q r s t : R}

theorem roots_cubic_roots_sum_of_squares (h1 : r + s + t = p) (h2 : r * s + r * t + s * t = q) :
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
sorry

end roots_cubic_roots_sum_of_squares_l544_544235


namespace sarah_flour_total_l544_544371

def rye_flour : ℕ := 5
def whole_wheat_bread_flour : ℕ := 10
def chickpea_flour : ℕ := 3
def whole_wheat_pastry_flour : ℕ := 2

def total_flour : ℕ := rye_flour + whole_wheat_bread_flour + chickpea_flour + whole_wheat_pastry_flour

theorem sarah_flour_total : total_flour = 20 := by
  sorry

end sarah_flour_total_l544_544371


namespace no_real_solution_cos_sin_l544_544180

theorem no_real_solution_cos_sin (x : ℝ) : ¬ (cos x = 1/2 ∧ sin x = 3/4) :=
by
  sorry

end no_real_solution_cos_sin_l544_544180


namespace prob_selected_first_eq_third_l544_544419

noncomputable def total_students_first := 800
noncomputable def total_students_second := 600
noncomputable def total_students_third := 500
noncomputable def selected_students_third := 25
noncomputable def prob_selected_third := selected_students_third / total_students_third

theorem prob_selected_first_eq_third :
  (selected_students_third / total_students_third = 1 / 20) →
  (prob_selected_third = 1 / 20) :=
by
  intros h
  sorry

end prob_selected_first_eq_third_l544_544419


namespace sum_n_k_l544_544768

theorem sum_n_k (n k : ℕ) (h1 : ∃ c1 c2 c3 : ℕ, c1:c2:c3 = 1:3:6) 
  (h2 : c1 = binomial n k) 
  (h3 : c2 = binomial n (k + 1)) 
  (h4 : c3 = binomial n (k + 2)) : 
  n + k = 9 := 
sorry

end sum_n_k_l544_544768


namespace elysse_bags_per_trip_l544_544173

-- Definitions from the problem conditions
def total_bags : ℕ := 30
def total_trips : ℕ := 5
def bags_per_trip : ℕ := total_bags / total_trips

def carries_same_amount (elysse_bags brother_bags : ℕ) : Prop := elysse_bags = brother_bags

-- Statement to prove
theorem elysse_bags_per_trip :
  ∀ (elysse_bags brother_bags : ℕ), 
  bags_per_trip = elysse_bags + brother_bags → 
  carries_same_amount elysse_bags brother_bags → 
  elysse_bags = 3 := 
by 
  intros elysse_bags brother_bags h1 h2
  sorry

end elysse_bags_per_trip_l544_544173


namespace moon_arrangements_l544_544923

theorem moon_arrangements :
  (∃ (MOON : Finset (List Char)), 
    {w : List Char |
      w ∈ MOON ∧ w = ['M', 'O', 'O', 'N']}.card = 12) :=
sorry

end moon_arrangements_l544_544923


namespace algebraic_expression_value_l544_544198

theorem algebraic_expression_value (b a c : ℝ) (h₁ : b < a) (h₂ : a < 0) (h₃ : 0 < c) :
  |b| - |b - a| + |c - a| - |a + b| = b + c - a :=
by
  sorry

end algebraic_expression_value_l544_544198


namespace sequence_term_2023_l544_544971

theorem sequence_term_2023 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h : ∀ n, 2 * S n = a n * (a n + 1)) : 
  a 2023 = 2023 :=
sorry

end sequence_term_2023_l544_544971


namespace angle_sum_eq_180_l544_544110

variables 
  (A B C D O W X Y Z : Type)
  [inhabited A] [inhabited B] [inhabited C] [inhabited D]
  [inhabited O] [inhabited W] [inhabited X] [inhabited Y] [inhabited Z]
  -- Geometry related definitions e.g. points, angles, etc
  (circle : CircumscribedCircleInQuadrilateral ABCD O W X Y Z)

-- Conditions
def inscribed_circle : Prop := 
  circle.is_inscribed_around ABCD 

-- Target statement to prove
theorem angle_sum_eq_180 (h : inscribed_circle) : 
  ∠ A O B + ∠ C O D = 180 :=
sorry

end angle_sum_eq_180_l544_544110


namespace min_right_triangle_side_l544_544785

theorem min_right_triangle_side (s : ℕ) : 
  (7^2 + 24^2 = s^2 ∧ 7 + 24 > s ∧ 24 + s > 7 ∧ 7 + s > 24) → s = 25 :=
by
  intro h
  sorry

end min_right_triangle_side_l544_544785


namespace magician_assistant_strategy_l544_544496

-- Define the possible states of a coin
inductive CoinState | heads | tails

-- Define the circle of coins
def coinCircle := Fin 11 → CoinState

-- The strategy ensures there are adjacent coins with the same state
theorem magician_assistant_strategy (c : coinCircle) :
  ∃ i : Fin 11, c i = c ((i + 1) % 11) := by sorry

end magician_assistant_strategy_l544_544496


namespace magicians_strategy_l544_544046

theorem magicians_strategy (n : ℕ) (hn : n > 0) :
  ∃ (strategy : (fin n) × (fin n) → (fin n) × (fin n) → (fin n) × (fin n) → bool),
    (∀ (board : (fin n) × (fin n) → bool)
        (C D : (fin n) × (fin n)),
      strategy board C D = (board D) ≠ (board (strategy board C D))) :=
sorry

end magicians_strategy_l544_544046


namespace min_sum_of_products_l544_544602

theorem min_sum_of_products (a : Fin 95 → ℤ) 
    (h : ∀ i, a i = 1 ∨ a i = -1) :
    ∃ N : ℤ, N = 13 ∧ (∑ i j in Finset.filter (λ ⟨i, j⟩, i < j) (Finset.product (Finset.univ : Finset (Fin 95)) (Finset.univ : Finset (Fin 95))), a i * a j) = 13 := 
by 
  sorry

end min_sum_of_products_l544_544602


namespace increasing_function_range_a_l544_544617

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (3 - a) * x - a else Real.logBase a x

theorem increasing_function_range_a : ∀ (x : ℝ), increasing_on (Ioo (-∞) (∞)) (f a x) →
  (3 / 2 ≤ a ∧ a < 3) :=
sorry

end increasing_function_range_a_l544_544617


namespace range_of_floor_f_l544_544018

def floor_function (x : ℝ) : ℤ := Int.floor x

def f (x : ℝ) : ℝ := x / (x ^ 2 + 3 * x + 4) + 8 / 9

theorem range_of_floor_f :
  ∀ y, y ∈ {floor_function (f x) | x : ℝ} ↔ y = -1 ∨ y = 0 ∨ y = 1 :=
sorry

end range_of_floor_f_l544_544018


namespace volume_tetrahedron_lt_one_eighth_l544_544392

noncomputable def volume_of_tetrahedron (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 6) * |((B.1 - A.1) * ((C.2 - A.2) * (D.3 - A.3) - (C.3 - A.3) * (D.2 - A.2)) -
             (B.2 - A.2) * ((C.1 - A.1) * (D.3 - A.3) - (C.3 - A.3) * (D.1 - A.1)) +
             (B.3 - A.3) * ((C.1 - A.1) * (D.2 - A.2) - (C.2 - A.2) * (D.1 - A.1)))|

theorem volume_tetrahedron_lt_one_eighth
  (A B C D : ℝ × ℝ × ℝ)
  (hAB : dist A B < 1)
  (hAC : dist A C < 1)
  (hAD : dist A D < 1)
  (hBC : dist B C < 1)
  (hBD : dist B D < 1) :
  volume_of_tetrahedron A B C D < 1 / 8 :=
by
  sorry

end volume_tetrahedron_lt_one_eighth_l544_544392


namespace girl_boy_lineup_probability_l544_544862

theorem girl_boy_lineup_probability :
  let total_configurations := Nat.choose 20 9
  let valid_case1 := Nat.choose 14 9
  let valid_subcases := 6 * Nat.choose 13 8
  let valid_configurations := valid_case1 + valid_subcases
  (valid_configurations : ℚ) / total_configurations = 0.058 :=
by
  let total_configurations := Nat.choose 20 9
  let valid_case1 := Nat.choose 14 9
  let valid_subcases := 6 * Nat.choose 13 8
  let valid_configurations := valid_case1 + valid_subcases
  have h : (valid_configurations : ℚ) / total_configurations = 0.058 := sorry
  exact h

end girl_boy_lineup_probability_l544_544862


namespace integer_part_of_result_is_40_l544_544775

noncomputable def numerator : ℝ := 0.1 + 1.2 + 2.3 + 3.4 + 4.5 + 5.6 + 6.7 + 7.8 + 8.9
noncomputable def denominator : ℝ := 0.01 + 0.03 + 0.05 + 0.07 + 0.09 + 0.11 + 0.13 + 0.15 + 0.17 + 0.19
noncomputable def result : ℝ := numerator / denominator

theorem integer_part_of_result_is_40 : ⌊result⌋ = 40 := 
by
  -- proof goes here
  sorry

end integer_part_of_result_is_40_l544_544775


namespace Elmo_books_count_l544_544172

-- Define the number of books each person has
def Stu_books : ℕ := 4
def Laura_books : ℕ := 2 * Stu_books
def Elmo_books : ℕ := 3 * Laura_books

-- The theorem we need to prove
theorem Elmo_books_count : Elmo_books = 24 := by
  -- this part is skipped since no proof is required
  sorry

end Elmo_books_count_l544_544172


namespace groups_of_nine_l544_544402

theorem groups_of_nine (boys girls students_per_group total_students groups : ℕ)
  (h_boys : boys = 31)
  (h_girls : girls = 32)
  (h_students_per_group : students_per_group = 9)
  (h_total_students : total_students = boys + girls)
  (h_groups : groups = total_students / students_per_group) :
  groups = 7 := 
by {
  rw [h_boys, h_girls, h_students_per_group] at h_total_students,
  rw [h_total_students],
  norm_num,
  sorry
}

end groups_of_nine_l544_544402


namespace find_x_such_that_sqrt_quad_eq_nine_l544_544949

theorem find_x_such_that_sqrt_quad_eq_nine (x : ℝ) :
  (sqrt (4 - 2 * x + x^2) = 9) ↔ (x = 1 + sqrt 78 ∨ x = 1 - sqrt 78) := 
by 
  sorry

end find_x_such_that_sqrt_quad_eq_nine_l544_544949


namespace mountain_height_l544_544296

theorem mountain_height
  (base_temp summit_temp : ℝ)
  (temp_change_rate : ℝ)
  (temperature_decrease : ∀ x : ℝ, summit_temp = base_temp - temp_change_rate * x) :
  ∃ x : ℝ, (x * 100 = 1700) :=
by
  let base_temp := 26
  let summit_temp := 14.1
  let temp_change_rate := 0.7
  have h : summit_temp = base_temp - temp_change_rate * 17 := by
    -- Summit temperature formula verification (skipped proof)
    sorry
  use 17
  rw mul_eq_mul_left_iff
  right
  norm_num

end mountain_height_l544_544296


namespace eq1_l544_544022

theorem eq1 (h1 : ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x0 : ℝ, x0^3 - x0^2 + 1 > 0)) 
            (h2 : ¬ ((∀ a b c : ℝ, b = sqrt (a * c) → (a*b*c) ≠ 0) ∧ (a*b*c = 0 → b ≠ sqrt (a * c))))
            (h3 : ∀ m : ℝ, ¬ ((m = -1 ↔ (∀ l1 l2 : ℝ, l1 * l2 = 0 → l1 ≠ l2)) ↔
                                (m = -1 ∧ (mx + (2*m-1)*y + 1 = 0 ∧ 3x + m*y + 2 = 0))))
            : (1 = 1) := sorry

end eq1_l544_544022


namespace overall_gain_percentage_l544_544116

def cost_of_A : ℝ := 100
def selling_price_of_A : ℝ := 125
def cost_of_B : ℝ := 200
def selling_price_of_B : ℝ := 250
def cost_of_C : ℝ := 150
def selling_price_of_C : ℝ := 180

theorem overall_gain_percentage :
  ((selling_price_of_A + selling_price_of_B + selling_price_of_C) - (cost_of_A + cost_of_B + cost_of_C)) / (cost_of_A + cost_of_B + cost_of_C) * 100 = 23.33 := 
by
  sorry

end overall_gain_percentage_l544_544116


namespace onions_on_scale_l544_544794

theorem onions_on_scale (N : ℕ) (W_total : ℕ) (W_removed : ℕ) (avg_remaining : ℕ) (avg_removed : ℕ) :
  W_total = 7680 →
  W_removed = 5 * 206 →
  avg_remaining = 190 →
  avg_removed = 206 →
  N = 40 :=
by
  sorry

end onions_on_scale_l544_544794


namespace cyclic_quadrilateral_relations_l544_544604

theorem cyclic_quadrilateral_relations
  (A B C D : ℝ)
  (hABCD : A + C = π ∧ B + D = π) :
  (sin A = sin C ∧ cos B + cos D = 0) ∧ 
  ¬(sin A + sin C = 0) ∧ ¬(cos B = cos D) :=
by sorry

end cyclic_quadrilateral_relations_l544_544604


namespace markup_is_correct_l544_544027

-- The mathematical interpretation of the given conditions
def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.05
def net_profit : ℝ := 12

-- Define the overhead cost calculation
def overhead_cost : ℝ := overhead_percentage * purchase_price

-- Define the total cost calculation
def total_cost : ℝ := purchase_price + overhead_cost

-- Define the selling price calculation
def selling_price : ℝ := total_cost + net_profit

-- Define the markup calculation
def markup : ℝ := selling_price - purchase_price

-- The statement we want to prove
theorem markup_is_correct : markup = 14.40 :=
by
  -- We will eventually prove this, but for now we use sorry as a placeholder
  sorry

end markup_is_correct_l544_544027


namespace proof_problem_l544_544209

def f (x : ℝ) : ℝ := Real.tan x + Real.sin x + 1

theorem proof_problem (b : ℝ) (h : f b = 2) : f (-b) = 0 := 
by 
  sorry

end proof_problem_l544_544209


namespace d_value_l544_544730

noncomputable def eq_area_half_unit_square (c d : ℝ) : Prop :=
  let line1 := λ x, (4 / (4 - c)) * (x - c)
  let line2 := λ x, -d / 4 * x + d
  let intersection_x := if (4 / (4 - c)) * 4 - (4 * 4 / (4 - c)) = -d / 4 * 4 + d
                        then 4 else 0 -- x = 4 when vertical line x=4 intersects
  let intersection_y := line2 intersection_x
  (1 / 2) * (4 - 0) * intersection_y = (1 / 2)

theorem d_value (c : ℝ) : eq_area_half_unit_square c 2 := by
  sorry

end d_value_l544_544730


namespace planted_fraction_l544_544177

theorem planted_fraction (a b c : ℕ) (x h : ℝ) 
  (h_right_triangle : a = 5 ∧ b = 12)
  (h_hypotenuse : c = 13)
  (h_square_dist : x = 3) : 
  (h * ((a * b) - (x^2))) / (a * b / 2) = (7 : ℝ) / 10 :=
by
  sorry

end planted_fraction_l544_544177


namespace games_bought_at_garage_sale_l544_544722

-- Definitions based on conditions
def games_from_friend : ℕ := 2
def defective_games : ℕ := 2
def good_games : ℕ := 2

-- Prove the number of games bought at the garage sale equals 2
theorem games_bought_at_garage_sale (G : ℕ) 
  (h : games_from_friend + G - defective_games = good_games) : G = 2 :=
by 
  -- use the given information and work out the proof here
  sorry

end games_bought_at_garage_sale_l544_544722


namespace incorrect_statements_count_is_3_l544_544820

def is_incorrect_statement (st : Nat) : Prop :=
  match st with
  | 1 => ¬(∀ x : ℕ, abs x ≥ abs 0) -- This statement asserts that no non-zero absolute value is smaller than 0
  | 2 => ¬ (∀ a : ℕ, -(3 * a - 2) = 2 - 3 * a)
  | 3 => ¬ (∀ R : ℕ, coefficient (5 * pi * R^2) = 5 * pi)
  | 4 => ¬ (∀ q : ℚ, is_rational q → (∃ z : ℤ, q = z) ∨ (∃ n d : ℤ, d ≠ 0 ∧ q = n / d))
  | 5 => ¬ (∀ x : ℕ, monomial_degree (3^4 * x^3) = 7)
  | _ => false

theorem incorrect_statements_count_is_3 :
  (count (λ st, is_incorrect_statement st) [1, 2, 3, 4, 5]) = 3 :=
sorry

end incorrect_statements_count_is_3_l544_544820


namespace percentage_reduction_l544_544870

theorem percentage_reduction (y x z p q : ℝ) (hy : y ≠ 0) (h1 : x = y - 10) (h2 : z = y - 20) :
  p = 1000 / y ∧ q = 2000 / y := by
  sorry

end percentage_reduction_l544_544870


namespace qinjiushao_value_at_neg1_l544_544634

noncomputable def f (x : ℤ) := x^5 + 47 * x^4 - 37 * x^2 + 1

theorem qinjiushao_value_at_neg1 : 
  let V0 := 1 in
  let V1 := V0 * (-1) + 47 in
  let V2 := V1 * (-1) in
  let V3 := V2 * (-1) - 37 in
  V3 = 9 :=
by
  let V0 := 1
  let V1 := V0 * (-1) + 47
  let V2 := V1 * (-1)
  let V3 := V2 * (-1) - 37
  show V3 = 9 from sorry

end qinjiushao_value_at_neg1_l544_544634


namespace heights_ratio_l544_544885

variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

noncomputable def ratio_heights_tetrahedron 
  (h1 : ∀ A' B' C' D' : MetricSpace, edge_touches_sphere A' B' C' D')
  (h2 : ∀ A' B' C' D' : MetricSpace, seg_midpoints_eq A' B' C' D')
  (h3 : ∠ A B C = 100) : Real := 
  2 * (Real.sqrt 3) * (Real.tan (50 * Real.pi / 180))

theorem heights_ratio 
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  (h_edges_touches_sphere : edge_touches_sphere A B C D) 
  (h_midpoints_eq : seg_midpoints_eq A B C D) 
  (angle_ABC : ∠ A B C = 100) : 
  ratio_heights_tetrahedron A B C D h_edges_touches_sphere h_midpoints_eq angle_ABC = 
  2 * (Real.sqrt 3) * (Real.tan (50 * Real.pi / 180)) := 
  sorry

end heights_ratio_l544_544885


namespace product_of_factors_l544_544781

theorem product_of_factors :
  (∏ n in Finset.range 9, (1 - 1 / (n + 2))) = 1 / 10 :=
by
  sorry

end product_of_factors_l544_544781


namespace number_of_rhombuses_l544_544534

-- Define the conditions as assumptions
def side_length_large_triangle : ℕ := 10
def total_small_triangles : ℕ := 100
def small_triangle_side_length : ℕ := 1

-- Define the proposition to prove the number of rhombuses
theorem number_of_rhombuses : 
  (side_length_large_triangle = 10) →
  (total_small_triangles = 100) →
  (small_triangle_side_length = 1) →
  ∃ n, n = 84 :=
by
  intros
  use 84
  sorry -- proof to be completed

end number_of_rhombuses_l544_544534


namespace sin_function_props_l544_544635

theorem sin_function_props (A ω m : ℝ) 
  (hA : A > 0) (hω : ω > 0) 
  (h_max : ∀ x, A * sin (ω * x + π / 6) + m ≤ 3) 
  (h_min : ∀ x, A * sin (ω * x + π / 6) + m ≥ -5)
  (h_sym : ∀ x, A * sin (ω * (x + π / (2 * ω)) + π / 6) + m = A * sin (ω * x + π / 6) + m) :
  A = 4 ∧ ω = 2 ∧ m = -1 :=
by
  sorry

end sin_function_props_l544_544635


namespace expansion_coeff_sum_l544_544767

theorem expansion_coeff_sum (n k : ℕ) (hn: (x + 2)^n = ∑ (i : ℕ) in range(n + 1), (choose n i) * x^(n - i) * 2^i) 
  (h1: choose n k * 1 = choose n (k + 1) * 3) 
  (h2: choose n (k + 1) * 3 = choose n (k + 2) * 6) : 
  n + k = 6 :=
by
  sorry

end expansion_coeff_sum_l544_544767


namespace closest_point_on_line_l544_544179

theorem closest_point_on_line : 
  ∃ t : ℝ, 
    let p := (3 - t, 1 + 4 * t, 2 - 2 * t)
    in p = (59 / 21, 25 / 21, 34 / 21) ∧
       ∀ q : ℝ × ℝ × ℝ, 
         q = (p.1, p.2, p.3) →
         (∑ i in Finset.range 3, (([1, 2, 3].nth_le i sorry) - ([p.1, p.2, p.3].nth_le i sorry))^2) ≤
         (∑ i in Finset.range 3, (([1, 2, 3].nth_le i sorry) - ([q.1, q.2, q.3].nth_le i sorry))^2) :=
sorry

end closest_point_on_line_l544_544179


namespace product_mod_remainder_l544_544075

open Int

theorem product_mod_remainder :
  (1897 * 2048) % 600 = 256 :=
by
  have h1 : 1897 % 600 = 97 := by norm_num
  have h2 : 2048 % 600 = -352 := by norm_num
  calc
    (1897 * 2048) % 600
        = (97 * -352) % 600 : by rw [Int.mul_mod, h1, h2]
    ... = (-34144) % 600  : by norm_num
    ... = 256             : by norm_num

end product_mod_remainder_l544_544075


namespace pentagon_side_lengths_l544_544356

noncomputable theory

def is_arithmetic_progression (a b c d : ℕ) (k : ℕ) : Prop :=
  b = a + k ∧ c = b + k ∧ d = c + k

def valid_side_lengths (a b c d e : ℕ) : Prop :=
  a = 30 ∧
  (is_arithmetic_progression b c d e 2) ∧
  (b ≤ 7) ∧
  (b + c + d + e > a)

theorem pentagon_side_lengths (a b c d e : ℕ) :
  valid_side_lengths a b c d e →
  (b = 5 ∧ c = 7 ∧ d = 9 ∧ e = 11 ∨
   b = 6 ∧ c = 8 ∧ d = 10 ∧ e = 12 ∨
   b = 7 ∧ c = 9 ∧ d = 11 ∧ e = 13) :=
by
  intros h
  cases h with ha hrest
  cases hrest with hap hbound
  cases hap with hbc hcd
  cases hcd with hde heq
  cases heq with hx sum_cond
  have h₁ : b = 5 ∨ b = 6 ∨ b = 7,
  sorry,
  cases h₁,
  {left, sorry},
  cases h₁,
  {right, left, sorry},
  {right, right, sorry}

end pentagon_side_lengths_l544_544356


namespace find_k_l544_544978

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ ∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def tangent_circle {r : ℝ} (k m : ℝ) : Prop :=
  m^2 = 1 + k^2

noncomputable def dot_product_condition (OA OB : ℝ) : Prop :=
  OA * OB = -3 / 2

theorem find_k (a b : ℝ) (h_ecc : a > b ∧ b > 0)
  (h_eccentricity : 1 / 2 = 1 / 2) (h_major_axis_len : 2 * a = 4) 
  (h_tangent : ∀ k m, tangent_circle k m)
  (h_condition : ∀ OA OB, dot_product_condition OA OB)
  : k = sqrt 2 / 2 ∨ k = - sqrt 2 / 2 :=
sorry

end find_k_l544_544978


namespace curve_symmetric_origin_l544_544405

theorem curve_symmetric_origin (x y : ℝ) (h : 3*x^2 - 8*x*y + 2*y^2 = 0) :
  3*(-x)^2 - 8*(-x)*(-y) + 2*(-y)^2 = 3*x^2 - 8*x*y + 2*y^2 :=
sorry

end curve_symmetric_origin_l544_544405


namespace inequality_solution_l544_544958

theorem inequality_solution (x : ℝ) (h : 1 + 2 * x ≥ 0) :
  (frac (4 * x^2) ((1 - sqrt (1 + 2 * x))^2) < 2 * x + 9) ↔ (-1/2 ≤ x ∧ x < 45/8) := 
sorry

end inequality_solution_l544_544958


namespace max_value_f_l544_544586

noncomputable def f (x : ℝ) : ℝ := log 2 * log 5 - log (2 * x) * log (5 * x)

theorem max_value_f : (∃ x : ℝ, x > 0 ∧ f x = 1/4) ∧ ∀ y > 0, f y ≤ 1/4 :=
by
  sorry

end max_value_f_l544_544586


namespace identity1_vandermonde_identity_l544_544746

theorem identity1 (n m r : ℕ) (h : n + m ≥ r) : 
  ∑ k in Finset.range (r + 1), Nat.choose n k * Nat.choose m (r - k) = Nat.choose (n + m) r := sorry

theorem vandermonde_identity (n : ℕ) : 
  ∑ k in Finset.range (n + 1), (Nat.choose n k) ^ 2 = Nat.choose (2 * n) n := sorry

end identity1_vandermonde_identity_l544_544746


namespace population_8000_in_2090_l544_544174

noncomputable def population_at_year (initial_population : ℕ) (quadrupling_period : ℕ) (current_year : ℕ) : ℕ :=
  initial_population * 4^(current_year - 2000) / quadrupling_period

theorem population_8000_in_2090 :
  (∀ (initial_population : ℕ) (quadrupling_period : ℕ) (target_population : ℕ), initial_population = 250 → quadrupling_period = 30 → target_population = 8000 → 
    ∃ (target_year : ℕ), target_year = 2090 ∧ population_at_year initial_population quadrupling_period target_year = 16000) :=
by
  intros initial_population quadrupling_period target_population initial_eq quad_period_eq target_eq
  use 2090
  rw [initial_eq, quad_period_eq, target_eq]
  have pop_calc := population_at_year 250 30 2090
  norm_num at pop_calc
  exact pop_calc
  sorry

end population_8000_in_2090_l544_544174


namespace solve_exponential_eq_l544_544753

theorem solve_exponential_eq :
  ∀ (x : ℝ), 3^(27^x) = 27^(3^x) → x = 1 :=
by
  intro x h
  sorry

end solve_exponential_eq_l544_544753


namespace no_positive_n_lcm_gcd_120_eq_gcd_360_l544_544077

theorem no_positive_n_lcm_gcd_120_eq_gcd_360 (n : ℕ) (h_pos : n > 0) :
  (nat.lcm n 120 = nat.gcd n 120 + 360) → n = 0 :=
by
  sorry

end no_positive_n_lcm_gcd_120_eq_gcd_360_l544_544077


namespace correct_computation_gives_l544_544652

variable (x : ℝ)

theorem correct_computation_gives :
  ((3 * x - 12) / 6 = 60) → ((x / 3) + 12 = 160 / 3) :=
by
  sorry

end correct_computation_gives_l544_544652


namespace count_3_digit_multiples_30_not_40_l544_544650

def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def count_multiples (a b : ℕ) (pred : ℕ → Prop) : ℕ :=
  (finset.range (b - a + 1)).filter pred).card

theorem count_3_digit_multiples_30_not_40 :
  (count_multiples 100 999 (λ n, is_multiple_of n 30 ∧ ¬is_multiple_of n 40)) = 22 :=
by 
  sorry

end count_3_digit_multiples_30_not_40_l544_544650


namespace nearest_integer_to_power_six_l544_544067

noncomputable def nearest_integer (x : ℝ) : ℤ := 
if x - real.floor x < real.ceil x - x then real.floor x else real.ceil x

theorem nearest_integer_to_power_six : 
  let x := 3 + real.sqrt 5 in
  let y := 3 - real.sqrt 5 in
  nearest_integer (x^6) = 2654 :=
by
  sorry

end nearest_integer_to_power_six_l544_544067


namespace apples_fit_into_basket_l544_544314

theorem apples_fit_into_basket :
  ∀ (J_full_capacity J_current_space : ℕ),
    J_full_capacity = 12 → J_current_space = 4 →
    let J_current_apples := J_full_capacity - J_current_space in
    let J_full_capacity := 12 in
    let J_full_capacity_double := J_full_capacity * 2 in
    let fit_times := J_full_capacity_double / J_current_apples in
    fit_times = 3 :=
by
  intros J_full_capacity J_current_space h1 h2,
  let J_current_apples := J_full_capacity - J_current_space,
  let J_full_capacity := 12,
  let J_full_capacity_double := J_full_capacity * 2,
  let fit_times := J_full_capacity_double / J_current_apples,
  have : fit_times = 3 := by sorry,
  exact this

end apples_fit_into_basket_l544_544314


namespace part1_part2_part3_l544_544253

-- Definitions according to conditions
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 - 3 * x
def f_prime (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x - 3

theorem part1 (a b : ℝ) (h1 : f_prime 1 = 0) (h2 : f_prime (-1) = 0) : 
  a = 1 ∧ b = 0 := by
  sorry

def specific_f (x : ℝ) : ℝ := x^3 - 3 * x

theorem part2 (x1 x2 : ℝ) (h : x1 ∈ Icc (-1 : ℝ) 1 ∧ x2 ∈ Icc (-1 : ℝ) 1) : 
  |specific_f x1 - specific_f x2| ≤ 4 := by
  sorry

def tangent_condition (x₀ m : ℝ) : ℝ := 2 * x₀^3 - 3 * x₀^2 + m + 3

theorem part3 (m : ℝ) (h : ∃ h : ℝ, tangent_condition h m = 0 ∧ tangent_condition 1 m = 0) : 
  -3 < m ∧ m < -2 := by 
  sorry

end part1_part2_part3_l544_544253


namespace cover_black_squares_with_trominos_l544_544951

noncomputable def can_cover_black_squares_with_trominos (n : ℕ) (n_odd : n % 2 = 1) (n_large_enough : n ≥ 7) : Prop :=
  let k := (n - 1) / 2 in (k + 1) ^ 2

theorem cover_black_squares_with_trominos (n : ℕ) (n_odd : n % 2 = 1) (n_large_enough : n ≥ 7) :
  ∃ (trominos_needed : ℕ), trominos_needed = (let k := (n - 1) / 2 in (k + 1) ^ 2) :=
by
  let k := (n - 1) / 2
  use (k + 1) ^ 2
  sorry

end cover_black_squares_with_trominos_l544_544951


namespace unique_identity_function_l544_544166

theorem unique_identity_function (f : ℕ+ → ℕ+) :
  (∀ (x y : ℕ+), 
    let a := x 
    let b := f y 
    let c := f (y + f x - 1)
    a + b > c ∧ a + c > b ∧ b + c > a) →
  (∀ x, f x = x) :=
by
  intro h
  sorry

end unique_identity_function_l544_544166


namespace Emily_age_is_23_l544_544806

variable (UncleBob Daniel Emily Zoe : ℕ)

-- Conditions
axiom h1 : UncleBob = 54
axiom h2 : Daniel = UncleBob / 2
axiom h3 : Emily = Daniel - 4
axiom h4 : Emily = 2 * Zoe / 3

-- Question: Prove that Emily's age is 23
theorem Emily_age_is_23 : Emily = 23 :=
by
  sorry

end Emily_age_is_23_l544_544806


namespace probability_factor_less_than_10_l544_544074

def is_factor (n k : ℕ) : Prop := k ∣ n

def factors (n : ℕ) : List ℕ := List.filter (λ k => is_factor n k) (List.range (n + 1))

def count_less_than (lst : List ℕ) (k : ℕ) : ℕ := List.length (List.filter (λ x => x < k) lst)

def probability {α : Type} [Fintype α] (P : α → Prop) [DecidablePred P] : ℚ :=
  (Fintype.card {x // P x} : ℚ) / Fintype.card α

theorem probability_factor_less_than_10 (n : ℕ) (h : n = 90) :
  probability (λ k, is_factor n k ∧ k < 10) = 1 / 2 := sorry

end probability_factor_less_than_10_l544_544074


namespace angle_of_l2_inclination_l544_544626

-- Define the conditions
def slope_l1 : ℝ := 1
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define the angle calculation based on the slope
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.atan m * 180 / Real.pi

-- State the theorem
theorem angle_of_l2_inclination :
  ∀ (m1 m2 : ℝ),
  m1 = slope_l1 →
  perpendicular m1 m2 →
  angle_of_inclination m2 = 135 :=
by
  intros m1 m2 h_slope h_perp
  sorry

end angle_of_l2_inclination_l544_544626


namespace count_valid_positive_n_values_l544_544592

def positive_divisors_of_24 : List ℕ := [1, 2, 3, 4, 6, 8, 12, 24]

def valid_positive_n_values : List ℕ := List.filter (λ d, d - 1 > 0) (List.map (λ d, d - 1) positive_divisors_of_24)

theorem count_valid_positive_n_values : List.length valid_positive_n_values = 7 := by
  -- The following proof step is skipped.
  sorry

end count_valid_positive_n_values_l544_544592


namespace problem1_problem2_l544_544791

/-- 
  Problem 1: Arrangement of Athletes
  Prove that the number of ways to arrange 7 athletes such that all female athletes are together is 720.
  Conditions: There are 4 male athletes and 3 female athletes.
--/
theorem problem1 (males females : ℕ) (h_males : males = 4) (h_females : females = 3) : 
  let total_ways := factorial (males + 1) * factorial females 
  in total_ways = 720 := by 
  sorry

/-- 
  Problem 2: Allocation to Two Venues
  Prove that the number of ways to allocate 4 male athletes to two venues, with at least one athlete in each venue, is 14.
  Conditions: There are 4 male athletes.
--/
theorem problem2 (males : ℕ) (h_males : males = 4) : 
  let case1_ways := combination males 1
  let case2_ways := combination males 2
  let total_ways := case1_ways + case2_ways
  in total_ways = 14 := by 
  sorry

end problem1_problem2_l544_544791


namespace two_digit_number_problem_l544_544217

theorem two_digit_number_problem (a b : ℕ) :
  let M := 10 * b + a
  let N := 10 * a + b
  2 * M - N = 19 * b - 8 * a := by
  sorry

end two_digit_number_problem_l544_544217


namespace max_sum_a1_to_a19_and_b1_to_b85_l544_544706

noncomputable def a : ℕ → ℕ := sorry -- a is the non-decreasing sequence
noncomputable def b (m : ℕ) : ℕ := sorry -- b_m = min {n : a_n ≥ m}

theorem max_sum_a1_to_a19_and_b1_to_b85 :
  (∀ n, a n > 0) →
  (∀ n m, n < m → a n ≤ a m) →
  a 19 = 85 →
  (∀ m, b m = Nat.find (λ n, a n ≥ m)) →
  (∑ i in finset.range 19, a (i + 1)) + (∑ j in finset.range 85, b (j + 1)) = 1700 :=
by
  intros a_pos a_nondec a_19_eq85 b_def
  sorry

end max_sum_a1_to_a19_and_b1_to_b85_l544_544706


namespace calculation_proof_l544_544548

theorem calculation_proof : 441 + 2 * 21 * 7 + 49 = 784 := by
  sorry

end calculation_proof_l544_544548


namespace find_m_l544_544888

noncomputable def first_series_sum : ℝ := 
  let a1 : ℝ := 18
  let a2 : ℝ := 6
  let r : ℝ := a2 / a1
  a1 / (1 - r)

noncomputable def second_series_sum (m : ℝ) : ℝ := 
  let b1 : ℝ := 18
  let b2 : ℝ := 6 + m
  let s : ℝ := b2 / b1
  b1 / (1 - s)

theorem find_m : 
  (3 : ℝ) * first_series_sum = second_series_sum m → m = 8 := 
by 
  sorry

end find_m_l544_544888


namespace domino_no_chain_l544_544694

open Finset

/-- Define a domino tile as a pair of non-negative integers (i, j) such that 0 ≤ i, j ≤ 6. --/
structure Dominoe where
  first : ℕ
  second : ℕ
  property : first ≤ 6 ∧ second ≤ 6

/-- A list of all 28 dominoes in a standard double-six set. --/
def double_six_set : Finset Dominoe := 
  (Finset.range 7).bUnion (λ i => Finset.range 7 |>.image (λ j => ⟨i, j, ⟨le_refl _, le_refl _⟩⟩))

/-- Checking if there exists a chain of dominoes matching the given condition. --/
def possible_chain (tiles : Finset Dominoe) : Prop :=
  ∃ (chain : List Dominoe), 
    (chain.head? = some ⟨6, 0, some_condition⟩ ∨ chain.head? = some ⟨0, 6, some_condition⟩) ∧
    (chain.last? = some ⟨5, 0, some_condition⟩ ∨ chain.last? = some ⟨0, 5, some_condition⟩) ∧
    (∀ t ∈ chain, t ∈ tiles) ∧
    (∀ i t1 t2, t1 ∈ chain ∧ t2 ∈ chain ∧ chain.nth (i - 1) = some t1 ∧ chain.nth i = some t2 → t1.second = t2.first)

theorem domino_no_chain :
  ¬ possible_chain double_six_set :=
sorry

end domino_no_chain_l544_544694


namespace no_solution_range_has_solution_range_l544_544935

open Real

theorem no_solution_range (a : ℝ) : (∀ x, ¬ (|x - 4| + |3 - x| < a)) ↔ a ≤ 1 := 
sorry

theorem has_solution_range (a : ℝ) : (∃ x, |x - 4| + |3 - x| < a) ↔ 1 < a :=
sorry

end no_solution_range_has_solution_range_l544_544935


namespace matrix_multiplication_correct_l544_544904

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 0, -1; 1, 3, -2; -2, 3, 1]
def B : Matrix (Fin 3) (Fin 3) ℝ := !![3, -3, 1; 2, 1, -4; 5, 0, 0]
def C : Matrix (Fin 3) (Fin 3) ℝ := !![1, -6, 2; -1, 0, -11; 5, 9, -14]

theorem matrix_multiplication_correct : A ⬝ B = C :=
by
  sorry

end matrix_multiplication_correct_l544_544904


namespace find_m_l544_544948

theorem find_m (m : ℕ) (h : ∀ n k : ℕ, n.choose k = n.choose (n - k)) (h_condition : 15.choose m = 15.choose (m - 3)) : m = 9 :=
by
  sorry

end find_m_l544_544948


namespace nearest_integer_3_add_sqrt_5_pow_6_l544_544060

noncomputable def approx (x : ℝ) : ℕ := Real.floor (x + 0.5)

theorem nearest_integer_3_add_sqrt_5_pow_6 : 
  approx ((3 + Real.sqrt 5)^6) = 22608 :=
by
  -- Proof omitted, sorry
  sorry

end nearest_integer_3_add_sqrt_5_pow_6_l544_544060


namespace cant_be_factored_53_l544_544085

-- Define the integers to be checked
def integers_to_check : List ℕ := [6, 27, 53, 39, 77]

-- Prime number check for any integer
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Check if an integer can be factored into two integers greater than 1
def can_be_factored (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Main theorem: 53 cannot be factored into two integers greater than 1
theorem cant_be_factored_53 : ¬ can_be_factored 53 := 
by {
  have H := is_prime 53,
  sorry
}

end cant_be_factored_53_l544_544085


namespace nearest_integer_to_expr_l544_544051

theorem nearest_integer_to_expr : 
  let a := 3 + Real.sqrt 5
  let b := (a)^6
  abs (b - 2744) < 1
:= sorry

end nearest_integer_to_expr_l544_544051


namespace water_level_drop_l544_544853

noncomputable def volume_cylinder (r h : ℝ) := π * r^2 * h
noncomputable def volume_sphere (r : ℝ) := 4 / 3 * π * r^3

theorem water_level_drop :
  ∀ (r r_sphere h : ℝ),
  r = 5 → r_sphere = 2.5 →
  volume_cylinder r h = 2 * volume_sphere r_sphere →
  h = 5 / 3 :=
by
  intros r r_sphere h hr hsphere heq
  sorry

end water_level_drop_l544_544853


namespace price_per_vanilla_cookie_l544_544823

theorem price_per_vanilla_cookie (P : ℝ) (h1 : 220 + 70 * P = 360) : P = 2 := 
by 
  sorry

end price_per_vanilla_cookie_l544_544823


namespace angles_equal_or_supplementary_l544_544687

theorem angles_equal_or_supplementary 
  (O A B C D E : Type)
  (angle_1 angle_2 angle_3 angle_4 angle_5 angle_6 angle_7 angle_8 : ℝ)
  (angle_9 angle_10 : ℝ)
  (h1 : angle_1 = angle_2)
  (h2 : angle_3 = angle_4)
  (h3 : angle_5 = angle_6)
  (h4 : angle_7 = angle_8)
  : angle_10 = angle_9 ∨ angle_10 = 180 - angle_9 := 
begin
  sorry
end

end angles_equal_or_supplementary_l544_544687


namespace angle_GDA_measure_l544_544298

-- Definitions based on the conditions
def right_isosceles_triangle (A B C : Type) :=
  ∃ (angle_A angle_B : ℝ), angle_A = angle_B ∧ angle_A + angle_B = 90 -- conditions for right isosceles triangle

def square (A B C D : Type) :=
  ∀ (angle_ABD angle_BDC : ℝ), angle_ABD = 90 ∧ angle_BDC = 90 -- conditions for square

-- Hypotheses based on the problem
variables (C D E F G A B : Type)
variables (h1 : right_isosceles_triangle C D E)
variables (h2 : square A B C D)
variables (h3 : square D E F G)

-- Prove that the measure of ∠GDA is 135 degrees
theorem angle_GDA_measure : ∠ G D A = 135 :=
sorry

end angle_GDA_measure_l544_544298


namespace convert_521_to_binary_l544_544162

theorem convert_521_to_binary :
  let n := 521
  in let binary_n := "1000001001"
  in ∀ m : ℕ, m = n → Nat.toDigits 2 m = (binary_n.toList.map (λ c => c.toNat - '0'.toNat)) :=
by
  intros
  sorry

end convert_521_to_binary_l544_544162


namespace angle_between_vectors_l544_544261

open Real

noncomputable def vec_ba : ℝ × ℝ := (1/2, sqrt 3 / 2)
noncomputable def vec_bc : ℝ × ℝ := (sqrt 3 / 2, 1/2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

def cos_angle (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

-- The theorem to prove the measure of the angle between the two vectors.
theorem angle_between_vectors :
  ∠ (vec_ba) (vec_bc) = π / 6 :=
by
  have h1 : cos_angle vec_ba vec_bc = sqrt 3 / 2,
  -- Here, add the logical steps to verify the above.
  sorry
-- The expected angle between the two vectors is π / 6.

end angle_between_vectors_l544_544261


namespace minimize_quadratic_l544_544437

def f (x : ℝ) := 3 * x^2 - 18 * x + 7

theorem minimize_quadratic : ∃ x : ℝ, f x = -20 ∧ ∀ y : ℝ, f y ≥ -20 := by
  sorry

end minimize_quadratic_l544_544437


namespace min_value_l544_544946

def f (x: ℝ) : ℝ := (sin x)^8 + (cos x)^8 + 3 / (sin x)^6 + (cos x)^6 + 3

theorem min_value : ∃ x : ℝ, f x = 2 / 3 :=
by
  sorry

end min_value_l544_544946


namespace transform_expression_to_product_l544_544045

open Real

noncomputable def transform_expression (α : ℝ) : ℝ :=
  4.66 * sin (5 * π / 2 + 4 * α) - (sin (5 * π / 2 + 2 * α)) ^ 6 + (cos (7 * π / 2 - 2 * α)) ^ 6

theorem transform_expression_to_product (α : ℝ) :
  transform_expression α = (1 / 8) * sin (4 * α) * sin (8 * α) :=
by
  sorry

end transform_expression_to_product_l544_544045


namespace locus_of_X_is_star_l544_544711

-- Define n-gon and the relevant points
variables {n : ℕ} (h : n ≥ 5)
variables (O A B X Y Z : E) (polygon : Polygon n (EuclideanSpace ℝ (_root_.fin n)))

-- Assume the conditions of the problem
variables
  (h1 : polygon.is_regular O) -- The n-gon is regular
  (h2 : A ∈ polygon.vertices)
  (h3 : B ∈ polygon.vertices)
  (h4 : are_adjacent polygons A B)
  (h5 : triangle XYZ ≃ triangle OAB) -- The triangle XYZ is congruent to OAB
  (h6 : ∀ t, Z = polygon.trace_boundary(parameterized_path t)) -- Z traces the boundary of the polygon
  (h7 : ∀ t, Y = polygon.trace_boundary(parameterized_path t)) -- Y traces the boundary of the polygon
  (h8 : ∀ t, X.inside_polygon(polygon)) -- X remains inside the polygon

-- The theorem stating the locus of X
theorem locus_of_X_is_star (n ≥ 5) :
  locus_of_X(polygon_trace) = star_shape(O n) :=
sorry

end locus_of_X_is_star_l544_544711


namespace no_solution_for_m_eq_7_l544_544190

theorem no_solution_for_m_eq_7 (x m : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : 
  (x-3)/(x-4) = (x-m)/(x-8) → m ≠ 7 :=
by
  intro h
  have : (m-7)x = 4m - 24 := sorry
  sorry

end no_solution_for_m_eq_7_l544_544190


namespace solution_set_of_inequality_l544_544952

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2 * a * x - 3 * a^2 < 0} = {x | 3 * a < x ∧ x < -a} :=
sorry

end solution_set_of_inequality_l544_544952


namespace ten_pow_x_sub_2y_l544_544612

theorem ten_pow_x_sub_2y (x y : ℝ) (h1 : 10^x = 3) (h2 : 10^y = 4) : 10^(x - 2 * y) = 3 / 16 := by
  sorry

end ten_pow_x_sub_2y_l544_544612


namespace inequality_solution_l544_544915

noncomputable def condition (x : ℝ) : Prop :=
  2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))
  ∧ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2

theorem inequality_solution (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * Real.pi) (h₁ : condition x) :
  Real.cos x ≤ Real.sqrt (2:ℝ) / 2 ∧ x ∈ [Real.pi/4, 7 * Real.pi/4] := sorry

end inequality_solution_l544_544915


namespace album_distribution_l544_544875

-- Define the conditions
def albums (photo: ℕ) (stamp: ℕ) := photo = 2 ∧ stamp = 3
def friends (count: ℕ) := count = 5

-- Define the theorem to prove the distribution ways
theorem album_distribution (photo stamp count: ℕ) (h_albums : albums photo stamp) (h_friends : friends count) :
  ∃ ways : ℕ, ways = 10 := 
by {
  -- As per the problem, since the distribution can be done in C(5,2) ways,
  -- we will assert that the ways equals 10
  use 𝔹(nCk 5 2),
  sorry
}

end album_distribution_l544_544875


namespace true_propositions_l544_544630

theorem true_propositions (a : ℝ) (h1 : a > 0 ∧ a ≠ 1) :
  (¬ (∀ x : ℝ, sin x^2 + 3 / sin x^2 ≥ 2 * sqrt 3)) ∧
  (∀ x : ℝ, (x - sqrt 11) / (x - sqrt 10) < (4 - sqrt 11) / (4 - sqrt 10) ∧
             (x - sqrt 11) / (x - sqrt 10) > (3 - sqrt 11) / (3 - sqrt 10)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 1) = -f x) ∧ f 2016 = 0) ∧
  (∀ x : ℝ, monotone (λ x, log a (2 + a ^ x))) :=
by sorry

end true_propositions_l544_544630


namespace TruckCapacities_RentalPlanExists_MinimumRentalCost_l544_544848

-- Problem 1
theorem TruckCapacities (x y : ℕ) (h1: 2 * x + y = 10) (h2: x + 2 * y = 11) :
  x = 3 ∧ y = 4 :=
by
  sorry

-- Problem 2
theorem RentalPlanExists (a b : ℕ) (h: 3 * a + 4 * b = 31) :
  (a = 9 ∧ b = 1) ∨ (a = 5 ∧ b = 4) ∨ (a = 1 ∧ b = 7) :=
by
  sorry

-- Problem 3
theorem MinimumRentalCost (a b : ℕ) (h1: 3 * a + 4 * b = 31) 
  (h2: 100 * a + 120 * b = 940) :
  ∃ a b, a = 1 ∧ b = 7 :=
by
  sorry

end TruckCapacities_RentalPlanExists_MinimumRentalCost_l544_544848


namespace correct_option_l544_544817

theorem correct_option (a b y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hy : y ≠ 0):
  (a^2 / (1 / a) = a^3) ∧
  ¬(1 / (2 * a) + 1 / (3 * a) = 1 / (5 * a)) ∧
  ¬(1 / a - 1 / b = (a - b) / (a * b)) ∧
  ¬((-y)^2 * (-y)^(-1) = y) :=
by {
  sorry  -- Proof steps are not required as per the instructions
}

end correct_option_l544_544817


namespace find_k_l544_544689

theorem find_k (k : ℝ) (h_line : ∀ x y : ℝ, 3 * x + 5 * y + k = 0)
    (h_sum_intercepts : - (k / 3) - (k / 5) = 16) : k = -30 := by
  sorry

end find_k_l544_544689


namespace factor_product_l544_544078

theorem factor_product : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end factor_product_l544_544078


namespace midpoint_lies_on_perpendicular_line_l544_544020

-- Define the given points and angles.
variables {A B C I C' B' C1 B1 : Point}
variables {α β γ : Angle}

-- Assume properties about I being the incenter and other geometric conditions.
variables (hI : Incenter I A B C)
variables (h1 : LineThrough I C' ∧ Perpendicular (LineThrough I C') (Bisector A I))
variables (h2 : LineThrough I B' ∧ Perpendicular (LineThrough I B') (Bisector A I))
variables (h3 : Altitude C' C1 B I)
variables (h4 : Altitude B' B1 C I)

-- Formalize the proof statement
theorem midpoint_lies_on_perpendicular_line
  (midpoint_cond : Midpoint (Segment B1 C1) liesOn (LineThrough I ∧ Perpendicular (LineThrough I) (LineThrough B C))) :
  ∃ T : Point, T = Midpoint (Segment B1 C1) ∧ T liesOn (LineThrough I ∧ Perpendicular (LineThrough I) (LineThrough B C)) := sorry

end midpoint_lies_on_perpendicular_line_l544_544020


namespace correct_answers_are_75_l544_544095

def num_correct_answers (c w : ℕ) : Prop :=
  (c + w = 120) ∧ (3 * c - w = 180)

theorem correct_answers_are_75 : ∃ c w : ℕ, num_correct_answers c w ∧ c = 75 :=
by
  use 75, 45
  split
  · simp [num_correct_answers]
  · sorry

end correct_answers_are_75_l544_544095


namespace largest_integer_of_four_l544_544759

theorem largest_integer_of_four (A B C D : ℤ)
  (h_diff: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_order: A < B ∧ B < C ∧ C < D)
  (h_avg: (A + B + C + D) / 4 = 74)
  (h_A_min: A ≥ 29) : D = 206 :=
by
  sorry

end largest_integer_of_four_l544_544759


namespace find_f1_l544_544616

noncomputable def f (a x : ℝ) : ℝ :=
if x < 2 then 2 * a^x else log (x^2 - 1) / log a

theorem find_f1 (a : ℝ) (h_pos_a : a > 0) (h_neq1_a : a ≠ 1) (h_f2 : f a 2 = 1) :
  f a 1 = 6 :=
by
  sorry

end find_f1_l544_544616


namespace problem_statement_l544_544081

theorem problem_statement : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end problem_statement_l544_544081


namespace smallest_n_for_at_least_64_candies_l544_544855

theorem smallest_n_for_at_least_64_candies :
  ∃ n : ℕ, (n > 0) ∧ (n * (n + 1) / 2 ≥ 64) ∧ (∀ m : ℕ, (m > 0) ∧ (m * (m + 1) / 2 ≥ 64) → n ≤ m) := 
sorry

end smallest_n_for_at_least_64_candies_l544_544855


namespace sin_alpha_beta_eq_l544_544982

theorem sin_alpha_beta_eq 
  (α β : ℝ) 
  (h1 : π / 4 < α) (h2 : α < 3 * π / 4)
  (h3 : 0 < β) (h4 : β < π / 4)
  (h5: Real.sin (α + π / 4) = 3 / 5)
  (h6: Real.cos (π / 4 + β) = 5 / 13) :
  Real.sin (α + β) = 56 / 65 :=
sorry

end sin_alpha_beta_eq_l544_544982


namespace cardinality_S_bound_l544_544134

variables {p k : ℕ} (a : ℕ) (S : Finset ℕ) (n : ℕ)

noncomputable def mod_p (b p : ℕ) : ℕ := (b % p + p) % p

def set_S (p k : ℕ) (a : ℕ) : Finset ℕ :=
  { n | 1 ≤ n ∧ n ≤ p - 1 ∧ 
      ∀ i j, 1 ≤ i → i < j → j ≤ k → mod_p (n * a * i) p < mod_p (n * a * j) p }.to_finset

theorem cardinality_S_bound (p k : ℕ) (a : ℕ) (h : a ≠ 0) :
  (set_S p k a).card < 2 * p / (k + 1) :=
sorry

end cardinality_S_bound_l544_544134


namespace fence_perimeter_262_l544_544415

structure FencePosts where
  total_posts : ℕ
  post_width : ℚ -- 6 inches in feet
  space_between_posts : ℚ -- in feet
  aspect_ratio : ℚ -- 2:1 aspect ratio

theorem fence_perimeter_262 (conditions : FencePosts)
  (h1 : conditions.total_posts = 32)
  (h2 : conditions.post_width = 0.5)
  (h3 : conditions.space_between_posts = 6)
  (h4 : conditions.aspect_ratio = 2) :
  let w := 6 in
  let length := (w - 1) * conditions.space_between_posts + w * conditions.post_width in
  let w := 8 in
  let width := (w - 1) * conditions.space_between_posts + w * conditions.post_width in
  2 * (length + width) = 262 := by
  sorry

end fence_perimeter_262_l544_544415


namespace vector_magnitude_addition_l544_544721

variable {V : Type*} [InnerProductSpace ℝ V]

theorem vector_magnitude_addition 
  (a b : V) 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = 1) 
  (hab : ⟪a, b⟫ = -1) : ∥a + 2 • b∥ = 1 :=
sorry

end vector_magnitude_addition_l544_544721


namespace convex_polyhedron_edge_labeling_exists_l544_544363

structure ConvexPolyhedron (V : Type) :=
(is_convex : Prop)
(vertices : Finset V)
(edges : Finset (V × V))
(is_valid_edge : ∀ e ∈ edges, e.1 ≠ e.2)
(is_vertex_connected : ∀ v ∈ vertices, ∃ e ∈ edges, v = e.1 ∨ v = e.2)

def edge_labeling (V : Type) := (V × V) → ℤ

def valid_labeling {V : Type} (poly : ConvexPolyhedron V) (labeling : edge_labeling V) : Prop :=
  ∀ v ∈ poly.vertices, 
    (∏ (e ∈ (poly.edges.filter (λ e, v = e.1 ∨ v = e.2))), labeling e) = -1

noncomputable def does_valid_labeling_exist (V : Type) [fintype V] (poly : ConvexPolyhedron V) : Prop :=
  (fintype.card V = 100) → (∃ (labeling : edge_labeling V), valid_labeling poly labeling)

theorem convex_polyhedron_edge_labeling_exists (V : Type) [fintype V] 
  (poly : ConvexPolyhedron V) : 
  does_valid_labeling_exist V poly :=
begin
  -- Proof omitted
  sorry,
end

end convex_polyhedron_edge_labeling_exists_l544_544363


namespace trig_problem_l544_544270

variables (θ : ℝ)

theorem trig_problem (h : Real.sin (2 * θ) = 1 / 2) : 
  Real.tan θ + 1 / Real.tan θ = 4 :=
sorry

end trig_problem_l544_544270


namespace equation_no_solution_for_k_7_l544_544189

theorem equation_no_solution_for_k_7 :
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ 5) → ¬ (x ^ 2 - 1) / (x - 3) = (x ^ 2 - 7) / (x - 5) :=
by
  intro x h
  have h1 : x ≠ 3 := h.1
  have h2 : x ≠ 5 := h.2
  sorry

end equation_no_solution_for_k_7_l544_544189


namespace at_least_one_pass_l544_544106

noncomputable def probability_A_correct : ℚ := 2 / 3
noncomputable def probability_B_correct : ℚ := 1 / 2

def binomial (n k : ℕ) : ℕ := choose n k

noncomputable def probability_A_fail (questions_correct : ℕ) : ℚ :=
  binomial 3 questions_correct * (probability_A_correct ^ questions_correct) *
  ((1 - probability_A_correct) ^ (3 - questions_correct))

noncomputable def probability_B_fail (questions_correct : ℕ) : ℚ :=
  binomial 3 questions_correct * (probability_B_correct ^ questions_correct) *
  ((1 - probability_B_correct) ^ (3 - questions_correct))

noncomputable def probability_both_fail : ℚ :=
  (probability_A_fail 1 + probability_A_fail 0) *
  (probability_B_fail 1 + probability_B_fail 0)

noncomputable def probability_at_least_one_pass : ℚ :=
  1 - probability_both_fail

theorem at_least_one_pass :
  probability_at_least_one_pass = 47 / 54 := by
  sorry

end at_least_one_pass_l544_544106


namespace lyla_laps_when_isabelle_passes_fifth_time_l544_544345

-- Define initial conditions
def lyla_speed : ℝ := 1   -- Suppose Lyla's speed is 1 lap per unit time
def isabelle_speed : ℝ := 1.25 -- Isabelle's speed is 25% faster than Lyla's

-- Lyla completes one-third of a lap before Isabelle starts running
def initial_distance_lyla : ℝ := 1 / 3

-- Total distance after 5th passing
def total_passing_distance : ℕ := 5

-- Total laps completed by Lyla when Isabelle passes her 5th time is 17 laps
theorem lyla_laps_when_isabelle_passes_fifth_time : 
  (total_passing_distance * 12 - initial_distance_lyla * total_passing_distance * 3) / 12 = 17 :=
by
  sorry

end lyla_laps_when_isabelle_passes_fifth_time_l544_544345


namespace product_base_n_number_l544_544591

open Nat

theorem product_base_n_number :
  let b_n (n : ℕ) := (n + 2)^2 / (n^3 - 1)
  (prod_eq_p_div_fact : ∀ p q : ℕ, 
  (∏ n in Finset.range (100 - 5 + 1), b_n (n + 5)) = p / q!
  → q = 100 ∧ p = 2014) :=
  sorry

end product_base_n_number_l544_544591


namespace selection_schemes_count_l544_544375

-- Definition of the problem in Lean 4
def people := Fin 6 -- 6 people, represented by elements of Fin 6
def places := Fin 4 -- 4 places, represented by elements of Fin 4

-- Define persons A and B
def A : people := 0
def B : people := 1

-- Define Xijiang Miao Village as a particular place
def Xijiang_Miao_Village : places := 2

-- Define a function to count valid scenarios
noncomputable def count_valid_scenarios : Nat :=
  let total := Nat.desc_factorial 6 4 -- A_6^4
  let A_scenarios := Nat.desc_factorial 5 3 -- A_5^3
  let B_scenarios := Nat.desc_factorial 5 3 -- A_5^3
  total - A_scenarios - B_scenarios

theorem selection_schemes_count :
  count_valid_scenarios = 240 :=
by
  sorry -- proof not required

end selection_schemes_count_l544_544375


namespace wilson_theorem_problem_remainder_l544_544181

-- Definitions and conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem wilson_theorem (p : ℕ) [fact (nat.prime p)] : (factorial (p-1)) + 1 ≡ 0 [MOD p] :=
by sorry

theorem problem_remainder : ((factorial 30) - 1) % 930 = 29 :=
by
  have prime_factors : 930 = 2 * 3 * 5 * 31 := by norm_num
  have h31 := wilson_theorem 31
  have h2 : (factorial 30) % 2 = 0 := by sorry  -- factorial 30 is even
  have h3 : (factorial 30) % 3 = 0 := by sorry  -- factorial 30 divisible by 3
  have h5 : (factorial 30) % 5 = 0 := by sorry  -- factorial 30 divisible by 5
  have h31_mod : (factorial 30 - 1) % 31 = 30 := by sorry  -- from Wilson's theorem
  apply chinese_remainder 29 31 _ _ _ _ _ _ sorry sorry sorry sorry

end wilson_theorem_problem_remainder_l544_544181


namespace bipartite_graph_non_equiv_l544_544325

variables {A B : Type} [Infinite A] [Infinite B] {E : A × B → Prop}

def E_X (X : set A) : set B := {b : B | ∃ a : A, a ∈ X ∧ E (a, b)}

theorem bipartite_graph_non_equiv :
  (¬ (∃ f : A → B, ∀ a1 a2 : A, f a1 = f a2 → a1 = a2 ∧ ∀ b : B, ∃ a : A, E (a, b))) ∧
  (∀ X : set A, finite X → (set.infinite (E_X X) ∨ X.to_finset.card ≤ (E_X X).to_finset.card)) :=
sorry

end bipartite_graph_non_equiv_l544_544325


namespace parabola_vertex_highest_point_l544_544007

def vertex_x (a b : ℝ) := -b / (2 * a)

def parabola_y (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_vertex_highest_point :
  vertex_x (-2) 28 = 7 ∧ parabola_y (-2) 28 418 7 = 516 := by
    sorry

end parabola_vertex_highest_point_l544_544007


namespace eccentricity_of_ellipse_equation_of_ellipse_at_max_area_l544_544561

variable (a b : ℝ) (k1 k2 : ℝ) (D : ℝ × ℝ)
variable (P Q : ℝ × ℝ)
variable (k1k2 : k1 * k2 = -(2/3))
variable (h : a > b ∧ b > 0)
variable (line_intersect : D = (-5, 0))
variable (DP2DQ : (dist D P) = 2 * (dist D Q))
variable (ellipse_eq : ∀ (x y : ℝ), (x, y) ∈ [P, Q] → x^2 / a^2 + y^2 / b^2 = 1)

theorem eccentricity_of_ellipse :
  sqrt(3) / 3 :=
sorry

theorem equation_of_ellipse_at_max_area : 
  ∀ (x y : ℝ), (∃ c, 2 * x^2 + 3 * y^2 = 250) := 
sorry

end eccentricity_of_ellipse_equation_of_ellipse_at_max_area_l544_544561


namespace equivalent_f_A_is_identical_to_1_over_x_l544_544444

def f_A (x : ℝ) : ℝ := x / (x^2)
def f_B (x : ℝ) : ℝ := 1 / Real.sqrt (x^2)
def f_C (x : ℝ) : ℝ := 1 / Real.sqrt (x)
def f_D (x a : ℝ) [Fact (0 < a)] [Fact (a ≠ 1)] : ℝ := a^(Real.logBase a (1/x))

theorem equivalent_f_A_is_identical_to_1_over_x (x : ℝ) (h : x ≠ 0) : f_A x = 1 / x :=
by sorry

end equivalent_f_A_is_identical_to_1_over_x_l544_544444


namespace probability_multiple_of_3_l544_544520

theorem probability_multiple_of_3 (die_outcomes : Finset ℕ)
  (die_vals : die_outcomes = {1, 2, 3, 4, 5, 6})
  (rolls : list (Fin 8 → die_outcomes)) :
  let not_3_or_6 := 4 / 6 := (2 / 3)
  let prob_not_3_or_6_8_rolls := (2 / 3) ^ 8
  let prob_1_or_6_once := 1 - prob_not_3_or_6_8_rolls
  (prob_1_or_6_once = 6305 / 6561) :=
begin
  sorry,
end

end probability_multiple_of_3_l544_544520


namespace image_of_point_under_mapping_l544_544605

def f (x y : ℝ) : ℝ × ℝ :=
  (Real.sqrt (1 / (x * y + 6 * y^2)), x^2 + y^3)

theorem image_of_point_under_mapping : f 6 (-3) = (1/6, 9) :=
by
  sorry

end image_of_point_under_mapping_l544_544605


namespace number_of_incorrect_statements_l544_544297

-- Definition of the conditions
def wall_thickness : ℕ := 5

-- Boring rates for big and small rats
def big_rat_boring_rate (n : ℕ) : ℕ := 2^(n - 1)
def small_rat_boring_rate (n : ℕ) : ℕ := nat.rec_on n 0 (λ _ IH, IH / 2)

-- Statements to be checked
def statement1 : Prop := small_rat_boring_rate 2 = 1 / 2
def statement2 : Prop := (2^2 - 1) + (2 - (1 / (2^(2 - 1)))) = 5
def statement3 : Prop := 
  let big_bored := big_rat_boring_rate 1 + big_rat_boring_rate 2 in
  let small_bored := small_rat_boring_rate 1 + small_rat_boring_rate 2 in
  big_bored + small_bored < wall_thickness

-- Main proof goal
theorem number_of_incorrect_statements : 
  (¬ statement1 ∨ ¬statement2 ∨ ¬statement3) ∧
  (statement1 ∧ ¬statement2 ∧ statement3) :=
begin
  sorry,
end

end number_of_incorrect_statements_l544_544297


namespace chord_length_is_sqrt_6_l544_544609

open Real

def circle_eq (x y : ℝ) := x^2 + y^2 + 4 * x - 4 * y + 6 = 0

def line_eq (k x y : ℝ) := k * x + y + 4 = 0

def point_on_line (k : ℝ) := (0, k)

def line_m (k x y : ℝ) := y = x + k

noncomputable def chord_length (k : ℝ) : ℝ := 
  let cx := -2
  let cy := 2
  let d := abs (cx + cy + k) / sqrt 2 in
  2 * sqrt (2 - d^2)

theorem chord_length_is_sqrt_6 (k : ℝ) 
  (h_line_symm : ∀ x y : ℝ, line_eq k x y → circle_eq x y)
  (h_point_on_line : (0, k) = point_on_line k) :
  chord_length k = sqrt 6 := by
  sorry

end chord_length_is_sqrt_6_l544_544609


namespace suitable_census_option_l544_544087

def isSuitableForCensus (option : Type) : Prop :=
  option = "standing long jump scores of students in a class"

theorem suitable_census_option : 
  isSuitableForCensus "standing long jump scores of students in a class" :=
by
  -- We assume based on the problem conditions that option 'A' is suitable for a census
  sorry

end suitable_census_option_l544_544087


namespace moon_arrangements_l544_544924

theorem moon_arrangements :
  (∃ (MOON : Finset (List Char)), 
    {w : List Char |
      w ∈ MOON ∧ w = ['M', 'O', 'O', 'N']}.card = 12) :=
sorry

end moon_arrangements_l544_544924


namespace bottle_and_beverage_weight_l544_544481

theorem bottle_and_beverage_weight 
  (B : ℝ)  -- Weight of the bottle in kilograms
  (x : ℝ)  -- Original weight of the beverage in kilograms
  (h1 : B + 2 * x = 5)  -- Condition: double the beverage weight total
  (h2 : B + 4 * x = 9)  -- Condition: quadruple the beverage weight total
: x = 2 ∧ B = 1 := 
by
  sorry

end bottle_and_beverage_weight_l544_544481


namespace problem1_problem2_l544_544899

variable {a b : ℝ}

theorem problem1 (ha : a ≠ 0) (hb : b ≠ 0) :
  (4 * b^3 / a) / (2 * b / a^2) = 2 * a * b^2 :=
by 
  sorry

theorem problem2 (ha : a ≠ b) :
  (a^2 / (a - b)) + (b^2 / (a - b)) - (2 * a * b / (a - b)) = a - b :=
by 
  sorry

end problem1_problem2_l544_544899


namespace length_of_platform_l544_544450

-- Conditions given in the problem
def train_length : ℝ := 300
def time_cross_pole : ℝ := 18
def time_cross_platform : ℝ := 30

-- Derived speed of the train
def train_speed : ℝ := train_length / time_cross_pole

-- Total distance covered when crossing the platform
def total_distance_cross_platform : ℝ := train_speed * time_cross_platform

-- The statement to be proven: length of the platform
theorem length_of_platform : total_distance_cross_platform - train_length = 200.1 := by
  -- This is where the proof would go, but we will replace it with sorry to indicate the proof is skipped
  sorry

end length_of_platform_l544_544450


namespace hyperbola_standard_equation_l544_544092

-- Definitions based on the given conditions:
def fociOnYAxis : Prop := true
def a : ℝ := 6
def b : ℝ := Real.sqrt 35

-- The theorem statement to be proved:
theorem hyperbola_standard_equation (h_foci : fociOnYAxis = true) (h_a : a = 6) (h_b : b = Real.sqrt 35) :
  (∀ x y : ℝ, (y^2 / 36 - x^2 / 35 = 1)) :=
sorry

end hyperbola_standard_equation_l544_544092


namespace nearest_integer_to_power_six_l544_544064

noncomputable def nearest_integer (x : ℝ) : ℤ := 
if x - real.floor x < real.ceil x - x then real.floor x else real.ceil x

theorem nearest_integer_to_power_six : 
  let x := 3 + real.sqrt 5 in
  let y := 3 - real.sqrt 5 in
  nearest_integer (x^6) = 2654 :=
by
  sorry

end nearest_integer_to_power_six_l544_544064


namespace magnitude_b_l544_544277

variable {α : Type*} [InnerProductSpace ℝ α]
variables (a b : α)

-- Define conditions
def condition_1 : Prop := ∥a∥ = 1
def condition_2 : Prop := inner (a + b) a = 0
def condition_3 : Prop := inner (3 • a + b) b = 0

-- Define the theorem to prove |b| = sqrt(3)
theorem magnitude_b :
  condition_1 a ∧ condition_2 a b ∧ condition_3 a b → ∥b∥ = Real.sqrt 3 :=
by 
  intros h,
  sorry

end magnitude_b_l544_544277


namespace max_parts_divided_by_three_planes_l544_544797

theorem max_parts_divided_by_three_planes : ∀ (planes : list Plane), 
  (length planes = 3) →
  (∀ p1 p2 p3, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    (exists c1 c2 c3, c1 ∥ p1 ∧ c1 ∥ p2 ∧ c1 ∥ p3 ↔ 4)
      ∨ (exists c1 c2 c3, c1 ∥ p1 ∧ c1 ∥ p2 ∧ c3 ∈ c1 ∩ c2 ∧ c2 ∥ p3 ↔ 6)
      ∨ (exists l, p1 ∩ p2 = l ∧ p2 ∩ p3 = l ∧ p3 ∩ p1 = l ↔ 6)
      ∨ (exists l1 l2 l3, p1 ∩ p2 = l1 ∧ p2 ∩ p3 = l2 ∧ p3 ∩ p1 = l3 ∧ parallel l1 l2 l3 ↔ 7)
      ∨ (exists l1 l2 l3, p1 ∩ p2 = l1 ∧ (p3 ∩ p1 ∪ p3 ∩ p2 = l2 ∧ p1 ∩ p2 ∩ p3 = l3) ↔ 8))
→ max (parts_divided_by_three_planes planes) = 8 :=
sorry

end max_parts_divided_by_three_planes_l544_544797


namespace angle_ABC_108_l544_544294

-- Condition 1: ABCDE is a pentagon
inductive Pentagon (A B C D E : Type) 

-- Condition 2: AB = BC = CD = DE = EA
structure RegularPentagon (A B C D E : Type) :=
(side_eq : A = B ∧ B = C ∧ C = D ∧ D = E ∧ E = A)

-- Condition 3: Each external angle at vertices A, B, C, D, and E equals 72 degrees
axiom external_angle_72 (A B C D E : Type) [Pentagon A B C D E] [RegularPentagon A B C D E] : 
  ∀ (P : Type), P ∈ {A, B, C, D, E} → ∀ (ext_angle : ∀ P, ℝ), ext_angle P = 72

-- Question: Prove that angle ABC = 108 degrees
theorem angle_ABC_108 (A B C D E : Type) [Pentagon A B C D E] [RegularPentagon A B C D E] :
  ∀ (internal_angle : ∀ P Q R : Type, ℝ), (internal_angle A B C = 108) :=
sorry

end angle_ABC_108_l544_544294


namespace magician_trick_l544_544491

theorem magician_trick (coins : Fin 11 → Bool) 
  (exists_pair : ∃ i : Fin 11, coins i = coins ((i + 1) % 11)) 
  (assistant_left_uncovered : Fin 11) 
  (uncovered_coin_same_face : coins assistant_left_uncovered = coins (some (exists_pair))) :
  ∃ j : Fin 11, j ≠ assistant_left_uncovered ∧ coins j = coins assistant_left_uncovered := 
  sorry

end magician_trick_l544_544491


namespace ellipse_standard_equation_ellipse_line_range_l544_544629

theorem ellipse_standard_equation (a b c : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : e = 1 / 2) (h4 : a - c = 1) :
  (e = c / a) → (b^2 = a^2 - c^2) → (∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1) := sorry

theorem ellipse_line_range (a b c k m : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : e = 1 / 2) (h4 : a - c = 1) :
  (exists x y : ℝ, x * y = 0) → (7 * m^2 = 12 + 12 * k^2) → 
  (-∞ < m ∧ m ≤ - (2 * real.sqrt 21) / 7) ∨ ((2 * real.sqrt 21) / 7 ≤ m ∧ m < +∞) := sorry

end ellipse_standard_equation_ellipse_line_range_l544_544629


namespace volleyball_team_math_count_l544_544142

theorem volleyball_team_math_count (total_players taking_physics taking_both : ℕ) 
  (h1 : total_players = 30) 
  (h2 : taking_physics = 15) 
  (h3 : taking_both = 6) 
  (h4 : total_players = 30 ∧ total_players = (taking_physics + (total_players - taking_physics - taking_both))) 
  : (total_players - (taking_physics - taking_both) + taking_both) = 21 := 
by
  sorry

end volleyball_team_math_count_l544_544142


namespace sqrt_quart_root_0_008_l544_544894

theorem sqrt_quart_root_0_008 : (Real.sqrt (Real.sqrt (Real.sqrt (0.008)) : ℝ) : ℝ) ≈ 0.55 :=
by
  -- prove the following approximation holds
  sorry

end sqrt_quart_root_0_008_l544_544894


namespace upper_limit_b_l544_544659

theorem upper_limit_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : 3 < b) (h4 : (a : ℚ) / b ≤ 3.75) : b ≤ 4 := by
  sorry

end upper_limit_b_l544_544659


namespace distance_car_to_stream_l544_544310

theorem distance_car_to_stream (total_distance : ℝ) (stream_to_meadow : ℝ) (meadow_to_campsite : ℝ) (h1 : total_distance = 0.7) (h2 : stream_to_meadow = 0.4) (h3 : meadow_to_campsite = 0.1) :
  (total_distance - (stream_to_meadow + meadow_to_campsite) = 0.2) :=
by
  sorry

end distance_car_to_stream_l544_544310


namespace prism_base_faces_and_lateral_edges_l544_544445

-- Define the concept of a prism in terms of its geometric properties
structure Prism (BaseFace : Type) :=
  (parallel_faces : ∃ f1 f2 : BaseFace, f1 ≠ f2 ∧ ∀ p : Point, p ∈ f1 ↔ p ∈ f2)
  (lateral_faces_are_parallelograms : ∀ lf : Face, is_lateral_face lf → is_parallelogram lf)

-- Define a predicate for base faces being parallel
def base_faces_are_parallel (P : Prism BaseFace) : Prop :=
  ∃ f1 f2 : BaseFace, f1 ≠ f2 ∧ (∀ p : Point, p ∈ f1 ↔ p ∈ f2)

-- Define a predicate for lateral edges being parallel
def lateral_edges_are_parallel (P : Prism BaseFace) : Prop :=
  ∀ e1 e2 : Edge, is_lateral_edge e1 → is_lateral_edge e2 → e1 ∥ e2

-- The main proof statement
theorem prism_base_faces_and_lateral_edges (P : Prism BaseFace) : 
  base_faces_are_parallel P ∧ lateral_edges_are_parallel P := 
by
  sorry

end prism_base_faces_and_lateral_edges_l544_544445


namespace pigeonhole_divisibility_l544_544931

theorem pigeonhole_divisibility (S : Finset ℕ) (hS : S.card = 51) (h_range : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 100) :
  ∃ (x y ∈ S), x ≠ y ∧ (x ∣ y ∨ y ∣ x) := sorry

end pigeonhole_divisibility_l544_544931


namespace initial_butterfly_count_l544_544937

theorem initial_butterfly_count (n : ℕ) (h : (2 / 3 : ℚ) * n = 6) : n = 9 :=
sorry

end initial_butterfly_count_l544_544937


namespace area_increase_percentage_l544_544028

def circle_perc_area_increase (r : ℝ) : ℝ :=
  let r_new := 1.5 * r
  let orig_area := π * r^2
  let new_area := π * r_new^2
  let area_increase := new_area - orig_area
  (area_increase / orig_area) * 100

theorem area_increase_percentage (r : ℝ) (h : r > 0) :
  circle_perc_area_increase r = 125 := by
  sorry

end area_increase_percentage_l544_544028


namespace correct_proposition_l544_544263

section Geometry

variable (l : Line) (m : Line) (α : Plane) (β : Plane)

-- Definitions based on the conditions
def line_perpendicular_plane (l : Line) (α : Plane) : Prop :=
  l.is_perpendicular α

def line_contained_in_plane (m : Line) (β : Plane) : Prop :=
  m.is_contained β

def planes_parallel (α : Plane) (β : Plane) : Prop :=
  α.is_parallel β

def planes_perpendicular (α : Plane) (β : Plane) : Prop :=
  α.is_perpendicular β

def lines_parallel (l : Line) (m : Line) : Prop :=
  l.is_parallel m

def lines_perpendicular (l : Line) (m : Line) : Prop :=
  l.is_perpendicular m

-- Propositions
def prop_a (l : Line) (m : Line) (α : Plane) (β : Plane) :=
  planes_parallel α β → l.is_skew m

def prop_b (l : Line) (m : Line) (α : Plane) (β : Plane) :=
  lines_parallel l m → planes_perpendicular α β

def prop_c (l : Line) (m : Line) (α : Plane) (β : Plane) :=
  planes_perpendicular α β → lines_parallel l m

def prop_d (l : Line) (m : Line) (α : Plane) (β : Plane) :=
  lines_perpendicular l m → planes_parallel α β

-- Correct answer proof problem
theorem correct_proposition (hl : line_perpendicular_plane l α) (hm : line_contained_in_plane m β) :
  prop_b l m α β :=
by sorry

end Geometry

end correct_proposition_l544_544263


namespace solve_for_a_l544_544223

noncomputable def a : ℝ := Real.root 4 3

theorem solve_for_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^b = b^a) (h4 : b = 9 * a) : a = Real.root 4 3 := by
  sorry

end solve_for_a_l544_544223


namespace sampling_proportion_l544_544112

noncomputable def stratified_sampling (total_employees : ℕ)
    (group1 : ℕ) (group2 : ℕ) (group3 : ℕ) (sample_size : ℕ) : (ℕ × ℕ × ℕ) :=
    let r := sample_size / total_employees in
    let sample1 := group1 * r in
    let sample2 := group2 * r in
    let sample3 := group3 * r in
    (sample1, sample2, sample3)

theorem sampling_proportion :
    stratified_sampling 500 125 280 95 100 = (25, 56, 19) :=
by
    -- Proof omitted
    sorry

end sampling_proportion_l544_544112


namespace even_and_period_of_pi_l544_544772

theorem even_and_period_of_pi (x : ℝ) : 
  (∀ x, f(-x) = f(x)) ∧ (∀ T > 0, T = π) :=
by
  sorry

def f (x : ℝ) : ℝ := 2 * (sin x) ^ 2 - 1

example : even_and_period_of_pi := by sorry

end even_and_period_of_pi_l544_544772


namespace total_profit_is_100_l544_544114

-- Conditions
variables (a_investment : ℕ) (a_period : ℕ)
variables (b_investment : ℕ) (b_period : ℕ)
variables (a_share : ℕ)

-- Setting values for the conditions
def a_investment := 300
def a_period := 12
def b_investment := 200
def b_period := 6
def a_share := 75

-- Theorem statement
theorem total_profit_is_100 (ainv := a_investment) (aper := a_period)
  (binv := b_investment) (bper := b_period) (ash := a_share) :
  let a_inv_share := ainv * aper,
      b_inv_share := binv * bper,
      total_inv_share := a_inv_share + b_inv_share,
      a_ratio := a_inv_share / total_inv_share in
  a_share / a_ratio = 100 :=
by
  -- Proof needed here
  sorry

end total_profit_is_100_l544_544114


namespace shortest_distance_to_line_from_circle_point_l544_544360

noncomputable def circle_center : (ℝ × ℝ) := (5, 3)
noncomputable def circle_radius : ℝ := 3

theorem shortest_distance_to_line_from_circle_point :
  ∀ (M : ℝ × ℝ), (M.1 - 5)^2 + (M.2 - 3)^2 = 9 → (3 * M.1 + 4 * M.2 - 2) / (sqrt (3^2 + 4^2)) = 2 :=
by
  sorry

end shortest_distance_to_line_from_circle_point_l544_544360


namespace concave_quadratic_g_range_l544_544570

def is_concave (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, f ((x1 + x2) / 2) ≤ (f x1 + f x2) / 2

def quadratic (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

theorem concave_quadratic (a : ℝ) (h : 0 < a) : is_concave (quadratic a) :=
  sorry

def g (a : ℝ) : ℝ :=
  if 0 < a ∧ a ≤ 1/2 then a - 1 else -1/(4*a)

theorem g_range : set.Ioo (-1 : ℝ) 0 = { y : ℝ | ∃ a : ℝ, (0 < a → y = g a ∧ (0 < a ∧ a ≤ 1/2 ∨ a > 1/2)) } :=
  sorry

end concave_quadratic_g_range_l544_544570


namespace find_max_area_triangle_PDE_l544_544361

noncomputable def triangle_max_area {α : Type*} [LinearOrderedField α] (AB AC : α) (sin_BAC : α) 
  (s t : α) (h_s : 0 < s) (h_s1 : s < 1) (h_t : 0 < t) (h_t1 : t < 1) : α :=
  5 * Real.sqrt 2 - 7

-- Lean 4 statement for the proof problem
theorem find_max_area_triangle_PDE : 
  (∀ (D E : Point) (hD : D ∈ segment AB) (hE : E ∈ segment AC),
  ∃ P ∈ line BE ∩ line CD,
  let s := AD / AB
  let t := AE / AC
  ∀ (h_area : area quadrilateral BCED = 2 * area (triangle PBC)),
  area (triangle PDE) ≤ 5 * Real.sqrt 2 - 7) := 
sorry

end find_max_area_triangle_PDE_l544_544361


namespace special_divisors_count_of_20_30_l544_544274

def prime_number (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def number_of_divisors (a : ℕ) (α β : ℕ) : ℕ := (α + 1) * (β + 1)

def count_special_divisors (m n : ℕ) : ℕ :=
  let total_divisors_m := (m + 1) * (n + 1)
  let total_divisors_n := (n + 1) * (n / 2 + 1)
  (total_divisors_m - 1) / 2 - total_divisors_n + 1

theorem special_divisors_count_of_20_30 (d_20_30 d_20_15 : ℕ) :
  let α := 60
  let β := 30
  let γ := 30
  let δ := 15
  prime_number 2 ∧ prime_number 5 ∧
  count_special_divisors α β = 1891 ∧
  count_special_divisors γ δ = 496 →
  d_20_30 = 2 * 1891 / 2 ∧
  d_20_15 = 2 * 496 →
  count_special_divisors 60 30 - count_special_divisors 30 15 + 1 = 450
:= by
  sorry

end special_divisors_count_of_20_30_l544_544274


namespace tank_filled_correct_l544_544125

variables (p q : ℝ) (z w : ℝ)

def three_fourths_filled : Prop := z + (3 / 4) * w = p
def one_third_filled : Prop := z + (1 / 3) * w = q
def totally_filled := z + w = (8 * p - 3 * q) / 5

theorem tank_filled_correct :
  three_fourths_filled p q z w →
  one_third_filled p q z w →
  totally_filled p q z w :=
by
  intros h1 h2
  sorry

end tank_filled_correct_l544_544125


namespace annual_interest_approx_l544_544902

noncomputable def P : ℝ := 10000
noncomputable def r : ℝ := 0.05
noncomputable def t : ℝ := 1
noncomputable def e : ℝ := Real.exp 1

theorem annual_interest_approx :
  let A := P * Real.exp (r * t)
  let interest := A - P
  abs (interest - 512.71) < 0.01 := sorry

end annual_interest_approx_l544_544902


namespace number_of_ways_to_select_team_at_least_2_boys_2_girls_l544_544352

def boys : ℕ := 10
def girls : ℕ := 10
def team_size : ℕ := 8

theorem number_of_ways_to_select_team_at_least_2_boys_2_girls : 
  ∃ ways: ℕ, ways = 123480 ∧ 
                boys = 10 ∧ 
                girls = 10 ∧ 
                team_size = 8 ∧ 
                (2 ≤ boys) ∧ 
                (2 ≤ girls) ∧ 
                (ways = (choose 10 2 * choose 10 6 + 
                         choose 10 3 * choose 10 5 + 
                         choose 10 4 * choose 10 4 + 
                         choose 10 5 * choose 10 3 + 
                         choose 10 6 * choose 10 2)) := 
by {
    use 123480,
    sorry
}

end number_of_ways_to_select_team_at_least_2_boys_2_girls_l544_544352


namespace function_property_verification_l544_544763

theorem function_property_verification :
  (∀ a : ℝ, a > 0 ∧ a ≠ 1 → 
    (set.univ = {x : ℝ | true} ∧ set.univ = {x : ℝ | true})) ∧
  (∀ k : ℝ, k > 0 → 
    (∃ t : ℝ, k = 3^t) ∧ true) ∧
  (∀ x : ℝ, x ≠ 0 → 
    (y = (1/2 + 1/(2^x - 1)) = - (y =  (1/2 + 1/(2^(-x) - 1))) ) ∧ 
    (y = x * (1/(3^x - 1) + 1/2) = y = (-x) * (1/(3^(-x) - 1) + 1/2)) ) ∧
  (∀ (f : ℝ → ℝ) (x₁ m n : ℝ), 
    f x₁ = 0 ∧ m < x₁ ∧ x₁ < n →
    ¬ (f(m) * f(n) < 0)) :=
by
  sorry

end function_property_verification_l544_544763


namespace price_of_one_rose_l544_544347

theorem price_of_one_rose
  (tulips1 tulips2 tulips3 roses1 roses2 roses3 : ℕ)
  (price_tulip : ℕ)
  (total_earnings : ℕ)
  (R : ℕ) :
  tulips1 = 30 →
  roses1 = 20 →
  tulips2 = 2 * tulips1 →
  roses2 = 2 * roses1 →
  tulips3 = 10 * tulips2 / 100 →  -- simplification of 0.1 * tulips2
  roses3 = 16 →
  price_tulip = 2 →
  total_earnings = 420 →
  (96 * price_tulip + 76 * R) = total_earnings →
  R = 3 :=
by
  intros
  -- Proof will go here
  sorry

end price_of_one_rose_l544_544347


namespace area_of_smaller_part_l544_544860

noncomputable def average (a b : ℝ) : ℝ :=
  (a + b) / 2

theorem area_of_smaller_part:
  ∃ A B : ℝ, A + B = 900 ∧ (B - A) = (1 / 5) * average A B ∧ A = 405 :=
by
  sorry

end area_of_smaller_part_l544_544860


namespace combined_weight_proof_l544_544658

variable (Jake_weight : ℕ := 93)
variable (loss_weight : ℕ := 15)
variable (twice : ℕ := 2)

def sister_weight (Jake_weight loss_weight twice : ℕ) : ℕ :=
  (Jake_weight - loss_weight) / twice

def combined_weight (Jake_weight : ℕ) (sister_weight : ℕ) : ℕ :=
  Jake_weight + sister_weight Jake_weight loss_weight twice

theorem combined_weight_proof : combined_weight Jake_weight (sister_weight Jake_weight loss_weight twice) = 132 := 
by 
  sorry

end combined_weight_proof_l544_544658


namespace at_least_one_nonnegative_l544_544220

theorem at_least_one_nonnegative
  (a1 a2 a3 a4 a5 a6 a7 a8 : ℝ)
  (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (h3 : a3 ≠ 0) (h4 : a4 ≠ 0)
  (h5 : a5 ≠ 0) (h6 : a6 ≠ 0) (h7 : a7 ≠ 0) (h8 : a8 ≠ 0)
  : (a1 * a3 + a2 * a4 ≥ 0) ∨ (a1 * a5 + a2 * a6 ≥ 0) ∨ (a1 * a7 + a2 * a8 ≥ 0) ∨
    (a3 * a5 + a4 * a6 ≥ 0) ∨ (a3 * a7 + a4 * a8 ≥ 0) ∨ (a5 * a7 + a6 * a8 ≥ 0) := 
sorry

end at_least_one_nonnegative_l544_544220


namespace triangle_area_and_height_l544_544129

theorem triangle_area_and_height (a b c : ℝ) (h_a : a = 9) (h_b : b = 12) (h_c : c = 15) (h_right_triangle : a^2 + b^2 = c^2) :
  (1/2 * a * b = 54) ∧ (a * b / c = 7.2) :=
by
  -- Area calculation
  have area_calc : 1 / 2 * 9 * 12 = 54 := sorry,
  -- Height calculation
  have height_calc : 9 * 12 / 15 = 7.2 := sorry,
  -- Combine the results into the proof of the theorem
  exact ⟨area_calc, height_calc⟩

end triangle_area_and_height_l544_544129


namespace find_range_of_a_l544_544713

noncomputable def range_of_a : Set ℝ :=
  {a | let z_1 := 1 + 2 * Complex.i,
           z_2 := 1 - 2 * Complex.i,
           z_3 := -a + Complex.sqrt(a^2 - 1),
           z_4 := -a - Complex.sqrt(a^2 - 1)
       in (a ≠ 1) ∧ (a ≠ -1) ∧ (|1 + a| = 2 ∧ (a = -3)) ∨ -1 < a ∧ a < 1 } 

theorem find_range_of_a (a: ℝ) (h: ∃z, (z^2 - 2*z + 5)*(z^2 + 2*a*z + 1) = 0 ∧ 
  let z_1 := 1 + 2 * Complex.i,
      z_2 := 1 - 2 * Complex.i,
      z_3 := -a + Complex.sqrt(a^2 - 1),
      z_4 := -a - Complex.sqrt(a^2 - 1) in 
  z ∈ {z_1, z_2, z_3, z_4}) : a ∈ {-3} ∪ Ioo (-1:ℝ) (1:ℝ) :=
by
  sorry

end find_range_of_a_l544_544713


namespace caterer_preparations_l544_544545

theorem caterer_preparations :
  let b_guests := 84
  let a_guests := (2/3) * b_guests
  let total_guests := b_guests + a_guests
  let extra_plates := 10
  let total_plates := total_guests + extra_plates

  let cherry_tomatoes_per_plate := 5
  let regular_asparagus_per_plate := 8
  let vegetarian_asparagus_per_plate := 6
  let larger_asparagus_per_plate := 12
  let larger_asparagus_portion_guests := 0.1 * total_plates

  let blueberries_per_plate := 15
  let raspberries_per_plate := 8
  let blackberries_per_plate := 10

  let cherry_tomatoes_needed := cherry_tomatoes_per_plate * total_plates

  let regular_portion_guests := 0.9 * total_plates
  let regular_asparagus_needed := regular_asparagus_per_plate * regular_portion_guests
  let larger_asparagus_needed := larger_asparagus_per_plate * larger_asparagus_portion_guests
  let asparagus_needed := regular_asparagus_needed + larger_asparagus_needed

  let blueberries_needed := blueberries_per_plate * total_plates
  let raspberries_needed := raspberries_per_plate * total_plates
  let blackberries_needed := blackberries_per_plate * total_plates

  cherry_tomatoes_needed = 750 ∧
  asparagus_needed = 1260 ∧
  blueberries_needed = 2250 ∧
  raspberries_needed = 1200 ∧
  blackberries_needed = 1500 :=
by
  -- Proof goes here
  sorry

end caterer_preparations_l544_544545


namespace only_solution_l544_544574

def phi : ℕ → ℕ := sorry  -- Euler's totient function
def d : ℕ → ℕ := sorry    -- Divisor function

theorem only_solution (n : ℕ) (h1 : n ∣ (phi n)^(d n) + 1) (h2 : ¬ d n ^ 5 ∣ n ^ (phi n) - 1) : n = 2 :=
sorry

end only_solution_l544_544574


namespace five_digit_numbers_with_one_odd_between_even_l544_544648

theorem five_digit_numbers_with_one_odd_between_even :
  ∃ n, n = 36 ∧
       (∀ (lst : List ℕ), lst.length = 5 ∧ lst.nodup ∧
        lst ⊆ [1, 2, 3, 4, 5] →
        (∃ (o1 o2 : ℕ), (o1 ∈ lst ∧ o2 ∈ lst ∧
        o1 % 2 = 1 ∧ o2 % 2 = 1 ∧ list.filter (λ x, x % 2 = 0) lst = [___, o1, ___, ___]
       ) ∨ 
        ∃ (e1 e2 e3 : ℕ), (e1 % 2 = 0 ∧ e2 % 2 = 0 ∧ e3 % 2 = 0 ∧ list.filter (λ x, x % 2 = 1) lst = [e1, e2, e3])))
  :=
sorry

end five_digit_numbers_with_one_odd_between_even_l544_544648


namespace moon_arrangement_l544_544918

theorem moon_arrangement : 
  let M_count := 1
  let O_count := 2
  let N_count := 1
  let total_letters := 4
  ∑ perm : List Nat, perm.permutations.length = total_letters! // (M_count! * O_count! * N_count!) :=
  12
 :=
begin
  -- Definitions from the condition
  have M_count := 1,
  have O_count := 2,
  have N_count := 1,
  have total_letters := 4,
  
  -- Applying formulas for permutation counts
  let num_unique_arrangements := total_letters.factorial / (M_count.factorial * O_count.factorial * N_count.factorial),
  show num_unique_arrangements = 12 from sorry
end

end moon_arrangement_l544_544918


namespace part_I_part_II_l544_544230

theorem part_I (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a + 4 / b) ≥ 9 := 
by 
  sorry

theorem part_II (a b x : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) (hineq : 1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) : 
  x ∈ Icc (-7) 11 := 
by 
  sorry

end part_I_part_II_l544_544230


namespace equilateral_triangle_ABO_side_length_l544_544743

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ :=
  (x, - (1 / 4) * x ^ 2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem equilateral_triangle_ABO_side_length :
  ∀ x : ℝ, x ≠ 0 → let A := point_on_parabola x in
                     let B := point_on_parabola (-x) in
                     let O := (0, 0) in
                     distance A O = distance A B →
                     distance A O = 4 * Real.sqrt 15 :=
by
  intros x hx A B O h
  sorry -- Proof to be completed

end equilateral_triangle_ABO_side_length_l544_544743


namespace segments_to_start_l544_544382

-- Define the problem statement conditions in Lean 4
def concentric_circles : Prop := sorry -- Placeholder, as geometry involving tangents and arcs isn't directly supported

def chord_tangent_small_circle (AB : Prop) : Prop := sorry -- Placeholder, detailing tangency

def angle_ABC_eq_60 (A B C : Prop) : Prop := sorry -- Placeholder, situating angles in terms of Lean formalism

-- Proof statement
theorem segments_to_start (A B C : Prop) :
  concentric_circles →
  chord_tangent_small_circle (A ↔ B) →
  chord_tangent_small_circle (B ↔ C) →
  angle_ABC_eq_60 A B C →
  ∃ n : ℕ, n = 3 :=
sorry

end segments_to_start_l544_544382


namespace question_to_prove_l544_544185

variable (a_n b_n : ℕ → ℤ) (S_n T_n : ℕ → ℤ)

noncomputable def Sn (n : ℕ) : ℤ := n * (38 * n + 14)
noncomputable def Tn (n : ℕ) : ℤ := n * (2 * n + 1)

axiom arithmetic_sequences
  (h1 : ∀ n, S_n n = Sn n)
  (h2 : ∀ n, T_n n = Tn n)
  (hn_pos : ∀ n, n > 0 → S_n n / T_n n = (38 * n + 14) / (2 * n + 1))
  (ha : ∀ n, a_n n = 76 * n - 24)
  (hb : ∀ n, b_n n = 4 * n - 1)

theorem question_to_prove : 
  ∀ n > 0, a_n 6 / b_n 7 = 16 :=
by
  sorry

end question_to_prove_l544_544185


namespace rectangle_width_correct_l544_544024

noncomputable def W : ℝ :=
  (4 * Real.pi + 5) / 3

theorem rectangle_width_correct :
  (let P_rectangle : ℝ := 2 * (2 * W + W);
       C_circle : ℝ := 2 * Real.pi * 4 in
   P_rectangle = C_circle + 10) →
  W ≈ 5.85 :=
by
  sorry

end rectangle_width_correct_l544_544024


namespace probability_of_spinner_in_shaded_region_l544_544858

-- Definitions representing the conditions
def is_equilateral_triangle (Δ : Type) : Prop := sorry -- Placeholder for defining an equilateral triangle
def is_median (Δ : Type) (m : Type) : Prop := sorry -- Placeholder for defining a median
def divides_equal_regions (Δ : Type) (regions : set (set Δ)) : Prop := sorry -- Placeholder for definition of equal regions division

-- Given definitions
variables {Δ : Type} [is_equilateral_triangle Δ]
variables {regions : set (set Δ)}
variable {shaded : set Δ}

-- The primary statement to be proved
theorem probability_of_spinner_in_shaded_region :
  divides_equal_regions Δ regions →
  ∃ (shaded : set (set Δ)), (card shaded = 3) ∧ (shaded ⊆ regions) →
  (probability (λ x, x ∈ shaded) (uniform_space regions) = (1 / 2)) :=
sorry

end probability_of_spinner_in_shaded_region_l544_544858


namespace proof_of_problem_statement_l544_544547

def problem_statement : Prop :=
  (1 * nat.C 10 1) + (2 * nat.C 10 2) + (3 * nat.C 10 3) + (4 * nat.C 10 4) + (5 * nat.C 10 5) + 
  (6 * nat.C 10 6) + (7 * nat.C 10 7) + (8 * nat.C 10 8) + (9 * nat.C 10 9) + (10 * nat.C 10 10) = 10240

theorem proof_of_problem_statement : problem_statement := 
by sorry

end proof_of_problem_statement_l544_544547


namespace F_properties_l544_544273

variable {R : Type*} [LinearOrderedField R]

/-- Definition of increasing function on reals -/
def is_increasing (f : R → R) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The core proof theorem statement -/
theorem F_properties (f : R → R) (h : is_increasing f) :
  ( ∀ x, F x = f x - f (-x) ) ∧ is_increasing F ∧ ∀ x, F x = -F (-x) :=
by
  let F := λ x : R, f x - f (-x)
  sorry

end F_properties_l544_544273


namespace Kindergarten_students_percentage_l544_544033

def total_students_Annville := 200
def total_students_Cleona := 150
def total_students_Brixton := 180

def percent_Kindergarten_Annville := 18 / 100
def percent_Kindergarten_Cleona := 10 / 100
def percent_Kindergarten_Brixton := 20 / 100

def Kindergarten_Annville := total_students_Annville * percent_Kindergarten_Annville
def Kindergarten_Cleona := total_students_Cleona * percent_Kindergarten_Cleona
def Kindergarten_Brixton := total_students_Brixton * percent_Kindergarten_Brixton

def total_Kindergarten_students := Kindergarten_Annville + Kindergarten_Cleona + Kindergarten_Brixton
def total_students := total_students_Annville + total_students_Cleona + total_students_Brixton

theorem Kindergarten_students_percentage :
  (total_Kindergarten_students / total_students) * 100 = 16.42 := by
  sorry

end Kindergarten_students_percentage_l544_544033


namespace large_block_dimension_ratio_l544_544473

theorem large_block_dimension_ratio
  (V_normal V_large : ℝ) 
  (k : ℝ)
  (h1 : V_normal = 4)
  (h2 : V_large = 32) 
  (h3 : V_large = k^3 * V_normal) :
  k = 2 := by
  sorry

end large_block_dimension_ratio_l544_544473


namespace triangle_perimeter_l544_544678

-- Define the isosceles triangle ABC with given conditions
variables {A B C : Type} [triangle ABC]
variables {sin_A sin_B : ℝ} (angle_ratio : sin_A / sin_B = 1 / 2)
variables (BC : ℝ) (isosceles_condition : is_isosceles ABC)
variables (BC_value : BC = 10)

-- Define the lengths of sides
variables (a b c : ℝ) (side_a : BC = a)
variables (side_b : 2 * a = b)
variables (side_c : b = c)

-- Define the perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem to prove the perimeter
theorem triangle_perimeter : perimeter a b c = 50 :=
by
  sorry -- The actual proof is omitted.

end triangle_perimeter_l544_544678


namespace sum_of_digits_base3_of_345_eq_5_l544_544812

theorem sum_of_digits_base3_of_345_eq_5 : ∑ d in [1, 1, 0, 2, 0, 1], d = 5 := by
  sorry

end sum_of_digits_base3_of_345_eq_5_l544_544812


namespace decompose_series_l544_544165

-- Define the 11-arithmetic Fibonacci sequence using the given series
def Φ₁₁₀ (n : ℕ) : ℕ :=
  if n % 11 = 0 then 0 else
  if n % 11 = 1 then 1 else
  if n % 11 = 2 then 1 else
  if n % 11 = 3 then 2 else
  if n % 11 = 4 then 3 else
  if n % 11 = 5 then 5 else
  if n % 11 = 6 then 8 else
  if n % 11 = 7 then 2 else
  if n % 11 = 8 then 10 else
  if n % 11 = 9 then 1 else
  0

-- Define the two geometric progressions
def G₁ (n : ℕ) : ℤ := 3 * (8 ^ n)
def G₂ (n : ℕ) : ℤ := 8 * (4 ^ n)

-- The decomposed sequence
def decomposedSequence (n : ℕ) : ℤ := G₁ n + G₂ n

-- The theorem to prove the decomposition
theorem decompose_series : ∀ n : ℕ, Φ₁₁₀ n = decomposedSequence n := by
  sorry

end decompose_series_l544_544165


namespace students_facing_coach_count_sixty_sixth_student_facing_coach_l544_544411

-- Definition of the students' numbers and conditions provided.
def students := {1, 2, ..., 240}
def multiples (n : ℕ) : list ℕ := (list.range' 1 240).filter (λ x, x % n = 0)

-- Define the sets given in the problem
def multiples_of_3 := multiples 3
def multiples_of_5 := multiples 5
def multiples_of_7 := multiples 7
def multiples_of_15 := multiples 15
def multiples_of_21 := multiples 21
def multiples_of_35 := multiples 35
def multiples_of_105 := multiples 105

-- Statement 1: Prove the number of students still facing the coach is 136
theorem students_facing_coach_count :
  students.card - 
    (multiples_of_3.card + multiples_of_5.card + multiples_of_7.card) + 
    (multiples_of_15.card + multiples_of_21.card + multiples_of_35.card) - 
    multiples_of_105.card = 136 := 
sorry

-- Statement 2: Prove the number reported by the 66th student facing the coach is 118
def non_multiples := (list.range' 1 240).filter (λ x, x % 3 ≠ 0 ∧ x % 5 ≠ 0 ∧ x % 7 ≠ 0)
theorem sixty_sixth_student_facing_coach : 
  non_multiples.nth 65 = some 118 :=
sorry

end students_facing_coach_count_sixty_sixth_student_facing_coach_l544_544411


namespace remainder_5n_div_3_l544_544813

theorem remainder_5n_div_3 (n : ℤ) (h : n % 3 = 2) : (5 * n) % 3 = 1 := by
  sorry

end remainder_5n_div_3_l544_544813


namespace g_value_at_five_sixths_l544_544999

noncomputable def f (x : ℝ) : ℝ := (sqrt 2) * sin (2 * π * x + π / 4)

def g : ℝ → ℝ
| x := if x < 0 then sin (2 * π * x) else g (x - 1)

theorem g_value_at_five_sixths : g (5 / 6) = - (sqrt 3) / 2 := 
by sorry

end g_value_at_five_sixths_l544_544999


namespace average_speed_is_approx_20_86_l544_544118

noncomputable def average_speed (distance_up: ℝ) (speed_up: ℝ) (distance_down: ℝ) (speed_down: ℝ) : ℝ :=
  let time_up := distance_up / speed_up
  let time_down := distance_down / speed_down
  let total_distance := distance_up + distance_down
  let total_time := time_up + time_down
  total_distance / total_time

theorem average_speed_is_approx_20_86 :
  average_speed 0.2 16 0.2 30 ≈ 20.86 :=
by
  sorry

end average_speed_is_approx_20_86_l544_544118


namespace distinct_integer_roots_of_quartic_eq_l544_544160

-- Declare a polynomial with integer coefficients
variables {a b c d : ℤ}

def polynomial := λ x : ℤ, x^4 + a*x^3 + b*x^2 + c*x + d

-- State that the possible number of distinct integer roots of the polynomial 
-- must be values from {0, 1, 2, 4}
theorem distinct_integer_roots_of_quartic_eq :
  ∃ m ∈ ({0, 1, 2, 4} : set ℕ), ∀ x : ℤ, polynomial x = 0 → m = {r ∈ ({0, 1, 2, 4} : set ℕ) | ∃ k, polynomial k = 0} :=
by
  sorry

end distinct_integer_roots_of_quartic_eq_l544_544160


namespace minimum_value_f_l544_544394

-- Define the function f
def f (x : ℝ) : ℝ := x + 9 / (2 * x - 2)

-- State the theorem
theorem minimum_value_f : 
  ∃ x : ℝ, x > 1 ∧ (∀ y : ℝ, y > 1 → f y ≥ 3 * Real.sqrt 2 + 1) ∧ f x = 3 * Real.sqrt 2 + 1 :=
by
  sorry

end minimum_value_f_l544_544394


namespace solve_inequality_l544_544754

theorem solve_inequality (a : ℝ) : 
  {x : ℝ | x^2 - (a + 2) * x + 2 * a > 0} = 
  (if a > 2 then {x | x < 2 ∨ x > a}
   else if a = 2 then {x | x ≠ 2}
   else {x | x < a ∨ x > 2}) :=
sorry

end solve_inequality_l544_544754


namespace general_term_l544_544303

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * a n + 1

theorem general_term (n : ℕ) : a (n+1) + 1 = (2:ℕ)^((n+1):ℕ) :=
by {
  sorry -- Proof goes here
}

end general_term_l544_544303


namespace find_a_l544_544300

theorem find_a (a : ℝ) : 3 * a + 150 = 360 → a = 70 := 
by 
  intro h
  sorry

end find_a_l544_544300


namespace sine_of_transformed_angle_l544_544269

variable {α m : Real}

theorem sine_of_transformed_angle (h1 : cos (π / 6 - α) = m) (h2 : abs m ≤ 1) : 
  sin (2 * π / 3 - α) = m := 
sorry

end sine_of_transformed_angle_l544_544269


namespace factorize_expression_l544_544176

theorem factorize_expression (x y : ℝ) : 2 * x^2 * y - 8 * y = 2 * y * (x + 2) * (x - 2) :=
  sorry

end factorize_expression_l544_544176


namespace obtuse_angle_range_l544_544195

def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (2, 2 * x + 3)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def range_of_x : set ℝ := {x | x < -2} ∪ {x | -2 < x ∧ x < -3 / 4}

theorem obtuse_angle_range (x : ℝ) :
  dot_product (vector_a x) (vector_b x) < 0 →
  x ∈ range_of_x :=
sorry

end obtuse_angle_range_l544_544195


namespace bulb_standard_probability_l544_544540

noncomputable def prob_A 
  (P_H1 : ℝ) (P_H2 : ℝ) (P_A_given_H1 : ℝ) (P_A_given_H2 : ℝ) :=
  P_A_given_H1 * P_H1 + P_A_given_H2 * P_H2

theorem bulb_standard_probability 
  (P_H1 : ℝ := 0.6) (P_H2 : ℝ := 0.4) 
  (P_A_given_H1 : ℝ := 0.95) (P_A_given_H2 : ℝ := 0.85) :
  prob_A P_H1 P_H2 P_A_given_H1 P_A_given_H2 = 0.91 :=
by
  sorry

end bulb_standard_probability_l544_544540


namespace binary_101110_to_octal_l544_544163

-- Definition: binary number 101110 represents some decimal number
def binary_101110 : ℕ := 0 * 2^0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5

-- Definition: decimal number 46 represents some octal number
def decimal_46 := 46

-- A utility function to convert decimal to octal (returns the digits as a list)
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else decimal_to_octal (n / 8) ++ [n % 8]

-- Hypothesis: the binary 101110 equals the decimal 46
lemma binary_101110_eq_46 : binary_101110 = decimal_46 := by sorry

-- Hypothesis: the decimal 46 converts to the octal number 56 (in list form)
def octal_56 := [5, 6]

-- Theorem: binary 101110 converts to the octal number 56
theorem binary_101110_to_octal :
  decimal_to_octal binary_101110 = octal_56 := by
  rw [binary_101110_eq_46]
  sorry

end binary_101110_to_octal_l544_544163


namespace parallelogram_has_4_altitudes_l544_544646

-- Define a parallelogram in geometric terms
structure Parallelogram (α : Type) [OrderedRing α] :=
  (A B C D : α × α)
  (AB || CD : ∀ x, x = A ∨ x = B -> x = C ∨ x = D)
  (BC || AD : ∀ x, x = B ∨ x = C -> x = A ∨ x = D)
  (AB = CD : distance A B = distance C D)
  (BC = AD : distance B C = distance A D)

-- Define altitude in terms of a line segment forming a right angle with a side and connecting the opposite side
def isAltitude (α : Type) [OrderedRing α] (P Q : α × α) (side : α × α × α × α) : Prop :=
  ∃ M : α × α, M ∈ line_segment side ∧ right_angle P Q M

-- State that every parallelogram has exactly 4 altitudes
theorem parallelogram_has_4_altitudes (α : Type) [OrderedRing α] (P : Parallelogram α) : ∃ (altitudes : Set (α × α × α × α)), altitudes.card = 4 ∧ ∀ (alt : α × α × α × α), alt ∈ altitudes → isAltitude α alt.1.1 alt.2.1 P := 
  sorry

end parallelogram_has_4_altitudes_l544_544646


namespace find_b_in_quadratic_eqn_l544_544003

theorem find_b_in_quadratic_eqn :
  ∃ (b : ℝ), ∃ (p : ℝ), 
  (∀ x, x^2 + b*x + 64 = (x + p)^2 + 16) → 
  b = 8 * Real.sqrt 3 :=
by 
  sorry

end find_b_in_quadratic_eqn_l544_544003


namespace magician_assistant_trick_l544_544486

/-- A coin can be either heads or tails. -/
inductive Coin
| heads : Coin
| tails : Coin

/-- Given a cyclic arrangement of 11 coins, there exists at least one pair of adjacent coins with the same face. -/
theorem magician_assistant_trick (coins : Fin 11 → Coin) : 
  ∃ i : Fin 11, coins i = coins (i + 1) := 
by
  sorry

end magician_assistant_trick_l544_544486


namespace quadratic_inequality_false_iff_l544_544783

theorem quadratic_inequality_false_iff (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x^2 - 3 * a * x + 9 < 0) ↔ (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
by sorry

end quadratic_inequality_false_iff_l544_544783


namespace ellipse_foci_x_axis_l544_544014

theorem ellipse_foci_x_axis (k : ℝ) : 
  (0 < k ∧ k < 2) ↔ (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∧ a > b) := 
sorry

end ellipse_foci_x_axis_l544_544014


namespace smallest_value_expr_geq_16_div_3_l544_544330

noncomputable def smallest_value_expr (a b c d : ℝ) : ℝ :=
  (a + b + c + d) * ((1 / (a + b + c)) + (1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem smallest_value_expr_geq_16_div_3 (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_value_expr a b c d ≥ 16 / 3 :=
begin
  sorry
end

end smallest_value_expr_geq_16_div_3_l544_544330


namespace average_speed_to_work_calculation_l544_544555

noncomputable def coles_average_speed_on_work : ℝ :=
let v := v in
  let distance_to_work := v * (7 / 6) in
  let time_to_work := (7 / 6) in
  let return_speed := 105 in
  let return_time := 2 - (7 / 6) in
  let distance_return := return_speed * return_time in
  v
  
theorem average_speed_to_work_calculation (v : ℝ) : 
    let distance_to_work := v * (7 / 6),
        time_to_work := (7 / 6),
        return_speed := 105,
        return_time := 2 - (7 / 6) in
    let distance_return := return_speed * return_time in
    distance_to_work = distance_return ∧ v = 75 :=
by 
    sorry

end average_speed_to_work_calculation_l544_544555


namespace find_P_Q_l544_544940

theorem find_P_Q :
  ∃ P Q, (∀ x : ℝ, x ≠ 12 ∧ x ≠ -4 →
    (7*x + 4) / ((x - 12) * (x + 4)) = P / (x - 12) + Q / (x + 4)) ∧ P = 5.5 ∧ Q = 1.5 :=
by 
  use 5.5, 1.5
  split
  { intros x x_cond,
    assume h : x ≠ 12 ∧ x ≠ -4,
    rw [mul_assoc, mul_inv_cancel, add_mul, sub_mul, mul_add, mul_sub, ←add_assoc],
    sorry },
    { split; refl }

end find_P_Q_l544_544940


namespace complement_union_l544_544720

def U : set ℤ := {-2, -1, 0, 1, 2}
def A : set ℤ := {-1, 2}
def B : set ℤ := {-2, 2}

-- Prove that the union of the complement of A in U and B equals {-2, 0, 1, 2}
theorem complement_union (CUA : set ℤ := U \ A) (CUA_union_B : set ℤ := CUA ∪ B) :
  CUA_union_B = {-2, 0, 1, 2} :=
by sorry

end complement_union_l544_544720


namespace sin_A_value_l544_544679

theorem sin_A_value {A B C : Type} [normed_add_torsor' A B C] (TriangleRight : ∀ (ABC : Triangle A), (angle ABC.C B A = π / 2)) (HypAB_2BC : ∀ (ABC : Triangle A), (distance ABC.C ABC.A = 2 * distance ABC.C ABC.B)) :
  ∃ (ABC : Triangle A), sin (angle ABC.B ABC.C ABC.A) = 1 / 2 :=
by
  sorry

end sin_A_value_l544_544679


namespace equation_of_parabola_slopes_product_l544_544258

-- Variables
variables (a : ℝ) (p : ℝ) (M : ℝ × ℝ := (a, -2 * p)) (focus : ℝ × ℝ := (0, p / 2))

-- Conditions and definitions
noncomputable def parabola := { x : ℝ × ℝ | x.1^2 = 2 * p * x.2 }
noncomputable def tangent_points := { A B : ℝ × ℝ // A ∈ parabola ∧ B ∈ parabola }
noncomputable def l_line (p : ℝ) := { x : ℝ × ℝ | x.1 / 2 + 2 * x.2 / p = 1 }
noncomputable def projections := (Q' N' : ℝ × ℝ × ℝ) 

-- Length condition |Q'N'| = 2√5
axiom abs_distance {Q N : ℝ × ℝ} : (Q.1 - N.1)^2 + (Q.2 - N.2)^2 = 20

-- Slopes of lines AM and BM
noncomputable def slope_AM (A : ℝ × ℝ) : ℝ := (A.2 - M.2) / (A.1 - M.1)
noncomputable def slope_BM (B : ℝ × ℝ) : ℝ := (B.2 - M.2) / (B.1 - M.1)

-- Define statement to prove equation of parabola
theorem equation_of_parabola : p = 2 → ∀ C₁ ∈ parabola, C₁.1^2 = 4 * C₁.2 :=
by sorry

-- Define statement to prove slopes multiplication
theorem slopes_product (A B : ℝ × ℝ) (hA : A ∈ parabola) (hB : B ∈ parabola) : 
  slope_AM A * slope_BM B = -4 :=
by sorry

end equation_of_parabola_slopes_product_l544_544258


namespace rudy_total_running_time_l544_544367

-- Definitions for the conditions
def time_first_part := 5 * 10
def time_second_part := 4 * 9.5

-- Statement to be proved
theorem rudy_total_running_time : time_first_part + time_second_part = 88 := by
  sorry

end rudy_total_running_time_l544_544367


namespace cost_per_box_l544_544847

noncomputable def cost_per_box_per_month : ℝ :=
  let box_volume := 15 * 12 * 10
  let total_volume := 1080000
  let total_cost := 120
  let number_of_boxes := total_volume / box_volume
  total_cost / number_of_boxes

theorem cost_per_box : cost_per_box_per_month = 0.20 := by
  let box_volume := 15 * 12 * 10
  let total_volume := 1080000
  let total_cost := 120
  let number_of_boxes := total_volume / box_volume
  have number_of_boxes_eq : number_of_boxes = 600 := by
    calc
      number_of_boxes = (total_volume / box_volume) : by rfl
      ... = (1080000 / (15 * 12 * 10)) : by rfl
      ... = 600 : by norm_num
  calc
    cost_per_box_per_month = total_cost / number_of_boxes : by rfl
    ... = 120 / 600 : by rw number_of_boxes_eq
    ... = 0.20 : by norm_num

end cost_per_box_l544_544847


namespace proof_cost_A_B_schools_proof_renovation_plans_l544_544043

noncomputable def cost_A_B_schools : Prop :=
  ∃ (x y : ℝ), 2 * x + 3 * y = 78 ∧ 3 * x + y = 54 ∧ x = 12 ∧ y = 18

noncomputable def renovation_plans : Prop :=
  ∃ (a : ℕ), 3 ≤ a ∧ a ≤ 5 ∧ 
    (1200 - 300) * a + (1800 - 500) * (10 - a) ≤ 11800 ∧
    300 * a + 500 * (10 - a) ≥ 4000

theorem proof_cost_A_B_schools : cost_A_B_schools :=
sorry

theorem proof_renovation_plans : renovation_plans :=
sorry

end proof_cost_A_B_schools_proof_renovation_plans_l544_544043


namespace platform_length_correct_l544_544840

noncomputable def length_of_platform (train_length: ℝ) (train_speed_kmh: ℝ) (crossing_time: ℝ): ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

theorem platform_length_correct :
  length_of_platform 250 55 50.395968322534195 ≈ 520.00 :=
begin
  -- We setup the conditions
  let train_length := 250
  let train_speed_kmh := 55
  let crossing_time := 50.395968322534195

  -- Calculate the speed in m/s
  let train_speed_ms := train_speed_kmh * 1000 / 3600

  -- Calculate the total distance
  let total_distance := train_speed_ms * crossing_time

  -- Determine the platform length
  let platform_length := total_distance - train_length

  -- Assert that the platform length is approximately 520
  have : abs (platform_length - 520) < 0.0001, by sorry,

  exact this,
end

end platform_length_correct_l544_544840


namespace find_a_l544_544611

open Set

noncomputable def A : Set ℝ := {x | x^2 - 2 * x - 8 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + a * x + a^2 - 12 = 0}

theorem find_a (a : ℝ) : (A ∪ (B a) = A) ↔ (a = -2 ∨ a ≥ 4 ∨ a < -4) := by
  sorry

end find_a_l544_544611


namespace race_outcomes_l544_544881

theorem race_outcomes (P : Fin 6 → Type*) :
  ∃ (n : ℕ), n = 360 ∧ ∀ (positions : Fin 4 → Fin 6), injective positions :=
sorry

end race_outcomes_l544_544881


namespace smallest_value_a2_b2_c2_l544_544717

theorem smallest_value_a2_b2_c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 3 * b + 4 * c = 120) : 
  a^2 + b^2 + c^2 ≥ 14400 / 29 :=
by sorry

end smallest_value_a2_b2_c2_l544_544717


namespace problem_b_value_l544_544383

theorem problem_b_value (b : ℤ)
  (h1 : 0 ≤ b)
  (h2 : b ≤ 20)
  (h3 : (3 - b) % 17 = 0) : b = 3 :=
sorry

end problem_b_value_l544_544383


namespace vasya_gift_choices_l544_544737

theorem vasya_gift_choices :
  let cars := 7
  let construction_sets := 5
  (cars * construction_sets + Nat.choose cars 2 + Nat.choose construction_sets 2) = 66 :=
by
  sorry

end vasya_gift_choices_l544_544737


namespace largest_angle_in_triangle_l544_544404

theorem largest_angle_in_triangle (a b c : ℝ)
  (h1 : a + b = (4 / 3) * 90)
  (h2 : b = a + 36)
  (h3 : a + b + c = 180) :
  max a (max b c) = 78 :=
sorry

end largest_angle_in_triangle_l544_544404


namespace simplify_and_evaluate_expr_l544_544752

-- Define the variables and their values
def a : ℝ := 2
def b : ℝ := -1/2
def c : ℝ := -1

-- Define the expression to be evaluated
def expr (a b c : ℝ) :=
  a * b * c - (2 * a * b - (3 * a * b * c - b * c) + 4 * a * b * c)

-- The theorem stating that the expression evaluates to 3/2 given the conditions
theorem simplify_and_evaluate_expr : expr a b c = 3 / 2 :=
by
  sorry

end simplify_and_evaluate_expr_l544_544752


namespace hyperbola_eccentricity_l544_544161

noncomputable def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_eccentricity (a b x y c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (hM : hyperbola c (b^2 / a) a b)
  (h_angle : tan (π / 6) = (b^2 / a) / (2 * c))
  (h_perpendicular : (c^2 - a^2) / (2 * a * c) = sqrt 3 / 3) :
  ∃ e : ℝ, e = sqrt 3 :=
by 
  sorry

end hyperbola_eccentricity_l544_544161


namespace pattern_of_multiples_l544_544788

theorem pattern_of_multiples (P : set ℕ) (h1 : ∀ x ∈ P, x % 3 = 0) 
(h2 : ∑ x in P, x = 72) (h3 : 27 ∈ P) (h4 : ∀ x ∈ P, x ≤ 27) : 
P = {21, 24, 27} :=
sorry

end pattern_of_multiples_l544_544788


namespace jenny_proposal_time_l544_544313

theorem jenny_proposal_time (total_time research_time report_time proposal_time : ℕ) 
  (h1 : total_time = 20) 
  (h2 : research_time = 10) 
  (h3 : report_time = 8) 
  (h4 : proposal_time = total_time - research_time - report_time) : 
  proposal_time = 2 := 
by
  sorry

end jenny_proposal_time_l544_544313


namespace melanie_statistics_2020_l544_544906

theorem melanie_statistics_2020 :
  let data := λ n, if 1 ≤ n ∧ n ≤ 29 then 12 else if 30 ≤ n ∧ n ≤ 31 then 11 else 0 in
  let days := (Σ n, vector.repeat n (data n)).length in
  let median := 16 in
  let mean := 5891 / 366 in
  let modes := list.range' 1 29 in
  let d := 15 in
  d < mean ∧ mean < median :=
by
  sorry

end melanie_statistics_2020_l544_544906


namespace nearest_integer_to_power_six_l544_544066

noncomputable def nearest_integer (x : ℝ) : ℤ := 
if x - real.floor x < real.ceil x - x then real.floor x else real.ceil x

theorem nearest_integer_to_power_six : 
  let x := 3 + real.sqrt 5 in
  let y := 3 - real.sqrt 5 in
  nearest_integer (x^6) = 2654 :=
by
  sorry

end nearest_integer_to_power_six_l544_544066


namespace angle_BFC_eq_90_l544_544318

-- Definitions and conditions
variables {A B C D E F : Type} 
          [geometry A] [geometry B] [geometry C] [geometry D] [geometry E] [geometry F]

-- Given data
-- Define types and geometric objects
noncomputable def Trapezium (AB CD AD AC BD F : Type) : Prop :=
AB + CD = AD ∧
(AC ∩ BD = E) ∧
(Line_through E ∥ Line_through AB) ∧ (Line_through E ∥ Line_through CD) ∧ 
(Line_through E ∩ Line_through AD = F)

-- Target statement to prove
theorem angle_BFC_eq_90 (AB CD AD AC BD F : Type) (h : Trapezium AB CD AD AC BD F) : 
  angle B F C = 90 :=
by sorry

end angle_BFC_eq_90_l544_544318


namespace combined_age_of_four_siblings_l544_544527

theorem combined_age_of_four_siblings :
  let aaron_age := 15
  let sister_age := 3 * aaron_age
  let henry_age := 4 * sister_age
  let alice_age := aaron_age - 2
  aaron_age + sister_age + henry_age + alice_age = 253 :=
by
  let aaron_age := 15
  let sister_age := 3 * aaron_age
  let henry_age := 4 * sister_age
  let alice_age := aaron_age - 2
  have h1 : aaron_age + sister_age + henry_age + alice_age = 15 + 3 * 15 + 4 * (3 * 15) + (15 - 2) := by sorry
  have h2 : 15 + 3 * 15 + 4 * (3 * 15) + (15 - 2) = 253 := by sorry
  exact h1.trans h2

end combined_age_of_four_siblings_l544_544527


namespace nearest_integer_is_11304_l544_544071

def nearest_integer_to_a_plus_b_pow_six (a b : ℝ) (h : b = Real.sqrt 5) : ℝ :=
  (a + b) ^ 6

theorem nearest_integer_is_11304 : nearest_integer_to_a_plus_b_pow_six 3 (Real.sqrt 5) rfl = 11304 := 
  sorry

end nearest_integer_is_11304_l544_544071


namespace intersection_complement_l544_544610

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement :
  A ∩ (compl B) = {x | 0 < x ∧ x < 2} := by
  sorry

end intersection_complement_l544_544610


namespace triangle_solution_correct_l544_544532

theorem triangle_solution_correct :
  (∀ (a b : ℝ) (A : ℝ), 
    (a = 4 ∧ b = 5 ∧ A = real.pi / 6) → ¬∃ B, sin B = 5 * (1/2) / 4) ∧
  (∀ (a b : ℝ) (A : ℝ), 
    (a = 5 ∧ b = 4 ∧ A = real.pi / 3) → ¬∃ B, sin B = 4 * (real.sin (real.pi / 3)) / 5) ∧
  
  (∀ (a b : ℝ) (A : ℝ),
    (a = real.sqrt 3 ∧ b = real.sqrt 2 ∧ A = (2 * real.pi) / 3) → ¬∃ B, sin B = real.sqrt 2 * (real.sin ((2 * real.pi) / 3)) / real.sqrt 3) ∧

  (∀ (a b : ℝ) (A : ℝ),
    (a = real.sqrt 3 ∧ b = real.sqrt 6 ∧ A = real.pi / 3) → ∃ B, sin B > 1) :=
by
  sorry

end triangle_solution_correct_l544_544532


namespace union_sets_l544_544639

def setA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def setB : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 3 } :=
sorry

end union_sets_l544_544639


namespace jordan_rectangle_width_l544_544153

theorem jordan_rectangle_width :
  let carol_length := 5
  let carol_width := 24
  let jordan_length := 8
  ∃ (W : ℝ), carol_length * carol_width = jordan_length * W ∧ W = 15 := 
by
  let carol_length := 5
  let carol_width := 24
  let jordan_length := 8
  let carol_area := carol_length * carol_width
  let jordan_width := 15
  exists_resolve_left (15: ℝ)
  sorry

end jordan_rectangle_width_l544_544153


namespace transformed_function_is_correct_l544_544911

noncomputable def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then -x - 3
  else if -1 < x ∧ x ≤ 1 then -real.sqrt (1 - (x + 1)^2) + 1
  else if 1 < x ∧ x ≤ 4 then 3 * (x - 1)
  else 0  -- default value outside the defined range

noncomputable def transformed_function (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then 2 * x - 12
  else if 0 < x ∧ x ≤ 2 then 2 * real.sqrt (1 - x^2) - 6
  else if 2 < x ∧ x ≤ 5 then -6 * x + 8
  else 0  -- default value outside the defined range

theorem transformed_function_is_correct :
  ∀ x : ℝ, transformed_function x = -2 * g (x - 1) - 4 :=
sorry

end transformed_function_is_correct_l544_544911


namespace problem_statement_l544_544188
noncomputable def a : ℕ := 10
noncomputable def b : ℕ := a^3

theorem problem_statement (a b : ℕ) (a_pos : 0 < a) (b_eq : b = a^3)
    (log_ab : Real.logb a (b : ℝ) = 3) (b_minus_a : b = a + 891) :
    a + b = 1010 :=
by
  sorry

end problem_statement_l544_544188


namespace diagonals_concurrent_l544_544042

open EuclideanGeometry

-- Defining our points and parallelogram
variables (A B C D O M N P Q : Point)
variables (h_parallelogram : parallelogram A B C D)
variables (h_O : on_diagonal O B D)
variables (h_MN_parallel_AB : parallel M N A B)
variables (h_M_on_AD : on_segment M A D)
variables (h_N_on_BC : on_segment N B C)
variables (h_PQ_parallel_AD : parallel P Q A D)
variables (h_P_on_BC : on_segment P B C)
variables (h_Q_on_AB : on_segment Q A B)

-- Theorem statement
theorem diagonals_concurrent 
    (h_AO : line_through A O)
    (h_BP : line_through B P)
    (h_DN : line_through D N) : concurrent (line_through A O) (line_through B P) (line_through D N) :=
sorry

end diagonals_concurrent_l544_544042


namespace deck_cost_l544_544420

variable (rareCount : ℕ := 19)
variable (uncommonCount : ℕ := 11)
variable (commonCount : ℕ := 30)
variable (rareCost : ℝ := 1.0)
variable (uncommonCost : ℝ := 0.5)
variable (commonCost : ℝ := 0.25)

theorem deck_cost : rareCount * rareCost + uncommonCount * uncommonCost + commonCount * commonCost = 32 := by
  sorry

end deck_cost_l544_544420


namespace probability_two_red_two_blue_from_bag_l544_544843

theorem probability_two_red_two_blue_from_bag
  (num_red : ℕ)
  (num_blue : ℕ)
  (total_draws : ℕ)
  (prob_draw : ℚ) :
  num_red = 15 →
  num_blue = 10 →
  total_draws = 4 →
  prob_draw = 21 / 56 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- Proof goes here...
  sorry

end probability_two_red_two_blue_from_bag_l544_544843


namespace part_a_part_b_l544_544456

-- Condition that α, β, and γ are the angles of a triangle implies their sum is π.
variables {α β γ : ℝ}
hypothesis angles_sum_pi : α + β + γ = real.pi

-- Part (a)
theorem part_a :
  real.cot α * real.cot β + real.cot β * real.cot γ + real.cot α * real.cot γ = 1 :=
by sorry

-- Part (b)
theorem part_b :
  real.cot α + real.cot β + real.cot γ - real.cot α * real.cot β * real.cot γ = 
  1 / (real.sin α * real.sin β * real.sin γ) :=
by sorry

end part_a_part_b_l544_544456


namespace max_value_of_quadratic_l544_544393

theorem max_value_of_quadratic : 
  ∃ M, ∀ x : ℝ, -2 * x^2 + 9 ≤ M ∧ (∀ x : ℝ, M = -2 * 0^2 + 9) := 
begin
  use 9,
  intros x,
  split,
  { sorry },
  { intros x,
    simp,
    ring,
  },
end

end max_value_of_quadratic_l544_544393


namespace hyperbola_eccentricity_l544_544994

-- Definitions for the problem
variables (m n : ℝ) (h_cond : n > m ∧ m > 0 ∧ (m / 2 = n / 2))
variables (ell_eccurity : ℝ) (hyp_eccurity : ℝ)

-- The conditions on the ellipse and its eccentricity
def ellipse_conditions := mx^2 + ny^2 = 1 ∧ ell_eccurity = sqrt 2 / 2

-- The goal is to prove that the eccentricity of the hyperbola is sqrt(6) / 2
theorem hyperbola_eccentricity :
  ellipse_conditions m n ell_eccurity →
  hyp_eccurity = sqrt 6 / 2 ∧
  (mx^2 - ny^2 = 1) :=
begin
  sorry
end

end hyperbola_eccentricity_l544_544994


namespace partition_three_groups_no_calls_l544_544355

-- Assume that the residents and their phone call relations can be modeled as a simple graph.
-- Each resident is a vertex and each phone call is an edge.

open Finset

theorem partition_three_groups_no_calls (n : ℕ) (calls : Finset (Fin n × Fin n)) 
  (h_calls : ∀ {x y : Fin n}, (x, y) ∈ calls → x ≠ y) :
  ∃ (G₁ G₂ G₃ : Finset (Fin n)), 
    (∀ {x y : Fin n}, x ∈ G₁ → y ∈ G₁ → (x, y) ∉ calls) ∧
    (∀ {x y : Fin n}, x ∈ G₂ → y ∈ G₂ → (x, y) ∉ calls) ∧
    (∀ {x y : Fin n}, x ∈ G₃ → y ∈ G₃ → (x, y) ∉ calls) ∧ 
    (G₁ ∪ G₂ ∪ G₃ = Finset.univ : Finset (Fin n)) :=
sorry

end partition_three_groups_no_calls_l544_544355


namespace trees_after_planting_l544_544719

variable (x : ℕ)

theorem trees_after_planting (x : ℕ) : 
  let additional_trees := 12
  let days := 12 / 2
  let trees_removed := days * 3
  x + additional_trees - trees_removed = x - 6 :=
by
  let additional_trees := 12
  let days := 12 / 2
  let trees_removed := days * 3
  sorry

end trees_after_planting_l544_544719


namespace point_on_circle_x_value_l544_544681

/-
In the xy-plane, the segment with endpoints (-3,0) and (21,0) is the diameter of a circle.
If the point (x,12) is on the circle, then x = 9.
-/
theorem point_on_circle_x_value :
  let c := (9, 0) -- center of the circle
  let r := 12 -- radius of the circle
  let circle := {p | (p.1 - 9)^2 + p.2^2 = 144} -- equation of the circle
  ∀ x : Real, (x, 12) ∈ circle → x = 9 :=
by
  intros
  sorry

end point_on_circle_x_value_l544_544681


namespace speed_calculation_l544_544871

def distance := 600 -- in meters
def time := 2 -- in minutes

def distance_km := distance / 1000 -- converting meters to kilometers
def time_hr := time / 60 -- converting minutes to hours

theorem speed_calculation : (distance_km / time_hr = 18) :=
 by
  sorry

end speed_calculation_l544_544871


namespace non_empty_odd_subsets_count_l544_544649

theorem non_empty_odd_subsets_count : 
  let S := {1, 2, 3, 5, 7, 9} in
  (∀ x ∈ S, x % 2 = 1) →
  card (S℘.filter (λ T, T ≠ ∅)) = 31 :=
by
  let S := {1, 2, 3, 5, 7, 9};
  intros;
  sorry

end non_empty_odd_subsets_count_l544_544649


namespace other_root_of_quadratic_l544_544357

theorem other_root_of_quadratic (α β : ℝ) :
  (7 * α^2 - 3 * α - 10 = 0) → 
  (α = -2) → 
  (7 * β^2 - 3 * β - 10 = 0) ∧ 
  (α * β = -10 / 7) := 
by 
  intro h₁ h₂
  have hα : α = -2 := h₂
  have h_pro : 7 * (-2)^2 - 3 * (-2) - 10 = 0 := by linarith
  have h_eq : 7 * β^2 - 3 * β - 10 = 0 → α * β = -10 / 7 :=
  by
    intro hβ
    have vieta := calc
      α * β = -10 / 7 := by sorry -- Use Vieta's formula for proof step.
    exact ⟨hβ, vieta⟩
  exact h_eq

end other_root_of_quadratic_l544_544357


namespace optimize_profit_l544_544857

-- Given conditions are modeled in Lean.
def initial_daily_sales : ℕ := 20
def initial_profit_per_pot : ℕ := 40
def additional_pots_per_yuan_reduction : ℕ := 2

-- Mathematical expressions translated to Lean.
def profit_expression (x : ℕ) : ℕ :=
  (initial_profit_per_pot - x) * (initial_daily_sales + additional_pots_per_yuan_reduction * x)

-- Requirement: Finding maximum profit and corresponding reduction in price per pot.
theorem optimize_profit : 
  ∃ x : ℕ, 
    let y := profit_expression x in
    x = 15 ∧ y = 1250 := 
by
  sorry

end optimize_profit_l544_544857


namespace monotonicity_a_zero_extreme_points_a_less_neg2_no_extreme_points_minimum_point_l544_544254

noncomputable def f (x a : ℝ) : ℝ := x * (abs (x + a)) - (1 / 2) * log x

-- Problem 1: Monotonicity when a = 0
theorem monotonicity_a_zero :
  (∀ x, (1 / 2 < x → deriv (λ x, f x 0) x > 0) ∧ (0 < x ∧ x < 1 / 2 → deriv (λ x, f x 0) x < 0)) :=
sorry

-- Problem 2: Extreme points when a < 0

-- Case 1: a < -2
theorem extreme_points_a_less_neg2 {a : ℝ} (h : a < -2) :
  (∃ x_min x_max, x_min = (-a - sqrt (a^2 - 4)) / 4 ∧ 
                   x_max = (-a + sqrt (a^2 - 4)) / 4 ∧ 
                   is_min (λ x, f x a) x_min ∧ 
                   is_max (λ x, f x a) x_max) :=
sorry

-- Case 2: -2 ≤ a ≤ -sqrt(2)/2
theorem no_extreme_points {a : ℝ} (h : -2 ≤ a ∧ a ≤ -sqrt 2 / 2) :
  (∀ x, ¬is_min (λ x, f x a) x ∧ ¬is_max (λ x, f x a) x) :=
sorry

-- Case 3: -sqrt(2)/2 < a < 0
theorem minimum_point {a : ℝ} (h : -sqrt 2 / 2 < a ∧ a < 0) :
  (∃ x_min, x_min = (-a + sqrt (a^2 + 4)) / 4 ∧ is_min (λ x, f x a) x_min) :=
sorry

end monotonicity_a_zero_extreme_points_a_less_neg2_no_extreme_points_minimum_point_l544_544254


namespace magician_trick_l544_544510

def coin (a : Fin 11 → Bool) : Prop :=
  ∃ i : Fin 11, a i = a (i + 1) % 11

theorem magician_trick (a : Fin 11 → Bool) 
(H : ∃ i : Fin 11, a i = a (i + 1) % 11) : 
  ∃ j : Fin 11, j ≠ 0 ∧ a j = a 0 :=
sorry

end magician_trick_l544_544510


namespace geometric_sequence_from_second_term_l544_544968

theorem geometric_sequence_from_second_term
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (hS1 : S 1 = 1)
  (hS2 : S 2 = 2)
  (h_recurrence : ∀ (n : ℕ), 2 ≤ n → S (n+1) - 3 * S n + 2 * S (n-1) = 0)
  (haa : ∀ (n : ℕ), S (n+1) - S n = a (n+1)) 
  (h_a_def : ∀ (n : ℕ), S n = ∑ i in finset.range (n + 1), a i) :
  ∀ (n : ℕ), 2 ≤ n → a (n+1) = 2 * a n :=
by
  assume n hn
  sorry

end geometric_sequence_from_second_term_l544_544968


namespace question1_question2_l544_544599

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {y | y ≤ 9}
def C (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

theorem question1 : 
  (A = {x : ℝ | 3 ≤ x ∧ x < 7}) ∧ 
  ((set.univ \ A) ∩ B = {x : ℝ | x < 3 ∨ (7 ≤ x ∧ x ≤ 9)}) :=
sorry

theorem question2 (a : ℝ) : 
  A ∪ C a = C a → (a ≥ 7 ∨ a + 1 < 3) :=
sorry

end question1_question2_l544_544599


namespace arcsin_one_half_eq_pi_six_l544_544158

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := 
by
  sorry

end arcsin_one_half_eq_pi_six_l544_544158


namespace moon_arrangements_l544_544925

theorem moon_arrangements :
  (∃ (MOON : Finset (List Char)), 
    {w : List Char |
      w ∈ MOON ∧ w = ['M', 'O', 'O', 'N']}.card = 12) :=
sorry

end moon_arrangements_l544_544925


namespace tenth_term_geometric_sequence_l544_544910

theorem tenth_term_geometric_sequence :
  let a := (8 : ℚ)
  let r := (-2 / 3 : ℚ)
  a * r^9 = -4096 / 19683 :=
by
  sorry

end tenth_term_geometric_sequence_l544_544910


namespace find_a_l544_544627

theorem find_a (a : ℝ) (α : ℝ) (h1 : ∃ (y : ℝ), (a, y) = (a, -2))
(h2 : Real.tan (π + α) = 1 / 3) : a = -6 :=
sorry

end find_a_l544_544627


namespace father_age_l544_544856

theorem father_age : 
  ∀ (S F : ℕ), (S - 5 = 11) ∧ (F - S = S) → F = 32 := 
by
  intros S F h
  -- Use the conditions to derive further equations and steps
  sorry

end father_age_l544_544856


namespace head_ninth_flip_probability_l544_544569

noncomputable def fair_coin_flip (n : ℕ) : Probability := 
  if n % 2 = 0 then 1 / 2 else 1 / 2

theorem head_ninth_flip_probability :
  P(fair_coin_flip 9 = 1 / 2) :=
sorry

end head_ninth_flip_probability_l544_544569


namespace length_of_PB_l544_544513

theorem length_of_PB 
  (AB BC : ℝ) 
  (PA PD PC PB : ℝ)
  (h1 : AB = 2 * BC) 
  (h2 : PA = 5) 
  (h3 : PD = 12) 
  (h4 : PC = 13) 
  (h5 : PA^2 + PB^2 = (AB^2 + BC^2) / 5) -- derived from question
  (h6 : PB^2 = ((2 * BC)^2) - PA^2) : 
  PB = 10.5 :=
by 
  -- We would insert proof steps here (not required as per instructions)
  sorry

end length_of_PB_l544_544513


namespace find_range_of_z1z2_l544_544240

noncomputable def complex_modulus (z : Complex) : ℝ := Complex.abs z

theorem find_range_of_z1z2 (z1 z2 : ℂ) 
  (h1 : complex_modulus z1 < 1) 
  (h2 : complex_modulus z2 < 1) 
  (h3 : (z1 + z2).im = 0) -- z1 + z2 is real
  (h4 : z1 + z2 + z1 * z2 = 0) : 
  z1 * z2 ∈ Set.Ioo (-(1/2) : ℝ) (1 : ℝ) :=
sorry

end find_range_of_z1z2_l544_544240


namespace basketball_probability_third_shot_l544_544467

theorem basketball_probability_third_shot
  (p1 : ℚ) (p2_given_made1 : ℚ) (p2_given_missed1 : ℚ) (p3_given_made2 : ℚ) (p3_given_missed2 : ℚ) :
  p1 = 2 / 3 → p2_given_made1 = 2 / 3 → p2_given_missed1 = 1 / 3 → p3_given_made2 = 2 / 3 → p3_given_missed2 = 2 / 3 →
  (p1 * p2_given_made1 * p3_given_made2 + p1 * p2_given_missed1 * p3_given_misseds2 + 
   (1 - p1) * p2_given_made1 * p3_given_made2 + (1 - p1) * p2_given_missed1 * p3_given_missed2) = 14 / 27 :=
by
  sorry

end basketball_probability_third_shot_l544_544467


namespace part_one_part_two_l544_544967

variable {a : Nat → ℝ}

theorem part_one
    (h₁: ∀ n, (a (n + 1) - 1) * (a n - 1) = (1 / 2) * (a n - a (n + 1)))
    (h₂: a 1 = 2) :
    ∃ (b : Nat → ℝ), (∀ n, b (n + 1) = b n + 2) ∧ b 1 = 1 :=
begin
    sorry
end

theorem part_two
    {a : ℕ → ℝ}
    {b : ℕ → ℝ}
    (h₁: ∀ n, (a (n + 1) - 1) * (a n - 1) = (1 / 2) * (a n - a (n + 1)))
    (h₂: a 1 = 2)
    (h₃: ∀ n, b n = 1 / (a n - 1))
    {c : ℕ → ℝ}
    (h₄: ∀ n, c n = Real.sqrt (2 / (b n + 1)))
    {T : ℕ → ℝ}
    (h₅: ∀ n, T n = ∑ i in Finset.range n, c (i + 1)):
    ∀ n, T n ≥ Real.sqrt n :=
begin
    sorry
end

end part_one_part_two_l544_544967


namespace division_quotient_less_dividend_l544_544084

theorem division_quotient_less_dividend
  (a1 : (6 : ℝ) > 0)
  (a2 : (5 / 7 : ℝ) > 0)
  (a3 : (3 / 8 : ℝ) > 0)
  (h1 : (3 / 5 : ℝ) < 1)
  (h2 : (5 / 4 : ℝ) > 1)
  (h3 : (5 / 12 : ℝ) < 1):
  (6 / (3 / 5) > 6) ∧ (5 / 7 / (5 / 4) < 5 / 7) ∧ (3 / 8 / (5 / 12) > 3 / 8) :=
by
  sorry

end division_quotient_less_dividend_l544_544084


namespace vasya_gift_choices_l544_544736

theorem vasya_gift_choices :
  let cars := 7
  let construction_sets := 5
  (cars * construction_sets + Nat.choose cars 2 + Nat.choose construction_sets 2) = 66 :=
by
  sorry

end vasya_gift_choices_l544_544736


namespace find_positive_integer_n_l544_544587

theorem find_positive_integer_n :
  (∃ (n : ℕ), 0 < n ∧ (n + 2)! + (n + 3)! = n! * 720) →
  (∃ (n : ℕ), 0 < n ∧ (n + 2)! + (n + 3)! = n! * 720 ∧ n = 24) :=
by
  sorry

end find_positive_integer_n_l544_544587


namespace sin_66_deg_approx_l544_544614

theorem sin_66_deg_approx (h : Real.cos (78 * Real.pi / 180) ≈ 0.20) : Real.sin (66 * Real.pi / 180) ≈ 0.92 :=
sorry

end sin_66_deg_approx_l544_544614


namespace conjugate_of_z_is_minus_one_minus_i_l544_544233

open Complex

-- Define the condition:
def z := (1 + I)^2 / (1 - I)

-- State the goal:
theorem conjugate_of_z_is_minus_one_minus_i :
  conj z = -1 - I :=
by
  -- Placeholder for proof
  sorry

end conjugate_of_z_is_minus_one_minus_i_l544_544233


namespace fixed_point_l544_544229

noncomputable def passes_through_fixed_point (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y = 2 + Real.logBase a (x + 2)

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : passes_through_fixed_point a (-1) 2 :=
by
  sorry

end fixed_point_l544_544229


namespace sum_not_fourteen_l544_544426

theorem sum_not_fourteen (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) 
  (hprod : a * b * c * d = 120) : a + b + c + d ≠ 14 :=
sorry

end sum_not_fourteen_l544_544426


namespace smallest_clock_equiv_square_l544_544732

theorem smallest_clock_equiv_square (h : ℕ) (h_gt_6 : h > 6) : 
  (∃ hmin, hmin = 9 ∧ hmin > 6 ∧ (hmin^2 - hmin) % 24 = 0) :=
begin
  sorry
end

end smallest_clock_equiv_square_l544_544732


namespace magician_trick_l544_544509

def coin (a : Fin 11 → Bool) : Prop :=
  ∃ i : Fin 11, a i = a (i + 1) % 11

theorem magician_trick (a : Fin 11 → Bool) 
(H : ∃ i : Fin 11, a i = a (i + 1) % 11) : 
  ∃ j : Fin 11, j ≠ 0 ∧ a j = a 0 :=
sorry

end magician_trick_l544_544509


namespace monotonicity_of_f_solve_inequality_l544_544334

noncomputable def f (x : ℝ) : ℝ := sorry

def f_defined : ∀ x > 0, ∃ y, f y = f x := sorry

axiom functional_eq : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y 

axiom f_gt_zero : ∀ x, x > 1 → f x > 0

theorem monotonicity_of_f : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 < f x2 :=
sorry

theorem solve_inequality (x : ℝ) (h1 : f 2 = 1) (h2 : 0 < x) : 
  f x + f (x - 3) ≤ 2 ↔ 3 < x ∧ x ≤ 4 :=
sorry

end monotonicity_of_f_solve_inequality_l544_544334


namespace nearest_integer_to_power_six_l544_544065

noncomputable def nearest_integer (x : ℝ) : ℤ := 
if x - real.floor x < real.ceil x - x then real.floor x else real.ceil x

theorem nearest_integer_to_power_six : 
  let x := 3 + real.sqrt 5 in
  let y := 3 - real.sqrt 5 in
  nearest_integer (x^6) = 2654 :=
by
  sorry

end nearest_integer_to_power_six_l544_544065


namespace basketball_team_girls_l544_544845

theorem basketball_team_girls (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : B + (1 / 3) * G = 18) : 
  G = 18 :=
by
  have h3 : G - (1 / 3) * G = 30 - 18 := by sorry
  have h4 : (2 / 3) * G = 12 := by sorry
  have h5 : G = 12 * (3 / 2) := by sorry
  have h6 : G = 18 := by sorry
  exact h6

end basketball_team_girls_l544_544845


namespace sales_on_same_date_in_July_l544_544469

theorem sales_on_same_date_in_July :
  let bookstore_sale_days := [1, 5, 10, 15, 20, 25, 30]
  let shoe_store_sale_days := [1, 8, 15, 22, 29]
  (list.inter bookstore_sale_days shoe_store_sale_days).length = 2 :=
by
  let bookstore_sale_days := [1, 5, 10, 15, 20, 25, 30]
  let shoe_store_sale_days := [1, 8, 15, 22, 29]
  have h : list.inter bookstore_sale_days shoe_store_sale_days = [1, 15] := by sorry
  rw h
  simp

end sales_on_same_date_in_July_l544_544469


namespace collinear_points_inequalities_l544_544742

theorem collinear_points_inequalities
  (A B C D : Type) [linear_order A]
  (x y z : ℝ)
  (hAB : A < B)
  (hBC : B < C)
  (hCD : C < D)
  (hAB_length : x = dist A B)
  (hAC_length : y = dist A C)
  (hAD_length : z = dist A D)
  (hTriangle : ∀ B C, ∃ ΔABC : triangle AB BC CA, ΔABC.figure)
  : x < z / 2 ∧ y < x + z / 2 :=
by {
  sorry
}

end collinear_points_inequalities_l544_544742


namespace general_term_formula_l544_544969

noncomputable def sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n, n ≥ 1 → S n = 2 * a n + (-1) ^ n

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  sequence a S →
  a 1 = 1 ∧ a 2 = 0 ∧ a 3 = 2 ∧
  (∀ n, a n = (2^(n-2) + (-1)^(n-1)) / 3) ∧
  (∀ n, n > 4 → (∑ k in (range (n - 4)).map (λ i, k + 4 + 1:1/a (k + 4))) < 7 / 8) :=
by
  sorry

end general_term_formula_l544_544969


namespace min_value_of_slopes_l544_544225

-- Defining the existing hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

-- Defining the conditions of the problem
variables (A B P : ℝ × ℝ)
variables (O : ℝ × ℝ := (0, 0))

-- A and B are distinct points on the hyperbola
def is_on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

-- P also lies on the hyperbola
def condition_P_on_hyperbola : Prop := is_on_hyperbola P

-- The vector condition given in the problem
def vector_condition : Prop :=
  let PA := (P.1 - A.1, P.2 - A.2)
  let PB := (P.1 - B.1, P.2 - B.2)
  let PO := (P.1 - O.1, P.2 - O.2)
  PA.1 + PB.1 = 2 * PO.1 ∧ PA.2 + PB.2 = 2 * PO.2

-- The slopes of lines PA and PB
def slope_P_A : ℝ := (P.2 - A.2) / (P.1 - A.1)
def slope_P_B : ℝ := (P.2 + A.2) / (P.1 + A.1)

-- The minimum value we need to prove
theorem min_value_of_slopes :
  is_on_hyperbola A → is_on_hyperbola B → condition_P_on_hyperbola →
  vector_condition →
  ∃ (m n : ℝ), m = slope_P_A ∧ n = slope_P_B ∧ m^2 + n^2 / 9 = 8 / 3 :=
by
  sorry

end min_value_of_slopes_l544_544225


namespace homes_distance_is_65_l544_544348

noncomputable def distance_between_homes
  (maxwell_speed : ℕ)
  (brad_speed : ℕ)
  (maxwell_distance : ℕ)
  (time : ℕ) : ℕ :=
  maxwell_distance + brad_speed * time

theorem homes_distance_is_65
  (maxwell_speed : ℕ := 2)
  (brad_speed : ℕ := 3)
  (maxwell_distance : ℕ := 26)
  (time : ℕ := maxwell_distance / maxwell_speed) :
  distance_between_homes maxwell_speed brad_speed maxwell_distance time = 65 :=
by 
  sorry

end homes_distance_is_65_l544_544348


namespace wheel_distance_l544_544878

noncomputable def diameter : ℝ := 9
noncomputable def revolutions : ℝ := 18.683651804670912
noncomputable def pi_approx : ℝ := 3.14159
noncomputable def circumference (d : ℝ) : ℝ := pi_approx * d
noncomputable def distance (r : ℝ) (c : ℝ) : ℝ := r * c

theorem wheel_distance : distance revolutions (circumference diameter) = 528.219 :=
by
  unfold distance circumference diameter revolutions pi_approx
  -- Here we would perform the calculation and show that the result is approximately 528.219
  sorry

end wheel_distance_l544_544878


namespace games_bought_at_garage_sale_l544_544725

theorem games_bought_at_garage_sale (G : ℕ)
  (h1 : 2 + G - 2  = 2) :
  G = 2 :=
by {
  sorry
}

end games_bought_at_garage_sale_l544_544725


namespace find_w_over_y_l544_544246

theorem find_w_over_y 
  (w x y : ℝ) 
  (h1 : w / x = 2 / 3) 
  (h2 : (x + y) / y = 1.6) : 
  w / y = 0.4 := 
  sorry

end find_w_over_y_l544_544246


namespace a_eq_b_if_b2_ab_1_divides_a2_ab_1_l544_544712

theorem a_eq_b_if_b2_ab_1_divides_a2_ab_1 (a b : ℕ) (ha_pos : a > 0) (hb_pos : b > 0)
  (h : b^2 + a * b + 1 ∣ a^2 + a * b + 1) : a = b :=
by
  sorry

end a_eq_b_if_b2_ab_1_divides_a2_ab_1_l544_544712


namespace min_value_of_fraction_sum_l544_544329

theorem min_value_of_fraction_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 3 / 2 :=
sorry

end min_value_of_fraction_sum_l544_544329


namespace fibonacci_120_mod_5_l544_544431

noncomputable def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem fibonacci_120_mod_5 : fibonacci 120 % 5 = 0 :=
by sorry

end fibonacci_120_mod_5_l544_544431


namespace percent_democrats_l544_544665

/-- The percentage of registered voters in the city who are democrats and republicans -/
def D : ℝ := sorry -- Percent of democrats
def R : ℝ := sorry -- Percent of republicans

-- Given conditions
axiom H1 : D + R = 100
axiom H2 : 0.65 * D + 0.20 * R = 47

-- Statement to prove
theorem percent_democrats : D = 60 :=
by
  sorry

end percent_democrats_l544_544665


namespace no_exact_purchase_l544_544291

noncomputable def złoty_to_grosz : ℕ := 1001

noncomputable def total_grosz (złoty : ℕ) : ℕ := złoty * złoty_to_grosz

noncomputable def item_price_grosz : ℕ := 1987

theorem no_exact_purchase (złoty_amount : ℕ) :
  złoty_amount = 1986 →
  ¬ (total_grosz złoty_amount % item_price_grosz = 0) :=
by {
  intros h,
  have h₁ : total_grosz 1986 = 1986001 := rfl,
  rw [h, h₁],
  norm_num,
  exact dec_trivial,
}

end no_exact_purchase_l544_544291


namespace partnership_total_profit_l544_544842

theorem partnership_total_profit
  {a b t u : ℝ}
  (h1 : a = 3 * b)
  (h2 : t = 2 * u)
  (B_profit : B_received = 3000)
  (total_profit : (b * u) * 6 = total_profit):
  total_profit = 21000 := sorry

end partnership_total_profit_l544_544842


namespace seven_positive_integers_divide_12n_l544_544594

theorem seven_positive_integers_divide_12n (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, k = 7 ∧ n = some (List.nth [1, 2, 3, 5, 7, 11, 23] k)) → 
  (12 * n) % ((n * (n + 1)) / 2) = 0 := 
by
  sorry

end seven_positive_integers_divide_12n_l544_544594


namespace find_cos_angle_AGF_l544_544340

variables (A B C D E F G : Type*) [metric_space A] [metric_space B]

-- Define points A, B, C as elements of the metric space
variables (A B C : A)
-- Define specific distances between points
variable (hAB : dist A B = 13)
variable (hBC : dist B C = 14)
variable (hCA : dist C A = 15)

-- Define point D and E such that triangles ACD and BCE are isosceles right
variable (D : A)
variable (hAngleACD : angle A C D = real.pi / 2)
variable (hOnSameSideAC : same_side_of_line C A D B)

variable (E : A)
variable (hAngleBCE : angle B C E = real.pi / 2)
variable (hOnSameSideBC : same_side_of_line B C E A)

-- Define points F and G as intersection points
variable (F : A)
variable (hIntersectF : intersect_lines (line_through B C) (line_through A D) = F)

variable (G : A)
variable (hIntersectG : intersect_lines (line_through A C) (line_through B E) = G)

-- The actual proof statement to be shown
theorem find_cos_angle_AGF :
  cos (angle A G F) = - (5 / 13) :=
sorry

end find_cos_angle_AGF_l544_544340


namespace desired_gain_percentage_l544_544546

theorem desired_gain_percentage (cp16 sp16 cp12881355932203391 sp12881355932203391 : ℝ) :
  sp16 = 1 →
  sp16 = 0.95 * cp16 →
  sp12881355932203391 = 1 →
  cp12881355932203391 = (12.881355932203391 / 16) * cp16 →
  (sp12881355932203391 - cp12881355932203391) / cp12881355932203391 * 100 = 18.75 :=
by sorry

end desired_gain_percentage_l544_544546


namespace remaining_meals_solution_l544_544482

-- Define the conditions
def total_meals_for_children := 90
def total_meals_for_adults := 70
def meals_per_adult : ℚ := total_meals_for_children / total_meals_for_adults
def adults_had_meal := 42
def total_children := 70
def total_adults := 55

-- The main statement to prove
theorem remaining_meals_solution :
  let
    children's_meal_consumed := (adults_had_meal : ℚ) * meals_per_adult,
    remaining_meals := (total_meals_for_children : ℕ) - (children's_meal_consumed : ℕ)
  in remaining_meals = 36 :=
  by 
    -- Introduce the definitions in the proof context to show they are correct.
    intros children's_meal_consumed remaining_meals
    sorry

end remaining_meals_solution_l544_544482


namespace first_number_difference_l544_544009

theorem first_number_difference
    (avg_with_errors : ℝ) (correct_avg : ℝ) (wrong_second_num correct_second_num : ℝ) 
    (sum_with_errors : ℝ) (sum_correct : ℝ) 
    (sum_difference : ℝ) (num_count : ℕ)
    (H_avg_with_errors : avg_with_errors = 40.2) 
    (H_correct_avg : correct_avg = 40)
    (H_wrong_second_num : wrong_second_num = 13)
    (H_correct_second_num : correct_second_num = 31)
    (H_sum_with_errors : sum_with_errors = avg_with_errors * num_count)
    (H_sum_correct : sum_correct = correct_avg * num_count)
    (H_num_count : num_count = 10)
    (H_sum_difference : sum_difference = H_sum_with_errors - H_sum_correct) : 
    ∃ d : ℝ, d = 20 :=
by
  sorry

end first_number_difference_l544_544009


namespace left_handed_like_jazz_l544_544036

theorem left_handed_like_jazz (total_members left_handed like_jazz right_handed_dislike_jazz : ℕ)
  (h1 : total_members = 20)
  (h2 : left_handed = 8)
  (h3 : like_jazz = 15)
  (h4 : right_handed_dislike_jazz = 2)
  (h5 : left_handed + (total_members - left_handed) = total_members) :
  ∃ x, x = 5 ∧ 
    left_handed + (left_handed - x) + (like_jazz - x) + right_handed_dislike_jazz = total_members := 
by
  use 5
  split
  · rfl
  · sorry

end left_handed_like_jazz_l544_544036


namespace optimal_well_position_is_median_l544_544741

-- Define positions of the houses
variables {x1 x2 x3 x4 x5 x6 : ℝ}

-- Ensure the positions are in increasing order
axiom house_positions : x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5 ∧ x5 < x6

-- Define the function that calculates the optimal position for the well
noncomputable def optimal_well_position : ℝ :=
  (x3 + x4) / 2

-- Prove that this is the optimal position
theorem optimal_well_position_is_median :
  ∃ x, x = optimal_well_position :=
begin
  use (x3 + x4) / 2,
  refl,
end

end optimal_well_position_is_median_l544_544741


namespace percentage_of_sales_not_books_or_magazines_l544_544758

theorem percentage_of_sales_not_books_or_magazines
  (books_percentage : ℕ) (magazines_percentage : ℕ)
  (books_eq : books_percentage = 45)
  (magazines_eq : magazines_percentage = 25) :
  100 - (books_percentage + magazines_percentage) = 30 :=
by
  rw [books_eq, magazines_eq]
  simp
  sorry

end percentage_of_sales_not_books_or_magazines_l544_544758


namespace part1_part2_l544_544615

variable {α : Type}
variables (α : Real) (sin cos tan : α → α)

-- Assuming the conditions given in the problem
axiom tan_alpha_eq_neg_two : tan α = -2
axiom cos_alpha_ne_zero : cos α ≠ 0

-- Theorem for part (1)
theorem part1 : (sin α - 3 * cos α) / (sin α + cos α) = 5 := sorry

-- Theorem for part (2)
theorem part2 : 1 / (sin α * cos α) = -5 / 2 := sorry

end part1_part2_l544_544615


namespace largest_m_inequality_l544_544944

theorem largest_m_inequality 
  (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_sum : a + b + c = 1) :
  (∃ m : ℝ, (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a + b + c = 1 → 
    10 * (a^3 + b^3 + c^3) - m * (a^5 + b^5 + c^5) ≥ 1) ∧ m = 9) :=
begin
  sorry
end

end largest_m_inequality_l544_544944


namespace magician_assistant_strategy_l544_544500

-- Define the possible states of a coin
inductive CoinState | heads | tails

-- Define the circle of coins
def coinCircle := Fin 11 → CoinState

-- The strategy ensures there are adjacent coins with the same state
theorem magician_assistant_strategy (c : coinCircle) :
  ∃ i : Fin 11, c i = c ((i + 1) % 11) := by sorry

end magician_assistant_strategy_l544_544500


namespace probability_and_expectation_l544_544664

theorem probability_and_expectation (n m k x : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) 
(h3 : k ≤ x) (h4 : x ≤ k + m) :
  let P_x := (Nat.choose (x-1) (k-1) * Nat.choose (m+n-x) (n-k)) / Nat.choose (m+n) n in
  (P_x = P(x_k = x)) ∧ (∑ x in (range (m+1)).map (λ y => y + k), (y + k) * P_x) = k * (m + n + 1) / (n + 1) :=
begin
  sorry,
end

end probability_and_expectation_l544_544664


namespace probability_above_parabola_l544_544756

theorem probability_above_parabola :
  let total_points := 81 in
  let valid_points := ∑ a in finset.range 10 \ 0, ∑ b in finset.range 10 \ 0, 
      if b > a * a * a / (a + 1) then 1 else 0 in
  valid_points = 19 ∧ 
  (valid_points : ℝ) / total_points = 19 / 81 := by
  sorry

end probability_above_parabola_l544_544756


namespace geom_seq_sum_problem_l544_544688

noncomputable def geom_sum_first_n_terms (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

noncomputable def geom_sum_specific_terms (a₃ q : ℕ) (n m : ℕ) : ℕ :=
  a₃ * ((1 - (q^m) ^ n) / (1 - q^m))

theorem geom_seq_sum_problem :
  ∀ (a₁ q S₈₇ : ℕ),
  q = 2 →
  S₈₇ = 140 →
  geom_sum_first_n_terms a₁ q 87 = S₈₇ →
  ∃ a₃, a₃ = ((q * q) * a₁) →
  geom_sum_specific_terms a₃ q 29 3 = 80 := 
by
  intros a₁ q S₈₇ hq₁ hS₈₇ hsum
  -- Further proof would go here
  sorry

end geom_seq_sum_problem_l544_544688


namespace problem_statement_l544_544996

open Real

noncomputable def f (ω varphi : ℝ) (x : ℝ) := 2 * sin (ω * x + varphi)

theorem problem_statement (ω varphi : ℝ) (x1 x2 : ℝ) (hω_pos : ω > 0) (hvarphi_abs : abs varphi < π / 2)
    (hf0 : f ω varphi 0 = -1) (hmonotonic : ∀ x y, π / 18 < x ∧ x < y ∧ y < π / 3 → f ω varphi x < f ω varphi y)
    (hshift : ∀ x, f ω varphi (x + π) = f ω varphi x)
    (hx1x2_interval : -17 * π / 12 < x1 ∧ x1 < -2 * π / 3 ∧ -17 * π / 12 < x2 ∧ x2 < -2 * π / 3 ∧ x1 ≠ x2)
    (heq_fx : f ω varphi x1 = f ω varphi x2) :
    f ω varphi (x1 + x2) = -1 :=
sorry

end problem_statement_l544_544996


namespace longest_line_segment_squared_l544_544475

theorem longest_line_segment_squared (d : ℝ) (n : ℕ) (h1 : d = 10) (h2 : n = 4) : 
  let r := d / 2
  let θ := (2 * Real.pi) / n in
  let l := 2 * r * Real.sin (θ / 2) in
  l^2 = 50 :=
by
  intros
  sorry

end longest_line_segment_squared_l544_544475


namespace number_of_officers_l544_544010

theorem number_of_officers :
  ∀ (O N : ℕ) (avg_salary total_salary officer_salary non_officer_salary : ℝ),
  avg_salary = 120 →
  officer_salary = 430 →
  non_officer_salary = 110 →
  N = 465 →
  total_salary = avg_salary * (O + N) →
  total_salary = officer_salary * O + non_officer_salary * N →
  O = 15 :=
by
  intros O N avg_salary total_salary officer_salary non_officer_salary
  assume h_avg_sal h_officer_sal h_non_officer_sal h_n h_total_salary_avg h_total_salary_sum
  sorry

end number_of_officers_l544_544010


namespace locus_of_P_tangent_to_fixed_circle_l544_544455

-- Definitions and conditions
def circle_omega (x y : ℝ) : Prop := x^2 + y^2 = 2
def point_E : ℝ × ℝ := (1, 1)
def point_F : ℝ × ℝ := (1, -1)
def point_D (t : ℝ) : ℝ × ℝ := (sqrt 2 * Real.cos t, sqrt 2 * Real.sin t)
def triangle_ABC (E F : ℝ × ℝ) (D : ℝ × ℝ) : Prop := 
  let A := (2, 0) in
  -- Tangents and intersections should be defined
  -- Here we assume the process derived these points
  let B := (sorry, sorry) in 
  let C := (sorry, sorry) in
  true -- Placeholder for triangle definition

def circumcenter_P (A B C : ℝ × ℝ) : ℝ × ℝ := sorry -- Compute coordinates of P

noncomputable def fixed_circle_center_G : ℝ × ℝ := (-2, 0)
def fixed_circle_radius : ℝ := 2 * sqrt 2

-- Proof statements
theorem locus_of_P : 
  ∀ (t : ℝ),
  let D := point_D t in
  let A := (2, 0) in
  let B := (sorry, sorry) in
  let C := (sorry, sorry) in
  let P := circumcenter_P A B C in
  circle_omega P.1 P.2 →
  P.1^2 - P.2^2 = 2 :=
sorry

theorem tangent_to_fixed_circle :
  ∀ (t : ℝ),
  let D := point_D t in
  let A := (2, 0) in
  let B := (sorry, sorry) in
  let C := (sorry, sorry) in
  let P := circumcenter_P A B C in
  circle_omega P.1 P.2 →
  let circumcircle := sorry in
  let fixed_circle := sorry in
  -- Assume circumcircle and fixed circle definition
  -- Circumcircle is calculated based on points A, B, C
  -- Fixed circle with center G and radius = 2√2
  tangent circumcircle fixed_circle :=
sorry

end locus_of_P_tangent_to_fixed_circle_l544_544455


namespace find_int_k_l544_544427

theorem find_int_k (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 1500) (h3 : K > 1) (h4 : Z = K^3) :
  K = 11 :=
by
  sorry

end find_int_k_l544_544427


namespace find_a_l544_544211

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then (2 * a - 1) * x + 4 * a else a / x

theorem find_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x2 - f a x1) / (x2 - x1) < 0) ↔ (1 / 5 ≤ a ∧ a < 1 / 2) :=
by
  sorry

end find_a_l544_544211


namespace sum_first_10_terms_l544_544215

def sequence (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (3 * n - 2)

theorem sum_first_10_terms :
  (∑ i in Finset.range 10, sequence (i + 1)) = 15 := 
by
  sorry

end sum_first_10_terms_l544_544215


namespace sufficient_not_necessary_condition_parallel_lines_l544_544088

theorem sufficient_not_necessary_condition_parallel_lines :
  ∀ (a : ℝ), (a = 1/2 → (∀ x y : ℝ, x + 2*a*y = 1 ↔ (x - x + 1) ≠ 0) 
            ∧ ((∃ a', a' ≠ 1/2 ∧ (∀ x y : ℝ, x + 2*a'*y = 1 ↔ (x - x + 1) ≠ 0)) → (a ≠ 1/2))) :=
by
  intro a
  sorry

end sufficient_not_necessary_condition_parallel_lines_l544_544088


namespace election_votes_l544_544292

theorem election_votes (percent_A percent_valid : ℝ) (total_votes num_votes_for_A : ℕ) 
  (h_percent_A : percent_A = 0.65) (h_percent_valid : percent_valid = 0.85) 
  (h_total_votes : total_votes = 560000):
  num_votes_for_A = percent_A * (percent_valid * total_votes) := 
begin
  sorry
end

end election_votes_l544_544292


namespace sin_cos_15_degrees_proof_l544_544099

noncomputable
def sin_cos_15_degrees : Prop := (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4)

theorem sin_cos_15_degrees_proof : sin_cos_15_degrees :=
by
  sorry

end sin_cos_15_degrees_proof_l544_544099


namespace time_in_1867_minutes_correct_l544_544388

def current_time := (3, 15) -- (hours, minutes)
def minutes_in_hour := 60
def total_minutes := 1867
def hours_after := total_minutes / minutes_in_hour
def remainder_minutes := total_minutes % minutes_in_hour
def result_time := ((current_time.1 + hours_after) % 24, current_time.2 + remainder_minutes)
def expected_time := (22, 22) -- 10:22 p.m. in 24-hour format

theorem time_in_1867_minutes_correct : result_time = expected_time := 
by
    -- No proof is required according to the instructions.
    sorry

end time_in_1867_minutes_correct_l544_544388


namespace equal_angles_l544_544397

open EuclideanGeometry

variables {A B C D M : Point}

-- Given conditions
axiom mid_of_ab (h_mid: midpoint M A B)
axiom C_D_on_halfline (ha: is_halfline_from M C) (hb: is_halfline_from M D)
axiom equal_length (h_equal: dist A C = dist B D)

-- The theorem to prove
theorem equal_angles (h_mid : midpoint M A B) (ha: is_halfline_from M C) (hb: is_halfline_from M D) (h_equal: dist A C = dist B D) : 
  angle A C M = angle B D M := 
sorry

end equal_angles_l544_544397


namespace nearest_integer_to_power_six_l544_544068

noncomputable def nearest_integer (x : ℝ) : ℤ := 
if x - real.floor x < real.ceil x - x then real.floor x else real.ceil x

theorem nearest_integer_to_power_six : 
  let x := 3 + real.sqrt 5 in
  let y := 3 - real.sqrt 5 in
  nearest_integer (x^6) = 2654 :=
by
  sorry

end nearest_integer_to_power_six_l544_544068


namespace greatest_x_value_l544_544280

theorem greatest_x_value (x : ℤ) (h : 6.1 * 10 ^ x < 620) : x ≤ 2 := sorry

example : ∃ x : ℤ, 6.1 * 10 ^ x < 620 ∧ x = 2 := 
by {
  existsi (2 : ℤ),
  show 6.1 * 10 ^ 2 < 620,
  norm_num,
  split,
  { linarith, },
  { refl, }
}

end greatest_x_value_l544_544280


namespace find_angle_between_vectors_l544_544620

variable (a b : ℝ)
variables (vector_a vector_b : EuclideanSpace ℝ (Fin 2))
variable (theta : ℝ)

-- Condition definitions for Lean
def vector_a_norm_one : Prop := ∥vector_a∥ = 1
def vector_b_norm_sqrt_two : Prop := ∥vector_b∥ = Real.sqrt 2
def orthogonal : Prop := (vector_a - vector_b) ⬝ vector_a = 0

-- The theorem we want to prove
theorem find_angle_between_vectors :
  vector_a_norm_one vector_a →
  vector_b_norm_sqrt_two vector_b →
  orthogonal vector_a vector_b →
  theta = Real.arccos ((vector_a ⬝ vector_b) / (∥vector_a∥ * ∥vector_b∥)) →
  theta = π / 4 := 
sorry

end find_angle_between_vectors_l544_544620


namespace magician_trick_l544_544494

theorem magician_trick (coins : Fin 11 → Bool) 
  (exists_pair : ∃ i : Fin 11, coins i = coins ((i + 1) % 11)) 
  (assistant_left_uncovered : Fin 11) 
  (uncovered_coin_same_face : coins assistant_left_uncovered = coins (some (exists_pair))) :
  ∃ j : Fin 11, j ≠ assistant_left_uncovered ∧ coins j = coins assistant_left_uncovered := 
  sorry

end magician_trick_l544_544494


namespace sequence_an_sum_sequence_Tn_l544_544214

theorem sequence_an (k c : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n, S n = k * c ^ n - k) (ha2 : a 2 = 4) (ha6 : a 6 = 8 * a 3) :
  ∀ n, a n = 2 ^ n :=
by
  -- Proof is assumed to be given
  sorry

theorem sum_sequence_Tn (a : ℕ → ℝ) (T : ℕ → ℝ)
  (ha : ∀ n, a n = 2 ^ n) :
  ∀ n, T n = (n - 1) * 2 ^ (n + 1) + 2 :=
by
  -- Proof is assumed to be given
  sorry

end sequence_an_sum_sequence_Tn_l544_544214


namespace prob_two_hearts_l544_544874

theorem prob_two_hearts (cards : Finset ℕ)
  (deck_size : cards.cardinality = 52)
  (hearts : Finset ℕ)
  (hearts_size : hearts.cardinality = 13)
  (subset_hearts : hearts ⊆ cards) :
  (13 / 52) * (12 / 51) = 1 / 17 :=
by
  sorry

end prob_two_hearts_l544_544874


namespace problem_statement_l544_544080

theorem problem_statement : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end problem_statement_l544_544080


namespace marbles_in_container_l544_544476

theorem marbles_in_container (V1 V2 : ℝ) (m1 : ℕ) (ratio : V1 / m1 = V2 / 225) : m1 = 75 ∧ V1 = 24 ∧ V2 = 72 → (V2 / 225 = V1 / 75) :=
by
  intros h
  cases h with h1 h2
  cases h2 with hV1 hV2
  rw [h1, hV1, hV2] at ratio
  exact ratio

end marbles_in_container_l544_544476


namespace nearest_integer_is_11304_l544_544073

def nearest_integer_to_a_plus_b_pow_six (a b : ℝ) (h : b = Real.sqrt 5) : ℝ :=
  (a + b) ^ 6

theorem nearest_integer_is_11304 : nearest_integer_to_a_plus_b_pow_six 3 (Real.sqrt 5) rfl = 11304 := 
  sorry

end nearest_integer_is_11304_l544_544073


namespace work_problem_l544_544460

variable (r_A r_B r_C : ℝ)

theorem work_problem : 
  r_A + r_B = 1/3 ∧
  r_B + r_C = 1/6 ∧
  r_A + r_C = 2/9 →
  1 / r_B = 7.2 :=
begin
  sorry,
end

end work_problem_l544_544460


namespace smallest_four_digit_int_l544_544432

theorem smallest_four_digit_int (n : ℕ) (h: 1000 ≤ n ∧ n ≤ 9999) 
  (Sn := n * (n + 1) / 2) : 
  (∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (Sn % 12 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m < n) → ((m * (m + 1) / 2) % 12 ≠ 0))) := 
by 
  use 1001
  split
  · exact ⟨by linarith, by linarith⟩
  · split
    · sorry
    · intros m hm
      sorry

end smallest_four_digit_int_l544_544432


namespace triangle_area_correct_l544_544428

def line1 (x : ℝ) : ℝ := 8
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define the intersection points
def intersection1 : ℝ × ℝ := (6, line1 6)
def intersection2 : ℝ × ℝ := (-6, line1 (-6))
def intersection3 : ℝ × ℝ := (0, line2 0)

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_correct :
  triangle_area intersection1 intersection2 intersection3 = 36 :=
by
  sorry

end triangle_area_correct_l544_544428


namespace determine_S6_l544_544100

noncomputable def x_from_condition : ℝ := sorry

def S (m : ℕ) (x : ℝ) : ℝ := x^m + 1/x^m

theorem determine_S6 (h : x_from_condition + 1 / x_from_condition = 5) : S 6 x_from_condition = 12196 := 
sorry

end determine_S6_l544_544100


namespace chord_half_radius_equal_intercepts_tangent_l544_544247

-- Part (1)
theorem chord_half_radius (m : ℚ) (h : (∃ (x : ℚ) (y : ℚ), x^2 + y^2 + 2 * x - 4 * y + m = 0 ∧ (radius C)^2 = 5 - m) : 
  (radius C)/2 = radius C :=
  m = 11/15
:= 
sorry

-- Part (2)
theorem equal_intercepts_tangent (m : ℚ) (h : m = 3) (hT : (∃ a b : ℚ, a^2 + b^2 + 2 * a - 4 * b + 3 = 0 
  ∧ tangent_eq a b x + tangent_eq a b y = 0)
:= 
  (y = (2 + sqrt 6) * x) ∨ (y = (2 - sqrt 6) * x) ∨ (x + y + 1 = 0) ∨ (x + y - 3 = 0)
:= 
sorry

end chord_half_radius_equal_intercepts_tangent_l544_544247


namespace triangle_similarity_l544_544677

-- Definition of circle with relevant points
variables {Ω : Type} [circle Ω]
variables {A D R M N G : Ω}
variables {BC AD : segment Ω}
variables {E F : point Ω}

-- Given conditions
-- chord BC intersects chord AD at point R
-- EF is the perpendicular bisector of AD, intersecting at M
-- N is on AD between A and M
-- line EN extended meets the circle at G

def chord_intersects (BC AD : segment Ω) (R : Ω) : Prop := 
  midpoint R (chord_intersection BC AD)

def perpendicular_bisector (EF : segment Ω) (AD : segment Ω) (E F M : Ω) : Prop :=
  is_perpendicular_bisector EF AD ∧ midpoint E F M

def point_between (N : Ω) (A D : Ω) (M : Ω) : Prop := between A N M

-- The proof statement: triangle ENM is similar to triangle EFG
theorem triangle_similarity 
  (hc : chord_intersects BC AD R)
  (hefb : perpendicular_bisector EF AD E F M)
  (hpb : point_between N A D M)
  (HEG : extend EN G ⊆ Ω) :
  similar (triangle ENM) (triangle EFG) :=
sorry

end triangle_similarity_l544_544677


namespace slope_of_tangent_line_at_P_equation_of_tangent_line_at_P_l544_544248

noncomputable def slope_at_point : ℝ × ℝ → ℝ → ℝ
| (x, y), a := deriv (λ x, (1/3) * x^3) a

theorem slope_of_tangent_line_at_P (x : ℝ) (y : ℝ) (P : ℝ × ℝ) (hP : P = (2, 8/3)) : 
  slope_at_point P (fst P) = 4 :=
by 
  simp [slope_at_point]
  have P_def : 2 = fst P := by simp [hP]
  rw P_def
  simp
  sorry

noncomputable def tangent_line_eq : ℝ × ℝ → (ℝ → ℝ)
| (x₁, y₁) := λ x, 4 * (x - x₁) + y₁

theorem equation_of_tangent_line_at_P (x : ℝ) (y : ℝ) (P : ℝ × ℝ) (hP : P = (2, 8/3)) : 
  ∃ (m b : ℝ), ∀ x, 4 * x - y = m * x + b →  m = 12 ∧ b = -16 := 
by 
  simp [tangent_line_eq]
  have P_def : 2 = fst P ∧ 8/3 = snd P := by simp [hP]
  use [12, -16]
  sorry

end slope_of_tangent_line_at_P_equation_of_tangent_line_at_P_l544_544248


namespace nearest_integer_3_add_sqrt_5_pow_6_l544_544059

noncomputable def approx (x : ℝ) : ℕ := Real.floor (x + 0.5)

theorem nearest_integer_3_add_sqrt_5_pow_6 : 
  approx ((3 + Real.sqrt 5)^6) = 22608 :=
by
  -- Proof omitted, sorry
  sorry

end nearest_integer_3_add_sqrt_5_pow_6_l544_544059


namespace inverse_function_of_log_l544_544201

theorem inverse_function_of_log :
  ∀ x, x ≥ 0 → (∃ y, f y = x → y = (λ x, sqrt (2^x - 2)) x ∧ x ≥ 1) := by
  sorry

def f : ℝ → ℝ := λ x, log 2 (x^2 + 2)

end inverse_function_of_log_l544_544201


namespace volume_of_larger_solid_l544_544568

noncomputable def larger_solid_volume (A P Q : ℝ × ℝ × ℝ) := 
  let edge_length := 2 in 
  let volume := 28 / 3 in 
  (A = (0,0,0)) ∧ 
  (P = (2, 2, 1)) ∧ 
  (Q = (0, 2, 1)) →
  larger_solid_volume A P Q = volume

theorem volume_of_larger_solid :
  ∀ (A P Q : ℝ × ℝ × ℝ),
    A = (0,0,0) ∧ P = (2, 2, 1) ∧ Q = (0, 2, 1) → larger_solid_volume A P Q = 28 / 3 :=
begin
  intros,
  sorry
end

end volume_of_larger_solid_l544_544568


namespace vertex_angle_isosceles_triangle_l544_544795

theorem vertex_angle_isosceles_triangle (base_angle : ℝ) (h1 : base_angle = 34) :
  let vertex_angle := 180 - 2 * base_angle in
  vertex_angle = 112 :=
by
  sorry

end vertex_angle_isosceles_triangle_l544_544795


namespace modulus_of_z_a_b_values_l544_544210
open Complex

noncomputable def z : ℂ := ((1 + I)^2 + 3*(1 - I)) / (2 + I)

axiom z_eq : z = 1 - I

theorem modulus_of_z : |z| = Real.sqrt 2 := by
  -- proof to be provided
  sorry

theorem a_b_values (a b : ℝ) (h1 : z^2 + a * z + b = 1 + I) : a = -3 ∧ b = 4 := by
  -- proof to be provided
  sorry

end modulus_of_z_a_b_values_l544_544210


namespace kevin_started_with_cards_l544_544698

-- The definitions corresponding to the conditions in the problem
def ended_with : Nat := 54
def found_cards : Nat := 47
def started_with (ended_with found_cards : Nat) : Nat := ended_with - found_cards

-- The Lean statement for the proof problem itself
theorem kevin_started_with_cards : started_with ended_with found_cards = 7 := by
  sorry

end kevin_started_with_cards_l544_544698


namespace treasure_code_conditions_l544_544901

-- Define the 8-digit number code and conditions
def code : ℕ

-- Define the conditions for the code
theorem treasure_code_conditions :
  (code ≥ 20000000 ∧ code < 30000000) ∧ -- condition (3)
  (code % 3 = 0) ∧ (code % 25 = 0) ∧ -- condition (2)
  (((code / 10000000) % 10) = ((code / 100000) % 10)) ∧ -- condition (4)
  (((code / 1000) % 10) = ((code / 10000) % 10) - 2) ∧ -- condition (5)
  (( (((code / 100000) % 10) * 100 + ((code / 10000) % 10) * 10 + ((code / 1000) % 10)) /
     ((code / 10000000) * 10 + ((code / 1000000) % 10))) = 25) -- condition (6)
  → ∃ (code : ℕ), code ≥ 20000000 ∧ code < 30000000.
sorry  -- Proof is omitted

end treasure_code_conditions_l544_544901


namespace ratio_bound_l544_544702

-- Define the objects in our problem
variables {A B C M P D : Point}
variable TriangleABC : Triangle A B C

-- Define conditions given
def midpoint (M A B : Point) := dist A M = dist M B
def on_segment (P A B : Point) : Prop := dist A P + dist P B = dist A B
def between (P M B : Point) : Prop := dist M B > dist M P

-- Define areas of triangles and the ratio
noncomputable def area (T : Triangle) : ℝ := sorry
noncomputable def ratio_area (T1 T2 : Triangle) : ℝ := (area T1) / (area T2)

-- Prove ratio r satisfies the condition
theorem ratio_bound (MidpointM: midpoint M A B) 
    (PointBetween: between P M B) 
    (line_PD : Prop) :
    0 < ratio_area (Triangle P B D) (TriangleABC) ∧ 
    ratio_area (Triangle P B D) (TriangleABC) < 1 / 2 :=
sorry

end ratio_bound_l544_544702


namespace pete_age_is_35_l544_544740

-- Define variables
variable (P : ℕ)

-- Define conditions
axiom son_age : ℕ := 9
axiom in_4_years : P + 4 = 3 * (son_age + 4)

-- Theorem stating that Pete's current age is 35
theorem pete_age_is_35 (h1 : P + 4 = 3 * (son_age + 4)) : P = 35 := 
by
  sorry

end pete_age_is_35_l544_544740


namespace value_of_M_l544_544879

theorem value_of_M (M : ℕ) (h1 : 9 < M / 4) (h2 : M / 4 < 9.5) : M = 37 :=
sorry

end value_of_M_l544_544879


namespace perimeter_difference_l544_544384

theorem perimeter_difference :
  ∃ (a b c d : ℕ), a * b = 2015 ∧ c * d = 2015 ∧ 
  (2 * (a + b) = 4032) ∧ (2 * (c + d) = 192) ∧
  (2 * (a + b) - 2 * (c + d) = 3840) :=
begin
  sorry
end

end perimeter_difference_l544_544384


namespace max_min_value_x_eq_1_l544_544773

noncomputable def f (x k : ℝ) : ℝ := x^2 - 2 * (2 * k - 1) * x + 3 * k^2 - 2 * k + 6

theorem max_min_value_x_eq_1 :
  ∀ (k : ℝ), (∀ x : ℝ, ∃ m : ℝ, f x k = m → k = 1 → m = 6) → (∃ x : ℝ, x = 1) :=
by
  sorry

end max_min_value_x_eq_1_l544_544773


namespace min_fraction_value_l544_544218

theorem min_fraction_value 
    (a : ℕ → ℝ) 
    (S : ℕ → ℝ) 
    (d : ℝ) 
    (n : ℕ) 
    (h1 : ∀ {n}, a n = 5 + (n - 1) * d)
    (h2 : (a 2) * (a 10) = (a 4 - 1)^2) 
    (h3 : S n = (n * (a 1 + a n)) / 2)
    (h4 : a 1 = 5)
    (h5 : d > 0) :
    2 * S n + n + 32 ≥ (20 / 3) * (a n + 1) := sorry

end min_fraction_value_l544_544218


namespace pipes_fill_cistern_l544_544096

-- Define the conditions as constants
def P1_rate : ℝ := 1 / 10   -- Rate of first pipe
def P2_rate : ℝ := 1 / 12   -- Rate of second pipe
def P3_rate : ℝ := -1 / 20  -- Rate of third pipe (negative because it empties)

-- Define the statement to prove the time to fill the cistern
theorem pipes_fill_cistern : 
  let combined_rate := P1_rate + P2_rate + P3_rate in
  (1 / combined_rate) = 7.5 :=
by 
  -- This is the statement, the proof is not required.
  sorry

end pipes_fill_cistern_l544_544096


namespace largest_n_triangle_property_l544_544565

def has_triangle_property (s : Set ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a < b ∧ b < c ∧ c < a + b

def max_n_triangle_property : ℕ := 2206

theorem largest_n_triangle_property :
  ∀ S : Set ℕ, (∀ t : Set ℕ, t ⊆ S ∧ t.card = 15 → has_triangle_property t) ↔ 
              (∀ n ≤ 2206, S = {k : ℕ | 3 ≤ k ∧ k ≤ n}) :=
by
  sorry

end largest_n_triangle_property_l544_544565


namespace triangle_must_be_isosceles_l544_544877

def is_fibonacci (n : ℕ) : Prop :=
  n ∈ {2, 3, 5, 8, 13, 21, 34, 55, 89, 144}

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem triangle_must_be_isosceles (a b c : ℕ) (h1 : is_triangle a b c) (h2 : a ≠ b ∨ b ≠ c ∨ a ≠ c) :
  is_fibonacci a → is_fibonacci b → is_fibonacci c → is_isosceles a b c :=
by
  sorry

end triangle_must_be_isosceles_l544_544877


namespace first_positive_term_is_7_l544_544992

-- Define the conditions and the sequence
def a1 : ℚ := -1
def d : ℚ := 1 / 5

-- Define the general term of the sequence
def a_n (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Define the proposition that the 7th term is the first positive term
theorem first_positive_term_is_7 :
  ∀ n : ℕ, (0 < a_n n) → (7 <= n) :=
by
  intro n h
  sorry

end first_positive_term_is_7_l544_544992


namespace factor_product_l544_544079

theorem factor_product : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end factor_product_l544_544079


namespace ellipse_and_line_properties_l544_544987

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

noncomputable def line_equation (k : ℝ) : Prop :=
  ∀ (x y : ℝ), y = k * x + 3 / 2

theorem ellipse_and_line_properties :
  (∃ a b : ℝ, a = sqrt 3 ∧ b = 1 ∧ ellipse_equation a b) ∧ 
  (∃ k : ℝ, (k = sqrt 6 / 3 ∨ k = -sqrt 6 / 3) ∧ line_equation k) :=
by sorry

end ellipse_and_line_properties_l544_544987


namespace modulus_of_power_of_complex_l544_544579

theorem modulus_of_power_of_complex (z : ℂ) (n : ℕ) : 
  |(2 + 1*I)^8| = 625 :=
by
  sorry

end modulus_of_power_of_complex_l544_544579


namespace sum_of_lengths_geq_one_div_k_l544_544343

-- Given conditions:
def non_overlapping_segments (M : Set (Set ℝ)) (s : Fin k → Set ℝ) : Prop :=
  ∀ i j : Fin k, i < j → s i ∩ s j = ∅ ∧ s i ∈ M ∧ s j ∈ M

def endpoints_in_M (M : Set (Set ℝ)) (s : Fin k → Set ℝ) : Prop :=
  ∀ (l : ℝ), l ≤ 1 → ∃ (i : Fin k), ∃ (x y : ℝ), x ∈ s i ∧ y ∈ s i ∧ y - x = l

-- Proof problem:
theorem sum_of_lengths_geq_one_div_k
  (M : Set (Set ℝ)) (s : Fin k → Set ℝ)
  (h_non_overlap : non_overlapping_segments M s)
  (h_endpoints_in_M : endpoints_in_M M s) :
  ∑ i, (s i).end - (s i).start ≥ 1 / k :=
by
  sorry

end sum_of_lengths_geq_one_div_k_l544_544343


namespace particle_path_count_l544_544512

def lattice_path_count (n : ℕ) : ℕ :=
sorry -- Placeholder for the actual combinatorial function

theorem particle_path_count : lattice_path_count 7 = sorry :=
sorry -- Placeholder for the actual count

end particle_path_count_l544_544512


namespace perpendicular_distance_from_D_to_plane_ABC_is_six_l544_544907

def point (ℝ × ℝ × ℝ) := ℝ

/-- Vertices definitions --/
def A : point := (0, 0, 0)
def B : point := (5, 0, 0)
def C : point := (0, 3, 0)
def D : point := (0, 0, 6)

/-- Function to determine the perpendicular distance from point D to the plane ABC --/
noncomputable def distance_from_point_to_plane (D A B C : point) : ℝ := 
    sorry -- Define the function by using vector and plane equations

theorem perpendicular_distance_from_D_to_plane_ABC_is_six :
  distance_from_point_to_plane D A B C = 6 :=
sorry

end perpendicular_distance_from_D_to_plane_ABC_is_six_l544_544907


namespace PropA_is_nec_but_not_suff_condition_l544_544203

-- Definitions for the conditions
variable {f : ℝ → ℝ} [Differentiable ℝ f]

def PropositionA (x : ℝ) : Prop := deriv f x = 0

def PropositionB (x : ℝ) : Prop := 
  ∃ ε > 0, ∀ y, y ≠ x → abs (y - x) < ε → (f y > f x ∨ f y < f x)

-- Statement that PropositionA is a necessary but not sufficient condition for PropositionB
theorem PropA_is_nec_but_not_suff_condition (x : ℝ) : 
  (PropositionB x → PropositionA x) ∧ ¬ (PropositionA x → PropositionB x) := sorry

end PropA_is_nec_but_not_suff_condition_l544_544203


namespace giraffes_difference_l544_544039

theorem giraffes_difference :
  ∃ n : ℕ, (300 = 3 * n) ∧ (300 - n = 200) :=
by
  sorry

end giraffes_difference_l544_544039


namespace sixth_individual_is_nineteen_l544_544965

def random_table : list (list string) :=
[["1818", "0792", "4544", "1716", "5809", "7983", "8619"],
 ["6206", "7650", "0310", "5523", "6405", "0526", "6238"]]

def select_individuals (table : list (list string)) : list ℕ :=
let digits := (table.head!.drop 2 ++ table.tail!.head!).map (λ s, s.to_list.to_chunks 2) in
digits.join.map (λ chunk, chunk.as_string.to_nat).filter (λ n, n < 20)

theorem sixth_individual_is_nineteen : 
  select_individuals random_table = [18, 7, 17, 16, 9, 19] → 
  (select_individuals random_table).nth 5 = some 19 :=
by
  intro h
  sorry

end sixth_individual_is_nineteen_l544_544965


namespace train_crossing_bridge_time_l544_544522

theorem train_crossing_bridge_time :
  ∀ (train_length bridge_and_train_length : ℝ) (train_speed_km_per_hr : ℝ),
  train_length = 180 →
  bridge_and_train_length = 195 →
  train_speed_km_per_hr = 45 →
  let train_speed_m_per_s := train_speed_km_per_hr * (1000 / 3600) in
  let time_to_cross := bridge_and_train_length / train_speed_m_per_s in
  time_to_cross = 15.6 :=
by
  sorry

end train_crossing_bridge_time_l544_544522


namespace line_through_point_equal_intercepts_l544_544017

theorem line_through_point_equal_intercepts (a b : ℝ) : 
  ((∃ (k : ℝ), k ≠ 0 ∧ (3 = 2 * k) ∧ b = k) ∨ ((a ≠ 0) ∧ (5/a = 1))) → 
  (a = 1 ∧ b = 1) ∨ (3 * a - 2 * b = 0) := 
by 
  sorry

end line_through_point_equal_intercepts_l544_544017


namespace total_rocks_needed_l544_544351

def rocks_already_has : ℕ := 64
def rocks_needed : ℕ := 61

theorem total_rocks_needed : rocks_already_has + rocks_needed = 125 :=
by
  sorry

end total_rocks_needed_l544_544351


namespace sum_prime_factors_245490_l544_544434

theorem sum_prime_factors_245490 : 
  let n := 245490
  let prime_factors := [2, 5, 3, 101]
  let largest_prime_factor := 101
  ∑ pf in prime_factors.erase_dup, pf + largest_prime_factor = 212 :=
by
  let n := 245490
  let prime_factors := [2, 5, 3, 101]
  let largest_prime_factor := 101
  have h : prime_factors.erase_dup = [2, 5, 3, 101] := sorry
  have hsum : ∑ pf in prime_factors.erase_dup, pf = 111 := sorry
  have hlargest : largest_prime_factor = 101 := sorry
  have hadd : 111 + 101 = 212 := rfl
  exact hadd

end sum_prime_factors_245490_l544_544434


namespace four_divides_n_l544_544171

theorem four_divides_n (n : ℕ) (x : ℕ → ℤ) 
  (hx1 : ∀ i, x i = 1 ∨ x i = -1)
  (hx2 : ∑ i in Finset.range n, x i * x (i + 1) * x (i + 2) * x (i + 3) = 0)
  (hx3 : ∀ i, x (n + i) = x i) :
  4 ∣ n := 
sorry

end four_divides_n_l544_544171


namespace find_m_of_perpendicular_l544_544262

variables (m : ℝ)
def vector_a := (2, 1)
def vector_b := (m, -2)

theorem find_m_of_perpendicular (h : vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 0) : m = 1 :=
by {
  sorry
}

end find_m_of_perpendicular_l544_544262


namespace missing_village_population_l544_544025

theorem missing_village_population : 
  let village_populations := [803, 900, 1100, 945, 980, 1249]
  let average_population := 1000
  ∑ (pop : Nat) in village_populations, pop = 5977 →
  (average_population * 7) - ∑ (pop : Nat) in village_populations, pop = 1023 := by
  intros village_populations average_population pop_sum_eq
  have pop_sum : ∑ (pop : Nat) in village_populations, pop = 5977 := pop_sum_eq
  have total_sum : average_population * 7 = 7000 := rfl
  have missing_pop := 7000 - 5977
  show missing_pop = 1023  
  sorry

end missing_village_population_l544_544025


namespace calculate_expression_l544_544896

theorem calculate_expression : 103^3 - 3 * 103^2 + 3 * 103 - 1 = 1061208 := by
  sorry

end calculate_expression_l544_544896


namespace sin_inequality_in_triangle_l544_544283

theorem sin_inequality_in_triangle (A B C : ℝ) (hA_leq_B : A ≤ B) (hB_leq_C : B ≤ C)
  (hSum : A + B + C = π) (hA_pos : 0 < A) (hB_pos : 0 < B) (hC_pos : 0 < C)
  (hA_lt_pi : A < π) (hB_lt_pi : B < π) (hC_lt_pi : C < π) :
  0 < Real.sin A + Real.sin B - Real.sin C ∧ Real.sin A + Real.sin B - Real.sin C ≤ Real.sqrt 3 / 2 := 
sorry

end sin_inequality_in_triangle_l544_544283


namespace triangle_properties_l544_544282

noncomputable def tan_A := 1/4
noncomputable def tan_B := 3/5
noncomputable def shortest_side := Real.sqrt 2
noncomputable def angle_C := 3 * Real.pi / 4
noncomputable def area_triangle := 3/2

theorem triangle_properties
  (A B C : ℝ) -- Angles
  (a b c : ℝ) -- Sides
  (h₁ : tan A = tan_A)
  (h₂ : tan B = tan_B)
  (h₃ : a = shortest_side) :
  C = angle_C ∧ (1 / 2 * a * c * sin B = area_triangle) :=
by 
  sorry

end triangle_properties_l544_544282


namespace line_MN_fixed_point_through_max_triangle_area_l544_544562

def ellipse := {p : ℝ × ℝ // 3 * p.1 ^ 2 + 2 * p.2 ^ 2 = 6}
noncomputable def focusF : ℝ × ℝ := (sqrt(1), 0)
noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem line_MN_fixed_point_through {M N : ℝ × ℝ} (hM : ∀ (A B : ellipse), M = midpoint A.1 B.1)
    (hN : ∀ (C D : ellipse), N = midpoint C.1 D.1)
    (hPerp : ∀ (A B C D : ellipse), (B.2 - A.2) * (D.2 - C.2) + (B.1 - A.1) * (D.1 - C.1) = 0) :
    ∀ (M N : ℝ × ℝ), lineThrough M N (3 / 5, 0) :=
sorry

theorem max_triangle_area {M N : ℝ × ℝ} (hM : ∀ (A B : ellipse), M = midpoint A.1 B.1)
    (hN : ∀ (C D : ellipse), N = midpoint C.1 D.1)
    (hPerp : ∀ (A B C D : ellipse), (B.2 - A.2) * (D.2 - C.2) + (B.1 - A.1) * (D.1 - C.1) = 0) :
    ∃ (area : ℝ), area = 4 / 25 :=
sorry

end line_MN_fixed_point_through_max_triangle_area_l544_544562


namespace total_pixels_l544_544867

theorem total_pixels (width height dpi : ℕ) (h_width : width = 21) (h_height : height = 12) (h_dpi : dpi = 100) :
  width * dpi * height * dpi = 2520000 := 
by
  rw [h_width, h_height, h_dpi]
  simp
  sorry

end total_pixels_l544_544867


namespace holidays_fraction_is_correct_l544_544543

-- Define the total bonus
def total_bonus : ℝ := 1496

-- Define the allocations
def kitchen_allocation : ℝ := total_bonus / 22
def christmas_gifts_allocation : ℝ := total_bonus / 8

-- Define the remaining amount after expenses
def remaining_amount : ℝ := 867

-- Define the fraction for holidays allocation
def holidays_allocation_fraction : ℝ := 187 / 748

-- Statement to prove
theorem holidays_fraction_is_correct : 
  (total_bonus - (kitchen_allocation + christmas_gifts_allocation + remaining_amount)) / total_bonus = holidays_allocation_fraction :=
by
  sorry

end holidays_fraction_is_correct_l544_544543


namespace polynomial_form_l544_544221

variable {R : Type*} [Semiring R] (f : R → R) (c k a : ℕ)
variable (a_seq : ℕ → ℕ)

-- Condition 1: f(x) is a polynomial with non-negative integral coefficients
variable [IsPolynomial f] (coeff_non_neg : ∀ i, 0 ≤ Polynomial.coeff f i)

-- Condition 2: a is a positive integer
variable (a_pos : 0 < a)

-- Condition 3: The sequence {a_n} is defined by a₁ = a, a_{n+1} = f(a_n)
def seq_update (n : ℕ) : ℕ := if n = 0 then a else f (seq_update (n - 1))

-- Condition 4: The set of primes dividing at least one term of this sequence is finite
variable (finite_primes : ∃ S : Set ℕ, (∀ n, ∃ p ∈ S, Prime p ∧ p ∣ seq_update n) ∧ S.Finite)

-- Theorem: f(x) = cx^k for some non-negative integers c and k
theorem polynomial_form :
  ∃ c k : ℕ, ∀ n, f n = c * (n ^ k) :=
sorry

end polynomial_form_l544_544221


namespace min_Sn_div_2n_l544_544976

def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) :=
  ∑ i in finset.range n, a i

def is_geometric_sequence {α : Type*} [field α] (a b c : α) :=
  b * b = a * c

theorem min_Sn_div_2n :
  ∀ (a : ℕ → ℤ) (S : ℕ → ℤ),
  (a 1 = 1) →
  (is_arithmetic_sequence a) →
  (is_geometric_sequence (a 1 + d) (a 1 + 2 * d) (a 1 + 5 * d)) →
  (∀ n, S n = sum_first_n_terms a n) →
  ∃ (n : ℕ), S n / 2 ^ n = -1 / 2 :=
by
  sorry

end min_Sn_div_2n_l544_544976


namespace circle_diameter_point_x_l544_544683

-- Define the endpoints of the circle's diameter
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (21, 0)

-- Define the point (x, 12)
def P_x : ℝ → ℝ × ℝ := λ x, (x, 12)

-- Mathematical statement to prove: for a point on the circle with diameter endpoints (-3,0) and (21,0), 
--   and y-coordinate 12, the x-coordinate must be 9.
theorem circle_diameter_point_x (x : ℝ) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in  -- center of the circle
  let r := (Mathlib.Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) / 2 in  -- radius of the circle
  let D := (9, 0) in  -- center is (9, 0)
  let R := 12 in  -- radius is 12
  (P_x x).fst = 9 := by
  -- stating that the point (x,12) lies on the circle with the center (9,0) and radius 12
  sorry

end circle_diameter_point_x_l544_544683


namespace arcsin_one_half_l544_544156

theorem arcsin_one_half : real.arcsin (1 / 2) = real.pi / 6 :=
by
  sorry

end arcsin_one_half_l544_544156


namespace boys_neither_happy_nor_sad_correct_l544_544453

def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def total_boys : ℕ := 16
def total_girls : ℕ := 44
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- The number of boys who are neither happy nor sad
def boys_neither_happy_nor_sad : ℕ :=
  total_boys - happy_boys - (sad_children - sad_girls)

theorem boys_neither_happy_nor_sad_correct : boys_neither_happy_nor_sad = 4 := by
  sorry

end boys_neither_happy_nor_sad_correct_l544_544453


namespace arithmetic_evaluation_l544_544939

theorem arithmetic_evaluation : (64 / 0.08) - 2.5 = 797.5 :=
by
  sorry

end arithmetic_evaluation_l544_544939


namespace base_rate_of_first_company_is_7_l544_544127

noncomputable def telephone_company_base_rate_proof : Prop :=
  ∃ (base_rate1 base_rate2 charge_per_minute1 charge_per_minute2 minutes : ℝ),
  base_rate1 = 7 ∧
  charge_per_minute1 = 0.25 ∧
  base_rate2 = 12 ∧
  charge_per_minute2 = 0.20 ∧
  minutes = 100 ∧
  (base_rate1 + charge_per_minute1 * minutes) =
  (base_rate2 + charge_per_minute2 * minutes) ∧
  base_rate1 = 7

theorem base_rate_of_first_company_is_7 :
  telephone_company_base_rate_proof :=
by
  -- The proof step will go here
  sorry

end base_rate_of_first_company_is_7_l544_544127


namespace median_of_consecutive_integers_l544_544832

theorem median_of_consecutive_integers (a n : ℤ) (N : ℕ) (h1 : (a + (n - 1)) + (a + (N - n)) = 110) : 
  (2 * a + N - 1) / 2 = 55 := 
by {
  -- The proof goes here.
  sorry
}

end median_of_consecutive_integers_l544_544832


namespace sin_addition_l544_544961

theorem sin_addition (α β : ℝ) 
  (h1: 0 < α ∧ α < π) 
  (h2: 0 < β ∧ β < π) 
  (h3: cos α = -1/2)
  (h4: sin β = sqrt 3 / 2) 
  : sin (α + β) = -3 / 4 := 
by
  sorry

end sin_addition_l544_544961


namespace find_speed_of_man_l544_544117

def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
(v_m + v_s = 6) ∧ (v_m - v_s = 8)

theorem find_speed_of_man :
  ∃ v_m v_s : ℝ, speed_of_man_in_still_water v_m v_s ∧ v_m = 7 :=
by
  sorry

end find_speed_of_man_l544_544117


namespace chess_tournament_games_l544_544465

def num_games (n : Nat) : Nat := n * (n - 1) * 2

theorem chess_tournament_games : num_games 7 = 84 :=
by
  sorry

end chess_tournament_games_l544_544465


namespace translated_function_equals_sin2x_l544_544422

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 3) + 2

noncomputable def translate_right (h : ℝ) (f : ℝ → ℝ) : ℝ → ℝ :=
  sorry -- We assume a translation function is correctly implemented

noncomputable def translate_down (k : ℝ) (f : ℝ → ℝ) : ℝ → ℝ :=
  sorry -- We assume a translation function is correctly implemented

theorem translated_function_equals_sin2x :
  let g := translate_down 2 (translate_right (π / 6) f)
  g = sin ∘ (λ x, 2 * x) :=
by
  sorry

end translated_function_equals_sin2x_l544_544422


namespace KM_plus_ML_eq_BC_l544_544676

variables {A B C L M K : Type*} [triangle A B C] [isosceles A B A C]
variables (hAngleA : ∠ A = 30)
variables (hALeqCM : AL = CM)
variables (hAngleAMK : ∠ AMK = 45)
variables (hAngleLMC : ∠ LMC = 75)

theorem KM_plus_ML_eq_BC
  (h1 : isosceles △ ABC)
  (h2 : ∠A = 30)
  (h3 : AL = CM)
  (h4 : ∠AMK = 45)
  (h5 : ∠LMC = 75) :
  KM + ML = BC :=
sorry

end KM_plus_ML_eq_BC_l544_544676


namespace picture_arrangements_l544_544578

theorem picture_arrangements :
  let F := 1
  let Mi := 2
  let M := 3
  let B := 4
  let units := {F, Mi, (M, B)}
  units.card = 3 → 
  (∀ (x ∈ units) (y ∈ units), x ≠ y) →
  (units.perm 3).card = 6 :=
by
  intros F Mi M B units h_card h_distinct
  exact sorry


end picture_arrangements_l544_544578


namespace modulus_of_complex_raised_to_eight_l544_544582

-- Define the complex number 2 + i in Lean
def z : Complex := Complex.mk 2 1

-- State the proof problem with conditions
theorem modulus_of_complex_raised_to_eight : Complex.abs (z ^ 8) = 625 := by
  sorry

end modulus_of_complex_raised_to_eight_l544_544582


namespace lucas_average_speed_palindrome_l544_544733

-- Define the theorem using conditions and the target average speed
theorem lucas_average_speed_palindrome : 
  ∀ (dist : ℕ) (t : ℕ), 
    dist = 100 → 
    t = 3 → 
    ∃ avg_speed : ℝ, avg_speed = dist / t ∧ avg_speed = 33.33 :=
begin
  sorry
end

end lucas_average_speed_palindrome_l544_544733


namespace great_eighteen_hockey_league_games_l544_544006

theorem great_eighteen_hockey_league_games :
  (let teams_per_division := 9
   let games_intra_division_per_team := 8 * 3
   let games_inter_division_per_team := teams_per_division * 2
   let total_games_per_team := games_intra_division_per_team + games_inter_division_per_team
   let total_game_instances := 18 * total_games_per_team
   let unique_games := total_game_instances / 2
   unique_games = 378) :=
by
  sorry

end great_eighteen_hockey_league_games_l544_544006


namespace kevin_initial_cards_l544_544696

theorem kevin_initial_cards (found_cards : ℕ) (total_cards : ℕ) (h1 : found_cards = 47) (h2 : total_cards = 54) : total_cards - found_cards = 7 := 
by
  rw [h1, h2]
  norm_num
  done

end kevin_initial_cards_l544_544696


namespace A_not_losing_prob_correct_l544_544441

def probability_draw : ℚ := 1 / 2
def probability_A_wins : ℚ := 1 / 3
def probability_A_not_losing : ℚ := 5 / 6

theorem A_not_losing_prob_correct : 
  probability_draw + probability_A_wins = probability_A_not_losing := 
by sorry

end A_not_losing_prob_correct_l544_544441


namespace cyclic_quadrilateral_tangents_concurrent_l544_544317

theorem cyclic_quadrilateral_tangents_concurrent 
  (A B C D K L M N : Point) 
  (h_cyclic : cyclic_quadrilateral A B C D)
  (h_KLMN_rhombus : rhombus K L M N ∧ parallel K L A C ∧ parallel L M B D)
  (h_K_on_AB : on_segment K A B) 
  (h_L_on_BC : on_segment L B C) 
  (h_M_on_CD : on_segment M C D) 
  (h_N_on_DA : on_segment N D A)
  (ω_A ω_B ω_C ω_D : Circle) 
  (h_ωA_incirc_AN_K : incircle ω_A (triangle A N K))
  (h_ωB_incirc_BK_L : incircle ω_B (triangle B K L))
  (h_ωC_incirc_CL_M : incircle ω_C (triangle C L M))
  (h_ωD_incirc_DM_N : incircle ω_D (triangle D M N)) :
  concurrent_tangents (common_internal_tangents ω_A ω_C) (common_internal_tangents ω_B ω_D) :=
sorry

end cyclic_quadrilateral_tangents_concurrent_l544_544317


namespace sin_sum_angle_eq_sqrt15_div5_l544_544202

variable {x : Real}
variable (h1 : 0 < x ∧ x < Real.pi) (h2 : Real.sin (2 * x) = 1 / 5)

theorem sin_sum_angle_eq_sqrt15_div5 : Real.sin (Real.pi / 4 + x) = Real.sqrt 15 / 5 := by
  -- The proof is omitted as instructed.
  sorry

end sin_sum_angle_eq_sqrt15_div5_l544_544202


namespace more_cost_effective_mall_l544_544839

def original_price : ℕ := 80

def mallA_discount (n : ℕ) : ℕ :=
  if n * 4 <= original_price / 2 then n * 4 else original_price / 2

def mallA_price (n : ℕ) : ℕ :=
  original_price - mallA_discount(n)

def mallB_price : ℕ := original_price * 70 / 100

theorem more_cost_effective_mall (n : ℕ) :
  (n < 6 → mallB_price < mallA_price n) ∧
  (n = 6 → mallB_price = mallA_price n) ∧
  (n > 6 → mallB_price > mallA_price n) :=
by
  sorry

end more_cost_effective_mall_l544_544839


namespace number_of_sets_l544_544031

def set_A {α : Type} (a b c d e : α) : set (set α) :=
  {A | {a, b} ⊆ A ∧ A ⊆ {a, b, c, d, e}}

theorem number_of_sets (α : Type) (a b c d e : α) :
  (set_A a b c d e).card = 8 := by
  sorry

end number_of_sets_l544_544031


namespace man_l544_544824

-- Define the speeds and values given in the problem conditions
def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

-- Define the man's speed in still water as a variable
def man_speed_in_still_water : ℝ := man_speed_with_current - speed_of_current

-- The theorem we need to prove
theorem man's_speed_against_current_is_correct :
  (man_speed_in_still_water - speed_of_current = man_speed_against_current) :=
by
  -- Placeholder for proof
  sorry

end man_l544_544824


namespace circle_radius_six_l544_544959

theorem circle_radius_six (d : ℝ) : d = 5 → ∃ r, r = 6 ∧ ( ∀ x y, x^2 - 8 * x + y^2 + 10 * y + d = 0 → (x - 4)^2 + (y + 5)^2 = r^2 ) :=
by 
  intro h
  rw h
  use 6
  split
  { refl }
  { intros x y
    simp
    sorry
  }

end circle_radius_six_l544_544959


namespace first_sales_amount_l544_544126

theorem first_sales_amount :
  ∃ S : ℝ, 
    (5 / S = real.rat_approx  five_over_s (5/S).to_rnlf :
    1.2000000000000001 * (5 / S)) ∧
    S = 15 :=
begin
  sorry
end

end first_sales_amount_l544_544126


namespace min_max_modulus_eq_one_l544_544320

open Complex Polynomial

noncomputable def min_max_modulus (n : ℕ) (h : 0 < n) : ℂ :=
let P := {p : Polynomial ℂ // p.degree = n ∧ p.monic} in
(min_value :=
  real.Inf {m | ∃ p ∈ P, m = ⟨max_norm_on_unit_circle p⟩})

theorem min_max_modulus_eq_one {n : ℕ} (h : 0 < n) :
  min_max_modulus n h = 1 := sorry

end min_max_modulus_eq_one_l544_544320


namespace distinct_pos_reals_are_ints_l544_544705

def floor (x : ℝ) : ℤ := Int.floor x

theorem distinct_pos_reals_are_ints (a b : ℝ) (h1 : a ≠ b) (h2 : 0 < a) (h3 : 0 < b)
  (h4 : ∀ (n : ℕ), 0 < n → floor (n * a) ∣ floor (n * b)) :
  ∃ (m : ℤ) (n : ℤ), a = m ∧ b = n :=
by
  sorry

end distinct_pos_reals_are_ints_l544_544705


namespace problem1_problem2_l544_544234

theorem problem1 (m : ℝ) (H : m > 0) (p : ∀ x : ℝ, (x+1)*(x-5) ≤ 0 → 1 - m ≤ x ∧ x ≤ 1 + m) : m ≥ 4 :=
sorry

theorem problem2 (x : ℝ) (m : ℝ) (H : m = 5) (disj : ∀ x : ℝ, ((x+1)*(x-5) ≤ 0 ∨ (1 - m ≤ x ∧ x ≤ 1 + m))
) (conj : ¬ ∃ x : ℝ, (x+1)*(x-5) ≤ 0 ∧ (1 - m ≤ x ∧ x ≤ 1 + m)) : (-4 ≤ x ∧ x < -1) ∨ (5 < x ∧ x < 6) :=
sorry

end problem1_problem2_l544_544234


namespace high_school_students_total_l544_544109

theorem high_school_students_total
    (students_taking_music : ℕ)
    (students_taking_art : ℕ)
    (students_taking_both_music_and_art : ℕ)
    (students_taking_neither : ℕ)
    (h1 : students_taking_music = 50)
    (h2 : students_taking_art = 20)
    (h3 : students_taking_both_music_and_art = 10)
    (h4 : students_taking_neither = 440) :
    students_taking_music - students_taking_both_music_and_art + students_taking_art - students_taking_both_music_and_art + students_taking_both_music_and_art + students_taking_neither = 500 :=
by
  sorry

end high_school_students_total_l544_544109


namespace minimum_value_is_81_l544_544331

noncomputable def minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) : ℝ :=
a^2 + 9 * a * b + 9 * b^2 + 3 * c^2

theorem minimum_value_is_81 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minimum_value a b c h1 h2 h3 h4 = 81 :=
sorry

end minimum_value_is_81_l544_544331


namespace hyperbola_equation_l544_544981

variable (O : Point) (C : Hyperbola) (P A B : Point)
variable (a b m : ℝ)
variable (PAOB_area : ℝ)
variables (b_gt_a : b > a) (a_pos : a > 0) (m_pos : m > 0)
variable (P_on_hyperbola : is_on_hyperbola P a b)
variable (P_projection_at_focus : is_focus_projection P x_axis focus_right)
variable (PAOB_area_one : PAOB_area = 1)

theorem hyperbola_equation :
  b = 2 ∧ a = 1 → (∃ b = 4, a = 1,
  standard_hyperbola_equation : ∀ x y, (x^2)/1^2 - (y^2)/b^2 = 1) :=
by
  sorry

end hyperbola_equation_l544_544981


namespace part_a_part_b_part_c_l544_544184

def tau (n : ℕ) : ℕ := ∑ d in (finset.divisors n), d
def phi (n : ℕ) : ℕ := (finset.range n).filter (λ x, nat.coprime x n).card

theorem part_a (n : ℕ) (hn : n > 1) : phi n * tau n < n ^ 2 :=
sorry

theorem part_b : ∀ n : ℕ, phi n * tau n + 1 = n ^ 2 → nat.prime n :=
sorry

theorem part_c : ¬ ∃ n : ℕ, phi n * tau n + 2023 = n ^ 2 :=
sorry

end part_a_part_b_part_c_l544_544184


namespace arrange_moon_l544_544921

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ways_to_arrange_moon : ℕ :=
  let total_letters := 4
  let repeated_O_count := 2
  factorial total_letters / factorial repeated_O_count

theorem arrange_moon : ways_to_arrange_moon = 12 := 
by {
  sorry -- Proof is omitted as instructed
}

end arrange_moon_l544_544921


namespace no_solution_range_has_solution_range_l544_544936

open Real

theorem no_solution_range (a : ℝ) : (∀ x, ¬ (|x - 4| + |3 - x| < a)) ↔ a ≤ 1 := 
sorry

theorem has_solution_range (a : ℝ) : (∃ x, |x - 4| + |3 - x| < a) ↔ 1 < a :=
sorry

end no_solution_range_has_solution_range_l544_544936


namespace part1_part2_l544_544143

-- Define the constants based on given conditions
def cost_price : ℕ := 5
def initial_selling_price : ℕ := 9
def initial_sales_volume : ℕ := 32
def price_increment : ℕ := 2
def sales_decrement : ℕ := 8

-- Part 1: Define the elements 
def selling_price_part1 : ℕ := 11
def profit_per_item_part1 : ℕ := 6
def daily_sales_volume_part1 : ℕ := 24

theorem part1 :
  (selling_price_part1 - cost_price = profit_per_item_part1) ∧ 
  (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price_part1 - initial_selling_price) = daily_sales_volume_part1) := 
by
  sorry

-- Part 2: Define the elements 
def target_daily_profit : ℕ := 140
def selling_price1_part2 : ℕ := 12
def selling_price2_part2 : ℕ := 10

theorem part2 :
  (((selling_price1_part2 - cost_price) *
    (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price1_part2 - initial_selling_price)) = target_daily_profit) ∨
  ((selling_price2_part2 - cost_price) *
    (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price2_part2 - initial_selling_price)) = target_daily_profit)) :=
by
  sorry

end part1_part2_l544_544143


namespace problem_l544_544632

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * a * x^2 - (x - 1) * Real.exp x

theorem problem (a : ℝ) :
  (∀ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ x3 ∧ x3 ≤ 1 →
                  f a x1 + f a x2 ≥ f a x3) →
  1 ≤ a ∧ a ≤ 4 :=
sorry

end problem_l544_544632


namespace harkamal_mangoes_l544_544264

theorem harkamal_mangoes (m : ℕ) (h1: 8 * 70 = 560) (h2 : m * 50 + 560 = 1010) : m = 9 :=
by
  sorry

end harkamal_mangoes_l544_544264


namespace minimize_expression_l544_544438

theorem minimize_expression : ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  exists 3
  split
  -- Here, you would prove the inequality and the fact that x = 3 gives the minimum.
  sorry

end minimize_expression_l544_544438


namespace large_seat_capacity_indeterminate_l544_544386

theorem large_seat_capacity_indeterminate 
  (small_seats : ℕ) 
  (large_seats : ℕ) 
  (people_per_small_seat : ℕ)
  (total_small_capacity : ℕ) 
  (h_small_seats : small_seats = 2)
  (h_large_seats : large_seats = 23)
  (h_people_per_small_seat : people_per_small_seat = 14)
  (h_total_small_capacity : total_small_capacity = 28) : 
  ∃ x, false := 
by 
  sorry

end large_seat_capacity_indeterminate_l544_544386


namespace hyperbola_eccentricity_l544_544680

theorem hyperbola_eccentricity:
  ∀ (x : ℝ), (y = 2 * x + 1 / x) →
  let C := hyperbola_rotated_around_origin y in
  let e := eccentricity_of C in
  e = Real.sqrt (10 - 4 * Real.sqrt 5) :=
by
  sorry

end hyperbola_eccentricity_l544_544680


namespace inequality_and_equality_condition_l544_544716

theorem inequality_and_equality_condition (x : ℝ) (hx : x > 0) : 
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) :=
  sorry

end inequality_and_equality_condition_l544_544716


namespace sarah_flour_total_l544_544370

def rye_flour : ℕ := 5
def whole_wheat_bread_flour : ℕ := 10
def chickpea_flour : ℕ := 3
def whole_wheat_pastry_flour : ℕ := 2

def total_flour : ℕ := rye_flour + whole_wheat_bread_flour + chickpea_flour + whole_wheat_pastry_flour

theorem sarah_flour_total : total_flour = 20 := by
  sorry

end sarah_flour_total_l544_544370


namespace number_of_left_handed_jazz_lovers_l544_544034

variable (total_members left_handed jazz_lovers rh_non_jazz_lovers left_handed_jazz_lovers : ℕ)

axiom club_conditions :
  total_members = 20 ∧
  left_handed = 8 ∧
  jazz_lovers = 15 ∧
  rh_non_jazz_lovers = 2

theorem number_of_left_handed_jazz_lovers :
  ∀ (total_members left_handed jazz_lovers rh_non_jazz_lovers : ℕ),
    total_members = 20 → 
    left_handed = 8 → 
    jazz_lovers = 15 → 
    rh_non_jazz_lovers = 2 → 
    (∃ x, x + (left_handed - x) + (jazz_lovers - x) + rh_non_jazz_lovers = total_members ∧ x = 5) :=
by {
  intros, 
  sorry
}

end number_of_left_handed_jazz_lovers_l544_544034


namespace set_representation_l544_544260

def is_Natural (n : ℕ) : Prop :=
  n ≠ 0

def condition (x : ℕ) : Prop :=
  x^2 - 3*x < 0

theorem set_representation :
  {x : ℕ | condition x ∧ is_Natural x} = {1, 2} := 
sorry

end set_representation_l544_544260


namespace deriv_frac_deriv_prod_l544_544102

-- Define and prove y' for y = (1 - x^2) / e^x
theorem deriv_frac (x : ℝ) : 
  ∃ y', deriv (λ x, (1 - x^2) / exp x) x = y' ∧ y' = (x^2 - 2 * x - 1) / exp x := 
by
  exists (((x^2) - 2*x - 1) / exp x)
  apply deriv_eq
  -- Proof steps skipped
  sorry

-- Define and prove y' for y = x^2 * sin (3x + π)
theorem deriv_prod (x : ℝ) : 
  ∃ y', deriv (λ x, x^2 * sin (3 * x + π)) x = y' ∧ y' = -2 * x * sin (3 * x) - 3 * x^2 * cos (3 * x) := 
by
  exists (-2 * x * sin (3 * x) - 3 * x^2 * cos (3 * x))
  apply deriv_eq
  -- Proof steps skipped
  sorry

end deriv_frac_deriv_prod_l544_544102


namespace expr1_eq_pi_expr2_eq_neg1_l544_544151

-- Define the first mathematical expression
def expr1 : ℝ :=
  real.sqrt ((3 - real.pi) ^ 4) + (0.008) ^ (-1/3) - (0.25) ^ (1/2) * (1 / real.sqrt 2) ^ (-4)

-- Prove that the first expression is equal to π
theorem expr1_eq_pi : expr1 = real.pi :=
by sorry

-- Define the second mathematical expression
def expr2 : ℝ :=
  real.logb 3 (real.sqrt 27) - real.logb 3 (real.sqrt 3) - real.log 625 - real.log 4 + real.ln (real.exp 2) - (4 / 3) * real.log (real.sqrt 8)

-- Prove that the second expression is equal to -1
theorem expr2_eq_neg1 : expr2 = -1 :=
by sorry

end expr1_eq_pi_expr2_eq_neg1_l544_544151


namespace circle_area_less_than_circumference_probability_l544_544803

noncomputable def probability_circle_area_less_than_circumference : ℚ :=
  let r : ℕ := 2 in
  let dice_sides := 8 in
  let total_possible_outcomes : ℕ := dice_sides * dice_sides in
  let favorable_outcomes : ℕ := 1 in
  favorable_outcomes / total_possible_outcomes

theorem circle_area_less_than_circumference_probability :
  probability_circle_area_less_than_circumference = 1 / 64 :=
by
  sorry

end circle_area_less_than_circumference_probability_l544_544803


namespace problem_statement_l544_544349

def g (x : ℕ) : ℕ := x^2 - 4 * x

theorem problem_statement :
  g (g (g (g (g (g 2))))) = L := sorry

end problem_statement_l544_544349


namespace abscissa_of_tangent_point_is_2_l544_544916

noncomputable def f (x : ℝ) : ℝ := (x^2) / 4 - 3 * Real.log x

noncomputable def f' (x : ℝ) : ℝ := (1/2) * x - 3 / x

theorem abscissa_of_tangent_point_is_2 : 
  ∃ x0 : ℝ, f' x0 = -1/2 ∧ x0 = 2 :=
by
  sorry

end abscissa_of_tangent_point_is_2_l544_544916


namespace solve_inequality_l544_544889

theorem solve_inequality (a x : ℝ) (h : a < 0) : 
    (ax^2 - (a-1)x - 1 < 0) ↔ 
    ((-1 < a ∧ a < 0 ∧ (x > - 1/a ∨ x < 1)) ∨
     (a = -1 ∧ x ≠ 1) ∨
     (a < -1 ∧ (x > 1 ∨ x < -1/a))) :=
sorry

end solve_inequality_l544_544889


namespace balloons_left_after_distribution_l544_544091

-- Definitions for the conditions
def red_balloons : ℕ := 23
def blue_balloons : ℕ := 39
def green_balloons : ℕ := 71
def yellow_balloons : ℕ := 89
def total_balloons : ℕ := red_balloons + blue_balloons + green_balloons + yellow_balloons
def number_of_friends : ℕ := 10

-- Statement to prove the correct answer
theorem balloons_left_after_distribution : total_balloons % number_of_friends = 2 :=
by
  -- The proof would go here
  sorry

end balloons_left_after_distribution_l544_544091


namespace water_parts_in_solution_l544_544518

def lemonade_syrup_parts : ℝ := 7
def target_percentage : ℝ := 0.30
def adjusted_parts : ℝ := 2.1428571428571423

-- Original equation: L = 0.30 * (L + W)
-- Substitute L = 7 for the particular instance.
-- Therefore, 7 = 0.30 * (7 + W)

theorem water_parts_in_solution (W : ℝ) : 
  (7 = 0.30 * (7 + W)) → 
  W = 16.333333333333332 := 
by
  sorry

end water_parts_in_solution_l544_544518


namespace sequence_count_zeros_ones_l544_544651

/--
The number of sequences of zeros and ones of length 10, with all zeros
at either end of the sequence, or all ones at either end of the sequence, or both, is 38.
-/

theorem sequence_count_zeros_ones : 
  let n := 10,
      seq_count := ∑ k in finset.range (n + 1), 2 * 1 - 2
  in seq_count = 38 :=
by
  sorry

end sequence_count_zeros_ones_l544_544651


namespace solve_for_daily_plough_rate_l544_544478

noncomputable def farmer_plough_rate (area days_overshoot leftover_area daily_plough actual_plough : ℝ) 
  (h_area : area = 3780)
  (h_leftover : leftover_area = 40)
  (h_daily_plough : daily_plough = 85)
  (h_overshoot : days_overshoot = 2)
  (h_actual_days : actual_plough = actual_plough) : Prop :=
  let x := (3780 / 90) in 3780 / daily_plough - days_overshoot = actual_plough ∧ actual_plough = (area - leftover_area) / daily_plough

theorem solve_for_daily_plough_rate : farmer_plough_rate 3780 2 40 85 42 := 
by 
  sorry

end solve_for_daily_plough_rate_l544_544478


namespace equal_triangle_areas_l544_544966

theorem equal_triangle_areas (A B C D : Point) (b c k l : ℝ) 
  (H_A : A = ⟨0, 0⟩) 
  (H_B : B = ⟨b, 0⟩)
  (H_C : C = ⟨b, c⟩)
  (H_D : D = ⟨0, c⟩)
  (H_K : K = ⟨b, k⟩)
  (H_L : L = ⟨l, c⟩)
  (H_ABK : area A B K = area A K L)
  (H_AKL : area A K L = area A D L) :
  k = (c * (Real.sqrt 5 - 1)) / 2 ∧ l = (b * (Real.sqrt 5 - 1)) / 2 :=
by
  sorry

end equal_triangle_areas_l544_544966


namespace tan_beta_minus_2alpha_l544_544227

variable {α β : ℝ}

-- Conditions
def condition1 : Prop :=
  (1 - Real.cos (2 * α)) / (Real.sin α * Real.cos α) = 1

def condition2 : Prop :=
  Real.tan (β - α) = -1 / 3

-- Final proof statement
theorem tan_beta_minus_2alpha (h1 : condition1) (h2 : condition2) :
  Real.tan (β - 2 * α) = -1 :=
sorry

end tan_beta_minus_2alpha_l544_544227


namespace arcsin_one_half_eq_pi_six_l544_544157

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := 
by
  sorry

end arcsin_one_half_eq_pi_six_l544_544157


namespace log_exp_equality_l544_544459

theorem log_exp_equality :
  log 2 + log 5 - 42 * 8^(1 / 4) - 2017^0 = -2 := by
  sorry

end log_exp_equality_l544_544459


namespace distance_between_cyclists_correct_l544_544801

noncomputable def distance_between_cyclists : ℝ :=
  if h : ∃ (a b : ℝ), a = b^2 ∧ a - 1 = 3 * (b - 1)
  then 
    let ⟨a, b, ha, hb⟩ := h in
    a - b
  else 
    0

theorem distance_between_cyclists_correct :
  (∃ (a b : ℝ), a = b^2 ∧ a - 1 = 3 * (b - 1)) →
  (distance_between_cyclists = 0 ∨ distance_between_cyclists = 2) :=
by
  sorry

end distance_between_cyclists_correct_l544_544801


namespace find_g_inv_f_10_l544_544002

variables (α β : Type) [Nonempty α] [Nonempty β]
variables (f : β → α) (g : α → β) (f_inv : α → β) (g_inv : β → α)

-- Assuming that f_inv is the inverse of f and g_inv is the inverse of g
axiom f_inv_def : ∀ x, f_inv (f x) = x
axiom g_inv_def : ∀ x, g_inv (g x) = x

-- Given condition
axiom condition : ∀ x, f_inv (g x) = 3 * x - 1

-- Statement to prove
theorem find_g_inv_f_10 : g_inv (f 10) = 11 / 3 :=
by
  sorry

end find_g_inv_f_10_l544_544002


namespace sugar_mixture_problem_l544_544470

theorem sugar_mixture_problem (x w b : ℕ) (hconditions : 320 - x + x = 320)
    (hY_ratio : w * (320 - x) = b * x)
    (hZ_ratio : 49 * (320 - (320 * x - x^2 // 320)) = 15 * (2 * x - x^2 // 320)) :
    x + w + b = 48 := 
sorry

end sugar_mixture_problem_l544_544470


namespace beth_wins_in_configuration_l544_544289

def nim_value (walls : List Nat) : Nat :=
  walls.foldr (λ n acc, n ⊕ acc) 0

theorem beth_wins_in_configuration :
  nim_value [7, 2, 1] = 0 :=
by 
  -- Calculation of nim value for each wall according to the new rules.
  -- Nim values are pre-calculated: 7 -> 3, 2 -> 2, 1 -> 1
  -- The result should satisfy nim_value [7, 2, 1] = 3 ⊕ 2 ⊕ 1 = 0
  sorry

end beth_wins_in_configuration_l544_544289


namespace incorrect_statement_c_l544_544090

-- Definitions based on conditions
variable (p q : Prop)

-- Lean 4 statement to check the logical proposition
theorem incorrect_statement_c (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
by
  sorry

end incorrect_statement_c_l544_544090


namespace arith_seq_general_term_sum_of_first_10_terms_l544_544608

open Nat

variable {a : ℕ → ℝ}  -- Defining the arithmetic sequence as a function ℕ → ℝ.

noncomputable def arith_seq_property (a : ℕ → ℝ) : Prop :=
  (a 3 = 1) ∧ (a 4 = real.sqrt (a 3 * a 7))

noncomputable def general_term (a : ℕ → ℝ) : ℕ → ℝ := fun n => 2 * n - 5

noncomputable def sum_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ :=
  fun n => (n * (2 * n - 10 + 5)) / 2

theorem arith_seq_general_term (a : ℕ → ℝ) (h : arith_seq_property a) :
  ∀ n, a n = general_term a n :=
sorry

theorem sum_of_first_10_terms (a : ℕ → ℝ) (h : arith_seq_property a) :
  sum_first_n_terms a 10 = 60 :=
sorry

end arith_seq_general_term_sum_of_first_10_terms_l544_544608


namespace parametric_convert_min_max_distances_l544_544991

-- Define the given polar equation of curve C
def polar_eq (ρ θ : ℝ) : Prop := ρ^2 - 4 * ρ * Real.sin θ + 3 = 0

-- Define the parametric equations of the curve
def parametric_eq (α : ℝ) : ℝ × ℝ := (Real.cos α, 2 + Real.sin α)

-- Cartesian coordinate representations of points A and B
def pointA : ℝ × ℝ := (-1, 0)
def pointB : ℝ × ℝ := (1, 0)

-- Define the distance square function
def dist_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the sum of squared distances for a point P on the curve
def sum_of_squares (α : ℝ) : ℝ :=
  let P := parametric_eq α in
  dist_squared pointA P + dist_squared pointB P

theorem parametric_convert (ρ θ : ℝ) (h : polar_eq ρ θ) : 
  ∃ α, parametric_eq α = (Real.cos α, 2 + Real.sin α) := sorry

theorem min_max_distances :
  ∃ α_min α_max, sum_of_squares α_min = 4 ∧ sum_of_squares α_max = 20 := sorry

end parametric_convert_min_max_distances_l544_544991


namespace salary_distribution_l544_544798

theorem salary_distribution (
  total_value : ℝ := 325500,
  first_team_workers : ℕ := 15,
  first_team_days : ℕ := 21,
  second_team_workers : ℕ := 14,
  second_team_days : ℕ := 25,
  third_team_days : ℕ := 20,
  workers_increase_percentage : ℝ := 0.4,
  ruble_per_day_per_worker : ℝ := 300
) :
  let third_team_workers := (1 + workers_increase_percentage) * first_team_workers
  let first_team_salary := first_team_workers * first_team_days * ruble_per_day_per_worker
  let second_team_salary := second_team_workers * second_team_days * ruble_per_day_per_worker
  let third_team_salary := third_team_workers * third_team_days * ruble_per_day_per_worker in
  first_team_salary + second_team_salary + third_team_salary = total_value ∧
  first_team_salary = 94500 ∧
  second_team_salary = 105000 ∧
  third_team_salary = 126000 :=
sorry

end salary_distribution_l544_544798


namespace inequality_proof_l544_544242

variables {a b c : ℝ}

theorem inequality_proof (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * c < b * c :=
by
  sorry

end inequality_proof_l544_544242


namespace angle_B_measure_max_area_triangle_l544_544305

theorem angle_B_measure (a c : ℝ) (h₁ : b = 4) (h₂ : ∃ (cosB cosC : ℝ), cosB / cosC = 4 / (2 * a - c)) :
  B = π / 3 :=
by
  -- sorry, skip proof
  sorry

theorem max_area_triangle (a c : ℝ) (h₁ : b = 4) (h₂ : ∃ (cosB cosC : ℝ), cosB / cosC = 4 / (2 * a - c)) :
  ∃ ac : ℝ, ac <= 16 ∧ S = 1/2 * ac * sin (π / 3) ∧ S = 4 * √3 :=
by
  -- sorry, skip proof
  sorry

end angle_B_measure_max_area_triangle_l544_544305


namespace percentage_previous_deadlift_l544_544146

-- Condition definitions
def deadlift_at_13 := 300
def weight_added_per_year := 110
def age_start := 13
def age_end := 18
def years := age_end - age_start
def total_added := weight_added_per_year * years
def deadlift_at_18 := deadlift_at_13 + total_added
def extra_weight := 100

-- Calculating the percentage
def percentage_of_previous_deadlift := 11.76

-- Statement to prove
theorem percentage_previous_deadlift :
  (deadlift_at_18 + extra_weight) = deadlift_at_18 * (1 + percentage_of_previous_deadlift / 100) :=
sorry

end percentage_previous_deadlift_l544_544146


namespace total_running_time_l544_544365

-- Defining the conditions
def rudy_runs_first_part : ℕ := 5 * 10
def rudy_runs_second_part : ℕ := 4 * 9.5.toInt

-- The theorem statement to prove the total time
theorem total_running_time : rudy_runs_first_part + rudy_runs_second_part = 88 :=
by
  sorry

end total_running_time_l544_544365


namespace find_value_l544_544995

theorem find_value
  (x a y b z c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 :=
by 
  sorry

end find_value_l544_544995


namespace y_percent_of_N_correct_l544_544657

variables (N x y : ℝ)
variables (x_prime y_prime : Prop)
variables (h_prime_x : Prime x) (h_prime_y : Prime y)
variables (h_diff_primes : x ≠ y)
variable (h_condition : 70 = (x / 100) * N)

noncomputable def y_percent_of_N := (y / 100) * N

theorem y_percent_of_N_correct :
  y_percent_of_N x y N = (70 * y) / x :=
by
  unfold y_percent_of_N
  sorry

end y_percent_of_N_correct_l544_544657


namespace find_base_k_l544_544954

theorem find_base_k (k : ℕ) (hk : 0 < k) (h : 7/51 = (2 * k + 3) / (k^2 - 1)) : k = 16 :=
sorry

end find_base_k_l544_544954


namespace range_of_a_l544_544662

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
sorry

end range_of_a_l544_544662


namespace tree_growth_at_least_29_feet_in_15_years_l544_544044

/--
Tony planted a 4-foot tree.
- After the first year, the tree grows at a rate of 5 feet.
- After the second year, the tree grows at a rate of 4 feet.
- From the third year onwards, the growth rate of the tree decreases by 1 foot each year until it reaches a minimum growth rate of 1 foot per year.

We need to prove that the tree will be at least 29 feet tall in 15 years.
-/
theorem tree_growth_at_least_29_feet_in_15_years :
  ∃ t : ℕ, t = 15 ∧ 
  let h₀ := 4 in
  let g₁ := 5 in 
  let g₂ := 4 in
  let growth_rate (n : ℕ) := 
    if n = 0 then g₁ 
    else if n = 1 then g₂
    else 
      max (g₂ - (n - 1)) 1 
  in
  let height_after_t_years := h₀ + ∑ i in finset.range t, growth_rate i in 
  height_after_t_years ≥ 29 :=
begin 
  let h₀ := 4,
  let g₁ := 5,
  let g₂ := 4,
  let growth_rate := λ (n : ℕ), 
    if n = 0 then g₁
    else if n = 1 then g₂
    else 
      max (g₂ - (n - 1)) 1,
  let height_after_5_years := h₀ + g₁ + g₂ + (g₂ - 1) + (g₂ - 2) + ((g₂ - 3) + 1),
  have t1: height_after_5_years = 19, from by simp [height_after_5_years, g₂, g₁, h₀],
  have height_after_t_years := (h₀ + g₁ + g₂ + (g₂ - 1) + (g₂ - 2) + ((g₂ - 3) + 1)) + 10,
  use 15,
  split,
  refl,
  simp [height_after_t_years, t1, t],
  linarith,
end

end tree_growth_at_least_29_feet_in_15_years_l544_544044


namespace magician_can_identify_matching_coin_l544_544504

-- Define the types and conditions
variable (coins : Fin 11 → Bool) -- Each coin is either heads (true) or tails (false).
variable (uncovered_index : Fin 11) -- The index of the initially uncovered coin.

-- Define the property to be proved: the magician can point to an adjacent coin with the same face as the uncovered one.
theorem magician_can_identify_matching_coin (h : ∃ i : Fin 11, coins i = coins (i + 1) % 11) :
  ∃ j : Fin 11, (j ≠ uncovered_index) ∧ coins j = coins uncovered_index :=
  sorry

end magician_can_identify_matching_coin_l544_544504


namespace basket_placards_count_l544_544119

theorem basket_placards_count (n_people : ℕ) (placards_per_person : ℕ) (top_count : ℕ)
  (h1 : n_people = 2317) 
  (h2 : placards_per_person = 2) :
  top_count = 4634 :=
begin
  sorry
end

end basket_placards_count_l544_544119


namespace nonagon_lines_intersect_at_single_point_l544_544322

theorem nonagon_lines_intersect_at_single_point :
  ∀ (O A1 A2 A4 A5 A7 B1 B4 : Type),
    (RegularNonagon K9 O A1 A2 A3 A4 A5 A6 A7 A8 A9)
    → Midpoint(B1, A1, A2) → Midpoint(B4, A4, A5) 
    → (∃ P : Type, (OnLine P A1 A4) ∧ (OnLine P B1 B4) ∧ (OnLine P A7 O)) := 
sorry

end nonagon_lines_intersect_at_single_point_l544_544322


namespace max_length_interval_l544_544625

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := ((m ^ 2 + m) * x - 1) / (m ^ 2 * x)

theorem max_length_interval (a b m : ℝ) (h1 : m ≠ 0) (h2 : ∀ x, f m x = x → x ∈ Set.Icc a b) :
  |b - a| = (2 * Real.sqrt 3) / 3 := sorry

end max_length_interval_l544_544625


namespace log_equality_solution_l544_544584

theorem log_equality_solution (x : ℝ) : 
  (log (x^2) (x^2 - 3*x + 2) = 1 ∨ log (x^2) (x^2 / (x - 2)) = 1 ∨ log (x^2) (x^2 / (x - 1)) = 1) 
  ↔ x = 3 :=
begin
  sorry
end

end log_equality_solution_l544_544584


namespace num_correct_propositions_l544_544838

-- Define the propositions
def proposition1 : Prop :=
  ∀ (P : Type) [prism P], (∀ f : Face P, is_square f) → is_right_prism P

def proposition2 : Prop :=
  ∀ (S : Type) [sphere S] (p1 p2 : Point S), 
  p1 ≠ p2 → (∃! c : Circle S, great_circle c ∧ p1 ∈ c ∧ p2 ∈ c)

def proposition3 : Prop :=
  ∀ (L1 L2 : Line) [skew L1 L2],
  ∃ (proj : ¬Parallel), parallel_projection proj L1 = parallel_projection proj L2 

def proposition4 : Prop :=
  ∀ (P P' : Plane), ¬parallel P P' ∧ ∃ l : Line, l ∉ P → (∃1 p : Plane, parallel p P ∧ l ∈ p)

--  Formulate the proof problem about the propositions
theorem num_correct_propositions : 
  (proposition1 → false) ∧ 
  (proposition2 → false) ∧ 
  (proposition3) ∧ 
  (proposition4 → false) → 
  (num_true_propositions [proposition1, proposition2, proposition3, proposition4] = 1) :=
sorry

end num_correct_propositions_l544_544838


namespace custom_op_9_3_l544_544396

def custom_op (a b : ℝ) : ℝ := a + 4 * a / (3 * b)

theorem custom_op_9_3 : custom_op 9 3 = 13 :=
by {
  sorry
}

end custom_op_9_3_l544_544396


namespace div_coeff_roots_l544_544389

theorem div_coeff_roots :
  ∀ (a b c d e : ℝ), (∀ x, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4)
  → (d / e = -25 / 12) :=
by
  intros a b c d e h
  sorry

end div_coeff_roots_l544_544389


namespace problem1_problem2_l544_544449

variable (α : ℝ)

theorem problem1 (hα : α ∈ ℝ) : 
  4.48 * cot (4 * α - 3 * Real.pi / 2) + (1 / cos (4 * α - 3 * Real.pi)) = cot (2 * α - Real.pi / 4) := sorry

theorem problem2 (hα : α ∈ ℝ) : 
  4.493 - 4 * cos (4 * α - 3 * Real.pi) - cos (5 * Real.pi + 8 * α) = 8 * cos (2 * α)^4 := sorry

end problem1_problem2_l544_544449


namespace part1_part2_l544_544934

-- Assuming x is a real number
variable (x : ℝ) (a : ℝ)

theorem part1 : ∀ a : ℝ, (∀ x : ℝ, ¬ (| x - 4 | + | 3 - x | < a)) → a ≤ 1 :=
by sorry

theorem part2 : ∀ a : ℝ, (∃ x : ℝ, | x - 4 | + | 3 - x | < a) → a > 1 :=
by sorry

end part1_part2_l544_544934


namespace parallel_line_eq_perpendicular_line_eq_l544_544485

-- Define the conditions: A line passing through (1, -4) and the given line equation 2x + 3y + 5 = 0
def passes_through (x y : ℝ) (a b c : ℝ) : Prop := a * x + b * y + c = 0

-- Define the theorem statements for parallel and perpendicular lines
theorem parallel_line_eq (m : ℝ) :
  passes_through 1 (-4) 2 3 m → m = 10 := 
sorry

theorem perpendicular_line_eq (n : ℝ) :
  passes_through 1 (-4) 3 (-2) (-n) → n = 11 :=
sorry

end parallel_line_eq_perpendicular_line_eq_l544_544485


namespace sum_floor_log2_l544_544550

theorem sum_floor_log2 : (∑ N in finset.range 2048.succ, ⌊real.log2 N⌋₊) = 20445 := sorry

end sum_floor_log2_l544_544550


namespace nearest_integer_is_11304_l544_544069

def nearest_integer_to_a_plus_b_pow_six (a b : ℝ) (h : b = Real.sqrt 5) : ℝ :=
  (a + b) ^ 6

theorem nearest_integer_is_11304 : nearest_integer_to_a_plus_b_pow_six 3 (Real.sqrt 5) rfl = 11304 := 
  sorry

end nearest_integer_is_11304_l544_544069


namespace equal_horizontal_vertical_segment_lengths_l544_544354

def grid_vertex (P : Type) := P

def convex_polygon (P : Type) [vertex: grid_vertex P] (vertices : set P) : Prop := 
  true -- this is a placeholder for the actual definition

def grid_segment_length (P : Type) [vertex: grid_vertex P] (a b : P) : ℝ := 
  sorry -- placeholder for actual segment length function

def horizontal_vertical_segment_lengths_equal (P : Type) 
  [vertex: grid_vertex P] 
  (vertices : set P) 
  (H : convex_polygon P vertices) 
  : Prop :=
  ∀ (horizontal_lengths vertical_lengths : list ℝ),
    (∀ ab, -- condition for horizontal segments
      ab ∈ vertices →
      (grid_segment_length P ab.1 ab.2 = sorry)) → -- calculate grid segment length for horizontal
    (∀ ab, -- condition for vertical segments
      ab ∈ vertices →
      (grid_segment_length P ab.1 ab.2 = sorry)) → -- calculate grid segment length for vertical
    horizontal_lengths.sum = vertical_lengths.sum

-- Statement to be proved
theorem equal_horizontal_vertical_segment_lengths 
  (P : Type) [vertex: grid_vertex P] (vertices : set P) 
  (H : convex_polygon P vertices) :
  horizontal_vertical_segment_lengths_equal P vertices H :=
by sorry

end equal_horizontal_vertical_segment_lengths_l544_544354


namespace exercise_proof_l544_544463

noncomputable theory

open_locale real

def line_l (t : ℝ) : ℝ × ℝ := (3 - (real.sqrt 2 / 2) * t, real.sqrt 5 + (real.sqrt 2 / 2) * t)

def circle_C (ρ θ : ℝ) : Prop := ρ = 2 * real.sqrt 5 * real.sin θ

def cartesian_circle_C (x y : ℝ) : Prop :=
  x^2 + (y - real.sqrt 5)^2 = 5

def P : ℝ × ℝ := (3, real.sqrt 5)

theorem exercise_proof :
  (∀ (ρ θ : ℝ), circle_C ρ θ → exists (x y : ℝ), x^2 + y^2 - 2 * real.sqrt 5 * y = 0 ∧ cartesian_circle_C x y) ∧
  ∀ (A B : ℝ × ℝ), (∃ t : ℝ, line_l t = A ∧ (∃ t : ℝ, line_l t = B)) →
    (sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)) = 3 * real.sqrt 2 :=
by
  sorry

end exercise_proof_l544_544463


namespace remainder_degrees_l544_544814

-- Define the divisor polynomial
def divisor : Polynomial ℝ := 2 * (Polynomial.X ^ 2) - 7 * Polynomial.X + 18

-- Prove the possible degrees of the remainder
theorem remainder_degrees (P : Polynomial ℝ) : ∃ r : Polynomial ℝ, r.degree < divisor.degree ∧ r.degree ∈ {0, 1} :=
sorry

end remainder_degrees_l544_544814


namespace bridge_length_l544_544523

theorem bridge_length (train_length : ℕ) (train_cross_bridge_time : ℕ) (train_cross_lamp_time : ℕ) (bridge_length : ℕ) :
  train_length = 600 →
  train_cross_bridge_time = 70 →
  train_cross_lamp_time = 20 →
  bridge_length = 1500 :=
by
  intro h1 h2 h3
  sorry

end bridge_length_l544_544523


namespace max_n_for_integer_division_l544_544236

-- Definitions based on given conditions
def sequence_product := ∏ x in finset.range(1201) + 810, x
def divisor_power := (810 : ℕ) ^ 150

-- The theorem to prove the given problem statement
theorem max_n_for_integer_division :
  (sequence_product : ℚ) / divisor_power ∈ ℤ :=
sorry

end max_n_for_integer_division_l544_544236


namespace evie_gave_2_shells_to_brother_l544_544577

def daily_shells : ℕ := 10
def days : ℕ := 6
def remaining_shells : ℕ := 58

def total_shells : ℕ := daily_shells * days
def shells_given : ℕ := total_shells - remaining_shells

theorem evie_gave_2_shells_to_brother :
  shells_given = 2 :=
by
  sorry

end evie_gave_2_shells_to_brother_l544_544577


namespace hyperbola_eccentricity_l544_544484

-- Definitions of geometric properties and relationships
variable (O F A B : Point)
variable (OA OB AB : ℝ)
variable (l1 l2 : Line)

-- Given Conditions
def condition1 : O = 0 := sorry
def condition2 : (foci_of_hyperbola (x_axis)) := sorry
def condition3 : asymptotes_hyperbola l1 l2 := sorry
def condition4 : line_perpendicular_to l1 (passes_through F) := sorry
def condition5 : intersects_at AB l1 A l2 B := sorry
def condition6 : arithmetic_sequence OA AB OB := sorry
def condition7 : same_direction BF FA := sorry

-- Translation of the solution in equivalence statement
theorem hyperbola_eccentricity (c1 : condition1) (c2 : condition2) (c3 : condition3)
  (c4 : condition4) (c5 : condition5) (c6 : condition6) (c7 : condition7) : ecc(hyperbola) = (sqrt 5) / 2 := 
  by
    -- the proof steps
    sorry

end hyperbola_eccentricity_l544_544484


namespace sum_digits_property_compute_s_l544_544715

def sum_of_digits (n : ℕ) : ℕ :=
  list.sum (n.digits 10)

theorem sum_digits_property (n : ℕ) : (n - sum_of_digits n) % 9 = 0 :=
  sorry

theorem compute_s (x : ℕ) : sum_of_digits (sum_of_digits (sum_of_digits (2 ^ 2009))) = 5 :=
  sorry

end sum_digits_property_compute_s_l544_544715


namespace magician_trick_l544_544492

theorem magician_trick (coins : Fin 11 → Bool) 
  (exists_pair : ∃ i : Fin 11, coins i = coins ((i + 1) % 11)) 
  (assistant_left_uncovered : Fin 11) 
  (uncovered_coin_same_face : coins assistant_left_uncovered = coins (some (exists_pair))) :
  ∃ j : Fin 11, j ≠ assistant_left_uncovered ∧ coins j = coins assistant_left_uncovered := 
  sorry

end magician_trick_l544_544492


namespace func_D_is_exponential_l544_544136

-- Define the exponential function condition
def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f x = a ^ x

-- Define the given function that needs to be proved as exponential
def func_D (x : ℝ) : ℝ := 3 ^ (-x)

-- The theorem statement that needs to be proved
theorem func_D_is_exponential : is_exponential_function func_D :=
sorry

end func_D_is_exponential_l544_544136


namespace monomial_properties_l544_544089

def monomial_coef (m : ℕ → ℕ → ℤ) : ℤ :=
  m 2 1

def monomial_deg (m : ℕ → ℕ → ℤ) : ℕ :=
  2 + 1

theorem monomial_properties :
  ∃ m : ℕ → ℕ → ℤ, monomial_coef m = -3 ∧ monomial_deg m = 3 :=
by
  let m := λ x y, if x = 2 ∧ y = 1 then -3 else 0
  use m
  split
  { simp [monomial_coef] }
  { simp [monomial_deg] }

end monomial_properties_l544_544089


namespace indefinite_integral_solution_l544_544098

open Real

theorem indefinite_integral_solution (c : ℝ) : 
  ∫ x, (1 - cos x) / (x - sin x) ^ 2 = - 1 / (x - sin x) + c := 
sorry

end indefinite_integral_solution_l544_544098


namespace simplify_expression1_simplify_expression2_l544_544380

variables (α : ℝ)
-- Condition: α is in the third quadrant
def third_quadrant : Prop :=
  α > π ∧ α < 3 * π / 2

-- Hypothesis taken from the third quadrant properties
axiom h_sin : third_quadrant α → -1 < Real.sin α ∧ Real.sin α < 0
axiom h_cos : third_quadrant α → -1 < Real.cos α ∧ Real.cos α < 0

theorem simplify_expression1 (h : third_quadrant α) :
  (Real.cos (α - π / 2) / Real.sin (5 * π / 2 + α) * Real.sin (α - 2 * π) * Real.cos (π - α)) =
  - (Real.sin α) ^ 2 :=
sorry

theorem simplify_expression2 (h : third_quadrant α) :
  (Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α))) =
  -2 * Real.tan α :=
sorry

end simplify_expression1_simplify_expression2_l544_544380


namespace right_triangle_area_l544_544423

def triangle_area (a b : ℝ) : ℝ :=
  0.5 * a * b

theorem right_triangle_area
  (A B C : Type)
  [IsRightTriangle A B C (angle := 90)]
  (AC BC : ℝ)
  (h_AC : AC = 6)
  (h_AB : BC = 8) :
  triangle_area AC BC = 24 :=
by
  -- placeholder for actual proof step
  sorry

end right_triangle_area_l544_544423


namespace magnus_score_in_third_game_l544_544726

-- Definitions and conditions
def game1_score (a b: ℕ) : Prop := a = b + 2
def game2_score (b c: ℕ) : Prop := b = (a + c) / 2
def unique_scores (s: Finset ℕ) : Prop := s.card = 6

def problem_conditions (m1 m2 m3 v1 v2 v3: ℕ) : Prop :=
  -- Positive integer scores and unique across games
  0 < m1 ∧ 0 < m2 ∧ 0 < m3 ∧ 0 < v1 ∧ 0 < v2 ∧ 0 < v3 ∧
  unique_scores {m1, m2, m3, v1, v2, v3} ∧
  -- Win and scoring conditions
  (v1 ≥ 25 ∨ v2 ≥ 25 ∨ v3 = 25) ∧
  ((v1 = 25 → m1 ≤ 23) ∧ (v2 = 25 → m2 ≤ 23)) ∧
  ((v1 > 25 → m1 = v1 - 2) ∧ (v2 > 25 → m2 = v2 - 2)) ∧
  (v1 ≠ v2) ∧ (v2 ≠ v3) ∧ (v1 ≠ v3) ∧
  -- Specific conditions
  v3 = 25 ∧
  -- Each player's second game score is the average of their 1st and 3rd game scores
  game1_score m1 m2 ∧ game2_score m2 m3 ∧
  game1_score v1 v2 ∧ game2_score v2 v3

-- Proving Magnus's score in the 3rd game equals 19 given conditions.
theorem magnus_score_in_third_game (m1 m2 m3 v1 v2 v3: ℕ) :
  problem_conditions m1 m2 m3 v1 v2 v3 → m3 = 19 :=
by
  sorry

end magnus_score_in_third_game_l544_544726


namespace remainder_8437_by_9_l544_544811

theorem remainder_8437_by_9 : 8437 % 9 = 4 :=
by
  -- proof goes here
  sorry

end remainder_8437_by_9_l544_544811


namespace number_of_possible_n_l544_544776

theorem number_of_possible_n : 
  (log 15 / log 5) + (log 125 / log 5) > log n / log 5 ∧ 
  (log 15 / log 5) + (log n / log 5) > log 125 / log 5 ∧ 
  (log 125 / log 5) + (log n / log 5) > log 15 / log 5 ∧ 
  n > 0 → 
  ∃ k : ℕ, k = 1866 :=
sorry

end number_of_possible_n_l544_544776


namespace quadratic_value_at_point_l544_544121

variable (a b c : ℝ)

-- Given: A quadratic function f(x) = ax^2 + bx + c that passes through the point (3,10)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_value_at_point
  (h : f a b c 3 = 10) :
  5 * a - 3 * b + c = -4 * a - 6 * b + 10 := by
  sorry

end quadratic_value_at_point_l544_544121


namespace initially_working_machines_l544_544850

theorem initially_working_machines (N R x : ℝ) 
  (h1 : N * R = x / 3) 
  (h2 : 45 * R = x / 2) : 
  N = 30 := by
  sorry

end initially_working_machines_l544_544850


namespace min_example_inequality_solution_l544_544327

open Set

def min_fn (a b : ℝ) : ℝ := min a b

def min_example_fn (x : ℝ) : ℝ := min_fn (x + 2) (x^2 - 6 * x + 8)

theorem min_example_inequality_solution : 
  {x : ℝ | min_example_fn x ≥ 0} = Icc (-2 : ℝ) 2 ∪ Ici 4 :=
by
  sorry

end min_example_inequality_solution_l544_544327


namespace conner_collected_on_day_two_l544_544004

variable (s0 : ℕ) (c0 : ℕ) (s1 : ℕ) (c1 : ℕ) (c2 : ℕ) (s3 : ℕ) (c3 : ℕ) (total_sydney : ℕ) (total_conner : ℕ)

theorem conner_collected_on_day_two :
  s0 = 837 ∧ c0 = 723 ∧ 
  s1 = 4 ∧ c1 = 8 * s1 ∧
  s3 = 2 * c1 ∧ c3 = 27 ∧
  total_sydney = s0 + s1 + s3 ∧
  total_conner = c0 + c1 + c2 + c3 ∧
  total_conner >= total_sydney
  → c2 = 123 :=
by
  sorry

end conner_collected_on_day_two_l544_544004


namespace positive_difference_AX_AY_l544_544316

-- Given definitions and conditions
variables (A B C X Y T1 T2 : Type*) [metric_space A] [metric_space B] 
          [metric_space C] [metric_space X] [metric_space Y] [metric_space T1] [metric_space T2]
          (AB AC BC : ℝ) (XY : ℝ)

-- Conditions
def triangle_sides : Prop := AB = 13 ∧ BC = 14 ∧ AC = 15
def incircle_tangent_points : Prop := true -- Details of incircle tangency not specified
def tangent_line_intersects : Prop := XY = 5

-- Question to prove
def positive_difference : Prop := 
  ∃ (AX AY : ℝ), (AX + AY = 9) ∧ (AX * AY = 130 / 7) ∧ |AX - AY| = (sqrt 329) / 7

-- Main theorem statement
theorem positive_difference_AX_AY
  (h_triangle_sides : triangle_sides AB AC BC)
  (h_incircle_tangent_points : incircle_tangent_points)
  (h_tangent_line_intersects : tangent_line_intersects XY)
  : positive_difference := 
sorry

end positive_difference_AX_AY_l544_544316


namespace normal_dist_symm_l544_544213

-- Definitions for the problem
variables (ξ : ℝ → ℝ) (σ : ℝ)
hypothesis h1 : ∀ x, ξ x ~ normal(4, σ^2)
hypothesis h2 : P(ξ > 8) = 0.4

-- Theorem to prove
theorem normal_dist_symm : P(ξ < 0) = 0.4 :=
sorry

end normal_dist_symm_l544_544213


namespace Raja_work_alone_days_l544_544364

theorem Raja_work_alone_days :
  ∃ x : ℕ, (1 / (x : ℝ) + 1 / 6 = 1 / 4) ∧ x = 3 :=
begin
  sorry
end

end Raja_work_alone_days_l544_544364


namespace sampling_method_l544_544960

/-
  Define the problem statement and relevant conditions.
-/
noncomputable def number_of_basketballs : ℕ := 10
noncomputable def selected_basketball : fin number_of_basketballs := 0 -- assuming 0-indexed for simplicity

-- Define the property we want to prove: Using simple random sampling for selecting one basketball out of 10.
theorem sampling_method (n : ℕ) (sel : fin n) : n = 10 → sel = 0 → ∃ method : String, method = "Simple random sampling" :=
by
  intros h₁ h₂
  use "Simple random sampling"
  sorry

end sampling_method_l544_544960


namespace games_bought_at_garage_sale_l544_544724

theorem games_bought_at_garage_sale (G : ℕ)
  (h1 : 2 + G - 2  = 2) :
  G = 2 :=
by {
  sorry
}

end games_bought_at_garage_sale_l544_544724


namespace minimize_quadratic_l544_544436

def f (x : ℝ) := 3 * x^2 - 18 * x + 7

theorem minimize_quadratic : ∃ x : ℝ, f x = -20 ∧ ∀ y : ℝ, f y ≥ -20 := by
  sorry

end minimize_quadratic_l544_544436


namespace f_2015_eq_2_l544_544199

def f : ℤ → ℤ
| x => if x ≤ 0 then int.floor (real.log (2) (3 - x)) else f(x-1) - f(x-2)

theorem f_2015_eq_2 : f 2015 = 2 := 
  sorry

end f_2015_eq_2_l544_544199


namespace students_like_both_l544_544669

def num_students := 40
def like_basketball := 20
def like_tabletennis := 15
def not_like_either := 8

theorem students_like_both :
  let like_either := num_students - not_like_either in
  like_basketball + like_tabletennis - like_either = 3 :=
by
  sorry

end students_like_both_l544_544669


namespace power_sum_log_l544_544601

noncomputable def log (x : ℝ) := Real.log x

theorem power_sum_log (a b : ℝ) (h1 : a = log 25) (h2 : b = log 36) :
  5 ^ (a / b) + 6 ^ (b / a) = 11 :=
by
  sorry

end power_sum_log_l544_544601


namespace x_coordinate_of_vertex_A_l544_544755

theorem x_coordinate_of_vertex_A (x y : ℝ) (a : ℝ) 
  (h1 : a = 2) 
  (h2 : y = log a x) 
  (h3 : y = log a (x-5)) 
  (h4 : y = log a (x-10)) 
  (h5 : ∀ x y, square_of_side_length (5 : ℝ)) 
  : x = 15 :=
sorry

end x_coordinate_of_vertex_A_l544_544755


namespace Mike_spent_on_new_mower_blades_l544_544728

theorem Mike_spent_on_new_mower_blades:
  let money_made := 42
  let game_price := 8
  let games_bought := 4
  let money_left := games_bought * game_price
  ∃ money_spent, money_spent = money_made - money_left :=
by
  let money_made := 42
  let game_price := 8
  let games_bought := 4
  let money_left := games_bought * game_price
  exists (money_made - money_left)
  sorry

end Mike_spent_on_new_mower_blades_l544_544728


namespace dot_product_computation_l544_544557

theorem dot_product_computation :
  let v1 := (vector.mk [-7, 3] 2)
  let v2 := (vector.mk [4, -5] 2)
  (v1.toList.head * v2.toList.head + v1.toList.tail.head * v2.toList.tail.head) = -43 :=
by
  sorry

end dot_product_computation_l544_544557


namespace hyperbola_equation_l544_544342

theorem hyperbola_equation (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
    (h_line : ∃ (m c : ℝ), ∀ (x : ℝ), ((b = -1) → (x = 0 ∧ c = 0 ∧ m = -1 ∧ x = 1)) 
    (C_asymptote_parallel : y = (b/a)x) (C_asymptote_perpendicular : y = -(a/b)x) :
    a = 1 ∧ b = 1 :=
by
  sorry

end hyperbola_equation_l544_544342


namespace det_inverse_l544_544226

variable {B : Matrix ℝ ℝ} -- Declaring B as a matrix over the reals

-- Given condition
axiom det_B_is_neg3 : det B = -3

-- Proof statement to show that det (B⁻¹) = -1 / 3
theorem det_inverse : det (B⁻¹) = -1 / 3 :=
by
  -- The proof would go here, but we are using sorry to skip the proof.
  sorry

end det_inverse_l544_544226


namespace expansion_coeff_sum_l544_544766

theorem expansion_coeff_sum (n k : ℕ) (hn: (x + 2)^n = ∑ (i : ℕ) in range(n + 1), (choose n i) * x^(n - i) * 2^i) 
  (h1: choose n k * 1 = choose n (k + 1) * 3) 
  (h2: choose n (k + 1) * 3 = choose n (k + 2) * 6) : 
  n + k = 6 :=
by
  sorry

end expansion_coeff_sum_l544_544766


namespace income_of_deceased_member_l544_544094

theorem income_of_deceased_member
  (A B C : ℝ) -- Incomes of the three members
  (h1 : (A + B + C) / 3 = 735)
  (h2 : (A + B) / 2 = 650) :
  C = 905 :=
by
  sorry

end income_of_deceased_member_l544_544094


namespace mulberry_sales_l544_544458

theorem mulberry_sales (x : ℝ) (p : ℝ) (h1 : 3000 = x * p)
    (h2 : 150 * (p * 1.4) + (x - 150) * (p * 0.8) - 3000 = 750) :
    x = 200 := by sorry

end mulberry_sales_l544_544458


namespace row_arrangement_possible_circle_arrangement_impossible_l544_544412

/-- This structure represents either standing in a row or in a circle -/
def Arrangement := List ℕ

/-- The condition that the sum of any two adjacent elements in the list is a prime number -/
def adjacencyPrime (arr : Arrangement) : Prop :=
  ∀ i, ∃ k : ℕ, Prime (arr.get! i + arr.get! ((i + 1) % arr.length))

/-- The jerseys numbered from 1 to 41 -/
def jerseys := (List.range 41).map (λ x => x + 1)

/-- Proves that it is possible to arrange athletes in a row such that sum of adjacent jersey numbers is prime -/
theorem row_arrangement_possible : ∃ arr : Arrangement, adjacencyPrime arr ∧ (arr.perm jerseys) :=
sorry

/-- Proves that it is not possible to arrange athletes in a circle such that sum of adjacent jersey numbers is prime -/
theorem circle_arrangement_impossible : ¬ ∃ arr : Arrangement, adjacencyPrime arr ∧ arr.head = arr.last :=
sorry

end row_arrangement_possible_circle_arrangement_impossible_l544_544412


namespace Sarah_ate_one_apple_l544_544374

theorem Sarah_ate_one_apple:
  ∀ (total_apples apples_given_to_teachers apples_given_to_friends apples_left: ℕ), 
  total_apples = 25 →
  apples_given_to_teachers = 16 →
  apples_given_to_friends = 5 →
  apples_left = 3 →
  total_apples - (apples_given_to_teachers + apples_given_to_friends + apples_left) = 1 :=
by
  intros total_apples apples_given_to_teachers apples_given_to_friends apples_left
  intro ht ht gt hf
  sorry

end Sarah_ate_one_apple_l544_544374


namespace units_digit_of_product_l544_544695

def KarolinasNumbers : Set ℕ := {123, 321}
def NikolasOddNumbers : Set ℕ := {465, 645}

theorem units_digit_of_product 
  (a ∈ KarolinasNumbers) 
  (b ∈ NikolasOddNumbers)
  (even (a + b)) : (a * b) % 10 = 5 :=
sorry

end units_digit_of_product_l544_544695


namespace f_7_is_neg_2_l544_544984

variables {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x : ℝ, f (x + p) = f x

theorem f_7_is_neg_2 (h_odd : is_odd_function f) (h_periodic : is_periodic f 4) 
  (h_on_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) : f 7 = -2 :=
begin
  sorry,
end

end f_7_is_neg_2_l544_544984


namespace find_two_numbers_l544_544047

noncomputable def num1 : ℝ := 5 + 2 * Real.sqrt 5
noncomputable def num2 : ℝ := 5 - 2 * Real.sqrt 5

theorem find_two_numbers (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = 5) (h4 : a + b = 10) : 
  {a, b} = {num1, num2} :=
by
  sorry

end find_two_numbers_l544_544047


namespace total_pixels_correct_l544_544866

-- Define the monitor's dimensions and pixel density as given conditions
def width_inches : ℕ := 21
def height_inches : ℕ := 12
def pixels_per_inch : ℕ := 100

-- Define the width and height in pixels based on the given conditions
def width_pixels : ℕ := width_inches * pixels_per_inch
def height_pixels : ℕ := height_inches * pixels_per_inch

-- State the objective: proving the total number of pixels on the monitor
theorem total_pixels_correct : width_pixels * height_pixels = 2520000 := by
  sorry

end total_pixels_correct_l544_544866


namespace restaurant_A2_probability_l544_544385

noncomputable def prob_A2 (P_A1 P_B1 P_A2_given_A1 P_A2_given_B1 : ℝ) : ℝ :=
  P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1

theorem restaurant_A2_probability :
  let P_A1 := 0.4
  let P_B1 := 0.6
  let P_A2_given_A1 := 0.6
  let P_A2_given_B1 := 0.5
  prob_A2 P_A1 P_B1 P_A2_given_A1 P_A2_given_B1 = 0.54 :=
by
  sorry

end restaurant_A2_probability_l544_544385


namespace sale_price_relative_to_original_l544_544421

variable (x : ℝ)

def increased_price (x : ℝ) := 1.30 * x
def sale_price (increased_price : ℝ) := 0.90 * increased_price

theorem sale_price_relative_to_original (x : ℝ) :
  sale_price (increased_price x) = 1.17 * x :=
by
  sorry

end sale_price_relative_to_original_l544_544421


namespace transportation_modes_l544_544132

-- Define the people
inductive Person
| Alex : Person
| Borya : Person
| Vitya : Person

-- Define the modes of transportation
inductive TransportMode
| Bus : TransportMode
| Tram : TransportMode
| Trolleybus : TransportMode

-- Define the conditions
def different_modes (p1 p2 p3: TransportMode) : Prop :=
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

axiom condition1 :
  different_modes TransportMode.Bus TransportMode.Tram TransportMode.Trolleybus

axiom condition2 :
  ∀ (m : TransportMode), m ≠ TransportMode.Bus → (Person.Alex, m) ≠ (Person.Alex, TransportMode.Bus)

axiom condition3 :
  ∀ (m : TransportMode), (Person.Borya, m) = (Person.Borya, TransportMode.Trolleybus) → (Person.Alex, m) ≠ (Person.Alex, TransportMode.Trolleybus)

-- The proof statement to be proved
theorem transportation_modes :
  (Person.Alex, TransportMode.Tram) ∧ (Person.Borya, TransportMode.Bus) ∧ (Person.Vitya, TransportMode.Trolleybus) :=
sorry

end transportation_modes_l544_544132


namespace three_distinct_intersections_l544_544232

def f (x : ℝ) : ℝ := (-x^2 + x - 1) * Real.exp x
def g (x m : ℝ) : ℝ := 1/3 * x^3 + 1/2 * x^2 + m

theorem three_distinct_intersections (m : ℝ) :
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = g a m ∧ f b = g b m ∧ f c = g c m) ↔
  -3 / Real.exp(1) - 1/6 < m ∧ m < -1 :=
by { sorry }

end three_distinct_intersections_l544_544232


namespace compute_expression_l544_544154

theorem compute_expression :
  ( (12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) )
  /
  ( (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) )
  = 221 := 
by sorry

end compute_expression_l544_544154


namespace find_base_k_l544_544955

theorem find_base_k (k : ℕ) (hk : 0 < k) (h : 7/51 = (2 * k + 3) / (k^2 - 1)) : k = 16 :=
sorry

end find_base_k_l544_544955


namespace total_red_peaches_l544_544413

theorem total_red_peaches (num_baskets : ℕ) (red_per_basket : ℕ) (num_baskets_eq_6 : num_baskets = 6) (red_per_basket_eq_16 : red_per_basket = 16) : num_baskets * red_per_basket = 96 :=
by
  rw [num_baskets_eq_6, red_per_basket_eq_16]
  simp
  sorry

end total_red_peaches_l544_544413


namespace geometric_progression_identical_numbers_l544_544792

theorem geometric_progression_identical_numbers (n : ℕ) (numbers : Fin (4 * n) → ℝ) 
  (h_pos : ∀ i, 0 < numbers i) (h_geom : ∀ i j k l, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
    numbers i / numbers j = numbers k / numbers l) :
  ∃ m, n ≤ (Finset.univ : Finset (Fin (4 * n))).filter (λ i, numbers i = m).card := sorry

end geometric_progression_identical_numbers_l544_544792


namespace increasing_interval_l544_544019

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 2 - Real.log x

theorem increasing_interval :
  (∀ x, x ∈ Ioc 0 (1 / 2) → f x < f (1 / 2)) ∧ (∀ x, x ∈ Ioc (1 / 2) ∞ → f (x / 2) < f x) :=
sorry

end increasing_interval_l544_544019


namespace number_of_lattice_points_in_triangle_l544_544140

theorem number_of_lattice_points_in_triangle (L : ℕ) (hL : L > 1) :
  ∃ I, I = (L^2 - 1) / 2 :=
by
  sorry

end number_of_lattice_points_in_triangle_l544_544140


namespace number_of_left_handed_jazz_lovers_l544_544035

variable (total_members left_handed jazz_lovers rh_non_jazz_lovers left_handed_jazz_lovers : ℕ)

axiom club_conditions :
  total_members = 20 ∧
  left_handed = 8 ∧
  jazz_lovers = 15 ∧
  rh_non_jazz_lovers = 2

theorem number_of_left_handed_jazz_lovers :
  ∀ (total_members left_handed jazz_lovers rh_non_jazz_lovers : ℕ),
    total_members = 20 → 
    left_handed = 8 → 
    jazz_lovers = 15 → 
    rh_non_jazz_lovers = 2 → 
    (∃ x, x + (left_handed - x) + (jazz_lovers - x) + rh_non_jazz_lovers = total_members ∧ x = 5) :=
by {
  intros, 
  sorry
}

end number_of_left_handed_jazz_lovers_l544_544035


namespace average_skips_per_round_l544_544369

theorem average_skips_per_round (S : ℝ) : 
  let skips_round_1 := S - 1,
      skips_round_2 := S - 3,
      skips_round_3 := S + 4,
      skips_round_4 := S / 2,
      total_skips := skips_round_1 + skips_round_2 + skips_round_3 + skips_round_4,
      average_skips := total_skips / 4
  in average_skips = 9 * S / 8 :=
by
  sorry

end average_skips_per_round_l544_544369


namespace words_reduced_to_length_8_l544_544319

def is_word (w : string) : Prop :=
  ∀ c ∈ w.to_list, c = 'a' ∨ c = 'b' ∨ c = 'c'

def rel (u v : string) : Prop :=
  (u + u = v) ∨ (∃ w x y, (u = w ++ x ++ y ∧ v = w ++ x ++ x ++ y) ∨ (u = w ++ x ++ x ++ y ∧ v = w ++ x ++ y))

def equiv (u v : string) : Prop :=
  ∃ n (us : list string), us.length = n + 1 ∧ us.head! = u ∧ us.last! = v ∧
  ∀ i, i < n → rel (us.nth_le i (nat.lt_of_lt_pred i)) (us.nth_le (i + 1) (nat.lt_succ_of_lt i))

theorem words_reduced_to_length_8 (w : string) (hw: is_word w) : ∃ w', equiv w w' ∧ w'.length ≤ 8 :=
sorry

end words_reduced_to_length_8_l544_544319


namespace num_distinct_pairs_l544_544947

theorem num_distinct_pairs : ∃ (s : Finset (ℝ × ℝ)), 
  (∀ (p : ℝ × ℝ), p ∈ s ↔ (p.1 = 3 * (p.1^2 + p.2^2) ∧ p.2 = 3 * p.1 * p.2)) ∧ s.card = 2 :=
begin
  sorry
end

end num_distinct_pairs_l544_544947


namespace problem1_problem2_l544_544837

-- Part 1
theorem problem1 (f : ℝ → ℝ) (h : ∀ x, f (1/x) = x / (1 - x^2)) (x ≠ 0) : f x = 1 / (x^2 - 1) := 
sorry

-- Part 2
theorem problem2 (f : ℝ → ℝ) (h₀ : f 0 = 2) (h : ∀ x, f (x + 1) - f x = 2 * x - 1) : 
  ∃ a b c, f = (fun x => a * x^2 + b * x + c) ∧ a ≠ 0 ∧ f = (fun x => x^2 - 2 * x + 2) := 
sorry

end problem1_problem2_l544_544837


namespace last_10_digits_repeat_periodically_l544_544563

theorem last_10_digits_repeat_periodically :
  ∃ (p : ℕ) (n₀ : ℕ), p = 4 * 10^9 ∧ n₀ = 10 ∧ 
  ∀ n, (2^(n + p) % 10^10 = 2^n % 10^10) :=
by sorry

end last_10_digits_repeat_periodically_l544_544563


namespace find_k_l544_544774

noncomputable def y (k x : ℝ) : ℝ := k / x

theorem find_k (k : ℝ) (h₁ : k ≠ 0) (h₂ : 1 ≤ 3) 
  (h₃ : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → x = 1 ∨ x = 3) 
  (h₄ : |y k 1 - y k 3| = 4) : k = 6 ∨ k = -6 :=
  sorry

end find_k_l544_544774


namespace parabola_constant_c_l544_544511

theorem parabola_constant_c (b c : ℝ): 
  (∀ x : ℝ, y = x^2 + b * x + c) ∧ 
  (10 = 2^2 + b * 2 + c) ∧ 
  (31 = 4^2 + b * 4 + c) → 
  c = -3 :=
by
  sorry

end parabola_constant_c_l544_544511


namespace remainder_div_357_l544_544734

theorem remainder_div_357 (N : ℤ) (h : N % 17 = 2) : N % 357 = 2 :=
sorry

end remainder_div_357_l544_544734


namespace total_students_l544_544409

theorem total_students (third_grade_students fourth_grade_students second_grade_boys second_grade_girls : ℕ)
  (h1 : third_grade_students = 19)
  (h2 : fourth_grade_students = 2 * third_grade_students)
  (h3 : second_grade_boys = 10)
  (h4 : second_grade_girls = 19) :
  third_grade_students + fourth_grade_students + (second_grade_boys + second_grade_girls) = 86 :=
by
  rw [h1, h3, h4, h2]
  norm_num
  sorry

end total_students_l544_544409


namespace true_propositions_l544_544631

theorem true_propositions :
  let prop1 := ∀ Δy Δx : ℝ, Δx ≠ 0 → Δx > 0
  let prop2 := ∀ (f : ℝ → ℝ) (x : ℝ), ∀ (tangent_line : ℝ → ℝ), ∃! p, tangent_line p = f p ∧ p = x
  let prop3 := ∀ f, f = λ x, sin (π / 3) → deriv f = λ x, cos (π / 3)
  let prop4 := ∀ (f : ℝ → ℝ) (a b : ℝ), (∀ x ∈ Ioo a b, f x < f (x + ε) ∀ ε > 0) → ∀ x ∈ Ioo a b, deriv f x ≥ 0
  let prop5 := ∀ (f : ℝ → ℝ) (a b : ℝ), continuous_on f (Icc a b) → ∃ c ∈ Icc a b, ∀ y ∈ Icc a b, f y ≤ f c ∨ f y ≥ f c
  prop4 ∧ prop5 :=
by
  have prop1 := ∀ Δy Δx : ℝ, Δx ≠ 0 → Δx > 0
  have prop2 := ∀ (f : ℝ → ℝ) (x : ℝ), ∀ (tangent_line : ℝ → ℝ), ∃! p, tangent_line p = f p ∧ p = x
  have prop3 := ∀ f, f = λ x, sin (π / 3) → deriv f = λ x, cos (π / 3)
  have prop4 := ∀ (f : ℝ → ℝ) (a b : ℝ), (∀ x ∈ Ioo a b, f x < f (x + ε) ∀ ε > 0) → ∀ x ∈ Ioo a b, deriv f x ≥ 0
  have prop5 := ∀ (f : ℝ → ℝ) (a b : ℝ), continuous_on f (Icc a b) → ∃ c ∈ Icc a b, ∀ y ∈ Icc a b, f y ≤ f c ∨ f y ≥ f c
  exact ⟨prop4, prop5⟩

end true_propositions_l544_544631


namespace log_inequalities_l544_544462

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

theorem log_inequalities : c < b ∧ b < a :=
  sorry

end log_inequalities_l544_544462


namespace mrs_jane_total_coins_l544_544130

theorem mrs_jane_total_coins (Jayden_coins Jason_coins : ℕ) (h1 : Jayden_coins = 300) (h2 : Jason_coins = Jayden_coins + 60) :
  Jayden_coins + Jason_coins = 660 :=
sorry

end mrs_jane_total_coins_l544_544130


namespace junk_mail_per_block_l544_544410

theorem junk_mail_per_block (houses_per_block : ℕ) (mail_per_house : ℕ) (total_mail : ℕ) :
  houses_per_block = 20 → mail_per_house = 32 → total_mail = 640 := by
  intros hpb_price mph_correct
  sorry

end junk_mail_per_block_l544_544410


namespace ten_thousand_points_length_l544_544353

theorem ten_thousand_points_length (a b : ℝ) (d : ℝ) 
  (h1 : d = a / 99) 
  (h2 : b = 9999 * d) : b = 101 * a := by
  sorry

end ten_thousand_points_length_l544_544353


namespace nearest_integer_to_expression_correct_l544_544056

open Real

noncomputable def nearest_integer_to_expression : ℝ :=
  let a := (3 + (sqrt 5))^6
  let b := (3 - (sqrt 5))^6
  let s := a + b
  floor s

theorem nearest_integer_to_expression_correct :
  nearest_integer_to_expression = 74608 :=
begin
  sorry
end

end nearest_integer_to_expression_correct_l544_544056


namespace usable_field_area_l544_544661

open Float

def breadth_of_field (P : ℕ) (extra_length : ℕ) := (P / 2 - extra_length) / 2

def length_of_field (b : ℕ) (extra_length : ℕ) := b + extra_length

def effective_length (l : ℕ) (obstacle_length : ℕ) := l - obstacle_length

def effective_breadth (b : ℕ) (obstacle_breadth : ℕ) := b - obstacle_breadth

def field_area (length : ℕ) (breadth : ℕ) := length * breadth 

theorem usable_field_area : 
  ∀ (P extra_length obstacle_length obstacle_breadth : ℕ), 
  P = 540 -> extra_length = 30 -> obstacle_length = 10 -> obstacle_breadth = 5 -> 
  field_area (effective_length (length_of_field (breadth_of_field P extra_length) extra_length) obstacle_length) (effective_breadth (breadth_of_field P extra_length) obstacle_breadth) = 16100 := by
  sorry

end usable_field_area_l544_544661


namespace polar_to_cartesian_parabola_l544_544167

theorem polar_to_cartesian_parabola (r θ : ℝ) (h : r = 1 / (1 - Real.sin θ)) :
  ∃ x y : ℝ, x^2 = 2 * y + 1 :=
by
  sorry

end polar_to_cartesian_parabola_l544_544167


namespace train_crossing_time_is_9_seconds_l544_544524

def km_per_hour_to_m_per_s (v : ℕ) : ℝ :=
  v * (1000.0 / 3600.0)

def train_crossing_time (speed_kmh : ℕ) (length_m : ℕ) : ℝ :=
  (length_m : ℝ) / km_per_hour_to_m_per_s speed_kmh

theorem train_crossing_time_is_9_seconds :
  train_crossing_time 126 315 = 9 :=
by sorry

end train_crossing_time_is_9_seconds_l544_544524


namespace sum_of_max_min_g_l544_544341

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - abs (2 * x - 10)

theorem sum_of_max_min_g :
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 →
  let max_val := if x < 2.5 then 4 else if 2.5 ≤ x ∧ x < 5 then 14 - 2 * x else 4 in
  let min_val := if x < 2.5 then 4 else if 2.5 ≤ x ∧ x < 5 then 14 - 2 * x else 4 in
  max_val + min_val = 13 :=
sorry

end sum_of_max_min_g_l544_544341


namespace trapezoid_area_is_4_l544_544295

-- Define the geometrical setup
structure Point :=
  (x : ℝ)
  (y : ℝ)

def Midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def area_of_trapezoid (A B C D : Point) : ℝ :=
  let base1 := (B.x - A.x).abs
  let base2 := (D.x - C.x).abs
  let height := (C.y - A.y).abs
  0.5 * (base1 + base2) * height

-- Points defined as per the problem statement
def A := Point.mk 0 0
def B := Point.mk 2 0
def C := Point.mk 2 4
def D := Point.mk 0 4

def E := Midpoint B C
def F := Midpoint C D
def G := Midpoint A D
def H := Midpoint G E

-- Intersection point of diagonal AC with line HF
def I := Point.mk 1 2

-- The trapezoid is formed by points F, G, I, H
def trapezoid_area := area_of_trapezoid F G I H

-- Lean proof statement to show the area of trapezoid is 4
theorem trapezoid_area_is_4 : trapezoid_area = 4 := by
  sorry

end trapezoid_area_is_4_l544_544295


namespace median_of_custom_list_l544_544301

theorem median_of_custom_list : 
  let list := (List.range' 1 301).bind (λ n, List.replicate n n) in
  let sorted_list := list.sorted in
  (sorted_list.nth (sorted_list.length / 2 - 1) = some 212 ∧ 
  sorted_list.nth (sorted_list.length / 2) = some 212) :=
by
  let list := (List.range' 1 301).bind (λ n, List.replicate n n)
  let sorted_list := list.sorted
  have h_len : sorted_list.length = 45150 := by sorry
  have h_nth_1 : sorted_list.nth (22575 - 1) = some 212 := by sorry
  have h_nth_2 : sorted_list.nth 22575 = some 212 := by sorry
  exact ⟨h_nth_1, h_nth_2⟩

end median_of_custom_list_l544_544301


namespace distinct_positive_integer_quadruples_l544_544237

theorem distinct_positive_integer_quadruples 
  (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a + b = c * d) (h8 : a * b = c + d) :
  (a, b, c, d) = (1, 5, 2, 3)
  ∨ (a, b, c, d) = (1, 5, 3, 2)
  ∨ (a, b, c, d) = (5, 1, 2, 3)
  ∨ (a, b, c, d) = (5, 1, 3, 2)
  ∨ (a, b, c, d) = (2, 3, 1, 5)
  ∨ (a, b, c, d) = (2, 3, 5, 1)
  ∨ (a, b, c, d) = (3, 2, 1, 5)
  ∨ (a, b, c, d) = (3, 2, 5, 1) :=
  sorry

end distinct_positive_integer_quadruples_l544_544237


namespace positive_integers_prime_powers_l544_544573

noncomputable def exists_poly_real_coeffs (n : ℕ) : Prop :=
  ∃ (f : ℝ[X]), (∀ k : ℤ, (f.eval k) ∈ ℤ ↔ ¬ (k % n = 0)) ∧ (f.natDegree < n)

theorem positive_integers_prime_powers :
  {n : ℕ | ∃ f : ℝ[X], (∀ k : ℤ, (f.eval k) ∈ ℤ ↔ ¬ (k % n = 0)) ∧ (f.natDegree < n)} = { p^m | p ∈ PrimeNumbers ∧ m > 0 } :=
sorry

end positive_integers_prime_powers_l544_544573


namespace simplify_and_evaluate_expression_l544_544379

theorem simplify_and_evaluate_expression (x y : ℝ) (hx : x = 1/3) (hy : y = -2) :
  [x * (x + y) - (x - y)^2] / y = 3 := 
by
  -- Proof goes here
  sorry

end simplify_and_evaluate_expression_l544_544379


namespace Mabel_remaining_petals_l544_544346

def initial_daisies : ℕ := 5
def initial_petals_per_daisy : ℕ := 8
def daisies_given_away : ℕ := 2
def new_daisies : ℕ := 3
def new_petals_per_daisy : ℕ := 7
def petals_lost_from_new_daisies : ℕ := 4
def petals_lost_from_original_daisies : ℕ := 2

theorem Mabel_remaining_petals :
  let initial_petals := initial_daisies * initial_petals_per_daisy,
      petals_given_away := daisies_given_away * initial_petals_per_daisy,
      remaining_petals := initial_petals - petals_given_away,
      new_petals_added := new_daisies * new_petals_per_daisy,
      total_petals := remaining_petals + new_petals_added,
      petals_lost := petals_lost_from_new_daisies + petals_lost_from_original_daisies,
      final_petals := total_petals - petals_lost
  in final_petals = 39 :=
by
  sorry

end Mabel_remaining_petals_l544_544346


namespace initial_saltwater_amount_l544_544846

variable (x y : ℝ)
variable (h1 : 0.04 * x = (x - y) * 0.1)
variable (h2 : ((x - y) * 0.1 + 300 * 0.04) / (x - y + 300) = 0.064)

theorem initial_saltwater_amount : x = 500 :=
by
  sorry

end initial_saltwater_amount_l544_544846


namespace distinct_points_count_l544_544023

theorem distinct_points_count :
    let circle_eq (x y : ℝ) := x^2 + y^2 = 16
    let parabola_eq (x y : ℝ) := y = x^2 - 4
    ∃ s : set (ℝ × ℝ), (∀ p, p ∈ s ↔ ∃ x y, circle_eq x y ∧ parabola_eq x y ∧ p = (x, y) ) ∧ s.card = 3 :=
by
  sorry

end distinct_points_count_l544_544023


namespace max_k_for_inequality_l544_544276

/-- Given any triangle ABC, the inequality 2 * sin^2 C + sin A * sin B > k * sin B * sin C 
holds for k <= 2 * sqrt 2 - 1./
theorem max_k_for_inequality (A B C : ℝ) (ha : sin A = (Real.sin A))
                             (hb : sin B = (Real.sin B))
                             (hc : sin C = (Real.sin C)) :
  (2 * (sin C)^2 + sin A * sin B > k * sin B * sin C) → k <= 2 * Real.sqrt 2 - 1 :=
by
  sorry

end max_k_for_inequality_l544_544276


namespace option_C_cannot_be_true_l544_544477

def f : ℕ → ℝ := sorry

theorem option_C_cannot_be_true :
  ¬ (∃ n ∈ {1, 2, ..., 98}, f n < f (n + 1) ∧ f (n + 1) = f (n + 2)) :=
sorry

end option_C_cannot_be_true_l544_544477


namespace henry_earnings_l544_544265

theorem henry_earnings :
  (let lawns_to_mow := 12 in 
   let lawns_forgot := 7 in
   let earnings_per_lawn := 5 in
   let lawns_mowed := lawns_to_mow - lawns_forgot in
   lawns_mowed * earnings_per_lawn = 25) := sorry

end henry_earnings_l544_544265


namespace shaded_area_l544_544538

theorem shaded_area (grid_side : ℕ) (small_square_side : ℕ) (shade_rect_length : ℕ) (shade_rect_width : ℕ) (removed_area_side : ℕ) : 
  grid_side = 3 ∧ small_square_side = 1 ∧ shade_rect_length = 3 ∧ shade_rect_width = 1 ∧ removed_area_side = 1 → 
  shade_rect_length * shade_rect_width - removed_area_side * removed_area_side = 2 :=
begin
  sorry
end

end shaded_area_l544_544538


namespace k_value_for_z_perfect_square_l544_544048

theorem k_value_for_z_perfect_square (Z K : ℤ) (h1 : 500 < Z ∧ Z < 1000) (h2 : K > 1) (h3 : Z = K * K^2) :
  ∃ K : ℤ, Z = 729 ∧ K = 9 :=
by {
  sorry
}

end k_value_for_z_perfect_square_l544_544048


namespace min_attempts_for_6_suitcases_and_keys_min_attempts_for_10_suitcases_and_keys_l544_544040

theorem min_attempts_for_6_suitcases_and_keys :
  ∃ (attempts : ℕ), attempts = 15 :=
begin
  let attempts := 5 + 4 + 3 + 2 + 1,
  use attempts,
  norm_num,
end

theorem min_attempts_for_10_suitcases_and_keys :
  ∃ (attempts : ℕ), attempts = 45 :=
begin
  let attempts := 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1,
  use attempts,
  norm_num,
end

end min_attempts_for_6_suitcases_and_keys_min_attempts_for_10_suitcases_and_keys_l544_544040


namespace smallest_positive_period_max_min_values_l544_544251

open Real -- Use the real number space

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := sin (2 * x) + a * (cos x)^2

theorem smallest_positive_period (a : ℝ) (h : f (π / 4) a = 0) : 
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) a = f (x) a := sorry

theorem max_min_values (a : ℝ) (h : f (π / 4) a = 0) :
  let f_interval := λ x, x ∈ Icc (π / 24) (11 * π / 24)
  ∀ x ∈ Icc (π/24) (11 * π/24),
    f x a ≤ sqrt 2 - 1 ∧ f x a ≥ - ((sqrt 2) / 2) - 1 := sorry

end smallest_positive_period_max_min_values_l544_544251


namespace crayons_loss_l544_544358

def initial_crayons : ℕ := 479
def final_crayons : ℕ := 134
def crayons_lost : ℕ := initial_crayons - final_crayons

theorem crayons_loss :
  crayons_lost = 345 := by
  sorry

end crayons_loss_l544_544358


namespace wednesdays_in_january_and_february_2012_l544_544692

theorem wednesdays_in_january_and_february_2012 :
  let january_days := 31 in
  let february_days := 29 in
  let jan_start_day := 0 in -- 0 represents Sunday
  let days_in_week := 7 in
  (nat.count (λ d, (jan_start_day + d) % days_in_week = 3) (finset.range january_days) = 4) ∧
  (nat.count (λ d, (jan_start_day + 31 + d) % days_in_week = 3) (finset.range february_days) = 5) := 
by {
  sorry -- Proof to be filled in
}

end wednesdays_in_january_and_february_2012_l544_544692


namespace triangle_perimeter_l544_544938

theorem triangle_perimeter (P₁ P₂ P₃ : ℝ) (hP₁ : P₁ = 12) (hP₂ : P₂ = 14) (hP₃ : P₃ = 16) : 
  P₁ + P₂ + P₃ = 42 := by
  sorry

end triangle_perimeter_l544_544938


namespace velocity_at_specific_time_acceleration_at_specific_time_acceleration_proportional_to_displacement_l544_544120

noncomputable def x (A ω t : ℝ) : ℝ := A * Real.sin (ω * t)
noncomputable def v (A ω t : ℝ) : ℝ := deriv (x A ω) t
noncomputable def α (A ω t : ℝ) : ℝ := deriv (v A ω) t

theorem velocity_at_specific_time (A ω : ℝ) : 
  v A ω (2 * Real.pi / ω) = A * ω := 
sorry

theorem acceleration_at_specific_time (A ω : ℝ) :
  α A ω (2 * Real.pi / ω) = 0 :=
sorry

theorem acceleration_proportional_to_displacement (A ω t : ℝ) :
  α A ω t = -ω^2 * x A ω t :=
sorry

end velocity_at_specific_time_acceleration_at_specific_time_acceleration_proportional_to_displacement_l544_544120


namespace sequence_term_2023_l544_544970

theorem sequence_term_2023 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h : ∀ n, 2 * S n = a n * (a n + 1)) : 
  a 2023 = 2023 :=
sorry

end sequence_term_2023_l544_544970


namespace nearest_integer_to_expr_l544_544052

theorem nearest_integer_to_expr : 
  let a := 3 + Real.sqrt 5
  let b := (a)^6
  abs (b - 2744) < 1
:= sorry

end nearest_integer_to_expr_l544_544052


namespace magician_can_identify_matching_coin_l544_544503

-- Define the types and conditions
variable (coins : Fin 11 → Bool) -- Each coin is either heads (true) or tails (false).
variable (uncovered_index : Fin 11) -- The index of the initially uncovered coin.

-- Define the property to be proved: the magician can point to an adjacent coin with the same face as the uncovered one.
theorem magician_can_identify_matching_coin (h : ∃ i : Fin 11, coins i = coins (i + 1) % 11) :
  ∃ j : Fin 11, (j ≠ uncovered_index) ∧ coins j = coins uncovered_index :=
  sorry

end magician_can_identify_matching_coin_l544_544503


namespace playerB_should_choose_best_of_three_l544_544757

noncomputable def player_awinsin4games (pa : ℚ) (pb : ℚ) (independent : Prop) : ℚ :=
  (3 * (pa ^ 2) * (pb) * pa) = 3 * ((2 : ℚ) / 3)^2 * ((1 : ℚ) / 3) * ((2 : ℚ) / 3)

noncomputable def games_distribution : ℚ → ℚ → (ℚ × ℚ × ℚ) :=
  λ pa pb, (
    (pa ^ 3 + pb ^ 3),
    3 * (pa ^ 2) * pb * pa + 3 * (pb ^ 2) * pa * pb,
    1 - (pa ^ 3 + pb ^ 3) - (3 * (pa ^ 2) * pb * pa + 3 * (pb ^ 2) * pa * pb)
  )

noncomputable def expected_value (pa : ℚ) (pb : ℚ) : ℚ :=
  (3 * (pa ^ 3 + pb ^ 3) + 4 * (3 * (pa ^ 2) * pb * pa + 3 * (pb ^ 2) * pa * pb) + 5 * (1 - (pa ^ 3 + pb ^ 3) - (3 * (pa ^ 2) * pb * pa + 3 * (pb ^ 2) * pa * pb)))
  =
  3 * (1 : ℚ) / 3 + 4 * 10 / 27 + 5 * 8 / 27

theorem playerB_should_choose_best_of_three(pa : ℚ) (pb : ℚ) : Prop :=
  let p1 := (pb^2 + 2 * pa * pb * pb),
  let p2 := (pb^3 + 3 * pa * pb^2 + 6 * pa^2 * pb * pb) in
  p1 > p2

def pa := (2 : ℚ) / 3
def pb := (1 : ℚ) / 3

example : player_awinsin4games pa pb True := by {
  unfold player_awinsin4games,
  simp,
  sorry
}

example : games_distribution pa pb = (1 / 3, 10 / 27, 8 / 27) := by {
  unfold games_distribution,
  simp,
  sorry
}

example : expected_value pa pb = 107 / 27 := by {
  unfold expected_value,
  simp,
  sorry
}

example : playerB_should_choose_best_of_three pa pb := by {
  unfold playerB_should_choose_best_of_three,
  simp,
  sorry
}

end playerB_should_choose_best_of_three_l544_544757


namespace function_same_graph_l544_544531

theorem function_same_graph :
  ∀ x : ℝ, x ≥ 0 → (sqrt x)^2 = x := by
  sorry

end function_same_graph_l544_544531


namespace symmetric_curve_intersection_l544_544239

theorem symmetric_curve_intersection (k : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ ln x = k * x) ↔ (k ≤ 0 ∨ k = 1 / Real.exp 1) := 
sorry

end symmetric_curve_intersection_l544_544239


namespace hyperbola_eccentricity_is_sqrt2_l544_544624

noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := 
  let e := Real.sqrt (1 + 2/a) in e

theorem hyperbola_eccentricity_is_sqrt2 :
  ∀ (a : ℝ), (∃ (c : ℝ), c^2 = 4 ∧ ∃ (b : ℝ), b^2 = 2 ∧ c = Real.sqrt (a + 2)) →
  hyperbola_eccentricity a = Real.sqrt 2 :=
by
  intros a h
  obtain ⟨c, hc1, b, hb1, hc2⟩ := h
  have h1 : c = 2, from Real.eq_of_sq_eq_sq c 2 (by linarith) (by linarith),
  have h2 : a + 2 = 4, from hc2.trans (by rw [h1, pow_two]),
  have h3 : a = 2, from sub_eq_iff_eq_add'.mp h2,
  have h4 : hyperbola_eccentricity a = Real.sqrt 2, by {
    rw [hyperbola_eccentricity, h3],
    simp [Real.sqrt_div, Real.sqrt_mul_self (by norm_num), divide_eq]
  },
  exact h4

end hyperbola_eccentricity_is_sqrt2_l544_544624


namespace hyperbola_a_squared_l544_544212

theorem hyperbola_a_squared
  (parabola_focus : (4, 0))
  (parabola_directrix : ∀ x, x = -4)
  (hyperbola_focus : (4, 0))
  (hyperbola_directrix : ∀ x, x = -3)
  (c : ℝ)
  (h : c = 4)
  (relation : - (12 : ℝ) = - (a^2 : ℝ) / c) :
  a^2 = 12 := by
  sorry

end hyperbola_a_squared_l544_544212


namespace sum_of_coefficients_l544_544175

def poly : Polynomial ℤ := 27 * X ^ 9 - 512 * Y ^ 9

theorem sum_of_coefficients : 
  let f := (X - 2 * Y) * (3 * X ^ 2 + 6 * X * Y + 12 * Y ^ 2) * (X + 2 * Y) * (3 * X ^ 2 - 6 * X * Y + 12 * Y ^ 2)
  in Polynomial.coeff_sum f = 32 := 
sorry

end sum_of_coefficients_l544_544175


namespace union_set_subset_range_intersection_empty_l544_544224

-- Define the sets A and B
def A : Set ℝ := { x | 1 < x ∧ x < 3 }
def B (m : ℝ) : Set ℝ := { x | 2 * m < x ∧ x < 1 - m }

-- Question 1: When m = -1, prove A ∪ B = { x | -2 < x < 3 }
theorem union_set (m : ℝ) (h : m = -1) : A ∪ B m = { x | -2 < x ∧ x < 3 } := by
  sorry

-- Question 2: If A ⊆ B, prove m ∈ (-∞, -2]
theorem subset_range (m : ℝ) (h : A ⊆ B m) : m ∈ Set.Iic (-2) := by
  sorry

-- Question 3: If A ∩ B = ∅, prove m ∈ [0, +∞)
theorem intersection_empty (m : ℝ) (h : A ∩ B m = ∅) : m ∈ Set.Ici 0 := by
  sorry

end union_set_subset_range_intersection_empty_l544_544224


namespace workers_complete_task_together_in_4_8_days_l544_544446

-- Definitions for the time taken by individual workers to complete the task
def WorkerB_time : ℝ := 12
def WorkerA_time : ℝ := WorkerB_time
def WorkerC_time : ℝ := 2 * WorkerB_time

-- Definition for the rate at which individual workers complete the task per day
def WorkerA_rate : ℝ := 1 / WorkerA_time
def WorkerB_rate : ℝ := 1 / WorkerB_time
def WorkerC_rate : ℝ := 1 / WorkerC_time

-- Definition of the combined rate of the three workers
def Combined_rate : ℝ := WorkerA_rate + WorkerB_rate + WorkerC_rate

-- The total number of days for Workers A, B, and C to complete the task together
def Total_days : ℝ := 1 / Combined_rate

theorem workers_complete_task_together_in_4_8_days :
  Total_days = 4.8 := 
by sorry

end workers_complete_task_together_in_4_8_days_l544_544446


namespace find_k_l544_544598

theorem find_k (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) : 
  k = Real.sqrt 15 := 
sorry

end find_k_l544_544598


namespace accept_candidate_l544_544675

theorem accept_candidate
    (divide_19_angle : ∃ f : ℝ → ℝ, f 19 = 1) -- 19° divided into 19 equal parts
    (divide_17_angle : ∃ f : ℝ → ℝ, f 17 = 1) -- 17° divided into 17 equal parts
    (not_divide_18_angle : ¬∃ f : ℝ → ℝ, f 18 = 1) -- 18° cannot be divided into 18 equal parts
    : Prop :=
  true

noncomputable def proof_accept_candidate : accept_candidate
    (λ x, if x = 19 then 1 else 0) -- Illustration for proof purpose
    (λ x, if x = 17 then 1 else 0) -- Illustration for proof purpose
    (λ f, if f 18 = 1 then false else true) -- Illustration for proof purpose
    := by
  sorry

end accept_candidate_l544_544675


namespace exp_gt_f_y_bounds_l544_544909

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  ∑ i in Finset.range (n + 1), x^i / i.factorial

theorem exp_gt_f (x : ℝ) (n : ℕ) (h₁ : 0 < x) (h₂ : 0 < n) :
  Real.exp x > f n x :=
sorry

theorem y_bounds (x : ℝ) (n : ℕ) (y : ℝ) (h₁ : 0 < x) (h₂ : 0 < n) (h₃ : Real.exp x = f n x + Real.exp y / (n + 1).factorial * x^(n+1)) :
  0 < y ∧ y < x :=
sorry

end exp_gt_f_y_bounds_l544_544909


namespace residue_of_neg_2035_mod_47_l544_544926

theorem residue_of_neg_2035_mod_47 : (-2035 : ℤ) % 47 = 33 := 
by
  sorry

end residue_of_neg_2035_mod_47_l544_544926


namespace percentage_difference_j_p_l544_544452

theorem percentage_difference_j_p (j p t : ℝ) (h1 : j = t * 80 / 100) 
  (h2 : t = p * (100 - t) / 100) (h3 : t = 6.25) : 
  ((p - j) / p) * 100 = 25 := 
by
  sorry

end percentage_difference_j_p_l544_544452


namespace sequence_formula_l544_544259

theorem sequence_formula :
  ∀ (a : ℕ → ℕ),
  (a 1 = 11) ∧
  (a 2 = 102) ∧
  (a 3 = 1003) ∧
  (a 4 = 10004) →
  ∀ n, a n = 10^n + n := by
  sorry

end sequence_formula_l544_544259


namespace equilateral_coloring_forces_monochrome_l544_544887

-- Define the types for points and colors
inductive Color : Type
| green : Color
| blue : Color

-- Define the points A1, A2, ..., A15
constants A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15 : Point

-- Definition of "mutually equidistant"
def mutually_equidistant (p q r : Point) : Prop := dist p q = dist q r ∧ dist q r = dist r p

-- Definition of "same color"
def same_color (c : Point → Color) (p q r : Point) : Prop := c p = c q ∧ c q = c r

-- The theorem to state that there exist three mutually equidistant points of the same color
theorem equilateral_coloring_forces_monochrome (c : Point → Color) :
  ∃ i j k, mutually_equidistant A_i A_j A_k ∧ same_color c A_i A_j A_k :=
sorry

end equilateral_coloring_forces_monochrome_l544_544887


namespace kevin_started_with_cards_l544_544699

-- The definitions corresponding to the conditions in the problem
def ended_with : Nat := 54
def found_cards : Nat := 47
def started_with (ended_with found_cards : Nat) : Nat := ended_with - found_cards

-- The Lean statement for the proof problem itself
theorem kevin_started_with_cards : started_with ended_with found_cards = 7 := by
  sorry

end kevin_started_with_cards_l544_544699


namespace vasya_gift_ways_l544_544739

theorem vasya_gift_ways :
  let cars := 7
  let constructor_sets := 5
  (cars * constructor_sets) + (Nat.choose cars 2) + (Nat.choose constructor_sets 2) = 66 :=
by
  let cars := 7
  let constructor_sets := 5
  sorry

end vasya_gift_ways_l544_544739


namespace magician_assistant_trick_l544_544488

/-- A coin can be either heads or tails. -/
inductive Coin
| heads : Coin
| tails : Coin

/-- Given a cyclic arrangement of 11 coins, there exists at least one pair of adjacent coins with the same face. -/
theorem magician_assistant_trick (coins : Fin 11 → Coin) : 
  ∃ i : Fin 11, coins i = coins (i + 1) := 
by
  sorry

end magician_assistant_trick_l544_544488


namespace probability_of_sum_9_is_2_by_9_l544_544425

def first_die : list ℤ := [1, 3, 3, 5, 5, 7]
def second_die : list ℤ := [2, 2, 4, 4, 6, 6]

def possible_sums := list.bind first_die (λ x, list.map (λ y, x + y) second_die)

noncomputable def probability_sum_9 := 
  (list.count (λ z, z = 9) possible_sums : ℚ) / (list.length possible_sums)

theorem probability_of_sum_9_is_2_by_9 : probability_sum_9 = 2 / 9 := 
  by sorry

end probability_of_sum_9_is_2_by_9_l544_544425


namespace trig_identity_l544_544448

theorem trig_identity (α β γ : ℝ) :
  sin α + sin β + sin γ = 4 * cos (α / 2) * cos (β / 2) * cos (γ / 2) :=
by
  sorry

end trig_identity_l544_544448


namespace area_of_triangle_l544_544622

-- Definitions based on conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 463^2) - (y^2 / 389^2) = 1

def midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

def line_intersects_asymptotes (A B : ℝ × ℝ) : Prop :=
  A.2 = (389 / 463) * A.1 ∧ B.2 = -(389 / 463) * B.1

-- Area calculation based on the above conditions
noncomputable def area_triangle_OAB (A B : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * B.2 - B.1 * A.2|

theorem area_of_triangle :
  ∃ (A B P : ℝ × ℝ), 
    (hyperbola P.1 P.2) ∧
    (midpoint P A B) ∧
    (line_intersects_asymptotes A B) ∧
    (area_triangle_OAB A B = 180107) :=
  sorry

end area_of_triangle_l544_544622


namespace mutual_knowing_odd_even_l544_544745

-- Define a structure to represent mutual knowing relationships
structure Person (α : Type) :=
(known : set α → ℕ)

-- Define the theorem to prove the number of people knowing an odd number of others is even
theorem mutual_knowing_odd_even {α : Type} (G : finset α) (know : α → α → Prop)
  (h_symm : ∀ a b, know a b ↔ know b a) :
  ∃ k, (∃ (d : ∀ i ∈ G, ℕ), (∀ i ∈ G, d i ∈ G → d i = (∑ x in G, if know i x then 1 else 0)) ∧ (∑ i in G, if (d i % 2 = 1) then 1 else 0) = 2 * k) := 
sorry

end mutual_knowing_odd_even_l544_544745


namespace cos_90_identity_l544_544159

noncomputable def angle_36 : ℝ := 36 * (real.pi / 180)
noncomputable def angle_54 : ℝ := 54 * (real.pi / 180)
noncomputable def angle_90 : ℝ := 90 * (real.pi / 180)

noncomputable def cos_36 : ℝ := (1 + real.sqrt 5) / 4
noncomputable def cos_54 : ℝ := (1 - real.sqrt 5) / 4
noncomputable def sin_36 : ℝ := real.sqrt (10 - 2 * real.sqrt 5) / 4
noncomputable def sin_54 : ℝ := real.sqrt (10 + 2 * real.sqrt 5) / 4

theorem cos_90_identity :
  real.cos (angle_90) = 0 :=
by
  -- state the trigonometric identity and substitutions
  let lhs := real.cos (angle_36 + angle_54)
  let rhs := cos_36 * cos_54 - sin_36 * sin_54
  show lhs = rhs
  -- insert arithmetic simplification to show lhs = rhs, concluding ∎
  sorry

end cos_90_identity_l544_544159


namespace number_of_irreducible_factors_l544_544440

theorem number_of_irreducible_factors (x : ℤ) : ∃ n, n = 5 ∧ ∀ k, irreducible_factor x k → k.factors.size = n :=
sorry

end number_of_irreducible_factors_l544_544440


namespace clock_angle_at_2_30_l544_544891

theorem clock_angle_at_2_30 : 
  let hour_angle := 2 * 30 + 15
  let minute_angle := 30 * 6
  let angle_diff := abs (minute_angle - hour_angle)
  angle_diff = 105 :=
by
  sorry

end clock_angle_at_2_30_l544_544891


namespace find_a_eq_transformed_a_l544_544943

noncomputable def find_a (n : ℤ) : Set ℝ :=
  { a | ∃ (k : ℤ), a = nat.fract ((-1) ^ (k+1) / 5 * Real.arcsin (119/169)) + k * (π/5) }

noncomputable def transformed_a (n : ℤ) : Set ℝ :=
  { a | ∃ (k : ℤ), a = π/10 + k * (2 * π / 5) ∨
                     a = π/10 ± 2/5 * atan (12/5) + k * (2 * π / 5) }

theorem find_a_eq_transformed_a (n : ℤ) : find_a n = transformed_a n :=
  sorry

end find_a_eq_transformed_a_l544_544943


namespace original_costs_l544_544145

theorem original_costs (P_old P_second_oldest : ℝ) (h1 : 0.9 * P_old = 1800) (h2 : 0.85 * P_second_oldest = 900) :
  P_old + P_second_oldest = 3058.82 :=
by sorry

end original_costs_l544_544145


namespace min_value_abc_l544_544962

def S_n (n : ℕ) : ℕ := (∑ i in finset.range (n + 1), ((2^(i + 1)) + (2*i + 1)))

theorem min_value_abc (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : S_n 10 = a * b * c) : a + b + c = 68 :=
sorry

end min_value_abc_l544_544962


namespace remainder_division_l544_544810

theorem remainder_division (x : ℕ) (dividend : ℕ) (divisor : ℕ) (correct_remainder : ℕ) 
    (h1 : dividend = 2^202 + 202) 
    (h2 : divisor = 2^101 + 2^51 + 1) 
    (h3 : correct_remainder = 201) :
    dividend % divisor = correct_remainder := 
by 
    rw [h1, h2, h3]
    sorry

end remainder_division_l544_544810


namespace markup_percentage_l544_544021

-- Definitions coming from conditions
variables (C : ℝ) (M : ℝ) (S : ℝ)
-- Markup formula
def markup_formula : Prop := M = 0.10 * C
-- Selling price formula
def selling_price_formula : Prop := S = C + M

-- Given the conditions, we need to prove that the markup is 9.09% of the selling price
theorem markup_percentage (h1 : markup_formula C M) (h2 : selling_price_formula C M S) :
  (M / S) * 100 = 9.09 :=
sorry

end markup_percentage_l544_544021


namespace number_of_subsets_l544_544700

theorem number_of_subsets (m n s : ℤ) (hm : m ≥ 1) (hn : n ≥ 1) 
  (coprime_mn : Int.gcd m n = 1) : 
  (∃ (A : Finset ℤ), A.card = m ∧ (A.Sum id) % n = s % n) → 
    (∃ x : ℤ, x = (Nat.choose (m + n - 1).toNat m.toNat) / n) := 
sorry

end number_of_subsets_l544_544700


namespace player_A_received_q_first_l544_544417

variable (p q r : ℕ)
variable (A B C : ℕ ) -- Final marble numbers
variable (n : ℕ) -- Number of rounds

def valid (g1 g2 g3 : ℕ) : Prop := 
  (0 < p ∧ p < q ∧ q < r) ∧
  (g1 = 20 ∧ g2 = 10 ∧ g3 = 9) ∧
  (p + q + r = g1 + g2 + g3) ∧
  (g2_last_round := r)

theorem player_A_received_q_first (h : valid p q r 20 10 9):
  sorry

end player_A_received_q_first_l544_544417


namespace nearest_integer_3_add_sqrt_5_pow_6_l544_544062

noncomputable def approx (x : ℝ) : ℕ := Real.floor (x + 0.5)

theorem nearest_integer_3_add_sqrt_5_pow_6 : 
  approx ((3 + Real.sqrt 5)^6) = 22608 :=
by
  -- Proof omitted, sorry
  sorry

end nearest_integer_3_add_sqrt_5_pow_6_l544_544062


namespace frosting_cupcakes_l544_544893

theorem frosting_cupcakes (t1 t2 delay total_time : ℝ)
  (c_rate := 1 / t1) -- rate at which Cagney frosts cupcakes
  (l_rate := 1 / t2) -- rate at which Lacey frosts cupcakes
  (delay := 60)      -- delay in seconds before Lacey starts
  (total_time := 600) -- total working time in seconds
  (time_lacey := total_time - delay) -- time Lacey works
  (cupcakes : ℝ := c_rate * total_time + l_rate * time_lacey) :
  t1 = 15 → t2 = 40 → cupcakes = 53.5 :=
by
  intros ht1 ht2
  rw [ht1, ht2]
  have cr : c_rate = 1 / 15 := by rw ht1
  have lr : l_rate = 1 / 40 := by rw ht2
  simp [c_rate, l_rate, cr, lr]
  norm_num
  sorry -- Placeholder for the computation steps leading to the final result

end frosting_cupcakes_l544_544893


namespace f_1234_eq_1_div_1_sub_x_l544_544256

-- Define the function f_1
def f1 (x : ℝ) : ℝ :=
  (2 * x - 1) / (x + 1)

-- Define the recursive function f_n
noncomputable def f : ℕ → ℝ → ℝ
| 0     => id
| (n+1) => λ x => f1 (f n x)

-- Define the theorem to prove f_{1234}(x) = 1 / (1 - x)
theorem f_1234_eq_1_div_1_sub_x (x : ℝ) : f 1234 x = 1 / (1 - x) :=
  sorry

end f_1234_eq_1_div_1_sub_x_l544_544256


namespace cyc_sum_inequality_l544_544749

variable {α : Type*} [LinearOrderedField α]
variables (a b c d : α)

theorem cyc_sum_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∑ cyc in {(a,b,2*c,3*d),(b,c,2*d,3*a),(c,d,2*a,3*b),(d,a,2*b,3*c)}, 
    \(λ (x, y, z, w), x / (y + z + w)} cyc ≥ 2 / 3 :=
  sorry

end cyc_sum_inequality_l544_544749


namespace min_2x3y2z_l544_544335

noncomputable def min_value (x y z : ℝ) : ℝ := 2 * (x^3) * (y^2) * z

theorem min_2x3y2z (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h : (1/x) + (1/y) + (1/z) = 9) :
  min_value x y z = 2 / 675 :=
sorry

end min_2x3y2z_l544_544335


namespace subset_relation_l544_544576

variables (a : ℝ) (A B : Set ℝ)

def A := {x : ℝ | x^2 - 8 * x + 15 = 0}
def B := {x : ℝ | a * x - 1 = 0}

theorem subset_relation : B ⊆ A ↔ a = 0 ∨ a = 1/3 ∨ a = 1/5 :=
by {
  sorry
}

end subset_relation_l544_544576


namespace area_of_quadrilateral_Q1Q2Q3Q4_l544_544653

variable (P : Fin 6 → EuclideanGeometry.Point ℝ)
variable (Q : Fin 4 → EuclideanGeometry.Point ℝ)
variable (side_length apothem : ℝ)

-- Definitions and assumptions based on problem conditions
def is_regular_hexagon (P : Fin 6 → EuclideanGeometry.Point ℝ) : Prop :=
∀ i : Fin 6, P (i + 1) = EuclideanGeometry.rotate_about (P i) (2 * π / 6) -- Rotation by 120 degrees

def is_midpoint (Q : Fin 4 → EuclideanGeometry.Point ℝ) (P : Fin 6 → EuclideanGeometry.Point ℝ) : Prop :=
∀ i : Fin 4, Q i = EuclideanGeometry.midpoint (P i) (P (i + 1))

-- The apothem condition
def is_apothem (P : Fin 6 → EuclideanGeometry.Point ℝ) (apothem : ℝ) : Prop :=
apothem = 2 ∧ ∀ i : Fin 6, EuclideanGeometry.dist (EuclideanGeometry.center_of_mass P) (EuclideanGeometry.midpoint (P i) (P (i + 1))) = apothem

theorem area_of_quadrilateral_Q1Q2Q3Q4 (h1 : is_regular_hexagon P)
  (h2 : is_midpoint Q P) (h3 : is_apothem P 2) : 
  EuclideanGeometry.area (λ i : Fin 4, Q i) = 4 * sqrt 3 := 
sorry

end area_of_quadrilateral_Q1Q2Q3Q4_l544_544653


namespace problem_part1_problem_part2_l544_544194

noncomputable def degree_to_radian (d : ℝ) : ℝ :=
  d * real.pi / 180

theorem problem_part1 : 
  let alpha := degree_to_radian 1690 
  in ∃ k β, k ∈ int ∧ β ∈ set.Ico 0 (2 * real.pi) ∧ alpha = 2 * k * real.pi + β := 
  by
    let alpha := degree_to_radian 1690
    use 4
    use (25 * real.pi / 18)
    split
    {
      norm_num
    },
    {
      split
      {
        norm_num
      },
      {
        sorry -- Proof of this equality
      }
    }

theorem problem_part2 :
  let alpha := degree_to_radian 1690 
  in ∃ θ, θ ∈ set.Ioo (-4 * real.pi) (-2 * real.pi) ∧ (∃ k, θ = 2 * k * real.pi + (25 * real.pi / 18)) := 
  by
    let theta := -47 * real.pi / 18
    use theta
    split
    {
      -- Proof that θ is in the interval
      split
      {
        have h1 : theta = (-47) * real.pi / 18 := by norm_num
        apply h1.ge -- To be completed
      },
      {
        have h2 : theta = (-47) * real.pi / 18 := by norm_num
        apply h2.le -- To be completed
      }
    },
    {
      use -2
      sorry -- Proof of this equivalence
    }

end problem_part1_problem_part2_l544_544194


namespace downstream_distance_l544_544864

theorem downstream_distance (t_d t_u : ℝ) (d_u : ℝ) (V_m : ℝ) (Vc : ℝ) (D : ℝ) 
  (h1 : t_d = 4) (h2 : t_u = 4) (h3 : d_u = 24) (h4 : V_m = 7) 
  (h5 : t_u * (V_m - Vc) = d_u) : 
  D = (V_m + Vc) * t_d :=
by
  have h6 : 4 * (7 - Vc) = 24 := h5
  have h7 : Vc = 1 := by linarith 
  have h8 : D = (7 + Vc) * 4 := by sorry  
  linarith
  sorry

end downstream_distance_l544_544864


namespace proof_problem_l544_544541

def intelligentFailRate (r1 r2 r3 : ℚ) : ℚ :=
  1 - r1 * r2 * r3

def phi (p : ℚ) : ℚ :=
  30 * p * (1 - p)^29

def derivativePhi (p : ℚ) : ℚ :=
  30 * (1 - p)^28 * (1 - 30 * p)

def qualifiedPassRate (intelligentPassRate comprehensivePassRate : ℚ) : ℚ :=
  intelligentPassRate * comprehensivePassRate

theorem proof_problem :
  let r1 := (99 : ℚ) / 100
  let r2 := (98 : ℚ) / 99
  let r3 := (97 : ℚ) / 98
  let p0 := (1 : ℚ) / 30
  let comprehensivePassRate := 1 - p0
  let qualifiedRate := qualifiedPassRate (r1 * r2 * r3) comprehensivePassRate
  (intelligentFailRate r1 r2 r3 = 3 / 100) ∧
  (derivativePhi p0 = 0) ∧
  (qualifiedRate < 96 / 100) :=
by
  sorry

end proof_problem_l544_544541


namespace ratio_of_men_to_women_l544_544830

variables (M W : ℕ)

theorem ratio_of_men_to_women (h1 : W = M + 4) (h2 : M + W = 14) : M = 5 ∧ W = 9 ∧ (5 : 9) :=
by 
  have : 2 * M + 4 = 14 := by linarith
  have : 2 * M = 10 := by linarith
  have : M = 5 := by linarith
  have : W = M + 4 := h1
  rw this
  have : W = 9 := by linarith
  split
  repeat { assumption }

end ratio_of_men_to_women_l544_544830


namespace find_b_l544_544636

theorem find_b (b : ℝ) (p : (1, 2) ∈ set_of (λ (x y : ℝ), y = 2 * x + b)) : b = 0 := 
sorry

end find_b_l544_544636


namespace school_total_spending_l544_544872

theorem school_total_spending:
  ∀ (seminar_fee_per_teacher : ℝ) 
    (num_teachers : ℕ) 
    (discount1 : ℝ) 
    (discount2 : ℝ)
    (threshold1 : ℕ)
    (threshold2 : ℕ)
    (food_allowance_per_teacher : ℝ)
    (tax_rate : ℝ),
  seminar_fee_per_teacher = 150 → 
  num_teachers = 22 → 
  discount1 = 0.05 → 
  discount2 = 0.075 →
  threshold1 = 10 → 
  threshold2 = 20 → 
  food_allowance_per_teacher = 10 → 
  tax_rate = 0.06 →
  let discounted_fee_per_teacher := if num_teachers ≥ threshold2 then
                                      seminar_fee_per_teacher * (1 - discount2)
                                    else if num_teachers ≥ threshold1 then
                                      seminar_fee_per_teacher * (1 - discount1)
                                    else
                                      seminar_fee_per_teacher in
  let total_seminar_fee := discounted_fee_per_teacher * num_teachers in
  let total_seminar_fee_with_tax := total_seminar_fee * (1 + tax_rate) in
  let total_food_allowance := food_allowance_per_teacher * num_teachers in
  let total_spending := total_seminar_fee_with_tax + total_food_allowance in
  total_spending = 3455.65 :=
by
  sorry

end school_total_spending_l544_544872


namespace company_stores_l544_544852

theorem company_stores (total_uniforms : ℕ) (uniforms_per_store : ℕ) 
  (h1 : total_uniforms = 121) (h2 : uniforms_per_store = 4) : 
  total_uniforms / uniforms_per_store = 30 :=
by
  sorry

end company_stores_l544_544852


namespace input_x_for_y_16_l544_544418

noncomputable def output_y_from_input_x (x : Int) : Int :=
if x < 0 then (x + 1) * (x + 1)
else (x - 1) * (x - 1)

theorem input_x_for_y_16 (x : Int) (y : Int) (h : y = 16) :
  output_y_from_input_x x = y ↔ (x = 5 ∨ x = -5) :=
by
  sorry

end input_x_for_y_16_l544_544418


namespace triangle_side_b_range_l544_544284

noncomputable def sin60 := Real.sin (Real.pi / 3)

theorem triangle_side_b_range (a b : ℝ) (A : ℝ)
  (ha : a = 2)
  (hA : A = 60 * Real.pi / 180)
  (h_2solutions : b * sin60 < a ∧ a < b) :
  (2 < b ∧ b < 4 * Real.sqrt 3 / 3) :=
by
  sorry

end triangle_side_b_range_l544_544284


namespace max_airlines_l544_544667

-- Definitions for the conditions
-- There are 200 cities
def num_cities : ℕ := 200

-- Calculate the total number of city pairs
def num_city_pairs (n : ℕ) : ℕ := (n * (n - 1)) / 2

def total_city_pairs : ℕ := num_city_pairs num_cities

-- Minimum spanning tree concept
def min_flights_per_airline (n : ℕ) : ℕ := n - 1

def total_flights_required : ℕ := num_cities * min_flights_per_airline num_cities

-- Claim: Maximum number of airlines
theorem max_airlines (n : ℕ) (h : n = 200) : ∃ m : ℕ, m = (total_city_pairs / (min_flights_per_airline n)) ∧ m = 100 :=
by sorry

end max_airlines_l544_544667


namespace polynomial_integral_bound_l544_544558

noncomputable def polynomial_bound_integral (p : polynomial ℝ) (n : ℕ) : Prop :=
  p.degree = n ∧ (∀ x : ℝ, polynomial.eval p x ≠ 0) → 
  ∃ I : ℝ, I = ∫ x in -∞..∞, ((polynomial.derivative p).eval x)^2 / ((p.eval x)^2 + ((polynomial.derivative p).eval x)^2) ∧ I ≤ n^(3/2) * Real.pi

theorem polynomial_integral_bound (p : polynomial ℝ) (n : ℕ) : polynomial_bound_integral p n :=
sorry

end polynomial_integral_bound_l544_544558


namespace intersection_A_B_intersection_CR_A_B_l544_544640

noncomputable def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
noncomputable def CR_A : Set ℝ := {x : ℝ | x < 3} ∪ {x : ℝ | 7 ≤ x}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 7} :=
by
  sorry

theorem intersection_CR_A_B :
  CR_A ∩ B = ({x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | 7 ≤ x ∧ x < 10}) :=
by
  sorry

end intersection_A_B_intersection_CR_A_B_l544_544640


namespace pipe_b_fills_tanker_in_40_minutes_l544_544861

theorem pipe_b_fills_tanker_in_40_minutes : 
  ∃ t : ℝ, 
    (∀ (A B : ℝ), 
      A = 60 → 
      t = 40 → 
      (1/2 * (1/B) + 1/2 * (1/A + 1/B)) * 30 = 1
    ) := 
exists.intro 40 
  (λ A B, assume hA : A = 60, assume ht : 30 = 30,
     by sorry)

end pipe_b_fills_tanker_in_40_minutes_l544_544861


namespace chord_length_intercepted_by_curve_l544_544012

theorem chord_length_intercepted_by_curve
(param_eqns : ∀ θ : ℝ, (x = 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ))
(line_eqn : 3 * x - 4 * y - 1 = 0) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 3 := 
sorry

end chord_length_intercepted_by_curve_l544_544012


namespace area_of_parallelogram_formed_by_vectors_l544_544149

variables {ℝ : Type*} [inner_product_space ℝ ℝ] [normed_space ℝ ℝ] [normed_field ℝ]

noncomputable def magnitude (v : ℝ → ℝ) : ℝ := sqrt (⟪v, v⟫)

noncomputable def cross_product (v w : ℝ → ℝ) : ℝ := magnitude (λ t, ∥v t ∥ * ∥w t ∥ * sin (π / 4))

noncomputable def area_of_parallelogram (a b : ℝ) : ℝ := 7 * cross_product a b

theorem area_of_parallelogram_formed_by_vectors
  (p q : ℝ → ℝ) 
  (h_p : magnitude p = 4)
  (h_q : magnitude q = 1)
  (h_angle : ∀ t, ⟪p t, q t⟫ = cos (π / 4)) : 
  area_of_parallelogram (3 * p + q) (p - 2 * q) = 14 * sqrt 2 :=
sorry

end area_of_parallelogram_formed_by_vectors_l544_544149


namespace probability_abc_eq_72_l544_544082

theorem probability_abc_eq_72 : 
  let die_results := ({1, 2, 3, 4, 5, 6} : Finset ℕ)
  in (∃ (a b c : ℕ), (a ∈ die_results) ∧ (b ∈ die_results) ∧ (c ∈ die_results) ∧ (a * b * c = 72)) → 
     (↑(die_results.card) ^ 3 = 216) →
     (∃ (count_abc_eq_72 : ℕ), count_abc_eq_72 = 9) →
     (count_abc_eq_72 / 216 = (1 : ℚ) / 24) :=
by
  sorry

end probability_abc_eq_72_l544_544082


namespace total_volume_of_spheres_sequence_l544_544673

-- Definitions of pyramid properties and constants
def regular_triangular_pyramid (S A : ℝ) (h : ℝ) : Prop :=
  A = 4 * S ∧ h = 130

-- Define the volume of the sum of the infinite sequence of spheres
def volume_of_spheres_sequence (S A : ℝ) (h : ℝ) : ℝ :=
  if regular_triangular_pyramid S A h then (8788 / 2598) * Real.pi else 0

-- The proof problem statement
theorem total_volume_of_spheres_sequence (S A : ℝ) (h : ℝ) :
  regular_triangular_pyramid S A h →
  volume_of_spheres_sequence S A h = (8788 / 2598) * Real.pi :=
by
  intro h₀,
  unfold volume_of_spheres_sequence,
  rw if_pos h₀,
  have : regular_triangular_pyramid S A h := h₀,
  sorry

end total_volume_of_spheres_sequence_l544_544673


namespace correct_conclusions_l544_544691

def in_class (a k : ℤ) : Prop := ∃ n : ℤ, a = 5 * n + k

def all_classes : Prop :=
  ∀ x : ℤ, ∃ k : fin 5, in_class x k

def same_class (a b : ℤ) : Prop := ∃ k : fin 5, in_class a k ∧ in_class b k

def mod_class (a b : ℤ) : Prop := in_class (a - b) 0

theorem correct_conclusions :
  in_class 2015 0 ∧ ¬in_class (-3) 3 ∧ all_classes ∧ (∀ a b : ℤ, same_class a b ↔ mod_class a b) :=
by
  split; sorry  -- in_class 2015 0
  split; sorry  -- ¬in_class (-3) 3
  split; sorry  -- all_classes
  sorry  -- ∀ a b, same_class a b ↔ mod_class a b

end correct_conclusions_l544_544691


namespace sum_of_primes_is_prime_l544_544780

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

theorem sum_of_primes_is_prime (P Q : ℕ) :
  is_prime P → is_prime Q → is_prime (P - Q) → is_prime (P + Q) →
  ∃ n : ℕ, n = P + Q + (P - Q) + (P + Q) ∧ is_prime n := by
  sorry

end sum_of_primes_is_prime_l544_544780


namespace angle_A_is_correct_l544_544663

-- Define the given conditions and the main theorem.
theorem angle_A_is_correct (A : ℝ) (m n : ℝ × ℝ) 
  (h_m : m = (Real.sin (A / 2), Real.cos (A / 2)))
  (h_n : n = (Real.cos (A / 2), -Real.cos (A / 2)))
  (h_eq : 2 * ((Prod.fst m * Prod.fst n) + (Prod.snd m * Prod.snd n)) + (Real.sqrt ((Prod.fst m)^2 + (Prod.snd m)^2)) = Real.sqrt 2 / 2) 
  : A = 5 * Real.pi / 12 := by
  sorry

end angle_A_is_correct_l544_544663


namespace geometric_sequence_ratio_l544_544761
-- Lean 4 Code

noncomputable def a_n (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * q ^ n

theorem geometric_sequence_ratio
  (a : ℝ)
  (q : ℝ)
  (h_pos : a > 0)
  (h_q_neq_1 : q ≠ 1)
  (h_arith_seq : 2 * a_n a q 4 = a_n a q 2 + a_n a q 5)
  : (a_n a q 2 + a_n a q 3) / (a_n a q 3 + a_n a q 4) = (Real.sqrt 5 - 1) / 2 :=
by {
  sorry
}

end geometric_sequence_ratio_l544_544761


namespace distinct_values_of_odd_integers_sum_product_l544_544564

theorem distinct_values_of_odd_integers_sum_product :
  let odd_numbers := {n | n < 16 ∧ n % 2 = 1}
  let possible_pairs := (odd_numbers × odd_numbers)
  ∃ distinct_values, distinct_values.size = 36 ∧ 
  distinct_values = {pq + p + q | (p, q) ∈ possible_pairs}
  sorry

end distinct_values_of_odd_integers_sum_product_l544_544564


namespace solution_set_inequality_l544_544403

theorem solution_set_inequality (x : ℝ) : 
  (x + 5) * (3 - 2 * x) ≤ 6 ↔ (x ≤ -9/2 ∨ x ≥ 1) :=
by
  sorry  -- proof skipped as instructed

end solution_set_inequality_l544_544403


namespace sum_n_k_l544_544769

theorem sum_n_k (n k : ℕ) (h1 : ∃ c1 c2 c3 : ℕ, c1:c2:c3 = 1:3:6) 
  (h2 : c1 = binomial n k) 
  (h3 : c2 = binomial n (k + 1)) 
  (h4 : c3 = binomial n (k + 2)) : 
  n + k = 9 := 
sorry

end sum_n_k_l544_544769


namespace danny_reach_time_l544_544164

theorem danny_reach_time (t : ℕ) (h1 : Steve's reach time = 2 * t) (h2 : (2 * t) / 2 - t / 2 = (17.5 : ℝ)) : 
  t = 35 :=
by
  sorry

end danny_reach_time_l544_544164


namespace vector_distributive_l544_544443

theorem vector_distributive (a b c : ℝ) (va vb vc : ℝ^n) :
  (a + b) * c = a * c + b * c →
  (va + vb) • vc = va • vc + vb • vc :=
sorry

end vector_distributive_l544_544443


namespace fraction_quaduple_l544_544113

variable (b a : ℤ)

theorem fraction_quaduple (h₁ : a ≠ 0) : (2 * b) / (a / 2) = 4 * (b / a) :=
by
  sorry

end fraction_quaduple_l544_544113


namespace factorial_equation_solution_unique_l544_544942

theorem factorial_equation_solution_unique :
  ∀ a b c : ℕ, (0 < a ∧ 0 < b ∧ 0 < c) →
  (a.factorial * b.factorial = a.factorial + b.factorial + c.factorial) →
  (a = 3 ∧ b = 3 ∧ c = 4) := 
by
  intros a b c h_positive h_eq
  sorry

end factorial_equation_solution_unique_l544_544942


namespace problem_solution_l544_544324

def T : ℕ := ∑ n in Finset.range 334, (-1 : ℤ) ^ n * Nat.choose 1002 (3 * n)

theorem problem_solution : T % 500 = 6 := by
  sorry

end problem_solution_l544_544324
